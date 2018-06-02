#!/usr/bin/env python

import sys
import math
import numpy as np
import random

import cv2
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.stats as stats

import pygame
from pygame.locals import *
from pygame.color import *
    
import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util

from information import *
from behavior_spatial_map import *
from behavior_voronoi import *
from behavior_move_towards import *
from behavior_obstacle_avoidance import *
from behavior_robot_control import *
from behavior_tight_passage import *
from behavior_door_perception import *
from behavior_open_door import *
from behavior_move_object import *
from behavior_object_perception import *
from behavior_push_object import *
from behavior_push_execution import *
from behavior_space_digging import *
from manual_control import *
from simulation_models import *
from environment_xml import *

def generateBehaviorsDotGraph(filename, behaviors):
    with open(filename, "w") as f:
        graph_str = "digraph behaviors {\n"
        data_type_names = set()
        for b in behaviors:
            graph_str += "    behavior_" + b.name + ' [label="' + b.name + '"];\n'
            for inp in b.inputs:
                data_type_names.add( inp )
                graph_str += "    " + inp + " -> behavior_" + b.name + ";\n"

            for outp in b.outputs:
                data_type_names.add( outp )
                graph_str += "    behavior_" + b.name + " -> " + outp + ";\n"

        for data_type in data_type_names:
            graph_str += "    " + data_type + " [shape=rectangle];\n"
        graph_str += "}\n"
        f.write( graph_str )

#
# polecenie: otworz_drzwi 
# - sprawca: agent
# - wywolanie ruchu
# - drzwi -> ruch obrotowy
# - interakcja
# - dotyk
# - zajecie odpowiedniego obszaru
#
# Potrzebna wiedza:
# - zdolnosc robota do przesuwania obiektow
# - sposoby przemieszczania obiektow, np. poprzez pchanie
# - ograniczenia kinematyczne drzwi (ruch po luku)
# - zmiana polozenia obiektu, ruch
# - zwiazek polecenia "otworz drzwi" z przemieszczeniem
# - kotwiczenie abstrakcyjnego pojecia do obserwacji
# - przyczyna ruchu, np. pchanie, czyli jeden obiekt w ruchu przesuwa inny obiekt
# - zdolnosc robota do dzialania, np. mozliwosc ruchu i generowania sily
# - stany posrednie, ciaglosc stanu w czasie oraz ciaglosc obiektow w czasie i przestrzeni
#
# Zadanie brzmi: "otworz drzwi"
# - wnioskowanie na podstawie wiedzy, interpretacja polecenia: "otworz drzwi" -> przemeszczenie drzwi
# - kotwiczenie: fragment srodowiska -> drzwi, zawiasy, przejscie ktore drzwi zamykaja
# - wiedza o drzwiach: ksztalt, kinematyka
# - wnioskowanie: efekt: przemieszczenie drzwi -> przyczyna: pchanie przez inny obiekt -> zdolnosc robota do ruchu -> robot pcha drzwi
#
# Schodzenie od ogolu do szczegolu. Najpierw odpowiedz na pytanie kto i co (robot wywoluje ruch drzwi), a dopiero pozniej
# odpowiedz na pytanie "jak?" (chwyt, pchanie, itp.).
#
# Planowanie zadania "otworz drzwi" zaczyna sie od okreslenia mozliwego zakresu ruchu. Zakres ten obejmuje duza czesc przestrzeni,
# na poczatku cala mozliwa. W miare doprecyzowywania planu (np. wybor chwytu), przestrzen ta sie zmniejsza.
# Juz na tym etapie mozna wyznaczyc sciezke dojscia (lub zblizenia sie) do celu.
# W przypadku otwierania drzwi mamy wiele mozliwych chwytow, ktore mozna podzielic w grupy w zaleznosci od roznic miedzy nimi.
# Plynne przejscie pomiedzy chwytami w jednej grupie jest mozliwe, ale przejscie pomiedzy grupami nie jest juz takie oczywiste.
#
# Efektem zadania "otworz drzwi" jest zmiana stanu srodowiska. Od tej zmiany wychodzimy, by nastepnie wywnioskowac, ze odpowiednia
# czynnoscia, ktora nalezy wykonac jest pchanie. Czynnosc ta ma byc wykonana przez robota. Wymaga ona interakcji miedzy robotem z drzwiami,
# wiec robot musi znalezc sie w okreslonej przestrzeni. Ruch drzwi jest spowodowany ruchem robota i jednoczesnym kontaktem miedzy nimi.
# Ruch robota jest wiec scisle zwiazany z przewidywanym ruchem drzwi. Majac dany przewidywany ruch drzwi, oraz stan biezacy, mozna okreslic
# mozliwe sposoby interakcji. Jesli interakcja nie jest mozliwa nalezy okreslic przyczyne tego utrudnienia i znalezc sposob rozwiazania problemu.
# Mozliwe utrudnienia w interakcji moga wynikac z braku dostepnych miejsc chwytu (drzwi nie maja klamki i sa zamkniete). Na pewnym etapie nalezy
# utworzyc graf mozliwych sposobow otwierania, np. ze zmiana chwytu w trakcie, z otwarciem drugiego skrzydla.
# Utrudnienia moga wystapic na wielu etapach planowania, np.:
# - zmiana stanu nie jest mozliwa, gdyz docelowe polozenie jest nieosiagalne ze wzgledu na przeszkody w srodowisku,
# - zmiana stanu nie jest mozliwa, gdyz drzwi sa zamkniete na klucz,
# - chwyt jest nieosiagalny, np. drzwi mozna jedynie pchac od wewnetrznej strony, a oba skrzydla drzwi sa zamkniete,
# - otwieranie drzwi zakonczy sie przedwczesnie ze wzgledu na ograniczenie w ruchu robota lub ze wzgledu na przeszkody
#   (podobnie jak w punkcie pierwszym)
# Jesli przeszkody w srodowisku uniemozliwiaja lub utrudniaja otwarcie drzwi, to nalezy dodac do biezacych motywacji zadanie utworzenia
# wolnej przestrzeni. Zadanie to zawiera tylko informacje o tym co nalezy odsunac i gdzie tego nie odkladac. Motywacje zostaja polaczone
# z innymi motywacjami (czyli zadaniami) i jest uaktualniany graf motywacji, zawierajacy informacje o tym, co z czego wynika.
# Graf ten jest rozbudowywany w miare uszczegolowiania planow. Kiedy jego stan sie ustala (lub nasyca), to znaczy, ze mozna rozpoczac wykonywanie
# zadania. Odkryte utrudnienia pozwalaja na doprecyzowanie poszczegolnych motywacji lub na dodanie nowych motywacji, poprzedzajacych inne.
# Bardzo istotne jest przewidywanie skutkow potencjalnych dzialan, np. otwierajace sie drzwi moga same odsunac przeszkode.
# Moze opisac skutki mozliwych dzialan jako zjawiska, np. "przesuniecie przeszkody". Wnioskowanie  w tym przypadku:
# - interpretacja polecenia "otworz drzwi" jako zmiana stanu (polozenia) drzwi
# - wyznaczenie ruchu drzwi i okresleie czy wystepuja jakies przeszkody - jest przeszkoda
# - sprawdzenie co mozna zrobic z przeszkoda - da sie ja przesunac
# - sa co najmniej dwie mozliwosci: 1) przesuniecie przeszkody razem z drzwiami, 2) odsuniecie przeszkody przed wykonaniem ruchu
# - tu nastepuje rozgalezienie:
# - 1) symulacja wspolnego ruchu drzwi i przeszkody - ruch ten wydaje sie mozliwy
# - 2) wyznaczenie nowej motywacji: zwolnij miejsce w obszarze ruchu drzwi - przemiesc przeszkode gdziekolwiek indziej
#    - motywacja ta jest oznaczona jako poprzednik motywacji "otworz drzwi"
# - teraz mozliwe sa dwa plany dzialan, ktore mozna doprecyzowac
# - doprecyzowujemy plan 1):
#   - aby przesunac drzwi, robot musi wejsc w interakcje z nimi,
#   - jedynym mozliwym miejscem chwytu jest klamka, jednak polozenie docelowe robota jest nieosiagalne ze wzgledu na przeszkode
#   - podobnie jak w poprzednim przypadku ruch przeszkody jest nieunikniony, a moze byc wykonany na wiele sposobow
#
# Ruch przeszkody przewija sie miedzy roznymi alternatywnymi planami oraz jest powiazany pomiedzy nastepujacymi po sobie czynnosciami i zjawiskami.
# Mozemy wyznaczyc abstrakcyjne zjawisko: "odsun przeszkode", ktore jest powiazanie z punktem 1) i 2) oraz z nastepnymi etapami planowania
# (tj. ruch robota) w obu tych punktach. Zjawisko to jednak wyglada inaczej na wyzszym poziomie szczegolowosci w kazdym z przypadkow.
#
# Odsloniecie przejscia, czy tez zwolnienie miejsca w przestrzeni wiaze sie z przemieszczeniem obiektu. Dla samego zadania odsloniecia przestrzeni
# docelowe miejsce dla przeszkody nie jest istotne, ale moze byc ono okreslone przez inne zadania (motywacje). Jedna z tych motywacji jest dbalosc
# o wolne przejscie do celu. Poczatkowa motywacja w tym przypadku jest "przemiesc sie do celu". Wymagane jest przy tym wolne przejscie, ktore jest
# zastawione obiektem, ktory mozna przemieszczac. Jako ze miejsce docelowe jest wyznaczone na koncu waskiego korytarza, to jedyna mozliwoscia jest
# wysuniecie przeszkody na zewnatrz i ponowne wejscie do korytarza. Symulacja w tym przypadku powinna przewidziec, ze pchanie przeszkody wglab
# nic nie da. Samo odkrycie mozliwosci przepchania przeszkody jest juz duzym wyzwaniem. Nalezaloby utworzyc mape topologiczna uwzgledniajaca
# ruchome przeszkody o roznym stopniu ruchomosci, czy tez inaczej - masie. Porownywanie mozliwych rozwiazan miedzy soba mogloby zachodzic na
# podstawie roznych kryteriow, np. czasu, wydatku energii, ryzyku niepowodzenia.
#
# Czy jest mozliwe symulowanie zadania na wielu poziomach abstrakcji?
# 1. otworz_drzwi
# 2. dojscie_do_drzwi, otwieranie_drzwi
# 3. dojscie_do_drzwi, odsuwanie_przeszkody, otwieranie_drzwi
#
# Odnosnie manipulacji, obiek moze byc manipulowany przez inny obiekt, ktory pelni funkcje "sterownika", np. dla przesuwania obiektu
# "sterownikiem" moze byc robot. Jesli robot uzywa narzedzia, to uwaga w zadaniu manipulacji skupiona jest na interakcji miedzy narzedziem
# a obrabianym obiektem. Z kolei robot jest sterownikiem dla narzedzia na kolejnym, nizszym poziomie. W przypadku otwierania drzwi robot jest
# "sterownikiem" dla drzwi. Przyklada do nich sile i nimi porusza. Mozliwa dlugosc lancucha sterownik - obiekt jest ograniczona i zalezna od
# mozliwosci predykcji, ktore z kolei sa zalezne od niepewnosci i dokladnosci modelu. Czym wlasciwie sterujemy przesuwajac obiekty? Dla zadania
# pchania mozemy przyjac, ze sterujemy polozeniem. Jaka wiedza jest potrzebna, zeby przesunac obiekt? Podobnie jak w przypadku ruchu samego robota,
# potrzebna jest znajomosc miejsca docelowego, pozadanej sciezki, modelu zachowania obiektu oraz mozliwosci robota, czyli w skrocie: celu oraz metody
# sterowania obiektem za pomoca robota.
#
# Istnienie przejscia lub jego brak moze byc rozpatrywany tylko w kontekscie obiektu, ktory ma przez dane przejscie przejsc.
#
# Podejmowanie decyzji odbywa sie za pomoca "wskazowek", czyli zebranego doswiadczenia.
# Wskazowkami moga byc np.:
# - przemieszczenie sie -> przez wolne przejscie
# - przemieszczenie sie -> kolizja z lekkim obiektem i i przesuniecie go
# Kazda decyzja musi byc rozpatrywana w szerszym kontekscie z obliczeniem jej kosztu oraz mozliwych konsekwencji.
# Przyklad: okreslanie mozliwej sciezki do celu:
# - ruch w wolnej przestrzeni -> sprawdzenie konsekwencji -> brak; dodatkowo sprawdzenie alternatyw -> rozmaitosc (szeroka przestrzen mozliwych
#   konfiguracji)
# - ruch w poblizu przeszkod -> sprawdzenie konsekwencji -> zderzenie, przesuniecie przeszkody -> ograniczenia w ruchu; sprawdzenie alternatyw ->
#   okreslenie miejsc mniej i bardziej "niewygodnych"
# Przeszukiwanie alternatyw powinno konczyc sie na bardzo plytkim poziomie, bez wnikania w dalsze etapy mozliwych scenariuszy, np. przebicie dziury
# w scianie jest kosztowna alternatywa, i choc ona istnieje, nie powinna byc brana pod uwage. Jesli przeszukiwanie
# plytkie konczy sie brakiem rozwiazania, nalezy wybrac obiecujace alternatywy o wyzszym koszcie i je przeanalizowac. Juz na poziomie przeszukiwania
# plytkiego powinien byc znany szacunkowy koszt poszczegolnych alternatyw, np. przejscie przez sciane jest znacznie bardziej kosztowne niz przepchanie
# lekkiej przeszkody, co jest z kolei bardziej kosztowne niz prejscie krotka i szeroka sciezka. Ocena poszczegolnych rozwiazan i sposob ich porownania
# jest dosc arbitralny, ale lepszy niz brak jakiegokolwiek porownania.
#
# Dodatkowo, podejmowanie decyzji wspomagane jest przez symulacje. Symulacja, podobnie jak generowanie wskazowek, zachodzi na wielu poziomach
# szczegolowosci. Przyklad wnioskowania z wykorzystaniem wskazowej i symulacji:
# - wskazowka: najkrotsza sciezka do celu (ze znanych drog, z pominieciem przeszkod)
# - symulacja: na wybranej sciezce jest przeszkoda
# - wskazowka: przesuniecie przeszkody
# - symulacja: przeszkoda usunieta, mozliwe dotarcie do celu
# - pojawilo sie nowe zadanie posredie: usuniecie przeszkody
# - wskazowka: usuniecie przeszkody przez przesuniecie
# - symulacja: przesuniecie przeszkody jest bardzo kosztowne
# - wskazowka: wyszukaj inna sciezke
# - symulacja: na wybranej sciezce jest wiele przeszkod
# - wskazowka: usuniecie przeszkod
# - symulacja: trudne do przewidzenia zachowanie
# - wskazowka: wybranie utorowanej drogi
# - symulacja: ...
#
# Wskazowki odnosnie przesuwania przeszkod sa bardzo zlozone i zalezne od wielu czynnikow, np. od sasiedztwa przeszkod, tarcia, zadania.
#
# Poziomy symulacji dla usuwania przeszkod:
# - znikniecie obiektu
# - przemieszczenie obiektu (zasieg mozliwego przemieszczenia, jego prawdopodobienstwo oraz interakcje z innymi obiektami).
#   Wybor docelowego polozenia oraz symulacja ruchu nie sa niezalezne.
# - przemieszczenie obiektu z uwzglednieniem mozliwosci robota
# Nalezaloby tu jeszcze dodac symulacje przemieszczenia grupy obiektow, przy czym dokladna symulacja nie bylaby wykonywana, lecz na podstawie
# ilosci wolnej przestrzeni pomiedzy obiektami moznaby oszacowac mozliwosc "upchania" obiektow. Przepychanie przeszkod mialoby nastepujace
# parametry: kierunek pchania, mozliwosc przeplywu (w zaleznosci od tarcia), mozliwosc upchania (w zaleznosci od ilosci wolnego miejsca).
#
# Jakie moga byc najprostsze wskazowki dot. odsuwania przeszkod?
# - unikanie zapychania wolnego przejscia, szegolnie tego, ktore ma zostac udroznione
# - wskazowki takie same jak dla przesuwania: mozliwe sterowanie posrednie i zwiazane z nim przesuniecie
#
# Wskazowki dotyczace przesuwania obiektow:
# - punkt kontaktu i sila -> przesuniecie
# - wstepne ustawienie obiektu do przesuwania (np. obrot przed pchaniem). Tu moznaby wyznaczyc najlepsze punkty do pchania i ciagniecia z punktu
#   widzenia mozliwosci manipulacji. Punkty te sa zalezne od ksztaltu czesci robota, ktora bedzie pchac.
# - kierunek przesuniecia -> punkt i sila kontaktu
# - kierunek przesuniecia -> ustawienie poczatkowe
# - nalezy odrozic przesuwanie precyzyjne od pchania na sile (upychania). W drugim przypadku nalezy rozwazyc mozliwosc uszkodzenia obiektow
#   oraz czy jest miejsce, w ktorym mozna upchac obiekty. Pomimo pozornego nieuporzadkowania, nalezy okreslic ktore obiekty beda upychane
#   i w jaki sposob.
#
# W przypadku udrozniania przejscia zadanie jest zdefioniowane podobnie jak w przypadku zwalniania przestrzeni. Przestrzenia do zwolnienia jest
# przejscie. W przypadku szerokiego, mocno zastawionego przejscia nalezy wyznaczyc korytarz miedzy przeszkodami i okreslic sposob jego udroznienia,
# np. upychanie lub precyzyjne przesuwanie. Wyznaczenie korytarza tez moze byc zdefiniowane przez wskazowki.
#
# Jak reprezentowac te wskazowki i symulacje, aby dodawanie nowych elementow bylo latwe, a ich integracja zachodzila automatycznie?
# Problemem jest uaktywnienie wskazowki. Skad wiadomo, ze wskazowka "odblokuj przejscie" ma zostac wybrana w sytuacji, kiedy 
#
# Opis srodowiska jest oparty na relacjach. Opisujemy polozenie jako np. "sloik jest w szafce na drugiej polce, po lewej, szafka jest w pokoju
# po prawej od wejscia, pokoj jest...". Relacje opisane sa z punktu widzenia obserwatora.
#
# - wybor 1 sciezki na podstawie dostepnych danych (np. graf bez uwzglednienia przeszkod, wczesniej realizowane sciezki, graf z oszacowanym kosztem usuwaia przeszkod)
# - przeanalizowanie sciezki, okreslenie jej kosztu
# - jesli sciezki nie udalo sie znalezc, nalezy rozwazyc udroznienie przejsc. Jak wybrac przejscia do udroznienia? Bierzemy jakas dobra (w przypadku bez przeszkod)
#   sciezke i obliczamy koszt jej udroznienia. Sciezka sklada sie z przejsc, wiec szacunkowy koszt udrozienia mozna zapamietac. Nalezy pamietac, ze udroznienie
#   przejscia wlasciwe dla jednej sciezki moze byc niewlasciwe dla innej. Jak oszacowac koszt udroznienia?
#   Bierzemy ksztalt przeszkody, wolne miejsce oraz mozliwosci robota w zakresie przesuwania przeszkod i szacujemy na ile udroznienie jest mozliwe, np. w waskim
#   przejsciu zakonczonym zakretem i przeszkoda, ktora mozna tylko pchac, koszt jest wysoki, a w przypadku przeszkody na ujsciu przejscia, ktora mozna wypchnac
#   do wolnej przestrzeni koszt jest niski. Do tego oszacowania potrzebne sa pewne zdolnosci do przewidywania zadania przesuwania obiektow. Wyjsciem tego zadania
#   sa mozliwe polozenia koncowe oraz koszt ich uzyskania lub prawdopodobienstwo powodzenia. Pytanie o udroznienie powinno wlasciwie brzmiec: "czy dla danej sciezki
#   mozna utworzyc korytarz?". Korytarz ten moze zajmowac cala dostepna szerokosc przejscia, a w przypadku przeciskania sie miedzy przeszkodami powinien byc zwezany
#   do niezbednego minimum. Ksztalt korytarza jest scisle zwiazane z mozliwosciami robota w kwestii przesuwania przeszkod. Mozna wyroznic kilka sposobow na przesuwanie
#   przeszkod, np. pchanie wzdluz przejscia, ciagniecie (o ile obiekt ma odpowiedni ksztalt), przesuwanie na bok przejscia (o ile jest na to miejsce).
#   Symboliczny opis mozliwosci udrozniania przejscia:
#   * jesli szerokosc przeszkody + wymagana szerokosc < szerokosc przejscia -> odsun na bok, czyli obroc tak, aby najmniejszy wymiar byl w poprzek i przyklej do sciany
#   * jesli mozna ciagnac i do poszerzenia z tylu jest blisko i jest tam miejsce -> wyciagnij przeszkode
#   * jesli do poszerzenia z przodu jest blisko i jest tam miejsce -> pchaj przeszkode
#   Widac tutaj, ze potrzebne jest rozpoznawanie sytuacji oraz sposob reakcji na nie. Taka sytuacja jest ruch do celu. Reakcja na te sytuacje jest wybor drogi (sciezki).
#   Moze okazac sie, ze takiej bezkolizyjnej sciezki nie ma i to jest kolejna sytuacja. Reakcja na nia jest opisana powyzej. W koncu, reakcja na sytuacje przemieszczenia
#   sie jest wybor korytarza. W przypadku sytuacji braku korytarza, reakcja jest udroznienie.
#   Dodatkowo, kazda akcja musi byc zgodna z wykonywanymi zadaniami. W przypadku niezgodnosci, nastepuje cofniecie we wnioskowaniu i wybor innego rozwiazania lub
#   modyfikacja celow zadania. Przykladem takiej niezgodnosci jest ruch bezkolizyjny do celu. Jesli nie udalo sie znalezc bezkolizyjnej sciezki, to ruch tem moze albo
#   zostac zaniechany albo moze nastapic modyfikacja zalozen, tj. zamiast ruchu bezkolizyjnego dopusci sie ruch z kolizjami. Idac dalej, jesli ruch z kolizjami wymaga
#   przesuniecia przeszkody, ktora jest ustawiona w odpowiednim miejscu dla dalszych zadan (czyli nie powina byc przemieszczana), to mozna rozwazyc m.in. tymczasowe
#   jej przemieszczenie (juz na poziomie zadania). Zadania na roznych poziomach oraz ich podzadania moga sie przenikac i wplywac jedno na drugie. Wiele z tych sytuacji
#   moze byc przewidzianych na etapie symulacji, lecz sprawdzanie wielu mozliwosci jest niepraktyczne. Zaleznosci te moga byc zauwazone na wyzszym poziomie abstrakcji.
#   Jak okreslic zaleznosci pomiedzy zadaniami?
#
#
# Podazanie do celu - motywacje:
# - wybieraj najkrotsza sciezke (pelen horyzont)
# - unikaj przeszkod (lokalnie)
# - miesc sie w ciasnych przejsciach (krotki horyzont)
# - wyrabiaj sie na zakretach (krotki horyzont)
# - zmierzaj do celu, uzywaj skrotow (sredni horyzont)
#
# Symulacja skomplikowanego ruchu dla env_02 z (500,300) do (100,300):
# 1. wyszukaj sciezke -> siezka przez wejscie w korytarz, na rozgalezieniu w lewo, zakret w prawo
# 2. zweryfikuj sciezke:
#    a) wejscie w ciasne przejscie -> dodaj ograniczenie przed wejsciem do korytarza: orientacja o1 lub o2
#    b) rozgalezienie, ciasny zakret w lewo -> symulacja przejscia -> dodaj ograniczenie o1 -> propagacja ograniczenia przez waski korytarz do 2.a
#    c) ciasny zakret w prawo z orientacja o1 -> brak mozliwosci przejscia -> sprawdz orientacje o2 -> przejscie mozliwe -> dodaj ograniczenie na o2,
#       sprzecznosc ograniczen w 2.c i 2.b,2.a, brak ciaglosci przejscia z 2.a do 2.c, nalezy dodac kolejne kroki
#    d) dodanie zmiany orientacji po 2.b -> zmiana mozliwa tylko w wolnej przestrzeni -> szukaj wolnej przestrzeni -> znaleziono za rozgalezieniem
#    e) zmiana orientacji po 2.b mozliwa w innym odgalezieniu, dodaj nowy cel posredni i ograniczenie orientacji na o2 przez powrotem do 2.c
#    f) teraz, z dodanym celem posrednim po 2.a, mozna usunac 2.b, gdyz zakret w lewo nie jest juz aktualny. Cel posredni moze zostac osiagniety na 2 sposoby:
#       ze skretem na rozgalezieniu w lewo lub w prawo. W zaleznosci od wariantu, ograniczeniem orientacji w 2.a jest tylko o1 lub tylko o2. Ograniczenie to
#       przestaje byc zwiazane z ograniczeniem orientacji w 2.c
#
# Pojecia na wielu poziomach abstrakcji:
# - korytarz
# - zakret_w_prawo
# - zakret_w_lewo
# - dlugi_korytarz (polaczone korytarze)
# Rozwiazania problemow:
# - rozne orientacje w dlugi_korytarz -> konieczna zmiana orientacji
# - zmiana orientacji w dlugi_korytarz -> wyszukanie mozliwej sciezki
# Pojecie dlugi_korytarz zostaje utworzone poprzez zaobserowawanie okreslonej konfiguracji pojec szczegolowych (korytarz).
# Rozpoznanie problemu odwoluje sie ograniczen zwiazanych z pojeciami szczegolowymi (zakret_w_prawo, zakret_w_lewo), przy czym to samo
# ograniczenie (np. o1 lub o2) moze byc zwiazane z roznymi pojeciami szczegolowymi.
#
# Plan hierarchiczny sklada sie z wezlow, z ktorych kazdy stanowi pewien podplan.
# Plan jest potencjalnym rozwiazaniem zadania.
# W przypadku powyzszym, dla znalezionego problemu ze zmiana orientacji w waskim dlugim korytarzu, mozna utworzyc nowe podzadanie:
# - przejdz z biezacej lokalizacji do tej samej lokalizacji z inna, okreslona orientacja (czyli zmien orientacje).
# Te podzadanie jest prostsze niz znajdowanie najkrotszej sciezki. Polega ono na przeszukaniu grafu wszerz i znalezieniu cyklu spelniajacego
# nastepujace zalozenia:
# - przechodzi przez biezace polozenie
# - przechodzi przez punkt, w ktorym mozna zmienic orientacje (wolna przestrzen lub odpowiednio ulozone skrzyzowanie T)
# - pomiedzy biezacym polozeniem, a punktem w ktorym mozna zmienic orientacje przechodza 2 siezki, ktore mozna pokonac z roznymi orientacjami
#
# Zadanie znajdowania najkrotszej sciezki dziala na wielu poziomach abstrakcji. W wersji podstawowej opiera sie ono na algorytmie Dijkstry,
# obliczajac dlugosc drogi do celu od poszczegolnych miejsc (wezlow grafu). Dopoki krawedzie spelniaja pewne zalozenia, stosowana jest
# podstawowa wersja algorytmu. Gdy jednak krawedz nie spelnia zalozen, brana pod uwage jest dodatkowa informacja, np. jesli przejscie
# pomiedzy miejscami (czyli krawedz) jest waskie, to dodawana jest np. informacja o mozliwych orientacjach robota w danym przejsciu.
# W tym przypadku, do kazdej krawedzi odpowiadajacej waskiemu przejsciu moze byc dodana informacja o ograniczeniach orientacji, czyli:
# - orientacja musi byc zachowana (brak mozliwosci zmiany orientacji)
# - orientacja musi byc rowna o1 (np. zakret w lewo)
# - orientacja musi byc rowna o2 (np. zakret w prawo)

def exportDotGraph(filename, graph):
    with open(filename, "w") as f:
        dot_str = "graph gr {\n"
        edges = set()
        for v1 in graph:
            for v2 in graph[v1]:
                if not (v1,v2) in edges and not (v2,v1) in edges:
                    edges.add( (v1,v2) )
        for v1,v2 in edges:
            dot_str += "    " + str(v1) + " -- " + str(v2) + ";\n"
        dot_str += "}\n"
        f.write(dot_str)

def exportDotDiGraph(filename, graph):
    with open(filename, "w") as f:
        dot_str = "digraph gr {\n"
        for v1 in graph:
            for v2 in graph[v1]:
                dot_str += "    " + str(v1) + " -> " + str(v2[1]) + "[label=\"" + v2[0] + "\"];\n"
        dot_str += "}\n"
        f.write(dot_str)

class PathPoint:
    def __init__(self):
        self.place_types = []
        self.place = None

class Path:
    def __init__(self):
        self.type = "path_instance"
        self.path_id = None
        self.points = []
        self.objects = {}
        self.constraints = {}

    def addConstraint(self, v, constraint, reason):
        if not v in self.constraints:
            self.constraints[v] = []
        self.constraints[v].append( (constraint, reason) )

class Corridor:
    def __init__(self):
        self.scope = "path"
        self.name = "corridor"
        self.requires = []

    def detect(self, path):
        #path.objects["corridor"] = []
        corr_idx = 0
        for p_idx in range(len(path.points)):
            pl = path.points[p_idx].place
            if pl == "k1" or pl == "k2" or pl == "k3" or pl == "s1" or pl == "z1":
                path.points[p_idx].place_types.append( "corridor" )
                path.objects[("corridor", corr_idx)] = [ ("node", p_idx) ]
                #path.addConstraint( ("node", p_idx), "oX", ("corridor", corr_idx) )
                corr_idx = corr_idx + 1

    def getConstraints(self):
        return [ "oX" ]

    def checkConstraints(self, path):
        pass

class Junction:
    def __init__(self):
        self.scope = "path"
        self.name = "junction"
        self.requires = []

    def detect(self, path):
        #path.objects["junction"] = []
        junction_idx = 0
        for p_idx in range(len(path.points)):
            pl = path.points[p_idx].place
            if pl == "s1":
                path.points[p_idx].place_types.append( "junction" )
                #path.objects["junction"].append( ("node", p_idx) )
                path.objects[("junction", junction_idx)] = [ ("node", p_idx) ]
                junction_idx = junction_idx + 1

    def checkConstraints(self, path):
        pass

class FreeSpace:
    def __init__(self):
        self.scope = "path"
        self.name = "free_space"
        self.requires = []

    def detect(self, path):
        #path.objects["free_space"] = []
        free_space_idx = 0
        for p_idx in range(len(path.points)):
            pl = path.points[p_idx].place
            if pl == "start" or pl == "cel" or pl == "w1":
                path.points[p_idx].place_types.append( "free_space" )
                #path.objects["free_space"].append( ("node", p_idx) )
                path.objects[("free_space", free_space_idx)] = [ ("node", p_idx) ]
                free_space_idx = free_space_idx + 1

    def checkConstraints(self, path):
        pass

class LeftTurn:
    def __init__(self):
        self.scope = "path"
        self.name = "left_turn"
        self.requires = []

    def detect(self, path):
        #path.objects["left_turn"] = []
        left_turn_idx = 0
        for p_idx in range(1, len(path.points)-1):
            p1 = path.points[p_idx-1].place
            p2 = path.points[p_idx].place
            p3 = path.points[p_idx+1].place
            if (p1 == "cel" and p2 == "z1" and p3 == "k2") or \
                    (p1 == "k1" and p2 == "s1" and p3 == "k2") or \
                    (p1 == "k3" and p2 == "s1" and p3 == "k1"):
                path.points[p_idx].place_types.append( "left_turn" )
                #path.objects["left_turn"].append( ("node", p_idx) )
                path.objects[("left_turn", left_turn_idx)] = [ ("node", p_idx-1), ("node", p_idx), ("node", p_idx+1) ]
                path.addConstraint( ("node", p_idx), "o1", ("left_turn", left_turn_idx) )
                left_turn_idx = left_turn_idx + 1

    def getConstraints(self):
        return [ "o1" ]

    def checkConstraints(self, path):
        pass

class RightTurn:
    def __init__(self):
        self.scope = "path"
        self.name = "right_turn"
        self.requires = []

    def detect(self, path):
        #path.objects["right_turn"] = []
        right_turn_idx = 0
        for p_idx in range(1, len(path.points)-1):
            p1 = path.points[p_idx-1].place
            p2 = path.points[p_idx].place
            p3 = path.points[p_idx+1].place
            if (p1 == "k2" and p2 == "z1" and p3 == "cel") or \
                    (p1 == "k2" and p2 == "s1" and p3 == "k1") or \
                    (p1 == "k1" and p2 == "s1" and p3 == "k3"):
                path.points[p_idx].place_types.append( "right_turn" )
                #path.objects["right_turn"].append( ("node", p_idx) )
                path.objects[("right_turn", right_turn_idx)] = [ ("node", p_idx-1), ("node", p_idx), ("node", p_idx+1) ]
                path.addConstraint( ("node", p_idx), "o2", ("right_turn", right_turn_idx) )
                right_turn_idx = right_turn_idx + 1

    def getConstraints(self):
        return [ "o2" ]

    def checkConstraints(self, path):
        pass

class LongCorridor:
    def __init__(self):
        self.scope = "path"
        self.name = "long_corridor"
        self.requires = [ "corridor" ]

    def detect(self, path):
        assert path.type == "path_instance"

        long_corridors = []
        corr_begin = None
        for p_idx in range(len(path.points)):
            if "corridor" in path.points[p_idx].place_types:
                if corr_begin is None:
                    corr_begin = p_idx
                corr_end = p_idx
            else:
                if not corr_begin is None:
                    long_corridors.append( (corr_begin, corr_end) )
                    #self.addLongCorridor(path_id, corr_begin, corr_end)
                    corr_begin = None

        if not corr_begin is None:
            long_corridors.append( (corr_begin, corr_end) )
            #self.addLongCorridor(path_id, corr_begin, corr_end)

#        path.objects["long_corridor"] = long_corridors
#        path.long_corridors = long_corridors
        #path.objects["long_corridor"] = []
#                path.objects[("right_turn", right_turn_idx)] = [ ("node", p_idx-1), ("node", p_idx), ("node", p_idx+1) ]
#                right_turn_idx = right_turn_idx + 1

        for long_corr_idx in range(len(long_corridors)):
            corr_begin, corr_end = long_corridors[long_corr_idx]
            path.objects[("long_corridor", long_corr_idx)] = []
            for p_idx in range(corr_begin, corr_end+1):
                corridor_id = None
                for key in path.objects:
                    if key[0] == "corridor":
                        if path.objects[key][0][1] == p_idx:
                            corridor_id = key
                            break
                assert not corridor_id is None
                #path.points[p_idx].place_types.append( ("long_corridor", long_corr_idx) )
                path.objects[("long_corridor", long_corr_idx)].append( corridor_id )

    def getConstraints(self):
        return [ "same_o" ]

    def checkConstraints(self, path):
        conflicts = []
        for key in path.objects:
            if key[0] != "long_corridor":
                continue

            ori = None
            prev_node_obj = None
            for corr_obj in path.objects[key]:
                node_obj = path.objects[corr_obj][0]
                if not node_obj in path.constraints:
                    continue
                for constraint, reason in path.constraints[node_obj]:
                    if constraint == "o1":
                        if ori == "o2":
                            conflicts.append( ("same_o", prev_node_obj, node_obj) )
                        ori = "o1"
                        prev_node_obj = node_obj
                        break

                for constraint, reason in path.constraints[node_obj]:
                    if constraint == "o2":
                        if ori == "o1":
                            conflicts.append( ("same_o", prev_node_obj, node_obj) )
                        ori = "o2"
                        prev_node_obj = node_obj
                        break
        print conflicts

        for conf in conflicts:
            if conf[0] == "same_o":
                node_s = conf[1]
                node_e = conf[2]
                # TODO: create suggestion: run "change orientation" behavior
                #for p_idx in range(node_s[1], node_e[1]+1):
                #    if "junction" in path.points[p_idx].place_types:
                #        print "try to use junction at node ", p_idx

    def checkConstraints2(self, path):
        for long_corr_idx in range(len(path.long_corridors)):
            corr_begin, corr_end = long_corridors[long_corr_idx]

            ori = "oX"
            ori_idx = None
            for p_idx in range(corr_begin, corr_end+1):
                if "o1" in path.points[p_idx].constraints[p_idx]:
                    if ori == "o2":
                        conflicts.append( ("same_o", ori_idx, p_idx ) )
                    ori = "o1"
                    ori_idx = p_idx
                elif "o2" in path.points[p_idx].constraints[p_idx]:
                    if ori == "o1":
                        conflicts.append( ("same_o", ori_idx, p_idx ) )
                    ori = "o2"
                    ori_idx = p_idx

class PlaningGraph:
    def __init__(self, places_graph):
        # TODO: determine sequence of operators automatically (using 'requires' member)
        self.operators = [Corridor(), Junction(), FreeSpace(), LeftTurn(), RightTurn(), LongCorridor()]
        self.operators_map = {}
        for op in self.operators:
            self.operators_map[op.name] = op

        self.places_graph = copy.deepcopy(places_graph)
#        self.graph = {}
#        for v1 in places_graph:
#            for v2 in places_graph[v1]:
#                self.addBiEdge("place_" + str(v1), "place_" + str(v2), "connects")
#        self.path_id = 0

        self.places = {}
        self.ids = {}
        self.paths = {}

    def getPath(self, path_id):
        assert path_id in self.paths
        return self.paths[path_id]

#    def addEdge(self, v1, v2, name):
#        if not v1 in self.graph:
#            self.graph[v1] = set()
#        self.graph[v1].add( (name, v2) )

#    def addBiEdge(self, v1, v2, name):
#        self.addEdge(v1, v2, name)
#        self.addEdge(v2, v1, name)

    def addPath(self, path):
        #places = self.recognizePlaces( path )
        p = Path()
        #p.objects["node"] = []
        p.path_id = self.getNextId( "path" )
        for p_idx in range(len(path)):
            p.objects[("node", p_idx)] = [ ("place", path[p_idx]) ]
            pt = PathPoint()
            assert path[p_idx] in self.places_graph
            pt.place = path[p_idx]
            #pt.place_types = places[p_idx]
            p.points.append( pt )
        self.paths[p.path_id] = p
        #self.findLongCorridors( p.path_id )
        for op in self.operators:
            op.detect( p )
        print p.objects
        for op in self.operators:
            op.checkConstraints( p )
        return

        path_name = "path_" + str(path_id)
        for p_idx in range(len(path)):
            node_id = path_name + "_" + str(p_idx)
            self.addEdge( path_name, node_id, "contains" )
            self.addEdge( node_id, path_name, "belongs_to" )

            self.addEdge( node_id, "place_" + path[p_idx], "is" )
            if p_idx > 0:
                prev_node_id = path_name + "_" + str(p_idx-1)
                self.addEdge(prev_node_id, node_id, "forw")
                self.addEdge(node_id, prev_node_id, "back")
        self.recognizePlaces(path)

    def exportPathObjectsDotGraph(self, filename, path):
        dot_str = "digraph gr {\n"
        for v1 in path.objects:
            for v2 in path.objects[v1]:
                dot_str += "    " + v1[0] + "_" + str(v1[1]) + " -> " + v2[0] + "_" + str(v2[1]) + ";\n"

        for v1 in path.constraints:
            for v2 in path.constraints[v1]:
                constraint, reason = v2
                dot_str += "    " + v1[0] + "_" + str(v1[1]) + " -> " + constraint[0] + "_" + str(constraint[1]) + ";\n"
                dot_str += "    " + constraint[0] + "_" + str(constraint[1]) + " [shape=\"rectangle\"];\n"
                dot_str += "    " + constraint[0] + "_" + str(constraint[1]) + " -> " + reason[0] + "_" + str(reason[1]) + " [style=\"dashed\"];\n"

        dot_str += "}\n"
        with open(filename, "w") as f:
            f.write(dot_str)
        
    def getNextId(self, entity_name):
        if not entity_name in self.ids:
            self.ids[entity_name] = 0
        self.ids[entity_name] += 1
        return self.ids[entity_name]

    def addPlace(self, place_type, place):
        if not place_type in self.places:
            self.places[place_type] = []
        self.places[place_type].append( place )

#    def addLongCorridor(self, path_id, corr_begin, corr_end):
#        path = self.getPath(path_id)
#        corridor_idx = self.getNextId( "long_corridor" )
#        self.addPlace( "long_corridor", (corridor_idx, path_id, corr_begin, corr_end) )
#        for p_idx in range(corr_begin, corr_end+1):
#            path.points[p_idx].place_types.append( ("long_corridor", corridor_idx) )

def main():
    sys.setrecursionlimit(10000)

    if False:
        plot_data_x = []
        plot_data_y = []
        for angle in np.linspace(0.0, math.pi*2.0, 500):
            plot_data_x.append( angle )
            plot_data_y.append( stats.norm.pdf(angle, math.pi/4.0, math.pi/16.0) + stats.norm.pdf(angle, math.pi/4.0+math.pi, math.pi/16.0) )

        plt.plot( plot_data_x, plot_data_y )
        plt.text(math.pi/4.0, stats.norm.pdf(0, 0, math.pi/16.0), "A")
        plt.text(math.pi/4.0+math.pi, stats.norm.pdf(0, 0, math.pi/16.0), "B")
        plt.show()
        exit(0)

    if False:
        places_graph = {
            "start" : [ "k1" ],
            "k1" : [ "start", "s1" ],
            "s1" : [ "k1", "k2", "k3" ],
            "k2" : [ "s1", "z1" ],
            "k3" : [ "s1", "w1" ],
            "w1" : [ "k3" ],
            "z1" : [ "k2", "cel" ],
            "cel" : [ "z1" ],
        }

        #def transformGraph(places_graph):
        #tf_graph = transformGraph(places_graph)

        def searchPath(places_graph, start, end):
            open_set = set()
            closed_set = set()
            open_set.add( start )
            while len(open_set) > 0:
                print open_set
                new_open_set = set()
                for v1 in open_set:
                    for v2 in places_graph[v1]:
                        if not v2 in open_set and not v2 in closed_set:
                            new_open_set.add( v2 )
                closed_set = closed_set.union( open_set )
                open_set = new_open_set

        searchPath(places_graph, "start", "cel")
        exit(0)

        def getPlacesByType(place_type):
            if place_type == "wolna_przestrzen":
                return [ "start", "cel", "w1" ]
            elif place_type == "skrzyzowanie":
                return [ "s1" ]

        def getPath(start, end):
            if start == "start" and end == "cel":
                return [ "start", "k1", "s1", "k2", "z1", "cel" ]
            elif start == "s1" and end == "w1":
                return [ "s1", "k3", "w1" ]
            elif start == "w1" and end == "s1":
                return [ "w1", "k3", "s1" ]
            elif start == "k2" and end == "w1":
                return [ "k2", "s1", "k3", "w1" ]
            elif start == "w1" and end == "k2":
                return [ "w1", "k3", "s1", "k2" ]
            elif start == "w1" and end == "cel":
                return [ "w1", "k3", "s1", "k2", "z1", "cel" ]
            raise Exception("case not implemented: start='" + start + "', end='" + end + "'", )

        exportDotGraph("/home/dseredyn/ws_stero/places_graph.dot", places_graph)

        planning_graph = PlaningGraph( places_graph )

        path = getPath("start", "cel")

        planning_graph.addPath(path)
        print "places", planning_graph.places
        p = planning_graph.getPath(1)
        for p_idx in range(len(p.points)):
            print p_idx, p.points[p_idx]. place, p.points[p_idx].place_types

        planning_graph.exportPathObjectsDotGraph("/home/dseredyn/ws_stero/planning_graph.dot", p)

#        places = recognizePlaces(path)
#        planning_graph.setRecognizedPlaces(places)

#        exportDotDiGraph("/home/dseredyn/ws_stero/planning_graph.dot", planning_graph.graph)

        exit(0)

        constraints = {
            "zakret_w_prawo" : [ "o1" ],
            "zakret_w_lewo" : [ "o2" ],
            "korytarz" : [ "oX" ],
            "long_corridor" : [ "same_o" ],
        }

        def getConstraints(places):
            constraints = []
            for p_idx in range(len(places)):
                constr_list = []
                for pl_name in places[p_idx]:
                    if pl_name == "zakret_w_prawo":
                        constr_list.append( "o1" )
                    elif pl_name == "zakret_w_lewo":
                        constr_list.append( "o2" )
                    elif pl_name == "korytarz":
                        constr_list.append( "oX" )
                constraints.append( constr_list )
            return constraints

        constraints = getConstraints(places)
        for p_idx in range(len(path)):
            print p_idx, path[p_idx], places[p_idx], constraints[p_idx]

        def getConflicts(places, constraints):
            # konflikt jest wtedy, kiedy w jednym korytarzu mamy rozne orientacje (tj. o1 i o2)
            corridors = []
            corr_begin = None
            for p_idx in range(len(places)):
                if "korytarz" in places[p_idx]:
                    if corr_begin == None:
                        corr_begin = p_idx
                    corr_end = p_idx
                else:
                    if corr_begin != None:
                        corridors.append( (corr_begin, corr_end) )
                        corr_begin = None
            if corr_begin != None:
                corridors.append( (corr_begin, len(places)-1) )

            conflicts = []
            for corr_begin, corr_end in corridors:
                ori = "oX"
                ori_idx = None
                for p_idx in range(corr_begin, corr_end+1):
                    if "o1" in constraints[p_idx]:
                        if ori == "o2":
                            conflicts.append( ("orientacje", ori_idx, p_idx ) )
                        ori = "o1"
                        ori_idx = p_idx
                    elif "o2" in constraints[p_idx]:
                        if ori == "o1":
                            conflicts.append( ("orientacje", ori_idx, p_idx ) )
                        ori = "o2"
                        ori_idx = p_idx

            return conflicts

        conflicts = getConflicts(places, constraints)
        print "conflicts", conflicts

        def getSolutions( conflicts ):
            solutions = []
            for c in conflicts:
                if c[0] == "orientacje":
                    solutions.append( ("do_wolnej_przestrzeni", c[1], c[2]) )
            return solutions
        solutions = getSolutions( conflicts )
        print "solutions", solutions

        def applySolutions( path, solutions ):
            for s in solutions:
                if s[0] == "do_wolnej_przestrzeni":
                    wp_places = getPlacesByType("wolna_przestrzen")
                    for p_idx in range(s[1], s[2]):
                        for wp in wp_places:
                            if wp in path:#[0:path[s[1]]]:
                                # this place is already in the path, so ignore it, as it does not solve the conflict
                                continue
                            print getPath(path[p_idx], wp)

        applySolutions(path, solutions)
        exit(0)        

    width, height = 690,600

    ### PyGame init
    pygame.init()
    screen = pygame.display.set_mode((width,height)) 
    clock = pygame.time.Clock()
    running = True
    font = pygame.font.SysFont("Arial", 16)
    
    ### Physics stuff
    space = pymunk.Space()   
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    env_xml = EnvironmentXml("/home/dseredyn/ws_stero/src/multilevel_planning/data/environments/env_03.xml")
    models = env_xml.getModels()

    #env = env3

    for model in models:
        model.addToSpace(space)

    damped_objects_list = []
    for model in models:
        if not model.damping is None:
            damped_objects_list.append( (model.body, model.damping[0], model.damping[1]) )

#    cab = ModelCabinet(200, 100, Vec2d(200,200), -0.05)
#    cab.addToSpace(space)

#    box = ModelBox(50, 10, Vec2d(200,200), 0*math.pi/4.0)
#    box.addToSpace( space )

#    damped_objects_list.append( (box.body, 300.0, 3000.0) )

    # robot
#    rob = ModelRobot( Vec2d(100,100), 0.0 )
#    rob = ModelRobot( Vec2d(220,225), 0.0 )
#    rob.addToSpace(space)
    for ob in models:
        if ob.name == "robot":
            rob = ob
        if ob.name == "box":
            box = ob

    control_mode = "target_pos"
    position_control = ManualPositionControl(1000.0, 500.0, 50000.0, 50000.0)

    control_mode = "auto"

    dest_point = Vec2d(550,270)

    info_exp_motion = InformationExpectedMotion()
    info_exp_motion.anchor = "instance_box"
    info_exp_motion.target_point = Vec2d(500,500)

    # provide information
    information = [
        InformationSpatialMapRange( (0, width), 100, (0, height), 100 ),        # range and resolution of spatial map
        InformationRobotPose( rob.robot_body.position, rob.robot_body.angle ),  # current pose of robot
        InformationPerceptionSpace( space ),                                    # percepted environment (exact, full perception)
#        InformationDestinationGeom( dest_point ),                               # target point for motion
#        InformationOpenDoorCommand(),
        info_exp_motion,
    ]

    behaviors = [
        BehaviorSpatialMapGeneration(models),
#        BehaviorSpaceDigging(env),
#        BehaviorSpatialPathPlanner(),
#        BehaviorMoveTowards(),
#        BehaviorObstacleAvoidance(),
#        BehaviorTightPassage(),
#        BehaviorDoorPerception(cab.left_door),
#        BehaviorOpenDoor(),
#        BehaviorMoveObject(),
        BehaviorObjectPerception(box, "box"),
        BehaviorPushObject(),
        BehaviorPushExecution(),
        BehaviorRobotControl(),
    ]

    behaviors[0].update(information)
    behaviors[0].plotTriangulation()
    behaviors[0].plotGraph()
#    plt.show()
#    behaviors[1].update(information)
#    behaviors[1].plotTriangulation()
#    behaviors[1].plotGraph()
#    behaviors[1].plotOccludedSimplices()
#    behaviors[1].plotBorder()
#    plt.show()
#    exit(0)

    generateBehaviorsDotGraph('/home/dseredyn/svn/phd/ds/doktorat/rozwazania_2018_04/img/zachowania.dot', behaviors)

    first_behavior_iteration = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or \
                event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):  
                running = False

        debug_info = []

        keys = pygame.key.get_pressed()
        # manual control
        if control_mode == "force":
            manualForceControl(rob.robot_body, keys)
        elif control_mode == "target_pos":
            position_control.update( rob.robot_body, keys )
        elif control_mode == "auto":
            position_control.update( rob.robot_body, keys )
            joinInformation( information, [ InformationRobotPose( rob.robot_body.position, rob.robot_body.angle ) ] )   # current pose of robot

            active_behaviors = []
            for b in behaviors:
                new_inf = b.update( information )
                joinInformation( information, new_inf )
                if len(new_inf) > 0:
                    active_behaviors.append( b.name )
#                    print new_inf[0].type
            inf_list = []
            for inf in information:
                inf_list.append( inf.type )
                if inf.type == "robot_total_control":
                    lin_damping = 300.0
                    lin_stiffness = 5000.0
                    rot_damping = 5000.0
                    rot_stiffness = 5000.0
                    #print inf.force, inf.torque
                    rob.robot_body.force = lin_stiffness*inf.force - lin_damping * rob.robot_body.velocity
                    rob.robot_body.torque = rot_stiffness*inf.torque - rot_damping * rob.robot_body.angular_velocity
            #print inf.torque
            #print inf_list
            # TODO: fix removing old information wrt. behavior execution sequence
            clearObsoleteInformation( information )
            #print active_behaviors
#        print "plan_idx", plan_idx
        #cab.debugVis(debug_info)

        #rob.debugVisQhull(debug_info)
        #rob.debugVisPushing(debug_info)

        applyDamping( damped_objects_list )

        mouse_position = pymunk.pygame_util.from_pygame( Vec2d(pygame.mouse.get_pos()), screen )
        mouse_position_munk = MunkToGame(mouse_position, height)

        ### Clear screen
        screen.fill(pygame.color.THECOLORS["black"])
        
        ### Draw stuff
        space.debug_draw(draw_options)

        if control_mode == "target_pos":
            position_control.debugInfoDraw( debug_info )
        elif control_mode == "auto":
            #behaviors[2].debugVisDraw(debug_info)
            #drawDebugCircle(debug_info, "green", 5, dest_point)
            for b in behaviors:
                if b.name == "push_object":
                    b.debugVisDraw(debug_info)

        # draw debug info
        drawDebugInfo(screen, height, debug_info)

        # Info and flip screen
        screen.blit(font.render("fps: " + str(clock.get_fps()), 1, THECOLORS["white"]), (0,0))
        screen.blit(font.render("Mouse position (in world coordinates): " + str(mouse_position[0]) + "," + str(mouse_position[1]), 1, THECOLORS["darkgrey"]), (5,height - 35))
        screen.blit(font.render("Press ESC or Q to quit", 1, THECOLORS["darkgrey"]), (5,height - 20))
        
        pygame.display.flip()
        
        ### Update physics
        fps = 60
        dt = 1./fps
        space.step(dt)
        
        clock.tick(fps)

if __name__ == '__main__':
    sys.exit(main())
