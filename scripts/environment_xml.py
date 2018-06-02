import xml.dom.minidom as minidom

from simulation_models import *

class Node:
    def __init__(self):
        self.node_type = None

class EnvironmentXml:
    def __init__(self, filename):
        xml_str = None
        with open(filename, "r") as f:
            xml_str = f.read()
        dom = minidom.parseString(xml_str)
        if False:
            print "COMMENT_NODE", dom.COMMENT_NODE
            print "ELEMENT_NODE", dom.ELEMENT_NODE
            print "ATTRIBUTE_NODE", dom.ATTRIBUTE_NODE
            print "TEXT_NODE", dom.TEXT_NODE
            print "CDATA_SECTION_NODE", dom.CDATA_SECTION_NODE
            print "ENTITY_NODE", dom.ENTITY_NODE
            print "PROCESSING_INSTRUCTION_NODE", dom.PROCESSING_INSTRUCTION_NODE
            print "COMMENT_NODE", dom.COMMENT_NODE
            print "DOCUMENT_NODE", dom.DOCUMENT_NODE
            print "DOCUMENT_TYPE_NODE", dom.DOCUMENT_TYPE_NODE
            print "NOTATION_NODE", dom.NOTATION_NODE

        self.models = []
        node_list = []
        for n in dom.childNodes:
            if n.nodeType == n.ELEMENT_NODE:
                if n.tagName == "world":
                    self.parseWorld( n )

    def parseWorld(self, xml):
        assert xml.tagName == "world"

        node_list = []
        for n in xml.childNodes:
            if n.nodeType == n.ELEMENT_NODE:
                if n.tagName == "segment":
                    self.parseSegment( n )
                elif n.tagName == "box":
                    self.parseBox( n )
                elif n.tagName == "cabinet":
                    self.parseCabinet( n )
                elif n.tagName == "robot":
                    self.parseRobot( n )

    def parseSegment(self, xml):        
        assert xml.tagName == "segment"

        a_str = xml.getAttribute("a")
        a_split = a_str.split()
        a = Vec2d( float(a_split[0]), float(a_split[1]) )

        b_str = xml.getAttribute("b")
        b_split = b_str.split()
        b = Vec2d( float(b_split[0]), float(b_split[1]) )

        width_str = xml.getAttribute("width")
        width = float(width_str)

        friction_str = xml.getAttribute("friction")
        friction = float(friction_str)

        static_str = xml.getAttribute("static")
        static = bool(static_str)

        self.models.append( ModelStaticWall( a, b, width, friction) )

    def parseBox(self, xml):        
        assert xml.tagName == "box"

        name = xml.getAttribute("name")

        dim_str = xml.getAttribute("dim")
        dim_split = dim_str.split()
        dim = Vec2d( float(dim_split[0]), float(dim_split[1]) )

        position_str = xml.getAttribute("position")
        position_split = position_str.split()
        position = Vec2d( float(position_split[0]), float(position_split[1]) )

        angle_str = xml.getAttribute("angle")
        angle = float(angle_str)

        static_str = xml.getAttribute("static")
        static = bool(static_str)

        damping_str = xml.getAttribute("damping")
        if damping_str is None:
            damping = None
        else:
            damping_split = damping_str.split()
            damping = (float(damping_split[0]), float(damping_split[1]))
            
        self.models.append( ModelBox(name, dim[0], dim[1], position, angle, damping=damping) )

    def parseCabinet(self, xml):        
        assert xml.tagName == "cabinet"

        name = xml.getAttribute("name")

        width_str = xml.getAttribute("width")
        width = float(width_str)

        depth_str = xml.getAttribute("depth")
        depth = float(depth_str)

        position_str = xml.getAttribute("position")
        position_split = position_str.split()
        position = Vec2d( float(position_split[0]), float(position_split[1]) )

        angle_str = xml.getAttribute("angle")
        angle = float(angle_str)

        cabinet_case = ModelCabinetCase(name, width, depth, position, angle)
        for m in cabinet_case.getModels():
            self.models.append( m )
        #self.models.append( ModelDoor(name + "_left_door", True, cabinet_case.width, cabinet_case.line_width, cabinet_case.hinge_pos_W, cabinet_case.rotation, 0) )

    def parseRobot(self, xml):        
        assert xml.tagName == "robot"

        position_str = xml.getAttribute("position")
        position_split = position_str.split()
        position = Vec2d( float(position_split[0]), float(position_split[1]) )

        angle_str = xml.getAttribute("angle")
        angle = float(angle_str)

        self.models.append( ModelRobot( position, angle) )

    def getModels(self):
        return self.models

#    def parseNode(self, xml):
#        result = Node()
#        for n in xml.childNodes:
#            if n.nodeType == n.ELEMENT_NODE:
#                if n.tagName == "type":
#                    result.node_type = self.parseTextNode(n)
#                elif n.tagName == "contains":
#                    pass
#                elif n.tagName == "constraints":
#                    pass
#        return result

#    def parseTextNode(self, xml):
#        assert len(xml.childNodes) == 1
#        assert xml.childNodes[0].nodeType == xml.TEXT_NODE
#        return xml.childNodes[0].data

#def str_to_bool(s):
#    if s.upper() == 'TRUE':
#        return True
#    if s.upper() == 'FALSE':
#        return False
#    raise ValueError("Wrong boolean value: " + s)


