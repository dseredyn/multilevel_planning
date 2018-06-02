cmake_minimum_required(VERSION 2.8)
project(multilevel_planning)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

find_package(catkin REQUIRED cv_bridge planer_utils)

find_package(Qhull REQUIRED)
if (HAVE_QHULL_2011)
  add_definitions(-DGEOMETRIC_SHAPES_HAVE_QHULL_2011)
endif()

find_package(Eigen3 REQUIRED)

find_package(OpenCV 3 REQUIRED)
if (${OpenCV_VERSION} MATCHES "3.3.1")
  foreach(__cvcomponent ${OpenCV_LIB_COMPONENTS})
    set (__original_cvcomponent ${__cvcomponent})
    if(NOT __cvcomponent MATCHES "^opencv_")
      set(__cvcomponent opencv_${__cvcomponent})
    endif()
    if (TARGET ${__cvcomponent})
      set_target_properties(${__cvcomponent} PROPERTIES
          MAP_IMPORTED_CONFIG_DEBUG ""
          MAP_IMPORTED_CONFIG_RELEASE ""
          MAP_IMPORTED_CONFIG_RELWITHDEBINFO ""
          MAP_IMPORTED_CONFIG_MINSIZEREL ""
      )
    endif()
  endforeach(__cvcomponent)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

catkin_package(CATKIN_DEPENDS roscpp cv_bridge planer_utils)

include_directories(${catkin_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS})

#message(WARNING "${catkin_LIBRARIES}")
#message(WARNING "${OpenCV_LIBRARIES}")

#add_executable(camera_calibration src/camera_calibration.cpp)
#target_link_libraries(camera_calibration ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

#add_executable(2d_slam src/2d_slam.cpp src/pdf.cpp)
#target_link_libraries(2d_slam ${catkin_LIBRARIES} ${QHULL_LIBRARIES} ${OpenCV_LIBRARIES})

#install(DIRECTORY data/
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/data
#  )

#install(DIRECTORY launch/
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
#  )

#catkin_install_python(PROGRAMS scripts/test1.py
#DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

