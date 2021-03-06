cmake_minimum_required(VERSION 2.8)
project(DBoW2)
include(ExternalProject)

option(BUILD_DBoW2   "Build DBoW2"            ON)
option(BUILD_Demo    "Build demo application" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -std=c++11")
endif()

set(HDRS
  include/DBoW2/BowVector.h           include/DBoW2/FBrief.h
  include/DBoW2/QueryResults.h        include/DBoW2/TemplatedDatabase.h   include/DBoW2/FORB.h
  include/DBoW2/DBoW2.h               include/DBoW2/FClass.h              include/DBoW2/FeatureVector.h
  include/DBoW2/ScoringObject.h       include/DBoW2/TemplatedVocabulary.h)
set(SRCS 
  src/BowVector.cpp     src/FBrief.cpp        src/FORB.cpp
  src/FeatureVector.cpp src/QueryResults.cpp  src/ScoringObject.cpp)

set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost COMPONENTS system filesystem python numpy REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

find_package(DLib REQUIRED
  PATHS ${DEPENDENCY_INSTALL_DIR})
#if(${DLib_FOUND})
#message("DLib library found, using it from the system")
include_directories(${DLib_INCLUDE_DIRS})
add_custom_target(Dependencies)
  #else(${DLib_FOUND})
  #message("DLib library not found in the system, it will be downloaded on build")
  #option(DOWNLOAD_DLib_dependency "Download DLib dependency" ON)
  #if(${DOWNLOAD_DLib_dependency})
  #  ExternalProject_Add(DLib
  #    PREFIX ${DEPENDENCY_DIR}
  #    GIT_REPOSITORY http://github.com/dorian3d/DLib
  #    GIT_TAG master
  #    INSTALL_DIR ${DEPENDENCY_INSTALL_DIR}
  #    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)
  #  add_custom_target(Dependencies ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR} DEPENDS DLib)
  #else()
  #  message(SEND_ERROR "Please, activate DOWNLOAD_DLib_dependency option or download manually")
  #endif(${DOWNLOAD_DLib_dependency})
  #endif(${DLib_FOUND})

if(BUILD_DBoW2)
  add_library(${PROJECT_NAME} SHARED ${SRCS})
  include_directories(include/DBoW2/)
  add_dependencies(${PROJECT_NAME} Dependencies)
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${DLib_LIBS})
endif(BUILD_DBoW2)

if(BUILD_Demo)
  add_executable(demo demo/demo.cpp)
  target_link_libraries(demo ${PROJECT_NAME} ${OpenCV_LIBS} ${DLib_LIBS})
  file(COPY demo/images DESTINATION ${CMAKE_BINARY_DIR}/)
endif(BUILD_Demo)

configure_file(src/DBoW2.cmake.in
  "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_DBoW2)
  install(DIRECTORY include/DBoW2 DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2/)
install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)



#### Python wrapper #################
find_package( PythonLibs 2.7 REQUIRED)

#find_package( Boost COMPONENTS python REQUIRED )

execute_process( COMMAND export CPLUS_INCLUDE_PATH="${PYTHON_INCLUDE_DIRS}" )

include_directories(		
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/include/DBoW2
	${PYTHON_INCLUDE_DIRS}
	${Boost_INCLUDE_DIR}
)

add_library(pydbow2 SHARED
	${CMAKE_CURRENT_SOURCE_DIR}/src/pywrap.cpp
)

target_link_libraries(pydbow2
	${OpenCV_LIBS} 
	${DLib_LIBS}
	${PROJECT_NAME}
	${Python_LIBRARIES}
	${Boost_LIBRARIES}
)

# don't prepend wrapper library name with lib
set_target_properties( pydbow2 PROPERTIES PREFIX "" )

# Speed Test

add_executable(speed-test speed-test/speed-test.cpp)
target_link_libraries(speed-test
	${PROJECT_NAME}
	${OpenCV_LIBS}
	${DLib_LIBS}
	${Boost_LIBRARIES}	
)


add_executable(vary-db-size speed-test/vary-db-size.cpp)
target_link_libraries(vary-db-size
	${PROJECT_NAME}
	${OpenCV_LIBS}
	${DLib_LIBS}
	${Boost_LIBRARIES}	
)

execute_process(
	 COMMAND tar -xzf ORBvoc.txt.tar.gz
	 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Vocabulary
)
file(COPY Vocabulary/ORBvoc.txt DESTINATION ${CMAKE_BINARY_DIR}/)






