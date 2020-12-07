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
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
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

find_package(Boost COMPONENTS system filesystem python REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

if(BUILD_DBoW2)
  set(LIB_SHARED "SHARED")
  if(WIN32)
    set(LIB_SHARED "STATIC")
  endif(WIN32)
  add_library(${PROJECT_NAME} ${LIB_SHARED} ${SRCS})
  target_include_directories(${PROJECT_NAME} PUBLIC include/DBoW2/ include/)
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
endif(BUILD_DBoW2)

if(BUILD_Demo)
  add_executable(demo demo/demo.cpp)
  target_link_libraries(demo ${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(demo PROPERTIES CXX_STANDARD 11)
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






