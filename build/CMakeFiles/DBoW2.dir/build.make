# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nate/Libraries/DBoW2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nate/Libraries/DBoW2/build

# Include any dependencies generated for this target.
include CMakeFiles/DBoW2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/DBoW2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DBoW2.dir/flags.make

CMakeFiles/DBoW2.dir/src/BowVector.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/src/BowVector.cpp.o: ../src/BowVector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nate/Libraries/DBoW2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DBoW2.dir/src/BowVector.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DBoW2.dir/src/BowVector.cpp.o -c /home/nate/Libraries/DBoW2/src/BowVector.cpp

CMakeFiles/DBoW2.dir/src/BowVector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/src/BowVector.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nate/Libraries/DBoW2/src/BowVector.cpp > CMakeFiles/DBoW2.dir/src/BowVector.cpp.i

CMakeFiles/DBoW2.dir/src/BowVector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/src/BowVector.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nate/Libraries/DBoW2/src/BowVector.cpp -o CMakeFiles/DBoW2.dir/src/BowVector.cpp.s

CMakeFiles/DBoW2.dir/src/BowVector.cpp.o.requires:

.PHONY : CMakeFiles/DBoW2.dir/src/BowVector.cpp.o.requires

CMakeFiles/DBoW2.dir/src/BowVector.cpp.o.provides: CMakeFiles/DBoW2.dir/src/BowVector.cpp.o.requires
	$(MAKE) -f CMakeFiles/DBoW2.dir/build.make CMakeFiles/DBoW2.dir/src/BowVector.cpp.o.provides.build
.PHONY : CMakeFiles/DBoW2.dir/src/BowVector.cpp.o.provides

CMakeFiles/DBoW2.dir/src/BowVector.cpp.o.provides.build: CMakeFiles/DBoW2.dir/src/BowVector.cpp.o


CMakeFiles/DBoW2.dir/src/FBrief.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/src/FBrief.cpp.o: ../src/FBrief.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nate/Libraries/DBoW2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/DBoW2.dir/src/FBrief.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DBoW2.dir/src/FBrief.cpp.o -c /home/nate/Libraries/DBoW2/src/FBrief.cpp

CMakeFiles/DBoW2.dir/src/FBrief.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/src/FBrief.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nate/Libraries/DBoW2/src/FBrief.cpp > CMakeFiles/DBoW2.dir/src/FBrief.cpp.i

CMakeFiles/DBoW2.dir/src/FBrief.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/src/FBrief.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nate/Libraries/DBoW2/src/FBrief.cpp -o CMakeFiles/DBoW2.dir/src/FBrief.cpp.s

CMakeFiles/DBoW2.dir/src/FBrief.cpp.o.requires:

.PHONY : CMakeFiles/DBoW2.dir/src/FBrief.cpp.o.requires

CMakeFiles/DBoW2.dir/src/FBrief.cpp.o.provides: CMakeFiles/DBoW2.dir/src/FBrief.cpp.o.requires
	$(MAKE) -f CMakeFiles/DBoW2.dir/build.make CMakeFiles/DBoW2.dir/src/FBrief.cpp.o.provides.build
.PHONY : CMakeFiles/DBoW2.dir/src/FBrief.cpp.o.provides

CMakeFiles/DBoW2.dir/src/FBrief.cpp.o.provides.build: CMakeFiles/DBoW2.dir/src/FBrief.cpp.o


CMakeFiles/DBoW2.dir/src/FORB.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/src/FORB.cpp.o: ../src/FORB.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nate/Libraries/DBoW2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/DBoW2.dir/src/FORB.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DBoW2.dir/src/FORB.cpp.o -c /home/nate/Libraries/DBoW2/src/FORB.cpp

CMakeFiles/DBoW2.dir/src/FORB.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/src/FORB.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nate/Libraries/DBoW2/src/FORB.cpp > CMakeFiles/DBoW2.dir/src/FORB.cpp.i

CMakeFiles/DBoW2.dir/src/FORB.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/src/FORB.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nate/Libraries/DBoW2/src/FORB.cpp -o CMakeFiles/DBoW2.dir/src/FORB.cpp.s

CMakeFiles/DBoW2.dir/src/FORB.cpp.o.requires:

.PHONY : CMakeFiles/DBoW2.dir/src/FORB.cpp.o.requires

CMakeFiles/DBoW2.dir/src/FORB.cpp.o.provides: CMakeFiles/DBoW2.dir/src/FORB.cpp.o.requires
	$(MAKE) -f CMakeFiles/DBoW2.dir/build.make CMakeFiles/DBoW2.dir/src/FORB.cpp.o.provides.build
.PHONY : CMakeFiles/DBoW2.dir/src/FORB.cpp.o.provides

CMakeFiles/DBoW2.dir/src/FORB.cpp.o.provides.build: CMakeFiles/DBoW2.dir/src/FORB.cpp.o


CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o: ../src/FeatureVector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nate/Libraries/DBoW2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o -c /home/nate/Libraries/DBoW2/src/FeatureVector.cpp

CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nate/Libraries/DBoW2/src/FeatureVector.cpp > CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.i

CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nate/Libraries/DBoW2/src/FeatureVector.cpp -o CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.s

CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o.requires:

.PHONY : CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o.requires

CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o.provides: CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o.requires
	$(MAKE) -f CMakeFiles/DBoW2.dir/build.make CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o.provides.build
.PHONY : CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o.provides

CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o.provides.build: CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o


CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o: ../src/QueryResults.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nate/Libraries/DBoW2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o -c /home/nate/Libraries/DBoW2/src/QueryResults.cpp

CMakeFiles/DBoW2.dir/src/QueryResults.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/src/QueryResults.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nate/Libraries/DBoW2/src/QueryResults.cpp > CMakeFiles/DBoW2.dir/src/QueryResults.cpp.i

CMakeFiles/DBoW2.dir/src/QueryResults.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/src/QueryResults.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nate/Libraries/DBoW2/src/QueryResults.cpp -o CMakeFiles/DBoW2.dir/src/QueryResults.cpp.s

CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o.requires:

.PHONY : CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o.requires

CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o.provides: CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o.requires
	$(MAKE) -f CMakeFiles/DBoW2.dir/build.make CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o.provides.build
.PHONY : CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o.provides

CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o.provides.build: CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o


CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o: ../src/ScoringObject.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nate/Libraries/DBoW2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o -c /home/nate/Libraries/DBoW2/src/ScoringObject.cpp

CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nate/Libraries/DBoW2/src/ScoringObject.cpp > CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.i

CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nate/Libraries/DBoW2/src/ScoringObject.cpp -o CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.s

CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o.requires:

.PHONY : CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o.requires

CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o.provides: CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o.requires
	$(MAKE) -f CMakeFiles/DBoW2.dir/build.make CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o.provides.build
.PHONY : CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o.provides

CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o.provides.build: CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o


# Object files for target DBoW2
DBoW2_OBJECTS = \
"CMakeFiles/DBoW2.dir/src/BowVector.cpp.o" \
"CMakeFiles/DBoW2.dir/src/FBrief.cpp.o" \
"CMakeFiles/DBoW2.dir/src/FORB.cpp.o" \
"CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o" \
"CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o" \
"CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o"

# External object files for target DBoW2
DBoW2_EXTERNAL_OBJECTS =

libDBoW2.so: CMakeFiles/DBoW2.dir/src/BowVector.cpp.o
libDBoW2.so: CMakeFiles/DBoW2.dir/src/FBrief.cpp.o
libDBoW2.so: CMakeFiles/DBoW2.dir/src/FORB.cpp.o
libDBoW2.so: CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o
libDBoW2.so: CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o
libDBoW2.so: CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o
libDBoW2.so: CMakeFiles/DBoW2.dir/build.make
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_stitching3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_superres3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_videostab3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_aruco3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_bgsegm3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_bioinspired3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_ccalib3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_cvv3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_datasets3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_dpm3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_face3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_fuzzy3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_hdf3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_line_descriptor3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_optflow3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_plot3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_reg3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_saliency3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_stereo3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_structured_light3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_surface_matching3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_text3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_ximgproc3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_xobjdetect3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_xphoto3.so.3.2.0
libDBoW2.so: dependencies/install/lib/libDLib.so
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_shape3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_video3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_viz3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_rgbd3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_calib3d3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_features2d3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_flann3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_objdetect3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_ml3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_highgui3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_photo3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_videoio3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.2.0
libDBoW2.so: /opt/ros/kinetic/lib/libopencv_core3.so.3.2.0
libDBoW2.so: CMakeFiles/DBoW2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nate/Libraries/DBoW2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX shared library libDBoW2.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DBoW2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DBoW2.dir/build: libDBoW2.so

.PHONY : CMakeFiles/DBoW2.dir/build

CMakeFiles/DBoW2.dir/requires: CMakeFiles/DBoW2.dir/src/BowVector.cpp.o.requires
CMakeFiles/DBoW2.dir/requires: CMakeFiles/DBoW2.dir/src/FBrief.cpp.o.requires
CMakeFiles/DBoW2.dir/requires: CMakeFiles/DBoW2.dir/src/FORB.cpp.o.requires
CMakeFiles/DBoW2.dir/requires: CMakeFiles/DBoW2.dir/src/FeatureVector.cpp.o.requires
CMakeFiles/DBoW2.dir/requires: CMakeFiles/DBoW2.dir/src/QueryResults.cpp.o.requires
CMakeFiles/DBoW2.dir/requires: CMakeFiles/DBoW2.dir/src/ScoringObject.cpp.o.requires

.PHONY : CMakeFiles/DBoW2.dir/requires

CMakeFiles/DBoW2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DBoW2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DBoW2.dir/clean

CMakeFiles/DBoW2.dir/depend:
	cd /home/nate/Libraries/DBoW2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nate/Libraries/DBoW2 /home/nate/Libraries/DBoW2 /home/nate/Libraries/DBoW2/build /home/nate/Libraries/DBoW2/build /home/nate/Libraries/DBoW2/build/CMakeFiles/DBoW2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DBoW2.dir/depend

