# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.23.3/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.23.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/aayush/desktop/c-deeplearning

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/aayush/desktop/c-deeplearning

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/main.c.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/main.c.o: main.c
CMakeFiles/main.dir/main.c.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aayush/desktop/c-deeplearning/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/main.dir/main.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/main.dir/main.c.o -MF CMakeFiles/main.dir/main.c.o.d -o CMakeFiles/main.dir/main.c.o -c /Users/aayush/desktop/c-deeplearning/main.c

CMakeFiles/main.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/main.dir/main.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/aayush/desktop/c-deeplearning/main.c > CMakeFiles/main.dir/main.c.i

CMakeFiles/main.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/main.dir/main.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/aayush/desktop/c-deeplearning/main.c -o CMakeFiles/main.dir/main.c.s

CMakeFiles/main.dir/matrix.c.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/matrix.c.o: matrix.c
CMakeFiles/main.dir/matrix.c.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aayush/desktop/c-deeplearning/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/main.dir/matrix.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/main.dir/matrix.c.o -MF CMakeFiles/main.dir/matrix.c.o.d -o CMakeFiles/main.dir/matrix.c.o -c /Users/aayush/desktop/c-deeplearning/matrix.c

CMakeFiles/main.dir/matrix.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/main.dir/matrix.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/aayush/desktop/c-deeplearning/matrix.c > CMakeFiles/main.dir/matrix.c.i

CMakeFiles/main.dir/matrix.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/main.dir/matrix.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/aayush/desktop/c-deeplearning/matrix.c -o CMakeFiles/main.dir/matrix.c.s

CMakeFiles/main.dir/neural_network.c.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/neural_network.c.o: neural_network.c
CMakeFiles/main.dir/neural_network.c.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/aayush/desktop/c-deeplearning/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/main.dir/neural_network.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/main.dir/neural_network.c.o -MF CMakeFiles/main.dir/neural_network.c.o.d -o CMakeFiles/main.dir/neural_network.c.o -c /Users/aayush/desktop/c-deeplearning/neural_network.c

CMakeFiles/main.dir/neural_network.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/main.dir/neural_network.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/aayush/desktop/c-deeplearning/neural_network.c > CMakeFiles/main.dir/neural_network.c.i

CMakeFiles/main.dir/neural_network.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/main.dir/neural_network.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/aayush/desktop/c-deeplearning/neural_network.c -o CMakeFiles/main.dir/neural_network.c.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.c.o" \
"CMakeFiles/main.dir/matrix.c.o" \
"CMakeFiles/main.dir/neural_network.c.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/main.c.o
main: CMakeFiles/main.dir/matrix.c.o
main: CMakeFiles/main.dir/neural_network.c.o
main: CMakeFiles/main.dir/build.make
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/aayush/desktop/c-deeplearning/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /Users/aayush/desktop/c-deeplearning && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/aayush/desktop/c-deeplearning /Users/aayush/desktop/c-deeplearning /Users/aayush/desktop/c-deeplearning /Users/aayush/desktop/c-deeplearning /Users/aayush/desktop/c-deeplearning/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

