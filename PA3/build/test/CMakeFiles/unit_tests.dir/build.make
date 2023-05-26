# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/course/hpc/users/2019010874/PA3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/course/hpc/users/2019010874/PA3/build

# Include any dependencies generated for this target.
include test/CMakeFiles/unit_tests.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/unit_tests.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/unit_tests.dir/flags.make

test/CMakeFiles/unit_tests.dir/main.cpp.o: test/CMakeFiles/unit_tests.dir/flags.make
test/CMakeFiles/unit_tests.dir/main.cpp.o: ../test/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/course/hpc/users/2019010874/PA3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/unit_tests.dir/main.cpp.o"
	cd /home/course/hpc/users/2019010874/PA3/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/unit_tests.dir/main.cpp.o -c /home/course/hpc/users/2019010874/PA3/test/main.cpp

test/CMakeFiles/unit_tests.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/unit_tests.dir/main.cpp.i"
	cd /home/course/hpc/users/2019010874/PA3/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/course/hpc/users/2019010874/PA3/test/main.cpp > CMakeFiles/unit_tests.dir/main.cpp.i

test/CMakeFiles/unit_tests.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/unit_tests.dir/main.cpp.s"
	cd /home/course/hpc/users/2019010874/PA3/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/course/hpc/users/2019010874/PA3/test/main.cpp -o CMakeFiles/unit_tests.dir/main.cpp.s

test/CMakeFiles/unit_tests.dir/test_spmm.cu.o: test/CMakeFiles/unit_tests.dir/flags.make
test/CMakeFiles/unit_tests.dir/test_spmm.cu.o: ../test/test_spmm.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/course/hpc/users/2019010874/PA3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object test/CMakeFiles/unit_tests.dir/test_spmm.cu.o"
	cd /home/course/hpc/users/2019010874/PA3/build/test && /usr/local/cuda-11.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/course/hpc/users/2019010874/PA3/test/test_spmm.cu -o CMakeFiles/unit_tests.dir/test_spmm.cu.o

test/CMakeFiles/unit_tests.dir/test_spmm.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/unit_tests.dir/test_spmm.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test/CMakeFiles/unit_tests.dir/test_spmm.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/unit_tests.dir/test_spmm.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

test/CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.o: test/CMakeFiles/unit_tests.dir/flags.make
test/CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.o: ../test/test_spmm_cpu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/course/hpc/users/2019010874/PA3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object test/CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.o"
	cd /home/course/hpc/users/2019010874/PA3/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.o -c /home/course/hpc/users/2019010874/PA3/test/test_spmm_cpu.cpp

test/CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.i"
	cd /home/course/hpc/users/2019010874/PA3/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/course/hpc/users/2019010874/PA3/test/test_spmm_cpu.cpp > CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.i

test/CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.s"
	cd /home/course/hpc/users/2019010874/PA3/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/course/hpc/users/2019010874/PA3/test/test_spmm_cpu.cpp -o CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.s

# Object files for target unit_tests
unit_tests_OBJECTS = \
"CMakeFiles/unit_tests.dir/main.cpp.o" \
"CMakeFiles/unit_tests.dir/test_spmm.cu.o" \
"CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.o"

# External object files for target unit_tests
unit_tests_EXTERNAL_OBJECTS =

test/CMakeFiles/unit_tests.dir/cmake_device_link.o: test/CMakeFiles/unit_tests.dir/main.cpp.o
test/CMakeFiles/unit_tests.dir/cmake_device_link.o: test/CMakeFiles/unit_tests.dir/test_spmm.cu.o
test/CMakeFiles/unit_tests.dir/cmake_device_link.o: test/CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.o
test/CMakeFiles/unit_tests.dir/cmake_device_link.o: test/CMakeFiles/unit_tests.dir/build.make
test/CMakeFiles/unit_tests.dir/cmake_device_link.o: /usr/local/cuda-11.1/lib64/libcudart_static.a
test/CMakeFiles/unit_tests.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/librt.so
test/CMakeFiles/unit_tests.dir/cmake_device_link.o: lib/libgtest_main.a
test/CMakeFiles/unit_tests.dir/cmake_device_link.o: libexpspmm.a
test/CMakeFiles/unit_tests.dir/cmake_device_link.o: lib/libgtest.a
test/CMakeFiles/unit_tests.dir/cmake_device_link.o: test/CMakeFiles/unit_tests.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/course/hpc/users/2019010874/PA3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CUDA device code CMakeFiles/unit_tests.dir/cmake_device_link.o"
	cd /home/course/hpc/users/2019010874/PA3/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/unit_tests.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/unit_tests.dir/build: test/CMakeFiles/unit_tests.dir/cmake_device_link.o

.PHONY : test/CMakeFiles/unit_tests.dir/build

# Object files for target unit_tests
unit_tests_OBJECTS = \
"CMakeFiles/unit_tests.dir/main.cpp.o" \
"CMakeFiles/unit_tests.dir/test_spmm.cu.o" \
"CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.o"

# External object files for target unit_tests
unit_tests_EXTERNAL_OBJECTS =

test/unit_tests: test/CMakeFiles/unit_tests.dir/main.cpp.o
test/unit_tests: test/CMakeFiles/unit_tests.dir/test_spmm.cu.o
test/unit_tests: test/CMakeFiles/unit_tests.dir/test_spmm_cpu.cpp.o
test/unit_tests: test/CMakeFiles/unit_tests.dir/build.make
test/unit_tests: /usr/local/cuda-11.1/lib64/libcudart_static.a
test/unit_tests: /usr/lib/x86_64-linux-gnu/librt.so
test/unit_tests: lib/libgtest_main.a
test/unit_tests: libexpspmm.a
test/unit_tests: lib/libgtest.a
test/unit_tests: test/CMakeFiles/unit_tests.dir/cmake_device_link.o
test/unit_tests: test/CMakeFiles/unit_tests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/course/hpc/users/2019010874/PA3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable unit_tests"
	cd /home/course/hpc/users/2019010874/PA3/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/unit_tests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/unit_tests.dir/build: test/unit_tests

.PHONY : test/CMakeFiles/unit_tests.dir/build

test/CMakeFiles/unit_tests.dir/clean:
	cd /home/course/hpc/users/2019010874/PA3/build/test && $(CMAKE_COMMAND) -P CMakeFiles/unit_tests.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/unit_tests.dir/clean

test/CMakeFiles/unit_tests.dir/depend:
	cd /home/course/hpc/users/2019010874/PA3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/course/hpc/users/2019010874/PA3 /home/course/hpc/users/2019010874/PA3/test /home/course/hpc/users/2019010874/PA3/build /home/course/hpc/users/2019010874/PA3/build/test /home/course/hpc/users/2019010874/PA3/build/test/CMakeFiles/unit_tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/unit_tests.dir/depend

