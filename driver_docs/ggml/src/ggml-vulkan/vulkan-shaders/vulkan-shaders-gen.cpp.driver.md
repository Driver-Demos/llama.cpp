# Purpose
This C++ source code file is designed to automate the process of compiling Vulkan shader source files into SPIR-V binary format and generating corresponding C++ header and source files. The code is structured to handle both Windows and Unix-like operating systems, using conditional compilation to manage platform-specific operations such as process creation and file handling. The main functionality revolves around the [`process_shaders`](#(anonymous)::process_shaders) function, which orchestrates the compilation of various shader types by invoking the `glslc` compiler with appropriate options and definitions. The code supports a wide range of shader types, including different floating-point and quantized formats, and it manages concurrency using C++ standard library features like `std::mutex`, `std::condition_variable`, and `std::future` to handle multiple shader compilations simultaneously.

The file defines a command-line interface that allows users to specify paths for the `glslc` compiler, input and output directories, and target files for the generated C++ code. It also includes utility functions for string manipulation, path handling, and directory management. The [`write_output_files`](#(anonymous)::write_output_files) function is responsible for creating the C++ header and source files that contain the compiled shader binaries as byte arrays, which can be directly included in other C++ projects. This code is a comprehensive solution for shader compilation and integration into C++ applications, providing a streamlined workflow for developers working with Vulkan shaders.
# Imports and Dependencies

---
- `iostream`
- `fstream`
- `sstream`
- `string`
- `stdexcept`
- `array`
- `vector`
- `map`
- `thread`
- `mutex`
- `future`
- `queue`
- `condition_variable`
- `cstdio`
- `cstring`
- `cstdlib`
- `cassert`
- `algorithm`
- `sys/stat.h`
- `sys/types.h`
- `windows.h`
- `direct.h`
- `unistd.h`
- `sys/wait.h`
- `fcntl.h`


# Global Variables

---
### lock
- **Type**: `std::mutex`
- **Description**: The `lock` variable is a global instance of `std::mutex`, which is a synchronization primitive used to protect shared data from being simultaneously accessed by multiple threads. In this code, it is used to ensure that the `shader_fnames` vector is accessed in a thread-safe manner.
- **Use**: The `lock` variable is used to synchronize access to shared resources, specifically the `shader_fnames` vector, to prevent data races in a multithreaded environment.


---
### shader\_fnames
- **Type**: `std::vector<std::pair<std::string, std::string>>`
- **Description**: The `shader_fnames` variable is a global vector that stores pairs of strings, where each pair consists of a shader name and its corresponding output file name. This vector is used to keep track of the shaders that have been processed and their associated output files.
- **Use**: This variable is used to store and manage the names and output file paths of shaders that are compiled during the execution of the program.


---
### GLSLC
- **Type**: `std::string`
- **Description**: The `GLSLC` variable is a global string variable initialized with the value "glslc". It represents the default command or path to the GLSL compiler used in the program.
- **Use**: This variable is used to specify the GLSL compiler command when compiling shader files to SPIR-V format.


---
### input\_dir
- **Type**: `std::string`
- **Description**: The `input_dir` variable is a global string variable initialized to the value "vulkan-shaders". It represents the directory path where the shader source files are located.
- **Use**: This variable is used to specify the input directory for shader source files, which are processed and compiled into SPIR-V format.


---
### output\_dir
- **Type**: `std::string`
- **Description**: The `output_dir` variable is a global string variable that specifies the directory path where the SPIR-V output files will be stored. It is initialized with the default value of "/tmp".
- **Use**: This variable is used to determine the directory path for storing the compiled SPIR-V shader files.


---
### target\_hpp
- **Type**: `std::string`
- **Description**: The `target_hpp` variable is a global string variable that holds the name of the header file to be generated, which is `ggml-vulkan-shaders.hpp`. This file is intended to store declarations related to Vulkan shader data.
- **Use**: This variable is used to specify the output path for the generated header file containing shader data declarations.


---
### target\_cpp
- **Type**: `std::string`
- **Description**: The `target_cpp` variable is a global string variable that holds the name of a C++ source file, specifically "ggml-vulkan-shaders.cpp". This file is likely used as a target for output or processing within the program.
- **Use**: This variable is used to specify the path or name of the generated C++ source file that will be written to during the execution of the program.


---
### no\_clean
- **Type**: `bool`
- **Description**: The `no_clean` variable is a global boolean flag initialized to `false`. It is used to determine whether temporary SPIR-V files should be deleted after the build process.
- **Use**: This variable is used to control the cleanup of temporary files, where setting it to `true` prevents the deletion of these files.


---
### type\_names
- **Type**: ``const std::vector<std::string>``
- **Description**: The `type_names` variable is a constant vector of strings that contains a list of type names used in the shader compilation process. These type names represent different data types or formats that are likely used in the context of shader programming, such as floating-point types (`f32`, `f16`), quantized types (`q4_0`, `q4_1`, etc.), and others like `bf16`. The vector is initialized with a predefined set of type names.
- **Use**: This variable is used to iterate over different type names when generating and compiling shaders to SPIR-V, allowing the program to handle various data types in shader operations.


# Functions

---
### execute\_command<!-- {{#callable:(anonymous)::execute_command}} -->
The `execute_command` function executes a shell command and captures its standard output and standard error streams into provided string references.
- **Inputs**:
    - `command`: A constant reference to a string representing the shell command to be executed.
    - `stdout_str`: A reference to a string where the standard output of the command will be stored.
    - `stderr_str`: A reference to a string where the standard error of the command will be stored.
- **Control Flow**:
    - The function checks if the platform is Windows or POSIX-compliant (e.g., Linux, macOS) using preprocessor directives.
    - For Windows, it creates pipes for stdout and stderr, sets up security attributes, and uses `CreateProcessA` to execute the command, capturing output via the pipes.
    - For POSIX systems, it creates pipes, forks the process, and uses `execl` to execute the command in the child process, redirecting stdout and stderr to the pipes.
    - In both cases, the function reads from the pipes in a loop to capture the command's output and error streams into the provided strings.
    - The function waits for the process to complete and closes all handles or file descriptors.
- **Output**: The function does not return a value but modifies the `stdout_str` and `stderr_str` arguments to contain the command's output and error messages, respectively.


---
### directory\_exists<!-- {{#callable:(anonymous)::directory_exists}} -->
The `directory_exists` function checks if a given path corresponds to an existing directory.
- **Inputs**:
    - `path`: A constant reference to a `std::string` representing the path to be checked.
- **Control Flow**:
    - Declare a `struct stat` variable named `info` to hold information about the file system object at the given path.
    - Use the `stat` function to attempt to retrieve information about the path, passing the C-style string of `path` and the address of `info`.
    - If `stat` returns a non-zero value, indicating an error or that the path does not exist, return `false`.
    - Check if the `st_mode` field of `info` has the `S_IFDIR` bit set, which indicates the path is a directory.
    - Return `true` if the path is a directory, otherwise return `false`.
- **Output**: A boolean value: `true` if the path exists and is a directory, `false` otherwise.


---
### create\_directory<!-- {{#callable:(anonymous)::create_directory}} -->
The `create_directory` function attempts to create a directory at the specified path and returns true if successful or if the directory already exists.
- **Inputs**:
    - `path`: A constant reference to a string representing the path where the directory should be created.
- **Control Flow**:
    - The function checks if the code is being compiled on a Windows platform using the `_WIN32` preprocessor directive.
    - If on Windows, it uses the `_mkdir` function to attempt to create the directory and checks if the return value is 0 (success) or if `errno` is `EEXIST` (directory already exists).
    - If not on Windows, it uses the `mkdir` function with permissions set to `0755` to attempt to create the directory and checks if the return value is 0 (success) or if `errno` is `EEXIST` (directory already exists).
- **Output**: A boolean value indicating whether the directory was successfully created or already exists.


---
### to\_uppercase<!-- {{#callable:(anonymous)::to_uppercase}} -->
The `to_uppercase` function converts all characters in a given string to uppercase.
- **Inputs**:
    - `input`: A constant reference to a `std::string` that represents the input string to be converted to uppercase.
- **Control Flow**:
    - Initialize a new string `result` with the value of the input string `input`.
    - Iterate over each character `c` in the string `result`.
    - Convert each character `c` to its uppercase equivalent using `std::toupper` and assign it back to `c`.
    - Return the modified string `result`.
- **Output**: A `std::string` where all characters from the input string have been converted to uppercase.


---
### string\_starts\_with<!-- {{#callable:(anonymous)::string_starts_with}} -->
The `string_starts_with` function checks if a given string starts with a specified prefix.
- **Inputs**:
    - `str`: The main string to be checked.
    - `prefix`: The prefix string to check against the start of the main string.
- **Control Flow**:
    - Check if the size of the prefix is greater than the size of the main string; if so, return false.
    - Use the `std::equal` function to compare the prefix with the beginning of the main string and return the result.
- **Output**: A boolean value indicating whether the main string starts with the specified prefix.


---
### string\_ends\_with<!-- {{#callable:(anonymous)::string_ends_with}} -->
The `string_ends_with` function checks if a given string ends with a specified suffix.
- **Inputs**:
    - `str`: The main string to be checked.
    - `suffix`: The suffix to check for at the end of the main string.
- **Control Flow**:
    - Check if the length of the suffix is greater than the length of the main string; if so, return false.
    - Use the `std::equal` function to compare the suffix with the end of the main string in reverse order, and return the result.
- **Output**: Returns a boolean value: true if the main string ends with the specified suffix, false otherwise.


---
### join\_paths<!-- {{#callable:(anonymous)::join_paths}} -->
The `join_paths` function concatenates two file paths with a platform-specific path separator.
- **Inputs**:
    - `path1`: The first part of the file path to be joined.
    - `path2`: The second part of the file path to be joined.
- **Control Flow**:
    - Concatenate `path1` and `path2` with the `path_separator` character in between.
- **Output**: A single string representing the combined file path with the appropriate path separator.


---
### basename<!-- {{#callable:(anonymous)::basename}} -->
The `basename` function extracts the file name from a given file path by removing the directory components.
- **Inputs**:
    - `path`: A string representing the file path from which the base name (file name) is to be extracted.
- **Control Flow**:
    - Finds the last occurrence of either '/' or '\' in the input path string using `find_last_of`.
    - Extracts the substring from the character after the last '/' or '\' to the end of the string using `substr`.
- **Output**: Returns a string containing the base name (file name) extracted from the input path.


---
### string\_to\_spv\_func<!-- {{#callable:(anonymous)::string_to_spv_func}} -->
The `string_to_spv_func` function compiles a GLSL shader file into SPIR-V format using specified options and stores the result in a designated output directory.
- **Inputs**:
    - `_name`: A string representing the base name for the output SPIR-V file.
    - `in_fname`: A string representing the input filename of the GLSL shader to be compiled.
    - `defines`: A map of preprocessor definitions to be passed to the GLSL compiler.
    - `fp16`: A boolean flag indicating whether to use 16-bit floating point precision (default is true).
    - `coopmat`: A boolean flag indicating whether cooperative matrix operations are enabled (default is false).
    - `coopmat2`: A boolean flag indicating whether a second type of cooperative matrix operations are enabled (default is false).
    - `f16acc`: A boolean flag indicating whether to use 16-bit accumulation (default is false).
- **Control Flow**:
    - Constructs the output filename by appending specific suffixes based on the boolean flags to the base name.
    - Determines the target Vulkan environment version based on the presence of '_cm2' in the name.
    - Sets the optimization level for the GLSL compiler, disabling it for cooperative matrix shaders.
    - Builds the command to execute the GLSL compiler with the appropriate flags and input/output paths.
    - Appends debug information flag if `GGML_VULKAN_SHADER_DEBUG_INFO` is defined.
    - Adds preprocessor definitions from the `defines` map to the command.
    - Executes the constructed command using [`execute_command`](#(anonymous)::execute_command), capturing standard output and error.
    - If there is an error in the standard error output, logs the error and returns early.
    - On successful compilation, locks a mutex and appends the shader name and output filename to a global vector.
    - Decrements a global compile count and notifies all waiting threads.
- **Output**: The function does not return a value but modifies global state by adding the compiled shader's name and output path to a global vector and managing a compile count.
- **Functions called**:
    - [`(anonymous)::join_paths`](#(anonymous)::join_paths)
    - [`(anonymous)::execute_command`](#(anonymous)::execute_command)


---
### merge\_maps<!-- {{#callable:(anonymous)::merge_maps}} -->
The `merge_maps` function combines two maps into a single map, with the second map's entries potentially overwriting those of the first if keys overlap.
- **Inputs**:
    - `a`: A constant reference to the first map of type `std::map<std::string, std::string>` to be merged.
    - `b`: A constant reference to the second map of type `std::map<std::string, std::string>` to be merged.
- **Control Flow**:
    - Initialize a new map `result` with the contents of map `a`.
    - Insert all elements from map `b` into `result`, potentially overwriting existing entries with the same keys.
    - Return the `result` map.
- **Output**: A `std::map<std::string, std::string>` containing all key-value pairs from both input maps, with `b`'s values overwriting `a`'s in case of key collisions.


---
### string\_to\_spv<!-- {{#callable:(anonymous)::string_to_spv}} -->
The `string_to_spv` function manages the asynchronous compilation of shader source files into SPIR-V format, ensuring that no more than 16 compilations are in progress simultaneously.
- **Inputs**:
    - `_name`: A string representing the name of the shader.
    - `in_fname`: A string representing the input filename of the shader source.
    - `defines`: A map of string key-value pairs representing preprocessor definitions to be used during compilation.
    - `fp16`: A boolean flag indicating whether to use 16-bit floating point precision (default is true).
    - `coopmat`: A boolean flag indicating whether cooperative matrix operations are enabled (default is false).
    - `coopmat2`: A boolean flag indicating whether a second type of cooperative matrix operations is enabled (default is false).
    - `f16acc`: A boolean flag indicating whether to use 16-bit floating point accumulation (default is false).
- **Control Flow**:
    - Acquire a unique lock on the `compile_count_mutex` to ensure thread safety when accessing `compile_count`.
    - Wait until the number of ongoing compilations (`compile_count`) is less than 16, using a condition variable to block the thread if necessary.
    - Increment the `compile_count` to indicate a new compilation is starting.
    - Launch an asynchronous task to compile the shader using `string_to_spv_func` with the provided parameters.
    - Add the future object returned by `std::async` to the `compiles` vector to keep track of the ongoing compilation.
- **Output**: The function does not return any value; it initiates an asynchronous task for shader compilation and manages the concurrency of such tasks.


---
### matmul\_shaders<!-- {{#callable:(anonymous)::matmul_shaders}} -->
The `matmul_shaders` function generates and compiles various shader configurations for matrix multiplication using different data types and optimization settings.
- **Inputs**:
    - `fp16`: A boolean indicating whether to use 16-bit floating point precision.
    - `matmul_id`: A boolean indicating whether to use a specific matrix multiplication identifier.
    - `coopmat`: A boolean indicating whether to use cooperative matrix operations.
    - `coopmat2`: A boolean indicating whether to use a second type of cooperative matrix operations.
    - `f16acc`: A boolean indicating whether to use 16-bit floating point accumulation.
- **Control Flow**:
    - Initialize variables `load_vec`, `aligned_b_type_f32`, and `aligned_b_type_f16` based on the input flags `coopmat2` and `fp16`.
    - Create a base dictionary `base_dict` with initial shader configuration settings.
    - Set `shader_name` to 'matmul' and modify it to 'matmul_id' if `matmul_id` is true, updating `base_dict` accordingly.
    - Add 'FLOAT16' to `base_dict` if `fp16` is true and set 'ACC_TYPE' based on `f16acc`.
    - Add 'COOPMAT' to `base_dict` if `coopmat` is true.
    - Determine `source_name` based on `coopmat2`.
    - Define a lambda `FLOAT_TYPE` to determine the float type based on input type and cooperative matrix settings.
    - Generate and compile shaders for different configurations using [`string_to_spv`](#(anonymous)::string_to_spv) function, passing different combinations of shader names, source names, and merged dictionaries.
    - Handle special cases for 'bf16' type, adjusting load vectors and conversion functions, and compile shaders accordingly.
    - Iterate over `type_names` to generate and compile shaders for each type, adjusting load vectors and shader configurations based on type and cooperative matrix settings.
    - Use preprocessor directives to conditionally compile certain shader configurations based on defined support macros.
- **Output**: The function does not return a value; it generates and compiles shader files, storing them in a specified output directory.
- **Functions called**:
    - [`(anonymous)::string_to_spv`](#(anonymous)::string_to_spv)
    - [`(anonymous)::merge_maps`](#(anonymous)::merge_maps)
    - [`(anonymous)::to_uppercase`](#(anonymous)::to_uppercase)


---
### process\_shaders<!-- {{#callable:(anonymous)::process_shaders}} -->
The `process_shaders` function generates and compiles various shader programs into SPIR-V format for Vulkan using different configurations and types.
- **Inputs**: None
- **Control Flow**:
    - Prints a message indicating the start of shader generation and compilation.
    - Initializes a base dictionary with a default float type.
    - Iterates over two boolean values for `matmul_id` to generate matrix multiplication shaders with different configurations, including support for cooperative matrices if defined.
    - Iterates over two boolean values for `f16acc` to generate flash attention shaders for various data types, with conditional support for cooperative matrices.
    - Iterates over a list of type names to generate shaders for matrix-vector multiplication, dequantization, and row retrieval, with special handling for certain types.
    - Generates additional shaders for operations like norms, copy, arithmetic operations, and others, using the [`string_to_spv`](#(anonymous)::string_to_spv) function to compile them.
    - Waits for all asynchronous shader compilation tasks to complete.
- **Output**: The function does not return a value but generates compiled SPIR-V shader files and stores them in a specified output directory.
- **Functions called**:
    - [`(anonymous)::matmul_shaders`](#(anonymous)::matmul_shaders)
    - [`(anonymous)::string_to_spv`](#(anonymous)::string_to_spv)
    - [`(anonymous)::merge_maps`](#(anonymous)::merge_maps)
    - [`(anonymous)::to_uppercase`](#(anonymous)::to_uppercase)
    - [`(anonymous)::string_ends_with`](#(anonymous)::string_ends_with)
    - [`(anonymous)::string_starts_with`](#(anonymous)::string_starts_with)


---
### write\_output\_files<!-- {{#callable:(anonymous)::write_output_files}} -->
The `write_output_files` function generates C++ header and source files containing SPIR-V shader data and metadata from compiled shader files.
- **Inputs**: None
- **Control Flow**:
    - Open the target header and source files for writing.
    - Include necessary headers in the target files.
    - Sort the list of shader filenames.
    - Iterate over each shader file, opening it and reading its binary data.
    - Write the binary data and its size as C++ arrays and constants in the header and source files.
    - If `no_clean` is false, delete the temporary shader files after processing.
    - Iterate over a set of operations ('add', 'sub', 'mul', 'div') and write external declarations and definitions for data and length arrays in the header and source files.
    - Close the header and source files.
- **Output**: The function outputs two files: a header file (`target_hpp`) and a source file (`target_cpp`) containing C++ declarations and definitions for shader data arrays and their lengths.
- **Functions called**:
    - [`(anonymous)::basename`](#(anonymous)::basename)


---
### main<!-- {{#callable:main}} -->
The `main` function processes command-line arguments to configure shader compilation and output directories, verifies directory existence, and orchestrates the shader processing and output file writing.
- **Inputs**:
    - `argc`: An integer representing the number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize a map `args` to store command-line arguments and their values.
    - Iterate over `argv` starting from index 1 to parse arguments prefixed with '--' and store them in `args`.
    - Check for specific arguments (`--glslc`, `--input-dir`, `--output-dir`, `--target-hpp`, `--target-cpp`, `--no-clean`) and update corresponding global variables.
    - Verify if `input_dir` exists; if not, print an error message and return `EXIT_FAILURE`.
    - Check if `output_dir` exists; if not, attempt to create it and return `EXIT_FAILURE` if creation fails.
    - Call `process_shaders()` to generate and compile shaders.
    - Call `write_output_files()` to write the compiled shader data to output files.
    - Return `EXIT_SUCCESS` to indicate successful execution.
- **Output**: Returns an integer status code: `EXIT_SUCCESS` on successful execution or `EXIT_FAILURE` if an error occurs.
- **Functions called**:
    - [`(anonymous)::directory_exists`](#(anonymous)::directory_exists)
    - [`(anonymous)::create_directory`](#(anonymous)::create_directory)
    - [`(anonymous)::process_shaders`](#(anonymous)::process_shaders)
    - [`(anonymous)::write_output_files`](#(anonymous)::write_output_files)


