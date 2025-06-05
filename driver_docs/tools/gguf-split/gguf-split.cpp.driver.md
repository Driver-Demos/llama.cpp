# Purpose
This C++ source code file is designed to handle operations on GGUF (Generic Graphical User Format) files, specifically focusing on splitting and merging these files. The code provides a command-line interface for users to specify input and output GGUF files and choose between splitting a single GGUF file into multiple smaller files or merging multiple GGUF files into a single file. The primary technical components include the `split_params` structure, which holds the parameters for the operation, and the [`split_strategy`](#split_strategysplit_strategy) class, which implements the logic for splitting GGUF files based on the specified parameters. The code also includes functions for parsing command-line arguments, converting string representations of sizes to bytes, and handling file I/O operations.

The file is structured to be an executable program, as indicated by the presence of a [`main`](#main) function, which serves as the entry point. It imports several headers, including custom ones like "ggml.h", "gguf.h", and "llama.h", suggesting dependencies on external libraries for handling GGUF and related data structures. The code defines public APIs through command-line options, allowing users to specify operations such as `--split`, `--merge`, `--split-max-tensors`, and `--split-max-size`. These options enable users to control the behavior of the program, such as the number of tensors per split or the maximum size of each split. The program is designed to be robust, with error handling for invalid arguments and file operations, ensuring that users receive informative feedback in case of incorrect usage or failures.
# Imports and Dependencies

---
- `ggml.h`
- `gguf.h`
- `llama.h`
- `common.h`
- `algorithm`
- `cinttypes`
- `climits`
- `cstdio`
- `cstdlib`
- `stdexcept`
- `cstring`
- `fstream`
- `string`
- `vector`
- `windows.h`
- `io.h`


# Data Structures

---
### split\_operation<!-- {{#data_structure:split_operation}} -->
- **Type**: `enum`
- **Members**:
    - `OP_NONE`: Represents no operation.
    - `OP_SPLIT`: Represents a split operation.
    - `OP_MERGE`: Represents a merge operation.
- **Description**: The `split_operation` enum defines a set of constants representing different operations that can be performed on GGUF files, specifically none, split, and merge operations. It is used to specify the type of operation to be executed in the context of file manipulation, such as splitting a GGUF file into multiple parts or merging multiple GGUF files into one.


---
### split\_mode<!-- {{#data_structure:split_mode}} -->
- **Type**: `enum`
- **Members**:
    - `MODE_NONE`: Represents a state where no specific split mode is applied.
    - `MODE_TENSOR`: Indicates a split mode based on the number of tensors.
    - `MODE_SIZE`: Indicates a split mode based on the size of the data.
- **Description**: The `split_mode` enum defines different modes for splitting data, specifically for operations involving GGUF files. It provides three modes: `MODE_NONE` for no specific split, `MODE_TENSOR` for splitting based on the number of tensors, and `MODE_SIZE` for splitting based on the size of the data. This enum is used to configure how data should be divided during processing.


---
### split\_params<!-- {{#data_structure:split_params}} -->
- **Type**: `struct`
- **Members**:
    - `operation`: Specifies the type of operation to perform, either split or merge, with a default of OP_NONE.
    - `mode`: Determines the mode of splitting, either by tensor or size, with a default of MODE_NONE.
    - `n_bytes_split`: Indicates the maximum number of bytes per split when splitting by size, defaulting to 0.
    - `n_split_tensors`: Specifies the maximum number of tensors per split when splitting by tensor, defaulting to 128.
    - `input`: Holds the input file path as a string.
    - `output`: Holds the output file path as a string.
    - `no_tensor_first_split`: A boolean flag indicating whether to exclude tensors from the first split, defaulting to false.
    - `dry_run`: A boolean flag indicating whether to perform a dry run without writing files, defaulting to false.
- **Description**: The `split_params` struct is designed to encapsulate parameters for splitting or merging operations on GGUF files. It includes settings for the type of operation (split or merge), the mode of splitting (by tensor count or size), and specific limits for each mode. Additionally, it holds file paths for input and output, and flags for controlling the behavior of the operation, such as whether to perform a dry run or exclude tensors from the first split. This struct is essential for configuring and executing the desired file manipulation operations.


---
### split\_strategy<!-- {{#data_structure:split_strategy}} -->
- **Type**: `struct`
- **Members**:
    - `params`: Holds the parameters for the split operation.
    - `f_input`: Reference to the input file stream.
    - `ctx_gguf`: Pointer to the GGUF context for the input data.
    - `ctx_meta`: Pointer to the GGML context for metadata, initialized to NULL.
    - `n_tensors`: Stores the number of tensors in the GGUF context.
    - `ctx_outs`: Vector of pointers to GGUF contexts for each output split.
    - `read_buf`: Temporary buffer for reading tensor data.
- **Description**: The `split_strategy` struct is designed to manage the process of splitting tensor data from a GGUF file into multiple output files based on specified parameters. It holds references to the input file and contexts for both the input and output data, as well as a buffer for reading tensor data. The struct is responsible for determining when to split the data based on size or tensor count, managing the output contexts, and writing the split data to files. It also provides functionality to print information about the splits and clean up resources upon destruction.
- **Member Functions**:
    - [`split_strategy::split_strategy`](#split_strategysplit_strategy)
    - [`split_strategy::~split_strategy`](#split_strategysplit_strategy)
    - [`split_strategy::should_split`](#split_strategyshould_split)
    - [`split_strategy::print_info`](#split_strategyprint_info)
    - [`split_strategy::write`](#split_strategywrite)
    - [`split_strategy::copy_file_to_file`](#split_strategycopy_file_to_file)

**Methods**

---
#### split\_strategy::split\_strategy<!-- {{#callable:split_strategy::split_strategy}} -->
The `split_strategy` constructor initializes a strategy for splitting tensor data from an input file into multiple output files based on specified parameters.
- **Inputs**:
    - `params`: A `split_params` structure containing parameters for the split operation, such as mode, size, and number of tensors per split.
    - `f_input`: A reference to an `std::ifstream` object representing the input file stream from which tensor data is read.
    - `ctx_gguf`: A pointer to a `gguf_context` structure representing the context of the input GGUF file.
    - `ctx_meta`: A pointer to a `ggml_context` structure representing the metadata context for tensor operations.
- **Control Flow**:
    - Initialize the number of tensors from the input GGUF context.
    - Define a lambda function `new_ctx_out` to create new output contexts and handle errors if a split has zero tensors.
    - Initialize the first output context using `new_ctx_out` with `allow_no_tensors` set to false.
    - If `no_tensor_first_split` is true in `params`, create a new output context allowing no tensors.
    - Iterate over each tensor, calculate its padded size, and determine if a new split is needed using [`should_split`](#split_strategyshould_split).
    - Add each tensor to the current output context and update the current tensor size.
    - Push the last output context to the list of contexts.
    - Set the correct number of splits for each output context.
- **Output**: The constructor does not return a value but initializes the `split_strategy` object with a list of output contexts (`ctx_outs`) for each split.
- **Functions called**:
    - [`gguf_get_n_tensors`](../../ggml/src/gguf.cpp.driver.md#gguf_get_n_tensors)
    - [`gguf_init_empty`](../../ggml/src/gguf.cpp.driver.md#gguf_init_empty)
    - [`gguf_set_kv`](../../ggml/src/gguf.cpp.driver.md#gguf_set_kv)
    - [`gguf_set_val_u16`](../../ggml/src/gguf.cpp.driver.md#gguf_set_val_u16)
    - [`gguf_set_val_i32`](../../ggml/src/gguf.cpp.driver.md#gguf_set_val_i32)
    - [`ggml_get_tensor`](../../ggml/src/ggml.c.driver.md#ggml_get_tensor)
    - [`gguf_get_tensor_name`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_name)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`split_strategy::should_split`](#split_strategyshould_split)
    - [`gguf_add_tensor`](../../ggml/src/gguf.cpp.driver.md#gguf_add_tensor)
- **See also**: [`split_strategy`](#split_strategy)  (Data Structure)


---
#### split\_strategy::\~split\_strategy<!-- {{#callable:split_strategy::~split_strategy}} -->
The destructor `~split_strategy` releases all allocated `gguf_context` objects stored in the `ctx_outs` vector.
- **Inputs**: None
- **Control Flow**:
    - Iterates over each `gguf_context` pointer in the `ctx_outs` vector.
    - Calls [`gguf_free`](../../ggml/src/gguf.cpp.driver.md#gguf_free) on each `gguf_context` pointer to release its allocated resources.
- **Output**: The function does not return any value; it ensures that resources are properly freed when a `split_strategy` object is destroyed.
- **Functions called**:
    - [`gguf_free`](../../ggml/src/gguf.cpp.driver.md#gguf_free)
- **See also**: [`split_strategy`](#split_strategy)  (Data Structure)


---
#### split\_strategy::should\_split<!-- {{#callable:split_strategy::should_split}} -->
The `should_split` function determines whether a split should occur based on the current mode and parameters.
- **Inputs**:
    - `i_tensor`: The index of the current tensor being processed.
    - `next_size`: The size of the next tensor or the cumulative size of tensors, depending on the mode.
- **Control Flow**:
    - Check if the mode is `MODE_SIZE`; if true, return true if `next_size` exceeds `params.n_bytes_split`.
    - If the mode is `MODE_TENSOR`, return true if `i_tensor` is greater than 0, less than `n_tensors`, and is a multiple of `params.n_split_tensors`.
    - If neither condition is met, abort the program with an 'invalid mode' error.
- **Output**: Returns a boolean indicating whether a split should occur based on the specified conditions.
- **See also**: [`split_strategy`](#split_strategy)  (Data Structure)


---
#### split\_strategy::print\_info<!-- {{#callable:split_strategy::print_info}} -->
The `print_info` function outputs the number of splits and details about each split, including the number of tensors and total size in megabytes.
- **Inputs**: None
- **Control Flow**:
    - Prints the total number of splits using the size of `ctx_outs`.
    - Initializes a counter `i_split` to zero for tracking the current split index.
    - Iterates over each `ctx_out` in `ctx_outs`.
    - For each `ctx_out`, calculates the total size by adding metadata size and the size of all tensors.
    - Converts the total size from bytes to megabytes.
    - Prints the split index, number of tensors, and total size in megabytes for each split.
    - Increments the `i_split` counter after processing each split.
- **Output**: The function outputs formatted information to the standard output, detailing the number of splits, and for each split, the number of tensors and the total size in megabytes.
- **Functions called**:
    - [`gguf_get_meta_size`](../../ggml/src/gguf.cpp.driver.md#gguf_get_meta_size)
    - [`gguf_get_n_tensors`](../../ggml/src/gguf.cpp.driver.md#gguf_get_n_tensors)
    - [`ggml_get_tensor`](../../ggml/src/ggml.c.driver.md#ggml_get_tensor)
    - [`gguf_get_tensor_name`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_name)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`split_strategy`](#split_strategy)  (Data Structure)


---
#### split\_strategy::write<!-- {{#callable:split_strategy::write}} -->
The `write` function writes metadata and tensor data from multiple contexts to separate output files, handling each context as a separate file split.
- **Inputs**:
    - `None`: The function does not take any parameters directly; it operates on the member variables of the `split_strategy` class.
- **Control Flow**:
    - Initialize `i_split` to 0 and determine the number of splits `n_split` from the size of `ctx_outs`.
    - Iterate over each `ctx_out` in `ctx_outs`.
    - For each `ctx_out`, construct a file path using `llama_split_path`.
    - Open a binary output file stream `fout` for the constructed file path, setting it to fail on write errors.
    - Retrieve and write metadata from `ctx_out` to the output file.
    - Iterate over each tensor in `ctx_out` using [`gguf_get_n_tensors`](../../ggml/src/gguf.cpp.driver.md#gguf_get_n_tensors).
    - For each tensor, retrieve its name and metadata, and prepare a buffer for its data.
    - Calculate the offset for the tensor data in the input file using [`gguf_find_tensor`](../../ggml/src/gguf.cpp.driver.md#gguf_find_tensor), [`gguf_get_data_offset`](../../ggml/src/gguf.cpp.driver.md#gguf_get_data_offset), and [`gguf_get_tensor_offset`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_offset).
    - Copy the tensor data from the input file to the output file using [`copy_file_to_file`](#split_strategycopy_file_to_file).
    - Pad the output file with zeros to align the tensor data size using [`zeros`](#zeros).
    - Print a completion message and close the output file.
    - Increment `i_split` to process the next context.
- **Output**: The function writes the metadata and tensor data to separate output files, one for each context in `ctx_outs`, and prints a message indicating the completion of each file write.
- **Functions called**:
    - [`gguf_get_meta_data`](../../ggml/src/gguf.cpp.driver.md#gguf_get_meta_data)
    - [`gguf_get_n_tensors`](../../ggml/src/gguf.cpp.driver.md#gguf_get_n_tensors)
    - [`gguf_get_tensor_name`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_name)
    - [`ggml_get_tensor`](../../ggml/src/ggml.c.driver.md#ggml_get_tensor)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`gguf_find_tensor`](../../ggml/src/gguf.cpp.driver.md#gguf_find_tensor)
    - [`gguf_get_data_offset`](../../ggml/src/gguf.cpp.driver.md#gguf_get_data_offset)
    - [`gguf_get_tensor_offset`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_offset)
    - [`split_strategy::copy_file_to_file`](#split_strategycopy_file_to_file)
    - [`zeros`](#zeros)
- **See also**: [`split_strategy`](#split_strategy)  (Data Structure)


---
#### split\_strategy::copy\_file\_to\_file<!-- {{#callable:split_strategy::copy_file_to_file}} -->
The `copy_file_to_file` function copies a specified portion of data from an input file stream to an output file stream.
- **Inputs**:
    - `f_in`: A reference to an input file stream (`std::ifstream`) from which data will be read.
    - `f_out`: A reference to an output file stream (`std::ofstream`) to which data will be written.
    - `in_offset`: A `size_t` value representing the offset in the input file from where the data reading should start.
    - `len`: A `size_t` value indicating the number of bytes to be copied from the input file to the output file.
- **Control Flow**:
    - Check if the `read_buf` size is less than `len` and resize it if necessary.
    - Seek the input file stream `f_in` to the position specified by `in_offset`.
    - Read `len` bytes of data from `f_in` into the `read_buf`.
    - Write the data from `read_buf` to the output file stream `f_out`.
- **Output**: The function does not return a value; it performs file I/O operations to copy data from the input file to the output file.
- **See also**: [`split_strategy`](#split_strategy)  (Data Structure)



# Functions

---
### split\_print\_usage<!-- {{#callable:split_print_usage}} -->
The `split_print_usage` function displays the usage instructions for a command-line tool that performs operations on GGUF files.
- **Inputs**:
    - `executable`: A constant character pointer representing the name of the executable file, typically used to display the command in the usage message.
- **Control Flow**:
    - Initialize a `split_params` object with default values.
    - Print a blank line for formatting purposes.
    - Print the usage message, including the executable name and the expected input and output files.
    - Print a description of the tool's purpose, which is to apply a GGUF operation on input to output.
    - Print a list of available options, including help, version, split, merge, and various split-related parameters, with default values where applicable.
- **Output**: The function does not return any value; it outputs the usage instructions to the standard output.


---
### split\_str\_to\_n\_bytes<!-- {{#callable:split_str_to_n_bytes}} -->
The function `split_str_to_n_bytes` converts a string representing a size in megabytes or gigabytes into the equivalent number of bytes.
- **Inputs**:
    - `str`: A string representing a size, ending with 'M' for megabytes or 'G' for gigabytes.
- **Control Flow**:
    - Initialize `n_bytes` to 0 and declare an integer `n`.
    - Check if the last character of `str` is 'M'; if true, parse the number and calculate bytes as `n * 1000 * 1000`.
    - Else, check if the last character of `str` is 'G'; if true, parse the number and calculate bytes as `n * 1000 * 1000 * 1000`.
    - If neither 'M' nor 'G' is found, throw an `invalid_argument` exception indicating unsupported units.
    - If the parsed number `n` is less than or equal to 0, throw an `invalid_argument` exception indicating the size must be positive.
    - Return the calculated `n_bytes`.
- **Output**: Returns the size in bytes as a `size_t`.


---
### split\_params\_parse\_ex<!-- {{#callable:split_params_parse_ex}} -->
The `split_params_parse_ex` function parses command-line arguments to configure the `split_params` structure for a GGUF operation, handling various options and validating input.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
    - `params`: A reference to a `split_params` structure that will be populated based on the parsed command-line arguments.
- **Control Flow**:
    - Initialize variables for argument parsing, including a string for the current argument and a boolean for invalid parameters.
    - Iterate over command-line arguments starting from index 1, checking if they start with '--'.
    - For each argument, replace underscores with hyphens and check if it matches known options.
    - If the argument is '-h' or '--help', print usage information and exit.
    - If the argument is '--version', print version and build information and exit.
    - For '--dry-run' and '--no-tensor-first-split', set corresponding flags in `params`.
    - For '--merge' and '--split', set the operation in `params`, ensuring only one of these can be set.
    - For '--split-max-tensors' and '--split-max-size', set the mode and corresponding values in `params`, ensuring only one of these can be set.
    - If an argument is not recognized, throw an invalid_argument exception.
    - Set default values for operation and mode if they are not specified.
    - Check for invalid parameters and throw an exception if any are found.
    - Ensure exactly two non-option arguments remain for input and output files, otherwise throw an exception.
    - Assign the remaining arguments to `params.input` and `params.output`.
- **Output**: The function does not return a value but populates the `split_params` structure with parsed and validated command-line options.
- **Functions called**:
    - [`split_print_usage`](#split_print_usage)
    - [`split_str_to_n_bytes`](#split_str_to_n_bytes)


---
### split\_params\_parse<!-- {{#callable:split_params_parse}} -->
The `split_params_parse` function parses command-line arguments to populate a `split_params` structure, handling errors and displaying usage information if necessary.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
    - `params`: A reference to a `split_params` structure that will be populated with parsed values from the command-line arguments.
- **Control Flow**:
    - The function initializes a boolean `result` to true, indicating successful parsing by default.
    - It calls [`split_params_parse_ex`](#split_params_parse_ex) to perform the actual parsing of command-line arguments and populate the `params` structure.
    - If [`split_params_parse_ex`](#split_params_parse_ex) throws a `std::invalid_argument` exception, the function catches it, prints the error message to standard error, calls [`split_print_usage`](#split_print_usage) to display usage information, and exits the program with a failure status.
    - Finally, the function returns the `result` variable, which remains true unless an exception is thrown.
- **Output**: A boolean value indicating whether the parsing was successful (always true unless an exception is thrown).
- **Functions called**:
    - [`split_params_parse_ex`](#split_params_parse_ex)
    - [`split_print_usage`](#split_print_usage)


---
### zeros<!-- {{#callable:zeros}} -->
The `zeros` function writes a specified number of zero bytes to a given output file stream.
- **Inputs**:
    - `file`: An output file stream (`std::ofstream`) where the zero bytes will be written.
    - `n`: The number of zero bytes to write to the file.
- **Control Flow**:
    - Initialize a character variable `zero` with the value 0.
    - Iterate `n` times, where `n` is the number of zero bytes to write.
    - In each iteration, write the `zero` character to the file stream using `file.write(&zero, 1)`.
- **Output**: The function does not return any value; it performs an operation on the file stream by writing zero bytes.


---
### gguf\_split<!-- {{#callable:gguf_split}} -->
The `gguf_split` function splits a GGUF file into multiple parts based on specified parameters.
- **Inputs**:
    - `split_params`: A `split_params` struct containing parameters for the split operation, including input file path, output file path, and options like dry run or no tensor in the first split.
- **Control Flow**:
    - Initialize a `ggml_context` pointer `ctx_meta` to NULL and set up `gguf_init_params` with no allocation and context pointing to `ctx_meta`.
    - Open the input GGUF file specified in `split_params` in binary mode using an `ifstream`.
    - Check if the file is successfully opened; if not, print an error message and exit.
    - Initialize a GGUF context from the input file using [`gguf_init_from_file`](../../ggml/src/gguf.cpp.driver.md#gguf_init_from_file); if it fails, print an error message and exit.
    - Create a `split_strategy` object with the given `split_params`, input file stream, GGUF context, and metadata context.
    - Retrieve the number of splits from the strategy and print split information.
    - If not a dry run, call the `write` method of the strategy to perform the actual file splitting.
    - Free the GGUF context and close the input file stream.
    - Print a message indicating the number of splits written and the total number of tensors.
- **Output**: The function does not return a value but writes the split GGUF files to the specified output locations and prints status messages to `stderr`.
- **Functions called**:
    - [`gguf_init_from_file`](../../ggml/src/gguf.cpp.driver.md#gguf_init_from_file)
    - [`gguf_free`](../../ggml/src/gguf.cpp.driver.md#gguf_free)


---
### gguf\_merge<!-- {{#callable:gguf_merge}} -->
The `gguf_merge` function merges multiple GGUF files into a single output file, handling metadata and tensor data appropriately.
- **Inputs**:
    - `split_params`: A `split_params` struct containing parameters for the merge operation, including input and output file paths, and flags for dry run and other options.
- **Control Flow**:
    - Prints the function name and input/output file paths to stderr.
    - Checks if the output file already exists and exits if it does to prevent overwriting.
    - Initializes an empty GGUF context for the output file.
    - Sets up vectors to store metadata and GGUF contexts for each split file.
    - Copies the input file path to a buffer and initializes a prefix buffer.
    - Iterates over each split file to read metadata and tensor information.
    - For the first split, retrieves the number of splits and verifies the file naming convention.
    - Sets metadata for the output context from the first split and adjusts the split count.
    - Iterates over tensors in each split, adding them to the output context and updating the total tensor count.
    - If not a dry run, opens the output file and writes placeholder metadata.
    - Iterates over each split again to read and write tensor data to the output file, handling padding as necessary.
    - Frees allocated contexts and closes files.
    - Writes the updated metadata to the beginning of the output file if not a dry run.
    - Prints a completion message with the number of splits and total tensors merged.
- **Output**: The function does not return a value but writes the merged GGUF data to the specified output file.
- **Functions called**:
    - [`gguf_init_empty`](../../ggml/src/gguf.cpp.driver.md#gguf_init_empty)
    - [`gguf_init_from_file`](../../ggml/src/gguf.cpp.driver.md#gguf_init_from_file)
    - [`gguf_find_key`](../../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`gguf_free`](../../ggml/src/gguf.cpp.driver.md#gguf_free)
    - [`ggml_free`](../../ggml/src/ggml.c.driver.md#ggml_free)
    - [`gguf_get_val_u16`](../../ggml/src/gguf.cpp.driver.md#gguf_get_val_u16)
    - [`gguf_set_val_u16`](../../ggml/src/gguf.cpp.driver.md#gguf_set_val_u16)
    - [`gguf_set_kv`](../../ggml/src/gguf.cpp.driver.md#gguf_set_kv)
    - [`gguf_get_n_tensors`](../../ggml/src/gguf.cpp.driver.md#gguf_get_n_tensors)
    - [`gguf_get_tensor_name`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_name)
    - [`ggml_get_tensor`](../../ggml/src/ggml.c.driver.md#ggml_get_tensor)
    - [`gguf_add_tensor`](../../ggml/src/gguf.cpp.driver.md#gguf_add_tensor)
    - [`gguf_get_meta_size`](../../ggml/src/gguf.cpp.driver.md#gguf_get_meta_size)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`gguf_get_data_offset`](../../ggml/src/gguf.cpp.driver.md#gguf_get_data_offset)
    - [`gguf_get_tensor_offset`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_offset)
    - [`zeros`](#zeros)
    - [`gguf_get_meta_data`](../../ggml/src/gguf.cpp.driver.md#gguf_get_meta_data)


---
### main<!-- {{#callable:main}} -->
The `main` function parses command-line arguments to determine whether to split or merge GGUF files and executes the corresponding operation.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize a `split_params` structure to hold parsed parameters.
    - Call [`split_params_parse`](#split_params_parse) to parse command-line arguments and populate the `params` structure.
    - Use a switch statement to determine the operation specified in `params.operation`.
    - If `params.operation` is `OP_SPLIT`, call [`gguf_split`](#gguf_split) with `params`.
    - If `params.operation` is `OP_MERGE`, call [`gguf_merge`](#gguf_merge) with `params`.
    - If the operation is not recognized, call [`split_print_usage`](#split_print_usage) to display usage information and exit with an error.
- **Output**: Returns 0 to indicate successful execution.
- **Functions called**:
    - [`split_params_parse`](#split_params_parse)
    - [`gguf_split`](#gguf_split)
    - [`gguf_merge`](#gguf_merge)
    - [`split_print_usage`](#split_print_usage)


