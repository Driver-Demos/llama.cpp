# Purpose
This C++ source code file is designed to perform hashing operations on GGUF files, which are likely a specific type of data file used in a particular application context. The code provides functionality to generate and verify hashes using multiple algorithms, including SHA-1, SHA-256, XXH64, and UUIDv5. It includes mechanisms to parse command-line arguments, allowing users to specify which hashing algorithms to use, whether to exclude per-layer hashing, and whether to verify hashes against a provided manifest file. The code is structured to handle both hash generation and verification, with detailed error handling and user feedback through console output.

The file includes several key components: it defines enumerations for exit codes and manifest verification results, structures for holding hashing parameters, and functions for parsing command-line arguments and performing hashing operations. The main function orchestrates the process by parsing input arguments, determining the appropriate hashing operations, and executing the hashing logic. The code also integrates external libraries for hashing algorithms, as indicated by the inclusion of headers like "xxhash.h", "sha1.h", and "sha256.h". This file is intended to be compiled into an executable, as evidenced by the presence of a [`main`](#main) function, and it provides a public API for command-line interaction, allowing users to specify options and input files directly.
# Imports and Dependencies

---
- `ggml.h`
- `gguf.h`
- `cstdlib`
- `cstddef`
- `cstdio`
- `string`
- `stdexcept`
- `algorithm`
- `cstring`
- `sstream`
- `fstream`
- `xxhash/xxhash.h`
- `sha1/sha1.h`
- `sha256/sha256.h`


# Data Structures

---
### hash\_exit\_code\_t<!-- {{#data_structure:hash_exit_code_t}} -->
- **Type**: `enum`
- **Members**:
    - `HASH_EXIT_SUCCESS`: Indicates that all hashes have been successfully generated or validated.
    - `HASH_EXIT_FAILURE`: Represents a generic failure during hash operations.
    - `HASH_EXIT_MISMATCH`: Indicates a hash mismatch occurred during validation.
    - `HASH_EXIT_MANIFEST_MISSING_ENTRY`: Signifies that a hash validation was attempted but an entry was missing in the manifest.
    - `HASH_EXIT_MANIFEST_UNKNOWN_HASH`: Indicates that the manifest is present, but the hash format is unknown.
    - `HASH_EXIT_MANIFEST_FILE_ERROR`: Represents an error where the manifest is either missing or not in a known format.
- **Description**: The `hash_exit_code_t` is an enumeration that defines various exit codes for hash operations, indicating the success or failure of hash generation and validation processes. It provides specific codes for successful operations, generic failures, mismatches, and issues related to manifest files, such as missing entries or unknown hash formats. This enum is used to standardize the response of hash-related functions, allowing for consistent error handling and reporting.


---
### hash\_manifest\_result\_t<!-- {{#data_structure:hash_manifest_result_t}} -->
- **Type**: `enum`
- **Members**:
    - `HASH_MANIFEST_NOT_FOUND`: Indicates that the hash manifest was not found.
    - `HASH_MANIFEST_MISMATCH`: Indicates that there is a mismatch in the hash manifest.
    - `HASH_MANIFEST_OK`: Indicates that the hash manifest is correct and matches.
- **Description**: The `hash_manifest_result_t` is an enumeration that represents the result of verifying a hash manifest. It provides three possible outcomes: `HASH_MANIFEST_NOT_FOUND` when the manifest is missing, `HASH_MANIFEST_MISMATCH` when there is a discrepancy between expected and actual hash values, and `HASH_MANIFEST_OK` when the manifest verification is successful and all hashes match as expected.


---
### hash\_params<!-- {{#data_structure:hash_params}} -->
- **Type**: `struct`
- **Members**:
    - `input`: A string representing the input file or data to be hashed.
    - `xxh64`: A boolean flag indicating whether the xxh64 hash algorithm should be used.
    - `sha1`: A boolean flag indicating whether the sha1 hash algorithm should be used.
    - `sha256`: A boolean flag indicating whether the sha256 hash algorithm should be used.
    - `uuid`: A boolean flag indicating whether a UUIDv5 should be generated.
    - `no_layer`: A boolean flag indicating whether to exclude per-layer hashing.
    - `manifest_is_usable`: A boolean flag indicating whether the manifest file is usable for hash verification.
    - `manifest_file`: A string representing the path to the manifest file for hash verification.
- **Description**: The `hash_params` struct is designed to encapsulate parameters for hashing operations, including the input data and various flags to specify which hash algorithms to use (xxh64, sha1, sha256, or UUIDv5). It also includes options to exclude per-layer hashing and to specify a manifest file for hash verification, indicating whether the manifest is usable for this purpose.


---
### manifest\_check\_params<!-- {{#data_structure:manifest_check_params}} -->
- **Type**: `struct`
- **Members**:
    - `xxh64`: A boolean flag indicating whether the xxh64 hash should be used.
    - `sha1`: A boolean flag indicating whether the sha1 hash should be used.
    - `sha256`: A boolean flag indicating whether the sha256 hash should be used.
    - `uuid`: A boolean flag indicating whether the uuid should be used.
- **Description**: The `manifest_check_params` struct is a simple data structure used to store boolean flags that indicate whether specific types of hashes (xxh64, sha1, sha256, and uuid) should be checked or used in a manifest verification process. Each member corresponds to a different hash type, allowing the program to determine which hash checks are applicable based on the manifest's contents.


# Functions

---
### hash\_manifest\_result\_to\_str<!-- {{#callable:hash_manifest_result_to_str}} -->
The function `hash_manifest_result_to_str` converts a `hash_manifest_result_t` enumeration value to its corresponding string representation.
- **Inputs**:
    - `value`: An enumeration value of type `hash_manifest_result_t` which indicates the result of a hash manifest operation.
- **Control Flow**:
    - The function uses a switch statement to match the input `value` against predefined enumeration cases.
    - If `value` is `HASH_MANIFEST_NOT_FOUND`, it returns the string "Not Found".
    - If `value` is `HASH_MANIFEST_MISMATCH`, it returns the string "Mismatch".
    - If `value` is `HASH_MANIFEST_OK`, it returns the string "Ok".
    - If `value` does not match any of the predefined cases, it returns the string "?".
- **Output**: A constant character pointer to a string that represents the input enumeration value.


---
### hash\_exit\_code\_to\_str<!-- {{#callable:hash_exit_code_to_str}} -->
The function `hash_exit_code_to_str` converts a `hash_exit_code_t` enumeration value to its corresponding string representation.
- **Inputs**:
    - `value`: An enumeration value of type `hash_exit_code_t` representing a specific exit code.
- **Control Flow**:
    - The function uses a switch statement to match the input `value` against predefined enumeration constants.
    - For each case, it returns a corresponding string literal that describes the exit code.
    - If the input value does not match any predefined case, the function returns a default string "?".
- **Output**: A constant character pointer to a string that describes the input exit code.


---
### hash\_print\_usage<!-- {{#callable:hash_print_usage}} -->
The `hash_print_usage` function displays the usage instructions for a command-line tool that hashes GGUF files, detailing available options and their descriptions.
- **Inputs**:
    - `executable`: A constant character pointer representing the name of the executable file, typically used to display the command in the usage message.
- **Control Flow**:
    - Initialize a `hash_params` object with default values.
    - Print a newline character to separate the usage message from previous output.
    - Print the usage format, substituting the `executable` argument into the command template.
    - Print a description of the tool's purpose: 'Hash a GGUF file'.
    - Print a list of available options, each with a brief description of its function.
- **Output**: The function does not return a value; it outputs the usage instructions directly to the standard output.


---
### hash\_params\_parse\_ex<!-- {{#callable:hash_params_parse_ex}} -->
The `hash_params_parse_ex` function parses command-line arguments to configure hash parameters for a file hashing operation.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
    - `params`: A reference to a `hash_params` structure where parsed parameters will be stored.
- **Control Flow**:
    - Initialize a string `arg` and a boolean `invalid_param` to track invalid parameters.
    - Set `arg_prefix` to "--" to identify long-form command-line options.
    - Iterate over `argv` starting from index 1, checking if each argument starts with "--".
    - Replace underscores with hyphens in arguments that start with the prefix `arg_prefix`.
    - Check each argument against known options (e.g., `--xxh64`, `--sha1`, `--uuid`, etc.) and set corresponding fields in `params`.
    - If `-h` or `--help` is encountered, print usage information and exit the program.
    - If `-c` or `--check` is encountered, expect a manifest file argument and store it in `params.manifest_file`.
    - If an unknown argument is encountered, throw an `std::invalid_argument` exception.
    - If `invalid_param` is true, throw an `std::invalid_argument` exception for invalid parameters.
    - Ensure at least one non-option argument remains for input, otherwise throw an `std::invalid_argument` exception.
    - Set `params.input` to the next argument in `argv`.
- **Output**: The function does not return a value but modifies the `params` structure to reflect the parsed command-line options.
- **Functions called**:
    - [`hash_print_usage`](#hash_print_usage)


---
### hash\_params\_parse<!-- {{#callable:hash_params_parse}} -->
The `hash_params_parse` function attempts to parse command-line arguments into a `hash_params` structure, handling any invalid arguments by printing an error message and exiting the program.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
    - `params`: A reference to a `hash_params` structure where parsed parameters will be stored.
- **Control Flow**:
    - Initialize a boolean variable `result` to true.
    - Attempt to call [`hash_params_parse_ex`](#hash_params_parse_ex) with the provided arguments to parse them into the `params` structure.
    - Catch any `std::invalid_argument` exceptions thrown by [`hash_params_parse_ex`](#hash_params_parse_ex).
    - If an exception is caught, print the error message to standard error, call [`hash_print_usage`](#hash_print_usage) to display usage information, and exit the program with a failure status.
    - Return the `result` variable, which remains true if no exceptions were caught.
- **Output**: A boolean value indicating whether the parsing was successful (always true unless an exception is caught, in which case the program exits).
- **Functions called**:
    - [`hash_params_parse_ex`](#hash_params_parse_ex)
    - [`hash_print_usage`](#hash_print_usage)


---
### manifest\_type<!-- {{#callable:manifest_type}} -->
The `manifest_type` function checks a manifest file for specific hash types and updates a `manifest_check_params` structure accordingly.
- **Inputs**:
    - `manifest_file`: A string representing the path to the manifest file to be checked.
    - `manifest_check`: A reference to a `manifest_check_params` structure that will be updated based on the hash types found in the manifest file.
- **Control Flow**:
    - Check if the `manifest_file` string is empty; if so, return false.
    - Attempt to open the file specified by `manifest_file`; if the file cannot be opened, return false.
    - Iterate over each line in the file, treating each line as a potential manifest entry.
    - For each line, extract the first word as the hash type and update the corresponding field in `manifest_check` if it matches a known hash type (SHA256, SHA1, XXH64, UUID).
    - Return true after processing all lines.
- **Output**: A boolean value indicating whether the manifest file was successfully processed and contained known hash types.


---
### manifest\_verify<!-- {{#callable:manifest_verify}} -->
The `manifest_verify` function checks if a given hash and tensor name match an entry in a specified manifest file.
- **Inputs**:
    - `manifest_file`: A string representing the path to the manifest file to be verified.
    - `hash_type_str`: A string representing the type of hash (e.g., 'sha256', 'sha1', etc.) to be verified.
    - `hash_str`: A string representing the hash value to be verified against the manifest.
    - `tensor_name`: A string representing the name of the tensor to be verified against the manifest.
- **Control Flow**:
    - Check if the manifest file path is empty; if so, return `HASH_MANIFEST_NOT_FOUND`.
    - Attempt to open the manifest file; if it fails, return `HASH_MANIFEST_NOT_FOUND`.
    - Iterate over each line in the manifest file, parsing the hash type, hash value, and tensor name from each line.
    - If the parsed hash type does not match `hash_type_str`, continue to the next line.
    - If the parsed tensor name does not match `tensor_name`, continue to the next line.
    - If both the hash type and tensor name match, compare the parsed hash value with `hash_str`.
    - Return `HASH_MANIFEST_OK` if the hash values match, otherwise return `HASH_MANIFEST_MISMATCH`.
    - If no matching entry is found after reading all lines, return `HASH_MANIFEST_NOT_FOUND`.
- **Output**: Returns a `hash_manifest_result_t` enum value indicating the result of the verification: `HASH_MANIFEST_OK`, `HASH_MANIFEST_MISMATCH`, or `HASH_MANIFEST_NOT_FOUND`.


---
### generate\_uuidv5<!-- {{#callable:generate_uuidv5}} -->
The `generate_uuidv5` function generates a UUID version 5 from a given SHA-1 digest by copying the first 16 bytes and setting specific bits for version and variant.
- **Inputs**:
    - `sha1_digest`: An array of 20 unsigned char elements representing the SHA-1 digest from which the UUID will be generated.
    - `uuid`: An array of 16 unsigned char elements where the generated UUID will be stored.
- **Control Flow**:
    - Copy the first 16 bytes of the `sha1_digest` array into the `uuid` array.
    - Modify the 7th byte of the `uuid` array to set the version to 5 by clearing the upper 4 bits and setting them to 0101.
    - Modify the 9th byte of the `uuid` array to set the variant to 0b10XX by clearing the upper 2 bits and setting them to 10.
- **Output**: The function does not return a value; it modifies the `uuid` array in place to contain the generated UUID version 5.


---
### gguf\_hash<!-- {{#callable:gguf_hash}} -->
The `gguf_hash` function computes and optionally verifies hash values for tensors in a GGUF file using specified hash algorithms and compares them against a manifest if provided.
- **Inputs**:
    - `hash_params`: A structure containing parameters for hashing, including the input file name, flags for using different hash algorithms (xxh64, sha1, sha256, uuid), a flag to exclude per-layer hashing, and manifest file details for verification.
- **Control Flow**:
    - Initialize hash contexts for xxh64, sha1, sha256, and uuid based on the flags in `hash_params`.
    - Open the GGUF file and retrieve the number of tensors.
    - Iterate over each tensor, compute per-layer hashes if `no_layer` is false, and update overall model hashes for each selected hash algorithm.
    - For each hash, if a manifest is provided, verify the computed hash against the manifest and print the results.
    - After processing all tensors, finalize the overall model hashes and verify them against the manifest if applicable.
    - Free allocated resources for the GGUF and GGML contexts.
    - Return a hash exit code based on the success or failure of hash generation or verification.
- **Output**: Returns a `hash_exit_code_t` indicating the result of the hash operation, such as success, failure, or manifest-related errors.
- **Functions called**:
    - [`XXH_errorcode::XXH64_reset`](deps/xxhash/xxhash.h.driver.md#XXH_errorcodeXXH64_reset)
    - [`SHA1Init`](deps/sha1/sha1.c.driver.md#SHA1Init)
    - [`sha256_init`](deps/sha256/sha256.c.driver.md#sha256_init)
    - [`SHA1Update`](deps/sha1/sha1.c.driver.md#SHA1Update)
    - [`gguf_init_from_file`](../../ggml/src/gguf.cpp.driver.md#gguf_init_from_file)
    - [`gguf_get_n_tensors`](../../ggml/src/gguf.cpp.driver.md#gguf_get_n_tensors)
    - [`gguf_get_tensor_name`](../../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_name)
    - [`ggml_get_tensor`](../../ggml/src/ggml.c.driver.md#ggml_get_tensor)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`manifest_verify`](#manifest_verify)
    - [`hash_manifest_result_to_str`](#hash_manifest_result_to_str)
    - [`XXH_errorcode::XXH64_update`](deps/xxhash/xxhash.h.driver.md#XXH_errorcodeXXH64_update)
    - [`SHA1`](deps/sha1/sha1.c.driver.md#SHA1)
    - [`sha256_hash`](deps/sha256/sha256.c.driver.md#sha256_hash)
    - [`sha256_update`](deps/sha256/sha256.c.driver.md#sha256_update)
    - [`XXH64_hash_t::XXH64_digest`](deps/xxhash/xxhash.h.driver.md#XXH64_hash_tXXH64_digest)
    - [`SHA1Final`](deps/sha1/sha1.c.driver.md#SHA1Final)
    - [`sha256_final`](deps/sha256/sha256.c.driver.md#sha256_final)
    - [`generate_uuidv5`](#generate_uuidv5)
    - [`ggml_free`](../../ggml/src/ggml.c.driver.md#ggml_free)
    - [`gguf_free`](../../ggml/src/gguf.cpp.driver.md#gguf_free)


---
### main<!-- {{#callable:main}} -->
The `main` function parses command-line arguments to configure hash parameters, checks and processes a manifest file if provided, and then computes or verifies hashes for a GGUF file, returning an appropriate exit code.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize `hash_params` and `manifest_check_params` structures to store hash configuration and manifest check parameters.
    - Parse command-line arguments using [`hash_params_parse`](#hash_params_parse) to populate the `params` structure.
    - Check if a manifest file is specified in `params.manifest_file`; if so, attempt to determine its type using [`manifest_type`](#manifest_type).
    - If the manifest file cannot be opened or contains no known hash formats, print an error message and return an error code.
    - Print the types of hashes found in the manifest file.
    - If no specific hash type is selected by the user, automatically select the most secure hash type available in the manifest.
    - Set `params.manifest_is_usable` to true if a valid manifest is provided.
    - If no hash type is selected, default to using `xxh64`.
    - Call [`gguf_hash`](#gguf_hash) with the configured `params` to compute or verify hashes.
    - If a manifest is used, print the verification results.
    - Return the exit code from [`gguf_hash`](#gguf_hash).
- **Output**: The function returns an integer exit code indicating the success or failure of the hash operation, based on the `hash_exit_code_t` enumeration.
- **Functions called**:
    - [`hash_params_parse`](#hash_params_parse)
    - [`manifest_type`](#manifest_type)
    - [`gguf_hash`](#gguf_hash)
    - [`hash_exit_code_to_str`](#hash_exit_code_to_str)


