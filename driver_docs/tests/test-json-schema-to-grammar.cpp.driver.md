# Purpose
This C++ source code file is designed to test the conversion of JSON schemas into grammar representations. It includes a series of test cases that validate the functionality of a JSON schema-to-grammar conversion tool. The file imports necessary headers and libraries, such as `nlohmann/json.hpp` for JSON handling and `llama-grammar.h` for grammar parsing. The core functionality is encapsulated in the `TestCase` struct, which defines methods to verify the correctness of the conversion by comparing expected and actual grammar outputs, checking the parseability of the expected grammar, and ensuring the status of the test case matches the expected outcome.

The file contains a [`main`](#main) function that orchestrates the execution of these tests across different programming environments, including C++, Python, and JavaScript, depending on the availability of the respective runtime environments. It uses the [`test_all`](#test_all) function to iterate over a series of predefined test cases, each specifying a JSON schema and its expected grammar output. The tests cover a wide range of scenarios, including various data types, constraints, and schema structures. The file also includes utility functions for reading and writing files, trimming strings, and handling test case execution. Overall, this file serves as a comprehensive test suite for ensuring the reliability and accuracy of the JSON schema-to-grammar conversion process.
# Imports and Dependencies

---
- `json-schema-to-grammar.h`
- `../src/llama-grammar.h`
- `nlohmann/json.hpp`
- `cassert`
- `fstream`
- `sstream`
- `regex`


# Data Structures

---
### TestCaseStatus<!-- {{#data_structure:TestCaseStatus}} -->
- **Type**: `enum`
- **Members**:
    - `SUCCESS`: Represents a successful test case status.
    - `FAILURE`: Represents a failed test case status.
- **Description**: The `TestCaseStatus` enum is used to represent the status of a test case, indicating whether it has succeeded or failed. It provides two possible values: `SUCCESS` and `FAILURE`, which can be used to easily check and handle the outcome of test cases in the code.


---
### TestCase<!-- {{#data_structure:TestCase}} -->
- **Type**: `struct`
- **Members**:
    - `expected_status`: Holds the expected status of the test case, indicating success or failure.
    - `name`: Stores the name of the test case.
    - `schema`: Contains the JSON schema associated with the test case.
    - `expected_grammar`: Represents the expected grammar output for the test case.
- **Description**: The `TestCase` struct is designed to encapsulate the details of a test case for verifying JSON schema to grammar conversion. It includes fields for the expected status, name, schema, and expected grammar of the test case. The struct provides methods to verify the actual grammar against the expected grammar, check the parseability of the expected grammar, and validate the status of the test case. This struct is integral to the testing framework, ensuring that JSON schemas are correctly converted to the expected grammar format.
- **Member Functions**:
    - [`TestCase::_print_failure_header`](#TestCase_print_failure_header)
    - [`TestCase::verify`](#TestCaseverify)
    - [`TestCase::verify_expectation_parseable`](#TestCaseverify_expectation_parseable)
    - [`TestCase::verify_status`](#TestCaseverify_status)

**Methods**

---
#### TestCase::\_print\_failure\_header<!-- {{#callable:TestCase::_print_failure_header}} -->
The `_print_failure_header` function outputs a formatted error message to the standard error stream indicating a test failure, including the test name and schema.
- **Inputs**: None
- **Control Flow**:
    - The function uses `fprintf` to write a formatted string to `stderr`.
    - The formatted string includes a header with the test name and the schema associated with the test case.
- **Output**: The function does not return any value; it outputs directly to the standard error stream.
- **See also**: [`TestCase`](#TestCase)  (Data Structure)


---
#### TestCase::verify<!-- {{#callable:TestCase::verify}} -->
The `verify` function checks if the trimmed version of the `actual_grammar` string matches the `expected_grammar` string, and if not, it prints a failure message and triggers an assertion failure.
- **Inputs**:
    - `actual_grammar`: A constant reference to a string representing the actual grammar to be verified against the expected grammar.
- **Control Flow**:
    - Trim the `actual_grammar` string using the [`trim`](#trim) function.
    - Compare the trimmed `actual_grammar` with the trimmed `expected_grammar`.
    - If they are not equal, call the [`_print_failure_header`](#TestCase_print_failure_header) method to print a failure message header.
    - Print the expected and actual grammar strings to the standard error output.
    - Trigger an assertion failure using `assert(false)` to indicate the verification failure.
- **Output**: The function does not return any value; it either completes successfully or triggers an assertion failure if the verification fails.
- **Functions called**:
    - [`trim`](#trim)
    - [`TestCase::_print_failure_header`](#TestCase_print_failure_header)
- **See also**: [`TestCase`](#TestCase)  (Data Structure)


---
#### TestCase::verify\_expectation\_parseable<!-- {{#callable:TestCase::verify_expectation_parseable}} -->
The `verify_expectation_parseable` function checks if the `expected_grammar` string can be successfully parsed by the `llama_grammar_parser` and contains a 'root' symbol, throwing an error if parsing fails.
- **Inputs**: None
- **Control Flow**:
    - The function attempts to parse the `expected_grammar` string using `llama_grammar_parser`.
    - If the 'root' symbol is not found in the parsed grammar, a `std::runtime_error` is thrown with a message indicating the grammar failed to parse.
    - If an exception is caught, the function prints a failure header and a grammar error message, then asserts false to indicate a test failure.
- **Output**: The function does not return any value; it asserts false if parsing fails, indicating a test failure.
- **Functions called**:
    - [`TestCase::_print_failure_header`](#TestCase_print_failure_header)
- **See also**: [`TestCase`](#TestCase)  (Data Structure)


---
#### TestCase::verify\_status<!-- {{#callable:TestCase::verify_status}} -->
The `verify_status` function checks if the provided `status` matches the `expected_status` of a test case and reports a failure if they do not match.
- **Inputs**:
    - `status`: The actual status of the test case, which is of type `TestCaseStatus` and can be either `SUCCESS` or `FAILURE`.
- **Control Flow**:
    - The function compares the input `status` with the `expected_status` of the `TestCase` object.
    - If the `status` does not match the `expected_status`, it calls the [`_print_failure_header`](#TestCase_print_failure_header) method to print a failure message header.
    - It then prints the expected and actual statuses to the standard error stream.
    - Finally, it triggers an assertion failure using `assert(false)` to indicate the test case failure.
- **Output**: The function does not return any value; it outputs failure information to the standard error stream and triggers an assertion failure if the statuses do not match.
- **Functions called**:
    - [`TestCase::_print_failure_header`](#TestCase_print_failure_header)
- **See also**: [`TestCase`](#TestCase)  (Data Structure)



# Functions

---
### trim<!-- {{#callable:trim}} -->
The `trim` function removes leading and trailing whitespace characters from a given string and replaces leading spaces in lines with a single space.
- **Inputs**:
    - `source`: A constant reference to a `std::string` that represents the input string to be trimmed.
- **Control Flow**:
    - Create a copy of the input string `source` into a new string `s`.
    - Remove leading whitespace characters (spaces, newlines, carriage returns, and tabs) from `s` using `erase` and `find_first_not_of`.
    - Remove trailing whitespace characters from `s` using `erase` and `find_last_not_of`.
    - Use `std::regex_replace` to replace leading spaces in each line of `s` with a single space.
    - Return the modified string `s`.
- **Output**: A `std::string` that is the trimmed version of the input string, with leading and trailing whitespace removed and leading spaces in lines replaced.


---
### write<!-- {{#callable:write}} -->
The `write` function writes a given string content to a specified file.
- **Inputs**:
    - `file`: A constant reference to a string representing the file path where the content will be written.
    - `content`: A constant reference to a string containing the content to be written to the file.
- **Control Flow**:
    - Create an ofstream object `f` to handle file operations.
    - Open the file specified by the `file` argument using `f.open()`.
    - Write the `content` string to the file using the stream insertion operator `<<`.
    - Close the file using `f.close()` to ensure all data is flushed and resources are released.
- **Output**: This function does not return any value.


---
### read<!-- {{#callable:read}} -->
The `read` function reads the entire content of a file into a string and returns it.
- **Inputs**:
    - `file`: A constant reference to a string representing the file path to be read.
- **Control Flow**:
    - Create an output string stream `actuals`.
    - Open the file specified by `file` using an input file stream and read its buffer into `actuals`.
    - Return the string content of `actuals`.
- **Output**: A string containing the entire content of the specified file.


---
### test\_all<!-- {{#callable:test_all}} -->
The `test_all` function runs a series of JSON schema conversion tests for a specified language using a provided test runner function.
- **Inputs**:
    - `lang`: A string representing the programming language for which the JSON schema conversion tests are being run.
    - `runner`: A function that takes a `TestCase` object and executes the test logic for that test case.
- **Control Flow**:
    - Prints a header indicating the start of JSON schema conversion tests for the specified language.
    - Defines a lambda function `test` that takes a `TestCase` object, prints the test name and expected status, and executes the `runner` function with the test case.
    - Calls the `test` lambda function with multiple predefined `TestCase` objects, each representing a different JSON schema conversion scenario.
    - Each `TestCase` object includes an expected status, a name, a JSON schema, and an expected grammar output.
    - The `runner` function is expected to handle the logic of verifying the conversion of the JSON schema to the expected grammar.
- **Output**: The function does not return any value; it performs tests and outputs results to standard error.


---
### main<!-- {{#callable:main}} -->
The `main` function initializes environment checks and executes a series of JSON schema conversion tests across different programming languages, handling errors and conditional test execution based on environment variables.
- **Inputs**: None
- **Control Flow**:
    - Prints the availability of LLAMA_NODE and LLAMA_PYTHON based on environment variables.
    - Executes C++ JSON schema conversion tests using [`test_all`](#test_all) function, verifying the conversion and status of each test case.
    - Checks if slow tests should be skipped on an emulator using the `LLAMA_SKIP_TESTS_SLOW_ON_EMULATOR` environment variable.
    - If not skipping, checks for Python availability and version, then runs Python JSON schema conversion tests if conditions are met, otherwise prints a warning.
    - Checks for Node.js availability, then runs JavaScript JSON schema conversion tests if conditions are met, otherwise prints a warning.
    - Executes a final set of tests to check the validity of expectations for test cases with a SUCCESS status.
- **Output**: The function does not return any value; it performs tests and outputs results and warnings to the standard error stream.
- **Functions called**:
    - [`test_all`](#test_all)
    - [`json_schema_to_grammar`](../common/json-schema-to-grammar.cpp.driver.md#json_schema_to_grammar)
    - [`write`](#write)
    - [`read`](#read)


