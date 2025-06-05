# Purpose
This C++ source code file is a test suite designed to validate the functionality of a custom regular expression library, particularly focusing on its support for partial final matches. The file includes two main test functions: [`test_regex`](#test_regex) and [`test_regex_to_reversed_partial_regex`](#test_regex_to_reversed_partial_regex). The [`test_regex`](#test_regex) function evaluates the behavior of the `common_regex` class by running a series of test cases that check for full and partial matches against various input strings and patterns. It uses a helper function, [`assert_equals`](#assert_equals), to compare expected and actual results, throwing an exception if they do not match. The [`test_regex_to_reversed_partial_regex`](#test_regex_to_reversed_partial_regex) function tests the conversion of regular expressions into a form that supports reversed partial matching, ensuring that the transformation logic is correct.

The file imports necessary components from "common.h" and "regex-partial.h", indicating that it relies on external definitions for the `common_regex` class and related types like `common_regex_match` and `common_regex_match_type`. The code is structured to be executed as a standalone program, with a [`main`](#main) function that calls the test functions and outputs a success message if all tests pass. This file does not define public APIs or external interfaces; instead, it serves as an internal validation tool to ensure the correctness and robustness of the regular expression handling capabilities provided by the associated library.
# Imports and Dependencies

---
- `common.h`
- `regex-partial.h`
- `sstream`
- `iostream`
- `optional`


# Data Structures

---
### test\_case<!-- {{#data_structure:test_case}} -->
- **Type**: `struct`
- **Members**:
    - `pattern`: A string representing the regex pattern to be tested.
    - `inputs_outputs`: A vector of input_output structs, each containing an input string and the expected regex match result.
- **Description**: The `test_case` struct is designed to encapsulate a regex pattern and a series of test cases for that pattern. Each test case consists of an input string and the expected output, represented by the `input_output` struct, which includes the input string and a `common_regex_match` object indicating the expected match result. This struct is used to facilitate testing of regex patterns against various inputs to verify correct matching behavior.


---
### input\_output<!-- {{#data_structure:test_case::input_output}} -->
- **Type**: `struct`
- **Members**:
    - `input`: A string representing the input data for the regex test.
    - `output`: A common_regex_match object representing the expected output of the regex match.
- **Description**: The `input_output` struct is designed to pair an input string with its expected regex match result, encapsulated in a `common_regex_match` object. This struct is used within a vector to store multiple test cases for regex pattern matching, allowing for systematic testing of regex functionality by comparing actual matches against expected results.


# Functions

---
### assert\_equals<!-- {{#callable:assert_equals}} -->
The `assert_equals` function compares two values and throws a runtime error if they are not equal, providing diagnostic output to the standard error stream.
- **Inputs**:
    - `expected`: The expected value of type T that the actual value is compared against.
    - `actual`: The actual value of type T that is being tested against the expected value.
- **Control Flow**:
    - The function checks if the `expected` value is not equal to the `actual` value using the `!=` operator.
    - If the values are not equal, it outputs the expected and actual values to the standard error stream using `std::cerr`.
    - The error messages are flushed to ensure they are immediately outputted.
    - A `std::runtime_error` is thrown with the message "Test failed" to indicate the test did not pass.
- **Output**: The function does not return a value; it throws a `std::runtime_error` if the expected and actual values are not equal.


---
### common\_regex\_match\_type\_name<!-- {{#callable:common_regex_match_type_name}} -->
The function `common_regex_match_type_name` converts a `common_regex_match_type` enum value to its corresponding string representation.
- **Inputs**:
    - `type`: An enum value of type `common_regex_match_type` which indicates the type of regex match (e.g., NONE, PARTIAL, FULL).
- **Control Flow**:
    - The function uses a switch statement to determine the string representation of the input enum value.
    - If the input is `COMMON_REGEX_MATCH_TYPE_NONE`, it returns the string "COMMON_REGEX_MATCH_TYPE_NONE".
    - If the input is `COMMON_REGEX_MATCH_TYPE_PARTIAL`, it returns the string "COMMON_REGEX_MATCH_TYPE_PARTIAL".
    - If the input is `COMMON_REGEX_MATCH_TYPE_FULL`, it returns the string "COMMON_REGEX_MATCH_TYPE_FULL".
    - If the input does not match any known enum values, it returns the string "?".
- **Output**: A string that represents the name of the `common_regex_match_type` enum value.


---
### test\_regex<!-- {{#callable:test_regex}} -->
The `test_regex` function tests various regular expression patterns against a set of input strings to verify the correctness of the `common_regex` class's matching capabilities, including partial and full matches.
- **Inputs**:
    - `None`: The function does not take any input parameters.
- **Control Flow**:
    - The function begins by printing the function name using `printf`.
    - A lambda function `test` is defined to handle individual test cases, which takes a `test_case` object as input.
    - For each test case, a `common_regex` object is created with the given pattern, and the pattern is printed.
    - The function iterates over each input-output pair in the test case.
    - For each input, it performs a search using the `common_regex` object and compares the result with the expected output.
    - If the actual output does not match the expected output, it constructs a string representation of both expected and actual matches and prints them.
    - If a mismatch is found, it throws a `std::runtime_error` indicating a test failure.
    - The function then calls the `test` lambda with various predefined test cases, each containing a pattern and a set of input-output pairs to validate different regex scenarios.
- **Output**: The function does not return any value; it throws an exception if any test case fails, otherwise it completes without output.
- **Functions called**:
    - [`common_regex_match_type_name`](#common_regex_match_type_name)
    - [`regex_to_reversed_partial_regex`](../common/regex-partial.cpp.driver.md#regex_to_reversed_partial_regex)


---
### test\_regex\_to\_reversed\_partial\_regex<!-- {{#callable:test_regex_to_reversed_partial_regex}} -->
The function `test_regex_to_reversed_partial_regex` tests the [`regex_to_reversed_partial_regex`](../common/regex-partial.cpp.driver.md#regex_to_reversed_partial_regex) function by asserting that it correctly transforms various regular expressions into their reversed partial regex forms.
- **Inputs**: None
- **Control Flow**:
    - Prints the function name using `printf`.
    - Uses `assert_equals` to compare the expected reversed partial regex with the actual output from [`regex_to_reversed_partial_regex`](../common/regex-partial.cpp.driver.md#regex_to_reversed_partial_regex) for various input regex patterns.
    - Throws a runtime error if any assertion fails, indicating a test failure.
- **Output**: The function does not return any value; it performs assertions to validate the correctness of the [`regex_to_reversed_partial_regex`](../common/regex-partial.cpp.driver.md#regex_to_reversed_partial_regex) function.
- **Functions called**:
    - [`regex_to_reversed_partial_regex`](../common/regex-partial.cpp.driver.md#regex_to_reversed_partial_regex)


---
### main<!-- {{#callable:main}} -->
The `main` function executes two test functions and outputs a success message if all tests pass.
- **Inputs**: None
- **Control Flow**:
    - The function [`test_regex_to_reversed_partial_regex`](#test_regex_to_reversed_partial_regex) is called to execute its tests.
    - The function [`test_regex`](#test_regex) is called to execute its tests.
    - If both test functions complete without throwing exceptions, a message 'All tests passed.' is printed to the standard output.
- **Output**: The function does not return any value; it outputs a message to the console.
- **Functions called**:
    - [`test_regex_to_reversed_partial_regex`](#test_regex_to_reversed_partial_regex)
    - [`test_regex`](#test_regex)


