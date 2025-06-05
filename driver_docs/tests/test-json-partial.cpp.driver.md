# Purpose
This C++ source code file is designed to test the functionality of a JSON parsing and "healing" mechanism. The code includes a series of test cases that evaluate the ability of the `common_json_parse` function to correctly parse JSON strings and handle incomplete or malformed JSON data by applying a "healing" process. The file imports necessary headers, including "common.h" and "json-partial.h", which likely contain the definitions and implementations of the JSON parsing and healing logic. The [`assert_equals`](#assert_equals) function is a utility used to verify that the expected and actual outputs of the parsing function match, throwing an exception if they do not.

The core functionality is encapsulated in the [`test_json_healing`](#test_json_healing) function, which defines several test scenarios using both complete and partial JSON strings. The tests check whether the parser can correctly identify and handle incomplete JSON literals, such as numbers, booleans, and null values, and whether it can apply a "healing marker" to indicate where the JSON was incomplete. The [`main`](#main) function executes the [`test_json_healing`](#test_json_healing) function and reports the success of all tests. This file is primarily focused on validating the robustness and correctness of the JSON parsing and healing process, making it a critical component for ensuring the reliability of applications that depend on this JSON handling functionality.
# Imports and Dependencies

---
- `common.h`
- `json-partial.h`
- `exception`
- `iostream`
- `stdexcept`


# Functions

---
### assert\_equals<!-- {{#callable:assert_equals}} -->
The `assert_equals` function checks if two values are equal and throws a runtime error if they are not, while also printing the expected and actual values to the standard error stream.
- **Inputs**:
    - `expected`: The expected value of type T that the actual value is compared against.
    - `actual`: The actual value of type T that is being tested for equality against the expected value.
- **Control Flow**:
    - The function compares the `expected` and `actual` values using the `!=` operator.
    - If the values are not equal, it prints the expected and actual values to the standard error stream using `std::cerr`.
    - It flushes the error stream to ensure all output is written immediately.
    - A `std::runtime_error` is thrown with the message "Test failed" to indicate the assertion failure.
- **Output**: The function does not return a value; it either completes silently if the values are equal or throws an exception if they are not.


---
### test\_json\_healing<!-- {{#callable:test_json_healing}} -->
The `test_json_healing` function tests the ability of a JSON parser to handle incomplete JSON strings and apply a 'healing' mechanism to produce valid JSON outputs.
- **Inputs**: None
- **Control Flow**:
    - Defines a lambda function `parse` to parse a given JSON string and check for a healing marker.
    - Defines a lambda function `parse_all` to parse all substrings of a given JSON string up to each character.
    - Calls `parse_all` with various JSON strings to test parsing and healing.
    - Defines a lambda function `test` to assert that parsing a list of JSON strings results in expected JSON outputs and healing markers.
    - Executes `test` with various JSON inputs to verify the parser's behavior with complete and incomplete JSON strings, checking for correct healing or error handling.
- **Output**: The function does not return a value but throws exceptions if parsing or assertions fail, indicating test failures.
- **Functions called**:
    - [`assert_equals`](#assert_equals)


---
### main<!-- {{#callable:main}} -->
The `main` function executes the [`test_json_healing`](#test_json_healing) function and outputs a success message if no exceptions are thrown.
- **Inputs**: None
- **Control Flow**:
    - The function [`test_json_healing`](#test_json_healing) is called to perform a series of JSON parsing tests.
    - If [`test_json_healing`](#test_json_healing) completes without throwing an exception, a success message 'All tests passed.' is printed to the standard error stream.
    - The function returns 0, indicating successful execution.
- **Output**: The function returns an integer value of 0, indicating successful execution.
- **Functions called**:
    - [`test_json_healing`](#test_json_healing)


