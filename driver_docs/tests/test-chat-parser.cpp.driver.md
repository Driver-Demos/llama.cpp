# Purpose
This C++ source code file is designed to test the functionality of a chat message parser, specifically focusing on grammar generation, parsing for tool calling, and handling various templates. The file serves as both a test suite and a command-line interface (CLI) tool to generate a Markdown summary of Jinja template formats. The code includes several test functions that validate different aspects of the `common_chat_msg_parser` class, such as reasoning parsing, regex consumption, JSON handling, and position management within the input string. These tests ensure that the parser can correctly interpret and manipulate chat messages, handle partial and complete JSON data, and manage string positions accurately.

The file is structured to include a series of static functions that perform assertions to verify expected outcomes against actual results, throwing exceptions when discrepancies are found. The [`assert_equals`](#assert_equals) and [`assert_throws`](#assert_throws) functions are central to this testing framework, providing a mechanism to validate the parser's behavior under various conditions. The code imports several headers, such as "chat-parser.h" and "regex-partial.h," indicating dependencies on external components for parsing and regex operations. The main function orchestrates the execution of all test cases, outputting a success message if all tests pass. This file is not intended to be a reusable library but rather a standalone executable for testing and validating the chat parser's functionality.
# Imports and Dependencies

---
- `exception`
- `iostream`
- `string`
- `chat-parser.h`
- `common.h`
- `log.h`
- `regex-partial.h`


# Global Variables

---
### barely\_healable\_jsons
- **Type**: `std::vector<std::string>`
- **Description**: The `barely_healable_jsons` is a constant vector of strings that contains various malformed or incomplete JSON snippets. These strings represent JSON structures that are not fully formed and may require 'healing' or correction to become valid JSON.
- **Use**: This variable is used in tests to simulate scenarios where JSON parsing needs to handle and potentially correct incomplete or malformed JSON inputs.


# Functions

---
### assert\_equals<!-- {{#callable:assert_equals}} -->
The `assert_equals` function compares an expected C-style string with an actual `std::string` and throws a runtime error if they are not equal.
- **Inputs**:
    - `expected`: A C-style string representing the expected value.
    - `actual`: A `std::string` representing the actual value to be compared against the expected value.
- **Control Flow**:
    - The function calls a templated version of `assert_equals` with `std::string` as the template argument, passing the `expected` and `actual` values.
    - The templated `assert_equals` function compares the two values for equality.
    - If the values are not equal, it outputs the expected and actual values to `std::cerr` and throws a `std::runtime_error`.
- **Output**: The function does not return a value; it throws an exception if the expected and actual values are not equal.


---
### assert\_throws<!-- {{#callable:assert_throws}} -->
The `assert_throws` function checks if a given function throws an exception and optionally verifies if the exception message matches a specified pattern.
- **Inputs**:
    - `fn`: A `std::function<void()>` representing the function that is expected to throw an exception.
    - `expected_exception_pattern`: An optional `std::string` representing a regex pattern that the exception message should match, defaulting to an empty string.
- **Control Flow**:
    - The function `fn` is executed within a try block.
    - If `fn` throws an exception, the catch block is executed, capturing the exception as `e`.
    - If `expected_exception_pattern` is empty, the function returns immediately, indicating that an exception was thrown as expected.
    - If `expected_exception_pattern` is not empty, a regex is created from it and the exception message is checked against this regex.
    - If the exception message matches the regex, the function returns, indicating the exception was as expected.
    - If the exception message does not match the regex, a `std::runtime_error` is thrown with a message indicating the mismatch.
    - If no exception is thrown by `fn`, a `std::runtime_error` is thrown indicating that an exception was expected but not thrown.
- **Output**: The function does not return a value but will throw a `std::runtime_error` if the function `fn` does not throw an exception or if the exception message does not match the expected pattern.


---
### test\_reasoning<!-- {{#callable:test_reasoning}} -->
The `test_reasoning` function tests the `common_chat_msg_parser` class's ability to parse reasoning content from chat messages under various configurations.
- **Inputs**: None
- **Control Flow**:
    - Initialize a `common_chat_msg_parser` object with a specific input string and configuration settings.
    - Call `try_parse_reasoning` with specific start and end tags and assert the expected boolean result.
    - Call `consume_rest` to get the remaining unparsed content and assert it matches the expected string.
    - Repeat the above steps with different configurations to test various parsing scenarios.
- **Output**: The function does not return any value; it uses assertions to validate the behavior of the `common_chat_msg_parser` class.
- **Functions called**:
    - [`assert_equals`](#assert_equals)


---
### test\_regex<!-- {{#callable:test_regex}} -->
The `test_regex` function tests various scenarios of regex consumption and matching using the `common_chat_msg_parser` class, including handling of partial and complete matches, and exception handling.
- **Inputs**: None
- **Control Flow**:
    - Defines a lambda function `test_throws` to test if consuming a regex throws an expected exception pattern.
    - Calls `test_throws` with different inputs and regex patterns to verify exception handling.
    - Creates a `common_chat_msg_parser` instance and tests consuming a regex and checking the remaining string.
    - Tests regex consumption in non-partial mode to verify if the regex was consumed or not.
    - Tests regex consumption with capturing groups and verifies the captured groups and position after the match.
    - Tests regex consumption in partial mode to ensure a partial match throws an exception.
    - Iterates over both partial and non-partial modes to test regex and literal consumption that should not match.
- **Output**: The function does not return any value; it performs assertions to validate regex consumption behavior and throws exceptions if tests fail.
- **Functions called**:
    - [`assert_throws`](#assert_throws)
    - [`common_regex`](../common/regex-partial.h.driver.md#common_regex)
    - [`assert_equals`](#assert_equals)


---
### test<!-- {{#callable:test}} -->
The `test` function validates the parsing and expected output of JSON data using a `common_chat_msg_parser` object.
- **Inputs**:
    - `input`: A string representing the input data to be parsed.
    - `is_partial`: A boolean indicating whether the input data is considered partial.
    - `args_paths`: A vector of vectors of strings representing paths to arguments within the JSON structure.
    - `content_paths`: A vector of vectors of strings representing paths to content within the JSON structure.
    - `expected`: A string representing the expected output after parsing and processing the input data.
- **Control Flow**:
    - A `common_chat_msg_parser` object is instantiated with the `input`, `is_partial`, and an empty options map.
    - The `try_consume_json_with_dumped_args` method is called on the parser object with `args_paths` and `content_paths` to attempt parsing the input.
    - An assertion checks that the parsing result (`js`) has a value, indicating successful parsing.
    - Another assertion checks that the `is_partial` property of the parsing result matches the input `is_partial`.
    - A final assertion checks that the parsed value matches the `expected` output, either as a string or a JSON dump, depending on the structure of `args_paths`.
- **Output**: The function does not return a value but throws an exception if any assertion fails, indicating a test failure.
- **Functions called**:
    - [`assert_equals`](#assert_equals)


---
### test\_with\_args<!-- {{#callable:test_with_args}} -->
The `test_with_args` function tests the parsing of JSON input with expected output, handling partial parsing and verifying the results.
- **Inputs**:
    - `input`: A `std::string` representing the JSON input to be parsed.
    - `expected`: A `std::string` representing the expected JSON output after parsing.
    - `parse_as_partial`: A `bool` indicating whether the input should be parsed as partial JSON (default is `true`).
    - `is_partial`: A `bool` indicating whether the expected result should be considered partial (default is `true`).
- **Control Flow**:
    - A `common_chat_msg_parser` object is instantiated with the `input` string and `parse_as_partial` flag.
    - The `try_consume_json_with_dumped_args` method is called on the parser with a predefined argument path `{{"args"}}`.
    - The function asserts that the JSON parsing result (`js`) has a value using [`assert_equals`](#assert_equals).
    - It asserts that the `is_partial` flag of the parsed JSON matches the `is_partial` input parameter.
    - Finally, it asserts that the dumped JSON value matches the `expected` string.
- **Output**: The function does not return a value; it performs assertions to validate the parsing behavior.
- **Functions called**:
    - [`assert_equals`](#assert_equals)


---
### test\_json\_with\_dumped\_args\_no\_args<!-- {{#callable:test_json_with_dumped_args_no_args}} -->
The function `test_json_with_dumped_args_no_args` tests the behavior of JSON parsing and healing when no arguments are provided in the JSON input.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`test`](#test) with a JSON string `{"name": "python"}` and checks that it remains unchanged when no arguments are specified.
    - It tests the same JSON string with an empty arguments path to ensure the output is the same.
    - The function iterates over a list of `barely_healable_jsons` and calls [`test`](#test) with each JSON string, expecting an empty JSON object `{}` as the output when arguments are specified further down.
    - Finally, it tests a JSON string `{"name": "python"` with a missing closing brace, expecting it to be healed to `{"name":"python"}` when arguments are specified.
- **Output**: The function does not return any value; it performs assertions to validate the expected behavior of JSON parsing and healing.
- **Functions called**:
    - [`test`](#test)


---
### test\_json\_with\_dumped\_args<!-- {{#callable:test_json_with_dumped_args}} -->
The `test_json_with_dumped_args` function tests the handling of JSON strings with potentially incomplete or malformed content and arguments, ensuring they are parsed and formatted correctly.
- **Inputs**: None
- **Control Flow**:
    - The function begins by testing JSON strings with partial content using the [`test`](#test) function, checking if they are correctly parsed and formatted.
    - It then tests JSON strings where the entire content is considered as arguments, ensuring that healing and dumping produce the same output as the input, albeit reformatted.
    - The function iterates over a list of barely healable JSON strings, testing each one to ensure they are handled correctly.
    - It tests full JSON strings with arguments using the [`test_with_args`](#test_with_args) function, verifying that they are parsed and formatted correctly regardless of whether they are parsed as partial or not.
    - The function proceeds to test various cases of partial JSON strings with partial arguments, including cases where the arguments are broken at different points (e.g., in object keys, values, or arrays), ensuring they are parsed and formatted correctly.
- **Output**: The function does not return any value; it performs assertions to validate the correctness of JSON parsing and formatting.
- **Functions called**:
    - [`test`](#test)
    - [`test_with_args`](#test_with_args)


---
### test\_positions<!-- {{#callable:test_positions}} -->
The `test_positions` function tests the position manipulation and boundary conditions of the `common_chat_msg_parser` class.
- **Inputs**: None
- **Control Flow**:
    - Initialize a `common_chat_msg_parser` object with the string "Hello, world!" and `is_partial` set to false.
    - Check that the initial position is 0 using [`assert_equals`](#assert_equals).
    - Attempt to move the position to 100, which should throw an exception, and verify the position remains 0.
    - Attempt to move the position back by 1, which should throw an exception, and verify the position remains 0.
    - Move the position to 8 and verify it using [`assert_equals`](#assert_equals).
    - Move the position back by 1 and verify it is now 7.
    - Consume the rest of the string from the current position and verify it equals "world!".
    - Move the position back to 0 and verify it.
    - Attempt to call `finish()` which should throw an exception, and verify the position remains 0.
    - Move the position to the end of the input string and call `finish()` without exceptions.
    - Repeat similar tests with `is_partial` set to true, ensuring the position can be moved to the end and `finish()` is called without exceptions.
- **Output**: The function does not return any value; it uses assertions to validate the behavior of the `common_chat_msg_parser` class.
- **Functions called**:
    - [`assert_throws`](#assert_throws)
    - [`assert_equals`](#assert_equals)


---
### main<!-- {{#callable:main}} -->
The `main` function executes a series of test functions to validate various aspects of chat handling and parsing, and outputs a success message if all tests pass.
- **Inputs**: None
- **Control Flow**:
    - The function calls `test_positions()` to test position handling in the parser.
    - It calls `test_json_with_dumped_args_no_args()` to test JSON parsing without arguments.
    - It calls `test_json_with_dumped_args()` to test JSON parsing with arguments.
    - It calls `test_reasoning()` to test reasoning parsing in chat messages.
    - It calls `test_regex()` to test regex handling in chat messages.
    - Finally, it outputs 'All tests passed!' to the console and returns 0.
- **Output**: The function returns an integer value of 0, indicating successful execution.
- **Functions called**:
    - [`test_positions`](#test_positions)
    - [`test_json_with_dumped_args_no_args`](#test_json_with_dumped_args_no_args)
    - [`test_json_with_dumped_args`](#test_json_with_dumped_args)
    - [`test_reasoning`](#test_reasoning)
    - [`test_regex`](#test_regex)


