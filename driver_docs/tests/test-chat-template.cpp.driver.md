# Purpose
This C++ source code file is designed to test and demonstrate the functionality of a chat templating system, which is part of a larger framework for handling chat interactions, likely in a conversational AI or chatbot context. The file includes several key components: it imports necessary libraries and headers, defines utility functions for message normalization and creation, and implements a [`main`](#main) function that orchestrates the testing of various chat templates. The code is structured to handle different chat message formats and templates, applying them to a predefined conversation and verifying the output against expected results.

The file is not a standalone executable but rather a test suite for validating the behavior of chat templates. It includes a series of test cases, each representing a different chat template, and checks if the formatted output matches the expected output. The code uses assertions to ensure correctness and provides detailed output for debugging purposes. The file also demonstrates the use of both built-in and custom templates, including those that utilize the Jinja templating engine. This setup is crucial for ensuring that the chat system can handle various message formats and templates, which is essential for supporting different conversational models and use cases.
# Imports and Dependencies

---
- `string`
- `vector`
- `sstream`
- `regex`
- `cassert`
- `llama.h`
- `common.h`
- `chat.h`


# Data Structures

---
### TestCase<!-- {{#data_structure:main::TestCase}} -->
- **Type**: `struct`
- **Members**:
    - `name`: A string representing the name of the test case.
    - `template_str`: A string containing the template used for the test case.
    - `expected_output`: A string representing the expected output of the test case.
    - `expected_output_jinja`: A string for the expected output when using Jinja templating.
    - `bos_token`: A string for the beginning-of-sequence token, defaulting to an empty string.
    - `eos_token`: A string for the end-of-sequence token, defaulting to an empty string.
    - `supported_with_jinja`: A boolean indicating if the test case supports Jinja templating, defaulting to true.
- **Description**: The `TestCase` struct is designed to encapsulate the details of a test case for chat template processing. It includes fields for the test case's name, the template string to be applied, and the expected output both with and without Jinja templating. Additionally, it contains optional fields for beginning and end-of-sequence tokens, and a boolean to indicate if Jinja templating is supported. This struct is used to validate the output of chat templates against expected results.


# Functions

---
### normalize\_newlines<!-- {{#callable:normalize_newlines}} -->
The `normalize_newlines` function replaces Windows-style newline sequences with Unix-style newline sequences in a given string, but only when compiled on Windows.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that represents the input string which may contain newline sequences.
- **Control Flow**:
    - The function checks if the code is being compiled on a Windows platform using the `_WIN32` preprocessor directive.
    - If on Windows, it defines a static `std::regex` to match Windows-style newline sequences (`\r\n`).
    - It uses `std::regex_replace` to replace all occurrences of `\r\n` with `\n` in the input string `s`.
    - If not on Windows, it simply returns the input string `s` unchanged.
- **Output**: A `std::string` with Windows-style newlines replaced by Unix-style newlines if on Windows, or the original string if not.


---
### simple\_msg<!-- {{#callable:simple_msg}} -->
The `simple_msg` function creates and returns a `common_chat_msg` object with specified role and content.
- **Inputs**:
    - `role`: A `std::string` representing the role of the message, such as 'user', 'assistant', or 'system'.
    - `content`: A `std::string` representing the content of the message.
- **Control Flow**:
    - A `common_chat_msg` object named `msg` is instantiated.
    - The `role` attribute of `msg` is set to the input `role`.
    - The `content` attribute of `msg` is set to the input `content`.
    - The `msg` object is returned.
- **Output**: A `common_chat_msg` object with its `role` and `content` attributes set to the provided input values.


---
### main<!-- {{#callable:main}} -->
The `main` function initializes a conversation and a set of test cases, applies various chat templates to the conversation, and verifies the output against expected results.
- **Inputs**:
    - `void`: The `main` function does not take any input arguments.
- **Control Flow**:
    - Initialize a vector `conversation` with predefined chat messages between a system, user, and assistant.
    - Define a `TestCase` struct to hold template information and expected outputs for various chat models.
    - Create a vector `test_cases` containing multiple `TestCase` instances, each representing a different chat model template and its expected output.
    - Initialize a vector `formatted_chat` to store formatted chat output and an integer `res` for result status.
    - List all supported chat templates using `llama_chat_builtin_templates` and print them.
    - Test an invalid chat template to ensure error handling works as expected.
    - Iterate over each `test_case`, apply the template to the `conversation` using `llama_chat_apply_template`, and compare the output to the expected result, asserting equality.
    - Convert `conversation` to a vector of `common_chat_msg` and repeat the template application and verification process for Jinja-supported templates.
    - Test `llama_chat_format_single` for formatting system and user messages with different templates, asserting expected outputs.
- **Output**: The function returns an integer `0` to indicate successful execution.
- **Functions called**:
    - [`simple_msg`](#simple_msg)
    - [`normalize_newlines`](#normalize_newlines)


