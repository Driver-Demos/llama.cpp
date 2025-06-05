# Purpose
This C++ source code file is designed to test chat handling functionalities, specifically focusing on grammar generation and parsing for tool calling within various templates. It serves a dual purpose: first, as a test suite to ensure the correct functionality of chat message handling, including the generation of message diffs and the parsing of JSON representations of chat messages and tools; second, as a command-line interface (CLI) tool to generate a Markdown summary of the formats of Jinja templates. The file includes several test functions that validate the conversion of chat messages and tools to and from JSON, ensuring compatibility with OpenAI's format, and it also tests the parsing of template outputs to verify that the expected message structures are correctly generated and parsed.

The code is structured around several key components, including the use of the nlohmann::json library for JSON handling, and it defines a series of static functions for reading files, normalizing messages, and asserting equality between expected and actual outputs. The file also defines a main function that either processes Jinja template files provided as command-line arguments or runs a series of predefined tests if no arguments are given. The tests cover a wide range of scenarios, including different message formats and tool call structures, ensuring robust validation of the chat handling logic. The file is not intended to be a standalone executable but rather a test suite that can be run to verify the correctness of the chat handling implementation.
# Imports and Dependencies

---
- `chat.h`
- `../src/unicode.h`
- `../src/llama-grammar.h`
- `nlohmann/json.hpp`
- `fstream`
- `iostream`
- `string`


# Global Variables

---
### special\_function\_tool
- **Type**: `common_chat_tool`
- **Description**: The `special_function_tool` is an instance of the `common_chat_tool` structure, representing a tool with a specific function name, description, and parameters. It is configured with the name "special_function", a description "I'm special", and a JSON schema for its parameters, which includes a required integer argument named "arg1".
- **Use**: This variable is used to define and configure a special function tool that can be utilized in chat applications for specific operations requiring an integer argument.


---
### python\_tool
- **Type**: `common_chat_tool`
- **Description**: The `python_tool` is a global variable of type `common_chat_tool` that represents a tool for executing Python code within a chat environment. It is configured with a name, description, and a JSON schema for its parameters, which specifies that it requires a string type 'code' property to execute Python code.
- **Use**: This variable is used to define and configure a tool that can be called to execute Python code in a chat application.


---
### code\_interpreter\_tool
- **Type**: `common_chat_tool`
- **Description**: The `code_interpreter_tool` is an instance of the `common_chat_tool` structure, representing a tool designed to execute Python code using an IPython interpreter. It is defined with a name, description, and a JSON schema for its parameters, which includes a required 'code' property of type string.
- **Use**: This variable is used to define a tool that can be called to execute Python code within a chat system, allowing for dynamic code execution.


---
### tools
- **Type**: `std::vector<common_chat_tool>`
- **Description**: The `tools` variable is a global vector containing instances of `common_chat_tool`. It is initialized with two specific tools: `special_function_tool` and `python_tool`. Each tool in the vector represents a chat tool with specific functionalities and parameters.
- **Use**: This variable is used to store and manage a collection of chat tools that can be utilized in chat handling and tool calling processes.


---
### llama\_3\_1\_tools
- **Type**: `std::vector<common_chat_tool>`
- **Description**: The `llama_3_1_tools` variable is a global vector containing instances of `common_chat_tool`. It is initialized with two specific tools: `special_function_tool` and `code_interpreter_tool`. Each tool in the vector represents a chat tool with specific functionalities and parameters.
- **Use**: This variable is used to store and manage a collection of chat tools that can be utilized in chat handling and processing within the application.


---
### message\_user
- **Type**: `const common_chat_msg`
- **Description**: The `message_user` variable is a constant instance of the `common_chat_msg` structure, initialized with a role of "user" and a content message of "Hey there!". It has empty vectors for `content_parts` and `tool_calls`, and empty strings for `reasoning_content`, `tool_name`, and `tool_call_id`. This structure is likely used to represent a chat message from a user in a chat handling system.
- **Use**: This variable is used to represent a predefined user message in chat handling tests or operations.


---
### message\_user\_parts
- **Type**: `const common_chat_msg`
- **Description**: The `message_user_parts` variable is a constant instance of the `common_chat_msg` structure, representing a chat message with a role of 'user'. It contains an empty content string and a list of content parts, each with a type 'text' and text values 'Hey' and 'there'. The other fields like `tool_calls`, `reasoning_content`, `tool_name`, and `tool_call_id` are initialized to empty values.
- **Use**: This variable is used to represent a user message in a chat system, specifically breaking down the message into parts for further processing or display.


---
### message\_assist
- **Type**: `const common_chat_msg`
- **Description**: The `message_assist` variable is a constant instance of the `common_chat_msg` structure, initialized using the `simple_assist_msg` function with the content "Hello, world!\nWhat's up?". This structure likely represents a chat message with a role of 'assistant' and contains the specified content.
- **Use**: This variable is used to represent a predefined assistant message in chat handling tests.


---
### message\_assist\_empty
- **Type**: `const common_chat_msg`
- **Description**: The `message_assist_empty` is a global constant variable of type `common_chat_msg` that is initialized using the `simple_assist_msg` function with an empty string as its content. This indicates that the message is intended to represent an empty or default assistant message with no content.
- **Use**: This variable is used to represent an empty assistant message in chat handling scenarios, likely serving as a placeholder or default value.


---
### message\_assist\_thoughts\_unparsed\_deepseek
- **Type**: `common_chat_msg`
- **Description**: The variable `message_assist_thoughts_unparsed_deepseek` is a constant instance of the `common_chat_msg` structure. It is initialized using the `simple_assist_msg` function with a content string that includes a thinking tag and a message: "<think>I'm\nthinking</think>Hello, world!\nWhat's up?". This suggests that the message is intended to represent an assistant's thought process followed by a greeting and a casual inquiry.
- **Use**: This variable is used to represent a chat message from an assistant, including unparsed thoughts and a greeting, likely for testing or demonstration purposes in chat handling scenarios.


---
### message\_assist\_thoughts\_unparsed\_md
- **Type**: `const common_chat_msg`
- **Description**: The variable `message_assist_thoughts_unparsed_md` is a constant instance of the `common_chat_msg` structure. It is initialized using the `simple_assist_msg` function with a content string that includes a thinking tag and a JSON code block. This variable represents a chat message from an assistant with unparsed thoughts in Markdown format.
- **Use**: This variable is used to represent a specific chat message format in tests or chat handling scenarios, particularly for testing the parsing and handling of Markdown content with embedded JSON.


---
### message\_assist\_thoughts\_unparsed\_md\_partial
- **Type**: `const common_chat_msg`
- **Description**: The variable `message_assist_thoughts_unparsed_md_partial` is a constant instance of the `common_chat_msg` structure. It is initialized using the `simple_assist_msg` function with a content string that includes a thinking tag and a JSON code block. This variable represents a chat message from an assistant with unparsed thoughts and a partial Markdown format.
- **Use**: This variable is used to represent a specific type of chat message in tests or chat handling scenarios, particularly for testing the parsing and handling of messages with unparsed thoughts and Markdown content.


---
### message\_assist\_thoughts\_unparsed\_r7b
- **Type**: `common_chat_msg`
- **Description**: The variable `message_assist_thoughts_unparsed_r7b` is a constant instance of the `common_chat_msg` structure. It is initialized using the `simple_assist_msg` function with a content string that includes special markers `<|START_THINKING|>` and `<|END_THINKING|>` to denote a thinking phase, followed by a message content 'Hello, world!\nWhat's up?'. This structure is used to represent a chat message with a role of 'assistant' and includes unparsed thoughts.
- **Use**: This variable is used to represent a chat message from an assistant, including unparsed thoughts, in a chat handling system.


---
### message\_assist\_thoughts
- **Type**: ``common_chat_msg``
- **Description**: The `message_assist_thoughts` variable is a constant instance of the `common_chat_msg` structure, initialized using the `simple_assist_msg` function. It represents a chat message from an assistant with the content "Hello, world!\nWhat's up?" and reasoning content "I'm\nthinking".
- **Use**: This variable is used to represent a predefined assistant message with specific content and reasoning, likely for testing or demonstration purposes in chat handling scenarios.


---
### message\_assist\_thoughts\_unopened\_unparsed
- **Type**: `const common_chat_msg`
- **Description**: The variable `message_assist_thoughts_unopened_unparsed` is a constant instance of the `common_chat_msg` structure. It is initialized using the `simple_assist_msg` function with a content string that includes a partial thinking tag and a message content: "I'm\nthinking</think>Hello, world!\nWhat's up?". This suggests that the message is intended to represent an assistant's thought process that is not fully parsed or opened, as indicated by the unclosed thinking tag.
- **Use**: This variable is used to represent a chat message from an assistant with unparsed thoughts, likely for testing or handling chat message parsing scenarios.


---
### message\_assist\_thoughts\_no\_content
- **Type**: `const common_chat_msg`
- **Description**: The `message_assist_thoughts_no_content` is a constant instance of the `common_chat_msg` structure, initialized using the `simple_assist_msg` function. It represents a chat message from an assistant with no main content but includes reasoning content that states "I'm\nthinking".
- **Use**: This variable is used to represent a chat message where the assistant is in a thinking state without providing any main content.


---
### message\_assist\_call
- **Type**: ``const common_chat_msg``
- **Description**: The `message_assist_call` is a constant instance of the `common_chat_msg` structure, initialized using the `simple_assist_msg` function. It represents a chat message with an empty content and reasoning content, but includes a tool call to a function named "special_function" with a JSON argument `{"arg1": 1}`.
- **Use**: This variable is used to represent a specific type of chat message that involves calling a special function with a predefined argument.


---
### message\_assist\_call\_content
- **Type**: ``const common_chat_msg``
- **Description**: The `message_assist_call_content` is a constant instance of the `common_chat_msg` structure, initialized using the `simple_assist_msg` function. It represents a chat message with the role of an assistant, containing the content "Hello, world!\nWhat's up?" and a tool call to a function named "special_function" with JSON arguments `{"arg1":1}`.
- **Use**: This variable is used to represent a predefined assistant message with specific content and a tool call, likely for testing or demonstration purposes in chat handling scenarios.


---
### message\_assist\_call\_empty\_args
- **Type**: ``common_chat_msg``
- **Description**: The `message_assist_call_empty_args` is a constant instance of the `common_chat_msg` structure, initialized using the `simple_assist_msg` function with empty strings for content and reasoning content, and a tool name of "special_function". This indicates a message from an assistant role with a tool call but no arguments provided.
- **Use**: This variable is used to represent an assistant message that calls a tool named "special_function" without any arguments.


---
### message\_assist\_call\_cutoff\_args
- **Type**: ``const common_chat_msg``
- **Description**: The `message_assist_call_cutoff_args` is a constant instance of the `common_chat_msg` structure, initialized using the `simple_assist_msg` function. It represents a chat message with an incomplete JSON argument string for a tool call to a function named "special_function".
- **Use**: This variable is used to simulate a scenario where the arguments for a tool call are incomplete, likely for testing purposes.


---
### message\_assist\_call\_thoughts
- **Type**: ``common_chat_msg``
- **Description**: The variable `message_assist_call_thoughts` is a constant instance of the `common_chat_msg` structure. It is initialized using the `simple_assist_msg` function with an empty content, a reasoning content of "I'm\nthinking", a tool name "special_function", and a JSON string as arguments.
- **Use**: This variable is used to represent a chat message from an assistant that includes a reasoning content and a tool call with specific arguments.


---
### message\_assist\_call\_thoughts\_unparsed
- **Type**: ``const common_chat_msg``
- **Description**: The variable `message_assist_call_thoughts_unparsed` is a constant instance of the `common_chat_msg` structure. It is initialized using the `simple_assist_msg` function with specific parameters, including a content string that contains a thinking tag, an empty reasoning content, a tool name "special_function", and a JSON string as arguments.
- **Use**: This variable is used to represent a chat message from an assistant that includes unparsed thoughts and a tool call with specific arguments.


---
### message\_assist\_call\_id
- **Type**: ``const common_chat_msg``
- **Description**: The `message_assist_call_id` is a constant instance of the `common_chat_msg` structure, initialized using the `simple_assist_msg` function. It represents a chat message with a tool call to a function named "special_function" and includes a JSON argument `{"arg1":1}` and an ID of "123456789".
- **Use**: This variable is used to represent a specific chat message scenario where a tool call is made with a predefined ID, likely for testing or demonstration purposes.


---
### message\_assist\_call\_idx
- **Type**: `const common_chat_msg`
- **Description**: The `message_assist_call_idx` is a constant instance of the `common_chat_msg` structure, initialized using the `simple_assist_msg` function. It represents a chat message with an empty content and reasoning content, but includes a tool call to a 'special_function' with a JSON argument and an ID of '0'.
- **Use**: This variable is used to represent a specific type of chat message that involves calling a special function with predefined arguments and an ID.


---
### message\_assist\_thoughts\_call\_idx
- **Type**: `const common_chat_msg`
- **Description**: The variable `message_assist_thoughts_call_idx` is a constant instance of the `common_chat_msg` structure. It is initialized using the `simple_assist_msg` function with specific parameters, including an empty content string, a reasoning content of "I'm\nthinking", a tool name "special_function", arguments in JSON format "{\"arg1\": 1}", and an ID of "0".
- **Use**: This variable is used to represent a chat message from an assistant that includes a reasoning statement and a tool call with specific arguments and an ID.


---
### message\_assist\_call\_python
- **Type**: ``common_chat_msg``
- **Description**: The `message_assist_call_python` is a constant instance of the `common_chat_msg` structure, initialized using the `simple_assist_msg` function. It represents a chat message from an assistant that includes a tool call to a Python interpreter with the code `print('hey')`. The message does not contain any direct content or reasoning content, but it specifies a tool call with the name 'python' and the argument containing the Python code to execute.
- **Use**: This variable is used to represent a chat message that triggers a Python code execution within a chat handling system.


---
### message\_assist\_call\_python\_lines
- **Type**: `const common_chat_msg`
- **Description**: The variable `message_assist_call_python_lines` is a constant instance of the `common_chat_msg` structure. It is initialized using the `simple_assist_msg` function with parameters that specify an empty content and reasoning content, a tool name of "python", and a JSON string representing a Python code snippet as arguments.
- **Use**: This variable is used to represent a chat message that includes a call to a Python tool with a specific code snippet, likely for testing or demonstration purposes.


---
### message\_assist\_call\_python\_lines\_unclosed
- **Type**: ``const common_chat_msg``
- **Description**: The variable `message_assist_call_python_lines_unclosed` is a constant instance of the `common_chat_msg` structure. It is initialized using the `simple_assist_msg` function with parameters that specify an empty content and reasoning content, a tool name of "python", and a JSON string representing a Python code snippet that is not properly closed.
- **Use**: This variable is used to represent a chat message that includes a call to a Python tool with an incomplete code snippet, likely for testing purposes.


---
### message\_assist\_call\_code\_interpreter
- **Type**: `const common_chat_msg`
- **Description**: The variable `message_assist_call_code_interpreter` is a constant instance of the `common_chat_msg` structure. It is initialized using the `simple_assist_msg` function with parameters that specify an empty content, an empty reasoning content, a tool name of "code_interpreter", and a JSON string representing a Python code snippet to execute.
- **Use**: This variable is used to represent a chat message where the assistant is calling a code interpreter tool to execute a specific Python code snippet.


# Data Structures

---
### delta\_data<!-- {{#data_structure:delta_data}} -->
- **Type**: `struct`
- **Members**:
    - `delta`: A string representing the delta or difference in chat content.
    - `params`: An instance of common_chat_params that holds parameters related to chat operations.
- **Description**: The `delta_data` struct is designed to encapsulate information about changes or differences in chat content, specifically within the context of chat handling and grammar parsing. It contains a `delta` string that represents the difference in chat content and a `params` object of type `common_chat_params` that holds various parameters related to the chat operation. This struct is likely used in scenarios where chat content is being compared or modified, and the differences need to be tracked and managed.


# Functions

---
### operator<<<!-- {{#callable:operator<<}} -->
The `operator<<` function serializes a `common_chat_msg` object into a human-readable format and writes it to an output stream.
- **Inputs**:
    - `os`: An output stream (std::ostream) where the serialized message will be written.
    - `msg`: A `common_chat_msg` object containing chat message details to be serialized.
- **Control Flow**:
    - Begin by writing the opening brace '{' to the output stream.
    - Write the `role` of the message followed by a semicolon to the output stream.
    - Write the `content` of the message followed by a semicolon to the output stream.
    - Write the opening bracket for `content_parts` and iterate over each part in `msg.content_parts`.
    - For each `content_part`, write its `type` and `text` to the output stream, followed by a comma.
    - Close the `content_parts` array with a bracket and semicolon.
    - Write the `reasoning_content` of the message followed by a semicolon to the output stream.
    - Write the opening bracket for `tool_calls` and iterate over each `tool_call` in `msg.tool_calls`.
    - For each `tool_call`, write its `name`, `arguments`, and `id` to the output stream, followed by a comma.
    - Close the `tool_calls` array with a bracket.
    - Write the closing brace '}' to the output stream.
    - Return the output stream.
- **Output**: The function returns the modified output stream `os` containing the serialized representation of the `common_chat_msg` object.


---
### equals<!-- {{#callable:equals}} -->
The `equals` function template specialization for `common_chat_msg` compares two `common_chat_msg` objects for equality after normalizing them.
- **Inputs**:
    - `expected`: A `common_chat_msg` object representing the expected message to compare.
    - `actual`: A `common_chat_msg` object representing the actual message to compare.
- **Control Flow**:
    - The function calls the [`normalize`](#normalize) function on both the `expected` and `actual` `common_chat_msg` objects.
    - It then compares the normalized versions of these objects using the equality operator `==`.
- **Output**: A boolean value indicating whether the normalized `expected` and `actual` `common_chat_msg` objects are equal.
- **Functions called**:
    - [`normalize`](#normalize)


---
### normalize<!-- {{#callable:normalize}} -->
The `normalize` function processes a `common_chat_msg` object by attempting to parse and reformat the JSON arguments of each tool call within the message.
- **Inputs**:
    - `msg`: A constant reference to a `common_chat_msg` object that contains the message to be normalized.
- **Control Flow**:
    - Create a copy of the input message `msg` named `normalized`.
    - Iterate over each `tool_call` in `normalized.tool_calls`.
    - For each `tool_call`, attempt to parse `tool_call.arguments` as JSON and reassign it to `tool_call.arguments` in a formatted string representation.
    - If parsing fails, catch the exception and do nothing.
    - Return the `normalized` message.
- **Output**: Returns a `common_chat_msg` object with potentially reformatted JSON arguments in its tool calls.


---
### assert\_equals<!-- {{#callable:assert_equals}} -->
The `assert_equals` function checks if two values are equal and throws a runtime error if they are not, while also printing the expected and actual values to the standard error stream.
- **Inputs**:
    - `expected`: The expected value of type T that the actual value is compared against.
    - `actual`: The actual value of type T that is compared to the expected value.
- **Control Flow**:
    - The function calls the [`equals`](#equals) function to compare the `expected` and `actual` values.
    - If the [`equals`](#equals) function returns false, indicating the values are not equal, the function prints the expected and actual values to the standard error stream.
    - The function then flushes the error stream to ensure all output is written immediately.
    - Finally, the function throws a `std::runtime_error` with the message "Test failed".
- **Output**: The function does not return a value; it either completes successfully if the values are equal or throws an exception if they are not.
- **Functions called**:
    - [`equals`](#equals)


---
### read\_file<!-- {{#callable:read_file}} -->
The `read_file` function reads the contents of a file specified by its path and returns it as a string.
- **Inputs**:
    - `path`: A constant reference to a string representing the file path to be read.
- **Control Flow**:
    - Prints a message to standard error indicating the file being read.
    - Attempts to open the file at the specified path in binary mode using an ifstream object.
    - If the file cannot be opened, attempts to open the file in a parent directory by prepending '../' to the path.
    - If the file still cannot be opened, throws a runtime error indicating failure to open the file.
    - Seeks to the end of the file to determine its size.
    - Resets the file position to the beginning of the file.
    - Resizes a string to accommodate the file's size.
    - Reads the file's contents into the string.
    - Returns the string containing the file's contents.
- **Output**: A string containing the contents of the file.


---
### read\_templates<!-- {{#callable:read_templates}} -->
The `read_templates` function reads a file from a specified path and initializes a `common_chat_templates` object with its contents.
- **Inputs**:
    - `path`: A string representing the file path from which to read the template data.
- **Control Flow**:
    - The function calls [`read_file`](#read_file) with the provided `path` to read the file contents into a string.
    - It then calls `common_chat_templates_init` with `nullptr` as the model and the file contents as arguments to initialize a `common_chat_templates` object.
    - Finally, it returns a `common_chat_templates_ptr` pointing to the initialized object.
- **Output**: A `common_chat_templates_ptr` which is a pointer to the initialized `common_chat_templates` object.
- **Functions called**:
    - [`read_file`](#read_file)


---
### build\_grammar<!-- {{#callable:build_grammar}} -->
The `build_grammar` function initializes and returns a unique pointer to a `llama_grammar` object using a given grammar string.
- **Inputs**:
    - `grammar_str`: A string representing the grammar to be used for initializing the `llama_grammar` object.
- **Control Flow**:
    - The function calls `llama_grammar_init_impl` with the provided `grammar_str` and other default parameters to initialize a `llama_grammar` object.
    - It then wraps the resulting `llama_grammar` pointer in a `std::unique_ptr` and returns it.
- **Output**: A `std::unique_ptr<llama_grammar>` that manages the lifetime of the `llama_grammar` object initialized with the provided grammar string.


---
### match\_string<!-- {{#callable:match_string}} -->
The `match_string` function checks if a given input string matches a specified grammar using Unicode code points.
- **Inputs**:
    - `input`: A constant reference to a `std::string` representing the input string to be matched against the grammar.
    - `grammar`: A pointer to a `llama_grammar` object that defines the grammar rules to match the input string against.
- **Control Flow**:
    - Convert the input string to a sequence of Unicode code points using `unicode_cpts_from_utf8`.
    - Retrieve the current stacks from the grammar using `llama_grammar_get_stacks`.
    - Iterate over each Unicode code point in the sequence.
    - For each code point, call `llama_grammar_accept` to process it with the grammar.
    - If the current stacks are empty after processing a code point, return `false` indicating a match failure.
    - After processing all code points, check if any stack is empty using `std::any_of`.
    - If any stack is empty, return `true` indicating the grammar has been completed successfully.
    - If no stack is empty, return `false` indicating the grammar was not completed.
- **Output**: A boolean value indicating whether the input string matches the grammar (`true` if matched, `false` otherwise).


---
### renormalize\_json<!-- {{#callable:renormalize_json}} -->
The `renormalize_json` function attempts to parse a JSON string and return it in a normalized format, or returns the original string if parsing fails.
- **Inputs**:
    - `json_str`: A string containing JSON data that needs to be parsed and normalized.
- **Control Flow**:
    - The function tries to parse the input JSON string using `json::parse` from the nlohmann::json library.
    - If parsing is successful, it returns the parsed JSON object as a string using `json_obj.dump()`.
    - If an exception is thrown during parsing, it catches the exception, logs an error message to `std::cerr`, and returns the original JSON string.
- **Output**: A string representing the normalized JSON if parsing is successful, or the original JSON string if parsing fails.


---
### assert\_msg\_equals<!-- {{#callable:assert_msg_equals}} -->
The `assert_msg_equals` function verifies that two `common_chat_msg` objects are equivalent by comparing their roles, content, content parts, reasoning content, and tool calls.
- **Inputs**:
    - `expected`: A `common_chat_msg` object representing the expected message to compare against.
    - `actual`: A `common_chat_msg` object representing the actual message to be compared.
- **Control Flow**:
    - The function begins by asserting that the roles of the expected and actual messages are equal using [`assert_equals`](#assert_equals).
    - It then asserts that the content of both messages is equal.
    - The function checks that the sizes of the `content_parts` vectors in both messages are equal.
    - It iterates over each part in the `content_parts` vectors, asserting that the type and text of each part are equal.
    - The function asserts that the `reasoning_content` of both messages is equal.
    - It checks that the sizes of the `tool_calls` vectors in both messages are equal.
    - The function iterates over each tool call, asserting that the name, renormalized arguments, and id of each tool call are equal.
- **Output**: The function does not return a value; it throws a runtime error if any assertion fails, indicating that the expected and actual messages are not equivalent.
- **Functions called**:
    - [`assert_equals`](#assert_equals)
    - [`renormalize_json`](#renormalize_json)


---
### init\_delta<!-- {{#callable:init_delta}} -->
The `init_delta` function initializes a `delta_data` structure by applying chat templates to user and delta messages, computing the difference between the resulting prompts, and stripping specified end tokens from the delta.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` structure containing the templates to be applied.
    - `end_tokens`: A vector of strings representing tokens that should be stripped from the end of the delta.
    - `user_message`: A `common_chat_msg` structure representing the user's message.
    - `delta_message`: A `common_chat_msg` structure representing the delta message to be processed.
    - `tools`: A vector of `common_chat_tool` structures representing the tools available for use in the chat.
    - `tool_choice`: A `common_chat_tool_choice` enumeration value indicating the tool choice strategy.
- **Control Flow**:
    - Initialize a `common_chat_templates_inputs` structure and set `parallel_tool_calls` to true.
    - Add the `user_message` to the `messages` vector of the inputs and set the `tools` and `tool_choice` fields.
    - Apply the templates to the inputs to get `params_prefix`.
    - Add the `delta_message` to the `messages` vector and set `add_generation_prompt` to false.
    - Apply the templates again to get `params_full`.
    - Extract the `prompt` fields from `params_prefix` and `params_full` as `prefix` and `full`, respectively.
    - Check if `full` is the same as `prefix` and throw a runtime error if they are identical.
    - Calculate the `common_prefix_length` by iterating over `prefix` and `full` until a difference is found, skipping over '<' characters.
    - Extract the `delta` from `full` starting from `common_prefix_length`.
    - Iterate over `end_tokens` and remove the last occurrence of any token found in `delta`.
    - Return a `delta_data` structure containing the `delta` and `params_full`.
- **Output**: A `delta_data` structure containing the computed `delta` string and the `params_full` from the template application.


---
### test\_templates<!-- {{#callable:test_templates}} -->
The `test_templates` function tests chat handling, including grammar generation and parsing for tool calling, using various templates and expected outputs.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` structure representing the chat templates to be tested.
    - `end_tokens`: A vector of strings representing the end tokens to be stripped from the generated delta.
    - `test_message`: A `common_chat_msg` object representing the message to be tested against the generated delta.
    - `tools`: An optional vector of `common_chat_tool` objects representing the tools available for use in the chat; defaults to an empty vector.
    - `expected_delta`: An optional string representing the expected delta output; defaults to an empty string.
    - `expect_grammar_triggered`: A boolean indicating whether grammar triggering is expected; defaults to true.
    - `test_grammar_if_triggered`: A boolean indicating whether to test the grammar if it is triggered; defaults to true.
    - `reasoning_format`: A `common_reasoning_format` enum value representing the reasoning format to be used; defaults to `COMMON_REASONING_FORMAT_NONE`.
- **Control Flow**:
    - Initialize a `common_chat_msg` object `user_message` with role 'user' and content 'Hello, world!'.
    - Iterate over two tool choices: `COMMON_CHAT_TOOL_CHOICE_AUTO` and `COMMON_CHAT_TOOL_CHOICE_REQUIRED`.
    - For each tool choice, call [`init_delta`](#init_delta) to generate a delta and parameters based on the templates, end tokens, user message, test message, tools, and tool choice.
    - If `expected_delta` is not empty, assert that it matches the generated delta.
    - If `expect_grammar_triggered` is true, parse the delta using `common_chat_parse` and assert that the parsed message matches `test_message`.
    - If `test_message` contains tool calls, assert that the generated parameters contain grammar.
    - If the parameters contain grammar, build the grammar using [`build_grammar`](#build_grammar) and check for grammar triggers in the delta.
    - For each grammar trigger, find its position in the delta and determine the earliest trigger position.
    - If a grammar trigger is found, constrain the delta to start from the earliest trigger position and set `grammar_triggered` to true.
    - If `grammar_lazy` is true, assert that `expect_grammar_triggered` matches `grammar_triggered`.
    - If `grammar_triggered` is true and `test_grammar_if_triggered` is true, assert that the constrained delta matches the grammar using [`match_string`](#match_string).
- **Output**: The function does not return a value; it performs assertions to validate the behavior of the chat templates and throws runtime errors if any assertions fail.
- **Functions called**:
    - [`init_delta`](#init_delta)
    - [`assert_equals`](#assert_equals)
    - [`assert_msg_equals`](#assert_msg_equals)
    - [`build_grammar`](#build_grammar)
    - [`match_string`](#match_string)


---
### simple\_assist\_msg<!-- {{#callable:simple_assist_msg}} -->
The `simple_assist_msg` function creates and returns a `common_chat_msg` object with specified content, reasoning content, and optional tool call information.
- **Inputs**:
    - `content`: A string representing the main content of the message.
    - `reasoning_content`: An optional string representing additional reasoning content for the message, defaulting to an empty string.
    - `tool_name`: An optional string representing the name of a tool to be called, defaulting to an empty string.
    - `arguments`: An optional string representing the arguments for the tool call, defaulting to an empty string.
    - `id`: An optional string representing the ID for the tool call, defaulting to an empty string.
- **Control Flow**:
    - Initialize a `common_chat_msg` object named `msg`.
    - Set the `role` of `msg` to "assistant".
    - Assign the `content` parameter to `msg.content`.
    - Assign the `reasoning_content` parameter to `msg.reasoning_content`.
    - Check if `tool_name` is not empty; if true, append a tool call with `tool_name`, `arguments`, and `id` to `msg.tool_calls`.
    - Return the `msg` object.
- **Output**: A `common_chat_msg` object with the specified role, content, reasoning content, and optional tool call information.


---
### test\_msgs\_oaicompat\_json\_conversion<!-- {{#callable:test_msgs_oaicompat_json_conversion}} -->
The function `test_msgs_oaicompat_json_conversion` tests the conversion of chat messages to and from a JSON format compatible with OpenAI's API.
- **Inputs**: None
- **Control Flow**:
    - Prints the function name using `printf`.
    - Initializes a vector `msgs` with various `common_chat_msg` objects representing different chat message scenarios.
    - Iterates over each message in `msgs`, converts it to JSON using `common_chat_msgs_to_json_oaicompat`, parses it back to a message using `common_chat_msgs_parse_oaicompat`, and asserts that the original and parsed messages are equal using [`assert_msg_equals`](#assert_msg_equals).
    - Performs specific JSON conversion and parsing tests for `message_user_parts` and `message_assist_call_python`, asserting the expected JSON output.
    - Parses a JSON string representing an assistant message with empty tool calls and asserts the parsed message properties.
    - Attempts to parse a JSON string missing the 'content' field, expecting an exception, and verifies the exception message.
- **Output**: The function does not return any value; it performs assertions to validate the correctness of JSON conversion and parsing.
- **Functions called**:
    - [`assert_equals`](#assert_equals)
    - [`assert_msg_equals`](#assert_msg_equals)


---
### test\_tools\_oaicompat\_json\_conversion<!-- {{#callable:test_tools_oaicompat_json_conversion}} -->
The function `test_tools_oaicompat_json_conversion` tests the conversion of `common_chat_tool` objects to and from a JSON format compatible with OpenAI's specifications.
- **Inputs**: None
- **Control Flow**:
    - Prints the function name using `printf`.
    - Initializes a vector `tools` with three `common_chat_tool` objects: `special_function_tool`, `python_tool`, and `code_interpreter_tool`.
    - Iterates over each `tool` in the `tools` vector.
    - Converts each `tool` to a JSON object using `common_chat_tools_to_json_oaicompat`.
    - Parses the JSON back into a `common_chat_tool` object using `common_chat_tools_parse_oaicompat`.
    - Asserts that the parsed tool has the same name, description, and parameters as the original tool.
    - Asserts that the JSON conversion of `special_function_tool` matches a predefined JSON string.
- **Output**: The function does not return any value; it performs assertions to validate the JSON conversion process.
- **Functions called**:
    - [`assert_equals`](#assert_equals)


---
### test\_template\_output\_parsers<!-- {{#callable:test_template_output_parsers}} -->
The `test_template_output_parsers` function tests the parsing and application of various chat templates to ensure correct format handling and message parsing.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Prints the function name using `printf`.
    - Initializes three `common_chat_templates_inputs` structures with different configurations of messages and tools.
    - Reads and applies various chat templates from specified file paths using [`read_templates`](#read_templates) and `common_chat_templates_apply`.
    - Asserts the expected format of the applied templates using [`assert_equals`](#assert_equals).
    - Parses messages using `common_chat_parse` and asserts the expected message content using [`assert_msg_equals`](#assert_msg_equals).
    - Tests templates with [`test_templates`](#test_templates) to ensure correct grammar and message parsing.
    - Iterates over different template files and performs similar operations to validate each template's behavior.
- **Output**: The function does not return any value; it performs assertions to validate template parsing and application.
- **Functions called**:
    - [`read_templates`](#read_templates)
    - [`assert_equals`](#assert_equals)
    - [`assert_msg_equals`](#assert_msg_equals)
    - [`test_templates`](#test_templates)
    - [`simple_assist_msg`](#simple_assist_msg)


---
### test\_msg\_diffs\_compute<!-- {{#callable:test_msg_diffs_compute}} -->
The `test_msg_diffs_compute` function tests the `compute_diffs` method of the `common_chat_msg_diff` class by asserting that the computed differences between various pairs of `common_chat_msg` objects match expected results.
- **Inputs**: None
- **Control Flow**:
    - Prints the function name using `printf`.
    - Defines two `common_chat_msg` objects, `msg1` and `msg2`, with different content and computes their difference using `common_chat_msg_diff::compute_diffs`.
    - Asserts that the computed difference matches the expected `common_chat_msg_diff` object with [`assert_equals`](#assert_equals).
    - Repeats the process for different pairs of `common_chat_msg` objects, including those with `tool_calls`, to test various scenarios of message differences.
- **Output**: The function does not return any value; it performs assertions to validate the correctness of the `compute_diffs` method.
- **Functions called**:
    - [`assert_equals`](#assert_equals)


---
### main<!-- {{#callable:main}} -->
The `main` function either processes Jinja template files to output their formats or runs a series of chat-related tests based on the command-line arguments provided.
- **Inputs**:
    - `argc`: An integer representing the number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - If the program is not running on Windows and more than one command-line argument is provided, it enters a loop to process each argument as a file path.
    - For each file path, it checks if the file has a ".jinja" extension; if not, it skips the file.
    - If the file is a Jinja template, it reads the template, extracts the file name, applies the template with predefined inputs, and prints the template name and its format.
    - If an exception occurs during file processing, it prints an error message.
    - If no command-line arguments are provided or the program is running on Windows, it runs a series of predefined test functions.
    - After running the tests, it prints a success message indicating all tests passed.
- **Output**: The function returns 0 to indicate successful execution.
- **Functions called**:
    - [`read_templates`](#read_templates)
    - [`test_msg_diffs_compute`](#test_msg_diffs_compute)
    - [`test_msgs_oaicompat_json_conversion`](#test_msgs_oaicompat_json_conversion)
    - [`test_tools_oaicompat_json_conversion`](#test_tools_oaicompat_json_conversion)
    - [`test_template_output_parsers`](#test_template_output_parsers)


