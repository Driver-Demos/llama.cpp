# Purpose
This C++ source file defines a class `common_chat_msg_parser` that is responsible for parsing chat messages, potentially in a partial state, and handling various components of these messages. The class is designed to work with a specific syntax defined by `common_chat_syntax` and can process input strings to extract structured data, such as JSON objects and tool calls. The parser provides methods to navigate through the input string, manage parsing positions, and handle errors through exceptions. It also includes functionality to consume and process JSON data, with options to handle partial JSON structures and convert specific subtrees into JSON strings. The file includes a custom exception class, [`common_chat_msg_partial_exception`](#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception), which is used to signal errors specific to the parsing process.

The file imports several headers, including `nlohmann/json.hpp` for JSON handling, and other project-specific headers like "chat.h", "json-partial.h", and "regex-partial.h", indicating that it is part of a larger codebase dealing with chat message processing. The class provides a public API with methods to access the parsed results, manipulate the parsing state, and add content or tool calls to the result. This file is likely intended to be part of a library or module that can be integrated into a larger application, providing specialized functionality for parsing and interpreting chat messages in a structured and extensible manner.
# Imports and Dependencies

---
- `chat.h`
- `json-partial.h`
- `regex-partial.h`
- `nlohmann/json.hpp`
- `optional`
- `string`
- `vector`


# Data Structures

---
### common\_chat\_msg\_partial\_exception<!-- {{#data_structure:common_chat_msg_partial_exception}} -->
- **Type**: `class`
- **Description**: The `common_chat_msg_partial_exception` class is a custom exception class that inherits from `std::runtime_error`. It is designed to handle exceptions specifically related to partial chat message parsing errors. The class constructor takes a string message as an argument, which is passed to the base class `std::runtime_error` to provide a descriptive error message when the exception is thrown.
- **Member Functions**:
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)
- **Inherits From**:
    - `std::runtime_error`

**Methods**

---
#### common\_chat\_msg\_partial\_exception::common\_chat\_msg\_partial\_exception<!-- {{#callable:common_chat_msg_partial_exception::common_chat_msg_partial_exception}} -->
The `common_chat_msg_partial_exception` constructor initializes an exception object with a given error message, inheriting from `std::runtime_error`.
- **Inputs**:
    - `message`: A constant reference to a `std::string` that contains the error message to be associated with the exception.
- **Control Flow**:
    - The constructor takes a single string argument `message`.
    - It calls the constructor of the base class `std::runtime_error` with the provided `message`.
- **Output**: An instance of `common_chat_msg_partial_exception` is created, which is a type of `std::runtime_error` with the specified error message.
- **See also**: [`common_chat_msg_partial_exception`](#common_chat_msg_partial_exception)  (Data Structure)



---
### common\_chat\_msg\_parser<!-- {{#data_structure:common_chat_msg_parser}} -->
- **Type**: `class`
- **Members**:
    - `input_`: Stores the input string to be parsed.
    - `is_partial_`: Indicates whether the input is a partial message.
    - `syntax_`: Holds the syntax rules for parsing the chat message.
    - `healing_marker_`: A unique marker used for healing partial JSON.
    - `pos_`: Tracks the current position in the input string.
    - `result_`: Stores the result of the parsing process as a common_chat_msg object.
- **Description**: The `common_chat_msg_parser` class is designed to parse chat messages, handling both complete and partial inputs. It maintains the current parsing position and applies syntax rules to extract meaningful content, reasoning, and tool calls from the input. The class also supports JSON parsing with special handling for partial JSON data, using a healing marker to manage incomplete structures. It provides various methods to manipulate the parsing process, such as moving the position, consuming literals, and handling regex matches.
- **Member Functions**:
    - [`common_chat_msg_parser::common_chat_msg_parser`](chat-parser.cpp.driver.md#common_chat_msg_parsercommon_chat_msg_parser)
    - [`common_chat_msg_parser::str`](chat-parser.cpp.driver.md#common_chat_msg_parserstr)
    - [`common_chat_msg_parser::add_content`](chat-parser.cpp.driver.md#common_chat_msg_parseradd_content)
    - [`common_chat_msg_parser::add_reasoning_content`](chat-parser.cpp.driver.md#common_chat_msg_parseradd_reasoning_content)
    - [`common_chat_msg_parser::add_tool_call`](chat-parser.cpp.driver.md#common_chat_msg_parseradd_tool_call)
    - [`common_chat_msg_parser::add_tool_call`](chat-parser.cpp.driver.md#common_chat_msg_parseradd_tool_call)
    - [`common_chat_msg_parser::add_tool_calls`](chat-parser.cpp.driver.md#common_chat_msg_parseradd_tool_calls)
    - [`common_chat_msg_parser::finish`](chat-parser.cpp.driver.md#common_chat_msg_parserfinish)
    - [`common_chat_msg_parser::consume_spaces`](chat-parser.cpp.driver.md#common_chat_msg_parserconsume_spaces)
    - [`common_chat_msg_parser::try_consume_literal`](chat-parser.cpp.driver.md#common_chat_msg_parsertry_consume_literal)
    - [`common_chat_msg_parser::try_find_literal`](chat-parser.cpp.driver.md#common_chat_msg_parsertry_find_literal)
    - [`common_chat_msg_parser::consume_literal`](chat-parser.cpp.driver.md#common_chat_msg_parserconsume_literal)
    - [`common_chat_msg_parser::try_parse_reasoning`](chat-parser.cpp.driver.md#common_chat_msg_parsertry_parse_reasoning)
    - [`common_chat_msg_parser::consume_rest`](chat-parser.cpp.driver.md#common_chat_msg_parserconsume_rest)
    - [`common_chat_msg_parser::try_find_regex`](chat-parser.cpp.driver.md#common_chat_msg_parsertry_find_regex)
    - [`common_chat_msg_parser::consume_regex`](chat-parser.cpp.driver.md#common_chat_msg_parserconsume_regex)
    - [`common_chat_msg_parser::try_consume_regex`](chat-parser.cpp.driver.md#common_chat_msg_parsertry_consume_regex)
    - [`common_chat_msg_parser::try_consume_json`](chat-parser.cpp.driver.md#common_chat_msg_parsertry_consume_json)
    - [`common_chat_msg_parser::consume_json`](chat-parser.cpp.driver.md#common_chat_msg_parserconsume_json)
    - [`common_chat_msg_parser::consume_json_with_dumped_args`](chat-parser.cpp.driver.md#common_chat_msg_parserconsume_json_with_dumped_args)
    - [`common_chat_msg_parser::try_consume_json_with_dumped_args`](chat-parser.cpp.driver.md#common_chat_msg_parsertry_consume_json_with_dumped_args)
    - [`common_chat_msg_parser::input`](#common_chat_msg_parserinput)
    - [`common_chat_msg_parser::pos`](#common_chat_msg_parserpos)
    - [`common_chat_msg_parser::healing_marker`](#common_chat_msg_parserhealing_marker)
    - [`common_chat_msg_parser::is_partial`](#common_chat_msg_parseris_partial)
    - [`common_chat_msg_parser::result`](#common_chat_msg_parserresult)
    - [`common_chat_msg_parser::syntax`](#common_chat_msg_parsersyntax)
    - [`common_chat_msg_parser::move_to`](#common_chat_msg_parsermove_to)
    - [`common_chat_msg_parser::move_back`](#common_chat_msg_parsermove_back)

**Methods**

---
#### common\_chat\_msg\_parser::input<!-- {{#callable:common_chat_msg_parser::input}} -->
The `input` function returns a constant reference to the `input_` string member of the `common_chat_msg_parser` class.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the `input_` member variable of the class.
- **Output**: A constant reference to the `input_` string member variable.
- **See also**: [`common_chat_msg_parser`](#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::pos<!-- {{#callable:common_chat_msg_parser::pos}} -->
The `pos` function returns the current position index within the input string of the `common_chat_msg_parser` class.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the value of the private member variable `pos_`.
- **Output**: The function outputs a `size_t` representing the current position index within the input string.
- **See also**: [`common_chat_msg_parser`](#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::healing\_marker<!-- {{#callable:common_chat_msg_parser::healing_marker}} -->
The `healing_marker` function returns a reference to the `healing_marker_` string member of the `common_chat_msg_parser` class.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the `healing_marker_` member variable of the class.
- **Output**: A constant reference to a `std::string` representing the `healing_marker_`.
- **See also**: [`common_chat_msg_parser`](#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::is\_partial<!-- {{#callable:common_chat_msg_parser::is_partial}} -->
The `is_partial` function returns a constant reference to the `is_partial_` boolean member variable of the `common_chat_msg_parser` class, indicating whether the parsing process is partial.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the `is_partial_` member variable without any additional logic or conditions.
- **Output**: A constant reference to a boolean value indicating if the parsing is partial.
- **See also**: [`common_chat_msg_parser`](#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::result<!-- {{#callable:common_chat_msg_parser::result}} -->
The `result` function returns a constant reference to the `common_chat_msg` object stored in the `result_` member variable of the `common_chat_msg_parser` class.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter that directly returns the `result_` member variable.
    - It is marked as `const`, indicating it does not modify any member variables of the class.
- **Output**: A constant reference to a `common_chat_msg` object.
- **See also**: [`common_chat_msg_parser`](#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::syntax<!-- {{#callable:common_chat_msg_parser::syntax}} -->
The `syntax` function returns a constant reference to the `common_chat_syntax` object associated with the `common_chat_msg_parser` instance.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the `syntax_` member variable of the `common_chat_msg_parser` class.
- **Output**: A constant reference to the `common_chat_syntax` object.
- **See also**: [`common_chat_msg_parser`](#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::move\_to<!-- {{#callable:common_chat_msg_parser::move_to}} -->
The `move_to` function sets the current position within the input string to a specified position, ensuring the position is valid.
- **Inputs**:
    - `pos`: A size_t value representing the position to move to within the input string.
- **Control Flow**:
    - Check if the provided position `pos` is greater than the size of the input string `input_`.
    - If `pos` is greater than the size of `input_`, throw a `std::runtime_error` with the message "Invalid position!".
    - If the position is valid, set the member variable `pos_` to the provided position `pos`.
- **Output**: The function does not return a value; it modifies the internal state of the object by updating the `pos_` member variable.
- **See also**: [`common_chat_msg_parser`](#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::move\_back<!-- {{#callable:common_chat_msg_parser::move_back}} -->
The `move_back` function adjusts the current position in the input string by moving it backwards by a specified number of characters, ensuring it does not move past the start of the string.
- **Inputs**:
    - `n`: The number of positions to move back from the current position `pos_`.
- **Control Flow**:
    - Check if the current position `pos_` is less than `n`.
    - If `pos_` is less than `n`, throw a `std::runtime_error` with the message "Can't move back that far!".
    - Subtract `n` from `pos_` to move the position backwards.
- **Output**: The function does not return a value; it modifies the member variable `pos_` of the `common_chat_msg_parser` class.
- **See also**: [`common_chat_msg_parser`](#common_chat_msg_parser)  (Data Structure)



---
### find\_regex\_result<!-- {{#data_structure:common_chat_msg_parser::find_regex_result}} -->
- **Type**: `struct`
- **Members**:
    - `prelude`: A string that represents the initial part of the matched content before the regex groups.
    - `groups`: A vector of common_string_range objects representing the captured groups from the regex match.
- **Description**: The `find_regex_result` struct is designed to encapsulate the result of a regex search operation. It contains a `prelude` string, which holds the portion of the input string preceding the matched groups, and a `groups` vector, which stores the ranges of strings captured by the regex groups. This structure is useful for parsing and processing text where regex patterns are used to identify and extract specific segments of the input.


---
### consume\_json\_result<!-- {{#data_structure:common_chat_msg_parser::consume_json_result}} -->
- **Type**: `struct`
- **Members**:
    - `value`: Holds a JSON object using the nlohmann::ordered_json type.
    - `is_partial`: Indicates whether the JSON object is partial or complete.
- **Description**: The `consume_json_result` struct is designed to encapsulate the result of a JSON consumption operation, specifically within the context of parsing potentially partial JSON data. It contains a `value` field, which stores the JSON object, and an `is_partial` boolean flag that signifies whether the JSON data is complete or if it has been truncated or is otherwise incomplete. This struct is useful in scenarios where JSON data may be incrementally parsed or when dealing with incomplete data streams.


