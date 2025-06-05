# Purpose
The provided C++ source code defines a class [`common_chat_msg_parser`](#common_chat_msg_parsercommon_chat_msg_parser) that is responsible for parsing chat messages with a specific syntax. This class is designed to handle both complete and partial message inputs, as indicated by the `is_partial` parameter. The parser processes the input string to extract various components such as content, reasoning content, and tool calls, which are encapsulated in a result structure. The class provides methods to consume and parse literals, spaces, and JSON data, and it uses regular expressions to identify and extract specific patterns from the input. The code also includes mechanisms to handle incomplete data gracefully, using optional return types and exceptions to manage parsing errors.

The [`common_chat_msg_parser`](#common_chat_msg_parsercommon_chat_msg_parser) class is part of a broader system, as suggested by the inclusion of headers like "chat-parser.h", "common.h", and "regex-partial.h". It appears to be a specialized component focused on parsing chat messages, likely for a chat application or a chatbot system. The class provides a public API for adding content and tool calls, consuming JSON, and handling reasoning content, making it a versatile tool for processing structured chat data. The use of JSON and regular expressions indicates that the parser is designed to handle complex message formats, and the presence of logging (commented out) suggests that debugging and monitoring are important aspects of its operation.
# Imports and Dependencies

---
- `chat-parser.h`
- `common.h`
- `log.h`
- `regex-partial.h`
- `optional`
- `stdexcept`
- `string`
- `vector`


# Data Structures

---
### common\_chat\_msg\_parser<!-- {{#data_structure:common_chat_msg_parser}} -->
- **Description**: [See definition](chat-parser.h.driver.md#common_chat_msg_parser)
- **Member Functions**:
    - [`common_chat_msg_parser::common_chat_msg_parser`](#common_chat_msg_parsercommon_chat_msg_parser)
    - [`common_chat_msg_parser::str`](#common_chat_msg_parserstr)
    - [`common_chat_msg_parser::add_content`](#common_chat_msg_parseradd_content)
    - [`common_chat_msg_parser::add_reasoning_content`](#common_chat_msg_parseradd_reasoning_content)
    - [`common_chat_msg_parser::add_tool_call`](#common_chat_msg_parseradd_tool_call)
    - [`common_chat_msg_parser::add_tool_call`](#common_chat_msg_parseradd_tool_call)
    - [`common_chat_msg_parser::add_tool_calls`](#common_chat_msg_parseradd_tool_calls)
    - [`common_chat_msg_parser::finish`](#common_chat_msg_parserfinish)
    - [`common_chat_msg_parser::consume_spaces`](#common_chat_msg_parserconsume_spaces)
    - [`common_chat_msg_parser::try_consume_literal`](#common_chat_msg_parsertry_consume_literal)
    - [`common_chat_msg_parser::try_find_literal`](#common_chat_msg_parsertry_find_literal)
    - [`common_chat_msg_parser::consume_literal`](#common_chat_msg_parserconsume_literal)
    - [`common_chat_msg_parser::try_parse_reasoning`](#common_chat_msg_parsertry_parse_reasoning)
    - [`common_chat_msg_parser::consume_rest`](#common_chat_msg_parserconsume_rest)
    - [`common_chat_msg_parser::try_find_regex`](#common_chat_msg_parsertry_find_regex)
    - [`common_chat_msg_parser::consume_regex`](#common_chat_msg_parserconsume_regex)
    - [`common_chat_msg_parser::try_consume_regex`](#common_chat_msg_parsertry_consume_regex)
    - [`common_chat_msg_parser::try_consume_json`](#common_chat_msg_parsertry_consume_json)
    - [`common_chat_msg_parser::consume_json`](#common_chat_msg_parserconsume_json)
    - [`common_chat_msg_parser::consume_json_with_dumped_args`](#common_chat_msg_parserconsume_json_with_dumped_args)
    - [`common_chat_msg_parser::try_consume_json_with_dumped_args`](#common_chat_msg_parsertry_consume_json_with_dumped_args)
    - [`common_chat_msg_parser::input`](chat-parser.h.driver.md#common_chat_msg_parserinput)
    - [`common_chat_msg_parser::pos`](chat-parser.h.driver.md#common_chat_msg_parserpos)
    - [`common_chat_msg_parser::healing_marker`](chat-parser.h.driver.md#common_chat_msg_parserhealing_marker)
    - [`common_chat_msg_parser::is_partial`](chat-parser.h.driver.md#common_chat_msg_parseris_partial)
    - [`common_chat_msg_parser::result`](chat-parser.h.driver.md#common_chat_msg_parserresult)
    - [`common_chat_msg_parser::syntax`](chat-parser.h.driver.md#common_chat_msg_parsersyntax)
    - [`common_chat_msg_parser::move_to`](chat-parser.h.driver.md#common_chat_msg_parsermove_to)
    - [`common_chat_msg_parser::move_back`](chat-parser.h.driver.md#common_chat_msg_parsermove_back)

**Methods**

---
#### common\_chat\_msg\_parser::common\_chat\_msg\_parser<!-- {{#callable:common_chat_msg_parser::common_chat_msg_parser}} -->
The `common_chat_msg_parser` constructor initializes a parser object for processing chat messages, setting up initial states and generating a unique healing marker not present in the input string.
- **Inputs**:
    - `input`: A constant reference to a `std::string` representing the input chat message to be parsed.
    - `is_partial`: A boolean indicating whether the input message is partial or complete.
    - `syntax`: A constant reference to a `common_chat_syntax` object that defines the syntax rules for parsing the chat message.
- **Control Flow**:
    - Initialize the member variables `input_`, `is_partial_`, and `syntax_` with the provided arguments.
    - Set the `role` of the `result_` object to "assistant".
    - Enter a loop to generate a random string `id` using `std::rand()` and convert it to a string.
    - Check if the generated `id` is not found in the `input` string.
    - If the `id` is not found, assign it to `healing_marker_` and break the loop.
- **Output**: The constructor does not return a value, as it is responsible for initializing the `common_chat_msg_parser` object.
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::str<!-- {{#callable:common_chat_msg_parser::str}} -->
The `str` function extracts a substring from the `input_` string of the `common_chat_msg_parser` class based on the specified range.
- **Inputs**:
    - `rng`: A `common_string_range` object that specifies the beginning and end indices for the substring extraction.
- **Control Flow**:
    - The function asserts that the beginning index is less than or equal to the end index using `GGML_ASSERT`.
    - It returns a substring of `input_` starting from `rng.begin` and spanning `rng.end - rng.begin` characters.
- **Output**: A `std::string` representing the extracted substring from the `input_`.
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::add\_content<!-- {{#callable:common_chat_msg_parser::add_content}} -->
The `add_content` function appends a given string to the `content` field of the `result_` member within the `common_chat_msg_parser` class.
- **Inputs**:
    - `content`: A constant reference to a `std::string` that represents the content to be appended to the `result_.content` field.
- **Control Flow**:
    - The function takes a single input parameter, `content`, which is a constant reference to a `std::string`.
    - It appends the value of `content` to the `content` field of the `result_` member, which is an instance of `common_chat_msg`.
- **Output**: This function does not return any value; it modifies the `result_.content` field in place.
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::add\_reasoning\_content<!-- {{#callable:common_chat_msg_parser::add_reasoning_content}} -->
The `add_reasoning_content` function appends a given string to the `reasoning_content` field of the `result_` object within the `common_chat_msg_parser` class.
- **Inputs**:
    - `reasoning_content`: A constant reference to a string that contains the reasoning content to be appended.
- **Control Flow**:
    - The function takes a single input parameter, `reasoning_content`, which is a string reference.
    - It appends the `reasoning_content` to the `reasoning_content` field of the `result_` object, which is an instance of `common_chat_msg`.
- **Output**: This function does not return any value; it modifies the `result_` object in place.
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::add\_tool\_call<!-- {{#callable:common_chat_msg_parser::add_tool_call}} -->
The `add_tool_call` function adds a tool call to the result if the tool call's name is not empty.
- **Inputs**:
    - `name`: A string representing the name of the tool call.
    - `id`: A string representing the identifier of the tool call.
    - `arguments`: A string representing the arguments for the tool call.
- **Control Flow**:
    - Check if the `name` is empty; if so, return `false`.
    - Create a `common_chat_tool_call` object and set its `name`, `id`, and `arguments` fields.
    - Add the `tool_call` object to the `result_.tool_calls` vector.
    - Return `true` to indicate the tool call was successfully added.
- **Output**: A boolean value indicating whether the tool call was successfully added (true) or not (false).
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::add\_tool\_calls<!-- {{#callable:common_chat_msg_parser::add_tool_calls}} -->
The `add_tool_calls` function iterates over a JSON array and attempts to add each item as a tool call, returning false if any addition fails.
- **Inputs**:
    - `arr`: A JSON array containing tool call objects, each with "name", "id", and "arguments" fields.
- **Control Flow**:
    - Iterate over each item in the JSON array `arr`.
    - For each item, call the [`add_tool_call`](#common_chat_msg_parseradd_tool_call) method to attempt adding the tool call.
    - If [`add_tool_call`](#common_chat_msg_parseradd_tool_call) returns false for any item, immediately return false.
    - If all items are successfully added, return true.
- **Output**: A boolean value indicating whether all tool calls were successfully added (true) or if any failed (false).
- **Functions called**:
    - [`common_chat_msg_parser::add_tool_call`](#common_chat_msg_parseradd_tool_call)
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::finish<!-- {{#callable:common_chat_msg_parser::finish}} -->
The `finish` method checks if the parsing process has reached the end of the input string when the parser is not in a partial state, and throws an exception if there is unexpected content remaining.
- **Inputs**: None
- **Control Flow**:
    - Check if the parser is not in a partial state (`!is_partial_`).
    - Check if the current position (`pos_`) is not equal to the size of the input string (`input_.size()`).
    - If both conditions are true, throw a `std::runtime_error` indicating unexpected content at the end of the input.
- **Output**: The function does not return any value; it throws an exception if there is unexpected content at the end of the input.
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::consume\_spaces<!-- {{#callable:common_chat_msg_parser::consume_spaces}} -->
The `consume_spaces` function advances the parser's position past any whitespace characters in the input string and returns whether any spaces were consumed.
- **Inputs**:
    - `none`: This function does not take any input parameters.
- **Control Flow**:
    - Initialize `length` with the size of the input string `input_`.
    - Set `consumed` to `false` to track if any spaces are consumed.
    - Enter a `while` loop that continues as long as `pos_` is less than `length` and the current character at `pos_` is a whitespace character.
    - Inside the loop, increment `pos_` to move past the whitespace character and set `consumed` to `true`.
    - Exit the loop when a non-whitespace character is encountered or the end of the string is reached.
    - Return the value of `consumed` indicating if any spaces were consumed.
- **Output**: A boolean value indicating whether any whitespace characters were consumed from the current position in the input string.
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::try\_consume\_literal<!-- {{#callable:common_chat_msg_parser::try_consume_literal}} -->
The `try_consume_literal` function attempts to match a given literal string with the current position in the input string and advances the position if successful.
- **Inputs**:
    - `literal`: A constant reference to a `std::string` representing the literal string to be matched against the input.
- **Control Flow**:
    - Initialize a local variable `pos` with the current position `pos_`.
    - Iterate over each character in the `literal` string.
    - Check if the current position `pos` is beyond the end of the input string; if so, return `false`.
    - Compare the character at the current position in the input string with the corresponding character in the `literal`; if they do not match, return `false`.
    - Increment the position `pos` after a successful character match.
    - After successfully matching all characters, update `pos_` to the new position `pos`.
    - Return `true` to indicate the literal was successfully matched and consumed.
- **Output**: Returns a `bool` indicating whether the literal was successfully matched and consumed from the input string.
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::try\_find\_literal<!-- {{#callable:common_chat_msg_parser::try_find_literal}} -->
The `try_find_literal` function attempts to locate a specified literal string within the input string of the `common_chat_msg_parser` class, returning a result with the prelude and matched groups if found.
- **Inputs**:
    - `literal`: A constant reference to a `std::string` representing the literal string to be searched for within the input.
- **Control Flow**:
    - The function searches for the `literal` starting from the current position `pos_` in the `input_` string.
    - If the `literal` is found, it creates a `find_regex_result` object, sets the `prelude` to the substring from `pos_` to the found index, and adds the range of the found literal to the `groups`.
    - The position `pos_` is updated to the end of the found literal, and the result is returned.
    - If the `literal` is not found and `is_partial_` is true, it attempts a partial search using `string_find_partial_stop`.
    - If a partial match is found, it creates a `find_regex_result` object similar to a full match, updates `pos_`, and returns the result.
    - If no match is found, the function returns `std::nullopt`.
- **Output**: An `std::optional<find_regex_result>` which contains the prelude and matched groups if the literal is found, or `std::nullopt` if not found.
- **Functions called**:
    - [`common_chat_msg_parser::move_to`](chat-parser.h.driver.md#common_chat_msg_parsermove_to)
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::consume\_literal<!-- {{#callable:common_chat_msg_parser::consume_literal}} -->
The `consume_literal` function attempts to consume a specified literal string from the input at the current position, throwing an exception if unsuccessful.
- **Inputs**:
    - `literal`: A constant reference to a `std::string` representing the literal string to be consumed from the input.
- **Control Flow**:
    - The function calls [`try_consume_literal`](#common_chat_msg_parsertry_consume_literal) with the provided `literal` string.
    - If [`try_consume_literal`](#common_chat_msg_parsertry_consume_literal) returns false, indicating the literal was not found at the current position, the function throws a [`common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception) with the `literal` as its argument.
- **Output**: The function does not return a value; it either successfully consumes the literal or throws an exception.
- **Functions called**:
    - [`common_chat_msg_parser::try_consume_literal`](#common_chat_msg_parsertry_consume_literal)
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::try\_parse\_reasoning<!-- {{#callable:common_chat_msg_parser::try_parse_reasoning}} -->
The `try_parse_reasoning` function attempts to parse reasoning content from a chat message, handling it according to specified syntax rules and returning a boolean indicating success.
- **Inputs**:
    - `start_think`: A string representing the starting delimiter for reasoning content.
    - `end_think`: A string representing the ending delimiter for reasoning content.
- **Control Flow**:
    - Define a lambda function `handle_reasoning` to process and add reasoning content based on syntax rules.
    - Check if the reasoning format is not `COMMON_REASONING_FORMAT_NONE`.
    - Attempt to consume the `start_think` literal or check if `thinking_forced_open` is true.
    - If successful, try to find the `end_think` literal in the input.
    - If `end_think` is found, process the reasoning content using `handle_reasoning` and return true.
    - If `end_think` is not found, consume the rest of the input and process it as reasoning content, allowing for unclosed tags.
    - Return false if the reasoning format is `COMMON_REASONING_FORMAT_NONE` or if `start_think` cannot be consumed.
- **Output**: A boolean value indicating whether reasoning content was successfully parsed and processed.
- **Functions called**:
    - [`common_chat_msg_parser::add_content`](#common_chat_msg_parseradd_content)
    - [`common_chat_msg_parser::add_reasoning_content`](#common_chat_msg_parseradd_reasoning_content)
    - [`common_chat_msg_parser::try_consume_literal`](#common_chat_msg_parsertry_consume_literal)
    - [`common_chat_msg_parser::try_find_literal`](#common_chat_msg_parsertry_find_literal)
    - [`common_chat_msg_parser::consume_spaces`](#common_chat_msg_parserconsume_spaces)
    - [`common_chat_msg_parser::consume_rest`](#common_chat_msg_parserconsume_rest)
    - [`common_chat_msg_parser::is_partial`](chat-parser.h.driver.md#common_chat_msg_parseris_partial)
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::consume\_rest<!-- {{#callable:common_chat_msg_parser::consume_rest}} -->
The `consume_rest` function returns the remaining unprocessed portion of the input string and updates the position to the end of the input.
- **Inputs**: None
- **Control Flow**:
    - Extracts the substring from the current position (`pos_`) to the end of the input string (`input_`).
    - Updates the position (`pos_`) to the size of the input string, effectively marking the entire input as processed.
    - Returns the extracted substring.
- **Output**: A `std::string` containing the unprocessed portion of the input string from the current position to the end.
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::try\_find\_regex<!-- {{#callable:common_chat_msg_parser::try_find_regex}} -->
The `try_find_regex` function attempts to find a regex match in the input string starting from a specified position and returns the prelude and matched groups if successful.
- **Inputs**:
    - `regex`: A `common_regex` object representing the regular expression to search for in the input string.
    - `from`: A `size_t` value indicating the starting position in the input string from which to begin the search; defaults to `std::string::npos`.
    - `add_prelude_to_content`: A `bool` indicating whether to add the prelude (text before the match) to the content of the result.
- **Control Flow**:
    - The function calls `regex.search` on the input string starting from the specified position or the current position if `from` is `std::string::npos`.
    - If no match is found (`COMMON_REGEX_MATCH_TYPE_NONE`), the function returns `std::nullopt`.
    - If a match is found, it calculates the prelude as the substring from the current position to the start of the match and updates the current position to the end of the match.
    - If `add_prelude_to_content` is true, it adds the prelude to the content.
    - If the match type is `COMMON_REGEX_MATCH_TYPE_PARTIAL` and the parser is in partial mode, it throws a [`common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception).
    - If the match type is `COMMON_REGEX_MATCH_TYPE_PARTIAL` and not in partial mode, it returns `std::nullopt`.
    - If a complete match is found, it returns a `find_regex_result` containing the prelude and matched groups.
- **Output**: An `std::optional<find_regex_result>` containing the prelude and matched groups if a complete match is found, or `std::nullopt` if no match or only a partial match is found.
- **Functions called**:
    - [`common_chat_msg_parser::add_content`](#common_chat_msg_parseradd_content)
    - [`common_chat_msg_parser::is_partial`](chat-parser.h.driver.md#common_chat_msg_parseris_partial)
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::consume\_regex<!-- {{#callable:common_chat_msg_parser::consume_regex}} -->
The `consume_regex` function attempts to consume a regex pattern from the input string and returns the match result, throwing an exception if the pattern is not found.
- **Inputs**:
    - `regex`: A `common_regex` object representing the regex pattern to be consumed from the input string.
- **Control Flow**:
    - Call [`try_consume_regex`](#common_chat_msg_parsertry_consume_regex) with the provided regex to attempt to find and consume the pattern from the input string.
    - If [`try_consume_regex`](#common_chat_msg_parsertry_consume_regex) returns a result, dereference and return it as the function's output.
    - If [`try_consume_regex`](#common_chat_msg_parsertry_consume_regex) returns `std::nullopt`, throw a [`common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception) with the regex pattern as the message.
- **Output**: Returns a `find_regex_result` structure containing the prelude and groups of the matched regex pattern.
- **Functions called**:
    - [`common_chat_msg_parser::try_consume_regex`](#common_chat_msg_parsertry_consume_regex)
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::try\_consume\_regex<!-- {{#callable:common_chat_msg_parser::try_consume_regex}} -->
The `try_consume_regex` function attempts to match a given regular expression at the current position in the input string and returns the match result if successful.
- **Inputs**:
    - `regex`: A `common_regex` object representing the regular expression to be matched against the input string.
- **Control Flow**:
    - The function calls `regex.search` with the current input and position to find a match.
    - If the match type is `COMMON_REGEX_MATCH_TYPE_NONE`, it returns `std::nullopt`.
    - If the match type is `COMMON_REGEX_MATCH_TYPE_PARTIAL` and the parser is in partial mode, it throws a [`common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception); otherwise, it returns `std::nullopt`.
    - If the match does not start at the current position, it returns `std::nullopt`.
    - If a match is found at the current position, it updates the position to the end of the match and returns a `find_regex_result` with the match groups.
- **Output**: An `std::optional<find_regex_result>` containing the match groups if a match is found at the current position, or `std::nullopt` if no match is found or if the match is partial and the parser is not in partial mode.
- **Functions called**:
    - [`common_chat_msg_parser::is_partial`](chat-parser.h.driver.md#common_chat_msg_parseris_partial)
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::try\_consume\_json<!-- {{#callable:common_chat_msg_parser::try_consume_json}} -->
The `try_consume_json` function attempts to parse a JSON object from the input string starting at the current position and returns it if successful, handling partial JSONs with a healing marker.
- **Inputs**:
    - `None`: This function does not take any explicit input arguments; it operates on the class member variables.
- **Control Flow**:
    - Initialize an iterator `it` to the current position in the input string and define `end` as the end of the input string.
    - Declare a `common_json` object `result` to store the parsed JSON.
    - Call `common_json_parse` with `it`, `end`, `healing_marker_`, and `result` to attempt parsing the JSON.
    - If parsing fails, return `std::nullopt`.
    - Update `pos_` to the new position after parsing using `std::distance`.
    - Check if the `healing_marker` in `result` is empty; if so, return the parsed JSON `result`.
    - If the `healing_marker` is not empty and the input is not partial, throw a [`common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception).
    - Return the parsed JSON `result` if the input is partial.
- **Output**: Returns an `std::optional<common_json>` containing the parsed JSON object if successful, or `std::nullopt` if parsing fails.
- **Functions called**:
    - [`common_chat_msg_parser::is_partial`](chat-parser.h.driver.md#common_chat_msg_parseris_partial)
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::consume\_json<!-- {{#callable:common_chat_msg_parser::consume_json}} -->
The `consume_json` function attempts to parse JSON from the input and returns the parsed JSON object, throwing an exception if parsing fails.
- **Inputs**: None
- **Control Flow**:
    - The function calls `try_consume_json()` to attempt parsing JSON from the input.
    - If `try_consume_json()` returns a valid result, the function returns the parsed JSON object.
    - If `try_consume_json()` returns no result, indicating a parsing failure, the function throws a [`common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception) with the message "JSON".
- **Output**: The function returns a `common_json` object representing the parsed JSON data.
- **Functions called**:
    - [`common_chat_msg_parser::try_consume_json`](#common_chat_msg_parsertry_consume_json)
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::consume\_json\_with\_dumped\_args<!-- {{#callable:common_chat_msg_parser::consume_json_with_dumped_args}} -->
The `consume_json_with_dumped_args` function attempts to consume JSON data, converting specific subtrees to JSON strings based on provided paths, and throws an exception if the operation fails.
- **Inputs**:
    - `args_paths`: A vector of vectors of strings representing paths in the JSON where subtrees should be converted to JSON strings.
    - `content_paths`: A vector of vectors of strings representing paths in the JSON where string values should be kept truncated and possibly converted to JSON strings.
- **Control Flow**:
    - Call [`try_consume_json_with_dumped_args`](#common_chat_msg_parsertry_consume_json_with_dumped_args) with `args_paths` and `content_paths` to attempt JSON consumption.
    - If [`try_consume_json_with_dumped_args`](#common_chat_msg_parsertry_consume_json_with_dumped_args) returns a result, dereference and return it.
    - If no result is returned, throw a [`common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception) with the message "JSON".
- **Output**: Returns a `consume_json_result` struct containing the JSON value and a boolean indicating if the JSON is partial.
- **Functions called**:
    - [`common_chat_msg_parser::try_consume_json_with_dumped_args`](#common_chat_msg_parsertry_consume_json_with_dumped_args)
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)


---
#### common\_chat\_msg\_parser::try\_consume\_json\_with\_dumped\_args<!-- {{#callable:common_chat_msg_parser::try_consume_json_with_dumped_args}} -->
The function `try_consume_json_with_dumped_args` attempts to parse JSON input, handling partial JSON and converting specified subtrees to JSON strings, while managing healing markers for incomplete data.
- **Inputs**:
    - `args_paths`: A vector of vectors of strings representing paths in the JSON that should be treated as arguments and potentially converted to JSON strings.
    - `content_paths`: A vector of vectors of strings representing paths in the JSON that should be treated as content and potentially converted to JSON strings.
- **Control Flow**:
    - Attempt to parse JSON using [`try_consume_json`](#common_chat_msg_parsertry_consume_json); if unsuccessful, return `std::nullopt`.
    - Define lambda functions `is_arguments_path` and `is_content_path` to check if a given path matches any in `args_paths` or `content_paths`, respectively.
    - If the JSON is fully parsed and no healing marker is present, return the JSON as is or as a dumped string if the entire JSON is an argument.
    - Log the parsed partial JSON and its healing marker.
    - Define a recursive lambda `remove_unsupported_healings_and_dump_args` to traverse the JSON, removing unsupported healings and dumping arguments as needed.
    - Within the lambda, handle paths identified as arguments or content, managing healing markers and truncating strings as necessary.
    - Traverse JSON objects and arrays, applying the lambda recursively to handle nested structures.
    - Log the cleaned JSON and return it along with a flag indicating if a healing marker was found.
- **Output**: An `std::optional<consume_json_result>` containing the cleaned JSON and a boolean indicating if the JSON is partial due to healing markers.
- **Functions called**:
    - [`common_chat_msg_parser::try_consume_json`](#common_chat_msg_parsertry_consume_json)
    - [`common_chat_msg_parser::is_partial`](chat-parser.h.driver.md#common_chat_msg_parseris_partial)
- **See also**: [`common_chat_msg_parser`](chat-parser.h.driver.md#common_chat_msg_parser)  (Data Structure)



