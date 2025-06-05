# Purpose
This C++ source code file is designed to handle chat message processing, particularly in the context of chatbots or conversational AI systems. It provides a comprehensive set of functionalities for parsing, formatting, and managing chat messages, including the handling of tool calls and JSON-based message structures. The file includes various template and parsing mechanisms to support different chat formats, such as Mistral Nemo, Llama 3.x, DeepSeek R1, and others, each with specific requirements for message formatting and tool call integration.

The code is structured around several key components: it defines functions for formatting time, computing string differences, and checking message content. It also includes template specializations for converting chat messages and tools to and from JSON, ensuring compatibility with OpenAI's JSON format. The file supports parsing and applying chat templates using both legacy and Jinja-based methods, allowing for flexible integration with different chat systems. Additionally, it provides error handling and validation mechanisms to ensure the integrity of message parsing and tool call execution. Overall, this file serves as a robust framework for managing chat interactions in a structured and extensible manner.
# Imports and Dependencies

---
- `chat.h`
- `chat-parser.h`
- `common.h`
- `json-partial.h`
- `json-schema-to-grammar.h`
- `log.h`
- `regex-partial.h`
- `minja/chat-template.hpp`
- `minja/minja.hpp`
- `cstdio`
- `exception`
- `iostream`
- `optional`
- `stdexcept`
- `string`
- `vector`


# Data Structures

---
### common\_chat\_templates<!-- {{#data_structure:common_chat_templates}} -->
- **Type**: `struct`
- **Members**:
    - `has_explicit_template`: Indicates if a built-in template or an override was specified.
    - `template_default`: A unique pointer to the default chat template, always set to a default value.
    - `template_tool_use`: A unique pointer to a chat template used for tool interactions.
- **Description**: The `common_chat_templates` struct is designed to manage chat templates within a chat system. It contains a boolean flag `has_explicit_template` to indicate whether a built-in template or an override was specified. The struct also holds two unique pointers to `common_chat_template` objects: `template_default`, which is always set to a default template (typically 'chatml'), and `template_tool_use`, which is used specifically for tool-related interactions. This structure facilitates the customization and application of different chat templates based on the context of the chat.


---
### templates\_params<!-- {{#data_structure:templates_params}} -->
- **Type**: `struct`
- **Members**:
    - `messages`: A JSON object representing the messages.
    - `tools`: A JSON object representing the tools.
    - `tool_choice`: An instance of common_chat_tool_choice indicating the tool choice.
    - `json_schema`: A JSON object representing the JSON schema.
    - `parallel_tool_calls`: A boolean indicating if parallel tool calls are allowed.
    - `stream`: A boolean indicating if streaming is enabled.
    - `grammar`: A string representing the grammar.
    - `add_generation_prompt`: A boolean indicating if a generation prompt should be added, defaulting to true.
    - `enable_thinking`: A boolean indicating if thinking is enabled, defaulting to true.
    - `now`: A time point representing the current system time, defaulting to the current time.
- **Description**: The `templates_params` struct is designed to encapsulate various parameters related to chat templates, including messages, tools, tool choice, JSON schema, and other configuration options such as parallel tool calls, streaming, grammar, and time settings. It provides a flexible structure to manage and configure chat interactions, supporting both default and customizable settings for generation prompts and thinking capabilities.


---
### common\_chat\_msg<!-- {{#data_structure:common_chat_msg}} -->
- **Description**: [See definition](chat.h.driver.md#common_chat_msg)
- **Member Functions**:
    - [`common_chat_msg::empty`](chat.h.driver.md#common_chat_msgempty)
    - [`common_chat_msg::ensure_tool_call_ids_set`](chat.h.driver.md#common_chat_msgensure_tool_call_ids_set)
    - [`common_chat_msg::operator==`](chat.h.driver.md#common_chat_msgoperator==)
    - [`common_chat_msg::operator!=`](chat.h.driver.md#common_chat_msgoperator!=)
    - [`common_chat_msg::to_json_oaicompat`](#common_chat_msgto_json_oaicompat)

**Methods**

---
#### common\_chat\_msg::to\_json\_oaicompat<!-- {{#callable:common_chat_msg::to_json_oaicompat}} -->
The `to_json_oaicompat` function converts a `common_chat_msg` object into a JSON object compatible with OpenAI's format, including handling of content, reasoning content, and tool calls.
- **Inputs**:
    - `none`: This function does not take any input parameters directly, as it is a method of the `common_chat_msg` class and operates on the instance data.
- **Control Flow**:
    - Initialize a JSON object `message` with a default role of 'assistant'.
    - Check if `reasoning_content` is not empty and add it to the `message` JSON object if true.
    - Determine the `content` field of the `message` JSON object based on whether `content` is empty and `tool_calls` is not empty.
    - If `tool_calls` is not empty, iterate over each tool call, convert it to a JSON object with fields 'type', 'function', and 'id', and add it to a JSON array `arr`.
    - Add the `arr` JSON array to the `message` JSON object under the key 'tool_calls' if it is not empty.
    - Return the `message` JSON object.
- **Output**: A JSON object representing the `common_chat_msg` instance in a format compatible with OpenAI's expected structure.
- **See also**: [`common_chat_msg`](chat.h.driver.md#common_chat_msg)  (Data Structure)



---
### common\_chat\_msg\_diff<!-- {{#data_structure:common_chat_msg_diff}} -->
- **Description**: [See definition](chat.h.driver.md#common_chat_msg_diff)
- **Member Functions**:
    - [`common_chat_msg_diff::operator==`](chat.h.driver.md#common_chat_msg_diffoperator==)
    - [`common_chat_msg_diff::compute_diffs`](#common_chat_msg_diffcompute_diffs)

**Methods**

---
#### common\_chat\_msg\_diff::compute\_diffs<!-- {{#callable:common_chat_msg_diff::compute_diffs}} -->
The `compute_diffs` function calculates the differences between two `common_chat_msg` objects and returns a vector of `common_chat_msg_diff` objects representing those differences.
- **Inputs**:
    - `previous_msg`: A `common_chat_msg` object representing the previous state of a chat message.
    - `new_msg`: A `common_chat_msg` object representing the new state of a chat message.
- **Control Flow**:
    - Initialize an empty vector `diffs` to store differences.
    - Check if `reasoning_content` differs between `previous_msg` and `new_msg`; if so, compute the difference and add it to `diffs`.
    - Check if `content` differs between `previous_msg` and `new_msg`; if so, compute the difference and add it to `diffs`.
    - Throw a runtime error if `new_msg` has fewer tool calls than `previous_msg`.
    - If `previous_msg` has tool calls, compare the last tool call's name and arguments; if they differ, compute the differences and add them to `diffs`.
    - For any additional tool calls in `new_msg` that are not in `previous_msg`, add them to `diffs`.
- **Output**: A vector of `common_chat_msg_diff` objects, each representing a difference between the `previous_msg` and `new_msg`.
- **Functions called**:
    - [`string_diff`](#string_diff)
- **See also**: [`common_chat_msg_diff`](chat.h.driver.md#common_chat_msg_diff)  (Data Structure)



# Functions

---
### format\_time<!-- {{#callable:format_time}} -->
The `format_time` function formats a given time point into a string based on a specified format.
- **Inputs**:
    - `now`: A `std::chrono::system_clock::time_point` representing the current time point to be formatted.
    - `format`: A `std::string` specifying the desired format for the output time string, compatible with `std::put_time`.
- **Control Flow**:
    - Convert the `now` time point to a `time_t` object using `std::chrono::system_clock::to_time_t`.
    - Convert the `time_t` object to a `tm` structure representing local time using `std::localtime`.
    - Create a `std::ostringstream` object to build the formatted time string.
    - Use `std::put_time` to format the `tm` structure into the string stream according to the specified format.
    - Extract the formatted string from the string stream and return it.
- **Output**: A `std::string` containing the formatted time.


---
### string\_diff<!-- {{#callable:string_diff}} -->
The `string_diff` function computes the difference between two strings, `last` and `current`, by returning the portion of `current` that follows `last` if `last` is a prefix of `current`, or handling specific cases where `last` is empty or not a prefix.
- **Inputs**:
    - `last`: A reference to the previous string, which is expected to be a prefix of the current string.
    - `current`: A reference to the current string, from which the difference is to be computed.
- **Control Flow**:
    - Check if `last` is empty; if so, return `current` as the difference.
    - Check if `current` starts with `last` using `string_starts_with`; if not, check if `last` starts with `current`.
    - If `last` starts with `current`, return an empty string, indicating a special case where the last generation ended on a partial stop word.
    - If neither condition is met, throw a runtime error indicating an invalid diff.
    - If `current` starts with `last`, return the substring of `current` starting from the length of `last`.
- **Output**: A string representing the difference between `current` and `last`, or an empty string in specific cases, or an exception if the inputs are invalid.


---
### has\_content\_or\_tool\_calls<!-- {{#callable:has_content_or_tool_calls}} -->
The function `has_content_or_tool_calls` checks if a `common_chat_msg` object has non-empty content or tool calls.
- **Inputs**:
    - `msg`: A `common_chat_msg` object that contains content and tool calls to be checked for non-emptiness.
- **Control Flow**:
    - The function checks if the `content` field of `msg` is not empty.
    - It also checks if the `tool_calls` field of `msg` is not empty.
    - The function returns true if either the `content` or `tool_calls` is not empty, otherwise it returns false.
- **Output**: A boolean value indicating whether the `msg` has non-empty content or tool calls.


---
### common\_chat\_tool\_choice\_parse\_oaicompat<!-- {{#callable:common_chat_tool_choice_parse_oaicompat}} -->
The function `common_chat_tool_choice_parse_oaicompat` parses a string representing a tool choice and returns the corresponding `common_chat_tool_choice` enumeration value.
- **Inputs**:
    - `tool_choice`: A string representing the tool choice, which can be 'auto', 'none', or 'required'.
- **Control Flow**:
    - The function checks if the input string `tool_choice` is equal to 'auto', 'none', or 'required'.
    - If `tool_choice` is 'auto', it returns `COMMON_CHAT_TOOL_CHOICE_AUTO`.
    - If `tool_choice` is 'none', it returns `COMMON_CHAT_TOOL_CHOICE_NONE`.
    - If `tool_choice` is 'required', it returns `COMMON_CHAT_TOOL_CHOICE_REQUIRED`.
    - If `tool_choice` does not match any of the expected values, it throws a `std::runtime_error` with a message indicating the invalid tool choice.
- **Output**: Returns a `common_chat_tool_choice` enumeration value corresponding to the input string, or throws a `std::runtime_error` if the input is invalid.


---
### common\_chat\_msgs\_parse\_oaicompat<!-- {{#callable:common_chat_msgs_parse_oaicompat}} -->
The function [`common_chat_msgs_parse_oaicompat`](#common_chat_msgs_parse_oaicompat) parses a JSON string representing chat messages into a vector of `common_chat_msg` objects.
- **Inputs**:
    - `messages`: A JSON string representing an array of chat messages.
- **Control Flow**:
    - The function first parses the input string into a JSON object.
    - It checks if the JSON object is an array, throwing an error if not.
    - For each message in the array, it verifies that the message is an object and contains a 'role' field.
    - It processes the 'content' field if present, handling it as a string or an array of content parts.
    - It processes the 'tool_calls' field if present, extracting tool call details.
    - It checks for the presence of either 'content' or 'tool_calls', throwing an error if neither is present.
    - It extracts optional fields like 'reasoning_content', 'name', and 'tool_call_id'.
    - Each processed message is added to the result vector.
- **Output**: A vector of `common_chat_msg` objects representing the parsed chat messages.
- **Functions called**:
    - [`common_chat_msgs_parse_oaicompat`](#common_chat_msgs_parse_oaicompat)


---
### common\_chat\_msgs\_to\_json\_oaicompat<!-- {{#callable:common_chat_msgs_to_json_oaicompat}} -->
The function `common_chat_msgs_to_json_oaicompat` converts a vector of `common_chat_msg` objects into a JSON array, optionally concatenating text parts if specified.
- **Inputs**:
    - `msgs`: A constant reference to a vector of `common_chat_msg` objects, each representing a chat message with various attributes like role, content, content parts, reasoning content, tool name, tool call ID, and tool calls.
    - `concat_typed_text`: A boolean flag indicating whether to concatenate text parts from `content_parts` into a single string if they exist.
- **Control Flow**:
    - Initialize an empty JSON array named `messages`.
    - Iterate over each `common_chat_msg` object in the `msgs` vector.
    - For each message, check if both `content` and `content_parts` are non-empty, and throw a runtime error if so.
    - Create a JSON object `jmsg` with the `role` of the message.
    - If `content` is non-empty, add it to `jmsg` under the key `content`.
    - If `content_parts` is non-empty and `concat_typed_text` is true, concatenate all text parts of type `text` into a single string and add it to `jmsg` under the key `content`.
    - If `concat_typed_text` is false, add each part in `content_parts` as a JSON object with `type` and `text` to a JSON array under the key `content`.
    - If neither `content` nor `content_parts` is non-empty, set `content` in `jmsg` to null.
    - Add `reasoning_content`, `tool_name`, `tool_call_id`, and `tool_calls` to `jmsg` if they are non-empty, with `tool_calls` being a JSON array of function call objects.
    - Append `jmsg` to the `messages` JSON array.
    - Return the `messages` JSON array.
- **Output**: A JSON array representing the chat messages, with each message converted into a JSON object containing its attributes.


---
### common\_chat\_tools\_parse\_oaicompat<!-- {{#callable:common_chat_tools_parse_oaicompat}} -->
The function [`common_chat_tools_parse_oaicompat`](#common_chat_tools_parse_oaicompat) parses a JSON string representing chat tools into a vector of `common_chat_tool` objects.
- **Inputs**:
    - `tools`: A JSON string representing an array of chat tools.
- **Control Flow**:
    - The function takes a JSON string as input.
    - It parses the JSON string into a JSON object using `json::parse`.
    - It calls another overloaded version of [`common_chat_tools_parse_oaicompat`](#common_chat_tools_parse_oaicompat) that takes a JSON object as input.
- **Output**: A vector of `common_chat_tool` objects parsed from the input JSON string.
- **Functions called**:
    - [`common_chat_tools_parse_oaicompat`](#common_chat_tools_parse_oaicompat)


---
### common\_chat\_tools\_to\_json\_oaicompat<!-- {{#callable:common_chat_tools_to_json_oaicompat}} -->
The function `common_chat_tools_to_json_oaicompat` converts a vector of `common_chat_tool` objects into a JSON array compatible with OpenAI's format.
- **Inputs**:
    - `tools`: A constant reference to a vector of `common_chat_tool` objects, each containing a name, description, and parameters in JSON format.
- **Control Flow**:
    - Check if the `tools` vector is empty; if so, return an empty JSON object.
    - Initialize a JSON array named `result`.
    - Iterate over each `common_chat_tool` object in the `tools` vector.
    - For each tool, parse its `parameters` from a string to a JSON object and create a JSON object with keys `type`, `function`, `name`, `description`, and `parameters`.
    - Add the created JSON object to the `result` array.
    - Return the `result` JSON array.
- **Output**: A JSON array where each element represents a `common_chat_tool` object with its details formatted for OpenAI compatibility.


---
### common\_chat\_msg\_diff\_to\_json\_oaicompat<!-- {{#callable:common_chat_msg_diff_to_json_oaicompat}} -->
The function `common_chat_msg_diff_to_json_oaicompat` converts a `common_chat_msg_diff` object into a JSON object that is compatible with OpenAI's format.
- **Inputs**:
    - `diff`: An instance of `common_chat_msg_diff` containing the differences in chat messages, including reasoning content, content, and tool call deltas.
- **Control Flow**:
    - Initialize a JSON object `delta` to store the differences.
    - Check if `reasoning_content_delta` in `diff` is not empty, and if so, add it to `delta` under the key `reasoning_content`.
    - Check if `content_delta` in `diff` is not empty, and if so, add it to `delta` under the key `content`.
    - Check if `tool_call_index` in `diff` is not `std::string::npos`, indicating a valid tool call index.
    - If a valid tool call index exists, create a JSON object `tool_call` and add the index to it.
    - If `tool_call_delta.id` is not empty, add it to `tool_call` with the key `id` and set the type to `function`.
    - Create a JSON object `function` and add `tool_call_delta.name` and `tool_call_delta.arguments` to it if `name` is not empty.
    - Add the `function` object to `tool_call` under the key `function`.
    - Add the `tool_call` object to `delta` under the key `tool_calls` as an array.
    - Return the `delta` JSON object.
- **Output**: A JSON object representing the differences in the chat message, formatted to be compatible with OpenAI's expected structure.


---
### common\_chat\_verify\_template<!-- {{#callable:common_chat_verify_template}} -->
The `common_chat_verify_template` function verifies the validity of a chat template by applying it to a test message, using either a Jinja-based or legacy template system, and returns a boolean indicating success or failure.
- **Inputs**:
    - `tmpl`: A string representing the chat template to be verified.
    - `use_jinja`: A boolean flag indicating whether to use the Jinja-based template system (true) or the legacy system (false).
- **Control Flow**:
    - If `use_jinja` is true, the function attempts to initialize and apply the template using the Jinja-based system.
    - A `common_chat_msg` object is created with a role of 'user' and content 'test'.
    - The template is initialized using [`common_chat_templates_init`](#common_chat_templates_init) with the provided template string.
    - The template is applied to the test message using [`common_chat_templates_apply`](#common_chat_templates_apply).
    - If the application is successful, the function returns true; otherwise, it catches any exceptions, logs an error, and returns false.
    - If `use_jinja` is false, a `llama_chat_message` array is created with a single message having a role of 'user' and content 'test'.
    - The template is applied using `llama_chat_apply_template`, and the function returns true if the result is non-negative, indicating success.
- **Output**: A boolean value indicating whether the template was successfully applied without errors.
- **Functions called**:
    - [`common_chat_templates_init`](#common_chat_templates_init)
    - [`common_chat_templates_apply`](#common_chat_templates_apply)


---
### common\_chat\_format\_single<!-- {{#callable:common_chat_format_single}} -->
The function `common_chat_format_single` formats a new chat message by applying templates to both past and new messages, and returns the formatted difference between them.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` structure containing chat templates to be applied.
    - `past_msg`: A vector of `common_chat_msg` objects representing past chat messages.
    - `new_msg`: A `common_chat_msg` object representing the new chat message to be formatted.
    - `add_ass`: A boolean flag indicating whether to add an assistant generation prompt.
    - `use_jinja`: A boolean flag indicating whether to use Jinja templates for formatting.
- **Control Flow**:
    - Initialize a `common_chat_templates_inputs` object and set its `use_jinja` property to the `use_jinja` input value.
    - Check if `past_msg` is not empty; if so, set `inputs.messages` to `past_msg` and apply the templates to get the formatted past message.
    - If `add_ass` is true and the formatted past message ends with a newline, append a newline to the output stream.
    - Add `new_msg` to `inputs.messages`, set `inputs.add_generation_prompt` to `add_ass`, and apply the templates to get the formatted new message.
    - Calculate the difference between the formatted new message and the formatted past message, and append this difference to the output stream.
    - Return the resulting formatted difference as a string.
- **Output**: A string representing the formatted difference between the new message and the past messages.
- **Functions called**:
    - [`common_chat_templates_apply`](#common_chat_templates_apply)


---
### common\_chat\_format\_example<!-- {{#callable:common_chat_format_example}} -->
The function `common_chat_format_example` creates a formatted chat prompt using a predefined sequence of messages and applies a chat template to generate the final prompt string.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` structure, which contains the chat templates to be used for formatting the chat messages.
    - `use_jinja`: A boolean flag indicating whether to use Jinja templates for formatting the chat messages.
- **Control Flow**:
    - Initialize a `common_chat_templates_inputs` structure and set its `use_jinja` field to the provided `use_jinja` argument.
    - Define a lambda function `add_simple_msg` to create and add a `common_chat_msg` to the `inputs.messages` vector with a specified role and content.
    - Use `add_simple_msg` to add a series of predefined messages to `inputs.messages`, simulating a simple chat interaction.
    - Call [`common_chat_templates_apply`](#common_chat_templates_apply) with the provided `tmpls` and the populated `inputs` to apply the chat template and generate the formatted prompt.
    - Return the `prompt` field from the result of [`common_chat_templates_apply`](#common_chat_templates_apply).
- **Output**: A `std::string` containing the formatted chat prompt generated by applying the chat template to the predefined messages.
- **Functions called**:
    - [`common_chat_templates_apply`](#common_chat_templates_apply)


---
### common\_chat\_templates\_free<!-- {{#callable:common_chat_templates_free}} -->
The `common_chat_templates_free` function deallocates memory for a `common_chat_templates` structure.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` structure that needs to be deallocated.
- **Control Flow**:
    - The function takes a pointer to a `common_chat_templates` structure as an argument.
    - It uses the `delete` operator to deallocate the memory associated with the `tmpls` pointer.
- **Output**: The function does not return any value.


---
### common\_chat\_templates\_was\_explicit<!-- {{#callable:common_chat_templates_was_explicit}} -->
The function `common_chat_templates_was_explicit` checks if a `common_chat_templates` structure has an explicit template set.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` structure, which contains information about chat templates, including whether an explicit template is set.
- **Control Flow**:
    - Access the `has_explicit_template` member of the `common_chat_templates` structure pointed to by `tmpls`.
    - Return the value of `has_explicit_template`.
- **Output**: A boolean value indicating whether the `common_chat_templates` structure has an explicit template set.


---
### common\_chat\_templates\_source<!-- {{#callable:common_chat_templates_source}} -->
The function `common_chat_templates_source` retrieves the source of a specified chat template variant from a `common_chat_templates` structure.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` structure containing chat templates.
    - `variant`: A C-style string representing the variant of the template to retrieve, such as "tool_use".
- **Control Flow**:
    - Check if `variant` is not null.
    - If `variant` is "tool_use", check if `tmpls->template_tool_use` is not null and return its source; otherwise, return null.
    - If `variant` is not "tool_use", log a debug message indicating an unknown template variant.
    - If `variant` is null or not "tool_use", return the source of `tmpls->template_default`.
- **Output**: A C-style string representing the source of the specified template variant, or null if the variant is "tool_use" and the corresponding template is not set.


---
### common\_chat\_templates\_init<!-- {{#callable:common_chat_templates_init}} -->
The `common_chat_templates_init` function initializes and returns a pointer to a `common_chat_templates` object, configuring chat templates based on provided overrides and a model's default settings.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure, which may provide default chat templates and vocabulary tokens.
    - `chat_template_override`: A string that can override the default chat template source.
    - `bos_token_override`: A string that can override the beginning-of-sequence token.
    - `eos_token_override`: A string that can override the end-of-sequence token.
- **Control Flow**:
    - Initialize `default_template_src` and `template_tool_use_src` as empty strings.
    - Check if `chat_template_override` is empty; if so, assert `model` is not null and attempt to retrieve default templates from the model.
    - If `chat_template_override` is not empty, set `default_template_src` to this override.
    - If `default_template_src` is empty or equals 'chatml', set it to `template_tool_use_src` if available, otherwise use a predefined `CHATML_TEMPLATE_SRC`.
    - Set `token_bos` and `token_eos` to the provided overrides; if `model` is available, retrieve and convert the model's BOS and EOS tokens.
    - Create a new `common_chat_templates` object and set its `has_explicit_template` flag.
    - Attempt to create a `minja::chat_template` for `template_default` using `default_template_src`, logging errors and defaulting to `CHATML_TEMPLATE_SRC` on failure.
    - If `template_tool_use_src` is not empty, attempt to create a `minja::chat_template` for `template_tool_use`, logging errors on failure.
    - Return the initialized `common_chat_templates` pointer.
- **Output**: A `common_chat_templates_ptr`, which is a pointer to a `common_chat_templates` object containing the configured chat templates.
- **Functions called**:
    - [`namespace::parser::get_token`](../vendor/nlohmann/json.hpp.driver.md#parserget_token)


---
### common\_chat\_format\_name<!-- {{#callable:common_chat_format_name}} -->
The function `common_chat_format_name` returns a string representation of a given `common_chat_format` enumeration value.
- **Inputs**:
    - `format`: An enumeration value of type `common_chat_format` representing a specific chat format.
- **Control Flow**:
    - The function uses a switch statement to match the input `format` with predefined cases.
    - Each case corresponds to a specific `common_chat_format` enumeration value and returns a string literal representing the format name.
    - If the input `format` does not match any predefined case, the function throws a `std::runtime_error` indicating an unknown chat format.
- **Output**: A constant character pointer (`const char *`) to a string literal representing the name of the chat format.


---
### common\_reasoning\_format\_name<!-- {{#callable:common_reasoning_format_name}} -->
The function `common_reasoning_format_name` returns a string representation of a given `common_reasoning_format` enumeration value.
- **Inputs**:
    - `format`: An enumeration value of type `common_reasoning_format` which specifies the reasoning format to be converted to a string.
- **Control Flow**:
    - The function uses a switch statement to determine the string representation of the `format` argument.
    - If `format` is `COMMON_REASONING_FORMAT_NONE`, it returns "none".
    - If `format` is `COMMON_REASONING_FORMAT_DEEPSEEK`, it returns "deepseek".
    - If `format` is `COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY`, it returns "deepseek-legacy".
    - If `format` does not match any known case, it throws a `std::runtime_error` with the message "Unknown reasoning format".
- **Output**: A constant character pointer to a string representing the name of the reasoning format.


---
### wrap\_code\_as\_arguments<!-- {{#callable:wrap_code_as_arguments}} -->
The `wrap_code_as_arguments` function wraps a given code string into a JSON object as an argument, handling partial code scenarios by appending a healing marker if necessary.
- **Inputs**:
    - `builder`: An instance of `common_chat_msg_parser` used to determine if the code is partial and to provide a healing marker.
    - `code`: A `std::string` representing the code to be wrapped as an argument.
- **Control Flow**:
    - Initialize an empty string `arguments` to store the JSON-wrapped code.
    - Check if the `builder` indicates a partial code using `is_partial()`.
    - If partial, append the healing marker to the code and wrap it in a JSON object with the key "code".
    - Find the position of the healing marker in the JSON string and resize the string to remove the marker.
    - If not partial, wrap the code directly in a JSON object with the key "code".
    - Return the JSON string `arguments`.
- **Output**: A `std::string` containing the JSON representation of the code, potentially truncated to remove a healing marker if the code was partial.


---
### parse\_json\_tool\_calls<!-- {{#callable:parse_json_tool_calls}} -->
The `parse_json_tool_calls` function parses tool call information from a chat message using specified regex patterns and adds the parsed tool calls to the message builder.
- **Inputs**:
    - `builder`: An instance of `common_chat_msg_parser` used to parse and build the chat message.
    - `block_open`: An optional `common_regex` that defines the opening pattern of a block to be parsed.
    - `function_regex_start_only`: An optional `common_regex` that matches the start of a function call, used only for the first match.
    - `function_regex`: An optional `common_regex` that matches the entire function call pattern.
    - `close_regex`: A `common_regex` that matches the closing pattern of a function call.
    - `block_close`: An optional `common_regex` that defines the closing pattern of a block to be parsed.
    - `allow_raw_python`: A boolean flag indicating whether raw Python code is allowed in the tool call.
    - `get_function_name`: An optional function that extracts the function name from a regex match result.
- **Control Flow**:
    - Define a lambda function `parse_tool_calls` to encapsulate the parsing logic.
    - Initialize `from` to `std::string::npos` and `first` to `true`.
    - Enter a loop to repeatedly find and parse function calls using the provided regex patterns.
    - Use `function_regex_start_only` for the first match if available, otherwise use `function_regex`.
    - If a match is found, extract the function name using `get_function_name` or by accessing the regex group.
    - If the function name is empty, skip the match and continue parsing from the next position.
    - Check if the function name is 'python' and `allow_raw_python` is true to handle raw Python code.
    - If the input at the current position is '{' or raw Python is not allowed, attempt to parse JSON arguments.
    - Add the parsed tool call to the builder and consume the closing regex.
    - If raw Python is allowed, wrap the remaining code as arguments and add the tool call to the builder.
    - If no match is found, break the loop.
    - If `block_close` is provided, consume the closing regex for the block.
    - Consume any remaining spaces and add the rest of the content to the builder.
    - If `block_open` is provided, attempt to find and parse the block using `parse_tool_calls`.
    - If no block is found, add the remaining content to the builder.
- **Output**: The function does not return a value; it modifies the `builder` by adding parsed tool calls and content.
- **Functions called**:
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)
    - [`wrap_code_as_arguments`](#wrap_code_as_arguments)


---
### parse\_prefixed\_json\_tool\_call\_array<!-- {{#callable:parse_prefixed_json_tool_call_array}} -->
The function `parse_prefixed_json_tool_call_array` attempts to parse a JSON array of tool calls prefixed by a regex pattern and adds them to a message parser, handling incomplete tool call arrays by throwing an exception.
- **Inputs**:
    - `builder`: An instance of `common_chat_msg_parser` used to parse and build the chat message content.
    - `prefix`: A `common_regex` object representing the regex pattern that prefixes the JSON tool call array.
    - `rstrip_prefix`: An optional size_t parameter indicating how many characters to move back in the builder after finding the prefix; defaults to 0.
- **Control Flow**:
    - The function defines a static vector `args_paths` with a single path `{"arguments"}` to locate arguments in the JSON structure.
    - It checks if the `builder` can find the `prefix` regex in the input using `try_find_regex`.
    - If the prefix is found, it moves the builder's position back by `rstrip_prefix` characters.
    - It then attempts to consume a JSON array of tool calls using `consume_json_with_dumped_args` with `args_paths`.
    - If the tool calls are successfully added to the builder and are not partial, the function completes.
    - If the tool calls are partial or cannot be added, it throws a [`common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception).
    - If the prefix is not found, it adds the remaining content of the builder to the message content using `add_content`.
- **Output**: The function does not return a value but modifies the `builder` by adding parsed tool calls or remaining content.
- **Functions called**:
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)


---
### foreach\_function<!-- {{#callable:foreach_function}} -->
The `foreach_function` function iterates over a JSON array of tools and applies a given function to each tool that is of type 'function' and contains a 'function' key.
- **Inputs**:
    - `tools`: A JSON array containing tool objects, each of which may have various attributes including 'type' and 'function'.
    - `fn`: A function object that takes a JSON object as an argument and performs an operation on it.
- **Control Flow**:
    - Iterate over each tool in the 'tools' JSON array.
    - Check if the current tool contains a 'type' key, if the 'type' is 'function', and if it contains a 'function' key.
    - If any of these conditions are not met, log a message indicating the tool is being skipped and continue to the next tool.
    - If all conditions are met, apply the provided function 'fn' to the current tool.
- **Output**: The function does not return any value; it performs operations via the provided function 'fn' on each valid tool.


---
### apply<!-- {{#callable:apply}} -->
The `apply` function processes a chat template with given inputs and returns a formatted string result, ensuring no double BOS/EOS tokens.
- **Inputs**:
    - `tmpl`: A `common_chat_template` object representing the chat template to be applied.
    - `messages`: A `nlohmann::ordered_json` object containing the messages to be included in the template.
    - `tools`: A `nlohmann::ordered_json` object containing the tools to be used in the template.
    - `add_generation_prompt`: A boolean flag indicating whether to add a generation prompt to the template.
    - `extra_context`: An optional `nlohmann::ordered_json` object providing additional context for the template, defaulting to an empty JSON object.
- **Control Flow**:
    - Initialize a `minja::chat_template_inputs` object and set its properties with the provided inputs.
    - Create a `minja::chat_template_options` object for template options.
    - Apply the template using `tmpl.apply` with the initialized inputs and options.
    - Check if the result starts with the BOS token and remove it if present.
    - Check if the result ends with the EOS token and remove it if present.
    - Return the processed result string.
- **Output**: A `std::string` containing the processed result of the chat template application, with any leading BOS or trailing EOS tokens removed.


---
### common\_chat\_params\_init\_generic<!-- {{#callable:common_chat_params_init_generic}} -->
The function `common_chat_params_init_generic` initializes and returns a `common_chat_params` object based on a given chat template and input parameters, configuring tool call schemas and JSON response handling.
- **Inputs**:
    - `tmpl`: A `common_chat_template` object representing the chat template to be used for generating the chat prompt.
    - `inputs`: A `templates_params` structure containing various parameters such as messages, tools, tool choice, JSON schema, and flags for parallel tool calls and generation prompts.
- **Control Flow**:
    - Initialize a `common_chat_params` object named `data`.
    - Create a JSON array `tool_call_schemas` to store tool call schemas.
    - Iterate over each tool in `inputs.tools` and construct a JSON schema for each tool, adding it to `tool_call_schemas`.
    - If `inputs.parallel_tool_calls` is true, add an 'id' property to the tool schema and mark it as required.
    - Determine the `tool_call` JSON schema based on whether parallel tool calls are allowed.
    - Construct the final `schema` JSON object, which may include a response schema if `inputs.tool_choice` is not `COMMON_CHAT_TOOL_CHOICE_REQUIRED`.
    - Set `data.grammar_lazy` to false and build a grammar using the constructed `schema`.
    - Modify the input messages to include a system message instructing responses in JSON format.
    - Generate the chat prompt using the [`apply`](#apply) function with the modified messages and tools.
    - Set the format of `data` to `COMMON_CHAT_FORMAT_GENERIC`.
    - Return the initialized `common_chat_params` object `data`.
- **Output**: A `common_chat_params` object configured with the specified template, input parameters, and generated prompt.
- **Functions called**:
    - [`foreach_function`](#foreach_function)
    - [`apply`](#apply)


---
### common\_chat\_parse\_generic<!-- {{#callable:common_chat_parse_generic}} -->
The function `common_chat_parse_generic` processes a chat message using a parser, extracting tool calls or responses from JSON data and handling incomplete data scenarios.
- **Inputs**:
    - `builder`: An instance of `common_chat_msg_parser` that provides methods to parse and build chat messages.
- **Control Flow**:
    - Check if the parser's syntax allows parsing tool calls; if not, consume the rest of the input as content and return.
    - Define static vectors for content and argument paths used in JSON parsing.
    - Consume JSON data using the defined paths and store the result in `data`.
    - Check if the JSON data contains 'tool_calls', 'tool_call', or 'response' and handle each case accordingly.
    - If 'tool_calls' is present, attempt to add them to the builder; throw an exception if incomplete.
    - If 'tool_call' is present, attempt to add it to the builder; throw an exception if incomplete.
    - If 'response' is present, add it as content to the builder; throw an exception if incomplete.
    - If none of the expected keys are present in the JSON, throw an exception indicating the expected keys.
- **Output**: The function does not return a value but modifies the `builder` object by adding content or tool calls based on the parsed JSON data.
- **Functions called**:
    - [`common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exception)


---
### common\_chat\_params\_init\_mistral\_nemo<!-- {{#callable:common_chat_params_init_mistral_nemo}} -->
The function `common_chat_params_init_mistral_nemo` initializes and returns a `common_chat_params` object configured for the Mistral Nemo chat format using the provided template and input parameters.
- **Inputs**:
    - `tmpl`: A `common_chat_template` object representing the chat template to be used for generating the prompt.
    - `inputs`: A `templates_params` structure containing various parameters such as messages, tools, tool choice, and other configuration options for initializing the chat parameters.
- **Control Flow**:
    - Initialize a `common_chat_params` object named `data`.
    - Set `data.grammar_lazy` based on whether the tool choice is not required.
    - Build a grammar using a lambda function that iterates over the tools in `inputs` to construct JSON schemas for each tool's function, including properties like `name`, `arguments`, and `id`.
    - Create a JSON schema for an array of tool call objects, with constraints on the number of items based on `inputs.parallel_tool_calls`.
    - Add a grammar rule named `root` that includes a trigger word `[TOOL_CALLS]` followed by the tool calls schema.
    - Add `[TOOL_CALLS]` to `data.grammar_triggers` and `data.preserved_tokens`.
    - Generate a prompt using the [`apply`](#apply) function with the template, messages, tools, and generation prompt flag from `inputs`.
    - Set `data.format` to `COMMON_CHAT_FORMAT_MISTRAL_NEMO`.
    - Return the initialized `data` object.
- **Output**: A `common_chat_params` object configured for the Mistral Nemo chat format, containing grammar rules, triggers, preserved tokens, a generated prompt, and the format type.
- **Functions called**:
    - [`foreach_function`](#foreach_function)
    - [`apply`](#apply)


---
### common\_chat\_parse\_mistral\_nemo<!-- {{#callable:common_chat_parse_mistral_nemo}} -->
The function `common_chat_parse_mistral_nemo` processes a chat message using a specific parsing strategy for the Mistral Nemo format, handling tool calls if enabled.
- **Inputs**:
    - `builder`: A reference to a `common_chat_msg_parser` object that manages the parsing state and accumulates the parsed message content.
- **Control Flow**:
    - Check if the `parse_tool_calls` flag in the builder's syntax is false; if so, consume the rest of the input as content and return.
    - Define a static regex `prefix` to match the string '[TOOL_CALLS]'.
    - Call [`parse_prefixed_json_tool_call_array`](#parse_prefixed_json_tool_call_array) with the builder and the prefix to parse tool calls prefixed by '[TOOL_CALLS]'.
- **Output**: The function does not return a value; it modifies the `builder` object to include parsed content or tool calls.
- **Functions called**:
    - [`parse_prefixed_json_tool_call_array`](#parse_prefixed_json_tool_call_array)


---
### common\_chat\_params\_init\_command\_r7b<!-- {{#callable:common_chat_params_init_command_r7b}} -->
The function `common_chat_params_init_command_r7b` initializes and configures a `common_chat_params` object for the Command R7B chat format, processing input messages and tools to generate a prompt and grammar rules.
- **Inputs**:
    - `tmpl`: A `common_chat_template` object representing the chat template to be applied.
    - `inputs`: A `templates_params` struct containing input parameters such as messages, tools, tool choice, and other configuration options.
- **Control Flow**:
    - Initialize a `common_chat_params` object named `data`.
    - Create an empty JSON array `adjusted_messages`.
    - Iterate over each message in `inputs.messages`.
    - For each message, check if it contains both `reasoning_content` and `tool_calls`.
    - If both are present, move `reasoning_content` to `tool_plan` and remove `reasoning_content` from the message.
    - Add the adjusted or original message to `adjusted_messages`.
    - Generate a prompt using the [`apply`](#apply) function with the template, adjusted messages, tools, and other parameters.
    - Set the `format` of `data` to `COMMON_CHAT_FORMAT_COMMAND_R7B`.
    - Check if the prompt ends with `<|START_THINKING|>`, and modify the prompt or set `thinking_forced_open` based on `inputs.enable_thinking`.
    - Configure grammar rules using `build_grammar`, defining schemas for tool calls based on the tools provided.
    - Add grammar triggers and preserved tokens to `data`.
    - Return the configured `data` object.
- **Output**: A `common_chat_params` object configured for the Command R7B format, containing the generated prompt, grammar rules, and other settings.
- **Functions called**:
    - [`apply`](#apply)
    - [`foreach_function`](#foreach_function)


---
### common\_chat\_parse\_command\_r7b<!-- {{#callable:common_chat_parse_command_r7b}} -->
The function `common_chat_parse_command_r7b` parses a chat message using specific regex patterns to identify and process tool calls or responses within the message.
- **Inputs**:
    - `builder`: An instance of `common_chat_msg_parser` used to parse the chat message and manage the parsing state.
- **Control Flow**:
    - The function begins by attempting to parse reasoning content between the markers `<|START_THINKING|>` and `<|END_THINKING|>`.
    - It defines regex patterns for identifying the start and end of actions and responses within the chat message.
    - The function checks if the start action regex is found in the message; if so, it attempts to parse tool calls as JSON with specific parameters.
    - For each tool call, it extracts the tool name, call ID, and arguments, adding them to the builder; if any tool call is incomplete, it throws an exception.
    - If the start response regex is found, it checks for the end response regex; if not found, it adds the remaining content to the builder and throws an exception.
    - If neither start action nor start response regex is found, it adds the remaining content to the builder.
- **Output**: The function does not return a value but modifies the `builder` to include parsed tool calls or content, and may throw exceptions if parsing is incomplete.
- **Functions called**:
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)


---
### expect\_tool\_parameters<!-- {{#callable:expect_tool_parameters}} -->
The `expect_tool_parameters` function validates that a given JSON object representing tool parameters contains specific required properties and matches an expected structure.
- **Inputs**:
    - `name`: A string representing the name of the tool whose parameters are being validated.
    - `parameters`: A JSON object representing the parameters of the tool, which should include 'type', 'properties', and 'required' fields.
    - `expected_properties`: A vector of strings representing the properties that are expected to be present in the tool's parameters.
- **Control Flow**:
    - Check if 'parameters' is an object and contains 'type', 'properties', and 'required' fields, and if 'type' is 'object'; throw an error if any condition is not met.
    - Retrieve 'properties' and 'required' fields from 'parameters'.
    - Iterate over each property in 'expected_properties'.
    - For each property, check if it exists in 'parameters_properties'; throw an error if it does not.
    - Check if each property is marked as required in 'parameters_required'; throw an error if it is not.
    - Check if the size of 'parameters_properties' matches the size of 'expected_properties'; throw an error if they do not match.
- **Output**: The function does not return a value; it throws a runtime error if any validation fails.


---
### common\_chat\_params\_init\_llama\_3\_x<!-- {{#callable:common_chat_params_init_llama_3_x}} -->
The function `common_chat_params_init_llama_3_x` initializes and returns a `common_chat_params` object configured for the Llama 3.x chat format, potentially with built-in tools, based on the provided template and input parameters.
- **Inputs**:
    - `tmpl`: A `common_chat_template` object representing the chat template to be used for generating the chat prompt.
    - `inputs`: A `templates_params` struct containing various parameters such as messages, tools, tool choice, and other configuration options for the chat.
    - `allow_python_tag_builtin_tools`: A boolean flag indicating whether built-in tools that use the `<|python_tag|>` should be allowed.
- **Control Flow**:
    - Initialize an empty JSON array `builtin_tools` and a `common_chat_params` object `data`.
    - Check if `inputs.tools` is not null; if true, proceed to configure grammar and tool rules.
    - Set `data.grammar_lazy` based on whether tool choice is required.
    - Define a lambda `handle_builtin_tool` to handle specific built-in tools like 'wolfram_alpha', 'web_search', 'brave_search', 'python', and 'code_interpreter'.
    - Iterate over each tool in `inputs.tools` using [`foreach_function`](#foreach_function), resolving references and adding rules for each tool.
    - If `allow_python_tag_builtin_tools` is true, attempt to handle built-in tools using `handle_builtin_tool`.
    - Add grammar triggers and preserved tokens if any built-in tools are present.
    - Add a root rule to the grammar builder using the constructed tool rules.
    - Set `data.format` based on whether built-in tools are allowed and present.
    - If `inputs.tools` is null, set `data.format` to `COMMON_CHAT_FORMAT_CONTENT_ONLY`.
    - Generate the chat prompt using the [`apply`](#apply) function with the template, messages, tools, and additional context.
    - Return the configured `common_chat_params` object `data`.
- **Output**: A `common_chat_params` object configured with grammar, format, and prompt based on the input parameters and template.
- **Functions called**:
    - [`expect_tool_parameters`](#expect_tool_parameters)
    - [`foreach_function`](#foreach_function)
    - [`apply`](#apply)
    - [`format_time`](#format_time)


---
### common\_chat\_parse\_llama\_3\_1<!-- {{#callable:common_chat_parse_llama_3_1}} -->
The function `common_chat_parse_llama_3_1` parses chat messages to identify and process tool calls, supporting both JSON-based and built-in tool call formats.
- **Inputs**:
    - `builder`: An instance of `common_chat_msg_parser` used to build and parse chat messages.
    - `with_builtin_tools`: A boolean flag indicating whether to parse built-in tool calls using a specific format.
- **Control Flow**:
    - Check if tool call parsing is enabled in the builder's syntax; if not, consume the rest of the input as content and return.
    - Define static regular expressions for parsing function calls and closing braces.
    - If `with_builtin_tools` is true, define a regex for built-in tool calls and attempt to find a match in the input.
    - If a built-in tool call is found, extract the function name and arguments, then add the tool call to the builder.
    - If no built-in tool call is found or `with_builtin_tools` is false, use [`parse_json_tool_calls`](#parse_json_tool_calls) to parse JSON-based tool calls.
- **Output**: The function does not return a value but modifies the `builder` to include parsed tool calls or content.
- **Functions called**:
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)
    - [`parse_json_tool_calls`](#parse_json_tool_calls)


---
### common\_chat\_params\_init\_deepseek\_r1<!-- {{#callable:common_chat_params_init_deepseek_r1}} -->
The function `common_chat_params_init_deepseek_r1` initializes and configures chat parameters for the DeepSeek R1 format, applying necessary template adjustments and grammar rules based on input parameters.
- **Inputs**:
    - `tmpl`: A `common_chat_template` object representing the chat template to be used.
    - `inputs`: A `templates_params` struct containing various parameters such as messages, tools, tool choice, JSON schema, and flags for additional configurations.
- **Control Flow**:
    - Initialize a `common_chat_params` object named `data`.
    - Apply the template `tmpl` with the provided `inputs` to generate a `prompt`.
    - Check if the template source contains a specific broken prompt pattern and apply fixes if necessary.
    - Set the `prompt` and `format` fields of `data` to the generated prompt and `COMMON_CHAT_FORMAT_DEEPSEEK_R1`, respectively.
    - Check if the prompt ends with `<think>\n` and adjust the prompt or set `thinking_forced_open` based on `inputs.enable_thinking`.
    - If `inputs.tools` is an array and not empty, configure grammar rules and triggers for tool calls using a `common_grammar_builder`.
    - Return the configured `data` object.
- **Output**: A `common_chat_params` object configured with the prompt, format, grammar, and other settings for the DeepSeek R1 chat format.
- **Functions called**:
    - [`minja::chat_template::apply`](../vendor/minja/chat-template.hpp.driver.md#chat_templateapply)
    - [`foreach_function`](#foreach_function)


---
### common\_chat\_parse\_deepseek\_r1<!-- {{#callable:common_chat_parse_deepseek_r1}} -->
The function `common_chat_parse_deepseek_r1` parses a chat message using a specific format, extracting tool calls and reasoning content if applicable.
- **Inputs**:
    - `builder`: A reference to a `common_chat_msg_parser` object that is used to build and parse the chat message.
- **Control Flow**:
    - The function begins by attempting to parse reasoning content enclosed in '<think>' and '</think>' tags using `builder.try_parse_reasoning`.
    - It checks if tool call parsing is enabled by evaluating `builder.syntax().parse_tool_calls`.
    - If tool call parsing is disabled, it adds the remaining content to the builder and returns.
    - If tool call parsing is enabled, it defines several regular expressions for identifying tool call sections and function calls within the message.
    - The function then calls [`parse_json_tool_calls`](#parse_json_tool_calls) with the builder and the defined regular expressions to parse and extract tool calls from the message.
- **Output**: The function does not return a value; it modifies the `builder` object to include parsed content and tool calls.
- **Functions called**:
    - [`parse_json_tool_calls`](#parse_json_tool_calls)


---
### common\_chat\_params\_init\_firefunction\_v2<!-- {{#callable:common_chat_params_init_firefunction_v2}} -->
The function `common_chat_params_init_firefunction_v2` initializes and returns a `common_chat_params` object configured for the FireFunction V2 chat format, based on the provided template and input parameters.
- **Inputs**:
    - `tmpl`: A `common_chat_template` object representing the chat template to be used for generating the prompt.
    - `inputs`: A `templates_params` structure containing various parameters such as messages, tools, tool choice, and other settings needed for initializing the chat parameters.
- **Control Flow**:
    - Log the function entry using `LOG_DBG`.
    - Initialize a `common_chat_params` object named `data`.
    - Generate a prompt using the [`apply`](../vendor/minja/chat-template.hpp.driver.md#chat_templateapply) function with the provided template, messages, and additional context including the current datetime and tools, and assign it to `data.prompt`.
    - Check if `inputs.tools` is an array and not empty.
    - If true, set `data.grammar_lazy` based on the tool choice and build a grammar using `build_grammar` with schemas derived from the tools, then set `data.grammar_triggers` and `data.preserved_tokens` accordingly, and set `data.format` to `COMMON_CHAT_FORMAT_FIREFUNCTION_V2`.
    - If false, set `data.format` to `COMMON_CHAT_FORMAT_CONTENT_ONLY`.
    - Return the `data` object.
- **Output**: A `common_chat_params` object configured with the generated prompt, grammar, format, and other settings based on the input parameters.
- **Functions called**:
    - [`minja::chat_template::apply`](../vendor/minja/chat-template.hpp.driver.md#chat_templateapply)
    - [`format_time`](#format_time)
    - [`foreach_function`](#foreach_function)


---
### common\_chat\_parse\_firefunction\_v2<!-- {{#callable:common_chat_parse_firefunction_v2}} -->
The function `common_chat_parse_firefunction_v2` processes a chat message using a specific parsing format, adding content or parsing tool calls based on the message's syntax.
- **Inputs**:
    - `builder`: An instance of `common_chat_msg_parser` that contains the chat message to be parsed and provides methods for parsing and modifying the message content.
- **Control Flow**:
    - Check if the `builder`'s syntax allows parsing tool calls; if not, add the remaining content to the builder and return.
    - Define a static regex `prefix` to match the string ' functools['.
    - Call [`parse_prefixed_json_tool_call_array`](#parse_prefixed_json_tool_call_array) with the `builder`, `prefix`, and a `rstrip_prefix` value of 1 to parse tool calls prefixed by the regex.
- **Output**: The function does not return a value; it modifies the `builder` by adding content or tool calls based on the parsed message.
- **Functions called**:
    - [`parse_prefixed_json_tool_call_array`](#parse_prefixed_json_tool_call_array)


---
### common\_chat\_params\_init\_functionary\_v3\_2<!-- {{#callable:common_chat_params_init_functionary_v3_2}} -->
The function `common_chat_params_init_functionary_v3_2` initializes a `common_chat_params` object for the Functionary v3.2 chat format, setting up the prompt and grammar based on the provided template and input parameters.
- **Inputs**:
    - `tmpl`: A `common_chat_template` object representing the chat template to be applied.
    - `inputs`: A `templates_params` struct containing various parameters such as messages, tools, tool choice, and other configuration options.
- **Control Flow**:
    - Initialize a `common_chat_params` object named `data`.
    - Set the `prompt` field of `data` by applying the template `tmpl` with the input messages, tools, and generation prompt flag from `inputs`.
    - Set the `format` field of `data` to `COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2`.
    - Check if `inputs.tools` is an array and not empty.
    - If true, set `data.grammar_lazy` based on the tool choice and build a grammar using a lambda function.
    - Within the lambda, iterate over each tool in `inputs.tools`, extracting the function name and parameters.
    - Resolve references in the parameters and create rules for each function call, handling special cases for 'python' functions.
    - Add grammar triggers for each function name pattern and set preserved tokens.
    - Define rules for the first and subsequent tool calls, and add them to the grammar.
- **Output**: Returns a `common_chat_params` object with initialized prompt, format, grammar, and other related fields.
- **Functions called**:
    - [`minja::chat_template::apply`](../vendor/minja/chat-template.hpp.driver.md#chat_templateapply)
    - [`foreach_function`](#foreach_function)


---
### common\_chat\_parse\_functionary\_v3\_2<!-- {{#callable:common_chat_parse_functionary_v3_2}} -->
The function `common_chat_parse_functionary_v3_2` parses chat messages using specific regex patterns to identify function calls and their arguments, allowing for raw Python code execution.
- **Inputs**:
    - `builder`: An instance of `common_chat_msg_parser` used to build and parse chat messages.
- **Control Flow**:
    - Define static regex patterns for function start, function continuation, and closing sequences.
    - Call [`parse_json_tool_calls`](#parse_json_tool_calls) with the builder and regex patterns to parse the input for function calls.
    - Within [`parse_json_tool_calls`](#parse_json_tool_calls), iterate over the input to find matches for the function start regex, and extract function names and arguments.
    - If a function name is 'python', allow raw Python code execution by consuming the rest of the input as code.
    - Handle incomplete tool calls by throwing exceptions if necessary.
- **Output**: The function does not return a value; it modifies the `builder` to include parsed tool calls and content.
- **Functions called**:
    - [`parse_json_tool_calls`](#parse_json_tool_calls)


---
### common\_chat\_params\_init\_functionary\_v3\_1\_llama\_3\_1<!-- {{#callable:common_chat_params_init_functionary_v3_1_llama_3_1}} -->
The function `common_chat_params_init_functionary_v3_1_llama_3_1` initializes and returns a `common_chat_params` object based on the provided chat template and input parameters, specifically for the Functionary v3.1 Llama 3.1 format.
- **Inputs**:
    - `tmpl`: A `common_chat_template` object representing the chat template to be used.
    - `inputs`: A `templates_params` struct containing various parameters such as messages, tools, tool choice, and other configuration options.
- **Control Flow**:
    - Initialize a `common_chat_params` object named `data`.
    - Check if `inputs.tools` is not null.
    - If tools are present, initialize variables for Python code argument name and a flag for raw Python.
    - Set `data.grammar_lazy` based on the tool choice.
    - Build grammar rules using a lambda function that iterates over each tool in `inputs.tools`.
    - For each tool, extract function name and parameters, and handle special cases for Python tools.
    - Add grammar rules for tool calls and raw Python if applicable.
    - Set `data.format` to `COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1` if tools are present, otherwise set to `COMMON_CHAT_FORMAT_CONTENT_ONLY`.
    - Apply the template using the [`apply`](#apply) function with the provided messages, tools, and other parameters.
    - Return the `data` object.
- **Output**: A `common_chat_params` object configured with grammar rules, format, and prompt based on the input parameters and template.
- **Functions called**:
    - [`foreach_function`](#foreach_function)
    - [`apply`](#apply)


---
### common\_chat\_parse\_functionary\_v3\_1\_llama\_3\_1<!-- {{#callable:common_chat_parse_functionary_v3_1_llama_3_1}} -->
The function `common_chat_parse_functionary_v3_1_llama_3_1` processes chat messages using a specific parsing format, handling both JSON tool calls and a legacy Python tool call format.
- **Inputs**:
    - `builder`: An instance of `common_chat_msg_parser` that manages the parsing state and stores the parsed content or tool calls.
- **Control Flow**:
    - Check if the builder's syntax allows parsing tool calls; if not, consume the rest of the input as content and return.
    - Define static regular expressions for parsing Python tags and function calls.
    - Call [`parse_json_tool_calls`](#parse_json_tool_calls) to handle JSON tool calls using the defined regular expressions.
    - Check for a Python tag using `try_find_regex`; if found, wrap the remaining input as arguments and add a Python tool call to the builder.
- **Output**: The function does not return a value; it modifies the `builder` object by adding content or tool calls based on the parsed input.
- **Functions called**:
    - [`parse_json_tool_calls`](#parse_json_tool_calls)
    - [`wrap_code_as_arguments`](#wrap_code_as_arguments)


---
### common\_chat\_params\_init\_hermes\_2\_pro<!-- {{#callable:common_chat_params_init_hermes_2_pro}} -->
The function `common_chat_params_init_hermes_2_pro` initializes and returns a `common_chat_params` object configured for the Hermes 2 Pro chat format, incorporating tool call handling and optional thinking prompts.
- **Inputs**:
    - `tmpl`: A `common_chat_template` object representing the chat template to be applied.
    - `inputs`: A `templates_params` struct containing parameters such as messages, tools, tool choice, and other configuration options for the chat.
- **Control Flow**:
    - Initialize a `common_chat_params` object named `data`.
    - Create a JSON object `additional_context` with the `enable_thinking` parameter from `inputs`.
    - Apply the template `tmpl` with the provided messages, tools, and additional context to generate the `prompt` for `data`.
    - Set the `format` of `data` to `COMMON_CHAT_FORMAT_HERMES_2_PRO`.
    - Check if the `prompt` ends with `<think>\n`; if so, append `</think>` if `enable_thinking` is false, or set `thinking_forced_open` to true if it is true.
    - If tools are provided, configure the grammar for tool calls using a `common_grammar_builder`, adding rules and triggers for tool call syntax and patterns.
    - Return the configured `common_chat_params` object `data`.
- **Output**: A `common_chat_params` object configured for the Hermes 2 Pro chat format, including the generated prompt, format, grammar, and thinking settings.
- **Functions called**:
    - [`minja::chat_template::apply`](../vendor/minja/chat-template.hpp.driver.md#chat_templateapply)
    - [`foreach_function`](#foreach_function)


---
### common\_chat\_parse\_hermes\_2\_pro<!-- {{#callable:common_chat_parse_hermes_2_pro}} -->
The function `common_chat_parse_hermes_2_pro` parses a chat message using the Hermes 2 Pro format, extracting tool calls and content based on specific regex patterns.
- **Inputs**:
    - `builder`: An instance of `common_chat_msg_parser` used to build and parse the chat message.
- **Control Flow**:
    - The function starts by attempting to parse reasoning content enclosed in `<think>` tags using `builder.try_parse_reasoning`.
    - If `builder.syntax().parse_tool_calls` is false, it adds the remaining content to the builder and returns.
    - A static regex `open_regex` is defined to match various tool call and function call patterns.
    - The function attempts to find a match using `builder.try_find_regex(open_regex)`.
    - If a match is found, it checks if the match corresponds to a named tool call or a function call.
    - For a named tool call, it moves the builder to the start of the JSON object and attempts to consume JSON arguments.
    - If successful, it adds the tool call to the builder, consumes spaces, and checks for closing tags and block ends.
    - For a function call, it extracts the function name, consumes JSON arguments, and adds the tool call to the builder.
    - If no match is found, it adds the remaining content to the builder.
- **Output**: The function does not return a value; it modifies the `builder` to include parsed tool calls and content.
- **Functions called**:
    - [`common_chat_msg_partial_exception::common_chat_msg_partial_exception`](chat-parser.h.driver.md#common_chat_msg_partial_exceptioncommon_chat_msg_partial_exception)


---
### common\_chat\_params\_init\_without\_tools<!-- {{#callable:common_chat_params_init_without_tools}} -->
The function `common_chat_params_init_without_tools` initializes a `common_chat_params` object using a chat template and input parameters, without utilizing any tools.
- **Inputs**:
    - `tmpl`: A `common_chat_template` object that provides the template for generating the chat prompt.
    - `inputs`: A `templates_params` struct containing various input parameters such as messages, tools, JSON schema, grammar, and other flags for prompt generation.
- **Control Flow**:
    - Initialize a `common_chat_params` object named `data`.
    - Set `data.prompt` by applying the template `tmpl` with the input messages, tools (if any), and the `add_generation_prompt` flag.
    - Set `data.format` to `COMMON_CHAT_FORMAT_CONTENT_ONLY`.
    - Set `data.grammar_lazy` to `false`.
    - Check if `inputs.json_schema` is not null; if true, ensure `inputs.grammar` is empty, then convert `inputs.json_schema` to a grammar and assign it to `data.grammar`.
    - If `inputs.json_schema` is null, assign `inputs.grammar` to `data.grammar`.
    - Return the initialized `data` object.
- **Output**: A `common_chat_params` object initialized with the provided template and input parameters, without using any tools.
- **Functions called**:
    - [`apply`](#apply)
    - [`json_schema_to_grammar`](json-schema-to-grammar.cpp.driver.md#json_schema_to_grammar)


---
### common\_chat\_templates\_apply\_jinja<!-- {{#callable:common_chat_templates_apply_jinja}} -->
The function `common_chat_templates_apply_jinja` applies a Jinja-based chat template to generate chat parameters based on the provided inputs and template capabilities.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` structure containing the chat templates to be used.
    - `inputs`: A reference to a `common_chat_templates_inputs` structure containing the inputs for the chat template, such as messages, tools, and other parameters.
- **Control Flow**:
    - Initialize a `templates_params` structure and convert input tools to JSON format compatible with OpenAI using `common_chat_tools_to_json_oaicompat`.
    - Select the appropriate template based on whether tools are used and the availability of a tool-specific template.
    - Convert input messages to JSON format compatible with OpenAI using `common_chat_msgs_to_json_oaicompat`.
    - Set various parameters in `params` based on the inputs, such as `add_generation_prompt`, `tool_choice`, `enable_thinking`, `grammar`, and `now`.
    - Parse the `json_schema` from inputs if it is not empty and assign it to `params.json_schema`.
    - Check if `parallel_tool_calls` is supported by the template and adjust `params.parallel_tool_calls` accordingly.
    - If tools are used, ensure that grammar is not specified and log a warning if the template supports tool calls but not tools.
    - Determine the appropriate handler function to call based on the template's source content and capabilities, such as [`common_chat_params_init_deepseek_r1`](#common_chat_params_init_deepseek_r1), [`common_chat_params_init_command_r7b`](#common_chat_params_init_command_r7b), etc.
    - Return the result of the selected handler function, which initializes and returns a `common_chat_params` structure.
- **Output**: A `common_chat_params` structure containing the initialized chat parameters based on the applied template and inputs.
- **Functions called**:
    - [`common_chat_params_init_deepseek_r1`](#common_chat_params_init_deepseek_r1)
    - [`common_chat_params_init_command_r7b`](#common_chat_params_init_command_r7b)
    - [`common_chat_params_init_hermes_2_pro`](#common_chat_params_init_hermes_2_pro)
    - [`common_chat_params_init_generic`](#common_chat_params_init_generic)
    - [`common_chat_params_init_functionary_v3_2`](#common_chat_params_init_functionary_v3_2)
    - [`common_chat_params_init_firefunction_v2`](#common_chat_params_init_firefunction_v2)
    - [`common_chat_params_init_functionary_v3_1_llama_3_1`](#common_chat_params_init_functionary_v3_1_llama_3_1)
    - [`common_chat_params_init_llama_3_x`](#common_chat_params_init_llama_3_x)
    - [`common_chat_params_init_without_tools`](#common_chat_params_init_without_tools)
    - [`common_chat_params_init_mistral_nemo`](#common_chat_params_init_mistral_nemo)


---
### common\_chat\_templates\_apply\_legacy<!-- {{#callable:common_chat_templates_apply_legacy}} -->
The function `common_chat_templates_apply_legacy` processes chat messages using a legacy template system, applying a template to generate a chat prompt and handling potential errors or buffer resizing.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` structure containing the chat templates to be applied.
    - `inputs`: A reference to a `common_chat_templates_inputs` structure containing the input messages and additional parameters for template application.
- **Control Flow**:
    - Initialize `alloc_size` to 0 and create vectors `chat` and `contents` to store processed messages and their contents.
    - Iterate over each message in `inputs.messages`, concatenating text parts and ignoring non-text parts, then store the result in `contents`.
    - For each message, create a `llama_chat_message` with the role and content, adding its size to `alloc_size`.
    - Allocate a buffer `buf` with size `alloc_size` to store the generated prompt.
    - Apply the template using `llama_chat_apply_template` to generate the prompt, checking for errors and resizing the buffer if necessary.
    - If the buffer is too small, resize it and reapply the template to ensure the prompt fits.
    - Create a `common_chat_params` object, setting its `prompt` to the generated prompt and its `grammar` based on `inputs.json_schema` or `inputs.grammar`.
    - Return the `common_chat_params` object.
- **Output**: A `common_chat_params` object containing the generated chat prompt and grammar settings.
- **Functions called**:
    - [`json_schema_to_grammar`](json-schema-to-grammar.cpp.driver.md#json_schema_to_grammar)


---
### common\_chat\_templates\_apply<!-- {{#callable:common_chat_templates_apply}} -->
The `common_chat_templates_apply` function applies a chat template to given inputs, choosing between Jinja or legacy template processing based on the `use_jinja` flag.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` structure containing the chat templates to be applied.
    - `inputs`: A reference to a `common_chat_templates_inputs` structure containing the inputs for the template application, including messages, tool choice, and other parameters.
- **Control Flow**:
    - The function asserts that `tmpls` is not null using `GGML_ASSERT`.
    - It checks the `use_jinja` flag in the `inputs` structure.
    - If `use_jinja` is true, it calls [`common_chat_templates_apply_jinja`](#common_chat_templates_apply_jinja) with `tmpls` and `inputs`.
    - If `use_jinja` is false, it calls [`common_chat_templates_apply_legacy`](#common_chat_templates_apply_legacy) with `tmpls` and `inputs`.
- **Output**: Returns a `common_chat_params` object containing the result of the template application, including the generated prompt and any applicable grammar.
- **Functions called**:
    - [`common_chat_templates_apply_jinja`](#common_chat_templates_apply_jinja)
    - [`common_chat_templates_apply_legacy`](#common_chat_templates_apply_legacy)


---
### common\_chat\_parse\_content\_only<!-- {{#callable:common_chat_parse_content_only}} -->
The function `common_chat_parse_content_only` adds the remaining content from the `common_chat_msg_parser` to the builder's content.
- **Inputs**:
    - `builder`: A reference to a `common_chat_msg_parser` object that is used to parse and build chat messages.
- **Control Flow**:
    - The function calls `builder.consume_rest()` to get the remaining content from the parser.
    - It then calls `builder.add_content()` with the consumed content to add it to the builder's content.
- **Output**: The function does not return any value; it modifies the `builder` object by adding the remaining content to it.


---
### common\_chat\_parse<!-- {{#callable:common_chat_parse}} -->
The [`common_chat_parse`](#common_chat_parse) function parses a chat message input string into a `common_chat_msg` object using a specified syntax and handles partial parsing exceptions.
- **Inputs**:
    - `input`: A `std::string` representing the chat message input to be parsed.
    - `is_partial`: A `bool` indicating whether the input is a partial message that may not be fully complete.
    - `syntax`: A `common_chat_syntax` object that defines the parsing rules and format for the chat message.
- **Control Flow**:
    - Initialize a `common_chat_msg_parser` object named `builder` with the input, is_partial flag, and syntax.
    - Attempt to parse the input using the [`common_chat_parse`](#common_chat_parse) function with the `builder` object.
    - Catch any `common_chat_msg_partial_exception` exceptions, log a debug message, and rethrow as a `std::runtime_error` if `is_partial` is false.
    - Retrieve the parsed message result from the `builder` object.
    - Log a debug message with the parsed message in JSON format.
- **Output**: Returns a `common_chat_msg` object representing the parsed chat message.
- **Functions called**:
    - [`common_chat_parse`](#common_chat_parse)


