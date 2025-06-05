# Purpose
This C++ header file provides a comprehensive framework for handling chat support, including tool call grammar constraining and output parsing, with both generic and custom template handlers. The file defines several key structures and enumerations that facilitate the management of chat messages, tool calls, and chat formats. The primary components include `common_chat_msg`, which represents a chat message with various attributes such as role, content, and associated tool calls, and `common_chat_tool`, which describes tools that can be invoked during a chat session. The file also defines enumerations like `common_chat_tool_choice` and `common_chat_format` to manage tool selection and chat formatting options, respectively.

Additionally, the file provides a set of functions and templates for parsing and formatting chat messages and tools, particularly in compatibility with OpenAI's chat completion API format. Functions such as `common_chat_format_single` and `common_chat_format_example` are used to format chat messages, while `common_chat_parse` and `common_chat_msgs_parse_oaicompat` handle parsing of input data. The file also includes utility functions for managing chat templates, such as `common_chat_templates_init` and `common_chat_templates_apply`, which initialize and apply chat templates to input data. Overall, this header file serves as a foundational component for building chat applications that require structured message handling and tool integration.
# Imports and Dependencies

---
- `common.h`
- `functional`
- `chrono`
- `string`
- `vector`


# Global Variables

---
### common\_chat\_templates\_source
- **Type**: `const char*`
- **Description**: The `common_chat_templates_source` is a function that returns a constant character pointer. It takes a pointer to a `common_chat_templates` structure and an optional character pointer `variant` as parameters. This function is likely used to retrieve a source or identifier related to the chat templates, possibly for logging or debugging purposes.
- **Use**: This function is used to obtain a source string associated with a given set of chat templates, optionally modified by a variant.


---
### common\_chat\_format\_name
- **Type**: `function`
- **Description**: The `common_chat_format_name` is a function that takes a `common_chat_format` enumeration value as an argument and returns a constant character pointer to the name of the format.
- **Use**: This function is used to retrieve the string representation of a chat format based on its enumeration value.


---
### common\_reasoning\_format\_name
- **Type**: `const char*`
- **Description**: The `common_reasoning_format_name` is a function that takes a `common_reasoning_format` enumeration value as an argument and returns a constant character pointer representing the name of the reasoning format.
- **Use**: This function is used to obtain a human-readable name for a given reasoning format, which can be useful for logging, debugging, or displaying format information to users.


# Data Structures

---
### common\_chat\_tool\_call<!-- {{#data_structure:common_chat_tool_call}} -->
- **Type**: `struct`
- **Members**:
    - `name`: A string representing the name of the tool call.
    - `arguments`: A string containing the arguments for the tool call.
    - `id`: A string serving as the unique identifier for the tool call.
- **Description**: The `common_chat_tool_call` struct is a data structure used to represent a tool call within a chat system. It contains three string members: `name`, which specifies the name of the tool being called; `arguments`, which holds any arguments that need to be passed to the tool; and `id`, which uniquely identifies the tool call instance. This struct also includes an equality operator to compare two instances of `common_chat_tool_call` for equivalence based on these three fields.
- **Member Functions**:
    - [`common_chat_tool_call::operator==`](#common_chat_tool_calloperator==)

**Methods**

---
#### common\_chat\_tool\_call::operator==<!-- {{#callable:common_chat_tool_call::operator==}} -->
The `operator==` function compares two `common_chat_tool_call` objects for equality based on their `name`, `arguments`, and `id` attributes.
- **Inputs**:
    - `other`: A reference to another `common_chat_tool_call` object to compare against the current object.
- **Control Flow**:
    - The function checks if the `name` attribute of the current object is equal to the `name` attribute of the `other` object.
    - It then checks if the `arguments` attribute of the current object is equal to the `arguments` attribute of the `other` object.
    - Finally, it checks if the `id` attribute of the current object is equal to the `id` attribute of the `other` object.
    - The function returns `true` if all three attributes are equal, otherwise it returns `false`.
- **Output**: A boolean value indicating whether the two `common_chat_tool_call` objects are equal.
- **See also**: [`common_chat_tool_call`](#common_chat_tool_call)  (Data Structure)



---
### common\_chat\_msg\_content\_part<!-- {{#data_structure:common_chat_msg_content_part}} -->
- **Type**: `struct`
- **Members**:
    - `type`: A string representing the type of the chat message content part.
    - `text`: A string containing the text of the chat message content part.
- **Description**: The `common_chat_msg_content_part` struct is a simple data structure used to represent a part of a chat message's content, consisting of a type and the actual text. It includes an equality operator to compare two instances of this struct, which checks if both the type and text are identical between the two instances.
- **Member Functions**:
    - [`common_chat_msg_content_part::operator==`](#common_chat_msg_content_partoperator==)

**Methods**

---
#### common\_chat\_msg\_content\_part::operator==<!-- {{#callable:common_chat_msg_content_part::operator==}} -->
The `operator==` function compares two `common_chat_msg_content_part` objects for equality based on their `type` and `text` attributes.
- **Inputs**:
    - `other`: A reference to another `common_chat_msg_content_part` object to compare against the current object.
- **Control Flow**:
    - The function checks if the `type` attribute of the current object is equal to the `type` attribute of the `other` object.
    - It then checks if the `text` attribute of the current object is equal to the `text` attribute of the `other` object.
    - The function returns `true` if both the `type` and `text` attributes are equal; otherwise, it returns `false`.
- **Output**: A boolean value indicating whether the two `common_chat_msg_content_part` objects are equal.
- **See also**: [`common_chat_msg_content_part`](#common_chat_msg_content_part)  (Data Structure)



---
### common\_chat\_msg<!-- {{#data_structure:common_chat_msg}} -->
- **Type**: `struct`
- **Members**:
    - `role`: Represents the role of the message sender, such as 'assistant' or 'user'.
    - `content`: Holds the main text content of the chat message.
    - `content_parts`: A vector of content parts, each represented by a common_chat_msg_content_part, for more granular message content.
    - `tool_calls`: A vector of tool calls, each represented by a common_chat_tool_call, indicating tools invoked by the message.
    - `reasoning_content`: Contains additional reasoning or explanation content related to the message.
    - `tool_name`: Specifies the name of the tool associated with the message.
    - `tool_call_id`: Stores the identifier for the tool call associated with the message.
- **Description**: The `common_chat_msg` struct is designed to encapsulate a chat message within a system that supports tool calls and reasoning content. It includes fields for the role of the message sender, the main content of the message, and additional content parts for more detailed message breakdowns. The struct also supports tool calls, allowing for the integration of external tools within the chat, and includes fields for reasoning content and tool identifiers. This structure is essential for managing complex chat interactions that involve multiple components and external tool integrations.
- **Member Functions**:
    - [`common_chat_msg::empty`](#common_chat_msgempty)
    - [`common_chat_msg::ensure_tool_call_ids_set`](#common_chat_msgensure_tool_call_ids_set)
    - [`common_chat_msg::operator==`](#common_chat_msgoperator==)
    - [`common_chat_msg::operator!=`](#common_chat_msgoperator!=)
    - [`common_chat_msg::to_json_oaicompat`](chat.cpp.driver.md#common_chat_msgto_json_oaicompat)

**Methods**

---
#### common\_chat\_msg::empty<!-- {{#callable:common_chat_msg::empty}} -->
The `empty` function checks if all the attributes of a `common_chat_msg` object are empty.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the `content` string is empty.
    - It checks if the `content_parts` vector is empty.
    - It checks if the `tool_calls` vector is empty.
    - It checks if the `reasoning_content` string is empty.
    - It checks if the `tool_name` string is empty.
    - It checks if the `tool_call_id` string is empty.
    - The function returns true if all these checks are true, otherwise it returns false.
- **Output**: A boolean value indicating whether all the attributes of the `common_chat_msg` object are empty.
- **See also**: [`common_chat_msg`](#common_chat_msg)  (Data Structure)


---
#### common\_chat\_msg::ensure\_tool\_call\_ids\_set<!-- {{#callable:common_chat_msg::ensure_tool_call_ids_set}} -->
The `ensure_tool_call_ids_set` function ensures that each tool call in a `common_chat_msg` has a unique identifier, generating new IDs if necessary, and caches them for future use.
- **Inputs**:
    - `ids_cache`: A reference to a vector of strings that stores cached tool call IDs.
    - `gen_tool_call_id`: A function that generates a new tool call ID as a string.
- **Control Flow**:
    - Iterate over each tool call in the `tool_calls` vector of the `common_chat_msg` object.
    - Check if the current index `i` is greater than or equal to the size of `ids_cache`.
    - If true, check if the tool call's ID is empty; if so, generate a new ID using `gen_tool_call_id`.
    - Push the generated or existing ID to `ids_cache`.
    - Assign the ID from `ids_cache` to the tool call's ID.
- **Output**: The function does not return a value; it modifies the `ids_cache` and the `tool_calls` vector in place.
- **See also**: [`common_chat_msg`](#common_chat_msg)  (Data Structure)


---
#### common\_chat\_msg::operator==<!-- {{#callable:common_chat_msg::operator==}} -->
The `operator==` function compares two `common_chat_msg` objects for equality by checking if all their corresponding attributes are equal.
- **Inputs**:
    - `other`: A reference to another `common_chat_msg` object to compare against the current object.
- **Control Flow**:
    - The function checks if the `role` attribute of the current object is equal to the `role` attribute of the `other` object.
    - It checks if the `content` attribute of the current object is equal to the `content` attribute of the `other` object.
    - It compares the `content_parts` vector of the current object with the `content_parts` vector of the `other` object for equality.
    - It compares the `tool_calls` vector of the current object with the `tool_calls` vector of the `other` object for equality.
    - It checks if the `reasoning_content` attribute of the current object is equal to the `reasoning_content` attribute of the `other` object.
    - It checks if the `tool_name` attribute of the current object is equal to the `tool_name` attribute of the `other` object.
    - It checks if the `tool_call_id` attribute of the current object is equal to the `tool_call_id` attribute of the `other` object.
- **Output**: A boolean value indicating whether the two `common_chat_msg` objects are equal (true) or not (false).
- **See also**: [`common_chat_msg`](#common_chat_msg)  (Data Structure)


---
#### common\_chat\_msg::operator\!=<!-- {{#callable:common_chat_msg::operator!=}} -->
The `operator!=` function checks if two `common_chat_msg` objects are not equal by negating the result of the equality operator.
- **Inputs**:
    - `other`: A reference to another `common_chat_msg` object to compare against the current object.
- **Control Flow**:
    - The function calls the equality operator `operator==` to compare the current object with the `other` object.
    - It negates the result of the equality operator to determine if the objects are not equal.
- **Output**: A boolean value indicating whether the two `common_chat_msg` objects are not equal.
- **See also**: [`common_chat_msg`](#common_chat_msg)  (Data Structure)



---
### common\_chat\_msg\_diff<!-- {{#data_structure:common_chat_msg_diff}} -->
- **Type**: `struct`
- **Members**:
    - `reasoning_content_delta`: Stores the difference in reasoning content between two chat messages.
    - `content_delta`: Holds the difference in main content between two chat messages.
    - `tool_call_index`: Indicates the index of the tool call that has changed, defaulting to `std::string::npos` if no change.
    - `tool_call_delta`: Represents the difference in tool call details between two chat messages.
- **Description**: The `common_chat_msg_diff` struct is designed to capture and represent the differences between two chat messages, specifically focusing on changes in reasoning content, main content, and tool call details. It includes fields to store deltas for reasoning content and main content, as well as an index and delta for tool calls. This struct is useful for tracking modifications in chat interactions, particularly in systems where tool calls and content reasoning are integral to the chat's functionality. The `compute_diffs` static method facilitates the comparison of two `common_chat_msg` instances to generate a list of `common_chat_msg_diff` instances, highlighting the specific changes between them.
- **Member Functions**:
    - [`common_chat_msg_diff::operator==`](#common_chat_msg_diffoperator==)
    - [`common_chat_msg_diff::compute_diffs`](chat.cpp.driver.md#common_chat_msg_diffcompute_diffs)

**Methods**

---
#### common\_chat\_msg\_diff::operator==<!-- {{#callable:common_chat_msg_diff::operator==}} -->
The `operator==` function compares two `common_chat_msg_diff` objects for equality based on their `content_delta`, `tool_call_index`, and `tool_call_delta` attributes.
- **Inputs**:
    - `other`: A `common_chat_msg_diff` object to compare against the current object.
- **Control Flow**:
    - The function checks if the `content_delta` of the current object is equal to the `content_delta` of the `other` object.
    - It then checks if the `tool_call_index` of the current object is equal to the `tool_call_index` of the `other` object.
    - Finally, it checks if the `tool_call_delta` of the current object is equal to the `tool_call_delta` of the `other` object.
    - If all three conditions are true, the function returns `true`; otherwise, it returns `false`.
- **Output**: A boolean value indicating whether the two `common_chat_msg_diff` objects are equal.
- **See also**: [`common_chat_msg_diff`](#common_chat_msg_diff)  (Data Structure)



---
### common\_chat\_tool<!-- {{#data_structure:common_chat_tool}} -->
- **Type**: `struct`
- **Members**:
    - `name`: A string representing the name of the chat tool.
    - `description`: A string providing a description of the chat tool.
    - `parameters`: A string detailing the parameters associated with the chat tool.
- **Description**: The `common_chat_tool` struct is a simple data structure used to encapsulate information about a chat tool, including its name, a description, and any parameters it may require. This struct is likely used within a larger chat system to manage and utilize various chat tools, providing a standardized way to store and access tool-related information.


---
### common\_chat\_tool\_choice<!-- {{#data_structure:common_chat_tool_choice}} -->
- **Type**: `enum`
- **Members**:
    - `COMMON_CHAT_TOOL_CHOICE_AUTO`: Represents an automatic choice for the chat tool.
    - `COMMON_CHAT_TOOL_CHOICE_REQUIRED`: Indicates that the chat tool is required.
    - `COMMON_CHAT_TOOL_CHOICE_NONE`: Specifies that no chat tool is chosen.
- **Description**: The `common_chat_tool_choice` enum defines three possible states for selecting a chat tool: automatic selection, mandatory requirement, or no selection. This enum is used to manage the behavior of chat tools within the chat support system, allowing for flexible configuration based on the needs of the application.


---
### common\_chat\_format<!-- {{#data_structure:common_chat_format}} -->
- **Type**: `enum`
- **Members**:
    - `COMMON_CHAT_FORMAT_CONTENT_ONLY`: Represents a chat format that includes only the content.
    - `COMMON_CHAT_FORMAT_GENERIC`: Represents a generic chat format.
    - `COMMON_CHAT_FORMAT_MISTRAL_NEMO`: Represents the Mistral Nemo chat format.
    - `COMMON_CHAT_FORMAT_LLAMA_3_X`: Represents the Llama 3.x chat format.
    - `COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS`: Represents the Llama 3.x chat format with built-in tools.
    - `COMMON_CHAT_FORMAT_DEEPSEEK_R1`: Represents the DeepSeek R1 chat format.
    - `COMMON_CHAT_FORMAT_FIREFUNCTION_V2`: Represents the FireFunction V2 chat format.
    - `COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2`: Represents the Functionary V3.2 chat format.
    - `COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1`: Represents the Functionary V3.1 chat format with Llama 3.1.
    - `COMMON_CHAT_FORMAT_HERMES_2_PRO`: Represents the Hermes 2 Pro chat format.
    - `COMMON_CHAT_FORMAT_COMMAND_R7B`: Represents the Command R7B chat format.
    - `COMMON_CHAT_FORMAT_COUNT`: Indicates the number of chat formats, not an actual format.
- **Description**: The `common_chat_format` enum defines a set of constants representing different chat formats used in a chat system. Each constant corresponds to a specific format, such as content-only, generic, or various named formats like Mistral Nemo or Llama 3.x. The enum also includes a special constant, `COMMON_CHAT_FORMAT_COUNT`, which is used to indicate the total number of formats defined, rather than representing an actual format.


---
### common\_chat\_templates\_inputs<!-- {{#data_structure:common_chat_templates_inputs}} -->
- **Type**: `struct`
- **Members**:
    - `messages`: A vector of common_chat_msg objects representing chat messages.
    - `grammar`: A string representing the grammar used in chat templates.
    - `json_schema`: A string representing the JSON schema for chat templates.
    - `add_generation_prompt`: A boolean indicating whether to add a generation prompt, defaulting to true.
    - `use_jinja`: A boolean indicating whether to use Jinja templates, defaulting to true.
    - `tools`: A vector of common_chat_tool objects, used only if use_jinja is true.
    - `tool_choice`: An enum value of type common_chat_tool_choice, defaulting to COMMON_CHAT_TOOL_CHOICE_AUTO.
    - `parallel_tool_calls`: A boolean indicating if tool calls should be parallel, defaulting to false.
    - `reasoning_format`: An enum value of type common_reasoning_format, defaulting to COMMON_REASONING_FORMAT_NONE.
    - `enable_thinking`: A boolean indicating if thinking is enabled, defaulting to true.
    - `now`: A time_point representing the current time, defaulting to the system's current time.
- **Description**: The `common_chat_templates_inputs` struct is designed to encapsulate the input parameters for chat template processing, including message data, grammar, and JSON schema. It supports optional Jinja template usage, with additional parameters for tool integration and reasoning format. The struct also includes settings for generation prompts, parallel tool calls, and a timestamp for the current time, making it a comprehensive configuration holder for chat template operations.


---
### common\_chat\_params<!-- {{#data_structure:common_chat_params}} -->
- **Type**: `struct`
- **Members**:
    - `format`: Specifies the chat format, defaulting to COMMON_CHAT_FORMAT_CONTENT_ONLY.
    - `prompt`: A string representing the initial prompt for the chat.
    - `grammar`: A string defining the grammar rules for the chat.
    - `grammar_lazy`: A boolean indicating if grammar checking is performed lazily.
    - `thinking_forced_open`: A boolean indicating if thinking is forced to be open.
    - `grammar_triggers`: A vector of common_grammar_trigger objects that define triggers for grammar rules.
    - `preserved_tokens`: A vector of strings representing tokens that should be preserved during processing.
    - `additional_stops`: A vector of strings representing additional stop words for the chat.
- **Description**: The `common_chat_params` struct is designed to encapsulate various parameters and settings for configuring a chat session. It includes options for specifying the chat format, initial prompt, and grammar rules, as well as flags for lazy grammar checking and forced open thinking. Additionally, it supports defining grammar triggers, preserved tokens, and additional stop words to customize the chat behavior.


---
### common\_chat\_syntax<!-- {{#data_structure:common_chat_syntax}} -->
- **Type**: `struct`
- **Members**:
    - `format`: Specifies the chat format, defaulting to COMMON_CHAT_FORMAT_CONTENT_ONLY.
    - `reasoning_format`: Specifies the reasoning format, defaulting to COMMON_REASONING_FORMAT_NONE.
    - `reasoning_in_content`: Indicates if reasoning content should be inlined in the main content.
    - `thinking_forced_open`: Indicates if thinking is forced to be open.
    - `parse_tool_calls`: Determines if tool calls should be parsed, defaulting to true.
- **Description**: The `common_chat_syntax` struct defines the configuration for chat syntax, including the format of the chat, the reasoning format, and various boolean flags that control whether reasoning content is inlined, if thinking is forced open, and if tool calls should be parsed. This struct is used to manage and configure the behavior of chat interactions, particularly in contexts where different formats and reasoning styles are required.


---
### common\_chat\_templates\_deleter<!-- {{#data_structure:common_chat_templates_deleter}} -->
- **Type**: `struct`
- **Description**: The `common_chat_templates_deleter` is a custom deleter struct designed to manage the memory of `common_chat_templates` objects. It provides an overloaded `operator()` that takes a pointer to a `common_chat_templates` object and calls the `common_chat_templates_free` function to properly release the resources associated with the object. This struct is typically used in conjunction with smart pointers, such as `std::unique_ptr`, to ensure that `common_chat_templates` objects are automatically and safely deleted when they go out of scope.
- **Member Functions**:
    - [`common_chat_templates_deleter::operator()`](#common_chat_templates_deleteroperator())

**Methods**

---
#### common\_chat\_templates\_deleter::operator\(\)<!-- {{#callable:common_chat_templates_deleter::operator()}} -->
The `operator()` function in `common_chat_templates_deleter` is a custom deleter that frees memory allocated for `common_chat_templates` objects.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` object that needs to be freed.
- **Control Flow**:
    - The function takes a pointer to a `common_chat_templates` object as an argument.
    - It calls the [`common_chat_templates_free`](chat.cpp.driver.md#common_chat_templates_free) function, passing the pointer to it, which handles the deallocation of the memory.
- **Output**: The function does not return any value; it performs a side effect by freeing the memory of the `common_chat_templates` object.
- **Functions called**:
    - [`common_chat_templates_free`](chat.cpp.driver.md#common_chat_templates_free)
- **See also**: [`common_chat_templates_deleter`](#common_chat_templates_deleter)  (Data Structure)



