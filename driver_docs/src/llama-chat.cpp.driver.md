# Purpose
This C++ source code file is designed to handle chat templates for a language model, specifically focusing on the detection and application of various chat templates. The file includes a comprehensive mapping of string identifiers to specific chat template constants, which are used to identify and apply the correct template format to chat messages. The code provides functions to detect the template type from a given string and to apply the detected template to a series of chat messages, formatting them according to the rules of the identified template. This functionality is crucial for ensuring that chat messages are correctly structured and formatted for processing by different language models.

The file includes several key components: a static function to trim whitespace from strings, a map of template names to template constants, and functions to detect and apply chat templates. The [`llm_chat_detect_template`](#llm_chat_detect_template) function uses heuristic checks to determine the appropriate template based on the presence of specific markers in the input string. The [`llm_chat_apply_template`](#llm_chat_apply_template) function formats chat messages according to the detected template, supporting a wide range of templates such as "chatml," "mistral," "llama," and others. Additionally, the file provides a public interface function, [`llama_chat_builtin_templates`](#llama_chat_builtin_templates), which returns the available built-in template names. This code is intended to be part of a larger system, likely a library or module, that processes chat interactions for language models, providing a standardized way to handle different chat formats.
# Imports and Dependencies

---
- `llama-chat.h`
- `llama.h`
- `map`
- `sstream`
- `algorithm`


# Global Variables

---
### LLM\_CHAT\_TEMPLATES
- **Type**: ``std::map<std::string, llm_chat_template>``
- **Description**: `LLM_CHAT_TEMPLATES` is a static constant map that associates string keys with `llm_chat_template` values. Each key represents a specific chat template name, and the corresponding value is a predefined template used for chat interactions in the system.
- **Use**: This variable is used to retrieve specific chat templates based on their string identifiers for use in chat applications.


# Functions

---
### trim<!-- {{#callable:trim}} -->
The `trim` function removes leading and trailing whitespace from a given string.
- **Inputs**:
    - `str`: A constant reference to a `std::string` from which leading and trailing whitespace will be removed.
- **Control Flow**:
    - Initialize `start` to 0 and `end` to the size of the input string `str`.
    - Iterate from the beginning of the string, incrementing `start` while the current character is a whitespace character.
    - Iterate from the end of the string, decrementing `end` while the character before `end` is a whitespace character.
    - Return a substring of `str` from `start` to `end`, effectively removing leading and trailing whitespace.
- **Output**: A `std::string` that is a copy of the input string `str` with leading and trailing whitespace removed.


---
### llm\_chat\_template\_from\_str<!-- {{#callable:llm_chat_template_from_str}} -->
The function `llm_chat_template_from_str` retrieves a chat template from a predefined map using a given template name.
- **Inputs**:
    - `name`: A string representing the name of the chat template to retrieve.
- **Control Flow**:
    - The function accesses the `LLM_CHAT_TEMPLATES` map using the `at` method with the provided `name` as the key.
    - If the `name` exists in the map, the corresponding `llm_chat_template` is returned.
    - If the `name` does not exist, an `std::out_of_range` exception is thrown.
- **Output**: Returns the `llm_chat_template` associated with the given `name` from the `LLM_CHAT_TEMPLATES` map.


---
### llm\_chat\_detect\_template<!-- {{#callable:llm_chat_detect_template}} -->
The `llm_chat_detect_template` function identifies and returns the appropriate chat template type based on the content of the input string `tmpl`.
- **Inputs**:
    - `tmpl`: A constant reference to a `std::string` representing the template string to be analyzed.
- **Control Flow**:
    - Attempt to convert the input string `tmpl` to a chat template using [`llm_chat_template_from_str`](#llm_chat_template_from_str); if successful, return the result.
    - If the conversion throws an `std::out_of_range` exception, ignore it and proceed with further checks.
    - Define a lambda function `tmpl_contains` to check if a substring exists within `tmpl`.
    - Check for specific substrings or patterns within `tmpl` to determine the appropriate chat template type, using a series of conditional statements.
    - Return the corresponding chat template constant based on the detected pattern.
    - If no known pattern is detected, return `LLM_CHAT_TEMPLATE_UNKNOWN`.
- **Output**: Returns an `llm_chat_template` enumeration value that corresponds to the detected chat template type, or `LLM_CHAT_TEMPLATE_UNKNOWN` if no known template is detected.
- **Functions called**:
    - [`llm_chat_template_from_str`](#llm_chat_template_from_str)


---
### llm\_chat\_apply\_template<!-- {{#callable:llm_chat_apply_template}} -->
The `llm_chat_apply_template` function formats a sequence of chat messages according to a specified template and appends the result to a destination string.
- **Inputs**:
    - `tmpl`: An `llm_chat_template` enum value indicating the template to be applied to the chat messages.
    - `chat`: A vector of pointers to `llama_chat_message` objects, each containing a role and content for a chat message.
    - `dest`: A reference to a `std::string` where the formatted chat messages will be stored.
    - `add_ass`: A boolean flag indicating whether to append an assistant prompt at the end of the formatted messages.
- **Control Flow**:
    - Initialize a `std::stringstream` to build the formatted chat messages.
    - Check the template type (`tmpl`) and apply the corresponding formatting rules to each message in the `chat` vector.
    - For each message, extract the role and content, and format them according to the template's specifications.
    - If `add_ass` is true, append an assistant prompt to the formatted messages.
    - Store the formatted result in the `dest` string.
    - Return the size of the `dest` string.
- **Output**: The function returns an `int32_t` representing the size of the formatted string stored in `dest`, or -1 if the template is not supported.
- **Functions called**:
    - [`trim`](#trim)


---
### llama\_chat\_builtin\_templates<!-- {{#callable:llama_chat_builtin_templates}} -->
The function `llama_chat_builtin_templates` populates an array with the names of built-in chat templates and returns the total number of such templates.
- **Inputs**:
    - `output`: A pointer to an array of C-style strings where the names of the chat templates will be stored.
    - `len`: The maximum number of template names to store in the output array.
- **Control Flow**:
    - Initialize an iterator to the beginning of the `LLM_CHAT_TEMPLATES` map.
    - Iterate over the map up to the minimum of `len` and the size of the map.
    - For each iteration, store the current template name in the `output` array and advance the iterator.
    - Return the total number of templates in the `LLM_CHAT_TEMPLATES` map.
- **Output**: The function returns the total number of chat templates available in the `LLM_CHAT_TEMPLATES` map as an `int32_t`.


