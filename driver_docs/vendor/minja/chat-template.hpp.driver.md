# Purpose
This C++ source file defines a class [`chat_template`](#chat_templatechat_template) within the `minja` namespace, which is designed to handle and render chat templates with various capabilities and options. The file includes several standard and third-party libraries, such as `<chrono>`, `<string>`, and `nlohmann/json.hpp`, indicating its reliance on JSON data structures and time-related functionalities. The [`chat_template`](#chat_templatechat_template) class encapsulates the logic for managing chat templates, including their capabilities (`chat_template_caps`), inputs (`chat_template_inputs`), and options (`chat_template_options`). It provides methods to apply these templates to given inputs, potentially with polyfills to accommodate different template capabilities, such as support for tools, tool calls, and system roles.

The class is structured to support a variety of chat template features, such as handling different message roles (user, system, assistant), managing tool calls and responses, and applying polyfills to ensure compatibility with templates that may not natively support certain features. The [`apply`](#chat_templateapply) method is central to the class, allowing for the rendering of chat templates based on the provided inputs and options. The file also includes logic to determine the capabilities of a given template, such as whether it requires typed content or supports parallel tool calls. Overall, this file provides a comprehensive framework for managing and rendering chat templates, making it a crucial component for applications that require dynamic and flexible chat interactions.
# Imports and Dependencies

---
- `minja.hpp`
- `chrono`
- `cstddef`
- `cstdio`
- `ctime`
- `exception`
- `iomanip`
- `memory`
- `sstream`
- `stdexcept`
- `string`
- `vector`
- `nlohmann/json.hpp`


# Data Structures

---
### chat\_template\_caps<!-- {{#data_structure:minja::chat_template_caps}} -->
- **Type**: `struct`
- **Members**:
    - `supports_tools`: Indicates if the template supports tools.
    - `supports_tool_calls`: Indicates if the template supports tool calls.
    - `supports_tool_responses`: Indicates if the template supports tool responses.
    - `supports_system_role`: Indicates if the template supports a system role.
    - `supports_parallel_tool_calls`: Indicates if the template supports parallel tool calls.
    - `supports_tool_call_id`: Indicates if the template supports tool call IDs.
    - `requires_object_arguments`: Indicates if the template requires arguments to be objects.
    - `requires_non_null_content`: Indicates if the template requires non-null content.
    - `requires_typed_content`: Indicates if the template requires typed content.
- **Description**: The `chat_template_caps` struct is a configuration structure that defines various capabilities and requirements of a chat template system. It includes boolean flags that specify whether the template supports certain features such as tools, tool calls, tool responses, system roles, and parallel tool calls. Additionally, it indicates specific requirements like whether arguments need to be objects, content must be non-null, or content must be typed. This struct is used to configure and adapt the behavior of chat templates to different systems and requirements.


---
### chat\_template\_inputs<!-- {{#data_structure:minja::chat_template_inputs}} -->
- **Type**: `struct`
- **Members**:
    - `messages`: A JSON object to store chat messages in an ordered manner.
    - `tools`: A JSON object to store tools information in an ordered manner.
    - `add_generation_prompt`: A boolean flag indicating whether to add a generation prompt, defaulting to true.
    - `extra_context`: A JSON object to store additional context information in an ordered manner.
    - `now`: A time point representing the current system time, initialized to the current time.
- **Description**: The `chat_template_inputs` struct is designed to encapsulate the input parameters required for generating chat templates. It includes ordered JSON objects for messages, tools, and extra context, allowing for structured and prioritized data handling. Additionally, it features a boolean flag to control the inclusion of a generation prompt and a time point to capture the current system time, facilitating time-sensitive operations within the chat template generation process.


---
### chat\_template\_options<!-- {{#data_structure:minja::chat_template_options}} -->
- **Type**: `struct`
- **Members**:
    - `apply_polyfills`: Indicates whether polyfills should be applied, defaulting to true.
    - `use_bos_token`: Specifies if the beginning-of-sequence token should be used, defaulting to true.
    - `use_eos_token`: Specifies if the end-of-sequence token should be used, defaulting to true.
    - `define_strftime_now`: Determines if the current time should be defined using strftime, defaulting to true.
    - `polyfill_tools`: Indicates if tools should be polyfilled, defaulting to true.
    - `polyfill_tool_call_examples`: Specifies if tool call examples should be polyfilled, defaulting to true.
    - `polyfill_tool_calls`: Indicates if tool calls should be polyfilled, defaulting to true.
    - `polyfill_tool_responses`: Specifies if tool responses should be polyfilled, defaulting to true.
    - `polyfill_system_role`: Indicates if the system role should be polyfilled, defaulting to true.
    - `polyfill_object_arguments`: Specifies if object arguments should be polyfilled, defaulting to true.
    - `polyfill_typed_content`: Indicates if typed content should be polyfilled, defaulting to true.
- **Description**: The `chat_template_options` struct is a configuration data structure used to specify various options for applying polyfills and handling tokens in a chat template system. It includes boolean flags to control the application of polyfills for tools, tool calls, tool responses, system roles, object arguments, and typed content, as well as options for using beginning-of-sequence and end-of-sequence tokens and defining the current time with strftime. Each option has a default value of true, allowing for flexible customization of the chat template behavior.


---
### chat\_template<!-- {{#data_structure:minja::chat_template}} -->
- **Type**: `class`
- **Members**:
    - `caps_`: Stores the capabilities of the chat template, such as support for tools and system roles.
    - `source_`: Holds the source string for the chat template.
    - `bos_token_`: Represents the beginning-of-sequence token for the template.
    - `eos_token_`: Represents the end-of-sequence token for the template.
    - `template_root_`: Points to the root node of the parsed template structure.
    - `tool_call_example_`: Contains an example of a tool call syntax for the template.
- **Description**: The `chat_template` class is designed to manage and render chat templates, providing functionality to handle various chat capabilities such as tool support, system roles, and content typing. It uses a source string to parse and create a template structure, which can then be rendered with specific inputs and options. The class also determines the capabilities of the template, such as whether it supports tools, tool calls, and system roles, and provides methods to apply these templates with given inputs and options.
- **Member Functions**:
    - [`minja::chat_template::try_raw_render`](#chat_templatetry_raw_render)
    - [`minja::chat_template::chat_template`](#chat_templatechat_template)
    - [`minja::chat_template::source`](#chat_templatesource)
    - [`minja::chat_template::bos_token`](#chat_templatebos_token)
    - [`minja::chat_template::eos_token`](#chat_templateeos_token)
    - [`minja::chat_template::original_caps`](#chat_templateoriginal_caps)
    - [`minja::chat_template::apply`](#chat_templateapply)
    - [`minja::chat_template::apply`](#chat_templateapply)
    - [`minja::chat_template::add_system`](#chat_templateadd_system)

**Methods**

---
#### chat\_template::try\_raw\_render<!-- {{#callable:minja::chat_template::try_raw_render}} -->
The `try_raw_render` function attempts to render a chat template using provided messages, tools, and context, returning the generated prompt or an empty string if an exception occurs.
- **Inputs**:
    - `messages`: A JSON object containing the messages to be used in the chat template rendering.
    - `tools`: A JSON object containing the tools to be used in the chat template rendering.
    - `add_generation_prompt`: A boolean indicating whether to add a generation prompt to the chat template.
    - `extra_context`: An optional JSON object providing additional context for the chat template rendering, defaulting to an empty JSON object.
- **Control Flow**:
    - Initialize a `chat_template_inputs` object with the provided messages, tools, add_generation_prompt, and extra_context.
    - Set the `now` field of `inputs` to a fixed date for testing purposes.
    - Initialize a `chat_template_options` object with `apply_polyfills` set to false.
    - Call the [`apply`](#chat_templateapply) method with the `inputs` and `opts` to generate the prompt.
    - Return the generated prompt if successful.
    - Catch any exceptions and return an empty string if an error occurs.
- **Output**: A string representing the generated prompt from the chat template, or an empty string if an exception is caught.
- **Functions called**:
    - [`minja::chat_template::apply`](#chat_templateapply)
- **See also**: [`minja::chat_template`](#minjachat_template)  (Data Structure)


---
#### chat\_template::chat\_template<!-- {{#callable:minja::chat_template::chat_template}} -->
The `chat_template` constructor initializes a chat template object by parsing a source template and determining the capabilities of the template regarding tool calls, system roles, and content requirements.
- **Inputs**:
    - `source`: A string representing the source template to be parsed.
    - `bos_token`: A string representing the beginning-of-sequence token.
    - `eos_token`: A string representing the end-of-sequence token.
- **Control Flow**:
    - The constructor initializes member variables `source_`, `bos_token_`, and `eos_token_` with the provided arguments.
    - It parses the `source_` using `minja::Parser::parse` to create `template_root_`.
    - A lambda function [`contains`](minja.hpp.driver.md#Valuecontains) is defined to check if a string contains a substring.
    - Dummy user and system messages are created to test the template's capabilities.
    - The constructor checks if the template requires typed content by rendering dummy messages and checking for specific needles.
    - It determines if the template supports system roles by rendering a system message and checking for a system needle.
    - The constructor checks if the template supports tools by rendering a dummy tool and checking for its presence.
    - It defines helper functions `make_tool_calls_msg` and `make_tool_call` to create tool call messages.
    - The constructor tests if tool calls render string or object arguments and sets the corresponding capabilities.
    - It checks if the template supports parallel tool calls, tool responses, and tool call IDs by rendering various scenarios.
    - If the template does not support tools, it attempts to generate a tool call example by comparing rendered outputs.
    - The constructor handles exceptions by logging errors if tool call example generation fails.
- **Output**: The constructor does not return a value; it initializes the `chat_template` object and sets its capabilities based on the parsed template.
- **Functions called**:
    - [`minja::Value::contains`](minja.hpp.driver.md#Valuecontains)
    - [`minja::chat_template::try_raw_render`](#chat_templatetry_raw_render)
    - [`minja::chat_template::apply`](#chat_templateapply)
- **See also**: [`minja::chat_template`](#minjachat_template)  (Data Structure)


---
#### chat\_template::source<!-- {{#callable:minja::chat_template::source}} -->
The `source` function returns a constant reference to the `source_` string member of the `chat_template` class.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the `source_` member variable of the `chat_template` class.
- **Output**: A constant reference to the `source_` string member variable.
- **See also**: [`minja::chat_template`](#minjachat_template)  (Data Structure)


---
#### chat\_template::bos\_token<!-- {{#callable:minja::chat_template::bos_token}} -->
The `bos_token` function returns a constant reference to the beginning-of-sequence token string stored in the `chat_template` class.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter that directly returns the private member variable `bos_token_`.
- **Output**: A constant reference to a `std::string` representing the beginning-of-sequence token.
- **See also**: [`minja::chat_template`](#minjachat_template)  (Data Structure)


---
#### chat\_template::eos\_token<!-- {{#callable:minja::chat_template::eos_token}} -->
The `eos_token` function returns a constant reference to the `eos_token_` string member of the `chat_template` class.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter that directly returns the `eos_token_` member variable.
    - It does not perform any additional logic or checks.
- **Output**: A constant reference to the `eos_token_` string member of the `chat_template` class.
- **See also**: [`minja::chat_template`](#minjachat_template)  (Data Structure)


---
#### chat\_template::original\_caps<!-- {{#callable:minja::chat_template::original_caps}} -->
The `original_caps` function returns a constant reference to the `chat_template_caps` object, which holds the capabilities of the chat template.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter that returns the private member `caps_`.
- **Output**: A constant reference to the `chat_template_caps` object `caps_`.
- **See also**: [`minja::chat_template`](#minjachat_template)  (Data Structure)


---
#### chat\_template::apply<!-- {{#callable:minja::chat_template::apply}} -->
The [`apply`](#chat_templateapply) function is a deprecated method that constructs `chat_template_inputs` and `chat_template_options` from given parameters and calls another [`apply`](#chat_templateapply) method with these objects.
- **Inputs**:
    - `messages`: A JSON object containing the messages to be processed.
    - `tools`: A JSON object containing the tools available for processing the messages.
    - `add_generation_prompt`: A boolean indicating whether to add a generation prompt to the inputs.
    - `extra_context`: An optional JSON object providing additional context for the processing, defaulting to an empty JSON object.
    - `apply_polyfills`: A boolean indicating whether to apply polyfills, defaulting to true.
- **Control Flow**:
    - Prints a deprecation warning to standard error.
    - Initializes a `chat_template_inputs` object with the provided parameters and the current system time.
    - Initializes a `chat_template_options` object with the `apply_polyfills` parameter.
    - Calls the overloaded [`apply`](#chat_templateapply) method with the constructed `inputs` and `opts` objects.
- **Output**: Returns a string result from the overloaded [`apply`](#chat_templateapply) method.
- **Functions called**:
    - [`minja::chat_template::apply`](#chat_templateapply)
- **See also**: [`minja::chat_template`](#minjachat_template)  (Data Structure)


---
#### chat\_template::apply<!-- {{#callable:minja::chat_template::apply}} -->
The `apply` function processes chat template inputs and options to generate a rendered string output, potentially applying various polyfills to the input messages based on the capabilities and options provided.
- **Inputs**:
    - `inputs`: An instance of `chat_template_inputs` containing messages, tools, a flag for adding a generation prompt, extra context, and a timestamp.
    - `opts`: An instance of [`chat_template_options`](#minjachat_template_options) specifying various polyfill and token options, with default values if not provided.
- **Control Flow**:
    - Initialize `actual_messages` as an empty JSON array.
    - Determine the presence of tools, tool calls, tool responses, and string content in the input messages.
    - Evaluate the need for various polyfills based on the input options and the capabilities of the chat template.
    - If polyfills are needed, adjust the messages by applying necessary transformations such as converting string content to typed content, handling tool calls, and adjusting roles.
    - If no polyfills are needed, use the input messages directly.
    - Create a context using the adjusted messages and set various context parameters based on the options and inputs.
    - Render the template using the context and return the resulting string.
- **Output**: A string representing the rendered output of the chat template, potentially modified by polyfills and context settings.
- **Functions called**:
    - [`minja::chat_template_options`](#minjachat_template_options)
    - [`minja::chat_template::add_system`](#chat_templateadd_system)
- **See also**: [`minja::chat_template`](#minjachat_template)  (Data Structure)


---
#### chat\_template::add\_system<!-- {{#callable:minja::chat_template::add_system}} -->
The `add_system` function inserts or updates a system message in a JSON array of messages, appending a new system prompt if a system message already exists.
- **Inputs**:
    - `messages`: A JSON array of messages, where each message is expected to have a 'role' and 'content' field.
    - `system_prompt`: A string representing the system prompt to be added or appended to the existing system message.
- **Control Flow**:
    - Initialize `messages_with_system` as a copy of `messages`.
    - Check if `messages_with_system` is not empty and the first message has the role 'system'.
    - If true, append `system_prompt` to the existing system message's content with a newline separator.
    - If false, insert a new system message with the `system_prompt` at the beginning of `messages_with_system`.
- **Output**: Returns a JSON array of messages with the system prompt added or updated.
- **See also**: [`minja::chat_template`](#minjachat_template)  (Data Structure)



