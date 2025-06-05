# Purpose
The provided C++ source code is part of a server application designed to handle various tasks related to natural language processing (NLP) and machine learning models. The code is structured to manage tasks such as text completion, embedding generation, and reranking, using a server-client architecture. It includes functionalities for handling HTTP requests, processing tasks in parallel using slots, and managing server state and metrics.

Key components of the code include the definition of various task types and server states, the use of JSON for data interchange, and the integration of a machine learning model for processing tasks. The code is organized into several structures and functions that manage tasks, slots, and server responses. It also includes error handling and logging mechanisms to ensure robust operation. The server is designed to be extensible, with support for additional functionalities such as multimodal processing and LoRA adapters for model fine-tuning. The code is intended to be part of a larger system that provides NLP services over HTTP, with support for OpenAI-compatible APIs and other custom endpoints.
# Imports and Dependencies

---
- `chat.h`
- `utils.hpp`
- `arg.h`
- `common.h`
- `json-schema-to-grammar.h`
- `llama.h`
- `log.h`
- `sampling.h`
- `speculative.h`
- `mtmd.h`
- `mtmd-helper.h`
- `index.html.gz.hpp`
- `loading.html.hpp`
- `atomic`
- `chrono`
- `condition_variable`
- `cstddef`
- `cinttypes`
- `deque`
- `memory`
- `mutex`
- `signal.h`
- `thread`
- `unordered_map`
- `unordered_set`


# Global Variables

---
### HTTP\_POLLING\_SECONDS
- **Type**: ``int``
- **Description**: `HTTP_POLLING_SECONDS` is a global constant integer variable defined using `constexpr`, which means its value is known at compile time and cannot be changed during runtime. It is set to the value `1`. This variable is likely used to specify a time interval in seconds for polling operations, such as checking for updates or waiting for a response in a network communication context.
- **Use**: This variable is used to define the polling interval in seconds for HTTP operations, ensuring consistent timing for network requests or responses.


---
### shutdown\_handler
- **Type**: `std::function<void(int)>`
- **Description**: The `shutdown_handler` is a global variable of type `std::function<void(int)>`, which is a function object that can store, copy, and invoke any callable target (like a function, lambda expression, or bind expression) that matches the signature `void(int)`. This means it can hold a function that takes an integer as an argument and returns nothing.
- **Use**: This variable is used to store a function that handles shutdown signals, allowing the program to define custom behavior when a shutdown signal is received.


---
### is\_terminating
- **Type**: `std::atomic_flag`
- **Description**: The `is_terminating` variable is a global atomic flag used to indicate whether the server is in the process of terminating. It is initialized with `ATOMIC_FLAG_INIT`, which sets the flag to a clear state.
- **Use**: This variable is used to manage the server's shutdown process, ensuring that termination signals are handled correctly and preventing multiple termination attempts.


# Data Structures

---
### stop\_type<!-- {{#data_structure:stop_type}} -->
- **Type**: `enum`
- **Members**:
    - `STOP_TYPE_NONE`: Represents no stop condition.
    - `STOP_TYPE_EOS`: Represents an end-of-sequence stop condition.
    - `STOP_TYPE_WORD`: Represents a stop condition based on a specific word.
    - `STOP_TYPE_LIMIT`: Represents a stop condition based on a predefined limit.
- **Description**: The `stop_type` enum defines various conditions under which a process or operation should be stopped. It includes options for no stop condition, stopping at the end of a sequence, stopping when a specific word is encountered, and stopping when a predefined limit is reached. This enum is useful for controlling the flow of operations that require specific stopping criteria.


---
### slot\_state<!-- {{#data_structure:slot_state}} -->
- **Type**: `enum`
- **Members**:
    - `SLOT_STATE_IDLE`: Represents the idle state of a slot.
    - `SLOT_STATE_STARTED`: Indicates that the slot has started, primarily for initial prompt processing.
    - `SLOT_STATE_PROCESSING_PROMPT`: Denotes that the slot is currently processing a prompt.
    - `SLOT_STATE_DONE_PROMPT`: Indicates that the prompt processing is completed.
    - `SLOT_STATE_GENERATING`: Represents the state where the slot is generating output.
- **Description**: The `slot_state` enum defines the various states a slot can be in during its lifecycle. It includes states for when a slot is idle, has started processing, is actively processing a prompt, has completed prompt processing, and is generating output. This enum is used to manage and track the state transitions of slots in a system that processes tasks or requests.


---
### server\_state<!-- {{#data_structure:server_state}} -->
- **Type**: `enum`
- **Members**:
    - `SERVER_STATE_LOADING_MODEL`: Indicates that the server is starting up and the model is not fully loaded yet.
    - `SERVER_STATE_READY`: Indicates that the server is ready and the model is fully loaded.
- **Description**: The `server_state` enum represents the operational state of a server in a machine learning context. It has two states: `SERVER_STATE_LOADING_MODEL`, which indicates that the server is in the process of starting up and the model is not yet fully loaded, and `SERVER_STATE_READY`, which indicates that the server is fully operational and the model is loaded and ready for use. This enum is used to manage and track the server's readiness to handle requests.


---
### server\_task\_type<!-- {{#data_structure:server_task_type}} -->
- **Type**: `enum`
- **Members**:
    - `SERVER_TASK_TYPE_COMPLETION`: Represents a task type for completion.
    - `SERVER_TASK_TYPE_EMBEDDING`: Represents a task type for embedding.
    - `SERVER_TASK_TYPE_RERANK`: Represents a task type for reranking.
    - `SERVER_TASK_TYPE_INFILL`: Represents a task type for infill.
    - `SERVER_TASK_TYPE_CANCEL`: Represents a task type for canceling a task.
    - `SERVER_TASK_TYPE_NEXT_RESPONSE`: Represents a task type for fetching the next response.
    - `SERVER_TASK_TYPE_METRICS`: Represents a task type for gathering metrics.
    - `SERVER_TASK_TYPE_SLOT_SAVE`: Represents a task type for saving a slot.
    - `SERVER_TASK_TYPE_SLOT_RESTORE`: Represents a task type for restoring a slot.
    - `SERVER_TASK_TYPE_SLOT_ERASE`: Represents a task type for erasing a slot.
    - `SERVER_TASK_TYPE_SET_LORA`: Represents a task type for setting LoRA (Low-Rank Adaptation).
- **Description**: The `server_task_type` enum defines various task types that a server can handle, such as completion, embedding, reranking, infill, and others. Each enumerator represents a specific kind of task that the server can execute, allowing for organized task management and processing within the server's architecture.


---
### oaicompat\_type<!-- {{#data_structure:oaicompat_type}} -->
- **Type**: `enum`
- **Members**:
    - `OAICOMPAT_TYPE_NONE`: Represents a state where no specific OpenAI compatibility type is set.
    - `OAICOMPAT_TYPE_CHAT`: Represents a compatibility type for chat-based interactions.
    - `OAICOMPAT_TYPE_COMPLETION`: Represents a compatibility type for text completion tasks.
    - `OAICOMPAT_TYPE_EMBEDDING`: Represents a compatibility type for embedding tasks.
- **Description**: The `oaicompat_type` enum defines various types of compatibility modes for OpenAI-related tasks, such as chat, completion, and embedding. Each enumerator represents a specific mode that can be used to configure the behavior of a system or application to align with OpenAI's expected input and output formats. This enum is useful for distinguishing between different types of tasks and ensuring that the system processes them according to the appropriate standards.


---
### error\_type<!-- {{#data_structure:error_type}} -->
- **Type**: `enum`
- **Members**:
    - `ERROR_TYPE_INVALID_REQUEST`: Represents an error due to an invalid request.
    - `ERROR_TYPE_AUTHENTICATION`: Represents an error due to authentication failure.
    - `ERROR_TYPE_SERVER`: Represents a server-side error.
    - `ERROR_TYPE_NOT_FOUND`: Represents an error when a requested resource is not found.
    - `ERROR_TYPE_PERMISSION`: Represents an error due to insufficient permissions.
    - `ERROR_TYPE_UNAVAILABLE`: Represents a custom error indicating a service is unavailable.
    - `ERROR_TYPE_NOT_SUPPORTED`: Represents a custom error indicating a feature is not supported.
- **Description**: The `error_type` enum defines a set of constants representing different types of errors that can occur in a system, such as invalid requests, authentication failures, server errors, and custom errors like unavailable services or unsupported features. Each constant in the enum corresponds to a specific error scenario, allowing for structured error handling and reporting.


---
### slot\_params<!-- {{#data_structure:slot_params}} -->
- **Type**: `struct`
- **Members**:
    - `stream`: Indicates if streaming is enabled.
    - `cache_prompt`: Determines if the prompt should be cached to avoid reprocessing.
    - `return_tokens`: Specifies if tokens should be returned.
    - `n_keep`: Number of tokens to keep from the initial prompt.
    - `n_discard`: Number of tokens that can be discarded after n_keep when shifting context.
    - `n_predict`: Number of new tokens to predict.
    - `n_indent`: Minimum line indentation for generated text in whitespace characters.
    - `t_max_prompt_ms`: Maximum time in milliseconds for processing the prompt.
    - `t_max_predict_ms`: Maximum time in milliseconds for the generation phase.
    - `lora`: List of LoRA (Low-Rank Adaptation) information.
    - `antiprompt`: List of strings that act as stop sequences.
    - `response_fields`: List of fields to include in the response.
    - `timings_per_token`: Indicates if timings should be recorded per token.
    - `post_sampling_probs`: Specifies if probabilities should be recorded after sampling.
    - `ignore_eos`: Determines if the end-of-sequence token should be ignored.
    - `sampling`: Parameters related to sampling.
    - `speculative`: Parameters related to speculative execution.
    - `verbose`: Indicates if verbose logging is enabled.
    - `oaicompat`: Type of OpenAI compatibility.
    - `oaicompat_model`: Name of the OpenAI-compatible model.
    - `oaicompat_cmpl_id`: Completion ID for OpenAI compatibility.
    - `oaicompat_chat_syntax`: Syntax settings for OpenAI-compatible chat.
- **Description**: The `slot_params` struct is a configuration data structure used to manage various parameters for text generation tasks. It includes settings for streaming, caching, token prediction, and context management. Additionally, it supports OpenAI compatibility features and allows for detailed control over sampling and speculative execution. The struct also provides options for handling LoRA adapters, response formatting, and timing measurements.
- **Member Functions**:
    - [`slot_params::to_json`](#slot_paramsto_json)

**Methods**

---
#### slot\_params::to\_json<!-- {{#callable:slot_params::to_json}} -->
The `to_json` function converts the `slot_params` object into a JSON representation, capturing various configuration and state parameters.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Initialize a vector `samplers` to store string representations of sampler types and reserve space based on the size of `sampling.samplers`.
    - Iterate over `sampling.samplers`, convert each sampler to a string using [`common_sampler_type_to_str`](../../common/sampling.cpp.driver.md#common_sampler_type_to_str), and add it to the `samplers` vector.
    - Create a JSON array `lora` and populate it with objects containing `id` and `scale` for each element in `this->lora`.
    - Initialize a JSON array `grammar_triggers` and populate it by converting each `sampling.grammar_triggers` element to JSON using `server_grammar_trigger`.
    - Return a JSON object containing various parameters from the `slot_params` structure, including prediction settings, sampling parameters, penalties, and other configuration options.
- **Output**: A JSON object representing the `slot_params` structure, including various configuration and state parameters.
- **Functions called**:
    - [`common_sampler_type_to_str`](../../common/sampling.cpp.driver.md#common_sampler_type_to_str)
    - [`format_logit_bias`](utils.hpp.driver.md#format_logit_bias)
- **See also**: [`slot_params`](#slot_params)  (Data Structure)



---
### server\_task<!-- {{#data_structure:server_task}} -->
- **Type**: `struct`
- **Members**:
    - `id`: An integer identifier for the task, initialized to -1.
    - `index`: An integer index used for batch requests, initialized to -1.
    - `type`: The type of server task, defined by the `server_task_type` enum.
    - `id_target`: An integer target ID used specifically for cancel tasks, initialized to -1.
    - `params`: Parameters for inference tasks, encapsulated in a `slot_params` structure.
    - `prompt_tokens`: Tokens representing the prompt for inference tasks, stored in a `server_tokens` structure.
    - `id_selected_slot`: An integer representing the selected slot ID for inference tasks, initialized to -1.
    - `slot_action`: A nested structure containing details for slot actions like save, restore, or erase.
    - `metrics_reset_bucket`: A boolean flag indicating whether to reset metrics, used in metrics tasks.
    - `set_lora`: A vector of `common_adapter_lora_info` used for setting LoRA adapters.
- **Description**: The `server_task` struct is a comprehensive data structure designed to encapsulate various types of tasks that a server can handle, such as inference, cancellation, slot management, and metrics collection. It includes fields for task identification, type specification, and parameters specific to the task type. The struct also contains a nested `slot_action` structure for managing slot-related tasks and a vector for handling LoRA adapter settings. This struct is integral to managing and executing tasks within a server environment, providing a flexible and extensible framework for task management.
- **Member Functions**:
    - [`server_task::server_task`](#server_taskserver_task)
    - [`server_task::params_from_json_cmpl`](#server_taskparams_from_json_cmpl)
    - [`server_task::get_list_id`](#server_taskget_list_id)

**Methods**

---
#### server\_task::server\_task<!-- {{#callable:server_task::server_task}} -->
The `server_task` constructor initializes a `server_task` object with a specified task type.
- **Inputs**:
    - `type`: A `server_task_type` enumeration value that specifies the type of the server task to be initialized.
- **Control Flow**:
    - The constructor takes a `server_task_type` argument named `type`.
    - It initializes the `type` member of the `server_task` struct with the provided `type` argument.
- **Output**: A `server_task` object is created with its `type` member set to the specified `server_task_type`.
- **See also**: [`server_task`](#server_task)  (Data Structure)


---
#### server\_task::params\_from\_json\_cmpl<!-- {{#callable:server_task::params_from_json_cmpl}} -->
The `params_from_json_cmpl` function initializes and returns a `slot_params` object by extracting and processing various parameters from a JSON object, using defaults and context-specific settings.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which provides the context for the model and vocabulary.
    - `params_base`: A reference to a `common_params` object that contains base parameters, including sampling and speculative settings.
    - `data`: A JSON object containing various parameters that may override the defaults or base parameters.
- **Control Flow**:
    - Retrieve the model and vocabulary from the context using `llama_get_model` and `llama_model_get_vocab` functions.
    - Initialize a `slot_params` object and set its default values using `params_base`.
    - Set various parameters in `slot_params` by extracting values from the JSON `data`, using defaults from `params_base` if not specified in `data`.
    - Perform sanity checks on certain parameters, throwing runtime errors if conditions are not met (e.g., `repeat_last_n` must be >= -1).
    - Adjust speculative parameters to ensure `n_min` is within valid bounds.
    - Handle special cases such as `logprobs` and `lora` settings, parsing and validating them as needed.
    - Check and set grammar-related parameters, converting JSON schema to grammar if necessary.
    - Process chat format and syntax settings, adjusting based on the presence of certain flags in `data`.
    - Handle preserved tokens and grammar triggers, ensuring they are correctly tokenized and validated.
    - Clear and set logit biases based on `data`, ensuring valid tokens and biases are used.
    - Process stop words and samplers, setting them in `slot_params` based on `data`.
    - Return the fully initialized `slot_params` object.
- **Output**: A `slot_params` object containing the initialized parameters for a server task, ready for use in processing requests.
- **Functions called**:
    - [`json_value`](utils.hpp.driver.md#json_value)
    - [`parse_lora_request`](utils.hpp.driver.md#parse_lora_request)
    - [`json_schema_to_grammar`](../../common/json-schema-to-grammar.cpp.driver.md#json_schema_to_grammar)
    - [`common_sampler_types_from_names`](../../common/sampling.cpp.driver.md#common_sampler_types_from_names)
    - [`common_sampler_types_from_chars`](../../common/sampling.cpp.driver.md#common_sampler_types_from_chars)
- **See also**: [`server_task`](#server_task)  (Data Structure)


---
#### server\_task::get\_list\_id<!-- {{#callable:server_task::get_list_id}} -->
The `get_list_id` function extracts and returns a set of unique IDs from a vector of `server_task` objects.
- **Inputs**:
    - `tasks`: A constant reference to a vector of `server_task` objects, each containing an integer ID.
- **Control Flow**:
    - Initialize an unordered set `ids` with a size equal to the number of tasks.
    - Iterate over each `server_task` object in the `tasks` vector.
    - Insert the `id` of each `server_task` into the `ids` set.
    - Return the `ids` set containing unique IDs.
- **Output**: An unordered set of integers representing the unique IDs extracted from the `server_task` objects.
- **See also**: [`server_task`](#server_task)  (Data Structure)



---
### slot\_action<!-- {{#data_structure:server_task::slot_action}} -->
- **Type**: `struct`
- **Members**:
    - `slot_id`: An integer representing the unique identifier for a slot.
    - `filename`: A string representing the name of the file associated with the slot action.
    - `filepath`: A string representing the path to the file associated with the slot action.
- **Description**: The `slot_action` struct is a simple data structure used to encapsulate information related to a specific action on a slot, such as saving or restoring. It contains an integer `slot_id` to uniquely identify the slot, and two strings, `filename` and `filepath`, which specify the name and path of the file associated with the action. This struct is typically used in contexts where file operations related to slots are performed, such as saving the state of a slot to a file or restoring it from a file.


---
### result\_timings<!-- {{#data_structure:result_timings}} -->
- **Type**: `struct`
- **Members**:
    - `prompt_n`: Stores the number of prompt tokens processed, initialized to -1.
    - `prompt_ms`: Records the total time taken to process the prompt in milliseconds.
    - `prompt_per_token_ms`: Calculates the average time per token for the prompt in milliseconds.
    - `prompt_per_second`: Indicates the rate of prompt processing in tokens per second.
    - `predicted_n`: Stores the number of predicted tokens processed, initialized to -1.
    - `predicted_ms`: Records the total time taken to process the predicted tokens in milliseconds.
    - `predicted_per_token_ms`: Calculates the average time per token for the predicted tokens in milliseconds.
    - `predicted_per_second`: Indicates the rate of predicted token processing in tokens per second.
    - `draft_n`: Optional metric for the number of draft tokens generated, initialized to 0.
    - `draft_n_accepted`: Optional metric for the number of draft tokens accepted, initialized to 0.
- **Description**: The `result_timings` struct is designed to capture and store various timing metrics related to the processing of prompt and predicted tokens in a computational task. It includes fields for the number of tokens processed, the total time taken, and the average time per token for both prompt and predicted tokens. Additionally, it optionally records speculative metrics for draft tokens, which are only included if they are greater than zero. This struct is useful for performance analysis and optimization by providing detailed insights into the timing aspects of token processing.
- **Member Functions**:
    - [`result_timings::to_json`](#result_timingsto_json)

**Methods**

---
#### result\_timings::to\_json<!-- {{#callable:result_timings::to_json}} -->
The `to_json` function converts the `result_timings` structure's data into a JSON object, including optional speculative metrics if applicable.
- **Inputs**: None
- **Control Flow**:
    - Initialize a JSON object `base` with key-value pairs for prompt and predicted metrics from the `result_timings` structure.
    - Check if `draft_n` is greater than 0, indicating the presence of speculative metrics.
    - If speculative metrics are present, add `draft_n` and `draft_n_accepted` to the `base` JSON object.
    - Return the `base` JSON object.
- **Output**: A JSON object representing the `result_timings` structure, including optional speculative metrics if `draft_n` is greater than 0.
- **See also**: [`result_timings`](#result_timings)  (Data Structure)



---
### server\_task\_result<!-- {{#data_structure:server_task_result}} -->
- **Type**: `struct`
- **Members**:
    - `id`: An integer identifier for the task result, initialized to -1.
    - `id_slot`: An integer identifier for the slot associated with the task result, initialized to -1.
- **Description**: The `server_task_result` struct serves as a base class for representing the result of a server task. It contains basic identifiers such as `id` and `id_slot` to track the task and its associated slot. The struct is designed to be extended by other classes that implement specific task result types, as indicated by its virtual methods like `is_error`, `is_stop`, `get_index`, and `to_json`. These methods provide a polymorphic interface for handling different types of task results, such as errors or completion signals, and for converting the result to a JSON format. The struct also includes a virtual destructor, ensuring proper cleanup of derived classes.
- **Member Functions**:
    - [`server_task_result::is_error`](#server_task_resultis_error)
    - [`server_task_result::is_stop`](#server_task_resultis_stop)
    - [`server_task_result::get_index`](#server_task_resultget_index)
    - [`server_task_result::~server_task_result`](#server_task_resultserver_task_result)

**Methods**

---
#### server\_task\_result::is\_error<!-- {{#callable:server_task_result::is_error}} -->
The `is_error` function is a virtual method in the `server_task_result` structure that returns `false` by default, indicating that the task result is not an error.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as a virtual method, allowing derived classes to override it.
    - It contains a single return statement that returns `false`.
- **Output**: The function returns a boolean value `false`, indicating that the task result is not an error by default.
- **See also**: [`server_task_result`](#server_task_result)  (Data Structure)


---
#### server\_task\_result::is\_stop<!-- {{#callable:server_task_result::is_stop}} -->
The `is_stop` method in the `server_task_result` structure is a virtual function that returns `false` by default, indicating that the task result is not a stop condition.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as a virtual method within the `server_task_result` structure.
    - It contains a single line of code that returns `false`.
    - The comment indicates that this method is only used by `server_task_result_cmpl_*` classes.
- **Output**: The function returns a boolean value `false`, indicating that the task result is not a stop condition by default.
- **See also**: [`server_task_result`](#server_task_result)  (Data Structure)


---
#### server\_task\_result::get\_index<!-- {{#callable:server_task_result::get_index}} -->
The `get_index` function returns a default index value of -1.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as a virtual method, allowing derived classes to override it.
    - It simply returns the integer value -1.
- **Output**: The function returns an integer value, specifically -1.
- **See also**: [`server_task_result`](#server_task_result)  (Data Structure)


---
#### server\_task\_result::\~server\_task\_result<!-- {{#callable:server_task_result::~server_task_result}} -->
The destructor `~server_task_result` is a virtual default destructor for the `server_task_result` struct.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual, ensuring that derived class destructors are called when an object is deleted through a base class pointer.
    - The destructor is defined as default, indicating that the compiler should generate the default destructor implementation.
- **Output**: The destructor does not return any value or output.
- **See also**: [`server_task_result`](#server_task_result)  (Data Structure)



---
### completion\_token\_output<!-- {{#data_structure:completion_token_output}} -->
- **Type**: `struct`
- **Members**:
    - `tok`: Represents a token of type `llama_token`.
    - `prob`: Stores the probability of the token as a `float`.
    - `text_to_send`: Holds the text representation of the token as a `std::string`.
    - `probs`: A vector of `prob_info` structs, each containing detailed probability information for tokens.
- **Description**: The `completion_token_output` struct is designed to encapsulate the output of a token completion process, including the token itself, its probability, and the text representation to be sent. It also contains a nested `prob_info` struct to store detailed probability information for multiple tokens, which is useful for post-processing and analysis of token probabilities. This struct is integral to handling the results of token generation and their associated probabilities in a structured manner.
- **Member Functions**:
    - [`completion_token_output::to_json`](#completion_token_outputto_json)
    - [`completion_token_output::probs_vector_to_json`](#completion_token_outputprobs_vector_to_json)
    - [`completion_token_output::logarithm`](#completion_token_outputlogarithm)
    - [`completion_token_output::str_to_bytes`](#completion_token_outputstr_to_bytes)

**Methods**

---
#### completion\_token\_output::to\_json<!-- {{#callable:completion_token_output::to_json}} -->
The `to_json` function converts a vector of `prob_info` structures into a JSON array, with each element containing token information and either probability or log probability based on the `post_sampling_probs` flag.
- **Inputs**:
    - `post_sampling_probs`: A boolean flag indicating whether to include probabilities (`true`) or log probabilities (`false`) in the JSON output.
- **Control Flow**:
    - Initialize an empty JSON array `probs_for_token`.
    - Iterate over each `prob_info` structure in the `probs` vector.
    - For each `prob_info`, convert the `txt` string to a valid UTF-8 string and resize it accordingly.
    - Convert the `txt` string to a byte array using [`str_to_bytes`](#completion_token_outputstr_to_bytes).
    - Determine whether to use `prob` or `logprob` based on the `post_sampling_probs` flag and calculate the appropriate value.
    - Add a JSON object to `probs_for_token` containing the token ID, token text, byte array, and either probability or log probability.
    - Return the `probs_for_token` JSON array.
- **Output**: A JSON array where each element is a JSON object containing the token ID, token text, byte array, and either probability or log probability.
- **Functions called**:
    - [`validate_utf8`](utils.hpp.driver.md#validate_utf8)
    - [`completion_token_output::str_to_bytes`](#completion_token_outputstr_to_bytes)
    - [`completion_token_output::logarithm`](#completion_token_outputlogarithm)
- **See also**: [`completion_token_output`](#completion_token_output)  (Data Structure)


---
#### completion\_token\_output::probs\_vector\_to\_json<!-- {{#callable:completion_token_output::probs_vector_to_json}} -->
The `probs_vector_to_json` function converts a vector of `completion_token_output` objects into a JSON array, with each object represented as a JSON object containing token information and probabilities.
- **Inputs**:
    - `probs`: A constant reference to a vector of `completion_token_output` objects, each containing token information and associated probabilities.
    - `post_sampling_probs`: A boolean flag indicating whether to use post-sampling probabilities (`true`) or log probabilities (`false`) in the JSON output.
- **Control Flow**:
    - Initialize an empty JSON array `out` to store the converted objects.
    - Iterate over each `completion_token_output` object `p` in the `probs` vector.
    - For each `p`, create a string `txt` from `p.text_to_send` and resize it to ensure valid UTF-8 encoding using [`validate_utf8`](utils.hpp.driver.md#validate_utf8).
    - Push a JSON object to `out` containing the token ID, token text, byte representation of the text, and either the probability or log probability based on `post_sampling_probs`.
    - Include either `top_probs` or `top_logprobs` in the JSON object, using `p.to_json(post_sampling_probs)` to get the appropriate JSON representation of the token's probabilities.
    - Return the JSON array `out` containing all the converted objects.
- **Output**: A JSON array where each element is a JSON object representing a `completion_token_output` with token ID, text, byte representation, and probabilities or log probabilities.
- **Functions called**:
    - [`validate_utf8`](utils.hpp.driver.md#validate_utf8)
    - [`completion_token_output::str_to_bytes`](#completion_token_outputstr_to_bytes)
    - [`completion_token_output::logarithm`](#completion_token_outputlogarithm)
- **See also**: [`completion_token_output`](#completion_token_output)  (Data Structure)


---
#### completion\_token\_output::logarithm<!-- {{#callable:completion_token_output::logarithm}} -->
The `logarithm` function calculates the natural logarithm of a given float value, returning the lowest possible float value if the input is zero to prevent JSON conversion issues.
- **Inputs**:
    - `x`: A float value for which the natural logarithm is to be calculated.
- **Control Flow**:
    - Check if the input `x` is equal to 0.0f.
    - If `x` is 0.0f, return the lowest possible float value using `std::numeric_limits<float>::lowest()`.
    - Otherwise, return the natural logarithm of `x` using `std::log(x)`.
- **Output**: A float value representing the natural logarithm of `x`, or the lowest possible float value if `x` is zero.
- **See also**: [`completion_token_output`](#completion_token_output)  (Data Structure)


---
#### completion\_token\_output::str\_to\_bytes<!-- {{#callable:completion_token_output::str_to_bytes}} -->
The `str_to_bytes` function converts a given string into a vector of unsigned char bytes.
- **Inputs**:
    - `str`: A constant reference to a std::string that needs to be converted into bytes.
- **Control Flow**:
    - Initialize an empty vector of unsigned char named `bytes`.
    - Iterate over each character `c` in the input string `str`.
    - For each character `c`, cast it to an unsigned char and append it to the `bytes` vector.
    - Return the `bytes` vector containing the converted bytes of the input string.
- **Output**: A std::vector<unsigned char> containing the byte representation of the input string.
- **See also**: [`completion_token_output`](#completion_token_output)  (Data Structure)



---
### prob\_info<!-- {{#data_structure:completion_token_output::prob_info}} -->
- **Type**: `struct`
- **Members**:
    - `tok`: Represents a token of type `llama_token`.
    - `txt`: Holds the text associated with the token as a `std::string`.
    - `prob`: Stores the probability of the token as a `float`.
- **Description**: The `prob_info` struct is designed to encapsulate information about a token, including its identifier (`tok`), the corresponding text (`txt`), and the probability of the token (`prob`). This struct is used to store and manage token-related data, particularly in contexts where token probabilities are relevant, such as in language models or text generation tasks. The `probs` vector is a collection of `prob_info` instances, allowing for the management of multiple tokens and their associated data.


---
### server\_task\_result\_cmpl\_final<!-- {{#data_structure:server_task_result_cmpl_final}} -->
- **Type**: `struct`
- **Members**:
    - `index`: An integer representing the index of the task result.
    - `content`: A string containing the content of the task result.
    - `tokens`: A llama_tokens object representing the tokens associated with the task result.
    - `stream`: A boolean indicating if the result is part of a stream.
    - `timings`: A result_timings object containing timing information for the task result.
    - `prompt`: A string representing the prompt used for the task.
    - `truncated`: A boolean indicating if the result was truncated.
    - `n_decoded`: An integer representing the number of decoded tokens.
    - `n_prompt_tokens`: An integer representing the number of prompt tokens.
    - `n_tokens_cached`: An integer representing the number of cached tokens.
    - `has_new_line`: A boolean indicating if the result contains a new line.
    - `stopping_word`: A string representing the stopping word for the task result.
    - `stop`: A stop_type enum indicating the stop condition of the task.
    - `post_sampling_probs`: A boolean indicating if post-sampling probabilities are included.
    - `probs_output`: A vector of completion_token_output objects representing the probabilities of the output tokens.
    - `response_fields`: A vector of strings representing the response fields.
    - `generation_params`: A slot_params object containing the generation parameters.
    - `verbose`: A boolean indicating if verbose output is enabled.
    - `oaicompat`: An oaicompat_type enum indicating the OpenAI compatibility type.
    - `oaicompat_model`: A string representing the OpenAI compatible model name.
    - `oaicompat_cmpl_id`: A string representing the OpenAI compatible completion ID.
    - `oaicompat_msg`: A common_chat_msg object representing the OpenAI compatible message.
    - `oaicompat_msg_diffs`: A vector of common_chat_msg_diff objects representing the differences in OpenAI compatible messages.
- **Description**: The `server_task_result_cmpl_final` struct is a data structure that extends `server_task_result` to represent the final result of a server task, specifically for completion tasks. It includes various fields to store information about the task result, such as the content, tokens, timing information, and OpenAI compatibility details. The struct also provides methods to convert the result to JSON format, supporting both standard and OpenAI-compatible formats, and to determine if the result signifies a stop condition in a streaming context.
- **Member Functions**:
    - [`server_task_result_cmpl_final::get_index`](#server_task_result_cmpl_finalget_index)
    - [`server_task_result_cmpl_final::is_stop`](#server_task_result_cmpl_finalis_stop)
    - [`server_task_result_cmpl_final::to_json`](#server_task_result_cmpl_finalto_json)
    - [`server_task_result_cmpl_final::to_json_non_oaicompat`](#server_task_result_cmpl_finalto_json_non_oaicompat)
    - [`server_task_result_cmpl_final::to_json_oaicompat`](#server_task_result_cmpl_finalto_json_oaicompat)
    - [`server_task_result_cmpl_final::to_json_oaicompat_chat`](#server_task_result_cmpl_finalto_json_oaicompat_chat)
    - [`server_task_result_cmpl_final::to_json_oaicompat_chat_stream`](#server_task_result_cmpl_finalto_json_oaicompat_chat_stream)
- **Inherits From**:
    - [`server_task_result`](#server_task_result)

**Methods**

---
#### server\_task\_result\_cmpl\_final::get\_index<!-- {{#callable:server_task_result_cmpl_final::get_index}} -->
The `get_index` method returns the value of the `index` member variable of the `server_task_result_cmpl_final` class.
- **Inputs**: None
- **Control Flow**:
    - The method directly returns the value of the `index` member variable.
- **Output**: An integer representing the `index` of the `server_task_result_cmpl_final` instance.
- **See also**: [`server_task_result_cmpl_final`](#server_task_result_cmpl_final)  (Data Structure)


---
#### server\_task\_result\_cmpl\_final::is\_stop<!-- {{#callable:server_task_result_cmpl_final::is_stop}} -->
The `is_stop` method in the `server_task_result_cmpl_final` class always returns `true`, indicating that final responses in stream mode are considered as stop.
- **Inputs**: None
- **Control Flow**:
    - The method is a virtual override of the `is_stop` method from the parent class `server_task_result`.
    - It simply returns `true` without any conditions or additional logic.
- **Output**: The method returns a boolean value `true`.
- **See also**: [`server_task_result_cmpl_final`](#server_task_result_cmpl_final)  (Data Structure)


---
#### server\_task\_result\_cmpl\_final::to\_json<!-- {{#callable:server_task_result_cmpl_final::to_json}} -->
The `to_json` method converts the `server_task_result_cmpl_final` object into a JSON representation based on the `oaicompat` type.
- **Inputs**:
    - `None`: This method does not take any input arguments.
- **Control Flow**:
    - The method uses a switch statement to determine the `oaicompat` type of the object.
    - If `oaicompat` is `OAICOMPAT_TYPE_NONE`, it calls `to_json_non_oaicompat()` to generate the JSON.
    - If `oaicompat` is `OAICOMPAT_TYPE_COMPLETION`, it calls `to_json_oaicompat()` to generate the JSON.
    - If `oaicompat` is `OAICOMPAT_TYPE_CHAT`, it checks the `stream` flag to decide between `to_json_oaicompat_chat_stream()` and `to_json_oaicompat_chat()`.
    - If none of the cases match, it asserts with an error message indicating an invalid `oaicompat_type`.
- **Output**: The method returns a JSON object representing the `server_task_result_cmpl_final` instance, formatted according to the specified `oaicompat` type.
- **Functions called**:
    - [`server_task_result_cmpl_final::to_json_non_oaicompat`](#server_task_result_cmpl_finalto_json_non_oaicompat)
    - [`server_task_result_cmpl_final::to_json_oaicompat`](#server_task_result_cmpl_finalto_json_oaicompat)
    - [`server_task_result_cmpl_final::to_json_oaicompat_chat_stream`](#server_task_result_cmpl_finalto_json_oaicompat_chat_stream)
    - [`server_task_result_cmpl_final::to_json_oaicompat_chat`](#server_task_result_cmpl_finalto_json_oaicompat_chat)
- **See also**: [`server_task_result_cmpl_final`](#server_task_result_cmpl_final)  (Data Structure)


---
#### server\_task\_result\_cmpl\_final::to\_json\_non\_oaicompat<!-- {{#callable:server_task_result_cmpl_final::to_json_non_oaicompat}} -->
The `to_json_non_oaicompat` function generates a JSON representation of a server task result in a non-OpenAI-compatible format, including various task-related details and metrics.
- **Inputs**: None
- **Control Flow**:
    - Initialize a JSON object `res` with various fields from the class, such as `index`, `content`, `tokens`, `id_slot`, `stop`, `model`, `tokens_predicted`, `tokens_evaluated`, `generation_settings`, `prompt`, `has_new_line`, `truncated`, `stop_type`, `stopping_word`, `tokens_cached`, and `timings`.
    - Check if `stream` is false and `probs_output` is not empty; if true, add `completion_probabilities` to `res` using `completion_token_output::probs_vector_to_json`.
    - Return `res` if `response_fields` is empty; otherwise, return `json_get_nested_values(response_fields, res)` to filter the JSON output based on `response_fields`.
- **Output**: A JSON object representing the server task result with various fields and metrics.
- **Functions called**:
    - [`stop_type_to_str`](#stop_type_to_str)
    - [`json_get_nested_values`](utils.hpp.driver.md#json_get_nested_values)
- **See also**: [`server_task_result_cmpl_final`](#server_task_result_cmpl_final)  (Data Structure)


---
#### server\_task\_result\_cmpl\_final::to\_json\_oaicompat<!-- {{#callable:server_task_result_cmpl_final::to_json_oaicompat}} -->
The `to_json_oaicompat` function generates a JSON object compatible with OpenAI's API format, representing the result of a text completion task, including metadata and optional debugging information.
- **Inputs**: None
- **Control Flow**:
    - Initialize the current time `t` and set `logprobs` to `null` by default.
    - Check if `stream` is false and `probs_output` is not empty, then populate `logprobs` with token probabilities converted to JSON.
    - Determine the `finish_reason` based on the `stop` type, setting it to 'stop' if the stop type is `STOP_TYPE_WORD` or `STOP_TYPE_EOS`, otherwise 'length'.
    - Construct a JSON object `res` with fields such as `choices`, `created`, `model`, `system_fingerprint`, `object`, `usage`, and `id`.
    - If `verbose` is true, add a `__verbose` field to `res` with non-OAI compatible JSON data.
    - If `timings.prompt_n` is non-negative, append `timings` JSON data to `res`.
    - Return the constructed JSON object `res`.
- **Output**: A JSON object representing the completion result, including choices, model information, usage statistics, and optional debugging data.
- **Functions called**:
    - [`server_task_result_cmpl_final::to_json_non_oaicompat`](#server_task_result_cmpl_finalto_json_non_oaicompat)
- **See also**: [`server_task_result_cmpl_final`](#server_task_result_cmpl_final)  (Data Structure)


---
#### server\_task\_result\_cmpl\_final::to\_json\_oaicompat\_chat<!-- {{#callable:server_task_result_cmpl_final::to_json_oaicompat_chat}} -->
The `to_json_oaicompat_chat` function generates a JSON object representing a chat completion response compatible with OpenAI's API format.
- **Inputs**:
    - `None`: This function does not take any input parameters.
- **Control Flow**:
    - Initialize `finish_reason` to "length" and declare a `common_chat_msg` object `msg`.
    - Check if `oaicompat_msg` is not empty; if so, assign it to `msg`, otherwise set `msg.role` to "assistant" and `msg.content` to `content`.
    - Determine `finish_reason` based on `stop` type and whether `msg.tool_calls` is empty.
    - Create a JSON object `choice` with `finish_reason`, `index`, and `message` fields.
    - If `stream` is false and `probs_output` is not empty, add `logprobs` to `choice`.
    - Get the current time and store it in `t`.
    - Construct the final JSON object `res` with fields like `choices`, `created`, `model`, `system_fingerprint`, `object`, `usage`, and `id`.
    - If `verbose` is true, add `__verbose` field to `res`.
    - If `timings.prompt_n` is non-negative, append `timings` to `res`.
    - Return the JSON object `res`.
- **Output**: A JSON object representing a chat completion response, including details like finish reason, message, model, and usage statistics.
- **Functions called**:
    - [`server_task_result_cmpl_final::to_json_non_oaicompat`](#server_task_result_cmpl_finalto_json_non_oaicompat)
- **See also**: [`server_task_result_cmpl_final`](#server_task_result_cmpl_final)  (Data Structure)


---
#### server\_task\_result\_cmpl\_final::to\_json\_oaicompat\_chat\_stream<!-- {{#callable:server_task_result_cmpl_final::to_json_oaicompat_chat_stream}} -->
The `to_json_oaicompat_chat_stream` function generates a JSON array of chat completion chunks, each containing deltas of message differences and metadata, for OpenAI-compatible streaming chat responses.
- **Inputs**: None
- **Control Flow**:
    - Initialize the current time `t` and set `finish_reason` to 'length'.
    - Check if `stop` is `STOP_TYPE_WORD` or `STOP_TYPE_EOS` and update `finish_reason` based on `oaicompat_msg.tool_calls`.
    - Create a JSON array `deltas` to store message differences.
    - Iterate over `oaicompat_msg_diffs` and for each difference, convert it to JSON and add it to `deltas` with metadata.
    - Add a final JSON object to `deltas` with the `finish_reason`, metadata, and usage statistics.
    - If `timings.prompt_n` is non-negative, append timing information to the last element of `deltas`.
    - If `verbose` is true and `deltas` is not empty, add verbose information to the first element of `deltas`.
    - Return the `deltas` JSON array.
- **Output**: A JSON array representing chat completion chunks with message differences and metadata for streaming responses.
- **Functions called**:
    - [`server_task_result_cmpl_final::to_json_non_oaicompat`](#server_task_result_cmpl_finalto_json_non_oaicompat)
- **See also**: [`server_task_result_cmpl_final`](#server_task_result_cmpl_final)  (Data Structure)



---
### server\_task\_result\_cmpl\_partial<!-- {{#data_structure:server_task_result_cmpl_partial}} -->
- **Type**: `struct`
- **Members**:
    - `index`: An integer representing the index of the task result.
    - `content`: A string containing the content of the task result.
    - `tokens`: A collection of tokens associated with the task result.
    - `n_decoded`: An integer representing the number of decoded tokens.
    - `n_prompt_tokens`: An integer representing the number of prompt tokens.
    - `post_sampling_probs`: A boolean indicating if post-sampling probabilities are included.
    - `prob_output`: An object containing the output probabilities of the completion tokens.
    - `timings`: An object containing timing information for the task result.
    - `verbose`: A boolean indicating if verbose output is enabled.
    - `oaicompat`: An enum indicating the OpenAI compatibility type.
    - `oaicompat_model`: A string representing the model used for OpenAI compatibility.
    - `oaicompat_cmpl_id`: A string representing the completion ID for OpenAI compatibility.
    - `oaicompat_msg_diffs`: A vector of message differences for OpenAI compatibility.
- **Description**: The `server_task_result_cmpl_partial` struct is a specialized data structure that extends `server_task_result` to represent partial results of a server task, particularly in a streaming context. It includes fields for managing the index, content, and tokens of the task result, as well as additional metadata such as the number of decoded and prompt tokens. The struct also supports OpenAI compatibility with fields for verbose output, model information, and message differences. Timing information and probability outputs are included to provide detailed insights into the task's execution.
- **Member Functions**:
    - [`server_task_result_cmpl_partial::get_index`](#server_task_result_cmpl_partialget_index)
    - [`server_task_result_cmpl_partial::is_stop`](#server_task_result_cmpl_partialis_stop)
    - [`server_task_result_cmpl_partial::to_json`](#server_task_result_cmpl_partialto_json)
    - [`server_task_result_cmpl_partial::to_json_non_oaicompat`](#server_task_result_cmpl_partialto_json_non_oaicompat)
    - [`server_task_result_cmpl_partial::to_json_oaicompat`](#server_task_result_cmpl_partialto_json_oaicompat)
    - [`server_task_result_cmpl_partial::to_json_oaicompat_chat`](#server_task_result_cmpl_partialto_json_oaicompat_chat)
- **Inherits From**:
    - [`server_task_result`](#server_task_result)

**Methods**

---
#### server\_task\_result\_cmpl\_partial::get\_index<!-- {{#callable:server_task_result_cmpl_partial::get_index}} -->
The `get_index` function returns the value of the `index` member variable of the `server_task_result_cmpl_partial` class.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the `index` member variable.
- **Output**: An integer representing the `index` of the `server_task_result_cmpl_partial` instance.
- **See also**: [`server_task_result_cmpl_partial`](#server_task_result_cmpl_partial)  (Data Structure)


---
#### server\_task\_result\_cmpl\_partial::is\_stop<!-- {{#callable:server_task_result_cmpl_partial::is_stop}} -->
The `is_stop` method in the `server_task_result_cmpl_partial` class always returns `false`, indicating that partial responses are not considered a stop in stream mode.
- **Inputs**: None
- **Control Flow**:
    - The method is defined as a virtual function and overrides a base class method.
    - It simply returns the boolean value `false`.
- **Output**: The method returns a boolean value `false`.
- **See also**: [`server_task_result_cmpl_partial`](#server_task_result_cmpl_partial)  (Data Structure)


---
#### server\_task\_result\_cmpl\_partial::to\_json<!-- {{#callable:server_task_result_cmpl_partial::to_json}} -->
The `to_json` method converts the `server_task_result_cmpl_partial` object into a JSON representation based on the `oaicompat` type.
- **Inputs**:
    - `None`: This method does not take any input arguments.
- **Control Flow**:
    - The method uses a switch statement to determine the `oaicompat` type of the object.
    - If `oaicompat` is `OAICOMPAT_TYPE_NONE`, it calls `to_json_non_oaicompat()` to generate a non-OAI-compatible JSON.
    - If `oaicompat` is `OAICOMPAT_TYPE_COMPLETION`, it calls `to_json_oaicompat()` to generate an OAI-compatible JSON for completion.
    - If `oaicompat` is `OAICOMPAT_TYPE_CHAT`, it calls `to_json_oaicompat_chat()` to generate an OAI-compatible JSON for chat.
    - If none of the cases match, it asserts with an error message indicating an invalid `oaicompat_type`.
- **Output**: The method returns a JSON object representing the `server_task_result_cmpl_partial` object, formatted according to the specified `oaicompat` type.
- **Functions called**:
    - [`server_task_result_cmpl_final::to_json_non_oaicompat`](#server_task_result_cmpl_finalto_json_non_oaicompat)
    - [`server_task_result_cmpl_final::to_json_oaicompat`](#server_task_result_cmpl_finalto_json_oaicompat)
    - [`server_task_result_cmpl_final::to_json_oaicompat_chat`](#server_task_result_cmpl_finalto_json_oaicompat_chat)
- **See also**: [`server_task_result_cmpl_partial`](#server_task_result_cmpl_partial)  (Data Structure)


---
#### server\_task\_result\_cmpl\_partial::to\_json\_non\_oaicompat<!-- {{#callable:server_task_result_cmpl_partial::to_json_non_oaicompat}} -->
The `to_json_non_oaicompat` function generates a JSON object representing a non-OAI-compatible partial server task result, including various task-related data and optional timing and probability information.
- **Inputs**:
    - `None`: This function does not take any input parameters.
- **Control Flow**:
    - Initialize a JSON object `res` with key-value pairs representing task-related data such as `index`, `content`, `tokens`, `stop`, `id_slot`, `tokens_predicted`, and `tokens_evaluated`.
    - Check if `timings.prompt_n` is greater than 0; if true, add a `timings` entry to the JSON object using `timings.to_json()`.
    - Check if `prob_output.probs` is not empty; if true, add a `completion_probabilities` entry to the JSON object using `completion_token_output::probs_vector_to_json`.
    - Return the JSON object `res`.
- **Output**: A JSON object containing task-related data, and optionally, timing and probability information.
- **See also**: [`server_task_result_cmpl_partial`](#server_task_result_cmpl_partial)  (Data Structure)


---
#### server\_task\_result\_cmpl\_partial::to\_json\_oaicompat<!-- {{#callable:server_task_result_cmpl_partial::to_json_oaicompat}} -->
The `to_json_oaicompat` function generates a JSON object compatible with OpenAI's API format for text completion results, including details like choices, model information, and optional debugging fields.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Initialize a `std::time_t` variable `t` with the current time.
    - Set `logprobs` to `json(nullptr)` as a default value.
    - Check if `prob_output.probs` is not empty; if true, set `logprobs` to a JSON object containing the content of `prob_output` converted to JSON.
    - Create a JSON object `res` with fields for choices, created time, model, system fingerprint, object type, and ID.
    - If `verbose` is true, add a `__verbose` field to `res` with non-OAI-compatible JSON data.
    - If `timings.prompt_n` is non-negative, add a `timings` field to `res`.
    - Return the JSON object `res`.
- **Output**: A JSON object representing the text completion result in a format compatible with OpenAI's API.
- **Functions called**:
    - [`server_task_result_cmpl_final::to_json_non_oaicompat`](#server_task_result_cmpl_finalto_json_non_oaicompat)
- **See also**: [`server_task_result_cmpl_partial`](#server_task_result_cmpl_partial)  (Data Structure)


---
#### server\_task\_result\_cmpl\_partial::to\_json\_oaicompat\_chat<!-- {{#callable:server_task_result_cmpl_partial::to_json_oaicompat_chat}} -->
The `to_json_oaicompat_chat` function generates a JSON representation of chat completion data in a format compatible with OpenAI's API, including handling message deltas and optional log probabilities.
- **Inputs**:
    - `None`: This function does not take any input parameters.
- **Control Flow**:
    - Initialize a boolean `first` to check if this is the first decoded message and get the current time `t`.
    - Define a vector `deltas` to store JSON objects representing message deltas.
    - Define a lambda function `add_delta` to add a new delta to the `deltas` vector with specific fields like `choices`, `created`, `id`, `model`, `system_fingerprint`, and `object`.
    - If `first` is true, add an initial delta with `role` as `assistant` and `content` as `nullptr` to conform to OpenAI behavior.
    - Iterate over `oaicompat_msg_diffs` and convert each diff to JSON using `common_chat_msg_diff_to_json_oaicompat`, then add it to `deltas` using `add_delta`.
    - If `deltas` is not empty, assert that the last delta has at least one choice.
    - If `prob_output.probs` is not empty, add `logprobs` to the last delta's first choice with probabilities converted to JSON.
    - If `timings.prompt_n` is non-negative, append `timings` converted to JSON to the last delta.
- **Output**: Returns a JSON array `deltas` containing the chat completion data in OpenAI-compatible format.
- **See also**: [`server_task_result_cmpl_partial`](#server_task_result_cmpl_partial)  (Data Structure)



---
### server\_task\_result\_embd<!-- {{#data_structure:server_task_result_embd}} -->
- **Type**: `struct`
- **Members**:
    - `index`: An integer representing the index of the task result.
    - `embedding`: A 2D vector of floats representing the embedding data.
    - `n_tokens`: An integer representing the number of tokens evaluated.
    - `oaicompat`: An enum value indicating the OpenAI compatibility type.
- **Description**: The `server_task_result_embd` struct is a specialized data structure that extends `server_task_result` to handle the results of embedding tasks on a server. It includes an index to identify the task, a 2D vector to store the embedding data, and a token count to track the number of tokens evaluated. Additionally, it supports OpenAI compatibility through the `oaicompat` field, which determines how the result is serialized into JSON, either in a standard or OpenAI-compatible format.
- **Member Functions**:
    - [`server_task_result_embd::get_index`](#server_task_result_embdget_index)
    - [`server_task_result_embd::to_json`](#server_task_result_embdto_json)
    - [`server_task_result_embd::to_json_non_oaicompat`](#server_task_result_embdto_json_non_oaicompat)
    - [`server_task_result_embd::to_json_oaicompat`](#server_task_result_embdto_json_oaicompat)
- **Inherits From**:
    - [`server_task_result`](#server_task_result)

**Methods**

---
#### server\_task\_result\_embd::get\_index<!-- {{#callable:server_task_result_embd::get_index}} -->
The `get_index` function returns the value of the `index` member variable of the `server_task_result_embd` structure.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the `index` member variable.
- **Output**: An integer representing the value of the `index` member variable.
- **See also**: [`server_task_result_embd`](#server_task_result_embd)  (Data Structure)


---
#### server\_task\_result\_embd::to\_json<!-- {{#callable:server_task_result_embd::to_json}} -->
The `to_json` method returns a JSON representation of the `server_task_result_embd` object, choosing between two formats based on the `oaicompat` field.
- **Inputs**:
    - `None`: This method does not take any input arguments.
- **Control Flow**:
    - Check if `oaicompat` is equal to `OAICOMPAT_TYPE_EMBEDDING`.
    - If true, call `to_json_oaicompat()` to get the JSON representation.
    - If false, call `to_json_non_oaicompat()` to get the JSON representation.
- **Output**: A JSON object representing the `server_task_result_embd` object, formatted according to the `oaicompat` field.
- **Functions called**:
    - [`server_task_result_cmpl_final::to_json_oaicompat`](#server_task_result_cmpl_finalto_json_oaicompat)
    - [`server_task_result_cmpl_final::to_json_non_oaicompat`](#server_task_result_cmpl_finalto_json_non_oaicompat)
- **See also**: [`server_task_result_embd`](#server_task_result_embd)  (Data Structure)


---
#### server\_task\_result\_embd::to\_json\_non\_oaicompat<!-- {{#callable:server_task_result_embd::to_json_non_oaicompat}} -->
The `to_json_non_oaicompat` function converts the `server_task_result_embd` object into a JSON object with non-OAI-compatible fields.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - The function creates a JSON object using the `json` library.
    - It populates the JSON object with two key-value pairs: `index` and `embedding`.
    - The `index` is set to the `index` member of the `server_task_result_embd` struct.
    - The `embedding` is set to the `embedding` member of the `server_task_result_embd` struct.
- **Output**: A JSON object containing the `index` and `embedding` fields of the `server_task_result_embd` struct.
- **See also**: [`server_task_result_embd`](#server_task_result_embd)  (Data Structure)


---
#### server\_task\_result\_embd::to\_json\_oaicompat<!-- {{#callable:server_task_result_embd::to_json_oaicompat}} -->
The `to_json_oaicompat` function converts the `server_task_result_embd` object into a JSON object with specific fields for OpenAI compatibility.
- **Inputs**: None
- **Control Flow**:
    - The function creates a JSON object using the `json` library.
    - It populates the JSON object with three key-value pairs: `index`, `embedding`, and `tokens_evaluated`.
    - The `index` is directly taken from the `server_task_result_embd` object.
    - The `embedding` is set to the first element of the `embedding` vector from the `server_task_result_embd` object.
    - The `tokens_evaluated` is set to the `n_tokens` value from the `server_task_result_embd` object.
- **Output**: A JSON object containing the `index`, the first `embedding`, and `tokens_evaluated` fields.
- **See also**: [`server_task_result_embd`](#server_task_result_embd)  (Data Structure)



---
### server\_task\_result\_rerank<!-- {{#data_structure:server_task_result_rerank}} -->
- **Type**: `struct`
- **Members**:
    - `index`: An integer representing the index of the rerank result.
    - `score`: A floating-point number representing the score of the rerank result, initialized to a very low value (-1e6).
    - `n_tokens`: An integer representing the number of tokens evaluated in the rerank task.
- **Description**: The `server_task_result_rerank` struct is a specialized data structure derived from `server_task_result` that represents the result of a rerank task on the server. It contains fields to store the index, score, and the number of tokens evaluated during the rerank process. The struct also provides an overridden method `get_index` to return the index and a `to_json` method to serialize the rerank result into a JSON format, including the index, score, and tokens evaluated.
- **Member Functions**:
    - [`server_task_result_rerank::get_index`](#server_task_result_rerankget_index)
    - [`server_task_result_rerank::to_json`](#server_task_result_rerankto_json)
- **Inherits From**:
    - [`server_task_result`](#server_task_result)

**Methods**

---
#### server\_task\_result\_rerank::get\_index<!-- {{#callable:server_task_result_rerank::get_index}} -->
The `get_index` function returns the value of the `index` member variable of the `server_task_result_rerank` structure.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the `index` member variable without any additional logic or conditions.
- **Output**: The function returns an integer value representing the `index` of the `server_task_result_rerank` instance.
- **See also**: [`server_task_result_rerank`](#server_task_result_rerank)  (Data Structure)


---
#### server\_task\_result\_rerank::to\_json<!-- {{#callable:server_task_result_rerank::to_json}} -->
The `to_json` method converts the `server_task_result_rerank` object into a JSON representation containing its index, score, and number of tokens evaluated.
- **Inputs**: None
- **Control Flow**:
    - The method constructs a JSON object with three key-value pairs: 'index', 'score', and 'tokens_evaluated'.
    - Each key is associated with the corresponding member variable of the `server_task_result_rerank` class.
- **Output**: A JSON object representing the `server_task_result_rerank` instance.
- **See also**: [`server_task_result_rerank`](#server_task_result_rerank)  (Data Structure)



---
### server\_task\_result\_error<!-- {{#data_structure:server_task_result_error}} -->
- **Type**: `struct`
- **Members**:
    - `index`: An integer representing the index of the error.
    - `err_type`: An enumeration value indicating the type of error, defaulting to ERROR_TYPE_SERVER.
    - `err_msg`: A string containing the error message.
- **Description**: The `server_task_result_error` struct is a specialized data structure derived from `server_task_result` that represents an error result in a server task. It includes an index to identify the error, an error type to categorize the error, and an error message to provide details about the error. This struct overrides the `is_error` method to always return true, indicating that it represents an error state, and provides a `to_json` method to format the error information into a JSON response.
- **Member Functions**:
    - [`server_task_result_error::is_error`](#server_task_result_erroris_error)
    - [`server_task_result_error::to_json`](#server_task_result_errorto_json)
- **Inherits From**:
    - [`server_task_result`](#server_task_result)

**Methods**

---
#### server\_task\_result\_error::is\_error<!-- {{#callable:server_task_result_error::is_error}} -->
The `is_error` function in the `server_task_result_error` struct always returns `true`, indicating that the result represents an error.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as a virtual method in the `server_task_result_error` struct, which inherits from `server_task_result`.
    - The function overrides the base class method `is_error`.
    - The function simply returns `true`, indicating that the instance represents an error.
- **Output**: The function returns a boolean value `true`.
- **See also**: [`server_task_result_error`](#server_task_result_error)  (Data Structure)


---
#### server\_task\_result\_error::to\_json<!-- {{#callable:server_task_result_error::to_json}} -->
The `to_json` function in the `server_task_result_error` class returns a JSON object representing an error response using the [`format_error_response`](#format_error_response) function.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - The function calls [`format_error_response`](#format_error_response) with `err_msg` and `err_type` as arguments.
    - The result of [`format_error_response`](#format_error_response) is returned as the output of the function.
- **Output**: A JSON object representing an error response, which includes the error code, message, and type.
- **Functions called**:
    - [`format_error_response`](#format_error_response)
- **See also**: [`server_task_result_error`](#server_task_result_error)  (Data Structure)



---
### server\_task\_result\_metrics<!-- {{#data_structure:server_task_result_metrics}} -->
- **Type**: `struct`
- **Members**:
    - `n_idle_slots`: Stores the number of idle slots available on the server.
    - `n_processing_slots`: Stores the number of slots currently processing tasks.
    - `n_tasks_deferred`: Stores the number of tasks that have been deferred.
    - `t_start`: Records the start time of the server task.
    - `n_prompt_tokens_processed_total`: Tracks the total number of prompt tokens processed.
    - `t_prompt_processing_total`: Tracks the total time spent processing prompts.
    - `n_tokens_predicted_total`: Tracks the total number of tokens predicted.
    - `t_tokens_generation_total`: Tracks the total time spent generating tokens.
    - `n_prompt_tokens_processed`: Tracks the number of prompt tokens processed in the current session.
    - `t_prompt_processing`: Tracks the time spent processing prompts in the current session.
    - `n_tokens_predicted`: Tracks the number of tokens predicted in the current session.
    - `t_tokens_generation`: Tracks the time spent generating tokens in the current session.
    - `n_decode_total`: Tracks the total number of decode operations performed.
    - `n_busy_slots_total`: Tracks the total number of busy slots over time.
    - `slots_data`: Stores JSON data representing the current state of server slots.
- **Description**: The `server_task_result_metrics` struct is a specialized data structure that extends `server_task_result` to capture and store various metrics related to server task processing. It includes fields for tracking the number of idle and processing slots, deferred tasks, and timing information for prompt processing and token generation. Additionally, it maintains cumulative totals for prompt tokens processed and tokens predicted, as well as JSON data for slot states, providing a comprehensive overview of server performance and task handling efficiency.
- **Member Functions**:
    - [`server_task_result_metrics::to_json`](#server_task_result_metricsto_json)
- **Inherits From**:
    - [`server_task_result`](#server_task_result)

**Methods**

---
#### server\_task\_result\_metrics::to\_json<!-- {{#callable:server_task_result_metrics::to_json}} -->
The `to_json` method converts the `server_task_result_metrics` object into a JSON representation containing various metrics and slot data.
- **Inputs**: None
- **Control Flow**:
    - The method constructs a JSON object using the `json` library.
    - It maps various member variables of the `server_task_result_metrics` class to corresponding JSON key-value pairs.
    - The JSON object includes metrics such as idle slots, processing slots, deferred tasks, and various token processing statistics.
    - The `slots_data` member, which is a JSON array, is also included in the JSON object.
- **Output**: A JSON object representing the metrics and slot data of the `server_task_result_metrics` instance.
- **See also**: [`server_task_result_metrics`](#server_task_result_metrics)  (Data Structure)



---
### server\_task\_result\_slot\_save\_load<!-- {{#data_structure:server_task_result_slot_save_load}} -->
- **Type**: `struct`
- **Members**:
    - `filename`: Stores the name of the file associated with the save or load operation.
    - `is_save`: Indicates whether the operation is a save (true) or load (false).
    - `n_tokens`: Represents the number of tokens involved in the save or load operation.
    - `n_bytes`: Represents the number of bytes written or read during the operation.
    - `t_ms`: Stores the time taken for the operation in milliseconds.
- **Description**: The `server_task_result_slot_save_load` struct is a specialized data structure that extends `server_task_result` to handle the results of slot save and load operations on a server. It contains fields to store the filename, operation type (save or load), the number of tokens and bytes processed, and the time taken for the operation. The struct provides a `to_json` method to serialize its data into a JSON format, which varies depending on whether the operation is a save or load.
- **Member Functions**:
    - [`server_task_result_slot_save_load::to_json`](#server_task_result_slot_save_loadto_json)
- **Inherits From**:
    - [`server_task_result`](#server_task_result)

**Methods**

---
#### server\_task\_result\_slot\_save\_load::to\_json<!-- {{#callable:server_task_result_slot_save_load::to_json}} -->
The `to_json` function serializes the state of a `server_task_result_slot_save_load` object into a JSON object, differentiating between save and load operations.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Check if the `is_save` flag is true.
    - If `is_save` is true, create a JSON object with keys `id_slot`, `filename`, `n_saved`, `n_written`, and `timings` containing `save_ms`.
    - If `is_save` is false, create a JSON object with keys `id_slot`, `filename`, `n_restored`, `n_read`, and `timings` containing `restore_ms`.
    - Return the created JSON object.
- **Output**: A JSON object representing the state of the `server_task_result_slot_save_load` object, with different fields for save and load operations.
- **See also**: [`server_task_result_slot_save_load`](#server_task_result_slot_save_load)  (Data Structure)



---
### server\_task\_result\_slot\_erase<!-- {{#data_structure:server_task_result_slot_erase}} -->
- **Type**: `struct`
- **Members**:
    - `n_erased`: Represents the number of slots erased.
- **Description**: The `server_task_result_slot_erase` struct is a specialized data structure that inherits from `server_task_result`. It is designed to represent the result of a server task that involves erasing slots. The primary attribute, `n_erased`, indicates the number of slots that have been successfully erased. This struct overrides the `to_json` method to provide a JSON representation of the result, including the slot ID and the number of erased slots.
- **Member Functions**:
    - [`server_task_result_slot_erase::to_json`](#server_task_result_slot_eraseto_json)
- **Inherits From**:
    - [`server_task_result`](#server_task_result)

**Methods**

---
#### server\_task\_result\_slot\_erase::to\_json<!-- {{#callable:server_task_result_slot_erase::to_json}} -->
The `to_json` function converts the `server_task_result_slot_erase` object into a JSON object containing the `id_slot` and `n_erased` fields.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - The function constructs a JSON object using the `json` initializer list syntax.
    - It includes two key-value pairs: `"id_slot"` with the value of the `id_slot` member variable, and `"n_erased"` with the value of the `n_erased` member variable.
    - The constructed JSON object is returned as the output of the function.
- **Output**: A JSON object containing the `id_slot` and `n_erased` fields of the `server_task_result_slot_erase` object.
- **See also**: [`server_task_result_slot_erase`](#server_task_result_slot_erase)  (Data Structure)



---
### server\_task\_result\_apply\_lora<!-- {{#data_structure:server_task_result_apply_lora}} -->
- **Type**: `struct`
- **Members**:
    - `to_json`: A virtual function that returns a JSON object indicating success.
- **Description**: The `server_task_result_apply_lora` struct is a specialized data structure derived from `server_task_result`. It overrides the `to_json` method to return a JSON object with a success status, indicating the successful application of a LoRA (Low-Rank Adaptation) task. This struct is part of a larger system handling server tasks and results, specifically for tasks related to applying LoRA configurations.
- **Member Functions**:
    - [`server_task_result_apply_lora::to_json`](#server_task_result_apply_lorato_json)
- **Inherits From**:
    - [`server_task_result`](#server_task_result)

**Methods**

---
#### server\_task\_result\_apply\_lora::to\_json<!-- {{#callable:server_task_result_apply_lora::to_json}} -->
The `to_json` method in the `server_task_result_apply_lora` class returns a JSON object indicating a successful operation.
- **Inputs**: None
- **Control Flow**:
    - The method constructs a JSON object with a single key-value pair where the key is 'success' and the value is `true`.
    - The method returns this JSON object.
- **Output**: A JSON object with a single key-value pair: { "success": true }.
- **See also**: [`server_task_result_apply_lora`](#server_task_result_apply_lora)  (Data Structure)



---
### server\_slot<!-- {{#data_structure:server_slot}} -->
- **Type**: `struct`
- **Members**:
    - `id`: An integer representing the unique identifier for the server slot.
    - `id_task`: An integer representing the task identifier associated with the server slot, defaulting to -1.
    - `task_type`: An enumeration indicating the type of task the server slot is handling, defaulting to SERVER_TASK_TYPE_COMPLETION.
    - `batch_spec`: A llama_batch structure specifying the batch configuration for the server slot.
    - `ctx`: A pointer to a llama_context, representing the context for the server slot.
    - `ctx_dft`: A pointer to a default llama_context, used for speculative decoding.
    - `mctx`: A pointer to an mtmd_context, used for multimodal processing.
    - `spec`: A pointer to a common_speculative structure, used for speculative decoding.
    - `lora`: A vector of common_adapter_lora_info structures, representing LoRA adapters applied to the server slot.
    - `index`: A size_t representing the index relative to a completion multi-task request.
    - `params`: A slot_params structure containing parameters for the server slot.
    - `state`: An enumeration indicating the current state of the server slot, defaulting to SLOT_STATE_IDLE.
    - `t_last_used`: An int64_t representing the timestamp of the last use of the server slot, defaulting to -1.
    - `n_ctx`: An int32_t representing the context size per slot.
    - `n_past`: An int32_t representing the number of past tokens processed.
    - `n_decoded`: An int32_t representing the number of tokens decoded.
    - `n_remaining`: An int32_t representing the number of remaining tokens to be processed, defaulting to -1.
    - `i_batch`: An int32_t representing the batch index, defaulting to -1.
    - `n_predict`: An int32_t representing the number of tokens to predict, defaulting to -1.
    - `n_prompt_tokens`: An int32_t representing the number of prompt tokens, which may differ from the size of prompt_tokens due to truncation.
    - `n_prompt_tokens_processed`: An int32_t representing the number of prompt tokens that have been processed.
    - `prompt_tokens`: A server_tokens structure containing the input prompt tokens.
    - `last_nl_pos`: A size_t representing the position of the last newline character in the generated text.
    - `generated_text`: A string containing the text generated by the server slot.
    - `generated_tokens`: A llama_tokens structure containing the tokens generated by the server slot.
    - `chat_msg`: A common_chat_msg structure representing the chat message associated with the server slot.
    - `cache_tokens`: A server_tokens structure containing cached tokens for the server slot.
    - `generated_token_probs`: A vector of completion_token_output structures representing the probabilities of generated tokens.
    - `has_next_token`: A boolean indicating whether there is a next token to be processed, defaulting to true.
    - `has_new_line`: A boolean indicating whether a new line has been generated, defaulting to false.
    - `truncated`: A boolean indicating whether the prompt has been truncated, defaulting to false.
    - `stop`: An enumeration indicating the stop type for the server slot.
    - `stopping_word`: A string representing the word that caused the generation to stop.
    - `json_schema`: A json object representing the JSON schema for sampling.
    - `smpl`: A pointer to a common_sampler structure used for sampling.
    - `sampled`: A llama_token representing the last sampled token.
    - `chat_format`: An enumeration indicating the chat format, defaulting to COMMON_CHAT_FORMAT_CONTENT_ONLY.
    - `generated_tool_call_ids`: A vector of strings representing the IDs of generated tool calls.
    - `n_sent_text`: A size_t representing the number of sent text characters.
    - `t_start_process_prompt`: An int64_t representing the start time of prompt processing.
    - `t_start_generation`: An int64_t representing the start time of token generation.
    - `t_prompt_processing`: A double representing the time taken for prompt processing in milliseconds.
    - `t_token_generation`: A double representing the time taken for token generation in milliseconds.
    - `callback_on_release`: A std::function that is called when the server slot is released.
    - `n_draft_total`: An int32_t representing the total number of draft tokens generated.
    - `n_draft_accepted`: An int32_t representing the number of draft tokens that were accepted.
- **Description**: The `server_slot` struct is a comprehensive data structure used to manage and track the state and operations of a server slot in a task processing system. It contains various fields to store identifiers, task types, context pointers, parameters, state information, and statistics related to token generation and processing. The struct supports multimodal processing, speculative decoding, and LoRA adapters, and it includes mechanisms for managing prompt tokens, generated text, and sampling operations. It is designed to handle complex task management scenarios, including batching, speculative decoding, and context management, making it a critical component in a server's task processing architecture.
- **Member Functions**:
    - [`server_slot::reset`](#server_slotreset)
    - [`server_slot::is_non_causal`](#server_slotis_non_causal)
    - [`server_slot::can_batch_with`](#server_slotcan_batch_with)
    - [`server_slot::has_budget`](#server_slothas_budget)
    - [`server_slot::is_processing`](#server_slotis_processing)
    - [`server_slot::can_speculate`](#server_slotcan_speculate)
    - [`server_slot::add_token`](#server_slotadd_token)
    - [`server_slot::release`](#server_slotrelease)
    - [`server_slot::get_timings`](#server_slotget_timings)
    - [`server_slot::update_chat_msg`](#server_slotupdate_chat_msg)
    - [`server_slot::find_stopping_strings`](#server_slotfind_stopping_strings)
    - [`server_slot::print_timings`](#server_slotprint_timings)
    - [`server_slot::to_json`](#server_slotto_json)

**Methods**

---
#### server\_slot::reset<!-- {{#callable:server_slot::reset}} -->
The `reset` function reinitializes various attributes of a `server_slot` object to their default states, effectively clearing any existing data and preparing the slot for a new task.
- **Inputs**: None
- **Control Flow**:
    - Logs a debug message indicating the reset operation.
    - Resets integer attributes like `n_prompt_tokens`, `last_nl_pos`, `n_past`, and `n_sent_text` to 0.
    - Resets boolean attributes like `has_new_line` and `truncated` to false.
    - Sets `stop` to `STOP_TYPE_NONE` and `task_type` to `SERVER_TASK_TYPE_COMPLETION`.
    - Sets `chat_format` to `COMMON_CHAT_FORMAT_CONTENT_ONLY`.
    - Clears string attributes like `generated_text` and `stopping_word`.
    - Clears vectors like `generated_tokens`, `generated_token_probs`, and `generated_tool_call_ids`.
    - Resets `chat_msg` to an empty state and `json_schema` to a new JSON object.
    - Resets speculative decoding stats `n_draft_total` and `n_draft_accepted` to 0.
- **Output**: The function does not return any value; it modifies the state of the `server_slot` object it is called on.
- **See also**: [`server_slot`](#server_slot)  (Data Structure)


---
#### server\_slot::is\_non\_causal<!-- {{#callable:server_slot::is_non_causal}} -->
The `is_non_causal` function checks if the `task_type` of a `server_slot` is either `SERVER_TASK_TYPE_EMBEDDING` or `SERVER_TASK_TYPE_RERANK`, indicating non-causal tasks.
- **Inputs**: None
- **Control Flow**:
    - The function checks if `task_type` is equal to `SERVER_TASK_TYPE_EMBEDDING` or `SERVER_TASK_TYPE_RERANK`.
    - If either condition is true, the function returns `true`.
    - If neither condition is true, the function returns `false`.
- **Output**: A boolean value indicating whether the `task_type` is non-causal (either `SERVER_TASK_TYPE_EMBEDDING` or `SERVER_TASK_TYPE_RERANK`).
- **See also**: [`server_slot`](#server_slot)  (Data Structure)


---
#### server\_slot::can\_batch\_with<!-- {{#callable:server_slot::can_batch_with}} -->
The `can_batch_with` function checks if two `server_slot` objects can be batched together based on their causal properties and LoRA configurations.
- **Inputs**:
    - `other_slot`: A reference to another `server_slot` object to compare with the current instance.
- **Control Flow**:
    - Check if the current slot is non-causal using `is_non_causal()` and compare it with the `other_slot`'s non-causal status.
    - Check if the LoRA configurations (`lora` vectors) of both slots are equal using `are_lora_equal()`.
    - Return true if both conditions are met, otherwise return false.
- **Output**: A boolean value indicating whether the two slots can be batched together.
- **Functions called**:
    - [`server_slot::is_non_causal`](#server_slotis_non_causal)
    - [`are_lora_equal`](utils.hpp.driver.md#are_lora_equal)
- **See also**: [`server_slot`](#server_slot)  (Data Structure)


---
#### server\_slot::has\_budget<!-- {{#callable:server_slot::has_budget}} -->
The `has_budget` function checks if there is a remaining budget for predictions based on the parameters provided.
- **Inputs**:
    - `global_params`: A constant reference to a `common_params` structure that contains global prediction parameters.
- **Control Flow**:
    - Check if both `params.n_predict` and `global_params.n_predict` are -1, indicating limitless predictions, and return true.
    - Initialize `n_remaining` to -1.
    - If `params.n_predict` is not -1, calculate `n_remaining` as `params.n_predict - n_decoded`.
    - Else if `global_params.n_predict` is not -1, calculate `n_remaining` as `global_params.n_predict - n_decoded`.
    - Return true if `n_remaining` is greater than 0, indicating there is a budget left; otherwise, return false.
- **Output**: A boolean value indicating whether there is a remaining budget for predictions.
- **See also**: [`server_slot`](#server_slot)  (Data Structure)


---
#### server\_slot::is\_processing<!-- {{#callable:server_slot::is_processing}} -->
The `is_processing` function checks if the server slot is currently in a state other than idle.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the `state` of the server slot is not equal to `SLOT_STATE_IDLE`.
    - It returns `true` if the state is not idle, indicating that the slot is processing, otherwise it returns `false`.
- **Output**: A boolean value indicating whether the server slot is currently processing (i.e., not idle).
- **See also**: [`server_slot`](#server_slot)  (Data Structure)


---
#### server\_slot::can\_speculate<!-- {{#callable:server_slot::can_speculate}} -->
The `can_speculate` function checks if speculative execution is possible based on certain conditions in the `server_slot` structure.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - The function checks if `ctx_dft` is not null, indicating a default context is available.
    - It verifies that `params.speculative.n_max` is greater than 0, ensuring there is a maximum number of speculative tokens allowed.
    - It checks if `params.cache_prompt` is true, indicating that the prompt caching is enabled.
- **Output**: The function returns a boolean value: `true` if all conditions are met, allowing speculative execution, otherwise `false`.
- **See also**: [`server_slot`](#server_slot)  (Data Structure)


---
#### server\_slot::add\_token<!-- {{#callable:server_slot::add_token}} -->
The `add_token` function adds a `completion_token_output` object to the `generated_token_probs` vector if the server slot is currently processing.
- **Inputs**:
    - `token`: A `completion_token_output` object representing a token and its associated probabilities to be added to the `generated_token_probs` vector.
- **Control Flow**:
    - Check if the server slot is currently processing by calling `is_processing()`.
    - If the slot is not processing, log a warning message and return immediately.
    - If the slot is processing, add the `token` to the `generated_token_probs` vector.
- **Output**: The function does not return any value.
- **Functions called**:
    - [`server_slot::is_processing`](#server_slotis_processing)
- **See also**: [`server_slot`](#server_slot)  (Data Structure)


---
#### server\_slot::release<!-- {{#callable:server_slot::release}} -->
The `release` function stops processing a server slot if it is currently processing, updates timing statistics, sets the slot state to idle, and triggers a callback function.
- **Inputs**:
    - `None`: The function does not take any input arguments.
- **Control Flow**:
    - Check if the server slot is currently processing using `is_processing()`.
    - If processing, log an informational message with `n_past` and `truncated` values.
    - Update `t_last_used` with the current time in microseconds using `ggml_time_us()`.
    - Calculate `t_token_generation` as the elapsed time since `t_start_generation` in milliseconds.
    - Set the slot's state to `SLOT_STATE_IDLE`.
    - Invoke the `callback_on_release` function with the slot's `id`.
- **Output**: The function does not return any value.
- **Functions called**:
    - [`server_slot::is_processing`](#server_slotis_processing)
- **See also**: [`server_slot`](#server_slot)  (Data Structure)


---
#### server\_slot::get\_timings<!-- {{#callable:server_slot::get_timings}} -->
The `get_timings` function calculates and returns various timing metrics related to prompt processing and token generation for a server slot.
- **Inputs**: None
- **Control Flow**:
    - Initialize a `result_timings` object named `timings`.
    - Set `timings.prompt_n` to the number of prompt tokens processed.
    - Set `timings.prompt_ms` to the time taken for prompt processing in milliseconds.
    - Calculate and set `timings.prompt_per_token_ms` as the average time per prompt token.
    - Calculate and set `timings.prompt_per_second` as the rate of prompt tokens processed per second.
    - Set `timings.predicted_n` to the number of tokens decoded.
    - Set `timings.predicted_ms` to the time taken for token generation in milliseconds.
    - Calculate and set `timings.predicted_per_token_ms` as the average time per predicted token.
    - Calculate and set `timings.predicted_per_second` as the rate of predicted tokens generated per second.
    - If speculative decoding metrics are available (i.e., `n_draft_total` > 0), set `timings.draft_n` and `timings.draft_n_accepted`.
    - Return the `timings` object.
- **Output**: A `result_timings` object containing metrics for prompt processing and token generation, including speculative metrics if applicable.
- **See also**: [`server_slot`](#server_slot)  (Data Structure)


---
#### server\_slot::update\_chat\_msg<!-- {{#callable:server_slot::update_chat_msg}} -->
The `update_chat_msg` function updates the current chat message with a new parsed message and computes the differences between the previous and new messages.
- **Inputs**:
    - `diffs`: A reference to a vector of `common_chat_msg_diff` objects that will be populated with the differences between the previous and new chat messages.
- **Control Flow**:
    - Store the current `chat_msg` in `previous_msg`.
    - Log the parsing of the chat message using `SRV_DBG`.
    - Parse the `generated_text` into a new chat message `new_msg` using `common_chat_parse`, considering if the message is partial based on the `stop` type and `params.oaicompat_chat_syntax`.
    - If `new_msg` is not empty, ensure tool call IDs are set in `new_msg` using `ensure_tool_call_ids_set`.
    - Update `chat_msg` with `new_msg`.
    - Compute the differences between `previous_msg` and `new_msg` using `common_chat_msg_diff::compute_diffs` and store them in `diffs`.
    - Return the updated `chat_msg`.
- **Output**: Returns a constant reference to the updated `common_chat_msg` object.
- **See also**: [`server_slot`](#server_slot)  (Data Structure)


---
#### server\_slot::find\_stopping\_strings<!-- {{#callable:server_slot::find_stopping_strings}} -->
The `find_stopping_strings` function searches for specific stopping words in a given text and returns the position of the first occurrence of any such word, considering whether a full stop is required.
- **Inputs**:
    - `text`: A reference to a constant string representing the text in which to search for stopping words.
    - `last_token_size`: A size_t value representing the size of the last token, used to calculate the search range when a full stop is required.
    - `is_full_stop`: A boolean indicating whether the search should consider a full stop (true) or a partial stop (false).
- **Control Flow**:
    - Initialize `stop_pos` to `std::string::npos` to indicate no stopping word found initially.
    - Iterate over each word in `params.antiprompt`, which contains the stopping words.
    - If `is_full_stop` is true, calculate `tmp` as the sum of the word size and `last_token_size`, and determine `from_pos` as the starting position for the search in `text`.
    - Use `text.find(word, from_pos)` to find the position of the word starting from `from_pos`.
    - If `is_full_stop` is false, use `string_find_partial_stop(text, word)` to find the position of the word considering partial stops.
    - If a word is found (`pos` is not `std::string::npos`) and it is the first found or occurs earlier than any previously found word, update `stop_pos` to `pos`.
    - If `is_full_stop` is true and a word is found, set `stop` to `STOP_TYPE_WORD`, update `stopping_word` to the found word, and set `has_next_token` to false.
    - Return `stop_pos`, which is the position of the first found stopping word or `std::string::npos` if none are found.
- **Output**: Returns a `size_t` representing the position of the first found stopping word in the text, or `std::string::npos` if no stopping word is found.
- **See also**: [`server_slot`](#server_slot)  (Data Structure)


---
#### server\_slot::print\_timings<!-- {{#callable:server_slot::print_timings}} -->
The `print_timings` function calculates and logs the average processing times and rates for prompt and token generation, and optionally logs the draft acceptance rate if applicable.
- **Inputs**: None
- **Control Flow**:
    - Calculate average time per token and tokens per second for prompt processing using `t_prompt_processing` and `n_prompt_tokens_processed`.
    - Calculate average time per token and tokens per second for token generation using `t_token_generation` and `n_decoded`.
    - Log the calculated prompt processing and token generation times and rates using `SLT_INF`.
    - If `n_draft_total` is greater than 0, calculate the draft acceptance rate and log it.
- **Output**: The function does not return any value; it logs the calculated timings and draft acceptance rate to the console.
- **See also**: [`server_slot`](#server_slot)  (Data Structure)


---
#### server\_slot::to\_json<!-- {{#callable:server_slot::to_json}} -->
The `to_json` function serializes the `server_slot` object into a JSON representation containing various attributes and states of the slot.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - The function constructs a JSON object using the `nlohmann::ordered_json` library.
    - It includes key-value pairs for attributes such as `id`, `id_task`, `n_ctx`, and others.
    - It calls methods like `can_speculate()`, `is_processing()`, and `is_non_causal()` to determine the values for `speculative`, [`is_processing`](#server_slotis_processing), and `non_causal` keys respectively.
    - The `params` attribute is serialized by calling its `to_json()` method.
    - The `prompt_tokens` are detokenized using the `detokenize` method with the context `ctx` and a boolean flag `true`.
    - A nested JSON object is created for `next_token` containing attributes like `has_next_token`, `has_new_line`, `n_remain`, `n_decoded`, and `stopping_word`.
- **Output**: The function returns a JSON object representing the state and attributes of the `server_slot` instance.
- **Functions called**:
    - [`server_slot::can_speculate`](#server_slotcan_speculate)
    - [`server_slot::is_processing`](#server_slotis_processing)
    - [`server_slot::is_non_causal`](#server_slotis_non_causal)
- **See also**: [`server_slot`](#server_slot)  (Data Structure)



---
### server\_metrics<!-- {{#data_structure:server_metrics}} -->
- **Type**: `struct`
- **Members**:
    - `t_start`: Stores the start time of the server metrics in microseconds.
    - `n_prompt_tokens_processed_total`: Total number of prompt tokens processed by the server.
    - `t_prompt_processing_total`: Total time spent processing prompts in microseconds.
    - `n_tokens_predicted_total`: Total number of tokens predicted by the server.
    - `t_tokens_generation_total`: Total time spent generating tokens in microseconds.
    - `n_prompt_tokens_processed`: Number of prompt tokens processed in the current bucket.
    - `t_prompt_processing`: Time spent processing prompts in the current bucket.
    - `n_tokens_predicted`: Number of tokens predicted in the current bucket.
    - `t_tokens_generation`: Time spent generating tokens in the current bucket.
    - `n_decode_total`: Total number of decode operations performed by the server.
    - `n_busy_slots_total`: Total number of busy slots during decode operations.
- **Description**: The `server_metrics` struct is designed to track various performance metrics of a server, particularly in the context of processing and generating tokens. It maintains both cumulative totals and current bucket values for prompt processing and token generation, allowing for detailed performance analysis over time. The struct also includes methods to initialize the start time, update metrics based on server slot activities, and reset the current bucket metrics.
- **Member Functions**:
    - [`server_metrics::init`](#server_metricsinit)
    - [`server_metrics::on_prompt_eval`](#server_metricson_prompt_eval)
    - [`server_metrics::on_prediction`](#server_metricson_prediction)
    - [`server_metrics::on_decoded`](#server_metricson_decoded)
    - [`server_metrics::reset_bucket`](#server_metricsreset_bucket)

**Methods**

---
#### server\_metrics::init<!-- {{#callable:server_metrics::init}} -->
The `init` function initializes the `t_start` member of the `server_metrics` structure with the current time in microseconds.
- **Inputs**: None
- **Control Flow**:
    - The function calls `ggml_time_us()` to get the current time in microseconds.
    - The result is assigned to the `t_start` member of the `server_metrics` structure.
- **Output**: The function does not return any value.
- **See also**: [`server_metrics`](#server_metrics)  (Data Structure)


---
#### server\_metrics::on\_prompt\_eval<!-- {{#callable:server_metrics::on_prompt_eval}} -->
The `on_prompt_eval` function updates the server metrics by accumulating the number of prompt tokens processed and the time taken for prompt processing from a given server slot.
- **Inputs**:
    - `slot`: A `server_slot` object containing metrics data such as the number of prompt tokens processed and the time taken for prompt processing.
- **Control Flow**:
    - The function accesses the `n_prompt_tokens_processed` and `t_prompt_processing` fields from the `slot` parameter.
    - It adds the `n_prompt_tokens_processed` from the `slot` to both `n_prompt_tokens_processed_total` and `n_prompt_tokens_processed` in the `server_metrics` structure.
    - It adds the `t_prompt_processing` from the `slot` to both `t_prompt_processing_total` and `t_prompt_processing` in the `server_metrics` structure.
- **Output**: The function does not return any value; it updates the metrics fields in the `server_metrics` structure.
- **See also**: [`server_metrics`](#server_metrics)  (Data Structure)


---
#### server\_metrics::on\_prediction<!-- {{#callable:server_metrics::on_prediction}} -->
The `on_prediction` function updates the server metrics related to token prediction and generation time using data from a given server slot.
- **Inputs**:
    - `slot`: A `server_slot` object containing metrics data such as the number of decoded tokens and token generation time.
- **Control Flow**:
    - Increment `n_tokens_predicted_total` by the number of decoded tokens in the slot.
    - Increment `n_tokens_predicted` by the number of decoded tokens in the slot.
    - Add the token generation time from the slot to `t_tokens_generation`.
    - Add the token generation time from the slot to `t_tokens_generation_total`.
- **Output**: This function does not return any value; it updates the server metrics in place.
- **See also**: [`server_metrics`](#server_metrics)  (Data Structure)


---
#### server\_metrics::on\_decoded<!-- {{#callable:server_metrics::on_decoded}} -->
The `on_decoded` function updates the total number of decode operations and counts the number of busy server slots from a given list of slots.
- **Inputs**:
    - `slots`: A constant reference to a vector of `server_slot` objects, representing the slots to be checked for processing status.
- **Control Flow**:
    - Increment the `n_decode_total` counter to track the total number of decode operations.
    - Iterate over each `server_slot` object in the `slots` vector.
    - For each `server_slot`, check if it is currently processing by calling its `is_processing()` method.
    - If a `server_slot` is processing, increment the `n_busy_slots_total` counter.
- **Output**: The function does not return any value; it updates the `n_decode_total` and `n_busy_slots_total` counters in the `server_metrics` structure.
- **See also**: [`server_metrics`](#server_metrics)  (Data Structure)


---
#### server\_metrics::reset\_bucket<!-- {{#callable:server_metrics::reset_bucket}} -->
The `reset_bucket` function resets specific server metrics related to token processing and generation to zero.
- **Inputs**: None
- **Control Flow**:
    - The function sets `n_prompt_tokens_processed` to 0.
    - The function sets `t_prompt_processing` to 0.
    - The function sets `n_tokens_predicted` to 0.
    - The function sets `t_tokens_generation` to 0.
- **Output**: The function does not return any value; it modifies the state of the `server_metrics` object by resetting certain fields to zero.
- **See also**: [`server_metrics`](#server_metrics)  (Data Structure)



---
### server\_queue<!-- {{#data_structure:server_queue}} -->
- **Type**: `struct`
- **Members**:
    - `id`: An integer identifier for the server queue, initialized to 0.
    - `running`: A boolean indicating whether the server queue is currently running.
    - `queue_tasks`: A deque of server_task objects representing the main task queue.
    - `queue_tasks_deferred`: A deque of server_task objects representing deferred tasks.
    - `mutex_tasks`: A mutex for synchronizing access to the task queues.
    - `condition_tasks`: A condition variable used to notify threads about task queue changes.
    - `callback_new_task`: A function to be called when a new task is added to the queue.
    - `callback_update_slots`: A function to be called when all slots data is ready to be processed.
- **Description**: The `server_queue` struct is a data structure designed to manage and process tasks in a server environment. It maintains two task queues: `queue_tasks` for active tasks and `queue_tasks_deferred` for tasks that are deferred until a slot is available. The struct uses a mutex and a condition variable to ensure thread-safe operations on the task queues. It also supports callback functions for handling new tasks and updating slots when tasks are processed. The `post` methods allow adding tasks to the queue, either at the front or back, and the `defer` method allows deferring tasks until a slot is available. The `start_loop` method runs the main loop for processing tasks, and the `terminate` method stops the loop.
- **Member Functions**:
    - [`server_queue::post`](#server_queuepost)
    - [`server_queue::post`](#server_queuepost)
    - [`server_queue::defer`](#server_queuedefer)
    - [`server_queue::get_new_id`](#server_queueget_new_id)
    - [`server_queue::on_new_task`](#server_queueon_new_task)
    - [`server_queue::on_update_slots`](#server_queueon_update_slots)
    - [`server_queue::pop_deferred_task`](#server_queuepop_deferred_task)
    - [`server_queue::terminate`](#server_queueterminate)
    - [`server_queue::start_loop`](#server_queuestart_loop)
    - [`server_queue::cleanup_pending_task`](#server_queuecleanup_pending_task)

**Methods**

---
#### server\_queue::post<!-- {{#callable:server_queue::post}} -->
The `post` function adds a new `server_task` to the `server_queue`, either at the front or back, and notifies a condition variable.
- **Inputs**:
    - `task`: A `server_task` object to be added to the queue, which must have a valid `id` (not equal to -1).
    - `front`: A boolean flag indicating whether the task should be added to the front of the queue (default is `false`).
- **Control Flow**:
    - Acquire a unique lock on `mutex_tasks` to ensure thread safety when accessing the task queue.
    - Assert that the task's `id` is not -1, ensuring it is valid.
    - If the task type is `SERVER_TASK_TYPE_CANCEL`, call [`cleanup_pending_task`](#server_queuecleanup_pending_task) to remove any pending tasks targeting the same `id_target`.
    - Log the new task's `id` and whether it is being added to the front of the queue.
    - Depending on the `front` flag, add the task to the front or back of `queue_tasks`.
    - Notify one waiting thread that a new task has been added by calling `condition_tasks.notify_one()`.
- **Output**: Returns the `id` of the task that was added to the queue.
- **Functions called**:
    - [`server_queue::cleanup_pending_task`](#server_queuecleanup_pending_task)
- **See also**: [`server_queue`](#server_queue)  (Data Structure)


---
#### server\_queue::post<!-- {{#callable:server_queue::post}} -->
The `post` function adds a batch of server tasks to a queue, optionally at the front, and notifies a condition variable to signal task availability.
- **Inputs**:
    - `tasks`: A vector of `server_task` objects to be added to the task queue.
    - `front`: A boolean flag indicating whether to add the tasks to the front of the queue (default is false).
- **Control Flow**:
    - Acquire a unique lock on the `mutex_tasks` to ensure thread safety while modifying the task queue.
    - Iterate over each task in the `tasks` vector.
    - For each task, if its `id` is -1, assign it a new unique `id` by incrementing the `id` member of the `server_queue`.
    - If the task type is `SERVER_TASK_TYPE_CANCEL`, call [`cleanup_pending_task`](#server_queuecleanup_pending_task) to remove any pending tasks with the target ID from the queue.
    - Log a debug message indicating the new task's ID, the total number of tasks, and whether it is being added to the front.
    - Depending on the `front` flag, add the task to the front or back of the `queue_tasks`.
    - Notify one waiting thread that a new task is available by calling `condition_tasks.notify_one()`.
- **Output**: Returns 0 to indicate successful execution.
- **Functions called**:
    - [`server_queue::cleanup_pending_task`](#server_queuecleanup_pending_task)
- **See also**: [`server_queue`](#server_queue)  (Data Structure)


---
#### server\_queue::defer<!-- {{#callable:server_queue::defer}} -->
The `defer` function adds a server task to a deferred task queue and notifies a condition variable.
- **Inputs**:
    - `task`: A `server_task` object that is being deferred, passed as an rvalue reference.
- **Control Flow**:
    - Acquire a unique lock on the `mutex_tasks` to ensure thread safety when accessing shared resources.
    - Log a debug message indicating the task is being deferred, including the task's ID.
    - Move the `task` into the `queue_tasks_deferred` deque to store it for later processing.
    - Notify one waiting thread that a new task has been added to the deferred queue using `condition_tasks.notify_one()`.
- **Output**: The function does not return any value.
- **See also**: [`server_queue`](#server_queue)  (Data Structure)


---
#### server\_queue::get\_new\_id<!-- {{#callable:server_queue::get_new_id}} -->
The `get_new_id` function generates a new unique identifier for tasks in a thread-safe manner by incrementing a shared integer.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Acquire a unique lock on the `mutex_tasks` to ensure thread safety.
    - Increment the shared `id` member variable and store the result in `new_id`.
    - Release the lock automatically when the function scope ends.
    - Return the newly generated `new_id`.
- **Output**: The function returns an integer representing the new unique task identifier.
- **See also**: [`server_queue`](#server_queue)  (Data Structure)


---
#### server\_queue::on\_new\_task<!-- {{#callable:server_queue::on_new_task}} -->
The `on_new_task` function registers a callback function to be invoked when a new server task is processed.
- **Inputs**:
    - `callback`: A `std::function` that takes a `server_task` rvalue reference as an argument, representing the function to be called when a new task is processed.
- **Control Flow**:
    - The function assigns the provided callback function to the `callback_new_task` member variable of the `server_queue` structure.
    - The callback function is moved into `callback_new_task` using `std::move` to transfer ownership.
- **Output**: This function does not return any value.
- **See also**: [`server_queue`](#server_queue)  (Data Structure)


---
#### server\_queue::on\_update\_slots<!-- {{#callable:server_queue::on_update_slots}} -->
The `on_update_slots` function registers a callback function to be called when all slots data is ready to be processed.
- **Inputs**:
    - `callback`: A `std::function<void(void)>` representing the callback function to be registered.
- **Control Flow**:
    - The function assigns the provided callback function to the `callback_update_slots` member variable of the `server_queue` structure.
- **Output**: The function does not return any value.
- **See also**: [`server_queue`](#server_queue)  (Data Structure)


---
#### server\_queue::pop\_deferred\_task<!-- {{#callable:server_queue::pop_deferred_task}} -->
The `pop_deferred_task` function moves a task from the deferred task queue to the main task queue and notifies a condition variable.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Acquire a unique lock on the `mutex_tasks` to ensure thread safety.
    - Check if the `queue_tasks_deferred` is not empty.
    - If not empty, move the front task from `queue_tasks_deferred` to the back of `queue_tasks`.
    - Remove the front task from `queue_tasks_deferred`.
    - Notify one waiting thread that a task has been added to `queue_tasks`.
- **Output**: The function does not return any value.
- **See also**: [`server_queue`](#server_queue)  (Data Structure)


---
#### server\_queue::terminate<!-- {{#callable:server_queue::terminate}} -->
The `terminate` function stops the server queue's task processing loop by setting the `running` flag to false and notifying all waiting threads.
- **Inputs**:
    - `None`: The function does not take any input arguments.
- **Control Flow**:
    - Acquire a unique lock on the `mutex_tasks` to ensure thread-safe access to shared resources.
    - Set the `running` flag to `false` to indicate that the server queue should stop processing tasks.
    - Call `condition_tasks.notify_all()` to wake up all threads that are waiting on the `condition_tasks` condition variable, allowing them to check the `running` flag and exit if necessary.
- **Output**: The function does not return any value.
- **See also**: [`server_queue`](#server_queue)  (Data Structure)


---
#### server\_queue::start\_loop<!-- {{#callable:server_queue::start_loop}} -->
The `start_loop` function continuously processes tasks from a queue, executes them using registered callbacks, and waits for new tasks while managing the running state of the server.
- **Inputs**: None
- **Control Flow**:
    - Set `running` to true to indicate the loop is active.
    - Enter an infinite loop to continuously process tasks.
    - Log the start of task processing.
    - Enter a nested loop to process tasks from `queue_tasks`.
    - Acquire a lock on `mutex_tasks` to safely access shared resources.
    - Check if `running` is false; if so, log termination and exit the function.
    - Check if `queue_tasks` is empty; if so, release the lock and break out of the nested loop.
    - Move the front task from `queue_tasks` and remove it from the queue.
    - Release the lock on `mutex_tasks`.
    - Log the processing of the current task and execute `callback_new_task` with the task.
    - After processing all tasks, log the update of slots and execute `callback_update_slots`.
    - Log waiting for new tasks and acquire a lock on `mutex_tasks`.
    - Check if `running` is false; if so, log termination and exit the function.
    - If `queue_tasks` is empty, wait on `condition_tasks` until a new task is available or `running` is false.
    - Release the lock on `mutex_tasks` and continue the outer loop.
- **Output**: The function does not return any value; it operates indefinitely until `running` is set to false.
- **See also**: [`server_queue`](#server_queue)  (Data Structure)


---
#### server\_queue::cleanup\_pending\_task<!-- {{#callable:server_queue::cleanup_pending_task}} -->
The `cleanup_pending_task` function removes tasks with a specific target ID from two task queues within the `server_queue` structure.
- **Inputs**:
    - `id_target`: An integer representing the target ID of tasks to be removed from the queues.
- **Control Flow**:
    - A lambda function `rm_func` is defined to check if a task's `id_target` matches the given `id_target`.
    - The `std::remove_if` algorithm is used with `rm_func` to identify tasks to be removed from `queue_tasks`.
    - The `erase` method is called on `queue_tasks` to remove the identified tasks.
    - The same process is repeated for `queue_tasks_deferred` to remove tasks with the matching `id_target`.
- **Output**: The function does not return any value; it modifies the `queue_tasks` and `queue_tasks_deferred` in place by removing tasks with the specified `id_target`.
- **See also**: [`server_queue`](#server_queue)  (Data Structure)



---
### server\_response<!-- {{#data_structure:server_response}} -->
- **Type**: `struct`
- **Members**:
    - `running`: Indicates whether the server is currently running.
    - `waiting_task_ids`: Stores the IDs of tasks that are waiting for a response.
    - `queue_results`: Holds the results of server tasks, using pointers for polymorphism.
    - `mutex_results`: A mutex to protect access to the results queue.
    - `condition_results`: A condition variable to manage waiting and notification of results.
- **Description**: The `server_response` struct is designed to manage the state and results of tasks processed by a server. It includes a flag to indicate if the server is running, a set to track task IDs that are awaiting responses, and a vector to store the results of these tasks. The struct also incorporates synchronization mechanisms, such as a mutex and a condition variable, to ensure thread-safe access and coordination when handling task results. This struct is crucial for managing asynchronous task processing and response delivery in a server environment.
- **Member Functions**:
    - [`server_response::add_waiting_task_id`](#server_responseadd_waiting_task_id)
    - [`server_response::add_waiting_tasks`](#server_responseadd_waiting_tasks)
    - [`server_response::remove_waiting_task_id`](#server_responseremove_waiting_task_id)
    - [`server_response::remove_waiting_task_ids`](#server_responseremove_waiting_task_ids)
    - [`server_response::recv`](#server_responserecv)
    - [`server_response::recv_with_timeout`](#server_responserecv_with_timeout)
    - [`server_response::recv`](#server_responserecv)
    - [`server_response::send`](#server_responsesend)
    - [`server_response::terminate`](#server_responseterminate)

**Methods**

---
#### server\_response::add\_waiting\_task\_id<!-- {{#callable:server_response::add_waiting_task_id}} -->
The `add_waiting_task_id` function adds a task ID to the set of task IDs that are waiting for a response in a thread-safe manner.
- **Inputs**:
    - `id_task`: An integer representing the ID of the task to be added to the waiting list.
- **Control Flow**:
    - Logs a debug message indicating the task ID being added and the current size of the waiting list before the addition.
    - Acquires a unique lock on the `mutex_results` to ensure thread safety while modifying the `waiting_task_ids` set.
    - Inserts the `id_task` into the `waiting_task_ids` set.
- **Output**: This function does not return any value.
- **See also**: [`server_response`](#server_response)  (Data Structure)


---
#### server\_response::add\_waiting\_tasks<!-- {{#callable:server_response::add_waiting_tasks}} -->
The `add_waiting_tasks` function adds a list of server tasks to the waiting list by inserting their IDs into a set, ensuring thread safety with a mutex lock.
- **Inputs**:
    - `tasks`: A constant reference to a vector of `server_task` objects, each containing an ID that needs to be added to the waiting list.
- **Control Flow**:
    - Acquire a unique lock on the `mutex_results` to ensure thread safety while modifying shared data.
    - Iterate over each `server_task` object in the `tasks` vector.
    - For each task, log a debug message indicating the task ID being added and the current size of the waiting list before the addition.
    - Insert the task's ID into the `waiting_task_ids` unordered set.
- **Output**: The function does not return any value; it modifies the `waiting_task_ids` set in place.
- **See also**: [`server_response`](#server_response)  (Data Structure)


---
#### server\_response::remove\_waiting\_task\_id<!-- {{#callable:server_response::remove_waiting_task_id}} -->
The `remove_waiting_task_id` function removes a specified task ID from the waiting list and cleans up any pending results associated with that task ID.
- **Inputs**:
    - `id_task`: An integer representing the ID of the task to be removed from the waiting list.
- **Control Flow**:
    - Logs the removal of the task ID from the waiting list, including the current size of the waiting list before removal.
    - Acquires a unique lock on the `mutex_results` to ensure thread safety while modifying shared data.
    - Removes the specified `id_task` from the `waiting_task_ids` unordered set.
    - Uses `std::remove_if` to find and erase any results in `queue_results` that are associated with the `id_task`.
- **Output**: The function does not return any value.
- **See also**: [`server_response`](#server_response)  (Data Structure)


---
#### server\_response::remove\_waiting\_task\_ids<!-- {{#callable:server_response::remove_waiting_task_ids}} -->
The `remove_waiting_task_ids` function removes a set of task IDs from the `waiting_task_ids` set in a thread-safe manner.
- **Inputs**:
    - `id_tasks`: A constant reference to an unordered set of integers representing task IDs to be removed from the waiting list.
- **Control Flow**:
    - Acquire a unique lock on the `mutex_results` to ensure thread safety while modifying shared data.
    - Iterate over each task ID in the `id_tasks` set.
    - For each task ID, log a debug message indicating the task ID being removed and the current size of the waiting list before removal.
    - Remove the task ID from the `waiting_task_ids` set.
- **Output**: The function does not return any value; it modifies the `waiting_task_ids` set in place.
- **See also**: [`server_response`](#server_response)  (Data Structure)


---
#### server\_response::recv<!-- {{#callable:server_response::recv}} -->
The `recv` function blocks the calling thread until a result for one of the specified task IDs is available in the result queue, and then returns that result.
- **Inputs**:
    - `id_tasks`: A constant reference to an unordered set of integers representing the IDs of tasks for which the function is waiting to receive results.
- **Control Flow**:
    - The function enters an infinite loop to continuously check for results.
    - A unique lock is acquired on the `mutex_results` to ensure thread safety when accessing shared resources.
    - The function waits on the `condition_results` condition variable until the queue is not empty or the `running` flag is false.
    - If `running` is false, the function logs a debug message and calls `std::terminate()` to stop execution, as it cannot return due to HTTP code constraints.
    - Once the queue is not empty, the function iterates over the `queue_results` to find a result with an ID that matches one in `id_tasks`.
    - If a matching result is found, it is moved out of the queue and returned to the caller.
- **Output**: A `server_task_result_ptr`, which is a unique pointer to a `server_task_result` object, representing the result of a task with an ID in `id_tasks`.
- **See also**: [`server_response`](#server_response)  (Data Structure)


---
#### server\_response::recv\_with\_timeout<!-- {{#callable:server_response::recv_with_timeout}} -->
The `recv_with_timeout` function waits for a response from a set of task IDs with a specified timeout, returning the result if available or nullptr if the timeout is reached.
- **Inputs**:
    - `id_tasks`: A constant reference to an unordered set of integers representing the IDs of tasks for which a response is awaited.
    - `timeout`: An integer specifying the timeout duration in seconds for waiting for a response.
- **Control Flow**:
    - Enter an infinite loop to continuously check for task results.
    - Acquire a unique lock on the mutex `mutex_results` to ensure thread-safe access to shared resources.
    - Iterate over the `queue_results` to find a result whose ID matches any in the `id_tasks` set.
    - If a matching result is found, move it from the queue, erase it, and return it.
    - Wait on the condition variable `condition_results` for the specified `timeout` duration.
    - If the `running` flag is false, log a debug message and terminate the program as the caller is HTTP code and cannot return.
    - If the wait operation times out, return nullptr indicating no result was received within the timeout period.
- **Output**: Returns a `server_task_result_ptr`, which is a unique pointer to a `server_task_result`. It returns the result if found within the timeout, or nullptr if the timeout is reached.
- **See also**: [`server_response`](#server_response)  (Data Structure)


---
#### server\_response::recv<!-- {{#callable:server_response::recv}} -->
The [`recv`](#server_responserecv) function retrieves the result of a server task by blocking the thread until a response for the specified task ID is available.
- **Inputs**:
    - `id_task`: An integer representing the ID of the task for which the result is being requested.
- **Control Flow**:
    - The function creates an unordered set containing the single task ID provided as input.
    - It calls the overloaded [`recv`](#server_responserecv) function that accepts a set of task IDs, passing the created set as an argument.
    - The [`recv`](#server_responserecv) function blocks the thread until a response for one of the task IDs in the set is available.
    - Once a response is available, it returns the result associated with the task ID.
- **Output**: A `server_task_result_ptr`, which is a pointer to the result of the server task associated with the given task ID.
- **Functions called**:
    - [`server_response::recv`](#server_responserecv)
- **See also**: [`server_response`](#server_response)  (Data Structure)


---
#### server\_response::send<!-- {{#callable:server_response::send}} -->
The `send` function adds a task result to the result queue if the task ID is in the waiting list and notifies all waiting threads.
- **Inputs**:
    - `result`: A `server_task_result_ptr` object representing the result of a server task, which is an rvalue reference.
- **Control Flow**:
    - Log the task ID of the result being sent.
    - Acquire a unique lock on the `mutex_results` to ensure thread safety when accessing shared resources.
    - Iterate over the `waiting_task_ids` to check if the result's task ID is in the waiting list.
    - If the result's task ID matches a waiting task ID, log that the task ID is being pushed to the result queue.
    - Move the result into the `queue_results` vector to store it.
    - Notify all threads waiting on `condition_results` to signal that a new result is available.
    - Return from the function, ending execution.
- **Output**: The function does not return a value; it modifies the state of the `queue_results` and potentially wakes up waiting threads.
- **See also**: [`server_response`](#server_response)  (Data Structure)


---
#### server\_response::terminate<!-- {{#callable:server_response::terminate}} -->
The `terminate` function stops the server's operation by setting the `running` flag to false and notifying all threads waiting on the `condition_results` condition variable.
- **Inputs**: None
- **Control Flow**:
    - Set the `running` flag to false, indicating the server should stop running.
    - Notify all threads waiting on the `condition_results` condition variable to wake up and check the `running` flag.
- **Output**: The function does not return any value.
- **See also**: [`server_response`](#server_response)  (Data Structure)



---
### server\_context<!-- {{#data_structure:server_context}} -->
- **Type**: `struct`
- **Members**:
    - `params_base`: Holds common parameters for server configuration.
    - `llama_init`: Stores the result of the initial model initialization.
    - `llama_init_dft`: Stores the result of the draft model initialization.
    - `model`: Pointer to the main llama model.
    - `ctx`: Pointer to the main llama context.
    - `mctx`: Pointer to the multimodal context, if used.
    - `vocab`: Pointer to the llama vocabulary.
    - `model_dft`: Pointer to the draft llama model, if used.
    - `cparams_dft`: Holds context parameters for the draft model.
    - `batch`: Represents a batch of llama tokens for processing.
    - `clean_kv_cache`: Indicates whether the key-value cache should be cleaned.
    - `add_bos_token`: Indicates whether to add a beginning-of-sequence token.
    - `has_eos_token`: Indicates whether an end-of-sequence token is present.
    - `n_ctx`: Total context size available for all clients or slots.
    - `slots`: Vector of server slots for handling multiple clients.
    - `default_generation_settings_for_props`: JSON object holding default generation settings.
    - `queue_tasks`: Queue for managing server tasks.
    - `queue_results`: Queue for managing server task results.
    - `metrics`: Tracks server performance metrics.
    - `slot_prompt_similarity`: Threshold for prompt similarity when selecting slots.
    - `chat_templates`: Pointer to chat templates used for formatting.
    - `oai_parser_opt`: Options for parsing OpenAI-compatible requests.
- **Description**: The `server_context` struct is a comprehensive data structure designed to manage the state and operations of a server handling llama models. It includes pointers to the main and draft llama models and contexts, as well as configurations for multimodal processing. The struct maintains a collection of server slots to handle multiple client requests concurrently, and it manages task and result queues to facilitate asynchronous processing. Additionally, it tracks server metrics and holds settings for chat templates and OpenAI-compatible parsing options. The struct is central to the server's operation, ensuring efficient model loading, task processing, and response generation.
- **Member Functions**:
    - [`server_context::~server_context`](#server_contextserver_context)
    - [`server_context::load_model`](#server_contextload_model)
    - [`server_context::init`](#server_contextinit)
    - [`server_context::get_slot_by_id`](#server_contextget_slot_by_id)
    - [`server_context::get_available_slot`](#server_contextget_available_slot)
    - [`server_context::launch_slot_with_task`](#server_contextlaunch_slot_with_task)
    - [`server_context::kv_cache_clear`](#server_contextkv_cache_clear)
    - [`server_context::process_token`](#server_contextprocess_token)
    - [`server_context::populate_token_probs`](#server_contextpopulate_token_probs)
    - [`server_context::send_error`](#server_contextsend_error)
    - [`server_context::send_error`](#server_contextsend_error)
    - [`server_context::send_error`](#server_contextsend_error)
    - [`server_context::ensure_no_mtmd`](#server_contextensure_no_mtmd)
    - [`server_context::send_partial_response`](#server_contextsend_partial_response)
    - [`server_context::send_final_response`](#server_contextsend_final_response)
    - [`server_context::send_embedding`](#server_contextsend_embedding)
    - [`server_context::send_rerank`](#server_contextsend_rerank)
    - [`server_context::cancel_tasks`](#server_contextcancel_tasks)
    - [`server_context::receive_multi_results`](#server_contextreceive_multi_results)
    - [`server_context::receive_cmpl_results_stream`](#server_contextreceive_cmpl_results_stream)
    - [`server_context::process_single_task`](#server_contextprocess_single_task)
    - [`server_context::update_slots`](#server_contextupdate_slots)
    - [`server_context::model_meta`](#server_contextmodel_meta)

**Methods**

---
#### server\_context::\~server\_context<!-- {{#callable:server_context::~server_context}} -->
The destructor `~server_context()` is responsible for freeing resources and clearing contexts associated with the `server_context` object.
- **Inputs**: None
- **Control Flow**:
    - Call `mtmd_free(mctx)` to free the multimodal context `mctx`.
    - Iterate over each `server_slot` in `slots` to free associated resources.
    - For each `server_slot`, call `common_sampler_free(slot.smpl)` and set `slot.smpl` to `nullptr`.
    - Call `llama_free(slot.ctx_dft)` and set `slot.ctx_dft` to `nullptr`.
    - Call `common_speculative_free(slot.spec)` and set `slot.spec` to `nullptr`.
    - Call `llama_batch_free(slot.batch_spec)` to free the batch specification.
    - Call `llama_batch_free(batch)` to free the main batch.
- **Output**: The function does not return any value as it is a destructor.
- **Functions called**:
    - [`mtmd_free`](../mtmd/mtmd.cpp.driver.md#mtmd_free)
    - [`common_sampler_free`](../../common/sampling.cpp.driver.md#common_sampler_free)
    - [`common_speculative_free`](../../common/speculative.cpp.driver.md#common_speculative_free)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::load\_model<!-- {{#callable:server_context::load_model}} -->
The `load_model` function initializes and loads a machine learning model and its associated components based on the provided parameters, handling both primary and speculative models, and setting up necessary contexts and templates.
- **Inputs**:
    - `params`: A `common_params` structure containing configuration details for loading the model, including paths, devices, context settings, and speculative model parameters.
- **Control Flow**:
    - Log the start of the model loading process using the model path from `params`.
    - Assign `params` to `params_base` for further use.
    - Initialize the primary model and context using `common_init_from_params` with `params_base`.
    - Check if the model was successfully loaded; if not, log an error and return `false`.
    - Retrieve the vocabulary from the loaded model and set context-related flags (`add_bos_token`, `has_eos_token`).
    - If speculative model parameters are provided, initialize a draft model with adjusted parameters and check compatibility with the primary model.
    - Initialize chat templates and handle potential parsing errors, falling back to a default template if necessary.
    - If a multimodal project path is provided, initialize a multimodal context and handle unsupported features by disabling them.
    - Check if context shifting is supported and adjust related parameters accordingly.
    - Return `true` if all initializations are successful.
- **Output**: A boolean value indicating whether the model and its components were successfully loaded and initialized (`true` for success, `false` for failure).
- **Functions called**:
    - [`common_speculative_are_compatible`](../../common/speculative.cpp.driver.md#common_speculative_are_compatible)
    - [`mtmd_context_params_default`](../mtmd/mtmd.cpp.driver.md#mtmd_context_params_default)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::init<!-- {{#callable:server_context::init}} -->
The `init` function initializes server slots and related settings for handling tasks in a server context.
- **Inputs**: None
- **Control Flow**:
    - Calculate the number of context slots per parallel task by dividing total context by the number of parallel tasks.
    - Log the initialization of slots with the number of parallel tasks.
    - Iterate over the number of parallel tasks to initialize each server slot.
    - For each slot, set its ID, context, context size, prediction count, multimodal context, and cache token settings.
    - If a draft model is available, initialize batch specifications and draft context for the slot, logging errors if initialization fails.
    - Log the creation of a new slot with its context size.
    - Set the slot's sampling parameters and define a callback for when the slot is released.
    - Reset the slot and add it to the list of slots.
    - Set default generation settings based on the first slot's JSON representation.
    - Initialize a batch with the maximum of batch size or parallel tasks, considering non-causal models.
    - Initialize server metrics.
    - Set options for the OpenAI-compatible parser based on server parameters and multimodal support.
- **Output**: The function does not return any value; it initializes the server context's slots and related settings.
- **Functions called**:
    - [`common_speculative_init`](../../common/speculative.cpp.driver.md#common_speculative_init)
    - [`mtmd_support_vision`](../mtmd/mtmd.cpp.driver.md#mtmd_support_vision)
    - [`mtmd_support_audio`](../mtmd/mtmd.cpp.driver.md#mtmd_support_audio)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::get\_slot\_by\_id<!-- {{#callable:server_context::get_slot_by_id}} -->
The `get_slot_by_id` function retrieves a pointer to a `server_slot` object from a vector of slots based on a given slot ID.
- **Inputs**:
    - `id`: An integer representing the ID of the slot to be retrieved.
- **Control Flow**:
    - Iterate over each `server_slot` object in the `slots` vector.
    - Check if the `id` of the current `server_slot` matches the input `id`.
    - If a match is found, return a pointer to the matching `server_slot`.
    - If no match is found after iterating through all slots, return `nullptr`.
- **Output**: A pointer to the `server_slot` object with the matching ID, or `nullptr` if no such slot is found.
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::get\_available\_slot<!-- {{#callable:server_context::get_available_slot}} -->
The `get_available_slot` function selects an available server slot based on prompt similarity or least recently used criteria.
- **Inputs**:
    - `task`: A `server_task` object containing the prompt tokens to be matched against available slots.
- **Control Flow**:
    - Initialize `ret` to `nullptr` to store the selected slot.
    - Check if `ret` is `nullptr` and `slot_prompt_similarity` is not zero to find a slot with prompt similarity.
    - Iterate over each `server_slot` in `slots`.
    - Skip slots that are processing or have empty cached tokens.
    - Calculate the longest common subsequence (LCS) length and similarity between the slot's cached tokens and the task's prompt tokens.
    - Update `ret` to the current slot if it has a higher LCS length and similarity than previously found slots.
    - Log the selected slot if found by LCS similarity.
    - If no slot is found by similarity, find the least recently used slot by iterating over `slots` again.
    - Skip slots that are processing.
    - Update `ret` to the slot with the earliest `t_last_used` timestamp.
    - Log the selected slot if found by least recently used criteria.
    - Return the selected slot `ret`.
- **Output**: A pointer to a `server_slot` object that is available for processing the task.
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::launch\_slot\_with\_task<!-- {{#callable:server_context::launch_slot_with_task}} -->
The `launch_slot_with_task` function initializes a server slot with a given task, ensuring compatibility and readiness for processing.
- **Inputs**:
    - `slot`: A reference to a `server_slot` object that will be initialized and configured for the task.
    - `task`: An rvalue reference to a `server_task` object containing the task details to be assigned to the slot.
- **Control Flow**:
    - Reset the slot to clear any previous state.
    - Assign the task's ID, index, type, parameters, and prompt tokens to the slot.
    - Check if the LoRA (Low-Rank Adaptation) parameters have changed; if so, clear the slot's cache tokens and update the LoRA settings.
    - Validate the prompt tokens against the server context; if invalid, send an error and return false.
    - Log the slot's launch details for debugging purposes.
    - Check if the task's prediction count exceeds the slot's limit; if so, adjust the task's prediction count to the slot's limit and log a warning.
    - If the task is set to ignore the end-of-sequence token and the server has such a token, adjust the sampling logit bias to ignore it.
    - Initialize or reinitialize the slot's sampler with the task's sampling parameters; if initialization fails, send an error and return false.
    - If the slot has a draft context, free the existing batch specification and initialize a new one with the task's speculative parameters.
    - Set the slot's state to `SLOT_STATE_STARTED` to indicate it is ready to process the task.
    - Log an informational message indicating the task is being processed.
    - Return true to indicate successful initialization and readiness.
- **Output**: A boolean value indicating whether the slot was successfully initialized with the task (true) or if an error occurred (false).
- **Functions called**:
    - [`are_lora_equal`](utils.hpp.driver.md#are_lora_equal)
    - [`server_context::send_error`](#server_contextsend_error)
    - [`safe_json_to_str`](utils.hpp.driver.md#safe_json_to_str)
    - [`common_sampler_free`](../../common/sampling.cpp.driver.md#common_sampler_free)
    - [`common_sampler_init`](../../common/sampling.cpp.driver.md#common_sampler_init)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::kv\_cache\_clear<!-- {{#callable:server_context::kv_cache_clear}} -->
The `kv_cache_clear` function clears the entire key-value (KV) cache and sets the `clean_kv_cache` flag to false.
- **Inputs**: None
- **Control Flow**:
    - Logs a debug message indicating the KV cache is being cleared.
    - Calls `llama_kv_self_clear(ctx)` to clear the entire KV cache.
    - Sets the `clean_kv_cache` flag to false.
- **Output**: The function does not return any value.
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::process\_token<!-- {{#callable:server_context::process_token}} -->
The `process_token` function processes a token for a server slot, handling text generation, stopping conditions, and context management.
- **Inputs**:
    - `result`: A reference to a `completion_token_output` object containing the token and text to be processed.
    - `slot`: A reference to a `server_slot` object representing the current server slot where the token is being processed.
- **Control Flow**:
    - The function starts by storing the token string and updating the slot's sampled token and generated text.
    - If `return_tokens` is true in the slot's parameters, the token is added to the slot's generated tokens.
    - The function checks for incomplete UTF-8 characters at the end of the generated text.
    - If the text is complete, it searches for stopping strings and removes them if found, updating the text to send.
    - If the text is incomplete, it sets `has_next_token` to true.
    - The function checks if context shifting is disabled and if the context limit is reached, setting stop conditions if necessary.
    - It checks if the slot has exceeded its budget and sets stop conditions if necessary.
    - If a new line is detected, it checks for indentation limits and sets stop conditions if necessary.
    - The function checks for a new line in the text to send and applies a time limit if necessary.
    - It checks if the context limit is reached and sets stop conditions if necessary.
    - The function checks if the token is an end-of-sequence token and sets stop conditions if necessary.
    - It checks if the number of predicted tokens is set for infinite generation and limits it to avoid infinite loops.
    - Finally, it logs the current state and returns whether there is a next token to process.
- **Output**: Returns a boolean indicating whether there is a next token to process (`true` if there is, `false` otherwise).
- **Functions called**:
    - [`validate_utf8`](utils.hpp.driver.md#validate_utf8)
    - [`server_context::send_partial_response`](#server_contextsend_partial_response)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::populate\_token\_probs<!-- {{#callable:server_context::populate_token_probs}} -->
The `populate_token_probs` function populates the probabilities of tokens for a given server slot and completion token output, either post-sampling or pre-sampling, based on the provided parameters.
- **Inputs**:
    - `slot`: A `server_slot` object that contains parameters and context for token sampling.
    - `result`: A `completion_token_output` object where the probabilities of tokens will be stored.
    - `post_sampling`: A boolean indicating whether the probabilities should be populated post-sampling.
    - `special`: A boolean indicating whether special tokens should be considered.
    - `idx`: An integer index used to retrieve token probabilities when not post-sampling.
- **Control Flow**:
    - Determine the number of probabilities (`n_probs`) and vocabulary size (`n_vocab`) from the slot's parameters and context.
    - If `post_sampling` is true, retrieve the current candidates from the sampler and set the probability for the sampled token in `result`.
    - Reserve space in `result.probs` for the top `n_probs` tokens and populate it with token IDs, their text pieces, and probabilities from the sampler's candidates.
    - If `post_sampling` is false, retrieve token probabilities using the context and index, then set the probability for the sampled token in `result`.
    - Reserve space in `result.probs` for the top `n_probs` tokens and populate it with token IDs, their text pieces, and probabilities from the retrieved token probabilities.
- **Output**: The function modifies the `result` object to include the probability of the sampled token and a list of probabilities for the top `n_probs` tokens.
- **Functions called**:
    - [`common_sampler_get_candidates`](../../common/sampling.cpp.driver.md#common_sampler_get_candidates)
    - [`get_token_probabilities`](utils.hpp.driver.md#get_token_probabilities)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::send\_error<!-- {{#callable:server_context::send_error}} -->
The [`send_error`](#server_contextsend_error) function sends an error message related to a specific server task to the results queue.
- **Inputs**:
    - `task`: A `server_task` object representing the task for which the error is being reported.
    - `error`: A `std::string` containing the error message to be sent.
    - `type`: An `enum error_type` indicating the type of error, defaulting to `ERROR_TYPE_SERVER`.
- **Control Flow**:
    - The function calls another [`send_error`](#server_contextsend_error) function, passing the task's ID, the error message, and the error type.
    - The called [`send_error`](#server_contextsend_error) function logs the error message with the task ID.
    - A `server_task_result_error` object is created and populated with the task ID, error type, and error message.
    - The error result is sent to the `queue_results` using the `send` method.
- **Output**: There is no return value; the function sends an error result to the results queue.
- **Functions called**:
    - [`server_context::send_error`](#server_contextsend_error)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::send\_error<!-- {{#callable:server_context::send_error}} -->
The [`send_error`](#server_contextsend_error) function sends an error message related to a specific task or slot to the server's result queue.
- **Inputs**:
    - `slot`: A reference to a `server_slot` object, which contains information about the task and context associated with the error.
    - `error`: A string containing the error message to be sent.
    - `type`: An optional enumeration value of type `error_type` indicating the type of error, defaulting to `ERROR_TYPE_SERVER`.
- **Control Flow**:
    - The function calls another overloaded [`send_error`](#server_contextsend_error) function, passing the `id_task` from the `slot`, the `error` message, and the `type` of error.
    - The called [`send_error`](#server_contextsend_error) function logs the error message with the task ID and creates a `server_task_result_error` object.
    - The error object is populated with the task ID, error type, and error message.
    - The error object is then sent to the `queue_results` to be processed.
- **Output**: There is no return value; the function sends an error message to the server's result queue.
- **Functions called**:
    - [`server_context::send_error`](#server_contextsend_error)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::send\_error<!-- {{#callable:server_context::send_error}} -->
The `send_error` function logs an error message and sends an error result to a queue for a specific task ID.
- **Inputs**:
    - `id_task`: An integer representing the ID of the task for which the error occurred.
    - `error`: A string containing the error message to be logged and sent.
    - `type`: An optional enumeration value of type `error_type` indicating the type of error, defaulting to `ERROR_TYPE_SERVER`.
- **Control Flow**:
    - Log the error message using the `SRV_ERR` macro, including the task ID and error message.
    - Create a unique pointer to a `server_task_result_error` object.
    - Set the `id`, `err_type`, and `err_msg` fields of the error result object with the provided task ID, error type, and error message.
    - Send the error result object to the `queue_results` using the `send` method.
- **Output**: The function does not return a value; it sends an error result to a queue.
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::ensure\_no\_mtmd<!-- {{#callable:server_context::ensure_no_mtmd}} -->
The `ensure_no_mtmd` function checks if the multimodal context (`mctx`) is enabled and sends an error if it is, returning false; otherwise, it returns true.
- **Inputs**:
    - `id_task`: An integer representing the task ID for which the check is being performed.
- **Control Flow**:
    - Check if the multimodal context (`mctx`) is not null.
    - If `mctx` is not null, call [`send_error`](#server_contextsend_error) with the task ID, an error message indicating the feature is not supported by multimodal, and the error type `ERROR_TYPE_NOT_SUPPORTED`.
    - Return false if `mctx` is not null.
    - Return true if `mctx` is null.
- **Output**: A boolean value indicating whether the multimodal context is not enabled (true) or is enabled (false).
- **Functions called**:
    - [`server_context::send_error`](#server_contextsend_error)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::send\_partial\_response<!-- {{#callable:server_context::send_partial_response}} -->
The `send_partial_response` function sends a partial response for a server task by creating a result object with task details and sending it to the results queue.
- **Inputs**:
    - `slot`: A reference to a `server_slot` object that contains information about the current task and its parameters.
    - `tkn`: A constant reference to a `completion_token_output` object that contains the token and text to be sent as part of the response.
- **Control Flow**:
    - Create a unique pointer to a `server_task_result_cmpl_partial` object.
    - Set the `id`, `index`, `content`, and `tokens` fields of the result object using the `slot` and `tkn` inputs.
    - Copy various parameters from the `slot` to the result object, such as `n_decoded`, `n_prompt_tokens`, `post_sampling_probs`, `verbose`, `oaicompat`, `oaicompat_model`, and `oaicompat_cmpl_id`.
    - Update the chat message in the `slot` using `update_chat_msg` and store the differences in `oaicompat_msg_diffs`.
    - If `n_probs` is greater than 0, copy the token probabilities from `tkn` to `prob_output`.
    - If the task is a final response or `timings_per_token` is enabled, populate the `timings` field using `get_timings`.
    - Send the result object to the `queue_results` using `queue_results.send`.
- **Output**: The function does not return a value; it sends a `server_task_result_cmpl_partial` object to the `queue_results`.
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::send\_final\_response<!-- {{#callable:server_context::send_final_response}} -->
The `send_final_response` function constructs a final response object with the results of a server task and sends it to the results queue.
- **Inputs**:
    - `slot`: A reference to a `server_slot` object containing the task's execution context and results.
- **Control Flow**:
    - Create a unique pointer `res` to a `server_task_result_cmpl_final` object.
    - Populate `res` with various fields from the `slot`, including task ID, slot ID, generated text, tokens, timings, prompt, and response fields.
    - Check if sampling probabilities should be included and populate `res->probs_output` accordingly.
    - Copy the generation parameters from `slot` to `res->generation_params`.
    - Send the populated `res` object to the `queue_results`.
- **Output**: The function does not return a value; it sends the final response object to the `queue_results`.
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::send\_embedding<!-- {{#callable:server_context::send_embedding}} -->
The `send_embedding` function processes a batch of tokens to generate embeddings and sends the results to a queue for further processing.
- **Inputs**:
    - `slot`: A `server_slot` object containing task-specific information such as task ID, index, number of prompt tokens, and parameters.
    - `batch`: A `llama_batch` object containing tokens and associated data for which embeddings need to be generated.
- **Control Flow**:
    - Create a unique pointer `res` to a `server_task_result_embd` object and initialize it with task-specific information from `slot`.
    - Determine the number of embeddings `n_embd` using the `llama_model_n_embd` function.
    - Initialize a vector `embd_res` to store normalized embeddings, if needed.
    - Iterate over each token in `batch` to process embeddings.
    - For each token, check if it is valid and belongs to the current slot; if not, continue to the next token.
    - Retrieve embeddings using `llama_get_embeddings_seq` or `llama_get_embeddings_ith` functions.
    - If embeddings retrieval fails, log an error and append a zero-filled vector to `res->embedding`.
    - If pooling is enabled, normalize the embeddings using `common_embd_normalize` and append to `res->embedding`; otherwise, directly append the embeddings.
    - Log a debug message indicating that embeddings are being sent.
    - Send the `res` object to the `queue_results` for further processing.
- **Output**: The function does not return a value; it sends the generated embeddings as a `server_task_result_embd` object to a results queue.
- **Functions called**:
    - [`llama_pooling_type`](../../include/llama.h.driver.md#llama_pooling_type)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::send\_rerank<!-- {{#callable:server_context::send_rerank}} -->
The `send_rerank` function processes a batch of tokens to compute and send a rerank score for a specific server slot.
- **Inputs**:
    - `slot`: A `server_slot` object representing the server slot for which the rerank score is being computed.
    - `batch`: A `llama_batch` object containing the batch of tokens to be processed for reranking.
- **Control Flow**:
    - Create a unique pointer `res` to a `server_task_result_rerank` object and initialize its `id`, `index`, and `n_tokens` fields using the `slot` data.
    - Iterate over each token in the `batch` using a for loop.
    - For each token, check if the token's logits are valid and if the sequence ID matches the slot ID; if not, continue to the next token.
    - Attempt to retrieve embeddings for the current token using `llama_get_embeddings_seq` and `llama_get_embeddings_ith`; if both return `NULL`, log an error and set `res->score` to a large negative value, then continue to the next token.
    - If embeddings are successfully retrieved, set `res->score` to the first element of the embeddings array.
    - Log the rerank result score using `SLT_DBG`.
    - Send the `res` object to the `queue_results` using `queue_results.send`.
- **Output**: The function does not return a value; it sends a `server_task_result_rerank` object with the computed score to the `queue_results`.
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::cancel\_tasks<!-- {{#callable:server_context::cancel_tasks}} -->
The `cancel_tasks` function cancels a set of tasks identified by their IDs by removing them from the waiting queue and posting them to the task queue with high priority.
- **Inputs**:
    - `id_tasks`: A constant reference to an unordered set of integers representing the IDs of the tasks to be canceled.
- **Control Flow**:
    - Initialize a vector `cancel_tasks` to store tasks to be canceled and reserve space based on the size of `id_tasks`.
    - Iterate over each task ID in `id_tasks`.
    - Log a warning message indicating the cancellation of the task with the current ID.
    - Create a `server_task` object of type `SERVER_TASK_TYPE_CANCEL` and set its `id_target` to the current task ID.
    - Remove the task ID from the `queue_results` waiting list.
    - Move the created task into the `cancel_tasks` vector.
    - Post the `cancel_tasks` vector to the `queue_tasks` with high priority (at the front of the queue).
- **Output**: This function does not return any value.
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::receive\_multi\_results<!-- {{#callable:server_context::receive_multi_results}} -->
The `receive_multi_results` function retrieves results for a set of task IDs, handling errors and connection closures, and then processes the results using a provided handler.
- **Inputs**:
    - `id_tasks`: A set of integers representing the IDs of tasks for which results are to be received.
    - `result_handler`: A function that takes a vector of server_task_result_ptr and processes the results.
    - `error_handler`: A function that takes a JSON object and handles any errors that occur during result retrieval.
    - `is_connection_closed`: A function that returns a boolean indicating whether the connection is closed.
- **Control Flow**:
    - Initialize a vector `results` to store the results for each task ID.
    - Iterate over the size of `id_tasks` to receive results for each task.
    - Use `queue_results.recv_with_timeout` to attempt to receive a result for the current task ID with a timeout.
    - Check if the connection is closed using `is_connection_closed`; if true, cancel the tasks and return.
    - If no result is received (result is nullptr), decrement the loop counter to retry receiving the result.
    - If the result indicates an error, call `error_handler` with the error JSON, cancel the tasks, and return.
    - Assert that the result is of a valid type (either `server_task_result_cmpl_final`, `server_task_result_embd`, or `server_task_result_rerank`).
    - Retrieve the index of the result and assert it is within the bounds of the `results` vector.
    - Store the result in the `results` vector at the appropriate index.
    - After all results are received, call `result_handler` with the `results` vector.
- **Output**: The function does not return a value; it processes results through the `result_handler` and handles errors through the `error_handler`.
- **Functions called**:
    - [`server_context::cancel_tasks`](#server_contextcancel_tasks)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::receive\_cmpl\_results\_stream<!-- {{#callable:server_context::receive_cmpl_results_stream}} -->
The `receive_cmpl_results_stream` function processes and handles streaming results from server tasks, managing task completion and error handling.
- **Inputs**:
    - `id_tasks`: A constant reference to an unordered set of integers representing the IDs of tasks to receive results for.
    - `result_handler`: A constant reference to a function that takes a server_task_result_ptr and returns a boolean, used to handle each result.
    - `error_handler`: A constant reference to a function that takes a JSON object, used to handle errors.
    - `is_connection_closed`: A constant reference to a function that returns a boolean, used to check if the connection is closed.
- **Control Flow**:
    - Initialize a counter `n_finished` to track the number of finished tasks.
    - Enter an infinite loop to continuously receive results.
    - Call `recv_with_timeout` on `queue_results` to get a result for the given `id_tasks` with a timeout.
    - Check if the connection is closed using `is_connection_closed`; if true, cancel tasks and return.
    - If the result is null, continue to retry receiving results.
    - If the result indicates an error, call `error_handler`, cancel tasks, and return.
    - Assert that the result is of type `server_task_result_cmpl_partial` or `server_task_result_cmpl_final`.
    - Call `result_handler` with the result; if it returns false, cancel tasks and break the loop.
    - If the result indicates a stop, increment `n_finished`; if `n_finished` equals the size of `id_tasks`, break the loop.
- **Output**: The function does not return a value; it manages task completion and error handling through the provided handlers.
- **Functions called**:
    - [`server_context::cancel_tasks`](#server_contextcancel_tasks)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::process\_single\_task<!-- {{#callable:server_context::process_single_task}} -->
The `process_single_task` function processes a given server task based on its type, handling various server operations such as task completion, cancellation, metrics collection, slot management, and more.
- **Inputs**:
    - `task`: A `server_task` object passed by rvalue reference, representing the task to be processed, which includes its type, parameters, and other relevant data.
- **Control Flow**:
    - The function begins by switching on the `task.type` to determine the type of task to process.
    - For task types like `SERVER_TASK_TYPE_COMPLETION`, `SERVER_TASK_TYPE_INFILL`, `SERVER_TASK_TYPE_EMBEDDING`, and `SERVER_TASK_TYPE_RERANK`, it attempts to find an appropriate slot for processing the task.
    - If no slot is available or the requested slot is busy, the task is deferred for later processing.
    - If a slot is available and not busy, it attempts to launch the task on the slot using [`launch_slot_with_task`](#server_contextlaunch_slot_with_task).
    - For `SERVER_TASK_TYPE_CANCEL`, it releases the slot associated with the task's target ID.
    - For `SERVER_TASK_TYPE_NEXT_RESPONSE`, it does nothing as this is a placeholder for future responses.
    - For `SERVER_TASK_TYPE_METRICS`, it collects and sends metrics data about the server's slots and tasks.
    - For `SERVER_TASK_TYPE_SLOT_SAVE`, it saves the state of a specified slot to a file, handling errors if the slot is invalid or busy.
    - For `SERVER_TASK_TYPE_SLOT_RESTORE`, it restores a slot's state from a file, handling errors similarly to slot save.
    - For `SERVER_TASK_TYPE_SLOT_ERASE`, it clears the token cache of a specified slot.
    - For `SERVER_TASK_TYPE_SET_LORA`, it applies a new set of LoRA adapters to the server context.
- **Output**: The function does not return a value; it performs operations based on the task type, such as deferring tasks, sending results, or modifying server state.
- **Functions called**:
    - [`server_context::get_slot_by_id`](#server_contextget_slot_by_id)
    - [`server_context::get_available_slot`](#server_contextget_available_slot)
    - [`server_context::launch_slot_with_task`](#server_contextlaunch_slot_with_task)
    - [`server_context::ensure_no_mtmd`](#server_contextensure_no_mtmd)
    - [`server_context::send_error`](#server_contextsend_error)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::update\_slots<!-- {{#callable:server_context::update_slots}} -->
The `update_slots` function manages the processing of server slots, handling tasks such as checking for idle slots, posting tasks, applying context shifts, and processing tokens in batches.
- **Inputs**: None
- **Control Flow**:
    - Check if all slots are idle; if so, log a message and clear the KV cache if needed, then return.
    - Post a 'NEXT_RESPONSE' task to the task queue.
    - Iterate over each slot to apply context shifts if needed, releasing slots and sending errors if context shift is disabled or unsupported.
    - Clear the batch for the current iteration and determine if slots can be batched together.
    - Add sampled tokens from ongoing sequences to the batch, updating slot states and cache tokens.
    - Process pending prompts in slots, checking for conditions like prompt size and context shift, and updating slot states accordingly.
    - If no tokens are in the batch, log a warning and return.
    - Decode the batch of tokens, handling errors and adjusting batch size if necessary.
    - For each slot in the batch, handle different task types (e.g., embedding, rerank) and update slot states, processing tokens and speculative decoding if applicable.
    - Log completion of slot processing.
- **Output**: The function does not return a value; it operates on the server slots and task queue, updating their states and processing tasks as needed.
- **Functions called**:
    - [`server_context::kv_cache_clear`](#server_contextkv_cache_clear)
    - [`server_context::send_error`](#server_contextsend_error)
    - [`server_context::send_final_response`](#server_contextsend_final_response)
    - [`llama_pooling_type`](../../include/llama.h.driver.md#llama_pooling_type)
    - [`common_sampler_reset`](../../common/sampling.cpp.driver.md#common_sampler_reset)
    - [`common_sampler_accept`](../../common/sampling.cpp.driver.md#common_sampler_accept)
    - [`server_context::send_embedding`](#server_contextsend_embedding)
    - [`server_context::send_rerank`](#server_contextsend_rerank)
    - [`common_sampler_sample`](../../common/sampling.cpp.driver.md#common_sampler_sample)
    - [`server_context::populate_token_probs`](#server_contextpopulate_token_probs)
    - [`server_context::process_token`](#server_contextprocess_token)
    - [`common_speculative_gen_draft`](../../common/speculative.cpp.driver.md#common_speculative_gen_draft)
- **See also**: [`server_context`](#server_context)  (Data Structure)


---
#### server\_context::model\_meta<!-- {{#callable:server_context::model_meta}} -->
The `model_meta` function returns a JSON object containing metadata about the model and vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function constructs a JSON object using the `json` library.
    - It retrieves various metadata attributes from the `vocab` and `model` objects using specific functions like [`llama_vocab_type`](../../include/llama.h.driver.md#llama_vocab_type), `llama_vocab_n_tokens`, `llama_model_n_ctx_train`, `llama_model_n_embd`, `llama_model_n_params`, and `llama_model_size`.
    - These attributes are added to the JSON object with corresponding keys.
- **Output**: A JSON object containing metadata about the model and vocabulary, including `vocab_type`, `n_vocab`, `n_ctx_train`, `n_embd`, `n_params`, and `size`.
- **Functions called**:
    - [`llama_vocab_type`](../../include/llama.h.driver.md#llama_vocab_type)
- **See also**: [`server_context`](#server_context)  (Data Structure)



# Functions

---
### stop\_type\_to\_str<!-- {{#callable:stop_type_to_str}} -->
The function `stop_type_to_str` converts a `stop_type` enum value to its corresponding string representation.
- **Inputs**:
    - `type`: An input of type `stop_type`, which is an enumeration representing different stop conditions such as `STOP_TYPE_EOS`, `STOP_TYPE_WORD`, `STOP_TYPE_LIMIT`, or `STOP_TYPE_NONE`.
- **Control Flow**:
    - The function uses a switch statement to check the value of the `type` parameter.
    - If `type` is `STOP_TYPE_EOS`, the function returns the string "eos".
    - If `type` is `STOP_TYPE_WORD`, the function returns the string "word".
    - If `type` is `STOP_TYPE_LIMIT`, the function returns the string "limit".
    - For any other value (default case), the function returns the string "none".
- **Output**: A `std::string` representing the string equivalent of the `stop_type` enum value.


---
### format\_error\_response<!-- {{#callable:format_error_response}} -->
The `format_error_response` function generates a JSON object representing an error response based on a given error message and error type.
- **Inputs**:
    - `message`: A string containing the error message to be included in the response.
    - `type`: An enumeration value of type `error_type` that specifies the type of error.
- **Control Flow**:
    - Initialize `type_str` as an empty string and `code` as 500.
    - Use a switch statement to determine the `type_str` and `code` based on the `type` argument.
    - For each case in the switch statement, set `type_str` to a specific error type string and `code` to the corresponding HTTP status code.
    - Return a JSON object containing the `code`, `message`, and `type_str`.
- **Output**: A JSON object with keys `code`, `message`, and `type`, representing the error code, error message, and error type string respectively.


---
### log\_server\_request<!-- {{#callable:log_server_request}} -->
The `log_server_request` function logs HTTP request and response details, excluding specific paths, to the server's log system.
- **Inputs**:
    - `req`: An `httplib::Request` object representing the HTTP request received by the server, containing details such as the request method, path, remote address, and body.
    - `res`: An `httplib::Response` object representing the HTTP response sent by the server, containing details such as the response status and body.
- **Control Flow**:
    - Check if the request path is "/v1/health" or "/v1/completions"; if so, return immediately without logging.
    - Log the request method, path, remote address, and response status using the `SRV_INF` macro.
    - Log the request body and response body using the `SRV_DBG` macro.
- **Output**: This function does not return any value; it performs logging as a side effect.


---
### signal\_handler<!-- {{#callable:signal_handler}} -->
The `signal_handler` function handles termination signals to gracefully shut down the server, allowing for a forced termination if the signal is received twice.
- **Inputs**:
    - `signal`: An integer representing the signal number that triggered the handler.
- **Control Flow**:
    - Check if the server is already in the process of terminating using `is_terminating.test_and_set()`.
    - If the server is already terminating, print a message to `stderr` and exit immediately with status code 1.
    - If the server is not yet terminating, call the `shutdown_handler` function with the received signal.
- **Output**: The function does not return a value; it may terminate the program if the signal is received twice.


---
### main<!-- {{#callable:main}} -->
The `main` function initializes and starts an HTTP server for handling various API requests related to a machine learning model, including loading the model, setting up server configurations, and managing request handling and server shutdown.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize `common_params` structure to hold server parameters.
    - Parse command-line arguments using [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse) and exit if parsing fails.
    - Initialize common resources with `common_init`.
    - Create a `server_context` object to manage the server's state and tasks.
    - Initialize the backend and NUMA settings for the server.
    - Log system information including thread counts.
    - Set up an HTTP server using `httplib::Server`, with optional SSL support.
    - Configure server middlewares for API key validation and server state checks.
    - Define various route handlers for different API endpoints, such as health checks, model information, and task handling.
    - Set up the server's thread pool and bind it to the specified hostname and port.
    - Start the server in a separate thread and wait for it to be ready.
    - Load the machine learning model and initialize the server context.
    - Set up signal handlers for graceful shutdown on Unix and Windows systems.
    - Enter the main loop to process tasks until termination is signaled.
    - Clean up resources and join the server thread before exiting.
- **Output**: Returns an integer status code, where 0 indicates successful execution and 1 indicates an error occurred during initialization or execution.
- **Functions called**:
    - [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse)
    - [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_numa_init`](../../src/llama.cpp.driver.md#llama_numa_init)
    - [`safe_json_to_str`](utils.hpp.driver.md#safe_json_to_str)
    - [`json_value`](utils.hpp.driver.md#json_value)
    - [`format_error_response`](#format_error_response)
    - [`gen_chatcmplid`](utils.hpp.driver.md#gen_chatcmplid)
    - [`fnv_hash`](utils.hpp.driver.md#fnv_hash)
    - [`mtmd_tokenize`](../mtmd/mtmd.cpp.driver.md#mtmd_tokenize)
    - [`tokenize_input_prompts`](utils.hpp.driver.md#tokenize_input_prompts)
    - [`server_task::server_task`](#server_taskserver_task)
    - [`server_sent_event`](utils.hpp.driver.md#server_sent_event)
    - [`oaicompat_completion_params_parse`](utils.hpp.driver.md#oaicompat_completion_params_parse)
    - [`format_infill`](utils.hpp.driver.md#format_infill)
    - [`oaicompat_chat_params_parse`](utils.hpp.driver.md#oaicompat_chat_params_parse)
    - [`tokenize_mixed`](utils.hpp.driver.md#tokenize_mixed)
    - [`is_valid_utf8`](utils.hpp.driver.md#is_valid_utf8)
    - [`format_tokenizer_response`](utils.hpp.driver.md#format_tokenizer_response)
    - [`tokens_to_str`](utils.hpp.driver.md#tokens_to_str)
    - [`format_detokenized_response`](utils.hpp.driver.md#format_detokenized_response)
    - [`llama_pooling_type`](../../include/llama.h.driver.md#llama_pooling_type)
    - [`format_embeddings_response_oaicompat`](utils.hpp.driver.md#format_embeddings_response_oaicompat)
    - [`format_rerank`](utils.hpp.driver.md#format_rerank)
    - [`format_response_rerank`](utils.hpp.driver.md#format_response_rerank)
    - [`parse_lora_request`](utils.hpp.driver.md#parse_lora_request)
    - [`llama_backend_free`](../../src/llama.cpp.driver.md#llama_backend_free)
    - [`signal_handler`](#signal_handler)


