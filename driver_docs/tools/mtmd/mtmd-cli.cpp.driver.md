# Purpose
This C++ source code file is an experimental command-line interface (CLI) for multimodal support in the `llama.cpp` project. It is designed to facilitate the integration and testing of multimodal capabilities, such as processing text, images, and audio, within the context of a language model. The file includes several header files that provide essential functionalities, such as argument parsing, logging, common utilities, and specific multimodal and chat functionalities. The code is structured to handle different operating systems, including Unix-like systems and Windows, by setting up appropriate signal handling for user interruptions.

The main technical components of this file include the [`mtmd_cli_context`](#mtmd_cli_contextmtmd_cli_context) structure, which encapsulates the context for multimodal operations, and functions for generating responses and evaluating messages. The [`mtmd_cli_context`](#mtmd_cli_contextmtmd_cli_context) manages resources such as the language model, vision context, and chat templates, ensuring that the multimodal operations are executed correctly. The [`main`](#main) function initializes the necessary parameters, parses command-line arguments, and manages the interaction loop, allowing users to input commands and receive responses. The code supports both single-turn interactions and chat mode, providing flexibility in how users can engage with the multimodal capabilities. This file is not intended for production use but serves as a playground for contributors to experiment with and develop multimodal support in the `llama.cpp` project.
# Imports and Dependencies

---
- `arg.h`
- `log.h`
- `common.h`
- `sampling.h`
- `llama.h`
- `ggml.h`
- `console.h`
- `chat.h`
- `mtmd.h`
- `mtmd-helper.h`
- `vector`
- `limits.h`
- `cinttypes`
- `signal.h`
- `unistd.h`
- `windows.h`


# Global Variables

---
### g\_is\_generating
- **Type**: `bool`
- **Description**: The `g_is_generating` variable is a global, static, and volatile boolean flag used to indicate whether a generation process is currently active. It is initialized to `false`, meaning that by default, no generation is occurring.
- **Use**: This variable is used to control the flow of the program, particularly in handling interruptions during a generation process, such as when a SIGINT signal is received.


---
### g\_is\_interrupted
- **Type**: `bool`
- **Description**: The `g_is_interrupted` variable is a global, static, and volatile boolean flag used to indicate whether an interrupt signal (such as SIGINT) has been received by the program. It is initialized to `false` and is set to `true` when an interrupt signal is detected, allowing the program to handle the interruption gracefully.
- **Use**: This variable is used to manage the program's response to interrupt signals, ensuring that ongoing processes can be safely stopped or exited.


# Data Structures

---
### mtmd\_cli\_context<!-- {{#data_structure:mtmd_cli_context}} -->
- **Type**: `struct`
- **Members**:
    - `ctx_vision`: A pointer to the vision context used for multimodal processing.
    - `llama_init`: Holds the result of the common initialization process for the llama model.
    - `model`: A pointer to the llama model being used.
    - `lctx`: A pointer to the llama context for managing model state.
    - `vocab`: A pointer to the llama vocabulary used for tokenization.
    - `smpl`: A pointer to the common sampler used for generating tokens.
    - `batch`: Represents a batch of tokens for the next token generation.
    - `n_batch`: The number of batches to process.
    - `bitmaps`: Stores bitmaps for media processing.
    - `tmpls`: Pointer to chat templates used for generating chat responses.
    - `antiprompt_tokens`: Tokens used to identify the end of a prompt in legacy templates.
    - `n_threads`: The number of threads to use for processing.
    - `n_past`: Tracks the number of past tokens processed.
- **Description**: The `mtmd_cli_context` struct is designed to manage the context and state for a command-line interface that supports multimodal processing using the llama model. It encapsulates various components such as the vision context, model, vocabulary, sampler, and chat templates, and provides mechanisms for initializing these components based on input parameters. The struct also handles media loading and token generation, making it a central part of the CLI's functionality for processing and responding to user inputs.
- **Member Functions**:
    - [`mtmd_cli_context::mtmd_cli_context`](#mtmd_cli_contextmtmd_cli_context)
    - [`mtmd_cli_context::~mtmd_cli_context`](#mtmd_cli_contextmtmd_cli_context)
    - [`mtmd_cli_context::init_vision_context`](#mtmd_cli_contextinit_vision_context)
    - [`mtmd_cli_context::check_antiprompt`](#mtmd_cli_contextcheck_antiprompt)
    - [`mtmd_cli_context::load_media`](#mtmd_cli_contextload_media)

**Methods**

---
#### mtmd\_cli\_context::mtmd\_cli\_context<!-- {{#callable:mtmd_cli_context::mtmd_cli_context}} -->
The `mtmd_cli_context` constructor initializes a multimodal command-line interface context by setting up model, context, vocabulary, sampler, batch, and chat templates based on provided parameters, and handles legacy template support.
- **Inputs**:
    - `params`: A reference to a `common_params` object containing configuration parameters for initializing the context, including model paths, sampling settings, chat templates, and CPU parameters.
- **Control Flow**:
    - Initialize `llama_init` using `common_init_from_params` with `params`.
    - Retrieve and assign the model and context pointers from `llama_init`.
    - Get the vocabulary from the model using `llama_model_get_vocab`.
    - Initialize the sampler using [`common_sampler_init`](../../common/sampling.cpp.driver.md#common_sampler_init) with the model and sampling parameters.
    - Set the number of threads from `params.cpuparams.n_threads`.
    - Initialize a batch for next token generation using `llama_batch_init`.
    - Set the batch size from `params.n_batch`.
    - Check if the model or context is null and exit if true.
    - Check if the model has a chat template or if `params.chat_template` is empty, log errors, and exit if true.
    - Initialize chat templates using `common_chat_templates_init` with the model and chat template parameters.
    - Log an example of the chat template format using `common_chat_format_example`.
    - Call [`init_vision_context`](#mtmd_cli_contextinit_vision_context) to initialize the vision context with `params`.
    - Load antiprompt tokens for legacy templates based on the chat template specified in `params`.
- **Output**: The constructor does not return a value; it initializes the `mtmd_cli_context` object with the necessary components for multimodal interaction.
- **Functions called**:
    - [`common_sampler_init`](../../common/sampling.cpp.driver.md#common_sampler_init)
    - [`mtmd_cli_context::init_vision_context`](#mtmd_cli_contextinit_vision_context)
- **See also**: [`mtmd_cli_context`](#mtmd_cli_context)  (Data Structure)


---
#### mtmd\_cli\_context::\~mtmd\_cli\_context<!-- {{#callable:mtmd_cli_context::~mtmd_cli_context}} -->
The destructor `~mtmd_cli_context` is responsible for freeing resources associated with the `batch` and `smpl` members of the `mtmd_cli_context` structure.
- **Inputs**: None
- **Control Flow**:
    - The destructor is called when an instance of `mtmd_cli_context` is destroyed.
    - It calls `llama_batch_free(batch)` to release resources associated with the `batch`.
    - It calls `common_sampler_free(smpl)` to release resources associated with the `smpl`.
- **Output**: The function does not return any value as it is a destructor.
- **Functions called**:
    - [`common_sampler_free`](../../common/sampling.cpp.driver.md#common_sampler_free)
- **See also**: [`mtmd_cli_context`](#mtmd_cli_context)  (Data Structure)


---
#### mtmd\_cli\_context::init\_vision\_context<!-- {{#callable:mtmd_cli_context::init_vision_context}} -->
The `init_vision_context` function initializes the vision context by loading a vision model from a specified file path using given parameters.
- **Inputs**:
    - `params`: A reference to a `common_params` structure containing configuration parameters for initializing the vision context, including file paths, GPU usage, thread count, and verbosity level.
- **Control Flow**:
    - Retrieve the file path for the vision model from `params.mmproj.path` and store it in `clip_path`.
    - Initialize `mparams` with default context parameters using `mtmd_context_params_default()`.
    - Set `mparams.use_gpu` to the value of `params.mmproj_use_gpu`.
    - Enable timing prints by setting `mparams.print_timings` to `true`.
    - Set the number of threads in `mparams.n_threads` to `params.cpuparams.n_threads`.
    - Determine the verbosity level based on `params.verbosity` and set `mparams.verbosity` accordingly.
    - Initialize the vision context `ctx_vision` by calling `mtmd_init_from_file` with `clip_path`, `model`, and `mparams`.
    - Check if `ctx_vision` is successfully initialized; if not, log an error message and terminate the program with `exit(1)`.
- **Output**: The function does not return a value but initializes the `ctx_vision` member of the `mtmd_cli_context` structure with a vision model context.
- **Functions called**:
    - [`mtmd_context_params_default`](mtmd.cpp.driver.md#mtmd_context_params_default)
- **See also**: [`mtmd_cli_context`](#mtmd_cli_context)  (Data Structure)


---
#### mtmd\_cli\_context::check\_antiprompt<!-- {{#callable:mtmd_cli_context::check_antiprompt}} -->
The `check_antiprompt` function checks if the end of a sequence of generated tokens matches a predefined sequence of antiprompt tokens.
- **Inputs**:
    - `generated_tokens`: A reference to a `llama_tokens` object representing the sequence of tokens generated by the model.
- **Control Flow**:
    - Check if `antiprompt_tokens` is empty or if `generated_tokens` is shorter than `antiprompt_tokens`; if either is true, return false.
    - Use `std::equal` to compare the last `antiprompt_tokens.size()` tokens of `generated_tokens` with `antiprompt_tokens` and return the result.
- **Output**: A boolean value indicating whether the end of `generated_tokens` matches `antiprompt_tokens`.
- **See also**: [`mtmd_cli_context`](#mtmd_cli_context)  (Data Structure)


---
#### mtmd\_cli\_context::load\_media<!-- {{#callable:mtmd_cli_context::load_media}} -->
The `load_media` function attempts to load a bitmap image from a file and add it to a collection of bitmaps if successful.
- **Inputs**:
    - `fname`: A constant reference to a string representing the filename of the media file to be loaded.
- **Control Flow**:
    - Create a `mtmd::bitmap` object by initializing it with a helper function that loads a bitmap from the file specified by `fname` using the vision context `ctx_vision`.
    - Check if the `ptr` member of the `mtmd::bitmap` object is null, indicating a failure to load the bitmap.
    - If the bitmap loading fails (i.e., `ptr` is null), return `false`.
    - If the bitmap is successfully loaded, move the `mtmd::bitmap` object into the `bitmaps.entries` vector.
    - Return `true` to indicate successful loading of the media.
- **Output**: A boolean value indicating whether the media was successfully loaded (`true`) or not (`false`).
- **See also**: [`mtmd_cli_context`](#mtmd_cli_context)  (Data Structure)



# Functions

---
### show\_additional\_info<!-- {{#callable:show_additional_info}} -->
The `show_additional_info` function logs usage instructions for an experimental CLI tool designed for multimodal support.
- **Inputs**:
    - `argv`: A pointer to an array of character strings representing the command-line arguments passed to the program.
- **Control Flow**:
    - The function uses the `LOG` macro to print a formatted string to the console.
    - The string provides usage instructions for the CLI, detailing required and optional arguments.
    - It explains the purpose of each argument and provides guidance on how to disable GPU usage for the `mmproj` model.
- **Output**: The function does not return any value; it outputs information to the console via the `LOG` macro.


---
### sigint\_handler<!-- {{#callable:sigint_handler}} -->
The `sigint_handler` function handles the SIGINT signal to manage the state of a program, particularly during generation processes, and ensures proper cleanup and exit if necessary.
- **Inputs**:
    - `signo`: An integer representing the signal number, specifically SIGINT in this context.
- **Control Flow**:
    - Check if the received signal number is SIGINT.
    - If `g_is_generating` is true, set it to false to stop the generation process.
    - If `g_is_generating` is false, call `console::cleanup()` to perform cleanup operations.
    - Check if `g_is_interrupted` is true; if so, exit the program with status 1.
    - If `g_is_interrupted` is false, set it to true to indicate the program has been interrupted.
- **Output**: The function does not return a value; it performs actions based on the signal received and the current state of the program.


---
### generate\_response<!-- {{#callable:generate_response}} -->
The `generate_response` function generates a sequence of tokens based on a given context and prediction count, handling interruptions and end-of-generation conditions.
- **Inputs**:
    - `ctx`: A reference to an `mtmd_cli_context` object that contains the context for token generation, including model, vocabulary, sampler, and batch information.
    - `n_predict`: An integer specifying the number of tokens to predict and generate.
- **Control Flow**:
    - Initialize an empty `llama_tokens` vector to store generated tokens.
    - Iterate up to `n_predict` times to generate tokens.
    - Check if the current iteration exceeds `n_predict`, or if generation is interrupted, and break the loop if so.
    - Sample a token using [`common_sampler_sample`](../../common/sampling.cpp.driver.md#common_sampler_sample) and add it to `generated_tokens`.
    - Accept the sampled token using [`common_sampler_accept`](../../common/sampling.cpp.driver.md#common_sampler_accept).
    - Check if the token is an end-of-generation token or matches an antiprompt, and break the loop if so.
    - Log the generated token and flush the output to ensure it is displayed immediately.
    - Clear and add the token to the batch for evaluation, incrementing the past token count.
    - Decode the token using `llama_decode`, and return 1 if decoding fails.
    - Return 0 upon successful completion of token generation.
- **Output**: Returns an integer, 0 if the token generation completes successfully, or 1 if there is a decoding failure.
- **Functions called**:
    - [`common_sampler_sample`](../../common/sampling.cpp.driver.md#common_sampler_sample)
    - [`common_sampler_accept`](../../common/sampling.cpp.driver.md#common_sampler_accept)


---
### eval\_message<!-- {{#callable:eval_message}} -->
The `eval_message` function processes a chat message by formatting it, tokenizing the prompt, and evaluating it using a vision context, updating the past position in the process.
- **Inputs**:
    - `ctx`: A reference to an `mtmd_cli_context` object, which contains the context and state for the multimodal chat application.
    - `msg`: A reference to a `common_chat_msg` object, representing the chat message to be evaluated.
    - `add_bos`: A boolean flag indicating whether to add a beginning-of-sequence token to the input text.
- **Control Flow**:
    - Initialize `common_chat_templates_inputs` and set its properties using the input message.
    - Apply chat templates to format the message into a prompt string.
    - Log the formatted prompt for debugging purposes.
    - Check if the global interrupt flag `g_is_interrupted` is set; if so, return 0 immediately.
    - Initialize `mtmd_input_text` with the formatted prompt and set special parsing flags.
    - Initialize `mtmd::input_chunks` for tokenization output.
    - Tokenize the prompt using [`mtmd_tokenize`](mtmd.cpp.driver.md#mtmd_tokenize) and handle any errors by logging and returning 1.
    - Clear the bitmap entries in the context after tokenization.
    - Evaluate the tokenized chunks using [`mtmd_helper_eval_chunks`](mtmd-helper.cpp.driver.md#mtmd_helper_eval_chunks), updating the past position `ctx.n_past`.
    - Log an error and return 1 if evaluation fails.
    - Update the context's past position with the new value from evaluation.
    - Log a newline for separation and return 0 to indicate success.
- **Output**: Returns an integer status code: 0 for success, 1 for failure during tokenization or evaluation.
- **Functions called**:
    - [`mtmd_tokenize`](mtmd.cpp.driver.md#mtmd_tokenize)
    - [`mtmd_helper_eval_chunks`](mtmd-helper.cpp.driver.md#mtmd_helper_eval_chunks)


---
### main<!-- {{#callable:main}} -->
The `main` function initializes and runs a multimodal command-line interface (CLI) for interacting with a model, handling both single-turn and chat mode interactions, and managing user inputs and interruptions.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize the timing system with `ggml_time_init()`.
    - Set default parameters for sampling and parse command-line arguments using [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse).
    - Check if the required `--mmproj` argument is provided; if not, display additional information and return an error.
    - Initialize the multimodal CLI context with `mtmd_cli_context` using the parsed parameters.
    - Determine if the interaction is a single-turn or chat mode based on the presence of a prompt and image.
    - Set up signal handling for Ctrl+C interruptions on Unix and Windows systems.
    - If in single-turn mode, prepare the prompt and load media, then evaluate the message and generate a response.
    - If in chat mode, enter a loop to handle user inputs, supporting commands like `/image`, `/audio`, `/clear`, `/quit`, and `/exit`.
    - Within the chat loop, process user inputs, load media if specified, evaluate messages, and generate responses.
    - Handle interruptions and print performance context before exiting.
- **Output**: Returns 0 on successful completion, 1 on error, and 130 if interrupted by the user.
- **Functions called**:
    - [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse)
    - [`show_additional_info`](#show_additional_info)
    - [`sigint_handler`](#sigint_handler)
    - [`mtmd_default_marker`](mtmd.cpp.driver.md#mtmd_default_marker)
    - [`eval_message`](#eval_message)
    - [`generate_response`](#generate_response)
    - [`mtmd_support_vision`](mtmd.cpp.driver.md#mtmd_support_vision)
    - [`mtmd_support_audio`](mtmd.cpp.driver.md#mtmd_support_audio)


