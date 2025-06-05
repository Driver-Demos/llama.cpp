# Purpose
This Swift source code file defines a class `LlamaContext` that serves as an interface for interacting with a machine learning model, likely related to natural language processing or text generation, as suggested by the naming conventions and functionality. The file imports the `Foundation` and `llama` modules, indicating that it leverages Apple's Foundation framework for basic data types and structures, and a custom or third-party `llama` library for model operations. The `LlamaContext` class encapsulates the model, context, vocabulary, and sampling mechanisms, providing methods for initializing, managing, and interacting with these components.

The `LlamaContext` class includes several methods that facilitate the initialization and management of the model and its context, such as `create_context`, which loads a model from a file and initializes the context with specified parameters. The class also provides functionality for text processing and generation, including methods like `completion_init` and `completion_loop`, which handle tokenization, batch processing, and text completion tasks. Additionally, the class includes a benchmarking method, `bench`, which measures the performance of prompt processing and text generation, outputting results in a formatted string.

Overall, this file provides a focused and specialized functionality for managing and interacting with a machine learning model, specifically tailored for tasks involving text processing and generation. The `LlamaContext` class acts as a comprehensive interface, encapsulating the necessary components and operations required to utilize the model effectively, while also offering utility functions for performance evaluation and context management.
# Imports and Dependencies

---
- `Foundation`
- `llama`


# Data Structures

---
### LlamaContext
- **Type**: `actor`
- **Members**:
    - `model`: An opaque pointer to the model used in the LlamaContext.
    - `context`: An opaque pointer to the context used in the LlamaContext.
    - `vocab`: An opaque pointer to the vocabulary used in the LlamaContext.
    - `sampling`: A pointer to the llama_sampler used for sampling operations.
    - `batch`: A llama_batch structure used to manage tokens and their associated data.
    - `tokens_list`: A list of llama_token used to store tokens for processing.
    - `is_done`: A boolean flag indicating if the processing is complete.
    - `temporary_invalid_cchars`: An array of CChar used to temporarily store invalid characters.
    - `n_len`: An Int32 representing the maximum length of tokens.
    - `n_cur`: An Int32 representing the current number of tokens processed.
    - `n_decode`: An Int32 representing the number of tokens decoded.
- **Description**: The LlamaContext is an actor that encapsulates the state and operations for managing a language model context, including model and context pointers, vocabulary, sampling mechanisms, and batch processing for token management. It provides methods for initializing, processing, and benchmarking language model operations, handling tokenization, and managing the lifecycle of the context and associated resources.


# Functions

---
### llama\_batch\_clear
The `llama_batch_clear` function resets the number of tokens in a `llama_batch` to zero.
- **Inputs**:
    - `batch`: A reference to a `llama_batch` object whose `n_tokens` field will be reset to zero.
- **Control Flow**:
    - The function takes a mutable reference to a `llama_batch` object as input.
    - It sets the `n_tokens` field of the `llama_batch` to zero, effectively clearing the batch of any tokens.
- **Output**: The function does not return any value; it modifies the input `llama_batch` in place.


---
### llama\_batch\_add
The `llama_batch_add` function adds a token, its position, sequence IDs, and logits flag to a llama batch, incrementing the token count.
- **Inputs**:
    - `batch`: A reference to a `llama_batch` structure that will be modified to include the new token information.
    - `id`: A `llama_token` representing the token to be added to the batch.
    - `pos`: A `llama_pos` indicating the position of the token within the sequence.
    - `seq_ids`: An array of `llama_seq_id` representing the sequence IDs associated with the token.
    - `logits`: A Boolean flag indicating whether logits should be considered for the token (true if logits are to be used, false otherwise).
- **Control Flow**:
    - The function accesses the `batch` structure and assigns the `id`, `pos`, and `logits` values to the respective arrays at the index of the current token count (`n_tokens`).
    - It sets the number of sequence IDs for the current token to the count of `seq_ids`.
    - A loop iterates over the `seq_ids` array, assigning each sequence ID to the `seq_id` array in the `batch` at the current token index.
    - The `logits` value is converted to an integer (1 for true, 0 for false) and stored in the `logits` array at the current token index.
    - Finally, the function increments the `n_tokens` count in the `batch` by 1.
- **Output**: The function does not return a value; it modifies the `batch` in place to include the new token information.


---
### create\_context
The `create_context` function initializes a `LlamaContext` object by loading a model and context from a specified file path, setting up necessary parameters and resources for further operations.
- **Inputs**:
    - `path`: A `String` representing the file path to the model that needs to be loaded.
- **Control Flow**:
    - Initialize the llama backend using `llama_backend_init()`.
    - Set default model parameters using `llama_model_default_params()`.
    - Load the model from the specified file path using `llama_model_load_from_file(path, model_params)`.
    - If the model cannot be loaded, print an error message and throw `LlamaError.couldNotInitializeContext`.
    - Determine the number of threads to use based on the processor count, ensuring it is between 1 and 8.
    - Set default context parameters using `llama_context_default_params()` and configure them with the number of threads and context size.
    - Initialize the context from the model using `llama_init_from_model(model, ctx_params)`.
    - If the context cannot be initialized, print an error message and throw `LlamaError.couldNotInitializeContext`.
    - Return a new `LlamaContext` instance initialized with the loaded model and context.
- **Output**: Returns a `LlamaContext` object initialized with the specified model and context, or throws an error if initialization fails.


---
### model\_info
The `model_info` function retrieves and returns a string description of the model associated with the `LlamaContext`.
- **Inputs**: None
- **Control Flow**:
    - Allocate memory for a C-style string buffer with a capacity of 256 characters.
    - Initialize the allocated buffer with zeros to ensure it is clean.
    - Call the `llama_model_desc` function to fill the buffer with the model's description, capturing the number of characters written.
    - Create a Swift string by iterating over the buffer and converting each character to a Swift `Character`, appending it to the Swift string.
    - Deallocate the buffer to free up memory.
    - Return the constructed Swift string containing the model description.
- **Output**: A `String` containing the description of the model.


---
### get\_n\_tokens
The `get_n_tokens` function returns the current number of tokens in the `llama_batch`.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `n_tokens` property of the `batch` object.
    - It returns the value of `n_tokens`.
- **Output**: The function outputs an `Int32` representing the number of tokens currently in the `llama_batch`.


---
### completion\_init
The `completion_init` function initializes the tokenization and batch setup for a given text input to prepare for text completion using the Llama model.
- **Inputs**:
    - `text`: A `String` representing the text input that needs to be tokenized and prepared for completion.
- **Control Flow**:
    - Prints the text that is being attempted for completion.
    - Tokenizes the input text with an option to add a beginning-of-sequence token.
    - Initializes an empty list for temporarily invalid characters.
    - Calculates the required key-value cache size and checks if it exceeds the context size, printing an error if it does.
    - Iterates over the tokenized list, converting each token to a string and printing it.
    - Clears the current batch of tokens.
    - Adds each token from the tokenized list to the batch with its position and sequence ID.
    - Sets the logits flag for the last token in the batch to true.
    - Attempts to decode the batch using the Llama context, printing an error if decoding fails.
    - Updates the current number of tokens processed.
- **Output**: The function does not return a value; it prepares the internal state of the `LlamaContext` for text completion.


---
### completion\_loop
The `completion_loop` function generates a new token using a sampling method and updates the context and batch for further processing.
- **Inputs**: None
- **Control Flow**:
    - Initialize a variable `new_token_id` to store the new token ID.
    - Sample a new token ID using the `llama_sampler_sample` function with the current context and batch.
    - Check if the new token ID indicates the end of generation or if the current token count `n_cur` has reached the maximum length `n_len`.
    - If the end of generation is reached or the maximum length is hit, set `is_done` to true, convert the temporary invalid characters to a string, clear them, and return the string.
    - If not, convert the new token ID to characters and append them to `temporary_invalid_cchars`.
    - Attempt to convert `temporary_invalid_cchars` to a valid UTF-8 string; if successful, clear `temporary_invalid_cchars` and set `new_token_str` to the string.
    - Clear the batch and add the new token to it, updating the current token count `n_cur` and decode count `n_decode`.
    - Attempt to decode the batch with the current context, printing an error message if it fails.
    - Return the generated string `new_token_str`.
- **Output**: The function returns a `String` representing the newly generated token or sequence of tokens.


---
### bench
The `bench` function measures and reports the performance of prompt processing and text generation in a LlamaContext by executing these tasks multiple times and calculating average speeds and standard deviations.
- **Inputs**:
    - `pp`: The number of tokens to process during the prompt processing phase.
    - `tg`: The number of text generation iterations to perform.
    - `pl`: The number of tokens to process per text generation iteration.
    - `nr`: The number of repetitions for the benchmarking process, defaulting to 1.
- **Control Flow**:
    - Initialize average and standard deviation variables for prompt processing (pp) and text generation (tg).
    - Loop `nr` times to perform the benchmarking process.
    - Clear the batch and add `pp` tokens to it for prompt processing.
    - Measure the time taken to decode the batch for prompt processing and calculate the speed.
    - Clear the context's key-value cache and synchronize the context.
    - Clear the batch and add `pl` tokens for each of the `tg` text generation iterations.
    - Measure the time taken to decode the batch for text generation and calculate the speed.
    - Calculate average speeds and standard deviations for both prompt processing and text generation.
    - Retrieve model information, size, and parameters.
    - Format and return the results as a markdown table string.
- **Output**: A markdown table string summarizing the model's performance in terms of tokens per second for prompt processing and text generation, including average speeds and standard deviations.


---
### clear
The `clear` function resets the `tokens_list` and `temporary_invalid_cchars` arrays and clears the key-value cache in the LlamaContext.
- **Inputs**: None
- **Control Flow**:
    - The function begins by removing all elements from the `tokens_list` array.
    - It then clears all elements from the `temporary_invalid_cchars` array.
    - Finally, it calls `llama_kv_self_clear` to clear the key-value cache associated with the `context`.
- **Output**: The function does not return any value.


---
### tokenize
The `tokenize` function converts a given text into a list of llama tokens, optionally adding a beginning-of-sequence token.
- **Inputs**:
    - `text`: A `String` representing the text to be tokenized.
    - `add_bos`: A `Bool` indicating whether to add a beginning-of-sequence token to the token list.
- **Control Flow**:
    - Calculate the number of tokens needed, including space for a beginning-of-sequence token if `add_bos` is true.
    - Allocate memory for the tokens using `UnsafeMutablePointer<llama_token>`.
    - Call `llama_tokenize` to tokenize the text into llama tokens, storing the result in the allocated memory.
    - Iterate over the tokenized result to convert it into a Swift array of `llama_token`.
    - Deallocate the memory used for the tokens.
- **Output**: Returns an array of `llama_token` representing the tokenized version of the input text.


---
### token\_to\_piece
The `token_to_piece` function converts a given llama token into its corresponding character representation without a null-terminator.
- **Inputs**:
    - `token`: A `llama_token` representing the token to be converted into a character array.
- **Control Flow**:
    - Allocate memory for an 8-character buffer to store the result.
    - Initialize the buffer with zeros.
    - Call `llama_token_to_piece` to convert the token into a character array, storing the result in the buffer.
    - Check if the number of tokens returned (`nTokens`) is negative, indicating the buffer was too small.
    - If `nTokens` is negative, allocate a new buffer with the required size and call `llama_token_to_piece` again to fill it.
    - Convert the buffer into an array of `CChar` and return it.
- **Output**: An array of `CChar` representing the character sequence of the token, excluding a null-terminator.


