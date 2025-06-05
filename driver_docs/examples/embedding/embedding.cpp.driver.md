# Purpose
This C++ source code file is an executable program designed to process and compute embeddings from input text prompts using a model referred to as "llama." The code is structured to handle command-line arguments, initialize necessary parameters and contexts, and perform tokenization and embedding computations. The main technical components include functions for splitting input text into lines, adding sequences to a batch, and decoding batches to obtain embeddings. The program utilizes a model to tokenize input prompts, processes them in batches, and computes embeddings, which are then outputted in various formats, including plain text and JSON. The code also includes logging for debugging and performance monitoring.

The file imports several headers, indicating its reliance on external libraries or modules for argument parsing, logging, and model operations. It defines a main function, which is the entry point of the program, and several static helper functions to manage batch processing and embedding computations. The code is designed to be executed as a standalone application, rather than being a library or header file intended for import into other projects. It provides a public interface through command-line arguments, allowing users to specify input prompts and output preferences, and it includes error handling and logging to ensure robust operation and user feedback.
# Imports and Dependencies

---
- `arg.h`
- `common.h`
- `log.h`
- `llama.h`
- `ctime`
- `algorithm`


# Functions

---
### split\_lines<!-- {{#callable:split_lines}} -->
The `split_lines` function splits a given string into a vector of strings based on a specified separator.
- **Inputs**:
    - `s`: The input string to be split into lines.
    - `separator`: The string used as a delimiter to split the input string, defaulting to a newline character ("\n").
- **Control Flow**:
    - Initialize an empty vector `lines` to store the resulting substrings.
    - Set `start` to 0 and find the first occurrence of `separator` in `s`, storing the position in `end`.
    - Enter a loop that continues as long as `end` is not `std::string::npos` (indicating no more separators are found).
    - Within the loop, extract the substring from `start` to `end` and add it to `lines`.
    - Update `start` to the position after the current `separator` and search for the next occurrence of `separator` starting from `start`.
    - After the loop, add the remaining part of the string (from `start` to the end of `s`) to `lines`.
- **Output**: A vector of strings, each representing a segment of the input string `s` split by the `separator`.


---
### batch\_add\_seq<!-- {{#callable:batch_add_seq}} -->
The `batch_add_seq` function adds a sequence of tokens to a batch with a specified sequence ID.
- **Inputs**:
    - `batch`: A reference to a `llama_batch` object where tokens will be added.
    - `tokens`: A constant reference to a vector of `int32_t` representing the tokens to be added to the batch.
    - `seq_id`: A `llama_seq_id` representing the sequence ID associated with the tokens being added.
- **Control Flow**:
    - Determine the number of tokens in the `tokens` vector.
    - Iterate over each token in the `tokens` vector.
    - For each token, call `common_batch_add` with the current token, its index, the sequence ID wrapped in a list, and a boolean value `true`.
- **Output**: The function does not return any value; it modifies the `batch` object by adding the specified tokens with the given sequence ID.


---
### batch\_decode<!-- {{#callable:batch_decode}} -->
The `batch_decode` function processes a batch of tokens in a given context, retrieves their embeddings, and normalizes them into an output array.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, which holds the context for the decoding process.
    - `batch`: A reference to a `llama_batch` structure, which contains the batch of tokens to be processed.
    - `output`: A pointer to a float array where the normalized embeddings will be stored.
    - `n_seq`: An integer representing the number of sequences in the batch.
    - `n_embd`: An integer representing the number of embedding dimensions.
    - `embd_norm`: An integer indicating whether the embeddings should be normalized.
- **Control Flow**:
    - Determine the pooling type using `llama_pooling_type(ctx)`.
    - Clear previous key-value cache values using `llama_kv_self_clear(ctx)`.
    - Log the number of tokens and sequences, then attempt to decode the batch with `llama_decode(ctx, batch)`.
    - Iterate over each token in the batch, skipping those without logits.
    - For each token, retrieve its embedding based on the pooling type (either token or sequence embeddings).
    - Normalize the retrieved embeddings using `common_embd_normalize` and store them in the output array.
- **Output**: The function does not return a value but populates the `output` array with normalized embeddings.
- **Functions called**:
    - [`llama_pooling_type`](../../include/llama.h.driver.md#llama_pooling_type)


---
### main<!-- {{#callable:main}} -->
The `main` function initializes and processes a machine learning model to compute embeddings from input prompts, handling various configurations and output formats.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize `common_params` structure to hold configuration parameters.
    - Parse command-line arguments into `params` and check for successful parsing; return 1 on failure.
    - Initialize common resources and set `params.embedding` to true.
    - Adjust batch size if it is less than context size and set `n_ubatch` to `n_batch`.
    - Initialize backend and NUMA settings based on `params`.
    - Load the model and context using `common_init_from_params`; return 1 if model loading fails.
    - Retrieve vocabulary and context information from the model.
    - Check for unsupported encoder-decoder model configurations and warn if context size exceeds training context size.
    - Log system information.
    - Split the input prompt into lines and tokenize each line, checking for token count limits and SEP token presence.
    - Log tokenization details if verbose mode is enabled.
    - Initialize a batch for processing prompts and calculate the number of embeddings needed.
    - Allocate memory for embeddings and process prompts in batches, encoding them and storing results.
    - Handle final batch processing and output embeddings in various formats based on `params.embd_out`.
    - Log performance metrics and clean up resources before exiting with a return code of 0.
- **Output**: The function returns an integer status code, 0 for successful execution and 1 for errors encountered during processing.
- **Functions called**:
    - [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse)
    - [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_numa_init`](../../src/llama.cpp.driver.md#llama_numa_init)
    - [`llama_pooling_type`](../../include/llama.h.driver.md#llama_pooling_type)
    - [`split_lines`](#split_lines)
    - [`batch_decode`](#batch_decode)
    - [`batch_add_seq`](#batch_add_seq)
    - [`llama_backend_free`](../../src/llama.cpp.driver.md#llama_backend_free)


