# Purpose
This C++ source code file is designed to perform text encoding and generation tasks using a language model, specifically the GritLM model. The file includes functions for encoding text into vector representations and generating text based on a given prompt. The [`encode`](#encode) function processes a list of sentences and an instruction to produce embeddings, which are normalized vector representations of the input text. This is achieved by tokenizing the input, running it through the model, and performing mean pooling on the token embeddings. The [`generate`](#generate) function takes a prompt and generates text by sampling tokens from the model until an end-of-sequence token is encountered. The code also includes a main function that demonstrates the use of these functionalities by encoding and comparing the cosine similarity of text embeddings and generating text based on a user prompt.

The file is structured to be an executable program, as indicated by the presence of the [`main`](#main) function. It integrates with the GritLM model through a series of function calls that initialize the model, process input data, and handle the output. The code relies on several external components, such as `llama_context`, `llama_model`, and `llama_sampler`, which are likely part of a larger library or framework for handling language models. The file does not define public APIs or external interfaces but rather serves as a standalone application that demonstrates the capabilities of the GritLM model in encoding and generating text. The use of conditional compilation with `#ifdef GRIT_DEBUG` suggests that the code can be compiled in a debug mode to provide additional output for development and testing purposes.
# Imports and Dependencies

---
- `arg.h`
- `common.h`
- `llama.h`
- `string`
- `vector`


# Functions

---
### encode<!-- {{#callable:encode}} -->
The `encode` function generates normalized embeddings for a list of sentences by processing them through a language model with a given instruction.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which contains the context for the language model operations.
    - `sentences`: A vector of strings, each representing a sentence to be encoded into embeddings.
    - `instruction`: A string containing an instruction that is prepended to each sentence before encoding.
- **Control Flow**:
    - Initialize an empty vector `result` to store the embeddings.
    - Retrieve the model and vocabulary from the context `ctx`.
    - Initialize a batch for processing the sentences.
    - Iterate over each sentence in the `sentences` vector.
    - For each sentence, clear the batch and concatenate the `instruction` with the sentence to form `input_string`.
    - Tokenize the `input_string` using the vocabulary, and determine the number of tokens `n_toks`.
    - Tokenize the `instruction` separately to determine the number of instruction tokens `n_inst`.
    - Add each token from the `input_string` to the batch, marking tokens beyond the instruction tokens for mean pooling.
    - Clear the key-value cache and set the context for embeddings and causal attention.
    - Run the model to decode the batch and obtain embeddings.
    - Retrieve the number of embedding dimensions `n_embd` from the model.
    - Initialize a vector `emb_unorm` to accumulate unnormalized embeddings.
    - Sum the embeddings for each token beyond the instruction tokens.
    - Perform mean pooling by dividing the accumulated embeddings by the number of sentence tokens.
    - Normalize the pooled embeddings and store them in `emb_norm`.
    - Add the normalized embeddings to the `result` vector.
    - Free the batch resources after processing all sentences.
    - Return the `result` vector containing the normalized embeddings for each sentence.
- **Output**: A vector of vectors of floats, where each inner vector represents the normalized embedding of a corresponding sentence from the input.


---
### generate<!-- {{#callable:generate}} -->
The `generate` function generates text based on a given prompt using a language model, optionally streaming the output as it is generated.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which holds the context for the language model.
    - `smpl`: A pointer to a `llama_sampler` object, which is used to sample tokens during text generation.
    - `prompt`: A constant reference to a `std::string` that contains the initial text prompt for the generation.
    - `stream`: A boolean flag indicating whether to stream the generated text to the standard output as it is produced.
- **Control Flow**:
    - Initialize an empty string `result` to store the generated text.
    - Retrieve the model and vocabulary from the context `ctx`.
    - Get the end-of-sequence token from the vocabulary.
    - Clear the key-value cache in the context and set embeddings and causal attention flags.
    - Initialize a batch for processing tokens.
    - Tokenize the input `prompt` into a vector of tokens.
    - Enter a loop to generate text until an end-of-sequence token is encountered.
    - Clear the batch and add the current input tokens to it, incrementing the token index.
    - Decode the batch to generate the next token.
    - Sample a token using the sampler `smpl`.
    - If the sampled token is the end-of-sequence token, break the loop.
    - Convert the token to a string piece and append it to `result`.
    - If streaming is enabled, print the piece to the standard output.
    - Add the token to the input vector for the next iteration.
    - After exiting the loop, if streaming is enabled, print a newline.
    - Free the batch resources.
    - Return the generated text stored in `result`.
- **Output**: A `std::string` containing the generated text based on the input prompt.


---
### gritlm\_instruction<!-- {{#callable:gritlm_instruction}} -->
The `gritlm_instruction` function formats a given instruction string by wrapping it with specific markers for user and embed sections, or returns a default embed marker if the instruction is empty.
- **Inputs**:
    - `instruction`: A constant reference to a `std::string` representing the instruction to be formatted.
- **Control Flow**:
    - Check if the `instruction` string is not empty.
    - If not empty, concatenate the user marker, the instruction, and the embed marker, and return the result.
    - If empty, return only the embed marker.
- **Output**: A `std::string` that contains the formatted instruction with user and embed markers, or just the embed marker if the instruction is empty.


---
### main<!-- {{#callable:main}} -->
The `main` function initializes and configures a language model to perform text embedding and generation tasks, then calculates cosine similarities between embeddings and generates text based on a prompt.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize `common_params` structure to hold parameters.
    - Parse command-line arguments using [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse); exit with code 1 if parsing fails.
    - Call `common_init` to perform common initialization tasks.
    - Convert parsed parameters to `llama_model_params` and `llama_context_params`.
    - Initialize the llama backend with [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init).
    - Load a llama model from a file using `llama_model_load_from_file`.
    - Create a generation context with `llama_init_from_model`.
    - Initialize a sampler chain with default parameters and add a greedy sampler.
    - Encode document and query texts into embeddings using the [`encode`](#encode) function.
    - Calculate cosine similarities between query and document embeddings.
    - Print the cosine similarity results for each query-document pair.
    - Generate text based on a prompt using the [`generate`](#generate) function.
    - Free resources allocated for the sampler, context, model, and backend.
    - Return 0 to indicate successful execution.
- **Output**: The function returns an integer, 0 for successful execution or 1 if parameter parsing fails.
- **Functions called**:
    - [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse)
    - [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_sampler_chain_default_params`](../../src/llama.cpp.driver.md#llama_sampler_chain_default_params)
    - [`encode`](#encode)
    - [`gritlm_instruction`](#gritlm_instruction)
    - [`generate`](#generate)
    - [`llama_backend_free`](../../src/llama.cpp.driver.md#llama_backend_free)


