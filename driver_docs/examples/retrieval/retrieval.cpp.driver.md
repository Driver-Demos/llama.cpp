# Purpose
This C++ source code file is an executable program designed to process text files and perform similarity-based retrieval using a language model. The program reads text files, divides them into chunks based on specified size and separator parameters, and then tokenizes these chunks. It uses a language model, likely based on the "llama" library, to generate embeddings for each chunk. The main functionality of the program is to allow users to input a query, tokenize it, and compute its embedding. The program then calculates cosine similarities between the query embedding and the embeddings of the text chunks, sorting and displaying the top-k most similar chunks based on these similarities.

The code is structured around several key components: file reading and chunking, tokenization, embedding generation, and similarity computation. It utilizes a variety of helper functions and structures, such as [`chunk_file`](#chunk_file) for dividing files into chunks, [`batch_add_seq`](#batch_add_seq) and [`batch_process`](#batch_process) for handling token sequences and processing them through the model, and a main loop for interactive query processing. The program is designed to be run as a standalone application, as indicated by the presence of a [`main`](#main) function, and it relies on external libraries and headers like "llama.h" for model operations and "common.h" for shared utilities. The code also includes logging for debugging and information purposes, and it handles various initialization and cleanup tasks for the language model and its context.
# Imports and Dependencies

---
- `arg.h`
- `common.h`
- `log.h`
- `llama.h`
- `algorithm`
- `fstream`
- `iostream`


# Data Structures

---
### chunk<!-- {{#data_structure:chunk}} -->
- **Type**: `struct`
- **Members**:
    - `filename`: Stores the name of the file associated with the chunk.
    - `filepos`: Indicates the original position of the chunk within the file.
    - `textdata`: Contains the original text data of the chunk.
    - `tokens`: Holds the tokenized version of the text data.
    - `embedding`: Stores the embedding vector for the chunk.
- **Description**: The `chunk` struct is designed to represent a segment of a file, capturing both its original and processed forms. It includes the filename and position within the file, the raw text data, its tokenized form, and an embedding vector for further processing or analysis. This struct is useful in applications that require text segmentation and analysis, such as natural language processing tasks.


# Functions

---
### print\_usage<!-- {{#callable:print_usage}} -->
The `print_usage` function logs an example usage message for the program, demonstrating how to run it with specific command-line arguments.
- **Inputs**:
    - `int`: An unused integer parameter, typically representing the argument count.
    - `char ** argv`: An array of C-style strings representing the command-line arguments passed to the program.
- **Control Flow**:
    - Logs a message indicating example usage of the program.
    - Logs a formatted string showing a sample command with various options, using the first element of `argv` to represent the program name.
    - Logs a newline character to separate the usage message from subsequent output.
- **Output**: The function does not return any value; it outputs the usage message to the log.


---
### chunk\_file<!-- {{#callable:chunk_file}} -->
The `chunk_file` function reads a file and divides its content into chunks based on a specified separator and minimum chunk size.
- **Inputs**:
    - `filename`: A string representing the name of the file to be read and chunked.
    - `chunk_size`: An integer specifying the minimum size of each chunk in terms of characters.
    - `chunk_separator`: A string used as a separator to determine where chunks should be split.
- **Control Flow**:
    - Initialize an empty vector `chunks` to store the resulting chunks and open the file specified by `filename` using an ifstream object `f`.
    - Check if the file is successfully opened; if not, log an error and return the empty `chunks` vector.
    - Initialize a [`chunk`](#chunk) object `current_chunk`, a character buffer, a file position tracker `filepos`, and a string `current` to accumulate file content.
    - Read the file in 1024-byte blocks into the buffer, appending the read content to `current`.
    - Within a loop, search for the `chunk_separator` in `current` and extract substrings up to and including the separator to `current_chunk.textdata`.
    - If `current_chunk.textdata` exceeds `chunk_size`, save the chunk with its file position and filename, update `filepos`, and reset `current_chunk`.
    - Continue processing until the file is fully read, then handle any remaining data in `current_chunk` by appending it to the last chunk or creating a new one if `chunks` is empty.
    - Close the file and return the `chunks` vector.
- **Output**: A vector of [`chunk`](#chunk) objects, each containing a portion of the file's text data, its original file position, and the filename.
- **Functions called**:
    - [`chunk`](#chunk)


---
### batch\_add\_seq<!-- {{#callable:batch_add_seq}} -->
The `batch_add_seq` function adds a sequence of tokens to a batch with a specified sequence ID.
- **Inputs**:
    - `batch`: A reference to a `llama_batch` object where the tokens will be added.
    - `tokens`: A constant reference to a vector of `int32_t` representing the tokens to be added to the batch.
    - `seq_id`: A `llama_seq_id` representing the sequence ID to associate with the tokens in the batch.
- **Control Flow**:
    - Determine the number of tokens in the input vector `tokens`.
    - Iterate over each token in the `tokens` vector.
    - For each token, call `common_batch_add` to add the token to the batch with the current index, the specified sequence ID, and a flag set to true.
- **Output**: The function does not return a value; it modifies the `batch` object by adding the tokens with the specified sequence ID.


---
### batch\_process<!-- {{#callable:batch_process}} -->
The `batch_process` function processes a batch of tokens in a llama context to generate normalized embeddings for each token sequence.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which holds the context for processing the batch.
    - `batch`: A reference to a `llama_batch` object, which contains the tokens and associated data to be processed.
    - `output`: A pointer to a float array where the resulting embeddings will be stored.
    - `n_seq`: An integer representing the number of sequences in the batch.
    - `n_embd`: An integer representing the number of embedding dimensions.
- **Control Flow**:
    - Clear the previous key-value cache in the context using `llama_kv_self_clear` as it is irrelevant for embeddings.
    - Log the number of tokens and sequences, then attempt to decode the batch using `llama_decode`; log an error if decoding fails.
    - Iterate over each token in the batch, skipping tokens without logits.
    - Attempt to retrieve sequence embeddings using `llama_get_embeddings_seq`; if unsuccessful, try `llama_get_embeddings_ith` and log an error if both fail.
    - Normalize the retrieved embeddings using `common_embd_normalize` and store them in the output array.
- **Output**: The function does not return a value, but it populates the `output` array with normalized embeddings for each token sequence in the batch.


---
### main<!-- {{#callable:main}} -->
The `main` function initializes and processes text data into chunks, tokenizes them, computes embeddings, and continuously accepts user queries to find and display the most similar text chunks based on cosine similarity.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of character pointers listing all the arguments.
- **Control Flow**:
    - Initialize `common_params` and parse command-line arguments using [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse); if parsing fails, print usage and return 1.
    - Initialize common settings with `common_init` and set parameters for BERT models.
    - Check if `chunk_size` is positive and `context_files` is specified; if not, log an error and return 1.
    - Log the context files being processed and chunk each file using [`chunk_file`](#chunk_file), storing the results in `chunks`.
    - Initialize the llama backend and NUMA settings, then load the model and context using `common_init_from_params`.
    - Check if the model is loaded and if the pooling type is supported; log errors and return 1 if not.
    - Log system information and assert that the batch size is greater than or equal to the context size.
    - Tokenize each chunk's text data, ensuring it does not exceed the batch size, and append an end-of-sequence token if necessary.
    - If verbose prompt is enabled, log tokenization details for each chunk.
    - Initialize a batch and allocate memory for embeddings, then process chunks in batches, encoding them and storing embeddings.
    - Clear tokens from chunks after processing and initialize a query batch for user input.
    - Enter a loop to accept user queries, tokenize them, process the query batch, and compute cosine similarities with chunk embeddings.
    - Sort and log the top k similar chunks based on cosine similarity for each query.
    - Print performance context information and clean up resources before exiting.
- **Output**: The function does not return a value; it continuously processes user queries and outputs the top k similar text chunks based on cosine similarity.
- **Functions called**:
    - [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse)
    - [`chunk_file`](#chunk_file)
    - [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_numa_init`](../../src/llama.cpp.driver.md#llama_numa_init)
    - [`llama_pooling_type`](../../include/llama.h.driver.md#llama_pooling_type)
    - [`batch_process`](#batch_process)
    - [`batch_add_seq`](#batch_add_seq)
    - [`llama_backend_free`](../../src/llama.cpp.driver.md#llama_backend_free)


