# Purpose
This C++ source code file is designed to perform operations related to the collection and processing of indirect matrix multiplications, specifically in the context of machine learning models, likely involving neural networks. The file includes functionality for collecting, saving, and loading matrix data, as well as computing softmax and log-softmax operations, which are common in neural network computations. The primary class, [`IMatrixCollector`](#IMatrixCollectorIMatrixCollector), is responsible for managing the collection of matrix data, ensuring thread safety with mutexes, and handling the persistence of this data through file operations. The code also includes a [`main`](#main) function, indicating that this is an executable program, which processes command-line arguments to configure its operations, such as specifying input files, output files, and various processing parameters.

The file imports several headers, both custom (e.g., "arg.h", "common.h", "log.h", "llama.h") and standard C++ libraries (e.g., `<chrono>`, `<thread>`, `<mutex>`), indicating a reliance on both custom and standard functionalities. The code is structured to handle multi-threaded operations, as seen in the use of `std::thread` and `std::mutex`, which are crucial for parallel processing of data. The program is designed to be flexible, allowing for the combination of precomputed matrices and the computation of new matrices based on input data. It also includes detailed logging and error handling, which are essential for debugging and ensuring the reliability of the software. The presence of functions like [`compute_imatrix`](#compute_imatrix) and [`process_logits`](#process_logits) suggests that the code is tailored for specific machine learning tasks, such as evaluating model performance or processing model outputs.
# Imports and Dependencies

---
- `arg.h`
- `common.h`
- `log.h`
- `llama.h`
- `chrono`
- `cmath`
- `cstdio`
- `cstring`
- `ctime`
- `thread`
- `mutex`
- `vector`
- `fstream`
- `unordered_map`
- `algorithm`


# Global Variables

---
### g\_collector
- **Type**: `IMatrixCollector`
- **Description**: `g_collector` is a global instance of the `IMatrixCollector` class, which is responsible for collecting and managing data related to indirect matrix multiplications in a machine learning context. It maintains statistics about the collected data, such as values and counts, and provides methods to save and load this data to and from files.
- **Use**: `g_collector` is used to collect, save, and load matrix data during the execution of the program, particularly in the context of processing machine learning models.


# Data Structures

---
### Stats<!-- {{#data_structure:Stats}} -->
- **Type**: `struct`
- **Members**:
    - `values`: A vector of floats to store statistical values.
    - `counts`: A vector of integers to store counts corresponding to the values.
    - `ncall`: An integer to track the number of times the statistics have been updated.
- **Description**: The `Stats` struct is designed to hold statistical data, specifically a collection of float values and their corresponding counts, along with a counter to track the number of updates or calls made to the statistics. This structure is useful for accumulating and managing statistical data over multiple operations or iterations.


---
### IMatrixCollector<!-- {{#data_structure:IMatrixCollector}} -->
- **Type**: `class`
- **Members**:
    - `m_stats`: An unordered map that associates string keys with Stats objects, used to store statistical data.
    - `m_params`: Holds common parameters for the matrix collection process.
    - `m_mutex`: A mutex used to ensure thread safety when accessing shared resources.
    - `m_last_call`: An integer tracking the last call number for matrix collection.
    - `m_src1_data`: A vector of characters used to store data from the source tensor.
    - `m_ids`: A vector of characters storing expert IDs from ggml_mul_mat_id operations.
- **Description**: The `IMatrixCollector` class is designed to manage the collection, storage, and retrieval of matrix data during tensor operations. It maintains statistical data about the matrices being processed, using a map to associate tensor names with their corresponding statistics. The class ensures thread safety with a mutex and provides methods to save and load matrix data to and from files. It also tracks the last call number to manage the frequency of saving operations. The class is integral to handling matrix operations in a multi-threaded environment, particularly in scenarios involving indirect matrix multiplications and expert ID management.
- **Member Functions**:
    - [`IMatrixCollector::IMatrixCollector`](#IMatrixCollectorIMatrixCollector)
    - [`IMatrixCollector::set_params`](#IMatrixCollectorset_params)
    - [`IMatrixCollector::collect_imatrix`](#IMatrixCollectorcollect_imatrix)
    - [`IMatrixCollector::save_imatrix`](#IMatrixCollectorsave_imatrix)
    - [`IMatrixCollector::load_imatrix`](#IMatrixCollectorload_imatrix)

**Methods**

---
#### IMatrixCollector::IMatrixCollector<!-- {{#callable:IMatrixCollector::IMatrixCollector}} -->
The `IMatrixCollector` class is responsible for managing and processing indirect matrix multiplication data, including setting parameters, collecting data, saving, and loading matrices.
- **Inputs**:
    - `params`: An instance of `common_params` that contains configuration settings for the matrix collection process.
- **Control Flow**:
    - The constructor `IMatrixCollector()` initializes an instance with default settings.
    - The `set_params` method assigns the provided `common_params` to the member variable `m_params`.
    - The `collect_imatrix` method processes a tensor `t` to determine if data should be collected based on the operation type and other conditions, and if so, collects and processes the data.
    - The `save_imatrix` method writes collected matrix data to a file, ensuring only complete data entries are stored.
    - The `load_imatrix` method reads matrix data from a file, reconstructing the internal state of the collector.
- **Output**: The `set_params` method does not return a value; it modifies the internal state of the `IMatrixCollector` instance by setting its parameters.
- **See also**: [`IMatrixCollector`](#IMatrixCollector)  (Data Structure)


---
#### IMatrixCollector::set\_params<!-- {{#callable:IMatrixCollector::set_params}} -->
The `set_params` function assigns a new set of parameters to the `IMatrixCollector` object by moving the provided `common_params` object into the member variable `m_params`.
- **Inputs**:
    - `params`: An object of type `common_params` that contains the parameters to be set for the `IMatrixCollector` instance.
- **Control Flow**:
    - The function takes a `common_params` object as an argument.
    - It uses `std::move` to transfer ownership of the `params` object to the member variable `m_params`.
- **Output**: The function does not return any value.
- **See also**: [`IMatrixCollector`](#IMatrixCollector)  (Data Structure)


---
#### IMatrixCollector::collect\_imatrix<!-- {{#callable:IMatrixCollector::collect_imatrix}} -->
The `collect_imatrix` function determines whether to collect data from a given tensor and performs the collection if required, updating internal statistics and potentially saving the collected data.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure representing the tensor from which data may be collected.
    - `ask`: A boolean flag indicating whether the function is being queried for interest in the tensor data (`true`) or if it should perform the actual data collection (`false`).
    - `user_data`: A pointer to user-defined data, which is unused in this function.
- **Control Flow**:
    - Check if `ask` is true to determine if the function is being queried for interest in the tensor data.
    - If `ask` is true, check the operation type of the tensor and other conditions to decide if data collection is needed, returning `true` or `false` accordingly.
    - If `ask` is false, lock a mutex to ensure thread safety during data collection.
    - Determine if the tensor data is on the host or needs to be copied from GPU memory.
    - If the operation is `GGML_OP_MUL_MAT_ID`, handle the collection of data for indirect matrix multiplications, updating statistics and checking for consistency.
    - If the operation is not `GGML_OP_MUL_MAT_ID`, handle the collection of data for direct matrix multiplications, updating statistics and checking for consistency.
    - Check if the number of calls exceeds the last recorded call and save the collected data if certain conditions are met.
- **Output**: Returns a boolean value `true` indicating successful interest or collection of data.
- **Functions called**:
    - [`filter_tensor_name`](#filter_tensor_name)
    - [`ggml_backend_buffer_is_host`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_backend_tensor_get`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get)
    - [`ggml_element_size`](../../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_op_name`](../../ggml/src/ggml.c.driver.md#ggml_op_name)
    - [`IMatrixCollector::save_imatrix`](#IMatrixCollectorsave_imatrix)
- **See also**: [`IMatrixCollector`](#IMatrixCollector)  (Data Structure)


---
#### IMatrixCollector::save\_imatrix<!-- {{#callable:IMatrixCollector::save_imatrix}} -->
The `save_imatrix` function saves the collected imatrix data to a file, ensuring only complete data entries are stored.
- **Inputs**:
    - `ncall`: An integer representing the number of calls; if greater than 0, it is used to append to the output filename.
- **Control Flow**:
    - Initialize the output filename from `m_params.out_file` and append '.at_' followed by `ncall` if `ncall` is greater than 0.
    - Iterate over `m_stats` to filter out entries with incomplete data, logging warnings for entries with no or partial data.
    - Count and store only entries with complete data in `to_store`.
    - Log a warning if not all entries are stored due to incomplete data.
    - Open a binary output file stream with the constructed filename.
    - Write the number of complete entries to the file.
    - For each entry in `to_store`, write the entry name, number of calls, and processed values to the file.
    - Write the `m_last_call` value to the file.
    - Write the input filename from `m_params.prompt_file` to the file.
    - Log debug information about the stored data.
- **Output**: The function outputs a binary file containing the number of complete entries, each entry's name, number of calls, processed values, the last call number, and the input filename.
- **See also**: [`IMatrixCollector`](#IMatrixCollector)  (Data Structure)


---
#### IMatrixCollector::load\_imatrix<!-- {{#callable:IMatrixCollector::load_imatrix}} -->
The `load_imatrix` function reads and loads statistical data from a binary file into the `m_stats` map of the `IMatrixCollector` class.
- **Inputs**:
    - `fname`: A constant character pointer representing the name of the file to be loaded.
- **Control Flow**:
    - Open the file specified by `fname` in binary mode using an `ifstream` object.
    - Check if the file was successfully opened; if not, log an error and return `false`.
    - Read the number of entries (`n_entries`) from the file and check for read errors or invalid values; if any, log an error and return `false`.
    - Iterate over each entry in the file, reading the length of the name, the name itself, the number of calls (`ncall`), and the number of values (`nval`).
    - For each entry, check for read errors or invalid values; if any, log an error, clear `m_stats`, and return `false`.
    - If the entry's `values` vector is empty, resize it and the `counts` vector to accommodate `nval` elements initialized to zero.
    - Read the values into a temporary vector and check for read errors; if any, log an error, clear `m_stats`, and return `false`.
    - Update the `values` and `counts` vectors of the entry in `m_stats` by adding the read values and incrementing the counts by `ncall`.
    - Increment the `ncall` of the entry by the read `ncall`.
    - Return `true` after successfully loading all entries.
- **Output**: A boolean value indicating whether the file was successfully loaded (`true`) or not (`false`).
- **See also**: [`IMatrixCollector`](#IMatrixCollector)  (Data Structure)



---
### results\_log\_softmax<!-- {{#data_structure:results_log_softmax}} -->
- **Type**: `struct`
- **Members**:
    - `log_softmax`: Stores the logarithm of the softmax value as a double.
    - `logit`: Represents the logit value as a float.
    - `prob`: Holds the probability value as a float.
- **Description**: The `results_log_softmax` struct is designed to encapsulate the results of a log softmax computation, including the log softmax value, the original logit, and the resulting probability. This struct is useful for storing and passing around these related values in a cohesive manner, particularly in contexts involving neural network output processing or probabilistic computations.


# Functions

---
### print\_usage<!-- {{#callable:print_usage}} -->
The `print_usage` function logs an example usage message for a command-line application, detailing the expected command-line arguments and options.
- **Inputs**:
    - `int`: An unused integer parameter, typically representing the argument count (argc) in a command-line application.
    - `char ** argv`: An array of C-style strings representing the command-line arguments passed to the program, where `argv[0]` is the program name.
- **Control Flow**:
    - The function begins by logging a message indicating that an example usage will be shown.
    - It then logs a formatted string that includes the program name (from `argv[0]`) and a list of command-line options and arguments that the program accepts.
    - The function ends by logging a newline character to separate the usage message from subsequent log entries.
- **Output**: The function does not return any value; it outputs the usage message to the log.


---
### filter\_tensor\_name<!-- {{#callable:filter_tensor_name}} -->
The `filter_tensor_name` function extracts and returns the core part of a tensor name by removing any prefix and suffix delimited by '#' characters.
- **Inputs**:
    - `name`: A C-style string representing the full name of a tensor, which may include prefixes and suffixes separated by '#' characters.
- **Control Flow**:
    - Initialize an empty string `wname` to store the filtered name.
    - Use `strchr` to find the first occurrence of '#' in the input `name`.
    - If a '#' is found, set `p` to point to the character immediately after the first '#'.
    - Search for another '#' starting from `p` using `strchr`.
    - If a second '#' is found, extract the substring between the two '#' characters and assign it to `wname`.
    - If no second '#' is found, assign the substring starting from `p` to the end of the string to `wname`.
    - If no '#' is found in the original `name`, assign the entire `name` to `wname`.
    - Return the filtered name stored in `wname`.
- **Output**: A `std::string` containing the core part of the tensor name, with any prefix and suffix removed.


---
### ik\_collect\_imatrix<!-- {{#callable:ik_collect_imatrix}} -->
The function `ik_collect_imatrix` serves as a wrapper to call the `collect_imatrix` method of the global `IMatrixCollector` instance, `g_collector`, with the provided tensor, ask flag, and user data.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure, representing the tensor to be processed.
    - `ask`: A boolean flag indicating whether the function is being called to check interest in the tensor data (true) or to perform the actual data collection (false).
    - `user_data`: A pointer to user-defined data that can be passed to the function, though it is unused in this implementation.
- **Control Flow**:
    - The function directly calls the `collect_imatrix` method of the `g_collector` object, passing the arguments `t`, `ask`, and `user_data`.
- **Output**: Returns a boolean value indicating the success or failure of the `collect_imatrix` method call.


---
### softmax<!-- {{#callable:softmax}} -->
The `softmax` function computes the softmax probabilities of a given vector of logits, ensuring numerical stability.
- **Inputs**:
    - `logits`: A constant reference to a vector of floats representing the input logits for which the softmax probabilities are to be calculated.
- **Control Flow**:
    - Initialize a vector `probs` of the same size as `logits` to store the softmax probabilities.
    - Find the maximum value in the `logits` vector to use for numerical stability.
    - Iterate over each element in `logits`, subtract the maximum logit from each logit, compute the exponential of the result, and accumulate the sum of these exponentials in `sum_exp`.
    - Store the exponential values in the `probs` vector.
    - Normalize each element in `probs` by dividing it by `sum_exp` to get the final softmax probabilities.
    - Return the `probs` vector containing the softmax probabilities.
- **Output**: A vector of floats representing the softmax probabilities corresponding to the input logits.


---
### log\_softmax<!-- {{#callable:log_softmax}} -->
The `log_softmax` function computes the log softmax value, the original logit, and the probability for a specified token index from a set of logits.
- **Inputs**:
    - `n_vocab`: The number of vocabulary entries, representing the size of the logits array.
    - `logits`: A pointer to an array of float values representing the logits for each vocabulary entry.
    - `tok`: The index of the token for which the log softmax, logit, and probability are to be computed.
- **Control Flow**:
    - Initialize `max_logit` with the first logit value.
    - Iterate over the logits to find the maximum logit value for numerical stability.
    - Initialize `sum_exp` to zero and iterate over the logits to compute the sum of exponentials of the logits adjusted by the maximum logit.
    - Return a `results_log_softmax` struct containing the log softmax value, the original logit, and the probability for the specified token index.
- **Output**: A `results_log_softmax` struct containing the log softmax value, the original logit, and the probability for the specified token index.


---
### process\_logits<!-- {{#callable:process_logits}} -->
The `process_logits` function calculates the negative log-likelihood (NLL) and its square for a sequence of tokens using multi-threading, while also storing the logit and probability history for each token.
- **Inputs**:
    - `n_vocab`: The number of vocabulary entries, representing the size of the vocabulary.
    - `logits`: A pointer to an array of floats representing the logits for each token.
    - `tokens`: A pointer to an array of integers representing the sequence of tokens.
    - `n_token`: The number of tokens in the sequence.
    - `workers`: A reference to a vector of threads used for parallel computation.
    - `nll`: A reference to a double where the cumulative negative log-likelihood will be stored.
    - `nll2`: A reference to a double where the cumulative square of the negative log-likelihood will be stored.
    - `logit_history`: A pointer to an array of floats where the logit history for each token will be stored.
    - `prob_history`: A pointer to an array of floats where the probability history for each token will be stored.
- **Control Flow**:
    - Initialize a mutex and a counter for thread synchronization and token processing.
    - Define a lambda function `compute` that calculates local NLL and NLL2 for a subset of tokens, updating shared NLL and NLL2 upon completion.
    - Within the lambda, acquire a lock to safely increment the counter and check if all tokens have been processed.
    - If not all tokens are processed, release the lock and compute the log-softmax for the current token, updating local NLL and NLL2, and storing logit and probability in their respective histories.
    - Spawn threads from the `workers` vector to execute the `compute` lambda concurrently.
    - Execute the `compute` lambda in the main thread as well.
    - Join all threads to ensure completion of all computations before proceeding.
- **Output**: The function does not return a value but updates the `nll`, `nll2`, `logit_history`, and `prob_history` with the computed results.
- **Functions called**:
    - [`log_softmax`](#log_softmax)


---
### compute\_imatrix<!-- {{#callable:compute_imatrix}} -->
The `compute_imatrix` function processes input text to compute and optionally evaluate perplexity over multiple chunks of tokens using a language model context.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which represents the context of the language model being used.
    - `params`: A `common_params` structure containing various parameters for the computation, such as the input prompt, number of chunks, batch size, and whether to compute perplexity.
- **Control Flow**:
    - Retrieve the model and vocabulary from the context.
    - Check if the beginning-of-sequence (BOS) token should be added and assert that the end-of-sequence (EOS) token is not added.
    - Tokenize the input prompt and log the time taken for tokenization.
    - If `i_chunk` is greater than 0, remove the specified number of initial chunks from the tokens.
    - Check if there are enough tokens for the specified context size; if not, log an error and return false.
    - Initialize vectors for logit and probability history if perplexity computation is enabled.
    - Determine the maximum number of chunks and set the number of chunks to process based on parameters.
    - Initialize variables for counting and tracking negative log likelihood (NLL).
    - Log the number of chunks and batch size being processed.
    - Create worker threads for parallel processing.
    - For each chunk, clear the key-value cache and initialize a batch for processing.
    - For each batch within a chunk, adjust tokens, clear the batch, and add tokens to the batch.
    - Decode the batch and handle errors if decoding fails.
    - If computing perplexity, gather logits and process them to update NLL and logit/probability history.
    - Log the estimated time per pass and perplexity if applicable.
    - After processing all chunks, compute and log the final perplexity estimate if enabled.
    - Return true to indicate successful computation.
- **Output**: A boolean value indicating whether the computation was successful (true) or not (false).
- **Functions called**:
    - [`process_logits`](#process_logits)


---
### main<!-- {{#callable:main}} -->
The `main` function initializes and configures parameters, processes input files, and manages the computation and saving of imatrix data using a specified model and context.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of character pointers listing all the arguments.
- **Control Flow**:
    - Initialize `common_params` structure with default values for output file, context size, and escape flag.
    - Parse command-line arguments using [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse); if parsing fails, print usage and return 1.
    - Initialize common resources with `common_init()`.
    - Set the batch size in `params` to the minimum of `params.n_batch` and `params.n_ctx`.
    - Set parameters for the global `IMatrixCollector` instance `g_collector`.
    - Iterate over input files in `params.in_files`, loading each imatrix using `g_collector.load_imatrix`; if loading fails, log an error and return 1.
    - If multiple input files are provided, save the combined imatrix using `g_collector.save_imatrix()`.
    - Initialize the backend and NUMA settings with `llama_backend_init()` and `llama_numa_init()`.
    - Set evaluation callback and user data in `params`.
    - Initialize the model and context using `common_init_from_params`; if initialization fails, log an error and return 1.
    - Check if the specified context size exceeds the model's training context size and log a warning if so.
    - Print system information using `common_params_get_system_info()`.
    - If no prompt is provided, check for input files; if none are present, log an error and return 1.
    - If a prompt is provided, compute the imatrix using [`compute_imatrix`](#compute_imatrix); if computation fails, return 1.
    - Save the imatrix using `g_collector.save_imatrix()`.
    - Print performance context information using `llama_perf_context_print()`.
    - Free backend resources with `llama_backend_free()`.
    - Return 0 to indicate successful execution.
- **Output**: The function returns an integer status code, 0 for success and 1 for failure.
- **Functions called**:
    - [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse)
    - [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_numa_init`](../../src/llama.cpp.driver.md#llama_numa_init)
    - [`compute_imatrix`](#compute_imatrix)
    - [`llama_backend_free`](../../src/llama.cpp.driver.md#llama_backend_free)


