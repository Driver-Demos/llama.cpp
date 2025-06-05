# Purpose
This C++ source code file is designed to facilitate the generation of control vectors using a machine learning model, specifically leveraging the GGML (General Graphical Model Library) framework. The code is structured to handle both CPU and GPU computations, as indicated by the conditional inclusion of CUDA and Metal headers. The primary functionality revolves around processing pairs of positive and negative prompts to compute difference vectors, which are then used to generate control vectors. These vectors are subsequently exported in the GGUF format, a file format likely used for storing model-related data.

The code is organized into several key components, including utility functions, data structures for managing callback and training contexts, and the main function that orchestrates the entire process. The `callback_data` and [`train_context`](#train_contexttrain_context) structures are central to managing the state and data flow during the computation of difference vectors. The code also includes functionality for tokenizing prompts, evaluating model layers, and applying dimensionality reduction techniques such as PCA (Principal Component Analysis) or mean calculation. The presence of detailed functions for loading prompt files, handling tensor operations, and exporting results indicates that this file is part of a larger system aimed at enhancing or modifying machine learning models through control vector manipulation.
# Imports and Dependencies

---
- `ggml.h`
- `gguf.h`
- `arg.h`
- `common.h`
- `llama.h`
- `pca.hpp`
- `mean.hpp`
- `ggml-cuda.h`
- `ggml-metal.h`
- `algorithm`
- `climits`
- `cstdio`
- `cstring`
- `fstream`
- `iostream`
- `string`
- `tuple`
- `vector`


# Data Structures

---
### callback\_data<!-- {{#data_structure:callback_data}} -->
- **Type**: `struct`
- **Members**:
    - `ctx_ggml`: A pointer to a ggml_context, used to manage memory for tensors.
    - `n_layers`: An integer representing the number of layers.
    - `n_tokens`: An integer representing the number of tokens.
    - `is_eval_pos`: A boolean indicating whether the current evaluation is positive.
    - `v_pos`: A vector of pointers to ggml_tensor structures, representing positive matrices.
    - `v_neg`: A vector of pointers to ggml_tensor structures, representing negative matrices.
    - `v_diff_filtered`: A vector of pointers to ggml_tensor structures, representing filtered difference matrices.
- **Description**: The `callback_data` struct is designed to manage and process tensor data for a machine learning model, specifically handling positive and negative evaluations. It contains pointers to a ggml_context for memory management, integers for layer and token counts, and vectors of ggml_tensor pointers to store matrices for positive, negative, and filtered difference data. The struct also includes a boolean to determine the evaluation type, and provides methods to save tensors, calculate differences, filter non-zero rows, and reset the data structure for reuse.
- **Member Functions**:
    - [`callback_data::save_tensor_for_layer`](#callback_datasave_tensor_for_layer)
    - [`callback_data::calc_diff`](#callback_datacalc_diff)
    - [`callback_data::filter_nonzero_rows`](#callback_datafilter_nonzero_rows)
    - [`callback_data::reset`](#callback_datareset)

**Methods**

---
#### callback\_data::save\_tensor\_for\_layer<!-- {{#callable:callback_data::save_tensor_for_layer}} -->
The `save_tensor_for_layer` function saves a given tensor into either the `v_pos` or `v_neg` vector, depending on the `is_eval_pos` flag, after ensuring the tensor is of type `GGML_TYPE_F32` and copying its data into a new tensor within a context.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure that represents the tensor to be saved.
- **Control Flow**:
    - Assert that the tensor type is `GGML_TYPE_F32`.
    - Check if `ctx_ggml` is `nullptr`; if so, initialize it with a new context using [`ggml_init`](../../ggml/src/ggml.c.driver.md#ggml_init) with parameters based on `n_layers`.
    - Calculate the number of bytes in the tensor `t` using [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes).
    - Create a new 2D tensor `t_layer` in the context `ctx_ggml` with the same type and dimensions as `t`.
    - Allocate memory for `t_layer->data` and copy the data from `t` into `t_layer->data` using [`ggml_backend_tensor_get`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get).
    - Set the name of `t_layer` to be the same as `t` using [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Depending on the value of `is_eval_pos`, push `t_layer` into either the `v_pos` or `v_neg` vector.
- **Output**: The function does not return a value; it modifies the `v_pos` or `v_neg` vector by adding the new tensor `t_layer`.
- **Functions called**:
    - [`ggml_tensor_overhead`](../../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_init`](../../ggml/src/ggml.c.driver.md#ggml_init)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_new_tensor_2d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_backend_tensor_get`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get)
    - [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_get_name`](../../ggml/src/ggml.c.driver.md#ggml_get_name)
- **See also**: [`callback_data`](#callback_data)  (Data Structure)


---
#### callback\_data::calc\_diff<!-- {{#callable:callback_data::calc_diff}} -->
The `calc_diff` function computes the element-wise difference between corresponding tensors in `v_pos` and `v_neg`, filters out zero rows from the resulting tensors, and stores the filtered tensors in `v_diff_filtered`.
- **Inputs**: None
- **Control Flow**:
    - Iterate over each tensor in `v_pos` using a loop indexed by `il`.
    - For each tensor, cast the data pointers of the corresponding tensors in `v_pos` and `v_neg` to `float *` and store them in `a` and `b`, respectively.
    - Determine the number of elements in the current tensor using [`ggml_nelements`](../../ggml/src/ggml.c.driver.md#ggml_nelements).
    - Perform an element-wise subtraction of `b` from `a` for all elements in the tensor.
    - Filter out zero rows from the resulting tensor using [`filter_nonzero_rows`](#callback_datafilter_nonzero_rows) and store the result in `diff_filtered`.
    - Append `diff_filtered` to the `v_diff_filtered` vector.
    - Return the `v_diff_filtered` vector.
- **Output**: A `std::vector` of `ggml_tensor *` containing the filtered difference tensors.
- **Functions called**:
    - [`ggml_nelements`](../../ggml/src/ggml.c.driver.md#ggml_nelements)
    - [`callback_data::filter_nonzero_rows`](#callback_datafilter_nonzero_rows)
- **See also**: [`callback_data`](#callback_data)  (Data Structure)


---
#### callback\_data::filter\_nonzero\_rows<!-- {{#callable:callback_data::filter_nonzero_rows}} -->
The `filter_nonzero_rows` function filters out rows from a given 2D tensor that contain only zero elements, returning a new tensor with only the non-zero rows.
- **Inputs**:
    - `a`: A pointer to a `ggml_tensor` structure representing a 2D tensor from which zero rows are to be filtered out.
- **Control Flow**:
    - Define a lambda function `is_row_all_zeros` to check if a given row in the tensor is composed entirely of zero elements, with a tolerance `eps` of 1e-6.
    - Iterate over each row of the input tensor `a` to determine if it is non-zero using the `is_row_all_zeros` lambda function.
    - Store the indices of non-zero rows in a vector `rows_to_copy`.
    - Calculate the number of non-zero rows `n_nonzero_rows` and assert that it is greater than zero.
    - Create a new tensor `diff_filtered` with dimensions `[n_embd, n_nonzero_rows]` to store the non-zero rows.
    - Copy the non-zero rows from the input tensor `a` to the new tensor `diff_filtered`.
    - Return the `diff_filtered` tensor containing only the non-zero rows.
- **Output**: A pointer to a new `ggml_tensor` structure containing only the non-zero rows from the input tensor `a`.
- **Functions called**:
    - [`ggml_get_f32_nd`](../../ggml/src/ggml-cpu/ggml-cpu.c.driver.md#ggml_get_f32_nd)
    - [`ggml_new_tensor_2d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_format_name`](../../ggml/src/ggml.c.driver.md#ggml_format_name)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_set_f32_nd`](../../ggml/src/ggml-cpu/ggml-cpu.c.driver.md#ggml_set_f32_nd)
- **See also**: [`callback_data`](#callback_data)  (Data Structure)


---
#### callback\_data::reset<!-- {{#callable:callback_data::reset}} -->
The `reset` function clears and deallocates memory for vectors and context within the `callback_data` structure.
- **Inputs**: None
- **Control Flow**:
    - Iterates over each pointer in `v_pos`, `v_neg`, and `v_diff_filtered` vectors, freeing the memory allocated for their `data`.
    - Clears the `v_pos`, `v_neg`, and `v_diff_filtered` vectors to remove all elements.
    - Checks if `ctx_ggml` is not null, and if so, frees the memory associated with `ctx_ggml`.
    - Sets `ctx_ggml` to null to indicate that the context is no longer valid.
- **Output**: The function does not return any value; it performs memory cleanup and resets the state of the `callback_data` object.
- **Functions called**:
    - [`ggml_free`](../../ggml/src/ggml.c.driver.md#ggml_free)
- **See also**: [`callback_data`](#callback_data)  (Data Structure)



---
### train\_context<!-- {{#data_structure:train_context}} -->
- **Type**: `struct`
- **Members**:
    - `ctx_ggml`: A pointer to a ggml_context, used for managing memory and operations related to ggml tensors.
    - `n_embd`: An integer representing the number of embeddings.
    - `n_layers`: An integer representing the number of layers in the model.
    - `positive_entries`: A vector of strings containing positive prompt entries.
    - `negative_entries`: A vector of strings containing negative prompt entries.
    - `v_diff`: A vector of ggml_tensor pointers, each representing a matrix of size [m, n_embd] for storing differences.
    - `v_final`: A vector of ggml_tensor pointers, each representing a vector of size [n_embd] to be written to a file.
    - `v_diff_tmp`: A vector of vectors of uint8_t used to temporarily store v_diff data before conversion.
- **Description**: The `train_context` struct is designed to manage the training context for processing and storing differences between positive and negative prompt entries in a machine learning model. It holds various vectors and tensors for managing embeddings and layers, and facilitates operations such as concatenating and building difference tensors. The struct is initialized with the number of embeddings and layers, and it manages memory allocation and deallocation for the ggml tensors used in the training process.
- **Member Functions**:
    - [`train_context::train_context`](#train_contexttrain_context)
    - [`train_context::concat_diff_tmp`](#train_contextconcat_diff_tmp)
    - [`train_context::build_v_diff`](#train_contextbuild_v_diff)
    - [`train_context::~train_context`](#train_contexttrain_context)

**Methods**

---
#### train\_context::train\_context<!-- {{#callable:train_context::train_context}} -->
The `train_context` constructor initializes a training context with specified embedding and layer counts, setting up memory and tensor structures for further processing.
- **Inputs**:
    - `n_embd_`: The number of embeddings, which determines the size of each tensor in the final vector.
    - `n_layers_`: The number of layers, which influences the memory allocation and the number of tensors created.
- **Control Flow**:
    - Assigns the input parameters `n_embd_` and `n_layers_` to the class members `n_embd` and `n_layers`.
    - Initializes `ggml_init_params` with calculated memory size based on the number of layers and sets `no_alloc` to true.
    - Calls [`ggml_init`](../../ggml/src/ggml.c.driver.md#ggml_init) with the initialized parameters to create a `ggml_context`.
    - Iterates over the range from 0 to `n_layers - 1`, creating an empty vector for `v_diff_tmp` and a new 1D tensor for `v_final` for each layer.
    - Allocates memory for each tensor's data using `malloc` and appends the tensor to `v_final`.
- **Output**: The function does not return a value; it initializes the `train_context` object with configured tensors and memory context.
- **Functions called**:
    - [`ggml_tensor_overhead`](../../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_init`](../../ggml/src/ggml.c.driver.md#ggml_init)
    - [`ggml_new_tensor_1d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`train_context`](#train_context)  (Data Structure)


---
#### train\_context::concat\_diff\_tmp<!-- {{#callable:train_context::concat_diff_tmp}} -->
The `concat_diff_tmp` function appends data from a vector of ggml_tensor pointers to a temporary storage vector for each layer, ensuring the size matches the expected number of layers minus one.
- **Inputs**:
    - `diff_filtered`: A constant reference to a vector of pointers to ggml_tensor structures, representing filtered difference tensors for each layer.
- **Control Flow**:
    - Assert that the size of `diff_filtered` matches `n_layers - 1`.
    - Iterate over each layer index from 0 to `n_layers - 2`.
    - For each layer, retrieve the corresponding tensor from `diff_filtered` and the temporary storage vector `v_diff_tmp`.
    - Calculate the current size of the temporary storage vector for the layer.
    - Resize the temporary storage vector to accommodate the new data from the tensor.
    - Copy the data from the tensor into the resized portion of the temporary storage vector.
- **Output**: The function does not return a value; it modifies the `v_diff_tmp` member of the `train_context` structure in place.
- **Functions called**:
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`train_context`](#train_context)  (Data Structure)


---
#### train\_context::build\_v\_diff<!-- {{#callable:train_context::build_v_diff}} -->
The `build_v_diff` function constructs and optionally transposes a series of ggml_tensor matrices from temporary data, storing them in a vector for further processing.
- **Inputs**:
    - `transpose`: A boolean flag indicating whether the resulting matrices should be transposed.
- **Control Flow**:
    - Prints 'build_v_diff' to indicate the function execution.
    - Iterates over each layer except the last one (from 0 to n_layers - 2).
    - For each layer, retrieves the temporary data from `v_diff_tmp` and calculates the number of elements and rows.
    - Asserts that the number of elements is divisible by `n_embd`.
    - Creates a new 2D tensor with dimensions based on the `transpose` flag.
    - Allocates memory for the tensor's data.
    - If `transpose` is true, transposes the data from `v_diff_tmp` into the new tensor; otherwise, directly copies the data.
    - Appends the new tensor to the `v_diff` vector.
    - Prints debug information for the tensor.
    - Clears the temporary data for the current layer.
- **Output**: The function does not return a value but modifies the `v_diff` vector by adding new ggml_tensor matrices.
- **Functions called**:
    - [`ggml_new_tensor_2d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_set_f32_nd`](../../ggml/src/ggml-cpu/ggml-cpu.c.driver.md#ggml_set_f32_nd)
    - [`print_debug_tensor`](pca.hpp.driver.md#print_debug_tensor)
- **See also**: [`train_context`](#train_context)  (Data Structure)


---
#### train\_context::\~train\_context<!-- {{#callable:train_context::~train_context}} -->
The destructor `~train_context` releases allocated memory for data in `v_final` and `v_diff` vectors and frees the `ggml_context`.
- **Inputs**: None
- **Control Flow**:
    - Iterates over each element in the `v_final` vector and frees the memory allocated for its `data`.
    - Iterates over each element in the `v_diff` vector and frees the memory allocated for its `data`.
    - Calls [`ggml_free`](../../ggml/src/ggml.c.driver.md#ggml_free) to release the `ggml_context` pointed to by `ctx_ggml`.
- **Output**: The function does not return any value; it performs cleanup operations to prevent memory leaks.
- **Functions called**:
    - [`ggml_free`](../../ggml/src/ggml.c.driver.md#ggml_free)
- **See also**: [`train_context`](#train_context)  (Data Structure)



---
### tokenized\_prompt<!-- {{#data_structure:tokenized_prompt}} -->
- **Type**: `struct`
- **Members**:
    - `tokens_pos`: A vector of llama_token representing the positive tokens.
    - `tokens_neg`: A vector of llama_token representing the negative tokens.
    - `max_seq_len`: The maximum sequence length between tokens_pos and tokens_neg.
- **Description**: The `tokenized_prompt` struct is designed to handle and store tokenized representations of positive and negative prompts. It contains vectors for both positive and negative tokens, and calculates the maximum sequence length between them. The constructor initializes these vectors by tokenizing the input strings and ensures they are padded to the same length for consistency in processing.
- **Member Functions**:
    - [`tokenized_prompt::tokenized_prompt`](#tokenized_prompttokenized_prompt)
    - [`tokenized_prompt::padding_seq`](#tokenized_promptpadding_seq)

**Methods**

---
#### tokenized\_prompt::tokenized\_prompt<!-- {{#callable:tokenized_prompt::tokenized_prompt}} -->
The `tokenized_prompt` constructor initializes a `tokenized_prompt` object by tokenizing positive and negative input strings, determining the maximum sequence length, and padding the token sequences to this length.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which provides the context for tokenization and other operations.
    - `pos`: A `std::string` representing the positive input string to be tokenized.
    - `neg`: A `std::string` representing the negative input string to be tokenized.
- **Control Flow**:
    - Retrieve the model from the context using `llama_get_model` and then get the vocabulary from the model using `llama_model_get_vocab`.
    - Determine whether to add a beginning-of-sequence token using `llama_vocab_get_add_bos`.
    - Tokenize the positive input string `pos` using `common_tokenize`, storing the result in `tokens_pos`.
    - Tokenize the negative input string `neg` using `common_tokenize`, storing the result in `tokens_neg`.
    - Calculate the maximum sequence length `max_seq_len` as the maximum of the sizes of `tokens_pos` and `tokens_neg`.
    - Pad `tokens_pos` to `max_seq_len` using the [`padding_seq`](#tokenized_promptpadding_seq) method.
    - Pad `tokens_neg` to `max_seq_len` using the [`padding_seq`](#tokenized_promptpadding_seq) method.
- **Output**: The function does not return a value; it initializes the `tokens_pos`, `tokens_neg`, and `max_seq_len` members of the `tokenized_prompt` object.
- **Functions called**:
    - [`tokenized_prompt::padding_seq`](#tokenized_promptpadding_seq)
- **See also**: [`tokenized_prompt`](#tokenized_prompt)  (Data Structure)


---
#### tokenized\_prompt::padding\_seq<!-- {{#callable:tokenized_prompt::padding_seq}} -->
The `padding_seq` function appends a padding token to a vector of tokens until it reaches a specified length.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which provides context for tokenization.
    - `tokens`: A reference to a vector of `llama_token` objects that will be padded.
    - `len`: The desired length of the `tokens` vector after padding.
- **Control Flow**:
    - The function begins by tokenizing a single space character using `common_tokenize` to obtain a padding token.
    - It retrieves the last token from the resulting vector as the padding token.
    - A while loop checks if the size of the `tokens` vector is less than `len`.
    - If true, the padding token is appended to the `tokens` vector.
    - The loop continues until the `tokens` vector reaches the specified length `len`.
- **Output**: The function does not return a value; it modifies the `tokens` vector in place by adding padding tokens.
- **See also**: [`tokenized_prompt`](#tokenized_prompt)  (Data Structure)



# Functions

---
### tokens\_to\_str<!-- {{#callable:tokens_to_str}} -->
The `tokens_to_str` function converts a range of tokens into a concatenated string representation using a given context.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which provides the necessary context for token conversion.
    - `begin`: An iterator pointing to the beginning of the token range to be converted.
    - `end`: An iterator pointing to the end of the token range to be converted.
- **Control Flow**:
    - Initialize an empty string `ret` to accumulate the converted token pieces.
    - Iterate over the range from `begin` to `end`.
    - For each token in the range, convert it to a string piece using `common_token_to_piece` and append it to `ret`.
    - Return the accumulated string `ret`.
- **Output**: A `std::string` containing the concatenated string representation of the tokens in the specified range.


---
### print\_usage<!-- {{#callable:print_usage}} -->
The `print_usage` function displays example command-line usage instructions for a program, using the provided executable name.
- **Inputs**:
    - `int`: This parameter is not used in the function and is likely a placeholder for a standard signature.
    - `char ** argv`: An array of C-style strings representing the command-line arguments, where `argv[0]` is the name of the executable.
- **Control Flow**:
    - The function begins by printing a header 'example usage:'.
    - It then prints four different example command-line usages, each demonstrating different options for running the program, using the executable name from `argv[0]`.
    - The examples include running the program with CPU only, with GPU, with advanced options, and using a specific method.
- **Output**: The function does not return any value; it outputs text directly to the standard output.


---
### to\_string<!-- {{#callable:to_string}} -->
The `to_string` function converts a given value of any type to its string representation using a stringstream.
- **Inputs**:
    - `val`: A constant reference to a value of any type `T` that needs to be converted to a string.
- **Control Flow**:
    - A stringstream object `ss` is created.
    - The value `val` is inserted into the stringstream `ss`.
    - The string representation of `val` is obtained by calling `ss.str()` and returned.
- **Output**: A `std::string` representing the string form of the input value `val`.


---
### ctrlvec\_load\_prompt\_file<!-- {{#callable:ctrlvec_load_prompt_file}} -->
The function `ctrlvec_load_prompt_file` reads a file line by line, processes each line to handle escape sequences, and returns a vector of strings, optionally skipping empty lines.
- **Inputs**:
    - `path`: A string representing the file path to be read.
    - `skip_empty_lines`: A boolean indicating whether to skip empty lines in the file.
- **Control Flow**:
    - Initialize an empty vector of strings named `output` to store the processed lines.
    - Open the file at the given `path` using an `ifstream` object named `file`.
    - Check if the file is successfully opened; if not, print an error message and exit the program.
    - Iterate over each line in the file using a `while` loop with `std::getline`.
    - For each line, determine if it should be skipped based on the `skip_empty_lines` flag and whether the line is empty.
    - If the line is not skipped, process escape sequences in the line using `string_process_escapes` and add the processed line to the `output` vector.
    - Close the file after reading all lines.
    - Return the `output` vector containing the processed lines.
- **Output**: A vector of strings containing the processed lines from the file, with optional skipping of empty lines.


---
### cb\_eval<!-- {{#callable:cb_eval}} -->
The `cb_eval` function evaluates a tensor to determine if it should be processed further based on its name and dimensions, and saves it to a context if applicable.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure representing the tensor to be evaluated.
    - `ask`: A boolean flag indicating whether the function is being queried for a condition check or for processing.
    - `user_data`: A pointer to user-defined data, expected to be of type `callback_data`, which contains context and state information for the evaluation.
- **Control Flow**:
    - The function casts `user_data` to a `callback_data` pointer named `cb_data`.
    - It checks if the tensor's name matches the string "l_out" to determine if it is the output layer tensor.
    - If `ask` is true, the function returns whether the tensor is the output layer (`is_l_out`).
    - If `ask` is false, it checks if the tensor is not the output layer or if its second dimension does not match the number of tokens in `cb_data`; if either condition is true, it returns true.
    - If the tensor is the output layer and its dimensions match, it calls `save_tensor_for_layer` on `cb_data` to save the tensor, then returns true.
- **Output**: A boolean value indicating whether the tensor should be processed further or not.


---
### get\_hidden\_layers<!-- {{#callable:get_hidden_layers}} -->
The `get_hidden_layers` function clears the key-value cache of a given llama context and attempts to decode a batch of tokens, returning a boolean indicating success or failure.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which represents the context in which the llama model operates.
    - `tokens`: A reference to a vector of `llama_token` objects, representing the tokens to be processed.
- **Control Flow**:
    - The function begins by clearing the key-value cache of the provided llama context using `llama_kv_self_clear(ctx)`.
    - It then attempts to decode the tokens by calling `llama_decode` with the context and a batch of tokens obtained from `llama_batch_get_one(tokens.data(), tokens.size())`.
    - If `llama_decode` returns a non-zero value, indicating a failure, an error message is printed to `stderr`, and the function returns `false`.
    - If the decoding is successful, the function returns `true`.
- **Output**: A boolean value indicating whether the decoding of the tokens was successful (`true`) or not (`false`).


---
### export\_gguf<!-- {{#callable:export_gguf}} -->
The `export_gguf` function exports a set of control vectors to a GGUF file with metadata about the architecture and model hint.
- **Inputs**:
    - `v_ctrl`: A vector of pointers to `ggml_tensor` structures representing the control vectors to be exported.
    - `fname`: A string representing the filename where the GGUF data will be written.
    - `model_hint`: A string providing a hint about the model architecture to be included in the GGUF metadata.
- **Control Flow**:
    - Initialize an empty GGUF context using `gguf_init_empty()`.
    - Set the architecture and model hint metadata in the GGUF context using `gguf_set_val_str()`.
    - Set the layer count metadata in the GGUF context using `gguf_set_val_i32()`, based on the size of `v_ctrl`.
    - Iterate over each tensor in `v_ctrl`, adding it to the GGUF context with `gguf_add_tensor()` and printing debug information.
    - Write the GGUF context to the specified file using `gguf_write_to_file()`.
    - Free the GGUF context using `gguf_free()`.
- **Output**: The function does not return a value; it writes the GGUF data to the specified file.
- **Functions called**:
    - [`gguf_init_empty`](../../ggml/src/gguf.cpp.driver.md#gguf_init_empty)
    - [`gguf_set_val_str`](../../ggml/src/gguf.cpp.driver.md#gguf_set_val_str)
    - [`gguf_set_val_i32`](../../ggml/src/gguf.cpp.driver.md#gguf_set_val_i32)
    - [`gguf_add_tensor`](../../ggml/src/gguf.cpp.driver.md#gguf_add_tensor)
    - [`print_debug_tensor`](pca.hpp.driver.md#print_debug_tensor)
    - [`gguf_write_to_file`](../../ggml/src/gguf.cpp.driver.md#gguf_write_to_file)
    - [`gguf_free`](../../ggml/src/gguf.cpp.driver.md#gguf_free)


---
### prepare\_entries<!-- {{#callable:prepare_entries}} -->
The `prepare_entries` function loads positive and negative prompt files, checks for equal and non-empty prompt pairs, and assigns them to the training context.
- **Inputs**:
    - `params`: A reference to a `common_params` object containing file paths for positive and negative prompt files.
    - `ctx_train`: A reference to a `train_context` object where the loaded prompts will be stored.
- **Control Flow**:
    - Load positive prompts from the file specified in `params.cvector_positive_file` using [`ctrlvec_load_prompt_file`](#ctrlvec_load_prompt_file).
    - Load negative prompts from the file specified in `params.cvector_negative_file` using [`ctrlvec_load_prompt_file`](#ctrlvec_load_prompt_file).
    - Check if the number of positive prompts is equal to the number of negative prompts; if not, print an error message and return 1.
    - Check if the positive prompts list is empty; if so, print an error message and return 1.
    - Assign the loaded positive prompts to `ctx_train.positive_entries`.
    - Assign the loaded negative prompts to `ctx_train.negative_entries`.
    - Return 0 to indicate successful execution.
- **Output**: Returns an integer, 0 on success or 1 on failure due to mismatched or empty prompt pairs.
- **Functions called**:
    - [`ctrlvec_load_prompt_file`](#ctrlvec_load_prompt_file)


---
### main<!-- {{#callable:main}} -->
The `main` function initializes and processes a machine learning model to evaluate tokenized prompts, calculate differences, and export the results using PCA or mean methods.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of character pointers listing all the arguments.
- **Control Flow**:
    - Initialize `common_params` and set the output file name.
    - Parse command-line arguments using [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse); exit with error if parsing fails.
    - Check if PCA iterations are a multiple of PCA batch size; exit with error if not.
    - Initialize `callback_data` and set evaluation callbacks in `params`.
    - Print build information and initialize backend and NUMA settings.
    - Load the model and retrieve hyperparameters like number of layers and embeddings.
    - Initialize `train_context` with model parameters.
    - Prepare training entries by loading positive and negative prompts.
    - Tokenize prompts and calculate total number of tokens.
    - Iterate over each tokenized prompt, evaluate positive and negative tokens, and calculate differences.
    - Concatenate filtered differences into `train_context`.
    - Free model resources after evaluation.
    - Determine whether to use PCA or mean for dimensionality reduction.
    - Build difference vectors in `train_context` and run PCA or mean as specified.
    - Export the final vectors to a file using [`export_gguf`](#export_gguf).
    - Free backend resources and return 0 to indicate successful execution.
- **Output**: The function returns an integer, 0 for successful execution or 1 for errors during parameter parsing or PCA configuration.
- **Functions called**:
    - [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse)
    - [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_numa_init`](../../src/llama.cpp.driver.md#llama_numa_init)
    - [`prepare_entries`](#prepare_entries)
    - [`tokens_to_str`](#tokens_to_str)
    - [`get_hidden_layers`](#get_hidden_layers)
    - [`export_gguf`](#export_gguf)
    - [`llama_backend_free`](../../src/llama.cpp.driver.md#llama_backend_free)


