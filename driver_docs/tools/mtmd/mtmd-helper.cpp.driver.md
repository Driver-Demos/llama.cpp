# Purpose
This C++ source code file provides a specialized functionality for processing multimedia data, specifically focusing on handling image and audio data. The file includes several components that work together to decode and evaluate multimedia input chunks, leveraging external libraries such as `miniaudio` for audio processing and `stb_image` for image handling. The code is structured to handle different types of input chunks, including text, image, and audio, and it provides mechanisms to decode these chunks into a format suitable for further processing, such as machine learning model evaluation.

The file defines several helper functions and structures, such as [`decode_embd_batch`](#decode_embd_batchdecode_embd_batch) and [`mtmd_helper_decode_image_chunk`](#mtmd_helper_decode_image_chunk), which facilitate the decoding of image and audio data into embeddings that can be processed by a model. It also includes functions for evaluating these chunks, such as [`mtmd_helper_eval_chunk_single`](#mtmd_helper_eval_chunk_single) and [`mtmd_helper_eval_chunks`](#mtmd_helper_eval_chunks), which manage the processing of input data in batches. The code is designed to be integrated into a larger system, likely involving machine learning models, as indicated by the use of functions like `llama_decode` and `llama_set_causal_attn`. Additionally, the file includes utility functions for initializing bitmap data from buffers or files, supporting both audio and image inputs. Overall, this code provides a focused set of functionalities for multimedia data processing, with a clear emphasis on preparing data for model evaluation.
# Imports and Dependencies

---
- `windows.h`
- `mtmd.h`
- `mtmd-helper.h`
- `llama.h`
- `algorithm`
- `cinttypes`
- `vector`
- `miniaudio/miniaudio.h`
- `stb/stb_image.h`


# Data Structures

---
### decode\_embd\_batch<!-- {{#data_structure:decode_embd_batch}} -->
- **Type**: `struct`
- **Members**:
    - `n_pos_per_embd`: Stores the number of positions per embedding.
    - `n_mmproj_embd`: Stores the number of multi-modal projection embeddings.
    - `pos`: A vector of llama_pos representing positions.
    - `pos_view`: A vector of llama_pos used by mrope for position views.
    - `n_seq_id`: A vector of int32_t representing sequence IDs.
    - `seq_id_0`: A vector of llama_seq_id storing the initial sequence ID.
    - `seq_ids`: A vector of pointers to llama_seq_id representing sequence IDs.
    - `logits`: A vector of int8_t representing logits.
    - `batch`: An instance of llama_batch used to store batch information.
- **Description**: The `decode_embd_batch` struct is designed to facilitate the handling of embedding batches, particularly in the context of multi-modal data processing. It manages various attributes related to positions, sequence IDs, and logits, and provides methods to set positions for different data types such as images and audio using M-RoPE (Multi-Resolution Positional Encoding). The struct also includes a constructor to initialize its members and a method to retrieve a view of the batch with a specified offset and number of tokens.
- **Member Functions**:
    - [`decode_embd_batch::decode_embd_batch`](#decode_embd_batchdecode_embd_batch)
    - [`decode_embd_batch::set_position_normal`](#decode_embd_batchset_position_normal)
    - [`decode_embd_batch::set_position_mrope_2d`](#decode_embd_batchset_position_mrope_2d)
    - [`decode_embd_batch::set_position_mrope_1d`](#decode_embd_batchset_position_mrope_1d)
    - [`decode_embd_batch::get_view`](#decode_embd_batchget_view)

**Methods**

---
#### decode\_embd\_batch::decode\_embd\_batch<!-- {{#callable:decode_embd_batch::decode_embd_batch}} -->
The `decode_embd_batch` constructor initializes a `decode_embd_batch` object with specified embedding data and token parameters, setting up internal vectors and a `llama_batch` structure for processing embeddings.
- **Inputs**:
    - `embd`: A pointer to a float array representing the embedding data.
    - `n_tokens`: An integer representing the number of tokens to process.
    - `n_pos_per_embd`: An integer specifying the number of positions per embedding.
    - `n_mmproj_embd`: An integer indicating the number of multi-modal projection embeddings.
- **Control Flow**:
    - Initialize `n_pos_per_embd` and `n_mmproj_embd` with the provided values.
    - Resize the `pos` vector to accommodate `n_tokens * n_pos_per_embd` elements.
    - Resize the `n_seq_id` vector to hold `n_tokens` elements.
    - Resize the `seq_ids` vector to hold `n_tokens + 1` elements and set the last element to `nullptr`.
    - Resize the `logits` vector to hold `n_tokens` elements.
    - Resize the `seq_id_0` vector to hold 1 element.
    - Initialize the `batch` structure with the provided `n_tokens`, `embd`, and the data pointers of the resized vectors.
- **Output**: The function does not return a value; it initializes the internal state of the `decode_embd_batch` object.
- **See also**: [`decode_embd_batch`](#decode_embd_batch)  (Data Structure)


---
#### decode\_embd\_batch::set\_position\_normal<!-- {{#callable:decode_embd_batch::set_position_normal}} -->
The `set_position_normal` function initializes the position, sequence ID, and logits for each token in a batch starting from a given position and sequence ID.
- **Inputs**:
    - `pos_0`: The starting position (of type `llama_pos`) for the tokens in the batch.
    - `seq_id`: The sequence ID (of type `llama_seq_id`) to be assigned to each token in the batch.
- **Control Flow**:
    - The function begins by setting the first element of `seq_id_0` to the provided `seq_id`.
    - A loop iterates over each token in the batch, indexed by `i`, from 0 to `batch.n_tokens - 1`.
    - Within the loop, the position for each token is set to `pos_0 + i`.
    - The sequence ID count for each token is set to 1.
    - The sequence ID pointer for each token is set to point to `seq_id_0`.
    - The logits flag for each token is set to `false`.
- **Output**: The function does not return a value; it modifies the `batch` member of the `decode_embd_batch` structure in place.
- **See also**: [`decode_embd_batch`](#decode_embd_batch)  (Data Structure)


---
#### decode\_embd\_batch::set\_position\_mrope\_2d<!-- {{#callable:decode_embd_batch::set_position_mrope_2d}} -->
The `set_position_mrope_2d` function initializes the positional encoding for a 2D grid of tokens in a batch, setting positions based on a starting position and sequence ID.
- **Inputs**:
    - `pos_0`: The initial position value of type `llama_pos` to be used as the base for positional encoding.
    - `nx`: The number of columns in the 2D grid, representing the width of the grid.
    - `ny`: The number of rows in the 2D grid, representing the height of the grid.
    - `seq_id`: The sequence ID of type `llama_seq_id` to be associated with the positional encoding.
- **Control Flow**:
    - Assert that `n_pos_per_embd` is equal to 4, ensuring the function is used in the correct context.
    - Set the first element of `seq_id_0` to the provided `seq_id`.
    - Iterate over each row `y` from 0 to `ny-1`.
    - For each row, iterate over each column `x` from 0 to `nx-1`.
    - Calculate the linear index `i` for the current position in the 2D grid.
    - Set the position at index `i` in the `pos` vector to `pos_0`.
    - Set the position at index `i + batch.n_tokens` to `pos_0 + y`, encoding the row position.
    - Set the position at index `i + batch.n_tokens * 2` to `pos_0 + x`, encoding the column position.
    - Set the position at index `i + batch.n_tokens * 3` to 0, as the last position dimension is unused.
    - Iterate over each token index `i` from 0 to `batch.n_tokens-1`.
    - Set `batch.n_seq_id[i]` to 1, indicating a single sequence ID per token.
    - Set `batch.seq_id[i]` to point to the data of `seq_id_0`.
    - Set `batch.logits[i]` to false, indicating no logits are associated with the token.
- **Output**: The function does not return a value; it modifies the `pos`, `n_seq_id`, `seq_id`, and `logits` vectors within the `batch` structure in place.
- **See also**: [`decode_embd_batch`](#decode_embd_batch)  (Data Structure)


---
#### decode\_embd\_batch::set\_position\_mrope\_1d<!-- {{#callable:decode_embd_batch::set_position_mrope_1d}} -->
The `set_position_mrope_1d` function initializes the position and sequence ID arrays for a batch of tokens in a 1D M-RoPE (Multi-dimensional Rotary Position Embedding) context for audio processing.
- **Inputs**:
    - `pos_0`: The initial position value of type `llama_pos` to be used for setting positions in the batch.
    - `seq_id`: The sequence ID of type `llama_seq_id` to be assigned to the batch.
- **Control Flow**:
    - Assert that the number of positions per embedding (`n_pos_per_embd`) is 4, ensuring the function is used in the correct context.
    - Set the first element of `seq_id_0` to the provided `seq_id`.
    - Iterate over the number of tokens in the batch (`batch.n_tokens`).
    - For each token, set four consecutive positions in the `pos` vector, with the first three being `pos_0 + i` and the fourth being 0, indicating the last position dimension is unused.
    - Iterate again over the number of tokens in the batch to set `n_seq_id` to 1, `seq_id` to point to `seq_id_0`, and `logits` to false for each token.
- **Output**: The function does not return a value; it modifies the `pos`, `n_seq_id`, `seq_id`, and `logits` arrays within the `decode_embd_batch` structure.
- **See also**: [`decode_embd_batch`](#decode_embd_batch)  (Data Structure)


---
#### decode\_embd\_batch::get\_view<!-- {{#callable:decode_embd_batch::get_view}} -->
The `get_view` function returns a `llama_batch` view of a specified number of tokens starting from a given offset, adjusting the position data based on the number of positions per embedding.
- **Inputs**:
    - `offset`: An integer representing the starting point in the token sequence from which the view should begin.
    - `n_tokens`: An integer representing the number of tokens to include in the view.
- **Control Flow**:
    - Initialize a pointer `pos_ptr` to hold position data.
    - Clear the `pos_view` vector and reserve space for `n_tokens * n_pos_per_embd` elements.
    - Check if `n_pos_per_embd` is greater than 1 to determine if M-RoPE (multi-dimensional rotary position embedding) is used.
    - If M-RoPE is used, iterate over each position per embedding, calculate the source index, and insert the corresponding position data into `pos_view`.
    - Set `pos_ptr` to point to the data in `pos_view`.
    - If M-RoPE is not used, set `pos_ptr` to point to the position data starting at the given offset.
    - Return a `llama_batch` object initialized with the specified number of tokens, adjusted embedding, position, sequence ID, and logits data.
- **Output**: A `llama_batch` object containing the view of the specified tokens with adjusted position and embedding data.
- **See also**: [`decode_embd_batch`](#decode_embd_batch)  (Data Structure)



# Functions

---
### mtmd\_helper\_get\_n\_tokens<!-- {{#callable:mtmd_helper_get_n_tokens}} -->
The function `mtmd_helper_get_n_tokens` calculates the total number of tokens across all input chunks.
- **Inputs**:
    - `chunks`: A pointer to an `mtmd_input_chunks` structure, which contains multiple input chunks to be processed.
- **Control Flow**:
    - Initialize a variable `n_tokens` to zero to store the total number of tokens.
    - Iterate over each chunk in the `chunks` using a loop that runs from 0 to the size of `chunks`.
    - For each chunk, retrieve the chunk using `mtmd_input_chunks_get` and add the number of tokens in that chunk to `n_tokens` using [`mtmd_input_chunk_get_n_tokens`](mtmd.cpp.driver.md#mtmd_input_chunk_get_n_tokens).
    - Return the total number of tokens accumulated in `n_tokens`.
- **Output**: Returns the total number of tokens as a `size_t` value, representing the sum of tokens in all chunks.
- **Functions called**:
    - [`mtmd_input_chunks_size`](mtmd.cpp.driver.md#mtmd_input_chunks_size)
    - [`mtmd_input_chunk_get_n_tokens`](mtmd.cpp.driver.md#mtmd_input_chunk_get_n_tokens)


---
### mtmd\_helper\_get\_n\_pos<!-- {{#callable:mtmd_helper_get_n_pos}} -->
The function `mtmd_helper_get_n_pos` calculates the total number of positions across all input chunks.
- **Inputs**:
    - `chunks`: A pointer to an `mtmd_input_chunks` structure, which contains multiple input chunks to be processed.
- **Control Flow**:
    - Initialize a variable `n_pos` to zero to accumulate the total positions.
    - Iterate over each chunk in the `chunks` using a loop that runs from 0 to the size of `chunks`.
    - For each chunk, retrieve the chunk using `mtmd_input_chunks_get` and add the number of positions in the chunk to `n_pos` using [`mtmd_input_chunk_get_n_pos`](mtmd.cpp.driver.md#mtmd_input_chunk_get_n_pos).
    - Return the accumulated `n_pos` value.
- **Output**: The function returns a `llama_pos` type, representing the total number of positions across all input chunks.
- **Functions called**:
    - [`mtmd_input_chunks_size`](mtmd.cpp.driver.md#mtmd_input_chunks_size)
    - [`mtmd_input_chunk_get_n_pos`](mtmd.cpp.driver.md#mtmd_input_chunk_get_n_pos)


---
### mtmd\_helper\_decode\_image\_chunk<!-- {{#callable:mtmd_helper_decode_image_chunk}} -->
The `mtmd_helper_decode_image_chunk` function decodes an image or audio chunk into embeddings using a specified context and updates the past position counter.
- **Inputs**:
    - `ctx`: A pointer to the `mtmd_context` structure, which holds the context for the decoding process.
    - `lctx`: A pointer to the `llama_context` structure, which is used for managing the llama model context.
    - `chunk`: A pointer to the `mtmd_input_chunk` structure, representing the input chunk to be decoded.
    - `encoded_embd`: A pointer to a float array where the encoded embeddings will be stored.
    - `n_past`: A `llama_pos` value representing the number of past positions processed.
    - `seq_id`: A `llama_seq_id` value representing the sequence identifier for the current decoding process.
    - `n_batch`: An integer representing the number of tokens to process in each batch.
    - `new_n_past`: A pointer to a `llama_pos` where the updated number of past positions will be stored after decoding.
- **Control Flow**:
    - Determine the type of the input chunk and log an error if it is not an image or audio type.
    - Retrieve the model and calculate the number of embeddings and positions per embedding based on the context settings.
    - Initialize a `decode_embd_batch` object to manage the embedding batch.
    - Set the position of the embeddings using either M-RoPE or normal positioning based on the context and chunk type.
    - If non-causal decoding is enabled, disable causal attention in the llama context.
    - Iterate over batches of tokens, decoding each batch and logging the time taken for each batch.
    - Update the past position counter and store it in `new_n_past`.
    - Restore causal attention if it was disabled and return the result of the decoding process.
- **Output**: Returns an integer status code, where 0 indicates success and -1 indicates an error occurred during decoding.
- **Functions called**:
    - [`mtmd_input_chunk_get_type`](mtmd.cpp.driver.md#mtmd_input_chunk_get_type)
    - [`mtmd_decode_use_mrope`](mtmd.cpp.driver.md#mtmd_decode_use_mrope)
    - [`mtmd_input_chunk_get_n_tokens`](mtmd.cpp.driver.md#mtmd_input_chunk_get_n_tokens)
    - [`mtmd_image_tokens_get_nx`](mtmd.cpp.driver.md#mtmd_image_tokens_get_nx)
    - [`mtmd_image_tokens_get_ny`](mtmd.cpp.driver.md#mtmd_image_tokens_get_ny)
    - [`mtmd_decode_use_non_causal`](mtmd.cpp.driver.md#mtmd_decode_use_non_causal)
    - [`mtmd_input_chunk_get_n_pos`](mtmd.cpp.driver.md#mtmd_input_chunk_get_n_pos)


---
### mtmd\_helper\_eval\_chunk\_single<!-- {{#callable:mtmd_helper_eval_chunk_single}} -->
The `mtmd_helper_eval_chunk_single` function processes a single input chunk, either text, image, or audio, by encoding and decoding it using the provided context and updates the past position counter.
- **Inputs**:
    - `ctx`: A pointer to the `mtmd_context` structure, which holds the context for the MTMD operations.
    - `lctx`: A pointer to the `llama_context` structure, which is used for llama operations such as decoding.
    - `chunk`: A pointer to the `mtmd_input_chunk` structure, representing the input data chunk to be processed.
    - `n_past`: A `llama_pos` value representing the number of past positions processed before this function call.
    - `seq_id`: A `llama_seq_id` value representing the sequence identifier for the current processing.
    - `n_batch`: An `int32_t` value indicating the maximum number of tokens to process in a single batch.
    - `logits_last`: A boolean flag indicating whether to set the logits flag for the last token in the batch.
    - `new_n_past`: A pointer to a `llama_pos` variable where the updated number of past positions will be stored after processing.
- **Control Flow**:
    - Initialize a llama batch with the specified batch size.
    - Determine the type of the input chunk using [`mtmd_input_chunk_get_type`](mtmd.cpp.driver.md#mtmd_input_chunk_get_type).
    - If the chunk is of type text, retrieve tokens and process them in batches, updating positions and sequence IDs, and decode using `llama_decode`.
    - If the chunk is of type image or audio, encode the chunk, retrieve embeddings, and decode using [`mtmd_helper_decode_image_chunk`](#mtmd_helper_decode_image_chunk).
    - If the chunk type is unsupported, abort the operation.
    - Free the llama batch resources before returning.
- **Output**: Returns an `int32_t` status code, where 0 indicates success and non-zero indicates an error occurred during processing.
- **Functions called**:
    - [`mtmd_input_chunk_get_type`](mtmd.cpp.driver.md#mtmd_input_chunk_get_type)
    - [`mtmd_encode_chunk`](mtmd.cpp.driver.md#mtmd_encode_chunk)
    - [`mtmd_get_output_embd`](mtmd.cpp.driver.md#mtmd_get_output_embd)
    - [`mtmd_helper_decode_image_chunk`](#mtmd_helper_decode_image_chunk)


---
### mtmd\_helper\_eval\_chunks<!-- {{#callable:mtmd_helper_eval_chunks}} -->
The `mtmd_helper_eval_chunks` function evaluates a series of input chunks using a given context and updates the past position counter.
- **Inputs**:
    - `ctx`: A pointer to the `mtmd_context` structure, which holds the context for the evaluation.
    - `lctx`: A pointer to the `llama_context` structure, which is used for llama-specific operations.
    - `chunks`: A pointer to the `mtmd_input_chunks` structure, which contains the chunks to be evaluated.
    - `n_past`: A `llama_pos` value representing the number of past positions to consider.
    - `seq_id`: A `llama_seq_id` value representing the sequence identifier for the evaluation.
    - `n_batch`: An `int32_t` value indicating the number of batches to process at a time.
    - `logits_last`: A boolean indicating whether to compute logits for the last chunk.
    - `new_n_past`: A pointer to a `llama_pos` where the updated number of past positions will be stored.
- **Control Flow**:
    - Determine the number of chunks using [`mtmd_input_chunks_size`](mtmd.cpp.driver.md#mtmd_input_chunks_size) and check if there are any chunks to evaluate.
    - Iterate over each chunk in the `chunks` array.
    - For each chunk, determine if it is the last chunk and set `chunk_logits_last` accordingly.
    - Retrieve the current chunk using `mtmd_input_chunks_get`.
    - Call [`mtmd_helper_eval_chunk_single`](#mtmd_helper_eval_chunk_single) to evaluate the current chunk with the provided parameters.
    - If the evaluation fails (returns non-zero), log an error and return the error code.
    - Update `new_n_past` with the current `n_past` value after each chunk evaluation.
    - Return 0 upon successful evaluation of all chunks.
- **Output**: Returns an `int32_t` value, which is 0 on success or an error code if any chunk evaluation fails.
- **Functions called**:
    - [`mtmd_input_chunks_size`](mtmd.cpp.driver.md#mtmd_input_chunks_size)
    - [`mtmd_helper_eval_chunk_single`](#mtmd_helper_eval_chunk_single)


---
### is\_audio\_file<!-- {{#callable:audio_helpers::is_audio_file}} -->
The `is_audio_file` function checks if a given buffer contains data for a WAV, MP3, or FLAC audio file.
- **Inputs**:
    - `buf`: A pointer to a character array (buffer) containing the file data to be checked.
    - `len`: The size of the buffer, indicating the length of the data to be checked.
- **Control Flow**:
    - Check if the length of the buffer is less than 12; if so, return false as it cannot be a valid audio file.
    - Check if the buffer starts with 'RIFF' and contains 'WAVE' at the correct position to identify it as a WAV file.
    - Check if the buffer starts with 'ID3' or has an MPEG sync word to identify it as an MP3 file.
    - Check if the buffer starts with 'fLaC' to identify it as a FLAC file.
    - Return true if any of the above conditions for WAV, MP3, or FLAC are met; otherwise, return false.
- **Output**: A boolean value indicating whether the buffer contains a valid audio file format (WAV, MP3, or FLAC).


---
### decode\_audio\_from\_buf<!-- {{#callable:audio_helpers::decode_audio_from_buf}} -->
The `decode_audio_from_buf` function decodes audio data from a buffer into a mono PCM float vector at a specified sample rate.
- **Inputs**:
    - `buf_in`: A pointer to the input buffer containing the audio data.
    - `len`: The length of the input buffer.
    - `target_sampler_rate`: The desired sample rate for the output audio.
    - `pcmf32_mono`: A reference to a vector where the decoded mono PCM float audio data will be stored.
- **Control Flow**:
    - Initialize a decoder configuration for a single channel with the specified target sample rate using [`ma_decoder_config_init`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_decoder_configma_decoder_config_init).
    - Attempt to initialize the decoder with the input buffer and configuration using [`ma_decoder_init_memory`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_decoder_init_memory); return false if unsuccessful.
    - Retrieve the total number of PCM frames in the audio using [`ma_decoder_get_length_in_pcm_frames`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_decoder_get_length_in_pcm_frames); return false if unsuccessful.
    - Resize the output vector `pcmf32_mono` to accommodate the frame count.
    - Read the PCM frames into `pcmf32_mono` using [`ma_decoder_read_pcm_frames`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_decoder_read_pcm_frames); return false if unsuccessful.
    - If `MTMD_AUDIO_DEBUG` is defined, encode the audio data to a WAV file for debugging purposes.
    - Uninitialize the decoder to free resources.
    - Return true to indicate successful decoding.
- **Output**: A boolean value indicating whether the audio decoding was successful (true) or not (false).
- **Functions called**:
    - [`ma_decoder_config::ma_decoder_config_init`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_decoder_configma_decoder_config_init)
    - [`ma_decoder_init_memory`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_decoder_init_memory)
    - [`ma_decoder_get_length_in_pcm_frames`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_decoder_get_length_in_pcm_frames)
    - [`ma_decoder_uninit`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_decoder_uninit)
    - [`ma_decoder_read_pcm_frames`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_decoder_read_pcm_frames)
    - [`ma_encoder_config::ma_encoder_config_init`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_encoder_configma_encoder_config_init)
    - [`ma_encoder_init_file`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_encoder_init_file)
    - [`ma_encoder_write_pcm_frames`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_encoder_write_pcm_frames)
    - [`ma_encoder_uninit`](../../vendor/miniaudio/miniaudio.h.driver.md#ma_encoder_uninit)


---
### mtmd\_helper\_bitmap\_init\_from\_buf<!-- {{#callable:mtmd_helper_bitmap_init_from_buf}} -->
The function `mtmd_helper_bitmap_init_from_buf` initializes a bitmap from a buffer containing either audio or image data.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` structure, which provides context for the operation.
    - `buf`: A pointer to an unsigned char buffer containing the data to be processed, which could be audio or image data.
    - `len`: The size of the buffer `buf`, indicating the length of the data to be processed.
- **Control Flow**:
    - Check if the buffer contains audio data using `audio_helpers::is_audio_file`.
    - If it is audio data, retrieve the audio bitrate using [`mtmd_get_audio_bitrate`](mtmd.cpp.driver.md#mtmd_get_audio_bitrate) and check if it is valid.
    - If the bitrate is valid, decode the audio data from the buffer into a vector of floats using `audio_helpers::decode_audio_from_buf`.
    - If decoding is successful, initialize a bitmap from the audio data using `mtmd_bitmap_init_from_audio` and return it.
    - If the buffer is not audio data, assume it is image data and attempt to load it using [`stbi_load_from_memory`](../../vendor/stb/stb_image.h.driver.md#stbi_load_from_memory).
    - If image loading is successful, initialize a bitmap from the image data using `mtmd_bitmap_init` and return it.
    - If any step fails, log an error message and return `nullptr`.
- **Output**: A pointer to an `mtmd_bitmap` structure initialized from the buffer data, or `nullptr` if initialization fails.
- **Functions called**:
    - [`mtmd_get_audio_bitrate`](mtmd.cpp.driver.md#mtmd_get_audio_bitrate)
    - [`stbi_load_from_memory`](../../vendor/stb/stb_image.h.driver.md#stbi_load_from_memory)
    - [`stbi_image_free`](../../vendor/stb/stb_image.h.driver.md#stbi_image_free)


---
### mtmd\_helper\_bitmap\_init\_from\_file<!-- {{#callable:mtmd_helper_bitmap_init_from_file}} -->
The function `mtmd_helper_bitmap_init_from_file` initializes a bitmap from a file by reading its contents into a buffer and then processing the buffer to create a bitmap.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` structure, which provides context for the bitmap initialization process.
    - `fname`: A constant character pointer representing the name of the file to be read and processed into a bitmap.
- **Control Flow**:
    - Open the file specified by `fname` in binary read mode.
    - Check if the file was successfully opened; if not, log an error and return `nullptr`.
    - Seek to the end of the file to determine its size, then return to the beginning of the file.
    - Resize the buffer to match the file size and read the file's contents into the buffer.
    - Close the file and check if the number of bytes read matches the file size; if not, log an error and return `nullptr`.
    - Call [`mtmd_helper_bitmap_init_from_buf`](#mtmd_helper_bitmap_init_from_buf) with the context, buffer data, and buffer size to initialize the bitmap from the buffer.
- **Output**: Returns a pointer to an `mtmd_bitmap` structure initialized from the file's contents, or `nullptr` if an error occurs during file reading or processing.
- **Functions called**:
    - [`mtmd_helper_bitmap_init_from_buf`](#mtmd_helper_bitmap_init_from_buf)


