# Purpose
The provided code is a C header file, `mtmd_helper.h`, which defines a set of helper functions for the `libmtmd` library. This library appears to be focused on processing multimedia data, as indicated by the functions that handle image and audio file formats. The header file includes functions for initializing bitmap structures from files or buffers, counting tokens and positions in data chunks, evaluating data chunks, and decoding image chunks. These functions are designed to facilitate operations on multimedia data, such as decoding and encoding, and are intended to be used in conjunction with the `ggml`, `llama`, and `mtmd` libraries, as suggested by the included headers.

The functions defined in this header file are marked with `MTMD_API`, indicating that they are part of the public API of the `libmtmd` library. The functions are designed to be thread-safe, except for those that evaluate chunks, which are explicitly noted as not thread-safe. The header file also includes C++ compatibility through the use of `extern "C"`, allowing the functions to be used in C++ projects. The documentation within the file warns that these helper functions are not guaranteed to be stable and may undergo breaking changes, suggesting that the library is still under active development or subject to frequent updates.
# Imports and Dependencies

---
- `ggml.h`
- `llama.h`
- `mtmd.h`
- `stddef.h`
- `stdint.h`
- `stdbool.h`


# Function Declarations (Public API)

---
### mtmd\_helper\_get\_n\_tokens<!-- {{#callable_declaration:mtmd_helper_get_n_tokens}} -->
Counts the total number of tokens in a list of input chunks.
- **Description**: This function is used to calculate the total number of tokens present in a given list of input chunks, which is useful for managing the KV cache in processing pipelines. It should be called when you need to determine the total token count across multiple chunks. The function expects a valid pointer to an `mtmd_input_chunks` structure and will return the total token count as a size_t value. Ensure that the `chunks` parameter is properly initialized and not null before calling this function.
- **Inputs**:
    - `chunks`: A pointer to an `mtmd_input_chunks` structure representing the list of input chunks. Must not be null. The function assumes the structure is properly initialized and accessible.
- **Output**: Returns the total number of tokens as a size_t value.
- **See also**: [`mtmd_helper_get_n_tokens`](mtmd-helper.cpp.driver.md#mtmd_helper_get_n_tokens)  (Implementation)


---
### mtmd\_helper\_get\_n\_pos<!-- {{#callable_declaration:mtmd_helper_get_n_pos}} -->
Counts the total position of tokens from a list of chunks.
- **Description**: This function calculates the total position of tokens from a given list of input chunks, which is useful for tracking the number of past positions (n_past) in certain contexts. It is particularly relevant when the position count differs from the token count, such as in M-RoPE scenarios. The function should be called when you need to determine the cumulative position of tokens across multiple chunks. Ensure that the input chunks are properly initialized and valid before calling this function.
- **Inputs**:
    - `chunks`: A pointer to an mtmd_input_chunks structure representing the list of input chunks. It must be a valid, non-null pointer, and the chunks should be properly initialized.
- **Output**: Returns the total position of tokens as a llama_pos value, which is an integer type.
- **See also**: [`mtmd_helper_get_n_pos`](mtmd-helper.cpp.driver.md#mtmd_helper_get_n_pos)  (Implementation)


---
### mtmd\_helper\_eval\_chunks<!-- {{#callable_declaration:mtmd_helper_eval_chunks}} -->
Evaluates a series of input chunks using the llama and mtmd contexts.
- **Description**: This function processes a list of input chunks by evaluating each one in sequence using the provided mtmd and llama contexts. It is designed to handle both text and image chunks, automatically managing the necessary encoding and decoding steps. The function should be used when you need to process multiple chunks in a batch, updating the past position counter as it progresses. It is important to note that this function is not thread-safe, so it should not be called concurrently from multiple threads. The function returns an error code if any chunk evaluation fails, otherwise it returns 0 on success.
- **Inputs**:
    - `ctx`: A pointer to an mtmd_context structure. This must be a valid, initialized context and must not be null. The caller retains ownership.
    - `lctx`: A pointer to a llama_context structure. This must be a valid, initialized context and must not be null. The caller retains ownership.
    - `chunks`: A pointer to an mtmd_input_chunks structure containing the chunks to be evaluated. This must not be null and should contain at least one chunk.
    - `n_past`: A llama_pos value representing the number of past positions to consider. It should be initialized to the current past position count.
    - `seq_id`: A llama_seq_id value representing the sequence identifier for the evaluation. It should be a valid sequence ID.
    - `n_batch`: An integer specifying the batch size for processing. It should be a positive integer.
    - `logits_last`: A boolean indicating whether to compute logits for the last chunk only. Set to true to compute logits for the last chunk, false otherwise.
    - `new_n_past`: A pointer to a llama_pos where the updated past position will be stored. This must not be null, and the function will update it with the new past position after processing.
- **Output**: Returns 0 on success, or an error code if any chunk evaluation fails.
- **See also**: [`mtmd_helper_eval_chunks`](mtmd-helper.cpp.driver.md#mtmd_helper_eval_chunks)  (Implementation)


---
### mtmd\_helper\_eval\_chunk\_single<!-- {{#callable_declaration:mtmd_helper_eval_chunk_single}} -->
Processes a single input chunk for evaluation.
- **Description**: Use this function to evaluate a single input chunk, which can be of text, image, or audio type, within a given context. It handles the decoding of text chunks and the encoding and subsequent decoding of image and audio chunks. This function is not thread-safe and should be used when you need to process individual chunks rather than a batch. It updates the position of tokens processed and returns an error code if any processing step fails.
- **Inputs**:
    - `ctx`: A pointer to an mtmd_context structure. Must not be null. The caller retains ownership.
    - `lctx`: A pointer to a llama_context structure. Must not be null. The caller retains ownership.
    - `chunk`: A pointer to an mtmd_input_chunk structure representing the input chunk to be processed. Must not be null. The caller retains ownership.
    - `n_past`: A llama_pos value representing the number of past tokens. It is used to track the position of tokens.
    - `seq_id`: A llama_seq_id value representing the sequence identifier for the chunk being processed.
    - `n_batch`: An integer specifying the maximum number of tokens to process in a single batch. Must be positive.
    - `logits_last`: A boolean indicating whether to compute logits for the last token in the batch.
    - `new_n_past`: A pointer to a llama_pos where the function will store the updated number of past tokens. Must not be null.
- **Output**: Returns 0 on success, or a non-zero error code if any processing step fails.
- **See also**: [`mtmd_helper_eval_chunk_single`](mtmd-helper.cpp.driver.md#mtmd_helper_eval_chunk_single)  (Implementation)


---
### mtmd\_helper\_decode\_image\_chunk<!-- {{#callable_declaration:mtmd_helper_decode_image_chunk}} -->
Decode an image chunk and handle batching and pre/post decoding setup.
- **Description**: Use this function to decode an image chunk whose embeddings have already been calculated. It manages the batching process and sets up the necessary pre and post decoding configurations, such as handling non-causal attention if required. This function should be used when you have a single image chunk to decode, and it is not thread-safe. Ensure that the input chunk is of a valid image type before calling this function, as it will return an error if the chunk is not an image or audio type.
- **Inputs**:
    - `ctx`: A pointer to an mtmd_context structure. Must not be null. The caller retains ownership.
    - `lctx`: A pointer to a llama_context structure. Must not be null. The caller retains ownership.
    - `chunk`: A pointer to an mtmd_input_chunk structure representing the image chunk to decode. Must not be null and must be of image or audio type.
    - `encoded_embd`: A pointer to a float array where the decoded embeddings will be stored. Must not be null and should have sufficient space to hold the embeddings.
    - `n_past`: A llama_pos value representing the number of past positions. Used for setting the position in the decoding process.
    - `seq_id`: A llama_seq_id value representing the sequence identifier for the decoding process.
    - `n_batch`: An integer specifying the number of tokens to process in each batch. Must be positive.
    - `new_n_past`: A pointer to a llama_pos where the updated number of past positions will be stored. Must not be null.
- **Output**: Returns 0 on success, -1 if the chunk is not a valid image or audio type, and 1 if there is a decode failure.
- **See also**: [`mtmd_helper_decode_image_chunk`](mtmd-helper.cpp.driver.md#mtmd_helper_decode_image_chunk)  (Implementation)


