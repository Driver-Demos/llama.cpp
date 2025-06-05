# Purpose
This C++ source code file is designed to handle multimedia data processing, specifically focusing on image and audio data. It provides a comprehensive set of functionalities for initializing, processing, and managing multimedia data, including images and audio, within a context that supports both vision and audio models. The code defines several structures, such as `mtmd_bitmap`, `mtmd_image_tokens`, `mtmd_audio_tokens`, and `mtmd_input_chunk`, which are used to represent raw image data, preprocessed image and audio tokens, and input data chunks, respectively. These structures facilitate the handling of multimedia data by encapsulating relevant attributes and methods for cloning and managing data.

The file also includes the implementation of a context management system through the [`mtmd_context`](#mtmd_contextmtmd_context) class, which initializes and manages the multimedia processing context, including the setup of vision and audio models, handling of special tokens, and preprocessing of image and audio data. The context is initialized using the [`mtmd_init_from_file`](#mtmd_init_from_file) function, which sets up the necessary parameters and validates the compatibility of the models. Additionally, the file provides public API functions for initializing and managing multimedia data, such as [`mtmd_bitmap_init`](#mtmd_bitmap_init), [`mtmd_input_chunks_init`](#mtmd_input_chunks_init), and [`mtmd_tokenize`](#mtmd_tokenize), which allow for the creation, manipulation, and tokenization of multimedia data. Overall, this code serves as a library for multimedia data processing, providing a structured approach to handling images and audio within a machine learning or data processing pipeline.
# Imports and Dependencies

---
- `clip.h`
- `clip-impl.h`
- `mtmd.h`
- `mtmd-audio.h`
- `llama.h`
- `algorithm`
- `cerrno`
- `cstdio`
- `cstdlib`
- `cstring`
- `limits`
- `vector`


# Data Structures

---
### mtmd\_bitmap<!-- {{#data_structure:mtmd_bitmap}} -->
- **Type**: `struct`
- **Members**:
    - `nx`: Represents the number of pixels in the x-direction of the bitmap.
    - `ny`: Represents the number of pixels in the y-direction of the bitmap.
    - `data`: Stores the raw image data in a vector of unsigned characters, with a layout of RGBRGBRGB.
    - `id`: An optional user-defined identifier, which can be set to an image hash for key-value cache tracking.
    - `is_audio`: A boolean flag indicating whether the bitmap represents audio data.
- **Description**: The `mtmd_bitmap` struct is designed to represent raw image data, with the layout of the data being RGBRGBRGB. It includes fields for the dimensions of the image (`nx` and `ny`), the actual image data stored in a vector of unsigned characters, and an optional identifier (`id`) that can be used for tracking purposes. Additionally, it has a boolean flag (`is_audio`) to indicate if the bitmap is actually audio data, allowing the struct to be used for both image and audio representations.


---
### mtmd\_image\_tokens<!-- {{#data_structure:mtmd_image_tokens}} -->
- **Type**: `struct`
- **Members**:
    - `nx`: Number of tokens in the x direction.
    - `ny`: Number of tokens in the y direction.
    - `use_mrope_pos`: Indicates whether to use M-RoPE position counting, treating the whole image as one temporal position.
    - `batch_f32`: Holds preprocessed image patches.
    - `id`: Optional user-defined ID for KV cache tracking.
- **Description**: The `mtmd_image_tokens` struct is designed to represent a collection of image tokens, specifically for handling image data in a tokenized form. It includes dimensions `nx` and `ny` to specify the number of tokens in the x and y directions, respectively. The `use_mrope_pos` boolean flag determines if M-RoPE position counting is used, which treats the entire image as a single temporal position. The `batch_f32` member stores preprocessed image patches, facilitating further processing or analysis. Additionally, the `id` field allows for an optional user-defined identifier, which can be useful for tracking purposes, such as in key-value cache systems. The struct also provides a method to calculate the total number of tokens and a clone function to create a copy of the struct.

**Methods**

---
#### mtmd\_image\_tokens::n\_tokens<!-- {{#callable:mtmd_image_tokens::n_tokens}} -->
The `n_tokens` function calculates the total number of tokens by multiplying the number of tokens in the x and y directions.
- **Inputs**: None
- **Control Flow**:
    - The function returns the product of `nx` and `ny`, which are the number of tokens in the x and y directions, respectively.
- **Output**: The function returns a `uint32_t` representing the total number of tokens.
- **See also**: [`mtmd_image_tokens`](mtmd.h.driver.md#mtmd_image_tokens)  (Data Structure)


---
#### mtmd\_image\_tokens::clone<!-- {{#callable:mtmd_image_tokens::clone}} -->
The `clone` function creates a copy of an `mtmd_image_tokens` object with the same properties as the original.
- **Inputs**: None
- **Control Flow**:
    - The function returns a new `mtmd_image_tokens` object.
    - The new object is initialized with the same `nx`, `ny`, `use_mrope_pos`, `batch_f32.clone()`, and `id` as the original object.
- **Output**: A new `mtmd_image_tokens` object that is a copy of the original.
- **See also**: [`mtmd_image_tokens`](mtmd.h.driver.md#mtmd_image_tokens)  (Data Structure)



---
### mtmd\_audio\_tokens<!-- {{#data_structure:mtmd_audio_tokens}} -->
- **Type**: `struct`
- **Members**:
    - `n_tokens`: Represents the number of tokens.
    - `batch_f32`: Holds preprocessed image patches.
    - `id`: An optional user-defined ID, useful for KV cache tracking.
- **Description**: The `mtmd_audio_tokens` struct is designed to encapsulate audio token data, including the number of tokens, preprocessed image patches, and an optional user-defined identifier for tracking purposes. This struct provides a `clone` method to create a copy of itself, ensuring that the data can be duplicated when necessary. It is used in conjunction with audio processing and tokenization tasks, particularly in contexts where audio data needs to be managed and tracked efficiently.
- **Member Functions**:
    - [`mtmd_audio_tokens::clone`](#mtmd_audio_tokensclone)

**Methods**

---
#### mtmd\_audio\_tokens::clone<!-- {{#callable:mtmd_audio_tokens::clone}} -->
The `clone` function creates a copy of an `mtmd_audio_tokens` object with the same number of tokens, a cloned batch of preprocessed image patches, and the same ID.
- **Inputs**: None
- **Control Flow**:
    - The function returns a new `mtmd_audio_tokens` object.
    - It initializes the new object with the same `n_tokens` value as the current object.
    - It calls the `clone` method on `batch_f32` to create a copy of the preprocessed image patches.
    - It assigns the same `id` to the new object as the current object.
- **Output**: A new `mtmd_audio_tokens` object that is a copy of the current object.
- **See also**: [`mtmd_audio_tokens`](#mtmd_audio_tokens)  (Data Structure)



---
### mtmd\_input\_chunk<!-- {{#data_structure:mtmd_input_chunk}} -->
- **Type**: `struct`
- **Members**:
    - `type`: Specifies the type of the input chunk, which can be text, image, or audio.
    - `tokens_text`: A vector of llama_token representing the text tokens in the chunk.
    - `tokens_image`: A unique pointer to mtmd_image_tokens, representing the image tokens in the chunk.
    - `tokens_audio`: A unique pointer to mtmd_audio_tokens, representing the audio tokens in the chunk.
- **Description**: The `mtmd_input_chunk` struct is a versatile data structure designed to encapsulate different types of media input chunks, including text, image, and audio. It uses a type field to distinguish between the chunk types and stores the corresponding tokens in separate fields. The text tokens are stored in a vector, while image and audio tokens are managed through unique pointers to their respective token structures, allowing for efficient memory management and flexibility in handling various media types.


---
### mtmd\_input\_chunks<!-- {{#data_structure:mtmd_input_chunks}} -->
- **Type**: `struct`
- **Members**:
    - `entries`: A vector containing multiple `mtmd_input_chunk` objects.
- **Description**: The `mtmd_input_chunks` struct is a container for a collection of `mtmd_input_chunk` objects, which can represent different types of input data such as text, image, or audio. This struct is used to manage and process multiple input chunks as a single entity, facilitating operations that involve handling various media types in a unified manner.


---
### mtmd\_slice\_tmpl<!-- {{#data_structure:mtmd_slice_tmpl}} -->
- **Type**: `enum`
- **Members**:
    - `MTMD_SLICE_TMPL_NONE`: Represents the absence of a slice template.
    - `MTMD_SLICE_TMPL_MINICPMV_2_5`: Represents the slice template for the MiniCPMV 2.5 format.
    - `MTMD_SLICE_TMPL_MINICPMV_2_6`: Represents the slice template for the MiniCPMV 2.6 format.
    - `MTMD_SLICE_TMPL_LLAMA4`: Represents the slice template for the Llama4 format.
- **Description**: The `mtmd_slice_tmpl` enum defines various slice templates used by certain models to correctly place special tokens around image embeddings. These templates are used to handle different formats such as MiniCPMV 2.5, MiniCPMV 2.6, and Llama4, each specifying a unique way to structure and process image embeddings with special tokens. The enum provides a way to specify which template to use, or to indicate that no template is used with `MTMD_SLICE_TMPL_NONE`.


---
### mtmd\_context<!-- {{#data_structure:mtmd_context}} -->
- **Type**: `struct`
- **Members**:
    - `ctx_v`: Pointer to a vision clip context.
    - `ctx_a`: Pointer to an audio clip context.
    - `text_model`: Pointer to a llama model for text processing.
    - `image_embd_v`: Vector of floats representing image embedding.
    - `print_timings`: Boolean flag to indicate if timings should be printed.
    - `n_threads`: Integer representing the number of threads to use.
    - `media_marker`: String used as a marker for media.
    - `n_embd_text`: Constant integer representing the number of text embeddings.
    - `img_beg`: String marking the beginning of image embeddings.
    - `img_end`: String marking the end of image embeddings.
    - `aud_beg`: String marking the beginning of audio embeddings.
    - `aud_end`: String marking the end of audio embeddings.
    - `slice_tmpl`: Enum indicating the slice template used for special tokens.
    - `tok_ov_img_start`: Token for the start of an overview image.
    - `tok_ov_img_end`: Token for the end of an overview image.
    - `tok_slices_start`: Token for the start of all slices.
    - `tok_slices_end`: Token for the end of all slices.
    - `tok_sli_img_start`: Token for the start of a single slice.
    - `tok_sli_img_end`: Token for the end of a single slice.
    - `tok_sli_img_mid`: Token for between two slices.
    - `tok_row_end`: Token for the end of a row.
    - `tok_row_end_trail`: Boolean indicating if there is a trailing end-of-row token.
    - `ov_img_first`: Boolean indicating if the overview image is first.
    - `use_mrope`: Boolean indicating if M-RoPE is used.
    - `w_filters`: Pre-calculated mel filter bank for whisper.
- **Description**: The `mtmd_context` struct is a comprehensive data structure designed to manage and process multi-modal data, specifically handling vision and audio contexts alongside text models. It includes pointers to vision and audio clip contexts, a text model, and various configurations for embedding vectors and processing parameters. The struct also manages special tokens and markers for image and audio embeddings, supporting different slice templates for models like llava-uhd. Additionally, it includes settings for thread management, media markers, and pre-calculated filters for audio processing, making it a versatile component for multi-modal data handling.

**Methods**

---
#### mtmd\_context::mtmd\_context<!-- {{#callable:mtmd_context::mtmd_context}} -->
The `mtmd_context` constructor initializes a multimedia context by setting up text, vision, and audio models, validating their compatibility, and preparing them for processing multimedia data.
- **Inputs**:
    - `mmproj_fname`: A constant character pointer representing the filename of the multimedia project to be loaded.
    - `text_model`: A pointer to a `llama_model` structure representing the text model to be used in the context.
    - `ctx_params`: A reference to an `mtmd_context_params` structure containing various parameters for initializing the context, such as GPU usage, verbosity, and media markers.
- **Control Flow**:
    - Initialize member variables with values from `ctx_params` and `text_model`.
    - Check if a custom image marker is used and throw an error if it is, as it is no longer supported.
    - Ensure the `media_marker` is not empty, throwing an error if it is.
    - Initialize CLIP context parameters and attempt to load the CLIP model using [`clip_init`](clip.cpp.driver.md#clip_init), storing the results in `ctx_v` and `ctx_a`.
    - If both vision and audio contexts are present, validate that their embedding dimensions match, throwing an error if they do not.
    - Validate that the text model's embedding dimension matches the CLIP model's embedding dimension, throwing an error if they do not.
    - Initialize vision and audio components if their respective contexts are present.
- **Output**: The constructor does not return a value, but it initializes the `mtmd_context` object and may throw runtime errors if certain conditions are not met.
- **Functions called**:
    - [`clip_init`](clip.cpp.driver.md#clip_init)
    - [`string_format`](clip-impl.h.driver.md#string_format)
    - [`clip_n_mmproj_embd`](clip.cpp.driver.md#clip_n_mmproj_embd)
    - [`mtmd_context::init_vision`](#mtmd_contextinit_vision)
    - [`mtmd_context::init_audio`](#mtmd_contextinit_audio)
- **See also**: [`mtmd_context`](mtmd.h.driver.md#mtmd_context)  (Data Structure)


---
#### mtmd\_context::init\_vision<!-- {{#callable:mtmd_context::init_vision}} -->
The `init_vision` function initializes the vision context by setting up the appropriate slice templates, tokens, and image embedding markers based on the projector type and version of the vision model.
- **Inputs**: None
- **Control Flow**:
    - Assert that the vision context (`ctx_v`) is not null.
    - Determine if M-RoPE should be used by checking if the context is Qwen2VL.
    - Retrieve the projector type and minicpmv version from the vision context.
    - Based on the minicpmv version, set the slice template and tokens for image and slice start/end markers, and determine if the overview image is first or last.
    - If the minicpmv version is unsupported, assert false.
    - For the llama 4 projector type, set the slice template and tokens for image and tile separators, and determine the position of the overview image.
    - Set the beginning and end markers for image embeddings based on the projector type, with specific handling for each type.
    - Log a warning if the projector type is llama 4, indicating known degraded quality.
- **Output**: The function does not return any value; it modifies the state of the `mtmd_context` object by setting various member variables related to vision processing.
- **Functions called**:
    - [`clip_is_qwen2vl`](clip.cpp.driver.md#clip_is_qwen2vl)
    - [`clip_get_projector_type`](clip.cpp.driver.md#clip_get_projector_type)
    - [`clip_is_minicpmv`](clip.cpp.driver.md#clip_is_minicpmv)
    - [`mtmd_context::lookup_token`](#mtmd_contextlookup_token)
- **See also**: [`mtmd_context`](mtmd.h.driver.md#mtmd_context)  (Data Structure)


---
#### mtmd\_context::init\_audio<!-- {{#callable:mtmd_context::init_audio}} -->
The `init_audio` function initializes audio processing by asserting the presence of an audio context, setting up whisper filters if applicable, and configuring audio embedding markers based on the projector type.
- **Inputs**: None
- **Control Flow**:
    - Assert that the audio context `ctx_a` is not null using `GGML_ASSERT`.
    - Retrieve the projector type for the audio context using [`clip_get_projector_type`](clip.cpp.driver.md#clip_get_projector_type).
    - Check if the audio context has a whisper encoder using [`clip_has_whisper_encoder`](clip.cpp.driver.md#clip_has_whisper_encoder); if true, set `w_filters` to 128-bin whisper filters.
    - Log a warning message indicating that audio input is experimental and may have reduced quality.
    - If the projector type is `PROJECTOR_TYPE_QWEN2A`, set `aud_beg` to `<|audio_bos|>` and `aud_end` to `<|audio_eos|>`.
- **Output**: The function does not return any value; it modifies the state of the audio context and related variables.
- **Functions called**:
    - [`clip_get_projector_type`](clip.cpp.driver.md#clip_get_projector_type)
    - [`clip_has_whisper_encoder`](clip.cpp.driver.md#clip_has_whisper_encoder)
- **See also**: [`mtmd_context`](mtmd.h.driver.md#mtmd_context)  (Data Structure)


---
#### mtmd\_context::get\_clip\_ctx<!-- {{#callable:mtmd_context::get_clip_ctx}} -->
The `get_clip_ctx` function returns the appropriate CLIP context based on the type of input chunk provided.
- **Inputs**:
    - `chunk`: A pointer to an `mtmd_input_chunk` structure, which contains information about the type of input (image or audio) and associated tokens.
- **Control Flow**:
    - Check if the `type` of the `chunk` is `MTMD_INPUT_CHUNK_TYPE_IMAGE`; if true, return `ctx_v` (vision context).
    - Check if the `type` of the `chunk` is `MTMD_INPUT_CHUNK_TYPE_AUDIO`; if true, return `ctx_a` (audio context).
    - If the `type` is neither image nor audio, call `GGML_ABORT` with an error message indicating an unknown chunk type.
- **Output**: Returns a pointer to a `clip_ctx` structure, which is either the vision context (`ctx_v`) or the audio context (`ctx_a`) based on the input chunk type.
- **See also**: [`mtmd_context`](mtmd.h.driver.md#mtmd_context)  (Data Structure)


---
#### mtmd\_context::proj\_type\_v<!-- {{#callable:mtmd_context::proj_type_v}} -->
The `proj_type_v` function returns the projector type for the vision context if it exists, otherwise it returns `PROJECTOR_TYPE_UNKNOWN`.
- **Inputs**: None
- **Control Flow**:
    - Check if `ctx_v` (vision context) is not null.
    - If `ctx_v` is not null, call `clip_get_projector_type(ctx_v)` to get the projector type.
    - If `ctx_v` is null, return `PROJECTOR_TYPE_UNKNOWN`.
- **Output**: The function returns a `projector_type` which indicates the type of projector used for the vision context, or `PROJECTOR_TYPE_UNKNOWN` if the vision context is not available.
- **Functions called**:
    - [`clip_get_projector_type`](clip.cpp.driver.md#clip_get_projector_type)
- **See also**: [`mtmd_context`](mtmd.h.driver.md#mtmd_context)  (Data Structure)


---
#### mtmd\_context::proj\_type\_a<!-- {{#callable:mtmd_context::proj_type_a}} -->
The `proj_type_a` function returns the projector type for the audio context if it exists, otherwise it returns `PROJECTOR_TYPE_UNKNOWN`.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Check if `ctx_a` (audio context) is not null.
    - If `ctx_a` is not null, call `clip_get_projector_type(ctx_a)` to get the projector type for the audio context.
    - If `ctx_a` is null, return `PROJECTOR_TYPE_UNKNOWN`.
- **Output**: Returns a `projector_type` which is either the type of the audio context or `PROJECTOR_TYPE_UNKNOWN` if the audio context is not available.
- **Functions called**:
    - [`clip_get_projector_type`](clip.cpp.driver.md#clip_get_projector_type)
- **See also**: [`mtmd_context`](mtmd.h.driver.md#mtmd_context)  (Data Structure)


---
#### mtmd\_context::\~mtmd\_context<!-- {{#callable:mtmd_context::~mtmd_context}} -->
The destructor `~mtmd_context` releases resources by freeing the audio and vision context pointers.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`clip_free`](clip.cpp.driver.md#clip_free) on `ctx_a` to free the audio context.
    - The function calls [`clip_free`](clip.cpp.driver.md#clip_free) on `ctx_v` to free the vision context.
- **Output**: The function does not return any value as it is a destructor.
- **Functions called**:
    - [`clip_free`](clip.cpp.driver.md#clip_free)
- **See also**: [`mtmd_context`](mtmd.h.driver.md#mtmd_context)  (Data Structure)


---
#### mtmd\_context::lookup\_token<!-- {{#callable:mtmd_context::lookup_token}} -->
The `lookup_token` function searches for a given token text in the vocabulary of a text model and returns its corresponding token index.
- **Inputs**:
    - `token_text`: A string representing the text of the token to be looked up in the vocabulary.
- **Control Flow**:
    - Retrieve the vocabulary from the text model using `llama_model_get_vocab`.
    - Determine the number of tokens in the vocabulary using `llama_vocab_n_tokens`.
    - Iterate over each token index in the vocabulary.
    - For each token, convert it to its text representation using [`token_to_piece`](#mtmd_contexttoken_to_piece).
    - Compare the text representation with the input `token_text`.
    - If a match is found, return the current token index.
    - If no match is found after iterating through all tokens, return `LLAMA_TOKEN_NULL`.
- **Output**: Returns the index of the token if found, otherwise returns `LLAMA_TOKEN_NULL`.
- **Functions called**:
    - [`mtmd_context::token_to_piece`](#mtmd_contexttoken_to_piece)
- **See also**: [`mtmd_context`](mtmd.h.driver.md#mtmd_context)  (Data Structure)


---
#### mtmd\_context::token\_to\_piece<!-- {{#callable:mtmd_context::token_to_piece}} -->
The `token_to_piece` function converts a given token from a vocabulary into its corresponding string representation, handling special tokens if necessary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure representing the vocabulary from which the token is derived.
    - `token`: A `llama_token` representing the token to be converted into a string piece.
    - `special`: A boolean indicating whether special handling for special tokens is required.
- **Control Flow**:
    - Initialize an empty string `piece` and resize it to use its internal cache.
    - Call `llama_token_to_piece` to convert the token into a string, storing the result in `piece`.
    - Check if the number of characters `n_chars` returned is negative, indicating the buffer was too small.
    - If `n_chars` is negative, resize `piece` to the required size and call `llama_token_to_piece` again to fill it.
    - Assert that the second call to `llama_token_to_piece` returns the expected negative size.
    - If `n_chars` is non-negative, resize `piece` to the actual number of characters.
    - Return the resulting string `piece`.
- **Output**: A `std::string` containing the string representation of the token.
- **See also**: [`mtmd_context`](mtmd.h.driver.md#mtmd_context)  (Data Structure)



---
### mtmd\_tokenizer<!-- {{#data_structure:mtmd_tokenizer}} -->
- **Type**: `struct`
- **Members**:
    - `ctx`: A pointer to an mtmd_context object, which holds the context for tokenization.
    - `bitmaps`: A vector of pointers to mtmd_bitmap objects, representing the bitmaps to be tokenized.
    - `input_text`: A string containing the input text to be tokenized.
    - `add_special`: A boolean indicating whether special tokens should be added during tokenization.
    - `parse_special`: A boolean indicating whether special tokens should be parsed during tokenization.
    - `vocab`: A pointer to a llama_vocab object, representing the vocabulary used for tokenization.
    - `cur`: An mtmd_input_chunks object that holds the current state of tokenized input chunks.
- **Description**: The `mtmd_tokenizer` struct is designed to handle the tokenization of text and media inputs, such as images and audio, within a given context. It maintains a reference to the context (`mtmd_context`), a collection of bitmaps to be processed, and the input text to be tokenized. The struct also manages the addition and parsing of special tokens, and utilizes a vocabulary (`llama_vocab`) for tokenization. The current state of tokenized input is stored in `mtmd_input_chunks`, allowing for the processing and conversion of input data into a structured format suitable for further analysis or model input.
- **Member Functions**:
    - [`mtmd_tokenizer::mtmd_tokenizer`](#mtmd_tokenizer::mtmd_tokenizer)
    - [`mtmd_tokenizer::tokenize`](#mtmd_tokenizer::tokenize)
    - [`mtmd_tokenizer::add_text`](#mtmd_tokenizer::add_text)
    - [`mtmd_tokenizer::add_text`](#mtmd_tokenizer::add_text)
    - [`mtmd_tokenizer::add_media`](#mtmd_tokenizer::add_media)
    - [`mtmd_tokenizer::split_batch_to_chunk`](#mtmd_tokenizer::split_batch_to_chunk)
    - [`mtmd_tokenizer::split_text`](#mtmd_tokenizer::split_text)
    - [`mtmd_tokenizer::mtmd_tokenize_text_internal`](#mtmd_tokenizer::mtmd_tokenize_text_internal)

**Methods**

---
#### mtmd\_tokenizer::mtmd\_tokenizer<!-- {{#callable:mtmd_tokenizer::mtmd_tokenizer}} -->
The `mtmd_tokenizer` constructor initializes a tokenizer object for processing multimedia text and media markers, setting up the context, input text, and vocabulary, while converting image markers to media markers for compatibility.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` object, which contains the context for processing text and media.
    - `text`: A pointer to an `mtmd_input_text` object, which contains the input text and flags for adding and parsing special tokens.
    - `bitmaps`: A pointer to an array of `mtmd_bitmap` pointers, representing the media (images or audio) to be processed.
    - `n_bitmaps`: The number of bitmaps in the `bitmaps` array.
- **Control Flow**:
    - Initialize the `ctx` member with the provided context pointer.
    - Initialize the `bitmaps` member as a vector containing the provided bitmaps.
    - Set the `add_special` and `parse_special` flags from the `text` input.
    - Set the `input_text` from the `text` input.
    - Retrieve the vocabulary from the context's text model and assign it to the `vocab` member.
    - Replace all occurrences of the default image marker in `input_text` with the media marker from the context.
- **Output**: The constructor does not return a value; it initializes the `mtmd_tokenizer` object.
- **Functions called**:
    - [`string_replace_all`](clip-impl.h.driver.md#string_replace_all)
- **See also**: [`mtmd_tokenizer`](#mtmd_tokenizer)  (Data Structure)


---
#### mtmd\_tokenizer::tokenize<!-- {{#callable:mtmd_tokenizer::tokenize}} -->
The `tokenize` function processes input text and associated media markers to generate a sequence of text and media chunks, optionally adding special tokens, and returns the result in an `mtmd_input_chunks` structure.
- **Inputs**:
    - `output`: A pointer to an `mtmd_input_chunks` structure where the tokenized output will be stored.
- **Control Flow**:
    - Clear the current entries in `cur.entries`.
    - Split the `input_text` into parts using the `media_marker` as a delimiter.
    - Iterate over each part in the split text.
    - If a part is a media marker, attempt to add the corresponding bitmap as a media chunk; log an error and return 1 if there are more markers than bitmaps.
    - If a part is text, add it as a text chunk, optionally parsing special tokens.
    - If `add_special` is true and the vocabulary supports adding a BOS token, add a BOS token to the first text chunk or create a new chunk with the BOS token.
    - If `add_special` is true and the vocabulary supports adding an EOS token, add an EOS token to the last text chunk.
    - Check if the number of processed bitmaps matches the number of markers; log an error and return 1 if they do not match.
    - Move the `cur` structure into the `output` parameter.
    - Return 0 to indicate success.
- **Output**: Returns an integer status code: 0 for success, 1 for a mismatch between the number of bitmaps and markers, or a non-zero value from [`add_media`](#mtmd_tokenizer::add_media) if an error occurs during media addition.
- **Functions called**:
    - [`mtmd_tokenizer::split_text`](#mtmd_tokenizer::split_text)
    - [`mtmd_tokenizer::add_media`](#mtmd_tokenizer::add_media)
    - [`mtmd_tokenizer::add_text`](#mtmd_tokenizer::add_text)
- **See also**: [`mtmd_tokenizer`](#mtmd_tokenizer)  (Data Structure)


---
#### mtmd\_tokenizer::add\_text<!-- {{#callable:mtmd_tokenizer::add_text}} -->
The [`add_text`](#mtmd_tokenizer::add_text) function logs the input text and tokenizes it using the internal tokenizer, then adds the resulting tokens to the current input chunks.
- **Inputs**:
    - `txt`: A constant reference to a string representing the text to be added.
    - `parse_special`: A boolean indicating whether special tokens should be parsed during tokenization.
- **Control Flow**:
    - Log the function name and the input text using `LOG_DBG`.
    - Tokenize the input text using [`mtmd_tokenize_text_internal`](#mtmd_tokenizer::mtmd_tokenize_text_internal) with the provided vocabulary, without adding special tokens, but considering the `parse_special` flag.
    - Call the overloaded [`add_text`](#mtmd_tokenizer::add_text) function with the resulting tokens to add them to the current input chunks.
- **Output**: This function does not return a value; it modifies the current input chunks by adding the tokenized text.
- **Functions called**:
    - [`mtmd_tokenizer::mtmd_tokenize_text_internal`](#mtmd_tokenizer::mtmd_tokenize_text_internal)
    - [`mtmd_tokenizer::add_text`](#mtmd_tokenizer::add_text)
- **See also**: [`mtmd_tokenizer`](#mtmd_tokenizer)  (Data Structure)


---
#### mtmd\_tokenizer::add\_text<!-- {{#callable:mtmd_tokenizer::add_text}} -->
The `add_text` function appends a vector of `llama_token` tokens to the current input chunks, either by adding them to the last text chunk if it exists or by creating a new text chunk.
- **Inputs**:
    - `tokens`: A constant reference to a vector of `llama_token` objects representing the tokens to be added.
- **Control Flow**:
    - Check if the `tokens` vector is empty; if so, return immediately.
    - Check if the last entry in `cur.entries` is a text chunk; if it is, append the `tokens` to this chunk's `tokens_text`.
    - If the last entry is not a text chunk, create a new `mtmd_input_chunk` of type `MTMD_INPUT_CHUNK_TYPE_TEXT` with the `tokens` and add it to `cur.entries`.
- **Output**: The function does not return a value; it modifies the `cur.entries` member of the `mtmd_tokenizer` class.
- **See also**: [`mtmd_tokenizer`](#mtmd_tokenizer)  (Data Structure)


---
#### mtmd\_tokenizer::add\_media<!-- {{#callable:mtmd_tokenizer::add_media}} -->
The `add_media` function processes a given `mtmd_bitmap` object, either as an image or audio, and adds the processed data as input chunks to the current tokenizer context.
- **Inputs**:
    - `bitmap`: A pointer to an `mtmd_bitmap` object representing either an image or audio data, with attributes such as dimensions, data buffer, and a flag indicating if it is audio.
- **Control Flow**:
    - Check if the bitmap is not audio; if true, handle it as an image.
    - Verify if the context supports vision input; if not, log an error and return 2.
    - Add an image begin token if available.
    - Convert the bitmap data to a `clip_image_u8` object and preprocess it into a `clip_image_f32_batch`.
    - If preprocessing fails, log an error and return 2.
    - If the context uses a specific slice template, split the batch into chunks and add them with appropriate tokens.
    - If not using a slice template, calculate the number of tokens and create an `mtmd_image_tokens` object, then add it as a chunk.
    - Add an image end token if available.
    - If the bitmap is audio, verify if the context supports audio input; if not, log an error and return 2.
    - Check if the audio data is empty; if true, log an error and return 2.
    - Add an audio begin token if available.
    - Preprocess the audio data into mel spectrogram chunks; if preprocessing fails, log an error and return 2.
    - For each mel spectrogram, convert it to a `clip_image_f32` object, calculate tokens, and create an `mtmd_audio_tokens` object, then add it as a chunk.
    - Add an audio end token if available.
    - Return 0 indicating successful processing.
- **Output**: Returns an integer status code: 0 for success, or 2 for errors such as unsupported input types or preprocessing failures.
- **Functions called**:
    - [`mtmd_tokenizer::add_text`](#mtmd_tokenizer::add_text)
    - [`clip_image_preprocess`](clip.cpp.driver.md#clip_image_preprocess)
    - [`mtmd_tokenizer::split_batch_to_chunk`](#mtmd_tokenizer::split_batch_to_chunk)
    - [`clip_n_output_tokens`](clip.cpp.driver.md#clip_n_output_tokens)
    - [`clip_n_output_tokens_x`](clip.cpp.driver.md#clip_n_output_tokens_x)
    - [`clip_n_output_tokens_y`](clip.cpp.driver.md#clip_n_output_tokens_y)
- **See also**: [`mtmd_tokenizer`](#mtmd_tokenizer)  (Data Structure)


---
#### mtmd\_tokenizer::split\_batch\_to\_chunk<!-- {{#callable:mtmd_tokenizer::split_batch_to_chunk}} -->
The `split_batch_to_chunk` function splits a batch of preprocessed image data into individual image chunks, each associated with a unique identifier.
- **Inputs**:
    - `batch_f32`: A `clip_image_f32_batch` object containing preprocessed image data entries to be split into chunks.
    - `id`: A `std::string` representing a unique identifier for the image tokens, used for tracking purposes.
- **Control Flow**:
    - Initialize an empty vector `chunks` to store the resulting image chunks.
    - Iterate over each entry in `batch_f32.entries`.
    - For each entry, create a new `mtmd_image_tokens` object and set its `nx` to the number of output tokens for the entry, `ny` to 1, and move the entry into `image_tokens->batch_f32.entries`.
    - Assign the provided `id` to `image_tokens->id`.
    - Create a new `mtmd_input_chunk` of type `MTMD_INPUT_CHUNK_TYPE_IMAGE`, with the `image_tokens` and no text or audio tokens.
    - Move the newly created chunk into the `chunks` vector.
    - Return the `chunks` vector containing all the created image chunks.
- **Output**: A `std::vector<mtmd_input_chunk>` containing the split image chunks, each with its own `mtmd_image_tokens` and associated identifier.
- **Functions called**:
    - [`clip_n_output_tokens`](clip.cpp.driver.md#clip_n_output_tokens)
- **See also**: [`mtmd_tokenizer`](#mtmd_tokenizer)  (Data Structure)


---
#### mtmd\_tokenizer::split\_text<!-- {{#callable:mtmd_tokenizer::split_text}} -->
The `split_text` function splits a given input string into a vector of substrings based on a specified delimiter, including the delimiter itself as separate elements in the result.
- **Inputs**:
    - `input`: A constant reference to a `std::string` representing the input text to be split.
    - `delimiter`: A constant reference to a `std::string` representing the delimiter used to split the input text.
- **Control Flow**:
    - Initialize an empty vector `result` to store the split substrings.
    - Check if the input string is empty; if so, return the empty `result` vector.
    - Initialize `start` and `pos` to zero to track the current position in the input string.
    - Use a while loop to find the position of the delimiter starting from `start` using `input.find(delimiter, start)`.
    - If a delimiter is found and there is text before it, extract the substring from `start` to `pos` and add it to `result`.
    - Add the delimiter itself to `result`.
    - Update `start` to the position after the delimiter.
    - Continue the loop until no more delimiters are found (i.e., `input.find` returns `std::string::npos`).
    - After the loop, if there is any remaining text after the last delimiter, add it to `result`.
    - Return the `result` vector containing the split substrings and delimiters.
- **Output**: A `std::vector<std::string>` containing the substrings of the input text split by the delimiter, including the delimiter itself as separate elements.
- **See also**: [`mtmd_tokenizer`](#mtmd_tokenizer)  (Data Structure)


---
#### mtmd\_tokenizer::mtmd\_tokenize\_text\_internal<!-- {{#callable:mtmd_tokenizer::mtmd_tokenize_text_internal}} -->
The `mtmd_tokenize_text_internal` function tokenizes a given text into a vector of `llama_token` objects, optionally adding special tokens and parsing special sequences.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure, which contains the vocabulary used for tokenization.
    - `text`: A constant reference to a `std::string` representing the text to be tokenized.
    - `add_special`: A boolean flag indicating whether special tokens should be added to the tokenized output.
    - `parse_special`: A boolean flag indicating whether special sequences in the text should be parsed during tokenization.
- **Control Flow**:
    - Calculate the upper limit for the number of tokens as the length of the text plus twice the value of `add_special`.
    - Initialize a `std::vector` of `llama_token` with the calculated size.
    - Call `llama_tokenize` to tokenize the text into the `result` vector, updating `n_tokens` with the actual number of tokens produced.
    - If `n_tokens` is negative, resize the `result` vector to the absolute value of `n_tokens` and re-tokenize the text, asserting that the re-tokenization produces the expected number of tokens.
    - If `n_tokens` is non-negative, resize the `result` vector to `n_tokens`.
- **Output**: A `std::vector<llama_token>` containing the tokenized representation of the input text.
- **See also**: [`mtmd_tokenizer`](#mtmd_tokenizer)  (Data Structure)



# Functions

---
### mtmd\_default\_marker<!-- {{#callable:mtmd_default_marker}} -->
The `mtmd_default_marker` function returns a default string marker used to denote media content.
- **Inputs**: None
- **Control Flow**:
    - The function is defined to return a constant character pointer.
    - It directly returns the string "<__media__>".
- **Output**: A constant character pointer to the string "<__media__>".


---
### mtmd\_context\_params\_default<!-- {{#callable:mtmd_context_params_default}} -->
The `mtmd_context_params_default` function initializes and returns a default set of parameters for an MTMD context.
- **Inputs**: None
- **Control Flow**:
    - Create an instance of `mtmd_context_params` named `params`.
    - Set `params.use_gpu` to `true`.
    - Set `params.print_timings` to `true`.
    - Set `params.n_threads` to `4`.
    - Set `params.verbosity` to `GGML_LOG_LEVEL_INFO`.
    - Set `params.image_marker` to `MTMD_DEFAULT_IMAGE_MARKER`.
    - Set `params.media_marker` to the result of `mtmd_default_marker()`.
    - Return the `params` object.
- **Output**: Returns an `mtmd_context_params` object with default settings for GPU usage, thread count, verbosity, and markers.
- **Functions called**:
    - [`mtmd_default_marker`](#mtmd_default_marker)


---
### mtmd\_init\_from\_file<!-- {{#callable:mtmd_init_from_file}} -->
The `mtmd_init_from_file` function initializes an `mtmd_context` object from a file, a text model, and context parameters, handling exceptions by logging errors and returning `nullptr` if initialization fails.
- **Inputs**:
    - `mmproj_fname`: A constant character pointer representing the file name of the multimedia project to be used for initialization.
    - `text_model`: A pointer to a `llama_model` structure representing the text model to be used in the context.
    - `ctx_params`: A structure of type `mtmd_context_params` containing parameters for initializing the context, such as GPU usage, verbosity, and markers.
- **Control Flow**:
    - The function attempts to create a new `mtmd_context` object using the provided file name, text model, and context parameters.
    - If the initialization is successful, it returns a pointer to the newly created `mtmd_context` object.
    - If an exception is thrown during initialization, it catches the exception, logs an error message with the function name and exception details, and returns `nullptr`.
- **Output**: Returns a pointer to an `mtmd_context` object if successful, or `nullptr` if an error occurs during initialization.


---
### mtmd\_free<!-- {{#callable:mtmd_free}} -->
The `mtmd_free` function deallocates memory for a given `mtmd_context` object if it is not null.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` object that needs to be deallocated.
- **Control Flow**:
    - Check if the `ctx` pointer is not null.
    - If `ctx` is not null, deallocate the memory using `delete`.
- **Output**: The function does not return any value.


---
### mtmd\_tokenize<!-- {{#callable:mtmd_tokenize}} -->
The `mtmd_tokenize` function initializes a tokenizer with the given context, text, and bitmaps, and then tokenizes the input text and media into output chunks.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` structure, which contains the context for tokenization, including model and configuration details.
    - `output`: A pointer to an `mtmd_input_chunks` structure where the tokenized output will be stored.
    - `text`: A pointer to an `mtmd_input_text` structure containing the input text to be tokenized.
    - `bitmaps`: A pointer to an array of `mtmd_bitmap` pointers, representing media (images or audio) to be tokenized alongside the text.
    - `n_bitmaps`: The number of bitmaps in the `bitmaps` array.
- **Control Flow**:
    - Create an `mtmd_tokenizer` object using the provided context, text, bitmaps, and number of bitmaps.
    - Call the `tokenize` method on the `mtmd_tokenizer` object, passing the `output` pointer to store the tokenized result.
    - Return the result of the `tokenize` method, which indicates success or failure.
- **Output**: An `int32_t` value indicating the success (0) or failure (non-zero) of the tokenization process.


---
### mtmd\_encode\_chunk<!-- {{#callable:mtmd_encode_chunk}} -->
The `mtmd_encode_chunk` function encodes a given input chunk based on its type (text, image, or audio) using the appropriate model context and returns a status code indicating success or failure.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` structure that contains the model context and configuration for encoding.
    - `chunk`: A pointer to an `mtmd_input_chunk` structure representing the input data to be encoded, which can be of type text, image, or audio.
- **Control Flow**:
    - Check if the chunk type is `MTMD_INPUT_CHUNK_TYPE_TEXT`; if so, log a warning and return 0 as text chunks are not processed.
    - If the chunk type is `MTMD_INPUT_CHUNK_TYPE_IMAGE`, check if the vision context (`ctx_v`) is available; if not, log an error and return 1. Otherwise, call [`mtmd_encode`](#mtmd_encode) with the image tokens and return its result.
    - If the chunk type is `MTMD_INPUT_CHUNK_TYPE_AUDIO`, check if the audio context (`ctx_a`) is available; if not, log an error and return 1. Otherwise, resize the image embedding vector, encode the audio tokens using [`clip_image_batch_encode`](clip.cpp.driver.md#clip_image_batch_encode), and return 0 if successful or 1 if not.
    - If the chunk type is unknown, log an error and return 1.
- **Output**: Returns an `int32_t` status code: 0 for successful encoding or no effect, and 1 for errors or unsupported chunk types.
- **Functions called**:
    - [`mtmd_encode`](#mtmd_encode)
    - [`clip_image_batch_encode`](clip.cpp.driver.md#clip_image_batch_encode)


---
### mtmd\_encode<!-- {{#callable:mtmd_encode}} -->
The `mtmd_encode` function encodes image tokens into an embedding vector using a vision context from a CLIP model.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` structure, which contains the vision context and other parameters for encoding.
    - `image_tokens`: A pointer to an `mtmd_image_tokens` structure, which contains preprocessed image patches and token information.
- **Control Flow**:
    - Retrieve the vision context (`ctx_v`) from the `mtmd_context` structure.
    - Check if the vision context is available; if not, log an error and return 1.
    - Determine the number of embedding dimensions (`n_mmproj_embd`) using the vision context.
    - Resize the image embedding vector in the context to accommodate the total number of tokens times the number of embedding dimensions.
    - Initialize a boolean `ok` to false to track the success of encoding.
    - Check if the vision context corresponds to specific models (`llava`, `minicpmv`, or `glm`) that do not support batched encoding.
    - If so, iterate over each entry in the image tokens, calculate the number of tokens per image, and encode each image individually using [`clip_image_encode`](clip.cpp.driver.md#clip_image_encode).
    - If not, perform a batched encoding of all image tokens using [`clip_image_batch_encode`](clip.cpp.driver.md#clip_image_batch_encode).
    - Return 0 if encoding was successful (`ok` is true), otherwise return 1.
- **Output**: Returns an `int32_t` indicating success (0) or failure (1) of the encoding process.
- **Functions called**:
    - [`clip_n_mmproj_embd`](clip.cpp.driver.md#clip_n_mmproj_embd)
    - [`clip_is_llava`](clip.cpp.driver.md#clip_is_llava)
    - [`clip_is_minicpmv`](clip.cpp.driver.md#clip_is_minicpmv)
    - [`clip_is_glm`](clip.cpp.driver.md#clip_is_glm)
    - [`clip_n_output_tokens`](clip.cpp.driver.md#clip_n_output_tokens)
    - [`clip_image_encode`](clip.cpp.driver.md#clip_image_encode)
    - [`clip_image_batch_encode`](clip.cpp.driver.md#clip_image_batch_encode)


---
### mtmd\_get\_output\_embd<!-- {{#callable:mtmd_get_output_embd}} -->
The function `mtmd_get_output_embd` retrieves the image embedding vector from the given `mtmd_context`.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` structure, which contains the image embedding vector among other context-related data.
- **Control Flow**:
    - The function accesses the `image_embd_v` member of the `mtmd_context` structure pointed to by `ctx`.
    - It returns the data pointer of the `image_embd_v` vector.
- **Output**: A pointer to a float array representing the image embedding vector stored in the `mtmd_context`.


---
### mtmd\_decode\_use\_non\_causal<!-- {{#callable:mtmd_decode_use_non_causal}} -->
The function `mtmd_decode_use_non_causal` checks if the vision context in the `mtmd_context` is using a projector of type `PROJECTOR_TYPE_GEMMA3` and returns true if so.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` structure, which contains information about the vision and audio contexts, among other things.
- **Control Flow**:
    - Check if the `ctx_v` member of the `mtmd_context` is not null, indicating that a vision context is present.
    - If a vision context is present, retrieve the projector type using `clip_get_projector_type(ctx->ctx_v)`.
    - Compare the retrieved projector type to `PROJECTOR_TYPE_GEMMA3`.
    - Return `true` if the projector type is `PROJECTOR_TYPE_GEMMA3`, otherwise return `false`.
- **Output**: A boolean value indicating whether the vision context uses a non-causal projector of type `PROJECTOR_TYPE_GEMMA3`.
- **Functions called**:
    - [`clip_get_projector_type`](clip.cpp.driver.md#clip_get_projector_type)


---
### mtmd\_decode\_use\_mrope<!-- {{#callable:mtmd_decode_use_mrope}} -->
The function `mtmd_decode_use_mrope` checks if the M-RoPE (Multi-Resolution Positional Encoding) is used in the given context.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` structure, which contains various settings and states for media processing.
- **Control Flow**:
    - The function accesses the `use_mrope` boolean member of the `mtmd_context` structure pointed to by `ctx`.
    - It returns the value of `ctx->use_mrope`.
- **Output**: A boolean value indicating whether M-RoPE is used in the context.


---
### mtmd\_support\_vision<!-- {{#callable:mtmd_support_vision}} -->
The `mtmd_support_vision` function checks if the given `mtmd_context` supports vision processing by verifying if the vision context (`ctx_v`) is not null.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` structure, which contains various context information including vision and audio processing capabilities.
- **Control Flow**:
    - The function checks if the `ctx_v` member of the `mtmd_context` structure is not null.
    - If `ctx_v` is not null, it indicates that vision processing is supported, and the function returns true.
    - If `ctx_v` is null, it indicates that vision processing is not supported, and the function returns false.
- **Output**: A boolean value indicating whether vision processing is supported (`true` if supported, `false` otherwise).


---
### mtmd\_support\_audio<!-- {{#callable:mtmd_support_audio}} -->
The `mtmd_support_audio` function checks if the audio context (`ctx_a`) in a given `mtmd_context` is initialized.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` structure, which contains various context information including audio and vision contexts.
- **Control Flow**:
    - The function checks if the `ctx_a` member of the `mtmd_context` structure is not a null pointer.
    - If `ctx_a` is not null, it indicates that the audio context is supported, and the function returns `true`.
    - If `ctx_a` is null, it indicates that the audio context is not supported, and the function returns `false`.
- **Output**: A boolean value indicating whether the audio context is supported (`true` if supported, `false` otherwise).


---
### mtmd\_get\_audio\_bitrate<!-- {{#callable:mtmd_get_audio_bitrate}} -->
The `mtmd_get_audio_bitrate` function retrieves the audio bitrate from a given `mtmd_context` if audio support is available.
- **Inputs**:
    - `ctx`: A pointer to an `mtmd_context` structure, which contains the context for multimedia processing, including audio and vision components.
- **Control Flow**:
    - Check if the `ctx_a` member of the `mtmd_context` is null.
    - If `ctx_a` is null, return -1 indicating that audio support is not available.
    - If `ctx_a` is not null, return a fixed audio bitrate of 16000 (16kHz).
- **Output**: Returns an integer representing the audio bitrate in Hz, or -1 if audio support is not available.


---
### mtmd\_bitmap\_init<!-- {{#callable:mtmd_bitmap_init}} -->
The `mtmd_bitmap_init` function initializes a new `mtmd_bitmap` structure with specified dimensions and raw image data.
- **Inputs**:
    - `nx`: The width of the bitmap in pixels.
    - `ny`: The height of the bitmap in pixels.
    - `data`: A pointer to the raw image data, expected to be in RGB format with a length of `nx * ny * 3` bytes.
- **Control Flow**:
    - Allocate memory for a new `mtmd_bitmap` object.
    - Set the `nx` and `ny` fields of the bitmap to the provided width and height.
    - Calculate the size of the data as `nx * ny * 3` to account for RGB data.
    - Resize the `data` vector in the bitmap to accommodate the image data size.
    - Copy the provided raw image data into the bitmap's `data` vector.
    - Return the pointer to the newly initialized `mtmd_bitmap` object.
- **Output**: A pointer to the newly created `mtmd_bitmap` object containing the initialized image data.


---
### mtmd\_bitmap\_init\_from\_audio<!-- {{#callable:mtmd_bitmap_init_from_audio}} -->
The `mtmd_bitmap_init_from_audio` function initializes a `mtmd_bitmap` structure from audio data, setting its properties to represent the audio data as a bitmap.
- **Inputs**:
    - `n_samples`: The number of audio samples, which determines the width of the bitmap.
    - `data`: A pointer to the audio data, represented as an array of floats.
- **Control Flow**:
    - Allocate memory for a new `mtmd_bitmap` object.
    - Set the `nx` property of the bitmap to `n_samples`, representing the number of audio samples.
    - Set the `ny` property of the bitmap to 1, indicating a single row for audio data.
    - Set the `is_audio` property of the bitmap to `true`, marking it as audio data.
    - Calculate the size of the audio data in bytes by multiplying `n_samples` by the size of a float.
    - Resize the `data` vector of the bitmap to accommodate the audio data size.
    - Copy the audio data from the input `data` pointer into the bitmap's `data` vector.
    - Return the initialized `mtmd_bitmap` object.
- **Output**: A pointer to the newly initialized `mtmd_bitmap` object representing the audio data.


---
### mtmd\_bitmap\_get\_nx<!-- {{#callable:mtmd_bitmap_get_nx}} -->
The function `mtmd_bitmap_get_nx` retrieves the width (number of pixels in the x-direction) of a given bitmap.
- **Inputs**:
    - `bitmap`: A pointer to an `mtmd_bitmap` structure from which the width (nx) is to be retrieved.
- **Control Flow**:
    - The function accesses the `nx` member of the `mtmd_bitmap` structure pointed to by `bitmap`.
    - It returns the value of `nx`.
- **Output**: The function returns a `uint32_t` representing the width of the bitmap in pixels.


---
### mtmd\_bitmap\_get\_ny<!-- {{#callable:mtmd_bitmap_get_ny}} -->
The function `mtmd_bitmap_get_ny` retrieves the height (ny) of a given bitmap.
- **Inputs**:
    - `bitmap`: A pointer to an `mtmd_bitmap` structure from which the height (ny) is to be retrieved.
- **Control Flow**:
    - The function accesses the `ny` member of the `mtmd_bitmap` structure pointed to by the `bitmap` argument.
    - It returns the value of `ny`.
- **Output**: The function returns a `uint32_t` representing the height (ny) of the bitmap.


---
### mtmd\_bitmap\_get\_data<!-- {{#callable:mtmd_bitmap_get_data}} -->
The `mtmd_bitmap_get_data` function retrieves the raw image data from a given `mtmd_bitmap` structure.
- **Inputs**:
    - `bitmap`: A pointer to an `mtmd_bitmap` structure from which the raw image data is to be retrieved.
- **Control Flow**:
    - The function accesses the `data` member of the `mtmd_bitmap` structure pointed to by `bitmap`.
    - It calls the `data()` method on the `data` vector to obtain a pointer to the raw image data.
- **Output**: A pointer to the raw image data stored in the `data` vector of the `mtmd_bitmap` structure.


---
### mtmd\_bitmap\_get\_n\_bytes<!-- {{#callable:mtmd_bitmap_get_n_bytes}} -->
The function `mtmd_bitmap_get_n_bytes` returns the number of bytes in the data vector of a given `mtmd_bitmap` structure.
- **Inputs**:
    - `bitmap`: A pointer to an `mtmd_bitmap` structure, which contains image or audio data.
- **Control Flow**:
    - The function accesses the `data` member of the `mtmd_bitmap` structure pointed to by `bitmap`.
    - It calls the `size()` method on the `data` vector to get the number of bytes it contains.
    - The function returns this size as the result.
- **Output**: The function returns a `size_t` value representing the number of bytes in the `data` vector of the `mtmd_bitmap`.


---
### mtmd\_bitmap\_is\_audio<!-- {{#callable:mtmd_bitmap_is_audio}} -->
The function `mtmd_bitmap_is_audio` checks if a given `mtmd_bitmap` object represents audio data.
- **Inputs**:
    - `bitmap`: A pointer to an `mtmd_bitmap` object, which contains metadata and data for an image or audio.
- **Control Flow**:
    - The function accesses the `is_audio` member of the `mtmd_bitmap` structure pointed to by `bitmap`.
    - It returns the value of `is_audio`, which is a boolean indicating whether the bitmap is audio.
- **Output**: A boolean value indicating whether the `mtmd_bitmap` object is audio (`true`) or not (`false`).


---
### mtmd\_bitmap\_get\_id<!-- {{#callable:mtmd_bitmap_get_id}} -->
The `mtmd_bitmap_get_id` function retrieves the ID of a given `mtmd_bitmap` object as a C-style string.
- **Inputs**:
    - `bitmap`: A pointer to an `mtmd_bitmap` object from which the ID is to be retrieved.
- **Control Flow**:
    - Access the `id` member of the `mtmd_bitmap` structure pointed to by `bitmap`.
    - Convert the `id` string to a C-style string using `c_str()`.
    - Return the C-style string representation of the `id`.
- **Output**: A C-style string representing the ID of the `mtmd_bitmap` object.


---
### mtmd\_bitmap\_set\_id<!-- {{#callable:mtmd_bitmap_set_id}} -->
The `mtmd_bitmap_set_id` function sets the `id` field of an `mtmd_bitmap` structure to a given string or clears it if the string is null.
- **Inputs**:
    - `bitmap`: A pointer to an `mtmd_bitmap` structure whose `id` field is to be set.
    - `id`: A constant character pointer representing the new ID to be assigned to the bitmap; if null, the ID is cleared.
- **Control Flow**:
    - Check if the `id` is not null.
    - If `id` is not null, assign the `id` to the `bitmap->id` as a `std::string`.
    - If `id` is null, clear the `bitmap->id` string.
- **Output**: This function does not return a value; it modifies the `id` field of the `mtmd_bitmap` structure in place.


---
### mtmd\_bitmap\_free<!-- {{#callable:mtmd_bitmap_free}} -->
The `mtmd_bitmap_free` function deallocates memory for a given `mtmd_bitmap` object if it is not null.
- **Inputs**:
    - `bitmap`: A pointer to an `mtmd_bitmap` object that needs to be deallocated.
- **Control Flow**:
    - Check if the `bitmap` pointer is not null.
    - If the `bitmap` is not null, deallocate the memory using `delete`.
- **Output**: The function does not return any value.


---
### mtmd\_input\_chunks\_init<!-- {{#callable:mtmd_input_chunks_init}} -->
The `mtmd_input_chunks_init` function initializes and returns a new instance of the `mtmd_input_chunks` structure.
- **Inputs**: None
- **Control Flow**:
    - The function creates a new instance of the `mtmd_input_chunks` structure using the `new` operator.
    - It returns the newly created instance.
- **Output**: A pointer to a newly allocated `mtmd_input_chunks` object.


---
### mtmd\_input\_chunks\_size<!-- {{#callable:mtmd_input_chunks_size}} -->
The `mtmd_input_chunks_size` function returns the number of entries in a given `mtmd_input_chunks` structure.
- **Inputs**:
    - `chunks`: A pointer to an `mtmd_input_chunks` structure, which contains a vector of `mtmd_input_chunk` entries.
- **Control Flow**:
    - Access the `entries` vector within the `chunks` structure.
    - Return the size of the `entries` vector, which represents the number of input chunks.
- **Output**: The function returns a `size_t` value representing the number of entries in the `mtmd_input_chunks` structure.


---
### mtmd\_input\_chunks\_get<!-- {{#callable:mtmd_input_chunks_get}} -->
The `mtmd_input_chunks_get` function retrieves a specific input chunk from a collection of input chunks based on the provided index.
- **Inputs**:
    - `chunks`: A pointer to an `mtmd_input_chunks` structure, which contains a vector of `mtmd_input_chunk` entries.
    - `idx`: A size_t index specifying which chunk to retrieve from the `entries` vector within the `chunks` structure.
- **Control Flow**:
    - Check if the provided index `idx` is greater than or equal to the size of the `entries` vector in `chunks`.
    - If the index is out of bounds, return `nullptr`.
    - If the index is valid, return a pointer to the `mtmd_input_chunk` at the specified index in the `entries` vector.
- **Output**: A pointer to the `mtmd_input_chunk` at the specified index, or `nullptr` if the index is out of bounds.


---
### mtmd\_input\_chunks\_free<!-- {{#callable:mtmd_input_chunks_free}} -->
The `mtmd_input_chunks_free` function deallocates memory for a `mtmd_input_chunks` object if it is not null.
- **Inputs**:
    - `chunks`: A pointer to a `mtmd_input_chunks` object that needs to be deallocated.
- **Control Flow**:
    - Check if the `chunks` pointer is not null.
    - If `chunks` is not null, deallocate the memory using `delete`.
- **Output**: The function does not return any value.


---
### mtmd\_input\_chunk\_get\_type<!-- {{#callable:mtmd_input_chunk_get_type}} -->
The function `mtmd_input_chunk_get_type` retrieves the type of a given `mtmd_input_chunk`.
- **Inputs**:
    - `chunk`: A pointer to an `mtmd_input_chunk` structure from which the type is to be retrieved.
- **Control Flow**:
    - The function accesses the `type` member of the `mtmd_input_chunk` structure pointed to by `chunk`.
    - It returns the value of the `type` member.
- **Output**: The function returns an `enum mtmd_input_chunk_type` which represents the type of the input chunk.


---
### mtmd\_input\_chunk\_get\_tokens\_text<!-- {{#callable:mtmd_input_chunk_get_tokens_text}} -->
The function `mtmd_input_chunk_get_tokens_text` retrieves the text tokens from a given input chunk if it is of type text.
- **Inputs**:
    - `chunk`: A pointer to an `mtmd_input_chunk` structure, which contains information about the type of input (text, image, or audio) and the associated tokens.
    - `n_tokens_output`: A pointer to a `size_t` variable where the function will store the number of text tokens if the chunk is of type text.
- **Control Flow**:
    - Check if the input chunk's type is `MTMD_INPUT_CHUNK_TYPE_TEXT`.
    - If true, set `n_tokens_output` to the size of the `tokens_text` vector and return a pointer to the data of `tokens_text`.
    - If false, set `n_tokens_output` to 0 and return `nullptr`.
- **Output**: A pointer to the array of `llama_token` if the chunk is of type text, otherwise `nullptr`.


---
### mtmd\_input\_chunk\_get\_tokens\_image<!-- {{#callable:mtmd_input_chunk_get_tokens_image}} -->
The function `mtmd_input_chunk_get_tokens_image` retrieves the image tokens from a given input chunk if the chunk is of type image.
- **Inputs**:
    - `chunk`: A pointer to an `mtmd_input_chunk` structure, which represents a chunk of input data that may contain text, image, or audio tokens.
- **Control Flow**:
    - Check if the `type` of the `chunk` is `MTMD_INPUT_CHUNK_TYPE_IMAGE`.
    - If true, return the image tokens by calling `get()` on `chunk->tokens_image`.
    - If false, return `nullptr`.
- **Output**: A pointer to an `mtmd_image_tokens` structure if the chunk is of type image, otherwise `nullptr`.


---
### mtmd\_input\_chunk\_get\_n\_tokens<!-- {{#callable:mtmd_input_chunk_get_n_tokens}} -->
The function `mtmd_input_chunk_get_n_tokens` returns the number of tokens in a given `mtmd_input_chunk` based on its type.
- **Inputs**:
    - `chunk`: A pointer to an `mtmd_input_chunk` structure, which contains information about the type of chunk and its associated tokens.
- **Control Flow**:
    - Check if the chunk type is `MTMD_INPUT_CHUNK_TYPE_TEXT`; if so, return the size of `tokens_text`.
    - If the chunk type is `MTMD_INPUT_CHUNK_TYPE_IMAGE`, call [`mtmd_image_tokens_get_n_tokens`](#mtmd_image_tokens_get_n_tokens) on `tokens_image` and return the result.
    - If the chunk type is `MTMD_INPUT_CHUNK_TYPE_AUDIO`, return the `n_tokens` value from `tokens_audio`.
    - If the chunk type is none of the above, abort the program with an error message indicating an invalid chunk type.
- **Output**: The function returns a `size_t` value representing the number of tokens in the specified input chunk.
- **Functions called**:
    - [`mtmd_image_tokens_get_n_tokens`](#mtmd_image_tokens_get_n_tokens)


---
### mtmd\_input\_chunk\_get\_n\_pos<!-- {{#callable:mtmd_input_chunk_get_n_pos}} -->
The function `mtmd_input_chunk_get_n_pos` returns the number of positions (tokens) for a given input chunk based on its type (text, image, or audio).
- **Inputs**:
    - `chunk`: A pointer to an `mtmd_input_chunk` structure, which contains information about the type of chunk (text, image, or audio) and the associated tokens.
- **Control Flow**:
    - Check if the chunk type is `MTMD_INPUT_CHUNK_TYPE_TEXT`; if so, return the size of `tokens_text`.
    - If the chunk type is `MTMD_INPUT_CHUNK_TYPE_IMAGE`, call [`mtmd_image_tokens_get_n_pos`](#mtmd_image_tokens_get_n_pos) on `tokens_image` and return the result.
    - If the chunk type is `MTMD_INPUT_CHUNK_TYPE_AUDIO`, return the `n_tokens` from `tokens_audio`.
    - If the chunk type is none of the above, abort the program with an error message indicating an invalid chunk type.
- **Output**: Returns a `llama_pos` value representing the number of positions (tokens) in the input chunk.
- **Functions called**:
    - [`mtmd_image_tokens_get_n_pos`](#mtmd_image_tokens_get_n_pos)


---
### mtmd\_input\_chunk\_get\_id<!-- {{#callable:mtmd_input_chunk_get_id}} -->
The `mtmd_input_chunk_get_id` function retrieves the ID of an `mtmd_input_chunk` based on its type, returning the ID of either the image or audio tokens, or `nullptr` if neither type is present.
- **Inputs**:
    - `chunk`: A pointer to an `mtmd_input_chunk` structure, which contains information about the type of chunk (image, audio, or text) and associated tokens.
- **Control Flow**:
    - Check if the chunk type is `MTMD_INPUT_CHUNK_TYPE_IMAGE` and return the ID of the image tokens if true.
    - Check if the chunk type is `MTMD_INPUT_CHUNK_TYPE_AUDIO` and return the ID of the audio tokens if true.
    - Return `nullptr` if the chunk type is neither image nor audio.
- **Output**: A `const char*` representing the ID of the image or audio tokens, or `nullptr` if the chunk type is not image or audio.


---
### mtmd\_input\_chunk\_copy<!-- {{#callable:mtmd_input_chunk_copy}} -->
The `mtmd_input_chunk_copy` function creates a deep copy of a given `mtmd_input_chunk` object, including its image and audio tokens if they exist.
- **Inputs**:
    - `chunk`: A pointer to a constant `mtmd_input_chunk` object that is to be copied.
- **Control Flow**:
    - Allocate memory for a new `mtmd_input_chunk` object and initialize it with the type and text tokens from the input `chunk`.
    - Check if the input `chunk` has image tokens; if so, allocate memory for a new `mtmd_image_tokens` object, clone the image tokens from the input `chunk`, and assign them to the new `chunk`.
    - Check if the input `chunk` has audio tokens; if so, allocate memory for a new `mtmd_audio_tokens` object, clone the audio tokens from the input `chunk`, and assign them to the new `chunk`.
    - Return the newly created `mtmd_input_chunk` object.
- **Output**: A pointer to a new `mtmd_input_chunk` object that is a deep copy of the input `chunk`.


---
### mtmd\_input\_chunk\_free<!-- {{#callable:mtmd_input_chunk_free}} -->
The `mtmd_input_chunk_free` function deallocates memory for a given `mtmd_input_chunk` object if it is not null.
- **Inputs**:
    - `chunk`: A pointer to an `mtmd_input_chunk` object that needs to be deallocated.
- **Control Flow**:
    - Check if the `chunk` pointer is not null.
    - If `chunk` is not null, deallocate the memory using `delete`.
- **Output**: The function does not return any value.


---
### mtmd\_image\_tokens\_get\_n\_tokens<!-- {{#callable:mtmd_image_tokens_get_n_tokens}} -->
The function `mtmd_image_tokens_get_n_tokens` returns the total number of tokens in an `mtmd_image_tokens` object by multiplying its `nx` and `ny` attributes.
- **Inputs**:
    - `image_tokens`: A pointer to an `mtmd_image_tokens` object, which contains information about the number of tokens in the x and y directions.
- **Control Flow**:
    - The function accesses the `n_tokens()` method of the `mtmd_image_tokens` object pointed to by `image_tokens`.
    - The `n_tokens()` method calculates the total number of tokens by multiplying the `nx` and `ny` attributes of the `mtmd_image_tokens` object.
- **Output**: The function returns a `size_t` value representing the total number of tokens in the `mtmd_image_tokens` object.


---
### mtmd\_image\_tokens\_get\_nx<!-- {{#callable:mtmd_image_tokens_get_nx}} -->
The function `mtmd_image_tokens_get_nx` retrieves the number of tokens in the x-direction from an `mtmd_image_tokens` structure.
- **Inputs**:
    - `image_tokens`: A pointer to an `mtmd_image_tokens` structure from which the number of tokens in the x-direction is to be retrieved.
- **Control Flow**:
    - The function accesses the `nx` member of the `mtmd_image_tokens` structure pointed to by `image_tokens`.
    - It returns the value of `nx`.
- **Output**: The function returns a `size_t` value representing the number of tokens in the x-direction of the image tokens.


---
### mtmd\_image\_tokens\_get\_ny<!-- {{#callable:mtmd_image_tokens_get_ny}} -->
The function `mtmd_image_tokens_get_ny` retrieves the number of tokens in the y-direction from an `mtmd_image_tokens` structure.
- **Inputs**:
    - `image_tokens`: A pointer to an `mtmd_image_tokens` structure from which the number of tokens in the y-direction is to be retrieved.
- **Control Flow**:
    - The function accesses the `ny` member of the `mtmd_image_tokens` structure pointed to by `image_tokens`.
    - It returns the value of `ny`.
- **Output**: The function returns a `size_t` value representing the number of tokens in the y-direction of the image tokens.


---
### mtmd\_image\_tokens\_get\_id<!-- {{#callable:mtmd_image_tokens_get_id}} -->
The function `mtmd_image_tokens_get_id` retrieves the ID of a given `mtmd_image_tokens` object as a C-style string.
- **Inputs**:
    - `image_tokens`: A pointer to an `mtmd_image_tokens` object from which the ID is to be retrieved.
- **Control Flow**:
    - Access the `id` member of the `mtmd_image_tokens` object pointed to by `image_tokens`.
    - Convert the `id` from a `std::string` to a C-style string using `c_str()`.
    - Return the C-style string representation of the ID.
- **Output**: A C-style string representing the ID of the `mtmd_image_tokens` object.


---
### mtmd\_image\_tokens\_get\_n\_pos<!-- {{#callable:mtmd_image_tokens_get_n_pos}} -->
The function `mtmd_image_tokens_get_n_pos` returns the number of positional tokens for an image, considering whether M-RoPE positional encoding is used.
- **Inputs**:
    - `image_tokens`: A pointer to an `mtmd_image_tokens` structure, which contains information about image tokens, including dimensions and whether M-RoPE positional encoding is used.
- **Control Flow**:
    - Check if `use_mrope_pos` is true in the `image_tokens` structure.
    - If `use_mrope_pos` is true, return 1, indicating the whole image is considered as one temporal position in M-RoPE encoding.
    - If `use_mrope_pos` is false, return the total number of tokens calculated by `n_tokens()` method of `image_tokens`.
- **Output**: Returns a `llama_pos` type, which is an integer representing the number of positional tokens for the image.


---
### mtmd\_test\_create\_input\_chunks<!-- {{#callable:mtmd_test_create_input_chunks}} -->
The function `mtmd_test_create_input_chunks` initializes and returns a collection of input chunks containing both text and image data.
- **Inputs**: None
- **Control Flow**:
    - Initialize a new `mtmd_input_chunks` object using [`mtmd_input_chunks_init`](#mtmd_input_chunks_init).
    - Check if the initialization was successful; if not, return `nullptr`.
    - Create a text chunk with predefined tokens and add it to the `entries` vector of the `chunks` object.
    - Create an image chunk with specified dimensions and an ID, then add it to the `entries` vector of the `chunks` object.
    - Return the `chunks` object containing the text and image chunks.
- **Output**: A pointer to an `mtmd_input_chunks` object containing a text chunk and an image chunk, or `nullptr` if initialization fails.
- **Functions called**:
    - [`mtmd_input_chunks_init`](#mtmd_input_chunks_init)


