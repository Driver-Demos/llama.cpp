# Purpose
The provided C++ header file, `mtmd.h`, defines the interface for the `libmtmd` library, which is designed to add multimodal support to the `llama.cpp` project. This library is experimental and subject to changes, as indicated by the warning in the comments. The primary purpose of this library is to facilitate the processing and integration of different types of input data, such as text, images, and audio, into the `llama` model framework. The file includes both C and C++ interfaces, making it versatile for use in various programming environments. It defines several opaque data structures and functions that manage contexts, bitmaps, and input chunks, which are essential for handling multimodal data.

The file provides a comprehensive API, including functions for initializing and freeing contexts, handling bitmap data, and tokenizing input data. It also includes functionality to check model capabilities, such as support for vision and audio inputs, and to manage input chunks and their associated tokens. The header file uses conditional compilation to ensure compatibility with different platforms and compilers, and it defines macros for managing shared library exports. Additionally, the file includes C++ wrappers that utilize smart pointers for automatic resource management, enhancing memory safety and ease of use in C++ applications. Overall, `mtmd.h` serves as a crucial component for developers looking to extend the `llama` model with multimodal capabilities, providing a structured and flexible interface for integrating diverse data types.
# Imports and Dependencies

---
- `ggml.h`
- `llama.h`
- `stddef.h`
- `stdint.h`
- `stdbool.h`
- `string`
- `vector`
- `cinttypes`
- `memory`


# Data Structures

---
### mtmd\_input\_chunk\_type<!-- {{#data_structure:mtmd_input_chunk_type}} -->
- **Type**: `enum`
- **Members**:
    - `MTMD_INPUT_CHUNK_TYPE_TEXT`: Represents a text input chunk type.
    - `MTMD_INPUT_CHUNK_TYPE_IMAGE`: Represents an image input chunk type.
    - `MTMD_INPUT_CHUNK_TYPE_AUDIO`: Represents an audio input chunk type.
- **Description**: The `mtmd_input_chunk_type` is an enumeration that defines the types of input chunks that can be processed by the libmtmd library, which supports multimodal input in the llama.cpp framework. It categorizes input data into three distinct types: text, image, and audio, allowing the library to handle different media formats appropriately.


---
### mtmd\_input\_text<!-- {{#data_structure:mtmd_input_text}} -->
- **Type**: `struct`
- **Members**:
    - `text`: A pointer to a constant character array representing the text input.
    - `add_special`: A boolean indicating whether to add special tokens to the text.
    - `parse_special`: A boolean indicating whether to parse special tokens in the text.
- **Description**: The `mtmd_input_text` structure is designed to represent a text input within the libmtmd library, which supports multimodal operations in llama.cpp. It contains a pointer to the text data and two boolean flags that control the handling of special tokens, allowing for customization of text processing in multimodal contexts.


---
### mtmd\_context<!-- {{#data_structure:mtmd_context}} -->
- **Type**: `struct`
- **Members**:
    - `use_gpu`: Indicates whether GPU is used for processing.
    - `print_timings`: Specifies whether to print timing information.
    - `n_threads`: Defines the number of threads to be used.
    - `verbosity`: Sets the verbosity level for logging.
    - `image_marker`: A deprecated marker for images, replaced by media_marker.
    - `media_marker`: A marker used for media inputs.
- **Description**: The `mtmd_context_params` struct is a configuration structure used to initialize an `mtmd_context` object. It contains parameters that control the behavior of the multimodal context, such as whether to use GPU acceleration, the number of threads to utilize, and the verbosity level for logging. Additionally, it includes markers for media inputs, with `image_marker` being deprecated in favor of `media_marker`. This struct is crucial for setting up the environment in which the multimodal operations will be executed, ensuring that the context is configured according to the user's requirements.
- **Member Functions**:
    - [`mtmd_context::mtmd_context`](mtmd.cpp.driver.md#mtmd_contextmtmd_context)
    - [`mtmd_context::init_vision`](mtmd.cpp.driver.md#mtmd_contextinit_vision)
    - [`mtmd_context::init_audio`](mtmd.cpp.driver.md#mtmd_contextinit_audio)
    - [`mtmd_context::get_clip_ctx`](mtmd.cpp.driver.md#mtmd_contextget_clip_ctx)
    - [`mtmd_context::proj_type_v`](mtmd.cpp.driver.md#mtmd_contextproj_type_v)
    - [`mtmd_context::proj_type_a`](mtmd.cpp.driver.md#mtmd_contextproj_type_a)
    - [`mtmd_context::~mtmd_context`](mtmd.cpp.driver.md#mtmd_contextmtmd_context)
    - [`mtmd_context::lookup_token`](mtmd.cpp.driver.md#mtmd_contextlookup_token)
    - [`mtmd_context::token_to_piece`](mtmd.cpp.driver.md#mtmd_contexttoken_to_piece)


---
### mtmd\_bitmap<!-- {{#data_structure:mtmd_bitmap}} -->
- **Type**: `struct`
- **Description**: The `mtmd_bitmap` is an opaque data structure used within the libmtmd library, which is designed to support multimodal data processing in llama.cpp. This structure is utilized to represent bitmap data, which can be either image or audio data. For images, the data is stored in an RGB format, while for audio, it is stored in a float format. The structure is managed through a series of API functions that allow for initialization, data retrieval, and memory management, but the internal details of the structure are not exposed, as it is defined as an opaque type.


---
### mtmd\_image\_tokens<!-- {{#data_structure:mtmd_image_tokens}} -->
- **Type**: `struct`
- **Members**:
    - `nx`: Represents the number of tokens along the x-axis.
    - `ny`: Represents the number of tokens along the y-axis.
    - `use_mrope_pos`: Indicates whether M-RoPE positional encoding is used.
    - `batch_f32`: Holds a batch of floating-point data, likely related to the image tokens.
    - `id`: A unique identifier for the image tokens.
- **Description**: The `mtmd_image_tokens` struct is part of the libmtmd library, which provides multimodal support for llama.cpp. This struct is used to represent a collection of image tokens, which are likely used in the context of image processing or analysis. The struct includes dimensions `nx` and `ny` to specify the size of the token grid, a boolean `use_mrope_pos` to indicate the use of M-RoPE positional encoding, a `batch_f32` for storing associated floating-point data, and an `id` for uniquely identifying the token set. The struct is constructed via the `mtmd_tokenize` function and is managed alongside `mtmd_input_chunk` instances.
- **Member Functions**:
    - [`mtmd_image_tokens::n_tokens`](mtmd.cpp.driver.md#mtmd_image_tokensn_tokens)
    - [`mtmd_image_tokens::clone`](mtmd.cpp.driver.md#mtmd_image_tokensclone)


---
### mtmd\_input\_chunk<!-- {{#data_structure:mtmd_input_chunk}} -->
- **Type**: `struct`
- **Description**: The `mtmd_input_chunk` is an opaque data structure used within the libmtmd library, which is designed to support multimodal input processing in the llama.cpp framework. This structure represents a single input chunk, which can be of various types such as text, image, or audio, as indicated by the `mtmd_input_chunk_type` enum. The `mtmd_input_chunk` is primarily managed through the library's API functions, which allow for operations such as retrieving the type of the chunk, accessing tokens associated with the chunk, and managing its lifecycle. The actual contents and fields of this structure are not exposed in the provided code, indicating that it is intended to be used through the API rather than directly manipulated by the user.


---
### mtmd\_input\_chunks<!-- {{#data_structure:mtmd_input_chunks}} -->
- **Type**: `struct`
- **Description**: The `mtmd_input_chunks` is an opaque data structure in the `libmtmd` library, which is used to represent a list of `mtmd_input_chunk` elements. These elements are populated through the `mtmd_tokenize()` function, and the structure is primarily managed through the C API functions provided, such as initialization, retrieval of size, access to individual chunks, and deallocation. This structure is integral to handling multimodal input data, such as text, images, and audio, within the library.


---
### mtmd\_context\_params<!-- {{#data_structure:mtmd_context_params}} -->
- **Type**: `struct`
- **Members**:
    - `use_gpu`: A boolean flag indicating whether to use GPU for processing.
    - `print_timings`: A boolean flag indicating whether to print timing information.
    - `n_threads`: An integer specifying the number of threads to use.
    - `verbosity`: An enum value specifying the level of logging verbosity.
    - `image_marker`: A deprecated string pointer for image markers, replaced by media_marker.
    - `media_marker`: A string pointer for media markers used in processing.
- **Description**: The `mtmd_context_params` struct is a configuration structure used to initialize and control the behavior of the MTMD context in the libmtmd library. It includes options for enabling GPU usage, printing timing information, setting the number of threads, and controlling logging verbosity. Additionally, it provides a media marker string for identifying media content within input data, with a deprecated image marker field for backward compatibility.


---
### mtmd\_context\_deleter<!-- {{#data_structure:mtmd::mtmd_context_deleter}} -->
- **Type**: `struct`
- **Description**: The `mtmd_context_deleter` is a C++ struct that defines a custom deleter for managing the lifecycle of `mtmd_context` objects. It overloads the function call operator to invoke `mtmd_free` on a given `mtmd_context` pointer, ensuring proper resource deallocation when used in conjunction with smart pointers like `std::unique_ptr`. This struct is part of a larger library designed to support multimodal operations in the `llama.cpp` framework.
- **Member Functions**:
    - [`mtmd::mtmd_context_deleter::operator()`](#mtmd_context_deleteroperator())

**Methods**

---
#### mtmd\_context\_deleter::operator\(\)<!-- {{#callable:mtmd::mtmd_context_deleter::operator()}} -->
The `operator()` function is a custom deleter for `mtmd_context` objects, which calls [`mtmd_free`](mtmd.cpp.driver.md#mtmd_free) to release the resources associated with the context.
- **Inputs**:
    - `val`: A pointer to an `mtmd_context` object that needs to be freed.
- **Control Flow**:
    - The function takes a pointer to an `mtmd_context` object as its input.
    - It calls the [`mtmd_free`](mtmd.cpp.driver.md#mtmd_free) function, passing the `mtmd_context` pointer to it.
    - The [`mtmd_free`](mtmd.cpp.driver.md#mtmd_free) function is responsible for releasing the resources associated with the `mtmd_context` object.
- **Output**: The function does not return any value; it performs a cleanup operation on the `mtmd_context` object.
- **Functions called**:
    - [`mtmd_free`](mtmd.cpp.driver.md#mtmd_free)
- **See also**: [`mtmd::mtmd_context_deleter`](#mtmdmtmd_context_deleter)  (Data Structure)



---
### mtmd\_bitmap\_deleter<!-- {{#data_structure:mtmd::mtmd_bitmap_deleter}} -->
- **Type**: `struct`
- **Description**: The `mtmd_bitmap_deleter` is a C++ struct designed to serve as a custom deleter for `std::unique_ptr` managing `mtmd_bitmap` objects. It provides an overloaded function call operator that takes a pointer to an `mtmd_bitmap` and calls `mtmd_bitmap_free` to properly release the resources associated with the bitmap. This struct is used in conjunction with `std::unique_ptr` to ensure that `mtmd_bitmap` objects are automatically and safely deallocated when they go out of scope, preventing memory leaks.
- **Member Functions**:
    - [`mtmd::mtmd_bitmap_deleter::operator()`](#mtmd_bitmap_deleteroperator())

**Methods**

---
#### mtmd\_bitmap\_deleter::operator\(\)<!-- {{#callable:mtmd::mtmd_bitmap_deleter::operator()}} -->
The `operator()` function in the `mtmd_bitmap_deleter` struct is a functor that frees a given `mtmd_bitmap` object using the [`mtmd_bitmap_free`](mtmd.cpp.driver.md#mtmd_bitmap_free) function.
- **Inputs**:
    - `val`: A pointer to an `mtmd_bitmap` object that needs to be freed.
- **Control Flow**:
    - The function takes a pointer to an `mtmd_bitmap` object as its input.
    - It calls the [`mtmd_bitmap_free`](mtmd.cpp.driver.md#mtmd_bitmap_free) function, passing the `mtmd_bitmap` pointer to it.
    - The [`mtmd_bitmap_free`](mtmd.cpp.driver.md#mtmd_bitmap_free) function is responsible for deallocating the memory associated with the `mtmd_bitmap` object.
- **Output**: The function does not return any value; it performs a side effect by freeing the memory of the `mtmd_bitmap` object.
- **Functions called**:
    - [`mtmd_bitmap_free`](mtmd.cpp.driver.md#mtmd_bitmap_free)
- **See also**: [`mtmd::mtmd_bitmap_deleter`](#mtmdmtmd_bitmap_deleter)  (Data Structure)



---
### mtmd\_input\_chunks\_deleter<!-- {{#data_structure:mtmd::mtmd_input_chunks_deleter}} -->
- **Type**: `struct`
- **Description**: The `mtmd_input_chunks_deleter` is a C++ struct that defines a custom deleter for the `mtmd_input_chunks` type, which is used in conjunction with `std::unique_ptr` to manage the memory of `mtmd_input_chunks` objects. It provides an overloaded `operator()` that calls the `mtmd_input_chunks_free` function to properly release the resources associated with an `mtmd_input_chunks` instance, ensuring that memory is managed safely and efficiently.
- **Member Functions**:
    - [`mtmd::mtmd_input_chunks_deleter::operator()`](#mtmd_input_chunks_deleteroperator())

**Methods**

---
#### mtmd\_input\_chunks\_deleter::operator\(\)<!-- {{#callable:mtmd::mtmd_input_chunks_deleter::operator()}} -->
The `operator()` function in the `mtmd_input_chunks_deleter` struct is a custom deleter that frees memory associated with `mtmd_input_chunks` objects.
- **Inputs**:
    - `val`: A pointer to an `mtmd_input_chunks` object that needs to be freed.
- **Control Flow**:
    - The function takes a pointer to an `mtmd_input_chunks` object as its argument.
    - It calls the [`mtmd_input_chunks_free`](mtmd.cpp.driver.md#mtmd_input_chunks_free) function, passing the pointer to free the associated memory.
- **Output**: This function does not return any value; it performs a side effect by freeing memory.
- **Functions called**:
    - [`mtmd_input_chunks_free`](mtmd.cpp.driver.md#mtmd_input_chunks_free)
- **See also**: [`mtmd::mtmd_input_chunks_deleter`](#mtmdmtmd_input_chunks_deleter)  (Data Structure)



---
### mtmd\_input\_chunk\_deleter<!-- {{#data_structure:mtmd::mtmd_input_chunk_deleter}} -->
- **Type**: `struct`
- **Description**: The `mtmd_input_chunk_deleter` is a C++ struct that defines a custom deleter for `mtmd_input_chunk` objects. It overloads the function call operator to call `mtmd_input_chunk_free` on a given `mtmd_input_chunk` pointer, ensuring proper resource management and cleanup when used with smart pointers like `std::unique_ptr`. This struct is part of a larger library designed to handle multimodal data in the context of the `llama.cpp` project.
- **Member Functions**:
    - [`mtmd::mtmd_input_chunk_deleter::operator()`](#mtmd_input_chunk_deleteroperator())

**Methods**

---
#### mtmd\_input\_chunk\_deleter::operator\(\)<!-- {{#callable:mtmd::mtmd_input_chunk_deleter::operator()}} -->
The `operator()` function in the `mtmd_input_chunk_deleter` struct is a functor that frees a given `mtmd_input_chunk` object using the [`mtmd_input_chunk_free`](mtmd.cpp.driver.md#mtmd_input_chunk_free) function.
- **Inputs**:
    - `val`: A pointer to an `mtmd_input_chunk` object that needs to be freed.
- **Control Flow**:
    - The function takes a pointer to an `mtmd_input_chunk` object as its argument.
    - It calls the [`mtmd_input_chunk_free`](mtmd.cpp.driver.md#mtmd_input_chunk_free) function, passing the pointer as an argument to free the associated resources.
- **Output**: This function does not return any value; it performs a cleanup operation on the input `mtmd_input_chunk` object.
- **Functions called**:
    - [`mtmd_input_chunk_free`](mtmd.cpp.driver.md#mtmd_input_chunk_free)
- **See also**: [`mtmd::mtmd_input_chunk_deleter`](#mtmdmtmd_input_chunk_deleter)  (Data Structure)



---
### bitmap<!-- {{#data_structure:mtmd::bitmap}} -->
- **Type**: `struct`
- **Members**:
    - `ptr`: A unique pointer to an mtmd_bitmap object, managing its lifetime.
- **Description**: The `bitmap` struct is a C++ wrapper around the `mtmd_bitmap` type, providing a managed interface for handling bitmap data, which can represent either image or audio data. It encapsulates a unique pointer (`bitmap_ptr`) to an `mtmd_bitmap` object, ensuring proper resource management and memory deallocation. The struct offers constructors for initializing the bitmap with dimensions and data, as well as move semantics for efficient resource transfer. It provides member functions to access bitmap properties such as dimensions (`nx`, `ny`), data, byte size, and an optional identifier (`id`).
- **Member Functions**:
    - [`mtmd::bitmap::bitmap`](#bitmapbitmap)
    - [`mtmd::bitmap::bitmap`](#bitmapbitmap)
    - [`mtmd::bitmap::bitmap`](#bitmapbitmap)
    - [`mtmd::bitmap::bitmap`](#bitmapbitmap)
    - [`mtmd::bitmap::~bitmap`](#bitmapbitmap)
    - [`mtmd::bitmap::nx`](#bitmapnx)
    - [`mtmd::bitmap::ny`](#bitmapny)
    - [`mtmd::bitmap::data`](#bitmapdata)
    - [`mtmd::bitmap::n_bytes`](#bitmapn_bytes)
    - [`mtmd::bitmap::id`](#bitmapid)
    - [`mtmd::bitmap::set_id`](#bitmapset_id)

**Methods**

---
#### bitmap::bitmap<!-- {{#callable:mtmd::bitmap::bitmap}} -->
The `bitmap` constructor initializes a `bitmap` object with either a null pointer or a given `mtmd_bitmap` pointer.
- **Inputs**:
    - `bitmap`: A pointer to an `mtmd_bitmap` object, which is used to initialize the `ptr` member of the `bitmap` object.
- **Control Flow**:
    - The default constructor initializes the `ptr` member to `nullptr`.
    - The parameterized constructor initializes the `ptr` member with the provided `mtmd_bitmap` pointer.
- **Output**: A `bitmap` object with its `ptr` member initialized to either `nullptr` or the provided `mtmd_bitmap` pointer.
- **See also**: [`mtmd::bitmap`](#mtmdbitmap)  (Data Structure)


---
#### bitmap::bitmap<!-- {{#callable:mtmd::bitmap::bitmap}} -->
The `bitmap` constructor initializes a `bitmap` object by transferring ownership of a `mtmd_bitmap` pointer or by moving another `bitmap` object.
- **Inputs**:
    - `bitmap`: A pointer to an `mtmd_bitmap` object used to initialize the `bitmap` object.
    - `other`: An rvalue reference to another `bitmap` object whose `mtmd_bitmap` pointer is moved to the new `bitmap` object.
- **Control Flow**:
    - The constructor `bitmap(mtmd_bitmap * bitmap)` initializes the `ptr` member with the provided `mtmd_bitmap` pointer.
    - The move constructor `bitmap(bitmap && other) noexcept` transfers ownership of the `mtmd_bitmap` pointer from `other` to the new `bitmap` object using `std::move`.
- **Output**: A `bitmap` object is constructed with its `ptr` member initialized to point to the given `mtmd_bitmap` or moved from another `bitmap` object.
- **See also**: [`mtmd::bitmap`](#mtmdbitmap)  (Data Structure)


---
#### bitmap::bitmap<!-- {{#callable:mtmd::bitmap::bitmap}} -->
The `bitmap` constructor initializes a `bitmap` object either by moving an existing `bitmap` object or by creating a new `bitmap` from given dimensions and data.
- **Inputs**:
    - `other`: An rvalue reference to another `bitmap` object, used for move construction.
    - `nx`: A `uint32_t` representing the width of the bitmap in pixels.
    - `ny`: A `uint32_t` representing the height of the bitmap in pixels.
    - `data`: A pointer to an array of unsigned char representing the bitmap data, expected to be in RGB format for images.
- **Control Flow**:
    - The move constructor initializes the `ptr` member by moving the `ptr` from the `other` bitmap object, effectively transferring ownership.
    - The constructor with parameters `nx`, `ny`, and `data` calls `mtmd_bitmap_init` to initialize a new `mtmd_bitmap` with the given dimensions and data, and assigns the result to `ptr`.
- **Output**: A `bitmap` object is constructed, either by moving an existing object or by creating a new one with specified dimensions and data.
- **See also**: [`mtmd::bitmap`](#mtmdbitmap)  (Data Structure)


---
#### bitmap::bitmap<!-- {{#callable:mtmd::bitmap::bitmap}} -->
The `bitmap` constructor initializes a `bitmap` object by creating a new `mtmd_bitmap` with specified dimensions and data, and manages its memory using a smart pointer.
- **Inputs**:
    - `nx`: The width of the bitmap in pixels, represented as a 32-bit unsigned integer.
    - `ny`: The height of the bitmap in pixels, represented as a 32-bit unsigned integer.
    - `data`: A pointer to an array of unsigned char representing the bitmap data, expected to be in RGB format if the bitmap is an image.
- **Control Flow**:
    - The constructor is called with parameters `nx`, `ny`, and `data`.
    - It calls the function `mtmd_bitmap_init` with these parameters to create a new `mtmd_bitmap` object.
    - The resulting `mtmd_bitmap` pointer is then managed by a `std::unique_ptr` named `ptr`, which is reset to own the newly created `mtmd_bitmap`.
- **Output**: The function does not return a value; it initializes the `bitmap` object with a managed `mtmd_bitmap`.
- **See also**: [`mtmd::bitmap`](#mtmdbitmap)  (Data Structure)


---
#### bitmap::\~bitmap<!-- {{#callable:mtmd::bitmap::~bitmap}} -->
The `~bitmap` function is the default destructor for the `bitmap` class, which automatically handles the cleanup of resources when a `bitmap` object is destroyed.
- **Inputs**: None
- **Control Flow**:
    - The `~bitmap` function is defined as `default`, indicating that the compiler will generate the default destructor implementation.
    - The destructor will automatically be called when a `bitmap` object goes out of scope or is explicitly deleted, ensuring that the `bitmap_ptr` resource is properly released.
- **Output**: The function does not return any value as it is a destructor.
- **See also**: [`mtmd::bitmap`](#mtmdbitmap)  (Data Structure)


---
#### bitmap::nx<!-- {{#callable:mtmd::bitmap::nx}} -->
The `nx` function retrieves the width (number of columns) of a bitmap image stored in the `bitmap` class.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`mtmd_bitmap_get_nx`](mtmd.cpp.driver.md#mtmd_bitmap_get_nx) with the `mtmd_bitmap` pointer obtained from `ptr.get()`.
    - The result of the call is returned as the output of the function.
- **Output**: The function returns a `uint32_t` representing the width (number of columns) of the bitmap image.
- **Functions called**:
    - [`mtmd_bitmap_get_nx`](mtmd.cpp.driver.md#mtmd_bitmap_get_nx)
- **See also**: [`mtmd::bitmap`](#mtmdbitmap)  (Data Structure)


---
#### bitmap::ny<!-- {{#callable:mtmd::bitmap::ny}} -->
The `ny` function retrieves the number of rows (ny) of a bitmap image from a `mtmd_bitmap` object.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`mtmd_bitmap_get_ny`](mtmd.cpp.driver.md#mtmd_bitmap_get_ny) with the `mtmd_bitmap` pointer obtained from `ptr.get()`.
    - It returns the result of the [`mtmd_bitmap_get_ny`](mtmd.cpp.driver.md#mtmd_bitmap_get_ny) function call.
- **Output**: The function returns a `uint32_t` representing the number of rows (ny) in the bitmap.
- **Functions called**:
    - [`mtmd_bitmap_get_ny`](mtmd.cpp.driver.md#mtmd_bitmap_get_ny)
- **See also**: [`mtmd::bitmap`](#mtmdbitmap)  (Data Structure)


---
#### bitmap::data<!-- {{#callable:mtmd::bitmap::data}} -->
The `data` function retrieves the raw data of a bitmap as an array of unsigned characters.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`mtmd_bitmap_get_data`](mtmd.cpp.driver.md#mtmd_bitmap_get_data) with the internal pointer to the bitmap object.
    - It returns the result of the [`mtmd_bitmap_get_data`](mtmd.cpp.driver.md#mtmd_bitmap_get_data) function call.
- **Output**: A pointer to an array of unsigned characters representing the bitmap data.
- **Functions called**:
    - [`mtmd_bitmap_get_data`](mtmd.cpp.driver.md#mtmd_bitmap_get_data)
- **See also**: [`mtmd::bitmap`](#mtmdbitmap)  (Data Structure)


---
#### bitmap::n\_bytes<!-- {{#callable:mtmd::bitmap::n_bytes}} -->
The `n_bytes` function returns the number of bytes used by the bitmap data.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`mtmd_bitmap_get_n_bytes`](mtmd.cpp.driver.md#mtmd_bitmap_get_n_bytes) with the bitmap pointer obtained from `ptr.get()`.
    - The result of the call is returned as the output of the function.
- **Output**: The function returns a `size_t` value representing the number of bytes used by the bitmap data.
- **Functions called**:
    - [`mtmd_bitmap_get_n_bytes`](mtmd.cpp.driver.md#mtmd_bitmap_get_n_bytes)
- **See also**: [`mtmd::bitmap`](#mtmdbitmap)  (Data Structure)


---
#### bitmap::id<!-- {{#callable:mtmd::bitmap::id}} -->
The `id` function retrieves the unique identifier of a bitmap object.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`mtmd_bitmap_get_id`](mtmd.cpp.driver.md#mtmd_bitmap_get_id) with the bitmap pointer obtained from `ptr.get()`.
    - It returns the result of the [`mtmd_bitmap_get_id`](mtmd.cpp.driver.md#mtmd_bitmap_get_id) function call, which is a string representing the bitmap's ID.
- **Output**: A `std::string` containing the ID of the bitmap.
- **Functions called**:
    - [`mtmd_bitmap_get_id`](mtmd.cpp.driver.md#mtmd_bitmap_get_id)
- **See also**: [`mtmd::bitmap`](#mtmdbitmap)  (Data Structure)


---
#### bitmap::set\_id<!-- {{#callable:mtmd::bitmap::set_id}} -->
The `set_id` function assigns a new identifier to a bitmap object by calling an external function to set the ID on the underlying bitmap pointer.
- **Inputs**:
    - `id`: A constant character pointer representing the new identifier to be set for the bitmap.
- **Control Flow**:
    - The function takes a constant character pointer `id` as an argument.
    - It calls the [`mtmd_bitmap_set_id`](mtmd.cpp.driver.md#mtmd_bitmap_set_id) function, passing the internal bitmap pointer and the `id` to set the new identifier.
- **Output**: This function does not return any value.
- **Functions called**:
    - [`mtmd_bitmap_set_id`](mtmd.cpp.driver.md#mtmd_bitmap_set_id)
- **See also**: [`mtmd::bitmap`](#mtmdbitmap)  (Data Structure)



---
### bitmaps<!-- {{#data_structure:mtmd::bitmaps}} -->
- **Type**: `struct`
- **Members**:
    - `entries`: A vector that holds multiple bitmap objects.
- **Description**: The `bitmaps` struct is a container for managing a collection of `bitmap` objects, which are likely used to represent image or audio data in a multimodal context. It provides a method `c_ptr()` to retrieve a list of raw pointers to the underlying `mtmd_bitmap` objects, facilitating their use in functions that require direct access to these pointers, such as the `mtmd_tokenize` function.
- **Member Functions**:
    - [`mtmd::bitmaps::~bitmaps`](#bitmapsbitmaps)
    - [`mtmd::bitmaps::c_ptr`](#bitmapsc_ptr)

**Methods**

---
#### bitmaps::\~bitmaps<!-- {{#callable:mtmd::bitmaps::~bitmaps}} -->
The destructor `~bitmaps()` is a default destructor for the `bitmaps` struct, which automatically handles cleanup of its resources.
- **Inputs**: None
- **Control Flow**:
    - The destructor `~bitmaps()` is defined as `default`, meaning it relies on the compiler-generated destructor to handle resource cleanup.
    - Since `bitmaps` contains a `std::vector` of `bitmap` objects, the default destructor will automatically call the destructors of each `bitmap` object in the vector.
- **Output**: There is no explicit output from the destructor as it is responsible for cleanup and resource deallocation.
- **See also**: [`mtmd::bitmaps`](#mtmdbitmaps)  (Data Structure)


---
#### bitmaps::c\_ptr<!-- {{#callable:mtmd::bitmaps::c_ptr}} -->
The `c_ptr` function returns a vector of raw pointers to `mtmd_bitmap` objects stored in the `entries` vector of `bitmap` objects.
- **Inputs**: None
- **Control Flow**:
    - Initialize a vector `res` of type `std::vector<const mtmd_bitmap *>` with the same size as `entries`.
    - Iterate over each `bitmap` object in the `entries` vector.
    - For each `bitmap`, retrieve the raw pointer to the `mtmd_bitmap` using `ptr.get()` and store it in the corresponding position in `res`.
    - Return the vector `res` containing the raw pointers.
- **Output**: A `std::vector` containing raw pointers to `mtmd_bitmap` objects.
- **See also**: [`mtmd::bitmaps`](#mtmdbitmaps)  (Data Structure)



---
### input\_chunks<!-- {{#data_structure:mtmd::input_chunks}} -->
- **Type**: `struct`
- **Members**:
    - `ptr`: A unique pointer to an mtmd_input_chunks object, managing its lifetime.
- **Description**: The `input_chunks` struct is a C++ wrapper around the `mtmd_input_chunks` structure, providing RAII-style management of the underlying C structure using a unique pointer. It offers a default constructor, a constructor that initializes the pointer with a given `mtmd_input_chunks` object, and a destructor. The struct provides methods to access the size of the input chunks and to retrieve a specific input chunk by index, facilitating interaction with the underlying multimodal input data.
- **Member Functions**:
    - [`mtmd::input_chunks::input_chunks`](#input_chunksinput_chunks)
    - [`mtmd::input_chunks::input_chunks`](#input_chunksinput_chunks)
    - [`mtmd::input_chunks::~input_chunks`](#input_chunksinput_chunks)
    - [`mtmd::input_chunks::size`](#input_chunkssize)
    - [`mtmd::input_chunks::operator[]`](llama.cpp/tools/mtmd/mtmd.h#callable:mtmd::input_chunks::operator[])

**Methods**

---
#### input\_chunks::input\_chunks<!-- {{#callable:mtmd::input_chunks::input_chunks}} -->
The `input_chunks` constructor initializes an `input_chunks` object with an optional pointer to `mtmd_input_chunks`.
- **Inputs**:
    - `chunks`: A pointer to an `mtmd_input_chunks` object, which is used to initialize the `ptr` member of the `input_chunks` object.
- **Control Flow**:
    - The default constructor `input_chunks()` initializes an `input_chunks` object with default values.
    - The parameterized constructor `input_chunks(mtmd_input_chunks * chunks)` initializes the `ptr` member with the provided `chunks` pointer.
- **Output**: An `input_chunks` object is constructed, optionally initialized with a pointer to `mtmd_input_chunks`.
- **See also**: [`mtmd::input_chunks`](#mtmdinput_chunks)  (Data Structure)


---
#### input\_chunks::input\_chunks<!-- {{#callable:mtmd::input_chunks::input_chunks}} -->
The `input_chunks` constructor initializes an `input_chunks` object with a pointer to `mtmd_input_chunks`.
- **Inputs**:
    - `chunks`: A pointer to an `mtmd_input_chunks` object, which is used to initialize the `ptr` member of the `input_chunks` object.
- **Control Flow**:
    - The constructor takes a pointer to `mtmd_input_chunks` as an argument.
    - It initializes the `ptr` member of the `input_chunks` object with the provided `chunks` pointer.
- **Output**: An `input_chunks` object initialized with the given `mtmd_input_chunks` pointer.
- **See also**: [`mtmd::input_chunks`](#mtmdinput_chunks)  (Data Structure)


---
#### input\_chunks::\~input\_chunks<!-- {{#callable:mtmd::input_chunks::~input_chunks}} -->
The destructor `~input_chunks` is a default destructor for the `input_chunks` class, which automatically handles the cleanup of resources when an `input_chunks` object is destroyed.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as `default`, meaning it relies on the compiler-generated destructor to handle resource cleanup.
    - The `input_chunks` class uses a smart pointer `input_chunks_ptr` to manage the `mtmd_input_chunks` resource, ensuring automatic deallocation when the object goes out of scope.
- **Output**: There is no explicit output from the destructor, as it is responsible for resource cleanup.
- **See also**: [`mtmd::input_chunks`](#mtmdinput_chunks)  (Data Structure)


---
#### input\_chunks::size<!-- {{#callable:mtmd::input_chunks::size}} -->
The `size` function returns the number of input chunks in the `input_chunks` structure.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`mtmd_input_chunks_size`](mtmd.cpp.driver.md#mtmd_input_chunks_size) with the pointer to the `mtmd_input_chunks` structure obtained from `ptr.get()`.
    - The result of [`mtmd_input_chunks_size`](mtmd.cpp.driver.md#mtmd_input_chunks_size) is returned as the output of the function.
- **Output**: The function returns a `size_t` value representing the number of input chunks.
- **Functions called**:
    - [`mtmd_input_chunks_size`](mtmd.cpp.driver.md#mtmd_input_chunks_size)
- **See also**: [`mtmd::input_chunks`](#mtmdinput_chunks)  (Data Structure)


---
#### input\_chunks::operator\[\]<!-- {{#callable:mtmd::input_chunks::operator[]}} -->
The `operator[]` function retrieves a pointer to an `mtmd_input_chunk` at a specified index from an `input_chunks` object.
- **Inputs**:
    - `idx`: The index of the `mtmd_input_chunk` to retrieve from the `input_chunks` object.
- **Control Flow**:
    - The function calls `mtmd_input_chunks_get` with the internal pointer to `mtmd_input_chunks` and the provided index `idx`.
    - It returns the result of `mtmd_input_chunks_get`, which is a pointer to the `mtmd_input_chunk` at the specified index.
- **Output**: A pointer to the `mtmd_input_chunk` at the specified index within the `input_chunks` object.
- **See also**: [`mtmd::input_chunks`](#mtmdinput_chunks)  (Data Structure)



