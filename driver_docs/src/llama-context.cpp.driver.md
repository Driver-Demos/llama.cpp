# Purpose
The provided C++ source code defines the implementation of a class named [`llama_context`](#llama_contextllama_context), which is part of a larger software system for managing and processing machine learning models, specifically those related to the "LLaMA" architecture. This file is not a standalone executable but rather a component of a larger library, as indicated by the inclusion of multiple header files and the absence of a `main` function. The primary purpose of this code is to manage the context in which a LLaMA model operates, including initialization, configuration, and execution of model-related tasks such as encoding and decoding sequences.

The [`llama_context`](#llama_contextllama_context) class is responsible for setting up and managing the execution environment for a LLaMA model. It handles various parameters and configurations, such as the number of threads, batch sizes, and memory management. The class also provides methods for encoding and decoding sequences, managing memory states, and interfacing with different computational backends. It includes functionality for logging, error handling, and performance tracking, which are crucial for debugging and optimizing the model's execution. Additionally, the class supports saving and loading the state of the model and its context, which is essential for tasks like checkpointing and resuming training or inference. Overall, this code provides a comprehensive framework for managing the lifecycle and execution of LLaMA models within a larger machine learning system.
# Imports and Dependencies

---
- `llama-context.h`
- `llama-impl.h`
- `llama-io.h`
- `llama-memory.h`
- `llama-mmap.h`
- `llama-model.h`
- `cinttypes`
- `cstring`
- `limits`
- `stdexcept`


# Data Structures

---
### llama\_io\_write\_dummy<!-- {{#data_structure:llama_io_write_dummy}} -->
- **Type**: `class`
- **Members**:
    - `size_written`: Tracks the total number of bytes written.
- **Description**: The `llama_io_write_dummy` class is a dummy implementation of the `llama_io_write_i` interface, designed to simulate writing operations without actually performing any I/O. It keeps track of the total number of bytes that would have been written through its `size_written` member. This class is useful for scenarios where you need to measure or simulate the amount of data being written without affecting any actual storage or file systems.
- **Member Functions**:
    - [`llama_io_write_dummy::llama_io_write_dummy`](#llama_io_write_dummyllama_io_write_dummy)
    - [`llama_io_write_dummy::write`](#llama_io_write_dummywrite)
    - [`llama_io_write_dummy::write_tensor`](#llama_io_write_dummywrite_tensor)
    - [`llama_io_write_dummy::n_bytes`](#llama_io_write_dummyn_bytes)
- **Inherits From**:
    - [`llama_io_write_i::llama_io_write_i`](llama-io.h.driver.md#llama_io_write_illama_io_write_i)

**Methods**

---
#### llama\_io\_write\_dummy::llama\_io\_write\_dummy<!-- {{#callable:llama_io_write_dummy::llama_io_write_dummy}} -->
The `llama_io_write_dummy` class is a dummy implementation of the `llama_io_write_i` interface that tracks the number of bytes written without actually storing any data.
- **Inputs**: None
- **Control Flow**:
    - The constructor `llama_io_write_dummy()` is defined as default, meaning it does not perform any specific initialization.
    - The `write` method increments the `size_written` member by the size of the data that is supposedly written.
    - The `write_tensor` method also increments the `size_written` member by the size of the tensor data that is supposedly written.
    - The `n_bytes` method returns the total number of bytes that have been 'written' as tracked by `size_written`.
- **Output**: The output is the total number of bytes 'written', as tracked by the `size_written` member variable.
- **See also**: [`llama_io_write_dummy`](#llama_io_write_dummy)  (Data Structure)


---
#### llama\_io\_write\_dummy::write<!-- {{#callable:llama_io_write_dummy::write}} -->
The `write` function updates the `size_written` member variable by adding the given `size` to it.
- **Inputs**:
    - `src`: A pointer to the source data to be written; however, it is not used in this function.
    - `size`: The size of the data to be written, which is added to the `size_written` member variable.
- **Control Flow**:
    - The function takes two parameters: a pointer `src` and a `size` value.
    - The `src` parameter is not used in the function body.
    - The `size` parameter is added to the `size_written` member variable, updating its value.
- **Output**: The function does not return any value; it modifies the `size_written` member variable of the class.
- **See also**: [`llama_io_write_dummy`](#llama_io_write_dummy)  (Data Structure)


---
#### llama\_io\_write\_dummy::write\_tensor<!-- {{#callable:llama_io_write_dummy::write_tensor}} -->
The `write_tensor` function updates the `size_written` member variable by adding the given `size` to it.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` object, which is not used in the function.
    - `offset`: A `size_t` value representing the offset, which is not used in the function.
    - `size`: A `size_t` value representing the size to be added to `size_written`.
- **Control Flow**:
    - The function takes three parameters: a tensor pointer, an offset, and a size.
    - The function does not use the tensor pointer or the offset.
    - The function adds the `size` parameter to the `size_written` member variable.
- **Output**: The function does not return any value; it modifies the `size_written` member variable of the class.
- **See also**: [`llama_io_write_dummy`](#llama_io_write_dummy)  (Data Structure)


---
#### llama\_io\_write\_dummy::n\_bytes<!-- {{#callable:llama_io_write_dummy::n_bytes}} -->
The `n_bytes` function returns the total number of bytes written by the `llama_io_write_dummy` class.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the value of the `size_written` member variable.
- **Output**: The function outputs a `size_t` value representing the total number of bytes written.
- **See also**: [`llama_io_write_dummy`](#llama_io_write_dummy)  (Data Structure)



---
### llama\_io\_write\_buffer<!-- {{#data_structure:llama_io_write_buffer}} -->
- **Type**: `class`
- **Members**:
    - `ptr`: A pointer to the current position in the buffer where data will be written.
    - `buf_size`: The remaining size of the buffer available for writing.
    - `size_written`: The total number of bytes that have been written to the buffer.
- **Description**: The `llama_io_write_buffer` class is a specialized implementation of the `llama_io_write_i` interface, designed to manage writing operations to a buffer in memory. It maintains a pointer to the current position in the buffer (`ptr`), tracks the available buffer size (`buf_size`), and records the total number of bytes written (`size_written`). The class provides methods to write raw data and tensor data to the buffer, ensuring that writes do not exceed the buffer's capacity, throwing an exception if an overflow is attempted.
- **Member Functions**:
    - [`llama_io_write_buffer::llama_io_write_buffer`](#llama_io_write_bufferllama_io_write_buffer)
    - [`llama_io_write_buffer::write`](#llama_io_write_bufferwrite)
    - [`llama_io_write_buffer::write_tensor`](#llama_io_write_bufferwrite_tensor)
    - [`llama_io_write_buffer::n_bytes`](#llama_io_write_buffern_bytes)
- **Inherits From**:
    - [`llama_io_write_i::llama_io_write_i`](llama-io.h.driver.md#llama_io_write_illama_io_write_i)

**Methods**

---
#### llama\_io\_write\_buffer::llama\_io\_write\_buffer<!-- {{#callable:llama_io_write_buffer::llama_io_write_buffer}} -->
The `llama_io_write_buffer` constructor initializes a buffer for writing data with a specified pointer and buffer size.
- **Inputs**:
    - `p`: A pointer to a uint8_t array where data will be written.
    - `len`: The size of the buffer in bytes.
- **Control Flow**:
    - The constructor initializes the `ptr` member with the provided pointer `p`.
    - The constructor initializes the `buf_size` member with the provided size `len`.
- **Output**: This constructor does not return any value; it initializes the object state.
- **See also**: [`llama_io_write_buffer`](#llama_io_write_buffer)  (Data Structure)


---
#### llama\_io\_write\_buffer::write<!-- {{#callable:llama_io_write_buffer::write}} -->
The `write` function writes data from a source buffer to a destination buffer, updating the buffer pointers and sizes accordingly.
- **Inputs**:
    - `src`: A pointer to the source buffer containing the data to be written.
    - `size`: The number of bytes to write from the source buffer to the destination buffer.
- **Control Flow**:
    - Check if the size of data to be written exceeds the available buffer size; if so, throw a runtime error.
    - Copy the specified number of bytes from the source buffer to the destination buffer using `memcpy`.
    - Advance the destination buffer pointer by the number of bytes written.
    - Update the total number of bytes written by adding the size of the current write operation.
    - Decrease the available buffer size by the number of bytes written.
- **Output**: This function does not return a value; it modifies the internal state of the buffer by updating the pointer and size variables.
- **See also**: [`llama_io_write_buffer`](#llama_io_write_buffer)  (Data Structure)


---
#### llama\_io\_write\_buffer::write\_tensor<!-- {{#callable:llama_io_write_buffer::write_tensor}} -->
The `write_tensor` function writes a portion of a tensor to a buffer, updating the buffer's pointer and size tracking variables.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` object, representing the tensor to be written to the buffer.
    - `offset`: A `size_t` value indicating the starting point within the tensor from which data should be written.
    - `size`: A `size_t` value specifying the number of bytes to write from the tensor to the buffer.
- **Control Flow**:
    - Check if the `size` to be written exceeds the available buffer size (`buf_size`).
    - If `size` exceeds `buf_size`, throw a `std::runtime_error` indicating the buffer has been exceeded.
    - Call [`ggml_backend_tensor_get`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get) to write the specified portion of the tensor to the buffer starting at `ptr`.
    - Increment the buffer pointer `ptr` by `size` to reflect the new position after writing.
    - Update `size_written` by adding `size` to track the total number of bytes written.
    - Decrease `buf_size` by `size` to reflect the remaining buffer capacity.
- **Output**: The function does not return a value; it modifies the buffer state by writing data and updating internal tracking variables.
- **Functions called**:
    - [`ggml_backend_tensor_get`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get)
- **See also**: [`llama_io_write_buffer`](#llama_io_write_buffer)  (Data Structure)


---
#### llama\_io\_write\_buffer::n\_bytes<!-- {{#callable:llama_io_write_buffer::n_bytes}} -->
The `n_bytes` function returns the total number of bytes written to the buffer.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the value of the `size_written` member variable.
- **Output**: The function returns a `size_t` value representing the total number of bytes written to the buffer.
- **See also**: [`llama_io_write_buffer`](#llama_io_write_buffer)  (Data Structure)



---
### llama\_io\_read\_buffer<!-- {{#data_structure:llama_io_read_buffer}} -->
- **Type**: `class`
- **Members**:
    - `ptr`: A pointer to the current position in the buffer.
    - `buf_size`: The remaining size of the buffer.
    - `size_read`: The total number of bytes read from the buffer.
- **Description**: The `llama_io_read_buffer` class is a specialized implementation of the `llama_io_read_i` interface, designed to facilitate reading operations from a buffer in memory. It maintains a pointer to the current position in the buffer and tracks the remaining size of the buffer as well as the total number of bytes read. This class provides methods to read a specified number of bytes from the buffer and to copy data directly to a destination, throwing an exception if the read operation exceeds the buffer's size.
- **Member Functions**:
    - [`llama_io_read_buffer::llama_io_read_buffer`](#llama_io_read_bufferllama_io_read_buffer)
    - [`llama_io_read_buffer::read`](#llama_io_read_bufferread)
    - [`llama_io_read_buffer::read_to`](#llama_io_read_bufferread_to)
    - [`llama_io_read_buffer::n_bytes`](#llama_io_read_buffern_bytes)
- **Inherits From**:
    - [`llama_io_read_i::llama_io_read_i`](llama-io.h.driver.md#llama_io_read_illama_io_read_i)

**Methods**

---
#### llama\_io\_read\_buffer::llama\_io\_read\_buffer<!-- {{#callable:llama_io_read_buffer::llama_io_read_buffer}} -->
The `llama_io_read_buffer` constructor initializes a buffer reader with a pointer to a data buffer and its size.
- **Inputs**:
    - `p`: A pointer to a constant uint8_t array representing the data buffer to be read.
    - `len`: A size_t value representing the size of the data buffer.
- **Control Flow**:
    - The constructor initializes the `ptr` member with the provided pointer `p`.
    - The constructor initializes the `buf_size` member with the provided size `len`.
- **Output**: This constructor does not return any value as it is used to initialize an object of the `llama_io_read_buffer` class.
- **See also**: [`llama_io_read_buffer`](#llama_io_read_buffer)  (Data Structure)


---
#### llama\_io\_read\_buffer::read<!-- {{#callable:llama_io_read_buffer::read}} -->
The `read` function reads a specified number of bytes from a buffer and updates the buffer's state accordingly.
- **Inputs**:
    - `size`: The number of bytes to read from the buffer.
- **Control Flow**:
    - Initialize `base_ptr` to the current position of the buffer pointer `ptr`.
    - Check if the requested `size` exceeds the available `buf_size`; if so, throw a runtime error indicating the end of the buffer has been reached.
    - Increment the buffer pointer `ptr` by `size` to reflect the bytes read.
    - Update `size_read` by adding `size` to it, indicating the total number of bytes read so far.
    - Decrease `buf_size` by `size` to reflect the remaining buffer size.
    - Return the `base_ptr`, which points to the start of the read data.
- **Output**: Returns a pointer to the start of the read data in the buffer.
- **See also**: [`llama_io_read_buffer`](#llama_io_read_buffer)  (Data Structure)


---
#### llama\_io\_read\_buffer::read\_to<!-- {{#callable:llama_io_read_buffer::read_to}} -->
The `read_to` function copies a specified number of bytes from a buffer to a destination memory location.
- **Inputs**:
    - `dst`: A pointer to the destination memory location where the data will be copied.
    - `size`: The number of bytes to read and copy from the buffer to the destination.
- **Control Flow**:
    - The function calls the [`read`](#llama_io_read_bufferread) method with the specified size to get a pointer to the data to be copied.
    - It then uses `memcpy` to copy the data from the buffer to the destination memory location.
- **Output**: The function does not return any value; it performs the copy operation as a side effect.
- **Functions called**:
    - [`llama_io_read_buffer::read`](#llama_io_read_bufferread)
- **See also**: [`llama_io_read_buffer`](#llama_io_read_buffer)  (Data Structure)


---
#### llama\_io\_read\_buffer::n\_bytes<!-- {{#callable:llama_io_read_buffer::n_bytes}} -->
The `n_bytes` function returns the total number of bytes read from the buffer.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the `size_read` member variable, which tracks the number of bytes read from the buffer.
- **Output**: The function returns a `size_t` value representing the total number of bytes read from the buffer.
- **See also**: [`llama_io_read_buffer`](#llama_io_read_buffer)  (Data Structure)



---
### llama\_io\_write\_file<!-- {{#data_structure:llama_io_write_file}} -->
- **Type**: `class`
- **Members**:
    - `file`: A pointer to a `llama_file` object used for file operations.
    - `size_written`: A size_t variable that tracks the total number of bytes written.
    - `temp_buffer`: A vector of uint8_t used as a temporary buffer for data operations.
- **Description**: The `llama_io_write_file` class is a specialized implementation of the `llama_io_write_i` interface, designed to handle file writing operations. It maintains a reference to a `llama_file` object for performing raw file writes and keeps track of the total number of bytes written through the `size_written` member. Additionally, it uses a temporary buffer, `temp_buffer`, to facilitate writing tensor data to the file. This class provides methods to write raw data and tensor data, ensuring that the file operations are efficiently managed.
- **Member Functions**:
    - [`llama_io_write_file::llama_io_write_file`](#llama_io_write_filellama_io_write_file)
    - [`llama_io_write_file::write`](#llama_io_write_filewrite)
    - [`llama_io_write_file::write_tensor`](#llama_io_write_filewrite_tensor)
    - [`llama_io_write_file::n_bytes`](#llama_io_write_filen_bytes)
- **Inherits From**:
    - [`llama_io_write_i::llama_io_write_i`](llama-io.h.driver.md#llama_io_write_illama_io_write_i)

**Methods**

---
#### llama\_io\_write\_file::llama\_io\_write\_file<!-- {{#callable:llama_io_write_file::llama_io_write_file}} -->
The `llama_io_write_file` constructor initializes an instance of the `llama_io_write_file` class with a given `llama_file` pointer.
- **Inputs**:
    - `f`: A pointer to a `llama_file` object that the `llama_io_write_file` instance will use for writing operations.
- **Control Flow**:
    - The constructor takes a `llama_file` pointer as an argument.
    - It initializes the `file` member variable with the provided `llama_file` pointer.
- **Output**: There is no output from this constructor as it is used to initialize an object.
- **See also**: [`llama_io_write_file`](#llama_io_write_file)  (Data Structure)


---
#### llama\_io\_write\_file::write<!-- {{#callable:llama_io_write_file::write}} -->
The `write` function writes raw data from a source to a file and updates the total size written.
- **Inputs**:
    - `src`: A pointer to the source data to be written to the file.
    - `size`: The size of the data to be written, in bytes.
- **Control Flow**:
    - Call the `write_raw` method on the `file` object, passing `src` and `size` as arguments to write the data.
    - Increment the `size_written` member variable by `size` to update the total size of data written.
- **Output**: This function does not return any value.
- **See also**: [`llama_io_write_file`](#llama_io_write_file)  (Data Structure)


---
#### llama\_io\_write\_file::write\_tensor<!-- {{#callable:llama_io_write_file::write_tensor}} -->
The `write_tensor` function writes a portion of a tensor to a file by first resizing a temporary buffer, retrieving the tensor data into this buffer, and then writing the buffer's contents to the file.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` object, representing the tensor whose data is to be written.
    - `offset`: A `size_t` value indicating the starting point within the tensor data from which to begin writing.
    - `size`: A `size_t` value specifying the number of bytes to write from the tensor.
- **Control Flow**:
    - Resize the `temp_buffer` to the specified `size`.
    - Retrieve the tensor data starting from `offset` and of length `size` into `temp_buffer` using [`ggml_backend_tensor_get`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get).
    - Write the data from `temp_buffer` to the file using the [`write`](#llama_io_write_dummywrite) method.
- **Output**: This function does not return any value; it performs its operations as a side effect.
- **Functions called**:
    - [`ggml_backend_tensor_get`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get)
    - [`llama_io_write_dummy::write`](#llama_io_write_dummywrite)
- **See also**: [`llama_io_write_file`](#llama_io_write_file)  (Data Structure)


---
#### llama\_io\_write\_file::n\_bytes<!-- {{#callable:llama_io_write_file::n_bytes}} -->
The `n_bytes` function returns the total number of bytes written to a file.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the `size_written` member variable.
- **Output**: The function returns a `size_t` value representing the total number of bytes written.
- **See also**: [`llama_io_write_file`](#llama_io_write_file)  (Data Structure)



---
### llama\_io\_read\_file<!-- {{#data_structure:llama_io_read_file}} -->
- **Type**: `class`
- **Members**:
    - `file`: A pointer to a `llama_file` object used for file operations.
    - `size_read`: A size_t variable that tracks the number of bytes read from the file.
    - `temp_buffer`: A vector of uint8_t used as a temporary buffer for reading data.
- **Description**: The `llama_io_read_file` class is a specialized implementation of the `llama_io_read_i` interface, designed to facilitate reading operations from a file. It maintains a reference to a `llama_file` object for performing raw read operations and keeps track of the total number of bytes read through the `size_read` member. The class also utilizes a temporary buffer, `temp_buffer`, to store data temporarily during read operations, allowing for efficient data handling and processing.
- **Member Functions**:
    - [`llama_io_read_file::llama_io_read_file`](#llama_io_read_filellama_io_read_file)
    - [`llama_io_read_file::read_to`](#llama_io_read_fileread_to)
    - [`llama_io_read_file::read`](#llama_io_read_fileread)
    - [`llama_io_read_file::n_bytes`](#llama_io_read_filen_bytes)
- **Inherits From**:
    - [`llama_io_read_i::llama_io_read_i`](llama-io.h.driver.md#llama_io_read_illama_io_read_i)

**Methods**

---
#### llama\_io\_read\_file::llama\_io\_read\_file<!-- {{#callable:llama_io_read_file::llama_io_read_file}} -->
The `llama_io_read_file` constructor initializes an instance of the `llama_io_read_file` class with a given `llama_file` pointer.
- **Inputs**:
    - `f`: A pointer to a `llama_file` object, which represents the file to be read.
- **Control Flow**:
    - The constructor takes a `llama_file` pointer as an argument.
    - It initializes the `file` member variable with the provided `llama_file` pointer.
- **Output**: There is no output from this constructor; it initializes the object state.
- **See also**: [`llama_io_read_file`](#llama_io_read_file)  (Data Structure)


---
#### llama\_io\_read\_file::read\_to<!-- {{#callable:llama_io_read_file::read_to}} -->
The `read_to` function reads raw data from a file into a specified destination buffer and updates the total size of data read.
- **Inputs**:
    - `dst`: A pointer to the destination buffer where the data will be read into.
    - `size`: The number of bytes to read from the file into the destination buffer.
- **Control Flow**:
    - Call the `read_raw` method on the `file` object to read `size` bytes of data into the `dst` buffer.
    - Increment the `size_read` member variable by `size` to update the total number of bytes read.
- **Output**: This function does not return a value; it modifies the destination buffer and updates the internal state of the object.
- **See also**: [`llama_io_read_file`](#llama_io_read_file)  (Data Structure)


---
#### llama\_io\_read\_file::read<!-- {{#callable:llama_io_read_file::read}} -->
The `read` function reads a specified number of bytes from a file into a temporary buffer and returns a pointer to the buffer.
- **Inputs**:
    - `size`: The number of bytes to read from the file.
- **Control Flow**:
    - Resize the `temp_buffer` to the specified `size`.
    - Call [`read_to`](#llama_io_read_fileread_to) to read `size` bytes from the file into `temp_buffer`.
    - Return a pointer to the data in `temp_buffer`.
- **Output**: A pointer to the `temp_buffer` containing the read data.
- **Functions called**:
    - [`llama_io_read_file::read_to`](#llama_io_read_fileread_to)
- **See also**: [`llama_io_read_file`](#llama_io_read_file)  (Data Structure)


---
#### llama\_io\_read\_file::n\_bytes<!-- {{#callable:llama_io_read_file::n_bytes}} -->
The `n_bytes` function returns the total number of bytes read so far by the `llama_io_read_file` object.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the value of the `size_read` member variable, which tracks the number of bytes read.
- **Output**: The function returns a `size_t` value representing the number of bytes read.
- **See also**: [`llama_io_read_file`](#llama_io_read_file)  (Data Structure)



---
### llama\_context<!-- {{#data_structure:llama_context}} -->
- **Description**: [See definition](llama-context.h.driver.md#llama_context)
- **Member Functions**:
    - [`llama_context::llama_context`](#llama_contextllama_context)
    - [`llama_context::~llama_context`](#llama_contextllama_context)
    - [`llama_context::synchronize`](#llama_contextsynchronize)
    - [`llama_context::get_model`](#llama_contextget_model)
    - [`llama_context::get_cparams`](#llama_contextget_cparams)
    - [`llama_context::get_sched`](#llama_contextget_sched)
    - [`llama_context::get_ctx_compute`](#llama_contextget_ctx_compute)
    - [`llama_context::n_ctx`](#llama_contextn_ctx)
    - [`llama_context::n_ctx_per_seq`](#llama_contextn_ctx_per_seq)
    - [`llama_context::n_batch`](#llama_contextn_batch)
    - [`llama_context::n_ubatch`](#llama_contextn_ubatch)
    - [`llama_context::n_seq_max`](#llama_contextn_seq_max)
    - [`llama_context::n_threads`](#llama_contextn_threads)
    - [`llama_context::n_threads_batch`](#llama_contextn_threads_batch)
    - [`llama_context::get_memory`](#llama_contextget_memory)
    - [`llama_context::kv_self_defrag_sched`](#llama_contextkv_self_defrag_sched)
    - [`llama_context::kv_self_update`](#llama_contextkv_self_update)
    - [`llama_context::pooling_type`](#llama_contextpooling_type)
    - [`llama_context::get_logits`](#llama_contextget_logits)
    - [`llama_context::get_logits_ith`](#llama_contextget_logits_ith)
    - [`llama_context::get_embeddings`](#llama_contextget_embeddings)
    - [`llama_context::get_embeddings_ith`](#llama_contextget_embeddings_ith)
    - [`llama_context::get_embeddings_seq`](#llama_contextget_embeddings_seq)
    - [`llama_context::attach_threadpool`](#llama_contextattach_threadpool)
    - [`llama_context::detach_threadpool`](#llama_contextdetach_threadpool)
    - [`llama_context::set_n_threads`](#llama_contextset_n_threads)
    - [`llama_context::set_abort_callback`](#llama_contextset_abort_callback)
    - [`llama_context::set_embeddings`](#llama_contextset_embeddings)
    - [`llama_context::set_causal_attn`](#llama_contextset_causal_attn)
    - [`llama_context::set_warmup`](#llama_contextset_warmup)
    - [`llama_context::set_adapter_lora`](#llama_contextset_adapter_lora)
    - [`llama_context::rm_adapter_lora`](#llama_contextrm_adapter_lora)
    - [`llama_context::clear_adapter_lora`](#llama_contextclear_adapter_lora)
    - [`llama_context::apply_adapter_cvec`](#llama_contextapply_adapter_cvec)
    - [`llama_context::process_ubatch`](#llama_contextprocess_ubatch)
    - [`llama_context::encode`](#llama_contextencode)
    - [`llama_context::decode`](#llama_contextdecode)
    - [`llama_context::output_reserve`](#llama_contextoutput_reserve)
    - [`llama_context::graph_max_nodes`](#llama_contextgraph_max_nodes)
    - [`llama_context::graph_init`](#llama_contextgraph_init)
    - [`llama_context::graph_reserve`](#llama_contextgraph_reserve)
    - [`llama_context::graph_build`](#llama_contextgraph_build)
    - [`llama_context::graph_compute`](#llama_contextgraph_compute)
    - [`llama_context::graph_get_cb`](#llama_contextgraph_get_cb)
    - [`llama_context::state_get_size`](#llama_contextstate_get_size)
    - [`llama_context::state_get_data`](#llama_contextstate_get_data)
    - [`llama_context::state_set_data`](#llama_contextstate_set_data)
    - [`llama_context::state_seq_get_size`](#llama_contextstate_seq_get_size)
    - [`llama_context::state_seq_get_data`](#llama_contextstate_seq_get_data)
    - [`llama_context::state_seq_set_data`](#llama_contextstate_seq_set_data)
    - [`llama_context::state_load_file`](#llama_contextstate_load_file)
    - [`llama_context::state_save_file`](#llama_contextstate_save_file)
    - [`llama_context::state_seq_load_file`](#llama_contextstate_seq_load_file)
    - [`llama_context::state_seq_save_file`](#llama_contextstate_seq_save_file)
    - [`llama_context::state_write_data`](#llama_contextstate_write_data)
    - [`llama_context::state_read_data`](#llama_contextstate_read_data)
    - [`llama_context::state_seq_write_data`](#llama_contextstate_seq_write_data)
    - [`llama_context::state_seq_read_data`](#llama_contextstate_seq_read_data)
    - [`llama_context::perf_get_data`](#llama_contextperf_get_data)
    - [`llama_context::perf_reset`](#llama_contextperf_reset)
    - [`llama_context::opt_init`](#llama_contextopt_init)
    - [`llama_context::opt_epoch_iter`](#llama_contextopt_epoch_iter)
    - [`llama_context::opt_epoch`](#llama_contextopt_epoch)

**Methods**

---
#### llama\_context::llama\_context<!-- {{#callable:llama_context::llama_context}} -->
Constructs a `llama_context` object using a specified `llama_model` and parameters.
- **Inputs**:
    - `model`: A constant reference to a `llama_model` object that contains the model's parameters and state.
    - `params`: A `llama_context_params` structure containing various configuration parameters for the context.
- **Control Flow**:
    - Logs the start of the context construction.
    - Initializes timing variables from the model.
    - Validates and sets maximum sequence length, ensuring it does not exceed predefined limits.
    - Configures context parameters based on the provided parameters and model hyperparameters.
    - Checks and adjusts the batch size and context size based on causal attention settings.
    - Initializes GPU backends and allocates necessary resources for computation.
    - Reserves memory for the worst-case graph based on the maximum number of sequences and tokens.
    - Logs various parameters for debugging and performance monitoring.
    - Handles potential warnings related to context size and sequence limits.
- **Output**: Constructs and returns a `llama_context` object that is ready for use in model inference or training.
- **Functions called**:
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`ggml_backend_dev_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_name)
    - [`ggml_backend_dev_count`](../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_dev_count)
    - [`ggml_backend_dev_type`](../ggml/include/ggml-backend.h.driver.md#ggml_backend_dev_type)
    - [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)
    - [`llama_set_abort_callback`](#llama_set_abort_callback)
    - [`llama_context::output_reserve`](#llama_contextoutput_reserve)
    - [`ggml_backend_buffer_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_name)
    - [`ggml_backend_buffer_get_size`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_get_size)
    - [`ggml_tensor_overhead`](../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_graph_overhead_custom`](../ggml/src/ggml.c.driver.md#ggml_graph_overhead_custom)
    - [`ggml_backend_dev_get_props`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_get_props)
    - [`ggml_backend_sched_get_n_copies`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_get_n_copies)
    - [`llama_context::graph_reserve`](#llama_contextgraph_reserve)
    - [`ggml_backend_sched_get_n_splits`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_get_n_splits)
    - [`ggml_graph_n_nodes`](../ggml/src/ggml.c.driver.md#ggml_graph_n_nodes)
    - [`ggml_backend_sched_get_buffer_size`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_get_buffer_size)
    - [`ggml_backend_buft_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buft_name)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::\~llama\_context<!-- {{#callable:llama_context::~llama_context}} -->
Destructor for the `llama_context` class that frees the optimization context.
- **Inputs**: None
- **Control Flow**:
    - Calls [`ggml_opt_free`](../ggml/src/ggml-opt.cpp.driver.md#ggml_opt_free) to release the resources associated with the optimization context (`opt_ctx`).
- **Output**: This function does not return a value; it performs cleanup by freeing resources.
- **Functions called**:
    - [`ggml_opt_free`](../ggml/src/ggml-opt.cpp.driver.md#ggml_opt_free)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::synchronize<!-- {{#callable:llama_context::synchronize}} -->
The `synchronize` method in the `llama_context` class synchronizes the backend scheduler and updates evaluation statistics.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - Calls [`ggml_backend_sched_synchronize`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_synchronize) to synchronize the backend scheduler.
    - Checks the number of queued tokens and updates evaluation time statistics accordingly.
    - If this is the first evaluation, it calculates the load time.
    - Resets the number of queued tokens and the compute start time.
- **Output**: This method does not return any value; it modifies the internal state of the `llama_context` instance.
- **Functions called**:
    - [`ggml_backend_sched_synchronize`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_synchronize)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::get\_model<!-- {{#callable:llama_context::get_model}} -->
The `get_model` function returns a constant reference to the `llama_model` associated with the `llama_context`.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the member variable `model` of the `llama_context` class.
    - It does not perform any computations or checks before returning the reference.
- **Output**: The output is a constant reference to a `llama_model` object, which represents the model associated with the current context.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::get\_cparams<!-- {{#callable:llama_context::get_cparams}} -->
The `get_cparams` method retrieves the current configuration parameters of the `llama_context`.
- **Inputs**: None
- **Control Flow**:
    - The method directly returns the member variable `cparams` of the `llama_context` instance.
- **Output**: The method returns a constant reference to a `llama_cparams` object, which contains the configuration parameters for the context.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::get\_sched<!-- {{#callable:llama_context::get_sched}} -->
The `get_sched` method retrieves the current scheduler from the `llama_context`.
- **Inputs**: None
- **Control Flow**:
    - The method directly accesses the `sched` member of the `llama_context` class.
    - It calls the `get` method on `sched` to obtain the current scheduler.
- **Output**: Returns the current scheduler of type `ggml_backend_sched_t`.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::get\_ctx\_compute<!-- {{#callable:llama_context::get_ctx_compute}} -->
The `get_ctx_compute` method retrieves the compute context associated with the `llama_context`.
- **Inputs**: None
- **Control Flow**:
    - The method directly accesses the `ctx_compute` member of the `llama_context` class.
    - It calls the `get()` method on `ctx_compute`, which is a smart pointer, to return the raw pointer.
- **Output**: The method returns a pointer of type `ggml_context*`, which represents the compute context.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::n\_ctx<!-- {{#callable:llama_context::n_ctx}} -->
The `n_ctx` function returns the number of context tokens configured in the `llama_context`.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `cparams` member of the `llama_context` instance.
    - It retrieves the value of `n_ctx` from `cparams` and returns it.
- **Output**: The output is a `uint32_t` representing the number of context tokens.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::n\_ctx\_per\_seq<!-- {{#callable:llama_context::n_ctx_per_seq}} -->
The `n_ctx_per_seq` function calculates the number of context tokens available per sequence.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `cparams` member of the `llama_context` class.
    - It performs a division of `cparams.n_ctx` by `cparams.n_seq_max` to compute the context tokens per sequence.
- **Output**: The function returns a `uint32_t` value representing the number of context tokens available for each sequence.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::n\_batch<!-- {{#callable:llama_context::n_batch}} -->
The `n_batch` function returns the number of batches configured in the `llama_context`.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `n_batch` member of the `cparams` structure.
    - No conditional statements or loops are present in this function.
- **Output**: The function outputs a `uint32_t` value representing the number of batches.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::n\_ubatch<!-- {{#callable:llama_context::n_ubatch}} -->
Returns the number of unbatched sequences in the `llama_context`.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `n_ubatch` member of the `cparams` structure.
    - No conditional logic or loops are present in the function.
- **Output**: Returns a `uint32_t` representing the number of unbatched sequences.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::n\_seq\_max<!-- {{#callable:llama_context::n_seq_max}} -->
The `n_seq_max` function returns the maximum number of sequences that can be processed in parallel.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `cparams` member of the `llama_context` class.
    - It retrieves the value of `n_seq_max` from `cparams` and returns it.
- **Output**: The output is a `uint32_t` value representing the maximum number of sequences that can be processed in parallel.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::n\_threads<!-- {{#callable:llama_context::n_threads}} -->
The `n_threads` function returns the number of threads configured in the `llama_context`.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `n_threads` member of the `cparams` structure.
    - No conditional statements or loops are present in the function.
- **Output**: The function outputs a `uint32_t` value representing the number of threads.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::n\_threads\_batch<!-- {{#callable:llama_context::n_threads_batch}} -->
The `n_threads_batch` function returns the number of threads allocated for batch processing in the `llama_context`.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `n_threads_batch` member of the `cparams` structure.
    - No conditional statements or loops are present, making the function straightforward.
- **Output**: The function outputs a `uint32_t` value representing the number of threads allocated for batch processing.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::get\_memory<!-- {{#callable:llama_context::get_memory}} -->
The `get_memory` method retrieves the memory associated with the `llama_context`.
- **Inputs**: None
- **Control Flow**:
    - The method directly accesses the `memory` member of the `llama_context` class.
    - It calls the `get` method on the `memory` object to retrieve the current memory state.
- **Output**: The method returns a `llama_memory_t` object, which represents the current memory state of the context.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::kv\_self\_defrag\_sched<!-- {{#callable:llama_context::kv_self_defrag_sched}} -->
The `kv_self_defrag_sched` function schedules a memory optimization for the key-value cache if memory is available.
- **Inputs**: None
- **Control Flow**:
    - The function first checks if the `memory` member of the `llama_context` instance is null.
    - If `memory` is null, the function returns immediately without performing any actions.
    - If `memory` is not null, it sets the `memory_force_optimize` flag to true.
- **Output**: The function does not return a value; it modifies the internal state of the `llama_context` instance by setting a flag to indicate that memory optimization should be forced.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::kv\_self\_update<!-- {{#callable:llama_context::kv_self_update}} -->
The `kv_self_update` method updates the key-value memory cache of the `llama_context` if memory is available.
- **Inputs**:
    - `optimize`: A boolean flag indicating whether to optimize the memory update.
- **Control Flow**:
    - Checks if the `memory` is null; if so, returns false.
    - Combines the `optimize` flag with `memory_force_optimize` and resets `memory_force_optimize` to false.
    - Initializes the memory update and checks its status, handling different cases for success, no update, and failure.
    - If the update is successful, applies the memory state.
    - Initializes the full memory state and reserves a new worst-case graph based on the current context parameters.
    - Logs errors if graph reservation fails.
- **Output**: Returns true if the memory update was successful, otherwise false.
- **Functions called**:
    - [`llama_context::graph_reserve`](#llama_contextgraph_reserve)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::pooling\_type<!-- {{#callable:llama_context::pooling_type}} -->
Returns the current pooling type used in the `llama_context`.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `pooling_type` member of the `cparams` structure.
    - It returns the value of `cparams.pooling_type` without any additional logic or conditions.
- **Output**: Returns an enumeration value of type `llama_pooling_type` that indicates the pooling strategy currently set in the context parameters.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::get\_logits<!-- {{#callable:llama_context::get_logits}} -->
The `get_logits` function returns a pointer to the logits array stored in the `llama_context`.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `logits` member variable of the `llama_context` class.
    - It does not perform any checks or computations before returning the pointer.
- **Output**: The output is a pointer to a float array representing the logits, which is used for further processing in the model.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::get\_logits\_ith<!-- {{#callable:llama_context::get_logits_ith}} -->
Retrieves the logits for the ith output from the logits buffer.
- **Inputs**:
    - `i`: An integer index representing the output for which logits are requested.
- **Control Flow**:
    - Check if the `logits` pointer is null; if so, throw a runtime error.
    - If `i` is negative, calculate the corresponding positive index `j` by adding it to `n_outputs` and check if `j` is valid.
    - If `i` is non-negative, check if it is within the bounds of `output_ids`; if not, throw an out-of-range error.
    - Assign `j` to the value at `output_ids[i]`.
    - Check if `j` is valid; if not, throw an error indicating a corrupt output buffer.
    - Return the address of the logits corresponding to the index `j`.
- **Output**: Returns a pointer to the logits corresponding to the ith output, or nullptr if an error occurs.
- **Functions called**:
    - [`format`](llama-impl.cpp.driver.md#format)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::get\_embeddings<!-- {{#callable:llama_context::get_embeddings}} -->
The `get_embeddings` function returns a pointer to the embeddings array.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the member variable `embd`, which is a pointer to the embeddings array.
- **Output**: The output is a pointer to a float array containing the embeddings.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::get\_embeddings\_ith<!-- {{#callable:llama_context::get_embeddings_ith}} -->
The `get_embeddings_ith` function retrieves the embedding vector at a specified index from the llama context.
- **Inputs**:
    - `i`: An integer index specifying the position of the embedding to retrieve.
- **Control Flow**:
    - The function starts by initializing an integer `j` to -1.
    - It checks if the `embd` pointer is null, throwing an error if it is, indicating that no embeddings are available.
    - If the index `i` is negative, it calculates `j` as `n_outputs + i`, ensuring it is within valid bounds.
    - If `i` is non-negative, it checks if `i` is out of range of `output_ids`, throwing an error if it is.
    - If `i` is valid, it assigns `j` to the corresponding value in `output_ids`.
    - The function checks if `j` is negative or exceeds `n_outputs`, throwing errors for both cases.
    - Finally, it returns a pointer to the embedding vector located at `embd + j * model.hparams.n_embd`.
- **Output**: Returns a pointer to the embedding vector corresponding to the specified index, or nullptr if an error occurs.
- **Functions called**:
    - [`format`](llama-impl.cpp.driver.md#format)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::get\_embeddings\_seq<!-- {{#callable:llama_context::get_embeddings_seq}} -->
The `get_embeddings_seq` function retrieves the embeddings associated with a specific sequence ID from the `llama_context`.
- **Inputs**:
    - `seq_id`: A `llama_seq_id` representing the identifier of the sequence whose embeddings are to be retrieved.
- **Control Flow**:
    - The function first attempts to find the sequence ID in the `embd_seq` map.
    - If the sequence ID is not found, it returns a null pointer.
    - If found, it returns a pointer to the data of the corresponding vector of embeddings.
- **Output**: Returns a pointer to a float array containing the embeddings for the specified sequence ID, or nullptr if the sequence ID does not exist.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::attach\_threadpool<!-- {{#callable:llama_context::attach_threadpool}} -->
Attaches a thread pool to the `llama_context` for managing concurrent tasks.
- **Inputs**:
    - `threadpool`: A pointer to a `ggml_threadpool_t` representing the thread pool for general tasks.
    - `threadpool_batch`: A pointer to a `ggml_threadpool_t` representing the thread pool for batch tasks, which can be null.
- **Control Flow**:
    - Logs the function call for debugging purposes.
    - Assigns the provided `threadpool` to the context's `threadpool` member.
    - If `threadpool_batch` is not null, assigns it to the context's `threadpool_batch` member; otherwise, assigns `threadpool` to `threadpool_batch`.
- **Output**: This function does not return a value; it modifies the state of the `llama_context` by setting its thread pool members.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::detach\_threadpool<!-- {{#callable:llama_context::detach_threadpool}} -->
Detaches the thread pool from the `llama_context` instance.
- **Inputs**: None
- **Control Flow**:
    - Logs a debug message indicating the function call.
    - Sets the `threadpool` and `threadpool_batch` members of the `llama_context` instance to `nullptr`.
- **Output**: This function does not return any value; it modifies the state of the `llama_context` instance by detaching its thread pools.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::set\_n\_threads<!-- {{#callable:llama_context::set_n_threads}} -->
Sets the number of threads for computation and batch processing in the `llama_context`.
- **Inputs**:
    - `n_threads`: An integer representing the number of threads to be used for computation.
    - `n_threads_batch`: An integer representing the number of threads to be used for batch processing.
- **Control Flow**:
    - Logs the values of `n_threads` and `n_threads_batch` for debugging purposes.
    - Updates the `cparams.n_threads` with the value of `n_threads`.
    - Updates the `cparams.n_threads_batch` with the value of `n_threads_batch`.
- **Output**: This function does not return a value; it modifies the internal state of the `llama_context` by updating the number of threads used for computation and batch processing.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::set\_abort\_callback<!-- {{#callable:llama_context::set_abort_callback}} -->
Sets a callback function to handle abort events during processing.
- **Inputs**:
    - `abort_callback`: A pointer to a function that takes a void pointer as an argument and returns a boolean, which will be called to check if the operation should be aborted.
    - `abort_callback_data`: A pointer to user-defined data that will be passed to the abort callback function.
- **Control Flow**:
    - Logs the entry into the function for debugging purposes.
    - Assigns the provided `abort_callback` and `abort_callback_data` to the class members.
    - Iterates over each backend in the `backends` vector.
    - For each backend, retrieves the backend registration and attempts to get the address of the `ggml_backend_set_abort_callback` function.
    - If the function is found, it calls this function with the current backend, the abort callback, and the associated data.
- **Output**: The function does not return a value; it sets the abort callback for each backend to allow for aborting operations based on user-defined conditions.
- **Functions called**:
    - [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::set\_embeddings<!-- {{#callable:llama_context::set_embeddings}} -->
Sets the `embeddings` parameter in the `llama_context` structure.
- **Inputs**:
    - `value`: A boolean indicating whether embeddings should be enabled (true) or disabled (false).
- **Control Flow**:
    - Logs the function call and the value of the `value` parameter using `LLAMA_LOG_DEBUG`.
    - Assigns the `value` parameter to the `embeddings` field of the `cparams` member of the `llama_context` instance.
- **Output**: This function does not return a value; it modifies the internal state of the `llama_context` by setting the `embeddings` parameter.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::set\_causal\_attn<!-- {{#callable:llama_context::set_causal_attn}} -->
Sets the causal attention parameter for the `llama_context`.
- **Inputs**:
    - `value`: A boolean indicating whether to enable or disable causal attention.
- **Control Flow**:
    - Logs the function call and the value of the input parameter.
    - Updates the `cparams.causal_attn` member of the `llama_context` structure with the provided value.
- **Output**: This function does not return a value; it modifies the internal state of the `llama_context`.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::set\_warmup<!-- {{#callable:llama_context::set_warmup}} -->
Sets the warmup parameter in the `llama_context` structure.
- **Inputs**:
    - `value`: A boolean indicating whether to enable or disable the warmup phase.
- **Control Flow**:
    - Logs the current function name and the value of the input parameter using `LLAMA_LOG_DEBUG`.
    - Assigns the input boolean value to the `warmup` field of the `cparams` member of the `llama_context`.
- **Output**: This function does not return a value; it modifies the internal state of the `llama_context` by setting the warmup parameter.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::set\_adapter\_lora<!-- {{#callable:llama_context::set_adapter_lora}} -->
Sets the scale for a specified `llama_adapter_lora` in the `llama_context`.
- **Inputs**:
    - `adapter`: A pointer to a `llama_adapter_lora` instance that represents the adapter to be set.
    - `scale`: A float value representing the scale to be applied to the specified adapter.
- **Control Flow**:
    - Logs the function call with the adapter pointer and scale value for debugging purposes.
    - Assigns the provided scale to the specified adapter in the `loras` map of the `llama_context`.
- **Output**: This function does not return a value; it modifies the internal state of the `llama_context` by associating the given scale with the specified adapter.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::rm\_adapter\_lora<!-- {{#callable:llama_context::rm_adapter_lora}} -->
Removes a `llama_adapter_lora` from the `llama_context` if it exists.
- **Inputs**:
    - `adapter`: A pointer to the `llama_adapter_lora` instance that needs to be removed from the context.
- **Control Flow**:
    - Logs the address of the adapter being removed for debugging purposes.
    - Searches for the adapter in the `loras` collection.
    - If the adapter is found, it is erased from the collection and the function returns true.
    - If the adapter is not found, the function returns false.
- **Output**: Returns a boolean indicating whether the adapter was successfully removed (true) or not found (false).
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::clear\_adapter\_lora<!-- {{#callable:llama_context::clear_adapter_lora}} -->
Clears all `llama_adapter_lora` entries from the `loras` container.
- **Inputs**: None
- **Control Flow**:
    - Logs a debug message indicating the function call.
    - Clears the `loras` container, effectively removing all stored adapter Lora configurations.
- **Output**: This function does not return any value; it modifies the internal state of the `llama_context` by clearing the `loras` container.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::apply\_adapter\_cvec<!-- {{#callable:llama_context::apply_adapter_cvec}} -->
Applies an adapter represented as a context vector to the model using the provided data and specified indices.
- **Inputs**:
    - `data`: A pointer to an array of floats representing the input data to be processed by the adapter.
    - `len`: The length of the input data array.
    - `n_embd`: The dimensionality of the embeddings used in the model.
    - `il_start`: The starting index for the input layer where the adapter will be applied.
    - `il_end`: The ending index for the input layer where the adapter will be applied.
- **Control Flow**:
    - Logs the starting and ending indices for the adapter application.
    - Calls the `apply` method of the `cvec` object, passing the model, data, length, embedding dimension, and input layer indices.
- **Output**: Returns a boolean indicating the success or failure of the adapter application.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::process\_ubatch<!-- {{#callable:llama_context::process_ubatch}} -->
Processes a user batch (`ubatch`) using a specified graph type (`gtype`), applying memory state if provided, and returns the result of the computation.
- **Inputs**:
    - `ubatch`: A constant reference to a `llama_ubatch` object that contains the input data for processing.
    - `gtype`: An enumeration value of type `llm_graph_type` that specifies the type of graph to be used for processing the batch.
    - `mstate`: A pointer to a `llama_memory_state_i` object that represents the memory state to be applied before processing, or nullptr if no memory state is to be applied.
    - `ret`: A reference to a `ggml_status` variable that will be set to indicate the success or failure of the operation.
- **Control Flow**:
    - Checks if `mstate` is provided and applies it; if it fails, logs an error and sets `ret` to `GGML_STATUS_FAILED`.
    - Initializes a computation graph using [`graph_init`](#llama_contextgraph_init); if it fails, logs an error and sets `ret` to `GGML_STATUS_FAILED`.
    - Builds the graph with the provided `ubatch`, `gtype`, and `mstate`; if it fails, logs an error and sets `ret` to `GGML_STATUS_FAILED`.
    - Allocates the graph in the backend scheduler; if it fails, logs an error and sets `ret` to `GGML_STATUS_ALLOC_FAILED`.
    - Sets the inputs of the result to the `ubatch`.
    - Computes the graph; if the computation fails, logs an error and sets `ret` to the corresponding status.
    - If successful, sets `ret` to `GGML_STATUS_SUCCESS` and returns the result.
- **Output**: Returns a pointer to a `llm_graph_result_ptr` containing the result of the computation, or nullptr if an error occurred.
- **Functions called**:
    - [`llama_context::graph_init`](#llama_contextgraph_init)
    - [`llama_context::graph_build`](#llama_contextgraph_build)
    - [`ggml_backend_sched_alloc_graph`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_alloc_graph)
    - [`llama_context::graph_compute`](#llama_contextgraph_compute)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::encode<!-- {{#callable:llama_context::encode}} -->
Encodes a batch of tokens into embeddings using a llama model.
- **Inputs**:
    - `inp_batch`: A reference to a `llama_batch` structure containing the tokens to be encoded.
- **Control Flow**:
    - Checks if the number of tokens in `inp_batch` is zero and logs an error if true.
    - Allocates memory for the input batch using `llama_batch_allocr`.
    - Validates the tokens in the batch, ensuring they are within valid ranges.
    - Processes the batch in a single shot due to non-causal encoding constraints.
    - Reserves output buffer space for the number of tokens being processed.
    - Sets up the computation graph and processes the input batch using [`process_ubatch`](#llama_contextprocess_ubatch).
    - Extracts embeddings based on the specified pooling type and stores them in the output buffers.
    - Resets the scheduler state for the next token processing.
- **Output**: Returns 0 on success, -1 if there are no tokens, -2 if output buffer reservation fails, or other negative values indicating specific errors.
- **Functions called**:
    - [`llama_sbatch::llama_sbatch`](llama-batch.h.driver.md#llama_sbatchllama_sbatch)
    - [`llama_context::output_reserve`](#llama_contextoutput_reserve)
    - [`ggml_backend_sched_reset`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_reset)
    - [`ggml_backend_sched_set_eval_callback`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_set_eval_callback)
    - [`llama_context::process_ubatch`](#llama_contextprocess_ubatch)
    - [`ggml_backend_tensor_get_async`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get_async)
    - [`llama_context::synchronize`](#llama_contextsynchronize)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::decode<!-- {{#callable:llama_context::decode}} -->
The `decode` method processes a batch of tokens to generate outputs using the llama model, handling memory management and validation throughout the process.
- **Inputs**:
    - `inp_batch`: A reference to a `llama_batch` object that contains the input tokens and associated metadata for decoding.
- **Control Flow**:
    - Checks if memory is available; if not, it calls the [`encode`](#llama_contextencode) method instead.
    - Validates the number of tokens in the input batch and checks for the presence of position and sequence ID.
    - Allocates memory for the input batch if necessary.
    - Validates the tokens in the batch against the model's vocabulary.
    - Counts the number of outputs based on the logits and pooling type.
    - Handles memory state initialization and updates, retrying if necessary.
    - Processes the input batch in a loop, extracting logits and embeddings as needed.
    - Sorts the output mappings to maintain the order of the original input batch.
- **Output**: Returns 0 on success, -1 for invalid input, -2 for memory allocation failures, and other error codes for specific issues encountered during processing.
- **Functions called**:
    - [`llama_context::encode`](#llama_contextencode)
    - [`llama_context::kv_self_update`](#llama_contextkv_self_update)
    - [`llama_context::output_reserve`](#llama_contextoutput_reserve)
    - [`ggml_backend_sched_reset`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_reset)
    - [`ggml_backend_sched_set_eval_callback`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_set_eval_callback)
    - [`llama_context::process_ubatch`](#llama_contextprocess_ubatch)
    - [`ggml_backend_tensor_get_async`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get_async)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::output\_reserve<!-- {{#callable:llama_context::output_reserve}} -->
Reserves output space for logits and embeddings based on the number of outputs requested.
- **Inputs**:
    - `n_outputs`: An integer representing the number of outputs for which space needs to be reserved.
- **Control Flow**:
    - Calculates the maximum number of outputs by comparing `n_outputs` with the maximum sequence length.
    - Determines the sizes required for logits and embeddings based on model parameters and architecture.
    - Resizes the `output_ids` vector if it is empty to accommodate the batch size.
    - Checks the current size of the output buffer and reallocates it if the new size exceeds the current capacity.
    - Allocates a new buffer for the output if necessary, using the appropriate backend buffer type.
    - Sets the pointers for `logits` and `embd` based on the allocated buffer and whether logits and embeddings are required.
    - Initializes the `output_ids` to -1 to indicate invalid IDs.
    - Updates the maximum number of outputs and returns the maximum number of outputs reserved.
- **Output**: Returns the maximum number of outputs for which space was reserved.
- **Functions called**:
    - [`llama_context::n_seq_max`](#llama_contextn_seq_max)
    - [`ggml_backend_buffer_get_size`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_get_size)
    - [`ggml_backend_buffer_get_base`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_get_base)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::graph\_max\_nodes<!-- {{#callable:llama_context::graph_max_nodes}} -->
Calculates the maximum number of nodes that can be used in a computation graph based on the number of tensors in the model.
- **Inputs**: None
- **Control Flow**:
    - The function uses `std::max` to determine the maximum value between 65536 and 5 times the number of tensors in the model.
    - It retrieves the number of tensors from the `model` member of the `llama_context` class.
- **Output**: Returns an `int32_t` representing the maximum number of nodes for the computation graph.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::graph\_init<!-- {{#callable:llama_context::graph_init}} -->
Initializes a computation graph for the `llama_context` using pre-allocated memory.
- **Inputs**: None
- **Control Flow**:
    - Creates a `ggml_init_params` structure with memory size and buffer information.
    - Resets the `ctx_compute` context with the initialized parameters.
    - Creates a new computation graph with a maximum number of nodes.
- **Output**: Returns a pointer to the newly created `ggml_cgraph`.
- **Functions called**:
    - [`ggml_init`](../ggml/src/ggml.c.driver.md#ggml_init)
    - [`ggml_new_graph_custom`](../ggml/src/ggml.c.driver.md#ggml_new_graph_custom)
    - [`llama_context::graph_max_nodes`](#llama_contextgraph_max_nodes)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::graph\_reserve<!-- {{#callable:llama_context::graph_reserve}} -->
Reserves a computation graph for processing a batch of tokens in the `llama_context`.
- **Inputs**:
    - `n_tokens`: The total number of tokens to be processed in the batch.
    - `n_seqs`: The number of sequences in the batch.
    - `n_outputs`: The number of outputs expected from the graph.
    - `mstate`: A pointer to the memory state that may be applied to the context's memory.
- **Control Flow**:
    - Logs the initial parameters for graph reservation.
    - Checks if `n_tokens` is a multiple of `n_seqs` and adjusts it if necessary.
    - Saves the current number of outputs and sets it to the new value.
    - Initializes a token for the batch and creates a `llama_ubatch` structure.
    - Calls `graph_init()` to initialize a new computation graph.
    - Attempts to build the graph using `graph_build()` with the initialized context and ubatch.
    - Restores the original number of outputs after graph building.
    - Resets the scheduler and attempts to reserve the graph in the backend scheduler.
    - Returns the graph pointer if successful, or logs an error and returns nullptr if any step fails.
- **Output**: Returns a pointer to the reserved `ggml_cgraph` if successful, or nullptr if the graph could not be built or reserved.
- **Functions called**:
    - [`llama_context::graph_init`](#llama_contextgraph_init)
    - [`llama_context::graph_build`](#llama_contextgraph_build)
    - [`ggml_backend_sched_reset`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_reset)
    - [`ggml_backend_sched_reserve`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_reserve)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::graph\_build<!-- {{#callable:llama_context::graph_build}} -->
Builds a computation graph for processing a batch of inputs in a specified context.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` that provides the context for computation.
    - `gf`: A pointer to a `ggml_cgraph` structure that represents the computation graph to be built.
    - `ubatch`: A reference to a `llama_ubatch` structure containing the input data for the batch.
    - `gtype`: An enumeration value of type `llm_graph_type` that specifies the type of graph to build.
    - `mstate`: A pointer to a `llama_memory_state_i` structure that holds the memory state, or nullptr if not used.
- **Control Flow**:
    - The function calls `model.build_graph` with a structured input containing various parameters including the context, architecture, hyperparameters, and the input batch.
    - It passes the graph structure `gf` and the graph type `gtype` to the `build_graph` method, which constructs the computation graph based on the provided inputs.
- **Output**: Returns a pointer to a `llm_graph_result_ptr` that contains the result of the graph building process, or nullptr if the process fails.
- **Functions called**:
    - [`llama_context::graph_get_cb`](#llama_contextgraph_get_cb)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::graph\_compute<!-- {{#callable:llama_context::graph_compute}} -->
Computes the graph for the llama context using a specified graph structure and thread settings.
- **Inputs**:
    - `gf`: A pointer to a `ggml_cgraph` structure representing the computation graph to be executed.
    - `batched`: A boolean indicating whether the computation should be performed in a batched manner, affecting the number of threads used.
- **Control Flow**:
    - Determine the number of threads to use based on the `batched` parameter.
    - If a CPU backend is available, set the thread pool for the backend.
    - Iterate over all backend functions to set the number of threads for each backend.
    - Schedule the graph computation asynchronously using the [`ggml_backend_sched_graph_compute_async`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_graph_compute_async) function.
    - Log an error if the scheduling fails.
- **Output**: Returns a `ggml_status` indicating the success or failure of the graph computation.
- **Functions called**:
    - [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)
    - [`ggml_backend_sched_graph_compute_async`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_graph_compute_async)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::graph\_get\_cb<!-- {{#callable:llama_context::graph_get_cb}} -->
The `graph_get_cb` method returns a callback function that configures tensor properties based on the current layer index and tensor name during graph computation.
- **Inputs**:
    - `ubatch`: A reference to a `llama_ubatch` structure that contains information about the current batch of tokens being processed.
    - `cur`: A pointer to a `ggml_tensor` that represents the current tensor being processed in the graph.
    - `name`: A string representing the name of the current tensor, used for identifying and configuring the tensor.
    - `il`: An integer representing the index of the current layer in the graph, used to determine specific configurations for the tensor.
- **Control Flow**:
    - The function first checks if the layer index `il` is non-negative; if so, it formats the name of the tensor by appending the layer index.
    - If the `offload_kqv` parameter is not set, it checks if the tensor name is 'kqv_merged_cont' and sets its backend to CPU.
    - It then checks if the number of tokens in the `ubatch` is less than 32 or if full offloading is required, and if the tensor name is 'norm', it assigns the appropriate backend based on the device layer.
- **Output**: The output is a lambda function that encapsulates the logic for configuring tensor properties during graph computation, allowing for dynamic adjustments based on the current processing context.
- **Functions called**:
    - [`ggml_format_name`](../ggml/src/ggml.c.driver.md#ggml_format_name)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_backend_sched_set_tensor_backend`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_sched_set_tensor_backend)
    - [`ggml_backend_supports_op`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_supports_op)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_get\_size<!-- {{#callable:llama_context::state_get_size}} -->
The `state_get_size` function retrieves the size of the state data by writing to a dummy IO object.
- **Inputs**: None
- **Control Flow**:
    - A `llama_io_write_dummy` object is instantiated to simulate writing data.
    - The function attempts to call [`state_write_data`](#llama_contextstate_write_data) with the dummy IO object.
    - If [`state_write_data`](#llama_contextstate_write_data) executes successfully, it returns the size of the state data.
    - If an exception occurs during the execution, an error is logged and the function returns 0.
- **Output**: The function returns the size of the state data as a `size_t` value, or 0 if an error occurs.
- **Functions called**:
    - [`llama_context::state_write_data`](#llama_contextstate_write_data)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_get\_data<!-- {{#callable:llama_context::state_get_data}} -->
Retrieves the current state data of the `llama_context` and writes it to the provided destination buffer.
- **Inputs**:
    - `dst`: A pointer to a buffer where the state data will be written.
    - `size`: The size of the destination buffer in bytes.
- **Control Flow**:
    - Creates a `llama_io_write_buffer` object using the provided destination buffer and size.
    - Attempts to write the state data using the [`state_write_data`](#llama_contextstate_write_data) method.
    - Catches any exceptions thrown during the write operation, logs an error message, and returns 0 if an error occurs.
- **Output**: Returns the number of bytes written to the destination buffer, or 0 if an error occurred.
- **Functions called**:
    - [`llama_context::state_write_data`](#llama_contextstate_write_data)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_set\_data<!-- {{#callable:llama_context::state_set_data}} -->
Sets the state data of the `llama_context` from a given source buffer.
- **Inputs**:
    - `src`: A pointer to a buffer containing the state data to be set.
    - `size`: The size of the data in bytes to be read from the source buffer.
- **Control Flow**:
    - Creates a `llama_io_read_buffer` object to read from the provided source buffer.
    - Attempts to read the state data using the [`state_read_data`](#llama_contextstate_read_data) method.
    - If an exception occurs during the read operation, logs an error message and returns 0.
- **Output**: Returns the number of bytes successfully read and set from the source buffer, or 0 if an error occurred.
- **Functions called**:
    - [`llama_context::state_read_data`](#llama_contextstate_read_data)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_seq\_get\_size<!-- {{#callable:llama_context::state_seq_get_size}} -->
Retrieves the size of the state sequence data for a given sequence ID.
- **Inputs**:
    - `seq_id`: An identifier for the sequence whose state size is being queried.
- **Control Flow**:
    - Creates a dummy IO object for writing data.
    - Attempts to call [`state_seq_write_data`](#llama_contextstate_seq_write_data) to get the size of the state data for the specified sequence ID.
    - Catches any exceptions thrown during the process and logs an error message.
    - Returns 0 if an error occurs, otherwise returns the size obtained from [`state_seq_write_data`](#llama_contextstate_seq_write_data).
- **Output**: Returns the size of the state sequence data as a size_t value, or 0 if an error occurs.
- **Functions called**:
    - [`llama_context::state_seq_write_data`](#llama_contextstate_seq_write_data)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_seq\_get\_data<!-- {{#callable:llama_context::state_seq_get_data}} -->
Retrieves data associated with a specific sequence ID and writes it to a provided buffer.
- **Inputs**:
    - `seq_id`: An identifier for the sequence whose data is to be retrieved.
    - `dst`: A pointer to a buffer where the retrieved data will be written.
    - `size`: The size of the buffer pointed to by `dst`.
- **Control Flow**:
    - Creates a `llama_io_write_buffer` object to handle writing data to the specified buffer.
    - Attempts to call the [`state_seq_write_data`](#llama_contextstate_seq_write_data) function to write the sequence data to the buffer.
    - Catches any exceptions thrown during the write operation, logs an error message, and returns 0 if an error occurs.
- **Output**: Returns the number of bytes written to the buffer, or 0 if an error occurred.
- **Functions called**:
    - [`llama_context::state_seq_write_data`](#llama_contextstate_seq_write_data)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_seq\_set\_data<!-- {{#callable:llama_context::state_seq_set_data}} -->
Sets the state sequence data for a given sequence ID from a source buffer.
- **Inputs**:
    - `seq_id`: An identifier for the sequence whose state data is being set.
    - `src`: A pointer to the source buffer containing the state data to be set.
    - `size`: The size of the data to be read from the source buffer.
- **Control Flow**:
    - Creates a `llama_io_read_buffer` object to read from the provided source buffer.
    - Attempts to read the state sequence data using the [`state_seq_read_data`](#llama_contextstate_seq_read_data) function.
    - Catches any exceptions thrown during the read operation, logs an error message, and returns 0.
- **Output**: Returns the number of bytes read from the source buffer, or 0 if an error occurred.
- **Functions called**:
    - [`llama_context::state_seq_read_data`](#llama_contextstate_seq_read_data)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_load\_file<!-- {{#callable:llama_context::state_load_file}} -->
Loads the state of a `llama_context` from a specified file, validating its format and restoring the context's tokens and state.
- **Inputs**:
    - `filepath`: A pointer to a null-terminated string representing the path to the file from which the state will be loaded.
    - `tokens_out`: A pointer to an array of `llama_token` where the loaded tokens will be stored.
    - `n_token_capacity`: The maximum number of tokens that can be stored in the `tokens_out` array.
    - `n_token_count_out`: A pointer to a size_t variable where the actual number of tokens read from the file will be stored.
- **Control Flow**:
    - A `llama_file` object is created to handle file operations for the specified `filepath`.
    - The function reads a 32-bit magic number and version from the file to validate its format.
    - If the magic number or version does not match expected values, an error is logged and the function returns false.
    - The function reads the number of tokens from the file and checks if it exceeds the provided capacity.
    - If the token count is valid, the tokens are read into the `tokens_out` array and the count is updated.
    - The remaining state data is read from the file, and if the number of bytes read does not match the expected size, an error is logged and the function returns false.
    - If all operations succeed, the function returns true.
- **Output**: Returns true if the state was successfully loaded, otherwise returns false.
- **Functions called**:
    - [`llama_context::state_read_data`](#llama_contextstate_read_data)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_save\_file<!-- {{#callable:llama_context::state_save_file}} -->
Saves the current state of the `llama_context` to a specified file along with a sequence of tokens.
- **Inputs**:
    - `filepath`: A pointer to a character array (C-string) representing the path to the file where the state will be saved.
    - `tokens`: A pointer to an array of `llama_token` representing the tokens to be saved.
    - `n_token_count`: A size_t value indicating the number of tokens to be saved.
- **Control Flow**:
    - A `llama_file` object is created to handle file operations in binary write mode using the provided `filepath`.
    - The function writes a magic number and version number to the file to identify the session format.
    - The number of tokens (`n_token_count`) is written to the file, followed by the raw token data.
    - A `llama_io_write_file` object is created to facilitate writing the context state to the file.
    - The [`state_write_data`](#llama_contextstate_write_data) method is called to write the context's state to the file.
    - The function returns true to indicate successful completion.
- **Output**: Returns a boolean value indicating whether the state was successfully saved to the file.
- **Functions called**:
    - [`llama_context::state_write_data`](#llama_contextstate_write_data)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_seq\_load\_file<!-- {{#callable:llama_context::state_seq_load_file}} -->
Loads a sequence state from a file, including token data and context state.
- **Inputs**:
    - `seq_id`: An identifier for the sequence whose state is being loaded.
    - `filepath`: The path to the file from which the sequence state will be loaded.
    - `tokens_out`: An output array where the loaded tokens will be stored.
    - `n_token_capacity`: The maximum number of tokens that can be stored in the tokens_out array.
    - `n_token_count_out`: A pointer to a variable that will hold the actual number of tokens loaded.
- **Control Flow**:
    - Opens the specified file in binary read mode.
    - Reads the magic number and version from the file to verify its integrity.
    - If the magic number or version does not match expected values, logs an error and returns 0.
    - Reads the number of tokens from the file and checks if it exceeds the provided capacity.
    - If the token count is valid, reads the tokens into the tokens_out array and updates n_token_count_out.
    - Calculates the remaining size of the file to read the context state.
    - Reads the context state using a helper function and checks for success.
    - Logs an error if restoring the state fails and returns 0.
    - Returns the total number of bytes read from the file.
- **Output**: Returns the number of bytes read from the file, or 0 if an error occurred.
- **Functions called**:
    - [`llama_context::state_seq_read_data`](#llama_contextstate_seq_read_data)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_seq\_save\_file<!-- {{#callable:llama_context::state_seq_save_file}} -->
Saves the state of a sequence to a specified file, including the prompt and context state.
- **Inputs**:
    - `seq_id`: An identifier for the sequence whose state is being saved.
    - `filepath`: The path to the file where the state will be saved.
    - `tokens`: An array of tokens representing the prompt to be saved.
    - `n_token_count`: The number of tokens in the array.
- **Control Flow**:
    - A `llama_file` object is created to handle file operations in binary write mode.
    - The magic number and version are written to the file to identify the format.
    - The number of tokens is written to the file, followed by the raw token data.
    - A `llama_io_write_file` object is created to facilitate writing the context state.
    - The [`state_seq_write_data`](#llama_contextstate_seq_write_data) function is called to write the context state to the file.
    - The total number of bytes written is calculated and returned.
- **Output**: Returns the total number of bytes written to the file, including the header and context state.
- **Functions called**:
    - [`llama_context::state_seq_write_data`](#llama_contextstate_seq_write_data)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_write\_data<!-- {{#callable:llama_context::state_write_data}} -->
The `state_write_data` function writes the current state of the `llama_context` to a specified output stream.
- **Inputs**:
    - `io`: An instance of `llama_io_write_i` that provides the interface for writing data to an output stream.
- **Control Flow**:
    - Logs the start of the state writing process.
    - Writes model architecture information to the output stream.
    - Writes the number of output IDs and their positions to the output stream.
    - Writes the size and data of logits to the output stream if available.
    - Writes the size and data of embeddings to the output stream if available.
    - If memory is not null, writes the key-value (KV) self state to the output stream.
    - Returns the total number of bytes written to the output stream.
- **Output**: Returns the total number of bytes written to the output stream.
- **Functions called**:
    - [`llm_arch_name`](llama-arch.cpp.driver.md#llm_arch_name)
    - [`llama_context::n_batch`](#llama_contextn_batch)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_read\_data<!-- {{#callable:llama_context::state_read_data}} -->
Reads the state data from the provided input stream and updates the context's internal state.
- **Inputs**:
    - `io`: An instance of `llama_io_read_i` that provides the interface for reading data from a source.
- **Control Flow**:
    - Logs the start of the state reading process.
    - Reads the model architecture string and checks it against the current model architecture, throwing an error if they do not match.
    - Reads the number of output IDs and reserves space for them, throwing an error if the reservation fails.
    - Reads the output IDs and validates them against the batch size, updating the context's output ID mapping.
    - Reads the logits size and checks if the buffer is sufficient, throwing an error if it is not.
    - Reads the logits data into the context's logits buffer if the size is valid.
    - Reads the embeddings size and checks if the buffer is sufficient, throwing an error if it is not.
    - Reads the embeddings data into the context's embeddings buffer if the size is valid.
    - If memory is used, reads the KV self state from the input stream.
    - Returns the total number of bytes read from the input stream.
- **Output**: Returns the number of bytes read from the input stream.
- **Functions called**:
    - [`llm_arch_name`](llama-arch.cpp.driver.md#llm_arch_name)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`llama_context::output_reserve`](#llama_contextoutput_reserve)
    - [`llama_context::n_batch`](#llama_contextn_batch)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_seq\_write\_data<!-- {{#callable:llama_context::state_seq_write_data}} -->
Writes the state of a sequence to the provided I/O interface.
- **Inputs**:
    - `io`: An instance of `llama_io_write_i` that provides the interface for writing data.
    - `seq_id`: An identifier for the sequence whose state is being written.
- **Control Flow**:
    - The function first checks if the `memory` member of the `llama_context` instance is not null.
    - If `memory` is valid, it calls the `state_write` method of the `memory` object, passing the `io` and `seq_id` as arguments.
    - Finally, it returns the number of bytes written by the `io` interface.
- **Output**: Returns the number of bytes written to the `io` interface.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::state\_seq\_read\_data<!-- {{#callable:llama_context::state_seq_read_data}} -->
Reads sequence state data from the provided input stream.
- **Inputs**:
    - `io`: An instance of `llama_io_read_i` that provides the input stream to read data from.
    - `seq_id`: An identifier for the sequence whose state data is being read.
- **Control Flow**:
    - The function first marks the `seq_id` as unused to avoid compiler warnings.
    - If the `memory` member of the `llama_context` instance is not null, it calls the `state_read` method of the `memory` object, passing the `io` and `seq_id` as arguments.
    - Finally, it returns the number of bytes read from the input stream.
- **Output**: Returns the number of bytes read from the input stream.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::perf\_get\_data<!-- {{#callable:llama_context::perf_get_data}} -->
Retrieves performance-related data from the `llama_context` instance.
- **Inputs**:
    - `this`: A constant reference to the `llama_context` instance from which performance data is being retrieved.
- **Control Flow**:
    - Initializes a `llama_perf_context_data` structure to hold the performance data.
    - Converts the time measurements from microseconds to milliseconds for various performance metrics.
    - Uses `std::max` to ensure that the number of evaluations is at least 1, preventing division by zero in subsequent calculations.
    - Returns the populated `llama_perf_context_data` structure.
- **Output**: Returns a `llama_perf_context_data` structure containing performance metrics such as start time, load time, evaluation time, and counts of evaluations.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::perf\_reset<!-- {{#callable:llama_context::perf_reset}} -->
Resets the performance metrics of the `llama_context` instance.
- **Inputs**: None
- **Control Flow**:
    - Sets the start time for performance measurement using `ggml_time_us()`.
    - Resets the evaluation time (`t_eval_us`) and the number of evaluations (`n_eval`) to zero.
    - Resets the prompt evaluation time (`t_p_eval_us`) and the number of prompt evaluations (`n_p_eval`) to zero.
- **Output**: The function does not return any value.
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::opt\_init<!-- {{#callable:llama_context::opt_init}} -->
Initializes the optimization context for a given model with specified optimization parameters.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure that contains the model parameters and hyperparameters.
    - `lopt_params`: A `llama_opt_params` structure containing optimization parameters such as context size and parameter filters.
- **Control Flow**:
    - Asserts that the optimization context (`opt_ctx`) is not already initialized.
    - Sets the training context size (`n_ctx_train`) based on the provided optimization parameters or defaults to the model's context size.
    - Calculates the batch size (`n_batch`) and unbatched size (`n_ubatch`) based on the model's context size and optimization parameters.
    - Validates that the context size is divisible by the batch size and the batch size is divisible by the unbatched size.
    - Initializes optimization parameters using default settings and updates them with values from `lopt_params`.
    - Calls `ggml_opt_init` to initialize the optimization context with the prepared parameters.
    - Applies parameter filters to various model parameters using the [`llama_set_param`](#llama_set_param) function.
    - Iterates through each layer of the model and applies the parameter filters to each tensor.
- **Output**: The function does not return a value but initializes the optimization context for training.
- **Functions called**:
    - [`llama_context::n_ctx`](#llama_contextn_ctx)
    - [`ggml_opt_default_params`](../ggml/src/ggml-opt.cpp.driver.md#ggml_opt_default_params)
    - [`llama_set_param`](#llama_set_param)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::opt\_epoch\_iter<!-- {{#callable:llama_context::opt_epoch_iter}} -->
The `opt_epoch_iter` function performs a single iteration of optimization over a dataset using a specified batch of tokens and labels, updating the model's parameters based on the computed results.
- **Inputs**:
    - `dataset`: A dataset of type `ggml_opt_dataset_t` that contains the training data for the optimization process.
    - `result`: A result object of type `ggml_opt_result_t` that will store the outcome of the optimization iteration.
    - `tokens`: A vector of `llama_token` representing the input tokens for the current optimization iteration.
    - `labels_sparse`: A vector of `llama_token` containing the sparse labels corresponding to the input tokens.
    - `batch`: A reference to a `llama_batch` structure that holds the current batch of tokens and their associated metadata.
    - `callback`: An optional callback function of type `ggml_opt_epoch_callback` that is called after each optimization step.
    - `train`: A boolean flag indicating whether the optimization is for training (true) or evaluation (false).
    - `idata_in_loop`: An integer representing the current index of the data point being processed in the loop.
    - `ndata_in_loop`: An integer representing the total number of data points processed in the current loop iteration.
    - `t_loop_start`: A timestamp indicating the start time of the current loop iteration.
- **Control Flow**:
    - The function begins by asserting that the optimization context (`opt_ctx`) is valid.
    - It retrieves the context size (`n_ctx`) and determines the batch sizes (`n_batch` and `n_ubatch`) based on the model's parameters.
    - The memory is cleared to prepare for the new optimization iteration.
    - A loop iterates over the context size in increments of the batch size, processing each batch of tokens.
    - Within the loop, the function populates the `batch` structure with the current tokens and their positions.
    - It initializes the memory state for the current batch and checks for successful initialization.
    - The output buffer is reserved based on the number of tokens in the batch.
    - Another loop processes the current batch, applying the memory state and building the computation graph.
    - The optimization context is prepared and allocated for the current iteration.
    - The labels are set for the optimization process, and the optimization evaluation is performed.
    - If a callback is provided, it is invoked with the current training state and results.
    - The loop continues until all tokens in the current context have been processed.
- **Output**: The function does not return a value but updates the `result` parameter with the outcome of the optimization and may invoke a callback function with the current state.
- **Functions called**:
    - [`llama_context::output_reserve`](#llama_contextoutput_reserve)
    - [`llama_context::graph_init`](#llama_contextgraph_init)
    - [`llama_context::graph_build`](#llama_contextgraph_build)
    - [`ggml_graph_size`](../ggml/src/ggml.c.driver.md#ggml_graph_size)
    - [`ggml_tensor_overhead`](../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_graph_overhead_custom`](../ggml/src/ggml.c.driver.md#ggml_graph_overhead_custom)
    - [`ggml_init`](../ggml/src/ggml.c.driver.md#ggml_init)
    - [`ggml_opt_prepare_alloc`](../ggml/src/ggml-opt.cpp.driver.md#ggml_opt_prepare_alloc)
    - [`ggml_opt_alloc`](../ggml/src/ggml-opt.cpp.driver.md#ggml_opt_alloc)
    - [`ggml_opt_labels`](../ggml/src/ggml-opt.cpp.driver.md#ggml_opt_labels)
    - [`ggml_set_zero`](../ggml/src/ggml.c.driver.md#ggml_set_zero)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_opt_eval`](../ggml/src/ggml-opt.cpp.driver.md#ggml_opt_eval)
    - [`ggml_free`](../ggml/src/ggml.c.driver.md#ggml_free)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)


---
#### llama\_context::opt\_epoch<!-- {{#callable:llama_context::opt_epoch}} -->
The `opt_epoch` function performs a training and evaluation epoch on a dataset using specified training and evaluation results, while allowing for callbacks during the process.
- **Inputs**:
    - `dataset`: A dataset of type `ggml_opt_dataset_t` used for training and evaluation.
    - `result_train`: A result structure of type `ggml_opt_result_t` to store training results.
    - `result_eval`: A result structure of type `ggml_opt_result_t` to store evaluation results.
    - `idata_split`: An integer indicating the index at which to split the dataset for training and evaluation.
    - `callback_train`: A callback function of type `ggml_opt_epoch_callback` to be called during training.
    - `callback_eval`: A callback function of type `ggml_opt_epoch_callback` to be called during evaluation.
- **Control Flow**:
    - The function begins by retrieving the context size (`n_ctx`), batch size (`n_batch`), and the number of unbatched samples (`n_ubatch`).
    - It asserts that the `idata_split` is within valid bounds of the dataset size.
    - The function initializes a batch structure and two vectors for tokens and labels.
    - It enters a loop to process the training data up to the `idata_split`, fetching batches of tokens and labels, and calling [`opt_epoch_iter`](#llama_contextopt_epoch_iter) for training.
    - After processing the training data, it resets the loop to process the evaluation data from `idata_split` to the end of the dataset, calling [`opt_epoch_iter`](#llama_contextopt_epoch_iter) for evaluation.
    - Finally, it frees the allocated batch resources.
- **Output**: The function does not return a value but updates the provided `result_train` and `result_eval` structures with the results of the training and evaluation processes.
- **Functions called**:
    - [`ggml_opt_dataset_ndata`](../ggml/src/ggml-opt.cpp.driver.md#ggml_opt_dataset_ndata)
    - [`ggml_opt_dataset_get_batch_host`](../ggml/src/ggml-opt.cpp.driver.md#ggml_opt_dataset_get_batch_host)
    - [`llama_context::opt_epoch_iter`](#llama_contextopt_epoch_iter)
- **See also**: [`llama_context`](llama-context.h.driver.md#llama_context)  (Data Structure)



# Functions

---
### llama\_set\_param<!-- {{#callable:llama_set_param}} -->
Sets parameters for a given tensor based on a filter function.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that represents the tensor whose parameters are to be set.
    - `param_filter`: A function pointer of type `llama_opt_param_filter` that is used to filter which parameters should be set.
    - `userdata`: A pointer to user-defined data that can be passed to the `param_filter` function.
- **Control Flow**:
    - Check if the `tensor` is null or if its type is not `GGML_TYPE_F32`, in which case the function returns immediately.
    - Invoke the `param_filter` function with the `tensor` and `userdata` as arguments; if it returns false, exit the function.
    - Check if the `tensor` name is 'token_embd.weight' or 'rope_freqs.weight'; if so, return without setting parameters.
    - Call [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param) to set the parameters of the tensor.
- **Output**: The function does not return a value; it modifies the tensor's parameters directly if the conditions are met.
- **Functions called**:
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)


---
### llama\_context\_default\_params<!-- {{#callable:llama_context_default_params}} -->
Returns a `llama_context_params` structure initialized with default values.
- **Inputs**: None
- **Control Flow**:
    - A `llama_context_params` structure named `result` is created and initialized with default values for various parameters such as context size, batch size, number of threads, and others.
    - The function then returns the initialized `result` structure.
- **Output**: Returns a `llama_context_params` structure containing default parameters for the llama context.


---
### llama\_init\_from\_model<!-- {{#callable:llama_init_from_model}} -->
Initializes a `llama_context` from a given `llama_model` and specified parameters, performing various validation checks and logging errors if any conditions are not met.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure that contains the model to be used for context initialization.
    - `params`: A `llama_context_params` structure containing various parameters for context initialization, such as batch sizes and context length.
- **Control Flow**:
    - Checks if the `model` pointer is null and logs an error if it is, returning nullptr.
    - Validates that both `n_batch` and `n_ubatch` are not zero, logging an error if they are.
    - Ensures that either `n_ctx` or the model's training context size is non-zero, logging an error if both are zero.
    - Checks for compatibility of `flash_attn` with the model architecture and disables it if necessary.
    - Validates that if `type_v` is quantized, `flash_attn` must be enabled, logging an error if it is not.
    - Attempts to create a new `llama_context` using the provided model and parameters, catching any exceptions and logging errors if initialization fails.
- **Output**: Returns a pointer to a newly initialized `llama_context` if successful, or nullptr if any validation checks fail or an exception occurs during initialization.
- **Functions called**:
    - [`ggml_is_quantized`](../ggml/src/ggml.c.driver.md#ggml_is_quantized)


---
### llama\_new\_context\_with\_model<!-- {{#callable:llama_new_context_with_model}} -->
Creates a new `llama_context` initialized with the specified `llama_model` and parameters.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure that contains the model to be used for the context.
    - `params`: A `llama_context_params` structure that contains various parameters for initializing the context.
- **Control Flow**:
    - Checks if the `model` pointer is null and logs an error if it is.
    - Validates that at least one of `n_batch` or `n_ubatch` in `params` is non-zero.
    - Validates that at least one of `n_ctx` or `model->hparams.n_ctx_train` is non-zero.
    - Checks compatibility of `flash_attn` with the model architecture and logs a warning if incompatible.
    - Checks if the vector cache quantization requires `flash_attn` and logs an error if not satisfied.
    - Attempts to create a new `llama_context` using the `llama_context` constructor with the provided model and parameters.
    - Catches any exceptions during the context initialization and logs an error message.
- **Output**: Returns a pointer to the newly created `llama_context`, or null if initialization fails.
- **Functions called**:
    - [`llama_init_from_model`](#llama_init_from_model)


---
### llama\_free<!-- {{#callable:llama_free}} -->
Frees the memory allocated for a `llama_context` instance.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` instance that needs to be freed.
- **Control Flow**:
    - The function takes a pointer to a `llama_context` as an argument.
    - It calls the `delete` operator on the provided pointer to deallocate the memory.
- **Output**: This function does not return any value; it simply frees the memory associated with the `llama_context`.


---
### llama\_n\_ctx<!-- {{#callable:llama_n_ctx}} -->
Returns the number of context tokens available in the given `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that contains the context information for the Llama model.
- **Control Flow**:
    - The function directly accesses the `n_ctx` method of the `llama_context` instance pointed to by `ctx`.
    - It returns the value obtained from the `n_ctx` method, which represents the number of context tokens.
- **Output**: Returns a `uint32_t` value representing the number of context tokens available in the `llama_context`.


---
### llama\_n\_batch<!-- {{#callable:llama_n_batch}} -->
Returns the batch size configured in the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a constant `llama_context` object, which contains the configuration and state for the Llama model.
- **Control Flow**:
    - The function directly accesses the `n_batch` method of the `llama_context` object pointed to by `ctx`.
    - It returns the value obtained from the `n_batch` method.
- **Output**: Returns a `uint32_t` representing the batch size configured in the `llama_context`.


---
### llama\_n\_ubatch<!-- {{#callable:llama_n_ubatch}} -->
Returns the number of unbatched sequences for the given `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a constant `llama_context` object, which contains the configuration and state for the Llama model.
- **Control Flow**:
    - The function directly accesses the `n_ubatch` method of the `llama_context` object pointed to by `ctx`.
    - It returns the result of the `n_ubatch` method call.
- **Output**: Returns a `uint32_t` value representing the number of unbatched sequences.


---
### llama\_n\_seq\_max<!-- {{#callable:llama_n_seq_max}} -->
Retrieves the maximum number of sequences allowed in the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the context for the Llama model.
- **Control Flow**:
    - The function directly accesses the `n_seq_max` method of the `llama_context` instance pointed to by `ctx`.
    - It returns the value obtained from the `n_seq_max` method.
- **Output**: Returns a `uint32_t` representing the maximum number of sequences that can be processed in the context.


---
### llama\_get\_model<!-- {{#callable:llama_get_model}} -->
The `llama_get_model` function retrieves a pointer to the `llama_model` associated with a given `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, which contains the model and its parameters.
- **Control Flow**:
    - The function directly accesses the `get_model` method of the `llama_context` instance pointed to by `ctx`.
    - It returns the address of the `llama_model` object that is encapsulated within the `llama_context`.
- **Output**: Returns a pointer to a constant `llama_model` object.


---
### llama\_get\_kv\_self<!-- {{#callable:llama_get_kv_self}} -->
The `llama_get_kv_self` function retrieves the key-value cache from the given `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object from which the key-value cache is to be retrieved.
- **Control Flow**:
    - The function uses `dynamic_cast` to attempt to cast the result of `ctx->get_memory()` to a `llama_kv_cache` type.
    - If the cast is successful, it returns the casted pointer; otherwise, it returns a null pointer.
- **Output**: Returns a pointer to a `llama_kv_cache` object if the cast is successful, otherwise returns null.


---
### llama\_kv\_self\_update<!-- {{#callable:llama_kv_self_update}} -->
Updates the key-value memory cache of the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object that contains the state and memory of the model.
- **Control Flow**:
    - Calls the `kv_self_update` method of the `llama_context` object with the argument `false`.
- **Output**: The function does not return a value; it performs an update operation on the context's memory.


---
### llama\_pooling\_type<!-- {{#callable:llama_pooling_type}} -->
The `llama_pooling_type` function retrieves the pooling type from the given `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a constant `llama_context` object from which the pooling type is to be retrieved.
- **Control Flow**:
    - The function directly calls the `pooling_type` method of the `llama_context` object pointed to by `ctx`.
    - The result of the `pooling_type` method is returned as the output of the function.
- **Output**: Returns an enumeration value of type `llama_pooling_type` that indicates the pooling strategy used in the context.


---
### llama\_attach\_threadpool<!-- {{#callable:llama_attach_threadpool}} -->
The `llama_attach_threadpool` function attaches a specified thread pool to a given llama context.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that represents the context to which the thread pool will be attached.
    - `threadpool`: A `ggml_threadpool_t` representing the thread pool to be attached for general use.
    - `threadpool_batch`: A `ggml_threadpool_t` representing the thread pool to be used for batch processing; if null, the general thread pool is used.
- **Control Flow**:
    - The function calls the `attach_threadpool` method of the `llama_context` instance, passing the provided thread pools.
    - If `threadpool_batch` is null, it defaults to using `threadpool` for batch processing.
- **Output**: The function does not return a value; it modifies the state of the `llama_context` by attaching the specified thread pools.


---
### llama\_detach\_threadpool<!-- {{#callable:llama_detach_threadpool}} -->
The `llama_detach_threadpool` function detaches the thread pool from the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object from which the thread pool will be detached.
- **Control Flow**:
    - The function calls the `detach_threadpool` method on the `ctx` object.
    - This effectively sets the thread pool pointers in the `llama_context` to null, indicating that no thread pool is currently attached.
- **Output**: The function does not return any value; it modifies the state of the `llama_context` by detaching its thread pool.


---
### llama\_set\_n\_threads<!-- {{#callable:llama_set_n_threads}} -->
Sets the number of threads for processing in the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure, which holds the context for the LLaMA model.
    - `n_threads`: An integer specifying the number of threads to be used for processing.
    - `n_threads_batch`: An integer specifying the number of threads to be used for batch processing.
- **Control Flow**:
    - The function calls the `set_n_threads` method of the `llama_context` instance pointed to by `ctx`.
    - It passes the `n_threads` and `n_threads_batch` parameters to the `set_n_threads` method.
- **Output**: This function does not return a value; it modifies the state of the `llama_context` by setting the number of threads.


---
### llama\_n\_threads<!-- {{#callable:llama_n_threads}} -->
The `llama_n_threads` function retrieves the number of threads configured in the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, which contains the configuration and state for the Llama model.
- **Control Flow**:
    - The function directly accesses the `n_threads` method of the `llama_context` structure pointed to by `ctx`.
    - It returns the value obtained from the `n_threads` method.
- **Output**: The function returns an integer representing the number of threads configured for the `llama_context`.


---
### llama\_n\_threads\_batch<!-- {{#callable:llama_n_threads_batch}} -->
The `llama_n_threads_batch` function retrieves the number of threads allocated for batch processing in a `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which contains the configuration and state for the Llama model.
- **Control Flow**:
    - The function directly calls the `n_threads_batch` method on the `llama_context` object pointed to by `ctx`.
    - It returns the result of the `n_threads_batch` method, which is an integer value.
- **Output**: Returns an integer representing the number of threads allocated for batch processing.


---
### llama\_set\_abort\_callback<!-- {{#callable:llama_set_abort_callback}} -->
Sets an abort callback function for the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure that holds the context for the Llama model.
    - `abort_callback`: A pointer to a function that takes a void pointer and returns a boolean, which will be called to check if the operation should be aborted.
    - `abort_callback_data`: A pointer to user-defined data that will be passed to the abort callback function.
- **Control Flow**:
    - The function calls the `set_abort_callback` method on the `ctx` object, passing the `abort_callback` and `abort_callback_data` as arguments.
    - This method sets the abort callback for the context and updates all backend devices with the new callback.
- **Output**: This function does not return a value; it modifies the state of the `llama_context` by setting the abort callback.


---
### llama\_set\_embeddings<!-- {{#callable:llama_set_embeddings}} -->
Sets the embeddings flag in the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure that holds the state of the model.
    - `embeddings`: A boolean value indicating whether to enable or disable embeddings.
- **Control Flow**:
    - The function directly calls the `set_embeddings` method on the `ctx` object, passing the `embeddings` boolean value.
    - No conditional logic or loops are present; the function performs a single operation.
- **Output**: This function does not return a value; it modifies the state of the `llama_context` by setting the embeddings flag.


---
### llama\_set\_causal\_attn<!-- {{#callable:llama_set_causal_attn}} -->
Sets the causal attention flag in the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the context for the model.
    - `causal_attn`: A boolean value indicating whether to enable causal attention.
- **Control Flow**:
    - The function calls the `set_causal_attn` method on the `ctx` object, passing the `causal_attn` value.
- **Output**: This function does not return a value; it modifies the state of the `llama_context`.


---
### llama\_set\_warmup<!-- {{#callable:llama_set_warmup}} -->
Sets the warmup state of the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure that holds the context for the Llama model.
    - `warmup`: A boolean value indicating whether to enable or disable the warmup state.
- **Control Flow**:
    - The function directly calls the `set_warmup` method of the `llama_context` instance pointed to by `ctx`.
    - The value of `warmup` is passed to the `set_warmup` method to update the warmup state.
- **Output**: This function does not return a value; it modifies the state of the `llama_context`.


---
### llama\_synchronize<!-- {{#callable:llama_synchronize}} -->
The `llama_synchronize` function synchronizes the execution of the `llama_context` by ensuring that all scheduled tasks are completed.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object that contains the state and configuration for the Llama model.
- **Control Flow**:
    - Calls the `synchronize` method on the `llama_context` object pointed to by `ctx`.
    - The `synchronize` method handles the synchronization of backend tasks and updates performance metrics.
- **Output**: This function does not return a value; it performs synchronization and updates internal state metrics within the `llama_context`.


---
### llama\_get\_logits<!-- {{#callable:llama_get_logits}} -->
`llama_get_logits` retrieves the logits from the given `llama_context` after synchronizing its state.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object which holds the state and parameters for the model.
- **Control Flow**:
    - The function first calls `ctx->synchronize()` to ensure that the context is in a consistent state before accessing the logits.
    - It then calls `ctx->get_logits()` to retrieve the logits from the context.
- **Output**: Returns a pointer to a float array containing the logits.


---
### llama\_get\_logits\_ith<!-- {{#callable:llama_get_logits_ith}} -->
Retrieves the logits for the ith output from the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the state and parameters of the model.
    - `i`: An integer index representing the position of the desired logits in the output.
- **Control Flow**:
    - Calls the `synchronize` method on the `ctx` to ensure that all previous computations are completed.
    - Calls the `get_logits_ith` method on the `ctx` to retrieve the logits for the specified index `i`.
- **Output**: Returns a pointer to a float array containing the logits for the ith output.


---
### llama\_get\_embeddings<!-- {{#callable:llama_get_embeddings}} -->
The `llama_get_embeddings` function retrieves the embeddings from the given `llama_context` after synchronizing its state.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which contains the state and parameters necessary for retrieving embeddings.
- **Control Flow**:
    - The function first calls the `synchronize` method on the `ctx` object to ensure that any pending computations are completed.
    - After synchronization, it calls the `get_embeddings` method on the `ctx` object to retrieve the embeddings.
- **Output**: Returns a pointer to a float array containing the embeddings from the `llama_context`.


---
### llama\_get\_embeddings\_ith<!-- {{#callable:llama_get_embeddings_ith}} -->
The `llama_get_embeddings_ith` function retrieves the i-th embedding from the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the state and parameters for the model.
    - `i`: An integer index specifying which embedding to retrieve.
- **Control Flow**:
    - The function first calls `ctx->synchronize()` to ensure that any pending computations are completed before accessing the embeddings.
    - It then calls `ctx->get_embeddings_ith(i)` to retrieve the i-th embedding from the context.
- **Output**: Returns a pointer to a float array representing the i-th embedding.


---
### llama\_get\_embeddings\_seq<!-- {{#callable:llama_get_embeddings_seq}} -->
The `llama_get_embeddings_seq` function retrieves the embeddings for a specified sequence ID from the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the context for the Llama model.
    - `seq_id`: A `llama_seq_id` representing the identifier of the sequence for which embeddings are to be retrieved.
- **Control Flow**:
    - The function first calls the `synchronize` method on the `ctx` to ensure that any pending computations are completed.
    - It then calls the `get_embeddings_seq` method of the `ctx` with the provided `seq_id` to retrieve the corresponding embeddings.
- **Output**: Returns a pointer to a float array containing the embeddings for the specified sequence ID, or nullptr if the sequence ID is not found.


---
### llama\_set\_adapter\_lora<!-- {{#callable:llama_set_adapter_lora}} -->
Sets a LoRA adapter for the given llama context with a specified scaling factor.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure that represents the current context.
    - `adapter`: A pointer to the `llama_adapter_lora` structure representing the LoRA adapter to be set.
    - `scale`: A float value representing the scaling factor to be applied to the adapter.
- **Control Flow**:
    - The function calls the `set_adapter_lora` method on the `ctx` object, passing the `adapter` and `scale` as arguments.
    - The function does not perform any checks or validations on the inputs.
    - After setting the adapter, the function returns a success code.
- **Output**: Returns 0 to indicate successful execution.


---
### llama\_rm\_adapter\_lora<!-- {{#callable:llama_rm_adapter_lora}} -->
The `llama_rm_adapter_lora` function removes a specified LoRA adapter from the `llama_context` and returns a status code indicating success or failure.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that represents the current context in which the LoRA adapter is being removed.
    - `adapter`: A pointer to a `llama_adapter_lora` structure that represents the LoRA adapter to be removed from the context.
- **Control Flow**:
    - The function calls the `rm_adapter_lora` method on the `ctx` object, passing the `adapter` as an argument.
    - The result of the `rm_adapter_lora` call is stored in the `res` variable, which is a boolean indicating whether the removal was successful.
    - The function then returns 0 if `res` is true, indicating success, or -1 if `res` is false, indicating failure.
- **Output**: The function returns an integer status code: 0 for successful removal of the adapter, and -1 for failure.


---
### llama\_clear\_adapter\_lora<!-- {{#callable:llama_clear_adapter_lora}} -->
The `llama_clear_adapter_lora` function clears all LORA adapters associated with the given `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object that contains the state and configuration for the Llama model.
- **Control Flow**:
    - The function calls the `clear_adapter_lora` method on the `ctx` object.
    - This method is responsible for removing all LORA adapters from the context.
- **Output**: The function does not return any value; it performs an action that modifies the state of the `llama_context` by clearing its LORA adapters.


---
### llama\_apply\_adapter\_cvec<!-- {{#callable:llama_apply_adapter_cvec}} -->
Applies an adapter represented as a context vector to the llama model's context.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure that holds the model's context.
    - `data`: A pointer to an array of floats representing the adapter context vector to be applied.
    - `len`: The length of the `data` array.
    - `n_embd`: The dimensionality of the embeddings.
    - `il_start`: The starting index for applying the adapter.
    - `il_end`: The ending index for applying the adapter.
- **Control Flow**:
    - The function first calls the `apply_adapter_cvec` method of the `llama_context` instance, passing the input parameters.
    - The result of the adapter application is stored in a boolean variable `res`.
    - The function then returns 0 if the adapter application was successful (i.e., `res` is true), otherwise it returns -1.
- **Output**: Returns 0 on successful application of the adapter context vector, or -1 if the application fails.


---
### llama\_get\_memory<!-- {{#callable:llama_get_memory}} -->
The `llama_get_memory` function retrieves the memory associated with a given `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a constant `llama_context` structure, which contains the context for the Llama model.
- **Control Flow**:
    - The function directly accesses the `get_memory` method of the `ctx` object.
    - It returns the result of the `get_memory` method call.
- **Output**: The function returns a `llama_memory_t` type, which represents the memory associated with the provided context.


---
### llama\_memory\_clear<!-- {{#callable:llama_memory_clear}} -->
The `llama_memory_clear` function clears the memory associated with a given `llama_memory_t` object.
- **Inputs**:
    - `mem`: A pointer to a `llama_memory_t` object whose memory is to be cleared.
- **Control Flow**:
    - The function calls the `clear` method on the `mem` object to perform the clearing operation.
- **Output**: The function does not return any value; it performs an in-place operation to clear the memory.


---
### llama\_memory\_seq\_rm<!-- {{#callable:llama_memory_seq_rm}} -->
Removes a sequence from memory within a specified range.
- **Inputs**:
    - `mem`: A pointer to the `llama_memory_t` structure representing the memory from which the sequence will be removed.
    - `seq_id`: An identifier for the sequence to be removed.
    - `p0`: The starting position of the range to be removed.
    - `p1`: The ending position of the range to be removed.
- **Control Flow**:
    - The function calls the `seq_rm` method on the `mem` object, passing the `seq_id`, `p0`, and `p1` as arguments.
    - The `seq_rm` method is responsible for the actual removal of the sequence from memory.
- **Output**: Returns a boolean indicating whether the removal operation was successful.


---
### llama\_memory\_seq\_cp<!-- {{#callable:llama_memory_seq_cp}} -->
Copies a sequence of memory from one sequence ID to another within the specified memory.
- **Inputs**:
    - `mem`: A pointer to the `llama_memory_t` structure representing the memory context.
    - `seq_id_src`: The source sequence ID from which memory will be copied.
    - `seq_id_dst`: The destination sequence ID to which memory will be copied.
    - `p0`: The starting position in the source sequence from which to begin copying.
    - `p1`: The ending position in the source sequence up to which to copy.
- **Control Flow**:
    - The function calls the `seq_cp` method on the `mem` object.
    - It passes the source sequence ID, destination sequence ID, and the positions p0 and p1 to the `seq_cp` method.
- **Output**: The function does not return a value; it performs the copy operation directly on the memory.


---
### llama\_memory\_seq\_keep<!-- {{#callable:llama_memory_seq_keep}} -->
The `llama_memory_seq_keep` function retains a specified sequence in the memory.
- **Inputs**:
    - `mem`: A pointer to a `llama_memory_t` structure representing the memory context.
    - `seq_id`: An identifier of the sequence to be retained in memory.
- **Control Flow**:
    - The function calls the `seq_keep` method on the `mem` object, passing the `seq_id` as an argument.
    - No conditional logic or loops are present; the function directly invokes the method.
- **Output**: The function does not return a value; it performs an action to retain the specified sequence in memory.


---
### llama\_memory\_seq\_add<!-- {{#callable:llama_memory_seq_add}} -->
Adds a sequence to the memory with specified positions and a delta.
- **Inputs**:
    - `mem`: A pointer to the `llama_memory_t` structure representing the memory where the sequence will be added.
    - `seq_id`: An identifier for the sequence being added.
    - `p0`: The starting position in the sequence.
    - `p1`: The ending position in the sequence.
    - `delta`: The delta value to be added to the sequence.
- **Control Flow**:
    - The function calls the `seq_add` method of the `llama_memory_t` structure.
    - It passes the `seq_id`, `p0`, `p1`, and `delta` parameters to the `seq_add` method.
- **Output**: The function does not return a value; it modifies the state of the memory by adding the specified sequence.


---
### llama\_memory\_seq\_div<!-- {{#callable:llama_memory_seq_div}} -->
The `llama_memory_seq_div` function divides a specified sequence in memory by a given integer.
- **Inputs**:
    - `mem`: A pointer to a `llama_memory_t` structure representing the memory context.
    - `seq_id`: An identifier for the sequence to be modified.
    - `p0`: The starting position in the sequence.
    - `p1`: The ending position in the sequence.
    - `d`: The integer by which the sequence will be divided.
- **Control Flow**:
    - The function calls the `seq_div` method on the `mem` object.
    - It passes the `seq_id`, `p0`, `p1`, and `d` parameters to the `seq_div` method.
- **Output**: The function does not return a value; it modifies the sequence in memory directly.


---
### llama\_memory\_seq\_pos\_min<!-- {{#callable:llama_memory_seq_pos_min}} -->
The `llama_memory_seq_pos_min` function retrieves the minimum position of a specified sequence in the memory.
- **Inputs**:
    - `mem`: A pointer to a `llama_memory_t` structure representing the memory context.
    - `seq_id`: An identifier for the sequence whose minimum position is to be retrieved.
- **Control Flow**:
    - The function calls the `seq_pos_min` method on the `mem` object, passing the `seq_id` as an argument.
    - The result of the `seq_pos_min` method is returned directly.
- **Output**: Returns the minimum position of the specified sequence as a `llama_pos` type.


---
### llama\_memory\_seq\_pos\_max<!-- {{#callable:llama_memory_seq_pos_max}} -->
The `llama_memory_seq_pos_max` function retrieves the maximum position of a specified sequence in the memory.
- **Inputs**:
    - `mem`: A pointer to a `llama_memory_t` structure representing the memory from which to retrieve the sequence position.
    - `seq_id`: An identifier for the sequence whose maximum position is to be retrieved.
- **Control Flow**:
    - The function directly calls the `seq_pos_max` method on the `mem` object, passing the `seq_id` as an argument.
    - The result of the `seq_pos_max` method is returned as the output of the function.
- **Output**: Returns a `llama_pos` value representing the maximum position of the specified sequence in the memory.


---
### llama\_memory\_can\_shift<!-- {{#callable:llama_memory_can_shift}} -->
Checks if the memory can shift.
- **Inputs**:
    - `mem`: A pointer to a `llama_memory_t` structure representing the memory to be checked.
- **Control Flow**:
    - Calls the `get_can_shift()` method on the `mem` object to determine if shifting is possible.
- **Output**: Returns a boolean value indicating whether the memory can shift.


---
### llama\_kv\_self\_n\_tokens<!-- {{#callable:llama_kv_self_n_tokens}} -->
Calculates the total number of tokens stored in the key-value memory for the given context.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that contains the context for the computation.
- **Control Flow**:
    - Retrieve the key-value memory associated with the context using `llama_get_memory(ctx)`.
    - If the key-value memory is null, return 0.
    - Initialize a result counter `res` to 0.
    - Iterate over each sequence index from 0 to `n_seq_max` (maximum number of sequences).
    - For each sequence index, get the minimum and maximum positions in the key-value memory using `kv->seq_pos_min(s)` and `kv->seq_pos_max(s)`.
    - If the minimum position is valid (greater than or equal to 0), calculate the number of tokens for that sequence and add it to `res`.
    - After the loop, return the total count of tokens stored in `res`.
- **Output**: Returns an integer representing the total number of tokens stored in the key-value memory.
- **Functions called**:
    - [`llama_get_memory`](#llama_get_memory)


---
### llama\_kv\_self\_used\_cells<!-- {{#callable:llama_kv_self_used_cells}} -->
Calculates the number of used cells in the key-value memory for the given `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that contains the context for the llama model.
- **Control Flow**:
    - Retrieve the key-value memory associated with the provided `ctx` using [`llama_get_memory`](#llama_get_memory).
    - If the key-value memory is null, return 0.
    - Initialize a result variable `res` to 0.
    - Iterate over the sequence indices from 0 to `n_seq_max` (maximum number of sequences).
    - For each sequence index, retrieve the minimum and maximum positions in the key-value memory using `seq_pos_min` and `seq_pos_max`.
    - If the minimum position is non-negative, calculate the number of used cells for that sequence and add it to `res`.
    - After the loop, return the total count of used cells stored in `res`.
- **Output**: Returns an integer representing the total number of used cells in the key-value memory.
- **Functions called**:
    - [`llama_get_memory`](#llama_get_memory)


---
### llama\_kv\_self\_clear<!-- {{#callable:llama_kv_self_clear}} -->
The `llama_kv_self_clear` function clears the key-value memory cache associated with a given `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that contains the context for the Llama model, including its memory management.
- **Control Flow**:
    - The function retrieves the memory associated with the provided `llama_context` using `llama_get_memory(ctx)`.
    - If the retrieved memory pointer `kv` is null (indicating no memory is allocated), the function returns immediately without performing any action.
    - If `kv` is valid, the function calls `llama_memory_clear(kv)` to clear the memory cache.
- **Output**: The function does not return a value; it performs an in-place operation to clear the memory cache.
- **Functions called**:
    - [`llama_get_memory`](#llama_get_memory)
    - [`llama_memory_clear`](#llama_memory_clear)


---
### llama\_kv\_self\_seq\_rm<!-- {{#callable:llama_kv_self_seq_rm}} -->
Removes a sequence from the key-value memory cache.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure, which holds the context for the Llama model.
    - `seq_id`: An identifier for the sequence to be removed from the memory.
    - `p0`: The starting position of the sequence to be removed.
    - `p1`: The ending position of the sequence to be removed.
- **Control Flow**:
    - The function retrieves the memory associated with the given context using `llama_get_memory(ctx)`.
    - If the memory is not available (i.e., `kv` is null), the function returns true, indicating no action is needed.
    - If memory is available, it calls `llama_memory_seq_rm(kv, seq_id, p0, p1)` to remove the specified sequence from the memory.
- **Output**: Returns a boolean value indicating the success of the sequence removal operation.
- **Functions called**:
    - [`llama_get_memory`](#llama_get_memory)
    - [`llama_memory_seq_rm`](#llama_memory_seq_rm)


---
### llama\_kv\_self\_seq\_cp<!-- {{#callable:llama_kv_self_seq_cp}} -->
Copies a sequence of memory from one sequence ID to another within the context's memory.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure, which holds the state and configuration for the LLaMA model.
    - `seq_id_src`: The source sequence ID from which the memory will be copied.
    - `seq_id_dst`: The destination sequence ID to which the memory will be copied.
    - `p0`: The starting position in the source sequence from which to begin copying.
    - `p1`: The ending position in the source sequence up to which to copy.
- **Control Flow**:
    - The function retrieves the memory associated with the provided `ctx` using [`llama_get_memory`](#llama_get_memory).
    - If the memory is not available (i.e., `kv` is null), the function returns immediately without performing any operations.
    - If memory is available, it calls [`llama_memory_seq_cp`](#llama_memory_seq_cp) to perform the actual copy operation from the source sequence to the destination sequence.
- **Output**: The function does not return a value; it performs an in-place operation to copy memory from one sequence to another.
- **Functions called**:
    - [`llama_get_memory`](#llama_get_memory)
    - [`llama_memory_seq_cp`](#llama_memory_seq_cp)


---
### llama\_kv\_self\_seq\_keep<!-- {{#callable:llama_kv_self_seq_keep}} -->
The `llama_kv_self_seq_keep` function keeps a specified sequence in the key-value memory.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the context for the Llama model.
    - `seq_id`: An identifier for the sequence to be kept in memory.
- **Control Flow**:
    - The function retrieves the memory associated with the provided context using [`llama_get_memory`](#llama_get_memory).
    - If the memory is not available (i.e., `kv` is null), the function returns immediately.
    - If memory is available, it calls [`llama_memory_seq_keep`](#llama_memory_seq_keep) to keep the specified sequence identified by `seq_id`.
- **Output**: The function does not return a value; it modifies the state of the memory by keeping the specified sequence.
- **Functions called**:
    - [`llama_get_memory`](#llama_get_memory)
    - [`llama_memory_seq_keep`](#llama_memory_seq_keep)


---
### llama\_kv\_self\_seq\_add<!-- {{#callable:llama_kv_self_seq_add}} -->
Adds a sequence to the key-value memory cache in the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure that holds the context for the operation.
    - `seq_id`: An identifier for the sequence being added to the memory.
    - `p0`: The starting position in the sequence for the addition.
    - `p1`: The ending position in the sequence for the addition.
    - `delta`: The value to be added to the sequence in the memory.
- **Control Flow**:
    - Retrieve the memory associated with the given `ctx` using [`llama_get_memory`](#llama_get_memory).
    - If the memory is not available (i.e., `kv` is null), the function returns immediately.
    - If memory is available, call [`llama_memory_seq_add`](#llama_memory_seq_add) to add the sequence defined by `seq_id`, `p0`, `p1`, and `delta`.
- **Output**: The function does not return a value; it modifies the state of the memory cache by adding the specified sequence.
- **Functions called**:
    - [`llama_get_memory`](#llama_get_memory)
    - [`llama_memory_seq_add`](#llama_memory_seq_add)


---
### llama\_kv\_self\_seq\_div<!-- {{#callable:llama_kv_self_seq_div}} -->
This function divides a specified sequence in the key-value memory of a llama context.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure, which holds the state and parameters for the llama model.
    - `seq_id`: An identifier for the sequence to be divided within the key-value memory.
    - `p0`: The starting position in the sequence from which to begin the division.
    - `p1`: The ending position in the sequence up to which the division should occur.
    - `d`: An integer that specifies the divisor for the division operation.
- **Control Flow**:
    - The function retrieves the key-value memory associated with the provided `ctx` using [`llama_get_memory`](#llama_get_memory).
    - If the memory is not available (i.e., `kv` is null), the function returns immediately without performing any operations.
    - If the memory is available, it calls [`llama_memory_seq_div`](#llama_memory_seq_div) to perform the division operation on the specified sequence from `p0` to `p1` using the divisor `d`.
- **Output**: The function does not return a value; it modifies the state of the key-value memory directly.
- **Functions called**:
    - [`llama_get_memory`](#llama_get_memory)
    - [`llama_memory_seq_div`](#llama_memory_seq_div)


---
### llama\_kv\_self\_seq\_pos\_min<!-- {{#callable:llama_kv_self_seq_pos_min}} -->
The `llama_kv_self_seq_pos_min` function retrieves the minimum position of a specified sequence ID from the key-value memory.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the context for the llama model.
    - `seq_id`: A `llama_seq_id` representing the sequence ID for which the minimum position is to be retrieved.
- **Control Flow**:
    - The function first calls [`llama_get_memory`](#llama_get_memory) with the context `ctx` to obtain a pointer to the key-value memory.
    - If the memory pointer `kv` is null, the function returns -1, indicating an error.
    - If the memory is valid, it calls [`llama_memory_seq_pos_min`](#llama_memory_seq_pos_min) with the memory pointer `kv` and the provided `seq_id` to get the minimum position.
- **Output**: The function returns the minimum position of the specified sequence ID, or -1 if the memory is not available.
- **Functions called**:
    - [`llama_get_memory`](#llama_get_memory)
    - [`llama_memory_seq_pos_min`](#llama_memory_seq_pos_min)


---
### llama\_kv\_self\_seq\_pos\_max<!-- {{#callable:llama_kv_self_seq_pos_max}} -->
This function retrieves the maximum position of a specified sequence ID from the key-value memory.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the context for the llama model.
    - `seq_id`: A `llama_seq_id` representing the sequence ID for which the maximum position is to be retrieved.
- **Control Flow**:
    - The function first retrieves the key-value memory associated with the provided context using `llama_get_memory(ctx)`.
    - If the key-value memory is not available (i.e., `kv` is null), the function returns -1 to indicate an error.
    - If the key-value memory is successfully retrieved, it calls `llama_memory_seq_pos_max(kv, seq_id)` to get the maximum position for the specified sequence ID.
- **Output**: The function returns the maximum position of the specified sequence ID if successful, or -1 if there was an error retrieving the key-value memory.
- **Functions called**:
    - [`llama_get_memory`](#llama_get_memory)
    - [`llama_memory_seq_pos_max`](#llama_memory_seq_pos_max)


---
### llama\_kv\_self\_defrag<!-- {{#callable:llama_kv_self_defrag}} -->
The `llama_kv_self_defrag` function triggers the self-defragmentation of the key-value memory cache in the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object that contains the state and configuration for the model.
- **Control Flow**:
    - The function calls the `kv_self_defrag_sched` method on the provided `ctx` object to initiate the defragmentation process.
- **Output**: The function does not return a value; it performs an action to optimize memory usage.


---
### llama\_kv\_self\_can\_shift<!-- {{#callable:llama_kv_self_can_shift}} -->
Checks if the key-value memory can shift.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the context for the Llama model.
- **Control Flow**:
    - Retrieve the key-value memory associated with the provided context using [`llama_get_memory`](#llama_get_memory).
    - If the key-value memory is null, return false.
    - Call [`llama_memory_can_shift`](#llama_memory_can_shift) with the key-value memory and return its result.
- **Output**: Returns a boolean indicating whether the key-value memory can shift.
- **Functions called**:
    - [`llama_get_memory`](#llama_get_memory)
    - [`llama_memory_can_shift`](#llama_memory_can_shift)


---
### llama\_get\_state\_size<!-- {{#callable:llama_get_state_size}} -->
The `llama_get_state_size` function retrieves the size of the state associated with a given `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, which contains the context for the Llama model.
- **Control Flow**:
    - The function calls [`llama_state_get_size`](#llama_state_get_size) with the provided `ctx` argument.
    - The result from [`llama_state_get_size`](#llama_state_get_size) is returned directly.
- **Output**: Returns the size of the state as a `size_t` value.
- **Functions called**:
    - [`llama_state_get_size`](#llama_state_get_size)


---
### llama\_copy\_state\_data<!-- {{#callable:llama_copy_state_data}} -->
Copies the state data from a `llama_context` to a specified destination buffer.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the state data to be copied.
    - `dst`: A pointer to a buffer where the state data will be copied.
- **Control Flow**:
    - The function calls [`llama_state_get_data`](#llama_state_get_data) with the context `ctx`, the destination buffer `dst`, and a size parameter of -1.
    - The size parameter of -1 indicates that the function should determine the size of the state data to copy automatically.
- **Output**: Returns the size of the copied state data.
- **Functions called**:
    - [`llama_state_get_data`](#llama_state_get_data)


---
### llama\_set\_state\_data<!-- {{#callable:llama_set_state_data}} -->
Sets the state data of the `llama_context` from a source buffer.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure that holds the state to be set.
    - `src`: A pointer to a buffer containing the source data to set the state from.
- **Control Flow**:
    - The function first calls [`llama_state_set_data`](#llama_state_set_data) with the context pointer `ctx`, the source pointer `src`, and a size of -1.
    - The size of -1 indicates that the function should determine the size of the data to be set automatically.
- **Output**: Returns the size of the data that was set in the state.
- **Functions called**:
    - [`llama_state_set_data`](#llama_state_set_data)


---
### llama\_load\_session\_file<!-- {{#callable:llama_load_session_file}} -->
The `llama_load_session_file` function loads a session state from a specified file into a given context.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the current state of the model.
    - `path_session`: A string representing the file path from which the session state will be loaded.
    - `tokens_out`: A pointer to an array of `llama_token` where the loaded tokens will be stored.
    - `n_token_capacity`: A size_t value indicating the maximum number of tokens that can be stored in the `tokens_out` array.
    - `n_token_count_out`: A pointer to a size_t variable that will hold the actual number of tokens loaded.
- **Control Flow**:
    - The function first calls [`llama_state_load_file`](#llama_state_load_file), passing the context, session file path, output tokens array, token capacity, and a pointer to the token count variable.
    - If [`llama_state_load_file`](#llama_state_load_file) returns true, the session state is successfully loaded; otherwise, it indicates a failure.
- **Output**: Returns a boolean value indicating whether the session state was successfully loaded from the file.
- **Functions called**:
    - [`llama_state_load_file`](#llama_state_load_file)


---
### llama\_save\_session\_file<!-- {{#callable:llama_save_session_file}} -->
The `llama_save_session_file` function saves the current session state to a specified file.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the current state of the Llama model.
    - `path_session`: A string representing the file path where the session state will be saved.
    - `tokens`: An array of `llama_token` representing the tokens associated with the session.
    - `n_token_count`: A size_t value indicating the number of tokens to be saved.
- **Control Flow**:
    - The function calls [`llama_state_save_file`](#llama_state_save_file), passing the same parameters to save the session state.
    - It returns the result of the [`llama_state_save_file`](#llama_state_save_file) function, which indicates success or failure.
- **Output**: Returns a boolean value indicating whether the session was successfully saved.
- **Functions called**:
    - [`llama_state_save_file`](#llama_state_save_file)


---
### llama\_state\_get\_size<!-- {{#callable:llama_state_get_size}} -->
The `llama_state_get_size` function retrieves the size of the state associated with a given `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, which contains the state information.
- **Control Flow**:
    - The function calls the `state_get_size` method on the `llama_context` instance pointed to by `ctx`.
    - It returns the size of the state as determined by the `state_get_size` method.
- **Output**: Returns the size of the state as a `size_t` value.


---
### llama\_state\_get\_data<!-- {{#callable:llama_state_get_data}} -->
Retrieves the state data from the `llama_context` and writes it to the provided destination buffer.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` from which the state data will be retrieved.
    - `dst`: A pointer to a buffer where the state data will be written.
    - `size`: The size of the destination buffer.
- **Control Flow**:
    - The function first calls `ctx->synchronize()` to ensure that any pending operations are completed before accessing the state data.
    - It then calls `ctx->state_get_data(dst, size)` to retrieve the state data and write it to the provided buffer.
    - The return value of `state_get_data` is returned as the output of this function.
- **Output**: Returns the size of the state data that was written to the destination buffer.


---
### llama\_state\_set\_data<!-- {{#callable:llama_state_set_data}} -->
Sets the state data of the `llama_context` from a source buffer.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure that holds the state to be set.
    - `src`: A pointer to a buffer containing the source data to set the state from.
    - `size`: The size of the source data buffer.
- **Control Flow**:
    - The function first calls `ctx->synchronize()` to ensure that any previous operations on the context are completed before setting new state data.
    - It then calls `ctx->state_set_data(src, size)` to actually set the state data from the provided source buffer.
- **Output**: Returns the size of the data that was successfully set in the state.


---
### llama\_state\_load\_file<!-- {{#callable:llama_state_load_file}} -->
Loads the state of a `llama_context` from a specified session file.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure that holds the state to be loaded.
    - `path_session`: A string representing the file path to the session file that contains the saved state.
    - `tokens_out`: An output parameter that will hold the loaded tokens.
    - `n_token_capacity`: The maximum number of tokens that can be stored in the `tokens_out` array.
    - `n_token_count_out`: An output parameter that will store the actual number of tokens loaded.
- **Control Flow**:
    - The function begins by calling `ctx->synchronize()` to ensure that any previous operations on the context are completed.
    - It then attempts to load the state from the specified session file by calling `ctx->state_load_file()` with the provided parameters.
    - If the loading operation is successful, it returns true.
    - If an exception occurs during the loading process, it catches the exception, logs an error message, and returns false.
- **Output**: Returns true if the state was successfully loaded from the file, otherwise returns false.


---
### llama\_state\_save\_file<!-- {{#callable:llama_state_save_file}} -->
Saves the current state of the `llama_context` to a specified file.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` instance that holds the state to be saved.
    - `path_session`: A string representing the file path where the session state will be saved.
    - `tokens`: An array of `llama_token` representing the tokens associated with the session.
    - `n_token_count`: The number of tokens in the `tokens` array.
- **Control Flow**:
    - The function begins by calling `ctx->synchronize()` to ensure that all operations on the context are completed before saving.
    - It then attempts to save the state by calling `ctx->state_save_file(path_session, tokens, n_token_count)`.
    - If the save operation is successful, it returns true; otherwise, it catches any exceptions.
    - In case of an exception, it logs an error message and returns false.
- **Output**: Returns true if the state was successfully saved, otherwise returns false.


---
### llama\_state\_seq\_get\_size<!-- {{#callable:llama_state_seq_get_size}} -->
Retrieves the size of the state sequence for a given sequence ID.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the context for the Llama model.
    - `seq_id`: A `llama_seq_id` representing the specific sequence for which the size is being queried.
- **Control Flow**:
    - The function calls the `state_seq_get_size` method of the `llama_context` instance pointed to by `ctx`.
    - It passes the `seq_id` to this method to retrieve the size of the specified state sequence.
- **Output**: Returns the size of the state sequence as a `size_t` value.


---
### llama\_state\_seq\_get\_data<!-- {{#callable:llama_state_seq_get_data}} -->
Retrieves data from a specified sequence in the `llama_context` and stores it in a provided destination buffer.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure, which holds the state and configuration for the Llama model.
    - `dst`: A pointer to a buffer where the retrieved data will be stored.
    - `size`: The size of the buffer `dst`, indicating how much data can be written to it.
    - `seq_id`: An identifier for the specific sequence from which data is to be retrieved.
- **Control Flow**:
    - The function begins by calling `ctx->synchronize()` to ensure that any previous operations on the context are completed before proceeding.
    - It then calls the `state_seq_get_data` method of the `llama_context` to retrieve the data for the specified sequence identified by `seq_id`, writing the data into the `dst` buffer and respecting the specified `size`.
- **Output**: Returns the number of bytes of data that were successfully written to the `dst` buffer.


---
### llama\_state\_seq\_set\_data<!-- {{#callable:llama_state_seq_set_data}} -->
Sets the sequence state data in the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure that holds the state of the model.
    - `src`: A pointer to a buffer containing the source data to be set for the sequence.
    - `size`: The size of the data in bytes that is to be set in the sequence state.
    - `seq_id`: An identifier for the sequence whose state is being set.
- **Control Flow**:
    - The function first calls `ctx->synchronize()` to ensure that any previous operations on the context are completed before proceeding.
    - It then calls `ctx->state_seq_set_data(seq_id, src, size)` to set the data for the specified sequence ID using the provided source buffer and size.
- **Output**: Returns the size of the data that was successfully set in the sequence state.


---
### llama\_state\_seq\_save\_file<!-- {{#callable:llama_state_seq_save_file}} -->
The `llama_state_seq_save_file` function saves the state of a sequence to a specified file.
- **Inputs**:
    - `seq_id`: An identifier for the sequence whose state is being saved.
    - `filepath`: The path to the file where the sequence state will be saved.
    - `tokens`: An array of tokens associated with the sequence.
    - `n_token_count`: The number of tokens to be saved.
- **Control Flow**:
    - The function first attempts to synchronize the context using `ctx->synchronize()`.
    - It then attempts to call `ctx->state_seq_save_file` to save the sequence state.
    - If an exception occurs during the save operation, it logs an error message and returns 0.
- **Output**: The function returns the size of the saved state or 0 if an error occurred.


---
### llama\_state\_seq\_load\_file<!-- {{#callable:llama_state_seq_load_file}} -->
Loads a sequence state from a file into the specified context.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure that holds the state and configuration for the model.
    - `filepath`: A string representing the path to the file from which the sequence state will be loaded.
    - `dest_seq_id`: An identifier for the destination sequence where the state will be loaded.
    - `tokens_out`: An output parameter that will hold the loaded tokens after the file is read.
    - `n_token_capacity`: The maximum number of tokens that can be stored in the `tokens_out` array.
    - `n_token_count_out`: An output parameter that will store the actual number of tokens loaded from the file.
- **Control Flow**:
    - The function begins by synchronizing the context to ensure all previous operations are complete.
    - It attempts to read the sequence state from the specified file using the `state_seq_load_file` method of the context.
    - If the file reading operation is successful, it returns the number of tokens loaded.
    - If an exception occurs during the file reading, it logs an error message and returns 0.
- **Output**: Returns the number of tokens successfully loaded from the file, or 0 if an error occurred.


---
### llama\_encode<!-- {{#callable:llama_encode}} -->
Encodes a given batch of tokens using the `llama_context` and logs an error if the encoding fails.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object that contains the model and parameters for encoding.
    - `batch`: A `llama_batch` object that contains the tokens to be encoded.
- **Control Flow**:
    - Calls the `encode` method of the `llama_context` object with the provided batch.
    - Checks the return value of the `encode` method; if it is not zero, logs an error message indicating the failure.
- **Output**: Returns an integer indicating the result of the encoding operation, where a non-zero value indicates an error.


---
### llama\_decode<!-- {{#callable:llama_decode}} -->
The `llama_decode` function decodes a given batch of tokens using the provided context.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the state and parameters for decoding.
    - `batch`: A `llama_batch` structure containing the tokens to be decoded.
- **Control Flow**:
    - The function calls the `decode` method on the `ctx` object with the provided `batch`.
    - If the return value from `decode` is not 0 or 1, an error message is logged indicating the failure to decode.
    - The function then returns the result of the `decode` call.
- **Output**: The function returns an integer indicating the result of the decoding operation, where 0 indicates success, 1 indicates a partial success, and any other value indicates an error.


---
### llama\_perf\_context<!-- {{#callable:llama_perf_context}} -->
The `llama_perf_context` function retrieves performance data from a given `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which holds the performance metrics.
- **Control Flow**:
    - The function initializes a `llama_perf_context_data` structure to hold the performance data.
    - It checks if the input `ctx` is a null pointer; if it is, the function returns the initialized data structure without modification.
    - If `ctx` is valid, it calls the `perf_get_data` method on the `ctx` object to populate the performance data.
- **Output**: Returns a `llama_perf_context_data` structure containing the performance metrics of the specified `llama_context`.


---
### llama\_perf\_context\_print<!-- {{#callable:llama_perf_context_print}} -->
Prints performance metrics of the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that contains performance data.
- **Control Flow**:
    - Calls [`llama_perf_context`](#llama_perf_context) to retrieve performance data from the provided `ctx`.
    - Calculates the end time in milliseconds using `ggml_time_us()`.
    - Logs the load time, prompt evaluation time, evaluation time, and total time using `LLAMA_LOG_INFO`.
- **Output**: This function does not return a value; it outputs performance metrics to the log.
- **Functions called**:
    - [`llama_perf_context`](#llama_perf_context)


---
### llama\_perf\_context\_reset<!-- {{#callable:llama_perf_context_reset}} -->
Resets the performance metrics of the `llama_context`.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the performance metrics to be reset.
- **Control Flow**:
    - Calls the `perf_reset` method on the `ctx` object to reset its performance metrics.
- **Output**: This function does not return a value; it modifies the state of the `llama_context` by resetting its performance metrics.


---
### llama\_opt\_param\_filter\_all<!-- {{#callable:llama_opt_param_filter_all}} -->
The `llama_opt_param_filter_all` function is a filter that always returns true, indicating that all parameters should be included.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor to be filtered.
    - `userdata`: A pointer to user-defined data that can be used for additional context or information during filtering.
- **Control Flow**:
    - The function uses `GGML_UNUSED` to indicate that the `tensor` and `userdata` parameters are not utilized within the function body.
    - It directly returns `true`, indicating that the filtering condition is satisfied for all inputs.
- **Output**: The function returns a boolean value, specifically `true`, indicating that all parameters pass the filter.


---
### llama\_opt\_init<!-- {{#callable:llama_opt_init}} -->
Initializes the optimization parameters for the `llama_context` using the provided model and optimization parameters.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` structure that holds the context for the model.
    - `model`: A pointer to the `llama_model` structure that contains the model parameters and architecture.
    - `lopt_params`: A structure containing optimization parameters such as context size and callbacks.
- **Control Flow**:
    - The function first asserts that the optimization context (`opt_ctx`) is not already initialized.
    - It sets the training context size based on the provided optimization parameters or defaults to the context size of the model.
    - It calculates the batch sizes and ensures they are valid.
    - The function initializes the optimization parameters and sets up the optimization context.
    - It applies the parameter filter to various model parameters to set them for optimization.
- **Output**: The function does not return a value but initializes the optimization context for the `llama_context`.


---
### llama\_opt\_epoch<!-- {{#callable:llama_opt_epoch}} -->
The `llama_opt_epoch` function orchestrates the optimization process for a given epoch in the training of a model.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure that holds the context for the optimization process.
    - `dataset`: A `ggml_opt_dataset_t` structure representing the dataset used for training.
    - `result_train`: A `ggml_opt_result_t` structure to store the results of the training phase.
    - `result_eval`: A `ggml_opt_result_t` structure to store the results of the evaluation phase.
    - `idata_split`: An integer representing the index of the data split for the current epoch.
    - `callback_train`: A callback function to be executed during the training phase.
    - `callback_eval`: A callback function to be executed during the evaluation phase.
- **Control Flow**:
    - The function begins by asserting the validity of the input parameters.
    - It retrieves the maximum context size and batch sizes from the context.
    - The function initializes a batch structure to hold the tokens and labels for the current epoch.
    - It enters a loop to process the dataset, iterating through the data splits.
    - For each data split, it retrieves a batch of tokens and labels, and calls the `opt_epoch_iter` function to perform the optimization for that batch.
    - The training and evaluation results are updated based on the callbacks provided.
- **Output**: The function does not return a value; instead, it updates the training and evaluation results through the provided result structures.


---
### llama\_io\_write\_i<!-- {{#callable:llama_io_write_i::llama_io_write_i}} -->
The `llama_io_write_i` class provides an interface for writing data to various output streams, including buffers and files.
- **Inputs**:
    - `src`: A pointer to the source data that needs to be written.
    - `size`: The size in bytes of the data to be written.
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor data to be written.
    - `offset`: The offset in the tensor from which to start writing.
- **Control Flow**:
    - The `write` method checks if the size exceeds the buffer size and throws an error if it does.
    - If the size is valid, it copies the data from the source pointer to the internal buffer.
    - The `write_tensor` method retrieves the tensor data and writes it to the buffer, also checking for size constraints.
- **Output**: The output is the total number of bytes written to the output stream, which can be retrieved using the `n_bytes` method.


---
### llama\_io\_read\_i<!-- {{#callable:llama_io_read_i::llama_io_read_i}} -->
The `llama_io_read_i` class provides an interface for reading data from a buffer, allowing for reading raw bytes and tensors.
- **Inputs**:
    - `src`: A pointer to the source buffer from which data will be read.
    - `size`: The size in bytes of the data to be read from the buffer.
- **Control Flow**:
    - The method first checks if the requested size exceeds the remaining buffer size.
    - If the size is valid, it updates the internal pointer to reflect the bytes read and returns the pointer to the data.
- **Output**: The method returns a pointer to the data read from the buffer.


