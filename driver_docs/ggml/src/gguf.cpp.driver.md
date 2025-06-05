# Purpose
The provided C++ source code file is a comprehensive implementation of a system for handling GGUF (Generic Graphical User Format) files, which are likely used for storing and managing data related to graphical or tensor-based applications. The code defines a variety of structures and functions to facilitate the reading, writing, and manipulation of GGUF files, focusing on key-value pairs and tensor data. The file includes several template specializations to map C++ types to GGUF types, ensuring type safety and consistency when handling data. The code also defines a `gguf_context` structure to maintain the state of a GGUF file, including its version, key-value pairs, tensor information, and data alignment.

Key components of the code include the [`gguf_kv`](#gguf_kvgguf_kv) structure for managing key-value pairs, the `gguf_tensor_info` structure for storing tensor metadata, and the [`gguf_reader`](#gguf_readergguf_reader) and [`gguf_writer`](#gguf_writergguf_writer) classes for reading from and writing to GGUF files, respectively. The code provides a public API for initializing GGUF contexts from files, retrieving and setting key-value pairs, managing tensor data, and writing GGUF data back to files. The use of assertions and error logging throughout the code ensures robustness and aids in debugging. Overall, this file serves as a library for applications that need to interact with GGUF files, providing a structured and efficient way to manage complex data formats.
# Imports and Dependencies

---
- `ggml.h`
- `ggml-backend.h`
- `ggml-impl.h`
- `gguf.h`
- `cinttypes`
- `cstddef`
- `cstdint`
- `cstdio`
- `cstdlib`
- `cstring`
- `map`
- `new`
- `stdexcept`
- `string`
- `vector`


# Global Variables

---
### GGUF\_TYPE\_SIZE
- **Type**: `std::map<gguf_type, size_t>`
- **Description**: `GGUF_TYPE_SIZE` is a static constant map that associates each `gguf_type` enumeration value with its corresponding size in bytes. The map includes various data types such as integers, floats, and booleans, with special cases for strings and arrays where the size is set to 0, indicating undefined size.
- **Use**: This map is used to determine the size of different `gguf_type` data types in bytes, which is essential for memory allocation and data handling operations.


---
### GGUF\_TYPE\_NAME
- **Type**: `std::map<gguf_type, const char *>`
- **Description**: `GGUF_TYPE_NAME` is a static constant map that associates `gguf_type` enumeration values with their corresponding string representations. This map is used to provide a human-readable name for each type defined in the `gguf_type` enumeration.
- **Use**: This variable is used to retrieve the string representation of a `gguf_type` when needed, such as for logging or displaying type information.


# Data Structures

---
### gguf\_kv<!-- {{#data_structure:gguf_kv}} -->
- **Type**: `struct`
- **Members**:
    - `key`: A string representing the key for the key-value pair.
    - `is_array`: A boolean indicating whether the value is an array.
    - `type`: An enum representing the type of the value.
    - `data`: A vector of int8_t storing the raw data for non-string types.
    - `data_string`: A vector of strings storing the data for string types.
- **Description**: The `gguf_kv` struct is a versatile data structure designed to store key-value pairs where the value can be of various types, including primitive types and strings, and can also be an array. It supports different constructors to initialize the key-value pair with a single value or an array of values, and it uses templates to determine the type of the value. The struct provides methods to retrieve the key, type, and number of elements, as well as to access the value at a specific index. It also includes a method to cast the value to a different type, ensuring that the data size is compatible with the new type.
- **Member Functions**:
    - [`gguf_kv::gguf_kv`](#gguf_kvgguf_kv)
    - [`gguf_kv::gguf_kv`](#gguf_kvgguf_kv)
    - [`gguf_kv::gguf_kv`](#gguf_kvgguf_kv)
    - [`gguf_kv::gguf_kv`](#gguf_kvgguf_kv)
    - [`gguf_kv::get_key`](#gguf_kvget_key)
    - [`gguf_kv::get_type`](#gguf_kvget_type)
    - [`gguf_kv::get_ne`](#gguf_kvget_ne)
    - [`gguf_kv::get_val`](#gguf_kvget_val)
    - [`gguf_kv::cast`](#gguf_kvcast)

**Methods**

---
#### gguf\_kv::gguf\_kv<!-- {{#callable:gguf_kv::gguf_kv}} -->
The `gguf_kv` constructor initializes a key-value pair with a given key and a single value of any type, storing the value in a byte vector.
- **Inputs**:
    - `key`: A constant reference to a `std::string` representing the key for the key-value pair.
    - `value`: A value of any type `T` to be stored in the key-value pair.
- **Control Flow**:
    - The constructor initializes the `key` member with the provided key and sets `is_array` to `false` indicating a single value.
    - It determines the type of the value using the `type_to_gguf_type` template specialization and assigns it to the `type` member.
    - An assertion checks that the key is not empty.
    - The `data` vector is resized to the size of the type `T`.
    - The value is copied into the `data` vector using `memcpy`.
- **Output**: The constructor does not return a value as it is used to initialize an instance of the `gguf_kv` struct.
- **See also**: [`gguf_kv`](#gguf_kv)  (Data Structure)


---
#### gguf\_kv::gguf\_kv<!-- {{#callable:gguf_kv::gguf_kv}} -->
The `gguf_kv` constructor initializes a key-value pair with a key and a vector of values, storing the values in a byte array.
- **Inputs**:
    - `key`: A string representing the key for the key-value pair, which must not be empty.
    - `value`: A vector of type T containing the values to be stored in the key-value pair.
- **Control Flow**:
    - The constructor initializes the `key` member with the provided key and sets `is_array` to true.
    - It determines the type of the values using the `type_to_gguf_type` template specialization and assigns it to the `type` member.
    - An assertion checks that the key is not empty.
    - The `data` vector is resized to accommodate the byte representation of all elements in the `value` vector.
    - A loop iterates over each element in the `value` vector, copying its byte representation into the `data` vector using `memcpy`.
- **Output**: The constructor does not return a value; it initializes the `gguf_kv` object with the provided key and vector of values.
- **See also**: [`gguf_kv`](#gguf_kv)  (Data Structure)


---
#### gguf\_kv::gguf\_kv<!-- {{#callable:gguf_kv::gguf_kv}} -->
The `gguf_kv` constructor initializes a key-value pair with a string key and a string value, storing the value in a vector of strings.
- **Inputs**:
    - `key`: A constant reference to a `std::string` representing the key for the key-value pair.
    - `value`: A constant reference to a `std::string` representing the value to be associated with the key.
- **Control Flow**:
    - The constructor initializes the `key` member with the provided key string.
    - It sets `is_array` to `false` indicating that the value is not an array.
    - The `type` is set to `GGUF_TYPE_STRING`, indicating the type of the value is a string.
    - An assertion checks that the key is not empty using `GGML_ASSERT`.
    - The provided value is added to the `data_string` vector.
- **Output**: The constructor does not return a value; it initializes the object state.
- **See also**: [`gguf_kv`](#gguf_kv)  (Data Structure)


---
#### gguf\_kv::gguf\_kv<!-- {{#callable:gguf_kv::gguf_kv}} -->
The `gguf_kv` constructor initializes a key-value pair with a string key and a vector of strings as the value, marking it as an array of strings.
- **Inputs**:
    - `key`: A constant reference to a `std::string` representing the key for the key-value pair.
    - `value`: A constant reference to a `std::vector<std::string>` representing the value associated with the key, which is a collection of strings.
- **Control Flow**:
    - The constructor initializes the `key` member with the provided key argument.
    - It sets the `is_array` member to `true`, indicating that the value is an array.
    - The `type` member is set to `GGUF_TYPE_STRING`, indicating the type of the value is a string.
    - An assertion checks that the key is not empty using `GGML_ASSERT(!key.empty())`.
    - The `data_string` member is assigned the provided vector of strings, `value`.
- **Output**: The function does not return a value as it is a constructor for the `gguf_kv` struct.
- **See also**: [`gguf_kv`](#gguf_kv)  (Data Structure)


---
#### gguf\_kv::get\_key<!-- {{#callable:gguf_kv::get_key}} -->
The `get_key` function returns a constant reference to the `key` member of the `gguf_kv` structure.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the `key` member of the `gguf_kv` structure.
- **Output**: A constant reference to a `std::string` representing the key of the key-value pair.
- **See also**: [`gguf_kv`](#gguf_kv)  (Data Structure)


---
#### gguf\_kv::get\_type<!-- {{#callable:gguf_kv::get_type}} -->
The `get_type` function returns the type of the `gguf_kv` object as an enumeration of `gguf_type`.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the `type` member of the `gguf_kv` structure.
- **Output**: A constant reference to an enumeration of type `gguf_type` representing the type of the `gguf_kv` object.
- **See also**: [`gguf_kv`](#gguf_kv)  (Data Structure)


---
#### gguf\_kv::get\_ne<!-- {{#callable:gguf_kv::get_ne}} -->
The `get_ne` function calculates and returns the number of elements in the `gguf_kv` data structure based on its type and data storage.
- **Inputs**: None
- **Control Flow**:
    - Check if the `type` is `GGUF_TYPE_STRING`.
    - If true, calculate `ne` as the size of `data_string` and assert that `is_array` is true or `ne` is 1.
    - Return `ne` if `type` is `GGUF_TYPE_STRING`.
    - If `type` is not `GGUF_TYPE_STRING`, calculate `type_size` using `gguf_type_size(type)`.
    - Assert that the size of `data` is divisible by `type_size`.
    - Calculate `ne` as the size of `data` divided by `type_size`.
    - Assert that `is_array` is true or `ne` is 1.
    - Return `ne`.
- **Output**: The function returns a `size_t` representing the number of elements in the `gguf_kv` data structure.
- **Functions called**:
    - [`gguf_type_size`](#gguf_type_size)
- **See also**: [`gguf_kv`](#gguf_kv)  (Data Structure)


---
#### gguf\_kv::get\_val<!-- {{#callable:gguf_kv::get_val}} -->
The `get_val` function retrieves a value of a specified type from a `gguf_kv` object at a given index, ensuring type and boundary checks.
- **Inputs**:
    - `i`: An optional index of type `size_t` with a default value of 0, indicating the position of the value to retrieve.
- **Control Flow**:
    - Assert that the type of the template parameter `T` matches the `type` of the `gguf_kv` object using `GGML_ASSERT`.
    - If `T` is `std::string`, assert that `data_string` has enough elements and return the string at index `i`.
    - Calculate the size of the type using [`gguf_type_size`](#gguf_type_size) and assert that `data` is properly aligned and has enough elements.
    - Return the value at index `i` from the `data` vector, cast to type `T`.
- **Output**: Returns a constant reference to the value of type `T` at the specified index `i` from the `gguf_kv` object.
- **Functions called**:
    - [`gguf_type_size`](#gguf_type_size)
- **See also**: [`gguf_kv`](#gguf_kv)  (Data Structure)


---
#### gguf\_kv::cast<!-- {{#callable:gguf_kv::cast}} -->
The `cast` function changes the type of the data stored in a `gguf_kv` object to a new specified type, ensuring the data size is compatible with the new type.
- **Inputs**:
    - `new_type`: An enumeration value of type `gguf_type` representing the new type to which the data should be cast.
- **Control Flow**:
    - Calculate the size of the new type using `gguf_type_size(new_type)`.
    - Assert that the current data size is a multiple of the new type size to ensure compatibility.
    - Set the `type` member of the `gguf_kv` object to the new type.
- **Output**: This function does not return a value; it modifies the `type` member of the `gguf_kv` object in place.
- **Functions called**:
    - [`gguf_type_size`](#gguf_type_size)
- **See also**: [`gguf_kv`](#gguf_kv)  (Data Structure)



---
### gguf\_tensor\_info<!-- {{#data_structure:gguf_tensor_info}} -->
- **Type**: `struct`
- **Members**:
    - `t`: Holds the equivalent tensor information using the ggml_tensor structure.
    - `offset`: Specifies the offset from the start of the data, which must be a multiple of ALIGNMENT.
- **Description**: The `gguf_tensor_info` struct is designed to encapsulate information about a tensor, specifically using the `ggml_tensor` structure to hold the tensor's details. It also includes an `offset` field that indicates the position of the tensor data within a larger data block, ensuring that this offset is aligned according to a specified alignment requirement. This struct is likely used in contexts where tensor data needs to be managed or accessed efficiently, particularly in scenarios involving serialized data or file I/O operations.


---
### gguf\_context<!-- {{#data_structure:gguf_context}} -->
- **Type**: `struct`
- **Members**:
    - `version`: Holds the version number of the GGUF format.
    - `kv`: A vector of key-value pairs, each represented by a gguf_kv struct.
    - `info`: A vector of gguf_tensor_info structs, each containing metadata about a tensor.
    - `alignment`: Specifies the alignment requirement for the data section.
    - `offset`: Indicates the offset of the data section from the beginning of the file.
    - `size`: Represents the size of the data section in bytes.
    - `data`: A pointer to the data section, initialized to nullptr.
- **Description**: The gguf_context struct is a central data structure used to manage and store metadata and data for GGUF files. It contains versioning information, a collection of key-value pairs for storing various metadata, and a list of tensor information structures that describe the properties and locations of tensors within the data section. The struct also manages alignment, offset, and size details for the data section, and provides a pointer to the data itself, facilitating efficient data handling and access.


---
### gguf\_reader<!-- {{#data_structure:gguf_reader}} -->
- **Type**: `struct`
- **Members**:
    - `file`: A pointer to a FILE object used for reading data from a file.
- **Description**: The `gguf_reader` struct is designed to facilitate reading various data types from a file stream. It encapsulates a FILE pointer and provides multiple overloaded `read` methods to handle different data types, including primitive types, vectors, strings, and enums. The struct ensures that data is read correctly from the file, handling type conversions and ensuring that the correct number of bytes are read for each data type. This makes it a versatile utility for file I/O operations in applications that require reading structured data from files.
- **Member Functions**:
    - [`gguf_reader::gguf_reader`](#gguf_readergguf_reader)
    - [`gguf_reader::read`](#gguf_readerread)
    - [`gguf_reader::read`](#gguf_readerread)
    - [`gguf_reader::read`](#gguf_readerread)
    - [`gguf_reader::read`](#gguf_readerread)
    - [`gguf_reader::read`](#gguf_readerread)
    - [`gguf_reader::read`](#gguf_readerread)
    - [`gguf_reader::read`](#gguf_readerread)

**Methods**

---
#### gguf\_reader::gguf\_reader<!-- {{#callable:gguf_reader::gguf_reader}} -->
The `gguf_reader` constructor initializes a `gguf_reader` object with a given file pointer.
- **Inputs**:
    - `file`: A pointer to a FILE object that represents the file to be read.
- **Control Flow**:
    - The constructor takes a FILE pointer as an argument.
    - It initializes the `file` member of the `gguf_reader` struct with the provided FILE pointer.
- **Output**: An instance of the `gguf_reader` struct initialized with the provided file pointer.
- **See also**: [`gguf_reader`](#gguf_reader)  (Data Structure)


---
#### gguf\_reader::read<!-- {{#callable:gguf_reader::read}} -->
The `read` function template reads a specified number of bytes from a file into a given destination variable and returns whether the read operation was successful.
- **Inputs**:
    - `T & dst`: A reference to a variable of type T where the data read from the file will be stored.
- **Control Flow**:
    - The function uses `fread` to read data from the file associated with the `gguf_reader` object into the `dst` variable.
    - It reads `sizeof(dst)` bytes from the file.
    - The function checks if the number of bytes read is equal to `sizeof(dst)`.
    - If the number of bytes read matches `sizeof(dst)`, the function returns `true`, indicating a successful read; otherwise, it returns `false`.
- **Output**: A boolean value indicating whether the read operation was successful (true if successful, false otherwise).
- **See also**: [`gguf_reader`](#gguf_reader)  (Data Structure)


---
#### gguf\_reader::read<!-- {{#callable:gguf_reader::read}} -->
The [`read`](#gguf_readerread) function reads `n` elements of type `T` from a file into a vector, handling boolean types separately.
- **Inputs**:
    - `dst`: A reference to a vector of type `T` where the read elements will be stored.
    - `n`: The number of elements to read from the file into the vector.
- **Control Flow**:
    - Resize the vector `dst` to hold `n` elements.
    - Iterate over each element in the vector `dst`.
    - For each element, check if the type `T` is `bool`.
    - If `T` is `bool`, read a temporary boolean value and assign it to the current position in `dst`.
    - If `T` is not `bool`, read directly into the current position in `dst`.
    - If any read operation fails, return `false`.
    - If all read operations succeed, return `true`.
- **Output**: Returns `true` if all elements are successfully read, otherwise returns `false`.
- **Functions called**:
    - [`gguf_reader::read`](#gguf_readerread)
- **See also**: [`gguf_reader`](#gguf_reader)  (Data Structure)


---
#### gguf\_reader::read<!-- {{#callable:gguf_reader::read}} -->
The [`read`](#gguf_readerread) function reads a boolean value from a file and assigns it to the provided reference, returning true if successful.
- **Inputs**:
    - `dst`: A reference to a boolean variable where the read value will be stored.
- **Control Flow**:
    - Declare an int8_t variable `tmp` and initialize it to -1.
    - Call the [`read`](#gguf_readerread) function with `tmp` as the argument to read a byte from the file.
    - If the read operation fails, return false.
    - Assign the result of `tmp != 0` to `dst`, converting the byte to a boolean value.
    - Return true to indicate the read operation was successful.
- **Output**: A boolean value indicating whether the read operation was successful.
- **Functions called**:
    - [`gguf_reader::read`](#gguf_readerread)
- **See also**: [`gguf_reader`](#gguf_reader)  (Data Structure)


---
#### gguf\_reader::read<!-- {{#callable:gguf_reader::read}} -->
The [`read`](#gguf_readerread) function reads an integer from a file and assigns it to a [`ggml_type`](../include/ggml.h.driver.md#ggml_type) enum variable.
- **Inputs**:
    - `dst`: A reference to a [`ggml_type`](../include/ggml.h.driver.md#ggml_type) enum variable where the read value will be stored.
- **Control Flow**:
    - Initialize a temporary integer variable `tmp` to -1.
    - Call the [`read`](#gguf_readerread) function to read an integer value into `tmp`.
    - If the read operation fails, return `false`.
    - Convert the integer `tmp` to a [`ggml_type`](../include/ggml.h.driver.md#ggml_type) and assign it to `dst`.
    - Return `true` to indicate successful reading and conversion.
- **Output**: A boolean value indicating whether the read and conversion operation was successful.
- **Functions called**:
    - [`gguf_reader::read`](#gguf_readerread)
    - [`ggml_type`](../include/ggml.h.driver.md#ggml_type)
- **See also**: [`gguf_reader`](#gguf_reader)  (Data Structure)


---
#### gguf\_reader::read<!-- {{#callable:gguf_reader::read}} -->
The [`read`](#gguf_readerread) function reads an integer from a file and assigns it to a [`gguf_type`](../include/gguf.h.driver.md#gguf_type) enum variable.
- **Inputs**:
    - `dst`: A reference to a [`gguf_type`](../include/gguf.h.driver.md#gguf_type) enum variable where the read value will be stored.
- **Control Flow**:
    - Initialize a temporary `int32_t` variable `tmp` to -1.
    - Call the [`read`](#gguf_readerread) function to read an integer from the file into `tmp`.
    - If the read operation fails, return `false`.
    - Convert `tmp` to [`gguf_type`](../include/gguf.h.driver.md#gguf_type) and assign it to `dst`.
    - Return `true` indicating the read operation was successful.
- **Output**: A boolean value indicating whether the read operation was successful (`true`) or not (`false`).
- **Functions called**:
    - [`gguf_reader::read`](#gguf_readerread)
    - [`gguf_type`](../include/gguf.h.driver.md#gguf_type)
- **See also**: [`gguf_reader`](#gguf_reader)  (Data Structure)


---
#### gguf\_reader::read<!-- {{#callable:gguf_reader::read}} -->
The [`read`](#gguf_readerread) function reads a string from a file, resizing the string to match the size read from the file, and returns whether the read operation was successful.
- **Inputs**:
    - `dst`: A reference to a std::string that will be populated with data read from the file.
- **Control Flow**:
    - Initialize a uint64_t variable `size` to -1.
    - Attempt to read the size of the string from the file using another [`read`](#gguf_readerread) function; if unsuccessful, return false.
    - Resize the `dst` string to the size read from the file.
    - Read the data from the file into the `dst` string and return whether the number of bytes read matches the expected length.
- **Output**: A boolean value indicating whether the string was successfully read from the file.
- **Functions called**:
    - [`gguf_reader::read`](#gguf_readerread)
- **See also**: [`gguf_reader`](#gguf_reader)  (Data Structure)


---
#### gguf\_reader::read<!-- {{#callable:gguf_reader::read}} -->
The `read` function reads a specified number of bytes from a file into a destination buffer and checks if the read operation was successful.
- **Inputs**:
    - `dst`: A pointer to the destination buffer where the data will be read into.
    - `size`: The number of bytes to read from the file.
- **Control Flow**:
    - The function calls `fread` to read `size` bytes from the file into the buffer pointed to by `dst`.
    - It compares the number of bytes read by `fread` to `size` to determine if the read operation was successful.
- **Output**: Returns `true` if the number of bytes read equals `size`, indicating a successful read operation; otherwise, returns `false`.
- **See also**: [`gguf_reader`](#gguf_reader)  (Data Structure)



---
### gguf\_writer<!-- {{#data_structure:gguf_writer}} -->
- **Type**: `struct`
- **Members**:
    - `buf`: A reference to a vector of int8_t that serves as the buffer for writing data.
- **Description**: The `gguf_writer` struct is designed to facilitate writing various types of data into a buffer, specifically a vector of int8_t. It provides multiple overloaded `write` methods to handle different data types, including primitive types, strings, and custom types like `gguf_kv` and `gguf_tensor_info`. The struct ensures that data is written in a contiguous manner and supports padding to align data according to specified alignment requirements. This makes it suitable for serializing data structures into a binary format for storage or transmission.
- **Member Functions**:
    - [`gguf_writer::gguf_writer`](#gguf_writergguf_writer)
    - [`gguf_writer::write`](#gguf_writerwrite)
    - [`gguf_writer::write`](#gguf_writerwrite)
    - [`gguf_writer::write`](#gguf_writerwrite)
    - [`gguf_writer::write`](#gguf_writerwrite)
    - [`gguf_writer::write`](#gguf_writerwrite)
    - [`gguf_writer::write`](#gguf_writerwrite)
    - [`gguf_writer::write`](#gguf_writerwrite)
    - [`gguf_writer::write`](#gguf_writerwrite)
    - [`gguf_writer::write_tensor_meta`](#gguf_writerwrite_tensor_meta)
    - [`gguf_writer::pad`](#gguf_writerpad)
    - [`gguf_writer::write_tensor_data`](#gguf_writerwrite_tensor_data)

**Methods**

---
#### gguf\_writer::gguf\_writer<!-- {{#callable:gguf_writer::gguf_writer}} -->
The `gguf_writer` constructor initializes a `gguf_writer` object with a reference to a buffer of type `std::vector<int8_t>`.
- **Inputs**:
    - `buf`: A reference to a `std::vector<int8_t>` that will be used as the buffer for writing data.
- **Control Flow**:
    - The constructor takes a reference to a `std::vector<int8_t>` as an argument.
    - It initializes the member variable `buf` with the provided reference.
- **Output**: An instance of the `gguf_writer` class with its buffer initialized.
- **See also**: [`gguf_writer`](#gguf_writer)  (Data Structure)


---
#### gguf\_writer::write<!-- {{#callable:gguf_writer::write}} -->
The `write` function template serializes a given value of any type into a byte buffer by iterating over its bytes and appending them to the buffer.
- **Inputs**:
    - `val`: A constant reference to a value of any type `T` that is to be serialized into the buffer.
- **Control Flow**:
    - The function iterates over each byte of the input value `val` using a loop that runs from 0 to the size of `val` in bytes.
    - Within the loop, each byte of `val` is accessed by casting the address of `val` to a pointer to `int8_t` and indexing into it.
    - Each byte is then appended to the `buf` vector, which is a member of the `gguf_writer` struct.
- **Output**: The function does not return a value; it modifies the `buf` vector by appending the bytes of the input value.
- **See also**: [`gguf_writer`](#gguf_writer)  (Data Structure)


---
#### gguf\_writer::write<!-- {{#callable:gguf_writer::write}} -->
The `write` function appends the contents of a given vector of `int8_t` to the end of an internal buffer.
- **Inputs**:
    - `val`: A constant reference to a vector of `int8_t` values that need to be appended to the buffer.
- **Control Flow**:
    - The function takes a vector of `int8_t` as input.
    - It uses the `insert` method to append the contents of the input vector to the end of the `buf` vector, which is a member of the `gguf_writer` struct.
- **Output**: The function does not return any value; it modifies the `buf` vector in place by appending the input vector's contents.
- **See also**: [`gguf_writer`](#gguf_writer)  (Data Structure)


---
#### gguf\_writer::write<!-- {{#callable:gguf_writer::write}} -->
The [`write`](#gguf_writerwrite) function converts a boolean value to an 8-bit integer and writes it to a buffer.
- **Inputs**:
    - `val`: A boolean reference that is to be written to the buffer.
- **Control Flow**:
    - Convert the boolean `val` to an 8-bit integer `val8`, where `true` becomes 1 and `false` becomes 0.
    - Call the overloaded [`write`](#gguf_writerwrite) function with `val8` to write it to the buffer.
- **Output**: The function does not return a value; it writes the converted boolean value to the buffer.
- **Functions called**:
    - [`gguf_writer::write`](#gguf_writerwrite)
- **See also**: [`gguf_writer`](#gguf_writer)  (Data Structure)


---
#### gguf\_writer::write<!-- {{#callable:gguf_writer::write}} -->
The [`write`](#gguf_writerwrite) method in the `gguf_writer` class serializes a given string into a buffer by first writing its length and then its content as bytes.
- **Inputs**:
    - `val`: A constant reference to a `std::string` that needs to be serialized into the buffer.
- **Control Flow**:
    - Calculate the length of the string `val` and store it in a `uint64_t` variable `n`.
    - Call the [`write`](#gguf_writerwrite) method to serialize the length `n` into the buffer.
    - Iterate over each character in the string `val`, cast it to `int8_t`, and append it to the buffer `buf`.
- **Output**: The function does not return a value; it modifies the `buf` member of the `gguf_writer` instance by appending the serialized string data.
- **Functions called**:
    - [`gguf_writer::write`](#gguf_writerwrite)
- **See also**: [`gguf_writer`](#gguf_writer)  (Data Structure)


---
#### gguf\_writer::write<!-- {{#callable:gguf_writer::write}} -->
The [`write`](#gguf_writerwrite) function converts a C-style string to a `std::string` and writes it to a buffer.
- **Inputs**:
    - `val`: A C-style string (const char*) to be written to the buffer.
- **Control Flow**:
    - The function takes a C-style string as input.
    - It converts the C-style string to a `std::string`.
    - It calls another [`write`](#gguf_writerwrite) function that handles `std::string` inputs to write the string to the buffer.
- **Output**: The function does not return any value; it writes the string data to the buffer.
- **Functions called**:
    - [`gguf_writer::write`](#gguf_writerwrite)
- **See also**: [`gguf_writer`](#gguf_writer)  (Data Structure)


---
#### gguf\_writer::write<!-- {{#callable:gguf_writer::write}} -->
The [`write`](#gguf_writerwrite) function converts a `ggml_type` enum value to an `int32_t` and writes it to a buffer.
- **Inputs**:
    - `val`: An enum value of type `ggml_type` that needs to be written to the buffer.
- **Control Flow**:
    - The function takes a `ggml_type` enum value as input.
    - It casts the enum value to an `int32_t`.
    - It calls another [`write`](#gguf_writerwrite) function with the `int32_t` value to write it to the buffer.
- **Output**: The function does not return any value; it writes the converted integer representation of the enum to a buffer.
- **Functions called**:
    - [`gguf_writer::write`](#gguf_writerwrite)
- **See also**: [`gguf_writer`](#gguf_writer)  (Data Structure)


---
#### gguf\_writer::write<!-- {{#callable:gguf_writer::write}} -->
The [`write`](#gguf_writerwrite) function converts a `gguf_type` enum value to an `int32_t` and writes it to a buffer.
- **Inputs**:
    - `val`: An enum value of type `gguf_type` that needs to be written to the buffer.
- **Control Flow**:
    - The function takes an enum value `val` of type `gguf_type`.
    - It casts the enum value to an `int32_t`.
    - It calls another [`write`](#gguf_writerwrite) function with the `int32_t` value to write it to the buffer.
- **Output**: The function does not return any value; it writes the converted integer representation of the enum to the buffer.
- **Functions called**:
    - [`gguf_writer::write`](#gguf_writerwrite)
- **See also**: [`gguf_writer`](#gguf_writer)  (Data Structure)


---
#### gguf\_writer::write<!-- {{#callable:gguf_writer::write}} -->
The [`write`](#gguf_writerwrite) function serializes a `gguf_kv` structure into a buffer, handling different data types and arrays.
- **Inputs**:
    - `kv`: A `gguf_kv` structure containing a key-value pair, which may be a single value or an array, and includes the key, type, and data.
- **Control Flow**:
    - Retrieve the number of elements `ne` from the `kv` structure using `kv.get_ne()`.
    - Write the key of the `kv` structure using `write(kv.get_key())`.
    - Check if `kv` is an array; if true, write `GGUF_TYPE_ARRAY`, the type, and `ne`; otherwise, write only the type.
    - Use a switch statement to handle different types of data in `kv`:
    - For numeric types (e.g., `GGUF_TYPE_UINT8`, `GGUF_TYPE_INT8`, etc.), write the data directly.
    - For `GGUF_TYPE_BOOL`, iterate over each element and write each boolean value.
    - For `GGUF_TYPE_STRING`, iterate over each element and write each string value.
    - If the type is `GGUF_TYPE_ARRAY` or an invalid type, abort the operation with an error.
- **Output**: The function writes serialized data to a buffer, modifying the buffer in place.
- **Functions called**:
    - [`gguf_writer::write`](#gguf_writerwrite)
- **See also**: [`gguf_writer`](#gguf_writer)  (Data Structure)


---
#### gguf\_writer::write\_tensor\_meta<!-- {{#callable:gguf_writer::write_tensor_meta}} -->
The `write_tensor_meta` function serializes the metadata of a tensor, including its name, dimensions, type, and offset, into a buffer.
- **Inputs**:
    - `info`: A constant reference to a `gguf_tensor_info` structure containing the tensor metadata to be written.
- **Control Flow**:
    - The function begins by writing the tensor's name from `info.t.name` to the buffer.
    - It retrieves the number of dimensions of the tensor using `ggml_n_dims(&info.t)` and writes this value to the buffer.
    - A loop iterates over each dimension, writing the size of each dimension from `info.t.ne[j]` to the buffer.
    - The function writes the tensor's type from `info.t.type` to the buffer.
    - Finally, it writes the tensor's offset from `info.offset` to the buffer.
- **Output**: The function does not return a value; it writes data directly to the buffer associated with the `gguf_writer` instance.
- **Functions called**:
    - [`gguf_writer::write`](#gguf_writerwrite)
    - [`ggml_n_dims`](ggml.c.driver.md#ggml_n_dims)
- **See also**: [`gguf_writer`](#gguf_writer)  (Data Structure)


---
#### gguf\_writer::pad<!-- {{#callable:gguf_writer::pad}} -->
The `pad` function ensures that the buffer size is a multiple of a specified alignment by appending zero bytes as needed.
- **Inputs**:
    - `alignment`: A size_t value representing the alignment requirement, which the buffer size should be a multiple of.
- **Control Flow**:
    - The function enters a while loop that continues as long as the buffer size modulo the alignment is not zero.
    - Inside the loop, a zero byte is defined and written to the buffer using the [`write`](#gguf_writerwrite) method, effectively increasing the buffer size by one byte each iteration.
- **Output**: The function does not return a value; it modifies the buffer in place to meet the alignment requirement.
- **Functions called**:
    - [`gguf_writer::write`](#gguf_writerwrite)
- **See also**: [`gguf_writer`](#gguf_writer)  (Data Structure)


---
#### gguf\_writer::write\_tensor\_data<!-- {{#callable:gguf_writer::write_tensor_data}} -->
The `write_tensor_data` function writes tensor data to a buffer, ensuring proper alignment and handling both contiguous and non-contiguous tensor data.
- **Inputs**:
    - `info`: A reference to a `gguf_tensor_info` structure containing metadata about the tensor, including its offset and data.
    - `offset_data`: A size_t value representing the offset in the buffer where the tensor data should be written.
    - `alignment`: A size_t value specifying the alignment requirement for the buffer.
- **Control Flow**:
    - Assert that the current buffer size minus the offset_data equals the tensor's offset to ensure correct positioning.
    - Assert that the tensor is contiguous using [`ggml_is_contiguous`](ggml.c.driver.md#ggml_is_contiguous).
    - Calculate the current offset in the buffer and the number of bytes required for the tensor data using [`ggml_nbytes`](ggml.c.driver.md#ggml_nbytes).
    - Resize the buffer to accommodate the new tensor data.
    - If the tensor has a buffer, use [`ggml_backend_tensor_get`](ggml-backend.cpp.driver.md#ggml_backend_tensor_get) to retrieve the data and store it in the buffer.
    - If the tensor does not have a buffer, assert that it has data and use `memcpy` to copy the data into the buffer.
    - Call the [`pad`](#gguf_writerpad) function to ensure the buffer size is aligned according to the specified alignment.
- **Output**: The function does not return a value; it modifies the buffer in place to include the tensor data.
- **Functions called**:
    - [`ggml_is_contiguous`](ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_nbytes`](ggml.c.driver.md#ggml_nbytes)
    - [`ggml_backend_tensor_get`](ggml-backend.cpp.driver.md#ggml_backend_tensor_get)
    - [`gguf_writer::pad`](#gguf_writerpad)
- **See also**: [`gguf_writer`](#gguf_writer)  (Data Structure)



# Functions

---
### gguf\_type\_size<!-- {{#callable:gguf_type_size}} -->
Returns the size in bytes of a specified `gguf_type`.
- **Inputs**:
    - `type`: An enumeration value of type `gguf_type` representing the type for which the size is requested.
- **Control Flow**:
    - The function attempts to find the size of the specified `gguf_type` in the `GGUF_TYPE_SIZE` map.
    - If the type is found, the corresponding size is returned; otherwise, 0 is returned.
- **Output**: Returns the size in bytes of the specified `gguf_type`, or 0 if the type is not found.


---
### gguf\_init\_empty<!-- {{#callable:gguf_init_empty}} -->
Initializes and returns a new instance of `gguf_context`.
- **Inputs**: None
- **Control Flow**:
    - The function allocates memory for a new `gguf_context` structure using the `new` operator.
    - It does not perform any checks or initializations beyond memory allocation.
- **Output**: Returns a pointer to the newly created `gguf_context` instance.


---
### gguf\_read\_emplace\_helper<!-- {{#callable:gguf_read_emplace_helper}} -->
The `gguf_read_emplace_helper` function reads data from a `gguf_reader` and emplaces it into a vector of key-value pairs, handling both single values and arrays.
- **Inputs**:
    - `gr`: A constant reference to a `gguf_reader` object used to read data.
    - `kv`: A reference to a vector of `gguf_kv` structures where the read key-value pairs will be stored.
    - `key`: A string representing the key associated with the value being read.
    - `is_array`: A boolean indicating whether the value being read is an array.
    - `n`: A size_t representing the number of elements to read if the value is an array.
- **Control Flow**:
    - The function first checks if the `is_array` flag is true to determine how to read the data.
    - If `is_array` is true, it attempts to read a vector of type `T` from the `gguf_reader`, catching any exceptions related to memory allocation or length errors.
    - If the read operation is successful, it emplaces a new `gguf_kv` object into the `kv` vector with the provided key and the read vector.
    - If `is_array` is false, it reads a single value of type `T` and emplaces it into the `kv` vector.
    - The function returns true if the read operation was successful, otherwise it returns false.
- **Output**: The function returns a boolean indicating the success or failure of the read operation.


---
### gguf\_init\_from\_file\_impl<!-- {{#callable:gguf_init_from_file_impl}} -->
Initializes a `gguf_context` from a GGUF file, reading its metadata and tensor information.
- **Inputs**:
    - `file`: A pointer to a `FILE` object representing the GGUF file to read from.
    - `params`: A `gguf_init_params` structure containing initialization parameters, including a context pointer and a no-alloc flag.
- **Control Flow**:
    - Creates a `gguf_reader` to facilitate reading from the file.
    - Reads the file magic number to verify the file format.
    - Reads the version of the GGUF file and checks for compatibility.
    - Reads the number of tensors and key-value pairs from the file header.
    - Iterates through the key-value pairs, reading each key, type, and associated data, while checking for duplicates and validity.
    - Reads tensor information including names, shapes, types, and offsets, ensuring no duplicates and valid dimensions.
    - Aligns the data section in the file and calculates the total size of the tensor data.
    - If requested, initializes a `ggml_context` and loads the tensor data into it.
- **Output**: Returns a pointer to a newly allocated `gguf_context` containing the initialized data, or nullptr if an error occurs.
- **Functions called**:
    - [`gguf_free`](#gguf_free)
    - [`gguf_type`](../include/gguf.h.driver.md#gguf_type)
    - [`gguf_find_key`](#gguf_find_key)
    - [`gguf_get_val_u32`](#gguf_get_val_u32)
    - [`ggml_set_name`](ggml.c.driver.md#ggml_set_name)
    - [`ggml_type_name`](ggml.c.driver.md#ggml_type_name)
    - [`ggml_type_size`](ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](ggml.c.driver.md#ggml_blck_size)
    - [`ggml_nbytes`](ggml.c.driver.md#ggml_nbytes)
    - [`ggml_tensor_overhead`](ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_init`](ggml.c.driver.md#ggml_init)
    - [`ggml_new_tensor_1d`](ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_free`](ggml.c.driver.md#ggml_free)
    - [`ggml_set_no_alloc`](ggml.c.driver.md#ggml_set_no_alloc)
    - [`ggml_new_tensor`](ggml.c.driver.md#ggml_new_tensor)


---
### gguf\_init\_from\_file<!-- {{#callable:gguf_init_from_file}} -->
Initializes a `gguf_context` from a specified file.
- **Inputs**:
    - `fname`: A pointer to a constant character string representing the filename of the GGUF file to be opened.
    - `params`: A `gguf_init_params` structure containing initialization parameters for the context.
- **Control Flow**:
    - Attempts to open the file specified by `fname` in binary read mode.
    - If the file cannot be opened, logs an error and returns a null pointer.
    - Calls the [`gguf_init_from_file_impl`](#gguf_init_from_file_impl) function, passing the opened file and parameters to initialize the context.
    - Closes the file after the context has been initialized.
    - Returns the initialized `gguf_context` pointer.
- **Output**: Returns a pointer to a `gguf_context` structure initialized with data from the specified file, or null if an error occurred.
- **Functions called**:
    - [`ggml_fopen`](ggml.c.driver.md#ggml_fopen)
    - [`gguf_init_from_file_impl`](#gguf_init_from_file_impl)


---
### gguf\_free<!-- {{#callable:gguf_free}} -->
Frees the memory allocated for a `gguf_context` structure.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that needs to be freed.
- **Control Flow**:
    - Checks if the input pointer `ctx` is null.
    - If `ctx` is not null, it proceeds to delete the allocated memory for `ctx`.
- **Output**: This function does not return a value; it performs a memory deallocation operation.


---
### gguf\_type\_name<!-- {{#callable:gguf_type_name}} -->
Returns the name associated with a given `gguf_type` enumeration.
- **Inputs**:
    - `type`: An enumeration of type `gguf_type` representing the type for which the name is to be retrieved.
- **Control Flow**:
    - The function attempts to find the `type` in the `GGUF_TYPE_NAME` map.
    - If the `type` is found, it returns the corresponding name.
    - If the `type` is not found, it returns a null pointer.
- **Output**: Returns a pointer to a string containing the name of the type, or nullptr if the type is not found.


---
### gguf\_get\_version<!-- {{#callable:gguf_get_version}} -->
The `gguf_get_version` function retrieves the version number from a given `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the version information.
- **Control Flow**:
    - The function directly accesses the `version` member of the `gguf_context` structure pointed to by `ctx`.
    - It returns the value of the `version` member without any additional checks or computations.
- **Output**: Returns a `uint32_t` representing the version number stored in the `gguf_context`.


---
### gguf\_get\_alignment<!-- {{#callable:gguf_get_alignment}} -->
The `gguf_get_alignment` function retrieves the alignment value from a given `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the alignment value.
- **Control Flow**:
    - The function directly accesses the `alignment` member of the `gguf_context` structure pointed to by `ctx`.
    - It returns the value of the `alignment` member without any additional computation or checks.
- **Output**: The function returns the alignment value as a `size_t`, which represents the alignment setting in the context.


---
### gguf\_get\_data\_offset<!-- {{#callable:gguf_get_data_offset}} -->
The `gguf_get_data_offset` function retrieves the data offset from a `gguf_context` structure.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains metadata and state information.
- **Control Flow**:
    - The function directly accesses the `offset` member of the `gguf_context` structure.
    - It returns the value of the `offset` member without any additional computation or checks.
- **Output**: Returns the `offset` value, which indicates the position of the data section in the file associated with the `gguf_context`.


---
### gguf\_get\_n\_kv<!-- {{#callable:gguf_get_n_kv}} -->
The `gguf_get_n_kv` function returns the number of key-value pairs stored in a `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the key-value pairs.
- **Control Flow**:
    - The function accesses the `kv` member of the `ctx` structure, which is a vector of key-value pairs.
    - It calls the `size()` method on the `kv` vector to determine the number of key-value pairs.
- **Output**: The function returns an integer of type `int64_t` representing the count of key-value pairs in the `gguf_context`.


---
### gguf\_find\_key<!-- {{#callable:gguf_find_key}} -->
The `gguf_find_key` function searches for a specified key in a `gguf_context` and returns its index or -1 if not found.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains key-value pairs.
    - `key`: A string representing the key to search for in the context.
- **Control Flow**:
    - Initialize `keyfound` to -1 to indicate that the key has not been found.
    - Retrieve the number of key-value pairs in the context using [`gguf_get_n_kv`](#gguf_get_n_kv).
    - Iterate over each key-value pair in the context.
    - For each key, compare it with the provided `key` using `strcmp`.
    - If a match is found, set `keyfound` to the current index and break the loop.
- **Output**: Returns the index of the found key in the key-value pairs, or -1 if the key is not found.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)
    - [`gguf_get_key`](#gguf_get_key)


---
### gguf\_get\_key<!-- {{#callable:gguf_get_key}} -->
Retrieves the key associated with a specified key ID from a `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains key-value pairs.
    - `key_id`: An integer representing the index of the key to retrieve, which must be within the valid range.
- **Control Flow**:
    - The function first asserts that the `key_id` is within the valid range (0 to the number of key-value pairs in the context).
    - If the assertion passes, it retrieves the key from the `kv` vector in the `gguf_context` using the provided `key_id`.
    - Finally, it returns the key as a C-style string using `c_str()`.
- **Output**: Returns a pointer to a constant character string representing the key associated with the specified `key_id`.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_kv\_type<!-- {{#callable:gguf_get_kv_type}} -->
Retrieves the type of a key-value pair from a `gguf_context` based on its key ID.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains key-value pairs.
    - `key_id`: An integer representing the index of the key-value pair whose type is to be retrieved.
- **Control Flow**:
    - The function first asserts that the `key_id` is within valid bounds (non-negative and less than the number of key-value pairs).
    - It then checks if the key-value pair at the specified `key_id` is an array.
    - If it is an array, the function returns `GGUF_TYPE_ARRAY`; otherwise, it retrieves and returns the type of the key-value pair using the `get_type()` method.
- **Output**: Returns an enumeration value of type `gguf_type` indicating the type of the key-value pair.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_arr\_type<!-- {{#callable:gguf_get_arr_type}} -->
Retrieves the type of an array stored in a key-value pair from a `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains key-value pairs and their associated metadata.
    - `key_id`: An integer representing the index of the key-value pair in the context, which must correspond to an array.
- **Control Flow**:
    - The function first asserts that the `key_id` is within valid bounds (non-negative and less than the number of key-value pairs).
    - It then asserts that the key-value pair at the specified `key_id` is indeed an array.
    - Finally, it retrieves and returns the type of the array from the key-value pair.
- **Output**: Returns an enumeration value of type `gguf_type` that indicates the data type of the array associated with the specified key.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_arr\_data<!-- {{#callable:gguf_get_arr_data}} -->
Retrieves a pointer to the data of an array stored in the `gguf_context` structure.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains key-value pairs and other context information.
    - `key_id`: An integer representing the index of the key-value pair from which to retrieve the array data.
- **Control Flow**:
    - The function first asserts that the `key_id` is within valid bounds (non-negative and less than the number of key-value pairs).
    - It then asserts that the type of the key-value pair at `key_id` is not a string, as strings are handled differently.
    - Finally, it returns a pointer to the data of the array associated with the specified key.
- **Output**: A pointer to the data of the array associated with the specified `key_id` in the `gguf_context`, or nullptr if the type is a string.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_arr\_str<!-- {{#callable:gguf_get_arr_str}} -->
Retrieves a specific string from an array of strings stored in a `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains key-value pairs.
    - `key_id`: An integer identifier for the key in the context, which must be valid.
    - `i`: An index specifying which string to retrieve from the array of strings.
- **Control Flow**:
    - Asserts that the `key_id` is within the valid range of key-value pairs in the context.
    - Asserts that the type of the key-value pair at `key_id` is a string type.
    - Returns the C-style string at index `i` from the array of strings associated with the specified key.
- **Output**: Returns a pointer to the C-style string located at the specified index in the array of strings.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_arr\_n<!-- {{#callable:gguf_get_arr_n}} -->
The `gguf_get_arr_n` function retrieves the number of elements in an array stored in a `gguf_context` based on a specified key.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains key-value pairs and other metadata.
    - `key_id`: An integer representing the index of the key in the key-value pairs of the `gguf_context`.
- **Control Flow**:
    - The function first asserts that the `key_id` is within valid bounds (non-negative and less than the number of key-value pairs).
    - It checks if the type of the data associated with the given `key_id` is a string; if so, it returns the size of the string data.
    - If the type is not a string, it calculates the size of the data type using [`gguf_type_size`](#gguf_type_size) and asserts that the data size is a multiple of this type size.
    - Finally, it returns the number of elements by dividing the total data size by the size of the data type.
- **Output**: The function returns the number of elements in the array associated with the specified key, or the size of the string if the type is a string.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)
    - [`gguf_type_size`](#gguf_type_size)


---
### gguf\_get\_val\_u8<!-- {{#callable:gguf_get_val_u8}} -->
Retrieves a single `uint8_t` value from the key-value store in the `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the key-value pairs.
    - `key_id`: An integer representing the index of the key-value pair to retrieve.
- **Control Flow**:
    - The function first asserts that the `key_id` is within the valid range of key-value pairs in the context.
    - It then asserts that the number of elements for the specified key is exactly one.
    - Finally, it retrieves and returns the value associated with the specified key as a `uint8_t`.
- **Output**: Returns the `uint8_t` value associated with the specified `key_id` from the `gguf_context`.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_val\_i8<!-- {{#callable:gguf_get_val_i8}} -->
Retrieves the value associated with a specified key as an `int8_t` from a `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains key-value pairs and other context information.
    - `key_id`: An integer representing the index of the key in the key-value pairs, which must be non-negative and less than the total number of key-value pairs.
- **Control Flow**:
    - The function first asserts that the `key_id` is within valid bounds (non-negative and less than the number of key-value pairs).
    - It then asserts that the number of elements associated with the specified key is exactly one.
    - Finally, it retrieves and returns the value associated with the key as an `int8_t`.
- **Output**: Returns the value of type `int8_t` associated with the specified key in the `gguf_context`.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_val\_u16<!-- {{#callable:gguf_get_val_u16}} -->
Retrieves a 16-bit unsigned integer value from the key-value store in the `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the key-value pairs from which the value is to be retrieved.
    - `key_id`: An integer representing the index of the key-value pair in the context's key-value store.
- **Control Flow**:
    - The function first asserts that the `key_id` is within the valid range of indices for the key-value pairs in the context.
    - It then asserts that the number of elements associated with the specified key is exactly one.
    - Finally, it retrieves and returns the value associated with the specified key as a `uint16_t`.
- **Output**: Returns the 16-bit unsigned integer value associated with the specified key in the `gguf_context`.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_val\_i16<!-- {{#callable:gguf_get_val_i16}} -->
Retrieves a 16-bit signed integer value from the key-value store in the given context.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the key-value pairs.
    - `key_id`: An integer representing the index of the key-value pair to retrieve.
- **Control Flow**:
    - The function first asserts that the `key_id` is within the valid range of key-value pairs in the context.
    - It then asserts that the number of elements for the specified key is exactly one.
    - Finally, it retrieves and returns the value associated with the specified key as a 16-bit signed integer.
- **Output**: Returns the 16-bit signed integer value associated with the specified key in the context.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_val\_u32<!-- {{#callable:gguf_get_val_u32}} -->
Retrieves a 32-bit unsigned integer value from the key-value store in the `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the key-value pairs.
    - `key_id`: An integer representing the index of the key-value pair from which to retrieve the value.
- **Control Flow**:
    - The function first asserts that `key_id` is within the valid range of key-value pairs in the context.
    - It then asserts that the number of elements for the specified key is exactly one.
    - Finally, it retrieves and returns the value as a `uint32_t` from the key-value pair.
- **Output**: Returns the 32-bit unsigned integer value associated with the specified `key_id`.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_val\_i32<!-- {{#callable:gguf_get_val_i32}} -->
Retrieves a 32-bit integer value from the key-value store in the given context.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the key-value pairs.
    - `key_id`: An integer representing the index of the key-value pair to retrieve.
- **Control Flow**:
    - Asserts that `key_id` is within the valid range of key-value pairs in the context.
    - Asserts that the number of elements for the specified key is exactly one.
    - Returns the value associated with the specified key as a 32-bit integer.
- **Output**: Returns the 32-bit integer value associated with the specified key in the context.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_val\_f32<!-- {{#callable:gguf_get_val_f32}} -->
Retrieves a 32-bit floating-point value from the key-value store in the given context.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the key-value store from which the value will be retrieved.
    - `key_id`: An integer representing the index of the key-value pair in the context's key-value store.
- **Control Flow**:
    - The function first asserts that the `key_id` is within the valid range of indices for the key-value pairs in the context.
    - It then asserts that the number of elements (`ne`) for the specified key is exactly 1, ensuring that a single value is expected.
    - Finally, it retrieves and returns the value associated with the specified key as a float.
- **Output**: Returns the 32-bit floating-point value associated with the specified `key_id` from the key-value store.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_val\_u64<!-- {{#callable:gguf_get_val_u64}} -->
Retrieves a 64-bit unsigned integer value from the key-value store in the given context.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the key-value pairs.
    - `key_id`: An integer representing the index of the key-value pair from which to retrieve the value.
- **Control Flow**:
    - The function first asserts that the `key_id` is within the valid range of key-value pairs in the context.
    - It then asserts that the number of elements associated with the specified key is exactly one.
    - Finally, it retrieves and returns the value as a `uint64_t` from the key-value store.
- **Output**: Returns the 64-bit unsigned integer value associated with the specified key in the context.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_val\_i64<!-- {{#callable:gguf_get_val_i64}} -->
Retrieves a 64-bit integer value associated with a specified key from a `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains key-value pairs and other context information.
    - `key_id`: An integer identifier for the key whose associated value is to be retrieved, which must be non-negative and less than the number of key-value pairs in the context.
- **Control Flow**:
    - The function first asserts that the `key_id` is within valid bounds (non-negative and less than the total number of key-value pairs).
    - It then asserts that the number of elements associated with the specified key is exactly one.
    - Finally, it retrieves and returns the value associated with the key as a 64-bit integer.
- **Output**: Returns the 64-bit integer value associated with the specified key in the `gguf_context`.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_val\_f64<!-- {{#callable:gguf_get_val_f64}} -->
Retrieves a double value from the key-value store in the `gguf_context` structure.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the key-value pairs.
    - `key_id`: An integer representing the index of the key-value pair to retrieve.
- **Control Flow**:
    - Asserts that `key_id` is within the valid range of key-value pairs in the context.
    - Asserts that the number of elements for the specified key is exactly one.
    - Returns the double value associated with the specified key from the key-value store.
- **Output**: Returns the double value stored at the specified key in the `gguf_context`.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_val\_bool<!-- {{#callable:gguf_get_val_bool}} -->
Retrieves a boolean value from the key-value store in the `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the key-value pairs.
    - `key_id`: An integer representing the index of the key-value pair to retrieve, which must be within valid bounds.
- **Control Flow**:
    - The function first asserts that `key_id` is within the valid range of key-value pairs in the context.
    - It then asserts that the number of elements (ne) for the specified key is exactly 1, indicating that it is a single boolean value.
    - Finally, it retrieves and returns the boolean value associated with the specified key.
- **Output**: Returns the boolean value stored at the specified key in the `gguf_context`.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_val\_str<!-- {{#callable:gguf_get_val_str}} -->
Retrieves the string value associated with a specified key ID from a `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains key-value pairs.
    - `key_id`: An integer representing the index of the key-value pair to retrieve.
- **Control Flow**:
    - Asserts that `key_id` is within the valid range of key-value pairs in the context.
    - Asserts that the number of elements for the specified key is exactly one.
    - Returns the C-style string representation of the value associated with the specified key.
- **Output**: Returns a pointer to a constant character string representing the value associated with the specified key ID.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_get\_val\_data<!-- {{#callable:gguf_get_val_data}} -->
Retrieves the raw data associated with a specified key from the `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains key-value pairs and associated data.
    - `key_id`: An integer identifier for the key whose associated data is to be retrieved.
- **Control Flow**:
    - The function first asserts that the `key_id` is within valid bounds (non-negative and less than the number of key-value pairs).
    - It then asserts that the number of elements associated with the specified key is exactly one.
    - Next, it checks that the type of the data associated with the key is not a string.
    - Finally, it returns a pointer to the raw data associated with the specified key.
- **Output**: Returns a pointer to the raw data associated with the specified key in the `gguf_context`, or nullptr if the key is invalid.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)


---
### gguf\_find\_tensor<!-- {{#callable:gguf_find_tensor}} -->
The `gguf_find_tensor` function searches for a tensor by its name within a given `gguf_context` and returns its index or -1 if not found.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains information about tensors.
    - `name`: A string representing the name of the tensor to search for.
- **Control Flow**:
    - Initialize `tensor_id` to -1 to indicate that the tensor has not been found.
    - Retrieve the total number of tensors using `gguf_get_n_tensors(ctx)`.
    - Iterate over each tensor index from 0 to `n_tensors - 1`.
    - For each index, compare the provided `name` with the name of the tensor obtained from `gguf_get_tensor_name(ctx, i)`.
    - If a match is found, set `tensor_id` to the current index and break the loop.
- **Output**: Returns the index of the tensor if found; otherwise, returns -1.
- **Functions called**:
    - [`gguf_get_n_tensors`](#gguf_get_n_tensors)
    - [`gguf_get_tensor_name`](#gguf_get_tensor_name)


---
### gguf\_get\_tensor\_offset<!-- {{#callable:gguf_get_tensor_offset}} -->
Retrieves the byte offset of a tensor identified by its ID from the context.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains information about tensors.
    - `tensor_id`: An integer representing the ID of the tensor whose offset is to be retrieved.
- **Control Flow**:
    - The function first asserts that the `tensor_id` is within valid bounds (non-negative and less than the total number of tensors).
    - If the assertion passes, it retrieves the offset of the specified tensor from the `info` array in the `gguf_context` structure.
- **Output**: Returns the byte offset of the specified tensor within the data section of the context.
- **Functions called**:
    - [`gguf_get_n_tensors`](#gguf_get_n_tensors)


---
### gguf\_get\_tensor\_name<!-- {{#callable:gguf_get_tensor_name}} -->
Retrieves the name of a tensor given its ID from the `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains information about tensors.
    - `tensor_id`: An integer representing the ID of the tensor whose name is to be retrieved.
- **Control Flow**:
    - The function asserts that the `tensor_id` is within valid bounds (greater than or equal to 0 and less than the total number of tensors).
    - If the assertion passes, it accesses the `info` array in the `gguf_context` structure to retrieve the name of the tensor at the specified `tensor_id`.
- **Output**: Returns a pointer to a constant character string representing the name of the tensor.
- **Functions called**:
    - [`gguf_get_n_tensors`](#gguf_get_n_tensors)


---
### gguf\_get\_tensor\_type<!-- {{#callable:gguf_get_tensor_type}} -->
Retrieves the type of a tensor identified by its ID from a given context.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains information about tensors.
    - `tensor_id`: An integer representing the ID of the tensor whose type is to be retrieved.
- **Control Flow**:
    - The function asserts that the `tensor_id` is within valid bounds (greater than or equal to 0 and less than the total number of tensors in the context).
    - If the assertion passes, it accesses the `info` array of the `gguf_context` structure to retrieve the type of the specified tensor.
- **Output**: Returns the `ggml_type` of the tensor identified by `tensor_id`.
- **Functions called**:
    - [`gguf_get_n_tensors`](#gguf_get_n_tensors)


---
### gguf\_get\_tensor\_size<!-- {{#callable:gguf_get_tensor_size}} -->
Retrieves the size in bytes of a tensor identified by its ID from a given context.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains information about tensors.
    - `tensor_id`: An integer representing the ID of the tensor whose size is to be retrieved.
- **Control Flow**:
    - The function first asserts that the `tensor_id` is within valid bounds (greater than or equal to 0 and less than the total number of tensors).
    - It then retrieves the size of the tensor by calling the [`ggml_nbytes`](ggml.c.driver.md#ggml_nbytes) function with the tensor information from the context.
- **Output**: Returns the size of the specified tensor in bytes.
- **Functions called**:
    - [`gguf_get_n_tensors`](#gguf_get_n_tensors)
    - [`ggml_nbytes`](ggml.c.driver.md#ggml_nbytes)


---
### gguf\_remove\_key<!-- {{#callable:gguf_remove_key}} -->
Removes a key-value pair from the `gguf_context` if the specified key exists.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds key-value pairs.
    - `key`: A string representing the key to be removed from the context.
- **Control Flow**:
    - Calls [`gguf_find_key`](#gguf_find_key) to locate the index of the specified key in the context.
    - If the key is found (i.e., `key_id` is non-negative), it erases the key-value pair from the `kv` vector in the context.
    - Returns the index of the removed key, or -1 if the key was not found.
- **Output**: Returns the index of the removed key if it existed, or -1 if the key was not found.
- **Functions called**:
    - [`gguf_find_key`](#gguf_find_key)


---
### gguf\_check\_reserved\_keys<!-- {{#callable:gguf_check_reserved_keys}} -->
Checks if the provided key is a reserved key and validates its associated value.
- **Inputs**:
    - `key`: A string representing the key to check against reserved keys.
    - `val`: A value of type T that is associated with the key.
- **Control Flow**:
    - The function first checks if the provided `key` matches the reserved key `GGUF_KEY_GENERAL_ALIGNMENT`.
    - If the key matches, it checks if the type of `val` is `uint32_t` using `std::is_same`.
    - If `val` is of type `uint32_t`, it asserts that `val` is greater than 0 and is a power of 2.
    - If `val` is not of type `uint32_t`, it marks `val` as unused and aborts the operation with an error message.
- **Output**: The function does not return a value; it asserts conditions or aborts the program based on the validity of the inputs.


---
### gguf\_set\_val\_u8<!-- {{#callable:gguf_set_val_u8}} -->
Sets a key-value pair in the `gguf_context` for an 8-bit unsigned integer.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure where the key-value pair will be stored.
    - `key`: A string representing the key under which the value will be stored.
    - `val`: An 8-bit unsigned integer value to be associated with the specified key.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and value against reserved keys.
    - Calls [`gguf_remove_key`](#gguf_remove_key) to remove any existing entry for the specified key in the context.
    - Adds the new key-value pair to the `kv` vector of the `gguf_context`.
- **Output**: This function does not return a value; it modifies the `gguf_context` by adding or updating the specified key-value pair.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_val\_i8<!-- {{#callable:gguf_set_val_i8}} -->
Sets a key-value pair in the `gguf_context` for an 8-bit integer value.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds key-value pairs and other context information.
    - `key`: A string representing the key under which the value will be stored.
    - `val`: An 8-bit integer value to be associated with the specified key.
- **Control Flow**:
    - The function first calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and value.
    - It then calls [`gguf_remove_key`](#gguf_remove_key) to remove any existing entry for the specified key in the context.
    - Finally, it adds the new key-value pair to the context's key-value vector using `emplace_back`.
- **Output**: The function does not return a value; it modifies the `gguf_context` in place by adding or updating the key-value pair.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_val\_u16<!-- {{#callable:gguf_set_val_u16}} -->
Sets a 16-bit unsigned integer value in the `gguf_context` associated with a specified key.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds key-value pairs.
    - `key`: A string representing the key under which the value will be stored.
    - `val`: A 16-bit unsigned integer value to be associated with the specified key.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and value.
    - Calls [`gguf_remove_key`](#gguf_remove_key) to remove any existing entry associated with the key.
    - Adds a new key-value pair to the context's key-value vector using `emplace_back`.
- **Output**: This function does not return a value; it modifies the `gguf_context` by adding or updating the key-value pair.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_val\_i16<!-- {{#callable:gguf_set_val_i16}} -->
Sets a 16-bit integer value in the `gguf_context` associated with a specified key.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure where the key-value pair will be stored.
    - `key`: A string representing the key under which the value will be stored.
    - `val`: An `int16_t` value that will be associated with the specified key.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and value.
    - Invokes [`gguf_remove_key`](#gguf_remove_key) to remove any existing entry for the specified key in the context.
    - Adds the new key-value pair to the `kv` vector of the context using `emplace_back`.
- **Output**: This function does not return a value; it modifies the `gguf_context` by adding or updating the key-value pair.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_val\_u32<!-- {{#callable:gguf_set_val_u32}} -->
Sets a 32-bit unsigned integer value in the `gguf_context` associated with a specified key.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds key-value pairs.
    - `key`: A string representing the key associated with the value to be set.
    - `val`: A 32-bit unsigned integer value to be stored in the context.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and value.
    - Invokes [`gguf_remove_key`](#gguf_remove_key) to remove any existing entry associated with the provided key.
    - Adds a new key-value pair to the context's key-value vector using `emplace_back`.
- **Output**: This function does not return a value; it modifies the `gguf_context` by adding or updating the specified key with the provided value.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_val\_i32<!-- {{#callable:gguf_set_val_i32}} -->
Sets a key-value pair in the `gguf_context` for a 32-bit integer.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure where the key-value pair will be stored.
    - `key`: A string representing the key under which the integer value will be stored.
    - `val`: An `int32_t` value that will be associated with the specified key.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and value against reserved keys.
    - Calls [`gguf_remove_key`](#gguf_remove_key) to remove any existing entry for the specified key in the context.
    - Adds the new key-value pair to the `kv` vector in the `gguf_context`.
- **Output**: This function does not return a value; it modifies the `gguf_context` by adding or updating the specified key with the provided integer value.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_val\_f32<!-- {{#callable:gguf_set_val_f32}} -->
Sets a float value associated with a key in the `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds key-value pairs.
    - `key`: A string representing the key to associate with the float value.
    - `val`: A float value to be set in the context.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and value.
    - Calls [`gguf_remove_key`](#gguf_remove_key) to remove any existing entry for the specified key.
    - Adds the new key-value pair to the `kv` vector in the `gguf_context`.
- **Output**: This function does not return a value; it modifies the `gguf_context` by adding or updating the key-value pair.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_val\_u64<!-- {{#callable:gguf_set_val_u64}} -->
Sets a 64-bit unsigned integer value in the `gguf_context` associated with a specified key.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure where the key-value pair will be stored.
    - `key`: A string representing the key under which the value will be stored.
    - `val`: A 64-bit unsigned integer value to be associated with the specified key.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and value against reserved keys.
    - Calls [`gguf_remove_key`](#gguf_remove_key) to remove any existing entry associated with the specified key from the context.
    - Adds a new key-value pair to the context's key-value vector using `emplace_back`.
- **Output**: This function does not return a value; it modifies the `gguf_context` by adding or updating the key-value pair.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_val\_i64<!-- {{#callable:gguf_set_val_i64}} -->
Sets a key-value pair in the `gguf_context` for a 64-bit integer.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure where the key-value pair will be stored.
    - `key`: A string representing the key under which the value will be stored.
    - `val`: An `int64_t` value that will be associated with the specified key.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and value against reserved keys.
    - Calls [`gguf_remove_key`](#gguf_remove_key) to remove any existing entry for the specified key in the context.
    - Adds the new key-value pair to the `kv` vector of the context using `emplace_back`.
- **Output**: This function does not return a value; it modifies the `gguf_context` by adding or updating the specified key-value pair.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_val\_f64<!-- {{#callable:gguf_set_val_f64}} -->
Sets a key-value pair in the `gguf_context` for a double value.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure where the key-value pair will be stored.
    - `key`: A string representing the key under which the double value will be stored.
    - `val`: The double value to be associated with the specified key.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and value against reserved keys.
    - Calls [`gguf_remove_key`](#gguf_remove_key) to remove any existing entry for the specified key in the context.
    - Adds the new key-value pair (key and double value) to the `kv` vector of the context.
- **Output**: This function does not return a value; it modifies the `gguf_context` by adding or updating the specified key with the double value.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_val\_bool<!-- {{#callable:gguf_set_val_bool}} -->
Sets a boolean value in the `gguf_context` associated with a specified key.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure where the key-value pair will be stored.
    - `key`: A string representing the key under which the boolean value will be stored.
    - `val`: A boolean value to be associated with the specified key.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and value.
    - Invokes [`gguf_remove_key`](#gguf_remove_key) to remove any existing entry for the specified key in the context.
    - Adds the new key-value pair (key and boolean value) to the `kv` vector of the `gguf_context`.
- **Output**: The function does not return a value; it modifies the `gguf_context` by adding or updating the key-value pair.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_val\_str<!-- {{#callable:gguf_set_val_str}} -->
Sets a string value in the `gguf_context` associated with a specified key.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure where the key-value pair will be stored.
    - `key`: A constant character pointer representing the key under which the value will be stored.
    - `val`: A constant character pointer representing the string value to be associated with the key.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and value against reserved keys.
    - Invokes [`gguf_remove_key`](#gguf_remove_key) to remove any existing entry for the specified key in the context.
    - Adds a new key-value pair to the context's key-value vector, converting the string value to a `std::string`.
- **Output**: The function does not return a value; it modifies the `gguf_context` by adding or updating the key-value pair.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_arr\_data<!-- {{#callable:gguf_set_arr_data}} -->
Sets an array of data in the `gguf_context` associated with a specified key.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure where the data will be stored.
    - `key`: A string representing the key under which the data will be stored.
    - `type`: An enumeration value of type `gguf_type` indicating the data type of the array.
    - `data`: A pointer to the data array that will be copied into the context.
    - `n`: The number of elements in the data array.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to ensure the key is not a reserved key.
    - Removes any existing entry for the specified key in the context using [`gguf_remove_key`](#gguf_remove_key).
    - Calculates the total number of bytes required for the data based on the number of elements and their type size.
    - Creates a temporary vector to hold the data and copies the input data into this vector.
    - Stores the key and the copied data in the context's key-value store.
    - Casts the stored data to the specified type using the `cast` method.
- **Output**: This function does not return a value; it modifies the `gguf_context` by adding or updating the key-value pair with the provided data.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)
    - [`gguf_type_size`](#gguf_type_size)


---
### gguf\_set\_arr\_str<!-- {{#callable:gguf_set_arr_str}} -->
Sets an array of strings in the `gguf_context` associated with a specified key.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure where the key-value pair will be stored.
    - `key`: A string representing the key under which the array of strings will be stored.
    - `data`: A pointer to an array of C-style strings (const char**) that will be converted to std::string.
    - `n`: The number of strings in the array pointed to by `data`.
- **Control Flow**:
    - Calls [`gguf_check_reserved_keys`](#gguf_check_reserved_keys) to validate the key and data.
    - Removes any existing key-value pair associated with `key` from the context using [`gguf_remove_key`](#gguf_remove_key).
    - Creates a temporary vector of strings (`tmp`) of size `n`.
    - Iterates over the input array `data`, converting each C-style string to a `std::string` and storing it in `tmp`.
    - Adds the new key-value pair (key and the vector of strings) to the context's key-value store.
- **Output**: This function does not return a value; it modifies the `gguf_context` by adding a new key-value pair.
- **Functions called**:
    - [`gguf_check_reserved_keys`](#gguf_check_reserved_keys)
    - [`gguf_remove_key`](#gguf_remove_key)


---
### gguf\_set\_kv<!-- {{#callable:gguf_set_kv}} -->
The `gguf_set_kv` function sets or adds key-value pairs from one `gguf_context` to another.
- **Inputs**:
    - `ctx`: A pointer to the destination `gguf_context` where key-value pairs will be set.
    - `src`: A pointer to the source `gguf_context` from which key-value pairs will be copied.
- **Control Flow**:
    - Retrieve the number of key-value pairs from the source context using `gguf_get_n_kv(src)`.
    - Iterate over each key-value pair in the source context.
    - For each key-value pair, check if it is an array or a single value.
    - If it is a single value, determine its type and call the appropriate `gguf_set_val_*` function to set the value in the destination context.
    - If it is an array, retrieve the number of elements and call [`gguf_set_arr_data`](#gguf_set_arr_data) or [`gguf_set_arr_str`](#gguf_set_arr_str) based on the type to set the array data in the destination context.
    - Continue this process until all key-value pairs from the source context have been processed.
- **Output**: The function does not return a value; it modifies the destination `gguf_context` by adding or updating key-value pairs based on the source context.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)
    - [`gguf_set_val_u8`](#gguf_set_val_u8)
    - [`gguf_set_val_i8`](#gguf_set_val_i8)
    - [`gguf_set_val_u16`](#gguf_set_val_u16)
    - [`gguf_set_val_i16`](#gguf_set_val_i16)
    - [`gguf_set_val_u32`](#gguf_set_val_u32)
    - [`gguf_set_val_i32`](#gguf_set_val_i32)
    - [`gguf_set_val_f32`](#gguf_set_val_f32)
    - [`gguf_set_val_u64`](#gguf_set_val_u64)
    - [`gguf_set_val_i64`](#gguf_set_val_i64)
    - [`gguf_set_val_f64`](#gguf_set_val_f64)
    - [`gguf_set_val_bool`](#gguf_set_val_bool)
    - [`gguf_set_val_str`](#gguf_set_val_str)
    - [`gguf_set_arr_data`](#gguf_set_arr_data)
    - [`gguf_set_arr_str`](#gguf_set_arr_str)


---
### gguf\_add\_tensor<!-- {{#callable:gguf_add_tensor}} -->
Adds a tensor to the `gguf_context`, ensuring no duplicate tensor names exist.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure where the tensor will be added.
    - `tensor`: A pointer to the `ggml_tensor` structure representing the tensor to be added.
- **Control Flow**:
    - Asserts that the `tensor` pointer is not null.
    - Checks if a tensor with the same name already exists in the context using [`gguf_find_tensor`](#gguf_find_tensor).
    - If a duplicate tensor name is found, the function aborts with an error message.
    - Creates a `gguf_tensor_info` structure to hold the tensor information.
    - Calculates the offset for the new tensor based on the previous tensor's offset and alignment.
    - Pushes the new tensor information into the `info` vector of the `gguf_context`.
- **Output**: The function does not return a value; it modifies the `gguf_context` by adding the new tensor information.
- **Functions called**:
    - [`gguf_find_tensor`](#gguf_find_tensor)
    - [`ggml_nbytes`](ggml.c.driver.md#ggml_nbytes)


---
### gguf\_set\_tensor\_type<!-- {{#callable:gguf_set_tensor_type}} -->
Sets the type of a tensor in the given context based on its name.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds information about tensors.
    - `name`: A string representing the name of the tensor whose type is to be set.
    - `type`: An enumeration value of type `ggml_type` representing the new type to be assigned to the tensor.
- **Control Flow**:
    - The function first calls [`gguf_find_tensor`](#gguf_find_tensor) to locate the tensor by its name, returning its ID.
    - If the tensor is not found (ID is negative), it aborts execution with an error message.
    - It retrieves the tensor structure from the context using the found tensor ID.
    - The size of the new type and its block size are calculated using [`ggml_type_size`](ggml.c.driver.md#ggml_type_size) and [`ggml_blck_size`](ggml.c.driver.md#ggml_blck_size) respectively.
    - The tensor's type is updated to the new type.
    - An assertion checks that the first dimension of the tensor is divisible by the block size of the new type.
    - The tensor's byte sizes for each dimension are recalculated based on the new type.
    - Finally, it updates the offsets of all subsequent tensors in the context to maintain correct memory alignment.
- **Output**: The function does not return a value; it modifies the tensor's type and its associated metadata directly within the context.
- **Functions called**:
    - [`gguf_find_tensor`](#gguf_find_tensor)
    - [`ggml_type_size`](ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](ggml.c.driver.md#ggml_blck_size)
    - [`gguf_get_n_tensors`](#gguf_get_n_tensors)
    - [`ggml_nbytes`](ggml.c.driver.md#ggml_nbytes)


---
### gguf\_set\_tensor\_data<!-- {{#callable:gguf_set_tensor_data}} -->
Sets the data for a specified tensor in the given context.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds the context in which the tensor is defined.
    - `name`: A string representing the name of the tensor whose data is to be set.
    - `data`: A pointer to the data that will be assigned to the tensor.
- **Control Flow**:
    - The function first calls [`gguf_find_tensor`](#gguf_find_tensor) to locate the tensor by its name in the context.
    - If the tensor is not found (i.e., `tensor_id` is negative), it triggers an abort with an error message.
    - If the tensor is found, it assigns the provided data to the tensor's data field, casting it to a non-const pointer.
- **Output**: The function does not return a value; it modifies the tensor's data in place.
- **Functions called**:
    - [`gguf_find_tensor`](#gguf_find_tensor)


---
### gguf\_write\_to\_buf<!-- {{#callable:gguf_write_to_buf}} -->
Writes the contents of a `gguf_context` to a buffer in a specific format.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure containing the data to be written.
    - `buf`: A reference to a `std::vector<int8_t>` that will hold the written data.
    - `only_meta`: A boolean flag indicating whether to write only metadata (true) or include tensor data (false).
- **Control Flow**:
    - Creates a `gguf_writer` instance to facilitate writing to the buffer.
    - Retrieves the number of key-value pairs and tensors from the `gguf_context`.
    - Writes a header to the buffer, including magic numbers, version, number of tensors, and number of key-value pairs.
    - Iterates over the key-value pairs and writes each to the buffer.
    - Iterates over the tensor information and writes each tensor's metadata to the buffer.
    - Pads the buffer to ensure alignment as specified in the context.
    - If `only_meta` is false, writes the actual tensor data to the buffer.
- **Output**: The function does not return a value; it modifies the provided buffer to contain the serialized data from the `gguf_context`.
- **Functions called**:
    - [`gguf_get_n_kv`](#gguf_get_n_kv)
    - [`gguf_get_n_tensors`](#gguf_get_n_tensors)


---
### gguf\_write\_to\_file<!-- {{#callable:gguf_write_to_file}} -->
Writes the contents of a `gguf_context` to a specified file.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure containing the data to be written.
    - `fname`: A string representing the name of the file to which the data will be written.
    - `only_meta`: A boolean flag indicating whether to write only metadata (true) or both metadata and data (false).
- **Control Flow**:
    - Attempts to open the specified file in binary write mode.
    - If the file cannot be opened, logs an error and returns false.
    - Creates a buffer to hold the data to be written by calling [`gguf_write_to_buf`](#gguf_write_to_buf).
    - Writes the contents of the buffer to the file and checks if the write operation was successful.
    - Closes the file and returns the result of the write operation.
- **Output**: Returns true if the data was successfully written to the file, otherwise returns false.
- **Functions called**:
    - [`ggml_fopen`](ggml.c.driver.md#ggml_fopen)
    - [`gguf_write_to_buf`](#gguf_write_to_buf)


---
### gguf\_get\_meta\_size<!-- {{#callable:gguf_get_meta_size}} -->
Calculates the size of the metadata for a given `gguf_context`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains metadata information.
- **Control Flow**:
    - Creates an empty vector `buf` to hold the serialized metadata.
    - Calls the [`gguf_write_to_buf`](#gguf_write_to_buf) function with `ctx`, `buf`, and a flag indicating that only metadata should be written.
    - Returns the size of the `buf` vector, which contains the serialized metadata.
- **Output**: Returns the size of the metadata in bytes as a `size_t` value.
- **Functions called**:
    - [`gguf_write_to_buf`](#gguf_write_to_buf)


---
### gguf\_get\_meta\_data<!-- {{#callable:gguf_get_meta_data}} -->
Retrieves metadata from a `gguf_context` and copies it into a provided buffer.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the metadata to be retrieved.
    - `data`: A pointer to a memory location where the metadata will be copied.
- **Control Flow**:
    - A vector of `int8_t` is created to hold the metadata buffer.
    - The function [`gguf_write_to_buf`](#gguf_write_to_buf) is called with the context and the buffer to fill it with metadata.
    - The contents of the buffer are then copied to the memory location pointed to by `data` using `memcpy`.
- **Output**: The function does not return a value; instead, it populates the provided `data` pointer with the metadata from the `gguf_context`.
- **Functions called**:
    - [`gguf_write_to_buf`](#gguf_write_to_buf)


