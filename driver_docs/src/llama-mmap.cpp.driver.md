# Purpose
This C++ source code file provides a comprehensive implementation for handling file operations and memory management across different operating systems, specifically focusing on Windows and POSIX-compliant systems. The file defines three main classes: [`llama_file`](#llama_filellama_file), [`llama_mmap`](#llama_mmapllama_mmap), and [`llama_mlock`](#llama_mlockllama_mlock), each encapsulating specific functionalities related to file handling, memory mapping, and memory locking, respectively. The [`llama_file`](#llama_filellama_file) class provides an abstraction for file operations, including opening, reading, writing, and seeking within files, with platform-specific implementations to handle differences between Windows and POSIX systems. The [`llama_mmap`](#llama_mmapllama_mmap) class facilitates memory mapping of files, allowing for efficient file access by mapping file contents directly into memory, again with distinct implementations for Windows and POSIX systems. The [`llama_mlock`](#llama_mlockllama_mlock) class is responsible for locking memory pages to prevent them from being swapped out, ensuring that critical data remains in physical memory.

The code is structured to handle platform-specific differences using preprocessor directives, ensuring compatibility and optimized performance across different environments. It includes error handling mechanisms to provide informative error messages and suggestions for resolving common issues, such as insufficient memory lock limits. The file also defines utility functions and constants, such as [`llama_format_win_err`](#llama_format_win_err) for formatting Windows error messages and [`llama_path_max`](#llama_path_max) for retrieving the maximum path length. Overall, this file serves as a foundational component for applications requiring robust file and memory management capabilities, providing a unified interface that abstracts away the complexities of cross-platform differences.
# Imports and Dependencies

---
- `llama-mmap.h`
- `llama-impl.h`
- `ggml.h`
- `cstring`
- `climits`
- `stdexcept`
- `cerrno`
- `algorithm`
- `unistd.h`
- `sys/mman.h`
- `fcntl.h`
- `sys/resource.h`
- `windows.h`
- `io.h`
- `TargetConditionals.h`


# Global Variables

---
### SUPPORTED
- **Type**: `const bool`
- **Description**: The `SUPPORTED` variable is a constant boolean that indicates whether the `llama_mlock` feature is supported on the current platform. It is set to `false` in the provided code, suggesting that the feature is not supported.
- **Use**: This variable is used to conditionally enable or disable functionality related to memory locking based on platform support.


# Data Structures

---
### llama\_file<!-- {{#data_structure:llama_file}} -->
- **Description**: [See definition](llama-mmap.h.driver.md#llama_file)
- **Member Functions**:
    - [`llama_file::llama_file`](#llama_filellama_file)
    - [`llama_file::~llama_file`](#llama_filellama_file)
    - [`llama_file::tell`](#llama_filetell)
    - [`llama_file::size`](#llama_filesize)
    - [`llama_file::file_id`](#llama_filefile_id)
    - [`llama_file::seek`](#llama_fileseek)
    - [`llama_file::read_raw`](#llama_fileread_raw)
    - [`llama_file::read_u32`](#llama_fileread_u32)
    - [`llama_file::write_raw`](#llama_filewrite_raw)
    - [`llama_file::write_u32`](#llama_filewrite_u32)

**Methods**

---
#### llama\_file::llama\_file<!-- {{#callable:llama_file::llama_file}} -->
The `llama_file` constructor initializes a `llama_file` object by opening a file with the specified name and mode, and setting up its internal implementation details.
- **Inputs**:
    - `fname`: A constant character pointer representing the name of the file to be opened.
    - `mode`: A constant character pointer representing the mode in which the file should be opened (e.g., read, write).
- **Control Flow**:
    - The constructor initializes the `pimpl` member with a new instance of the `impl` struct, passing `fname` and `mode` to its constructor.
    - The `impl` constructor attempts to open the file using `ggml_fopen` with the provided `fname` and `mode`.
    - If the file cannot be opened, a `std::runtime_error` is thrown with an appropriate error message.
    - On Windows, the file handle is retrieved using `_get_osfhandle` and stored in `fp_win32`.
    - The file pointer is moved to the end to determine the file size, then reset to the beginning.
- **Output**: A `llama_file` object is constructed and initialized, ready for file operations.
- **See also**: [`llama_file`](llama-mmap.h.driver.md#llama_file)  (Data Structure)


---
#### llama\_file::\~llama\_file<!-- {{#callable:llama_file::~llama_file}} -->
The `llama_file::~llama_file()` function is the destructor for the `llama_file` class, which automatically cleans up resources when a `llama_file` object is destroyed.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as `= default`, indicating that the compiler will generate the default destructor implementation.
    - The destructor will automatically be called when a `llama_file` object goes out of scope or is explicitly deleted.
    - The destructor will ensure that the `std::unique_ptr<impl> pimpl` member is properly destroyed, which in turn will invoke the destructor of the `impl` class, handling any necessary cleanup of file resources.
- **Output**: The function does not return any value as it is a destructor.
- **See also**: [`llama_file`](llama-mmap.h.driver.md#llama_file)  (Data Structure)


---
#### llama\_file::tell<!-- {{#callable:llama_file::tell}} -->
The `tell` function returns the current position of the file pointer within the file associated with the `llama_file` object.
- **Inputs**: None
- **Control Flow**:
    - The function calls the `tell` method of the `impl` struct through the `pimpl` pointer.
    - The `impl::tell` method retrieves the current file pointer position using platform-specific methods (either `SetFilePointerEx` on Windows or `ftell` on other systems).
    - If the retrieval of the file pointer position fails, an exception is thrown with an appropriate error message.
- **Output**: The function returns a `size_t` value representing the current position of the file pointer.
- **See also**: [`llama_file`](llama-mmap.h.driver.md#llama_file)  (Data Structure)


---
#### llama\_file::size<!-- {{#callable:llama_file::size}} -->
The `size` function of the `llama_file` class returns the size of the file as stored in the private implementation (`impl`) of the class.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private member `pimpl` of the `llama_file` class, which is a unique pointer to an `impl` object.
    - It returns the `size` member of the `impl` object, which represents the size of the file.
- **Output**: The function returns a `size_t` value representing the size of the file.
- **See also**: [`llama_file`](llama-mmap.h.driver.md#llama_file)  (Data Structure)


---
#### llama\_file::file\_id<!-- {{#callable:llama_file::file_id}} -->
The `file_id` function returns the file descriptor associated with the file pointer in the `llama_file` class, using platform-specific methods.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Check if the platform is Windows (_WIN32).
    - If on Windows, use `_fileno` to get the file descriptor from `pimpl->fp`.
    - If not on Windows, check if `fileno` is defined.
    - If `fileno` is defined, use it to get the file descriptor from `pimpl->fp`.
    - If `fileno` is not defined, use the global `::fileno` function to get the file descriptor from `pimpl->fp`.
- **Output**: Returns an integer representing the file descriptor of the file associated with the `llama_file` object.
- **See also**: [`llama_file`](llama-mmap.h.driver.md#llama_file)  (Data Structure)


---
#### llama\_file::seek<!-- {{#callable:llama_file::seek}} -->
The `seek` function in the `llama_file` class sets the file position to a specified offset based on a given reference point.
- **Inputs**:
    - `offset`: A `size_t` value representing the number of bytes to offset from the reference point specified by `whence`.
    - `whence`: An `int` value that specifies the reference point for the offset, which can be `SEEK_SET`, `SEEK_CUR`, or `SEEK_END`.
- **Control Flow**:
    - The function calls the `seek` method of the `impl` class, passing the `offset` and `whence` parameters.
    - The `impl::seek` method uses platform-specific code to adjust the file pointer.
    - On Windows, it uses `SetFilePointerEx` to set the file pointer based on the `offset` and `whence`.
    - On non-Windows systems, it uses `fseek` or `_fseeki64` to adjust the file pointer.
    - If the operation fails, a runtime error is thrown with a descriptive error message.
- **Output**: This function does not return a value; it performs an action on the file's position.
- **See also**: [`llama_file`](llama-mmap.h.driver.md#llama_file)  (Data Structure)


---
#### llama\_file::read\_raw<!-- {{#callable:llama_file::read_raw}} -->
The `read_raw` function reads a specified number of bytes from a file into a given memory location.
- **Inputs**:
    - `ptr`: A pointer to the memory location where the data read from the file will be stored.
    - `len`: The number of bytes to read from the file.
- **Control Flow**:
    - The function calls the `read_raw` method of the `impl` class, passing the `ptr` and `len` arguments.
    - The `impl::read_raw` method reads data from the file in chunks, ensuring that the entire specified length is read.
    - If an error occurs during reading, or if the end of the file is unexpectedly reached, an exception is thrown.
- **Output**: The function does not return a value; it modifies the memory pointed to by `ptr` with the data read from the file.
- **See also**: [`llama_file`](llama-mmap.h.driver.md#llama_file)  (Data Structure)


---
#### llama\_file::read\_u32<!-- {{#callable:llama_file::read_u32}} -->
The `read_u32` function reads a 32-bit unsigned integer from a file using the `llama_file` class.
- **Inputs**: None
- **Control Flow**:
    - The function `read_u32` is a member of the `llama_file` class and is marked as `const`, indicating it does not modify the state of the object.
    - It calls the `read_u32` method on the `pimpl` object, which is a pointer to the `impl` structure that handles the actual file operations.
    - The `impl::read_u32` method reads raw data from the file into a `uint32_t` variable using the `read_raw` method, which handles reading a specified number of bytes from the file.
    - The `read_raw` method ensures that the entire requested length is read, handling partial reads and errors appropriately.
    - Finally, the `impl::read_u32` method returns the read `uint32_t` value.
- **Output**: The function returns a `uint32_t` value, which is the 32-bit unsigned integer read from the file.
- **See also**: [`llama_file`](llama-mmap.h.driver.md#llama_file)  (Data Structure)


---
#### llama\_file::write\_raw<!-- {{#callable:llama_file::write_raw}} -->
The `write_raw` function writes a specified number of bytes from a given memory location to a file.
- **Inputs**:
    - `ptr`: A pointer to the memory location from which data will be written to the file.
    - `len`: The number of bytes to write from the memory location to the file.
- **Control Flow**:
    - The function calls the `write_raw` method of the `impl` class, passing the `ptr` and `len` arguments.
    - The `impl::write_raw` method writes data in chunks to the file, handling errors and ensuring all bytes are written.
- **Output**: The function does not return any value.
- **See also**: [`llama_file`](llama-mmap.h.driver.md#llama_file)  (Data Structure)


---
#### llama\_file::write\_u32<!-- {{#callable:llama_file::write_u32}} -->
The `write_u32` function writes a 32-bit unsigned integer to a file using the implementation details encapsulated in the `impl` class.
- **Inputs**:
    - `val`: A 32-bit unsigned integer (`uint32_t`) that is to be written to the file.
- **Control Flow**:
    - The function `write_u32` is a member of the `llama_file` class and is marked as `const`, indicating it does not modify the state of the `llama_file` object.
    - It delegates the task of writing the 32-bit unsigned integer to the `write_u32` method of the `impl` class, which is accessed through the `pimpl` pointer.
    - The `impl` class handles the actual file writing operation, ensuring the integer is written to the file correctly.
- **Output**: This function does not return any value.
- **See also**: [`llama_file`](llama-mmap.h.driver.md#llama_file)  (Data Structure)



---
### llama\_mmap<!-- {{#data_structure:llama_mmap}} -->
- **Description**: [See definition](llama-mmap.h.driver.md#llama_mmap)
- **Member Functions**:
    - [`llama_mmap::llama_mmap`](llama-mmap.h.driver.md#llama_mmapllama_mmap)
    - [`llama_mmap::llama_mmap`](#llama_mmapllama_mmap)
    - [`llama_mmap::~llama_mmap`](#llama_mmapllama_mmap)
    - [`llama_mmap::size`](#llama_mmapsize)
    - [`llama_mmap::addr`](#llama_mmapaddr)
    - [`llama_mmap::unmap_fragment`](#llama_mmapunmap_fragment)

**Methods**

---
#### llama\_mmap::llama\_mmap<!-- {{#callable:llama_mmap::llama_mmap}} -->
The `llama_mmap` constructor initializes a memory-mapped file object using the provided file, prefetch size, and NUMA settings.
- **Inputs**:
    - `file`: A pointer to a `llama_file` structure representing the file to be memory-mapped.
    - `prefetch`: A size_t value indicating the number of bytes to prefetch; defaults to -1 if not specified.
    - `numa`: A boolean indicating whether NUMA (Non-Uniform Memory Access) should be used; defaults to false.
- **Control Flow**:
    - The constructor initializes the `pimpl` member with a new `impl` object, passing the `file`, `prefetch`, and `numa` parameters to the `impl` constructor.
    - The `impl` constructor handles the platform-specific logic for memory-mapping the file, including setting up the memory map and handling prefetching and NUMA settings if applicable.
- **Output**: The function does not return a value; it initializes the `llama_mmap` object.
- **See also**: [`llama_mmap`](llama-mmap.h.driver.md#llama_mmap)  (Data Structure)


---
#### llama\_mmap::\~llama\_mmap<!-- {{#callable:llama_mmap::~llama_mmap}} -->
The `llama_mmap::~llama_mmap` function is the default destructor for the `llama_mmap` class, which automatically handles the cleanup of resources when an instance of `llama_mmap` is destroyed.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as `= default`, indicating that the compiler will generate the default destructor implementation.
    - The destructor will automatically clean up the resources managed by the `llama_mmap` class, specifically the `std::unique_ptr<impl> pimpl` member, which will invoke the destructor of the `impl` class.
- **Output**: The function does not return any output as it is a destructor.
- **See also**: [`llama_mmap`](llama-mmap.h.driver.md#llama_mmap)  (Data Structure)


---
#### llama\_mmap::size<!-- {{#callable:llama_mmap::size}} -->
The `size` function in the `llama_mmap` class returns the size of the memory-mapped file.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private member `pimpl` which is a unique pointer to an `impl` object.
    - It returns the `size` member of the `impl` object, which represents the size of the memory-mapped file.
- **Output**: The function returns a `size_t` value representing the size of the memory-mapped file.
- **See also**: [`llama_mmap`](llama-mmap.h.driver.md#llama_mmap)  (Data Structure)


---
#### llama\_mmap::addr<!-- {{#callable:llama_mmap::addr}} -->
The `addr` function returns the memory address mapped by the `llama_mmap` object.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private member `pimpl` of the `llama_mmap` class, which is a unique pointer to an `impl` object.
    - It returns the `addr` member of the `impl` object, which is a pointer to the mapped memory address.
- **Output**: A pointer to the memory address mapped by the `llama_mmap` object.
- **See also**: [`llama_mmap`](llama-mmap.h.driver.md#llama_mmap)  (Data Structure)


---
#### llama\_mmap::unmap\_fragment<!-- {{#callable:llama_mmap::unmap_fragment}} -->
The `unmap_fragment` function in the `llama_mmap` class unmaps a specified memory fragment from a memory-mapped file.
- **Inputs**:
    - `first`: The starting byte position of the memory fragment to be unmapped.
    - `last`: The ending byte position of the memory fragment to be unmapped.
- **Control Flow**:
    - The function calls the `unmap_fragment` method of the `impl` class, passing the `first` and `last` parameters.
    - In the `impl` class, the `unmap_fragment` method aligns the `first` and `last` positions to the system's page size.
    - If the length of the fragment to be unmapped is zero, the function returns immediately.
    - The function checks that the aligned `first` and `last` positions are valid and then calls `munmap` to unmap the memory region.
    - The function updates the `mapped_fragments` vector to reflect the unmapped region by removing or splitting existing fragments as necessary.
- **Output**: The function does not return any value.
- **See also**: [`llama_mmap`](llama-mmap.h.driver.md#llama_mmap)  (Data Structure)



---
### llama\_mlock<!-- {{#data_structure:llama_mlock}} -->
- **Description**: [See definition](llama-mmap.h.driver.md#llama_mlock)
- **Member Functions**:
    - [`llama_mlock::llama_mlock`](#llama_mlockllama_mlock)
    - [`llama_mlock::~llama_mlock`](#llama_mlockllama_mlock)
    - [`llama_mlock::init`](#llama_mlockinit)
    - [`llama_mlock::grow_to`](#llama_mlockgrow_to)

**Methods**

---
#### llama\_mlock::llama\_mlock<!-- {{#callable:llama_mlock::llama_mlock}} -->
The `llama_mlock` constructor initializes a `llama_mlock` object by creating a unique pointer to its implementation.
- **Inputs**: None
- **Control Flow**:
    - The constructor `llama_mlock::llama_mlock()` is called.
    - A unique pointer `pimpl` is created and initialized with a new instance of the `impl` struct.
- **Output**: A `llama_mlock` object with its implementation initialized.
- **See also**: [`llama_mlock`](llama-mmap.h.driver.md#llama_mlock)  (Data Structure)


---
#### llama\_mlock::\~llama\_mlock<!-- {{#callable:llama_mlock::~llama_mlock}} -->
The destructor `~llama_mlock()` is a default destructor for the `llama_mlock` class, which automatically cleans up resources when an instance of the class is destroyed.
- **Inputs**: None
- **Control Flow**:
    - The destructor `~llama_mlock()` is defined as `= default;`, indicating that it uses the compiler-generated default behavior for destructors.
    - This default behavior will automatically handle the destruction of the `pimpl` unique pointer, which will in turn call the destructor of the `impl` class, releasing any resources managed by `impl`.
- **Output**: The destructor does not return any value or output.
- **See also**: [`llama_mlock`](llama-mmap.h.driver.md#llama_mlock)  (Data Structure)


---
#### llama\_mlock::init<!-- {{#callable:llama_mlock::init}} -->
The `init` function initializes the `llama_mlock` object with a given memory address.
- **Inputs**:
    - `ptr`: A pointer to a memory address that the `llama_mlock` object will be initialized with.
- **Control Flow**:
    - The function calls the `init` method of the `impl` class, passing the `ptr` argument to it.
    - The `impl::init` method asserts that the current address (`addr`) is `NULL` and the size is `0`, ensuring it hasn't been initialized before.
    - The `impl::init` method then sets the `addr` member to the provided `ptr`.
- **Output**: The function does not return any value.
- **See also**: [`llama_mlock`](llama-mmap.h.driver.md#llama_mlock)  (Data Structure)


---
#### llama\_mlock::grow\_to<!-- {{#callable:llama_mlock::grow_to}} -->
The `grow_to` function in the `llama_mlock` class increases the memory lock size to a specified target size.
- **Inputs**:
    - `target_size`: The desired size to which the memory lock should be grown, specified in bytes.
- **Control Flow**:
    - The function calls the `grow_to` method on the `pimpl` object, which is a unique pointer to an `impl` instance.
    - The `impl` class's `grow_to` method checks if the current memory lock size is less than the target size.
    - If the target size is greater, it attempts to lock additional memory up to the target size using platform-specific mechanisms.
    - If the lock operation fails, it sets a flag to prevent further attempts.
- **Output**: The function does not return any value.
- **See also**: [`llama_mlock`](llama-mmap.h.driver.md#llama_mlock)  (Data Structure)



# Functions

---
### llama\_format\_win\_err<!-- {{#callable:llama_format_win_err}} -->
The `llama_format_win_err` function formats a Windows error code into a human-readable string using the system's error message.
- **Inputs**:
    - `err`: A `DWORD` representing the Windows error code to be formatted.
- **Control Flow**:
    - Allocate a buffer for the error message using `FormatMessageA` with flags to allocate the buffer, use system messages, and ignore inserts.
    - Check if the `FormatMessageA` call was successful by verifying the size of the message; if not, return an error message indicating failure.
    - If successful, construct a `std::string` from the buffer and the size of the message.
    - Free the allocated buffer using `LocalFree`.
    - Return the constructed string containing the formatted error message.
- **Output**: A `std::string` containing the formatted error message or an error message indicating that `FormatMessageA` failed.


---
### GetErrorMessageWin32<!-- {{#callable:llama_file::impl::GetErrorMessageWin32}} -->
The `GetErrorMessageWin32` function retrieves a human-readable error message string corresponding to a given Win32 error code.
- **Inputs**:
    - `error_code`: A `DWORD` representing the Win32 error code for which the error message is to be retrieved.
- **Control Flow**:
    - Initialize an empty string `ret` to store the error message.
    - Declare a pointer `lpMsgBuf` to hold the formatted message buffer.
    - Call `FormatMessageA` with flags to allocate a buffer and retrieve the system message for the given `error_code`.
    - Check if `FormatMessageA` returns a non-zero buffer length (`bufLen`).
    - If `bufLen` is zero, format the error code into a string and assign it to `ret`.
    - If `bufLen` is non-zero, assign the message in `lpMsgBuf` to `ret` and free the allocated buffer using `LocalFree`.
    - Return the error message string `ret`.
- **Output**: A `std::string` containing the error message corresponding to the provided Win32 error code, or a formatted string with the error code if the message retrieval fails.
- **Functions called**:
    - [`format`](llama-impl.cpp.driver.md#format)


---
### impl<!-- {{#callable:llama_mlock::impl::impl}} -->
The `impl` constructor initializes a `llama_mlock::impl` object with default values for its member variables.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes the `addr` member to `NULL`, indicating no memory address is currently locked.
    - The `size` member is set to `0`, indicating no memory size is currently locked.
    - The `failed_already` member is set to `false`, indicating that no previous lock attempt has failed.
- **Output**: The constructor does not return any value as it is a default constructor for the `llama_mlock::impl` class.


---
### tell<!-- {{#callable:llama_file::impl::tell}} -->
The `tell` function returns the current position of the file pointer in a file stream.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the code is being compiled on a Windows platform using the `_WIN32` macro.
    - If on Windows, it uses `_ftelli64` to get the current file position; otherwise, it uses `std::ftell`.
    - The function checks if the return value is `-1`, indicating an error, and throws a `std::runtime_error` with an error message if so.
    - Finally, it returns the file position cast to a `size_t`.
- **Output**: The function returns the current position of the file pointer as a `size_t`.
- **Functions called**:
    - [`format`](llama-impl.cpp.driver.md#format)


---
### seek<!-- {{#callable:llama_file::impl::seek}} -->
The `seek` function adjusts the file position indicator for a file stream to a specified offset based on a given reference point.
- **Inputs**:
    - `offset`: A `size_t` value representing the number of bytes to offset the file position indicator.
    - `whence`: An `int` value indicating the reference point for the offset, such as `SEEK_SET`, `SEEK_CUR`, or `SEEK_END`.
- **Control Flow**:
    - The function checks if the code is being compiled on a Windows platform using the `_WIN32` macro.
    - If on Windows, it uses `_fseeki64` to set the file position indicator to the specified offset from the reference point `whence`.
    - If not on Windows, it uses `std::fseek` to perform the same operation.
    - The function checks the return value of the seek operation; if it is not zero, it throws a `std::runtime_error` with an error message indicating a seek error.
- **Output**: The function does not return a value but may throw a `std::runtime_error` if the seek operation fails.
- **Functions called**:
    - [`format`](llama-impl.cpp.driver.md#format)


---
### read\_raw<!-- {{#callable:llama_file::impl::read_raw}} -->
The `read_raw` function reads a specified number of bytes from a file into a buffer and handles errors related to file reading.
- **Inputs**:
    - `ptr`: A pointer to the buffer where the read data will be stored.
    - `len`: The number of bytes to read from the file.
- **Control Flow**:
    - Check if the length `len` is zero; if so, return immediately as no reading is needed.
    - Set `errno` to 0 to clear any previous error state.
    - Attempt to read `len` bytes from the file into the buffer pointed to by `ptr` using `std::fread`.
    - Check if a file error occurred using `ferror`; if true, throw a runtime error with the error message.
    - Check if the number of items read (`ret`) is not equal to 1, indicating an unexpected end of file, and throw a runtime error if so.
- **Output**: The function does not return a value but throws a runtime error if a read error occurs or if the end of the file is unexpectedly reached.
- **Functions called**:
    - [`format`](llama-impl.cpp.driver.md#format)


---
### read\_u32<!-- {{#callable:llama_file::impl::read_u32}} -->
The `read_u32` function reads a 32-bit unsigned integer from a file or memory-mapped file.
- **Inputs**: None
- **Control Flow**:
    - Declare a variable `ret` of type `uint32_t` to store the result.
    - Call the [`read_raw`](#llama_fileread_raw) method with the address of `ret` and the size of `ret` to read raw data into `ret`.
    - Return the value stored in `ret`.
- **Output**: A 32-bit unsigned integer read from the file or memory-mapped file.
- **Functions called**:
    - [`llama_file::read_raw`](#llama_fileread_raw)


---
### write\_raw<!-- {{#callable:llama_file::impl::write_raw}} -->
The `write_raw` function writes a specified number of bytes from a given memory location to a file, throwing an error if the write operation fails.
- **Inputs**:
    - `ptr`: A pointer to the memory location from which data is to be written to the file.
    - `len`: The number of bytes to write from the memory location pointed to by `ptr`.
- **Control Flow**:
    - Check if `len` is zero; if so, return immediately without performing any write operation.
    - Set `errno` to zero to clear any previous error state.
    - Attempt to write `len` bytes from `ptr` to the file using `std::fwrite`.
    - Check if the number of items written is not equal to 1, indicating a write error.
    - If a write error occurs, throw a `std::runtime_error` with a formatted error message including the error string from `strerror(errno)`.
- **Output**: The function does not return a value, but it may throw a `std::runtime_error` if the write operation fails.
- **Functions called**:
    - [`format`](llama-impl.cpp.driver.md#format)


---
### write\_u32<!-- {{#callable:llama_file::impl::write_u32}} -->
The `write_u32` function writes a 32-bit unsigned integer to a file using a raw write operation.
- **Inputs**:
    - `val`: A 32-bit unsigned integer (`uint32_t`) that is to be written to the file.
- **Control Flow**:
    - The function calls [`write_raw`](#llama_filewrite_raw), passing the address of `val` and the size of `val` (which is 4 bytes for a `uint32_t`).
- **Output**: The function does not return any value.
- **Functions called**:
    - [`llama_file::write_raw`](#llama_filewrite_raw)


---
### \~impl<!-- {{#callable:llama_mmap::impl::~impl}} -->
The destructor `~impl` attempts to unmap a memory-mapped file view and logs a warning if it fails.
- **Inputs**: None
- **Control Flow**:
    - The function checks if `UnmapViewOfFile(addr)` returns false, indicating a failure to unmap the file view.
    - If the unmapping fails, it logs a warning message using `LLAMA_LOG_WARN`, including the formatted error message obtained from `llama_format_win_err(GetLastError())`.
- **Output**: This function does not return any value; it is a destructor and its purpose is to clean up resources.
- **Functions called**:
    - [`llama_format_win_err`](#llama_format_win_err)


---
### align\_range<!-- {{#callable:llama_mmap::impl::align_range}} -->
The `align_range` function adjusts the range defined by two pointers to align with a specified page size.
- **Inputs**:
    - `first`: A pointer to the starting address of the range to be aligned.
    - `last`: A pointer to the ending address of the range to be aligned.
    - `page_size`: The size of the page to which the range should be aligned.
- **Control Flow**:
    - Calculate the offset of the `first` pointer within the page using bitwise AND with `page_size - 1`.
    - Determine the offset needed to align `first` to the next page boundary, which is zero if already aligned, otherwise `page_size - offset_in_page`.
    - Increment the `first` pointer by the calculated offset to align it to the page boundary.
    - Align the `last` pointer to the previous page boundary by clearing the lower bits using bitwise AND with the negation of `page_size - 1`.
    - If the aligned `last` pointer is less than or equal to the aligned `first` pointer, set `last` to the value of `first`.
- **Output**: The function modifies the values pointed to by `first` and `last` to ensure they are aligned to the specified page size.


---
### unmap\_fragment<!-- {{#callable:llama_mmap::impl::unmap_fragment}} -->
The `unmap_fragment` function throws a runtime error indicating that memory mapping (mmap) is not supported.
- **Inputs**:
    - `first`: The starting index of the memory fragment to unmap.
    - `last`: The ending index of the memory fragment to unmap.
- **Control Flow**:
    - The function marks the input parameters `first` and `last` as unused using `GGML_UNUSED` macro.
    - It then throws a `std::runtime_error` with the message "mmap not supported".
- **Output**: The function does not return any value as it throws an exception.


---
### lock\_granularity<!-- {{#callable:llama_mlock::impl::lock_granularity}} -->
The `lock_granularity` function returns a fixed size value representing the lock granularity.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as a static method, meaning it can be called without an instance of the class.
    - It simply returns a constant value of 65536 cast to a `size_t` type.
- **Output**: The function returns a `size_t` value of 65536, representing the lock granularity.


---
### raw\_lock<!-- {{#callable:llama_mlock::impl::raw_lock}} -->
The `raw_lock` function logs a warning and returns false, indicating that memory locking is not supported on the current system.
- **Inputs**:
    - `addr`: A pointer to the starting address of the memory region to be locked.
    - `len`: The length in bytes of the memory region to be locked.
- **Control Flow**:
    - Logs a warning message indicating that memory locking is not supported on the system.
    - Returns false to indicate failure to lock the memory.
- **Output**: A boolean value, always false, indicating that the memory lock operation is not supported.


---
### raw\_unlock<!-- {{#callable:llama_mlock::impl::raw_unlock}} -->
The `raw_unlock` function is a static method intended to unlock a memory region, but it is currently implemented as an empty function.
- **Inputs**:
    - `addr`: A pointer to the starting address of the memory region to be unlocked.
    - `len`: The size of the memory region to be unlocked, in bytes.
- **Control Flow**:
    - The function is defined as a static method, indicating it is intended to be used without an instance of a class.
    - The function takes two parameters, `addr` and `len`, but does not perform any operations with them.
    - The function body is empty, suggesting it is either a placeholder or not needed in the current implementation.
- **Output**: The function does not return any value or perform any operations.


---
### init<!-- {{#callable:llama_mlock::impl::init}} -->
The `init` function initializes a memory lock by setting the address pointer to a given memory location, ensuring that it has not been previously set.
- **Inputs**:
    - `ptr`: A pointer to a memory location that will be used to initialize the memory lock.
- **Control Flow**:
    - The function asserts that the current address (`addr`) is `NULL` and the size is `0`, ensuring that the memory lock has not been initialized before.
    - The function then sets the `addr` to the provided `ptr`, effectively initializing the memory lock with the given memory location.
- **Output**: The function does not return any value.


---
### grow\_to<!-- {{#callable:llama_mlock::impl::grow_to}} -->
The `grow_to` function attempts to increase the size of a memory-locked region to a specified target size, ensuring alignment with system page size and handling potential locking failures.
- **Inputs**:
    - `target_size`: The desired size to which the memory-locked region should grow, specified in bytes.
- **Control Flow**:
    - Assert that the memory address (`addr`) is valid.
    - Check if a previous lock attempt has failed (`failed_already`), and return immediately if so.
    - Determine the system's memory lock granularity using `lock_granularity()`.
    - Adjust the `target_size` to align with the granularity by rounding up to the nearest multiple of the granularity.
    - If the adjusted `target_size` is greater than the current size, attempt to lock the additional memory region from the current size to the target size using `raw_lock()`.
    - If the lock is successful, update the current size to the new `target_size`; otherwise, set `failed_already` to true to indicate a failure.
- **Output**: The function does not return a value, but it updates the internal state of the object by potentially increasing the `size` and setting `failed_already` if locking fails.
- **Functions called**:
    - [`llama_mlock::impl::lock_granularity`](#impllock_granularity)
    - [`llama_mlock::impl::raw_lock`](#implraw_lock)


---
### llama\_path\_max<!-- {{#callable:llama_path_max}} -->
The `llama_path_max` function returns the maximum allowable path length for the system.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the macro `PATH_MAX`.
- **Output**: The function outputs a `size_t` value representing the maximum path length defined by `PATH_MAX`.


