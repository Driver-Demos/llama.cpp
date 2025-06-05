# Purpose
This C++ header file provides a narrow set of functionalities related to file handling, memory mapping, and memory locking, encapsulated within three main structures: `llama_file`, [`llama_mmap`](#llama_mmapllama_mmap), and `llama_mlock`. Each structure is designed to manage specific resources, such as file operations, memory mapping, and memory locking, using the Pimpl idiom to hide implementation details. The `llama_file` structure offers methods for file manipulation, including reading, writing, and seeking, while [`llama_mmap`](#llama_mmapllama_mmap) handles memory mapping operations with support for NUMA (Non-Uniform Memory Access) configurations. The `llama_mlock` structure provides functionality for memory locking and resizing. The use of `std::unique_ptr` ensures automatic resource management and memory safety. This header is intended to be included in other C++ files or libraries, providing a modular and encapsulated approach to resource management.
# Imports and Dependencies

---
- `cstdint`
- `memory`
- `vector`


# Data Structures

---
### llama\_file<!-- {{#data_structure:llama_file}} -->
- **Type**: `struct`
- **Members**:
    - `pimpl`: A unique pointer to an implementation struct, used for the Pimpl idiom to hide implementation details.
- **Description**: The `llama_file` struct is a file handling abstraction that provides an interface for file operations such as reading, writing, seeking, and obtaining file metadata. It uses the Pimpl idiom to encapsulate its implementation details, allowing for a clean separation between the interface and the implementation. This struct is designed to manage file operations efficiently and is part of a larger system that includes memory mapping and locking functionalities.
- **Member Functions**:
    - [`llama_file::llama_file`](llama-mmap.cpp.driver.md#llama_filellama_file)
    - [`llama_file::~llama_file`](llama-mmap.cpp.driver.md#llama_filellama_file)
    - [`llama_file::tell`](llama-mmap.cpp.driver.md#llama_filetell)
    - [`llama_file::size`](llama-mmap.cpp.driver.md#llama_filesize)
    - [`llama_file::file_id`](llama-mmap.cpp.driver.md#llama_filefile_id)
    - [`llama_file::seek`](llama-mmap.cpp.driver.md#llama_fileseek)
    - [`llama_file::read_raw`](llama-mmap.cpp.driver.md#llama_fileread_raw)
    - [`llama_file::read_u32`](llama-mmap.cpp.driver.md#llama_fileread_u32)
    - [`llama_file::write_raw`](llama-mmap.cpp.driver.md#llama_filewrite_raw)
    - [`llama_file::write_u32`](llama-mmap.cpp.driver.md#llama_filewrite_u32)


---
### llama\_mmap<!-- {{#data_structure:llama_mmap}} -->
- **Type**: `struct`
- **Members**:
    - `SUPPORTED`: A static constant boolean indicating if the feature is supported.
    - `pimpl`: A unique pointer to an implementation struct, used for the Pimpl idiom.
- **Description**: The `llama_mmap` struct is a memory-mapped file handler that encapsulates the details of memory mapping operations. It uses the Pimpl idiom to hide implementation details, providing a clean interface for operations such as retrieving the size of the mapped region, accessing the address of the mapped memory, and unmapping specific fragments. The struct is designed to be non-copyable, ensuring that each instance uniquely manages its resources. It also includes a static constant to indicate whether the memory mapping feature is supported.
- **Member Functions**:
    - [`llama_mmap::llama_mmap`](#llama_mmapllama_mmap)
    - [`llama_mmap::llama_mmap`](llama-mmap.cpp.driver.md#llama_mmapllama_mmap)
    - [`llama_mmap::~llama_mmap`](llama-mmap.cpp.driver.md#llama_mmapllama_mmap)
    - [`llama_mmap::size`](llama-mmap.cpp.driver.md#llama_mmapsize)
    - [`llama_mmap::addr`](llama-mmap.cpp.driver.md#llama_mmapaddr)
    - [`llama_mmap::unmap_fragment`](llama-mmap.cpp.driver.md#llama_mmapunmap_fragment)

**Methods**

---
#### llama\_mmap::llama\_mmap<!-- {{#callable:llama_mmap::llama_mmap}} -->
The `llama_mmap` constructor initializes a memory-mapped file object with optional prefetching and NUMA support.
- **Inputs**:
    - `file`: A pointer to a `llama_file` structure representing the file to be memory-mapped.
    - `prefetch`: An optional size_t value indicating the number of bytes to prefetch; defaults to the maximum size_t value if not specified.
    - `numa`: A boolean flag indicating whether NUMA (Non-Uniform Memory Access) support should be enabled; defaults to false.
- **Control Flow**:
    - The constructor is called with a `llama_file` pointer, an optional prefetch size, and an optional NUMA flag.
    - The constructor initializes the `llama_mmap` object, potentially setting up memory mapping with the specified prefetch size and NUMA settings.
    - The constructor does not allow copying of `llama_mmap` objects, as indicated by the deleted copy constructor.
- **Output**: The constructor does not return a value; it initializes the `llama_mmap` object.
- **See also**: [`llama_mmap`](#llama_mmap)  (Data Structure)



---
### llama\_mlock<!-- {{#data_structure:llama_mlock}} -->
- **Type**: `struct`
- **Members**:
    - `SUPPORTED`: A static constant boolean indicating if the feature is supported.
    - `pimpl`: A unique pointer to an implementation struct, used for the Pimpl idiom.
- **Description**: The `llama_mlock` struct is designed to manage memory locking operations, providing an interface to initialize and grow memory locks. It uses the Pimpl idiom to hide implementation details, encapsulated in a private `impl` struct, and offers a static constant `SUPPORTED` to indicate feature availability. The struct includes a constructor and destructor for managing its lifecycle, and methods to initialize and adjust the size of the memory lock.
- **Member Functions**:
    - [`llama_mlock::llama_mlock`](llama-mmap.cpp.driver.md#llama_mlockllama_mlock)
    - [`llama_mlock::~llama_mlock`](llama-mmap.cpp.driver.md#llama_mlockllama_mlock)
    - [`llama_mlock::init`](llama-mmap.cpp.driver.md#llama_mlockinit)
    - [`llama_mlock::grow_to`](llama-mmap.cpp.driver.md#llama_mlockgrow_to)


