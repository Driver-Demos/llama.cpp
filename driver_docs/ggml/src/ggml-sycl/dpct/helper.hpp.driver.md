# Purpose
The provided C++ source code file is a header file designed to facilitate the use of SYCL (a C++-based parallel programming model) for heterogeneous computing, particularly in the context of the LLVM project. It provides a comprehensive set of utilities and abstractions to manage SYCL devices, memory, and operations, making it easier to write code that can run on different hardware backends such as CPUs, GPUs, and accelerators. The file includes conditional compilation directives to support different platforms (Linux and Windows) and backends (Intel oneMKL, NVIDIA, AMD), ensuring compatibility across various environments.

Key components of this file include device management classes ([`device_ext`](#device_extdevice_ext), [`dev_mgr`](#dev_mgrdev_mgr)), memory management utilities ([`mem_mgr`](#mem_mgrmem_mgr), [`device_memory`](#device_memorydevice_memory)), and a variety of SYCL-related functions and templates for operations like matrix multiplication ([`gemm`](#dpctgemm), [`gemm_batch`](#dpctgemm_batch)) and memory copying ([`dpct_memcpy`](#detaildpct_memcpy)). It also defines several enumerations and structures to handle different data types and memory regions, providing a robust framework for SYCL-based development. The file is intended to be included in other C++ projects that require SYCL support, offering a broad range of functionalities to manage and utilize SYCL devices and resources efficiently.
# Imports and Dependencies

---
- `sycl/sycl.hpp`
- `sycl/half_type.hpp`
- `syclcompat/math.hpp`
- `map`
- `oneapi/mkl.hpp`
- `oneapi/math.hpp`
- `ggml.h`
- `sys/mman.h`
- `windows.h`
- `unistd.h`
- `sys/syscall.h`
- `sycl/info/aspects.def`
- `sycl/info/aspects_deprecated.def`


# Data Structures

---
### matrix\_info\_t<!-- {{#data_structure:matrix_info_t}} -->
- **Type**: `struct`
- **Members**:
    - `transpose_info`: An array of two elements of type `oneapi::math::transpose` that stores transpose information for matrices.
    - `value_info`: An array of two elements of type `Ts` that stores value-related information.
    - `size_info`: An array of three elements of type `std::int64_t` that stores size-related information for matrices.
    - `ld_info`: An array of three elements of type `std::int64_t` that stores leading dimension information for matrices.
    - `groupsize_info`: A single `std::int64_t` value that stores the group size information.
- **Description**: The `matrix_info_t` struct is a template data structure designed to encapsulate various matrix-related information, including transpose states, value information, size dimensions, leading dimensions, and group size. It is used in the context of matrix operations, particularly in batch processing scenarios, to store and manage metadata necessary for performing matrix computations efficiently. The template parameter `Ts` allows for flexibility in the type of value information stored, making the struct adaptable to different data types used in matrix operations.


---
### error\_code<!-- {{#data_structure:dpct::error_code}} -->
- **Type**: `enum`
- **Members**:
    - `success`: Represents a successful operation with a value of 0.
    - `default_error`: Represents a default error with a value of 999.
- **Description**: The `error_code` enum is a simple enumeration used to represent error states within the code. It defines two possible values: `success`, which indicates a successful operation, and `default_error`, which indicates a generic error condition. This enum can be used to standardize error handling by providing a consistent set of error codes that can be checked against when performing operations that may fail.


---
### memcpy\_direction<!-- {{#data_structure:dpct::memcpy_direction}} -->
- **Type**: `enum`
- **Members**:
    - `host_to_host`: Represents a memory copy operation from host to host.
    - `host_to_device`: Represents a memory copy operation from host to device.
    - `device_to_host`: Represents a memory copy operation from device to host.
    - `device_to_device`: Represents a memory copy operation from device to device.
    - `automatic`: Automatically determines the direction of the memory copy operation.
- **Description**: The `memcpy_direction` enum defines various directions for memory copy operations, specifically in the context of host and device interactions. It includes options for copying data between host and device, as well as between devices, and provides an automatic option to deduce the direction based on the context. This enum is useful in scenarios involving data transfer in heterogeneous computing environments, such as those using SYCL or other parallel computing frameworks.


---
### memory\_region<!-- {{#data_structure:dpct::memory_region}} -->
- **Type**: `enum`
- **Members**:
    - `global`: Represents device global memory.
    - `constant`: Represents device constant memory.
    - `local`: Represents device local memory.
    - `shared`: Represents memory accessible by both host and device.
- **Description**: The `memory_region` enum defines different types of memory regions available in a device, such as global, constant, local, and shared memory. Each enumerator corresponds to a specific type of memory that can be used in device programming, allowing developers to specify the memory region they intend to use for various operations.


---
### library\_data\_t<!-- {{#data_structure:dpct::library_data_t}} -->
- **Type**: `enum class`
- **Members**:
    - `real_float`: Represents a real floating-point data type.
    - `complex_float`: Represents a complex floating-point data type.
    - `real_double`: Represents a real double-precision floating-point data type.
    - `complex_double`: Represents a complex double-precision floating-point data type.
    - `real_half`: Represents a real half-precision floating-point data type.
    - `complex_half`: Represents a complex half-precision floating-point data type.
    - `real_bfloat16`: Represents a real bfloat16 data type.
    - `complex_bfloat16`: Represents a complex bfloat16 data type.
    - `real_int4`: Represents a real 4-bit integer data type.
    - `complex_int4`: Represents a complex 4-bit integer data type.
    - `real_uint4`: Represents a real 4-bit unsigned integer data type.
    - `complex_uint4`: Represents a complex 4-bit unsigned integer data type.
    - `real_int8`: Represents a real 8-bit integer data type.
    - `complex_int8`: Represents a complex 8-bit integer data type.
    - `real_uint8`: Represents a real 8-bit unsigned integer data type.
    - `complex_uint8`: Represents a complex 8-bit unsigned integer data type.
    - `real_int16`: Represents a real 16-bit integer data type.
    - `complex_int16`: Represents a complex 16-bit integer data type.
    - `real_uint16`: Represents a real 16-bit unsigned integer data type.
    - `complex_uint16`: Represents a complex 16-bit unsigned integer data type.
    - `real_int32`: Represents a real 32-bit integer data type.
    - `complex_int32`: Represents a complex 32-bit integer data type.
    - `real_uint32`: Represents a real 32-bit unsigned integer data type.
    - `complex_uint32`: Represents a complex 32-bit unsigned integer data type.
    - `real_int64`: Represents a real 64-bit integer data type.
    - `complex_int64`: Represents a complex 64-bit integer data type.
    - `real_uint64`: Represents a real 64-bit unsigned integer data type.
    - `complex_uint64`: Represents a complex 64-bit unsigned integer data type.
    - `real_int8_4`: Represents a real 8-bit integer data type with 4 components.
    - `real_int8_32`: Represents a real 8-bit integer data type with 32 components.
    - `real_uint8_4`: Represents a real 8-bit unsigned integer data type with 4 components.
    - `library_data_t_size`: Represents the size of the library_data_t enum.
- **Description**: The `library_data_t` enum class defines a set of constants representing various data types used in a library, including real and complex numbers in different precisions and integer types of various bit widths. This enum is used to specify the type of data being handled, allowing for operations to be performed on different data types in a consistent manner. The enum values are of type `unsigned char`, ensuring compact storage and efficient comparisons.


---
### DataType<!-- {{#data_structure:dpct::DataType}} -->
- **Type**: `struct`
- **Members**:
    - `T2`: An alias for the template parameter T.
- **Description**: The `DataType` struct is a templated data structure that provides a type alias `T2` for the template parameter `T`. This allows for a more convenient reference to the type `T` within the context where `DataType` is used. Additionally, there is a specialization of the `DataType` struct for `sycl::vec<T, 2>`, which redefines `T2` as `std::complex<T>`, indicating a specific handling for 2-element SYCL vectors by converting them to complex numbers.


---
### generic\_error\_type<!-- {{#data_structure:dpct::detail::generic_error_type}} -->
- **Type**: `class`
- **Members**:
    - `value`: A private member variable of type T that stores the error value.
- **Description**: The `generic_error_type` is a templated class designed to encapsulate an error value of a specified type T, associated with a tag type. It provides a default constructor and a parameterized constructor to initialize the error value. The class also includes a type conversion operator to allow implicit conversion to the underlying type T, facilitating easy retrieval of the stored error value. This class is useful for creating strongly-typed error codes or values in a type-safe manner.
- **Member Functions**:
    - [`dpct::detail::generic_error_type::generic_error_type`](#generic_error_typegeneric_error_type)
    - [`dpct::detail::generic_error_type::generic_error_type`](#generic_error_typegeneric_error_type)

**Methods**

---
#### generic\_error\_type::generic\_error\_type<!-- {{#callable:dpct::detail::generic_error_type::generic_error_type}} -->
The `generic_error_type` class is a template class that encapsulates an error value of type `T` and provides a conversion operator to retrieve this value.
- **Inputs**:
    - `T`: The type of the error value that the `generic_error_type` class will encapsulate.
- **Control Flow**:
    - The class has a default constructor that initializes the object without setting a value.
    - There is a parameterized constructor that initializes the object with a given value of type `T`.
    - The class provides a conversion operator to convert the `generic_error_type` object to its encapsulated value of type `T`.
- **Output**: An instance of `generic_error_type` can be implicitly converted to the encapsulated value of type `T`.
- **See also**: [`dpct::detail::generic_error_type`](#detailgeneric_error_type)  (Data Structure)


---
#### generic\_error\_type::generic\_error\_type<!-- {{#callable:dpct::detail::generic_error_type::generic_error_type}} -->
The `generic_error_type` class template provides a mechanism to encapsulate an error value of a specified type and allows implicit conversion back to that type.
- **Inputs**:
    - `tag`: A unique type used to differentiate between different specializations of the `generic_error_type` class template.
    - `T`: The type of the error value that the `generic_error_type` will encapsulate.
- **Control Flow**:
    - The class has a default constructor that initializes the error value to its default state.
    - A parameterized constructor is provided to initialize the error value with a specific value of type `T`.
    - An implicit conversion operator is defined to allow the `generic_error_type` object to be used as if it were of type `T`, returning the encapsulated error value.
- **Output**: The class does not produce an output directly, but it allows the encapsulated error value to be accessed and used as if it were of type `T`.
- **See also**: [`dpct::detail::generic_error_type`](#detailgeneric_error_type)  (Data Structure)



---
### pitched\_data<!-- {{#data_structure:dpct::pitched_data}} -->
- **Type**: `class`
- **Members**:
    - `_data`: A pointer to the data stored in the pitched memory.
    - `_pitch`: The pitch (or stride) of the memory, representing the number of bytes between consecutive rows.
    - `_x`: The width of the memory region in terms of elements.
    - `_y`: The height of the memory region in terms of elements.
- **Description**: The `pitched_data` class is designed to manage pitched 2D or 3D memory data, commonly used in graphics and parallel computing to handle memory with padding for alignment. It encapsulates a pointer to the data, the pitch (or stride) of the memory, and the dimensions of the memory region (width and height). The class provides constructors for initialization and methods to get and set the data pointer, pitch, and dimensions, allowing for flexible manipulation of pitched memory.
- **Member Functions**:
    - [`dpct::pitched_data::pitched_data`](#pitched_datapitched_data)
    - [`dpct::pitched_data::pitched_data`](#pitched_datapitched_data)
    - [`dpct::pitched_data::get_data_ptr`](#pitched_dataget_data_ptr)
    - [`dpct::pitched_data::set_data_ptr`](#pitched_dataset_data_ptr)
    - [`dpct::pitched_data::get_pitch`](#pitched_dataget_pitch)
    - [`dpct::pitched_data::set_pitch`](#pitched_dataset_pitch)
    - [`dpct::pitched_data::get_x`](#pitched_dataget_x)
    - [`dpct::pitched_data::set_x`](#pitched_dataset_x)
    - [`dpct::pitched_data::get_y`](#pitched_dataget_y)
    - [`dpct::pitched_data::set_y`](#pitched_dataset_y)

**Methods**

---
#### pitched\_data::pitched\_data<!-- {{#callable:dpct::pitched_data::pitched_data}} -->
The `pitched_data` constructor initializes a `pitched_data` object with a data pointer, pitch, and dimensions x and y.
- **Inputs**:
    - `data`: A pointer to the data that the `pitched_data` object will manage.
    - `pitch`: The pitch (or stride) of the data, representing the number of bytes in a row of the data.
    - `x`: The width of the data in terms of elements.
    - `y`: The height of the data in terms of elements.
- **Control Flow**:
    - The constructor initializes the private member `_data` with the provided `data` pointer.
    - The constructor initializes the private member `_pitch` with the provided `pitch` value.
    - The constructor initializes the private member `_x` with the provided `x` value.
    - The constructor initializes the private member `_y` with the provided `y` value.
- **Output**: A `pitched_data` object initialized with the specified data pointer, pitch, and dimensions.
- **See also**: [`dpct::pitched_data`](#dpctpitched_data)  (Data Structure)


---
#### pitched\_data::pitched\_data<!-- {{#callable:dpct::pitched_data::pitched_data}} -->
The `pitched_data` constructor initializes a `pitched_data` object with given data pointer, pitch, and dimensions x and y.
- **Inputs**:
    - `data`: A pointer to the data that the `pitched_data` object will manage.
    - `pitch`: The pitch (or stride) of the data, representing the number of bytes between consecutive rows.
    - `x`: The width of the data in terms of number of elements.
    - `y`: The height of the data in terms of number of elements.
- **Control Flow**:
    - The constructor initializes the private member `_data` with the provided `data` pointer.
    - The constructor initializes the private member `_pitch` with the provided `pitch` value.
    - The constructor initializes the private member `_x` with the provided `x` value.
    - The constructor initializes the private member `_y` with the provided `y` value.
- **Output**: A `pitched_data` object initialized with the specified data pointer, pitch, and dimensions.
- **See also**: [`dpct::pitched_data`](#dpctpitched_data)  (Data Structure)


---
#### pitched\_data::get\_data\_ptr<!-- {{#callable:dpct::pitched_data::get_data_ptr}} -->
The `get_data_ptr` function returns a pointer to the data stored in the `pitched_data` class.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the private member `_data` of the `pitched_data` class.
- **Output**: A `void*` pointer to the data stored in the `pitched_data` class.
- **See also**: [`dpct::pitched_data`](#dpctpitched_data)  (Data Structure)


---
#### pitched\_data::set\_data\_ptr<!-- {{#callable:dpct::pitched_data::set_data_ptr}} -->
The `set_data_ptr` function sets the internal `_data` pointer of the `pitched_data` class to a new data pointer provided as an argument.
- **Inputs**:
    - `data`: A void pointer to the new data that will be assigned to the internal `_data` member of the `pitched_data` class.
- **Control Flow**:
    - The function takes a single argument, `data`, which is a void pointer.
    - It assigns the value of `data` to the private member `_data` of the `pitched_data` class.
- **Output**: The function does not return any value.
- **See also**: [`dpct::pitched_data`](#dpctpitched_data)  (Data Structure)


---
#### pitched\_data::get\_pitch<!-- {{#callable:dpct::pitched_data::get_pitch}} -->
The `get_pitch` function retrieves the pitch value from a `pitched_data` object.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_pitch`.
- **Output**: The function returns a `size_t` value representing the pitch of the data.
- **See also**: [`dpct::pitched_data`](#dpctpitched_data)  (Data Structure)


---
#### pitched\_data::set\_pitch<!-- {{#callable:dpct::pitched_data::set_pitch}} -->
The `set_pitch` function sets the `_pitch` member variable of the `pitched_data` class to a specified value.
- **Inputs**:
    - `pitch`: A `size_t` value representing the new pitch to be set for the `pitched_data` object.
- **Control Flow**:
    - The function takes a single argument `pitch` of type `size_t`.
    - It assigns the value of `pitch` to the private member variable `_pitch` of the `pitched_data` class.
- **Output**: The function does not return any value.
- **See also**: [`dpct::pitched_data`](#dpctpitched_data)  (Data Structure)


---
#### pitched\_data::get\_x<!-- {{#callable:dpct::pitched_data::get_x}} -->
The `get_x` function retrieves the value of the private member variable `_x` from the `pitched_data` class.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_x`.
- **Output**: The function returns a `size_t` value, which is the current value of the `_x` member variable.
- **See also**: [`dpct::pitched_data`](#dpctpitched_data)  (Data Structure)


---
#### pitched\_data::set\_x<!-- {{#callable:dpct::pitched_data::set_x}} -->
The `set_x` function sets the private member variable `_x` of the `pitched_data` class to the given value `x`.
- **Inputs**:
    - `x`: A `size_t` value representing the new value to be assigned to the private member variable `_x` of the `pitched_data` class.
- **Control Flow**:
    - The function takes a single argument `x` of type `size_t`.
    - It assigns the value of `x` to the private member variable `_x` of the `pitched_data` class.
- **Output**: The function does not return any value.
- **See also**: [`dpct::pitched_data`](#dpctpitched_data)  (Data Structure)


---
#### pitched\_data::get\_y<!-- {{#callable:dpct::pitched_data::get_y}} -->
The `get_y` function retrieves the value of the private member variable `_y` from the `pitched_data` class.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_y`.
- **Output**: The function returns a `size_t` value representing the current value of the `_y` member variable.
- **See also**: [`dpct::pitched_data`](#dpctpitched_data)  (Data Structure)


---
#### pitched\_data::set\_y<!-- {{#callable:dpct::pitched_data::set_y}} -->
The `set_y` function sets the private member variable `_y` of the `pitched_data` class to the given value `y`.
- **Inputs**:
    - `y`: A `size_t` value representing the new value to be assigned to the private member variable `_y` of the `pitched_data` class.
- **Control Flow**:
    - The function takes a single argument `y` of type `size_t`.
    - It assigns the value of `y` to the private member variable `_y`.
- **Output**: The function does not return any value.
- **See also**: [`dpct::pitched_data`](#dpctpitched_data)  (Data Structure)



---
### device\_info<!-- {{#data_structure:dpct::device_info}} -->
- **Type**: `class`
- **Members**:
    - `_name`: A character array of size 256 to store the device name.
    - `_max_work_item_sizes_i`: An integer array of size 3 to store the maximum work item sizes.
    - `_host_unified_memory`: A boolean indicating if the device supports host unified memory.
    - `_major`: An integer representing the major version of the device.
    - `_minor`: An integer representing the minor version of the device.
    - `_integrated`: An integer indicating if the device is integrated.
    - `_frequency`: An integer representing the maximum clock frequency of the device.
    - `_memory_clock_rate`: An unsigned integer representing the memory clock rate in kHz, defaulting to 3200000 kHz.
    - `_memory_bus_width`: An unsigned integer representing the memory bus width in bits, defaulting to 64 bits.
    - `_global_mem_cache_size`: An unsigned integer representing the global memory cache size in bytes.
    - `_max_compute_units`: An integer representing the maximum number of compute units.
    - `_max_work_group_size`: An integer representing the maximum work group size.
    - `_max_sub_group_size`: An integer representing the maximum sub-group size.
    - `_max_work_items_per_compute_unit`: An integer representing the maximum work items per compute unit.
    - `_max_register_size_per_work_group`: An integer representing the maximum register size per work group.
    - `_global_mem_size`: A size_t representing the global memory size.
    - `_local_mem_size`: A size_t representing the local memory size.
    - `_max_mem_alloc_size`: A size_t representing the maximum memory allocation size.
    - `_max_nd_range_size`: A size_t array of size 3 representing the maximum ND range size.
    - `_max_nd_range_size_i`: An integer array of size 3 representing the maximum ND range size in integer form.
    - `_device_id`: A uint32_t representing the device ID.
    - `_uuid`: A std::array of 16 unsigned chars representing the device UUID.
- **Description**: The `device_info` class is a comprehensive data structure designed to encapsulate various attributes and capabilities of a computing device, particularly in the context of SYCL and parallel computing. It includes a wide range of properties such as device name, version information, memory specifications, and computational capabilities. The class provides both getter and setter methods for these attributes, allowing for detailed configuration and querying of device properties. This structure is essential for managing and optimizing device-specific operations in high-performance computing applications.
- **Member Functions**:
    - [`dpct::device_info::get_name`](#device_infoget_name)
    - [`dpct::device_info::get_name`](#device_infoget_name)
    - [`dpct::device_info::get_max_work_item_sizes`](#device_infoget_max_work_item_sizes)
    - [`dpct::device_info::get_max_work_item_sizes`](#device_infoget_max_work_item_sizes)
    - [`dpct::device_info::get_host_unified_memory`](#device_infoget_host_unified_memory)
    - [`dpct::device_info::get_major_version`](#device_infoget_major_version)
    - [`dpct::device_info::get_minor_version`](#device_infoget_minor_version)
    - [`dpct::device_info::get_integrated`](#device_infoget_integrated)
    - [`dpct::device_info::get_max_clock_frequency`](#device_infoget_max_clock_frequency)
    - [`dpct::device_info::get_max_compute_units`](#device_infoget_max_compute_units)
    - [`dpct::device_info::get_max_work_group_size`](#device_infoget_max_work_group_size)
    - [`dpct::device_info::get_max_sub_group_size`](#device_infoget_max_sub_group_size)
    - [`dpct::device_info::get_max_work_items_per_compute_unit`](#device_infoget_max_work_items_per_compute_unit)
    - [`dpct::device_info::get_max_register_size_per_work_group`](#device_infoget_max_register_size_per_work_group)
    - [`dpct::device_info::get_max_nd_range_size`](#device_infoget_max_nd_range_size)
    - [`dpct::device_info::get_max_nd_range_size`](#device_infoget_max_nd_range_size)
    - [`dpct::device_info::get_global_mem_size`](#device_infoget_global_mem_size)
    - [`dpct::device_info::get_local_mem_size`](#device_infoget_local_mem_size)
    - [`dpct::device_info::get_max_mem_alloc_size`](#device_infoget_max_mem_alloc_size)
    - [`dpct::device_info::get_memory_clock_rate`](#device_infoget_memory_clock_rate)
    - [`dpct::device_info::get_memory_bus_width`](#device_infoget_memory_bus_width)
    - [`dpct::device_info::get_device_id`](#device_infoget_device_id)
    - [`dpct::device_info::get_uuid`](#device_infoget_uuid)
    - [`dpct::device_info::get_global_mem_cache_size`](#device_infoget_global_mem_cache_size)
    - [`dpct::device_info::set_name`](#device_infoset_name)
    - [`dpct::device_info::set_max_work_item_sizes`](#device_infoset_max_work_item_sizes)
    - [`dpct::device_info::set_max_work_item_sizes`](#device_infoset_max_work_item_sizes)
    - [`dpct::device_info::set_host_unified_memory`](#device_infoset_host_unified_memory)
    - [`dpct::device_info::set_major_version`](#device_infoset_major_version)
    - [`dpct::device_info::set_minor_version`](#device_infoset_minor_version)
    - [`dpct::device_info::set_integrated`](#device_infoset_integrated)
    - [`dpct::device_info::set_max_clock_frequency`](#device_infoset_max_clock_frequency)
    - [`dpct::device_info::set_max_compute_units`](#device_infoset_max_compute_units)
    - [`dpct::device_info::set_global_mem_size`](#device_infoset_global_mem_size)
    - [`dpct::device_info::set_local_mem_size`](#device_infoset_local_mem_size)
    - [`dpct::device_info::set_max_mem_alloc_size`](#device_infoset_max_mem_alloc_size)
    - [`dpct::device_info::set_max_work_group_size`](#device_infoset_max_work_group_size)
    - [`dpct::device_info::set_max_sub_group_size`](#device_infoset_max_sub_group_size)
    - [`dpct::device_info::set_max_work_items_per_compute_unit`](#device_infoset_max_work_items_per_compute_unit)
    - [`dpct::device_info::set_max_nd_range_size`](#device_infoset_max_nd_range_size)
    - [`dpct::device_info::set_memory_clock_rate`](#device_infoset_memory_clock_rate)
    - [`dpct::device_info::set_memory_bus_width`](#device_infoset_memory_bus_width)
    - [`dpct::device_info::set_max_register_size_per_work_group`](#device_infoset_max_register_size_per_work_group)
    - [`dpct::device_info::set_device_id`](#device_infoset_device_id)
    - [`dpct::device_info::set_uuid`](#device_infoset_uuid)
    - [`dpct::device_info::set_global_mem_cache_size`](#device_infoset_global_mem_cache_size)

**Methods**

---
#### device\_info::get\_name<!-- {{#callable:dpct::device_info::get_name}} -->
The `get_name` function retrieves the name of the device stored in the `device_info` class.
- **Inputs**: None
- **Control Flow**:
    - The function is overloaded to provide both a const and non-const version.
    - The const version returns a const char pointer to the `_name` member variable.
    - The non-const version returns a char pointer to the `_name` member variable.
- **Output**: A pointer to the `_name` member variable, which is a character array representing the device's name.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_name<!-- {{#callable:dpct::device_info::get_name}} -->
The `get_name` function returns a pointer to the `_name` member variable of the `device_info` class.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the `_name` member variable without any additional logic or conditions.
- **Output**: A pointer to the `_name` character array, which is a private member of the `device_info` class.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_max\_work\_item\_sizes<!-- {{#callable:dpct::device_info::get_max_work_item_sizes}} -->
The `get_max_work_item_sizes` function retrieves the maximum work item sizes for a device, returning them as either a `sycl::range<3>` or an integer array, depending on the template type.
- **Inputs**:
    - `WorkItemSizesTy`: A template parameter that can be either `sycl::range<3>` or `int*`, determining the return type of the function.
- **Control Flow**:
    - The function checks if the template type `WorkItemSizesTy` is `sycl::range<3>` using `if constexpr`.
    - If `WorkItemSizesTy` is `sycl::range<3>`, it returns a `sycl::range<3>` object constructed from the `_max_work_item_sizes_i` array.
    - If `WorkItemSizesTy` is not `sycl::range<3>`, it returns the `_max_work_item_sizes_i` array directly.
- **Output**: The function returns either a `sycl::range<3>` object or an `int*` pointing to the `_max_work_item_sizes_i` array, depending on the template type `WorkItemSizesTy`.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_max\_work\_item\_sizes<!-- {{#callable:dpct::device_info::get_max_work_item_sizes}} -->
The `get_max_work_item_sizes` function retrieves the maximum work item sizes for a device, returning them as either a `sycl::range<3>` or an integer pointer array, depending on the template type specified.
- **Inputs**:
    - `WorkItemSizesTy`: A template parameter that can be either `sycl::range<3>` or `int*`, determining the return type of the function.
- **Control Flow**:
    - The function checks if `WorkItemSizesTy` is `sycl::range<3>` using `std::is_same_v`.
    - If true, it returns a `sycl::range<3>` constructed from the `_max_work_item_sizes_i` array.
    - If false, it returns the `_max_work_item_sizes_i` array directly.
- **Output**: Returns the maximum work item sizes as a `sycl::range<3>` if `WorkItemSizesTy` is `sycl::range<3>`, otherwise returns an integer pointer to the `_max_work_item_sizes_i` array.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_host\_unified\_memory<!-- {{#callable:dpct::device_info::get_host_unified_memory}} -->
The `get_host_unified_memory` function returns the value of the `_host_unified_memory` member variable, indicating whether the device supports host unified memory.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter method that directly returns the value of the `_host_unified_memory` member variable.
- **Output**: A boolean value indicating if the device supports host unified memory.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_major\_version<!-- {{#callable:dpct::device_info::get_major_version}} -->
The `get_major_version` function returns the major version number of a device.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_major`.
- **Output**: An integer representing the major version number of the device.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_minor\_version<!-- {{#callable:dpct::device_info::get_minor_version}} -->
The `get_minor_version` function retrieves the minor version number of a device from a `sycl::device` object.
- **Inputs**:
    - `dev`: A `sycl::device` object from which the minor version number is to be retrieved.
- **Control Flow**:
    - The function calls `detail::get_version` with the `sycl::device` object `dev` and two integer references `major` and `minor`.
    - The `detail::get_version` function extracts the version string from the device and parses it to obtain the major and minor version numbers.
    - The function returns the `minor` version number obtained from the `detail::get_version` function.
- **Output**: An integer representing the minor version number of the device.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_integrated<!-- {{#callable:dpct::device_info::get_integrated}} -->
The `get_integrated` function returns the value of the `_integrated` member variable from the `device_info` class.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter method that directly returns the value of the `_integrated` member variable.
- **Output**: An integer representing whether the device is integrated or not, as stored in the `_integrated` member variable.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_max\_clock\_frequency<!-- {{#callable:dpct::device_info::get_max_clock_frequency}} -->
The `get_max_clock_frequency` function returns the maximum clock frequency of a device in the `device_info` class.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_frequency`.
- **Output**: An integer representing the maximum clock frequency of the device.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_max\_compute\_units<!-- {{#callable:dpct::device_info::get_max_compute_units}} -->
The `get_max_compute_units` function returns the maximum number of compute units available on a device.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_max_compute_units`.
- **Output**: An integer representing the maximum number of compute units available on the device.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_max\_work\_group\_size<!-- {{#callable:dpct::device_info::get_max_work_group_size}} -->
The `get_max_work_group_size` function returns the maximum work group size for a device.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_max_work_group_size`.
- **Output**: An integer representing the maximum work group size of the device.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_max\_sub\_group\_size<!-- {{#callable:dpct::device_info::get_max_sub_group_size}} -->
The `get_max_sub_group_size` function returns the maximum size of a sub-group for a device.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_max_sub_group_size`.
- **Output**: An integer representing the maximum sub-group size of the device.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_max\_work\_items\_per\_compute\_unit<!-- {{#callable:dpct::device_info::get_max_work_items_per_compute_unit}} -->
The `get_max_work_items_per_compute_unit` function returns the maximum number of work items that can be executed per compute unit on a device.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_max_work_items_per_compute_unit`.
- **Output**: An integer representing the maximum number of work items per compute unit.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_max\_register\_size\_per\_work\_group<!-- {{#callable:dpct::device_info::get_max_register_size_per_work_group}} -->
The `get_max_register_size_per_work_group` function returns the maximum register size available per work group for a device.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_max_register_size_per_work_group`.
- **Output**: An integer representing the maximum register size per work group.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_max\_nd\_range\_size<!-- {{#callable:dpct::device_info::get_max_nd_range_size}} -->
The `get_max_nd_range_size` function retrieves the maximum ND range size for a device, returning either a `size_t*` or `int*` based on the template type.
- **Inputs**:
    - `NDRangeSizeTy`: A template parameter that can be either `size_t*` or `int*`, determining the type of the returned maximum ND range size.
- **Control Flow**:
    - The function uses a template parameter `NDRangeSizeTy` with a default type of `size_t*` and a constraint that it must be either `size_t*` or `int*`.
    - It checks the type of `NDRangeSizeTy` using `if constexpr` to determine which member variable to return.
    - If `NDRangeSizeTy` is `size_t*`, it returns `_max_nd_range_size`.
    - Otherwise, it returns `_max_nd_range_size_i`.
- **Output**: Returns a pointer to the maximum ND range size, either `_max_nd_range_size` (of type `size_t*`) or `_max_nd_range_size_i` (of type `int*`), depending on the template type.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_max\_nd\_range\_size<!-- {{#callable:dpct::device_info::get_max_nd_range_size}} -->
The `get_max_nd_range_size` function returns the maximum ND range size for a device, either as a `size_t*` or `int*`, depending on the template type `NDRangeSizeTy`.
- **Inputs**:
    - `NDRangeSizeTy`: A template parameter that can be either `size_t*` or `int*`, determining the type of the returned maximum ND range size.
- **Control Flow**:
    - The function uses a template parameter `NDRangeSizeTy` with a default type of `size_t*` and a constraint that it must be either `size_t*` or `int*`.
    - It checks the type of `NDRangeSizeTy` using `std::is_same_v` to determine which member variable to return.
    - If `NDRangeSizeTy` is `size_t*`, it returns `_max_nd_range_size`.
    - Otherwise, it returns `_max_nd_range_size_i`.
- **Output**: Returns either `_max_nd_range_size` or `_max_nd_range_size_i` based on the type of `NDRangeSizeTy`.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_global\_mem\_size<!-- {{#callable:dpct::device_info::get_global_mem_size}} -->
The `get_global_mem_size` function returns the size of the global memory for a device.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_global_mem_size`.
- **Output**: The function outputs a `size_t` value representing the size of the global memory.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_local\_mem\_size<!-- {{#callable:dpct::device_info::get_local_mem_size}} -->
The `get_local_mem_size` function returns the size of the local memory available on the device.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_local_mem_size`.
- **Output**: The function returns a `size_t` value representing the size of the local memory.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_max\_mem\_alloc\_size<!-- {{#callable:dpct::device_info::get_max_mem_alloc_size}} -->
The `get_max_mem_alloc_size` function returns the maximum memory allocation size for a device.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_max_mem_alloc_size`.
- **Output**: The function outputs a `size_t` value representing the maximum memory allocation size for the device.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_memory\_clock\_rate<!-- {{#callable:dpct::device_info::get_memory_clock_rate}} -->
The `get_memory_clock_rate` function returns the maximum clock rate of the device's global memory in kHz.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_memory_clock_rate`.
- **Output**: An unsigned integer representing the memory clock rate in kHz.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_memory\_bus\_width<!-- {{#callable:dpct::device_info::get_memory_bus_width}} -->
The `get_memory_bus_width` function returns the maximum bus width between the device and memory in bits.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_memory_bus_width`.
- **Output**: An unsigned integer representing the memory bus width in bits.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_device\_id<!-- {{#callable:dpct::device_info::get_device_id}} -->
The `get_device_id` function returns the device ID of the device_info object.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter that directly returns the private member variable `_device_id` of the `device_info` class.
- **Output**: The function returns a `uint32_t` representing the device ID.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_uuid<!-- {{#callable:dpct::device_info::get_uuid}} -->
The `get_uuid` function returns the UUID of the device as a 16-element array of unsigned characters.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the private member variable `_uuid` which is an array of 16 unsigned characters.
- **Output**: A `std::array<unsigned char, 16>` representing the UUID of the device.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::get\_global\_mem\_cache\_size<!-- {{#callable:dpct::device_info::get_global_mem_cache_size}} -->
The `get_global_mem_cache_size` function returns the size of the global memory cache in bytes for a device.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_global_mem_cache_size`.
- **Output**: An unsigned integer representing the global memory cache size in bytes.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_name<!-- {{#callable:dpct::device_info::set_name}} -->
The `set_name` function sets the `_name` member of the `device_info` class to a given string, ensuring it does not exceed 255 characters.
- **Inputs**:
    - `name`: A constant character pointer representing the name to be set for the device.
- **Control Flow**:
    - Calculate the length of the input string `name` using `strlen`.
    - Check if the length is less than 256.
    - If true, copy the entire string including the null terminator into `_name` using `std::memcpy`.
    - If false, copy only the first 255 characters into `_name` and manually set the 256th character to the null terminator.
- **Output**: The function does not return any value; it modifies the `_name` member of the `device_info` class.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_max\_work\_item\_sizes<!-- {{#callable:dpct::device_info::set_max_work_item_sizes}} -->
The `set_max_work_item_sizes` function sets the maximum work item sizes for a device by copying values from a given `sycl::range<3>` to a private member array.
- **Inputs**:
    - `max_work_item_sizes`: A `sycl::range<3>` object representing the maximum work item sizes to be set.
- **Control Flow**:
    - Iterate over the three dimensions of the `max_work_item_sizes` range.
    - Assign each dimension's value to the corresponding index in the `_max_work_item_sizes_i` array.
- **Output**: This function does not return any value; it modifies the private member `_max_work_item_sizes_i` of the `device_info` class.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_max\_work\_item\_sizes<!-- {{#callable:dpct::device_info::set_max_work_item_sizes}} -->
The `set_max_work_item_sizes` function sets the maximum work item sizes for a device by copying values from a given `sycl::id<3>` object to a private member array.
- **Inputs**:
    - `max_work_item_sizes`: A `sycl::id<3>` object representing the maximum work item sizes to be set.
- **Control Flow**:
    - The function iterates over the three dimensions of the `max_work_item_sizes` object.
    - For each dimension, it assigns the corresponding value from `max_work_item_sizes` to the private member array `_max_work_item_sizes_i`.
- **Output**: This function does not return any value; it modifies the private member array `_max_work_item_sizes_i` of the `device_info` class.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_host\_unified\_memory<!-- {{#callable:dpct::device_info::set_host_unified_memory}} -->
The `set_host_unified_memory` function sets the `_host_unified_memory` member variable of the `device_info` class to the provided boolean value.
- **Inputs**:
    - `host_unified_memory`: A boolean value indicating whether host unified memory is enabled or not.
- **Control Flow**:
    - The function takes a boolean parameter `host_unified_memory`.
    - It assigns the value of `host_unified_memory` to the private member variable `_host_unified_memory`.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_major\_version<!-- {{#callable:dpct::device_info::set_major_version}} -->
The `set_major_version` function sets the major version number of a device in the `device_info` class.
- **Inputs**:
    - `major`: An integer representing the major version number to be set for the device.
- **Control Flow**:
    - The function takes an integer input `major`.
    - It assigns the value of `major` to the private member variable `_major` of the `device_info` class.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_minor\_version<!-- {{#callable:dpct::device_info::set_minor_version}} -->
The `set_minor_version` function sets the minor version of a device in the `device_info` class.
- **Inputs**:
    - `minor`: An integer representing the minor version to be set for the device.
- **Control Flow**:
    - The function takes an integer input `minor`.
    - It assigns the value of `minor` to the private member variable `_minor` of the `device_info` class.
- **Output**: The function does not return any value; it modifies the internal state of the `device_info` object by setting its `_minor` member variable.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_integrated<!-- {{#callable:dpct::device_info::set_integrated}} -->
The `set_integrated` function sets the `_integrated` member variable of the `device_info` class to the provided integer value.
- **Inputs**:
    - `integrated`: An integer value representing whether the device is integrated or not.
- **Control Flow**:
    - The function directly assigns the input value to the `_integrated` member variable of the `device_info` class.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_max\_clock\_frequency<!-- {{#callable:dpct::device_info::set_max_clock_frequency}} -->
The `set_max_clock_frequency` function sets the maximum clock frequency for a device by assigning the provided frequency value to the `_frequency` member variable.
- **Inputs**:
    - `frequency`: An integer representing the maximum clock frequency to be set for the device.
- **Control Flow**:
    - The function takes an integer input `frequency`.
    - It assigns the input `frequency` to the private member variable `_frequency`.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_max\_compute\_units<!-- {{#callable:dpct::device_info::set_max_compute_units}} -->
The `set_max_compute_units` function sets the maximum number of compute units for a device by assigning the provided value to the `_max_compute_units` member variable.
- **Inputs**:
    - `max_compute_units`: An integer representing the maximum number of compute units to be set for the device.
- **Control Flow**:
    - The function takes an integer input `max_compute_units`.
    - It assigns the input value to the private member variable `_max_compute_units` of the `device_info` class.
- **Output**: The function does not return any value; it modifies the state of the `device_info` object by setting the `_max_compute_units` member variable.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_global\_mem\_size<!-- {{#callable:dpct::device_info::set_global_mem_size}} -->
The `set_global_mem_size` function sets the global memory size for a device by assigning the provided value to the `_global_mem_size` member variable.
- **Inputs**:
    - `global_mem_size`: A `size_t` value representing the size of the global memory to be set.
- **Control Flow**:
    - The function takes a single input parameter `global_mem_size`.
    - It assigns the value of `global_mem_size` to the private member variable `_global_mem_size`.
- **Output**: The function does not return any value; it modifies the internal state of the `device_info` object by setting the `_global_mem_size`.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_local\_mem\_size<!-- {{#callable:dpct::device_info::set_local_mem_size}} -->
The `set_local_mem_size` function sets the local memory size for a device by assigning the provided value to the `_local_mem_size` member variable.
- **Inputs**:
    - `local_mem_size`: A `size_t` value representing the size of local memory to be set for the device.
- **Control Flow**:
    - The function takes a single input parameter `local_mem_size`.
    - It assigns the value of `local_mem_size` to the private member variable `_local_mem_size`.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_max\_mem\_alloc\_size<!-- {{#callable:dpct::device_info::set_max_mem_alloc_size}} -->
The `set_max_mem_alloc_size` function sets the maximum memory allocation size for a device by updating the `_max_mem_alloc_size` member variable.
- **Inputs**:
    - `max_mem_alloc_size`: A `size_t` value representing the maximum memory allocation size to be set.
- **Control Flow**:
    - The function takes a single input parameter `max_mem_alloc_size`.
    - It assigns the value of `max_mem_alloc_size` to the private member variable `_max_mem_alloc_size`.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_max\_work\_group\_size<!-- {{#callable:dpct::device_info::set_max_work_group_size}} -->
The `set_max_work_group_size` function sets the maximum work group size for a device by assigning the provided value to the `_max_work_group_size` member variable.
- **Inputs**:
    - `max_work_group_size`: An integer representing the maximum work group size to be set for the device.
- **Control Flow**:
    - The function takes an integer input `max_work_group_size`.
    - It assigns the input value to the private member variable `_max_work_group_size`.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_max\_sub\_group\_size<!-- {{#callable:dpct::device_info::set_max_sub_group_size}} -->
The `set_max_sub_group_size` function sets the maximum sub-group size for a device by assigning the provided value to the `_max_sub_group_size` member variable.
- **Inputs**:
    - `max_sub_group_size`: An integer representing the maximum sub-group size to be set for the device.
- **Control Flow**:
    - The function takes an integer input `max_sub_group_size`.
    - It assigns the input value to the private member variable `_max_sub_group_size`.
- **Output**: The function does not return any value; it modifies the state of the `device_info` object by setting the `_max_sub_group_size`.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_max\_work\_items\_per\_compute\_unit<!-- {{#callable:dpct::device_info::set_max_work_items_per_compute_unit}} -->
The function `set_max_work_items_per_compute_unit` sets the maximum number of work items per compute unit for a device.
- **Inputs**:
    - `max_work_items_per_compute_unit`: An integer representing the maximum number of work items that can be executed per compute unit.
- **Control Flow**:
    - The function takes an integer input `max_work_items_per_compute_unit`.
    - It assigns this input value to the private member variable `_max_work_items_per_compute_unit` of the `device_info` class.
- **Output**: The function does not return any value; it modifies the internal state of the `device_info` object by setting the `_max_work_items_per_compute_unit` member variable.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_max\_nd\_range\_size<!-- {{#callable:dpct::device_info::set_max_nd_range_size}} -->
The `set_max_nd_range_size` function sets the maximum ND range size for a device by copying values from the input array to two internal arrays.
- **Inputs**:
    - `max_nd_range_size`: An array of integers representing the maximum ND range size to be set, expected to have at least 3 elements.
- **Control Flow**:
    - Iterates over the first three elements of the input array `max_nd_range_size`.
    - Copies each element to the corresponding index in the internal arrays `_max_nd_range_size` and `_max_nd_range_size_i`.
- **Output**: This function does not return any value; it modifies internal state variables of the `device_info` class.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_memory\_clock\_rate<!-- {{#callable:dpct::device_info::set_memory_clock_rate}} -->
The `set_memory_clock_rate` function sets the memory clock rate of a device to a specified value.
- **Inputs**:
    - `memory_clock_rate`: An unsigned integer representing the desired memory clock rate in kHz to be set for the device.
- **Control Flow**:
    - The function takes an unsigned integer as an argument.
    - It assigns this value to the private member variable `_memory_clock_rate` of the `device_info` class.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_memory\_bus\_width<!-- {{#callable:dpct::device_info::set_memory_bus_width}} -->
The `set_memory_bus_width` function sets the memory bus width of a device to a specified value.
- **Inputs**:
    - `memory_bus_width`: An unsigned integer representing the new memory bus width to be set for the device.
- **Control Flow**:
    - The function assigns the input `memory_bus_width` to the private member variable `_memory_bus_width` of the `device_info` class.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_max\_register\_size\_per\_work\_group<!-- {{#callable:dpct::device_info::set_max_register_size_per_work_group}} -->
The function `set_max_register_size_per_work_group` sets the maximum register size per work group for a device.
- **Inputs**:
    - `max_register_size_per_work_group`: An integer representing the maximum register size per work group to be set.
- **Control Flow**:
    - The function assigns the input value `max_register_size_per_work_group` to the private member variable `_max_register_size_per_work_group`.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_device\_id<!-- {{#callable:dpct::device_info::set_device_id}} -->
The `set_device_id` function assigns a given device ID to the `_device_id` member variable of the `device_info` class.
- **Inputs**:
    - `device_id`: A 32-bit unsigned integer representing the device ID to be set.
- **Control Flow**:
    - The function takes a single argument, `device_id`, of type `uint32_t`.
    - It assigns the value of `device_id` to the private member variable `_device_id` of the `device_info` class.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_uuid<!-- {{#callable:dpct::device_info::set_uuid}} -->
The `set_uuid` function assigns a given UUID to the `_uuid` member variable of the `device_info` class.
- **Inputs**:
    - `uuid`: A `std::array` of 16 `unsigned char` elements representing the UUID to be set.
- **Control Flow**:
    - The function takes a `std::array` of 16 `unsigned char` elements as input, representing the UUID.
    - It uses `std::move` to efficiently transfer the input UUID to the `_uuid` member variable of the `device_info` class.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)


---
#### device\_info::set\_global\_mem\_cache\_size<!-- {{#callable:dpct::device_info::set_global_mem_cache_size}} -->
The `set_global_mem_cache_size` function sets the global memory cache size for a device by assigning the provided value to the `_global_mem_cache_size` member variable.
- **Inputs**:
    - `global_mem_cache_size`: An unsigned integer representing the size of the global memory cache to be set.
- **Control Flow**:
    - The function takes an unsigned integer as an argument.
    - It assigns this value to the private member variable `_global_mem_cache_size`.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_info`](#dpctdevice_info)  (Data Structure)



---
### device\_ext<!-- {{#data_structure:dpct::device_ext}} -->
- **Type**: `class`
- **Members**:
    - `_q_in_order`: An in-order SYCL queue for managing tasks that must be executed in the order they are submitted.
    - `_q_out_of_order`: An out-of-order SYCL queue for managing tasks that can be executed in any order.
    - `_saved_queue`: A SYCL queue that is saved for later use or retrieval.
    - `_queues`: A vector of SYCL queues used to manage multiple queues for task execution.
    - `m_mutex`: A mutable mutex used to synchronize access to shared resources within the class.
- **Description**: The `device_ext` class is an extension of the `sycl::device` class, providing additional functionality for managing SYCL queues and device information. It includes methods for creating and managing in-order and out-of-order queues, retrieving device information such as version and memory details, and handling synchronization through mutexes. The class also supports queue creation with optional exception handling and maintains a list of queues for task execution. It is designed to facilitate advanced device management and task scheduling in SYCL-based applications.
- **Member Functions**:
    - [`dpct::device_ext::device_ext`](#device_extdevice_ext)
    - [`dpct::device_ext::~device_ext`](#device_extdevice_ext)
    - [`dpct::device_ext::device_ext`](#device_extdevice_ext)
    - [`dpct::device_ext::is_native_atomic_supported`](#device_extis_native_atomic_supported)
    - [`dpct::device_ext::get_major_version`](#device_extget_major_version)
    - [`dpct::device_ext::get_minor_version`](#device_extget_minor_version)
    - [`dpct::device_ext::get_max_compute_units`](#device_extget_max_compute_units)
    - [`dpct::device_ext::get_max_clock_frequency`](#device_extget_max_clock_frequency)
    - [`dpct::device_ext::get_integrated`](#device_extget_integrated)
    - [`dpct::device_ext::get_max_sub_group_size`](#device_extget_max_sub_group_size)
    - [`dpct::device_ext::get_max_register_size_per_work_group`](#device_extget_max_register_size_per_work_group)
    - [`dpct::device_ext::get_max_work_group_size`](#device_extget_max_work_group_size)
    - [`dpct::device_ext::get_mem_base_addr_align`](#device_extget_mem_base_addr_align)
    - [`dpct::device_ext::get_global_mem_size`](#device_extget_global_mem_size)
    - [`dpct::device_ext::get_max_mem_alloc_size`](#device_extget_max_mem_alloc_size)
    - [`dpct::device_ext::get_memory_info`](#device_extget_memory_info)
    - [`dpct::device_ext::get_device_info`](#device_extget_device_info)
    - [`dpct::device_ext::get_device_info`](#device_extget_device_info)
    - [`dpct::device_ext::reset`](#device_extreset)
    - [`dpct::device_ext::in_order_queue`](#device_extin_order_queue)
    - [`dpct::device_ext::out_of_order_queue`](#device_extout_of_order_queue)
    - [`dpct::device_ext::default_queue`](#device_extdefault_queue)
    - [`dpct::device_ext::queues_wait_and_throw`](#device_extqueues_wait_and_throw)
    - [`dpct::device_ext::create_queue`](#device_extcreate_queue)
    - [`dpct::device_ext::create_queue`](#device_extcreate_queue)
    - [`dpct::device_ext::create_in_order_queue`](#device_extcreate_in_order_queue)
    - [`dpct::device_ext::create_in_order_queue`](#device_extcreate_in_order_queue)
    - [`dpct::device_ext::create_out_of_order_queue`](#device_extcreate_out_of_order_queue)
    - [`dpct::device_ext::destroy_queue`](#device_extdestroy_queue)
    - [`dpct::device_ext::set_saved_queue`](#device_extset_saved_queue)
    - [`dpct::device_ext::get_saved_queue`](#device_extget_saved_queue)
    - [`dpct::device_ext::clear_queues`](#device_extclear_queues)
    - [`dpct::device_ext::init_queues`](#device_extinit_queues)
    - [`dpct::device_ext::create_queue_impl`](#device_extcreate_queue_impl)
    - [`dpct::device_ext::create_queue_impl`](#device_extcreate_queue_impl)
    - [`dpct::device_ext::get_version`](#device_extget_version)
- **Inherits From**:
    - `sycl::device`

**Methods**

---
#### device\_ext::device\_ext<!-- {{#callable:dpct::device_ext::device_ext}} -->
The `device_ext` constructor initializes a `device_ext` object by calling the base class `sycl::device` constructor.
- **Inputs**: None
- **Control Flow**:
    - The constructor `device_ext()` is called, which in turn calls the base class `sycl::device` constructor.
    - No additional operations are performed in this constructor.
- **Output**: A `device_ext` object is created, initialized as a `sycl::device`.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::\~device\_ext<!-- {{#callable:dpct::device_ext::~device_ext}} -->
The destructor `~device_ext` ensures thread-safe cleanup of device queues by locking a mutex and clearing the queues.
- **Inputs**: None
- **Control Flow**:
    - The destructor is called when an instance of `device_ext` is destroyed.
    - A `std::lock_guard` is used to lock the `m_mutex` to ensure thread safety during the cleanup process.
    - The [`clear_queues`](#device_extclear_queues) method is called to clear the device queues.
- **Output**: The function does not return any value as it is a destructor.
- **Functions called**:
    - [`dpct::device_ext::clear_queues`](#device_extclear_queues)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::device\_ext<!-- {{#callable:dpct::device_ext::device_ext}} -->
The `device_ext` constructor initializes a `device_ext` object by copying a base `sycl::device` and setting up its internal queues with thread safety.
- **Inputs**:
    - `base`: A constant reference to a `sycl::device` object that serves as the base device for the `device_ext` object.
- **Control Flow**:
    - The constructor is called with a `sycl::device` reference as an argument.
    - A lock is acquired on the mutex `m_mutex` to ensure thread safety during initialization.
    - The base `sycl::device` is passed to the parent class constructor to initialize the `device_ext` object.
    - The [`init_queues`](#device_extinit_queues) method is called to initialize the internal queues of the `device_ext` object.
- **Output**: The function does not return any value as it is a constructor.
- **Functions called**:
    - [`dpct::device_ext::init_queues`](#device_extinit_queues)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::is\_native\_atomic\_supported<!-- {{#callable:dpct::device_ext::is_native_atomic_supported}} -->
The `is_native_atomic_supported` function checks if native atomic operations are supported on the device, but currently always returns 0, indicating no support.
- **Inputs**: None
- **Control Flow**:
    - The function is defined to return an integer value.
    - It directly returns the integer 0 without any conditions or checks.
- **Output**: The function returns an integer value of 0, indicating that native atomic operations are not supported.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_major\_version<!-- {{#callable:dpct::device_ext::get_major_version}} -->
The `get_major_version` function retrieves the major version number of the device associated with the `device_ext` object.
- **Inputs**:
    - `this`: A constant reference to the current `device_ext` object, which is implicitly passed to the function.
- **Control Flow**:
    - The function calls `dpct::get_major_version` with the current `device_ext` object as an argument.
    - The `dpct::get_major_version` function internally calls `detail::get_version` to extract the major version from the device's version string.
    - The major version is returned by `dpct::get_major_version` and subsequently by `get_major_version`.
- **Output**: An integer representing the major version number of the device.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_minor\_version<!-- {{#callable:dpct::device_ext::get_minor_version}} -->
The `get_minor_version` function retrieves the minor version number of the device by calling the `dpct::get_minor_version` function with the current device object as an argument.
- **Inputs**:
    - `this`: The current instance of the `device_ext` class, which is implicitly passed as the argument to the `dpct::get_minor_version` function.
- **Control Flow**:
    - The function is a simple one-liner that directly returns the result of the `dpct::get_minor_version` function call.
    - The `dpct::get_minor_version` function is called with the current instance of the `device_ext` class as its argument.
- **Output**: The function returns an integer representing the minor version number of the device.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_max\_compute\_units<!-- {{#callable:dpct::device_ext::get_max_compute_units}} -->
The `get_max_compute_units` function retrieves the maximum number of compute units available on a SYCL device.
- **Inputs**: None
- **Control Flow**:
    - The function calls `get_device_info()` to obtain a `device_info` object.
    - It then calls `get_max_compute_units()` on the `device_info` object to retrieve the maximum compute units.
- **Output**: Returns an integer representing the maximum number of compute units available on the device.
- **Functions called**:
    - [`dpct::device_ext::get_device_info`](#device_extget_device_info)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_max\_clock\_frequency<!-- {{#callable:dpct::device_ext::get_max_clock_frequency}} -->
The `get_max_clock_frequency` function retrieves the maximum clock frequency of the device in KHz.
- **Inputs**: None
- **Control Flow**:
    - The function calls `get_device_info()` to obtain a `device_info` object.
    - It then calls `get_max_clock_frequency()` on the `device_info` object to retrieve the maximum clock frequency.
- **Output**: Returns an integer representing the maximum clock frequency of the device in KHz.
- **Functions called**:
    - [`dpct::device_ext::get_device_info`](#device_extget_device_info)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_integrated<!-- {{#callable:dpct::device_ext::get_integrated}} -->
The `get_integrated` function retrieves the integrated status of a device by calling the `get_integrated` method on the `device_info` object associated with the device.
- **Inputs**: None
- **Control Flow**:
    - The function calls `get_device_info()` to obtain a `device_info` object.
    - It then calls `get_integrated()` on the `device_info` object to retrieve the integrated status.
- **Output**: An integer representing the integrated status of the device.
- **Functions called**:
    - [`dpct::device_ext::get_device_info`](#device_extget_device_info)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_max\_sub\_group\_size<!-- {{#callable:dpct::device_ext::get_max_sub_group_size}} -->
The `get_max_sub_group_size` function retrieves the maximum sub-group size supported by the device.
- **Inputs**: None
- **Control Flow**:
    - The function calls `get_device_info()` to obtain a `device_info` object.
    - It then calls `get_max_sub_group_size()` on the `device_info` object to retrieve the maximum sub-group size.
- **Output**: Returns an integer representing the maximum sub-group size supported by the device.
- **Functions called**:
    - [`dpct::device_ext::get_device_info`](#device_extget_device_info)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_max\_register\_size\_per\_work\_group<!-- {{#callable:dpct::device_ext::get_max_register_size_per_work_group}} -->
The `get_max_register_size_per_work_group` function retrieves the maximum register size available per work group for the device.
- **Inputs**: None
- **Control Flow**:
    - The function calls `get_device_info()` to obtain a `device_info` object.
    - It then calls `get_max_register_size_per_work_group()` on the `device_info` object to retrieve the maximum register size per work group.
    - The retrieved value is returned as the output of the function.
- **Output**: An integer representing the maximum register size per work group for the device.
- **Functions called**:
    - [`dpct::device_ext::get_device_info`](#device_extget_device_info)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_max\_work\_group\_size<!-- {{#callable:dpct::device_ext::get_max_work_group_size}} -->
The `get_max_work_group_size` function retrieves the maximum work group size for the device associated with the `device_ext` class.
- **Inputs**: None
- **Control Flow**:
    - The function calls `get_device_info()` to obtain a `device_info` object.
    - It then calls `get_max_work_group_size()` on the `device_info` object to retrieve the maximum work group size.
- **Output**: Returns an integer representing the maximum work group size for the device.
- **Functions called**:
    - [`dpct::device_ext::get_device_info`](#device_extget_device_info)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_mem\_base\_addr\_align<!-- {{#callable:dpct::device_ext::get_mem_base_addr_align}} -->
The `get_mem_base_addr_align` function retrieves the memory base address alignment information for a SYCL device.
- **Inputs**: None
- **Control Flow**:
    - The function calls `get_info` with the template parameter `sycl::info::device::mem_base_addr_align` to obtain the memory base address alignment information.
- **Output**: Returns an integer representing the memory base address alignment of the device.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_global\_mem\_size<!-- {{#callable:dpct::device_ext::get_global_mem_size}} -->
The `get_global_mem_size` function retrieves the global memory size of the device associated with the `device_ext` object.
- **Inputs**: None
- **Control Flow**:
    - The function calls `get_device_info()` on the `device_ext` object to obtain a `device_info` object.
    - It then calls `get_global_mem_size()` on the `device_info` object to retrieve the global memory size.
- **Output**: The function returns a `size_t` value representing the global memory size of the device.
- **Functions called**:
    - [`dpct::device_ext::get_device_info`](#device_extget_device_info)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_max\_mem\_alloc\_size<!-- {{#callable:dpct::device_ext::get_max_mem_alloc_size}} -->
The `get_max_mem_alloc_size` function retrieves the maximum memory allocation size for the device.
- **Inputs**: None
- **Control Flow**:
    - The function calls `get_device_info()` to obtain a `device_info` object.
    - It then calls `get_max_mem_alloc_size()` on the `device_info` object to retrieve the maximum memory allocation size.
- **Output**: Returns a `size_t` value representing the maximum memory allocation size for the device.
- **Functions called**:
    - [`dpct::device_ext::get_device_info`](#device_extget_device_info)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_memory\_info<!-- {{#callable:dpct::device_ext::get_memory_info}} -->
The `get_memory_info` function retrieves the total and free memory available on a SYCL device, with a fallback mechanism if querying free memory is not supported.
- **Inputs**:
    - `free_memory`: A reference to a size_t variable where the function will store the number of bytes of free memory on the SYCL device.
    - `total_memory`: A reference to a size_t variable where the function will store the total number of bytes of memory on the SYCL device.
- **Control Flow**:
    - Retrieve the total memory size from the device and store it in `total_memory`.
    - Define a warning message indicating that querying free memory is not supported unless a specific environment variable is set.
    - Check if the SYCL compiler version is defined and meets a minimum version requirement.
    - If the `ext_intel_free_memory` aspect is not supported, print the warning message and set `free_memory` to `total_memory`.
    - If the `ext_intel_free_memory` aspect is supported, retrieve the free memory information and store it in `free_memory`.
    - If the SYCL compiler version is not defined or does not meet the requirement, print the warning message and set `free_memory` to `total_memory`.
- **Output**: The function outputs the total and free memory sizes through the reference parameters `total_memory` and `free_memory`.
- **Functions called**:
    - [`dpct::device_ext::get_device_info`](#device_extget_device_info)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_device\_info<!-- {{#callable:dpct::device_ext::get_device_info}} -->
The `get_device_info` function populates a `device_info` object with information about the current device using the `dpct::get_device_info` function.
- **Inputs**:
    - `out`: A reference to a `device_info` object that will be populated with the device information.
- **Control Flow**:
    - The function calls `dpct::get_device_info` with the `out` parameter and the current device (`*this`) as arguments.
- **Output**: The function does not return a value; it modifies the `out` parameter to contain the device information.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_device\_info<!-- {{#callable:dpct::device_ext::get_device_info}} -->
The `get_device_info` function retrieves and returns the device information for the current device.
- **Inputs**: None
- **Control Flow**:
    - Create a `device_info` object named `prop`.
    - Call `dpct::get_device_info` with `prop` and the current device (`*this`) to populate `prop` with device information.
    - Return the populated `device_info` object `prop`.
- **Output**: A `device_info` object containing the information of the current device.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::reset<!-- {{#callable:dpct::device_ext::reset}} -->
The `reset` function in the `device_ext` class clears and reinitializes the SYCL queues while ensuring thread safety using a mutex.
- **Inputs**: None
- **Control Flow**:
    - Acquire a lock on the mutex `m_mutex` to ensure thread safety.
    - Call `clear_queues()` to clear the existing queues.
    - Call `init_queues()` to initialize the queues again.
- **Output**: The function does not return any value.
- **Functions called**:
    - [`dpct::device_ext::clear_queues`](#device_extclear_queues)
    - [`dpct::device_ext::init_queues`](#device_extinit_queues)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::in\_order\_queue<!-- {{#callable:dpct::device_ext::in_order_queue}} -->
The `in_order_queue` function returns a reference to the in-order SYCL queue associated with the `device_ext` class.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the `_q_in_order` member variable, which is a `sycl::queue` object.
- **Output**: A reference to a `sycl::queue` object representing the in-order queue.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::out\_of\_order\_queue<!-- {{#callable:dpct::device_ext::out_of_order_queue}} -->
The `out_of_order_queue` function returns a reference to the out-of-order SYCL queue associated with the `device_ext` class.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the `_q_out_of_order` member variable, which is a `sycl::queue` object.
- **Output**: A reference to a `sycl::queue` object representing the out-of-order queue.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::default\_queue<!-- {{#callable:dpct::device_ext::default_queue}} -->
The `default_queue` method returns a reference to the in-order SYCL queue associated with the `device_ext` class.
- **Inputs**: None
- **Control Flow**:
    - The method directly calls and returns the result of the `in_order_queue()` method.
- **Output**: A reference to a `sycl::queue` object representing the in-order queue.
- **Functions called**:
    - [`dpct::device_ext::in_order_queue`](#device_extin_order_queue)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::queues\_wait\_and\_throw<!-- {{#callable:dpct::device_ext::queues_wait_and_throw}} -->
The `queues_wait_and_throw` function iterates over a collection of SYCL queues, calling `wait_and_throw` on each to ensure all queued tasks are completed and any exceptions are thrown.
- **Inputs**: None
- **Control Flow**:
    - Acquire a unique lock on the mutex `m_mutex` to ensure thread safety.
    - Immediately unlock the mutex to allow other operations while waiting on queues.
    - Iterate over each queue in the `_queues` vector.
    - For each queue, call the `wait_and_throw` method to wait for all tasks to complete and throw any exceptions that occurred.
    - Re-lock the mutex to safely manage the reference count of the queues.
- **Output**: The function does not return any value; it ensures all queues have completed their tasks and handles any exceptions.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::create\_queue<!-- {{#callable:dpct::device_ext::create_queue}} -->
The `create_queue` function creates and returns a SYCL in-order queue, optionally with an exception handler.
- **Inputs**:
    - `enable_exception_handler`: A boolean flag indicating whether to enable an exception handler for the queue; defaults to false.
- **Control Flow**:
    - The function calls [`create_in_order_queue`](#device_extcreate_in_order_queue) with the `enable_exception_handler` argument.
    - The [`create_in_order_queue`](#device_extcreate_in_order_queue) function is responsible for creating the queue with the specified properties.
- **Output**: Returns a `sycl::queue` object configured as an in-order queue.
- **Functions called**:
    - [`dpct::device_ext::create_in_order_queue`](#device_extcreate_in_order_queue)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::create\_queue<!-- {{#callable:dpct::device_ext::create_queue}} -->
The `create_queue` function creates and returns a SYCL in-order queue for a specified device, with an optional exception handler.
- **Inputs**:
    - `device`: A `sycl::device` object representing the device for which the queue is to be created.
    - `enable_exception_handler`: A boolean flag indicating whether to enable an exception handler for the queue; defaults to `false`.
- **Control Flow**:
    - The function calls [`create_in_order_queue`](#device_extcreate_in_order_queue) with the provided `device` and `enable_exception_handler` arguments.
    - The [`create_in_order_queue`](#device_extcreate_in_order_queue) function is responsible for creating the actual queue with the specified properties.
- **Output**: Returns a `sycl::queue` object that is an in-order queue associated with the specified device.
- **Functions called**:
    - [`dpct::device_ext::create_in_order_queue`](#device_extcreate_in_order_queue)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::create\_in\_order\_queue<!-- {{#callable:dpct::device_ext::create_in_order_queue}} -->
The `create_in_order_queue` function creates and returns a SYCL queue with in-order execution properties, optionally with an exception handler.
- **Inputs**:
    - `enable_exception_handler`: A boolean flag indicating whether to enable an exception handler for the queue; defaults to false.
- **Control Flow**:
    - Acquire a lock on the mutex `m_mutex` to ensure thread safety.
    - Call [`create_queue_impl`](#device_extcreate_queue_impl) with the `enable_exception_handler` flag and the `sycl::property::queue::in_order()` property to create the queue.
    - Return the created queue.
- **Output**: Returns a `sycl::queue` object configured for in-order execution.
- **Functions called**:
    - [`dpct::device_ext::create_queue_impl`](#device_extcreate_queue_impl)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::create\_in\_order\_queue<!-- {{#callable:dpct::device_ext::create_in_order_queue}} -->
The `create_in_order_queue` function creates and returns a SYCL queue configured to execute tasks in order on a specified device, with an optional exception handler.
- **Inputs**:
    - `device`: A `sycl::device` object representing the device on which the queue will be created.
    - `enable_exception_handler`: A boolean flag indicating whether to enable an exception handler for the queue; defaults to `false`.
- **Control Flow**:
    - Acquire a lock on the mutex `m_mutex` to ensure thread safety during queue creation.
    - Call [`create_queue_impl`](#device_extcreate_queue_impl) with the provided device, exception handler flag, and the `sycl::property::queue::in_order` property to create the queue.
    - Return the created in-order queue.
- **Output**: Returns a `sycl::queue` object configured to execute tasks in order on the specified device.
- **Functions called**:
    - [`dpct::device_ext::create_queue_impl`](#device_extcreate_queue_impl)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::create\_out\_of\_order\_queue<!-- {{#callable:dpct::device_ext::create_out_of_order_queue}} -->
The `create_out_of_order_queue` function creates and returns a SYCL queue that allows out-of-order execution, with an optional exception handler.
- **Inputs**:
    - `enable_exception_handler`: A boolean flag indicating whether to enable an exception handler for the queue; defaults to false.
- **Control Flow**:
    - Acquire a lock on the mutex `m_mutex` to ensure thread safety.
    - Call the [`create_queue_impl`](#device_extcreate_queue_impl) function with the `enable_exception_handler` argument to create the queue.
    - Return the created SYCL queue.
- **Output**: A `sycl::queue` object configured for out-of-order execution.
- **Functions called**:
    - [`dpct::device_ext::create_queue_impl`](#device_extcreate_queue_impl)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::destroy\_queue<!-- {{#callable:dpct::device_ext::destroy_queue}} -->
The `destroy_queue` function removes a specified SYCL queue from a list of queues within a `device_ext` object, ensuring thread safety with a mutex lock.
- **Inputs**:
    - `queue`: A `sycl::queue` object that specifies the queue to be removed from the internal list of queues.
- **Control Flow**:
    - Acquire a lock on the mutex `m_mutex` to ensure thread safety.
    - Use `std::remove_if` to find and remove the specified queue from the `_queues` vector.
    - Erase the removed queue from the `_queues` vector using the `erase` method.
- **Output**: The function does not return any value; it modifies the internal state of the `device_ext` object by removing the specified queue from the `_queues` vector.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::set\_saved\_queue<!-- {{#callable:dpct::device_ext::set_saved_queue}} -->
The `set_saved_queue` function sets the `_saved_queue` member variable to the provided SYCL queue `q` in a thread-safe manner.
- **Inputs**:
    - `q`: A `sycl::queue` object that is to be saved as the `_saved_queue` member variable of the `device_ext` class.
- **Control Flow**:
    - Acquire a lock on the `m_mutex` to ensure thread safety.
    - Assign the input queue `q` to the `_saved_queue` member variable.
- **Output**: This function does not return any value.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_saved\_queue<!-- {{#callable:dpct::device_ext::get_saved_queue}} -->
The `get_saved_queue` function retrieves the saved SYCL queue from a `device_ext` object, ensuring thread safety with a mutex lock.
- **Inputs**: None
- **Control Flow**:
    - A `std::lock_guard` is used to lock the mutex `m_mutex` to ensure thread safety when accessing the `_saved_queue`.
    - The function returns the `_saved_queue` member variable.
- **Output**: The function returns a `sycl::queue` object, which is the saved queue from the `device_ext` instance.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::clear\_queues<!-- {{#callable:dpct::device_ext::clear_queues}} -->
The `clear_queues` function clears all the SYCL queues stored in the `_queues` vector of the `device_ext` class.
- **Inputs**: None
- **Control Flow**:
    - The function calls the `clear` method on the `_queues` vector, which removes all elements from the vector, effectively clearing it.
- **Output**: The function does not return any value.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::init\_queues<!-- {{#callable:dpct::device_ext::init_queues}} -->
The `init_queues` function initializes three SYCL queues: an in-order queue, an out-of-order queue, and sets a saved queue to the default queue.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`create_queue_impl`](#device_extcreate_queue_impl) with `true` and `sycl::property::queue::in_order()` to create an in-order queue and assigns it to `_q_in_order`.
    - It calls [`create_queue_impl`](#device_extcreate_queue_impl) with `true` to create an out-of-order queue and assigns it to `_q_out_of_order`.
    - It sets `_saved_queue` to the result of `default_queue()`.
- **Output**: The function does not return any value; it initializes member variables of the `device_ext` class.
- **Functions called**:
    - [`dpct::device_ext::create_queue_impl`](#device_extcreate_queue_impl)
    - [`dpct::device_ext::default_queue`](#device_extdefault_queue)
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::create\_queue\_impl<!-- {{#callable:dpct::device_ext::create_queue_impl}} -->
The `create_queue_impl` function creates and returns a SYCL queue with specified properties and an optional exception handler.
- **Inputs**:
    - `enable_exception_handler`: A boolean flag indicating whether to enable an exception handler for the queue.
    - `properties`: A variadic template parameter representing additional properties to be applied to the queue.
- **Control Flow**:
    - Initialize an empty `sycl::async_handler` named `eh`.
    - If `enable_exception_handler` is true, set `eh` to `exception_handler`.
    - Create a `sycl::queue` with the current object (`*this`), the async handler `eh`, and a `sycl::property_list` containing the provided properties.
    - Add the created queue to the `_queues` vector.
    - Return the last element of the `_queues` vector, which is the newly created queue.
- **Output**: Returns a `sycl::queue` object that has been added to the `_queues` vector.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::create\_queue\_impl<!-- {{#callable:dpct::device_ext::create_queue_impl}} -->
The `create_queue_impl` function creates and returns a SYCL queue for a specified device with optional exception handling and additional properties.
- **Inputs**:
    - `device`: A `sycl::device` object representing the device for which the queue is to be created.
    - `enable_exception_handler`: A boolean flag indicating whether to enable an exception handler for the queue.
    - `properties`: A variadic template parameter representing additional properties to be applied to the queue.
- **Control Flow**:
    - Initialize an empty `sycl::async_handler` named `eh`.
    - If `enable_exception_handler` is true, set `eh` to the global `exception_handler`.
    - Create a `sycl::queue` with the specified `device`, `eh`, and a `sycl::property_list` containing the provided properties and optionally enable profiling if `DPCT_PROFILING_ENABLED` is defined.
    - Add the created queue to the `_queues` vector.
    - Return the last element of the `_queues` vector, which is the newly created queue.
- **Output**: Returns a `sycl::queue` object that was created and added to the `_queues` vector.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)


---
#### device\_ext::get\_version<!-- {{#callable:dpct::device_ext::get_version}} -->
The `get_version` function retrieves the major and minor version numbers of a device by calling a detailed version retrieval function.
- **Inputs**:
    - `major`: A reference to an integer where the major version number will be stored.
    - `minor`: A reference to an integer where the minor version number will be stored.
- **Control Flow**:
    - The function calls `detail::get_version` with the current device object and the references to the major and minor version integers.
    - The `detail::get_version` function extracts the version information from the device and assigns the major and minor version numbers to the provided references.
- **Output**: The function does not return a value but modifies the input references to store the major and minor version numbers.
- **See also**: [`dpct::device_ext`](#dpctdevice_ext)  (Data Structure)



---
### dev\_mgr<!-- {{#data_structure:dpct::dev_mgr}} -->
- **Type**: `class`
- **Members**:
    - `m_mutex`: A recursive mutex used for thread-safe operations on the device manager.
    - `_devs`: A vector of shared pointers to device_ext objects representing the managed devices.
    - `DEFAULT_DEVICE_ID`: A constant unsigned integer representing the default device ID.
    - `_thread2dev_map`: A map that associates thread IDs with device IDs.
    - `_cpu_device`: An integer representing the index of the CPU device in the _devs vector.
- **Description**: The `dev_mgr` class is a singleton device manager responsible for managing and interacting with SYCL devices. It provides functionality to select, retrieve, and manage devices, including CPU and GPU devices, within a SYCL platform. The class maintains a list of devices, allows for device selection based on thread ID, and ensures thread-safe operations using a recursive mutex. It also provides methods to get the current device, CPU device, and device count, as well as to select a preferred GPU platform based on environment variables or default settings.
- **Member Functions**:
    - [`dpct::dev_mgr::current_device`](#dev_mgrcurrent_device)
    - [`dpct::dev_mgr::cpu_device`](#dev_mgrcpu_device)
    - [`dpct::dev_mgr::get_device`](#dev_mgrget_device)
    - [`dpct::dev_mgr::current_device_id`](#dev_mgrcurrent_device_id)
    - [`dpct::dev_mgr::select_device`](#dev_mgrselect_device)
    - [`dpct::dev_mgr::device_count`](#dev_mgrdevice_count)
    - [`dpct::dev_mgr::get_device_id`](#dev_mgrget_device_id)
    - [`dpct::dev_mgr::get_preferred_gpu_platform_name`](#dev_mgrget_preferred_gpu_platform_name)
    - [`dpct::dev_mgr::select_device`](#dev_mgrselect_device)
    - [`dpct::dev_mgr::instance`](#dev_mgrinstance)
    - [`dpct::dev_mgr::dev_mgr`](#dev_mgrdev_mgr)
    - [`dpct::dev_mgr::operator=`](#dev_mgroperator=)
    - [`dpct::dev_mgr::dev_mgr`](#dev_mgrdev_mgr)
    - [`dpct::dev_mgr::operator=`](#dev_mgroperator=)
    - [`dpct::dev_mgr::compare_dev`](#dev_mgrcompare_dev)
    - [`dpct::dev_mgr::convert_backend_index`](#dev_mgrconvert_backend_index)
    - [`dpct::dev_mgr::compare_backend`](#dev_mgrcompare_backend)
    - [`dpct::dev_mgr::dev_mgr`](#dev_mgrdev_mgr)
    - [`dpct::dev_mgr::check_id`](#dev_mgrcheck_id)

**Methods**

---
#### dev\_mgr::current\_device<!-- {{#callable:dpct::dev_mgr::current_device}} -->
The `current_device` function retrieves the current device object based on the current thread's device ID.
- **Inputs**: None
- **Control Flow**:
    - Call `current_device_id()` to get the current device ID for the thread.
    - Invoke `check_id(dev_id)` to validate the device ID.
    - Return the device object from the `_devs` vector using the validated device ID.
- **Output**: Returns a reference to the `device_ext` object representing the current device.
- **Functions called**:
    - [`dpct::dev_mgr::current_device_id`](#dev_mgrcurrent_device_id)
    - [`dpct::dev_mgr::check_id`](#dev_mgrcheck_id)
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::cpu\_device<!-- {{#callable:dpct::dev_mgr::cpu_device}} -->
The `cpu_device` function returns a reference to the CPU device from the device manager if it is valid, otherwise it throws an exception.
- **Inputs**: None
- **Control Flow**:
    - The function acquires a lock on a recursive mutex to ensure thread safety.
    - It checks if the `_cpu_device` index is -1, indicating no valid CPU device is set.
    - If `_cpu_device` is -1, it throws a `std::runtime_error` with the message 'no valid cpu device'.
    - If `_cpu_device` is valid, it returns a reference to the device at the `_cpu_device` index in the `_devs` vector.
- **Output**: A reference to a `device_ext` object representing the CPU device.
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::get\_device<!-- {{#callable:dpct::dev_mgr::get_device}} -->
The `get_device` function retrieves a reference to a device object from a device manager based on a given device ID.
- **Inputs**:
    - `id`: An unsigned integer representing the ID of the device to be retrieved.
- **Control Flow**:
    - A lock is acquired on a recursive mutex to ensure thread safety when accessing shared resources.
    - The function [`check_id`](#dev_mgrcheck_id) is called to validate the provided device ID, ensuring it is within the valid range of available devices.
    - The function returns a reference to the device object corresponding to the provided ID from the `_devs` vector.
- **Output**: A reference to a `device_ext` object corresponding to the specified device ID.
- **Functions called**:
    - [`dpct::dev_mgr::check_id`](#dev_mgrcheck_id)
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::current\_device\_id<!-- {{#callable:dpct::dev_mgr::current_device_id}} -->
The `current_device_id` function retrieves the device ID associated with the current thread or returns a default device ID if no association exists.
- **Inputs**: None
- **Control Flow**:
    - The function acquires a lock on a recursive mutex to ensure thread safety.
    - It retrieves the current thread ID using the `get_tid()` function.
    - It searches for the thread ID in the `_thread2dev_map` to find the associated device ID.
    - If the thread ID is found in the map, it returns the corresponding device ID.
    - If the thread ID is not found, it returns the `DEFAULT_DEVICE_ID`.
- **Output**: The function returns an `unsigned int` representing the device ID associated with the current thread or the default device ID if no association is found.
- **Functions called**:
    - [`dpct::get_tid`](#dpctget_tid)
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::select\_device<!-- {{#callable:dpct::dev_mgr::select_device}} -->
The `select_device` function assigns a specified device ID to the current thread in a thread-to-device map, ensuring thread-safe access with a mutex lock.
- **Inputs**:
    - `id`: An unsigned integer representing the ID of the device to be selected for the current thread.
- **Control Flow**:
    - Acquire a lock on the recursive mutex `m_mutex` to ensure thread-safe access to shared resources.
    - Call the [`check_id`](#dev_mgrcheck_id) function to validate the provided device ID against the available devices.
    - Map the current thread ID, obtained via `get_tid()`, to the specified device ID in the `_thread2dev_map`.
- **Output**: The function does not return any value; it modifies the internal state of the `dev_mgr` class by updating the `_thread2dev_map`.
- **Functions called**:
    - [`dpct::dev_mgr::check_id`](#dev_mgrcheck_id)
    - [`dpct::get_tid`](#dpctget_tid)
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::device\_count<!-- {{#callable:dpct::dev_mgr::device_count}} -->
The `device_count` function returns the number of devices managed by the `dev_mgr` class.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `_devs` vector, which stores shared pointers to `device_ext` objects representing devices.
    - It returns the size of the `_devs` vector, which indicates the number of devices currently managed.
- **Output**: The function returns an unsigned integer representing the count of devices.
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::get\_device\_id<!-- {{#callable:dpct::dev_mgr::get_device_id}} -->
The `get_device_id` function retrieves the index of a given SYCL device from a list of devices managed by the `dev_mgr` class.
- **Inputs**:
    - `dev`: A constant reference to a `sycl::device` object representing the device whose index is to be retrieved.
- **Control Flow**:
    - Initialize an unsigned integer `id` to 0 to serve as the index counter.
    - Iterate over each device in the `_devs` vector, which stores shared pointers to `device_ext` objects.
    - For each device, check if it matches the input device `dev` using the dereferenced pointer comparison.
    - If a match is found, return the current value of `id` as the index of the device.
    - If no match is found after iterating through all devices, return -1 to indicate the device is not in the list.
- **Output**: Returns an unsigned integer representing the index of the device in the `_devs` vector, or -1 if the device is not found.
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::get\_preferred\_gpu\_platform\_name<!-- {{#callable:dpct::dev_mgr::get_preferred_gpu_platform_name}} -->
The `get_preferred_gpu_platform_name` function determines and returns the name of the preferred GPU platform based on environment variables or default device selection.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty string `result` and `filter`.
    - Check if the environment variable `ONEAPI_DEVICE_SELECTOR` is set.
    - If set, determine the `filter` based on the value of `ONEAPI_DEVICE_SELECTOR` (e.g., 'level_zero', 'opencl', 'cuda', 'hip').
    - If not set, use the default SYCL device to determine the `filter` based on the platform name or if the device is a CPU.
    - Retrieve the list of available SYCL platforms.
    - Iterate over each platform to find a GPU device that matches the `filter`.
    - If a matching platform is found, set `result` to the platform's name.
    - If no matching platform is found, throw a runtime error.
    - Return the `result` containing the preferred GPU platform name.
- **Output**: Returns a string representing the name of the preferred GPU platform.
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::select\_device<!-- {{#callable:dpct::dev_mgr::select_device}} -->
The [`select_device`](#dev_mgrselect_device) function selects a SYCL device based on a provided device selector and updates the current device ID in the device manager.
- **Inputs**:
    - `DeviceSelector`: A callable object that takes a `const sycl::device &` and returns an `int`, used to select a device.
    - `selector`: An optional device selector, defaulting to `sycl::gpu_selector_v`, used to choose a SYCL device.
- **Control Flow**:
    - A `sycl::device` is created using the provided `selector`.
    - The `get_device_id` method is called to retrieve the ID of the selected device.
    - The [`select_device`](#dev_mgrselect_device) method is called with the retrieved device ID to update the current device in the device manager.
- **Output**: This function does not return a value; it updates the current device in the device manager.
- **Functions called**:
    - [`dpct::dev_mgr::select_device`](#dev_mgrselect_device)
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::instance<!-- {{#callable:dpct::dev_mgr::instance}} -->
The `instance` function returns a singleton instance of the `dev_mgr` class.
- **Inputs**: None
- **Control Flow**:
    - The function defines a static local variable `d_m` of type `dev_mgr`.
    - The function returns a reference to `d_m`.
- **Output**: A reference to the singleton instance of `dev_mgr`.
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::dev\_mgr<!-- {{#callable:dpct::dev_mgr::dev_mgr}} -->
The `dev_mgr` constructor initializes a singleton device manager by selecting a preferred GPU platform and organizing available SYCL devices based on their backend and compute capabilities.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes a default SYCL device using the default selector and adds it to the `_devs` vector.
    - If the default device is a CPU, it sets `_cpu_device` to 0.
    - It retrieves all available SYCL platforms and determines the preferred GPU platform name using `get_preferred_gpu_platform_name()`.
    - For each platform, if its name matches the preferred platform, it retrieves its devices and categorizes them by backend type.
    - The devices are sorted by backend priority using `compare_backend` and by compute capability using `compare_dev`.
    - Each device, except the default device, is added to the `_devs` vector, and if a CPU device is found, `_cpu_device` is updated.
- **Output**: The constructor does not return any value as it is responsible for initializing the `dev_mgr` instance.
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::operator=<!-- {{#callable:dpct::dev_mgr::operator=}} -->
The `operator=` for the `dev_mgr` class is deleted to prevent assignment of `dev_mgr` instances.
- **Inputs**: None
- **Control Flow**:
    - The `operator=` is explicitly deleted, meaning any attempt to assign one `dev_mgr` instance to another will result in a compile-time error.
- **Output**: There is no output as the function is deleted and cannot be used.
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::dev\_mgr<!-- {{#callable:dpct::dev_mgr::dev_mgr}} -->
The `dev_mgr` class constructor is deleted to prevent move operations, ensuring that instances of the class cannot be moved.
- **Inputs**: None
- **Control Flow**:
    - The move constructor `dev_mgr(dev_mgr &&)` is deleted, preventing the creation of a new `dev_mgr` instance by moving an existing one.
    - The move assignment operator `dev_mgr &operator=(dev_mgr &&)` is also deleted, preventing the assignment of a `dev_mgr` instance by moving from another instance.
- **Output**: There is no output from these operations as they are deleted constructors and assignment operators.
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::operator=<!-- {{#callable:dpct::dev_mgr::operator=}} -->
The move assignment operator for the `dev_mgr` class is deleted, preventing move assignment of `dev_mgr` objects.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as a deleted function, meaning it cannot be used or called.
    - No operations or logic are performed within this function.
- **Output**: There is no output as the function is deleted and cannot be invoked.
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::compare\_dev<!-- {{#callable:dpct::dev_mgr::compare_dev}} -->
The `compare_dev` function compares two SYCL devices based on their backend type and maximum compute units.
- **Inputs**:
    - `device1`: A reference to the first SYCL device to be compared.
    - `device2`: A reference to the second SYCL device to be compared.
- **Control Flow**:
    - Retrieve the backend type of both devices using `get_backend()` method.
    - Check if the first device's backend is `ext_oneapi_level_zero` and the second is not, returning true if so.
    - Check if the second device's backend is `ext_oneapi_level_zero` and the first is not, returning false if so.
    - Retrieve device information for both devices using `dpct::get_device_info`.
    - Compare the maximum compute units of both devices and return true if the first device has more, otherwise return false.
- **Output**: Returns a boolean indicating whether the first device should be considered 'less than' the second device based on the specified criteria.
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::convert\_backend\_index<!-- {{#callable:dpct::dev_mgr::convert_backend_index}} -->
The `convert_backend_index` function maps a given backend string to a corresponding integer index, or aborts if the backend is unrecognized.
- **Inputs**:
    - `backend`: A reference to a string representing the backend type, such as 'ext_oneapi_level_zero:gpu' or 'opencl:gpu'.
- **Control Flow**:
    - Check if the input string matches 'ext_oneapi_level_zero:gpu'; if so, return 0.
    - Check if the input string matches 'opencl:gpu'; if so, return 1.
    - Check if the input string matches 'ext_oneapi_cuda:gpu'; if so, return 2.
    - Check if the input string matches 'ext_oneapi_hip:gpu'; if so, return 3.
    - Check if the input string matches 'opencl:cpu'; if so, return 4.
    - Check if the input string matches 'opencl:acc'; if so, return 5.
    - If none of the above conditions are met, print an error message and abort the program.
- **Output**: An integer representing the index of the backend, or the program aborts if the backend is unrecognized.
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::compare\_backend<!-- {{#callable:dpct::dev_mgr::compare_backend}} -->
The `compare_backend` function compares two backend strings by converting them to indices and checking their order.
- **Inputs**:
    - `backend1`: A reference to a string representing the first backend to compare.
    - `backend2`: A reference to a string representing the second backend to compare.
- **Control Flow**:
    - The function calls [`convert_backend_index`](#dev_mgrconvert_backend_index) on `backend1` to get its index.
    - The function calls [`convert_backend_index`](#dev_mgrconvert_backend_index) on `backend2` to get its index.
    - The function returns `true` if the index of `backend1` is less than the index of `backend2`, otherwise it returns `false`.
- **Output**: A boolean value indicating whether the first backend is considered less than the second based on their indices.
- **Functions called**:
    - [`dpct::dev_mgr::convert_backend_index`](#dev_mgrconvert_backend_index)
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::dev\_mgr<!-- {{#callable:dpct::dev_mgr::dev_mgr}} -->
The `dev_mgr` constructor initializes a device manager by selecting a default SYCL device, collecting all available devices from preferred platforms, and storing them in a sorted order based on backend and compute capabilities.
- **Inputs**: None
- **Control Flow**:
    - Initialize a default SYCL device using the default selector and add it to the device list `_devs`.
    - Check if the default device is a CPU and set `_cpu_device` to 0 if true.
    - Retrieve all available SYCL platforms and determine the preferred GPU platform name using `get_preferred_gpu_platform_name()`.
    - Iterate over the platforms, filtering out those that do not match the preferred platform name.
    - For each matching platform, retrieve its devices and categorize them by backend type using `get_device_backend_and_type()`.
    - Sort the backend types and devices within each backend type using `compare_backend` and `compare_dev` respectively.
    - Add all devices, except the default device, to `_devs` and update `_cpu_device` if a CPU device is found.
- **Output**: The constructor does not return any value; it initializes the internal state of the `dev_mgr` instance.
- **Functions called**:
    - [`dpct::dev_mgr::get_preferred_gpu_platform_name`](#dev_mgrget_preferred_gpu_platform_name)
    - [`get_device_backend_and_type`](#get_device_backend_and_type)
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)


---
#### dev\_mgr::check\_id<!-- {{#callable:dpct::dev_mgr::check_id}} -->
The `check_id` function verifies if a given device ID is valid by checking if it is within the bounds of the `_devs` vector.
- **Inputs**:
    - `id`: An unsigned integer representing the device ID to be checked.
- **Control Flow**:
    - The function checks if the provided `id` is greater than or equal to the size of the `_devs` vector.
    - If the condition is true, it throws a `std::runtime_error` with the message "invalid device id".
- **Output**: The function does not return any value; it throws an exception if the ID is invalid.
- **See also**: [`dpct::dev_mgr`](#dpctdev_mgr)  (Data Structure)



---
### pointer\_access\_attribute<!-- {{#data_structure:dpct::detail::pointer_access_attribute}} -->
- **Type**: `enum class`
- **Members**:
    - `host_only`: Represents a pointer that is accessible only by the host.
    - `device_only`: Represents a pointer that is accessible only by the device.
    - `host_device`: Represents a pointer that is accessible by both the host and the device.
    - `end`: Marks the end of the enumeration, used for iteration or boundary checks.
- **Description**: The `pointer_access_attribute` is an enumeration class that defines different access attributes for pointers in a SYCL context. It specifies whether a pointer is accessible only by the host, only by the device, or by both, which is crucial for managing memory in heterogeneous computing environments. The `end` member is used to denote the end of the enumeration, which can be useful for iteration or validation purposes.


---
### mem\_mgr<!-- {{#data_structure:dpct::detail::mem_mgr}} -->
- **Type**: `class`
- **Members**:
    - `m_map`: A map that associates device pointers with their corresponding allocation details.
    - `m_mutex`: A mutex to ensure thread-safe access to the memory manager's resources.
    - `mapped_address_space`: A pointer to the start of the reserved address space for memory allocation.
    - `next_free`: A pointer to the next free address in the reserved address space.
    - `mapped_region_size`: The total size of the reserved address space, set to 128 GB.
    - `alignment`: The alignment size for memory allocations, set to 256 bytes.
    - `extra_padding`: An optional padding size for debugging out-of-bound accesses, set to 0.
- **Description**: The `mem_mgr` class is a singleton memory manager designed to handle memory allocation and deallocation in a virtual memory pool. It reserves a large address space (128 GB) without actual memory allocation initially, and manages allocations within this space using a map to track device pointers and their corresponding allocation details. The class ensures thread safety with a mutex and provides methods for memory allocation, deallocation, and pointer translation. It supports both Linux and Windows platforms, using `mmap` and `VirtualAlloc` respectively for reserving address space.
- **Member Functions**:
    - [`dpct::detail::mem_mgr::mem_mgr`](#mem_mgrmem_mgr)
    - [`dpct::detail::mem_mgr::~mem_mgr`](#mem_mgrmem_mgr)
    - [`dpct::detail::mem_mgr::mem_mgr`](#mem_mgrmem_mgr)
    - [`dpct::detail::mem_mgr::operator=`](#mem_mgroperator=)
    - [`dpct::detail::mem_mgr::mem_mgr`](#mem_mgrmem_mgr)
    - [`dpct::detail::mem_mgr::operator=`](#mem_mgroperator=)
    - [`dpct::detail::mem_mgr::mem_alloc`](#mem_mgrmem_alloc)
    - [`dpct::detail::mem_mgr::mem_free`](#mem_mgrmem_free)
    - [`dpct::detail::mem_mgr::translate_ptr`](#mem_mgrtranslate_ptr)
    - [`dpct::detail::mem_mgr::is_device_ptr`](#mem_mgris_device_ptr)
    - [`dpct::detail::mem_mgr::instance`](#mem_mgrinstance)
    - [`dpct::detail::mem_mgr::get_map_iterator`](#mem_mgrget_map_iterator)

**Methods**

---
#### mem\_mgr::mem\_mgr<!-- {{#callable:dpct::detail::mem_mgr::mem_mgr}} -->
The `mem_mgr` constructor initializes a memory manager by reserving a large address space for memory allocation without actually allocating physical memory.
- **Inputs**: None
- **Control Flow**:
    - The constructor checks the operating system using preprocessor directives.
    - If the system is Linux, it uses `mmap` to reserve address space with no access permissions.
    - If the system is Windows, it uses `VirtualAlloc` to reserve address space with no access permissions.
    - An error is thrown if the system is neither Windows nor Linux.
    - The `next_free` pointer is initialized to the start of the reserved address space.
- **Output**: The constructor does not return any value; it initializes the `mem_mgr` object state.
- **See also**: [`dpct::detail::mem_mgr`](#detailmem_mgr)  (Data Structure)


---
#### mem\_mgr::\~mem\_mgr<!-- {{#callable:dpct::detail::mem_mgr::~mem_mgr}} -->
The destructor `~mem_mgr` releases the reserved memory address space used by the `mem_mgr` class, depending on the operating system.
- **Inputs**: None
- **Control Flow**:
    - The destructor checks if the code is being compiled for a Linux environment using `#if defined(__linux__)`.
    - If it is Linux, it calls `munmap` to unmap the memory region specified by `mapped_address_space` and `mapped_region_size`.
    - If the code is being compiled for a Windows environment using `#elif defined(_WIN64)`, it calls `VirtualFree` to release the memory region specified by `mapped_address_space`.
    - If neither Linux nor Windows is defined, it raises a compilation error with `#error "Only support Windows and Linux."`.
- **Output**: The function does not return any value as it is a destructor.
- **See also**: [`dpct::detail::mem_mgr`](#detailmem_mgr)  (Data Structure)


---
#### mem\_mgr::mem\_mgr<!-- {{#callable:dpct::detail::mem_mgr::mem_mgr}} -->
The `mem_mgr` constructor is a private constructor that initializes a memory manager by reserving a large address space without actual memory allocation, and sets up the initial state for memory allocation.
- **Inputs**: None
- **Control Flow**:
    - The constructor is private, ensuring that instances of `mem_mgr` can only be created within the class itself, typically for singleton pattern implementation.
    - The constructor checks the operating system using preprocessor directives to determine whether to use `mmap` for Linux or `VirtualAlloc` for Windows to reserve address space.
    - For Linux, `mmap` is used to reserve a large address space with no access permissions, while for Windows, `VirtualAlloc` is used to reserve pages with no access.
    - The `next_free` pointer is initialized to point to the start of the reserved address space, setting up the initial state for future memory allocations.
- **Output**: The constructor does not return any value as it is a constructor for initializing an instance of the `mem_mgr` class.
- **See also**: [`dpct::detail::mem_mgr`](#detailmem_mgr)  (Data Structure)


---
#### mem\_mgr::operator=<!-- {{#callable:dpct::detail::mem_mgr::operator=}} -->
The `operator=` for the `mem_mgr` class is deleted to prevent assignment of `mem_mgr` objects.
- **Inputs**: None
- **Control Flow**:
    - The `operator=` is explicitly deleted, meaning any attempt to use the assignment operator on `mem_mgr` objects will result in a compile-time error.
- **Output**: There is no output as the function is deleted and cannot be used.
- **See also**: [`dpct::detail::mem_mgr`](#detailmem_mgr)  (Data Structure)


---
#### mem\_mgr::mem\_mgr<!-- {{#callable:dpct::detail::mem_mgr::mem_mgr}} -->
The `mem_mgr` class is a singleton memory manager that handles allocation and deallocation of memory in a reserved address space, ensuring thread safety and alignment.
- **Inputs**: None
- **Control Flow**:
    - The constructor `mem_mgr()` reserves address space using `mmap` on Linux or `VirtualAlloc` on Windows, without actual memory allocation, and initializes `next_free` to the start of this space.
    - The destructor `~mem_mgr()` releases the reserved address space using `munmap` on Linux or `VirtualFree` on Windows.
    - The `mem_alloc(size_t size)` method checks if the requested size is zero, then locks a mutex for thread safety, checks if there is enough space for the allocation, and if so, allocates a buffer, updates the `next_free` pointer with alignment and padding, and returns the allocated pointer.
    - The `mem_free(const void *ptr)` method checks if the pointer is null, locks a mutex, retrieves the allocation map iterator for the pointer, and erases the allocation from the map.
    - The `translate_ptr(const void *ptr)` method locks a mutex, retrieves the allocation map iterator for the pointer, and returns the allocation details.
    - The `is_device_ptr(const void *ptr)` method locks a mutex and checks if the pointer is within the reserved address space.
    - The `instance()` method returns a static instance of `mem_mgr`, ensuring it is a singleton.
- **Output**: The function does not return any output directly, but manages memory allocations and deallocations within the reserved address space.
- **See also**: [`dpct::detail::mem_mgr`](#detailmem_mgr)  (Data Structure)


---
#### mem\_mgr::operator=<!-- {{#callable:dpct::detail::mem_mgr::operator=}} -->
The move assignment operator for the `mem_mgr` class is deleted, preventing move assignment of `mem_mgr` objects.
- **Inputs**: None
- **Control Flow**:
    - The move assignment operator is explicitly deleted using `= delete;`, which means any attempt to move-assign a `mem_mgr` object will result in a compilation error.
- **Output**: There is no output as the function is deleted and cannot be used.
- **See also**: [`dpct::detail::mem_mgr`](#detailmem_mgr)  (Data Structure)


---
#### mem\_mgr::mem\_alloc<!-- {{#callable:dpct::detail::mem_mgr::mem_alloc}} -->
The `mem_alloc` function allocates a block of memory from a pre-mapped virtual memory pool, ensuring thread safety and alignment.
- **Inputs**:
    - `size`: The size of the memory block to allocate, in bytes.
- **Control Flow**:
    - Check if the requested size is zero and return nullptr if true.
    - Acquire a lock on the mutex to ensure thread safety during allocation.
    - Check if there is enough space in the mapped address space for the requested allocation; throw a runtime error if not.
    - Create a SYCL buffer of the requested size.
    - Create an allocation record with the buffer, current free pointer, and size.
    - Map the allocation to the current free pointer and store it in the map.
    - Calculate the next free pointer position, considering alignment and padding.
    - Return the pointer to the allocated memory block.
- **Output**: A pointer to the allocated memory block, or nullptr if the size is zero.
- **See also**: [`dpct::detail::mem_mgr`](#detailmem_mgr)  (Data Structure)


---
#### mem\_mgr::mem\_free<!-- {{#callable:dpct::detail::mem_mgr::mem_free}} -->
The `mem_free` function deallocates a memory block by removing its entry from a map that tracks allocations.
- **Inputs**:
    - `ptr`: A constant pointer to the memory block that needs to be deallocated.
- **Control Flow**:
    - Check if the input pointer `ptr` is null; if so, return immediately without doing anything.
    - Acquire a lock on the mutex `m_mutex` to ensure thread safety during the deallocation process.
    - Retrieve the iterator for the map entry corresponding to the memory block pointed to by `ptr` using the [`get_map_iterator`](#mem_mgrget_map_iterator) function.
    - Erase the map entry from `m_map` using the iterator, effectively deallocating the memory block.
- **Output**: The function does not return any value.
- **Functions called**:
    - [`dpct::detail::mem_mgr::get_map_iterator`](#mem_mgrget_map_iterator)
- **See also**: [`dpct::detail::mem_mgr`](#detailmem_mgr)  (Data Structure)


---
#### mem\_mgr::translate\_ptr<!-- {{#callable:dpct::detail::mem_mgr::translate_ptr}} -->
The `translate_ptr` function retrieves the allocation details associated with a given device pointer from a map, ensuring thread safety with a mutex lock.
- **Inputs**:
    - `ptr`: A constant void pointer representing the device pointer whose allocation details are to be retrieved.
- **Control Flow**:
    - The function begins by acquiring a lock on a mutex to ensure thread safety when accessing shared resources.
    - It then calls the [`get_map_iterator`](#mem_mgrget_map_iterator) function to find the iterator pointing to the map entry corresponding to the given pointer.
    - Finally, it returns the `allocation` object associated with the found iterator.
- **Output**: An `allocation` object containing the buffer, allocation pointer, and size associated with the given device pointer.
- **Functions called**:
    - [`dpct::detail::mem_mgr::get_map_iterator`](#mem_mgrget_map_iterator)
- **See also**: [`dpct::detail::mem_mgr`](#detailmem_mgr)  (Data Structure)


---
#### mem\_mgr::is\_device\_ptr<!-- {{#callable:dpct::detail::mem_mgr::is_device_ptr}} -->
The `is_device_ptr` function checks if a given pointer falls within the reserved device memory address space.
- **Inputs**:
    - `ptr`: A constant void pointer representing the memory address to be checked.
- **Control Flow**:
    - The function acquires a lock on a mutex to ensure thread safety.
    - It checks if the pointer `ptr` is greater than or equal to `mapped_address_space` and less than `mapped_address_space + mapped_region_size`.
    - The function returns true if both conditions are met, indicating that `ptr` is within the device memory range; otherwise, it returns false.
- **Output**: A boolean value indicating whether the pointer is within the device memory address space.
- **See also**: [`dpct::detail::mem_mgr`](#detailmem_mgr)  (Data Structure)


---
#### mem\_mgr::instance<!-- {{#callable:dpct::detail::mem_mgr::instance}} -->
The `instance` function returns a singleton instance of the `mem_mgr` class.
- **Inputs**: None
- **Control Flow**:
    - The function defines a static local variable `m` of type `mem_mgr`.
    - The function returns the reference to this static variable `m`.
- **Output**: A reference to a singleton instance of the `mem_mgr` class.
- **See also**: [`dpct::detail::mem_mgr`](#detailmem_mgr)  (Data Structure)


---
#### mem\_mgr::get\_map\_iterator<!-- {{#callable:dpct::detail::mem_mgr::get_map_iterator}} -->
The `get_map_iterator` function retrieves an iterator to a map entry corresponding to a given pointer, ensuring the pointer is within valid bounds of a virtual memory allocation.
- **Inputs**:
    - `ptr`: A constant void pointer representing the address to be checked against the map of allocations.
- **Control Flow**:
    - Convert the input pointer `ptr` to a `byte_t` pointer and find the first map entry with a key greater than this pointer using `upper_bound`.
    - Check if the iterator `it` is at the end of the map, indicating that the pointer is not a virtual pointer, and throw a runtime error if so.
    - Retrieve the allocation associated with the iterator `it`.
    - Check if the input pointer `ptr` is less than the allocation's pointer `alloc_ptr`, indicating an out-of-bounds access, and throw a runtime error if so.
    - Return the iterator `it` if all checks pass.
- **Output**: An iterator to the map entry corresponding to the given pointer, if the pointer is valid and within bounds.
- **See also**: [`dpct::detail::mem_mgr`](#detailmem_mgr)  (Data Structure)



---
### allocation<!-- {{#data_structure:dpct::detail::mem_mgr::allocation}} -->
- **Type**: `struct`
- **Members**:
    - `buffer`: A buffer_t object representing the buffer associated with the allocation.
    - `alloc_ptr`: A pointer to the allocated memory of type byte_t.
    - `size`: A size_t value indicating the size of the allocation.
- **Description**: The `allocation` struct is a simple data structure used to represent a memory allocation. It contains a buffer, a pointer to the allocated memory, and the size of the allocation. This struct is likely used in the context of memory management, particularly in environments where memory allocations need to be tracked and managed, such as in a SYCL or similar parallel computing framework.


---
### memory\_traits<!-- {{#data_structure:dpct::detail::memory_traits}} -->
- **Type**: `class`
- **Members**:
    - `target`: Specifies the SYCL access target, defaulting to device.
    - `mode`: Defines the SYCL access mode, set to read for constant memory and read_write otherwise.
    - `type_size`: Holds the size of the type T in bytes.
    - `element_t`: Defines the type of elements, const T for constant memory and T otherwise.
    - `value_t`: Represents the non-const version of type T.
    - `accessor_t`: Defines a type for SYCL accessors, using local_accessor for local memory and accessor for other memory types.
    - `pointer_t`: Typedef for a pointer to type T.
- **Description**: The `memory_traits` class template is designed to encapsulate various traits and types associated with different memory regions in a SYCL environment. It provides static constants and type definitions that help in managing memory access modes, target specifications, and type sizes for different memory regions such as global, constant, and local. The class also defines accessor types for handling memory access in SYCL kernels, adapting to the specific memory region being used. This makes it easier to write generic code that can operate on different types of memory in a SYCL application.


---
### host\_buffer<!-- {{#data_structure:dpct::dpct_memcpy::host_buffer}} -->
- **Type**: `class`
- **Members**:
    - `_buf`: A pointer to the allocated buffer in host memory.
    - `_size`: The size of the buffer in bytes.
    - `_q`: A reference to a SYCL queue used for submitting tasks.
    - `_deps`: A reference to a vector of SYCL events that the free operation depends on.
- **Description**: The `host_buffer` class is a resource management class that encapsulates a buffer allocated in host memory, along with its size, and manages its lifecycle using a SYCL queue. It ensures that the buffer is properly freed after use, taking into account any dependencies specified by a vector of SYCL events. The class provides methods to access the buffer pointer and its size, and it automatically handles the deallocation of the buffer in a SYCL task, ensuring that any dependent events are completed before the buffer is freed.
- **Member Functions**:
    - [`dpct::dpct_memcpy::host_buffer::host_buffer`](#host_bufferhost_buffer)
    - [`dpct::dpct_memcpy::host_buffer::get_ptr`](#host_bufferget_ptr)
    - [`dpct::dpct_memcpy::host_buffer::get_size`](#host_bufferget_size)
    - [`dpct::dpct_memcpy::host_buffer::~host_buffer`](#host_bufferhost_buffer)

**Methods**

---
#### host\_buffer::host\_buffer<!-- {{#callable:dpct::dpct_memcpy::host_buffer::host_buffer}} -->
The `host_buffer` constructor initializes a buffer of a specified size and associates it with a SYCL queue and a list of dependency events.
- **Inputs**:
    - `size`: The size of the buffer to be allocated in bytes.
    - `q`: A reference to a SYCL queue that will be used for submitting tasks related to this buffer.
    - `deps`: A constant reference to a vector of SYCL events that represent dependencies for operations on this buffer.
- **Control Flow**:
    - Allocate memory of the specified size using `std::malloc` and assign it to `_buf`.
    - Initialize `_size` with the provided size.
    - Store the reference to the provided SYCL queue in `_q`.
    - Store the reference to the provided vector of SYCL events in `_deps`.
- **Output**: The constructor does not return a value; it initializes the `host_buffer` object.
- **See also**: [`dpct::dpct_memcpy::host_buffer`](#dpct_memcpyhost_buffer)  (Data Structure)


---
#### host\_buffer::get\_ptr<!-- {{#callable:dpct::dpct_memcpy::host_buffer::get_ptr}} -->
The `get_ptr` function returns a pointer to the buffer managed by the `host_buffer` class.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the private member `_buf` of the `host_buffer` class, which is a pointer to the allocated memory buffer.
- **Output**: A `void*` pointer to the buffer managed by the `host_buffer` class.
- **See also**: [`dpct::dpct_memcpy::host_buffer`](#dpct_memcpyhost_buffer)  (Data Structure)


---
#### host\_buffer::get\_size<!-- {{#callable:dpct::dpct_memcpy::host_buffer::get_size}} -->
The `get_size` function returns the size of the buffer managed by the `host_buffer` class.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the value of the private member variable `_size`.
- **Output**: The function returns a `size_t` value representing the size of the buffer.
- **See also**: [`dpct::dpct_memcpy::host_buffer`](#dpct_memcpyhost_buffer)  (Data Structure)


---
#### host\_buffer::\~host\_buffer<!-- {{#callable:dpct::dpct_memcpy::host_buffer::~host_buffer}} -->
The destructor `~host_buffer` releases the allocated buffer memory asynchronously using a SYCL queue and its associated dependencies.
- **Inputs**: None
- **Control Flow**:
    - Check if the buffer `_buf` is not null.
    - If `_buf` is not null, submit a task to the SYCL queue `_q`.
    - In the submitted task, set dependencies using `cgh.depends_on(_deps)`.
    - Define a host task to free the buffer using `std::free(buf)` within the submitted task.
- **Output**: The function does not return any value; it performs cleanup operations.
- **See also**: [`dpct::dpct_memcpy::host_buffer`](#dpct_memcpyhost_buffer)  (Data Structure)



---
### usm\_allocator<!-- {{#data_structure:dpct::detail::deprecated::usm_allocator}} -->
- **Type**: `class`
- **Members**:
    - `_impl`: A private member of type `sycl::usm_allocator<T, AllocKind>` used to implement the allocator.
- **Description**: The `usm_allocator` class is a template class that provides a custom allocator for Unified Shared Memory (USM) in SYCL. It is parameterized by a type `T` and an allocation kind `AllocKind`. The class encapsulates a SYCL USM allocator and provides various type definitions and methods to manage memory allocation and deallocation. It supports operations like address retrieval, memory allocation, deallocation, and comparison of allocator instances. The class also includes a nested `rebind` struct template to allow the allocator to be rebound to a different type.
- **Member Functions**:
    - [`dpct::detail::deprecated::usm_allocator::usm_allocator`](#usm_allocatorusm_allocator)
    - [`dpct::detail::deprecated::usm_allocator::~usm_allocator`](#usm_allocatorusm_allocator)
    - [`dpct::detail::deprecated::usm_allocator::usm_allocator`](#usm_allocatorusm_allocator)
    - [`dpct::detail::deprecated::usm_allocator::usm_allocator`](#usm_allocatorusm_allocator)
    - [`dpct::detail::deprecated::usm_allocator::address`](#usm_allocatoraddress)
    - [`dpct::detail::deprecated::usm_allocator::address`](#usm_allocatoraddress)
    - [`dpct::detail::deprecated::usm_allocator::allocate`](#usm_allocatorallocate)
    - [`dpct::detail::deprecated::usm_allocator::deallocate`](#usm_allocatordeallocate)
    - [`dpct::detail::deprecated::usm_allocator::max_size`](#usm_allocatormax_size)
    - [`dpct::detail::deprecated::usm_allocator::operator==`](#usm_allocatoroperator==)
    - [`dpct::detail::deprecated::usm_allocator::operator!=`](#usm_allocatoroperator!=)

**Methods**

---
#### usm\_allocator::usm\_allocator<!-- {{#callable:dpct::detail::deprecated::usm_allocator::usm_allocator}} -->
The `usm_allocator` constructor initializes a USM allocator with the default SYCL queue.
- **Inputs**: None
- **Control Flow**:
    - The constructor `usm_allocator()` is called, which initializes the `_impl` member with the default SYCL queue obtained from `dpct::get_default_queue()`.
- **Output**: The function does not return any value as it is a constructor.
- **See also**: [`dpct::detail::deprecated::usm_allocator`](#deprecatedusm_allocator)  (Data Structure)


---
#### usm\_allocator::\~usm\_allocator<!-- {{#callable:dpct::detail::deprecated::usm_allocator::~usm_allocator}} -->
The destructor `~usm_allocator` is a default destructor for the `usm_allocator` class template, which does not perform any specific operations.
- **Inputs**: None
- **Control Flow**:
    - The destructor `~usm_allocator` is defined as an empty function, indicating that it does not perform any specific cleanup or resource deallocation tasks.
- **Output**: There is no output from this destructor as it is empty and does not perform any operations.
- **See also**: [`dpct::detail::deprecated::usm_allocator`](#deprecatedusm_allocator)  (Data Structure)


---
#### usm\_allocator::usm\_allocator<!-- {{#callable:dpct::detail::deprecated::usm_allocator::usm_allocator}} -->
The `usm_allocator` function is a constructor for the `usm_allocator` class that initializes a new instance by copying or moving the implementation details from another `usm_allocator` instance.
- **Inputs**:
    - `other`: A reference to another `usm_allocator` instance from which the implementation details are copied or moved.
- **Control Flow**:
    - The copy constructor initializes the `_impl` member by copying the `_impl` from the `other` instance.
    - The move constructor initializes the `_impl` member by moving the `_impl` from the `other` instance using `std::move`.
- **Output**: A new `usm_allocator` instance with its `_impl` member initialized from another instance.
- **See also**: [`dpct::detail::deprecated::usm_allocator`](#deprecatedusm_allocator)  (Data Structure)


---
#### usm\_allocator::usm\_allocator<!-- {{#callable:dpct::detail::deprecated::usm_allocator::usm_allocator}} -->
The `usm_allocator` move constructor initializes a new `usm_allocator` object by transferring ownership of the internal `_impl` allocator from another `usm_allocator` object.
- **Inputs**:
    - `other`: An rvalue reference to another `usm_allocator` object from which the internal `_impl` allocator will be moved.
- **Control Flow**:
    - The constructor is called with an rvalue reference to another `usm_allocator` object.
    - The internal `_impl` allocator of the current object is initialized by moving the `_impl` allocator from the `other` object using `std::move`.
- **Output**: A new `usm_allocator` object with the internal `_impl` allocator moved from the `other` object.
- **See also**: [`dpct::detail::deprecated::usm_allocator`](#deprecatedusm_allocator)  (Data Structure)


---
#### usm\_allocator::address<!-- {{#callable:dpct::detail::deprecated::usm_allocator::address}} -->
The `address` function returns the memory address of a given reference or constant reference.
- **Inputs**:
    - `r`: A reference to a value of type `value_type`.
    - `r`: A constant reference to a value of type `value_type`.
- **Control Flow**:
    - The function takes a reference `r` as input.
    - It returns the address of the reference using the address-of operator `&`.
- **Output**: A pointer to the input reference, which is of type `pointer` for non-const references and `const_pointer` for const references.
- **See also**: [`dpct::detail::deprecated::usm_allocator`](#deprecatedusm_allocator)  (Data Structure)


---
#### usm\_allocator::address<!-- {{#callable:dpct::detail::deprecated::usm_allocator::address}} -->
The `address` function returns the memory address of a given constant reference to a value of type `T`.
- **Inputs**:
    - `r`: A constant reference to a value of type `T` for which the memory address is to be obtained.
- **Control Flow**:
    - The function takes a constant reference `r` as input.
    - It returns the address of `r` using the address-of operator `&`.
- **Output**: A `const_pointer`, which is the memory address of the input constant reference `r`.
- **See also**: [`dpct::detail::deprecated::usm_allocator`](#deprecatedusm_allocator)  (Data Structure)


---
#### usm\_allocator::allocate<!-- {{#callable:dpct::detail::deprecated::usm_allocator::allocate}} -->
The `allocate` function allocates memory using the allocator traits of the `usm_allocator` class.
- **Inputs**:
    - `cnt`: The number of elements to allocate.
    - `hint`: An optional pointer that serves as a hint for the allocation, defaulting to `nullptr`.
- **Control Flow**:
    - The function calls `std::allocator_traits<Alloc>::allocate` with the internal allocator `_impl`, the count `cnt`, and the hint `hint`.
- **Output**: Returns a pointer to the allocated memory of type `pointer`.
- **See also**: [`dpct::detail::deprecated::usm_allocator`](#deprecatedusm_allocator)  (Data Structure)


---
#### usm\_allocator::deallocate<!-- {{#callable:dpct::detail::deprecated::usm_allocator::deallocate}} -->
The `deallocate` function releases memory previously allocated by the `usm_allocator` using the specified pointer and count.
- **Inputs**:
    - `p`: A pointer to the memory block that needs to be deallocated.
    - `cnt`: The number of elements to be deallocated from the memory block pointed to by `p`.
- **Control Flow**:
    - The function calls `std::allocator_traits<Alloc>::deallocate` with `_impl`, `p`, and `cnt` as arguments to perform the deallocation.
- **Output**: The function does not return any value.
- **See also**: [`dpct::detail::deprecated::usm_allocator`](#deprecatedusm_allocator)  (Data Structure)


---
#### usm\_allocator::max\_size<!-- {{#callable:dpct::detail::deprecated::usm_allocator::max_size}} -->
The `max_size` function returns the maximum number of elements that the `usm_allocator` can allocate.
- **Inputs**: None
- **Control Flow**:
    - The function calls `std::allocator_traits<Alloc>::max_size(_impl)` to determine the maximum size.
    - It returns the result of this call.
- **Output**: The function returns a `size_type` value representing the maximum number of elements that can be allocated by the allocator.
- **See also**: [`dpct::detail::deprecated::usm_allocator`](#deprecatedusm_allocator)  (Data Structure)


---
#### usm\_allocator::operator==<!-- {{#callable:dpct::detail::deprecated::usm_allocator::operator==}} -->
The `operator==` function checks if two `usm_allocator` objects are equal by comparing their internal `_impl` members.
- **Inputs**:
    - `other`: A reference to another `usm_allocator` object to compare with the current object.
- **Control Flow**:
    - The function compares the `_impl` member of the current `usm_allocator` object with the `_impl` member of the `other` `usm_allocator` object using the equality operator `==`.
- **Output**: A boolean value indicating whether the two `usm_allocator` objects are equal (true) or not (false).
- **See also**: [`dpct::detail::deprecated::usm_allocator`](#deprecatedusm_allocator)  (Data Structure)


---
#### usm\_allocator::operator\!=<!-- {{#callable:dpct::detail::deprecated::usm_allocator::operator!=}} -->
The `operator!=` function checks if two `usm_allocator` objects are not equal by comparing their internal `_impl` members.
- **Inputs**:
    - `other`: A reference to another `usm_allocator` object to compare against the current object.
- **Control Flow**:
    - The function compares the `_impl` member of the current `usm_allocator` object with the `_impl` member of the `other` `usm_allocator` object using the `!=` operator.
    - If the `_impl` members are not equal, the function returns `true`; otherwise, it returns `false`.
- **Output**: A boolean value indicating whether the two `usm_allocator` objects are not equal.
- **See also**: [`dpct::detail::deprecated::usm_allocator`](#deprecatedusm_allocator)  (Data Structure)



---
### rebind<!-- {{#data_structure:dpct::detail::deprecated::usm_allocator::rebind}} -->
- **Type**: `struct`
- **Members**:
    - `other`: Defines a type alias for `usm_allocator<U, AllocKind>`.
- **Description**: The `rebind` struct template is a utility within the context of allocators, specifically designed to facilitate the creation of an allocator for a different type. It contains a single member, `other`, which is a type alias for `usm_allocator<U, AllocKind>`. This allows for the redefinition of an allocator to allocate objects of a different type `U` while maintaining the same allocation kind `AllocKind`. This pattern is commonly used in C++ to support allocator-aware containers that need to allocate different types of objects.


---
### vectorized\_binary<!-- {{#data_structure:dpct::detail::vectorized_binary}} -->
- **Type**: `class`
- **Description**: The `vectorized_binary` class template is designed to perform element-wise binary operations on two vector-like objects of type `VecT` using a specified binary operation. It provides an overloaded `operator()` that takes two vectors and a binary operation, applying the operation to each corresponding pair of elements from the vectors and returning a new vector with the results. This class is useful for vectorized computations where operations need to be applied across elements of two vectors in a parallel or efficient manner.
- **Member Functions**:
    - [`dpct::detail::vectorized_binary::operator()`](#vectorized_binaryoperator())

**Methods**

---
#### vectorized\_binary::operator\(\)<!-- {{#callable:dpct::detail::vectorized_binary::operator()}} -->
The `operator()` function applies a binary operation element-wise to two input vectors and returns the resulting vector.
- **Inputs**:
    - `a`: A vector of type `VecT` representing the first operand in the binary operation.
    - `b`: A vector of type `VecT` representing the second operand in the binary operation.
    - `binary_op`: A binary operation function or functor that takes two elements of type `VecT` and returns a result of the same type.
- **Control Flow**:
    - Initialize an empty vector `v4` of type `VecT` to store the result.
    - Iterate over each element index `i` of the vector `v4`.
    - For each index `i`, apply the `binary_op` to the corresponding elements of vectors `a` and `b`, and store the result in `v4[i]`.
    - Return the vector `v4` containing the results of the binary operations.
- **Output**: A vector of type `VecT` containing the results of applying the binary operation to each pair of elements from the input vectors `a` and `b`.
- **See also**: [`dpct::detail::vectorized_binary`](#detailvectorized_binary)  (Data Structure)



---
### sub\_sat<!-- {{#data_structure:dpct::sub_sat}} -->
- **Type**: `struct`
- **Description**: The `sub_sat` struct is a simple functor that provides an operator to perform saturated subtraction using the SYCL `sub_sat` function. It is a template struct that can operate on any type `T` that supports the `sub_sat` operation, allowing for the subtraction of two values `x` and `y` with saturation, meaning the result is clamped to the minimum or maximum value representable by the type if it would otherwise overflow or underflow.
- **Member Functions**:
    - [`dpct::sub_sat::operator()`](#sub_satoperator())

**Methods**

---
#### sub\_sat::operator\(\)<!-- {{#callable:dpct::sub_sat::operator()}} -->
The `operator()` function performs a saturated subtraction of two values using SYCL's `sub_sat` function.
- **Inputs**:
    - `x`: The first operand of type T for the saturated subtraction.
    - `y`: The second operand of type T for the saturated subtraction.
- **Control Flow**:
    - The function takes two parameters, `x` and `y`, both of type T.
    - It calls the `sycl::sub_sat` function with `x` and `y` as arguments.
    - The result of `sycl::sub_sat(x, y)` is returned.
- **Output**: The result of the saturated subtraction of `x` and `y`, which is of the same type T.
- **See also**: [`dpct::sub_sat`](#dpctsub_sat)  (Data Structure)



---
### device\_memory<!-- {{#data_structure:dpct::detail::device_memory}} -->
- **Type**: `class`
- **Members**:
    - `_size`: Stores the size of the memory in bytes.
    - `_range`: Represents the range of the memory in terms of dimensions.
    - `_reference`: Indicates whether the memory is a reference to an existing memory block.
    - `_host_ptr`: Pointer to the host memory.
    - `_device_ptr`: Pointer to the device memory.
- **Description**: The `device_memory` class template is a versatile data structure designed to manage memory allocation and access for device memory in SYCL applications. It supports different memory regions such as global, constant, and shared memory, and can handle multi-dimensional memory allocations. The class provides constructors for initializing memory with ranges and initializer lists, and includes methods for memory initialization, assignment, and access. It ensures proper memory management by freeing allocated memory upon destruction and supports both host and device memory operations.
- **Member Functions**:
    - [`dpct::detail::device_memory::device_memory`](#device_memorydevice_memory)
    - [`dpct::detail::device_memory::device_memory`](#device_memorydevice_memory)
    - [`dpct::detail::device_memory::device_memory`](#device_memorydevice_memory)
    - [`dpct::detail::device_memory::device_memory`](#device_memorydevice_memory)
    - [`dpct::detail::device_memory::device_memory`](#device_memorydevice_memory)
    - [`dpct::detail::device_memory::~device_memory`](#device_memorydevice_memory)
    - [`dpct::detail::device_memory::init`](#device_memoryinit)
    - [`dpct::detail::device_memory::init`](#device_memoryinit)
    - [`dpct::detail::device_memory::assign`](#device_memoryassign)
    - [`dpct::detail::device_memory::get_ptr`](#device_memoryget_ptr)
    - [`dpct::detail::device_memory::get_ptr`](#device_memoryget_ptr)
    - [`dpct::detail::device_memory::get_size`](#device_memoryget_size)
    - [`dpct::detail::device_memory::operator[]`](llama.cpp/ggml/src/ggml-sycl/dpct/helper.hpp#callable:dpct::detail::device_memory::operator[])
    - [`dpct::detail::device_memory::get_access`](#device_memoryget_access)
    - [`dpct::detail::device_memory::device_memory`](#device_memorydevice_memory)
    - [`dpct::detail::device_memory::allocate_device`](#device_memoryallocate_device)

**Methods**

---
#### device\_memory::device\_memory<!-- {{#callable:dpct::detail::device_memory::device_memory}} -->
The `device_memory` default constructor initializes a `device_memory` object with a default range of size 1.
- **Inputs**: None
- **Control Flow**:
    - The constructor calls another constructor of the same class, `device_memory`, with a `sycl::range` of size 1 as the argument.
    - This effectively initializes the `device_memory` object with a default range of size 1.
- **Output**: A `device_memory` object initialized with a default range of size 1.
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::device\_memory<!-- {{#callable:dpct::detail::device_memory::device_memory}} -->
The `device_memory` constructor initializes a 1-D device memory object with a given range and an initializer list, allocating and copying data to host memory.
- **Inputs**:
    - `in_range`: A `sycl::range<Dimension>` object specifying the size of the memory to be allocated.
    - `init_list`: An `std::initializer_list<value_t>` containing initial values to be copied into the allocated memory.
- **Control Flow**:
    - The constructor is called with a range and an initializer list as arguments.
    - It asserts that the size of the initializer list does not exceed the size of the range.
    - Memory is allocated on the host using `std::malloc` for the specified size.
    - The allocated memory is initialized to zero using `std::memset`.
    - The contents of the initializer list are copied into the allocated memory using `std::memcpy`.
- **Output**: The function does not return a value; it initializes the internal state of the `device_memory` object.
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::device\_memory<!-- {{#callable:dpct::detail::device_memory::device_memory}} -->
The `device_memory` constructor initializes a 2-D device memory object using a given range and a nested initializer list.
- **Inputs**:
    - `in_range`: A `sycl::range<2>` object specifying the dimensions of the 2-D memory to be allocated.
    - `init_list`: A nested initializer list of type `std::initializer_list<std::initializer_list<value_t>>` containing the initial values for the 2-D memory.
- **Control Flow**:
    - The constructor is enabled only if the template parameter `D` is equal to 2, ensuring it is used for 2-D memory initialization.
    - The constructor first calls another `device_memory` constructor with `in_range` to initialize the base class.
    - An assertion checks that the size of `init_list` does not exceed the first dimension of `in_range`.
    - Memory is allocated for the host pointer `_host_ptr` using `std::malloc` with the size `_size`.
    - The allocated memory is initialized to zero using `std::memset`.
    - A temporary pointer `tmp_data` is set to `_host_ptr` to iterate over the initializer list.
    - For each sublist in `init_list`, an assertion checks that its size does not exceed the second dimension of `in_range`.
    - The contents of each sublist are copied into the allocated memory using `std::memcpy`, and `tmp_data` is incremented by the second dimension of `in_range` to move to the next row.
- **Output**: The constructor does not return a value; it initializes the 2-D device memory object with the specified range and initial values.
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::device\_memory<!-- {{#callable:dpct::detail::device_memory::device_memory}} -->
The `device_memory` constructor initializes a device memory object with a specified range and ensures that memory management and device management singletons are instantiated.
- **Inputs**:
    - `range_in`: A `sycl::range<Dimension>` object representing the dimensions of the memory to be allocated.
- **Control Flow**:
    - The constructor initializes the `_size` member variable by calculating the total size of the memory in bytes using the provided range and the size of the template type `T`.
    - The `_range` member variable is set to the provided `range_in`.
    - The `_reference`, `_host_ptr`, and `_device_ptr` member variables are initialized to `false`, `nullptr`, and `nullptr` respectively.
    - A static assertion checks that the `Memory` template parameter is one of `global`, `constant`, or `shared`.
    - The constructor ensures that the `mem_mgr` and `dev_mgr` singleton instances are created by calling their `instance()` methods.
- **Output**: The constructor does not return a value; it initializes the state of the `device_memory` object.
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::device\_memory<!-- {{#callable:dpct::detail::device_memory::device_memory}} -->
The `device_memory` constructor template initializes a `device_memory` object using a variadic list of arguments to define its range.
- **Inputs**:
    - `Args... Arguments`: A variadic list of arguments used to construct a `sycl::range<Dimension>` object, which defines the dimensions of the `device_memory` object.
- **Control Flow**:
    - The constructor takes a variadic list of arguments `Args... Arguments`.
    - It constructs a `sycl::range<Dimension>` object using these arguments.
    - It then calls another `device_memory` constructor with this `sycl::range<Dimension>` object.
- **Output**: A `device_memory` object is initialized with the specified range.
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::\~device\_memory<!-- {{#callable:dpct::detail::device_memory::~device_memory}} -->
The destructor `~device_memory` releases allocated device and host memory if they are not referenced.
- **Inputs**: None
- **Control Flow**:
    - Check if `_device_ptr` is not null and `_reference` is false, then free the device memory using `dpct::dpct_free`.
    - Check if `_host_ptr` is not null, then free the host memory using `std::free`.
- **Output**: The function does not return any value.
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::init<!-- {{#callable:dpct::detail::device_memory::init}} -->
The [`init`](#device_memoryinit) function initializes device memory by allocating it with the default SYCL queue and copying any initial values from host memory to device memory if they exist.
- **Inputs**: None
- **Control Flow**:
    - The function calls another [`init`](#device_memoryinit) function with the default SYCL queue as an argument.
    - The [`init`](#device_memoryinit) function with the queue checks if the device pointer is already allocated; if so, it returns immediately.
    - If the size of the memory is zero, it also returns immediately.
    - Otherwise, it calls `allocate_device` to allocate memory on the device using the specified queue.
    - If there is a host pointer with initial values, it copies these values to the device memory using `dpct_memcpy`.
- **Output**: The function does not return any value; it initializes the device memory state.
- **Functions called**:
    - [`dpct::detail::device_memory::init`](#device_memoryinit)
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::init<!-- {{#callable:dpct::detail::device_memory::init}} -->
The `init` function initializes device memory by allocating it on the specified SYCL queue and optionally copying data from host memory if available.
- **Inputs**:
    - `q`: A reference to a `sycl::queue` object, which specifies the SYCL queue to be used for memory operations.
- **Control Flow**:
    - Check if `_device_ptr` is already initialized; if so, return immediately.
    - Check if `_size` is zero; if so, return immediately.
    - Call `allocate_device(q)` to allocate memory on the device using the specified queue.
    - If `_host_ptr` is not null, copy data from `_host_ptr` to `_device_ptr` using `detail::dpct_memcpy` with the direction `host_to_device`.
- **Output**: The function does not return any value; it modifies the state of the `device_memory` object by initializing its device memory.
- **Functions called**:
    - [`dpct::detail::device_memory::allocate_device`](#device_memoryallocate_device)
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::assign<!-- {{#callable:dpct::detail::device_memory::assign}} -->
The `assign` method reinitializes a `device_memory` object with a new memory source and size.
- **Inputs**:
    - `src`: A pointer to the source memory of type `value_t` that will be assigned to the `device_memory` object.
    - `size`: The size of the memory to be assigned, in bytes.
- **Control Flow**:
    - The method first calls the destructor of the current `device_memory` object to release any existing resources.
    - It then uses placement new to reinitialize the `device_memory` object with the new source memory and size.
- **Output**: The method does not return any value; it modifies the `device_memory` object in place.
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::get\_ptr<!-- {{#callable:dpct::detail::device_memory::get_ptr}} -->
The [`get_ptr`](#device_memoryget_ptr) function retrieves the memory pointer of a device memory object, initializing the memory if necessary, and returns a virtual pointer when USM is not used or a device pointer when USM is used.
- **Inputs**:
    - `q`: A `sycl::queue` reference used to initialize the memory if it hasn't been initialized yet.
- **Control Flow**:
    - The function first calls the `init` method with the provided queue `q` to ensure the memory is initialized.
    - If the memory is not already allocated, the `init` method will allocate the device memory and copy any host data to the device.
    - Finally, the function returns the `_device_ptr`, which is the pointer to the device memory.
- **Output**: A pointer of type `value_t*` to the device memory, which can be a virtual pointer or a device pointer depending on whether USM is used.
- **Functions called**:
    - [`dpct::detail::device_memory::get_ptr`](#device_memoryget_ptr)
    - [`dpct::get_default_queue`](#dpctget_default_queue)
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::get\_ptr<!-- {{#callable:dpct::detail::device_memory::get_ptr}} -->
The `get_ptr` function initializes the device memory using a specified SYCL queue and returns a pointer to the device memory.
- **Inputs**:
    - `q`: A reference to a SYCL queue object used for initializing the device memory.
- **Control Flow**:
    - The function calls the [`init`](#device_memoryinit) method with the provided SYCL queue `q` to ensure the device memory is initialized.
    - After initialization, the function returns the `_device_ptr`, which is a pointer to the device memory.
- **Output**: A pointer to the device memory (`_device_ptr`).
- **Functions called**:
    - [`dpct::detail::device_memory::init`](#device_memoryinit)
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::get\_size<!-- {{#callable:dpct::detail::device_memory::get_size}} -->
The `get_size` function returns the size of the device memory object in bytes.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `_size`.
- **Output**: The function returns a `size_t` value representing the size of the device memory object in bytes.
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::operator\[\]<!-- {{#callable:dpct::detail::device_memory::operator[]}} -->
The `operator[]` function provides access to elements of a 1-dimensional device memory array by index.
- **Inputs**:
    - `index`: A `size_t` representing the index of the element to access in the 1-dimensional device memory array.
- **Control Flow**:
    - The function checks if the template parameter `D` is equal to 1 using `std::enable_if` to ensure it is only used for 1-dimensional arrays.
    - The `init()` method is called to ensure the device memory is initialized before accessing it.
    - The function returns a reference to the element at the specified `index` in the `_device_ptr` array.
- **Output**: A reference to the element of type `T` at the specified index in the device memory array.
- **Functions called**:
    - [`dpct::detail::device_memory::init`](#device_memoryinit)
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::get\_access<!-- {{#callable:dpct::detail::device_memory::get_access}} -->
The `get_access` function returns a `dpct_accessor_t` object for device memory with dimensions greater than 1.
- **Inputs**:
    - `cgh`: A `sycl::handler` object, which is marked as `[[maybe_unused]]`, indicating it is not used in the function body.
- **Control Flow**:
    - The function is templated with a default template parameter `D` set to `Dimension`.
    - It uses `std::enable_if` to ensure the function is only instantiated when `D` is not equal to 1.
    - The function returns a `dpct_accessor_t` object, constructed with the device pointer `_device_ptr` cast to type `T*` and the range `_range`.
- **Output**: A `dpct_accessor_t` object, which is an accessor for device memory with dimension information.
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::device\_memory<!-- {{#callable:dpct::detail::device_memory::device_memory}} -->
The `device_memory` constructor initializes a `device_memory` object with a given memory pointer and size, setting it as a reference to existing device memory.
- **Inputs**:
    - `memory_ptr`: A pointer to the existing device memory that the `device_memory` object will reference.
    - `size`: The size of the memory block in bytes that the `device_memory` object will manage.
- **Control Flow**:
    - The constructor initializes the `_size` member with the provided size.
    - It calculates the `_range` by dividing the size by the size of the template type `T`.
    - The `_reference` member is set to `true`, indicating that this object is a reference to existing memory rather than owning the memory.
    - The `_device_ptr` is set to the provided `memory_ptr`, pointing to the existing device memory.
- **Output**: The function does not return any value; it initializes the `device_memory` object.
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)


---
#### device\_memory::allocate\_device<!-- {{#callable:dpct::detail::device_memory::allocate_device}} -->
The `allocate_device` function allocates memory on a SYCL device based on the specified memory region type (shared, constant, or global).
- **Inputs**:
    - `q`: A reference to a `sycl::queue` object, which represents the queue to be used for memory allocation on the device.
- **Control Flow**:
    - Check if the preprocessor directive `DPCT_USM_LEVEL_NONE` is not defined.
    - If the `Memory` type is `shared`, allocate shared memory using `sycl::malloc_shared` and assign it to `_device_ptr`.
    - If the `Memory` type is `constant` and `SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY` is defined, allocate device memory with read-only property using `sycl::malloc_device` and assign it to `_device_ptr`.
    - If none of the above conditions are met, allocate global device memory using `detail::dpct_malloc` and assign it to `_device_ptr`.
- **Output**: The function does not return a value; it assigns the allocated memory pointer to the `_device_ptr` member variable.
- **See also**: [`dpct::detail::device_memory`](#detaildevice_memory)  (Data Structure)



# Functions

---
### get\_device\_type\_name<!-- {{#callable:get_device_type_name}} -->
The `get_device_type_name` function returns a string representation of the type of a given SYCL device.
- **Inputs**:
    - `Device`: A constant reference to a `sycl::device` object from which the device type information is retrieved.
- **Control Flow**:
    - Retrieve the device type information from the `Device` object using `get_info<sycl::info::device::device_type>()`.
    - Use a switch statement to match the retrieved device type against known SYCL device types: `cpu`, `gpu`, `host`, and `accelerator`.
    - Return the corresponding string representation for each matched device type.
    - If the device type does not match any known types, return "unknown".
- **Output**: A `std::string` representing the type of the device, such as "cpu", "gpu", "host", "acc", or "unknown".


---
### get\_device\_backend\_and\_type<!-- {{#callable:get_device_backend_and_type}} -->
The function `get_device_backend_and_type` returns a string representation of a SYCL device's backend and type.
- **Inputs**:
    - `device`: A `sycl::device` object representing the device whose backend and type information is to be retrieved.
- **Control Flow**:
    - Initialize a `std::stringstream` object named `device_type`.
    - Retrieve the backend of the device using `device.get_backend()` and store it in a `sycl::backend` variable named `backend`.
    - Append the backend and the result of `get_device_type_name(device)` to the `device_type` stream, separated by a colon.
- **Output**: A `std::string` containing the backend and type of the device, formatted as "backend:type".
- **Functions called**:
    - [`get_device_type_name`](#get_device_type_name)


---
### destroy\_event<!-- {{#callable:dpct::destroy_event}} -->
The `destroy_event` function deallocates memory for a given SYCL event pointer.
- **Inputs**:
    - `event`: A pointer to a SYCL event (`sycl::event *`) that needs to be destroyed.
- **Control Flow**:
    - The function takes a single argument, `event`, which is a pointer to a SYCL event.
    - It uses the `delete` operator to deallocate the memory associated with the `event` pointer.
- **Output**: The function does not return any value.


---
### get\_tid<!-- {{#callable:dpct::get_tid}} -->
The `get_tid` function retrieves the current thread ID for the executing thread on either Linux or Windows platforms.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the code is being compiled on a Linux platform using the `__linux__` preprocessor directive.
    - If on Linux, it calls the `syscall` function with `SYS_gettid` to get the thread ID.
    - If the code is being compiled on a Windows platform (checked using `_WIN64`), it calls `GetCurrentThreadId` to get the thread ID.
    - If neither Linux nor Windows is detected, a preprocessor error is raised indicating that only Windows and Linux are supported.
- **Output**: The function returns an `unsigned int` representing the current thread ID.


---
### get\_version<!-- {{#callable:dpct::detail::get_version}} -->
The `get_version` function extracts the major and minor version numbers from a SYCL device's version string.
- **Inputs**:
    - `dev`: A `sycl::device` object representing the device from which the version information is to be retrieved.
    - `major`: An integer reference where the major version number will be stored.
    - `minor`: An integer reference where the minor version number will be stored.
- **Control Flow**:
    - Retrieve the version string from the device using `dev.get_info<sycl::info::device::version>()`.
    - Initialize a string index `i` to 0 and iterate through the version string until a digit is found, indicating the start of the version number.
    - Convert the substring starting at the first digit to an integer and store it in `major`.
    - Continue iterating through the string until a period `.` is found, indicating the separator between major and minor version numbers.
    - If a period is found, increment the index and convert the following substring to an integer to store in `minor`.
    - If no period is found, set `minor` to 0, indicating a version format without a minor version.
- **Output**: The function outputs the major and minor version numbers through the reference parameters `major` and `minor`.


---
### get\_major\_version<!-- {{#callable:dpct::get_major_version}} -->
The `get_major_version` function retrieves the major version number of a given SYCL device.
- **Inputs**:
    - `dev`: A constant reference to a `sycl::device` object representing the device whose major version is to be retrieved.
- **Control Flow**:
    - Declare two integer variables, `major` and `minor`.
    - Call the `detail::get_version` function with `dev`, `major`, and `minor` as arguments to populate the `major` and `minor` version numbers of the device.
    - Return the `major` version number.
- **Output**: An integer representing the major version number of the specified SYCL device.


---
### get\_minor\_version<!-- {{#callable:dpct::get_minor_version}} -->
The `get_minor_version` function retrieves the minor version number of a given SYCL device.
- **Inputs**:
    - `dev`: A constant reference to a `sycl::device` object representing the device whose minor version is to be retrieved.
- **Control Flow**:
    - Declare two integer variables, `major` and `minor`.
    - Call the `detail::get_version` function with `dev`, `major`, and `minor` as arguments to populate the version numbers.
    - Return the `minor` version number.
- **Output**: An integer representing the minor version number of the specified SYCL device.


---
### get\_device\_info<!-- {{#callable:dpct::get_device_info}} -->
The `get_device_info` function populates a `device_info` object with various properties of a given SYCL device.
- **Inputs**:
    - `out`: A reference to a `device_info` object that will be populated with the device's properties.
    - `dev`: A constant reference to a `sycl::device` object from which the device properties will be retrieved.
- **Control Flow**:
    - Initialize a `device_info` object `prop` to store device properties.
    - Retrieve and set the device name using `dev.get_info<sycl::info::device::name>()`.
    - Call `detail::get_version` to get the major and minor version numbers of the device and set them in `prop`.
    - Set the maximum work item sizes using a conditional compilation directive to handle different SYCL versions.
    - Set the host unified memory support flag using `dev.has(sycl::aspect::usm_host_allocations)`.
    - Retrieve and set various device properties such as max clock frequency, max compute units, max work group size, global memory size, local memory size, and max memory allocation size using `dev.get_info` calls.
    - For Intel-specific extensions, check for support and set properties like memory clock rate, memory bus width, device ID, and UUID if available.
    - Determine the maximum sub-group size by iterating over the sub-group sizes retrieved from the device.
    - Set additional properties like max work items per compute unit, max ND range size, max register size per work group, and global memory cache size.
    - Assign the populated `prop` object to the `out` parameter.
- **Output**: The function does not return a value but modifies the `out` parameter to contain the device's properties.


---
### get\_default\_queue<!-- {{#callable:dpct::get_default_queue}} -->
The `get_default_queue` function retrieves the default SYCL queue for the current device managed by the device manager singleton.
- **Inputs**: None
- **Control Flow**:
    - The function calls `dev_mgr::instance()` to get the singleton instance of the device manager.
    - It then calls `current_device()` on the device manager instance to get the current device.
    - Finally, it calls `default_queue()` on the current device to retrieve the default SYCL queue.
- **Output**: A reference to the default SYCL queue for the current device.


---
### get\_pointer\_attribute<!-- {{#callable:dpct::detail::get_pointer_attribute}} -->
The `get_pointer_attribute` function determines the access attribute of a pointer based on its allocation type in a SYCL context.
- **Inputs**:
    - `q`: A reference to a `sycl::queue` object, which provides the context for determining the pointer type.
    - `ptr`: A constant void pointer to the memory location whose access attribute is to be determined.
- **Control Flow**:
    - The function uses a switch statement to evaluate the result of `sycl::get_pointer_type(ptr, q.get_context())`.
    - If the pointer type is `sycl::usm::alloc::unknown`, it returns `pointer_access_attribute::host_only`.
    - If the pointer type is `sycl::usm::alloc::device`, it returns `pointer_access_attribute::device_only`.
    - If the pointer type is `sycl::usm::alloc::shared` or `sycl::usm::alloc::host`, it returns `pointer_access_attribute::host_device`.
- **Output**: Returns a `pointer_access_attribute` enum value indicating the access attribute of the pointer.


---
### get\_type\_combination\_id<!-- {{#callable:dpct::detail::get_type_combination_id}} -->
The [`get_type_combination_id`](#detailget_type_combination_id) function computes a unique identifier for a combination of types, represented by `library_data_t` values, by recursively shifting and combining these values into a 64-bit integer.
- **Inputs**:
    - `FirstT`: The type of the first argument, which must be `library_data_t`.
    - `FirstVal`: The first value of type `FirstT`, representing a type in the `library_data_t` enumeration.
    - `RestT`: A variadic template parameter representing the types of the remaining arguments, which must also be `library_data_t`.
    - `RestVal`: The remaining values of types `RestT`, each representing a type in the `library_data_t` enumeration.
- **Control Flow**:
    - The function begins by asserting that the size of `library_data_t` does not exceed the maximum value of an unsigned char.
    - It asserts that the number of additional parameters (`RestT`) does not exceed 8.
    - It asserts that the type of `FirstT` is `library_data_t`.
    - The function calls itself recursively with the rest of the values (`RestVal...`), shifting the result left by 8 bits.
    - It combines the shifted result with the current `FirstVal` cast to `std::uint64_t` using a bitwise OR operation.
- **Output**: A `std::uint64_t` value representing a unique identifier for the combination of types specified by the input arguments.
- **Functions called**:
    - [`dpct::detail::get_type_combination_id`](#detailget_type_combination_id)


---
### dpct\_malloc<!-- {{#callable:dpct::detail::dpct_malloc}} -->
The [`dpct_malloc`](#detaildpct_malloc) function allocates memory for a 3D array on a SYCL device and returns a pointer to the allocated memory.
- **Inputs**:
    - `pitch`: A reference to a `size_t` variable where the function will store the computed pitch value, which is the aligned width of the memory allocation.
    - `x`: The width of the 3D array in elements.
    - `y`: The height of the 3D array in elements.
    - `z`: The depth of the 3D array in elements.
    - `q`: A reference to a `sycl::queue` object that represents the SYCL queue where the memory allocation will be performed.
- **Control Flow**:
    - The function first calculates the pitch by aligning the width `x` using the `PITCH_DEFAULT_ALIGN` macro.
    - It then calls another [`dpct_malloc`](#detaildpct_malloc) function, passing the total size of the memory to be allocated (`pitch * y * z`) and the SYCL queue `q`.
    - The result of the [`dpct_malloc`](#detaildpct_malloc) call, which is a pointer to the allocated memory, is returned.
- **Output**: A pointer to the allocated memory on the SYCL device.
- **Functions called**:
    - [`dpct::detail::dpct_malloc`](#detaildpct_malloc)


---
### dpct\_memset<!-- {{#callable:dpct::detail::dpct_memset}} -->
The [`dpct_memset`](#detaildpct_memset) function sets a specified value to a 2D pitched memory region on a SYCL device queue.
- **Inputs**:
    - `q`: A reference to a SYCL queue where the operation is performed.
    - `ptr`: A pointer to the starting address of the memory region to be set.
    - `pitch`: The pitch size in bytes, including padding, of the memory region.
    - `val`: The value to be set in the memory region.
    - `x`: The width of the memory region in terms of number of elements.
    - `y`: The height of the memory region in terms of number of elements.
- **Control Flow**:
    - The function constructs a [`pitched_data`](#pitched_datapitched_data) object using the provided `ptr`, `pitch`, `x`, and a fixed value of 1 for the depth.
    - It then calls another overloaded [`dpct_memset`](#detaildpct_memset) function, passing the queue `q`, the constructed [`pitched_data`](#pitched_datapitched_data) object, the value `val`, and a `sycl::range<3>` object representing the dimensions of the memory region (x, y, 1).
- **Output**: A vector of SYCL events representing the memset operations performed on the memory region.
- **Functions called**:
    - [`dpct::detail::dpct_memset`](#detaildpct_memset)
    - [`dpct::pitched_data::pitched_data`](#pitched_datapitched_data)


---
### deduce\_memcpy\_direction<!-- {{#callable:dpct::detail::deduce_memcpy_direction}} -->
The `deduce_memcpy_direction` function determines the direction of memory copy operations based on given pointers and a specified direction, defaulting to automatic deduction if necessary.
- **Inputs**:
    - `q`: A reference to a `sycl::queue` object, representing the SYCL queue in which the operation is performed.
    - `to_ptr`: A pointer to the destination memory location where data is to be copied.
    - `from_ptr`: A pointer to the source memory location from where data is to be copied.
    - `dir`: An enumerated value of type `memcpy_direction` indicating the intended direction of the memory copy operation.
- **Control Flow**:
    - The function begins by checking the value of the `dir` parameter using a switch statement.
    - If `dir` is one of the explicit directions (`host_to_host`, `host_to_device`, `device_to_host`, `device_to_device`), it returns `dir` immediately.
    - If `dir` is `automatic`, the function uses a static direction table to determine the direction based on the attributes of the `to_ptr` and `from_ptr` pointers.
    - The function calls [`get_pointer_attribute`](#detailget_pointer_attribute) to get the attributes of the pointers, which are then used to index into the direction table.
    - If `dir` is not a valid value, the function throws a `std::runtime_error` indicating an invalid direction value.
- **Output**: The function returns a `memcpy_direction` value indicating the determined direction of the memory copy operation.
- **Functions called**:
    - [`dpct::detail::get_pointer_attribute`](#detailget_pointer_attribute)


---
### dpct\_memcpy<!-- {{#callable:dpct::dpct_memcpy}} -->
The [`dpct_memcpy`](#detaildpct_memcpy) function performs a 2D memory copy operation between two pointers with specified pitches and dimensions, using a SYCL queue and a specified direction for the copy.
- **Inputs**:
    - `q`: A reference to a `sycl::queue` object, which represents the SYCL queue where the memory copy operation will be enqueued.
    - `to_ptr`: A pointer to the destination memory location where data will be copied to.
    - `from_ptr`: A pointer to the source memory location from where data will be copied.
    - `to_pitch`: The pitch (or stride) of the destination memory in bytes, representing the number of bytes between consecutive rows.
    - `from_pitch`: The pitch (or stride) of the source memory in bytes, representing the number of bytes between consecutive rows.
    - `x`: The width of the memory region to be copied, in bytes.
    - `y`: The height of the memory region to be copied, in rows.
    - `direction`: An optional `memcpy_direction` enum value indicating the direction of the memory copy (e.g., host to device, device to host, etc.), with a default value of `automatic`.
- **Control Flow**:
    - The function calls another overloaded [`dpct_memcpy`](#detaildpct_memcpy) function, passing the queue, pointers, and dimensions as `sycl::range<3>` and `sycl::id<3>` objects.
    - The `sycl::range<3>` objects are constructed using the provided pitches and dimensions, while the `sycl::id<3>` objects are initialized to zero.
    - The function returns the result of the called [`dpct_memcpy`](#detaildpct_memcpy) function, which is a vector of `sycl::event` objects representing the memory copy operations.
- **Output**: A `std::vector<sycl::event>` containing the events associated with the memory copy operations, which can be used to synchronize or query the status of the operations.
- **Functions called**:
    - [`dpct::detail::dpct_memcpy`](#detaildpct_memcpy)


---
### get\_copy\_range<!-- {{#callable:dpct::get_copy_range}} -->
The `get_copy_range` function calculates the total number of elements to be copied in a 3D memory region based on the given dimensions, slice size, and pitch.
- **Inputs**:
    - `size`: A `sycl::range<3>` object representing the dimensions of the 3D memory region.
    - `slice`: A `size_t` value representing the size of a slice in the 3D memory region.
    - `pitch`: A `size_t` value representing the pitch (or stride) of the memory region in the second dimension.
- **Control Flow**:
    - The function takes three parameters: `size`, `slice`, and `pitch`.
    - It calculates the total number of elements to be copied by multiplying `slice` with the size of the third dimension minus one, adding the product of `pitch` and the size of the second dimension minus one, and finally adding the size of the first dimension.
- **Output**: The function returns a `size_t` value representing the total number of elements to be copied in the 3D memory region.


---
### get\_offset<!-- {{#callable:dpct::get_offset}} -->
The `get_offset` function calculates a linear offset in a 3D array based on a given 3D index and the dimensions of the array slices and pitches.
- **Inputs**:
    - `id`: A `sycl::id<3>` object representing a 3D index within the array.
    - `slice`: A `size_t` representing the size of a slice in the array, typically the product of the pitch and the second dimension size.
    - `pitch`: A `size_t` representing the pitch of the array, which is the number of elements in a row of the array.
- **Control Flow**:
    - The function takes a 3D index `id` and calculates the offset by multiplying the slice size with the third component of the index `id.get(2)`, the pitch with the second component `id.get(1)`, and adding the first component `id.get(0)`.
    - The calculated offset is returned as the result of the function.
- **Output**: A `size_t` representing the linear offset in the 3D array.


---
### dpct\_free<!-- {{#callable:dpct::dpct_free}} -->
The `dpct_free` function deallocates memory pointed to by `ptr` using a specified SYCL queue.
- **Inputs**:
    - `ptr`: A pointer to the memory that needs to be deallocated.
    - `q`: A SYCL queue used for the deallocation operation; defaults to the result of `get_default_queue()` if not provided.
- **Control Flow**:
    - The function calls `detail::dpct_free` with the provided `ptr` and `q` arguments.
    - The `detail::dpct_free` function handles the actual memory deallocation using the SYCL context associated with the queue.
- **Output**: The function does not return any value.
- **Functions called**:
    - [`dpct::get_default_queue`](#dpctget_default_queue)


---
### get\_memory<!-- {{#callable:dpct::detail::get_memory}} -->
The `get_memory` function casts a constant void pointer to a mutable pointer of a specified type.
- **Inputs**:
    - `x`: A constant void pointer that needs to be cast to a different type.
- **Control Flow**:
    - The function takes a constant void pointer `x` as input.
    - It uses `const_cast` to remove the constness of `x`, allowing it to be cast to a mutable pointer.
    - The function then uses `reinterpret_cast` to cast the pointer to a pointer of type `T`.
    - The casted pointer is returned.
- **Output**: A pointer of type `T*` that points to the same memory location as the input pointer `x`, but allows for modification of the data.


---
### get\_value<!-- {{#callable:dpct::get_value}} -->
The `get_value` function retrieves the value pointed to by a device pointer, potentially transferring it from device to host memory if necessary.
- **Inputs**:
    - `s`: A pointer to a value of type `T` that may reside in device memory.
    - `q`: A reference to a SYCL queue used for managing device operations and memory transfers.
- **Control Flow**:
    - The function calls `detail::get_value` with the provided pointer `s` and queue `q`.
    - Inside `detail::get_value`, it checks if the pointer `s` is a device-only pointer using `get_pointer_attribute`.
    - If `s` is a device-only pointer, it performs a memory copy from device to host using `dpct_memcpy`.
    - If `s` is not a device-only pointer, it directly reads the value from the pointer `s`.
- **Output**: The function returns the value of type `T2` (a potentially transformed type of `T`) that `s` points to, either directly or after copying from device to host.


---
### gemm\_impl<!-- {{#callable:dpct::detail::gemm_impl}} -->
The `gemm_impl` function performs a general matrix-matrix multiplication (GEMM) operation using the SYCL queue and oneAPI math library.
- **Inputs**:
    - `q`: A reference to a SYCL queue used for executing the operation.
    - `a_trans`: Specifies whether matrix A should be transposed or not, using the `oneapi::math::transpose` enumeration.
    - `b_trans`: Specifies whether matrix B should be transposed or not, using the `oneapi::math::transpose` enumeration.
    - `m`: The number of rows in the matrix op(A) and the matrix C.
    - `n`: The number of columns in the matrix op(B) and the matrix C.
    - `k`: The number of columns in the matrix op(A) and the number of rows in the matrix op(B).
    - `alpha`: A pointer to a scalar multiplier for the matrix product.
    - `a`: A pointer to the first input matrix A.
    - `lda`: The leading dimension of matrix A.
    - `b`: A pointer to the second input matrix B.
    - `ldb`: The leading dimension of matrix B.
    - `beta`: A pointer to a scalar multiplier for matrix C.
    - `c`: A pointer to the output matrix C.
    - `ldc`: The leading dimension of matrix C.
- **Control Flow**:
    - Retrieve the scalar value of alpha from the pointer using `dpct::get_value` and the SYCL queue.
    - Retrieve the scalar value of beta from the pointer using `dpct::get_value` and the SYCL queue.
    - Convert the input pointers for matrices A, B, and C to their respective types using `get_memory`.
    - Call the `oneapi::math::blas::column_major::gemm` function to perform the matrix multiplication with the specified parameters.
- **Output**: The function does not return a value; it modifies the matrix C in place.


---
### operator\(\)<!-- {{#callable:dpct::detail::operator()}} -->
The `operator()` function applies a binary operation on two vector inputs and returns the result as a vector of the same type.
- **Inputs**:
    - `a`: A vector of type `VecT` representing the first operand.
    - `b`: A vector of type `VecT` representing the second operand.
    - `binary_op`: A binary operation function or functor that takes two `VecT` vectors and returns a result.
- **Control Flow**:
    - The function takes two vectors `a` and `b` and a binary operation `binary_op` as inputs.
    - It applies the `binary_op` to `a` and `b`.
    - The result of the binary operation is then cast to the type `VecT` using the `as<VecT>()` method.
    - The function returns the cast result.
- **Output**: A vector of type `VecT` that is the result of applying the binary operation to the input vectors `a` and `b`.


---
### gemm\_batch\_impl<!-- {{#callable:dpct::detail::gemm_batch_impl}} -->
The `gemm_batch_impl` function performs a batched matrix-matrix multiplication using the SYCL queue and oneAPI math library.
- **Inputs**:
    - `q`: A reference to a SYCL queue where the operation will be executed.
    - `a_trans`: Specifies the operation applied to matrix A (transpose or not).
    - `b_trans`: Specifies the operation applied to matrix B (transpose or not).
    - `m`: The number of rows of the matrix op(A) and of the matrix C.
    - `n`: The number of columns of the matrix op(B) and of the matrix C.
    - `k`: The number of columns of the matrix op(A) and the number of rows of the matrix op(B).
    - `alpha`: A pointer to the scaling factor for the matrix-matrix product.
    - `a`: A pointer to the input matrix A.
    - `lda`: The leading dimension of matrix A.
    - `stride_a`: The stride between the different A matrices in the batch.
    - `b`: A pointer to the input matrix B.
    - `ldb`: The leading dimension of matrix B.
    - `stride_b`: The stride between the different B matrices in the batch.
    - `beta`: A pointer to the scaling factor for matrix C.
    - `c`: A pointer to the input/output matrix C.
    - `ldc`: The leading dimension of matrix C.
    - `stride_c`: The stride between the different C matrices in the batch.
    - `batch_size`: The number of matrix multiply operations to perform in the batch.
- **Control Flow**:
    - Retrieve the value of alpha from the device using `dpct::get_value` and store it in `alpha_value`.
    - Retrieve the value of beta from the device using `dpct::get_value` and store it in `beta_value`.
    - Convert the pointers `a`, `b`, and `c` to their respective data types using `get_memory`.
    - Call `oneapi::math::blas::column_major::gemm_batch` with the appropriate parameters to perform the batched matrix multiplication.
- **Output**: The function does not return a value; it performs operations on the matrices pointed to by the input pointers.


---
### vectorized\_binary<!-- {{#callable:dpct::vectorized_binary}} -->
The `vectorized_binary` function performs a binary operation on two unsigned integers using a specified vector type and binary operation, and returns the result as an unsigned integer.
- **Inputs**:
    - `a`: An unsigned integer representing the first operand.
    - `b`: An unsigned integer representing the second operand.
    - `binary_op`: A binary operation function or functor to be applied to the operands.
- **Control Flow**:
    - Convert the unsigned integer `a` to a SYCL vector of type `VecT`.
    - Convert the unsigned integer `b` to a SYCL vector of type `VecT`.
    - Invoke the `vectorized_binary` operation from the `detail` namespace, passing the converted vectors and the binary operation.
    - Convert the result back to a SYCL vector of type `sycl::vec<unsigned, 1>`.
    - Return the result as an unsigned integer.
- **Output**: An unsigned integer representing the result of the binary operation applied to the inputs.


---
### async\_dpct\_memcpy<!-- {{#callable:dpct::async_dpct_memcpy}} -->
The `async_dpct_memcpy` function performs an asynchronous memory copy operation between two pitched 2D memory regions using a specified SYCL queue and direction.
- **Inputs**:
    - `to_ptr`: A pointer to the destination memory region where data will be copied to.
    - `to_pitch`: The pitch (width in bytes) of the destination memory region.
    - `from_ptr`: A pointer to the source memory region from where data will be copied.
    - `from_pitch`: The pitch (width in bytes) of the source memory region.
    - `x`: The width of the memory region to be copied, in bytes.
    - `y`: The height of the memory region to be copied, in elements.
    - `direction`: The direction of the memory copy, which can be host-to-host, host-to-device, device-to-host, device-to-device, or automatic (default is automatic).
    - `q`: The SYCL queue to be used for the memory copy operation (default is the default queue).
- **Control Flow**:
    - The function calls `detail::dpct_memcpy` with the provided SYCL queue, destination and source pointers, pitches, dimensions, and direction.
    - The `detail::dpct_memcpy` function handles the actual memory copy operation based on the specified direction and queue.
- **Output**: The function does not return any value; it performs the memory copy operation asynchronously.
- **Functions called**:
    - [`dpct::get_default_queue`](#dpctget_default_queue)


---
### select\_device<!-- {{#callable:dpct::select_device}} -->
The `select_device` function sets the current device to the specified device ID and returns that ID.
- **Inputs**:
    - `id`: An unsigned integer representing the ID of the device to be selected.
- **Control Flow**:
    - The function calls `dev_mgr::instance().select_device(id)` to set the current device to the specified ID.
    - The function then returns the provided device ID.
- **Output**: The function returns the same unsigned integer ID that was passed as an argument.


---
### permute\_sub\_group\_by\_xor<!-- {{#callable:dpct::permute_sub_group_by_xor}} -->
The `permute_sub_group_by_xor` function permutes elements within a SYCL sub-group using an XOR operation on the local linear ID and a mask.
- **Inputs**:
    - `g`: A SYCL sub_group object representing the sub-group within which the permutation is performed.
    - `x`: A value of type T that represents the element to be permuted within the sub-group.
    - `mask`: An unsigned integer used as a mask for the XOR operation to determine the target offset for permutation.
    - `logical_sub_group_size`: An optional unsigned integer specifying the logical size of the sub-group, defaulting to 32.
- **Control Flow**:
    - Retrieve the local linear ID of the current work item within the sub-group using `g.get_local_linear_id()`.
    - Calculate the start index of the logical sub-group by dividing the local linear ID by the logical sub-group size and multiplying by the logical sub-group size.
    - Compute the target offset by performing an XOR operation between the local linear ID modulo the logical sub-group size and the mask.
    - Use `sycl::select_from_group` to select the element from the sub-group at the calculated target offset if it is within bounds, otherwise select the element at the current ID.
- **Output**: Returns the permuted element of type T from the sub-group based on the calculated target offset.


---
### dp4a<!-- {{#callable:dpct::dp4a}} -->
The `dp4a` function is a template function that performs a dot product of four-element vectors and adds a scalar, utilizing the `syclcompat::dp4a` function.
- **Inputs**:
    - `T1 a`: The first input vector for the dot product, of type T1.
    - `T2 b`: The second input vector for the dot product, of type T2.
    - `T3 c`: The scalar value to be added to the dot product result, of type T3.
- **Control Flow**:
    - The function is a template function, allowing for different types for the inputs.
    - It calls the `syclcompat::dp4a` function with the provided arguments `a`, `b`, and `c`.
- **Output**: The function returns the result of the `syclcompat::dp4a` operation, which is the dot product of `a` and `b` plus `c`.


---
### vectorized\_min<!-- {{#callable:dpct::vectorized_min}} -->
The `vectorized_min` function computes the minimum of two values using SYCL vector operations and type casting.
- **Inputs**:
    - `S`: A template parameter representing the type to which the input values will be cast for comparison.
    - `T`: A template parameter representing the type of the input values `a` and `b`.
    - `a`: The first input value of type `T` to be compared.
    - `b`: The second input value of type `T` to be compared.
- **Control Flow**:
    - Create SYCL vector `v0` initialized with value `a` and vector `v1` initialized with value `b`.
    - Cast `v0` and `v1` to type `S` using the `as` method, resulting in `v2` and `v3`.
    - Compute the minimum of `v2` and `v3` using `sycl::min`, storing the result in `v4`.
    - Cast `v4` back to a SYCL vector of type `sycl::vec<T, 1>` and assign it to `v0`.
    - Return the value of `v0`.
- **Output**: The function returns the minimum of the two input values `a` and `b`, cast back to the original type `T`.


---
### pow<!-- {{#callable:dpct::pow}} -->
The `pow` function template computes the power of two non-floating-point numbers by converting them to double and using the SYCL `pow` function.
- **Inputs**:
    - `a`: The base of the power operation, a non-floating-point number of type T.
    - `b`: The exponent of the power operation, a non-floating-point number of type U.
- **Control Flow**:
    - The function checks if the type T is not a floating-point type using `std::enable_if_t` and `std::is_floating_point_v`.
    - If T is not a floating-point type, the function proceeds to convert both `a` and `b` to `double`.
    - The function then calls `sycl::pow` with the converted `double` values of `a` and `b`.
- **Output**: Returns the result of raising `a` to the power of `b` as a `double`.


---
### min<!-- {{#callable:dpct::min}} -->
The `min` function returns the minimum of two numbers, a 32-bit unsigned integer and a 64-bit unsigned integer, by casting the 32-bit integer to a 64-bit integer and using the SYCL `min` function.
- **Inputs**:
    - `a`: A 32-bit unsigned integer (`std::uint32_t`) to be compared.
    - `b`: A 64-bit unsigned integer (`std::uint64_t`) to be compared.
- **Control Flow**:
    - The function takes two arguments: a 32-bit unsigned integer `a` and a 64-bit unsigned integer `b`.
    - The 32-bit integer `a` is cast to a 64-bit integer to ensure both numbers are of the same type.
    - The SYCL `min` function is called with the casted `a` and `b` to determine the minimum value.
    - The result of the `min` function is returned.
- **Output**: A 64-bit unsigned integer (`std::uint64_t`) representing the minimum of the two input values.


---
### max<!-- {{#callable:dpct::max}} -->
The `max` function returns the maximum of two numbers, a 32-bit unsigned integer and a 64-bit unsigned integer, by casting the 32-bit integer to a 64-bit integer and using the SYCL `max` function.
- **Inputs**:
    - `a`: A 32-bit unsigned integer (`std::uint32_t`) to be compared.
    - `b`: A 64-bit unsigned integer (`std::uint64_t`) to be compared.
- **Control Flow**:
    - The function takes two parameters: a 32-bit unsigned integer `a` and a 64-bit unsigned integer `b`.
    - The 32-bit integer `a` is cast to a 64-bit integer to ensure both numbers are of the same type for comparison.
    - The SYCL `max` function is called with the casted `a` and `b` to determine the maximum value.
    - The result of the `max` function is returned.
- **Output**: The function returns a 64-bit unsigned integer (`std::uint64_t`) which is the maximum of the two input values.


---
### has\_capability\_or\_fail<!-- {{#callable:dpct::has_capability_or_fail}} -->
The `has_capability_or_fail` function checks if a given SYCL device supports specified aspects and throws an error if any aspect is not supported.
- **Inputs**:
    - `dev`: A `sycl::device` object representing the SYCL device to be checked for capabilities.
    - `props`: An `std::initializer_list` of `sycl::aspect` enumerations representing the capabilities to check for on the device.
- **Control Flow**:
    - Iterate over each aspect in the `props` list.
    - For each aspect, check if the device supports it using `dev.has(it)`.
    - If the device supports the aspect, continue to the next aspect.
    - If the device does not support the aspect, check if it is `sycl::aspect::fp64` or `sycl::aspect::fp16` and throw a runtime error with a specific message if so.
    - For other unsupported aspects, use a lambda function to get the aspect name and throw a runtime error with a message indicating the aspect is not supported.
- **Output**: The function does not return a value; it throws a `std::runtime_error` if any aspect is not supported by the device.


---
### get\_current\_device\_id<!-- {{#callable:dpct::get_current_device_id}} -->
The `get_current_device_id` function retrieves the ID of the current device being used by the device manager.
- **Inputs**: None
- **Control Flow**:
    - The function calls `dev_mgr::instance()` to get the singleton instance of the device manager.
    - It then calls `current_device_id()` on this instance to get the ID of the current device.
- **Output**: The function returns an unsigned integer representing the current device ID.


---
### get\_current\_device<!-- {{#callable:dpct::get_current_device}} -->
The `get_current_device` function retrieves the current device being used by the device manager.
- **Inputs**: None
- **Control Flow**:
    - The function calls `dev_mgr::instance()` to get the singleton instance of the device manager.
    - It then calls `current_device()` on this instance to retrieve the current device.
- **Output**: A reference to the `device_ext` object representing the current device.


---
### get\_device<!-- {{#callable:dpct::get_device}} -->
The `get_device` function retrieves a reference to a `device_ext` object corresponding to a given device ID from the device manager singleton.
- **Inputs**:
    - `id`: An unsigned integer representing the ID of the device to retrieve.
- **Control Flow**:
    - The function calls `dev_mgr::instance()` to get the singleton instance of the device manager.
    - It then calls `get_device(id)` on the device manager instance, passing the provided `id` as an argument.
    - The `get_device(id)` method of `dev_mgr` returns a reference to the `device_ext` object corresponding to the specified device ID.
- **Output**: A reference to a `device_ext` object representing the device with the specified ID.


---
### get\_in\_order\_queue<!-- {{#callable:dpct::get_in_order_queue}} -->
The `get_in_order_queue` function retrieves the in-order queue of the current device managed by the device manager singleton.
- **Inputs**: None
- **Control Flow**:
    - The function calls `dev_mgr::instance()` to get the singleton instance of the device manager.
    - It then calls `current_device()` on the device manager instance to get the current device.
    - Finally, it calls `in_order_queue()` on the current device to retrieve the in-order queue and returns it.
- **Output**: A reference to a `sycl::queue` object representing the in-order queue of the current device.


---
### gemm<!-- {{#callable:dpct::gemm}} -->
The `gemm` function performs a general matrix-matrix multiplication (GEMM) operation on matrices A and B, with optional transposition and scaling, and stores the result in matrix C using SYCL for parallel execution.
- **Inputs**:
    - `q`: A reference to a SYCL queue where the operation will be executed.
    - `a_trans`: Specifies whether matrix A should be transposed before multiplication.
    - `b_trans`: Specifies whether matrix B should be transposed before multiplication.
    - `m`: The number of rows in the matrix op(A) and the matrix C.
    - `n`: The number of columns in the matrix op(B) and the matrix C.
    - `k`: The number of columns in the matrix op(A) and the number of rows in the matrix op(B).
    - `alpha`: A pointer to a scaling factor for the matrix-matrix product.
    - `a`: A pointer to the input matrix A.
    - `a_type`: The data type of the matrix A.
    - `lda`: The leading dimension of matrix A.
    - `b`: A pointer to the input matrix B.
    - `b_type`: The data type of the matrix B.
    - `ldb`: The leading dimension of matrix B.
    - `beta`: A pointer to a scaling factor for matrix C.
    - `c`: A pointer to the input/output matrix C.
    - `c_type`: The data type of the matrix C.
    - `ldc`: The leading dimension of matrix C.
    - `scaling_type`: The data type of the scaling factors.
- **Control Flow**:
    - Check if the scaling type needs to be adjusted based on the data type of matrix C.
    - Compute a unique key based on the data types of matrices A, B, C, and the scaling type.
    - Use a switch statement to select the appropriate implementation of the GEMM operation based on the computed key.
    - For each case in the switch statement, call the `gemm_impl` function with the appropriate template parameters and arguments.
    - If no matching case is found, throw a runtime error indicating an unsupported combination of data types.
- **Output**: The function does not return a value; it modifies the matrix C in place with the result of the matrix-matrix multiplication.


---
### gemm\_batch<!-- {{#callable:dpct::gemm_batch}} -->
The `gemm_batch` function performs a batch of general matrix-matrix multiplications (GEMM) on matrices A, B, and C with specified data types and scaling factors, using the SYCL queue for execution.
- **Inputs**:
    - `q`: A reference to a SYCL queue where the routine should be executed.
    - `a_trans`: Specifies the operation applied to matrix A (transpose or not).
    - `b_trans`: Specifies the operation applied to matrix B (transpose or not).
    - `m`: The number of rows of the matrix op(A) and of the matrix C.
    - `n`: The number of columns of the matrix op(B) and of the matrix C.
    - `k`: The number of columns of the matrix op(A) and the number of rows of the matrix op(B).
    - `alpha`: A pointer to the scaling factor for the matrix-matrix product.
    - `a`: A pointer to the input matrix A.
    - `a_type`: The data type of the matrix A.
    - `lda`: The leading dimension of matrix A.
    - `stride_a`: The stride between the different A matrices.
    - `b`: A pointer to the input matrix B.
    - `b_type`: The data type of the matrix B.
    - `ldb`: The leading dimension of matrix B.
    - `stride_b`: The stride between the different B matrices.
    - `beta`: A pointer to the scaling factor for matrix C.
    - `c`: A pointer to the input/output matrix C.
    - `c_type`: The data type of the matrix C.
    - `ldc`: The leading dimension of matrix C.
    - `stride_c`: The stride between the different C matrices.
    - `batch_size`: The number of matrix multiply operations to perform.
    - `scaling_type`: The data type of the scaling factors.
- **Control Flow**:
    - The function first checks and adjusts the scaling type if necessary, based on the data type of matrix C.
    - It calculates a unique key based on the data types of matrices A, B, C, and the scaling type.
    - A switch statement is used to select the appropriate implementation of the GEMM batch operation based on the calculated key.
    - For each case in the switch statement, the function calls `detail::gemm_batch_impl` with the appropriate template parameters and arguments.
    - If the data type combination is unsupported, the function throws a runtime error.
- **Output**: The function does not return a value; it performs operations directly on the input/output matrix C.


---
### accessor<!-- {{#callable:dpct::accessor}} -->
The `accessor` constructor initializes a 2D SYCL accessor using a pointer from another accessor and a specified range.
- **Inputs**:
    - `acc`: A constant reference to an `accessor_t` object, which provides a pointer to the data to be accessed.
    - `in_range`: A constant reference to a `sycl::range<2>` object, specifying the range of the 2D data to be accessed.
- **Control Flow**:
    - The constructor is called with two parameters: `acc` and `in_range`.
    - It initializes the base class `accessor` with the pointer obtained from `acc.get_pointer()` and the specified `in_range`.
- **Output**: A new `accessor` object is created, initialized to access a 2D range of data using the pointer from the provided `accessor_t` object.


---
### operator\[\]<!-- {{#callable:dpct::operator[]}} -->
The `operator[]` function returns a pointer to an element in a 2D array-like structure based on a given index.
- **Inputs**:
    - `index`: A `size_t` value representing the index of the element to access in the 2D array-like structure.
- **Control Flow**:
    - The function calculates the offset by multiplying the second dimension size (obtained from `_range.get(1)`) with the given `index`.
    - It returns a pointer to the element at the calculated offset from the base pointer `_data`.
- **Output**: A `pointer_t` pointing to the element at the specified index in the 2D array-like structure.


---
### get\_ptr<!-- {{#callable:dpct::get_ptr}} -->
The `get_ptr` function returns a pointer to the data stored in the `pitched_data` class.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter that returns the private member `_data` of the `pitched_data` class.
- **Output**: A pointer of type `pointer_t` to the data stored in the `pitched_data` class.


---
### device\_memory<!-- {{#callable:dpct::detail::device_memory}} -->
The `device_memory` constructor initializes a `device_memory` object with a base size of 1.
- **Inputs**: None
- **Control Flow**:
    - The constructor `device_memory()` is called.
    - The constructor initializes the base class with a size of 1.
- **Output**: A `device_memory` object is created with a base size of 1.


---
### atomic\_fetch\_add<!-- {{#callable:dpct::atomic_fetch_add}} -->
The `atomic_fetch_add` function performs an atomic addition operation on a value at a specified memory address, using a specified memory order.
- **Inputs**:
    - `addr`: A pointer to the memory location where the atomic addition will be performed.
    - `operand`: The value to be added to the value at the memory location pointed to by `addr`.
    - `memoryOrder`: The memory order to be used for the atomic operation, which can be `sycl::memory_order::relaxed`, `sycl::memory_order::acq_rel`, or `sycl::memory_order::seq_cst`.
- **Control Flow**:
    - The function is a template with default template parameters for `addressSpace` set to `sycl::access::address_space::global_space` and `memoryOrder` set to `sycl::memory_order::relaxed`.
    - The function calls another overloaded version of `atomic_fetch_add` with the same parameters, effectively performing the atomic addition operation.
- **Output**: The function returns the original value at the memory location before the addition.


