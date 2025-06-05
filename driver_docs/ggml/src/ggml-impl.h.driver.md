# Purpose
This C header file is an internal component of the GGML library, which appears to be a machine learning or numerical computation library. The file provides a variety of utility functions and definitions that support the core operations of the library. It includes functions for logging, memory alignment, and mathematical operations, as well as data structures for managing tensors and computation graphs. The file also contains platform-specific optimizations for floating-point conversions, particularly between different precision formats like FP16, FP32, and BF16, which are crucial for efficient numerical computations in machine learning applications.

The file defines several static inline functions and macros for common operations, such as alignment and bit manipulation, which are essential for performance optimization. It also includes data structures and functions for managing hash sets and bitsets, which are likely used for efficient storage and retrieval of tensor data. Additionally, the file provides logging utilities with different log levels, which are important for debugging and monitoring the library's operations. The presence of conditional compilation directives indicates that the file is designed to be portable across different hardware architectures, including ARM and x86, and it includes specific optimizations for these platforms. Overall, this header file is a foundational component of the GGML library, providing essential utilities and optimizations for its numerical and machine learning computations.
# Imports and Dependencies

---
- `ggml.h`
- `gguf.h`
- `assert.h`
- `math.h`
- `stdlib.h`
- `stdbool.h`
- `stdint.h`
- `string.h`
- `arm_sve.h`
- `arm_neon.h`
- `immintrin.h`
- `vector`


# Data Structures

---
### ggml\_map\_custom1\_op\_params
- **Type**: `struct`
- **Members**:
    - `fun`: A function pointer of type `ggml_custom1_op_t` that represents a custom operation.
    - `n_tasks`: An integer representing the number of tasks to be executed.
    - `userdata`: A pointer to user-defined data that can be used within the custom operation.
- **Description**: The `ggml_map_custom1_op_params` structure is designed to encapsulate parameters for a custom operation in the GGML library. It includes a function pointer `fun` to specify the custom operation, an integer `n_tasks` to indicate how many tasks the operation should handle, and a `userdata` pointer to pass additional user-specific data to the operation. This structure allows for flexible and customizable operation execution within the GGML framework.


---
### ggml\_map\_custom2\_op\_params
- **Type**: `struct`
- **Members**:
    - `fun`: A function pointer of type `ggml_custom2_op_t` that represents a custom operation.
    - `n_tasks`: An integer representing the number of tasks to be executed.
    - `userdata`: A pointer to user-defined data that can be passed to the custom operation.
- **Description**: The `ggml_map_custom2_op_params` structure is designed to encapsulate parameters for a custom operation in the GGML library. It includes a function pointer `fun` for the custom operation, an integer `n_tasks` to specify how many tasks the operation should handle, and a `userdata` pointer for any additional data the user wishes to associate with the operation. This structure allows for flexible and customizable operation handling within the GGML framework.


---
### ggml\_map\_custom3\_op\_params
- **Type**: `struct`
- **Members**:
    - `fun`: A function pointer of type `ggml_custom3_op_t` that represents a custom operation.
    - `n_tasks`: An integer representing the number of tasks to be executed.
    - `userdata`: A pointer to user-defined data that can be used within the custom operation.
- **Description**: The `ggml_map_custom3_op_params` structure is designed to encapsulate parameters for a custom operation in the GGML library. It includes a function pointer `fun` for the custom operation, an integer `n_tasks` to specify how many tasks the operation should handle, and a `userdata` pointer for any additional data the user wishes to associate with the operation. This structure allows for flexible and customizable operations within the GGML framework, enabling users to define and execute their own operations with specific parameters.


---
### ggml\_custom\_op\_params
- **Type**: `struct`
- **Members**:
    - `fun`: A function pointer of type `ggml_custom_op_t` representing the custom operation to be executed.
    - `n_tasks`: An integer representing the number of tasks to be executed for the custom operation.
    - `userdata`: A pointer to user-defined data that can be used within the custom operation.
- **Description**: The `ggml_custom_op_params` structure is designed to encapsulate parameters for a custom operation within the GGML framework. It includes a function pointer to the custom operation, an integer specifying the number of tasks to be executed, and a pointer to user-defined data that can be utilized during the operation. This structure allows for flexible and customizable operations to be integrated into the GGML computation graph.


---
### ggml\_hash\_set
- **Type**: `struct`
- **Members**:
    - `size`: Represents the total capacity of the hash set.
    - `used`: A bitset indicating which keys are currently in use.
    - `keys`: An array of pointers to ggml_tensor structures, representing the actual keys in the set.
- **Description**: The `ggml_hash_set` is a data structure designed to manage a collection of `ggml_tensor` pointers using a hash set approach. It uses a bitset to track which slots in the `keys` array are occupied, allowing for efficient insertion, deletion, and lookup operations. The `size` member indicates the total number of slots available in the hash set, while the `used` bitset and `keys` array work together to manage the presence and retrieval of tensor keys. This structure is particularly useful for scenarios where quick access to a set of unique tensor objects is required.


---
### ggml\_cgraph\_eval\_order
- **Type**: `enum`
- **Members**:
    - `GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT`: Represents the evaluation order from left to right.
    - `GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT`: Represents the evaluation order from right to left.
    - `GGML_CGRAPH_EVAL_ORDER_COUNT`: Represents the count of evaluation orders available.
- **Description**: The `ggml_cgraph_eval_order` is an enumeration that defines the possible evaluation orders for a computation graph in the GGML library. It includes options for evaluating the graph from left to right or right to left, and also provides a count of the available evaluation orders. This enumeration is used to control the sequence in which nodes in a computation graph are processed during evaluation.


---
### ggml\_cgraph
- **Type**: `struct`
- **Members**:
    - `size`: Maximum number of nodes, leafs, gradients, and gradient accumulators.
    - `n_nodes`: Number of nodes currently in use.
    - `n_leafs`: Number of leafs currently in use.
    - `nodes`: Array of pointers to tensors with data that can change if the graph is evaluated.
    - `grads`: Array of pointers to tensors representing the gradients of the nodes.
    - `grad_accs`: Array of pointers to tensors that accumulate node gradients.
    - `leafs`: Array of pointers to tensors with constant data.
    - `visited_hash_set`: Hash set to track visited nodes.
    - `order`: Evaluation order of the computation graph.
- **Description**: The `ggml_cgraph` structure represents a computation graph used in machine learning or numerical computations. It manages a collection of nodes and leafs, where nodes are tensors that can change during graph evaluation, and leafs are constant tensors. The structure also maintains gradients and gradient accumulators for nodes, facilitating backpropagation in neural networks. The `visited_hash_set` helps in tracking visited nodes to avoid redundant computations, and the `order` field specifies the order in which the graph should be evaluated.


# Functions

---
### ggml\_up32<!-- {{#callable:ggml_up32}} -->
The `ggml_up32` function rounds up an integer to the nearest multiple of 32.
- **Inputs**:
    - `n`: An integer value that needs to be rounded up to the nearest multiple of 32.
- **Control Flow**:
    - The function takes an integer `n` as input.
    - It adds 31 to `n` to ensure rounding up to the next multiple of 32 if `n` is not already a multiple of 32.
    - It then performs a bitwise AND operation with the bitwise NOT of 31 (`~31`), effectively zeroing out the last 5 bits of the result, which rounds `n` up to the nearest multiple of 32.
- **Output**: The function returns an integer that is the smallest multiple of 32 greater than or equal to the input `n`.


---
### ggml\_up<!-- {{#callable:ggml_up}} -->
The `ggml_up` function rounds up an integer `n` to the nearest multiple of `m`, where `m` is a power of 2.
- **Inputs**:
    - `n`: The integer value to be rounded up.
    - `m`: The integer value representing the power of 2 to which `n` should be rounded up.
- **Control Flow**:
    - The function begins by asserting that `m` is a power of 2 using the condition `(m & (m - 1)) == 0`.
    - It then calculates the smallest multiple of `m` that is greater than or equal to `n` using the expression `(n + m - 1) & ~(m - 1)`.
- **Output**: The function returns an integer which is the smallest multiple of `m` that is greater than or equal to `n`.


---
### ggml\_set\_op\_params<!-- {{#callable:ggml_set_op_params}} -->
The `ggml_set_op_params` function sets the operation parameters for a given tensor by copying data from a provided source.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure where the operation parameters will be set.
    - `params`: A pointer to the source data containing the operation parameters to be copied.
    - `params_size`: The size of the data in bytes to be copied from `params` to the tensor's operation parameters.
- **Control Flow**:
    - The function begins by asserting that the `tensor` pointer is not NULL to prevent array bounds warnings.
    - It then checks that the `params_size` does not exceed the maximum allowed size for operation parameters (`GGML_MAX_OP_PARAMS`).
    - Finally, it uses `memcpy` to copy the data from `params` to the `op_params` field of the `tensor`.
- **Output**: The function does not return any value; it modifies the `op_params` field of the provided `ggml_tensor` structure.


---
### ggml\_get\_op\_params\_i32<!-- {{#callable:ggml_get_op_params_i32}} -->
The function `ggml_get_op_params_i32` retrieves a 32-bit integer operation parameter from a tensor at a specified index.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure from which the operation parameter is to be retrieved.
    - `i`: An unsigned 32-bit integer representing the index of the operation parameter to retrieve.
- **Control Flow**:
    - The function asserts that the index `i` is less than the maximum number of operation parameters divided by the size of an `int32_t`, ensuring the index is within bounds.
    - It then casts the `op_params` field of the `tensor` to a pointer to `int32_t` and returns the value at the specified index `i`.
- **Output**: The function returns the 32-bit integer operation parameter located at the specified index `i` within the tensor's operation parameters.


---
### ggml\_get\_op\_params\_f32<!-- {{#callable:ggml_get_op_params_f32}} -->
The function `ggml_get_op_params_f32` retrieves a float parameter from a tensor's operation parameters at a specified index.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure from which the operation parameter is to be retrieved.
    - `i`: An unsigned 32-bit integer representing the index of the float parameter to retrieve from the tensor's operation parameters.
- **Control Flow**:
    - The function asserts that the index `i` is less than the maximum number of operation parameters divided by the size of a float, ensuring the index is within bounds.
    - It then casts the `op_params` field of the `tensor` to a pointer to a float array and returns the float value at the specified index `i`.
- **Output**: The function returns a float value from the tensor's operation parameters at the specified index.


---
### ggml\_set\_op\_params\_i32<!-- {{#callable:ggml_set_op_params_i32}} -->
The function `ggml_set_op_params_i32` sets a specific integer operation parameter for a given tensor at a specified index.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure where the operation parameter will be set.
    - `i`: An unsigned 32-bit integer representing the index at which the operation parameter should be set.
    - `value`: A signed 32-bit integer representing the value to be set at the specified index in the tensor's operation parameters.
- **Control Flow**:
    - The function begins by asserting that the index `i` is within the valid range, which is less than `GGML_MAX_OP_PARAMS / sizeof(int32_t)`, ensuring that the index does not exceed the maximum allowed operation parameters for a tensor.
    - The function then casts the `op_params` field of the `tensor` to an `int32_t` pointer and assigns the `value` to the specified index `i`.
- **Output**: The function does not return any value; it modifies the `op_params` field of the `tensor` in place.


---
### ggml\_set\_op\_params\_f32<!-- {{#callable:ggml_set_op_params_f32}} -->
The function `ggml_set_op_params_f32` sets a specific float value in the operation parameters of a given tensor at a specified index.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure whose operation parameters are to be modified.
    - `i`: An unsigned 32-bit integer representing the index in the operation parameters array where the float value should be set.
    - `value`: A float value to be set at the specified index in the tensor's operation parameters.
- **Control Flow**:
    - The function begins by asserting that the index `i` is within the valid range, which is less than `GGML_MAX_OP_PARAMS / sizeof(float)`, ensuring it does not exceed the maximum allowed operation parameters for a float.
    - It then casts the `op_params` field of the `tensor` to a float pointer and assigns the `value` to the specified index `i`.
- **Output**: The function does not return any value; it modifies the operation parameters of the tensor in place.


---
### ggml\_bitset\_size<!-- {{#callable:ggml_bitset_size}} -->
The `ggml_bitset_size` function calculates the number of `ggml_bitset_t` elements required to store a bitset of a given size `n`.
- **Inputs**:
    - `n`: The number of bits that need to be stored in the bitset.
- **Control Flow**:
    - The function adds `BITSET_MASK` to the input `n`.
    - It then performs a right bitwise shift by `BITSET_SHR` on the result.
    - The final result is returned as the size of the bitset in terms of `ggml_bitset_t` elements.
- **Output**: The function returns a `size_t` value representing the number of `ggml_bitset_t` elements needed to store `n` bits.


---
### ggml\_bitset\_get<!-- {{#callable:ggml_bitset_get}} -->
The `ggml_bitset_get` function retrieves the boolean value of a specific bit in a bitset array.
- **Inputs**:
    - `bitset`: A pointer to a `ggml_bitset_t` array, representing the bitset from which a bit value is to be retrieved.
    - `i`: An index of type `size_t` indicating the position of the bit to be retrieved within the bitset.
- **Control Flow**:
    - The function calculates the index in the bitset array by right-shifting the input index `i` by `BITSET_SHR` bits.
    - It then calculates the bit position within the selected bitset element by applying a bitwise AND operation between `i` and `BITSET_MASK`.
    - The function checks if the bit at the calculated position is set by performing a bitwise AND operation between the bitset element and a bitmask with a single bit set at the calculated position.
    - The result of the bitwise operation is converted to a boolean value using double negation (`!!`).
- **Output**: A boolean value indicating whether the specified bit in the bitset is set (true) or not (false).


---
### ggml\_bitset\_set<!-- {{#callable:ggml_bitset_set}} -->
The `ggml_bitset_set` function sets a specific bit in a bitset to 1.
- **Inputs**:
    - `bitset`: A pointer to a `ggml_bitset_t` array, representing the bitset where a bit will be set.
    - `i`: An index of type `size_t` indicating the position of the bit to be set within the bitset.
- **Control Flow**:
    - Calculate the index in the bitset array by right-shifting `i` by `BITSET_SHR` bits.
    - Calculate the bit position within the bitset element by applying a bitwise AND between `i` and `BITSET_MASK`.
    - Set the bit at the calculated position to 1 using a bitwise OR operation.
- **Output**: The function does not return a value; it modifies the bitset in place.


---
### ggml\_bitset\_clear<!-- {{#callable:ggml_bitset_clear}} -->
The `ggml_bitset_clear` function clears a specific bit in a bitset at a given index.
- **Inputs**:
    - `bitset`: A pointer to a `ggml_bitset_t` array, representing the bitset where a bit will be cleared.
    - `i`: A `size_t` index indicating the position of the bit to be cleared in the bitset.
- **Control Flow**:
    - Calculate the index in the bitset array by right-shifting `i` by `BITSET_SHR` bits.
    - Calculate the bit position within the bitset element by applying a bitwise AND between `i` and `BITSET_MASK`.
    - Clear the bit at the calculated position by applying a bitwise AND with the negation of a left-shifted 1 at the bit position.
- **Output**: The function does not return a value; it modifies the bitset in place.


---
### ggml\_hash<!-- {{#callable:ggml_hash}} -->
The `ggml_hash` function computes a hash value for a given `ggml_tensor` pointer by shifting its address to the right by 4 bits.
- **Inputs**:
    - `p`: A pointer to a `ggml_tensor` structure, which represents the tensor whose address is to be hashed.
- **Control Flow**:
    - The function takes a pointer `p` to a `ggml_tensor` as input.
    - It casts the pointer to an unsigned integer type (`uintptr_t`) to perform arithmetic operations on the address.
    - The address is then shifted right by 4 bits, effectively discarding the last 4 bits, which are zero due to alignment constraints.
    - The result of the shift operation is cast to a `size_t` type and returned as the hash value.
- **Output**: A `size_t` value representing the hash of the input tensor's address, with the last 4 bits removed.


---
### ggml\_hash\_find<!-- {{#callable:ggml_hash_find}} -->
The `ggml_hash_find` function searches for a given tensor key in a hash set and returns its index or indicates if the hash set is full.
- **Inputs**:
    - `hash_set`: A pointer to a `ggml_hash_set` structure, representing the hash set in which to search for the key.
    - `key`: A pointer to a `ggml_tensor` structure, representing the key to search for in the hash set.
- **Control Flow**:
    - Compute the hash value of the key using [`ggml_hash`](#ggml_hash) and take modulo with the hash set size to get the initial index `h`.
    - Initialize `i` to `h` and start a linear probing loop.
    - In the loop, check if the current index `i` is used and if the key at this index is not equal to the given key.
    - If both conditions are true, increment `i` and wrap around using modulo with the hash set size.
    - If `i` returns to the initial index `h`, return `GGML_HASHSET_FULL` indicating the hash set is full and the key was not found.
    - If the loop exits without returning `GGML_HASHSET_FULL`, return the current index `i` where the key is found or can be inserted.
- **Output**: Returns the index of the key in the hash set if found, or `GGML_HASHSET_FULL` if the hash set is full and the key is not found.
- **Functions called**:
    - [`ggml_hash`](#ggml_hash)
    - [`ggml_bitset_get`](#ggml_bitset_get)


---
### ggml\_hash\_contains<!-- {{#callable:ggml_hash_contains}} -->
The `ggml_hash_contains` function checks if a given tensor key is present in a hash set.
- **Inputs**:
    - `hash_set`: A pointer to a `ggml_hash_set` structure, representing the hash set to be searched.
    - `key`: A pointer to a `ggml_tensor` structure, representing the key to be checked for presence in the hash set.
- **Control Flow**:
    - Call [`ggml_hash_find`](#ggml_hash_find) with `hash_set` and `key` to find the index of the key in the hash set.
    - Check if the returned index is not equal to `GGML_HASHSET_FULL`, indicating the hash set is not full and the key might be present.
    - Use [`ggml_bitset_get`](#ggml_bitset_get) to check if the bit at the found index is set, confirming the presence of the key in the hash set.
- **Output**: Returns `true` if the key is present in the hash set, otherwise returns `false`.
- **Functions called**:
    - [`ggml_hash_find`](#ggml_hash_find)
    - [`ggml_bitset_get`](#ggml_bitset_get)


---
### ggml\_hash\_insert<!-- {{#callable:ggml_hash_insert}} -->
The `ggml_hash_insert` function inserts a tensor key into a hash set, using linear probing to resolve collisions, and returns the index of the inserted key or a special value if the key already exists.
- **Inputs**:
    - `hash_set`: A pointer to a `ggml_hash_set` structure where the key will be inserted.
    - `key`: A pointer to a `ggml_tensor` structure that serves as the key to be inserted into the hash set.
- **Control Flow**:
    - Compute the hash value of the key and take modulo with the hash set size to get the initial index `h`.
    - Initialize `i` to `h` and start a loop to find an empty slot or the key itself.
    - Check if the slot at index `i` is unused using [`ggml_bitset_get`](#ggml_bitset_get); if unused, mark it as used, store the key, and return the index `i`.
    - If the key at index `i` matches the input key, return `GGML_HASHSET_ALREADY_EXISTS`.
    - Increment `i` using linear probing and wrap around using modulo operation with the hash set size.
    - Continue the loop until `i` returns to `h`, indicating the table is full, and abort with a fatal error.
- **Output**: Returns the index where the key was inserted, or `GGML_HASHSET_ALREADY_EXISTS` if the key already exists in the hash set. If the hash set is full, the function aborts with a fatal error.
- **Functions called**:
    - [`ggml_hash`](#ggml_hash)
    - [`ggml_bitset_get`](#ggml_bitset_get)
    - [`ggml_bitset_set`](#ggml_bitset_set)


---
### ggml\_hash\_find\_or\_insert<!-- {{#callable:ggml_hash_find_or_insert}} -->
The `ggml_hash_find_or_insert` function attempts to find a given tensor key in a hash set and inserts it if not already present, using linear probing for collision resolution.
- **Inputs**:
    - `hash_set`: A pointer to a `ggml_hash_set` structure, which contains the hash table and associated metadata.
    - `key`: A pointer to a `ggml_tensor` structure, which serves as the key to be found or inserted in the hash set.
- **Control Flow**:
    - Compute the hash value of the key and take modulo with the hash set size to get the initial index `h`.
    - Initialize `i` to `h` and start a do-while loop for linear probing.
    - Check if the current index `i` in the hash set is unused using [`ggml_bitset_get`](#ggml_bitset_get).
    - If unused, mark it as used with [`ggml_bitset_set`](#ggml_bitset_set), store the key at this index, and return the index `i`.
    - If the key at index `i` matches the input key, return the index `i`.
    - Increment `i` and wrap around using modulo with the hash set size, continuing the loop until `i` returns to `h`.
    - If the loop completes without finding an unused slot or matching key, abort the program with a fatal error message.
- **Output**: Returns the index of the key in the hash set if found or inserted successfully; aborts the program if the hash set is full and the key cannot be inserted.
- **Functions called**:
    - [`ggml_hash`](#ggml_hash)
    - [`ggml_bitset_get`](#ggml_bitset_get)
    - [`ggml_bitset_set`](#ggml_bitset_set)


---
### ggml\_compute\_fp16\_to\_fp32<!-- {{#callable:ggml_compute_fp16_to_fp32}} -->
The function `ggml_compute_fp16_to_fp32` converts a 16-bit floating-point number (FP16) to a 32-bit floating-point number (FP32).
- **Inputs**:
    - `h`: A 16-bit floating-point number (ggml_fp16_t) to be converted to a 32-bit floating-point number.
- **Control Flow**:
    - The input 16-bit number is left-shifted by 16 bits to form a 32-bit integer `w`.
    - The sign bit is extracted from `w` using a bitwise AND with `0x80000000`.
    - The value `w` is doubled to form `two_w`.
    - An exponent offset is defined as `0xE0 << 23`.
    - A conditional compilation directive sets `exp_scale` to `0x1.0p-112f` if certain C standards or compilers are used, otherwise it uses `fp32_from_bits(0x7800000)`.
    - A normalized value is calculated by converting `(two_w >> 4) + exp_offset` to a float and multiplying by `exp_scale`.
    - A magic mask and bias are defined for denormalized values, and a denormalized value is calculated by converting `(two_w >> 17) | magic_mask` to a float and subtracting `magic_bias`.
    - A cutoff value for denormalized numbers is defined as `1 << 27`.
    - The result is determined by checking if `two_w` is less than the denormalized cutoff; if true, the denormalized value is used, otherwise the normalized value is used.
    - The final result is constructed by combining the sign bit with the selected value and converting it back to a float.
- **Output**: A 32-bit floating-point number (float) representing the converted value from the input 16-bit floating-point number.
- **Functions called**:
    - [`fp32_from_bits`](#fp32_from_bits)
    - [`fp32_to_bits`](#fp32_to_bits)


---
### ggml\_compute\_fp32\_to\_fp16<!-- {{#callable:ggml_compute_fp32_to_fp16}} -->
The function `ggml_compute_fp32_to_fp16` converts a 32-bit floating-point number to a 16-bit floating-point representation.
- **Inputs**:
    - `f`: A 32-bit floating-point number (float) to be converted to a 16-bit floating-point representation.
- **Control Flow**:
    - Check if the environment supports certain C standards or compilers to define constants `scale_to_inf` and `scale_to_zero` for scaling operations.
    - Calculate a base value by scaling the absolute value of the input float `f` using `scale_to_inf` and `scale_to_zero`.
    - Convert the float `f` to its bit representation `w` and perform bitwise operations to extract the sign and calculate a bias.
    - Adjust the bias if it is below a certain threshold to ensure proper conversion.
    - Recalculate the base using the adjusted bias and convert it back to bits to extract exponent and mantissa bits.
    - Combine the sign, exponent, and mantissa bits to form the final 16-bit representation, handling special cases for overflow.
- **Output**: Returns a 16-bit floating-point representation (ggml_fp16_t) of the input 32-bit float.
- **Functions called**:
    - [`fp32_from_bits`](#fp32_from_bits)
    - [`fp32_to_bits`](#fp32_to_bits)


---
### fp32\_from\_bits<!-- {{#callable:fp32_from_bits}} -->
The `fp32_from_bits` function converts a 32-bit unsigned integer representation of a floating-point number into a `float` type.
- **Inputs**:
    - `w`: A 32-bit unsigned integer representing the bit pattern of a floating-point number.
- **Control Flow**:
    - A union is defined with two members: a 32-bit unsigned integer `as_bits` and a `float` `as_value`.
    - The input integer `w` is assigned to the `as_bits` member of the union.
    - The function returns the `as_value` member of the union, which interprets the bit pattern as a `float`.
- **Output**: A `float` value that corresponds to the bit pattern provided by the input integer.


---
### fp32\_to\_bits<!-- {{#callable:fp32_to_bits}} -->
The `fp32_to_bits` function converts a 32-bit floating-point number to its equivalent 32-bit binary representation.
- **Inputs**:
    - `f`: A 32-bit floating-point number (float) to be converted to its binary representation.
- **Control Flow**:
    - A union is defined with two members: a float and a uint32_t, allowing shared memory space.
    - The input float 'f' is assigned to the float member of the union.
    - The function returns the uint32_t member of the union, which now contains the binary representation of the float.
- **Output**: A 32-bit unsigned integer representing the binary form of the input float.


---
### ggml\_lookup\_fp16\_to\_fp32<!-- {{#callable:ggml_lookup_fp16_to_fp32}} -->
The function `ggml_lookup_fp16_to_fp32` converts a 16-bit floating-point value to a 32-bit floating-point value using a precomputed lookup table.
- **Inputs**:
    - `f`: A 16-bit floating-point value (`ggml_fp16_t`) to be converted to a 32-bit floating-point value.
- **Control Flow**:
    - The function begins by declaring a 16-bit unsigned integer `s`.
    - It uses `memcpy` to copy the contents of the input `f` into `s`.
    - The function then returns the 32-bit floating-point value from the `ggml_table_f32_f16` lookup table at the index `s`.
- **Output**: A 32-bit floating-point value corresponding to the input 16-bit floating-point value.


---
### ggml\_compute\_bf16\_to\_fp32<!-- {{#callable:ggml_compute_bf16_to_fp32}} -->
The function `ggml_compute_bf16_to_fp32` converts a 16-bit bfloat16 value to a 32-bit float value.
- **Inputs**:
    - `h`: A `ggml_bf16_t` type representing a 16-bit bfloat16 value to be converted to a 32-bit float.
- **Control Flow**:
    - A union is defined with a float and a uint32_t to facilitate bit manipulation.
    - The 16-bit bfloat16 value's bits are shifted left by 16 to align with the 32-bit float format.
    - The resulting 32-bit integer is stored in the union, and the float value is returned.
- **Output**: A 32-bit float value that represents the converted bfloat16 input.


---
### ggml\_compute\_fp32\_to\_bf16<!-- {{#callable:ggml_compute_fp32_to_bf16}} -->
The function `ggml_compute_fp32_to_bf16` converts a 32-bit floating-point number to a 16-bit bfloat16 representation.
- **Inputs**:
    - `s`: A 32-bit floating-point number (float) to be converted to bfloat16.
- **Control Flow**:
    - The function begins by declaring a variable `h` of type `ggml_bf16_t` to store the result.
    - A union `u` is used to access the bits of the input float `s` as a 32-bit unsigned integer.
    - The float `s` is assigned to the union `u`, allowing its bit representation to be accessed via `u.i`.
    - The function checks if the input is a NaN by examining if the exponent bits are all ones and the mantissa is non-zero.
    - If the input is NaN, the function sets the bfloat16 bits to a quiet NaN by shifting the bits and setting a specific bit.
    - If the input is not NaN, the function rounds the float to the nearest bfloat16 by adding a rounding bias and shifting the bits right by 16.
    - The resulting bfloat16 bits are stored in `h.bits`.
    - The function returns the bfloat16 representation `h`.
- **Output**: The function returns a `ggml_bf16_t` structure containing the bfloat16 representation of the input float.


# Function Declarations (Public API)

---
### ggml\_log\_internal<!-- {{#callable_declaration:ggml_log_internal}} -->
Logs a formatted message at a specified log level.
- **Description**: Use this function to log messages with varying levels of severity, such as debug, info, warning, error, or continuation. It is designed to handle formatted strings similar to printf, allowing for dynamic message content. This function is typically used for internal logging purposes and should be called with the appropriate log level to categorize the message. Ensure that the format string and any additional arguments match the expected types to avoid undefined behavior.
- **Inputs**:
    - `level`: Specifies the severity level of the log message. It must be a valid value from the ggml_log_level enumeration.
    - `format`: A C-style format string that specifies how subsequent arguments are converted for output. Must not be null.
    - `...`: A variable number of arguments that correspond to the format specifiers in the format string. The types and number of these arguments must match the format string.
- **Output**: None
- **See also**: [`ggml_log_internal`](ggml.c.driver.md#ggml_log_internal)  (Implementation)


---
### ggml\_log\_callback\_default<!-- {{#callable_declaration:ggml_log_callback_default}} -->
Logs a message to the standard error stream.
- **Description**: This function is used to log messages to the standard error stream, typically for debugging or error reporting purposes. It takes a log level, a message text, and user data as parameters, but only the message text is used to output to stderr. The function is suitable for use as a default logging callback in systems where logging to stderr is appropriate. It does not perform any operations based on the log level or user data, and it flushes the stderr stream after writing the message.
- **Inputs**:
    - `level`: Specifies the log level of the message. This parameter is not used in the function, so any value is acceptable.
    - `text`: A pointer to a null-terminated string containing the message to be logged. Must not be null, as the function will attempt to write the string to stderr.
    - `user_data`: A pointer to user-defined data. This parameter is not used in the function, so any value is acceptable.
- **Output**: None
- **See also**: [`ggml_log_callback_default`](ggml.c.driver.md#ggml_log_callback_default)  (Implementation)


---
### ggml\_hash\_set\_new<!-- {{#callable_declaration:ggml_hash_set_new}} -->
Creates a new hash set with a specified initial size.
- **Description**: This function initializes a new hash set structure with a size that is adjusted to be optimal for the given input size. It allocates memory for the keys and a bitset to track used slots. This function should be used when you need to create a hash set to store tensor pointers, ensuring that the size is appropriate for the expected number of elements. The caller is responsible for managing the memory of the returned hash set, including freeing it when no longer needed.
- **Inputs**:
    - `size`: Specifies the initial size of the hash set. The function adjusts this size to an optimal value. The input should be a positive integer, and the function will handle any necessary adjustments internally.
- **Output**: Returns a `ggml_hash_set` structure initialized with the specified size, ready to store tensor pointers.
- **See also**: [`ggml_hash_set_new`](ggml.c.driver.md#ggml_hash_set_new)  (Implementation)


---
### ggml\_hash\_set\_free<!-- {{#callable_declaration:ggml_hash_set_free}} -->
Frees the resources associated with a hash set.
- **Description**: Use this function to release the memory allocated for a `ggml_hash_set` structure when it is no longer needed. This function should be called to prevent memory leaks after the hash set is no longer in use. It is important to ensure that the `hash_set` parameter is not null before calling this function, as passing a null pointer may lead to undefined behavior.
- **Inputs**:
    - `hash_set`: A pointer to a `ggml_hash_set` structure whose resources are to be freed. The pointer must not be null, and the caller retains ownership of the `ggml_hash_set` structure itself.
- **Output**: None
- **See also**: [`ggml_hash_set_free`](ggml.c.driver.md#ggml_hash_set_free)  (Implementation)


---
### ggml\_hash\_size<!-- {{#callable_declaration:ggml_hash_size}} -->
Determine the minimum hash set size for a given number of elements.
- **Description**: Use this function to calculate the smallest size for a hash set that can accommodate at least the specified number of elements. It is particularly useful when initializing a hash set to ensure efficient space utilization. The function returns the smallest prime number that is greater than or equal to the provided minimum size, ensuring optimal hash table performance. If the specified size exceeds the largest predefined prime, the function returns the next odd number greater than or equal to the specified size.
- **Inputs**:
    - `min_sz`: The minimum number of elements the hash set should accommodate. It must be a non-negative integer. If the value is larger than the largest predefined prime, the function will return the next odd number greater than or equal to this value.
- **Output**: Returns the smallest prime number greater than or equal to `min_sz`, or the next odd number if `min_sz` exceeds the largest predefined prime.
- **See also**: [`ggml_hash_size`](ggml.c.driver.md#ggml_hash_size)  (Implementation)


---
### ggml\_hash\_set\_reset<!-- {{#callable_declaration:ggml_hash_set_reset}} -->
Resets the state of a hash set.
- **Description**: Use this function to clear all elements from a given hash set, effectively resetting its state. This function is useful when you need to reuse a hash set without retaining any of its previous contents. It must be called with a valid `ggml_hash_set` structure that has been properly initialized. The function does not deallocate memory or change the size of the hash set; it only marks all entries as unused.
- **Inputs**:
    - `hash_set`: A pointer to a `ggml_hash_set` structure. This parameter must not be null and should point to a valid, initialized hash set. The function will reset the state of this hash set, marking all entries as unused.
- **Output**: None
- **See also**: [`ggml_hash_set_reset`](ggml.c.driver.md#ggml_hash_set_reset)  (Implementation)


---
### ggml\_hash\_contains<!-- {{#callable_declaration:ggml_hash_contains}} -->
Checks if a key is present in a hash set.
- **Description**: Use this function to determine if a specific tensor key is present within a given hash set. It is useful for checking membership before performing operations that require the key to be present. The function assumes that the hash set has been properly initialized and that the key is a valid tensor object. It does not modify the hash set or the key.
- **Inputs**:
    - `hash_set`: A pointer to a ggml_hash_set structure representing the hash set to be checked. Must not be null and should be properly initialized.
    - `key`: A pointer to a ggml_tensor structure representing the key to be checked for presence in the hash set. Must not be null.
- **Output**: Returns true if the key is present in the hash set, false otherwise.
- **See also**: [`ggml_hash_contains`](#ggml_hash_contains)  (Implementation)


---
### ggml\_hash\_find<!-- {{#callable_declaration:ggml_hash_find}} -->
Finds the index of a key in a hash set or indicates if the set is full.
- **Description**: Use this function to locate the index of a given key within a hash set. It is useful for checking if a key is present or determining where it should be inserted. The function employs linear probing to resolve hash collisions. If the key is not found and the hash set is full, it returns a special value indicating the set is full. This function does not modify the hash set and should be used when you need to check for the presence of a key without altering the set.
- **Inputs**:
    - `hash_set`: A pointer to a ggml_hash_set structure representing the hash set to search. Must not be null, and the hash set should be properly initialized.
    - `key`: A pointer to a ggml_tensor structure representing the key to find in the hash set. Must not be null.
- **Output**: Returns the index of the key if found, or GGML_HASHSET_FULL if the key is not found and the hash set is full.
- **See also**: [`ggml_hash_find`](#ggml_hash_find)  (Implementation)


---
### ggml\_hash\_insert<!-- {{#callable_declaration:ggml_hash_insert}} -->
Inserts a tensor into a hash set.
- **Description**: Use this function to insert a tensor into a hash set, ensuring that the tensor is not already present. The function uses linear probing to find an available slot in the hash set. It is important to ensure that the hash set is not full before calling this function, as it will abort execution if no empty slot is found. This function is typically used in scenarios where unique storage of tensor references is required.
- **Inputs**:
    - `hash_set`: A pointer to a ggml_hash_set structure where the tensor will be inserted. The hash set must be initialized and not full. The caller retains ownership.
    - `key`: A pointer to a ggml_tensor structure that represents the tensor to be inserted. The tensor must be valid and not null. The caller retains ownership.
- **Output**: Returns the index where the tensor was inserted, or GGML_HASHSET_ALREADY_EXISTS if the tensor is already in the hash set. Aborts if the hash set is full.
- **See also**: [`ggml_hash_insert`](#ggml_hash_insert)  (Implementation)


---
### ggml\_hash\_find\_or\_insert<!-- {{#callable_declaration:ggml_hash_find_or_insert}} -->
Finds or inserts a tensor in the hash set.
- **Description**: Use this function to locate a tensor within a hash set or insert it if it is not already present. It is suitable for managing unique tensor entries in a hash set. The function assumes that the hash set has been properly initialized and has sufficient capacity to accommodate new entries. If the hash set is full, the function will terminate the program. This function is useful in scenarios where you need to ensure that a tensor is part of a collection, either by confirming its presence or by adding it.
- **Inputs**:
    - `hash_set`: A pointer to a ggml_hash_set structure where the tensor will be searched for or inserted. Must not be null and should be properly initialized.
    - `key`: A pointer to a ggml_tensor structure that represents the tensor to be found or inserted. Must not be null.
- **Output**: Returns the index of the tensor in the hash set. If the hash set is full, the function will abort the program.
- **See also**: [`ggml_hash_find_or_insert`](#ggml_hash_find_or_insert)  (Implementation)


---
### ggml\_graph\_view<!-- {{#callable_declaration:ggml_graph_view}} -->
Returns a subgraph view of the specified range of nodes from a computation graph.
- **Description**: Use this function to create a view of a subset of nodes from an existing computation graph, specified by the range [i0, i1). This view does not include leaf nodes or gradients, which must be accessed from the original graph if needed. The function is useful for operations that require working with a specific segment of the graph without modifying the original structure. Ensure that the indices i0 and i1 are within the bounds of the original graph's node array.
- **Inputs**:
    - `cgraph`: A pointer to the original computation graph from which a subgraph view is to be created. Must not be null, and the graph should be properly initialized.
    - `i0`: The starting index of the node range to include in the subgraph view. Must be non-negative and less than or equal to i1.
    - `i1`: The ending index (exclusive) of the node range to include in the subgraph view. Must be greater than or equal to i0 and within the bounds of the original graph's node array.
- **Output**: A new ggml_cgraph structure representing the subgraph view, containing nodes from index i0 to i1-1 of the original graph.
- **See also**: [`ggml_graph_view`](ggml.c.driver.md#ggml_graph_view)  (Implementation)


---
### ggml\_aligned\_malloc<!-- {{#callable_declaration:ggml_aligned_malloc}} -->
Allocates memory with a specific alignment.
- **Description**: Use this function to allocate a block of memory with a specified alignment, which is useful for optimizing data access patterns on certain hardware architectures. The function returns a pointer to the allocated memory block, or NULL if the allocation fails. It is important to note that requesting an allocation of 0 bytes will result in a NULL return, and a warning will be logged. The alignment is platform-dependent, typically 64 bytes, but may vary on certain architectures. Ensure to free the allocated memory using `ggml_aligned_free` to avoid memory leaks.
- **Inputs**:
    - `size`: Specifies the number of bytes to allocate. Must be greater than 0 to avoid unexpected behavior. If 0, the function returns NULL and logs a warning.
- **Output**: Returns a pointer to the allocated memory block if successful, or NULL if the allocation fails or if size is 0.
- **See also**: [`ggml_aligned_malloc`](ggml.c.driver.md#ggml_aligned_malloc)  (Implementation)


---
### ggml\_aligned\_free<!-- {{#callable_declaration:ggml_aligned_free}} -->
Frees memory allocated with alignment considerations.
- **Description**: Use this function to free memory that was previously allocated with alignment requirements. It is essential to call this function to release resources when they are no longer needed, especially when the memory was allocated using platform-specific alignment methods. This function handles different platforms and conditions, ensuring that the memory is freed correctly according to the system's requirements. It is important to ensure that the pointer provided is valid and was allocated with the corresponding aligned allocation function.
- **Inputs**:
    - `ptr`: A pointer to the memory block to be freed. Must not be null if the memory was allocated and needs to be freed. The caller retains ownership and responsibility for ensuring it points to a valid memory block.
    - `size`: The size of the memory block to be freed. This parameter is not used in the function, but it should match the size used during allocation for consistency.
- **Output**: None
- **See also**: [`ggml_aligned_free`](ggml.c.driver.md#ggml_aligned_free)  (Implementation)


---
### gguf\_type\_size<!-- {{#callable_declaration:gguf_type_size}} -->
Returns the size in bytes of a given gguf_type.
- **Description**: Use this function to determine the size in bytes of a specific gguf_type, which is essential for memory allocation and data handling when working with GGUF types. This function is particularly useful when you need to allocate memory or perform operations that depend on the size of the data type. It returns 0 if the provided type is not recognized, indicating an invalid or unsupported type.
- **Inputs**:
    - `type`: An enumeration value of type gguf_type representing the data type whose size is to be determined. The value must be a valid gguf_type; otherwise, the function returns 0.
- **Output**: The function returns the size in bytes of the specified gguf_type. If the type is not recognized, it returns 0.
- **See also**: [`gguf_type_size`](gguf.cpp.driver.md#gguf_type_size)  (Implementation)


---
### gguf\_init\_from\_file\_impl<!-- {{#callable_declaration:gguf_init_from_file_impl}} -->
Initializes a GGUF context from a file.
- **Description**: This function initializes a GGUF context by reading data from a specified file. It is used to load tensor and key-value pair information from a GGUF file, which is a specific file format. The function must be called with a valid file pointer and initialization parameters. It handles various error conditions, such as invalid file format, unsupported GGUF versions, and memory allocation failures, by returning a null pointer. The function also supports loading tensor data into a provided context if specified in the parameters.
- **Inputs**:
    - `file`: A pointer to a FILE object representing the file to read from. Must not be null. The file should be opened in a mode that allows reading.
    - `params`: A struct gguf_init_params containing initialization parameters. This includes options for memory allocation and whether to load tensor data into a provided context.
- **Output**: Returns a pointer to a gguf_context struct if successful, or null if an error occurs during initialization.
- **See also**: [`gguf_init_from_file_impl`](gguf.cpp.driver.md#gguf_init_from_file_impl)  (Implementation)


---
### gguf\_write\_to\_buf<!-- {{#callable_declaration:gguf_write_to_buf}} -->
Writes GGUF data to a buffer.
- **Description**: This function serializes data from a GGUF context into a provided buffer. It writes the header, key-value pairs, and tensor metadata to the buffer. If the `only_meta` parameter is false, it also writes the tensor data. This function is useful for exporting GGUF data to a binary format for storage or transmission. Ensure that the context is properly initialized and contains valid data before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure containing the data to be serialized. Must not be null and should be properly initialized with valid data.
    - `buf`: A reference to a `std::vector<int8_t>` that will be filled with the serialized data. The caller retains ownership of the buffer.
    - `only_meta`: A boolean flag indicating whether to write only metadata (true) or both metadata and tensor data (false).
- **Output**: None
- **See also**: [`gguf_write_to_buf`](gguf.cpp.driver.md#gguf_write_to_buf)  (Implementation)


