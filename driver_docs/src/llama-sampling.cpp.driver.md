# Purpose
The provided C++ source code file is a comprehensive implementation of a sampling framework for a language model, specifically designed to handle various sampling strategies and techniques. The file defines a series of structures and functions that facilitate the initialization, application, and management of different sampling methods, such as greedy, top-k, top-p, temperature-based, and more advanced techniques like Mirostat and infill sampling. These sampling strategies are crucial for generating text from a language model by selecting the next token based on probabilities derived from the model's output logits.

The code is organized into several key components, including the definition of a [`ring_buffer`](#ring_bufferring_buffer) template for managing a fixed-capacity buffer, and a series of `llama_sampler` structures and functions that encapsulate different sampling strategies. Each sampler is implemented with a specific interface (`llama_sampler_i`) that includes methods for applying the sampling logic, resetting the sampler state, and cloning the sampler. The file also includes utility functions for managing random number generation seeds and performance measurement of the sampling process. This code is intended to be part of a larger system, likely a language model framework, where it can be imported and used to provide flexible and efficient sampling capabilities for text generation tasks.
# Imports and Dependencies

---
- `llama-sampling.h`
- `llama-impl.h`
- `llama-vocab.h`
- `llama-grammar.h`
- `algorithm`
- `cassert`
- `cfloat`
- `chrono`
- `cmath`
- `cstdlib`
- `cstring`
- `ctime`
- `numeric`
- `random`
- `unordered_map`
- `stdexcept`


# Global Variables

---
### llama\_sampler\_chain\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_chain_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for operations related to a sampler chain. These operations include naming, accepting tokens, applying sampling logic, resetting, cloning, and freeing resources associated with the sampler chain.
- **Use**: This variable is used to define the interface for a sampler chain, allowing it to perform various operations through the specified function pointers.


---
### llama\_sampler\_greedy\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_greedy_i` is a static instance of the `llama_sampler_i` structure, which defines a greedy sampling interface for the llama sampling system. It specifies the functions to be used for the greedy sampling strategy, including the name and apply functions, while leaving other functions like accept, reset, clone, and free as nullptr, indicating they are not used in this context.
- **Use**: This variable is used to define the behavior of a greedy sampler, which selects the token with the highest probability during the sampling process.


---
### llama\_sampler\_dist\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_dist_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for operations related to a distribution-based sampler. This includes functions for applying the sampler, resetting it, cloning, and freeing resources.
- **Use**: This variable is used to define the interface for a distribution-based sampler, allowing it to be initialized and used in sampling operations.


---
### llama\_sampler\_softmax\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_softmax_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for a softmax sampling interface. It includes a name function, an apply function, and placeholders for accept, reset, clone, and free functions, which are set to nullptr.
- **Use**: This variable is used to define the behavior of a softmax sampler by providing function pointers for operations like applying the softmax function to a token data array.


---
### llama\_sampler\_top\_k\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_top_k_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for handling top-k sampling operations. This structure includes pointers to functions for naming, applying, cloning, and freeing the sampler, but does not include functions for accepting or resetting.
- **Use**: This variable is used to define the interface for top-k sampling operations, allowing the application of top-k logic to a set of token data.


---
### llama\_sampler\_top\_p\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_top_p_i` is a static global variable of type `struct llama_sampler_i` that defines an interface for a top-p sampling strategy in the llama sampling framework. It contains function pointers for operations such as applying the sampling strategy, cloning, and freeing resources.
- **Use**: This variable is used to define the behavior of a top-p sampler, which is a probabilistic sampling method that selects tokens based on cumulative probability thresholds.


---
### llama\_sampler\_min\_p\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_min_p_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for operations related to the 'min-p' sampling strategy. This structure includes pointers to functions for naming, applying, cloning, and freeing the sampler, but notably lacks an 'accept' or 'reset' function.
- **Use**: This variable is used to define the interface for a 'min-p' sampler, specifying how it should be applied, cloned, and freed.


---
### llama\_sampler\_typical\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_typical_i` is a static instance of the `llama_sampler_i` structure, which defines a typical sampling interface for the Llama sampling system. It includes function pointers for operations such as applying the sampler, cloning, and freeing resources.
- **Use**: This variable is used to define the behavior and operations of a typical sampler in the Llama sampling framework.


---
### llama\_sampler\_temp\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_temp_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for operations related to a temperature-based sampling strategy in the Llama sampling framework. This structure includes pointers to functions for naming, applying, cloning, and freeing the sampler, with some functions set to `nullptr` indicating they are not implemented for this sampler.
- **Use**: This variable is used to define the interface for a temperature-based sampler, allowing it to be initialized and used within the Llama sampling framework.


---
### llama\_sampler\_temp\_ext\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_temp_ext_i` is a static instance of the `llama_sampler_i` structure, which defines an interface for a sampler with extended temperature control. This structure includes function pointers for operations such as applying the sampler, cloning, and freeing resources.
- **Use**: This variable is used to define the behavior of a sampler that applies extended temperature scaling to token probabilities in a language model.


---
### llama\_sampler\_xtc\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_xtc_i` is a static global variable of type `struct llama_sampler_i` that defines an interface for a specific sampling strategy called 'xtc'. It includes function pointers for operations such as applying the sampler, resetting it, cloning, and freeing resources.
- **Use**: This variable is used to define the behavior of the 'xtc' sampler by providing function pointers for its operations.


---
### llama\_sampler\_mirostat\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_mirostat_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for the Mirostat sampling method. This structure is used to implement a specific sampling strategy in the Llama library.
- **Use**: This variable is used to define the interface for the Mirostat sampling method, including functions for applying, resetting, cloning, and freeing the sampler.


---
### llama\_sampler\_mirostat\_v2\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_mirostat_v2_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for the Mirostat V2 sampling algorithm. This structure includes pointers to functions for naming, applying, resetting, cloning, and freeing the sampler.
- **Use**: This variable is used to define the interface for the Mirostat V2 sampling algorithm, allowing it to be used within the llama sampling framework.


---
### llama\_sampler\_init\_grammar\_impl
- **Type**: `struct llama_sampler *`
- **Description**: The `llama_sampler_init_grammar_impl` is a static function that initializes a `llama_sampler` structure for grammar-based sampling. It takes various parameters including a vocabulary, grammar string, grammar root, and trigger words or patterns to configure the grammar sampler.
- **Use**: This function is used to create and configure a `llama_sampler` instance that applies grammar rules during the sampling process.


---
### llama\_sampler\_grammar\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_grammar_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for handling grammar-based sampling operations. It includes functions for naming, accepting, applying, resetting, cloning, and freeing a grammar sampler.
- **Use**: This variable is used to define the interface for grammar-based sampling operations in the llama sampling system.


---
### llama\_sampler\_penalties\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_penalties_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for handling penalties in a sampling process. It includes functions for naming, accepting, applying, resetting, cloning, and freeing the sampler penalties.
- **Use**: This variable is used to manage and apply penalties during the sampling process in a llama-based system, affecting how tokens are selected based on their past occurrences.


---
### llama\_sampler\_top\_n\_sigma\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_top_n_sigma_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for operations related to the 'top-n-sigma' sampling strategy. This structure includes pointers to functions for naming, applying, cloning, and freeing the sampler, but notably lacks an 'accept' and 'reset' function.
- **Use**: This variable is used to define the interface for a 'top-n-sigma' sampler, specifying how it should be applied, cloned, and freed.


---
### llama\_sampler\_dry\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_dry_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for operations related to the 'dry' sampling method. This includes functions for naming, accepting, applying, resetting, cloning, and freeing the sampler.
- **Use**: This variable is used to define the interface for the 'dry' sampling method, allowing it to be used in the llama sampling framework.


---
### llama\_sampler\_logit\_bias\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_logit_bias_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for handling logit bias operations in a sampling context. It includes pointers to functions for naming, applying, cloning, and freeing the sampler, but does not include functions for accepting or resetting.
- **Use**: This variable is used to define the interface for logit bias operations in the llama sampling process.


---
### llama\_sampler\_infill\_i
- **Type**: `struct llama_sampler_i`
- **Description**: The `llama_sampler_infill_i` is a static instance of the `llama_sampler_i` structure, which defines a set of function pointers for operations related to infill sampling in the Llama library. It includes pointers to functions for naming, applying, cloning, and freeing the sampler, but does not include functions for accepting or resetting.
- **Use**: This variable is used to define the behavior of an infill sampler by providing specific function implementations for operations like applying the sampler and managing its lifecycle.


# Data Structures

---
### ring\_buffer<!-- {{#data_structure:ring_buffer}} -->
- **Type**: `struct`
- **Members**:
    - `capacity`: Stores the maximum number of elements the ring buffer can hold.
    - `sz`: Tracks the current number of elements in the ring buffer.
    - `first`: Index of the first element in the ring buffer.
    - `pos`: Index where the next element will be inserted in the ring buffer.
    - `data`: A vector that holds the elements of the ring buffer.
- **Description**: The `ring_buffer` is a templated data structure that implements a circular buffer with a fixed capacity. It allows for efficient insertion and removal of elements in a FIFO manner. The buffer automatically overwrites the oldest data when it becomes full, maintaining a constant size. It provides methods to access the front and back elements, push new elements to the back, and pop elements from the front. The internal state is managed using indices to track the start and end of the buffer, and a vector to store the elements.

**Methods**

---
#### ring\_buffer::ring\_buffer<!-- {{#callable:ring_buffer::ring_buffer}} -->
The `ring_buffer` constructor initializes a ring buffer with a specified capacity.
- **Inputs**:
    - `cap`: The capacity of the ring buffer, which determines the maximum number of elements it can hold.
- **Control Flow**:
    - The constructor initializes the `capacity` member variable with the provided `cap` value.
    - It initializes the `data` member variable as a vector with a size equal to `cap`.
- **Output**: A `ring_buffer` object is created with the specified capacity and an internal data storage vector of the same size.
- **See also**: [`ring_buffer`](../common/sampling.cpp.driver.md#ring_buffer)  (Data Structure)


---
#### ring\_buffer::front<!-- {{#callable:ring_buffer::front}} -->
The `front` function returns a reference to the first element in the ring buffer, throwing an exception if the buffer is empty.
- **Inputs**: None
- **Control Flow**:
    - Check if the size of the ring buffer (`sz`) is zero.
    - If the buffer is empty, throw a `std::runtime_error` with the message "ring buffer is empty".
    - Return a reference to the element at the `first` index of the `data` vector.
- **Output**: A reference to the first element in the ring buffer.
- **See also**: [`ring_buffer`](../common/sampling.cpp.driver.md#ring_buffer)  (Data Structure)


---
#### ring\_buffer::back<!-- {{#callable:ring_buffer::back}} -->
The `back` function returns a reference to the last element in the ring buffer, throwing an exception if the buffer is empty.
- **Inputs**: None
- **Control Flow**:
    - Check if the size of the ring buffer (`sz`) is zero.
    - If the buffer is empty, throw a `std::runtime_error` with the message "ring buffer is empty".
    - Return a reference to the element at the current position (`pos`) in the data vector.
- **Output**: A reference to the last element in the ring buffer.
- **See also**: [`ring_buffer`](../common/sampling.cpp.driver.md#ring_buffer)  (Data Structure)


---
#### ring\_buffer::push\_back<!-- {{#callable:ring_buffer::push_back}} -->
The `push_back` function adds a new element to the ring buffer, handling buffer overflow by overwriting the oldest element if necessary.
- **Inputs**:
    - `value`: A constant reference to the element of type `T` to be added to the ring buffer.
- **Control Flow**:
    - Check if the buffer's capacity is zero and throw a runtime error if true.
    - If the buffer is full (size equals capacity), increment the `first` index to overwrite the oldest element.
    - If the buffer is not full, increment the size `sz`.
    - Assign the new value to the current position `pos` in the buffer.
    - Update the position `pos` to the next index, wrapping around using modulo operation with the capacity.
- **Output**: The function does not return a value; it modifies the internal state of the ring buffer by adding the new element.
- **See also**: [`ring_buffer`](../common/sampling.cpp.driver.md#ring_buffer)  (Data Structure)


---
#### ring\_buffer::pop\_front<!-- {{#callable:ring_buffer::pop_front}} -->
The `pop_front` function removes and returns the first element from a ring buffer, updating the buffer's state accordingly.
- **Inputs**: None
- **Control Flow**:
    - Check if the buffer size `sz` is zero; if so, throw a `std::runtime_error` indicating the buffer is empty.
    - Retrieve the value at the `first` index of the `data` vector and store it in `value`.
    - Update the `first` index to the next position in the buffer using modulo arithmetic with `capacity`.
    - Decrement the buffer size `sz` by one.
    - Return the retrieved `value`.
- **Output**: The function returns the first element of the ring buffer of type `T`.
- **See also**: [`ring_buffer`](../common/sampling.cpp.driver.md#ring_buffer)  (Data Structure)


---
#### ring\_buffer::rat<!-- {{#callable:ring_buffer::rat}} -->
The `rat` function retrieves an element from a ring buffer in reverse order based on the given index.
- **Inputs**:
    - `i`: An index of type `size_t` representing the position in reverse order from which to retrieve the element in the ring buffer.
- **Control Flow**:
    - Check if the input index `i` is greater than or equal to the current size `sz` of the ring buffer.
    - If the index is out of bounds, throw a `std::runtime_error` with the message "ring buffer: index out of bounds".
    - Calculate the position in the buffer using the formula `(first + sz - i - 1) % capacity` to access the element in reverse order.
    - Return the element at the calculated position in the buffer.
- **Output**: A constant reference to the element of type `T` at the specified reverse index in the ring buffer.
- **See also**: [`ring_buffer`](../common/sampling.cpp.driver.md#ring_buffer)  (Data Structure)


---
#### ring\_buffer::to\_vector<!-- {{#callable:ring_buffer::to_vector}} -->
The `to_vector` function converts the contents of a `ring_buffer` into a standard `std::vector`.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty `std::vector` named `result` to store the elements of the ring buffer.
    - Reserve space in `result` for `sz` elements to optimize memory allocation.
    - Iterate over the range from 0 to `sz` (exclusive) to access each element in the ring buffer.
    - For each index `i`, calculate the actual index in the `data` vector using `(first + i) % capacity` to handle the circular nature of the buffer.
    - Push the element at the calculated index into the `result` vector.
    - Return the `result` vector containing all elements of the ring buffer in order.
- **Output**: A `std::vector<T>` containing the elements of the ring buffer in the order they appear from `first` to `pos`.
- **See also**: [`ring_buffer`](../common/sampling.cpp.driver.md#ring_buffer)  (Data Structure)


---
#### ring\_buffer::clear<!-- {{#callable:ring_buffer::clear}} -->
The `clear` function resets the state of the ring buffer by setting its size, first position, and current position to zero.
- **Inputs**: None
- **Control Flow**:
    - Set the size (`sz`) of the buffer to 0.
    - Set the first position (`first`) of the buffer to 0.
    - Set the current position (`pos`) of the buffer to 0.
- **Output**: The function does not return any value; it modifies the internal state of the ring buffer.
- **See also**: [`ring_buffer`](../common/sampling.cpp.driver.md#ring_buffer)  (Data Structure)


---
#### ring\_buffer::empty<!-- {{#callable:ring_buffer::empty}} -->
The `empty` function checks if the ring buffer is empty by comparing its size to zero.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the size (`sz`) of the ring buffer is equal to zero.
    - If `sz` is zero, it returns `true`, indicating the buffer is empty.
    - If `sz` is not zero, it returns `false`, indicating the buffer is not empty.
- **Output**: A boolean value indicating whether the ring buffer is empty (`true`) or not (`false`).
- **See also**: [`ring_buffer`](../common/sampling.cpp.driver.md#ring_buffer)  (Data Structure)


---
#### ring\_buffer::size<!-- {{#callable:ring_buffer::size}} -->
The `size` function returns the current number of elements in the ring buffer.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns the value of the member variable `sz`, which represents the current size of the ring buffer.
- **Output**: The function returns a `size_t` representing the number of elements currently stored in the ring buffer.
- **See also**: [`ring_buffer`](../common/sampling.cpp.driver.md#ring_buffer)  (Data Structure)



---
### probs\_iterator<!-- {{#data_structure:llama_sample_dist::probs_iterator}} -->
- **Type**: `struct`
- **Members**:
    - `iterator_category`: Defines the iterator category as an input iterator.
    - `value_type`: Specifies the type of value the iterator points to, which is a float.
    - `pointer`: Defines the pointer type for the iterator, which is a pointer to float.
    - `reference`: Defines the reference type for the iterator, which is a reference to float.
    - `difference_type`: Specifies the type used to represent the difference between two iterators, which is ptrdiff_t.
    - `data`: A pointer to a constant llama_token_data structure, representing the data the iterator operates on.
- **Description**: The `probs_iterator` struct is a custom iterator designed to traverse through a collection of `llama_token_data` structures, specifically focusing on the probability values within these structures. It is defined as an input iterator, meaning it can read from the pointed-to data but not modify it. The iterator provides standard operations such as equality comparison, dereferencing to access the probability value, and increment operations to move through the data. This struct is particularly useful in scenarios where probability data needs to be iterated over in a read-only manner, such as in sampling or statistical analysis tasks.
- **Member Functions**:
    - [`llama_sample_dist::probs_iterator::operator==`](#probs_iteratoroperator==)
    - [`llama_sample_dist::probs_iterator::operator!=`](#probs_iteratoroperator!=)
    - [`llama_sample_dist::probs_iterator::operator*`](#probs_iteratoroperator*)
    - [`llama_sample_dist::probs_iterator::operator++`](#probs_iteratoroperator++)
    - [`llama_sample_dist::probs_iterator::operator++`](#probs_iteratoroperator++)

**Methods**

---
#### probs\_iterator::operator==<!-- {{#callable:llama_sample_dist::probs_iterator::operator==}} -->
The `operator==` function checks if two `probs_iterator` objects point to the same data.
- **Inputs**:
    - `other`: A reference to another `probs_iterator` object to compare with the current object.
- **Control Flow**:
    - The function compares the `data` member of the current `probs_iterator` object with the `data` member of the `other` `probs_iterator` object.
    - If the `data` members are equal, the function returns `true`; otherwise, it returns `false`.
- **Output**: A boolean value indicating whether the two `probs_iterator` objects are equal (i.e., point to the same data).
- **See also**: [`llama_sample_dist::probs_iterator`](#llama_sample_distprobs_iterator)  (Data Structure)


---
#### probs\_iterator::operator\!=<!-- {{#callable:llama_sample_dist::probs_iterator::operator!=}} -->
The `operator!=` function checks if two `probs_iterator` objects are not equal by comparing their `data` pointers.
- **Inputs**:
    - `other`: A reference to another `probs_iterator` object to compare against the current object.
- **Control Flow**:
    - The function compares the `data` pointer of the current `probs_iterator` object with the `data` pointer of the `other` `probs_iterator` object.
    - If the pointers are not equal, the function returns `true`; otherwise, it returns `false`.
- **Output**: A boolean value indicating whether the two `probs_iterator` objects are not equal.
- **See also**: [`llama_sample_dist::probs_iterator`](#llama_sample_distprobs_iterator)  (Data Structure)


---
#### probs\_iterator::operator\*<!-- {{#callable:llama_sample_dist::probs_iterator::operator*}} -->
The `operator*` function returns a constant reference to a float value from the `llama_token_data` structure pointed to by the `data` member of the `probs_iterator`.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `data` member of the `probs_iterator` structure, which is a pointer to a `llama_token_data` object.
    - It returns the `p` member of the `llama_token_data` object, which is a float, as a constant reference.
- **Output**: A constant reference to a float value, specifically the `p` member of the `llama_token_data` structure.
- **See also**: [`llama_sample_dist::probs_iterator`](#llama_sample_distprobs_iterator)  (Data Structure)


---
#### probs\_iterator::operator\+\+<!-- {{#callable:llama_sample_dist::probs_iterator::operator++}} -->
The `operator++` functions in the `probs_iterator` struct increment the iterator to point to the next element in the sequence.
- **Inputs**: None
- **Control Flow**:
    - The prefix `operator++()` increments the `data` pointer and returns a reference to the current iterator.
    - The postfix `operator++(int)` creates a temporary copy of the current iterator, increments the `data` pointer, and returns the temporary copy.
- **Output**: The prefix `operator++()` returns a reference to the incremented iterator, while the postfix `operator++(int)` returns a copy of the iterator before it was incremented.
- **See also**: [`llama_sample_dist::probs_iterator`](#llama_sample_distprobs_iterator)  (Data Structure)


---
#### probs\_iterator::operator\+\+<!-- {{#callable:llama_sample_dist::probs_iterator::operator++}} -->
The `operator++(int)` function for the `probs_iterator` class provides a post-increment operation that returns the current iterator state before advancing the iterator.
- **Inputs**: None
- **Control Flow**:
    - Create a temporary `probs_iterator` object `tmp` and initialize it with the current state of `*this`.
    - Increment the `data` pointer to advance the iterator.
    - Return the temporary `probs_iterator` object `tmp` which holds the state before the increment.
- **Output**: Returns a `probs_iterator` object representing the state of the iterator before it was incremented.
- **See also**: [`llama_sample_dist::probs_iterator`](#llama_sample_distprobs_iterator)  (Data Structure)



---
### llama\_sampler\_dist<!-- {{#data_structure:llama_sampler_dist}} -->
- **Type**: `struct`
- **Members**:
    - `seed`: A constant 32-bit unsigned integer representing the initial seed value for the random number generator.
    - `seed_cur`: A 32-bit unsigned integer representing the current seed value, which can be updated during the lifetime of the struct.
    - `rng`: An instance of the std::mt19937 random number generator, used for generating random numbers.
- **Description**: The `llama_sampler_dist` struct is designed to manage a random number generator for sampling purposes. It holds a constant initial seed (`seed`) and a mutable current seed (`seed_cur`) to allow for reseeding the random number generator (`rng`) as needed. The `rng` is an instance of the Mersenne Twister engine (`std::mt19937`), which is a standard random number generator in C++ known for its high-quality randomness and performance. This struct is likely used in contexts where controlled random sampling is required, such as in probabilistic algorithms or simulations.


---
### llama\_sampler\_top\_k<!-- {{#data_structure:llama_sampler_top_k}} -->
- **Type**: `struct`
- **Members**:
    - `k`: A constant integer representing the number of top elements to consider in sampling.
- **Description**: The `llama_sampler_top_k` struct is a simple data structure used in the context of sampling algorithms, specifically for top-k sampling. It contains a single member, `k`, which is a constant integer that specifies the number of top elements to consider when performing sampling operations. This struct is likely used to configure or initialize a sampling process where only the top `k` elements, based on some criteria, are selected or considered.


---
### llama\_sampler\_top\_p<!-- {{#data_structure:llama_sampler_top_p}} -->
- **Type**: `struct`
- **Members**:
    - `p`: A constant float representing the probability threshold for top-p sampling.
    - `min_keep`: A constant size_t representing the minimum number of tokens to keep during sampling.
- **Description**: The `llama_sampler_top_p` struct is used in the context of top-p sampling, a technique in natural language processing where tokens are selected based on their cumulative probability until a specified threshold `p` is reached. The `min_keep` member ensures that at least a certain number of tokens are retained, even if the cumulative probability threshold is met. This struct is part of a larger system for sampling tokens in a probabilistic manner, allowing for more controlled and varied text generation.


---
### llama\_sampler\_min\_p<!-- {{#data_structure:llama_sampler_min_p}} -->
- **Type**: `struct`
- **Members**:
    - `p`: A constant float representing a probability threshold.
    - `min_keep`: A constant size_t representing the minimum number of tokens to keep.
- **Description**: The `llama_sampler_min_p` struct is a data structure used to define parameters for a sampling strategy in a language model. It contains two members: `p`, which is a probability threshold used to determine the minimum acceptable probability for a token to be considered, and `min_keep`, which specifies the minimum number of tokens that must be retained regardless of their probability. This struct is likely used in a context where tokens are sampled based on their probabilities, ensuring that a certain number of tokens are always kept to maintain diversity or completeness in the output.


---
### llama\_sampler\_typical<!-- {{#data_structure:llama_sampler_typical}} -->
- **Type**: `struct`
- **Members**:
    - `p`: A constant float representing a probability threshold.
    - `min_keep`: A constant size_t representing the minimum number of tokens to keep.
- **Description**: The `llama_sampler_typical` struct is a data structure used in the context of sampling algorithms, specifically for typical sampling. It contains two members: `p`, which is a probability threshold used to determine the typicality of tokens, and `min_keep`, which specifies the minimum number of tokens that should be retained during the sampling process. This struct is likely used to configure the behavior of a sampling algorithm that selects tokens based on their typicality relative to a given probability distribution.


---
### llama\_sampler\_temp<!-- {{#data_structure:llama_sampler_temp}} -->
- **Type**: `struct`
- **Members**:
    - `temp`: A constant float representing the temperature value for the sampler.
- **Description**: The `llama_sampler_temp` struct is a simple data structure that holds a single constant float member named `temp`. This member represents the temperature value used in the sampling process, which is a common parameter in probabilistic models to control the randomness of predictions. The struct is likely used in conjunction with other functions or classes to apply temperature scaling to logits or probabilities in a sampling algorithm.


---
### llama\_sampler\_temp\_ext<!-- {{#data_structure:llama_sampler_temp_ext}} -->
- **Type**: `struct`
- **Members**:
    - `temp`: A constant float representing the temperature value.
    - `delta`: A constant float representing the delta value for temperature adjustment.
    - `exponent`: A constant float representing the exponent used in temperature scaling.
- **Description**: The `llama_sampler_temp_ext` struct is designed to hold parameters for an extended temperature sampling mechanism. It includes three constant float members: `temp`, `delta`, and `exponent`, which are used to control the temperature scaling process in a sampling algorithm. The `temp` member represents the base temperature, `delta` allows for adjustments around this base, and `exponent` is used to apply a power function to the temperature scaling, providing flexibility in how temperature affects the sampling process.


---
### llama\_sampler\_xtc<!-- {{#data_structure:llama_sampler_xtc}} -->
- **Type**: `struct`
- **Members**:
    - `probability`: A constant float representing the probability threshold for sampling.
    - `threshold`: A constant float representing the threshold value for sampling.
    - `min_keep`: A constant size_t indicating the minimum number of tokens to keep.
    - `seed`: A constant uint32_t representing the initial seed for random number generation.
    - `seed_cur`: A uint32_t representing the current seed used in random number generation.
    - `rng`: An instance of std::mt19937 used for random number generation.
- **Description**: The `llama_sampler_xtc` struct is designed to facilitate sampling operations with specific constraints, such as probability thresholds and minimum token retention. It includes parameters for setting a probability and threshold, as well as a minimum number of tokens to keep during sampling. The struct also manages random number generation through a seed and a random number generator instance, allowing for controlled randomness in sampling processes.


---
### llama\_sampler\_mirostat<!-- {{#data_structure:llama_sampler_mirostat}} -->
- **Type**: `struct`
- **Members**:
    - `n_vocab`: Represents the number of vocabulary tokens.
    - `seed`: Initial seed value for random number generation.
    - `seed_cur`: Current seed value for random number generation.
    - `tau`: Target surprise value for the sampler.
    - `eta`: Learning rate for updating the surprise value.
    - `m`: Number of most probable tokens considered for estimating surprise.
    - `mu`: Current surprise value used in the sampling process.
    - `rng`: Random number generator used for sampling.
- **Description**: The `llama_sampler_mirostat` struct is designed to facilitate the Mirostat sampling algorithm, which aims to control the surprise value of generated tokens in a language model. It includes parameters for vocabulary size, random number generation, and the Mirostat algorithm's specific parameters such as target surprise (`tau`), learning rate (`eta`), and the number of tokens (`m`) used for estimating surprise. The struct also maintains a mutable surprise value (`mu`) and a random number generator (`rng`) to support the dynamic sampling process.


---
### llama\_sampler\_mirostat\_v2<!-- {{#data_structure:llama_sampler_mirostat_v2}} -->
- **Type**: `struct`
- **Members**:
    - `seed`: A constant unsigned 32-bit integer representing the initial seed value for random number generation.
    - `seed_cur`: An unsigned 32-bit integer representing the current seed value, which may change over time.
    - `tau`: A constant floating-point value representing a parameter used in the sampling process.
    - `eta`: A constant floating-point value representing the learning rate for updating the parameter mu.
    - `mu`: A floating-point value that is dynamically updated during the sampling process to control the surprise value.
    - `rng`: An instance of the std::mt19937 random number generator used for sampling operations.
- **Description**: The `llama_sampler_mirostat_v2` struct is a data structure used in the Mirostat V2 sampling algorithm, which is designed to control the surprise value of generated tokens in a probabilistic model. It contains parameters such as `tau` and `eta` that influence the sampling behavior, as well as a mutable `mu` value that is adjusted during the sampling process to maintain a target surprise level. The struct also includes a random number generator (`rng`) initialized with a seed (`seed` and `seed_cur`) to ensure reproducibility of the sampling results.


---
### llama\_sampler\_grammar<!-- {{#data_structure:llama_sampler_grammar}} -->
- **Type**: `struct`
- **Members**:
    - `vocab`: A pointer to a llama_vocab structure, representing the vocabulary used.
    - `grammar_str`: A string representing the grammar in use.
    - `grammar_root`: A string representing the root of the grammar.
    - `grammar`: A pointer to a llama_grammar structure, representing the grammar rules.
- **Description**: The `llama_sampler_grammar` struct is designed to encapsulate grammar-related data for a llama sampler. It includes a pointer to a vocabulary structure, strings for the grammar and its root, and a pointer to a grammar structure. This struct is likely used in conjunction with other components to apply grammar rules during the sampling process in a language model.


---
### llama\_sampler\_penalties<!-- {{#data_structure:llama_sampler_penalties}} -->
- **Type**: `struct`
- **Members**:
    - `penalty_last_n`: Specifies the number of recent tokens to consider for applying penalties.
    - `penalty_repeat`: Defines the penalty factor for repeated tokens.
    - `penalty_freq`: Specifies the penalty factor based on token frequency.
    - `penalty_present`: Defines the penalty factor for token presence.
    - `prev`: A ring buffer storing the most recent tokens.
    - `token_count`: An unordered map that counts occurrences of each token.
- **Description**: The `llama_sampler_penalties` struct is designed to manage and apply penalties to token sampling processes in a language model. It includes parameters for penalizing repeated tokens, token frequency, and token presence, which are used to adjust the likelihood of token selection. The struct maintains a history of recent tokens using a ring buffer and tracks token occurrences with an unordered map, allowing for dynamic adjustment of token probabilities based on their recent usage patterns.


---
### llama\_sampler\_top\_n\_sigma<!-- {{#data_structure:llama_sampler_top_n_sigma}} -->
- **Type**: `struct`
- **Members**:
    - `n`: A constant float representing a parameter for the sampler.
- **Description**: The `llama_sampler_top_n_sigma` struct is a simple data structure that contains a single constant float member `n`. This struct is likely used to configure or parameterize a sampling process, specifically in the context of a top-n sigma sampling strategy, where `n` might represent a threshold or scaling factor for determining which elements are included in the sampling process.


---
### llama\_sampler\_dry<!-- {{#data_structure:llama_sampler_dry}} -->
- **Type**: `struct`
- **Members**:
    - `total_context_size`: Represents the total size of the context in which the sampler operates.
    - `dry_multiplier`: A constant multiplier used in the dry sampling process.
    - `dry_base`: The base value used for calculating penalties in the dry sampling process.
    - `dry_allowed_length`: Specifies the allowed length for dry sampling before penalties are applied.
    - `dry_penalty_last_n`: Indicates the number of last tokens to consider for applying penalties.
    - `dry_processed_breakers`: Stores processed token sequences that act as breakers in the dry sampling process.
    - `dry_repeat_count`: Tracks the count of repeated tokens in the context.
    - `dry_max_token_repeat`: Maps tokens to their maximum repeat count in the context.
    - `last_tokens`: A ring buffer that stores the last tokens processed in the context.
- **Description**: The `llama_sampler_dry` struct is designed to manage and apply penalties during the dry sampling process in a token-based context. It includes parameters for controlling the sampling behavior, such as multipliers and base values for penalties, as well as mechanisms for tracking token repetitions and managing restart sequences. The struct utilizes various data structures like unordered maps and vectors to efficiently handle token sequences and repetitions, ensuring that the sampling process adheres to specified constraints and penalties.


---
### llama\_sampler\_logit\_bias<!-- {{#data_structure:llama_sampler_logit_bias}} -->
- **Type**: `struct`
- **Members**:
    - `n_vocab`: Represents the number of vocabulary entries.
    - `logit_bias`: A constant vector of `llama_logit_bias` objects representing biases applied to logits.
    - `to_search`: A vector of `llama_logit_bias` objects used for searching biases not directly mapped by index.
- **Description**: The `llama_sampler_logit_bias` struct is designed to manage and apply biases to logits in a vocabulary-based sampling process. It contains a fixed number of vocabulary entries (`n_vocab`), a vector of logit biases (`logit_bias`) that are directly applied to corresponding tokens, and another vector (`to_search`) for biases that need to be searched and applied to tokens not directly indexed. This structure is essential for adjusting the probability distribution of tokens during sampling by modifying their logits.


---
### llama\_sampler\_infill<!-- {{#data_structure:llama_sampler_infill}} -->
- **Type**: `struct`
- **Members**:
    - `vocab`: A pointer to a `llama_vocab` structure, representing the vocabulary used by the sampler.
    - `buf0`: A vector of characters used as a buffer for processing tokens.
    - `buf1`: Another vector of characters used as a secondary buffer for processing tokens.
- **Description**: The `llama_sampler_infill` struct is designed to facilitate the infill sampling process in a language model. It holds a reference to a vocabulary (`vocab`) and utilizes two character buffers (`buf0` and `buf1`) to manage and process token data during sampling operations. This structure is part of a larger system for token sampling, where it likely plays a role in managing the selection and combination of tokens based on their probabilities and other criteria.


# Functions

---
### llama\_sample\_dist<!-- {{#callable:llama_sample_dist}} -->
Samples a token based on a discrete probability distribution derived from the provided token data.
- **Inputs**:
    - `cur_p`: A pointer to a `llama_token_data_array` structure containing the token data and their associated probabilities.
    - `rng`: A reference to a `std::mt19937` random number generator used to produce random samples.
- **Control Flow**:
    - Defines a nested `probs_iterator` struct to iterate over the probabilities of tokens.
    - Creates a `std::discrete_distribution<int>` object using the `probs_iterator` to represent the probability distribution of the tokens.
    - Calls the `dist` object with the random number generator `rng` to sample a token based on the defined distribution.
- **Output**: Returns an integer representing the index of the sampled token from the `llama_token_data_array`.


---
### llama\_sampler\_temp\_impl<!-- {{#callable:llama_sampler_temp_impl}} -->
The `llama_sampler_temp_impl` function adjusts the logits of tokens based on a specified temperature value.
- **Inputs**:
    - `cur_p`: A pointer to a `llama_token_data_array` structure containing the current token data, including logits.
    - `temp`: A float representing the temperature value used to scale the logits.
- **Control Flow**:
    - If the temperature is less than or equal to zero, the function identifies the token with the highest logit and sets all other logits to negative infinity.
    - It iterates through the `cur_p` array to find the maximum logit and updates the corresponding index while setting others to -INFINITY.
    - If the temperature is greater than zero, it divides each logit in the `cur_p` array by the temperature value.
- **Output**: The function modifies the logits in the `cur_p` array in place, either by setting all but the highest logit to -INFINITY or by scaling all logits by the temperature.


---
### llama\_sampler\_softmax\_impl<!-- {{#callable:llama_sampler_softmax_impl}} -->
The `llama_sampler_softmax_impl` function computes the softmax probabilities for a given array of token data.
- **Inputs**:
    - `cur_p`: A pointer to a `llama_token_data_array` structure that contains the token data, including logits and probabilities.
- **Control Flow**:
    - The function asserts that the size of the `cur_p` array is greater than zero.
    - If the logits are not sorted, it sorts the `cur_p->data` array in descending order based on the logit values.
    - It initializes `max_l` with the highest logit value and `cum_sum` to zero.
    - It iterates over the sorted logits, calculating the exponentiated values and accumulating the sum of these values.
    - Finally, it normalizes the probabilities by dividing each exponentiated value by the cumulative sum.
- **Output**: The function modifies the `cur_p` array in place, setting the `p` field of each token data to its corresponding softmax probability.


---
### llama\_sampler\_top\_k\_impl<!-- {{#callable:llama_sampler_top_k_impl}} -->
The `llama_sampler_top_k_impl` function modifies a given array of token data to retain only the top `k` tokens based on their logit scores.
- **Inputs**:
    - `cur_p`: A pointer to a `llama_token_data_array` structure that contains the current token data including logits.
    - `k`: An integer representing the number of top tokens to retain based on their logit scores.
- **Control Flow**:
    - The function first checks if `k` is less than or equal to zero, returning immediately if true.
    - It then adjusts `k` to be the minimum of the provided `k` and the size of the token data array.
    - If the token data is not already sorted, it sorts the tokens based on their logit scores in descending order.
    - For `k` values less than or equal to 128, it uses `std::partial_sort` to sort the top `k` tokens directly.
    - For larger values of `k`, it employs a bucket sort strategy to efficiently categorize and sort the logits.
    - After sorting, it copies the top `k` tokens back into the original array and updates the size of the token data array.
- **Output**: The function modifies the `cur_p` array in place, retaining only the top `k` tokens based on their logit scores and updating the size of the array to `k`.


---
### get\_rng\_seed<!-- {{#callable:get_rng_seed}} -->
Generates a random seed for a random number generator based on the provided seed.
- **Inputs**:
    - `seed`: A `uint32_t` value representing the seed for the random number generator.
- **Control Flow**:
    - Checks if the input `seed` is equal to `LLAMA_DEFAULT_SEED`.
    - If it is, it checks if `std::random_device` is a true random number generator by evaluating its entropy.
    - If `std::random_device` is not a true RNG, it returns the current time since epoch as a seed.
    - If `std::random_device` is a true RNG, it generates and returns a random number using `std::random_device`.
- **Output**: Returns a `uint32_t` value that serves as the seed for the random number generator.


---
### llama\_sampler\_init<!-- {{#callable:llama_sampler_init}} -->
Initializes a new `llama_sampler` instance with a specified interface and context.
- **Inputs**:
    - `iface`: A pointer to a `llama_sampler_i` structure that defines the interface for the sampler.
    - `ctx`: A context of type `llama_sampler_context_t` that holds additional information for the sampler.
- **Control Flow**:
    - Allocates memory for a new `llama_sampler` object using the `new` operator.
    - Initializes the `iface` and `ctx` members of the `llama_sampler` structure with the provided arguments.
- **Output**: Returns a pointer to the newly created `llama_sampler` instance.


---
### llama\_sampler\_name<!-- {{#callable:llama_sampler_name}} -->
Returns the name of the sampler interface associated with the given sampler.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the sampler interface.
- **Control Flow**:
    - Checks if the `iface` member of the `llama_sampler` structure is null.
    - If `iface` is null, returns the string '(null)'.
    - If `iface` is not null, calls the `name` method of the `iface` and returns its result.
- **Output**: Returns a pointer to a string containing the name of the sampler interface, or '(null)' if the interface is not set.


---
### llama\_sampler\_accept<!-- {{#callable:llama_sampler_accept}} -->
The `llama_sampler_accept` function accepts a token for a given sampler interface if the accept method is defined.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the sampler interface.
    - `token`: A `llama_token` representing the token to be accepted by the sampler.
- **Control Flow**:
    - The function first checks if the `accept` method is defined in the sampler's interface.
    - If the `accept` method is defined, it is called with the sampler and the token as arguments.
- **Output**: The function does not return a value; it performs an action based on the acceptance of the token.


---
### llama\_sampler\_apply<!-- {{#callable:llama_sampler_apply}} -->
The `llama_sampler_apply` function applies a sampling method defined in the sampler interface to a given token data array.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the sampling method to be applied.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data to which the sampling method will be applied.
- **Control Flow**:
    - The function asserts that the `apply` method of the sampler interface is not null using `GGML_ASSERT`.
    - It then calls the `apply` method of the sampler interface, passing the `smpl` and `cur_p` as arguments.
- **Output**: The function does not return a value; it modifies the `cur_p` token data array in place based on the sampling method applied.


---
### llama\_sampler\_reset<!-- {{#callable:llama_sampler_reset}} -->
Resets the state of the `llama_sampler` if a reset function is defined.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the sampler's state and interface.
- **Control Flow**:
    - Checks if the `reset` function is defined in the `iface` of the `llama_sampler`.
    - If defined, it calls the `reset` function, passing the `llama_sampler` pointer.
- **Output**: This function does not return a value; it performs an action based on the state of the sampler.


---
### llama\_sampler\_clone<!-- {{#callable:llama_sampler_clone}} -->
Clones a `llama_sampler` instance, if supported, or initializes a new instance with the same interface.
- **Inputs**:
    - `smpl`: A pointer to the `llama_sampler` instance to be cloned.
- **Control Flow**:
    - Checks if the `clone` function is defined in the `iface` of the provided `llama_sampler` instance.
    - If the `clone` function exists, it calls this function to create a clone of the sampler.
    - If the `ctx` of the sampler is `nullptr`, it initializes a new sampler with the same interface and a `nullptr` context.
    - If cloning is not supported and the context is not `nullptr`, it aborts the operation with an error message.
- **Output**: Returns a pointer to the cloned `llama_sampler` instance, or a newly initialized sampler if cloning is not supported.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_free<!-- {{#callable:llama_sampler_free}} -->
The `llama_sampler_free` function deallocates memory associated with a `llama_sampler` instance.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that needs to be freed.
- **Control Flow**:
    - The function first checks if the `smpl` pointer is null; if it is, the function returns immediately without performing any operations.
    - If the `smpl` pointer is not null, it checks if the `free` function pointer in the `iface` structure of `smpl` is not null.
    - If the `free` function pointer is valid, it calls this function, passing `smpl` to it, allowing for any custom cleanup defined by the sampler interface.
    - Finally, the function deletes the `smpl` pointer, releasing the memory allocated for the `llama_sampler` instance.
- **Output**: The function does not return a value; it performs memory deallocation and cleanup.


---
### llama\_sampler\_sample<!-- {{#callable:llama_sampler_sample}} -->
Samples a token from the llama model's vocabulary based on the logits provided.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the sampling strategy.
    - `ctx`: A pointer to a `llama_context` structure that holds the model context.
    - `idx`: An integer index indicating which logits to sample from.
- **Control Flow**:
    - Retrieve the logits for the specified index from the context using `llama_get_logits_ith`.
    - Obtain the model and vocabulary from the context.
    - Determine the number of tokens in the vocabulary.
    - Allocate a vector to hold token data and populate it with token IDs and their corresponding logits.
    - Create a `llama_token_data_array` structure to hold the current token data.
    - Apply the sampling strategy using [`llama_sampler_apply`](#llama_sampler_apply).
    - Assert that a valid token has been selected from the sampling process.
    - Retrieve the selected token ID from the current token data.
    - Accept the selected token using [`llama_sampler_accept`](#llama_sampler_accept).
    - Return the selected token ID.
- **Output**: Returns the ID of the sampled token from the vocabulary.
- **Functions called**:
    - [`llama_sampler_apply`](#llama_sampler_apply)
    - [`llama_sampler_accept`](#llama_sampler_accept)


---
### llama\_sampler\_chain\_name<!-- {{#callable:llama_sampler_chain_name}} -->
The `llama_sampler_chain_name` function returns the name of the sampler chain as a constant string.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements as it directly returns a string.
- **Output**: The function outputs a constant string 'chain' which represents the name of the sampler chain.


---
### llama\_sampler\_chain\_accept<!-- {{#callable:llama_sampler_chain_accept}} -->
The `llama_sampler_chain_accept` function accepts a token and updates all samplers in a chain.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure representing the sampler chain.
    - `token`: A `llama_token` representing the token to be accepted by the samplers.
- **Control Flow**:
    - The function retrieves the `llama_sampler_chain` context from the provided `llama_sampler` pointer.
    - A timing measurement is initiated to track sampling performance.
    - The function iterates over each sampler in the chain, calling [`llama_sampler_accept`](#llama_sampler_accept) for each with the provided token.
    - The sample count for the chain is incremented.
- **Output**: The function does not return a value; it modifies the state of the samplers in the chain.
- **Functions called**:
    - [`llama_sampler_accept`](#llama_sampler_accept)


---
### llama\_sampler\_chain\_apply<!-- {{#callable:llama_sampler_chain_apply}} -->
The `llama_sampler_chain_apply` function applies a series of samplers in a chain to a given token data array.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure representing the sampler chain.
    - `cur_p`: A pointer to a `llama_token_data_array` structure containing the current token data to be processed.
- **Control Flow**:
    - The function retrieves the `llama_sampler_chain` context from the provided `llama_sampler` pointer.
    - A timing measurement is initiated to track the sampling duration.
    - The function iterates over each sampler in the chain, applying each sampler to the `cur_p` token data array using the [`llama_sampler_apply`](#llama_sampler_apply) function.
- **Output**: The function does not return a value; it modifies the `cur_p` token data array in place based on the applied samplers.
- **Functions called**:
    - [`llama_sampler_apply`](#llama_sampler_apply)


---
### llama\_sampler\_chain\_reset<!-- {{#callable:llama_sampler_chain_reset}} -->
Resets the state of all samplers in a `llama_sampler_chain`.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that represents the sampler chain to be reset.
- **Control Flow**:
    - Retrieve the `llama_sampler_chain` context from the provided `llama_sampler` pointer.
    - Iterate over each sampler in the chain's `samplers` vector.
    - Call the [`llama_sampler_reset`](#llama_sampler_reset) function on each sampler to reset its state.
    - Reset the total sample time (`t_sample_us`) to zero.
    - Reset the sample count (`n_sample`) to zero.
- **Output**: This function does not return a value; it modifies the state of the samplers in the chain directly.
- **Functions called**:
    - [`llama_sampler_reset`](#llama_sampler_reset)


---
### llama\_sampler\_chain\_clone<!-- {{#callable:llama_sampler_chain_clone}} -->
Clones a `llama_sampler_chain` by creating a new instance and copying its samplers.
- **Inputs**:
    - `smpl`: A pointer to the source `llama_sampler` that is to be cloned.
- **Control Flow**:
    - The function retrieves the source sampler chain from the input `smpl`.
    - It initializes a new sampler chain using the parameters from the source chain.
    - It iterates over each sampler in the source chain, cloning each one and adding it to the new chain.
    - Finally, it returns the newly created sampler chain.
- **Output**: Returns a pointer to the newly cloned `llama_sampler` chain.
- **Functions called**:
    - [`llama_sampler_chain_init`](#llama_sampler_chain_init)
    - [`llama_sampler_chain_add`](#llama_sampler_chain_add)
    - [`llama_sampler_clone`](#llama_sampler_clone)


---
### llama\_sampler\_chain\_free<!-- {{#callable:llama_sampler_chain_free}} -->
The `llama_sampler_chain_free` function deallocates memory associated with a `llama_sampler_chain` and its contained samplers.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which contains a context pointer to a `llama_sampler_chain`.
- **Control Flow**:
    - The function retrieves the `llama_sampler_chain` from the `smpl` context.
    - It iterates over each sampler in the `chain->samplers` vector and calls [`llama_sampler_free`](#llama_sampler_free) on each sampler to free their resources.
    - Finally, it deletes the `chain` itself to free the memory allocated for the `llama_sampler_chain`.
- **Output**: This function does not return a value; it performs cleanup and deallocation of resources.
- **Functions called**:
    - [`llama_sampler_free`](#llama_sampler_free)


---
### llama\_sampler\_chain\_init<!-- {{#callable:llama_sampler_chain_init}} -->
Initializes a `llama_sampler` instance for a chain of samplers.
- **Inputs**:
    - `params`: A `llama_sampler_chain_params` structure containing parameters for the sampler chain.
- **Control Flow**:
    - Calls [`llama_sampler_init`](#llama_sampler_init) to create a new `llama_sampler` instance.
    - Passes a pointer to the `llama_sampler_chain_i` interface and a newly allocated `llama_sampler_chain` structure initialized with the provided parameters.
    - The `llama_sampler_chain` structure is initialized with an empty list of samplers, and counters for sampling time and number of samples set to zero.
- **Output**: Returns a pointer to the newly initialized `llama_sampler` instance.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_chain\_add<!-- {{#callable:llama_sampler_chain_add}} -->
Adds a new `llama_sampler` to the `llama_sampler_chain`.
- **Inputs**:
    - `chain`: A pointer to a `llama_sampler` structure representing the chain to which the sampler will be added.
    - `smpl`: A pointer to a `llama_sampler` structure representing the sampler to be added to the chain.
- **Control Flow**:
    - The function retrieves the context of the `chain` by casting its `ctx` member to a `llama_sampler_chain` pointer.
    - It then adds the `smpl` sampler to the `samplers` vector of the `llama_sampler_chain`.
- **Output**: The function does not return a value; it modifies the internal state of the `llama_sampler_chain` by adding the specified sampler.


---
### llama\_sampler\_chain\_get<!-- {{#callable:llama_sampler_chain_get}} -->
Retrieves a specific `llama_sampler` from a `llama_sampler_chain` based on the provided index.
- **Inputs**:
    - `chain`: A pointer to a `llama_sampler` structure representing the chain from which to retrieve the sampler.
    - `i`: An integer index specifying the position of the sampler to retrieve from the chain.
- **Control Flow**:
    - The function casts the `ctx` member of the `chain` to a `llama_sampler_chain` type to access the underlying data.
    - It checks if the provided index `i` is out of bounds, either negative or greater than or equal to the size of the `samplers` vector.
    - If the index is invalid, the function returns a null pointer.
    - If the index is valid, it retrieves and returns the sampler at the specified index from the `samplers` vector.
- **Output**: Returns a pointer to the `llama_sampler` at the specified index if valid, otherwise returns nullptr.


---
### llama\_sampler\_chain\_remove<!-- {{#callable:llama_sampler_chain_remove}} -->
Removes a `llama_sampler` from a `llama_sampler_chain` at a specified index.
- **Inputs**:
    - `chain`: A pointer to the `llama_sampler` chain from which a sampler will be removed.
    - `i`: An integer index specifying the position of the sampler to be removed from the chain.
- **Control Flow**:
    - The function first casts the `chain` pointer to a `llama_sampler_chain` type to access its context.
    - It checks if the provided index `i` is valid (i.e., not negative and within the bounds of the `samplers` vector).
    - If the index is invalid, the function returns a null pointer.
    - If the index is valid, it retrieves the sampler at that index, removes it from the `samplers` vector, and returns the removed sampler.
- **Output**: Returns a pointer to the removed `llama_sampler` if successful, or null if the index was invalid.


---
### llama\_sampler\_chain\_n<!-- {{#callable:llama_sampler_chain_n}} -->
Returns the number of samplers in a `llama_sampler_chain`.
- **Inputs**:
    - `chain`: A pointer to a `llama_sampler` structure that represents a sampler chain.
- **Control Flow**:
    - The function casts the `ctx` member of the `chain` to a pointer of type `llama_sampler_chain`.
    - It accesses the `samplers` vector from the `llama_sampler_chain` structure.
    - Finally, it returns the size of the `samplers` vector.
- **Output**: An integer representing the number of samplers in the chain.


---
### llama\_sampler\_greedy\_name<!-- {{#callable:llama_sampler_greedy_name}} -->
The `llama_sampler_greedy_name` function returns the name of the greedy sampling strategy.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which is not used in this function.
- **Control Flow**:
    - The function does not contain any control flow statements such as loops or conditionals.
    - It directly returns a string literal.
- **Output**: The function outputs a constant string "greedy", representing the name of the sampling strategy.


---
### llama\_sampler\_greedy\_apply<!-- {{#callable:llama_sampler_greedy_apply}} -->
The `llama_sampler_greedy_apply` function selects the token with the highest logit value from a given array of token data.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which is not used in this function.
    - `cur_p`: A pointer to a `llama_token_data_array` structure containing the token data from which the highest logit will be selected.
- **Control Flow**:
    - The function initializes the `selected` index of `cur_p` to 0, assuming the first token is the highest initially.
    - It then iterates through the `cur_p->data` array starting from the second token (index 1).
    - For each token, it compares its logit value with the logit of the currently selected token.
    - If a token with a higher logit is found, it updates the `selected` index to the current token's index.
- **Output**: The function does not return a value; instead, it modifies the `cur_p` structure to indicate which token has been selected based on the highest logit.


---
### llama\_sampler\_init\_greedy<!-- {{#callable:llama_sampler_init_greedy}} -->
Initializes a `llama_sampler` instance using a greedy sampling strategy.
- **Inputs**:
    - `none`: This function does not take any input parameters.
- **Control Flow**:
    - Calls the [`llama_sampler_init`](#llama_sampler_init) function with a specific interface for greedy sampling and a null context.
    - The interface used is `llama_sampler_greedy_i`, which defines the behavior of the greedy sampler.
- **Output**: Returns a pointer to a newly initialized `llama_sampler` instance configured for greedy sampling.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_dist\_name<!-- {{#callable:llama_sampler_dist_name}} -->
The `llama_sampler_dist_name` function returns the name of the distribution sampler.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a string literal 'dist'.
- **Output**: The output is a constant string 'dist', representing the name of the distribution sampler.


---
### llama\_sampler\_dist\_apply<!-- {{#callable:llama_sampler_dist_apply}} -->
The `llama_sampler_dist_apply` function applies a sampling distribution to a given token data array.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context and state for the sampling process.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data, including logits and probabilities.
- **Control Flow**:
    - The function retrieves the context from the `llama_sampler` structure, specifically casting it to `llama_sampler_dist`.
    - It calls [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl) to compute the softmax probabilities from the logits in `cur_p`.
    - Finally, it samples a token based on the computed probabilities using [`llama_sample_dist`](#llama_sample_dist), storing the selected token index back in `cur_p->selected`.
- **Output**: The function does not return a value; instead, it modifies the `cur_p` structure to indicate which token was selected based on the sampling distribution.
- **Functions called**:
    - [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl)
    - [`llama_sample_dist`](#llama_sample_dist)


---
### llama\_sampler\_dist\_clone<!-- {{#callable:llama_sampler_dist_clone}} -->
Clones a `llama_sampler_dist` instance, copying its state.
- **Inputs**:
    - `smpl`: A pointer to the source `llama_sampler` instance that is to be cloned.
- **Control Flow**:
    - The function casts the `ctx` of the input `smpl` to a `llama_sampler_dist` type.
    - It initializes a new `llama_sampler` instance using the [`llama_sampler_init_dist`](#llama_sampler_init_dist) function with the seed from the original sampler's context.
    - The state of the original sampler's context is copied to the new sampler's context, specifically the random number generator (RNG).
    - Finally, the newly created sampler is returned.
- **Output**: Returns a pointer to a new `llama_sampler` instance that is a clone of the input sampler.
- **Functions called**:
    - [`llama_sampler_init_dist`](#llama_sampler_init_dist)


---
### llama\_sampler\_dist\_reset<!-- {{#callable:llama_sampler_dist_reset}} -->
Resets the random number generator state for the `llama_sampler`.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context for the sampler.
- **Control Flow**:
    - Retrieve the context of the sampler by casting `smpl->ctx` to `llama_sampler_dist`.
    - Generate a new seed using the [`get_rng_seed`](#get_rng_seed) function based on the original seed stored in the context.
    - Set the current seed of the context to the newly generated seed.
    - Seed the random number generator (`rng`) with the current seed.
- **Output**: The function does not return a value; it modifies the state of the `llama_sampler` by resetting its random number generator.
- **Functions called**:
    - [`get_rng_seed`](#get_rng_seed)


---
### llama\_sampler\_dist\_free<!-- {{#callable:llama_sampler_dist_free}} -->
The `llama_sampler_dist_free` function deallocates the context associated with a `llama_sampler` instance.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure whose context needs to be freed.
- **Control Flow**:
    - The function takes a pointer to a `llama_sampler` structure as input.
    - It accesses the `ctx` member of the `llama_sampler` structure, which is expected to be a pointer to a `llama_sampler_dist` structure.
    - The function then calls `delete` on the `ctx` pointer to free the allocated memory.
- **Output**: This function does not return a value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_dist<!-- {{#callable:llama_sampler_init_dist}} -->
Initializes a `llama_sampler` for distribution sampling with a specified random seed.
- **Inputs**:
    - `seed`: A 32-bit unsigned integer used to initialize the random number generator.
- **Control Flow**:
    - Calls [`get_rng_seed`](#get_rng_seed) to obtain a current random seed based on the provided seed.
    - Creates a new instance of `llama_sampler_dist` with the original seed, current seed, and a random number generator initialized with the current seed.
    - Calls [`llama_sampler_init`](#llama_sampler_init) to initialize the sampler interface with the created context.
- **Output**: Returns a pointer to a newly initialized `llama_sampler` instance configured for distribution sampling.
- **Functions called**:
    - [`get_rng_seed`](#get_rng_seed)
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_softmax\_name<!-- {{#callable:llama_sampler_softmax_name}} -->
The `llama_sampler_softmax_name` function returns the name of the softmax sampler.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a string literal 'softmax'.
- **Output**: The output is a constant string 'softmax', representing the name of the sampler.


---
### llama\_sampler\_softmax\_apply<!-- {{#callable:llama_sampler_softmax_apply}} -->
Applies the softmax function to the logits in the `cur_p` token data array.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which is not used in this function.
    - `cur_p`: A pointer to a `llama_token_data_array` structure containing the logits to be processed.
- **Control Flow**:
    - Calls the [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl) function, passing the `cur_p` argument to it.
    - The [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl) function handles the actual computation of the softmax probabilities.
- **Output**: This function does not return a value; it modifies the `cur_p` structure in place to contain the softmax probabilities.
- **Functions called**:
    - [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl)


---
### llama\_sampler\_init\_softmax<!-- {{#callable:llama_sampler_init_softmax}} -->
Initializes a `llama_sampler` instance using the softmax sampling interface.
- **Inputs**:
    - `none`: This function does not take any input parameters.
- **Control Flow**:
    - Calls the [`llama_sampler_init`](#llama_sampler_init) function with the softmax interface and a null context.
    - Returns the initialized `llama_sampler` instance.
- **Output**: Returns a pointer to a newly created `llama_sampler` instance configured for softmax sampling.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_top\_k\_name<!-- {{#callable:llama_sampler_top_k_name}} -->
The `llama_sampler_top_k_name` function returns the name of the top-k sampling method.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements as it directly returns a string.
- **Output**: The function outputs a constant string 'top-k' which represents the name of the sampling method.


---
### llama\_sampler\_top\_k\_apply<!-- {{#callable:llama_sampler_top_k_apply}} -->
Applies the top-k sampling strategy to the current token data array.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the sampling context.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data to be processed.
- **Control Flow**:
    - The function retrieves the context from the `llama_sampler` structure, specifically casting it to `llama_sampler_top_k`.
    - It then calls the [`llama_sampler_top_k_impl`](#llama_sampler_top_k_impl) function, passing the current token data array and the value of k from the context.
- **Output**: The function does not return a value; it modifies the `cur_p` array in place based on the top-k sampling strategy.
- **Functions called**:
    - [`llama_sampler_top_k_impl`](#llama_sampler_top_k_impl)


---
### llama\_sampler\_top\_k\_clone<!-- {{#callable:llama_sampler_top_k_clone}} -->
Clones a `llama_sampler_top_k` instance by initializing a new sampler with the same top-k value.
- **Inputs**:
    - `smpl`: A pointer to a constant `llama_sampler` structure that contains the context for the sampler to be cloned.
- **Control Flow**:
    - The function retrieves the context of the input sampler `smpl` and casts it to a `llama_sampler_top_k` type.
    - It then calls [`llama_sampler_init_top_k`](#llama_sampler_init_top_k) with the `k` value from the context to create a new sampler instance.
- **Output**: Returns a pointer to a new `llama_sampler` instance initialized with the same top-k value as the input sampler.
- **Functions called**:
    - [`llama_sampler_init_top_k`](#llama_sampler_init_top_k)


---
### llama\_sampler\_top\_k\_free<!-- {{#callable:llama_sampler_top_k_free}} -->
The `llama_sampler_top_k_free` function deallocates the memory associated with the `llama_sampler_top_k` context.
- **Inputs**: None
- **Control Flow**:
    - The function directly calls the `delete` operator on the `ctx` member of the `llama_sampler` structure, which is cast to `llama_sampler_top_k`.
- **Output**: The function does not return any value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_top\_k<!-- {{#callable:llama_sampler_init_top_k}} -->
Initializes a `llama_sampler` for top-k sampling.
- **Inputs**:
    - `k`: An integer representing the number of top tokens to consider during sampling.
- **Control Flow**:
    - Calls the [`llama_sampler_init`](#llama_sampler_init) function to create a new sampler instance.
    - Passes a pointer to the `llama_sampler_top_k_i` interface and a new instance of `llama_sampler_top_k` containing the value of `k`.
- **Output**: Returns a pointer to a newly initialized `llama_sampler` configured for top-k sampling.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_top\_p\_name<!-- {{#callable:llama_sampler_top_p_name}} -->
The `llama_sampler_top_p_name` function returns the name of the top-p sampling method.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements as it directly returns a string.
- **Output**: The function outputs a constant string "top-p" which represents the name of the sampling method.


---
### llama\_sampler\_top\_p\_apply<!-- {{#callable:llama_sampler_top_p_apply}} -->
The `llama_sampler_top_p_apply` function applies top-p sampling to a given token data array.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context for the sampling method.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token probabilities and their associated data.
- **Control Flow**:
    - Check if the top-p value `p` is greater than or equal to 1.0; if so, exit the function early.
    - Call [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl) to compute the softmax probabilities for the tokens in `cur_p`.
    - Initialize a cumulative sum variable `cum_sum` to 0.0 and set `last_idx` to the size of `cur_p`.
    - Iterate over the tokens in `cur_p`, updating `cum_sum` with the probability of each token.
    - If `cum_sum` exceeds `p` and the number of kept tokens is at least `min_keep`, update `last_idx` to include the current token.
    - After the loop, resize `cur_p` to keep only the top-p tokens based on `last_idx`.
- **Output**: The function modifies the `cur_p` array in place to retain only the top-p tokens based on their cumulative probabilities.
- **Functions called**:
    - [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl)


---
### llama\_sampler\_top\_p\_clone<!-- {{#callable:llama_sampler_top_p_clone}} -->
Clones a `llama_sampler` object using the top-p sampling strategy.
- **Inputs**:
    - `smpl`: A pointer to a constant `llama_sampler` structure that contains the context and parameters for the sampling strategy.
- **Control Flow**:
    - The function retrieves the context of the input sampler `smpl` and casts it to a `llama_sampler_top_p` type.
    - It then calls the [`llama_sampler_init_top_p`](#llama_sampler_init_top_p) function with the parameters `p` and `min_keep` from the context to create a new sampler instance.
    - Finally, it returns the newly created sampler instance.
- **Output**: Returns a pointer to a new `llama_sampler` instance that is a clone of the original sampler, configured for top-p sampling.
- **Functions called**:
    - [`llama_sampler_init_top_p`](#llama_sampler_init_top_p)


---
### llama\_sampler\_top\_p\_free<!-- {{#callable:llama_sampler_top_p_free}} -->
The `llama_sampler_top_p_free` function deallocates the context associated with a `llama_sampler` instance.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure whose context needs to be freed.
- **Control Flow**:
    - The function takes a pointer to a `llama_sampler` structure as input.
    - It accesses the `ctx` member of the `llama_sampler` structure, which is expected to be of type `llama_sampler_top_p`.
    - The function then deletes the context object pointed to by `ctx`, effectively freeing the allocated memory.
- **Output**: The function does not return any value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_top\_p<!-- {{#callable:llama_sampler_init_top_p}} -->
Initializes a `llama_sampler` for top-p sampling with specified parameters.
- **Inputs**:
    - `p`: A float representing the cumulative probability threshold for top-p sampling.
    - `min_keep`: A size_t indicating the minimum number of tokens to keep regardless of the probability threshold.
- **Control Flow**:
    - Calls [`llama_sampler_init`](#llama_sampler_init) to create a new `llama_sampler` instance.
    - Passes a pointer to the `llama_sampler_top_p_i` interface and a new instance of `llama_sampler_top_p` containing the parameters `p` and `min_keep`.
- **Output**: Returns a pointer to a newly initialized `llama_sampler` configured for top-p sampling.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_min\_p\_name<!-- {{#callable:llama_sampler_min_p_name}} -->
The `llama_sampler_min_p_name` function returns the name of the minimum probability sampler.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a string literal without any conditional logic or iterations.
- **Output**: The output is a constant string "min-p" which represents the name of the sampler.


---
### llama\_sampler\_min\_p\_apply<!-- {{#callable:llama_sampler_min_p_apply}} -->
The `llama_sampler_min_p_apply` function applies a minimum probability filter to a given array of token data.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context and parameters for the sampling.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data to be filtered.
- **Control Flow**:
    - Check if the minimum probability `p` is less than or equal to 0 or if the size of `cur_p` is zero; if so, return immediately.
    - Initialize a boolean flag `min_p_applied` to track if the minimum probability filter was successfully applied.
    - If `cur_p` is not sorted, attempt to filter tokens based on the unsorted implementation by calculating the maximum logit and determining a minimum logit threshold.
    - If enough tokens meet the minimum logit criteria, update `cur_p` with the filtered tokens and set `min_p_applied` to true.
    - If the unsorted implementation fails or `cur_p` is sorted, sort the tokens by logit in descending order and apply the minimum logit filter again.
    - Resize `cur_p` to keep only the tokens that meet the minimum logit criteria.
- **Output**: The function modifies the `cur_p` array in place, filtering out tokens that do not meet the minimum probability criteria, and updates its size accordingly.


---
### llama\_sampler\_min\_p\_clone<!-- {{#callable:llama_sampler_min_p_clone}} -->
The `llama_sampler_min_p_clone` function creates a clone of a `llama_sampler` object specifically for the `min-p` sampling strategy.
- **Inputs**:
    - `smpl`: A pointer to a constant `llama_sampler` structure that contains the context and parameters for the sampling strategy.
- **Control Flow**:
    - The function retrieves the context of the input sampler `smpl` and casts it to a pointer of type `llama_sampler_min_p`.
    - It then calls the [`llama_sampler_init_min_p`](#llama_sampler_init_min_p) function with the parameters extracted from the context to create a new sampler instance.
- **Output**: Returns a pointer to a new `llama_sampler` instance initialized with the parameters from the original sampler's context.
- **Functions called**:
    - [`llama_sampler_init_min_p`](#llama_sampler_init_min_p)


---
### llama\_sampler\_min\_p\_free<!-- {{#callable:llama_sampler_min_p_free}} -->
The `llama_sampler_min_p_free` function deallocates the context associated with a `llama_sampler` instance.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure whose context needs to be freed.
- **Control Flow**:
    - The function takes a pointer to a `llama_sampler` structure as input.
    - It accesses the `ctx` member of the `llama_sampler` structure.
    - It casts the `ctx` pointer to a `llama_sampler_min_p` type and deletes it.
- **Output**: This function does not return a value; it performs a cleanup operation by freeing allocated memory.


---
### llama\_sampler\_init\_min\_p<!-- {{#callable:llama_sampler_init_min_p}} -->
Initializes a `llama_sampler` instance with a minimum probability threshold and a minimum number of tokens to keep.
- **Inputs**:
    - `p`: A float representing the minimum probability threshold for token selection.
    - `min_keep`: A size_t value indicating the minimum number of tokens that must be kept after applying the sampling method.
- **Control Flow**:
    - Calls [`llama_sampler_init`](#llama_sampler_init) function to create a new `llama_sampler` instance.
    - Passes a pointer to the `llama_sampler_min_p_i` interface and a new instance of `llama_sampler_min_p` containing the parameters `p` and `min_keep`.
- **Output**: Returns a pointer to the newly initialized `llama_sampler` instance.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_typical\_name<!-- {{#callable:llama_sampler_typical_name}} -->
The `llama_sampler_typical_name` function returns a constant string representing the name of the typical sampling method.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements as it directly returns a string.
- **Output**: The function outputs a constant string 'typical'.


---
### llama\_sampler\_typical\_apply<!-- {{#callable:llama_sampler_typical_apply}} -->
The `llama_sampler_typical_apply` function applies typical sampling to a given set of token probabilities.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context and parameters for the sampling.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token probabilities and their associated data.
- **Control Flow**:
    - Check if the probability threshold `ctx->p` is greater than or equal to 1.0; if so, exit the function early.
    - Compute the softmax of the logits in `cur_p` to convert them into probabilities.
    - Calculate the entropy of the probability distribution using the formula for entropy.
    - Compute the absolute difference between the negative log probabilities and the calculated entropy for each token, storing these in a vector of shifted scores.
    - Sort the tokens based on their shifted scores while maintaining their original indices.
    - Iterate through the sorted indices to compute cumulative probabilities until the cumulative sum exceeds `ctx->p` or the minimum number of tokens to keep is reached.
    - Create a new vector of token data containing only the tokens that are considered locally typical based on the cumulative probabilities.
    - Replace the original data in `cur_p` with the new filtered data and update its size.
- **Output**: The function modifies the `cur_p` structure in place, updating it to contain only the locally typical tokens based on the sampling criteria.
- **Functions called**:
    - [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl)


---
### llama\_sampler\_typical\_clone<!-- {{#callable:llama_sampler_typical_clone}} -->
Creates a clone of a `llama_sampler_typical` instance.
- **Inputs**:
    - `smpl`: A pointer to a constant `llama_sampler` structure that contains the context of the sampler to be cloned.
- **Control Flow**:
    - The function casts the `ctx` member of the input `smpl` to a pointer of type `const llama_sampler_typical`.
    - It then calls the [`llama_sampler_init_typical`](#llama_sampler_init_typical) function with the parameters extracted from the `ctx` to create a new sampler instance.
- **Output**: Returns a pointer to a new `llama_sampler` instance that is a clone of the original sampler.
- **Functions called**:
    - [`llama_sampler_init_typical`](#llama_sampler_init_typical)


---
### llama\_sampler\_typical\_free<!-- {{#callable:llama_sampler_typical_free}} -->
The `llama_sampler_typical_free` function deallocates the context associated with a `llama_sampler` instance.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context to be freed.
- **Control Flow**:
    - The function takes a pointer to a `llama_sampler` structure as input.
    - It accesses the `ctx` member of the `llama_sampler` structure, which is expected to be a pointer to a `llama_sampler_typical` structure.
    - The function then calls `delete` on the `ctx` pointer to free the allocated memory.
- **Output**: This function does not return a value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_typical<!-- {{#callable:llama_sampler_init_typical}} -->
Initializes a `llama_sampler` instance with typical sampling parameters.
- **Inputs**:
    - `p`: A floating-point value representing the probability threshold for typical sampling.
    - `min_keep`: A size_t value indicating the minimum number of tokens to keep during sampling.
- **Control Flow**:
    - Calls the [`llama_sampler_init`](#llama_sampler_init) function to create a new `llama_sampler` instance.
    - Passes a pointer to the `llama_sampler_typical_i` interface and a new instance of `llama_sampler_typical` initialized with the provided parameters.
- **Output**: Returns a pointer to the newly created `llama_sampler` instance.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_temp\_name<!-- {{#callable:llama_sampler_temp_name}} -->
The `llama_sampler_temp_name` function returns a constant string representing the name of the temperature sampler.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which is not used in this function.
- **Control Flow**:
    - The function does not contain any control flow statements such as loops or conditionals.
    - It directly returns a constant string.
- **Output**: The function outputs a constant string 'temp', which signifies the name of the temperature sampler.


---
### llama\_sampler\_temp\_apply<!-- {{#callable:llama_sampler_temp_apply}} -->
Applies temperature scaling to the logits of the current token data array.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context for the sampler.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data to which temperature scaling will be applied.
- **Control Flow**:
    - The function retrieves the temperature context from the `llama_sampler` structure.
    - It then calls the [`llama_sampler_temp_impl`](#llama_sampler_temp_impl) function, passing the current token data array and the temperature value.
- **Output**: The function does not return a value; it modifies the logits of the `cur_p` token data array in place based on the specified temperature.
- **Functions called**:
    - [`llama_sampler_temp_impl`](#llama_sampler_temp_impl)


---
### llama\_sampler\_temp\_clone<!-- {{#callable:llama_sampler_temp_clone}} -->
Clones a temporary `llama_sampler` instance from an existing one.
- **Inputs**:
    - `smpl`: A pointer to the original `llama_sampler` instance that is to be cloned.
- **Control Flow**:
    - The function retrieves the context from the original sampler by casting `smpl->ctx` to `const llama_sampler_temp*`.
    - It then calls [`llama_sampler_init_temp`](#llama_sampler_init_temp) with the temperature value from the context to create a new sampler instance.
- **Output**: Returns a pointer to the newly created `llama_sampler` instance that is a clone of the original.
- **Functions called**:
    - [`llama_sampler_init_temp`](#llama_sampler_init_temp)


---
### llama\_sampler\_temp\_free<!-- {{#callable:llama_sampler_temp_free}} -->
The `llama_sampler_temp_free` function deallocates the memory associated with the temperature sampler context.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which contains the context to be freed.
- **Control Flow**:
    - The function directly accesses the `ctx` member of the `llama_sampler` structure.
    - It casts the `ctx` pointer to `llama_sampler_temp*` type.
    - It then calls `delete` on the casted pointer to free the allocated memory.
- **Output**: This function does not return a value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_temp<!-- {{#callable:llama_sampler_init_temp}} -->
Initializes a `llama_sampler` with a temperature-based sampling strategy.
- **Inputs**:
    - `temp`: A floating-point value representing the temperature parameter used to control the randomness of the sampling process.
- **Control Flow**:
    - Calls the [`llama_sampler_init`](#llama_sampler_init) function to create a new `llama_sampler` instance.
    - Passes a pointer to the `llama_sampler_temp_i` interface and a new instance of `llama_sampler_temp` initialized with the provided temperature.
- **Output**: Returns a pointer to a newly created `llama_sampler` that uses temperature-based sampling.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_temp\_ext\_name<!-- {{#callable:llama_sampler_temp_ext_name}} -->
The `llama_sampler_temp_ext_name` function returns the name of the temperature extension sampler.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a string literal 'temp-ext'.
- **Output**: The output is a constant string 'temp-ext' representing the name of the sampler.


---
### llama\_sampler\_temp\_ext\_apply<!-- {{#callable:llama_sampler_temp_ext_apply}} -->
Applies dynamic temperature scaling to a token probability distribution based on entropy.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context and parameters for sampling.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token probabilities and logits.
- **Control Flow**:
    - Checks if the `delta` value in the context is greater than 0 to determine if dynamic temperature scaling should be applied.
    - Calculates the minimum and maximum temperature based on the `temp` and `delta` values.
    - Returns early if the size of `cur_p` is less than or equal to 1, as no scaling is needed.
    - Calculates the maximum possible entropy based on the size of `cur_p`.
    - Calls [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl) to compute softmax probabilities from the logits.
    - Calculates the entropy of the softmax probabilities and normalizes it.
    - Maps the normalized entropy to a dynamic temperature using a power function.
    - Applies the dynamic temperature to the logits using [`llama_sampler_temp_impl`](#llama_sampler_temp_impl).
    - Re-computes the softmax probabilities after applying the dynamic temperature.
    - Logs various debug information if in DEBUG mode.
- **Output**: The function modifies the `cur_p` structure in place, updating the probabilities based on the dynamically calculated temperature.
- **Functions called**:
    - [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl)
    - [`llama_sampler_temp_impl`](#llama_sampler_temp_impl)


---
### llama\_sampler\_temp\_ext\_clone<!-- {{#callable:llama_sampler_temp_ext_clone}} -->
Clones a `llama_sampler_temp_ext` structure by initializing a new sampler with the same context parameters.
- **Inputs**:
    - `smpl`: A pointer to a constant `llama_sampler` structure that contains the context and parameters to be cloned.
- **Control Flow**:
    - The function retrieves the context from the input `llama_sampler` structure, casting it to `llama_sampler_temp_ext`.
    - It then calls [`llama_sampler_init_temp_ext`](#llama_sampler_init_temp_ext) with the parameters extracted from the context to create a new sampler.
- **Output**: Returns a pointer to a newly initialized `llama_sampler` that is a clone of the input sampler.
- **Functions called**:
    - [`llama_sampler_init_temp_ext`](#llama_sampler_init_temp_ext)


---
### llama\_sampler\_temp\_ext\_free<!-- {{#callable:llama_sampler_temp_ext_free}} -->
The `llama_sampler_temp_ext_free` function deallocates the context associated with a `llama_sampler` instance.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure whose context needs to be freed.
- **Control Flow**:
    - The function directly accesses the `ctx` member of the `llama_sampler` structure.
    - It casts the `ctx` pointer to `llama_sampler_temp_ext` type.
    - The `delete` operator is used to free the memory allocated for the `llama_sampler_temp_ext` context.
- **Output**: This function does not return a value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_temp\_ext<!-- {{#callable:llama_sampler_init_temp_ext}} -->
Initializes a `llama_sampler` with extended temperature settings.
- **Inputs**:
    - `temp`: A float representing the base temperature for sampling.
    - `delta`: A float that defines the range of temperature variation.
    - `exponent`: A float that determines the exponent used for dynamic temperature scaling.
- **Control Flow**:
    - Calls [`llama_sampler_init`](#llama_sampler_init) to create a new sampler instance.
    - Allocates a new `llama_sampler_temp_ext` structure with the provided `temp`, `delta`, and `exponent` values.
    - Passes the interface pointer `&llama_sampler_temp_ext_i` and the newly created context to [`llama_sampler_init`](#llama_sampler_init).
- **Output**: Returns a pointer to a newly initialized `llama_sampler` configured for extended temperature sampling.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_xtc\_name<!-- {{#callable:llama_sampler_xtc_name}} -->
The `llama_sampler_xtc_name` function returns the name of the sampler as a constant string.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements.
    - It directly returns a string literal.
- **Output**: The output is a constant string "xtc" representing the name of the sampler.


---
### llama\_sample\_xtc\_apply<!-- {{#callable:llama_sample_xtc_apply}} -->
The `llama_sample_xtc_apply` function applies a sampling technique based on a probability threshold and minimum retention criteria.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the sampling context and parameters.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data and probabilities.
- **Control Flow**:
    - Checks if the `probability` is less than or equal to 0, if the `threshold` is greater than 0.5, or if the size of `cur_p` is less than 2; if any condition is true, the function returns immediately.
    - Generates a random float `chance` using a uniform distribution and compares it to `ctx->probability`; if `chance` exceeds `ctx->probability`, the function returns.
    - Calls [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl) to ensure the probabilities in `cur_p` are calculated and sorted.
    - Iterates through the `cur_p->data` array to find the last position where the probability is greater than or equal to `ctx->threshold` and stores this position in `pos_last`.
    - If the number of tokens from `pos_last` to the end of `cur_p` is greater than or equal to `ctx->min_keep` and `pos_last` is greater than 0, it adjusts `cur_p->data` and `cur_p->size` to keep only the relevant tokens.
- **Output**: The function modifies the `cur_p` array in place, potentially reducing its size and updating the data to retain only tokens that meet the specified probability threshold and minimum retention criteria.
- **Functions called**:
    - [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl)


---
### llama\_sampler\_xtc\_clone<!-- {{#callable:llama_sampler_xtc_clone}} -->
Clones a `llama_sampler_xtc` instance, copying its state.
- **Inputs**:
    - `smpl`: A pointer to the original `llama_sampler` instance to be cloned.
- **Control Flow**:
    - Cast the context of the input `llama_sampler` to `llama_sampler_xtc` type.
    - Initialize a new `llama_sampler` instance using the parameters from the original sampler's context.
    - Copy the random number generator state from the original sampler's context to the new sampler's context.
    - Return the newly created sampler instance.
- **Output**: Returns a pointer to the newly cloned `llama_sampler` instance.
- **Functions called**:
    - [`llama_sampler_init_xtc`](#llama_sampler_init_xtc)


---
### llama\_sampler\_xtc\_free<!-- {{#callable:llama_sampler_xtc_free}} -->
Frees the resources associated with the `llama_sampler_xtc` context.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context to be freed.
- **Control Flow**:
    - The function takes a pointer to a `llama_sampler` structure as an argument.
    - It accesses the `ctx` member of the `llama_sampler` structure, which is expected to be of type `llama_sampler_xtc`.
    - The function then calls `delete` on the `ctx` pointer to free the associated resources.
- **Output**: This function does not return a value; it performs a cleanup operation by deallocating memory.


---
### llama\_sampler\_xtc\_reset<!-- {{#callable:llama_sampler_xtc_reset}} -->
Resets the state of the `llama_sampler_xtc` by reinitializing its random number generator seed.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context for the sampler.
- **Control Flow**:
    - Retrieve the context of the sampler by casting `smpl->ctx` to `llama_sampler_xtc`.
    - Get the current seed using [`get_rng_seed`](#get_rng_seed) function and assign it to `ctx->seed_cur`.
    - Set the random number generator's seed using `ctx->rng.seed(ctx->seed_cur)`.
- **Output**: This function does not return a value; it modifies the state of the `llama_sampler_xtc` instance directly.
- **Functions called**:
    - [`get_rng_seed`](#get_rng_seed)


---
### llama\_sampler\_init\_xtc<!-- {{#callable:llama_sampler_init_xtc}} -->
Initializes a `llama_sampler_xtc` structure with specified parameters.
- **Inputs**:
    - `p`: A float representing the probability threshold for sampling.
    - `t`: A float representing the threshold for token selection.
    - `min_keep`: A size_t indicating the minimum number of tokens to keep after sampling.
    - `seed`: A uint32_t value used to seed the random number generator.
- **Control Flow**:
    - Calls [`get_rng_seed`](#get_rng_seed) to obtain a current random seed based on the provided seed.
    - Creates a new instance of `llama_sampler_xtc` with the provided parameters.
    - Calls [`llama_sampler_init`](#llama_sampler_init) to initialize the sampler interface with the created context.
- **Output**: Returns a pointer to a newly initialized `llama_sampler` structure.
- **Functions called**:
    - [`get_rng_seed`](#get_rng_seed)
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_mirostat\_name<!-- {{#callable:llama_sampler_mirostat_name}} -->
Returns the name of the 'mirostat' sampler.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a string literal 'mirostat'.
- **Output**: A constant string 'mirostat' representing the name of the sampler.


---
### llama\_sampler\_mirostat\_apply<!-- {{#callable:llama_sampler_mirostat_apply}} -->
The `llama_sampler_mirostat_apply` function applies the Mirostat sampling algorithm to adjust token probabilities based on estimated surprise.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context and parameters for the Mirostat sampling.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token probabilities and their associated data.
- **Control Flow**:
    - Calls [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl) to compute the softmax probabilities for the current token data.
    - Estimates `s_hat` using the most probable `m` tokens from `cur_p` to calculate the surprise value.
    - Computes `k` based on the estimated `s_hat` and the target surprise value `tau`.
    - Applies the top-k filtering to the token probabilities using [`llama_sampler_top_k_impl`](#llama_sampler_top_k_impl).
    - Selects a token based on the updated probabilities using [`llama_sample_dist`](#llama_sample_dist).
    - Calculates the observed surprise and updates the `mu` parameter using the learning rate and error.
- **Output**: The function modifies the `cur_p` structure in place, selecting a token based on the adjusted probabilities and updating the `mu` parameter in the sampler's context.
- **Functions called**:
    - [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl)
    - [`llama_sampler_top_k_impl`](#llama_sampler_top_k_impl)
    - [`llama_sample_dist`](#llama_sample_dist)


---
### llama\_sampler\_mirostat\_clone<!-- {{#callable:llama_sampler_mirostat_clone}} -->
Clones a `llama_sampler_mirostat` instance, initializing a new sampler with the same context parameters.
- **Inputs**:
    - `smpl`: A pointer to the original `llama_sampler` instance that is to be cloned.
- **Control Flow**:
    - The function retrieves the context of the original sampler, casting it to `llama_sampler_mirostat`.
    - It initializes a new sampler using [`llama_sampler_init_mirostat`](#llama_sampler_init_mirostat) with parameters from the original context.
    - The state of the original sampler's context is copied to the new sampler's context, specifically the `mu` and `rng` values.
    - Finally, the new sampler instance is returned.
- **Output**: Returns a pointer to the newly cloned `llama_sampler` instance.
- **Functions called**:
    - [`llama_sampler_init_mirostat`](#llama_sampler_init_mirostat)


---
### llama\_sampler\_mirostat\_reset<!-- {{#callable:llama_sampler_mirostat_reset}} -->
Resets the state of the `llama_sampler_mirostat` context by initializing the `mu` parameter and reseeding the random number generator.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context for the Mirostat sampler.
- **Control Flow**:
    - Retrieve the `llama_sampler_mirostat` context from the provided `llama_sampler` pointer.
    - Set the `mu` parameter to twice the value of `tau` from the context.
    - Obtain a new random seed using the [`get_rng_seed`](#get_rng_seed) function based on the current seed.
    - Seed the random number generator with the newly obtained seed.
- **Output**: The function does not return a value; it modifies the state of the `llama_sampler_mirostat` context directly.
- **Functions called**:
    - [`get_rng_seed`](#get_rng_seed)


---
### llama\_sampler\_mirostat\_free<!-- {{#callable:llama_sampler_mirostat_free}} -->
The `llama_sampler_mirostat_free` function deallocates the memory used by the `ctx` member of a `llama_sampler` structure.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure whose context needs to be freed.
- **Control Flow**:
    - The function takes a pointer to a `llama_sampler` structure as input.
    - It accesses the `ctx` member of the `llama_sampler` structure, which is expected to be a pointer to a `llama_sampler_mirostat` structure.
    - The function then uses the `delete` operator to free the memory allocated for the `llama_sampler_mirostat` context.
- **Output**: This function does not return any value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_mirostat<!-- {{#callable:llama_sampler_init_mirostat}} -->
Initializes a `llama_sampler` instance using the Mirostat sampling algorithm.
- **Inputs**:
    - `n_vocab`: The number of vocabulary tokens available for sampling.
    - `seed`: The seed value for random number generation.
    - `tau`: A parameter that influences the target surprise value.
    - `eta`: A learning rate parameter for updating the surprise value.
    - `m`: The number of most probable tokens to consider for estimating surprise.
- **Control Flow**:
    - Calls [`get_rng_seed`](#get_rng_seed) to obtain a current random seed based on the provided seed.
    - Creates a new instance of `llama_sampler_mirostat` with the provided parameters.
    - Calls [`llama_sampler_init`](#llama_sampler_init) to initialize the sampler with the Mirostat interface and the newly created context.
- **Output**: Returns a pointer to a newly initialized `llama_sampler` instance configured for Mirostat sampling.
- **Functions called**:
    - [`get_rng_seed`](#get_rng_seed)
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_mirostat\_v2\_name<!-- {{#callable:llama_sampler_mirostat_v2_name}} -->
The `llama_sampler_mirostat_v2_name` function returns the name of the Mirostat v2 sampler.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a string literal without any conditional logic or loops.
- **Output**: The output is a constant string "mirostat-v2" representing the name of the sampler.


---
### llama\_sampler\_mirostat\_v2\_apply<!-- {{#callable:llama_sampler_mirostat_v2_apply}} -->
Applies the Mirostat v2 sampling algorithm to a given token data array.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context and state for the sampling process.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data, including probabilities and selected token.
- **Control Flow**:
    - Calls [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl) to compute the softmax probabilities for the tokens in `cur_p`.
    - Truncates the token data array to only include tokens with surprise values less than or equal to `mu`.
    - If no tokens remain after truncation, sets the size of `cur_p` to 1 to ensure at least one token is selected.
    - Normalizes the probabilities of the remaining tokens by calling [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl) again.
    - Samples a token index from the remaining tokens using [`llama_sample_dist`](#llama_sample_dist).
    - Calculates the observed surprise for the selected token and updates the `mu` value based on the learning rate and error.
- **Output**: The function does not return a value but modifies the `cur_p` structure to reflect the selected token and updates the `mu` value in the sampler context.
- **Functions called**:
    - [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl)
    - [`llama_sample_dist`](#llama_sample_dist)


---
### llama\_sampler\_mirostat\_v2\_reset<!-- {{#callable:llama_sampler_mirostat_v2_reset}} -->
Resets the state of the `llama_sampler_mirostat_v2` context.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context to be reset.
- **Control Flow**:
    - Retrieve the context of type `llama_sampler_mirostat_v2` from the `llama_sampler` structure.
    - Set the `mu` parameter to twice the value of `tau` from the context.
    - Obtain a new random seed using the [`get_rng_seed`](#get_rng_seed) function with the current seed from the context.
    - Seed the random number generator with the newly obtained seed.
- **Output**: This function does not return a value; it modifies the state of the `llama_sampler_mirostat_v2` context directly.
- **Functions called**:
    - [`get_rng_seed`](#get_rng_seed)


---
### llama\_sampler\_mirostat\_v2\_clone<!-- {{#callable:llama_sampler_mirostat_v2_clone}} -->
Clones a `llama_sampler_mirostat_v2` instance by copying its state.
- **Inputs**:
    - `smpl`: A pointer to the original `llama_sampler` instance that is to be cloned.
- **Control Flow**:
    - Cast the context of the input sampler to `llama_sampler_mirostat_v2` type.
    - Initialize a new sampler using [`llama_sampler_init_mirostat_v2`](#llama_sampler_init_mirostat_v2) with the seed, tau, and eta from the original sampler's context.
    - Copy the state variables `mu` and `rng` from the original context to the new context.
    - Return the newly created sampler.
- **Output**: Returns a pointer to the newly cloned `llama_sampler` instance.
- **Functions called**:
    - [`llama_sampler_init_mirostat_v2`](#llama_sampler_init_mirostat_v2)


---
### llama\_sampler\_mirostat\_v2\_free<!-- {{#callable:llama_sampler_mirostat_v2_free}} -->
The `llama_sampler_mirostat_v2_free` function deallocates the context associated with a `llama_sampler` instance.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context to be freed.
- **Control Flow**:
    - The function takes a pointer to a `llama_sampler` structure as input.
    - It accesses the `ctx` member of the `llama_sampler` structure, which is expected to be a pointer to a `llama_sampler_mirostat_v2` instance.
    - The function then calls `delete` on the `ctx` pointer to free the allocated memory.
- **Output**: This function does not return a value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_mirostat\_v2<!-- {{#callable:llama_sampler_init_mirostat_v2}} -->
Initializes a `llama_sampler` using the Mirostat v2 algorithm with specified parameters.
- **Inputs**:
    - `seed`: A 32-bit unsigned integer used to initialize the random number generator.
    - `tau`: A float representing the target surprise value for the Mirostat algorithm.
    - `eta`: A float representing the learning rate for updating the surprise value.
- **Control Flow**:
    - Calls [`get_rng_seed`](#get_rng_seed) to obtain a current random seed based on the provided `seed`.
    - Creates a new instance of `llama_sampler_mirostat_v2` with the provided parameters and the current seed.
    - Invokes [`llama_sampler_init`](#llama_sampler_init) to initialize the sampler with the interface and context.
- **Output**: Returns a pointer to a newly initialized `llama_sampler` configured for the Mirostat v2 sampling algorithm.
- **Functions called**:
    - [`get_rng_seed`](#get_rng_seed)
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_grammar\_name<!-- {{#callable:llama_sampler_grammar_name}} -->
The `llama_sampler_grammar_name` function returns the name of the grammar sampler.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a string literal 'grammar'.
- **Output**: The output is a constant string 'grammar' representing the name of the grammar sampler.


---
### llama\_sampler\_grammar\_accept\_impl<!-- {{#callable:llama_sampler_grammar_accept_impl}} -->
The `llama_sampler_grammar_accept_impl` function accepts a token into the grammar context of a sampler.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context for the sampler.
    - `token`: A `llama_token` representing the token to be accepted by the grammar.
- **Control Flow**:
    - The function retrieves the grammar context from the `llama_sampler` structure.
    - It checks if the grammar context is valid (not null).
    - If valid, it calls the [`llama_grammar_accept_impl`](llama-grammar.cpp.driver.md#llama_grammar_accept_impl) function, passing the grammar and the token.
- **Output**: The function does not return a value; it modifies the state of the grammar context by accepting the provided token.
- **Functions called**:
    - [`llama_grammar_accept_impl`](llama-grammar.cpp.driver.md#llama_grammar_accept_impl)


---
### llama\_sampler\_grammar\_apply<!-- {{#callable:llama_sampler_grammar_apply}} -->
Applies grammar rules to the current token data array if a grammar context is available.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context and interface for the sampler.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data to which grammar rules will be applied.
- **Control Flow**:
    - The function retrieves the grammar context from the `llama_sampler` structure.
    - It checks if the grammar context is not null.
    - If the grammar context is valid, it calls the [`llama_grammar_apply_impl`](llama-grammar.cpp.driver.md#llama_grammar_apply_impl) function, passing the grammar and the current token data array.
- **Output**: The function does not return a value; it modifies the `cur_p` token data array in place based on the applied grammar rules.
- **Functions called**:
    - [`llama_grammar_apply_impl`](llama-grammar.cpp.driver.md#llama_grammar_apply_impl)


---
### llama\_sampler\_grammar\_reset<!-- {{#callable:llama_sampler_grammar_reset}} -->
Resets the grammar of the `llama_sampler` by reinitializing it with the current grammar settings.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context and grammar to be reset.
- **Control Flow**:
    - Checks if the `grammar` field in the `ctx` of the `llama_sampler` is null; if it is, the function returns immediately.
    - Creates a vector to hold the C-style string representations of the trigger patterns from the grammar.
    - Iterates over the `trigger_patterns` in the grammar, converting each pattern to a C-style string and storing it in the vector.
    - Calls `llama_grammar_init_impl` to initialize a new grammar using the current settings and the collected trigger patterns.
    - Frees the old grammar using [`llama_grammar_free_impl`](llama-grammar.cpp.driver.md#llama_grammar_free_impl).
    - Updates the `grammar` field in the context with the newly initialized grammar.
- **Output**: This function does not return a value; it modifies the state of the `llama_sampler` by resetting its grammar.
- **Functions called**:
    - [`llama_grammar_free_impl`](llama-grammar.cpp.driver.md#llama_grammar_free_impl)


---
### llama\_sampler\_grammar\_clone<!-- {{#callable:llama_sampler_grammar_clone}} -->
Clones a `llama_sampler` instance by duplicating its grammar state.
- **Inputs**:
    - `smpl`: A pointer to a constant `llama_sampler` structure that contains the grammar state to be cloned.
- **Control Flow**:
    - The function retrieves the context from the input `llama_sampler` instance, casting it to `llama_sampler_grammar`.
    - It initializes a new `llama_sampler` instance using [`llama_sampler_init_grammar_impl`](#llama_sampler_init_grammar_impl) with the vocabulary from the context.
    - An assertion is made to ensure that the new sampler was created successfully.
    - If the original sampler's grammar is not null, it copies the grammar string and root from the original to the new sampler.
    - The grammar is cloned using [`llama_grammar_clone_impl`](llama-grammar.cpp.driver.md#llama_grammar_clone_impl) and assigned to the new sampler's context.
- **Output**: Returns a pointer to the newly cloned `llama_sampler` instance.
- **Functions called**:
    - [`llama_sampler_init_grammar_impl`](#llama_sampler_init_grammar_impl)
    - [`llama_grammar_clone_impl`](llama-grammar.cpp.driver.md#llama_grammar_clone_impl)


---
### llama\_sampler\_grammar\_free<!-- {{#callable:llama_sampler_grammar_free}} -->
Frees the resources associated with a `llama_sampler` instance.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context and resources to be freed.
- **Control Flow**:
    - The function retrieves the context from the `llama_sampler` instance.
    - If the context contains a valid `grammar`, it calls [`llama_grammar_free_impl`](llama-grammar.cpp.driver.md#llama_grammar_free_impl) to free the grammar resources.
    - Finally, it deletes the context itself.
- **Output**: This function does not return a value; it performs cleanup operations to free allocated resources.
- **Functions called**:
    - [`llama_grammar_free_impl`](llama-grammar.cpp.driver.md#llama_grammar_free_impl)


---
### llama\_sampler\_init\_grammar\_impl<!-- {{#callable:llama_sampler_init_grammar_impl}} -->
Initializes a grammar-based llama sampler with specified vocabulary and grammar parameters.
- **Inputs**:
    - `vocab`: A pointer to the `llama_vocab` structure that contains the vocabulary used by the sampler.
    - `grammar_str`: A string representing the grammar rules to be used by the sampler.
    - `grammar_root`: A string indicating the root of the grammar structure.
    - `lazy`: A boolean flag indicating whether to initialize the grammar lazily.
    - `trigger_words`: An array of strings representing words that trigger specific grammar rules.
    - `num_trigger_words`: The number of trigger words provided.
    - `trigger_tokens`: An array of tokens corresponding to the trigger words.
    - `num_trigger_tokens`: The number of trigger tokens provided.
    - `trigger_patterns`: An array of strings representing regex patterns for triggers.
    - `num_trigger_patterns`: The number of trigger patterns provided.
- **Control Flow**:
    - Allocates memory for a new `llama_sampler_grammar` context.
    - Checks if the `grammar_str` is not null or empty; if so, it processes trigger words to create a regex pattern.
    - If trigger words are provided, it constructs a regex pattern that matches any of the trigger words.
    - Initializes the grammar using `llama_grammar_init_impl` with the provided parameters.
    - If the grammar initialization fails, it cleans up and returns null.
    - If the grammar string is empty, it initializes the context with default values.
    - Finally, it returns a new `llama_sampler` initialized with the grammar context.
- **Output**: Returns a pointer to a `llama_sampler` structure initialized with the grammar context, or null if initialization fails.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_init\_grammar<!-- {{#callable:llama_sampler_init_grammar}} -->
Initializes a `llama_sampler` with grammar-based sampling.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary used for sampling.
    - `grammar_str`: A string representing the grammar to be used for sampling.
    - `grammar_root`: A string representing the root of the grammar.
- **Control Flow**:
    - Calls the [`llama_sampler_init_grammar_impl`](#llama_sampler_init_grammar_impl) function with the provided vocabulary, grammar string, and grammar root.
    - The [`llama_sampler_init_grammar_impl`](#llama_sampler_init_grammar_impl) function handles the actual initialization of the grammar sampler.
- **Output**: Returns a pointer to a `llama_sampler` initialized with the specified grammar.
- **Functions called**:
    - [`llama_sampler_init_grammar_impl`](#llama_sampler_init_grammar_impl)


---
### llama\_sampler\_init\_grammar\_lazy<!-- {{#callable:llama_sampler_init_grammar_lazy}} -->
Initializes a lazy grammar sampler for the LLaMA model.
- **Inputs**:
    - `vocab`: A pointer to the vocabulary structure used by the sampler.
    - `grammar_str`: A string representing the grammar to be used.
    - `grammar_root`: A string indicating the root of the grammar.
    - `trigger_words`: An array of strings representing words that trigger specific grammar rules.
    - `num_trigger_words`: The number of trigger words provided.
    - `trigger_tokens`: An array of tokens corresponding to the trigger words.
    - `num_trigger_tokens`: The number of trigger tokens provided.
- **Control Flow**:
    - Calls [`llama_sampler_init_grammar_impl`](#llama_sampler_init_grammar_impl) with the provided parameters, setting the lazy flag to true.
    - The function handles the initialization of the grammar sampler, which may involve setting up grammar rules based on the provided strings and tokens.
- **Output**: Returns a pointer to a `llama_sampler` structure initialized with the specified grammar and vocabulary.
- **Functions called**:
    - [`llama_sampler_init_grammar_impl`](#llama_sampler_init_grammar_impl)


---
### llama\_sampler\_init\_grammar\_lazy\_patterns<!-- {{#callable:llama_sampler_init_grammar_lazy_patterns}} -->
Initializes a lazy grammar sampler with specified patterns.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary used by the sampler.
    - `grammar_str`: A string representing the grammar rules to be used by the sampler.
    - `grammar_root`: A string indicating the root of the grammar structure.
    - `trigger_patterns`: An array of strings representing patterns that trigger specific grammar rules.
    - `num_trigger_patterns`: The number of trigger patterns provided in the `trigger_patterns` array.
    - `trigger_tokens`: An array of tokens that correspond to the trigger patterns.
    - `num_trigger_tokens`: The number of trigger tokens provided in the `trigger_tokens` array.
- **Control Flow**:
    - Calls [`llama_sampler_init_grammar_impl`](#llama_sampler_init_grammar_impl) with the provided parameters, setting the lazy flag to true.
    - The function passes `nullptr` for the trigger words and zero for the number of trigger words, indicating that they are not used in this initialization.
- **Output**: Returns a pointer to a `llama_sampler` structure initialized with the specified grammar and patterns.
- **Functions called**:
    - [`llama_sampler_init_grammar_impl`](#llama_sampler_init_grammar_impl)


---
### llama\_sampler\_penalties\_name<!-- {{#callable:llama_sampler_penalties_name}} -->
The `llama_sampler_penalties_name` function returns the name of the penalties sampler.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements as it directly returns a string.
    - It takes a pointer to a `llama_sampler` structure as an argument, but does not use it in the function body.
- **Output**: The function outputs a constant string "penalties".


---
### llama\_sampler\_penalties\_accept<!-- {{#callable:llama_sampler_penalties_accept}} -->
The `llama_sampler_penalties_accept` function updates the penalty context for a given token in a llama sampler.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context for the sampler.
    - `token`: A `llama_token` representing the token to be accepted and processed for penalties.
- **Control Flow**:
    - Check if the `penalty_last_n` is zero; if so, exit the function early.
    - Increment the count of the accepted `token` in the `token_count` map.
    - If the ring buffer is full (i.e., it has reached the size of `penalty_last_n`), remove the oldest token from the buffer.
    - Decrement the count of the removed token in the `token_count` map, and erase it if its count reaches zero.
    - Add the new `token` to the ring buffer.
- **Output**: The function does not return a value; it modifies the internal state of the `llama_sampler` context.


---
### llama\_sampler\_penalties\_apply<!-- {{#callable:llama_sampler_penalties_apply}} -->
Applies penalties to the logits of tokens based on their frequency and presence in the last N tokens.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context for applying penalties.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data including logits.
- **Control Flow**:
    - Check if the penalty parameters are set to zero or if all penalty values are neutral, in which case the function returns early.
    - Iterate over each token in the `cur_p` array.
    - For each token, check if it exists in the `token_count` map; if not, continue to the next token.
    - Retrieve the count of occurrences for the token and assert that it is within valid bounds.
    - Adjust the logit of the token based on its count and the penalty values, applying either multiplication or division based on the logit value.
    - Subtract frequency and presence penalties from the logit based on the token's count.
    - Mark the `cur_p` array as unsorted after applying penalties.
- **Output**: The function modifies the `cur_p` array in place, adjusting the logits of the tokens based on the applied penalties.


---
### llama\_sampler\_penalties\_reset<!-- {{#callable:llama_sampler_penalties_reset}} -->
Resets the penalties for the `llama_sampler` by clearing the previous tokens and token count.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context for penalties.
- **Control Flow**:
    - Retrieve the context of type `llama_sampler_penalties` from the `llama_sampler` structure.
    - Clear the `prev` ring buffer which stores the last tokens.
    - Clear the `token_count` unordered map which tracks the frequency of tokens.
- **Output**: This function does not return a value; it modifies the state of the `llama_sampler` by resetting its penalties.


---
### llama\_sampler\_penalties\_clone<!-- {{#callable:llama_sampler_penalties_clone}} -->
Clones a `llama_sampler` object with penalties by copying its context and initializing a new sampler.
- **Inputs**:
    - `smpl`: A pointer to the original `llama_sampler` object that is to be cloned.
- **Control Flow**:
    - The function retrieves the context from the original sampler, which contains penalty parameters.
    - It initializes a new `llama_sampler` with the penalty parameters from the context.
    - The state of the previous tokens is copied from the original sampler's context to the new sampler's context.
    - Finally, the new sampler is returned.
- **Output**: Returns a pointer to the newly cloned `llama_sampler` object.
- **Functions called**:
    - [`llama_sampler_init_penalties`](#llama_sampler_init_penalties)


---
### llama\_sampler\_penalties\_free<!-- {{#callable:llama_sampler_penalties_free}} -->
Frees the memory allocated for the penalties context in a `llama_sampler`.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which contains the context to be freed.
- **Control Flow**:
    - The function takes a pointer to a `llama_sampler` structure as input.
    - It accesses the `ctx` member of the `llama_sampler` structure, which is expected to be a pointer to a `llama_sampler_penalties` structure.
    - The function then deletes the memory allocated for the `llama_sampler_penalties` context.
- **Output**: This function does not return a value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_penalties<!-- {{#callable:llama_sampler_init_penalties}} -->
Initializes a `llama_sampler` with specified penalties for token sampling.
- **Inputs**:
    - `penalty_last_n`: An integer specifying the number of last tokens to consider for penalties.
    - `penalty_repeat`: A float representing the penalty factor for repeated tokens.
    - `penalty_freq`: A float representing the penalty factor for token frequency.
    - `penalty_present`: A float representing the penalty factor for the presence of tokens.
- **Control Flow**:
    - The function first ensures that `penalty_last_n` is non-negative by taking the maximum of `penalty_last_n` and 0.
    - It then calls [`llama_sampler_init`](#llama_sampler_init) to create a new `llama_sampler` instance, passing a pointer to the penalties interface and a new `llama_sampler_penalties` context.
    - The context is initialized with the provided penalty parameters and a ring buffer for tracking the last `penalty_last_n` tokens.
- **Output**: Returns a pointer to a newly initialized `llama_sampler` configured with the specified penalties.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_top\_n\_sigma\_name<!-- {{#callable:llama_sampler_top_n_sigma_name}} -->
The `llama_sampler_top_n_sigma_name` function returns the name of the top-n-sigma sampling method.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a string literal without any conditional logic or loops.
- **Output**: The output is a constant string "top-n-sigma" representing the name of the sampling method.


---
### llama\_sampler\_top\_n\_sigma\_apply<!-- {{#callable:llama_sampler_top_n_sigma_apply}} -->
The `llama_sampler_top_n_sigma_apply` function applies a top-n sigma sampling strategy to a given array of token data.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context for the sampling operation.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data to be processed.
- **Control Flow**:
    - The function first retrieves the context from the `llama_sampler` structure.
    - It checks if the parameter `n` in the context is less than or equal to 0 or if the size of `cur_p` is less than or equal to 1, in which case it returns early.
    - It initializes variables to find the maximum logit and calculate the mean of the logits, iterating through `cur_p` to compute these values while ignoring negative infinity logits.
    - It calculates the standard deviation of the logits using the mean and the valid logits count.
    - It applies a mask to the logits in `cur_p`, setting logits below the threshold defined by the maximum logit minus `n` times the standard deviation to negative infinity.
    - Finally, it calls the [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl) function to normalize the logits in `cur_p`.
- **Output**: The function modifies the `cur_p` array in place, applying the top-n sigma sampling strategy and normalizing the logits.
- **Functions called**:
    - [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl)


---
### llama\_sampler\_top\_n\_sigma\_clone<!-- {{#callable:llama_sampler_top_n_sigma_clone}} -->
Clones a `llama_sampler` instance using the `top-n-sigma` sampling strategy.
- **Inputs**:
    - `smpl`: A pointer to a constant `llama_sampler` structure that contains the context for the sampler to be cloned.
- **Control Flow**:
    - The function retrieves the context of the provided `llama_sampler` by casting `smpl->ctx` to `const llama_sampler_top_n_sigma*`.
    - It then calls [`llama_sampler_init_top_n_sigma`](#llama_sampler_init_top_n_sigma) with the value of `ctx->n` to create a new `llama_sampler` instance.
- **Output**: Returns a pointer to a new `llama_sampler` instance initialized with the same parameters as the original sampler.
- **Functions called**:
    - [`llama_sampler_init_top_n_sigma`](#llama_sampler_init_top_n_sigma)


---
### llama\_sampler\_top\_n\_sigma\_free<!-- {{#callable:llama_sampler_top_n_sigma_free}} -->
The `llama_sampler_top_n_sigma_free` function deallocates the memory associated with the `llama_sampler_top_n_sigma` context.
- **Inputs**: None
- **Control Flow**:
    - The function takes a pointer to a `llama_sampler` structure as an argument.
    - It accesses the `ctx` member of the `llama_sampler` structure, which is expected to point to a `llama_sampler_top_n_sigma` instance.
    - The function then calls `delete` on this context pointer to free the allocated memory.
- **Output**: The function does not return any value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_top\_n\_sigma<!-- {{#callable:llama_sampler_init_top_n_sigma}} -->
Initializes a `llama_sampler` for the top-n-sigma sampling strategy.
- **Inputs**:
    - `n`: A float value representing the number of standard deviations to use for masking logits.
- **Control Flow**:
    - Calls [`llama_sampler_init`](#llama_sampler_init) with a specific interface for top-n-sigma sampling and a new context containing the value of `n`.
- **Output**: Returns a pointer to a newly initialized `llama_sampler` structure configured for top-n-sigma sampling.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### get\_overlapping\_token\_sequences<!-- {{#callable:get_overlapping_token_sequences}} -->
The `get_overlapping_token_sequences` function identifies and stores overlapping token sequences from a given vocabulary that match a specified substring.
- **Inputs**:
    - `vocab`: A reference to a `llama_vocab` object that contains the vocabulary used for tokenization.
    - `str`: A `std::string` representing the substring to search for within the vocabulary words.
    - `token_sequences`: An `unordered_multimap` that will store the token IDs and their corresponding tokenizations as vectors.
    - `max_tail_len`: An integer specifying the maximum length of the tail tokens to be included; defaults to -1 for no limit.
- **Control Flow**:
    - Iterates over each token ID in the vocabulary.
    - Detokenizes the token ID to obtain the corresponding word.
    - Checks if the word contains the specified substring; if so, it adds an empty vector for that token ID to `token_sequences`.
    - If the substring is not found, it searches for occurrences of the first character of the substring in the word.
    - For each occurrence, it checks if the subsequent characters match the substring.
    - If a match is found, it tokenizes the remaining part of the substring and checks against `max_tail_len` to limit the size.
    - Before adding the tokenization to `token_sequences`, it checks for duplicates to avoid redundancy.
- **Output**: The function does not return a value but populates the `token_sequences` multimap with token IDs and their corresponding tokenizations.


---
### llama\_sampler\_dry\_name<!-- {{#callable:llama_sampler_dry_name}} -->
Returns the name of the dry sampler as a constant string.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which is not used in this function.
- **Control Flow**:
    - The function directly returns a constant string without any conditional logic or loops.
- **Output**: A constant string "dry" representing the name of the dry sampler.


---
### llama\_sampler\_dry\_accept<!-- {{#callable:llama_sampler_dry_accept}} -->
The `llama_sampler_dry_accept` function records a token in the dry sampler's context if certain conditions are met.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context for the sampler.
    - `token`: A `llama_token` representing the token to be accepted and recorded.
- **Control Flow**:
    - The function retrieves the context from the `llama_sampler` structure, casting it to `llama_sampler_dry`.
    - It checks if the `dry_multiplier` is zero, `dry_base` is less than 1.0, or `dry_penalty_last_n` is zero.
    - If any of these conditions are true, the function returns early without recording the token.
    - If the conditions are not met, the token is added to the `last_tokens` vector in the context.
- **Output**: The function does not return a value; it modifies the internal state of the `llama_sampler_dry` context by adding the token to the `last_tokens` vector.


---
### llama\_sampler\_dry\_apply<!-- {{#callable:llama_sampler_dry_apply}} -->
The `llama_sampler_dry_apply` function applies a dry sampling strategy to adjust the logits of tokens based on their repetition in the context.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context and parameters for the dry sampling.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data and logits to be adjusted.
- **Control Flow**:
    - Check if the dry sampling parameters are valid; if not, return immediately.
    - Calculate the effective penalty length and the number of last tokens to consider for repetition.
    - If the number of repeated tokens is less than the allowed length, return.
    - Initialize the repeat count and clear the maximum token repeat map.
    - Search for restart sequences in the last tokens to determine the maximum repetition limit.
    - If the repetition limit is less than the allowed length, return.
    - Use the Z-algorithm to compute the lengths of suffixes in the last tokens.
    - Iterate over the repeat counts to determine the maximum repeat length for each token.
    - Apply logit penalties based on the maximum repeat lengths for relevant tokens.
- **Output**: The function modifies the logits in `cur_p` to apply penalties for repeated tokens, potentially reducing their likelihood based on the defined dry sampling strategy.


---
### llama\_sampler\_dry\_reset<!-- {{#callable:llama_sampler_dry_reset}} -->
Resets the state of the `llama_sampler_dry` context by clearing its last tokens and repeat counts.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context to be reset.
- **Control Flow**:
    - The function retrieves the context of type `llama_sampler_dry` from the provided `llama_sampler` pointer.
    - It clears the `last_tokens`, `dry_repeat_count`, and `dry_max_token_repeat` vectors in the context.
- **Output**: This function does not return a value; it modifies the state of the `llama_sampler_dry` context directly.


---
### llama\_sampler\_dry\_clone<!-- {{#callable:llama_sampler_dry_clone}} -->
The `llama_sampler_dry_clone` function creates a deep copy of a `llama_sampler` instance, specifically for the dry sampling strategy.
- **Inputs**:
    - `smpl`: A pointer to the original `llama_sampler` instance that is to be cloned.
- **Control Flow**:
    - The function retrieves the context from the original sampler, casting it to `llama_sampler_dry`.
    - A dummy vocabulary is initialized since it is only needed for processing sequence breakers, which have already been handled.
    - The function calls [`llama_sampler_init_dry`](#llama_sampler_init_dry) to create a new sampler instance with the parameters from the original sampler's context.
    - The state of the original sampler, including processed breakers and repeat counts, is copied to the new sampler's context.
- **Output**: Returns a pointer to the newly created `llama_sampler` instance that is a clone of the original.
- **Functions called**:
    - [`llama_sampler_init_dry`](#llama_sampler_init_dry)


---
### llama\_sampler\_dry\_free<!-- {{#callable:llama_sampler_dry_free}} -->
The `llama_sampler_dry_free` function deallocates the memory associated with the `ctx` member of a `llama_sampler` structure.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure whose context needs to be freed.
- **Control Flow**:
    - The function takes a pointer to a `llama_sampler` structure as input.
    - It accesses the `ctx` member of the `llama_sampler` structure, which is expected to be of type `llama_sampler_dry`.
    - It then calls `delete` on the `ctx` pointer to free the allocated memory.
- **Output**: The function does not return any value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_dry<!-- {{#callable:llama_sampler_init_dry}} -->
Initializes a dry sampling strategy for the llama model with specified parameters.
- **Inputs**:
    - `vocab`: A pointer to the vocabulary structure used for tokenization.
    - `context_size`: The total size of the context for the sampling.
    - `dry_multiplier`: A multiplier that affects the penalty applied during sampling.
    - `dry_base`: The base used for calculating penalties.
    - `dry_allowed_length`: The maximum length of allowed repetitions.
    - `dry_penalty_last_n`: The number of last tokens to consider for applying penalties.
    - `seq_breakers`: An array of sequence breaker strings that can interrupt the sampling.
    - `num_breakers`: The number of sequence breakers provided.
- **Control Flow**:
    - Calculates the effective dry penalty length based on the input parameters.
    - Checks if dry sampling is enabled based on the provided parameters.
    - If dry sampling is enabled and there are sequence breakers, processes each breaker string.
    - For each breaker, validates its content and truncates it if necessary, then retrieves overlapping token sequences.
    - Finally, initializes and returns a new llama sampler instance with the processed parameters.
- **Output**: Returns a pointer to a newly initialized `llama_sampler` structure configured for dry sampling.
- **Functions called**:
    - [`get_overlapping_token_sequences`](#get_overlapping_token_sequences)
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_init\_dry\_testing<!-- {{#callable:llama_sampler_init_dry_testing}} -->
Initializes a `llama_sampler` for dry testing with specified parameters and processes sequence breakers.
- **Inputs**:
    - `context_size`: An integer representing the size of the context for the sampler.
    - `dry_multiplier`: A float that determines the multiplier for the dry penalty.
    - `dry_base`: A float that serves as the base for the dry penalty calculation.
    - `dry_allowed_length`: An integer that specifies the maximum allowed length for dry sequences.
    - `dry_penalty_last_n`: An integer that indicates how many of the last tokens to consider for penalties.
    - `seq_breakers`: A vector of vectors containing `llama_token` sequences that act as breakers for the dry sampling.
- **Control Flow**:
    - A dummy vocabulary is created to initialize the sampler.
    - The [`llama_sampler_init_dry`](#llama_sampler_init_dry) function is called to create a sampler with the provided parameters.
    - The context of the sampler is cast to `llama_sampler_dry` to access its specific fields.
    - The `dry_processed_breakers` map is cleared to prepare for new sequence breakers.
    - If the `seq_breakers` vector is empty, a warning is logged.
    - For each sequence breaker in `seq_breakers`, if it is not empty, the first token is treated as the head and the rest as tail tokens, which are stored in `dry_processed_breakers`.
    - If no valid sequence breakers are processed, a warning is logged.
- **Output**: Returns a pointer to the initialized `llama_sampler` for dry testing.
- **Functions called**:
    - [`llama_sampler_init_dry`](#llama_sampler_init_dry)


---
### llama\_sampler\_logit\_bias\_name<!-- {{#callable:llama_sampler_logit_bias_name}} -->
The `llama_sampler_logit_bias_name` function returns the name of the logit bias sampler.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a string literal 'logit-bias'.
- **Output**: The output is a constant string 'logit-bias' representing the name of the logit bias sampler.


---
### llama\_sampler\_logit\_bias\_apply<!-- {{#callable:llama_sampler_logit_bias_apply}} -->
Applies logit bias adjustments to a token data array.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context for the sampler.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data to which logit biases will be applied.
- **Control Flow**:
    - Check if the logit bias vector is empty; if so, return immediately.
    - Clear the `to_search` vector in the context.
    - Iterate over the `logit_bias` vector and apply biases to the corresponding tokens in `cur_p` if they are valid.
    - If any tokens were not found in the previous step, store them in `to_search`.
    - If `to_search` is empty after the first pass, return.
    - Iterate over the `cur_p` array again to apply biases for any remaining tokens in `to_search`.
- **Output**: The function modifies the `logit` values of the tokens in `cur_p` based on the specified biases, but does not return a value.


---
### llama\_sampler\_logit\_bias\_clone<!-- {{#callable:llama_sampler_logit_bias_clone}} -->
Clones a `llama_sampler` instance with logit bias settings.
- **Inputs**:
    - `smpl`: A pointer to a constant `llama_sampler` structure that contains the context and settings to be cloned.
- **Control Flow**:
    - The function retrieves the context from the input `llama_sampler` instance, specifically casting it to `llama_sampler_logit_bias`.
    - It then calls [`llama_sampler_init_logit_bias`](#llama_sampler_init_logit_bias) with the vocabulary size, the size of the logit bias vector, and the logit bias data from the context to create a new sampler instance.
- **Output**: Returns a pointer to a new `llama_sampler` instance that is a clone of the original, initialized with the same logit bias settings.
- **Functions called**:
    - [`llama_sampler_init_logit_bias`](#llama_sampler_init_logit_bias)


---
### llama\_sampler\_logit\_bias\_free<!-- {{#callable:llama_sampler_logit_bias_free}} -->
The `llama_sampler_logit_bias_free` function deallocates the memory associated with the logit bias context of a given sampler.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which contains the context to be freed.
- **Control Flow**:
    - The function takes a pointer to a `llama_sampler` structure as input.
    - It accesses the `ctx` member of the `llama_sampler` structure, which is expected to be a pointer to a `llama_sampler_logit_bias` structure.
    - The function then calls `delete` on this context pointer to free the allocated memory.
- **Output**: The function does not return any value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_logit\_bias<!-- {{#callable:llama_sampler_init_logit_bias}} -->
Initializes a `llama_sampler` with logit bias settings.
- **Inputs**:
    - `n_vocab`: The total number of vocabulary tokens.
    - `n_logit_bias`: The number of logit bias entries.
    - `logit_bias`: An array of `llama_logit_bias` structures containing token IDs and their corresponding bias values.
- **Control Flow**:
    - Calls [`llama_sampler_init`](#llama_sampler_init) to create a new sampler instance.
    - Allocates a new `llama_sampler_logit_bias` context with the provided vocabulary size and logit biases.
    - Copies the logit bias data into a vector for internal use.
- **Output**: Returns a pointer to the initialized `llama_sampler` instance with logit bias applied.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_infill\_name<!-- {{#callable:llama_sampler_infill_name}} -->
The `llama_sampler_infill_name` function returns the name of the infill sampler.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which is not used in this function.
- **Control Flow**:
    - The function directly returns a string literal 'infill'.
- **Output**: The function outputs a constant string 'infill', representing the name of the infill sampler.


---
### llama\_sampler\_infill\_apply<!-- {{#callable:llama_sampler_infill_apply}} -->
The `llama_sampler_infill_apply` function processes a token data array to adjust probabilities based on specific criteria, including the ratio of end-of-group tokens.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure that contains the context and configuration for the sampling process.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current token data, including their IDs, probabilities, and logits.
- **Control Flow**:
    - The function begins by invoking [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl) to normalize the probabilities in `cur_p` based on their logits.
    - It then initializes two sums, `p_txt_sum` and `p_eog_sum`, to accumulate probabilities of text and end-of-group (EOG) tokens respectively.
    - A loop iterates through the tokens in `cur_p`, updating the sums based on whether each token is an EOG token or not.
    - If the ratio of text to EOG probabilities is too low, the function filters `cur_p` to retain only EOG tokens and normalizes their probabilities.
    - If the ratio is acceptable, the function combines tokens with common prefixes and applies a threshold to filter out low-probability tokens.
    - Finally, it normalizes the probabilities of the remaining tokens and ensures at least one EOT token is present if no non-EOG tokens remain.
- **Output**: The function modifies the `cur_p` array in place, adjusting the size and probabilities of the tokens based on the defined criteria, and does not return a value.
- **Functions called**:
    - [`llama_sampler_softmax_impl`](#llama_sampler_softmax_impl)


---
### llama\_sampler\_infill\_clone<!-- {{#callable:llama_sampler_infill_clone}} -->
Clones a `llama_sampler` for infill sampling.
- **Inputs**:
    - `smpl`: A pointer to a constant `llama_sampler` structure that is to be cloned.
- **Control Flow**:
    - The function retrieves the context from the input `llama_sampler` by casting `smpl->ctx` to a `const llama_sampler_infill` type.
    - It then calls [`llama_sampler_init_infill`](#llama_sampler_init_infill) with the vocabulary from the retrieved context to create a new `llama_sampler` instance.
- **Output**: Returns a pointer to a new `llama_sampler` that is a clone of the input sampler, initialized for infill sampling.
- **Functions called**:
    - [`llama_sampler_init_infill`](#llama_sampler_init_infill)


---
### llama\_sampler\_infill\_free<!-- {{#callable:llama_sampler_infill_free}} -->
The `llama_sampler_infill_free` function deallocates the context associated with a `llama_sampler`.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure whose context needs to be freed.
- **Control Flow**:
    - The function takes a pointer to a `llama_sampler` structure as input.
    - It accesses the `ctx` member of the `llama_sampler` structure, which is expected to be of type `llama_sampler_infill`.
    - The function then deletes the context object pointed to by `ctx`, effectively freeing the allocated memory.
- **Output**: This function does not return a value; it performs a memory deallocation operation.


---
### llama\_sampler\_init\_infill<!-- {{#callable:llama_sampler_init_infill}} -->
Initializes a `llama_sampler` for infilling tasks using a specified vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary used for sampling.
- **Control Flow**:
    - Calls [`llama_sampler_init`](#llama_sampler_init) to create a new sampler instance.
    - Passes a pointer to the `llama_sampler_infill_i` interface and a new `llama_sampler_infill` context containing the vocabulary and two buffers initialized to a size of 512 characters each.
- **Output**: Returns a pointer to a newly initialized `llama_sampler` for infilling.
- **Functions called**:
    - [`llama_sampler_init`](#llama_sampler_init)


---
### llama\_sampler\_get\_seed<!-- {{#callable:llama_sampler_get_seed}} -->
Retrieves the current seed value from a `llama_sampler` instance based on its interface type.
- **Inputs**:
    - `smpl`: A pointer to a `llama_sampler` structure, which contains the sampler's context and interface.
- **Control Flow**:
    - Checks the interface type of the `llama_sampler` instance pointed to by `smpl`.
    - If the interface is `llama_sampler_dist_i`, retrieves the current seed from the `llama_sampler_dist` context.
    - If the interface is `llama_sampler_mirostat_i`, retrieves the current seed from the `llama_sampler_mirostat` context.
    - If the interface is `llama_sampler_mirostat_v2_i`, retrieves the current seed from the `llama_sampler_mirostat_v2` context.
    - If the interface is `llama_sampler_chain_i`, iterates through the chain of samplers in reverse order to find the first non-default seed.
    - If no valid seed is found, returns a default seed value.
- **Output**: Returns a `uint32_t` representing the current seed value for the sampler, or a default seed if none is found.


---
### llama\_perf\_sampler<!-- {{#callable:llama_perf_sampler}} -->
The `llama_perf_sampler` function retrieves performance metrics from a given sampler chain.
- **Inputs**:
    - `chain`: A pointer to a `llama_sampler` structure representing the sampler chain from which performance data is to be retrieved.
- **Control Flow**:
    - The function initializes a `llama_perf_sampler_data` structure to hold the performance data.
    - It checks if the `chain` pointer is null or if the interface of the sampler does not match the expected type, aborting if either condition is true.
    - It retrieves the context from the `chain` and populates the `data` structure with the sampling time in milliseconds and the number of samples taken.
- **Output**: Returns a `llama_perf_sampler_data` structure containing the sampling time in milliseconds and the number of samples.


---
### llama\_perf\_sampler\_print<!-- {{#callable:llama_perf_sampler_print}} -->
Prints performance metrics for a given `llama_sampler` chain.
- **Inputs**:
    - `chain`: A pointer to a `llama_sampler` structure representing the sampler chain whose performance metrics are to be printed.
- **Control Flow**:
    - Calls [`llama_perf_sampler`](#llama_perf_sampler) to retrieve performance data from the provided `chain`.
    - Logs the sampling time, number of runs, average time per token, and tokens per second using `LLAMA_LOG_INFO`.
- **Output**: No return value; outputs performance metrics to the log.
- **Functions called**:
    - [`llama_perf_sampler`](#llama_perf_sampler)


---
### llama\_perf\_sampler\_reset<!-- {{#callable:llama_perf_sampler_reset}} -->
Resets the performance metrics of a `llama_sampler` chain.
- **Inputs**:
    - `chain`: A pointer to a `llama_sampler` structure that represents the sampler chain to be reset.
- **Control Flow**:
    - Checks if the `chain` pointer is null or if its interface does not match the expected type, aborting if either condition is true.
    - Retrieves the context from the `chain` pointer, casting it to a `llama_sampler_chain` structure.
    - Resets the `t_sample_us` and `n_sample` fields of the context to zero.
- **Output**: This function does not return a value; it modifies the state of the `llama_sampler` chain directly.


