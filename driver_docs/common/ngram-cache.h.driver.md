# Purpose
This C++ source code file is designed to manage and manipulate n-gram caches, which are data structures used to map sequences of tokens (n-grams) to their empirical probabilities. The file provides a collection of functions and data structures that facilitate the creation, updating, and manipulation of these n-gram caches. The primary components include the [`common_ngram`](#common_ngramcommon_ngram) structure, which represents an n-gram of tokens, and the `common_ngram_cache`, which is a specialized unordered map that associates n-grams with their observed token distributions. The file also defines hash functions for these structures to enable efficient storage and retrieval in hash-based containers.

The code offers a range of functionalities, including updating n-gram caches with new token sequences, drafting new tokens based on existing caches, and saving/loading caches to/from files. It also provides a mechanism to merge two n-gram caches. These operations are crucial for applications that require statistical language modeling or predictive text generation, where understanding the likelihood of token sequences is essential. The file is not an executable but rather a library intended to be included and used in other C++ projects, providing a public API for managing n-gram data structures and their associated operations.
# Imports and Dependencies

---
- `llama.h`
- `unordered_map`
- `string`
- `vector`


# Data Structures

---
### common\_ngram<!-- {{#data_structure:common_ngram}} -->
- **Type**: `struct`
- **Members**:
    - `tokens`: An array of llama_token with a fixed size of LLAMA_NGRAM_MAX, initialized to LLAMA_TOKEN_NULL.
- **Description**: The `common_ngram` struct is designed to represent a sequence of tokens, or n-gram, with a fixed maximum size defined by `LLAMA_NGRAM_MAX`. It provides constructors for initializing the n-gram with either default null tokens or a specified sequence of tokens up to the n-gram size. The struct also includes an equality operator to compare two n-grams, ensuring that all tokens in the sequence match. This data structure is primarily used in mapping n-grams to empirical token probabilities within a larger system for processing and analyzing token sequences.
- **Member Functions**:
    - [`common_ngram::common_ngram`](#common_ngramcommon_ngram)
    - [`common_ngram::common_ngram`](#common_ngramcommon_ngram)
    - [`common_ngram::operator==`](#common_ngramoperator==)

**Methods**

---
#### common\_ngram::common\_ngram<!-- {{#callable:common_ngram::common_ngram}} -->
The default constructor for the `common_ngram` struct initializes all elements of the `tokens` array to `LLAMA_TOKEN_NULL`.
- **Inputs**: None
- **Control Flow**:
    - The constructor iterates over each index from 0 to `LLAMA_NGRAM_MAX - 1`.
    - For each index, it sets the corresponding element in the `tokens` array to `LLAMA_TOKEN_NULL`.
- **Output**: There is no return value as this is a constructor for initializing an instance of the `common_ngram` struct.
- **See also**: [`common_ngram`](#common_ngram)  (Data Structure)


---
#### common\_ngram::common\_ngram<!-- {{#callable:common_ngram::common_ngram}} -->
The `common_ngram` constructor initializes a `common_ngram` object by copying a specified number of tokens from an input array and filling the rest with a null token.
- **Inputs**:
    - `input`: A pointer to an array of `llama_token` elements from which the n-gram tokens are copied.
    - `ngram_size`: An integer specifying the number of tokens to copy from the input array.
- **Control Flow**:
    - Iterates over a fixed range defined by `LLAMA_NGRAM_MAX`.
    - For each index `i`, checks if `i` is less than `ngram_size`.
    - If `i` is less than `ngram_size`, assigns `input[i]` to `tokens[i]`.
    - If `i` is not less than `ngram_size`, assigns `LLAMA_TOKEN_NULL` to `tokens[i]`.
- **Output**: The function does not return a value; it initializes the `tokens` array within the `common_ngram` object.
- **See also**: [`common_ngram`](#common_ngram)  (Data Structure)


---
#### common\_ngram::operator==<!-- {{#callable:common_ngram::operator==}} -->
The `operator==` function compares two `common_ngram` objects to determine if they are equal by checking if all their tokens are identical.
- **Inputs**:
    - `other`: A reference to another `common_ngram` object to compare against the current object.
- **Control Flow**:
    - Iterate over each index from 0 to `LLAMA_NGRAM_MAX - 1`.
    - For each index, compare the token at that index in the current object with the token at the same index in the `other` object.
    - If any pair of tokens are not equal, return `false`.
    - If all tokens are equal, return `true`.
- **Output**: A boolean value indicating whether the two `common_ngram` objects are equal (`true`) or not (`false`).
- **See also**: [`common_ngram`](#common_ngram)  (Data Structure)



---
### common\_token\_hash\_function<!-- {{#data_structure:common_token_hash_function}} -->
- **Type**: `struct`
- **Description**: The `common_token_hash_function` is a C++ struct that defines a custom hash function for `llama_token` types. It uses a specific constant multiplier to perform Fibonacci hashing, which is a technique that optimizes hash distribution by using a large prime number. This struct is designed to be used with hash-based containers like `std::unordered_map` to efficiently store and retrieve `llama_token` values.
- **Member Functions**:
    - [`common_token_hash_function::operator()`](#common_token_hash_functionoperator())

**Methods**

---
#### common\_token\_hash\_function::operator\(\)<!-- {{#callable:common_token_hash_function::operator()}} -->
The `operator()` function in `common_token_hash_function` computes a hash value for a given `llama_token` using a Fibonacci hashing technique.
- **Inputs**:
    - `token`: A `llama_token` which is an input to the hash function, representing a token to be hashed.
- **Control Flow**:
    - The function takes a `llama_token` as input.
    - It multiplies the token by a large constant `11400714819323198485llu`, which is a part of the Fibonacci hashing technique.
    - The result of the multiplication is returned as the hash value.
- **Output**: The function returns a `size_t` value, which is the hash of the input token.
- **See also**: [`common_token_hash_function`](#common_token_hash_function)  (Data Structure)



---
### common\_ngram\_hash\_function<!-- {{#data_structure:common_ngram_hash_function}} -->
- **Type**: `struct`
- **Description**: The `common_ngram_hash_function` is a functor struct designed to compute a hash value for a `common_ngram` object. It utilizes the `common_token_hash_function` to hash each token within the `ngram` and combines these hashes using the XOR operation. This struct is primarily used to enable the use of `common_ngram` objects as keys in hash-based containers like `std::unordered_map`, ensuring efficient retrieval and storage of n-gram data.
- **Member Functions**:
    - [`common_ngram_hash_function::operator()`](#common_ngram_hash_functionoperator())

**Methods**

---
#### common\_ngram\_hash\_function::operator\(\)<!-- {{#callable:common_ngram_hash_function::operator()}} -->
The `operator()` function computes a hash value for a `common_ngram` object by combining the hash values of its tokens.
- **Inputs**:
    - `ngram`: A `common_ngram` object containing an array of tokens to be hashed.
- **Control Flow**:
    - Initialize a hash value using the hash of the first token in the `ngram`.
    - Iterate over the remaining tokens in the `ngram` up to `LLAMA_NGRAM_MAX`.
    - For each token, compute its hash and XOR it with the current hash value.
    - Return the final computed hash value.
- **Output**: A `size_t` hash value representing the combined hash of the tokens in the `ngram`.
- **See also**: [`common_ngram_hash_function`](#common_ngram_hash_function)  (Data Structure)



