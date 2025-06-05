# Purpose
This C++ source code file is designed to manage and manipulate n-gram caches, which are data structures used to store sequences of tokens and their occurrences. The primary functionality revolves around updating, drafting, saving, loading, and merging n-gram caches. The code is structured to handle n-grams of varying sizes, specified by `ngram_min` and `ngram_max`, and it operates on sequences of `llama_token` objects. The [`common_ngram_cache_update`](#common_ngram_cache_update) function updates the cache with new tokens, while the [`common_ngram_cache_draft`](#common_ngram_cache_draft) function attempts to draft new tokens based on existing cache data, using both dynamic and static caches for validation. The file also includes functions for saving ([`common_ngram_cache_save`](#common_ngram_cache_save)) and loading ([`common_ngram_cache_load`](#common_ngram_cache_load)) the cache to and from binary files, ensuring persistence of the n-gram data. Additionally, the [`common_ngram_cache_merge`](#common_ngram_cache_merge) function allows for the combination of two caches, integrating their data.

The code is a collection of functions that provide a cohesive set of operations for managing n-gram caches, which are likely used in natural language processing or similar applications where token sequences are analyzed and predicted. The file includes both utility functions and core operations, making it a comprehensive module for n-gram cache management. It does not define a public API or external interfaces directly, but it is intended to be integrated into a larger system where these functions are called to perform specific tasks related to n-gram analysis and manipulation. The use of helper functions and constants indicates a focus on efficiency and accuracy in token drafting and cache management.
# Imports and Dependencies

---
- `ngram-cache.h`
- `common.h`
- `log.h`
- `cinttypes`
- `cstdint`
- `cstdio`
- `fstream`
- `thread`
- `algorithm`


# Global Variables

---
### draft\_min\_sample\_size\_lax
- **Type**: `constexpr int`
- **Description**: `draft_min_sample_size_lax` is a constant integer array with a size defined by `LLAMA_NGRAM_MAX`. It contains threshold values for the minimum sample size required for a draft to proceed under lax conditions.
- **Use**: This array is used to determine if the sample size is sufficient to continue drafting a token in the `try_draft` function.


---
### draft\_min\_percent\_lax
- **Type**: `constexpr int[]`
- **Description**: The `draft_min_percent_lax` is a constant array of integers that defines the minimum percentage thresholds for drafting operations in a lax mode. It contains four elements, each corresponding to a different n-gram size, with values 66, 50, 50, and 50.
- **Use**: This array is used to determine if a draft operation should proceed based on the percentage of the most frequent token in the n-gram cache compared to the total count of tokens.


---
### draft\_min\_sample\_size\_strict
- **Type**: `constexpr int`
- **Description**: The `draft_min_sample_size_strict` is a constant integer array that defines the minimum sample size required for strict drafting conditions in a n-gram based token drafting system. It is indexed by the n-gram size, with each element specifying the minimum number of samples needed for that n-gram size.
- **Use**: This array is used to determine if the sample size is sufficient to proceed with drafting a token under strict conditions.


---
### draft\_min\_percent\_strict
- **Type**: `constexpr int[]`
- **Description**: The `draft_min_percent_strict` is a constant array of integers that defines the minimum percentage thresholds for strict drafting conditions across different n-gram sizes, as specified by the `LLAMA_NGRAM_MAX` constant. The array contains four elements with values {75, 66, 66, 66}, which correspond to the minimum percentage thresholds for each n-gram size.
- **Use**: This array is used to determine whether a draft should be aborted early based on strict percentage criteria during the token drafting process.


# Functions

---
### common\_ngram\_cache\_update<!-- {{#callable:common_ngram_cache_update}} -->
The `common_ngram_cache_update` function updates an n-gram cache with new tokens from a given input sequence, tracking occurrences of each token following specific n-grams.
- **Inputs**:
    - `ngram_cache`: A reference to a `common_ngram_cache` object that stores n-grams and their associated token counts.
    - `ngram_min`: An integer specifying the minimum size of n-grams to consider.
    - `ngram_max`: An integer specifying the maximum size of n-grams to consider.
    - `inp`: A reference to a vector of `llama_token` objects representing the input sequence of tokens.
    - `nnew`: An integer indicating the number of new tokens to process from the input sequence.
    - `print_progress`: A boolean flag indicating whether to print progress updates during the cache update process.
- **Control Flow**:
    - Initialize the start time and input size variables.
    - Calculate the total number of n-grams to process based on input size and n-gram range.
    - Iterate over n-gram sizes from `ngram_min` to `ngram_max`.
    - For each n-gram size, determine the starting index for processing based on input size and `nnew`.
    - Iterate over the input sequence from the starting index to the end, creating n-grams and updating the cache.
    - For each n-gram, check if it exists in the cache; if not, add it with the current token and a count of 1.
    - If the n-gram exists, update the count of the current token in the cache.
    - Increment the count of processed n-grams and, if `print_progress` is true, print progress updates every 10 million n-grams.
- **Output**: The function does not return a value; it updates the `ngram_cache` in place.


---
### get\_token<!-- {{#callable:get_token}} -->
The `get_token` function retrieves a token from a combined sequence of input and draft vectors based on a specified index.
- **Inputs**:
    - `inp`: A constant reference to a vector of `llama_token` representing the input sequence.
    - `draft`: A constant reference to a vector of `llama_token` representing the draft sequence.
    - `i`: A size_t index indicating the position of the token to retrieve from the combined sequence.
- **Control Flow**:
    - Check if the index `i` is less than the size of the `inp` vector.
    - If true, return the token at index `i` from the `inp` vector.
    - If false, calculate the index in the `draft` vector as `1 + i - inp.size()` and return the token from the `draft` vector at this calculated index.
- **Output**: Returns a `llama_token` from either the `inp` or `draft` vector based on the index `i`.


---
### try\_draft<!-- {{#callable:try_draft}} -->
The `try_draft` function attempts to draft a token from a primary n-gram cache, validating it against a static cache, based on specified minimum sample size and percentage thresholds.
- **Inputs**:
    - `nc_primary`: A reference to the primary n-gram cache, which is a data structure mapping n-grams to their token counts.
    - `ngrams_primary`: A vector of primary n-grams to be considered for drafting a token.
    - `part_static`: A reference to a static n-gram cache part, used for validation of the drafted token.
    - `min_sample_size`: A pointer to an array of minimum sample sizes required for each n-gram size.
    - `min_percent`: A pointer to an array of minimum percentage thresholds required for each n-gram size.
- **Control Flow**:
    - Initialize `drafted_token` to `LLAMA_TOKEN_NULL` to indicate no token has been drafted yet.
    - Iterate over the `ngrams_primary` vector in reverse order, stopping if a token is successfully drafted.
    - For each n-gram, check if it exists in the `nc_primary` cache; if not, continue to the next n-gram.
    - Retrieve the token counts for the current n-gram from the primary cache.
    - Initialize variables to track the maximum token count and sum of token counts in the primary cache.
    - Iterate over each token in the primary cache part, checking if it also exists in the static cache and calculating a combined score based on primary and static counts.
    - Update the maximum token and its counts if the current token's score is higher than the current maximum.
    - After iterating through tokens, check if the sum of primary counts meets the minimum sample size; if not, continue to the next n-gram.
    - Check if the maximum primary count meets the minimum percentage threshold; if not, continue to the next n-gram.
    - If both conditions are met, set `drafted_token` to the token with the highest score.
    - Return the `drafted_token`, which may still be `LLAMA_TOKEN_NULL` if no suitable token was found.
- **Output**: Returns a `llama_token`, which is either the drafted token or `LLAMA_TOKEN_NULL` if no suitable token was found.


---
### common\_ngram\_cache\_draft<!-- {{#callable:common_ngram_cache_draft}} -->
The `common_ngram_cache_draft` function attempts to draft a sequence of tokens by leveraging static, dynamic, and context n-gram caches to predict and append tokens to a draft sequence.
- **Inputs**:
    - `inp`: A reference to a vector of `llama_token` representing the input sequence.
    - `draft`: A reference to a vector of `llama_token` representing the draft sequence, which initially contains one token.
    - `n_draft`: An integer specifying the number of tokens to draft.
    - `ngram_min`: An integer specifying the minimum size of n-grams to consider.
    - `ngram_max`: An integer specifying the maximum size of n-grams to consider.
    - `nc_context`: A reference to a `common_ngram_cache` representing the context n-gram cache.
    - `nc_dynamic`: A reference to a `common_ngram_cache` representing the dynamic n-gram cache.
    - `nc_static`: A reference to a `common_ngram_cache` representing the static n-gram cache.
- **Control Flow**:
    - Assert that the draft size is initially 1.
    - Calculate the size of the input sequence and return immediately if it is smaller than `LLAMA_NGRAM_STATIC`.
    - Enter a loop that continues until the draft size minus one is less than `n_draft`.
    - Initialize `drafted_token` to `LLAMA_TOKEN_NULL`.
    - Calculate the starting index for static n-grams and populate `ngram_static` with tokens from the input and draft sequences.
    - Search for `ngram_static` in the static cache and retrieve its associated part if found.
    - Iterate over n-gram sizes from `ngram_min` to `ngram_max` to populate `ngrams_cd` with context and dynamic n-grams.
    - Attempt to draft a token using the context cache, then the dynamic cache, and finally the static cache, updating `drafted_token` if successful.
    - Break the loop if no token is drafted (i.e., `drafted_token` remains `LLAMA_TOKEN_NULL`).
    - Log the drafted token and append it to the draft sequence.
- **Output**: The function modifies the `draft` vector by appending drafted tokens to it, potentially up to `n_draft` tokens.
- **Functions called**:
    - [`get_token`](#get_token)
    - [`try_draft`](#try_draft)


---
### common\_ngram\_cache\_save<!-- {{#callable:common_ngram_cache_save}} -->
The `common_ngram_cache_save` function serializes and writes the contents of a `common_ngram_cache` to a binary file.
- **Inputs**:
    - `ngram_cache`: A reference to a `common_ngram_cache` object, which is a data structure containing n-grams and their associated token counts.
    - `filename`: A reference to a `std::string` representing the name of the file where the cache will be saved.
- **Control Flow**:
    - Open a binary output file stream using the provided filename.
    - Iterate over each n-gram and its associated token counts in the `ngram_cache`.
    - For each n-gram, assert that the token counts are not empty and determine the number of tokens.
    - Write the n-gram and the number of tokens to the file.
    - Iterate over each token and its count in the token counts.
    - For each token, assert that the count is greater than zero, then write the token and its count to the file.
- **Output**: The function does not return any value; it writes the serialized n-gram cache to the specified file.


---
### common\_ngram\_cache\_load<!-- {{#callable:common_ngram_cache_load}} -->
The function `common_ngram_cache_load` reads a binary file containing n-gram data and populates a `common_ngram_cache` with this data.
- **Inputs**:
    - `filename`: A reference to a string representing the name of the file to be loaded.
- **Control Flow**:
    - Open the file specified by `filename` in binary mode using an `ifstream` object.
    - Check if the file was successfully opened; if not, throw an exception.
    - Initialize a `common_ngram_cache` object to store the loaded n-gram data.
    - Declare variables for `common_ngram`, `int32_t` for number of tokens, `llama_token`, and `int32_t` for count.
    - Use a while loop to read `common_ngram` objects from the file until EOF is reached.
    - For each `common_ngram`, read the number of tokens and ensure it is greater than zero.
    - Initialize a `common_ngram_cache_part` to store token counts for the current n-gram.
    - For each token in the n-gram, read the token and its count, ensuring the count is greater than zero, and store them in `common_ngram_cache_part`.
    - Add the `common_ngram` and its associated `common_ngram_cache_part` to the `common_ngram_cache`.
    - Assert that the end of the file has been reached after reading all data.
    - Return the populated `common_ngram_cache`.
- **Output**: A `common_ngram_cache` object containing the n-gram data read from the file.


---
### common\_ngram\_cache\_merge<!-- {{#callable:common_ngram_cache_merge}} -->
The `common_ngram_cache_merge` function merges two n-gram caches by adding the counts of matching n-grams and tokens from the source cache to the target cache.
- **Inputs**:
    - `ngram_cache_target`: A reference to the target `common_ngram_cache` where the n-grams and their counts will be merged into.
    - `ngram_cache_add`: A reference to the source `common_ngram_cache` whose n-grams and their counts will be added to the target cache.
- **Control Flow**:
    - Iterate over each n-gram and its associated cache part in `ngram_cache_add`.
    - For each n-gram, check if it exists in `ngram_cache_target`.
    - If the n-gram does not exist in `ngram_cache_target`, add it along with its cache part.
    - If the n-gram exists, iterate over each token and its count in the cache part.
    - For each token, check if it exists in the corresponding cache part in `ngram_cache_target`.
    - If the token does not exist, add it with its count.
    - If the token exists, increment its count by the count from `ngram_cache_add`.
- **Output**: The function does not return a value; it modifies `ngram_cache_target` in place by merging data from `ngram_cache_add`.


