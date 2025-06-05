# Purpose
This C++ source code file is designed to test and benchmark various sampling strategies for a language model, likely related to the "llama" model, as indicated by the inclusion of "llama.h" and the use of llama-specific functions and data structures. The file includes a main function, making it an executable program. The primary functionality revolves around testing different sampling techniques such as temperature sampling, top-k sampling, top-p sampling, and others, which are common in natural language processing tasks to generate text from probabilistic models. The code defines a [`sampler_tester`](#sampler_testersampler_tester) class that facilitates the testing of these sampling methods by applying them to a set of probabilities and comparing the results against expected outcomes.

The file is structured to include a series of static functions that each test a specific sampling method. These functions utilize the [`sampler_tester`](#sampler_testersampler_tester) class to initialize, apply, and verify the results of the sampling strategies. The code also includes a benchmarking function, [`bench`](#bench), which measures the performance of these sampling methods in terms of execution time per iteration. The use of macros like `DUMP` and `BENCH` aids in debugging and performance evaluation. The file is comprehensive in its approach, covering a wide range of sampling scenarios and ensuring that the implemented methods meet expected performance and accuracy standards. This makes it a critical component for validating and optimizing the sampling mechanisms used in the language model's text generation process.
# Imports and Dependencies

---
- `ggml.h`
- `llama.h`
- `algorithm`
- `cmath`
- `string`
- `vector`


# Global Variables

---
### llama\_sampler\_init\_dry\_testing
- **Type**: `struct llama_sampler *`
- **Description**: The `llama_sampler_init_dry_testing` is a global function that returns a pointer to a `llama_sampler` structure. It is designed to initialize a sampler for dry testing with specific parameters such as context size, dry multiplier, dry base, allowed length, penalty for the last n tokens, and sequence breakers.
- **Use**: This function is used to create and configure a `llama_sampler` for testing purposes with specific dry run parameters.


# Data Structures

---
### sampler\_tester<!-- {{#data_structure:sampler_tester}} -->
- **Type**: `struct`
- **Members**:
    - `cur_p`: A public member of type `llama_token_data_array` that holds the current token data array.
    - `probs_expected`: A private constant vector of floats representing the expected probabilities for the tokens.
    - `cur`: A private vector of `llama_token_data` that stores the current token data.
- **Description**: The `sampler_tester` struct is designed to facilitate testing of token sampling mechanisms by managing a collection of token data and expected probabilities. It initializes token data arrays based on vocabulary size or given probabilities, applies sampling functions, and verifies the results against expected probabilities. The struct provides a mechanism to test various sampling strategies by applying them to the token data and checking the outcomes.
- **Member Functions**:
    - [`sampler_tester::sampler_tester`](#sampler_testersampler_tester)
    - [`sampler_tester::sampler_tester`](#sampler_testersampler_tester)
    - [`sampler_tester::apply`](#sampler_testerapply)
    - [`sampler_tester::check`](#sampler_testercheck)

**Methods**

---
#### sampler\_tester::sampler\_tester<!-- {{#callable:sampler_tester::sampler_tester}} -->
The `sampler_tester` constructor initializes a `sampler_tester` object by reserving space for tokens and populating it with token data based on the given vocabulary size.
- **Inputs**:
    - `n_vocab`: The number of vocabulary tokens to initialize the sampler with.
- **Control Flow**:
    - Reserve space in the `cur` vector for `n_vocab` tokens.
    - Iterate over each token ID from 0 to `n_vocab - 1`.
    - For each token ID, calculate its logit using the natural logarithm function `logf`.
    - Create a `llama_token_data` object with the token ID, calculated logit, and a probability of 0.0f, and add it to the `cur` vector.
    - Initialize `cur_p` as a `llama_token_data_array` using the data from `cur`, its size, and default values for other fields.
- **Output**: The function does not return a value; it initializes the `sampler_tester` object with token data.
- **See also**: [`sampler_tester`](#sampler_tester)  (Data Structure)


---
#### sampler\_tester::sampler\_tester<!-- {{#callable:sampler_tester::sampler_tester}} -->
The `sampler_tester` constructor initializes a `sampler_tester` object with token data based on given probabilities and expected probabilities.
- **Inputs**:
    - `probs`: A vector of floats representing the probabilities of each token.
    - `probs_expected`: A vector of floats representing the expected probabilities of each token after sampling.
- **Control Flow**:
    - The constructor reserves space in the `cur` vector for the number of tokens equal to the size of `probs`.
    - It iterates over each token ID from 0 to the size of `probs`, calculating the log of each probability and creating a `llama_token_data` object with the token ID, logit, and probability.
    - Each `llama_token_data` object is added to the `cur` vector.
    - The `cur_p` member is initialized as a `llama_token_data_array` with the data from `cur`, its size, and default values for other fields.
- **Output**: The constructor does not return a value; it initializes the `sampler_tester` object.
- **See also**: [`sampler_tester`](#sampler_tester)  (Data Structure)


---
#### sampler\_tester::apply<!-- {{#callable:sampler_tester::apply}} -->
The `apply` function applies a given `llama_sampler` to the current token data array and then frees the sampler.
- **Inputs**:
    - `sampler`: A pointer to a `llama_sampler` object that will be applied to the current token data array `cur_p`.
- **Control Flow**:
    - The function calls `llama_sampler_apply` with the provided `sampler` and the current token data array `cur_p`.
    - After applying the sampler, it calls `llama_sampler_free` to release the resources associated with the `sampler`.
- **Output**: This function does not return any value; it performs operations on the `sampler` and `cur_p` in place.
- **See also**: [`sampler_tester`](#sampler_tester)  (Data Structure)


---
#### sampler\_tester::check<!-- {{#callable:sampler_tester::check}} -->
The `check` function verifies that the size of the `cur_p` array matches the size of the `probs_expected` vector and that each probability in `cur_p` is within a small tolerance of the corresponding expected probability.
- **Inputs**: None
- **Control Flow**:
    - Assert that the size of `cur_p` is equal to the size of `probs_expected`.
    - Iterate over each element in `cur_p`.
    - For each element, assert that the absolute difference between the probability in `cur_p` and the corresponding expected probability is less than `1e-5`.
- **Output**: The function does not return any value; it performs assertions to ensure the correctness of the data.
- **See also**: [`sampler_tester`](#sampler_tester)  (Data Structure)



# Functions

---
### dump<!-- {{#callable:dump}} -->
The `dump` function prints the id, probability, and logit of each token in a given `llama_token_data_array`.
- **Inputs**:
    - `cur_p`: A pointer to a `llama_token_data_array` structure containing an array of token data, including size, id, probability, and logit values.
- **Control Flow**:
    - Iterates over each element in the `llama_token_data_array` using a for loop.
    - For each token, it prints the token's id, probability (p), and logit using the `printf` function.
- **Output**: The function does not return any value; it outputs formatted text to the standard output.


---
### test\_temp<!-- {{#callable:test_temp}} -->
The `test_temp` function tests the application of temperature-based sampling on a given set of probabilities and checks if the resulting probabilities match the expected values.
- **Inputs**:
    - `probs`: A vector of floats representing the initial probabilities of tokens.
    - `probs_expected`: A vector of floats representing the expected probabilities after applying the temperature-based sampling.
    - `temp`: A float representing the temperature parameter to be used in the sampling process.
- **Control Flow**:
    - Initialize a `sampler_tester` object with the given `probs` and `probs_expected`.
    - Dump the current state of probabilities using the `DUMP` macro.
    - Apply temperature-based sampling using `llama_sampler_init_temp(temp)` and update the probabilities.
    - Apply a distribution initialization with `llama_sampler_init_dist(0)` to further update the probabilities.
    - Dump the updated state of probabilities using the `DUMP` macro again.
    - Check if the current probabilities match the expected probabilities using the `check` method of `sampler_tester`.
- **Output**: The function does not return any value; it performs assertions to ensure the probabilities match the expected values.


---
### test\_temp\_ext<!-- {{#callable:test_temp_ext}} -->
The `test_temp_ext` function tests the extended temperature-based sampling method by applying it to a set of probabilities and comparing the results to expected values.
- **Inputs**:
    - `probs`: A vector of float values representing the initial probabilities of tokens.
    - `probs_expected`: A vector of float values representing the expected probabilities after sampling.
    - `temp`: A float value representing the temperature parameter for the sampling method.
    - `delta`: A float value representing the delta parameter for the extended temperature sampling.
    - `exponent`: A float value representing the exponent parameter for the extended temperature sampling.
- **Control Flow**:
    - Initialize a `sampler_tester` object with the given `probs` and `probs_expected`.
    - Dump the current state of the `sampler_tester`'s probability array using the `DUMP` macro.
    - Apply the extended temperature-based sampler initialized with `temp`, `delta`, and `exponent` to the `sampler_tester`.
    - Apply a distribution sampler initialized with 0 to the `sampler_tester`.
    - Dump the current state of the `sampler_tester`'s probability array again using the `DUMP` macro.
    - Check that the resulting probabilities match the expected probabilities within a small tolerance.
- **Output**: The function does not return a value; it performs assertions to ensure the sampled probabilities match the expected values.


---
### test\_top\_k<!-- {{#callable:test_top_k}} -->
The `test_top_k` function tests the top-k sampling method by applying it to a given set of probabilities and comparing the result to expected probabilities.
- **Inputs**:
    - `probs`: A vector of float values representing the initial probabilities of tokens.
    - `probs_expected`: A vector of float values representing the expected probabilities after applying the top-k sampling.
    - `k`: An integer specifying the number of top probabilities to retain during sampling.
- **Control Flow**:
    - Initialize a `sampler_tester` object with the given probabilities and expected probabilities.
    - Dump the current state of the `sampler_tester`'s probability array using the `DUMP` macro.
    - Apply the top-k sampling method by initializing a sampler with `llama_sampler_init_top_k(k)` and applying it to the tester.
    - Apply a distribution initialization with `llama_sampler_init_dist(0)` to the tester.
    - Dump the state of the `sampler_tester`'s probability array again to observe changes.
    - Check the final probabilities against the expected probabilities using the `check` method of `sampler_tester`.
- **Output**: The function does not return a value; it performs assertions to ensure the sampled probabilities match the expected probabilities.


---
### test\_top\_p<!-- {{#callable:test_top_p}} -->
The `test_top_p` function tests the behavior of a top-p sampling strategy on a given set of probabilities and compares the results to expected probabilities.
- **Inputs**:
    - `probs`: A vector of float values representing the initial probabilities of tokens.
    - `probs_expected`: A vector of float values representing the expected probabilities after applying the top-p sampling.
    - `p`: A float value representing the top-p threshold for sampling.
- **Control Flow**:
    - Initialize a `sampler_tester` object with the given probabilities and expected probabilities.
    - Dump the current state of the `sampler_tester` object using the `DUMP` macro.
    - Apply the top-p sampling strategy using `llama_sampler_init_top_p` with the given threshold `p`.
    - Apply a distribution initialization using `llama_sampler_init_dist`.
    - Dump the state of the `sampler_tester` object again to observe changes.
    - Check if the resulting probabilities match the expected probabilities using the `check` method of `sampler_tester`.
- **Output**: The function does not return a value; it performs assertions to ensure the sampled probabilities match the expected probabilities.


---
### test\_min\_p<!-- {{#callable:test_min_p}} -->
The `test_min_p` function tests the behavior of a sampler initialized with a minimum probability threshold by comparing the resulting probabilities against expected values.
- **Inputs**:
    - `probs`: A vector of float values representing the initial probabilities of tokens.
    - `probs_expected`: A vector of float values representing the expected probabilities after applying the sampler.
    - `p`: A float value representing the minimum probability threshold for the sampler.
- **Control Flow**:
    - A `sampler_tester` object is instantiated with the given `probs` and `probs_expected` vectors.
    - The current state of the sampler (`cur_p`) is dumped to the console for debugging purposes.
    - The `llama_sampler_init_min_p` function is called with the minimum probability `p` and applied to the sampler using the `apply` method of `sampler_tester`.
    - The `llama_sampler_init_dist` function is called with a parameter of 0 and applied to the sampler.
    - The current state of the sampler (`cur_p`) is dumped again to the console for debugging purposes.
    - The `check` method of `sampler_tester` is called to assert that the resulting probabilities match the expected probabilities within a small tolerance.
- **Output**: The function does not return a value; it performs assertions to validate the sampler's behavior.


---
### test\_xtc<!-- {{#callable:test_xtc}} -->
The `test_xtc` function tests the XTC sampling method by applying it to a set of probabilities and comparing the results to expected probabilities.
- **Inputs**:
    - `probs`: A vector of float values representing the initial probabilities of tokens.
    - `probs_expected`: A vector of float values representing the expected probabilities after applying the sampler.
    - `p`: A float value representing a parameter for the XTC sampler.
    - `t`: A float value representing another parameter for the XTC sampler.
- **Control Flow**:
    - Initialize a `sampler_tester` object with the given `probs` and `probs_expected` vectors.
    - Dump the current state of the `sampler_tester`'s probability array using the `DUMP` macro.
    - Apply the XTC sampler initialized with parameters `p` and `t` to the `sampler_tester`.
    - Dump the state of the `sampler_tester`'s probability array again after applying the sampler.
    - Check if the resulting probabilities match the expected probabilities using the `check` method of `sampler_tester`.
- **Output**: The function does not return a value; it performs assertions to ensure the sampler's output matches the expected probabilities.


---
### test\_typical<!-- {{#callable:test_typical}} -->
The `test_typical` function tests the typical sampling method by initializing a sampler with given probabilities and expected results, applying the typical sampling, and verifying the output.
- **Inputs**:
    - `probs`: A vector of float values representing the probabilities of different tokens.
    - `probs_expected`: A vector of float values representing the expected probabilities after sampling.
    - `p`: A float value representing the typical sampling parameter.
- **Control Flow**:
    - Initialize a `sampler_tester` object with `probs` and `probs_expected`.
    - Dump the current state of the sampler using the `DUMP` macro.
    - Apply the typical sampling method using `llama_sampler_init_typical` with parameter `p`.
    - Dump the state of the sampler again to observe changes.
    - Call the `check` method of `sampler_tester` to assert that the current probabilities match the expected probabilities.
- **Output**: The function does not return a value; it performs assertions to verify the correctness of the sampling process.


---
### test\_penalties<!-- {{#callable:test_penalties}} -->
The `test_penalties` function tests the application of penalties on a set of probabilities using a sampler initialized with specific penalty parameters.
- **Inputs**:
    - `probs`: A vector of float values representing the initial probabilities of tokens.
    - `last_tokens`: A vector of `llama_token` representing the last tokens to be considered for penalty application.
    - `probs_expected`: A vector of float values representing the expected probabilities after penalties are applied.
    - `repeat_penalty`: A float value representing the penalty for repeating tokens.
    - `alpha_frequency`: A float value representing the frequency penalty parameter.
    - `alpha_presence`: A float value representing the presence penalty parameter.
- **Control Flow**:
    - Assert that the size of `probs` is equal to the size of `probs_expected`.
    - Initialize a `sampler_tester` object with `probs` and `probs_expected`.
    - Initialize a sampler with penalties using `llama_sampler_init_penalties` with the size of `last_tokens` and the penalty parameters.
    - Iterate over `last_tokens` and accept each token into the sampler using `llama_sampler_accept`.
    - Dump the current state of the tester's probabilities using the `DUMP` macro.
    - Apply the sampler to the tester's current probabilities using `tester.apply`.
    - Apply a distribution initialization sampler with `llama_sampler_init_dist(0)` to the tester.
    - Dump the current state of the tester's probabilities again using the `DUMP` macro.
    - Check the tester's probabilities against the expected probabilities using `tester.check`.
- **Output**: The function does not return a value; it performs assertions to ensure the probabilities after applying penalties match the expected values.


---
### test\_dry<!-- {{#callable:test_dry}} -->
The `test_dry` function tests a dry run of a sampling process using a given set of probabilities, tokens, and parameters, and verifies the output against expected probabilities.
- **Inputs**:
    - `probs`: A vector of float values representing the initial probabilities for each token.
    - `last_tokens`: A vector of `llama_token` representing the last tokens processed in the sequence.
    - `expected_probs`: A vector of float values representing the expected probabilities after processing.
    - `dry_multiplier`: A float value used as a multiplier in the dry run sampling process.
    - `dry_base`: A float value used as a base in the dry run sampling process.
    - `dry_allowed_length`: An integer specifying the allowed length for the dry run.
    - `dry_penalty_last_n`: An integer specifying the number of last tokens to apply a penalty to in the dry run.
    - `seq_breakers`: A vector of vectors of `llama_token` representing sequences that can break the sampling process.
- **Control Flow**:
    - Assert that the size of `probs` matches the size of `expected_probs`.
    - Initialize a `sampler_tester` object with `probs` and `expected_probs`.
    - Initialize a sampler using [`llama_sampler_init_dry_testing`](../src/llama-sampling.cpp.driver.md#llama_sampler_init_dry_testing) with the provided parameters.
    - Iterate over `last_tokens` and accept each token into the sampler.
    - Dump the current state of the sampler using the `DUMP` macro.
    - Apply the sampler to the `sampler_tester`.
    - Apply a distribution initialization sampler with `llama_sampler_init_dist(0)`.
    - Dump the current state again using the `DUMP` macro.
    - Check the results using the `tester.check()` method to ensure they match `expected_probs`.
- **Output**: The function does not return a value but performs assertions to ensure the sampler's output matches the expected probabilities.
- **Functions called**:
    - [`llama_sampler_init_dry_testing`](../src/llama-sampling.cpp.driver.md#llama_sampler_init_dry_testing)


---
### test\_top\_n\_sigma<!-- {{#callable:test_top_n_sigma}} -->
The `test_top_n_sigma` function tests the behavior of a sampler initialized with a top-n sigma strategy on a given set of probabilities and expected results.
- **Inputs**:
    - `probs`: A vector of float values representing the initial probabilities of tokens.
    - `probs_expected`: A vector of float values representing the expected probabilities after applying the sampler.
    - `n`: An integer representing the parameter for the top-n sigma strategy.
- **Control Flow**:
    - A `sampler_tester` object is instantiated with the given `probs` and `probs_expected` vectors.
    - The current state of the sampler is dumped using the `DUMP` macro.
    - The `llama_sampler_init_top_n_sigma` function is called with `n` to initialize a sampler, which is then applied to the `sampler_tester` object.
    - The `llama_sampler_init_dist` function is called with 0 to apply a distribution initialization to the sampler.
    - The current state of the sampler is dumped again using the `DUMP` macro.
    - The `check` method of `sampler_tester` is called to assert that the resulting probabilities match the expected probabilities.
- **Output**: The function does not return a value; it performs assertions to ensure the sampler's output matches the expected probabilities.


---
### test\_sampler\_queue<!-- {{#callable:test_sampler_queue}} -->
The `test_sampler_queue` function tests a sequence of sampling strategies on a vocabulary of tokens, ensuring that the resulting token distributions meet expected criteria based on the specified sampling parameters.
- **Inputs**:
    - `n_vocab`: The size of the vocabulary, representing the total number of tokens available.
    - `samplers_sequence`: A string representing the sequence of samplers to be applied, where each character corresponds to a specific sampling strategy ('k' for top-k, 'p' for top-p, 'm' for min-p, etc.).
    - `top_k`: The maximum number of top tokens to consider in the top-k sampling strategy.
    - `top_p`: The cumulative probability threshold for the top-p sampling strategy.
    - `min_p`: The minimum probability threshold for the min-p sampling strategy.
- **Control Flow**:
    - Initialize a `sampler_tester` object with the given vocabulary size `n_vocab`.
    - Set `min_token_id` to 0 and `max_token_id` to `n_vocab - 1`.
    - Iterate over each character in `samplers_sequence` to determine the sampling strategy to apply.
    - For each character, apply the corresponding sampler to the `sampler_tester` object using the `apply` method.
    - After applying the sampler, apply a distribution initialization sampler with `llama_sampler_init_dist(0)`.
    - Retrieve the current probability distribution `cur_p` from the `sampler_tester`.
    - Depending on the sampler type ('k', 'p', or 'm'), calculate the expected size of the token distribution and adjust `min_token_id` accordingly.
    - Use assertions to verify that the size of the current probability distribution matches the expected size and that the token IDs at the boundaries are correct.
    - Print a confirmation message indicating the successful application of the sampler queue with the given parameters.
- **Output**: The function does not return a value but prints a confirmation message indicating the successful execution of the sampler queue with the specified parameters.


---
### bench<!-- {{#callable:bench}} -->
The `bench` function measures the average time taken per iteration to apply a sampler to a set of token data over a specified number of iterations.
- **Inputs**:
    - `cnstr`: A pointer to a `llama_sampler` object that will be applied to the token data.
    - `cnstr_name`: A string representing the name of the sampler, used for output display.
    - `data`: A constant reference to a vector of `llama_token_data` objects, representing the token data to be processed.
    - `n_iter`: An integer specifying the number of iterations to perform the sampling process.
- **Control Flow**:
    - Initialize a vector `cur` with the same size as `data` and copy the contents of `data` into `cur`.
    - Create a `llama_token_data_array` `cur_p` from `cur` and apply the sampler `cnstr` to `cur_p`.
    - Reset the sampler `cnstr` after applying it.
    - Record the start time using `ggml_time_us()`.
    - Iterate `n_iter` times, copying `data` to `cur`, creating `cur_p`, applying the sampler, and resetting it each time.
    - Record the end time after the loop completes.
    - Free the sampler `cnstr` to release resources.
    - Calculate the average time per iteration and print it along with the sampler name.
- **Output**: The function outputs the average time per iteration in microseconds, formatted and printed to the console.


---
### test\_perf<!-- {{#callable:test_perf}} -->
The `test_perf` function benchmarks various llama sampler initialization methods using a large set of randomly generated token data.
- **Inputs**: None
- **Control Flow**:
    - Initialize a constant `n_vocab` to 2^17, representing the number of vocabulary tokens.
    - Create a vector `data` to store `llama_token_data` objects and reserve space for `n_vocab` elements.
    - Populate the `data` vector with `n_vocab` elements, each having a random logit value calculated from a uniform distribution.
    - Call the `BENCH` macro to benchmark different llama sampler initialization methods (`top_k`, `top_p`, `min_p`, `typical`, `xtc`) using the `data` vector and 32 iterations each.
- **Output**: The function does not return any value; it outputs benchmark results to the console.


---
### main<!-- {{#callable:main}} -->
The `main` function initializes timing, executes a series of test functions for various sampling strategies, and performs performance benchmarking.
- **Inputs**: None
- **Control Flow**:
    - Initialize timing with `ggml_time_init()`.
    - Execute [`test_temp`](#test_temp) with different probability vectors and temperature values.
    - Execute [`test_temp_ext`](#test_temp_ext) with additional parameters for delta and exponent.
    - Execute [`test_top_k`](#test_top_k) with different probability vectors and k-values.
    - Execute [`test_top_p`](#test_top_p) with different probability vectors and p-values.
    - Execute [`test_min_p`](#test_min_p) with different probability vectors and p-values.
    - Print messages indicating expected and unexpected outcomes for [`test_xtc`](#test_xtc).
    - Execute [`test_xtc`](#test_xtc) with different probability vectors and parameters.
    - Execute [`test_typical`](#test_typical) with different probability vectors and p-values.
    - Execute [`test_penalties`](#test_penalties) with different probability vectors, last tokens, and penalty parameters.
    - Execute [`test_dry`](#test_dry) with different probability vectors, last tokens, and dry parameters.
    - Execute [`test_top_n_sigma`](#test_top_n_sigma) with different probability vectors and n-values.
    - Execute [`test_sampler_queue`](#test_sampler_queue) with different vocabulary sizes, sampler sequences, and parameters.
    - Print 'OK' to indicate successful execution of tests.
    - Execute [`test_perf`](#test_perf) to benchmark performance of various samplers.
    - Return 0 to indicate successful completion.
- **Output**: The function returns an integer value of 0, indicating successful execution.
- **Functions called**:
    - [`test_temp`](#test_temp)
    - [`test_temp_ext`](#test_temp_ext)
    - [`test_top_k`](#test_top_k)
    - [`test_top_p`](#test_top_p)
    - [`test_min_p`](#test_min_p)
    - [`test_xtc`](#test_xtc)
    - [`test_typical`](#test_typical)
    - [`test_penalties`](#test_penalties)
    - [`test_dry`](#test_dry)
    - [`test_top_n_sigma`](#test_top_n_sigma)
    - [`test_sampler_queue`](#test_sampler_queue)
    - [`test_perf`](#test_perf)


