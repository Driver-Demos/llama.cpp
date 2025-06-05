# Purpose
This C++ source code file is designed to perform various natural language processing (NLP) tasks, primarily focusing on evaluating language models using metrics such as perplexity, KL divergence, and multiple-choice question answering. The file includes a main function that initializes the necessary parameters and context for running these evaluations. It imports several headers, indicating dependencies on external libraries or modules for argument parsing, logging, and model handling.

The code defines several structures and functions to compute metrics like perplexity and KL divergence, which are crucial for assessing the performance of language models. It includes functions for processing logits, calculating softmax and log-softmax values, and handling multi-threaded computations to improve performance. The file also contains specific functions for evaluating tasks like HellaSwag, Winogrande, and multiple-choice questions, which are common benchmarks in NLP. These functions tokenize input data, compute probabilities, and compare model predictions against ground truth to calculate accuracy and other metrics. The code is structured to handle large datasets efficiently, leveraging multi-threading and optimized mathematical computations.
# Imports and Dependencies

---
- `arg.h`
- `common.h`
- `log.h`
- `llama.h`
- `chrono`
- `algorithm`
- `array`
- `atomic`
- `cmath`
- `cstdio`
- `cstring`
- `ctime`
- `fstream`
- `mutex`
- `random`
- `sstream`
- `thread`
- `vector`


# Data Structures

---
### results\_perplexity<!-- {{#data_structure:results_perplexity}} -->
- **Type**: `struct`
- **Members**:
    - `tokens`: A vector of llama_token representing the sequence of tokens.
    - `ppl_value`: A double representing the perplexity value.
    - `logits`: A vector of floats representing the logits for each token.
    - `probs`: A vector of floats representing the probabilities for each token.
- **Description**: The `results_perplexity` struct is designed to encapsulate the results of a perplexity calculation in a language model context. It contains a sequence of tokens, the calculated perplexity value, and the associated logits and probabilities for each token. This struct is useful for storing and accessing the detailed results of a perplexity evaluation, which is a common metric used to assess the performance of language models.


---
### results\_log\_softmax<!-- {{#data_structure:results_log_softmax}} -->
- **Type**: `struct`
- **Members**:
    - `log_softmax`: Stores the log of the softmax value as a double.
    - `logit`: Represents the logit value as a float.
    - `prob`: Holds the probability value as a float.
- **Description**: The `results_log_softmax` struct is designed to encapsulate the results of a log softmax computation. It contains three fields: `log_softmax`, which stores the logarithm of the softmax value; `logit`, which represents the logit value; and `prob`, which holds the probability value. This struct is useful for handling and storing the results of log softmax operations in a structured manner, facilitating further processing or analysis.


---
### kl\_divergence\_result<!-- {{#data_structure:kl_divergence_result}} -->
- **Type**: `struct`
- **Members**:
    - `sum_nll`: Represents the sum of negative log-likelihood values.
    - `sum_nll2`: Represents the sum of squared negative log-likelihood values.
    - `sum_nll_base`: Represents the sum of base negative log-likelihood values.
    - `sum_nll_base2`: Represents the sum of squared base negative log-likelihood values.
    - `sum_nll_nll_base`: Represents the sum of the product of negative log-likelihood and base negative log-likelihood values.
    - `sum_kld`: Represents the sum of Kullback-Leibler divergence values.
    - `sum_kld2`: Represents the sum of squared Kullback-Leibler divergence values.
    - `sum_p_diff`: Represents the sum of probability differences.
    - `sum_p_diff2`: Represents the sum of squared probability differences.
    - `sum_p_diff4`: Represents the sum of fourth power of probability differences.
    - `max_p_diff`: Stores the maximum probability difference encountered.
    - `n_same_top`: Counts the number of times the top probability is the same.
    - `count`: Counts the number of observations or iterations.
- **Description**: The `kl_divergence_result` struct is designed to store various statistical measures related to the calculation of Kullback-Leibler divergence and negative log-likelihoods. It includes fields for summing these values and their squares, as well as tracking probability differences and their maximum values. This struct is useful for accumulating and analyzing data over multiple iterations or samples, particularly in the context of evaluating model performance or comparing distributions.


---
### hs\_data\_t<!-- {{#data_structure:hellaswag_score::hs_data_t}} -->
- **Type**: `struct`
- **Members**:
    - `context`: A string representing the context of the task.
    - `gold_ending_idx`: An index indicating the correct ending among the four options.
    - `ending`: An array of four strings, each representing a possible ending for the context.
    - `ending_logprob_count`: An array of four size_t values, each representing the count of log probabilities for the corresponding ending.
    - `ending_logprob`: An array of four doubles, each representing the log probability of the corresponding ending.
    - `i_logits`: The starting index of logits in the llama_batch.
    - `common_prefix`: The maximum number of initial tokens that are the same in all sentences.
    - `required_tokens`: The number of tokens needed to evaluate all four endings.
    - `seq_tokens`: An array of four vectors, each containing llama_tokens for the corresponding ending sequence.
- **Description**: The `hs_data_t` struct is designed to hold data related to a task in the HellaSwag dataset, which involves evaluating multiple possible endings for a given context. It stores the context, possible endings, and their respective log probabilities, as well as metadata such as the index of the correct ending, the common prefix length among endings, and the required tokens for evaluation. This struct is used to facilitate the processing and scoring of tasks in the HellaSwag benchmark.


---
### winogrande\_entry<!-- {{#data_structure:winogrande_entry}} -->
- **Type**: `struct`
- **Members**:
    - `first`: A string representing the first part of a sentence.
    - `second`: A string representing the second part of a sentence.
    - `choices`: An array of two strings representing the two choices for filling the blank in the sentence.
    - `answer`: An integer representing the correct choice index (1 or 2).
    - `i_logits`: A size_t representing the starting index of logits in the llama_batch.
    - `common_prefix`: A size_t representing the maximum number of initial tokens that are the same in both sequences.
    - `required_tokens`: A size_t representing the number of tokens needed to evaluate both choices.
    - `n_base1`: A size_t representing the number of tokens for context plus choice 1.
    - `n_base2`: A size_t representing the number of tokens for context plus choice 2.
    - `seq_tokens`: An array of two vectors of llama_token representing the tokenized sequences for each choice.
- **Description**: The `winogrande_entry` struct is designed to represent an entry in the Winogrande dataset, which is used for evaluating language models on tasks that require commonsense reasoning. Each entry consists of a sentence split into two parts, with a blank to be filled by one of two given choices. The struct stores the sentence parts, the choices, and the correct answer. Additionally, it includes metadata for processing the entry, such as token indices and counts, which are used to evaluate the model's performance on the task.


---
### multiple\_choice\_answers<!-- {{#data_structure:multiple_choice_answers}} -->
- **Type**: `struct`
- **Members**:
    - `answers`: A vector of strings representing the possible answers for a multiple-choice question.
    - `labels`: A vector of integers representing labels associated with each answer, typically used to indicate correctness.
- **Description**: The `multiple_choice_answers` struct is designed to store and manage multiple-choice answers for a question. It contains two main members: `answers`, which is a vector of strings holding the possible answers, and `labels`, a vector of integers that typically represent the correctness of each answer. The struct also includes a `deserialize` method that reads data from an input stream to populate the `answers` and `labels` vectors, ensuring that the number of answers does not exceed a practical limit of 100.
- **Member Functions**:
    - [`multiple_choice_answers::deserialize`](#multiple_choice_answersdeserialize)

**Methods**

---
#### multiple\_choice\_answers::deserialize<!-- {{#callable:multiple_choice_answers::deserialize}} -->
The `deserialize` function reads data from an input stream to populate the `answers` and `labels` vectors of a `multiple_choice_answers` structure, ensuring the data is valid and within expected limits.
- **Inputs**:
    - `in`: An input stream (`std::istream&`) from which the function reads serialized data.
- **Control Flow**:
    - Read a 32-bit unsigned integer `n` from the input stream to determine the number of answers.
    - Check if the read operation failed or if `n` exceeds 100; if so, return `false`.
    - Resize the `answers` and `labels` vectors to accommodate `n` elements.
    - Iterate over the `answers` vector and use [`deserialize_string`](#deserialize_string) to populate each string; return `false` if any deserialization fails.
    - Read `n` integers from the input stream into the `labels` vector.
    - Return `true` if all operations succeed without stream failures.
- **Output**: A boolean value indicating whether the deserialization was successful (`true`) or if an error occurred (`false`).
- **Functions called**:
    - [`deserialize_string`](#deserialize_string)
- **See also**: [`multiple_choice_answers`](#multiple_choice_answers)  (Data Structure)



---
### multiple\_choice\_task<!-- {{#data_structure:multiple_choice_task}} -->
- **Type**: `struct`
- **Members**:
    - `question`: A string representing the question or context that needs to be continued.
    - `mc1`: An instance of `multiple_choice_answers` representing possible answers with a single correct answer.
    - `mc2`: An instance of `multiple_choice_answers` representing possible answers with multiple correct answers, not yet handled.
    - `i_logits`: A size_t indicating the starting index of logits in the llama_batch.
    - `common_prefix`: A size_t representing the maximum number of initial tokens that are the same in all sentences.
    - `required_tokens`: A size_t indicating the needed number of tokens to evaluate all answers.
    - `seq_tokens`: A vector of vectors of `llama_token` representing sequences of tokens for each answer.
    - `log_probs`: A vector of floats representing the log probabilities of the answers.
- **Description**: The `multiple_choice_task` struct is designed to represent a multiple-choice question task, including the question itself and two sets of possible answers, one with a single correct answer and another with multiple correct answers. It also includes fields for evaluation purposes, such as the starting index of logits, the common prefix of tokens, the required number of tokens for evaluation, and the sequences of tokens and their log probabilities. This struct is used in contexts where multiple-choice questions need to be processed and evaluated, particularly in machine learning or natural language processing tasks.
- **Member Functions**:
    - [`multiple_choice_task::deserialize`](#multiple_choice_taskdeserialize)

**Methods**

---
#### multiple\_choice\_task::deserialize<!-- {{#callable:multiple_choice_task::deserialize}} -->
The `deserialize` function reads data from an input stream to populate a `multiple_choice_task` object with a question and two sets of multiple-choice answers.
- **Inputs**:
    - `in`: An input stream (`std::istream&`) from which the function reads data to populate the `multiple_choice_task` object.
- **Control Flow**:
    - The function first attempts to deserialize a string from the input stream into the `question` member of the `multiple_choice_task` object using the [`deserialize_string`](#deserialize_string) function.
    - If the deserialization of the `question` fails, the function returns `false`.
    - If the `question` is successfully deserialized, the function proceeds to deserialize the `mc1` and `mc2` members, which are of type `multiple_choice_answers`, by calling their respective `deserialize` methods.
    - The function returns the logical AND of the results of the deserialization of `mc1` and `mc2`, indicating success only if both are successfully deserialized.
- **Output**: A boolean value indicating whether the deserialization of the `multiple_choice_task` object was successful.
- **Functions called**:
    - [`deserialize_string`](#deserialize_string)
- **See also**: [`multiple_choice_task`](#multiple_choice_task)  (Data Structure)



# Functions

---
### softmax<!-- {{#callable:softmax}} -->
The `softmax` function computes the softmax probabilities of a given vector of logits, ensuring numerical stability.
- **Inputs**:
    - `logits`: A constant reference to a vector of floats representing the input logits for which the softmax probabilities are to be calculated.
- **Control Flow**:
    - Initialize a vector `probs` of the same size as `logits` to store the softmax probabilities.
    - Find the maximum value in `logits` and store it in `max_logit` for numerical stability.
    - Iterate over each element in `logits`, subtract `max_logit` from each logit, compute the exponential, and accumulate the sum of these exponentials in `sum_exp`.
    - Store the exponentials in the `probs` vector.
    - Normalize each element in `probs` by dividing it by `sum_exp` to get the final softmax probabilities.
    - Return the `probs` vector containing the softmax probabilities.
- **Output**: A vector of floats representing the softmax probabilities corresponding to the input logits.


---
### log\_softmax<!-- {{#callable:log_softmax}} -->
The `log_softmax` function computes the log-softmax of a given set of logits, updates a KL divergence result structure, and returns a pair consisting of a sum of weighted differences and a probability difference.
- **Inputs**:
    - `n_vocab`: The number of vocabulary entries, representing the size of the logits array.
    - `logits`: A pointer to an array of float values representing the logits for each vocabulary entry.
    - `base_log_prob`: A pointer to an array of uint16_t values representing the base log probabilities, which is used for comparison and scaling.
    - `tok`: An integer representing the index of the token for which the log-softmax is being calculated.
    - `kld`: A reference to a `kl_divergence_result` structure that accumulates various statistics related to KL divergence and probability differences.
- **Control Flow**:
    - Initialize `max_logit` with the first logit and find the maximum logit value and its index in the `logits` array.
    - Calculate the sum of exponentials of the logits adjusted by the maximum logit for numerical stability.
    - Compute the log of the sum of exponentials (`log_sum_exp`) and retrieve scaling factors from `base_log_prob`.
    - Calculate the negative log-likelihood (`nll`) for the given token and update the `kld` structure with this value and its square.
    - Compute the base negative log-likelihood (`nll_base`) using the base log probabilities and update the `kld` structure with this value and its square.
    - Update the `kld` structure with the product of `nll` and `nll_base`.
    - Adjust `max_logit` by adding `log_sum_exp` and initialize variables for summing weighted differences.
    - Iterate over the vocabulary to compute a weighted sum of differences between base log probabilities and logits, updating the `kld` structure with this sum and its square.
    - Increment the count of processed tokens in the `kld` structure and check if the maximum logit index matches the maximum base log probability index, updating `n_same_top` if they match.
    - Calculate the probability difference between the computed and base probabilities, updating the `kld` structure with this difference, its square, and its fourth power.
    - Update the maximum probability difference in the `kld` structure if the current difference is larger.
    - Return a pair consisting of the sum of weighted differences and the probability difference.
- **Output**: A `std::pair` containing a double representing the sum of weighted differences and a float representing the probability difference.


---
### nearest\_int<!-- {{#callable:nearest_int}} -->
The `nearest_int` function converts a floating-point number to the nearest integer using bit manipulation.
- **Inputs**:
    - `fval`: A floating-point number that needs to be converted to the nearest integer.
- **Control Flow**:
    - The function adds a large constant (12582912.f) to the input float `fval` to manipulate its bits for conversion.
    - It then copies the bit representation of the resulting float into an integer variable `i` using `memcpy`.
    - The function extracts the integer part by applying a bitmask `0x007fffff` to `i` and subtracting a constant `0x00400000` to adjust the result.
- **Output**: Returns the integer closest to the input floating-point number `fval`.


---
### process\_logits<!-- {{#callable:process_logits}} -->
The `process_logits` function processes logits for a given vocabulary size and tokens, computes Kullback-Leibler divergence, and updates the results using multithreading.
- **Inputs**:
    - `n_vocab`: The number of vocabulary entries.
    - `logits`: A pointer to an array of logits, which are the raw prediction scores for each token.
    - `tokens`: A pointer to an array of token indices.
    - `n_token`: The number of tokens to process.
    - `workers`: A reference to a vector of threads used for parallel processing.
    - `base_log_probs`: A constant reference to a vector of base log probabilities used for comparison in KL divergence calculation.
    - `kld`: A reference to a `kl_divergence_result` structure to store the results of the KL divergence calculations.
    - `kld_values`: A pointer to an array where the computed KL divergence values will be stored.
    - `p_diff_values`: A pointer to an array where the probability difference values will be stored.
- **Control Flow**:
    - Initialize a mutex and calculate `nv` as a function of `n_vocab`.
    - Define a lambda function `compute` that processes each token's logits in a thread-safe manner using a mutex to control access to a shared counter.
    - Within the lambda, calculate the log softmax for each token, update local KL divergence results, and store the results in `kld_values` and `p_diff_values`.
    - If all tokens are processed, update the global `kld` results with the local results and exit the loop.
    - Create threads in the `workers` vector to execute the `compute` lambda function.
    - Execute the `compute` function in the main thread as well.
    - Join all threads to ensure completion of processing.
- **Output**: The function does not return a value but updates the `kld`, `kld_values`, and `p_diff_values` with the computed results.
- **Functions called**:
    - [`log_softmax`](#log_softmax)


---
### perplexity\_v2<!-- {{#callable:perplexity_v2}} -->
The `perplexity_v2` function calculates the perplexity of a given text prompt using a language model, evaluating it over multiple chunks and batches, and returns the tokens, perplexity value, and history of logits and probabilities.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which contains the context for the language model evaluation.
    - `params`: A `common_params` structure containing parameters for the perplexity calculation, such as the prompt, stride, number of chunks, batch size, and output type.
- **Control Flow**:
    - Retrieve the model and vocabulary from the context.
    - Check if BOS tokens should be added and assert that EOS tokens should not be added.
    - Tokenize the input prompt and check if there are enough tokens for the context size.
    - Initialize vectors for storing logit and probability history.
    - Check if the stride parameter is valid and calculate the maximum number of chunks.
    - Iterate over each chunk, clearing the KV cache and initializing a batch for each chunk.
    - For each batch, add tokens to the batch and decode them using the model, storing logits.
    - Calculate the probability of the next token using the softmax function and update the negative log-likelihood.
    - Calculate and log the perplexity for each chunk based on the average negative log-likelihood.
    - Return the tokens, calculated perplexity, and history of logits and probabilities.
- **Output**: A `results_perplexity` structure containing the tokens, calculated perplexity value, and vectors of logits and probabilities for each token.
- **Functions called**:
    - [`softmax`](#softmax)


---
### perplexity<!-- {{#callable:perplexity}} -->
The `perplexity` function calculates the perplexity of a given text using a language model, optionally saving logits to a file, and returns the tokens, perplexity value, and history of logits and probabilities.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` which contains the model and its state.
    - `params`: A `common_params` structure containing various parameters for the perplexity calculation, such as prompt, batch size, and file paths.
    - `n_ctx`: An integer representing the context size, i.e., the number of tokens to consider in each evaluation window.
- **Control Flow**:
    - Check if `params.ppl_stride` is greater than 0; if so, call [`perplexity_v2`](#perplexity_v2) and return its result.
    - Retrieve the model and vocabulary from the context.
    - Check if a logits file is specified in `params`; if so, open the file for writing and write initial metadata.
    - Tokenize the input prompt using `common_tokenize`.
    - Check if the number of tokens is sufficient for the given context size; if not, log an error and return.
    - Initialize vectors for storing logit and probability history.
    - Calculate the maximum number of chunks and determine the number of chunks to process based on `params.n_chunks`.
    - Initialize a batch for processing tokens and prepare for multi-threaded processing if necessary.
    - Iterate over chunks, processing each in sequence, and calculate logits using the model's forward pass.
    - For each chunk, calculate the negative log-likelihood (NLL) and update the logit and probability history.
    - Calculate and log the perplexity for each chunk and the final estimate, including standard deviation if applicable.
    - Free the batch resources and return the results containing tokens, perplexity value, and histories.
- **Output**: A `results_perplexity` structure containing the tokens, calculated perplexity value, and histories of logits and probabilities.
- **Functions called**:
    - [`perplexity_v2`](#perplexity_v2)
    - [`process_logits`](#process_logits)


---
### decode\_helper<!-- {{#callable:decode_helper}} -->
The `decode_helper` function processes a batch of tokens in a llama context, decodes them, and stores the resulting logits in a provided vector.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which represents the context in which the decoding is performed.
    - `batch`: A reference to a `llama_batch` object, which contains the tokens to be decoded and other related information.
    - `batch_logits`: A reference to a vector of floats where the decoded logits will be stored.
    - `n_batch`: An integer representing the number of tokens to process in each batch.
    - `n_vocab`: An integer representing the size of the vocabulary, used to determine the size of the logits for each token.
- **Control Flow**:
    - Initialize `prev_outputs` to 0 to keep track of the number of outputs processed so far.
    - Iterate over the tokens in the batch in steps of `n_batch`.
    - For each step, determine the number of tokens to process (`n_tokens`) as the minimum of `n_batch` and the remaining tokens.
    - Create a `batch_view` object representing the current subset of the batch to be processed.
    - Call `llama_decode` with the context and `batch_view`; if it fails, log an error and return false.
    - Count the number of non-zero logits in `batch_view` and update `n_outputs`.
    - Copy the logits from the context to `batch_logits` using `memcpy`, starting at the position indicated by `prev_outputs`.
    - Update `prev_outputs` by adding `n_outputs`.
    - Return true if all batches are successfully processed.
- **Output**: A boolean value indicating whether the decoding was successful for all batches (true) or if an error occurred (false).


---
### compute\_logprobs<!-- {{#callable:compute_logprobs}} -->
The `compute_logprobs` function calculates the log probabilities of specific tokens from a batch of logits using multithreading.
- **Inputs**:
    - `batch_logits`: A pointer to an array of floats representing the logits for a batch of data.
    - `n_vocab`: An integer representing the number of vocabulary tokens.
    - `workers`: A reference to a vector of threads used for parallel computation.
    - `eval_pairs`: A reference to a vector of pairs, where each pair contains a size_t index and a llama_token, representing the positions and tokens to evaluate.
    - `eval_results`: A reference to a vector of floats where the computed log probabilities will be stored.
- **Control Flow**:
    - Check if the size of `eval_results` matches `eval_pairs` and resize if necessary.
    - Return immediately if `eval_pairs` is empty.
    - Determine the maximum number of threads to use based on the size of `eval_pairs` and the number of available workers.
    - Initialize an atomic counter to manage the distribution of work among threads.
    - Define a lambda function `compute` that calculates log probabilities for chunks of tokens.
    - Within the lambda, fetch the next chunk of work using the atomic counter and break if all work is done.
    - For each token in the current chunk, calculate the log probability using the softmax function for numerical stability.
    - Copy the calculated log probabilities into the `eval_results` vector.
    - Launch threads to execute the `compute` lambda function in parallel.
    - Join all threads to ensure completion of all computations.
- **Output**: The function outputs the computed log probabilities in the `eval_results` vector.


---
### hellaswag\_score<!-- {{#callable:hellaswag_score}} -->
The `hellaswag_score` function calculates the HellaSwag score (acc_norm) from a given prompt using a language model context.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which represents the context of the language model.
    - `params`: A constant reference to a `common_params` object, which contains parameters such as the prompt and the number of tasks to evaluate.
- **Control Flow**:
    - Retrieve the model and vocabulary from the context.
    - Split the input prompt into lines and store them in a vector.
    - Check if the number of lines in the prompt is a multiple of 6; if not, log an error and return.
    - Determine the number of tasks to evaluate based on the prompt size and the specified number of tasks in `params`.
    - Randomize the order of tasks if `randomize_tasks` is true.
    - For each task, extract the context, gold ending index, and possible endings from the prompt lines.
    - Tokenize the context and each ending, and calculate the common prefix and required tokens for evaluation.
    - Batch tasks into the available context window, ensuring each task fits within the context size.
    - Decode the batched tasks to obtain logits and compute log probabilities for each ending.
    - Determine the ending with the maximum log probability and compare it to the gold ending index to calculate accuracy.
    - Log the accumulated accuracy and confidence interval for each task batch.
- **Output**: The function does not return a value; it logs the HellaSwag score and confidence intervals for the evaluated tasks.
- **Functions called**:
    - [`llama_vocab_type`](../../include/llama.h.driver.md#llama_vocab_type)
    - [`decode_helper`](#decode_helper)
    - [`compute_logprobs`](#compute_logprobs)
    - [`softmax`](#softmax)


---
### load\_winogrande\_from\_csv<!-- {{#callable:load_winogrande_from_csv}} -->
The function `load_winogrande_from_csv` parses a CSV-formatted string to extract Winogrande task entries and returns them as a vector of `winogrande_entry` structures.
- **Inputs**:
    - `prompt`: A string containing CSV-formatted data where each line represents a Winogrande task with fields: index, sentence, choice1, choice2, and answer.
- **Control Flow**:
    - Initialize an empty vector `result` to store `winogrande_entry` objects.
    - Create an input string stream `in` from the `prompt` string.
    - Iterate over each line in the input stream using a loop until the end of the stream is reached.
    - For each line, initialize an array `comma_pos` to store positions of commas and a boolean `quote_open` to track quoted sections.
    - Iterate over each character in the line to find comma positions, considering quoted sections to avoid counting commas within quotes.
    - If four comma positions are not found, log an error and continue to the next line.
    - Extract the sentence, choice1, choice2, and answer from the line using the comma positions.
    - Find the position of the underscore character in the sentence to split it into two parts: `first` and `second`.
    - Convert the answer from a string to an integer and validate it to be either 1 or 2; log an error if invalid.
    - Create a `winogrande_entry` object with the extracted data and add it to the `result` vector.
    - Return the `result` vector containing all parsed `winogrande_entry` objects.
- **Output**: A vector of `winogrande_entry` structures, each containing the parsed sentence parts, choices, and answer from the CSV data.


---
### winogrande\_score<!-- {{#callable:winogrande_score}} -->
The `winogrande_score` function evaluates the Winogrande score by processing a set of tasks from a CSV file, tokenizing them, and calculating the accuracy of predictions based on token probabilities.
- **Inputs**:
    - `ctx`: A pointer to the `llama_context` which provides the context for the model operations.
    - `params`: A constant reference to `common_params` which contains parameters such as the prompt and the number of tasks to process.
- **Control Flow**:
    - Retrieve the model and vocabulary from the context.
    - Load tasks from a CSV file using the prompt provided in `params`.
    - If no tasks are loaded, log an error and return.
    - Log the number of tasks loaded.
    - If a specific number of tasks is requested, randomly select that many tasks from the loaded data.
    - Tokenize the selected tasks, calculating common prefixes and required tokens for each task.
    - Initialize variables for context size, batch size, and vocabulary size.
    - Iterate over the tasks, batching them based on context size constraints.
    - For each batch, clear the batch context and add tokens, marking where logits are needed.
    - Decode the batch and compute log probabilities for each task's choices.
    - Calculate scores for each choice and determine the correct choice based on scores.
    - Log the accuracy of predictions and calculate the final Winogrande score with confidence intervals.
- **Output**: The function does not return a value but logs the Winogrande score and accuracy statistics.
- **Functions called**:
    - [`load_winogrande_from_csv`](#load_winogrande_from_csv)
    - [`decode_helper`](#decode_helper)
    - [`compute_logprobs`](#compute_logprobs)


---
### deserialize\_string<!-- {{#callable:deserialize_string}} -->
The `deserialize_string` function reads a serialized string from an input stream and stores it in a provided string variable.
- **Inputs**:
    - `in`: A reference to an input stream (`std::istream`) from which the serialized string data will be read.
    - `str`: A reference to a string (`std::string`) where the deserialized string will be stored.
- **Control Flow**:
    - Declare a `uint32_t` variable `size` to hold the size of the string to be read.
    - Attempt to read the size of the string from the input stream `in` into the `size` variable.
    - If reading the size is successful, resize the `str` to the specified `size`.
    - Attempt to read `size` bytes from the input stream `in` into the `str`.
    - If reading the string data is successful, return `true`.
    - If any read operation fails, return `false`.
- **Output**: A boolean value indicating whether the deserialization was successful (`true`) or not (`false`).


---
### multiple\_choice\_prepare\_one\_task<!-- {{#callable:multiple_choice_prepare_one_task}} -->
The function `multiple_choice_prepare_one_task` prepares a multiple choice task by tokenizing the question and answers, and calculates the common prefix and required tokens for evaluation.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which is used for tokenization.
    - `task`: A reference to a `multiple_choice_task` object, which contains the question and possible answers.
    - `log_error`: A boolean flag indicating whether to log errors if the task is malformed.
- **Control Flow**:
    - Check if the task's question or answers are empty; if so, log an error (if `log_error` is true) and return false.
    - Reserve space in `task.seq_tokens` for the number of answers.
    - For each answer, check if it is empty; if so, log an error (if `log_error` is true) and return false.
    - Tokenize the concatenation of the question and each answer, storing the result in `task.seq_tokens`.
    - Determine the minimum length of the tokenized sequences.
    - Calculate the common prefix length by comparing tokens at each position across all sequences.
    - Calculate the total number of required tokens by summing the lengths of the sequences minus the common prefix.
    - Return true if the task is successfully prepared.
- **Output**: A boolean value indicating whether the task was successfully prepared (true) or not (false).


---
### multiple\_choice\_score<!-- {{#callable:multiple_choice_score}} -->
The `multiple_choice_score` function evaluates multiple-choice tasks using a language model, calculating accuracy and providing statistical results.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which represents the context of the language model.
    - `params`: A constant reference to a `common_params` object, which contains parameters for the evaluation, including the prompt and the number of tasks to evaluate.
- **Control Flow**:
    - Retrieve the model and vocabulary from the context.
    - Read the number of tasks from the prompt and check for errors.
    - If no specific number of tasks is specified, read all tasks; otherwise, select a random subset of tasks.
    - Prepare task data, either in parallel if there are more than 500 tasks or sequentially otherwise.
    - Initialize variables for batch processing and evaluation results.
    - Iterate over tasks, batching them into the available context window and preparing them for evaluation.
    - Decode tasks and compute log-probabilities in parallel using multiple threads.
    - Calculate the log-probabilities for each task's possible answers and determine the most likely answer.
    - Accumulate the number of correct answers and total answers, and log the accuracy after each task.
    - If enough tasks are completed, calculate and log the final accuracy and statistical results.
- **Output**: The function does not return a value but logs the accuracy of the multiple-choice tasks and statistical results to the console.
- **Functions called**:
    - [`multiple_choice_prepare_one_task`](#multiple_choice_prepare_one_task)
    - [`decode_helper`](#decode_helper)
    - [`compute_logprobs`](#compute_logprobs)
    - [`softmax`](#softmax)


---
### kl\_divergence<!-- {{#callable:kl_divergence}} -->
The `kl_divergence` function calculates the Kullback-Leibler divergence between a model's predictions and a base model's log probabilities, using data from a specified file.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, representing the current context of the model.
    - `params`: A `common_params` structure containing parameters for the function, including the file name for the base model's log probabilities.
- **Control Flow**:
    - Retrieve the model and vocabulary from the context.
    - Check if the `logits_file` parameter is empty and log an error if it is.
    - Open the specified `logits_file` and check for errors in opening or reading the file header.
    - Read the context size (`n_ctx`) from the file and compare it with the current context size, logging a warning if they differ.
    - Read the vocabulary size (`n_vocab`) and number of chunks (`n_chunk`) from the file, checking for consistency with the current vocabulary.
    - Read evaluation tokens from the file into a vector.
    - Initialize vectors for storing log probabilities, KL divergence values, and probability differences.
    - Iterate over each chunk, processing log probabilities and calculating KL divergence for each batch of tokens.
    - Log the results, including perplexity, KL divergence, and probability differences.
    - Sort and log statistics for KL divergence and token probability differences.
- **Output**: The function does not return a value; it logs various statistics and results related to KL divergence and token probabilities.
- **Functions called**:
    - [`process_logits`](#process_logits)


---
### main<!-- {{#callable:main}} -->
The `main` function initializes parameters, parses command-line arguments, sets up the environment, loads a model, and performs various scoring or evaluation tasks based on the specified parameters.
- **Inputs**:
    - `argc`: An integer representing the number of command-line arguments.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize `common_params` structure with default values for `n_ctx` and `escape`.
    - Parse command-line arguments using [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse); if parsing fails, return 1.
    - Initialize common resources with `common_init`.
    - Check if `n_ctx` is valid; if not, log an error and return 1.
    - Determine if perplexity calculation (`ppl`) is needed based on parameter flags.
    - Adjust parameters for parallel processing and context size based on `ppl` and other flags.
    - Initialize backend and NUMA settings with [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init) and [`llama_numa_init`](../../src/llama.cpp.driver.md#llama_numa_init).
    - Load the model and context using `common_init_from_params`; if loading fails, log an error and return 1.
    - Check if the specified context size exceeds the model's training context size and log a warning if so.
    - Print system information using `common_params_get_system_info`.
    - Perform the appropriate scoring or evaluation task based on the parameters (`hellaswag`, `winogrande`, `multiple_choice`, [`kl_divergence`](#kl_divergence), or [`perplexity`](#perplexity)).
    - Log performance context and free backend resources.
    - Return 0 to indicate successful execution.
- **Output**: Returns an integer status code, 0 for success and 1 for failure.
- **Functions called**:
    - [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse)
    - [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_numa_init`](../../src/llama.cpp.driver.md#llama_numa_init)
    - [`hellaswag_score`](#hellaswag_score)
    - [`winogrande_score`](#winogrande_score)
    - [`multiple_choice_score`](#multiple_choice_score)
    - [`kl_divergence`](#kl_divergence)
    - [`perplexity`](#perplexity)
    - [`llama_backend_free`](../../src/llama.cpp.driver.md#llama_backend_free)


