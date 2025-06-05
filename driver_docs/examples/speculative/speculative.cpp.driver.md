# Purpose
This C++ source code file is an executable program designed to perform speculative sampling using two language models, referred to as the "target" and "draft" models. The program initializes and loads these models, then processes input text (a prompt) to generate predictions. The primary functionality revolves around speculative sampling, where the draft model generates multiple potential sequences (drafts) of tokens, and the target model verifies these sequences. The program uses a probabilistic approach to decide whether to accept or reject tokens from the draft sequences, aiming to efficiently generate text by leveraging the draft model's predictions while ensuring accuracy with the target model.

Key technical components include the initialization and management of the language models, tokenization of input text, and the speculative sampling process. The code defines a `seq_draft` structure to manage draft sequences, and it uses various functions to handle model initialization, token sampling, and sequence management. The program also includes logging and error handling to ensure proper execution and debugging. The code is structured to be executed as a standalone application, with a [`main`](#main) function that orchestrates the entire process, from parsing command-line arguments to performing the speculative sampling and outputting the results.
# Imports and Dependencies

---
- `arg.h`
- `common.h`
- `sampling.h`
- `log.h`
- `llama.h`
- `algorithm`
- `cstdio`
- `cstring`
- `random`
- `set`
- `string`
- `vector`


# Data Structures

---
### seq\_draft<!-- {{#data_structure:seq_draft}} -->
- **Type**: `struct`
- **Members**:
    - `active`: Indicates whether the sequence is currently active.
    - `drafting`: Indicates whether the sequence is in the drafting phase.
    - `skip`: Indicates whether the sequence should be skipped.
    - `i_batch_dft`: Stores the index of the current batch in the draft sequence.
    - `i_batch_tgt`: Holds indices of the target batch for the sequence.
    - `tokens`: Contains a list of tokens associated with the sequence.
    - `dists`: Stores distributions of token data for each token in the sequence.
    - `smpl`: Pointer to a common_sampler structure for sampling operations.
- **Description**: The `seq_draft` struct is designed to manage and track the state of a sequence during a drafting process in a speculative execution model. It includes flags to indicate the activity and drafting status of the sequence, as well as mechanisms to skip certain sequences. The struct also maintains indices for batch processing, a list of tokens, and their associated distributions. Additionally, it holds a pointer to a `common_sampler` structure, which is used for sampling operations during the drafting process.


# Functions

---
### main<!-- {{#callable:main}} -->
The `main` function initializes and runs a speculative language model generation process using both target and draft models, handling input parsing, model loading, tokenization, and speculative sampling with error handling and performance logging.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize `common_params` and set `params.sampling.n_probs` to 128.
    - Parse command-line arguments using [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse); return 1 on failure.
    - Check if `params.n_predict` is valid; log error and return 1 if not.
    - Initialize common resources with `common_init`.
    - Check if the draft model path is provided; log error and return 1 if not.
    - Initialize random number generator and distribution for sampling.
    - Initialize llama backend and NUMA settings.
    - Load target and draft models using `common_init_from_params`.
    - Verify that vocabularies of target and draft models match; log errors and return 1 if mismatched.
    - Tokenize the input prompt and check its length; log error and return 1 if too long.
    - Evaluate the prompt with both models and log the time taken.
    - Initialize variables for speculative sampling, including draft sequences and samplers.
    - Enter a loop to perform speculative sampling, checking and accepting tokens based on probabilities.
    - Log performance metrics and free resources before exiting.
- **Output**: Returns 0 on successful execution, or 1 if any error occurs during initialization, model loading, or token verification.
- **Functions called**:
    - [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse)
    - [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_numa_init`](../../src/llama.cpp.driver.md#llama_numa_init)
    - [`llama_vocab_type`](../../include/llama.h.driver.md#llama_vocab_type)
    - [`common_sampler_init`](../../common/sampling.cpp.driver.md#common_sampler_init)
    - [`common_sampler_sample`](../../common/sampling.cpp.driver.md#common_sampler_sample)
    - [`common_sampler_get_candidates`](../../common/sampling.cpp.driver.md#common_sampler_get_candidates)
    - [`common_sampler_accept`](../../common/sampling.cpp.driver.md#common_sampler_accept)
    - [`common_sampler_free`](../../common/sampling.cpp.driver.md#common_sampler_free)
    - [`common_sampler_clone`](../../common/sampling.cpp.driver.md#common_sampler_clone)
    - [`common_perf_print`](../../common/sampling.cpp.driver.md#common_perf_print)
    - [`llama_backend_free`](../../src/llama.cpp.driver.md#llama_backend_free)


