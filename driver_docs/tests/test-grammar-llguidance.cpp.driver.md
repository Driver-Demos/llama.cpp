# Purpose
This C++ source code file is designed to perform integration tests for a grammar and schema validation system, specifically using a library or framework related to "llama" and "llguidance." The file includes a main function, indicating that it is an executable program. The primary purpose of the code is to validate the functionality of grammar and schema parsing, tokenization, and sampling processes. It achieves this by defining a series of test cases that check the correctness of grammar rules and JSON schema constraints against various input strings. The tests cover a wide range of scenarios, including simple and complex grammars, special characters, quantifiers, and JSON schema properties like type, minimum and maximum values, and pattern matching.

The code is structured around several static functions that encapsulate different test scenarios, such as [`test_simple_grammar`](#test_simple_grammar), [`test_complex_grammar`](#test_complex_grammar), [`test_special_chars`](#test_special_chars), [`test_quantifiers`](#test_quantifiers), and [`test_json_schema`](#test_json_schema). Each function defines specific test cases with expected passing and failing strings, which are then validated using the [`match_string`](#match_string) function and other helper functions. The code also includes a [`test_sampler_chain`](#test_sampler_chain) function to test the integration of multiple samplers in a chain. The use of assertions ensures that the tests fail if the actual behavior does not match the expected results. The file relies on external components like `llama_vocab`, `llama_sampler`, and related functions, indicating that it is part of a larger system or library focused on language processing or grammar validation.
# Imports and Dependencies

---
- `sampling.h`
- `cassert`
- `string`
- `vector`


# Global Variables

---
### vocab
- **Type**: ``const llama_vocab *``
- **Description**: The `vocab` variable is a global pointer to a constant `llama_vocab` structure. It is used to store the vocabulary data required for tokenization and other operations related to language processing.
- **Use**: This variable is used throughout the code to access vocabulary-related functions and data, such as tokenizing input strings and retrieving the number of tokens in the vocabulary.


# Functions

---
### match\_string<!-- {{#callable:match_string}} -->
The `match_string` function checks if a given input string can be matched by a specified grammar using a token-based approach.
- **Inputs**:
    - `input`: A constant reference to a `std::string` representing the input string to be matched against the grammar.
    - `grammar`: A pointer to a `llama_sampler` object representing the grammar used for matching the input string.
- **Control Flow**:
    - The function begins by resetting the grammar using `llama_sampler_reset`.
    - The input string is tokenized using `common_tokenize`, resulting in a vector of tokens.
    - The number of tokens in the vocabulary is retrieved using `llama_vocab_n_tokens`.
    - A vector `cur` of `llama_token_data` is initialized and reserved to hold token data for each token in the vocabulary.
    - A `llama_token_data_array` named `tok_arr` is created to hold the token data array.
    - For each token in the input string, the logit values in `cur` are reset to 0.0.
    - The grammar is applied to `tok_arr` using `llama_sampler_apply`.
    - If the logit of the current token is less than 0.0, the function returns `false`, indicating a mismatch.
    - If the token is accepted by the grammar, `llama_sampler_accept` is called.
    - After processing all tokens, the function checks if the End Of String (EOS) token is allowed by the grammar.
    - The EOS token is identified using `llama_vocab_eot` or `llama_vocab_eos`.
    - The logit for the EOS token is set to 0.0, and the grammar is applied again.
    - The function returns `true` if the EOS token's logit is greater than or equal to 0.0, indicating a successful match.
- **Output**: A boolean value indicating whether the input string matches the specified grammar (`true` for a match, `false` otherwise).


---
### test<!-- {{#callable:test}} -->
The `test` function evaluates a given grammar against a set of strings to determine if they match or fail according to the grammar rules, and provides debugging information for mismatches.
- **Inputs**:
    - `test_desc`: A description of the test being performed, used for logging purposes.
    - `grammar_str`: A string representing the grammar rules to be tested.
    - `passing_strings`: A vector of strings that are expected to match the grammar.
    - `failing_strings`: A vector of strings that are expected to not match the grammar.
- **Control Flow**:
    - Log the test description and grammar string to standard error.
    - Initialize a grammar object using `llama_sampler_init_llg` with the provided grammar string.
    - Log the section header for valid strings to standard error.
    - Iterate over each string in `passing_strings`, log the string, and check if it matches the grammar using [`match_string`](#match_string).
    - If a string in `passing_strings` does not match, log an error, write the grammar and string to files for debugging, and provide a command for further analysis.
    - Assert that each string in `passing_strings` matches the grammar.
    - Log the section header for invalid strings to standard error.
    - Iterate over each string in `failing_strings`, log the string, and check if it matches the grammar using [`match_string`](#match_string).
    - If a string in `failing_strings` matches, log an error.
    - Assert that each string in `failing_strings` does not match the grammar.
    - Free the grammar object using `llama_sampler_free`.
- **Output**: The function does not return a value; it logs results to standard output and standard error, and may write files for debugging purposes.
- **Functions called**:
    - [`match_string`](#match_string)


---
### test\_grammar<!-- {{#callable:test_grammar}} -->
The `test_grammar` function runs a grammar test by invoking the [`test`](#test) function with a description, grammar string, and lists of passing and failing strings.
- **Inputs**:
    - `test_desc`: A description of the test being performed.
    - `grammar_str`: The grammar string that defines the rules to be tested.
    - `passing_strings`: A vector of strings that are expected to match the grammar.
    - `failing_strings`: A vector of strings that are expected not to match the grammar.
- **Control Flow**:
    - The function constructs a test description by appending the grammar string to the provided test description.
    - It calls the [`test`](#test) function with the constructed test description, the grammar string, and the vectors of passing and failing strings.
- **Output**: The function does not return any value; it is a void function.
- **Functions called**:
    - [`test`](#test)


---
### test\_schema<!-- {{#callable:test_schema}} -->
The `test_schema` function tests a JSON schema against a set of passing and failing strings by converting the schema into a grammar and invoking the [`test`](#test) function.
- **Inputs**:
    - `test_desc`: A description of the test being performed, used for logging and identification purposes.
    - `schema_str`: A string representation of the JSON schema to be tested.
    - `passing_strings`: A vector of strings that are expected to pass validation against the schema.
    - `failing_strings`: A vector of strings that are expected to fail validation against the schema.
- **Control Flow**:
    - The function constructs a test description by appending the schema string to the provided test description.
    - It constructs a grammar string by prefixing the schema string with '%llguidance {}\nstart: %json '.
    - The function calls the [`test`](#test) function with the constructed test description, grammar string, passing strings, and failing strings.
- **Output**: The function does not return a value; it performs testing and outputs results to standard error and standard output.
- **Functions called**:
    - [`test`](#test)


---
### test\_simple\_grammar<!-- {{#callable:test_simple_grammar}} -->
The `test_simple_grammar` function tests various JSON schema constraints and a simple arithmetic grammar against a set of passing and failing string inputs to validate their correctness.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`test_schema`](#test_schema) multiple times with different JSON schema definitions, each specifying constraints like minimum, maximum, and exclusive bounds for integers.
    - For each schema, it provides a list of strings that should pass and a list that should fail the schema validation.
    - The function also calls [`test_grammar`](#test_grammar) with a simple arithmetic grammar and provides strings that should pass or fail the grammar validation.
    - Each call to [`test_schema`](#test_schema) or [`test_grammar`](#test_grammar) internally uses the `test` function to perform the validation and assert the expected outcomes.
- **Output**: The function does not return any value; it performs assertions to ensure that the strings match the expected outcomes based on the provided schemas and grammar.
- **Functions called**:
    - [`test_schema`](#test_schema)
    - [`test_grammar`](#test_grammar)


---
### test\_complex\_grammar<!-- {{#callable:test_complex_grammar}} -->
The `test_complex_grammar` function tests a medium complexity grammar by verifying that a set of strings either match or fail to match the defined grammar rules.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - The function calls [`test_grammar`](#test_grammar) with a description, a grammar string, a list of passing strings, and a list of failing strings.
    - The grammar string defines a set of rules for parsing expressions, terms, factors, numbers, variables, function calls, and whitespace.
    - The [`test_grammar`](#test_grammar) function checks each string in the passing list to ensure it matches the grammar and each string in the failing list to ensure it does not match.
    - If a passing string fails to match or a failing string matches, the function logs an error and asserts to ensure correctness.
- **Output**: The function does not return any value; it performs assertions to validate the grammar against the test strings.
- **Functions called**:
    - [`test_grammar`](#test_grammar)


---
### test\_special\_chars<!-- {{#callable:test_special_chars}} -->
The `test_special_chars` function tests a grammar that matches strings containing the sequence 'abc' surrounded by any three characters, including multi-byte characters.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`test_grammar`](#test_grammar) with a description 'special characters', a grammar string, a list of passing strings, and a list of failing strings.
    - The grammar string specifies a pattern where 'abc' is surrounded by any three characters on both sides.
    - The passing strings include examples like 'abcabcabc', 'aaaabcccc', and a string with multi-byte characters surrounding 'abc'.
    - The failing strings include examples that do not match the pattern, such as 'aaabcccc' and strings with incorrect placement of 'abc'.
- **Output**: The function does not return any value; it asserts that the strings match or do not match the grammar as expected.
- **Functions called**:
    - [`test_grammar`](#test_grammar)


---
### test\_quantifiers<!-- {{#callable:test_quantifiers}} -->
The `test_quantifiers` function performs a series of tests on different quantifiers (*, +, ?, and repetition) in grammar rules to validate their behavior with various input strings.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`test_grammar`](#test_grammar) multiple times, each with a different grammar rule and corresponding passing and failing strings.
    - For each grammar rule, it specifies a description, the grammar itself, a list of strings that should pass the grammar, and a list of strings that should fail.
    - The [`test_grammar`](#test_grammar) function is responsible for executing the tests and asserting the correctness of the grammar against the provided strings.
- **Output**: The function does not return any value; it performs assertions to ensure the grammar rules behave as expected with the given input strings.
- **Functions called**:
    - [`test_grammar`](#test_grammar)


---
### test\_json\_schema<!-- {{#callable:test_json_schema}} -->
The `test_json_schema` function validates various JSON schemas by converting them into grammars and testing them against a set of passing and failing strings.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`test_schema`](#test_schema) multiple times, each with a different JSON schema, a description, and sets of passing and failing strings.
    - Each call to [`test_schema`](#test_schema) converts the JSON schema into a grammar and then tests the provided strings to see if they match the schema.
    - The function uses assertions to ensure that passing strings match the schema and failing strings do not.
    - If a string does not match as expected, debug information is written to files for further analysis.
- **Output**: The function does not return any value; it performs assertions to validate the schemas and will terminate the program if any assertion fails.
- **Functions called**:
    - [`test_schema`](#test_schema)


---
### one\_hot<!-- {{#callable:one_hot}} -->
The `one_hot` function sets the logit of a selected token to a high value while setting all other tokens' logits to zero in a given token data array.
- **Inputs**:
    - `tok_arr`: A reference to a `llama_token_data_array` structure that contains token data and metadata.
    - `selected`: A `llama_token` representing the index of the token to be highlighted with a high logit value.
- **Control Flow**:
    - Retrieve the size of the vocabulary from `tok_arr.size` and store it in `n_vocab`.
    - Set `tok_arr.selected` to -1 and `tok_arr.sorted` to false to reset selection and sorting status.
    - Iterate over each token ID from 0 to `n_vocab - 1`.
    - For each token, set its `id` to the current token ID and its `logit` to 0.0f.
    - Set the `logit` of the token at the `selected` index to 100.0f.
- **Output**: The function does not return a value; it modifies the `tok_arr` in place.


---
### test\_sampler\_chain<!-- {{#callable:test_sampler_chain}} -->
The `test_sampler_chain` function initializes a sampler chain with specific parameters and grammar, processes a given input string by tokenizing it, applies the sampler to each token, and verifies the output against expected results.
- **Inputs**: None
- **Control Flow**:
    - Initialize default sampler chain parameters and set `no_perf` to false.
    - Initialize a sampler chain with the specified parameters.
    - Define grammar data for the sampler chain.
    - Add two samplers to the chain: one initialized with grammar data and another with a distribution.
    - Tokenize the input string using the vocabulary.
    - Determine the number of tokens in the vocabulary.
    - Create a vector to store token data and reserve space for the number of vocabulary tokens.
    - Initialize a token data array with the vector data.
    - Iterate over each token in the tokenized input string.
    - For each token, apply a one-hot encoding to the token data array.
    - Apply the sampler to the token data array and print the applied token and selected token information.
    - Assert that the selected token matches the current token and accept the token in the sampler.
    - Determine the end-of-sequence token from the vocabulary.
    - Apply a one-hot encoding to the token data array for the end-of-sequence token.
    - Apply the sampler to the token data array and assert that the selected token matches the end-of-sequence token.
- **Output**: The function does not return any value; it performs assertions to verify the correctness of the sampler chain processing.
- **Functions called**:
    - [`llama_sampler_chain_default_params`](../src/llama.cpp.driver.md#llama_sampler_chain_default_params)
    - [`one_hot`](#one_hot)


---
### main<!-- {{#callable:main}} -->
The `main` function initializes the llama backend, loads a vocabulary file, and runs a series of integration tests on grammar and sampling functionalities.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments, where `argv[0]` is the program name and `argv[1]` should be the path to the vocabulary file.
- **Control Flow**:
    - Prints a message indicating the start of integration tests.
    - Checks if the number of arguments is exactly 2; if not, prints usage information and returns 1.
    - Extracts the vocabulary file path from `argv[1]`.
    - Initializes the llama backend.
    - Loads the vocabulary model from the specified file with default parameters set to `vocab_only`.
    - If the model fails to load, prints an error message and returns 1.
    - Initializes the llama context from the loaded model.
    - If the context fails to initialize, prints an error message, frees the model, and returns 1.
    - Retrieves the vocabulary from the loaded model.
    - Runs a series of test functions: [`test_simple_grammar`](#test_simple_grammar), [`test_complex_grammar`](#test_complex_grammar), [`test_special_chars`](#test_special_chars), [`test_quantifiers`](#test_quantifiers), [`test_json_schema`](#test_json_schema), and [`test_sampler_chain`](#test_sampler_chain).
    - Prints a message indicating all tests passed and returns 0.
- **Output**: Returns 0 if all tests pass successfully, or 1 if there is an error in loading the vocabulary or initializing the context.
- **Functions called**:
    - [`llama_backend_init`](../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_model_default_params`](../src/llama-model.cpp.driver.md#llama_model_default_params)
    - [`llama_context_default_params`](../src/llama-context.cpp.driver.md#llama_context_default_params)
    - [`test_simple_grammar`](#test_simple_grammar)
    - [`test_complex_grammar`](#test_complex_grammar)
    - [`test_special_chars`](#test_special_chars)
    - [`test_quantifiers`](#test_quantifiers)
    - [`test_json_schema`](#test_json_schema)
    - [`test_sampler_chain`](#test_sampler_chain)


