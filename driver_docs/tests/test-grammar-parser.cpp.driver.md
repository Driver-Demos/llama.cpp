# Purpose
This C++ source code file is designed to test the functionality of a grammar parser, specifically for a library named "llama." The file includes a series of test cases that verify the parsing of various grammar rules and expressions. The primary function, [`verify_parsing`](#verify_parsing), checks if the parsed grammar matches expected symbol IDs and rules, while [`verify_failure`](#verify_failure) ensures that certain invalid grammar inputs fail as expected. The code uses assertions to validate the correctness of the parsing process, and it provides detailed error messages to assist in debugging when expectations are not met. The file includes a main function that executes a series of predefined grammar tests, demonstrating the parser's ability to handle different grammar constructs, such as character ranges, optional elements, and repetitions.

The file includes several technical components, such as the `llama_grammar_parser` class, which is responsible for parsing the grammar strings, and the `llama_grammar_element` structure, which represents individual elements of the grammar. The [`type_str`](#type_str) function is used to convert grammar element types to human-readable strings, aiding in debugging and error reporting. The code is structured to facilitate testing and debugging of the grammar parser, with the ability to print detailed information about the parsed grammar when an environment variable is set. This file is not intended to be a standalone executable but rather a test suite for validating the functionality of the llama grammar parsing library.
# Imports and Dependencies

---
- `llama.h`
- `../src/llama-grammar.h`
- `cassert`


# Functions

---
### type\_str<!-- {{#callable:type_str}} -->
The `type_str` function returns a string representation of a given `llama_gretype` enumeration value.
- **Inputs**:
    - `type`: An enumeration value of type `llama_gretype` which represents a specific grammar element type.
- **Control Flow**:
    - The function uses a switch statement to match the input `type` against various `llama_gretype` enumeration values.
    - For each case, it returns a corresponding string literal that represents the enumeration value.
    - If the `type` does not match any known enumeration values, it returns a default string "?".
- **Output**: A constant character pointer to a string that represents the name of the `llama_gretype` enumeration value.


---
### verify\_parsing<!-- {{#callable:verify_parsing}} -->
The `verify_parsing` function checks if a given grammar string is correctly parsed into expected symbols and rules, and asserts the correctness of the parsing against expected values.
- **Inputs**:
    - `grammar_bytes`: A C-style string representing the grammar to be parsed.
    - `expected`: A vector of pairs, where each pair contains a string and a uint32_t, representing the expected symbol names and their corresponding IDs.
    - `expected_rules`: A vector of `llama_grammar_element` objects representing the expected parsed rules.
- **Control Flow**:
    - Initialize an index to zero and create a `llama_grammar_parser` object to parse the `grammar_bytes`.
    - Populate a map `symbol_names` with symbol IDs as keys and their corresponding names as values from the parsed grammar.
    - Define a lambda function `print_all` to print the current grammar parsing state and expected values for debugging purposes.
    - Check if the environment variable `TEST_GRAMMAR_PARSER_PRINT_ALL` is set; if so, call `print_all` and return.
    - Print the grammar being tested and check if the number of parsed symbols matches the expected size; if not, print the update code and assert the sizes match.
    - Iterate over the parsed symbols, comparing each with the expected pair; print detailed error messages and assert equality if they do not match.
    - Reset the index and iterate over the parsed rules, comparing each element with the expected rule element; print detailed error messages and assert equality if they do not match.
- **Output**: The function does not return a value but asserts the correctness of the parsed grammar against expected values, potentially printing debugging information to stderr.
- **Functions called**:
    - [`type_str`](#type_str)


---
### verify\_failure<!-- {{#callable:verify_failure}} -->
The `verify_failure` function tests whether parsing a given grammar string results in a failure by asserting that the parsed rules are empty.
- **Inputs**:
    - `grammar_bytes`: A C-style string representing the grammar to be parsed, which is expected to fail.
- **Control Flow**:
    - Prints a message to standard error indicating the grammar being tested for expected failure.
    - Initializes a `llama_grammar_parser` object named `result`.
    - Calls the `parse` method on `result` with `grammar_bytes` as the argument to attempt parsing the grammar.
    - Asserts that the `rules` vector in `result` is empty, indicating a parsing failure, and triggers an assertion failure if not.
- **Output**: The function does not return any value; it asserts that the parsing fails by checking if the `rules` vector is empty.


---
### main<!-- {{#callable:main}} -->
The `main` function tests various grammar parsing scenarios using the [`verify_failure`](#verify_failure) and [`verify_parsing`](#verify_parsing) functions to ensure correct parsing and expected failures of grammar rules.
- **Inputs**: None
- **Control Flow**:
    - The function begins by calling [`verify_failure`](#verify_failure) with a grammar string that should fail parsing due to incorrect repetition syntax, expecting an empty rule set.
    - It then calls [`verify_parsing`](#verify_parsing) multiple times with different grammar strings, expected symbol mappings, and expected parsing rules to validate correct parsing behavior.
    - Each call to [`verify_parsing`](#verify_parsing) checks if the parsed grammar matches the expected symbol IDs and rules, asserting if there are discrepancies.
    - The function concludes by returning 0, indicating successful execution.
- **Output**: The function returns an integer 0, indicating successful execution of all tests.
- **Functions called**:
    - [`verify_failure`](#verify_failure)
    - [`verify_parsing`](#verify_parsing)


