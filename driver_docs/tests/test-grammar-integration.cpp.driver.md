# Purpose
This C++ source code file is designed to perform integration tests for a grammar processing system, specifically focusing on validating JSON schemas and grammars. The file includes a series of test functions that evaluate the system's ability to parse and validate various grammar rules and JSON schemas. The code utilizes the nlohmann::json library for JSON parsing and includes several custom functions to build and test grammars, such as [`build_grammar`](#build_grammar), [`test_build_grammar_fails`](#test_build_grammar_fails), and [`match_string`](#match_string). These functions are used to construct grammars from strings, test for expected failures, and match input strings against the grammars.

The file is structured to run a comprehensive suite of tests, including simple and complex grammar tests, tests for special characters and quantifiers, and tests for JSON schema validation. It also includes tests for specific failure cases, such as missing root nodes, missing references, and left recursion detection. The main function orchestrates the execution of these tests, outputting results to the console. The code is intended to be executed as a standalone program, as indicated by the presence of the [`main`](#main) function, and it does not define public APIs or external interfaces. The primary purpose of this file is to ensure the robustness and correctness of the grammar processing system by systematically verifying its behavior against a wide range of test cases.
# Imports and Dependencies

---
- `json-schema-to-grammar.h`
- `../src/unicode.h`
- `../src/llama-grammar.h`
- `nlohmann/json.hpp`
- `cassert`
- `string`
- `vector`


# Functions

---
### build\_grammar<!-- {{#callable:build_grammar}} -->
The `build_grammar` function initializes a llama grammar object using a given grammar string.
- **Inputs**:
    - `grammar_str`: A constant reference to a `std::string` that contains the grammar definition to be used for initializing the llama grammar.
- **Control Flow**:
    - The function calls `llama_grammar_init_impl` with the provided `grammar_str` converted to a C-style string using `c_str()`, along with other default parameters such as "root" for the root rule and `false` for some boolean flag.
    - The function returns the result of the `llama_grammar_init_impl` call, which is a pointer to a `llama_grammar` object.
- **Output**: A pointer to a `llama_grammar` object initialized with the provided grammar string.


---
### test\_build\_grammar\_fails<!-- {{#callable:test_build_grammar_fails}} -->
The `test_build_grammar_fails` function checks if building a grammar from a given string fails as expected.
- **Inputs**:
    - `grammar_str`: A string representing the grammar to be tested for build failure.
- **Control Flow**:
    - Prints a message to stderr indicating the start of a failure test for the provided grammar string.
    - Initializes a boolean variable `grammar_fails` to false.
    - Attempts to build a grammar using the [`build_grammar`](#build_grammar) function with the provided `grammar_str`.
    - Checks if the returned grammar pointer is not null, indicating a build success when a failure was expected, and prints an error message to stderr.
    - If the grammar pointer is null, indicating a build failure as expected, sets `grammar_fails` to true and prints a success message to stdout.
    - Returns the value of `grammar_fails`, indicating whether the build failure was correctly detected.
- **Output**: A boolean value indicating whether the grammar build failed as expected (true if it failed, false if it succeeded unexpectedly).
- **Functions called**:
    - [`build_grammar`](#build_grammar)


---
### match\_string<!-- {{#callable:match_string}} -->
The `match_string` function checks if a given input string matches a specified grammar by processing its Unicode code points and evaluating the grammar's acceptance stacks.
- **Inputs**:
    - `input`: A constant reference to a `std::string` representing the input string to be matched against the grammar.
    - `grammar`: A pointer to a `llama_grammar` object that defines the grammar rules against which the input string is to be matched.
- **Control Flow**:
    - Convert the input string into a sequence of Unicode code points using `unicode_cpts_from_utf8`.
    - Retrieve the current grammar stacks using `llama_grammar_get_stacks`.
    - Iterate over each Unicode code point in the input string.
    - For each code point, call `llama_grammar_accept` to process it with the grammar.
    - If the current grammar stacks are empty after processing a code point, return `false` indicating a match failure.
    - After processing all code points, iterate over the current grammar stacks.
    - If any stack is empty, return `true` indicating a successful match.
    - If no stack is empty, return `false` indicating a match failure.
- **Output**: A boolean value indicating whether the input string successfully matches the grammar (`true` for a match, `false` otherwise).


---
### test<!-- {{#callable:test}} -->
The `test` function evaluates a given grammar against a set of passing and failing strings, reporting matches and mismatches, and generating debug files for failed matches.
- **Inputs**:
    - `test_desc`: A description of the test being performed, used for logging purposes.
    - `grammar_str`: A string representation of the grammar to be tested.
    - `passing_strings`: A vector of strings that are expected to match the grammar.
    - `failing_strings`: A vector of strings that are expected not to match the grammar.
- **Control Flow**:
    - Log the test description and grammar string to standard error.
    - Build the grammar from the provided grammar string using [`build_grammar`](#build_grammar).
    - Retrieve and store the original grammar stacks for resetting after each test string.
    - Iterate over each string in `passing_strings`, logging the string and checking if it matches the grammar using [`match_string`](#match_string).
    - If a passing string does not match, log a failure message, generate debug files for the grammar and string, and provide a command for further analysis.
    - Assert that each passing string matches the grammar, resetting the grammar stacks after each test.
    - Iterate over each string in `failing_strings`, logging the string and checking if it matches the grammar using [`match_string`](#match_string).
    - If a failing string matches, log an incorrect match message.
    - Assert that each failing string does not match the grammar, resetting the grammar stacks after each test.
    - Free the allocated memory for the grammar using `llama_grammar_free_impl`.
- **Output**: The function does not return a value; it logs results to standard output and standard error, and generates debug files for failed matches.
- **Functions called**:
    - [`build_grammar`](#build_grammar)
    - [`match_string`](#match_string)


---
### test\_grammar<!-- {{#callable:test_grammar}} -->
The `test_grammar` function tests a given grammar string against a set of passing and failing strings to verify its correctness.
- **Inputs**:
    - `test_desc`: A description of the test being performed.
    - `grammar_str`: The grammar string to be tested.
    - `passing_strings`: A vector of strings that should be successfully matched by the grammar.
    - `failing_strings`: A vector of strings that should not be matched by the grammar.
- **Control Flow**:
    - The function constructs a test description by appending the grammar string to the provided test description.
    - It calls the [`test`](#test) function with the constructed test description, the grammar string, and the vectors of passing and failing strings.
- **Output**: The function does not return a value; it performs tests and outputs results to standard error and standard output.
- **Functions called**:
    - [`test`](#test)


---
### test\_schema<!-- {{#callable:test_schema}} -->
The `test_schema` function tests a JSON schema by converting it to a grammar and validating a set of passing and failing strings against it.
- **Inputs**:
    - `test_desc`: A description of the test being performed, used for logging and identification purposes.
    - `schema_str`: A string representation of the JSON schema to be tested.
    - `passing_strings`: A vector of strings that are expected to pass validation against the schema.
    - `failing_strings`: A vector of strings that are expected to fail validation against the schema.
- **Control Flow**:
    - The function constructs a test description by appending the schema string to the provided test description.
    - It parses the JSON schema string into a JSON object using `json::parse`.
    - The parsed JSON schema is converted into a grammar using [`json_schema_to_grammar`](../common/json-schema-to-grammar.cpp.driver.md#json_schema_to_grammar).
    - The [`test`](#test) function is called with the constructed test description, the generated grammar, and the vectors of passing and failing strings.
- **Output**: The function does not return a value; it performs validation and asserts the correctness of the schema against the provided strings.
- **Functions called**:
    - [`test`](#test)
    - [`json_schema_to_grammar`](../common/json-schema-to-grammar.cpp.driver.md#json_schema_to_grammar)


---
### test\_simple\_grammar<!-- {{#callable:test_simple_grammar}} -->
The `test_simple_grammar` function tests various JSON schemas and a simple grammar against a set of passing and failing string inputs to validate their correctness.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`test_schema`](#test_schema) multiple times with different JSON schema strings, each specifying integer constraints like minimum, maximum, and exclusive bounds, along with sets of passing and failing string inputs.
    - Each call to [`test_schema`](#test_schema) converts the JSON schema into a grammar and tests the provided strings to ensure they match or fail as expected based on the schema constraints.
    - The function also calls [`test_grammar`](#test_grammar) with a simple arithmetic grammar and tests it against strings that should pass or fail according to the grammar rules.
- **Output**: The function does not return any value; it performs assertions to ensure that all tests pass, and any failure will trigger an assertion error.
- **Functions called**:
    - [`test_schema`](#test_schema)
    - [`test_grammar`](#test_grammar)


---
### test\_complex\_grammar<!-- {{#callable:test_complex_grammar}} -->
The `test_complex_grammar` function tests a medium complexity grammar by verifying that a set of strings either match or fail to match the defined grammar rules.
- **Inputs**:
    - `None`: This function does not take any input parameters.
- **Control Flow**:
    - The function calls [`test_grammar`](#test_grammar) with a description, a grammar string, a list of passing strings, and a list of failing strings.
    - The [`test_grammar`](#test_grammar) function builds the grammar from the provided string and tests each string in the passing list to ensure they match the grammar.
    - For each passing string, it asserts that the string matches the grammar and resets the grammar stacks.
    - It then tests each string in the failing list to ensure they do not match the grammar, asserting that they do not match and resetting the grammar stacks.
    - The function concludes by cleaning up the allocated memory for the grammar.
- **Output**: The function does not return any value; it performs assertions to validate the grammar against the test strings.
- **Functions called**:
    - [`test_grammar`](#test_grammar)


---
### test\_special\_chars<!-- {{#callable:test_special_chars}} -->
The `test_special_chars` function tests a grammar rule that matches strings containing the substring 'abc' and verifies the handling of special and multi-byte characters.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`test_grammar`](#test_grammar) with a description 'special characters', a grammar rule that includes 'abc', a list of passing strings, and a list of failing strings.
    - The grammar rule is defined to match any string that contains the substring 'abc'.
    - The passing strings include variations of 'abc' and a string with multi-byte characters surrounding 'abc'.
    - The failing strings include variations that do not contain 'abc' or have 'abc' in incorrect positions.
- **Output**: The function does not return any value; it asserts the correctness of the grammar matching for the given test cases.
- **Functions called**:
    - [`test_grammar`](#test_grammar)


---
### test\_quantifiers<!-- {{#callable:test_quantifiers}} -->
The `test_quantifiers` function performs a series of grammar tests to validate the behavior of quantifiers (*, +, ?) and repetition patterns in grammar rules.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`test_grammar`](#test_grammar) multiple times, each with a different grammar rule and corresponding passing and failing strings.
    - For each grammar rule, it specifies a description, the grammar itself, a list of strings that should pass the grammar, and a list of strings that should fail.
    - The [`test_grammar`](#test_grammar) function is used to validate each set of strings against the specified grammar, ensuring that passing strings match the grammar and failing strings do not.
- **Output**: The function does not return any value; it performs assertions to ensure the grammar tests pass or fail as expected.
- **Functions called**:
    - [`test_grammar`](#test_grammar)


---
### test\_failure\_missing\_root<!-- {{#callable:test_failure_missing_root}} -->
The function `test_failure_missing_root` tests a grammar parsing scenario where the root node is missing, ensuring that the grammar is parsed correctly but does not contain a root node.
- **Inputs**: None
- **Control Flow**:
    - Prints a message indicating the start of the test for a missing root node.
    - Defines a grammar string that lacks a root rule, using 'rot' instead of 'root'.
    - Initializes a `llama_grammar_parser` object and parses the grammar string.
    - Asserts that the parsed grammar contains rules, indicating successful parsing.
    - Asserts that the 'root' symbol is not found in the parsed grammar's symbol IDs, confirming the absence of a root node.
    - Prints a success message if all assertions pass.
- **Output**: The function does not return any value; it performs assertions to validate the test case and prints messages to standard error.


---
### test\_failure\_missing\_reference<!-- {{#callable:test_failure_missing_reference}} -->
The function `test_failure_missing_reference` tests the parsing of a grammar string that is missing a referenced rule and asserts that the parsing fails as expected.
- **Inputs**: None
- **Control Flow**:
    - Prints a message indicating the start of the test for a missing reference node.
    - Defines a grammar string with a missing rule reference ('numero' instead of 'number').
    - Prints an expected error message.
    - Parses the grammar string using `llama_grammar_parser`.
    - Asserts that the parsed grammar's rules are empty, indicating a parsing failure.
    - Prints a message indicating the end of the expected error and that the test passed.
- **Output**: The function does not return any value; it asserts the expected failure of parsing a grammar with a missing reference.


---
### test\_failure\_left\_recursion<!-- {{#callable:test_failure_left_recursion}} -->
The `test_failure_left_recursion` function tests the detection of left recursion in grammar definitions by asserting that certain grammars fail to build due to left recursion issues.
- **Inputs**: None
- **Control Flow**:
    - Prints a message indicating the start of left recursion detection tests.
    - Defines a simple grammar string with direct left recursion and asserts that building this grammar fails.
    - Defines a medium complexity grammar string with indirect left recursion and asserts that building this grammar fails.
    - Defines a more complex grammar string with indirect left recursion involving multiple rules and asserts that building this grammar fails.
    - Defines the most complex grammar string with indirect left recursion and an empty rule, asserting that building this grammar fails.
    - Prints a message indicating that all tests passed.
- **Output**: The function does not return any value; it uses assertions to ensure that the grammar fails to build due to left recursion.
- **Functions called**:
    - [`test_build_grammar_fails`](#test_build_grammar_fails)


---
### test\_json\_schema<!-- {{#callable:test_json_schema}} -->
The `test_json_schema` function tests various JSON schemas by converting them to grammars and validating strings against these grammars to ensure correct parsing behavior.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`test_schema`](#test_schema) multiple times, each with a different JSON schema, a set of passing strings, and a set of failing strings.
    - Each call to [`test_schema`](#test_schema) involves converting the JSON schema to a grammar and then testing the provided strings against this grammar.
    - For each schema, the function checks if the passing strings are correctly matched by the grammar and if the failing strings are correctly rejected.
    - Assertions are used to ensure that the expected behavior (matching or not matching) is achieved for each string.
    - The function covers a wide range of JSON schema features, including types, formats, patterns, and properties.
- **Output**: The function does not return any value; it performs assertions to validate the behavior of the JSON schema parsing.
- **Functions called**:
    - [`test_schema`](#test_schema)


---
### main<!-- {{#callable:main}} -->
The `main` function executes a series of grammar integration tests and outputs the results to the console.
- **Inputs**: None
- **Control Flow**:
    - Prints a message indicating the start of grammar integration tests.
    - Calls [`test_simple_grammar`](#test_simple_grammar) to test simple grammar rules.
    - Calls [`test_complex_grammar`](#test_complex_grammar) to test more complex grammar rules.
    - Calls [`test_special_chars`](#test_special_chars) to test grammars with special characters.
    - Calls [`test_quantifiers`](#test_quantifiers) to test grammars with quantifiers like '*', '+', and '?'.
    - Calls [`test_failure_missing_root`](#test_failure_missing_root) to test grammars missing a root node.
    - Calls [`test_failure_missing_reference`](#test_failure_missing_reference) to test grammars with missing references.
    - Calls [`test_failure_left_recursion`](#test_failure_left_recursion) to test grammars for left recursion issues.
    - Calls [`test_json_schema`](#test_json_schema) to test JSON schema conversions to grammar.
    - Prints a message indicating all tests have passed.
    - Returns 0 to indicate successful execution.
- **Output**: The function returns an integer value of 0, indicating successful execution.
- **Functions called**:
    - [`test_simple_grammar`](#test_simple_grammar)
    - [`test_complex_grammar`](#test_complex_grammar)
    - [`test_special_chars`](#test_special_chars)
    - [`test_quantifiers`](#test_quantifiers)
    - [`test_failure_missing_root`](#test_failure_missing_root)
    - [`test_failure_missing_reference`](#test_failure_missing_reference)
    - [`test_failure_left_recursion`](#test_failure_left_recursion)
    - [`test_json_schema`](#test_json_schema)


