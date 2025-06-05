# Purpose
This C++ source code file is an executable program designed to validate an input string against a specified grammar defined in a GBNF (Grammar Backus-Naur Form) file. The program reads two files: a grammar file and an input file, both specified as command-line arguments. It utilizes functions from the included headers, `unicode.h` and `llama-grammar.h`, to process and validate the input string. The core functionality is encapsulated in the [`llama_grammar_validate`](#llama_grammar_validate) function, which checks if the input string conforms to the rules defined in the grammar. If the input string is invalid, the program provides detailed error messages, including the position of the error and a visual indication of the problematic character.

The program is structured around a [`main`](#main) function that orchestrates the reading of files, initialization of the grammar, validation of the input, and cleanup of resources. It uses standard C++ libraries for file handling and string manipulation, and it outputs results to the standard output. The code is a standalone executable, not intended to be a library or header file for inclusion in other projects. It provides a specific utility for grammar validation, making it a narrow-purpose tool focused on parsing and validating text against a defined grammar.
# Imports and Dependencies

---
- `../src/unicode.h`
- `../src/llama-grammar.h`
- `cstdio`
- `cstdlib`
- `sstream`
- `fstream`
- `string`
- `vector`


# Functions

---
### llama\_grammar\_validate<!-- {{#callable:llama_grammar_validate}} -->
The `llama_grammar_validate` function checks if a given input string conforms to a specified grammar and identifies any errors in the input.
- **Inputs**:
    - `grammar`: A pointer to a `llama_grammar` structure that defines the grammar rules against which the input string is validated.
    - `input_str`: A constant reference to a `std::string` containing the input string to be validated.
    - `error_pos`: A reference to a `size_t` variable where the position of the first error in the input string will be stored if validation fails.
    - `error_msg`: A reference to a `std::string` where an error message will be stored if validation fails.
- **Control Flow**:
    - Convert the input string `input_str` into a sequence of Unicode code points using `unicode_cpts_from_utf8`.
    - Retrieve the current grammar stacks using `llama_grammar_get_stacks`.
    - Iterate over each Unicode code point in the sequence.
    - For each code point, call `llama_grammar_accept` to process it according to the grammar rules.
    - If the current grammar stacks are empty after processing a code point, set `error_pos` and `error_msg` to indicate an unexpected character error and return `false`.
    - Increment the position counter `pos` for each code point processed.
    - After processing all code points, check if any stack in `stacks_cur` is empty, indicating successful validation, and return `true`.
    - If no stack is empty, set `error_pos` and `error_msg` to indicate an unexpected end of input and return `false`.
- **Output**: Returns a boolean value: `true` if the input string is valid according to the grammar, or `false` if an error is found.


---
### print\_error\_message<!-- {{#callable:print_error_message}} -->
The `print_error_message` function outputs a formatted error message indicating the position and nature of an error in a given input string.
- **Inputs**:
    - `input_str`: A constant reference to a string representing the input text that was validated.
    - `error_pos`: A size_t value indicating the position in the input string where the error occurred.
    - `error_msg`: A constant reference to a string containing the error message describing the nature of the error.
- **Control Flow**:
    - Prints a message indicating the input string is invalid according to the grammar.
    - Prints the error message and the position of the error in the input string.
    - Prints the input string up to the error position.
    - If the error position is within the bounds of the input string, it highlights the erroneous character in red.
    - If there are characters following the error position, it prints them in a different shade of red.
    - Resets the text color to default after printing the error context.
- **Output**: This function does not return a value; it outputs the error message directly to the standard output (stdout).


---
### main<!-- {{#callable:main}} -->
The `main` function reads a grammar and an input file, validates the input against the grammar, and outputs whether the input is valid or not.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments, where argv[0] is the program name, argv[1] is the grammar filename, and argv[2] is the input filename.
- **Control Flow**:
    - Check if the number of arguments (argc) is not equal to 3; if so, print usage information and return 1.
    - Extract the grammar filename and input filename from argv.
    - Attempt to open the grammar file; if it fails, print an error message and return 1.
    - Read the entire contents of the grammar file into a string.
    - Initialize a llama_grammar object using the grammar string; if initialization fails, print an error message and return 1.
    - Read the entire contents of the input file into a string.
    - Validate the input string against the grammar using llama_grammar_validate; capture the result and any error position/message.
    - If the input is valid, print a success message; otherwise, print an error message with details about the invalid input.
    - Free the llama_grammar object to clean up resources.
    - Return 0 to indicate successful execution.
- **Output**: The function returns an integer, 0 for successful execution and 1 for failure due to incorrect arguments, file opening errors, or grammar initialization failure.
- **Functions called**:
    - [`llama_grammar_validate`](#llama_grammar_validate)
    - [`print_error_message`](#print_error_message)


