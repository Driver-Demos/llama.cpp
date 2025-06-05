# Purpose
The provided C++ source code file is a comprehensive implementation of a grammar parser and processor, specifically designed to handle UTF-8 encoded text. It is part of a larger system, likely related to natural language processing or text parsing, as indicated by the inclusion of files like "llama-grammar.h", "llama-impl.h", "llama-vocab.h", and "llama-sampling.h". The file defines a series of functions and classes that facilitate the parsing of grammar rules, decoding of UTF-8 sequences, and the application of these rules to input text. The core functionality revolves around the `llama_grammar_parser` class, which is responsible for parsing grammar definitions, managing symbol IDs, and adding rules to a collection. Additionally, the file includes utility functions for handling UTF-8 decoding, character parsing, and grammar rule validation.

The code is structured to provide a robust framework for grammar parsing, with functions to handle various parsing tasks such as parsing alternates, sequences, and specific character types. It also includes mechanisms for detecting and handling left recursion in grammar rules, which is crucial for ensuring the correctness of the parsing process. The file defines public APIs for initializing, applying, and managing grammar rules, making it a critical component of a larger system that requires precise text parsing capabilities. The presence of functions for printing grammar rules and handling errors further indicates its role in debugging and validating grammar definitions. Overall, this file is a specialized library intended to be integrated into a larger application, providing essential grammar parsing and processing functionalities.
# Imports and Dependencies

---
- `llama-grammar.h`
- `llama-impl.h`
- `llama-vocab.h`
- `llama-sampling.h`
- `cmath`
- `algorithm`
- `stdexcept`


# Data Structures

---
### llama\_grammar\_parser<!-- {{#data_structure:llama_grammar_parser}} -->
- **Description**: [See definition](llama-grammar.h.driver.md#llama_grammar_parser)
- **Member Functions**:
    - [`llama_grammar_parser::get_symbol_id`](#llama_grammar_parserget_symbol_id)
    - [`llama_grammar_parser::generate_symbol_id`](#llama_grammar_parsergenerate_symbol_id)
    - [`llama_grammar_parser::add_rule`](#llama_grammar_parseradd_rule)
    - [`llama_grammar_parser::parse_alternates`](#llama_grammar_parserparse_alternates)
    - [`llama_grammar_parser::parse_sequence`](#llama_grammar_parserparse_sequence)
    - [`llama_grammar_parser::parse_rule`](#llama_grammar_parserparse_rule)
    - [`llama_grammar_parser::parse`](#llama_grammar_parserparse)
    - [`llama_grammar_parser::print`](#llama_grammar_parserprint)
    - [`llama_grammar_parser::c_rules`](#llama_grammar_parserc_rules)

**Methods**

---
#### llama\_grammar\_parser::get\_symbol\_id<!-- {{#callable:llama_grammar_parser::get_symbol_id}} -->
The `get_symbol_id` function retrieves or assigns a unique identifier for a given symbol string in the `llama_grammar_parser` class.
- **Inputs**:
    - `src`: A pointer to a character array representing the symbol string to be processed.
    - `len`: The length of the symbol string pointed to by `src`.
- **Control Flow**:
    - Calculate the next available unique identifier by determining the current size of the `symbol_ids` map.
    - Attempt to insert the symbol string (constructed from `src` and `len`) into the `symbol_ids` map with the calculated identifier.
    - If the symbol string already exists in the map, the existing identifier is used; otherwise, the new identifier is assigned.
    - Return the identifier associated with the symbol string.
- **Output**: Returns a `uint32_t` representing the unique identifier for the given symbol string.
- **See also**: [`llama_grammar_parser`](llama-grammar.h.driver.md#llama_grammar_parser)  (Data Structure)


---
#### llama\_grammar\_parser::generate\_symbol\_id<!-- {{#callable:llama_grammar_parser::generate_symbol_id}} -->
The `generate_symbol_id` function generates a unique symbol ID for a given base name by appending an underscore and a numeric ID to the base name, and then stores this mapping in a map.
- **Inputs**:
    - `base_name`: A string representing the base name for which a unique symbol ID is to be generated.
- **Control Flow**:
    - Calculate the next available ID by getting the size of the `symbol_ids` map and casting it to `uint32_t`.
    - Concatenate the `base_name` with an underscore and the `next_id` to form a unique key.
    - Insert this key-value pair into the `symbol_ids` map, where the key is the concatenated string and the value is `next_id`.
    - Return the `next_id` as the generated symbol ID.
- **Output**: Returns a `uint32_t` representing the newly generated unique symbol ID.
- **See also**: [`llama_grammar_parser`](llama-grammar.h.driver.md#llama_grammar_parser)  (Data Structure)


---
#### llama\_grammar\_parser::add\_rule<!-- {{#callable:llama_grammar_parser::add_rule}} -->
The `add_rule` function adds or updates a grammar rule in the `rules` vector at a specified index, resizing the vector if necessary.
- **Inputs**:
    - `rule_id`: An unsigned 32-bit integer representing the index at which the rule should be added or updated in the `rules` vector.
    - `rule`: A constant reference to a `llama_grammar_rule` object that represents the grammar rule to be added or updated.
- **Control Flow**:
    - Check if the current size of the `rules` vector is less than or equal to `rule_id`.
    - If true, resize the `rules` vector to accommodate the new rule by increasing its size to `rule_id + 1`.
    - Assign the `rule` to the `rules` vector at the index specified by `rule_id`.
- **Output**: This function does not return any value.
- **See also**: [`llama_grammar_parser`](llama-grammar.h.driver.md#llama_grammar_parser)  (Data Structure)


---
#### llama\_grammar\_parser::parse\_alternates<!-- {{#callable:llama_grammar_parser::parse_alternates}} -->
The `parse_alternates` function parses a sequence of grammar rules from a source string, handling alternation operators ('|'), and adds the parsed rule to the grammar rules list.
- **Inputs**:
    - `src`: A pointer to the source string from which the grammar rules are to be parsed.
    - `rule_name`: A string representing the name of the rule being parsed.
    - `rule_id`: A unique identifier for the rule being parsed.
    - `is_nested`: A boolean indicating whether the parsing is occurring within a nested context.
- **Control Flow**:
    - Initialize a `llama_grammar_rule` object to store the parsed rule.
    - Call [`parse_sequence`](#llama_grammar_parserparse_sequence) to parse the initial sequence from `src` and update the `rule` object.
    - Enter a loop that continues as long as the current position in the source string points to a '|' character, indicating an alternation.
    - Within the loop, add an alternation element (`LLAMA_GRETYPE_ALT`) to the rule, parse any spaces, and then parse the next sequence using [`parse_sequence`](#llama_grammar_parserparse_sequence).
    - After exiting the loop, append an end element (`LLAMA_GRETYPE_END`) to the rule to signify the end of the alternation.
    - Add the completed rule to the grammar rules list using [`add_rule`](#llama_grammar_parseradd_rule).
    - Return the current position in the source string.
- **Output**: Returns a pointer to the current position in the source string after parsing the alternates.
- **Functions called**:
    - [`llama_grammar_parser::parse_sequence`](#llama_grammar_parserparse_sequence)
    - [`parse_space`](#parse_space)
    - [`llama_grammar_parser::add_rule`](#llama_grammar_parseradd_rule)
- **See also**: [`llama_grammar_parser`](llama-grammar.h.driver.md#llama_grammar_parser)  (Data Structure)


---
#### llama\_grammar\_parser::parse\_sequence<!-- {{#callable:llama_grammar_parser::parse_sequence}} -->
The `parse_sequence` function parses a sequence of grammar elements from a source string and updates a grammar rule accordingly, handling various grammar constructs like literals, character ranges, rule references, groupings, and repetitions.
- **Inputs**:
    - `src`: A pointer to the source string from which the grammar sequence is to be parsed.
    - `rule_name`: A string representing the name of the rule being parsed.
    - `rule`: A reference to a `llama_grammar_rule` object where the parsed elements will be stored.
    - `is_nested`: A boolean indicating whether the parsing is occurring within a nested context.
- **Control Flow**:
    - Initialize `last_sym_start` to track the start of the last symbol in the rule and set `pos` to the start of the source string.
    - Define a lambda function `handle_repetitions` to handle repetition constructs like `*`, `+`, `?`, and `{m,n}` by transforming the previous symbols in the rule according to specified repetition rules.
    - Enter a loop to process each character in the source string `pos` until the end of the string is reached or an unrecognized character is encountered.
    - Handle different grammar constructs based on the current character: literals ("), character ranges ([ ]), rule references (word characters), groupings (( )), any character (.), and repetition operators (*, +, ?, { }).
    - For each recognized construct, update the `rule` with the appropriate grammar elements and adjust `pos` to point to the next character to be processed.
    - For repetition operators, call `handle_repetitions` with the appropriate minimum and maximum repetition counts.
    - Return the updated position `pos` after parsing the sequence.
- **Output**: Returns a pointer to the position in the source string immediately following the parsed sequence.
- **Functions called**:
    - [`llama_grammar_parser::generate_symbol_id`](#llama_grammar_parsergenerate_symbol_id)
    - [`llama_grammar_parser::add_rule`](#llama_grammar_parseradd_rule)
    - [`parse_char`](#parse_char)
    - [`parse_space`](#parse_space)
    - [`is_word_char`](#is_word_char)
    - [`parse_name`](#parse_name)
    - [`llama_grammar_parser::get_symbol_id`](#llama_grammar_parserget_symbol_id)
    - [`llama_grammar_parser::parse_alternates`](#llama_grammar_parserparse_alternates)
    - [`is_digit_char`](#is_digit_char)
    - [`parse_int`](#parse_int)
- **See also**: [`llama_grammar_parser`](llama-grammar.h.driver.md#llama_grammar_parser)  (Data Structure)


---
#### llama\_grammar\_parser::parse\_rule<!-- {{#callable:llama_grammar_parser::parse_rule}} -->
The `parse_rule` function parses a grammar rule from a given source string, ensuring it follows the expected format and structure, and returns the position in the string after the rule has been parsed.
- **Inputs**:
    - `src`: A pointer to a constant character array representing the source string containing the grammar rule to be parsed.
- **Control Flow**:
    - Call [`parse_name`](#parse_name) to determine the end of the rule name in the source string.
    - Call [`parse_space`](#parse_space) to skip any whitespace following the rule name.
    - Calculate the length of the rule name and obtain its symbol ID using [`get_symbol_id`](#llama_grammar_parserget_symbol_id).
    - Check for the presence of the '::=' delimiter; if not found, throw a runtime error.
    - Call [`parse_space`](#parse_space) to skip whitespace after the '::=' delimiter.
    - Call [`parse_alternates`](#llama_grammar_parserparse_alternates) to parse the rule's alternates, passing the current position, rule name, and rule ID.
    - Check for newline characters or end of input after parsing alternates; if not found, throw a runtime error.
    - Return the position in the source string after parsing the rule and any trailing whitespace.
- **Output**: A pointer to a constant character array indicating the position in the source string after the parsed rule and any trailing whitespace.
- **Functions called**:
    - [`parse_name`](#parse_name)
    - [`parse_space`](#parse_space)
    - [`llama_grammar_parser::get_symbol_id`](#llama_grammar_parserget_symbol_id)
    - [`llama_grammar_parser::parse_alternates`](#llama_grammar_parserparse_alternates)
- **See also**: [`llama_grammar_parser`](llama-grammar.h.driver.md#llama_grammar_parser)  (Data Structure)


---
#### llama\_grammar\_parser::parse<!-- {{#callable:llama_grammar_parser::parse}} -->
The `parse` function attempts to parse a grammar from a given source string and validates that all rules are defined, returning a boolean indicating success or failure.
- **Inputs**:
    - `src`: A C-style string representing the source grammar to be parsed.
- **Control Flow**:
    - The function begins by calling [`parse_space`](#parse_space) to skip any initial whitespace or comments in the source string.
    - It enters a loop to parse each rule in the source string using [`parse_rule`](#llama_grammar_parserparse_rule) until the end of the string is reached.
    - After parsing, it iterates over all rules to ensure they are defined and checks for any undefined rule references.
    - If any rule is undefined or a rule reference is invalid, it throws a runtime error with a descriptive message.
    - If an exception is caught, it logs the error, clears the rules, and returns false.
    - If no exceptions occur, it returns true, indicating successful parsing.
- **Output**: A boolean value indicating whether the parsing was successful (true) or if an error occurred (false).
- **Functions called**:
    - [`parse_space`](#parse_space)
    - [`llama_grammar_parser::parse_rule`](#llama_grammar_parserparse_rule)
- **See also**: [`llama_grammar_parser`](llama-grammar.h.driver.md#llama_grammar_parser)  (Data Structure)


---
#### llama\_grammar\_parser::print<!-- {{#callable:llama_grammar_parser::print}} -->
The `print` function in the `llama_grammar_parser` class outputs the grammar rules to a specified file, mapping symbol IDs to their names and printing each rule using a helper function.
- **Inputs**:
    - `file`: A pointer to a `FILE` object where the grammar rules will be printed.
- **Control Flow**:
    - A `try` block is initiated to handle any exceptions that may occur during the printing process.
    - A `std::map` named `symbol_id_names` is created to map symbol IDs to their corresponding names by iterating over the `symbol_ids` map.
    - The function iterates over the `rules` vector, and for each rule, it calls the [`print_rule`](#print_rule) function, passing the file pointer, the rule index, the rule itself, and the `symbol_id_names` map.
    - If an exception is caught, an error message is printed to `stderr` indicating an error occurred while printing the grammar.
- **Output**: The function does not return a value; it outputs the grammar rules to the specified file.
- **Functions called**:
    - [`print_rule`](#print_rule)
- **See also**: [`llama_grammar_parser`](llama-grammar.h.driver.md#llama_grammar_parser)  (Data Structure)


---
#### llama\_grammar\_parser::c\_rules<!-- {{#callable:llama_grammar_parser::c_rules}} -->
The `c_rules` function returns a stack of pointers to the data of each rule in the `rules` collection of the `llama_grammar_parser` class.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty `llama_grammar_stack` named `ret`.
    - Reserve space in `ret` for the number of elements in `rules`.
    - Iterate over each `rule` in `rules`.
    - For each `rule`, push a pointer to its data into `ret`.
    - Return the populated `ret` stack.
- **Output**: A `llama_grammar_stack` containing pointers to the data of each rule in the `rules` collection.
- **See also**: [`llama_grammar_parser`](llama-grammar.h.driver.md#llama_grammar_parser)  (Data Structure)



# Functions

---
### decode\_utf8<!-- {{#callable:decode_utf8}} -->
The `decode_utf8` function decodes a UTF-8 encoded string into a vector of Unicode code points, handling partial sequences and invalid sequences.
- **Inputs**:
    - `src`: A `std::string` representing the UTF-8 encoded input string to be decoded.
    - `partial_start`: A `llama_partial_utf8` structure containing the state of a partially decoded UTF-8 sequence, with fields `value` and `n_remain` indicating the current value and the number of remaining bytes to complete the sequence.
- **Control Flow**:
    - Initialize a lookup table for determining the number of bytes in a UTF-8 sequence based on the high bits of the first byte.
    - Set up a pointer `pos` to iterate over the input string `src` and initialize a vector `code_points` to store the decoded Unicode code points.
    - Reserve space in `code_points` based on the size of `src` plus one for a terminating zero.
    - Continue decoding from a previous partial sequence if `n_remain` is greater than zero, checking for valid continuation bytes and updating `value`.
    - If a partial sequence is completed, add the decoded value to `code_points`.
    - Iterate over the remaining bytes in `src`, determining the length of each UTF-8 sequence using the lookup table, and decode each sequence into a Unicode code point.
    - Handle invalid sequences by clearing `code_points` and returning a pair with a zero code point and an error state.
    - Add a terminating zero to `code_points` and return a pair containing the decoded code points and the state of any remaining partial sequence.
- **Output**: A `std::pair` consisting of a `std::vector<uint32_t>` containing the decoded Unicode code points and a `llama_partial_utf8` structure representing the state of any remaining partial UTF-8 sequence.


---
### is\_digit\_char<!-- {{#callable:is_digit_char}} -->
The `is_digit_char` function checks if a given character is a numeric digit (0-9).
- **Inputs**:
    - `c`: A character to be checked if it is a numeric digit.
- **Control Flow**:
    - The function uses a simple comparison to check if the character `c` is between '0' and '9' inclusive.
    - It returns the result of this comparison as a boolean value.
- **Output**: A boolean value indicating whether the character is a numeric digit.


---
### is\_word\_char<!-- {{#callable:is_word_char}} -->
The `is_word_char` function checks if a given character is a letter, a digit, or a hyphen.
- **Inputs**:
    - `c`: A character to be checked if it is a word character.
- **Control Flow**:
    - The function checks if the character `c` is a lowercase letter between 'a' and 'z'.
    - If not, it checks if `c` is an uppercase letter between 'A' and 'Z'.
    - If not, it checks if `c` is a hyphen ('-').
    - If none of the above, it calls `is_digit_char(c)` to check if `c` is a digit.
- **Output**: Returns `true` if the character is a letter, a digit, or a hyphen; otherwise, returns `false`.
- **Functions called**:
    - [`is_digit_char`](#is_digit_char)


---
### parse\_hex<!-- {{#callable:parse_hex}} -->
The `parse_hex` function parses a hexadecimal string of a specified size and returns the corresponding integer value along with the position in the string where parsing stopped.
- **Inputs**:
    - `src`: A pointer to the source string containing the hexadecimal characters to be parsed.
    - `size`: An integer specifying the number of hexadecimal characters expected to be parsed from the source string.
- **Control Flow**:
    - Initialize `pos` to point to the start of the source string and `end` to point to the end of the expected parsing range.
    - Initialize `value` to 0 to accumulate the parsed integer value.
    - Iterate over the characters in the source string from `pos` to `end`, shifting `value` left by 4 bits for each character to make room for the next nibble.
    - For each character, check if it is a valid hexadecimal digit ('0'-'9', 'a'-'f', 'A'-'F') and update `value` accordingly by adding the numeric value of the character.
    - If a non-hexadecimal character is encountered, break out of the loop.
    - If the loop ends before reaching `end`, throw a runtime error indicating the expected number of hexadecimal characters was not met.
    - Return a pair containing the parsed integer value and the position in the string where parsing stopped.
- **Output**: A `std::pair` containing the parsed integer value as `uint32_t` and a pointer to the position in the string where parsing stopped.


---
### parse\_space<!-- {{#callable:parse_space}} -->
The `parse_space` function advances a pointer through whitespace, tabs, comments, and optionally newlines in a given string.
- **Inputs**:
    - `src`: A pointer to the beginning of the string to be parsed.
    - `newline_ok`: A boolean flag indicating whether newlines should be considered as whitespace to skip.
- **Control Flow**:
    - Initialize a pointer `pos` to the start of the input string `src`.
    - Enter a loop that continues as long as `pos` points to a space, tab, comment character ('#'), or newline (if `newline_ok` is true).
    - If `pos` points to a comment character ('#'), enter a nested loop to skip all characters until a newline or the end of the string is reached.
    - Otherwise, increment `pos` to skip the current whitespace or newline character.
    - Return the updated pointer `pos` after all applicable characters have been skipped.
- **Output**: A pointer to the first non-whitespace character in the string, or the end of the string if only whitespace is present.


---
### parse\_name<!-- {{#callable:parse_name}} -->
The `parse_name` function parses a name from a given string, ensuring it consists of valid word characters and throws an error if no valid name is found.
- **Inputs**:
    - `src`: A pointer to a constant character array (C-style string) from which the function will attempt to parse a name.
- **Control Flow**:
    - Initialize a pointer `pos` to the start of the input string `src`.
    - Iterate through the string, advancing `pos` while the current character is a valid word character (letters, digits, or hyphen).
    - If `pos` has not advanced from `src`, throw a runtime error indicating that a name was expected at the given position.
    - Return the position `pos` where the parsing stopped.
- **Output**: A pointer to the position in the string immediately after the last valid word character of the parsed name.
- **Functions called**:
    - [`is_word_char`](#is_word_char)


---
### parse\_int<!-- {{#callable:parse_int}} -->
The `parse_int` function parses a string to find an integer and returns a pointer to the position after the integer, throwing an error if no integer is found.
- **Inputs**:
    - `src`: A pointer to a constant character array (string) from which the function will attempt to parse an integer.
- **Control Flow**:
    - Initialize a pointer `pos` to the start of the input string `src`.
    - Iterate through the string using `pos` as long as the current character is a digit, incrementing `pos` with each digit found.
    - If `pos` is equal to `src` after the loop, throw a `std::runtime_error` indicating that an integer was expected at the start of the string.
    - Return the pointer `pos`, which now points to the character immediately following the last digit of the parsed integer.
- **Output**: A pointer to the character in the input string immediately following the last digit of the parsed integer.
- **Functions called**:
    - [`is_digit_char`](#is_digit_char)


---
### parse\_char<!-- {{#callable:parse_char}} -->
The `parse_char` function parses a character from a given source string, handling escape sequences and UTF-8 encoding.
- **Inputs**:
    - `src`: A pointer to a constant character array (C-style string) from which the character is to be parsed.
- **Control Flow**:
    - Check if the first character in `src` is a backslash '\'.
    - If it is a backslash, use a switch statement to handle various escape sequences such as '\x', '\u', '\U', '\t', '\r', '\n', '\\', '"', '[', and ']'.
    - For '\x', '\u', and '\U', call [`parse_hex`](#parse_hex) with appropriate length to parse hexadecimal values.
    - For '\t', '\r', and '\n', return the corresponding character and advance the pointer by 2.
    - For '\\', '"', '[', and ']', return the character itself and advance the pointer by 2.
    - If the escape sequence is unknown, throw a runtime error with a message indicating the unknown escape.
    - If the first character is not a backslash and is not null, call [`decode_utf8`](#decode_utf8) to handle UTF-8 decoding.
    - If the first character is null, throw a runtime error indicating an unexpected end of input.
- **Output**: A `std::pair` containing a `uint32_t` representing the parsed character and a `const char*` pointing to the next character in the source string after the parsed character.
- **Functions called**:
    - [`parse_hex`](#parse_hex)
    - [`decode_utf8`](#decode_utf8)


---
### print\_grammar\_char<!-- {{#callable:print_grammar_char}} -->
The `print_grammar_char` function prints a character to a file, either as its ASCII representation if it's a printable ASCII character or as a Unicode code point if it's not.
- **Inputs**:
    - `file`: A pointer to a FILE object where the character will be printed.
    - `c`: A 32-bit unsigned integer representing the character to be printed.
- **Control Flow**:
    - Check if the character `c` is within the printable ASCII range (0x20 to 0x7f).
    - If `c` is within the printable ASCII range, print it as a character to the file using `fprintf`.
    - If `c` is not within the printable ASCII range, print it as a Unicode code point in the format `<U+XXXX>` using `fprintf`.
- **Output**: The function does not return a value; it outputs directly to the specified file.


---
### is\_char\_element<!-- {{#callable:is_char_element}} -->
The `is_char_element` function checks if a given `llama_grammar_element` is of a character-related type.
- **Inputs**:
    - `elem`: A `llama_grammar_element` object whose type is to be checked.
- **Control Flow**:
    - The function uses a switch statement to check the type of the `elem` argument.
    - If `elem.type` matches any of the character-related types (`LLAMA_GRETYPE_CHAR`, `LLAMA_GRETYPE_CHAR_NOT`, `LLAMA_GRETYPE_CHAR_ALT`, `LLAMA_GRETYPE_CHAR_RNG_UPPER`, `LLAMA_GRETYPE_CHAR_ANY`), the function returns `true`.
    - If `elem.type` does not match any of these types, the function returns `false`.
- **Output**: A boolean value indicating whether the `elem` is of a character-related type.


---
### print\_rule\_binary<!-- {{#callable:print_rule_binary}} -->
The `print_rule_binary` function outputs a binary representation of a grammar rule to a specified file.
- **Inputs**:
    - `file`: A pointer to a FILE object where the output will be written.
    - `rule`: A constant reference to a `llama_grammar_rule` object, which is a collection of grammar elements to be printed.
- **Control Flow**:
    - Iterates over each element in the `rule` collection.
    - For each element, it checks the type of the element using a switch statement.
    - Depending on the type, it prints a corresponding string (e.g., "END", "ALT", "RULE_REF", etc.) to the file.
    - For certain types (LLAMA_GRETYPE_END, LLAMA_GRETYPE_ALT, LLAMA_GRETYPE_RULE_REF), it prints the element's value as an unsigned integer.
    - For character-related types (LLAMA_GRETYPE_CHAR, LLAMA_GRETYPE_CHAR_NOT, etc.), it prints the character value using the [`print_grammar_char`](#print_grammar_char) function.
    - Ends the output with a newline character.
- **Output**: The function does not return a value; it writes formatted output to the specified file.
- **Functions called**:
    - [`print_grammar_char`](#print_grammar_char)


---
### print\_rule<!-- {{#callable:print_rule}} -->
The `print_rule` function formats and prints a grammar rule to a specified file, ensuring the rule is well-formed and ends with a specific type.
- **Inputs**:
    - `file`: A pointer to a FILE object where the formatted rule will be printed.
    - `rule_id`: An unsigned 32-bit integer representing the unique identifier of the grammar rule.
    - `rule`: A constant reference to a `llama_grammar_rule`, which is a collection of `llama_grammar_element` objects representing the grammar rule to be printed.
    - `symbol_id_names`: A constant reference to a map that associates rule IDs with their corresponding string names.
- **Control Flow**:
    - Check if the rule is empty or does not end with `LLAMA_GRETYPE_END`, throwing a runtime error if so.
    - Print the rule's name followed by '::=' to the file using the rule ID to look up its name in `symbol_id_names`.
    - Iterate over each element in the rule, except the last one, and handle each element type with a switch statement.
    - For `LLAMA_GRETYPE_ALT`, print '|'.
    - For `LLAMA_GRETYPE_RULE_REF`, print the referenced rule's name using `symbol_id_names`.
    - For `LLAMA_GRETYPE_CHAR`, `LLAMA_GRETYPE_CHAR_NOT`, `LLAMA_GRETYPE_CHAR_RNG_UPPER`, and `LLAMA_GRETYPE_CHAR_ALT`, print the character or character range, handling errors if the range is malformed.
    - For `LLAMA_GRETYPE_CHAR_ANY`, print '.'.
    - Close character ranges with ']' if the current element is a character element and the next element is not a continuation of the range.
    - Print a newline character at the end.
- **Output**: The function does not return a value; it outputs the formatted rule directly to the specified file.
- **Functions called**:
    - [`print_grammar_char`](#print_grammar_char)
    - [`is_char_element`](#is_char_element)


---
### llama\_grammar\_is\_end\_of\_sequence<!-- {{#callable:llama_grammar_is_end_of_sequence}} -->
The function `llama_grammar_is_end_of_sequence` checks if a given grammar element marks the end of a sequence in a grammar rule.
- **Inputs**:
    - `pos`: A pointer to a `llama_grammar_element` structure, representing a position in a grammar rule.
- **Control Flow**:
    - The function uses a switch statement to check the type of the grammar element pointed to by `pos`.
    - If the type is `LLAMA_GRETYPE_END` or `LLAMA_GRETYPE_ALT`, the function returns `true`.
    - For any other type, the function returns `false`.
- **Output**: A boolean value indicating whether the given grammar element marks the end of a sequence (true) or not (false).


---
### llama\_grammar\_match\_char<!-- {{#callable:llama_grammar_match_char}} -->
The function `llama_grammar_match_char` checks if a given character matches a specified grammar element, which can be a single character, a range, or any character, and returns a pair indicating the match result and the next position in the grammar.
- **Inputs**:
    - `pos`: A pointer to the current position in the grammar element array, representing the grammar rule to be matched against.
    - `chr`: A 32-bit unsigned integer representing the character to be matched against the grammar rule.
- **Control Flow**:
    - Initialize `found` to false and determine if the current grammar element is a positive character match type.
    - Assert that the grammar element is either a positive or negative character type.
    - Enter a loop to evaluate the grammar element at `pos` against `chr`.
    - If the next element is a range upper bound, check if `chr` falls within the range and update `found`.
    - If the current element is a wildcard (any character), set `found` to true.
    - Otherwise, check for an exact character match and update `found`.
    - Advance `pos` to the next element in the grammar.
    - Continue the loop if the current element is an alternate character type.
    - Return a pair indicating if the match result aligns with the positive character type and the updated position.
- **Output**: A `std::pair` consisting of a boolean indicating if the character matches the grammar rule and a pointer to the next position in the grammar element array.


---
### llama\_grammar\_match\_partial\_char<!-- {{#callable:llama_grammar_match_partial_char}} -->
The function `llama_grammar_match_partial_char` checks if a given partial UTF-8 sequence can potentially match a character range defined by a grammar element.
- **Inputs**:
    - `pos`: A pointer to a `llama_grammar_element` which defines the character range or type to match against.
    - `partial_utf8`: A `llama_partial_utf8` structure containing the current value of the partial UTF-8 sequence and the number of remaining bytes needed to complete it.
- **Control Flow**:
    - Check if the grammar element at `pos` is a positive character type (either `LLAMA_GRETYPE_CHAR` or `LLAMA_GRETYPE_CHAR_ANY`) or assert that it is `LLAMA_GRETYPE_CHAR_NOT`.
    - Extract the `value` and `n_remain` from the `partial_utf8` structure.
    - Return `false` if the sequence is invalid or if a 7-bit character is split across two bytes (overlong).
    - Calculate the range of possible code points (`low` and `high`) that the partial UTF-8 sequence could complete to.
    - Adjust `low` for overlong sequences if necessary.
    - Iterate over the grammar elements starting at `pos` to check if the range of possible code points can match the character range or type defined by the grammar element.
    - Return `true` if a match is found according to the character type, otherwise continue checking the next elements.
    - Return `false` if no match is found and the character type is positive, or `true` if the character type is negative.
- **Output**: A boolean value indicating whether the partial UTF-8 sequence could potentially match the character range defined by the grammar element.


---
### llama\_grammar\_advance\_stack<!-- {{#callable:llama_grammar_advance_stack}} -->
The function `llama_grammar_advance_stack` transforms a grammar pushdown stack into multiple possible stacks, all ending at a character range (terminal element).
- **Inputs**:
    - `rules`: A reference to `llama_grammar_rules`, which is a collection of grammar rules.
    - `stack`: A reference to `llama_grammar_stack`, representing the current state of the grammar stack.
    - `new_stacks`: A reference to `llama_grammar_stacks`, where new stacks will be stored after processing.
- **Control Flow**:
    - Check if the input stack is empty; if so, add it to `new_stacks` if it's not already present, and return.
    - Retrieve the last element from the stack to determine the type of grammar element.
    - If the element is a rule reference (`LLAMA_GRETYPE_RULE_REF`), iterate over its alternates, creating new stacks by removing the top element and adding subsequent elements or alternates.
    - Recursively call `llama_grammar_advance_stack` for each new stack created from rule references.
    - If the element is a character type (`LLAMA_GRETYPE_CHAR`, `LLAMA_GRETYPE_CHAR_NOT`, `LLAMA_GRETYPE_CHAR_ANY`), add the stack to `new_stacks` if it's not a duplicate.
    - Handle unexpected element types by aborting the program with an error message.
- **Output**: The function does not return a value; it modifies `new_stacks` by adding new stacks derived from the input `stack`.
- **Functions called**:
    - [`llama_grammar_is_end_of_sequence`](#llama_grammar_is_end_of_sequence)


---
### llama\_grammar\_reject\_candidates<!-- {{#callable:llama_grammar_reject_candidates}} -->
The `llama_grammar_reject_candidates` function filters out candidates that do not match the grammar rules across multiple stacks.
- **Inputs**:
    - `rules`: A reference to `llama_grammar_rules`, which contains the grammar rules to be applied.
    - `stacks`: A reference to `llama_grammar_stacks`, which represents the current state of the grammar parsing stacks.
    - `candidates`: A reference to `llama_grammar_candidates`, which contains the candidates to be evaluated against the grammar rules.
- **Control Flow**:
    - Assert that the `stacks` is not empty using `GGML_ASSERT`.
    - Check if `candidates` is empty; if so, return an empty `llama_grammar_candidates` object.
    - Call [`llama_grammar_reject_candidates_for_stack`](#llama_grammar_reject_candidates_for_stack) with the first stack to get initial rejects.
    - Iterate over the remaining stacks, updating the rejects by calling [`llama_grammar_reject_candidates_for_stack`](#llama_grammar_reject_candidates_for_stack) for each stack.
    - Return the final list of rejects.
- **Output**: Returns a `llama_grammar_candidates` object containing the candidates that do not match the grammar rules.
- **Functions called**:
    - [`llama_grammar_reject_candidates_for_stack`](#llama_grammar_reject_candidates_for_stack)


---
### llama\_grammar\_detect\_left\_recursion<!-- {{#callable:llama_grammar_detect_left_recursion}} -->
The function `llama_grammar_detect_left_recursion` checks for left recursion in a grammar rule set starting from a specified rule index.
- **Inputs**:
    - `rules`: A reference to a collection of grammar rules (`llama_grammar_rules`) to be checked for left recursion.
    - `rule_index`: The index of the rule in the `rules` collection from which to start checking for left recursion.
    - `rules_visited`: A pointer to a vector of booleans indicating which rules have already been visited.
    - `rules_in_progress`: A pointer to a vector of booleans indicating which rules are currently being processed.
    - `rules_may_be_empty`: A pointer to a vector of booleans indicating which rules may produce an empty string.
- **Control Flow**:
    - Check if the rule at `rule_index` is already in progress; if so, return true indicating left recursion.
    - Mark the rule at `rule_index` as in progress.
    - Retrieve the rule at `rule_index` from the `rules` collection.
    - Iterate over the elements of the rule to determine if it might produce an empty string, updating `rules_may_be_empty` if so.
    - Iterate over the elements of the rule to recursively check for left recursion in leftmost nonterminals, considering whether previous nonterminals may be empty.
    - If left recursion is detected in any recursive call, return true.
    - Mark the rule at `rule_index` as no longer in progress and as visited.
    - Return false if no left recursion is detected.
- **Output**: Returns a boolean value: true if left recursion is detected starting from the specified rule index, otherwise false.
- **Functions called**:
    - [`llama_grammar_is_end_of_sequence`](#llama_grammar_is_end_of_sequence)


---
### llama\_grammar\_get\_rules<!-- {{#callable:llama_grammar_get_rules}} -->
The function `llama_grammar_get_rules` retrieves the grammar rules from a given `llama_grammar` structure.
- **Inputs**:
    - `grammar`: A pointer to a `llama_grammar` structure from which the grammar rules are to be retrieved.
- **Control Flow**:
    - The function accesses the `rules` member of the `llama_grammar` structure pointed to by `grammar`.
    - It returns the `rules` member directly.
- **Output**: A reference to the `llama_grammar_rules` contained within the provided `llama_grammar` structure.


---
### llama\_grammar\_get\_stacks<!-- {{#callable:llama_grammar_get_stacks}} -->
The function `llama_grammar_get_stacks` returns a reference to the `stacks` member of a `llama_grammar` structure.
- **Inputs**:
    - `grammar`: A pointer to a `llama_grammar` structure from which the `stacks` member will be accessed.
- **Control Flow**:
    - The function takes a pointer to a `llama_grammar` structure as input.
    - It directly accesses the `stacks` member of the `llama_grammar` structure.
    - The function returns a reference to the `stacks` member.
- **Output**: A reference to the `stacks` member of the `llama_grammar` structure.


---
### llama\_grammar\_accept<!-- {{#callable:llama_grammar_accept}} -->
The `llama_grammar_accept` function processes a character input against a grammar's current state, updating the grammar's stacks based on character matches and grammar rules.
- **Inputs**:
    - `grammar`: A pointer to a `llama_grammar` structure, which contains the current state of the grammar including rules and stacks.
    - `chr`: A `uint32_t` representing the character to be processed against the grammar.
- **Control Flow**:
    - Initialize a new stack container `stacks_new` with reserved space equal to the current number of stacks in the grammar.
    - Iterate over each stack in the grammar's current stacks.
    - Skip processing for any empty stack.
    - For each non-empty stack, check if the character `chr` matches the top element of the stack using [`llama_grammar_match_char`](#llama_grammar_match_char).
    - If a match is found, create a new stack by copying the current stack minus its top element.
    - If the matched position is not the end of a sequence, add the matched position to the new stack.
    - Advance the new stack using [`llama_grammar_advance_stack`](#llama_grammar_advance_stack), updating `stacks_new` with possible new stack configurations.
    - Replace the grammar's current stacks with `stacks_new`.
- **Output**: The function does not return a value; it modifies the `grammar`'s stacks in place.
- **Functions called**:
    - [`llama_grammar_match_char`](#llama_grammar_match_char)
    - [`llama_grammar_is_end_of_sequence`](#llama_grammar_is_end_of_sequence)
    - [`llama_grammar_advance_stack`](#llama_grammar_advance_stack)


---
### llama\_grammar\_reject\_candidates\_for\_stack<!-- {{#callable:llama_grammar_reject_candidates_for_stack}} -->
The function `llama_grammar_reject_candidates_for_stack` filters out candidates that do not match the current grammar stack position and returns the rejected candidates.
- **Inputs**:
    - `rules`: A constant reference to `llama_grammar_rules`, which defines the grammar rules to be used for matching.
    - `stack`: A constant reference to `llama_grammar_stack`, representing the current position in the grammar stack to be matched against.
    - `candidates`: A constant reference to `llama_grammar_candidates`, which is a list of candidate tokens to be evaluated against the grammar stack.
- **Control Flow**:
    - Initialize an empty `rejects` list and reserve space equal to the size of `candidates`.
    - Check if the `stack` is empty; if so, iterate over `candidates` and add those with non-zero code points or remaining partial UTF-8 sequences to `rejects`.
    - If the `stack` is not empty, set `stack_pos` to the last element of the stack and initialize `next_candidates`.
    - Iterate over `candidates`, checking if the token's code points are zero and if it ends in a partial sequence that cannot satisfy the grammar position, adding such tokens to `rejects`.
    - For non-zero code points, check if the token matches the current grammar position using [`llama_grammar_match_char`](#llama_grammar_match_char); if it matches, add it to `next_candidates` with incremented code points, otherwise add it to `rejects`.
    - Determine the next position in the stack using [`llama_grammar_match_char`](#llama_grammar_match_char) with a zero character.
    - Update the stack to `stack_after` by removing the last element and adding the next position if it is not the end of a sequence.
    - Advance the stack using [`llama_grammar_advance_stack`](#llama_grammar_advance_stack) to get `next_stacks`.
    - Call [`llama_grammar_reject_candidates`](#llama_grammar_reject_candidates) recursively with `next_stacks` and `next_candidates` to get `next_rejects`.
    - Add `next_rejects` to `rejects`, adjusting the code points by decrementing them by one.
- **Output**: Returns a `llama_grammar_candidates` object containing the list of rejected candidates.
- **Functions called**:
    - [`llama_grammar_match_partial_char`](#llama_grammar_match_partial_char)
    - [`llama_grammar_match_char`](#llama_grammar_match_char)
    - [`llama_grammar_is_end_of_sequence`](#llama_grammar_is_end_of_sequence)
    - [`llama_grammar_advance_stack`](#llama_grammar_advance_stack)
    - [`llama_grammar_reject_candidates`](#llama_grammar_reject_candidates)


---
### llama\_grammar\_init\_impl<!-- {{#callable:llama_grammar_init_impl}} -->
The `llama_grammar_init_impl` function initializes a `llama_grammar` structure by parsing a given grammar string, checking for errors, and setting up grammar rules and trigger patterns.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure, representing the vocabulary used in the grammar.
    - `grammar_str`: A C-style string containing the grammar definition to be parsed.
    - `grammar_root`: A C-style string specifying the root symbol of the grammar.
    - `lazy`: A boolean indicating whether the grammar should be initialized in lazy mode, affecting trigger handling.
    - `trigger_patterns`: An array of C-style strings representing regex patterns that can trigger grammar processing.
    - `num_trigger_patterns`: The number of trigger patterns provided in the `trigger_patterns` array.
    - `trigger_tokens`: An array of `llama_token` representing specific tokens that can trigger grammar processing.
    - `num_trigger_tokens`: The number of trigger tokens provided in the `trigger_tokens` array.
- **Control Flow**:
    - Initialize a `llama_grammar_parser` object to parse the provided grammar string.
    - Check if the grammar string is successfully parsed and contains rules; if not, print an error and return `nullptr`.
    - Verify the presence of a 'root' symbol in the parsed grammar; if missing, print an error and return `nullptr`.
    - Convert parsed grammar rules into a vector of `llama_grammar_element` pointers.
    - Check for left recursion in the grammar rules using helper functions; if detected, log an error and return `nullptr`.
    - Initialize grammar stacks by iterating over alternates of the start rule and advancing stacks accordingly.
    - Convert trigger tokens and patterns into vectors, asserting their non-nullity.
    - Return a new `llama_grammar` object, moving the vectors of rules, stacks, trigger tokens, and patterns into it.
- **Output**: Returns a pointer to a newly allocated `llama_grammar` structure initialized with the parsed grammar rules, stacks, and trigger configurations, or `nullptr` if initialization fails.
- **Functions called**:
    - [`llama_grammar_detect_left_recursion`](#llama_grammar_detect_left_recursion)
    - [`llama_grammar_is_end_of_sequence`](#llama_grammar_is_end_of_sequence)
    - [`llama_grammar_advance_stack`](#llama_grammar_advance_stack)


---
### llama\_grammar\_free\_impl<!-- {{#callable:llama_grammar_free_impl}} -->
The `llama_grammar_free_impl` function deallocates memory for a `llama_grammar` structure if it is not null.
- **Inputs**:
    - `grammar`: A pointer to a `llama_grammar` structure that is to be deallocated.
- **Control Flow**:
    - Check if the `grammar` pointer is `nullptr` and return immediately if it is.
    - Use the `delete` operator to deallocate the memory for the `llama_grammar` structure.
- **Output**: The function does not return any value.


---
### llama\_grammar\_clone\_impl<!-- {{#callable:llama_grammar_clone_impl}} -->
The `llama_grammar_clone_impl` function creates a deep copy of a given `llama_grammar` structure, ensuring that all internal pointers in the stacks are redirected to the new rules in the cloned structure.
- **Inputs**:
    - `grammar`: A reference to a `llama_grammar` structure that is to be cloned.
- **Control Flow**:
    - Allocate memory for a new `llama_grammar` structure and initialize it with the same values as the input `grammar`.
    - Iterate over each stack in the `result` grammar's stacks.
    - For each element in the stack, iterate over all rules in the original `grammar`.
    - If an element in the stack points to a rule in the original `grammar`, update it to point to the corresponding rule in the `result` grammar.
    - Return the newly created `llama_grammar` structure.
- **Output**: A pointer to a newly allocated `llama_grammar` structure that is a deep copy of the input `grammar`.


---
### llama\_grammar\_apply\_impl<!-- {{#callable:llama_grammar_apply_impl}} -->
The `llama_grammar_apply_impl` function processes a set of token data against a grammar, adjusting the logit values of tokens based on grammar rules and whether they are allowed to end a sequence.
- **Inputs**:
    - `grammar`: A reference to a `llama_grammar` structure containing the grammar rules, vocabulary, and stacks used for processing tokens.
    - `cur_p`: A pointer to a `llama_token_data_array` structure that holds the current set of tokens and their associated logit values to be processed.
- **Control Flow**:
    - The function asserts that the grammar's vocabulary is not null.
    - It checks if the grammar is awaiting a trigger and returns immediately if so.
    - It initializes a boolean `allow_eog` to false and iterates over the grammar's stacks to set `allow_eog` to true if any stack is empty.
    - It prepares two vectors, `candidates_decoded` and `candidates_grammar`, to store decoded token data and grammar candidates respectively.
    - For each token in `cur_p`, it retrieves the token's ID and corresponding piece from the vocabulary.
    - If the token is an end-of-grammar (EOG) token and `allow_eog` is false, or if the piece is empty or starts with a null character, it sets the token's logit to negative infinity.
    - Otherwise, it decodes the UTF-8 piece and adds the decoded data to `candidates_decoded` and `candidates_grammar`.
    - It calls [`llama_grammar_reject_candidates`](#llama_grammar_reject_candidates) to determine which candidates should be rejected based on the grammar rules and stacks.
    - For each rejected candidate, it sets the corresponding token's logit to negative infinity.
- **Output**: The function modifies the logit values of tokens in `cur_p` based on grammar rules, potentially setting some to negative infinity to indicate rejection.
- **Functions called**:
    - [`decode_utf8`](#decode_utf8)
    - [`llama_grammar_reject_candidates`](#llama_grammar_reject_candidates)


---
### llama\_grammar\_accept\_impl<!-- {{#callable:llama_grammar_accept_impl}} -->
The `llama_grammar_accept_impl` function processes a given token within a grammar context, handling trigger conditions and updating the grammar state accordingly.
- **Inputs**:
    - `grammar`: A reference to a `llama_grammar` structure that contains the grammar rules, stacks, and other state information.
    - `token`: A `llama_token` representing the token to be processed within the grammar.
- **Control Flow**:
    - Assert that the grammar's vocabulary is not null.
    - Retrieve the string representation of the token using the grammar's vocabulary.
    - Check if the grammar is awaiting a trigger; if so, determine if the token matches any trigger tokens.
    - If a trigger token is matched, reset the trigger state, clear the trigger buffer, and accept the token string into the grammar.
    - If no trigger token is matched, append the token string to the trigger buffer and check against trigger patterns using regex.
    - If a regex pattern matches, reset the trigger state, extract the matched substring, clear the trigger buffer, and accept the substring into the grammar.
    - If still awaiting a trigger, log the current state and return.
    - If not awaiting a trigger, check if the token is an end-of-grammar (EOG) token.
    - If EOG and all stacks are empty, return; otherwise, abort with a fatal error.
    - If not EOG, accept the token string into the grammar.
- **Output**: The function does not return a value; it modifies the state of the `llama_grammar` structure in place.
- **Functions called**:
    - [`llama_grammar_accept_str`](#llama_grammar_accept_str)


---
### llama\_grammar\_accept\_str<!-- {{#callable:llama_grammar_accept_str}} -->
The `llama_grammar_accept_str` function processes a UTF-8 encoded string, updating a grammar structure by accepting each decoded code point and handling partial UTF-8 sequences.
- **Inputs**:
    - `grammar`: A reference to a `llama_grammar` structure that maintains the current state of the grammar parsing.
    - `piece`: A constant reference to a `std::string` representing the UTF-8 encoded string to be processed.
- **Control Flow**:
    - The function begins by decoding the UTF-8 string `piece` using the [`decode_utf8`](#decode_utf8) function, which returns a pair consisting of a vector of code points and a `llama_partial_utf8` structure.
    - It iterates over the decoded code points, excluding the last one, and calls [`llama_grammar_accept`](#llama_grammar_accept) for each code point to update the grammar state.
    - The `partial_utf8` member of the `grammar` is updated with the second element of the decoded pair, which represents any remaining partial UTF-8 sequence.
    - Finally, the function checks if the `stacks` member of the `grammar` is empty, and if so, throws a `std::runtime_error` indicating an unexpected empty grammar stack.
- **Output**: The function does not return a value; it modifies the `grammar` structure in place.
- **Functions called**:
    - [`decode_utf8`](#decode_utf8)
    - [`llama_grammar_accept`](#llama_grammar_accept)


