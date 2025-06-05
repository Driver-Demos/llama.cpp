# Purpose
This C++ source code file provides specialized functionality for handling regular expressions, specifically focusing on both full and partial matches. The code defines a class [`common_regex`](#common_regexcommon_regex) that encapsulates a regular expression pattern and provides a method [`search`](#common_regexsearch) to perform regex matching on input strings. The [`search`](#common_regexsearch) method can handle both full matches and partial matches by utilizing a reversed version of the regex pattern. This is achieved through the function [`regex_to_reversed_partial_regex`](#regex_to_reversed_partial_regex), which transforms a given regex pattern into a form that can be used to identify partial matches at the end of a string by reversing the input and applying the transformed pattern.

The file is not a standalone executable but rather a component intended to be used as part of a larger system, likely as a library or module. It includes headers such as "regex-partial.h" and "common.h", suggesting that it is part of a broader codebase. The code does not define a public API or external interface directly but provides internal functionality that can be leveraged by other parts of the system. The transformation of regex patterns to handle partial matches is a key technical component, addressing the limitation of the standard library's regex capabilities by implementing a custom solution for partial matching.
# Imports and Dependencies

---
- `regex-partial.h`
- `common.h`
- `functional`
- `optional`


# Data Structures

---
### common\_regex<!-- {{#data_structure:common_regex}} -->
- **Description**: [See definition](regex-partial.h.driver.md#common_regex)
- **Member Functions**:
    - [`common_regex::common_regex`](#common_regexcommon_regex)
    - [`common_regex::search`](#common_regexsearch)
    - [`common_regex::str`](regex-partial.h.driver.md#common_regexstr)

**Methods**

---
#### common\_regex::common\_regex<!-- {{#callable:common_regex::common_regex}} -->
The `common_regex` constructor initializes a `common_regex` object with a given pattern, compiling it into a standard regex and a reversed partial regex.
- **Inputs**:
    - `pattern`: A string representing the regular expression pattern to be used for matching.
- **Control Flow**:
    - The constructor initializes the `pattern` member variable with the provided pattern string.
    - It compiles the `pattern` into a `std::regex` object and assigns it to the `rx` member variable.
    - It calls the [`regex_to_reversed_partial_regex`](#regex_to_reversed_partial_regex) function with the `pattern` to generate a reversed partial regex and assigns it to the `rx_reversed_partial` member variable.
- **Output**: The constructor does not return a value; it initializes the member variables of the `common_regex` object.
- **Functions called**:
    - [`regex_to_reversed_partial_regex`](#regex_to_reversed_partial_regex)
- **See also**: [`common_regex`](regex-partial.h.driver.md#common_regex)  (Data Structure)


---
#### common\_regex::search<!-- {{#callable:common_regex::search}} -->
The `search` function performs a regex search or match on a given input string starting from a specified position and returns a `common_regex_match` object containing the match details.
- **Inputs**:
    - `input`: A constant reference to a `std::string` representing the input text to be searched.
    - `pos`: A `size_t` representing the starting position in the input string from which the search or match should begin.
    - `as_match`: A `bool` indicating whether to perform a full match (`true`) or a search (`false`).
- **Control Flow**:
    - Check if the starting position `pos` is greater than the input string size and throw a runtime error if true.
    - Initialize a `std::smatch` object to store match results.
    - Determine whether to use `std::regex_match` or `std::regex_search` based on the `as_match` flag and perform the operation from the specified starting position.
    - If a match is found, create a `common_regex_match` object, set its type to `COMMON_REGEX_MATCH_TYPE_FULL`, and populate its `groups` with the positions and lengths of the matches.
    - If no match is found, attempt a partial match using a reversed regex pattern and reversed input string.
    - If a partial match is found, create a `common_regex_match` object, set its type to `COMMON_REGEX_MATCH_TYPE_PARTIAL`, and populate its `groups` with the range of the partial match.
    - Return the `common_regex_match` object if a match or partial match is found, otherwise return an empty `common_regex_match` object.
- **Output**: A `common_regex_match` object containing the type of match (full or partial) and the positions of the matched groups, or an empty object if no match is found.
- **See also**: [`common_regex`](regex-partial.h.driver.md#common_regex)  (Data Structure)



# Functions

---
### regex\_to\_reversed\_partial\_regex<!-- {{#callable:regex_to_reversed_partial_regex}} -->
The function `regex_to_reversed_partial_regex` transforms a given regex pattern into a partial match pattern that operates on a reversed input string to find partial final matches of the original pattern.
- **Inputs**:
    - `pattern`: A string representing the regex pattern to be transformed.
- **Control Flow**:
    - Initialize iterators `it` and `end` to traverse the input pattern.
    - Define a lambda function `process` to handle the transformation of the regex pattern.
    - Within `process`, initialize a vector of alternatives to store different sequences of regex components.
    - Iterate over the pattern using `it`, handling different regex components such as character classes, quantifiers, repetition ranges, groups, and alternation.
    - For character classes, ensure matching brackets and add the class to the current sequence.
    - For quantifiers, ensure they follow a valid element and append them to the last element of the sequence.
    - For repetition ranges, parse the range, validate it, and expand the sequence accordingly.
    - For groups, recursively call `process` to handle nested patterns and ensure matching parentheses.
    - For alternation, create new sequences for each alternative pattern.
    - After processing, construct the reversed partial regex by reversing sequences and adding non-capturing groups.
    - Return the final transformed regex pattern with a capturing group and a wildcard match at the end.
- **Output**: A string representing the transformed regex pattern suitable for matching reversed input strings.


