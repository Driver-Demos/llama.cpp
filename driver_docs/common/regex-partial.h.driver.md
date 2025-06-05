# Purpose
This C++ header file defines a set of classes and structures for handling regular expression matching with a focus on capturing match types and string ranges. The primary components include an enumeration `common_regex_match_type` to categorize the type of match (none, partial, or full), and a [`common_string_range`](#common_string_rangecommon_string_range) structure to represent a range within a string, ensuring that the range is valid upon construction. The `common_regex_match` structure encapsulates the match type and a collection of string ranges, providing equality operators for comparison. The `common_regex` class is central to this file, encapsulating a regular expression pattern and providing a method to search for matches within a given input string. This class also maintains a reversed partial regex for specific matching scenarios, although the details of its usage are not fully elaborated in this snippet.

The file is designed to be included in other C++ source files, as indicated by the `#pragma once` directive, which prevents multiple inclusions. It provides a public API through the `common_regex` class, allowing users to instantiate it with a pattern and perform searches on input strings. The file also includes a utility function `regex_to_reversed_partial_regex`, intended for testing purposes, which likely transforms a regex pattern into a form suitable for reversed partial matching. Overall, this code offers a focused functionality for regex operations, emphasizing match categorization and range management, and is likely part of a larger library or application dealing with text processing or pattern matching.
# Imports and Dependencies

---
- `regex`
- `string`


# Data Structures

---
### common\_regex\_match\_type<!-- {{#data_structure:common_regex_match_type}} -->
- **Type**: `enum`
- **Members**:
    - `COMMON_REGEX_MATCH_TYPE_NONE`: Represents no match found in the regex operation.
    - `COMMON_REGEX_MATCH_TYPE_PARTIAL`: Indicates a partial match was found in the regex operation.
    - `COMMON_REGEX_MATCH_TYPE_FULL`: Denotes a full match was found in the regex operation.
- **Description**: The `common_regex_match_type` enum defines three possible states for the result of a regex match operation: no match, partial match, and full match. This enum is used to categorize the outcome of regex operations, allowing for clear differentiation between different levels of matching success.


---
### common\_string\_range<!-- {{#data_structure:common_string_range}} -->
- **Type**: `struct`
- **Members**:
    - `begin`: The starting index of the string range.
    - `end`: The ending index of the string range.
- **Description**: The `common_string_range` struct represents a range within a string, defined by a starting index (`begin`) and an ending index (`end`). It includes a constructor that initializes these indices and checks for validity by ensuring that `begin` is not greater than `end`, throwing a runtime error if this condition is violated. The struct also provides a method to check if the range is empty and an equality operator to compare two ranges.
- **Member Functions**:
    - [`common_string_range::common_string_range`](#common_string_rangecommon_string_range)
    - [`common_string_range::common_string_range`](#common_string_rangecommon_string_range)
    - [`common_string_range::empty`](#common_string_rangeempty)
    - [`common_string_range::operator==`](#common_string_rangeoperator==)

**Methods**

---
#### common\_string\_range::common\_string\_range<!-- {{#callable:common_string_range::common_string_range}} -->
The `common_string_range` constructor initializes a range with specified begin and end indices, ensuring the begin index is not greater than the end index.
- **Inputs**:
    - `begin`: The starting index of the range, of type `size_t`.
    - `end`: The ending index of the range, of type `size_t`.
- **Control Flow**:
    - The constructor initializes the `begin` and `end` member variables with the provided arguments.
    - It checks if the `begin` index is greater than the `end` index.
    - If `begin` is greater than `end`, it throws a `std::runtime_error` with the message "Invalid range".
- **Output**: The constructor does not return a value, but it initializes the `common_string_range` object or throws an exception if the range is invalid.
- **See also**: [`common_string_range`](#common_string_range)  (Data Structure)


---
#### common\_string\_range::common\_string\_range<!-- {{#callable:common_string_range::common_string_range}} -->
The `empty` method checks if the `common_string_range` object represents an empty range by comparing its `begin` and `end` values.
- **Inputs**: None
- **Control Flow**:
    - The method compares the `begin` and `end` member variables of the `common_string_range` object.
    - If `begin` is equal to `end`, the method returns `true`, indicating the range is empty.
    - Otherwise, it returns `false`, indicating the range is not empty.
- **Output**: A boolean value indicating whether the range is empty (`true`) or not (`false`).
- **See also**: [`common_string_range`](#common_string_range)  (Data Structure)


---
#### common\_string\_range::empty<!-- {{#callable:common_string_range::empty}} -->
The `empty` function checks if the `common_string_range` object represents an empty range by comparing its `begin` and `end` values.
- **Inputs**: None
- **Control Flow**:
    - The function compares the `begin` and `end` member variables of the `common_string_range` object.
    - If `begin` is equal to `end`, the function returns `true`, indicating the range is empty.
    - If `begin` is not equal to `end`, the function returns `false`, indicating the range is not empty.
- **Output**: A boolean value indicating whether the range is empty (`true`) or not (`false`).
- **See also**: [`common_string_range`](#common_string_range)  (Data Structure)


---
#### common\_string\_range::operator==<!-- {{#callable:common_string_range::operator==}} -->
The `operator==` function compares two `common_string_range` objects for equality by checking if their `begin` and `end` values are the same.
- **Inputs**:
    - `other`: A reference to another `common_string_range` object to compare against the current object.
- **Control Flow**:
    - The function checks if the `begin` value of the current object is equal to the `begin` value of the `other` object.
    - It then checks if the `end` value of the current object is equal to the `end` value of the `other` object.
    - The function returns true if both the `begin` and `end` values are equal, otherwise it returns false.
- **Output**: A boolean value indicating whether the two `common_string_range` objects are equal.
- **See also**: [`common_string_range`](#common_string_range)  (Data Structure)



---
### common\_regex\_match<!-- {{#data_structure:common_regex_match}} -->
- **Type**: `struct`
- **Members**:
    - `type`: Specifies the type of regex match, initialized to COMMON_REGEX_MATCH_TYPE_NONE.
    - `groups`: A vector of common_string_range objects representing matched groups.
- **Description**: The `common_regex_match` struct is designed to encapsulate the result of a regex match operation. It includes a `type` field to indicate the nature of the match (none, partial, or full) and a `groups` vector to store the ranges of matched substrings. The struct also provides equality and inequality operators to compare match results.
- **Member Functions**:
    - [`common_regex_match::operator==`](#common_regex_matchoperator==)
    - [`common_regex_match::operator!=`](#common_regex_matchoperator!=)

**Methods**

---
#### common\_regex\_match::operator==<!-- {{#callable:common_regex_match::operator==}} -->
The `operator==` function compares two `common_regex_match` objects for equality by checking if their `type` and `groups` attributes are equal.
- **Inputs**:
    - `other`: A reference to another `common_regex_match` object to compare against the current object.
- **Control Flow**:
    - The function checks if the `type` attribute of the current object is equal to the `type` attribute of the `other` object.
    - It then checks if the `groups` vector of the current object is equal to the `groups` vector of the `other` object.
    - The function returns `true` if both the `type` and `groups` are equal, otherwise it returns `false`.
- **Output**: A boolean value indicating whether the current `common_regex_match` object is equal to the `other` object.
- **See also**: [`common_regex_match`](#common_regex_match)  (Data Structure)


---
#### common\_regex\_match::operator\!=<!-- {{#callable:common_regex_match::operator!=}} -->
The `operator!=` function checks if two `common_regex_match` objects are not equal by negating the result of the equality operator.
- **Inputs**:
    - `other`: A reference to another `common_regex_match` object to compare against the current object.
- **Control Flow**:
    - The function calls the equality operator `operator==` to compare the current object with the `other` object.
    - It negates the result of the equality check to determine inequality.
- **Output**: A boolean value indicating whether the two `common_regex_match` objects are not equal.
- **See also**: [`common_regex_match`](#common_regex_match)  (Data Structure)



---
### common\_regex<!-- {{#data_structure:common_regex}} -->
- **Type**: `class`
- **Members**:
    - `pattern`: Stores the regex pattern as a string.
    - `rx`: Holds the compiled regex object for the pattern.
    - `rx_reversed_partial`: Contains the compiled regex object for the reversed partial pattern.
- **Description**: The `common_regex` class encapsulates a regular expression pattern and provides functionality to search for matches within a given input string. It maintains the original pattern as a string and compiles it into a regex object for efficient searching. Additionally, it supports searching for both full and partial matches, utilizing a reversed partial regex for the latter. The class offers a constructor to initialize the pattern and a search method to perform regex operations on input strings, returning a `common_regex_match` object that details the match type and groups found.
- **Member Functions**:
    - [`common_regex::common_regex`](regex-partial.cpp.driver.md#common_regexcommon_regex)
    - [`common_regex::search`](regex-partial.cpp.driver.md#common_regexsearch)
    - [`common_regex::str`](#common_regexstr)

**Methods**

---
#### common\_regex::str<!-- {{#callable:common_regex::str}} -->
The `str` method returns the regex pattern string stored in the `common_regex` object.
- **Inputs**: None
- **Control Flow**:
    - The method directly returns the `pattern` member variable of the `common_regex` class.
- **Output**: A constant reference to the `std::string` representing the regex pattern.
- **See also**: [`common_regex`](#common_regex)  (Data Structure)



