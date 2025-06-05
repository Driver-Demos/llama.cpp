# Purpose
This C++ header file provides a comprehensive set of utilities for handling Unicode code points and UTF-8 encoded strings. It defines a [`unicode_cpt_flags`](#unicode_cpt_flagsunicode_cpt_flags) structure that categorizes Unicode code points into various types such as numbers, letters, separators, and more, using both bit flags and helper flags. This structure allows for efficient storage and manipulation of Unicode character properties. The file includes functions for converting between Unicode code points and UTF-8 strings, normalizing Unicode code points, and performing case transformations. Additionally, it offers functionality for splitting strings based on regular expressions, which is useful for text processing tasks.

The file is designed to be included in other C++ projects, providing a public API for Unicode handling. It includes functions for decoding and encoding UTF-8, normalizing Unicode strings to NFD (Normalization Form D), and converting between different character representations. The use of bit flags in the [`unicode_cpt_flags`](#unicode_cpt_flagsunicode_cpt_flags) structure allows for efficient checking and manipulation of character properties, making this file a valuable resource for applications that require detailed Unicode text processing. The inclusion of regex-based string splitting further enhances its utility in text analysis and manipulation tasks.
# Imports and Dependencies

---
- `cstdint`
- `string`
- `vector`


# Data Structures

---
### unicode\_cpt\_flags<!-- {{#data_structure:unicode_cpt_flags}} -->
- **Type**: `struct`
- **Members**:
    - `is_undefined`: Indicates if the codepoint is undefined.
    - `is_number`: Indicates if the codepoint is a number, matching regex \p{N}.
    - `is_letter`: Indicates if the codepoint is a letter, matching regex \p{L}.
    - `is_separator`: Indicates if the codepoint is a separator, matching regex \p{Z}.
    - `is_accent_mark`: Indicates if the codepoint is an accent mark, matching regex \p{M}.
    - `is_punctuation`: Indicates if the codepoint is punctuation, matching regex \p{P}.
    - `is_symbol`: Indicates if the codepoint is a symbol, matching regex \p{S}.
    - `is_control`: Indicates if the codepoint is a control character, matching regex \p{C}.
    - `is_whitespace`: Indicates if the codepoint is whitespace, matching regex \s.
    - `is_lowercase`: Indicates if the codepoint is lowercase.
    - `is_uppercase`: Indicates if the codepoint is uppercase.
    - `is_nfd`: Indicates if the codepoint is in Normalization Form D (NFD).
- **Description**: The `unicode_cpt_flags` struct is designed to represent various properties of a Unicode codepoint using bit fields. It includes flags for different Unicode categories such as numbers, letters, separators, accent marks, punctuation, symbols, and control characters, as well as helper flags for whitespace, case, and normalization form. The struct provides methods to initialize from a `uint16_t` value and to extract category flags, facilitating efficient storage and manipulation of Unicode codepoint properties.
- **Member Functions**:
    - [`unicode_cpt_flags::unicode_cpt_flags`](#unicode_cpt_flagsunicode_cpt_flags)
    - [`unicode_cpt_flags::as_uint`](#unicode_cpt_flagsas_uint)
    - [`unicode_cpt_flags::category_flag`](#unicode_cpt_flagscategory_flag)

**Methods**

---
#### unicode\_cpt\_flags::unicode\_cpt\_flags<!-- {{#callable:unicode_cpt_flags::unicode_cpt_flags}} -->
The `unicode_cpt_flags` constructor initializes an instance of the `unicode_cpt_flags` structure by setting its bit fields based on a given 16-bit integer flag value.
- **Inputs**:
    - `flags`: A 16-bit unsigned integer representing the flags to initialize the `unicode_cpt_flags` structure with; defaults to 0 if not provided.
- **Control Flow**:
    - The constructor takes a 16-bit unsigned integer `flags` as an argument, defaulting to 0 if not provided.
    - It uses a reinterpret cast to treat the `this` pointer as a pointer to a `uint16_t`, allowing direct assignment of the `flags` value to the bit fields of the structure.
- **Output**: The function does not return a value; it initializes the bit fields of the `unicode_cpt_flags` structure based on the provided `flags`.
- **See also**: [`unicode_cpt_flags`](#unicode_cpt_flags)  (Data Structure)


---
#### unicode\_cpt\_flags::as\_uint<!-- {{#callable:unicode_cpt_flags::as_uint}} -->
The `as_uint` function returns the `unicode_cpt_flags` structure as a `uint16_t` integer by reinterpreting its memory layout.
- **Inputs**: None
- **Control Flow**:
    - The function uses `reinterpret_cast` to treat the `unicode_cpt_flags` object as a pointer to a `uint16_t`.
    - It dereferences this pointer to obtain the `uint16_t` value representing the entire structure's bitfield.
- **Output**: A `uint16_t` integer representing the bitfield of the `unicode_cpt_flags` structure.
- **See also**: [`unicode_cpt_flags`](#unicode_cpt_flags)  (Data Structure)


---
#### unicode\_cpt\_flags::category\_flag<!-- {{#callable:unicode_cpt_flags::category_flag}} -->
The `category_flag` function returns the category flags of a Unicode code point by masking the relevant bits from the internal representation.
- **Inputs**: None
- **Control Flow**:
    - Call the `as_uint` method to get the internal 16-bit unsigned integer representation of the `unicode_cpt_flags` object.
    - Perform a bitwise AND operation between the result of `as_uint` and `MASK_CATEGORIES` to isolate the category flags.
- **Output**: A 16-bit unsigned integer representing the category flags of the Unicode code point.
- **See also**: [`unicode_cpt_flags`](#unicode_cpt_flags)  (Data Structure)



