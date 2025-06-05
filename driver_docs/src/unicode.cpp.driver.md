# Purpose
This C++ source code file provides a comprehensive set of functions for handling Unicode text, specifically focusing on UTF-8 encoding and decoding, as well as text processing using regular expressions. The file includes functions to convert between Unicode code points and UTF-8 strings, normalize Unicode text, and perform regex-based text splitting. It also defines mappings between bytes and UTF-8 strings, and provides utilities for handling Unicode character properties such as whitespace, letter, and number classifications. The code is structured to support efficient text processing, including custom implementations for regex-based tokenization tailored for specific use cases like GPT-2 and LLAMA3 systems.

The file is not an executable but rather a library intended to be used by other parts of a software system that require Unicode text processing capabilities. It defines several public APIs, such as [`unicode_cpt_to_utf8`](#unicode_cpt_to_utf8), [`unicode_cpts_from_utf8`](#unicode_cpts_from_utf8), and [`unicode_regex_split`](#unicode_regex_split), which can be used to convert and manipulate Unicode text. The code also includes static functions and data structures to support internal operations, such as mapping Unicode code points to their properties and handling custom regex patterns. The inclusion of headers like `<regex>`, `<string>`, and `<unordered_map>` indicates the use of standard C++ libraries for string manipulation and regular expression processing.
# Imports and Dependencies

---
- `unicode.h`
- `unicode-data.h`
- `algorithm`
- `cassert`
- `codecvt`
- `cstddef`
- `cstdint`
- `locale`
- `map`
- `regex`
- `stdexcept`
- `string`
- `unordered_map`
- `utility`
- `vector`


# Functions

---
### unicode\_len\_utf8<!-- {{#callable:unicode_len_utf8}} -->
The `unicode_len_utf8` function determines the number of bytes required to represent a UTF-8 encoded character based on its leading byte.
- **Inputs**:
    - `src`: A single character (of type `char`) representing the leading byte of a UTF-8 encoded character.
- **Control Flow**:
    - The function defines a lookup table `lookup` with 16 elements, where each element represents the number of bytes required for a UTF-8 character based on the high 4 bits of the leading byte.
    - The function calculates `highbits` by right-shifting the input character `src` by 4 bits, effectively isolating the high 4 bits of the byte.
    - The function returns the value from the `lookup` table at the index specified by `highbits`, which corresponds to the number of bytes needed for the UTF-8 character.
- **Output**: The function returns a `size_t` value indicating the number of bytes required to represent the UTF-8 character.


---
### unicode\_cpts\_to\_utf8<!-- {{#callable:unicode_cpts_to_utf8}} -->
The function `unicode_cpts_to_utf8` converts a vector of Unicode code points into a UTF-8 encoded string.
- **Inputs**:
    - `cps`: A constant reference to a vector of 32-bit unsigned integers, each representing a Unicode code point.
- **Control Flow**:
    - Initialize an empty string `result` to store the UTF-8 encoded output.
    - Iterate over each code point in the input vector `cps`.
    - For each code point, convert it to a UTF-8 string using the [`unicode_cpt_to_utf8`](#unicode_cpt_to_utf8) function and append the result to the `result` string.
    - Return the `result` string containing the concatenated UTF-8 encoded characters.
- **Output**: A string containing the UTF-8 encoded representation of the input Unicode code points.
- **Functions called**:
    - [`unicode_cpt_to_utf8`](#unicode_cpt_to_utf8)


---
### unicode\_cpt\_from\_utf8<!-- {{#callable:unicode_cpt_from_utf8}} -->
The function `unicode_cpt_from_utf8` decodes a UTF-8 encoded string starting from a given offset and returns the corresponding Unicode code point, updating the offset to the next character position.
- **Inputs**:
    - `utf8`: A constant reference to a UTF-8 encoded string from which the Unicode code point is to be extracted.
    - `offset`: A reference to a size_t variable indicating the starting position in the string; it is updated to point to the next character position after decoding.
- **Control Flow**:
    - Assert that the offset is within the bounds of the string.
    - Check if the first byte indicates a single-byte character (ASCII) and return it if true, updating the offset by 1.
    - Check if the first byte is invalid (not starting with 10xxxxxx) and throw an exception if true.
    - Check if the first byte indicates a two-byte character, validate the second byte, decode the character, update the offset by 2, and return the result.
    - Check if the first byte indicates a three-byte character, validate the next two bytes, decode the character, update the offset by 3, and return the result.
    - Check if the first byte indicates a four-byte character, validate the next three bytes, decode the character, update the offset by 4, and return the result.
    - If none of the conditions are met, throw an exception indicating a failure to convert UTF-8 to a code point.
- **Output**: Returns a uint32_t representing the Unicode code point extracted from the UTF-8 string.


---
### unicode\_cpt\_flags\_array<!-- {{#callable:unicode_cpt_flags_array}} -->
The `unicode_cpt_flags_array` function initializes and returns a vector of `unicode_cpt_flags` for all Unicode code points, setting specific flags based on predefined ranges and mappings.
- **Inputs**: None
- **Control Flow**:
    - Initialize a vector `cpt_flags` with `MAX_CODEPOINTS` elements, all set to `unicode_cpt_flags::UNDEFINED`.
    - Assert that the first and last elements of `unicode_ranges_flags` cover the entire range of code points from 0 to `MAX_CODEPOINTS`.
    - Iterate over `unicode_ranges_flags` to set the flags for each code point in the range to the corresponding flag from `range_ini`.
    - Iterate over `unicode_set_whitespace` to set the `is_whitespace` flag for each code point in the set.
    - Iterate over `unicode_map_lowercase` to set the `is_lowercase` flag for each mapped code point.
    - Iterate over `unicode_map_uppercase` to set the `is_uppercase` flag for each mapped code point.
    - Iterate over `unicode_ranges_nfd` to set the `is_nfd` flag for each `nfd` code point in the range.
    - Return the populated `cpt_flags` vector.
- **Output**: A vector of `unicode_cpt_flags` where each element corresponds to a Unicode code point with specific flags set based on predefined ranges and mappings.


---
### unicode\_byte\_to\_utf8\_map<!-- {{#callable:unicode_byte_to_utf8_map}} -->
The function `unicode_byte_to_utf8_map` creates and returns a map that associates each byte value (0-255) with its corresponding UTF-8 encoded string representation.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty unordered map `map` to store byte-to-UTF-8 mappings.
    - Iterate over the range of byte values from 0x21 to 0x7E, 0xA1 to 0xAC, and 0xAE to 0xFF, converting each byte to its UTF-8 representation using [`unicode_cpt_to_utf8`](#unicode_cpt_to_utf8) and storing it in `map`.
    - For any byte values not covered in the previous ranges (0 to 255), map them to a UTF-8 representation of a code point starting from 256, incrementing for each unmapped byte.
    - Return the completed map.
- **Output**: An unordered map where each key is a byte (uint8_t) and each value is the corresponding UTF-8 encoded string.
- **Functions called**:
    - [`unicode_cpt_to_utf8`](#unicode_cpt_to_utf8)


---
### unicode\_utf8\_to\_byte\_map<!-- {{#callable:unicode_utf8_to_byte_map}} -->
The function `unicode_utf8_to_byte_map` creates a mapping from UTF-8 encoded strings to their corresponding byte values for a specific range of Unicode characters.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty unordered map `map` to store string to byte mappings.
    - Iterate over Unicode code points from 0x21 to 0x7E, 0xA1 to 0xAC, and 0xAE to 0xFF, converting each to UTF-8 and storing the mapping in `map`.
    - For each code point, assert that it is within the valid byte range (0 to 255).
    - Iterate over all possible byte values (0 to 255) and check if their UTF-8 representation is already in `map`.
    - If a UTF-8 representation is not found in `map`, map it to a new byte value starting from 256, incrementing `n` for each new mapping.
    - Return the completed map.
- **Output**: An unordered map where keys are UTF-8 encoded strings and values are their corresponding byte values.
- **Functions called**:
    - [`unicode_cpt_to_utf8`](#unicode_cpt_to_utf8)


---
### unicode\_wstring\_from\_utf8<!-- {{#callable:unicode_wstring_from_utf8}} -->
The function `unicode_wstring_from_utf8` converts a UTF-8 encoded `std::string` to a `std::wstring` using a deprecated codecvt facet.
- **Inputs**:
    - `s`: A `std::string` containing UTF-8 encoded text to be converted to a wide string.
- **Control Flow**:
    - The function begins by checking if the code is being compiled with Clang and suppresses the deprecation warning for `std::codecvt_utf8`.
    - A `std::wstring_convert` object is created using `std::codecvt_utf8<wchar_t>` to handle the conversion from UTF-8 to wide string.
    - The function then calls `from_bytes` on the `std::wstring_convert` object, passing the input string `s` to convert it to a wide string.
    - Finally, the function returns the resulting `std::wstring`.
- **Output**: A `std::wstring` that represents the wide character equivalent of the input UTF-8 encoded string.


---
### unicode\_byte\_encoding\_process<!-- {{#callable:unicode_byte_encoding_process}} -->
The function `unicode_byte_encoding_process` converts a list of words from UTF-8 encoding to a custom byte encoding using Unicode code points.
- **Inputs**:
    - `bpe_words`: A constant reference to a vector of strings, where each string represents a word in UTF-8 encoding.
- **Control Flow**:
    - Initialize an empty vector `bpe_encoded_words` to store the encoded words.
    - Iterate over each word in the input vector `bpe_words`.
    - For each word, convert it from UTF-8 to a vector of Unicode code points using [`unicode_cpts_from_utf8`](#unicode_cpts_from_utf8).
    - Convert each Unicode code point back to a UTF-8 string using [`unicode_cpt_to_utf8`](#unicode_cpt_to_utf8) and concatenate them to form `text_utf`.
    - For each character in `text_utf`, convert it to a custom byte encoding using [`unicode_byte_to_utf8`](#unicode_byte_to_utf8) and concatenate the results to form `encoded_token`.
    - Add `encoded_token` to the `bpe_encoded_words` vector.
    - Return the `bpe_encoded_words` vector.
- **Output**: A vector of strings, where each string is the encoded version of the corresponding input word using a custom byte encoding.
- **Functions called**:
    - [`unicode_cpts_from_utf8`](#unicode_cpts_from_utf8)
    - [`unicode_cpt_to_utf8`](#unicode_cpt_to_utf8)
    - [`unicode_byte_to_utf8`](#unicode_byte_to_utf8)


---
### unicode\_regex\_split\_custom\_gpt2<!-- {{#callable:unicode_regex_split_custom_gpt2}} -->
The `unicode_regex_split_custom_gpt2` function splits a given UTF-8 encoded text into segments based on specific regex patterns, returning the lengths of these segments.
- **Inputs**:
    - `text`: A UTF-8 encoded string that needs to be split into segments.
    - `offsets`: A vector of size_t values representing the lengths of segments in the text to be processed.
- **Control Flow**:
    - Initialize an empty vector `bpe_offsets` to store the lengths of the segments and reserve memory based on the size of `offsets`.
    - Convert the input `text` into a vector of Unicode code points using [`unicode_cpts_from_utf8`](#unicode_cpts_from_utf8).
    - Iterate over each offset in `offsets`, setting `offset_ini` and `offset_end` to define the range of code points to process.
    - For each code point in the range, use lambda functions `_get_cpt` and `_get_flags` to retrieve the code point and its flags, respectively.
    - Use a series of regex-like conditions to identify segments such as contractions (e.g., 's, 't), letters, numbers, non-whitespace symbols, and whitespace sequences.
    - For each identified segment, calculate its length and add it to `bpe_offsets` using the `_add_token` lambda function.
    - Continue processing until all code points in the current offset range are handled.
    - Return the `bpe_offsets` vector containing the lengths of the identified segments.
- **Output**: A vector of size_t values representing the lengths of the segments identified in the input text.
- **Functions called**:
    - [`unicode_cpts_from_utf8`](#unicode_cpts_from_utf8)
    - [`unicode_cpt_flags_from_cpt`](#unicode_cpt_flags_from_cpt)


---
### unicode\_regex\_split\_custom\_llama3<!-- {{#callable:unicode_regex_split_custom_llama3}} -->
The function `unicode_regex_split_custom_llama3` splits a given UTF-8 encoded text into segments based on specific regex-like rules, returning the lengths of these segments.
- **Inputs**:
    - `text`: A UTF-8 encoded string that needs to be split into segments.
    - `offsets`: A vector of size_t values representing the initial offsets for splitting the text.
- **Control Flow**:
    - Initialize a vector `bpe_offsets` to store the lengths of the segments and reserve memory based on the size of `offsets`.
    - Convert the input `text` into a vector of Unicode code points using [`unicode_cpts_from_utf8`](#unicode_cpts_from_utf8).
    - Iterate over each offset in `offsets`, setting `offset_ini` and `offset_end` to define the range of code points to process.
    - For each code point in the range, use lambda functions `_get_cpt` and `_get_flags` to retrieve the code point and its flags.
    - Use a series of regex-like conditions to determine how to split the text, such as handling contractions, letters, numbers, and whitespace.
    - For each match, use the `_add_token` lambda to calculate the length of the segment and add it to `bpe_offsets`.
    - Continue processing until all code points in the range are evaluated.
- **Output**: A vector of size_t values representing the lengths of the segments obtained by splitting the input text.
- **Functions called**:
    - [`unicode_cpts_from_utf8`](#unicode_cpts_from_utf8)
    - [`unicode_cpt_flags_from_cpt`](#unicode_cpt_flags_from_cpt)
    - [`unicode_tolower`](#unicode_tolower)


---
### unicode\_regex\_split\_stl<!-- {{#callable:unicode_regex_split_stl}} -->
The `unicode_regex_split_stl` function splits a given text into segments based on a specified regular expression and returns the offsets of these segments.
- **Inputs**:
    - `text`: A `std::string` representing the text to be split.
    - `regex_expr`: A `std::string` containing the regular expression used to split the text.
    - `offsets`: A `std::vector<size_t>` representing the initial offsets of segments in the text.
- **Control Flow**:
    - Initialize a `std::regex` object with the given `regex_expr`.
    - Reserve memory in `bpe_offsets` vector based on the size of `offsets`.
    - Iterate over each offset in `offsets`, adjusting the starting position for each segment.
    - For each segment, use `std::cregex_iterator` to find matches of the regular expression within the segment.
    - For each match, calculate the position and length, and store these in `bpe_offsets`.
    - If there is any remaining unmatched text in the segment, calculate its length and store it in `bpe_offsets`.
    - Update the starting position for the next segment by adding the current offset.
- **Output**: A `std::vector<size_t>` containing the lengths of the segments resulting from the split operation.


---
### unicode\_regex\_split\_custom<!-- {{#callable:unicode_regex_split_custom}} -->
The `unicode_regex_split_custom` function splits a given text into segments based on specified regular expressions and returns the offsets of these segments.
- **Inputs**:
    - `text`: A string representing the text to be split.
    - `regex_expr`: A string representing the regular expression used to determine the splitting criteria.
    - `offsets`: A vector of size_t representing the initial offsets for splitting the text.
- **Control Flow**:
    - Initialize an empty vector `bpe_offsets` to store the resulting offsets.
    - Check if `regex_expr` matches a specific pattern for GPT2; if so, call [`unicode_regex_split_custom_gpt2`](#unicode_regex_split_custom_gpt2) with `text` and `offsets` to get the offsets.
    - Check if `regex_expr` matches a specific pattern for LLAMA3; if so, call [`unicode_regex_split_custom_llama3`](#unicode_regex_split_custom_llama3) with `text` and `offsets` to get the offsets.
    - Return the `bpe_offsets` vector.
- **Output**: A vector of size_t containing the offsets of the segments in the text after splitting based on the regular expression.
- **Functions called**:
    - [`unicode_regex_split_custom_gpt2`](#unicode_regex_split_custom_gpt2)
    - [`unicode_regex_split_custom_llama3`](#unicode_regex_split_custom_llama3)


---
### unicode\_cpt\_to\_utf8<!-- {{#callable:unicode_cpt_to_utf8}} -->
The function `unicode_cpt_to_utf8` converts a Unicode code point to its corresponding UTF-8 encoded string.
- **Inputs**:
    - `cpt`: A 32-bit unsigned integer representing a Unicode code point.
- **Control Flow**:
    - Initialize an empty string `result` to store the UTF-8 encoded characters.
    - Check if the code point `cpt` is in the range 0x00 to 0x7F (1-byte UTF-8 encoding) and append it directly to `result`.
    - Check if `cpt` is in the range 0x80 to 0x7FF (2-byte UTF-8 encoding), calculate and append the corresponding bytes to `result`.
    - Check if `cpt` is in the range 0x800 to 0xFFFF (3-byte UTF-8 encoding), calculate and append the corresponding bytes to `result`.
    - Check if `cpt` is in the range 0x10000 to 0x10FFFF (4-byte UTF-8 encoding), calculate and append the corresponding bytes to `result`.
    - If `cpt` does not fall into any of the valid ranges, throw an `std::invalid_argument` exception indicating an invalid code point.
- **Output**: A string containing the UTF-8 encoded representation of the input Unicode code point.


---
### unicode\_cpts\_normalize\_nfd<!-- {{#callable:unicode_cpts_normalize_nfd}} -->
The function `unicode_cpts_normalize_nfd` normalizes a vector of Unicode code points to their NFD (Normalization Form D) equivalents using predefined Unicode ranges.
- **Inputs**:
    - `cpts`: A constant reference to a vector of `uint32_t` representing Unicode code points to be normalized.
- **Control Flow**:
    - Define a lambda function `comp` to compare a code point with the start of a Unicode range.
    - Initialize a result vector `result` with the same size as `cpts`.
    - Iterate over each code point in `cpts`.
    - For each code point, use `std::upper_bound` to find the appropriate Unicode range in `unicode_ranges_nfd`.
    - Check if the code point falls within the found range; if so, replace it with its NFD equivalent, otherwise keep the original code point.
    - Store the result in the `result` vector.
    - Return the `result` vector.
- **Output**: A vector of `uint32_t` containing the normalized Unicode code points.


---
### unicode\_cpts\_from\_utf8<!-- {{#callable:unicode_cpts_from_utf8}} -->
The function `unicode_cpts_from_utf8` converts a UTF-8 encoded string into a vector of Unicode code points, handling invalid UTF-8 sequences by inserting a replacement character.
- **Inputs**:
    - `utf8`: A constant reference to a UTF-8 encoded string that needs to be converted into Unicode code points.
- **Control Flow**:
    - Initialize an empty vector `result` to store Unicode code points and reserve space equivalent to the size of the input string.
    - Set an offset variable to 0 to track the current position in the input string.
    - Enter a while loop that continues until the offset reaches the end of the input string.
    - Within the loop, attempt to convert the current UTF-8 sequence starting at the offset to a Unicode code point using [`unicode_cpt_from_utf8`](#unicode_cpt_from_utf8).
    - If the conversion is successful, append the code point to the `result` vector and update the offset accordingly.
    - If an `std::invalid_argument` exception is caught (indicating an invalid UTF-8 sequence), increment the offset by one and append the Unicode replacement character (0xFFFD) to the `result` vector.
    - Continue the loop until the entire input string is processed.
- **Output**: A vector of `uint32_t` representing the Unicode code points extracted from the input UTF-8 string, with invalid sequences replaced by the Unicode replacement character.
- **Functions called**:
    - [`unicode_cpt_from_utf8`](#unicode_cpt_from_utf8)


---
### unicode\_cpt\_flags\_from\_cpt<!-- {{#callable:unicode_cpt_flags_from_cpt}} -->
The function `unicode_cpt_flags_from_cpt` retrieves the Unicode code point flags for a given code point, returning a default undefined flag if the code point is out of range.
- **Inputs**:
    - `cpt`: A 32-bit unsigned integer representing a Unicode code point.
- **Control Flow**:
    - Define a static `unicode_cpt_flags` object `undef` initialized with `UNDEFINED` flag.
    - Define a static array `cpt_flags` by calling `unicode_cpt_flags_array()`.
    - Check if the input `cpt` is less than the size of `cpt_flags`.
    - If true, return the flag at index `cpt` from `cpt_flags`.
    - If false, return the `undef` flag.
- **Output**: Returns a `unicode_cpt_flags` object representing the flags associated with the given code point, or an undefined flag if the code point is out of range.
- **Functions called**:
    - [`unicode_cpt_flags_array`](#unicode_cpt_flags_array)


---
### unicode\_cpt\_flags\_from\_utf8<!-- {{#callable:unicode_cpt_flags_from_utf8}} -->
The function `unicode_cpt_flags_from_utf8` converts a UTF-8 encoded string into its corresponding Unicode code point flags.
- **Inputs**:
    - `utf8`: A constant reference to a `std::string` representing a UTF-8 encoded string.
- **Control Flow**:
    - Check if the input string `utf8` is empty; if so, return a static `unicode_cpt_flags` object representing an undefined state.
    - Initialize a `size_t` variable `offset` to 0 to track the position within the UTF-8 string.
    - Call [`unicode_cpt_from_utf8`](#unicode_cpt_from_utf8) with the input string and `offset` to convert the first UTF-8 character to a Unicode code point.
    - Pass the resulting code point to [`unicode_cpt_flags_from_cpt`](#unicode_cpt_flags_from_cpt) to obtain the corresponding Unicode code point flags.
    - Return the obtained Unicode code point flags.
- **Output**: Returns a `unicode_cpt_flags` object that represents the flags associated with the first Unicode code point in the UTF-8 string.
- **Functions called**:
    - [`unicode_cpt_flags_from_cpt`](#unicode_cpt_flags_from_cpt)
    - [`unicode_cpt_from_utf8`](#unicode_cpt_from_utf8)


---
### unicode\_byte\_to\_utf8<!-- {{#callable:unicode_byte_to_utf8}} -->
The `unicode_byte_to_utf8` function converts a single byte into its corresponding UTF-8 string representation using a predefined mapping.
- **Inputs**:
    - `byte`: A single byte (uint8_t) that represents a Unicode character to be converted to UTF-8.
- **Control Flow**:
    - A static unordered map is initialized using the [`unicode_byte_to_utf8_map`](#unicode_byte_to_utf8_map) function, which maps bytes to their UTF-8 string equivalents.
    - The function retrieves the UTF-8 string corresponding to the input byte from the map using the `at` method.
- **Output**: A string representing the UTF-8 encoding of the input byte.
- **Functions called**:
    - [`unicode_byte_to_utf8_map`](#unicode_byte_to_utf8_map)


---
### unicode\_utf8\_to\_byte<!-- {{#callable:unicode_utf8_to_byte}} -->
The `unicode_utf8_to_byte` function converts a UTF-8 encoded string to its corresponding byte value using a predefined mapping.
- **Inputs**:
    - `utf8`: A UTF-8 encoded string that represents a Unicode character.
- **Control Flow**:
    - A static unordered map `map` is initialized using the [`unicode_utf8_to_byte_map`](#unicode_utf8_to_byte_map) function, which creates a mapping from UTF-8 strings to byte values.
    - The function retrieves the byte value corresponding to the input UTF-8 string from the map using the `at` method, which throws an exception if the key is not found.
- **Output**: Returns a `uint8_t` byte value that corresponds to the input UTF-8 string.
- **Functions called**:
    - [`unicode_utf8_to_byte_map`](#unicode_utf8_to_byte_map)


---
### unicode\_tolower<!-- {{#callable:unicode_tolower}} -->
The `unicode_tolower` function converts a given Unicode code point to its lowercase equivalent if a mapping exists, otherwise it returns the original code point.
- **Inputs**:
    - `cpt`: A 32-bit unsigned integer representing a Unicode code point to be converted to lowercase.
- **Control Flow**:
    - Perform a binary search on the `unicode_map_lowercase` to find the input code point `cpt`.
    - If a matching code point is found in the map, return its corresponding lowercase code point.
    - If no matching code point is found, return the original code point `cpt`.
- **Output**: A 32-bit unsigned integer representing the lowercase equivalent of the input code point, or the original code point if no lowercase mapping is found.


---
### unicode\_regex\_split<!-- {{#callable:unicode_regex_split}} -->
The `unicode_regex_split` function splits a given UTF-8 encoded text into segments based on a list of regular expressions, handling Unicode categories and collapsing codepoints as necessary.
- **Inputs**:
    - `text`: A UTF-8 encoded string that needs to be split based on the provided regular expressions.
    - `regex_exprs`: A vector of regular expressions that define the splitting criteria, potentially including Unicode category patterns.
- **Control Flow**:
    - Initialize static maps for Unicode categories and their corresponding codepoints and ranges.
    - Determine if any of the regular expressions require collapsing Unicode categories into single-byte representations.
    - Convert the input text into a vector of Unicode codepoints.
    - If collapsing is needed, create a collapsed version of the text where each codepoint is replaced by a single byte based on its category.
    - Initialize a vector to store offsets for byte pair encoding (BPE) splits, starting with the size of the codepoints vector.
    - Iterate over each regular expression in `regex_exprs`.
    - For each regex, attempt to use a custom regex splitting function; if successful, update the BPE offsets.
    - If custom splitting is not possible, check if the regex uses Unicode categories and decide whether to use the collapsed text or the original text for splitting.
    - If using the collapsed text, ensure the regex does not contain non-ASCII characters and create a collapsed version of the regex.
    - Use either `std::regex` or `std::wregex` to perform the splitting based on the chosen text and regex representation.
    - Catch and handle any regex errors, throwing a runtime error if regex processing fails.
    - Convert the BPE offsets into segments of the original text, converting each segment back to UTF-8.
    - Return the processed segments after applying byte encoding.
- **Output**: A vector of strings, each representing a segment of the original text split according to the provided regular expressions.
- **Functions called**:
    - [`unicode_cpts_from_utf8`](#unicode_cpts_from_utf8)
    - [`unicode_cpt_flags_from_cpt`](#unicode_cpt_flags_from_cpt)
    - [`unicode_regex_split_custom`](#unicode_regex_split_custom)
    - [`unicode_regex_split_stl`](#unicode_regex_split_stl)
    - [`unicode_wstring_from_utf8`](#unicode_wstring_from_utf8)
    - [`unicode_cpt_to_utf8`](#unicode_cpt_to_utf8)
    - [`unicode_byte_encoding_process`](#unicode_byte_encoding_process)


