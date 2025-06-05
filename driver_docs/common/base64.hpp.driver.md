# Purpose
This C++ header file, `public_domain_base64.hpp`, provides a comprehensive implementation of Base64 encoding and decoding functionality. The file defines a `base64` class that offers methods to encode and decode data using the Base64 scheme, which is commonly used for encoding binary data into ASCII text. The class supports different Base64 alphabets, including the standard and URL/filename-safe variants, and provides options for handling padding and decoding behavior. The file includes template functions for encoding and decoding data from various input types, such as strings and character arrays, and supports both in-place and separate buffer operations. Additionally, it provides utility functions to calculate the required buffer sizes for encoding and decoding operations.

The file also defines a custom exception class, `base64_error`, which inherits from `std::runtime_error` to handle errors specific to Base64 operations. The `base64` class is designed to be flexible and efficient, allowing users to specify the desired alphabet and decoding behavior. The implementation is released into the public domain, as indicated by the Unlicense declaration at the top of the file, allowing unrestricted use, modification, and distribution. This header file is intended to be included in other C++ projects to provide Base64 functionality, making it a reusable component for developers needing to encode or decode data in this format.
# Imports and Dependencies

---
- `cstdint`
- `iterator`
- `stdexcept`
- `string`


# Data Structures

---
### base64\_error<!-- {{#data_structure:base64_error}} -->
- **Type**: `class`
- **Description**: The `base64_error` class is a custom exception type that inherits from `std::runtime_error`. It is designed to handle errors specifically related to Base64 encoding and decoding operations. By inheriting from `std::runtime_error`, it allows for the use of standard exception handling mechanisms in C++ to manage errors that occur during Base64 processing, providing a clear and specific error type for these operations.
- **Inherits From**:
    - `std::runtime_error`


---
### base64<!-- {{#data_structure:base64}} -->
- **Type**: `class`
- **Members**:
    - `alphabet`: An enum class that defines different base64 alphabets, including automatic detection, standard, and URL/filename safe variants.
    - `decoding_behavior`: An enum class that specifies the behavior during decoding, such as moderate or loose handling of padding.
- **Description**: The `base64` class provides functionality for encoding and decoding data using the Base64 encoding scheme. It supports different alphabets for encoding, including standard and URL/filename safe variants, and allows for different behaviors during decoding, such as handling padding characters. The class offers static methods for encoding and decoding strings and character arrays, both in-place and to new buffers, and includes utility functions to calculate the required buffer sizes for encoding and decoding operations.
- **Member Functions**:
    - [`base64::encode`](#base64encode)
    - [`base64::encode`](#base64encode)
    - [`base64::encode`](#base64encode)
    - [`base64::decode`](#base64decode)
    - [`base64::decode`](#base64decode)
    - [`base64::decode`](#base64decode)
    - [`base64::decode_inplace`](#base64decode_inplace)
    - [`base64::decode_inplace`](#base64decode_inplace)
    - [`base64::max_decode_size`](#base64max_decode_size)
    - [`base64::required_encode_size`](#base64required_encode_size)
    - [`base64::_base64_value`](#base64_base64_value)

**Methods**

---
#### base64::encode<!-- {{#callable:base64::encode}} -->
The `encode` function encodes a sequence of bytes into a Base64 string using a specified alphabet.
- **Inputs**:
    - `in_begin`: An iterator pointing to the beginning of the input sequence to be encoded.
    - `in_end`: An iterator pointing to the end of the input sequence to be encoded.
    - `out`: An output iterator where the encoded Base64 characters will be written.
    - `alphabet`: An optional parameter specifying which Base64 alphabet to use, either standard or URL/filename safe.
- **Control Flow**:
    - Initialize the padding character as '=' and select the appropriate Base64 alphabet based on the `alphabet` parameter.
    - Iterate over the input sequence in chunks of up to three bytes.
    - For each byte or partial byte, convert it to Base64 by mapping it to the corresponding character in the selected alphabet.
    - If fewer than three bytes remain, pad the output with '=' characters to complete the Base64 encoding block.
    - Write the encoded characters to the output iterator.
    - Return the output iterator pointing to the position after the last written character.
- **Output**: The function returns an output iterator pointing to the position just past the last encoded character.
- **See also**: [`base64`](#base64)  (Data Structure)


---
#### base64::encode<!-- {{#callable:base64::encode}} -->
The [`encode`](#base64encode) function encodes a given string into a base64 format using a specified alphabet.
- **Inputs**:
    - `str`: The string that should be encoded.
    - `alphabet`: The base64 alphabet to use for encoding, defaulting to `alphabet::standard`.
- **Control Flow**:
    - Initialize an empty string `result` to store the encoded output.
    - Reserve space in `result` for the encoded data using [`required_encode_size`](#base64required_encode_size) based on the input string's length.
    - Call the [`encode`](#base64encode) function template with the input string's iterators, a back inserter for `result`, and the specified alphabet.
    - Return the encoded `result` string.
- **Output**: The function returns the encoded base64 string.
- **Functions called**:
    - [`base64::required_encode_size`](#base64required_encode_size)
    - [`base64::encode`](#base64encode)
- **See also**: [`base64`](#base64)  (Data Structure)


---
#### base64::encode<!-- {{#callable:base64::encode}} -->
The [`encode`](#base64encode) function encodes a given character buffer into a Base64 string using a specified alphabet.
- **Inputs**:
    - `buffer`: A pointer to the character array that needs to be encoded.
    - `size`: The size of the character array to be encoded.
    - `alphabet`: An optional parameter specifying which Base64 alphabet to use, defaulting to `alphabet::standard`.
- **Control Flow**:
    - Initialize an empty string `result` to store the encoded output.
    - Reserve space in `result` for the encoded data using `required_encode_size(size) + 1` to ensure sufficient capacity.
    - Call the overloaded [`encode`](#base64encode) function with the buffer range and a back inserter for `result`, using the specified alphabet.
    - Return the encoded string `result`.
- **Output**: A `std::string` containing the Base64 encoded representation of the input buffer.
- **Functions called**:
    - [`base64::required_encode_size`](#base64required_encode_size)
    - [`base64::encode`](#base64encode)
- **See also**: [`base64`](#base64)  (Data Structure)


---
#### base64::decode<!-- {{#callable:base64::decode}} -->
The `decode` function decodes a range of base64-encoded input elements into a specified output iterator, handling different alphabets and decoding behaviors.
- **Inputs**:
    - `in_begin`: An iterator pointing to the beginning of the base64-encoded input range.
    - `in_end`: An iterator pointing to the end of the base64-encoded input range.
    - `out`: An output iterator where the decoded bytes will be written.
    - `alphabet`: Specifies which base64 alphabet to use for decoding, with a default of `alphabet::auto_`.
    - `behavior`: Specifies the behavior when encountering errors, with a default of `decoding_behavior::moderate`.
- **Control Flow**:
    - Initialize `last` to 0 and `bits` to 0 to keep track of the last processed value and the number of bits processed.
    - Iterate over the input range from `in_begin` to `in_end`.
    - For each character, check if it is a padding character ('='); if so, break the loop.
    - Convert the character to its base64 value using [`_base64_value`](#base64_base64_value) and store it in `part`.
    - If there are enough bits to form a byte (i.e., `bits + 6 >= 8`), calculate the byte value, write it to the output iterator, and adjust `bits`.
    - If not enough bits are available, add 6 to `bits` and update `last` with `part`.
    - After processing all input characters, if the behavior is not `decoding_behavior::loose`, check for any remaining non-padding characters and throw an error if found.
    - Return the output iterator pointing to the next position after the last written byte.
- **Output**: The function returns an output iterator pointing to the next position after the last decoded byte.
- **Functions called**:
    - [`base64::_base64_value`](#base64_base64_value)
    - [`base64_error`](#base64_error)
- **See also**: [`base64`](#base64)  (Data Structure)


---
#### base64::decode<!-- {{#callable:base64::decode}} -->
The [`decode`](#base64decode) function decodes a base64 encoded string into its original form using a specified alphabet and decoding behavior.
- **Inputs**:
    - `str`: The base64 encoded string to be decoded.
    - `alphabet`: The alphabet to be used for decoding, with a default value of `alphabet::auto_`.
    - `behavior`: The behavior to follow when an error is detected during decoding, with a default value of `decoding_behavior::moderate`.
- **Control Flow**:
    - A `std::string` named `result` is initialized to store the decoded output.
    - The `result` string is reserved with a size calculated by [`max_decode_size`](#base64max_decode_size) based on the length of the input string `str`.
    - The [`decode`](#base64decode) function template is called with the input string's iterators, a back inserter for `result`, and the specified `alphabet` and `behavior`.
    - The decoded result is returned as a `std::string`.
- **Output**: A `std::string` containing the decoded version of the input base64 encoded string.
- **Functions called**:
    - [`base64::max_decode_size`](#base64max_decode_size)
    - [`base64::decode`](#base64decode)
- **See also**: [`base64`](#base64)  (Data Structure)


---
#### base64::decode<!-- {{#callable:base64::decode}} -->
The [`decode`](#base64decode) function decodes a base64 encoded character buffer into a standard string using a specified alphabet and decoding behavior.
- **Inputs**:
    - `buffer`: A pointer to the base64 encoded character array to be decoded.
    - `size`: The size of the character array to be decoded.
    - `alphabet`: The base64 alphabet to use for decoding, with a default value of `alphabet::auto_`.
    - `behavior`: The behavior to follow when an error is detected during decoding, with a default value of `decoding_behavior::moderate`.
- **Control Flow**:
    - Initialize an empty `std::string` named `result` to store the decoded output.
    - Reserve space in `result` using `max_decode_size(size)` to ensure it can hold the decoded data.
    - Call the [`decode`](#base64decode) function template with the buffer range, a back inserter for `result`, and the specified `alphabet` and `behavior`.
    - Return the `result` string containing the decoded data.
- **Output**: A `std::string` containing the decoded data from the input buffer.
- **Functions called**:
    - [`base64::max_decode_size`](#base64max_decode_size)
    - [`base64::decode`](#base64decode)
- **See also**: [`base64`](#base64)  (Data Structure)


---
#### base64::decode\_inplace<!-- {{#callable:base64::decode_inplace}} -->
The `decode_inplace` function decodes a base64-encoded string in place, modifying the original string to contain the decoded data.
- **Inputs**:
    - `str`: A reference to the base64-encoded string that will be decoded in place.
    - `alphabet`: An optional parameter specifying which base64 alphabet to use; defaults to `alphabet::auto_`.
    - `behavior`: An optional parameter specifying the behavior when an error is detected during decoding; defaults to `decoding_behavior::moderate`.
- **Control Flow**:
    - The function calls the [`decode`](#base64decode) method, passing the beginning and end iterators of the string, the beginning iterator as the output location, and the specified alphabet and behavior.
    - The [`decode`](#base64decode) method processes the input string and writes the decoded data back to the beginning of the string.
    - The function then resizes the string to the length of the decoded data by subtracting the beginning iterator from the returned iterator of the [`decode`](#base64decode) method.
- **Output**: The function does not return a value; it modifies the input string in place to contain the decoded data.
- **Functions called**:
    - [`base64::decode`](#base64decode)
- **See also**: [`base64`](#base64)  (Data Structure)


---
#### base64::decode\_inplace<!-- {{#callable:base64::decode_inplace}} -->
The `decode_inplace` function decodes a base64 encoded character array in place, modifying the original array.
- **Inputs**:
    - `str`: A pointer to the character array that contains the base64 encoded data to be decoded.
    - `size`: The size of the character array to be decoded.
    - `alphabet`: An optional parameter specifying which base64 alphabet to use; defaults to `alphabet::auto_`.
    - `behavior`: An optional parameter specifying the behavior when an error is detected during decoding; defaults to `decoding_behavior::moderate`.
- **Control Flow**:
    - The function calls the [`decode`](#base64decode) function, passing the start and end of the character array, the same array as the output, and the specified alphabet and behavior.
    - The [`decode`](#base64decode) function performs the decoding operation in place, modifying the original character array.
- **Output**: Returns a pointer to the next element past the last element decoded in the character array.
- **Functions called**:
    - [`base64::decode`](#base64decode)
- **See also**: [`base64`](#base64)  (Data Structure)


---
#### base64::max\_decode\_size<!-- {{#callable:base64::max_decode_size}} -->
The `max_decode_size` function calculates the maximum possible size of a decoded buffer from a given encoded input size.
- **Inputs**:
    - `size`: The size of the encoded input, which is a `std::size_t` value representing the number of characters in the encoded data.
- **Control Flow**:
    - The function divides the input size by 4 to determine the number of complete 4-character base64 blocks.
    - It checks if there is a remainder when dividing the size by 4, and if so, it adds 1 to account for an incomplete block.
    - The result is then multiplied by 3 to convert the number of base64 blocks to the maximum number of bytes they can decode into.
- **Output**: The function returns a `std::size_t` value representing the maximum size of the decoded buffer.
- **See also**: [`base64`](#base64)  (Data Structure)


---
#### base64::required\_encode\_size<!-- {{#callable:base64::required_encode_size}} -->
The `required_encode_size` function calculates the necessary buffer size for encoding a given number of bytes into Base64 format.
- **Inputs**:
    - `size`: The size of the input data in bytes that needs to be encoded.
- **Control Flow**:
    - The function divides the input size by 3 to determine how many full 3-byte groups are present.
    - It checks if there is a remainder when dividing the size by 3, which indicates additional bytes that need padding.
    - The function adds 1 to the quotient if there is a remainder, ensuring that any partial group is accounted for.
    - The result is multiplied by 4 to calculate the total number of Base64 characters needed, as each 3-byte group is encoded into 4 Base64 characters.
- **Output**: The function returns the calculated size as a `std::size_t`, representing the number of characters required to encode the input data in Base64 format.
- **See also**: [`base64`](#base64)  (Data Structure)


---
#### base64::\_base64\_value<!-- {{#callable:base64::_base64_value}} -->
The `_base64_value` function converts a base64 character to its corresponding integer value based on the specified alphabet, and throws an error if the character is invalid.
- **Inputs**:
    - `alphabet`: A reference to an `alphabet` enum indicating the base64 alphabet type to use (standard, url_filename_safe, or auto_).
    - `c`: A character representing a base64 encoded value to be converted to its integer equivalent.
- **Control Flow**:
    - Check if the character `c` is an uppercase letter ('A'-'Z') and return its zero-based index.
    - Check if the character `c` is a lowercase letter ('a'-'z') and return its index offset by 26.
    - Check if the character `c` is a digit ('0'-'9') and return its index offset by 52.
    - If the alphabet is `standard`, check if `c` is '+' or '/' and return 62 or 63 respectively.
    - If the alphabet is `url_filename_safe`, check if `c` is '-' or '_' and return 62 or 63 respectively.
    - If the alphabet is `auto_`, determine the alphabet based on the character and return the corresponding value.
    - If none of the conditions are met, throw a [`base64_error`](#base64_error) indicating an invalid base64 character.
- **Output**: Returns a `std::uint8_t` representing the integer value of the base64 character `c`.
- **Functions called**:
    - [`base64_error`](#base64_error)
- **See also**: [`base64`](#base64)  (Data Structure)



---
### alphabet<!-- {{#data_structure:base64::alphabet}} -->
- **Type**: `enum class`
- **Members**:
    - `auto_`: The alphabet is detected automatically.
    - `standard`: The standard base64 alphabet is used.
    - `url_filename_safe`: Like `standard` except that the characters `+` and `/` are replaced by `-` and `_` respectively.
- **Description**: The `alphabet` enum class defines different types of base64 alphabets that can be used for encoding and decoding operations. It includes three options: `auto_`, which automatically detects the alphabet; `standard`, which uses the standard base64 alphabet; and `url_filename_safe`, which modifies the standard alphabet to be safe for URLs and filenames by replacing `+` and `/` with `-` and `_`.


---
### decoding\_behavior<!-- {{#data_structure:base64::decoding_behavior}} -->
- **Type**: `enum class`
- **Members**:
    - `moderate`: If the input is not padded, the remaining bits are ignored.
    - `loose`: If a padding character is encountered, decoding is finished.
- **Description**: The `decoding_behavior` enum class defines two modes of behavior for handling base64 decoding errors related to padding. The `moderate` mode allows for ignoring remaining bits if the input is not padded, while the `loose` mode stops decoding upon encountering a padding character. This enum is used to specify how strictly the base64 decoding process should adhere to expected input formats.


