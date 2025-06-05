# Purpose
This C++ source code file is designed to handle JSON parsing with a focus on error detection and recovery. It utilizes the nlohmann::json library to parse JSON data and implements a custom SAX (Simple API for XML) interface to track parsing errors and manage a stack of JSON elements. The primary functionality of this code is encapsulated in the [`common_json_parse`](#common_json_parse) function, which attempts to parse a JSON string and, in the event of an error, employs a "healing" mechanism to attempt to recover from incomplete or malformed JSON data. This is achieved by analyzing the parsing stack and strategically inserting a "healing marker" to close any open JSON structures, allowing the parser to continue processing the input.

The code is structured around a few key components: an enumeration `common_json_stack_element_type` to represent different JSON element types, a struct `common_json_stack_element` to store the type and key of JSON elements, and a [`json_error_locator`](#common_json_parse::json_error_locator::json_error_locator) class that extends the SAX interface to handle parsing events and errors. The [`json_error_locator`](#common_json_parse::json_error_locator::json_error_locator) class is responsible for maintaining the parsing state, detecting errors, and managing the stack of JSON elements. The code is intended to be part of a larger system, likely as a utility for robust JSON parsing, and does not define a public API or external interface directly. Instead, it provides a specialized function for internal use, focusing on enhancing the resilience of JSON parsing operations.
# Imports and Dependencies

---
- `json-partial.h`
- `log.h`
- `nlohmann/json.hpp`
- `string`


# Data Structures

---
### common\_json\_stack\_element\_type<!-- {{#data_structure:common_json_stack_element_type}} -->
- **Type**: `enum`
- **Members**:
    - `COMMON_JSON_STACK_ELEMENT_OBJECT`: Represents a JSON object in the stack.
    - `COMMON_JSON_STACK_ELEMENT_KEY`: Represents a key within a JSON object in the stack.
    - `COMMON_JSON_STACK_ELEMENT_ARRAY`: Represents a JSON array in the stack.
- **Description**: The `common_json_stack_element_type` is an enumeration that defines the types of elements that can be present in a JSON parsing stack. It includes three possible values: `COMMON_JSON_STACK_ELEMENT_OBJECT` for JSON objects, `COMMON_JSON_STACK_ELEMENT_KEY` for keys within JSON objects, and `COMMON_JSON_STACK_ELEMENT_ARRAY` for JSON arrays. This enum is used to track the structure of JSON data during parsing, helping to manage the state of the parser as it processes different parts of a JSON document.


---
### common\_json\_stack\_element<!-- {{#data_structure:common_json_stack_element}} -->
- **Type**: `struct`
- **Members**:
    - `type`: Specifies the type of the JSON stack element, using the enum common_json_stack_element_type.
    - `key`: Holds the key associated with the JSON stack element, represented as a string.
- **Description**: The `common_json_stack_element` struct is used to represent an element in a JSON parsing stack, where each element can be of a specific type (object, key, or array) and may have an associated key. This struct is part of a mechanism to track the structure of JSON data during parsing, allowing for error handling and recovery in the event of malformed JSON input.


---
### json\_error\_locator<!-- {{#data_structure:common_json_parse::json_error_locator}} -->
- **Type**: `struct`
- **Members**:
    - `position`: Stores the position in the JSON input where an error was detected.
    - `found_error`: Indicates whether an error was found during JSON parsing.
    - `last_token`: Holds the last token processed before an error occurred.
    - `exception_message`: Contains the exception message associated with the parsing error.
    - `stack`: A vector that tracks the stack of JSON elements being processed, such as objects and arrays.
- **Description**: The `json_error_locator` struct is a specialized SAX (Simple API for XML) handler for JSON parsing, designed to detect and report errors during the parsing process. It extends the `nlohmann::json_sax` interface and provides mechanisms to track the position of errors, the last token processed, and the exception message associated with the error. Additionally, it maintains a stack of JSON elements to assist in error recovery and reporting. This struct is particularly useful for applications that require detailed error handling and recovery during JSON parsing.
- **Member Functions**:
    - [`common_json_parse::json_error_locator::json_error_locator`](#common_json_parse::json_error_locator::json_error_locator)
    - [`common_json_parse::json_error_locator::parse_error`](#common_json_parse::json_error_locator::parse_error)
    - [`common_json_parse::json_error_locator::close_value`](#common_json_parse::json_error_locator::close_value)
    - [`common_json_parse::json_error_locator::null`](#common_json_parse::json_error_locator::null)
    - [`common_json_parse::json_error_locator::boolean`](#common_json_parse::json_error_locator::boolean)
    - [`common_json_parse::json_error_locator::number_integer`](#common_json_parse::json_error_locator::number_integer)
    - [`common_json_parse::json_error_locator::number_unsigned`](#common_json_parse::json_error_locator::number_unsigned)
    - [`common_json_parse::json_error_locator::number_float`](#common_json_parse::json_error_locator::number_float)
    - [`common_json_parse::json_error_locator::string`](#common_json_parse::json_error_locator::string)
    - [`common_json_parse::json_error_locator::binary`](#common_json_parse::json_error_locator::binary)
    - [`common_json_parse::json_error_locator::start_object`](#common_json_parse::json_error_locator::start_object)
    - [`common_json_parse::json_error_locator::end_object`](#common_json_parse::json_error_locator::end_object)
    - [`common_json_parse::json_error_locator::key`](#common_json_parse::json_error_locator::key)
    - [`common_json_parse::json_error_locator::start_array`](#common_json_parse::json_error_locator::start_array)
    - [`common_json_parse::json_error_locator::end_array`](#common_json_parse::json_error_locator::end_array)
- **Inherits From**:
    - `nlohmann::json_sax<json>`

**Methods**

---
#### json\_error\_locator::json\_error\_locator<!-- {{#callable:common_json_parse::json_error_locator::json_error_locator}} -->
The `json_error_locator` constructor initializes a JSON error locator object with default values for tracking parsing errors.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes the `position` member to 0, indicating the starting position for error tracking.
    - The `found_error` member is set to `false`, indicating no errors have been found initially.
- **Output**: An instance of the `json_error_locator` struct with initialized `position` and `found_error` members.
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::parse\_error<!-- {{#callable:common_json_parse::json_error_locator::parse_error}} -->
The `parse_error` function updates the state of a `json_error_locator` object to reflect a parsing error at a given position with a specific token and exception message.
- **Inputs**:
    - `position`: The position in the JSON input where the error occurred.
    - `last_token`: The last token encountered before the error.
    - `ex`: The exception object containing details about the parsing error.
- **Control Flow**:
    - The function sets the `position` member variable to one less than the provided position to adjust for zero-based indexing.
    - It sets the `found_error` member variable to `true` to indicate that an error was found.
    - The `last_token` member variable is updated with the provided last token.
    - The `exception_message` member variable is set to the message from the provided exception object.
    - The function returns `false` to indicate that parsing should not continue.
- **Output**: The function returns `false` to indicate a parsing error.
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::close\_value<!-- {{#callable:common_json_parse::json_error_locator::close_value}} -->
The `close_value` function checks if the last element in the stack is a key and removes it if true.
- **Inputs**: None
- **Control Flow**:
    - Check if the stack is not empty and the last element's type is `COMMON_JSON_STACK_ELEMENT_KEY`.
    - If both conditions are true, remove the last element from the stack.
- **Output**: This function does not return any value.
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::null<!-- {{#callable:common_json_parse::json_error_locator::null}} -->
The `null` function in the `json_error_locator` class handles the parsing of a JSON null value by closing the current value context and returning true.
- **Inputs**: None
- **Control Flow**:
    - The function calls `close_value()` to handle any necessary cleanup or state update related to the current JSON value context.
    - The function returns `true` to indicate successful handling of the null value.
- **Output**: The function returns a boolean value `true` to indicate successful processing of a JSON null value.
- **Functions called**:
    - [`common_json_parse::json_error_locator::close_value`](#common_json_parse::json_error_locator::close_value)
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::boolean<!-- {{#callable:common_json_parse::json_error_locator::boolean}} -->
The `boolean` method in the `json_error_locator` struct handles boolean values during JSON SAX parsing by closing any open JSON key context and returning true.
- **Inputs**:
    - `bool`: A boolean value that is passed to the method, though it is not used within the method body.
- **Control Flow**:
    - The method calls `close_value()` to handle any necessary cleanup of the JSON parsing stack, specifically removing any open key context.
    - The method then returns `true`, indicating successful handling of the boolean value.
- **Output**: The method returns a boolean value `true`, indicating successful processing of the boolean input.
- **Functions called**:
    - [`common_json_parse::json_error_locator::close_value`](#common_json_parse::json_error_locator::close_value)
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::number\_integer<!-- {{#callable:common_json_parse::json_error_locator::number_integer}} -->
The `number_integer` function handles the parsing of integer numbers in a JSON SAX parser by closing any open JSON key context and returning true.
- **Inputs**:
    - `number_integer_t`: A parameter representing an integer number, though it is not used within the function body.
- **Control Flow**:
    - The function calls `close_value()` to handle any open JSON key context.
    - The function returns `true` to indicate successful handling of the integer number.
- **Output**: The function returns a boolean value `true`.
- **Functions called**:
    - [`common_json_parse::json_error_locator::close_value`](#common_json_parse::json_error_locator::close_value)
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::number\_unsigned<!-- {{#callable:common_json_parse::json_error_locator::number_unsigned}} -->
The `number_unsigned` function processes an unsigned number in a JSON parsing context and manages the JSON stack state accordingly.
- **Inputs**:
    - `number_unsigned_t`: An unsigned integer type representing the unsigned number being processed.
- **Control Flow**:
    - The function calls `close_value()` to manage the JSON stack state.
    - The function returns `true` to indicate successful processing of the unsigned number.
- **Output**: A boolean value `true`, indicating that the unsigned number was processed successfully.
- **Functions called**:
    - [`common_json_parse::json_error_locator::close_value`](#common_json_parse::json_error_locator::close_value)
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::number\_float<!-- {{#callable:common_json_parse::json_error_locator::number_float}} -->
The `number_float` function handles the parsing of floating-point numbers in a JSON SAX parser by closing the current JSON value context and returning true.
- **Inputs**:
    - `number_float_t`: A floating-point number type representing the number being parsed.
    - `const string_t &`: A constant reference to a string representing the textual representation of the floating-point number.
- **Control Flow**:
    - Invoke the `close_value()` method to handle the closure of the current JSON value context.
    - Return `true` to indicate successful handling of the floating-point number.
- **Output**: The function returns a boolean value `true` to indicate successful processing of the floating-point number.
- **Functions called**:
    - [`common_json_parse::json_error_locator::close_value`](#common_json_parse::json_error_locator::close_value)
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::string<!-- {{#callable:common_json_parse::json_error_locator::string}} -->
The `string` method in the `json_error_locator` struct handles JSON string values by closing any open key-value context and returning true.
- **Inputs**:
    - `string_t &`: A reference to a string type, representing the JSON string value being processed.
- **Control Flow**:
    - The method calls `close_value()` to handle any open key-value context in the JSON stack.
    - The method returns `true` to indicate successful handling of the string value.
- **Output**: A boolean value `true`, indicating that the string value was successfully processed.
- **Functions called**:
    - [`common_json_parse::json_error_locator::close_value`](#common_json_parse::json_error_locator::close_value)
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::binary<!-- {{#callable:common_json_parse::json_error_locator::binary}} -->
The `binary` function processes a binary JSON value by closing the current JSON stack element if it is a key and always returns true.
- **Inputs**:
    - `binary_t &`: A reference to a binary_t object, representing the binary JSON value being processed.
- **Control Flow**:
    - The function calls `close_value()` to handle the current JSON stack element.
    - The function returns true, indicating successful processing of the binary value.
- **Output**: The function returns a boolean value, always true, indicating successful processing of the binary JSON value.
- **Functions called**:
    - [`common_json_parse::json_error_locator::close_value`](#common_json_parse::json_error_locator::close_value)
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::start\_object<!-- {{#callable:common_json_parse::json_error_locator::start_object}} -->
The `start_object` function adds a new object element to the JSON parsing stack to track the beginning of a JSON object.
- **Inputs**:
    - `std::size_t`: This parameter represents the number of elements in the JSON object being started, but it is not used in the function body.
- **Control Flow**:
    - The function pushes a new `common_json_stack_element` with type `COMMON_JSON_STACK_ELEMENT_OBJECT` and an empty key onto the `stack` vector.
    - The function returns `true` to indicate successful handling of the start of a JSON object.
- **Output**: The function returns a boolean value `true` to indicate that the start of a JSON object was successfully processed.
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::end\_object<!-- {{#callable:common_json_parse::json_error_locator::end_object}} -->
The `end_object` function finalizes the parsing of a JSON object by ensuring the stack is correctly managed and closed.
- **Inputs**: None
- **Control Flow**:
    - Assert that the stack is not empty and the last element is of type COMMON_JSON_STACK_ELEMENT_OBJECT.
    - Remove the last element from the stack.
    - Call the [`close_value`](#common_json_parse::json_error_locator::close_value) function to handle any necessary cleanup.
    - Return true to indicate successful completion.
- **Output**: A boolean value `true` indicating the successful end of a JSON object parsing.
- **Functions called**:
    - [`common_json_parse::json_error_locator::close_value`](#common_json_parse::json_error_locator::close_value)
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::key<!-- {{#callable:common_json_parse::json_error_locator::key}} -->
The `key` function pushes a key-value pair onto a stack with the key type and the provided key string.
- **Inputs**:
    - `key`: A reference to a string_t object representing the key to be pushed onto the stack.
- **Control Flow**:
    - The function takes a reference to a string_t object as input.
    - It pushes a new common_json_stack_element onto the stack with the type COMMON_JSON_STACK_ELEMENT_KEY and the provided key.
    - The function returns true, indicating successful execution.
- **Output**: The function returns a boolean value, always true, indicating successful execution.
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::start\_array<!-- {{#callable:common_json_parse::json_error_locator::start_array}} -->
The `start_array` function adds a new array element to the JSON parsing stack to track the beginning of an array structure.
- **Inputs**:
    - `std::size_t`: The size of the array, which is not used in the function body.
- **Control Flow**:
    - The function pushes a new `common_json_stack_element` onto the `stack` vector, indicating the start of an array with type `COMMON_JSON_STACK_ELEMENT_ARRAY` and an empty key.
    - The function returns `true` to indicate successful handling of the start of an array.
- **Output**: A boolean value `true` indicating the successful start of an array.
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)


---
#### json\_error\_locator::end\_array<!-- {{#callable:common_json_parse::json_error_locator::end_array}} -->
The `end_array` function finalizes the parsing of a JSON array by ensuring the stack is correctly managed and closing any open values.
- **Inputs**: None
- **Control Flow**:
    - Assert that the stack is not empty and the last element is of type COMMON_JSON_STACK_ELEMENT_ARRAY.
    - Remove the last element from the stack.
    - Call the close_value() function to handle any open values.
    - Return true to indicate successful completion.
- **Output**: A boolean value, always true, indicating the successful end of an array parsing.
- **Functions called**:
    - [`common_json_parse::json_error_locator::close_value`](#common_json_parse::json_error_locator::close_value)
- **See also**: [`common_json_parse::json_error_locator`](#common_json_parsejson_error_locator)  (Data Structure)



# Functions

---
### common\_json\_parse<!-- {{#callable:common_json_parse}} -->
The `common_json_parse` function attempts to parse a JSON string, handling errors by attempting to 'heal' incomplete JSON data using a specified marker.
- **Inputs**:
    - `it`: A reference to a string iterator pointing to the current position in the JSON string to be parsed.
    - `end`: A constant string iterator marking the end of the JSON string to be parsed.
    - `healing_marker`: A string used as a marker to attempt to 'heal' or complete the JSON string if parsing errors occur.
    - `out`: A reference to a `common_json` object where the parsed JSON data will be stored.
- **Control Flow**:
    - Initialize a `json_error_locator` struct to track parsing errors and JSON structure.
    - Use `json::sax_parse` to attempt parsing the JSON string from `it` to `end`, updating `err_loc` with any errors.
    - If a parsing error is found, reset `it` to the start position and attempt to parse up to the error position.
    - If parsing up to the error position fails, check if healing is possible using the `healing_marker` and the error stack.
    - If healing is possible, construct a valid JSON string by appending necessary closing brackets and the healing marker, then parse it.
    - If healing is successful, update `it` to the new position and return true.
    - If no errors are found, parse the JSON from `it` to `end` and store it in `out`, then update `it` to `end` and return true.
- **Output**: Returns a boolean indicating whether the JSON parsing (and potential healing) was successful.
- **Functions called**:
    - [`common_json_parse::json_error_locator::close_value`](#common_json_parse::json_error_locator::close_value)


