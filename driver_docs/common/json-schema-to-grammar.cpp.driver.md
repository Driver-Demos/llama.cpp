# Purpose
This C++ source code file is designed to convert JSON schemas into grammar rules, specifically for use in parsing or validating JSON data. The file includes a variety of components that work together to achieve this conversion. It utilizes the nlohmann::json library for handling JSON data and defines several static functions and classes to process and transform JSON schema elements into grammar rules. The key components include the [`SchemaConverter`](#SchemaConverterSchemaConverter) class, which is responsible for traversing and interpreting the JSON schema, resolving references, and generating grammar rules. The file also defines utility functions like [`build_repetition`](#build_repetition) and [`_build_min_max_int`](#_build_min_max_int) to handle specific schema constraints such as repetition and numeric ranges.

The file is not an executable on its own but rather a library intended to be used by other parts of a software system that require JSON schema validation or parsing capabilities. It defines a public API through functions like [`json_schema_to_grammar`](#json_schema_to_grammar) and [`build_grammar`](#build_grammar), which are used to initiate the conversion process. The code also includes mechanisms for handling errors and warnings during the conversion process, ensuring that any issues with the schema are reported. Overall, the file provides a focused functionality for converting JSON schemas into a grammar format, which can be used for various applications such as data validation, parsing, or code generation.
# Imports and Dependencies

---
- `json-schema-to-grammar.h`
- `common.h`
- `nlohmann/json.hpp`
- `algorithm`
- `map`
- `regex`
- `sstream`
- `string`
- `unordered_map`
- `unordered_set`
- `vector`


# Global Variables

---
### SPACE\_RULE
- **Type**: `std::string`
- **Description**: The `SPACE_RULE` is a constant string that defines a grammar rule for matching spaces in a text. It specifies that a space can be a single space character, one or two newline characters followed by up to 20 tab characters.
- **Use**: This variable is used to define a pattern for spaces in grammar rules, likely for parsing or formatting purposes.


---
### PRIMITIVE\_RULES
- **Type**: `std::unordered_map<std::string, BuiltinRule>`
- **Description**: The `PRIMITIVE_RULES` variable is a global unordered map that associates string keys with `BuiltinRule` objects. Each key represents a primitive type or structure, such as 'boolean', 'number', or 'object', and the corresponding `BuiltinRule` contains a string representation of a grammar rule and a list of dependencies for parsing that type.
- **Use**: This variable is used to define and store grammar rules for various primitive types, which can be utilized in parsing or validating JSON schemas.


---
### STRING\_FORMAT\_RULES
- **Type**: `std::unordered_map<std::string, BuiltinRule>`
- **Description**: `STRING_FORMAT_RULES` is a global variable that is an unordered map associating string keys with `BuiltinRule` values. Each key represents a specific string format, such as 'date', 'time', or 'date-time', and the corresponding `BuiltinRule` contains a regular expression pattern for validating that format and a list of dependencies on other rules.
- **Use**: This variable is used to define and store the rules for validating various string formats in a JSON schema to grammar conversion process.


---
### INVALID\_RULE\_CHARS\_RE
- **Type**: `std::regex`
- **Description**: The `INVALID_RULE_CHARS_RE` is a global variable of type `std::regex` that defines a regular expression pattern. This pattern matches any character that is not an alphanumeric character (a-z, A-Z, 0-9) or a hyphen (-).
- **Use**: It is used to identify and potentially replace invalid characters in rule names within the code.


---
### GRAMMAR\_LITERAL\_ESCAPE\_RE
- **Type**: `std::regex`
- **Description**: The `GRAMMAR_LITERAL_ESCAPE_RE` is a global variable of type `std::regex` that is initialized with a regular expression pattern. This pattern is designed to match any of the characters carriage return (`\r`), newline (`\n`), or double quote (`"`).
- **Use**: This variable is used to identify and potentially escape these specific characters in strings when processing grammar literals.


---
### GRAMMAR\_RANGE\_LITERAL\_ESCAPE\_RE
- **Type**: `std::regex`
- **Description**: The `GRAMMAR_RANGE_LITERAL_ESCAPE_RE` is a regular expression object that is used to match specific characters that need to be escaped in a grammar range literal. These characters include carriage return (`\r`), newline (`\n`), double quote (`"`), closing square bracket (`]`), hyphen (`-`), and backslash (`\\`).
- **Use**: This variable is used to identify characters in a string that need to be escaped when constructing grammar range literals.


---
### GRAMMAR\_LITERAL\_ESCAPES
- **Type**: `std::unordered_map<char, std::string>`
- **Description**: The `GRAMMAR_LITERAL_ESCAPES` variable is a global unordered map that associates certain special characters with their corresponding escape sequences as strings. It is used to handle characters that need to be escaped in grammar literals, such as carriage return, newline, double quotes, hyphen, and closing square bracket.
- **Use**: This variable is used to replace special characters in grammar literals with their escape sequences to ensure proper parsing and representation.


---
### NON\_LITERAL\_SET
- **Type**: `std::unordered_set<char>`
- **Description**: The `NON_LITERAL_SET` is a global variable defined as an unordered set of characters. It contains a collection of special characters that are typically used in regular expressions to denote non-literal operations, such as alternation, grouping, and quantification.
- **Use**: This variable is used to identify characters that are treated as non-literal in regular expressions, likely for parsing or processing purposes.


---
### ESCAPED\_IN\_REGEXPS\_BUT\_NOT\_IN\_LITERALS
- **Type**: `std::unordered_set<char>`
- **Description**: The variable `ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS` is a global unordered set of characters that are typically escaped in regular expressions but not in string literals. It includes characters such as '^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', and '?'. These characters have special meanings in regular expressions and are often escaped to be used as literal characters.
- **Use**: This variable is used to identify characters that need to be escaped in regular expressions but not in string literals, aiding in the correct processing of patterns.


# Data Structures

---
### string\_view<!-- {{#data_structure:string_view}} -->
- **Type**: `class`
- **Members**:
    - `_str`: A constant reference to a std::string that the string_view is viewing.
    - `_start`: A constant size_t indicating the starting index of the view within the string.
    - `_end`: A constant size_t indicating the ending index of the view within the string.
- **Description**: The `string_view` class is a minimalistic replacement for `std::string_view`, which is only available from C++17 onwards. It provides a non-owning view over a portion of a `std::string`, defined by a start and end index. This class allows for efficient string manipulation and access without copying the string data. It includes methods for obtaining the size of the view, converting the view to a `std::string`, and accessing substrings and individual characters. The class also supports comparison operations with other `string_view` instances.
- **Member Functions**:
    - [`string_view::string_view`](#string_viewstring_view)
    - [`string_view::size`](#string_viewsize)
    - [`string_view::length`](#string_viewlength)
    - [`string_view::str`](#string_viewstr)
    - [`string_view::substr`](#string_viewsubstr)
    - [`string_view::operator[]`](llama.cpp/common/json-schema-to-grammar.cpp#callable:string_view::operator[])
    - [`string_view::operator==`](#string_viewoperator==)

**Methods**

---
#### string\_view::string\_view<!-- {{#callable:string_view::string_view}} -->
The `string_view` constructor initializes a `string_view` object to represent a substring of a given `std::string` from a specified start position to an end position.
- **Inputs**:
    - `str`: A constant reference to a `std::string` from which the `string_view` will be created.
    - `start`: An optional size_t parameter specifying the starting index of the substring within `str`; defaults to 0.
    - `end`: An optional size_t parameter specifying the ending index of the substring within `str`; defaults to `std::string::npos`, which means the end of the string.
- **Control Flow**:
    - The constructor initializes the `_str` member with the provided `str` reference.
    - The `_start` member is initialized with the `start` parameter.
    - The `_end` member is initialized with the `end` parameter if it is not `std::string::npos`; otherwise, it is set to the length of `str`.
- **Output**: A `string_view` object representing the specified substring of the input `std::string`.
- **See also**: [`string_view`](#string_view)  (Data Structure)


---
#### string\_view::size<!-- {{#callable:string_view::size}} -->
The `size` function calculates the length of the substring represented by the `string_view` object.
- **Inputs**: None
- **Control Flow**:
    - The function calculates the difference between the `_end` and `_start` member variables of the `string_view` class.
- **Output**: The function returns a `size_t` value representing the length of the substring.
- **See also**: [`string_view`](#string_view)  (Data Structure)


---
#### string\_view::length<!-- {{#callable:string_view::length}} -->
The `length` function returns the length of the string view by calling the [`size`](../vendor/nlohmann/json.hpp.driver.md#size) method.
- **Inputs**: None
- **Control Flow**:
    - The function calls the [`size`](../vendor/nlohmann/json.hpp.driver.md#size) method of the `string_view` class.
    - The [`size`](../vendor/nlohmann/json.hpp.driver.md#size) method calculates the length by subtracting `_start` from `_end`.
    - The result from [`size`](../vendor/nlohmann/json.hpp.driver.md#size) is returned as the output of `length`.
- **Output**: The function returns a `size_t` representing the length of the string view.
- **Functions called**:
    - [`size`](../vendor/nlohmann/json.hpp.driver.md#size)
- **See also**: [`string_view`](#string_view)  (Data Structure)


---
#### string\_view::str<!-- {{#callable:string_view::str}} -->
The `str` method returns a substring of the original string referenced by the `string_view` object, defined by the `_start` and `_end` indices.
- **Inputs**: None
- **Control Flow**:
    - The method calls the `substr` function on the `_str` member, which is a reference to the original string.
    - It passes `_start` as the starting position and `_end - _start` as the length of the substring to be extracted.
- **Output**: A `std::string` object representing the substring from the original string, starting at `_start` and ending at `_end`.
- **See also**: [`string_view`](#string_view)  (Data Structure)


---
#### string\_view::substr<!-- {{#callable:string_view::substr}} -->
The `substr` method returns a substring view of the original string view starting from a specified position and extending for a specified length.
- **Inputs**:
    - `pos`: The starting position within the string view from which the substring view should begin.
    - `len`: The length of the substring view to be returned; defaults to the maximum possible length if not specified.
- **Control Flow**:
    - The method calculates the starting position of the substring view by adding the `pos` to the `_start` of the current string view.
    - It determines the ending position of the substring view by checking if `len` is `std::string::npos`; if so, it uses `_end` as the ending position, otherwise it calculates the end by adding `len` to the starting position.
    - A new [`string_view`](#string_viewstring_view) object is created and returned using the original string, the calculated starting position, and the determined ending position.
- **Output**: A [`string_view`](#string_viewstring_view) object representing the specified substring view of the original string view.
- **Functions called**:
    - [`string_view::string_view`](#string_viewstring_view)
- **See also**: [`string_view`](#string_view)  (Data Structure)


---
#### string\_view::operator\[\]<!-- {{#callable:string_view::operator[]}} -->
The `operator[]` function provides access to a character at a specified position within a `string_view` object, throwing an exception if the position is out of range.
- **Inputs**:
    - `pos`: The position within the `string_view` from which to retrieve the character.
- **Control Flow**:
    - Calculate the index by adding the `_start` position to the `pos` argument.
    - Check if the calculated index is greater than or equal to `_end`; if so, throw a `std::out_of_range` exception.
    - Return the character at the calculated index from the underlying string `_str`.
- **Output**: A character from the underlying string at the specified position within the `string_view`.
- **See also**: [`string_view`](#string_view)  (Data Structure)


---
#### string\_view::operator==<!-- {{#callable:string_view::operator==}} -->
The `operator==` function compares two `string_view` objects for equality by converting them to `std::string` and using the `std::string` equality operator.
- **Inputs**:
    - `other`: A reference to another `string_view` object to compare with the current object.
- **Control Flow**:
    - Convert the current `string_view` object to a `std::string` using the conversion operator.
    - Convert the `other` `string_view` object to a `std::string`.
    - Compare the two `std::string` objects using the `==` operator.
    - Return the result of the comparison.
- **Output**: A boolean value indicating whether the two `string_view` objects are equal.
- **See also**: [`string_view`](#string_view)  (Data Structure)



---
### BuiltinRule<!-- {{#data_structure:BuiltinRule}} -->
- **Type**: `struct`
- **Members**:
    - `content`: A string representing the content of the rule.
    - `deps`: A vector of strings representing dependencies of the rule.
- **Description**: The `BuiltinRule` struct is a simple data structure used to represent a rule with its associated content and dependencies. It contains two members: `content`, which is a string that holds the rule's content, and `deps`, which is a vector of strings that lists the dependencies required by the rule. This struct is likely used in the context of defining and managing rules within a larger system, possibly for parsing or interpreting structured data.


---
### SchemaConverter<!-- {{#data_structure:SchemaConverter}} -->
- **Type**: `class`
- **Members**:
    - `_fetch_json`: A function to fetch JSON data given a string input.
    - `_dotall`: A boolean flag indicating whether dotall mode is enabled.
    - `_rules`: A map storing grammar rules with their names as keys.
    - `_refs`: An unordered map storing JSON references.
    - `_refs_being_resolved`: An unordered set tracking references currently being resolved.
    - `_errors`: A vector storing error messages encountered during processing.
    - `_warnings`: A vector storing warning messages encountered during processing.
- **Description**: The `SchemaConverter` class is designed to convert JSON schemas into grammar rules. It manages the conversion process by maintaining a set of rules, resolving references, and handling errors and warnings. The class uses a function to fetch JSON data and supports various schema constructs such as objects, arrays, and patterns. It also provides methods to add rules, resolve references, and format the resulting grammar.
- **Member Functions**:
    - [`SchemaConverter::_add_rule`](#SchemaConverter_add_rule)
    - [`SchemaConverter::_generate_union_rule`](#SchemaConverter_generate_union_rule)
    - [`SchemaConverter::_visit_pattern`](#SchemaConverter_visit_pattern)
    - [`SchemaConverter::_not_strings`](#SchemaConverter_not_strings)
    - [`SchemaConverter::_resolve_ref`](#SchemaConverter_resolve_ref)
    - [`SchemaConverter::_build_object_rule`](#SchemaConverter_build_object_rule)
    - [`SchemaConverter::_add_primitive`](#SchemaConverter_add_primitive)
    - [`SchemaConverter::SchemaConverter`](#SchemaConverterSchemaConverter)
    - [`SchemaConverter::resolve_refs`](#SchemaConverterresolve_refs)
    - [`SchemaConverter::_generate_constant_rule`](#SchemaConverter_generate_constant_rule)
    - [`SchemaConverter::visit`](#SchemaConvertervisit)
    - [`SchemaConverter::check_errors`](#SchemaConvertercheck_errors)
    - [`SchemaConverter::format_grammar`](#SchemaConverterformat_grammar)

**Methods**

---
#### SchemaConverter::\_add\_rule<!-- {{#callable:SchemaConverter::_add_rule}} -->
The `_add_rule` function adds a rule to the `_rules` map, ensuring unique keys by appending an index if necessary.
- **Inputs**:
    - `name`: A string representing the name of the rule to be added.
    - `rule`: A string representing the rule content to be associated with the name.
- **Control Flow**:
    - The function first replaces invalid characters in the `name` with hyphens to create `esc_name`.
    - It checks if `esc_name` is not already in `_rules` or if the existing rule matches the new `rule`.
    - If either condition is true, it adds or updates the rule in `_rules` with `esc_name` as the key and returns `esc_name`.
    - If the rule already exists with a different value, it enters a loop to find a unique key by appending an index to `esc_name`.
    - The loop increments the index until a unique key is found or a matching rule is found.
    - Once a unique key is found, it adds the rule to `_rules` with this key and returns the key.
- **Output**: A string representing the key under which the rule was added to the `_rules` map.
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)


---
#### SchemaConverter::\_generate\_union\_rule<!-- {{#callable:SchemaConverter::_generate_union_rule}} -->
The `_generate_union_rule` function generates a union rule string by visiting each alternative schema and joining the resulting rules with a '|' separator.
- **Inputs**:
    - `name`: A string representing the base name to be used for generating rule names for each alternative schema.
    - `alt_schemas`: A vector of JSON objects representing alternative schemas to be processed into rules.
- **Control Flow**:
    - Initialize an empty vector `rules` to store the generated rules for each alternative schema.
    - Iterate over each schema in `alt_schemas` using an index `i`.
    - For each schema, call the [`visit`](#SchemaConvertervisit) function with the schema and a generated name based on `name` and `i`, then add the result to `rules`.
    - Join all the rules in `rules` with a '|' separator and return the resulting string.
- **Output**: A string representing the union of rules generated from the alternative schemas, separated by '|'.
- **Functions called**:
    - [`SchemaConverter::visit`](#SchemaConvertervisit)
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)


---
#### SchemaConverter::\_visit\_pattern<!-- {{#callable:SchemaConverter::_visit_pattern}} -->
The `_visit_pattern` function processes a regex pattern string, validates its format, and converts it into a grammar rule for JSON schema conversion.
- **Inputs**:
    - `pattern`: A string representing a regex pattern that must start with '^' and end with '$'.
    - `name`: A string representing the name to be used for the generated rule.
- **Control Flow**:
    - Check if the pattern starts with '^' and ends with '$'; if not, log an error and return an empty string.
    - Extract the sub-pattern by removing the leading '^' and trailing '$'.
    - Initialize variables for tracking position and length of the sub-pattern.
    - Define a lambda `to_rule` to convert a literal or rule pair into a string representation.
    - Define a recursive lambda `transform` to process the sub-pattern and convert it into a sequence of literals and rules.
    - Within `transform`, iterate over the sub-pattern characters, handling special regex characters like '.', '(', ')', '[', ']', '|', '*', '+', '?', and '{' to build the sequence.
    - For each special character, perform specific actions such as adding rules, handling repetitions, and managing nested patterns.
    - Use helper functions like `get_dot`, `join_seq`, and [`build_repetition`](#build_repetition) to manage specific pattern transformations.
    - Return the final rule by adding it to the rules map using [`_add_rule`](#SchemaConverter_add_rule).
- **Output**: A string representing the name of the rule added to the rules map, or an empty string if the pattern is invalid.
- **Functions called**:
    - [`SchemaConverter::_add_rule`](#SchemaConverter_add_rule)
    - [`build_repetition`](#build_repetition)
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)


---
#### SchemaConverter::\_not\_strings<!-- {{#callable:SchemaConverter::_not_strings}} -->
The function `_not_strings` generates a grammar rule that matches a JSON string which is none of the provided strings.
- **Inputs**:
    - `strings`: A vector of strings that the generated grammar rule should not match.
- **Control Flow**:
    - Define a `TrieNode` structure to represent a trie with children nodes and an end-of-string flag.
    - Insert each string from the input vector into the trie to build a prefix tree.
    - Retrieve the character rule for 'char' using [`_add_primitive`](#SchemaConverter_add_primitive).
    - Initialize an output stream to construct the grammar rule.
    - Define a recursive lambda function [`visit`](#SchemaConvertervisit) to traverse the trie and build the grammar rule.
    - For each node in the trie, append character options to the output stream, handling branches and end-of-string conditions.
    - If a node has children, append a negated character class to the output stream to exclude those characters.
    - Invoke the [`visit`](#SchemaConvertervisit) function starting from the root of the trie.
    - Finalize the grammar rule by appending closing syntax and return the constructed string.
- **Output**: A string representing a grammar rule that matches a JSON string not equal to any of the input strings.
- **Functions called**:
    - [`SchemaConverter::_add_primitive`](#SchemaConverter_add_primitive)
    - [`SchemaConverter::visit`](#SchemaConvertervisit)
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)


---
#### SchemaConverter::\_resolve\_ref<!-- {{#callable:SchemaConverter::_resolve_ref}} -->
The `_resolve_ref` function resolves a reference string to a rule name by checking if it is already resolved or being resolved, and if not, it processes the reference to generate a rule name.
- **Inputs**:
    - `ref`: A string representing the reference to be resolved, typically in the form of a URL or a path.
- **Control Flow**:
    - Extract the reference name from the input `ref` by taking the substring after the last '/' character.
    - Check if the reference name is not already in the `_rules` map and the full reference is not in the `_refs_being_resolved` set.
    - If the reference is not already resolved or being resolved, insert the reference into `_refs_being_resolved` to mark it as being processed.
    - Retrieve the JSON object associated with the reference from the `_refs` map.
    - Call the [`visit`](#SchemaConvertervisit) function with the retrieved JSON object and the reference name to process the reference and generate a rule name.
    - Remove the reference from `_refs_being_resolved` after processing.
    - Return the resolved reference name.
- **Output**: A string representing the resolved reference name, which is used as a rule name.
- **Functions called**:
    - [`SchemaConverter::visit`](#SchemaConvertervisit)
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)


---
#### SchemaConverter::\_build\_object\_rule<!-- {{#callable:SchemaConverter::_build_object_rule}} -->
The function `_build_object_rule` constructs a grammar rule for a JSON object based on its properties, required fields, and additional properties.
- **Inputs**:
    - `properties`: A vector of pairs where each pair consists of a property name (string) and its corresponding JSON schema (json).
    - `required`: An unordered set of strings representing the names of required properties.
    - `name`: A string representing the base name for the rule being constructed.
    - `additional_properties`: A JSON object or boolean indicating additional properties allowed in the object.
- **Control Flow**:
    - Initialize vectors for required and optional properties, and a map for property key-value rule names.
    - Iterate over each property in `properties`, generating a rule name for each property schema using the [`visit`](#SchemaConvertervisit) function and adding it to `prop_kv_rule_names`.
    - Classify each property as required or optional based on the `required` set and add to respective vectors.
    - If `additional_properties` is allowed, generate rules for additional properties and add to `optional_props`.
    - Construct the initial part of the rule string for required properties, concatenating their rules with commas.
    - If there are optional properties, construct a recursive rule string to handle combinations of optional properties, using a helper function `get_recursive_refs`.
    - Finalize the rule string by appending the closing brace and return the complete rule.
- **Output**: A string representing the constructed grammar rule for the JSON object.
- **Functions called**:
    - [`SchemaConverter::visit`](#SchemaConvertervisit)
    - [`SchemaConverter::_add_rule`](#SchemaConverter_add_rule)
    - [`format_literal`](#format_literal)
    - [`SchemaConverter::_add_primitive`](#SchemaConverter_add_primitive)
    - [`SchemaConverter::_not_strings`](#SchemaConverter_not_strings)
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)


---
#### SchemaConverter::\_add\_primitive<!-- {{#callable:SchemaConverter::_add_primitive}} -->
The `_add_primitive` function adds a primitive rule to the internal rules map and recursively ensures all its dependencies are also added.
- **Inputs**:
    - `name`: A string representing the name of the primitive rule to be added.
    - `rule`: A `BuiltinRule` object containing the content and dependencies of the rule to be added.
- **Control Flow**:
    - Call [`_add_rule`](#SchemaConverter_add_rule) with the provided `name` and `rule.content` to add the rule and get the normalized rule name.
    - Iterate over each dependency in `rule.deps`.
    - For each dependency, check if it exists in `PRIMITIVE_RULES` or `STRING_FORMAT_RULES`.
    - If a dependency is not found in either map, add an error message to `_errors` and continue to the next dependency.
    - If a dependency is found and not already in `_rules`, recursively call `_add_primitive` to add the dependency.
- **Output**: Returns the normalized name of the rule that was added.
- **Functions called**:
    - [`SchemaConverter::_add_rule`](#SchemaConverter_add_rule)
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)


---
#### SchemaConverter::SchemaConverter<!-- {{#callable:SchemaConverter::SchemaConverter}} -->
The `SchemaConverter` constructor initializes a `SchemaConverter` object with a JSON fetching function and a boolean flag, and sets up initial grammar rules.
- **Inputs**:
    - `fetch_json`: A `std::function` that takes a `std::string` and returns a `json` object, used to fetch JSON data.
    - `dotall`: A boolean flag indicating whether the dot (.) in patterns should match all characters, including newlines.
- **Control Flow**:
    - The constructor initializes the `_fetch_json` member with the provided `fetch_json` function.
    - The constructor initializes the `_dotall` member with the provided `dotall` boolean value.
    - The constructor sets the initial rule for "space" in the `_rules` map to `SPACE_RULE`.
- **Output**: The constructor does not return any value; it initializes the `SchemaConverter` object.
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)


---
#### SchemaConverter::resolve\_refs<!-- {{#callable:SchemaConverter::resolve_refs}} -->
The `resolve_refs` function resolves all `$ref` fields in a JSON schema, fetching remote schemas if necessary, and replaces each `$ref` with an absolute reference URL while populating a map with the referenced subschema dictionaries.
- **Inputs**:
    - `schema`: A JSON object representing the schema in which `$ref` fields need to be resolved.
    - `url`: A string representing the base URL used to resolve relative references within the schema.
- **Control Flow**:
    - Define a lambda function `visit_refs` to recursively visit each node in the JSON schema.
    - If the node is an array, iterate over each element and recursively call `visit_refs`.
    - If the node is an object and contains a `$ref` field, check if the reference is already resolved in `_refs`.
    - If the reference is not resolved, determine if it is a remote reference (starting with 'https://') or a local reference (starting with '#/').
    - For remote references, fetch the referenced schema using `_fetch_json`, resolve its references, and store it in `_refs`.
    - For local references, adjust the `$ref` to an absolute URL using the provided `url`.
    - Split the reference pointer by '/' and navigate through the target schema to resolve the reference.
    - Store the resolved target in `_refs` using the absolute reference URL as the key.
    - If the node is an object without a `$ref`, recursively call `visit_refs` on each value in the object.
- **Output**: The function does not return a value but modifies the input `schema` in place and updates the `_refs` map with resolved references.
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)


---
#### SchemaConverter::\_generate\_constant\_rule<!-- {{#callable:SchemaConverter::_generate_constant_rule}} -->
The function `_generate_constant_rule` generates a grammar rule for a constant JSON value by formatting it as a literal string.
- **Inputs**:
    - `value`: A JSON object representing the constant value to be converted into a grammar rule.
- **Control Flow**:
    - The function takes a JSON object `value` as input.
    - It calls the `dump` method on the JSON object to convert it into a string representation.
    - The resulting string is passed to the [`format_literal`](#format_literal) function, which formats it as a literal string suitable for grammar rules.
    - The formatted string is returned as the output.
- **Output**: A string representing the formatted grammar rule for the given constant JSON value.
- **Functions called**:
    - [`format_literal`](#format_literal)
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)


---
#### SchemaConverter::visit<!-- {{#callable:SchemaConverter::visit}} -->
The `visit` function processes a JSON schema to generate a grammar rule based on the schema's structure and properties.
- **Inputs**:
    - `schema`: A JSON object representing the schema to be processed.
    - `name`: A string representing the name of the rule to be generated.
- **Control Flow**:
    - Initialize `schema_type` and `schema_format` from the schema if available.
    - Determine `rule_name` based on whether the name is reserved or empty.
    - Check if the schema contains a `$ref` and resolve it if present.
    - Handle `oneOf` or `anyOf` by generating a union rule for alternative schemas.
    - If `schema_type` is an array, generate a union rule for each type in the array.
    - Handle `const` by generating a constant rule.
    - Handle `enum` by generating a rule for enumerated values.
    - For object types, build an object rule considering properties and additional properties.
    - For `allOf`, combine properties from all schemas and build an object rule.
    - For array types, handle `items` or `prefixItems` to generate rules for array elements.
    - For string types with a `pattern`, visit the pattern to generate a rule.
    - Handle specific string formats like `uuid` or other known formats.
    - For strings with `minLength` or `maxLength`, generate a rule with repetition constraints.
    - For integers with min/max constraints, build a rule with numeric range constraints.
    - If the schema is empty or an object, add a primitive rule for objects.
    - If none of the above conditions are met, handle unrecognized schemas and add primitive rules if possible.
- **Output**: Returns a string representing the name of the generated rule.
- **Functions called**:
    - [`is_reserved_name`](#is_reserved_name)
    - [`SchemaConverter::_add_rule`](#SchemaConverter_add_rule)
    - [`SchemaConverter::_resolve_ref`](#SchemaConverter_resolve_ref)
    - [`SchemaConverter::_generate_union_rule`](#SchemaConverter_generate_union_rule)
    - [`SchemaConverter::_generate_constant_rule`](#SchemaConverter_generate_constant_rule)
    - [`SchemaConverter::_build_object_rule`](#SchemaConverter_build_object_rule)
    - [`build_repetition`](#build_repetition)
    - [`SchemaConverter::_visit_pattern`](#SchemaConverter_visit_pattern)
    - [`SchemaConverter::_add_primitive`](#SchemaConverter_add_primitive)
    - [`_build_min_max_int`](#_build_min_max_int)
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)


---
#### SchemaConverter::check\_errors<!-- {{#callable:SchemaConverter::check_errors}} -->
The `check_errors` function checks for any errors or warnings in the JSON schema conversion process and throws an exception or logs a warning accordingly.
- **Inputs**: None
- **Control Flow**:
    - Check if the `_errors` vector is not empty.
    - If `_errors` is not empty, throw a `std::runtime_error` with a message containing the joined error messages.
    - Check if the `_warnings` vector is not empty.
    - If `_warnings` is not empty, print a warning message to `stderr` with the joined warning messages.
- **Output**: The function does not return any value; it either throws an exception or logs a warning.
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)


---
#### SchemaConverter::format\_grammar<!-- {{#callable:SchemaConverter::format_grammar}} -->
The `format_grammar` function generates a formatted string representation of grammar rules stored in the `_rules` map of the `SchemaConverter` class.
- **Inputs**: None
- **Control Flow**:
    - Initialize a `std::stringstream` object `ss`.
    - Iterate over each key-value pair in the `_rules` map.
    - For each pair, append the key, the string ' ::= ', and the value to the `stringstream`, followed by a newline character.
    - Convert the `stringstream` to a string and return it.
- **Output**: A `std::string` containing the formatted grammar rules.
- **See also**: [`SchemaConverter`](#SchemaConverter)  (Data Structure)



---
### TrieNode<!-- {{#data_structure:SchemaConverter::_not_strings::TrieNode}} -->
- **Type**: `struct`
- **Members**:
    - `children`: A map that associates characters with their corresponding child TrieNode.
    - `is_end_of_string`: A boolean flag indicating if the node marks the end of a string.
- **Description**: The `TrieNode` struct is a fundamental building block for implementing a Trie data structure, which is used for efficient retrieval of strings. Each `TrieNode` contains a map of child nodes, allowing for branching based on character input, and a boolean flag to denote if the node represents the end of a valid string in the Trie. This structure supports operations like insertion of strings, enabling the construction of a Trie for various applications such as autocomplete and spell-checking.
- **Member Functions**:
    - [`SchemaConverter::_not_strings::TrieNode::TrieNode`](#SchemaConverter_not_strings::TrieNode::TrieNode)
    - [`SchemaConverter::_not_strings::TrieNode::insert`](#SchemaConverter_not_strings::TrieNode::insert)

**Methods**

---
#### TrieNode::TrieNode<!-- {{#callable:SchemaConverter::_not_strings::TrieNode::TrieNode}} -->
The `TrieNode` constructor initializes a `TrieNode` object with the `is_end_of_string` flag set to `false`.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes the `is_end_of_string` member variable to `false`.
- **Output**: A `TrieNode` object with its `is_end_of_string` member variable initialized to `false`.
- **See also**: [`SchemaConverter::_not_strings::TrieNode`](#SchemaConverter::_not_strings::TrieNode)  (Data Structure)


---
#### TrieNode::insert<!-- {{#callable:SchemaConverter::_not_strings::TrieNode::insert}} -->
The `insert` function adds a string to a Trie data structure by creating or navigating through nodes for each character in the string.
- **Inputs**:
    - `string`: A constant reference to a `std::string` that represents the string to be inserted into the Trie.
- **Control Flow**:
    - Initialize a pointer `node` to the current TrieNode instance (`this`).
    - Iterate over each character `c` in the input string.
    - For each character, update `node` to point to the child node corresponding to `c`, creating a new TrieNode if necessary.
    - After processing all characters, mark the current node as the end of a string by setting `is_end_of_string` to `true`.
- **Output**: The function does not return any value; it modifies the Trie structure in place.
- **See also**: [`SchemaConverter::_not_strings::TrieNode`](#SchemaConverter::_not_strings::TrieNode)  (Data Structure)



# Functions

---
### build\_repetition<!-- {{#callable:build_repetition}} -->
The `build_repetition` function generates a string representation of a repetition pattern based on specified minimum and maximum occurrences, with an optional separator.
- **Inputs**:
    - `item_rule`: A string representing the rule or pattern for a single item.
    - `min_items`: An integer specifying the minimum number of times the item_rule should be repeated.
    - `max_items`: An integer specifying the maximum number of times the item_rule can be repeated.
    - `separator_rule`: An optional string representing the rule or pattern for separating repeated items; defaults to an empty string if not provided.
- **Control Flow**:
    - Check if max_items is zero, returning an empty string if true.
    - Check if min_items is zero and max_items is one, returning item_rule followed by '?' if true.
    - If separator_rule is empty, determine the appropriate repetition pattern based on min_items and max_items, returning item_rule followed by '+', '*', or a range in curly braces.
    - If separator_rule is not empty, recursively build a repetition pattern with the separator, adjusting min_items and max_items accordingly.
    - If min_items is zero, wrap the result in parentheses followed by '?'.
- **Output**: A string representing the repetition pattern of the item_rule, formatted according to the specified min_items, max_items, and separator_rule.


---
### \_build\_min\_max\_int<!-- {{#callable:_build_min_max_int}} -->
The function `_build_min_max_int` generates a regular expression pattern for integers within a specified range and writes it to a given output stream.
- **Inputs**:
    - `min_value`: An integer representing the minimum value of the range.
    - `max_value`: An integer representing the maximum value of the range.
    - `out`: A reference to a `std::stringstream` where the generated pattern will be written.
    - `decimals_left`: An optional integer specifying the number of decimal places left to consider, defaulting to 16.
    - `top_level`: An optional boolean indicating if the current call is at the top level, defaulting to true.
- **Control Flow**:
    - Check if `min_value` and `max_value` are not the minimum and maximum integer limits, respectively, to determine if they are set.
    - Define helper lambdas `digit_range` and `more_digits` to append digit range patterns and repetition patterns to the output stream.
    - Define a recursive lambda `uniform_range` to handle uniform digit ranges between two string representations of numbers.
    - If both `min_value` and `max_value` are set, handle negative ranges by recursively calling `_build_min_max_int` with negated values, and handle positive ranges by iterating through digit lengths and using `uniform_range`.
    - If only `min_value` is set, handle negative and positive ranges separately, using `digit_range` and `more_digits` to construct patterns.
    - If only `max_value` is set, handle positive and negative ranges separately, using recursive calls to `_build_min_max_int`.
    - Throw a runtime error if neither `min_value` nor `max_value` is set.
- **Output**: The function outputs a regular expression pattern to the provided `std::stringstream` that matches integers within the specified range.


---
### is\_reserved\_name<!-- {{#callable:is_reserved_name}} -->
The `is_reserved_name` function checks if a given string is a reserved name by comparing it against a set of predefined reserved names.
- **Inputs**:
    - `name`: A constant reference to a `std::string` representing the name to be checked against reserved names.
- **Control Flow**:
    - A static unordered set `RESERVED_NAMES` is declared to store reserved names.
    - The function checks if `RESERVED_NAMES` is empty, and if so, initializes it by inserting the string "root" and the keys from the `PRIMITIVE_RULES` and `STRING_FORMAT_RULES` maps.
    - The function then checks if the provided `name` exists in the `RESERVED_NAMES` set.
    - The function returns `true` if the `name` is found in the set, otherwise it returns `false`.
- **Output**: A boolean value indicating whether the provided name is a reserved name (`true` if it is reserved, `false` otherwise).


---
### replacePattern<!-- {{#callable:replacePattern}} -->
The `replacePattern` function replaces all occurrences of a regex pattern in a given input string with a replacement string generated by a provided function.
- **Inputs**:
    - `input`: A constant reference to a `std::string` representing the input text in which the pattern replacements will occur.
    - `regex`: A constant reference to a `std::regex` object that defines the pattern to search for within the input string.
    - `replacement`: A constant reference to a `std::function` that takes a `std::smatch` object and returns a `std::string`, used to generate the replacement string for each match found.
- **Control Flow**:
    - Initialize a `std::smatch` object to store match results and a `std::string` to accumulate the result.
    - Set iterators `searchStart` and `searchEnd` to the beginning and end of the input string, respectively.
    - Enter a loop that continues as long as `std::regex_search` finds matches in the input string between `searchStart` and `searchEnd`.
    - For each match found, append the substring from `searchStart` to the start of the match to the result string.
    - Use the `replacement` function to generate a replacement string for the current match and append it to the result string.
    - Update `searchStart` to the end of the current match's suffix to continue searching for further matches.
    - After the loop, append any remaining part of the input string from `searchStart` to `searchEnd` to the result string.
    - Return the accumulated result string.
- **Output**: A `std::string` containing the input string with all occurrences of the regex pattern replaced by the generated replacement strings.


---
### format\_literal<!-- {{#callable:format_literal}} -->
The `format_literal` function escapes special characters in a string and returns it enclosed in double quotes.
- **Inputs**:
    - `literal`: A constant reference to a `std::string` that represents the input string to be formatted.
- **Control Flow**:
    - The function calls [`replacePattern`](#replacePattern) with the input `literal`, a regex pattern `GRAMMAR_LITERAL_ESCAPE_RE`, and a lambda function to replace matched characters with their escaped equivalents from `GRAMMAR_LITERAL_ESCAPES`.
    - The lambda function captures each match, retrieves the first character, and returns its corresponding escaped string from the `GRAMMAR_LITERAL_ESCAPES` map.
    - The result of [`replacePattern`](#replacePattern) is stored in the `escaped` variable.
    - The function returns the `escaped` string enclosed in double quotes.
- **Output**: A `std::string` that is the input string with special characters escaped and enclosed in double quotes.
- **Functions called**:
    - [`replacePattern`](#replacePattern)


---
### json\_schema\_to\_grammar<!-- {{#callable:json_schema_to_grammar}} -->
The `json_schema_to_grammar` function converts a JSON schema into a grammar format, optionally using GBNF if specified.
- **Inputs**:
    - `schema`: A JSON object representing the schema to be converted into a grammar.
    - `force_gbnf`: A boolean flag indicating whether to force the use of GBNF (Generalized Backus-Naur Form) for the grammar conversion.
- **Control Flow**:
    - If the `LLAMA_USE_LLGUIDANCE` preprocessor directive is defined and `force_gbnf` is false, the function returns a string with `%llguidance` and the JSON schema dumped as a string.
    - If `LLAMA_USE_LLGUIDANCE` is not defined, the function proceeds to build a grammar using the [`build_grammar`](#build_grammar) function.
    - Within [`build_grammar`](#build_grammar), a copy of the schema is made, and the `resolve_refs` and `add_schema` callbacks are invoked to process the schema.
- **Output**: A string representing the grammar derived from the JSON schema, formatted according to the specified options.
- **Functions called**:
    - [`build_grammar`](#build_grammar)


---
### build\_grammar<!-- {{#callable:build_grammar}} -->
The `build_grammar` function constructs a grammar string from a JSON schema using a callback to define rules and options for grammar building.
- **Inputs**:
    - `cb`: A callback function that takes a `common_grammar_builder` object and defines rules and schemas for the grammar.
    - `options`: A `common_grammar_options` object that contains options for grammar building, such as whether to use dotall mode.
- **Control Flow**:
    - Initialize a `SchemaConverter` object with a lambda function to fetch JSON and the `dotall` option from `options`.
    - Create a `common_grammar_builder` object with three lambda functions: `add_rule`, `add_schema`, and `resolve_refs`, each calling corresponding methods on the `SchemaConverter`.
    - Invoke the callback `cb` with the `common_grammar_builder` object to allow the user to define grammar rules and schemas.
    - Call `check_errors` on the `SchemaConverter` to ensure no errors occurred during schema conversion.
    - Return the formatted grammar string by calling `format_grammar` on the `SchemaConverter`.
- **Output**: A string representing the constructed grammar based on the provided JSON schema and options.


