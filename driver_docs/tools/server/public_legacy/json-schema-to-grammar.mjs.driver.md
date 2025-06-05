# Purpose
This source code file is a JavaScript module designed to convert JSON schemas into grammar rules. The primary functionality is encapsulated within the `SchemaConverter` class, which processes JSON schema definitions and generates corresponding grammar rules. The module includes several helper functions and constants to facilitate this conversion, such as `_buildRepetition`, `_generateMinMaxInt`, and various regular expression patterns. These components work together to handle different schema types, including objects, arrays, strings, numbers, and more, by defining rules for each type and their constraints.

The file defines a set of primitive and string format rules, such as `PRIMITIVE_RULES` and `STRING_FORMAT_RULES`, which are used to map JSON schema types to grammar rules. The `SchemaConverter` class uses these rules to build a comprehensive grammar representation of a given JSON schema. It supports various schema features, including references (`$ref`), pattern matching, and constraints like `minItems` and `maxItems` for arrays, or `minLength` and `maxLength` for strings. The class also includes methods for resolving schema references, generating union rules, and formatting literals and range characters.

Overall, this module provides a robust framework for translating JSON schemas into grammar rules, which can be used for tasks such as validation, parsing, or code generation. The code is structured to handle a wide range of schema features and constraints, making it a versatile tool for developers working with JSON schema-based applications.
# Imports and Dependencies

---
- `node-fetch`


# Global Variables

---
### SPACE\_RULE
- **Type**: `string`
- **Description**: The `SPACE_RULE` variable is a string that defines a grammar rule for matching spaces and newlines in a specific format. It allows for a single space character or one to two newline characters followed by up to 20 tab characters.
- **Use**: This variable is used to define a rule for handling whitespace in grammar parsing.


---
### PRIMITIVE\_RULES
- **Type**: `object`
- **Description**: `PRIMITIVE_RULES` is a global object that maps various primitive data types and structures to their corresponding grammar rules using the `BuiltinRule` class. Each key in the object represents a primitive type or structure, such as 'boolean', 'number', 'integer', 'value', 'object', 'array', 'uuid', 'char', 'string', and 'null', and is associated with a `BuiltinRule` instance that defines the grammar rule for that type and any dependencies it may have.
- **Use**: This variable is used to define and store grammar rules for primitive data types and structures, which can be referenced and utilized throughout the codebase for schema conversion and validation.


---
### STRING\_FORMAT\_RULES
- **Type**: `object`
- **Description**: `STRING_FORMAT_RULES` is a global object that defines a set of built-in rules for parsing and validating specific string formats such as 'date', 'time', and 'date-time'. Each key in the object corresponds to a string format and maps to a `BuiltinRule` instance, which contains a grammar rule for validating that format.
- **Use**: This variable is used to provide predefined grammar rules for specific string formats, which can be utilized in schema validation processes.


---
### RESERVED\_NAMES
- **Type**: `object`
- **Description**: The `RESERVED_NAMES` variable is a global object that combines a set of predefined rules for primitive types and string formats. It includes rules for JSON schema primitive types like boolean, number, integer, and complex types like object and array, as well as string formats like date and time.
- **Use**: This variable is used to store and provide access to a collection of predefined grammar rules for various data types and formats, which are utilized in schema conversion and validation processes.


---
### INVALID\_RULE\_CHARS\_RE
- **Type**: `RegExp`
- **Description**: The `INVALID_RULE_CHARS_RE` is a regular expression pattern that matches any character that is not a digit, uppercase or lowercase letter, or a hyphen. It is used to identify invalid characters in rule names.
- **Use**: This variable is used to sanitize rule names by replacing invalid characters with a hyphen.


---
### GRAMMAR\_LITERAL\_ESCAPE\_RE
- **Type**: `RegExp`
- **Description**: The `GRAMMAR_LITERAL_ESCAPE_RE` is a regular expression pattern that matches newline characters (`\n`), carriage return characters (`\r`), and double quote characters (`"`). This pattern is used to identify these specific characters within strings that need to be escaped.
- **Use**: This variable is used to find and escape newline, carriage return, and double quote characters in strings to ensure they are properly formatted for grammar rules.


---
### GRAMMAR\_RANGE\_LITERAL\_ESCAPE\_RE
- **Type**: `RegExp`
- **Description**: `GRAMMAR_RANGE_LITERAL_ESCAPE_RE` is a regular expression pattern that matches specific characters within a string. It is designed to identify newline characters (`\n` and `\r`), double quotes (`"`), closing square brackets (`]`), hyphens (`-`), and backslashes (`\\`).
- **Use**: This variable is used to escape certain characters in grammar range literals to ensure they are correctly processed in regular expressions.


---
### GRAMMAR\_LITERAL\_ESCAPES
- **Type**: `object`
- **Description**: `GRAMMAR_LITERAL_ESCAPES` is a global object that maps specific characters to their escaped string representations. It includes mappings for carriage return (`\r`), newline (`\n`), double quote (`\"`), hyphen (`\-`), and closing square bracket (`\]`).
- **Use**: This variable is used to replace specific characters in strings with their escaped versions during grammar processing.


---
### NON\_LITERAL\_SET
- **Type**: `Set`
- **Description**: The `NON_LITERAL_SET` is a global variable defined as a JavaScript `Set` containing a collection of special characters used in regular expressions and grammar rules. These characters include `|`, `.`, `(`, `)`, `[`, `]`, `{`, `}`, `*`, `+`, and `?`. The set is used to identify characters that are not considered literals in the context of grammar and regular expression parsing.
- **Use**: This variable is used to determine which characters in a pattern are not literals and may require special handling during parsing or transformation.


---
### ESCAPED\_IN\_REGEXPS\_BUT\_NOT\_IN\_LITERALS
- **Type**: `Set`
- **Description**: The `ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS` is a global variable defined as a Set containing special characters that are typically escaped in regular expressions but not in string literals. These characters include `^`, `$`, `.`, `[`, `]`, `(`, `)`, `|`, `{`, `}`, `*`, `+`, and `?`. This distinction is important for handling patterns and literals differently in the context of grammar and schema conversion.
- **Use**: This variable is used to identify characters that need special handling when converting patterns to grammar rules, ensuring they are treated correctly in different contexts.


# Data Structures

---
### BuiltinRule
- **Type**: `class`
- **Members**:
    - `content`: A string representing the rule content.
    - `deps`: An array of dependencies required by the rule.
- **Description**: The `BuiltinRule` class is a data structure used to define grammar rules in a schema conversion context. Each instance of `BuiltinRule` contains a `content` string that specifies the rule's pattern or structure, and a `deps` array that lists any dependencies the rule has on other rules. This class is utilized to encapsulate the logic and dependencies of various grammar rules, facilitating the conversion of JSON schemas into grammar representations.


---
### SchemaConverter
- **Type**: `class`
- **Members**:
    - `_propOrder`: Stores the order of properties as specified in options.
    - `_allowFetch`: Indicates whether fetching remote schemas is allowed.
    - `_dotall`: Determines if dotall mode is enabled for regex patterns.
    - `_rules`: Holds the grammar rules for schema conversion.
    - `_refs`: Caches resolved schema references.
    - `_refsBeingResolved`: Tracks references currently being resolved to prevent circular dependencies.
- **Description**: The `SchemaConverter` class is designed to convert JSON schemas into grammar rules, facilitating the parsing and validation of JSON data against these schemas. It manages schema properties, handles references, and constructs grammar rules for various schema types, including objects, arrays, and strings. The class supports options for property order, remote schema fetching, and regex pattern handling, ensuring flexibility and extensibility in schema conversion.


# Functions

---
### \_buildRepetition
The `_buildRepetition` function generates a grammar rule for repeating an item rule a specified number of times, with optional separators.
- **Inputs**:
    - `itemRule`: A string representing the rule for the item to be repeated.
    - `minItems`: An integer specifying the minimum number of times the item should be repeated.
    - `maxItems`: An integer specifying the maximum number of times the item can be repeated, or undefined for no maximum.
    - `opts`: An optional object containing additional options, such as `separatorRule` for specifying a separator between items and `itemRuleIsLiteral` to indicate if the item rule is a literal.
- **Control Flow**:
    - Check if `maxItems` is 0, return an empty string if true.
    - Check if `minItems` is 0 and `maxItems` is 1, return the item rule followed by '?' if true.
    - Retrieve `separatorRule` and `itemRuleIsLiteral` from `opts`, defaulting to empty string and false respectively.
    - If `separatorRule` is empty, determine the repetition pattern based on `minItems` and `maxItems` and return the appropriate grammar rule.
    - If `separatorRule` is not empty, recursively build a repetition rule with the separator and return the result, optionally wrapped in parentheses with a '?' if `minItems` is 0.
- **Output**: A string representing the grammar rule for repeating the item rule according to the specified parameters.


---
### \_generateMinMaxInt
The function `_generateMinMaxInt` generates a grammar rule for matching integers within a specified range, considering both minimum and maximum values.
- **Inputs**:
    - `minValue`: The minimum integer value for the range, or null if there is no minimum.
    - `maxValue`: The maximum integer value for the range, or null if there is no maximum.
    - `out`: An array to which the generated grammar rule components are appended.
    - `decimalsLeft`: The number of decimal places left to consider, defaulting to 16.
    - `topLevel`: A boolean indicating if the current call is at the top level, defaulting to true.
- **Control Flow**:
    - Check if both `minValue` and `maxValue` are provided, and handle negative ranges by recursively calling `_generateMinMaxInt` with negated values.
    - If both `minValue` and `maxValue` are provided, generate grammar rules for numbers with varying digit lengths between the two values.
    - If only `minValue` is provided, handle negative and positive ranges separately, generating rules for numbers greater than or equal to `minValue`.
    - If only `maxValue` is provided, handle negative and positive ranges separately, generating rules for numbers less than or equal to `maxValue`.
    - If neither `minValue` nor `maxValue` is provided, throw an error.
- **Output**: The function appends components of a grammar rule to the `out` array, which represents the integer range specified by `minValue` and `maxValue`.


---
### digitRange
The `digitRange` function generates a character class pattern for a range of digits and appends it to an output array.
- **Inputs**:
    - `fromChar`: The starting character of the digit range.
    - `toChar`: The ending character of the digit range.
- **Control Flow**:
    - The function starts by pushing a '[' character to the output array to begin a character class.
    - It checks if the starting and ending characters are the same.
    - If they are the same, it pushes the single character to the output array.
    - If they are different, it pushes the starting character, a '-' character, and the ending character to the output array to represent a range.
    - Finally, it pushes a ']' character to the output array to close the character class.
- **Output**: The function does not return a value; it modifies the `out` array by appending a character class pattern.


---
### moreDigits
The `moreDigits` function appends a regex pattern to the output array that matches a specified range of digit repetitions.
- **Inputs**:
    - `minDigits`: The minimum number of digits to match.
    - `maxDigits`: The maximum number of digits to match.
- **Control Flow**:
    - The function starts by appending the regex pattern for a single digit '[0-9]' to the output array.
    - It checks if the minimum and maximum digits are equal and both are 1, in which case it returns immediately as the pattern is already complete.
    - If the minimum and maximum digits are not equal, it appends a repetition pattern '{minDigits,maxDigits}' to the output array.
    - If the maximum digits are not equal to the minimum digits, it appends a comma and the maximum digits to the pattern, unless the maximum is the maximum safe integer.
- **Output**: The function does not return a value; it modifies the `out` array in place by appending the constructed regex pattern.


---
### uniformRange
The `uniformRange` function generates a regular expression pattern to match numbers within a specified range, considering shared prefixes and digit constraints.
- **Inputs**:
    - `fromStr`: A string representing the lower bound of the range.
    - `toStr`: A string representing the upper bound of the range.
- **Control Flow**:
    - Initialize index `i` to 0 and increment it while characters at the same position in `fromStr` and `toStr` are equal.
    - If `i` is greater than 0, append the common prefix to the output array.
    - If `i` is less than the length of `fromStr`, handle the remaining digits by considering sub-ranges and digit constraints.
    - If the remaining substring length is greater than 0, recursively call `uniformRange` for sub-ranges and handle digit ranges and repetitions.
    - If the remaining substring length is 0, handle the range directly using character classes.
- **Output**: The function appends parts of a regular expression pattern to the `out` array, which matches numbers within the specified range.


---
### resolveRefs
The `resolveRefs` function resolves JSON schema references, potentially fetching remote schemas if allowed, and returns the fully resolved schema.
- **Inputs**:
    - `schema`: The JSON schema object that may contain references to be resolved.
    - `url`: The base URL used to resolve relative references within the schema.
- **Control Flow**:
    - Defines an asynchronous helper function `visit` to recursively traverse the schema.
    - Checks if the current node is an array and recursively visits each element if true.
    - Checks if the current node is an object and processes `$ref` if present.
    - If `$ref` is a remote URL and fetching is allowed, fetches the schema and resolves it recursively.
    - If `$ref` is a local reference, resolves it using the base URL and updates the reference.
    - Recursively visits all values of the object if no `$ref` is present.
    - Returns the fully resolved schema after all references are processed.
- **Output**: The function returns a Promise that resolves to the schema with all references resolved.


---
### \_generateUnionRule
The `_generateUnionRule` function generates a grammar rule for a union of alternative schemas by visiting each alternative schema and joining their rules with a pipe ('|') symbol.
- **Inputs**:
    - `name`: The base name for the rule being generated, which may be used to construct unique rule names for each alternative.
    - `altSchemas`: An array of alternative schemas that need to be combined into a union rule.
- **Control Flow**:
    - Iterate over each alternative schema in `altSchemas`.
    - For each alternative schema, call the `visit` method with the schema and a constructed name based on the base `name` and the index of the alternative.
    - Join the results of the `visit` calls with a pipe ('|') symbol to form the union rule.
- **Output**: A string representing the grammar rule for the union of the alternative schemas.


---
### \_visitPattern
The `_visitPattern` function processes a regex pattern string, transforming it into a grammar rule and adding it to the internal rules of the `SchemaConverter` class.
- **Inputs**:
    - `pattern`: A string representing a regex pattern that must start with '^' and end with '$'.
    - `name`: A string representing the name of the rule to be created from the pattern.
- **Control Flow**:
    - Check if the pattern starts with '^' and ends with '$', throwing an error if not.
    - Remove the leading '^' and trailing '$' from the pattern.
    - Initialize an empty object `subRuleIds` to store sub-rule identifiers.
    - Iterate over the pattern string, processing each character to build a sequence of grammar components.
    - Handle special characters like '.', '(', ')', '[', ']', '|', '*', '+', '?', and '{min,max}' to construct the grammar rule.
    - For each component, determine if it is a literal or a rule and transform it accordingly.
    - Join the sequence of components into a single rule string.
    - Add the constructed rule to the internal rules of the `SchemaConverter` class using `_addRule`.
- **Output**: Returns the name of the rule added to the internal rules of the `SchemaConverter` class.


---
### \_notStrings
The `_notStrings` function generates a grammar rule that matches strings not present in a given list of strings.
- **Inputs**:
    - `strings`: An array of strings that should not be matched by the generated grammar rule.
- **Control Flow**:
    - Define a `TrieNode` class to represent nodes in a trie data structure, with methods to insert strings.
    - Create a new `TrieNode` instance called `trie` and insert each string from the `strings` array into the trie.
    - Add a primitive rule for `char` using the `PRIMITIVE_RULES` dictionary.
    - Initialize an output array `out` with the beginning of a grammar rule that starts with a double quote and an opening parenthesis.
    - Define a recursive `visit` function to traverse the trie and build the grammar rule.
    - For each child node in the current trie node, add a character class to `out` that matches the current character, and recursively call `visit` on the child node if it has children.
    - If a child node is the end of a string, append a rule to match one or more characters using the `char` rule.
    - If the current node has children, add a rule to match any character not in the set of child characters, followed by zero or more characters using the `char` rule.
    - Close the parenthesis and add an optional quantifier if the root node is not the end of a string, then close the double quote and add a space rule.
    - Return the joined `out` array as the final grammar rule.
- **Output**: A string representing a grammar rule that matches any string not present in the input `strings` array.


---
### \_resolveRef
The `_resolveRef` function resolves a JSON schema reference to its corresponding schema definition, caching the result to avoid redundant resolutions.
- **Inputs**:
    - `ref`: A string representing the reference to be resolved, typically in the form of a URI or a JSON pointer.
- **Control Flow**:
    - Extract the reference name from the `ref` by splitting the string and taking the last segment.
    - Check if the reference name is already in the `_rules` or if the reference is currently being resolved to prevent circular dependencies.
    - If the reference is not resolved, add it to the `_refsBeingResolved` set to mark it as being processed.
    - Retrieve the resolved schema from the `_refs` cache using the `ref` as the key.
    - If the schema is not cached, resolve it by visiting the schema and generating a rule name.
    - Remove the reference from the `_refsBeingResolved` set after resolution.
    - Return the resolved reference name.
- **Output**: The function returns the name of the resolved reference, which is a string representing the rule name associated with the resolved schema.


---
### \_generateConstantRule
The `_generateConstantRule` function formats a given constant value into a grammar-compatible literal string.
- **Inputs**:
    - `value`: The constant value to be formatted into a grammar-compatible literal string.
- **Control Flow**:
    - The function takes a single input, `value`, which is expected to be a constant.
    - It uses the `JSON.stringify` method to convert the `value` into a JSON string representation.
    - The resulting JSON string is then passed to the `_formatLiteral` method to escape any special characters and format it as a grammar-compatible literal.
    - The formatted literal string is returned as the output.
- **Output**: A string representing the formatted constant value as a grammar-compatible literal.


---
### visit
The `visit` function recursively resolves JSON schema references, fetching remote schemas if necessary, and returns the resolved schema.
- **Inputs**:
    - `schema`: The JSON schema object that may contain references to be resolved.
    - `url`: The base URL used to resolve relative references within the schema.
- **Control Flow**:
    - Define an asynchronous function `visit` that takes a node `n` as input.
    - Check if `n` is an array, and if so, recursively call `visit` on each element using `Promise.all`.
    - If `n` is an object, check for a `$ref` property to determine if it needs to be resolved.
    - If `$ref` is a URL and not already resolved, fetch the schema from the URL if allowed, and resolve it recursively.
    - If `$ref` is a local reference (starts with '#/'), resolve it within the current schema using the provided `url`.
    - If `$ref` is not present, recursively call `visit` on each value of the object `n`.
    - Return the resolved node `n`.
- **Output**: The function returns the resolved schema object with all references replaced by their target definitions.


---
### \_addPrimitive
The `_addPrimitive` function adds a primitive rule to the internal ruleset of the `SchemaConverter` class, ensuring all dependencies are also added.
- **Inputs**:
    - `name`: The name of the primitive rule to be added.
    - `rule`: An instance of `BuiltinRule` containing the content and dependencies of the rule.
- **Control Flow**:
    - The function first adds the rule content to the internal ruleset using `_addRule`, which returns the rule name.
    - It iterates over each dependency in the rule's dependencies list.
    - For each dependency, it checks if the dependency is already in the ruleset.
    - If a dependency is not in the ruleset, it recursively calls `_addPrimitive` to add the dependency rule.
- **Output**: The function returns the name of the rule that was added to the internal ruleset.


---
### \_buildObjectRule
The `_buildObjectRule` function constructs a grammar rule for JSON objects based on specified properties, required fields, and additional properties.
- **Inputs**:
    - `properties`: An array of tuples where each tuple contains a property name and its corresponding schema.
    - `required`: A set of property names that are required in the object.
    - `name`: A string representing the base name for the rule being constructed.
    - `additionalProperties`: A schema or boolean indicating whether additional properties are allowed and their schema if applicable.
- **Control Flow**:
    - Sort properties based on their order in `propOrder` or their original order if not specified.
    - Iterate over properties to generate rule names for each key-value pair using the `visit` method.
    - Separate properties into required and optional based on the `required` set.
    - If `additionalProperties` is allowed, create a rule for additional key-value pairs using a generic key and value rule.
    - Construct the grammar rule string by concatenating rules for required properties, optional properties, and additional properties if applicable.
    - Return the constructed grammar rule string.
- **Output**: A string representing the grammar rule for the JSON object, including required, optional, and additional properties.


---
### formatGrammar
The `formatGrammar` function generates a formatted grammar string from a set of rules defined in the `SchemaConverter` class.
- **Inputs**: None
- **Control Flow**:
    - Iterates over the sorted entries of the `_rules` object in the `SchemaConverter` class.
    - For each rule, appends a formatted string representation of the rule to the `grammar` string.
    - Returns the complete `grammar` string after processing all rules.
- **Output**: A string representing the formatted grammar, with each rule defined in the `SchemaConverter` class.


---
### groupBy
The `groupBy` function is a helper function that groups elements of an iterable based on a key function, yielding pairs of keys and grouped elements.
- **Inputs**:
    - `iterable`: An iterable collection of elements to be grouped.
    - `keyFn`: A function that takes an element from the iterable and returns a key to group by.
- **Control Flow**:
    - Initialize `lastKey` to null and `group` to an empty array.
    - Iterate over each `element` in the `iterable`.
    - For each `element`, compute the `key` using `keyFn`.
    - If `lastKey` is not null and `key` is different from `lastKey`, yield the current `lastKey` and `group`, then reset `group` to an empty array.
    - Add the current `element` to `group` and update `lastKey` to the current `key`.
    - After the loop, if `group` is not empty, yield the final `lastKey` and `group`.
- **Output**: Yields pairs of keys and lists of grouped elements, where each pair consists of a key and the list of elements that share that key.


