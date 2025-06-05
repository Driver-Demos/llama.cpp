# Purpose
This Python script is designed to convert JSON schemas into a grammar format suitable for use with a tool like `llama-cli`. The script provides a comprehensive solution for parsing JSON schemas, resolving references, and generating grammar rules that can produce JSON data conforming to the specified schema. The main components of the script include functions for building repetition patterns, generating integer ranges, and handling various JSON schema constructs such as objects, arrays, and primitive types. The script also supports handling of regular expressions within JSON schemas, converting them into a grammar format.

The script is structured as a command-line tool, utilizing the `argparse` module to handle input arguments, which include options for property order, fetching remote schemas, and handling regular expression patterns. The core functionality is encapsulated within the `SchemaConverter` class, which manages the conversion process by visiting each part of the schema and generating corresponding grammar rules. The script is designed to be executed as a standalone program, reading a JSON schema from a file or URL, and outputting the generated grammar to standard output. This makes it a versatile tool for developers working with JSON schemas who need to generate compatible grammar definitions for further processing or validation tasks.
# Imports and Dependencies

---
- `__future__.annotations`
- `argparse`
- `itertools`
- `json`
- `re`
- `sys`
- `typing.Any`
- `typing.List`
- `typing.Optional`
- `typing.Set`
- `typing.Tuple`
- `typing.Union`
- `requests`


# Global Variables

---
### SPACE\_RULE
- **Type**: `str`
- **Description**: The `SPACE_RULE` variable is a string that defines a regular expression pattern for matching spaces, newlines, and tabs. It allows for a single space character, one or two newline characters, and up to 20 tab characters.
- **Use**: This variable is used to define a pattern for whitespace handling in the grammar rules of the program.


---
### PRIMITIVE\_RULES
- **Type**: `dict`
- **Description**: `PRIMITIVE_RULES` is a dictionary that maps string keys representing different primitive data types and structures to instances of the `BuiltinRule` class. Each `BuiltinRule` instance contains a string pattern that defines the syntax for the corresponding data type or structure, and a list of dependencies that are other rules required to define the pattern.
- **Use**: This variable is used to define and store grammar rules for parsing and validating various primitive data types and structures in a JSON-like format.


---
### STRING\_FORMAT\_RULES
- **Type**: `dict`
- **Description**: `STRING_FORMAT_RULES` is a dictionary that maps string format names to `BuiltinRule` objects, which define regular expression patterns for validating specific string formats such as 'date', 'time', and 'date-time'. Each key in the dictionary corresponds to a format name, and the associated value is a `BuiltinRule` instance containing a regex pattern and a list of dependencies on other rules.
- **Use**: This variable is used to store and provide access to predefined string format rules for validating and parsing date and time strings in various formats.


---
### DOTALL
- **Type**: `str`
- **Description**: The `DOTALL` variable is a string that represents a regular expression pattern matching any Unicode character from U+0000 to U+10FFFF. This range covers all possible Unicode code points, effectively allowing the pattern to match any character, including those outside the Basic Multilingual Plane.
- **Use**: This variable is used in regular expression operations to match any character across the full range of Unicode code points.


---
### DOT
- **Type**: `str`
- **Description**: The variable `DOT` is a string that represents a regular expression pattern. This pattern matches any character except for the newline characters '\x0A' (line feed) and '\x0D' (carriage return).
- **Use**: It is used in regular expression operations to match any character except newline characters.


---
### RESERVED\_NAMES
- **Type**: `set`
- **Description**: `RESERVED_NAMES` is a set containing a collection of reserved names used within the program. It includes the strings 'root' and 'dot', as well as all the keys from the `PRIMITIVE_RULES` and `STRING_FORMAT_RULES` dictionaries. This set is likely used to ensure that certain names are not used for other purposes within the program, maintaining consistency and avoiding conflicts.
- **Use**: This variable is used to store and manage a list of names that are reserved and should not be used for other identifiers in the program.


---
### INVALID\_RULE\_CHARS\_RE
- **Type**: `re.Pattern`
- **Description**: The variable `INVALID_RULE_CHARS_RE` is a compiled regular expression pattern that matches any character that is not an uppercase or lowercase letter, a digit, or a hyphen. This pattern is used to identify invalid characters in a rule name or identifier.
- **Use**: This variable is used to sanitize or validate rule names by ensuring they only contain valid characters.


---
### GRAMMAR\_LITERAL\_ESCAPE\_RE
- **Type**: `re.Pattern`
- **Description**: `GRAMMAR_LITERAL_ESCAPE_RE` is a compiled regular expression pattern that matches carriage return, newline, and double quote characters. It is used to identify these specific characters in a string for further processing.
- **Use**: This variable is used to escape certain characters in grammar literals by identifying them for replacement or handling.


---
### GRAMMAR\_RANGE\_LITERAL\_ESCAPE\_RE
- **Type**: `re.Pattern`
- **Description**: `GRAMMAR_RANGE_LITERAL_ESCAPE_RE` is a compiled regular expression pattern that matches any of the characters carriage return (\r), newline (\n), double quote ("), closing square bracket (]), hyphen (-), or backslash (\\).
- **Use**: This variable is used to identify and potentially escape specific characters within a grammar range literal.


---
### GRAMMAR\_LITERAL\_ESCAPES
- **Type**: `dict`
- **Description**: `GRAMMAR_LITERAL_ESCAPES` is a dictionary that maps certain special characters to their escaped string representations. This includes characters like carriage return (`\r`), newline (`\n`), double quote (`\"`), hyphen (`\-`), and closing square bracket (`\]`).
- **Use**: This variable is used to replace special characters with their escaped versions in grammar literals, ensuring they are correctly interpreted in string processing.


---
### NON\_LITERAL\_SET
- **Type**: `set`
- **Description**: The `NON_LITERAL_SET` is a set containing characters that are considered non-literal in regular expressions. These characters include special symbols like '|', '.', '(', ')', '[', ']', '{', '}', '*', '+', and '?', which are used for defining patterns and repetitions in regex.
- **Use**: This variable is used to identify and handle non-literal characters in regular expression processing.


---
### ESCAPED\_IN\_REGEXPS\_BUT\_NOT\_IN\_LITERALS
- **Type**: `set`
- **Description**: The variable `ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS` is a set containing special characters that are typically escaped in regular expressions but not in string literals. These characters include `^`, `$`, `.`, `[`, `]`, `(`, `)`, `|`, `{`, `}`, `*`, `+`, and `?`.
- **Use**: This set is used to identify characters that need special handling when converting regular expressions to other formats or when processing them in contexts where they are not automatically escaped.


# Classes

---
### BuiltinRule<!-- {{#class:llama.cpp/examples/json_schema_to_grammar.BuiltinRule}} -->
- **Members**:
    - `content`: Stores the main content or rule as a string.
    - `deps`: Holds a list of dependencies for the rule, defaulting to an empty list if not provided.
- **Description**: The BuiltinRule class is a simple data structure designed to encapsulate a rule with its associated content and dependencies. It initializes with a content string and an optional list of dependencies, defaulting to an empty list if none are provided. This class is used to define various primitive and format rules within a schema conversion context.
- **Methods**:
    - [`llama.cpp/examples/json_schema_to_grammar.BuiltinRule.__init__`](#BuiltinRule__init__)

**Methods**

---
#### BuiltinRule\.\_\_init\_\_<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.BuiltinRule.__init__}} -->
The `__init__` method initializes a `BuiltinRule` object with content and optional dependencies.
- **Inputs**:
    - `content`: A string representing the content of the rule.
    - `deps`: An optional list of dependencies for the rule, defaulting to an empty list if not provided.
- **Control Flow**:
    - Assigns the `content` parameter to the `content` attribute of the instance.
    - Assigns the `deps` parameter to the `deps` attribute of the instance, defaulting to an empty list if `deps` is None.
- **Output**: None, as it is a constructor method for initializing an object.
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.BuiltinRule`](#cpp/examples/json_schema_to_grammarBuiltinRule)  (Base Class)



---
### SchemaConverter<!-- {{#class:llama.cpp/examples/json_schema_to_grammar.SchemaConverter}} -->
- **Members**:
    - `_prop_order`: Stores the order of properties for schema conversion.
    - `_allow_fetch`: Indicates if fetching remote schemas is allowed.
    - `_dotall`: Determines if dot (.) matches all characters including line breaks in regex patterns.
    - `_raw_pattern`: Specifies if string patterns are treated as raw patterns without quotes.
    - `_rules`: Holds the grammar rules for schema conversion.
    - `_refs`: Caches resolved schema references.
    - `_refs_being_resolved`: Tracks references currently being resolved to prevent circular dependencies.
- **Description**: The SchemaConverter class is designed to convert JSON schemas into a grammar format suitable for use in generating JSON that conforms to the schema. It manages schema properties, handles references, and supports various schema features like pattern matching and property ordering. The class also includes mechanisms to resolve schema references, including fetching remote schemas if allowed, and to transform regular expression patterns into grammar rules. It maintains internal state for rules and references, ensuring efficient and accurate schema conversion.
- **Methods**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.__init__`](#SchemaConverter__init__)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._format_literal`](#SchemaConverter_format_literal)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.not_literal`](#SchemaConverternot_literal)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._not_strings`](#SchemaConverter_not_strings)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._add_rule`](#SchemaConverter_add_rule)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.resolve_refs`](#SchemaConverterresolve_refs)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._generate_union_rule`](#SchemaConverter_generate_union_rule)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._visit_pattern`](#SchemaConverter_visit_pattern)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._resolve_ref`](#SchemaConverter_resolve_ref)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._generate_constant_rule`](#SchemaConverter_generate_constant_rule)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.visit`](#SchemaConvertervisit)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._add_primitive`](#SchemaConverter_add_primitive)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._build_object_rule`](#SchemaConverter_build_object_rule)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.format_grammar`](#SchemaConverterformat_grammar)

**Methods**

---
#### SchemaConverter\.\_\_init\_\_<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter.__init__}} -->
The `__init__` method initializes a `SchemaConverter` object with configuration options for property order, fetching permissions, regex dotall behavior, and raw pattern handling.
- **Inputs**:
    - `prop_order`: A parameter that specifies the order of properties, likely as a dictionary or list, to determine precedence in object properties.
    - `allow_fetch`: A boolean parameter that indicates whether fetching of remote schemas over HTTPS is allowed.
    - `dotall`: A boolean parameter that determines if the dot ('.') in regular expressions should match all characters, including line breaks.
    - `raw_pattern`: A boolean parameter that specifies whether string patterns should be treated as raw patterns without quotes or quote escapes.
- **Control Flow**:
    - Assigns the `prop_order` parameter to the instance variable `_prop_order`.
    - Assigns the `allow_fetch` parameter to the instance variable `_allow_fetch`.
    - Assigns the `dotall` parameter to the instance variable `_dotall`.
    - Assigns the `raw_pattern` parameter to the instance variable `_raw_pattern`.
    - Initializes the `_rules` dictionary with a predefined rule for 'space'.
    - Initializes `_refs` as an empty dictionary to store schema references.
    - Initializes `_refs_being_resolved` as an empty set to track references currently being resolved.
- **Output**: The method does not return any value; it initializes the instance variables of the `SchemaConverter` object.
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.\_format\_literal<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter._format_literal}} -->
The `_format_literal` method escapes special characters in a given string literal and returns it enclosed in double quotes.
- **Inputs**:
    - `literal`: A string that may contain special characters such as carriage return, newline, or double quotes that need to be escaped.
- **Control Flow**:
    - The method uses a regular expression `GRAMMAR_LITERAL_ESCAPE_RE` to find special characters in the input `literal`.
    - For each match found, it uses a lambda function to replace the character with its corresponding escape sequence from the `GRAMMAR_LITERAL_ESCAPES` dictionary, or leaves it unchanged if no escape sequence is defined.
    - The escaped string is then formatted into a new string enclosed in double quotes and returned.
- **Output**: A string with special characters escaped and enclosed in double quotes.
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.not\_literal<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter.not_literal}} -->
The `not_literal` method generates a string pattern that matches any string not containing the specified literal sequence, with optional handling for escaped underscores.
- **Inputs**:
    - `literal`: A non-empty string representing the sequence of characters to be excluded from matching.
    - `dotall`: A boolean flag indicating whether the dot (.) should match all characters, including line breaks, in regular expression patterns. Defaults to True.
    - `maybe_escaped_underscores`: A boolean flag indicating whether underscores in the literal should be treated as possibly escaped. Defaults to False.
- **Control Flow**:
    - The method asserts that the `literal` string is not empty.
    - Defines a recursive inner function `recurse` that processes each character in the `literal`.
    - For each character `c` in `literal`, if `maybe_escaped_underscores` is True and `c` is an underscore, it yields a pattern that excludes `c` and optionally matches an escaped underscore.
    - If `c` is not an underscore or `maybe_escaped_underscores` is False, it yields a pattern that excludes `c`.
    - If there are more characters in `literal`, it yields a pattern that includes `c` and recursively processes the next character.
    - The method returns a string that combines all yielded patterns into a single expression enclosed in parentheses.
- **Output**: A string representing a pattern that matches any string not containing the specified literal sequence, formatted as a regular expression.
- **Functions called**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._format_literal`](#SchemaConverter_format_literal)
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.\_not\_strings<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter._not_strings}} -->
The `_not_strings` method generates a grammar rule that matches strings not present in a given list of strings using a trie data structure.
- **Inputs**:
    - `strings`: A list of strings that the generated grammar rule should not match.
- **Control Flow**:
    - A [`TrieNode`](#cpp/examples/json_schema_to_grammarSchemaConverter._not_strings.TrieNode) class is defined with methods to initialize and insert strings into a trie structure.
    - A [`TrieNode`](#cpp/examples/json_schema_to_grammarSchemaConverter._not_strings.TrieNode) instance is created and each string from the input list is inserted into the trie.
    - A character rule is added using [`_add_primitive`](#SchemaConverter_add_primitive) method for handling individual characters.
    - An output list `out` is initialized with a starting pattern for a string.
    - A recursive [`visit`](#SchemaConvertervisit) function is defined to traverse the trie and build the grammar rule by appending patterns to `out`.
    - The [`visit`](#SchemaConvertervisit) function is called with the root of the trie to populate `out` with the appropriate patterns.
    - The final pattern is completed by appending closing elements to `out` and the result is returned as a concatenated string.
- **Output**: A string representing a grammar rule that matches any string not in the input list of strings.
- **Functions called**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._not_strings.TrieNode`](#cpp/examples/json_schema_to_grammarSchemaConverter._not_strings.TrieNode)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._not_strings.TrieNode.insert`](#SchemaConverter_not_strings.TrieNode.insert)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._add_primitive`](#SchemaConverter_add_primitive)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.visit`](#SchemaConvertervisit)
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.\_add\_rule<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter._add_rule}} -->
The `_add_rule` method adds a rule to the `_rules` dictionary, ensuring unique keys by appending an index if necessary, and returns the key used.
- **Inputs**:
    - `name`: The name of the rule to be added, which may contain invalid characters.
    - `rule`: The rule content to be associated with the given name.
- **Control Flow**:
    - The method first sanitizes the `name` by replacing invalid characters with a hyphen to create `esc_name`.
    - It checks if `esc_name` is not already in `_rules` or if the existing rule for `esc_name` is the same as the new `rule`.
    - If the above condition is true, it sets `key` to `esc_name`.
    - If the condition is false, it enters a loop to find a unique key by appending an index to `esc_name` until a unique key is found or the existing rule matches the new rule.
    - The rule is then added to `_rules` with the determined `key`.
    - Finally, the method returns the `key` used to store the rule.
- **Output**: The method returns the key under which the rule was stored in the `_rules` dictionary.
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.resolve\_refs<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter.resolve_refs}} -->
The `resolve_refs` method resolves all `$ref` fields in a given JSON schema, fetching remote schemas if necessary, and updates the schema with absolute reference URLs while populating a reference dictionary with the resolved subschemas.
- **Inputs**:
    - `schema`: A dictionary representing the JSON schema that may contain `$ref` fields to be resolved.
    - `url`: A string representing the base URL used to resolve relative `$ref` fields within the schema.
- **Control Flow**:
    - Defines a nested function [`visit`](#SchemaConvertervisit) to recursively traverse the schema.
    - Checks if the current node is a list and recursively visits each element if true.
    - Checks if the current node is a dictionary and looks for a `$ref` field.
    - If a `$ref` is found and not already resolved, it checks if the reference is a remote URL or a local reference.
    - For remote URLs, it asserts that fetching is allowed, fetches the schema using `requests`, and recursively resolves it.
    - For local references, it constructs an absolute URL and updates the `$ref` field.
    - Traverses the reference path to resolve the target subschema and updates the `_refs` dictionary with the resolved subschema.
    - If no `$ref` is found, it recursively visits all values in the dictionary.
- **Output**: Returns the schema with all `$ref` fields resolved to their respective subschemas.
- **Functions called**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.visit`](#SchemaConvertervisit)
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.\_generate\_union\_rule<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter._generate_union_rule}} -->
The `_generate_union_rule` method generates a union rule string by visiting each alternative schema and joining their results with a '|' separator.
- **Inputs**:
    - `name`: A string representing the base name for the rule being generated.
    - `alt_schemas`: A list of alternative schema dictionaries to be processed and combined into a union rule.
- **Control Flow**:
    - Iterates over the `alt_schemas` list using `enumerate` to get both the index and the schema.
    - For each `alt_schema`, it calls the [`visit`](#SchemaConvertervisit) method with the schema and a constructed name that includes the base `name` and the index `i`.
    - Joins the results of the [`visit`](#SchemaConvertervisit) method calls with a ' | ' separator to form the final union rule string.
- **Output**: A string representing the union of the rules generated from the alternative schemas, separated by ' | '.
- **Functions called**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.visit`](#SchemaConvertervisit)
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.\_visit\_pattern<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter._visit_pattern}} -->
The `_visit_pattern` method transforms a regular expression pattern into a GBNF rule, handling specific regex features and translating them into grammar rules.
- **Inputs**:
    - `pattern`: A string representing a regular expression pattern that must start with '^' and end with '$'.
    - `name`: A string representing the name of the rule to be generated from the pattern.
- **Control Flow**:
    - The method asserts that the pattern starts with '^' and ends with '$', then strips these characters.
    - Initializes an index `i` and a dictionary `sub_rule_ids` to track sub-rules.
    - Defines a helper function `to_rule` to convert a tuple of text and a boolean into a rule string.
    - Defines a nested function `transform` to parse the pattern and convert it into a sequence of rules, handling literals, groups, character classes, and quantifiers.
    - Iterates over the pattern, character by character, to build a sequence of rules, handling special characters like '.', '(', ')', '[', ']', '|', '*', '+', '?', and '{x,y}' quantifiers.
    - For each special character, the method applies specific logic to transform it into a GBNF rule, using helper functions like `get_dot` and `join_seq`.
    - Handles quantifiers by creating sub-rules if necessary to keep the output lean.
    - Returns the final rule by calling `self._add_rule` with the transformed pattern.
- **Output**: Returns the name of the rule added to the `_rules` dictionary, which represents the transformed GBNF rule.
- **Functions called**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._add_rule`](#SchemaConverter_add_rule)
    - [`llama.cpp/examples/json_schema_to_grammar._build_repetition`](#cpp/examples/json_schema_to_grammar_build_repetition)
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.\_resolve\_ref<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter._resolve_ref}} -->
The `_resolve_ref` method resolves a reference by checking if it is already resolved or being resolved, and if not, it resolves the reference and returns the resolved name.
- **Inputs**:
    - `ref`: A string representing the reference to be resolved.
- **Control Flow**:
    - Extract the reference name from the input `ref` by splitting the string on '/' and taking the last element.
    - Check if the reference name is not in `_rules` and the reference is not in `_refs_being_resolved`.
    - If the reference is not already being resolved, add it to `_refs_being_resolved`.
    - Retrieve the resolved reference from `_refs` using the input `ref`.
    - Call the [`visit`](#SchemaConvertervisit) method with the resolved reference and the reference name to process it further.
    - Remove the reference from `_refs_being_resolved` after processing.
    - Return the reference name.
- **Output**: The method returns the resolved reference name as a string.
- **Functions called**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.visit`](#SchemaConvertervisit)
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.\_generate\_constant\_rule<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter._generate_constant_rule}} -->
The `_generate_constant_rule` method converts a given value into a JSON-formatted string and then formats it as a grammar literal.
- **Inputs**:
    - `value`: The input value to be converted into a JSON-formatted string and then formatted as a grammar literal.
- **Control Flow**:
    - The method takes the input value and converts it into a JSON-formatted string using `json.dumps(value)`.
    - It then calls the [`_format_literal`](#SchemaConverter_format_literal) method with the JSON-formatted string to escape any special characters and format it as a grammar literal.
    - The formatted literal is returned as the output of the method.
- **Output**: A string representing the input value formatted as a grammar literal.
- **Functions called**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._format_literal`](#SchemaConverter_format_literal)
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.visit<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter.visit}} -->
The `visit` method processes a JSON schema to generate grammar rules based on the schema's structure and properties.
- **Inputs**:
    - `schema`: A dictionary representing the JSON schema to be processed.
    - `name`: A string representing the name of the rule to be generated, which is adjusted if it is a reserved name.
- **Control Flow**:
    - Retrieve the schema type and format from the schema dictionary.
    - Determine the rule name based on whether the provided name is a reserved name or not.
    - Check if the schema contains a '$ref' key and resolve it if present, adding the rule using the resolved reference.
    - Handle 'oneOf' or 'anyOf' keys by generating a union rule for the alternatives and adding it as a rule.
    - If the schema type is a list, generate a union rule for each type and add it as a rule.
    - For 'const' keys, generate a constant rule and add it with a space suffix.
    - For 'enum' keys, generate a rule with all possible values joined by '|' and add it with a space suffix.
    - If the schema type is 'object' or unspecified and has 'properties' or 'additionalProperties', build an object rule and add it.
    - If the schema type is 'object' or unspecified and has 'allOf', combine properties from all components and add the rule.
    - For 'array' types with 'items' or 'prefixItems', generate rules for each item and add them as a rule.
    - For 'string' types with 'pattern', transform the pattern into a grammar rule and add it.
    - For 'string' types with a UUID format, add a primitive rule for UUIDs.
    - For 'string' types with a specific format, add a primitive rule based on the format.
    - For 'string' types with 'minLength' or 'maxLength', generate a rule for character repetition and add it.
    - For 'integer' types with min/max constraints, generate a rule for the range and add it.
    - If the schema type is 'object' or the schema is empty, add a primitive rule for objects.
    - For any other schema type, assert that it is recognized and add a primitive rule based on the type.
- **Output**: Returns the name of the rule added to the internal rules dictionary.
- **Functions called**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._add_rule`](#SchemaConverter_add_rule)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._resolve_ref`](#SchemaConverter_resolve_ref)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._generate_union_rule`](#SchemaConverter_generate_union_rule)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._generate_constant_rule`](#SchemaConverter_generate_constant_rule)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._build_object_rule`](#SchemaConverter_build_object_rule)
    - [`llama.cpp/examples/json_schema_to_grammar._build_repetition`](#cpp/examples/json_schema_to_grammar_build_repetition)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._visit_pattern`](#SchemaConverter_visit_pattern)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._add_primitive`](#SchemaConverter_add_primitive)
    - [`llama.cpp/examples/json_schema_to_grammar._generate_min_max_int`](#cpp/examples/json_schema_to_grammar_generate_min_max_int)
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.\_add\_primitive<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter._add_primitive}} -->
The `_add_primitive` method adds a primitive rule to the schema converter's rule set, ensuring all dependencies are also added recursively.
- **Inputs**:
    - `name`: A string representing the name of the primitive rule to be added.
    - `rule`: An instance of `BuiltinRule` containing the content and dependencies of the rule to be added.
- **Control Flow**:
    - Call [`_add_rule`](#SchemaConverter_add_rule) with the provided name and rule content to add the rule to the internal rules dictionary and store the returned key in `n`.
    - Iterate over each dependency in `rule.deps`.
    - For each dependency, retrieve the corresponding rule from `PRIMITIVE_RULES` or `STRING_FORMAT_RULES`.
    - Assert that the dependency rule exists, raising an error if not.
    - If the dependency is not already in the internal rules dictionary, recursively call `_add_primitive` to add the dependency rule.
- **Output**: Returns the key `n` which is the name under which the rule was added to the internal rules dictionary.
- **Functions called**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._add_rule`](#SchemaConverter_add_rule)
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.\_build\_object\_rule<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter._build_object_rule}} -->
The `_build_object_rule` method constructs a grammar rule for JSON objects based on specified properties, required fields, and additional properties.
- **Inputs**:
    - `properties`: A list of tuples where each tuple contains a property name and its corresponding schema.
    - `required`: A set of property names that are required in the object.
    - `name`: A string representing the base name for the rule being constructed.
    - `additional_properties`: An optional parameter that can be a boolean or a schema, indicating whether additional properties are allowed and their schema if applicable.
- **Control Flow**:
    - Initialize `prop_order` from the instance variable `_prop_order` to determine the order of properties.
    - Sort the properties based on their order in `prop_order` and their original order.
    - Iterate over each property to generate a rule name and add it to `prop_kv_rule_names`.
    - Separate properties into `required_props` and `optional_props` based on the `required` set.
    - If `additional_properties` is specified and not False, create rules for additional properties and append '*' to `optional_props`.
    - Construct the initial part of the rule with required properties.
    - If there are optional properties, construct a recursive rule to handle them, allowing for optional and additional properties.
    - Finalize the rule by appending the closing brace and space.
- **Output**: Returns a string representing the constructed grammar rule for the JSON object.
- **Functions called**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.visit`](#SchemaConvertervisit)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._add_rule`](#SchemaConverter_add_rule)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._format_literal`](#SchemaConverter_format_literal)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._add_primitive`](#SchemaConverter_add_primitive)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._not_strings`](#SchemaConverter_not_strings)
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)


---
#### SchemaConverter\.format\_grammar<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter.format_grammar}} -->
The `format_grammar` method formats and returns the grammar rules stored in the `_rules` dictionary as a string, with each rule on a new line.
- **Inputs**: None
- **Control Flow**:
    - The method accesses the `_rules` dictionary of the class instance.
    - It sorts the items of the `_rules` dictionary by their keys.
    - For each sorted key-value pair, it formats them into a string in the format 'name ::= rule'.
    - It joins all formatted strings with newline characters to create a single string output.
- **Output**: A string representing the formatted grammar rules, with each rule on a new line.
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)  (Base Class)



---
### TrieNode<!-- {{#class:llama.cpp/examples/json_schema_to_grammar.SchemaConverter._not_strings.TrieNode}} -->
- **Members**:
    - `children`: A dictionary that stores child TrieNode objects for each character.
    - `is_end_of_string`: A boolean flag indicating if the node represents the end of a string.
- **Description**: The TrieNode class represents a node in a trie data structure, which is used to store strings efficiently. Each node contains a dictionary of children nodes, where each key is a character and the value is another TrieNode, allowing for the construction of a tree-like structure. The is_end_of_string flag is used to mark the end of a valid string within the trie, enabling the trie to differentiate between prefixes and complete strings.
- **Methods**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._not_strings.TrieNode.__init__`](#SchemaConverter_not_strings.TrieNode.__init__)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._not_strings.TrieNode.insert`](#SchemaConverter_not_strings.TrieNode.insert)

**Methods**

---
#### TrieNode\.\_\_init\_\_<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter._not_strings.TrieNode.__init__}} -->
The `__init__` method initializes a `TrieNode` object with an empty dictionary for children and a boolean flag indicating the end of a string.
- **Inputs**: None
- **Control Flow**:
    - The method initializes the `children` attribute as an empty dictionary.
    - The method sets the `is_end_of_string` attribute to `False`.
- **Output**: The method does not return any value; it initializes the instance attributes of a `TrieNode` object.
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._not_strings.TrieNode`](#cpp/examples/json_schema_to_grammarSchemaConverter._not_strings.TrieNode)  (Base Class)


---
#### TrieNode\.insert<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.SchemaConverter._not_strings.TrieNode.insert}} -->
The `insert` method adds a string to the Trie by creating nodes for each character if they don't already exist.
- **Inputs**:
    - `string`: The string to be inserted into the Trie, where each character will be represented as a node.
- **Control Flow**:
    - Initialize the current node to the root of the Trie (self).
    - Iterate over each character in the input string.
    - For each character, check if it exists as a child of the current node; if not, create a new TrieNode for it.
    - Move the current node to the child node corresponding to the current character.
    - After processing all characters, mark the current node as the end of a string.
- **Output**: The method does not return any value; it modifies the Trie in place by adding nodes for the input string.
- **See also**: [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter._not_strings.TrieNode`](#cpp/examples/json_schema_to_grammarSchemaConverter._not_strings.TrieNode)  (Base Class)



# Functions

---
### \_build\_repetition<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar._build_repetition}} -->
The `_build_repetition` function generates a string representation of a repetition pattern for a given item rule, with optional separators, based on specified minimum and maximum occurrences.
- **Inputs**:
    - `item_rule`: A string representing the rule for the item to be repeated.
    - `min_items`: An integer specifying the minimum number of times the item should appear.
    - `max_items`: An integer or None specifying the maximum number of times the item can appear; if None, there is no upper limit.
    - `separator_rule`: An optional string representing the rule for the separator between repeated items; defaults to None if not provided.
- **Control Flow**:
    - Check if `max_items` is 0, return an empty string if true.
    - Check if `min_items` is 0 and `max_items` is 1, return the item rule followed by '?' if true.
    - If `separator_rule` is not provided, determine the repetition pattern based on `min_items` and `max_items` and return the appropriate string representation.
    - If `separator_rule` is provided, recursively build the repetition pattern with the separator included, adjusting `min_items` and `max_items` accordingly.
    - Return the constructed repetition pattern, optionally wrapped in parentheses and followed by '?' if `min_items` is 0.
- **Output**: A string representing the repetition pattern for the item rule, formatted according to the specified minimum and maximum occurrences, and optionally including a separator.


---
### \_generate\_min\_max\_int<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar._generate_min_max_int}} -->
The `_generate_min_max_int` function generates a regular expression pattern to match integers within a specified minimum and maximum range.
- **Inputs**:
    - `min_value`: An optional integer representing the minimum value of the range.
    - `max_value`: An optional integer representing the maximum value of the range.
    - `out`: A list to which the generated regular expression components are appended.
    - `decimals_left`: An integer representing the number of decimal places left to consider, defaulting to 16.
    - `top_level`: A boolean indicating if the function is being called at the top level, defaulting to True.
- **Control Flow**:
    - Check if `min_value` and `max_value` are provided and set flags `has_min` and `has_max` accordingly.
    - Define helper functions `digit_range`, `more_digits`, and `uniform_range` to assist in building parts of the regular expression.
    - If both `min_value` and `max_value` are provided, handle negative ranges, adjust for negative minimums, and generate patterns for each digit length between the minimum and maximum values.
    - If only `min_value` is provided, handle negative values, zero, single-digit, and multi-digit cases to generate appropriate patterns.
    - If only `max_value` is provided, handle positive and negative cases to generate patterns.
    - Raise a `RuntimeError` if neither `min_value` nor `max_value` is provided.
- **Output**: The function appends components of a regular expression pattern to the `out` list, which represents the range of integers specified by `min_value` and `max_value`.


---
### main<!-- {{#callable:llama.cpp/examples/json_schema_to_grammar.main}} -->
The `main` function parses command-line arguments to generate a grammar from a JSON schema, supporting a subset of JSON schema features.
- **Inputs**:
    - `args_in`: Optional list of command-line arguments to parse; if None, defaults to `sys.argv`.
- **Control Flow**:
    - An `ArgumentParser` is created to handle command-line arguments with descriptions for each option.
    - The `--prop-order` argument is parsed to determine the order of properties in the generated grammar.
    - The `--allow-fetch`, `--dotall`, and `--raw-pattern` flags are parsed to control schema fetching, regex behavior, and pattern handling, respectively.
    - The `schema` argument is parsed to determine the source of the JSON schema, which can be a URL, a file, or standard input.
    - If the schema is a URL, it is fetched using `requests.get`; if it is '-', it is read from standard input; otherwise, it is read from a file.
    - A [`SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter) object is instantiated with the parsed arguments to handle schema conversion.
    - The schema is resolved for references using `SchemaConverter.resolve_refs`.
    - The schema is visited and processed using `SchemaConverter.visit`.
    - The generated grammar is printed using `SchemaConverter.format_grammar`.
- **Output**: The function outputs the generated grammar to the standard output, which is a string representation of the grammar rules derived from the JSON schema.
- **Functions called**:
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter`](#cpp/examples/json_schema_to_grammarSchemaConverter)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.resolve_refs`](#SchemaConverterresolve_refs)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.visit`](#SchemaConvertervisit)
    - [`llama.cpp/examples/json_schema_to_grammar.SchemaConverter.format_grammar`](#SchemaConverterformat_grammar)


