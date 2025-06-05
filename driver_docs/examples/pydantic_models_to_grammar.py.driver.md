# Purpose
This Python code is a comprehensive utility for generating Generalized Backus-Naur Form (GBNF) grammar and documentation from Pydantic models. It provides a broad functionality that includes mapping Pydantic data types to GBNF, generating GBNF rules for various data types, and creating dynamic Pydantic models from dictionaries or functions. The code is structured to handle complex data types, including lists, sets, unions, and custom classes, and it supports generating both markdown and text documentation for the models.

Key components of the code include the `PydanticDataType` enum, which defines the supported data types, and functions like [`map_pydantic_type_to_gbnf`](#cpp/examples/pydantic_models_to_grammarmap_pydantic_type_to_gbnf), [`generate_gbnf_grammar`](#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar), and [`generate_markdown_documentation`](#cpp/examples/pydantic_models_to_grammargenerate_markdown_documentation), which are responsible for converting Pydantic models into GBNF grammar and documentation. The code also includes utilities for saving the generated grammar and documentation to files, and for creating dynamic models from dictionaries, which can be particularly useful for applications that need to dynamically generate and validate data structures. This code is designed to be used as a library, providing a public API for generating grammar and documentation from Pydantic models, and it is intended to be imported and used in other Python scripts or applications.
# Imports and Dependencies

---
- `__future__.annotations`
- `inspect`
- `json`
- `re`
- `copy.copy`
- `enum.Enum`
- `inspect.getdoc`
- `inspect.isclass`
- `typing.TYPE_CHECKING`
- `typing.Any`
- `typing.Callable`
- `typing.List`
- `typing.Optional`
- `typing.Union`
- `typing.get_args`
- `typing.get_origin`
- `typing.get_type_hints`
- `docstring_parser.parse`
- `pydantic.BaseModel`
- `pydantic.create_model`
- `types.GenericAlias`
- `typing._GenericAlias`


# Classes

---
### PydanticDataType<!-- {{#class:llama.cpp/examples/pydantic_models_to_grammar.PydanticDataType}} -->
- **Members**:
    - `STRING`: Represents a string data type.
    - `TRIPLE_QUOTED_STRING`: Represents a triple quoted string data type.
    - `MARKDOWN_CODE_BLOCK`: Represents a markdown code block data type.
    - `BOOLEAN`: Represents a boolean data type.
    - `INTEGER`: Represents an integer data type.
    - `FLOAT`: Represents a float data type.
    - `OBJECT`: Represents an object data type.
    - `ARRAY`: Represents an array data type.
    - `ENUM`: Represents an enum data type.
    - `ANY`: Represents any data type.
    - `NULL`: Represents a null data type.
    - `CUSTOM_CLASS`: Represents a custom class data type.
    - `CUSTOM_DICT`: Represents a custom dictionary data type.
    - `SET`: Represents a set data type.
- **Description**: The `PydanticDataType` class is an enumeration that defines various data types supported by the grammar generator, including basic types like string, boolean, integer, and float, as well as more complex types like object, array, enum, and custom class. It serves as a standardized way to represent these data types within the system, facilitating the mapping and conversion of Pydantic types to grammar-based representations.
- **Inherits From**:
    - `Enum`


# Functions

---
### map\_pydantic\_type\_to\_gbnf<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.map_pydantic_type_to_gbnf}} -->
The function `map_pydantic_type_to_gbnf` maps a Pydantic type to a corresponding GBNF (Generalized Backus-Naur Form) string representation.
- **Inputs**:
    - `pydantic_type`: A Pydantic type, which is a type hint that can be any Python type.
- **Control Flow**:
    - Retrieve the origin type of the provided Pydantic type using `get_origin`.
    - If the origin type is `None`, set it to the provided Pydantic type.
    - Check if the origin type is a subclass of `str`, `bool`, `int`, `float`, or `Enum`, and return the corresponding GBNF string value from `PydanticDataType`.
    - If the origin type is a subclass of `BaseModel`, return a formatted model and field name.
    - If the origin type is `list` or `set`, recursively map the element type to GBNF and append '-list' or '-set' to the result.
    - If the origin type is `Union`, map each type in the union to GBNF and join them with '-or-'.
    - If the origin type is `Optional`, map the element type to GBNF and prepend 'optional-'.
    - If the origin type is a class, return a custom class GBNF string with the formatted model and field name.
    - If the origin type is `dict`, map the key and value types to GBNF and format them into a custom dictionary GBNF string.
    - If none of the conditions match, return 'unknown'.
- **Output**: A string representing the GBNF equivalent of the provided Pydantic type.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name)


---
### format\_model\_and\_field\_name<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.format_model_and_field_name}} -->
The function `format_model_and_field_name` formats a given model name into a lowercase, hyphen-separated string based on its camel case structure.
- **Inputs**:
    - `model_name`: A string representing the name of the model to be formatted.
- **Control Flow**:
    - The function uses a regular expression to split the `model_name` into parts based on uppercase letters, capturing each uppercase letter followed by any number of lowercase letters.
    - If the `parts` list is empty, indicating that the `model_name` does not contain any uppercase letters, the function returns the `model_name` converted to lowercase with underscores replaced by hyphens.
    - If the `parts` list is not empty, the function joins the parts into a single string, converting each part to lowercase and replacing underscores with hyphens, and returns this formatted string.
- **Output**: A string that is the formatted version of the input `model_name`, with camel case parts separated by hyphens and converted to lowercase.


---
### generate\_list\_rule<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_list_rule}} -->
The `generate_list_rule` function creates a GBNF rule for a list of elements of a specified type.
- **Inputs**:
    - `element_type`: The type of the elements in the list, such as 'string'.
- **Control Flow**:
    - The function starts by determining the rule name by appending '-list' to the GBNF representation of the element type.
    - It then maps the element type to its corresponding GBNF rule using the [`map_pydantic_type_to_gbnf`](#cpp/examples/pydantic_models_to_grammarmap_pydantic_type_to_gbnf) function.
    - A GBNF rule string is constructed for a list, which includes the element type and allows for multiple elements separated by commas within square brackets.
    - Finally, the constructed GBNF rule string is returned.
- **Output**: A string representing the GBNF rule for a list of the specified element type.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.map_pydantic_type_to_gbnf`](#cpp/examples/pydantic_models_to_grammarmap_pydantic_type_to_gbnf)


---
### get\_members\_structure<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.get_members_structure}} -->
The `get_members_structure` function generates a grammar rule string for a given class and rule name, handling different class types like Enums and Pydantic models.
- **Inputs**:
    - `cls`: The class for which the grammar rule is being generated; it can be an Enum, a Pydantic model, or a custom class.
    - `rule_name`: The name of the rule to be used in the generated grammar string.
- **Control Flow**:
    - Check if the class is a subclass of Enum; if true, generate a grammar rule string for Enum members.
    - If the class has annotations, generate a grammar rule string using the class's type hints, excluding 'self'.
    - If the rule name is 'custom-class-any', return a simple rule string with 'value'.
    - Otherwise, inspect the class's __init__ method to generate a grammar rule string using its parameters, excluding 'self' and parameters without annotations.
- **Output**: A string representing the grammar rule for the given class and rule name.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.map_pydantic_type_to_gbnf`](#cpp/examples/pydantic_models_to_grammarmap_pydantic_type_to_gbnf)


---
### regex\_to\_gbnf<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.regex_to_gbnf}} -->
The function `regex_to_gbnf` translates a basic regex pattern into a GBNF rule.
- **Inputs**:
    - `regex_pattern`: A string representing the regex pattern to be translated into a GBNF rule.
- **Control Flow**:
    - Initialize `gbnf_rule` with the value of `regex_pattern`.
    - Replace occurrences of '\\d' in `gbnf_rule` with '[0-9]'.
    - Replace occurrences of '\\s' in `gbnf_rule` with '[ \t\n]'.
    - Return the modified `gbnf_rule`.
- **Output**: A string representing the GBNF rule equivalent of the input regex pattern.


---
### generate\_gbnf\_integer\_rules<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_integer_rules}} -->
The `generate_gbnf_integer_rules` function generates GBNF rules for integers based on specified maximum and minimum digit constraints.
- **Inputs**:
    - `max_digit`: The maximum number of digits allowed for the integer; defaults to None if not specified.
    - `min_digit`: The minimum number of digits required for the integer; defaults to None if not specified.
- **Control Flow**:
    - Initialize an empty list `additional_rules` to store additional GBNF rules.
    - Set the base rule identifier `integer_rule` to 'integer-part'.
    - If `max_digit` is provided, append '-max' followed by `max_digit` to `integer_rule`.
    - If `min_digit` is provided, append '-min' followed by `min_digit` to `integer_rule`.
    - If either `max_digit` or `min_digit` is provided, proceed to construct the integer rule part.
    - Initialize an empty string `integer_rule_part` to build the rule.
    - If `min_digit` is specified, append '[0-9] ' repeated `min_digit` times to `integer_rule_part`.
    - If `max_digit` is specified, calculate the number of optional digits and append '[0-9]? ' for each optional digit to `integer_rule_part`.
    - Trim any trailing spaces from `integer_rule_part` and, if not empty, append the complete rule to `additional_rules`.
- **Output**: Returns a tuple containing the `integer_rule` string and a list of `additional_rules` strings.


---
### generate\_gbnf\_float\_rules<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_float_rules}} -->
The function generates Generalized Backus-Naur Form (GBNF) rules for floating-point numbers based on specified constraints for integer and fractional parts.
- **Inputs**:
    - `max_digit`: Maximum number of digits allowed in the integer part of the float (default: None).
    - `min_digit`: Minimum number of digits required in the integer part of the float (default: None).
    - `max_precision`: Maximum number of digits allowed in the fractional part of the float (default: None).
    - `min_precision`: Minimum number of digits required in the fractional part of the float (default: None).
- **Control Flow**:
    - Initialize an empty list `additional_rules` to store the generated rules.
    - Construct the `integer_part_rule` string based on `max_digit` and `min_digit` constraints.
    - Construct the `fractional_part_rule` string based on `max_precision` and `min_precision` constraints, and append the corresponding rule to `additional_rules`.
    - Construct the `float_rule` string using the integer and fractional part rules, and append it to `additional_rules`.
    - If `max_digit` or `min_digit` is specified, generate the integer part rule definition and append it to `additional_rules`.
- **Output**: A tuple containing the float rule as a string and a list of additional rules.


---
### generate\_gbnf\_rule\_for\_type<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_rule_for_type}} -->
The `generate_gbnf_rule_for_type` function generates a GBNF rule for a given field type, considering its characteristics and dependencies.
- **Inputs**:
    - `model_name`: Name of the model.
    - `field_name`: Name of the field.
    - `field_type`: Type of the field.
    - `is_optional`: Boolean indicating whether the field is optional.
    - `processed_models`: List of processed models to avoid duplication.
    - `created_rules`: List of created rules to track generated rules.
    - `field_info`: Additional information about the field, optional.
- **Control Flow**:
    - Initialize an empty list `rules` to store generated rules.
    - Format the `field_name` using [`format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name).
    - Map the `field_type` to a GBNF type using [`map_pydantic_type_to_gbnf`](#cpp/examples/pydantic_models_to_grammarmap_pydantic_type_to_gbnf).
    - Determine the `origin_type` of the `field_type`.
    - If `origin_type` is a subclass of `BaseModel`, generate nested model rules and update `gbnf_type` and `rules`.
    - If `origin_type` is a subclass of `Enum`, generate an enum rule and update `gbnf_type` and `rules`.
    - If `origin_type` is a list or set, recursively generate rules for the element type and create an array rule.
    - If `gbnf_type` starts with 'custom-class-', append the member structure to `rules`.
    - If `gbnf_type` starts with 'custom-dict-', recursively generate rules for key and value types and create a dictionary rule.
    - If `gbnf_type` starts with 'union-', handle each union type separately, generate rules, and create a union grammar rule.
    - If `origin_type` is a string, handle special cases like triple-quoted strings, markdown code blocks, or regex patterns.
    - If `origin_type` is a float or int with additional field info, generate rules considering precision or digit constraints.
    - Return the `gbnf_type` and `rules`.
- **Output**: A tuple containing the GBNF type as a string and a list of additional rules as strings.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name)
    - [`llama.cpp/examples/pydantic_models_to_grammar.map_pydantic_type_to_gbnf`](#cpp/examples/pydantic_models_to_grammarmap_pydantic_type_to_gbnf)
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar`](#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar)
    - [`llama.cpp/examples/pydantic_models_to_grammar.get_members_structure`](#cpp/examples/pydantic_models_to_grammarget_members_structure)
    - [`llama.cpp/examples/pydantic_models_to_grammar.regex_to_gbnf`](#cpp/examples/pydantic_models_to_grammarregex_to_gbnf)
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_float_rules`](#cpp/examples/pydantic_models_to_grammargenerate_gbnf_float_rules)
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_integer_rules`](#cpp/examples/pydantic_models_to_grammargenerate_gbnf_integer_rules)


---
### generate\_gbnf\_grammar<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar}} -->
The `generate_gbnf_grammar` function generates a GBnF grammar for a given Pydantic model, ensuring no duplicate or recursive rules are created.
- **Inputs**:
    - `model`: A Pydantic model class to generate the grammar for, which must be a subclass of BaseModel.
    - `processed_models`: A set of already processed models to prevent infinite recursion.
    - `created_rules`: A dictionary containing already created rules to prevent duplicates.
- **Control Flow**:
    - Check if the model is already in the processed_models set; if so, return an empty list and False.
    - Add the model to the processed_models set and format the model name.
    - Determine the model fields based on whether the model is a Pydantic model or not.
    - Iterate over each field in the model fields to generate rules for each field type using the [`generate_gbnf_rule_for_type`](#cpp/examples/pydantic_models_to_grammargenerate_gbnf_rule_for_type) function.
    - Check if the generated rule name is 'markdown_code_block' or 'triple_quoted_string' to set flags accordingly.
    - If the rule name is not a special string, add it to created_rules and append the rule to model_rule_parts and nested_rules.
    - Join the model_rule_parts to form the complete model rule and append special string rules if necessary.
    - Return the list of all rules and a boolean indicating if a special string is present.
- **Output**: A tuple containing a list of GBnF grammar rules in string format and a boolean indicating if a special string (markdown or triple quoted) is present in the grammar.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name)
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_rule_for_type`](#cpp/examples/pydantic_models_to_grammargenerate_gbnf_rule_for_type)


---
### generate\_gbnf\_grammar\_from\_pydantic\_models<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar_from_pydantic_models}} -->
This function generates a GBNF grammar string from a list of Pydantic models, optionally wrapping it with an outer object rule.
- **Inputs**:
    - `models`: A list of Pydantic model classes from which to generate the GBNF grammar.
    - `outer_object_name`: An optional string specifying the name of an outer object for the GBNF grammar; if None, no outer object is generated.
    - `outer_object_content`: An optional string specifying the content for the outer rule in the GBNF grammar.
    - `list_of_outputs`: A boolean indicating whether the output should be a list of objects.
- **Control Flow**:
    - Initialize a set to track processed models and lists to store all rules and created rules.
    - If 'outer_object_name' is None, iterate over each model in 'models' to generate GBNF rules and append them to 'all_rules'.
    - If 'list_of_outputs' is True, define a root rule for a list of grammar models; otherwise, define a root rule for a single grammar model.
    - Insert the root rule at the beginning of 'all_rules' and return the joined rules as a string.
    - If 'outer_object_name' is not None, define a root rule and a model rule using 'outer_object_name' and 'outer_object_content'.
    - Iterate over each model in 'models' to generate GBNF rules, appending them to 'all_rules', and handle special string cases.
    - Insert the combined root, model, and grammar model rules at the beginning of 'all_rules' and return the joined rules as a string.
- **Output**: A string representing the generated GBNF grammar.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar`](#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar)
    - [`llama.cpp/examples/pydantic_models_to_grammar.format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name)


---
### get\_primitive\_grammar<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.get_primitive_grammar}} -->
The `get_primitive_grammar` function generates and returns a GBNF primitive grammar string based on the provided GBNF grammar string.
- **Inputs**:
    - `grammar`: A string containing the GBNF grammar.
- **Control Flow**:
    - Initialize an empty list `type_list` to store types based on the grammar string.
    - Check for specific substrings ('string-list', 'boolean-list', 'integer-list', 'float-list') in the `grammar` and append corresponding types (str, bool, int, float) to `type_list`.
    - Generate additional grammar rules for each type in `type_list` using [`generate_list_rule`](#cpp/examples/pydantic_models_to_grammargenerate_list_rule) and store them in `additional_grammar`.
    - Define a base `primitive_grammar` string containing default GBNF rules for boolean, null, string, whitespace, float, and integer.
    - Check for 'custom-class-any' in `grammar` and append additional rules for value, object, array, and number to `any_block`.
    - Check for 'markdown-code-block' in `grammar` and append markdown code block rules to `markdown_code_block_grammar`.
    - Check for 'triple-quoted-string' in `grammar` and append triple-quoted string rules to `markdown_code_block_grammar`.
    - Return a concatenated string of `additional_grammar`, `any_block`, `primitive_grammar`, and `markdown_code_block_grammar`.
- **Output**: A string representing the GBNF primitive grammar.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_list_rule`](#cpp/examples/pydantic_models_to_grammargenerate_list_rule)


---
### generate\_markdown\_documentation<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_markdown_documentation}} -->
The `generate_markdown_documentation` function generates markdown documentation for a list of Pydantic models, including model and field descriptions.
- **Inputs**:
    - `pydantic_models`: A list of Pydantic model classes to document.
    - `model_prefix`: A string prefix for the model section in the documentation, defaulting to 'Model'.
    - `fields_prefix`: A string prefix for the fields section in the documentation, defaulting to 'Fields'.
    - `documentation_with_field_description`: A boolean indicating whether to include field descriptions in the documentation, defaulting to True.
- **Control Flow**:
    - Initialize an empty string `documentation` to accumulate the markdown content.
    - Create a list `pyd_models` containing tuples of each model and a boolean `True` indicating whether to add a prefix.
    - Iterate over each model in `pyd_models`, appending the model name to `documentation` with the appropriate prefix.
    - Retrieve and format the model's docstring if it exists and is different from the base class docstring, appending it to `documentation`.
    - Append the fields section header to `documentation`, using the specified prefix.
    - For each field in the model, determine its type and append its markdown documentation to `documentation`, potentially adding nested models to `pyd_models`.
    - If the model has a `Config` class with a `json_schema_extra` containing an example, append the formatted example to `documentation`.
- **Output**: A string containing the generated markdown documentation for the provided Pydantic models.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.format_multiline_description`](#cpp/examples/pydantic_models_to_grammarformat_multiline_description)
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_field_markdown`](#cpp/examples/pydantic_models_to_grammargenerate_field_markdown)
    - [`llama.cpp/examples/pydantic_models_to_grammar.format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name)


---
### generate\_field\_markdown<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_field_markdown}} -->
The `generate_field_markdown` function generates markdown documentation for a specific field of a Pydantic model, including its type, description, and examples if available.
- **Inputs**:
    - `field_name`: Name of the field to document.
    - `field_type`: Type of the field, which can be any type.
    - `model`: The Pydantic model class that contains the field.
    - `depth`: Indentation depth for the generated markdown, defaulting to 1.
    - `documentation_with_field_description`: Boolean flag indicating whether to include field descriptions in the documentation, defaulting to True.
- **Control Flow**:
    - Initialize the indentation string based on the depth parameter.
    - Retrieve field information from the model using the field name.
    - Determine the field's description if available.
    - Identify the origin type of the field using `get_origin`; if none, use the field type itself.
    - If the field is a list, format the field text to include the element type and append a description if available.
    - If the field is a Union, format the field text to include all possible types and append a description if available.
    - For other types, format the field text to include the field type and append a description if available.
    - If `documentation_with_field_description` is False, return the field text without further processing.
    - If a description is available, append it to the field text.
    - Check if the model has a Config with a JSON schema example and append the example to the field text if available.
    - If the field type is a subclass of BaseModel, recursively generate markdown for its fields and append to the field text.
    - Return the complete field text.
- **Output**: A string containing the generated markdown documentation for the specified field.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name)


---
### format\_json\_example<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.format_json_example}} -->
The `format_json_example` function formats a JSON dictionary into a readable string with specified indentation.
- **Inputs**:
    - `example`: A dictionary representing the JSON example to be formatted.
    - `depth`: An integer specifying the indentation depth for formatting.
- **Control Flow**:
    - Initialize an indentation string based on the provided depth.
    - Start the formatted string with an opening curly brace and a newline.
    - Iterate over each key-value pair in the example dictionary.
    - For each key-value pair, format the value as a string if it is a string, otherwise keep it as is.
    - Append each formatted key-value pair to the formatted string with the specified indentation and a comma.
    - Remove the trailing comma and newline from the formatted string.
    - Close the formatted string with a newline and the indentation followed by a closing curly brace.
    - Return the formatted string.
- **Output**: A string representing the formatted JSON example with the specified indentation.


---
### generate\_text\_documentation<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_text_documentation}} -->
The `generate_text_documentation` function generates text documentation for a list of Pydantic models, including model and field descriptions.
- **Inputs**:
    - `pydantic_models`: A list of Pydantic model classes to document.
    - `model_prefix`: A string prefix for the model section in the documentation, defaulting to 'Model'.
    - `fields_prefix`: A string prefix for the fields section in the documentation, defaulting to 'Fields'.
    - `documentation_with_field_description`: A boolean indicating whether to include field descriptions in the documentation, defaulting to True.
- **Control Flow**:
    - Initialize an empty string `documentation` to accumulate the generated documentation.
    - Create a list `pyd_models` containing tuples of each model and a boolean indicating if a prefix should be added.
    - Iterate over each model in `pyd_models`.
    - For each model, append the model name to `documentation`, prefixed by `model_prefix` if `add_prefix` is True.
    - Retrieve and append the model's class description if it exists and differs from the base class description.
    - Check if the model is a subclass of `BaseModel` and iterate over its fields using `get_type_hints`.
    - For each field, check if it is a list or a union type and append any nested Pydantic models to `pyd_models`.
    - Generate field documentation using [`generate_field_text`](#cpp/examples/pydantic_models_to_grammargenerate_field_text) and append it to `documentation_fields`.
    - If `documentation_fields` is not empty, append it to `documentation`, prefixed by `fields_prefix` if `add_prefix` is True.
    - Check if the model has a `Config` with `json_schema_extra` containing an 'example', and append the example to `documentation`.
- **Output**: A string containing the generated text documentation for the provided Pydantic models.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.format_multiline_description`](#cpp/examples/pydantic_models_to_grammarformat_multiline_description)
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_field_text`](#cpp/examples/pydantic_models_to_grammargenerate_field_text)
    - [`llama.cpp/examples/pydantic_models_to_grammar.format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name)


---
### generate\_field\_text<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_field_text}} -->
The `generate_field_text` function generates text documentation for a field in a Pydantic model, including its type, description, and examples if available.
- **Inputs**:
    - `field_name`: Name of the field as a string.
    - `field_type`: Type of the field, specified as a type hint.
    - `model`: The Pydantic model class to which the field belongs.
    - `depth`: Indentation depth for the generated documentation, defaulting to 1.
    - `documentation_with_field_description`: Boolean flag indicating whether to include field descriptions in the documentation, defaulting to True.
- **Control Flow**:
    - Retrieve the field information from the model using the field name.
    - Determine the field description if available.
    - Check if the field type is a list, union, or a simple type and format the field text accordingly.
    - If `documentation_with_field_description` is True, append the field description to the field text.
    - Check for field-specific examples in the model's configuration and append them to the field text if available.
    - If the field type is a subclass of `BaseModel`, recursively generate documentation for its fields and append it to the field text.
- **Output**: Returns a string containing the generated text documentation for the specified field.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name)


---
### format\_multiline\_description<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.format_multiline_description}} -->
The function formats a multiline string with a specified indentation level.
- **Inputs**:
    - `description`: A string representing the multiline description to be formatted.
    - `indent_level`: An integer specifying the level of indentation to apply to the description.
- **Control Flow**:
    - Calculate the indentation string by repeating four spaces for the given indent level.
    - Replace each newline character in the description with a newline followed by the calculated indentation.
    - Return the newly formatted string.
- **Output**: A string that is the formatted version of the input description with the specified indentation applied.


---
### save\_gbnf\_grammar\_and\_documentation<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.save_gbnf_grammar_and_documentation}} -->
The function saves GBNF grammar and documentation to specified file paths, handling potential I/O errors.
- **Inputs**:
    - `grammar`: A string representing the GBNF grammar to be saved.
    - `documentation`: A string containing the documentation to be saved.
    - `grammar_file_path`: A string specifying the file path where the GBNF grammar should be saved, defaulting to './grammar.gbnf'.
    - `documentation_file_path`: A string specifying the file path where the documentation should be saved, defaulting to './grammar_documentation.md'.
- **Control Flow**:
    - Attempts to open the file at 'grammar_file_path' in write mode.
    - Writes the 'grammar' string concatenated with the result of 'get_primitive_grammar(grammar)' to the file.
    - Prints a success message if the grammar is saved successfully.
    - Catches and prints an error message if an IOError occurs while saving the grammar.
    - Attempts to open the file at 'documentation_file_path' in write mode.
    - Writes the 'documentation' string to the file.
    - Prints a success message if the documentation is saved successfully.
    - Catches and prints an error message if an IOError occurs while saving the documentation.
- **Output**: The function does not return any value; it performs file operations and prints messages to the console.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.get_primitive_grammar`](#cpp/examples/pydantic_models_to_grammarget_primitive_grammar)


---
### remove\_empty\_lines<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.remove_empty_lines}} -->
The `remove_empty_lines` function removes empty lines from a given string.
- **Inputs**:
    - `string`: The input string from which empty lines are to be removed.
- **Control Flow**:
    - The input string is split into lines using the `splitlines()` method, resulting in a list of lines.
    - A list comprehension is used to filter out lines that are empty or contain only whitespace, creating a list of non-empty lines.
    - The non-empty lines are joined back into a single string with newline characters separating them.
    - The resulting string, which has no empty lines, is returned.
- **Output**: A string with all empty lines removed.


---
### generate\_and\_save\_gbnf\_grammar\_and\_documentation<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_and_save_gbnf_grammar_and_documentation}} -->
This function generates GBNF grammar and documentation from a list of Pydantic models and saves them to specified file paths.
- **Inputs**:
    - `pydantic_model_list`: A list of Pydantic model classes to generate the grammar and documentation from.
    - `grammar_file_path`: The file path where the generated GBNF grammar will be saved, defaulting to './generated_grammar.gbnf'.
    - `documentation_file_path`: The file path where the generated documentation will be saved, defaulting to './generated_grammar_documentation.md'.
    - `outer_object_name`: An optional string specifying the outer object name for the GBNF grammar; if None, no outer object will be generated.
    - `outer_object_content`: An optional string specifying the content for the outer rule in the GBNF grammar.
    - `model_prefix`: A string prefix for the model section in the documentation, defaulting to 'Output Model'.
    - `fields_prefix`: A string prefix for the fields section in the documentation, defaulting to 'Output Fields'.
    - `list_of_outputs`: A boolean indicating whether the output is a list of items, defaulting to False.
    - `documentation_with_field_description`: A boolean indicating whether to include field descriptions in the documentation, defaulting to True.
- **Control Flow**:
    - Generate markdown documentation using the provided Pydantic models and specified prefixes.
    - Generate GBNF grammar from the Pydantic models, considering optional outer object name and content, and whether the output is a list.
    - Remove empty lines from the generated grammar string.
    - Save the generated grammar and documentation to the specified file paths using the save_gbnf_grammar_and_documentation function.
- **Output**: The function does not return any value; it performs file operations to save the generated grammar and documentation.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_markdown_documentation`](#cpp/examples/pydantic_models_to_grammargenerate_markdown_documentation)
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar_from_pydantic_models`](#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_from_pydantic_models)
    - [`llama.cpp/examples/pydantic_models_to_grammar.remove_empty_lines`](#cpp/examples/pydantic_models_to_grammarremove_empty_lines)
    - [`llama.cpp/examples/pydantic_models_to_grammar.save_gbnf_grammar_and_documentation`](#cpp/examples/pydantic_models_to_grammarsave_gbnf_grammar_and_documentation)


---
### generate\_gbnf\_grammar\_and\_documentation<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar_and_documentation}} -->
The function generates GBNF grammar and documentation for a list of Pydantic models.
- **Inputs**:
    - `pydantic_model_list`: List of Pydantic model classes.
    - `outer_object_name`: Optional string for the outer object name in the GBNF grammar.
    - `outer_object_content`: Optional string for the content of the outer rule in the GBNF grammar.
    - `model_prefix`: String prefix for the model section in the documentation.
    - `fields_prefix`: String prefix for the fields section in the documentation.
    - `list_of_outputs`: Boolean indicating if the output is a list of items.
    - `documentation_with_field_description`: Boolean indicating whether to include field descriptions in the documentation.
- **Control Flow**:
    - The function first generates markdown documentation for the provided Pydantic models using the [`generate_markdown_documentation`](#cpp/examples/pydantic_models_to_grammargenerate_markdown_documentation) function.
    - It then generates GBNF grammar from the Pydantic models using the [`generate_gbnf_grammar_from_pydantic_models`](#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_from_pydantic_models) function.
    - The generated grammar is combined with primitive grammar using the [`get_primitive_grammar`](#cpp/examples/pydantic_models_to_grammarget_primitive_grammar) function.
    - Empty lines are removed from the combined grammar using the [`remove_empty_lines`](#cpp/examples/pydantic_models_to_grammarremove_empty_lines) function.
    - Finally, the function returns a tuple containing the grammar and documentation strings.
- **Output**: A tuple containing the GBNF grammar string and the documentation string.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_markdown_documentation`](#cpp/examples/pydantic_models_to_grammargenerate_markdown_documentation)
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar_from_pydantic_models`](#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_from_pydantic_models)
    - [`llama.cpp/examples/pydantic_models_to_grammar.remove_empty_lines`](#cpp/examples/pydantic_models_to_grammarremove_empty_lines)
    - [`llama.cpp/examples/pydantic_models_to_grammar.get_primitive_grammar`](#cpp/examples/pydantic_models_to_grammarget_primitive_grammar)


---
### generate\_gbnf\_grammar\_and\_documentation\_from\_dictionaries<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar_and_documentation_from_dictionaries}} -->
This function generates GBNF grammar and documentation from a list of dictionaries representing Pydantic models.
- **Inputs**:
    - `dictionaries`: A list of dictionaries where each dictionary represents a Pydantic model.
    - `outer_object_name`: An optional string specifying the outer object name for the GBNF grammar; if None, no outer object is generated.
    - `outer_object_content`: An optional string specifying the content for the outer rule in the GBNF grammar.
    - `model_prefix`: A string prefix for the model section in the documentation, defaulting to 'Output Model'.
    - `fields_prefix`: A string prefix for the fields section in the documentation, defaulting to 'Output Fields'.
    - `list_of_outputs`: A boolean indicating whether the output is a list of items, defaulting to False.
    - `documentation_with_field_description`: A boolean indicating whether to include field descriptions in the documentation, defaulting to True.
- **Control Flow**:
    - Create dynamic Pydantic models from the provided dictionaries using [`create_dynamic_models_from_dictionaries`](#cpp/examples/pydantic_models_to_grammarcreate_dynamic_models_from_dictionaries) function.
    - Generate markdown documentation for the models using [`generate_markdown_documentation`](#cpp/examples/pydantic_models_to_grammargenerate_markdown_documentation) function.
    - Generate GBNF grammar from the Pydantic models using [`generate_gbnf_grammar_from_pydantic_models`](#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_from_pydantic_models) function.
    - Remove empty lines from the generated grammar and append primitive grammar using [`remove_empty_lines`](#cpp/examples/pydantic_models_to_grammarremove_empty_lines) and [`get_primitive_grammar`](#cpp/examples/pydantic_models_to_grammarget_primitive_grammar) functions.
    - Return the generated grammar and documentation as a tuple.
- **Output**: A tuple containing the GBNF grammar string and the documentation string.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.create_dynamic_models_from_dictionaries`](#cpp/examples/pydantic_models_to_grammarcreate_dynamic_models_from_dictionaries)
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_markdown_documentation`](#cpp/examples/pydantic_models_to_grammargenerate_markdown_documentation)
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar_from_pydantic_models`](#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_from_pydantic_models)
    - [`llama.cpp/examples/pydantic_models_to_grammar.remove_empty_lines`](#cpp/examples/pydantic_models_to_grammarremove_empty_lines)
    - [`llama.cpp/examples/pydantic_models_to_grammar.get_primitive_grammar`](#cpp/examples/pydantic_models_to_grammarget_primitive_grammar)


---
### create\_dynamic\_model\_from\_function<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.create_dynamic_model_from_function}} -->
Creates a dynamic Pydantic model from a function's type hints and adds the function as a 'run' method.
- **Inputs**:
    - `func`: A callable function with type hints from which to create the dynamic model.
- **Control Flow**:
    - Retrieve the function's signature using `inspect.signature`.
    - Parse the function's docstring using `parse` from `docstring_parser`.
    - Iterate over the function's parameters, excluding 'self', to ensure each has a type annotation and a description in the docstring.
    - Raise a `TypeError` if a parameter lacks a type annotation, or a `ValueError` if it lacks a description.
    - Create a dictionary `dynamic_fields` to store parameter names, types, and default values.
    - Use `create_model` from Pydantic to create a dynamic model class with the function's parameters as fields.
    - Assign descriptions from the docstring to the model fields.
    - Define a `run_method_wrapper` function that calls the original function with arguments from the model instance.
    - Add the `run_method_wrapper` as a 'run' method to the dynamic model.
    - Return the dynamic model class.
- **Output**: A dynamic Pydantic model class with the provided function as a 'run' method.


---
### add\_run\_method\_to\_dynamic\_model<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.add_run_method_to_dynamic_model}} -->
Adds a 'run' method to a dynamic Pydantic model using a provided function.
- **Inputs**:
    - `model`: A dynamic Pydantic model class, which is a subclass of BaseModel.
    - `func`: A callable function that will be added as a 'run' method to the model.
- **Control Flow**:
    - Defines an inner function 'run_method_wrapper' that collects model field values into a dictionary 'func_args'.
    - Calls the provided function 'func' with 'func_args' as arguments and returns the result.
    - Uses 'setattr' to add 'run_method_wrapper' as a 'run' method to the model class.
    - Returns the modified model class with the new 'run' method.
- **Output**: Returns the modified Pydantic model class with the added 'run' method.


---
### create\_dynamic\_models\_from\_dictionaries<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.create_dynamic_models_from_dictionaries}} -->
This function generates a list of dynamic Pydantic model classes from a list of dictionaries representing model structures.
- **Inputs**:
    - `dictionaries`: A list of dictionaries, each representing the structure of a model to be converted into a Pydantic model class.
- **Control Flow**:
    - Initialize an empty list `dynamic_models` to store the generated Pydantic model classes.
    - Iterate over each dictionary in the `dictionaries` list.
    - For each dictionary, retrieve the model name using the [`format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name) function, defaulting to an empty string if the 'name' key is not present.
    - Convert the dictionary into a Pydantic model using the [`convert_dictionary_to_pydantic_model`](#cpp/examples/pydantic_models_to_grammarconvert_dictionary_to_pydantic_model) function, passing the dictionary and the formatted model name.
    - Append the generated Pydantic model to the `dynamic_models` list.
    - Return the `dynamic_models` list containing all the generated Pydantic model classes.
- **Output**: A list of generated dynamic Pydantic model classes, each corresponding to a dictionary from the input list.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name)
    - [`llama.cpp/examples/pydantic_models_to_grammar.convert_dictionary_to_pydantic_model`](#cpp/examples/pydantic_models_to_grammarconvert_dictionary_to_pydantic_model)


---
### map\_grammar\_names\_to\_pydantic\_model\_class<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.map_grammar_names_to_pydantic_model_class}} -->
The function `map_grammar_names_to_pydantic_model_class` maps the names of Pydantic models to their corresponding model classes in a dictionary.
- **Inputs**:
    - `pydantic_model_list`: A list of Pydantic model classes to be mapped.
- **Control Flow**:
    - Initialize an empty dictionary `output`.
    - Iterate over each model in `pydantic_model_list`.
    - For each model, format its name using [`format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name) and use it as a key in the `output` dictionary, with the model itself as the value.
    - Return the `output` dictionary.
- **Output**: A dictionary where keys are formatted model names and values are the corresponding Pydantic model classes.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.format_model_and_field_name`](#cpp/examples/pydantic_models_to_grammarformat_model_and_field_name)


---
### json\_schema\_to\_python\_types<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.json_schema_to_python_types}} -->
Converts a JSON schema type to its corresponding Python type.
- **Inputs**:
    - `schema`: A string representing the JSON schema type to be converted.
- **Control Flow**:
    - A dictionary `type_map` is defined to map JSON schema types to Python types.
    - The function returns the Python type corresponding to the input `schema` by looking it up in the `type_map` dictionary.
- **Output**: The Python type corresponding to the input JSON schema type.


---
### list\_to\_enum<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.list_to_enum}} -->
The `list_to_enum` function creates an enumeration from a list of values.
- **Inputs**:
    - `enum_name`: The name of the enumeration to be created.
    - `values`: A list of values that will be used as both the names and values of the enumeration members.
- **Control Flow**:
    - The function uses a dictionary comprehension to create a dictionary where each key-value pair is a value from the input list, effectively mapping each value to itself.
    - The `Enum` class is then called with the provided `enum_name` and the dictionary created from the list of values, resulting in a new enumeration.
- **Output**: An `Enum` object with members corresponding to the values in the input list, each having the same name and value.


---
### convert\_dictionary\_to\_pydantic\_model<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar.convert_dictionary_to_pydantic_model}} -->
The function `convert_dictionary_to_pydantic_model` converts a dictionary representing a model structure into a Pydantic model class.
- **Inputs**:
    - `dictionary`: A dictionary representing the model structure, with keys like 'properties', 'function', and 'parameters' to define the model's fields and structure.
    - `model_name`: An optional string specifying the name of the generated Pydantic model, defaulting to 'CustomModel'.
- **Control Flow**:
    - Initialize an empty dictionary `fields` to store the model's fields.
    - Check if the dictionary contains a 'properties' key, and iterate over its items to define fields.
    - For each field, determine its type and handle special cases like 'object', 'array', and 'enum'.
    - Recursively call `convert_dictionary_to_pydantic_model` for nested objects and arrays to create submodels.
    - Handle 'required' fields by marking non-required fields as optional using `Optional` type.
    - Check for 'function' and 'parameters' keys to adjust the model name and structure accordingly.
    - Create and return a Pydantic model using `create_model` with the defined fields.
- **Output**: The function returns a Pydantic model class generated based on the provided dictionary structure.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.list_to_enum`](#cpp/examples/pydantic_models_to_grammarlist_to_enum)
    - [`llama.cpp/examples/pydantic_models_to_grammar.json_schema_to_python_types`](#cpp/examples/pydantic_models_to_grammarjson_schema_to_python_types)


