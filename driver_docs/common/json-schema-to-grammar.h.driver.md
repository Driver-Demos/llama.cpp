# Purpose
This C++ header file provides a narrow functionality focused on converting JSON schemas into grammar representations. It includes declarations for a function `json_schema_to_grammar` that transforms a JSON schema into a grammar string, potentially using a specific grammar format if `force_gbnf` is true. Additionally, it defines two structures: `common_grammar_builder`, which offers a set of function pointers for adding rules and schemas and resolving references, and `common_grammar_options`, which includes options like `dotall` to modify grammar building behavior. The `build_grammar` function is declared to facilitate the construction of grammar using a callback function and optional settings. This header is intended to be included in other C++ files where JSON schema to grammar conversion is needed.
# Imports and Dependencies

---
- `nlohmann/json_fwd.hpp`
- `functional`
- `string`


# Data Structures

---
### common\_grammar\_builder<!-- {{#data_structure:common_grammar_builder}} -->
- **Type**: `struct`
- **Members**:
    - `add_rule`: A function to add a grammar rule, taking two strings as input and returning a string.
    - `add_schema`: A function to add a schema, taking a string and a JSON object as input and returning a string.
    - `resolve_refs`: A function to resolve references within a JSON object, modifying it in place.
- **Description**: The `common_grammar_builder` struct is designed to facilitate the construction of grammar rules and schemas, providing function members that allow for the addition of rules and schemas, as well as the resolution of references within JSON objects. It leverages the `std::function` template to define flexible function signatures for these operations, enabling users to customize the behavior of grammar building and reference resolution.


---
### common\_grammar\_options<!-- {{#data_structure:common_grammar_options}} -->
- **Type**: `struct`
- **Members**:
    - `dotall`: A boolean flag indicating whether the dotall mode is enabled, defaulting to false.
- **Description**: The `common_grammar_options` struct is a simple data structure that contains configuration options for grammar processing. It currently includes a single boolean member, `dotall`, which determines if the dotall mode is active, affecting how certain grammar rules are interpreted. This struct is used to pass options to functions that build or manipulate grammar structures.


