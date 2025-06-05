# Purpose
This C++ header file provides functionality for parsing JSON strings, with a specific focus on handling and "healing" partial JSON data. The file defines two main structures: `common_healing_marker` and `common_json`. The `common_healing_marker` structure is used to store markers that help identify where a JSON string was healed, allowing for the reconstruction of the original partial JSON string. The `common_json` structure represents a parsed JSON object, which includes an optional healing marker to indicate the position of healing within the JSON dump string. This is particularly useful for scenarios where JSON outputs from models need to be parsed and potentially reconstructed for further processing.

The file also declares two overloaded functions, `common_json_parse`, which are responsible for parsing JSON strings. These functions can handle partial JSON data by using a healing marker to complete the JSON structure, allowing for successful parsing. The first function takes a JSON string and a healing marker as input, while the second function operates on iterators, advancing them to the end of the input upon successful parsing. This functionality is crucial for applications that require robust handling of incomplete JSON data, such as those involving partial tool calls in OpenAI (OAI) format. The use of the `nlohmann::ordered_json` type from the popular JSON library indicates that the code leverages external libraries to manage JSON data efficiently.
# Imports and Dependencies

---
- `nlohmann/json.hpp`


# Data Structures

---
### common\_healing\_marker<!-- {{#data_structure:common_healing_marker}} -->
- **Type**: `struct`
- **Members**:
    - `marker`: A raw marker string used in the healing process of JSON parsing.
    - `json_dump_marker`: A string used to identify the position in the JSON dump where the original partial JSON string can be reconstructed.
- **Description**: The `common_healing_marker` struct is designed to facilitate the process of healing or completing partial JSON strings during parsing. It contains two string members: `marker`, which serves as a raw marker, and `json_dump_marker`, which is used to identify the position in a JSON dump where the original partial JSON can be reconstructed. This struct is particularly useful in scenarios where JSON outputs from models need to be parsed and potentially modified to form complete JSON objects.


---
### common\_json<!-- {{#data_structure:common_json}} -->
- **Type**: `struct`
- **Members**:
    - `json`: An ordered JSON object from the nlohmann::json library.
    - `healing_marker`: An instance of common_healing_marker that holds information for JSON healing.
- **Description**: The `common_json` struct represents a parsed JSON object using the nlohmann::ordered_json library, with an optional healing marker. The healing marker, encapsulated in the `common_healing_marker` struct, is used to manage and identify positions in a JSON dump string where healing (or completion) of partial JSON data is necessary. This struct is particularly useful in scenarios where JSON data may be incomplete or require reconstruction, such as when dealing with outputs from models that generate partial JSON strings.


