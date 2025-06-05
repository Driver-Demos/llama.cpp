# Purpose
This C++ source code file is designed to handle various functionalities related to JSON processing, HTTP communication, and tokenization, particularly in the context of a server application that interacts with language models. The file includes a variety of utility functions and structures that facilitate the parsing and handling of JSON data, tokenization of input prompts, and the management of server events and responses. It leverages several external libraries, such as `nlohmann/json` for JSON manipulation and `cpp-httplib` for HTTP server functionalities, indicating its role in a networked application environment.

The code defines several macros for logging purposes, which are used to standardize the logging of information, warnings, errors, and debug messages. It also includes template functions for JSON value extraction with type safety and default value handling. The file provides a comprehensive set of utilities for tokenizing input data, including handling mixed types of input (strings and tokens) and ensuring UTF-8 validity. Additionally, it includes functions for encoding and decoding base64 data, generating random strings for identifiers, and formatting responses in a manner compatible with OpenAI's API specifications. The presence of structures and functions related to server-side grammar triggers and multi-tasking input processing suggests that the file is part of a larger system designed to handle complex interactions with language models, possibly in a chatbot or AI assistant context.
# Imports and Dependencies

---
- `common.h`
- `log.h`
- `llama.h`
- `arg.h`
- `base64.hpp`
- `mtmd.h`
- `mtmd-helper.h`
- `chat.h`
- `cpp-httplib/httplib.h`
- `nlohmann/json.hpp`
- `random`
- `sstream`
- `string`
- `vector`
- `memory`
- `cinttypes`


# Global Variables

---
### build\_info
- **Type**: `std::string`
- **Description**: The `build_info` variable is a constant static string that concatenates the build number and commit hash of the software. It is constructed using the `LLAMA_BUILD_NUMBER` and `LLAMA_COMMIT` macros, which are likely defined elsewhere in the codebase to represent the current build number and commit hash, respectively.
- **Use**: This variable is used to store and provide information about the specific build and commit of the software, which can be useful for debugging or version tracking.


---
### base64\_chars
- **Type**: `std::string`
- **Description**: The `base64_chars` variable is a constant string that contains the characters used in the Base64 encoding scheme. It includes uppercase and lowercase letters, digits, and the '+' and '/' symbols.
- **Use**: This variable is used to map indices to characters during Base64 encoding and decoding processes.


# Data Structures

---
### server\_grammar\_trigger<!-- {{#data_structure:server_grammar_trigger}} -->
- **Type**: `struct`
- **Members**:
    - `value`: Holds an instance of the `common_grammar_trigger` type.
- **Description**: The `server_grammar_trigger` struct is a thin wrapper around the `common_grammar_trigger` type, providing additional functionality for JSON (de)serialization. It includes a default constructor, a copy constructor that initializes the `value` member with a `common_grammar_trigger` instance, and a constructor that initializes the `value` member from a JSON object. The struct also provides a `to_json` method to serialize the `value` member back into a JSON object, handling specific cases based on the type of the grammar trigger.
- **Member Functions**:
    - [`server_grammar_trigger::server_grammar_trigger`](#server_grammar_triggerserver_grammar_trigger)
    - [`server_grammar_trigger::server_grammar_trigger`](#server_grammar_triggerserver_grammar_trigger)
    - [`server_grammar_trigger::server_grammar_trigger`](#server_grammar_triggerserver_grammar_trigger)
    - [`server_grammar_trigger::to_json`](#server_grammar_triggerto_json)

**Methods**

---
#### server\_grammar\_trigger::server\_grammar\_trigger<!-- {{#callable:server_grammar_trigger::server_grammar_trigger}} -->
The `server_grammar_trigger` constructor initializes a `server_grammar_trigger` object using a `common_grammar_trigger` object.
- **Inputs**:
    - `value`: A reference to a `common_grammar_trigger` object used to initialize the `server_grammar_trigger` object.
- **Control Flow**:
    - The constructor takes a `common_grammar_trigger` object as an argument.
    - It initializes the `value` member of the `server_grammar_trigger` object with the provided `common_grammar_trigger` object.
- **Output**: A `server_grammar_trigger` object initialized with the provided `common_grammar_trigger` object.
- **See also**: [`server_grammar_trigger`](#server_grammar_trigger)  (Data Structure)


---
#### server\_grammar\_trigger::server\_grammar\_trigger<!-- {{#callable:server_grammar_trigger::server_grammar_trigger}} -->
The `server_grammar_trigger` constructor initializes a `server_grammar_trigger` object from a `common_grammar_trigger` or a JSON object.
- **Inputs**:
    - `value`: A `common_grammar_trigger` object used to initialize the `server_grammar_trigger`.
    - `in`: A JSON object containing the fields 'type', 'value', and optionally 'token' to initialize the `server_grammar_trigger`.
- **Control Flow**:
    - The constructor `server_grammar_trigger(const common_grammar_trigger & value)` directly assigns the input `value` to the member `value`.
    - The constructor `server_grammar_trigger(const json & in)` extracts the 'type' and 'value' from the JSON object and assigns them to the corresponding fields in the `value` member.
    - If the 'type' is `COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN`, it also extracts the 'token' from the JSON and assigns it to the `value.token`.
- **Output**: The constructors do not return a value; they initialize the `server_grammar_trigger` object.
- **See also**: [`server_grammar_trigger`](#server_grammar_trigger)  (Data Structure)


---
#### server\_grammar\_trigger::server\_grammar\_trigger<!-- {{#callable:server_grammar_trigger::server_grammar_trigger}} -->
The `server_grammar_trigger` constructor initializes a `server_grammar_trigger` object from a JSON input by setting its type and value, and optionally its token if the type is `COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN`.
- **Inputs**:
    - `in`: A JSON object containing the fields 'type', 'value', and optionally 'token'.
- **Control Flow**:
    - Extracts the 'type' field from the JSON input and casts it to `common_grammar_trigger_type`, assigning it to `value.type`.
    - Extracts the 'value' field from the JSON input as a string and assigns it to `value.value`.
    - Checks if `value.type` is `COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN`; if true, extracts the 'token' field from the JSON input, casts it to `llama_token`, and assigns it to `value.token`.
- **Output**: This constructor does not return a value; it initializes the `server_grammar_trigger` object.
- **See also**: [`server_grammar_trigger`](#server_grammar_trigger)  (Data Structure)


---
#### server\_grammar\_trigger::to\_json<!-- {{#callable:server_grammar_trigger::to_json}} -->
The `to_json` function serializes a `server_grammar_trigger` object into a JSON object.
- **Inputs**: None
- **Control Flow**:
    - Initialize a JSON object `out` with the `type` and `value` fields from the `value` member of the `server_grammar_trigger` object.
    - Check if the `type` of `value` is `COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN`.
    - If true, add the `token` field to the JSON object `out`.
    - Return the JSON object `out`.
- **Output**: A JSON object representing the `server_grammar_trigger` with fields `type`, `value`, and optionally `token`.
- **See also**: [`server_grammar_trigger`](#server_grammar_trigger)  (Data Structure)



---
### oaicompat\_parser\_options<!-- {{#data_structure:oaicompat_parser_options}} -->
- **Type**: `struct`
- **Members**:
    - `use_jinja`: A boolean flag indicating whether to use Jinja templates.
    - `prefill_assistant`: A boolean flag indicating whether to prefill the assistant.
    - `reasoning_format`: An instance of `common_reasoning_format` specifying the reasoning format.
    - `tmpls`: A pointer to `common_chat_templates` for chat templates.
    - `allow_image`: A boolean flag indicating whether image input is allowed.
    - `allow_audio`: A boolean flag indicating whether audio input is allowed.
    - `enable_thinking`: A boolean flag indicating whether thinking is enabled, defaulting to true.
- **Description**: The `oaicompat_parser_options` struct is a configuration data structure used to specify various options for parsing OpenAI-compatible requests. It includes flags for enabling Jinja templates, pre-filling the assistant, and allowing image and audio inputs. Additionally, it holds a reasoning format and a pointer to chat templates, with an option to enable or disable thinking, which defaults to true.


---
### server\_tokens<!-- {{#data_structure:server_tokens}} -->
- **Type**: `struct`
- **Members**:
    - `has_mtmd`: Indicates whether the server_tokens instance has multi-token media data (mtmd).
    - `map_pos_to_media`: Maps a start position in tokens to an image chunk, used to associate media with token positions.
    - `tokens`: Stores a list of tokens, which can include LLAMA_TOKEN_NULL to indicate non-text tokens.
- **Description**: The `server_tokens` struct is designed to manage input tokens and associated media chunks for a server, particularly in contexts where multi-token media data (mtmd) is involved. It maintains a list of tokens, which can include special null tokens to represent non-text data, and a mapping from token positions to media chunks, allowing for efficient handling of mixed text and media inputs. The struct provides functionality to add, access, and manipulate tokens and media chunks, while ensuring synchronization and preventing invalid operations, such as removing tokens in the middle of a media chunk. It supports operations like detokenization, validation, and processing of media chunks, making it a versatile tool for managing complex input data in server applications.
- **Member Functions**:
    - [`server_tokens::server_tokens`](#server_tokensserver_tokens)
    - [`server_tokens::~server_tokens`](#server_tokensserver_tokens)
    - [`server_tokens::server_tokens`](#server_tokensserver_tokens)
    - [`server_tokens::operator=`](#server_tokensoperator=)
    - [`server_tokens::server_tokens`](#server_tokensserver_tokens)
    - [`server_tokens::operator=`](#server_tokensoperator=)
    - [`server_tokens::operator[]`](llama.cpp/tools/server/utils.hpp#callable:server_tokens::operator[])
    - [`server_tokens::operator[]`](llama.cpp/tools/server/utils.hpp#callable:server_tokens::operator[])
    - [`server_tokens::server_tokens`](#server_tokensserver_tokens)
    - [`server_tokens::server_tokens`](#server_tokensserver_tokens)
    - [`server_tokens::str`](#server_tokensstr)
    - [`server_tokens::find_chunk`](#server_tokensfind_chunk)
    - [`server_tokens::push_back`](#server_tokenspush_back)
    - [`server_tokens::push_back`](#server_tokenspush_back)
    - [`server_tokens::insert`](#server_tokensinsert)
    - [`server_tokens::get_text_tokens`](#server_tokensget_text_tokens)
    - [`server_tokens::set_token`](#server_tokensset_token)
    - [`server_tokens::size`](#server_tokenssize)
    - [`server_tokens::empty`](#server_tokensempty)
    - [`server_tokens::clear`](#server_tokensclear)
    - [`server_tokens::keep_first`](#server_tokenskeep_first)
    - [`server_tokens::detokenize`](#server_tokensdetokenize)
    - [`server_tokens::get_common_prefix`](#server_tokensget_common_prefix)
    - [`server_tokens::validate`](#server_tokensvalidate)
    - [`server_tokens::process_chunk`](#server_tokensprocess_chunk)

**Methods**

---
#### server\_tokens::server\_tokens<!-- {{#callable:server_tokens::server_tokens}} -->
The `server_tokens` constructor initializes a `server_tokens` object with either a list of `mtmd` input chunks or a list of `llama` tokens, and sets the `has_mtmd` flag accordingly.
- **Inputs**:
    - `mtmd_chunks`: A reference to a list of `mtmd::input_chunks` used to initialize the `server_tokens` object.
    - `has_mtmd`: A boolean flag indicating whether the `server_tokens` object should handle `mtmd` input chunks.
    - `tokens`: A reference to a list of `llama_tokens` used to initialize the `server_tokens` object.
- **Control Flow**:
    - The constructor checks if it is being initialized with `mtmd_chunks` or `tokens`.
    - If initialized with `mtmd_chunks`, it iterates over each chunk and calls `push_back` to add it to the `server_tokens` object.
    - If initialized with `tokens`, it directly assigns the tokens to the `tokens` member of the `server_tokens` object.
    - The `has_mtmd` flag is set based on the input parameter.
- **Output**: A `server_tokens` object initialized with the provided input chunks or tokens.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::\~server\_tokens<!-- {{#callable:server_tokens::~server_tokens}} -->
The destructor `~server_tokens()` is a default destructor for the `server_tokens` struct, which performs no specific actions upon object destruction.
- **Inputs**: None
- **Control Flow**:
    - The destructor `~server_tokens()` is defined as `default`, meaning it relies on the compiler-generated default behavior for destructors.
    - No custom cleanup or resource deallocation is performed in this destructor.
- **Output**: There is no output from this destructor as it performs no operations.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::server\_tokens<!-- {{#callable:server_tokens::server_tokens}} -->
The `server_tokens` constructor is deleted to prevent copying of `server_tokens` objects.
- **Inputs**: None
- **Control Flow**:
    - The function is a deleted copy constructor, which means it is not implemented and cannot be used to copy `server_tokens` objects.
    - This prevents the copying of `server_tokens` instances, ensuring that each instance is unique and not accidentally duplicated.
- **Output**: There is no output as this is a deleted function, meaning it cannot be called or used.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::operator=<!-- {{#callable:server_tokens::operator=}} -->
The `operator=` function for the `server_tokens` struct is deleted to prevent assignment of `server_tokens` objects.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as deleted, which means it cannot be used to assign one `server_tokens` object to another.
- **Output**: There is no output as the function is deleted and cannot be invoked.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::server\_tokens<!-- {{#callable:server_tokens::server_tokens}} -->
The `server_tokens` move constructor and move assignment operator allow for efficient transfer of resources from one `server_tokens` instance to another.
- **Inputs**:
    - `server_tokens&&`: A rvalue reference to a `server_tokens` object, representing the source from which resources will be moved.
- **Control Flow**:
    - The move constructor and move assignment operator are both set to `default`, indicating that the compiler will automatically generate them.
    - These operations transfer ownership of resources from the source object to the target object, leaving the source in a valid but unspecified state.
- **Output**: The output is a new `server_tokens` object with resources moved from the source object, or an existing `server_tokens` object with its resources replaced by those from the source object.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::operator=<!-- {{#callable:server_tokens::operator=}} -->
The `operator=` function is a default move assignment operator for the `server_tokens` struct.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as a default move assignment operator, meaning it will automatically handle the move assignment of `server_tokens` objects by transferring ownership of resources from the source object to the target object.
    - No custom logic is implemented in this operator, so it relies on the compiler-generated behavior for moving the struct's members.
- **Output**: The function returns a reference to the `server_tokens` object that has been assigned the values from the moved object.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::operator\[\]<!-- {{#callable:server_tokens::operator[]}} -->
The `operator[]` function provides access to elements in the `tokens` list within the `server_tokens` structure, allowing both mutable and immutable access.
- **Inputs**:
    - `index`: A `size_t` index specifying the position of the token to access in the `tokens` list.
- **Control Flow**:
    - The function retrieves the token at the specified `index` from the `tokens` list.
    - There are two overloads: one for mutable access and one for immutable access.
- **Output**: The function returns a `llama_token` at the specified index for mutable access, and a `const llama_token&` for immutable access.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::operator\[\]<!-- {{#callable:server_tokens::operator[]}} -->
The `operator[]` function provides read-only access to a `llama_token` at a specified index within the `tokens` list of a `server_tokens` object.
- **Inputs**:
    - `index`: A `size_t` representing the position in the `tokens` list from which to retrieve the `llama_token`.
- **Control Flow**:
    - The function takes an index as input.
    - It accesses the `tokens` list at the specified index.
    - It returns the `llama_token` located at that index.
- **Output**: A constant reference to the `llama_token` at the specified index in the `tokens` list.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::server\_tokens<!-- {{#callable:server_tokens::server_tokens}} -->
The `server_tokens` constructor initializes a `server_tokens` object by populating its token list with tokens from the provided `mtmd_chunks` and sets the `has_mtmd` flag.
- **Inputs**:
    - `mtmd_chunks`: A reference to an `mtmd::input_chunks` object, which is a collection of input chunks to be processed and added to the token list.
    - `has_mtmd`: A boolean flag indicating whether the `mtmd` (multi-token media data) feature is enabled.
- **Control Flow**:
    - The constructor initializes the `has_mtmd` member variable with the provided `has_mtmd` argument.
    - It iterates over each chunk in the `mtmd_chunks` collection.
    - For each chunk, it calls the [`push_back`](#server_tokenspush_back) method to add the chunk to the token list.
- **Output**: The constructor does not return a value; it initializes the `server_tokens` object.
- **Functions called**:
    - [`server_tokens::push_back`](#server_tokenspush_back)
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::server\_tokens<!-- {{#callable:server_tokens::server_tokens}} -->
The `server_tokens` constructor initializes a `server_tokens` object with a given set of tokens and a flag indicating the presence of multimedia data (mtmd).
- **Inputs**:
    - `tokens`: A reference to a `llama_tokens` object, which is a list of tokens that may include LLAMA_TOKEN_NULL to indicate non-text tokens.
    - `has_mtmd`: A boolean flag indicating whether multimedia data (mtmd) is present.
- **Control Flow**:
    - The constructor initializes the `has_mtmd` member with the provided `has_mtmd` argument.
    - The constructor initializes the `tokens` member with the provided `tokens` argument.
- **Output**: This constructor does not return any value as it is a constructor for initializing an object of the `server_tokens` class.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::str<!-- {{#callable:server_tokens::str}} -->
The `str` method generates a string representation of the `server_tokens` object, detailing the tokens and their associated image positions.
- **Inputs**: None
- **Control Flow**:
    - Initialize an output string stream `oss`.
    - Append the string 'tokens: ' to `oss`.
    - Iterate over each token `t` in the `tokens` list.
    - For each token, check if it is `LLAMA_TOKEN_NULL`; if so, append '<embd> ' to `oss`, otherwise append the token value followed by a space.
    - Append a newline character to `oss`.
    - Append the string 'image pos: ' to `oss`.
    - Iterate over each entry `it` in the `map_pos_to_media` map.
    - For each entry, append the key (position) followed by a comma and space to `oss`.
    - Return the accumulated string from `oss`.
- **Output**: A string that represents the tokens and their image positions in the `server_tokens` object.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::find\_chunk<!-- {{#callable:server_tokens::find_chunk}} -->
The `find_chunk` function retrieves a media chunk associated with a given position from a map, throwing an error if the position is not found.
- **Inputs**:
    - `pos`: A `llama_pos` type representing the position in the token sequence for which the associated media chunk is to be retrieved.
- **Control Flow**:
    - Searches for the given position `pos` in the `map_pos_to_media` map.
    - If the position is found, returns the associated media chunk pointer.
    - If the position is not found, throws a `std::runtime_error` with the message "Chunk not found".
- **Output**: Returns a reference to the `mtmd::input_chunk_ptr` associated with the given position if found; otherwise, throws an exception.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::push\_back<!-- {{#callable:server_tokens::push_back}} -->
The `push_back` function adds a non-null `llama_token` to the `tokens` list of the `server_tokens` structure.
- **Inputs**:
    - `tok`: A `llama_token` to be added to the `tokens` list; it must not be `LLAMA_TOKEN_NULL`.
- **Control Flow**:
    - Check if the input token `tok` is `LLAMA_TOKEN_NULL`.
    - If `tok` is `LLAMA_TOKEN_NULL`, throw a `std::runtime_error` with the message "Invalid token".
    - If `tok` is not `LLAMA_TOKEN_NULL`, add it to the `tokens` list using `emplace_back`.
- **Output**: The function does not return a value; it modifies the `tokens` list by adding the provided token.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::push\_back<!-- {{#callable:server_tokens::push_back}} -->
The [`push_back`](#server_tokenspush_back) function adds a new `mtmd_input_chunk` to the `server_tokens` object, handling different types of chunks (image, audio, or text) appropriately.
- **Inputs**:
    - `chunk`: A pointer to an `mtmd_input_chunk` object, which represents a chunk of input data that can be of type image, audio, or text.
- **Control Flow**:
    - Retrieve the type of the input chunk using `mtmd_input_chunk_get_type(chunk)`.
    - If the chunk type is image or audio, assert that `has_mtmd` is true, then determine the number of positions (`n_pos`) the chunk occupies.
    - Calculate the starting position (`start_pos`) in the `tokens` list, and insert `LLAMA_TOKEN_NULL` for each position the chunk occupies.
    - Create a copy of the chunk and store it in `map_pos_to_media` with `start_pos` as the key.
    - If the chunk type is text, retrieve the text tokens and their count, then recursively call [`push_back`](#server_tokenspush_back) for each token.
    - If the chunk type is invalid, abort the operation with an error message.
- **Output**: The function does not return a value; it modifies the `server_tokens` object by adding the chunk's data to the `tokens` list and updating the `map_pos_to_media` if necessary.
- **Functions called**:
    - [`mtmd_input_chunk_get_type`](../mtmd/mtmd.cpp.driver.md#mtmd_input_chunk_get_type)
    - [`mtmd_input_chunk_get_n_pos`](../mtmd/mtmd.cpp.driver.md#mtmd_input_chunk_get_n_pos)
    - [`server_tokens::push_back`](#server_tokenspush_back)
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::insert<!-- {{#callable:server_tokens::insert}} -->
The `insert` function appends a sequence of tokens to the existing token list in the `server_tokens` structure, provided that the `has_mtmd` flag is false.
- **Inputs**:
    - `inp_tokens`: A constant reference to a `llama_tokens` object, representing the sequence of tokens to be inserted.
- **Control Flow**:
    - The function begins by asserting that the `has_mtmd` flag is false, using `GGML_ASSERT(!has_mtmd)`, to ensure that the operation is only allowed when mtmd is disabled.
    - If the assertion passes, the function proceeds to insert the tokens from `inp_tokens` at the end of the `tokens` list using the `insert` method of the `tokens` vector.
- **Output**: The function does not return any value; it modifies the `tokens` list in place.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::get\_text\_tokens<!-- {{#callable:server_tokens::get_text_tokens}} -->
The `get_text_tokens` function returns the list of tokens stored in the `server_tokens` object, ensuring that the `has_mtmd` flag is not set.
- **Inputs**: None
- **Control Flow**:
    - The function asserts that the `has_mtmd` flag is false using `GGML_ASSERT`.
    - It then returns the `tokens` member of the `server_tokens` object.
- **Output**: A constant reference to the `llama_tokens` object, which is the list of tokens stored in the `server_tokens` object.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::set\_token<!-- {{#callable:server_tokens::set_token}} -->
The `set_token` function assigns a given token ID to a specified position in the `tokens` list, ensuring that the `has_mtmd` flag is not set.
- **Inputs**:
    - `pos`: The position in the `tokens` list where the token ID should be set.
    - `id`: The token ID to be assigned to the specified position.
- **Control Flow**:
    - The function begins by asserting that the `has_mtmd` flag is not set, using `GGML_ASSERT(!has_mtmd);` to ensure this condition.
    - If the assertion passes, the function assigns the token ID `id` to the position `pos` in the `tokens` list.
- **Output**: The function does not return any value; it modifies the `tokens` list in place.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::size<!-- {{#callable:server_tokens::size}} -->
The `size` function returns the number of tokens in the `tokens` list of the `server_tokens` structure.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `tokens` member of the `server_tokens` structure.
    - It calls the `size()` method on the `tokens` list to get the number of elements.
    - The function returns the result of the `tokens.size()` call.
- **Output**: The function returns a `size_t` value representing the number of tokens in the `tokens` list.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::empty<!-- {{#callable:server_tokens::empty}} -->
The `empty` function checks if the `tokens` container in the `server_tokens` structure is empty.
- **Inputs**: None
- **Control Flow**:
    - The function calls the `empty` method on the `tokens` container.
    - It returns the result of the `tokens.empty()` call.
- **Output**: A boolean value indicating whether the `tokens` container is empty (true if empty, false otherwise).
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::clear<!-- {{#callable:server_tokens::clear}} -->
The `clear` function empties the `tokens` list within the `server_tokens` structure.
- **Inputs**: None
- **Control Flow**:
    - The function calls the `clear` method on the `tokens` member, which is a `llama_tokens` list, to remove all elements from it.
- **Output**: The function does not return any value.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::keep\_first<!-- {{#callable:server_tokens::keep_first}} -->
The `keep_first` function resizes the `tokens` list to retain only the first `n` tokens, ensuring that no tokens are removed from the middle of an image chunk if `has_mtmd` is true.
- **Inputs**:
    - `n`: The number of tokens to retain from the beginning of the `tokens` list.
- **Control Flow**:
    - Assert that `n` is less than or equal to the size of `tokens`.
    - If `has_mtmd` is true and `n` equals the size of `tokens`, return immediately as no resizing is needed.
    - If `has_mtmd` is true and `n` is greater than 0, check if the last token to be retained is `LLAMA_TOKEN_NULL`; if so, ensure it is the start of a chunk by calling [`find_chunk`](#server_tokensfind_chunk).
    - Iterate over `map_pos_to_media` and remove any entries where the position is greater than or equal to `n`.
    - Resize the `tokens` list to `n`.
- **Output**: The function does not return a value; it modifies the `tokens` list in place.
- **Functions called**:
    - [`server_tokens::find_chunk`](#server_tokensfind_chunk)
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::detokenize<!-- {{#callable:server_tokens::detokenize}} -->
The `detokenize` function converts a list of tokens into a string representation, excluding any null tokens, using a specified context and special handling flag.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which provides the context for detokenization.
    - `special`: A boolean flag indicating whether special handling should be applied during detokenization.
- **Control Flow**:
    - Initialize an empty `llama_tokens` vector named `text_tokens` and reserve space equal to the size of `tokens`.
    - Iterate over each token `t` in the `tokens` vector.
    - If `t` is not equal to `LLAMA_TOKEN_NULL`, append `t` to `text_tokens`.
    - Call `common_detokenize` with `ctx`, `text_tokens`, and `special` as arguments, and return its result.
- **Output**: A `std::string` that represents the detokenized text from the input tokens, excluding null tokens.
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::get\_common\_prefix<!-- {{#callable:server_tokens::get_common_prefix}} -->
The `get_common_prefix` function calculates the length of the longest common prefix between two `server_tokens` objects' token sequences.
- **Inputs**:
    - `b`: A reference to another `server_tokens` object whose tokens are compared with the current object's tokens.
- **Control Flow**:
    - Determine the maximum index to compare by taking the minimum size of the two token sequences.
    - Iterate over the tokens up to the maximum index.
    - For each token pair, check if both are `LLAMA_TOKEN_NULL`, indicating a media chunk, and if so, verify the chunk IDs and positions match; if they do, skip ahead by the chunk's position count minus one.
    - If the tokens are equal, continue to the next pair.
    - If any token pair is not equal, return the current index as the length of the common prefix.
    - If all tokens are equal up to the maximum index, return the maximum index as the length of the common prefix.
- **Output**: The function returns a `size_t` representing the length of the common prefix between the two token sequences.
- **Functions called**:
    - [`server_tokens::find_chunk`](#server_tokensfind_chunk)
    - [`mtmd_input_chunk_get_id`](../mtmd/mtmd.cpp.driver.md#mtmd_input_chunk_get_id)
    - [`mtmd_input_chunk_get_n_pos`](../mtmd/mtmd.cpp.driver.md#mtmd_input_chunk_get_n_pos)
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::validate<!-- {{#callable:server_tokens::validate}} -->
The `validate` function checks if all tokens in the `server_tokens` object are valid within the given vocabulary context.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, which provides the context for accessing the model and vocabulary.
- **Control Flow**:
    - Retrieve the model from the context using `llama_get_model` and then get the vocabulary from the model using `llama_model_get_vocab`.
    - Determine the number of tokens in the vocabulary using `llama_vocab_n_tokens`.
    - Iterate over each token in the `tokens` list of the `server_tokens` object.
    - For each token, check if it is `LLAMA_TOKEN_NULL`. If so, attempt to find the corresponding chunk using [`find_chunk`](#server_tokensfind_chunk). If the chunk is found, adjust the loop index by the number of positions the chunk occupies minus one.
    - If the token is not `LLAMA_TOKEN_NULL`, check if it is within the valid range of the vocabulary (i.e., greater than or equal to 0 and less than `n_vocab`).
    - If any token is invalid or an exception is caught during chunk retrieval, return `false`.
    - If all tokens are valid, return `true`.
- **Output**: A boolean value indicating whether all tokens are valid within the given vocabulary context.
- **Functions called**:
    - [`server_tokens::find_chunk`](#server_tokensfind_chunk)
    - [`mtmd_input_chunk_get_n_pos`](../mtmd/mtmd.cpp.driver.md#mtmd_input_chunk_get_n_pos)
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)


---
#### server\_tokens::process\_chunk<!-- {{#callable:server_tokens::process_chunk}} -->
The `process_chunk` function processes a media chunk (image or audio) from a given position in a sequence, updating the position and returning a status code.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which provides context for processing the chunk.
    - `mctx`: A pointer to a `mtmd_context` object, which is used for media processing.
    - `n_past`: A `llama_pos` indicating the starting position in the sequence from which the chunk is to be processed.
    - `seq_id`: An integer representing the sequence identifier for the processing task.
    - `n_pos_out`: A reference to a `llama_pos` that will be updated with the new position after processing the chunk.
- **Control Flow**:
    - Retrieve the media chunk from the position `n_past` using [`find_chunk`](#server_tokensfind_chunk).
    - Determine the type of the chunk (image or audio) and log the processing start message.
    - Get the batch size using `llama_n_batch` from the context `ctx`.
    - Record the current time in milliseconds for performance logging.
    - Call [`mtmd_helper_eval_chunk_single`](../mtmd/mtmd-helper.cpp.driver.md#mtmd_helper_eval_chunk_single) to process the chunk, passing necessary parameters including `mctx`, `ctx`, chunk data, `n_past`, `seq_id`, batch size, and a flag for logits.
    - Log the processing time once the evaluation is complete.
    - Check the result of the evaluation; if non-zero, log an error, set `n_pos_out` to `n_past`, and return the result code.
    - If successful, update `n_pos_out` with the new position `new_n_past` and return 0.
- **Output**: Returns an integer status code, where 0 indicates success and any non-zero value indicates an error during processing.
- **Functions called**:
    - [`server_tokens::find_chunk`](#server_tokensfind_chunk)
    - [`mtmd_input_chunk_get_type`](../mtmd/mtmd.cpp.driver.md#mtmd_input_chunk_get_type)
    - [`mtmd_helper_eval_chunk_single`](../mtmd/mtmd-helper.cpp.driver.md#mtmd_helper_eval_chunk_single)
- **See also**: [`server_tokens`](#server_tokens)  (Data Structure)



# Functions

---
### json\_value<!-- {{#callable:json_value}} -->
The `json_value` function retrieves a value from a JSON object by key, returning a default value if the key is not present or the value is null or of the wrong type.
- **Inputs**:
    - `body`: A JSON object from which the value is to be retrieved.
    - `key`: A string representing the key whose value is to be retrieved from the JSON object.
    - `default_value`: A default value to return if the key is not present, the value is null, or the value is of the wrong type.
- **Control Flow**:
    - Check if the JSON object `body` contains the specified `key` and the value is not null.
    - If the key exists and the value is not null, attempt to return the value associated with the key.
    - If a type error occurs during retrieval, log a warning and return the `default_value`.
    - If the key does not exist or the value is null, return the `default_value`.
- **Output**: Returns the value associated with the specified key in the JSON object, or the default value if the key is not present, the value is null, or the value is of the wrong type.


---
### json\_is\_array\_of\_numbers<!-- {{#callable:json_is_array_of_numbers}} -->
The function `json_is_array_of_numbers` checks if a given JSON object is an array consisting solely of integer numbers.
- **Inputs**:
    - `data`: A JSON object to be checked if it is an array of integer numbers.
- **Control Flow**:
    - Check if the input JSON object `data` is an array using `data.is_array()`.
    - If `data` is an array, iterate over each element `e` in the array.
    - For each element `e`, check if it is an integer number using `e.is_number_integer()`.
    - If any element is not an integer, return `false`.
    - If all elements are integers, return `true`.
    - If `data` is not an array, return `false`.
- **Output**: A boolean value indicating whether the JSON object is an array of integer numbers (true) or not (false).


---
### json\_is\_array\_of\_mixed\_numbers\_strings<!-- {{#callable:json_is_array_of_mixed_numbers_strings}} -->
The function `json_is_array_of_mixed_numbers_strings` checks if a JSON array contains both integer numbers and strings.
- **Inputs**:
    - `data`: A JSON object that is expected to be an array, which will be checked for containing both numbers and strings.
- **Control Flow**:
    - Initialize two boolean flags `seen_string` and `seen_number` to false.
    - Check if the input `data` is an array.
    - Iterate over each element `e` in the array.
    - Update `seen_string` if the element is a string and `seen_number` if the element is an integer number.
    - If both `seen_string` and `seen_number` are true, return true immediately.
    - If the loop completes without finding both types, return false.
- **Output**: A boolean value indicating whether the array contains both integer numbers and strings.


---
### json\_get\_nested\_values<!-- {{#callable:json_get_nested_values}} -->
The `json_get_nested_values` function retrieves values from a JSON object based on a list of path strings and returns a JSON object containing these values.
- **Inputs**:
    - `paths`: A vector of strings, where each string represents a path to a value in the JSON object, with keys separated by '/'.
    - `js`: A JSON object from which values are to be retrieved based on the provided paths.
- **Control Flow**:
    - Initialize an empty JSON object `result` to store the retrieved values.
    - Iterate over each path in the `paths` vector.
    - For each path, split the path string into keys using '/' as the separator.
    - Initialize a boolean `valid_path` to true and set `current` to the input JSON `js`.
    - Iterate over each key in the split path keys.
    - Check if `current` is an object and contains the key; if so, update `current` to the value associated with the key, otherwise set `valid_path` to false.
    - If `valid_path` remains true after processing all keys, add the final `current` value to the `result` JSON object with the path as the key.
    - Return the `result` JSON object containing all successfully retrieved values.
- **Output**: A JSON object containing key-value pairs where each key is a path from the input `paths` and the value is the corresponding value from the input JSON `js` if the path is valid.


---
### tokenize\_mixed<!-- {{#callable:tokenize_mixed}} -->
The `tokenize_mixed` function tokenizes a JSON prompt that can be a string or an array of mixed strings and tokens, using a specified vocabulary and options for adding and parsing special tokens.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure, which contains the vocabulary used for tokenization.
    - `json_prompt`: A JSON object that represents the prompt to be tokenized; it can be a string or an array containing strings and/or tokens.
    - `add_special`: A boolean flag indicating whether special tokens should be added during tokenization.
    - `parse_special`: A boolean flag indicating whether special tokens should be parsed during tokenization.
- **Control Flow**:
    - Initialize an empty `llama_tokens` vector named `prompt_tokens` to store the resulting tokens.
    - Check if `json_prompt` is an array; if so, iterate over each element in the array.
    - For each element in the array, check if it is a string.
    - If the element is a string, tokenize it using `common_tokenize` with the `add_special` flag set to true for the first string and false for subsequent strings, then append the tokens to `prompt_tokens`.
    - If the element is not a string, convert it to a `llama_token` and append it to `prompt_tokens`.
    - If `json_prompt` is not an array, treat it as a string, tokenize it using `common_tokenize` with the `add_special` flag, and store the result in `prompt_tokens`.
    - Return the `prompt_tokens` vector containing the tokenized prompt.
- **Output**: A `llama_tokens` vector containing the tokenized representation of the input JSON prompt.


---
### tokenize\_input\_prompts<!-- {{#callable:tokenize_input_prompts}} -->
The `tokenize_input_prompts` function tokenizes a JSON input prompt into a vector of llama_tokens based on its type and structure.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure used for tokenization.
    - `json_prompt`: A JSON object representing the input prompt, which can be a string, an array of numbers, a mixed array of strings and numbers, or an array of such prompts.
    - `add_special`: A boolean flag indicating whether special tokens should be added during tokenization.
    - `parse_special`: A boolean flag indicating whether special tokens should be parsed during tokenization.
- **Control Flow**:
    - Initialize an empty vector `result` to store the tokenized prompts.
    - Check if `json_prompt` is a string or a mixed array of numbers and strings; if so, tokenize it using [`tokenize_mixed`](#tokenize_mixed) and add to `result`.
    - Check if `json_prompt` is an array of numbers; if so, convert it directly to `llama_tokens` and add to `result`.
    - If `json_prompt` is an array, iterate over each element and tokenize based on its type (string, mixed, or numbers) and add to `result`.
    - Throw a runtime error if `json_prompt` is not a valid type or if `result` is empty.
    - Return the `result` vector containing the tokenized prompts.
- **Output**: A vector of `llama_tokens` representing the tokenized input prompts.
- **Functions called**:
    - [`json_is_array_of_mixed_numbers_strings`](#json_is_array_of_mixed_numbers_strings)
    - [`tokenize_mixed`](#tokenize_mixed)
    - [`json_is_array_of_numbers`](#json_is_array_of_numbers)


---
### validate\_utf8<!-- {{#callable:validate_utf8}} -->
The `validate_utf8` function checks if the last few bytes of a given string form a valid UTF-8 sequence and returns the index of the last valid character.
- **Inputs**:
    - `text`: A constant reference to a `std::string` that represents the text to be validated for UTF-8 encoding.
- **Control Flow**:
    - Initialize `len` with the size of the input string `text`.
    - If `len` is 0, return 0 immediately as there is nothing to validate.
    - Iterate over the last 1 to 4 bytes of the string, checking if they form the start of a multi-byte UTF-8 sequence.
    - For each byte, determine if it is the start of a 2-byte, 3-byte, or 4-byte UTF-8 sequence using bitwise operations.
    - If a valid start of a multi-byte sequence is found but the sequence is incomplete (i.e., not enough bytes are present), return the index before the start of this sequence.
    - If no incomplete multi-byte sequence is found, return the full length of the string.
- **Output**: The function returns a `size_t` value representing the index of the last valid character in the string, or the full length if the string is valid UTF-8.


---
### format\_rerank<!-- {{#callable:format_rerank}} -->
The `format_rerank` function formats a rerank task by combining query and document tokens with special tokens for beginning, end, and separation.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure, which provides access to special tokens like BOS, EOS, and SEP.
    - `query`: A `llama_tokens` object representing the query tokens to be included in the formatted rerank task.
    - `doc`: A `llama_tokens` object representing the document tokens to be included in the formatted rerank task.
- **Control Flow**:
    - Initialize an empty `llama_tokens` object named `result`.
    - Retrieve the EOS token from the vocabulary; if not available, use the SEP token as a fallback.
    - Reserve space in `result` for the combined size of `doc`, `query`, and four additional tokens.
    - Insert the BOS token at the beginning of `result`.
    - Append the `query` tokens to `result`.
    - Insert the EOS token after the `query` tokens.
    - Insert the SEP token after the EOS token.
    - Append the `doc` tokens to `result`.
    - Insert the EOS token after the `doc` tokens.
    - Return the `result` containing the formatted rerank task.
- **Output**: A `llama_tokens` object containing the formatted rerank task with special tokens and the input query and document tokens.


---
### format\_infill<!-- {{#callable:format_infill}} -->
The `format_infill` function formats and tokenizes input data for an infill task using a specified vocabulary and context parameters.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure used for tokenization.
    - `input_prefix`: A JSON object representing the prefix input data to be tokenized.
    - `input_suffix`: A JSON object representing the suffix input data to be tokenized.
    - `input_extra`: A JSON array containing additional input data chunks, each with 'text' and 'filename' fields.
    - `n_batch`: An integer specifying the batch size for processing.
    - `n_predict`: An integer specifying the number of predictions to be made.
    - `n_ctx`: An integer specifying the context size for tokenization.
    - `spm_infill`: A boolean indicating whether to use SPM infill mode.
    - `tokens_prompt`: A `llama_tokens` object containing the prompt tokens to be included in the infill task.
- **Control Flow**:
    - Initialize an empty `llama_tokens` object `extra_tokens` and reserve space for `n_ctx` tokens.
    - Tokenize the `input_prefix` and `input_suffix` using [`tokenize_mixed`](#tokenize_mixed).
    - If a FIM repository token is available, add it and a static project name to `extra_tokens`.
    - Iterate over `input_extra`, extracting 'text' and 'filename', and tokenize them, adding to `extra_tokens` with appropriate separators.
    - Determine the number of prefix and suffix tokens to take based on `n_batch` and `tokens_prompt` size.
    - Adjust `tokens_prefix` and `tokens_suffix` to fit the determined sizes, adding special tokens for FIM prefix and suffix.
    - Select the input and end tokens based on `spm_infill` mode, and optionally add a beginning-of-sequence token.
    - Insert `extra_tokens` into the beginning of the selected input tokens and append the end tokens.
    - Add a FIM middle token to the end of the input tokens.
    - Return the formatted and tokenized input as `embd_inp`.
- **Output**: A `llama_tokens` object containing the formatted and tokenized input data for the infill task.
- **Functions called**:
    - [`tokenize_mixed`](#tokenize_mixed)
    - [`json_value`](#json_value)


---
### is\_base64<!-- {{#callable:is_base64}} -->
The `is_base64` function checks if a given character is a valid base64 character.
- **Inputs**:
    - `c`: A single character of type `uint8_t` to be checked if it is a base64 character.
- **Control Flow**:
    - The function checks if the character `c` is alphanumeric using `isalnum(c)`.
    - If `c` is not alphanumeric, it checks if `c` is either '+' or '/'.
    - The function returns `true` if any of the above conditions are met, indicating that `c` is a valid base64 character.
- **Output**: A boolean value indicating whether the character is a valid base64 character (`true`) or not (`false`).


---
### base64\_decode<!-- {{#callable:base64_decode}} -->
The `base64_decode` function decodes a Base64-encoded string into a raw binary buffer.
- **Inputs**:
    - `encoded_string`: A constant reference to a `std::string` that contains the Base64-encoded data to be decoded.
- **Control Flow**:
    - Initialize variables for indexing and length tracking.
    - Create arrays to hold 4 Base64 characters and 3 decoded bytes.
    - Iterate over the input string, checking for valid Base64 characters and '=' padding character.
    - For every 4 valid Base64 characters, convert them to 3 bytes and append to the result buffer.
    - If there are remaining characters after the loop, process them to extract the remaining bytes.
    - Return the decoded binary data as a `raw_buffer`.
- **Output**: A `raw_buffer` (which is a `std::vector<uint8_t>`) containing the decoded binary data.
- **Functions called**:
    - [`is_base64`](#is_base64)


---
### random\_string<!-- {{#callable:random_string}} -->
The `random_string` function generates a random 32-character alphanumeric string.
- **Inputs**: None
- **Control Flow**:
    - A static string `str` containing alphanumeric characters is defined.
    - A random device `rd` and a Mersenne Twister generator `generator` are initialized.
    - A string `result` of length 32 is initialized with spaces.
    - A loop iterates 32 times, each time selecting a random character from `str` and assigning it to the corresponding position in `result`.
    - The `result` string is returned.
- **Output**: A random 32-character alphanumeric string.


---
### gen\_chatcmplid<!-- {{#callable:gen_chatcmplid}} -->
The `gen_chatcmplid` function generates a unique chat completion identifier by concatenating a fixed prefix with a random string.
- **Inputs**: None
- **Control Flow**:
    - The function calls `random_string()` to generate a random string.
    - It concatenates the prefix 'chatcmpl-' with the random string.
    - The concatenated result is returned as the output.
- **Output**: A string that represents a unique chat completion identifier, prefixed with 'chatcmpl-'.
- **Functions called**:
    - [`random_string`](#random_string)


---
### gen\_tool\_call\_id<!-- {{#callable:gen_tool_call_id}} -->
The `gen_tool_call_id` function generates a random string to be used as a tool call identifier.
- **Inputs**: None
- **Control Flow**:
    - The function calls the [`random_string`](#random_string) function.
    - The [`random_string`](#random_string) function generates a random 32-character string consisting of alphanumeric characters.
    - The generated string is returned as the tool call identifier.
- **Output**: A random 32-character alphanumeric string.
- **Functions called**:
    - [`random_string`](#random_string)


---
### tokens\_to\_str<!-- {{#callable:tokens_to_str}} -->
The `tokens_to_str` function converts a range of tokens into a string representation using a given context.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which provides the context needed for token conversion.
    - `begin`: An iterator pointing to the beginning of the token range to be converted.
    - `end`: An iterator pointing to the end of the token range to be converted.
- **Control Flow**:
    - Initialize an empty string `ret` to store the result.
    - Iterate over the range from `begin` to `end`.
    - For each token in the range, convert it to a string piece using `common_token_to_piece` and append it to `ret`.
    - Return the concatenated string `ret`.
- **Output**: A `std::string` that represents the concatenated string pieces of the tokens in the specified range.


---
### tokens\_to\_output\_formatted\_string<!-- {{#callable:tokens_to_output_formatted_string}} -->
The function `tokens_to_output_formatted_string` converts a given token into a formatted string representation, handling special cases for partial UTF-8 characters.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, which provides context for token processing.
    - `token`: A `llama_token` representing the token to be converted into a string.
- **Control Flow**:
    - Initialize the output string by checking if the token is `LLAMA_TOKEN_NULL`; if not, convert the token to a string using `common_token_to_piece`.
    - Check if the output string size is 1 and the first bit of the character is 1, indicating a partial UTF-8 character.
    - If the above condition is true, convert the character to a hexadecimal string and format it as a byte representation.
    - Return the formatted output string.
- **Output**: A `std::string` representing the formatted output of the token, with special handling for partial UTF-8 characters.


---
### server\_sent\_event<!-- {{#callable:server_sent_event}} -->
The `server_sent_event` function formats a server-sent event message and writes it to a data sink.
- **Inputs**:
    - `sink`: A reference to an `httplib::DataSink` object where the formatted event message will be written.
    - `event`: A C-style string representing the event type or name.
    - `data`: A JSON object containing the data to be included in the event message.
- **Control Flow**:
    - Concatenate the event name and the JSON data, formatted as a string, with a colon and two newline characters to comply with RFC 8895.
    - Log the formatted string for debugging purposes.
    - Write the formatted string to the provided data sink using the `write` method.
- **Output**: Returns a boolean indicating whether the write operation to the data sink was successful.


---
### oaicompat\_completion\_params\_parse<!-- {{#callable:oaicompat_completion_params_parse}} -->
The function `oaicompat_completion_params_parse` parses a JSON object containing parameters for a completion request, validates certain fields, and returns a modified JSON object with parameters suitable for llama.cpp.
- **Inputs**:
    - `body`: A JSON object containing parameters for a completion request, which may include fields like "prompt", "stop", "n", "echo", and others.
- **Control Flow**:
    - Initialize an empty JSON object `llama_params` to store parsed parameters.
    - Check if the "prompt" field is present in `body`; if not, throw a runtime error.
    - If the "stop" field is present and is a string, convert it to a JSON array and add it to `llama_params`; otherwise, use a default value.
    - Retrieve the "n" field from `body`, defaulting to 1, and throw an error if it is not 1.
    - Check the "echo" field and throw an error if it is true, as only no echo is supported.
    - Iterate over a list of unsupported parameters ("best_of", "suffix") and throw an error if any are present in `body`.
    - Copy remaining properties from `body` to `llama_params`, with a special case for "n_predict" to overwrite "max_tokens" if present.
    - Return the `llama_params` JSON object.
- **Output**: A JSON object `llama_params` containing validated and possibly modified parameters for llama.cpp.
- **Functions called**:
    - [`json_value`](#json_value)


---
### oaicompat\_chat\_params\_parse<!-- {{#callable:oaicompat_chat_params_parse}} -->
The `oaicompat_chat_params_parse` function processes and validates JSON input for chat completions, handling various parameters and preparing them for use with the llama.cpp library.
- **Inputs**:
    - `body`: A JSON object representing the input parameters for the OpenAI API, including fields like 'tools', 'stream', 'tool_choice', 'stop', 'response_format', 'messages', etc.
    - `opt`: An `oaicompat_parser_options` struct containing options for parsing, such as whether to use Jinja, allow images or audio, and other configuration settings.
    - `out_files`: A reference to a vector of `raw_buffer` where any processed file data (e.g., images or audio) will be stored.
- **Control Flow**:
    - Initialize a JSON object `llama_params` to store processed parameters.
    - Extract and validate 'tools', 'stream', and 'tool_choice' from the input JSON `body`.
    - Check if Jinja is required for 'tools' and 'tool_choice' and throw an error if not enabled.
    - Process the 'stop' field, ensuring it is an array, and handle potential errors.
    - Validate that 'json_schema' and 'grammar' are not both set, throwing an error if they are.
    - Handle 'response_format' to determine the type and schema, throwing errors for unsupported types.
    - Ensure 'messages' is present and is an array, throwing errors for missing or invalid content.
    - Iterate over 'messages' to validate and process each message, handling content types like 'image_url' and 'input_audio'.
    - Download or decode images and audio, storing the data in `out_files`, and replace content with markers.
    - Prepare `common_chat_templates_inputs` with parsed messages, tools, and other parameters.
    - Apply chat templates to the messages and append any prefilled assistant messages if necessary.
    - Set various fields in `llama_params` based on the processed chat parameters, including 'chat_format', 'prompt', 'grammar', and others.
    - Validate the 'n' field to ensure only one completion choice is allowed, throwing an error otherwise.
    - Handle 'logprobs' field, ensuring compatibility with tools and stream, and set 'n_probs' if applicable.
    - Copy remaining properties from `body` to `llama_params`, allowing for llama.cpp-specific parameters.
- **Output**: A JSON object `llama_params` containing the processed and validated parameters ready for use with the llama.cpp library.
- **Functions called**:
    - [`json_value`](#json_value)
    - [`base64_decode`](#base64_decode)
    - [`mtmd_default_marker`](../mtmd/mtmd.cpp.driver.md#mtmd_default_marker)


---
### format\_embeddings\_response\_oaicompat<!-- {{#callable:format_embeddings_response_oaicompat}} -->
The function `format_embeddings_response_oaicompat` formats a JSON response for embeddings in a way that is compatible with OpenAI's API, optionally encoding the embeddings in base64.
- **Inputs**:
    - `request`: A JSON object containing the request details, including the model name.
    - `embeddings`: A JSON array of embeddings, each containing an 'embedding' and optionally 'tokens_evaluated'.
    - `use_base64`: A boolean flag indicating whether to encode the embeddings in base64 format; defaults to false.
- **Control Flow**:
    - Initialize an empty JSON array `data` and integer `n_tokens` to 0.
    - Iterate over each element in the `embeddings` JSON array.
    - For each element, check if `use_base64` is true.
    - If `use_base64` is true, convert the 'embedding' vector to a base64 string and create a JSON object with 'embedding', 'index', 'object', and 'encoding_format'.
    - If `use_base64` is false, create a JSON object with 'embedding', 'index', and 'object'.
    - Add the created JSON object to the `data` array.
    - Accumulate the 'tokens_evaluated' value from each element into `n_tokens`.
    - Create a JSON response object `res` with 'model', 'object', 'usage', and 'data' fields.
    - Return the JSON response object `res`.
- **Output**: A JSON object containing the formatted response with model information, usage statistics, and the processed embeddings data.
- **Functions called**:
    - [`json_value`](#json_value)


---
### format\_response\_rerank<!-- {{#callable:format_response_rerank}} -->
The `format_response_rerank` function formats a response based on ranking data, either in TEI or Jina format, depending on the specified format flag.
- **Inputs**:
    - `request`: A JSON object containing the request data, which may include parameters like 'return_text' and 'model'.
    - `ranks`: A JSON array of ranking data, where each element contains an 'index' and a 'score', and optionally 'tokens_evaluated'.
    - `is_tei_format`: A boolean flag indicating whether the response should be formatted in TEI format (true) or Jina format (false).
    - `texts`: A vector of strings containing text data, used when 'return_text' is true in TEI format.
- **Control Flow**:
    - Initialize an empty JSON object 'res'.
    - Check if 'is_tei_format' is true.
    - If true, initialize 'res' as a JSON array and check if 'return_text' is true in the 'request'.
    - Iterate over each 'rank' in 'ranks', extract 'index' and 'score', and optionally add 'text' from 'texts' if 'return_text' is true.
    - Push each formatted element into 'res'.
    - If 'is_tei_format' is false, initialize 'results' as a JSON array and 'n_tokens' as 0.
    - Iterate over each 'rank' in 'ranks', extract 'index', 'score', and 'tokens_evaluated', and accumulate 'n_tokens'.
    - Push each formatted element into 'results'.
    - Construct 'res' as a JSON object with 'model', 'object', 'usage', and 'results'.
    - Return the formatted JSON object 'res'.
- **Output**: A JSON object formatted according to the specified response format, containing ranking data and optionally usage statistics.
- **Functions called**:
    - [`json_value`](#json_value)


---
### is\_valid\_utf8<!-- {{#callable:is_valid_utf8}} -->
The function `is_valid_utf8` checks if a given string is a valid UTF-8 encoded string.
- **Inputs**:
    - `str`: A constant reference to a `std::string` that represents the string to be checked for valid UTF-8 encoding.
- **Control Flow**:
    - The function begins by casting the string data to an unsigned char pointer for byte-level operations.
    - It sets a pointer `end` to the end of the byte array to mark the boundary for iteration.
    - A while loop iterates over the bytes of the string until the `bytes` pointer reaches `end`.
    - For each byte, it checks if the byte is a valid UTF-8 lead byte and if the subsequent bytes form a valid UTF-8 sequence.
    - If a 1-byte sequence is detected (0xxxxxxx), it increments the pointer by 1.
    - If a 2-byte sequence is detected (110xxxxx 10xxxxxx), it checks if there are at least 2 bytes remaining and if the second byte is valid; if not, it returns false.
    - If a 3-byte sequence is detected (1110xxxx 10xxxxxx 10xxxxxx), it checks for 3 valid bytes; if not, it returns false.
    - If a 4-byte sequence is detected (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx), it checks for 4 valid bytes; if not, it returns false.
    - If none of these valid sequences are detected, it returns false indicating an invalid UTF-8 sequence.
    - If the loop completes without returning false, it returns true indicating the string is valid UTF-8.
- **Output**: A boolean value: `true` if the string is valid UTF-8, `false` otherwise.


---
### format\_tokenizer\_response<!-- {{#callable:format_tokenizer_response}} -->
The function `format_tokenizer_response` creates a JSON object with a single key-value pair where the key is "tokens" and the value is the input JSON object `tokens`.
- **Inputs**:
    - `tokens`: A JSON object representing tokens, which will be included in the output JSON under the key "tokens".
- **Control Flow**:
    - The function takes a JSON object `tokens` as input.
    - It constructs a new JSON object with a single key-value pair.
    - The key is "tokens" and the value is the input `tokens`.
    - The constructed JSON object is returned.
- **Output**: A JSON object containing the input `tokens` under the key "tokens".


---
### format\_detokenized\_response<!-- {{#callable:format_detokenized_response}} -->
The function `format_detokenized_response` creates a JSON object with a single key-value pair where the key is "content" and the value is the provided string.
- **Inputs**:
    - `content`: A string that represents the content to be included in the JSON response.
- **Control Flow**:
    - The function takes a single string input named `content`.
    - It constructs a JSON object using the `nlohmann::ordered_json` library.
    - The JSON object is initialized with a single key-value pair: the key is "content" and the value is the input string `content`.
    - The function returns the constructed JSON object.
- **Output**: A JSON object with a single key-value pair where the key is "content" and the value is the input string.


---
### format\_logit\_bias<!-- {{#callable:format_logit_bias}} -->
The `format_logit_bias` function converts a vector of `llama_logit_bias` objects into a JSON array format.
- **Inputs**:
    - `logit_bias`: A constant reference to a vector of `llama_logit_bias` objects, each containing a bias and a token.
- **Control Flow**:
    - Initialize a JSON array named `data`.
    - Iterate over each `llama_logit_bias` object in the `logit_bias` vector.
    - For each object, create a JSON object with "bias" and "token" fields and push it into the `data` array.
    - Return the `data` JSON array.
- **Output**: A JSON array where each element is a JSON object representing a `llama_logit_bias` with "bias" and "token" fields.


---
### safe\_json\_to\_str<!-- {{#callable:safe_json_to_str}} -->
The function `safe_json_to_str` converts a JSON object to a string representation with error handling for invalid UTF-8 sequences.
- **Inputs**:
    - `data`: A JSON object from the nlohmann::json library that needs to be converted to a string.
- **Control Flow**:
    - The function calls the `dump` method on the JSON object `data`.
    - It specifies parameters for the `dump` method: `-1` for no indentation, `' '` for the space character, `false` to not ensure ASCII, and `json::error_handler_t::replace` to replace invalid UTF-8 sequences.
- **Output**: A string representation of the JSON object with invalid UTF-8 sequences replaced.


---
### get\_token\_probabilities<!-- {{#callable:get_token_probabilities}} -->
The `get_token_probabilities` function calculates and returns the probabilities of tokens based on their logits for a given index in a llama context.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which contains the context for the llama model.
    - `idx`: An integer representing the index for which the token probabilities are to be calculated.
- **Control Flow**:
    - Retrieve the logits for the specified index using `llama_get_logits_ith` function.
    - Get the model and vocabulary from the context using `llama_get_model` and `llama_model_get_vocab` functions respectively.
    - Determine the number of tokens in the vocabulary using `llama_vocab_n_tokens`.
    - Resize the `cur` vector to hold data for each token in the vocabulary.
    - Populate the `cur` vector with `llama_token_data` objects, each containing a token ID, its corresponding logit, and an initial probability of 0.0.
    - Sort the `cur` vector in descending order based on the logit values.
    - Apply the softmax function to convert logits into probabilities, normalizing them so that they sum to 1.
    - Return the `cur` vector containing the token data with calculated probabilities.
- **Output**: A `std::vector` of `llama_token_data` objects, each containing a token ID, its logit, and its calculated probability.


---
### are\_lora\_equal<!-- {{#callable:are_lora_equal}} -->
The function `are_lora_equal` checks if two vectors of `common_adapter_lora_info` objects are equal based on their size and specific attributes.
- **Inputs**:
    - `l1`: A constant reference to the first vector of `common_adapter_lora_info` objects to be compared.
    - `l2`: A constant reference to the second vector of `common_adapter_lora_info` objects to be compared.
- **Control Flow**:
    - Check if the sizes of the two vectors `l1` and `l2` are different; if so, return `false`.
    - Iterate over the elements of the vectors, comparing the `scale` and `ptr` attributes of corresponding elements.
    - If any pair of elements have different `scale` or `ptr` values, return `false`.
    - If all elements are equal in terms of `scale` and `ptr`, return `true`.
- **Output**: A boolean value indicating whether the two vectors are equal based on the specified criteria.


---
### parse\_lora\_request<!-- {{#callable:parse_lora_request}} -->
The `parse_lora_request` function updates the scale values of a vector of `common_adapter_lora_info` objects based on a JSON input, ensuring valid adapter IDs.
- **Inputs**:
    - `lora_base`: A vector of `common_adapter_lora_info` objects representing the base LoRa configuration.
    - `data`: A JSON object containing the new scale values and corresponding adapter IDs.
- **Control Flow**:
    - Initialize a new vector `lora` as a copy of `lora_base` and determine its size as `max_idx`.
    - Iterate over each entry in `lora` and set its `scale` to 0.0f to clear existing values.
    - Iterate over each entry in the JSON `data`.
    - For each entry, extract the `id` and `scale` values using [`json_value`](#json_value).
    - Check if the `id` is within the valid range (0 to `max_idx - 1`).
    - If valid, update the `scale` of the corresponding `lora` entry with the new `scale` value.
    - If invalid, throw a `std::runtime_error` indicating an invalid adapter ID.
    - Return the updated `lora` vector.
- **Output**: A vector of `common_adapter_lora_info` objects with updated scale values based on the JSON input.
- **Functions called**:
    - [`json_value`](#json_value)


---
### fnv\_hash<!-- {{#callable:fnv_hash}} -->
The `fnv_hash` function computes the FNV-1a hash of a given data buffer and returns it as a string.
- **Inputs**:
    - `data`: A pointer to the data buffer (array of bytes) to be hashed.
    - `len`: The length of the data buffer, indicating how many bytes to hash.
- **Control Flow**:
    - Initialize the FNV-1a hash with a specific offset basis value.
    - Iterate over each byte in the data buffer.
    - For each byte, XOR the hash with the byte and then multiply the hash by the FNV prime.
    - Convert the final hash value to a string and return it.
- **Output**: A string representation of the computed FNV-1a hash value.


