# Purpose
This Python script is designed to manipulate GGUF (Generic Graphical User Format) files by creating a copy with updated metadata. It serves as a command-line utility that allows users to specify an input GGUF file and an output file, along with various options to modify or remove metadata fields. The script leverages the `gguf` package to read and write GGUF files, and it provides functionality to add, modify, or remove metadata fields such as general name, description, chat templates, and special tokens. The script also includes safety checks to prevent accidental overwriting of existing files and warns users about the potential risks of removing essential metadata.

The script is structured around a main function that parses command-line arguments using `argparse`, sets up logging, and manages the flow of reading from the input file, processing metadata, and writing to the output file. Key components include the `MetadataDetails` named tuple for handling metadata attributes, and the [`copy_with_new_metadata`](#cpp/gguf-py/gguf/scripts/gguf_new_metadatacopy_with_new_metadata) function, which performs the core task of transferring and updating metadata between the reader and writer objects. The script is intended to be executed as a standalone tool, as indicated by the `if __name__ == '__main__':` block, and it provides a public interface for users to interact with GGUF files through the command line.
# Imports and Dependencies

---
- `__future__.annotations`
- `logging`
- `argparse`
- `os`
- `sys`
- `json`
- `pathlib.Path`
- `tqdm.tqdm`
- `typing.Any`
- `typing.Sequence`
- `typing.NamedTuple`
- `gguf`


# Global Variables

---
### logger
- **Type**: `logging.Logger`
- **Description**: The `logger` variable is an instance of the `Logger` class from the `logging` module, configured to handle logging for the application with the name 'gguf-new-metadata'. This logger is used to output debug and informational messages throughout the script, aiding in tracking the flow of execution and any issues that arise.
- **Use**: The `logger` is used to log messages at various levels (e.g., debug, info, warning) to provide insights into the application's operations and to help with debugging.


# Classes

---
### MetadataDetails<!-- {{#class:llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.MetadataDetails}} -->
- **Decorators**: `@NamedTuple`
- **Members**:
    - `type`: Specifies the type of the metadata value using gguf.GGUFValueType.
    - `value`: Holds the actual metadata value, which can be of any type.
    - `description`: Provides an optional description for the metadata.
    - `sub_type`: Indicates the sub-type of the metadata if applicable, or None.
- **Description**: The MetadataDetails class is a NamedTuple designed to encapsulate metadata information, including its type, value, optional description, and an optional sub-type. It is used to manage metadata details in the context of GGUF files, providing a structured way to handle metadata attributes.
- **Inherits From**:
    - `NamedTuple`


# Functions

---
### get\_field\_data<!-- {{#callable:llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.get_field_data}} -->
The `get_field_data` function retrieves the contents of a specified field from a GGUFReader object, returning None if the field does not exist.
- **Inputs**:
    - `reader`: A GGUFReader object from which the field data is to be retrieved.
    - `key`: A string representing the key of the field whose data is to be retrieved.
- **Control Flow**:
    - Call the `get_field` method on the `reader` object with `key` to retrieve the field.
    - Check if the field exists; if it does, return the result of calling `contents()` on the field.
    - If the field does not exist, return `None`.
- **Output**: The function returns the contents of the specified field if it exists, otherwise it returns None.


---
### find\_token<!-- {{#callable:llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.find_token}} -->
The `find_token` function searches for a specific token in a list and returns the indices where it is found.
- **Inputs**:
    - `token_list`: A sequence of integers representing the list of tokens to search through.
    - `token`: A string representing the token to find within the token_list.
- **Control Flow**:
    - The function uses a list comprehension to iterate over the token_list, enumerating each element to find indices where the element matches the specified token.
    - If no indices are found (i.e., the token is not present in the list), a LookupError is raised with a message indicating the token was not found.
    - If indices are found, the list of indices is returned.
- **Output**: A sequence of integers representing the indices in the token_list where the specified token is found.


---
### copy\_with\_new\_metadata<!-- {{#callable:llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.copy_with_new_metadata}} -->
The function `copy_with_new_metadata` copies metadata from a GGUFReader to a GGUFWriter, allowing for modifications and removals of specific metadata fields.
- **Inputs**:
    - `reader`: An instance of `gguf.GGUFReader` from which metadata fields are read.
    - `writer`: An instance of `gguf.GGUFWriter` to which metadata fields are written.
    - `new_metadata`: A dictionary mapping field names to [`MetadataDetails`](#cpp/gguf-py/gguf/scripts/gguf_new_metadataMetadataDetails) objects, representing new or modified metadata to be written.
    - `remove_metadata`: A sequence of field names indicating which metadata fields should be removed from the output.
- **Control Flow**:
    - Iterates over each field in the `reader`'s fields.
    - Suppresses fields that are virtual or written by `GGUFWriter`, logging the suppression.
    - Skips old chat templates if new ones are provided in `new_metadata`, logging the skip.
    - Removes fields listed in `remove_metadata`, logging the removal.
    - For each field, determines its type and subtype, and retrieves its current value.
    - Checks if the field is in `new_metadata` to modify it, logging the modification, or copies it if not, logging the copy.
    - Adds the field to the `writer` if its value is not `None`.
    - Adds new chat templates from `new_metadata` to the `writer`, logging the addition.
    - Adds any remaining new metadata fields to the `writer`, logging each addition.
    - Calculates the total bytes of tensor data from the `reader`.
    - Uses `tqdm` to create a progress bar for writing operations.
    - Writes header, key-value data, and tensor info data to the `writer`.
    - Writes tensor data from the `reader` to the `writer`, updating the progress bar.
    - Closes the `writer` after all operations are complete.
- **Output**: The function does not return any value; it performs operations on the `writer` to update it with the new metadata and tensor data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.MetadataDetails`](#cpp/gguf-py/gguf/scripts/gguf_new_metadataMetadataDetails)


---
### main<!-- {{#callable:llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.main}} -->
The `main` function processes command-line arguments to copy a GGUF file with updated metadata, handling special tokens and ensuring data integrity.
- **Inputs**: None
- **Control Flow**:
    - Initialize tokenizer metadata and token names from GGUF keys.
    - Set up an argument parser to handle various command-line options for input/output files and metadata updates.
    - Parse command-line arguments, defaulting to showing help if insufficient arguments are provided.
    - Configure logging based on verbosity flag.
    - Initialize dictionaries for new metadata and metadata to be removed based on parsed arguments.
    - Check and update new metadata entries for general name, description, chat templates, and pre-tokenizer if provided.
    - Warn the user about potential issues with removing metadata and require confirmation unless forced.
    - Load the input GGUF file using a GGUFReader and retrieve architecture and token list data.
    - Process special tokens by value or ID, updating new metadata accordingly and handling errors or warnings for unknown tokens.
    - Check if the output file already exists and warn the user, requiring confirmation to overwrite unless forced.
    - Create a GGUFWriter for the output file, setting custom alignment if specified.
    - Call [`copy_with_new_metadata`](#cpp/gguf-py/gguf/scripts/gguf_new_metadatacopy_with_new_metadata) to perform the actual copying and metadata updating process.
- **Output**: The function does not return any value; it performs file operations and updates metadata in the specified GGUF file.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.MetadataDetails`](#cpp/gguf-py/gguf/scripts/gguf_new_metadataMetadataDetails)
    - [`llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.get_field_data`](#cpp/gguf-py/gguf/scripts/gguf_new_metadataget_field_data)
    - [`llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.find_token`](#cpp/gguf-py/gguf/scripts/gguf_new_metadatafind_token)
    - [`llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.copy_with_new_metadata`](#cpp/gguf-py/gguf/scripts/gguf_new_metadatacopy_with_new_metadata)


