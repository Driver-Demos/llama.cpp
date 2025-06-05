# Purpose
The provided Python code defines a class `TensorNameMap` that serves as a mapping utility for tensor names across different model architectures. This class is designed to facilitate the translation of tensor names from various model implementations into a standardized format. The code includes a comprehensive set of mappings for different tensor types, such as token embeddings, attention mechanisms, and feed-forward layers, across a wide range of model architectures like GPT, BERT, and T5, among others. The mappings are organized into dictionaries, `mappings_cfg` and `block_mappings_cfg`, which associate model tensor types with their corresponding names in different architectures. Additionally, the class supports architecture-specific block mappings through `arch_block_mappings_cfg`.

The `TensorNameMap` class is initialized with a specific model architecture and the number of blocks in the model, allowing it to generate a complete mapping for that configuration. It provides methods to retrieve the type and name of a tensor given a key, with optional suffix handling for more flexible lookups. The class also implements dictionary-like behavior, allowing for key-based access to tensor names. The function [`get_tensor_name_map`](#cpp/gguf-py/gguf/tensor_mappingget_tensor_name_map) is a utility function that instantiates a `TensorNameMap` object for a given architecture and block count. This code is likely part of a larger system that deals with model conversion or interoperability, ensuring that tensor names are consistently interpreted across different model implementations.
# Imports and Dependencies

---
- `__future__.annotations`
- `typing.Sequence`
- `.constants.MODEL_ARCH`
- `.constants.MODEL_TENSOR`
- `.constants.MODEL_TENSORS`
- `.constants.TENSOR_NAMES`


# Classes

---
### TensorNameMap<!-- {{#class:llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap}} -->
- **Members**:
    - `mappings_cfg`: A dictionary mapping MODEL_TENSOR types to tuples of string identifiers for various model architectures.
    - `block_mappings_cfg`: A dictionary mapping MODEL_TENSOR types to tuples of string identifiers for block-specific configurations.
    - `arch_block_mappings_cfg`: A dictionary mapping MODEL_ARCH types to block-specific configurations for different architectures.
    - `mapping`: A dictionary mapping string keys to tuples of MODEL_TENSOR and string identifiers.
- **Description**: The TensorNameMap class is designed to manage and map tensor names across different model architectures and configurations. It maintains mappings for both general and block-specific tensor names, allowing for flexible retrieval of tensor types and names based on model architecture and block index. The class supports initialization with a specific architecture and number of blocks, updating its mappings accordingly, and provides methods to retrieve tensor types and names, ensuring compatibility with various model configurations.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.__init__`](#TensorNameMap__init__)
    - [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.get_type_and_name`](#TensorNameMapget_type_and_name)
    - [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.get_name`](#TensorNameMapget_name)
    - [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.get_type`](#TensorNameMapget_type)
    - [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.__getitem__`](#TensorNameMap__getitem__)
    - [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.__contains__`](#TensorNameMap__contains__)
    - [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.__repr__`](#TensorNameMap__repr__)

**Methods**

---
#### TensorNameMap\.\_\_init\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.__init__}} -->
The `__init__` method initializes a `TensorNameMap` object by setting up a mapping of tensor names to their corresponding model tensor types and names based on the specified architecture and number of blocks.
- **Inputs**:
    - `arch`: The architecture type of the model, specified as a `MODEL_ARCH` enum.
    - `n_blocks`: The number of blocks in the model, specified as an integer.
- **Control Flow**:
    - Initialize an empty dictionary `self.mapping` to store the tensor mappings.
    - Iterate over each tensor and its associated keys in `self.mappings_cfg`.
    - For each tensor, check if it is present in `MODEL_TENSORS` for the given architecture; if not, skip to the next tensor.
    - Retrieve the tensor name from `TENSOR_NAMES` and add it to `self.mapping` with the tensor and its name as a tuple.
    - For each key associated with the tensor, add it to `self.mapping` with the same tuple of tensor and name.
    - Check if the architecture is present in `self.arch_block_mappings_cfg`; if so, update `self.block_mappings_cfg` with the architecture-specific mappings.
    - Iterate over each block index from 0 to `n_blocks - 1`.
    - For each block, iterate over each tensor and its associated keys in `self.block_mappings_cfg`.
    - Check if the tensor is present in `MODEL_TENSORS` for the given architecture; if not, skip to the next tensor.
    - Format the tensor name with the current block index and add it to `self.mapping` with the tensor and formatted name as a tuple.
    - For each key associated with the tensor, format it with the current block index and add it to `self.mapping` with the same tuple of tensor and formatted name.
- **Output**: The method does not return any value; it initializes the `self.mapping` attribute of the `TensorNameMap` instance.
- **See also**: [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap`](#cpp/gguf-py/gguf/tensor_mappingTensorNameMap)  (Base Class)


---
#### TensorNameMap\.get\_type\_and\_name<!-- {{#callable:llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.get_type_and_name}} -->
The `get_type_and_name` method retrieves the tensor type and name associated with a given key, optionally considering suffixes.
- **Inputs**:
    - `key`: A string representing the key to look up in the mapping.
    - `try_suffixes`: An optional sequence of string suffixes to try if the key is not found directly.
- **Control Flow**:
    - Attempt to retrieve the value associated with the key from the mapping.
    - If the key is found, return the associated tuple (tensor type and name).
    - If the key is not found, iterate over the provided suffixes.
    - For each suffix, check if the key ends with the suffix.
    - If it does, attempt to retrieve the value associated with the key minus the suffix from the mapping.
    - If a match is found, return the tensor type and the name concatenated with the suffix.
    - If no match is found after checking all suffixes, return None.
- **Output**: Returns a tuple containing the tensor type and name if found, or None if no match is found.
- **See also**: [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap`](#cpp/gguf-py/gguf/tensor_mappingTensorNameMap)  (Base Class)


---
#### TensorNameMap\.get\_name<!-- {{#callable:llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.get_name}} -->
The `get_name` method retrieves the name associated with a given key from a mapping, optionally considering suffixes.
- **Inputs**:
    - `key`: A string representing the key for which the associated name is to be retrieved.
    - `try_suffixes`: An optional sequence of string suffixes to try if the key is not directly found in the mapping.
- **Control Flow**:
    - Call the [`get_type_and_name`](#TensorNameMapget_type_and_name) method with the provided key and suffixes to retrieve a tuple containing the type and name associated with the key.
    - Check if the result from [`get_type_and_name`](#TensorNameMapget_type_and_name) is `None`.
    - If the result is `None`, return `None`.
    - If the result is not `None`, return the second element of the tuple, which is the name.
- **Output**: Returns a string representing the name associated with the key, or `None` if no association is found.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.get_type_and_name`](#TensorNameMapget_type_and_name)
- **See also**: [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap`](#cpp/gguf-py/gguf/tensor_mappingTensorNameMap)  (Base Class)


---
#### TensorNameMap\.get\_type<!-- {{#callable:llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.get_type}} -->
The `get_type` method retrieves the type of a model tensor associated with a given key, optionally considering suffixes.
- **Inputs**:
    - `key`: A string representing the key for which the tensor type is to be retrieved.
    - `try_suffixes`: An optional sequence of string suffixes to try if the key is not found directly.
- **Control Flow**:
    - Call the [`get_type_and_name`](#TensorNameMapget_type_and_name) method with the provided key and suffixes.
    - Check if the result is `None`; if so, return `None`.
    - If a result is found, return the first element of the result, which is the tensor type.
- **Output**: Returns a `MODEL_TENSOR` type if found, otherwise `None`.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.get_type_and_name`](#TensorNameMapget_type_and_name)
- **See also**: [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap`](#cpp/gguf-py/gguf/tensor_mappingTensorNameMap)  (Base Class)


---
#### TensorNameMap\.\_\_getitem\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.__getitem__}} -->
The `__getitem__` method retrieves the tensor name associated with a given key from the `mapping` dictionary, raising a `KeyError` if the key is not found.
- **Inputs**:
    - `key`: A string representing the key for which the associated tensor name is to be retrieved from the `mapping` dictionary.
- **Control Flow**:
    - Attempts to return the second element of the tuple associated with the given `key` in the `mapping` dictionary.
    - If the `key` is not found in the `mapping` dictionary, a `KeyError` is raised with the `key` as the error message.
- **Output**: Returns a string, which is the tensor name associated with the given `key` in the `mapping` dictionary.
- **See also**: [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap`](#cpp/gguf-py/gguf/tensor_mappingTensorNameMap)  (Base Class)


---
#### TensorNameMap\.\_\_contains\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.__contains__}} -->
The `__contains__` method checks if a given key exists in the `mapping` dictionary of the `TensorNameMap` class.
- **Inputs**:
    - `key`: A string representing the key to be checked for existence in the `mapping` dictionary.
- **Control Flow**:
    - The method uses the `in` keyword to check if the `key` is present in the `mapping` dictionary.
- **Output**: A boolean value indicating whether the `key` is present in the `mapping` dictionary.
- **See also**: [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap`](#cpp/gguf-py/gguf/tensor_mappingTensorNameMap)  (Base Class)


---
#### TensorNameMap\.\_\_repr\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap.__repr__}} -->
The `__repr__` method returns a string representation of the `mapping` attribute of the `TensorNameMap` class.
- **Inputs**: None
- **Control Flow**:
    - The method directly returns the string representation of the `mapping` attribute using the built-in `repr` function.
- **Output**: A string that represents the `mapping` attribute of the `TensorNameMap` instance.
- **See also**: [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap`](#cpp/gguf-py/gguf/tensor_mappingTensorNameMap)  (Base Class)



# Functions

---
### get\_tensor\_name\_map<!-- {{#callable:llama.cpp/gguf-py/gguf/tensor_mapping.get_tensor_name_map}} -->
The `get_tensor_name_map` function creates and returns a [`TensorNameMap`](#cpp/gguf-py/gguf/tensor_mappingTensorNameMap) object for a given model architecture and number of blocks.
- **Inputs**:
    - `arch`: The model architecture, specified as a `MODEL_ARCH` type, which determines the configuration of the tensor mappings.
    - `n_blocks`: An integer representing the number of blocks in the model, used to configure block-specific tensor mappings.
- **Control Flow**:
    - The function calls the [`TensorNameMap`](#cpp/gguf-py/gguf/tensor_mappingTensorNameMap) constructor with the provided `arch` and `n_blocks` arguments.
    - The [`TensorNameMap`](#cpp/gguf-py/gguf/tensor_mappingTensorNameMap) constructor initializes a mapping of tensor names based on the architecture and number of blocks.
    - The constructor iterates over predefined tensor mappings and block mappings, updating the internal mapping dictionary with formatted tensor names.
- **Output**: A [`TensorNameMap`](#cpp/gguf-py/gguf/tensor_mappingTensorNameMap) object configured with tensor mappings for the specified architecture and number of blocks.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/tensor_mapping.TensorNameMap`](#cpp/gguf-py/gguf/tensor_mappingTensorNameMap)


