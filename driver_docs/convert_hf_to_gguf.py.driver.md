# Purpose
The provided Python script is a versatile utility designed to convert Hugging Face models into the GGUF (Generic Graph Universal Format), catering to machine learning and natural language processing applications that require compatibility with specific execution environments. It is structured to support a wide array of model architectures, including LLaMA, GPTNeoX, Bloom, T5, and ChatGLM, by employing a modular class-based approach with a base class `ModelBase` and subclasses for each model type. The script processes model configurations and weights, handling tasks such as tensor transformations, vocabulary processing, and metadata extraction, and writes the results into GGUF files. It is intended to be executed as a standalone command-line tool, offering a command-line interface (CLI) for users to specify conversion options, including output file types and handling of remote models from the Hugging Face Hub. This extensible and flexible script serves as a bridge between Hugging Face models and the GGUF format, without defining public APIs or external interfaces beyond its CLI.
# Imports and Dependencies

---
- `__future__.annotations`
- `ast`
- `logging`
- `argparse`
- `contextlib`
- `json`
- `os`
- `re`
- `sys`
- `enum.IntEnum`
- `pathlib.Path`
- `hashlib.sha256`
- `typing.TYPE_CHECKING`
- `typing.Any`
- `typing.Callable`
- `typing.ContextManager`
- `typing.Iterable`
- `typing.Iterator`
- `typing.Literal`
- `typing.Sequence`
- `typing.TypeVar`
- `typing.cast`
- `itertools.chain`
- `transformers.AutoConfig`
- `math`
- `numpy`
- `torch`
- `torch.Tensor`
- `gguf`
- `safetensors.safe_open`
- `transformers.AutoTokenizer`
- `sentencepiece.SentencePieceProcessor`
- `copy`
- `transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode`
- `sentencepiece.sentencepiece_model_pb2`
- `base64.b64decode`
- `huggingface_hub.snapshot_download`


# Global Variables

---
### logger
- **Type**: `logging.Logger`
- **Description**: The `logger` variable is an instance of the `Logger` class from the Python `logging` module. It is configured to log messages for the application or module identified by the name 'hf-to-gguf'. This allows for centralized and consistent logging throughout the application, facilitating debugging and monitoring.
- **Use**: This variable is used to log messages with varying levels of severity, such as debug, info, warning, error, and critical, for the 'hf-to-gguf' application or module.


---
### AnyModel
- **Type**: `TypeVar`
- **Description**: `AnyModel` is a type variable that is constrained to types that are subclasses of `ModelBase`. This allows for type checking and ensures that any type assigned to `AnyModel` is a subtype of `ModelBase`. It is used in generic programming to create flexible and reusable code components that can operate on any model type derived from `ModelBase`. 
- **Use**: `AnyModel` is used to define generic functions or classes that can work with any model type that is a subclass of `ModelBase`.


# Classes

---
### SentencePieceTokenTypes<!-- {{#class:llama.cpp/convert_hf_to_gguf.SentencePieceTokenTypes}} -->
- **Members**:
    - `NORMAL`: Represents a normal token type.
    - `UNKNOWN`: Represents an unknown token type.
    - `CONTROL`: Represents a control token type.
    - `USER_DEFINED`: Represents a user-defined token type.
    - `UNUSED`: Represents an unused token type.
    - `BYTE`: Represents a byte token type.
- **Description**: The `SentencePieceTokenTypes` class is an enumeration that defines various types of tokens used in sentence piece processing, categorizing them into normal, unknown, control, user-defined, unused, and byte types.
- **Inherits From**:
    - `IntEnum`


---
### ModelType<!-- {{#class:llama.cpp/convert_hf_to_gguf.ModelType}} -->
- **Members**:
    - `TEXT`: Represents the text model type with a value of 1.
    - `MMPROJ`: Represents the MMPROJ model type with a value of 2.
- **Description**: `ModelType` is an enumeration class that defines two model types, `TEXT` and `MMPROJ`, each associated with a unique integer value.
- **Inherits From**:
    - `IntEnum`


---
### ModelBase<!-- {{#class:llama.cpp/convert_hf_to_gguf.ModelBase}} -->
- **Members**:
    - `_model_classes`: A class variable that maps model types to their corresponding model classes.
    - `dir_model`: The directory path where the model files are located.
    - `ftype`: The file type of the model, specified as a `gguf.LlamaFileType`.
    - `fname_out`: The output file name for the processed model.
    - `is_big_endian`: A boolean indicating if the model uses big-endian byte order.
    - `endianess`: The endianess of the model, represented as a `gguf.GGUFEndian`.
    - `use_temp_file`: A boolean indicating if a temporary file should be used.
    - `lazy`: A boolean indicating if lazy loading of tensors is enabled.
    - `part_names`: A list of part names for the model.
    - `is_safetensors`: A boolean indicating if the model is in safetensors format.
    - `hparams`: A dictionary containing hyperparameters for the model.
    - `tensor_names`: A set of tensor names or None if not initialized.
    - `gguf_writer`: An instance of `gguf.GGUFWriter` for writing model data.
    - `model_name`: The name of the model or None if not specified.
    - `metadata_override`: A path for overriding metadata or None.
    - `dir_model_card`: The directory path for the model card.
    - `remote_hf_model_id`: The Hugging Face model ID for remote models.
    - `model_arch`: A class variable that should be defined by subclasses to specify the model architecture.
    - `block_count`: A class variable that should be initialized by subclasses to specify the block count.
    - `tensor_map`: A class variable that should be initialized by subclasses to specify the tensor mapping.
- **Description**: The `ModelBase` class serves as a base for model handling, providing a structure for loading, processing, and writing model data, while enforcing that subclasses define specific model architecture and tensor mapping details.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.add_prefix_to_filename`](#ModelBaseadd_prefix_to_filename)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.get_tensors`](#ModelBaseget_tensors)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.match_model_tensor_name`](#ModelBasematch_model_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.modify_tensors`](#ModelBasemodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.tensor_force_quant`](#ModelBasetensor_force_quant)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.generate_extra_tensors`](#ModelBasegenerate_extra_tensors)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_type`](#ModelBaseset_type)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_metadata`](#ModelBaseprepare_metadata)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.write_vocab`](#ModelBasewrite_vocab)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.write`](#ModelBasewrite)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.get_model_part_names`](#ModelBaseget_model_part_names)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.load_hparams`](#ModelBaseload_hparams)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.register`](#ModelBaseregister)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.print_registered_models`](#ModelBaseprint_registered_models)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.from_model_architecture`](#ModelBasefrom_model_architecture)

**Methods**

---
#### ModelBase\.add\_prefix\_to\_filename<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.add_prefix_to_filename}} -->
This method adds a specified prefix to the filename of a given file path.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `path`: A `Path` object representing the file path to which the prefix will be added.
    - `prefix`: A string that represents the prefix to be added to the filename.
- **Control Flow**:
    - The method extracts the stem (filename without extension) and suffix (file extension) from the provided `path`.
    - It constructs a new filename by concatenating the `prefix`, `stem`, and `suffix`.
    - Finally, it returns a new `Path` object with the updated filename.
- **Output**: Returns a new `Path` object with the updated filename that includes the specified prefix.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.find\_hparam<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam}} -->
The `find_hparam` method retrieves a hyperparameter value from the model's hyperparameters based on a list of provided keys.
- **Inputs**:
    - `keys`: An iterable of strings representing the keys to search for in the hyperparameters.
    - `optional`: A boolean flag indicating whether to return None if no key is found; defaults to False.
- **Control Flow**:
    - The method uses a generator expression to find the first key from `keys` that exists in `self.hparams`.
    - If a matching key is found, the corresponding value from `self.hparams` is returned.
    - If no key is found and `optional` is True, the method returns None.
    - If no key is found and `optional` is False, a KeyError is raised with a message indicating the missing keys.
- **Output**: The method returns the value associated with the found key in `self.hparams`, None if `optional` is True and no key is found, or raises a KeyError if no key is found and `optional` is False.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.get\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.get_tensors}} -->
The `get_tensors` method retrieves tensor data from model files, yielding each tensor's name and data.
- **Inputs**: None
- **Control Flow**:
    - Initializes a set to track tensor names from model parts.
    - Determines the index file name based on whether the model is in safetensors format.
    - Checks if the index file exists; if it does, loads the weight map and updates tensor names.
    - Iterates over model part names, loading each part and extracting tensor data.
    - Handles both safetensors and regular PyTorch model formats for loading tensors.
    - Yields each tensor's name and data as a tuple.
    - Verifies the presence of tensor names and checks for any discrepancies between the loaded tensors and the weight map.
- **Output**: Yields tuples containing the name and data of each tensor loaded from the model parts.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LazyTorchTensor.from_safetensors_slice`](#LazyTorchTensorfrom_safetensors_slice)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.format\_tensor\_name<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name}} -->
Formats the name of a tensor based on a given key, optional block ID, and suffix.
- **Inputs**:
    - `key`: A tensor key of type `gguf.MODEL_TENSOR` used to retrieve the tensor name.
    - `bid`: An optional integer representing the block ID, used for formatting the tensor name.
    - `suffix`: A string suffix to append to the tensor name, defaulting to '.weight'.
- **Control Flow**:
    - Checks if the provided `key` exists in the `MODEL_TENSORS` for the current model architecture, raising a ValueError if not.
    - Retrieves the base tensor name associated with the `key`.
    - If the base name contains a placeholder for `bid`, it asserts that `bid` is not None and formats the name with the provided `bid`.
    - Returns the formatted tensor name concatenated with the specified `suffix`.
- **Output**: Returns the formatted tensor name as a string.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.match\_model\_tensor\_name<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.match_model_tensor_name}} -->
The `match_model_tensor_name` method checks if a given tensor name matches the expected format based on a specified key and optional block ID.
- **Inputs**:
    - `name`: A string representing the tensor name to be matched.
    - `key`: A key of type `gguf.MODEL_TENSOR` that identifies the expected tensor.
    - `bid`: An optional integer representing the block ID, or None if not applicable.
    - `suffix`: A string that defaults to '.weight', which is appended to the tensor name for matching.
- **Control Flow**:
    - The method first checks if the provided `key` exists in the `MODEL_TENSORS` dictionary for the current model architecture.
    - If the `key` is valid, it retrieves the corresponding tensor name from `TENSOR_NAMES`.
    - If the tensor name contains a placeholder for `bid`, it checks if `bid` is provided; if not, it returns False.
    - If the tensor name does not contain a placeholder but `bid` is provided, it also returns False.
    - Finally, it compares the constructed tensor name (with the suffix) to the provided `name` and returns True if they match, otherwise False.
- **Output**: Returns a boolean indicating whether the provided tensor name matches the expected name based on the key and optional block ID.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.map\_tensor\_name<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name}} -->
Maps a tensor name to a new name using a specified suffix.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `name`: The original name of the tensor to be mapped.
    - `try_suffixes`: A sequence of suffixes to try appending to the original name for mapping.
- **Control Flow**:
    - Calls `get_name` method on `self.tensor_map` to attempt to find a new name for the provided tensor name.
    - If no new name is found, raises a `ValueError` indicating that the tensor cannot be mapped.
    - Returns the new name if found.
- **Output**: Returns the mapped tensor name as a string.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters}} -->
`set_gguf_parameters` is an abstract method that must be implemented by subclasses of `ModelBase`.
- **Inputs**: None
- **Control Flow**:
    - The method raises a `NotImplementedError` to enforce that subclasses must provide their own implementation.
- **Output**: The method does not return any value; it is intended to be overridden in subclasses.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.modify_tensors}} -->
The `modify_tensors` method maps a tensor name and returns a tuple containing the mapped name and the tensor.
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the data to be modified.
    - `name`: A string representing the name of the tensor.
    - `bid`: An optional integer that is currently unused in the method.
- **Control Flow**:
    - The method begins by deleting the `bid` parameter as it is not used.
    - It then calls the [`map_tensor_name`](#ModelBasemap_tensor_name) method to get the mapped name for the provided tensor name.
    - Finally, it returns a list containing a tuple of the mapped name and the original tensor.
- **Output**: An iterable of tuples, each containing a string (the mapped tensor name) and a `Tensor` (the original tensor).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.tensor\_force\_quant<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.tensor_force_quant}} -->
The `tensor_force_quant` method is a placeholder that currently does not perform any operations and always returns False.
- **Inputs**:
    - `name`: A string representing the original name of the tensor.
    - `new_name`: A string representing the new name to be assigned to the tensor.
    - `bid`: An optional integer representing the block ID associated with the tensor.
    - `n_dims`: An integer representing the number of dimensions of the tensor.
- **Control Flow**:
    - The method begins by deleting the input parameters as they are not used in the current implementation.
    - It then returns a boolean value of False, indicating that no quantization is performed.
- **Output**: The method returns False, indicating that no quantization operation was performed.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.generate\_extra\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.generate_extra_tensors}} -->
Generates additional tensors for a model, but currently returns an empty iterable.
- **Inputs**: None
- **Control Flow**:
    - The method is defined to return an iterable of tuples containing string and Tensor, but it currently has no implementation.
    - It directly returns an empty tuple, indicating that no extra tensors are generated.
- **Output**: Returns an empty iterable, indicating that there are no additional tensors generated by this method.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors}} -->
Prepares and quantizes tensors for a model, ensuring they are in the correct format and type.
- **Inputs**: None
- **Control Flow**:
    - Calculates the maximum length of tensor names to format logging output.
    - Iterates over generated and retrieved tensors, skipping certain tensor names.
    - Converts unsupported tensor data types to float32.
    - Extracts a block ID from the tensor name if it contains a decimal part.
    - Modifies tensor names and data using the [`modify_tensors`](#ModelBasemodify_tensors) method.
    - Handles quantization of tensor data based on various conditions and types.
    - Logs the old and new data types along with the shape of the tensor.
    - Adds the processed tensor to the `gguf_writer` for output.
- **Output**: The method does not return a value but modifies the internal state by adding processed tensors to the `gguf_writer`.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.generate_extra_tensors`](#ModelBasegenerate_extra_tensors)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.get_tensors`](#ModelBaseget_tensors)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.modify_tensors`](#ModelBasemodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.LazyTorchTensor.numpy`](#LazyTorchTensornumpy)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.tensor_force_quant`](#ModelBasetensor_force_quant)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.match_model_tensor_name`](#ModelBasematch_model_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.set\_type<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.set_type}} -->
Sets the type of the GGUF writer to indicate that the model is of type MODEL.
- **Inputs**: None
- **Control Flow**:
    - Calls the `add_type` method on the `gguf_writer` instance, passing in the `gguf.GGUFType.MODEL` constant.
- **Output**: No output is returned; the method modifies the state of the `gguf_writer` by setting its type.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.prepare\_metadata<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.prepare_metadata}} -->
Prepares and sets metadata for a model, including parameter counts and model identification.
- **Inputs**:
    - `vocab_only`: A boolean flag indicating whether to prepare metadata only for vocabulary.
- **Control Flow**:
    - Retrieves total parameter counts and initializes metadata using the provided model information.
    - Sets the metadata name based on the remote Hugging Face model ID or falls back to the model directory name.
    - Generates a size label for the model if it has not been set and total parameters are greater than zero.
    - Calls methods to set the model type, parameters, and quantization version, logging each step.
- **Output**: The method does not return a value but updates the instance's metadata attribute with the prepared metadata.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_type`](#ModelBaseset_type)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.write\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.write_vocab}} -->
The `write_vocab` method is an abstract method that must be implemented by subclasses of `ModelBase`.
- **Inputs**: None
- **Control Flow**:
    - The method raises a `NotImplementedError`, indicating that it is intended to be overridden in a subclass.
- **Output**: The method does not return any value; it is meant to enforce implementation in derived classes.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.write<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.write}} -->
The `write` method orchestrates the process of preparing tensors and metadata, and writing them to a specified output file.
- **Inputs**: None
- **Control Flow**:
    - Calls `self.prepare_tensors()` to prepare the tensor data for writing.
    - Calls `self.prepare_metadata(vocab_only=False)` to prepare the metadata associated with the model.
    - Invokes `self.gguf_writer.write_header_to_file(path=self.fname_out)` to write the header to the output file.
    - Calls `self.gguf_writer.write_kv_data_to_file()` to write key-value data to the output file.
    - Calls `self.gguf_writer.write_tensors_to_file(progress=True)` to write the tensor data to the output file with progress indication.
    - Finally, calls `self.gguf_writer.close()` to close the writer and finalize the file.
- **Output**: The method does not return a value; instead, it writes the prepared tensors and metadata to a file specified by `self.fname_out`.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_metadata`](#ModelBaseprepare_metadata)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.get\_model\_part\_names<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.get_model_part_names}} -->
Retrieves and sorts model part filenames from a specified directory based on given prefix and suffix.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `dir_model`: A `Path` object representing the directory containing model files.
    - `prefix`: A string that specifies the prefix that model filenames should start with.
    - `suffix`: A string that specifies the suffix that model filenames should end with.
- **Control Flow**:
    - Initializes an empty list `part_names` to store valid filenames.
    - Iterates over each filename in the specified directory using `os.listdir(dir_model)`.
    - Checks if each filename starts with the specified `prefix` and ends with the specified `suffix`.
    - If a filename matches the criteria, it is appended to the `part_names` list.
    - After collecting all valid filenames, the list is sorted alphabetically.
    - Returns the sorted list of model part names.
- **Output**: A sorted list of strings representing the filenames of model parts that match the specified prefix and suffix.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.load\_hparams<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.load_hparams}} -->
Loads hyperparameters from a specified model directory, attempting to read from a configuration file and falling back to a default if necessary.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `dir_model`: A `Path` object representing the directory where the model's configuration files are located.
- **Control Flow**:
    - Attempts to load the model configuration using `AutoConfig.from_pretrained` with security settings to prevent loading remote code.
    - If the initial loading fails, it logs a warning and attempts to load a `config.json` file from the specified directory.
    - Checks for specific keys in the loaded configuration and renames them for compatibility with different model architectures.
    - Returns the final configuration dictionary.
- **Output**: Returns a dictionary containing the model's hyperparameters and configuration settings.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.register<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.register}} -->
The `register` method is a class method that registers model classes under specified names in a class-level dictionary based on their architecture type.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `cls`: The class itself, which is automatically passed to class methods.
    - `names`: A variable-length list of strings representing the names under which the model class will be registered.
    - `modelcls`: The model class that is being registered, which is passed to the inner function.
- **Control Flow**:
    - The method asserts that at least one name is provided.
    - It defines an inner function `func` that takes a model class as an argument.
    - Inside `func`, it determines the model type based on the architecture of the provided model class.
    - It iterates over the provided names and registers the model class in the `_model_classes` dictionary under the determined model type.
    - Finally, it returns the model class.
- **Output**: The output of the `register` method is the inner function `func`, which when called, returns the model class after registering it.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.print\_registered\_models<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.print_registered_models}} -->
Prints the names of all registered model classes grouped by their model type.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `cls`: The class itself, used to access class-level attributes and methods.
- **Control Flow**:
    - Iterates over the `_model_classes` dictionary, which contains model types as keys and their corresponding model classes as values.
    - Logs an error message for each model type indicating the type of models being printed.
    - Sorts the model class names alphabetically and logs each name under its respective model type.
- **Output**: No return value; the method outputs error logs containing the names of registered models.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)


---
#### ModelBase\.from\_model\_architecture<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ModelBase.from_model_architecture}} -->
This class method retrieves a model class based on the specified architecture and model type.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `arch`: A string representing the architecture of the model to retrieve.
    - `model_type`: An optional parameter indicating the type of model, defaulting to ModelType.TEXT.
- **Control Flow**:
    - The method attempts to access the `_model_classes` dictionary using the provided `model_type` and `arch` as keys.
    - If the specified architecture is not found, a `NotImplementedError` is raised with a message indicating that the architecture is not supported.
- **Output**: Returns the model class associated with the specified architecture and model type.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)  (Base Class)



---
### TextModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.TextModel}} -->
- **Members**:
    - `model_type`: Specifies the type of model as TEXT.
    - `hf_arch`: Holds the architecture of the Hugging Face model.
    - `block_count`: Represents the number of blocks in the model.
    - `tensor_map`: Maps tensor names based on the model architecture and block count.
- **Description**: The `TextModel` class extends `ModelBase` and is designed to handle text-based models, managing their architecture, configuration, and vocabulary, while providing methods for metadata preparation and vocabulary management.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel.__init__`](#TextModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.TextModel.__init_subclass__`](#TextModel__init_subclass__)
    - [`llama.cpp/convert_hf_to_gguf.TextModel.set_vocab`](#TextModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.TextModel.prepare_metadata`](#TextModelprepare_metadata)
    - [`llama.cpp/convert_hf_to_gguf.TextModel.set_gguf_parameters`](#TextModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.TextModel.write_vocab`](#TextModelwrite_vocab)
    - [`llama.cpp/convert_hf_to_gguf.TextModel.does_token_look_special`](#TextModeldoes_token_look_special)
    - [`llama.cpp/convert_hf_to_gguf.TextModel.get_vocab_base`](#TextModelget_vocab_base)
    - [`llama.cpp/convert_hf_to_gguf.TextModel.get_vocab_base_pre`](#TextModelget_vocab_base_pre)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_none`](#TextModel_set_vocab_none)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_qwen`](#TextModel_set_vocab_qwen)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._create_vocab_sentencepiece`](#TextModel_create_vocab_sentencepiece)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_llama_hf`](#TextModel_set_vocab_llama_hf)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_rwkv_world`](#TextModel_set_vocab_rwkv_world)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_builtin`](#TextModel_set_vocab_builtin)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._try_set_pooling_type`](#TextModel_try_set_pooling_type)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)

**Methods**

---
#### TextModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel.__init__}} -->
Initializes a `TextModel` instance by setting up model architecture, hyperparameters, and tensor mappings.
- **Inputs**:
    - `args`: Positional arguments passed to the parent class initializer.
    - `kwargs`: Keyword arguments passed to the parent class initializer.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method to ensure proper initialization.
    - Retrieves the model architecture based on hyperparameters and model type.
    - Checks if 'text_config' is present in hyperparameters and merges it into the root level of `hparams`.
    - Determines the block count by searching for specific hyperparameter keys.
    - Generates a tensor name map based on the model architecture and block count.
- **Output**: No explicit output; initializes the instance variables of the `TextModel` class.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
    - [`llama.cpp/convert_hf_to_gguf.get_model_architecture`](#cpp/convert_hf_to_ggufget_model_architecture)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.\_\_init\_subclass\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel.__init_subclass__}} -->
The `__init_subclass__` method ensures that any subclass of `TextModel` defines a `model_arch` property.
- **Decorators**: `@classmethod`
- **Inputs**: None
- **Control Flow**:
    - Checks if the `model_arch` property is defined in the subclass's dictionary.
    - Raises a `TypeError` if `model_arch` is not found, indicating that the subclass is improperly defined.
- **Output**: The method does not return a value; it raises an exception if the required property is missing.
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel.set_vocab}} -->
The `set_vocab` method initializes the vocabulary for the text model using the GPT-2 tokenizer.
- **Inputs**: None
- **Control Flow**:
    - The method calls the [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) private method to set the vocabulary specifically for the GPT-2 model.
    - The [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) method retrieves the vocabulary base and adds it to the `gguf_writer` along with the tokenizer model and special vocabulary.
- **Output**: The method does not return a value; it modifies the internal state of the `gguf_writer` to include the GPT-2 vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.prepare\_metadata<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel.prepare_metadata}} -->
Prepares metadata for the model by generating a filename based on the model's specifications and setting the vocabulary.
- **Inputs**:
    - `vocab_only`: A boolean flag indicating whether to prepare metadata for vocabulary only.
- **Control Flow**:
    - Calls the parent class's [`prepare_metadata`](#ModelBaseprepare_metadata) method with the `vocab_only` argument.
    - Retrieves the total parameter count from the `gguf_writer`.
    - Extracts the encoding scheme from the file type name.
    - Checks if the output filename path is a directory.
    - If it is a directory and `vocab_only` is false, generates a default filename using model metadata; otherwise, generates a filename for vocabulary only.
    - If the output path is not a directory, processes a templated filename based on the output file type.
    - Logs the action of setting the model tokenizer and calls the [`set_vocab`](#TextModelset_vocab) method.
- **Output**: The method does not return a value but updates the `fname_out` attribute with the generated filename and prepares the vocabulary for the model.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_metadata`](#ModelBaseprepare_metadata)
    - [`llama.cpp/convert_hf_to_gguf.TextModel.set_vocab`](#TextModelset_vocab)
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on hyperparameters.
- **Inputs**: None
- **Control Flow**:
    - The method starts by adding the block count to the GGUF writer.
    - It checks for various hyperparameters using the [`find_hparam`](#ModelBasefind_hparam) method and adds corresponding values to the GGUF writer if they exist.
    - For each parameter found, it logs the value using the logger.
    - The method concludes by adding the file type to the GGUF writer.
- **Output**: The method does not return a value; it modifies the state of the GGUF writer by adding parameters and logging their values.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.write\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel.write_vocab}} -->
The `write_vocab` method writes the vocabulary data to a file after ensuring that the vocabulary is not split.
- **Inputs**: None
- **Control Flow**:
    - Checks if the length of `self.gguf_writer.tensors` is not equal to 1, raising a ValueError if true, indicating that splitting the vocabulary is unsupported.
    - Calls `self.prepare_metadata` with `vocab_only=True` to prepare the metadata for the vocabulary.
    - Writes the header to the output file specified by `self.fname_out` using `self.gguf_writer.write_header_to_file`.
    - Writes the key-value data to the output file using `self.gguf_writer.write_kv_data_to_file`.
    - Closes the `gguf_writer` to finalize the writing process.
- **Output**: The method does not return a value; it performs file operations to write the vocabulary data to the specified output file.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_metadata`](#ModelBaseprepare_metadata)
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.does\_token\_look\_special<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel.does_token_look_special}} -->
Determines if a given token appears to be a special control token.
- **Inputs**:
    - `token`: A string or bytes representation of the token to be checked.
- **Control Flow**:
    - Checks the type of the input `token` to determine how to decode it.
    - If the token is of type `bytes` or `bytearray`, it decodes it to a UTF-8 string.
    - If the token is a `memoryview`, it converts it to bytes and then decodes it.
    - If the token is neither, it is assumed to be a string and assigned directly.
    - Checks if the decoded token matches known special tokens or follows specific patterns indicating it is special.
    - Returns a boolean indicating whether the token is considered special.
- **Output**: Returns a boolean value indicating whether the token is deemed special based on predefined criteria.
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.get\_vocab\_base<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel.get_vocab_base}} -->
The `get_vocab_base` method retrieves the vocabulary and token types from a pre-trained tokenizer, normalizing certain tokens and handling special cases.
- **Inputs**:
    - `self`: An instance of the class that contains the method, which holds model parameters and configurations.
- **Control Flow**:
    - Imports the `AutoTokenizer` from the `transformers` library and initializes it with the model directory.
    - Retrieves the vocabulary size from the model parameters or the tokenizer's vocabulary length.
    - Asserts that the maximum token ID in the tokenizer's vocabulary is less than the specified vocabulary size.
    - Calls the [`get_vocab_base_pre`](#TextModelget_vocab_base_pre) method to get a pre-tokenization identifier.
    - Creates a reverse mapping of token IDs to tokens from the tokenizer's vocabulary.
    - Iterates through the range of vocabulary size to populate the `tokens` and `toktypes` lists based on the presence of token IDs in the reverse vocabulary.
    - Normalizes tokens that are not pre-normalized and categorizes them as CONTROL, USER_DEFINED, or NORMAL based on specific conditions.
- **Output**: Returns a tuple containing a list of tokens, a list of their corresponding token types, and a string identifier for the pre-tokenization method.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel.get_vocab_base_pre`](#TextModelget_vocab_base_pre)
    - [`llama.cpp/convert_hf_to_gguf.TextModel.does_token_look_special`](#TextModeldoes_token_look_special)
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.get\_vocab\_base\_pre<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel.get_vocab_base_pre}} -->
The `get_vocab_base_pre` method generates a unique identifier for the BPE pre-tokenizer used by a model by encoding a specific string and hashing the resulting tokens.
- **Inputs**:
    - `tokenizer`: An instance of a tokenizer that is used to encode a predefined string.
- **Control Flow**:
    - A string `chktxt` is defined, which contains various characters and emojis.
    - The `tokenizer.encode` method is called with `chktxt` to produce a tokenized representation.
    - The SHA-256 hash of the string representation of the tokenized output is computed.
    - A series of conditional checks compare the computed hash against known values to determine the corresponding pre-tokenizer name.
    - If a match is found, the corresponding pre-tokenizer name is assigned to `res`; otherwise, a warning is logged and a NotImplementedError is raised.
- **Output**: The method returns a string representing the name of the BPE pre-tokenizer if recognized; otherwise, it raises a NotImplementedError.
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.\_set\_vocab\_none<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_none}} -->
Sets the tokenizer model to 'none' in the GGUF writer.
- **Inputs**: None
- **Control Flow**:
    - The method directly calls the `add_tokenizer_model` method of the `gguf_writer` instance.
    - It passes the string 'none' as an argument to indicate that no tokenizer model is being set.
- **Output**: The method does not return any value; it modifies the state of the `gguf_writer` by setting the tokenizer model.
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.\_set\_vocab\_gpt2<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2}} -->
Sets the vocabulary for the GPT-2 tokenizer by retrieving tokens and their types, and adding them to the GGUF writer.
- **Inputs**: None
- **Control Flow**:
    - Calls `get_vocab_base()` to retrieve the list of tokens, their types, and the pre-tokenization method.
    - Adds the tokenizer model name 'gpt2' to the GGUF writer.
    - Adds the pre-tokenization method to the GGUF writer.
    - Adds the list of tokens to the GGUF writer.
    - Adds the list of token types to the GGUF writer.
    - Creates a `SpecialVocab` instance and loads special vocabulary merges, then adds it to the GGUF writer.
- **Output**: The method does not return any value; it modifies the state of the `gguf_writer` by adding vocabulary information.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel.get_vocab_base`](#TextModelget_vocab_base)
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.\_set\_vocab\_qwen<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_qwen}} -->
Sets the vocabulary for the Qwen model by processing tokens and their types from a pretrained tokenizer.
- **Inputs**: None
- **Control Flow**:
    - Initializes the tokenizer using the model directory and hyperparameters.
    - Asserts that the maximum vocabulary index from the tokenizer is less than the specified vocabulary size.
    - Retrieves the base vocabulary pre-tokenization method.
    - Iterates through the mergeable ranks of tokens to create a vocabulary and corresponding merges.
    - Combines the added vocabulary with the main vocabulary and assigns token types based on their presence in the added vocabulary.
    - Writes the tokenizer model, pre-tokenization information, token list, and token types to the GGUF writer.
    - Handles special tokens and merges, ensuring they are added correctly to the GGUF writer.
- **Output**: The method does not return a value but updates the GGUF writer with the tokenizer model, token list, token types, and special vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel.get_vocab_base_pre`](#TextModelget_vocab_base_pre)
    - [`llama.cpp/convert_hf_to_gguf.QwenModel.token_bytes_to_string`](#QwenModeltoken_bytes_to_string)
    - [`llama.cpp/convert_hf_to_gguf.QwenModel.bpe`](#QwenModelbpe)
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.\_set\_vocab\_sentencepiece<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece}} -->
Sets the vocabulary for a SentencePiece tokenizer and adds it to the GGUF writer.
- **Inputs**:
    - `add_to_gguf`: A boolean flag indicating whether to add the vocabulary to the GGUF writer.
- **Control Flow**:
    - Calls the [`_create_vocab_sentencepiece`](#TextModel_create_vocab_sentencepiece) method to generate tokens, scores, and token types.
    - Adds the tokenizer model, pre-tokenizer, token list, token scores, and token types to the GGUF writer.
    - Creates a `SpecialVocab` instance with the number of tokens and adds it to the GGUF writer.
- **Output**: None, as the method modifies the state of the GGUF writer directly.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._create_vocab_sentencepiece`](#TextModel_create_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.\_create\_vocab\_sentencepiece<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel._create_vocab_sentencepiece}} -->
Creates vocabulary tokens, scores, and types from a SentencePiece tokenizer model.
- **Inputs**: None
- **Control Flow**:
    - Checks if the tokenizer model file exists and raises a FileNotFoundError if it does not.
    - Loads the tokenizer model and retrieves the vocabulary size.
    - Initializes lists for tokens, scores, and token types with default values.
    - Iterates through the vocabulary to populate tokens, scores, and types based on the tokenizer's properties.
    - Checks for additional tokens from 'added_tokens.json' and updates the lists accordingly.
    - Checks for a tokenizer configuration file and updates tokens and types based on its content.
    - Pads the token lists if the vocabulary size exceeds the number of tokens generated.
- **Output**: Returns a tuple containing the lists of tokens, scores, and token types.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel.does_token_look_special`](#TextModeldoes_token_look_special)
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.\_set\_vocab\_llama\_hf<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_llama_hf}} -->
Sets the vocabulary for the Llama model using Hugging Face's vocabulary structure.
- **Inputs**: None
- **Control Flow**:
    - Creates a vocabulary object from the model directory using `gguf.LlamaHfVocab`.
    - Iterates over all tokens in the vocabulary, appending each token's text, score, and type to respective lists.
    - Asserts that the number of tokens matches the expected vocabulary size.
    - Adds the tokenizer model, pre-tokenizer, token list, token scores, and token types to the `gguf_writer`.
    - Creates a special vocabulary object and adds it to the `gguf_writer`.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` to include the Llama model's vocabulary.
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.\_set\_vocab\_rwkv\_world<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_rwkv_world}} -->
Sets the vocabulary for the RWKV model by reading from a specified vocabulary file and populating token lists.
- **Inputs**: None
- **Control Flow**:
    - Asserts that the vocabulary file exists at the specified path.
    - Retrieves the vocabulary size from hyperparameters, defaulting to 65536 if not specified.
    - Initializes token lists with a start token and its type.
    - Reads the vocabulary file line by line, parsing each line to extract tokens and their lengths, and appends them to the token lists.
    - Calculates the remaining tokens needed to reach the specified vocabulary size and appends padding tokens.
    - Adds the tokenizer model and token lists to the GGUF writer.
    - Sets special tokens for the vocabulary and adds them to the GGUF writer.
- **Output**: The method does not return a value but modifies the internal state of the object by populating the tokenizer model and vocabulary in the GGUF writer.
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.\_set\_vocab\_builtin<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_builtin}} -->
Sets the vocabulary for a specified model by reading from a tokenizer file and populating various tokenizer attributes.
- **Inputs**:
    - `model_name`: A string literal that specifies the model type, either 'gpt-neox' or 'llama-spm'.
    - `vocab_size`: An integer that defines the maximum number of tokens to be included in the vocabulary.
- **Control Flow**:
    - Constructs the path to the tokenizer file based on the model name.
    - Logs a warning indicating the tokenizer being used.
    - Reads the tokenizer file using a GGUFReader instance.
    - Sets a default pre-tokenizer based on the model name.
    - Retrieves and asserts the tokenizer model field, then adds it to the writer.
    - Retrieves and adds the pre-tokenizer field, using the default if not found.
    - Retrieves and asserts the token list field, then adds the specified number of tokens to the writer.
    - If the model is 'llama-spm', retrieves and adds token scores.
    - Retrieves and asserts the token type field, then adds it to the writer.
    - If the model is not 'llama-spm', retrieves and adds token merges.
    - Retrieves and conditionally adds various special token IDs and attributes if they exist.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` with the vocabulary data.
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)


---
#### TextModel\.\_try\_set\_pooling\_type<!-- {{#callable:llama.cpp/convert_hf_to_gguf.TextModel._try_set_pooling_type}} -->
Attempts to set the pooling type for a model based on configuration files.
- **Inputs**: None
- **Control Flow**:
    - Checks if the 'modules.json' file exists in the model directory.
    - If the file exists, it loads the JSON content and searches for a module of type 'sentence_transformers.models.Pooling'.
    - If found, it retrieves the path to the pooling configuration file.
    - If the pooling path is valid, it loads the 'config.json' file to determine the pooling type.
    - Based on the configuration, it sets the pooling type to MEAN, CLS, or LAST, or raises an error if none are applicable.
- **Output**: The method does not return a value but updates the pooling type in the 'gguf_writer' based on the configuration.
- **See also**: [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)  (Base Class)



---
### MmprojModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.MmprojModel}} -->
- **Members**:
    - `model_type`: Specifies the type of model as MMPROJ.
    - `model_arch`: Defines the architecture of the model as MMPROJ.
    - `preprocessor_config`: Holds configuration settings for preprocessing.
    - `global_config`: Stores global configuration settings.
    - `n_block_keys`: A list of keys related to the number of layers in the model.
    - `has_vision_encoder`: Indicates if the model includes a vision encoder, defaulting to True.
    - `has_audio_encoder`: Indicates if the model includes an audio encoder, defaulting to False.
    - `hparams_vision`: Holds hyperparameters specific to the vision encoder.
    - `hparams_audio`: Holds hyperparameters specific to the audio encoder.
- **Description**: The `MmprojModel` class extends `ModelBase` and is designed to handle models with vision and audio encoders, managing their configurations and hyperparameters while ensuring compatibility with the MMPROJ architecture.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel.__init__`](#MmprojModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel.get_vision_config`](#MmprojModelget_vision_config)
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel.get_audio_config`](#MmprojModelget_audio_config)
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel.set_type`](#MmprojModelset_type)
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel.set_gguf_parameters`](#MmprojModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel.write_vocab`](#MmprojModelwrite_vocab)
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel.find_vparam`](#MmprojModelfind_vparam)
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel.find_aparam`](#MmprojModelfind_aparam)
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel._find_param`](#MmprojModel_find_param)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase`](#cpp/convert_hf_to_ggufModelBase)

**Methods**

---
#### MmprojModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MmprojModel.__init__}} -->
Initializes the `MmprojModel` class, validating architecture and configuring parameters for vision and audio encoders.
- **Inputs**:
    - `args`: Positional arguments passed to the parent class constructor.
    - `kwargs`: Keyword arguments passed to the parent class constructor.
- **Control Flow**:
    - Calls the parent class constructor using `super().__init__(*args, **kwargs)`.
    - Checks if the `model_arch` is set to `gguf.MODEL_ARCH.MMPROJ`, raising a `TypeError` if not.
    - Initializes `text_config` and retrieves the embedding size from it, asserting it is greater than zero.
    - Creates a deep copy of `hparams` into `global_config` and retrieves vision and audio configurations.
    - Raises a `ValueError` if both vision and audio configurations are missing.
    - Sets `hparams` to the available configuration (vision or audio).
    - Determines the `block_count` based on the presence of multiple encoders.
    - Loads the preprocessor configuration from a JSON file.
- **Output**: No explicit output; initializes the instance with configured parameters and validates the model architecture.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel.get_vision_config`](#MmprojModelget_vision_config)
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel.get_audio_config`](#MmprojModelget_audio_config)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)  (Base Class)


---
#### MmprojModel\.get\_vision\_config<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MmprojModel.get_vision_config}} -->
Retrieves the vision configuration from the global configuration dictionary.
- **Inputs**: None
- **Control Flow**:
    - Accesses the `global_config` attribute of the instance to retrieve the value associated with the key 'vision_config'.
    - Returns the retrieved value, which can be a dictionary or None if the key does not exist.
- **Output**: Returns a dictionary containing the vision configuration if it exists, otherwise returns None.
- **See also**: [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)  (Base Class)


---
#### MmprojModel\.get\_audio\_config<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MmprojModel.get_audio_config}} -->
The `get_audio_config` method retrieves the audio configuration from the global configuration dictionary.
- **Inputs**: None
- **Control Flow**:
    - The method accesses the `global_config` attribute of the instance, which is expected to be a dictionary.
    - It attempts to retrieve the value associated with the key 'audio_config' from the `global_config` dictionary.
- **Output**: The method returns the value associated with the 'audio_config' key, which can be a dictionary or None if the key does not exist.
- **See also**: [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)  (Base Class)


---
#### MmprojModel\.set\_type<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MmprojModel.set_type}} -->
Sets the type of the model in the GGUF writer to MMPROJ.
- **Decorators**: `@some_decorator_if_any`
- **Inputs**: None
- **Control Flow**:
    - Calls the `add_type` method on the `gguf_writer` attribute, passing in the `GGUFType.MMPROJ` constant.
- **Output**: No output is returned; the method modifies the state of the `gguf_writer` by adding a type.
- **See also**: [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)  (Base Class)


---
#### MmprojModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MmprojModel.set_gguf_parameters}} -->
Sets parameters for the GGUF writer based on the presence of vision and audio encoders.
- **Inputs**: None
- **Control Flow**:
    - The method begins by adding the file type to the GGUF writer.
    - If a vision encoder is present, it adds various vision-related parameters such as image size, patch size, and embedding lengths.
    - If an audio encoder is present, it adds audio-related parameters including embedding lengths and block counts.
    - If neither encoder is present, a ValueError is raised indicating that at least one encoder must be available.
- **Output**: The method does not return a value but configures the GGUF writer with the specified parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel.find_vparam`](#MmprojModelfind_vparam)
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel.find_aparam`](#MmprojModelfind_aparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)  (Base Class)


---
#### MmprojModel\.write\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MmprojModel.write_vocab}} -->
The `write_vocab` method raises a ValueError indicating that vocabulary writing is not supported in the `MmprojModel` class.
- **Inputs**: None
- **Control Flow**:
    - The method immediately raises a `ValueError` exception when called, indicating that vocabulary writing is not supported.
- **Output**: The method does not return a value; instead, it raises a `ValueError` with a specific message.
- **See also**: [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)  (Base Class)


---
#### MmprojModel\.find\_vparam<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MmprojModel.find_vparam}} -->
The `find_vparam` method retrieves a parameter from the vision hyperparameters based on provided keys.
- **Inputs**:
    - `keys`: An iterable of strings representing the keys to search for in the vision hyperparameters.
    - `optional`: A boolean flag indicating whether the search for the key is optional.
- **Control Flow**:
    - The method asserts that `self.hparams_vision` is not None to ensure that vision hyperparameters are available.
    - It calls the [`_find_param`](#MmprojModel_find_param) method with `self.hparams_vision`, the provided `keys`, and the `optional` flag to retrieve the desired parameter.
- **Output**: Returns the value associated with the first matching key found in the vision hyperparameters, or None if the key is not found and `optional` is True.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel._find_param`](#MmprojModel_find_param)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)  (Base Class)


---
#### MmprojModel\.find\_aparam<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MmprojModel.find_aparam}} -->
The `find_aparam` method retrieves a parameter from the audio hyperparameters dictionary based on provided keys.
- **Inputs**:
    - `keys`: An iterable of strings representing the keys to search for in the audio hyperparameters.
    - `optional`: A boolean flag indicating whether the search for the key is optional.
- **Control Flow**:
    - The method asserts that `self.hparams_audio` is not None to ensure that audio hyperparameters are available.
    - It calls the [`_find_param`](#MmprojModel_find_param) method with `self.hparams_audio`, the provided `keys`, and the `optional` flag to retrieve the desired parameter.
- **Output**: Returns the value associated with the first matching key found in the audio hyperparameters, or None if `optional` is True and no key is found.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel._find_param`](#MmprojModel_find_param)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)  (Base Class)



---
### GPTNeoXModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.GPTNeoXModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as GPTNeoX.
- **Description**: The `GPTNeoXModel` class extends `TextModel` and is designed for causal language modeling, incorporating methods to set parameters for the GGUF format and modify tensor data for model layers.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.GPTNeoXModel.set_gguf_parameters`](#GPTNeoXModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.GPTNeoXModel.modify_tensors`](#GPTNeoXModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### GPTNeoXModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.GPTNeoXModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on model hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `GPTNeoXModel` class, which contains model hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves the number of hidden layers from the model's hyperparameters.
    - Calls multiple methods on the `gguf_writer` to set parameters such as context length, embedding length, block count, feed-forward length, rotary dimension count, head count, parallel residual usage, and layer normalization epsilon.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.GPTNeoXModel`](#cpp/convert_hf_to_ggufGPTNeoXModel)  (Base Class)


---
#### GPTNeoXModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.GPTNeoXModel.modify_tensors}} -->
The `modify_tensors` method reformats tensor weights and biases for attention layers in a neural network model.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` containing the weights or biases to be modified.
    - `name`: A string representing the name of the tensor, which determines how it will be processed.
    - `bid`: An optional integer that is unused in the current implementation.
- **Control Flow**:
    - The method retrieves the number of attention heads (`n_head`) and the embedding size (`n_embed`) from the model's hyperparameters.
    - It checks if the `name` matches a specific pattern for weights or biases of the attention layer.
    - If the name matches the weight pattern, it reshapes and concatenates the tensor accordingly, logging the operation.
    - If the name matches the bias pattern, it reshapes and concatenates the bias tensor, also logging the operation.
    - Finally, it appends the modified tensor along with its mapped name to the `tensors` list and returns this list.
- **Output**: An iterable of tuples, each containing a string (the mapped tensor name) and the modified `Tensor`.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.GPTNeoXModel`](#cpp/convert_hf_to_ggufGPTNeoXModel)  (Base Class)



---
### BloomModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.BloomModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as BLOOM.
- **Description**: The `BloomModel` class extends `TextModel` and is designed for handling the BLOOM architecture in causal language modeling, providing methods to set parameters and modify tensor data for model training and inference.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.BloomModel.set_gguf_parameters`](#BloomModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.BloomModel.modify_tensors`](#BloomModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### BloomModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BloomModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on model hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `BloomModel` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves the embedding size (`n_embed`) from hyperparameters, defaulting to `n_embed` if `hidden_size` is not set.
    - Retrieves the number of attention heads (`n_head`) from hyperparameters, defaulting to `num_attention_heads` if `n_head` is not set.
    - Calls methods on `gguf_writer` to set various parameters such as context length, embedding length, feed-forward length, block count, head count, and layer normalization epsilon using the retrieved hyperparameters.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.BloomModel`](#cpp/convert_hf_to_ggufBloomModel)  (Base Class)


---
#### BloomModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BloomModel.modify_tensors}} -->
The `modify_tensors` method reformats tensor data for attention mechanisms in a neural network model.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` containing the weights or biases to be modified.
    - `name`: A string representing the name of the tensor, which determines how it will be processed.
    - `bid`: An optional integer that is unused in this method.
- **Control Flow**:
    - The method retrieves the number of attention heads (`n_head`) and the embedding size (`n_embed`) from the model's hyperparameters.
    - It removes the 'transformer.' prefix from the tensor name for easier matching.
    - If the tensor name matches the pattern for weights, it reshapes and concatenates the tensor accordingly, logging the operation.
    - If the tensor name matches the pattern for biases, it reshapes and concatenates the tensor accordingly, logging the operation.
    - The modified tensor is then added to a list along with its mapped name.
- **Output**: Returns an iterable of tuples, each containing a string (the mapped tensor name) and the modified `Tensor`.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BloomModel`](#cpp/convert_hf_to_ggufBloomModel)  (Base Class)



---
### MPTModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.MPTModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `MPTModel` class extends `TextModel` and is designed for causal language modeling, providing methods to set vocabulary and configure model parameters, while also allowing for tensor modifications during processing.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.MPTModel.set_vocab`](#MPTModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.MPTModel.set_gguf_parameters`](#MPTModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.MPTModel.modify_tensors`](#MPTModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### MPTModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MPTModel.set_vocab}} -->
The `set_vocab` method initializes the vocabulary for the model, attempting to set it using the GPT-2 method and falling back to a SentencePiece method if an exception occurs.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `MPTModel` class, which contains methods and attributes necessary for setting the vocabulary.
- **Control Flow**:
    - The method first attempts to call `self._set_vocab_gpt2()` to set the vocabulary using the GPT-2 method.
    - If an exception occurs during the GPT-2 vocabulary setting, it falls back to `self._set_vocab_sentencepiece()`.
    - After setting the vocabulary with the fallback method, it configures additional tokens for the model by calling methods on `self.gguf_writer` to add special tokens such as the beginning-of-sequence (BOS), padding, end-of-sequence (EOS), and unknown (UNK) tokens.
- **Output**: The method does not return a value; instead, it modifies the internal state of the model by setting the vocabulary and configuring special tokens.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MPTModel`](#cpp/convert_hf_to_ggufMPTModel)  (Base Class)


---
#### MPTModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MPTModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on model hyperparameters.
- **Inputs**: None
- **Control Flow**:
    - Retrieves the number of layers from `self.hparams` and assigns it to `block_count`.
    - Adds context length, embedding length, block count, feed-forward length, and head count to the GGUF writer using values from `self.hparams`.
    - Checks if `kv_n_heads` exists in the attention configuration and adds it to the GGUF writer if present.
    - Sets a fixed layer normalization epsilon value.
    - Checks if `clip_qkv` is specified in the attention configuration and adds it to the GGUF writer if present.
    - Checks if `alibi` is enabled in the attention configuration and adds the corresponding maximum alibi bias to the GGUF writer, defaulting to 0.0 if not enabled.
- **Output**: No explicit return value; modifies the state of the GGUF writer with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.MPTModel`](#cpp/convert_hf_to_ggufMPTModel)  (Base Class)


---
#### MPTModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MPTModel.modify_tensors}} -->
The `modify_tensors` method modifies the name of a tensor based on its original name and returns a tuple containing the new name and the tensor.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the original name of the tensor.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method first deletes the `bid` parameter as it is unused.
    - It checks if the string 'scales' is present in the `name` argument.
    - If 'scales' is found, it maps the tensor name with specific suffixes and replaces 'scales' with 'act.scales'.
    - If 'scales' is not found, it simply maps the tensor name with other suffixes.
    - Finally, it returns a list containing a tuple of the new name and the original tensor.
- **Output**: The output is an iterable containing a single tuple, where the first element is the modified tensor name and the second element is the original tensor.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MPTModel`](#cpp/convert_hf_to_ggufMPTModel)  (Base Class)



---
### OrionModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.OrionModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as Orion.
- **Description**: The `OrionModel` class extends `TextModel` and is designed for causal language modeling, incorporating specific parameters and configurations for the Orion architecture.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.OrionModel.set_vocab`](#OrionModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.OrionModel.set_gguf_parameters`](#OrionModelset_gguf_parameters)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### OrionModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OrionModel.set_vocab}} -->
Sets the vocabulary for the `OrionModel` by invoking a specific method to handle the vocabulary setup.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `OrionModel` class.
- **Control Flow**:
    - Calls the [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece) method to perform the actual vocabulary setting.
- **Output**: No output is returned; the method modifies the internal state of the `OrionModel` instance.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.OrionModel`](#cpp/convert_hf_to_ggufOrionModel)  (Base Class)


---
#### OrionModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OrionModel.set_gguf_parameters}} -->
Sets parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `OrionModel` class containing hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves the number of hidden layers and attention heads from `self.hparams`.
    - Checks for the context length parameter in `self.hparams`, prioritizing `max_sequence_length`, `max_position_embeddings`, and `model_max_length`, raising an error if none are found.
    - Calls various methods on `self.gguf_writer` to set the GGUF model parameters using the retrieved hyperparameters.
- **Output**: No explicit output; the method configures the GGUF writer with model parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.OrionModel`](#cpp/convert_hf_to_ggufOrionModel)  (Base Class)



---
### BaichuanModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.BaichuanModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `BaichuanModel` class extends `TextModel` and is designed for causal language modeling, incorporating methods for setting vocabulary and configuring model parameters, as well as modifying tensor data for attention mechanisms.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.BaichuanModel.set_vocab`](#BaichuanModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.BaichuanModel.set_gguf_parameters`](#BaichuanModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.BaichuanModel.modify_tensors`](#BaichuanModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.BaichuanModel._reverse_hf_permute`](#BaichuanModel_reverse_hf_permute)
    - [`llama.cpp/convert_hf_to_gguf.BaichuanModel._reverse_hf_permute_part`](#BaichuanModel_reverse_hf_permute_part)
    - [`llama.cpp/convert_hf_to_gguf.BaichuanModel._reverse_hf_part`](#BaichuanModel_reverse_hf_part)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### BaichuanModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BaichuanModel.set_vocab}} -->
Sets the vocabulary for the model using a sentencepiece tokenizer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `BaichuanModel` class.
- **Control Flow**:
    - Calls the [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece) method to initialize the vocabulary.
- **Output**: No output is returned; the method modifies the internal state of the model by setting the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BaichuanModel`](#cpp/convert_hf_to_ggufBaichuanModel)  (Base Class)


---
#### BaichuanModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BaichuanModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `BaichuanModel` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves the number of hidden layers and attention heads from the hyperparameters.
    - Determines the context length based on available hyperparameters, raising an error if none are found.
    - Adds various model parameters to the GGUF writer, including tensor data layout, context length, embedding length, block count, feed-forward length, ROPE dimension count, head counts, layer normalization epsilon, and file type.
    - Checks for ROPE scaling parameters and adds them to the GGUF writer if applicable.
- **Output**: The method does not return a value; it configures the GGUF writer with model parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.BaichuanModel`](#cpp/convert_hf_to_ggufBaichuanModel)  (Base Class)


---
#### BaichuanModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BaichuanModel.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on specific conditions related to attention layers.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name of the tensor, which is used to determine how to process the tensor.
    - `bid`: An optional integer that indicates the specific layer index for attention mechanisms.
- **Control Flow**:
    - The method retrieves the number of attention heads and key-value heads from the model's hyperparameters.
    - It checks if `bid` is not None and if the `name` matches a specific pattern related to attention weights.
    - If the conditions are met, it logs the unpacking and permuting action and creates a list of modified tensors using helper methods.
    - If the conditions are not met, it simply maps the tensor name and returns the original tensor.
- **Output**: The method returns an iterable of tuples, each containing a string (the formatted tensor name) and a `Tensor` (the modified tensor data).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.BaichuanModel._reverse_hf_permute_part`](#BaichuanModel_reverse_hf_permute_part)
    - [`llama.cpp/convert_hf_to_gguf.BaichuanModel._reverse_hf_part`](#BaichuanModel_reverse_hf_part)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BaichuanModel`](#cpp/convert_hf_to_ggufBaichuanModel)  (Base Class)


---
#### BaichuanModel\.\_reverse\_hf\_permute\_part<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BaichuanModel._reverse_hf_permute_part}} -->
This method extracts a specific part of the weights tensor and applies a permutation operation to it.
- **Inputs**:
    - `weights`: A `Tensor` containing the weights from which a part will be extracted.
    - `n_part`: An integer indicating which part of the weights to extract.
    - `n_head`: An integer representing the number of attention heads.
    - `n_head_kv`: An optional integer representing the number of key-value heads.
- **Control Flow**:
    - Calculates `r`, which is one third of the first dimension of `weights`.
    - Extracts a slice of `weights` corresponding to the specified `n_part` and passes it to the [`_reverse_hf_permute`](#BaichuanModel_reverse_hf_permute) method along with `n_head` and `n_head_kv`.
- **Output**: Returns a `Tensor` that has been permuted based on the extracted part of the weights.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.BaichuanModel._reverse_hf_permute`](#BaichuanModel_reverse_hf_permute)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BaichuanModel`](#cpp/convert_hf_to_ggufBaichuanModel)  (Base Class)



---
### XverseModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.XverseModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `XverseModel` class extends `TextModel` and is designed for handling specific functionalities related to the Xverse architecture, including vocabulary management and tensor modifications for model parameters.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.XverseModel.set_vocab`](#XverseModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.XverseModel.set_gguf_parameters`](#XverseModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.XverseModel.modify_tensors`](#XverseModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.XverseModel._reverse_hf_permute`](#XverseModel_reverse_hf_permute)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### XverseModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.XverseModel.set_vocab}} -->
Sets the vocabulary for the model by loading a tokenizer and processing its tokens.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class containing the method, which holds model parameters and configurations.
- **Control Flow**:
    - Asserts that the tokenizer configuration file exists in the specified model directory.
    - Loads the tokenizer from the specified directory using the `AutoTokenizer` class.
    - Retrieves the vocabulary size from the model parameters or the tokenizer's vocabulary.
    - Checks if the maximum vocabulary index exceeds the expected vocabulary size, raising an error if it does.
    - Creates a reverse mapping of token IDs to token strings from the tokenizer's vocabulary.
    - Iterates through the range of vocabulary size to process each token, determining its type based on specific conditions.
    - Appends the processed token text and its type to respective lists.
    - Adds the tokenizer model and pre-tokenizer information to the `gguf_writer`.
    - Adds the list of tokens and their types to the `gguf_writer`.
    - Creates a special vocabulary object and adds it to the `gguf_writer`.
- **Output**: The method does not return a value but updates the internal state of the model by writing the processed vocabulary and its types to the `gguf_writer`.
- **See also**: [`llama.cpp/convert_hf_to_gguf.XverseModel`](#cpp/convert_hf_to_ggufXverseModel)  (Base Class)


---
#### XverseModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.XverseModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves the number of hidden layers and attention heads from the hyperparameters.
    - Determines the context length based on available hyperparameters, raising an error if none are found.
    - Adds various model parameters to the GGUF writer, including tensor data layout, context length, embedding length, block count, and others.
    - Checks for rope scaling parameters and adds them to the GGUF writer if applicable.
- **Output**: The method does not return a value but updates the GGUF writer with the specified model parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.XverseModel`](#cpp/convert_hf_to_ggufXverseModel)  (Base Class)


---
#### XverseModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.XverseModel.modify_tensors}} -->
The `modify_tensors` method adjusts tensor weights based on their names to reverse any permutations applied by Hugging Face models.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the weights to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be modified.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method retrieves the number of attention heads from the model's hyperparameters.
    - It checks if the tensor name ends with 'q_proj.weight' or 'k_proj.weight' to determine if a permutation reversal is needed.
    - If a reversal is needed, it calls the [`_reverse_hf_permute`](#ChameleonModel_reverse_hf_permute) method with the appropriate parameters.
    - Finally, it returns a list containing a tuple of the modified tensor name and the modified tensor.
- **Output**: The method returns an iterable of tuples, each containing a string (the mapped tensor name) and a `Tensor` (the modified tensor).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ChameleonModel._reverse_hf_permute`](#ChameleonModel_reverse_hf_permute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.XverseModel`](#cpp/convert_hf_to_ggufXverseModel)  (Base Class)


---
#### XverseModel\.\_reverse\_hf\_permute<!-- {{#callable:llama.cpp/convert_hf_to_gguf.XverseModel._reverse_hf_permute}} -->
Reverses the permutation of weights for a multi-head attention mechanism based on the number of heads and key-value heads.
- **Inputs**:
    - `weights`: A `Tensor` representing the weights to be permuted.
    - `n_head`: An integer representing the number of attention heads.
    - `n_kv_head`: An optional integer representing the number of key-value heads; if not provided, defaults to None.
- **Control Flow**:
    - Checks if `n_kv_head` is provided and if it differs from `n_head`, adjusting `n_head` accordingly.
    - Reshapes the `weights` tensor to a new shape that accommodates the multi-head structure, swaps axes to reorder dimensions, and reshapes it back to the original shape.
- **Output**: Returns a `Tensor` that has been reshaped and permuted to reverse the original permutation applied to the weights.
- **See also**: [`llama.cpp/convert_hf_to_gguf.XverseModel`](#cpp/convert_hf_to_ggufXverseModel)  (Base Class)



---
### FalconModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.FalconModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as FALCON.
- **Description**: The `FalconModel` class extends `TextModel` and is designed for causal language modeling, incorporating methods to set parameters for the GGUF format and modify tensor structures for efficient processing.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.FalconModel.set_gguf_parameters`](#FalconModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.FalconModel.modify_tensors`](#FalconModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### FalconModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.FalconModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class `FalconModel`, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves the number of hidden layers from `self.hparams`, using a fallback to an older parameter name if necessary.
    - Retrieves the number of attention heads from `self.hparams`, using a fallback to an older parameter name if necessary.
    - Retrieves the number of key-value heads from `self.hparams`, using a fallback to an older parameter name with a default value of 1 if necessary.
    - Calls methods on `self.gguf_writer` to set various model parameters such as context length, tensor data layout, embedding length, feed-forward length, block count, head count, key-value head count, layer normalization epsilon, and file type.
- **Output**: The method does not return a value; it configures the GGUF writer with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.FalconModel`](#cpp/convert_hf_to_ggufFalconModel)  (Base Class)


---
#### FalconModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.FalconModel.modify_tensors}} -->
Transforms a query-key-value tensor layout for compatibility with GGML.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A PyTorch tensor representing the query-key-value data to be modified.
    - `name`: A string representing the name of the tensor, which determines the transformation applied.
    - `bid`: An optional integer that is unused in this method.
- **Control Flow**:
    - The method first checks if 'query_key_value' is part of the tensor name to determine if transformation is needed.
    - It retrieves the number of attention heads and key-value heads from the model's hyperparameters.
    - The method reshapes the input tensor into a specific format to separate query, key, and value weights.
    - Finally, it concatenates the transformed tensors and returns them as a list of tuples with the mapped name.
- **Output**: Returns an iterable of tuples, each containing the mapped tensor name and the modified tensor.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.FalconModel`](#cpp/convert_hf_to_ggufFalconModel)  (Base Class)



---
### StarCoderModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.StarCoderModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as `STARCODER`.
- **Description**: The `StarCoderModel` class extends `TextModel` and is designed to represent a specific architecture for a causal language model, incorporating various parameters for model configuration.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.StarCoderModel.set_gguf_parameters`](#StarCoderModelset_gguf_parameters)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### StarCoderModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.StarCoderModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on model hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `StarCoderModel` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves the number of layers from the model's hyperparameters.
    - Calls methods on the `gguf_writer` to set various parameters such as context length, embedding length, feed-forward length, block count, head count, key-value head count, layer normalization epsilon, and file type.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.StarCoderModel`](#cpp/convert_hf_to_ggufStarCoderModel)  (Base Class)



---
### RefactModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.RefactModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `RefactModel` class extends `TextModel` and is designed for a specific architecture in the context of causal language modeling, providing methods to set vocabulary and model parameters, as well as to modify tensors based on the model's configuration.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.RefactModel.set_vocab`](#RefactModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.RefactModel.set_gguf_parameters`](#RefactModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.RefactModel.modify_tensors`](#RefactModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### RefactModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.RefactModel.set_vocab}} -->
The [`set_vocab`](#TextModelset_vocab) method initializes a special vocabulary for the model by setting specific token types and adding them to the GGUF writer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains the method, which is expected to have attributes like `dir_model` and `gguf_writer`.
- **Control Flow**:
    - Calls the parent class's [`set_vocab`](#TextModelset_vocab) method to ensure any base vocabulary setup is completed.
    - Creates an instance of `gguf.SpecialVocab` with specified parameters including the model directory and token types.
    - Sets special tokens for 'prefix', 'suffix', and 'middle' using the `_set_special_token` method.
    - Ensures that the `chat_template` attribute is set to None to avoid duplication.
    - Adds the special vocabulary to the GGUF writer using the `add_to_gguf` method.
- **Output**: The method does not return a value but modifies the state of the `special_vocab` and updates the `gguf_writer` with the special tokens.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel.set_vocab`](#TextModelset_vocab)
- **See also**: [`llama.cpp/convert_hf_to_gguf.RefactModel`](#cpp/convert_hf_to_ggufRefactModel)  (Base Class)


---
#### RefactModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.RefactModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `RefactModel` class containing hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calculates the `hidden_dim`, `inner_dim`, and `ff_dim` based on hyperparameters.
    - Retrieves the `block_count` from the hyperparameters.
    - Uses the `gguf_writer` to add various model parameters such as context length, embedding length, feed-forward length, block count, head count, and layer normalization epsilon.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.RefactModel`](#cpp/convert_hf_to_ggufRefactModel)  (Base Class)


---
#### RefactModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.RefactModel.modify_tensors}} -->
The `modify_tensors` method processes input tensors based on their names and an optional bid, returning a list of formatted tensor names and their corresponding data.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` containing the data to be modified.
    - `name`: A string representing the name of the tensor to be modified.
    - `bid`: An optional integer that specifies the block ID associated with the tensor.
- **Control Flow**:
    - Calculates dimensions based on hyperparameters such as `n_embd`, `n_head`, and `n_layer`.
    - Checks if `bid` is not None and matches the `name` against specific patterns to determine how to split `data_torch`.
    - If a match is found, appends formatted tensor names and corresponding slices of `data_torch` to the `tensors` list.
    - If no matches are found, maps the tensor name using [`map_tensor_name`](#ModelBasemap_tensor_name) and appends it with the original `data_torch`.
- **Output**: Returns an iterable of tuples, each containing a formatted tensor name and its corresponding `Tensor` data.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.RefactModel`](#cpp/convert_hf_to_ggufRefactModel)  (Base Class)



---
### StableLMModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.StableLMModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
    - `_q_norms`: Holds the query layer normalization tensors.
    - `_k_norms`: Holds the key layer normalization tensors.
- **Description**: The `StableLMModel` class extends `TextModel` and is designed for handling the StableLM architecture, providing methods for setting vocabulary, configuring model parameters, and managing tensor modifications related to attention mechanisms.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.StableLMModel.set_vocab`](#StableLMModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.StableLMModel.set_gguf_parameters`](#StableLMModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.StableLMModel.modify_tensors`](#StableLMModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.StableLMModel._stack_qk_norm`](#StableLMModel_stack_qk_norm)
    - [`llama.cpp/convert_hf_to_gguf.StableLMModel.prepare_tensors`](#StableLMModelprepare_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### StableLMModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.StableLMModel.set_vocab}} -->
Sets the vocabulary for the model based on the presence of a tokenizer configuration file.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `StableLMModel` class.
- **Control Flow**:
    - Checks if the file 'tokenizer.json' exists in the model directory.
    - If the file exists, calls the [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) method to set the vocabulary.
    - If the file does not exist, calls the [`_set_vocab_qwen`](#TextModel_set_vocab_qwen) method to set the vocabulary in a different format.
- **Output**: No explicit output; the method modifies the internal state of the model by setting the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_qwen`](#TextModel_set_vocab_qwen)
- **See also**: [`llama.cpp/convert_hf_to_gguf.StableLMModel`](#cpp/convert_hf_to_ggufStableLMModel)  (Base Class)


---
#### StableLMModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.StableLMModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class containing hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves hyperparameters from the instance's `hparams` attribute.
    - Adds context length, embedding length, block count, and feed-forward length to the GGUF writer using values from `hparams`.
    - Calculates the rope dimension count based on the rotary factor and adds it to the GGUF writer.
    - Adds head count and key-value head count to the GGUF writer.
    - Checks for the presence of the `use_parallel_residual` key in `hparams` and adds its value to the GGUF writer, defaulting to True if not present.
    - Adds layer normalization epsilon and file type to the GGUF writer.
- **Output**: No explicit output; modifies the state of the GGUF writer with the provided parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.StableLMModel`](#cpp/convert_hf_to_ggufStableLMModel)  (Base Class)


---
#### StableLMModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.StableLMModel.modify_tensors}} -->
The `modify_tensors` method processes tensor data based on the specified layer normalization type and returns stacked tensors when a sufficient number of tensors have been collected.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the tensor data to be modified.
    - `name`: A string representing the name of the tensor, which indicates its type (e.g., 'q_layernorm.norms' or 'k_layernorm.norms').
    - `bid`: An optional integer representing the block ID, which is required for certain tensor types.
- **Control Flow**:
    - The method first retrieves the number of attention heads and key-value heads from the model's hyperparameters.
    - It checks if the `name` indicates a query layer normalization; if so, it asserts that `bid` is not None and initializes `_q_norms` if it is None.
    - The tensor is stored in `_q_norms` under the specified `bid` and `name`, and if the number of stored tensors reaches `n_head`, it calls [`_stack_qk_norm`](#StableLMModel_stack_qk_norm) to stack them.
    - If the `name` indicates a key layer normalization, a similar process is followed for `_k_norms` and `n_kv_head`.
    - If the `name` does not match either layer normalization type, it returns a tuple containing the mapped name and the original tensor.
- **Output**: Returns an iterable of tuples, where each tuple contains a mapped tensor name and the corresponding tensor, or an empty list if not enough tensors have been collected for stacking.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.StableLMModel._stack_qk_norm`](#StableLMModel_stack_qk_norm)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.StableLMModel`](#cpp/convert_hf_to_ggufStableLMModel)  (Base Class)


---
#### StableLMModel\.\_stack\_qk\_norm<!-- {{#callable:llama.cpp/convert_hf_to_gguf.StableLMModel._stack_qk_norm}} -->
The `_stack_qk_norm` method consolidates and stacks tensor norms for a specified layer in a model architecture.
- **Inputs**:
    - `bid`: An integer representing the block ID of the model layer.
    - `n_head`: An integer indicating the number of attention heads.
    - `norms`: A dictionary mapping string keys to tensor values, containing the norms to be stacked.
    - `layer_name`: A string specifying the name of the layer, defaulting to 'q_layernorm'.
- **Control Flow**:
    - Initializes an empty list `datas` to hold the tensor norms.
    - Iterates over the range of `n_head` to extract norms from the `norms` dictionary based on constructed keys.
    - Appends each extracted tensor to the `datas` list and removes the entry from `norms`.
    - Stacks the tensors in `datas` along a new dimension using `torch.stack`.
    - Constructs a new name for the stacked tensor and maps it using `self.map_tensor_name`.
    - Returns a list containing the new name and the stacked tensor.
- **Output**: Returns a list containing a tuple with the new tensor name and the stacked tensor of norms.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.StableLMModel`](#cpp/convert_hf_to_ggufStableLMModel)  (Base Class)


---
#### StableLMModel\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.StableLMModel.prepare_tensors}} -->
The [`prepare_tensors`](#ModelBaseprepare_tensors) method prepares tensor data by invoking the parent class's method and checks for unprocessed norms in the model.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls the [`prepare_tensors`](#ModelBaseprepare_tensors) method from the parent class to perform initial tensor preparation.
    - Checks if either `_q_norms` or `_k_norms` is not None, indicating that there are norms to process.
    - If norms are present, it flattens the lists of dictionaries into a single list of keys.
    - If the flattened list of norms is not empty, it raises a ValueError indicating unprocessed norms.
- **Output**: The method does not return a value but raises a ValueError if there are unprocessed norms.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.StableLMModel`](#cpp/convert_hf_to_ggufStableLMModel)  (Base Class)



---
### LlamaModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.LlamaModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
    - `undo_permute`: Indicates whether to undo the permutation of weights.
    - `_experts`: Holds a list of expert tensors or None.
- **Description**: The `LlamaModel` class extends `TextModel` and is designed for causal language modeling, incorporating various configurations and parameters specific to the LLaMA architecture, including handling vocabulary and tensor modifications for model training and inference.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.__init__`](#LlamaModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.set_vocab`](#LlamaModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.set_gguf_parameters`](#LlamaModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.modify_tensors`](#LlamaModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.generate_extra_tensors`](#LlamaModelgenerate_extra_tensors)
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.prepare_tensors`](#LlamaModelprepare_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### LlamaModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.LlamaModel.__init__}} -->
Initializes the `LlamaModel` class, ensuring proper configuration for the model architecture and setting default parameters if necessary.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `args`: Variable length argument list for additional parameters to be passed to the parent class initializer.
    - `kwargs`: Keyword arguments for additional configuration options to be passed to the parent class initializer.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method to ensure proper initialization.
    - Checks if the model architecture is `VLlama3ForCausalLM` and sets the `num_attention_heads` parameter to a default value of 32 if it is not already specified.
- **Output**: No explicit output; the method initializes the instance of the class and configures its parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
- **See also**: [`llama.cpp/convert_hf_to_gguf.LlamaModel`](#cpp/convert_hf_to_ggufLlamaModel)  (Base Class)


---
#### LlamaModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.LlamaModel.set_vocab}} -->
The `set_vocab` method initializes the vocabulary for the model by attempting to load it from various sources and configuring special tokens based on the model's parameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class `LlamaModel`, which contains model parameters and methods for vocabulary management.
- **Control Flow**:
    - The method first attempts to set the vocabulary using [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece), and if it fails due to a `FileNotFoundError`, it tries [`_set_vocab_llama_hf`](#TextModel_set_vocab_llama_hf).
    - If both attempts fail, it defaults to using [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2).
    - It checks if the vocabulary size is 32016 to apply special vocabulary settings for CodeLlama, including setting special tokens and adding them to the GGUF writer.
    - The method then checks for the existence of a `tokenizer_config.json` file to potentially add a prefix space to the tokenizer.
    - Finally, if the vocabulary size is 49152, it modifies the GGUF writer to not add a beginning-of-sequence token.
- **Output**: The method does not return a value but modifies the internal state of the model by setting the vocabulary and configuring special tokens as needed.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_llama_hf`](#TextModel_set_vocab_llama_hf)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.LlamaModel`](#cpp/convert_hf_to_ggufLlamaModel)  (Base Class)


---
#### LlamaModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.LlamaModel.set_gguf_parameters}} -->
Sets the parameters for the GGUF model by configuring vocabulary size, rope dimensions, and scaling factors.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base configurations are applied.
    - Retrieves hyperparameters from `self.hparams` to configure vocabulary size and rope dimensions.
    - Checks if 'head_dim' is present in hyperparameters to determine the rope dimension; otherwise, calculates it based on hidden size and number of attention heads.
    - Retrieves the rope scaling configuration and checks if it is of type 'linear' to apply the corresponding scaling type and factor.
- **Output**: The method does not return a value but updates the GGUF writer with the configured parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.LlamaModel`](#cpp/convert_hf_to_ggufLlamaModel)  (Base Class)


---
#### LlamaModel\.permute<!-- {{#callable:llama.cpp/convert_hf_to_gguf.LlamaModel.permute}} -->
The `permute` method rearranges the dimensions of a tensor based on the number of attention heads and optionally adjusts the number of heads if a key-value head count is provided.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `weights`: A `Tensor` representing the weights to be permuted.
    - `n_head`: An integer representing the number of attention heads.
    - `n_head_kv`: An optional integer that specifies the number of key-value heads; if provided and different from `n_head`, it will override `n_head`.
- **Control Flow**:
    - Checks if `n_head_kv` is not None and differs from `n_head`, in which case it assigns `n_head_kv` to `n_head`.
    - Reshapes the `weights` tensor into a new shape that includes the number of heads and a fixed dimension of 2.
    - Swaps axes 1 and 2 of the reshaped tensor to rearrange the dimensions.
    - Reshapes the tensor back to its original shape before returning it.
- **Output**: Returns a tensor that has been permuted according to the specified number of heads, maintaining the original shape.
- **See also**: [`llama.cpp/convert_hf_to_gguf.LlamaModel`](#cpp/convert_hf_to_ggufLlamaModel)  (Base Class)


---
#### LlamaModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.LlamaModel.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on specific naming conventions and conditions related to attention heads and expert models.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the data to be modified.
    - `name`: A string representing the name associated with the tensor, which influences how the tensor is processed.
    - `bid`: An optional integer that indicates the block ID for expert processing.
- **Control Flow**:
    - Checks if the tensor is a vision tensor and returns an empty list if true.
    - Modifies the `name` based on specific conditions related to the architecture and naming conventions.
    - If `undo_permute` is true, permutes the tensor based on its name and the number of attention heads.
    - Handles expert tensors by storing them in a dictionary and merging them into a single tensor when a certain count is reached.
    - Returns a list of tuples containing the modified tensor name and the tensor itself.
- **Output**: Returns an iterable of tuples, each containing a modified tensor name and the corresponding tensor, or an empty list if no modifications are applicable.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.LlamaModel`](#cpp/convert_hf_to_ggufLlamaModel)  (Base Class)


---
#### LlamaModel\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.LlamaModel.prepare_tensors}} -->
The [`prepare_tensors`](#ModelBaseprepare_tensors) method prepares tensor data by invoking the parent class's method and checks for unprocessed expert tensors.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls the [`prepare_tensors`](#ModelBaseprepare_tensors) method of the parent class using `super()`.
    - Checks if the `_experts` attribute is not `None`.
    - Flattens the `_experts` list of dictionaries to extract keys (expert names).
    - Raises a `ValueError` if there are any unprocessed expert names.
- **Output**: The method does not return a value but raises a `ValueError` if there are unprocessed experts.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.LlamaModel`](#cpp/convert_hf_to_ggufLlamaModel)  (Base Class)



---
### LlavaVisionModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.LlavaVisionModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `img_break_tok_id`: An integer representing the image break token ID, initialized to -1.
- **Description**: The `LlavaVisionModel` class extends `MmprojModel` and is designed for conditional generation tasks, specifically tailored for the 'pixtral' model type, managing parameters and token IDs related to image processing in a vision model context.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.LlavaVisionModel.__init__`](#LlavaVisionModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.LlavaVisionModel.get_token_id`](#LlavaVisionModelget_token_id)
    - [`llama.cpp/convert_hf_to_gguf.LlavaVisionModel.set_gguf_parameters`](#LlavaVisionModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.LlavaVisionModel.modify_tensors`](#LlavaVisionModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)

**Methods**

---
#### LlavaVisionModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.LlavaVisionModel.__init__}} -->
Initializes the `LlavaVisionModel` class, setting parameters based on the model type.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `*args`: Variable length argument list for additional parameters.
    - `**kwargs`: Keyword arguments for additional parameters, including model hyperparameters.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method to initialize inherited attributes.
    - Checks if the `model_type` in `hparams` is 'pixtral'.
    - If 'pixtral', sets a default value for `layer_norm_eps` and retrieves the token ID for '[IMG_BREAK]'.
    - Logs the image break token ID.
    - If the `model_type` is not 'pixtral', raises a ValueError indicating unsupported model type.
- **Output**: No explicit output; initializes the instance with specific parameters and configurations.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
    - [`llama.cpp/convert_hf_to_gguf.LlavaVisionModel.get_token_id`](#LlavaVisionModelget_token_id)
- **See also**: [`llama.cpp/convert_hf_to_gguf.LlavaVisionModel`](#cpp/convert_hf_to_ggufLlavaVisionModel)  (Base Class)


---
#### LlavaVisionModel\.get\_token\_id<!-- {{#callable:llama.cpp/convert_hf_to_gguf.LlavaVisionModel.get_token_id}} -->
Retrieves the integer ID associated with a specified token from a tokenizer configuration file.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `token`: A string representing the token whose ID is to be retrieved.
- **Control Flow**:
    - Constructs the path to the tokenizer configuration file named 'tokenizer_config.json'.
    - Opens the configuration file and loads its content as a JSON object.
    - Iterates through the 'added_tokens_decoder' dictionary to find a matching token.
    - If a match is found, returns the corresponding token ID as an integer.
    - If no match is found after the iteration, raises a ValueError indicating the token was not found.
- **Output**: Returns the integer ID of the specified token if found; otherwise, raises a ValueError.
- **See also**: [`llama.cpp/convert_hf_to_gguf.LlavaVisionModel`](#cpp/convert_hf_to_ggufLlavaVisionModel)  (Base Class)


---
#### LlavaVisionModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.LlavaVisionModel.set_gguf_parameters}} -->
Sets parameters for the GGUF writer based on model hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class parameters are set.
    - Checks if the model type is 'pixtral' to proceed with specific parameter settings.
    - Adds the clip projector type and layer normalization epsilon to the GGUF writer.
    - Determines the activation function to use based on the `hidden_act` parameter, raising an error for unsupported types.
    - If `spatial_merge_size` is present in the global configuration, it adds this size to the GGUF writer.
- **Output**: No explicit return value; modifies the state of the GGUF writer based on the provided hyperparameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.LlavaVisionModel`](#cpp/convert_hf_to_ggufLlavaVisionModel)  (Base Class)


---
#### LlavaVisionModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.LlavaVisionModel.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on specific naming conventions and conditions.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that contains the data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method first deletes the unused `bid` argument.
    - It retrieves the number of attention heads from the model's hyperparameters.
    - If the `name` starts with 'multi_modal_projector.' or 'vision_tower.', it checks for specific suffixes to permute the tensor accordingly.
    - If the `name` contains 'embed_tokens.weight' and the `img_break_tok_id` is valid, it extracts a specific token embedding and maps its name.
    - If none of the conditions are met, it returns an empty list, indicating that no modifications were made.
- **Output**: The method returns an iterable of tuples, each containing a mapped tensor name and the modified tensor, or an empty list if no modifications were applicable.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.LlavaVisionModel`](#cpp/convert_hf_to_ggufLlavaVisionModel)  (Base Class)



---
### SmolVLMModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.SmolVLMModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `hparams`: A dictionary containing hyperparameters for the model.
    - `gguf_writer`: An object responsible for writing GGUF model parameters.
    - `global_config`: A configuration object that holds global settings for the model.
- **Description**: The `SmolVLMModel` class extends `MmprojModel` and is designed for conditional generation tasks, specifically tailored for the 'smolvlm_vision' model type, with custom handling of hyperparameters and tensor modifications.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.SmolVLMModel.__init__`](#SmolVLMModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.SmolVLMModel.set_gguf_parameters`](#SmolVLMModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.SmolVLMModel.tensor_force_quant`](#SmolVLMModeltensor_force_quant)
    - [`llama.cpp/convert_hf_to_gguf.SmolVLMModel.modify_tensors`](#SmolVLMModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)

**Methods**

---
#### SmolVLMModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.SmolVLMModel.__init__}} -->
Initializes the `SmolVLMModel` class, setting default hyperparameters for the model if it is of type 'smolvlm_vision'.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `*args`: Variable length argument list for passing positional arguments to the parent class.
    - `**kwargs`: Variable length keyword argument dictionary for passing keyword arguments to the parent class.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method to ensure proper initialization.
    - Checks if the `model_type` in `hparams` is 'smolvlm_vision'.
    - If the condition is true, it sets default values for `hidden_size`, `num_attention_heads`, and `intermediate_size` in `hparams` if they are not already provided.
- **Output**: No explicit output; the method initializes the instance's state based on the provided parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
- **See also**: [`llama.cpp/convert_hf_to_gguf.SmolVLMModel`](#cpp/convert_hf_to_ggufSmolVLMModel)  (Base Class)


---
#### SmolVLMModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.SmolVLMModel.set_gguf_parameters}} -->
Sets specific parameters for the GGUF model related to vision processing.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains the method, which holds configuration parameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base configurations are applied.
    - Adds a specific clip projector type to the GGUF writer.
    - Retrieves and sets the layer normalization epsilon value from the hyperparameters, defaulting to 1e-5 if not specified.
    - Retrieves and sets the projector scale factor from the global configuration, defaulting to 2 if not specified.
    - Enables the use of GELU activation in the vision model.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` with specific parameters for vision processing.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.SmolVLMModel`](#cpp/convert_hf_to_ggufSmolVLMModel)  (Base Class)


---
#### SmolVLMModel\.tensor\_force\_quant<!-- {{#callable:llama.cpp/convert_hf_to_gguf.SmolVLMModel.tensor_force_quant}} -->
Determines the quantization type for a tensor based on its name.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `name`: The name of the tensor, which is used to determine its quantization type.
    - `new_name`: An unused parameter that is not utilized in the method.
    - `bid`: An unused parameter that is not utilized in the method.
    - `n_dims`: An unused parameter that is not utilized in the method.
- **Control Flow**:
    - The method begins by deleting the unused parameters `bid`, `new_name`, and `n_dims`.
    - It checks if the string '.embeddings.' is present in the `name` argument.
    - If the condition is met, it returns the quantization type `gguf.GGMLQuantizationType.F32`.
    - If the condition is not met, it returns `False`.
- **Output**: Returns the quantization type `gguf.GGMLQuantizationType.F32` if the tensor name indicates it is an embedding; otherwise, it returns `False`.
- **See also**: [`llama.cpp/convert_hf_to_gguf.SmolVLMModel`](#cpp/convert_hf_to_ggufSmolVLMModel)  (Base Class)


---
#### SmolVLMModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.SmolVLMModel.modify_tensors}} -->
The `modify_tensors` method processes a given tensor based on its name, returning a modified tensor if it is identified as a vision tensor.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name of the tensor, which is used to determine if it is a vision tensor.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method begins by deleting the `bid` parameter as it is not utilized.
    - It checks if the `name` contains specific substrings that indicate it is a vision tensor.
    - If the tensor is identified as a vision tensor, it maps the tensor name using `self.map_tensor_name(name)` and returns a list containing a tuple of the mapped name and the original tensor.
    - If the tensor is not a vision tensor, it returns an empty list.
- **Output**: The method outputs an iterable of tuples, where each tuple contains a string (the mapped tensor name) and a `Tensor` (the original tensor), or an empty list if the tensor is not a vision tensor.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.SmolVLMModel`](#cpp/convert_hf_to_ggufSmolVLMModel)  (Base Class)



---
### Llama4Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.Llama4Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type for the model as Llama4.
    - `undo_permute`: A boolean flag indicating whether to undo permutation.
- **Description**: The `Llama4Model` class extends `LlamaModel` and is designed for conditional generation tasks, incorporating specific model architecture parameters and methods for setting vocabulary and modifying tensor data during processing.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Llama4Model.__init__`](#Llama4Model__init__)
    - [`llama.cpp/convert_hf_to_gguf.Llama4Model.set_vocab`](#Llama4Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.Llama4Model.set_gguf_parameters`](#Llama4Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Llama4Model.modify_tensors`](#Llama4Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel`](#cpp/convert_hf_to_ggufLlamaModel)

**Methods**

---
#### Llama4Model\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Llama4Model.__init__}} -->
Initializes the `Llama4Model` by setting specific hyperparameters related to intermediate sizes.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `args`: Variable length argument list that can be passed to the parent class constructor.
    - `kwargs`: Keyword arguments that can be passed to the parent class constructor.
- **Control Flow**:
    - Calls the parent class constructor using `super().__init__(*args, **kwargs)` to ensure proper initialization.
    - Sets the `intermediate_size_moe` hyperparameter to the value of the original `intermediate_size`.
    - Updates the `intermediate_size` hyperparameter to the value of `intermediate_size_mlp`.
- **Output**: No explicit output; the method initializes the instance's hyperparameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Llama4Model`](#cpp/convert_hf_to_ggufLlama4Model)  (Base Class)


---
#### Llama4Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Llama4Model.set_vocab}} -->
Sets the vocabulary for the model by configuring GPT-2 specific settings and enabling the addition of a beginning-of-sequence token.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls the [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) method to configure the vocabulary settings specific to the GPT-2 model.
    - Invokes the `add_add_bos_token` method on `gguf_writer` to enable the addition of a beginning-of-sequence token.
- **Output**: No return value; the method modifies the internal state of the model and its writer.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Llama4Model`](#cpp/convert_hf_to_ggufLlama4Model)  (Base Class)


---
#### Llama4Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Llama4Model.set_gguf_parameters}} -->
Sets specific parameters for the GGUF writer in the Llama4Model class.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the Llama4Model class.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class parameters are set.
    - Adds an interleave MOE layer step using the value from `self.hparams['interleave_moe_layer_step']`.
    - Adds the expert feed-forward length using the value from `self.hparams['intermediate_size_moe']`.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` with the specified parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Llama4Model`](#cpp/convert_hf_to_ggufLlama4Model)  (Base Class)


---
#### Llama4Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Llama4Model.modify_tensors}} -->
Modifies tensor data based on specific naming conventions and conditions.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name associated with the tensor, which influences how the tensor is modified.
    - `bid`: An optional integer that may be used for additional identification or processing.
- **Control Flow**:
    - Checks if the `name` starts with 'language_model.' and modifies it accordingly.
    - If `name` contains 'gate_up_proj', it splits the tensor into two parts: gate and up projections, and returns them with their mapped names.
    - If `name` ends with 'down_proj', it appends '.weight' to the name and transposes the tensor.
    - If `name` contains 'multi_modal_projector' or 'vision_model', it returns an empty list.
    - If none of the above conditions are met, it calls the parent class's [`modify_tensors`](#ModelBasemodify_tensors) method with the modified parameters.
- **Output**: Returns a list of tuples containing modified tensor names and their corresponding tensors, or an empty list, or the result of the parent class's [`modify_tensors`](#ModelBasemodify_tensors) method.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.modify_tensors`](#ModelBasemodify_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Llama4Model`](#cpp/convert_hf_to_ggufLlama4Model)  (Base Class)



---
### Llama4VisionModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.Llama4VisionModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `gguf_writer`: An instance of a writer for GGUF model parameters.
    - `hparams`: A dictionary containing hyperparameters for the model.
- **Description**: The `Llama4VisionModel` class extends `MmprojModel` and is designed for conditional generation tasks, specifically integrating vision capabilities with a focus on handling and modifying tensors related to vision models.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Llama4VisionModel.set_gguf_parameters`](#Llama4VisionModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Llama4VisionModel.modify_tensors`](#Llama4VisionModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)

**Methods**

---
#### Llama4VisionModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Llama4VisionModel.set_gguf_parameters}} -->
Sets specific parameters for the GGUF writer in the Llama4VisionModel.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the Llama4VisionModel class.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base parameters are set.
    - Adds a specific clip projector type for the GGUF writer.
    - Sets the layer normalization epsilon value from the model's hyperparameters.
    - Calculates and sets the projector scale factor based on the pixel shuffle ratio from hyperparameters.
    - Asserts that the hidden activation function is 'gelu' to ensure compatibility.
    - Enables the use of the 'gelu' activation function in the GGUF writer.
- **Output**: No explicit output; modifies the state of the GGUF writer with the specified parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Llama4VisionModel`](#cpp/convert_hf_to_ggufLlama4VisionModel)  (Base Class)


---
#### Llama4VisionModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Llama4VisionModel.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on the provided name, returning a tuple of modified tensor names and their corresponding tensors.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name associated with the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method begins by deleting the unused `bid` parameter.
    - It checks if the `name` contains either 'multi_modal_projector' or 'vision_model' to determine if the tensor should be processed.
    - If 'positional_embedding_vlm' is in the `name` and '.weight' is not, it appends '.weight' to the `name`.
    - If the `name` matches 'multi_modal_projector.linear_1', it returns a specific tensor name and the `data_torch` tensor.
    - For other valid names, it maps the name using `self.map_tensor_name` and returns it with the `data_torch` tensor.
    - If the `name` does not match the specified conditions, it returns an empty list.
- **Output**: The method returns an iterable of tuples, each containing a string (the modified tensor name) and the corresponding `Tensor` (data_torch), or an empty list if no conditions are met.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Llama4VisionModel`](#cpp/convert_hf_to_ggufLlama4VisionModel)  (Base Class)



---
### Mistral3Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.Mistral3Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as LLAMA.
- **Description**: The `Mistral3Model` class extends `LlamaModel` and is registered with the `ModelBase` under the identifier 'Mistral3ForConditionalGeneration', providing a specific architecture type and a method to modify tensors based on certain conditions.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Mistral3Model.modify_tensors`](#Mistral3Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel`](#cpp/convert_hf_to_ggufLlamaModel)

**Methods**

---
#### Mistral3Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Mistral3Model.modify_tensors}} -->
Modifies tensor data based on the provided name and bid, returning an empty list for specific names.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the data to be modified.
    - `name`: A string representing the name associated with the tensor.
    - `bid`: An optional integer or None representing an identifier.
- **Control Flow**:
    - The method first removes the prefix 'language_model.' from the `name` string.
    - It checks if the modified `name` contains 'multi_modal_projector' or 'vision_tower'; if so, it returns an empty list.
    - If the name does not match the specified conditions, it calls the parent class's [`modify_tensors`](#ModelBasemodify_tensors) method with the original arguments.
- **Output**: Returns an empty list if the name matches certain conditions; otherwise, it returns the result of the parent class's [`modify_tensors`](#ModelBasemodify_tensors) method.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.modify_tensors`](#ModelBasemodify_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Mistral3Model`](#cpp/convert_hf_to_ggufMistral3Model)  (Base Class)



---
### DeciModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.DeciModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type for the model.
    - `_num_kv_heads`: Stores the number of key-value heads for each block configuration.
    - `_num_heads`: Stores the number of attention heads for each block configuration.
    - `_ffn_dims`: Holds the dimensions for feed-forward networks calculated from multipliers.
- **Description**: The `DeciModel` class extends `TextModel` and is designed for causal language modeling, incorporating specific configurations for attention mechanisms and feed-forward networks based on provided hyperparameters.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.DeciModel._ffn_mult_to_intermediate_size`](#DeciModel_ffn_mult_to_intermediate_size)
    - [`llama.cpp/convert_hf_to_gguf.DeciModel._find_multiple`](#DeciModel_find_multiple)
    - [`llama.cpp/convert_hf_to_gguf.DeciModel.__init__`](#DeciModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.DeciModel.set_vocab`](#DeciModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.DeciModel.set_gguf_parameters`](#DeciModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.DeciModel.permute`](#DeciModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.DeciModel.modify_tensors`](#DeciModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.DeciModel.generate_extra_tensors`](#DeciModelgenerate_extra_tensors)
    - [`llama.cpp/convert_hf_to_gguf.DeciModel.prepare_tensors`](#DeciModelprepare_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### DeciModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeciModel.__init__}} -->
Initializes the `DeciModel` class, setting up attention and feed-forward network parameters based on provided hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `args`: Positional arguments passed to the parent class constructor.
    - `kwargs`: Keyword arguments passed to the parent class constructor, which may include hyperparameters.
- **Control Flow**:
    - Calls the parent class constructor using `super().__init__(*args, **kwargs)`.
    - Checks if 'block_configs' is present in `self.hparams` to configure model parameters.
    - Iterates over each block configuration to determine the number of key-value heads and attention heads based on the configuration.
    - Appends calculated values to `_num_kv_heads`, `_num_heads`, and `_ffn_multipliers` lists.
    - Validates the lengths of the lists against `self.block_count` to ensure consistency.
    - Calculates feed-forward dimensions using the [`_ffn_mult_to_intermediate_size`](#DeciModel_ffn_mult_to_intermediate_size) method based on multipliers.
- **Output**: No explicit return value; initializes instance variables for attention and feed-forward network configurations.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
    - [`llama.cpp/convert_hf_to_gguf.DeciModel._ffn_mult_to_intermediate_size`](#DeciModel_ffn_mult_to_intermediate_size)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeciModel`](#cpp/convert_hf_to_ggufDeciModel)  (Base Class)


---
#### DeciModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeciModel.set_vocab}} -->
Sets the vocabulary for the model based on its configuration.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `DeciModel` class, which contains model parameters and methods.
- **Control Flow**:
    - Checks if the `vocab_size` in `hparams` is equal to the default value of 128256.
    - If true, retrieves the base vocabulary using `get_vocab_base()` and adds the tokenizer model, pre-tokenizer, token list, and token types to the `gguf_writer`.
    - Creates a `SpecialVocab` instance and adds it to the `gguf_writer`.
    - If `vocab_size` is not equal to 128256, calls the `_set_vocab_llama_hf()` method for a different vocabulary setup.
- **Output**: No explicit return value; modifies the internal state of the `gguf_writer` with the vocabulary data.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel.get_vocab_base`](#TextModelget_vocab_base)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_llama_hf`](#TextModel_set_vocab_llama_hf)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeciModel`](#cpp/convert_hf_to_ggufDeciModel)  (Base Class)


---
#### DeciModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeciModel.set_gguf_parameters}} -->
Sets the GGUF parameters based on model hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class containing model hyperparameters and a GGUF writer.
- **Control Flow**:
    - Checks if 'block_configs' is present in the hyperparameters to determine the model type.
    - Asserts that the block count matches the lengths of key-value heads, heads, and feed-forward dimensions.
    - If 'rope_theta' is specified, adds it to the GGUF writer.
    - Adds various model parameters to the GGUF writer including head counts, block count, context length, and embedding length.
    - If 'block_configs' is not present, calls the parent class's method and checks for 'num_key_value_heads_per_layer'.
    - Adds vocabulary size and rope dimension count to the GGUF writer.
    - Handles rope scaling if specified in the hyperparameters.
- **Output**: No explicit return value; modifies the state of the GGUF writer with the set parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeciModel`](#cpp/convert_hf_to_ggufDeciModel)  (Base Class)


---
#### DeciModel\.permute<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeciModel.permute}} -->
The `permute` method rearranges the dimensions of a tensor based on the specified number of heads and optionally adjusts the number of heads if a key-value head count is provided.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `weights`: A `Tensor` that contains the weights to be permuted.
    - `n_head`: An integer representing the number of heads to be used in the permutation.
    - `n_head_kv`: An optional integer that specifies the number of key-value heads; if provided and different from `n_head`, it will override `n_head`.
- **Control Flow**:
    - Checks if `n_head_kv` is not None and differs from `n_head`, in which case it assigns `n_head_kv` to `n_head`.
    - Reshapes the `weights` tensor into a new shape based on `n_head`, then swaps axes 1 and 2, and finally reshapes it back to the original shape.
- **Output**: Returns a tensor that has been permuted according to the specified number of heads.
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeciModel`](#cpp/convert_hf_to_ggufDeciModel)  (Base Class)


---
#### DeciModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeciModel.modify_tensors}} -->
The `modify_tensors` method adjusts tensor data based on the specified parameters and returns a mapped tensor name with the modified tensor.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be modified.
    - `bid`: An optional integer that specifies the block ID, which influences the number of key-value heads used in the modification.
- **Control Flow**:
    - The method retrieves the number of attention heads from the model's hyperparameters.
    - If `bid` is provided, it checks for the number of key-value heads based on the hyperparameters or block configurations.
    - If `name` ends with 'q_proj.weight' or 'q_proj.bias', it permutes the tensor using the number of heads.
    - If `name` ends with 'k_proj.weight' or 'k_proj.bias', it permutes the tensor using the number of heads and key-value heads.
    - Finally, it returns a list containing the mapped tensor name and the modified tensor.
- **Output**: An iterable of tuples, each containing a string (the mapped tensor name) and a `Tensor` (the modified tensor).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeciModel`](#cpp/convert_hf_to_ggufDeciModel)  (Base Class)


---
#### DeciModel\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeciModel.prepare_tensors}} -->
This method calls the [`prepare_tensors`](#ModelBaseprepare_tensors) method of its superclass to perform tensor preparation.
- **Inputs**: None
- **Control Flow**:
    - The method directly invokes the [`prepare_tensors`](#ModelBaseprepare_tensors) method from the superclass without any additional logic.
- **Output**: The method does not return any value; it performs an operation defined in the superclass.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeciModel`](#cpp/convert_hf_to_ggufDeciModel)  (Base Class)



---
### BitnetModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.BitnetModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as Bitnet.
- **Description**: The `BitnetModel` class extends `TextModel` and is designed for causal language modeling, incorporating specific methods for vocabulary setting, parameter configuration, and tensor modification, particularly focusing on weight quantization for efficient processing.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.BitnetModel.set_vocab`](#BitnetModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.BitnetModel.set_gguf_parameters`](#BitnetModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.BitnetModel.weight_quant`](#BitnetModelweight_quant)
    - [`llama.cpp/convert_hf_to_gguf.BitnetModel.modify_tensors`](#BitnetModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### BitnetModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BitnetModel.set_vocab}} -->
Sets the vocabulary for the model using a sentencepiece tokenizer.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls the private method `_set_vocab_sentencepiece()` to perform the actual vocabulary setting.
- **Output**: No output is returned; the method modifies the internal state of the model by setting the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BitnetModel`](#cpp/convert_hf_to_ggufBitnetModel)  (Base Class)


---
#### BitnetModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BitnetModel.set_gguf_parameters}} -->
Sets specific parameters for the GGUF model, including rope scaling type and factor.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base parameters are set.
    - Adds a linear rope scaling type to the `gguf_writer`.
    - Sets the rope scaling factor to 1.0 in the `gguf_writer`.
- **Output**: No output is returned; the method modifies the internal state of the `gguf_writer`.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BitnetModel`](#cpp/convert_hf_to_ggufBitnetModel)  (Base Class)


---
#### BitnetModel\.weight\_quant<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BitnetModel.weight_quant}} -->
The `weight_quant` method quantizes the input tensor `weight` by scaling and rounding its values to fit within the range of -1 to 1.
- **Inputs**:
    - `weight`: A `Tensor` representing the weights to be quantized.
- **Control Flow**:
    - The method retrieves the data type of the input `weight` tensor.
    - The `weight` tensor is converted to a float type to ensure precision during calculations.
    - The mean of the absolute values of the `weight` tensor is computed and clamped to a minimum of 1e-5 to avoid division by zero.
    - An inverse scale (`iscale`) is calculated from the computed scale.
    - The `weight` tensor is scaled by `iscale`, rounded to the nearest integer, clamped to the range of -1 to 1, and then divided by `iscale` to revert the scaling.
    - Finally, the result is converted back to the original data type of the input tensor and returned.
- **Output**: The method returns a quantized `Tensor` with values rounded and clamped to the range of -1 to 1, maintaining the original data type.
- **See also**: [`llama.cpp/convert_hf_to_gguf.BitnetModel`](#cpp/convert_hf_to_ggufBitnetModel)  (Base Class)


---
#### BitnetModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BitnetModel.modify_tensors}} -->
The `modify_tensors` method transforms tensor weights based on their names and yields the modified tensor along with a new name.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the weights to be modified.
    - `name`: A string representing the original name of the tensor.
    - `bid`: An optional integer that may be used to identify a specific model or context.
- **Control Flow**:
    - The method first maps the original tensor name to a new name using `self.map_tensor_name(name)`.
    - It checks if the new name matches any predefined model tensor names using `self.match_model_tensor_name` for a list of keys.
    - If a match is found, it quantizes the tensor weights using the `self.weight_quant(data_torch)` method.
    - Finally, it yields a tuple containing the new name and the (possibly modified) tensor.
- **Output**: The method outputs an iterable of tuples, each containing a string (the new tensor name) and a `Tensor` (the modified tensor).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.match_model_tensor_name`](#ModelBasematch_model_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.BitnetModel.weight_quant`](#BitnetModelweight_quant)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BitnetModel`](#cpp/convert_hf_to_ggufBitnetModel)  (Base Class)



---
### GrokModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.GrokModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture of the model as GROK.
    - `_experts`: Holds a list of dictionaries for expert tensors, initialized to None.
- **Description**: The `GrokModel` class extends `TextModel` and is designed for causal language modeling, incorporating a specific model architecture and handling expert tensors for enhanced processing.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.GrokModel.set_vocab`](#GrokModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.GrokModel.__init__`](#GrokModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.GrokModel.set_gguf_parameters`](#GrokModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.GrokModel.modify_tensors`](#GrokModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### GrokModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.GrokModel.__init__}} -->
Initializes a `GrokModel` instance by calling the parent class's [`__init__`](#ModelBase__init__) method with any provided arguments.
- **Inputs**:
    - `args`: A variable-length argument list that can include any number of positional arguments.
    - `kwargs`: A variable-length keyword argument dictionary that can include any number of named arguments.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method using `super()` to ensure proper initialization of the base class.
    - Passes all received positional and keyword arguments to the parent class's [`__init__`](#ModelBase__init__) method.
- **Output**: The method does not return any value; it initializes the instance of `GrokModel`.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
- **See also**: [`llama.cpp/convert_hf_to_gguf.GrokModel`](#cpp/convert_hf_to_ggufGrokModel)  (Base Class)


---
#### GrokModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.GrokModel.set_gguf_parameters}} -->
The [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method in the `GrokModel` class calls the same method from its parent class.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - The method directly invokes the [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method of the superclass without any additional logic or parameters.
- **Output**: The method does not return any value; it simply executes the parent class's method.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.GrokModel`](#cpp/convert_hf_to_ggufGrokModel)  (Base Class)



---
### DbrxModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.DbrxModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as DBRX.
- **Description**: The `DbrxModel` class extends `TextModel` and is designed for handling the DBRX architecture, providing methods to set parameters for GGUF format and modify tensor data specific to the model's requirements.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.DbrxModel.set_gguf_parameters`](#DbrxModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.DbrxModel.modify_tensors`](#DbrxModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.DbrxModel.tensor_force_quant`](#DbrxModeltensor_force_quant)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### DbrxModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DbrxModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on model hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `DbrxModel` class, which contains model hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves feed-forward and attention configurations from the model's hyperparameters.
    - Adds various model parameters to the GGUF writer, including block count, context length, embedding length, and others.
    - Logs the file type of the GGUF writer.
- **Output**: The method does not return a value; it modifies the state of the GGUF writer with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.DbrxModel`](#cpp/convert_hf_to_ggufDbrxModel)  (Base Class)


---
#### DbrxModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DbrxModel.modify_tensors}} -->
The `modify_tensors` method modifies tensor data based on specific naming conventions and configurations for expert models.
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name of the tensor, which is used to determine how to modify the tensor.
    - `bid`: An optional integer that is unused in the current implementation.
- **Control Flow**:
    - The method retrieves the number of experts, feed-forward size, and embedding size from the model's hyperparameters.
    - It defines a mapping of expert tensor names to their corresponding permutation configurations.
    - The method checks if the provided tensor name matches any expert tensor names and modifies the tensor's shape and order accordingly.
    - Finally, it maps the tensor name to a new name, appending '.weight' if it corresponds to an expert tensor, and returns a list containing the new name and modified tensor.
- **Output**: Returns an iterable of tuples, each containing the modified tensor name and the modified tensor data.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DbrxModel`](#cpp/convert_hf_to_ggufDbrxModel)  (Base Class)


---
#### DbrxModel\.tensor\_force\_quant<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DbrxModel.tensor_force_quant}} -->
Determines if quantization is necessary based on the number of dimensions.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `name`: A string representing the original name of the tensor.
    - `new_name`: A string representing the new name for the tensor.
    - `bid`: An optional integer that may represent a batch ID, but is unused in this method.
    - `n_dims`: An integer representing the number of dimensions of the tensor.
- **Control Flow**:
    - The method begins by deleting the unused input arguments `name`, `new_name`, and `bid`.
    - It then evaluates whether `n_dims` is greater than 1, returning `True` if it is, and `False` otherwise.
- **Output**: Returns a boolean indicating whether the tensor has more than one dimension, which suggests that quantization may be applicable.
- **See also**: [`llama.cpp/convert_hf_to_gguf.DbrxModel`](#cpp/convert_hf_to_ggufDbrxModel)  (Base Class)



---
### MiniCPMModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.MiniCPMModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as MiniCPM.
- **Description**: The `MiniCPMModel` class extends `TextModel` and is designed for causal language modeling, incorporating methods for setting model parameters, generating tensors, and modifying tensor data specific to the MiniCPM architecture.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.MiniCPMModel.set_gguf_parameters`](#MiniCPMModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.MiniCPMModel.generate_extra_tensors`](#MiniCPMModelgenerate_extra_tensors)
    - [`llama.cpp/convert_hf_to_gguf.MiniCPMModel.set_vocab`](#MiniCPMModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.MiniCPMModel.modify_tensors`](#MiniCPMModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### MiniCPMModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MiniCPMModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `MiniCPMModel` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class parameters are set.
    - Retrieves the embedding scale from `self.hparams` and adds it to the GGUF writer.
    - Logs the embedding scale value.
    - Calculates the residual scale based on depth and number of hidden layers, adds it to the GGUF writer, and logs the value.
    - Calculates the logit scale from hidden size and model base dimension, adds it to the GGUF writer, and logs the value.
    - Checks for rope scaling parameters and adds the rope scaling type to the GGUF writer if it is set to 'longrope', logging the type.
- **Output**: The method does not return a value but modifies the state of the GGUF writer with the specified parameters and logs the actions taken.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MiniCPMModel`](#cpp/convert_hf_to_ggufMiniCPMModel)  (Base Class)


---
#### MiniCPMModel\.generate\_extra\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MiniCPMModel.generate_extra_tensors}} -->
Generates additional tensors for rope scaling based on model hyperparameters.
- **Inputs**:
    - `self`: An instance of the class containing hyperparameters and methods for tensor generation.
- **Control Flow**:
    - Calculates `rope_dims` by dividing the hidden size by the number of attention heads.
    - Retrieves the `rope_scaling` parameters from the hyperparameters.
    - Checks if `rope_scaling` is not None and retrieves `long_factors` and `short_factors`.
    - Raises a KeyError if either `long_factors` or `short_factors` is missing.
    - Validates the lengths of `long_factors` and `short_factors` against `rope_dims`.
    - Yields formatted tensor names and their corresponding tensor values for long and short factors.
- **Output**: Yields tuples containing the names and tensor representations of the long and short rope factors.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MiniCPMModel`](#cpp/convert_hf_to_ggufMiniCPMModel)  (Base Class)


---
#### MiniCPMModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MiniCPMModel.set_vocab}} -->
Sets the vocabulary for the model using the SentencePiece tokenizer.
- **Inputs**: None
- **Control Flow**:
    - Calls the private method `_set_vocab_sentencepiece()` to perform the actual vocabulary setting.
- **Output**: No output is returned; the method modifies the internal state of the model by setting the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MiniCPMModel`](#cpp/convert_hf_to_ggufMiniCPMModel)  (Base Class)


---
#### MiniCPMModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MiniCPMModel.modify_tensors}} -->
The `modify_tensors` method adjusts tensor data based on the specified tensor name and returns a mapped name with the modified tensor.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the tensor data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be modified.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method retrieves the number of attention heads and key-value heads from the model's hyperparameters.
    - It checks if the tensor name ends with 'q_proj.weight' or 'k_proj.weight' to determine if the tensor needs to be permuted.
    - If the name matches, it calls `LlamaModel.permute` to adjust the tensor accordingly.
- **Output**: Returns an iterable of tuples, each containing a mapped tensor name and the modified tensor.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MiniCPMModel`](#cpp/convert_hf_to_ggufMiniCPMModel)  (Base Class)



---
### MiniCPM3Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.MiniCPM3Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `MiniCPM3Model` class extends `TextModel` and is designed for causal language modeling, incorporating methods for setting model parameters, generating tensors, and managing vocabulary.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.MiniCPM3Model.set_gguf_parameters`](#MiniCPM3Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.MiniCPM3Model.generate_extra_tensors`](#MiniCPM3Modelgenerate_extra_tensors)
    - [`llama.cpp/convert_hf_to_gguf.MiniCPM3Model.set_vocab`](#MiniCPM3Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.MiniCPM3Model._reverse_hf_permute`](#MiniCPM3Model_reverse_hf_permute)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### MiniCPM3Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MiniCPM3Model.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `MiniCPM3Model` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves hyperparameters from the instance's `hparams` attribute.
    - Calls methods on the `gguf_writer` to set various model parameters such as file type, context length, embedding length, and others based on the retrieved hyperparameters.
    - Checks if 'q_lora_rank' exists in `hparams` and is not None, and if so, adds it to the `gguf_writer`.
    - Adds the key length and rope dimension count to the `gguf_writer`.
- **Output**: The method does not return a value; it configures the GGUF writer with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.MiniCPM3Model`](#cpp/convert_hf_to_ggufMiniCPM3Model)  (Base Class)


---
#### MiniCPM3Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MiniCPM3Model.set_vocab}} -->
Sets the vocabulary for the model using a sentencepiece tokenizer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `MiniCPM3Model` class.
- **Control Flow**:
    - Calls the private method `_set_vocab_sentencepiece()` to perform the actual vocabulary setting.
- **Output**: No output is returned; the method modifies the internal state of the model by setting the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MiniCPM3Model`](#cpp/convert_hf_to_ggufMiniCPM3Model)  (Base Class)



---
### QwenModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.QwenModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `QwenModel` class extends `TextModel` and is designed for handling specific model architecture and tokenization processes, including methods for converting byte sequences to strings and applying byte pair encoding.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.QwenModel.token_bytes_to_string`](#QwenModeltoken_bytes_to_string)
    - [`llama.cpp/convert_hf_to_gguf.QwenModel.bpe`](#QwenModelbpe)
    - [`llama.cpp/convert_hf_to_gguf.QwenModel.set_vocab`](#QwenModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.QwenModel.set_gguf_parameters`](#QwenModelset_gguf_parameters)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### QwenModel\.token\_bytes\_to\_string<!-- {{#callable:llama.cpp/convert_hf_to_gguf.QwenModel.token_bytes_to_string}} -->
Converts a byte string into a human-readable string using a specific byte encoding.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `b`: A byte string that needs to be converted to a human-readable string.
- **Control Flow**:
    - Imports the `bytes_to_unicode` function from the `transformers` library.
    - Creates a `byte_encoder` mapping using the `bytes_to_unicode` function.
    - Decodes the byte string `b` using 'latin-1' encoding.
    - Iterates over each character in the decoded string, converting it to its corresponding Unicode character using the `byte_encoder`.
    - Joins the list of characters into a single string and returns it.
- **Output**: A human-readable string representation of the input byte string.
- **See also**: [`llama.cpp/convert_hf_to_gguf.QwenModel`](#cpp/convert_hf_to_ggufQwenModel)  (Base Class)


---
#### QwenModel\.bpe<!-- {{#callable:llama.cpp/convert_hf_to_gguf.QwenModel.bpe}} -->
The `bpe` method performs byte pair encoding on a given token using specified mergeable ranks.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `mergeable_ranks`: A dictionary mapping byte pairs to their corresponding ranks, indicating how mergeable they are.
    - `token`: A byte sequence representing the token to be processed.
    - `max_rank`: An optional integer that specifies the maximum rank for merging; if provided, merging stops when the rank reaches or exceeds this value.
- **Control Flow**:
    - The method initializes `parts` as a list of single-byte sequences from the input `token`.
    - It enters a loop that continues until no more merges can be performed based on the ranks.
    - Within the loop, it iterates over adjacent byte pairs in `parts` to find the pair with the lowest rank from `mergeable_ranks`.
    - If a valid minimum rank is found and it is below `max_rank` (if specified), the corresponding byte pair is merged into a single byte, and the list `parts` is updated.
    - The loop breaks when no more valid merges can be found or when the minimum rank meets or exceeds `max_rank`.
- **Output**: The method returns a list of bytes representing the token after applying byte pair encoding, with merges performed according to the specified ranks.
- **See also**: [`llama.cpp/convert_hf_to_gguf.QwenModel`](#cpp/convert_hf_to_ggufQwenModel)  (Base Class)


---
#### QwenModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.QwenModel.set_vocab}} -->
The `set_vocab` method initializes the vocabulary for the Qwen model by invoking a specific internal method.
- **Decorators**: `@staticmethod`
- **Inputs**: None
- **Control Flow**:
    - The method calls the internal method `_set_vocab_qwen()` to perform the vocabulary setup.
- **Output**: The method does not return any value; it is intended to set up the vocabulary for the model internally.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_qwen`](#TextModel_set_vocab_qwen)
- **See also**: [`llama.cpp/convert_hf_to_gguf.QwenModel`](#cpp/convert_hf_to_ggufQwenModel)  (Base Class)


---
#### QwenModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.QwenModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on model hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `QwenModel` class, which contains model hyperparameters and a GGUF writer.
- **Control Flow**:
    - The method accesses the `hparams` dictionary to retrieve various hyperparameters related to the model architecture.
    - It calls multiple methods on the `gguf_writer` object to set parameters such as context length, block count, embedding length, and others using the retrieved hyperparameters.
- **Output**: This method does not return a value; it modifies the state of the `gguf_writer` by setting various parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.QwenModel`](#cpp/convert_hf_to_ggufQwenModel)  (Base Class)



---
### Qwen2Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.Qwen2Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as QWEN2.
- **Description**: The `Qwen2Model` class extends `TextModel` and is designed for handling various model types, including causal language modeling and audio generation, while providing methods for vocabulary setting and tensor modification.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Qwen2Model.set_vocab`](#Qwen2Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.Qwen2Model.set_gguf_parameters`](#Qwen2Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Qwen2Model.modify_tensors`](#Qwen2Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### Qwen2Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2Model.set_vocab}} -->
The `set_vocab` method attempts to set the vocabulary using a SentencePiece model and falls back to a GPT-2 model if a FileNotFoundError occurs.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - The method first attempts to call the [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece) method to set the vocabulary.
    - If a `FileNotFoundError` is raised during the execution of [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece), it catches the exception and calls the [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) method instead.
- **Output**: The method does not return a value; it modifies the internal state of the object by setting the vocabulary based on the available model.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2Model`](#cpp/convert_hf_to_ggufQwen2Model)  (Base Class)


---
#### Qwen2Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2Model.set_gguf_parameters}} -->
Sets parameters related to the GGUF model, specifically handling rope scaling configurations.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains the method, which holds model parameters and configurations.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base configurations are set.
    - Attempts to set the pooling type by calling [`_try_set_pooling_type`](#TextModel_try_set_pooling_type).
    - Retrieves the `rope_scaling` configuration from the model's hyperparameters, defaulting to an empty dictionary if not found.
    - Checks if the `rope_type` is 'yarn' and if a 'factor' is present in the `rope_scaling` dictionary.
    - If the conditions are met, adds the rope scaling type, factor, and original context length to the `gguf_writer`.
- **Output**: No explicit return value; modifies the state of the `gguf_writer` with rope scaling parameters if conditions are satisfied.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._try_set_pooling_type`](#TextModel_try_set_pooling_type)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2Model`](#cpp/convert_hf_to_ggufQwen2Model)  (Base Class)


---
#### Qwen2Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2Model.modify_tensors}} -->
The [`modify_tensors`](#ModelBasemodify_tensors) method modifies tensor names based on specific conditions and yields modified tensors from the parent class.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that contains the data to be modified.
    - `name`: A string representing the name of the tensor.
    - `bid`: An optional integer that may represent a batch ID.
- **Control Flow**:
    - Checks if the architecture is 'Qwen2Model' and modifies the tensor name accordingly.
    - Replaces 'language_model.' in the tensor name for compatibility with InternVL.
    - Checks if the tensor name starts with specific prefixes related to vision and audio models, and returns an empty list if it does.
    - Yields modified tensors from the parent class's [`modify_tensors`](#ModelBasemodify_tensors) method.
- **Output**: An iterable of tuples, each containing a string (the modified tensor name) and a `Tensor` object.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.modify_tensors`](#ModelBasemodify_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2Model`](#cpp/convert_hf_to_ggufQwen2Model)  (Base Class)



---
### Qwen2VLModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.Qwen2VLModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `Qwen2VLModel` class extends `TextModel` and is designed for conditional generation tasks, incorporating specific model architecture and methods for setting parameters and vocabulary.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Qwen2VLModel.set_gguf_parameters`](#Qwen2VLModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Qwen2VLModel.set_vocab`](#Qwen2VLModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.Qwen2VLModel.modify_tensors`](#Qwen2VLModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### Qwen2VLModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2VLModel.set_gguf_parameters}} -->
Sets the GGUF parameters for the model by updating the rope section and adding it to the GGUF writer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `Qwen2VLModel` class.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class setup is performed.
    - Retrieves the `mrope_section` from the model's hyperparameters, specifically from the `rope_scaling` dictionary.
    - Extends the `mrope_section` list with zeros to ensure it has a length of at least 4.
    - Adds the updated `mrope_section` to the GGUF writer using the `add_rope_dimension_sections` method.
- **Output**: The method does not return a value; it modifies the internal state of the model by updating the GGUF parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2VLModel`](#cpp/convert_hf_to_ggufQwen2VLModel)  (Base Class)


---
#### Qwen2VLModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2VLModel.set_vocab}} -->
The `set_vocab` method attempts to set the vocabulary using a SentencePiece model, falling back to a GPT-2 model if the SentencePiece model file is not found.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - The method first attempts to call the [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece) method to set the vocabulary.
    - If a `FileNotFoundError` is raised during the execution of [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece), it catches the exception and calls the [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) method instead.
- **Output**: The method does not return a value; it modifies the internal state of the object by setting the vocabulary based on the available model.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2VLModel`](#cpp/convert_hf_to_ggufQwen2VLModel)  (Base Class)


---
#### Qwen2VLModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2VLModel.modify_tensors}} -->
The `modify_tensors` method processes a tensor based on its name, modifying the name if it starts with 'thinker.' and filtering out certain multimodal tensor names.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name associated with the tensor.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method begins by deleting the `bid` parameter as it is unused.
    - If the `name` starts with 'thinker.', it is modified to remove this prefix.
    - The method checks if the `name` starts with any of the specified prefixes ('visual', 'audio', 'talker', 'token2wav') to determine if it should skip processing.
    - If the name matches any of the specified prefixes, the method returns an empty list, indicating no tensors to process.
    - If the name does not match the specified prefixes, the method returns a list containing a tuple of the modified name and the original tensor.
- **Output**: The output is an iterable of tuples, where each tuple contains a modified tensor name and the original tensor, or an empty list if the tensor is to be skipped.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2VLModel`](#cpp/convert_hf_to_ggufQwen2VLModel)  (Base Class)



---
### Qwen2VLVisionModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `hparams_vision`: A dictionary containing hyperparameters related to vision processing.
    - `global_config`: A dictionary holding global configuration settings for the model.
    - `gguf_writer`: An object responsible for writing GGUF model parameters.
- **Description**: The `Qwen2VLVisionModel` class extends `MmprojModel` and is designed for vision-related tasks, incorporating various hyperparameters and configurations to manage model behavior, particularly in relation to image processing and tensor modifications.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel.__init__`](#Qwen2VLVisionModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel.set_gguf_parameters`](#Qwen2VLVisionModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel.tensor_force_quant`](#Qwen2VLVisionModeltensor_force_quant)
    - [`llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel.modify_tensors`](#Qwen2VLVisionModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)

**Methods**

---
#### Qwen2VLVisionModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel.__init__}} -->
Initializes the `Qwen2VLVisionModel` by setting up hyperparameters and renaming configuration values.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `*args`: Variable length argument list for additional parameters.
    - `**kwargs`: Keyword arguments for additional parameters.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method to initialize inherited attributes.
    - Asserts that `hparams_vision` is not None to ensure necessary parameters are provided.
    - Sets a default value for `image_size` if it is not already specified in `hparams_vision`.
    - Renames configuration values from `num_heads` to `num_attention_heads` and from `depth` to `num_hidden_layers`.
    - If `embed_dim` is present in `hparams_vision`, it sets `intermediate_size` and `hidden_size` based on `hidden_size` and `embed_dim` respectively.
- **Output**: No explicit output; initializes the instance with configured hyperparameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel`](#cpp/convert_hf_to_ggufQwen2VLVisionModel)  (Base Class)


---
#### Qwen2VLVisionModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel.set_gguf_parameters}} -->
Sets parameters for the GGUF model based on the vision model type.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains the method.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to initialize parameters.
    - Asserts that `hparams_vision` is not None to ensure necessary parameters are available.
    - Retrieves the model type from `global_config` to determine the appropriate settings.
    - Based on the model type, adds the corresponding clip projector type to `gguf_writer`.
    - For 'qwen2_5_vl' and 'qwen2_5_omni', retrieves and validates `fullatt_block_indexes` to compute the window attention pattern.
    - Raises a ValueError if the model type is unknown or if the `fullatt_block_indexes` validation fails.
    - Adds default values for attention layer normalization epsilon from the global configuration.
- **Output**: No explicit return value; modifies the state of the `gguf_writer` with the set parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel`](#cpp/convert_hf_to_ggufQwen2VLVisionModel)  (Base Class)


---
#### Qwen2VLVisionModel\.tensor\_force\_quant<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel.tensor_force_quant}} -->
Determines the quantization type for a tensor based on its name.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `name`: An identifier for the tensor, which is not used in the method.
    - `new_name`: The new name of the tensor, which is used to determine the quantization type.
    - `bid`: An unused argument in the method.
    - `n_dims`: An unused argument in the method.
- **Control Flow**:
    - The method first deletes the unused arguments `bid`, `name`, and `n_dims`.
    - It checks if the string '.patch_embd.' is present in `new_name`, returning `gguf.GGMLQuantizationType.F16` if true.
    - If the first condition is not met, it checks if '.position_embd.' is in `new_name`, returning `gguf.GGMLQuantizationType.F32` if true.
    - If neither condition is satisfied, it returns `False`.
- **Output**: Returns the quantization type as `gguf.GGMLQuantizationType.F16`, `gguf.GGMLQuantizationType.F32`, or `False` based on the conditions checked.
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel`](#cpp/convert_hf_to_ggufQwen2VLVisionModel)  (Base Class)


---
#### Qwen2VLVisionModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel.modify_tensors}} -->
The `modify_tensors` method processes visual tensors, potentially splitting them based on their names and dimensions.
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that is currently unused in the method.
- **Control Flow**:
    - The method first checks if the `name` starts with 'visual.' to determine if it should process the tensor.
    - If the name contains '.qkv.', it splits the tensor into three parts (Q, K, V) based on its dimensions.
    - If the name contains 'patch_embed.proj.weight', it splits a 3D convolution tensor into two 2D tensors.
    - If the name does not match any specific conditions, it returns the tensor as is, mapped to its name.
    - If the name does not start with 'visual.', the method returns an empty list, skipping the tensor.
- **Output**: The method returns an iterable of tuples, each containing a modified tensor name and the corresponding tensor, or an empty list if the tensor is not processed.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel`](#cpp/convert_hf_to_ggufQwen2VLVisionModel)  (Base Class)



---
### Qwen25OmniModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.Qwen25OmniModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `has_vision_encoder`: Indicates if the model has a vision encoder.
    - `has_audio_encoder`: Indicates if the model has an audio encoder.
- **Description**: The `Qwen25OmniModel` class extends `Qwen2VLVisionModel` and is designed to handle both vision and audio processing, initializing specific audio parameters and providing methods to configure and modify tensors related to audio and vision data.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.__init__`](#Qwen25OmniModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.set_gguf_parameters`](#Qwen25OmniModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.get_vision_config`](#Qwen25OmniModelget_vision_config)
    - [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.get_audio_config`](#Qwen25OmniModelget_audio_config)
    - [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.generate_extra_tensors`](#Qwen25OmniModelgenerate_extra_tensors)
    - [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.tensor_force_quant`](#Qwen25OmniModeltensor_force_quant)
    - [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.modify_tensors`](#Qwen25OmniModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.Qwen2VLVisionModel`](#cpp/convert_hf_to_ggufQwen2VLVisionModel)

**Methods**

---
#### Qwen25OmniModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.__init__}} -->
Initializes the `Qwen25OmniModel` class, setting up audio hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `*args`: Variable length argument list for additional parameters.
    - `**kwargs`: Keyword arguments for additional parameters.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method to ensure proper initialization.
    - Asserts that the `hparams_audio` attribute is not None to prevent errors.
    - Sets the `hidden_size`, `intermediate_size`, and `num_attention_heads` in `hparams_audio` based on existing parameters.
- **Output**: No explicit output; initializes the instance with configured audio hyperparameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel`](#cpp/convert_hf_to_ggufQwen25OmniModel)  (Base Class)


---
#### Qwen25OmniModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.set_gguf_parameters}} -->
Sets parameters related to audio processing in the GGUF format.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `Qwen25OmniModel` class.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class parameters are set.
    - Asserts that `self.hparams_audio` is not None to ensure audio parameters are available.
    - Adds the number of mel bins from `self.hparams_audio` to the `gguf_writer`.
    - Adds the layer normalization epsilon value from `self.hparams_audio` to the `gguf_writer`, defaulting to 1e-5 if not specified.
- **Output**: No explicit output; modifies the state of the `gguf_writer` with audio parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel`](#cpp/convert_hf_to_ggufQwen25OmniModel)  (Base Class)


---
#### Qwen25OmniModel\.get\_vision\_config<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.get_vision_config}} -->
Retrieves the vision configuration from the global configuration dictionary.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `Qwen25OmniModel` class.
- **Control Flow**:
    - Accesses the `global_config` attribute of the instance.
    - Retrieves the value associated with the key 'thinker_config'.
    - Returns the value of 'vision_config' from the 'thinker_config' dictionary, or None if it does not exist.
- **Output**: Returns a dictionary containing the vision configuration or None if the configuration is not found.
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel`](#cpp/convert_hf_to_ggufQwen25OmniModel)  (Base Class)


---
#### Qwen25OmniModel\.get\_audio\_config<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.get_audio_config}} -->
Retrieves the audio configuration from the global configuration dictionary.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `Qwen25OmniModel` class.
- **Control Flow**:
    - Accesses the `global_config` attribute of the instance.
    - Retrieves the value associated with the key 'thinker_config' from `global_config`.
    - Attempts to get the 'audio_config' from the 'thinker_config' dictionary.
- **Output**: Returns the audio configuration as a dictionary or None if it does not exist.
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel`](#cpp/convert_hf_to_ggufQwen25OmniModel)  (Base Class)


---
#### Qwen25OmniModel\.generate\_extra\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.generate_extra_tensors}} -->
Generates sinusoidal position embeddings for audio input.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `Qwen25OmniModel` class, which contains audio hyperparameters.
- **Control Flow**:
    - Asserts that `self.hparams_audio` is not None to ensure audio parameters are available.
    - Defines constants for maximum timescale, length of the embedding, and number of channels based on audio parameters.
    - Calculates the logarithmic increment for timescales and generates inverse timescales using exponential decay.
    - Creates a scaled time tensor by multiplying the time range with the inverse timescales.
    - Computes the position embedding by concatenating sine and cosine values of the scaled time tensor.
    - Yields a tuple containing the name of the tensor and the computed position embedding.
- **Output**: Yields a tuple where the first element is the string 'audio_tower.embed_positions.weight' and the second element is a tensor containing the sinusoidal position embeddings.
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel`](#cpp/convert_hf_to_ggufQwen25OmniModel)  (Base Class)


---
#### Qwen25OmniModel\.tensor\_force\_quant<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.tensor_force_quant}} -->
The `tensor_force_quant` method determines the quantization type for a given tensor name based on specific substring conditions.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `name`: A string representing the name of the tensor.
    - `new_name`: A string representing a new name for the tensor, which is unused in the method.
    - `bid`: An identifier for the tensor, which is unused in the method.
    - `n_dims`: The number of dimensions of the tensor, which is unused in the method.
- **Control Flow**:
    - The method begins by deleting the unused parameters `bid`, `new_name`, and `n_dims`.
    - It checks if the `name` contains both '.conv' and '.weight'.
    - If both conditions are met, it returns the quantization type `gguf.GGMLQuantizationType.F16`.
    - If the conditions are not met, it returns `False`.
- **Output**: The output is either the quantization type `F16` if the conditions are satisfied, or `False` if they are not.
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel`](#cpp/convert_hf_to_ggufQwen25OmniModel)  (Base Class)


---
#### Qwen25OmniModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen25OmniModel.modify_tensors}} -->
The [`modify_tensors`](#ModelBasemodify_tensors) method processes and modifies tensor data based on specific naming conventions.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that may be used for additional identification or processing, but is not utilized in this method.
- **Control Flow**:
    - The method first checks if the `name` starts with 'thinker.' and removes this prefix if present.
    - If the modified `name` starts with 'audio_tower', it processes the tensor accordingly.
    - For 'conv1.bias' or 'conv2.bias', it applies an unsqueeze operation to the tensor.
    - If the name is 'audio_bos_eos_token', it returns an empty list, indicating that this tensor is unused.
    - If the name does not match the specific conditions, it calls the parent class's [`modify_tensors`](#ModelBasemodify_tensors) method for further processing.
- **Output**: The method returns an iterable of tuples containing the modified tensor name and the modified tensor, or an empty list if the tensor is unused.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.modify_tensors`](#ModelBasemodify_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen25OmniModel`](#cpp/convert_hf_to_ggufQwen25OmniModel)  (Base Class)



---
### InternVisionModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.InternVisionModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `gguf_writer`: An instance of a writer for GGUF format, used to add various model parameters.
    - `hparams`: A dictionary containing hyperparameters for the model.
    - `global_config`: A configuration object that holds global settings for the model.
- **Description**: The `InternVisionModel` class extends `MmprojModel` and is designed to manage and modify parameters specific to a vision model, including setting GGUF parameters and modifying tensor data for model training.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.InternVisionModel.set_gguf_parameters`](#InternVisionModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.InternVisionModel.tensor_force_quant`](#InternVisionModeltensor_force_quant)
    - [`llama.cpp/convert_hf_to_gguf.InternVisionModel.modify_tensors`](#InternVisionModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)

**Methods**

---
#### InternVisionModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.InternVisionModel.set_gguf_parameters}} -->
Sets parameters for the GGUF model configuration based on hyperparameters and global configuration.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and configuration settings.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to initialize base parameters.
    - Retrieves hyperparameters from `self.hparams`.
    - Adds a clip projector type to the GGUF writer.
    - Sets the layer normalization epsilon value in the GGUF writer.
    - Checks the `hidden_act` hyperparameter and configures the GGUF writer accordingly, raising an error for unsupported values.
    - Retrieves the `downsample_ratio` from the global configuration and asserts it is not None.
    - Calculates the scale factor for the projector and adds it to the GGUF writer.
- **Output**: No explicit return value; modifies the state of the `gguf_writer` with the configured parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.InternVisionModel`](#cpp/convert_hf_to_ggufInternVisionModel)  (Base Class)


---
#### InternVisionModel\.tensor\_force\_quant<!-- {{#callable:llama.cpp/convert_hf_to_gguf.InternVisionModel.tensor_force_quant}} -->
Determines the quantization type for a tensor based on its name.
- **Inputs**:
    - `name`: A string representing the original name of the tensor.
    - `new_name`: A string representing the modified name of the tensor used to determine quantization.
    - `bid`: An unused argument that is intended to represent a batch ID.
    - `n_dims`: An unused argument that represents the number of dimensions of the tensor.
- **Control Flow**:
    - The method begins by deleting the unused arguments 'bid', 'name', and 'n_dims'.
    - It checks if the string '.patch_embd.' is present in 'new_name' and returns 'F16' if true.
    - It checks if the string '.position_embd.' is present in 'new_name' and returns 'F32' if true.
    - If neither condition is met, it returns 'False'.
- **Output**: Returns a quantization type (F16 or F32) based on the presence of specific substrings in 'new_name', or 'False' if neither condition is satisfied.
- **See also**: [`llama.cpp/convert_hf_to_gguf.InternVisionModel`](#cpp/convert_hf_to_ggufInternVisionModel)  (Base Class)


---
#### InternVisionModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.InternVisionModel.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on the provided name, returning a list of tuples containing modified tensor names and their corresponding tensors.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the tensor data to be modified.
    - `name`: A string representing the name associated with the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method first checks if the `name` starts with 'vision_model' or 'mlp' to determine if the tensor should be processed.
    - If the `name` starts with 'vision_model', it prefixes it with 'vision_tower.'; if it contains '.ls' or 'position_embedding' and does not end with '.weight', it appends '.weight' to the name.
    - If the `name` contains '.qkv.', it splits the tensor into three parts (query, key, value) based on its dimensions and returns them as separate tuples with modified names.
    - If the `name` does not contain '.qkv.', it returns a single tuple with the modified name and the original tensor.
    - If the `name` does not start with 'vision_model' or 'mlp', it returns an empty list, skipping the tensor.
- **Output**: Returns an iterable of tuples, where each tuple contains a modified tensor name and its corresponding tensor, or an empty list if the tensor is not processed.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.InternVisionModel`](#cpp/convert_hf_to_ggufInternVisionModel)  (Base Class)



---
### WavTokenizerDecModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.WavTokenizerDecModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type for the WavTokenizerDec model.
- **Description**: The `WavTokenizerDecModel` class extends `TextModel` and is designed for handling the WavTokenizer decoding process, including tensor modifications and vocabulary settings.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.WavTokenizerDecModel.modify_tensors`](#WavTokenizerDecModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.WavTokenizerDecModel.set_vocab`](#WavTokenizerDecModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.WavTokenizerDecModel.set_gguf_parameters`](#WavTokenizerDecModelset_gguf_parameters)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### WavTokenizerDecModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.WavTokenizerDecModel.modify_tensors}} -->
The `modify_tensors` method processes a tensor based on its name and returns a tuple of the mapped name and the tensor unless the name matches certain excluded patterns.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name associated with the tensor.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method begins by deleting the `bid` parameter as it is unused.
    - It checks if the `name` ends with specific substrings that indicate it should be skipped.
    - If the name matches any of the excluded patterns, a debug message is logged and an empty list is returned.
    - If the name does not match the excluded patterns, an info message is logged showing the mapped name and the shape of the tensor.
    - Finally, the method returns a list containing a tuple of the mapped name and the tensor.
- **Output**: The output is an iterable of tuples, each containing a string (the mapped name) and a `Tensor` (the original tensor), or an empty list if the name was excluded.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.WavTokenizerDecModel`](#cpp/convert_hf_to_ggufWavTokenizerDecModel)  (Base Class)


---
#### WavTokenizerDecModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.WavTokenizerDecModel.set_vocab}} -->
Sets the vocabulary by invoking a method to handle the vocabulary initialization.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls the private method `_set_vocab_none()` to perform the vocabulary setup.
- **Output**: No output is returned; the method is likely intended to modify the internal state of the object.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_none`](#TextModel_set_vocab_none)
- **See also**: [`llama.cpp/convert_hf_to_gguf.WavTokenizerDecModel`](#cpp/convert_hf_to_ggufWavTokenizerDecModel)  (Base Class)


---
#### WavTokenizerDecModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.WavTokenizerDecModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class parameters are set.
    - Adds vocabulary size to the GGUF writer using the `vocab_size` from `hparams`.
    - Adds the length of features, feed-forward layers, group normalization epsilon, and group normalization groups from `hparams`.
    - Adds positional network embedding length and block count from the `posnet` section of `hparams`.
    - Adds ConvNext embedding length and block count from the `convnext` section of `hparams`.
    - Sets causal attention to False in the GGUF writer.
- **Output**: The method does not return a value but configures the GGUF writer with specific parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.WavTokenizerDecModel`](#cpp/convert_hf_to_ggufWavTokenizerDecModel)  (Base Class)



---
### Qwen2MoeModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.Qwen2MoeModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
    - `_experts`: Holds a list of expert tensors or None.
- **Description**: The `Qwen2MoeModel` class extends `TextModel` and is designed for a mixture of experts architecture, allowing for dynamic handling of expert parameters and tensor modifications during model training and inference.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Qwen2MoeModel.set_gguf_parameters`](#Qwen2MoeModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Qwen2MoeModel.modify_tensors`](#Qwen2MoeModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.Qwen2MoeModel.prepare_tensors`](#Qwen2MoeModelprepare_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### Qwen2MoeModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2MoeModel.set_gguf_parameters}} -->
Sets parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to initialize base parameters.
    - Checks if `num_experts` is defined in `hparams` and adds it to the GGUF writer if present.
    - Checks if `moe_intermediate_size` is defined in `hparams`, adds it to the GGUF writer, and logs the value.
    - Checks if `shared_expert_intermediate_size` is defined in `hparams`, adds it to the GGUF writer, and logs the value.
    - Retrieves the `rope_scaling` configuration from `hparams` and checks if it is set to use 'yarn' with a defined 'factor'.
    - If the conditions for `rope_scaling` are met, adds the scaling type, factor, and original context length to the GGUF writer.
- **Output**: No explicit return value; modifies the state of the GGUF writer with the provided parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2MoeModel`](#cpp/convert_hf_to_ggufQwen2MoeModel)  (Base Class)


---
#### Qwen2MoeModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2MoeModel.modify_tensors}} -->
The `modify_tensors` method processes and merges tensor data for experts in a model, returning the modified tensors.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name associated with the tensor data.
    - `bid`: An optional integer representing the block ID, which is required when processing expert tensors.
- **Control Flow**:
    - Checks if the `name` contains the substring 'experts' to determine if expert processing is needed.
    - Asserts that `bid` is not None when processing experts.
    - Initializes the `_experts` list if it is None, creating a dictionary for each block.
    - Stores the `data_torch` tensor in the `_experts` dictionary for the specified `bid` and `name`.
    - If the number of stored expert tensors reaches three times the number of experts, it merges them into a single 3D tensor.
    - Iterates over the projection names ('down_proj', 'gate_proj', 'up_proj') to collect and stack the expert tensors.
    - Maps the merged tensor name using [`map_tensor_name`](#ModelBasemap_tensor_name) and appends the result to the output list.
    - Returns the list of modified tensors or an empty list if conditions are not met.
- **Output**: Returns an iterable of tuples, each containing a string (the mapped tensor name) and a `Tensor` (the modified tensor), or an empty list if no modifications were made.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2MoeModel`](#cpp/convert_hf_to_ggufQwen2MoeModel)  (Base Class)


---
#### Qwen2MoeModel\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Qwen2MoeModel.prepare_tensors}} -->
Validates that all expert tensors have been processed and raises an error if any remain unprocessed.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls the [`prepare_tensors`](#ModelBaseprepare_tensors) method of the parent class to perform any necessary setup.
    - Checks if the `_experts` attribute is not `None`.
    - Flattens the list of expert dictionaries into a list of keys (expert names).
    - If there are any unprocessed expert names, raises a `ValueError` with the list of those names.
- **Output**: Raises a `ValueError` if there are unprocessed expert tensors; otherwise, the method completes without returning a value.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Qwen2MoeModel`](#cpp/convert_hf_to_ggufQwen2MoeModel)  (Base Class)



---
### Qwen3Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.Qwen3Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as QWEN3.
- **Description**: The `Qwen3Model` class extends the `Qwen2Model` and registers itself with the `ModelBase` under the identifier 'Qwen3ForCausalLM', defining its architecture as QWEN3.
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.Qwen2Model`](#cpp/convert_hf_to_ggufQwen2Model)


---
### Qwen3MoeModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.Qwen3MoeModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the model architecture for the `Qwen3MoeModel` class.
- **Description**: The `Qwen3MoeModel` class extends the `Qwen2MoeModel` and registers itself with a specific model architecture for causal language modeling.
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.Qwen2MoeModel`](#cpp/convert_hf_to_ggufQwen2MoeModel)


---
### GPT2Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.GPT2Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type for the GPT-2 model.
- **Description**: The `GPT2Model` class extends `TextModel` and is designed to represent a GPT-2 language model, providing functionality to set parameters for the model and modify tensor data during processing.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.GPT2Model.set_gguf_parameters`](#GPT2Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.GPT2Model.modify_tensors`](#GPT2Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### GPT2Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.GPT2Model.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on model hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `GPT2Model` class, which contains model hyperparameters and a GGUF writer.
- **Control Flow**:
    - The method accesses the `hparams` dictionary to retrieve model hyperparameters such as number of layers, context length, embedding length, etc.
    - Each hyperparameter is passed to the corresponding method of the `gguf_writer` to set the respective parameter.
- **Output**: This method does not return a value; it modifies the state of the `gguf_writer` by setting various parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.GPT2Model`](#cpp/convert_hf_to_ggufGPT2Model)  (Base Class)


---
#### GPT2Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.GPT2Model.modify_tensors}} -->
The `modify_tensors` method processes a tensor based on its name and returns a list of tuples containing the modified tensor and its new name.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name associated with the tensor.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method begins by deleting the unused `bid` parameter.
    - It initializes an empty list `tensors` to store the results.
    - If the `name` ends with specific suffixes related to attention biases, it returns the empty list immediately.
    - If the `name` ends with certain weight suffixes, it transposes the `data_torch` tensor.
    - The method then maps the original `name` to a new name using `self.map_tensor_name(name)`.
    - Finally, it appends a tuple of the new name and the (possibly modified) tensor to the `tensors` list and returns it.
- **Output**: Returns an iterable of tuples, each containing a string (the new tensor name) and a `Tensor` (the modified tensor).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.GPT2Model`](#cpp/convert_hf_to_ggufGPT2Model)  (Base Class)



---
### Phi2Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.Phi2Model}} -->
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as PHI2.
- **Description**: `Phi2Model` is a specialized class that extends `TextModel` and is designed to configure and manage parameters specific to the PHI2 architecture for causal language modeling.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Phi2Model.set_gguf_parameters`](#Phi2Modelset_gguf_parameters)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### Phi2Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Phi2Model.set_gguf_parameters}} -->
The `set_gguf_parameters` method configures various parameters for the GGUF model by retrieving hyperparameters and passing them to a writer object.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `Phi2Model` class, which contains methods and attributes necessary for setting GGUF parameters.
- **Control Flow**:
    - The method retrieves the number of hidden layers and assigns it to `block_count`.
    - It retrieves the partial rotary factor and assigns it to `rot_pct`.
    - It retrieves the hidden size and assigns it to `n_embd`.
    - It retrieves the number of attention heads and assigns it to `n_head`.
    - The method then uses the `gguf_writer` to add various parameters including context length, embedding length, feed-forward length, block count, head count, layer normalization epsilon, rope dimension count, file type, and a flag for adding a beginning-of-sequence token.
- **Output**: The method does not return a value; instead, it configures the GGUF writer with the specified model parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Phi2Model`](#cpp/convert_hf_to_ggufPhi2Model)  (Base Class)



---
### Phi3MiniModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.Phi3MiniModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `Phi3MiniModel` class extends `TextModel` and is designed for handling the Phi-3 architecture for causal language modeling, including methods for setting vocabulary and model parameters, as well as generating additional tensors for model training.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Phi3MiniModel.set_vocab`](#Phi3MiniModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.Phi3MiniModel.set_gguf_parameters`](#Phi3MiniModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Phi3MiniModel.generate_extra_tensors`](#Phi3MiniModelgenerate_extra_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### Phi3MiniModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Phi3MiniModel.set_vocab}} -->
The `set_vocab` method initializes the vocabulary for a language model by loading tokenizer configurations and processing tokens from specified files.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class `Phi3MiniModel`, which contains model parameters and directory paths for loading tokenizer files.
- **Control Flow**:
    - Checks for the existence of a tokenizer configuration file and loads it to determine the tokenizer class.
    - If the tokenizer class is 'GPT2Tokenizer', it calls a specific method to set the vocabulary for GPT-2.
    - Loads a SentencePiece tokenizer model from a specified file and retrieves the vocabulary size.
    - Initializes lists for tokens, scores, and token types based on the vocabulary size.
    - Iterates through the vocabulary to populate tokens, scores, and types based on the tokenizer's properties.
    - Checks for an 'added_tokens.json' file to include user-defined tokens and their properties.
    - Updates tokens and scores based on additional configurations from 'tokenizer_config.json' and 'tokenizer.json'.
    - Adds the processed vocabulary to a GGUF writer for further use in the model.
- **Output**: The method does not return a value but populates the model's vocabulary, scores, and types, which are then used for tokenization in the language model.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Phi3MiniModel`](#cpp/convert_hf_to_ggufPhi3MiniModel)  (Base Class)


---
#### Phi3MiniModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Phi3MiniModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves various hyperparameters using the [`find_hparam`](#ModelBasefind_hparam) method.
    - Calculates the number of rope dimensions based on the rotary factor and embedding size.
    - Adds context length, embedding length, and other parameters to the GGUF writer.
    - Handles the sliding window parameter, defaulting to zero if not set.
- **Output**: The method does not return a value; it modifies the state of the GGUF writer with the set parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Phi3MiniModel`](#cpp/convert_hf_to_ggufPhi3MiniModel)  (Base Class)


---
#### Phi3MiniModel\.generate\_extra\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Phi3MiniModel.generate_extra_tensors}} -->
Generates additional tensors for rope scaling factors based on model hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class containing the method, which holds model hyperparameters and methods for tensor generation.
- **Control Flow**:
    - Retrieve various hyperparameters related to the model architecture, such as embedding size, number of attention heads, and position embeddings.
    - Check if rope scaling parameters are provided; if not, exit the method.
    - Calculate the scaling factor based on maximum position embeddings.
    - Determine the type of rope scaling and compute the attention factor based on the scaling type.
    - Validate the presence and length of long and short factors for rope scaling.
    - Yield the formatted tensor names and their corresponding tensor values for long and short factors.
- **Output**: Yields tuples containing the names and tensor values for long and short rope scaling factors.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Phi3MiniModel`](#cpp/convert_hf_to_ggufPhi3MiniModel)  (Base Class)



---
### PhiMoeModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.PhiMoeModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
    - `_experts`: Holds a list of dictionaries containing expert tensors or None.
- **Description**: The `PhiMoeModel` class extends `Phi3MiniModel` and is designed for a mixture of experts model architecture, managing expert tensors and their processing for causal language modeling.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.PhiMoeModel.set_gguf_parameters`](#PhiMoeModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.PhiMoeModel.modify_tensors`](#PhiMoeModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.PhiMoeModel.prepare_tensors`](#PhiMoeModelprepare_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.Phi3MiniModel`](#cpp/convert_hf_to_ggufPhi3MiniModel)

**Methods**

---
#### PhiMoeModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.PhiMoeModel.set_gguf_parameters}} -->
Sets parameters related to the GGUF model by invoking the parent method and updating expert counts.
- **Inputs**: None
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class parameters are set.
    - Adds the count of experts used per token to the `gguf_writer` using the `num_experts_per_tok` parameter from `hparams`.
    - Adds the total number of local experts to the `gguf_writer` using the `num_local_experts` parameter from `hparams`.
- **Output**: The method does not return a value; it updates the state of the `gguf_writer` with expert-related parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.PhiMoeModel`](#cpp/convert_hf_to_ggufPhiMoeModel)  (Base Class)


---
#### PhiMoeModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.PhiMoeModel.modify_tensors}} -->
The `modify_tensors` method processes and merges tensor data for experts in a model, returning a list of modified tensor names and their corresponding data.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name associated with the tensor data.
    - `bid`: An optional integer representing the block ID for the experts.
- **Control Flow**:
    - Checks if the `name` contains 'block_sparse_moe.experts' to determine if expert processing is needed.
    - Asserts that `bid` is not None if expert processing is required.
    - Initializes `_experts` if it is None, creating a list of dictionaries for each block.
    - Stores the `data_torch` in the appropriate expert dictionary based on `bid` and `name`.
    - If the number of stored experts reaches three times the number of local experts, it merges the tensors into a single 3D tensor.
    - Iterates over the weight names 'w1', 'w2', and 'w3', stacking the tensors from each expert and creating a new name for the merged tensor.
    - Returns a list of tuples containing the new tensor names and their corresponding stacked tensors.
    - If the expert count is insufficient, returns an empty list.
    - If the `name` does not match the expert pattern, returns a tuple with the mapped name and the original tensor.
- **Output**: Returns an iterable of tuples, each containing a modified tensor name and its corresponding tensor data, or an empty list if conditions for merging are not met.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.PhiMoeModel`](#cpp/convert_hf_to_ggufPhiMoeModel)  (Base Class)


---
#### PhiMoeModel\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.PhiMoeModel.prepare_tensors}} -->
The [`prepare_tensors`](#ModelBaseprepare_tensors) method checks for unprocessed expert tensors and raises an error if any are found.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class, which contains the `_experts` attribute.
- **Control Flow**:
    - Calls the [`prepare_tensors`](#ModelBaseprepare_tensors) method of the parent class to perform any necessary setup.
    - Checks if the `_experts` attribute is not `None`.
    - Flattens the list of dictionaries in `_experts` to extract the keys (expert names).
    - If there are any unprocessed experts, raises a `ValueError` with a message listing the unprocessed expert names.
- **Output**: Raises a `ValueError` if there are unprocessed expert tensors; otherwise, it completes without returning a value.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.PhiMoeModel`](#cpp/convert_hf_to_ggufPhiMoeModel)  (Base Class)



---
### PlamoModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.PlamoModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `PlamoModel` class extends `TextModel` and is designed for causal language modeling, incorporating methods for setting vocabulary and GGUF parameters, as well as modifying tensor weights for attention mechanisms.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.PlamoModel.set_vocab`](#PlamoModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.PlamoModel.set_gguf_parameters`](#PlamoModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.PlamoModel.shuffle_attn_q_weight`](#PlamoModelshuffle_attn_q_weight)
    - [`llama.cpp/convert_hf_to_gguf.PlamoModel.shuffle_attn_output_weight`](#PlamoModelshuffle_attn_output_weight)
    - [`llama.cpp/convert_hf_to_gguf.PlamoModel.modify_tensors`](#PlamoModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### PlamoModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.PlamoModel.set_vocab}} -->
The `set_vocab` method initializes the vocabulary for the model by invoking the [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece) method.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - The method directly calls the [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece) method without any conditions or loops.
- **Output**: The method does not return any value; it performs an internal operation to set the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.PlamoModel`](#cpp/convert_hf_to_ggufPlamoModel)  (Base Class)


---
#### PlamoModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.PlamoModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on model hyperparameters.
- **Inputs**: None
- **Control Flow**:
    - Retrieves hyperparameters from the model's `hparams` attribute.
    - Extracts the number of hidden layers from the hyperparameters.
    - Calls multiple methods on the `gguf_writer` to set various model parameters such as context length, embedding length, feed-forward length, block count, head count, and layer normalization epsilon.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.PlamoModel`](#cpp/convert_hf_to_ggufPlamoModel)  (Base Class)


---
#### PlamoModel\.shuffle\_attn\_q\_weight<!-- {{#callable:llama.cpp/convert_hf_to_gguf.PlamoModel.shuffle_attn_q_weight}} -->
Shuffles the attention query weight tensor by reshaping and permuting its dimensions.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A PyTorch tensor of shape (5120, 5120) representing the attention query weights.
- **Control Flow**:
    - Asserts that the input tensor `data_torch` has the correct shape of (5120, 5120).
    - Reshapes `data_torch` into a tensor of shape (8, 5, 128, 5120).
    - Permutes the dimensions of the reshaped tensor to rearrange its axes.
    - Reshapes the permuted tensor back to the original shape of (5120, 5120).
- **Output**: Returns the reshaped and permuted tensor of attention query weights with the shape (5120, 5120).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
- **See also**: [`llama.cpp/convert_hf_to_gguf.PlamoModel`](#cpp/convert_hf_to_ggufPlamoModel)  (Base Class)


---
#### PlamoModel\.shuffle\_attn\_output\_weight<!-- {{#callable:llama.cpp/convert_hf_to_gguf.PlamoModel.shuffle_attn_output_weight}} -->
The `shuffle_attn_output_weight` method reshapes and permutes a given tensor to shuffle its attention output weights.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A PyTorch tensor of shape (5120, 5120) representing the attention output weights.
- **Control Flow**:
    - The method asserts that the input tensor `data_torch` has the correct shape of (5120, 5120).
    - It reshapes `data_torch` into a new shape of (5120, 8, 5, 128).
    - The tensor is then permuted to rearrange its dimensions according to the specified order (0, 2, 1, 3).
    - Finally, the permuted tensor is reshaped back to (5120, 5120) before being returned.
- **Output**: The method returns a reshaped and permuted tensor of the same shape (5120, 5120) as the input, effectively shuffling the attention output weights.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
- **See also**: [`llama.cpp/convert_hf_to_gguf.PlamoModel`](#cpp/convert_hf_to_ggufPlamoModel)  (Base Class)


---
#### PlamoModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.PlamoModel.modify_tensors}} -->
Modifies a tensor based on its name by potentially shuffling its weights for specific attention mechanisms.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the data to be modified.
    - `name`: A string representing the name associated with the tensor.
    - `bid`: An optional integer that is unused in the method.
- **Control Flow**:
    - The method begins by deleting the unused `bid` parameter.
    - It maps the input `name` to a new name using `self.map_tensor_name(name)`.
    - If the new name ends with 'attn_q.weight', it shuffles the tensor using `self.shuffle_attn_q_weight(data_torch)`.
    - If the new name ends with 'attn_output.weight', it shuffles the tensor using `self.shuffle_attn_output_weight(data_torch)`.
    - Finally, it returns a list containing a tuple of the new name and the modified tensor.
- **Output**: Returns an iterable containing a tuple with the modified tensor's new name and the modified tensor itself.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.PlamoModel.shuffle_attn_q_weight`](#PlamoModelshuffle_attn_q_weight)
    - [`llama.cpp/convert_hf_to_gguf.PlamoModel.shuffle_attn_output_weight`](#PlamoModelshuffle_attn_output_weight)
- **See also**: [`llama.cpp/convert_hf_to_gguf.PlamoModel`](#cpp/convert_hf_to_ggufPlamoModel)  (Base Class)



---
### CodeShellModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.CodeShellModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
    - `_has_tok_embd`: Indicates whether the token embedding has been encountered.
- **Description**: The `CodeShellModel` class extends `TextModel` and is designed for managing and configuring a code shell model architecture, specifically for causal language modeling, with methods to set parameters and modify tensor data.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.CodeShellModel.set_gguf_parameters`](#CodeShellModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.CodeShellModel.modify_tensors`](#CodeShellModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### CodeShellModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.CodeShellModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `CodeShellModel` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves the number of layers from the hyperparameters and stores it in `block_count`.
    - Calls multiple methods on `self.gguf_writer` to set various parameters such as context length, embedding length, feed-forward length, block count, head count, and others using values from `self.hparams`.
- **Output**: The method does not return a value; it configures the GGUF writer with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.CodeShellModel`](#cpp/convert_hf_to_ggufCodeShellModel)  (Base Class)


---
#### CodeShellModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.CodeShellModel.modify_tensors}} -->
The `modify_tensors` method processes a tensor and its associated name, potentially modifying internal state based on the tensor's identity.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the data to be modified.
    - `name`: A string representing the name associated with the tensor.
    - `bid`: An optional integer that is unused in the method.
- **Control Flow**:
    - The method begins by deleting the unused `bid` parameter.
    - It formats the output tensor name and token embedding name using the [`format_tensor_name`](#ModelBaseformat_tensor_name) method.
    - It maps the provided tensor name to a new name using the [`map_tensor_name`](#ModelBasemap_tensor_name) method.
    - If the token embedding has not been seen and the new name matches the output tensor name, it checks if the tensor names contain 'transformer.wte.weight' and logs a debug message if it does, removing it from the tensor names.
    - If the new name matches the token embedding name, it sets the `_has_tok_embd` flag to True.
- **Output**: Returns an iterable containing a tuple of the new tensor name and the original tensor data.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.CodeShellModel`](#cpp/convert_hf_to_ggufCodeShellModel)  (Base Class)



---
### InternLM2Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.InternLM2Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `InternLM2Model` class extends `TextModel` and is designed for causal language modeling, incorporating methods for setting vocabulary and model parameters, as well as modifying tensors for attention mechanisms.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.InternLM2Model.set_vocab`](#InternLM2Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.InternLM2Model.set_gguf_parameters`](#InternLM2Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.InternLM2Model.modify_tensors`](#InternLM2Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### InternLM2Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.InternLM2Model.set_vocab}} -->
The `set_vocab` method initializes the vocabulary for a tokenizer model by loading token data from various files and processing it to handle special cases.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class `InternLM2Model`, which contains model parameters and methods for processing the vocabulary.
- **Control Flow**:
    - Check if the tokenizer model file exists; if not, log an error and exit.
    - Load the tokenizer model and extract vocabulary size.
    - Iterate through the vocabulary, processing each token to handle special cases like the null character.
    - Load additional tokens from `added_tokens.json` and `tokenizer_config.json`, updating the token lists accordingly.
    - Add the processed tokens, scores, and types to the `gguf_writer` for further use.
    - Handle special end-of-sequence tokens for chat models, replacing them as necessary.
- **Output**: The method does not return a value but updates the internal state of the model with the processed vocabulary and writes it to the `gguf_writer`.
- **See also**: [`llama.cpp/convert_hf_to_gguf.InternLM2Model`](#cpp/convert_hf_to_ggufInternLM2Model)  (Base Class)


---
#### InternLM2Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.InternLM2Model.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on hyperparameters.
- **Inputs**: None
- **Control Flow**:
    - The method retrieves various hyperparameters from `self.hparams` and uses them to configure the `gguf_writer`.
    - It checks if the `rope_scaling` parameter is set to 'linear' and if a 'factor' exists, then it adds the corresponding scaling type and factor to the `gguf_writer`.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` by adding parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.InternLM2Model`](#cpp/convert_hf_to_ggufInternLM2Model)  (Base Class)


---
#### InternLM2Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.InternLM2Model.modify_tensors}} -->
The `modify_tensors` method processes and reshapes tensor data based on specified conditions and returns a list of formatted tensor names and their corresponding tensors.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name of the tensor.
    - `bid`: An optional integer that specifies the block ID, or None.
- **Control Flow**:
    - The method retrieves hyperparameters related to attention heads and embedding size from `self.hparams`.
    - It modifies the `name` by removing the prefix 'language_model.' and checks if it starts with 'mlp' or 'vision_model' to skip visual tensors.
    - If `bid` is not None and the `name` matches a specific pattern, it reshapes the `data_torch` tensor into query, key, and value tensors.
    - The query and key tensors are further permuted to match the required dimensions for the model.
    - Finally, it returns a list of tuples containing formatted tensor names and their corresponding tensors, or a single tuple with the mapped name and original tensor if conditions are not met.
- **Output**: An iterable of tuples, where each tuple contains a string (formatted tensor name) and a `Tensor` (modified tensor data).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.InternLM2Model`](#cpp/convert_hf_to_ggufInternLM2Model)  (Base Class)



---
### InternLM3Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.InternLM3Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture of the model as LLAMA.
- **Description**: The `InternLM3Model` class extends `TextModel` and is designed for causal language modeling, providing methods to set vocabulary and GGUF parameters, as well as modify tensors for model training.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.InternLM3Model.set_vocab`](#InternLM3Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.InternLM3Model.set_gguf_parameters`](#InternLM3Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.InternLM3Model.modify_tensors`](#InternLM3Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### InternLM3Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.InternLM3Model.set_vocab}} -->
The `set_vocab` method initializes the vocabulary for a language model by creating a vocabulary from sentence pieces and configuring the tokenizer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `InternLM3Model` class, which contains the model architecture and methods for setting up the vocabulary.
- **Control Flow**:
    - Calls the [`_create_vocab_sentencepiece`](#TextModel_create_vocab_sentencepiece) method to generate tokens, scores, and token types.
    - Adds the tokenizer model and its configurations to the `gguf_writer`.
    - Checks for the existence of a tokenizer configuration file and reads it if present.
    - If the configuration specifies an 'add_prefix_space', it adds this to the `gguf_writer`.
    - Processes any special tokens defined in the configuration and updates the special vocabulary accordingly.
    - Finally, adds the special vocabulary to the `gguf_writer`.
- **Output**: The method does not return a value but configures the tokenizer and special vocabulary for the model, preparing it for use in language processing tasks.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._create_vocab_sentencepiece`](#TextModel_create_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.InternLM3Model`](#cpp/convert_hf_to_ggufInternLM3Model)  (Base Class)


---
#### InternLM3Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.InternLM3Model.set_gguf_parameters}} -->
Sets parameters for the GGUF model including vocabulary size, rope dimensions, and scaling.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure base parameters are set.
    - Retrieves hyperparameters from `self.hparams`.
    - Adds the vocabulary size to the GGUF writer using the retrieved vocabulary size.
    - Checks if 'head_dim' is present in hyperparameters to determine the rope dimension; if not, calculates it based on hidden size and number of attention heads.
    - Adds the calculated rope dimension count to the GGUF writer.
    - Retrieves the rope scaling configuration and checks if the type is 'linear' and if a scaling factor is provided.
    - If conditions are met, adds the rope scaling type and factor to the GGUF writer.
- **Output**: No explicit return value; modifies the state of the GGUF writer with the set parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.InternLM3Model`](#cpp/convert_hf_to_ggufInternLM3Model)  (Base Class)


---
#### InternLM3Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.InternLM3Model.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on the provided name and model parameters.
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name of the tensor, which influences how the tensor is modified.
    - `bid`: An optional integer that may represent a batch ID, though it is not used in the method.
- **Control Flow**:
    - The method retrieves the number of attention heads and key-value heads from the model's hyperparameters.
    - It modifies the `name` by removing the prefix 'language_model.' for internal processing.
    - If the `name` starts with 'mlp' or 'vision_model', the method returns an empty list, skipping further processing.
    - If the `name` ends with 'q_proj.weight' or 'q_proj.bias', the method permutes the `data_torch` tensor using the number of attention heads.
    - If the `name` ends with 'k_proj.weight' or 'k_proj.bias', the method permutes the `data_torch` tensor using the number of attention heads and key-value heads.
    - Finally, the method returns a list containing a tuple of the modified tensor name and the modified tensor.
- **Output**: The method outputs an iterable of tuples, each containing a string (the modified tensor name) and a `Tensor` (the possibly modified tensor data).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.InternLM3Model`](#cpp/convert_hf_to_ggufInternLM3Model)  (Base Class)



---
### BertModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.BertModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as BERT.
    - `vocab_size`: Holds the size of the vocabulary used by the model.
    - `cls_out_labels`: Stores the mapping of class output labels for classification tasks.
- **Description**: The `BertModel` class extends `TextModel` and is designed to implement the BERT architecture for various NLP tasks, including masked language modeling and sequence classification, while managing vocabulary and model parameters effectively.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.BertModel.__init__`](#BertModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.BertModel.set_gguf_parameters`](#BertModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.BertModel.set_vocab`](#BertModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.BertModel.modify_tensors`](#BertModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.BertModel._xlmroberta_tokenizer_init`](#BertModel_xlmroberta_tokenizer_init)
    - [`llama.cpp/convert_hf_to_gguf.BertModel._xlmroberta_set_vocab`](#BertModel_xlmroberta_set_vocab)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### BertModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BertModel.set_gguf_parameters}} -->
Sets parameters for the GGUF writer, including causal attention and classifier output labels.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains the method.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class parameters are set.
    - Disables causal attention by calling `add_causal_attention` with `False`.
    - Attempts to set the pooling type by calling [`_try_set_pooling_type`](#TextModel_try_set_pooling_type).
    - Checks if `cls_out_labels` is defined; if so, adds the sorted classifier output labels to the GGUF writer.
- **Output**: This method does not return a value; it modifies the state of the GGUF writer by adding parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._try_set_pooling_type`](#TextModel_try_set_pooling_type)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BertModel`](#cpp/convert_hf_to_ggufBertModel)  (Base Class)


---
#### BertModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BertModel.set_vocab}} -->
The `set_vocab` method initializes the vocabulary for a BERT model by processing tokens and their types, converting them to a specific format, and adding them to a GGUF writer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `BertModel` class, which contains model parameters and methods.
- **Control Flow**:
    - Calls [`get_vocab_base`](#TextModelget_vocab_base) to retrieve the base vocabulary, token types, and pre-tokenization information.
    - Sets the vocabulary size based on the number of tokens retrieved.
    - Adds the token type count to the GGUF writer using a parameter from the model's hyperparameters.
    - Defines a nested function `phantom` to transform tokens into a phantom space format.
    - Maps the original tokens to their phantom representations using the `phantom` function.
    - Adds the tokenizer model, pre-tokenization information, token list, and token types to the GGUF writer.
    - Creates a `SpecialVocab` instance to handle special tokens and adds it to the GGUF writer.
- **Output**: The method does not return a value but updates the GGUF writer with the processed vocabulary and special tokens.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel.get_vocab_base`](#TextModelget_vocab_base)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BertModel`](#cpp/convert_hf_to_ggufBertModel)  (Base Class)


---
#### BertModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BertModel.modify_tensors}} -->
The `modify_tensors` method modifies tensor names based on specific rules and returns a tuple of the modified name and the tensor.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name of the tensor that may need modification.
    - `bid`: An optional integer that is unused in the method.
- **Control Flow**:
    - The method first deletes the unused `bid` argument.
    - It checks if the `name` starts with 'bert.' and modifies it by removing this prefix.
    - If the `name` ends with '.gamma', it changes it to end with '.weight'.
    - If the `name` ends with '.beta', it changes it to end with '.bias'.
    - The method checks if the `name` corresponds to certain tensor names that are not needed and returns an empty list for those cases.
    - If `self.cls_out_labels` is set, it modifies the `name` for the classifier's weight and bias accordingly.
    - Finally, it returns a list containing a tuple of the modified name and the original tensor.
- **Output**: The output is an iterable of tuples, each containing a modified tensor name and the corresponding `data_torch` tensor.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BertModel`](#cpp/convert_hf_to_ggufBertModel)  (Base Class)


---
#### BertModel\.\_xlmroberta\_tokenizer\_init<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BertModel._xlmroberta_tokenizer_init}} -->
Initializes the tokenizer for the XLM-RoBERTa model by adjusting the position embeddings based on the padding token ID.
- **Inputs**: None
- **Control Flow**:
    - Checks if the 'pad_token_id' is present in the model's hyperparameters.
    - If 'pad_token_id' is found, it calculates the position offset and adjusts 'max_position_embeddings' accordingly.
    - If 'pad_token_id' is not found, it sets the position offset to None.
- **Output**: The method does not return any value; it modifies the internal state of the object by setting the position offset and potentially adjusting the maximum position embeddings.
- **See also**: [`llama.cpp/convert_hf_to_gguf.BertModel`](#cpp/convert_hf_to_ggufBertModel)  (Base Class)



---
### DistilBertModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.DistilBertModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the model architecture as BERT.
- **Description**: The `DistilBertModel` class extends the `BertModel` class and is registered with the `ModelBase` for various tasks, providing specific configurations for the DistilBERT architecture, including methods to set parameters and modify tensor data.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.DistilBertModel.set_gguf_parameters`](#DistilBertModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.DistilBertModel.modify_tensors`](#DistilBertModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.BertModel`](#cpp/convert_hf_to_ggufBertModel)

**Methods**

---
#### DistilBertModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DistilBertModel.set_gguf_parameters}} -->
Sets specific parameters for the GGUF writer in the `DistilBertModel` class.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls `add_layer_norm_eps` method on `gguf_writer` with a fixed value of 1e-12.
    - Logs the layer norm epsilon value using the logger.
    - Invokes the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any additional setup is performed.
- **Output**: No explicit output; the method modifies the state of the `gguf_writer` and logs information.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DistilBertModel`](#cpp/convert_hf_to_ggufDistilBertModel)  (Base Class)


---
#### DistilBertModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DistilBertModel.modify_tensors}} -->
Modifies tensor data based on the provided name and bid, filtering out specific layers.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name of the layer or component to be modified.
    - `bid`: An optional integer that may represent a batch ID or similar identifier.
- **Control Flow**:
    - Checks if the `name` starts with 'distilbert.' and modifies it by removing this prefix.
    - Checks if the modified `name` starts with 'vocab_' and returns an empty list if true, indicating that these layers should not be processed.
    - If the `name` does not match the above conditions, it calls the parent class's [`modify_tensors`](#ModelBasemodify_tensors) method with the original arguments.
- **Output**: Returns an iterable of tuples, each containing a string and a `Tensor`, representing the modified tensor data.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.modify_tensors`](#ModelBasemodify_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DistilBertModel`](#cpp/convert_hf_to_ggufDistilBertModel)  (Base Class)



---
### RobertaModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.RobertaModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the model architecture as BERT.
    - `_position_offset`: Holds the position offset for embeddings based on the pad token ID.
- **Description**: The `RobertaModel` class extends `BertModel` to implement a Roberta model specifically for sequence classification tasks, incorporating adjustments for position embeddings and vocabulary settings based on the model's hyperparameters.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.RobertaModel.__init__`](#RobertaModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.RobertaModel.set_vocab`](#RobertaModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.RobertaModel.modify_tensors`](#RobertaModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.BertModel`](#cpp/convert_hf_to_ggufBertModel)


---
### NomicBertModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.NomicBertModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
    - `is_moe`: Indicates whether the model uses a mixture of experts.
    - `_tokenizer_is_xlmroberta`: Determines if the tokenizer is of type XLM-RoBERTa.
- **Description**: The `NomicBertModel` class extends `BertModel` to implement a specialized BERT architecture that can utilize a mixture of experts, with additional configurations for tokenizer handling and model parameters based on the provided hyperparameters.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.NomicBertModel.__init__`](#NomicBertModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.NomicBertModel.set_vocab`](#NomicBertModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.NomicBertModel.modify_tensors`](#NomicBertModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.NomicBertModel.set_gguf_parameters`](#NomicBertModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.NomicBertModel._is_tokenizer_xlmroberta`](#NomicBertModel_is_tokenizer_xlmroberta)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.BertModel`](#cpp/convert_hf_to_ggufBertModel)

**Methods**

---
#### NomicBertModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.NomicBertModel.__init__}} -->
Initializes a NomicBertModel instance with specified model parameters and configurations.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `dir_model`: A `Path` object representing the directory of the model.
    - `ftype`: An instance of `gguf.LlamaFileType` indicating the file type of the model.
    - `fname_out`: A `Path` object specifying the output filename.
    - `**kwargs`: Additional keyword arguments that may include hyperparameters.
- **Control Flow**:
    - Checks if 'hparams' is provided in kwargs; if not, loads it from the model directory.
    - Determines if the model is a mixture of experts (MoE) based on 'hparams'.
    - Initializes the model architecture based on whether it is MoE or not.
    - Calls the parent class's [`__init__`](#ModelBase__init__) method to initialize inherited properties.
    - Checks if the tokenizer is of type XLM-RoBERTa and initializes it if true.
    - Validates and adjusts the number of positions based on the trained model parameters.
    - Asserts various conditions on hyperparameters to ensure they meet expected values.
- **Output**: No explicit output; the method initializes the instance and sets up its internal state.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.load_hparams`](#ModelBaseload_hparams)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
    - [`llama.cpp/convert_hf_to_gguf.NomicBertModel._is_tokenizer_xlmroberta`](#NomicBertModel_is_tokenizer_xlmroberta)
    - [`llama.cpp/convert_hf_to_gguf.BertModel._xlmroberta_tokenizer_init`](#BertModel_xlmroberta_tokenizer_init)
- **See also**: [`llama.cpp/convert_hf_to_gguf.NomicBertModel`](#cpp/convert_hf_to_ggufNomicBertModel)  (Base Class)


---
#### NomicBertModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.NomicBertModel.set_vocab}} -->
Sets the vocabulary for the model based on the tokenizer type.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `NomicBertModel` class.
- **Control Flow**:
    - Checks if the tokenizer is of type XLM-RoBERTa.
    - If true, calls the [`_xlmroberta_set_vocab`](#BertModel_xlmroberta_set_vocab) method to set the vocabulary.
    - If false, calls the parent class's [`set_vocab`](#TextModelset_vocab) method.
- **Output**: Returns None; the method modifies the vocabulary setting internally based on the tokenizer type.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.BertModel._xlmroberta_set_vocab`](#BertModel_xlmroberta_set_vocab)
    - [`llama.cpp/convert_hf_to_gguf.TextModel.set_vocab`](#TextModelset_vocab)
- **See also**: [`llama.cpp/convert_hf_to_gguf.NomicBertModel`](#cpp/convert_hf_to_ggufNomicBertModel)  (Base Class)


---
#### NomicBertModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.NomicBertModel.modify_tensors}} -->
The `modify_tensors` method processes a given tensor based on its name, modifying its shape for specific cases.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `torch.Tensor` that is to be modified based on its name.
    - `name`: A string representing the name associated with the tensor.
    - `bid`: An optional integer that may represent a batch ID, though it is not used in the method.
- **Control Flow**:
    - The method first checks if the `name` contains 'mlp.experts.bias'; if so, it returns an empty list, skipping further processing.
    - If the `name` contains 'mlp.experts.mlp.w1', the tensor is reshaped to a specific view and the name is updated to include '.weight'.
    - If the `name` contains 'mlp.experts.mlp.w2', the tensor is reshaped and transposed, and the name is similarly updated.
    - Finally, the method returns a list containing a tuple of the modified name and tensor.
- **Output**: The method returns an iterable containing a tuple with the modified tensor name and the processed `torch.Tensor`.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.NomicBertModel`](#cpp/convert_hf_to_ggufNomicBertModel)  (Base Class)


---
#### NomicBertModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.NomicBertModel.set_gguf_parameters}} -->
Sets parameters for the GGUF writer based on model hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class, which contains model hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base functionality is executed.
    - Adds the rotary embedding frequency base to the GGUF writer using the `rotary_emb_base` hyperparameter.
    - Checks if the model is a mixture of experts (MoE) and, if so, adds additional parameters related to MoE to the GGUF writer.
- **Output**: The method does not return a value; it modifies the state of the GGUF writer by adding specific parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.NomicBertModel`](#cpp/convert_hf_to_ggufNomicBertModel)  (Base Class)



---
### XLMRobertaModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.XLMRobertaModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the model architecture as BERT.
- **Description**: The `XLMRobertaModel` class extends `BertModel` and is designed for sequence classification tasks, incorporating specific initialization and vocabulary settings for the XLM-RoBERTa model.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.XLMRobertaModel.__init__`](#XLMRobertaModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.XLMRobertaModel.set_vocab`](#XLMRobertaModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.XLMRobertaModel.modify_tensors`](#XLMRobertaModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.BertModel`](#cpp/convert_hf_to_ggufBertModel)

**Methods**

---
#### XLMRobertaModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.XLMRobertaModel.__init__}} -->
Initializes an instance of the `XLMRobertaModel` class, setting up the tokenizer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `*args`: Positional arguments passed to the parent class constructor.
    - `**kwargs`: Keyword arguments passed to the parent class constructor.
- **Control Flow**:
    - Calls the parent class (`BertModel`) constructor with the provided arguments using `super().__init__(*args, **kwargs)`.
    - Invokes the [`_xlmroberta_tokenizer_init`](#BertModel_xlmroberta_tokenizer_init) method to initialize the tokenizer specific to the XLM-Roberta model.
- **Output**: No explicit output; the method initializes the instance and prepares the tokenizer for use.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
    - [`llama.cpp/convert_hf_to_gguf.BertModel._xlmroberta_tokenizer_init`](#BertModel_xlmroberta_tokenizer_init)
- **See also**: [`llama.cpp/convert_hf_to_gguf.XLMRobertaModel`](#cpp/convert_hf_to_ggufXLMRobertaModel)  (Base Class)


---
#### XLMRobertaModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.XLMRobertaModel.set_vocab}} -->
Sets the vocabulary for the XLMRoberta model by invoking a specific internal method.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls the internal method `_xlmroberta_set_vocab()` to perform the vocabulary setting operation.
- **Output**: No output is returned; the method modifies the internal state of the model by setting the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.BertModel._xlmroberta_set_vocab`](#BertModel_xlmroberta_set_vocab)
- **See also**: [`llama.cpp/convert_hf_to_gguf.XLMRobertaModel`](#cpp/convert_hf_to_ggufXLMRobertaModel)  (Base Class)


---
#### XLMRobertaModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.XLMRobertaModel.modify_tensors}} -->
The [`modify_tensors`](#ModelBasemodify_tensors) method modifies tensor data based on specific conditions related to the tensor's name and then calls the parent class's [`modify_tensors`](#ModelBasemodify_tensors) method.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that contains the data to be modified.
    - `name`: A string representing the name of the tensor, which may include a prefix.
    - `bid`: An optional integer that may be used for additional identification or processing.
- **Control Flow**:
    - Checks if the `name` starts with 'roberta.' and removes this prefix if present.
    - If the `name` is 'embeddings.position_embeddings.weight', it checks if `_position_offset` is not None and slices the `data_torch` tensor accordingly.
    - Calls the parent class's [`modify_tensors`](#ModelBasemodify_tensors) method with the potentially modified `data_torch`, `name`, and `bid`.
- **Output**: Returns an iterable of tuples containing the modified tensor name and the corresponding tensor data.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.modify_tensors`](#ModelBasemodify_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.XLMRobertaModel`](#cpp/convert_hf_to_ggufXLMRobertaModel)  (Base Class)



---
### GemmaModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.GemmaModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as GEMMA.
- **Description**: The `GemmaModel` class extends `TextModel` and is designed for causal language modeling, incorporating methods for setting vocabulary and GGUF parameters, as well as modifying tensors during model loading.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.GemmaModel.set_vocab`](#GemmaModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.GemmaModel.set_gguf_parameters`](#GemmaModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.GemmaModel.modify_tensors`](#GemmaModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### GemmaModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.GemmaModel.set_vocab}} -->
The `set_vocab` method initializes special vocabulary tokens for the Gemma model and integrates them into the GGUF writer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `GemmaModel` class, which contains model-specific attributes and methods.
- **Control Flow**:
    - Calls the [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece) method to set up the sentencepiece vocabulary.
    - Creates an instance of `gguf.SpecialVocab` with specified token types and the model directory.
    - Sets special tokens with predefined IDs for 'prefix', 'suffix', 'middle', 'fsep', and 'eot'.
    - Assigns `None` to `chat_template` to avoid duplication.
    - Adds the special vocabulary to the GGUF writer.
    - Disables the addition of space prefix in the GGUF writer.
- **Output**: The method does not return a value but modifies the internal state of the `GemmaModel` instance by setting up the vocabulary and updating the GGUF writer.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.GemmaModel`](#cpp/convert_hf_to_ggufGemmaModel)  (Base Class)


---
#### GemmaModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.GemmaModel.set_gguf_parameters}} -->
The `set_gguf_parameters` method configures various hyperparameters for the GGUF writer based on the model's hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `GemmaModel` class, which contains model hyperparameters and a GGUF writer.
- **Control Flow**:
    - The method retrieves hyperparameters from the `self.hparams` dictionary.
    - It extracts the number of hidden layers to set the block count.
    - The method then calls various methods on `self.gguf_writer` to set parameters such as context length, embedding length, block count, feed-forward length, head count, and others based on the hyperparameters.
- **Output**: The method does not return any value; it modifies the state of the `gguf_writer` by adding various hyperparameter settings.
- **See also**: [`llama.cpp/convert_hf_to_gguf.GemmaModel`](#cpp/convert_hf_to_ggufGemmaModel)  (Base Class)


---
#### GemmaModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.GemmaModel.modify_tensors}} -->
The `modify_tensors` method processes a tensor based on its name, modifying it if necessary, and returns a mapped name along with the modified tensor.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method first deletes the unused `bid` parameter.
    - If the `name` is 'lm_head.weight', it logs a debug message and returns an empty list to skip processing.
    - If the `name` ends with 'norm.weight', it increments the `data_torch` tensor by 1.
    - Finally, it returns a list containing a tuple of the mapped tensor name and the (possibly modified) `data_torch` tensor.
- **Output**: Returns an iterable of tuples, each containing a string (the mapped tensor name) and a `Tensor` (the modified tensor).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.GemmaModel`](#cpp/convert_hf_to_ggufGemmaModel)  (Base Class)



---
### Gemma2Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.Gemma2Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as GEMMA2.
- **Description**: The `Gemma2Model` class extends `TextModel` and is designed for causal language modeling, incorporating methods to set vocabulary and configure model parameters, while also handling tensor modifications during model loading.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Gemma2Model.set_vocab`](#Gemma2Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.Gemma2Model.set_gguf_parameters`](#Gemma2Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Gemma2Model.modify_tensors`](#Gemma2Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### Gemma2Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Gemma2Model.set_vocab}} -->
The `set_vocab` method initializes the vocabulary settings for the model by configuring the sentencepiece tokenizer and adjusting the space prefix in the GGUF writer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `Gemma2Model` class, which contains the model architecture and methods for configuring the model.
- **Control Flow**:
    - Calls the [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece) method to set up the vocabulary using the sentencepiece tokenizer.
    - Invokes the `add_add_space_prefix` method on the `gguf_writer` object with a parameter of `False` to configure the space prefix setting.
- **Output**: The method does not return any value; it modifies the internal state of the model by setting up the vocabulary and configuring the GGUF writer.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Gemma2Model`](#cpp/convert_hf_to_ggufGemma2Model)  (Base Class)


---
#### Gemma2Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Gemma2Model.set_gguf_parameters}} -->
The `set_gguf_parameters` method configures various parameters for the GGUF writer based on hyperparameters defined in the model.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `Gemma2Model` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - The method retrieves hyperparameters from `self.hparams`.
    - It extracts the number of hidden layers from the hyperparameters to set the block count.
    - The method sequentially calls various methods on `self.gguf_writer` to set parameters such as context length, embedding length, block count, and others based on the hyperparameters.
- **Output**: The method does not return any value; it modifies the state of the `gguf_writer` by adding various configuration parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.Gemma2Model`](#cpp/convert_hf_to_ggufGemma2Model)  (Base Class)


---
#### Gemma2Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Gemma2Model.modify_tensors}} -->
The `modify_tensors` method processes a tensor based on its name, modifying it if necessary and returning a mapped name with the tensor.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method begins by deleting the unused `bid` parameter.
    - If the `name` is 'lm_head.weight', a debug message is logged and an empty list is returned to skip processing.
    - If the `name` ends with 'norm.weight', the `data_torch` tensor is incremented by 1.
    - Finally, the method returns a list containing a tuple of the mapped tensor name and the possibly modified tensor.
- **Output**: An iterable of tuples, each containing a string (the mapped tensor name) and a `Tensor` (the modified tensor).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Gemma2Model`](#cpp/convert_hf_to_ggufGemma2Model)  (Base Class)



---
### Gemma3Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.Gemma3Model}} -->
- **Members**:
    - `model_arch`: Specifies the architecture type for the model.
- **Description**: The `Gemma3Model` class extends `TextModel` and is designed for causal language modeling and conditional generation, incorporating various methods to set vocabulary and configure model parameters based on hyperparameters.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Gemma3Model.set_vocab`](#Gemma3Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.Gemma3Model.set_gguf_parameters`](#Gemma3Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Gemma3Model.modify_tensors`](#Gemma3Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### Gemma3Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Gemma3Model.set_vocab}} -->
The `set_vocab` method initializes the vocabulary for the model by invoking a specific method to set the vocabulary and configures the space prefix for the GGUF writer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `Gemma3Model` class, which contains methods and properties related to the model.
- **Control Flow**:
    - The method first calls `self._set_vocab_sentencepiece()` to set the vocabulary using a sentencepiece model.
    - Then, it calls `self.gguf_writer.add_add_space_prefix(False)` to configure the GGUF writer not to add a space prefix.
- **Output**: The method does not return any value; it performs actions that modify the internal state of the model and its associated GGUF writer.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Gemma3Model`](#cpp/convert_hf_to_ggufGemma3Model)  (Base Class)


---
#### Gemma3Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Gemma3Model.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves hyperparameters from the instance's `hparams` attribute.
    - Adds context length, embedding length, block count, feed-forward length, head count, layer normalization epsilon, key length, value length, file type, and rope frequency base to the GGUF writer using values from `hparams`.
    - Checks for the presence of `attn_logit_softcapping` and `final_logit_softcapping` in `hparams` and asserts they are None.
    - Adds sliding window and key-value head count to the GGUF writer.
    - If `rope_scaling` is specified in `hparams`, asserts its type is linear and adds the corresponding scaling type and factor to the GGUF writer.
- **Output**: The method does not return a value; it configures the GGUF writer with various parameters derived from the hyperparameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.Gemma3Model`](#cpp/convert_hf_to_ggufGemma3Model)  (Base Class)


---
#### Gemma3Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Gemma3Model.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on the provided name, returning a mapped name and the modified tensor.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the data to be modified.
    - `name`: A string representing the name associated with the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that is unused in the current implementation.
- **Control Flow**:
    - The method first checks if the `name` starts with 'language_model.' and modifies it accordingly.
    - If the `name` starts with any of the specified prefixes related to vision models, the method returns an empty list to skip processing.
    - If the `name` contains 'embed_tokens.weight', it creates a vocabulary and truncates `data_torch` to match the length of the tokens.
    - If the `name` ends with 'norm.weight', it increments the `data_torch` tensor by 1.
    - Finally, the method returns a list containing a tuple of the mapped name and the modified tensor.
- **Output**: Returns an iterable of tuples, each containing a string (the mapped tensor name) and a `Tensor` (the modified tensor).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._create_vocab_sentencepiece`](#TextModel_create_vocab_sentencepiece)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Gemma3Model`](#cpp/convert_hf_to_ggufGemma3Model)  (Base Class)



---
### Gemma3VisionModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.Gemma3VisionModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `gguf_writer`: An instance of a writer for GGUF format used in the model.
    - `hparams`: Hyperparameters for configuring the model.
    - `preprocessor_config`: Configuration settings for preprocessing input data.
- **Description**: The `Gemma3VisionModel` class extends `MmprojModel` and is designed for conditional generation tasks, incorporating methods to set GGUF parameters, modify tensors, and handle quantization for vision-related components.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Gemma3VisionModel.set_gguf_parameters`](#Gemma3VisionModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Gemma3VisionModel.tensor_force_quant`](#Gemma3VisionModeltensor_force_quant)
    - [`llama.cpp/convert_hf_to_gguf.Gemma3VisionModel.modify_tensors`](#Gemma3VisionModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)

**Methods**

---
#### Gemma3VisionModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Gemma3VisionModel.set_gguf_parameters}} -->
Sets parameters for the GGUF model related to vision processing.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and configuration for the GGUF model.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to initialize base parameters.
    - Retrieves hyperparameters from `self.hparams`.
    - Adds a specific clip projector type to the GGUF writer.
    - Sets the layer normalization epsilon and specifies the use of GELU activation.
    - Calculates the projection scale factor based on image sequence length, image size, and patch size.
    - Conditionally adds the projection scale factor to the GGUF writer if it is not the default value.
- **Output**: No explicit return value; modifies the state of the GGUF writer with the specified parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Gemma3VisionModel`](#cpp/convert_hf_to_ggufGemma3VisionModel)  (Base Class)


---
#### Gemma3VisionModel\.tensor\_force\_quant<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Gemma3VisionModel.tensor_force_quant}} -->
Determines the quantization type for a tensor based on its name.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `name`: A string representing the name of the tensor.
    - `new_name`: A string intended for a new name, but not used in the method.
    - `bid`: An identifier that is not used in the method.
    - `n_dims`: The number of dimensions of the tensor, which is also not used.
- **Control Flow**:
    - The method starts by deleting the unused parameters 'bid', 'new_name', and 'n_dims'.
    - It checks if the string 'input_projection' is present in the 'name' argument; if so, it returns the quantization type F16.
    - If the string '.embeddings.' is found in 'name', it returns the quantization type F32.
    - If neither condition is met, it returns False.
- **Output**: Returns a quantization type (F16 or F32) based on the tensor's name, or False if no conditions are met.
- **See also**: [`llama.cpp/convert_hf_to_gguf.Gemma3VisionModel`](#cpp/convert_hf_to_ggufGemma3VisionModel)  (Base Class)


---
#### Gemma3VisionModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Gemma3VisionModel.modify_tensors}} -->
The `modify_tensors` method processes specific tensor names and modifies their values based on predefined conditions.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the data to be modified.
    - `name`: A string representing the name of the tensor.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method begins by deleting the unused `bid` parameter.
    - It checks if the `name` contains 'vision_model.head.' and returns an empty list if true, skipping further processing.
    - If the `name` starts with specific prefixes related to vision models, it modifies the `name` by replacing '_weight' with '.weight'.
    - If the modified `name` is 'soft_emb_norm.weight', it logs a correction message and increments the `data_torch` value by 1.
    - Finally, it returns a list containing a tuple of the mapped tensor name and the modified `data_torch`.
    - If none of the conditions are met, it returns an empty list.
- **Output**: The method outputs an iterable of tuples, each containing a modified tensor name and its corresponding modified tensor, or an empty list if no modifications are made.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Gemma3VisionModel`](#cpp/convert_hf_to_ggufGemma3VisionModel)  (Base Class)



---
### StarCoder2Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.StarCoder2Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type for the model as `STARCODER2`.
- **Description**: The `StarCoder2Model` class extends `TextModel` and is registered with the `ModelBase` under the identifier 'Starcoder2ForCausalLM', defining a specific architecture for causal language modeling.
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)


---
### Rwkv6Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.Rwkv6Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as RWKV6.
    - `lerp_weights`: Stores a dictionary of linear interpolation weights indexed by integer keys.
- **Description**: The `Rwkv6Model` class extends `TextModel` and is designed for causal language modeling using the RWKV6 architecture, providing methods to set vocabulary and configure model parameters, as well as to modify tensor data during processing.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Rwkv6Model.set_vocab`](#Rwkv6Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.Rwkv6Model.set_gguf_parameters`](#Rwkv6Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Rwkv6Model.modify_tensors`](#Rwkv6Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### Rwkv6Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Rwkv6Model.set_vocab}} -->
Sets the vocabulary for the RWKV model by invoking a specific internal method.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `Rwkv6Model` class, which contains the model's parameters and methods.
- **Control Flow**:
    - The method calls the [`_set_vocab_rwkv_world`](#TextModel_set_vocab_rwkv_world) method on the instance, which is responsible for setting the vocabulary specific to the RWKV model architecture.
- **Output**: The method does not return any value; it performs an internal operation to configure the model's vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_rwkv_world`](#TextModel_set_vocab_rwkv_world)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Rwkv6Model`](#cpp/convert_hf_to_ggufRwkv6Model)  (Base Class)


---
#### Rwkv6Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Rwkv6Model.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves hyperparameters from `self.hparams` such as number of hidden layers, head size, hidden size, layer normalization epsilon, and others.
    - Calculates `intermediate_size` based on the hidden size if it is not provided.
    - Determines `time_mix_extra_dim` and `time_decay_extra_dim` based on the hidden size.
    - Calls methods on `self.gguf_writer` to set various parameters including context length, embedding length, block count, and others.
- **Output**: The method does not return a value; it configures the GGUF writer with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.Rwkv6Model`](#cpp/convert_hf_to_ggufRwkv6Model)  (Base Class)


---
#### Rwkv6Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Rwkv6Model.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on specific naming conventions and conditions.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name associated with the tensor.
    - `bid`: An optional integer that may represent a block ID for certain operations.
- **Control Flow**:
    - The method begins by mapping the input `name` to a new tensor name using `self.map_tensor_name`.
    - If the new name does not end with '.weight' or '.bias', it appends '.weight' to the name.
    - Depending on the suffix of the new name, the method may transpose or permute the `data_torch` tensor.
    - If the new name indicates a decay or contains 'lerp', the tensor is squeezed to remove dimensions of size one.
    - The method attempts to access a parameter for rescaling every N layers, and if applicable, rescales the tensor based on the block ID.
    - If `bid` is provided and the new name indicates a 'time_mix_lerp', it stores the tensor in a dictionary and checks if all required tensors are present to yield a fused tensor.
    - Finally, the method yields the modified tensor along with its new name.
- **Output**: The method yields an iterable of tuples, each containing the modified tensor's new name and the modified tensor itself.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Rwkv6Model`](#cpp/convert_hf_to_ggufRwkv6Model)  (Base Class)



---
### RWKV6Qwen2Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.RWKV6Qwen2Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `RWKV6Qwen2Model` class extends `Rwkv6Model` and is designed for causal language modeling, incorporating specific methods for setting vocabulary and configuring model parameters, particularly for the RWKV6QWEN2 architecture.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.RWKV6Qwen2Model.set_vocab`](#RWKV6Qwen2Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.RWKV6Qwen2Model.set_gguf_parameters`](#RWKV6Qwen2Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.RWKV6Qwen2Model.modify_tensors`](#RWKV6Qwen2Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.Rwkv6Model`](#cpp/convert_hf_to_ggufRwkv6Model)

**Methods**

---
#### RWKV6Qwen2Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.RWKV6Qwen2Model.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `RWKV6Qwen2Model` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves hyperparameters from `self.hparams` such as number of hidden layers, attention heads, and hidden size.
    - Calculates `head_size` as the hidden size divided by the number of attention heads.
    - Uses default values for `time_mix_extra_dim` and `time_decay_extra_dim` based on the hidden size if not explicitly set.
    - Calls various methods on `self.gguf_writer` to set parameters like context length, embedding length, and block count.
    - Sets special parameters for time mixing and head counts for the model.
- **Output**: The method does not return a value; it configures the GGUF writer with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.RWKV6Qwen2Model`](#cpp/convert_hf_to_ggufRWKV6Qwen2Model)  (Base Class)



---
### Rwkv7Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.Rwkv7Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
    - `lerp_weights`: Stores the linear interpolation weights for model layers.
    - `lora_needs_transpose`: Indicates whether the LoRA weights need to be transposed.
- **Description**: The `Rwkv7Model` class extends `TextModel` and is designed for causal language modeling using the RWKV7 architecture, providing methods for vocabulary setting, parameter configuration, and tensor modification, while managing specific model attributes such as layer weights and LoRA configurations.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Rwkv7Model.set_vocab`](#Rwkv7Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.Rwkv7Model.calc_lora_rank`](#Rwkv7Modelcalc_lora_rank)
    - [`llama.cpp/convert_hf_to_gguf.Rwkv7Model.set_gguf_parameters`](#Rwkv7Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.Rwkv7Model.modify_tensors`](#Rwkv7Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### Rwkv7Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Rwkv7Model.set_vocab}} -->
Sets the vocabulary for the RWKV model by invoking a specific internal method.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - The method directly calls the [`_set_vocab_rwkv_world`](#TextModel_set_vocab_rwkv_world) method without any conditions or loops.
- **Output**: The method does not return any value; it performs an internal operation to set the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_rwkv_world`](#TextModel_set_vocab_rwkv_world)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Rwkv7Model`](#cpp/convert_hf_to_ggufRwkv7Model)  (Base Class)


---
#### Rwkv7Model\.calc\_lora\_rank<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Rwkv7Model.calc_lora_rank}} -->
Calculates the LoRA rank based on the given hidden size, exponent, and multiplier.
- **Inputs**:
    - `hidden_size`: The size of the hidden layer, which influences the rank calculation.
    - `exponent`: The exponent used in the calculation to adjust the influence of hidden size.
    - `multiplier`: A multiplier that scales the result of the calculation.
- **Control Flow**:
    - Calculates the value of `hidden_size` raised to the power of `exponent`.
    - Multiplies the result by `multiplier` and divides by 32.
    - Rounds the result to the nearest integer.
    - Ensures the final result is at least 1 by using `max(1, ...)`.
    - Multiplies the final result by 32 to get the LoRA rank.
- **Output**: Returns the calculated LoRA rank as an integer.
- **See also**: [`llama.cpp/convert_hf_to_gguf.Rwkv7Model`](#cpp/convert_hf_to_ggufRwkv7Model)  (Base Class)


---
#### Rwkv7Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Rwkv7Model.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves the number of hidden layers from hyperparameters to set the block count.
    - Attempts to retrieve 'head_size' and 'layer_norm_eps' from hyperparameters, falling back to alternative values if they are not found.
    - Calculates 'intermediate_size' based on 'hidden_size' or defaults to four times 'hidden_size'.
    - Attempts to retrieve various 'lora_rank' parameters from hyperparameters, using a calculation method if they are not found.
    - Adds various parameters to the GGUF writer, including context length, embedding length, and others.
- **Output**: The method does not return a value but configures the GGUF writer with the specified parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.Rwkv7Model.calc_lora_rank`](#Rwkv7Modelcalc_lora_rank)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Rwkv7Model`](#cpp/convert_hf_to_ggufRwkv7Model)  (Base Class)


---
#### Rwkv7Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Rwkv7Model.modify_tensors}} -->
The `modify_tensors` method processes and renames tensor data based on specific rules and conditions, yielding modified tensor names and their corresponding data.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the original name of the tensor.
    - `bid`: An optional integer that indicates the block ID, or None.
- **Control Flow**:
    - The method begins by standardizing the tensor name by replacing certain substrings to unify naming conventions.
    - It checks if the name contains '_lora.lora' to determine if transposition is needed for LoRA layers.
    - If the name indicates an attention vector and the block ID is 0, it may return early if the tensor is not used.
    - If the block ID is not None and the name indicates an attention tensor, it handles the tensor differently based on whether it has already been concatenated.
    - If the block ID is None, it processes the tensor data, potentially transposing it based on naming conventions.
    - Finally, it yields the modified tensor name and the processed tensor data.
- **Output**: The method yields an iterable of tuples, each containing a modified tensor name and its corresponding `Tensor` data.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Rwkv7Model`](#cpp/convert_hf_to_ggufRwkv7Model)  (Base Class)



---
### ARwkv7Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.ARwkv7Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as ARWKV7.
- **Description**: The `ARwkv7Model` class extends `Rwkv7Model` and is designed for hybrid causal language modeling, incorporating methods to set vocabulary and configure GGUF parameters based on model hyperparameters.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.ARwkv7Model.set_vocab`](#ARwkv7Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.ARwkv7Model.set_gguf_parameters`](#ARwkv7Modelset_gguf_parameters)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.Rwkv7Model`](#cpp/convert_hf_to_ggufRwkv7Model)

**Methods**

---
#### ARwkv7Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ARwkv7Model.set_vocab}} -->
The `set_vocab` method attempts to set the vocabulary using a SentencePiece model and falls back to a GPT-2 model if a FileNotFoundError occurs.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `ARwkv7Model` class, which contains methods for setting vocabulary.
- **Control Flow**:
    - The method first attempts to call the [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece) method to set the vocabulary.
    - If a `FileNotFoundError` is raised during the execution of [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece), it catches the exception and calls the [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) method as a fallback.
- **Output**: The method does not return a value; it modifies the internal state of the model by setting the vocabulary based on the available model.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ARwkv7Model`](#cpp/convert_hf_to_ggufARwkv7Model)  (Base Class)


---
#### ARwkv7Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ARwkv7Model.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class containing hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves hyperparameters from `self.hparams` such as number of hidden layers, hidden size, head size, RMS normalization epsilon, intermediate size, and whether WKV has a gate.
    - Asserts that the WKV version is 7 to ensure compatibility.
    - Defines several LORA rank parameters based on the retrieved hyperparameters.
    - Calls methods on `self.gguf_writer` to set various model parameters including context length, embedding length, block count, layer normalization RMS epsilon, head size, and LORA ranks.
- **Output**: The method does not return a value but configures the GGUF writer with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ARwkv7Model`](#cpp/convert_hf_to_ggufARwkv7Model)  (Base Class)



---
### MambaModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.MambaModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
    - `_tok_embd`: Holds the token embedding tensor.
- **Description**: The `MambaModel` class extends `TextModel` and is designed for causal language modeling, incorporating methods for setting vocabulary, configuring model parameters, and modifying tensor data during processing.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.MambaModel.set_vocab`](#MambaModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.MambaModel.set_gguf_parameters`](#MambaModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.MambaModel.modify_tensors`](#MambaModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### MambaModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MambaModel.set_vocab}} -->
Sets the vocabulary size and initializes the tokenizer based on available files.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `MambaModel` class, which contains hyperparameters and model directory information.
- **Control Flow**:
    - Retrieves the initial vocabulary size from the hyperparameters.
    - Rounds the vocabulary size up to the nearest multiple of 8 using ceiling division.
    - Updates the vocabulary size in the hyperparameters.
    - Checks for the existence of a 'tokenizer.json' file and calls `_set_vocab_gpt2()` if it exists.
    - Checks for the existence of a 'tokenizer.model' file and calls `_set_vocab_sentencepiece()` if it exists.
    - If neither tokenizer file is found, defaults to using the GPT-NeoX tokenizer by calling `_set_vocab_builtin()`.
- **Output**: The method does not return a value; it modifies the vocabulary size in the hyperparameters and sets the tokenizer accordingly.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_builtin`](#TextModel_set_vocab_builtin)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MambaModel`](#cpp/convert_hf_to_ggufMambaModel)  (Base Class)


---
#### MambaModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MambaModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains the method.
- **Control Flow**:
    - Retrieves hyperparameters using [`find_hparam`](#ModelBasefind_hparam) method, providing default values if not found.
    - Checks if the model type is 'falcon_mamba' to determine if RMS normalization should be applied.
    - Asserts that the `d_inner` parameter is twice the `d_model` parameter to ensure model consistency.
    - Calls various methods on `gguf_writer` to set the model parameters based on the retrieved hyperparameters.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` with the set parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MambaModel`](#cpp/convert_hf_to_ggufMambaModel)  (Base Class)


---
#### MambaModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.MambaModel.modify_tensors}} -->
The `modify_tensors` method processes a given tensor based on its name and returns a modified tensor along with its new name.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name associated with the tensor.
    - `bid`: An optional integer that may be used to identify a specific tensor or batch.
- **Control Flow**:
    - The method formats the output tensor name and token embedding name using [`format_tensor_name`](#ModelBaseformat_tensor_name).
    - It maps the input tensor name to a new name using [`map_tensor_name`](#ModelBasemap_tensor_name).
    - If the input name ends with '.A_log', it transforms the tensor by applying a negative exponential function.
    - If the new name matches a specific tensor type and the tensor has a certain shape, it squeezes the tensor to remove dimensions of size one.
    - If the token embedding has been previously set and the new name matches the output name, it checks for equality with the current tensor and omits it if they are equivalent.
    - If the new name matches the token embedding name, it updates the stored token embedding with the current tensor.
- **Output**: Returns an iterable of tuples, each containing the new tensor name and the modified tensor.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.match_model_tensor_name`](#ModelBasematch_model_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.MambaModel`](#cpp/convert_hf_to_ggufMambaModel)  (Base Class)



---
### CommandR2Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.CommandR2Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture of the model as COMMAND_R.
    - `hparams`: Holds hyperparameters for the model, including max position embeddings.
- **Description**: The `CommandR2Model` class extends `TextModel` and is designed for a specific model architecture, incorporating hyperparameter management and configuration for a causal language model.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.CommandR2Model.__init__`](#CommandR2Model__init__)
    - [`llama.cpp/convert_hf_to_gguf.CommandR2Model.set_gguf_parameters`](#CommandR2Modelset_gguf_parameters)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### CommandR2Model\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.CommandR2Model.__init__}} -->
Initializes the `CommandR2Model` by setting up hyperparameters and calling the parent class's initializer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `args`: Variable length argument list passed to the parent class initializer.
    - `kwargs`: Keyword arguments passed to the parent class initializer.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method using `super()` to ensure proper initialization.
    - Sets the `max_position_embeddings` hyperparameter by calling [`find_hparam`](#ModelBasefind_hparam) with a list of potential keys.
- **Output**: No explicit output; initializes the instance with specific hyperparameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.CommandR2Model`](#cpp/convert_hf_to_ggufCommandR2Model)  (Base Class)


---
#### CommandR2Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.CommandR2Model.set_gguf_parameters}} -->
Sets GGUF parameters for the model by invoking the parent method and adding specific hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `CommandR2Model` class.
- **Control Flow**:
    - Calls the [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method from the parent class using `super()`.
    - Adds the `logit_scale` parameter from the model's hyperparameters to the `gguf_writer`.
    - Sets the rope scaling type to `NONE` in the `gguf_writer`.
- **Output**: No explicit output; modifies the state of the `gguf_writer` with new parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.CommandR2Model`](#cpp/convert_hf_to_ggufCommandR2Model)  (Base Class)



---
### Cohere2Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.Cohere2Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as COHERE2.
- **Description**: The `Cohere2Model` class extends the `TextModel` class and is designed for causal language modeling, incorporating specific parameters and configurations for the COHERE2 architecture.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Cohere2Model.set_gguf_parameters`](#Cohere2Modelset_gguf_parameters)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### Cohere2Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Cohere2Model.set_gguf_parameters}} -->
Sets GGUF parameters for the `Cohere2Model` by configuring various hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `Cohere2Model` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class parameters are set.
    - Adds the logit scale, sliding window, and vocabulary size to the GGUF writer using values from the instance's hyperparameters.
    - Calculates the rope dimension count based on the rotary percentage, hidden size, and number of attention heads, and adds it to the GGUF writer.
    - Sets the rope scaling type to `NONE` in the GGUF writer.
- **Output**: The method does not return a value but configures the GGUF writer with specific model parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Cohere2Model`](#cpp/convert_hf_to_ggufCohere2Model)  (Base Class)



---
### OlmoModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.OlmoModel}} -->
- **Decorators**: `@ModelBase.register`, `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `OlmoModel` class extends `TextModel` and is designed for causal language modeling, incorporating specific model architecture settings and methods for parameter adjustment and tensor modification.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.OlmoModel.set_gguf_parameters`](#OlmoModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.OlmoModel.modify_tensors`](#OlmoModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### OlmoModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OlmoModel.set_gguf_parameters}} -->
Sets parameters for the GGUF model, including layer normalization and optional clamping for QKV.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure base parameters are set.
    - Adds a layer normalization epsilon value of 1e-5 to the GGUF writer.
    - Retrieves the `clip_qkv` parameter from the model's hyperparameters.
    - If `clip_qkv` is not None, adds a clamping operation for QKV to the GGUF writer.
- **Output**: No explicit output; modifies the state of the GGUF writer with new parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.OlmoModel`](#cpp/convert_hf_to_ggufOlmoModel)  (Base Class)


---
#### OlmoModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OlmoModel.modify_tensors}} -->
Modifies tensor data based on the name of the tensor, specifically for query and key projection weights.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be modified.
    - `bid`: An optional integer that is unused in this method.
- **Control Flow**:
    - The method retrieves the number of attention heads and key-value heads from the model's hyperparameters.
    - If the `name` ends with 'q_proj.weight', the method permutes the `data_torch` tensor using the number of attention heads.
    - If the `name` ends with 'k_proj.weight', the method permutes the `data_torch` tensor using the number of attention heads and key-value heads.
- **Output**: Returns an iterable of tuples, each containing a mapped tensor name and the modified tensor.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.OlmoModel`](#cpp/convert_hf_to_ggufOlmoModel)  (Base Class)



---
### Olmo2Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.Olmo2Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as OLMO2.
- **Description**: The `Olmo2Model` class extends the `TextModel` class and is registered with the `ModelBase` under the identifier 'Olmo2ForCausalLM', defining its architecture as OLMO2.
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)


---
### OlmoeModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.OlmoeModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture of the model as `gguf.MODEL_ARCH.OLMOE`.
    - `_experts`: Holds a list of dictionaries containing expert tensors or is None.
- **Description**: The `OlmoeModel` class extends `TextModel` and is designed for causal language modeling, incorporating expert layers and tensor management for model training.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.OlmoeModel.set_gguf_parameters`](#OlmoeModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.OlmoeModel.modify_tensors`](#OlmoeModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.OlmoeModel.prepare_tensors`](#OlmoeModelprepare_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### OlmoeModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OlmoeModel.set_gguf_parameters}} -->
Sets parameters for the GGUF model, including layer normalization and expert count.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class parameters are set.
    - Adds a layer normalization RMS epsilon value of 1e-5 to the GGUF writer.
    - Checks if the number of experts is specified in the hyperparameters; if so, adds this count to the GGUF writer.
- **Output**: No explicit return value; modifies the state of the GGUF writer with the specified parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.OlmoeModel`](#cpp/convert_hf_to_ggufOlmoeModel)  (Base Class)


---
#### OlmoeModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OlmoeModel.modify_tensors}} -->
The `modify_tensors` method processes and merges tensor data for experts in a model, returning the modified tensors.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name associated with the tensor, which may indicate if it relates to experts.
    - `bid`: An optional integer that specifies the block ID for the experts.
- **Control Flow**:
    - Checks if the `name` contains the substring 'experts' to determine if expert processing is needed.
    - Asserts that `bid` is not None if experts are being processed.
    - Initializes the `_experts` list if it is None, creating a dictionary for each block.
    - Stores the `data_torch` tensor in the `_experts` dictionary for the specified `bid` and `name`.
    - If the number of stored experts reaches three times the number of experts, it proceeds to merge them.
    - Iterates over the weight names ('down_proj', 'gate_proj', 'up_proj') to collect and stack the tensors from the experts.
    - Maps the merged tensor names and returns a list of tuples containing the new names and the stacked tensors.
    - If the expert count is not reached, returns an empty list.
    - If the `name` does not indicate experts, returns a tuple with the mapped name and the original tensor.
- **Output**: Returns an iterable of tuples, each containing a string (the mapped tensor name) and a `Tensor` (the modified tensor), or an empty list if no merging occurs.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.OlmoeModel`](#cpp/convert_hf_to_ggufOlmoeModel)  (Base Class)


---
#### OlmoeModel\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OlmoeModel.prepare_tensors}} -->
The [`prepare_tensors`](#ModelBaseprepare_tensors) method prepares the model's tensors and checks for any unprocessed expert tensors.
- **Decorators**: `@super`
- **Inputs**:
    - `self`: An instance of the class, which contains the model's parameters and state.
- **Control Flow**:
    - Calls the parent class's [`prepare_tensors`](#ModelBaseprepare_tensors) method to perform any necessary setup.
    - Checks if the `_experts` attribute is not `None`.
    - Flattens the `_experts` list of dictionaries into a list of keys (expert names).
    - If there are any unprocessed expert names, raises a `ValueError` with the list of these names.
- **Output**: The method does not return a value; it raises an exception if there are unprocessed experts.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.OlmoeModel`](#cpp/convert_hf_to_ggufOlmoeModel)  (Base Class)



---
### JinaBertV2Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.JinaBertV2Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type for the model.
    - `intermediate_size`: Holds the size of the intermediate layer as defined in the hyperparameters.
- **Description**: The `JinaBertV2Model` class extends `BertModel` to implement a specific variant of the BERT architecture, providing additional functionality for handling tensors and vocabulary based on the tokenizer type, while also registering itself with a model base for identification.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.JinaBertV2Model.__init__`](#JinaBertV2Model__init__)
    - [`llama.cpp/convert_hf_to_gguf.JinaBertV2Model.get_tensors`](#JinaBertV2Modelget_tensors)
    - [`llama.cpp/convert_hf_to_gguf.JinaBertV2Model.set_vocab`](#JinaBertV2Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.JinaBertV2Model.modify_tensors`](#JinaBertV2Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.BertModel`](#cpp/convert_hf_to_ggufBertModel)

**Methods**

---
#### JinaBertV2Model\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.JinaBertV2Model.__init__}} -->
Initializes the `JinaBertV2Model` by calling the parent class's initializer and setting the intermediate size from hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `args`: Variable length argument list passed to the parent class initializer.
    - `kwargs`: Keyword arguments passed to the parent class initializer.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method using `super()` to ensure proper initialization of the base class.
    - Sets the `intermediate_size` attribute using the value from the `hparams` dictionary.
- **Output**: No explicit output; initializes the instance of `JinaBertV2Model` with specific attributes.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
- **See also**: [`llama.cpp/convert_hf_to_gguf.JinaBertV2Model`](#cpp/convert_hf_to_ggufJinaBertV2Model)  (Base Class)


---
#### JinaBertV2Model\.get\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.JinaBertV2Model.get_tensors}} -->
The [`get_tensors`](#ModelBaseget_tensors) method retrieves and processes tensor data from the parent class, modifying the names and splitting the data for specific layers.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `JinaBertV2Model` class.
- **Control Flow**:
    - Iterates over the tensor names and data returned by the parent class's [`get_tensors`](#ModelBaseget_tensors) method.
    - Checks if the current tensor name contains 'gated_layer'.
    - If it does, splits the tensor data into two parts based on `self.intermediate_size` and modifies the tensor names accordingly.
    - Yields the modified tensor names and their corresponding data.
    - If the name does not contain 'gated_layer', yields the original name and data.
- **Output**: Yields tuples of tensor names and their corresponding data, with specific modifications for tensors associated with gated layers.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.get_tensors`](#ModelBaseget_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.JinaBertV2Model`](#cpp/convert_hf_to_ggufJinaBertV2Model)  (Base Class)


---
#### JinaBertV2Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.JinaBertV2Model.set_vocab}} -->
Sets the vocabulary for the model based on the tokenizer type specified in the configuration file.
- **Inputs**: None
- **Control Flow**:
    - Reads the tokenizer class from a JSON configuration file.
    - Checks the tokenizer class and calls the appropriate method to set the vocabulary.
    - Raises a NotImplementedError if the tokenizer class is not supported.
    - Adds special tokens for beginning and end of sequences to the writer.
- **Output**: No explicit output; modifies the internal state of the model and the writer.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel.set_vocab`](#TextModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.JinaBertV2Model`](#cpp/convert_hf_to_ggufJinaBertV2Model)  (Base Class)


---
#### JinaBertV2Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.JinaBertV2Model.modify_tensors}} -->
The [`modify_tensors`](#ModelBasemodify_tensors) method modifies the input tensor's name by removing a specific prefix and then delegates the processing to a superclass method.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name associated with the tensor, which may have a prefix to be removed.
    - `bid`: An optional integer that may be used for additional identification or processing.
- **Control Flow**:
    - Checks if the `name` starts with the prefix 'bert.' and removes it if present.
    - Calls the [`modify_tensors`](#ModelBasemodify_tensors) method of the superclass with the modified name and the original tensor and bid.
- **Output**: Returns an iterable of tuples, each containing a string (the modified name) and a `Tensor` (the modified data) from the superclass method.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.modify_tensors`](#ModelBasemodify_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.JinaBertV2Model`](#cpp/convert_hf_to_ggufJinaBertV2Model)  (Base Class)



---
### OpenELMModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.OpenELMModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `_n_embd`: Represents the model dimension for embeddings.
    - `_num_kv_heads`: Stores the number of key-value heads for the model.
    - `_num_query_heads`: Stores the number of query heads for the model.
    - `_ffn_dims`: Holds the dimensions for feed-forward networks, calculated based on multipliers.
- **Description**: The `OpenELMModel` class extends `TextModel` and is designed for causal language modeling, incorporating specific architecture parameters and methods for managing model dimensions, vocabulary, and hyperparameters.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.OpenELMModel._make_divisible`](#OpenELMModel_make_divisible)
    - [`llama.cpp/convert_hf_to_gguf.OpenELMModel.__init__`](#OpenELMModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.OpenELMModel.set_vocab`](#OpenELMModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.OpenELMModel.set_gguf_parameters`](#OpenELMModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.OpenELMModel.find_hparam`](#OpenELMModelfind_hparam)
    - [`llama.cpp/convert_hf_to_gguf.OpenELMModel.modify_tensors`](#OpenELMModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### OpenELMModel\.\_make\_divisible<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OpenELMModel._make_divisible}} -->
The `_make_divisible` method adjusts a given value to ensure it is divisible by a specified divisor, with additional logic to prevent significant reduction in value.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `v`: A float or integer value that needs to be adjusted to be divisible by the divisor.
    - `divisor`: An integer that specifies the divisor to which the value `v` should be made divisible.
- **Control Flow**:
    - Calculates `new_v` as the maximum of the divisor and the nearest multiple of the divisor to `v`.
    - Checks if `new_v` is less than 90% of the original value `v`, and if so, increments `new_v` by the divisor to ensure it does not decrease significantly.
- **Output**: Returns the adjusted integer value that is divisible by the specified divisor.
- **See also**: [`llama.cpp/convert_hf_to_gguf.OpenELMModel`](#cpp/convert_hf_to_ggufOpenELMModel)  (Base Class)


---
#### OpenELMModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OpenELMModel.__init__}} -->
Initializes an instance of the `OpenELMModel` class, setting up various hyperparameters and configurations for the model.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `args`: Variable length argument list for additional parameters.
    - `kwargs`: Keyword arguments for additional parameters.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method to initialize inherited attributes.
    - Retrieves hyperparameters from `self.hparams` for `ffn_multipliers`, `ffn_dim_divisor`, `model_dim`, `num_kv_heads`, and `num_query_heads`.
    - Calculates the feed-forward network dimensions using a list comprehension that applies the [`_make_divisible`](#OpenELMModel_make_divisible) method to each multiplier.
    - Asserts that `num_kv_heads` and `num_query_heads` are lists containing integers.
- **Output**: No explicit output; initializes instance variables for the model based on provided hyperparameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
    - [`llama.cpp/convert_hf_to_gguf.OpenELMModel._make_divisible`](#OpenELMModel_make_divisible)
- **See also**: [`llama.cpp/convert_hf_to_gguf.OpenELMModel`](#cpp/convert_hf_to_ggufOpenELMModel)  (Base Class)


---
#### OpenELMModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OpenELMModel.set_vocab}} -->
Sets the vocabulary for the model either from a SentencePiece model or a built-in vocabulary.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `OpenELMModel` class.
- **Control Flow**:
    - Attempts to call the [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece) method to set the vocabulary from a SentencePiece model.
    - If a `FileNotFoundError` is raised, it falls back to calling [`_set_vocab_builtin`](#TextModel_set_vocab_builtin) with a default vocabulary type and size.
- **Output**: No explicit output; the method modifies the internal state of the model by setting the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_builtin`](#TextModel_set_vocab_builtin)
- **See also**: [`llama.cpp/convert_hf_to_gguf.OpenELMModel`](#cpp/convert_hf_to_ggufOpenELMModel)  (Base Class)


---
#### OpenELMModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OpenELMModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on model configuration.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `OpenELMModel` class, which contains model parameters and configurations.
- **Control Flow**:
    - Initializes local variables `n_embd`, `head_dim`, and `rot_pct` from the instance attributes.
    - Asserts that the `block_count` matches the lengths of `_num_kv_heads`, `_num_query_heads`, and `_ffn_dims` to ensure consistency.
    - Calls various methods on `gguf_writer` to set parameters such as block count, context length, embedding length, feed-forward dimensions, head counts, and other model-specific configurations.
- **Output**: This method does not return a value; it modifies the state of the `gguf_writer` with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.OpenELMModel`](#cpp/convert_hf_to_ggufOpenELMModel)  (Base Class)


---
#### OpenELMModel\.find\_hparam<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OpenELMModel.find_hparam}} -->
The [`find_hparam`](#ModelBasefind_hparam) method retrieves specific hyperparameters based on provided keys.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `keys`: An iterable of strings representing the keys for which hyperparameters are requested.
    - `optional`: A boolean flag indicating whether the retrieval of the hyperparameter is optional.
- **Control Flow**:
    - Checks if the string 'n_layers' is present in the `keys` iterable.
    - If 'n_layers' is found, it returns the value of the hyperparameter 'num_transformer_layers' from `self.hparams`.
    - If 'n_layers' is not found, it calls the parent class's [`find_hparam`](#ModelBasefind_hparam) method with the same arguments.
- **Output**: Returns the value of 'num_transformer_layers' if 'n_layers' is in `keys`, otherwise returns the result of the parent class's [`find_hparam`](#ModelBasefind_hparam) method.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.OpenELMModel`](#cpp/convert_hf_to_ggufOpenELMModel)  (Base Class)


---
#### OpenELMModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.OpenELMModel.modify_tensors}} -->
The `modify_tensors` method processes a tensor based on its name and an optional bid, yielding modified tensor names and their corresponding values.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that specifies the index of the layer, used for conditional processing of the tensor.
- **Control Flow**:
    - Checks if `bid` is not None and if `name` matches a specific pattern related to feed-forward network weights.
    - If the conditions are met, it splits the `data_torch` tensor into two parts based on the feed-forward dimension and yields them with formatted names.
    - If the conditions are not met, it yields the tensor with a mapped name.
- **Output**: An iterable of tuples, where each tuple contains a string (the modified tensor name) and a `Tensor` (the corresponding tensor data).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.OpenELMModel`](#cpp/convert_hf_to_ggufOpenELMModel)  (Base Class)



---
### ArcticModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.ArcticModel}} -->
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
    - `_experts`: Holds a list of expert tensors or None.
- **Description**: The `ArcticModel` class extends `TextModel` and is designed for handling a specific model architecture, providing functionality to set vocabulary and manage tensor modifications, particularly in the context of a causal language model.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.ArcticModel.set_vocab`](#ArcticModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.ArcticModel.set_gguf_parameters`](#ArcticModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.ArcticModel.modify_tensors`](#ArcticModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.ArcticModel.prepare_tensors`](#ArcticModelprepare_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### ArcticModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ArcticModel.set_vocab}} -->
Sets the vocabulary for the model by loading tokens from a tokenizer model file and configuring additional tokens from a JSON configuration.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains the method, providing access to model parameters and file paths.
- **Control Flow**:
    - Checks if the tokenizer model file exists; if not, logs an error and exits.
    - Loads the vocabulary from the tokenizer model file using `SentencePieceProcessor`.
    - Initializes lists for tokens, scores, and token types based on the vocabulary size.
    - Iterates through each token ID to retrieve the corresponding token piece, score, and type, updating the lists accordingly.
    - Checks for an additional configuration file to modify tokens based on user-defined settings, logging any changes made.
    - Adds the configured tokens, scores, and types to the `gguf_writer` for further processing.
- **Output**: No explicit return value; the method modifies the internal state of the model by setting up the vocabulary and writing it to a GGUF format.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ArcticModel`](#cpp/convert_hf_to_ggufArcticModel)  (Base Class)


---
#### ArcticModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ArcticModel.set_gguf_parameters}} -->
Sets the GGUF parameters for the model by adding vocabulary size and rope dimension count.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class logic is executed.
    - Retrieves hyperparameters from `self.hparams`.
    - Adds the vocabulary size to the GGUF writer using the value from `hparams['vocab_size']`.
    - Calculates the rope dimension count by dividing `hparams['hidden_size']` by `hparams['num_attention_heads']` and adds it to the GGUF writer.
- **Output**: No explicit return value; modifies the state of the GGUF writer by adding parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ArcticModel`](#cpp/convert_hf_to_ggufArcticModel)  (Base Class)


---
#### ArcticModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ArcticModel.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on specific naming conventions and conditions related to attention heads and expert models.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that indicates the block ID for processing expert tensors.
- **Control Flow**:
    - The method retrieves the number of attention heads and key-value heads from the model's hyperparameters.
    - It checks if the tensor name indicates a query or key projection weight and permutes the tensor accordingly.
    - If the tensor name indicates it belongs to block sparse experts, it processes the experts based on the block ID.
    - It asserts that the block ID is not None and initializes the experts list if it is None.
    - The method stores the tensor in the appropriate expert dictionary and checks if enough tensors have been collected to merge.
    - If enough tensors are collected, it merges them into a single 3D tensor and returns a list of tuples containing the new tensor names and their corresponding tensors.
    - If not enough tensors are collected, it returns an empty list.
    - If the tensor does not belong to the expert category, it returns a tuple with the mapped tensor name and the modified tensor.
- **Output**: The method returns an iterable of tuples, each containing a string (the mapped tensor name) and a `Tensor` (the modified tensor), or an empty list if not enough expert tensors are collected.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ArcticModel`](#cpp/convert_hf_to_ggufArcticModel)  (Base Class)


---
#### ArcticModel\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ArcticModel.prepare_tensors}} -->
Prepares tensors for a model by checking for unprocessed expert tensors.
- **Decorators**: `@super`
- **Inputs**:
    - `self`: An instance of the class containing the method.
- **Control Flow**:
    - Calls the [`prepare_tensors`](#ModelBaseprepare_tensors) method of the parent class.
    - Checks if `_experts` is not None.
    - Flattens the list of expert tensors into a list of their keys.
    - Raises a ValueError if there are any unprocessed expert tensors.
- **Output**: Raises a ValueError if there are unprocessed expert tensors; otherwise, returns None.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ArcticModel`](#cpp/convert_hf_to_ggufArcticModel)  (Base Class)



---
### DeepseekModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.DeepseekModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
    - `_experts`: Holds a list of expert tensors or None.
- **Description**: The `DeepseekModel` class extends `TextModel` and is designed for deep learning applications, specifically for causal language modeling, incorporating advanced features such as expert layers and tensor manipulation for model parameters.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.DeepseekModel.set_vocab`](#DeepseekModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.DeepseekModel.set_gguf_parameters`](#DeepseekModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.DeepseekModel.permute`](#DeepseekModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.DeepseekModel.modify_tensors`](#DeepseekModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.DeepseekModel.prepare_tensors`](#DeepseekModelprepare_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### DeepseekModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeepseekModel.set_vocab}} -->
Sets the vocabulary for the model by attempting to load it from a SentencePiece model and falling back to a GPT-2 model if the former is not found.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `DeepseekModel` class.
- **Control Flow**:
    - The method first attempts to call the [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece) method to set the vocabulary.
    - If a `FileNotFoundError` is raised during the execution of [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece), it catches the exception and calls the [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) method instead.
- **Output**: The method does not return any value; it modifies the internal state of the model by setting the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeepseekModel`](#cpp/convert_hf_to_ggufDeepseekModel)  (Base Class)


---
#### DeepseekModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeepseekModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `DeepseekModel` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class parameters are set.
    - Retrieves hyperparameters from `self.hparams`.
    - Checks if 'head_dim' is present in `hparams` to determine the value of `rope_dim`; if not, calculates it using 'hidden_size' and 'num_attention_heads'.
    - Uses the `gguf_writer` to add various parameters such as rope dimension count, scaling type, leading dense block count, vocabulary size, expert feed-forward length, expert weights scale, expert count, and expert shared count.
- **Output**: No explicit return value; the method modifies the state of the `gguf_writer` with the specified parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeepseekModel`](#cpp/convert_hf_to_ggufDeepseekModel)  (Base Class)


---
#### DeepseekModel\.permute<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeepseekModel.permute}} -->
The `permute` method reshapes and rearranges a tensor based on specified head dimensions.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `weights`: A `Tensor` that contains the weights to be permuted.
    - `n_head`: An integer representing the number of heads for the attention mechanism.
    - `n_head_kv`: An optional integer that specifies the number of key-value heads; if provided, it can override `n_head`.
- **Control Flow**:
    - If `n_head_kv` is not None and differs from `n_head`, `n_head` is updated to `n_head_kv`.
    - The method reshapes the `weights` tensor into a new shape that organizes the data into a specified number of heads and dimensions.
    - The axes of the reshaped tensor are swapped to rearrange the data appropriately.
    - Finally, the tensor is reshaped back to its original shape before being returned.
- **Output**: The method returns a reshaped `Tensor` that maintains the original data but is organized according to the specified head dimensions.
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeepseekModel`](#cpp/convert_hf_to_ggufDeepseekModel)  (Base Class)


---
#### DeepseekModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeepseekModel.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on specific naming conventions and expert configurations.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that indicates the block ID for expert processing.
- **Control Flow**:
    - The method retrieves the number of attention heads and key-value heads from the model's hyperparameters.
    - If the tensor name indicates it is a query projection, the tensor is permuted using the [`permute`](#LlamaModelpermute) method.
    - If the tensor name indicates it is a key projection, the tensor is permuted with respect to the number of key-value heads.
    - If the tensor name contains 'mlp.experts', the method processes the tensor as part of an expert configuration.
    - The method checks if the number of stored expert tensors for the given block ID has reached a threshold.
    - If the threshold is met, it merges the expert tensors into a single 3D tensor and returns them with a new name.
    - If the threshold is not met, it returns an empty list.
    - If the tensor does not match any special conditions, it returns the tensor with its mapped name.
- **Output**: The method returns an iterable of tuples, each containing a string (the mapped tensor name) and a `Tensor` (the modified tensor data). If no modifications are made, it returns an empty list.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeepseekModel`](#cpp/convert_hf_to_ggufDeepseekModel)  (Base Class)


---
#### DeepseekModel\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeepseekModel.prepare_tensors}} -->
The [`prepare_tensors`](#ModelBaseprepare_tensors) method prepares tensor data for the model and checks for any unprocessed expert tensors.
- **Decorators**: `@super`
- **Inputs**: None
- **Control Flow**:
    - Calls the parent class's [`prepare_tensors`](#ModelBaseprepare_tensors) method to perform any necessary setup.
    - Checks if the `_experts` attribute is not `None`.
    - Flattens the `_experts` list of dictionaries to extract keys representing expert tensors.
    - Raises a `ValueError` if there are any unprocessed expert tensors.
- **Output**: The method does not return a value but raises an exception if unprocessed expert tensors are found.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeepseekModel`](#cpp/convert_hf_to_ggufDeepseekModel)  (Base Class)



---
### DeepseekV2Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.DeepseekV2Model}} -->
- **Decorators**: `@ModelBase.register`, `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type for the model.
    - `_experts`: Holds a list of expert tensors or None.
- **Description**: The `DeepseekV2Model` class extends `TextModel` and is designed for causal language modeling, incorporating advanced features such as expert layers and customizable parameters for model architecture and tensor manipulation.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.DeepseekV2Model.set_vocab`](#DeepseekV2Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.DeepseekV2Model.set_gguf_parameters`](#DeepseekV2Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.DeepseekV2Model.modify_tensors`](#DeepseekV2Modelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.DeepseekV2Model.prepare_tensors`](#DeepseekV2Modelprepare_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### DeepseekV2Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeepseekV2Model.set_vocab}} -->
Sets the vocabulary for the model using the GPT-2 vocabulary setting method.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `DeepseekV2Model` class.
- **Control Flow**:
    - Calls the [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) method to set the vocabulary.
- **Output**: No output is returned; the method modifies the internal state of the model by setting the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeepseekV2Model`](#cpp/convert_hf_to_ggufDeepseekV2Model)  (Base Class)


---
#### DeepseekV2Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeepseekV2Model.set_gguf_parameters}} -->
Sets the GGUF parameters for the model by configuring various hyperparameters and invoking methods on the `gguf_writer`.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains the method, which holds the hyperparameters and the GGUF writer.
- **Control Flow**:
    - Sets the number of key-value heads in the hyperparameters to 1.
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class configurations are applied.
    - Retrieves the hyperparameters from `self.hparams` for further configuration.
    - Adds various parameters to the `gguf_writer` based on the hyperparameters, including dense block counts, vocabulary size, and expert configurations.
    - Checks the scoring function type and adds the corresponding expert gating function to the `gguf_writer`, raising an error for unsupported types.
    - Handles optional rope scaling configurations if present in the hyperparameters.
- **Output**: The method does not return a value; instead, it configures the `gguf_writer` with various parameters based on the model's hyperparameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeepseekV2Model`](#cpp/convert_hf_to_ggufDeepseekV2Model)  (Base Class)


---
#### DeepseekV2Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeepseekV2Model.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on specific naming conventions and conditions related to model architecture.
- **Decorators**: `@ModelBase.register`, `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name of the tensor, which may dictate how the tensor is processed.
    - `bid`: An optional integer that indicates the block ID, used for processing expert layers.
- **Control Flow**:
    - If the `name` ends with 'e_score_correction_bias', it is renamed to 'e_score_correction.bias'.
    - The method checks if the `name` corresponds to a Multi-Token Prediction (MTP) layer and skips processing if it does.
    - If the `name` indicates an expert layer, it stores the tensor in a dictionary and checks if enough tensors have been collected to merge them.
    - If merging is required, it stacks the tensors into a 3D tensor and renames them accordingly.
    - If the `name` ends with 'kv_b_proj.weight', it splits the tensor into two parts and transposes one of them before returning.
    - If none of the above conditions are met, it simply returns the tensor with its mapped name.
- **Output**: The method returns an iterable of tuples, each containing a modified tensor name and the corresponding tensor data.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeepseekV2Model`](#cpp/convert_hf_to_ggufDeepseekV2Model)  (Base Class)


---
#### DeepseekV2Model\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.DeepseekV2Model.prepare_tensors}} -->
The [`prepare_tensors`](#ModelBaseprepare_tensors) method prepares tensor data by invoking the parent class's method and checks for unprocessed expert tensors.
- **Inputs**: None
- **Control Flow**:
    - Calls the [`prepare_tensors`](#ModelBaseprepare_tensors) method from the parent class to perform initial tensor preparation.
    - Checks if the `_experts` attribute is not `None` to determine if there are expert tensors to process.
    - Flattens the list of expert dictionaries into a list of keys (tensor names) and raises a ValueError if any unprocessed experts are found.
- **Output**: The method does not return a value; instead, it raises a ValueError if there are unprocessed expert tensors.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.DeepseekV2Model`](#cpp/convert_hf_to_ggufDeepseekV2Model)  (Base Class)



---
### PLMModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.PLMModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as PLM.
- **Description**: The `PLMModel` class extends `TextModel` and is designed for a specific type of language model, incorporating methods for setting vocabulary and configuring model parameters.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.PLMModel.set_vocab`](#PLMModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.PLMModel.set_gguf_parameters`](#PLMModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.PLMModel.modify_tensors`](#PLMModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.PLMModel.prepare_tensors`](#PLMModelprepare_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### PLMModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.PLMModel.set_vocab}} -->
Sets the vocabulary for the model using the GPT-2 vocabulary setting method.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls the [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) method to set the vocabulary.
- **Output**: No output is returned; the method modifies the internal state of the model by setting the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.PLMModel`](#cpp/convert_hf_to_ggufPLMModel)  (Base Class)


---
#### PLMModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.PLMModel.set_gguf_parameters}} -->
The [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method initializes various parameters for the GGUF writer based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class initialization is performed.
    - Retrieves hyperparameters from the instance's `hparams` attribute.
    - Adds vocabulary size, KV LoRA rank, key length, value length, and ROPE dimension count to the GGUF writer using the retrieved hyperparameters.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` by adding various parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.PLMModel`](#cpp/convert_hf_to_ggufPLMModel)  (Base Class)


---
#### PLMModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.PLMModel.modify_tensors}} -->
The `modify_tensors` method maps a tensor name to a given tensor and returns it as a tuple.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name to be mapped to the tensor.
    - `bid`: An optional integer that may represent an identifier, but is not used in the method.
- **Control Flow**:
    - The method calls `self.map_tensor_name(name)` to get the mapped name for the tensor.
    - It returns a list containing a single tuple with the mapped name and the original tensor.
- **Output**: An iterable containing a tuple where the first element is the mapped tensor name and the second element is the original tensor.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.PLMModel`](#cpp/convert_hf_to_ggufPLMModel)  (Base Class)


---
#### PLMModel\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.PLMModel.prepare_tensors}} -->
This method calls the [`prepare_tensors`](#ModelBaseprepare_tensors) method of its superclass to prepare tensors.
- **Inputs**: None
- **Control Flow**:
    - The method directly invokes the [`prepare_tensors`](#ModelBaseprepare_tensors) method from the superclass without any additional logic or processing.
- **Output**: The method does not return any value; it performs an action by calling the superclass method.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.PLMModel`](#cpp/convert_hf_to_ggufPLMModel)  (Base Class)



---
### T5Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.T5Model}} -->
- **Decorators**: `@ModelBase.register`, `@ModelBase.register`, `@ModelBase.register`, `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the T5 model.
    - `shared_token_embeddings_found`: Indicates whether shared token embeddings have been found during processing.
- **Description**: The `T5Model` class extends `TextModel` and is designed for handling various T5 model architectures, providing functionalities for setting vocabulary, configuring model parameters, and modifying tensor data specific to T5-based models.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.T5Model.__init__`](#T5Model__init__)
    - [`llama.cpp/convert_hf_to_gguf.T5Model.set_vocab`](#T5Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.T5Model.set_gguf_parameters`](#T5Modelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.T5Model.modify_tensors`](#T5Modelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### T5Model\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.T5Model.__init__}} -->
Initializes an instance of the `T5Model` class, setting up shared token embeddings and calling the parent class's initializer.
- **Inputs**:
    - `args`: Positional arguments passed to the parent class initializer.
    - `kwargs`: Keyword arguments passed to the parent class initializer.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method with the provided arguments.
    - Initializes the instance variable `shared_token_embeddings_found` to `False`.
- **Output**: No output is returned; the method initializes the state of the object.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
- **See also**: [`llama.cpp/convert_hf_to_gguf.T5Model`](#cpp/convert_hf_to_ggufT5Model)  (Base Class)


---
#### T5Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.T5Model.set_vocab}} -->
Sets up the vocabulary for a T5 model by loading a tokenizer model and processing its tokens.
- **Decorators**: `@ModelBase.register`, `@ModelBase.register`, `@ModelBase.register`, `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the T5Model class, which contains model parameters and methods.
- **Control Flow**:
    - Sets the environment variable to avoid TypeError when importing the SentencePiece model.
    - Checks for the existence of the tokenizer model file and raises an error if not found.
    - Parses the tokenizer model and determines its type (BPE or UNIGRAM).
    - Loads the tokenizer and initializes token lists, scores, and types based on the vocabulary size.
    - Processes each token to determine its type and score, updating the respective lists.
    - Checks for additional tokens from a JSON file and updates the token lists accordingly.
    - Pads the token lists if the vocabulary size exceeds the number of tokens found.
    - Writes the processed vocabulary and tokenizer settings to a GGUF writer.
- **Output**: No explicit return value; the method modifies the internal state of the model by setting up the vocabulary and writing it to a GGUF format.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.T5Model`](#cpp/convert_hf_to_ggufT5Model)  (Base Class)


---
#### T5Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.T5Model.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on hyperparameters and context length.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Checks for the context length parameter in the hyperparameters; if not found, defaults to 512 and logs a warning.
    - Adds various model parameters such as context length, embedding length, feed-forward length, block count, head count, key length, value length, layer normalization epsilon, relative attention buckets count, layer normalization RMS epsilon, decoder start token ID, and file type to the GGUF writer.
- **Output**: No explicit return value; the method modifies the state of the GGUF writer by adding parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.T5Model`](#cpp/convert_hf_to_ggufT5Model)  (Base Class)


---
#### T5Model\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.T5Model.modify_tensors}} -->
The `modify_tensors` method processes tensor data for T5-based models, ensuring that only the first shared token embedding tensor is used.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name of the tensor being processed.
    - `bid`: An optional integer that is unused in the method.
- **Control Flow**:
    - The method begins by deleting the unused `bid` parameter.
    - It checks if the `name` corresponds to one of the shared token embedding tensors.
    - If the shared token embeddings have not been found yet, it sets `name` to 'shared.weight' and marks it as found.
    - If the shared token embeddings have already been found, it logs a debug message and returns an empty list.
    - If the `name` is valid, it returns a tuple containing the mapped tensor name and the original tensor data.
- **Output**: The method returns an iterable of tuples, each containing a string (the mapped tensor name) and the original `Tensor` object.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.T5Model`](#cpp/convert_hf_to_ggufT5Model)  (Base Class)



---
### T5EncoderModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.T5EncoderModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the T5 encoder model.
    - `shared_token_embeddings_found`: Indicates whether shared token embeddings have been located.
- **Description**: The `T5EncoderModel` class extends `TextModel` and is designed to handle the T5 encoder architecture, providing functionalities for setting vocabulary, managing tokenizer models, and configuring model parameters for text processing tasks.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.T5EncoderModel.__init__`](#T5EncoderModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.T5EncoderModel.set_vocab`](#T5EncoderModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.T5EncoderModel.set_gguf_parameters`](#T5EncoderModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.T5EncoderModel.modify_tensors`](#T5EncoderModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### T5EncoderModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.T5EncoderModel.__init__}} -->
Initializes a `T5EncoderModel` instance, setting up shared token embeddings.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `*args`: Variable length argument list for additional parameters.
    - `**kwargs`: Keyword arguments for additional parameters.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method to ensure proper initialization of the base class.
    - Sets the instance variable `shared_token_embeddings_found` to `False` to indicate that shared token embeddings have not yet been located.
- **Output**: No output is returned; the method initializes the instance state.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
- **See also**: [`llama.cpp/convert_hf_to_gguf.T5EncoderModel`](#cpp/convert_hf_to_ggufT5EncoderModel)  (Base Class)


---
#### T5EncoderModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.T5EncoderModel.set_vocab}} -->
Sets up the vocabulary for a T5 encoder model using a tokenizer model file.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the T5EncoderModel class.
- **Control Flow**:
    - Sets the environment variable to avoid TypeError when importing the SentencePiece model.
    - Checks for the existence of the tokenizer model file and raises a FileNotFoundError if not found.
    - Parses the tokenizer model file to determine the model type (BPE or UNIGRAM).
    - Loads the tokenizer and initializes lists for tokens, scores, and token types.
    - Iterates through the tokenizer's vocabulary to populate the tokens, scores, and token types based on their properties.
    - Checks for an additional tokens file and updates the vocabulary if necessary.
    - Pads the vocabulary if the size is less than the expected vocabulary size.
    - Writes the vocabulary and tokenizer properties to the GGUF writer.
- **Output**: No explicit return value; the method modifies the internal state of the T5EncoderModel instance and writes the vocabulary to a GGUF format.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.T5EncoderModel`](#cpp/convert_hf_to_ggufT5EncoderModel)  (Base Class)


---
#### T5EncoderModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.T5EncoderModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on hyperparameters and context length.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `T5EncoderModel` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Checks if the context length (`n_ctx`) can be found from hyperparameters; if not, it defaults to 512 and logs a warning.
    - Adds various model parameters such as context length, embedding length, feed-forward length, block count, head count, key length, value length, layer normalization epsilon, relative attention buckets count, and file type to the GGUF writer.
- **Output**: The method does not return a value; it modifies the state of the GGUF writer by adding the specified parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.T5EncoderModel`](#cpp/convert_hf_to_ggufT5EncoderModel)  (Base Class)


---
#### T5EncoderModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.T5EncoderModel.modify_tensors}} -->
The `modify_tensors` method processes tensor data for T5-based models, managing shared token embeddings.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name of the tensor to be modified.
    - `bid`: An optional integer that is unused in the method.
- **Control Flow**:
    - The method begins by deleting the unused `bid` parameter.
    - It checks if the `name` corresponds to specific shared token embeddings.
    - If the shared token embeddings have not been found yet, it sets `name` to 'shared.weight' and marks it as found.
    - If the shared token embeddings have already been found, it logs a debug message and returns an empty list.
    - If the `name` does not match the specified embeddings, it returns a tuple containing the mapped tensor name and the original tensor data.
- **Output**: The method returns an iterable of tuples, each containing a string (the mapped tensor name) and the original `Tensor` object.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.T5EncoderModel`](#cpp/convert_hf_to_ggufT5EncoderModel)  (Base Class)



---
### JaisModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.JaisModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
    - `embeddings_scale`: Scale factor for the embeddings.
    - `width_scale`: Scale factor for the output width.
    - `max_alibi_bias`: Maximum bias value for ALiBi position embedding.
- **Description**: The `JaisModel` class extends `TextModel` and is designed for a specific model architecture, incorporating features like SwigLU activation and ALiBi position embeddings, while managing scaling factors for embeddings and output width.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.JaisModel.__init__`](#JaisModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.JaisModel.set_vocab`](#JaisModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.JaisModel.set_gguf_parameters`](#JaisModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.JaisModel.modify_tensors`](#JaisModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.JaisModel.prepare_tensors`](#JaisModelprepare_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### JaisModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.JaisModel.__init__}} -->
Initializes the `JaisModel` class, setting up activation functions, position embeddings, and scaling factors based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `args`: Positional arguments passed to the parent class constructor.
    - `kwargs`: Keyword arguments passed to the parent class constructor, which include hyperparameters for model configuration.
- **Control Flow**:
    - Calls the parent class constructor using `super().__init__(*args, **kwargs)`.
    - Asserts that the activation function specified in `hparams` is 'swiglu'.
    - Asserts that the position embedding type specified in `hparams` is 'alibi'.
    - Initializes `embeddings_scale` to 1.0 and updates it based on the presence of 'mup_embeddings_scale' or 'embeddings_scale' in `hparams`.
    - Initializes `width_scale` to 1.0 and updates it based on the presence of 'mup_output_alpha' and 'mup_width_scale' or 'width_scale' in `hparams`.
    - Sets `max_alibi_bias` to a constant value of 8.0.
- **Output**: No explicit output; the method initializes instance variables based on the provided hyperparameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
- **See also**: [`llama.cpp/convert_hf_to_gguf.JaisModel`](#cpp/convert_hf_to_ggufJaisModel)  (Base Class)


---
#### JaisModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.JaisModel.set_vocab}} -->
Sets the vocabulary for the model by invoking a specific method for GPT-2.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls the private method `_set_vocab_gpt2()` to perform the vocabulary setting.
- **Output**: The method does not return any value; it modifies the internal state of the model by setting the vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.JaisModel`](#cpp/convert_hf_to_ggufJaisModel)  (Base Class)


---
#### JaisModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.JaisModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `JaisModel` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the `add_block_count` method of `gguf_writer` with the number of layers from `hparams`.
    - Calls the `add_context_length` method of `gguf_writer` with the number of positions from `hparams`.
    - Calls the `add_embedding_length` method of `gguf_writer` with the embedding length from `hparams`.
    - Calls the `add_feed_forward_length` method of `gguf_writer` with the inner feed-forward length from `hparams`.
    - Calls the `add_head_count` method of `gguf_writer` with the number of heads from `hparams`.
    - Calls the `add_layer_norm_eps` method of `gguf_writer` with the layer normalization epsilon from `hparams`.
    - Calls the `add_file_type` method of `gguf_writer` with the file type stored in `ftype`.
- **Output**: No explicit return value; the method modifies the state of the `gguf_writer` by adding parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.JaisModel`](#cpp/convert_hf_to_ggufJaisModel)  (Base Class)


---
#### JaisModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.JaisModel.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on specific naming conventions and scaling factors.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name associated with the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method starts by initializing an empty list `tensors` to store the modified tensor tuples.
    - If the `name` ends with '.attn.bias', the method returns the empty `tensors` list immediately.
    - If the `name` ends with 'relative_pe.slopes', it calculates the maximum ALiBi bias and updates the instance variable `max_alibi_bias`, then returns the empty `tensors` list.
    - If the `name` ends with specific weight identifiers, the `data_torch` tensor is transposed.
    - The method maps the original `name` to a new name using [`map_tensor_name`](#ModelBasemap_tensor_name).
    - Based on the new name, the method scales the `data_torch` tensor accordingly and appends the result to the `tensors` list.
    - Finally, the method returns the `tensors` list containing the modified tensor tuples.
- **Output**: The method returns an iterable of tuples, each containing a string (the modified tensor name) and a `Tensor` (the modified tensor data).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.JaisModel`](#cpp/convert_hf_to_ggufJaisModel)  (Base Class)


---
#### JaisModel\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.JaisModel.prepare_tensors}} -->
The [`prepare_tensors`](#ModelBaseprepare_tensors) method prepares the model's tensors by invoking the parent class's method and adding the maximum ALiBi bias to the GGUF writer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `JaisModel` class, which contains model parameters and methods.
- **Control Flow**:
    - Calls the [`prepare_tensors`](#ModelBaseprepare_tensors) method from the parent class to perform any necessary tensor preparation defined there.
    - Adds the `max_alibi_bias` value to the `gguf_writer` to ensure it is included in the model's output.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` by adding the maximum ALiBi bias.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.JaisModel`](#cpp/convert_hf_to_ggufJaisModel)  (Base Class)



---
### Glm4Model<!-- {{#class:llama.cpp/convert_hf_to_gguf.Glm4Model}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture of the model as GLM4.
- **Description**: The `Glm4Model` class extends `TextModel` and is designed for causal language modeling, incorporating methods to set vocabulary and configure GGUF parameters for the model.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.Glm4Model.set_vocab`](#Glm4Modelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.Glm4Model.set_gguf_parameters`](#Glm4Modelset_gguf_parameters)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### Glm4Model\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Glm4Model.set_vocab}} -->
The `set_vocab` method initializes a tokenizer, retrieves vocabulary data, and updates a special vocabulary with specific tokens.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class `Glm4Model`, which contains model parameters and methods.
- **Control Flow**:
    - Imports the `AutoTokenizer` from the `transformers` library.
    - Initializes a tokenizer using a pre-trained model directory specified by `self.dir_model`.
    - Creates a `SpecialVocab` instance to manage special tokens.
    - Calls [`get_vocab_base`](#TextModelget_vocab_base) to retrieve a list of tokens, their types, and a prefix.
    - Adds the tokenizer model, prefix, token list, and token types to the `gguf_writer`.
    - Sets special tokens such as 'eos', 'eot', 'unk', and 'bos' using the vocabulary obtained from the tokenizer.
    - Finally, adds the special vocabulary to the `gguf_writer`.
- **Output**: The method does not return a value but updates the internal state of the `gguf_writer` with the tokenizer model, token lists, token types, and special tokens.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel.get_vocab_base`](#TextModelget_vocab_base)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Glm4Model`](#cpp/convert_hf_to_ggufGlm4Model)  (Base Class)


---
#### Glm4Model\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.Glm4Model.set_gguf_parameters}} -->
Sets parameters related to rotary embeddings for the GGUF model.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base parameters are set.
    - Retrieves the `head_dim` from the instance's hyperparameters to calculate the rope dimension count.
    - Adds the calculated rope dimension count to the GGUF writer.
    - Checks for the presence of `rope_scaling` parameters and if the type is 'yarn' with a defined factor.
    - If conditions are met, adds the rope scaling type, factor, and original context length to the GGUF writer.
- **Output**: No explicit return value; modifies the state of the GGUF writer with the set parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.Glm4Model`](#cpp/convert_hf_to_ggufGlm4Model)  (Base Class)



---
### ChatGLMModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.ChatGLMModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type for the ChatGLM model.
- **Description**: The `ChatGLMModel` class extends `TextModel` and is designed for handling the ChatGLM architecture, providing methods for setting vocabulary, managing tokenizer parameters, and modifying tensor data for model training and inference.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.ChatGLMModel.set_vocab_chatglm3`](#ChatGLMModelset_vocab_chatglm3)
    - [`llama.cpp/convert_hf_to_gguf.ChatGLMModel.token_bytes_to_string`](#ChatGLMModeltoken_bytes_to_string)
    - [`llama.cpp/convert_hf_to_gguf.ChatGLMModel.bpe`](#ChatGLMModelbpe)
    - [`llama.cpp/convert_hf_to_gguf.ChatGLMModel.set_vocab`](#ChatGLMModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.ChatGLMModel.set_gguf_parameters`](#ChatGLMModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.ChatGLMModel.modify_tensors`](#ChatGLMModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### ChatGLMModel\.set\_vocab\_chatglm3<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ChatGLMModel.set_vocab_chatglm3}} -->
Sets the vocabulary for the ChatGLM3 model by processing tokens and their types from a pretrained tokenizer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `ChatGLMModel` class, which contains model parameters and methods.
- **Control Flow**:
    - Initializes lists for tokens, token types, and scores.
    - Loads a tokenizer from the specified model directory.
    - Determines the vocabulary size and asserts it against the tokenizer's vocabulary.
    - Iterates over the range of vocabulary size to convert token IDs to their corresponding tokens.
    - Handles special cases for specific token IDs (0, 1, 2) to assign special tokens.
    - Checks if the token ID is valid and assigns scores and types based on the tokenizer's model.
    - Appends the processed tokens, scores, and types to their respective lists.
    - Adds the tokenizer model and processed data to the `gguf_writer` for further use.
- **Output**: No explicit return value; modifies the internal state of the `gguf_writer` with the processed vocabulary data.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ChatGLMModel`](#cpp/convert_hf_to_ggufChatGLMModel)  (Base Class)


---
#### ChatGLMModel\.bpe<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ChatGLMModel.bpe}} -->
The `bpe` method performs byte pair encoding on a given token using specified mergeable ranks.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `mergeable_ranks`: A dictionary mapping byte pairs to their corresponding merge ranks.
    - `token`: A byte sequence representing the token to be encoded.
    - `max_rank`: An optional integer specifying the maximum rank for merging; if provided, merging stops when the rank reaches or exceeds this value.
- **Control Flow**:
    - The method initializes `parts` as a list of individual bytes from the input `token`.
    - It enters a loop that continues until no more merges can be made based on the ranks.
    - Within the loop, it iterates over adjacent byte pairs to find the pair with the lowest merge rank.
    - If a valid merge is found and it does not exceed `max_rank`, the bytes are merged into a single byte.
    - The loop breaks when no valid merges are found or the minimum rank exceeds `max_rank`.
- **Output**: The method returns a list of bytes representing the token after applying byte pair encoding.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ChatGLMModel`](#cpp/convert_hf_to_ggufChatGLMModel)  (Base Class)


---
#### ChatGLMModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ChatGLMModel.set_vocab}} -->
The `set_vocab` method initializes the vocabulary for a model based on its configuration and tokenizer.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class containing the method, which holds model parameters and configurations.
- **Control Flow**:
    - Checks if the model name contains 'THUDM/chatglm3-6b' and calls [`set_vocab_chatglm3`](#ChatGLMModelset_vocab_chatglm3) if true.
    - Initializes the tokenizer using the model directory and retrieves the vocabulary size.
    - Asserts that the maximum vocabulary index from the tokenizer is less than the specified vocabulary size.
    - Calls [`get_vocab_base`](#TextModelget_vocab_base) to retrieve the base vocabulary tokens and types.
    - Adds the tokenizer model, pre-tokenizer, token list, and token types to the `gguf_writer`.
    - Creates a `SpecialVocab` instance and sets special tokens based on the tokenizer's added vocabulary.
    - Adds the special vocabulary to the `gguf_writer`.
- **Output**: The method does not return a value but modifies the internal state of the model by setting up the vocabulary and special tokens for use in the model.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ChatGLMModel.set_vocab_chatglm3`](#ChatGLMModelset_vocab_chatglm3)
    - [`llama.cpp/convert_hf_to_gguf.TextModel.get_vocab_base`](#TextModelget_vocab_base)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ChatGLMModel`](#cpp/convert_hf_to_ggufChatGLMModel)  (Base Class)


---
#### ChatGLMModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ChatGLMModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF writer based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class containing hyperparameters and a GGUF writer.
- **Control Flow**:
    - Retrieves various hyperparameters from `self.hparams` with fallback options.
    - Calls methods on `self.gguf_writer` to set context length, embedding length, feed-forward length, block count, head count, key-value head count, layer normalization epsilon, file type, rope dimension count, and rope frequency base.
    - Checks for the presence of 'attention_dim' and 'rope_ratio' in `self.hparams` to calculate `rope_dim` and `rope_freq` respectively.
- **Output**: The method does not return a value; it modifies the state of the GGUF writer with the specified parameters.
- **See also**: [`llama.cpp/convert_hf_to_gguf.ChatGLMModel`](#cpp/convert_hf_to_ggufChatGLMModel)  (Base Class)


---
#### ChatGLMModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ChatGLMModel.modify_tensors}} -->
The `modify_tensors` method processes a tensor based on its name and returns a modified representation if certain conditions are met.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name associated with the tensor.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method begins by deleting the `bid` parameter as it is unused.
    - It checks if the `name` ends with '.rotary_pos_emb.inv_freq' or starts with 'model.vision.'; if so, it returns an empty list.
    - If the `name` does not meet the above conditions, it removes the prefix 'transformer.' from the `name` and returns a list containing a tuple of the mapped tensor name and the original tensor.
- **Output**: The output is an iterable of tuples, each containing a string (the mapped tensor name) and the original `Tensor` object.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ChatGLMModel`](#cpp/convert_hf_to_ggufChatGLMModel)  (Base Class)



---
### NemotronModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.NemotronModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as Nemotron.
- **Description**: The `NemotronModel` class extends `TextModel` and is designed for causal language modeling, incorporating specific methods for setting vocabulary and GGUF parameters, as well as modifying tensor weights for layer normalization.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.NemotronModel.set_vocab`](#NemotronModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.NemotronModel.set_gguf_parameters`](#NemotronModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.NemotronModel.modify_tensors`](#NemotronModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### NemotronModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.NemotronModel.set_vocab}} -->
Sets the vocabulary for the `NemotronModel` by configuring special tokens and initializing the sentencepiece vocabulary.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `NemotronModel` class.
- **Control Flow**:
    - Calls the [`_set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece) method to initialize the sentencepiece vocabulary.
    - Adds a padding token ID with a value of 0 to the `gguf_writer`.
    - Adds an unknown token ID with a value of 1 to the `gguf_writer`.
- **Output**: The method does not return any value; it modifies the internal state of the `gguf_writer` to include vocabulary settings.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_sentencepiece`](#TextModel_set_vocab_sentencepiece)
- **See also**: [`llama.cpp/convert_hf_to_gguf.NemotronModel`](#cpp/convert_hf_to_ggufNemotronModel)  (Base Class)


---
#### NemotronModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.NemotronModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters defined in the model.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer for adding model parameters.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to initialize base parameters.
    - Retrieves the vocabulary size from `hparams` and adds it to the GGUF writer.
    - Finds and adds the layer normalization epsilon value to the GGUF writer.
    - Calculates the rope dimension count based on the partial rotary factor, hidden size, and number of attention heads, and adds it to the GGUF writer.
    - Checks if rope scaling is defined in `hparams`; if not, sets it to NONE, otherwise sets it to LINEAR and adds the scaling factor.
- **Output**: No explicit return value; modifies the state of the GGUF writer with the specified parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.NemotronModel`](#cpp/convert_hf_to_ggufNemotronModel)  (Base Class)


---
#### NemotronModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.NemotronModel.modify_tensors}} -->
The `modify_tensors` method adjusts the weights of LayerNorm tensors by incrementing them by one if their name ends with 'norm.weight'.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the weights to be modified.
    - `name`: A string representing the name of the tensor, which is used to determine if the tensor is a LayerNorm weight.
    - `bid`: An optional integer that may represent a batch ID, though it is not used in the method.
- **Control Flow**:
    - The method checks if the `name` ends with 'norm.weight'.
    - If the condition is true, it increments the `data_torch` tensor by 1.
    - Finally, it returns a list containing a tuple of the mapped tensor name and the modified tensor.
- **Output**: An iterable of tuples, where each tuple contains a string (the mapped tensor name) and the modified `Tensor` object.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.NemotronModel`](#cpp/convert_hf_to_ggufNemotronModel)  (Base Class)



---
### ExaoneModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.ExaoneModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
- **Description**: The `ExaoneModel` class extends `TextModel` and is designed for causal language modeling, incorporating specific parameters and configurations for the EXAONE architecture, including methods for setting GGUF parameters and generating additional tensors based on hyperparameters.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.ExaoneModel.set_gguf_parameters`](#ExaoneModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.ExaoneModel.generate_extra_tensors`](#ExaoneModelgenerate_extra_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### ExaoneModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ExaoneModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Asserts that the activation function specified in hyperparameters is 'silu'.
    - Retrieves various hyperparameters such as maximum position embeddings, embedding dimension, number of heads, and layer normalization epsilon.
    - Adds parameters to the GGUF writer including embedding length, head count, context length, and feed-forward length.
    - Checks for the presence of 'rope_theta' in hyperparameters and adds it to the GGUF writer if available.
    - Calculates the rotary factor and adds the corresponding rope dimension count to the GGUF writer.
    - Checks for rope scaling type and factor, and adds them to the GGUF writer if the conditions are met.
- **Output**: No explicit return value; modifies the state of the GGUF writer with the set parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ExaoneModel`](#cpp/convert_hf_to_ggufExaoneModel)  (Base Class)


---
#### ExaoneModel\.generate\_extra\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ExaoneModel.generate_extra_tensors}} -->
Generates additional tensors for rotary embeddings based on specified hyperparameters.
- **Inputs**: None
- **Control Flow**:
    - Checks for the presence of 'rope_scaling' hyperparameter.
    - If 'rope_type' is 'llama3', it retrieves various parameters such as 'rope_theta', 'head_dim', and frequency factors.
    - Calculates wavelengths and determines rope factors based on frequency conditions.
    - Yields a tuple containing the formatted tensor name and the computed tensor of rope factors.
- **Output**: Yields a tuple consisting of a string representing the tensor name and a tensor containing the calculated rope factors.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.find_hparam`](#ModelBasefind_hparam)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ExaoneModel`](#cpp/convert_hf_to_ggufExaoneModel)  (Base Class)



---
### GraniteModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.GraniteModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the model architecture for Granite.
- **Description**: The `GraniteModel` class is a specialized model for IBM's GraniteForCausalLM, extending the `LlamaModel` and incorporating specific parameter handling for the Granite architecture.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.GraniteModel.set_gguf_parameters`](#GraniteModelset_gguf_parameters)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel`](#cpp/convert_hf_to_ggufLlamaModel)

**Methods**

---
#### GraniteModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.GraniteModel.set_gguf_parameters}} -->
Sets GGUF parameters for the Granite model, handling specific parameter adjustments and logging.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `GraniteModel` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Checks for and removes the 'head_dim' parameter from `hparams`, logging a warning if it exists.
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to handle standard parameter settings.
    - Retrieves and processes the 'attention_multiplier', 'embedding_multiplier', 'residual_multiplier', and 'logits_scaling' parameters, logging their values and adding them to the GGUF writer.
- **Output**: The method does not return a value but modifies the state of the GGUF writer and logs relevant information.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.GraniteModel`](#cpp/convert_hf_to_ggufGraniteModel)  (Base Class)



---
### GraniteMoeModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.GraniteMoeModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the model architecture as GRANITE_MOE.
- **Description**: The `GraniteMoeModel` class extends `GraniteModel` and is designed for the conversion of IBM's `GraniteMoeForCausalLM`, incorporating specific parameters and tensor modifications for handling model architecture.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.GraniteMoeModel.set_gguf_parameters`](#GraniteMoeModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.GraniteMoeModel.modify_tensors`](#GraniteMoeModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.GraniteModel`](#cpp/convert_hf_to_ggufGraniteModel)

**Methods**

---
#### GraniteMoeModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.GraniteMoeModel.set_gguf_parameters}} -->
Sets GGUF parameters specific to the `GraniteMoeShared` model, including the shared feed-forward length.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains the method, which holds model parameters and configurations.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to initialize base parameters.
    - Checks if `shared_intermediate_size` is present in the model's hyperparameters.
    - If `shared_intermediate_size` is found, it adds the corresponding shared feed-forward length to the GGUF writer.
    - Logs the value of `shared_feed_forward_length` for debugging purposes.
- **Output**: No explicit return value; modifies the state of the GGUF writer and logs information.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.GraniteMoeModel`](#cpp/convert_hf_to_ggufGraniteMoeModel)  (Base Class)


---
#### GraniteMoeModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.GraniteMoeModel.modify_tensors}} -->
The [`modify_tensors`](#ModelBasemodify_tensors) method splits a merged tensor into its components based on the specified weight name.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` that contains the merged weights to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that may be used to identify the batch or instance.
- **Control Flow**:
    - Checks if the `name` ends with 'block_sparse_moe.input_linear.weight' to determine the processing path.
    - Retrieves the `intermediate_size` from `self.hparams` and asserts that the last dimension of `data_torch` is twice this size.
    - Splits `data_torch` into two tensors: `gate` and `up`, and formats their names before returning them as a list of tuples.
    - Checks if the `name` ends with 'shared_mlp.input_linear.weight' and follows a similar process using `shared_intermediate_size`.
    - If neither condition is met, it calls the parent class's [`modify_tensors`](#ModelBasemodify_tensors) method.
- **Output**: Returns an iterable of tuples, each containing a formatted tensor name and the corresponding tensor (either `gate` or `up`), or calls the parent method if no conditions are met.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.modify_tensors`](#ModelBasemodify_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.GraniteMoeModel`](#cpp/convert_hf_to_ggufGraniteMoeModel)  (Base Class)



---
### BailingMoeModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.BailingMoeModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model.
    - `_experts`: Holds a list of expert tensors or None.
- **Description**: The `BailingMoeModel` class extends `TextModel` and is designed for a specific model architecture, providing methods to set vocabulary and configure parameters for a mixture of experts model, while managing tensor modifications and expert handling.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.BailingMoeModel.set_vocab`](#BailingMoeModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.BailingMoeModel.set_gguf_parameters`](#BailingMoeModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.BailingMoeModel.permute`](#BailingMoeModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.BailingMoeModel.modify_tensors`](#BailingMoeModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.BailingMoeModel.prepare_tensors`](#BailingMoeModelprepare_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### BailingMoeModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BailingMoeModel.set_vocab}} -->
Sets the vocabulary for the model by invoking a specific method to configure it for GPT-2.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `BailingMoeModel` class, which contains the model's parameters and methods.
- **Control Flow**:
    - Calls the [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) method to configure the vocabulary for the model.
- **Output**: No output is returned; the method modifies the internal state of the model by setting its vocabulary.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BailingMoeModel`](#cpp/convert_hf_to_ggufBailingMoeModel)  (Base Class)


---
#### BailingMoeModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BailingMoeModel.set_gguf_parameters}} -->
Sets various parameters for the GGUF model based on hyperparameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `BailingMoeModel` class, which contains hyperparameters and a GGUF writer.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to initialize base parameters.
    - Retrieves hyperparameters from `self.hparams`.
    - Calculates the `rope_dim` based on `head_dim` or derived from `hidden_size` and `num_attention_heads`.
    - Adds the calculated `rope_dim` to the GGUF writer.
    - Checks for `rope_scaling` parameters and adds corresponding scaling type and factors to the GGUF writer.
    - Adds various model parameters such as `first_k_dense_replace`, `vocab_size`, `moe_intermediate_size`, `num_experts`, `num_shared_experts`, and `norm_topk_prob` to the GGUF writer.
- **Output**: No explicit return value; modifies the state of the `gguf_writer` with the set parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BailingMoeModel`](#cpp/convert_hf_to_ggufBailingMoeModel)  (Base Class)


---
#### BailingMoeModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BailingMoeModel.modify_tensors}} -->
The `modify_tensors` method processes and modifies tensor data based on specific naming conventions and parameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object containing the data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be processed.
    - `bid`: An optional integer that serves as an identifier for the batch or block of data.
- **Control Flow**:
    - The method retrieves hyperparameters such as the number of attention heads and hidden size from `self.hparams`.
    - It checks the tensor name to determine how to process the input tensor, handling cases for attention weights, query-key-value weights, and MLP expert weights.
    - For query-key-value weights, it splits the tensor into query, key, and value components and permutes them accordingly.
    - If the tensor name indicates MLP experts, it collects and merges expert weights into a single tensor when the required number of experts is reached.
    - Finally, it normalizes the tensor if it matches the output name and normalization is enabled, before returning the modified tensor with its new name.
- **Output**: Returns an iterable of tuples, each containing a modified tensor name and the corresponding modified tensor.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.format_tensor_name`](#ModelBaseformat_tensor_name)
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BailingMoeModel`](#cpp/convert_hf_to_ggufBailingMoeModel)  (Base Class)


---
#### BailingMoeModel\.prepare\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.BailingMoeModel.prepare_tensors}} -->
The [`prepare_tensors`](#ModelBaseprepare_tensors) method prepares the model's tensors and checks for any unprocessed expert tensors.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - Calls the parent class's [`prepare_tensors`](#ModelBaseprepare_tensors) method to perform any necessary setup.
    - Checks if the `_experts` attribute is not None, indicating that expert tensors are being used.
    - Flattens the list of dictionaries in `_experts` to extract the keys (expert names).
    - If there are any unprocessed expert names, raises a ValueError with the list of these names.
- **Output**: The method does not return a value; it raises an exception if there are unprocessed expert tensors.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.prepare_tensors`](#ModelBaseprepare_tensors)
- **See also**: [`llama.cpp/convert_hf_to_gguf.BailingMoeModel`](#cpp/convert_hf_to_ggufBailingMoeModel)  (Base Class)



---
### ChameleonModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.ChameleonModel}} -->
- **Decorators**: `@ModelBase.register`, `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the architecture type of the model as Chameleon.
- **Description**: The `ChameleonModel` class extends `TextModel` and is designed for conditional generation and causal language modeling, incorporating specific methods for setting parameters and modifying tensor data, while also defining its model architecture.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.ChameleonModel.set_gguf_parameters`](#ChameleonModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.ChameleonModel.set_vocab`](#ChameleonModelset_vocab)
    - [`llama.cpp/convert_hf_to_gguf.ChameleonModel.modify_tensors`](#ChameleonModelmodify_tensors)
    - [`llama.cpp/convert_hf_to_gguf.ChameleonModel._reverse_hf_permute`](#ChameleonModel_reverse_hf_permute)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### ChameleonModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ChameleonModel.set_gguf_parameters}} -->
Sets the GGUF parameters for the Chameleon model by invoking the parent method and adding a specific normalization parameter.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `ChameleonModel` class.
- **Control Flow**:
    - Calls the [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method of the parent class using `super()` to ensure any base class initialization is performed.
    - Retrieves the value of the 'swin_norm' parameter from the model's hyperparameters (`hparams`) with a default of False if not set.
    - Adds the 'swin_norm' parameter to the `gguf_writer` using the `add_swin_norm` method.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` by adding the 'swin_norm' parameter.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ChameleonModel`](#cpp/convert_hf_to_ggufChameleonModel)  (Base Class)


---
#### ChameleonModel\.set\_vocab<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ChameleonModel.set_vocab}} -->
The `set_vocab` method initializes the vocabulary for the Chameleon model by invoking the [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) method.
- **Decorators**: `@ModelBase.register`
- **Inputs**: None
- **Control Flow**:
    - The method directly calls the [`_set_vocab_gpt2`](#TextModel_set_vocab_gpt2) method without any conditional logic or iterations.
- **Output**: The method does not return any value; it is intended to set up the vocabulary for the model internally.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel._set_vocab_gpt2`](#TextModel_set_vocab_gpt2)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ChameleonModel`](#cpp/convert_hf_to_ggufChameleonModel)  (Base Class)


---
#### ChameleonModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.ChameleonModel.modify_tensors}} -->
The `modify_tensors` method adjusts tensor data based on the specified name and model parameters.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object representing the data to be modified.
    - `name`: A string representing the name of the tensor, which determines how the tensor will be modified.
    - `bid`: An optional integer that may represent a batch ID, though it is not used in the current implementation.
- **Control Flow**:
    - The method first checks if the `name` starts with 'model.vqmodel', returning an empty list if true.
    - It retrieves the number of attention heads, key-value heads, and hidden dimension from the model's hyperparameters.
    - Based on the suffix of the `name`, it conditionally permutes the `data_torch` tensor using either `LlamaModel.permute` or `ChameleonModel._reverse_hf_permute`.
    - Finally, it returns a list containing a tuple of the mapped tensor name and the modified tensor.
- **Output**: The method outputs an iterable of tuples, each containing a string (the mapped tensor name) and a modified `Tensor` object.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.LlamaModel.permute`](#LlamaModelpermute)
    - [`llama.cpp/convert_hf_to_gguf.BaichuanModel._reverse_hf_permute`](#BaichuanModel_reverse_hf_permute)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.ChameleonModel`](#cpp/convert_hf_to_ggufChameleonModel)  (Base Class)



---
### UltravoxModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.UltravoxModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `model_arch`: Specifies the model architecture as LLAMA.
- **Description**: The `UltravoxModel` class extends `TextModel` and is designed to register a specific model architecture, but it raises a `NotImplementedError` in its constructor to indicate that it does not support text decoding.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.UltravoxModel.__init__`](#UltravoxModel__init__)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.TextModel`](#cpp/convert_hf_to_ggufTextModel)

**Methods**

---
#### UltravoxModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.UltravoxModel.__init__}} -->
Initializes an instance of the `UltravoxModel` class, which raises a `NotImplementedError` indicating the absence of a text decoder.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `*args`: Variable length argument list that can accept any number of positional arguments.
    - `**kwargs`: Variable length keyword argument dictionary that can accept any number of keyword arguments.
- **Control Flow**:
    - Calls the [`__init__`](#ModelBase__init__) method of the parent class `TextModel` with the provided arguments and keyword arguments.
    - Immediately raises a `NotImplementedError` with a specific message regarding the lack of a text decoder in the `UltravoxModel`.
- **Output**: This method does not return a value; instead, it raises an exception to indicate that the functionality is not implemented.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
- **See also**: [`llama.cpp/convert_hf_to_gguf.UltravoxModel`](#cpp/convert_hf_to_ggufUltravoxModel)  (Base Class)



---
### WhisperEncoderModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.WhisperEncoderModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `has_vision_encoder`: Indicates that the model does not have a vision encoder.
    - `has_audio_encoder`: Indicates that the model has an audio encoder.
    - `hparams`: Holds hyperparameters for the model configuration.
- **Description**: The `WhisperEncoderModel` class is a specialized model for audio processing that inherits from `MmprojModel`, designed to handle audio encoding without a vision component, and includes methods for setting parameters and modifying tensor data.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.WhisperEncoderModel.__init__`](#WhisperEncoderModel__init__)
    - [`llama.cpp/convert_hf_to_gguf.WhisperEncoderModel.set_gguf_parameters`](#WhisperEncoderModelset_gguf_parameters)
    - [`llama.cpp/convert_hf_to_gguf.WhisperEncoderModel.tensor_force_quant`](#WhisperEncoderModeltensor_force_quant)
    - [`llama.cpp/convert_hf_to_gguf.WhisperEncoderModel.modify_tensors`](#WhisperEncoderModelmodify_tensors)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.MmprojModel`](#cpp/convert_hf_to_ggufMmprojModel)

**Methods**

---
#### WhisperEncoderModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/convert_hf_to_gguf.WhisperEncoderModel.__init__}} -->
Initializes the `WhisperEncoderModel` by setting specific hyperparameters based on the provided arguments.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `*args`: Variable length argument list for additional parameters to be passed to the parent class.
    - `**kwargs`: Keyword arguments for additional parameters to be passed to the parent class.
- **Control Flow**:
    - Calls the parent class's [`__init__`](#ModelBase__init__) method to ensure proper initialization of inherited attributes.
    - Sets the `hidden_size`, `intermediate_size`, and `num_attention_heads` in the `hparams` dictionary based on existing hyperparameters.
- **Output**: No explicit output; the method initializes the instance's state by modifying the `hparams` attribute.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.__init__`](#ModelBase__init__)
- **See also**: [`llama.cpp/convert_hf_to_gguf.WhisperEncoderModel`](#cpp/convert_hf_to_ggufWhisperEncoderModel)  (Base Class)


---
#### WhisperEncoderModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.WhisperEncoderModel.set_gguf_parameters}} -->
Sets specific parameters for the GGUF writer related to audio and vision projector types.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the class that contains the method.
- **Control Flow**:
    - Calls the parent class's [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method to ensure any base class parameters are set.
    - Adds a specific vision projector type (QWEN2A) to the GGUF writer.
    - Adds the number of mel bins for audio from the instance's hyperparameters.
    - Adds the layer normalization epsilon value for audio, defaulting to 1e-5 if not specified in hyperparameters.
- **Output**: The method does not return a value; it modifies the state of the `gguf_writer` by adding parameters.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.WhisperEncoderModel`](#cpp/convert_hf_to_ggufWhisperEncoderModel)  (Base Class)


---
#### WhisperEncoderModel\.tensor\_force\_quant<!-- {{#callable:llama.cpp/convert_hf_to_gguf.WhisperEncoderModel.tensor_force_quant}} -->
The `tensor_force_quant` method determines the quantization type for a tensor based on its name.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `name`: The name of the tensor, which is used to determine its quantization type.
    - `new_name`: An unused parameter that is not utilized in the method.
    - `bid`: An unused parameter that is not utilized in the method.
    - `n_dims`: An unused parameter that is not utilized in the method.
- **Control Flow**:
    - The method first deletes the unused parameters `bid`, `new_name`, and `n_dims`.
    - It checks if the `name` contains both '.conv' and '.weight'.
    - If the condition is met, it returns the quantization type `F16` from `gguf.GGMLQuantizationType`.
    - If the condition is not met, it returns `False`.
- **Output**: The output is either the quantization type `F16` if the conditions are satisfied, or `False` if they are not.
- **See also**: [`llama.cpp/convert_hf_to_gguf.WhisperEncoderModel`](#cpp/convert_hf_to_ggufWhisperEncoderModel)  (Base Class)


---
#### WhisperEncoderModel\.modify\_tensors<!-- {{#callable:llama.cpp/convert_hf_to_gguf.WhisperEncoderModel.modify_tensors}} -->
The `modify_tensors` method processes a tensor based on its name, modifying it if necessary and returning a mapped name with the tensor.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `data_torch`: A `Tensor` object that represents the data to be modified.
    - `name`: A string representing the name associated with the tensor.
    - `bid`: An optional integer that is not used in the method.
- **Control Flow**:
    - The method begins by deleting the unused `bid` parameter.
    - It checks if the `name` starts with 'language_model.' and returns an empty list if true, skipping further processing.
    - If the `name` starts with 'multi_modal_projector', it prefixes 'audio.' to the `name` to avoid naming conflicts.
    - If the `name` contains 'conv1.bias' or 'conv2.bias', it modifies the `data_torch` tensor by adding an extra dimension using `unsqueeze`.
    - Finally, it returns a list containing a tuple of the mapped name and the possibly modified tensor.
- **Output**: The method outputs an iterable of tuples, each containing a string (the mapped tensor name) and a `Tensor` (the modified tensor).
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.map_tensor_name`](#ModelBasemap_tensor_name)
- **See also**: [`llama.cpp/convert_hf_to_gguf.WhisperEncoderModel`](#cpp/convert_hf_to_ggufWhisperEncoderModel)  (Base Class)



---
### UltravoxWhisperEncoderModel<!-- {{#class:llama.cpp/convert_hf_to_gguf.UltravoxWhisperEncoderModel}} -->
- **Decorators**: `@ModelBase.register`
- **Members**:
    - `has_vision_encoder`: Indicates that this model does not include a vision encoder.
    - `has_audio_encoder`: Indicates that this model includes an audio encoder.
- **Description**: The `UltravoxWhisperEncoderModel` class extends the `WhisperEncoderModel` and is designed to handle audio encoding without a vision encoder, providing specific functionality for audio processing.
- **Methods**:
    - [`llama.cpp/convert_hf_to_gguf.UltravoxWhisperEncoderModel.set_gguf_parameters`](#UltravoxWhisperEncoderModelset_gguf_parameters)
- **Inherits From**:
    - [`llama.cpp/convert_hf_to_gguf.WhisperEncoderModel`](#cpp/convert_hf_to_ggufWhisperEncoderModel)

**Methods**

---
#### UltravoxWhisperEncoderModel\.set\_gguf\_parameters<!-- {{#callable:llama.cpp/convert_hf_to_gguf.UltravoxWhisperEncoderModel.set_gguf_parameters}} -->
Sets GGUF parameters for the `UltravoxWhisperEncoderModel` by invoking the parent method and adding an audio stack factor.
- **Decorators**: `@ModelBase.register`
- **Inputs**:
    - `self`: An instance of the `UltravoxWhisperEncoderModel` class.
- **Control Flow**:
    - Calls the [`set_gguf_parameters`](#ModelBaseset_gguf_parameters) method of the parent class using `super()`.
    - Adds an audio stack factor to the `gguf_writer` using a value from `global_config`.
- **Output**: No explicit output; modifies the state of the `gguf_writer` by adding an audio stack factor.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.set_gguf_parameters`](#ModelBaseset_gguf_parameters)
- **See also**: [`llama.cpp/convert_hf_to_gguf.UltravoxWhisperEncoderModel`](#cpp/convert_hf_to_ggufUltravoxWhisperEncoderModel)  (Base Class)



# Functions

---
### parse\_args<!-- {{#callable:llama.cpp/convert_hf_to_gguf.parse_args}} -->
The `parse_args` function parses command-line arguments for converting a Hugging Face model to a GGML compatible file.
- **Inputs**: None
- **Control Flow**:
    - An `ArgumentParser` object is created with a description of the program's purpose.
    - Multiple arguments are added to the parser, each with specific options such as type, default values, and help descriptions.
    - The `parse_args` method is called on the parser to parse the command-line arguments into a `Namespace` object.
    - A conditional check ensures that if the `--print-supported-models` flag is not set and no model is provided, an error is raised indicating that the model argument is required.
- **Output**: The function returns an `argparse.Namespace` object containing the parsed command-line arguments.


---
### split\_str\_to\_n\_bytes<!-- {{#callable:llama.cpp/convert_hf_to_gguf.split_str_to_n_bytes}} -->
The function `split_str_to_n_bytes` converts a string representing a size with optional units (K, M, G) into an integer number of bytes.
- **Inputs**:
    - `split_str`: A string representing a size, which may end with 'K', 'M', or 'G' to denote kilobytes, megabytes, or gigabytes, respectively, or be a plain numeric string.
- **Control Flow**:
    - Check if the input string ends with 'K', 'M', or 'G' and convert the numeric part to bytes by multiplying with the respective power of 1000.
    - If the string is purely numeric, convert it directly to an integer.
    - If the string does not match any of the expected formats, raise a ValueError indicating an invalid split size.
    - After conversion, check if the resulting number of bytes is negative and raise a ValueError if it is.
    - Return the computed number of bytes as an integer.
- **Output**: An integer representing the size in bytes.


---
### get\_model\_architecture<!-- {{#callable:llama.cpp/convert_hf_to_gguf.get_model_architecture}} -->
The function `get_model_architecture` determines the architecture of a model based on the provided hyperparameters and model type.
- **Inputs**:
    - `hparams`: A dictionary containing hyperparameters, including potential sub-configurations for text and vision models.
    - `model_type`: An instance of the `ModelType` enumeration indicating the type of model (e.g., TEXT, MMPROJ).
- **Control Flow**:
    - Initialize `text_config` and `vision_config` from `hparams` with default empty dictionaries if not present.
    - Set `arch` to the first element of the `architectures` list from `hparams`.
    - Check if `model_type` is `ModelType.TEXT` and if `text_config` contains an `architectures` key; if so, update `arch` to the first element of `text_config['architectures']`.
    - Check if `model_type` is `ModelType.MMPROJ` and if `vision_config` contains an `architectures` key; if so, update `arch` to the first element of `vision_config['architectures']`.
    - Return the determined architecture `arch`.
- **Output**: A string representing the architecture of the model based on the given hyperparameters and model type.


---
### main<!-- {{#callable:llama.cpp/convert_hf_to_gguf.main}} -->
The `main` function orchestrates the process of loading, configuring, and exporting a machine learning model based on command-line arguments.
- **Inputs**: None
- **Control Flow**:
    - Parse command-line arguments using `parse_args()`.
    - Check if `print_supported_models` flag is set; if so, print supported models and exit.
    - Set logging level based on `verbose` flag.
    - Handle remote model download if `remote` flag is set, updating `dir_model`.
    - Verify `dir_model` is a directory, otherwise log error and exit.
    - Map output file type strings to `gguf.LlamaFileType` using `ftype_map`.
    - Check for incompatible `use_temp_file` and splitting options, log error and exit if both are set.
    - Determine output file name based on `outfile` argument and `remote` flag.
    - Log the model being loaded.
    - Add prefix to output file name if `mmproj` flag is set.
    - Enter a `torch.inference_mode()` context for model operations.
    - Determine `output_type` and `model_type` based on arguments.
    - Load model hyperparameters and architecture, logging the architecture.
    - Attempt to instantiate model class from architecture, log error and exit if unsupported.
    - Create a model instance with specified parameters.
    - If `vocab_only` flag is set, export model vocabulary and log success.
    - Otherwise, export the model and log the output path.
- **Output**: The function does not return any value; it performs operations such as logging, downloading, and exporting models, and exits the program on certain conditions.
- **Functions called**:
    - [`llama.cpp/convert_hf_to_gguf.parse_args`](#cpp/convert_hf_to_ggufparse_args)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.print_registered_models`](#ModelBaseprint_registered_models)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.add_prefix_to_filename`](#ModelBaseadd_prefix_to_filename)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.load_hparams`](#ModelBaseload_hparams)
    - [`llama.cpp/convert_hf_to_gguf.get_model_architecture`](#cpp/convert_hf_to_ggufget_model_architecture)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.from_model_architecture`](#ModelBasefrom_model_architecture)
    - [`llama.cpp/convert_hf_to_gguf.split_str_to_n_bytes`](#cpp/convert_hf_to_ggufsplit_str_to_n_bytes)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.write_vocab`](#ModelBasewrite_vocab)
    - [`llama.cpp/convert_hf_to_gguf.ModelBase.write`](#ModelBasewrite)


