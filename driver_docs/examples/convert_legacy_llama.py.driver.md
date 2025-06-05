# Purpose
The provided Python script is a comprehensive tool designed to convert LLaMA models into a GGUF (General Graph Universal Format) compatible file. This script is intended for use in machine learning and natural language processing applications, where model conversion and optimization are crucial for deployment and performance tuning. The script supports various input formats, including PyTorch and SafeTensors, and can handle multi-file models. It offers functionality to extract and convert model parameters, manage vocabulary, and apply quantization techniques to optimize model size and performance.

Key components of the script include the use of data classes to define data types and parameters, lazy loading mechanisms to efficiently handle large model files, and parallel processing to speed up conversion tasks. The script also provides options for users to specify output formats, handle vocabulary separately, and manage metadata for model provenance. The script is designed to be flexible and extensible, allowing users to customize the conversion process through command-line arguments, such as specifying the output file type, concurrency level, and metadata overrides. Overall, this script is a powerful tool for researchers and engineers working with LLaMA models, enabling efficient conversion and deployment in various environments.
# Imports and Dependencies

---
- `__future__.annotations`
- `logging`
- `argparse`
- `concurrent.futures`
- `enum`
- `faulthandler`
- `functools`
- `itertools`
- `json`
- `math`
- `mmap`
- `os`
- `pickle`
- `re`
- `signal`
- `struct`
- `sys`
- `textwrap`
- `time`
- `zipfile`
- `abc.ABC`
- `abc.abstractmethod`
- `concurrent.futures.ProcessPoolExecutor`
- `concurrent.futures.ThreadPoolExecutor`
- `dataclasses.dataclass`
- `pathlib.Path`
- `typing.TYPE_CHECKING`
- `typing.Any`
- `typing.Callable`
- `typing.IO`
- `typing.Iterable`
- `typing.Literal`
- `typing.TypeVar`
- `numpy`
- `gguf`
- `gguf.BaseVocab`
- `gguf.Vocab`
- `gguf.NoVocab`
- `gguf.BpeVocab`
- `gguf.SentencePieceVocab`
- `gguf.LlamaHfVocab`
- `typing_extensions.Self`
- `typing_extensions.TypeAlias`


# Global Variables

---
### logger
- **Type**: `logging.Logger`
- **Description**: The `logger` variable is an instance of the `Logger` class from the `logging` module, configured to handle logging for the 'convert' component of the application. This logger is used to record log messages, which can include information, warnings, errors, and debugging messages, to help track the application's execution and diagnose issues.
- **Use**: This variable is used to log messages related to the 'convert' process, aiding in debugging and monitoring.


---
### NDArray
- **Type**: `TypeAlias`
- **Description**: `NDArray` is a type alias for `np.ndarray[Any, Any]`, which represents a NumPy array with any shape and any data type. This alias is used to simplify type annotations in the code.
- **Use**: This variable is used to annotate variables or function parameters that are expected to be NumPy arrays.


---
### ARCH
- **Type**: `gguf.MODEL_ARCH`
- **Description**: The variable `ARCH` is a global variable that is assigned the value `gguf.MODEL_ARCH.LLAMA`. This indicates that the architecture being used or referenced in the code is the 'LLAMA' model architecture as defined in the `gguf` module.
- **Use**: This variable is used to specify the model architecture type throughout the code, likely influencing how the model is processed or converted.


---
### DEFAULT\_CONCURRENCY
- **Type**: `int`
- **Description**: `DEFAULT_CONCURRENCY` is a global variable set to the integer value 8. It represents the default level of concurrency for operations that can be executed in parallel.
- **Use**: This variable is used to specify the default number of concurrent threads or processes for parallel execution tasks.


---
### ADDED\_TOKENS\_FILE
- **Type**: `str`
- **Description**: The `ADDED_TOKENS_FILE` variable is a string that holds the filename 'added_tokens.json'. This file is likely used to store additional tokens that are not part of the standard vocabulary in a tokenizer setup.
- **Use**: This variable is used to specify the filename for storing or accessing added tokens in a JSON format.


---
### FAST\_TOKENIZER\_FILE
- **Type**: `str`
- **Description**: `FAST_TOKENIZER_FILE` is a string variable that holds the filename 'tokenizer.json'. This filename is likely used to reference a JSON file containing tokenizer configurations or data.
- **Use**: This variable is used to specify the filename for a tokenizer configuration file, which can be accessed or modified in the program.


---
### DT\_F16
- **Type**: `UnquantizedDataType`
- **Description**: `DT_F16` is an instance of the `UnquantizedDataType` class, representing a data type with the name 'F16'. It uses the NumPy data type `np.float16` and allows valid conversions to 'F32' and 'Q8_0'. This data type is used to handle 16-bit floating-point numbers in the context of the software.
- **Use**: This variable is used to define and manage 16-bit floating-point data types, particularly for operations that may require conversion to other data types like 'F32' or 'Q8_0'.


---
### DT\_F32
- **Type**: `UnquantizedDataType`
- **Description**: `DT_F32` is an instance of the `UnquantizedDataType` class, representing a data type with the name 'F32'. It uses the NumPy data type `np.float32` and allows valid conversions to 'F16' and 'Q8_0'. This data type is used for handling unquantized floating-point data in 32-bit precision.
- **Use**: This variable is used to define and manage unquantized 32-bit floating-point data types, including their valid conversions.


---
### DT\_I32
- **Type**: `UnquantizedDataType`
- **Description**: `DT_I32` is an instance of the `UnquantizedDataType` class, representing a data type with the name 'I32'. It is associated with the NumPy data type `np.int16` and does not have any valid conversions specified. This indicates that it is a 16-bit integer data type without any predefined conversion paths to other data types.
- **Use**: This variable is used to define and manage the properties of the 'I32' data type within the context of the software, particularly in handling unquantized data.


---
### DT\_BF16
- **Type**: `UnquantizedDataType`
- **Description**: `DT_BF16` is an instance of the `UnquantizedDataType` class, representing a data type with the name 'BF16'. It uses a NumPy data type of `np.uint16` and allows valid conversions to 'F32', 'F16', and 'Q8_0'. This data type is used to handle unquantized data in the BF16 format.
- **Use**: This variable is used to define and manage data in the BF16 format, allowing conversions to other specified data types.


---
### DT\_Q8\_0
- **Type**: `Q8_0QuantizedDataType`
- **Description**: `DT_Q8_0` is an instance of the `Q8_0QuantizedDataType` class, which is a specialized data type for quantized data in the Q8_0 format. It is configured with a block size of 32 and a quantized data type consisting of a float and an integer array of size 32. This data type is used for quantizing arrays of `np.float32` type into a more compact representation.
- **Use**: This variable is used to define and handle quantized data in the Q8_0 format, particularly for operations that require quantization and dequantization of data blocks.


---
### NUMPY\_TYPE\_TO\_DATA\_TYPE
- **Type**: `dict[np.dtype[Any], DataType]`
- **Description**: `NUMPY_TYPE_TO_DATA_TYPE` is a dictionary that maps NumPy data types (`np.dtype`) to custom `DataType` instances. It is initialized as an empty dictionary and then populated with mappings for specific unquantized data types like `DT_BF16`, `DT_F16`, `DT_F32`, and `DT_I32`. Each entry in the dictionary associates a NumPy data type with a corresponding `DataType` object, which contains metadata about the data type, such as its name, NumPy dtype, and valid conversions.
- **Use**: This variable is used to map NumPy data types to custom `DataType` objects for handling data type conversions and metadata in the application.


---
### SAFETENSORS\_DATA\_TYPES
- **Type**: `dict[str, DataType]`
- **Description**: The `SAFETENSORS_DATA_TYPES` variable is a dictionary that maps string keys representing data type names to instances of the `DataType` class. Each key corresponds to a specific data type used in the safetensors format, such as 'BF16', 'F16', 'F32', and 'I32'. The values are instances of the `DataType` class, which encapsulate information about the data type, including its name, numpy dtype, and valid conversions.
- **Use**: This variable is used to define and access the data types supported by the safetensors format, allowing for type checking and conversion operations.


---
### GGML\_FILE\_TYPE\_TO\_DATA\_TYPE
- **Type**: `dict[GGMLFileType, DataType]`
- **Description**: `GGML_FILE_TYPE_TO_DATA_TYPE` is a dictionary that maps different file types, represented by the `GGMLFileType` enum, to corresponding data types, represented by instances of the `DataType` class. The dictionary includes mappings for three file types: `AllF32`, `MostlyF16`, and `MostlyQ8_0`, which are associated with the data types `DT_F32`, `DT_F16`, and `DT_Q8_0` respectively.
- **Use**: This variable is used to determine the appropriate data type for a given file type when processing GGML files.


---
### GGMLCompatibleTensor
- **Type**: `UnquantizedTensor`
- **Description**: `GGMLCompatibleTensor` is a global variable that is an alias for the `UnquantizedTensor` class. This class is used to represent tensors that are not quantized, meaning they retain their full precision data type, typically used for operations that require high precision.
- **Use**: This variable is used to handle tensors in their unquantized form, allowing for operations that require full precision.


---
### LazyModel
- **Type**: `TypeAlias`
- **Description**: `LazyModel` is a type alias for a dictionary where the keys are strings and the values are `LazyTensor` objects. This alias is used to represent a model where the tensors are loaded lazily, meaning they are only loaded into memory when needed.
- **Use**: This variable is used to define the structure of a model where tensors are accessed lazily, optimizing memory usage by loading tensor data only when required.


---
### ModelFormat
- **Type**: `TypeAlias`
- **Description**: `ModelFormat` is a type alias for a literal type that can take one of four string values: 'ggml', 'torch', 'safetensors', or 'none'. This type alias is used to specify the format of a model in the code.
- **Use**: This variable is used to define the expected format of a model, ensuring that only the specified string literals are used to represent model formats.


---
### In
- **Type**: `TypeVar`
- **Description**: `In` is a type variable defined using Python's `TypeVar` from the `typing` module. It is used to specify a generic type placeholder that can be replaced with any type when the function or class is instantiated.
- **Use**: `In` is used as a generic type placeholder in type annotations to allow for flexible and reusable code.


---
### Out
- **Type**: `TypeVar`
- **Description**: The variable `Out` is a type variable defined using Python's `TypeVar` from the `typing` module. It is used to create generic types, allowing for more flexible and reusable code by enabling type parameterization.
- **Use**: `Out` is used to define a generic type that can be used in type annotations for functions or classes, allowing them to operate on any type specified at runtime.


# Classes

---
### DataType<!-- {{#class:llama.cpp/examples/convert_legacy_llama.DataType}} -->
- **Decorators**: `@dataclass`
- **Members**:
    - `name`: The name of the data type.
    - `dtype`: The numpy data type associated with this data type.
    - `valid_conversions`: A list of valid conversion types for this data type.
- **Description**: The `DataType` class represents a data type with a name, a corresponding numpy data type, and a list of valid conversions. It is designed to be immutable, as indicated by the `frozen=True` parameter in the `@dataclass` decorator. This class provides a method to calculate the number of bytes required to store a given number of elements of this data type.
- **Methods**:
    - [`llama.cpp/examples/convert_legacy_llama.DataType.elements_to_bytes`](#DataTypeelements_to_bytes)

**Methods**

---
#### DataType\.elements\_to\_bytes<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.DataType.elements_to_bytes}} -->
The `elements_to_bytes` method calculates the total number of bytes required to store a given number of elements of a specific data type.
- **Inputs**:
    - `n_elements`: An integer representing the number of elements for which the byte size is to be calculated.
- **Control Flow**:
    - The method multiplies the number of elements (`n_elements`) by the item size of the data type (`self.dtype.itemsize`).
- **Output**: Returns an integer representing the total number of bytes required to store the specified number of elements.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.DataType`](#cpp/examples/convert_legacy_llamaDataType)  (Base Class)



---
### UnquantizedDataType<!-- {{#class:llama.cpp/examples/convert_legacy_llama.UnquantizedDataType}} -->
- **Decorators**: `@dataclass`, `@frozen=True`
- **Description**: The `UnquantizedDataType` class is a specialized data type that inherits from the `DataType` class, designed to represent data types that are not quantized. It is a frozen dataclass, meaning its instances are immutable once created. This class does not introduce any new attributes or methods beyond those inherited from `DataType`, and serves as a marker or specific type for unquantized data types within the system.
- **Inherits From**:
    - [`llama.cpp/examples/convert_legacy_llama.DataType`](#cpp/examples/convert_legacy_llamaDataType)


---
### QuantizedDataType<!-- {{#class:llama.cpp/examples/convert_legacy_llama.QuantizedDataType}} -->
- **Decorators**: `@dataclass`
- **Members**:
    - `block_size`: Defines the size of blocks for quantization.
    - `quantized_dtype`: Specifies the numpy data type used for quantized data.
    - `ggml_type`: Indicates the GGML quantization type associated with this data type.
- **Description**: The `QuantizedDataType` class is a specialized data type for handling quantized data in machine learning models. It extends the `DataType` class and includes additional attributes specific to quantization, such as `block_size`, `quantized_dtype`, and `ggml_type`. This class provides a framework for defining how data should be quantized and stored, although the actual quantization method is not implemented in this class and must be defined in subclasses or instances.
- **Methods**:
    - [`llama.cpp/examples/convert_legacy_llama.QuantizedDataType.quantize`](#QuantizedDataTypequantize)
    - [`llama.cpp/examples/convert_legacy_llama.QuantizedDataType.elements_to_bytes`](#QuantizedDataTypeelements_to_bytes)
- **Inherits From**:
    - [`llama.cpp/examples/convert_legacy_llama.DataType`](#cpp/examples/convert_legacy_llamaDataType)

**Methods**

---
#### QuantizedDataType\.quantize<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.QuantizedDataType.quantize}} -->
The `quantize` method is intended to perform quantization on an input array but is not implemented in the `QuantizedDataType` class.
- **Inputs**:
    - `arr`: An input array of type `NDArray` that is intended to be quantized.
- **Control Flow**:
    - The method immediately raises a `NotImplementedError`, indicating that the quantization functionality is not yet implemented for the specific data type.
- **Output**: The method does not return any output as it raises an exception.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.QuantizedDataType`](#cpp/examples/convert_legacy_llamaQuantizedDataType)  (Base Class)


---
#### QuantizedDataType\.elements\_to\_bytes<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.QuantizedDataType.elements_to_bytes}} -->
The `elements_to_bytes` method calculates the number of bytes required to store a given number of elements based on the quantized data type and block size.
- **Inputs**:
    - `n_elements`: An integer representing the number of elements to be converted to bytes.
- **Control Flow**:
    - The method asserts that the number of elements (`n_elements`) is a multiple of the `block_size`. If not, it raises an assertion error with a message indicating the invalid number of elements for the given block size.
    - If the assertion passes, it calculates the number of bytes by multiplying the item size of the `quantized_dtype` by the quotient of `n_elements` divided by `block_size`.
- **Output**: Returns an integer representing the number of bytes required to store the specified number of elements.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.QuantizedDataType`](#cpp/examples/convert_legacy_llamaQuantizedDataType)  (Base Class)



---
### Q8\_0QuantizedDataType<!-- {{#class:llama.cpp/examples/convert_legacy_llama.Q8_0QuantizedDataType}} -->
- **Decorators**: `@dataclass`
- **Members**:
    - `name`: The name of the quantized data type.
    - `dtype`: The numpy data type of the quantized data.
    - `valid_conversions`: A list of valid data type conversions.
    - `block_size`: The size of each block for quantization.
    - `quantized_dtype`: The numpy data type used for the quantized data.
    - `ggml_type`: The GGML quantization type.
- **Description**: The `Q8_0QuantizedDataType` class is a specialized data type for handling Q8_0 quantization in Python. It extends the `QuantizedDataType` class and is designed to perform block quantization on numpy arrays of type `float32`. The class is immutable, as indicated by the `frozen=True` parameter in the `@dataclass` decorator, ensuring that instances cannot be modified after creation. It includes attributes for the name, data type, valid conversions, block size, quantized data type, and GGML quantization type. The `quantize` method implements a block quantization algorithm that processes the input array in blocks, normalizing each block by its maximum absolute value and scaling it to fit within the range of an 8-bit signed integer.
- **Methods**:
    - [`llama.cpp/examples/convert_legacy_llama.Q8_0QuantizedDataType.quantize`](#Q8_0QuantizedDataTypequantize)
- **Inherits From**:
    - [`llama.cpp/examples/convert_legacy_llama.QuantizedDataType`](#cpp/examples/convert_legacy_llamaQuantizedDataType)

**Methods**

---
#### Q8\_0QuantizedDataType\.quantize<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.Q8_0QuantizedDataType.quantize}} -->
The `quantize` method performs block quantization on a given NumPy array of type `float32` using a Q8_0 quantization scheme.
- **Inputs**:
    - `arr`: A NumPy array (`NDArray`) of type `float32` that needs to be quantized. The size of the array must be a non-zero multiple of the `block_size` attribute of the class.
- **Control Flow**:
    - The method first asserts that the size of the input array is a non-zero multiple of the `block_size` and that the array's data type is `float32`.
    - It calculates the number of blocks by dividing the array size by the `block_size`.
    - The array is reshaped into a 2D array where each row represents a block of size `block_size`.
    - A nested function `quantize_blocks_q8_0` is defined to perform the quantization on each block.
    - Within `quantize_blocks_q8_0`, the maximum absolute value of each block is divided by 127 to compute a scaling factor `d`.
    - The blocks are then divided by their respective scaling factors and rounded to the nearest integer to form quantized values `qs`.
    - If the scaling factor `d` is zero, the corresponding quantized values are set to zero.
    - The function yields tuples of scaling factors and quantized values for each block.
    - The method returns a NumPy array created from the tuples generated by `quantize_blocks_q8_0`, with a specified count and data type.
- **Output**: A quantized NumPy array (`NDArray`) with a data type specified by the `quantized_dtype` attribute of the class, representing the quantized version of the input array.
- **Functions called**:
    - [`llama.cpp/convert_lora_to_gguf.LoraTorchTensor.reshape`](../convert_lora_to_gguf.py.driver.md#LoraTorchTensorreshape)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.Q8_0QuantizedDataType`](#cpp/examples/convert_legacy_llamaQ8_0QuantizedDataType)  (Base Class)



---
### GGMLFileType<!-- {{#class:llama.cpp/examples/convert_legacy_llama.GGMLFileType}} -->
- **Members**:
    - `AllF32`: Represents a file type where all tensors are stored in F32 format.
    - `MostlyF16`: Represents a file type where most tensors are stored in F16 format, except 1D tensors.
    - `MostlyQ8_0`: Represents a file type where most tensors are stored in Q8_0 format, except 1D tensors.
- **Description**: The `GGMLFileType` class is an enumeration that defines different file types for storing tensor data, specifically indicating the data type used for storing the tensors. It extends `enum.IntEnum` and provides three specific file types: `AllF32`, `MostlyF16`, and `MostlyQ8_0`, each associated with a specific integer value. This class is used to determine the appropriate data type for tensors based on the file type, with special handling for 1D tensors to ensure compatibility with the rest of the codebase.
- **Methods**:
    - [`llama.cpp/examples/convert_legacy_llama.GGMLFileType.type_for_tensor`](#GGMLFileTypetype_for_tensor)
- **Inherits From**:
    - `enum.IntEnum`

**Methods**

---
#### GGMLFileType\.type\_for\_tensor<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.GGMLFileType.type_for_tensor}} -->
The `type_for_tensor` method determines the appropriate data type for a given tensor based on its dimensionality and the file type.
- **Inputs**:
    - `name`: A string representing the name of the tensor.
    - `tensor`: An instance of `LazyTensor` representing the tensor whose data type is to be determined.
- **Control Flow**:
    - Retrieve the data type (`dt`) associated with the current `GGMLFileType` instance from the `GGML_FILE_TYPE_TO_DATA_TYPE` dictionary.
    - If `dt` is `None`, raise a `ValueError` with the current `GGMLFileType` instance as the error message.
    - Check the dimensionality of the `tensor` by evaluating the length of its shape.
    - If the tensor is 1-dimensional, return `DT_F32` as the data type.
    - Otherwise, return the data type `dt` retrieved earlier.
- **Output**: Returns a `DataType` object representing the data type for the given tensor, which is either `DT_F32` for 1D tensors or the data type associated with the current `GGMLFileType` instance for higher-dimensional tensors.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.GGMLFileType`](#cpp/examples/convert_legacy_llamaGGMLFileType)  (Base Class)



---
### Params<!-- {{#class:llama.cpp/examples/convert_legacy_llama.Params}} -->
- **Decorators**: `@dataclass`
- **Members**:
    - `n_vocab`: The size of the vocabulary.
    - `n_embd`: The size of the embedding layer.
    - `n_layer`: The number of layers in the model.
    - `n_ctx`: The context size for the model.
    - `n_ff`: The size of the feed-forward layer.
    - `n_head`: The number of attention heads.
    - `n_head_kv`: The number of key-value heads.
    - `n_experts`: The number of experts, if applicable.
    - `n_experts_used`: The number of experts used, if applicable.
    - `f_norm_eps`: The epsilon value for normalization.
    - `rope_scaling_type`: The type of rope scaling used, if any.
    - `f_rope_freq_base`: The base frequency for rope scaling, if any.
    - `f_rope_scale`: The scale factor for rope scaling, if any.
    - `n_ctx_orig`: The original context size, if applicable.
    - `rope_finetuned`: Indicates if rope scaling is finetuned.
    - `ftype`: The file type for GGML.
    - `path_model`: The path to the directory containing the model files.
- **Description**: The `Params` class is a data structure that encapsulates various configuration parameters for a machine learning model, particularly those related to model architecture and hyperparameters. It includes attributes such as vocabulary size, embedding dimensions, number of layers, context size, and other specialized parameters like the number of experts and rope scaling factors. This class is designed to facilitate the loading and configuration of model parameters from different sources, such as JSON configuration files or inferred directly from model files, and is used to ensure consistency and correctness in model setup.
- **Methods**:
    - [`llama.cpp/examples/convert_legacy_llama.Params.guessed`](#Paramsguessed)
    - [`llama.cpp/examples/convert_legacy_llama.Params.loadHFTransformerJson`](#ParamsloadHFTransformerJson)
    - [`llama.cpp/examples/convert_legacy_llama.Params.loadOriginalParamsJson`](#ParamsloadOriginalParamsJson)
    - [`llama.cpp/examples/convert_legacy_llama.Params.load`](#Paramsload)

**Methods**

---
#### Params\.guessed<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.Params.guessed}} -->
The `guessed` method attempts to infer model parameters such as vocabulary size, embedding size, number of layers, and other attributes from a given model's structure.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `model`: A `LazyModel` object representing the model from which parameters are to be inferred.
- **Control Flow**:
    - Check if 'model.embed_tokens.weight' or 'tok_embeddings.weight' exists in the model to determine vocabulary and embedding sizes.
    - Determine the number of layers by checking for the presence of specific weight keys in the model, using different naming conventions.
    - If the number of layers is less than 1, raise a KeyError with a suggestion to provide a 'config.json'.
    - Calculate the number of heads by dividing the embedding size by 128.
    - Set a multiplier value for further calculations.
    - Calculate the feed-forward size using a formula based on the embedding size and the multiplier.
    - Return a `Params` object with the inferred parameters.
- **Output**: A `Params` object containing inferred model parameters such as `n_vocab`, `n_embd`, `n_layer`, `n_ctx`, `n_ff`, `n_head`, `n_head_kv`, and `f_norm_eps`.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.Params`](#cpp/examples/convert_legacy_llamaParams)  (Base Class)


---
#### Params\.loadHFTransformerJson<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.Params.loadHFTransformerJson}} -->
The `loadHFTransformerJson` method loads a Hugging Face transformer model configuration from a JSON file and returns a `Params` object with the model's parameters.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `model`: A `LazyModel` object representing the model to be configured.
    - `config_path`: A `Path` object indicating the file path to the JSON configuration file.
- **Control Flow**:
    - Open the JSON configuration file specified by `config_path` and load its contents into a dictionary named `config`.
    - Initialize variables `rope_scaling_type`, `f_rope_scale`, `n_ctx_orig`, and `rope_finetuned` to `None`.
    - Retrieve the `rope_scaling` configuration from the `config` dictionary, if available.
    - If `rope_scaling` is not `None` and contains a `type`, set `rope_scaling_type` and `f_rope_scale` based on the type and factor specified.
    - Determine the context length `n_ctx` from either `max_sequence_length` or `max_position_embeddings` in the `config` dictionary, raising a `KeyError` if neither is found.
    - Initialize `n_experts` and `n_experts_used` to `None`.
    - If `num_local_experts` is present in the `config`, set `n_experts` and `n_experts_used` accordingly.
    - Return a `Params` object initialized with various parameters extracted from the `config` dictionary and the computed values.
- **Output**: A `Params` object containing the model's parameters, such as vocabulary size, hidden size, number of layers, context length, and other configuration details.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.Params`](#cpp/examples/convert_legacy_llamaParams)  (Base Class)


---
#### Params\.loadOriginalParamsJson<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.Params.loadOriginalParamsJson}} -->
The `loadOriginalParamsJson` method loads and returns model parameters from a JSON configuration file, determining specific settings based on the model's version and configuration.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `model`: A `LazyModel` object representing the model whose parameters are to be loaded.
    - `config_path`: A `Path` object indicating the file path to the JSON configuration file.
- **Control Flow**:
    - Open the JSON configuration file specified by `config_path` and load its contents into a dictionary named `config`.
    - Initialize variables `n_experts`, `n_experts_used`, `f_rope_freq_base`, and `n_ff` to `None`.
    - Determine the context size `n_ctx` based on the presence of certain keys and values in the `config` dictionary, distinguishing between different versions of the LLaMA model and CodeLlama.
    - Check if the model contains a specific layer weight to set `n_ff` based on its shape.
    - If the `config` contains a 'moe' key, set `n_ff`, `n_experts`, `n_experts_used`, and `f_rope_freq_base` based on the model's configuration.
    - Assert that `n_ff` is not `None` to ensure it has been set.
    - Return a `Params` object initialized with various parameters extracted from the model and configuration, including vocabulary size, embedding dimensions, number of layers, context size, feed-forward dimensions, number of heads, and other optional parameters.
- **Output**: A `Params` object containing the model's parameters, including vocabulary size, embedding dimensions, number of layers, context size, feed-forward dimensions, number of heads, and other optional parameters.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.Params`](#cpp/examples/convert_legacy_llamaParams)  (Base Class)


---
#### Params\.load<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.Params.load}} -->
The `load` method loads model parameters from specified configuration files or guesses them if necessary.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `model_plus`: An instance of ModelPlus containing the model and its associated paths and format.
- **Control Flow**:
    - Determine the path for 'config.json' and 'params.json' based on the first path in model_plus.
    - Check if 'config.json' exists; if so, load parameters using `Params.loadHFTransformerJson`.
    - If 'config.json' does not exist, check if 'params.json' exists; if so, load parameters using `Params.loadOriginalParamsJson`.
    - If neither 'config.json' nor 'params.json' exists and the model format is not 'none', guess the parameters using `Params.guessed`.
    - If the model format is 'none' and parameters cannot be guessed, raise a ValueError.
    - Set the `path_model` attribute of the loaded parameters to the parent directory of the first path in model_plus.
    - Return the loaded or guessed parameters.
- **Output**: Returns an instance of Params containing the loaded or guessed model parameters.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.Params.loadHFTransformerJson`](#ParamsloadHFTransformerJson)
    - [`llama.cpp/examples/convert_legacy_llama.Params.loadOriginalParamsJson`](#ParamsloadOriginalParamsJson)
    - [`llama.cpp/examples/convert_legacy_llama.Params.guessed`](#Paramsguessed)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.Params`](#cpp/examples/convert_legacy_llamaParams)  (Base Class)



---
### Tensor<!-- {{#class:llama.cpp/examples/convert_legacy_llama.Tensor}} -->
- **Decorators**: `@ABC`
- **Members**:
    - `ndarray`: Represents the underlying n-dimensional array data of the tensor.
    - `data_type`: Specifies the data type of the tensor, indicating how the data is stored and interpreted.
- **Description**: The `Tensor` class is an abstract base class that defines the structure and essential properties of a tensor, which is a multi-dimensional array used in numerical computations. It includes abstract methods for type conversion, permutation, and partitioning of the tensor, as well as a method to convert the tensor to a GGML-compatible format. The class is designed to be extended by concrete implementations that provide specific functionality for handling tensor data.
- **Methods**:
    - [`llama.cpp/examples/convert_legacy_llama.Tensor.astype`](#Tensorastype)
    - [`llama.cpp/examples/convert_legacy_llama.Tensor.permute`](#Tensorpermute)
    - [`llama.cpp/examples/convert_legacy_llama.Tensor.permute_part`](#Tensorpermute_part)
    - [`llama.cpp/examples/convert_legacy_llama.Tensor.part`](#Tensorpart)
    - [`llama.cpp/examples/convert_legacy_llama.Tensor.to_ggml`](#Tensorto_ggml)
- **Inherits From**:
    - `ABC`

**Methods**

---
#### Tensor\.astype<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.Tensor.astype}} -->
The `astype` method is an abstract method in the `Tensor` class that is intended to convert the tensor's data type to a specified `DataType`.
- **Decorators**: `@abstractmethod`
- **Inputs**:
    - `data_type`: A `DataType` object representing the target data type to which the tensor should be converted.
- **Control Flow**:
    - The method is defined as an abstract method, meaning it must be implemented by any subclass of `Tensor`.
    - The method takes a `DataType` as an argument, which specifies the desired data type for the tensor conversion.
- **Output**: The method returns an instance of the same class (`Self`), which is a tensor with the converted data type.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.Tensor`](#cpp/examples/convert_legacy_llamaTensor)  (Base Class)


---
#### Tensor\.permute<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.Tensor.permute}} -->
The `permute` method is an abstract method intended to rearrange the tensor data based on the number of heads and key-value heads.
- **Decorators**: `@abstractmethod`
- **Inputs**:
    - `n_head`: The number of heads to be used in the permutation.
    - `n_head_kv`: The number of key-value heads to be used in the permutation.
- **Control Flow**:
    - The method is abstract, so it does not have an implementation in the `Tensor` class.
    - The actual implementation should be provided in a subclass of `Tensor`.
- **Output**: The method is expected to return an instance of the class that implements it, which is a subclass of `Tensor`.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.Tensor`](#cpp/examples/convert_legacy_llamaTensor)  (Base Class)


---
#### Tensor\.permute\_part<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.Tensor.permute_part}} -->
The `permute_part` method is an abstract method intended to permute a specific part of a tensor based on the given number of parts, heads, and key-value heads.
- **Decorators**: `@abstractmethod`
- **Inputs**:
    - `n_part`: The integer representing the part of the tensor to permute.
    - `n_head`: The integer representing the number of heads for permutation.
    - `n_head_kv`: The integer representing the number of key-value heads for permutation.
- **Control Flow**:
    - The method is defined as abstract, meaning it must be implemented by any subclass of the `Tensor` class.
    - The method signature suggests it will perform a permutation operation on a part of a tensor, but the exact logic is not provided in the abstract method.
- **Output**: The method is expected to return an instance of the class implementing the method, as indicated by the return type `Self`.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.Tensor`](#cpp/examples/convert_legacy_llamaTensor)  (Base Class)


---
#### Tensor\.part<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.Tensor.part}} -->
The `part` method is an abstract method in the `Tensor` class that returns a portion of the tensor based on the specified part index.
- **Decorators**: `@abstractmethod`
- **Inputs**:
    - `n_part`: An integer representing the index of the part of the tensor to be returned.
- **Control Flow**:
    - The method is defined as an abstract method, meaning it must be implemented by any subclass of `Tensor`.
    - The method is expected to return a portion of the tensor corresponding to the specified part index `n_part`.
- **Output**: Returns an instance of the same type (`Self`), representing a portion of the tensor.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.Tensor`](#cpp/examples/convert_legacy_llamaTensor)  (Base Class)


---
#### Tensor\.to\_ggml<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.Tensor.to_ggml}} -->
The `to_ggml` method is an abstract method intended to convert a tensor to a GGML-compatible format.
- **Decorators**: `@abstractmethod`
- **Inputs**: None
- **Control Flow**:
    - The method is defined as an abstract method, indicating that it must be implemented by any subclass of the `Tensor` class.
- **Output**: The method returns a `GGMLCompatibleTensor`, which is a tensor compatible with the GGML format.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.Tensor`](#cpp/examples/convert_legacy_llamaTensor)  (Base Class)



---
### UnquantizedTensor<!-- {{#class:llama.cpp/examples/convert_legacy_llama.UnquantizedTensor}} -->
- **Members**:
    - `ndarray`: Holds the numpy array data for the tensor.
    - `data_type`: Stores the data type of the tensor, derived from the numpy array's dtype.
- **Description**: The `UnquantizedTensor` class is a specialized implementation of the abstract `Tensor` class, designed to handle unquantized tensor data using numpy arrays. It provides functionality to convert the tensor to different data types, permute its dimensions, and extract specific parts of the tensor. The class ensures that the tensor's data type is consistent with the numpy array's data type and supports operations like type conversion and permutation, which are essential for manipulating tensor data in machine learning models.
- **Methods**:
    - [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor.__init__`](#UnquantizedTensor__init__)
    - [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor.astype`](#UnquantizedTensorastype)
    - [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor.to_ggml`](#UnquantizedTensorto_ggml)
    - [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor.permute_part`](#UnquantizedTensorpermute_part)
    - [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor.part`](#UnquantizedTensorpart)
    - [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor.permute`](#UnquantizedTensorpermute)
- **Inherits From**:
    - [`llama.cpp/examples/convert_legacy_llama.Tensor`](#cpp/examples/convert_legacy_llamaTensor)

**Methods**

---
#### UnquantizedTensor\.\_\_init\_\_<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.UnquantizedTensor.__init__}} -->
The `__init__` method initializes an instance of the `UnquantizedTensor` class with a given NumPy array and determines its data type.
- **Inputs**:
    - `ndarray`: A NumPy array (`np.ndarray`) that is used to initialize the `UnquantizedTensor` instance.
- **Control Flow**:
    - The method asserts that the input `ndarray` is an instance of `np.ndarray` to ensure type safety.
    - It assigns the input `ndarray` to the instance variable `self.ndarray`.
    - It determines the data type of the `ndarray` using a predefined mapping (`NUMPY_TYPE_TO_DATA_TYPE`) and assigns it to the instance variable `self.data_type`.
- **Output**: The method does not return any value; it initializes the instance variables of the `UnquantizedTensor` object.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor`](#cpp/examples/convert_legacy_llamaUnquantizedTensor)  (Base Class)


---
#### UnquantizedTensor\.astype<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.UnquantizedTensor.astype}} -->
The `astype` method converts the data type of an `UnquantizedTensor` to a specified `DataType`.
- **Inputs**:
    - `data_type`: An instance of `DataType` that specifies the target data type for conversion.
- **Control Flow**:
    - Retrieve the target data type from the `data_type` argument.
    - Check if the current data type of the tensor is `DT_BF16`.
    - If the current data type is `DT_BF16`, convert the tensor's data from BF16 to FP32 using the [`bf16_to_fp32`](#cpp/examples/convert_legacy_llamabf16_to_fp32) function.
    - Convert the tensor's data to the specified target data type using NumPy's `astype` method.
    - Return a new `UnquantizedTensor` instance with the converted data.
- **Output**: Returns a new `UnquantizedTensor` instance with the data converted to the specified `DataType`.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.bf16_to_fp32`](#cpp/examples/convert_legacy_llamabf16_to_fp32)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor`](#cpp/examples/convert_legacy_llamaUnquantizedTensor)  (Base Class)


---
#### UnquantizedTensor\.to\_ggml<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.UnquantizedTensor.to_ggml}} -->
The `to_ggml` method returns the instance of the `UnquantizedTensor` class itself.
- **Inputs**: None
- **Control Flow**:
    - The method simply returns the instance of the class it is called on, without any modifications or additional logic.
- **Output**: The output is the instance of the `UnquantizedTensor` class itself, which is the object on which the method is called.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor`](#cpp/examples/convert_legacy_llamaUnquantizedTensor)  (Base Class)


---
#### UnquantizedTensor\.permute\_part<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.UnquantizedTensor.permute_part}} -->
The `permute_part` method permutes a specific part of the tensor based on the given parameters and returns a new `UnquantizedTensor` with the permuted data.
- **Inputs**:
    - `n_part`: An integer representing the part of the tensor to permute.
    - `n_head`: An integer representing the number of heads for permutation.
    - `n_head_kv`: An integer representing the number of key-value heads for permutation.
- **Control Flow**:
    - Calculate the size of each part of the tensor by dividing the first dimension of `self.ndarray` by 3 and store it in `r`.
    - Slice the tensor to get the part specified by `n_part` using the calculated size `r`.
    - Call the [`permute`](#cpp/examples/convert_legacy_llamapermute) function on the sliced tensor with `n_head` and `n_head_kv` as arguments.
    - Return a new `UnquantizedTensor` initialized with the permuted data.
- **Output**: Returns an `UnquantizedTensor` containing the permuted part of the original tensor.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.permute`](#cpp/examples/convert_legacy_llamapermute)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor`](#cpp/examples/convert_legacy_llamaUnquantizedTensor)  (Base Class)


---
#### UnquantizedTensor\.part<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.UnquantizedTensor.part}} -->
The `part` method extracts a specific segment of the tensor based on the given part index.
- **Inputs**:
    - `n_part`: An integer representing the part index to extract from the tensor.
- **Control Flow**:
    - Calculate the segment size `r` as one-third of the first dimension of the tensor's shape.
    - Extract the segment of the tensor corresponding to the specified `n_part` by slicing the array from `r * n_part` to `r * n_part + r`.
    - Return a new `UnquantizedTensor` object containing the extracted segment.
- **Output**: Returns an `UnquantizedTensor` object containing the specified segment of the original tensor.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor`](#cpp/examples/convert_legacy_llamaUnquantizedTensor)  (Base Class)


---
#### UnquantizedTensor\.permute<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.UnquantizedTensor.permute}} -->
The [`permute`](#cpp/examples/convert_legacy_llamapermute) method rearranges the elements of the tensor's underlying ndarray based on the specified number of heads and key-value heads.
- **Inputs**:
    - `n_head`: The number of heads to use for permutation.
    - `n_head_kv`: The number of key-value heads to use for permutation.
- **Control Flow**:
    - The method calls the [`permute`](#cpp/examples/convert_legacy_llamapermute) function with the tensor's ndarray, `n_head`, and `n_head_kv` as arguments.
    - The result of the [`permute`](#cpp/examples/convert_legacy_llamapermute) function is used to create a new `UnquantizedTensor` instance, which is then returned.
- **Output**: An `UnquantizedTensor` object with the permuted ndarray.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.permute`](#cpp/examples/convert_legacy_llamapermute)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor`](#cpp/examples/convert_legacy_llamaUnquantizedTensor)  (Base Class)



---
### LazyTensor<!-- {{#class:llama.cpp/examples/convert_legacy_llama.LazyTensor}} -->
- **Decorators**: `@dataclass`
- **Members**:
    - `_load`: A callable function that returns a Tensor when invoked.
    - `shape`: A list of integers representing the dimensions of the tensor.
    - `data_type`: The data type of the tensor, represented by a DataType object.
    - `description`: A string providing a description of the tensor.
- **Description**: The LazyTensor class represents a tensor that is loaded on demand, allowing for deferred computation and memory efficiency. It encapsulates a callable function to load the tensor, its shape, data type, and a description. This class provides methods to load the tensor, convert it to a different data type, and validate such conversions, ensuring that the tensor's data type is compatible with the desired type.
- **Methods**:
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor.load`](#LazyTensorload)
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor.astype`](#LazyTensorastype)
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor.validate_conversion_to`](#LazyTensorvalidate_conversion_to)

**Methods**

---
#### LazyTensor\.load<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.LazyTensor.load}} -->
The `load` method retrieves a `Tensor` object by invoking a callable and ensures its data type matches the expected data type of the `LazyTensor` instance.
- **Inputs**: None
- **Control Flow**:
    - Invoke the `_load` callable to retrieve a `Tensor` object and assign it to `ret`.
    - Check if the data type of `ret` matches the `data_type` of the `LazyTensor` instance or if their numpy data types are equivalent.
    - Raise an assertion error with a tuple containing the expected and actual data types and a description if the data types do not match.
    - Return the `ret` object.
- **Output**: Returns a `Tensor` object that matches the expected data type of the `LazyTensor` instance.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor)  (Base Class)


---
#### LazyTensor\.astype<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.LazyTensor.astype}} -->
The [`astype`](#Tensorastype) method converts a `LazyTensor` to a specified `DataType` and returns a new `LazyTensor` with the converted type.
- **Inputs**:
    - `data_type`: A `DataType` object representing the target data type to which the `LazyTensor` should be converted.
- **Control Flow**:
    - The method first calls [`validate_conversion_to`](#LazyTensorvalidate_conversion_to) to ensure the conversion to the specified `data_type` is valid.
    - A nested function [`load`](#Paramsload) is defined, which loads the current tensor and converts it to the specified `data_type` using the [`astype`](#Tensorastype) method of the `Tensor` class.
    - A new `LazyTensor` is created and returned, initialized with the [`load`](#Paramsload) function, the current tensor's shape, the new `data_type`, and an updated description indicating the conversion.
- **Output**: Returns a new `LazyTensor` object with the specified `data_type`.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor.validate_conversion_to`](#LazyTensorvalidate_conversion_to)
    - [`llama.cpp/examples/convert_legacy_llama.Params.load`](#Paramsload)
    - [`llama.cpp/examples/convert_legacy_llama.Tensor.astype`](#Tensorastype)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor)  (Base Class)


---
#### LazyTensor\.validate\_conversion\_to<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.LazyTensor.validate_conversion_to}} -->
The `validate_conversion_to` method checks if a conversion from the current data type to a specified data type is valid, raising an error if not.
- **Inputs**:
    - `data_type`: A `DataType` object representing the target data type to which conversion is being validated.
- **Control Flow**:
    - The method checks if the provided `data_type` is different from the current `data_type` of the object.
    - It then checks if the name of the provided `data_type` is not in the list of valid conversions for the current `data_type`.
    - If both conditions are true, it raises a `ValueError` indicating that the conversion cannot be validated.
- **Output**: The method does not return any value; it raises a `ValueError` if the conversion is not valid.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor)  (Base Class)



---
### ModelPlus<!-- {{#class:llama.cpp/examples/convert_legacy_llama.ModelPlus}} -->
- **Decorators**: `@dataclass`
- **Members**:
    - `model`: A dictionary mapping string keys to LazyTensor objects representing the model's tensors.
    - `paths`: A list of Path objects indicating where the model was read from.
    - `format`: A string literal indicating the format of the model (e.g., 'ggml', 'torch', 'safetensors', 'none').
    - `vocab`: An optional BaseVocab object representing the vocabulary for GGML models.
- **Description**: The ModelPlus class is a data structure that encapsulates a machine learning model, including its tensors, file paths, format, and optional vocabulary. It is designed to handle models that may be loaded lazily, allowing for efficient memory usage by deferring the loading of tensor data until it is needed. This class is particularly useful for managing models that are stored in various formats and may include built-in vocabularies, such as those used in GGML models.


---
### LazyStorageKind<!-- {{#class:llama.cpp/examples/convert_legacy_llama.LazyStorageKind}} -->
- **Decorators**: `@dataclass`
- **Members**:
    - `data_type`: Specifies the data type associated with the lazy storage kind.
- **Description**: The `LazyStorageKind` class is a simple data structure that represents a kind of storage with an associated data type. It is used to define the type of data that a `LazyStorage` can handle, encapsulating the data type information in a structured way. This class is typically used in scenarios where data types need to be managed or referenced in a lazy-loading context, such as when dealing with large datasets or files that are loaded on demand.


---
### LazyStorage<!-- {{#class:llama.cpp/examples/convert_legacy_llama.LazyStorage}} -->
- **Decorators**: `@dataclass`
- **Members**:
    - `load`: A callable that loads a numpy array given two integer parameters.
    - `kind`: Specifies the kind of lazy storage, encapsulating data type information.
    - `description`: A string providing a description of the lazy storage.
- **Description**: The LazyStorage class is designed to handle the deferred loading of numpy arrays, allowing for efficient memory usage by loading data only when needed. It encapsulates a callable for loading data, a kind that specifies the data type, and a description for additional context. This class is useful in scenarios where large datasets are involved, and immediate loading into memory is not feasible or necessary.


---
### LazyUnpickler<!-- {{#class:llama.cpp/examples/convert_legacy_llama.LazyUnpickler}} -->
- **Members**:
    - `data_base_path`: Stores the base path for data files within the zip archive.
    - `zip_file`: Holds the reference to the zip file containing the data.
    - `CLASSES`: Maps specific module and class names to their corresponding types or storage kinds.
- **Description**: The LazyUnpickler class extends the pickle.Unpickler to facilitate lazy loading of tensors from a zip file, allowing for efficient memory usage by loading data only when needed. It overrides the persistent_load method to handle custom storage types and provides static methods for rebuilding tensors and types lazily. This class is particularly useful for handling large models stored in zip files, enabling on-demand data access without loading the entire dataset into memory at once.
- **Methods**:
    - [`llama.cpp/examples/convert_legacy_llama.LazyUnpickler.__init__`](#LazyUnpickler__init__)
    - [`llama.cpp/examples/convert_legacy_llama.LazyUnpickler.persistent_load`](#LazyUnpicklerpersistent_load)
    - [`llama.cpp/examples/convert_legacy_llama.LazyUnpickler.lazy_rebuild_tensor_v2`](#LazyUnpicklerlazy_rebuild_tensor_v2)
    - [`llama.cpp/examples/convert_legacy_llama.LazyUnpickler.rebuild_from_type_v2`](#LazyUnpicklerrebuild_from_type_v2)
    - [`llama.cpp/examples/convert_legacy_llama.LazyUnpickler.find_class`](#LazyUnpicklerfind_class)
- **Inherits From**:
    - `pickle.Unpickler`

**Methods**

---
#### LazyUnpickler\.\_\_init\_\_<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.LazyUnpickler.__init__}} -->
The [`__init__`](../convert_lora_to_gguf.py.driver.md#LoraTorchTensor__init__) method initializes an instance of the `LazyUnpickler` class with a file pointer, a data base path, and a zip file.
- **Inputs**:
    - `fp`: A file pointer (IO[bytes]) used to read the data.
    - `data_base_path`: A string representing the base path for data within the zip file.
    - `zip_file`: A `zipfile.ZipFile` object representing the zip file containing the data.
- **Control Flow**:
    - Calls the superclass (`pickle.Unpickler`) [`__init__`](../convert_lora_to_gguf.py.driver.md#LoraTorchTensor__init__) method with the file pointer `fp` to initialize the base class.
    - Sets the `data_base_path` attribute to the provided `data_base_path` argument.
    - Sets the `zip_file` attribute to the provided `zip_file` argument.
- **Output**: This method does not return any value; it initializes the instance attributes.
- **Functions called**:
    - [`llama.cpp/convert_lora_to_gguf.LoraTorchTensor.__init__`](../convert_lora_to_gguf.py.driver.md#LoraTorchTensor__init__)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.LazyUnpickler`](#cpp/examples/convert_legacy_llamaLazyUnpickler)  (Base Class)


---
#### LazyUnpickler\.persistent\_load<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.LazyUnpickler.persistent_load}} -->
The `persistent_load` method loads a lazy storage object from a zip file based on a given persistent ID.
- **Inputs**:
    - `pid`: A persistent ID, which is a tuple where the first element is expected to be 'storage', the second is an instance of `LazyStorageKind`, and the third is a filename stem.
- **Control Flow**:
    - Assert that the first element of `pid` is 'storage'.
    - Assert that the second element of `pid` is an instance of `LazyStorageKind`.
    - Extract the data type from the `LazyStorageKind` object in `pid[1]`.
    - Construct the full filename using `self.data_base_path` and `pid[2]`.
    - Retrieve file information from the zip file using the constructed filename.
    - Define a nested function `load` that reads a specified number of elements from the file at a given offset, converts the data to a numpy array, and returns it.
    - Create a description string for the storage.
    - Return a [`LazyStorage`](#cpp/examples/convert_legacy_llamaLazyStorage) object initialized with the `load` function, the `LazyStorageKind` object, and the description.
- **Output**: Returns a [`LazyStorage`](#cpp/examples/convert_legacy_llamaLazyStorage) object that can load data from a specified offset and element count, using the data type and file information derived from the persistent ID.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.LazyStorage`](#cpp/examples/convert_legacy_llamaLazyStorage)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.LazyUnpickler`](#cpp/examples/convert_legacy_llamaLazyUnpickler)  (Base Class)


---
#### LazyUnpickler\.lazy\_rebuild\_tensor\_v2<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.LazyUnpickler.lazy_rebuild_tensor_v2}} -->
The `lazy_rebuild_tensor_v2` method reconstructs a [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) from given storage and tensor properties, allowing for lazy loading of tensor data.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `storage`: An instance of `LazyStorage` that provides the mechanism to load tensor data.
    - `storage_offset`: An offset in the storage from where the tensor data should be loaded.
    - `size`: The dimensions of the tensor to be reconstructed.
    - `stride`: The stride of the tensor, which helps in calculating the number of elements to load.
    - `requires_grad`: A flag indicating if the tensor requires gradient computation (not used in the function).
    - `backward_hooks`: Hooks for backward computation (not used in the function).
    - `metadata`: Optional metadata for the tensor (default is None, not used in the function).
- **Control Flow**:
    - The function asserts that the `storage` is an instance of `LazyStorage`.
    - Defines a nested function [`load`](#Paramsload) that calculates the number of elements to load using the first element of `stride` and `size`.
    - The [`load`](#Paramsload) function loads the data from storage, reshapes it according to `size`, and returns an [`UnquantizedTensor`](#cpp/examples/convert_legacy_llamaUnquantizedTensor).
    - Constructs a description string using `storage_offset` and `storage.description`.
    - Returns a [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) initialized with the [`load`](#Paramsload) function, the tensor's size, data type from `storage.kind`, and the constructed description.
- **Output**: Returns a [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) object that encapsulates the lazy loading mechanism for the tensor data.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor`](#cpp/examples/convert_legacy_llamaUnquantizedTensor)
    - [`llama.cpp/examples/convert_legacy_llama.Params.load`](#Paramsload)
    - [`llama.cpp/convert_lora_to_gguf.LoraTorchTensor.reshape`](../convert_lora_to_gguf.py.driver.md#LoraTorchTensorreshape)
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.LazyUnpickler`](#cpp/examples/convert_legacy_llamaLazyUnpickler)  (Base Class)


---
#### LazyUnpickler\.rebuild\_from\_type\_v2<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.LazyUnpickler.rebuild_from_type_v2}} -->
The `rebuild_from_type_v2` method executes a given function with specified arguments.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `func`: A callable function that will be executed.
    - `new_type`: A parameter that is not used in the function but may represent a new type for the function.
    - `args`: A tuple of arguments to be passed to the function `func`.
    - `state`: A parameter that is not used in the function but may represent some state information.
- **Control Flow**:
    - The function takes a callable `func` and a tuple of arguments `args`.
    - It calls the function `func` with the unpacked arguments from `args` using the `*args` syntax.
    - The result of the function call is returned.
- **Output**: The output is the result of executing the function `func` with the provided arguments `args`.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.LazyUnpickler`](#cpp/examples/convert_legacy_llamaLazyUnpickler)  (Base Class)


---
#### LazyUnpickler\.find\_class<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.LazyUnpickler.find_class}} -->
The `find_class` method in `LazyUnpickler` retrieves a class from a module, specifically handling classes from the 'torch' module using a predefined dictionary.
- **Inputs**:
    - `module`: A string representing the name of the module from which to find the class.
    - `name`: A string representing the name of the class to find within the specified module.
- **Control Flow**:
    - Check if the module name starts with 'torch'.
    - If the module does not start with 'torch', call the parent class's `find_class` method to find the class.
    - If the module starts with 'torch', return the class from the `CLASSES` dictionary using the module and name as the key.
- **Output**: Returns the class object corresponding to the specified module and name, either from the `CLASSES` dictionary or by calling the parent class's `find_class` method.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.LazyUnpickler`](#cpp/examples/convert_legacy_llamaLazyUnpickler)  (Base Class)



---
### OutputFile<!-- {{#class:llama.cpp/examples/convert_legacy_llama.OutputFile}} -->
- **Members**:
    - `gguf`: An instance of GGUFWriter used to write model data to a file.
- **Description**: The `OutputFile` class is responsible for managing the output of model data to a file using the GGUF format. It initializes a GGUFWriter instance to handle the writing process and provides methods to add metadata about the model, its architecture, vocabulary, and tensor information. The class also includes functionality to write the metadata and tensor data to the file, as well as static methods for handling specific tasks like writing only the vocabulary or processing tensor data in parallel. This class is crucial for converting and exporting model data into a structured and standardized format.
- **Methods**:
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.__init__`](#OutputFile__init__)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_model`](#OutputFileadd_meta_model)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_arch`](#OutputFileadd_meta_arch)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.extract_vocabulary_from_model`](#OutputFileextract_vocabulary_from_model)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_vocab`](#OutputFileadd_meta_vocab)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_special_vocab`](#OutputFileadd_meta_special_vocab)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_tensor_info`](#OutputFileadd_tensor_info)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.write_meta`](#OutputFilewrite_meta)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.write_tensor_info`](#OutputFilewrite_tensor_info)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.write_tensor_data`](#OutputFilewrite_tensor_data)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.close`](#OutputFileclose)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.write_vocab_only`](#OutputFilewrite_vocab_only)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.do_item`](#OutputFiledo_item)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.maybe_do_quantize`](#OutputFilemaybe_do_quantize)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.write_all`](#OutputFilewrite_all)

**Methods**

---
#### OutputFile\.\_\_init\_\_<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.__init__}} -->
The `__init__` method initializes an `OutputFile` instance by creating a `GGUFWriter` object with specified output file path and endianess.
- **Inputs**:
    - `fname_out`: A `Path` object representing the output file path where the GGUF data will be written.
    - `endianess`: An optional `gguf.GGUFEndian` value indicating the byte order for the GGUF file, defaulting to `gguf.GGUFEndian.LITTLE`.
- **Control Flow**:
    - The method initializes the `gguf` attribute of the `OutputFile` instance by creating a `GGUFWriter` object.
    - The `GGUFWriter` is initialized with the provided `fname_out`, a model architecture name from `gguf.MODEL_ARCH_NAMES` using a constant `ARCH`, and the specified `endianess`.
- **Output**: There is no return value as this is a constructor method for initializing an instance of the `OutputFile` class.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.add\_meta\_model<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_model}} -->
The `add_meta_model` method adds metadata about a model and its provenance to a GGUFWriter instance.
- **Inputs**:
    - `params`: An instance of the Params class containing model parameters such as path_model and n_ctx.
    - `metadata`: An optional instance of gguf.Metadata containing various metadata fields about the model, such as name, author, version, organization, and more.
- **Control Flow**:
    - Initialize the model name to 'LLaMA'.
    - Check if metadata is provided and has a name; if so, use it as the model name.
    - If metadata name is not available, check if params.path_model is not None and use its name as the model name.
    - If params.n_ctx equals 4096, set the model name to 'LLaMA v2' as a heuristic detection of LLaMA v2 model.
    - Add the determined model name to the GGUFWriter instance using self.gguf.add_name().
    - If metadata is provided, check each metadata field (author, version, organization, etc.) and add them to the GGUFWriter instance if they are not None.
    - If metadata contains base_models, iterate over each base model entry and add its details (name, author, version, etc.) to the GGUFWriter instance.
    - If metadata contains datasets, iterate over each dataset entry and add its details (name, author, version, etc.) to the GGUFWriter instance.
    - If metadata contains tags or languages, add them to the GGUFWriter instance.
- **Output**: The method does not return any value; it modifies the GGUFWriter instance by adding metadata information.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.add\_meta\_arch<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_arch}} -->
The `add_meta_arch` method adds metadata about the neural architecture to a GGUFWriter instance using parameters from a `Params` object.
- **Inputs**:
    - `params`: An instance of the `Params` class containing various parameters of the neural architecture, such as vocabulary size, context length, embedding length, number of layers, and other optional parameters.
- **Control Flow**:
    - The method begins by adding basic architecture parameters like vocabulary size, context length, embedding length, block count, feed-forward length, rope dimension count, head count, and head count for key-value pairs to the GGUFWriter instance.
    - It checks if `n_experts` is provided in `params` and adds the expert count if it is.
    - Similarly, it checks for `n_experts_used` and adds the expert used count if present.
    - The method ensures `f_norm_eps` is not None and adds it to the GGUFWriter; otherwise, it raises a `ValueError`.
    - If `f_rope_freq_base` is not None, it adds the rope frequency base to the GGUFWriter.
    - If `rope_scaling_type` is provided, it asserts that `f_rope_scale` is not None, then adds both the rope scaling type and factor to the GGUFWriter.
    - It checks for `n_ctx_orig` and `rope_finetuned` in `params` and adds them to the GGUFWriter if they are not None.
    - Finally, it checks if `ftype` is provided and adds the file type to the GGUFWriter.
- **Output**: The method does not return any value; it modifies the state of the GGUFWriter instance by adding metadata about the neural architecture.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.extract\_vocabulary\_from\_model<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.extract_vocabulary_from_model}} -->
The `extract_vocabulary_from_model` method extracts tokens, scores, and token types from a given vocabulary object and returns them as separate lists.
- **Inputs**:
    - `vocab`: An instance of the `Vocab` class, which provides access to the vocabulary tokens, scores, and types through the `all_tokens` method.
- **Control Flow**:
    - Initialize empty lists for tokens, scores, and token types.
    - Iterate over each token, score, and token type returned by `vocab.all_tokens()`.
    - Append each token, score, and token type to their respective lists.
    - Assert that the number of tokens matches the vocabulary size.
    - Return the lists of tokens, scores, and token types as a tuple.
- **Output**: A tuple containing three lists: a list of tokens (as bytes), a list of scores (as floats), and a list of token types (as `gguf.TokenType`).
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.add\_meta\_vocab<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_vocab}} -->
The `add_meta_vocab` method adds vocabulary metadata to a GGUF model by incorporating tokenizer model information and extracted token data from a given vocabulary.
- **Inputs**:
    - `vocab`: An instance of the `Vocab` class, representing the vocabulary to be added to the GGUF model.
- **Control Flow**:
    - The method starts by adding the tokenizer model from the provided `vocab` to the GGUF model using `self.gguf.add_tokenizer_model(vocab.tokenizer_model)`.
    - It then calls `self.extract_vocabulary_from_model(vocab)` to extract tokens, scores, and token types from the `vocab`.
    - The extracted tokens, scores, and token types are added to the GGUF model using `self.gguf.add_token_list(tokens)`, `self.gguf.add_token_scores(scores)`, and `self.gguf.add_token_types(toktypes)` respectively.
- **Output**: The method does not return any value; it modifies the GGUF model by adding vocabulary metadata.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.extract_vocabulary_from_model`](#OutputFileextract_vocabulary_from_model)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.add\_meta\_special\_vocab<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_special_vocab}} -->
The `add_meta_special_vocab` method adds special vocabulary metadata to a GGUF file using a `SpecialVocab` instance.
- **Inputs**:
    - `svocab`: An instance of `gguf.SpecialVocab` that contains special vocabulary information to be added to the GGUF file.
- **Control Flow**:
    - The method calls the `add_to_gguf` method on the `svocab` object, passing the `gguf` attribute of the `OutputFile` instance as an argument.
- **Output**: The method does not return any value (returns `None`).
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.add\_tensor\_info<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.add_tensor_info}} -->
The `add_tensor_info` method adds metadata about a tensor to a GGUF file, including its shape, data type, and size in bytes.
- **Inputs**:
    - `name`: A string representing the name of the tensor.
    - `tensor`: An instance of `LazyTensor` representing the tensor whose information is to be added.
- **Control Flow**:
    - Calculate the total number of elements in the tensor by taking the product of its shape dimensions.
    - Retrieve the raw data type of the tensor using the 'ggml_type' attribute if available.
    - Determine the data type of the tensor, preferring 'quantized_type' if available, otherwise using the tensor's dtype.
    - Calculate the number of bytes required to store the tensor's data using its data type and number of elements.
    - Add the tensor's information, including its name, shape, data type, number of bytes, and raw data type, to the GGUF file using the `add_tensor_info` method of the `gguf` object.
- **Output**: The method does not return any value (returns `None`).
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.DataType.elements_to_bytes`](#DataTypeelements_to_bytes)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.write\_meta<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.write_meta}} -->
The `write_meta` method writes the header and key-value data to a file using the `gguf` writer.
- **Inputs**: None
- **Control Flow**:
    - The method calls `self.gguf.write_header_to_file()` to write the header information to the file.
    - It then calls `self.gguf.write_kv_data_to_file()` to write key-value data to the file.
- **Output**: The method does not return any value; it performs file writing operations.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.write\_tensor\_info<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.write_tensor_info}} -->
The `write_tensor_info` method writes tensor information data to a file using the `gguf` writer.
- **Inputs**: None
- **Control Flow**:
    - The method calls `write_ti_data_to_file` on the `gguf` attribute of the class instance.
- **Output**: The method does not return any value (returns `None`).
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.write\_tensor\_data<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.write_tensor_data}} -->
The `write_tensor_data` method writes tensor data from a model to a file, potentially quantizing the data based on the specified file type, while logging the process.
- **Inputs**:
    - `ftype`: A `GGMLFileType` enum value indicating the file type, which determines whether quantization is applied to the tensor data.
    - `model`: A `LazyModel` object representing the model whose tensor data is to be written.
    - `concurrency`: An integer specifying the level of concurrency to use for parallel processing of tensor data.
- **Control Flow**:
    - The method begins by using [`bounded_parallel_map`](#cpp/examples/convert_legacy_llamabounded_parallel_map) to process model items with `OutputFile.do_item`, creating an iterable of ndarrays (`ndarrays_inner`).
    - If `ftype` is `GGMLFileType.MostlyQ8_0`, it applies `OutputFile.maybe_do_quantize` to `ndarrays_inner` using [`bounded_parallel_map`](#cpp/examples/convert_legacy_llamabounded_parallel_map) with a process pool executor, otherwise it uses a simple map function.
    - The method records the start time and iterates over the zipped pairs of model items and processed ndarrays.
    - For each tensor, it calculates the elapsed time, formats the tensor's size, and logs the writing process with details about the tensor's name, size, data type, and elapsed time.
    - Finally, it writes the tensor data to the file using `self.gguf.write_tensor_data(ndarray)` for each processed ndarray.
- **Output**: The method does not return any value; it writes tensor data to a file and logs the process.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.bounded_parallel_map`](#cpp/examples/convert_legacy_llamabounded_parallel_map)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.close<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.close}} -->
The `close` method finalizes and closes the GGUFWriter instance associated with the OutputFile object.
- **Inputs**: None
- **Control Flow**:
    - The method calls the `close` method on the `gguf` attribute, which is an instance of `gguf.GGUFWriter`.
- **Output**: The method does not return any value; it performs an action to close the GGUFWriter instance.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.write\_vocab\_only<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.write_vocab_only}} -->
The `write_vocab_only` method writes vocabulary and metadata information to a specified output file using the GGUF format.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `fname_out`: A `Path` object representing the output file path where the vocabulary and metadata will be written.
    - `params`: An instance of `Params` containing model parameters such as vocabulary size, embedding length, etc.
    - `vocab`: An instance of `Vocab` representing the vocabulary to be written to the output file.
    - `svocab`: An instance of `gguf.SpecialVocab` representing special vocabulary tokens to be added to the output file.
    - `endianess`: An optional `gguf.GGUFEndian` value indicating the byte order for the output file, defaulting to `gguf.GGUFEndian.LITTLE`.
    - `pad_vocab`: A boolean indicating whether to pad the vocabulary if the model's expected vocabulary size is larger than the actual size, defaulting to `False`.
    - `metadata`: An optional `gguf.Metadata` object containing additional metadata about the model, such as author, version, and description.
- **Control Flow**:
    - The method begins by checking the vocabulary size using the [`check_vocab_size`](#cpp/examples/convert_legacy_llamacheck_vocab_size) function, which ensures the vocabulary size matches the model's expected size or pads it if necessary.
    - An `OutputFile` object is instantiated with the specified output file path and endianess.
    - Metadata about the model, architecture, vocabulary, and special vocabulary is added to the `OutputFile` object using its respective methods ([`add_meta_model`](#OutputFileadd_meta_model), [`add_meta_arch`](#OutputFileadd_meta_arch), [`add_meta_vocab`](#OutputFileadd_meta_vocab), [`add_meta_special_vocab`](#OutputFileadd_meta_special_vocab)).
    - The metadata is written to the output file using the [`write_meta`](#OutputFilewrite_meta) method of the `OutputFile` object.
    - Finally, the `OutputFile` object is closed to finalize the writing process.
- **Output**: The method does not return any value; it writes the vocabulary and metadata to the specified output file.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.check_vocab_size`](#cpp/examples/convert_legacy_llamacheck_vocab_size)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_model`](#OutputFileadd_meta_model)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_arch`](#OutputFileadd_meta_arch)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_vocab`](#OutputFileadd_meta_vocab)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_special_vocab`](#OutputFileadd_meta_special_vocab)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.write_meta`](#OutputFilewrite_meta)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.close`](#OutputFileclose)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.do\_item<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.do_item}} -->
The `do_item` method processes a tuple containing a string and a `LazyTensor`, loading the tensor and converting it to a GGML-compatible format, then returning its data type and ndarray.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `item`: A tuple consisting of a string (name) and a `LazyTensor` object.
- **Control Flow**:
    - The method unpacks the input tuple into `name` and `lazy_tensor`.
    - It calls the [`load`](#LazyTensorload) method on `lazy_tensor` to retrieve the tensor.
    - The loaded tensor is converted to a GGML-compatible format using the [`to_ggml`](#Tensorto_ggml) method.
    - The method returns a tuple containing the data type of the `lazy_tensor` and the ndarray of the converted tensor.
- **Output**: A tuple containing the data type of the `lazy_tensor` and its corresponding ndarray.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor.load`](#LazyTensorload)
    - [`llama.cpp/examples/convert_legacy_llama.Tensor.to_ggml`](#Tensorto_ggml)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.maybe\_do\_quantize<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.maybe_do_quantize}} -->
The `maybe_do_quantize` method checks if a given data type is quantized and applies quantization to the associated array if it is.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `item`: A tuple consisting of a `DataType` and an `NDArray`.
- **Control Flow**:
    - Extracts the data type (`dt`) and array (`arr`) from the input tuple `item`.
    - Checks if `dt` is an instance of `QuantizedDataType`.
    - If `dt` is not a `QuantizedDataType`, returns the array `arr` as is.
    - If `dt` is a `QuantizedDataType`, applies the [`quantize`](#QuantizedDataTypequantize) method of `dt` to `arr` and returns the result.
- **Output**: Returns an `NDArray`, which is either the original array or the quantized array, depending on the data type.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.QuantizedDataType.quantize`](#QuantizedDataTypequantize)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)


---
#### OutputFile\.write\_all<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.OutputFile.write_all}} -->
The `write_all` method writes a complete model, including metadata, tensor information, and tensor data, to an output file.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `fname_out`: The output file path where the model data will be written.
    - `ftype`: The file type for the GGML model, indicating the data type for tensors.
    - `params`: An instance of `Params` containing model parameters such as vocabulary size, embedding dimensions, etc.
    - `model`: A `LazyModel` dictionary containing model tensors, where each tensor is a `LazyTensor` object.
    - `vocab`: An instance of `BaseVocab` representing the model's vocabulary.
    - `svocab`: An instance of `gguf.SpecialVocab` representing special vocabulary tokens.
    - `concurrency`: The number of concurrent threads or processes to use for writing tensor data.
    - `endianess`: The byte order for the output file, either little or big endian.
    - `pad_vocab`: A boolean indicating whether to pad the vocabulary to match the model's expected size.
    - `metadata`: Optional metadata about the model, such as author, version, and description.
- **Control Flow**:
    - The method begins by checking the vocabulary size using [`check_vocab_size`](#cpp/examples/convert_legacy_llamacheck_vocab_size) to ensure it matches the model's expectations.
    - An `OutputFile` object is instantiated with the specified output file path and endianess.
    - Metadata about the model and its architecture is added to the `OutputFile` using [`add_meta_model`](#OutputFileadd_meta_model) and [`add_meta_arch`](#OutputFileadd_meta_arch).
    - If the vocabulary is of type `Vocab`, metadata about the vocabulary and special vocabulary is added; otherwise, only the tokenizer model is added.
    - Tensor information for each tensor in the model is added to the `OutputFile` using [`add_tensor_info`](#OutputFileadd_tensor_info).
    - The metadata and tensor information are written to the output file using [`write_meta`](#OutputFilewrite_meta) and [`write_tensor_info`](#OutputFilewrite_tensor_info).
    - Tensor data is written to the output file using [`write_tensor_data`](#OutputFilewrite_tensor_data), which handles concurrency and optional quantization.
    - The `OutputFile` is closed to finalize the writing process.
- **Output**: The method does not return any value; it writes the model data to the specified output file.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.check_vocab_size`](#cpp/examples/convert_legacy_llamacheck_vocab_size)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_model`](#OutputFileadd_meta_model)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_arch`](#OutputFileadd_meta_arch)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_vocab`](#OutputFileadd_meta_vocab)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_meta_special_vocab`](#OutputFileadd_meta_special_vocab)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.add_tensor_info`](#OutputFileadd_tensor_info)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.write_meta`](#OutputFilewrite_meta)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.write_tensor_info`](#OutputFilewrite_tensor_info)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.write_tensor_data`](#OutputFilewrite_tensor_data)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.close`](#OutputFileclose)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.OutputFile`](#cpp/examples/convert_legacy_llamaOutputFile)  (Base Class)



---
### VocabFactory<!-- {{#class:llama.cpp/examples/convert_legacy_llama.VocabFactory}} -->
- **Members**:
    - `_VOCAB_CLASSES`: A list of vocabulary classes that the factory can create.
    - `path`: The file path where the vocabulary files are located.
- **Description**: The `VocabFactory` class is responsible for creating and managing vocabulary objects for different types of tokenizers. It maintains a list of supported vocabulary classes and provides methods to load a vocabulary based on specified types. The class can also create special vocabularies that include additional configurations such as loading merge files for BPE vocabularies. This factory pattern allows for flexible and dynamic creation of vocabulary objects based on the needs of the application.
- **Methods**:
    - [`llama.cpp/examples/convert_legacy_llama.VocabFactory.__init__`](#VocabFactory__init__)
    - [`llama.cpp/examples/convert_legacy_llama.VocabFactory._create_special_vocab`](#VocabFactory_create_special_vocab)
    - [`llama.cpp/examples/convert_legacy_llama.VocabFactory._create_vocab_by_path`](#VocabFactory_create_vocab_by_path)
    - [`llama.cpp/examples/convert_legacy_llama.VocabFactory.load_vocab`](#VocabFactoryload_vocab)

**Methods**

---
#### VocabFactory\.\_\_init\_\_<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.VocabFactory.__init__}} -->
The `__init__` method initializes an instance of the `VocabFactory` class with a specified path.
- **Inputs**:
    - `path`: A `Path` object representing the file path to be used by the `VocabFactory` instance.
- **Control Flow**:
    - The method assigns the provided `path` argument to the `path` attribute of the `VocabFactory` instance.
- **Output**: This method does not return any value; it initializes the instance attributes.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.VocabFactory`](#cpp/examples/convert_legacy_llamaVocabFactory)  (Base Class)


---
#### VocabFactory\.\_create\_special\_vocab<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.VocabFactory._create_special_vocab}} -->
The `_create_special_vocab` method creates a `gguf.SpecialVocab` object based on the provided vocabulary and model path.
- **Inputs**:
    - `vocab`: An instance of `BaseVocab` or its subclass, representing the vocabulary to be used.
    - `model_parent_path`: A `Path` object representing the parent directory path of the model.
- **Control Flow**:
    - Determine if `load_merges` should be `True` by checking if the vocabulary name is 'bpe'.
    - Set `n_vocab` to the vocabulary size if `vocab` is an instance of `Vocab`, otherwise set it to `None`.
    - Create and return a `gguf.SpecialVocab` object using the `model_parent_path`, `load_merges`, `special_token_types` (set to `None`), and `n_vocab`.
- **Output**: Returns a `gguf.SpecialVocab` object initialized with the specified parameters.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.VocabFactory`](#cpp/examples/convert_legacy_llamaVocabFactory)  (Base Class)


---
#### VocabFactory\.\_create\_vocab\_by\_path<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.VocabFactory._create_vocab_by_path}} -->
The `_create_vocab_by_path` method selects and initializes a vocabulary class based on the provided list of vocabulary types.
- **Inputs**:
    - `vocab_types`: A list of strings representing the types of vocabularies to be considered for selection and initialization.
- **Control Flow**:
    - Initialize a dictionary `vocab_classes` mapping vocabulary class names to their respective classes from `_VOCAB_CLASSES`.
    - Create an empty dictionary `selected_vocabs` to store the selected vocabulary classes.
    - Iterate over each vocabulary type in `vocab_types`.
    - For each type, attempt to add the corresponding class from `vocab_classes` to `selected_vocabs`. If the type is not found, raise a `ValueError`.
    - Iterate over the items in `selected_vocabs` to attempt initializing a vocabulary instance with the class and `self.path`.
    - If a `FileNotFoundError` is raised during initialization, continue to the next class without breaking the loop.
    - If no vocabulary is successfully initialized, raise a `FileNotFoundError`.
    - Log the successful loading of the vocabulary file and return the initialized vocabulary instance.
- **Output**: Returns an instance of the `Vocab` class that was successfully initialized from the provided path.
- **See also**: [`llama.cpp/examples/convert_legacy_llama.VocabFactory`](#cpp/examples/convert_legacy_llamaVocabFactory)  (Base Class)


---
#### VocabFactory\.load\_vocab<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.VocabFactory.load_vocab}} -->
The `load_vocab` method loads a vocabulary and a special vocabulary based on the provided vocabulary types and model parent path.
- **Inputs**:
    - `vocab_types`: A list of strings representing the types of vocabularies to load, or None if no specific types are provided.
    - `model_parent_path`: A Path object representing the parent directory of the model, used to create the special vocabulary.
- **Control Flow**:
    - Check if `vocab_types` is None; if so, instantiate `vocab` as `NoVocab()`.
    - If `vocab_types` is not None, call [`_create_vocab_by_path`](#VocabFactory_create_vocab_by_path) with `vocab_types` to create the `vocab`.
    - Call [`_create_special_vocab`](#VocabFactory_create_special_vocab) with `vocab` and `model_parent_path` to create the `special_vocab`.
    - Return a tuple containing `vocab` and `special_vocab`.
- **Output**: A tuple containing a `BaseVocab` object and a `gguf.SpecialVocab` object.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.VocabFactory._create_vocab_by_path`](#VocabFactory_create_vocab_by_path)
    - [`llama.cpp/examples/convert_legacy_llama.VocabFactory._create_special_vocab`](#VocabFactory_create_special_vocab)
- **See also**: [`llama.cpp/examples/convert_legacy_llama.VocabFactory`](#cpp/examples/convert_legacy_llamaVocabFactory)  (Base Class)



# Functions

---
### permute<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.permute}} -->
The `permute` function reshapes and permutes a given weight matrix based on the specified number of heads and key-value heads.
- **Inputs**:
    - `weights`: An NDArray representing the weight matrix to be permuted.
    - `n_head`: An integer representing the number of heads for reshaping the weight matrix.
    - `n_head_kv`: An integer representing the number of key-value heads, which may override `n_head` if they are different.
- **Control Flow**:
    - Check if `n_head_kv` is not None and `n_head` is different from `n_head_kv`; if so, set `n_head` to `n_head_kv`.
    - Reshape the `weights` NDArray into a shape that includes `n_head`, 2, and the appropriate dimensions based on the original shape of `weights`.
    - Swap the axes of the reshaped array to change the order of dimensions.
    - Reshape the permuted array back to the original shape of `weights`.
- **Output**: Returns an NDArray with the same shape as the input `weights`, but with its elements permuted according to the specified head configuration.
- **Functions called**:
    - [`llama.cpp/convert_lora_to_gguf.LoraTorchTensor.reshape`](../convert_lora_to_gguf.py.driver.md#LoraTorchTensorreshape)
    - [`llama.cpp/convert_lora_to_gguf.LoraTorchTensor.swapaxes`](../convert_lora_to_gguf.py.driver.md#LoraTorchTensorswapaxes)


---
### bf16\_to\_fp32<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.bf16_to_fp32}} -->
The function `bf16_to_fp32` converts a bfloat16 (BF16) numpy array to a float32 (FP32) numpy array.
- **Inputs**:
    - `bf16_arr`: A numpy array of dtype uint16 representing bfloat16 values.
- **Control Flow**:
    - Assert that the input array `bf16_arr` has dtype uint16, raising an error if not.
    - Convert the `bf16_arr` to dtype uint32 and left-shift the values by 16 bits to prepare for conversion to float32.
    - Return the resulting array viewed as a float32 numpy array.
- **Output**: A numpy array of dtype float32, representing the converted values from the input bfloat16 array.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.Tensor.astype`](#Tensorastype)
    - [`llama.cpp/convert_lora_to_gguf.LoraTorchTensor.view`](../convert_lora_to_gguf.py.driver.md#LoraTorchTensorview)


---
### load\_unquantized<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.load_unquantized}} -->
The `load_unquantized` function loads an unquantized tensor from a `LazyTensor`, optionally converting its data type if specified.
- **Inputs**:
    - `lazy_tensor`: A `LazyTensor` object representing the tensor to be loaded.
    - `expected_dtype`: An optional argument specifying the expected data type of the tensor's ndarray; defaults to `None`.
    - `convert`: A boolean flag indicating whether to convert the tensor's data type to `expected_dtype` if it does not match; defaults to `False`.
- **Control Flow**:
    - Load the tensor from the `lazy_tensor` using its [`load`](#LazyTensorload) method.
    - Assert that the loaded tensor is an instance of `UnquantizedTensor`.
    - Check that the shape of the loaded tensor matches the expected shape from `lazy_tensor`.
    - If `expected_dtype` is provided and does not match the tensor's current dtype, check the `convert` flag.
    - If `convert` is `True`, convert the tensor's ndarray to the `expected_dtype`.
    - If `convert` is `False`, raise a `ValueError` indicating the dtype mismatch.
- **Output**: Returns the ndarray of the loaded `UnquantizedTensor`.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor.load`](#LazyTensorload)
    - [`llama.cpp/examples/convert_legacy_llama.Tensor.astype`](#Tensorastype)


---
### merge\_sharded<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.merge_sharded}} -->
The `merge_sharded` function combines multiple sharded LazyModel instances into a single LazyModel by merging tensors along specified dimensions.
- **Inputs**:
    - `models`: A list of LazyModel instances, each representing a part of a sharded model.
- **Control Flow**:
    - Initialize a dictionary `names` to store unique tensor names from all models, preserving order.
    - Define an inner function `convert` that takes a tensor name and processes the corresponding tensors from all models.
    - For each tensor name, gather the corresponding LazyTensor instances from all models into `lazy_tensors`.
    - If there is only one LazyTensor, return it directly to avoid unnecessary processing.
    - If the tensor is 1-dimensional, return the first LazyTensor as it is duplicated across files.
    - Determine the axis to concatenate along based on the tensor name (columns for certain embeddings and weights, rows otherwise).
    - Calculate the concatenated shape by summing the dimensions along the chosen axis.
    - Define a `load` function to concatenate the unquantized tensors along the specified axis and return an UnquantizedTensor.
    - Create a LazyTensor with the `load` function, concatenated shape, data type, and a description, and return it.
    - Return a dictionary mapping each tensor name to its corresponding processed LazyTensor.
- **Output**: A LazyModel, which is a dictionary mapping tensor names to LazyTensor instances, representing the merged model.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.load_unquantized`](#cpp/examples/convert_legacy_llamaload_unquantized)
    - [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor`](#cpp/examples/convert_legacy_llamaUnquantizedTensor)
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor)


---
### merge\_multifile\_models<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.merge_multifile_models}} -->
The function `merge_multifile_models` combines multiple [`ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus) objects into a single [`ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus) object by merging their models, paths, and vocabularies.
- **Inputs**:
    - `models_plus`: A list of [`ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus) objects, each containing a model, paths, format, and optionally a vocabulary.
- **Control Flow**:
    - Extracts the set of formats from the input [`ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus) objects and asserts that all formats are the same.
    - Collects all paths from the input [`ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus) objects into a single list.
    - Attempts to find the first non-None vocabulary from the input [`ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus) objects.
    - Checks if any model in the input list contains the key 'model.embed_tokens.weight'.
    - If the key is found, it merges the models by updating a dictionary with each model's contents.
    - If the key is not found, it calls [`merge_sharded`](#cpp/examples/convert_legacy_llamamerge_sharded) to merge the models by concatenating tensors along specific axes.
    - Returns a new [`ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus) object with the merged model, paths, format, and vocabulary.
- **Output**: A [`ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus) object containing the merged model, paths, format, and vocabulary.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.merge_sharded`](#cpp/examples/convert_legacy_llamamerge_sharded)
    - [`llama.cpp/examples/convert_legacy_llama.ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus)


---
### permute\_lazy<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.permute_lazy}} -->
The `permute_lazy` function creates a new [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) that applies a permutation operation on the loaded tensor data using specified head dimensions.
- **Inputs**:
    - `lazy_tensor`: A [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) object representing the tensor to be permuted.
    - `n_head`: An integer representing the number of heads for the permutation.
    - `n_head_kv`: An integer representing the number of key-value heads for the permutation.
- **Control Flow**:
    - Defines an inner function [`load`](#Paramsload) that loads the tensor from `lazy_tensor` and applies the [`permute`](../convert_lora_to_gguf.py.driver.md#LoraTorchTensorpermute) method with `n_head` and `n_head_kv` as arguments.
    - Returns a new [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) object initialized with the [`load`](#Paramsload) function, the shape and data type of the original `lazy_tensor`, and a description indicating the permutation operation.
- **Output**: A [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) object that represents the permuted tensor.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.Params.load`](#Paramsload)
    - [`llama.cpp/convert_lora_to_gguf.LoraTorchTensor.permute`](../convert_lora_to_gguf.py.driver.md#LoraTorchTensorpermute)
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor)


---
### permute\_part\_lazy<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.permute_part_lazy}} -->
The `permute_part_lazy` function creates a new [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) that represents a permuted part of an existing [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) based on specified parameters.
- **Inputs**:
    - `lazy_tensor`: An instance of [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) that represents the tensor to be permuted.
    - `n_part`: An integer specifying the part of the tensor to permute.
    - `n_head`: An integer specifying the number of heads for permutation.
    - `n_head_kv`: An integer specifying the number of key-value heads for permutation.
- **Control Flow**:
    - Defines an inner function [`load`](#Paramsload) that loads the tensor and applies the [`permute_part`](#Tensorpermute_part) method with the given parameters `n_part`, `n_head`, and `n_head_kv`.
    - Copies the shape of the input `lazy_tensor` and modifies the first dimension by dividing it by 3.
    - Returns a new [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) with the [`load`](#Paramsload) function, modified shape, the same data type as the input `lazy_tensor`, and a description indicating the permutation.
- **Output**: A [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) object that represents the permuted part of the input tensor.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.Params.load`](#Paramsload)
    - [`llama.cpp/examples/convert_legacy_llama.Tensor.permute_part`](#Tensorpermute_part)
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor)


---
### part\_lazy<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.part_lazy}} -->
The `part_lazy` function creates a new [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) that represents a partitioned version of the input `lazy_tensor` based on the specified partition index `n_part`.
- **Inputs**:
    - `lazy_tensor`: A [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) object that represents a tensor to be partitioned.
    - `n_part`: An integer representing the partition index to extract from the `lazy_tensor`.
- **Control Flow**:
    - Defines an inner function [`load`](#Paramsload) that loads the tensor from `lazy_tensor` and extracts the specified partition using the [`part`](#Tensorpart) method.
    - Copies the shape of the input `lazy_tensor` and modifies the first dimension to be one-third of its original size, reflecting the partitioning.
    - Returns a new [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) object initialized with the [`load`](#Paramsload) function, the modified shape, the data type of the input `lazy_tensor`, and a description indicating it is a partition.
- **Output**: A [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) object representing the specified partition of the input `lazy_tensor`.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.Params.load`](#Paramsload)
    - [`llama.cpp/examples/convert_legacy_llama.Tensor.part`](#Tensorpart)
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor)


---
### pack\_experts\_lazy<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.pack_experts_lazy}} -->
The `pack_experts_lazy` function creates a new [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) by combining a list of [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) objects into a single tensor with an additional dimension representing the number of input tensors.
- **Inputs**:
    - `lazy_tensors`: A list of [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) objects that are to be packed into a single [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor).
- **Control Flow**:
    - Defines an inner function [`load`](#Paramsload) that loads each [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) in the `lazy_tensors` list and collects their underlying `Tensor` objects.
    - Creates a new [`UnquantizedTensor`](#cpp/examples/convert_legacy_llamaUnquantizedTensor) by stacking the `ndarray` of each loaded `Tensor` into a single numpy array.
    - Copies the shape of the first [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) in the list and inserts the number of [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) objects at the beginning of the shape to account for the new dimension.
    - Returns a new [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) initialized with the [`load`](#Paramsload) function, the new shape, the data type of the first [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor), and a description string.
- **Output**: Returns a [`LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor) that represents the packed tensor with an additional dimension for the number of input tensors.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.Params.load`](#Paramsload)
    - [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor`](#cpp/examples/convert_legacy_llamaUnquantizedTensor)
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor)


---
### lazy\_load\_torch\_file<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.lazy_load_torch_file}} -->
The function `lazy_load_torch_file` loads a PyTorch model from a zip file, unpickles it, and returns it as a [`ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus) object.
- **Inputs**:
    - `outer_fp`: A file-like object representing the zip file containing the PyTorch model.
    - `path`: A `Path` object representing the file path to the model.
- **Control Flow**:
    - Open the zip file using `zipfile.ZipFile` with `outer_fp` as the file pointer.
    - Extract the list of file names in the zip file that end with '.pkl'.
    - Assert that there is exactly one pickle file in the zip file.
    - Open the pickle file from the zip file for reading.
    - Create a [`LazyUnpickler`](#cpp/examples/convert_legacy_llamaLazyUnpickler) instance with the opened pickle file, the base path of the data, and the zip file.
    - Load the model using the [`LazyUnpickler`](#cpp/examples/convert_legacy_llamaLazyUnpickler) instance.
    - If the loaded model contains a 'model' key, extract the model from this key.
    - Convert the model to a dictionary using `dict(model.items())`.
    - Return a [`ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus) object initialized with the model dictionary, the path, the format 'torch', and `None` for the vocab.
- **Output**: A [`ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus) object containing the loaded model, the path, the format 'torch', and `None` for the vocab.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.LazyUnpickler`](#cpp/examples/convert_legacy_llamaLazyUnpickler)
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor.load`](#LazyTensorload)
    - [`llama.cpp/examples/convert_legacy_llama.ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus)


---
### lazy\_load\_safetensors\_file<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.lazy_load_safetensors_file}} -->
The `lazy_load_safetensors_file` function loads a model from a safetensors file using lazy loading techniques to minimize memory usage.
- **Inputs**:
    - `fp`: A file-like object (IO[bytes]) representing the safetensors file to be read.
    - `path`: A Path object representing the file path of the safetensors file.
- **Control Flow**:
    - Read the first 8 bytes from the file to determine the header size using struct unpacking.
    - Read the header from the file using the determined header size and parse it as a JSON object.
    - Create a memory-mapped view of the file to access the data without loading it entirely into memory.
    - Define a nested function `convert` that takes tensor information and returns a LazyTensor object.
    - Iterate over the items in the header, excluding metadata, and convert each tensor information into a LazyTensor using the `convert` function.
    - Return a ModelPlus object containing the model, file paths, format, and vocabulary.
- **Output**: A ModelPlus object containing the lazily loaded model, the file path, the format 'safetensors', and no vocabulary.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.UnquantizedTensor`](#cpp/examples/convert_legacy_llamaUnquantizedTensor)
    - [`llama.cpp/convert_lora_to_gguf.LoraTorchTensor.reshape`](../convert_lora_to_gguf.py.driver.md#LoraTorchTensorreshape)
    - [`llama.cpp/examples/convert_legacy_llama.LazyTensor`](#cpp/examples/convert_legacy_llamaLazyTensor)
    - [`llama.cpp/examples/convert_legacy_llama.ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus)


---
### must\_read<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.must_read}} -->
The `must_read` function reads a specified number of bytes from a file-like object and raises an EOFError if the end of the file is reached before reading the required number of bytes.
- **Inputs**:
    - `fp`: A file-like object that supports reading bytes.
    - `length`: An integer specifying the number of bytes to read from the file-like object.
- **Control Flow**:
    - Read the specified number of bytes from the file-like object using `fp.read(length)` and store it in `ret`.
    - Check if the length of `ret` is less than the specified `length`.
    - If the length of `ret` is less than `length`, raise an `EOFError` with a message indicating that the end of the file was reached unexpectedly.
- **Output**: Returns the bytes read from the file-like object if the specified number of bytes is successfully read.


---
### lazy\_load\_file<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.lazy_load_file}} -->
The `lazy_load_file` function loads a model file in either PyTorch or safetensors format, caching the result for future calls.
- **Decorators**: `@functools.lru_cache`
- **Inputs**:
    - `path`: A `Path` object representing the file path to the model file to be loaded.
- **Control Flow**:
    - Open the file at the given path in binary read mode.
    - Read the first 8 bytes of the file to determine its format.
    - If the first two bytes are 'PK', treat the file as a PyTorch zip file and call [`lazy_load_torch_file`](#cpp/examples/convert_legacy_llamalazy_load_torch_file).
    - If the unpacked value of the first 8 bytes is less than 16 MB, treat the file as a safetensors file and call [`lazy_load_safetensors_file`](#cpp/examples/convert_legacy_llamalazy_load_safetensors_file).
    - If neither condition is met, raise a `ValueError` indicating an unknown format.
- **Output**: Returns a `ModelPlus` object representing the loaded model, which includes the model data, file paths, format, and vocabulary.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.lazy_load_torch_file`](#cpp/examples/convert_legacy_llamalazy_load_torch_file)
    - [`llama.cpp/examples/convert_legacy_llama.lazy_load_safetensors_file`](#cpp/examples/convert_legacy_llamalazy_load_safetensors_file)


---
### bounded\_parallel\_map<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.bounded_parallel_map}} -->
The `bounded_parallel_map` function performs a parallel map operation with backpressure to prevent excessive memory usage by limiting the number of concurrent function calls.
- **Inputs**:
    - `func`: A callable function that takes an input of type `In` and returns an output of type `Out`.
    - `iterable`: An iterable collection of inputs of type `In` to be processed by the `func`.
    - `concurrency`: An integer specifying the maximum number of concurrent function calls to be made.
    - `max_workers`: An optional integer specifying the maximum number of worker threads or processes to use; defaults to `None`, which lets the executor decide.
    - `use_processpool_executor`: A boolean indicating whether to use `ProcessPoolExecutor` instead of `ThreadPoolExecutor`; defaults to `False`.
- **Control Flow**:
    - Check if concurrency is less than 2; if so, use a simple map function to yield results.
    - Convert the iterable to an iterator.
    - Determine the executor class to use based on the `use_processpool_executor` flag.
    - Initialize the executor with the specified `max_workers`.
    - Submit initial tasks to the executor up to the specified concurrency level.
    - Enter a loop to process futures as they complete, yielding results one by one.
    - Continue submitting new tasks as long as there are more items in the iterable and the concurrency limit is not reached.
    - Handle `StopIteration` exceptions to mark when the iterable is exhausted.
- **Output**: An iterable of outputs of type `Out`, produced by applying `func` to each input in the `iterable`.


---
### check\_vocab\_size<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.check_vocab_size}} -->
The `check_vocab_size` function verifies and adjusts the vocabulary size of a model against its parameters, raising errors or warnings if mismatches are found.
- **Inputs**:
    - `params`: An instance of the `Params` class containing model parameters, including the expected vocabulary size (`n_vocab`).
    - `vocab`: An instance of `BaseVocab` or its subclass, representing the vocabulary of the model, with attributes like `vocab_size` and `fname_tokenizer`.
    - `pad_vocab`: A boolean flag indicating whether to pad the vocabulary if the model's expected vocabulary size is greater than the current vocabulary size.
- **Control Flow**:
    - Check if the model's vocabulary size (`n_vocab`) is set to -1 in `params`; if so, raise a `ValueError` suggesting manual update.
    - If `vocab` is not an instance of `Vocab`, return immediately as the model has no vocabulary to check.
    - Compare `params.n_vocab` with `vocab.vocab_size`; if they match, log a warning about ignoring `added_tokens.json` and return.
    - If `pad_vocab` is `True` and `params.n_vocab` is greater than `vocab.vocab_size`, calculate the number of padding tokens needed, log the padding action, add dummy tokens to the vocabulary, update `vocab.vocab_size`, and return.
    - If there is a vocabulary size mismatch, construct an error message indicating the mismatch and possible missing `added_tokens.json`, and raise a `ValueError`.
- **Output**: The function does not return any value but may raise a `ValueError` if there is a vocabulary size mismatch or if `params.n_vocab` is set to -1.


---
### pick\_output\_type<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.pick_output_type}} -->
The function `pick_output_type` determines the appropriate `GGMLFileType` for a model based on the provided output type string and the data type of a specific tensor in the model.
- **Inputs**:
    - `model`: A `LazyModel` dictionary where keys are tensor names and values are `LazyTensor` objects, representing the model's tensors.
    - `output_type_str`: An optional string indicating the desired output type, which can be 'f32', 'f16', 'q8_0', or `None`.
- **Control Flow**:
    - Retrieve the data type of the tensor associated with the key `gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ATTN_Q].format(bid=0) + ".weight"` from the model.
    - Check if `output_type_str` is 'f32' or if it is `None` and the tensor's data type is either `DT_F32` or `DT_BF16`. If so, return `GGMLFileType.AllF32`.
    - Check if `output_type_str` is 'f16' or if it is `None` and the tensor's data type is `DT_F16`. If so, return `GGMLFileType.MostlyF16`.
    - Check if `output_type_str` is 'q8_0'. If so, return `GGMLFileType.MostlyQ8_0`.
    - If none of the conditions are met, create a dictionary mapping tensor names to their data types and raise a `ValueError` with this information.
- **Output**: Returns a `GGMLFileType` enum value indicating the selected output type for the model.


---
### per\_model\_weight\_count\_estimation<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.per_model_weight_count_estimation}} -->
The function estimates the total, shared, and expert parameter counts for a given set of model tensors.
- **Inputs**:
    - `tensors`: An iterable of tuples, where each tuple contains a string (tensor name) and a LazyTensor object.
- **Control Flow**:
    - Initialize total_params, shared_params, and expert_params to zero.
    - Iterate over each tensor name and LazyTensor in the input iterable.
    - Skip tensors with names ending in specific suffixes ('.attention.masked_bias', '.attention.bias', '.rotary_emb.inv_freq').
    - Calculate the product of dimensions for each tensor to get the total number of weights in that tensor.
    - If the tensor name contains '.experts.', check if it is the first expert ('.experts.0.') and add its weight count to expert_params.
    - Otherwise, add the weight count to shared_params.
    - Add the weight count to total_params for every tensor.
    - Return a tuple containing total_params, shared_params, and expert_params.
- **Output**: A tuple of three integers representing the total number of parameters, the number of shared parameters, and the number of expert parameters.


---
### convert\_to\_output\_type<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.convert_to_output_type}} -->
The function `convert_to_output_type` converts the data type of each tensor in a `LazyModel` to a specified `GGMLFileType`.
- **Inputs**:
    - `model`: A `LazyModel`, which is a dictionary mapping tensor names to `LazyTensor` objects.
    - `output_type`: A `GGMLFileType` that specifies the desired output data type for the tensors.
- **Control Flow**:
    - Iterates over each tensor in the `model` dictionary.
    - For each tensor, it calls the [`astype`](#Tensorastype) method on the `LazyTensor` to convert it to the data type specified by `output_type.type_for_tensor(name, tensor)`.
    - Constructs a new dictionary with the same keys as `model`, but with the converted tensors as values.
- **Output**: A new `LazyModel` dictionary with tensors converted to the specified `output_type`.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.Tensor.astype`](#Tensorastype)
    - [`llama.cpp/examples/convert_legacy_llama.GGMLFileType.type_for_tensor`](#GGMLFileTypetype_for_tensor)


---
### convert\_model\_names<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.convert_model_names}} -->
The `convert_model_names` function processes a `LazyModel` by merging expert tensors, permuting certain tensors, and renaming them according to a mapping, while optionally skipping unknown tensors.
- **Inputs**:
    - `model`: A `LazyModel` dictionary containing tensor names as keys and `LazyTensor` objects as values.
    - `params`: A `Params` object containing model parameters such as number of layers, number of experts, and head counts.
    - `skip_unknown`: A boolean flag indicating whether to skip unknown tensor names or raise an error.
- **Control Flow**:
    - Initialize a `TensorNameMap` object `tmap` with architecture and number of layers from `params`.
    - Create a set `should_skip` containing tensor types to skip based on the architecture.
    - If the model has experts, iterate over each layer and weight index to merge expert tensors into a single tensor, deleting the original expert tensors from the model.
    - Iterate over layers to permute or unpack and permute certain tensors, updating the model with the permuted tensors and deleting packed tensors if necessary.
    - Iterate over the model's items to rename tensors using `tmap`, skipping or raising an error for unknown tensor names based on `skip_unknown`, and skipping tensors in `should_skip`.
    - Log the renaming process and add the renamed tensors to the output model.
- **Output**: A new `LazyModel` dictionary with processed and renamed tensors.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.pack_experts_lazy`](#cpp/examples/convert_legacy_llamapack_experts_lazy)
    - [`llama.cpp/examples/convert_legacy_llama.permute_lazy`](#cpp/examples/convert_legacy_llamapermute_lazy)
    - [`llama.cpp/examples/convert_legacy_llama.permute_part_lazy`](#cpp/examples/convert_legacy_llamapermute_part_lazy)
    - [`llama.cpp/examples/convert_legacy_llama.part_lazy`](#cpp/examples/convert_legacy_llamapart_lazy)


---
### nth\_multifile\_path<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.nth_multifile_path}} -->
The function `nth_multifile_path` returns the nth path in a multi-file model sequence based on a given path and index.
- **Inputs**:
    - `path`: A `Path` object representing the initial path of a file in a multi-file model sequence.
    - `n`: An integer representing the index of the desired file in the multi-file model sequence.
- **Control Flow**:
    - Define a list of patterns to match different multi-file naming conventions.
    - Iterate over each pattern, checking if the pattern matches the name of the given path.
    - If a match is found, construct a new path by replacing the matched pattern with the nth file pattern.
    - Check if the newly constructed path exists; if it does, return this path.
    - If no matching path is found after checking all patterns, return `None`.
- **Output**: Returns a `Path` object representing the nth file in the multi-file model sequence if it exists, otherwise returns `None`.


---
### find\_multifile\_paths<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.find_multifile_paths}} -->
The function `find_multifile_paths` returns a list of all paths in a multi-file model given a path to one of the files.
- **Inputs**:
    - `path`: A `Path` object representing a file path that is part of a multi-file model.
- **Control Flow**:
    - Initialize an empty list `ret` to store the paths.
    - Iterate over an infinite sequence of integers using `itertools.count()`.
    - For each integer `i`, call [`nth_multifile_path`](#cpp/examples/convert_legacy_llamanth_multifile_path) with `path` and `i` to get the `nth_path`.
    - If `nth_path` is `None`, break the loop.
    - If `nth_path` is not `None`, append it to the `ret` list.
    - After the loop, check if `ret` is empty.
    - If `ret` is empty, return a list containing the original `path` as a single file.
    - If `ret` is not empty, return the list `ret`.
- **Output**: A list of `Path` objects representing all the paths in the multi-file model.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.nth_multifile_path`](#cpp/examples/convert_legacy_llamanth_multifile_path)


---
### load\_some\_model<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.load_some_model}} -->
The `load_some_model` function loads a model from a specified path, handling both single and multi-file models in various formats.
- **Inputs**:
    - `path`: A `Path` object representing the file or directory path from which to load the model.
- **Control Flow**:
    - Check if the provided path is a directory.
    - If it is a directory, search for model files using specific glob patterns for safetensors and PyTorch formats.
    - If no files are found, raise a `FileNotFoundError`.
    - If multiple files are found, raise a `ValueError`.
    - If a single file is found, set the path to this file.
    - Use [`find_multifile_paths`](#cpp/examples/convert_legacy_llamafind_multifile_paths) to get all paths related to the model file(s).
    - Iterate over each path, load the model file using [`lazy_load_file`](#cpp/examples/convert_legacy_llamalazy_load_file), and append the result to `models_plus`.
    - Merge all loaded models using [`merge_multifile_models`](#cpp/examples/convert_legacy_llamamerge_multifile_models).
- **Output**: Returns a `ModelPlus` object representing the loaded model, which includes the model data, paths, format, and vocabulary.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.find_multifile_paths`](#cpp/examples/convert_legacy_llamafind_multifile_paths)
    - [`llama.cpp/examples/convert_legacy_llama.lazy_load_file`](#cpp/examples/convert_legacy_llamalazy_load_file)
    - [`llama.cpp/examples/convert_legacy_llama.merge_multifile_models`](#cpp/examples/convert_legacy_llamamerge_multifile_models)


---
### default\_convention\_outfile<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.default_convention_outfile}} -->
The function `default_convention_outfile` generates a default output filename based on the provided file type, expert count, model parameters, and metadata.
- **Inputs**:
    - `file_type`: An instance of `GGMLFileType` that specifies the type of file (e.g., AllF32, MostlyF16, MostlyQ8_0).
    - `expert_count`: An integer or None indicating the number of experts, if applicable.
    - `model_params_count`: A tuple of three integers representing the model parameters count.
    - `metadata`: An instance of `gguf.Metadata` containing metadata information such as name, basename, finetune, version, and size_label.
- **Control Flow**:
    - Extracts the name, basename, finetune, version, and size_label from the metadata, using defaults if they are None.
    - Determines the output type string based on the provided `file_type`.
    - Calls `gguf.naming_convention` with the extracted and determined values to generate the output filename.
- **Output**: Returns a string representing the default output filename based on the naming convention.


---
### default\_outfile<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.default_outfile}} -->
The `default_outfile` function generates a default output file path for a model conversion process, ensuring it does not overwrite any input files.
- **Inputs**:
    - `model_paths`: A list of Path objects representing the paths to the model files.
    - `file_type`: An instance of GGMLFileType indicating the type of file to be generated.
    - `expert_count`: An optional integer representing the number of experts, or None if not applicable.
    - `model_params_count`: A tuple of three integers representing the count of model parameters.
    - `metadata`: An instance of gguf.Metadata containing metadata information about the model.
- **Control Flow**:
    - Call [`default_convention_outfile`](#cpp/examples/convert_legacy_llamadefault_convention_outfile) with the provided file_type, expert_count, model_params_count, and metadata to generate a default filename.
    - Construct a Path object `ret` by combining the parent directory of the first model path with the default filename suffixed by '.gguf'.
    - Check if the generated path `ret` is in the list of model_paths.
    - If `ret` is in model_paths, log an error message and exit the program with a status code of 1.
    - Return the generated Path object `ret`.
- **Output**: A Path object representing the default output file path for the model conversion.
- **Functions called**:
    - [`llama.cpp/examples/convert_legacy_llama.default_convention_outfile`](#cpp/examples/convert_legacy_llamadefault_convention_outfile)


---
### do\_dump\_model<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.do_dump_model}} -->
The `do_dump_model` function prints detailed information about a `ModelPlus` object, including its paths, format, vocab, and the shape, type, and description of each tensor in the model.
- **Inputs**:
    - `model_plus`: An instance of the `ModelPlus` class, which contains the model's data, paths, format, and vocabulary information.
- **Control Flow**:
    - Prints the paths of the `model_plus` object.
    - Prints the format of the `model_plus` object.
    - Prints the vocabulary of the `model_plus` object.
    - Iterates over each item in the `model_plus.model` dictionary.
    - For each tensor, prints its name, shape, data type, and description.
- **Output**: The function does not return any value; it outputs information to the console.


---
### main<!-- {{#callable:llama.cpp/examples/convert_legacy_llama.main}} -->
The `main` function processes command-line arguments to convert a LLaMA model to a GGML compatible file, handling various options for output format, verbosity, and metadata.
- **Inputs**:
    - `args_in`: A list of strings representing command-line arguments, or None to use sys.argv.
- **Control Flow**:
    - Initialize a list of output format choices, adding 'q8_0' if the system is little-endian.
    - Set up an argument parser with various options for model conversion, including output type, verbosity, and metadata.
    - Parse the provided command-line arguments.
    - Configure logging level based on verbosity and other flags.
    - Load metadata for the model using the provided metadata path, model directory, and model name.
    - If the '--get-outfile' flag is set, load the model, determine parameters, and print the default output file name, then return.
    - Raise an error if both '--no-vocab' and '--vocab-only' flags are set.
    - If the '--dump-single' flag is set, load and dump a single model file, then return.
    - Load the model if not in vocab-only mode, otherwise create a dummy model.
    - If the '--dump' flag is set, dump the model information and return.
    - Determine the system's endianess based on the '--big-endian' flag.
    - Load model parameters if needed, and set context size if not specified.
    - Load vocabulary and special vocabulary using the specified or default vocab directory and types.
    - If in vocab-only mode, write the vocabulary to the output file and return.
    - If the model has a built-in vocabulary and no vocab directory is specified, use the model's vocabulary.
    - Ensure parameters are loaded and set metadata name if not already set.
    - Estimate model parameters count and log information about the model and vocabulary.
    - Convert model names and types based on parameters and output type.
    - Determine the output file path, ensuring it does not overwrite input files.
    - Write the converted model to the output file using the specified concurrency and options.
- **Output**: The function does not return a value; it performs file conversion and writes output to a specified or default file path.
- **Functions called**:
    - [`llama.cpp/convert_lora_to_gguf.parse_args`](../convert_lora_to_gguf.py.driver.md#cpp/convert_lora_to_ggufparse_args)
    - [`llama.cpp/examples/convert_legacy_llama.load_some_model`](#cpp/examples/convert_legacy_llamaload_some_model)
    - [`llama.cpp/examples/convert_legacy_llama.convert_model_names`](#cpp/examples/convert_legacy_llamaconvert_model_names)
    - [`llama.cpp/examples/convert_legacy_llama.per_model_weight_count_estimation`](#cpp/examples/convert_legacy_llamaper_model_weight_count_estimation)
    - [`llama.cpp/examples/convert_legacy_llama.pick_output_type`](#cpp/examples/convert_legacy_llamapick_output_type)
    - [`llama.cpp/examples/convert_legacy_llama.default_convention_outfile`](#cpp/examples/convert_legacy_llamadefault_convention_outfile)
    - [`llama.cpp/examples/convert_legacy_llama.lazy_load_file`](#cpp/examples/convert_legacy_llamalazy_load_file)
    - [`llama.cpp/examples/convert_legacy_llama.do_dump_model`](#cpp/examples/convert_legacy_llamado_dump_model)
    - [`llama.cpp/examples/convert_legacy_llama.ModelPlus`](#cpp/examples/convert_legacy_llamaModelPlus)
    - [`llama.cpp/examples/convert_legacy_llama.VocabFactory`](#cpp/examples/convert_legacy_llamaVocabFactory)
    - [`llama.cpp/examples/convert_legacy_llama.VocabFactory.load_vocab`](#VocabFactoryload_vocab)
    - [`llama.cpp/examples/convert_legacy_llama.Params`](#cpp/examples/convert_legacy_llamaParams)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.write_vocab_only`](#OutputFilewrite_vocab_only)
    - [`llama.cpp/examples/convert_legacy_llama.convert_to_output_type`](#cpp/examples/convert_legacy_llamaconvert_to_output_type)
    - [`llama.cpp/examples/convert_legacy_llama.default_outfile`](#cpp/examples/convert_legacy_llamadefault_outfile)
    - [`llama.cpp/examples/convert_legacy_llama.OutputFile.write_all`](#OutputFilewrite_all)


