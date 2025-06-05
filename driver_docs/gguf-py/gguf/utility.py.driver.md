# Purpose
This Python code file provides a set of utilities for handling and processing model data, particularly focusing on the management of remote safetensor files from Hugging Face model repositories. The file includes functions for generating formatted filenames, calculating model parameter sizes in a human-readable format, and constructing naming conventions for models. The core functionality revolves around the `SafetensorRemote` class, which facilitates the retrieval and parsing of tensor metadata from remote safetensor files. This class includes methods for checking the existence of files, fetching metadata, and retrieving specific data ranges from remote files, making it a crucial component for applications that need to interact with large-scale model data stored remotely.

The code is structured to be used as a library, with no direct script execution, indicating its purpose for integration into larger systems or applications. It defines public APIs through its class methods and functions, allowing other parts of a software system to leverage its capabilities for model data management. The use of type annotations and the `dataclass` for `RemoteTensor` enhances code readability and maintainability, ensuring that the data structures and expected types are clear. Overall, this file provides specialized functionality for model data handling, particularly in the context of remote data access and manipulation, which is essential for machine learning workflows that rely on distributed model storage and processing.
# Imports and Dependencies

---
- `__future__.annotations`
- `dataclasses.dataclass`
- `typing.Literal`
- `os`
- `json`
- `requests`
- `urllib.parse.urlparse`


# Classes

---
### RemoteTensor<!-- {{#class:llama.cpp/gguf-py/gguf/utility.RemoteTensor}} -->
- **Decorators**: `@dataclass`
- **Members**:
    - `dtype`: The data type of the tensor.
    - `shape`: The shape of the tensor as a tuple of integers.
    - `offset_start`: The starting byte offset for the tensor data in the remote file.
    - `size`: The size in bytes of the tensor data.
    - `url`: The URL of the remote file containing the tensor data.
- **Description**: The `RemoteTensor` class is a data structure that represents a tensor stored remotely, encapsulating its data type, shape, byte offset, size, and the URL of the remote file. It provides a method to retrieve the tensor data as a bytearray from the specified remote location, facilitating the handling of tensor data stored in remote safetensor files.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/utility.RemoteTensor.data`](#RemoteTensordata)

**Methods**

---
#### RemoteTensor\.data<!-- {{#callable:llama.cpp/gguf-py/gguf/utility.RemoteTensor.data}} -->
The `data` method retrieves a specific range of byte data from a remote safetensor file and returns it as a bytearray.
- **Inputs**:
    - `self`: An instance of the RemoteTensor class, which contains attributes like dtype, shape, offset_start, size, and url.
- **Control Flow**:
    - The method calls `SafetensorRemote.get_data_by_range` with the URL, offset_start, and size attributes of the RemoteTensor instance to fetch the data.
    - The fetched data is converted into a bytearray to ensure it is writable, which is necessary for compatibility with PyTorch.
    - The method returns the bytearray containing the data.
- **Output**: A bytearray containing the data fetched from the specified range of the remote safetensor file.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_data_by_range`](#SafetensorRemoteget_data_by_range)
- **See also**: [`llama.cpp/gguf-py/gguf/utility.RemoteTensor`](#cpp/gguf-py/gguf/utilityRemoteTensor)  (Base Class)



---
### SafetensorRemote<!-- {{#class:llama.cpp/gguf-py/gguf/utility.SafetensorRemote}} -->
- **Members**:
    - `BASE_DOMAIN`: The base URL for accessing Hugging Face model repositories.
    - `ALIGNMENT`: The byte alignment used for data offsets in safetensor files.
- **Description**: The `SafetensorRemote` class is a utility designed to handle remote safetensor files, specifically for use with Hugging Face model repositories. It provides methods to retrieve lists of tensors from these repositories, either from a single safetensor file or multiple files, and to access the raw byte data of tensors by specifying a range. The class also includes functionality to check the existence of files at given URLs and to manage HTTP request headers, including authorization if a token is available. This class is essential for efficiently managing and accessing large model data stored remotely.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_list_tensors_hf_model`](#SafetensorRemoteget_list_tensors_hf_model)
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_list_tensors`](#SafetensorRemoteget_list_tensors)
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_metadata`](#SafetensorRemoteget_metadata)
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_data_by_range`](#SafetensorRemoteget_data_by_range)
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote.check_file_exist`](#SafetensorRemotecheck_file_exist)
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote._get_request_headers`](#SafetensorRemote_get_request_headers)

**Methods**

---
#### SafetensorRemote\.get\_list\_tensors\_hf\_model<!-- {{#callable:llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_list_tensors_hf_model}} -->
The `get_list_tensors_hf_model` method retrieves a dictionary of tensor names and their metadata from a Hugging Face model repository, handling both single and multiple safetensor files.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `model_id`: A string representing the identifier of the Hugging Face model repository from which to retrieve tensor data.
- **Control Flow**:
    - Check if a single safetensor file exists for the model using the [`check_file_exist`](#SafetensorRemotecheck_file_exist) method.
    - If a single file exists, construct its URL and retrieve the tensor list using [`get_list_tensors`](#SafetensorRemoteget_list_tensors).
    - If multiple files exist, check for an index file and retrieve its data using [`get_data_by_range`](#SafetensorRemoteget_data_by_range).
    - Parse the index file to extract a weight map and determine the list of shard files.
    - Iterate over each shard file, retrieve its tensor list using [`get_list_tensors`](#SafetensorRemoteget_list_tensors), and aggregate the results into a dictionary.
    - If neither a single file nor multiple files are found, raise a `ValueError`.
- **Output**: A dictionary where keys are tensor names and values are `RemoteTensor` objects containing metadata such as dtype, shape, offset_start, size, and the URL of the remote safetensor file.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote.check_file_exist`](#SafetensorRemotecheck_file_exist)
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_list_tensors`](#SafetensorRemoteget_list_tensors)
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_data_by_range`](#SafetensorRemoteget_data_by_range)
- **See also**: [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote`](#cpp/gguf-py/gguf/utilitySafetensorRemote)  (Base Class)


---
#### SafetensorRemote\.get\_list\_tensors<!-- {{#callable:llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_list_tensors}} -->
The `get_list_tensors` method retrieves a dictionary of tensor names and their metadata from a remote safetensor file.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `url`: A string representing the URL of the remote safetensor file.
- **Control Flow**:
    - Call `cls.get_metadata(url)` to retrieve metadata and data start offset from the remote file.
    - Initialize an empty dictionary `res` to store the results.
    - Iterate over each item in the metadata dictionary.
    - Skip the item if the name is '__metadata__'.
    - Check if the metadata is a dictionary; if not, raise a `ValueError`.
    - Extract `dtype`, `shape`, and `data_offsets` from the metadata.
    - Calculate `size` and `offset_start` using `data_offsets` and `data_start_offset`.
    - Create a [`RemoteTensor`](#cpp/gguf-py/gguf/utilityRemoteTensor) object with the extracted metadata and add it to the `res` dictionary.
    - Handle missing keys in metadata by raising a `ValueError`.
    - Return the `res` dictionary containing tensor names and their metadata.
- **Output**: A dictionary where keys are tensor names and values are [`RemoteTensor`](#cpp/gguf-py/gguf/utilityRemoteTensor) objects containing metadata such as dtype, shape, offset_start, size, and url.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_metadata`](#SafetensorRemoteget_metadata)
    - [`llama.cpp/gguf-py/gguf/utility.RemoteTensor`](#cpp/gguf-py/gguf/utilityRemoteTensor)
- **See also**: [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote`](#cpp/gguf-py/gguf/utilitySafetensorRemote)  (Base Class)


---
#### SafetensorRemote\.get\_metadata<!-- {{#callable:llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_metadata}} -->
The `get_metadata` method retrieves and parses JSON metadata from a remote safetensor file, returning the metadata and the data start offset.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `url`: A string representing the URL of the remote safetensor file from which to retrieve metadata.
- **Control Flow**:
    - The method requests the first 5MB of the file from the given URL to ensure it captures enough data for the metadata.
    - It checks if the retrieved data is at least 8 bytes long to read the metadata length; if not, it raises a ValueError.
    - The first 8 bytes of the data are interpreted as a little-endian unsigned 64-bit integer to determine the metadata length.
    - The data start offset is calculated as 8 plus the metadata length, adjusted for alignment based on the class's ALIGNMENT constant.
    - If the retrieved data is insufficient to cover the metadata length, a ValueError is raised.
    - The metadata bytes are extracted from the raw data, decoded as a UTF-8 string, and parsed as JSON.
    - If JSON parsing fails, a ValueError is raised with details of the JSONDecodeError.
- **Output**: A tuple containing the parsed metadata as a dictionary and the data start offset as an integer.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_data_by_range`](#SafetensorRemoteget_data_by_range)
- **See also**: [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote`](#cpp/gguf-py/gguf/utilitySafetensorRemote)  (Base Class)


---
#### SafetensorRemote\.get\_data\_by\_range<!-- {{#callable:llama.cpp/gguf-py/gguf/utility.SafetensorRemote.get_data_by_range}} -->
The `get_data_by_range` method retrieves raw byte data from a specified range of a remote file, using HTTP requests.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `url`: A string representing the URL of the remote file from which data is to be retrieved.
    - `start`: An integer indicating the starting byte position from which to begin reading data.
    - `size`: An optional integer specifying the number of bytes to read; defaults to -1, which means the entire file will be read.
- **Control Flow**:
    - Parse the provided URL using `urlparse` to ensure it is valid.
    - Raise a `ValueError` if the URL is invalid (i.e., missing scheme or netloc).
    - Retrieve request headers using the [`_get_request_headers`](#SafetensorRemote_get_request_headers) method.
    - If `size` is greater than -1, set the 'Range' header to specify the byte range to be retrieved.
    - Make an HTTP GET request to the specified URL with the prepared headers, allowing redirects.
    - Raise an HTTP error if the request fails using `response.raise_for_status()`.
    - Return the content of the response, sliced to the specified size if `size` is greater than -1.
- **Output**: The method returns the raw byte data retrieved from the specified range of the remote file.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote._get_request_headers`](#SafetensorRemote_get_request_headers)
- **See also**: [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote`](#cpp/gguf-py/gguf/utilitySafetensorRemote)  (Base Class)


---
#### SafetensorRemote\.check\_file\_exist<!-- {{#callable:llama.cpp/gguf-py/gguf/utility.SafetensorRemote.check_file_exist}} -->
The `check_file_exist` method checks if a file exists at a specified URL by sending a HEAD request and returns a boolean indicating the file's existence.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `url`: A string representing the URL of the file to check for existence.
- **Control Flow**:
    - Parse the provided URL using `urlparse` to ensure it has a valid scheme and netloc.
    - Raise a `ValueError` if the URL is invalid (missing scheme or netloc).
    - Attempt to send a HEAD request to the URL with headers that include a range request for the first byte.
    - Return `True` if the response status code indicates success (2xx) or redirection (3xx).
    - Return `False` if a `requests.RequestException` is raised during the request.
- **Output**: A boolean value: `True` if the file exists at the given URL, `False` otherwise.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote._get_request_headers`](#SafetensorRemote_get_request_headers)
- **See also**: [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote`](#cpp/gguf-py/gguf/utilitySafetensorRemote)  (Base Class)


---
#### SafetensorRemote\.\_get\_request\_headers<!-- {{#callable:llama.cpp/gguf-py/gguf/utility.SafetensorRemote._get_request_headers}} -->
The `_get_request_headers` method prepares and returns a dictionary of HTTP headers for making requests, including a User-Agent and optionally an Authorization header if an environment token is present.
- **Decorators**: `@classmethod`
- **Inputs**: None
- **Control Flow**:
    - Initialize a dictionary `headers` with a 'User-Agent' key set to 'convert_hf_to_gguf'.
    - Check if the environment variable 'HF_TOKEN' is set.
    - If 'HF_TOKEN' is present, add an 'Authorization' key to the `headers` dictionary with the value 'Bearer ' followed by the token.
    - Return the `headers` dictionary.
- **Output**: A dictionary containing HTTP headers, including 'User-Agent' and optionally 'Authorization' if the 'HF_TOKEN' environment variable is set.
- **See also**: [`llama.cpp/gguf-py/gguf/utility.SafetensorRemote`](#cpp/gguf-py/gguf/utilitySafetensorRemote)  (Base Class)



# Functions

---
### fill\_templated\_filename<!-- {{#callable:llama.cpp/gguf-py/gguf/utility.fill_templated_filename}} -->
The `fill_templated_filename` function replaces placeholders in a filename with formatted output type strings.
- **Inputs**:
    - `filename`: A string representing the filename template containing placeholders for type information.
    - `output_type`: A string or None representing the output type, which will be used to fill in the placeholders in the filename template.
- **Control Flow**:
    - Check if `output_type` is not None and convert it to lowercase and uppercase strings; otherwise, use empty strings.
    - Use the `format` method on the `filename` to replace placeholders with the lowercase and uppercase versions of `output_type`.
- **Output**: A string with the placeholders in the filename template replaced by the formatted output type strings.


---
### model\_weight\_count\_rounded\_notation<!-- {{#callable:llama.cpp/gguf-py/gguf/utility.model_weight_count_rounded_notation}} -->
The function `model_weight_count_rounded_notation` formats a model's parameter count into a human-readable string with a specified minimum number of significant digits.
- **Inputs**:
    - `model_params_count`: An integer representing the total number of parameters in the model.
    - `min_digits`: An optional integer specifying the minimum number of significant digits to display, defaulting to 2.
- **Control Flow**:
    - Check if `model_params_count` is greater than 1 trillion; if so, scale it down to trillions and set the suffix to 'T'.
    - If not, check if `model_params_count` is greater than 1 billion; if so, scale it down to billions and set the suffix to 'B'.
    - If not, check if `model_params_count` is greater than 1 million; if so, scale it down to millions and set the suffix to 'M'.
    - If none of the above, scale `model_params_count` down to thousands and set the suffix to 'K'.
    - Calculate the number of decimal places needed to meet the `min_digits` requirement by subtracting the length of the integer part of the scaled number from `min_digits`.
    - Return the scaled number formatted as a string with the calculated number of decimal places, followed by the appropriate suffix.
- **Output**: A string representing the scaled and formatted parameter count with a suffix indicating the scale (T, B, M, or K).


---
### size\_label<!-- {{#callable:llama.cpp/gguf-py/gguf/utility.size_label}} -->
The `size_label` function generates a human-readable size label for a model based on its parameter counts and expert configuration.
- **Inputs**:
    - `total_params`: The total number of parameters in the model.
    - `shared_params`: The number of shared parameters in the model.
    - `expert_params`: The number of expert parameters in the model.
    - `expert_count`: The number of expert models or components in the model.
- **Control Flow**:
    - Check if `expert_count` is greater than 0.
    - If `expert_count` is greater than 0, calculate the pretty size using the sum of absolute values of `shared_params` and `expert_params`, and format it with the expert count.
    - If `expert_count` is not greater than 0, calculate the pretty size using the absolute value of `total_params`.
    - Return the formatted size label.
- **Output**: A string representing the size label of the model, formatted based on the number of experts and parameter counts.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/utility.model_weight_count_rounded_notation`](#cpp/gguf-py/gguf/utilitymodel_weight_count_rounded_notation)


---
### naming\_convention<!-- {{#callable:llama.cpp/gguf-py/gguf/utility.naming_convention}} -->
The `naming_convention` function generates a standardized model name string based on various input parameters.
- **Inputs**:
    - `model_name`: An optional string representing the model's name.
    - `base_name`: An optional string representing the base name of the model.
    - `finetune_string`: An optional string representing the finetuning information.
    - `version_string`: An optional string representing the version information.
    - `size_label`: An optional string representing the size label of the model.
    - `output_type`: An optional string representing the output type.
    - `model_type`: An optional string that must be either 'vocab' or 'LoRA', representing the model type.
- **Control Flow**:
    - Check if `base_name` is provided; if so, use it as the base for the name after stripping and replacing spaces and slashes with hyphens.
    - If `base_name` is not provided, check if `model_name` is provided; if so, use it similarly as the base for the name.
    - If neither `base_name` nor `model_name` is provided, default the name to 'ggml-model'.
    - Append `size_label` to the name if it is provided, prefixed by a hyphen.
    - Append `finetune_string` to the name if it is provided, prefixed by a hyphen and with spaces replaced by hyphens.
    - Append `version_string` to the name if it is provided, prefixed by a hyphen and with spaces replaced by hyphens.
    - Append `output_type` to the name if it is provided, prefixed by a hyphen, with spaces replaced by hyphens, and converted to uppercase.
    - Append `model_type` to the name if it is provided, prefixed by a hyphen and with spaces replaced by hyphens.
- **Output**: A string that represents the constructed model name following the specified naming convention.


