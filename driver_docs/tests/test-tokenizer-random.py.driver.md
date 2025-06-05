# Purpose
This Python script is designed to test and compare the functionality of two different tokenizers: the `libllama` tokenizer and the `AutoTokenizer` from the Hugging Face Transformers library. The script is structured as a test suite that generates various types of text inputs, including random words, edge cases, and unicode characters, to evaluate the tokenization and detokenization processes of both tokenizers. The primary goal is to ensure that the `libllama` tokenizer, which interfaces with a C library using `cffi`, produces results consistent with the `AutoTokenizer`.

The script defines several classes and functions to facilitate this testing. The `LibLlama` and `LibLlamaModel` classes handle the loading and interaction with the `libllama` C library, providing methods for tokenization and detokenization. The `TokenizerGroundtruth` and `TokenizerLlamaCpp` classes implement the `Tokenizer` interface, representing the two tokenizers being compared. Various generator functions produce different types of text inputs for testing, and the [`compare_tokenizers`](#cpp/tests/test-tokenizer-randomcompare_tokenizers) function performs the actual comparison, logging any discrepancies. The script is intended to be run as a standalone program, with command-line arguments specifying the paths to the vocabulary file and tokenizer directory. The logging setup ensures detailed output for debugging and analysis.
# Imports and Dependencies

---
- `__future__.annotations`
- `time`
- `logging`
- `argparse`
- `subprocess`
- `random`
- `unicodedata`
- `pathlib.Path`
- `typing.Any`
- `typing.Iterator`
- `typing.cast`
- `typing_extensions.Buffer`
- `cffi`
- `transformers.AutoTokenizer`
- `transformers.PreTrainedTokenizer`


# Global Variables

---
### logger
- **Type**: `logging.Logger`
- **Description**: The `logger` variable is an instance of the `Logger` class from the `logging` module, configured to log messages for the 'test-tokenizer-random' context. This logger is used to record debug and error messages throughout the script, particularly during the comparison of tokenizers.
- **Use**: This variable is used to log information and errors during the execution of the script, aiding in debugging and monitoring the tokenizer tests.


# Classes

---
### LibLlama<!-- {{#class:llama.cpp/tests/test-tokenizer-random.LibLlama}} -->
- **Members**:
    - `DEFAULT_PATH_LLAMA_H`: The default file path for the llama.h header file.
    - `DEFAULT_PATH_INCLUDES`: The default list of include directories for the C library.
    - `DEFAULT_PATH_LIBLLAMA`: The default file path for the libllama shared library.
- **Description**: The `LibLlama` class is responsible for interfacing with the C library `libllama` using the CFFI (C Foreign Function Interface) in Python. It initializes the library by loading the necessary header and shared library files, and provides methods to retrieve default parameters for models and contexts. The class is designed to facilitate the integration of the `libllama` library into Python applications, allowing for the configuration and initialization of models and contexts with default or custom parameters.
- **Methods**:
    - [`llama.cpp/tests/test-tokenizer-random.LibLlama.__init__`](#LibLlama__init__)
    - [`llama.cpp/tests/test-tokenizer-random.LibLlama._load_libllama_cffi`](#LibLlama_load_libllama_cffi)
    - [`llama.cpp/tests/test-tokenizer-random.LibLlama.model_default_params`](#LibLlamamodel_default_params)
    - [`llama.cpp/tests/test-tokenizer-random.LibLlama.context_default_params`](#LibLlamacontext_default_params)

**Methods**

---
#### LibLlama\.\_\_init\_\_<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.LibLlama.__init__}} -->
The `__init__` method initializes an instance of the `LibLlama` class by setting up paths for necessary files and loading the `libllama` library using CFFI.
- **Inputs**:
    - `path_llama_h`: A string or None, representing the path to the 'llama.h' header file. Defaults to None, which will use the class's default path.
    - `path_includes`: A list of strings representing include paths for the C preprocessor. Defaults to an empty list, which will use the class's default include paths.
    - `path_libllama`: A string or None, representing the path to the 'libllama' shared library. Defaults to None, which will use the class's default path.
- **Control Flow**:
    - The method checks if `path_llama_h` is provided; if not, it assigns the default path from `DEFAULT_PATH_LLAMA_H`.
    - It checks if `path_includes` is provided; if not, it assigns the default paths from `DEFAULT_PATH_INCLUDES`.
    - It checks if `path_libllama` is provided; if not, it assigns the default path from `DEFAULT_PATH_LIBLLAMA`.
    - The method calls [`_load_libllama_cffi`](#LibLlama_load_libllama_cffi) with the resolved paths to load the library and assigns the returned FFI and library objects to `self.ffi` and `self.lib`.
    - It initializes the llama backend by calling `llama_backend_init` on the loaded library.
- **Output**: The method does not return any value; it initializes the instance's attributes `ffi` and `lib` and sets up the llama backend.
- **Functions called**:
    - [`llama.cpp/tests/test-tokenizer-random.LibLlama._load_libllama_cffi`](#LibLlama_load_libllama_cffi)
- **See also**: [`llama.cpp/tests/test-tokenizer-random.LibLlama`](#cpp/tests/test-tokenizer-randomLibLlama)  (Base Class)


---
#### LibLlama\.\_load\_libllama\_cffi<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.LibLlama._load_libllama_cffi}} -->
The `_load_libllama_cffi` method compiles a C header file using GCC, processes it with `cffi` to define C functions and types, and loads a shared library for use in Python.
- **Inputs**:
    - `path_llama_h`: A string representing the path to the llama.h C header file.
    - `path_includes`: A list of strings representing the include paths for the C preprocessor.
    - `path_libllama`: A string representing the path to the libllama shared library file.
- **Control Flow**:
    - Constructs a GCC command to preprocess the C header file with specified include paths and macros.
    - Executes the GCC command using `subprocess.run` and captures the output.
    - Asserts that the GCC command executed successfully by checking the return code.
    - Decodes the preprocessed C source code from the GCC output.
    - Initializes a `cffi.FFI` object to handle C definitions in Python.
    - Applies workarounds for `pycparser` by modifying the source code to handle specific C constructs.
    - Defines the C functions and types in the `ffi` object using the preprocessed source code.
    - Loads the shared library specified by `path_libllama` using `ffi.dlopen`.
    - Returns a tuple containing the `ffi` object and the loaded library.
- **Output**: A tuple containing a `cffi.FFI` object and the loaded shared library.
- **See also**: [`llama.cpp/tests/test-tokenizer-random.LibLlama`](#cpp/tests/test-tokenizer-randomLibLlama)  (Base Class)


---
#### LibLlama\.model\_default\_params<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.LibLlama.model_default_params}} -->
The `model_default_params` method retrieves default model parameters and updates them with any provided keyword arguments.
- **Inputs**:
    - `kwargs`: A dictionary of keyword arguments representing parameter names and their new values to update the default model parameters.
- **Control Flow**:
    - Call `self.lib.llama_model_default_params()` to retrieve the default model parameters.
    - Iterate over each key-value pair in `kwargs`.
    - For each key-value pair, use `setattr` to update the corresponding attribute in the default model parameters object.
    - Return the updated model parameters object.
- **Output**: The method returns an object representing the model parameters, updated with any provided keyword arguments.
- **See also**: [`llama.cpp/tests/test-tokenizer-random.LibLlama`](#cpp/tests/test-tokenizer-randomLibLlama)  (Base Class)


---
#### LibLlama\.context\_default\_params<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.LibLlama.context_default_params}} -->
The `context_default_params` method initializes and returns a default set of context parameters for the llama library, allowing for optional customization through keyword arguments.
- **Inputs**:
    - `kwargs`: A dictionary of keyword arguments representing context parameters to be customized.
- **Control Flow**:
    - Call the `llama_context_default_params` method from the `lib` attribute to get the default context parameters.
    - Iterate over the key-value pairs in `kwargs`.
    - For each key-value pair, set the attribute on the `cparams` object using `setattr`.
    - Return the modified `cparams` object.
- **Output**: The method returns an object representing the context parameters, potentially modified by the provided keyword arguments.
- **See also**: [`llama.cpp/tests/test-tokenizer-random.LibLlama`](#cpp/tests/test-tokenizer-randomLibLlama)  (Base Class)



---
### LibLlamaModel<!-- {{#class:llama.cpp/tests/test-tokenizer-random.LibLlamaModel}} -->
- **Members**:
    - `lib`: Holds the library interface from the LibLlama instance.
    - `ffi`: Stores the Foreign Function Interface from the LibLlama instance.
    - `model`: Represents the loaded model from the specified file.
    - `ctx`: Holds the context created for the model.
    - `token_ids`: An array to store token IDs with a maximum size determined by the context.
    - `text_buff`: A buffer for storing text data with a fixed size of 1024 bytes.
- **Description**: The LibLlamaModel class is responsible for managing the loading and interaction with a Llama model using the LibLlama library. It initializes the model and context using specified parameters, and provides methods for tokenizing and detokenizing text. The class ensures that the model and context are properly loaded and provides mechanisms to free resources when they are no longer needed.
- **Methods**:
    - [`llama.cpp/tests/test-tokenizer-random.LibLlamaModel.__init__`](#LibLlamaModel__init__)
    - [`llama.cpp/tests/test-tokenizer-random.LibLlamaModel.free`](#LibLlamaModelfree)
    - [`llama.cpp/tests/test-tokenizer-random.LibLlamaModel.tokenize`](#LibLlamaModeltokenize)
    - [`llama.cpp/tests/test-tokenizer-random.LibLlamaModel.detokenize`](#LibLlamaModeldetokenize)

**Methods**

---
#### LibLlamaModel\.\_\_init\_\_<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.LibLlamaModel.__init__}} -->
The `__init__` method initializes a `LibLlamaModel` instance by loading a model from a file and creating a context for it using specified parameters.
- **Inputs**:
    - `libllama`: An instance of the `LibLlama` class, providing access to the library and FFI interfaces.
    - `path_model`: A string representing the file path to the model to be loaded.
    - `mparams`: An optional dictionary of model parameters to customize the model loading process.
    - `cparams`: An optional dictionary of context parameters to customize the context creation process.
- **Control Flow**:
    - Assigns the library and FFI interfaces from the `libllama` instance to the instance variables `self.lib` and `self.ffi`.
    - Checks if `mparams` is a dictionary and converts it to default model parameters using `libllama.model_default_params`.
    - Loads the model from the specified file path using `self.lib.llama_model_load_from_file` with the encoded path and model parameters.
    - Raises a `RuntimeError` if the model fails to load.
    - Checks if `cparams` is a dictionary and converts it to default context parameters using `libllama.context_default_params`.
    - Creates a new context for the model using `self.lib.llama_new_context_with_model` with the model and context parameters.
    - Raises a `RuntimeError` if the context fails to create.
    - Determines the maximum number of tokens using `self.lib.llama_n_ctx` and initializes `self.token_ids` with this size.
    - Initializes `self.text_buff` with a buffer size of 1024 bytes.
- **Output**: The method does not return any value; it initializes the instance variables for the `LibLlamaModel` object.
- **Functions called**:
    - [`llama.cpp/tests/test-tokenizer-random.LibLlama.model_default_params`](#LibLlamamodel_default_params)
    - [`llama.cpp/tests/test-tokenizer-random.LibLlama.context_default_params`](#LibLlamacontext_default_params)
- **See also**: [`llama.cpp/tests/test-tokenizer-random.LibLlamaModel`](#cpp/tests/test-tokenizer-randomLibLlamaModel)  (Base Class)


---
#### LibLlamaModel\.free<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.LibLlamaModel.free}} -->
The `free` method releases resources associated with the model and context by freeing memory and setting related attributes to None.
- **Inputs**: None
- **Control Flow**:
    - Check if `self.ctx` is not None, and if so, call `self.lib.llama_free(self.ctx)` to free the context.
    - Check if `self.model` is not None, and if so, call `self.lib.llama_model_free(self.model)` to free the model.
    - Set `self.ctx`, `self.model`, and `self.lib` to None to indicate that the resources have been released.
- **Output**: The method does not return any value; it performs cleanup operations on the instance's attributes.
- **See also**: [`llama.cpp/tests/test-tokenizer-random.LibLlamaModel`](#cpp/tests/test-tokenizer-randomLibLlamaModel)  (Base Class)


---
#### LibLlamaModel\.tokenize<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.LibLlamaModel.tokenize}} -->
The `tokenize` method converts a given text into a list of integer token IDs using the llama model.
- **Inputs**:
    - `text`: A string representing the text to be tokenized.
    - `add_special`: A boolean indicating whether to add special tokens during tokenization, default is False.
    - `parse_special`: A boolean indicating whether to parse special tokens during tokenization, default is False.
- **Control Flow**:
    - The input text is encoded into UTF-8 bytes.
    - The `llama_tokenize` function from the llama library is called with the model, encoded text, and other parameters to tokenize the text.
    - If the number of tokens returned is negative and the token buffer is not too large, the token buffer is resized and `llama_tokenize` is called again.
    - The method returns a list of token IDs from the token buffer.
- **Output**: A list of integers representing the token IDs of the input text.
- **See also**: [`llama.cpp/tests/test-tokenizer-random.LibLlamaModel`](#cpp/tests/test-tokenizer-randomLibLlamaModel)  (Base Class)


---
#### LibLlamaModel\.detokenize<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.LibLlamaModel.detokenize}} -->
The `detokenize` method converts a list of token IDs back into a string, handling special tokens based on the provided flags.
- **Inputs**:
    - `ids`: A list of integers representing token IDs to be converted back into a string.
    - `remove_special`: A boolean flag indicating whether to remove special tokens during detokenization.
    - `unparse_special`: A boolean flag indicating whether to unparse special tokens during detokenization.
- **Control Flow**:
    - Check if the current token buffer size is less than the number of IDs; if so, resize the buffer to accommodate the IDs.
    - Iterate over the list of IDs and assign each ID to the token buffer.
    - Call the `llama_detokenize` function from the library to convert the token IDs into a string, storing the result in a text buffer.
    - If the detokenization result is negative and the text buffer is not too large, resize the text buffer and retry detokenization.
    - Convert the text buffer to a UTF-8 string, replacing any encoding errors with the Unicode replacement character.
- **Output**: A UTF-8 encoded string representing the detokenized text, with errors replaced by the Unicode replacement character.
- **See also**: [`llama.cpp/tests/test-tokenizer-random.LibLlamaModel`](#cpp/tests/test-tokenizer-randomLibLlamaModel)  (Base Class)



---
### Tokenizer<!-- {{#class:llama.cpp/tests/test-tokenizer-random.Tokenizer}} -->
- **Description**: The `Tokenizer` class is an abstract base class that defines the interface for tokenization operations, specifically encoding text into a list of integers and decoding a list of integers back into text. It serves as a blueprint for more specific tokenizer implementations, such as `TokenizerGroundtruth` and `TokenizerLlamaCpp`, which provide concrete implementations of these methods.
- **Methods**:
    - [`llama.cpp/tests/test-tokenizer-random.Tokenizer.encode`](#Tokenizerencode)
    - [`llama.cpp/tests/test-tokenizer-random.Tokenizer.decode`](#Tokenizerdecode)

**Methods**

---
#### Tokenizer\.encode<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.Tokenizer.encode}} -->
The `encode` method in the `Tokenizer` class is an abstract method intended to convert a given text string into a list of integer token IDs.
- **Inputs**:
    - `text`: A string representing the text to be encoded into token IDs.
- **Control Flow**:
    - The method is defined to raise a `NotImplementedError`, indicating it is meant to be overridden in subclasses.
- **Output**: The method is expected to return a list of integers, each representing a token ID corresponding to the input text.
- **See also**: [`llama.cpp/tests/test-tokenizer-random.Tokenizer`](#cpp/tests/test-tokenizer-randomTokenizer)  (Base Class)


---
#### Tokenizer\.decode<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.Tokenizer.decode}} -->
The `decode` method in the `Tokenizer` class is intended to convert a list of integer token IDs back into a string, but it is not implemented in the base class.
- **Inputs**:
    - `ids`: A list of integers representing token IDs that need to be converted back into a string.
- **Control Flow**:
    - The method is defined but not implemented, as it raises a `NotImplementedError`, indicating that subclasses should provide their own implementation.
- **Output**: The method is expected to return a string that represents the decoded text from the provided list of token IDs.
- **See also**: [`llama.cpp/tests/test-tokenizer-random.Tokenizer`](#cpp/tests/test-tokenizer-randomTokenizer)  (Base Class)



---
### TokenizerGroundtruth<!-- {{#class:llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth}} -->
- **Members**:
    - `model`: An instance of PreTrainedTokenizer initialized from a specified directory.
    - `add_bos_token`: A boolean indicating if a beginning-of-sequence token should be added.
    - `add_eos_token`: A boolean indicating if an end-of-sequence token should be added.
    - `vocab`: A sorted list of vocabulary tokens decoded from the model.
    - `special_tokens`: A list of special tokens used by the tokenizer.
    - `added_tokens`: A list of additional tokens encoded by the model.
    - `bos_token`: The beginning-of-sequence token used by the model.
    - `eos_token`: The end-of-sequence token used by the model.
- **Description**: The TokenizerGroundtruth class extends the Tokenizer class and is designed to handle tokenization using a pre-trained tokenizer model. It initializes a tokenizer model from a specified directory and determines whether to add beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens based on the encoding of a sample text. The class also constructs a vocabulary from the model's tokens, identifies special and added tokens, and stores the BOS and EOS tokens. It provides methods to encode text into token IDs and decode token IDs back into text, ensuring that special tokens are appropriately handled.
- **Methods**:
    - [`llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth.__init__`](#TokenizerGroundtruth__init__)
    - [`llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth.encode`](#TokenizerGroundtruthencode)
    - [`llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth.decode`](#TokenizerGroundtruthdecode)
- **Inherits From**:
    - [`llama.cpp/tests/test-tokenizer-random.Tokenizer`](#cpp/tests/test-tokenizer-randomTokenizer)

**Methods**

---
#### TokenizerGroundtruth\.\_\_init\_\_<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth.__init__}} -->
The `__init__` method initializes a `TokenizerGroundtruth` object by loading a pre-trained tokenizer model and setting up various token-related attributes.
- **Inputs**:
    - `dir_tokenizer`: A string representing the directory path to the pre-trained tokenizer model.
- **Control Flow**:
    - The method starts by loading a pre-trained tokenizer model using the `AutoTokenizer.from_pretrained` method with the provided `dir_tokenizer` path.
    - It encodes a sample string 'a' to determine the presence of beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens, asserting that the length of the encoded IDs is between 1 and 3.
    - Based on the encoded IDs, it sets the `add_bos_token` and `add_eos_token` attributes by checking if the first and last IDs match the model's BOS and EOS token IDs, respectively.
    - The method retrieves the vocabulary from the tokenizer model, decodes it into a list of tokens, and sorts it to set the `vocab` attribute.
    - It initializes the `special_tokens` attribute with a list of all special tokens from the model.
    - It decodes any added tokens from the model and assigns them to the `added_tokens` attribute.
    - Finally, it sets the `bos_token` and `eos_token` attributes to the model's BOS and EOS tokens.
- **Output**: The method does not return any value; it initializes the instance attributes of the `TokenizerGroundtruth` class.
- **Functions called**:
    - [`llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth.encode`](#TokenizerGroundtruthencode)
- **See also**: [`llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth`](#cpp/tests/test-tokenizer-randomTokenizerGroundtruth)  (Base Class)


---
#### TokenizerGroundtruth\.encode<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth.encode}} -->
The [`encode`](#Tokenizerencode) method encodes a given text string into a list of integer token IDs using a pre-trained tokenizer model.
- **Inputs**:
    - `text`: A string representing the text to be encoded into token IDs.
- **Control Flow**:
    - The method calls the [`encode`](#Tokenizerencode) function of the `self.model`, which is an instance of `PreTrainedTokenizer`, passing the input `text` and setting `add_special_tokens` to `True`.
- **Output**: A list of integers representing the token IDs of the encoded text.
- **Functions called**:
    - [`llama.cpp/tests/test-tokenizer-random.Tokenizer.encode`](#Tokenizerencode)
- **See also**: [`llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth`](#cpp/tests/test-tokenizer-randomTokenizerGroundtruth)  (Base Class)


---
#### TokenizerGroundtruth\.decode<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth.decode}} -->
The [`decode`](#Tokenizerdecode) method converts a list of token IDs into a string using the model's decoding function, including special tokens.
- **Inputs**:
    - `ids`: A list of integers representing token IDs to be decoded into a string.
- **Control Flow**:
    - The method calls the [`decode`](#Tokenizerdecode) function of the `model` attribute, passing the `ids` list and setting `skip_special_tokens` to `False`.
- **Output**: A string that represents the decoded text from the provided token IDs, including any special tokens.
- **Functions called**:
    - [`llama.cpp/tests/test-tokenizer-random.Tokenizer.decode`](#Tokenizerdecode)
- **See also**: [`llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth`](#cpp/tests/test-tokenizer-randomTokenizerGroundtruth)  (Base Class)



---
### TokenizerLlamaCpp<!-- {{#class:llama.cpp/tests/test-tokenizer-random.TokenizerLlamaCpp}} -->
- **Members**:
    - `libllama`: An instance of LibLlama or None, used to interface with the Llama library.
- **Description**: The TokenizerLlamaCpp class is a specialized tokenizer that extends the Tokenizer class, utilizing the LibLlama library to perform tokenization and detokenization of text. It initializes a LibLlamaModel with specific parameters for vocabulary and context, allowing it to encode text into token IDs and decode token IDs back into text. This class is designed to work with a specific vocabulary file and provides methods to handle special tokens during the encoding and decoding processes.
- **Methods**:
    - [`llama.cpp/tests/test-tokenizer-random.TokenizerLlamaCpp.__init__`](#TokenizerLlamaCpp__init__)
    - [`llama.cpp/tests/test-tokenizer-random.TokenizerLlamaCpp.encode`](#TokenizerLlamaCppencode)
    - [`llama.cpp/tests/test-tokenizer-random.TokenizerLlamaCpp.decode`](#TokenizerLlamaCppdecode)
- **Inherits From**:
    - [`llama.cpp/tests/test-tokenizer-random.Tokenizer`](#cpp/tests/test-tokenizer-randomTokenizer)

**Methods**

---
#### TokenizerLlamaCpp\.\_\_init\_\_<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.TokenizerLlamaCpp.__init__}} -->
The `__init__` method initializes an instance of the `TokenizerLlamaCpp` class by setting up a [`LibLlama`](#cpp/tests/test-tokenizer-randomLibLlama) instance and a [`LibLlamaModel`](#cpp/tests/test-tokenizer-randomLibLlamaModel) with specified parameters.
- **Inputs**:
    - `vocab_file`: A string representing the path to the vocabulary file used to initialize the [`LibLlamaModel`](#cpp/tests/test-tokenizer-randomLibLlamaModel).
- **Control Flow**:
    - Checks if `libllama` is `None` and initializes it with a new [`LibLlama`](#cpp/tests/test-tokenizer-randomLibLlama) instance if necessary.
    - Creates a [`LibLlamaModel`](#cpp/tests/test-tokenizer-randomLibLlamaModel) instance using the `libllama` instance, the provided `vocab_file`, and specific model and context parameters.
- **Output**: The method does not return any value; it initializes the instance attributes `libllama` and `model`.
- **Functions called**:
    - [`llama.cpp/tests/test-tokenizer-random.LibLlama`](#cpp/tests/test-tokenizer-randomLibLlama)
    - [`llama.cpp/tests/test-tokenizer-random.LibLlamaModel`](#cpp/tests/test-tokenizer-randomLibLlamaModel)
- **See also**: [`llama.cpp/tests/test-tokenizer-random.TokenizerLlamaCpp`](#cpp/tests/test-tokenizer-randomTokenizerLlamaCpp)  (Base Class)


---
#### TokenizerLlamaCpp\.encode<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.TokenizerLlamaCpp.encode}} -->
The `encode` method tokenizes a given text string into a list of integer token IDs using the `LibLlamaModel`.
- **Inputs**:
    - `text`: A string representing the text to be tokenized.
- **Control Flow**:
    - The method calls the [`tokenize`](#LibLlamaModeltokenize) function of the `LibLlamaModel` instance associated with the `TokenizerLlamaCpp` object.
    - The [`tokenize`](#LibLlamaModeltokenize) function is invoked with the `text` argument and additional parameters `add_special=True` and `parse_special=True`.
    - The [`tokenize`](#LibLlamaModeltokenize) function processes the text and returns a list of integer token IDs.
- **Output**: A list of integers representing the token IDs of the input text.
- **Functions called**:
    - [`llama.cpp/tests/test-tokenizer-random.LibLlamaModel.tokenize`](#LibLlamaModeltokenize)
- **See also**: [`llama.cpp/tests/test-tokenizer-random.TokenizerLlamaCpp`](#cpp/tests/test-tokenizer-randomTokenizerLlamaCpp)  (Base Class)


---
#### TokenizerLlamaCpp\.decode<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.TokenizerLlamaCpp.decode}} -->
The `decode` method converts a list of token IDs back into a string using the [`detokenize`](#LibLlamaModeldetokenize) method of the `LibLlamaModel`.
- **Inputs**:
    - `ids`: A list of integers representing token IDs to be converted back into a string.
- **Control Flow**:
    - The method calls `self.model.detokenize` with the provided `ids` list.
    - It sets `remove_special` to `False` and `unparse_special` to `True` when calling [`detokenize`](#LibLlamaModeldetokenize).
- **Output**: A string that represents the decoded text from the given list of token IDs.
- **Functions called**:
    - [`llama.cpp/tests/test-tokenizer-random.LibLlamaModel.detokenize`](#LibLlamaModeldetokenize)
- **See also**: [`llama.cpp/tests/test-tokenizer-random.TokenizerLlamaCpp`](#cpp/tests/test-tokenizer-randomTokenizerLlamaCpp)  (Base Class)



# Functions

---
### generator\_custom\_text<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.generator_custom_text}} -->
The `generator_custom_text` function yields a series of predefined text strings for testing purposes.
- **Inputs**: None
- **Control Flow**:
    - The function uses the `yield from` statement to yield each string from a predefined list.
    - The list contains various strings including empty strings, whitespace characters, special characters, and text in different languages.
    - The function does not take any input parameters and directly yields the strings one by one.
- **Output**: An iterator that yields strings from a predefined list.


---
### generator\_custom\_text\_edge\_cases<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.generator_custom_text_edge_cases}} -->
The function `generator_custom_text_edge_cases` generates an iterator of strings representing edge cases found during debugging.
- **Inputs**: None
- **Control Flow**:
    - The function uses a `yield from` statement to yield a list of strings.
    - Each string in the list represents a specific edge case, often involving special characters or sequences that may cause issues in text processing.
    - The strings include various Unicode characters, control characters, and sequences that have been identified as problematic in different contexts.
- **Output**: An iterator of strings, each representing a specific edge case for text processing.


---
### generator\_vocab\_words<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.generator_vocab_words}} -->
The function `generator_vocab_words` generates an iterator over all vocabulary words from a given `TokenizerGroundtruth` instance.
- **Inputs**:
    - `tokenizer`: An instance of `TokenizerGroundtruth` from which the vocabulary words are retrieved.
- **Control Flow**:
    - The function uses the `yield from` statement to iterate over the `vocab` attribute of the `tokenizer` object, which is a list of vocabulary words.
- **Output**: An iterator that yields each word in the vocabulary of the provided `TokenizerGroundtruth` instance.


---
### generator\_ascii\_lr\_strip<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.generator_ascii_lr_strip}} -->
The function `generator_ascii_lr_strip` generates all possible combinations of two ASCII characters with optional leading and trailing whitespace.
- **Inputs**: None
- **Control Flow**:
    - Initialize a list `WHITESPACES` containing empty strings and strings with one or two spaces.
    - Create a list `CHARACTERS` containing all ASCII characters from code point 1 to 127, plus an empty string.
    - Iterate over each character `char1` in `CHARACTERS`.
    - For each `char1`, iterate over each character `char2` in `CHARACTERS`.
    - For each pair of characters `char1` and `char2`, iterate over each string `lstrip` in `WHITESPACES`.
    - For each `lstrip`, iterate over each string `rstrip` in `WHITESPACES`.
    - Yield three different combinations of `char1`, `char2`, `lstrip`, and `rstrip` with different arrangements of whitespace.
- **Output**: An iterator that yields strings composed of two ASCII characters with various combinations of leading and trailing whitespace.


---
### generator\_apostrophe<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.generator_apostrophe}} -->
The `generator_apostrophe` function generates and yields strings with various combinations of ASCII characters and apostrophes, including optional leading and trailing whitespace.
- **Inputs**: None
- **Control Flow**:
    - Initialize a list `WHITESPACES` containing empty string, single space, and double space.
    - Initialize a list `CHARACTERS` containing ASCII characters from 1 to 127 and an empty string.
    - Iterate over each character `char1` in `CHARACTERS`.
    - For each `char1`, iterate over each character `char2` in `CHARACTERS`.
    - For each combination of `char1` and `char2`, iterate over each whitespace `lstrip` in `WHITESPACES`.
    - For each `lstrip`, iterate over each whitespace `rstrip` in `WHITESPACES`.
    - Yield the string formed by concatenating `char1`, `lstrip`, an apostrophe, `rstrip`, and `char2`.
    - Yield the string formed by concatenating `char1`, `char2`, `lstrip`, an apostrophe, `rstrip`, and 'z'.
    - Yield the string formed by concatenating 'a', `lstrip`, an apostrophe, `rstrip`, `char1`, and `char2`.
- **Output**: An iterator that yields strings composed of ASCII characters, apostrophes, and optional whitespace.


---
### generator\_added\_lr\_strip<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.generator_added_lr_strip}} -->
The function `generator_added_lr_strip` generates strings by combining special and added tokens from a tokenizer with various whitespace patterns on both sides.
- **Inputs**:
    - `tokenizer`: An instance of `TokenizerGroundtruth` that provides special and added tokens for string generation.
- **Control Flow**:
    - Define a list of whitespace patterns called `WHITESPACES`.
    - Combine the special and added tokens from the tokenizer into a sorted list called `all_tokens`.
    - Iterate over each token in `all_tokens`.
    - For each token, iterate over each left and right whitespace pattern from `WHITESPACES`.
    - Yield strings formed by concatenating the left whitespace, token, and right whitespace in various combinations, including with 'a' and 'z' added to the start and end.
- **Output**: An iterator that yields strings with special and added tokens surrounded by various whitespace patterns.


---
### generator\_random\_added\_tokens<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.generator_random_added_tokens}} -->
The function `generator_random_added_tokens` generates random sequences of tokens from a tokenizer's special and added tokens, ensuring no duplicate BOS or EOS tokens at the start or end of the sequence.
- **Inputs**:
    - `tokenizer`: An instance of `TokenizerGroundtruth` which provides the special and added tokens to be used in generating random sequences.
    - `iterations`: An integer specifying the number of iterations for generating random token sequences, defaulting to 100.
- **Control Flow**:
    - Initialize a list of separation tokens and combine them with the tokenizer's special and added tokens to form a list of all possible tokens.
    - Create a random number generator instance.
    - Iterate over the specified number of iterations.
    - For each iteration, seed the random generator with the current iteration index and select 500 random tokens from the list of all tokens.
    - Check if the first token is the BOS token and remove duplicate BOS tokens if present, leaving only one at the start.
    - Check if the last token is the EOS token and remove duplicate EOS tokens if present, leaving only one at the end.
    - Yield the concatenated string of tokens for each iteration.
- **Output**: An iterator that yields strings, each representing a sequence of randomly generated tokens.


---
### generator\_random\_chars<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.generator_random_chars}} -->
The function `generator_random_chars` generates random text composed of simple characters and whitespace over a specified number of iterations.
- **Inputs**:
    - `iterations`: An integer specifying the number of iterations for generating random text, defaulting to 100.
- **Control Flow**:
    - Initialize constants for the number of words per iteration, whitespace characters, and a set of characters to use in text generation.
    - Create a random number generator instance.
    - Iterate over the range of specified iterations, seeding the random generator with the current iteration index.
    - For each iteration, generate a list of random words by selecting a random number of characters from the character set and appending a random whitespace character.
    - Join the generated words into a single string and yield it as the output for the current iteration.
- **Output**: An iterator that yields strings of randomly generated text for each iteration.


---
### generator\_unicodes<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.generator_unicodes}} -->
The `generator_unicodes` function generates an iterator over valid Unicode characters up to a specified code point limit.
- **Decorators**: `@Iterator`
- **Inputs**: None
- **Control Flow**:
    - Define a constant `MAX_CODEPOINTS` set to 0x30000, which is the upper limit for Unicode code points to be considered.
    - Define a nested function `_valid(cpt)` that checks if a given code point `cpt` is valid by ensuring it is less than `MAX_CODEPOINTS` and not in the categories 'Cn', 'Cs', or 'Co'.
    - Create a list `characters` by iterating over all code points from 0 to `MAX_CODEPOINTS`, converting each valid code point to its corresponding character using `chr(cpt)`.
    - Use `yield from characters` to yield each character in the `characters` list.
- **Output**: An iterator that yields valid Unicode characters as strings.


---
### generator\_random\_unicodes<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.generator_random_unicodes}} -->
The function `generator_random_unicodes` generates random text composed of unicode characters and whitespace, iterating a specified number of times.
- **Inputs**:
    - `iterations`: An integer specifying the number of iterations for generating random unicode text, defaulting to 100.
- **Control Flow**:
    - Initialize a constant `NUM_WORDS` to 200, representing the number of words to generate per iteration.
    - Define a list `WHITESPACES` containing various whitespace characters to be used in the text generation.
    - Generate a list `characters` by calling `generator_unicodes()`, which provides a sequence of unicode characters.
    - Create a `Random` object `rand` for generating random numbers.
    - Iterate `m` from 0 to `iterations - 1`, seeding the random number generator with `m` for reproducibility.
    - For each iteration, initialize an empty list `text` to store the generated words.
    - Generate `NUM_WORDS` words by randomly selecting between 1 and 7 unicode characters from `characters`, appending a random whitespace from `WHITESPACES`, and joining them into a string.
    - Append each generated word to the `text` list.
    - Yield the concatenated string of all words in `text` as the output for the current iteration.
- **Output**: An iterator that yields strings of randomly generated unicode text with whitespace.
- **Functions called**:
    - [`llama.cpp/tests/test-tokenizer-random.generator_unicodes`](#cpp/tests/test-tokenizer-randomgenerator_unicodes)


---
### generator\_random\_vocab\_chars<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.generator_random_vocab_chars}} -->
The function generates random text using characters from a tokenizer's vocabulary over a specified number of iterations.
- **Inputs**:
    - `tokenizer`: An instance of TokenizerGroundtruth, which provides access to the vocabulary from which characters are drawn.
    - `iterations`: An optional integer specifying the number of iterations for generating random text, defaulting to 100.
- **Control Flow**:
    - Initialize an empty set to collect unique characters from the tokenizer's vocabulary.
    - Iterate over each word in the tokenizer's vocabulary, updating the set with characters from each word.
    - Convert the set of characters into a sorted list.
    - Create a random number generator instance.
    - For each iteration up to the specified number, seed the random generator with the current iteration index.
    - Generate a random sequence of 1024 characters from the sorted list of vocabulary characters.
    - Yield the generated sequence as a string.
- **Output**: An iterator that yields strings of random text composed of characters from the tokenizer's vocabulary.


---
### generator\_random\_vocab\_words<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.generator_random_vocab_words}} -->
The function `generator_random_vocab_words` generates random text using words from a tokenizer's vocabulary.
- **Inputs**:
    - `tokenizer`: An instance of `TokenizerGroundtruth` which provides the vocabulary words to be used for generating random text.
    - `iterations`: An optional integer specifying the number of iterations for generating random text, defaulting to 100.
- **Control Flow**:
    - The function first creates a list of vocabulary words by stripping whitespace from each word in the tokenizer's vocabulary.
    - It yields each word from the vocabulary list one by one.
    - A random number generator is initialized.
    - For each iteration (up to the specified number of iterations), the random generator is seeded with the current iteration number.
    - A random number of words (between 300 and 400) is chosen to form a text block.
    - For each word in the text block, a random number of words (between 1 and 3) is selected from the vocabulary.
    - A random separator (space, newline, carriage return, or tab) is chosen and appended to the word.
    - The constructed text block is yielded as a single string.
- **Output**: An iterator that yields strings of random text generated from the vocabulary words.


---
### compare\_tokenizers<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.compare_tokenizers}} -->
The `compare_tokenizers` function compares the encoding and decoding performance and accuracy of two different tokenizers using a text generator.
- **Inputs**:
    - `tokenizer1`: An instance of `TokenizerGroundtruth`, which is used as the reference tokenizer for comparison.
    - `tokenizer2`: An instance of `TokenizerLlamaCpp`, which is the tokenizer being compared against the reference.
    - `generator`: An iterator that yields strings of text to be tokenized and detokenized by both tokenizers.
- **Control Flow**:
    - Initialize timing and error counters for encoding and decoding processes.
    - Log the start of the comparison process with the generator's qualified name.
    - Iterate over each text string provided by the generator.
    - For each text, encode it using both tokenizers and measure the time taken for each encoding process.
    - Decode the encoded text using both tokenizers and measure the time taken for each decoding process.
    - Accumulate the time taken for encoding and decoding for both tokenizers.
    - If the number of encoding errors is below the maximum allowed and the encoded outputs differ, find the first mismatch and log the expected and actual results.
    - If the number of decoding errors is below the maximum allowed and the decoded outputs do not match the original text, find the first mismatch and log the expected and actual results.
    - If the number of encoding and decoding errors reaches the maximum allowed, log an exit message and break the loop.
    - Log the total time taken for the comparison process.
- **Output**: The function does not return any value; it logs the results of the comparison, including timing and error information, to a logger.
- **Functions called**:
    - [`llama.cpp/tests/test-tokenizer-random.Tokenizer.encode`](#Tokenizerencode)
    - [`llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth.decode`](#TokenizerGroundtruthdecode)


---
### main<!-- {{#callable:llama.cpp/tests/test-tokenizer-random.main}} -->
The `main` function initializes and compares two tokenizers using various text generators to test their encoding and decoding capabilities.
- **Inputs**:
    - `argv`: An optional list of strings representing command-line arguments; if not provided, defaults to None.
- **Control Flow**:
    - An argument parser is created to handle command-line inputs for 'vocab_file', 'dir_tokenizer', and an optional 'verbose' flag.
    - Logging is configured based on the verbosity flag provided in the arguments.
    - Two tokenizer objects are instantiated: [`TokenizerGroundtruth`](#cpp/tests/test-tokenizer-randomTokenizerGroundtruth) using the directory of the tokenizer model and [`TokenizerLlamaCpp`](#cpp/tests/test-tokenizer-randomTokenizerLlamaCpp) using the vocabulary file.
    - Several calls to [`compare_tokenizers`](#cpp/tests/test-tokenizer-randomcompare_tokenizers) are made with different text generators to compare the tokenization results of the two tokenizers.
    - The `model.free()` method is called on the [`TokenizerLlamaCpp`](#cpp/tests/test-tokenizer-randomTokenizerLlamaCpp) instance to release resources.
- **Output**: The function does not return any value; it performs logging and comparison operations as side effects.
- **Functions called**:
    - [`llama.cpp/tests/test-tokenizer-random.TokenizerGroundtruth`](#cpp/tests/test-tokenizer-randomTokenizerGroundtruth)
    - [`llama.cpp/tests/test-tokenizer-random.TokenizerLlamaCpp`](#cpp/tests/test-tokenizer-randomTokenizerLlamaCpp)
    - [`llama.cpp/tests/test-tokenizer-random.compare_tokenizers`](#cpp/tests/test-tokenizer-randomcompare_tokenizers)
    - [`llama.cpp/tests/test-tokenizer-random.generator_ascii_lr_strip`](#cpp/tests/test-tokenizer-randomgenerator_ascii_lr_strip)
    - [`llama.cpp/tests/test-tokenizer-random.generator_apostrophe`](#cpp/tests/test-tokenizer-randomgenerator_apostrophe)
    - [`llama.cpp/tests/test-tokenizer-random.generator_unicodes`](#cpp/tests/test-tokenizer-randomgenerator_unicodes)
    - [`llama.cpp/tests/test-tokenizer-random.generator_vocab_words`](#cpp/tests/test-tokenizer-randomgenerator_vocab_words)
    - [`llama.cpp/tests/test-tokenizer-random.generator_added_lr_strip`](#cpp/tests/test-tokenizer-randomgenerator_added_lr_strip)
    - [`llama.cpp/tests/test-tokenizer-random.LibLlamaModel.free`](#LibLlamaModelfree)


