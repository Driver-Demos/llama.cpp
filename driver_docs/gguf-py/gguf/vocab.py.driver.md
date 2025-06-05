# Purpose
This Python code defines a set of classes and functions for handling different types of vocabularies used in natural language processing models, specifically focusing on tokenization. The primary classes include `SpecialVocab`, `NoVocab`, `BpeVocab`, `SentencePieceVocab`, and `LlamaHfVocab`, each representing a different approach to managing vocabularies and tokenization. The `SpecialVocab` class is responsible for loading and managing special tokens and merges from various configuration files, while the `BpeVocab`, `SentencePieceVocab`, and `LlamaHfVocab` classes handle specific tokenization models like Byte Pair Encoding (BPE) and SentencePiece, with `LlamaHfVocab` specifically designed for handling Hugging Face's Llama tokenizer.

The code is structured as a library intended to be imported and used in other Python scripts or applications. It provides a comprehensive API for loading, managing, and interacting with different vocabulary types, including methods for loading vocabularies from files, adding special tokens, and iterating over all tokens. The use of protocols and runtime checks ensures that the classes conform to expected interfaces, allowing for flexible and extensible vocabulary management. The code also includes logging for debugging and tracking the processing of tokens and merges, making it suitable for integration into larger machine learning or natural language processing pipelines.
# Imports and Dependencies

---
- `__future__.annotations`
- `re`
- `logging`
- `json`
- `os`
- `pathlib.Path`
- `typing.Any`
- `typing.Callable`
- `typing.Sequence`
- `typing.Mapping`
- `typing.Iterable`
- `typing.Protocol`
- `typing.ClassVar`
- `typing.runtime_checkable`
- `sentencepiece.SentencePieceProcessor`
- `gguf`
- `.gguf_writer.GGUFWriter`
- `transformers.AutoTokenizer`


# Global Variables

---
### logger
- **Type**: `logging.Logger`
- **Description**: The `logger` variable is an instance of the `Logger` class from the `logging` module, configured to use the name of the current module (`__name__`). This allows for logging messages that are specific to the module's context, facilitating easier debugging and monitoring of the module's behavior.
- **Use**: The `logger` is used throughout the module to log informational messages, warnings, and errors, providing insights into the module's operations and potential issues.


# Classes

---
### SpecialVocab<!-- {{#class:llama.cpp/gguf-py/gguf/vocab.SpecialVocab}} -->
- **Members**:
    - `merges`: A list of string pairs representing token merges.
    - `add_special_token`: A dictionary mapping special token types to boolean values indicating their addition.
    - `special_token_ids`: A dictionary mapping special token types to their corresponding integer IDs.
    - `chat_template`: A string or sequence of mappings representing a chat template, or None if not set.
- **Description**: The `SpecialVocab` class is designed to manage special vocabulary tokens and their configurations for a tokenizer. It handles loading and storing token merges, special token IDs, and additional special token settings from various configuration files. The class also supports adding these configurations to a GGUFWriter instance, which is used for further processing or integration. It provides mechanisms to load data from JSON files and text files, ensuring that the special tokens and merges are correctly set up for use in tokenization tasks.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab.__init__`](#SpecialVocab__init__)
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab.__repr__`](#SpecialVocab__repr__)
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab.add_to_gguf`](#SpecialVocabadd_to_gguf)
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab._load`](#SpecialVocab_load)
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab._try_load_merges_txt`](#SpecialVocab_try_load_merges_txt)
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab._set_special_token`](#SpecialVocab_set_special_token)
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab._try_load_from_tokenizer_json`](#SpecialVocab_try_load_from_tokenizer_json)
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab._try_load_from_config_json`](#SpecialVocab_try_load_from_config_json)

**Methods**

---
#### SpecialVocab\.\_\_init\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SpecialVocab.__init__}} -->
The `__init__` method initializes a `SpecialVocab` object by setting up its attributes and loading configuration data from a specified path.
- **Inputs**:
    - `path`: A string or os.PathLike object representing the file path to load configuration data from.
    - `load_merges`: A boolean indicating whether to load merge data from the specified path.
    - `special_token_types`: An optional iterable of strings specifying special token types; defaults to a predefined set if not provided.
    - `n_vocab`: An optional integer specifying the number of vocabulary tokens; used to validate special token IDs.
- **Control Flow**:
    - Initialize `special_token_ids` and `add_special_token` as empty dictionaries.
    - Set `n_vocab`, `load_merges`, and `merges` attributes based on input parameters.
    - Set `chat_template` to `None`.
    - Assign `special_token_types` to the provided iterable or a default set if not provided.
    - Call the [`_load`](#SpecialVocab_load) method with the provided path to load additional configuration data.
- **Output**: The method does not return any value; it initializes the object's state.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab._load`](#SpecialVocab_load)
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab`](#cpp/gguf-py/gguf/vocabSpecialVocab)  (Base Class)


---
#### SpecialVocab\.\_\_repr\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SpecialVocab.__repr__}} -->
The `__repr__` method provides a string representation of the `SpecialVocab` object, detailing the number of merges, special tokens, and additional special tokens.
- **Inputs**: None
- **Control Flow**:
    - The method constructs a string using the `format` method.
    - It includes the length of the `merges` list, the `special_token_ids` dictionary, and the `add_special_token` dictionary.
    - If `special_token_ids` or `add_special_token` are empty, it defaults to 'unset'.
- **Output**: A string that describes the `SpecialVocab` instance, including the number of merges, special tokens, and additional special tokens.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab`](#cpp/gguf-py/gguf/vocabSpecialVocab)  (Base Class)


---
#### SpecialVocab\.add\_to\_gguf<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SpecialVocab.add_to_gguf}} -->
The `add_to_gguf` method adds token merges, special token IDs, additional special tokens, and a chat template to a `GGUFWriter` instance, with optional logging.
- **Inputs**:
    - `gw`: An instance of `GGUFWriter` to which the merges, special tokens, and chat template will be added.
    - `quiet`: A boolean flag that, when set to `True`, suppresses logging output; defaults to `False`.
- **Control Flow**:
    - Checks if there are any merges in `self.merges`; if so, logs the number of merges (unless `quiet` is `True`) and adds them to `gw` using `gw.add_token_merges`.
    - If `self.merges` is empty and `self.load_merges` is `True`, logs a warning about missing merges.
    - Iterates over `self.special_token_ids`, retrieves the appropriate handler method from `gw` for each token type, logs the action (unless `quiet` is `True`), and calls the handler with the token ID.
    - Iterates over `self.add_special_token`, retrieves the appropriate handler method from `gw` for each token type, logs the action (unless `quiet` is `True`), and calls the handler with the token value.
    - If `self.chat_template` is not `None`, logs the action (unless `quiet` is `True`) and adds the chat template to `gw` using `gw.add_chat_template`.
- **Output**: The method does not return any value (`None`).
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_merges`](gguf_writer.py.driver.md#GGUFWriteradd_token_merges)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_chat_template`](gguf_writer.py.driver.md#GGUFWriteradd_chat_template)
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab`](#cpp/gguf-py/gguf/vocabSpecialVocab)  (Base Class)


---
#### SpecialVocab\.\_load<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SpecialVocab._load}} -->
The `_load` method attempts to load tokenizer and configuration data from specified JSON files and, if necessary, merge data from a text file.
- **Inputs**:
    - `path`: A `Path` object representing the directory path where the tokenizer and configuration files are located.
- **Control Flow**:
    - Calls [`_try_load_from_tokenizer_json`](#SpecialVocab_try_load_from_tokenizer_json) with the provided path to load tokenizer data.
    - Calls [`_try_load_from_config_json`](#SpecialVocab_try_load_from_config_json) with the provided path to load configuration data.
    - Checks if `load_merges` is True and `merges` is empty, then calls [`_try_load_merges_txt`](#SpecialVocab_try_load_merges_txt) to load merge data from a text file.
- **Output**: The method does not return any value; it modifies the instance's state by potentially updating its `merges`, `special_token_ids`, `add_special_token`, and `chat_template` attributes.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab._try_load_from_tokenizer_json`](#SpecialVocab_try_load_from_tokenizer_json)
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab._try_load_from_config_json`](#SpecialVocab_try_load_from_config_json)
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab._try_load_merges_txt`](#SpecialVocab_try_load_merges_txt)
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab`](#cpp/gguf-py/gguf/vocabSpecialVocab)  (Base Class)


---
#### SpecialVocab\.\_try\_load\_merges\_txt<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SpecialVocab._try_load_merges_txt}} -->
The `_try_load_merges_txt` method attempts to load merge rules from a 'merges.txt' file and updates the `merges` attribute of the `SpecialVocab` class if successful.
- **Inputs**:
    - `path`: A `Path` object representing the directory path where the 'merges.txt' file is expected to be located.
- **Control Flow**:
    - Constructs the path to 'merges.txt' by appending it to the provided `path`.
    - Checks if 'merges.txt' exists as a file; returns `False` if it does not.
    - Opens 'merges.txt' for reading with UTF-8 encoding.
    - Reads the first line to determine if it starts with a comment ('#'); if not, resets the file pointer to the beginning.
    - Initializes a list `merges` to store valid merge entries.
    - Iterates over each line in the file, skipping empty lines and malformed entries that do not split into exactly two parts.
    - Logs a warning for any malformed entries and continues processing the next line.
    - Appends valid merge entries to the `merges` list in the format 'part1 part2'.
    - Updates the `merges` attribute of the class with the collected merge entries.
    - Returns `True` to indicate successful loading of merge entries.
- **Output**: A boolean value indicating whether the merge rules were successfully loaded from 'merges.txt'.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab`](#cpp/gguf-py/gguf/vocabSpecialVocab)  (Base Class)


---
#### SpecialVocab\.\_set\_special\_token<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SpecialVocab._set_special_token}} -->
The `_set_special_token` method assigns a special token ID to a specified type if the ID is valid and within the vocabulary range.
- **Inputs**:
    - `typ`: A string representing the type of the special token.
    - `tid`: An identifier for the special token, which can be of any type but is expected to be an integer.
- **Control Flow**:
    - Check if `tid` is an integer; if not, return immediately.
    - If `tid` is less than 0, raise a `ValueError` indicating an invalid token ID.
    - If `n_vocab` is `None` or `tid` is less than `n_vocab`, proceed to the next check.
    - Check if `typ` is already in `special_token_ids`; if it is, return without making changes.
    - If `typ` is not in `special_token_ids`, assign `tid` to `typ` in `special_token_ids`.
    - If `tid` is not within the valid range, log a warning message indicating the token ID is out of range.
- **Output**: The method does not return any value; it modifies the `special_token_ids` dictionary of the `SpecialVocab` instance if conditions are met.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab`](#cpp/gguf-py/gguf/vocabSpecialVocab)  (Base Class)


---
#### SpecialVocab\.\_try\_load\_from\_tokenizer\_json<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SpecialVocab._try_load_from_tokenizer_json}} -->
The `_try_load_from_tokenizer_json` method attempts to load tokenizer configuration and merges from JSON files, updating the `SpecialVocab` instance with this data.
- **Inputs**:
    - `path`: A `Path` object representing the directory path where the tokenizer JSON files are located.
- **Control Flow**:
    - Check if 'tokenizer.json' exists at the given path; if not, initialize `added_tokens` as an empty dictionary.
    - If 'tokenizer.json' exists, open and parse it to load the tokenizer data.
    - If `load_merges` is True, attempt to load merges from the tokenizer data, handling different formats and encoding spaces if necessary.
    - Check if 'tokenizer_config.json' exists; if not, return True.
    - If 'tokenizer_config.json' exists, open and parse it to load the tokenizer configuration.
    - Check if 'chat_template.json' exists and load an alternative chat template if available.
    - Set `self.chat_template` if the chat template is valid; otherwise, log a warning.
    - Iterate over `self.special_token_types` to configure special tokens and their addition flags based on the tokenizer configuration.
    - For each special token type, attempt to find a matching token ID in `added_tokens` and set it using [`_set_special_token`](#SpecialVocab_set_special_token).
- **Output**: Returns a boolean value, True, indicating the process was completed successfully.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab._set_special_token`](#SpecialVocab_set_special_token)
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab`](#cpp/gguf-py/gguf/vocabSpecialVocab)  (Base Class)


---
#### SpecialVocab\.\_try\_load\_from\_config\_json<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SpecialVocab._try_load_from_config_json}} -->
The `_try_load_from_config_json` method attempts to load special token IDs from a `config.json` file located at the specified path.
- **Inputs**:
    - `path`: A `Path` object representing the directory path where the `config.json` file is expected to be located.
- **Control Flow**:
    - Constructs the path to the `config.json` file by appending 'config.json' to the provided `path`.
    - Checks if the `config.json` file exists at the constructed path; if not, returns `False`.
    - Opens the `config.json` file and loads its content as a JSON object.
    - Iterates over each special token type defined in `self.special_token_types`.
    - For each token type, retrieves the corresponding token ID from the JSON configuration and calls `self._set_special_token` to set the token ID.
    - Returns `True` after successfully processing the configuration file.
- **Output**: Returns `True` if the `config.json` file is successfully loaded and processed, otherwise returns `False` if the file does not exist.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab._set_special_token`](#SpecialVocab_set_special_token)
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SpecialVocab`](#cpp/gguf-py/gguf/vocabSpecialVocab)  (Base Class)



---
### BaseVocab<!-- {{#class:llama.cpp/gguf-py/gguf/vocab.BaseVocab}} -->
- **Decorators**: `@runtime_checkable`
- **Members**:
    - `tokenizer_model`: A class variable representing the model of the tokenizer.
    - `name`: A class variable representing the name of the vocabulary.
- **Description**: The `BaseVocab` class is a protocol that defines a basic interface for vocabulary classes, requiring them to have a `tokenizer_model` and a `name` as class variables. It serves as a foundational structure for more specific vocabulary implementations, ensuring they adhere to a common interface for tokenizer model identification and naming.
- **Inherits From**:
    - `Protocol`


---
### Vocab<!-- {{#class:llama.cpp/gguf-py/gguf/vocab.Vocab}} -->
- **Decorators**: `@runtime_checkable`
- **Members**:
    - `vocab_size`: The total size of the vocabulary including base and added tokens.
    - `added_tokens_dict`: A dictionary mapping added token strings to their integer IDs.
    - `added_tokens_list`: A list of added token strings.
    - `fname_tokenizer`: The file path to the tokenizer file.
- **Description**: The `Vocab` class is a protocol that extends `BaseVocab` and is used to define a vocabulary structure with a specified size, added tokens, and a tokenizer file path. It provides a framework for managing vocabulary data, including the ability to iterate over all tokens, which are represented as tuples containing the token bytes, a float score, and a token type. This class is designed to be used as a base for more specific vocabulary implementations.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/vocab.Vocab.__init__`](#Vocab__init__)
    - [`llama.cpp/gguf-py/gguf/vocab.Vocab.all_tokens`](#Vocaball_tokens)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/vocab.BaseVocab`](#cpp/gguf-py/gguf/vocabBaseVocab)

**Methods**

---
#### Vocab\.\_\_init\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.Vocab.__init__}} -->
The `__init__` method initializes a `Vocab` object with a specified base path for loading vocabulary data.
- **Decorators**: `@runtime_checkable`
- **Inputs**:
    - `base_path`: A `Path` object representing the base directory path where vocabulary files are located.
- **Control Flow**:
    - The method is defined but not implemented in the `Vocab` class, indicating it is intended to be overridden by subclasses.
    - Subclasses like `BpeVocab`, `SentencePieceVocab`, and `LlamaHfVocab` provide specific implementations for loading vocabulary data from files located at the `base_path`.
- **Output**: The method does not return any value; it initializes the object state.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.Vocab`](#cpp/gguf-py/gguf/vocabVocab)  (Base Class)


---
#### Vocab\.all\_tokens<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.Vocab.all_tokens}} -->
The `all_tokens` method returns an iterable of all tokens, including both base and added tokens, each represented as a tuple of bytes, a float score, and a token type.
- **Inputs**: None
- **Control Flow**:
    - The method first yields tokens from the base vocabulary by calling a method specific to the vocabulary type (e.g., `bpe_tokens`, `sentencepiece_tokens`, or `hf_tokens`).
    - It then yields tokens from the added tokens list by calling the `added_tokens` method.
- **Output**: An iterable of tuples, where each tuple contains a token as bytes, a float score, and a `gguf.TokenType` indicating the type of the token.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.Vocab`](#cpp/gguf-py/gguf/vocabVocab)  (Base Class)



---
### NoVocab<!-- {{#class:llama.cpp/gguf-py/gguf/vocab.NoVocab}} -->
- **Members**:
    - `tokenizer_model`: A class variable indicating the tokenizer model type, set to 'no_vocab'.
    - `name`: A class variable representing the name of the vocabulary, set to 'no_vocab'.
- **Description**: The NoVocab class is a simple implementation of the BaseVocab protocol, designed for models that do not require an integrated vocabulary. It defines two class variables, 'tokenizer_model' and 'name', both set to 'no_vocab', indicating the absence of a vocabulary. This class is primarily used as a placeholder or default option when a vocabulary is not needed in a model.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/vocab.NoVocab.__repr__`](#NoVocab__repr__)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/vocab.BaseVocab`](#cpp/gguf-py/gguf/vocabBaseVocab)

**Methods**

---
#### NoVocab\.\_\_repr\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.NoVocab.__repr__}} -->
The `__repr__` method for the `NoVocab` class returns a string representation indicating that the model does not have an integrated vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The method directly returns a fixed string without any conditions or iterations.
- **Output**: A string that reads '<NoVocab for a model without integrated vocabulary>'.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.NoVocab`](#cpp/gguf-py/gguf/vocabNoVocab)  (Base Class)



---
### BpeVocab<!-- {{#class:llama.cpp/gguf-py/gguf/vocab.BpeVocab}} -->
- **Members**:
    - `tokenizer_model`: A class variable indicating the tokenizer model used, set to 'gpt2'.
    - `name`: A class variable representing the name of the vocabulary, set to 'bpe'.
    - `vocab`: An instance variable holding the main vocabulary loaded from a JSON file.
    - `added_tokens_dict`: A dictionary mapping added token strings to their IDs.
    - `added_tokens_list`: A list of added token strings sorted by their IDs.
    - `vocab_size_base`: An integer representing the size of the base vocabulary.
    - `vocab_size`: An integer representing the total size of the vocabulary including added tokens.
    - `fname_tokenizer`: A Path object indicating the file name of the tokenizer used.
- **Description**: The BpeVocab class is a specialized vocabulary handler for Byte Pair Encoding (BPE) tokenizers, specifically designed to work with GPT-2 models. It extends the Vocab protocol and manages both the base vocabulary and any additional tokens that may be included. The class is responsible for loading vocabulary data from JSON files, ensuring that added tokens do not overlap with the main vocabulary, and maintaining a consistent and sequential ID range for these tokens. It provides methods to iterate over both base and added tokens, yielding them as tuples containing the token bytes, a score, and a token type. The class also includes mechanisms to handle different tokenizer file formats, distinguishing between 'slow' and 'fast' tokenizers, and raises errors if the expected tokenizer format is not found.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/vocab.BpeVocab.__init__`](#BpeVocab__init__)
    - [`llama.cpp/gguf-py/gguf/vocab.BpeVocab.bpe_tokens`](#BpeVocabbpe_tokens)
    - [`llama.cpp/gguf-py/gguf/vocab.BpeVocab.added_tokens`](#BpeVocabadded_tokens)
    - [`llama.cpp/gguf-py/gguf/vocab.BpeVocab.all_tokens`](#BpeVocaball_tokens)
    - [`llama.cpp/gguf-py/gguf/vocab.BpeVocab.__repr__`](#BpeVocab__repr__)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/vocab.Vocab`](#cpp/gguf-py/gguf/vocabVocab)

**Methods**

---
#### BpeVocab\.\_\_init\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.BpeVocab.__init__}} -->
The `__init__` method initializes a `BpeVocab` object by loading vocabulary and added tokens from specified files, ensuring the integrity of token IDs, and setting up various attributes related to the vocabulary.
- **Inputs**:
    - `base_path`: A `Path` object representing the directory path where the tokenizer and vocabulary files are located.
- **Control Flow**:
    - Initialize an empty dictionary `added_tokens` to store added tokens.
    - Check if 'vocab.json' exists in the `base_path`; if it does, load the vocabulary from this file and attempt to load added tokens from 'added_tokens.json'.
    - If 'vocab.json' does not exist, assume a 'fast' tokenizer and load the tokenizer configuration from 'tokenizer.json'.
    - Verify that the tokenizer model is of type 'BPE' and uses a 'ByteLevel' decoder; raise a `FileNotFoundError` if not.
    - Extract the vocabulary from the tokenizer model and check for any added tokens, ensuring they do not duplicate existing vocabulary entries.
    - Calculate the expected range of IDs for added tokens and verify that the actual IDs match this range; raise a `ValueError` if they do not.
    - Sort the added tokens by their IDs and store them in `self.added_tokens_dict` and `self.added_tokens_list`.
    - Set various attributes such as `self.vocab_size_base`, `self.vocab_size`, and `self.fname_tokenizer` based on the loaded data.
- **Output**: The method does not return a value but initializes the `BpeVocab` object with attributes such as `vocab`, `added_tokens_dict`, `added_tokens_list`, `vocab_size_base`, `vocab_size`, and `fname_tokenizer`.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.BpeVocab`](#cpp/gguf-py/gguf/vocabBpeVocab)  (Base Class)


---
#### BpeVocab\.bpe\_tokens<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.BpeVocab.bpe_tokens}} -->
The `bpe_tokens` method yields tuples of base vocabulary tokens, a score of 0.0, and a token type of NORMAL from the BPE vocabulary.
- **Inputs**: None
- **Control Flow**:
    - A reverse mapping of the vocabulary is created from token IDs to encoded tokens.
    - The method iterates over the vocabulary using an index.
    - For each token in the vocabulary, it yields a tuple containing the encoded token, a score of 0.0, and the token type `gguf.TokenType.NORMAL`.
- **Output**: An iterable of tuples, each containing a token as bytes, a float score of 0.0, and a token type of `gguf.TokenType.NORMAL`.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.BpeVocab`](#cpp/gguf-py/gguf/vocabBpeVocab)  (Base Class)


---
#### BpeVocab\.added\_tokens<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.BpeVocab.added_tokens}} -->
The `added_tokens` method yields encoded added tokens with a fixed score and token type from the `added_tokens_list`.
- **Inputs**: None
- **Control Flow**:
    - Iterates over each text in `self.added_tokens_list`.
    - For each text, sets a fixed score of -1000.0.
    - Yields a tuple containing the UTF-8 encoded text, the fixed score, and the token type `gguf.TokenType.CONTROL`.
- **Output**: An iterable of tuples, each containing a UTF-8 encoded added token, a fixed score of -1000.0, and the token type `gguf.TokenType.CONTROL`.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.BpeVocab`](#cpp/gguf-py/gguf/vocabBpeVocab)  (Base Class)


---
#### BpeVocab\.all\_tokens<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.BpeVocab.all_tokens}} -->
The `all_tokens` method in the `BpeVocab` class yields all tokens from both the base BPE vocabulary and the added tokens.
- **Inputs**: None
- **Control Flow**:
    - The method first calls `self.bpe_tokens()` to yield tokens from the base BPE vocabulary.
    - It then calls `self.added_tokens()` to yield tokens from the list of added tokens.
- **Output**: An iterable of tuples, each containing a token as bytes, a float score, and a `gguf.TokenType` indicating the type of token.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/vocab.BpeVocab.bpe_tokens`](#BpeVocabbpe_tokens)
    - [`llama.cpp/gguf-py/gguf/vocab.BpeVocab.added_tokens`](#BpeVocabadded_tokens)
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.BpeVocab`](#cpp/gguf-py/gguf/vocabBpeVocab)  (Base Class)


---
#### BpeVocab\.\_\_repr\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.BpeVocab.__repr__}} -->
The `__repr__` method provides a string representation of the `BpeVocab` object, indicating the number of base and added tokens.
- **Inputs**: None
- **Control Flow**:
    - The method constructs a string using the `vocab_size_base` and the length of `added_tokens_list`.
    - It returns the constructed string in the format: '<BpeVocab with {vocab_size_base} base tokens and {len(added_tokens_list)} added tokens>'.
- **Output**: A string that describes the `BpeVocab` object, including the number of base and added tokens.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.BpeVocab`](#cpp/gguf-py/gguf/vocabBpeVocab)  (Base Class)



---
### SentencePieceVocab<!-- {{#class:llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab}} -->
- **Members**:
    - `tokenizer_model`: Specifies the model type for the tokenizer, set to 'llama'.
    - `name`: Defines the name of the vocabulary, set to 'spm'.
    - `added_tokens_dict`: Stores a dictionary of added tokens with their corresponding IDs.
    - `added_tokens_list`: Contains a list of added token pieces.
    - `vocab_size_base`: Holds the size of the base vocabulary.
    - `vocab_size`: Represents the total vocabulary size, including added tokens.
    - `fname_tokenizer`: Stores the file path to the tokenizer model.
- **Description**: The SentencePieceVocab class is a specialized vocabulary handler that extends the Vocab protocol, designed to work with SentencePiece tokenizers. It initializes by loading a tokenizer model from a specified path and manages both base and added tokens, ensuring that added token IDs are sequential and do not overlap with the base vocabulary. The class provides methods to iterate over all tokens, including both base and added tokens, and represents the vocabulary with a clear distinction between base and added tokens in its string representation.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab.__init__`](#SentencePieceVocab__init__)
    - [`llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab.sentencepiece_tokens`](#SentencePieceVocabsentencepiece_tokens)
    - [`llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab.added_tokens`](#SentencePieceVocabadded_tokens)
    - [`llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab.all_tokens`](#SentencePieceVocaball_tokens)
    - [`llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab.__repr__`](#SentencePieceVocab__repr__)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/vocab.Vocab`](#cpp/gguf-py/gguf/vocabVocab)

**Methods**

---
#### SentencePieceVocab\.\_\_init\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab.__init__}} -->
The `__init__` method initializes a `SentencePieceVocab` object by loading a tokenizer model and any additional tokens from specified file paths, and sets up the vocabulary size and added tokens.
- **Inputs**:
    - `base_path`: A `Path` object representing the base directory where the tokenizer model and added tokens JSON files are expected to be located.
- **Control Flow**:
    - Initialize an empty dictionary `added_tokens` to store additional tokens.
    - Check if the tokenizer model file exists at the normal location (`base_path / 'tokenizer.model'`).
    - If the tokenizer model file exists, attempt to load additional tokens from `added_tokens.json` in the same directory.
    - If the tokenizer model file does not exist at the normal location, check an alternate location (`base_path.parent / 'tokenizer.model'`).
    - If the tokenizer model file is not found in either location, raise a `FileNotFoundError`.
    - Load the tokenizer model using `SentencePieceProcessor` and retrieve the base vocabulary size.
    - Identify new tokens by filtering `added_tokens` for IDs greater than or equal to the base vocabulary size.
    - Check if the new token IDs are sequential and raise a `ValueError` if they are not.
    - Store the added tokens and calculate the total vocabulary size including added tokens.
- **Output**: The method does not return a value but initializes the `SentencePieceVocab` object with attributes such as `sentencepiece_tokenizer`, `added_tokens_dict`, `added_tokens_list`, `vocab_size_base`, `vocab_size`, and `fname_tokenizer`.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab`](#cpp/gguf-py/gguf/vocabSentencePieceVocab)  (Base Class)


---
#### SentencePieceVocab\.sentencepiece\_tokens<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab.sentencepiece_tokens}} -->
The `sentencepiece_tokens` method generates an iterable of tuples containing encoded text, score, and token type for each token in the SentencePiece vocabulary.
- **Inputs**: None
- **Control Flow**:
    - Retrieve the SentencePiece tokenizer instance from the class attribute.
    - Iterate over the range of the tokenizer's vocabulary size.
    - For each token ID, convert the token to a piece of text and encode it as UTF-8 bytes.
    - Retrieve the score of the token using the tokenizer's `GetScore` method.
    - Initialize the token type as `NORMAL`.
    - Check if the token is unknown, control, unused, or a byte, and update the token type accordingly.
    - Yield a tuple containing the encoded text, score, and token type.
- **Output**: An iterable of tuples, each containing a byte-encoded text, a float score, and a `gguf.TokenType` indicating the type of the token.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab`](#cpp/gguf-py/gguf/vocabSentencePieceVocab)  (Base Class)


---
#### SentencePieceVocab\.added\_tokens<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab.added_tokens}} -->
The `added_tokens` method yields encoded added tokens with a fixed score and token type.
- **Inputs**: None
- **Control Flow**:
    - Iterates over each text in `self.added_tokens_list`.
    - Encodes each text to UTF-8 bytes.
    - Assigns a fixed score of -1000.0 to each token.
    - Yields a tuple of the encoded text, score, and `gguf.TokenType.USER_DEFINED`.
- **Output**: An iterable of tuples, each containing a byte-encoded token, a float score, and a `gguf.TokenType` indicating the token type.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab`](#cpp/gguf-py/gguf/vocabSentencePieceVocab)  (Base Class)


---
#### SentencePieceVocab\.all\_tokens<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab.all_tokens}} -->
The `all_tokens` method yields all tokens from both the SentencePiece tokenizer and any added tokens, each represented as a tuple of encoded text, score, and token type.
- **Inputs**: None
- **Control Flow**:
    - The method first calls `self.sentencepiece_tokens()` to yield tokens from the SentencePiece tokenizer.
    - It then calls `self.added_tokens()` to yield any additional tokens that have been added.
- **Output**: An iterable of tuples, where each tuple contains a token's encoded text as bytes, a float score, and a `gguf.TokenType` indicating the type of the token.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab.sentencepiece_tokens`](#SentencePieceVocabsentencepiece_tokens)
    - [`llama.cpp/gguf-py/gguf/vocab.BpeVocab.added_tokens`](#BpeVocabadded_tokens)
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab`](#cpp/gguf-py/gguf/vocabSentencePieceVocab)  (Base Class)


---
#### SentencePieceVocab\.\_\_repr\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab.__repr__}} -->
The `__repr__` method provides a string representation of the `SentencePieceVocab` object, indicating the number of base and added tokens.
- **Inputs**: None
- **Control Flow**:
    - The method constructs a string using the `vocab_size_base` and the length of `added_tokens_list` attributes of the `SentencePieceVocab` instance.
    - It returns the constructed string in the format: '<SentencePieceVocab with {vocab_size_base} base tokens and {len(added_tokens_list)} added tokens>'.
- **Output**: A string that describes the `SentencePieceVocab` instance, including the number of base and added tokens.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.SentencePieceVocab`](#cpp/gguf-py/gguf/vocabSentencePieceVocab)  (Base Class)



---
### LlamaHfVocab<!-- {{#class:llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab}} -->
- **Members**:
    - `tokenizer_model`: Specifies the tokenizer model type as 'llama'.
    - `name`: Defines the name of the vocabulary as 'hfft'.
    - `tokenizer`: Holds the tokenizer instance loaded from the specified base path.
    - `added_tokens_list`: Stores a list of tokens that have been added to the base vocabulary.
    - `added_tokens_dict`: Maps added token strings to their corresponding indices.
    - `added_tokens_ids`: Contains a set of indices for the added tokens.
    - `specials`: Maps special token strings to their corresponding IDs.
    - `special_ids`: Contains a set of IDs for all special tokens.
    - `vocab_size_base`: Represents the size of the base vocabulary.
    - `vocab_size`: Represents the total vocabulary size including added tokens.
    - `fname_tokenizer`: Stores the file path to the tokenizer JSON file.
- **Description**: The LlamaHfVocab class is a specialized vocabulary handler for the Llama tokenizer model, designed to work with the Hugging Face Transformers library. It initializes by loading a tokenizer from a specified path, ensuring compatibility with the BPE tokenizer type, and manages both base and added tokens. The class provides mechanisms to handle special tokens, calculate vocabulary sizes, and yield tokens with their types and scores. It also checks for the presence of newline tokens and requires the 'transformers' package for full functionality.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.__init__`](#LlamaHfVocab__init__)
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.hf_tokens`](#LlamaHfVocabhf_tokens)
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.get_token_type`](#LlamaHfVocabget_token_type)
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.get_token_score`](#LlamaHfVocabget_token_score)
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.added_tokens`](#LlamaHfVocabadded_tokens)
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.has_newline_token`](#LlamaHfVocabhas_newline_token)
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.all_tokens`](#LlamaHfVocaball_tokens)
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.__repr__`](#LlamaHfVocab__repr__)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/vocab.Vocab`](#cpp/gguf-py/gguf/vocabVocab)

**Methods**

---
#### LlamaHfVocab\.\_\_init\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.__init__}} -->
The `__init__` method initializes a `LlamaHfVocab` object by loading a tokenizer from a specified path, verifying its compatibility, and setting up token lists and dictionaries.
- **Inputs**:
    - `base_path`: A `Path` object representing the directory path where the tokenizer files are located.
- **Control Flow**:
    - Constructs the path to the 'tokenizer.json' file using the provided `base_path`.
    - Opens and reads the 'tokenizer.json' file to load the tokenizer configuration.
    - Checks if the tokenizer model is compatible with Llama 3 and raises a `TypeError` if it is not.
    - Verifies the tokenizer type and decoder type, raising a `FileNotFoundError` if they do not match expected values.
    - Attempts to import the `AutoTokenizer` from the `transformers` package, raising an `ImportError` if the package is not installed.
    - Initializes the `AutoTokenizer` using the `from_pretrained` method with local file paths only.
    - Asserts that the tokenizer is a 'fast' tokenizer.
    - Initializes lists and dictionaries to store added tokens and their IDs.
    - Iterates over added tokens from the tokenizer, storing those not in the base vocabulary in the initialized lists and dictionaries.
    - Stores special tokens and their IDs from the tokenizer.
    - Calculates and sets the base and total vocabulary sizes.
- **Output**: The method does not return a value but initializes the `LlamaHfVocab` instance with attributes such as `tokenizer`, `added_tokens_list`, `added_tokens_dict`, `specials`, `special_ids`, `vocab_size_base`, `vocab_size`, and `fname_tokenizer`.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab`](#cpp/gguf-py/gguf/vocabLlamaHfVocab)  (Base Class)


---
#### LlamaHfVocab\.hf\_tokens<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.hf_tokens}} -->
The `hf_tokens` method generates an iterable of tuples containing token text in bytes, a score, and a token type for each base vocabulary token, excluding added tokens.
- **Inputs**: None
- **Control Flow**:
    - Create a reverse mapping of token IDs to token text from the tokenizer's vocabulary.
    - Iterate over each token ID in the base vocabulary size range.
    - Skip processing if the token ID is in the set of added token IDs.
    - Convert the token text to bytes using UTF-8 encoding.
    - Yield a tuple containing the token text in bytes, the token's score, and the token's type.
- **Output**: An iterable of tuples, each containing a token's text in bytes, its score as a float, and its type as a `gguf.TokenType`.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.get_token_score`](#LlamaHfVocabget_token_score)
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.get_token_type`](#LlamaHfVocabget_token_type)
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab`](#cpp/gguf-py/gguf/vocabLlamaHfVocab)  (Base Class)


---
#### LlamaHfVocab\.get\_token\_type<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.get_token_type}} -->
The `get_token_type` method determines the type of a token based on its ID, text, and whether it is a special token.
- **Inputs**:
    - `token_id`: An integer representing the unique identifier of the token.
    - `token_text`: A byte string representing the text of the token.
    - `special_ids`: A set of integers representing the IDs of special tokens.
- **Control Flow**:
    - Check if the token text matches the pattern for a byte token using a regular expression.
    - If the token text matches the byte token pattern, return `gguf.TokenType.BYTE`.
    - If the token ID is in the set of special IDs, return `gguf.TokenType.CONTROL`.
    - If the token ID is not in the set of special IDs, return `gguf.TokenType.NORMAL`.
- **Output**: Returns a `gguf.TokenType` indicating whether the token is a BYTE, CONTROL, or NORMAL token.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab`](#cpp/gguf-py/gguf/vocabLlamaHfVocab)  (Base Class)


---
#### LlamaHfVocab\.get\_token\_score<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.get_token_score}} -->
The `get_token_score` method returns a default score for a given token ID, with the logic for determining the actual score yet to be implemented.
- **Inputs**:
    - `token_id`: An integer representing the ID of the token for which the score is to be retrieved.
- **Control Flow**:
    - The method currently contains a placeholder comment indicating where the logic for determining the token's score should be implemented.
    - The method returns a default score of -1000.0 for any given token ID.
- **Output**: A float representing the score of the token, currently set to a default value of -1000.0.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab`](#cpp/gguf-py/gguf/vocabLlamaHfVocab)  (Base Class)


---
#### LlamaHfVocab\.added\_tokens<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.added_tokens}} -->
The `added_tokens` method yields encoded added tokens along with their scores and types, distinguishing between special and user-defined tokens.
- **Inputs**: None
- **Control Flow**:
    - Iterates over each token in `self.added_tokens_list`.
    - Checks if the token is in `self.specials` to determine if it is a special token.
    - If the token is special, retrieves its type and score using [`get_token_type`](#LlamaHfVocabget_token_type) and [`get_token_score`](#LlamaHfVocabget_token_score) methods.
    - If the token is not special, assigns it a type of `gguf.TokenType.USER_DEFINED` and a default score of -1000.0.
    - Yields a tuple containing the UTF-8 encoded token, its score, and its type.
- **Output**: An iterable of tuples, each containing a UTF-8 encoded token (as bytes), a float score, and a `gguf.TokenType` indicating the token type.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.get_token_type`](#LlamaHfVocabget_token_type)
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.get_token_score`](#LlamaHfVocabget_token_score)
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab`](#cpp/gguf-py/gguf/vocabLlamaHfVocab)  (Base Class)


---
#### LlamaHfVocab\.has\_newline\_token<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.has_newline_token}} -->
The `has_newline_token` method checks if the tokenizer's vocabulary contains a newline token represented as "<0x0A>" or "\n".
- **Inputs**: None
- **Control Flow**:
    - The method checks if the string "<0x0A>" is present in the tokenizer's vocabulary.
    - If not found, it checks if the newline character "\n" is present in the tokenizer's vocabulary.
    - The method returns `True` if either of the checks is successful, otherwise it returns `False`.
- **Output**: A boolean value indicating the presence of a newline token in the tokenizer's vocabulary.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab`](#cpp/gguf-py/gguf/vocabLlamaHfVocab)  (Base Class)


---
#### LlamaHfVocab\.all\_tokens<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.all_tokens}} -->
The `all_tokens` method yields all tokens from both the base vocabulary and the added tokens in the form of tuples containing the token's byte representation, score, and type.
- **Inputs**: None
- **Control Flow**:
    - The method first calls `self.hf_tokens()` to yield tokens from the base vocabulary.
    - It then calls `self.added_tokens()` to yield tokens that have been added to the vocabulary.
- **Output**: An iterable of tuples, each containing a token's byte representation, a float score, and a `gguf.TokenType` indicating the token's type.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.hf_tokens`](#LlamaHfVocabhf_tokens)
    - [`llama.cpp/gguf-py/gguf/vocab.BpeVocab.added_tokens`](#BpeVocabadded_tokens)
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab`](#cpp/gguf-py/gguf/vocabLlamaHfVocab)  (Base Class)


---
#### LlamaHfVocab\.\_\_repr\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab.__repr__}} -->
The `__repr__` method provides a string representation of the `LlamaHfVocab` object, indicating the number of base and added tokens.
- **Inputs**: None
- **Control Flow**:
    - The method constructs a string using the `vocab_size_base` and the length of `added_tokens_list` attributes of the `LlamaHfVocab` instance.
    - It returns the constructed string in the format: '<LlamaHfVocab with {base_tokens} base tokens and {added_tokens} added tokens>'.
- **Output**: A string that describes the `LlamaHfVocab` instance, including the count of base and added tokens.
- **See also**: [`llama.cpp/gguf-py/gguf/vocab.LlamaHfVocab`](#cpp/gguf-py/gguf/vocabLlamaHfVocab)  (Base Class)



