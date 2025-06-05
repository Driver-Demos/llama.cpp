# Purpose
This Python code file defines a comprehensive set of constants, classes, and enumerations related to the configuration and metadata of machine learning models, particularly those using the GGUF (General Graph Universal Format) and GGML (General Graph Machine Learning) frameworks. The file is structured to provide a detailed mapping of model architectures, tensor types, quantization types, and various metadata keys that are essential for defining and managing machine learning models. The code includes nested classes for organizing metadata keys into categories such as general information, LLM (Large Language Model) specifics, attention mechanisms, RoPE (Rotary Position Embedding) configurations, and more. Additionally, it defines several enumerations for model architectures, tensor types, quantization types, and other configuration parameters, which are crucial for ensuring consistency and interoperability across different components of a machine learning system.

The file serves as a library module intended to be imported and used in other parts of a software system that deals with machine learning models. It provides a standardized way to reference and manage model-related metadata and configurations, facilitating the development and deployment of machine learning models. The code does not define any public APIs or external interfaces directly but rather provides a foundational set of definitions that can be utilized by other modules or scripts to interact with machine learning models in a structured and consistent manner. The use of enumerations and constants ensures that the code is both extensible and maintainable, allowing for easy updates and modifications as new model architectures and configurations are developed.
# Imports and Dependencies

---
- `__future__.annotations`
- `enum.Enum`
- `enum.IntEnum`
- `enum.auto`
- `typing.Any`


# Global Variables

---
### GGUF\_MAGIC
- **Type**: `string`
- **Description**: `GGUF_MAGIC` is a constant that represents the magic number for the GGUF file format, specifically set to the hexadecimal value `0x46554747`, which corresponds to the ASCII string 'GGUF'. This magic number is used to identify GGUF files and ensure that they are correctly recognized by the software that processes them.
- **Use**: This variable is used to validate the format of files being read or written, ensuring they conform to the expected GGUF structure.


---
### GGUF\_VERSION
- **Type**: `integer`
- **Description**: `GGUF_VERSION` is a global constant that represents the version number of the GGUF format. It is set to `3`, indicating the specific iteration of the format being used.
- **Use**: This variable is used to identify the version of the GGUF format in the codebase.


---
### GGUF\_DEFAULT\_ALIGNMENT
- **Type**: `integer`
- **Description**: The `GGUF_DEFAULT_ALIGNMENT` variable is a constant that defines the default alignment value used in the GGUF (Generalized Graph Universal Format) data structure. It is set to 32, which is a common alignment size for memory allocation in many computing environments.
- **Use**: This variable is used to ensure that data structures are aligned in memory according to the specified alignment value, which can improve performance and compatibility.


---
### GGML\_QUANT\_VERSION
- **Type**: `int`
- **Description**: The `GGML_QUANT_VERSION` variable holds the quantization version number for the GGML model, which is set to 2. This version is referenced from the `ggml.h` header file, indicating the specific quantization scheme used.
- **Use**: This variable is used to specify the quantization version in the context of model loading and processing.


---
### MODEL\_ARCH\_NAMES
- **Type**: `dict[MODEL_ARCH, str]`
- **Description**: The `MODEL_ARCH_NAMES` variable is a dictionary that maps various model architecture enumerations from `MODEL_ARCH` to their corresponding string representations. This mapping provides a human-readable name for each architecture, facilitating easier identification and usage in the codebase.
- **Use**: This variable is used to retrieve the string name of a model architecture based on its enumeration value.


---
### VISION\_PROJECTOR\_TYPE\_NAMES
- **Type**: `dict[VISION_PROJECTOR_TYPE, str]`
- **Description**: The `VISION_PROJECTOR_TYPE_NAMES` variable is a dictionary that maps specific types of vision projectors, defined by the `VISION_PROJECTOR_TYPE` enumeration, to their corresponding string representations. This mapping allows for easy reference and identification of different projector types used in the vision processing context.
- **Use**: This variable is used to retrieve the string name associated with a specific `VISION_PROJECTOR_TYPE` for display or logging purposes.


---
### TENSOR\_NAMES
- **Type**: `dict[MODEL_TENSOR, str]`
- **Description**: The `TENSOR_NAMES` variable is a dictionary that maps `MODEL_TENSOR` enumeration values to their corresponding string representations. This mapping is essential for identifying and referencing various tensor types used in machine learning models, facilitating easier access and management of tensor names throughout the code.
- **Use**: This variable is used to retrieve the string name associated with a specific tensor type, aiding in the organization and clarity of tensor-related operations in the model.


---
### MODEL\_TENSORS
- **Type**: `dict[MODEL_ARCH, list[MODEL_TENSOR]]`
- **Description**: `MODEL_TENSORS` is a dictionary that maps various model architectures (defined by the `MODEL_ARCH` enum) to their corresponding lists of model tensors (defined by the `MODEL_TENSOR` enum). Each architecture has a specific set of tensors that are utilized during model operations, such as attention mechanisms and feed-forward networks.
- **Use**: This variable is used to retrieve the list of tensors associated with a specific model architecture, facilitating the configuration and execution of model operations.


---
### MODEL\_TENSOR\_SKIP
- **Type**: `dict[MODEL_ARCH, list[MODEL_TENSOR]]`
- **Description**: The `MODEL_TENSOR_SKIP` variable is a dictionary that maps various model architectures (defined by the `MODEL_ARCH` enum) to lists of tensors (defined by the `MODEL_TENSOR` enum) that should be excluded from serialization. This is useful for optimizing the model storage by skipping certain tensors that may not be necessary for all architectures.
- **Use**: This variable is used to determine which tensors to skip during the serialization process for different model architectures.


---
### QK\_K
- **Type**: `string`
- **Description**: `QK_K` is a constant integer variable set to 256, which likely represents a specific size or dimension used in quantization or model architecture. It is used in the context of defining quantization sizes for various types in the `GGML_QUANT_SIZES` dictionary.
- **Use**: `QK_K` is utilized to define the size of certain quantization types in the `GGML_QUANT_SIZES` dictionary.


---
### GGML\_QUANT\_SIZES
- **Type**: `dict[GGMLQuantizationType, tuple[int, int]]`
- **Description**: The `GGML_QUANT_SIZES` variable is a dictionary that maps different quantization types, defined by the `GGMLQuantizationType` enumeration, to their corresponding sizes. Each entry in the dictionary consists of a tuple containing two integers: the first integer represents the block size, and the second integer represents the type size for that quantization type.
- **Use**: This variable is used to retrieve the size information for various quantization types, which is essential for memory allocation and processing in machine learning models.


---
### KEY\_GENERAL\_ARCHITECTURE
- **Type**: `string`
- **Description**: `KEY_GENERAL_ARCHITECTURE` is a global constant that holds the key for the architecture metadata in a general context, specifically defined as `Keys.General.ARCHITECTURE`. This key is used to identify the architecture type of a model in metadata storage or processing.
- **Use**: This variable is used to access the architecture information of a model in metadata.


---
### KEY\_GENERAL\_QUANTIZATION\_VERSION
- **Type**: `string`
- **Description**: The variable `KEY_GENERAL_QUANTIZATION_VERSION` holds the key for the quantization version metadata in the general context of the application. It is derived from the `Keys.General` class, specifically referencing the `QUANTIZATION_VERSION` attribute, which is used to identify the quantization method applied to the model data.
- **Use**: This variable is used to access the quantization version information for models, ensuring compatibility and proper handling of quantized data.


---
### KEY\_GENERAL\_ALIGNMENT
- **Type**: `string`
- **Description**: `KEY_GENERAL_ALIGNMENT` is a global constant that holds the key for the general alignment metadata in the context of model configuration. It is defined as `Keys.General.ALIGNMENT`, which is a string representing the alignment property of a model. This key is used to access or specify the alignment attribute in metadata related to machine learning models.
- **Use**: This variable is used to reference the alignment key in metadata for models.


---
### KEY\_GENERAL\_NAME
- **Type**: `string`
- **Description**: The variable `KEY_GENERAL_NAME` is a string constant that represents the key for the general name metadata in a model's metadata structure. It is defined as `Keys.General.NAME`, which corresponds to the string 'general.name'.
- **Use**: This variable is used to access or reference the general name metadata in various contexts where model metadata is processed.


---
### KEY\_GENERAL\_AUTHOR
- **Type**: `string`
- **Description**: The variable `KEY_GENERAL_AUTHOR` is a string constant that holds the key for the author's metadata in a general context. It is defined as `Keys.General.AUTHOR`, which is part of a structured enumeration for metadata keys used in the application.
- **Use**: This variable is used to reference the author's key in metadata-related operations, ensuring consistency and clarity in accessing author information.


---
### KEY\_GENERAL\_URL
- **Type**: `string`
- **Description**: `KEY_GENERAL_URL` is a global constant that holds the URL key for general metadata in the `Keys.General` class. It is used to specify the location of a model's website or paper, providing a reference for users to access additional information about the model.
- **Use**: This variable is used to retrieve the URL associated with general metadata for models.


---
### KEY\_GENERAL\_DESCRIPTION
- **Type**: `string`
- **Description**: `KEY_GENERAL_DESCRIPTION` is a global constant that holds the key for accessing the general description metadata of a model. It is defined as `Keys.General.DESCRIPTION`, which is a string representing the metadata key used to retrieve the description of the model in a structured format.
- **Use**: This variable is used to reference the general description key when accessing model metadata.


---
### KEY\_GENERAL\_LICENSE
- **Type**: `string`
- **Description**: `KEY_GENERAL_LICENSE` is a global constant that holds the key for accessing the license information in the metadata structure defined within the `Keys.General` class. It is used to standardize the retrieval of licensing details across different components of the application.
- **Use**: This variable is used to reference the license key when accessing or storing licensing information in metadata.


---
### KEY\_GENERAL\_SOURCE\_URL
- **Type**: `string`
- **Description**: The variable `KEY_GENERAL_SOURCE_URL` is a global constant that holds the key for the source URL metadata in a model's general information. It is defined as `Keys.General.SOURCE_URL`, which is a string representing the key used to access the source URL of the model from a structured metadata format.
- **Use**: This variable is used to retrieve the source URL associated with a model's metadata.


---
### KEY\_GENERAL\_FILE\_TYPE
- **Type**: `string`
- **Description**: The variable `KEY_GENERAL_FILE_TYPE` is a constant that holds the key for the general file type metadata in the context of model metadata. It is defined as `Keys.General.FILE_TYPE`, which is a string identifier used to specify the type of file being processed or referenced in the system.
- **Use**: This variable is used to access the general file type key within the metadata structure, facilitating the retrieval and management of file type information.


---
### KEY\_VOCAB\_SIZE
- **Type**: `string`
- **Description**: `KEY_VOCAB_SIZE` is a global constant that holds the vocabulary size for a language model, defined as a string template in the `Keys.LLM` class. It is used to specify the number of unique tokens that the model can recognize and process.
- **Use**: This variable is utilized to configure the vocabulary size of the language model during its initialization or setup.


---
### KEY\_CONTEXT\_LENGTH
- **Type**: `string`
- **Description**: `KEY_CONTEXT_LENGTH` is a global constant that retrieves the context length for a language model from the `Keys.LLM` class. It is defined as a string template that can be formatted with specific architecture identifiers to specify the context length for different model architectures.
- **Use**: This variable is used to access the context length configuration for various language models.


---
### KEY\_EMBEDDING\_LENGTH
- **Type**: `string`
- **Description**: The variable `KEY_EMBEDDING_LENGTH` is a constant that retrieves the embedding length for a specific architecture from the `Keys.LLM` class. It is used to define the dimensionality of the embedding layer in a language model, which is crucial for processing input data effectively.
- **Use**: This variable is used to specify the embedding length when configuring or initializing language model architectures.


---
### KEY\_BLOCK\_COUNT
- **Type**: `string`
- **Description**: `KEY_BLOCK_COUNT` is a global constant that retrieves the block count configuration for a language model from the `Keys.LLM` class. It is defined as a string template that incorporates the architecture identifier, allowing for dynamic configuration based on the specific model architecture being used.
- **Use**: This variable is used to access the block count setting for different language model architectures.


---
### KEY\_FEED\_FORWARD\_LENGTH
- **Type**: `string`
- **Description**: `KEY_FEED_FORWARD_LENGTH` is a global constant that retrieves the feed-forward length configuration for a language model from the `Keys.LLM` class. This value is essential for defining the architecture of the model, particularly in the context of neural network layers that utilize feed-forward mechanisms.
- **Use**: It is used to specify the feed-forward length in the configuration of language models.


---
### KEY\_USE\_PARALLEL\_RESIDUAL
- **Type**: `string`
- **Description**: The variable `KEY_USE_PARALLEL_RESIDUAL` is a constant that references a specific key in the `Keys.LLM` class, which is used to indicate whether parallel residual connections should be utilized in a model architecture. This key is likely used in configurations or settings related to the model's architecture to enable or disable this feature.
- **Use**: This variable is used to configure the model's architecture regarding the use of parallel residual connections.


---
### KEY\_TENSOR\_DATA\_LAYOUT
- **Type**: `string`
- **Description**: `KEY_TENSOR_DATA_LAYOUT` is a global constant that holds a key for the tensor data layout configuration in a language model. It is defined as a string that references a specific layout format within the `Keys.LLM` class, allowing for consistent access to this configuration across the codebase.
- **Use**: This variable is used to retrieve the tensor data layout key for model configuration.


---
### KEY\_ATTENTION\_HEAD\_COUNT
- **Type**: `string`
- **Description**: `KEY_ATTENTION_HEAD_COUNT` is a global constant that retrieves the number of attention heads used in a model's attention mechanism. It is defined as a reference to `Keys.Attention.HEAD_COUNT`, which is formatted to include the architecture identifier. This variable is crucial for configuring the attention layers of neural network models, particularly in transformer architectures.
- **Use**: This variable is used to specify the number of attention heads in the model's architecture.


---
### KEY\_ATTENTION\_HEAD\_COUNT\_KV
- **Type**: `string`
- **Description**: The variable `KEY_ATTENTION_HEAD_COUNT_KV` is a string constant that represents a key for accessing the number of key-value attention heads in a model architecture. It is defined within the `Keys.Attention` class, which organizes various constants related to attention mechanisms in neural networks.
- **Use**: This variable is used to retrieve or reference the number of key-value attention heads in configurations or metadata related to model architectures.


---
### KEY\_ATTENTION\_MAX\_ALIBI\_BIAS
- **Type**: `string`
- **Description**: `KEY_ATTENTION_MAX_ALIBI_BIAS` is a global constant that retrieves the maximum alibi bias value for attention mechanisms from the `Keys.Attention` class. This value is likely used in the context of attention models to control the bias applied during attention calculations, which can affect how the model processes input sequences.
- **Use**: This variable is used to set or reference the maximum alibi bias in attention calculations within the model.


---
### KEY\_ATTENTION\_CLAMP\_KQV
- **Type**: `string`
- **Description**: `KEY_ATTENTION_CLAMP_KQV` is a global constant that holds a key for the attention mechanism's KQV clamping parameter, defined within the `Keys.Attention` class. This key is used to access specific configuration settings related to the attention mechanism in a model architecture.
- **Use**: It is utilized to retrieve the clamping value for KQV in attention calculations.


---
### KEY\_ATTENTION\_LAYERNORM\_EPS
- **Type**: `string`
- **Description**: The variable `KEY_ATTENTION_LAYERNORM_EPS` is a constant that holds the epsilon value used in layer normalization for attention mechanisms. It is derived from the `LAYERNORM_EPS` attribute of the `Keys.Attention` class, which is likely used to prevent division by zero during normalization calculations.
- **Use**: This variable is used to configure the layer normalization process in attention models, ensuring numerical stability.


---
### KEY\_ATTENTION\_LAYERNORM\_RMS\_EPS
- **Type**: `string`
- **Description**: The variable `KEY_ATTENTION_LAYERNORM_RMS_EPS` is a constant that holds the value of the layer normalization RMS epsilon used in attention mechanisms. It is defined as a reference to `Keys.Attention.LAYERNORM_RMS_EPS`, which likely contains a specific numerical value for stability in layer normalization calculations.
- **Use**: This variable is used to ensure numerical stability during the layer normalization process in attention models.


---
### KEY\_ROPE\_DIMENSION\_COUNT
- **Type**: `str`
- **Description**: The variable `KEY_ROPE_DIMENSION_COUNT` is a constant that holds the key for accessing the dimension count of the RoPE (Rotary Positional Encoding) configuration in a model architecture. It is derived from the `Keys.Rope` class, specifically referencing the `DIMENSION_COUNT` attribute.
- **Use**: This variable is used to retrieve or set the dimension count for RoPE in model configurations.


---
### KEY\_ROPE\_FREQ\_BASE
- **Type**: `string`
- **Description**: The variable `KEY_ROPE_FREQ_BASE` is a constant that holds the frequency base value used in the RoPE (Rotary Positional Encoding) mechanism, which is essential for encoding positional information in transformer models. It is derived from the `Keys.Rope.FREQ_BASE` attribute, indicating its role in the configuration of the RoPE parameters.
- **Use**: This variable is used to configure the frequency base for the RoPE mechanism in transformer architectures.


---
### KEY\_ROPE\_SCALING\_TYPE
- **Type**: `string`
- **Description**: `KEY_ROPE_SCALING_TYPE` is a global constant that holds the scaling type for the Rope mechanism, defined within the `Keys.Rope` class. It is used to specify how the scaling of the Rope embeddings is applied, allowing for different scaling strategies such as linear or none.
- **Use**: This variable is used to configure the scaling behavior of the Rope embeddings in the model.


---
### KEY\_ROPE\_SCALING\_FACTOR
- **Type**: `string`
- **Description**: `KEY_ROPE_SCALING_FACTOR` is a global constant that retrieves the scaling factor for the Rope mechanism from the `Keys.Rope` class. This scaling factor is likely used in the context of adjusting the dimensions or performance of the Rope implementation in a model. It is essential for ensuring that the Rope's behavior aligns with the expected scaling properties during computations.
- **Use**: This variable is used to access the scaling factor for the Rope mechanism in the model's configuration.


---
### KEY\_ROPE\_SCALING\_ORIG\_CTX\_LEN
- **Type**: `string`
- **Description**: The variable `KEY_ROPE_SCALING_ORIG_CTX_LEN` is a constant that holds the key for the original context length used in the scaling of the RoPE (Rotary Positional Encoding) mechanism. It is derived from the `Keys.Rope.SCALING_ORIG_CTX_LEN`, which is part of a structured set of keys used for managing various parameters related to the RoPE functionality in a model architecture.
- **Use**: This variable is used to reference the original context length in configurations or metadata related to the RoPE scaling mechanism.


---
### KEY\_ROPE\_SCALING\_FINETUNED
- **Type**: `string`
- **Description**: The variable `KEY_ROPE_SCALING_FINETUNED` is a constant that holds a specific key for accessing the fine-tuned scaling configuration of the RoPE (Rotary Position Embedding) mechanism in a model. It is defined as `Keys.Rope.SCALING_FINETUNED`, which indicates that it is part of a structured set of keys related to the RoPE settings.
- **Use**: This variable is used to reference the fine-tuned scaling configuration in the context of model metadata or configuration.


---
### KEY\_SSM\_CONV\_KERNEL
- **Type**: `string`
- **Description**: The variable `KEY_SSM_CONV_KERNEL` is a string constant that represents the key for the convolution kernel in the state space model (SSM) configuration. It is defined as part of the `Keys.SSM` class, which organizes various keys related to the SSM architecture.
- **Use**: This variable is used to access or reference the convolution kernel parameter within the SSM configuration.


---
### KEY\_SSM\_INNER\_SIZE
- **Type**: `string`
- **Description**: The variable `KEY_SSM_INNER_SIZE` is a constant that holds the inner size configuration for the state space model (SSM) architecture, as defined in the `Keys.SSM` class. It is used to specify the dimensionality of the inner representations within the SSM framework.
- **Use**: This variable is utilized to configure the inner size parameter for SSM-related computations and model architectures.


---
### KEY\_SSM\_STATE\_SIZE
- **Type**: `string`
- **Description**: `KEY_SSM_STATE_SIZE` is a global constant that retrieves the state size for the SSM (State Space Model) from the `Keys.SSM` class. It is used to define the dimensionality of the state representation in the model, which is crucial for its performance and functionality.
- **Use**: This variable is used to access the state size configuration for the SSM in the model.


---
### KEY\_SSM\_TIME\_STEP\_RANK
- **Type**: `str`
- **Description**: The variable `KEY_SSM_TIME_STEP_RANK` is a string constant that holds a key for accessing the time step rank in the SSM (State Space Model) configuration. It is defined as a reference to `Keys.SSM.TIME_STEP_RANK`, which is part of a structured set of keys used for metadata management.
- **Use**: This variable is used to retrieve or reference the time step rank in SSM-related operations or configurations.


---
### KEY\_SSM\_DT\_B\_C\_RMS
- **Type**: `string`
- **Description**: The variable `KEY_SSM_DT_B_C_RMS` is a string constant that represents a specific metadata key related to the state-space model (SSM) architecture, specifically the 'dt_b_c_rms' parameter. It is defined as part of the `Keys.SSM` class, which organizes various keys used in the model's metadata.
- **Use**: This variable is used to access the 'dt_b_c_rms' key in the context of model metadata, facilitating the retrieval of relevant information during model operations.


---
### KEY\_TOKENIZER\_MODEL
- **Type**: `string`
- **Description**: `KEY_TOKENIZER_MODEL` is a global constant that holds the key for accessing the model attribute of the tokenizer within the `Keys.Tokenizer` class. It is defined as a string value 'tokenizer.ggml.model', which is used to identify the model configuration in the context of tokenization.
- **Use**: This variable is used to retrieve or reference the model information associated with the tokenizer in various parts of the code.


---
### KEY\_TOKENIZER\_PRE
- **Type**: `string`
- **Description**: `KEY_TOKENIZER_PRE` is a global constant that holds the string value representing the prefix key for the tokenizer in the context of the GGUF format. It is defined as `Keys.Tokenizer.PRE`, which is part of a larger set of keys used for managing tokenizer configurations and metadata.
- **Use**: This variable is used to reference the prefix key for the tokenizer when processing or configuring tokenization settings.


---
### KEY\_TOKENIZER\_LIST
- **Type**: `string`
- **Description**: `KEY_TOKENIZER_LIST` is a global constant that holds the key for accessing the list of tokens in the tokenizer configuration. It is defined as `Keys.Tokenizer.LIST`, which is a string that represents the specific key used to retrieve the token list from a tokenizer's settings.
- **Use**: This variable is used to reference the token list key when configuring or utilizing a tokenizer.


---
### KEY\_TOKENIZER\_TOKEN\_TYPE
- **Type**: `string`
- **Description**: The `KEY_TOKENIZER_TOKEN_TYPE` variable holds a string constant that represents the key for the token type in the tokenizer configuration. It is defined as part of the `Keys.Tokenizer` class, which organizes various constants related to tokenization in a structured manner.
- **Use**: This variable is used to access the token type key when configuring or utilizing the tokenizer in the model.


---
### KEY\_TOKENIZER\_SCORES
- **Type**: `string`
- **Description**: The `KEY_TOKENIZER_SCORES` variable is a constant that holds the key for accessing the scores associated with the tokenizer in the model. It is defined as part of the `Keys.Tokenizer` class, which organizes various constants related to tokenization, including model parameters and token types.
- **Use**: This variable is used to retrieve or reference the scores during the tokenization process in the model.


---
### KEY\_TOKENIZER\_MERGES
- **Type**: `string`
- **Description**: The `KEY_TOKENIZER_MERGES` variable holds the key for accessing the merges information in the tokenizer configuration, specifically defined as `Keys.Tokenizer.MERGES`. This key is essential for understanding how tokens are merged during the tokenization process, which is crucial for models that utilize subword tokenization techniques.
- **Use**: This variable is used to retrieve the merges configuration for the tokenizer, which influences how input text is processed into tokens.


---
### KEY\_TOKENIZER\_BOS\_ID
- **Type**: `string`
- **Description**: `KEY_TOKENIZER_BOS_ID` is a global constant that holds the beginning-of-sequence (BOS) token ID for a tokenizer, specifically defined within the `Keys.Tokenizer` class. This ID is used to indicate the start of a sequence in natural language processing tasks, such as text generation or model training.
- **Use**: This variable is used to reference the BOS token ID when initializing or processing sequences in a tokenizer.


---
### KEY\_TOKENIZER\_EOS\_ID
- **Type**: `string`
- **Description**: The `KEY_TOKENIZER_EOS_ID` variable holds the end-of-sequence token ID used in tokenization processes. It is derived from the `EOS_ID` attribute of the `Tokenizer` class within the `Keys` namespace, which defines various constants related to tokenization. This ID is crucial for indicating the end of a sequence in natural language processing tasks.
- **Use**: This variable is used to reference the token ID that signifies the end of a sequence during tokenization.


---
### KEY\_TOKENIZER\_EOT\_ID
- **Type**: `string`
- **Description**: `KEY_TOKENIZER_EOT_ID` is a global constant that holds the end-of-text token identifier for the tokenizer, sourced from the `Keys.Tokenizer` class. This identifier is crucial for indicating the termination of text input during processing in natural language models.
- **Use**: It is used to signify the end of a text sequence in tokenization processes.


---
### KEY\_TOKENIZER\_EOM\_ID
- **Type**: `str`
- **Description**: The variable `KEY_TOKENIZER_EOM_ID` is a string constant that represents the end-of-message (EOM) token identifier used in the tokenizer. It is defined as part of the `Keys.Tokenizer` class, which organizes various token-related constants for a tokenizer implementation.
- **Use**: This variable is used to reference the specific token ID for the end-of-message token in tokenization processes.


---
### KEY\_TOKENIZER\_UNK\_ID
- **Type**: `string`
- **Description**: The `KEY_TOKENIZER_UNK_ID` variable holds the identifier for the unknown token in the tokenizer, which is used to represent tokens that are not recognized or do not exist in the vocabulary. This is crucial for handling out-of-vocabulary words during text processing.
- **Use**: This variable is used to reference the unknown token ID when tokenizing input text.


---
### KEY\_TOKENIZER\_SEP\_ID
- **Type**: `string`
- **Description**: The `KEY_TOKENIZER_SEP_ID` variable holds the identifier for the separator token used in the tokenizer, which is essential for distinguishing between different segments of input text during processing. It is derived from the `Keys.Tokenizer.SEP_ID` constant, ensuring consistency across the tokenizer's implementation.
- **Use**: This variable is used to reference the separator token ID in various tokenization processes.


---
### KEY\_TOKENIZER\_PAD\_ID
- **Type**: `string`
- **Description**: `KEY_TOKENIZER_PAD_ID` is a global constant that holds the padding token ID used in tokenization processes. It is derived from the `Keys.Tokenizer.PAD_ID`, which is part of a class that defines various constants related to tokenization. This ID is essential for ensuring that sequences of varying lengths can be processed uniformly by padding them to a consistent length.
- **Use**: This variable is used to identify the padding token in tokenization tasks, allowing models to handle input sequences of different lengths.


---
### KEY\_TOKENIZER\_MASK\_ID
- **Type**: `string`
- **Description**: `KEY_TOKENIZER_MASK_ID` is a global constant that holds the key for the mask token ID used in tokenization processes. It is defined as `Keys.Tokenizer.MASK_ID`, which is part of a class that organizes various keys related to tokenization. This key is essential for identifying the mask token in models that utilize masked language modeling.
- **Use**: This variable is used to reference the mask token ID in tokenization operations.


---
### KEY\_TOKENIZER\_HF\_JSON
- **Type**: `string`
- **Description**: `KEY_TOKENIZER_HF_JSON` is a global constant that holds the key for the Hugging Face JSON tokenizer configuration. It is defined as a string constant within the `Keys.Tokenizer` class, specifically representing the path to the Hugging Face tokenizer JSON file.
- **Use**: This variable is used to reference the Hugging Face tokenizer configuration in various parts of the code.


---
### KEY\_TOKENIZER\_RWKV
- **Type**: `string`
- **Description**: The variable `KEY_TOKENIZER_RWKV` is a constant that holds a reference to the RWKV tokenizer key defined within the `Keys.Tokenizer` class. This key is used to identify the RWKV tokenizer model in various contexts, such as configuration or metadata management.
- **Use**: This variable is used to access the RWKV tokenizer key for model configuration and processing tasks.


---
### KEY\_TOKENIZER\_FIM\_PRE\_ID
- **Type**: `string`
- **Description**: `KEY_TOKENIZER_FIM_PRE_ID` is a global constant that holds the key identifier for the prefix token used in the tokenizer, specifically for fine-tuning or infilling tasks. It is derived from the `Keys.Tokenizer` class, which organizes various token-related constants for model processing.
- **Use**: This variable is used to reference the prefix token ID in tokenizer operations.


---
### KEY\_TOKENIZER\_FIM\_SUF\_ID
- **Type**: `string`
- **Description**: `KEY_TOKENIZER_FIM_SUF_ID` is a global constant that holds the key for the suffix identifier used in the tokenizer, specifically defined as `Keys.Tokenizer.FIM_SUF_ID`. This key is part of a set of constants that facilitate the tokenization process in natural language processing tasks.
- **Use**: This variable is used to reference the suffix token ID in the tokenizer's configuration.


---
### KEY\_TOKENIZER\_FIM\_MID\_ID
- **Type**: `string`
- **Description**: The variable `KEY_TOKENIZER_FIM_MID_ID` is a constant that holds the identifier for the middle token used in the Fine-tuning Infill Model (FIM) within the tokenizer. It is derived from the `Keys.Tokenizer` class, specifically referencing the `FIM_MID_ID` attribute, which is essential for tokenization processes in natural language processing tasks.
- **Use**: This variable is used to access the specific token ID for the middle token in the FIM tokenizer, facilitating the correct tokenization of input data.


---
### KEY\_TOKENIZER\_FIM\_PAD\_ID
- **Type**: `string`
- **Description**: The `KEY_TOKENIZER_FIM_PAD_ID` variable is a global constant that holds the padding token ID used in the tokenizer for FIM (Fine-tuning Inference Model) operations. It is derived from the `Keys.Tokenizer` class, specifically referencing the `FIM_PAD_ID` attribute, which is essential for managing padding in sequences during tokenization.
- **Use**: This variable is used to identify the padding token in the tokenizer, ensuring that sequences are properly padded to the required length during processing.


---
### KEY\_TOKENIZER\_FIM\_REP\_ID
- **Type**: `string`
- **Description**: The variable `KEY_TOKENIZER_FIM_REP_ID` is a constant that holds the key for the 'finetune representation ID' used in the tokenizer. It is derived from the `Keys.Tokenizer` class, specifically referencing the `FIM_REP_ID` attribute, which is likely used to identify a specific token or representation in the context of fine-tuning models.
- **Use**: This variable is used to access the finetune representation ID within the tokenizer's configuration.


---
### KEY\_TOKENIZER\_FIM\_SEP\_ID
- **Type**: `string`
- **Description**: `KEY_TOKENIZER_FIM_SEP_ID` is a constant that holds the key for the FIM separator token ID used in the tokenizer. It is defined as `Keys.Tokenizer.FIM_SEP_ID`, which is a string representing the specific token ID for the separator in the tokenizer's vocabulary.
- **Use**: This variable is used to reference the separator token ID in the tokenizer's operations.


---
### KEY\_TOKENIZER\_PREFIX\_ID
- **Type**: `string`
- **Description**: `KEY_TOKENIZER_PREFIX_ID` is a global constant that holds the key for the prefix token ID used in the tokenizer. It is derived from the `Keys.Tokenizer` class, specifically referencing the `PREFIX_ID` attribute, which is a string identifier for the prefix token in the tokenization process.
- **Use**: This variable is used to access the prefix token ID in the context of tokenization.


---
### KEY\_TOKENIZER\_SUFFIX\_ID
- **Type**: `string`
- **Description**: The variable `KEY_TOKENIZER_SUFFIX_ID` is a constant that holds the key for the suffix token ID used in the tokenizer configuration. It is derived from the `Keys.Tokenizer` class, specifically referencing the `SUFFIX_ID` attribute, which is essential for identifying the suffix token in tokenization processes.
- **Use**: This variable is used to access the suffix token ID within the tokenizer's configuration, facilitating the handling of token sequences.


---
### KEY\_TOKENIZER\_MIDDLE\_ID
- **Type**: `string`
- **Description**: The variable `KEY_TOKENIZER_MIDDLE_ID` is a constant that holds the key for the middle token ID in the tokenizer configuration. It is derived from the `Keys.Tokenizer` class, specifically referencing the `MIDDLE_ID` attribute, which is used to identify the middle token in a sequence during tokenization.
- **Use**: This variable is used to access the middle token ID within the tokenizer's settings, facilitating the processing of input sequences.


# Classes

---
### Keys<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys}} -->
- **Description**: The `Keys` class serves as a comprehensive container for various metadata keys used in machine learning models, particularly those related to general model information, large language models (LLM), attention mechanisms, rope scaling, tokenization, and more. It is organized into several nested classes, each representing a specific category of metadata, such as `General`, `LLM`, `Attention`, `Rope`, `Tokenizer`, and others. Each nested class contains a set of string constants that define the keys used to access specific metadata attributes, facilitating structured and consistent access to model metadata across different components of a machine learning framework.


---
### General<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.General}} -->
- **Members**:
    - `TYPE`: Represents the general type of the model.
    - `ARCHITECTURE`: Specifies the architecture of the model.
    - `QUANTIZATION_VERSION`: Indicates the version of quantization used.
    - `ALIGNMENT`: Defines the alignment property of the model.
    - `FILE_TYPE`: Specifies the file type of the model.
    - `NAME`: Holds the name of the model.
    - `AUTHOR`: Contains the author's name of the model.
    - `VERSION`: Indicates the version of the model.
    - `ORGANIZATION`: Specifies the organization associated with the model.
    - `FINETUNE`: Indicates if the model is fine-tuned.
    - `BASENAME`: Represents the base name of the model.
    - `DESCRIPTION`: Provides a description of the model.
    - `QUANTIZED_BY`: Specifies who quantized the model.
    - `SIZE_LABEL`: Indicates the size label of the model.
    - `LICENSE`: Contains the license information of the model.
    - `LICENSE_NAME`: Specifies the name of the license.
    - `LICENSE_LINK`: Provides a link to the license.
    - `URL`: Contains the URL to the model's website or paper.
    - `DOI`: Holds the DOI of the model.
    - `UUID`: Specifies the UUID of the model.
    - `REPO_URL`: Contains the URL to the model's source repository.
    - `SOURCE_URL`: Provides the URL to the model's source during conversion.
    - `SOURCE_DOI`: Holds the DOI of the model's source.
    - `SOURCE_UUID`: Specifies the UUID of the model's source.
    - `SOURCE_REPO_URL`: Contains the URL to the model's source repository during conversion.
    - `BASE_MODEL_COUNT`: Indicates the count of base models if merged.
    - `BASE_MODEL_NAME`: Specifies the name of a base model.
    - `BASE_MODEL_AUTHOR`: Contains the author's name of a base model.
    - `BASE_MODEL_VERSION`: Indicates the version of a base model.
    - `BASE_MODEL_ORGANIZATION`: Specifies the organization associated with a base model.
    - `BASE_MODEL_DESCRIPTION`: Provides a description of a base model.
    - `BASE_MODEL_URL`: Contains the URL to a base model's website or paper.
    - `BASE_MODEL_DOI`: Holds the DOI of a base model.
    - `BASE_MODEL_UUID`: Specifies the UUID of a base model.
    - `BASE_MODEL_REPO_URL`: Contains the URL to a base model's source repository.
    - `DATASET_COUNT`: Indicates the count of datasets used.
    - `DATASET_NAME`: Specifies the name of a dataset.
    - `DATASET_AUTHOR`: Contains the author's name of a dataset.
    - `DATASET_VERSION`: Indicates the version of a dataset.
    - `DATASET_ORGANIZATION`: Specifies the organization associated with a dataset.
    - `DATASET_DESCRIPTION`: Provides a description of a dataset.
    - `DATASET_URL`: Contains the URL to a dataset's website or paper.
    - `DATASET_DOI`: Holds the DOI of a dataset.
    - `DATASET_UUID`: Specifies the UUID of a dataset.
    - `DATASET_REPO_URL`: Contains the URL to a dataset's source repository.
    - `TAGS`: Stores tags associated with the model.
    - `LANGUAGES`: Lists the languages supported by the model.
- **Description**: The `General` class serves as a comprehensive metadata container for models, encapsulating various attributes such as type, architecture, quantization version, and alignment. It includes authorship metadata like name, author, version, and organization, as well as licensing details and URLs for model and source repositories. The class also supports tracking of base models and datasets, providing fields for their names, authors, versions, and descriptions. Additionally, it includes fields for tags and languages, making it a versatile structure for managing model metadata in a structured and detailed manner.


---
### LLM<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.LLM}} -->
- **Members**:
    - `VOCAB_SIZE`: Represents the vocabulary size of the architecture.
    - `CONTEXT_LENGTH`: Indicates the context length for the architecture.
    - `EMBEDDING_LENGTH`: Specifies the embedding length for the architecture.
    - `FEATURES_LENGTH`: Defines the features length for the architecture.
    - `BLOCK_COUNT`: Denotes the number of blocks in the architecture.
    - `LEADING_DENSE_BLOCK_COUNT`: Indicates the count of leading dense blocks in the architecture.
    - `FEED_FORWARD_LENGTH`: Specifies the length of the feed-forward network in the architecture.
    - `EXPERT_FEED_FORWARD_LENGTH`: Defines the length of the expert feed-forward network.
    - `EXPERT_SHARED_FEED_FORWARD_LENGTH`: Specifies the length of the shared expert feed-forward network.
    - `USE_PARALLEL_RESIDUAL`: Indicates whether parallel residual connections are used.
    - `TENSOR_DATA_LAYOUT`: Specifies the data layout for tensors in the architecture.
    - `EXPERT_COUNT`: Denotes the total number of experts in the architecture.
    - `EXPERT_USED_COUNT`: Indicates the number of experts actively used.
    - `EXPERT_SHARED_COUNT`: Specifies the count of shared experts.
    - `EXPERT_WEIGHTS_SCALE`: Defines the scale for expert weights.
    - `EXPERT_WEIGHTS_NORM`: Indicates the normalization applied to expert weights.
    - `EXPERT_GATING_FUNC`: Specifies the gating function used for experts.
    - `MOE_EVERY_N_LAYERS`: Indicates the frequency of mixture of experts layers.
    - `POOLING_TYPE`: Defines the type of pooling used in the architecture.
    - `LOGIT_SCALE`: Specifies the scale applied to logits.
    - `DECODER_START_TOKEN_ID`: Indicates the start token ID for the decoder.
    - `ATTN_LOGIT_SOFTCAPPING`: Specifies the softcapping applied to attention logits.
    - `FINAL_LOGIT_SOFTCAPPING`: Indicates the softcapping applied to final logits.
    - `SWIN_NORM`: Defines the normalization used in SWIN layers.
    - `RESCALE_EVERY_N_LAYERS`: Specifies the frequency of rescaling across layers.
    - `TIME_MIX_EXTRA_DIM`: Indicates the extra dimension for time mixing.
    - `TIME_DECAY_EXTRA_DIM`: Specifies the extra dimension for time decay.
    - `RESIDUAL_SCALE`: Defines the scale applied to residual connections.
    - `EMBEDDING_SCALE`: Specifies the scale applied to embeddings.
    - `TOKEN_SHIFT_COUNT`: Indicates the count of token shifts applied.
    - `INTERLEAVE_MOE_LAYER_STEP`: Specifies the step for interleaving mixture of experts layers.
- **Description**: The LLM class encapsulates a comprehensive set of architectural parameters for a language model, including vocabulary size, context length, embedding dimensions, and various expert configurations. It defines numerous constants that represent different aspects of the model's structure, such as the number of blocks, feed-forward lengths, and expert-related settings. This class is crucial for configuring and understanding the architecture of a language model, providing detailed specifications for each component involved in the model's operation.


---
### Attention<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.ClipAudio.Attention}} -->
- **Members**:
    - `HEAD_COUNT`: Represents the number of attention heads in the audio attention mechanism.
    - `LAYERNORM_EPS`: Specifies the epsilon value used for layer normalization in the audio attention mechanism.
- **Description**: The `Attention` class is a simple data structure that defines constants related to the audio attention mechanism in the CLIP model, specifically focusing on the number of attention heads and the epsilon value for layer normalization.


---
### Rope<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.Rope}} -->
- **Members**:
    - `DIMENSION_COUNT`: Represents the key for the dimension count of the rope.
    - `DIMENSION_SECTIONS`: Represents the key for the dimension sections of the rope.
    - `FREQ_BASE`: Represents the key for the frequency base of the rope.
    - `SCALING_TYPE`: Represents the key for the scaling type of the rope.
    - `SCALING_FACTOR`: Represents the key for the scaling factor of the rope.
    - `SCALING_ATTN_FACTOR`: Represents the key for the scaling attention factor of the rope.
    - `SCALING_ORIG_CTX_LEN`: Represents the key for the original context length in scaling of the rope.
    - `SCALING_FINETUNED`: Represents the key for the finetuned scaling of the rope.
    - `SCALING_YARN_LOG_MUL`: Represents the key for the yarn log multiplier in scaling of the rope.
- **Description**: The `Rope` class is a collection of constants that define various configuration keys related to the rope architecture in a model. These keys are used to specify different parameters such as dimension count, frequency base, and various scaling factors, which are essential for configuring the rope's behavior in the model's architecture.


---
### Split<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.Split}} -->
- **Members**:
    - `LLM_KV_SPLIT_NO`: Represents a key for no split in the LLM key-value store.
    - `LLM_KV_SPLIT_COUNT`: Represents a key for the count of splits in the LLM key-value store.
    - `LLM_KV_SPLIT_TENSORS_COUNT`: Represents a key for the count of split tensors in the LLM key-value store.
- **Description**: The `Split` class defines constants used as keys in a key-value store related to splits in a language model (LLM) context. These constants are likely used to manage and reference different aspects of data splitting, such as the number of splits and the number of tensors involved in the splits.


---
### SSM<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.SSM}} -->
- **Members**:
    - `CONV_KERNEL`: A string template for the convolution kernel key in the SSM architecture.
    - `INNER_SIZE`: A string template for the inner size key in the SSM architecture.
    - `STATE_SIZE`: A string template for the state size key in the SSM architecture.
    - `TIME_STEP_RANK`: A string template for the time step rank key in the SSM architecture.
    - `DT_B_C_RMS`: A string template for the dt_b_c_rms key in the SSM architecture.
- **Description**: The SSM class defines a set of string templates for various keys related to the SSM (State Space Model) architecture. These keys are used to specify different parameters such as convolution kernel, inner size, state size, time step rank, and dt_b_c_rms within the architecture, allowing for dynamic configuration based on the architecture type.


---
### WKV<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.WKV}} -->
- **Members**:
    - `HEAD_SIZE`: A string template for the head size of the WKV architecture.
- **Description**: The `WKV` class is a simple container for a single class variable, `HEAD_SIZE`, which is a string template used to define the head size for the WKV architecture. This class is part of a larger set of classes that define various architectural and metadata keys for different components of a machine learning model.


---
### PosNet<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.PosNet}} -->
- **Members**:
    - `EMBEDDING_LENGTH`: A string template for the embedding length specific to the PosNet architecture.
    - `BLOCK_COUNT`: A string template for the block count specific to the PosNet architecture.
- **Description**: The PosNet class defines constants related to the PosNet architecture, specifically providing string templates for embedding length and block count, which are used to configure or identify these parameters within the architecture.


---
### ConvNext<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.ConvNext}} -->
- **Members**:
    - `EMBEDDING_LENGTH`: A string template for the embedding length of the ConvNext architecture.
    - `BLOCK_COUNT`: A string template for the block count of the ConvNext architecture.
- **Description**: The ConvNext class defines constants related to the ConvNext architecture, specifically providing string templates for embedding length and block count, which are likely used for configuring or identifying specific architectural parameters in a larger system.


---
### Classifier<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.Classifier}} -->
- **Members**:
    - `OUTPUT_LABELS`: A class variable that stores the format string for output labels specific to the classifier architecture.
- **Description**: The `Classifier` class is a simple container for a single class variable, `OUTPUT_LABELS`, which holds a format string used to define output labels for a classifier architecture. This class is part of a larger framework that likely deals with various model architectures and their associated metadata.


---
### Tokenizer<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.Tokenizer}} -->
- **Members**:
    - `MODEL`: Represents the model file for the tokenizer.
    - `PRE`: Indicates the pre-tokenization configuration.
    - `LIST`: Contains the list of tokens used by the tokenizer.
    - `TOKEN_TYPE`: Defines the type of tokens used in the tokenizer.
    - `TOKEN_TYPE_COUNT`: Specifies the count of token types, useful for BERT-style token types.
    - `SCORES`: Holds the scores associated with tokens.
    - `MERGES`: Contains merge rules for byte pair encoding.
    - `BOS_ID`: ID for the beginning-of-sequence token.
    - `EOS_ID`: ID for the end-of-sequence token.
    - `EOT_ID`: ID for the end-of-text token.
    - `EOM_ID`: ID for the end-of-message token.
    - `UNK_ID`: ID for the unknown token.
    - `SEP_ID`: ID for the separator token.
    - `PAD_ID`: ID for the padding token.
    - `MASK_ID`: ID for the mask token.
    - `ADD_BOS`: Flag to add a beginning-of-sequence token.
    - `ADD_EOS`: Flag to add an end-of-sequence token.
    - `ADD_PREFIX`: Flag to add a space prefix to tokens.
    - `REMOVE_EXTRA_WS`: Flag to remove extra whitespaces during tokenization.
    - `PRECOMPILED_CHARSMAP`: Contains a precompiled character map for tokenization.
    - `HF_JSON`: Path to the Hugging Face JSON configuration for the tokenizer.
    - `RWKV`: Configuration related to the RWKV world model.
    - `CHAT_TEMPLATE`: Template for chat-based tokenization.
    - `CHAT_TEMPLATE_N`: Named template for chat-based tokenization.
    - `CHAT_TEMPLATES`: Collection of chat templates for tokenization.
    - `FIM_PRE_ID`: ID for the FIM pre-token.
    - `FIM_SUF_ID`: ID for the FIM suffix token.
    - `FIM_MID_ID`: ID for the FIM middle token.
    - `FIM_PAD_ID`: ID for the FIM padding token.
    - `FIM_REP_ID`: ID for the FIM replacement token.
    - `FIM_SEP_ID`: ID for the FIM separator token.
    - `PREFIX_ID`: Deprecated ID for the prefix token.
    - `SUFFIX_ID`: Deprecated ID for the suffix token.
    - `MIDDLE_ID`: Deprecated ID for the middle token.
- **Description**: The `Tokenizer` class is a comprehensive configuration holder for various tokenization parameters and constants used in natural language processing models. It includes identifiers for special tokens like beginning-of-sequence, end-of-sequence, and unknown tokens, as well as configurations for handling token types, scores, and merges. Additionally, it supports chat templates and FIM (Fill-In-the-Middle) special tokens, providing a robust framework for managing tokenization in different contexts. The class also includes deprecated identifiers for backward compatibility.


---
### Adapter<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.Adapter}} -->
- **Members**:
    - `TYPE`: A string constant representing the type of the adapter.
    - `LORA_ALPHA`: A string constant representing the alpha parameter for LoRA (Low-Rank Adaptation).
- **Description**: The `Adapter` class is a simple container for constants related to adapter configurations, specifically defining the type of adapter and a parameter for LoRA (Low-Rank Adaptation). It serves as a centralized location for these constants, which can be used throughout the codebase to ensure consistency and avoid hardcoding these values in multiple places.


---
### Clip<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.Clip}} -->
- **Members**:
    - `PROJECTOR_TYPE`: A string constant representing the projector type for the Clip class.
    - `HAS_VISION_ENCODER`: A string constant indicating if the Clip class has a vision encoder.
    - `HAS_AUDIO_ENCODER`: A string constant indicating if the Clip class has an audio encoder.
    - `HAS_LLAVA_PROJECTOR`: A string constant indicating if the Clip class has a LLAVA projector.
- **Description**: The Clip class is a simple container for string constants that define various attributes related to the Clip architecture, such as the type of projector and the presence of vision and audio encoders, as well as a LLAVA projector. These constants are likely used as keys or identifiers in a larger system dealing with multimedia processing or machine learning models.


---
### ClipVision<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.ClipVision}} -->
- **Members**:
    - `IMAGE_SIZE`: Represents the image size for the vision model.
    - `PATCH_SIZE`: Defines the patch size used in the vision model.
    - `EMBEDDING_LENGTH`: Specifies the length of the embedding vector for the vision model.
    - `FEED_FORWARD_LENGTH`: Indicates the length of the feed-forward network in the vision model.
    - `PROJECTION_DIM`: Denotes the dimension of the projection layer in the vision model.
    - `BLOCK_COUNT`: Represents the number of blocks in the vision model.
    - `IMAGE_MEAN`: Specifies the mean value used for image normalization.
    - `IMAGE_STD`: Specifies the standard deviation used for image normalization.
    - `SPATIAL_MERGE_SIZE`: Defines the size for spatial merging in the vision model.
    - `USE_GELU`: Indicates whether the GELU activation function is used.
    - `USE_SILU`: Indicates whether the SiLU activation function is used.
    - `N_WA_PATTERN`: Used by qwen2.5vl, represents a specific pattern in the vision model.
- **Description**: The `ClipVision` class encapsulates various configuration parameters and constants for a vision model within the CLIP framework. It includes specifications for image processing such as image size, patch size, and normalization parameters, as well as architectural details like embedding length, feed-forward network length, and block count. The class also contains nested classes for `Attention` and `Projector`, which define specific parameters related to attention mechanisms and projection scaling, respectively. This class serves as a centralized repository for vision model settings, facilitating easy access and modification of these parameters.


---
### Projector<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.ClipAudio.Projector}} -->
- **Members**:
    - `STACK_FACTOR`: A constant string representing the stack factor for the audio projector in the clip module.
- **Description**: The `Projector` class is a simple container for a constant string, `STACK_FACTOR`, which is used to represent a specific configuration or parameter related to the audio projector within the clip module. This class serves as a namespace for this constant, ensuring that it is easily accessible and organized within the codebase.


---
### ClipAudio<!-- {{#class:llama.cpp/gguf-py/gguf/constants.Keys.ClipAudio}} -->
- **Members**:
    - `NUM_MEL_BINS`: Specifies the number of mel bins for audio processing.
    - `EMBEDDING_LENGTH`: Defines the length of the audio embedding.
    - `FEED_FORWARD_LENGTH`: Indicates the length of the feed-forward network for audio.
    - `PROJECTION_DIM`: Specifies the dimension of the audio projection.
    - `BLOCK_COUNT`: Denotes the number of blocks in the audio processing pipeline.
- **Description**: The `ClipAudio` class is designed to encapsulate various configuration parameters related to audio processing within a CLIP model. It includes constants for defining the number of mel bins, embedding length, feed-forward network length, projection dimension, and block count, which are essential for configuring the audio processing pipeline. Additionally, it contains nested classes for `Attention` and `Projector`, which further specify parameters related to attention mechanisms and projection stacking in the audio context.


---
### GGUFType<!-- {{#class:llama.cpp/gguf-py/gguf/constants.GGUFType}} -->
- **Members**:
    - `MODEL`: Represents the type 'model'.
    - `ADAPTER`: Represents the type 'adapter'.
    - `MMPROJ`: Represents the type 'mmproj', currently unused.
- **Description**: The GGUFType class defines a set of string constants that represent different types of models or components, such as 'model', 'adapter', and 'mmproj'. These constants are likely used as identifiers or keys within a larger system to categorize or specify the type of a model or component. The 'mmproj' type is noted as a dummy and is currently unused.


---
### MODEL\_ARCH<!-- {{#class:llama.cpp/gguf-py/gguf/constants.MODEL_ARCH}} -->
- **Description**: The `MODEL_ARCH` class is an enumeration that defines a comprehensive list of model architectures, each represented as an enumerated constant. This class is used to categorize and identify different model architectures, such as LLAMA, GPT2, BERT, and many others, by assigning them unique integer values automatically. It serves as a centralized reference for model architecture types within the software, facilitating the management and identification of various models in a structured manner.
- **Inherits From**:
    - `IntEnum`


---
### VISION\_PROJECTOR\_TYPE<!-- {{#class:llama.cpp/gguf-py/gguf/constants.VISION_PROJECTOR_TYPE}} -->
- **Members**:
    - `MLP`: Represents a Multi-Layer Perceptron projector type.
    - `LDP`: Represents a Linear Discriminant Projection type.
    - `LDPV2`: Represents a second version of Linear Discriminant Projection.
    - `RESAMPLER`: Represents a resampler projector type.
    - `GLM_EDGE`: Represents a GLM edge projector type.
    - `MERGER`: Represents a merger projector type.
    - `GEMMA3`: Represents a GEMMA3 projector type.
- **Description**: The VISION_PROJECTOR_TYPE class is an enumeration that defines various types of vision projectors used in machine learning models. Each member of the enumeration represents a specific type of projector, such as MLP, LDP, and RESAMPLER, which are used to transform or project data in different ways within vision-related tasks. This class is part of a larger framework for handling model architectures and their components, providing a standardized way to refer to these projector types.
- **Inherits From**:
    - `IntEnum`


---
### MODEL\_TENSOR<!-- {{#class:llama.cpp/gguf-py/gguf/constants.MODEL_TENSOR}} -->
- **Decorators**: `@IntEnum`
- **Description**: The `MODEL_TENSOR` class is an enumeration that defines a comprehensive list of tensor types used in various model architectures. Each member of this enumeration represents a specific tensor type, such as embeddings, attention mechanisms, feed-forward networks, and other components relevant to model processing in machine learning frameworks. This enumeration facilitates the standardized identification and handling of different tensor types across diverse model implementations.
- **Inherits From**:
    - `IntEnum`


---
### TokenType<!-- {{#class:llama.cpp/gguf-py/gguf/constants.TokenType}} -->
- **Decorators**: `@IntEnum`
- **Members**:
    - `NORMAL`: Represents a normal token type with a value of 1.
    - `UNKNOWN`: Represents an unknown token type with a value of 2.
    - `CONTROL`: Represents a control token type with a value of 3.
    - `USER_DEFINED`: Represents a user-defined token type with a value of 4.
    - `UNUSED`: Represents an unused token type with a value of 5.
    - `BYTE`: Represents a byte token type with a value of 6.
- **Description**: The TokenType class is an enumeration that defines various types of tokens, each associated with a unique integer value. It is used to categorize tokens into different types such as normal, unknown, control, user-defined, unused, and byte, facilitating the handling and processing of tokens in a structured manner.
- **Inherits From**:
    - `IntEnum`


---
### RopeScalingType<!-- {{#class:llama.cpp/gguf-py/gguf/constants.RopeScalingType}} -->
- **Description**: The `RopeScalingType` class is an enumeration that defines different types of rope scaling methods, including 'none', 'linear', 'yarn', and 'longrope'. This class is used to categorize and manage different scaling strategies for ropes, likely in a context where rope properties or behaviors need to be adjusted or simulated based on these types.
- **Inherits From**:
    - `Enum`


---
### PoolingType<!-- {{#class:llama.cpp/gguf-py/gguf/constants.PoolingType}} -->
- **Members**:
    - `NONE`: Represents no pooling with a value of 0.
    - `MEAN`: Represents mean pooling with a value of 1.
    - `CLS`: Represents CLS token pooling with a value of 2.
    - `LAST`: Represents last token pooling with a value of 3.
    - `RANK`: Represents rank pooling with a value of 4.
- **Description**: The `PoolingType` class is an enumeration that defines different types of pooling strategies used in neural network architectures. Each member of the enumeration corresponds to a specific pooling method, such as NONE, MEAN, CLS, LAST, and RANK, each associated with a unique integer value. This class is useful for selecting and applying different pooling techniques in model configurations.
- **Inherits From**:
    - `IntEnum`


---
### GGMLQuantizationType<!-- {{#class:llama.cpp/gguf-py/gguf/constants.GGMLQuantizationType}} -->
- **Description**: The `GGMLQuantizationType` class is an enumeration that defines various quantization types for use in machine learning models. Each member of the enumeration represents a specific quantization format, such as floating-point (F32, F16, F64), integer (I8, I16, I32, I64), and various quantized formats (Q4_0, Q4_1, Q5_0, etc.). These quantization types are used to optimize model storage and computation by reducing the precision of the data, which can lead to faster processing and reduced memory usage.
- **Inherits From**:
    - `IntEnum`


---
### ExpertGatingFuncType<!-- {{#class:llama.cpp/gguf-py/gguf/constants.ExpertGatingFuncType}} -->
- **Decorators**: `@IntEnum`
- **Members**:
    - `SOFTMAX`: Represents the softmax gating function with a value of 1.
    - `SIGMOID`: Represents the sigmoid gating function with a value of 2.
- **Description**: The `ExpertGatingFuncType` class is an enumeration that defines two types of expert gating functions, namely SOFTMAX and SIGMOID, each associated with a unique integer value. This class is used to specify the type of gating function to be applied in a model, providing a clear and standardized way to reference these functions within the code.
- **Inherits From**:
    - `IntEnum`


---
### LlamaFileType<!-- {{#class:llama.cpp/gguf-py/gguf/constants.LlamaFileType}} -->
- **Members**:
    - `ALL_F32`: Represents a file type where all tensors are in 32-bit floating point format.
    - `MOSTLY_F16`: Represents a file type where most tensors are in 16-bit floating point format, except 1D tensors.
    - `MOSTLY_Q4_0`: Represents a file type where most tensors are in Q4_0 quantization format, except 1D tensors.
    - `MOSTLY_Q4_1`: Represents a file type where most tensors are in Q4_1 quantization format, except 1D tensors.
    - `MOSTLY_Q8_0`: Represents a file type where most tensors are in Q8_0 quantization format, except 1D tensors.
    - `MOSTLY_Q5_0`: Represents a file type where most tensors are in Q5_0 quantization format, except 1D tensors.
    - `MOSTLY_Q5_1`: Represents a file type where most tensors are in Q5_1 quantization format, except 1D tensors.
    - `MOSTLY_Q2_K`: Represents a file type where most tensors are in Q2_K quantization format, except 1D tensors.
    - `MOSTLY_Q3_K_S`: Represents a file type where most tensors are in Q3_K_S quantization format, except 1D tensors.
    - `MOSTLY_Q3_K_M`: Represents a file type where most tensors are in Q3_K_M quantization format, except 1D tensors.
    - `MOSTLY_Q3_K_L`: Represents a file type where most tensors are in Q3_K_L quantization format, except 1D tensors.
    - `MOSTLY_Q4_K_S`: Represents a file type where most tensors are in Q4_K_S quantization format, except 1D tensors.
    - `MOSTLY_Q4_K_M`: Represents a file type where most tensors are in Q4_K_M quantization format, except 1D tensors.
    - `MOSTLY_Q5_K_S`: Represents a file type where most tensors are in Q5_K_S quantization format, except 1D tensors.
    - `MOSTLY_Q5_K_M`: Represents a file type where most tensors are in Q5_K_M quantization format, except 1D tensors.
    - `MOSTLY_Q6_K`: Represents a file type where most tensors are in Q6_K quantization format, except 1D tensors.
    - `MOSTLY_IQ2_XXS`: Represents a file type where most tensors are in IQ2_XXS quantization format, except 1D tensors.
    - `MOSTLY_IQ2_XS`: Represents a file type where most tensors are in IQ2_XS quantization format, except 1D tensors.
    - `MOSTLY_Q2_K_S`: Represents a file type where most tensors are in Q2_K_S quantization format, except 1D tensors.
    - `MOSTLY_IQ3_XS`: Represents a file type where most tensors are in IQ3_XS quantization format, except 1D tensors.
    - `MOSTLY_IQ3_XXS`: Represents a file type where most tensors are in IQ3_XXS quantization format, except 1D tensors.
    - `MOSTLY_IQ1_S`: Represents a file type where most tensors are in IQ1_S quantization format, except 1D tensors.
    - `MOSTLY_IQ4_NL`: Represents a file type where most tensors are in IQ4_NL quantization format, except 1D tensors.
    - `MOSTLY_IQ3_S`: Represents a file type where most tensors are in IQ3_S quantization format, except 1D tensors.
    - `MOSTLY_IQ3_M`: Represents a file type where most tensors are in IQ3_M quantization format, except 1D tensors.
    - `MOSTLY_IQ2_S`: Represents a file type where most tensors are in IQ2_S quantization format, except 1D tensors.
    - `MOSTLY_IQ2_M`: Represents a file type where most tensors are in IQ2_M quantization format, except 1D tensors.
    - `MOSTLY_IQ4_XS`: Represents a file type where most tensors are in IQ4_XS quantization format, except 1D tensors.
    - `MOSTLY_IQ1_M`: Represents a file type where most tensors are in IQ1_M quantization format, except 1D tensors.
    - `MOSTLY_BF16`: Represents a file type where most tensors are in BF16 format, except 1D tensors.
    - `MOSTLY_TQ1_0`: Represents a file type where most tensors are in TQ1_0 quantization format, except 1D tensors.
    - `MOSTLY_TQ2_0`: Represents a file type where most tensors are in TQ2_0 quantization format, except 1D tensors.
    - `GUESSED`: Represents a file type that is not specified in the model file.
- **Description**: The LlamaFileType class is an enumeration that defines various file types used in the Llama model, each associated with a specific quantization or data format. These file types are primarily used to specify the format of tensors within the model, with most types indicating a predominant format for tensors, except for 1D tensors. The class includes a variety of quantization formats such as Q4_0, Q5_0, and others, as well as floating-point formats like F32 and BF16. Additionally, it includes a special 'GUESSED' type for cases where the file type is not explicitly specified in the model file.
- **Inherits From**:
    - `IntEnum`


---
### GGUFEndian<!-- {{#class:llama.cpp/gguf-py/gguf/constants.GGUFEndian}} -->
- **Members**:
    - `LITTLE`: Represents the little-endian byte order with a value of 0.
    - `BIG`: Represents the big-endian byte order with a value of 1.
- **Description**: The GGUFEndian class is an enumeration that defines constants for representing byte order, specifically little-endian and big-endian, using integer values. It is a subclass of IntEnum, allowing for easy comparison and use in contexts where integer values are required to denote endianness.
- **Inherits From**:
    - `IntEnum`


---
### GGUFValueType<!-- {{#class:llama.cpp/gguf-py/gguf/constants.GGUFValueType}} -->
- **Members**:
    - `UINT8`: Represents an unsigned 8-bit integer type.
    - `INT8`: Represents a signed 8-bit integer type.
    - `UINT16`: Represents an unsigned 16-bit integer type.
    - `INT16`: Represents a signed 16-bit integer type.
    - `UINT32`: Represents an unsigned 32-bit integer type.
    - `INT32`: Represents a signed 32-bit integer type.
    - `FLOAT32`: Represents a 32-bit floating point type.
    - `BOOL`: Represents a boolean type.
    - `STRING`: Represents a string type.
    - `ARRAY`: Represents an array type.
    - `UINT64`: Represents an unsigned 64-bit integer type.
    - `INT64`: Represents a signed 64-bit integer type.
    - `FLOAT64`: Represents a 64-bit floating point type.
- **Description**: The GGUFValueType class is an enumeration that defines various data types used in the GGUF format, including integer, floating point, boolean, string, and array types. It provides a static method, get_type, to determine the corresponding GGUFValueType for a given Python value, facilitating type identification and conversion within the GGUF framework.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/constants.GGUFValueType.get_type`](#GGUFValueTypeget_type)
- **Inherits From**:
    - `IntEnum`

**Methods**

---
#### GGUFValueType\.get\_type<!-- {{#callable:llama.cpp/gguf-py/gguf/constants.GGUFValueType.get_type}} -->
The `get_type` method determines the `GGUFValueType` of a given value based on its Python type.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `val`: The input value whose type is to be determined; it can be of any type.
- **Control Flow**:
    - Check if the input value is an instance of `str`, `bytes`, or `bytearray`, and return `GGUFValueType.STRING` if true.
    - Check if the input value is a `list`, and return `GGUFValueType.ARRAY` if true.
    - Check if the input value is a `float`, and return `GGUFValueType.FLOAT32` if true.
    - Check if the input value is a `bool`, and return `GGUFValueType.BOOL` if true.
    - Check if the input value is an `int`, and return `GGUFValueType.INT32` if true.
    - If none of the above conditions are met, raise a `ValueError` indicating an unknown type.
- **Output**: Returns a `GGUFValueType` enumeration value corresponding to the type of the input value.
- **See also**: [`llama.cpp/gguf-py/gguf/constants.GGUFValueType`](#cpp/gguf-py/gguf/constantsGGUFValueType)  (Base Class)



---
### VisionProjectorType<!-- {{#class:llama.cpp/gguf-py/gguf/constants.VisionProjectorType}} -->
- **Members**:
    - `GEMMA3`: Represents the 'gemma3' vision projector type.
    - `IDEFICS3`: Represents the 'idefics3' vision projector type.
    - `PIXTRAL`: Represents the 'pixtral' vision projector type.
    - `LLAMA4`: Represents the 'llama4' vision projector type.
    - `QWEN2VL`: Represents the 'qwen2vl_merger' vision projector type.
    - `QWEN25VL`: Represents the 'qwen2.5vl_merger' vision projector type.
    - `ULTRAVOX`: Represents the 'ultravox' vision projector type.
    - `INTERNVL`: Represents the 'internvl' vision projector type.
    - `QWEN2A`: Represents the 'qwen2a' vision projector type, specifically for audio.
    - `QWEN25O`: Represents the 'qwen2.5o' vision projector type, specifically for omni.
- **Description**: The VisionProjectorType class defines a set of constants representing different types of vision projectors, each associated with a specific string identifier. These constants are used to categorize and identify various projector types within a vision processing context, including types for audio and omni projectors.


