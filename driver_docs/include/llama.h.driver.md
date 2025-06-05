# Purpose
The provided C header file, `llama.h`, defines a comprehensive API for a library that appears to be focused on machine learning model management, particularly for models related to natural language processing (NLP). The file includes a wide array of definitions, structures, enumerations, and function prototypes that facilitate the loading, configuration, execution, and management of machine learning models. It provides a C interface for interacting with models, handling vocabulary, managing memory, and performing tokenization and sampling operations. The file also includes support for model quantization, session management, and various types of sampling strategies, which are crucial for generating text or other outputs from language models.

Key components of this header file include structures for defining model parameters, context parameters, and batch processing, as well as enumerations for different types of vocabularies, token attributes, and model file types. The file also defines a set of functions for initializing and freeing models and contexts, loading and saving model states, and performing encoding and decoding operations. Additionally, it provides a detailed sampling API that allows for the customization of token selection strategies during text generation. The file is designed to be used as a public API, offering a broad range of functionalities for developers working with machine learning models in C, and it includes mechanisms for handling shared library exports, deprecation of older functions, and support for various backends and optimizations.
# Imports and Dependencies

---
- `ggml.h`
- `ggml-cpu.h`
- `ggml-backend.h`
- `ggml-opt.h`
- `stddef.h`
- `stdint.h`
- `stdio.h`
- `stdbool.h`


# Data Structures

---
### llama\_vocab\_type
- **Type**: `enum`
- **Members**:
    - `LLAMA_VOCAB_TYPE_NONE`: Represents models without a vocabulary.
    - `LLAMA_VOCAB_TYPE_SPM`: LLaMA tokenizer based on byte-level BPE with byte fallback.
    - `LLAMA_VOCAB_TYPE_BPE`: GPT-2 tokenizer based on byte-level BPE.
    - `LLAMA_VOCAB_TYPE_WPM`: BERT tokenizer based on WordPiece.
    - `LLAMA_VOCAB_TYPE_UGM`: T5 tokenizer based on Unigram.
    - `LLAMA_VOCAB_TYPE_RWKV`: RWKV tokenizer based on greedy tokenization.
- **Description**: The `llama_vocab_type` enumeration defines various types of vocabulary tokenizers used in different language models. Each enumerator corresponds to a specific tokenizer type, such as byte-level BPE, WordPiece, or Unigram, which are used by models like LLaMA, GPT-2, BERT, T5, and RWKV. This enumeration allows for the selection of the appropriate tokenizer type based on the model's requirements.


---
### llama\_vocab\_pre\_type
- **Type**: `enum`
- **Members**:
    - `LLAMA_VOCAB_PRE_TYPE_DEFAULT`: Represents the default pre-tokenization type with a value of 0.
    - `LLAMA_VOCAB_PRE_TYPE_LLAMA3`: Represents the LLAMA3 pre-tokenization type with a value of 1.
    - `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM`: Represents the DeepSeek LLM pre-tokenization type with a value of 2.
    - `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER`: Represents the DeepSeek Coder pre-tokenization type with a value of 3.
    - `LLAMA_VOCAB_PRE_TYPE_FALCON`: Represents the Falcon pre-tokenization type with a value of 4.
    - `LLAMA_VOCAB_PRE_TYPE_MPT`: Represents the MPT pre-tokenization type with a value of 5.
    - `LLAMA_VOCAB_PRE_TYPE_STARCODER`: Represents the StarCoder pre-tokenization type with a value of 6.
    - `LLAMA_VOCAB_PRE_TYPE_GPT2`: Represents the GPT-2 pre-tokenization type with a value of 7.
    - `LLAMA_VOCAB_PRE_TYPE_REFACT`: Represents the Refact pre-tokenization type with a value of 8.
    - `LLAMA_VOCAB_PRE_TYPE_COMMAND_R`: Represents the Command R pre-tokenization type with a value of 9.
    - `LLAMA_VOCAB_PRE_TYPE_STABLELM2`: Represents the StableLM2 pre-tokenization type with a value of 10.
    - `LLAMA_VOCAB_PRE_TYPE_QWEN2`: Represents the Qwen2 pre-tokenization type with a value of 11.
    - `LLAMA_VOCAB_PRE_TYPE_OLMO`: Represents the Olmo pre-tokenization type with a value of 12.
    - `LLAMA_VOCAB_PRE_TYPE_DBRX`: Represents the DBRX pre-tokenization type with a value of 13.
    - `LLAMA_VOCAB_PRE_TYPE_SMAUG`: Represents the Smaug pre-tokenization type with a value of 14.
    - `LLAMA_VOCAB_PRE_TYPE_PORO`: Represents the Poro pre-tokenization type with a value of 15.
    - `LLAMA_VOCAB_PRE_TYPE_CHATGLM3`: Represents the ChatGLM3 pre-tokenization type with a value of 16.
    - `LLAMA_VOCAB_PRE_TYPE_CHATGLM4`: Represents the ChatGLM4 pre-tokenization type with a value of 17.
    - `LLAMA_VOCAB_PRE_TYPE_VIKING`: Represents the Viking pre-tokenization type with a value of 18.
    - `LLAMA_VOCAB_PRE_TYPE_JAIS`: Represents the Jais pre-tokenization type with a value of 19.
    - `LLAMA_VOCAB_PRE_TYPE_TEKKEN`: Represents the Tekken pre-tokenization type with a value of 20.
    - `LLAMA_VOCAB_PRE_TYPE_SMOLLM`: Represents the Smollm pre-tokenization type with a value of 21.
    - `LLAMA_VOCAB_PRE_TYPE_CODESHELL`: Represents the CodeShell pre-tokenization type with a value of 22.
    - `LLAMA_VOCAB_PRE_TYPE_BLOOM`: Represents the Bloom pre-tokenization type with a value of 23.
    - `LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH`: Represents the GPT-3 Finnish pre-tokenization type with a value of 24.
    - `LLAMA_VOCAB_PRE_TYPE_EXAONE`: Represents the ExaOne pre-tokenization type with a value of 25.
    - `LLAMA_VOCAB_PRE_TYPE_CHAMELEON`: Represents the Chameleon pre-tokenization type with a value of 26.
    - `LLAMA_VOCAB_PRE_TYPE_MINERVA`: Represents the Minerva pre-tokenization type with a value of 27.
    - `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM`: Represents the DeepSeek3 LLM pre-tokenization type with a value of 28.
    - `LLAMA_VOCAB_PRE_TYPE_GPT4O`: Represents the GPT4O pre-tokenization type with a value of 29.
    - `LLAMA_VOCAB_PRE_TYPE_SUPERBPE`: Represents the SuperBPE pre-tokenization type with a value of 30.
    - `LLAMA_VOCAB_PRE_TYPE_TRILLION`: Represents the Trillion pre-tokenization type with a value of 31.
    - `LLAMA_VOCAB_PRE_TYPE_BAILINGMOE`: Represents the BailingMoe pre-tokenization type with a value of 32.
    - `LLAMA_VOCAB_PRE_TYPE_LLAMA4`: Represents the LLAMA4 pre-tokenization type with a value of 33.
    - `LLAMA_VOCAB_PRE_TYPE_PIXTRAL`: Represents the Pixtral pre-tokenization type with a value of 34.
    - `LLAMA_VOCAB_PRE_TYPE_SEED_CODER`: Represents the Seed Coder pre-tokenization type with a value of 35.
- **Description**: The `llama_vocab_pre_type` is an enumeration that defines various pre-tokenization types used in the LLaMA framework. Each enumerator represents a specific pre-tokenization strategy, identified by a unique integer value, which is used to process text data before tokenization. This enumeration allows for the selection of different pre-tokenization methods tailored to specific models or tasks, facilitating flexibility and customization in text processing workflows.


---
### llama\_rope\_type
- **Type**: `enum`
- **Members**:
    - `LLAMA_ROPE_TYPE_NONE`: Represents an undefined or no rope type with a value of -1.
    - `LLAMA_ROPE_TYPE_NORM`: Represents a normal rope type with a value of 0.
    - `LLAMA_ROPE_TYPE_NEOX`: Represents a Neox rope type, using the value from GGML_ROPE_TYPE_NEOX.
    - `LLAMA_ROPE_TYPE_MROPE`: Represents an Mrope type, using the value from GGML_ROPE_TYPE_MROPE.
    - `LLAMA_ROPE_TYPE_VISION`: Represents a Vision rope type, using the value from GGML_ROPE_TYPE_VISION.
- **Description**: The `llama_rope_type` is an enumeration that defines various types of rope configurations used within the LLaMA framework. It includes options for no rope type, a normal rope type, and specialized types like Neox, Mrope, and Vision, which are linked to corresponding values from the GGML library. This enumeration is likely used to specify the type of rope mechanism to be applied in different contexts or models within the LLaMA system.


---
### llama\_token\_type
- **Type**: `enum`
- **Members**:
    - `LLAMA_TOKEN_TYPE_UNDEFINED`: Represents an undefined token type with a value of 0.
    - `LLAMA_TOKEN_TYPE_NORMAL`: Represents a normal token type with a value of 1.
    - `LLAMA_TOKEN_TYPE_UNKNOWN`: Represents an unknown token type with a value of 2.
    - `LLAMA_TOKEN_TYPE_CONTROL`: Represents a control token type with a value of 3.
    - `LLAMA_TOKEN_TYPE_USER_DEFINED`: Represents a user-defined token type with a value of 4.
    - `LLAMA_TOKEN_TYPE_UNUSED`: Represents an unused token type with a value of 5.
    - `LLAMA_TOKEN_TYPE_BYTE`: Represents a byte token type with a value of 6.
- **Description**: The `llama_token_type` is an enumeration that defines various types of tokens that can be used within the LLaMA system. Each token type is associated with a specific integer value, allowing for easy identification and categorization of tokens. This enum is currently used as a placeholder until more detailed per-token attributes are available from the GGUF file.


---
### llama\_token\_attr
- **Type**: `enum`
- **Members**:
    - `LLAMA_TOKEN_ATTR_UNDEFINED`: Represents an undefined token attribute with a value of 0.
    - `LLAMA_TOKEN_ATTR_UNKNOWN`: Represents an unknown token attribute with a bitwise value of 1.
    - `LLAMA_TOKEN_ATTR_UNUSED`: Represents an unused token attribute with a bitwise value of 2.
    - `LLAMA_TOKEN_ATTR_NORMAL`: Represents a normal token attribute with a bitwise value of 4.
    - `LLAMA_TOKEN_ATTR_CONTROL`: Represents a control token attribute with a bitwise value of 8.
    - `LLAMA_TOKEN_ATTR_USER_DEFINED`: Represents a user-defined token attribute with a bitwise value of 16.
    - `LLAMA_TOKEN_ATTR_BYTE`: Represents a byte token attribute with a bitwise value of 32.
    - `LLAMA_TOKEN_ATTR_NORMALIZED`: Represents a normalized token attribute with a bitwise value of 64.
    - `LLAMA_TOKEN_ATTR_LSTRIP`: Represents a left-strip token attribute with a bitwise value of 128.
    - `LLAMA_TOKEN_ATTR_RSTRIP`: Represents a right-strip token attribute with a bitwise value of 256.
    - `LLAMA_TOKEN_ATTR_SINGLE_WORD`: Represents a single-word token attribute with a bitwise value of 512.
- **Description**: The `llama_token_attr` is an enumeration that defines various attributes for tokens in the LLaMA system. Each attribute is represented as a bitwise flag, allowing for combinations of attributes to be easily managed using bitwise operations. This enum is used to categorize tokens based on their characteristics, such as whether they are user-defined, control tokens, or have specific formatting like left or right stripping. The use of bitwise values allows for efficient storage and manipulation of these attributes.


---
### llama\_ftype
- **Type**: `enum`
- **Members**:
    - `LLAMA_FTYPE_ALL_F32`: Represents a model file type where all tensors are in 32-bit floating point format.
    - `LLAMA_FTYPE_MOSTLY_F16`: Represents a model file type where most tensors are in 16-bit floating point format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q4_0`: Represents a model file type where most tensors are quantized to Q4_0 format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q4_1`: Represents a model file type where most tensors are quantized to Q4_1 format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q8_0`: Represents a model file type where most tensors are quantized to Q8_0 format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q5_0`: Represents a model file type where most tensors are quantized to Q5_0 format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q5_1`: Represents a model file type where most tensors are quantized to Q5_1 format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q2_K`: Represents a model file type where most tensors are quantized to Q2_K format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q3_K_S`: Represents a model file type where most tensors are quantized to Q3_K_S format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q3_K_M`: Represents a model file type where most tensors are quantized to Q3_K_M format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q3_K_L`: Represents a model file type where most tensors are quantized to Q3_K_L format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q4_K_S`: Represents a model file type where most tensors are quantized to Q4_K_S format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q4_K_M`: Represents a model file type where most tensors are quantized to Q4_K_M format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q5_K_S`: Represents a model file type where most tensors are quantized to Q5_K_S format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q5_K_M`: Represents a model file type where most tensors are quantized to Q5_K_M format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q6_K`: Represents a model file type where most tensors are quantized to Q6_K format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_IQ2_XXS`: Represents a model file type where most tensors are quantized to IQ2_XXS format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_IQ2_XS`: Represents a model file type where most tensors are quantized to IQ2_XS format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_Q2_K_S`: Represents a model file type where most tensors are quantized to Q2_K_S format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_IQ3_XS`: Represents a model file type where most tensors are quantized to IQ3_XS format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_IQ3_XXS`: Represents a model file type where most tensors are quantized to IQ3_XXS format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_IQ1_S`: Represents a model file type where most tensors are quantized to IQ1_S format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_IQ4_NL`: Represents a model file type where most tensors are quantized to IQ4_NL format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_IQ3_S`: Represents a model file type where most tensors are quantized to IQ3_S format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_IQ3_M`: Represents a model file type where most tensors are quantized to IQ3_M format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_IQ2_S`: Represents a model file type where most tensors are quantized to IQ2_S format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_IQ2_M`: Represents a model file type where most tensors are quantized to IQ2_M format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_IQ4_XS`: Represents a model file type where most tensors are quantized to IQ4_XS format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_IQ1_M`: Represents a model file type where most tensors are quantized to IQ1_M format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_BF16`: Represents a model file type where most tensors are in BF16 format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_TQ1_0`: Represents a model file type where most tensors are quantized to TQ1_0 format, except 1D tensors.
    - `LLAMA_FTYPE_MOSTLY_TQ2_0`: Represents a model file type where most tensors are quantized to TQ2_0 format, except 1D tensors.
    - `LLAMA_FTYPE_GUESSED`: Represents a model file type that is not specified in the model file and is guessed.
- **Description**: The `llama_ftype` enumeration defines various model file types used in the LLaMA framework, each representing a different format or quantization level for storing model tensors. These types are primarily used to optimize the storage and processing of model data by specifying the precision or quantization method applied to the tensors, with most types excluding 1D tensors from these specifications. The enumeration includes types for full precision (F32), half precision (F16), and various quantized formats (Q4, Q5, Q8, etc.), as well as a special type for cases where the file type is not explicitly specified and must be guessed.


---
### llama\_rope\_scaling\_type
- **Type**: `enum`
- **Members**:
    - `LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED`: Represents an unspecified scaling type with a value of -1.
    - `LLAMA_ROPE_SCALING_TYPE_NONE`: Indicates no scaling with a value of 0.
    - `LLAMA_ROPE_SCALING_TYPE_LINEAR`: Represents linear scaling with a value of 1.
    - `LLAMA_ROPE_SCALING_TYPE_YARN`: Represents yarn scaling with a value of 2.
    - `LLAMA_ROPE_SCALING_TYPE_LONGROPE`: Represents longrope scaling with a value of 3.
    - `LLAMA_ROPE_SCALING_TYPE_MAX_VALUE`: Defines the maximum value for scaling types, set to LLAMA_ROPE_SCALING_TYPE_LONGROPE.
- **Description**: The `llama_rope_scaling_type` is an enumeration that defines various types of scaling methods for ropes in the context of the LLaMA library. It includes options for unspecified, none, linear, yarn, and longrope scaling, with each type represented by a unique integer value. The enumeration also defines a maximum value for the scaling types, which is set to the value of the longrope scaling type. This structure is used to specify the scaling behavior of ropes within the library's operations.


---
### llama\_pooling\_type
- **Type**: `enum`
- **Members**:
    - `LLAMA_POOLING_TYPE_UNSPECIFIED`: Represents an unspecified pooling type with a value of -1.
    - `LLAMA_POOLING_TYPE_NONE`: Indicates no pooling with a value of 0.
    - `LLAMA_POOLING_TYPE_MEAN`: Represents mean pooling with a value of 1.
    - `LLAMA_POOLING_TYPE_CLS`: Represents CLS pooling with a value of 2.
    - `LLAMA_POOLING_TYPE_LAST`: Represents last pooling with a value of 3.
    - `LLAMA_POOLING_TYPE_RANK`: Used by reranking models to attach the classification head to the graph with a value of 4.
- **Description**: The `llama_pooling_type` is an enumeration that defines various types of pooling operations that can be applied in a model. It includes options for unspecified, none, mean, CLS, last, and rank pooling types, each represented by a unique integer value. This enumeration is used to specify how embeddings or other data should be aggregated or processed in the context of a model, particularly in scenarios involving classification or reranking tasks.


---
### llama\_attention\_type
- **Type**: `enum`
- **Members**:
    - `LLAMA_ATTENTION_TYPE_UNSPECIFIED`: Represents an unspecified attention type with a value of -1.
    - `LLAMA_ATTENTION_TYPE_CAUSAL`: Represents causal attention with a value of 0.
    - `LLAMA_ATTENTION_TYPE_NON_CAUSAL`: Represents non-causal attention with a value of 1.
- **Description**: The `llama_attention_type` is an enumeration that defines different types of attention mechanisms that can be used in a model. It includes three members: `LLAMA_ATTENTION_TYPE_UNSPECIFIED` for cases where the attention type is not specified, `LLAMA_ATTENTION_TYPE_CAUSAL` for causal attention where the model attends only to past tokens, and `LLAMA_ATTENTION_TYPE_NON_CAUSAL` for non-causal attention where the model can attend to both past and future tokens. This enum is useful for configuring the attention behavior in models that require different attention strategies.


---
### llama\_split\_mode
- **Type**: `enum`
- **Members**:
    - `LLAMA_SPLIT_MODE_NONE`: Represents a mode where a single GPU is used.
    - `LLAMA_SPLIT_MODE_LAYER`: Indicates a mode where layers and key-value pairs are split across multiple GPUs.
    - `LLAMA_SPLIT_MODE_ROW`: Specifies a mode where layers and key-value pairs are split across GPUs with tensor parallelism if supported.
- **Description**: The `llama_split_mode` enumeration defines different modes for distributing computational tasks across GPUs in a multi-GPU setup. It provides options for using a single GPU, splitting layers and key-value pairs across multiple GPUs, and utilizing tensor parallelism for further optimization. This allows for flexible and efficient use of GPU resources depending on the hardware capabilities and the specific requirements of the task.


---
### llama\_token\_data
- **Type**: `struct`
- **Members**:
    - `id`: Represents the token identifier.
    - `logit`: Stores the log-odds of the token.
    - `p`: Holds the probability of the token.
- **Description**: The `llama_token_data` structure is designed to encapsulate information about a token in a language model. It includes an identifier for the token (`id`), the log-odds (`logit`) which is a measure of the token's likelihood in log scale, and the probability (`p`) which represents the likelihood of the token being selected. This structure is useful in contexts where token probabilities and their identifiers need to be managed and processed, such as in natural language processing tasks.


---
### llama\_token\_data\_array
- **Type**: `struct`
- **Members**:
    - `data`: A pointer to an array of `llama_token_data` structures.
    - `size`: The number of elements in the `data` array.
    - `selected`: An index indicating the selected element in the `data` array.
    - `sorted`: A boolean indicating whether the `data` array is sorted.
- **Description**: The `llama_token_data_array` structure is designed to manage an array of `llama_token_data` elements, which represent tokens with associated metadata such as log-odds and probabilities. It includes a pointer to the data array, the size of the array, an index for a selected element, and a flag indicating if the array is sorted. This structure is useful in scenarios where token data needs to be manipulated or accessed efficiently, such as in token sampling or processing tasks.


---
### llama\_batch
- **Type**: `struct`
- **Members**:
    - `n_tokens`: Stores the number of tokens in the batch.
    - `token`: Pointer to an array of token IDs, used when embeddings are not provided.
    - `embd`: Pointer to an array of token embeddings, used when token IDs are not provided.
    - `pos`: Pointer to an array of token positions in the sequence.
    - `n_seq_id`: Pointer to an array of sequence IDs, intended to be removed as it should belong to only one sequence.
    - `seq_id`: Pointer to an array of sequence IDs, intended to be changed to a single pointer.
    - `logits`: Pointer to an array of logits, intended to be renamed to 'output'.
- **Description**: The `llama_batch` structure is designed to encapsulate data for processing a batch of tokens in a sequence, supporting both token IDs and embeddings. It includes fields for the number of tokens, token IDs, embeddings, token positions, sequence IDs, and logits. The structure is intended for use in contexts where multiple sequences are processed, with some fields marked for future modification to streamline its design. The `llama_batch` is integral to the `llama_decode` function, which processes input sequences for language models.


---
### llama\_model\_kv\_override\_type
- **Type**: `enum`
- **Members**:
    - `LLAMA_KV_OVERRIDE_TYPE_INT`: Represents an integer key-value override type.
    - `LLAMA_KV_OVERRIDE_TYPE_FLOAT`: Represents a floating-point key-value override type.
    - `LLAMA_KV_OVERRIDE_TYPE_BOOL`: Represents a boolean key-value override type.
    - `LLAMA_KV_OVERRIDE_TYPE_STR`: Represents a string key-value override type.
- **Description**: The `llama_model_kv_override_type` is an enumeration that defines the types of key-value overrides that can be applied to a model's metadata. It includes integer, float, boolean, and string types, allowing for flexible customization of model parameters by specifying the type of value associated with a particular key.


---
### llama\_model\_kv\_override
- **Type**: `struct`
- **Members**:
    - `tag`: An enum indicating the type of value stored in the union.
    - `key`: A character array of size 128 used to store the key associated with the value.
    - `val_i64`: An int64_t integer value stored in the union.
    - `val_f64`: A double precision floating-point value stored in the union.
    - `val_bool`: A boolean value stored in the union.
    - `val_str`: A character array of size 128 used to store a string value in the union.
- **Description**: The `llama_model_kv_override` structure is designed to represent a key-value pair where the value can be of multiple types, as indicated by the `tag` member. The `key` is a fixed-size character array used to identify the value, while the value itself is stored in a union that can hold an integer, a floating-point number, a boolean, or a string. This structure allows for flexible storage and retrieval of different types of data associated with a specific key.


---
### llama\_model\_tensor\_buft\_override
- **Type**: `struct`
- **Members**:
    - `pattern`: A constant character pointer representing a pattern string.
    - `buft`: A variable of type `ggml_backend_buffer_type_t` representing the buffer type.
- **Description**: The `llama_model_tensor_buft_override` structure is used to define an override for tensor buffer types in a model. It contains a pattern string to match specific tensors and a buffer type to apply to those tensors. This allows for customization of how tensor data is handled in the backend, potentially optimizing performance or resource usage based on the specified buffer type.


---
### llama\_model\_params
- **Type**: `struct`
- **Members**:
    - `devices`: A NULL-terminated list of devices for offloading, using all available devices if NULL.
    - `tensor_buft_overrides`: A NULL-terminated list of buffer types for tensors matching a pattern.
    - `n_gpu_layers`: The number of layers to store in VRAM.
    - `split_mode`: Defines how to split the model across multiple GPUs.
    - `main_gpu`: The GPU used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE.
    - `tensor_split`: Proportion of the model to offload to each GPU, with size equal to llama_max_devices().
    - `progress_callback`: A callback function called with a progress value between 0.0 and 1.0.
    - `progress_callback_user_data`: Context pointer passed to the progress callback.
    - `kv_overrides`: Override key-value pairs of the model metadata.
    - `vocab_only`: Boolean indicating if only the vocabulary should be loaded, without weights.
    - `use_mmap`: Boolean indicating if mmap should be used if possible.
    - `use_mlock`: Boolean to force the system to keep the model in RAM.
    - `check_tensors`: Boolean to validate model tensor data.
- **Description**: The `llama_model_params` structure is designed to configure various parameters for loading and managing a machine learning model across multiple GPUs. It includes settings for device allocation, tensor buffer overrides, GPU layer storage, and model splitting strategies. Additionally, it provides options for progress tracking, metadata overrides, and memory management, such as using mmap or mlock. The structure also allows for loading only the vocabulary and validating tensor data, making it versatile for different model deployment scenarios.


---
### llama\_context\_params
- **Type**: `struct`
- **Members**:
    - `n_ctx`: Specifies the text context, with 0 indicating it should be derived from the model.
    - `n_batch`: Defines the logical maximum batch size for llama_decode submissions.
    - `n_ubatch`: Indicates the physical maximum batch size.
    - `n_seq_max`: Sets the maximum number of sequences for recurrent models.
    - `n_threads`: Determines the number of threads for generation.
    - `n_threads_batch`: Specifies the number of threads for batch processing.
    - `rope_scaling_type`: Defines the RoPE scaling type using the llama_rope_scaling_type enum.
    - `pooling_type`: Indicates whether to pool embedding results by sequence id using the llama_pooling_type enum.
    - `attention_type`: Specifies the attention type for embeddings using the llama_attention_type enum.
    - `rope_freq_base`: Sets the RoPE base frequency, with 0 indicating it should be derived from the model.
    - `rope_freq_scale`: Defines the RoPE frequency scaling factor, with 0 indicating it should be derived from the model.
    - `yarn_ext_factor`: Specifies the YaRN extrapolation mix factor, with negative values indicating it should be derived from the model.
    - `yarn_attn_factor`: Sets the YaRN magnitude scaling factor.
    - `yarn_beta_fast`: Defines the YaRN low correction dimension.
    - `yarn_beta_slow`: Specifies the YaRN high correction dimension.
    - `yarn_orig_ctx`: Indicates the original context size for YaRN.
    - `defrag_thold`: Sets the threshold for defragmenting the KV cache, with values <= 0 disabling it.
    - `cb_eval`: Callback for evaluation scheduling in the backend.
    - `cb_eval_user_data`: User data for the evaluation callback.
    - `type_k`: Specifies the data type for the K cache, marked as experimental.
    - `type_v`: Specifies the data type for the V cache, marked as experimental.
    - `abort_callback`: Callback to abort llama_decode execution if it returns true, currently only for CPU execution.
    - `abort_callback_data`: User data for the abort callback.
    - `embeddings`: Boolean indicating if embeddings should be extracted along with logits.
    - `offload_kqv`: Boolean to offload KQV operations, including the KV cache, to GPU.
    - `flash_attn`: Boolean to use flash attention, marked as experimental.
    - `no_perf`: Boolean to measure performance timings.
    - `op_offload`: Boolean to offload host tensor operations to a device.
    - `swa_full`: Boolean to use a full-size SWA cache, with performance implications if false when n_seq_max > 1.
- **Description**: The `llama_context_params` structure is a comprehensive configuration entity for managing the context and operational parameters of the llama model's execution environment. It includes settings for context size, batch processing, threading, and various scaling and attention types. Additionally, it provides options for callback functions, data types for caches, and several boolean flags to control features like embedding extraction, GPU offloading, and performance measurement. This structure is crucial for fine-tuning the model's behavior and optimizing its performance across different computational environments.


---
### llama\_model\_quantize\_params
- **Type**: `struct`
- **Members**:
    - `nthread`: Number of threads to use for quantizing, defaults to hardware concurrency if <=0.
    - `ftype`: Specifies the llama_ftype to quantize to.
    - `output_tensor_type`: Defines the type of the output tensor.
    - `token_embedding_type`: Specifies the type for token embeddings tensor.
    - `allow_requantize`: Indicates if non-f32/f16 tensors can be quantized.
    - `quantize_output_tensor`: Determines if the output.weight should be quantized.
    - `only_copy`: If true, only copies tensors, ignoring ftype, allow_requantize, and quantize_output_tensor.
    - `pure`: Quantizes all tensors to the default type if true.
    - `keep_split`: Ensures quantization maintains the same number of shards.
    - `imatrix`: Pointer to importance matrix data.
    - `kv_overrides`: Pointer to a vector containing key-value overrides.
    - `tensor_types`: Pointer to a vector containing tensor types.
- **Description**: The `llama_model_quantize_params` structure is designed to hold parameters for quantizing a model in the LLaMA framework. It includes settings for the number of threads, the target quantization type, and specific tensor types for output and token embeddings. Additionally, it provides options to allow requantization, quantize specific tensors, and maintain shard consistency. The structure also supports copying tensors without quantization and includes pointers to additional data such as importance matrices and key-value overrides.


---
### llama\_logit\_bias
- **Type**: `struct`
- **Members**:
    - `token`: Represents a token in the llama vocabulary, identified by an integer.
    - `bias`: A floating-point value representing the bias associated with the token.
- **Description**: The `llama_logit_bias` structure is used to associate a bias value with a specific token in the llama vocabulary. This structure is likely used in contexts where token biases need to be adjusted, such as in language models or token sampling processes, to influence the probability of selecting certain tokens during generation or decoding.


---
### llama\_sampler\_chain\_params
- **Type**: `struct`
- **Members**:
    - `no_perf`: Indicates whether to measure performance timings.
- **Description**: The `llama_sampler_chain_params` structure is a simple data structure used to configure parameters for a llama sampler chain. It currently contains a single boolean member, `no_perf`, which specifies whether performance timings should be measured during the sampling process. This structure is likely used to control the behavior of a sampler chain in terms of performance monitoring.


---
### llama\_chat\_message
- **Type**: `struct`
- **Members**:
    - `role`: A pointer to a constant character string representing the role of the message sender.
    - `content`: A pointer to a constant character string containing the message content.
- **Description**: The `llama_chat_message` structure is designed to represent a chat message within a chat application or system. It contains two members: `role`, which indicates the role of the message sender (such as 'user' or 'assistant'), and `content`, which holds the actual text of the message. This structure is useful for organizing and managing chat interactions, allowing for easy access to both the sender's role and the message content.


# Function Declarations (Public API)

---
### llama\_model\_default\_params<!-- {{#callable_declaration:llama_model_default_params}} -->
Returns the default parameters for a llama model.
- **Description**: This function provides a set of default parameters for initializing a llama model. It is useful when you want to start with a standard configuration and then modify specific parameters as needed. The function returns a `llama_model_params` structure with pre-set values, which include settings for GPU layers, split mode, and memory mapping options. This function is typically called before loading or configuring a model to ensure that all parameters have valid initial values.
- **Inputs**: None
- **Output**: A `llama_model_params` structure initialized with default values.
- **See also**: [`llama_model_default_params`](../src/llama-model.cpp.driver.md#llama_model_default_params)  (Implementation)


---
### llama\_context\_default\_params<!-- {{#callable_declaration:llama_context_default_params}} -->
Returns default parameters for a llama context.
- **Description**: Use this function to obtain a set of default parameters for initializing a llama context. This is useful when you want to start with a standard configuration and modify only specific parameters as needed. The function provides a pre-defined set of values that are generally suitable for typical use cases, ensuring a consistent starting point for context configuration.
- **Inputs**: None
- **Output**: A `llama_context_params` structure initialized with default values.
- **See also**: [`llama_context_default_params`](../src/llama-context.cpp.driver.md#llama_context_default_params)  (Implementation)


---
### llama\_sampler\_chain\_default\_params<!-- {{#callable_declaration:llama_sampler_chain_default_params}} -->
Returns the default parameters for a llama sampler chain.
- **Description**: Use this function to obtain a `llama_sampler_chain_params` structure initialized with default values. This is typically the first step when setting up a sampler chain, ensuring that all parameters are set to their default states before any modifications are made. This function does not require any input parameters and provides a convenient way to start with a known configuration.
- **Inputs**: None
- **Output**: A `llama_sampler_chain_params` structure with default settings, specifically with `no_perf` set to `true`.
- **See also**: [`llama_sampler_chain_default_params`](../src/llama.cpp.driver.md#llama_sampler_chain_default_params)  (Implementation)


---
### llama\_model\_quantize\_default\_params<!-- {{#callable_declaration:llama_model_quantize_default_params}} -->
Returns default quantization parameters for a llama model.
- **Description**: Use this function to obtain a set of default parameters for quantizing a llama model. This is useful when you want to start with a standard configuration and then modify specific parameters as needed for your application. The function does not require any input and provides a pre-configured set of parameters that can be used directly or adjusted before passing them to other functions that perform model quantization.
- **Inputs**: None
- **Output**: A `llama_model_quantize_params` structure initialized with default values for model quantization.
- **See also**: [`llama_model_quantize_default_params`](../src/llama-quant.cpp.driver.md#llama_model_quantize_default_params)  (Implementation)


---
### llama\_backend\_init<!-- {{#callable_declaration:llama_backend_init}} -->
Initialize the llama and ggml backend.
- **Description**: This function should be called once at the start of the program to initialize the llama and ggml backend. It sets up necessary internal structures and prepares the system for further operations. This initialization is crucial for ensuring that subsequent calls to other functions in the library operate correctly. It does not take any parameters and does not return any value, indicating that it is primarily used for setup purposes.
- **Inputs**: None
- **Output**: None
- **See also**: [`llama_backend_init`](../src/llama.cpp.driver.md#llama_backend_init)  (Implementation)


---
### llama\_backend\_free<!-- {{#callable_declaration:llama_backend_free}} -->
Frees resources used by the llama backend.
- **Description**: This function should be called once at the end of the program to release any resources allocated by the llama backend. It is essential to ensure that all resources are properly freed to prevent memory leaks. This function is typically used in conjunction with llama_backend_init, which initializes the backend. Ensure that no llama backend operations are pending or in progress when calling this function.
- **Inputs**: None
- **Output**: None
- **See also**: [`llama_backend_free`](../src/llama.cpp.driver.md#llama_backend_free)  (Implementation)


---
### llama\_numa\_init<!-- {{#callable_declaration:llama_numa_init}} -->
Initializes NUMA optimizations for the backend if enabled.
- **Description**: This function initializes NUMA (Non-Uniform Memory Access) optimizations for the backend if the specified strategy is not disabled. It should be called to enable NUMA optimizations when the application starts, provided that the CPU backend is loaded. This function does not perform any action if the strategy is set to `GGML_NUMA_STRATEGY_DISABLED`. Ensure that the CPU backend is properly loaded before calling this function to avoid assertion failures.
- **Inputs**:
    - `numa`: An enumeration value of type `enum ggml_numa_strategy` that specifies the NUMA strategy to be used. If set to `GGML_NUMA_STRATEGY_DISABLED`, the function will not perform any initialization. The caller retains ownership of this parameter.
- **Output**: None
- **See also**: [`llama_numa_init`](../src/llama.cpp.driver.md#llama_numa_init)  (Implementation)


---
### llama\_attach\_threadpool<!-- {{#callable_declaration:llama_attach_threadpool}} -->
Attach a threadpool to a llama context for processing.
- **Description**: This function is used to associate a threadpool with a given llama context, allowing for parallel processing within that context. It is typically called to optimize performance by utilizing multiple threads for operations within the context. The function should be called when a threadpool is available and you want to leverage it for batch processing or other parallelizable tasks. Ensure that the context and threadpool are properly initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a llama_context structure. This must be a valid, initialized context and must not be null. The caller retains ownership.
    - `threadpool`: A ggml_threadpool_t object representing the threadpool to be attached. It must be a valid threadpool object.
    - `threadpool_batch`: A ggml_threadpool_t object representing the batch threadpool to be attached. It must be a valid threadpool object.
- **Output**: None
- **See also**: [`llama_attach_threadpool`](../src/llama-context.cpp.driver.md#llama_attach_threadpool)  (Implementation)


---
### llama\_detach\_threadpool<!-- {{#callable_declaration:llama_detach_threadpool}} -->
Detaches the thread pool from the given llama context.
- **Description**: Use this function to detach any thread pool that has been associated with a llama context. This is typically necessary when you want to manage threading independently or when cleaning up resources. Ensure that the context is valid and properly initialized before calling this function. It should be called when the thread pool is no longer needed or before the context is destroyed to prevent resource leaks.
- **Inputs**:
    - `ctx`: A pointer to a llama_context structure. This must be a valid, non-null pointer to a llama context from which the thread pool will be detached. The function assumes the context is properly initialized and does not perform null checks.
- **Output**: None
- **See also**: [`llama_detach_threadpool`](../src/llama-context.cpp.driver.md#llama_detach_threadpool)  (Implementation)


