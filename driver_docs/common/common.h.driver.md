# Purpose
This C++ header file provides a comprehensive set of utilities and helper functions designed to support various operations related to the "llama" library, which appears to be a machine learning or natural language processing framework. The file includes a wide range of functionalities, from CPU utility functions and model parameter structures to string manipulation and tokenization utilities. It defines several data structures and enumerations for handling model parameters, sampling configurations, and CPU settings, which are crucial for configuring and optimizing the performance of machine learning models. Additionally, the file includes macros for error handling and build information logging, which are essential for debugging and maintaining the software.

The file is structured to offer both broad and specific functionalities, such as CPU parameter management, model initialization, and token processing. It defines public APIs and external interfaces that can be used by other components of the software to interact with the "llama" library. The inclusion of various utility functions for string operations, file system interactions, and model management indicates that this file serves as a foundational component for building and running machine learning models within the "llama" framework. The presence of detailed structures for model and sampling parameters suggests that the file is intended to be imported and used by other parts of the software to facilitate complex operations like model training, inference, and evaluation.
# Imports and Dependencies

---
- `llama-cpp.h`
- `set`
- `string`
- `string_view`
- `vector`
- `sstream`


# Global Variables

---
### LLAMA\_BUILD\_NUMBER
- **Type**: `int`
- **Description**: `LLAMA_BUILD_NUMBER` is a global integer variable that represents the build number of the software. It is used to track the specific version or iteration of the build during development and deployment.
- **Use**: This variable is used in the `print_build_info()` macro to display the build number along with other build-related information.


---
### LLAMA\_COMMIT
- **Type**: `const char *`
- **Description**: `LLAMA_COMMIT` is a global constant pointer to a character string that represents the commit hash of the source code repository for the LLAMA project. This variable is used to identify the specific version of the codebase that is being used or built.
- **Use**: It is used in the `print_build_info()` macro to display the commit hash as part of the build information.


---
### LLAMA\_COMPILER
- **Type**: `const char *`
- **Description**: `LLAMA_COMPILER` is a global constant pointer to a character string that holds the name of the compiler used to build the software. It is declared as an external variable, meaning its definition is expected to be provided elsewhere in the program or linked from another module.
- **Use**: This variable is used to display the compiler information in the build information output, helping to identify the environment in which the software was compiled.


---
### LLAMA\_BUILD\_TARGET
- **Type**: `const char *`
- **Description**: `LLAMA_BUILD_TARGET` is a global constant pointer to a character string that represents the target platform for which the software is built. It is used to store the name of the platform or architecture, such as 'Windows', 'Linux', or 'macOS', on which the build is intended to run.
- **Use**: This variable is used in the `print_build_info` macro to display the build target information when logging or debugging.


# Data Structures

---
### common\_adapter\_lora\_info<!-- {{#data_structure:common_adapter_lora_info}} -->
- **Type**: `struct`
- **Members**:
    - `path`: A string representing the file path associated with the LoRA adapter.
    - `scale`: A floating-point value indicating the scale factor for the LoRA adapter.
    - `ptr`: A pointer to a `llama_adapter_lora` structure, representing the LoRA adapter instance.
- **Description**: The `common_adapter_lora_info` struct is designed to encapsulate information related to a LoRA (Low-Rank Adaptation) adapter. It includes a file path to the adapter, a scale factor to adjust its influence, and a pointer to the actual LoRA adapter structure. This struct is likely used in contexts where multiple LoRA adapters need to be managed, allowing for dynamic loading and scaling of these adapters in machine learning or AI model applications.


---
### cpu\_params<!-- {{#data_structure:cpu_params}} -->
- **Type**: `struct`
- **Members**:
    - `n_threads`: Specifies the number of threads to use, defaulting to -1.
    - `cpumask`: An array representing the CPU affinity mask, initialized to false for all elements.
    - `mask_valid`: Indicates if the CPU mask is valid, defaulting to false.
    - `priority`: Defines the scheduling priority with possible values ranging from normal to realtime.
    - `strict_cpu`: Determines if strict CPU placement should be used, defaulting to false.
    - `poll`: Specifies the polling level for busy-waiting, with a default value of 50.
- **Description**: The `cpu_params` struct is designed to configure CPU-related parameters for a system, including the number of threads, CPU affinity mask, scheduling priority, and polling level. It allows for fine-tuning of CPU usage by setting thread count, enabling strict CPU placement, and adjusting the polling level for busy-waiting. The struct also includes a boolean to validate the CPU mask and an enumeration to set the scheduling priority, providing flexibility in managing CPU resources.


---
### llama\_example<!-- {{#data_structure:llama_example}} -->
- **Type**: `enum`
- **Members**:
    - `LLAMA_EXAMPLE_COMMON`: Represents a common example type.
    - `LLAMA_EXAMPLE_SPECULATIVE`: Represents a speculative example type.
    - `LLAMA_EXAMPLE_MAIN`: Represents the main example type.
    - `LLAMA_EXAMPLE_EMBEDDING`: Represents an embedding example type.
    - `LLAMA_EXAMPLE_PERPLEXITY`: Represents a perplexity example type.
    - `LLAMA_EXAMPLE_RETRIEVAL`: Represents a retrieval example type.
    - `LLAMA_EXAMPLE_PASSKEY`: Represents a passkey example type.
    - `LLAMA_EXAMPLE_IMATRIX`: Represents an imatrix example type.
    - `LLAMA_EXAMPLE_BENCH`: Represents a benchmark example type.
    - `LLAMA_EXAMPLE_SERVER`: Represents a server example type.
    - `LLAMA_EXAMPLE_CVECTOR_GENERATOR`: Represents a cvector generator example type.
    - `LLAMA_EXAMPLE_EXPORT_LORA`: Represents an export LORA example type.
    - `LLAMA_EXAMPLE_MTMD`: Represents an MTMD example type.
    - `LLAMA_EXAMPLE_LOOKUP`: Represents a lookup example type.
    - `LLAMA_EXAMPLE_PARALLEL`: Represents a parallel example type.
    - `LLAMA_EXAMPLE_TTS`: Represents a text-to-speech example type.
    - `LLAMA_EXAMPLE_COUNT`: Represents the count of example types.
- **Description**: The `llama_example` enum defines a set of constants representing different types of examples or modes that can be used within the application. Each enumerator corresponds to a specific example type, such as common, speculative, main, embedding, and others, which are likely used to categorize or control different functionalities or behaviors in the system. The `LLAMA_EXAMPLE_COUNT` enumerator is used to represent the total number of example types defined in this enumeration.


---
### common\_sampler\_type<!-- {{#data_structure:common_sampler_type}} -->
- **Type**: `enum`
- **Members**:
    - `COMMON_SAMPLER_TYPE_NONE`: Represents no sampler type, with a value of 0.
    - `COMMON_SAMPLER_TYPE_DRY`: Represents a dry sampler type, with a value of 1.
    - `COMMON_SAMPLER_TYPE_TOP_K`: Represents a top-k sampler type, with a value of 2.
    - `COMMON_SAMPLER_TYPE_TOP_P`: Represents a top-p sampler type, with a value of 3.
    - `COMMON_SAMPLER_TYPE_MIN_P`: Represents a minimum probability sampler type, with a value of 4.
    - `COMMON_SAMPLER_TYPE_TYPICAL_P`: Represents a typical probability sampler type, with a value of 6.
    - `COMMON_SAMPLER_TYPE_TEMPERATURE`: Represents a temperature-based sampler type, with a value of 7.
    - `COMMON_SAMPLER_TYPE_XTC`: Represents an XTC sampler type, with a value of 8.
    - `COMMON_SAMPLER_TYPE_INFILL`: Represents an infill sampler type, with a value of 9.
    - `COMMON_SAMPLER_TYPE_PENALTIES`: Represents a penalties sampler type, with a value of 10.
    - `COMMON_SAMPLER_TYPE_TOP_N_SIGMA`: Represents a top-n sigma sampler type, with a value of 11.
- **Description**: The `common_sampler_type` enum defines a set of constants representing different types of sampling strategies used in probabilistic models or algorithms. Each enumerator corresponds to a specific sampling method, such as top-k, top-p, temperature-based, and others, which are commonly used in machine learning and natural language processing tasks to control the randomness and diversity of generated outputs. The values assigned to each enumerator are unique integers, facilitating their use in switch-case statements or other control structures.


---
### dimre\_method<!-- {{#data_structure:dimre_method}} -->
- **Type**: `enum`
- **Members**:
    - `DIMRE_METHOD_PCA`: Represents the Principal Component Analysis method for dimensionality reduction.
    - `DIMRE_METHOD_MEAN`: Represents the Mean method for dimensionality reduction.
- **Description**: The `dimre_method` enum defines two methods for dimensionality reduction: Principal Component Analysis (PCA) and Mean. These methods are used by the cvector-generator to reduce the dimensionality of data, which can be useful in various machine learning and data processing tasks to simplify models and reduce computational load.


---
### common\_conversation\_mode<!-- {{#data_structure:common_conversation_mode}} -->
- **Type**: `enum`
- **Members**:
    - `COMMON_CONVERSATION_MODE_DISABLED`: Represents a disabled state for the conversation mode.
    - `COMMON_CONVERSATION_MODE_ENABLED`: Represents an enabled state for the conversation mode.
    - `COMMON_CONVERSATION_MODE_AUTO`: Represents an automatic state for the conversation mode.
- **Description**: The `common_conversation_mode` enum defines three possible states for a conversation mode: disabled, enabled, and automatic. This enum is likely used to control or configure the behavior of a conversation system, allowing it to be turned off, turned on, or set to automatically determine its state based on certain conditions.


---
### common\_grammar\_trigger\_type<!-- {{#data_structure:common_grammar_trigger_type}} -->
- **Type**: `enum`
- **Members**:
    - `COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN`: Represents a trigger type based on a specific token.
    - `COMMON_GRAMMAR_TRIGGER_TYPE_WORD`: Represents a trigger type based on a specific word.
    - `COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN`: Represents a trigger type based on a pattern.
    - `COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL`: Represents a trigger type based on a full pattern.
- **Description**: The `common_grammar_trigger_type` is an enumeration that defines different types of grammar triggers used in a system. These triggers can be based on tokens, words, patterns, or full patterns, allowing for flexible grammar rule application based on the type of input detected.


---
### common\_grammar\_trigger<!-- {{#data_structure:common_grammar_trigger}} -->
- **Type**: `struct`
- **Members**:
    - `type`: Specifies the type of grammar trigger using the `common_grammar_trigger_type` enum.
    - `value`: Holds a string value associated with the grammar trigger.
    - `token`: Represents a token associated with the grammar trigger, defaulting to `LLAMA_TOKEN_NULL`.
- **Description**: The `common_grammar_trigger` struct is designed to represent a grammar trigger within a system, encapsulating the type of trigger, its associated string value, and an optional token. This structure is likely used in contexts where grammar rules or patterns need to be dynamically applied or recognized, with the `type` field indicating the nature of the trigger (such as token, word, or pattern), the `value` field storing the relevant string data, and the `token` field providing an optional token reference for further processing or identification.


---
### common\_params\_sampling<!-- {{#data_structure:common_params_sampling}} -->
- **Type**: `struct`
- **Members**:
    - `seed`: The seed used to initialize the llama_sampler.
    - `n_prev`: Number of previous tokens to remember.
    - `n_probs`: If greater than 0, output the probabilities of top n_probs tokens.
    - `min_keep`: Minimum number of tokens samplers should return, 0 means disabled.
    - `top_k`: Top-k sampling parameter, <= 0 to use vocab size.
    - `top_p`: Top-p sampling parameter, 1.0 means disabled.
    - `min_p`: Minimum probability threshold, 0.0 means disabled.
    - `xtc_probability`: Probability for XTC sampling, 0.0 means disabled.
    - `xtc_threshold`: Threshold for XTC sampling, > 0.5 disables XTC.
    - `typ_p`: Typical probability parameter, 1.0 means disabled.
    - `temp`: Temperature for sampling, <= 0.0 to sample greedily.
    - `dynatemp_range`: Range for dynamic temperature sampling, 0.0 means disabled.
    - `dynatemp_exponent`: Exponent for dynamic temperature sampling.
    - `penalty_last_n`: Number of last tokens to penalize, 0 means disable penalty.
    - `penalty_repeat`: Repetition penalty, 1.0 means disabled.
    - `penalty_freq`: Frequency penalty, 0.0 means disabled.
    - `penalty_present`: Presence penalty, 0.0 means disabled.
    - `dry_multiplier`: Multiplier for DRY repetition penalty, 0.0 means disabled.
    - `dry_base`: Base for DRY repetition penalty, 0.0 means disabled.
    - `dry_allowed_length`: Allowed length for DRY repetition before penalty.
    - `dry_penalty_last_n`: Number of tokens to scan for DRY repetitions, 0 means disable penalty.
    - `mirostat`: Mirostat sampling mode, 0 means disabled.
    - `top_n_sigma`: Top-n sigma sampling parameter, -1.0 means disabled.
    - `mirostat_tau`: Target entropy for Mirostat.
    - `mirostat_eta`: Learning rate for Mirostat.
    - `ignore_eos`: Flag to ignore end-of-sequence tokens.
    - `no_perf`: Flag to disable performance metrics.
    - `timing_per_token`: Flag to enable timing per token.
    - `dry_sequence_breakers`: Default sequence breakers for DRY.
    - `samplers`: List of sampler types to use.
    - `grammar`: Optional BNF-like grammar to constrain sampling.
    - `grammar_lazy`: Flag to enable lazy grammar evaluation.
    - `grammar_triggers`: Optional triggers for lazy grammars.
    - `preserved_tokens`: Set of tokens to preserve during sampling.
    - `logit_bias`: Logit biases to apply during sampling.
- **Description**: The `common_params_sampling` struct is a comprehensive configuration structure used to define various parameters for token sampling in a language model. It includes settings for random seed initialization, token memory, probability outputs, and various sampling strategies such as top-k, top-p, and temperature sampling. Additionally, it provides options for dynamic temperature adjustments, penalties for token repetition, and specific configurations for advanced sampling techniques like Mirostat. The struct also supports grammar constraints and logit biases, making it highly customizable for different sampling needs in natural language processing tasks.
- **Member Functions**:
    - [`common_params_sampling::print`](sampling.cpp.driver.md#common_params_samplingprint)


---
### common\_params\_model<!-- {{#data_structure:common_params_model}} -->
- **Type**: `struct`
- **Members**:
    - `path`: Represents the local path to the model.
    - `url`: Represents the URL from which the model can be downloaded.
    - `hf_repo`: Specifies the Hugging Face repository associated with the model.
    - `hf_file`: Specifies the file within the Hugging Face repository.
- **Description**: The `common_params_model` struct is designed to encapsulate the essential parameters required for handling a model's location and source information. It includes fields for specifying the local path where the model is stored, the URL for downloading the model, and details about the associated Hugging Face repository and file. This struct is useful for managing model resources in a structured and consistent manner.


---
### common\_params\_speculative<!-- {{#data_structure:common_params_speculative}} -->
- **Type**: `struct`
- **Members**:
    - `devices`: A vector of devices used for offloading.
    - `n_ctx`: Draft context size.
    - `n_max`: Maximum number of tokens to draft during speculative decoding.
    - `n_min`: Minimum number of draft tokens to use for speculative decoding.
    - `n_gpu_layers`: Number of layers to store in VRAM for the draft model (-1 indicates default).
    - `p_split`: Speculative decoding split probability.
    - `p_min`: Minimum speculative decoding probability (greedy).
    - `cpuparams`: CPU parameters for the speculative decoding process.
    - `cpuparams_batch`: Batch CPU parameters for the speculative decoding process.
    - `model`: Model parameters for speculative decoding.
- **Description**: The `common_params_speculative` struct is designed to hold configuration parameters for speculative decoding in a machine learning context. It includes settings for device offloading, context size, token limits, and GPU layer management, as well as probabilities for speculative decoding. Additionally, it contains CPU and model parameters to fine-tune the speculative decoding process, allowing for efficient and flexible model execution.


---
### common\_params\_vocoder<!-- {{#data_structure:common_params_vocoder}} -->
- **Type**: `struct`
- **Members**:
    - `model`: A nested struct of type `common_params_model` that holds model-related parameters.
    - `speaker_file`: A string representing the file path to the speaker file.
    - `use_guide_tokens`: A boolean flag indicating whether guide tokens are used to improve TTS accuracy.
- **Description**: The `common_params_vocoder` struct is designed to encapsulate parameters specific to a vocoder, which is a component used in text-to-speech (TTS) systems. It includes a nested `common_params_model` struct for model-related parameters, a string for specifying the path to a speaker file, and a boolean flag to enable or disable the use of guide tokens, which can enhance the accuracy of TTS outputs.


---
### common\_reasoning\_format<!-- {{#data_structure:common_reasoning_format}} -->
- **Type**: `enum`
- **Members**:
    - `COMMON_REASONING_FORMAT_NONE`: Represents the absence of a reasoning format.
    - `COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY`: Extracts thinking tag contents and returns them as `message.reasoning_content`, or leaves them inline in <think> tags in stream mode.
    - `COMMON_REASONING_FORMAT_DEEPSEEK`: Extracts thinking tag contents and returns them as `message.reasoning_content`, including in streaming deltas.
- **Description**: The `common_reasoning_format` enum defines different formats for handling reasoning content in a system. It includes options for no reasoning format, a legacy format that extracts content from thinking tags, and a more advanced format that also supports streaming deltas. This enum is used to specify how reasoning content should be processed and returned in a message.


---
### common\_params<!-- {{#data_structure:common_params}} -->
- **Type**: `struct`
- **Members**:
    - `n_predict`: Specifies the number of new tokens to predict.
    - `n_ctx`: Defines the context size for processing.
    - `n_batch`: Logical batch size for prompt processing.
    - `n_ubatch`: Physical batch size for prompt processing.
    - `n_keep`: Number of tokens to retain from the initial prompt.
    - `n_chunks`: Maximum number of chunks to process.
    - `n_parallel`: Number of parallel sequences to decode.
    - `n_sequences`: Number of sequences to decode.
    - `grp_attn_n`: Group-attention factor.
    - `grp_attn_w`: Group-attention width.
    - `n_print`: Print token count every n tokens.
    - `rope_freq_base`: RoPE base frequency.
    - `rope_freq_scale`: RoPE frequency scaling factor.
    - `yarn_ext_factor`: YaRN extrapolation mix factor.
    - `yarn_attn_factor`: YaRN magnitude scaling factor.
    - `yarn_beta_fast`: YaRN low correction dimension.
    - `yarn_beta_slow`: YaRN high correction dimension.
    - `yarn_orig_ctx`: YaRN original context length.
    - `defrag_thold`: KV cache defragmentation threshold.
    - `devices`: Devices to use for offloading.
    - `n_gpu_layers`: Number of layers to store in VRAM.
    - `main_gpu`: The GPU used for scratch and small tensors.
    - `tensor_split`: Distribution of split tensors across GPUs.
    - `split_mode`: Mode for splitting the model across GPUs.
    - `cpuparams`: CPU parameters for processing.
    - `cpuparams_batch`: CPU parameters for batch processing.
    - `cb_eval`: Callback for evaluation scheduling.
    - `cb_eval_user_data`: User data for evaluation callback.
    - `numa`: NUMA strategy for memory allocation.
    - `rope_scaling_type`: Type of RoPE scaling.
    - `pooling_type`: Pooling type for embeddings.
    - `attention_type`: Attention type for embeddings.
    - `sampling`: Parameters for sampling.
    - `speculative`: Parameters for speculative decoding.
    - `vocoder`: Parameters for vocoder processing.
    - `model`: Model parameters.
    - `model_alias`: Alias for the model.
    - `hf_token`: Token for Hugging Face integration.
    - `prompt`: Initial prompt for processing.
    - `system_prompt`: System prompt for processing.
    - `prompt_file`: External prompt file name.
    - `path_prompt_cache`: Path for saving/loading prompt evaluation state.
    - `input_prefix`: Prefix for user inputs.
    - `input_suffix`: Suffix for user inputs.
    - `lookup_cache_static`: Path for static ngram cache file.
    - `lookup_cache_dynamic`: Path for dynamic ngram cache file.
    - `logits_file`: File for saving all logits.
    - `in_files`: List of all input files.
    - `antiprompt`: Strings for reverse prompts.
    - `kv_overrides`: Overrides for key-value pairs in the model.
    - `tensor_buft_overrides`: Overrides for tensor buffer types.
    - `lora_init_without_apply`: Flag to load LoRA without applying it.
    - `lora_adapters`: LoRA adapter paths with user-defined scale.
    - `control_vectors`: Control vectors with user-defined scale.
    - `verbosity`: Level of verbosity for logging.
    - `control_vector_layer_start`: Start layer for control vector application.
    - `control_vector_layer_end`: End layer for control vector application.
    - `offline`: Flag to indicate offline mode.
    - `ppl_stride`: Stride for perplexity calculations.
    - `ppl_output_type`: Output type for perplexity calculations.
    - `hellaswag`: Flag to compute HellaSwag score.
    - `hellaswag_tasks`: Number of tasks for HellaSwag score computation.
    - `winogrande`: Flag to compute Winogrande score.
    - `winogrande_tasks`: Number of tasks for Winogrande score computation.
    - `multiple_choice`: Flag to compute TruthfulQA score.
    - `multiple_choice_tasks`: Number of tasks for TruthfulQA score computation.
    - `kl_divergence`: Flag to compute KL divergence.
    - `usage`: Flag to print usage information.
    - `completion`: Flag to print completion script.
    - `use_color`: Flag to use color in outputs.
    - `special`: Flag to enable special token output.
    - `interactive`: Flag for interactive mode.
    - `interactive_first`: Flag to wait for user input immediately.
    - `prompt_cache_all`: Flag to save all user input and generations to cache.
    - `prompt_cache_ro`: Flag to open prompt cache read-only.
    - `escape`: Flag to escape special characters.
    - `multiline_input`: Flag to reverse the usage of backslash.
    - `simple_io`: Flag to improve compatibility with subprocesses.
    - `cont_batching`: Flag to insert new sequences for decoding on-the-fly.
    - `flash_attn`: Flag to enable flash attention.
    - `no_perf`: Flag to disable performance metrics.
    - `ctx_shift`: Flag for context shift in infinite text generation.
    - `swa_full`: Flag to use full-size SWA cache.
    - `input_prefix_bos`: Flag to prefix BOS to user inputs.
    - `use_mmap`: Flag to use mmap for faster loads.
    - `use_mlock`: Flag to use mlock to keep model in memory.
    - `verbose_prompt`: Flag to print prompt tokens before generation.
    - `display_prompt`: Flag to print prompt before generation.
    - `no_kv_offload`: Flag to disable KV offloading.
    - `warmup`: Flag for warmup run.
    - `check_tensors`: Flag to validate tensor data.
    - `no_op_offload`: Flag to disable offload host tensor operations.
    - `single_turn`: Flag for single turn chat conversation.
    - `cache_type_k`: KV cache data type for the K.
    - `cache_type_v`: KV cache data type for the V.
    - `conversation_mode`: Mode for conversation handling.
    - `mmproj`: Parameters for multimodal model projection.
    - `mmproj_use_gpu`: Flag to use GPU for multimodal model.
    - `no_mmproj`: Flag to disable multimodal model.
    - `image`: Paths to image files for processing.
    - `embedding`: Flag to get only sentence embedding.
    - `embd_normalize`: Normalization method for embeddings.
    - `embd_out`: Output format for embeddings.
    - `embd_sep`: Separator for embeddings.
    - `reranking`: Flag to enable reranking support.
    - `port`: Network port for server listening.
    - `timeout_read`: HTTP read timeout in seconds.
    - `timeout_write`: HTTP write timeout in seconds.
    - `n_threads_http`: Number of threads for HTTP request processing.
    - `n_cache_reuse`: Minimum chunk size for cache reuse.
    - `hostname`: Hostname for server.
    - `public_path`: Public path for server resources.
    - `chat_template`: Template for chat responses.
    - `use_jinja`: Flag to use Jinja for templating.
    - `enable_chat_template`: Flag to enable chat template.
    - `reasoning_format`: Format for reasoning in responses.
    - `reasoning_budget`: Budget for reasoning tasks.
    - `prefill_assistant`: Flag to prefill assistant messages.
    - `api_keys`: List of API keys for authentication.
    - `ssl_file_key`: Path to SSL key file.
    - `ssl_file_cert`: Path to SSL certificate file.
    - `webui`: Flag to enable web UI.
    - `endpoint_slots`: Flag to enable endpoint slots.
    - `endpoint_props`: Flag to control POST requests.
    - `endpoint_metrics`: Flag to enable endpoint metrics.
    - `log_json`: Flag to log in JSON format.
    - `slot_save_path`: Path to save slot data.
    - `slot_prompt_similarity`: Similarity threshold for slot prompts.
    - `is_pp_shared`: Flag for shared post-processing.
    - `n_pp`: List of post-processing parameters.
    - `n_tg`: List of target generation parameters.
    - `n_pl`: List of placeholder parameters.
    - `context_files`: List of context files for embedding.
    - `chunk_size`: Chunk size for context embedding.
    - `chunk_separator`: Separator for context embedding chunks.
    - `n_junk`: Number of times to repeat junk text.
    - `i_pos`: Position of passkey in junk text.
    - `n_out_freq`: Frequency of outputting the imatrix.
    - `n_save_freq`: Frequency of saving the imatrix.
    - `i_chunk`: Starting chunk for processing.
    - `process_output`: Flag to collect data for output tensor.
    - `compute_ppl`: Flag to compute perplexity.
    - `parse_special`: Flag to parse special tokens.
    - `n_pca_batch`: Batch size for PCA in cvector-generator.
    - `n_pca_iterations`: Number of PCA iterations in cvector-generator.
    - `cvector_dimre_method`: Dimensionality reduction method for cvector-generator.
    - `cvector_positive_file`: Path to positive file for cvector-generator.
    - `cvector_negative_file`: Path to negative file for cvector-generator.
    - `spm_infill`: Flag for infill pattern in SPM.
    - `batched_bench_output_jsonl`: Flag for JSONL output in batched-bench.
    - `out_file`: Output filename for example programs.
    - `load_progress_callback`: Callback for model loading progress.
    - `load_progress_callback_user_data`: User data for load progress callback.
- **Description**: The `common_params` struct is a comprehensive configuration structure used to manage various parameters and settings for a machine learning model, particularly in the context of natural language processing tasks. It includes fields for controlling prediction, context size, batch processing, and parallel decoding, as well as parameters for attention mechanisms, RoPE frequency, and YaRN factors. The struct also manages device offloading, GPU layer storage, and model splitting across GPUs. Additionally, it contains settings for sampling, speculative decoding, vocoder processing, and model management, along with various flags for controlling output, interaction, and performance metrics. The struct is designed to be highly configurable, allowing for detailed customization of model behavior and processing.


---
### common\_init\_result<!-- {{#data_structure:common_init_result}} -->
- **Type**: `struct`
- **Members**:
    - `model`: A pointer to a llama model.
    - `context`: A pointer to a llama context.
    - `lora`: A vector of pointers to llama adapter lora instances.
- **Description**: The `common_init_result` struct is designed to encapsulate the result of an initialization process involving a llama model and its context. It contains pointers to both the model and the context, as well as a collection of LoRA (Low-Rank Adaptation) adapters, which are used to modify or extend the model's capabilities. This struct is likely used to manage and store the state of a model and its associated resources after initialization.


---
### common\_control\_vector\_data<!-- {{#data_structure:common_control_vector_data}} -->
- **Type**: `struct`
- **Members**:
    - `n_embd`: An integer representing the number of embeddings.
    - `data`: A vector of floats storing data for layers, where the number of layers is determined by the size of the data divided by n_embd.
- **Description**: The `common_control_vector_data` struct is designed to store control vector data used in a layered model architecture. It contains an integer `n_embd` which specifies the number of embeddings, and a vector `data` that holds the actual data for the layers. The number of layers is implicitly defined by dividing the size of the `data` vector by `n_embd`, allowing for dynamic adjustment based on the data size.


---
### common\_control\_vector\_load\_info<!-- {{#data_structure:common_control_vector_load_info}} -->
- **Type**: `struct`
- **Members**:
    - `strength`: A float representing the strength of the control vector.
    - `fname`: A string representing the filename associated with the control vector.
- **Description**: The `common_control_vector_load_info` struct is used to store information about control vectors, specifically their strength and associated filename. This struct is likely used in the context of loading and managing control vectors, which are scaled by their strength and combined for further processing.


# Functions

---
### string\_split<!-- {{#callable:string_split}} -->
The `string_split` function template splits a given string into a vector of elements of type `T` using a specified delimiter, converting each substring to type `T` before adding it to the vector.
- **Inputs**:
    - `str`: A constant reference to the input string that needs to be split.
    - `delim`: A character that serves as the delimiter to split the string.
- **Control Flow**:
    - The function begins by asserting that the template type `T` is not `std::string`, as a specialized version exists for `std::string`.
    - An empty vector of type `T` is initialized to store the split values.
    - A string stream is created from the input string to facilitate reading tokens separated by the delimiter.
    - A loop iterates over the tokens obtained by splitting the input string using the delimiter.
    - For each token, a new string stream is created to convert the token to type `T`.
    - The converted value is added to the vector of type `T`.
    - After processing all tokens, the vector containing the split and converted values is returned.
- **Output**: A vector of type `T` containing the split and converted elements from the input string.


---
### string\_split<std::string><!-- {{#callable:string_split<std::string>}} -->
The `string_split<std::string>` function splits a given string into a vector of substrings based on a specified separator character.
- **Inputs**:
    - `input`: A constant reference to the input string that needs to be split.
    - `separator`: A character that serves as the delimiter for splitting the input string.
- **Control Flow**:
    - Initialize an empty vector `parts` to store the resulting substrings.
    - Set `begin_pos` to 0 to track the start of each substring.
    - Find the first occurrence of the separator in the input string using `input.find(separator)` and store the position in `separator_pos`.
    - Enter a while loop that continues as long as `separator_pos` is not `std::string::npos` (indicating the separator is found).
    - Within the loop, extract a substring from `begin_pos` to `separator_pos` and add it to the `parts` vector.
    - Update `begin_pos` to `separator_pos + 1` to move past the current separator.
    - Find the next occurrence of the separator starting from `begin_pos`.
    - After the loop, add the last substring (from `begin_pos` to the end of the string) to the `parts` vector.
    - Return the `parts` vector containing all the substrings.
- **Output**: A vector of strings, where each element is a substring of the input string split by the specified separator.


---
### string\_starts\_with<!-- {{#callable:string_starts_with}} -->
The `string_starts_with` function checks if a given string starts with a specified prefix.
- **Inputs**:
    - `str`: The main string to be checked.
    - `prefix`: The prefix to check against the start of the main string.
- **Control Flow**:
    - The function uses the `rfind` method of the `std::string` class to search for the prefix at the start of the string.
    - It checks if the position returned by `rfind` is 0, indicating that the prefix is found at the beginning of the string.
- **Output**: Returns `true` if the string starts with the specified prefix, otherwise returns `false`.


