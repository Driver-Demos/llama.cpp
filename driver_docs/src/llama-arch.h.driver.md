# Purpose
This C++ source code file defines a set of enumerations, structures, and functions related to handling constants and metadata for various large language model (LLM) architectures. The file includes enumerations for different LLM architectures (`llm_arch`), key-value pairs (`llm_kv`), and tensor types (`llm_tensor` and `llm_tensor_layer`). These enumerations provide a structured way to reference different components and configurations of LLMs, such as architecture types, tensor properties, and key-value metadata. The file also defines structures like `LLM_KV` and [`LLM_TN`](#LLM_TNLLM_TN) that facilitate the generation of standardized names for tensors and key-value pairs, which are crucial for managing and interacting with model components in a consistent manner.

The file serves as a header file intended to be included in other parts of a software project, providing a public API for accessing and manipulating LLM-related constants and metadata. It includes functions such as `llm_arch_name`, `llm_arch_from_string`, and `llm_tensor_info_for`, which offer utility for converting between string representations and enumerated types, as well as retrieving information about specific tensors. The inclusion of `#pragma once` indicates that this file is designed to be included only once per compilation unit, preventing duplicate definitions. Overall, the file provides a comprehensive framework for managing the diverse configurations and components of various LLM architectures, supporting the development and maintenance of software that interacts with these models.
# Imports and Dependencies

---
- `ggml.h`
- `string`


# Global Variables

---
### llm\_arch\_name
- **Type**: `const char *`
- **Description**: The `llm_arch_name` is a function that takes an `llm_arch` enumeration value as an argument and returns a constant character pointer. This function is likely used to map the architecture enumeration to a human-readable string representation of the architecture name.
- **Use**: This function is used to retrieve the name of a specific architecture as a string based on its enumeration value.


---
### llm\_tensor\_info\_for
- **Type**: `const llm_tensor_info &`
- **Description**: The `llm_tensor_info_for` is a function that returns a constant reference to an `llm_tensor_info` structure. This structure contains information about a specific tensor, including its layer and operation type, as defined by the `llm_tensor` enumeration.
- **Use**: This function is used to retrieve detailed information about a tensor, such as its layer and operation, based on the tensor type provided as an argument.


# Data Structures

---
### llm\_arch<!-- {{#data_structure:llm_arch}} -->
- **Type**: `enum`
- **Members**:
    - `LLM_ARCH_LLAMA`: Represents the LLAMA architecture.
    - `LLM_ARCH_LLAMA4`: Represents the LLAMA4 architecture.
    - `LLM_ARCH_DECI`: Represents the DECI architecture.
    - `LLM_ARCH_FALCON`: Represents the FALCON architecture.
    - `LLM_ARCH_BAICHUAN`: Represents the BAICHUAN architecture.
    - `LLM_ARCH_GROK`: Represents the GROK architecture.
    - `LLM_ARCH_GPT2`: Represents the GPT2 architecture.
    - `LLM_ARCH_GPTJ`: Represents the GPTJ architecture.
    - `LLM_ARCH_GPTNEOX`: Represents the GPTNEOX architecture.
    - `LLM_ARCH_MPT`: Represents the MPT architecture.
    - `LLM_ARCH_STARCODER`: Represents the STARCODER architecture.
    - `LLM_ARCH_REFACT`: Represents the REFACT architecture.
    - `LLM_ARCH_BERT`: Represents the BERT architecture.
    - `LLM_ARCH_NOMIC_BERT`: Represents the NOMIC_BERT architecture.
    - `LLM_ARCH_NOMIC_BERT_MOE`: Represents the NOMIC_BERT_MOE architecture.
    - `LLM_ARCH_JINA_BERT_V2`: Represents the JINA_BERT_V2 architecture.
    - `LLM_ARCH_BLOOM`: Represents the BLOOM architecture.
    - `LLM_ARCH_STABLELM`: Represents the STABLELM architecture.
    - `LLM_ARCH_QWEN`: Represents the QWEN architecture.
    - `LLM_ARCH_QWEN2`: Represents the QWEN2 architecture.
    - `LLM_ARCH_QWEN2MOE`: Represents the QWEN2MOE architecture.
    - `LLM_ARCH_QWEN2VL`: Represents the QWEN2VL architecture.
    - `LLM_ARCH_QWEN3`: Represents the QWEN3 architecture.
    - `LLM_ARCH_QWEN3MOE`: Represents the QWEN3MOE architecture.
    - `LLM_ARCH_PHI2`: Represents the PHI2 architecture.
    - `LLM_ARCH_PHI3`: Represents the PHI3 architecture.
    - `LLM_ARCH_PHIMOE`: Represents the PHIMOE architecture.
    - `LLM_ARCH_PLAMO`: Represents the PLAMO architecture.
    - `LLM_ARCH_CODESHELL`: Represents the CODESHELL architecture.
    - `LLM_ARCH_ORION`: Represents the ORION architecture.
    - `LLM_ARCH_INTERNLM2`: Represents the INTERNLM2 architecture.
    - `LLM_ARCH_MINICPM`: Represents the MINICPM architecture.
    - `LLM_ARCH_MINICPM3`: Represents the MINICPM3 architecture.
    - `LLM_ARCH_GEMMA`: Represents the GEMMA architecture.
    - `LLM_ARCH_GEMMA2`: Represents the GEMMA2 architecture.
    - `LLM_ARCH_GEMMA3`: Represents the GEMMA3 architecture.
    - `LLM_ARCH_STARCODER2`: Represents the STARCODER2 architecture.
    - `LLM_ARCH_MAMBA`: Represents the MAMBA architecture.
    - `LLM_ARCH_XVERSE`: Represents the XVERSE architecture.
    - `LLM_ARCH_COMMAND_R`: Represents the COMMAND_R architecture.
    - `LLM_ARCH_COHERE2`: Represents the COHERE2 architecture.
    - `LLM_ARCH_DBRX`: Represents the DBRX architecture.
    - `LLM_ARCH_OLMO`: Represents the OLMO architecture.
    - `LLM_ARCH_OLMO2`: Represents the OLMO2 architecture.
    - `LLM_ARCH_OLMOE`: Represents the OLMOE architecture.
    - `LLM_ARCH_OPENELM`: Represents the OPENELM architecture.
    - `LLM_ARCH_ARCTIC`: Represents the ARCTIC architecture.
    - `LLM_ARCH_DEEPSEEK`: Represents the DEEPSEEK architecture.
    - `LLM_ARCH_DEEPSEEK2`: Represents the DEEPSEEK2 architecture.
    - `LLM_ARCH_CHATGLM`: Represents the CHATGLM architecture.
    - `LLM_ARCH_GLM4`: Represents the GLM4 architecture.
    - `LLM_ARCH_BITNET`: Represents the BITNET architecture.
    - `LLM_ARCH_T5`: Represents the T5 architecture.
    - `LLM_ARCH_T5ENCODER`: Represents the T5ENCODER architecture.
    - `LLM_ARCH_JAIS`: Represents the JAIS architecture.
    - `LLM_ARCH_NEMOTRON`: Represents the NEMOTRON architecture.
    - `LLM_ARCH_EXAONE`: Represents the EXAONE architecture.
    - `LLM_ARCH_RWKV6`: Represents the RWKV6 architecture.
    - `LLM_ARCH_RWKV6QWEN2`: Represents the RWKV6QWEN2 architecture.
    - `LLM_ARCH_RWKV7`: Represents the RWKV7 architecture.
    - `LLM_ARCH_ARWKV7`: Represents the ARWKV7 architecture.
    - `LLM_ARCH_GRANITE`: Represents the GRANITE architecture.
    - `LLM_ARCH_GRANITE_MOE`: Represents the GRANITE_MOE architecture.
    - `LLM_ARCH_CHAMELEON`: Represents the CHAMELEON architecture.
    - `LLM_ARCH_WAVTOKENIZER_DEC`: Represents the WAVTOKENIZER_DEC architecture.
    - `LLM_ARCH_PLM`: Represents the PLM architecture.
    - `LLM_ARCH_BAILINGMOE`: Represents the BAILINGMOE architecture.
    - `LLM_ARCH_UNKNOWN`: Represents an unknown architecture.
- **Description**: The `llm_arch` enum defines a comprehensive list of architecture identifiers for various large language models (LLMs). Each enumerator within `llm_arch` corresponds to a specific architecture, such as LLAMA, GPT2, BERT, and many others, including both well-known and potentially proprietary or experimental architectures. This enumeration is used to categorize and identify different LLM architectures within a software system, facilitating operations that depend on the specific characteristics or capabilities of these models.


---
### llm\_kv<!-- {{#data_structure:llm_kv}} -->
- **Type**: `enum`
- **Members**:
    - `LLM_KV_GENERAL_TYPE`: Represents the general type of the LLM.
    - `LLM_KV_GENERAL_ARCHITECTURE`: Specifies the architecture of the LLM.
    - `LLM_KV_GENERAL_QUANTIZATION_VERSION`: Indicates the version of quantization used.
    - `LLM_KV_GENERAL_ALIGNMENT`: Defines the alignment of the LLM.
    - `LLM_KV_GENERAL_FILE_TYPE`: Specifies the file type associated with the LLM.
    - `LLM_KV_GENERAL_NAME`: Holds the name of the LLM.
    - `LLM_KV_GENERAL_AUTHOR`: Contains the author's name of the LLM.
    - `LLM_KV_GENERAL_VERSION`: Indicates the version of the LLM.
    - `LLM_KV_GENERAL_URL`: Provides the URL related to the LLM.
    - `LLM_KV_GENERAL_DESCRIPTION`: Contains a description of the LLM.
    - `LLM_KV_GENERAL_LICENSE`: Specifies the license under which the LLM is released.
    - `LLM_KV_GENERAL_SOURCE_URL`: Provides the source URL for the LLM.
    - `LLM_KV_GENERAL_SOURCE_HF_REPO`: Indicates the source repository on Hugging Face.
    - `LLM_KV_VOCAB_SIZE`: Specifies the vocabulary size of the LLM.
    - `LLM_KV_CONTEXT_LENGTH`: Defines the context length for the LLM.
    - `LLM_KV_EMBEDDING_LENGTH`: Indicates the length of embeddings used.
    - `LLM_KV_FEATURES_LENGTH`: Specifies the length of features in the LLM.
    - `LLM_KV_BLOCK_COUNT`: Defines the number of blocks in the LLM.
    - `LLM_KV_LEADING_DENSE_BLOCK_COUNT`: Indicates the count of leading dense blocks.
    - `LLM_KV_FEED_FORWARD_LENGTH`: Specifies the length of the feed-forward network.
    - `LLM_KV_EXPERT_FEED_FORWARD_LENGTH`: Defines the length of the expert feed-forward network.
    - `LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH`: Indicates the length of the shared expert feed-forward network.
    - `LLM_KV_USE_PARALLEL_RESIDUAL`: Specifies whether parallel residuals are used.
    - `LLM_KV_TENSOR_DATA_LAYOUT`: Defines the data layout of tensors.
    - `LLM_KV_EXPERT_COUNT`: Indicates the number of experts in the LLM.
    - `LLM_KV_EXPERT_USED_COUNT`: Specifies the count of used experts.
    - `LLM_KV_EXPERT_SHARED_COUNT`: Defines the count of shared experts.
    - `LLM_KV_EXPERT_WEIGHTS_SCALE`: Indicates the scale of expert weights.
    - `LLM_KV_EXPERT_WEIGHTS_NORM`: Specifies the normalization of expert weights.
    - `LLM_KV_EXPERT_GATING_FUNC`: Defines the gating function for experts.
    - `LLM_KV_MOE_EVERY_N_LAYERS`: Indicates the frequency of MoE layers.
    - `LLM_KV_POOLING_TYPE`: Specifies the type of pooling used.
    - `LLM_KV_LOGIT_SCALE`: Defines the scale of logits.
    - `LLM_KV_DECODER_START_TOKEN_ID`: Indicates the start token ID for the decoder.
    - `LLM_KV_ATTN_LOGIT_SOFTCAPPING`: Specifies the softcapping of attention logits.
    - `LLM_KV_FINAL_LOGIT_SOFTCAPPING`: Defines the softcapping of final logits.
    - `LLM_KV_SWIN_NORM`: Indicates the normalization used in SWIN.
    - `LLM_KV_RESCALE_EVERY_N_LAYERS`: Specifies the rescaling frequency across layers.
    - `LLM_KV_TIME_MIX_EXTRA_DIM`: Defines the extra dimension for time mixing.
    - `LLM_KV_TIME_DECAY_EXTRA_DIM`: Indicates the extra dimension for time decay.
    - `LLM_KV_RESIDUAL_SCALE`: Specifies the scale of residuals.
    - `LLM_KV_EMBEDDING_SCALE`: Defines the scale of embeddings.
    - `LLM_KV_TOKEN_SHIFT_COUNT`: Indicates the count of token shifts.
    - `LLM_KV_INTERLEAVE_MOE_LAYER_STEP`: Specifies the step for interleaving MoE layers.
    - `LLM_KV_ATTENTION_HEAD_COUNT`: Defines the number of attention heads.
    - `LLM_KV_ATTENTION_HEAD_COUNT_KV`: Indicates the count of key-value attention heads.
    - `LLM_KV_ATTENTION_MAX_ALIBI_BIAS`: Specifies the maximum ALiBi bias for attention.
    - `LLM_KV_ATTENTION_CLAMP_KQV`: Defines the clamping for KQV in attention.
    - `LLM_KV_ATTENTION_KEY_LENGTH`: Indicates the length of attention keys.
    - `LLM_KV_ATTENTION_VALUE_LENGTH`: Specifies the length of attention values.
    - `LLM_KV_ATTENTION_LAYERNORM_EPS`: Defines the epsilon for layer normalization in attention.
    - `LLM_KV_ATTENTION_LAYERNORM_RMS_EPS`: Indicates the RMS epsilon for layer normalization in attention.
    - `LLM_KV_ATTENTION_GROUPNORM_EPS`: Specifies the epsilon for group normalization in attention.
    - `LLM_KV_ATTENTION_GROUPNORM_GROUPS`: Defines the number of groups for group normalization in attention.
    - `LLM_KV_ATTENTION_CAUSAL`: Indicates if the attention is causal.
    - `LLM_KV_ATTENTION_Q_LORA_RANK`: Specifies the LoRA rank for Q in attention.
    - `LLM_KV_ATTENTION_KV_LORA_RANK`: Defines the LoRA rank for KV in attention.
    - `LLM_KV_ATTENTION_DECAY_LORA_RANK`: Indicates the decay LoRA rank in attention.
    - `LLM_KV_ATTENTION_ICLR_LORA_RANK`: Specifies the ICLR LoRA rank in attention.
    - `LLM_KV_ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK`: Defines the LoRA rank for value residual mix in attention.
    - `LLM_KV_ATTENTION_GATE_LORA_RANK`: Indicates the LoRA rank for gate in attention.
    - `LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT`: Specifies the count of relative buckets in attention.
    - `LLM_KV_ATTENTION_SLIDING_WINDOW`: Defines the sliding window size for attention.
    - `LLM_KV_ATTENTION_SCALE`: Indicates the scale of attention.
    - `LLM_KV_ATTENTION_KEY_LENGTH_MLA`: Specifies the key length for MLA in attention.
    - `LLM_KV_ATTENTION_VALUE_LENGTH_MLA`: Defines the value length for MLA in attention.
    - `LLM_KV_ROPE_DIMENSION_COUNT`: Indicates the count of dimensions for ROPE.
    - `LLM_KV_ROPE_DIMENSION_SECTIONS`: Specifies the sections of dimensions for ROPE.
    - `LLM_KV_ROPE_FREQ_BASE`: Defines the frequency base for ROPE.
    - `LLM_KV_ROPE_SCALE_LINEAR`: Indicates the linear scale for ROPE.
    - `LLM_KV_ROPE_SCALING_TYPE`: Specifies the type of scaling for ROPE.
    - `LLM_KV_ROPE_SCALING_FACTOR`: Defines the scaling factor for ROPE.
    - `LLM_KV_ROPE_SCALING_ATTN_FACTOR`: Indicates the attention scaling factor for ROPE.
    - `LLM_KV_ROPE_SCALING_ORIG_CTX_LEN`: Specifies the original context length for ROPE scaling.
    - `LLM_KV_ROPE_SCALING_FINETUNED`: Defines if ROPE scaling is fine-tuned.
    - `LLM_KV_ROPE_SCALING_YARN_LOG_MUL`: Indicates the YARN log multiplier for ROPE scaling.
    - `LLM_KV_SPLIT_NO`: Specifies the split number.
    - `LLM_KV_SPLIT_COUNT`: Defines the count of splits.
    - `LLM_KV_SPLIT_TENSORS_COUNT`: Indicates the count of split tensors.
    - `LLM_KV_SSM_INNER_SIZE`: Specifies the inner size for SSM.
    - `LLM_KV_SSM_CONV_KERNEL`: Defines the convolution kernel for SSM.
    - `LLM_KV_SSM_STATE_SIZE`: Indicates the state size for SSM.
    - `LLM_KV_SSM_TIME_STEP_RANK`: Specifies the time step rank for SSM.
    - `LLM_KV_SSM_DT_B_C_RMS`: Defines the DT B C RMS for SSM.
    - `LLM_KV_WKV_HEAD_SIZE`: Indicates the head size for WKV.
    - `LLM_KV_TOKENIZER_MODEL`: Specifies the tokenizer model.
    - `LLM_KV_TOKENIZER_PRE`: Defines the pre-tokenizer.
    - `LLM_KV_TOKENIZER_LIST`: Indicates the list of tokenizers.
    - `LLM_KV_TOKENIZER_TOKEN_TYPE`: Specifies the token type for the tokenizer.
    - `LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT`: Defines the count of token types for the tokenizer.
    - `LLM_KV_TOKENIZER_SCORES`: Indicates the scores for the tokenizer.
    - `LLM_KV_TOKENIZER_MERGES`: Specifies the merges for the tokenizer.
    - `LLM_KV_TOKENIZER_BOS_ID`: Defines the BOS ID for the tokenizer.
    - `LLM_KV_TOKENIZER_EOS_ID`: Indicates the EOS ID for the tokenizer.
    - `LLM_KV_TOKENIZER_EOT_ID`: Specifies the EOT ID for the tokenizer.
    - `LLM_KV_TOKENIZER_EOM_ID`: Defines the EOM ID for the tokenizer.
    - `LLM_KV_TOKENIZER_UNK_ID`: Indicates the UNK ID for the tokenizer.
    - `LLM_KV_TOKENIZER_SEP_ID`: Specifies the SEP ID for the tokenizer.
    - `LLM_KV_TOKENIZER_PAD_ID`: Defines the PAD ID for the tokenizer.
    - `LLM_KV_TOKENIZER_CLS_ID`: Indicates the CLS ID for the tokenizer.
    - `LLM_KV_TOKENIZER_MASK_ID`: Specifies the MASK ID for the tokenizer.
    - `LLM_KV_TOKENIZER_ADD_BOS`: Defines if BOS is added by the tokenizer.
    - `LLM_KV_TOKENIZER_ADD_EOS`: Indicates if EOS is added by the tokenizer.
    - `LLM_KV_TOKENIZER_ADD_PREFIX`: Specifies if a prefix is added by the tokenizer.
    - `LLM_KV_TOKENIZER_REMOVE_EXTRA_WS`: Defines if extra whitespace is removed by the tokenizer.
    - `LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP`: Indicates the precompiled character map for the tokenizer.
    - `LLM_KV_TOKENIZER_HF_JSON`: Specifies the Hugging Face JSON for the tokenizer.
    - `LLM_KV_TOKENIZER_RWKV`: Defines the RWKV for the tokenizer.
    - `LLM_KV_TOKENIZER_CHAT_TEMPLATE`: Indicates the chat template for the tokenizer.
    - `LLM_KV_TOKENIZER_CHAT_TEMPLATE_N`: Specifies the chat template number for the tokenizer.
    - `LLM_KV_TOKENIZER_FIM_PRE_ID`: Defines the FIM pre ID for the tokenizer.
    - `LLM_KV_TOKENIZER_FIM_SUF_ID`: Indicates the FIM suffix ID for the tokenizer.
    - `LLM_KV_TOKENIZER_FIM_MID_ID`: Specifies the FIM middle ID for the tokenizer.
    - `LLM_KV_TOKENIZER_FIM_PAD_ID`: Defines the FIM pad ID for the tokenizer.
    - `LLM_KV_TOKENIZER_FIM_REP_ID`: Indicates the FIM repeat ID for the tokenizer.
    - `LLM_KV_TOKENIZER_FIM_SEP_ID`: Specifies the FIM separator ID for the tokenizer.
    - `LLM_KV_ADAPTER_TYPE`: Defines the adapter type.
    - `LLM_KV_ADAPTER_LORA_ALPHA`: Indicates the LoRA alpha for the adapter.
    - `LLM_KV_POSNET_EMBEDDING_LENGTH`: Specifies the embedding length for PosNet.
    - `LLM_KV_POSNET_BLOCK_COUNT`: Defines the block count for PosNet.
    - `LLM_KV_CONVNEXT_EMBEDDING_LENGTH`: Indicates the embedding length for ConvNext.
    - `LLM_KV_CONVNEXT_BLOCK_COUNT`: Specifies the block count for ConvNext.
    - `LLM_KV_CLASSIFIER_OUTPUT_LABELS`: Defines the output labels for the classifier.
    - `LLM_KV_TOKENIZER_PREFIX_ID`: Deprecated: Specifies the prefix ID for the tokenizer.
    - `LLM_KV_TOKENIZER_SUFFIX_ID`: Deprecated: Indicates the suffix ID for the tokenizer.
    - `LLM_KV_TOKENIZER_MIDDLE_ID`: Deprecated: Specifies the middle ID for the tokenizer.
- **Description**: The `llm_kv` enum is a comprehensive enumeration that defines a wide range of key-value identifiers used in the context of large language models (LLMs). These identifiers cover various aspects of LLMs, including general properties like type, architecture, and version, as well as specific configurations related to vocabulary, context, embedding, attention mechanisms, and tokenization. The enum also includes identifiers for deprecated features, ensuring backward compatibility. This structure is essential for managing and accessing different parameters and settings within LLM systems, facilitating the organization and retrieval of model-specific information.


---
### llm\_tensor<!-- {{#data_structure:llm_tensor}} -->
- **Type**: `enum`
- **Members**:
    - `LLM_TENSOR_TOKEN_EMBD`: Represents the token embedding tensor.
    - `LLM_TENSOR_TOKEN_EMBD_NORM`: Represents the normalized token embedding tensor.
    - `LLM_TENSOR_TOKEN_TYPES`: Represents the tensor for token types.
    - `LLM_TENSOR_POS_EMBD`: Represents the positional embedding tensor.
    - `LLM_TENSOR_OUTPUT`: Represents the output tensor.
    - `LLM_TENSOR_OUTPUT_NORM`: Represents the normalized output tensor.
    - `LLM_TENSOR_ROPE_FREQS`: Represents the tensor for ROPE frequencies.
    - `LLM_TENSOR_ROPE_FACTORS_LONG`: Represents the tensor for long ROPE factors.
    - `LLM_TENSOR_ROPE_FACTORS_SHORT`: Represents the tensor for short ROPE factors.
    - `LLM_TENSOR_ATTN_Q`: Represents the attention query tensor.
    - `LLM_TENSOR_ATTN_K`: Represents the attention key tensor.
    - `LLM_TENSOR_ATTN_V`: Represents the attention value tensor.
    - `LLM_TENSOR_ATTN_QKV`: Represents the combined attention query, key, and value tensor.
    - `LLM_TENSOR_ATTN_OUT`: Represents the attention output tensor.
    - `LLM_TENSOR_ATTN_NORM`: Represents the normalized attention tensor.
    - `LLM_TENSOR_ATTN_NORM_2`: Represents a secondary normalized attention tensor.
    - `LLM_TENSOR_ATTN_OUT_NORM`: Represents the normalized attention output tensor.
    - `LLM_TENSOR_ATTN_POST_NORM`: Represents the post-normalization attention tensor.
    - `LLM_TENSOR_ATTN_ROT_EMBD`: Represents the rotational embedding for attention.
    - `LLM_TENSOR_FFN_GATE_INP`: Represents the feed-forward network gate input tensor.
    - `LLM_TENSOR_FFN_GATE_INP_SHEXP`: Represents the shared expert feed-forward network gate input tensor.
    - `LLM_TENSOR_FFN_NORM`: Represents the normalized feed-forward network tensor.
    - `LLM_TENSOR_FFN_POST_NORM`: Represents the post-normalization feed-forward network tensor.
    - `LLM_TENSOR_FFN_GATE`: Represents the feed-forward network gate tensor.
    - `LLM_TENSOR_FFN_DOWN`: Represents the feed-forward network down-scaling tensor.
    - `LLM_TENSOR_FFN_UP`: Represents the feed-forward network up-scaling tensor.
    - `LLM_TENSOR_FFN_ACT`: Represents the feed-forward network activation tensor.
    - `LLM_TENSOR_FFN_DOWN_EXP`: Represents the expert down-scaling tensor for backward compatibility.
    - `LLM_TENSOR_FFN_GATE_EXP`: Represents the expert gate tensor.
    - `LLM_TENSOR_FFN_UP_EXP`: Represents the expert up-scaling tensor.
    - `LLM_TENSOR_FFN_NORM_EXPS`: Represents the normalized expert feed-forward network tensor.
    - `LLM_TENSOR_FFN_DOWN_EXPS`: Represents the merged expert down-scaling tensor.
    - `LLM_TENSOR_FFN_GATE_EXPS`: Represents the merged expert gate tensor.
    - `LLM_TENSOR_FFN_UP_EXPS`: Represents the merged expert up-scaling tensor.
    - `LLM_TENSOR_FFN_DOWN_SHEXP`: Represents the shared expert down-scaling tensor.
    - `LLM_TENSOR_FFN_GATE_SHEXP`: Represents the shared expert gate tensor.
    - `LLM_TENSOR_FFN_UP_SHEXP`: Represents the shared expert up-scaling tensor.
    - `LLM_TENSOR_FFN_EXP_PROBS_B`: Represents the expert probabilities tensor.
    - `LLM_TENSOR_ATTN_Q_NORM`: Represents the normalized attention query tensor.
    - `LLM_TENSOR_ATTN_K_NORM`: Represents the normalized attention key tensor.
    - `LLM_TENSOR_LAYER_OUT_NORM`: Represents the normalized layer output tensor.
    - `LLM_TENSOR_POST_ATTN_NORM`: Represents the post-attention normalization tensor.
    - `LLM_TENSOR_POST_MLP_NORM`: Represents the post-MLP normalization tensor.
    - `LLM_TENSOR_SSM_IN`: Represents the SSM input tensor.
    - `LLM_TENSOR_SSM_CONV1D`: Represents the SSM 1D convolution tensor.
    - `LLM_TENSOR_SSM_X`: Represents the SSM X tensor.
    - `LLM_TENSOR_SSM_DT`: Represents the SSM DT tensor.
    - `LLM_TENSOR_SSM_A`: Represents the SSM A tensor.
    - `LLM_TENSOR_SSM_D`: Represents the SSM D tensor.
    - `LLM_TENSOR_SSM_OUT`: Represents the SSM output tensor.
    - `LLM_TENSOR_TIME_MIX_W0`: Represents the time mix W0 tensor.
    - `LLM_TENSOR_TIME_MIX_W1`: Represents the time mix W1 tensor.
    - `LLM_TENSOR_TIME_MIX_W2`: Represents the time mix W2 tensor.
    - `LLM_TENSOR_TIME_MIX_A0`: Represents the time mix A0 tensor.
    - `LLM_TENSOR_TIME_MIX_A1`: Represents the time mix A1 tensor.
    - `LLM_TENSOR_TIME_MIX_A2`: Represents the time mix A2 tensor.
    - `LLM_TENSOR_TIME_MIX_V0`: Represents the time mix V0 tensor.
    - `LLM_TENSOR_TIME_MIX_V1`: Represents the time mix V1 tensor.
    - `LLM_TENSOR_TIME_MIX_V2`: Represents the time mix V2 tensor.
    - `LLM_TENSOR_TIME_MIX_G1`: Represents the time mix G1 tensor.
    - `LLM_TENSOR_TIME_MIX_G2`: Represents the time mix G2 tensor.
    - `LLM_TENSOR_TIME_MIX_K_K`: Represents the time mix K_K tensor.
    - `LLM_TENSOR_TIME_MIX_K_A`: Represents the time mix K_A tensor.
    - `LLM_TENSOR_TIME_MIX_R_K`: Represents the time mix R_K tensor.
    - `LLM_TENSOR_TIME_MIX_LERP_X`: Represents the time mix LERP X tensor.
    - `LLM_TENSOR_TIME_MIX_LERP_W`: Represents the time mix LERP W tensor.
    - `LLM_TENSOR_TIME_MIX_LERP_K`: Represents the time mix LERP K tensor.
    - `LLM_TENSOR_TIME_MIX_LERP_V`: Represents the time mix LERP V tensor.
    - `LLM_TENSOR_TIME_MIX_LERP_R`: Represents the time mix LERP R tensor.
    - `LLM_TENSOR_TIME_MIX_LERP_G`: Represents the time mix LERP G tensor.
    - `LLM_TENSOR_TIME_MIX_LERP_FUSED`: Represents the fused time mix LERP tensor.
    - `LLM_TENSOR_TIME_MIX_FIRST`: Represents the first time mix tensor.
    - `LLM_TENSOR_TIME_MIX_DECAY`: Represents the time mix decay tensor.
    - `LLM_TENSOR_TIME_MIX_DECAY_W1`: Represents the time mix decay W1 tensor.
    - `LLM_TENSOR_TIME_MIX_DECAY_W2`: Represents the time mix decay W2 tensor.
    - `LLM_TENSOR_TIME_MIX_KEY`: Represents the time mix key tensor.
    - `LLM_TENSOR_TIME_MIX_VALUE`: Represents the time mix value tensor.
    - `LLM_TENSOR_TIME_MIX_RECEPTANCE`: Represents the time mix receptance tensor.
    - `LLM_TENSOR_TIME_MIX_GATE`: Represents the time mix gate tensor.
    - `LLM_TENSOR_TIME_MIX_LN`: Represents the time mix layer normalization tensor.
    - `LLM_TENSOR_TIME_MIX_OUTPUT`: Represents the time mix output tensor.
    - `LLM_TENSOR_CHANNEL_MIX_LERP_K`: Represents the channel mix LERP K tensor.
    - `LLM_TENSOR_CHANNEL_MIX_LERP_R`: Represents the channel mix LERP R tensor.
    - `LLM_TENSOR_CHANNEL_MIX_KEY`: Represents the channel mix key tensor.
    - `LLM_TENSOR_CHANNEL_MIX_RECEPTANCE`: Represents the channel mix receptance tensor.
    - `LLM_TENSOR_CHANNEL_MIX_VALUE`: Represents the channel mix value tensor.
    - `LLM_TENSOR_ATTN_Q_A`: Represents the attention query A tensor.
    - `LLM_TENSOR_ATTN_Q_B`: Represents the attention query B tensor.
    - `LLM_TENSOR_ATTN_KV_A_MQA`: Represents the attention key-value A MQA tensor.
    - `LLM_TENSOR_ATTN_KV_B`: Represents the attention key-value B tensor.
    - `LLM_TENSOR_ATTN_K_B`: Represents the attention key B tensor.
    - `LLM_TENSOR_ATTN_V_B`: Represents the attention value B tensor.
    - `LLM_TENSOR_ATTN_Q_A_NORM`: Represents the normalized attention query A tensor.
    - `LLM_TENSOR_ATTN_KV_A_NORM`: Represents the normalized attention key-value A tensor.
    - `LLM_TENSOR_ATTN_SUB_NORM`: Represents the sub-normalized attention tensor.
    - `LLM_TENSOR_FFN_SUB_NORM`: Represents the sub-normalized feed-forward network tensor.
    - `LLM_TENSOR_DEC_ATTN_NORM`: Represents the decoder attention normalization tensor.
    - `LLM_TENSOR_DEC_ATTN_Q`: Represents the decoder attention query tensor.
    - `LLM_TENSOR_DEC_ATTN_K`: Represents the decoder attention key tensor.
    - `LLM_TENSOR_DEC_ATTN_V`: Represents the decoder attention value tensor.
    - `LLM_TENSOR_DEC_ATTN_OUT`: Represents the decoder attention output tensor.
    - `LLM_TENSOR_DEC_ATTN_REL_B`: Represents the decoder attention relative B tensor.
    - `LLM_TENSOR_DEC_CROSS_ATTN_NORM`: Represents the decoder cross-attention normalization tensor.
    - `LLM_TENSOR_DEC_CROSS_ATTN_Q`: Represents the decoder cross-attention query tensor.
    - `LLM_TENSOR_DEC_CROSS_ATTN_K`: Represents the decoder cross-attention key tensor.
    - `LLM_TENSOR_DEC_CROSS_ATTN_V`: Represents the decoder cross-attention value tensor.
    - `LLM_TENSOR_DEC_CROSS_ATTN_OUT`: Represents the decoder cross-attention output tensor.
    - `LLM_TENSOR_DEC_CROSS_ATTN_REL_B`: Represents the decoder cross-attention relative B tensor.
    - `LLM_TENSOR_DEC_FFN_NORM`: Represents the decoder feed-forward network normalization tensor.
    - `LLM_TENSOR_DEC_FFN_GATE`: Represents the decoder feed-forward network gate tensor.
    - `LLM_TENSOR_DEC_FFN_DOWN`: Represents the decoder feed-forward network down-scaling tensor.
    - `LLM_TENSOR_DEC_FFN_UP`: Represents the decoder feed-forward network up-scaling tensor.
    - `LLM_TENSOR_DEC_OUTPUT_NORM`: Represents the decoder output normalization tensor.
    - `LLM_TENSOR_ENC_ATTN_NORM`: Represents the encoder attention normalization tensor.
    - `LLM_TENSOR_ENC_ATTN_Q`: Represents the encoder attention query tensor.
    - `LLM_TENSOR_ENC_ATTN_K`: Represents the encoder attention key tensor.
    - `LLM_TENSOR_ENC_ATTN_V`: Represents the encoder attention value tensor.
    - `LLM_TENSOR_ENC_ATTN_OUT`: Represents the encoder attention output tensor.
    - `LLM_TENSOR_ENC_ATTN_REL_B`: Represents the encoder attention relative B tensor.
    - `LLM_TENSOR_ENC_FFN_NORM`: Represents the encoder feed-forward network normalization tensor.
    - `LLM_TENSOR_ENC_FFN_GATE`: Represents the encoder feed-forward network gate tensor.
    - `LLM_TENSOR_ENC_FFN_DOWN`: Represents the encoder feed-forward network down-scaling tensor.
    - `LLM_TENSOR_ENC_FFN_UP`: Represents the encoder feed-forward network up-scaling tensor.
    - `LLM_TENSOR_ENC_OUTPUT_NORM`: Represents the encoder output normalization tensor.
    - `LLM_TENSOR_CLS`: Represents the classification tensor.
    - `LLM_TENSOR_CLS_OUT`: Represents the classification output tensor.
    - `LLM_TENSOR_CONV1D`: Represents the 1D convolution tensor.
    - `LLM_TENSOR_CONVNEXT_DW`: Represents the ConvNext depth-wise tensor.
    - `LLM_TENSOR_CONVNEXT_NORM`: Represents the ConvNext normalization tensor.
    - `LLM_TENSOR_CONVNEXT_PW1`: Represents the ConvNext point-wise 1 tensor.
    - `LLM_TENSOR_CONVNEXT_PW2`: Represents the ConvNext point-wise 2 tensor.
    - `LLM_TENSOR_CONVNEXT_GAMMA`: Represents the ConvNext gamma tensor.
    - `LLM_TENSOR_POS_NET_CONV1`: Represents the positional network convolution 1 tensor.
    - `LLM_TENSOR_POS_NET_CONV2`: Represents the positional network convolution 2 tensor.
    - `LLM_TENSOR_POS_NET_NORM`: Represents the positional network normalization tensor.
    - `LLM_TENSOR_POS_NET_NORM1`: Represents the first positional network normalization tensor.
    - `LLM_TENSOR_POS_NET_NORM2`: Represents the second positional network normalization tensor.
    - `LLM_TENSOR_POS_NET_ATTN_NORM`: Represents the positional network attention normalization tensor.
    - `LLM_TENSOR_POS_NET_ATTN_Q`: Represents the positional network attention query tensor.
    - `LLM_TENSOR_POS_NET_ATTN_K`: Represents the positional network attention key tensor.
    - `LLM_TENSOR_POS_NET_ATTN_V`: Represents the positional network attention value tensor.
    - `LLM_TENSOR_POS_NET_ATTN_OUT`: Represents the positional network attention output tensor.
- **Description**: The `llm_tensor` enum defines a comprehensive set of tensor types used in large language models (LLMs). Each enumerator represents a specific type of tensor that is utilized in various stages of the model's architecture, such as embeddings, attention mechanisms, feed-forward networks, and normalization processes. This enum is crucial for managing and referencing the different tensor components within the model, ensuring that each tensor is correctly identified and utilized in the model's operations. The enum covers a wide range of tensor types, reflecting the complexity and diversity of operations in modern LLM architectures.


---
### llm\_tensor\_layer<!-- {{#data_structure:llm_tensor_layer}} -->
- **Type**: `enum`
- **Members**:
    - `LLM_TENSOR_LAYER_INPUT`: Represents the input layer of a tensor.
    - `LLM_TENSOR_LAYER_REPEATING`: Represents a repeating layer within a tensor.
    - `LLM_TENSOR_LAYER_OUTPUT`: Represents the output layer of a tensor.
- **Description**: The `llm_tensor_layer` enum defines different types of layers that can be part of a tensor in a machine learning model. It categorizes the layers into three types: input, repeating, and output, which are essential for structuring the flow of data through the model's architecture.


---
### LLM\_KV<!-- {{#data_structure:LLM_KV}} -->
- **Type**: `struct`
- **Members**:
    - `arch`: Represents the architecture type of the LLM, defined by the `llm_arch` enum.
    - `suffix`: An optional string suffix that can be appended to the formatted output.
- **Description**: The `LLM_KV` struct is designed to handle key-value operations related to large language models (LLMs). It encapsulates an architecture type, specified by the `llm_arch` enum, and an optional suffix string. The struct provides a callable operator that formats and returns a string representation of a given key-value pair, using the architecture and suffix information. This struct is useful for generating standardized names or identifiers for various components of LLMs based on their architecture and specific key-value attributes.
- **Member Functions**:
    - [`LLM_KV::LLM_KV`](llama-arch.cpp.driver.md#LLM_KVLLM_KV)
    - [`LLM_KV::operator()`](llama-arch.cpp.driver.md#LLM_KVoperator())


---
### LLM\_TN\_IMPL<!-- {{#data_structure:LLM_TN_IMPL}} -->
- **Type**: `struct`
- **Members**:
    - `arch`: Represents the architecture type of the LLM, defined by the `llm_arch` enum.
    - `tensor`: Represents the tensor type, defined by the `llm_tensor` enum.
    - `suffix`: A constant character pointer to a suffix string, which can be appended to the tensor name.
    - `bid`: An integer representing a block identifier.
    - `xid`: An integer representing an additional identifier, possibly for extended identification.
- **Description**: The `LLM_TN_IMPL` struct is a data structure designed to encapsulate information about a specific tensor within a given architecture of a large language model (LLM). It includes fields for the architecture type, tensor type, and optional suffix, as well as identifiers for blocks and extended identification. The struct provides functionality to convert its data into a string representation, which is useful for generating human-readable names for tensors. It also supports comparison operations with strings, allowing for easy checking of tensor names against expected values.
- **Member Functions**:
    - [`LLM_TN_IMPL::str`](llama-arch.cpp.driver.md#LLM_TN_IMPLstr)
    - [`LLM_TN_IMPL::operator==`](#LLM_TN_IMPLoperator==)
    - [`LLM_TN_IMPL::operator!=`](#LLM_TN_IMPLoperator!=)

**Methods**

---
#### LLM\_TN\_IMPL::operator==<!-- {{#callable:LLM_TN_IMPL::operator==}} -->
The `operator==` function compares a `std::string` with an `LLM_TN_IMPL` object by checking if the string is equal to the string representation of the `LLM_TN_IMPL` object.
- **Inputs**:
    - `str`: A constant reference to a `std::string` that is to be compared with the `LLM_TN_IMPL` object.
    - `tn`: A constant reference to an `LLM_TN_IMPL` object whose string representation is to be compared with the input string.
- **Control Flow**:
    - The function calls the `str()` method on the `LLM_TN_IMPL` object `tn` to obtain its string representation.
    - It then compares the input string `str` with the string representation of `tn` using the equality operator `==`.
    - The result of the comparison is returned as a boolean value.
- **Output**: A boolean value indicating whether the input string is equal to the string representation of the `LLM_TN_IMPL` object.
- **See also**: [`LLM_TN_IMPL`](#LLM_TN_IMPL)  (Data Structure)


---
#### LLM\_TN\_IMPL::operator\!=<!-- {{#callable:LLM_TN_IMPL::operator!=}} -->
The `operator!=` function checks if a given string is not equal to the string representation of an `LLM_TN_IMPL` object.
- **Inputs**:
    - `str`: A constant reference to a `std::string` object that is compared against the `LLM_TN_IMPL` object.
    - `tn`: A constant reference to an `LLM_TN_IMPL` object whose string representation is compared against the input string.
- **Control Flow**:
    - The function calls the `str()` method on the `LLM_TN_IMPL` object to get its string representation.
    - It then compares the input string `str` with the string representation of the `LLM_TN_IMPL` object using the `!=` operator.
    - The result of this comparison is returned as the output of the function.
- **Output**: A boolean value indicating whether the input string is not equal to the string representation of the `LLM_TN_IMPL` object.
- **See also**: [`LLM_TN_IMPL`](#LLM_TN_IMPL)  (Data Structure)



---
### LLM\_TN<!-- {{#data_structure:LLM_TN}} -->
- **Type**: `struct`
- **Members**:
    - `arch`: Represents the architecture type of the LLM, defined by the llm_arch enum.
- **Description**: The `LLM_TN` struct is designed to encapsulate a specific architecture type for a language model, as indicated by the `arch` member, which is of type `llm_arch`. It provides overloaded function call operators to create instances of `LLM_TN_IMPL`, which are used to generate string representations of tensor names based on the architecture, tensor type, optional suffix, and optional block and extra identifiers. This struct is primarily used to facilitate the naming and identification of tensors within a language model architecture.
- **Member Functions**:
    - [`LLM_TN::LLM_TN`](#LLM_TNLLM_TN)
    - [`LLM_TN::operator()`](#LLM_TNoperator())
    - [`LLM_TN::operator()`](#LLM_TNoperator())

**Methods**

---
#### LLM\_TN::LLM\_TN<!-- {{#callable:LLM_TN::LLM_TN}} -->
The `LLM_TN` constructor initializes an instance of the `LLM_TN` struct with a specified architecture.
- **Inputs**:
    - `arch`: An enumeration value of type `llm_arch` representing the architecture to be associated with the `LLM_TN` instance.
- **Control Flow**:
    - The constructor takes a single argument `arch` of type `llm_arch`.
    - It initializes the member variable `arch` of the `LLM_TN` struct with the provided `arch` argument.
- **Output**: The constructor does not return any value as it is used to initialize an instance of the `LLM_TN` struct.
- **See also**: [`LLM_TN`](#LLM_TN)  (Data Structure)


---
#### LLM\_TN::operator\(\)<!-- {{#callable:LLM_TN::operator()}} -->
The `operator()` function constructs and returns an `LLM_TN_IMPL` object using the provided tensor, suffix, bid, and xid values, along with the `arch` member of the `LLM_TN` struct.
- **Inputs**:
    - `tensor`: An `llm_tensor` enum value representing the type of tensor.
    - `suffix`: A `const char*` representing an optional suffix for the tensor name.
    - `bid`: An integer representing a block identifier, defaulting to -1 if not provided.
    - `xid`: An integer representing an additional identifier, defaulting to -1 if not provided.
- **Control Flow**:
    - The function takes four parameters: `tensor`, `suffix`, `bid`, and `xid`, with `bid` and `xid` having default values of -1.
    - It constructs an `LLM_TN_IMPL` object using the `arch` member of the `LLM_TN` struct and the provided parameters.
    - The constructed `LLM_TN_IMPL` object is returned.
- **Output**: An `LLM_TN_IMPL` object initialized with the provided parameters and the `arch` member of the `LLM_TN` struct.
- **See also**: [`LLM_TN`](#LLM_TN)  (Data Structure)


---
#### LLM\_TN::operator\(\)<!-- {{#callable:LLM_TN::operator()}} -->
The `operator()` function creates and returns an `LLM_TN_IMPL` object using the provided tensor, bid, and xid, with the architecture from the parent `LLM_TN` object.
- **Inputs**:
    - `tensor`: An `llm_tensor` enum value representing the type of tensor.
    - `bid`: An optional integer representing a block identifier, defaulting to -1.
    - `xid`: An optional integer representing an extra identifier, defaulting to -1.
- **Control Flow**:
    - The function takes three parameters: `tensor`, `bid`, and `xid`, with `bid` and `xid` having default values of -1.
    - It constructs an `LLM_TN_IMPL` object using the `arch` from the parent `LLM_TN` object, the provided `tensor`, a `nullptr` for the suffix, and the `bid` and `xid` values.
    - The constructed `LLM_TN_IMPL` object is returned.
- **Output**: An `LLM_TN_IMPL` object initialized with the given parameters and the architecture from the parent `LLM_TN` object.
- **See also**: [`LLM_TN`](#LLM_TN)  (Data Structure)



---
### llm\_tensor\_info<!-- {{#data_structure:llm_tensor_info}} -->
- **Type**: `struct`
- **Members**:
    - `layer`: Represents the layer of the tensor in the model architecture.
    - `op`: Specifies the operation associated with the tensor.
- **Description**: The `llm_tensor_info` struct is a simple data structure used to encapsulate information about a tensor within a machine learning model. It contains two members: `layer`, which indicates the specific layer of the tensor in the model's architecture, and `op`, which defines the operation that the tensor is involved in. This struct is likely used to manage and organize tensor-related data within the context of a larger machine learning framework.


