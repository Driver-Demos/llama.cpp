# Purpose
This C++ header file defines a set of data structures and enumerations that are crucial for configuring and managing the hyperparameters of a machine learning model, specifically one that appears to be related to the "llama" project. The file includes definitions for various hyperparameter structures, such as `llama_hparams`, `llama_hparams_posnet`, and `llama_hparams_convnext`, which encapsulate a wide range of parameters used to configure different aspects of the model, including embedding dimensions, layer counts, attention mechanisms, and more. The file also defines several enumerations, such as `llama_expert_gating_func_type` and `llama_swa_type`, which categorize different functional types and configurations for model components like gating functions and sliding window attention.

The header file is designed to be included in other parts of the project, as indicated by the `#pragma once` directive, which prevents multiple inclusions. It provides a broad functionality by defining a comprehensive set of parameters that can be adjusted to fine-tune the model's behavior. The file does not define any executable code but rather serves as a configuration interface for the model, allowing developers to specify and manipulate the model's hyperparameters. The presence of functions like `set_swa_pattern` and `is_swa_any` suggests that the file also includes utility functions to manage specific configurations, such as sliding window attention patterns. The use of static assertions ensures that the `llama_hparams` structure is trivially copyable, which is important for efficient memory management and data handling in C++.
# Imports and Dependencies

---
- `llama.h`
- `array`


# Data Structures

---
### llama\_expert\_gating\_func\_type<!-- {{#data_structure:llama_expert_gating_func_type}} -->
- **Type**: `enum`
- **Members**:
    - `LLAMA_EXPERT_GATING_FUNC_TYPE_NONE`: Represents the absence of a gating function, with a value of 0.
    - `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`: Represents a softmax gating function, with a value of 1.
    - `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`: Represents a sigmoid gating function, with a value of 2.
- **Description**: The `llama_expert_gating_func_type` is an enumeration that defines different types of gating functions that can be used in the context of expert models. It includes three possible values: `LLAMA_EXPERT_GATING_FUNC_TYPE_NONE` for no gating function, `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` for a softmax gating function, and `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` for a sigmoid gating function. This enum is likely used to configure or select the gating mechanism applied to expert models within the larger system.


---
### llama\_swa\_type<!-- {{#data_structure:llama_swa_type}} -->
- **Type**: `enum`
- **Members**:
    - `LLAMA_SWA_TYPE_NONE`: Represents the absence of a sliding window attention type.
    - `LLAMA_SWA_TYPE_STANDARD`: Represents a standard sliding window attention type.
    - `LLAMA_SWA_TYPE_CHUNKED`: Represents a chunked sliding window attention type.
- **Description**: The `llama_swa_type` enumeration defines different types of Sliding Window Attention (SWA) mechanisms that can be used in a model. It includes three possible values: `LLAMA_SWA_TYPE_NONE` for no SWA, `LLAMA_SWA_TYPE_STANDARD` for a standard SWA approach, and `LLAMA_SWA_TYPE_CHUNKED` for a chunked SWA approach. This enumeration allows for the selection of the SWA strategy to be applied in the model's attention mechanism.


---
### llama\_hparams\_posnet<!-- {{#data_structure:llama_hparams_posnet}} -->
- **Type**: `struct`
- **Members**:
    - `n_embd`: Represents the number of embeddings in the structure.
    - `n_layer`: Indicates the number of layers in the structure.
- **Description**: The `llama_hparams_posnet` struct is a simple data structure designed to hold hyperparameters related to a positional network, specifically the number of embeddings (`n_embd`) and the number of layers (`n_layer`). This struct is likely used to configure or initialize a model or component within a larger system, providing essential parameters that define the model's architecture.


---
### llama\_hparams\_convnext<!-- {{#data_structure:llama_hparams_convnext}} -->
- **Type**: `struct`
- **Members**:
    - `n_embd`: Represents the number of embedding dimensions.
    - `n_layer`: Indicates the number of layers in the structure.
- **Description**: The `llama_hparams_convnext` struct is a simple data structure designed to hold hyperparameters for a ConvNeXt model, specifically the number of embedding dimensions (`n_embd`) and the number of layers (`n_layer`). This struct is likely used to configure or initialize a ConvNeXt model within the broader context of the llama framework, providing essential parameters that define the model's architecture.


---
### llama\_hparams<!-- {{#data_structure:llama_hparams}} -->
- **Type**: `struct`
- **Members**:
    - `vocab_only`: Indicates if only vocabulary is used.
    - `rope_finetuned`: Indicates if rope is fine-tuned.
    - `use_par_res`: Indicates if parallel residuals are used.
    - `swin_norm`: Indicates if SWIN normalization is used.
    - `n_ctx_train`: Context size the model was trained on.
    - `n_embd`: Number of embeddings.
    - `n_embd_features`: Number of embedding features, default is 0.
    - `n_layer`: Number of layers.
    - `n_rot`: Number of rotations.
    - `n_embd_head_k`: Dimension of keys (d_k) for embeddings.
    - `n_embd_head_v`: Dimension of values (d_v) for embeddings.
    - `n_expert`: Number of experts, default is 0.
    - `n_expert_used`: Number of experts used, default is 0.
    - `n_rel_attn_bkts`: Number of relative attention buckets, default is 0.
    - `n_embd_head_k_mla`: Dimension of keys for MLA, default is 0.
    - `n_embd_head_v_mla`: Dimension of values for MLA, default is 0.
    - `posnet`: Positional network parameters.
    - `convnext`: ConvNext network parameters.
    - `n_head_arr`: Array of head counts per layer.
    - `n_head_kv_arr`: Array of key-value head counts per layer.
    - `n_ff_arr`: Array of feed-forward network sizes per layer.
    - `n_layer_dense_lead`: Number of dense leading layers, default is 0.
    - `n_lora_q`: Number of LoRA queries, default is 0.
    - `n_lora_kv`: Number of LoRA key-values, default is 0.
    - `n_ff_exp`: Number of feed-forward expansions, default is 0.
    - `n_ff_shexp`: Number of feed-forward shared expansions, default is 0.
    - `n_expert_shared`: Number of shared experts, default is 0.
    - `n_norm_groups`: Number of normalization groups, default is 0.
    - `expert_weights_scale`: Scale for expert weights, default is 0.0.
    - `expert_weights_norm`: Indicates if expert weights are normalized, default is false.
    - `expert_gating_func`: Type of expert gating function, default is none.
    - `moe_every_n_layers`: Frequency of MoE layers, default is 0.
    - `f_norm_eps`: Epsilon for normalization.
    - `f_norm_rms_eps`: RMS epsilon for normalization.
    - `f_norm_group_eps`: Group epsilon for normalization.
    - `f_attn_logit_softcapping`: Softcapping for attention logits, default is 50.0f.
    - `f_final_logit_softcapping`: Softcapping for final logits, default is 30.0f.
    - `rescale_every_n_layers`: Rescale frequency for layers, default is 0.
    - `time_mix_extra_dim`: Extra dimension for time mixing, default is 0.
    - `time_decay_extra_dim`: Extra dimension for time decay, default is 0.
    - `wkv_head_size`: Size of WKV head, default is 0.
    - `token_shift_count`: Count of token shifts, default is 2.
    - `n_lora_decay`: Number of LoRA decays, default is 0.
    - `n_lora_iclr`: Number of LoRA ICLR, default is 0.
    - `n_lora_value_res_mix`: Number of LoRA value residual mixes, default is 0.
    - `n_lora_gate`: Number of LoRA gates, default is 0.
    - `rope_attn_factor`: Factor for rope attention, default is 1.0f.
    - `rope_freq_base_train`: Base frequency for rope training.
    - `rope_freq_base_train_swa`: Base frequency for rope training with SWA.
    - `rope_freq_scale_train`: Frequency scale for rope training.
    - `rope_freq_scale_train_swa`: Frequency scale for rope training with SWA.
    - `n_ctx_orig_yarn`: Original context size for yarn.
    - `rope_yarn_log_mul`: Log multiplier for rope yarn.
    - `rope_sections`: Sections for rope configuration.
    - `swa_type`: Type of sliding window attention, default is none.
    - `n_swa`: Size of the sliding window, default is 0.
    - `swa_layers`: Array indicating SWA layers.
    - `ssm_d_conv`: Dimension of convolution in state space models, default is 0.
    - `ssm_d_inner`: Inner dimension in state space models, default is 0.
    - `ssm_d_state`: State dimension in state space models, default is 0.
    - `ssm_dt_rank`: Rank for state space model time dimension, default is 0.
    - `ssm_dt_b_c_rms`: Indicates if RMS is used in state space model time dimension, default is false.
    - `f_clamp_kqv`: Clamp value for KQV, default is 0.0f.
    - `f_max_alibi_bias`: Maximum bias for ALiBi, default is 0.0f.
    - `f_logit_scale`: Scale for logits, default is 0.0f.
    - `f_residual_scale`: Scale for residuals, default is 0.0f.
    - `f_embedding_scale`: Scale for embeddings, default is 0.0f.
    - `f_attention_scale`: Scale for attention, default is 0.0f.
    - `causal_attn`: Indicates if causal attention is used, default is true.
    - `use_alibi`: Indicates if ALiBi is used, default is false.
    - `attn_soft_cap`: Indicates if attention soft cap is used, default is false.
    - `use_kq_norm`: Indicates if KQ normalization is used, default is true.
    - `n_cls_out`: Number of classifier outputs, default is 1.
    - `n_moe_layer_step`: Step size for MoE layers, default is 0.
    - `n_no_rope_layer_step`: Step size for layers without rope, default is 4.
    - `n_attn_temp_floor_scale`: Floor scale for attention temperature, default is 8192.
    - `f_attn_temp_scale`: Scale for attention temperature, default is 0.1.
    - `dec_start_token_id`: Start token ID for decoder models.
    - `pooling_type`: Type of pooling used, default is none.
    - `rope_type`: Type of rope used, default is none.
    - `rope_scaling_type_train`: Type of rope scaling used during training, default is none.
- **Description**: The `llama_hparams` struct is a comprehensive configuration structure used to define hyperparameters for a machine learning model, particularly in the context of neural networks with attention mechanisms. It includes a wide range of parameters such as embedding dimensions, layer counts, attention head configurations, and various scaling factors. The struct also supports advanced features like sliding window attention (SWA), state space models, and MoE (Mixture of Experts) configurations. Additionally, it provides settings for normalization, softcapping, and specific configurations for different model components like WavTokenizer and RWKV. This struct is designed to be flexible and extensible, allowing for detailed customization of model behavior and architecture.


