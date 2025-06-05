# Purpose
The provided Python code defines a class `GGUFWriter` that is responsible for writing data to a file format called GGUF. This class is designed to handle the serialization of tensor data and associated metadata into a structured file format, which is likely used for machine learning models or similar data-intensive applications. The code includes functionality for managing file output, handling different data types, and ensuring data alignment and endianess. The `GGUFWriter` class supports splitting data into multiple files (shards) if necessary, based on specified limits for the number of tensors or file size.

Key components of the code include the `TensorInfo` and `GGUFValue` data classes, which encapsulate information about tensors and key-value pairs, respectively. The `WriterState` enum defines various states of the file writing process, ensuring that operations occur in the correct sequence. The class provides a comprehensive API for adding metadata and tensor data, with methods to handle different data types and structures. The code also includes logging to track the writing process and ensure that the output files are correctly generated. Overall, this code is a specialized library intended for use in applications that require structured data serialization, particularly in the context of machine learning or data processing pipelines.
# Imports and Dependencies

---
- `__future__.annotations`
- `logging`
- `os`
- `shutil`
- `struct`
- `tempfile`
- `dataclasses.dataclass`
- `enum.Enum`
- `enum.auto`
- `math.prod`
- `pathlib.Path`
- `io.BufferedWriter`
- `typing.IO`
- `typing.Any`
- `typing.Sequence`
- `typing.Mapping`
- `string.ascii_letters`
- `string.digits`
- `numpy`
- `.constants.GGUF_DEFAULT_ALIGNMENT`
- `.constants.GGUF_MAGIC`
- `.constants.GGUF_VERSION`
- `.constants.GGMLQuantizationType`
- `.constants.GGUFEndian`
- `.constants.GGUFValueType`
- `.constants.Keys`
- `.constants.RopeScalingType`
- `.constants.PoolingType`
- `.constants.TokenType`
- `.constants.ExpertGatingFuncType`
- `.quants.quant_shape_from_byte_shape`
- `tqdm.tqdm`


# Global Variables

---
### logger
- **Type**: `logging.Logger`
- **Description**: The `logger` variable is an instance of the `Logger` class from the `logging` module, configured to use the current module's name as its logger name. This allows for logging messages that are specific to the module in which the logger is defined.
- **Use**: This variable is used to log informational and warning messages throughout the module, aiding in debugging and monitoring the module's operations.


---
### SHARD\_NAME\_FORMAT
- **Type**: `str`
- **Description**: `SHARD_NAME_FORMAT` is a string template used to format the names of shard files. It includes placeholders for a string identifier and two zero-padded integers, representing the shard index and total number of shards, respectively.
- **Use**: This variable is used to generate consistent and descriptive filenames for shard files in a GGUF (Generic Graphical User Format) file writing process.


# Classes

---
### TensorInfo<!-- {{#class:llama.cpp/gguf-py/gguf/gguf_writer.TensorInfo}} -->
- **Decorators**: `@dataclass`
- **Members**:
    - `shape`: Specifies the dimensions of the tensor as a sequence of integers.
    - `dtype`: Indicates the data type of the tensor, specifically the quantization type.
    - `nbytes`: Represents the number of bytes the tensor occupies in memory.
    - `tensor`: Holds the actual tensor data as a NumPy array or None if not set.
- **Description**: The `TensorInfo` class is a data structure that encapsulates metadata about a tensor, including its shape, data type, and memory size, with an optional field for the tensor's data itself. It is designed to facilitate the handling and manipulation of tensor information, particularly in contexts where quantization and memory management are important.


---
### GGUFValue<!-- {{#class:llama.cpp/gguf-py/gguf/gguf_writer.GGUFValue}} -->
- **Decorators**: `@dataclass`
- **Members**:
    - `value`: Holds the actual value of the GGUFValue instance.
    - `type`: Specifies the type of the value using GGUFValueType.
    - `sub_type`: Optional sub-type of the value, also using GGUFValueType.
- **Description**: The GGUFValue class is a data structure designed to encapsulate a value along with its type and an optional sub-type. It is used to represent metadata values in the GGUF file format, where each value is associated with a specific type defined by the GGUFValueType enumeration. The class is decorated with @dataclass to automatically generate special methods like __init__ and __repr__.


---
### WriterState<!-- {{#class:llama.cpp/gguf-py/gguf/gguf_writer.WriterState}} -->
- **Description**: The `WriterState` class is an enumeration that defines various states for a writer, such as NO_FILE, EMPTY, HEADER, KV_DATA, TI_DATA, and WEIGHTS. These states are used to track the progress and current status of a writing process, likely in the context of file or data output operations.
- **Inherits From**:
    - `Enum`


---
### GGUFWriter<!-- {{#class:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter}} -->
- **Members**:
    - `fout`: A list of BufferedWriter objects or None, representing the output file streams.
    - `path`: A Path object or None, representing the file path for output.
    - `temp_file`: A SpooledTemporaryFile object or None, used for temporary storage of tensor data.
    - `tensors`: A list of dictionaries mapping tensor names to TensorInfo objects.
    - `kv_data`: A list of dictionaries mapping keys to GGUFValue objects.
    - `state`: An instance of WriterState, representing the current state of the writer.
    - `_simple_value_packing`: A dictionary mapping GGUFValueType to their corresponding struct packing format strings.
- **Description**: The GGUFWriter class is responsible for managing the writing of GGUF (Generic Graph Universal Format) files, which involves handling tensor data, key-value metadata, and file output operations. It supports features like splitting tensor data into multiple files, using temporary files for intermediate storage, and managing different data types and their serialization formats. The class maintains the state of the writing process and provides methods to add various types of metadata and tensor information, ensuring the correct format and structure of the output GGUF files.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.__init__`](#GGUFWriter__init__)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.get_total_parameter_count`](#GGUFWriterget_total_parameter_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.format_shard_names`](#GGUFWriterformat_shard_names)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.open_output_file`](#GGUFWriteropen_output_file)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.print_plan`](#GGUFWriterprint_plan)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_shard_kv_data`](#GGUFWriteradd_shard_kv_data)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_header_to_file`](#GGUFWriterwrite_header_to_file)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_kv_data_to_file`](#GGUFWriterwrite_kv_data_to_file)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_ti_data_to_file`](#GGUFWriterwrite_ti_data_to_file)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint8`](#GGUFWriteradd_uint8)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_int8`](#GGUFWriteradd_int8)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint16`](#GGUFWriteradd_uint16)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_int16`](#GGUFWriteradd_int16)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_int32`](#GGUFWriteradd_int32)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint64`](#GGUFWriteradd_uint64)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_int64`](#GGUFWriteradd_int64)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float64`](#GGUFWriteradd_float64)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.ggml_pad`](#GGUFWriterggml_pad)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tensor_info`](#GGUFWriteradd_tensor_info)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tensor`](#GGUFWriteradd_tensor)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_padding`](#GGUFWriterwrite_padding)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_tensor_data`](#GGUFWriterwrite_tensor_data)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_tensors_to_file`](#GGUFWriterwrite_tensors_to_file)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.flush`](#GGUFWriterflush)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.close`](#GGUFWriterclose)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_type`](#GGUFWriteradd_type)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_architecture`](#GGUFWriteradd_architecture)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_quantization_version`](#GGUFWriteradd_quantization_version)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_custom_alignment`](#GGUFWriteradd_custom_alignment)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_file_type`](#GGUFWriteradd_file_type)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_name`](#GGUFWriteradd_name)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_author`](#GGUFWriteradd_author)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_version`](#GGUFWriteradd_version)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_organization`](#GGUFWriteradd_organization)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_finetune`](#GGUFWriteradd_finetune)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_basename`](#GGUFWriteradd_basename)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_description`](#GGUFWriteradd_description)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_quantized_by`](#GGUFWriteradd_quantized_by)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_size_label`](#GGUFWriteradd_size_label)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_license`](#GGUFWriteradd_license)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_license_name`](#GGUFWriteradd_license_name)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_license_link`](#GGUFWriteradd_license_link)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_url`](#GGUFWriteradd_url)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_doi`](#GGUFWriteradd_doi)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uuid`](#GGUFWriteradd_uuid)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_repo_url`](#GGUFWriteradd_repo_url)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_source_url`](#GGUFWriteradd_source_url)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_source_doi`](#GGUFWriteradd_source_doi)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_source_uuid`](#GGUFWriteradd_source_uuid)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_source_repo_url`](#GGUFWriteradd_source_repo_url)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_count`](#GGUFWriteradd_base_model_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_name`](#GGUFWriteradd_base_model_name)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_author`](#GGUFWriteradd_base_model_author)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_version`](#GGUFWriteradd_base_model_version)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_organization`](#GGUFWriteradd_base_model_organization)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_description`](#GGUFWriteradd_base_model_description)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_url`](#GGUFWriteradd_base_model_url)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_doi`](#GGUFWriteradd_base_model_doi)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_uuid`](#GGUFWriteradd_base_model_uuid)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_repo_url`](#GGUFWriteradd_base_model_repo_url)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_count`](#GGUFWriteradd_dataset_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_name`](#GGUFWriteradd_dataset_name)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_author`](#GGUFWriteradd_dataset_author)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_version`](#GGUFWriteradd_dataset_version)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_organization`](#GGUFWriteradd_dataset_organization)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_description`](#GGUFWriteradd_dataset_description)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_url`](#GGUFWriteradd_dataset_url)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_doi`](#GGUFWriteradd_dataset_doi)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_uuid`](#GGUFWriteradd_dataset_uuid)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_repo_url`](#GGUFWriteradd_dataset_repo_url)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tags`](#GGUFWriteradd_tags)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_languages`](#GGUFWriteradd_languages)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tensor_data_layout`](#GGUFWriteradd_tensor_data_layout)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vocab_size`](#GGUFWriteradd_vocab_size)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_context_length`](#GGUFWriteradd_context_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_embedding_length`](#GGUFWriteradd_embedding_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_features_length`](#GGUFWriteradd_features_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_posnet_embedding_length`](#GGUFWriteradd_posnet_embedding_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_posnet_block_count`](#GGUFWriteradd_posnet_block_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_convnext_embedding_length`](#GGUFWriteradd_convnext_embedding_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_convnext_block_count`](#GGUFWriteradd_convnext_block_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_block_count`](#GGUFWriteradd_block_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_leading_dense_block_count`](#GGUFWriteradd_leading_dense_block_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_feed_forward_length`](#GGUFWriteradd_feed_forward_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_feed_forward_length`](#GGUFWriteradd_expert_feed_forward_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_shared_feed_forward_length`](#GGUFWriteradd_expert_shared_feed_forward_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_parallel_residual`](#GGUFWriteradd_parallel_residual)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_decoder_start_token_id`](#GGUFWriteradd_decoder_start_token_id)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_head_count`](#GGUFWriteradd_head_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_head_count_kv`](#GGUFWriteradd_head_count_kv)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_length`](#GGUFWriteradd_key_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_value_length`](#GGUFWriteradd_value_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_length_mla`](#GGUFWriteradd_key_length_mla)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_value_length_mla`](#GGUFWriteradd_value_length_mla)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_max_alibi_bias`](#GGUFWriteradd_max_alibi_bias)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_clamp_kqv`](#GGUFWriteradd_clamp_kqv)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_logit_scale`](#GGUFWriteradd_logit_scale)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_attn_logit_softcapping`](#GGUFWriteradd_attn_logit_softcapping)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_final_logit_softcapping`](#GGUFWriteradd_final_logit_softcapping)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_count`](#GGUFWriteradd_expert_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_used_count`](#GGUFWriteradd_expert_used_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_shared_count`](#GGUFWriteradd_expert_shared_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_weights_scale`](#GGUFWriteradd_expert_weights_scale)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_weights_norm`](#GGUFWriteradd_expert_weights_norm)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_gating_func`](#GGUFWriteradd_expert_gating_func)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_moe_every_n_layers`](#GGUFWriteradd_moe_every_n_layers)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_swin_norm`](#GGUFWriteradd_swin_norm)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rescale_every_n_layers`](#GGUFWriteradd_rescale_every_n_layers)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_time_mix_extra_dim`](#GGUFWriteradd_time_mix_extra_dim)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_time_decay_extra_dim`](#GGUFWriteradd_time_decay_extra_dim)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_residual_scale`](#GGUFWriteradd_residual_scale)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_embedding_scale`](#GGUFWriteradd_embedding_scale)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_wkv_head_size`](#GGUFWriteradd_wkv_head_size)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_shift_count`](#GGUFWriteradd_token_shift_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_interleave_moe_layer_step`](#GGUFWriteradd_interleave_moe_layer_step)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_layer_norm_eps`](#GGUFWriteradd_layer_norm_eps)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_layer_norm_rms_eps`](#GGUFWriteradd_layer_norm_rms_eps)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_group_norm_eps`](#GGUFWriteradd_group_norm_eps)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_group_norm_groups`](#GGUFWriteradd_group_norm_groups)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_causal_attention`](#GGUFWriteradd_causal_attention)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_q_lora_rank`](#GGUFWriteradd_q_lora_rank)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_kv_lora_rank`](#GGUFWriteradd_kv_lora_rank)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_decay_lora_rank`](#GGUFWriteradd_decay_lora_rank)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_iclr_lora_rank`](#GGUFWriteradd_iclr_lora_rank)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_value_residual_mix_lora_rank`](#GGUFWriteradd_value_residual_mix_lora_rank)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_gate_lora_rank`](#GGUFWriteradd_gate_lora_rank)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_relative_attn_buckets_count`](#GGUFWriteradd_relative_attn_buckets_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_sliding_window`](#GGUFWriteradd_sliding_window)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_attention_scale`](#GGUFWriteradd_attention_scale)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_pooling_type`](#GGUFWriteradd_pooling_type)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_dimension_count`](#GGUFWriteradd_rope_dimension_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_dimension_sections`](#GGUFWriteradd_rope_dimension_sections)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_freq_base`](#GGUFWriteradd_rope_freq_base)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_scaling_type`](#GGUFWriteradd_rope_scaling_type)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_scaling_factor`](#GGUFWriteradd_rope_scaling_factor)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_scaling_attn_factors`](#GGUFWriteradd_rope_scaling_attn_factors)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_scaling_orig_ctx_len`](#GGUFWriteradd_rope_scaling_orig_ctx_len)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_scaling_finetuned`](#GGUFWriteradd_rope_scaling_finetuned)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_scaling_yarn_log_mul`](#GGUFWriteradd_rope_scaling_yarn_log_mul)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_ssm_conv_kernel`](#GGUFWriteradd_ssm_conv_kernel)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_ssm_inner_size`](#GGUFWriteradd_ssm_inner_size)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_ssm_state_size`](#GGUFWriteradd_ssm_state_size)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_ssm_time_step_rank`](#GGUFWriteradd_ssm_time_step_rank)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_ssm_dt_b_c_rms`](#GGUFWriteradd_ssm_dt_b_c_rms)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tokenizer_model`](#GGUFWriteradd_tokenizer_model)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tokenizer_pre`](#GGUFWriteradd_tokenizer_pre)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_list`](#GGUFWriteradd_token_list)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_merges`](#GGUFWriteradd_token_merges)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_types`](#GGUFWriteradd_token_types)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_type_count`](#GGUFWriteradd_token_type_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_scores`](#GGUFWriteradd_token_scores)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bos_token_id`](#GGUFWriteradd_bos_token_id)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_eos_token_id`](#GGUFWriteradd_eos_token_id)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_unk_token_id`](#GGUFWriteradd_unk_token_id)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_sep_token_id`](#GGUFWriteradd_sep_token_id)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_pad_token_id`](#GGUFWriteradd_pad_token_id)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_mask_token_id`](#GGUFWriteradd_mask_token_id)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_add_bos_token`](#GGUFWriteradd_add_bos_token)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_add_eos_token`](#GGUFWriteradd_add_eos_token)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_add_space_prefix`](#GGUFWriteradd_add_space_prefix)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_remove_extra_whitespaces`](#GGUFWriteradd_remove_extra_whitespaces)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_precompiled_charsmap`](#GGUFWriteradd_precompiled_charsmap)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_chat_template`](#GGUFWriteradd_chat_template)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_eot_token_id`](#GGUFWriteradd_eot_token_id)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_eom_token_id`](#GGUFWriteradd_eom_token_id)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_classifier_output_labels`](#GGUFWriteradd_classifier_output_labels)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_clip_has_vision_encoder`](#GGUFWriteradd_clip_has_vision_encoder)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_clip_has_audio_encoder`](#GGUFWriteradd_clip_has_audio_encoder)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_clip_projector_type`](#GGUFWriteradd_clip_projector_type)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_projection_dim`](#GGUFWriteradd_vision_projection_dim)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_patch_size`](#GGUFWriteradd_vision_patch_size)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_embedding_length`](#GGUFWriteradd_vision_embedding_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_feed_forward_length`](#GGUFWriteradd_vision_feed_forward_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_block_count`](#GGUFWriteradd_vision_block_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_head_count`](#GGUFWriteradd_vision_head_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_attention_layernorm_eps`](#GGUFWriteradd_vision_attention_layernorm_eps)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_image_size`](#GGUFWriteradd_vision_image_size)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_image_mean`](#GGUFWriteradd_vision_image_mean)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_image_std`](#GGUFWriteradd_vision_image_std)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_spatial_merge_size`](#GGUFWriteradd_vision_spatial_merge_size)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_use_gelu`](#GGUFWriteradd_vision_use_gelu)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_use_silu`](#GGUFWriteradd_vision_use_silu)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_projector_scale_factor`](#GGUFWriteradd_vision_projector_scale_factor)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_n_wa_pattern`](#GGUFWriteradd_vision_n_wa_pattern)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_projection_dim`](#GGUFWriteradd_audio_projection_dim)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_embedding_length`](#GGUFWriteradd_audio_embedding_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_feed_forward_length`](#GGUFWriteradd_audio_feed_forward_length)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_block_count`](#GGUFWriteradd_audio_block_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_head_count`](#GGUFWriteradd_audio_head_count)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_attention_layernorm_eps`](#GGUFWriteradd_audio_attention_layernorm_eps)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_num_mel_bins`](#GGUFWriteradd_audio_num_mel_bins)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_stack_factor`](#GGUFWriteradd_audio_stack_factor)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter._pack`](#GGUFWriter_pack)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter._pack_val`](#GGUFWriter_pack_val)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.format_n_bytes_to_str`](#GGUFWriterformat_n_bytes_to_str)

**Methods**

---
#### GGUFWriter\.\_\_init\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.__init__}} -->
Initializes a GGUFWriter instance with specified parameters for file handling and tensor management.
- **Inputs**:
    - `path`: The file path where the GGUF data will be written, can be a string or a Path object.
    - `arch`: A string representing the architecture type of the model being written.
    - `use_temp_file`: A boolean indicating whether to use a temporary file for writing data.
    - `endianess`: An enumeration value indicating the endianness (byte order) for the data, defaulting to little-endian.
    - `split_max_tensors`: An integer specifying the maximum number of tensors allowed before splitting into a new file.
    - `split_max_size`: An integer specifying the maximum size (in bytes) allowed for a single file before splitting.
    - `dry_run`: A boolean indicating whether to perform a dry run without actual file writing.
    - `small_first_shard`: A boolean indicating whether to create a smaller first shard when splitting files.
- **Control Flow**:
    - Initializes instance variables for file output, path, architecture, endianess, and other parameters.
    - Logs the endianess type being used for the GGUF file.
    - Sets the initial state of the writer to 'NO_FILE'.
    - If 'small_first_shard' is true, appends an empty dictionary to the tensors list.
    - Calls the 'add_architecture' method to store the architecture information.
- **Output**: No output is returned; the method initializes the state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_architecture`](#GGUFWriteradd_architecture)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.get\_total\_parameter\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.get_total_parameter_count}} -->
Calculates the total number of parameters in a model, distinguishing between shared and expert parameters.
- **Inputs**:
    - `self`: An instance of the `GGUFWriter` class, which contains the model's tensor data.
- **Control Flow**:
    - Initializes counters for total, shared, and expert parameters, as well as tracking for expert tensors.
    - Iterates through the list of tensors, checking each tensor's name and shape.
    - Handles special cases for tensors ending with '.lora_a' and '.lora_b' to correctly compute their sizes.
    - Accumulates the total size of parameters, distinguishing between shared and expert parameters based on naming conventions.
    - Calculates the average expert count if expert tensors are present.
    - Negates the total parameter count if a '.lora_a' tensor was found, indicating the count may not be exact.
- **Output**: Returns a tuple containing the total number of parameters (potentially negated), the number of shared parameters, the number of expert parameters, and the average expert count.
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.format\_shard\_names<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.format_shard_names}} -->
Formats the names of shard files based on the number of tensors.
- **Inputs**:
    - `path`: A `Path` object representing the base path for the shard files.
- **Control Flow**:
    - Checks if there is only one tensor; if so, returns a list containing the original path.
    - If there are multiple tensors, generates a list of paths formatted according to the `SHARD_NAME_FORMAT`, including the tensor index and total count.
- **Output**: Returns a list of `Path` objects representing the formatted shard names.
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.open\_output\_file<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.open_output_file}} -->
Opens an output file for writing data, ensuring the correct state and path are set.
- **Inputs**:
    - `path`: An optional `Path` object representing the file path to open; if None, retains the current path.
- **Control Flow**:
    - Checks if the current state is EMPTY and if the output file is already opened with the same path, allowing multiple calls without action.
    - Raises a ValueError if the state is not NO_FILE, indicating that the output file is already opened.
    - Updates the path if a new path is provided.
    - Calls `print_plan()` to get the filenames to open, then opens each file in binary write mode and updates the state to EMPTY.
- **Output**: None; the method modifies the internal state and opens files for writing.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.print_plan`](#GGUFWriterprint_plan)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.print\_plan<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.print_plan}} -->
Generates a list of file paths for output based on the provided tensor data and logs the details of the files to be written.
- **Inputs**:
    - `self`: An instance of the `GGUFWriter` class, which contains the necessary attributes for generating file paths and managing tensor data.
- **Control Flow**:
    - Logs the message indicating the files that will be written.
    - Asserts that the `path` attribute of the instance is not None.
    - Calls the [`format_shard_names`](#GGUFWriterformat_shard_names) method to generate a list of filenames based on the `path` and the number of tensors.
    - Asserts that the number of generated filenames matches the number of tensors.
    - Logs the details of each filename, including the number of tensors and their total size.
    - Checks if the `dry_run` attribute is set to True; if so, logs a message indicating that no files will be written and prints the filenames before exiting.
    - Returns the list of generated filenames.
- **Output**: Returns a list of `Path` objects representing the filenames for the output files.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.format_shard_names`](#GGUFWriterformat_shard_names)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.format_n_bytes_to_str`](#GGUFWriterformat_n_bytes_to_str)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_shard\_kv\_data<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_shard_kv_data}} -->
Adds key-value data for sharding based on the number of tensors.
- **Inputs**: None
- **Control Flow**:
    - Checks if the number of tensors is one; if so, it returns early without doing anything.
    - Calculates the total number of tensors across all shards.
    - Asserts that the output file list is not None.
    - Extends the `kv_data` list to match the number of output files.
    - Iterates over the `kv_data` to populate it with split number, total splits, and total tensor counts.
- **Output**: The method does not return any value; it modifies the internal state of the `kv_data` attribute.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFValue`](#cpp/gguf-py/gguf/gguf_writerGGUFValue)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.write\_header\_to\_file<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_header_to_file}} -->
Writes the header information to the output file for the GGUF format.
- **Inputs**:
    - `path`: An optional Path object specifying the output file path.
- **Control Flow**:
    - Checks if the number of tensors is one and if splitting conditions are met, logging a warning if not.
    - Calls the [`open_output_file`](#GGUFWriteropen_output_file) method to prepare the output file.
    - Validates that the output file is in the expected empty state.
    - Asserts that the output file handle and tensor data are correctly initialized.
    - Calls [`add_shard_kv_data`](#GGUFWriteradd_shard_kv_data) to prepare key-value data for sharding if necessary.
    - Iterates over the output file handles, tensors, and key-value data to write the header information, including magic number, version, tensor count, and key-value data count.
- **Output**: None, but the output file is updated with the header information.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.open_output_file`](#GGUFWriteropen_output_file)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_shard_kv_data`](#GGUFWriteradd_shard_kv_data)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter._pack`](#GGUFWriter_pack)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.flush`](#GGUFWriterflush)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.write\_kv\_data\_to\_file<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_kv_data_to_file}} -->
Writes key-value data to a file after ensuring the file is in the correct state.
- **Inputs**:
    - `self`: An instance of the `GGUFWriter` class, which contains the state and data to be written.
- **Control Flow**:
    - Checks if the current state is `HEADER`, raising a ValueError if not.
    - Asserts that the output file list `fout` is not None.
    - Iterates over pairs of output files and key-value data.
    - For each key-value pair, packs the key and value into bytes using the [`_pack_val`](#GGUFWriter_pack_val) method.
    - Writes the packed bytes to the corresponding output file.
    - Calls the [`flush`](#GGUFWriterflush) method to ensure all data is written to the files.
    - Updates the state to `KV_DATA` after writing.
- **Output**: None, but the method writes the key-value data to the specified output files.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter._pack_val`](#GGUFWriter_pack_val)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.flush`](#GGUFWriterflush)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.write\_ti\_data\_to\_file<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_ti_data_to_file}} -->
Writes tensor information data to a file.
- **Inputs**:
    - `self`: An instance of the `GGUFWriter` class, which contains the state and data to be written.
- **Control Flow**:
    - Checks if the current state is `KV_DATA`, raising a ValueError if not.
    - Asserts that the output file list `fout` is not None.
    - Iterates over pairs of output files and tensors.
    - For each tensor, constructs a bytearray containing its name, dimensions, shape, data type, and offset.
    - Writes the constructed bytearray to the corresponding output file and flushes the file.
- **Output**: The method does not return a value; it writes the tensor information directly to the output files.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter._pack_val`](#GGUFWriter_pack_val)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter._pack`](#GGUFWriter_pack)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.ggml_pad`](#GGUFWriterggml_pad)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.flush`](#GGUFWriterflush)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_key\_value<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value}} -->
Adds a key-value pair to the `kv_data` dictionary, ensuring that the key is unique.
- **Inputs**:
    - `key`: A string representing the key to be added.
    - `val`: The value associated with the key, which can be of any type.
    - `vtype`: An enumeration value of type `GGUFValueType` indicating the type of the value.
    - `sub_type`: An optional subtype of `GGUFValueType`, which can be used to specify additional type information.
- **Control Flow**:
    - Checks if the provided key already exists in any of the dictionaries within `kv_data`.
    - If the key is found, a `ValueError` is raised indicating a duplicate key.
    - If the key is unique, a new [`GGUFValue`](#cpp/gguf-py/gguf/gguf_writerGGUFValue) instance is created with the provided value, type, and optional subtype, and is added to the first dictionary in `kv_data`.
- **Output**: The method does not return any value; it modifies the internal state of the `kv_data` attribute.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFValue`](#cpp/gguf-py/gguf/gguf_writerGGUFValue)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_uint8<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint8}} -->
Adds a key-value pair to the internal storage with the value type set as unsigned 8-bit integer.
- **Inputs**:
    - `key`: A string representing the key under which the value will be stored.
    - `val`: An integer value that is expected to be in the range of an unsigned 8-bit integer (0 to 255).
- **Control Flow**:
    - Calls the [`add_key_value`](#GGUFWriteradd_key_value) method with the provided key, value, and the value type `GGUFValueType.UINT8`.
    - The [`add_key_value`](#GGUFWriteradd_key_value) method checks for duplicate keys and stores the key-value pair in the internal `kv_data` structure.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_int8<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_int8}} -->
Adds an 8-bit integer value associated with a specified key to the key-value data structure.
- **Inputs**:
    - `key`: A string representing the key to associate with the integer value.
    - `val`: An integer value to be added, which should fit within the 8-bit signed integer range.
- **Control Flow**:
    - Calls the [`add_key_value`](#GGUFWriteradd_key_value) method to store the key-value pair in the internal data structure.
    - The method specifies the value type as `GGUFValueType.INT8` to indicate that the value is an 8-bit integer.
- **Output**: The method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_uint16<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint16}} -->
Adds a 16-bit unsigned integer key-value pair to the internal key-value data structure.
- **Inputs**:
    - `key`: A string representing the key for the key-value pair.
    - `val`: An integer value that is to be stored as a 16-bit unsigned integer.
- **Control Flow**:
    - Calls the [`add_key_value`](#GGUFWriteradd_key_value) method with the provided key, value, and the type `GGUFValueType.UINT16`.
    - The [`add_key_value`](#GGUFWriteradd_key_value) method checks for duplicate keys and adds the key-value pair to the internal storage.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_int16<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_int16}} -->
Adds a 16-bit integer value associated with a specified key to the key-value data structure.
- **Inputs**:
    - `key`: A string representing the key to associate with the integer value.
    - `val`: An integer value that will be stored as a 16-bit integer.
- **Control Flow**:
    - Calls the [`add_key_value`](#GGUFWriteradd_key_value) method of the parent class `GGUFWriter`.
    - Passes the key, value, and the type `GGUFValueType.INT16` to the [`add_key_value`](#GGUFWriteradd_key_value) method.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_uint32<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32}} -->
Adds a key-value pair to the internal key-value data structure with the value type set to UINT32.
- **Inputs**:
    - `key`: A string representing the key for the key-value pair.
    - `val`: An integer value to be associated with the key, which must fit within the range of a 32-bit unsigned integer.
- **Control Flow**:
    - Calls the [`add_key_value`](#GGUFWriteradd_key_value) method with the provided key, value, and the value type set to `GGUFValueType.UINT32`.
    - The [`add_key_value`](#GGUFWriteradd_key_value) method checks for duplicate keys and adds the key-value pair to the internal data structure.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_int32<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_int32}} -->
Adds a key-value pair to the internal storage with the value type set to INT32.
- **Inputs**:
    - `key`: A string representing the key under which the value will be stored.
    - `val`: An integer value to be associated with the specified key.
- **Control Flow**:
    - Calls the [`add_key_value`](#GGUFWriteradd_key_value) method with the provided key, value, and the value type set to `GGUFValueType.INT32`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_float32<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32}} -->
Adds a float32 value associated with a key to the key-value data structure.
- **Inputs**:
    - `key`: A string representing the key to associate with the float32 value.
    - `val`: A float representing the value to be added.
- **Control Flow**:
    - Calls the [`add_key_value`](#GGUFWriteradd_key_value) method with the provided key, value, and the type `GGUFValueType.FLOAT32`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_uint64<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint64}} -->
Adds a key-value pair to the internal key-value data structure with the value type set to UINT64.
- **Inputs**:
    - `key`: A string representing the key for the key-value pair.
    - `val`: An integer value to be associated with the key, expected to be a 64-bit unsigned integer.
- **Control Flow**:
    - Calls the [`add_key_value`](#GGUFWriteradd_key_value) method with the provided key, value, and the value type set to `GGUFValueType.UINT64`.
    - The [`add_key_value`](#GGUFWriteradd_key_value) method checks for duplicate keys and adds the key-value pair to the internal data structure.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_int64<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_int64}} -->
Adds a key-value pair to the internal key-value data structure with the value type set to INT64.
- **Inputs**:
    - `key`: A string representing the key for the key-value pair.
    - `val`: An integer value to be associated with the key, specifically of type INT64.
- **Control Flow**:
    - Calls the [`add_key_value`](#GGUFWriteradd_key_value) method with the provided key, value, and the value type set to `GGUFValueType.INT64`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_float64<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float64}} -->
Adds a float64 value associated with a key to the key-value data structure.
- **Inputs**:
    - `key`: A string representing the key to associate with the float64 value.
    - `val`: A float representing the value to be added.
- **Control Flow**:
    - Calls the [`add_key_value`](#GGUFWriteradd_key_value) method to add the key-value pair.
    - Specifies the value type as `GGUFValueType.FLOAT64`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_bool<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool}} -->
Adds a boolean key-value pair to the internal key-value data structure.
- **Inputs**:
    - `key`: A string representing the key under which the boolean value will be stored.
    - `val`: A boolean value to be associated with the specified key.
- **Control Flow**:
    - Calls the [`add_key_value`](#GGUFWriteradd_key_value) method with the provided key, value, and the type `GGUFValueType.BOOL`.
    - The [`add_key_value`](#GGUFWriteradd_key_value) method checks for duplicate keys and adds the key-value pair to the internal storage.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_string<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string}} -->
Adds a string key-value pair to the internal key-value data structure if the value is not empty.
- **Inputs**:
    - `key`: A string representing the key to be added.
    - `val`: A string representing the value associated with the key.
- **Control Flow**:
    - Checks if the value is empty; if it is, the method returns immediately without adding anything.
    - If the value is not empty, it calls the [`add_key_value`](#GGUFWriteradd_key_value) method to add the key-value pair with the type `GGUFValueType.STRING`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_array<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array}} -->
Adds an array to the key-value data structure if the array is not empty.
- **Inputs**:
    - `key`: A string representing the key under which the array will be stored.
    - `val`: A sequence of any type representing the array to be added.
- **Control Flow**:
    - Checks if the length of `val` is zero; if so, the function returns immediately without making any changes.
    - If `val` is not empty, it calls the [`add_key_value`](#GGUFWriteradd_key_value) method to store the array under the specified key with the type `GGUFValueType.ARRAY`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the array to its key-value data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_value`](#GGUFWriteradd_key_value)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.ggml\_pad<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.ggml_pad}} -->
Calculates the nearest multiple of `n` that is greater than or equal to `x`.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `x`: An integer value that needs to be padded.
    - `n`: An integer value that specifies the multiple to which `x` should be padded.
- **Control Flow**:
    - The method computes the expression ((x + n - 1) // n) * n to find the nearest multiple of `n` that is greater than or equal to `x`.
    - The expression works by first adjusting `x` to ensure that any remainder when divided by `n` is accounted for, effectively rounding up to the next multiple.
- **Output**: Returns an integer that is the nearest multiple of `n` that is greater than or equal to `x`.
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_tensor\_info<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tensor_info}} -->
Adds information about a tensor to the writer's internal state.
- **Inputs**:
    - `name`: The name of the tensor to be added.
    - `tensor_shape`: A sequence representing the shape of the tensor.
    - `tensor_dtype`: The data type of the tensor, specified as a NumPy dtype.
    - `tensor_nbytes`: The number of bytes that the tensor occupies.
    - `raw_dtype`: An optional quantization type for the tensor, if applicable.
- **Control Flow**:
    - Check if the writer is in a valid state to add a tensor; if not, raise a ValueError.
    - Check for duplicate tensor names; if a duplicate is found, raise a ValueError.
    - Determine the quantization type based on the provided tensor data type or use the provided raw_dtype if available.
    - Check if the current tensor list exceeds the maximum allowed tensors or size; if so, append a new tensor dictionary to split the tensors.
    - Add the tensor information to the current tensor dictionary.
- **Output**: This method does not return a value; it modifies the internal state of the `GGUFWriter` by adding a new tensor's information.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/quants.quant_shape_from_byte_shape`](quants.py.driver.md#cpp/gguf-py/gguf/quantsquant_shape_from_byte_shape)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.TensorInfo`](#cpp/gguf-py/gguf/gguf_writerTensorInfo)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_tensor<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tensor}} -->
Adds a tensor to the GGUFWriter, handling endianess, temporary file usage, and tensor metadata.
- **Inputs**:
    - `name`: The name of the tensor to be added.
    - `tensor`: The numpy array representing the tensor.
    - `raw_shape`: An optional sequence defining the shape of the tensor.
    - `raw_dtype`: An optional quantization type for the tensor.
- **Control Flow**:
    - Checks if the tensor's endianess is BIG and swaps bytes if necessary.
    - Initializes a temporary file if required and not already created.
    - Determines the shape of the tensor, using raw_shape if provided, otherwise using the tensor's shape.
    - Calls add_tensor_info to store metadata about the tensor.
    - If no temporary file is used, assigns the tensor directly to the internal storage.
    - If a temporary file is used, writes the tensor data to the temporary file and adds padding.
- **Output**: The method does not return a value; it modifies the internal state of the GGUFWriter by adding the tensor and its metadata.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tensor_info`](#GGUFWriteradd_tensor_info)
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.tofile`](lazy.py.driver.md#LazyNumpyTensortofile)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_padding`](#GGUFWriterwrite_padding)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.write\_padding<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_padding}} -->
Writes padding bytes to a file stream to ensure proper alignment.
- **Inputs**:
    - `fp`: A file-like object (IO[bytes]) where the padding will be written.
    - `n`: The number of bytes that need to be padded.
    - `align`: An optional alignment value; if not provided, defaults to the instance's data alignment.
- **Control Flow**:
    - Calculates the amount of padding needed by calling [`ggml_pad`](#GGUFWriterggml_pad) with the number of bytes and the alignment value.
    - If padding is greater than zero, writes the calculated number of zero bytes to the file stream.
- **Output**: The function does not return any value; it modifies the file stream directly by writing padding bytes.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.ggml_pad`](#GGUFWriterggml_pad)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.write\_tensor\_data<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_tensor_data}} -->
Writes tensor data to a file after validating the current state and handling endianess.
- **Inputs**:
    - `tensor`: A NumPy array representing the tensor data to be written.
- **Control Flow**:
    - Checks if the current state is either TI_DATA or WEIGHTS; raises a ValueError if not.
    - Asserts that the output file handle is not None.
    - If the endianess is BIG, the tensor is byte-swapped.
    - Finds the first non-empty tensor group and retrieves the corresponding file handle.
    - Pops the first tensor info from the tensor group and asserts that the byte size matches the input tensor.
    - Writes padding to the file, followed by the tensor data, and then additional padding based on the tensor size.
    - Updates the state to WEIGHTS.
- **Output**: None, but the tensor data is written to the appropriate file.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_padding`](#GGUFWriterwrite_padding)
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.tofile`](lazy.py.driver.md#LazyNumpyTensortofile)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.write\_tensors\_to\_file<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_tensors_to_file}} -->
The `write_tensors_to_file` method writes tensor data to specified output files, optionally displaying progress.
- **Inputs**:
    - `progress`: A boolean flag indicating whether to display a progress bar during the writing process.
- **Control Flow**:
    - Calls [`write_ti_data_to_file`](#GGUFWriterwrite_ti_data_to_file) to write tensor info data before writing the actual tensor data.
    - Asserts that the output file list `fout` is not None.
    - Iterates over each output file in `fout` to write padding based on the current file position.
    - Checks if `temp_file` is None to determine the writing method: either directly writing tensors or copying from a temporary file.
    - If `progress` is True, initializes progress bars for overall and shard-specific writing.
    - For each tensor in the output files, writes the tensor data to the file, updating progress bars accordingly.
    - If `temp_file` is not None, seeks to the beginning and copies its content to the appropriate output file.
    - Finally, updates the state to indicate that the weights have been written.
- **Output**: The method does not return a value; it writes tensor data to files and updates the internal state of the writer.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_ti_data_to_file`](#GGUFWriterwrite_ti_data_to_file)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.write_padding`](#GGUFWriterwrite_padding)
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.tofile`](lazy.py.driver.md#LazyNumpyTensortofile)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.flush`](#GGUFWriterflush)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.close`](#GGUFWriterclose)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.flush<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.flush}} -->
Flushes the output buffers for all open file handles.
- **Inputs**: None
- **Control Flow**:
    - Asserts that the `fout` attribute is not None, ensuring that there are open file handles.
    - Iterates over each file handle in `fout` and calls the `flush` method on each to clear the buffer.
- **Output**: This method does not return any value; it performs an action on the file handles.
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.close<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.close}} -->
Closes all open file handles associated with the GGUFWriter instance.
- **Inputs**: None
- **Control Flow**:
    - Checks if the `fout` attribute is not None, indicating that there are open file handles.
    - Iterates over each file handle in `fout` and calls the `close()` method on it to close the file.
    - Sets the `fout` attribute to None after closing all file handles.
- **Output**: This method does not return any value; it simply performs the action of closing the file handles.
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_type<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_type}} -->
Adds a type string to the GGUFWriter's key-value data.
- **Inputs**:
    - `type_name`: A string representing the type to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a predefined key and the provided type_name.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the type information.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_architecture<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_architecture}} -->
Adds the architecture information to the GGUFWriter instance.
- **Inputs**:
    - `self`: The instance of the GGUFWriter class.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with the architecture key and the architecture value stored in the instance.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance by adding architecture information.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_quantization\_version<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_quantization_version}} -->
Adds a quantization version to the GGUFWriter instance.
- **Inputs**:
    - `quantization_version`: An integer representing the quantization version to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a specific key and the provided quantization version.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance by adding the quantization version.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_custom\_alignment<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_custom_alignment}} -->
Sets a custom data alignment value for the GGUFWriter.
- **Inputs**:
    - `alignment`: An integer representing the desired data alignment value.
- **Control Flow**:
    - The method assigns the provided alignment value to the instance variable `data_alignment`.
    - It then calls the [`add_uint32`](#GGUFWriteradd_uint32) method to store the alignment value in the key-value data structure.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_file\_type<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_file_type}} -->
Adds a file type identifier to the GGUFWriter instance.
- **Inputs**:
    - `ftype`: An integer representing the file type to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a specific key and the provided file type.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_name<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_name}} -->
Adds a name to the general keys of the GGUFWriter.
- **Inputs**:
    - `name`: A string representing the name to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with the key `Keys.General.NAME` and the provided name.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the name.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_author<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_author}} -->
Adds an author string to the GGUFWriter's key-value data.
- **Inputs**:
    - `author`: A string representing the author's name to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a predefined key for author and the provided author string.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the author information.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_version<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_version}} -->
Adds a version string to the GGUFWriter's key-value data.
- **Inputs**:
    - `version`: A string representing the version to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a predefined key and the provided version string.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the version information.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_organization<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_organization}} -->
Adds an organization string to the GGUFWriter's key-value data.
- **Inputs**:
    - `organization`: A string representing the name of the organization to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a predefined key and the organization string.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the organization information.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_finetune<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_finetune}} -->
Adds a fine-tuning string to the GGUFWriter's key-value data.
- **Inputs**:
    - `finetune`: A string representing the fine-tuning information to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method of the `GGUFWriter` class to add the fine-tuning information under the key `Keys.General.FINETUNE`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the fine-tuning information.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_basename<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_basename}} -->
Adds a basename string to the general key-value data of the GGUFWriter.
- **Inputs**:
    - `basename`: A string representing the basename to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with the key `Keys.General.BASENAME` and the provided `basename`.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the basename.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_description<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_description}} -->
Adds a description string to the general metadata of the GGUF file.
- **Inputs**:
    - `description`: A string containing the description to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method of the `GGUFWriter` class to store the description under a specific key.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_quantized\_by<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_quantized_by}} -->
Adds a quantization descriptor to the GGUFWriter instance.
- **Inputs**:
    - `quantized`: A string representing the quantization method used.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method of the GGUFWriter instance to store the quantization information.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_size\_label<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_size_label}} -->
Adds a size label to the GGUFWriter instance.
- **Inputs**:
    - `size_label`: A string representing the size label to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method of the GGUFWriter instance to add the size label.
    - Uses a predefined key from the Keys.General class to associate the size label.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_license<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_license}} -->
Adds a license string to the GGUFWriter's key-value data.
- **Inputs**:
    - `license`: A string representing the license information to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a predefined key and the provided license string.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the license information.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_license\_name<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_license_name}} -->
Adds a license name to the GGUFWriter's key-value data.
- **Inputs**:
    - `license`: A string representing the name of the license to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with the key `Keys.General.LICENSE_NAME` and the provided license string.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the license name.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_license\_link<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_license_link}} -->
Adds a license link to the GGUFWriter's key-value data.
- **Inputs**:
    - `license`: A string representing the license link to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a predefined key and the provided license string.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the license link.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_url<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_url}} -->
The `add_url` method adds a URL string to the general keys of the GGUFWriter.
- **Inputs**:
    - `url`: A string representing the URL to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with the key `Keys.General.URL` and the provided `url` argument.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the URL.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_doi<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_doi}} -->
Adds a DOI (Digital Object Identifier) string to the GGUFWriter instance.
- **Inputs**:
    - `doi`: A string representing the Digital Object Identifier to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method of the GGUFWriter class to store the DOI under the specified key.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance by adding the DOI.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_uuid<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uuid}} -->
Adds a UUID string to the general keys of the GGUFWriter.
- **Inputs**:
    - `uuid`: A string representing the UUID to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with the key `Keys.General.UUID` and the provided `uuid`.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the UUID.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_repo\_url<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_repo_url}} -->
Adds a repository URL to the general keys of the GGUFWriter.
- **Inputs**:
    - `repo_url`: A string representing the URL of the repository to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with the key `Keys.General.REPO_URL` and the provided `repo_url`.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the repository URL.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_source\_url<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_source_url}} -->
Adds a source URL to the GGUFWriter's key-value data.
- **Inputs**:
    - `url`: A string representing the source URL to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with the key `Keys.General.SOURCE_URL` and the provided `url`.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the source URL.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_source\_doi<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_source_doi}} -->
Adds a source DOI to the GGUFWriter's key-value data.
- **Inputs**:
    - `doi`: A string representing the DOI (Digital Object Identifier) to be added as a source.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method of the `GGUFWriter` class to store the DOI under the key `Keys.General.SOURCE_DOI`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the DOI.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_source\_uuid<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_source_uuid}} -->
Adds a source UUID to the GGUFWriter's key-value data.
- **Inputs**:
    - `uuid`: A string representing the UUID to be added as a source identifier.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with the key `Keys.General.SOURCE_UUID` and the provided `uuid`.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the UUID to its key-value data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_source\_repo\_url<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_source_repo_url}} -->
Adds a source repository URL to the GGUFWriter instance.
- **Inputs**:
    - `repo_url`: A string representing the URL of the source repository.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method of the GGUFWriter instance to store the repository URL under the key `Keys.General.SOURCE_REPO_URL`.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_base\_model\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_count}} -->
Adds the count of base models to the GGUFWriter's key-value data.
- **Inputs**:
    - `source_count`: An integer representing the count of base models to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a specific key and the provided `source_count`.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the base model count.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_base\_model\_name<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_name}} -->
Adds the name of a base model to the key-value data structure.
- **Inputs**:
    - `source_id`: An integer representing the identifier of the source model.
    - `name`: A string representing the name of the base model.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a formatted key and the provided name.
    - The key is generated using the `Keys.General.BASE_MODEL_NAME` format string, which incorporates the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the base model name to the key-value data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_base\_model\_author<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_author}} -->
Adds the author information for a specified base model to the GGUFWriter.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the base model.
    - `author`: A string containing the name of the author of the base model.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a formatted key and the author name.
    - The key is generated using the `Keys.General.BASE_MODEL_AUTHOR` format string, which incorporates the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the author information.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_base\_model\_version<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_version}} -->
Adds a version string for a base model identified by a source ID.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the source model.
    - `version`: A string representing the version of the base model.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a formatted key and the version string.
- **Output**: This method does not return any value; it updates the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_base\_model\_organization<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_organization}} -->
Adds the organization name associated with a base model to the internal data structure.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the source model.
    - `organization`: A string representing the name of the organization associated with the base model.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method to store the organization name using a formatted key that includes the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the organization information.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_base\_model\_description<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_description}} -->
Adds a description for a base model identified by a source ID.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the base model.
    - `description`: A string containing the description of the base model.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a formatted key and the provided description.
    - The key is generated using the `Keys.General.BASE_MODEL_DESCRIPTION` format method, which incorporates the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the base model description.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_base\_model\_url<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_url}} -->
Adds a URL for a base model identified by a source ID.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the base model.
    - `url`: A string containing the URL associated with the base model.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method to store the URL using a formatted key that includes the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the URL.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_base\_model\_doi<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_doi}} -->
Adds a DOI (Digital Object Identifier) for a base model identified by a source ID.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the source model.
    - `doi`: A string representing the Digital Object Identifier to be associated with the base model.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a formatted key and the DOI value.
    - The key is generated using the `Keys.General.BASE_MODEL_DOI` format string, which incorporates the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the DOI to its key-value data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_base\_model\_uuid<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_uuid}} -->
Adds a UUID for a base model associated with a given source ID.
- **Inputs**:
    - `source_id`: An integer representing the identifier for the source model.
    - `uuid`: A string representing the universally unique identifier for the base model.
- **Control Flow**:
    - The method calls [`add_string`](#GGUFWriteradd_string) with a formatted key and the provided UUID.
    - The key is generated using `Keys.General.BASE_MODEL_UUID` with the `source_id` inserted into the format.
- **Output**: The method does not return any value; it modifies the internal state by adding the UUID to the key-value store.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_base\_model\_repo\_url<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_base_model_repo_url}} -->
Adds a repository URL for a base model identified by a source ID.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the source model.
    - `repo_url`: A string containing the URL of the repository for the base model.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method to store the repository URL using a formatted key that includes the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the repository URL.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_dataset\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_count}} -->
Adds a dataset count to the GGUFWriter's key-value data.
- **Inputs**:
    - `source_count`: An integer representing the count of datasets to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method to store the dataset count in the key-value data using a predefined key.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the dataset count.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_dataset\_name<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_name}} -->
Adds a dataset name associated with a specific source ID to the GGUFWriter.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the dataset.
    - `name`: A string representing the name of the dataset.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method to store the dataset name using a formatted key that includes the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` by adding the dataset name.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_dataset\_author<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_author}} -->
Adds an author to a dataset identified by a source ID.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the dataset.
    - `author`: A string representing the name of the author to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method of the `GGUFWriter` class to store the author information.
    - The key for storing the author is formatted using `Keys.General.DATASET_AUTHOR` with the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_dataset\_version<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_version}} -->
Adds a version string for a dataset identified by a source ID.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the dataset.
    - `version`: A string representing the version of the dataset.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method to store the dataset version using a formatted key that includes the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the dataset version.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_dataset\_organization<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_organization}} -->
Adds an organization name to a dataset identified by a source ID.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the dataset.
    - `organization`: A string representing the name of the organization associated with the dataset.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method to store the organization name using a formatted key that includes the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the organization information.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_dataset\_description<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_description}} -->
Adds a description for a dataset identified by a source ID.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the dataset.
    - `description`: A string containing the description of the dataset.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a formatted key and the provided description.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the dataset description.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_dataset\_url<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_url}} -->
Adds a dataset URL associated with a given source ID to the internal key-value data structure.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the dataset.
    - `url`: A string containing the URL of the dataset.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with a formatted key and the provided URL.
    - The key is generated using the `Keys.General.DATASET_URL` format string, which incorporates the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_dataset\_doi<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_doi}} -->
Adds a dataset DOI (Digital Object Identifier) to the GGUFWriter instance.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the dataset.
    - `doi`: A string representing the Digital Object Identifier for the dataset.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method of the `GGUFWriter` class.
    - Formats the key for the DOI using the `Keys.General.DATASET_DOI` template with the provided `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the DOI.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_dataset\_uuid<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_uuid}} -->
Adds a UUID for a dataset associated with a given source ID.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the source.
    - `uuid`: A string representing the universally unique identifier for the dataset.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method to store the UUID in the internal key-value data structure using a formatted key that includes the source ID.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_dataset\_repo\_url<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_dataset_repo_url}} -->
Adds a repository URL for a dataset identified by a source ID.
- **Inputs**:
    - `source_id`: An integer representing the unique identifier for the dataset.
    - `repo_url`: A string containing the URL of the repository associated with the dataset.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method to store the repository URL in the internal key-value data structure.
    - The key for storing the URL is formatted using the `Keys.General.DATASET_REPO_URL` template, which incorporates the `source_id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the repository URL.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_tags<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tags}} -->
Adds a sequence of tags to the GGUFWriter instance.
- **Inputs**:
    - `tags`: A sequence of strings representing the tags to be added.
- **Control Flow**:
    - Calls the [`add_array`](#GGUFWriteradd_array) method of the GGUFWriter instance to add the tags under the key specified by `Keys.General.TAGS`.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance by adding the provided tags.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_languages<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_languages}} -->
Adds a list of languages to the GGUFWriter instance.
- **Inputs**:
    - `languages`: A sequence of strings representing the languages to be added.
- **Control Flow**:
    - Calls the [`add_array`](#GGUFWriteradd_array) method of the GGUFWriter instance to add the languages under the key specified by `Keys.General.LANGUAGES`.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_tensor\_data\_layout<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tensor_data_layout}} -->
Adds a tensor data layout string to the GGUFWriter instance.
- **Inputs**:
    - `layout`: A string representing the layout of the tensor data.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method to add the tensor data layout to the internal key-value data structure.
    - Formats the key using the architecture of the writer instance.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vocab\_size<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vocab_size}} -->
Adds the vocabulary size to the GGUFWriter instance.
- **Inputs**:
    - `size`: An integer representing the vocabulary size to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided size.
    - The key is generated using the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_context\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_context_length}} -->
Adds the context length to the GGUFWriter's key-value data.
- **Inputs**:
    - `length`: An integer representing the context length to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method to add the context length to the key-value data using a formatted key.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the context length.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_embedding\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_embedding_length}} -->
Adds the specified embedding length to the GGUFWriter's key-value data.
- **Inputs**:
    - `length`: An integer representing the embedding length to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to add the embedding length.
    - The key for the embedding length is formatted using the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the embedding length.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_features\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_features_length}} -->
Adds the specified length as a uint32 value to the GGUFWriter's key-value data.
- **Inputs**:
    - `length`: An integer representing the length to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided length.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding a key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_posnet\_embedding\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_posnet_embedding_length}} -->
Adds the length of the PosNet embedding to the GGUFWriter's key-value data.
- **Inputs**:
    - `length`: An integer representing the length of the PosNet embedding.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to store the embedding length.
    - The key for storing the length is formatted using `Keys.PosNet.EMBEDDING_LENGTH` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_posnet\_block\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_posnet_block_count}} -->
Adds a block count for the PosNet architecture to the GGUFWriter.
- **Inputs**:
    - `length`: An integer representing the number of blocks to be added for the PosNet architecture.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to store the block count.
    - The key for storage is formatted using the `Keys.PosNet.BLOCK_COUNT` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` by adding the specified block count.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_convnext\_embedding\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_convnext_embedding_length}} -->
Adds the ConvNext embedding length to the GGUFWriter.
- **Inputs**:
    - `length`: An integer representing the embedding length to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the GGUFWriter class.
    - Formats the key using `Keys.ConvNext.EMBEDDING_LENGTH` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the specified embedding length.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_convnext\_block\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_convnext_block_count}} -->
Adds a ConvNext block count to the GGUFWriter's key-value data.
- **Inputs**:
    - `length`: An integer representing the number of ConvNext blocks to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the GGUFWriter class to add the block count.
    - The key for the block count is formatted using the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the block count.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_block\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_block_count}} -->
Adds a block count to the GGUFWriter instance.
- **Inputs**:
    - `length`: An integer representing the block count to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided length.
- **Output**: This method does not return any value; it updates the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_leading\_dense\_block\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_leading_dense_block_count}} -->
Adds the leading dense block count to the GGUFWriter instance.
- **Inputs**:
    - `length`: An integer representing the count of leading dense blocks to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the GGUFWriter instance.
    - Formats the key using `Keys.LLM.LEADING_DENSE_BLOCK_COUNT` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_feed\_forward\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_feed_forward_length}} -->
Adds a feed-forward length to the GGUFWriter, either as a single integer or a sequence of integers.
- **Inputs**:
    - `length`: An integer or a sequence of integers representing the feed-forward length to be added.
- **Control Flow**:
    - Checks if the input `length` is an instance of `int`.
    - If `length` is an integer, it calls [`add_uint32`](#GGUFWriteradd_uint32) to add the feed-forward length.
    - If `length` is a sequence, it calls [`add_array`](#GGUFWriteradd_array) to add the feed-forward lengths.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the specified feed-forward length.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_expert\_feed\_forward\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_feed_forward_length}} -->
Adds the expert feed forward length to the GGUFWriter instance.
- **Inputs**:
    - `length`: An integer representing the length of the expert feed forward.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the GGUFWriter instance.
    - Formats the key using the architecture attribute and adds the length value.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_expert\_shared\_feed\_forward\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_shared_feed_forward_length}} -->
Adds the expert shared feed forward length to the GGUFWriter's key-value data.
- **Inputs**:
    - `length`: An integer representing the length of the expert shared feed forward.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided length.
    - The key is generated using the architecture attribute of the GGUFWriter instance.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_parallel\_residual<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_parallel_residual}} -->
Sets the parallel residual configuration for the model.
- **Inputs**:
    - `use`: A boolean indicating whether to enable or disable parallel residual.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method to store the value of `use` under a specific key formatted with the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding a boolean configuration.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_decoder\_start\_token\_id<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_decoder_start_token_id}} -->
Adds a decoder start token ID to the GGUFWriter instance.
- **Inputs**:
    - `id`: An integer representing the decoder start token ID to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to store the decoder start token ID.
    - The key for storage is formatted using the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the decoder start token ID.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_head\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_head_count}} -->
Adds head count information to the GGUFWriter instance.
- **Inputs**:
    - `count`: An integer or a sequence of integers representing the head count.
- **Control Flow**:
    - Checks if the input 'count' is an integer.
    - If 'count' is an integer, calls the method 'add_uint32' with a formatted key and the count.
    - If 'count' is a sequence, calls the method 'add_array' with a formatted key and the sequence.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance by adding head count data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_head\_count\_kv<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_head_count_kv}} -->
Adds head count key-value data to the GGUFWriter instance.
- **Inputs**:
    - `count`: An integer or a sequence of integers representing the head count.
- **Control Flow**:
    - Checks if the input 'count' is an integer.
    - If 'count' is an integer, calls 'add_uint32' to add the head count.
    - If 'count' is a sequence, calls 'add_array' to add the head count array.
- **Output**: The method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_key\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_length}} -->
Adds a key length value to the GGUFWriter's key-value data.
- **Inputs**:
    - `length`: An integer representing the length of the key to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided length.
    - The key is formatted using the architecture of the writer.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the key length.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_value\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_value_length}} -->
Adds the specified value length to the GGUFWriter's key-value data.
- **Inputs**:
    - `length`: An integer representing the length of the value to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided length.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the value length.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_key\_length\_mla<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_key_length_mla}} -->
Adds a key length value for the multi-layer attention mechanism to the GGUFWriter.
- **Inputs**:
    - `length`: An integer representing the length of the key for the multi-layer attention.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to add the key length value.
    - The key for the [`add_uint32`](#GGUFWriteradd_uint32) method is formatted using the `KEY_LENGTH_MLA` key from the `Keys.Attention` class, incorporating the architecture of the writer.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` by adding the key length.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_value\_length\_mla<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_value_length_mla}} -->
Adds a value length for the multi-layer attention mechanism in the GGUFWriter.
- **Inputs**:
    - `length`: An integer representing the length of the value to be added for the multi-layer attention.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to add the value length.
    - The key for the value length is formatted using the `Keys.Attention.VALUE_LENGTH_MLA` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` by adding the specified value length.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_max\_alibi\_bias<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_max_alibi_bias}} -->
Adds a maximum ALIBI bias value to the attention configuration.
- **Inputs**:
    - `bias`: A float representing the maximum ALIBI bias to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the parent class `GGUFWriter` to store the bias value.
    - The key for storing the bias is generated using the `Keys.Attention.MAX_ALIBI_BIAS` format string, which incorporates the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the specified bias.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_clamp\_kqv<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_clamp_kqv}} -->
Adds a float32 value to the key-value data structure with a specific key format.
- **Inputs**:
    - `value`: A float value to be added to the key-value data.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method with a formatted key and the provided value.
- **Output**: None, as it modifies the internal state of the object by adding the value to the key-value data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_logit\_scale<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_logit_scale}} -->
Adds a logit scale value to the GGUFWriter instance.
- **Inputs**:
    - `value`: A float representing the logit scale to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method with a formatted key and the provided value.
    - The key is generated using the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance by adding the logit scale.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_attn\_logit\_softcapping<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_attn_logit_softcapping}} -->
Adds a float32 value for attention logit softcapping to the GGUFWriter's key-value data.
- **Inputs**:
    - `value`: A float representing the attention logit softcapping value to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the `GGUFWriter` class to store the value.
    - The key for storing the value is generated using a formatted string that includes the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the specified float32 value.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_final\_logit\_softcapping<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_final_logit_softcapping}} -->
Adds a final logit softcapping value to the GGUFWriter instance.
- **Inputs**:
    - `value`: A float representing the final logit softcapping value to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the `GGUFWriter` class.
    - Formats the key using `Keys.LLM.FINAL_LOGIT_SOFTCAPPING` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_expert\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_count}} -->
Adds an expert count to the GGUFWriter instance.
- **Inputs**:
    - `count`: An integer representing the number of experts to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided count.
    - The key is generated using the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_expert\_used\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_used_count}} -->
Increments the count of experts used by a specified integer value.
- **Inputs**:
    - `count`: An integer representing the number of experts used to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided count.
- **Output**: This method does not return any value; it updates the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_expert\_shared\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_shared_count}} -->
This method adds a specified count of expert shared parameters to the GGUFWriter's key-value data.
- **Inputs**:
    - `count`: An integer representing the number of expert shared parameters to be added.
- **Control Flow**:
    - The method calls [`add_uint32`](#GGUFWriteradd_uint32) on the current instance, passing a formatted key and the count value.
    - The key is generated using the `Keys.LLM.EXPERT_SHARED_COUNT` format string, which incorporates the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the expert shared count.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_expert\_weights\_scale<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_weights_scale}} -->
Sets the expert weights scale for the model.
- **Inputs**:
    - `value`: A float representing the scale of the expert weights.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the parent class `GGUFWriter`.
    - Formats the key for the expert weights scale using the architecture attribute.
- **Output**: None, but updates the internal state of the `GGUFWriter` with the expert weights scale.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_expert\_weights\_norm<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_weights_norm}} -->
Sets the expert weights normalization flag in the GGUFWriter.
- **Inputs**:
    - `value`: A boolean indicating whether to enable or disable expert weights normalization.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method of the `GGUFWriter` class with a formatted key and the provided value.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_expert\_gating\_func<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_expert_gating_func}} -->
Adds an expert gating function value to the GGUFWriter instance.
- **Inputs**:
    - `value`: An instance of ExpertGatingFuncType representing the gating function value to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the GGUFWriter instance to store the expert gating function value.
    - The key for storage is formatted using the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance by adding the expert gating function value.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_moe\_every\_n\_layers<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_moe_every_n_layers}} -->
Adds a specified integer value to the configuration for the number of layers in a mixture of experts (MoE) model.
- **Inputs**:
    - `value`: An integer representing the number of layers for the mixture of experts.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the parent class `GGUFWriter`.
    - Formats the key using `Keys.LLM.MOE_EVERY_N_LAYERS` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the specified value.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_swin\_norm<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_swin_norm}} -->
Sets the SWIN normalization flag for the architecture.
- **Inputs**:
    - `value`: A boolean indicating whether SWIN normalization should be enabled or disabled.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method to add the SWIN normalization key-value pair to the internal data structure.
    - The key is formatted using the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding a key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_rescale\_every\_n\_layers<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rescale_every_n_layers}} -->
Adds a uint32 value representing the number of layers to rescale in a model.
- **Inputs**:
    - `count`: An integer representing the number of layers after which rescaling should occur.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the parent class `GGUFWriter`.
    - Formats the key using `Keys.LLM.RESCALE_EVERY_N_LAYERS` with the current architecture.
- **Output**: None, as it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_time\_mix\_extra\_dim<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_time_mix_extra_dim}} -->
Adds an extra dimension for time mixing in the model's configuration.
- **Inputs**:
    - `dim`: An integer representing the extra dimension to be added for time mixing.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided dimension value.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding a new configuration parameter.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_time\_decay\_extra\_dim<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_time_decay_extra_dim}} -->
Adds a time decay extra dimension to the GGUFWriter.
- **Inputs**:
    - `dim`: An integer representing the dimension to be added for time decay.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to add the time decay dimension.
    - The key for the dimension is formatted using the `Keys.LLM.TIME_DECAY_EXTRA_DIM` template with the architecture of the writer.
- **Output**: The method does not return any value; it modifies the internal state of the GGUFWriter by adding the specified dimension.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_residual\_scale<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_residual_scale}} -->
Adds a residual scale value to the GGUFWriter instance.
- **Inputs**:
    - `value`: A float representing the residual scale to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the `GGUFWriter` class to store the residual scale value.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_embedding\_scale<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_embedding_scale}} -->
Adds a float32 value representing the embedding scale to the GGUFWriter.
- **Inputs**:
    - `value`: A float representing the embedding scale to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the `GGUFWriter` class with a formatted key and the provided value.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` by adding the embedding scale.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_wkv\_head\_size<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_wkv_head_size}} -->
Adds the specified head size to the WKV (Weight Key Value) data structure.
- **Inputs**:
    - `size`: An integer representing the head size to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method to add the head size to the WKV data structure using a formatted key.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the head size.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_token\_shift\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_shift_count}} -->
Adds a token shift count to the GGUFWriter instance.
- **Inputs**:
    - `count`: An integer representing the token shift count to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided count.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance by adding the token shift count.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_interleave\_moe\_layer\_step<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_interleave_moe_layer_step}} -->
Adds a step value for interleaving in a mixture of experts layer.
- **Inputs**:
    - `value`: An integer representing the step value for the interleaved mixture of experts layer.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the parent class `GGUFWriter`.
    - Formats the key using `Keys.LLM.INTERLEAVE_MOE_LAYER_STEP` with the current architecture and passes the value.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the specified step value.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_layer\_norm\_eps<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_layer_norm_eps}} -->
Adds a layer normalization epsilon value to the GGUFWriter instance.
- **Inputs**:
    - `value`: A float representing the layer normalization epsilon value to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the GGUFWriter class to store the layer normalization epsilon value.
    - The key for storage is generated using the `Keys.Attention.LAYERNORM_EPS` format string, which incorporates the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance by adding the specified epsilon value.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_layer\_norm\_rms\_eps<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_layer_norm_rms_eps}} -->
Adds a layer normalization RMS epsilon value to the GGUFWriter instance.
- **Inputs**:
    - `value`: A float representing the RMS epsilon value to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the `GGUFWriter` class.
    - Formats the key using `Keys.Attention.LAYERNORM_RMS_EPS` with the current architecture.
- **Output**: The method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_group\_norm\_eps<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_group_norm_eps}} -->
Adds a float32 value representing the group normalization epsilon to the GGUFWriter's key-value data.
- **Inputs**:
    - `value`: A float representing the group normalization epsilon to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the `GGUFWriter` class to store the value.
    - The key for storage is generated using the `Keys.Attention.GROUPNORM_EPS` format string, which incorporates the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` by adding the specified epsilon value.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_group\_norm\_groups<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_group_norm_groups}} -->
Adds a group normalization parameter to the GGUFWriter instance.
- **Inputs**:
    - `value`: An integer representing the number of groups for group normalization.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class.
    - Formats the key for group normalization groups using the architecture of the writer.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_causal\_attention<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_causal_attention}} -->
Sets the causal attention flag in the GGUFWriter instance.
- **Inputs**:
    - `value`: A boolean indicating whether causal attention should be enabled.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method of the `GGUFWriter` class to store the causal attention setting.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_q\_lora\_rank<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_q_lora_rank}} -->
Adds the Q LoRA rank to the GGUFWriter instance.
- **Inputs**:
    - `length`: An integer representing the Q LoRA rank to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided length.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance by adding the Q LoRA rank.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_kv\_lora\_rank<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_kv_lora_rank}} -->
Adds a key-value pair representing the KV LoRA rank to the GGUFWriter.
- **Inputs**:
    - `length`: An integer representing the KV LoRA rank to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided length.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the KV LoRA rank.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_decay\_lora\_rank<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_decay_lora_rank}} -->
Adds a decay LoRA rank to the GGUFWriter instance.
- **Inputs**:
    - `length`: An integer representing the decay LoRA rank to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the GGUFWriter instance to add the decay LoRA rank.
    - The key for the decay LoRA rank is formatted using the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_iclr\_lora\_rank<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_iclr_lora_rank}} -->
Adds the ICLR LoRA rank to the GGUFWriter instance.
- **Inputs**:
    - `length`: An integer representing the ICLR LoRA rank to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the GGUFWriter class to add the ICLR LoRA rank.
    - The key for the rank is formatted using the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_value\_residual\_mix\_lora\_rank<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_value_residual_mix_lora_rank}} -->
Adds a value for the residual mix LoRA rank to the GGUFWriter.
- **Inputs**:
    - `length`: An integer representing the length of the value to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to add the value associated with the key formatted for the current architecture.
    - The key is generated using `Keys.Attention.VALUE_RESIDUAL_MIX_LORA_RANK` with the current architecture (`self.arch`) included in the format.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` by adding a key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_gate\_lora\_rank<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_gate_lora_rank}} -->
Adds a gate LoRA rank to the GGUFWriter instance.
- **Inputs**:
    - `length`: An integer representing the gate LoRA rank to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the GGUFWriter instance.
    - Formats the key using `Keys.Attention.GATE_LORA_RANK` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_relative\_attn\_buckets\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_relative_attn_buckets_count}} -->
Adds a count of relative attention buckets to the GGUFWriter instance.
- **Inputs**:
    - `value`: An integer representing the count of relative attention buckets to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to store the relative attention buckets count.
    - The key for storage is formatted using the architecture of the writer instance.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_sliding\_window<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_sliding_window}} -->
Adds a sliding window value to the attention configuration.
- **Inputs**:
    - `value`: An integer representing the sliding window size to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided value to store the sliding window size.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_attention\_scale<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_attention_scale}} -->
Adds a float32 value representing the attention scale to the GGUFWriter's key-value data.
- **Inputs**:
    - `value`: A float representing the attention scale to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the `GGUFWriter` class to store the attention scale value.
    - The key for the value is formatted using `Keys.Attention.SCALE` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the attention scale.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_pooling\_type<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_pooling_type}} -->
Adds a pooling type to the GGUFWriter instance.
- **Inputs**:
    - `value`: An instance of `PoolingType` that represents the pooling type to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method to add the pooling type value to the internal key-value data structure.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_rope\_dimension\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_dimension_count}} -->
Adds a specified count of rope dimensions to the GGUFWriter instance.
- **Inputs**:
    - `count`: An integer representing the number of rope dimensions to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to store the dimension count.
    - The dimension count is formatted using a key that includes the architecture of the writer.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_rope\_dimension\_sections<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_dimension_sections}} -->
Adds rope dimension sections to the GGUFWriter instance.
- **Inputs**:
    - `dims`: A sequence of integers representing the dimensions of the rope sections to be added.
- **Control Flow**:
    - Calls the [`add_array`](#GGUFWriteradd_array) method with a formatted key and the provided dimensions.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance by adding the specified rope dimension sections.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_rope\_freq\_base<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_freq_base}} -->
Adds a frequency base value for rope encoding to the GGUFWriter.
- **Inputs**:
    - `value`: A float representing the frequency base value to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the GGUFWriter class.
    - Formats the key using `Keys.Rope.FREQ_BASE` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the frequency base.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_rope\_scaling\_type<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_scaling_type}} -->
Adds a rope scaling type to the GGUFWriter instance.
- **Inputs**:
    - `value`: An instance of `RopeScalingType` that represents the scaling type to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method to add the scaling type to the internal key-value data structure.
    - Formats the key using `Keys.Rope.SCALING_TYPE` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the specified scaling type.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_rope\_scaling\_factor<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_scaling_factor}} -->
Adds a scaling factor for rope embeddings to the GGUFWriter.
- **Inputs**:
    - `value`: A float representing the scaling factor to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method with a formatted key and the provided value.
    - The key is generated using the `Keys.Rope.SCALING_FACTOR` format string, which includes the architecture of the model.
- **Output**: The method does not return any value; it updates the internal state of the GGUFWriter with the new scaling factor.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_rope\_scaling\_attn\_factors<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_scaling_attn_factors}} -->
Adds a scaling attention factor to the rope configuration in the GGUFWriter.
- **Inputs**:
    - `value`: A float representing the scaling attention factor to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the parent class `GGUFWriter`.
    - Formats the key for the scaling attention factor using the architecture attribute.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_rope\_scaling\_orig\_ctx\_len<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_scaling_orig_ctx_len}} -->
Adds the original context length for rope scaling to the GGUFWriter's key-value data.
- **Inputs**:
    - `value`: An integer representing the original context length for rope scaling.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class.
    - Formats the key for the context length using the `Keys.Rope.SCALING_ORIG_CTX_LEN` template with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` by adding a key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_rope\_scaling\_finetuned<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_scaling_finetuned}} -->
Sets a boolean value indicating whether rope scaling has been fine-tuned.
- **Inputs**:
    - `value`: A boolean indicating the fine-tuning status of rope scaling.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method of the parent class `GGUFWriter` to store the boolean value.
    - Formats the key for the boolean value using `Keys.Rope.SCALING_FINETUNED` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_rope\_scaling\_yarn\_log\_mul<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_rope_scaling_yarn_log_mul}} -->
This method adds a float32 value to a specific key in the GGUFWriter's key-value data structure.
- **Inputs**:
    - `value`: A float value to be added to the key corresponding to the scaling yarn log multiplier.
- **Control Flow**:
    - The method calls [`add_float32`](#GGUFWriteradd_float32) with a formatted key and the provided value.
    - The key is generated using the `Keys.Rope.SCALING_YARN_LOG_MUL` format, which incorporates the architecture of the writer.
- **Output**: The method does not return any value; it modifies the internal state of the GGUFWriter by adding the specified float value to its key-value data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_ssm\_conv\_kernel<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_ssm_conv_kernel}} -->
Adds a 32-bit unsigned integer value to the SSM convolution kernel key in the GGUFWriter.
- **Inputs**:
    - `value`: An integer representing the convolution kernel value to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to add the value to the specified key.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` by adding the specified value.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_ssm\_inner\_size<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_ssm_inner_size}} -->
Adds the inner size parameter for the SSM architecture to the GGUFWriter.
- **Inputs**:
    - `value`: An integer representing the inner size to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided value.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the inner size.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_ssm\_state\_size<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_ssm_state_size}} -->
Adds the state size for the SSM architecture to the GGUFWriter.
- **Inputs**:
    - `value`: An integer representing the state size to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided value.
- **Output**: This method does not return any value; it updates the internal state of the GGUFWriter.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_ssm\_time\_step\_rank<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_ssm_time_step_rank}} -->
Adds a time step rank value to the SSM (State Space Model) configuration.
- **Inputs**:
    - `value`: An integer representing the time step rank to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a formatted key and the provided value to store the time step rank.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_ssm\_dt\_b\_c\_rms<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_ssm_dt_b_c_rms}} -->
Adds a boolean value to the SSM data structure.
- **Inputs**:
    - `value`: A boolean value to be added to the SSM data.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method of the parent class `GGUFWriter`.
    - Formats the key using `Keys.SSM.DT_B_C_RMS` with the current architecture.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the boolean value.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_tokenizer\_model<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tokenizer_model}} -->
Adds a tokenizer model string to the GGUFWriter instance.
- **Inputs**:
    - `model`: A string representing the tokenizer model to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with the key `Keys.Tokenizer.MODEL` and the provided model string.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter instance by adding the tokenizer model.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_tokenizer\_pre<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_tokenizer_pre}} -->
Adds a string value to the tokenizer's pre-tokenization settings.
- **Inputs**:
    - `pre`: A string representing the pre-tokenization setting to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method with the key `Keys.Tokenizer.PRE` and the provided `pre` string.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_token\_list<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_list}} -->
Adds a list of tokens to the tokenizer's data structure.
- **Inputs**:
    - `tokens`: A sequence of tokens which can be strings, bytes, or bytearrays.
- **Control Flow**:
    - Calls the [`add_array`](#GGUFWriteradd_array) method of the parent class `GGUFWriter` to add the tokens under the key `Keys.Tokenizer.LIST`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the provided tokens.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_token\_merges<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_merges}} -->
Adds token merges to the tokenizer's configuration.
- **Inputs**:
    - `merges`: A sequence of strings, bytes, or bytearrays representing the token merges to be added.
- **Control Flow**:
    - Calls the [`add_array`](#GGUFWriteradd_array) method of the `GGUFWriter` class to add the merges to the tokenizer's configuration under the key `Keys.Tokenizer.MERGES`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the specified token merges.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_token\_types<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_types}} -->
Adds token types to the tokenizer's configuration.
- **Inputs**:
    - `types`: A sequence of `TokenType` or integer values representing the types of tokens to be added.
- **Control Flow**:
    - Calls the [`add_array`](#GGUFWriteradd_array) method of the parent class `GGUFWriter` to add the token types under the key `Keys.Tokenizer.TOKEN_TYPE`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the specified token types.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_token\_type\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_type_count}} -->
Adds a count of token types to the tokenizer's metadata.
- **Inputs**:
    - `value`: An integer representing the count of token types to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a specific key and the provided value to store the token type count.
- **Output**: This method does not return any value; it updates the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_token\_scores<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_token_scores}} -->
Adds token scores to the tokenizer's metadata.
- **Inputs**:
    - `scores`: A sequence of floating-point numbers representing the scores for tokens.
- **Control Flow**:
    - Calls the [`add_array`](#GGUFWriteradd_array) method to add the scores to the tokenizer's metadata under the key `Keys.Tokenizer.SCORES`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the provided scores.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_bos\_token\_id<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bos_token_id}} -->
Adds the beginning-of-sequence (BOS) token ID to the tokenizer's key-value data.
- **Inputs**:
    - `id`: An integer representing the ID of the beginning-of-sequence token.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with the key for the BOS ID and the provided ID.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the BOS token ID.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_eos\_token\_id<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_eos_token_id}} -->
Adds an end-of-sequence (EOS) token ID to the tokenizer.
- **Inputs**:
    - `id`: An integer representing the EOS token ID to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with the key `Keys.Tokenizer.EOS_ID` and the provided `id`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the EOS token ID.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_unk\_token\_id<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_unk_token_id}} -->
Adds an unknown token ID to the tokenizer's key-value data.
- **Inputs**:
    - `id`: An integer representing the unknown token ID to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with the key for the unknown token ID and the provided ID.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the unknown token ID.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_sep\_token\_id<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_sep_token_id}} -->
Adds a separator token ID to the tokenizer's key-value data.
- **Inputs**:
    - `id`: An integer representing the separator token ID to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with the key for the separator token ID and the provided ID.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the separator token ID.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_pad\_token\_id<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_pad_token_id}} -->
Adds a padding token ID to the tokenizer.
- **Inputs**:
    - `id`: An integer representing the padding token ID to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with the key for the padding token ID and the provided ID.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the padding token ID.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_mask\_token\_id<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_mask_token_id}} -->
Adds a mask token ID to the tokenizer's key-value data.
- **Inputs**:
    - `id`: An integer representing the mask token ID to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with the key for the mask ID and the provided ID.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the mask token ID.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_add\_bos\_token<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_add_bos_token}} -->
Sets the boolean value for adding a beginning-of-sequence (BOS) token in the tokenizer.
- **Inputs**:
    - `value`: A boolean indicating whether to add a beginning-of-sequence (BOS) token.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method of the parent class `GGUFWriter` to store the value associated with the key `Keys.Tokenizer.ADD_BOS`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_add\_eos\_token<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_add_eos_token}} -->
Adds a boolean value indicating whether to include an end-of-sequence (EOS) token in the tokenizer.
- **Inputs**:
    - `value`: A boolean indicating whether to add the EOS token.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method of the parent class `GGUFWriter` to store the EOS token setting.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_add\_space\_prefix<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_add_space_prefix}} -->
Sets a boolean value indicating whether to add a space prefix in the tokenizer.
- **Inputs**:
    - `value`: A boolean indicating whether to add a space prefix.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method with the key `Keys.Tokenizer.ADD_PREFIX` and the provided `value`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_remove\_extra\_whitespaces<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_remove_extra_whitespaces}} -->
Sets the flag to remove extra whitespaces in the tokenizer.
- **Inputs**:
    - `value`: A boolean indicating whether to remove extra whitespaces.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method with the key `Keys.Tokenizer.REMOVE_EXTRA_WS` and the provided value.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_precompiled\_charsmap<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_precompiled_charsmap}} -->
Adds a precompiled character mapping to the tokenizer's key-value data.
- **Inputs**:
    - `charsmap`: A bytes object representing the precompiled character mapping to be added.
- **Control Flow**:
    - Calls the [`add_array`](#GGUFWriteradd_array) method of the `GGUFWriter` class to add the `charsmap` to the key specified by `Keys.Tokenizer.PRECOMPILED_CHARSMAP`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the character mapping.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_chat\_template<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_chat_template}} -->
Adds chat templates to the writer, either as a single string or a sequence of mappings.
- **Inputs**:
    - `value`: A string or a sequence of mappings, where each mapping contains a 'name' and a 'template'.
- **Control Flow**:
    - Checks if the input 'value' is a string; if not, it processes each mapping in the sequence.
    - For each mapping, it retrieves the 'name' and 'template', filtering the name to allow only alphanumeric characters.
    - If the name is 'default', it sets 'template_default' to the corresponding template; otherwise, it adds the name to 'template_names' and stores the template.
    - If there are any template names, it adds them to the writer's chat templates.
    - If 'template_default' is not set, the function returns early; otherwise, it sets 'value' to 'template_default'.
    - Finally, it adds the final 'value' as a chat template string.
- **Output**: The method does not return a value; it modifies the internal state of the writer by adding chat templates.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_eot\_token\_id<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_eot_token_id}} -->
Adds the end-of-text (EOT) token ID to the tokenizer's key-value data.
- **Inputs**:
    - `id`: An integer representing the end-of-text token ID to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with the key for EOT ID and the provided ID.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the EOT token ID.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_eom\_token\_id<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_eom_token_id}} -->
Adds the end-of-message (EOM) token ID to the tokenizer's key-value data.
- **Inputs**:
    - `id`: An integer representing the end-of-message token ID to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with the key for the EOM ID and the provided ID.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_classifier\_output\_labels<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_classifier_output_labels}} -->
Adds classifier output labels to the GGUFWriter instance.
- **Inputs**:
    - `labels`: A sequence of strings representing the output labels for the classifier.
- **Control Flow**:
    - Calls the [`add_array`](#GGUFWriteradd_array) method of the `GGUFWriter` class to store the labels.
    - The key used for storage is formatted with the architecture of the model.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_clip\_has\_vision\_encoder<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_clip_has_vision_encoder}} -->
Sets the boolean value indicating whether the CLIP model has a vision encoder.
- **Inputs**:
    - `value`: A boolean indicating the presence of a vision encoder in the CLIP model.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method of the parent class `GGUFWriter` to store the value associated with the key `Keys.Clip.HAS_VISION_ENCODER`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_clip\_has\_audio\_encoder<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_clip_has_audio_encoder}} -->
Sets a boolean value indicating whether the clip has an audio encoder.
- **Inputs**:
    - `value`: A boolean indicating the presence of an audio encoder in the clip.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method of the parent class `GGUFWriter` to store the boolean value associated with the key `Keys.Clip.HAS_AUDIO_ENCODER`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_clip\_projector\_type<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_clip_projector_type}} -->
Adds a projector type string to the clip metadata.
- **Inputs**:
    - `value`: A string representing the type of projector to be added.
- **Control Flow**:
    - Calls the [`add_string`](#GGUFWriteradd_string) method of the parent class `GGUFWriter` to add the projector type.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_string`](#GGUFWriteradd_string)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_projection\_dim<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_projection_dim}} -->
Adds a projection dimension for vision models in the GGUFWriter.
- **Inputs**:
    - `value`: An integer representing the projection dimension to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a specific key and the provided value to store the projection dimension.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the projection dimension.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_patch\_size<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_patch_size}} -->
Adds a vision patch size to the GGUFWriter's key-value data.
- **Inputs**:
    - `value`: An integer representing the size of the vision patch to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with the key `Keys.ClipVision.PATCH_SIZE` and the provided `value`.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the patch size to its key-value data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_embedding\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_embedding_length}} -->
Adds the vision embedding length to the GGUFWriter's key-value data.
- **Inputs**:
    - `value`: An integer representing the length of the vision embedding.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to store the embedding length under a specific key.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the embedding length.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_feed\_forward\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_feed_forward_length}} -->
Adds a specified feed-forward length to the vision model's configuration.
- **Inputs**:
    - `value`: An integer representing the feed-forward length to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the parent class `GGUFWriter` to store the feed-forward length under a specific key.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the feed-forward length.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_block\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_block_count}} -->
Adds a vision block count to the GGUFWriter's key-value data.
- **Inputs**:
    - `value`: An integer representing the number of vision blocks to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a specific key and the provided value to store the block count.
- **Output**: This method does not return any value; it updates the internal state of the GGUFWriter by adding the block count.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_head\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_head_count}} -->
Adds a vision head count to the GGUFWriter's key-value data.
- **Inputs**:
    - `value`: An integer representing the number of heads in the vision model's attention mechanism.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to store the head count under a specific key.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_attention\_layernorm\_eps<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_attention_layernorm_eps}} -->
Adds a float32 value representing the layer normalization epsilon for vision attention.
- **Inputs**:
    - `value`: A float representing the layer normalization epsilon value to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the parent class `GGUFWriter`.
    - Uses a predefined key from `Keys.ClipVision.Attention.LAYERNORM_EPS` to store the value.
- **Output**: None, as the method modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_image\_size<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_image_size}} -->
Adds the image size for vision models to the GGUFWriter.
- **Inputs**:
    - `value`: An integer representing the image size to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to store the image size under the key `Keys.ClipVision.IMAGE_SIZE`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` by adding the image size.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_image\_mean<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_image_mean}} -->
Adds an array of float values representing the mean image values for vision processing.
- **Inputs**:
    - `values`: A sequence of float values representing the mean image values.
- **Control Flow**:
    - Calls the [`add_array`](#GGUFWriteradd_array) method of the parent class `GGUFWriter` to store the provided mean values under a specific key.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the mean image values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_image\_std<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_image_std}} -->
Adds an array of standard deviation values for vision images to the GGUFWriter.
- **Inputs**:
    - `values`: A sequence of float values representing the standard deviations for vision images.
- **Control Flow**:
    - Calls the [`add_array`](#GGUFWriteradd_array) method of the `GGUFWriter` class to store the provided standard deviation values under the key `Keys.ClipVision.IMAGE_STD`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the provided standard deviation values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_array`](#GGUFWriteradd_array)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_spatial\_merge\_size<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_spatial_merge_size}} -->
Adds a spatial merge size value to the GGUFWriter's key-value data.
- **Inputs**:
    - `value`: An integer representing the spatial merge size to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with a specific key and the provided value to store the spatial merge size.
- **Output**: This method does not return any value; it modifies the internal state of the GGUFWriter by adding the spatial merge size.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_use\_gelu<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_use_gelu}} -->
Sets the boolean value for using GELU in vision models.
- **Inputs**:
    - `value`: A boolean indicating whether to use GELU (Gaussian Error Linear Unit) activation function in the vision model.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method of the parent class `GGUFWriter` to store the value associated with the key `Keys.ClipVision.USE_GELU`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding a key-value pair.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_use\_silu<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_use_silu}} -->
Sets the boolean value for using the SiLU activation function in vision models.
- **Inputs**:
    - `value`: A boolean indicating whether to use the SiLU activation function.
- **Control Flow**:
    - Calls the [`add_bool`](#GGUFWriteradd_bool) method of the parent class `GGUFWriter`.
    - Passes the key `Keys.ClipVision.USE_SILU` along with the provided boolean value.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_bool`](#GGUFWriteradd_bool)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_projector\_scale\_factor<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_projector_scale_factor}} -->
Adds a scale factor for the vision projector in the GGUFWriter.
- **Inputs**:
    - `value`: An integer representing the scale factor to be added for the vision projector.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the parent class `GGUFWriter` to add the scale factor.
    - Uses a predefined key from `Keys.ClipVision.Projector.SCALE_FACTOR` to associate the value.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_vision\_n\_wa\_pattern<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_vision_n_wa_pattern}} -->
Adds a specified integer value to the `N_WA_PATTERN` key in the `ClipVision` section of the GGUFWriter.
- **Inputs**:
    - `value`: An integer value to be added to the `N_WA_PATTERN` key.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with the `N_WA_PATTERN` key and the provided value.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` by adding the specified value.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_audio\_projection\_dim<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_projection_dim}} -->
Adds an audio projection dimension value to the GGUFWriter instance.
- **Inputs**:
    - `value`: An integer representing the audio projection dimension to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to add the projection dimension value associated with the key `Keys.ClipAudio.PROJECTION_DIM`.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the specified projection dimension.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_audio\_embedding\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_embedding_length}} -->
Adds the audio embedding length to the GGUFWriter's key-value data.
- **Inputs**:
    - `value`: An integer representing the length of the audio embedding.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to store the audio embedding length.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_audio\_feed\_forward\_length<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_feed_forward_length}} -->
Adds the audio feed forward length to the GGUFWriter's key-value data.
- **Inputs**:
    - `value`: An integer representing the feed forward length to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to add the feed forward length.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_audio\_block\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_block_count}} -->
Adds an audio block count to the GGUFWriter instance.
- **Inputs**:
    - `value`: An integer representing the audio block count to be added.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to add the audio block count.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_audio\_head\_count<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_head_count}} -->
Adds an audio head count to the GGUFWriter's key-value data.
- **Inputs**:
    - `value`: An integer representing the head count for audio attention.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to add the head count to the key-value data.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_audio\_attention\_layernorm\_eps<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_attention_layernorm_eps}} -->
Adds a float32 value representing the layer normalization epsilon for audio attention.
- **Inputs**:
    - `value`: A float representing the layer normalization epsilon value to be added.
- **Control Flow**:
    - Calls the [`add_float32`](#GGUFWriteradd_float32) method of the parent class `GGUFWriter`.
    - Uses a predefined key from `Keys.ClipAudio.Attention.LAYERNORM_EPS` to store the value.
- **Output**: None, as the method modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_float32`](#GGUFWriteradd_float32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_audio\_num\_mel\_bins<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_num_mel_bins}} -->
Sets the number of Mel bins for audio processing in the GGUFWriter.
- **Inputs**:
    - `value`: An integer representing the number of Mel bins to be set.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method of the `GGUFWriter` class to store the number of Mel bins.
    - Uses a predefined key from `Keys.ClipAudio.NUM_MEL_BINS` to associate the value.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.add\_audio\_stack\_factor<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_audio_stack_factor}} -->
Sets the audio stack factor for the audio projector.
- **Inputs**:
    - `value`: An integer representing the stack factor to be set for the audio projector.
- **Control Flow**:
    - Calls the [`add_uint32`](#GGUFWriteradd_uint32) method with the key corresponding to the audio projector's stack factor and the provided value.
- **Output**: This method does not return any value; it modifies the internal state of the `GGUFWriter` instance by adding the specified stack factor.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.add_uint32`](#GGUFWriteradd_uint32)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.\_pack<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter._pack}} -->
The `_pack` method in the `GGUFWriter` class packs a given value into a binary format based on a specified format string and the endianess of the data.
- **Inputs**:
    - `fmt`: A format string that specifies the data type and layout for packing the value.
    - `value`: The value to be packed into binary format.
    - `skip_pack_prefix`: A boolean flag indicating whether to skip adding the endianess prefix to the packed data.
- **Control Flow**:
    - The method initializes an empty string for `pack_prefix`.
    - If `skip_pack_prefix` is False, it sets `pack_prefix` to '<' for little-endian or '>' for big-endian based on the `endianess` attribute of the class.
    - It then uses the `struct.pack` function to pack the `value` using the combined format string of `pack_prefix` and `fmt`.
- **Output**: The method returns the packed binary data as a bytes object.
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.\_pack\_val<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter._pack_val}} -->
The `_pack_val` method serializes various types of values into a byte format based on their specified type.
- **Inputs**:
    - `val`: The value to be serialized, which can be of any type.
    - `vtype`: An enumeration value indicating the type of the value being serialized (e.g., integer, string, array).
    - `add_vtype`: A boolean flag indicating whether to include the value type in the serialized output.
    - `sub_type`: An optional subtype for array values, indicating the type of elements within the array.
- **Control Flow**:
    - If `add_vtype` is true, the method first packs the value type into the output byte array.
    - The method checks if the value type has a predefined packing format; if so, it packs the value accordingly.
    - For string values, it encodes the string in UTF-8 and packs its length followed by the encoded string.
    - For array values, it validates the input, determines the element type, and packs the array length and each element recursively.
    - If the value type is invalid, a ValueError is raised.
- **Output**: The method returns a byte array containing the serialized representation of the input value.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter._pack`](#GGUFWriter_pack)
    - [`llama.cpp/gguf-py/gguf/constants.GGUFValueType.get_type`](constants.py.driver.md#GGUFValueTypeget_type)
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)


---
#### GGUFWriter\.format\_n\_bytes\_to\_str<!-- {{#callable:llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter.format_n_bytes_to_str}} -->
Converts a number of bytes into a human-readable string representation with appropriate units.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `num`: An integer representing the number of bytes to be formatted.
- **Control Flow**:
    - Checks if the input number is zero and returns a specific string if true.
    - Converts the number to a float for processing.
    - Iterates through a list of units (bytes, kilobytes, megabytes, gigabytes) to determine the appropriate unit for the number.
    - If the number is less than 1000, it formats and returns the number with the corresponding unit.
    - If the number exceeds the range of gigabytes, it divides the number by 1000 and continues until it finds the appropriate unit, returning a formatted string indicating that the size exceeds 1TB.
- **Output**: A string representing the formatted size in human-readable form, including the appropriate unit (e.g., '1.5M', '2.0G', or a message indicating the size is negligible or exceeds 1TB).
- **See also**: [`llama.cpp/gguf-py/gguf/gguf_writer.GGUFWriter`](#cpp/gguf-py/gguf/gguf_writerGGUFWriter)  (Base Class)



