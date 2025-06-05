# Purpose
The provided C++ source code defines a class named `llama_context`, which serves as a central component for managing and executing operations related to a machine learning model, likely within a broader framework or library. This class encapsulates various functionalities, including model initialization, synchronization, and execution of computational graphs. It interfaces with several other components, such as `llama_model`, `llama_batch`, and `llama_adapter`, indicating its role in handling model parameters, batch processing, and adapter configurations. The class also provides methods for managing threading, memory, and performance metrics, which are crucial for optimizing the execution of machine learning tasks.

The `llama_context` class is designed to be highly modular and flexible, supporting operations like encoding, decoding, and processing of micro-batches (`ubatch`). It includes methods for saving and loading state data, which is essential for model persistence and checkpointing. The class also supports training operations, with methods for initializing and executing optimization epochs. Additionally, it provides interfaces for attaching and detaching thread pools, setting various operational parameters, and handling performance data. The presence of private and public sections within the class indicates a clear separation of internal mechanisms and public APIs, ensuring encapsulation and ease of use for developers integrating this class into their applications.
# Imports and Dependencies

---
- `llama.h`
- `llama-batch.h`
- `llama-cparams.h`
- `llama-graph.h`
- `llama-adapter.h`
- `ggml-cpp.h`
- `ggml-opt.h`
- `map`
- `vector`


# Data Structures

---
### llama\_context<!-- {{#data_structure:llama_context}} -->
- **Type**: `struct`
- **Members**:
    - `model`: A reference to the llama_model associated with this context.
    - `cparams`: Holds the context parameters for the llama_context.
    - `cvec`: Represents the adapter vector for the context.
    - `loras`: Stores the adapter loras for the context.
    - `cross`: Handles cross-attention, though it is marked as temporary.
    - `memory`: A unique pointer to the llama_memory_i, managing memory operations.
    - `memory_force_optimize`: A boolean flag indicating if memory optimization is forced.
    - `logits_size`: The capacity for logits in terms of floats.
    - `logits`: A pointer to the logits output buffer.
    - `embd_size`: The capacity for embeddings in terms of floats.
    - `embd`: A pointer to the embeddings output buffer.
    - `embd_seq`: A map storing sequence embeddings as vectors of floats.
    - `n_outputs`: The number of outputs used in the current or last batch.
    - `n_outputs_max`: The maximum capacity for output buffers in terms of token positions.
    - `output_ids`: Maps batch token positions to IDs of the logits and embedding buffers.
    - `sched`: A pointer to the backend scheduler for managing computation.
    - `backend_cpu`: A pointer to the CPU backend used for computation.
    - `backends`: A vector of pointers to various backends used for computation.
    - `ctx_compute`: A pointer to the context used for computation graphs.
    - `opt_ctx`: A pointer to the optimization context for training.
    - `threadpool`: A pointer to the threadpool used for parallel computation.
    - `threadpool_batch`: A pointer to the threadpool used for batch parallel computation.
    - `abort_callback`: A function pointer for handling abort callbacks.
    - `abort_callback_data`: A pointer to data used by the abort callback function.
    - `set_n_threads_fns`: A vector of pairs mapping backends to their set_n_threads functions.
    - `backend_ptrs`: A vector of backend pointers used for compute buffers.
    - `backend_buft`: A vector of buffer types for each backend's compute buffer.
    - `buf_compute_meta`: A vector of bytes used for storing computation metadata.
    - `buf_output`: A pointer to the buffer used for model output (logits and embeddings).
    - `has_evaluated_once`: A boolean indicating if the context has been evaluated at least once.
    - `t_start_us`: The start time in microseconds for performance measurement.
    - `t_load_us`: The load time in microseconds for performance measurement.
    - `t_p_eval_us`: The prompt evaluation time in microseconds for performance measurement.
    - `t_eval_us`: The evaluation time in microseconds for performance measurement.
    - `t_compute_start_us`: The start time in microseconds for compute operations.
    - `n_queued_tokens`: The number of tokens queued for evaluation.
    - `n_p_eval`: The number of tokens in evaluation calls for the prompt.
    - `n_eval`: The number of evaluation calls made.
- **Description**: The `llama_context` struct is a comprehensive data structure designed to manage the context and state of a llama model during its operation. It encapsulates various parameters and settings required for model execution, including memory management, threading, and backend scheduling. The struct maintains references to the model, context parameters, and various buffers for logits and embeddings. It also handles the initialization and synchronization of computation graphs, manages output buffers, and supports state saving and loading. Additionally, it provides functionality for training, including optimization context management and epoch iteration. The `llama_context` is integral to the efficient execution and evaluation of the llama model, ensuring that all necessary resources and configurations are properly managed and utilized.
- **Member Functions**:
    - [`llama_context::llama_context`](llama-context.cpp.driver.md#llama_contextllama_context)
    - [`llama_context::~llama_context`](llama-context.cpp.driver.md#llama_contextllama_context)
    - [`llama_context::synchronize`](llama-context.cpp.driver.md#llama_contextsynchronize)
    - [`llama_context::get_model`](llama-context.cpp.driver.md#llama_contextget_model)
    - [`llama_context::get_cparams`](llama-context.cpp.driver.md#llama_contextget_cparams)
    - [`llama_context::get_sched`](llama-context.cpp.driver.md#llama_contextget_sched)
    - [`llama_context::get_ctx_compute`](llama-context.cpp.driver.md#llama_contextget_ctx_compute)
    - [`llama_context::n_ctx`](llama-context.cpp.driver.md#llama_contextn_ctx)
    - [`llama_context::n_ctx_per_seq`](llama-context.cpp.driver.md#llama_contextn_ctx_per_seq)
    - [`llama_context::n_batch`](llama-context.cpp.driver.md#llama_contextn_batch)
    - [`llama_context::n_ubatch`](llama-context.cpp.driver.md#llama_contextn_ubatch)
    - [`llama_context::n_seq_max`](llama-context.cpp.driver.md#llama_contextn_seq_max)
    - [`llama_context::n_threads`](llama-context.cpp.driver.md#llama_contextn_threads)
    - [`llama_context::n_threads_batch`](llama-context.cpp.driver.md#llama_contextn_threads_batch)
    - [`llama_context::get_memory`](llama-context.cpp.driver.md#llama_contextget_memory)
    - [`llama_context::kv_self_defrag_sched`](llama-context.cpp.driver.md#llama_contextkv_self_defrag_sched)
    - [`llama_context::kv_self_update`](llama-context.cpp.driver.md#llama_contextkv_self_update)
    - [`llama_context::pooling_type`](llama-context.cpp.driver.md#llama_contextpooling_type)
    - [`llama_context::get_logits`](llama-context.cpp.driver.md#llama_contextget_logits)
    - [`llama_context::get_logits_ith`](llama-context.cpp.driver.md#llama_contextget_logits_ith)
    - [`llama_context::get_embeddings`](llama-context.cpp.driver.md#llama_contextget_embeddings)
    - [`llama_context::get_embeddings_ith`](llama-context.cpp.driver.md#llama_contextget_embeddings_ith)
    - [`llama_context::get_embeddings_seq`](llama-context.cpp.driver.md#llama_contextget_embeddings_seq)
    - [`llama_context::attach_threadpool`](llama-context.cpp.driver.md#llama_contextattach_threadpool)
    - [`llama_context::detach_threadpool`](llama-context.cpp.driver.md#llama_contextdetach_threadpool)
    - [`llama_context::set_n_threads`](llama-context.cpp.driver.md#llama_contextset_n_threads)
    - [`llama_context::set_abort_callback`](llama-context.cpp.driver.md#llama_contextset_abort_callback)
    - [`llama_context::set_embeddings`](llama-context.cpp.driver.md#llama_contextset_embeddings)
    - [`llama_context::set_causal_attn`](llama-context.cpp.driver.md#llama_contextset_causal_attn)
    - [`llama_context::set_warmup`](llama-context.cpp.driver.md#llama_contextset_warmup)
    - [`llama_context::set_adapter_lora`](llama-context.cpp.driver.md#llama_contextset_adapter_lora)
    - [`llama_context::rm_adapter_lora`](llama-context.cpp.driver.md#llama_contextrm_adapter_lora)
    - [`llama_context::clear_adapter_lora`](llama-context.cpp.driver.md#llama_contextclear_adapter_lora)
    - [`llama_context::apply_adapter_cvec`](llama-context.cpp.driver.md#llama_contextapply_adapter_cvec)
    - [`llama_context::process_ubatch`](llama-context.cpp.driver.md#llama_contextprocess_ubatch)
    - [`llama_context::encode`](llama-context.cpp.driver.md#llama_contextencode)
    - [`llama_context::decode`](llama-context.cpp.driver.md#llama_contextdecode)
    - [`llama_context::output_reserve`](llama-context.cpp.driver.md#llama_contextoutput_reserve)
    - [`llama_context::graph_max_nodes`](llama-context.cpp.driver.md#llama_contextgraph_max_nodes)
    - [`llama_context::graph_init`](llama-context.cpp.driver.md#llama_contextgraph_init)
    - [`llama_context::graph_reserve`](llama-context.cpp.driver.md#llama_contextgraph_reserve)
    - [`llama_context::graph_build`](llama-context.cpp.driver.md#llama_contextgraph_build)
    - [`llama_context::graph_compute`](llama-context.cpp.driver.md#llama_contextgraph_compute)
    - [`llama_context::graph_get_cb`](llama-context.cpp.driver.md#llama_contextgraph_get_cb)
    - [`llama_context::state_get_size`](llama-context.cpp.driver.md#llama_contextstate_get_size)
    - [`llama_context::state_get_data`](llama-context.cpp.driver.md#llama_contextstate_get_data)
    - [`llama_context::state_set_data`](llama-context.cpp.driver.md#llama_contextstate_set_data)
    - [`llama_context::state_seq_get_size`](llama-context.cpp.driver.md#llama_contextstate_seq_get_size)
    - [`llama_context::state_seq_get_data`](llama-context.cpp.driver.md#llama_contextstate_seq_get_data)
    - [`llama_context::state_seq_set_data`](llama-context.cpp.driver.md#llama_contextstate_seq_set_data)
    - [`llama_context::state_load_file`](llama-context.cpp.driver.md#llama_contextstate_load_file)
    - [`llama_context::state_save_file`](llama-context.cpp.driver.md#llama_contextstate_save_file)
    - [`llama_context::state_seq_load_file`](llama-context.cpp.driver.md#llama_contextstate_seq_load_file)
    - [`llama_context::state_seq_save_file`](llama-context.cpp.driver.md#llama_contextstate_seq_save_file)
    - [`llama_context::state_write_data`](llama-context.cpp.driver.md#llama_contextstate_write_data)
    - [`llama_context::state_read_data`](llama-context.cpp.driver.md#llama_contextstate_read_data)
    - [`llama_context::state_seq_write_data`](llama-context.cpp.driver.md#llama_contextstate_seq_write_data)
    - [`llama_context::state_seq_read_data`](llama-context.cpp.driver.md#llama_contextstate_seq_read_data)
    - [`llama_context::perf_get_data`](llama-context.cpp.driver.md#llama_contextperf_get_data)
    - [`llama_context::perf_reset`](llama-context.cpp.driver.md#llama_contextperf_reset)
    - [`llama_context::opt_init`](llama-context.cpp.driver.md#llama_contextopt_init)
    - [`llama_context::opt_epoch_iter`](llama-context.cpp.driver.md#llama_contextopt_epoch_iter)
    - [`llama_context::opt_epoch`](llama-context.cpp.driver.md#llama_contextopt_epoch)


