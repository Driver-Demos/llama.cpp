# Purpose
The provided C++ code is a sophisticated implementation of a Vulkan-based backend for a machine learning library, likely intended for high-performance computations on GPUs, such as those used in the GGML framework. It is designed to leverage Vulkan's low-level graphics and compute API to perform a wide range of tensor operations, including matrix multiplications, element-wise operations, and more complex neural network tasks like softmax and convolution, thereby accelerating machine learning computations. The code is modular and extensible, with components for managing Vulkan devices, buffers, pipelines, and synchronization, as well as internal APIs for resource management and compute execution. It includes performance and debugging tools to ensure computational correctness and optimize GPU operations, making it a flexible and powerful backend library for integrating into larger machine learning systems. Overall, this code serves as a foundation for building high-performance applications by offloading tensor computations to Vulkan-supported GPUs, enhancing both functionality and performance.
# Imports and Dependencies

---
- `ggml-vulkan.h`
- `vulkan/vulkan_core.h`
- `chrono`
- `ggml-cpu.h`
- `vulkan/vulkan.hpp`
- `algorithm`
- `cmath`
- `iomanip`
- `iostream`
- `tuple`
- `vector`
- `sstream`
- `utility`
- `memory`
- `limits`
- `map`
- `unordered_map`
- `mutex`
- `future`
- `thread`
- `windows.h`
- `immintrin.h`
- `arm_acle.h`
- `ggml-impl.h`
- `ggml-backend-impl.h`
- `ggml-vulkan-shaders.hpp`


# Global Variables

---
### ggml\_backend\_vk\_buffer\_type\_name
- **Type**: `function`
- **Description**: `ggml_backend_vk_buffer_type_name` is a static function that takes a parameter of type `ggml_backend_buffer_type_t` and returns a constant character pointer. This function is likely used to retrieve a string representation of a buffer type for Vulkan backend operations.
- **Use**: This function is used to obtain the name of a Vulkan backend buffer type as a string.


---
### ggml\_backend\_vk\_buffer\_type\_interface
- **Type**: `ggml_backend_buffer_type_i`
- **Description**: The `ggml_backend_vk_buffer_type_interface` is a static instance of the `ggml_backend_buffer_type_i` structure, which is used to define a set of function pointers for managing Vulkan buffer types in the GGML backend. This structure includes functions for retrieving the buffer's name, allocating the buffer, getting alignment, determining the maximum size, and calculating the allocation size. The `is_host` function pointer is set to NULL, indicating that this functionality is not implemented or not applicable for this interface.
- **Use**: This variable is used to provide a concrete implementation of the `ggml_backend_buffer_type_i` interface for Vulkan buffer management in the GGML backend.


---
### mul\_mat\_vec\_max\_cols
- **Type**: ``uint32_t``
- **Description**: The variable `mul_mat_vec_max_cols` is a constant of type `uint32_t` that represents the maximum number of columns allowed in a matrix-vector multiplication operation. It is defined with a value of 8, indicating that the operation is optimized or constrained to handle matrices with up to 8 columns.
- **Use**: This variable is used to set a limit on the number of columns in matrix-vector multiplication operations, likely for performance optimization or hardware compatibility.


---
### p021\_max\_gqa\_ratio
- **Type**: ``uint32_t``
- **Description**: The variable `p021_max_gqa_ratio` is a global constant of type `uint32_t` with a value of 8. It is defined using the `constexpr` keyword, indicating that its value is a compile-time constant.
- **Use**: This variable is used to represent the maximum GQA (Global Quality Assessment) ratio allowed in the program.


---
### log\_mutex
- **Type**: `std::mutex`
- **Description**: The `log_mutex` is a static mutex object used to synchronize access to shared resources, specifically for logging purposes. It ensures that only one thread can access the logging mechanism at a time, preventing race conditions and ensuring thread-safe operations.
- **Use**: This variable is used to lock and unlock access to the logging system, ensuring that log entries are not interleaved or corrupted by concurrent thread execution.


---
### vk\_ptr\_base
- **Type**: `void * const`
- **Description**: The variable `vk_ptr_base` is a constant pointer to a void type, initialized to the memory address 0x1000. This means it is a pointer that cannot be changed to point to another address after its initialization.
- **Use**: This variable is used as a base address for memory operations or calculations, likely serving as a reference point in memory management or hardware interfacing.


---
### vk\_instance\_initialized
- **Type**: `bool`
- **Description**: The `vk_instance_initialized` is a static boolean variable that indicates whether the Vulkan instance has been successfully initialized. It is initially set to `false`, suggesting that the Vulkan instance is not yet initialized at the start of the program.
- **Use**: This variable is used to track the initialization state of the Vulkan instance, ensuring that initialization procedures are only executed once.


---
### vk\_instance
- **Type**: `vk_instance_t`
- **Description**: The `vk_instance` is a static global variable of type `vk_instance_t`. It is likely used to represent an instance of a Vulkan API context, which is essential for managing the connection between the application and the Vulkan library.
- **Use**: This variable is used to maintain a single instance of the Vulkan context throughout the application's lifecycle.


---
### vk\_perf\_logger\_enabled
- **Type**: `bool`
- **Description**: The `vk_perf_logger_enabled` is a static global boolean variable that indicates whether the Vulkan performance logger is enabled or not. It is initialized to `false`, meaning that by default, the performance logging is disabled.
- **Use**: This variable is used to control the activation of performance logging features in a Vulkan-based application.


---
### vk\_skip\_checks
- **Type**: `size_t`
- **Description**: The variable `vk_skip_checks` is a static global variable of type `size_t`. It is used to store a size-related value, typically representing a count or a size in bytes, depending on the context of its use.
- **Use**: This variable is used to control or track the number of checks that should be skipped in a given operation or process.


---
### vk\_output\_tensor
- **Type**: `size_t`
- **Description**: The `vk_output_tensor` is a static global variable of type `size_t`, which is an unsigned integer type used to represent the size of any object in bytes. It is declared at the top level scope, making it accessible throughout the file in which it is defined.
- **Use**: This variable is used to store the size or index related to an output tensor in a Vulkan-based application.


---
### compile\_count
- **Type**: `uint32_t`
- **Description**: The `compile_count` variable is a static unsigned 32-bit integer initialized to zero. It is used to keep track of the number of times a certain compilation process has been executed.
- **Use**: This variable is incremented in a thread-safe manner using the associated mutex to ensure accurate counting across multiple threads.


---
### compile\_count\_mutex
- **Type**: `std::mutex`
- **Description**: The `compile_count_mutex` is a static mutex object used to synchronize access to shared resources or critical sections of code related to compile count operations. It ensures that only one thread can access the protected resource at a time, preventing race conditions.
- **Use**: This mutex is used to lock and protect shared data or operations related to compile counts, ensuring thread-safe access.


---
### compile\_count\_cond
- **Type**: `std::condition_variable`
- **Description**: The `compile_count_cond` is a static instance of `std::condition_variable`, which is a synchronization primitive used to block a thread until a particular condition is met. It is typically used in conjunction with a mutex to manage access to shared data in concurrent programming.
- **Use**: This variable is used to synchronize threads, allowing them to wait for a condition to be met before proceeding.


---
### flash\_attention\_num\_small\_rows
- **Type**: `uint32_t`
- **Description**: The variable `flash_attention_num_small_rows` is a constant of type `uint32_t` that holds the value 32. It is defined as a static constexpr, indicating that it is a compile-time constant and its value cannot be changed during runtime.
- **Use**: This variable is used to define the number of small rows in a flash attention mechanism, likely to optimize or configure the attention computation process.


---
### scalar\_flash\_attention\_num\_small\_rows
- **Type**: `uint32_t`
- **Description**: The variable `scalar_flash_attention_num_small_rows` is a constant of type `uint32_t` that holds the value 1. It is defined as a static constexpr, indicating that it is a compile-time constant with internal linkage.
- **Use**: This variable is used to specify the number of small rows in a scalar flash attention mechanism.


---
### scalar\_flash\_attention\_num\_large\_rows
- **Type**: ``uint32_t``
- **Description**: The variable `scalar_flash_attention_num_large_rows` is a constant of type `uint32_t` that holds the value 8. It is defined as a static constexpr, indicating that it is a compile-time constant and its value cannot be changed during runtime.
- **Use**: This variable is used to define the number of large rows in a scalar flash attention mechanism, likely influencing the configuration or behavior of the attention computation.


---
### coopmat1\_flash\_attention\_num\_large\_rows
- **Type**: ``uint32_t``
- **Description**: The variable `coopmat1_flash_attention_num_large_rows` is a constant of type `uint32_t` that holds the value 16. It is defined as a static constexpr, indicating that its value is a compile-time constant and cannot be changed during runtime.
- **Use**: This variable is used to specify the number of large rows in a cooperative matrix operation related to flash attention.


---
### scalar\_flash\_attention\_Bc
- **Type**: `uint32_t`
- **Description**: The variable `scalar_flash_attention_Bc` is a constant unsigned 32-bit integer with a value of 64. It is defined using the `constexpr` keyword, indicating that its value is known at compile time and cannot be changed during program execution.
- **Use**: This variable is used to define a constant value, likely representing a block size or a similar parameter, in the context of scalar flash attention operations.


---
### scalar\_flash\_attention\_workgroup\_size
- **Type**: ``uint32_t``
- **Description**: The variable `scalar_flash_attention_workgroup_size` is a constant unsigned 32-bit integer that defines the size of the workgroup for scalar flash attention operations. It is set to a value of 128, indicating the number of threads or units of work that will be grouped together for processing.
- **Use**: This variable is used to configure the workgroup size for scalar flash attention operations, optimizing parallel processing.


---
### rdna1\_pipelines
- **Type**: `std::unordered_map<std::string, uint32_t>`
- **Description**: The `rdna1_pipelines` is a static constant unordered map that associates string keys with unsigned 32-bit integer values. It is used to store pipeline names as keys and their corresponding numerical identifiers or capacities as values.
- **Use**: This variable is used to map specific pipeline names to their respective numeric identifiers or capacities for RDNA1 architecture.


---
### rdna2\_pipelines
- **Type**: `std::unordered_map<std::string, uint32_t>`
- **Description**: The `rdna2_pipelines` is a static constant unordered map that associates string keys with unsigned 32-bit integer values. It is initialized with two key-value pairs: "soft_max" mapped to 64 and "im2col" mapped to 64.
- **Use**: This variable is used to store and provide quick access to specific pipeline configurations or identifiers associated with the RDNA2 architecture.


---
### RDNA\_DEFAULT\_SUBGROUP\_SIZE
- **Type**: ``uint32_t``
- **Description**: `RDNA_DEFAULT_SUBGROUP_SIZE` is a global constant variable of type `uint32_t` that represents the default size of a subgroup in RDNA architecture, set to 32. This value is typically used in graphics programming and GPU computations to define the number of threads in a subgroup for parallel processing.
- **Use**: This variable is used to specify the default number of threads in a subgroup for RDNA architecture in GPU programming.


---
### gpu\_pipeline\_configs
- **Type**: `std::vector<GpuPipelineConfig>`
- **Description**: The `gpu_pipeline_configs` is a static global variable that holds a vector of `GpuPipelineConfig` objects. Each `GpuPipelineConfig` object contains information about a specific GPU architecture, associated pipeline configurations, and a default subgroup size. This setup is used to manage and configure GPU pipelines for different AMD architectures, specifically RDNA1 and RDNA2.
- **Use**: This variable is used to store and manage GPU pipeline configurations for different AMD architectures, facilitating the selection and application of appropriate pipeline settings.


# Data Structures

---
### VkPhysicalDeviceShaderBfloat16FeaturesKHR<!-- {{#data_structure:VkPhysicalDeviceShaderBfloat16FeaturesKHR}} -->
- **Type**: `struct`
- **Members**:
    - `sType`: Specifies the type of this structure.
    - `pNext`: A pointer to the next structure in a structure chain.
    - `shaderBFloat16Type`: Indicates whether the bfloat16 shader type is supported.
    - `shaderBFloat16DotProduct`: Indicates support for bfloat16 dot product operations.
    - `shaderBFloat16CooperativeMatrix`: Indicates support for bfloat16 cooperative matrix operations.
- **Description**: The `VkPhysicalDeviceShaderBfloat16FeaturesKHR` structure is used in Vulkan to query and enable support for bfloat16 shader features on a physical device. It includes fields to specify the structure type, a pointer for extension chaining, and boolean flags to indicate support for bfloat16 shader types, dot product operations, and cooperative matrix operations. This structure is part of the Vulkan API's mechanism for feature querying and extension.


---
### vk\_queue<!-- {{#data_structure:vk_queue}} -->
- **Type**: `struct`
- **Members**:
    - `queue_family_index`: Stores the index of the queue family to which this queue belongs.
    - `queue`: Represents a Vulkan queue handle.
    - `pool`: Holds a Vulkan command pool associated with the queue.
    - `cmd_buffer_idx`: Tracks the current index of the command buffer in use.
    - `cmd_buffers`: Contains a list of Vulkan command buffers.
    - `stage_flags`: Specifies pipeline stage flags for synchronization purposes.
    - `transfer_only`: Indicates if the queue is used exclusively for transfer operations.
- **Description**: The `vk_queue` struct is a data structure used in Vulkan applications to encapsulate a Vulkan queue along with its associated resources and state. It includes a queue family index to identify the queue family, a Vulkan queue handle, and a command pool for managing command buffers. The struct also maintains a list of command buffers and an index to track the current command buffer in use. Additionally, it holds pipeline stage flags for synchronization and a boolean flag to indicate if the queue is dedicated to transfer operations only.


---
### vk\_pipeline\_struct<!-- {{#data_structure:vk_pipeline_struct}} -->
- **Type**: `struct`
- **Members**:
    - `name`: A string representing the name of the pipeline.
    - `shader_module`: A Vulkan ShaderModule object associated with the pipeline.
    - `dsl`: A Vulkan DescriptorSetLayout object defining the layout of descriptor sets.
    - `descriptor_pools`: A vector of Vulkan DescriptorPool objects used for allocating descriptor sets.
    - `descriptor_sets`: A vector of Vulkan DescriptorSet objects representing the descriptor sets used by the pipeline.
    - `descriptor_set_idx`: An unsigned integer indicating the index of the descriptor set.
    - `layout`: A Vulkan PipelineLayout object that defines the layout of the pipeline.
    - `pipeline`: A Vulkan Pipeline object representing the compiled pipeline.
    - `push_constant_size`: An unsigned integer specifying the size of the push constants.
    - `parameter_count`: An unsigned integer representing the number of parameters.
    - `wg_denoms`: An array of three unsigned integers used for workgroup size denominators.
    - `align`: An unsigned integer specifying alignment requirements.
    - `needed`: A boolean flag indicating if the pipeline needs to be compiled after a dry run.
    - `compiled`: A boolean flag indicating if the shader has been compiled.
- **Description**: The `vk_pipeline_struct` is a comprehensive data structure used in Vulkan graphics programming to encapsulate all necessary components and configurations for a graphics pipeline. It includes shader modules, descriptor set layouts, descriptor pools, and sets, as well as pipeline layout and the pipeline itself. Additionally, it manages various configuration parameters such as push constant size, parameter count, and workgroup denominators. The structure also contains flags to track the compilation state of the pipeline, ensuring that it is compiled when needed and indicating when the shader has been successfully compiled.


---
### vk\_matmul\_pipeline\_struct<!-- {{#data_structure:vk_matmul_pipeline_struct}} -->
- **Type**: `struct`
- **Members**:
    - `l`: A vk_pipeline object representing the 'l' pipeline.
    - `m`: A vk_pipeline object representing the 'm' pipeline.
    - `s`: A vk_pipeline object representing the 's' pipeline.
    - `a_l`: A vk_pipeline object representing the 'a_l' pipeline.
    - `a_m`: A vk_pipeline object representing the 'a_m' pipeline.
    - `a_s`: A vk_pipeline object representing the 'a_s' pipeline.
- **Description**: The `vk_matmul_pipeline_struct` is a structure that encapsulates six `vk_pipeline` objects, which are likely used to manage different stages or configurations of a matrix multiplication operation in a Vulkan-based application. The structure provides a convenient way to group these related pipeline objects, potentially representing different aspects or phases of the matrix multiplication process, such as loading, multiplying, and storing results.


---
### vk\_matmul\_pipeline2<!-- {{#data_structure:vk_matmul_pipeline2}} -->
- **Type**: `struct`
- **Members**:
    - `f32acc`: A shared pointer to a vk_matmul_pipeline_struct for 32-bit floating-point accumulation.
    - `f16acc`: A shared pointer to a vk_matmul_pipeline_struct for 16-bit floating-point accumulation.
- **Description**: The `vk_matmul_pipeline2` struct is designed to manage two separate matrix multiplication pipelines, one for 32-bit floating-point (f32acc) and another for 16-bit floating-point (f16acc) accumulations. It initializes these pipelines using shared pointers to `vk_matmul_pipeline_struct`, ensuring efficient memory management and resource sharing. This struct is likely used in contexts where different precision levels are required for matrix operations, providing flexibility and performance optimization in computational tasks.
- **Member Functions**:
    - [`vk_matmul_pipeline2::vk_matmul_pipeline2`](#vk_matmul_pipeline2vk_matmul_pipeline2)

**Methods**

---
#### vk\_matmul\_pipeline2::vk\_matmul\_pipeline2<!-- {{#callable:vk_matmul_pipeline2::vk_matmul_pipeline2}} -->
The `vk_matmul_pipeline2` constructor initializes two shared pointers to `vk_matmul_pipeline_struct` for 16-bit and 32-bit accumulators.
- **Inputs**: None
- **Control Flow**:
    - The constructor `vk_matmul_pipeline2` is called when an instance of the `vk_matmul_pipeline2` struct is created.
    - Inside the constructor, `f16acc` is initialized as a shared pointer to a new instance of `vk_matmul_pipeline_struct`.
    - Similarly, `f32acc` is initialized as a shared pointer to another new instance of `vk_matmul_pipeline_struct`.
- **Output**: The constructor does not return a value, but initializes two member variables `f16acc` and `f32acc` as shared pointers to `vk_matmul_pipeline_struct`.
- **See also**: [`vk_matmul_pipeline2`](#vk_matmul_pipeline2)  (Data Structure)



---
### ggml\_backend\_vk\_buffer\_type\_context<!-- {{#data_structure:ggml_backend_vk_buffer_type_context}} -->
- **Type**: `struct`
- **Members**:
    - `name`: A string representing the name of the Vulkan buffer type context.
    - `device`: An instance of vk_device associated with the Vulkan buffer type context.
- **Description**: The `ggml_backend_vk_buffer_type_context` struct is a data structure used to represent a context for Vulkan buffer types within the GGML backend. It contains a name, which is a string identifier for the context, and a `vk_device`, which is an instance representing the Vulkan device associated with this context. This struct is likely used to manage and organize Vulkan resources in a graphics or compute application.


---
### vk\_device\_architecture<!-- {{#data_structure:vk_device_architecture}} -->
- **Type**: `enum`
- **Members**:
    - `OTHER`: Represents an unspecified or unknown device architecture.
    - `AMD_GCN`: Represents the AMD Graphics Core Next architecture.
    - `AMD_RDNA1`: Represents the first generation of AMD's RDNA architecture.
    - `AMD_RDNA2`: Represents the second generation of AMD's RDNA architecture.
    - `AMD_RDNA3`: Represents the third generation of AMD's RDNA architecture.
    - `INTEL_XE2`: Represents the Intel Xe2 architecture.
- **Description**: The `vk_device_architecture` enum defines a set of constants representing different types of device architectures, primarily focusing on AMD and Intel architectures. It is used to categorize and identify the architecture of a device, which can be useful for optimizing performance or compatibility in graphics applications.


---
### vk\_device\_struct<!-- {{#data_structure:vk_device_struct}} -->
- **Type**: `struct`
- **Members**:
    - `mutex`: A mutex for synchronizing access to the device structure.
    - `physical_device`: Represents the physical device associated with this structure.
    - `properties`: Holds the properties of the physical device.
    - `name`: The name of the device.
    - `max_memory_allocation_size`: The maximum size for memory allocations.
    - `suballocation_block_size`: The block size for suballocations.
    - `fp16`: Indicates if FP16 is supported.
    - `pipeline_robustness`: Indicates if pipeline robustness is supported.
    - `device`: Represents the logical device.
    - `vendor_id`: The vendor ID of the device.
    - `driver_id`: The driver ID of the device.
    - `architecture`: The architecture of the device.
    - `compute_queue`: The compute queue associated with the device.
    - `transfer_queue`: The transfer queue associated with the device.
    - `single_queue`: Indicates if a single queue is used.
    - `subgroup_size`: The size of the subgroup.
    - `shader_core_count`: The number of shader cores.
    - `uma`: Indicates if unified memory architecture is supported.
    - `prefer_host_memory`: Indicates if host memory is preferred.
    - `float_controls_rte_fp16`: Indicates if float controls for RTE FP16 are supported.
    - `subgroup_add`: Indicates if subgroup addition is supported.
    - `subgroup_shuffle`: Indicates if subgroup shuffle is supported.
    - `integer_dot_product`: Indicates if integer dot product is supported.
    - `subgroup_size_control`: Indicates if subgroup size control is supported.
    - `subgroup_min_size`: The minimum size of the subgroup.
    - `subgroup_max_size`: The maximum size of the subgroup.
    - `subgroup_require_full_support`: Indicates if full support for subgroup is required.
    - `coopmat_support`: Indicates if cooperative matrix support is available.
    - `coopmat_acc_f32_support`: Indicates if cooperative matrix F32 accumulation is supported.
    - `coopmat_acc_f16_support`: Indicates if cooperative matrix F16 accumulation is supported.
    - `coopmat_bf16_support`: Indicates if cooperative matrix BF16 support is available.
    - `coopmat_support_16x16x16_f16acc`: Indicates if 16x16x16 cooperative matrix with F16 accumulation is supported.
    - `coopmat_support_16x16x16_f32acc`: Indicates if 16x16x16 cooperative matrix with F32 accumulation is supported.
    - `coopmat1_fa_support`: Indicates if cooperative matrix 1 FA support is available.
    - `coopmat_m`: The M dimension for cooperative matrix.
    - `coopmat_n`: The N dimension for cooperative matrix.
    - `coopmat_k`: The K dimension for cooperative matrix.
    - `coopmat_int_support`: Indicates if integer cooperative matrix support is available.
    - `coopmat_int_m`: The M dimension for integer cooperative matrix.
    - `coopmat_int_n`: The N dimension for integer cooperative matrix.
    - `coopmat_int_k`: The K dimension for integer cooperative matrix.
    - `coopmat2`: Indicates if a second cooperative matrix is supported.
    - `idx`: An index for the device structure.
    - `mul_mat_l`: Array indicating support for large matrix multiplication for each type.
    - `mul_mat_m`: Array indicating support for medium matrix multiplication for each type.
    - `mul_mat_s`: Array indicating support for small matrix multiplication for each type.
    - `mul_mat_id_l`: Array indicating support for large matrix multiplication with identity for each type.
    - `mul_mat_id_m`: Array indicating support for medium matrix multiplication with identity for each type.
    - `mul_mat_id_s`: Array indicating support for small matrix multiplication with identity for each type.
    - `need_compiles`: Indicates if shaders need to be compiled after a dry run.
    - `pipeline_matmul_f32`: Pipeline for F32 matrix multiplication.
    - `pipeline_matmul_f32_f16`: Pipeline for F32 to F16 matrix multiplication.
    - `pipeline_matmul_bf16`: Pipeline for BF16 matrix multiplication.
    - `pipeline_matmul_f16`: Pipeline for F16 matrix multiplication.
    - `pipeline_matmul_f16_f32`: Pipeline for F16 to F32 matrix multiplication.
    - `pipeline_dequant_mul_mat_mat`: Pipelines for dequantizing and multiplying matrices for each type.
    - `pipeline_dequant_mul_mat_mat_f16`: Pipelines for dequantizing and multiplying matrices with F16 for each type.
    - `pipeline_dequant_mul_mat_mat_q8_1`: Pipelines for dequantizing and multiplying matrices with Q8.1 for each type.
    - `pipeline_matmul_id_f32`: Pipeline for F32 matrix multiplication with identity.
    - `pipeline_matmul_id_bf16`: Pipeline for BF16 matrix multiplication with identity.
    - `pipeline_matmul_id_f16`: Pipeline for F16 matrix multiplication with identity.
    - `pipeline_matmul_id_f16_f32`: Pipeline for F16 to F32 matrix multiplication with identity.
    - `pipeline_dequant_mul_mat_mat_id`: Pipelines for dequantizing and multiplying matrices with identity for each type.
    - `pipeline_matmul_split_k_reduce`: Pipeline for split-K reduction in matrix multiplication.
    - `pipeline_quantize_q8_1`: Pipeline for quantizing to Q8.1.
    - `pipeline_dequant`: Pipelines for dequantizing for each type.
    - `pipeline_dequant_mul_mat_vec_f32_f32`: Pipelines for dequantizing and multiplying matrix-vector with F32 to F32 for each type.
    - `pipeline_dequant_mul_mat_vec_f16_f32`: Pipelines for dequantizing and multiplying matrix-vector with F16 to F32 for each type.
    - `pipeline_dequant_mul_mat_vec_id_f32`: Pipelines for dequantizing and multiplying matrix-vector with identity for F32 for each type.
    - `pipeline_mul_mat_vec_p021_f16_f32`: Pipelines for multiplying matrix-vector with P021 layout from F16 to F32.
    - `pipeline_mul_mat_vec_nc_f16_f32`: Pipeline for multiplying matrix-vector with NC layout from F16 to F32.
    - `pipeline_get_rows`: Pipelines for getting rows for each type.
    - `pipeline_get_rows_f32`: Pipelines for getting rows with F32 for each type.
    - `pipeline_acc_f32`: Pipeline for accumulating F32 values.
    - `pipeline_add`: Pipelines for addition operations with different data types.
    - `pipeline_add_norepeat`: Pipelines for addition operations without repeat with different data types.
    - `pipeline_sub`: Pipelines for subtraction operations with different data types.
    - `pipeline_sub_norepeat`: Pipelines for subtraction operations without repeat with different data types.
    - `pipeline_mul`: Pipelines for multiplication operations with different data types.
    - `pipeline_mul_norepeat`: Pipelines for multiplication operations without repeat with different data types.
    - `pipeline_div`: Pipelines for division operations with different data types.
    - `pipeline_div_norepeat`: Pipelines for division operations without repeat with different data types.
    - `pipeline_concat_f32`: Pipeline for concatenating F32 data.
    - `pipeline_concat_f16`: Pipeline for concatenating F16 data.
    - `pipeline_concat_i32`: Pipeline for concatenating I32 data.
    - `pipeline_upscale_f32`: Pipeline for upscaling F32 data.
    - `pipeline_scale_f32`: Pipeline for scaling F32 data.
    - `pipeline_sqr_f32`: Pipeline for squaring F32 data.
    - `pipeline_sin_f32`: Pipeline for computing sine of F32 data.
    - `pipeline_cos_f32`: Pipeline for computing cosine of F32 data.
    - `pipeline_clamp_f32`: Pipeline for clamping F32 data.
    - `pipeline_pad_f32`: Pipeline for padding F32 data.
    - `pipeline_repeat_f32`: Pipeline for repeating F32 data.
    - `pipeline_repeat_back_f32`: Pipeline for repeating F32 data backwards.
    - `pipeline_cpy_f32_f32`: Pipeline for copying F32 to F32 data.
    - `pipeline_cpy_f32_f16`: Pipeline for copying F32 to F16 data.
    - `pipeline_cpy_f16_f16`: Pipeline for copying F16 to F16 data.
    - `pipeline_cpy_f16_f32`: Pipeline for copying F16 to F32 data.
    - `pipeline_cpy_f32_bf16`: Pipeline for copying F32 to BF16 data.
    - `pipeline_contig_cpy_f32_f32`: Pipeline for contiguous copying of F32 to F32 data.
    - `pipeline_contig_cpy_f32_f16`: Pipeline for contiguous copying of F32 to F16 data.
    - `pipeline_contig_cpy_f16_f16`: Pipeline for contiguous copying of F16 to F16 data.
    - `pipeline_contig_cpy_f16_f32`: Pipeline for contiguous copying of F16 to F32 data.
    - `pipeline_contig_cpy_f32_bf16`: Pipeline for contiguous copying of F32 to BF16 data.
    - `pipeline_cpy_f32_quant`: Pipelines for copying F32 to quantized data for each type.
    - `pipeline_cpy_quant_f32`: Pipelines for copying quantized data to F32 for each type.
    - `pipeline_norm_f32`: Pipeline for normalizing F32 data.
    - `pipeline_group_norm_f32`: Pipeline for group normalization of F32 data.
    - `pipeline_rms_norm_f32`: Pipeline for RMS normalization of F32 data.
    - `pipeline_rms_norm_back_f32`: Pipeline for backward RMS normalization of F32 data.
    - `pipeline_l2_norm_f32`: Pipeline for L2 normalization of F32 data.
    - `pipeline_gelu`: Pipelines for GELU activation with different data types.
    - `pipeline_gelu_quick`: Pipelines for quick GELU activation with different data types.
    - `pipeline_silu`: Pipelines for SiLU activation with different data types.
    - `pipeline_relu`: Pipelines for ReLU activation with different data types.
    - `pipeline_tanh`: Pipelines for tanh activation with different data types.
    - `pipeline_sigmoid`: Pipelines for sigmoid activation with different data types.
    - `pipeline_leaky_relu_f32`: Pipeline for leaky ReLU activation with F32 data.
    - `pipeline_silu_back_f32`: Pipeline for backward SiLU activation with F32 data.
    - `pipeline_diag_mask_inf_f32`: Pipeline for diagonal masking with infinity for F32 data.
    - `pipeline_soft_max_f32`: Pipeline for softmax activation with F32 data.
    - `pipeline_soft_max_f32_f16`: Pipeline for softmax activation from F32 to F16 data.
    - `pipeline_soft_max_f32_wg512`: Pipeline for softmax activation with F32 data and workgroup size 512.
    - `pipeline_soft_max_f32_f16_wg512`: Pipeline for softmax activation from F32 to F16 data with workgroup size 512.
    - `pipeline_soft_max_back_f32`: Pipeline for backward softmax activation with F32 data.
    - `pipeline_rope_norm_f32`: Pipeline for rope normalization with F32 data.
    - `pipeline_rope_norm_f16`: Pipeline for rope normalization with F16 data.
    - `pipeline_rope_neox_f32`: Pipeline for rope normalization with NeoX and F32 data.
    - `pipeline_rope_neox_f16`: Pipeline for rope normalization with NeoX and F16 data.
    - `pipeline_rope_multi_f32`: Pipeline for multi-dimensional rope normalization with F32 data.
    - `pipeline_rope_multi_f16`: Pipeline for multi-dimensional rope normalization with F16 data.
    - `pipeline_rope_vision_f32`: Pipeline for vision rope normalization with F32 data.
    - `pipeline_rope_vision_f16`: Pipeline for vision rope normalization with F16 data.
    - `pipeline_argsort_f32`: Pipeline for argsort operation with F32 data.
    - `pipeline_sum_rows_f32`: Pipeline for summing rows with F32 data.
    - `pipeline_argmax_f32`: Pipeline for argmax operation with F32 data.
    - `pipeline_count_equal_i32`: Pipeline for counting equal elements with I32 data.
    - `pipeline_im2col_f32`: Pipeline for im2col operation with F32 data.
    - `pipeline_im2col_f32_f16`: Pipeline for im2col operation from F32 to F16 data.
    - `pipeline_timestep_embedding_f32`: Pipeline for timestep embedding with F32 data.
    - `pipeline_conv_transpose_1d_f32`: Pipeline for 1D transposed convolution with F32 data.
    - `pipeline_pool2d_f32`: Pipeline for 2D pooling with F32 data.
    - `pipeline_rwkv_wkv6_f32`: Pipeline for RWKV WKV6 operation with F32 data.
    - `pipeline_rwkv_wkv7_f32`: Pipeline for RWKV WKV7 operation with F32 data.
    - `pipeline_opt_step_adamw_f32`: Pipeline for AdamW optimization step with F32 data.
    - `pipeline_conv2d_dw_whcn_f32`: Pipeline for 2D depthwise convolution with WHCN layout and F32 data.
    - `pipeline_conv2d_dw_cwhn_f32`: Pipeline for 2D depthwise convolution with CWHN layout and F32 data.
    - `pipeline_flash_attn_f32_f16_D64_cm2`: Pipelines for flash attention with F32 to F16 data, D64, and CM2 layout.
    - `pipeline_flash_attn_f32_f16_D80_cm2`: Pipelines for flash attention with F32 to F16 data, D80, and CM2 layout.
    - `pipeline_flash_attn_f32_f16_D96_cm2`: Pipelines for flash attention with F32 to F16 data, D96, and CM2 layout.
    - `pipeline_flash_attn_f32_f16_D112_cm2`: Pipelines for flash attention with F32 to F16 data, D112, and CM2 layout.
    - `pipeline_flash_attn_f32_f16_D128_cm2`: Pipelines for flash attention with F32 to F16 data, D128, and CM2 layout.
    - `pipeline_flash_attn_f32_f16_D256_cm2`: Pipelines for flash attention with F32 to F16 data, D256, and CM2 layout.
    - `pipeline_flash_attn_f32_f16_D64_cm1`: Pipelines for flash attention with F32 to F16 data, D64, and CM1 layout.
    - `pipeline_flash_attn_f32_f16_D80_cm1`: Pipelines for flash attention with F32 to F16 data, D80, and CM1 layout.
    - `pipeline_flash_attn_f32_f16_D96_cm1`: Pipelines for flash attention with F32 to F16 data, D96, and CM1 layout.
    - `pipeline_flash_attn_f32_f16_D112_cm1`: Pipelines for flash attention with F32 to F16 data, D112, and CM1 layout.
    - `pipeline_flash_attn_f32_f16_D128_cm1`: Pipelines for flash attention with F32 to F16 data, D128, and CM1 layout.
    - `pipeline_flash_attn_f32_f16_D256_cm1`: Pipelines for flash attention with F32 to F16 data, D256, and CM1 layout.
    - `pipeline_flash_attn_f32_f16_D64`: Pipelines for flash attention with F32 to F16 data and D64.
    - `pipeline_flash_attn_f32_f16_D80`: Pipelines for flash attention with F32 to F16 data and D80.
    - `pipeline_flash_attn_f32_f16_D96`: Pipelines for flash attention with F32 to F16 data and D96.
    - `pipeline_flash_attn_f32_f16_D112`: Pipelines for flash attention with F32 to F16 data and D112.
    - `pipeline_flash_attn_f32_f16_D128`: Pipelines for flash attention with F32 to F16 data and D128.
    - `pipeline_flash_attn_f32_f16_D256`: Pipelines for flash attention with F32 to F16 data and D256.
    - `pipeline_flash_attn_split_k_reduce`: Pipeline for split-K reduction in flash attention.
    - `pipelines`: A map of pipeline references indexed by string keys.
    - `pipeline_descriptor_set_requirements`: A map of pipeline descriptor set requirements indexed by string keys.
    - `pinned_memory`: A vector of tuples representing pinned memory allocations.
    - `fence`: A fence for synchronizing operations.
    - `sync_staging`: A buffer for staging synchronization.
    - `buffer_type`: The type of backend buffer used.
    - `memory_logger`: A unique pointer to a memory logger for debugging.
    - `perf_logger`: A unique pointer to a performance logger.
    - `query_pool`: A query pool for performance queries.
    - `num_queries`: The number of queries available.
- **Description**: The `vk_device_struct` is a comprehensive data structure designed to encapsulate various properties and capabilities of a Vulkan device, including its physical and logical device representations, memory allocation parameters, and support for various computational features such as FP16, cooperative matrices, and subgroup operations. It also manages a wide array of Vulkan pipelines for different operations, ranging from matrix multiplications to activation functions, and includes synchronization primitives like fences and query pools for performance monitoring. This structure is integral to managing and optimizing the execution of Vulkan-based applications, providing detailed control over device-specific features and resources.
- **Member Functions**:
    - [`vk_device_struct::~vk_device_struct`](#vk_device_structvk_device_struct)

**Methods**

---
#### vk\_device\_struct::\~vk\_device\_struct<!-- {{#callable:vk_device_struct::~vk_device_struct}} -->
The destructor `~vk_device_struct` is responsible for cleaning up and releasing all Vulkan resources associated with a `vk_device_struct` instance.
- **Inputs**: None
- **Control Flow**:
    - Logs a debug message indicating the destruction of the device with its name.
    - Destroys the Vulkan fence associated with the device.
    - Destroys the Vulkan buffer used for synchronization staging.
    - Destroys the command pool associated with the compute queue.
    - Checks if the device is not using a single queue, and if so, destroys the command pool associated with the transfer queue.
    - Iterates over the `pipelines` map, checks if each pipeline is expired, and if not, locks and destroys the pipeline using [`ggml_vk_destroy_pipeline`](#ggml_vk_destroy_pipeline).
    - Clears the `pipelines` map to remove all entries.
    - Destroys the Vulkan device itself.
- **Output**: The function does not return any value as it is a destructor.
- **Functions called**:
    - [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer)
    - [`ggml_vk_destroy_pipeline`](#ggml_vk_destroy_pipeline)
- **See also**: [`vk_device_struct`](#vk_device_struct)  (Data Structure)



---
### vk\_buffer\_struct<!-- {{#data_structure:vk_buffer_struct}} -->
- **Type**: `struct`
- **Members**:
    - `buffer`: A Vulkan buffer handle initialized to VK_NULL_HANDLE.
    - `device_memory`: A Vulkan device memory handle initialized to VK_NULL_HANDLE.
    - `memory_property_flags`: Flags indicating the properties of the memory associated with the buffer.
    - `ptr`: A pointer to the memory mapped to the buffer.
    - `size`: The size of the buffer, initialized to 0.
    - `device`: A reference to the Vulkan device associated with the buffer.
- **Description**: The `vk_buffer_struct` is a C++ structure designed to encapsulate a Vulkan buffer and its associated memory management details. It includes handles for the buffer and device memory, memory property flags, a pointer to the mapped memory, and the size of the buffer. The structure also holds a reference to the Vulkan device used for resource management. The destructor ensures proper cleanup by freeing the device memory and destroying the buffer if the size is non-zero, thus preventing resource leaks.
- **Member Functions**:
    - [`vk_buffer_struct::~vk_buffer_struct`](#vk_buffer_structvk_buffer_struct)

**Methods**

---
#### vk\_buffer\_struct::\~vk\_buffer\_struct<!-- {{#callable:vk_buffer_struct::~vk_buffer_struct}} -->
The destructor `~vk_buffer_struct` releases Vulkan buffer resources if the buffer size is non-zero.
- **Inputs**: None
- **Control Flow**:
    - Check if the `size` attribute is zero; if so, exit the function immediately.
    - Log a debug message with the buffer and size information.
    - Free the device memory associated with `device_memory` using the Vulkan device.
    - Destroy the Vulkan buffer associated with `buffer` using the Vulkan device.
- **Output**: This function does not return any value; it performs cleanup operations on the Vulkan buffer resources.
- **See also**: [`vk_buffer_struct`](#vk_buffer_struct)  (Data Structure)



---
### vk\_subbuffer<!-- {{#data_structure:vk_subbuffer}} -->
- **Type**: `struct`
- **Members**:
    - `buffer`: A vk_buffer object representing the buffer associated with this subbuffer.
    - `offset`: A 64-bit unsigned integer indicating the offset within the buffer.
    - `size`: A 64-bit unsigned integer specifying the size of the subbuffer.
- **Description**: The `vk_subbuffer` struct represents a subregion of a Vulkan buffer, encapsulating a `vk_buffer` object along with an offset and size to define the specific portion of the buffer. It includes a conversion operator to `vk::DescriptorBufferInfo`, allowing it to be easily used in Vulkan descriptor sets, which require information about buffer regions.


---
### vk\_semaphore<!-- {{#data_structure:vk_semaphore}} -->
- **Type**: `struct`
- **Members**:
    - `s`: A Vulkan semaphore object used for synchronization.
    - `value`: A 64-bit unsigned integer representing the semaphore's value.
- **Description**: The `vk_semaphore` struct is a simple data structure used in Vulkan-based applications to manage synchronization between GPU operations. It contains a Vulkan semaphore object, `s`, which is used to signal and wait for operations to complete, and a `value` field that holds a 64-bit unsigned integer, typically used to track or manage the state or progress of the semaphore in synchronization tasks.


---
### vk\_submission<!-- {{#data_structure:vk_submission}} -->
- **Type**: `struct`
- **Members**:
    - `buffer`: A command buffer from the Vulkan API used to record commands for execution.
    - `wait_semaphores`: A vector of semaphores that the submission will wait on before execution.
    - `signal_semaphores`: A vector of semaphores that will be signaled once the submission is complete.
- **Description**: The `vk_submission` struct is designed to encapsulate a Vulkan command buffer along with associated synchronization primitives, specifically semaphores. It includes a command buffer for recording commands, and two vectors of semaphores: one for semaphores that must be waited on before the command buffer execution begins, and another for semaphores that will be signaled after the command buffer execution is completed. This struct is essential for managing synchronization in Vulkan applications, ensuring that command buffers are executed in the correct order with respect to other operations.


---
### vk\_mat\_mat\_push\_constants<!-- {{#data_structure:vk_mat_mat_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `M`: Represents the number of rows in the matrix.
    - `N`: Represents the number of columns in the matrix.
    - `K`: Represents the inner dimension for matrix multiplication.
    - `stride_a`: Specifies the stride for matrix A.
    - `stride_b`: Specifies the stride for matrix B.
    - `stride_d`: Specifies the stride for matrix D.
    - `batch_stride_a`: Specifies the batch stride for matrix A.
    - `batch_stride_b`: Specifies the batch stride for matrix B.
    - `batch_stride_d`: Specifies the batch stride for matrix D.
    - `k_split`: Indicates the split factor for the K dimension.
    - `ne02`: Represents an additional parameter for matrix operations.
    - `ne12`: Represents an additional parameter for matrix operations.
    - `broadcast2`: Indicates whether broadcasting is applied in a specific dimension.
    - `broadcast3`: Indicates whether broadcasting is applied in another specific dimension.
    - `padded_N`: Represents the padded size of the N dimension.
- **Description**: The `vk_mat_mat_push_constants` struct is designed to hold various parameters and configurations for matrix operations, particularly in the context of Vulkan-based computations. It includes dimensions for matrices (M, N, K), stride and batch stride information for matrices A, B, and D, as well as additional parameters for handling broadcasting and padding. This struct is likely used to pass constant data to shaders or compute pipelines, facilitating efficient matrix multiplication and related operations.


---
### vk\_mat\_vec\_push\_constants<!-- {{#data_structure:vk_mat_vec_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `ncols`: Represents the number of columns in a matrix.
    - `stride_a`: Specifies the stride for matrix A.
    - `stride_b`: Specifies the stride for matrix B.
    - `stride_d`: Specifies the stride for matrix D.
    - `batch_stride_a`: Defines the batch stride for matrix A.
    - `batch_stride_b`: Defines the batch stride for matrix B.
    - `batch_stride_d`: Defines the batch stride for matrix D.
    - `ne02`: An additional parameter, possibly related to matrix dimensions or operations.
    - `ne12`: Another parameter, possibly related to matrix dimensions or operations.
    - `broadcast2`: Indicates whether broadcasting is applied along the second dimension.
    - `broadcast3`: Indicates whether broadcasting is applied along the third dimension.
- **Description**: The `vk_mat_vec_push_constants` struct is designed to hold various parameters related to matrix operations, particularly in the context of Vulkan or similar graphics/compute APIs. It includes fields for specifying the number of columns, strides for different matrices, batch strides, and parameters for broadcasting, which are essential for efficient matrix computations and transformations.


---
### vk\_mat\_mat\_id\_push\_constants<!-- {{#data_structure:vk_mat_mat_id_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `M`: Represents the number of rows in the matrix.
    - `N`: Represents the number of columns in the matrix.
    - `K`: Represents the inner dimension for matrix multiplication.
    - `stride_a`: Specifies the stride for matrix A.
    - `stride_b`: Specifies the stride for matrix B.
    - `stride_d`: Specifies the stride for matrix D.
    - `batch_stride_a`: Specifies the batch stride for matrix A.
    - `batch_stride_b`: Specifies the batch stride for matrix B.
    - `batch_stride_d`: Specifies the batch stride for matrix D.
    - `nei0`: Represents a specific neighbor index or offset.
    - `nei1`: Represents another specific neighbor index or offset.
    - `nbi1`: Represents a specific batch index or offset.
    - `ne11`: Represents another specific neighbor index or offset.
    - `padded_N`: Represents the padded size of N for alignment purposes.
- **Description**: The `vk_mat_mat_id_push_constants` struct is designed to hold various parameters related to matrix operations, particularly for use in Vulkan compute shaders. It includes dimensions for matrix multiplication (M, N, K), stride values for accessing matrix data, batch stride values for handling batched operations, and additional indices or offsets for neighbor and batch processing. The `padded_N` field is used to ensure proper alignment of the matrix dimensions.


---
### vk\_mat\_vec\_id\_push\_constants<!-- {{#data_structure:vk_mat_vec_id_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `ncols`: Represents the number of columns.
    - `stride_a`: Specifies the stride for matrix A.
    - `stride_b`: Specifies the stride for matrix B.
    - `stride_d`: Specifies the stride for matrix D.
    - `batch_stride_a`: Defines the batch stride for matrix A.
    - `batch_stride_b`: Defines the batch stride for matrix B.
    - `batch_stride_d`: Defines the batch stride for matrix D.
    - `nei0`: Represents a specific index or element, possibly related to matrix operations.
    - `ne11`: Represents another specific index or element, possibly related to matrix operations.
- **Description**: The `vk_mat_vec_id_push_constants` struct is designed to hold various parameters related to matrix operations, particularly in the context of Vulkan or similar graphics APIs. It includes fields for specifying the number of columns, strides for different matrices (A, B, D), and batch strides, which are crucial for efficient data access and manipulation in batch processing. Additionally, it contains fields `nei0` and `ne11`, which likely serve as indices or identifiers for specific elements or operations within the matrix context.


---
### vk\_flash\_attn\_push\_constants<!-- {{#data_structure:vk_flash_attn_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `N`: Represents a 32-bit unsigned integer, likely used for a size or count.
    - `KV`: Represents a 32-bit unsigned integer, possibly related to key-value pairs.
    - `ne1`: Represents a 32-bit unsigned integer, possibly an index or size parameter.
    - `ne2`: Represents a 32-bit unsigned integer, possibly an index or size parameter.
    - `ne3`: Represents a 32-bit unsigned integer, possibly an index or size parameter.
    - `neq2`: Represents a 32-bit unsigned integer, possibly an index or size parameter.
    - `neq3`: Represents a 32-bit unsigned integer, possibly an index or size parameter.
    - `nek2`: Represents a 32-bit unsigned integer, possibly an index or size parameter.
    - `nek3`: Represents a 32-bit unsigned integer, possibly an index or size parameter.
    - `nev2`: Represents a 32-bit unsigned integer, possibly an index or size parameter.
    - `nev3`: Represents a 32-bit unsigned integer, possibly an index or size parameter.
    - `nem1`: Represents a 32-bit unsigned integer, possibly an index or size parameter.
    - `nb01`: Represents a 32-bit unsigned integer, possibly a block or batch size.
    - `nb02`: Represents a 32-bit unsigned integer, possibly a block or batch size.
    - `nb03`: Represents a 32-bit unsigned integer, possibly a block or batch size.
    - `nb11`: Represents a 32-bit unsigned integer, possibly a block or batch size.
    - `nb12`: Represents a 32-bit unsigned integer, possibly a block or batch size.
    - `nb13`: Represents a 32-bit unsigned integer, possibly a block or batch size.
    - `nb21`: Represents a 32-bit unsigned integer, possibly a block or batch size.
    - `nb22`: Represents a 32-bit unsigned integer, possibly a block or batch size.
    - `nb23`: Represents a 32-bit unsigned integer, possibly a block or batch size.
    - `nb31`: Represents a 32-bit unsigned integer, possibly a block or batch size.
    - `scale`: Represents a floating-point value, likely used for scaling operations.
    - `max_bias`: Represents a floating-point value, possibly used for bias adjustment.
    - `logit_softcap`: Represents a floating-point value, possibly used for softmax or similar operations.
    - `mask`: Represents a 32-bit unsigned integer, likely used for masking operations.
    - `n_head_log2`: Represents a 32-bit unsigned integer, possibly related to logarithmic head count.
    - `m0`: Represents a floating-point value, possibly a multiplier or coefficient.
    - `m1`: Represents a floating-point value, possibly a multiplier or coefficient.
    - `gqa_ratio`: Represents a 32-bit unsigned integer, possibly related to a ratio in GQA.
    - `split_kv`: Represents a 32-bit unsigned integer, possibly indicating a split in key-value pairs.
    - `k_num`: Represents a 32-bit unsigned integer, possibly indicating a count of keys.
- **Description**: The `vk_flash_attn_push_constants` struct is a data structure designed to hold a variety of parameters and constants used in a Vulkan-based flash attention mechanism. It includes numerous 32-bit unsigned integers and floating-point values that likely represent sizes, indices, scaling factors, biases, and other configuration parameters necessary for the operation of the attention mechanism. The struct is likely used to pass these constants efficiently to a shader or compute pipeline, enabling optimized execution of attention-related computations.


---
### vk\_op\_push\_constants<!-- {{#data_structure:vk_op_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `KX`: A 32-bit unsigned integer representing a constant value.
    - `KY`: A 32-bit unsigned integer representing another constant value.
    - `param1`: A floating-point number used as a parameter.
    - `param2`: A floating-point number used as another parameter.
- **Description**: The `vk_op_push_constants` struct is a simple data structure that holds two unsigned integer constants and two floating-point parameters. It is likely used in a graphics or compute context, possibly related to Vulkan operations, where these constants and parameters are pushed to a shader or similar processing unit to control or modify its behavior.


---
### vk\_op\_unary\_push\_constants<!-- {{#data_structure:vk_op_unary_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `ne`: Represents a 32-bit unsigned integer, possibly used as a count or identifier.
    - `ne00`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne01`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne02`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne03`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `nb00`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `nb01`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `nb02`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `nb03`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne10`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne11`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne12`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne13`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `nb10`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `nb11`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `nb12`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `nb13`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `misalign_offsets`: Represents a 32-bit unsigned integer, possibly used to indicate misalignment offsets.
    - `param1`: Represents a floating-point parameter, possibly used for calculations or configurations.
    - `param2`: Represents a floating-point parameter, possibly used for calculations or configurations.
    - `ne0_012mp`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne0_012L`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne0_01mp`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne0_01L`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne0_0mp`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne0_0L`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne1_012mp`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne1_012L`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne1_01mp`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne1_01L`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne1_0mp`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
    - `ne1_0L`: Represents a 32-bit unsigned integer, possibly used for a specific configuration or state.
- **Description**: The `vk_op_unary_push_constants` struct is a data structure designed to hold a collection of 32-bit unsigned integers and floating-point values, likely used for configuration or state management in a Vulkan-based application. The struct includes a variety of fields, each potentially representing different configuration states or parameters, such as `ne`, `nb`, and `param` values, which may be used in graphics or compute operations. The struct is constrained to a size of 128 bytes or less, ensuring it fits within certain memory limits, possibly for efficient GPU operations.


---
### vk\_op\_binary\_push\_constants<!-- {{#data_structure:vk_op_binary_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `ne`: Represents a 32-bit unsigned integer, possibly a count or index.
    - `ne00`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `ne01`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `ne02`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `ne03`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `nb00`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `nb01`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `nb02`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `nb03`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `ne10`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `ne11`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `ne12`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `ne13`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `nb10`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `nb11`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `nb12`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `nb13`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `ne20`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `ne21`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `ne22`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `ne23`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `nb20`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `nb21`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `nb22`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `nb23`: Represents a 32-bit unsigned integer, possibly part of a matrix or grid.
    - `misalign_offsets`: Represents a 32-bit unsigned integer, possibly indicating misalignment.
    - `param1`: Represents a floating-point parameter.
    - `param2`: Represents a floating-point parameter.
    - `param3`: Represents a 32-bit signed integer parameter.
- **Description**: The `vk_op_binary_push_constants` struct is a data structure that appears to be used for storing a series of 32-bit unsigned integers, possibly representing elements of a matrix or grid, along with some floating-point and signed integer parameters. The structure includes fields that might be used for operations involving binary push constants in a Vulkan-like graphics or compute context, with specific fields potentially indicating misalignment or other operational parameters.


---
### vk\_op\_diag\_mask\_push\_constants<!-- {{#data_structure:vk_op_diag_mask_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `ncols`: Represents the number of columns.
    - `rows_per_channel`: Indicates the number of rows per channel.
    - `n_past`: Stores the number of past elements or steps.
- **Description**: The `vk_op_diag_mask_push_constants` struct is a simple data structure used to store configuration parameters for a specific operation, likely related to a Vulkan operation involving diagonal masking. It contains three fields: `ncols` for the number of columns, `rows_per_channel` for the number of rows per channel, and `n_past` for tracking past elements or steps, which may be used in computations or rendering processes.


---
### vk\_op\_rope\_push\_constants<!-- {{#data_structure:vk_op_rope_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `ncols`: Represents the number of columns.
    - `n_dims`: Indicates the number of dimensions.
    - `freq_scale`: A scaling factor for frequency.
    - `p_delta_rows`: Represents the delta of rows.
    - `freq_base`: The base frequency value.
    - `ext_factor`: An external factor for calculations.
    - `attn_factor`: A factor related to attention mechanisms.
    - `corr_dims`: An array of two floats representing correlated dimensions.
    - `theta_scale`: A scaling factor for theta.
    - `has_ff`: Indicates the presence of a feed-forward component.
    - `ne02`: A specific parameter, possibly related to a version or type.
    - `s1`: A parameter, possibly a state or setting.
    - `s2`: Another parameter, possibly a state or setting.
    - `sections`: An array of four integers representing different sections.
    - `is_back`: Indicates if the operation is a backward pass.
- **Description**: The `vk_op_rope_push_constants` struct is a data structure used to encapsulate various parameters and constants related to operations in a Vulkan-based application, possibly involving rope or attention mechanisms. It includes fields for dimensions, frequency scaling, and other factors that influence the operation's behavior, such as attention and external factors. The struct also contains arrays for correlated dimensions and sections, as well as flags indicating specific states or configurations.


---
### vk\_op\_soft\_max\_push\_constants<!-- {{#data_structure:vk_op_soft_max_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `KX`: Represents a 32-bit unsigned integer value, likely used as a dimension or size parameter.
    - `KY`: Represents a 32-bit unsigned integer value, likely used as a dimension or size parameter.
    - `scale`: A floating-point value used to scale some aspect of the operation.
    - `max_bias`: A floating-point value representing a maximum bias applied in the operation.
    - `m0`: A floating-point value, possibly a coefficient or parameter in a calculation.
    - `m1`: A floating-point value, possibly a coefficient or parameter in a calculation.
    - `n_head_log2`: A 32-bit unsigned integer representing the logarithm base 2 of the number of heads, likely in a multi-head operation.
    - `nrows_x`: A 32-bit unsigned integer representing the number of rows in a matrix or grid.
- **Description**: The `vk_op_soft_max_push_constants` struct is a data structure used to store various parameters for a Vulkan operation, likely related to a softmax computation. It includes integer values for dimensions or sizes (`KX`, `KY`, `n_head_log2`, `nrows_x`) and floating-point values for scaling and biasing (`scale`, `max_bias`, `m0`, `m1`). This struct is designed to be used as push constants in Vulkan, which are small pieces of data that can be quickly accessed by shaders.


---
### vk\_op\_argsort\_push\_constants<!-- {{#data_structure:vk_op_argsort_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `ncols`: Represents the number of columns.
    - `ncols_pad`: Represents the padded number of columns.
    - `order`: Indicates the sorting order, typically ascending or descending.
- **Description**: The `vk_op_argsort_push_constants` struct is used to define constants for a Vulkan operation related to argument sorting. It includes fields for specifying the number of columns (`ncols`), the padded number of columns (`ncols_pad`), and the sorting order (`order`). This struct is likely used to pass these constants to a shader or compute operation in a Vulkan-based application, facilitating efficient sorting operations.


---
### vk\_op\_im2col\_push\_constants<!-- {{#data_structure:vk_op_im2col_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `batch_offset`: Specifies the offset for the batch in the operation.
    - `offset_delta`: Defines the change in offset for each operation.
    - `IC`: Represents the number of input channels.
    - `IW`: Indicates the input width.
    - `IH`: Indicates the input height.
    - `OW`: Indicates the output width.
    - `OH`: Indicates the output height.
    - `KW`: Specifies the kernel width.
    - `KH`: Specifies the kernel height.
    - `pelements`: Represents the number of processing elements.
    - `CHW`: Represents the product of channels, height, and width.
    - `s0`: Defines the stride in the first dimension.
    - `s1`: Defines the stride in the second dimension.
    - `p0`: Specifies the padding in the first dimension.
    - `p1`: Specifies the padding in the second dimension.
    - `d0`: Specifies the dilation in the first dimension.
    - `d1`: Specifies the dilation in the second dimension.
- **Description**: The `vk_op_im2col_push_constants` struct is used to define a set of parameters for an image-to-column operation in a Vulkan-based application. It includes various fields that specify dimensions and configurations such as input and output sizes, kernel dimensions, strides, padding, and dilation factors, which are essential for configuring the operation's behavior in a neural network or image processing context.


---
### vk\_op\_timestep\_embedding\_push\_constants<!-- {{#data_structure:vk_op_timestep_embedding_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `nb1`: Represents a 32-bit unsigned integer value, likely used as a parameter or identifier.
    - `dim`: Represents a 32-bit unsigned integer value, likely indicating a dimension or size.
    - `max_period`: Represents a 32-bit unsigned integer value, likely specifying a maximum period or limit.
- **Description**: The `vk_op_timestep_embedding_push_constants` struct is a simple data structure that contains three 32-bit unsigned integer fields. These fields are likely used to store parameters or configuration settings related to timestep embedding operations, possibly in a Vulkan graphics or compute context. The fields `nb1`, `dim`, and `max_period` suggest roles related to identifiers, dimensions, and period limits, respectively.


---
### vk\_op\_conv\_transpose\_1d\_push\_constants<!-- {{#data_structure:vk_op_conv_transpose_1d_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `Cout`: Represents the number of output channels.
    - `Cin`: Represents the number of input channels.
    - `K`: Represents the kernel size.
    - `L`: Represents the length of the input.
    - `KL`: Represents the product of kernel size and length.
    - `nb01`: Represents a specific block size or parameter related to the operation.
    - `nb02`: Represents another specific block size or parameter related to the operation.
    - `nb11`: Represents yet another specific block size or parameter related to the operation.
    - `nb1`: Represents a block size or parameter related to the operation.
    - `s0`: Represents a stride or step size for the operation.
- **Description**: The `vk_op_conv_transpose_1d_push_constants` struct is designed to hold various parameters for a 1D transposed convolution operation in a Vulkan-based application. It includes fields for the number of input and output channels, kernel size, input length, and a product of kernel size and length, as well as several block size parameters and a stride value. These parameters are likely used to configure and optimize the convolution operation on a GPU.


---
### vk\_op\_pool2d\_push\_constants<!-- {{#data_structure:vk_op_pool2d_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `IW`: Represents the input width.
    - `IH`: Represents the input height.
    - `OW`: Represents the output width.
    - `OH`: Represents the output height.
    - `OC`: Represents the output channels.
    - `pelements`: Represents the number of pooling elements.
    - `op`: Represents the operation type.
    - `k0`: Represents the kernel size in the first dimension.
    - `k1`: Represents the kernel size in the second dimension.
    - `s0`: Represents the stride in the first dimension.
    - `s1`: Represents the stride in the second dimension.
    - `p0`: Represents the padding in the first dimension.
    - `p1`: Represents the padding in the second dimension.
- **Description**: The `vk_op_pool2d_push_constants` struct is used to define the parameters for a 2D pooling operation in a Vulkan-based application. It includes dimensions for input and output, the number of output channels, and parameters for the pooling operation such as kernel size, stride, and padding. This struct is likely used to pass these parameters to a shader or compute operation in a Vulkan pipeline.


---
### vk\_op\_rwkv\_wkv6\_push\_constants<!-- {{#data_structure:vk_op_rwkv_wkv6_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `B`: Represents a 32-bit unsigned integer, likely used as a parameter or configuration value.
    - `T`: Represents a 32-bit unsigned integer, likely used as a parameter or configuration value.
    - `C`: Represents a 32-bit unsigned integer, likely used as a parameter or configuration value.
    - `H`: Represents a 32-bit unsigned integer, likely used as a parameter or configuration value.
- **Description**: The `vk_op_rwkv_wkv6_push_constants` struct is a simple data structure containing four 32-bit unsigned integer fields: B, T, C, and H. These fields are likely used to store configuration or parameter values for a Vulkan operation, possibly related to a specific shader or compute task. The naming of the fields suggests they may represent dimensions or other key parameters in a computational context.


---
### vk\_op\_rwkv\_wkv7\_push\_constants<!-- {{#data_structure:vk_op_rwkv_wkv7_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `B`: Represents a 32-bit unsigned integer value.
    - `T`: Represents a 32-bit unsigned integer value.
    - `C`: Represents a 32-bit unsigned integer value.
    - `H`: Represents a 32-bit unsigned integer value.
- **Description**: The `vk_op_rwkv_wkv7_push_constants` struct is a simple data structure that holds four 32-bit unsigned integer values, labeled B, T, C, and H. These fields are likely used as push constants in a Vulkan operation, which are small amounts of data passed to shaders that can be changed frequently and are used to control rendering operations.


---
### vk\_op\_conv2d\_dw\_push\_constants<!-- {{#data_structure:vk_op_conv2d_dw_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `ne`: Represents the number of elements.
    - `batches`: Indicates the number of batches.
    - `channels`: Specifies the number of channels.
    - `dst_w`: Denotes the destination width.
    - `dst_h`: Denotes the destination height.
    - `src_w`: Represents the source width.
    - `src_h`: Represents the source height.
    - `knl_w`: Specifies the kernel width.
    - `knl_h`: Specifies the kernel height.
    - `stride_x`: Indicates the stride in the x-direction.
    - `stride_y`: Indicates the stride in the y-direction.
    - `pad_x`: Represents the padding in the x-direction.
    - `pad_y`: Represents the padding in the y-direction.
    - `dilation_x`: Specifies the dilation in the x-direction.
    - `dilation_y`: Specifies the dilation in the y-direction.
- **Description**: The `vk_op_conv2d_dw_push_constants` struct is designed to encapsulate the parameters required for a depthwise convolution operation in a Vulkan-based application. It includes fields for specifying the number of elements, batches, and channels, as well as dimensions for both the source and destination images. Additionally, it defines kernel dimensions, stride, padding, and dilation values for both x and y directions, which are essential for configuring the convolution operation.


---
### vk\_op\_upscale\_push\_constants<!-- {{#data_structure:vk_op_upscale_push_constants}} -->
- **Type**: `struct`
- **Members**:
    - `ne`: Represents a 32-bit unsigned integer, possibly used for element count or similar purpose.
    - `a_offset`: Represents a 32-bit unsigned integer, likely used as an offset value for array or buffer operations.
    - `d_offset`: Represents a 32-bit unsigned integer, likely used as an offset value for data or buffer operations.
    - `nb00`: Represents a 32-bit unsigned integer, possibly used for a specific buffer or block index.
    - `nb01`: Represents a 32-bit unsigned integer, possibly used for a specific buffer or block index.
    - `nb02`: Represents a 32-bit unsigned integer, possibly used for a specific buffer or block index.
    - `nb03`: Represents a 32-bit unsigned integer, possibly used for a specific buffer or block index.
    - `ne10`: Represents a 32-bit unsigned integer, possibly used for a specific element or entity index.
    - `ne11`: Represents a 32-bit unsigned integer, possibly used for a specific element or entity index.
    - `ne12`: Represents a 32-bit unsigned integer, possibly used for a specific element or entity index.
    - `ne13`: Represents a 32-bit unsigned integer, possibly used for a specific element or entity index.
    - `sf0`: Represents a floating-point value, likely used as a scaling factor or similar purpose.
    - `sf1`: Represents a floating-point value, likely used as a scaling factor or similar purpose.
    - `sf2`: Represents a floating-point value, likely used as a scaling factor or similar purpose.
    - `sf3`: Represents a floating-point value, likely used as a scaling factor or similar purpose.
- **Description**: The `vk_op_upscale_push_constants` struct is a data structure used to store a set of constants that are likely used in a Vulkan operation related to upscaling. It contains several 32-bit unsigned integers and floating-point values, which are probably used as offsets, indices, and scaling factors in the upscaling process. The struct is designed to be passed to a Vulkan shader or compute operation, providing necessary parameters for the operation's execution.


---
### vk\_staging\_memcpy<!-- {{#data_structure:vk_staging_memcpy}} -->
- **Type**: `struct`
- **Members**:
    - `dst`: A pointer to the destination memory where data will be copied.
    - `src`: A constant pointer to the source memory from which data will be copied.
    - `n`: The number of bytes to copy from the source to the destination.
- **Description**: The `vk_staging_memcpy` struct is a simple data structure designed to facilitate memory copying operations. It encapsulates the necessary information for a memory copy operation, including pointers to the source and destination memory locations and the size of the data to be copied. This struct is typically used in scenarios where memory management and data transfer are critical, such as in graphics or system programming.
- **Member Functions**:
    - [`vk_staging_memcpy::vk_staging_memcpy`](#vk_staging_memcpyvk_staging_memcpy)

**Methods**

---
#### vk\_staging\_memcpy::vk\_staging\_memcpy<!-- {{#callable:vk_staging_memcpy::vk_staging_memcpy}} -->
The `vk_staging_memcpy` function is a constructor for the `vk_staging_memcpy` struct that initializes its members with the provided destination, source, and size parameters.
- **Inputs**:
    - `_dst`: A pointer to the destination memory where data will be copied.
    - `_src`: A pointer to the source memory from which data will be copied.
    - `_n`: The number of bytes to copy from the source to the destination.
- **Control Flow**:
    - The constructor initializes the `dst` member with the `_dst` parameter.
    - The constructor initializes the `src` member with the `_src` parameter.
    - The constructor initializes the `n` member with the `_n` parameter.
- **Output**: This constructor does not return a value; it initializes the members of the `vk_staging_memcpy` struct.
- **See also**: [`vk_staging_memcpy`](#vk_staging_memcpy)  (Data Structure)



---
### vk\_context\_struct<!-- {{#data_structure:vk_context_struct}} -->
- **Type**: `struct`
- **Members**:
    - `s`: A pointer to a vk_submission object.
    - `seqs`: A vector containing vk_sequence objects.
    - `exit_tensor_idx`: An integer representing the exit tensor index.
    - `in_memcpys`: A vector of vk_staging_memcpy objects for input memory copies.
    - `out_memcpys`: A vector of vk_staging_memcpy objects for output memory copies.
    - `q`: A pointer to a vk_queue object.
- **Description**: The `vk_context_struct` is a structure that encapsulates various components related to Vulkan operations, including submission, sequencing, and memory copying. It holds pointers to submission and queue objects, vectors for sequences and memory copy operations, and an integer for tracking the exit tensor index. This structure is designed to manage and coordinate Vulkan tasks efficiently, and it is often used in conjunction with a shared pointer typedef `vk_context` for easier memory management.


---
### ggml\_vk\_garbage\_collector<!-- {{#data_structure:ggml_vk_garbage_collector}} -->
- **Type**: `struct`
- **Members**:
    - `tl_semaphores`: A vector of Vulkan semaphores used for timeline synchronization.
    - `semaphores`: A vector of Vulkan semaphores for general synchronization purposes.
    - `events`: A vector of Vulkan events used for GPU-CPU synchronization.
    - `temp_buffers`: A vector of Vulkan buffers used temporarily during operations.
    - `contexts`: A vector of Vulkan contexts representing different execution environments.
- **Description**: The `ggml_vk_garbage_collector` struct is designed to manage and organize Vulkan resources that need to be cleaned up or synchronized. It contains vectors of Vulkan semaphores, events, buffers, and contexts, which are essential for handling synchronization and resource management in Vulkan-based applications. This struct helps in efficiently managing the lifecycle of Vulkan objects, ensuring that resources are properly released and synchronized across different stages of execution.


---
### vk\_memory\_logger<!-- {{#data_structure:vk_memory_logger}} -->
- **Type**: `class`
- **Members**:
    - `allocations`: A map that tracks memory allocations with vk::Buffer as the key and size_t as the value.
    - `total_device`: A size_t variable that keeps track of the total device memory allocated.
    - `total_host`: A size_t variable that keeps track of the total host memory allocated.
- **Description**: The `vk_memory_logger` class is designed to manage and log memory allocations and deallocations for Vulkan buffers. It maintains a map of current allocations, allowing for efficient tracking of memory usage. The class also keeps a running total of memory allocated on the device and host, providing a comprehensive overview of memory usage in a Vulkan application.
- **Member Functions**:
    - [`vk_memory_logger::vk_memory_logger`](#vk_memory_loggervk_memory_logger)
    - [`vk_memory_logger::log_allocation`](#vk_memory_loggerlog_allocation)
    - [`vk_memory_logger::log_deallocation`](#vk_memory_loggerlog_deallocation)

**Methods**

---
#### vk\_memory\_logger::vk\_memory\_logger<!-- {{#callable:vk_memory_logger::vk_memory_logger}} -->
The `vk_memory_logger` constructor initializes a memory logger with zeroed total device and host memory usage.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes the `total_device` and `total_host` member variables to zero.
- **Output**: An instance of the `vk_memory_logger` class with initialized memory tracking variables.
- **See also**: [`vk_memory_logger`](#vk_memory_logger)  (Data Structure)


---
#### vk\_memory\_logger::log\_allocation<!-- {{#callable:vk_memory_logger::log_allocation}} -->
The `log_allocation` function logs memory allocation details for a Vulkan buffer, updating total device and host memory usage.
- **Inputs**:
    - `buf_ref`: A reference to a Vulkan buffer object, encapsulated in a `vk_buffer_ref`.
    - `size`: The size of the memory allocation to be logged, in bytes.
- **Control Flow**:
    - Acquire a lock on `log_mutex` to ensure thread safety during logging.
    - Lock the `vk_buffer_ref` to obtain a `vk_buffer` object.
    - Determine if the buffer is device-local by checking its memory property flags.
    - Set the allocation type to 'device' if the buffer is device-local, otherwise 'host'.
    - Record the allocation size in the `allocations` map using the buffer as the key.
    - Update `total_device` or `total_host` based on the allocation type.
    - Log the allocation details, including buffer name, size, type, and updated totals, using `VK_LOG_MEMORY`.
- **Output**: The function does not return a value; it updates internal state and logs information.
- **Functions called**:
    - [`format_size`](#format_size)
- **See also**: [`vk_memory_logger`](#vk_memory_logger)  (Data Structure)


---
#### vk\_memory\_logger::log\_deallocation<!-- {{#callable:vk_memory_logger::log_deallocation}} -->
The `log_deallocation` function logs the deallocation of a Vulkan buffer, updating the total memory usage and removing the buffer from the allocation map.
- **Inputs**:
    - `buf_ref`: A `vk_buffer_ref` object representing a reference to a Vulkan buffer that is being deallocated.
- **Control Flow**:
    - Check if `buf_ref` is expired or if the buffer size is zero; if so, return immediately.
    - Acquire a lock on `log_mutex` to ensure thread-safe access to shared resources.
    - Lock the `buf_ref` to obtain a `vk_buffer` object and determine if the memory is device-local or host-local.
    - Find the buffer in the `allocations` map to retrieve its size.
    - Update `total_device` or `total_host` based on the memory type of the buffer being deallocated.
    - If the buffer is found in the `allocations` map, log the deallocation and erase the buffer from the map.
    - If the buffer is not found, log an error indicating an attempt to deallocate unknown memory.
- **Output**: The function does not return any value; it performs logging and updates internal state.
- **Functions called**:
    - [`format_size`](#format_size)
- **See also**: [`vk_memory_logger`](#vk_memory_logger)  (Data Structure)



---
### vk\_perf\_logger<!-- {{#data_structure:vk_perf_logger}} -->
- **Type**: `class`
- **Members**:
    - `timings`: A map that associates operation names with a vector of timing values in microseconds.
- **Description**: The `vk_perf_logger` class is designed to log and print performance timings for Vulkan operations. It maintains a private member `timings`, which is a map that stores operation names as keys and vectors of timing values as values. The class provides methods to log timings for different operations and to print the collected timings, which can be useful for performance analysis and debugging in Vulkan applications.
- **Member Functions**:
    - [`vk_perf_logger::print_timings`](#vk_perf_loggerprint_timings)
    - [`vk_perf_logger::log_timing`](#vk_perf_loggerlog_timing)

**Methods**

---
#### vk\_perf\_logger::print\_timings<!-- {{#callable:vk_perf_logger::print_timings}} -->
The `print_timings` function outputs the average execution time of various Vulkan operations and clears the stored timing data.
- **Inputs**: None
- **Control Flow**:
    - Prints a header 'Vulkan Timings:' to standard error output.
    - Iterates over each entry in the `timings` map, where each entry consists of a string key and a vector of uint64_t values representing timing data.
    - For each entry, calculates the total time by summing all timing values in the vector.
    - Calculates the average time by dividing the total time by the number of timing entries and converts it to microseconds.
    - Prints the operation name, the number of recorded timings, and the average time in microseconds to standard error output.
    - Clears the `timings` map to remove all stored timing data.
- **Output**: The function does not return any value; it outputs timing information to the standard error stream and clears the `timings` map.
- **See also**: [`vk_perf_logger`](#vk_perf_logger)  (Data Structure)


---
#### vk\_perf\_logger::log\_timing<!-- {{#callable:vk_perf_logger::log_timing}} -->
The `log_timing` function records the execution time of operations performed on a tensor node, categorizing them based on the operation type and dimensions.
- **Inputs**:
    - `node`: A pointer to a `ggml_tensor` structure representing the tensor node whose operation timing is being logged.
    - `time`: A `uint64_t` value representing the execution time of the operation in nanoseconds.
- **Control Flow**:
    - Check if the operation type of the node is `GGML_OP_UNARY`; if true, log the time under the unary operation name and return.
    - Check if the operation type is `GGML_OP_MUL_MAT` or `GGML_OP_MUL_MAT_ID`; if true, calculate dimensions `m`, `n`, and `k` from the node's source tensors.
    - Construct a name for the operation based on its type and dimensions, appending '_VEC' if `n` equals 1, and log the time under this name.
    - If none of the above conditions are met, log the time under the general operation name.
- **Output**: The function does not return a value; it updates the `timings` map with the execution time categorized by operation type and dimensions.
- **Functions called**:
    - [`ggml_unary_op_name`](../ggml.c.driver.md#ggml_unary_op_name)
    - [`ggml_get_unary_op`](../ggml.c.driver.md#ggml_get_unary_op)
    - [`ggml_op_name`](../ggml.c.driver.md#ggml_op_name)
- **See also**: [`vk_perf_logger`](#vk_perf_logger)  (Data Structure)



---
### ggml\_backend\_vk\_context<!-- {{#data_structure:ggml_backend_vk_context}} -->
- **Type**: `struct`
- **Members**:
    - `name`: A string representing the name of the Vulkan context.
    - `device`: An instance of vk_device representing the Vulkan device.
    - `semaphore_idx`: An index for semaphores used in synchronization.
    - `event_idx`: An index for events used in synchronization.
    - `gc`: An instance of ggml_vk_garbage_collector for managing resource cleanup.
    - `prealloc_size_x`: Size for preallocated buffer x.
    - `prealloc_size_y`: Size for preallocated buffer y.
    - `prealloc_size_split_k`: Size for preallocated buffer split_k.
    - `prealloc_x`: A Vulkan buffer for preallocated x data.
    - `prealloc_y`: A Vulkan buffer for preallocated y data.
    - `prealloc_split_k`: A Vulkan buffer for preallocated split_k data.
    - `fence`: A Vulkan fence for synchronization.
    - `almost_ready_fence`: A Vulkan fence indicating almost ready state.
    - `almost_ready_fence_pending`: A boolean indicating if the almost ready fence is pending.
    - `buffer_pool`: An array of Vulkan buffers for pooling resources.
    - `compute_ctx`: A reference to a Vulkan compute context.
    - `transfer_ctx`: A reference to a Vulkan transfer context.
    - `tensor_ctxs`: A vector of Vulkan context references for tensor operations.
- **Description**: The `ggml_backend_vk_context` struct is a comprehensive data structure designed to manage and organize various Vulkan resources and contexts for a backend system. It includes fields for device management, synchronization indices, garbage collection, and preallocated buffer sizes and instances. Additionally, it maintains synchronization fences and a pool of Vulkan buffers, along with references to compute and transfer contexts, and a collection of tensor contexts, facilitating efficient resource management and operations in a Vulkan-based environment.


---
### ggml\_backend\_vk\_buffer\_context<!-- {{#data_structure:ggml_backend_vk_buffer_context}} -->
- **Type**: `struct`
- **Members**:
    - `device`: A reference to a Vulkan device.
    - `dev_buffer`: A Vulkan buffer associated with the device.
    - `name`: A string representing the name of the buffer context.
- **Description**: The `ggml_backend_vk_buffer_context` struct is designed to manage a Vulkan buffer within a specific device context. It holds a reference to a Vulkan device, a Vulkan buffer, and a name for identification purposes. The constructor initializes these members, and the destructor ensures proper cleanup by destroying the Vulkan buffer when the context is no longer needed.
- **Member Functions**:
    - [`ggml_backend_vk_buffer_context::ggml_backend_vk_buffer_context`](#ggml_backend_vk_buffer_contextggml_backend_vk_buffer_context)
    - [`ggml_backend_vk_buffer_context::~ggml_backend_vk_buffer_context`](#ggml_backend_vk_buffer_contextggml_backend_vk_buffer_context)

**Methods**

---
#### ggml\_backend\_vk\_buffer\_context::ggml\_backend\_vk\_buffer\_context<!-- {{#callable:ggml_backend_vk_buffer_context::ggml_backend_vk_buffer_context}} -->
The `ggml_backend_vk_buffer_context` constructor initializes a Vulkan buffer context with a device reference, a buffer, and a name.
- **Inputs**:
    - `device`: A reference to a Vulkan device (`vk_device_ref`) that the buffer is associated with.
    - `dev_buffer`: An rvalue reference to a Vulkan buffer (`vk_buffer&&`) that will be managed by this context.
    - `name`: A reference to a string (`std::string&`) representing the name of the buffer context.
- **Control Flow**:
    - The constructor initializes the `device` member with the provided `device` argument.
    - The `dev_buffer` member is initialized with the provided `dev_buffer` argument using move semantics.
    - The `name` member is initialized with the provided `name` argument.
- **Output**: The constructor does not return a value; it initializes the members of the `ggml_backend_vk_buffer_context` structure.
- **See also**: [`ggml_backend_vk_buffer_context`](#ggml_backend_vk_buffer_context)  (Data Structure)


---
#### ggml\_backend\_vk\_buffer\_context::\~ggml\_backend\_vk\_buffer\_context<!-- {{#callable:ggml_backend_vk_buffer_context::~ggml_backend_vk_buffer_context}} -->
The destructor `~ggml_backend_vk_buffer_context` is responsible for cleaning up resources by destroying the Vulkan buffer associated with the context.
- **Inputs**: None
- **Control Flow**:
    - The destructor is called when an instance of `ggml_backend_vk_buffer_context` is destroyed.
    - It invokes the function [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer) with `dev_buffer` as the argument to release the Vulkan buffer resources.
- **Output**: There is no return value as this is a destructor.
- **Functions called**:
    - [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer)
- **See also**: [`ggml_backend_vk_buffer_context`](#ggml_backend_vk_buffer_context)  (Data Structure)



---
### vk\_instance\_t<!-- {{#data_structure:vk_instance_t}} -->
- **Type**: `struct`
- **Members**:
    - `instance`: A Vulkan instance handle from the Vulkan API.
    - `device_indices`: A vector containing indices of devices.
    - `devices`: An array of Vulkan devices with a maximum size defined by GGML_VK_MAX_DEVICES.
- **Description**: The `vk_instance_t` struct is a data structure used to encapsulate a Vulkan instance and its associated devices. It contains a Vulkan instance handle, a vector of device indices, and an array of Vulkan devices, allowing for the management and interaction with multiple Vulkan devices within a single instance. This struct is essential for applications that require handling multiple devices in a Vulkan environment.


---
### FaCodePath<!-- {{#data_structure:FaCodePath}} -->
- **Type**: `enum`
- **Members**:
    - `FA_SCALAR`: Represents a scalar code path.
    - `FA_COOPMAT1`: Represents the first cooperative matrix code path.
    - `FA_COOPMAT2`: Represents the second cooperative matrix code path.
- **Description**: The `FaCodePath` enumeration defines a set of constants that represent different code paths for handling various types of operations or data structures, such as scalar operations and cooperative matrix operations. This enum is likely used to select or switch between different processing strategies or optimizations in a program.


---
### GpuPipelineConfig<!-- {{#data_structure:GpuPipelineConfig}} -->
- **Type**: `struct`
- **Members**:
    - `arch`: GPU architecture identifier.
    - `pipelines`: Mapping of pipeline names to their specific subgroup sizes.
    - `default_subgroup_size`: Default subgroup size for this GPU, defaults to 0 if not explicitly provided.
- **Description**: The `GpuPipelineConfig` struct is designed to configure GPU pipeline settings, specifically tailored to different GPU architectures. It includes an architecture identifier (`arch`) to specify the GPU type, a mapping (`pipelines`) that associates pipeline names with their respective subgroup sizes, and a default subgroup size (`default_subgroup_size`) that is used when no specific size is provided. This configuration is crucial for optimizing GPU performance by aligning pipeline execution with the hardware's capabilities.


---
### ggml\_backend\_vk\_device\_context<!-- {{#data_structure:ggml_backend_vk_device_context}} -->
- **Type**: `struct`
- **Members**:
    - `device`: Represents the device identifier as a size_t.
    - `name`: Stores the name of the device as a std::string.
    - `description`: Holds a description of the device as a std::string.
- **Description**: The `ggml_backend_vk_device_context` struct is designed to encapsulate information about a Vulkan device in the context of the GGML backend. It includes a device identifier, a name, and a description, all of which are essential for identifying and describing the device within the Vulkan backend framework.


# Functions

---
### is\_pow2<!-- {{#callable:is_pow2}} -->
Determines if a given unsigned 32-bit integer is a power of two.
- **Inputs**:
    - `x`: An unsigned 32-bit integer to be checked if it is a power of two.
- **Control Flow**:
    - The function first checks if `x` is greater than 1, ensuring that only positive integers greater than 1 are considered.
    - It then uses the bitwise AND operation between `x` and `x-1` to determine if `x` is a power of two, as a power of two has exactly one bit set in its binary representation.
- **Output**: Returns a boolean value: true if `x` is a power of two, false otherwise.


---
### get\_device\_architecture<!-- {{#callable:get_device_architecture}} -->
Determines the architecture of a Vulkan physical device based on its properties and supported extensions.
- **Inputs**:
    - `device`: A reference to a `vk::PhysicalDevice` object representing the Vulkan physical device whose architecture is to be determined.
- **Control Flow**:
    - Retrieves the properties of the given `device` using `getProperties()`.
    - Checks if the vendor ID of the device is AMD or Intel to determine the architecture-specific logic.
    - For AMD devices, enumerates the supported extensions and checks for specific extensions related to shader core properties and subgroup size control.
    - If the required extensions are present, retrieves additional properties using `getProperties2()` and checks subgroup size to classify the architecture as AMD GCN, RDNA1, RDNA2, or RDNA3.
    - For Intel devices, enumerates the supported extensions and checks for subgroup size control, then retrieves properties to classify the architecture as Intel XE2 based on the minimum subgroup size.
    - If none of the conditions for AMD or Intel architectures are met, returns `vk_device_architecture::OTHER`.
- **Output**: Returns an enumeration value of type `vk_device_architecture` indicating the detected architecture of the Vulkan physical device, or `OTHER` if the architecture cannot be determined.


---
### init\_fastdiv\_values<!-- {{#callable:init_fastdiv_values}} -->
Initializes the values for fast division by computing a multiplier and the logarithmic ceiling of a divisor.
- **Inputs**:
    - `d`: The divisor for which the fast division values are being initialized.
    - `mp`: A reference to a variable that will store the computed multiplier for fast division.
    - `L`: A reference to a variable that will store the computed logarithmic ceiling of the divisor.
- **Control Flow**:
    - The function starts by initializing `L` to 0.
    - It enters a while loop that increments `L` until `2^L` is no longer less than `d` or `L` reaches 32.
    - After determining `L`, it calculates the multiplier `mp` using a formula that involves bit shifting and division.
- **Output**: The function does not return a value; instead, it modifies the references `mp` and `L` to store the computed multiplier and logarithmic ceiling, respectively.


---
### init\_pushconst\_fastdiv<!-- {{#callable:init_pushconst_fastdiv}} -->
Initializes fast division constants for a given `vk_op_unary_push_constants` object.
- **Inputs**:
    - `p`: A reference to a `vk_op_unary_push_constants` object containing various numerical values used for fast division calculations.
- **Control Flow**:
    - The function calls [`init_fastdiv_values`](#init_fastdiv_values) multiple times, each time with different parameters derived from the fields of the `p` object.
    - Each call to [`init_fastdiv_values`](#init_fastdiv_values) computes magic values for fast division based on the product of specific fields of `p`.
- **Output**: The function does not return a value; it modifies the state of the `vk_op_unary_push_constants` object by initializing its division constants.
- **Functions called**:
    - [`init_fastdiv_values`](#init_fastdiv_values)


---
### format\_size<!-- {{#callable:format_size}} -->
Converts a size in bytes to a human-readable string format with appropriate units.
- **Inputs**:
    - `size`: The size in bytes to be formatted.
- **Control Flow**:
    - Defines constants for kibibytes, mebibytes, and gibibytes.
    - Initializes a string stream to format the output.
    - Checks if the size is greater than or equal to gibibytes and formats accordingly.
    - If not, checks if the size is greater than or equal to mebibytes and formats accordingly.
    - If not, checks if the size is greater than or equal to kibibytes and formats accordingly.
    - If none of the above conditions are met, formats the size in bytes.
- **Output**: Returns a string representing the size in a human-readable format with the appropriate unit (GiB, MiB, KiB, or B).


---
### vk\_tensor\_offset<!-- {{#callable:vk_tensor_offset}} -->
Calculates the offset of a `ggml_tensor` from a base pointer.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure whose offset is to be calculated.
- **Control Flow**:
    - Checks if the `view_src` member of the `tensor` is not null.
    - If `view_src` is not null, calculates the offset using `view_src->data`.
    - If `view_src` is null, calculates the offset using `tensor->data`.
- **Output**: Returns the calculated offset as a `uint64_t` value.


---
### ggml\_vk\_wait\_for\_fence<!-- {{#callable:ggml_vk_wait_for_fence}} -->
The `ggml_vk_wait_for_fence` function waits for a Vulkan fence to signal that a GPU operation has completed, allowing the CPU to sleep during the wait.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the Vulkan device context and fence information.
- **Control Flow**:
    - Checks if there is a pending fence (`almost_ready_fence_pending`) and waits for it to be signaled using `waitForFences`.
    - Resets the pending fence after it has been signaled.
    - Enters a loop that continuously checks the status of another fence (`ctx->fence`) until it is signaled.
    - If the fence status is not ready, it logs an error and exits the program.
    - If the fence is not ready, it yields control multiple times to allow other processes to run.
- **Output**: The function does not return a value; it modifies the state of the Vulkan fences and may terminate the program on error.


---
### ggml\_vk\_create\_pipeline\_func<!-- {{#callable:ggml_vk_create_pipeline_func}} -->
Creates a Vulkan compute pipeline using specified parameters and shader module.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device used for pipeline creation.
    - `pipeline`: A reference to a `vk_pipeline` object that will hold the created pipeline.
    - `spv_size`: The size of the SPIR-V shader code in bytes.
    - `spv_data`: A pointer to the SPIR-V shader code data.
    - `entrypoint`: A string representing the entry point function name in the shader.
    - `parameter_count`: The number of parameters (descriptor bindings) for the pipeline.
    - `wg_denoms`: An array of three `uint32_t` values representing workgroup denominators.
    - `specialization_constants`: A vector of `uint32_t` values representing specialization constants for the shader.
    - `disable_robustness`: A boolean flag indicating whether to disable pipeline robustness.
    - `require_full_subgroups`: A boolean flag indicating whether full subgroup support is required.
    - `required_subgroup_size`: A `uint32_t` value specifying the required subgroup size for the pipeline.
- **Control Flow**:
    - Logs the parameters for pipeline creation using `VK_LOG_DEBUG`.
    - Asserts that `parameter_count` and each element of `wg_denoms` are greater than zero.
    - Creates a shader module from the provided SPIR-V data.
    - Sets up descriptor set layout bindings based on the parameter count.
    - Creates a descriptor set layout using the bindings and flags.
    - Creates a descriptor pool for the pipeline's parameters.
    - Creates a pipeline layout using the descriptor set layout and push constant range.
    - Prepares specialization map entries for the specialization constants.
    - Configures the pipeline shader stage creation info with the shader module and entry point.
    - Handles subgroup size requirements if specified.
    - Creates the compute pipeline and handles any errors during creation.
    - Marks the pipeline as compiled and adds it to the device's pipeline map.
    - Decrements the compile count and notifies any waiting threads.
- **Output**: The function does not return a value but modifies the `pipeline` object to include the created compute pipeline and related resources.


---
### ggml\_vk\_destroy\_pipeline<!-- {{#callable:ggml_vk_destroy_pipeline}} -->
The `ggml_vk_destroy_pipeline` function cleans up and deallocates resources associated with a Vulkan pipeline.
- **Inputs**:
    - `device`: A reference to a Vulkan device (`vk::Device`) used to perform destruction of Vulkan resources.
    - `pipeline`: A reference to a Vulkan pipeline object (`vk_pipeline`) that contains resources to be destroyed.
- **Control Flow**:
    - Logs the destruction process of the pipeline using `VK_LOG_DEBUG`.
    - Iterates over each descriptor pool in the pipeline's `descriptor_pools` and destroys them using the provided `device`.
    - Clears the `descriptor_pools` and `descriptor_sets` vectors in the pipeline.
    - Resets the `descriptor_set_idx` to 0.
    - Destroys the descriptor set layout associated with the pipeline.
    - Destroys the pipeline layout.
    - Destroys the shader module associated with the pipeline.
    - Finally, destroys the pipeline itself.
- **Output**: This function does not return a value; it performs cleanup operations to free Vulkan resources associated with the specified pipeline.


---
### ggml\_pipeline\_request\_descriptor\_sets<!-- {{#callable:ggml_pipeline_request_descriptor_sets}} -->
The `ggml_pipeline_request_descriptor_sets` function updates the descriptor set requirements for a given Vulkan pipeline.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device.
    - `pipeline`: A reference to a `vk_pipeline` object representing the Vulkan pipeline whose descriptor sets are being requested.
    - `n`: An unsigned integer representing the number of descriptor sets to request.
- **Control Flow**:
    - Logs the request for descriptor sets using the pipeline's name and the number of sets requested.
    - Increments the descriptor set requirements for the specified pipeline by the number of sets requested.
    - Checks if the pipeline has been compiled; if not, marks it as needed and indicates that the device requires compilation.
- **Output**: The function does not return a value; it modifies the state of the `device` and `pipeline` objects based on the request.


---
### ggml\_pipeline\_allocate\_descriptor\_sets<!-- {{#callable:ggml_pipeline_allocate_descriptor_sets}} -->
Allocates descriptor sets for Vulkan pipelines based on their requirements.
- **Inputs**:
    - `device`: A reference to a Vulkan device object that contains pipeline and descriptor set information.
- **Control Flow**:
    - Locks the device's mutex to ensure thread safety during descriptor set allocation.
    - Iterates over each pipeline and its corresponding descriptor set requirements.
    - Logs the pipeline name and the number of descriptor sets needed.
    - Checks if the required number of descriptor sets is already allocated; if so, continues to the next pipeline.
    - Calculates how many more descriptor sets need to be allocated and the remaining capacity in the current descriptor pool.
    - Enters a loop to allocate the required descriptor sets until all are allocated.
    - If the current pool is exhausted, creates a new descriptor pool and adds it to the pipeline's pool list.
    - Allocates the descriptor sets from the Vulkan device and appends them to the pipeline's descriptor sets.
- **Output**: The function does not return a value; it modifies the state of the pipeline by allocating the necessary descriptor sets.


---
### ggml\_pipeline\_cleanup<!-- {{#callable:ggml_pipeline_cleanup}} -->
Cleans up the specified Vulkan pipeline by resetting its descriptor set index.
- **Inputs**:
    - `pipeline`: A reference to a `vk_pipeline` object representing the Vulkan pipeline to be cleaned up.
- **Control Flow**:
    - Logs a debug message indicating the start of the cleanup process for the specified pipeline.
    - Resets the `descriptor_set_idx` of the pipeline to 0, effectively cleaning up its state.
- **Output**: The function does not return a value; it modifies the state of the input `pipeline` directly.


---
### ggml\_vk\_create\_cmd\_buffer<!-- {{#callable:ggml_vk_create_cmd_buffer}} -->
Creates and returns a Vulkan command buffer, reusing existing ones if available.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device.
    - `q`: A reference to a `vk_queue` object containing the command buffer pool and index.
- **Control Flow**:
    - Logs the function call for debugging purposes.
    - Acquires a lock on the device's mutex to ensure thread safety.
    - Checks if there are available command buffers in the queue's buffer list.
    - If available, reuses the existing command buffer and increments the index.
    - If not available, allocates a new command buffer from the Vulkan device using the specified pool.
    - Stores the newly allocated command buffer in the queue's buffer list and increments the index.
- **Output**: Returns a `vk::CommandBuffer` object that can be used for recording commands.


---
### ggml\_vk\_create\_submission<!-- {{#callable:ggml_vk_create_submission}} -->
Creates a Vulkan submission structure with command buffer and semaphore configurations.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device.
    - `q`: A reference to a `vk_queue` object representing the Vulkan queue for submission.
    - `wait_semaphores`: A vector of `vk_semaphore` objects that the submission will wait on before execution.
    - `signal_semaphores`: A vector of `vk_semaphore` objects that the submission will signal upon completion.
- **Control Flow**:
    - Logs the entry into the function using `VK_LOG_DEBUG`.
    - Creates a command buffer by calling [`ggml_vk_create_cmd_buffer`](#ggml_vk_create_cmd_buffer) with the provided `device` and `q`.
    - Moves the `wait_semaphores` and `signal_semaphores` into the `vk_submission` structure to avoid unnecessary copies.
    - Returns the populated `vk_submission` structure.
- **Output**: Returns a `vk_submission` structure containing the created command buffer and the specified wait and signal semaphores.
- **Functions called**:
    - [`ggml_vk_create_cmd_buffer`](#ggml_vk_create_cmd_buffer)


---
### ggml\_vk\_submit<!-- {{#callable:ggml_vk_submit}} -->
Submits a series of command sequences to a Vulkan queue, handling synchronization with semaphores.
- **Inputs**:
    - `ctx`: A reference to a `vk_context` object that contains the Vulkan queue and command sequences to be submitted.
    - `fence`: A `vk::Fence` object used to synchronize the submission; if provided, it will be signaled when the submission is complete.
- **Control Flow**:
    - Checks if the `seqs` vector in `ctx` is empty; if so, submits an empty command to the queue if a `fence` is provided and returns.
    - Logs the submission details for debugging purposes.
    - Calculates the total number of submissions to reserve space in various vectors to avoid reallocations.
    - Iterates over each sequence in `ctx->seqs`, and for each submission, collects wait and signal semaphore information.
    - Constructs `vk::SubmitInfo` objects for each submission, including semaphore synchronization details.
    - Submits the constructed `submit_infos` to the Vulkan queue along with the provided `fence`.
    - Clears the `seqs` vector in `ctx` after submission.
- **Output**: The function does not return a value; it performs a submission operation to the Vulkan queue and manages synchronization through the provided fence.


---
### ggml\_vk\_find\_queue\_family\_index<!-- {{#callable:ggml_vk_find_queue_family_index}} -->
The `ggml_vk_find_queue_family_index` function searches for a suitable queue family index based on specified requirements and constraints.
- **Inputs**:
    - `queue_family_props`: A vector of `vk::QueueFamilyProperties` that contains the properties of available queue families.
    - `required`: A `vk::QueueFlags` bitmask indicating the required queue capabilities.
    - `avoid`: A `vk::QueueFlags` bitmask indicating the queue capabilities to avoid.
    - `compute_index`: An integer index of a compute queue to avoid using, or -1 if not applicable.
    - `min_num_queues`: A minimum number of queues that must be available in the selected queue family.
- **Control Flow**:
    - The function first logs its invocation for debugging purposes.
    - It retrieves the size of the `queue_family_props` vector.
    - It iterates through the queue families to find one that meets the criteria of having enough queues, matching required flags, and not matching avoid flags.
    - If no suitable queue is found, it attempts to find a queue that only matches the required flags, ignoring the avoid flags.
    - If still unsuccessful, it looks for any queue that meets the minimum queue count and matches the required flags, regardless of the compute index.
    - If no queues are found yet, it relaxes the minimum queue count requirement and looks for any queue that matches the required flags.
    - If a valid `compute_index` is provided, it returns that index as a fallback.
    - If all attempts fail, it logs an error message and aborts the program.
- **Output**: The function returns a `uint32_t` representing the index of a suitable queue family, or aborts the program if no suitable index is found.


---
### ggml\_vk\_create\_queue<!-- {{#callable:ggml_vk_create_queue}} -->
Creates a Vulkan queue and associated command pool for a specified device.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device.
    - `q`: A reference to a `vk_queue` object that will be initialized with the created queue and command pool.
    - `queue_family_index`: An unsigned integer specifying the index of the queue family to use.
    - `queue_index`: An unsigned integer specifying the index of the queue within the specified queue family.
    - `stage_flags`: A rvalue reference to `vk::PipelineStageFlags` indicating the pipeline stages that the queue will be used for.
    - `transfer_only`: A boolean indicating whether the queue is intended for transfer operations only.
- **Control Flow**:
    - Logs the entry into the function for debugging purposes.
    - Acquires a lock on the device's mutex to ensure thread safety during queue creation.
    - Sets the `queue_family_index` and `transfer_only` properties of the `q` object.
    - Creates a command pool with transient properties for the specified queue family index.
    - Retrieves the Vulkan queue from the device using the provided queue family and index.
    - Assigns the retrieved queue and stage flags to the `q` object.
- **Output**: The function does not return a value; instead, it initializes the provided `vk_queue` object with the created Vulkan queue and command pool.


---
### ggml\_vk\_create\_context<!-- {{#callable:ggml_vk_create_context}} -->
Creates a Vulkan context and associates it with a given queue.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds Vulkan context information.
    - `q`: A reference to a `vk_queue` object that represents the Vulkan queue to be associated with the context.
- **Control Flow**:
    - A new `vk_context` is created using `std::make_shared` to allocate memory for the context structure.
    - A debug log is generated to indicate the creation of the context, including its address.
    - The newly created context is added to the `contexts` vector of the provided `ctx` structure.
    - The queue reference `q` is assigned to the `q` member of the newly created context.
    - The function returns the newly created Vulkan context.
- **Output**: Returns a shared pointer to the newly created `vk_context`.


---
### ggml\_vk\_create\_temporary\_context<!-- {{#callable:ggml_vk_create_temporary_context}} -->
Creates a temporary Vulkan context associated with a given queue.
- **Inputs**:
    - `q`: A reference to a `vk_queue` object that the temporary context will be associated with.
- **Control Flow**:
    - A new `vk_context` is created using `std::make_shared` to allocate memory for a `vk_context_struct`.
    - A debug log is generated to indicate the creation of the temporary context, including its address.
    - The queue reference `q` is assigned to the `q` member of the newly created context.
    - The function returns the newly created `vk_context`.
- **Output**: Returns a shared pointer to a `vk_context` that is initialized with the provided queue.


---
### ggml\_vk\_create\_binary\_semaphore<!-- {{#callable:ggml_vk_create_binary_semaphore}} -->
Creates a binary semaphore in Vulkan and stores it in the provided context.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the Vulkan device and a collection of semaphores.
- **Control Flow**:
    - Logs a debug message indicating the creation of a binary semaphore.
    - Creates a `vk::SemaphoreTypeCreateInfo` object specifying the semaphore type as binary.
    - Initializes a `vk::SemaphoreCreateInfo` object and sets its `pNext` pointer to the semaphore type create info.
    - Calls the Vulkan API to create a semaphore using the device associated with the context and the create info.
    - Stores the created semaphore in the context's semaphore collection along with an initial value of 0.
    - Returns a pointer to the newly created semaphore stored in the context's semaphore collection.
- **Output**: Returns a pointer to the created `vk_semaphore` object stored in the context's semaphore collection.


---
### ggml\_vk\_create\_timeline\_semaphore<!-- {{#callable:ggml_vk_create_timeline_semaphore}} -->
Creates a timeline semaphore in Vulkan if the current index exceeds the existing semaphores.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains Vulkan context information and semaphore management.
- **Control Flow**:
    - Logs the function call for debugging purposes.
    - Checks if the current semaphore index exceeds the size of the existing timeline semaphores.
    - If the index is exceeded, creates a new timeline semaphore using Vulkan API and adds it to the context's semaphore list.
    - Returns a pointer to the newly created or existing timeline semaphore and increments the semaphore index.
- **Output**: Returns a pointer to a `vk_semaphore` representing the created or retrieved timeline semaphore.


---
### ggml\_vk\_create\_event<!-- {{#callable:ggml_vk_create_event}} -->
Creates a Vulkan event and returns it, managing the event index within the provided context.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains Vulkan context information, including the event index and a collection of Vulkan events.
- **Control Flow**:
    - Checks if the current event index (`event_idx`) is greater than or equal to the size of the events vector in the context.
    - If the index is out of bounds, a new Vulkan event is created using the device's `createEvent` method and added to the events vector.
    - The event at the current index is returned, and the index is incremented for the next call.
- **Output**: Returns a `vk::Event` object representing the Vulkan event created or retrieved from the context.


---
### ggml\_vk\_queue\_cleanup<!-- {{#callable:ggml_vk_queue_cleanup}} -->
Cleans up the Vulkan queue by resetting its command pool.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device.
    - `q`: A reference to a `vk_queue` object representing the Vulkan queue to be cleaned up.
- **Control Flow**:
    - Logs the entry into the `ggml_vk_queue_cleanup` function for debugging purposes.
    - Acquires a lock on the device's mutex to ensure thread safety during cleanup.
    - Resets the command pool associated with the queue, which requires that all command buffers are completed.
    - Resets the command buffer index of the queue to zero.
- **Output**: The function does not return a value; it performs cleanup operations on the Vulkan queue.


---
### find\_properties<!-- {{#callable:find_properties}} -->
The `find_properties` function identifies a suitable memory type index based on specified memory requirements and property flags.
- **Inputs**:
    - `mem_props`: A pointer to a `vk::PhysicalDeviceMemoryProperties` structure that contains information about the memory types and heaps available on a physical device.
    - `mem_req`: A pointer to a `vk::MemoryRequirements` structure that specifies the memory requirements for a resource.
    - `flags`: A `vk::MemoryPropertyFlags` value that indicates the desired memory properties.
- **Control Flow**:
    - The function iterates over each memory type available in `mem_props` using a for loop.
    - For each memory type, it checks if the memory type is compatible with the requested memory requirements and properties.
    - If a suitable memory type is found, the function returns its index.
    - If no suitable memory type is found after checking all types, the function returns `UINT32_MAX`.
- **Output**: The function returns the index of the first memory type that meets the specified requirements, or `UINT32_MAX` if no suitable memory type is found.


---
### ggml\_vk\_create\_buffer<!-- {{#callable:ggml_vk_create_buffer}} -->
Creates a Vulkan buffer with specified size and memory properties, handling allocation and potential fallback.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device used for buffer creation.
    - `size`: A `size_t` value indicating the size of the buffer to be created.
    - `req_flags`: A `vk::MemoryPropertyFlags` value specifying the required memory properties for the buffer.
    - `fallback_flags`: An optional `vk::MemoryPropertyFlags` value specifying fallback memory properties if the required ones are not available.
- **Control Flow**:
    - Logs the function call with device name, size, and memory property flags.
    - Checks if the requested buffer size exceeds the device's maximum memory allocation size, throwing an error if it does.
    - Locks the device's mutex to ensure thread safety during buffer creation.
    - Creates a shared pointer for the buffer structure and initializes its size to zero if the requested size is zero.
    - Sets up the `vk::BufferCreateInfo` structure with the specified size and usage flags.
    - Creates the Vulkan buffer using the device's `createBuffer` method.
    - Retrieves memory requirements for the created buffer.
    - Finds a suitable memory type index based on the required flags, and if not found, attempts to use fallback flags.
    - If no suitable memory type is found, destroys the buffer and throws an error.
    - Attempts to allocate memory for the buffer, handling potential allocation errors by retrying with fallback flags if necessary.
    - Maps the memory to the buffer if the memory property flags indicate it is host visible.
    - Binds the allocated memory to the created buffer.
    - Logs the memory allocation if debugging is enabled.
- **Output**: Returns a `vk_buffer` object that encapsulates the created Vulkan buffer, its size, and associated memory.
- **Functions called**:
    - [`find_properties`](#find_properties)


---
### ggml\_vk\_create\_buffer\_check<!-- {{#callable:ggml_vk_create_buffer_check}} -->
The `ggml_vk_create_buffer_check` function attempts to create a Vulkan buffer and handles any memory allocation errors.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device used for buffer creation.
    - `size`: A `size_t` value indicating the size of the buffer to be created.
    - `req_flags`: A `vk::MemoryPropertyFlags` value specifying the required memory properties for the buffer.
    - `fallback_flags`: An optional `vk::MemoryPropertyFlags` value specifying fallback memory properties, defaulting to zero.
- **Control Flow**:
    - The function attempts to call [`ggml_vk_create_buffer`](#ggml_vk_create_buffer) with the provided parameters to create a Vulkan buffer.
    - If the buffer creation fails, it catches a `vk::SystemError` exception.
    - Upon catching the exception, it logs an error message indicating the failure and the size of the requested buffer.
    - Finally, it rethrows the caught exception to propagate the error.
- **Output**: Returns a `vk_buffer` object representing the created Vulkan buffer if successful; otherwise, it throws a `vk::SystemError` exception.
- **Functions called**:
    - [`ggml_vk_create_buffer`](#ggml_vk_create_buffer)


---
### ggml\_vk\_create\_buffer\_device<!-- {{#callable:ggml_vk_create_buffer_device}} -->
Creates a Vulkan buffer on a specified device with a given size, considering memory properties.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device on which the buffer will be created.
    - `size`: A `size_t` value indicating the size of the buffer to be created.
- **Control Flow**:
    - The function begins by declaring a `vk_buffer` variable named `buf` to hold the created buffer.
    - It checks if the device prefers host memory; if true, it creates a buffer with host-visible and coherent memory properties.
    - If the device is UMA (Unified Memory Architecture), it falls back to creating a buffer with device-local memory and host-visible properties.
    - If neither condition is met, it attempts to create a buffer with a combination of device-local and host-visible properties, using rebar if available.
    - If any memory allocation fails, it catches a `vk::SystemError`, logs an error message, and rethrows the exception.
- **Output**: Returns a `vk_buffer` object representing the created buffer on the specified Vulkan device.
- **Functions called**:
    - [`ggml_vk_create_buffer`](#ggml_vk_create_buffer)


---
### ggml\_vk\_destroy\_buffer<!-- {{#callable:ggml_vk_destroy_buffer}} -->
This function safely destroys a Vulkan buffer by checking for null and logging deallocation if debugging is enabled.
- **Inputs**:
    - `buf`: A reference to a `vk_buffer` object that represents the Vulkan buffer to be destroyed.
- **Control Flow**:
    - The function first checks if the `buf` is null; if it is, the function returns immediately without doing anything.
    - If debugging is enabled (i.e., `GGML_VULKAN_MEMORY_DEBUG` is defined), it checks if the `device` associated with the buffer is not null and logs the deallocation of the buffer using the memory logger.
    - Finally, the function calls `reset()` on the `buf`, which effectively releases the resources associated with the Vulkan buffer.
- **Output**: The function does not return a value; it modifies the state of the `buf` by releasing its resources.


---
### ggml\_vk\_subbuffer<!-- {{#callable:ggml_vk_subbuffer}} -->
Creates a `vk_subbuffer` from a given `vk_buffer`.
- **Inputs**:
    - `buf`: A reference to a `vk_buffer` object that serves as the source for creating the subbuffer.
- **Control Flow**:
    - Directly returns a `vk_subbuffer` initialized with the provided `vk_buffer`, an offset of 0, and a size of `VK_WHOLE_SIZE`.
- **Output**: Returns a `vk_subbuffer` structure that represents a subregion of the specified `vk_buffer`, starting from the beginning and extending to the full size of the buffer.


---
### ggml\_vk\_sync\_buffers<!-- {{#callable:ggml_vk_sync_buffers}} -->
Synchronizes Vulkan buffer operations by applying a pipeline barrier based on the context's queue settings.
- **Inputs**:
    - `ctx`: A reference to a `vk_context` object that contains information about the Vulkan context, including the command queue and buffer state.
- **Control Flow**:
    - Logs the entry into the function for debugging purposes.
    - Checks if the command queue is set to transfer only mode.
    - Calls the `pipelineBarrier` method on the buffer object to synchronize access to the buffers, using appropriate access flags based on the transfer queue status.
- **Output**: The function does not return a value; it performs synchronization of Vulkan buffer operations.


---
### ggml\_vk\_wait\_events<!-- {{#callable:ggml_vk_wait_events}} -->
The `ggml_vk_wait_events` function waits for a set of Vulkan events to be signaled.
- **Inputs**:
    - `ctx`: A reference to a `vk_context` object that contains Vulkan context information.
    - `events`: A vector of Vulkan `Event` objects that the function will wait on.
- **Control Flow**:
    - The function logs a debug message indicating its invocation.
    - It checks if the `events` vector is empty; if it is, the function returns immediately without further action.
    - If the `events` vector is not empty, it calls the `waitEvents` method on the Vulkan buffer, passing the events and relevant stage flags from the context.
- **Output**: The function does not return a value; it performs an operation to wait for the specified Vulkan events.


---
### get\_fa\_num\_small\_rows<!-- {{#callable:get_fa_num_small_rows}} -->
The `get_fa_num_small_rows` function returns the number of small rows based on the provided `FaCodePath`.
- **Inputs**:
    - `path`: An enumeration value of type `FaCodePath` that determines which number of small rows to return.
- **Control Flow**:
    - The function checks if the input `path` is equal to `FA_COOPMAT2`.
    - If the condition is true, it returns the value of `flash_attention_num_small_rows`.
    - If the condition is false, it returns the value of `scalar_flash_attention_num_small_rows`.
- **Output**: The function outputs a `uint32_t` value representing the number of small rows corresponding to the specified `FaCodePath`.


---
### fa\_rows\_cols<!-- {{#callable:fa_rows_cols}} -->
The `fa_rows_cols` function determines the number of rows and columns for a specific attention mechanism based on the provided parameters.
- **Inputs**:
    - `path`: An enumeration value of type `FaCodePath` that specifies the type of attention mechanism.
    - `D`: A `uint32_t` value representing a dimension parameter that may influence the output.
    - `clamp`: A `uint32_t` value that is unused in the function, likely intended for future use or compatibility.
    - `type`: A variable of type `ggml_type` that indicates the data type, which can affect the output based on quantization.
    - `small_rows`: A boolean flag indicating whether to use a configuration optimized for small rows.
- **Control Flow**:
    - The function first checks if the `path` is equal to `FA_SCALAR` and returns different row and column values based on the `small_rows` flag.
    - If the `path` is `FA_COOPMAT1`, it similarly returns values based on the `small_rows` flag, but uses different constants for rows.
    - If neither of the first two conditions are met and `small_rows` is true, it calls [`get_fa_num_small_rows`](#get_fa_num_small_rows) with `FA_COOPMAT2` to determine the number of rows, returning a fixed column size.
    - If `small_rows` is false, it checks if the `type` is quantized or if `D` equals 256, returning a specific row and column configuration.
    - If none of the above conditions are satisfied, it defaults to returning a standard configuration of 64 rows and 64 columns.
- **Output**: The function returns a `std::array<uint32_t, 2>` containing the number of rows and columns based on the input parameters.
- **Functions called**:
    - [`get_fa_num_small_rows`](#get_fa_num_small_rows)
    - [`ggml_is_quantized`](../ggml.c.driver.md#ggml_is_quantized)


---
### ggml\_vk\_matmul\_shmem\_support<!-- {{#callable:ggml_vk_matmul_shmem_support}} -->
The `ggml_vk_matmul_shmem_support` function checks if the shared memory size required for matrix multiplication is supported by the given Vulkan device.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device being queried.
    - `warptile`: A vector of `uint32_t` values that define the configuration of the warptile for matrix multiplication.
    - `mul_mat_id`: A boolean indicating whether to include matrix ID in the multiplication.
    - `src0_type`: An enumeration value of type `ggml_type` that specifies the data type of the first source matrix.
- **Control Flow**:
    - The function initializes a variable `lut_size` to store the lookup table size based on the `src0_type` input.
    - A switch statement determines the `lut_size` based on the value of `src0_type`, with specific cases for different types.
    - The function calculates `bank_conflict_offset`, `type_size`, `warps`, `load_bufs`, `mmid_row_ids`, and `coopmat_stage` based on the device properties and `warptile` configuration.
    - The total required shared memory size is computed by summing `load_bufs`, `mmid_row_ids`, `coopmat_stage`, and `lut_size`.
    - A boolean `supported` is determined by checking if the total size is less than or equal to the device's maximum compute shared memory size.
    - A debug log statement outputs the parameters and the support status before returning the result.
- **Output**: The function returns a boolean value indicating whether the required shared memory size for the matrix multiplication is supported by the Vulkan device.
- **Functions called**:
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)


---
### get\_subgroup\_size<!-- {{#callable:get_subgroup_size}} -->
The `get_subgroup_size` function retrieves the subgroup size for a specified pipeline name and device architecture.
- **Inputs**:
    - `pipeline_name`: A constant reference to a string representing the name of the pipeline for which the subgroup size is requested.
    - `arch`: A constant reference to a `vk_device_architecture` object representing the architecture of the device.
- **Control Flow**:
    - The function iterates over a collection of `gpu_pipeline_configs` to find a matching architecture.
    - If a matching architecture is found, it checks if the specified `pipeline_name` exists in the corresponding pipelines.
    - If the pipeline is found, its associated subgroup size is returned immediately.
    - If the pipeline is not found, the function creates a sorted list of pipelines based on their names' lengths in descending order.
    - It then iterates through the sorted list to find the first pipeline name that is a substring of the specified `pipeline_name` and returns its subgroup size.
    - If no suitable pipeline is found, the function returns the default subgroup size for the configuration.
    - If no matching architecture is found in the entire collection, the function returns 0.
- **Output**: The function returns a `uint32_t` value representing the subgroup size for the specified pipeline, or 0 if no matching configuration is found.


---
### ggml\_vk\_get\_device<!-- {{#callable:ggml_vk_get_device}} -->
Retrieves or initializes a Vulkan device based on the specified index.
- **Inputs**:
    - `idx`: An index representing the desired Vulkan device to retrieve or initialize.
- **Control Flow**:
    - Logs the function call with the provided index.
    - Checks if the Vulkan device at the specified index is already initialized.
    - If not initialized, creates a new `vk_device` instance and initializes it.
    - Retrieves the physical devices available in the Vulkan instance.
    - Validates the device index against the available physical devices and throws an error if invalid.
    - Initializes various properties and features of the device based on the physical device's capabilities.
    - Sets up device extensions and features based on the device's properties and environment variables.
    - Creates the Vulkan device and associated queues.
    - Returns the initialized or retrieved Vulkan device.
- **Output**: Returns a shared pointer to the initialized or existing `vk_device`.
- **Functions called**:
    - [`get_device_architecture`](#get_device_architecture)
    - [`ggml_vk_khr_cooperative_matrix_support`](#ggml_vk_khr_cooperative_matrix_support)
    - [`ggml_vk_find_queue_family_index`](#ggml_vk_find_queue_family_index)
    - [`ggml_vk_create_queue`](#ggml_vk_create_queue)
    - [`ggml_backend_vk_reg`](#ggml_backend_vk_reg)


---
### ggml\_vk\_instance\_init<!-- {{#callable:ggml_vk_instance_init}} -->
Initializes the Vulkan instance and sets up the necessary configurations for GPU usage.
- **Inputs**: None
- **Control Flow**:
    - Checks if the Vulkan instance has already been initialized; if so, it returns immediately.
    - Logs the initialization process and retrieves the Vulkan API version.
    - Validates that the API version is at least 1.2, aborting if it is not.
    - Creates an `ApplicationInfo` structure and enumerates available instance extensions.
    - Checks for the availability of validation extensions and sets up the corresponding layers and extensions.
    - Creates an `InstanceCreateInfo` structure with the application info, layers, and extensions.
    - If on macOS, adds portability enumeration flags if the extension is available.
    - Enables validation features if validation extensions are present.
    - Creates the Vulkan instance and marks it as initialized.
    - Checks for the environment variable `GGML_VK_VISIBLE_DEVICES` to determine which devices to use.
    - If the variable is set, parses the device indices; otherwise, enumerates all physical devices.
    - Prioritizes dedicated GPUs and handles potential duplicates based on driver IDs.
    - Logs the number of Vulkan devices found and prints information for each device.
- **Output**: The function does not return a value but initializes the Vulkan instance and sets up the device indices for further GPU operations.
- **Functions called**:
    - [`ggml_vk_instance_validation_ext_available`](#ggml_vk_instance_validation_ext_available)
    - [`ggml_vk_instance_portability_enumeration_ext_available`](#ggml_vk_instance_portability_enumeration_ext_available)


---
### ggml\_vk\_init<!-- {{#callable:ggml_vk_init}} -->
Initializes a Vulkan backend context for a specified device index.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context information.
    - `idx`: A size_t index representing the specific device to initialize within the Vulkan instance.
- **Control Flow**:
    - Logs the initialization call with the context name and index.
    - Calls `ggml_vk_instance_init()` to initialize the Vulkan instance.
    - Asserts that the provided index is valid and within the range of available device indices.
    - Sets the context name based on a predefined constant and the device index.
    - Retrieves the Vulkan device corresponding to the provided index and assigns it to the context.
    - Initializes semaphore and event indices to zero.
    - Sets preallocation sizes for x, y, and split_k to zero.
    - Creates fences for synchronization and assigns them to the context.
    - Optionally checks environment variables to configure Vulkan behavior based on runtime settings.
- **Output**: This function does not return a value; it initializes the provided `ggml_backend_vk_context` structure with Vulkan device and synchronization settings.
- **Functions called**:
    - [`ggml_vk_instance_init`](#ggml_vk_instance_init)
    - [`ggml_vk_get_device`](#ggml_vk_get_device)


---
### ggml\_vk\_get\_to\_fp16<!-- {{#callable:ggml_vk_get_to_fp16}} -->
The `ggml_vk_get_to_fp16` function retrieves a Vulkan pipeline for converting various tensor types to FP16 format.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the Vulkan device context.
    - `type`: An enumeration value of type `ggml_type` that specifies the tensor type to be converted.
- **Control Flow**:
    - The function logs a debug message indicating its invocation.
    - It checks the `type` against a list of valid tensor types that can be processed.
    - If the `type` is not valid, the function returns a null pointer.
    - If the `type` is valid, it retrieves and returns the corresponding Vulkan pipeline from the context's device.
- **Output**: The function returns a pointer to a Vulkan pipeline for the specified tensor type, or null if the type is invalid.


---
### ggml\_vk\_get\_mul\_mat\_mat\_pipeline<!-- {{#callable:ggml_vk_get_mul_mat_mat_pipeline}} -->
Retrieves the appropriate matrix multiplication pipeline based on the types of the source matrices and the precision.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains device-specific information and resources.
    - `src0_type`: The type of the first source matrix, represented as a `ggml_type` enumeration.
    - `src1_type`: The type of the second source matrix, represented as a `ggml_type` enumeration.
    - `prec`: The precision level for the operation, represented as a `ggml_prec` enumeration.
- **Control Flow**:
    - Logs the function call with the types of the source matrices and precision.
    - Checks combinations of `src0_type` and `src1_type` to return the corresponding matrix multiplication pipeline from the device context.
    - Handles special cases for quantized types and checks device capabilities like `coopmat_support` and `coopmat2`.
    - Uses a switch statement to validate `src0_type` against a set of allowed quantized types.
    - Returns `nullptr` if no valid pipeline is found based on the input types and device capabilities.
- **Output**: Returns a `vk_matmul_pipeline` that corresponds to the specified matrix types and precision, or `nullptr` if no valid pipeline is available.
- **Functions called**:
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)


---
### ggml\_vk\_get\_dequantize\_mul\_mat\_vec<!-- {{#callable:ggml_vk_get_dequantize_mul_mat_vec}} -->
The `ggml_vk_get_dequantize_mul_mat_vec` function retrieves a Vulkan pipeline for dequantizing and multiplying a matrix with a vector based on specified types and column count.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the Vulkan device context.
    - `a_type`: The data type of the matrix, represented as a `ggml_type` enumeration.
    - `b_type`: The data type of the vector, which must be either `GGML_TYPE_F32` or `GGML_TYPE_F16`.
    - `num_cols`: An unsigned integer representing the number of columns in the matrix, constrained between 1 and `mul_mat_vec_max_cols`.
- **Control Flow**:
    - Logs the function call for debugging purposes.
    - Asserts that `b_type` is either `GGML_TYPE_F32` or `GGML_TYPE_F16`.
    - Asserts that `num_cols` is within the valid range (1 to `mul_mat_vec_max_cols`).
    - Checks the value of `a_type` against a list of valid types; if it is invalid, the function returns nullptr.
    - Returns the appropriate Vulkan pipeline based on the value of `b_type` and the specified `num_cols`.
- **Output**: Returns a `vk_pipeline` object corresponding to the dequantization and multiplication operation, or nullptr if the input types are invalid.


---
### ggml\_vk\_get\_mul\_mat\_mat\_id\_pipeline<!-- {{#callable:ggml_vk_get_mul_mat_mat_id_pipeline}} -->
Retrieves the appropriate matrix multiplication pipeline based on the input types and precision.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that contains device information.
    - `src0_type`: The type of the first source matrix, represented as a `ggml_type`.
    - `src1_type`: The type of the second source matrix, represented as a `ggml_type`.
    - `prec`: The precision level, represented as a `ggml_prec`.
- **Control Flow**:
    - Logs the function call for debugging purposes.
    - Checks if both source matrix types are `GGML_TYPE_F32` and returns the corresponding pipeline if true.
    - Checks if both source matrix types are `GGML_TYPE_BF16` and returns the corresponding pipeline if true.
    - Evaluates the precision and device capabilities to determine the appropriate pipeline for `GGML_TYPE_F16` and `GGML_TYPE_F32` combinations.
    - Asserts that the second source type is either `GGML_TYPE_F32` or compatible with cooperative matrix multiplication.
    - Validates the first source type against a set of quantized types, returning `nullptr` if it does not match.
    - Returns the appropriate dequantized matrix multiplication pipeline based on the first source type and device capabilities.
- **Output**: Returns a `vk_matmul_pipeline` that corresponds to the input types and precision, or `nullptr` if the input types are invalid.


---
### ggml\_vk\_get\_dequantize\_mul\_mat\_vec\_id<!-- {{#callable:ggml_vk_get_dequantize_mul_mat_vec_id}} -->
Retrieves a Vulkan pipeline for dequantizing and multiplying a matrix with a vector based on the input types.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the Vulkan device context.
    - `a_type`: The type of the first input (matrix) which determines the dequantization method.
    - `b_type`: The type of the second input (vector), which must be `GGML_TYPE_F32`.
- **Control Flow**:
    - Logs a debug message indicating the function has been called.
    - Asserts that the `b_type` is `GGML_TYPE_F32`, ensuring the second input is of the correct type.
    - Checks the `a_type` against a list of valid types; if it is not valid, the function returns a null pointer.
    - If `a_type` is valid, retrieves and returns the corresponding Vulkan pipeline from the context's device.
- **Output**: Returns a Vulkan pipeline for dequantizing and multiplying a matrix with a vector if the input types are valid; otherwise, returns a null pointer.


---
### ggml\_vk\_pool\_malloc<!-- {{#callable:ggml_vk_pool_malloc}} -->
Allocates a Vulkan buffer from a pool, either by reusing an existing buffer or creating a new one if necessary.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the Vulkan buffer pool and device.
    - `size`: The size in bytes of the buffer to allocate.
- **Control Flow**:
    - Logs the allocation request and memory usage.
    - Initializes variables to track the best fitting and worst fitting buffers in the pool.
    - Iterates through the buffer pool to find the smallest buffer that can accommodate the requested size.
    - If a suitable buffer is found, it resets that buffer and returns it.
    - If no suitable buffer is found, it identifies the largest buffer and destroys it to free up memory.
    - Creates a new buffer of the requested size if no existing buffer can be reused.
- **Output**: Returns a `vk_buffer` that is either an existing buffer from the pool or a newly created buffer of the specified size.
- **Functions called**:
    - [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer)
    - [`ggml_vk_create_buffer_device`](#ggml_vk_create_buffer_device)


---
### ggml\_vk\_pool\_free<!-- {{#callable:ggml_vk_pool_free}} -->
The `ggml_vk_pool_free` function attempts to free a Vulkan buffer by adding it to a buffer pool or destroying it if the pool is full.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the buffer pool.
    - `buffer`: A reference to a `vk_buffer` that represents the Vulkan buffer to be freed.
- **Control Flow**:
    - The function logs the size of the buffer being freed for debugging purposes.
    - It iterates through the `buffer_pool` array in the `ctx` structure to find an empty slot (a `nullptr`).
    - If an empty slot is found, the function assigns the `buffer` to that slot and returns.
    - If no empty slots are available after checking all buffers, a warning message is printed to the standard error output.
    - Finally, the function calls [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer) to destroy the `buffer`.
- **Output**: The function does not return a value; it either adds the buffer to the pool or destroys it if the pool is full.
- **Functions called**:
    - [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer)


---
### ggml\_vk\_create\_buffer\_temp<!-- {{#callable:ggml_vk_create_buffer_temp}} -->
The `ggml_vk_create_buffer_temp` function creates or retrieves a temporary Vulkan buffer of a specified size.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the context for Vulkan operations, including a list of temporary buffers.
    - `size`: A `size_t` value representing the required size of the temporary buffer to be created or retrieved.
- **Control Flow**:
    - The function iterates over the existing temporary buffers in the `ctx->gc.temp_buffers` list to check if any buffer has a size that is greater than or equal to the requested size.
    - If a suitable buffer is found, it is returned immediately.
    - If no suitable buffer is found, a log message is generated to indicate the creation of a new buffer.
    - A new buffer is allocated using [`ggml_vk_pool_malloc`](#ggml_vk_pool_malloc), and this new buffer is added to the `ctx->gc.temp_buffers` list before being returned.
- **Output**: The function returns a `vk_buffer` which is either an existing buffer with sufficient size or a newly created buffer of the specified size.
- **Functions called**:
    - [`ggml_vk_pool_malloc`](#ggml_vk_pool_malloc)


---
### ggml\_vk\_host\_malloc<!-- {{#callable:ggml_vk_host_malloc}} -->
Allocates a memory buffer on the host using Vulkan and returns a pointer to the allocated memory.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device used for memory allocation.
    - `size`: A `size_t` value indicating the size in bytes of the memory to be allocated.
- **Control Flow**:
    - Logs the memory allocation request with the specified size.
    - Calls [`ggml_vk_create_buffer`](#ggml_vk_create_buffer) to create a Vulkan buffer with specified memory properties.
    - Checks if the allocated buffer's memory is host visible; if not, logs a warning, frees the allocated memory, and returns nullptr.
    - If the allocation is successful, stores the buffer pointer, size, and buffer object in the `pinned_memory` vector of the device.
    - Returns the pointer to the allocated memory.
- **Output**: Returns a pointer to the allocated memory if successful, or nullptr if the allocation fails.
- **Functions called**:
    - [`ggml_vk_create_buffer`](#ggml_vk_create_buffer)


---
### ggml\_vk\_host\_free<!-- {{#callable:ggml_vk_host_free}} -->
The `ggml_vk_host_free` function deallocates a memory block from a Vulkan device's pinned memory if it is found.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device managing the memory.
    - `ptr`: A pointer to the memory block that needs to be freed.
- **Control Flow**:
    - The function first checks if the input pointer `ptr` is null; if it is, the function returns immediately without doing anything.
    - It logs the memory free operation using `VK_LOG_MEMORY` with the address of the pointer.
    - The function iterates over the `pinned_memory` vector of the `device` to find the memory block that contains the pointer `ptr`.
    - If a matching memory block is found, it retrieves the associated `vk_buffer` and its index.
    - If no matching buffer is found, a warning is printed to `stderr`, and the function returns.
    - If a buffer is found, it calls [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer) to free the Vulkan buffer associated with the memory.
    - Finally, it removes the entry from the `pinned_memory` vector using the stored index.
- **Output**: The function does not return a value; it performs memory deallocation and modifies the state of the `device` by removing the freed memory from its `pinned_memory` list.
- **Functions called**:
    - [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer)


---
### ggml\_vk\_host\_get<!-- {{#callable:ggml_vk_host_get}} -->
The `ggml_vk_host_get` function retrieves a Vulkan buffer and its offset based on a given pointer from a device's pinned memory.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device.
    - `ptr`: A pointer to the memory location for which the corresponding Vulkan buffer is to be found.
    - `buf`: A reference to a `vk_buffer` that will be set to the found buffer.
    - `buf_offset`: A reference to a `size_t` that will be set to the offset of the pointer within the buffer.
- **Control Flow**:
    - The function initializes `buf` to `nullptr` and `buf_offset` to `0`.
    - It iterates over each entry in the `pinned_memory` vector of the `device`.
    - For each entry, it retrieves the address and size of the pinned memory block.
    - It checks if the provided pointer `ptr` falls within the range of the current pinned memory block.
    - If a match is found, it sets `buf` to the corresponding Vulkan buffer and calculates the offset of `ptr` within that buffer, then breaks the loop.
- **Output**: The function does not return a value; instead, it modifies the `buf` and `buf_offset` references to provide the Vulkan buffer and the offset of the pointer within that buffer.


---
### ggml\_vk\_begin\_submission<!-- {{#callable:ggml_vk_begin_submission}} -->
Begins a Vulkan command buffer submission for a specified device and queue.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device.
    - `q`: A reference to a `vk_queue` object representing the Vulkan queue to which the command buffer will be submitted.
    - `one_time`: A boolean flag indicating whether the command buffer should be set for one-time submission.
- **Control Flow**:
    - A `vk_submission` object `s` is initialized.
    - A command buffer is created by calling [`ggml_vk_create_cmd_buffer`](#ggml_vk_create_cmd_buffer) with the provided `device` and `q`.
    - If `one_time` is true, the command buffer is begun with the `eOneTimeSubmit` usage flag; otherwise, it is begun with default usage flags.
    - The `vk_submission` object `s` is returned.
- **Output**: Returns a `vk_submission` object containing the initialized command buffer ready for submission.
- **Functions called**:
    - [`ggml_vk_create_cmd_buffer`](#ggml_vk_create_cmd_buffer)


---
### push\_constant\_size<!-- {{#callable:push_constant_size}} -->
Calculates the total size in bytes of a constant-sized array of a given type.
- **Inputs**:
    - `t`: A constant reference to a `std::array` of type `T` and size `N`.
- **Control Flow**:
    - The function begins by marking the input parameter `t` as unused to avoid compiler warnings.
    - It then calculates the size of the array by multiplying the size of the type `T` by the number of elements `N`.
- **Output**: Returns the total size in bytes of the array, which is computed as `sizeof(T) * N`.


---
### push\_constant\_data<!-- {{#callable:push_constant_data}} -->
This function returns a pointer to the underlying data of a constant array.
- **Inputs**:
    - `t`: A constant reference to a `std::array` of type `T` with size `N`.
- **Control Flow**:
    - The function takes a single input parameter, `t`, which is a constant reference to a `std::array`.
    - It directly accesses the underlying data of the array using the `data()` member function.
- **Output**: Returns a pointer to the first element of the array `t`, which is of type `const T*`.


---
### ggml\_vk\_dispatch\_pipeline<!-- {{#callable:ggml_vk_dispatch_pipeline}} -->
Dispatches a compute pipeline in Vulkan with specified descriptor buffer information and push constants.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan sub-context for the dispatch.
    - `pipeline`: A reference to a `vk_pipeline` object that represents the compute pipeline to be dispatched.
    - `descriptor_buffer_infos`: An initializer list of `vk::DescriptorBufferInfo` objects that provide information about the buffers to be used in the pipeline.
    - `push_constants`: A constant reference to a template type `T` that contains the push constants to be sent to the shader.
    - `elements`: An array of three `uint32_t` values representing the dimensions of the workgroups to be dispatched.
- **Control Flow**:
    - Calculates the number of workgroups required in each dimension by dividing the number of elements by the corresponding workgroup denominator from the pipeline.
    - Logs the pipeline name, descriptor buffer information, and calculated workgroup sizes for debugging purposes.
    - Asserts that the current descriptor set index is valid and that the number of descriptor buffer infos matches the expected parameter count of the pipeline.
    - Updates the descriptor set with the provided buffer information.
    - Pushes the constants to the compute shader using the specified layout and stage.
    - Binds the pipeline and descriptor sets to the command buffer for execution.
    - Dispatches the compute shader with the calculated workgroup sizes.
- **Output**: This function does not return a value; it performs operations to set up and dispatch a compute pipeline in Vulkan.
- **Functions called**:
    - [`push_constant_size`](#push_constant_size)
    - [`push_constant_data`](#push_constant_data)


---
### ggml\_vk\_end\_submission<!-- {{#callable:ggml_vk_end_submission}} -->
The `ggml_vk_end_submission` function finalizes a Vulkan submission by ending the associated buffer and updating the wait and signal semaphores.
- **Inputs**:
    - `s`: A reference to a `vk_submission` object that represents the Vulkan submission being finalized.
    - `wait_semaphores`: A vector of `vk_semaphore` objects that will be used to wait for the completion of the submission.
    - `signal_semaphores`: A vector of `vk_semaphore` objects that will be signaled upon the completion of the submission.
- **Control Flow**:
    - The function first calls the `end` method on the `buffer` member of the `vk_submission` object `s`, which likely finalizes the buffer operations.
    - It then moves the contents of the `wait_semaphores` vector into the `wait_semaphores` member of the `vk_submission` object `s`, effectively transferring ownership.
    - Finally, it moves the contents of the `signal_semaphores` vector into the `signal_semaphores` member of the `vk_submission` object `s`, also transferring ownership.
- **Output**: The function does not return a value; it modifies the state of the `vk_submission` object `s` by finalizing the buffer and updating the semaphore vectors.


---
### ggml\_vk\_ctx\_end<!-- {{#callable:ggml_vk_ctx_end}} -->
Ends the Vulkan context by finalizing the buffer and resetting the context state.
- **Inputs**:
    - `ctx`: A reference to a `vk_context` object that holds the Vulkan context state.
- **Control Flow**:
    - Logs a debug message indicating the function call and the size of the sequence in the context.
    - Checks if the `s` member of the context is a null pointer; if it is, the function returns immediately.
    - Calls the `end` method on the `buffer` member of the `s` object to finalize the buffer.
    - Sets the `s` member of the context to null, effectively resetting the context state.
- **Output**: The function does not return a value; it modifies the state of the `vk_context` by finalizing the buffer and resetting the context.


---
### ggml\_vk\_ctx\_begin<!-- {{#callable:ggml_vk_ctx_begin}} -->
Begins a new Vulkan context by initializing a submission sequence.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device.
    - `subctx`: A reference to a `vk_context` object representing the Vulkan context to be initialized.
- **Control Flow**:
    - Logs the beginning of the context initialization with the device name.
    - Checks if the `subctx` has an existing submission sequence; if so, it calls [`ggml_vk_ctx_end`](#ggml_vk_ctx_end) to end it.
    - Calls [`ggml_vk_begin_submission`](#ggml_vk_begin_submission) to start a new submission sequence and stores it in `subctx->seqs`.
    - Updates `subctx->s` to point to the latest submission sequence.
- **Output**: The function does not return a value; it modifies the `subctx` to reflect the new Vulkan context state.
- **Functions called**:
    - [`ggml_vk_ctx_end`](#ggml_vk_ctx_end)
    - [`ggml_vk_begin_submission`](#ggml_vk_begin_submission)


---
### ggml\_vk\_align\_size<!-- {{#callable:ggml_vk_align_size}} -->
Calculates the aligned size based on a given width and alignment.
- **Inputs**:
    - `width`: The original size that needs to be aligned.
    - `align`: The alignment value to which the width should be aligned.
- **Control Flow**:
    - Logs the input parameters `width` and `align` for debugging purposes.
    - Calculates the aligned size by dividing `width` by `align`, rounding up to the nearest whole number, and then multiplying by `align`.
- **Output**: Returns the aligned size as a `size_t` value.


---
### deferred\_memcpy<!-- {{#callable:deferred_memcpy}} -->
The `deferred_memcpy` function conditionally performs a memory copy operation or stores the copy operation parameters for deferred execution.
- **Inputs**:
    - `dst`: A pointer to the destination memory location where data will be copied.
    - `src`: A pointer to the source memory location from which data will be copied.
    - `size`: The number of bytes to copy from the source to the destination.
    - `memcpys`: An optional pointer to a vector of `vk_staging_memcpy` objects that will store the copy operation parameters for deferred execution.
- **Control Flow**:
    - The function first checks if the `memcpys` pointer is null.
    - If `memcpys` is null, it directly performs the memory copy using `memcpy`.
    - If `memcpys` is not null, it creates a new `vk_staging_memcpy` object with the provided parameters and adds it to the `memcpys` vector.
- **Output**: The function does not return a value; it either performs a memory copy immediately or stores the parameters for a future copy operation.


---
### ggml\_vk\_ensure\_sync\_staging\_buffer<!-- {{#callable:ggml_vk_ensure_sync_staging_buffer}} -->
Ensures that the synchronization staging buffer for a Vulkan device is allocated and has sufficient size.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device.
    - `size`: A `size_t` value indicating the required size for the staging buffer.
- **Control Flow**:
    - Checks if the `sync_staging` buffer is null or if its size is less than the required size.
    - Logs a memory allocation request if the buffer needs to be created or resized.
    - Destroys the existing `sync_staging` buffer if it is not sufficient.
    - Creates a new `sync_staging` buffer with the specified size and memory properties.
- **Output**: The function does not return a value; it modifies the `sync_staging` buffer of the `device` if necessary.
- **Functions called**:
    - [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer)
    - [`ggml_vk_create_buffer_check`](#ggml_vk_create_buffer_check)


---
### ggml\_vk\_buffer\_write\_nc\_async<!-- {{#callable:ggml_vk_buffer_write_nc_async}} -->
Asynchronously writes data from a `ggml_tensor` to a Vulkan buffer, handling both pinned and non-pinned memory scenarios.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure, which contains Vulkan device context information.
    - `subctx`: A reference to a `vk_context` object that represents the current Vulkan context for operations.
    - `dst`: A reference to a `vk_buffer` object representing the destination Vulkan buffer where data will be written.
    - `offset`: A size_t value indicating the offset in the destination buffer where the data will be written.
    - `tensor`: A pointer to a `ggml_tensor` structure containing the source data to be written to the Vulkan buffer.
    - `sync_staging`: A boolean flag indicating whether to synchronize the staging buffer during the write operation.
- **Control Flow**:
    - Logs the function call and asserts that the tensor is not contiguous.
    - Checks if the destination buffer is host visible; if so, it aborts the operation.
    - Retrieves the source tensor's data and its dimensions and properties.
    - If the source buffer is pinned, it prepares a list of buffer copy operations for the Vulkan command buffer.
    - If the source buffer is not pinned and sync_staging is false, it aborts the operation.
    - If sync_staging is true, it ensures a staging buffer is available and performs a copy operation from the staging buffer to the destination buffer.
    - Uses deferred memory copy operations to handle the data transfer from the tensor to the staging buffer.
- **Output**: The function does not return a value; it performs asynchronous writes to the Vulkan buffer based on the provided tensor data.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_ensure_sync_staging_buffer`](#ggml_vk_ensure_sync_staging_buffer)
    - [`deferred_memcpy`](#deferred_memcpy)


---
### ggml\_vk\_buffer\_write\_2d\_async<!-- {{#callable:ggml_vk_buffer_write_2d_async}} -->
The `ggml_vk_buffer_write_2d_async` function asynchronously writes 2D data from a source buffer to a destination Vulkan buffer, handling both pinned and non-pinned memory scenarios.
- **Inputs**:
    - `subctx`: A `vk_context` representing the Vulkan context for the operation.
    - `dst`: A reference to a `vk_buffer` that represents the destination buffer where data will be written.
    - `offset`: A `size_t` indicating the offset in the destination buffer where the data will be written.
    - `src`: A pointer to the source data that will be written to the destination buffer.
    - `spitch`: A `size_t` representing the pitch (stride) of the source data.
    - `width`: A `size_t` indicating the width of the 2D data to be written.
    - `height`: A `size_t` indicating the height of the 2D data to be written.
    - `sync_staging`: A boolean flag indicating whether to synchronize the staging buffer; defaults to false.
- **Control Flow**:
    - The function begins by logging the dimensions of the data being written.
    - It checks if the destination buffer is host visible; if so, it aborts the operation with an error message.
    - The function retrieves the source buffer and its offset using [`ggml_vk_host_get`](#ggml_vk_host_get).
    - If the source buffer is pinned, it prepares a copy operation based on whether the width matches the source pitch.
    - If the source buffer is not pinned and `sync_staging` is false, it aborts the operation.
    - It ensures a synchronized staging buffer is available and prepares a copy command for the Vulkan command buffer.
    - Finally, it performs the actual memory copy from the source to the staging buffer, handling both single and multi-row writes.
- **Output**: The function does not return a value; it performs the write operation asynchronously and manages Vulkan buffer operations.
- **Functions called**:
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_ensure_sync_staging_buffer`](#ggml_vk_ensure_sync_staging_buffer)
    - [`deferred_memcpy`](#deferred_memcpy)


---
### ggml\_vk\_buffer\_write\_async<!-- {{#callable:ggml_vk_buffer_write_async}} -->
The `ggml_vk_buffer_write_async` function asynchronously writes data to a Vulkan buffer.
- **Inputs**:
    - `subctx`: A `vk_context` object representing the Vulkan context in which the operation is performed.
    - `dst`: A reference to a `vk_buffer` object that specifies the destination buffer where data will be written.
    - `offset`: A `size_t` value indicating the byte offset in the destination buffer where writing should begin.
    - `src`: A pointer to the source data that will be written to the destination buffer.
    - `size`: A `size_t` value representing the number of bytes to write from the source data.
    - `sync_staging`: A boolean flag indicating whether to synchronize the staging buffer after the write operation (default is false).
- **Control Flow**:
    - The function logs a debug message indicating the size of the data being written.
    - It then calls the [`ggml_vk_buffer_write_2d_async`](#ggml_vk_buffer_write_2d_async) function to perform the actual asynchronous write operation, passing along the context, destination buffer, offset, source data, size, and synchronization flag.
- **Output**: The function does not return a value; it performs an asynchronous operation to write data to the specified Vulkan buffer.
- **Functions called**:
    - [`ggml_vk_buffer_write_2d_async`](#ggml_vk_buffer_write_2d_async)


---
### ggml\_vk\_buffer\_write\_2d<!-- {{#callable:ggml_vk_buffer_write_2d}} -->
The `ggml_vk_buffer_write_2d` function writes 2D data from a source buffer to a destination Vulkan buffer, handling both mapped and unmapped memory scenarios.
- **Inputs**:
    - `dst`: A reference to a `vk_buffer` object representing the destination buffer where data will be written.
    - `offset`: A size_t value indicating the byte offset in the destination buffer where writing should begin.
    - `src`: A pointer to the source data buffer from which the 2D data will be copied.
    - `spitch`: A size_t value representing the pitch (in bytes) of the source buffer, which is the number of bytes between the start of one row and the start of the next.
    - `width`: A size_t value indicating the width (in bytes) of the data to be written for each row.
    - `height`: A size_t value indicating the number of rows of data to be written.
- **Control Flow**:
    - The function starts by logging the dimensions of the data being written.
    - It checks if the destination buffer is host-visible; if so, it asserts that it is also host-coherent.
    - If the buffer is host-visible, it uses a loop to copy each row of data from the source buffer to the destination buffer using `memcpy`.
    - If the buffer is not host-visible, it creates a temporary Vulkan context for asynchronous operations.
    - The function begins the Vulkan context, calls [`ggml_vk_buffer_write_2d_async`](#ggml_vk_buffer_write_2d_async) to handle the asynchronous write operation, and ends the context.
    - It then processes any in-memory copy operations recorded in the temporary context.
    - Finally, it submits the context and waits for the fence associated with the destination device to ensure the operation is complete.
- **Output**: The function does not return a value; it performs the operation of writing data to the destination buffer, ensuring synchronization and proper handling of memory properties.
- **Functions called**:
    - [`ggml_vk_create_temporary_context`](#ggml_vk_create_temporary_context)
    - [`ggml_vk_ctx_begin`](#ggml_vk_ctx_begin)
    - [`ggml_vk_buffer_write_2d_async`](#ggml_vk_buffer_write_2d_async)
    - [`ggml_vk_ctx_end`](#ggml_vk_ctx_end)
    - [`ggml_vk_submit`](#ggml_vk_submit)


---
### ggml\_vk\_buffer\_write<!-- {{#callable:ggml_vk_buffer_write}} -->
Writes data from a source pointer to a specified offset in a Vulkan buffer.
- **Inputs**:
    - `dst`: A reference to a `vk_buffer` object representing the destination buffer where data will be written.
    - `offset`: A `size_t` value indicating the byte offset in the destination buffer where writing should begin.
    - `src`: A pointer to the source data that will be written to the destination buffer.
    - `size`: A `size_t` value representing the number of bytes to write from the source to the destination buffer.
- **Control Flow**:
    - Logs the size of the data to be written using `VK_LOG_DEBUG` for debugging purposes.
    - Calls the helper function [`ggml_vk_buffer_write_2d`](#ggml_vk_buffer_write_2d) to perform the actual writing of data, passing the destination buffer, offset, source pointer, and size.
- **Output**: This function does not return a value; it performs a write operation to the specified Vulkan buffer.
- **Functions called**:
    - [`ggml_vk_buffer_write_2d`](#ggml_vk_buffer_write_2d)


---
### ggml\_vk\_buffer\_read\_2d\_async<!-- {{#callable:ggml_vk_buffer_read_2d_async}} -->
The `ggml_vk_buffer_read_2d_async` function asynchronously reads a 2D buffer from a Vulkan source buffer to a destination memory location, handling both pinned and non-pinned memory scenarios.
- **Inputs**:
    - `subctx`: A `vk_context` object representing the Vulkan context for the operation.
    - `src`: A reference to a `vk_buffer` object representing the source buffer from which data will be read.
    - `offset`: A `size_t` value indicating the starting offset in the source buffer from which to read data.
    - `dst`: A pointer to the destination memory where the read data will be stored.
    - `spitch`: A `size_t` value representing the stride (in bytes) of the source buffer.
    - `dpitch`: A `size_t` value representing the stride (in bytes) of the destination memory.
    - `width`: A `size_t` value indicating the width (in bytes) of the 2D data to be read.
    - `height`: A `size_t` value indicating the height (in number of rows) of the 2D data to be read.
    - `sync_staging`: A boolean flag indicating whether to synchronize the staging buffer; defaults to false.
- **Control Flow**:
    - The function begins by logging the parameters and asserting that width, height, and source buffer are valid.
    - It checks if the destination memory is pinned and retrieves the appropriate buffer and offset.
    - Depending on whether the width and strides are equal, it prepares a vector of `vk::BufferCopy` slices for the copy operation.
    - If the destination buffer is pinned, it synchronizes the buffers and performs the copy directly.
    - If the destination is not pinned and `sync_staging` is false, it aborts the operation.
    - If `sync_staging` is true, it ensures a staging buffer is available, synchronizes, and performs the copy from the staging buffer to the destination.
- **Output**: The function does not return a value; it performs the asynchronous read operation and writes the data to the specified destination memory.
- **Functions called**:
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_ensure_sync_staging_buffer`](#ggml_vk_ensure_sync_staging_buffer)
    - [`deferred_memcpy`](#deferred_memcpy)


---
### ggml\_vk\_buffer\_read\_async<!-- {{#callable:ggml_vk_buffer_read_async}} -->
This function asynchronously reads data from a Vulkan buffer into a specified destination.
- **Inputs**:
    - `subctx`: A `vk_context` object representing the Vulkan context used for the operation.
    - `src`: A reference to a `vk_buffer` object that specifies the source buffer from which data will be read.
    - `offset`: A `size_t` value indicating the byte offset in the source buffer from which to start reading.
    - `dst`: A pointer to the destination memory where the read data will be stored.
    - `size`: A `size_t` value representing the number of bytes to read from the source buffer.
    - `sync_staging`: A boolean flag that indicates whether to synchronize the staging buffer after the read operation (default is false).
- **Control Flow**:
    - The function calls [`ggml_vk_buffer_read_2d_async`](#ggml_vk_buffer_read_2d_async) with the provided parameters, including the source buffer, offset, destination pointer, and size.
    - The parameters are adjusted to fit the expected format of [`ggml_vk_buffer_read_2d_async`](#ggml_vk_buffer_read_2d_async), specifically passing the size multiple times to match its signature.
- **Output**: This function does not return a value; it initiates an asynchronous read operation from the Vulkan buffer.
- **Functions called**:
    - [`ggml_vk_buffer_read_2d_async`](#ggml_vk_buffer_read_2d_async)


---
### ggml\_vk\_buffer\_read<!-- {{#callable:ggml_vk_buffer_read}} -->
Reads data from a Vulkan buffer into a specified destination memory location.
- **Inputs**:
    - `src`: A reference to a `vk_buffer` object representing the source Vulkan buffer from which data will be read.
    - `offset`: A `size_t` value indicating the byte offset in the source buffer from which to start reading.
    - `dst`: A pointer to the destination memory location where the read data will be stored.
    - `size`: A `size_t` value specifying the number of bytes to read from the source buffer.
- **Control Flow**:
    - Logs the parameters of the function call for debugging purposes.
    - Checks if the source buffer's memory is host-visible and if the device is a Unified Memory Architecture (UMA) device.
    - If both conditions are true, asserts that the memory is also coherent and performs a direct memory copy from the source buffer to the destination.
    - If the conditions are not met, creates a temporary Vulkan context for asynchronous reading.
    - Begins the Vulkan context, calls [`ggml_vk_buffer_read_async`](#ggml_vk_buffer_read_async) to read the data asynchronously, and ends the context.
    - Submits the context and waits for the fence to ensure the read operation is complete.
    - Resets the fence after the operation is done.
    - Copies the data from the temporary context's output memory copies to the destination.
- **Output**: The function does not return a value; it directly modifies the memory at the destination pointer with the data read from the Vulkan buffer.
- **Functions called**:
    - [`ggml_vk_create_temporary_context`](#ggml_vk_create_temporary_context)
    - [`ggml_vk_ctx_begin`](#ggml_vk_ctx_begin)
    - [`ggml_vk_buffer_read_async`](#ggml_vk_buffer_read_async)
    - [`ggml_vk_ctx_end`](#ggml_vk_ctx_end)
    - [`ggml_vk_submit`](#ggml_vk_submit)


---
### ggml\_vk\_buffer\_copy\_async<!-- {{#callable:ggml_vk_buffer_copy_async}} -->
Copies data asynchronously from a source Vulkan buffer to a destination Vulkan buffer.
- **Inputs**:
    - `ctx`: A reference to the Vulkan context used for command buffer operations.
    - `dst`: A reference to the destination Vulkan buffer where data will be copied.
    - `dst_offset`: The offset in the destination buffer where the data will be copied to.
    - `src`: A reference to the source Vulkan buffer from which data will be copied.
    - `src_offset`: The offset in the source buffer from which data will be copied.
    - `size`: The number of bytes to copy from the source buffer to the destination buffer.
- **Control Flow**:
    - Logs the size of the data to be copied for debugging purposes.
    - Asserts that both the source and destination buffers are located on the same Vulkan device to prevent errors.
    - Creates a `VkBufferCopy` structure to define the copy operation parameters.
    - Calls `vkCmdCopyBuffer` to perform the actual copy operation using the Vulkan command buffer.
- **Output**: This function does not return a value; it performs an asynchronous copy operation on the Vulkan buffers.


---
### ggml\_vk\_buffer\_copy<!-- {{#callable:ggml_vk_buffer_copy}} -->
Copies data from one Vulkan buffer to another, handling both single-device and multi-device scenarios.
- **Inputs**:
    - `dst`: A reference to the destination `vk_buffer` where data will be copied to.
    - `dst_offset`: The offset in the destination buffer where the copy will begin.
    - `src`: A reference to the source `vk_buffer` from which data will be copied.
    - `src_offset`: The offset in the source buffer from where the copy will begin.
    - `size`: The number of bytes to copy from the source buffer to the destination buffer.
- **Control Flow**:
    - Checks if the source and destination buffers are on the same device.
    - If they are on the same device, it creates a temporary Vulkan context, begins the context, and performs an asynchronous buffer copy.
    - After the copy, it waits for the fence to ensure the operation is complete before resetting the fence.
    - If the buffers are on different devices, it ensures that synchronization staging buffers are available for both source and destination.
    - Copies data from the source buffer to the source staging buffer, then uses `memcpy` to transfer data from the source staging buffer to the destination staging buffer.
    - Finally, it copies the data from the destination staging buffer to the destination buffer.
- **Output**: This function does not return a value; it performs the copy operation directly on the specified Vulkan buffers.
- **Functions called**:
    - [`ggml_vk_create_temporary_context`](#ggml_vk_create_temporary_context)
    - [`ggml_vk_ctx_begin`](#ggml_vk_ctx_begin)
    - [`ggml_vk_buffer_copy_async`](#ggml_vk_buffer_copy_async)
    - [`ggml_vk_ctx_end`](#ggml_vk_ctx_end)
    - [`ggml_vk_submit`](#ggml_vk_submit)
    - [`ggml_vk_ensure_sync_staging_buffer`](#ggml_vk_ensure_sync_staging_buffer)


---
### ggml\_vk\_buffer\_memset\_async<!-- {{#callable:ggml_vk_buffer_memset_async}} -->
Asynchronously fills a specified range of a Vulkan buffer with a given value.
- **Inputs**:
    - `ctx`: A reference to the Vulkan context that manages the Vulkan state.
    - `dst`: A reference to the Vulkan buffer that will be modified.
    - `offset`: The starting position in the buffer where the fill operation will begin.
    - `c`: The value to fill the buffer with.
    - `size`: The number of bytes to fill in the buffer.
- **Control Flow**:
    - Logs the parameters received by the function for debugging purposes.
    - Calls the `fillBuffer` method on the Vulkan context's buffer manager to perform the fill operation.
- **Output**: This function does not return a value; it modifies the specified Vulkan buffer in place.


---
### ggml\_vk\_buffer\_memset<!-- {{#callable:ggml_vk_buffer_memset}} -->
The `ggml_vk_buffer_memset` function fills a specified region of a Vulkan buffer with a given value.
- **Inputs**:
    - `dst`: A reference to a `vk_buffer` object representing the Vulkan buffer to be modified.
    - `offset`: The starting position in the buffer where the fill operation begins.
    - `c`: The 32-bit unsigned integer value to fill the buffer with.
    - `size`: The number of bytes to fill in the buffer starting from the offset.
- **Control Flow**:
    - Logs the parameters passed to the function for debugging purposes.
    - Creates a temporary Vulkan context for the buffer operations.
    - Begins the Vulkan context to prepare for commands.
    - Fills the specified region of the buffer with the given value using the `fillBuffer` method.
    - Ends the Vulkan context after the fill operation is complete.
    - Submits the commands in the temporary context to the Vulkan device.
    - Waits for the fence associated with the device to ensure the fill operation is complete.
    - Resets the fence to prepare for future operations.
- **Output**: The function does not return a value; it modifies the specified Vulkan buffer in place.
- **Functions called**:
    - [`ggml_vk_create_temporary_context`](#ggml_vk_create_temporary_context)
    - [`ggml_vk_ctx_begin`](#ggml_vk_ctx_begin)
    - [`ggml_vk_ctx_end`](#ggml_vk_ctx_end)
    - [`ggml_vk_submit`](#ggml_vk_submit)


---
### ggml\_vk\_guess\_split\_k<!-- {{#callable:ggml_vk_guess_split_k}} -->
The `ggml_vk_guess_split_k` function estimates an optimal split factor for the `k` dimension based on the given matrix dimensions and the capabilities of the GPU.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains information about the GPU context.
    - `m`: An integer representing the number of rows in the matrix.
    - `n`: An integer representing the number of columns in the matrix.
    - `k`: An integer representing the depth or third dimension of the matrix.
    - `pipeline`: A reference to a `vk_pipeline` object that contains information about the workgroup denominators.
- **Control Flow**:
    - The function starts by logging the input parameters `m`, `n`, and `k` for debugging purposes.
    - It initializes `split_k` to 1, which will be adjusted based on the conditions checked.
    - It checks if the GPU's shader core count is non-zero and if the dimensions `m` and `n` are sufficient based on the workgroup denominators.
    - If `k` is large (greater than or equal to 2048) and the product of the number of tiles in `m` and `n` is less than half the shader core count, it calculates a potential `split_k` value.
    - The calculated `split_k` is clamped to a maximum of 4, and if it equals 3, it is adjusted to 2.
    - If the `coopmat2` flag is set in the context, it ensures that `split_k` is aligned to 256 by halving it until the condition is met.
- **Output**: The function returns a `uint32_t` value representing the optimal split factor for the `k` dimension, which can be used to optimize GPU workload distribution.


---
### ggml\_vk\_guess\_matmul\_pipeline<!-- {{#callable:ggml_vk_guess_matmul_pipeline}} -->
The `ggml_vk_guess_matmul_pipeline` function determines the appropriate Vulkan pipeline for matrix multiplication based on input dimensions and alignment.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains device and context information.
    - `mmp`: A reference to a `vk_matmul_pipeline` structure that holds different pipeline configurations.
    - `m`: An unsigned 32-bit integer representing the number of rows in the first matrix.
    - `n`: An unsigned 32-bit integer representing the number of columns in the second matrix.
    - `aligned`: A boolean indicating whether the matrices are aligned.
    - `src0_type`: A `ggml_type` enumeration value representing the data type of the first source matrix.
    - `src1_type`: A `ggml_type` enumeration value representing the data type of the second source matrix.
- **Control Flow**:
    - The function logs the input parameters for debugging purposes.
    - It checks if the device supports cooperative matrix multiplication (coopmat2).
    - If coopmat2 is supported, it evaluates the N dimension against crossover thresholds to select between large, medium, or small shaders based on the source matrix type and alignment.
    - If coopmat2 is not supported, it evaluates the M and N dimensions against fixed thresholds to determine the appropriate shader based on the source matrix type and alignment.
    - The function returns the selected pipeline based on the evaluations.
- **Output**: The function returns a `vk_pipeline` that corresponds to the best-suited matrix multiplication pipeline based on the input parameters.
- **Functions called**:
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)


---
### ggml\_vk\_guess\_matmul\_pipeline\_align<!-- {{#callable:ggml_vk_guess_matmul_pipeline_align}} -->
The `ggml_vk_guess_matmul_pipeline_align` function estimates the alignment for a matrix multiplication pipeline based on input parameters.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `mmp`: A reference to a `vk_matmul_pipeline` object that represents the matrix multiplication pipeline.
    - `m`: An integer representing the number of rows in the first matrix.
    - `n`: An integer representing the number of columns in the second matrix.
    - `src0_type`: A `ggml_type` enumeration value indicating the data type of the first source matrix.
    - `src1_type`: A `ggml_type` enumeration value indicating the data type of the second source matrix.
- **Control Flow**:
    - The function begins by logging a debug message that includes the dimensions of the matrices and their types.
    - It then calls the [`ggml_vk_guess_matmul_pipeline`](#ggml_vk_guess_matmul_pipeline) function with the provided parameters, including a boolean value set to true, which likely indicates a specific behavior or mode.
    - Finally, it retrieves and returns the `align` property from the result of the [`ggml_vk_guess_matmul_pipeline`](#ggml_vk_guess_matmul_pipeline) function call.
- **Output**: The function returns a `uint32_t` value representing the alignment required for the matrix multiplication pipeline.
- **Functions called**:
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)
    - [`ggml_vk_guess_matmul_pipeline`](#ggml_vk_guess_matmul_pipeline)


---
### ggml\_vk\_matmul<!-- {{#callable:ggml_vk_matmul}} -->
`ggml_vk_matmul` performs matrix multiplication using Vulkan, supporting both standard and split-k operations.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context used for managing Vulkan resources.
    - `subctx`: A reference to the Vulkan context for the current operation.
    - `pipeline`: A reference to the Vulkan pipeline used for executing the matrix multiplication.
    - `a`: A `vk_subbuffer` representing the first input matrix.
    - `b`: A `vk_subbuffer` representing the second input matrix.
    - `d`: A `vk_subbuffer` where the result of the matrix multiplication will be stored.
    - `split_k_buffer`: A `vk_subbuffer` used for intermediate results when performing split-k operations.
    - `m`: The number of rows in the first matrix.
    - `n`: The number of columns in the second matrix.
    - `k`: The number of columns in the first matrix (or rows in the second matrix).
    - `stride_a`: The stride (or step size) for accessing elements in matrix `a`.
    - `stride_b`: The stride for accessing elements in matrix `b`.
    - `stride_d`: The stride for accessing elements in the result matrix `d`.
    - `batch_stride_a`: The stride for batch processing in matrix `a`.
    - `batch_stride_b`: The stride for batch processing in matrix `b`.
    - `batch_stride_d`: The stride for batch processing in the result matrix `d`.
    - `split_k`: The number of splits for the `k` dimension in split-k operations.
    - `batch`: The number of batches to process.
    - `ne02`: An additional parameter related to the second dimension of the operation.
    - `ne12`: An additional parameter related to the first dimension of the operation.
    - `broadcast2`: A flag indicating whether to broadcast along the second dimension.
    - `broadcast3`: A flag indicating whether to broadcast along the third dimension.
    - `padded_n`: The padded size of the second dimension.
- **Control Flow**:
    - The function starts by logging the input parameters for debugging purposes.
    - It synchronizes the Vulkan buffers to ensure they are ready for processing.
    - If `split_k` is equal to 1, it prepares push constants and dispatches the pipeline for standard matrix multiplication.
    - If `split_k` is greater than 1, it asserts that the batch stride for the result matrix is correct, prepares push constants for the split-k operation, and dispatches the pipeline for the split-k multiplication.
    - After the split-k multiplication, it synchronizes the buffers again.
    - Finally, it dispatches another pipeline to reduce the results from the split-k operation into the final output.
- **Output**: The function does not return a value; instead, it writes the result of the matrix multiplication directly into the provided output buffer `d`.
- **Functions called**:
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)


---
### ggml\_vk\_guess\_matmul\_id\_pipeline<!-- {{#callable:ggml_vk_guess_matmul_id_pipeline}} -->
Determines the appropriate matrix multiplication pipeline based on input dimensions and alignment.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains device and context information.
    - `mmp`: A reference to a `vk_matmul_pipeline` structure that holds different matrix multiplication pipeline configurations.
    - `m`: An unsigned integer representing the number of rows in the first matrix.
    - `n`: An unsigned integer representing the number of columns in the second matrix.
    - `aligned`: A boolean indicating whether the matrices are aligned.
    - `src0_type`: An enumeration value of type `ggml_type` that specifies the type of the source matrix.
- **Control Flow**:
    - Logs the function call with input parameters for debugging purposes.
    - Checks if the device supports cooperative matrix multiplication (coopmat2).
    - If coopmat2 is supported, it evaluates the N dimension against crossover thresholds to select between large, medium, or small shaders based on the source type and alignment.
    - If coopmat2 is not supported, it evaluates the M and N dimensions against fixed thresholds to select the appropriate shader based on the source type and alignment.
- **Output**: Returns a `vk_pipeline` that corresponds to the selected matrix multiplication configuration based on the input parameters.
- **Functions called**:
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)


---
### ggml\_vk\_guess\_matmul\_id\_pipeline\_align<!-- {{#callable:ggml_vk_guess_matmul_id_pipeline_align}} -->
The `ggml_vk_guess_matmul_id_pipeline_align` function determines the alignment of a matrix multiplication pipeline based on given dimensions and source type.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `mmp`: A reference to a `vk_matmul_pipeline` object that represents the matrix multiplication pipeline.
    - `m`: An integer representing the number of rows in the matrix.
    - `n`: An integer representing the number of columns in the matrix.
    - `src0_type`: An enumeration value of type `ggml_type` that specifies the data type of the source matrix.
- **Control Flow**:
    - The function logs a debug message that includes the dimensions of the matrix and the source type name.
    - It calls the [`ggml_vk_guess_matmul_id_pipeline`](#ggml_vk_guess_matmul_id_pipeline) function with the provided context, pipeline, dimensions, a boolean value set to true, and the source type to retrieve the corresponding pipeline ID.
    - The function then accesses the `align` property of the returned pipeline ID and returns it.
- **Output**: The function returns a `uint32_t` value representing the alignment of the matrix multiplication pipeline.
- **Functions called**:
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)
    - [`ggml_vk_guess_matmul_id_pipeline`](#ggml_vk_guess_matmul_id_pipeline)


---
### ggml\_vk\_matmul\_id<!-- {{#callable:ggml_vk_matmul_id}} -->
Performs matrix multiplication using Vulkan with specified parameters and buffers.
- **Inputs**:
    - `ctx`: Pointer to the Vulkan backend context used for managing Vulkan resources.
    - `subctx`: Reference to the Vulkan context for the current operation.
    - `pipeline`: Reference to the Vulkan pipeline used for executing the matrix multiplication.
    - `a`: A `vk_subbuffer` representing the first input matrix.
    - `b`: A `vk_subbuffer` representing the second input matrix.
    - `d`: A `vk_subbuffer` where the result of the matrix multiplication will be stored.
    - `ids`: A `vk_subbuffer` containing identifiers for the operation.
    - `m`: The number of rows in the first matrix.
    - `n`: The number of columns in the second matrix.
    - `k`: The number of columns in the first matrix (and rows in the second matrix).
    - `stride_a`: The stride for the first matrix.
    - `stride_b`: The stride for the second matrix.
    - `stride_d`: The stride for the output matrix.
    - `batch_stride_a`: The batch stride for the first matrix.
    - `batch_stride_b`: The batch stride for the second matrix.
    - `batch_stride_d`: The batch stride for the output matrix.
    - `n_as`: The number of elements in the first matrix's batch.
    - `nei0`: The number of elements in the first dimension of the input.
    - `nei1`: The number of elements in the second dimension of the input.
    - `nbi1`: The number of batch elements in the second dimension.
    - `ne11`: The number of elements in the last dimension.
    - `padded_n`: The padded size for the second matrix.
- **Control Flow**:
    - Logs the input parameters for debugging purposes.
    - Synchronizes the Vulkan buffers to ensure they are ready for the operation.
    - Creates a structure of push constants containing the parameters for the matrix multiplication.
    - Dispatches the Vulkan pipeline to perform the matrix multiplication using the provided buffers and push constants.
- **Output**: The function does not return a value; instead, it performs the matrix multiplication operation and stores the result in the specified output buffer.
- **Functions called**:
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)


---
### ggml\_vk\_dim01\_contiguous<!-- {{#callable:ggml_vk_dim01_contiguous}} -->
Checks if a given tensor is contiguous in memory based on its dimensions and type.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that contains information about the tensor's dimensions, type, and other properties.
- **Control Flow**:
    - The function evaluates three conditions to determine if the tensor is contiguous.
    - It checks if the first dimension size matches the size of the tensor type.
    - It verifies if the second dimension size is equal to the product of the first dimension size and the number of elements in the first dimension divided by the block size of the tensor type.
    - It confirms that the third dimension size is equal to the product of the second dimension size and the number of elements in the second dimension.
- **Output**: Returns a boolean value indicating whether the tensor is contiguous in memory based on the evaluated conditions.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)


---
### ggml\_vk\_get\_cpy\_pipeline<!-- {{#callable:ggml_vk_get_cpy_pipeline}} -->
The `ggml_vk_get_cpy_pipeline` function retrieves the appropriate Vulkan pipeline for copying data between tensors based on their types and contiguity.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains device information for Vulkan operations.
    - `src`: A pointer to a `ggml_tensor` structure representing the source tensor from which data will be copied.
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor to which data will be copied.
    - `to`: An enumeration value of type `ggml_type` indicating the target data type for the copy operation.
- **Control Flow**:
    - The function first checks if both the source and destination tensors are contiguous using [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous).
    - It then evaluates the types of the source tensor and the target type to determine the appropriate Vulkan pipeline to use for the copy operation.
    - For each combination of source and target types, it checks if the tensors are contiguous and returns the corresponding pipeline from the `ctx->device` structure.
    - If the source and target types are the same, it checks the size of the type to decide between different copy pipelines.
    - If no matching pipeline is found, an error message is printed, and the function aborts execution.
- **Output**: The function returns a `vk_pipeline` object that corresponds to the selected Vulkan pipeline for the copy operation, or it aborts if no suitable pipeline is found.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_is_quantized`](../ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)


---
### ggml\_vk\_cpy\_to\_contiguous<!-- {{#callable:ggml_vk_cpy_to_contiguous}} -->
Copies data from a Vulkan subbuffer to a contiguous memory layout based on the properties of a given tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object representing the Vulkan context for the operation.
    - `pipeline`: A `vk_pipeline` object that defines the compute pipeline to be used for the operation.
    - `tensor`: A pointer to a `ggml_tensor` structure that contains metadata about the tensor, including its type and dimensions.
    - `in`: A `vk_subbuffer` representing the input buffer from which data will be copied.
    - `out`: A `vk_subbuffer` representing the output buffer to which data will be copied.
- **Control Flow**:
    - Logs the details of the tensor and the sizes of the input and output buffers.
    - Calculates the size of the tensor type using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size).
    - Determines the number of elements in the tensor using [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements).
    - Based on the number of elements, sets up the `elements` array to define the dispatch dimensions for the Vulkan operation.
    - Initializes a `vk_op_unary_push_constants` structure with various tensor properties and dimensions.
    - Calls [`init_pushconst_fastdiv`](#init_pushconst_fastdiv) to prepare the push constants for the Vulkan pipeline.
    - Synchronizes the buffers using [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers) to ensure data consistency.
    - Dispatches the Vulkan pipeline with the input and output buffers and the calculated push constants.
- **Output**: The function does not return a value; it performs a copy operation from the input subbuffer to the output subbuffer using Vulkan compute shaders.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`init_pushconst_fastdiv`](#init_pushconst_fastdiv)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)


---
### ggml\_vk\_get\_quantize\_pipeline<!-- {{#callable:ggml_vk_get_quantize_pipeline}} -->
Retrieves the appropriate quantization pipeline based on the specified `ggml_type`.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the Vulkan device context.
    - `type`: An enumeration of type `ggml_type` that specifies the quantization type to retrieve the pipeline for.
- **Control Flow**:
    - The function begins by evaluating the `type` argument using a `switch` statement.
    - If the `type` matches `GGML_TYPE_Q8_1`, it returns the corresponding quantization pipeline from the `ctx` structure.
    - If the `type` does not match any known cases, it outputs an error message to standard error and aborts the program.
- **Output**: Returns a `vk_pipeline` object representing the quantization pipeline for the specified type, or aborts if the type is unsupported.
- **Functions called**:
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)


---
### ggml\_vk\_quantize\_q8\_1<!-- {{#callable:ggml_vk_quantize_q8_1}} -->
The `ggml_vk_quantize_q8_1` function performs quantization of input data using a Vulkan pipeline.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context used for managing Vulkan resources.
    - `subctx`: A reference to a Vulkan context that holds the state for the current Vulkan operations.
    - `in`: A `vk_subbuffer` representing the input data buffer to be quantized.
    - `out`: A `vk_subbuffer` representing the output data buffer where the quantized results will be stored.
    - `ne`: A `uint32_t` representing the number of elements to be processed in the quantization.
- **Control Flow**:
    - Logs the sizes of the input and output buffers along with the number of elements to be processed.
    - Retrieves the Vulkan pipeline for quantization using the [`ggml_vk_get_quantize_pipeline`](#ggml_vk_get_quantize_pipeline) function.
    - Synchronizes the buffers in the Vulkan context to ensure they are ready for processing.
    - Dispatches the Vulkan pipeline with the input and output buffers, specifying the number of elements to process.
- **Output**: The function does not return a value; instead, it performs operations that modify the output buffer in place with quantized data.
- **Functions called**:
    - [`ggml_vk_get_quantize_pipeline`](#ggml_vk_get_quantize_pipeline)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)


---
### ggml\_vk\_mul\_mat\_q\_f16<!-- {{#callable:ggml_vk_mul_mat_q_f16}} -->
Performs matrix multiplication of two quantized tensors in a Vulkan backend context.
- **Inputs**:
    - `ctx`: Pointer to the Vulkan backend context used for managing resources and operations.
    - `subctx`: Reference to a Vulkan context for managing sub-operations.
    - `src0`: Pointer to the first source tensor to be multiplied.
    - `src1`: Pointer to the second source tensor to be multiplied.
    - `dst`: Pointer to the destination tensor where the result will be stored.
    - `dryrun`: Boolean flag indicating whether to perform a dry run without executing the multiplication.
- **Control Flow**:
    - Logs the details of the input tensors and the dry run status.
    - Validates the contiguity and types of the input tensors using assertions.
    - Calculates the dimensions and sizes of the input and output tensors.
    - Checks if the device supports Unified Memory Architecture (UMA) and retrieves buffers accordingly.
    - Determines if the input tensors need to be dequantized or converted to a different format based on their types and contiguity.
    - Requests necessary descriptor sets for the Vulkan pipeline based on the computed sizes and conditions.
    - If in dry run mode, checks memory allocation sizes and updates preallocation sizes without executing the multiplication.
    - If not in dry run mode, prepares the buffers and executes the matrix multiplication using Vulkan commands.
- **Output**: The function does not return a value but populates the destination tensor with the result of the matrix multiplication.
- **Functions called**:
    - [`ggml_vk_dim01_contiguous`](#ggml_vk_dim01_contiguous)
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_vk_get_mul_mat_mat_pipeline`](#ggml_vk_get_mul_mat_mat_pipeline)
    - [`ggml_vk_align_size`](#ggml_vk_align_size)
    - [`ggml_vk_guess_matmul_pipeline_align`](#ggml_vk_guess_matmul_pipeline_align)
    - [`ggml_vk_guess_matmul_pipeline`](#ggml_vk_guess_matmul_pipeline)
    - [`ggml_vk_guess_split_k`](#ggml_vk_guess_split_k)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`ggml_vk_get_cpy_pipeline`](#ggml_vk_get_cpy_pipeline)
    - [`ggml_vk_get_to_fp16`](#ggml_vk_get_to_fp16)
    - [`ggml_vk_get_quantize_pipeline`](#ggml_vk_get_quantize_pipeline)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_vk_cpy_to_contiguous`](#ggml_vk_cpy_to_contiguous)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)
    - [`ggml_vk_quantize_q8_1`](#ggml_vk_quantize_q8_1)
    - [`ggml_vk_matmul`](#ggml_vk_matmul)


---
### ggml\_vk\_mul\_mat\_vec\_q\_f16<!-- {{#callable:ggml_vk_mul_mat_vec_q_f16}} -->
Performs matrix-vector multiplication for quantized half-precision floating-point tensors using Vulkan.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to the Vulkan context used for dispatching commands.
    - `src0`: A pointer to the first input tensor (`ggml_tensor`) representing the matrix.
    - `src1`: A pointer to the second input tensor (`ggml_tensor`) representing the vector.
    - `dst`: A pointer to the output tensor (`ggml_tensor`) where the result will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (true) or execute the operation (false).
- **Control Flow**:
    - Logs the details of the input tensors and the output tensor.
    - Validates the contiguity and types of the input tensors using assertions.
    - Calculates the necessary dimensions and sizes for the input and output tensors.
    - Checks if the input tensors are in Unified Memory Architecture (UMA) and retrieves their data if applicable.
    - Determines if the input tensors need dequantization based on their contiguity and types.
    - If `dryrun` is true, it checks for memory allocation sizes and requests descriptor sets without executing the multiplication.
    - If `dryrun` is false, it prepares the buffers for the input and output tensors, handling any necessary dequantization.
    - Sets up the compute pipeline and dispatches the matrix-vector multiplication operation using Vulkan.
- **Output**: The function does not return a value but writes the result of the matrix-vector multiplication into the `dst` tensor.
- **Functions called**:
    - [`ggml_vk_dim01_contiguous`](#ggml_vk_dim01_contiguous)
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`ggml_vk_align_size`](#ggml_vk_align_size)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`ggml_vk_get_cpy_pipeline`](#ggml_vk_get_cpy_pipeline)
    - [`ggml_vk_get_to_fp16`](#ggml_vk_get_to_fp16)
    - [`ggml_vk_get_dequantize_mul_mat_vec`](#ggml_vk_get_dequantize_mul_mat_vec)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_vk_cpy_to_contiguous`](#ggml_vk_cpy_to_contiguous)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)


---
### ggml\_vk\_mul\_mat\_vec\_p021\_f16\_f32<!-- {{#callable:ggml_vk_mul_mat_vec_p021_f16_f32}} -->
Multiplies a matrix (`src0`) with a vector (`src1`) and stores the result in a destination tensor (`dst`) using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context (`ggml_backend_vk_context`) which holds the state and configuration for Vulkan operations.
    - `subctx`: A reference to a Vulkan context (`vk_context`) used for managing Vulkan resources and commands.
    - `src0`: A pointer to the source tensor (`ggml_tensor`) representing the matrix to be multiplied.
    - `src1`: A pointer to the source tensor (`ggml_tensor`) representing the vector to be multiplied with the matrix.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the result of the multiplication will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (true) or execute the multiplication (false). Default is false.
- **Control Flow**:
    - Logs the details of the input tensors and the dry run status.
    - Validates the input tensors to ensure they are in the correct format and type.
    - Calculates the necessary sizes and offsets for the buffers based on the input tensor dimensions.
    - If `dryrun` is true, requests descriptor sets and exits without performing the multiplication.
    - Retrieves the device buffers for the input and output tensors.
    - Calculates shader offsets and buffer offsets for the Vulkan dispatch.
    - Sets up the pipeline configuration and dispatches the multiplication operation to the GPU.
- **Output**: The function does not return a value; instead, it writes the result of the matrix-vector multiplication directly into the `dst` tensor.
- **Functions called**:
    - [`ggml_is_permuted`](../ggml.c.driver.md#ggml_is_permuted)
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`ggml_vk_align_size`](#ggml_vk_align_size)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)


---
### ggml\_vk\_mul\_mat\_vec\_nc\_f16\_f32<!-- {{#callable:ggml_vk_mul_mat_vec_nc_f16_f32}} -->
Performs matrix-vector multiplication using Vulkan for a half-precision floating-point matrix and a single-precision floating-point vector.
- **Inputs**:
    - `ctx`: Pointer to the Vulkan backend context used for managing Vulkan resources.
    - `subctx`: Reference to a Vulkan context for the current operation.
    - `src0`: Pointer to the source tensor representing the matrix in half-precision format.
    - `src1`: Pointer to the source tensor representing the vector in single-precision format.
    - `dst`: Pointer to the destination tensor where the result of the multiplication will be stored.
    - `dryrun`: Boolean flag indicating whether to perform a dry run (true) or execute the multiplication (false).
- **Control Flow**:
    - Logs the details of the input tensors and the dry run status.
    - Validates the input tensors to ensure they are not transposed or permuted and that they have the correct data types.
    - Calculates necessary parameters such as dimensions and strides for the matrix and vector.
    - If 'dryrun' is true, requests descriptor sets and exits without performing the multiplication.
    - Retrieves device buffers for the source and destination tensors.
    - Aligns buffer offsets according to Vulkan's storage buffer alignment requirements.
    - Sets up the pipeline parameters and dispatches the Vulkan compute pipeline to perform the multiplication.
- **Output**: The function does not return a value but writes the result of the matrix-vector multiplication into the destination tensor.
- **Functions called**:
    - [`ggml_is_transposed`](../ggml.c.driver.md#ggml_is_transposed)
    - [`ggml_is_permuted`](../ggml.c.driver.md#ggml_is_permuted)
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)


---
### ggml\_vk\_mul\_mat<!-- {{#callable:ggml_vk_mul_mat}} -->
Performs matrix multiplication of two tensors using Vulkan backend with various optimizations based on tensor properties.
- **Inputs**:
    - `ctx`: Pointer to the Vulkan backend context used for managing Vulkan resources.
    - `subctx`: Reference to a Vulkan context that may contain additional state or resources for the operation.
    - `src0`: Pointer to the first input tensor for the multiplication operation.
    - `src1`: Pointer to the second input tensor for the multiplication operation.
    - `dst`: Pointer to the output tensor where the result of the multiplication will be stored.
    - `dryrun`: Boolean flag indicating whether to perform a dry run (true) or execute the operation (false).
- **Control Flow**:
    - Logs the function call with input tensor addresses.
    - Checks if both input tensors are of type `GGML_TYPE_F16` and are permuted with specific conditions to call [`ggml_vk_mul_mat_vec_p021_f16_f32`](#ggml_vk_mul_mat_vec_p021_f16_f32) for optimized multiplication.
    - If the first tensor is `GGML_TYPE_F16` and not contiguous, and the second tensor is not transposed, calls [`ggml_vk_mul_mat_vec_nc_f16_f32`](#ggml_vk_mul_mat_vec_nc_f16_f32) for another optimized multiplication.
    - If the output tensor's second dimension is 1 or meets specific conditions, calls [`ggml_vk_mul_mat_vec_q_f16`](#ggml_vk_mul_mat_vec_q_f16) for quantized multiplication.
    - If none of the above conditions are met, defaults to calling [`ggml_vk_mul_mat_q_f16`](#ggml_vk_mul_mat_q_f16) for general multiplication.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the result of the matrix multiplication.
- **Functions called**:
    - [`ggml_is_permuted`](../ggml.c.driver.md#ggml_is_permuted)
    - [`ggml_vk_mul_mat_vec_p021_f16_f32`](#ggml_vk_mul_mat_vec_p021_f16_f32)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_is_transposed`](../ggml.c.driver.md#ggml_is_transposed)
    - [`ggml_vk_mul_mat_vec_nc_f16_f32`](#ggml_vk_mul_mat_vec_nc_f16_f32)
    - [`ggml_is_quantized`](../ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_vk_mul_mat_vec_q_f16`](#ggml_vk_mul_mat_vec_q_f16)
    - [`ggml_vk_mul_mat_q_f16`](#ggml_vk_mul_mat_q_f16)


---
### ggml\_vk\_mul\_mat\_id\_q\_f16<!-- {{#callable:ggml_vk_mul_mat_id_q_f16}} -->
Performs matrix multiplication of two tensors with an index tensor, utilizing Vulkan for GPU acceleration.
- **Inputs**:
    - `ctx`: Pointer to the Vulkan backend context, which holds device and memory management information.
    - `subctx`: Reference to a Vulkan context for managing command buffers and execution.
    - `src0`: Pointer to the first input tensor, which is involved in the matrix multiplication.
    - `src1`: Pointer to the second input tensor, which is involved in the matrix multiplication.
    - `ids`: Pointer to an index tensor that specifies which rows/columns to multiply.
    - `dst`: Pointer to the output tensor where the result of the multiplication will be stored.
    - `dryrun`: Boolean flag indicating whether to perform a dry run (true) or execute the multiplication (false).
- **Control Flow**:
    - Logs the details of the input tensors for debugging purposes.
    - Validates the types and dimensions of the input tensors using assertions.
    - Checks if the device supports Unified Memory Architecture (UMA) and retrieves data from host memory if necessary.
    - Determines if the input tensors are contiguous and sets up the appropriate Vulkan pipeline for matrix multiplication.
    - Handles the case for dry runs by checking memory allocation sizes without executing the multiplication.
    - Prepares the input tensors for multiplication, including potential dequantization and copying to contiguous memory.
    - Executes the matrix multiplication using the Vulkan pipeline and the specified parameters.
- **Output**: The function does not return a value but populates the `dst` tensor with the result of the matrix multiplication.
- **Functions called**:
    - [`ggml_vk_dim01_contiguous`](#ggml_vk_dim01_contiguous)
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`ggml_vk_get_mul_mat_mat_id_pipeline`](#ggml_vk_get_mul_mat_mat_id_pipeline)
    - [`ggml_vk_align_size`](#ggml_vk_align_size)
    - [`ggml_vk_guess_matmul_id_pipeline_align`](#ggml_vk_guess_matmul_id_pipeline_align)
    - [`ggml_vk_guess_matmul_id_pipeline`](#ggml_vk_guess_matmul_id_pipeline)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`ggml_vk_get_cpy_pipeline`](#ggml_vk_get_cpy_pipeline)
    - [`ggml_vk_get_to_fp16`](#ggml_vk_get_to_fp16)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_vk_cpy_to_contiguous`](#ggml_vk_cpy_to_contiguous)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)
    - [`ggml_vk_matmul_id`](#ggml_vk_matmul_id)


---
### ggml\_vk\_mul\_mat\_vec\_id\_q\_f16<!-- {{#callable:ggml_vk_mul_mat_vec_id_q_f16}} -->
Performs matrix-vector multiplication with an identity mapping using Vulkan for half-precision floating-point tensors.
- **Inputs**:
    - `ctx`: Pointer to the Vulkan backend context used for managing Vulkan resources.
    - `subctx`: Reference to a Vulkan context for managing sub-operations.
    - `src0`: Pointer to the first input tensor (matrix) for multiplication.
    - `src1`: Pointer to the second input tensor (vector) for multiplication.
    - `ids`: Pointer to a tensor containing indices for the operation.
    - `dst`: Pointer to the output tensor where the result will be stored.
    - `dryrun`: Boolean flag indicating whether to perform a dry run (true) or execute the operation (false).
- **Control Flow**:
    - Logs the details of the input tensors and the dry run status.
    - Validates the input tensor types and dimensions using assertions.
    - Checks if the input tensors are contiguous in memory and prepares Vulkan buffers accordingly.
    - Handles the case for dry run by calculating required memory sizes and requesting descriptor sets without executing the operation.
    - If not a dry run, prepares the Vulkan pipeline for the matrix-vector multiplication and handles any necessary data dequantization.
    - Dispatches the Vulkan compute pipeline to perform the matrix-vector multiplication using the prepared buffers and push constants.
- **Output**: The result of the matrix-vector multiplication is stored in the `dst` tensor.
- **Functions called**:
    - [`ggml_vk_dim01_contiguous`](#ggml_vk_dim01_contiguous)
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`ggml_vk_align_size`](#ggml_vk_align_size)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`ggml_vk_get_cpy_pipeline`](#ggml_vk_get_cpy_pipeline)
    - [`ggml_vk_get_to_fp16`](#ggml_vk_get_to_fp16)
    - [`ggml_vk_get_dequantize_mul_mat_vec_id`](#ggml_vk_get_dequantize_mul_mat_vec_id)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_vk_cpy_to_contiguous`](#ggml_vk_cpy_to_contiguous)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)


---
### ggml\_vk\_mul\_mat\_id<!-- {{#callable:ggml_vk_mul_mat_id}} -->
Multiplies two matrices or a matrix and a vector using Vulkan backend, with an option for a dry run.
- **Inputs**:
    - `ctx`: Pointer to the Vulkan backend context used for managing Vulkan resources.
    - `subctx`: Reference to a Vulkan context that provides additional context for Vulkan operations.
    - `src0`: Pointer to the first source tensor, which can be a matrix or a vector.
    - `src1`: Pointer to the second source tensor, which is typically a matrix.
    - `src2`: Pointer to the third source tensor, which is used in the multiplication operation.
    - `dst`: Pointer to the destination tensor where the result of the multiplication will be stored.
    - `dryrun`: Boolean flag indicating whether to perform a dry run (true) or execute the multiplication (false).
- **Control Flow**:
    - Logs the function call with the provided tensor pointers for debugging purposes.
    - Checks if the second dimension of `src2` is 1 and if `src0` is of type float or half-float or quantized.
    - If the condition is true, calls [`ggml_vk_mul_mat_vec_id_q_f16`](#ggml_vk_mul_mat_vec_id_q_f16) to perform matrix-vector multiplication.
    - If the condition is false, calls [`ggml_vk_mul_mat_id_q_f16`](#ggml_vk_mul_mat_id_q_f16) to perform matrix-matrix multiplication.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the result of the multiplication.
- **Functions called**:
    - [`ggml_is_quantized`](../ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_vk_mul_mat_vec_id_q_f16`](#ggml_vk_mul_mat_vec_id_q_f16)
    - [`ggml_vk_mul_mat_id_q_f16`](#ggml_vk_mul_mat_id_q_f16)


---
### ggml\_vk\_flash\_attn\_coopmat\_shmem\_support<!-- {{#callable:ggml_vk_flash_attn_coopmat_shmem_support}} -->
Determines if the shared memory size required for cooperative matrix flash attention is supported by the given Vulkan device.
- **Inputs**:
    - `device`: A reference to a `vk_device` object representing the Vulkan device to check for shared memory support.
    - `D`: A `uint32_t` representing the dimensionality of the data being processed.
    - `f32acc`: A boolean indicating whether to use 32-bit floating point accumulation.
- **Control Flow**:
    - Calculates various sizes based on the workgroup size, number of large rows, and other parameters.
    - Determines the total shared memory size required for the operation.
    - Compares the calculated total size against the maximum compute shared memory size supported by the device.
    - Logs the parameters and the result of the support check for debugging purposes.
- **Output**: Returns a boolean indicating whether the required shared memory size is supported by the Vulkan device.


---
### ggml\_vk\_flash\_attn<!-- {{#callable:ggml_vk_flash_attn}} -->
The `ggml_vk_flash_attn` function performs flash attention computation using Vulkan for given query, key, value tensors, and an optional mask, storing the result in a destination tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the current operation.
    - `q`: A pointer to a `ggml_tensor` representing the query tensor.
    - `k`: A pointer to a `ggml_tensor` representing the key tensor.
    - `v`: A pointer to a `ggml_tensor` representing the value tensor.
    - `mask`: An optional pointer to a `ggml_tensor` representing the mask tensor, which can be used to control attention.
    - `dst`: A pointer to a `ggml_tensor` where the result of the attention computation will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (true) or execute the actual computation (false).
- **Control Flow**:
    - The function begins by logging the details of the input tensors and the dry run status.
    - It extracts tensor dimensions and checks for various assertions to ensure the validity of the input tensors.
    - Based on the device capabilities and tensor dimensions, it determines the appropriate computation path (scalar, coopmat1, or coopmat2).
    - It calculates workgroup sizes and checks for alignment requirements for optimal performance.
    - If `dryrun` is true, it requests descriptor sets and returns without performing the computation.
    - It prepares the necessary parameters and buffers for the Vulkan pipeline dispatch.
    - Finally, it dispatches the Vulkan pipeline to perform the attention computation, handling both split-k and non-split-k cases.
- **Output**: The function does not return a value directly; instead, it writes the computed attention results into the `dst` tensor.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_vk_flash_attn_coopmat_shmem_support`](#ggml_vk_flash_attn_coopmat_shmem_support)
    - [`get_fa_num_small_rows`](#get_fa_num_small_rows)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)


---
### ggml\_vk\_op\_get\_pipeline<!-- {{#callable:ggml_vk_op_get_pipeline}} -->
Retrieves the appropriate Vulkan pipeline based on the specified operation and tensor types.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure, which contains device-specific information.
    - `src0`: A pointer to the first source tensor of type `ggml_tensor`.
    - `src1`: A pointer to the second source tensor of type `ggml_tensor`, can be null for some operations.
    - `src2`: A pointer to the third source tensor of type `ggml_tensor`, not used in the function.
    - `dst`: A pointer to the destination tensor of type `ggml_tensor`.
    - `op`: An enumeration value of type `ggml_op` that specifies the operation to be performed.
- **Control Flow**:
    - The function begins by checking the operation type specified by `op` using a switch statement.
    - For each case, it verifies the types of the source and destination tensors to ensure compatibility.
    - If the types are compatible, it retrieves the corresponding Vulkan pipeline from the context's device based on the operation and tensor types.
    - If the operation or tensor types do not match expected values, the function returns nullptr.
- **Output**: Returns a `vk_pipeline` object corresponding to the specified operation and tensor types, or nullptr if no valid pipeline is found.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_vk_get_cpy_pipeline`](#ggml_vk_get_cpy_pipeline)
    - [`ggml_get_unary_op`](../ggml.c.driver.md#ggml_get_unary_op)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_is_contiguous_channels`](../ggml.c.driver.md#ggml_is_contiguous_channels)


---
### ggml\_vk\_op\_supports\_incontiguous<!-- {{#callable:ggml_vk_op_supports_incontiguous}} -->
Determines if a given `ggml_op` operation supports non-contiguous memory access.
- **Inputs**:
    - `op`: An enumeration value of type `ggml_op` representing the operation to check for support of non-contiguous memory access.
- **Control Flow**:
    - The function uses a `switch` statement to evaluate the value of the input `op`.
    - If `op` matches any of the predefined operations that support non-contiguous access, the function returns `true`.
    - If `op` does not match any of the cases, the function returns `false`.
- **Output**: Returns a boolean value indicating whether the specified operation supports non-contiguous memory access.


---
### get\_misalign\_bytes<!-- {{#callable:get_misalign_bytes}} -->
Calculates the number of misaligned bytes for a given tensor in a Vulkan backend context.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the Vulkan context and device properties.
    - `t`: A pointer to a `ggml_tensor` structure representing the tensor for which misalignment is being calculated.
- **Control Flow**:
    - The function retrieves the offset of the tensor using `vk_tensor_offset(t)`.
    - It adds the tensor's view offset `t->view_offs` to the retrieved offset.
    - The sum is then bitwise ANDed with the minimum storage buffer offset alignment minus one, obtained from `ctx->device->properties.limits.minStorageBufferOffsetAlignment`.
- **Output**: Returns a `uint32_t` value representing the number of misaligned bytes for the specified tensor.
- **Functions called**:
    - [`vk_tensor_offset`](#vk_tensor_offset)


---
### init\_pushconst\_tensor\_offsets<!-- {{#callable:init_pushconst_tensor_offsets}} -->
Initializes push constant offsets for tensor operations in a Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `p`: A reference to a `vk_op_upscale_push_constants` structure where the computed offsets will be stored.
    - `src0`: A pointer to the first source `ggml_tensor` whose misalignment offset is calculated.
    - `src1`: A pointer to the second source `ggml_tensor`, which is unused in this function.
    - `src2`: A pointer to the third source `ggml_tensor`, which is also unused in this function.
    - `dst`: A pointer to the destination `ggml_tensor` whose misalignment offset is calculated.
- **Control Flow**:
    - Calculates the misalignment offset for `src0` using the [`get_misalign_bytes`](#get_misalign_bytes) function and divides it by the size of the tensor type to get `a_offset`.
    - Calculates the misalignment offset for `dst` in a similar manner to get `d_offset`.
    - Assigns the calculated offsets `a_offset` and `d_offset` to the respective fields in the `vk_op_upscale_push_constants` structure.
    - The `src1` and `src2` tensors are marked as unused to avoid compiler warnings.
- **Output**: This function does not return a value; instead, it modifies the `vk_op_upscale_push_constants` structure to include the calculated offsets.
- **Functions called**:
    - [`get_misalign_bytes`](#get_misalign_bytes)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)


---
### ggml\_vk\_op\_f32<!-- {{#callable:ggml_vk_op_f32}} -->
The `ggml_vk_op_f32` function performs a specified operation on tensors using Vulkan, handling various tensor configurations and ensuring proper memory management.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure, which contains the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan command buffer context.
    - `src0`: A pointer to the first source tensor (`ggml_tensor`) used in the operation.
    - `src1`: A pointer to the second source tensor (`ggml_tensor`), which can be null if not used.
    - `src2`: A pointer to the third source tensor (`ggml_tensor`), which can be null if not used.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the result will be stored.
    - `op`: An enumeration value of type `ggml_op` that specifies the operation to be performed.
    - `pc`: A parameter pack that allows for passing additional parameters to the operation.
    - `dryrun`: A boolean flag indicating whether to perform a dry run without executing the operation.
- **Control Flow**:
    - Logs the details of the input tensors and the operation being performed.
    - Validates the operation type and checks for tensor compatibility.
    - Calculates the necessary tensor sizes and offsets based on the input tensors.
    - Retrieves the appropriate Vulkan pipeline for the specified operation.
    - Handles special cases for dry runs and ensures proper memory management for the source and destination tensors.
    - Dispatches the Vulkan pipeline with the appropriate buffers and parameters based on the operation type.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place based on the specified operation.
- **Functions called**:
    - [`ggml_op_name`](../ggml.c.driver.md#ggml_op_name)
    - [`ggml_is_quantized`](../ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_vk_op_supports_incontiguous`](#ggml_vk_op_supports_incontiguous)
    - [`ggml_vk_dim01_contiguous`](#ggml_vk_dim01_contiguous)
    - [`init_pushconst_fastdiv`](#init_pushconst_fastdiv)
    - [`ggml_vk_op_get_pipeline`](#ggml_vk_op_get_pipeline)
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`init_pushconst_tensor_offsets`](#init_pushconst_tensor_offsets)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)
    - [`ggml_vk_buffer_memset_async`](#ggml_vk_buffer_memset_async)


---
### ggml\_vk\_get\_rows<!-- {{#callable:ggml_vk_get_rows}} -->
The `ggml_vk_get_rows` function retrieves specific rows from two source tensors and stores the result in a destination tensor using Vulkan backend operations.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to the first source `ggml_tensor` from which rows will be retrieved.
    - `src1`: A pointer to the second source `ggml_tensor` that may be used in conjunction with `src0`.
    - `dst`: A pointer to the destination `ggml_tensor` where the retrieved rows will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without actual execution).
- **Control Flow**:
    - The function begins by determining the size of each tensor type using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size) for `src0`, `src1`, and `dst`.
    - It then calls the `ggml_vk_op_f32` function, passing the Vulkan context, source tensors, destination tensor, operation type `GGML_OP_GET_ROWS`, and a list of parameters including tensor dimensions and strides.
    - The parameters are calculated based on the number of elements and the number of bytes for each dimension of the tensors.
    - Finally, the function executes the Vulkan operation, which retrieves the specified rows from the source tensors and stores them in the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the retrieved rows from the source tensors.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_acc<!-- {{#callable:ggml_vk_acc}} -->
The `ggml_vk_acc` function performs an accumulation operation on two source tensors and stores the result in a destination tensor using Vulkan backend operations.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to the first source `ggml_tensor` that will be used in the accumulation operation.
    - `src1`: A pointer to the second source `ggml_tensor` that will be used in the accumulation operation.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the accumulation will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation will not be executed).
- **Control Flow**:
    - The function begins by determining the size of each tensor type using the [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size) function for `src0`, `src1`, and `dst`.
    - It calculates the number of elements and offsets required for the destination tensor based on its operation parameters.
    - The `ggml_vk_op_f32` function is called with the Vulkan operation type `vk_op_binary_push_constants`, passing the context, source tensors, destination tensor, and various calculated parameters.
    - If `dryrun` is true, the operation is prepared but not executed, allowing for validation of parameters without performing the actual computation.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place with the result of the accumulation operation, or prepares for the operation if `dryrun` is true.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_add<!-- {{#callable:ggml_vk_add}} -->
The `ggml_vk_add` function performs an element-wise addition of two tensors using Vulkan for GPU acceleration.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to the first `ggml_tensor` that serves as the source operand for the addition.
    - `src1`: A pointer to the second `ggml_tensor` that serves as the source operand for the addition.
    - `dst`: A pointer to the `ggml_tensor` that will store the result of the addition.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function begins by determining the size of each tensor type using the [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size) function for `src0`, `src1`, and `dst`.
    - It then calls the `ggml_vk_op_f32` function, passing the Vulkan context, source tensors, destination tensor, operation type (addition), and various tensor properties.
    - The properties include the number of elements and the dimensions of each tensor, calculated from the tensor's `ne` and `nb` attributes.
    - If `dryrun` is true, the operation is simulated without performing the actual addition.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to hold the result of the addition of `src0` and `src1`.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_sub<!-- {{#callable:ggml_vk_sub}} -->
The `ggml_vk_sub` function performs a subtraction operation on two source tensors and stores the result in a destination tensor using Vulkan backend operations.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to the first `ggml_tensor` (source tensor) from which values will be subtracted.
    - `src1`: A pointer to the second `ggml_tensor` (source tensor) that will be subtracted from `src0`.
    - `dst`: A pointer to the `ggml_tensor` where the result of the subtraction will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function begins by determining the size of each tensor type using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size) for `src0`, `src1`, and `dst`.
    - It then calls the `ggml_vk_op_f32` function, passing the Vulkan context, source tensors, destination tensor, operation type (subtraction), and various tensor properties.
    - The properties include the number of elements and the dimensions of each tensor, calculated from their respective `ne` and `nb` attributes.
    - The function concludes by executing the Vulkan operation, unless `dryrun` is set to true, in which case it simulates the operation.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the result of the subtraction operation.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_mul<!-- {{#callable:ggml_vk_mul}} -->
Multiplies two tensors (`src0` and `src1`) and stores the result in a destination tensor (`dst`) using Vulkan backend operations.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context (`ggml_backend_vk_context`) used for managing Vulkan operations.
    - `subctx`: A reference to a Vulkan context (`vk_context`) that provides additional context for the operation.
    - `src0`: A pointer to the first source tensor (`ggml_tensor`) to be multiplied.
    - `src1`: A pointer to the second source tensor (`ggml_tensor`) to be multiplied.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the result of the multiplication will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without actual execution).
- **Control Flow**:
    - Calculates the type sizes of the source and destination tensors using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size).
    - Calls the `ggml_vk_op_f32` function to perform the multiplication operation, passing the necessary parameters including tensor dimensions and offsets.
    - The operation is executed in the Vulkan context, and if `dryrun` is true, the operation is simulated without actual execution.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the result of the multiplication of `src0` and `src1`.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_div<!-- {{#callable:ggml_vk_div}} -->
The `ggml_vk_div` function performs element-wise division of two tensors using Vulkan for GPU acceleration.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to the `vk_context` that represents the current Vulkan context.
    - `src0`: A pointer to the first source `ggml_tensor` which is the dividend.
    - `src1`: A pointer to the second source `ggml_tensor` which is the divisor.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the division will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function begins by determining the size of each tensor type using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size) for `src0`, `src1`, and `dst`.
    - It then calls the `ggml_vk_op_f32` function, passing the context, source tensors, destination tensor, operation type (division), and various tensor properties.
    - The properties include the number of elements and the dimensions of each tensor, calculated from their respective `ne` and `nb` attributes.
    - If `dryrun` is true, the operation is simulated without performing the actual division.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to store the result of the division operation.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_op\_f32\_wkv<!-- {{#callable:ggml_vk_op_f32_wkv}} -->
Executes a Vulkan operation for processing tensors based on the specified version and push constants.
- **Inputs**:
    - `ctx`: Pointer to the Vulkan backend context used for managing Vulkan resources.
    - `subctx`: Reference to a Vulkan context for executing commands.
    - `dst`: Pointer to the destination tensor where the result will be stored.
    - `pc`: Push constants containing parameters for the Vulkan operation.
    - `version`: Integer indicating the version of the operation (6 or 7).
    - `dryrun`: Boolean flag indicating whether to perform a dry run without executing the operation.
- **Control Flow**:
    - Asserts that the version is either 6 or 7 and determines the number of source tensors based on the version.
    - Validates that the destination tensor is not quantized and that its buffer is not null.
    - Retrieves the Vulkan pipeline for the operation based on the source and destination tensors.
    - If 'dryrun' is true, requests descriptor sets and exits the function without executing the operation.
    - Synchronizes buffers and prepares device buffers for the source and destination tensors.
    - Calculates the sizes and offsets for the source tensors and the destination tensor.
    - Dispatches the Vulkan pipeline with the appropriate parameters based on the version.
- **Output**: The function does not return a value; it performs operations directly on the GPU using Vulkan.
- **Functions called**:
    - [`ggml_is_quantized`](../ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_vk_op_get_pipeline`](#ggml_vk_op_get_pipeline)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)


---
### ggml\_vk\_rwkv\_wkv6<!-- {{#callable:ggml_vk_rwkv_wkv6}} -->
The `ggml_vk_rwkv_wkv6` function performs a specific operation on a tensor using Vulkan backend context.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor for the operation.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function begins by extracting the sequence length, number of embeddings, number of heads, and number of sequences from the `dst` tensor and its source tensors.
    - It then calls the [`ggml_vk_op_f32_wkv`](#ggml_vk_op_f32_wkv) function, passing the Vulkan context, destination tensor, and a set of parameters including the number of sequences, sequence length, number of embeddings, and number of heads.
    - The function also specifies the operation type (6) and the dry run flag.
- **Output**: The function does not return a value; it modifies the state of the destination tensor based on the Vulkan operation performed.
- **Functions called**:
    - [`ggml_vk_op_f32_wkv`](#ggml_vk_op_f32_wkv)


---
### ggml\_vk\_rwkv\_wkv7<!-- {{#callable:ggml_vk_rwkv_wkv7}} -->
The `ggml_vk_rwkv_wkv7` function performs a GPU operation for processing tensor data in a specific format.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context for GPU operations.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the current operation.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor for the operation.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function retrieves the sequence length, number of embeddings, number of heads, and number of sequences from the `dst` tensor and its source tensors.
    - It then calls the [`ggml_vk_op_f32_wkv`](#ggml_vk_op_f32_wkv) function, passing the Vulkan context, destination tensor, and relevant parameters including the dimensions of the tensor and the dry run flag.
- **Output**: The function does not return a value; it performs an operation on the GPU that modifies the state of the destination tensor based on the input parameters.
- **Functions called**:
    - [`ggml_vk_op_f32_wkv`](#ggml_vk_op_f32_wkv)


---
### ggml\_vk\_op\_f32\_opt\_step\_adamw<!-- {{#callable:ggml_vk_op_f32_opt_step_adamw}} -->
Executes the AdamW optimization step for a given set of tensors in a Vulkan backend context.
- **Inputs**:
    - `ctx`: Pointer to the Vulkan backend context used for managing Vulkan resources.
    - `subctx`: Reference to a Vulkan context for executing commands.
    - `dst`: Pointer to the destination tensor that holds the source tensors for the optimization.
    - `pc`: Push constants for the Vulkan pipeline, passed as an rvalue reference.
    - `dryrun`: Boolean flag indicating whether to perform a dry run without executing the optimization.
- **Control Flow**:
    - Asserts the types and shapes of the input tensors to ensure they are valid for the operation.
    - Retrieves the Vulkan pipeline for the AdamW optimization step.
    - If 'dryrun' is true, requests descriptor sets and exits without further processing.
    - Synchronizes the buffers in the Vulkan context.
    - Checks if the tensors are using Unified Memory Architecture (UMA) and retrieves device buffers accordingly.
    - Calculates the sizes of the input tensors.
    - Dispatches the Vulkan pipeline with the appropriate buffers and parameters.
- **Output**: The function does not return a value; it performs operations directly on the Vulkan context and the specified tensors.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_vk_op_get_pipeline`](#ggml_vk_op_get_pipeline)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`ggml_vk_sync_buffers`](#ggml_vk_sync_buffers)
    - [`ggml_vk_host_get`](#ggml_vk_host_get)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)


---
### ggml\_vk\_opt\_step\_adamw<!-- {{#callable:ggml_vk_opt_step_adamw}} -->
The `ggml_vk_opt_step_adamw` function performs an optimization step using the AdamW algorithm on a given tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor to be optimized.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without actual execution).
- **Control Flow**:
    - The function begins by determining the number of elements `n` in the source tensor of `dst` using [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements).
    - It then calls the [`ggml_vk_op_f32_opt_step_adamw`](#ggml_vk_op_f32_opt_step_adamw) function, passing the Vulkan context, the destination tensor, and a configuration structure containing the number of elements and two float values initialized to zero.
    - The `dryrun` flag is also passed to indicate whether the operation should be executed or simulated.
- **Output**: The function does not return a value; instead, it modifies the state of the `dst` tensor based on the AdamW optimization step.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_vk_op_f32_opt_step_adamw`](#ggml_vk_op_f32_opt_step_adamw)


---
### ggml\_vk\_concat<!-- {{#callable:ggml_vk_concat}} -->
The `ggml_vk_concat` function concatenates two tensors (`src0` and `src1`) into a destination tensor (`dst`) using Vulkan backend operations.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context (`ggml_backend_vk_context`) used for managing Vulkan resources.
    - `subctx`: A reference to a Vulkan context (`vk_context`) that provides additional context for Vulkan operations.
    - `src0`: A pointer to the first source tensor (`ggml_tensor`) to be concatenated.
    - `src1`: A pointer to the second source tensor (`ggml_tensor`) to be concatenated.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the concatenated result will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without actual execution).
- **Control Flow**:
    - The function begins by retrieving the operation parameters from the destination tensor's `op_params` array.
    - It calculates the size of each tensor type using the [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size) function for `src0`, `src1`, and `dst`.
    - The function then calls `ggml_vk_op_f32` with the Vulkan operation type `vk_op_binary_push_constants`, passing the context, source tensors, and destination tensor along with their respective dimensions and sizes.
    - The parameters for the concatenation operation are prepared, including the number of elements and the dimensions of the source and destination tensors.
    - Finally, the function executes the Vulkan operation, unless `dryrun` is set to true, in which case it simulates the operation.
- **Output**: The function does not return a value; instead, it modifies the destination tensor (`dst`) to contain the concatenated result of the two source tensors (`src0` and `src1`).
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_upscale<!-- {{#callable:ggml_vk_upscale}} -->
The `ggml_vk_upscale` function performs an upscale operation on a source tensor and stores the result in a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to a `ggml_tensor` structure representing the source tensor to be upscaled.
    - `dst`: A pointer to a `ggml_tensor` structure where the upscaled result will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without actual execution).
- **Control Flow**:
    - The function begins by calculating the size of the data type of the source tensor `src0` using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size).
    - It then computes the scaling factors for each dimension of the source tensor based on the dimensions of the destination tensor.
    - The function calls `ggml_vk_op_f32` with the appropriate parameters, including the calculated scaling factors and tensor sizes, to perform the upscale operation.
    - If `dryrun` is true, the operation is simulated without executing the actual upscale.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place with the upscaled data.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_scale<!-- {{#callable:ggml_vk_scale}} -->
The `ggml_vk_scale` function performs a scaling operation on a source tensor and stores the result in a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context (`ggml_backend_vk_context`) used for managing Vulkan resources.
    - `subctx`: A reference to a Vulkan context (`vk_context`) that provides additional context for Vulkan operations.
    - `src0`: A pointer to the source tensor (`ggml_tensor`) that will be scaled.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the scaled result will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without actual execution).
- **Control Flow**:
    - The function begins by retrieving the operation parameters from the destination tensor's `op_params` array.
    - It calculates the size of the data types for both the source and destination tensors using the [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size) function.
    - The function then calls `ggml_vk_op_f32` to perform the scaling operation, passing various parameters including the number of elements and dimensions of the source and destination tensors.
    - The scaling operation is configured with specific parameters, including the scaling factor and other operational settings, and is executed unless `dryrun` is true.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place with the scaled results from the source tensor.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_sqr<!-- {{#callable:ggml_vk_sqr}} -->
The `ggml_vk_sqr` function computes the square of elements in a source tensor and stores the result in a destination tensor using Vulkan backend operations.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to a `ggml_tensor` structure representing the source tensor whose elements will be squared.
    - `dst`: A pointer to a `ggml_tensor` structure where the squared results will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function begins by determining the size of the data types for the source and destination tensors using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size).
    - It then calls the `ggml_vk_op_f32` function, passing the Vulkan context, source tensor, and destination tensor along with various parameters related to the tensor dimensions and sizes.
    - The parameters include the number of elements in the source tensor and the respective dimensions and byte sizes for both the source and destination tensors.
    - The operation type is specified as `GGML_OP_SQR`, indicating that the square operation is to be performed.
    - If `dryrun` is true, the operation is simulated without actual execution.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place to contain the squared values of the source tensor's elements.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_sin<!-- {{#callable:ggml_vk_sin}} -->
The `ggml_vk_sin` function computes the sine of elements in a source tensor and stores the result in a destination tensor using Vulkan backend operations.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to a `ggml_tensor` structure that contains the input tensor whose elements will be processed.
    - `dst`: A pointer to a `ggml_tensor` structure that will store the output tensor containing the sine results.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function begins by determining the size of the data types for the source and destination tensors using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size).
    - It then calls the `ggml_vk_op_f32` function, passing the Vulkan context, source tensor, and destination tensor along with various parameters related to the tensor's dimensions and sizes.
    - The parameters include the number of elements in the source tensor, its dimensions, and the corresponding sizes for both source and destination tensors.
    - The operation type is specified as `GGML_OP_SIN`, indicating that the sine function will be applied to the elements of the source tensor.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place to contain the sine of the elements from the source tensor.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_cos<!-- {{#callable:ggml_vk_cos}} -->
The `ggml_vk_cos` function computes the cosine of elements in a source tensor and stores the result in a destination tensor using Vulkan backend operations.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to a `ggml_tensor` structure that contains the source tensor whose elements will be processed.
    - `dst`: A pointer to a `ggml_tensor` structure that will store the result of the cosine operation.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function begins by determining the size of the data types for the source and destination tensors using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size).
    - It then calls the `ggml_vk_op_f32` function, passing the Vulkan context, source tensor, and destination tensor along with various parameters related to the tensor's dimensions and sizes.
    - The parameters include the number of elements in the source tensor and the respective dimensions and byte sizes for both the source and destination tensors.
    - The operation type is specified as `GGML_OP_COS`, indicating that the cosine function will be applied to the elements of the source tensor.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place to contain the cosine of the elements from the source tensor.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_clamp<!-- {{#callable:ggml_vk_clamp}} -->
The `ggml_vk_clamp` function performs a clamping operation on a source tensor and stores the result in a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to the source `ggml_tensor` that contains the input data to be clamped.
    - `dst`: A pointer to the destination `ggml_tensor` where the clamped result will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without actual execution).
- **Control Flow**:
    - The function begins by retrieving the operation parameters from the destination tensor's `op_params` array.
    - It calculates the size of the data types for both the source and destination tensors using the [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size) function.
    - The `ggml_vk_op_f32` function is called to execute the clamping operation, passing the necessary parameters including the source tensor, destination tensor, and various tensor dimensions and strides.
    - The clamping operation is defined by the `GGML_OP_CLAMP` constant, and the function handles the dry run condition appropriately.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place with the clamped results derived from the source tensor `src0`.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_pad<!-- {{#callable:ggml_vk_pad}} -->
The `ggml_vk_pad` function performs padding operations on a source tensor and writes the result to a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to the source `ggml_tensor` that will be padded.
    - `dst`: A pointer to the destination `ggml_tensor` where the padded result will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without actual execution).
- **Control Flow**:
    - The function begins by determining the size of the data types for both the source and destination tensors using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size).
    - It then calls the `ggml_vk_op_f32` function, passing the Vulkan context, source tensor, and destination tensor along with various parameters related to the tensor's dimensions and sizes.
    - The parameters include the number of elements in the destination tensor and the dimensions and byte sizes of both the source and destination tensors.
    - The function also includes several zero-initialized parameters, likely reserved for future use or specific padding configurations.
    - The `dryrun` flag is passed to indicate whether the operation should be executed or just simulated.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place to contain the padded data based on the source tensor.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_repeat<!-- {{#callable:ggml_vk_repeat}} -->
The `ggml_vk_repeat` function performs a repeat operation on a source tensor and stores the result in a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context (`ggml_backend_vk_context`) used for managing Vulkan resources.
    - `subctx`: A reference to a Vulkan context (`vk_context`) that provides the necessary environment for executing Vulkan commands.
    - `src0`: A pointer to the source tensor (`ggml_tensor`) that contains the data to be repeated.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the repeated data will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without actual execution).
- **Control Flow**:
    - The function begins by determining the size of the data types for the source and destination tensors using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size).
    - It then calls the `ggml_vk_op_f32` function, passing the Vulkan context, source tensor, and destination tensor along with various parameters related to the tensor dimensions and sizes.
    - The parameters include the number of elements in the destination tensor and the dimensions and byte sizes of both the source and destination tensors.
    - The function also includes several zeroed parameters and constants, which may be placeholders for future use or specific operation configurations.
    - Finally, the `dryrun` flag is passed to indicate whether the operation should be executed or just simulated.
- **Output**: The function does not return a value; instead, it performs an operation that modifies the destination tensor in place based on the source tensor's data.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_repeat\_back<!-- {{#callable:ggml_vk_repeat_back}} -->
The `ggml_vk_repeat_back` function performs a repeat-back operation on a source tensor and stores the result in a destination tensor using Vulkan backend operations.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to the source `ggml_tensor` from which data will be repeated.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the repeat-back operation will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation will not be executed).
- **Control Flow**:
    - The function begins by determining the size of the data types for both the source and destination tensors using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size).
    - It then calls the `ggml_vk_op_f32` function, passing the Vulkan context, source tensor, and destination tensor along with various parameters related to the tensor dimensions and strides.
    - The parameters include the number of elements in the destination tensor and the dimensions and byte offsets of both the source and destination tensors.
    - The operation type is specified as `GGML_OP_REPEAT_BACK`, and the dry run flag is passed to control whether the operation is executed.
- **Output**: The function does not return a value; instead, it performs the repeat-back operation on the destination tensor in place, or simulates the operation if `dryrun` is true.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_cpy<!-- {{#callable:ggml_vk_cpy}} -->
Copies data from a source tensor to a destination tensor using Vulkan operations.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to a `ggml_tensor` structure representing the source tensor from which data will be copied.
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor to which data will be copied.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation will not execute the copy).
- **Control Flow**:
    - Calculates the size of the source and destination tensor types using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size).
    - Determines the number of elements in the source tensor using [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements).
    - If both source and destination tensors are quantized, adjusts the number of elements based on the block size and type size.
    - Calls the `ggml_vk_op_f32` function to perform the copy operation, passing the necessary parameters including tensor dimensions and offsets.
    - The operation can be executed or skipped based on the `dryrun` flag.
- **Output**: The function does not return a value; it performs a copy operation on the GPU, modifying the destination tensor in place.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_is_quantized`](../ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)


---
### ggml\_vk\_silu\_back<!-- {{#callable:ggml_vk_silu_back}} -->
The `ggml_vk_silu_back` function performs a backward pass of the Sigmoid Linear Unit (SiLU) operation in a Vulkan backend context.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to the first `ggml_tensor` which is the input tensor for the backward operation.
    - `src1`: A pointer to the second `ggml_tensor` which is used in the backward operation.
    - `dst`: A pointer to the `ggml_tensor` where the result of the backward operation will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation will not execute but will validate parameters).
- **Control Flow**:
    - The function calls `ggml_vk_op_f32` with the specified parameters to execute the backward operation.
    - It passes the operation type `GGML_OP_SILU_BACK` to indicate that this is a backward pass for the SiLU function.
    - The function also provides the number of elements in `src0` as part of the operation parameters.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor to contain the result of the backward SiLU operation.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_norm<!-- {{#callable:ggml_vk_norm}} -->
The `ggml_vk_norm` function performs a normalization operation on a source tensor and stores the result in a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to the source `ggml_tensor` that contains the data to be normalized.
    - `dst`: A pointer to the destination `ggml_tensor` where the normalized result will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation will not be executed).
- **Control Flow**:
    - The function begins by casting the `op_params` of the destination tensor to a float pointer to access operation parameters.
    - It then calls the `ggml_vk_op_f32` function, passing the Vulkan context, source tensor, and destination tensor along with specific parameters for the normalization operation.
    - The parameters include the dimensions of the source tensor and the first operation parameter from `op_params`, with a default value of 0.0f for the second parameter.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place with the normalized data.


---
### ggml\_vk\_group\_norm<!-- {{#callable:ggml_vk_group_norm}} -->
The `ggml_vk_group_norm` function performs group normalization on a source tensor and stores the result in a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context (`ggml_backend_vk_context`) used for managing Vulkan resources.
    - `subctx`: A reference to a Vulkan context (`vk_context`) that provides additional context for Vulkan operations.
    - `src0`: A pointer to the source tensor (`ggml_tensor`) that contains the input data to be normalized.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the normalized output will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the function will not execute the normalization but will prepare for it).
- **Control Flow**:
    - The function retrieves operation parameters from the destination tensor's `op_params`, interpreting them as integers and floats.
    - It calculates the number of groups and the group size based on the dimensions of the source tensor and the first operation parameter.
    - The function then calls `ggml_vk_op_f32` to execute the group normalization operation, passing the calculated parameters and the dry run flag.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place with the normalized data.


---
### ggml\_vk\_rms\_norm<!-- {{#callable:ggml_vk_rms_norm}} -->
The `ggml_vk_rms_norm` function computes the root mean square normalization of a source tensor and stores the result in a destination tensor using Vulkan backend operations.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context (`ggml_backend_vk_context`) used for managing Vulkan resources.
    - `subctx`: A reference to a Vulkan context (`vk_context`) that provides additional context for Vulkan operations.
    - `src0`: A pointer to the source tensor (`ggml_tensor`) that will be normalized.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the result of the normalization will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without actual execution).
- **Control Flow**:
    - The function begins by retrieving the operation parameters from the destination tensor's `op_params` array.
    - It calculates the type sizes for both the source and destination tensors using the [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size) function.
    - The function then calls `ggml_vk_op_f32` to perform the RMS normalization operation, passing the necessary parameters including the dimensions and strides of the source and destination tensors.
    - The `dryrun` flag is passed to determine if the operation should be executed or just simulated.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place to contain the result of the RMS normalization operation.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_rms\_norm\_back<!-- {{#callable:ggml_vk_rms_norm_back}} -->
The `ggml_vk_rms_norm_back` function performs a backward pass of RMS normalization in a Vulkan context.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to the `vk_context` that represents the Vulkan command buffer context for executing operations.
    - `src0`: A pointer to the first `ggml_tensor` which serves as the source tensor for the backward operation.
    - `src1`: A pointer to the second `ggml_tensor` which is used in conjunction with `src0` for the backward operation.
    - `dst`: A pointer to the `ggml_tensor` where the result of the backward operation will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation will not be executed).
- **Control Flow**:
    - The function begins by casting the `op_params` of the destination tensor `dst` to a float pointer.
    - It then calls the `ggml_vk_op_f32` function, passing the Vulkan context, source tensors, and operation parameters to execute the RMS normalization backward operation.
    - The operation type is specified as `GGML_OP_RMS_NORM_BACK`, and the parameters include the dimensions of `src0` and the first element of `op_params`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the results of the backward RMS normalization operation.


---
### ggml\_vk\_l2\_norm<!-- {{#callable:ggml_vk_l2_norm}} -->
Calculates the L2 norm of a tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context (`ggml_backend_vk_context`) used for managing Vulkan resources.
    - `subctx`: A reference to a Vulkan context (`vk_context`) that provides additional context for Vulkan operations.
    - `src0`: A pointer to the source tensor (`ggml_tensor`) from which the L2 norm is computed.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the result of the L2 norm will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - Extracts operation parameters from the destination tensor's `op_params` array.
    - Calls the `ggml_vk_op_f32` function to perform the L2 norm operation, passing the necessary parameters including the source tensor, destination tensor, and operation type.
- **Output**: The function does not return a value; instead, it modifies the destination tensor to contain the result of the L2 norm operation.


---
### ggml\_vk\_unary<!-- {{#callable:ggml_vk_unary}} -->
The `ggml_vk_unary` function performs a unary operation on a source tensor and stores the result in a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to the source `ggml_tensor` from which the unary operation will be applied.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the unary operation will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function calls `ggml_vk_op_f32` with the provided context and tensors.
    - It passes the operation type `GGML_OP_UNARY` along with the number of elements in the source tensor and additional parameters set to default values.
    - If `dryrun` is true, the operation is simulated without executing the actual computation.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place with the result of the unary operation applied to `src0`.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_diag\_mask\_inf<!-- {{#callable:ggml_vk_diag_mask_inf}} -->
The `ggml_vk_diag_mask_inf` function applies a diagonal mask operation to a source tensor and stores the result in a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to a `ggml_tensor` structure that serves as the source tensor for the diagonal mask operation.
    - `dst`: A pointer to a `ggml_tensor` structure where the result of the operation will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function begins by extracting the operation parameters from the destination tensor's `op_params` field.
    - It then calls the `ggml_vk_op_f32` function, passing the Vulkan context, source tensor, and other parameters to perform the diagonal mask operation.
    - The operation type is specified as `GGML_OP_DIAG_MASK_INF`, and the dimensions of the source tensor are included in the parameters.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place with the results of the diagonal mask operation.


---
### ggml\_vk\_soft\_max<!-- {{#callable:ggml_vk_soft_max}} -->
The `ggml_vk_soft_max` function computes the softmax operation on a given tensor using Vulkan backend, with optional parameters for scaling and bias.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context used for managing Vulkan resources.
    - `subctx`: A reference to a Vulkan context that provides additional context for the operation.
    - `src0`: A pointer to the source tensor on which the softmax operation is applied.
    - `src1`: An optional pointer to a second source tensor, which may influence the operation.
    - `dst`: A pointer to the destination tensor where the result of the softmax operation will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is not executed).
- **Control Flow**:
    - The function begins by extracting operation parameters from the destination tensor's `op_params` array.
    - It calculates the number of columns and rows for the source tensor `src0`.
    - The number of heads for key-value pairs and their logarithmic representation are computed based on the dimensions of `src0`.
    - Two scaling factors, `m0` and `m1`, are calculated using the maximum bias and the logarithmic head count.
    - Finally, the function calls `ggml_vk_op_f32` to perform the softmax operation, passing the necessary parameters and the computed values.
- **Output**: The function does not return a value directly; instead, it populates the `dst` tensor with the results of the softmax operation.
- **Functions called**:
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)


---
### ggml\_vk\_soft\_max\_back<!-- {{#callable:ggml_vk_soft_max_back}} -->
The `ggml_vk_soft_max_back` function performs a backward operation for the softmax function in a Vulkan context.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to the `vk_context` that represents the Vulkan command context.
    - `src0`: A pointer to the first `ggml_tensor` which is the input tensor for the backward operation.
    - `src1`: A pointer to the second `ggml_tensor` which is used in the backward operation.
    - `dst`: A pointer to the `ggml_tensor` where the result of the backward operation will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation will not be executed).
- **Control Flow**:
    - The function begins by casting the `op_params` of the destination tensor `dst` to a float pointer.
    - It then calls the `ggml_vk_op_f32` function, passing the context, subcontext, source tensors, and operation parameters to perform the softmax backward operation.
    - The operation type is specified as `GGML_OP_SOFT_MAX_BACK`, and the parameters include the dimensions of `src0` and two additional parameters from `op_params`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the results of the softmax backward operation.


---
### ggml\_vk\_rope<!-- {{#callable:ggml_vk_rope}} -->
The `ggml_vk_rope` function performs a specific tensor operation using Vulkan backend for GPU acceleration, applying a rotation encoding technique based on the provided parameters.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan context for GPU operations.
    - `subctx`: A reference to a `vk_context` object that represents a sub-context for Vulkan operations.
    - `src0`: A pointer to the first source `ggml_tensor` which is used as the primary input for the operation.
    - `src1`: A pointer to the second source `ggml_tensor` which is used in conjunction with `src0`.
    - `src2`: A pointer to the third source `ggml_tensor`, which may be optional and can be null.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the operation will be stored.
    - `backprop`: A boolean flag indicating whether backpropagation should be performed.
    - `dryrun`: An optional boolean flag (defaulting to false) indicating whether to perform a dry run without executing the operation.
- **Control Flow**:
    - The function begins by extracting various parameters from the `dst->op_params` array, including dimensions, mode, and frequency factors.
    - If the mode indicates a specific type of rotation encoding (MROPE), it copies additional section parameters from `dst->op_params`.
    - The function calculates correlation dimensions using the `ggml_rope_yarn_corr_dims` function based on the extracted parameters.
    - It computes a scaling factor `theta_scale` based on the frequency base and the number of dimensions.
    - The sizes of the second and third dimensions of `src0` are calculated to assist in the operation.
    - Finally, it calls the `ggml_vk_op_f32` function to perform the actual tensor operation, passing all necessary parameters and flags.
- **Output**: The function does not return a value directly; instead, it modifies the `dst` tensor in place with the results of the rotation encoding operation.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)


---
### ggml\_vk\_argsort<!-- {{#callable:ggml_vk_argsort}} -->
The `ggml_vk_argsort` function performs an argsort operation on a source tensor and stores the result in a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to a `ggml_tensor` structure that represents the source tensor to be sorted.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the sorted result.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function retrieves the operation parameters from the destination tensor's `op_params` array.
    - It calculates the number of columns in the source tensor and determines the padded size by doubling until it exceeds the number of columns.
    - An assertion checks that the padded size does not exceed 1024.
    - The function then calls `ggml_vk_op_f32` to perform the argsort operation, passing the necessary parameters including the source tensor, destination tensor, and the calculated sizes.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place to contain the sorted indices of the source tensor.


---
### ggml\_vk\_sum<!-- {{#callable:ggml_vk_sum}} -->
The `ggml_vk_sum` function performs a summation operation on a source tensor and stores the result in a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to the `vk_context` that represents the Vulkan context for the operation.
    - `src0`: A pointer to the `ggml_tensor` that serves as the source tensor for the summation.
    - `dst`: A pointer to the `ggml_tensor` where the result of the summation will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function calls `ggml_vk_op_f32` with parameters including the context, source tensor, and destination tensor.
    - It specifies the operation type as `GGML_OP_SUM` to indicate that a summation operation is to be performed.
    - The function also passes additional parameters such as the number of elements in the source tensor and default values for other parameters.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` to contain the result of the summation operation.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_sum\_rows<!-- {{#callable:ggml_vk_sum_rows}} -->
The `ggml_vk_sum_rows` function performs a summation operation across the rows of a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to a constant `ggml_tensor` that serves as the source tensor from which rows will be summed.
    - `dst`: A pointer to a `ggml_tensor` that will store the result of the row summation.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function calls `ggml_vk_op_f32` with parameters that specify the operation type as `GGML_OP_SUM_ROWS`.
    - It passes the dimensions of the first dimension of `src0` as part of the operation parameters.
    - If `dryrun` is true, the operation is simulated without affecting the actual data.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the summed results of the rows from `src0`.


---
### ggml\_vk\_argmax<!-- {{#callable:ggml_vk_argmax}} -->
The `ggml_vk_argmax` function computes the argmax operation on a source tensor and stores the result in a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: A pointer to the `ggml_tensor` structure that serves as the source tensor for which the argmax is computed.
    - `dst`: A pointer to the `ggml_tensor` structure where the result of the argmax operation will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation is simulated without execution).
- **Control Flow**:
    - The function calls `ggml_vk_op_f32` to perform the argmax operation on the source tensor `src0`.
    - It passes the Vulkan context `ctx`, the subcontext `subctx`, and the destination tensor `dst` to store the result.
    - The operation type is specified as `GGML_OP_ARGMAX`, and the dimensions of the source tensor are provided as parameters.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor to contain the indices of the maximum values from the `src0` tensor.


---
### ggml\_vk\_count\_equal<!-- {{#callable:ggml_vk_count_equal}} -->
Counts the number of equal elements between two tensors using Vulkan.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to the `vk_context` that represents the Vulkan context for the operation.
    - `src0`: A pointer to the first `ggml_tensor` which is the source tensor to compare.
    - `src1`: A pointer to the second `ggml_tensor` which is the source tensor to compare against.
    - `dst`: A pointer to the `ggml_tensor` where the result of the count will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation will not execute but will still validate parameters).
- **Control Flow**:
    - Calls the `ggml_vk_op_f32` function to perform the counting operation using Vulkan, passing the necessary parameters including the operation type `GGML_OP_COUNT_EQUAL` and the number of elements in `src0`.
- **Output**: The function does not return a value; instead, it writes the result of the count of equal elements into the `dst` tensor.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_im2col<!-- {{#callable:ggml_vk_im2col}} -->
The `ggml_vk_im2col` function performs an image-to-column transformation for a given source tensor using Vulkan backend context.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context (`ggml_backend_vk_context`) used for managing Vulkan resources.
    - `subctx`: A reference to a Vulkan context (`vk_context`) that provides additional context for Vulkan operations.
    - `src0`: A pointer to the source tensor (`ggml_tensor`) that contains the kernel or filter data.
    - `src1`: A pointer to the source tensor (`ggml_tensor`) that contains the input image data to be transformed.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the transformed data will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the function will not execute the transformation).
- **Control Flow**:
    - The function begins by extracting various parameters from the `dst` tensor's operation parameters, including stride, padding, and dilation values.
    - It checks if the operation is 2D based on a specific parameter from the `dst` tensor.
    - The function retrieves dimensions of the input tensors (`src0` and `src1`) based on whether the operation is 2D or not.
    - It calculates offsets for batch and element access based on the byte size of the tensor data.
    - The number of elements to be processed is computed based on the output width and kernel dimensions.
    - Finally, it calls the `ggml_vk_op_f32` function to perform the actual image-to-column transformation, passing all necessary parameters and the dry run flag.
- **Output**: The function does not return a value; instead, it modifies the destination tensor (`dst`) in place with the transformed data.


---
### ggml\_vk\_timestep\_embedding<!-- {{#callable:ggml_vk_timestep_embedding}} -->
The `ggml_vk_timestep_embedding` function computes a timestep embedding for a given tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: A pointer to the Vulkan backend context (`ggml_backend_vk_context`) used for managing Vulkan resources.
    - `subctx`: A reference to a Vulkan context (`vk_context`) that provides additional context for Vulkan operations.
    - `src0`: A pointer to the source tensor (`ggml_tensor`) from which the embedding is computed.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the computed timestep embedding will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the function will not execute the actual operation).
- **Control Flow**:
    - The function retrieves the dimensionality (`dim`) and maximum period (`max_period`) from the destination tensor's operation parameters.
    - It calculates the number of elements in the second dimension of the destination tensor (`nb1`) based on its size and type.
    - The function then calls `ggml_vk_op_f32` to perform the actual Vulkan operation for timestep embedding, passing the necessary parameters and the dry run flag.
- **Output**: The function does not return a value; instead, it modifies the destination tensor (`dst`) in place with the computed timestep embedding.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)


---
### ggml\_vk\_conv\_transpose\_1d<!-- {{#callable:ggml_vk_conv_transpose_1d}} -->
Performs a 1D transposed convolution operation on the input tensors using Vulkan backend.
- **Inputs**:
    - `ctx`: Pointer to the Vulkan backend context used for managing Vulkan resources.
    - `subctx`: Reference to a Vulkan context that provides additional context for the operation.
    - `src0`: Pointer to the first input tensor representing the convolution kernel with shape (K, Cout, Cin, 1).
    - `src1`: Pointer to the second input tensor representing the input data with shape (L, Cin, 1, 1).
    - `dst`: Pointer to the output tensor where the result of the transposed convolution will be stored, with shape (*, Cout, 1, 1).
    - `dryrun`: Boolean flag indicating whether to perform a dry run (true) or execute the operation (false). Default is false.
- **Control Flow**:
    - The function begins by asserting that the types of the input tensors `src0`, `src1`, and output tensor `dst` are all of type `GGML_TYPE_F32`.
    - It initializes local variables for binary operations using `GGML_TENSOR_BINARY_OP_LOCALS`.
    - It asserts the sizes of the tensor elements to ensure they are of type float.
    - The function retrieves the stride parameter from the output tensor's operation parameters.
    - It prepares a structure `vk_op_conv_transpose_1d_push_constants` to hold the parameters for the Vulkan operation, including the dimensions and strides of the tensors.
    - Finally, it calls [`ggml_vk_op_f32`](#ggml_vk_op_f32) to execute the transposed convolution operation, passing the context, input tensors, and the prepared parameters, while respecting the dry run flag.
- **Output**: The function does not return a value; instead, it populates the output tensor `dst` with the result of the transposed convolution operation.
- **Functions called**:
    - [`ggml_vk_op_f32`](#ggml_vk_op_f32)


---
### ggml\_vk\_pool\_2d<!-- {{#callable:ggml_vk_pool_2d}} -->
Performs a 2D pooling operation on a source tensor and stores the result in a destination tensor using Vulkan backend.
- **Inputs**:
    - `ctx`: Pointer to a `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: Reference to a `vk_context` object that represents the Vulkan context for the operation.
    - `src0`: Pointer to the source `ggml_tensor` from which the pooling operation will read data.
    - `dst`: Pointer to the destination `ggml_tensor` where the result of the pooling operation will be stored.
    - `dryrun`: Boolean flag indicating whether to perform a dry run (if true, the operation is simulated without actual execution).
- **Control Flow**:
    - Extracts operation parameters from the `dst` tensor's `op_params` array, including kernel sizes, strides, and paddings.
    - Determines the dimensions of the input tensor (`src0`) and the output tensor (`dst`).
    - Calculates the total number of parallel elements to be processed based on the output tensor's dimensions.
    - Calls the `ggml_vk_op_f32` function to execute the 2D pooling operation with the specified parameters, passing the Vulkan context and the dry run flag.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the results of the pooling operation.


---
### ggml\_vk\_conv\_2d\_dw<!-- {{#callable:ggml_vk_conv_2d_dw}} -->
Performs a 2D depthwise convolution operation on the input tensors using Vulkan backend.
- **Inputs**:
    - `ctx`: Pointer to the Vulkan backend context used for managing Vulkan resources.
    - `subctx`: Reference to a Vulkan context that holds state and resources for the current operation.
    - `src0`: Pointer to the first input tensor, which represents the convolution kernel.
    - `src1`: Pointer to the second input tensor, which is the source data to be convolved.
    - `dst`: Pointer to the destination tensor where the result of the convolution will be stored.
    - `dryrun`: Boolean flag indicating whether to perform a dry run (true) or execute the operation (false).
- **Control Flow**:
    - Initializes a structure `vk_op_conv2d_dw_push_constants` to hold parameters for the convolution operation.
    - Extracts dimensions and parameters from the destination tensor and the source tensors to populate the push constants structure.
    - Validates that the number of channels in `src0` matches the expected number of channels in the destination tensor.
    - Validates that the number of batches in `src1` matches the expected number of batches in the destination tensor.
    - Calls [`ggml_vk_op_f32`](#ggml_vk_op_f32) to perform the actual convolution operation using the Vulkan backend, passing the prepared parameters and the dry run flag.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the result of the convolution operation.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_vk_op_f32`](#ggml_vk_op_f32)


---
### ggml\_vk\_leaky\_relu<!-- {{#callable:ggml_vk_leaky_relu}} -->
Applies the Leaky ReLU activation function to a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `subctx`: A reference to the `vk_context` that represents the Vulkan context for the operation.
    - `src0`: A pointer to the source `ggml_tensor` which contains the input data to be processed.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the Leaky ReLU operation will be stored.
    - `dryrun`: A boolean flag indicating whether to perform a dry run (if true, the operation will not be executed).
- **Control Flow**:
    - The function retrieves the operation parameters from the destination tensor's `op_params` field.
    - It then calls the `ggml_vk_op_f32` function to perform the Leaky ReLU operation, passing the necessary parameters including the source tensor, destination tensor, and operation type.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place with the results of the Leaky ReLU activation applied to the source tensor.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_vk\_print\_matrix\_area<!-- {{#callable:ggml_vk_print_matrix_area}} -->
Prints a specified area of a matrix to the standard error output.
- **Inputs**:
    - `data`: Pointer to the matrix data.
    - `type`: The data type of the matrix elements, which can be either `GGML_TYPE_F32` or `GGML_TYPE_F16`.
    - `ne0`: The size of the first dimension of the matrix.
    - `ne1`: The size of the second dimension of the matrix.
    - `i0`: The index in the first dimension from which to start printing.
    - `i1`: The index in the second dimension from which to start printing.
    - `i2`: The index in the third dimension (if applicable) to specify which slice of the matrix to print.
- **Control Flow**:
    - Checks if the `type` is either `GGML_TYPE_F32` or `GGML_TYPE_F16`, returning early if not.
    - Adjusts the starting indices `i0`, `i1`, and `i2` to ensure they are within valid bounds.
    - Prints the header for the column indices based on `i1`.
    - Iterates over the range of indices for the first dimension, printing each row.
    - For each row, iterates over the range of indices for the second dimension, retrieving and printing the corresponding matrix value.
    - Handles out-of-bounds indices by printing spaces instead of values.
- **Output**: The function outputs a formatted representation of a specified area of the matrix to the standard error stream, displaying the values in a grid format.
- **Functions called**:
    - [`ggml_fp16_to_fp32`](../ggml.c.driver.md#ggml_fp16_to_fp32)


---
### ggml\_vk\_test\_matmul<!-- {{#callable:ggml_vk_test_matmul}} -->
The `ggml_vk_test_matmul` function performs a matrix multiplication test using Vulkan, measuring performance and validating results.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the Vulkan device context.
    - `m`: The number of rows in the first matrix.
    - `n`: The number of columns in the second matrix.
    - `k`: The number of columns in the first matrix and rows in the second matrix.
    - `batch`: The number of batches to process.
    - `num_it`: The number of iterations to run for performance measurement.
    - `split_k`: A parameter that determines if the computation should be split across multiple kernels.
    - `shader_size`: An integer that specifies the size of the shader to use for the computation.
- **Control Flow**:
    - Logs the input parameters for debugging purposes.
    - Calculates the number of elements for the input and output matrices based on the dimensions and batch size.
    - Selects the appropriate Vulkan pipeline based on the data types of the input matrices and the specified shader size.
    - Handles padding for the `k` dimension if necessary, adjusting the pipeline accordingly.
    - Requests descriptor sets for the Vulkan pipeline and manages memory allocation for split-k if applicable.
    - Allocates device buffers for the input and output matrices and initializes them with random data.
    - Executes the matrix multiplication operation in a loop for the specified number of iterations.
    - Measures the execution time and calculates the performance in TFLOPS.
    - Validates the results by comparing the output from the Vulkan computation with a CPU-based computation.
    - Logs any discrepancies in the results and cleans up allocated resources.
- **Output**: The function outputs performance metrics including execution time and TFLOPS, and logs any errors in the computed results compared to expected values.
- **Functions called**:
    - [`ggml_vk_align_size`](#ggml_vk_align_size)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer)
    - [`ggml_vk_create_buffer_check`](#ggml_vk_create_buffer_check)
    - [`ggml_pipeline_allocate_descriptor_sets`](#ggml_pipeline_allocate_descriptor_sets)
    - [`ggml_vk_buffer_write`](#ggml_vk_buffer_write)
    - [`ggml_vk_create_context`](#ggml_vk_create_context)
    - [`ggml_vk_ctx_begin`](#ggml_vk_ctx_begin)
    - [`ggml_vk_matmul`](#ggml_vk_matmul)
    - [`ggml_vk_subbuffer`](#ggml_vk_subbuffer)
    - [`ggml_vk_ctx_end`](#ggml_vk_ctx_end)
    - [`ggml_vk_submit`](#ggml_vk_submit)
    - [`ggml_vk_buffer_read`](#ggml_vk_buffer_read)
    - [`ggml_init`](../ggml.c.driver.md#ggml_init)
    - [`ggml_new_tensor_3d`](../ggml.c.driver.md#ggml_new_tensor_3d)
    - [`ggml_mul_mat`](../ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_new_graph`](../ggml.c.driver.md#ggml_new_graph)
    - [`ggml_build_forward_expand`](../ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_graph_compute_with_ctx`](../ggml-cpu/ggml-cpu.c.driver.md#ggml_graph_compute_with_ctx)
    - [`ggml_free`](../ggml.c.driver.md#ggml_free)
    - [`ggml_vk_print_matrix_area`](#ggml_vk_print_matrix_area)
    - [`ggml_vk_queue_cleanup`](#ggml_vk_queue_cleanup)
    - [`ggml_pipeline_cleanup`](#ggml_pipeline_cleanup)


---
### ggml\_vk\_print\_tensor\_area<!-- {{#callable:ggml_vk_print_tensor_area}} -->
Prints a formatted area of a tensor's data to standard error.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that contains metadata about the tensor, including its type and dimensions.
    - `data`: A pointer to the raw data of the tensor.
    - `i0`: An integer representing the first index in the first dimension of the tensor.
    - `i1`: An integer representing the first index in the second dimension of the tensor.
    - `i2`: An integer representing the first index in the third dimension of the tensor.
    - `i3`: An integer representing the first index in the fourth dimension of the tensor.
- **Control Flow**:
    - Checks if the tensor type is one of the supported types (F32, F16, I32); if not, the function returns immediately.
    - Adjusts the input indices (i0, i1, i2, i3) to ensure they are within valid bounds, with minimum values set for i0 and i1.
    - Prints a header row of indices for the second dimension (i1) to standard error.
    - Iterates over a range of indices around i0 and i1, checking if the indices are within the bounds of the tensor dimensions.
    - For valid indices, retrieves the corresponding value from the tensor data based on its type and prints it; otherwise, prints spaces.
- **Output**: The function does not return a value; it outputs formatted tensor data directly to standard error.
- **Functions called**:
    - [`ggml_fp16_to_fp32`](../ggml.c.driver.md#ggml_fp16_to_fp32)


---
### ggml\_vk\_quantize\_data<!-- {{#callable:ggml_vk_quantize_data}} -->
Quantizes a chunk of floating-point data into a specified format.
- **Inputs**:
    - `from`: Pointer to the source array of floating-point values to be quantized.
    - `to`: Pointer to the destination where the quantized data will be stored.
    - `ne`: The number of elements in the source array to be quantized.
    - `quant`: The quantization type that specifies how the data should be quantized.
- **Control Flow**:
    - Calls the [`ggml_quantize_chunk`](../ggml.c.driver.md#ggml_quantize_chunk) function with the provided parameters to perform the quantization.
    - The parameters passed include the quantization type, source data pointer, destination data pointer, and the number of elements to process.
- **Output**: The function does not return a value; instead, it modifies the data in the destination pointer `to` with the quantized results.
- **Functions called**:
    - [`ggml_quantize_chunk`](../ggml.c.driver.md#ggml_quantize_chunk)


---
### ggml\_vk\_dequantize\_data<!-- {{#callable:ggml_vk_dequantize_data}} -->
Dequantizes data from a specified format to floating-point representation.
- **Inputs**:
    - `from`: A pointer to the source data that needs to be dequantized.
    - `to`: A pointer to the destination array where the dequantized floating-point values will be stored.
    - `ne`: The number of elements to be dequantized.
    - `quant`: An enumeration value representing the data type of the source data.
- **Control Flow**:
    - Checks if the `quant` type is `GGML_TYPE_F32`, and if so, directly copies the data from `from` to `to` using `memcpy`.
    - If the `quant` type is not `GGML_TYPE_F32`, retrieves the type traits associated with the `quant` type.
    - Uses the `to_float` function from the type traits to convert the data from `from` to `to` for the specified number of elements `ne`.
- **Output**: The function does not return a value; instead, it populates the `to` array with the dequantized floating-point values.
- **Functions called**:
    - [`ggml_get_type_traits`](../ggml.c.driver.md#ggml_get_type_traits)


---
### ggml\_vk\_test\_dequant<!-- {{#callable:ggml_vk_test_dequant}} -->
The `ggml_vk_test_dequant` function tests the dequantization process of quantized data using Vulkan, measuring performance and error metrics.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the Vulkan device context.
    - `ne`: A size_t value representing the number of elements to process.
    - `quant`: A `ggml_type` enumeration value that specifies the quantization type to be used.
- **Control Flow**:
    - Logs the start of the function execution with the number of elements.
    - Calculates the sizes for various buffers based on the number of elements and quantization type.
    - Allocates memory for input and output buffers.
    - Generates random float values for the input data.
    - Retrieves the Vulkan pipeline for converting data to half-precision floats.
    - Quantizes the input data and then dequantizes it back to reference values.
    - Requests and allocates descriptor sets for the Vulkan pipeline.
    - Writes the quantized data to a Vulkan buffer and creates a Vulkan context for execution.
    - Dispatches the Vulkan pipeline to process the data.
    - Measures the time taken for the dequantization process.
    - Reads the dequantized data back from the Vulkan buffer.
    - Calculates the average error between the reference and dequantized data, logging any discrepancies.
    - Cleans up by destroying Vulkan buffers and freeing allocated memory.
- **Output**: The function outputs the time taken for the dequantization process and the average error between the expected and actual results, logging detailed error information if the average error exceeds a threshold.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`ggml_vk_create_buffer_check`](#ggml_vk_create_buffer_check)
    - [`ggml_vk_get_to_fp16`](#ggml_vk_get_to_fp16)
    - [`ggml_vk_quantize_data`](#ggml_vk_quantize_data)
    - [`ggml_vk_dequantize_data`](#ggml_vk_dequantize_data)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`ggml_pipeline_allocate_descriptor_sets`](#ggml_pipeline_allocate_descriptor_sets)
    - [`ggml_vk_buffer_write`](#ggml_vk_buffer_write)
    - [`ggml_vk_create_context`](#ggml_vk_create_context)
    - [`ggml_vk_ctx_begin`](#ggml_vk_ctx_begin)
    - [`ggml_vk_dispatch_pipeline`](#ggml_vk_dispatch_pipeline)
    - [`ggml_vk_ctx_end`](#ggml_vk_ctx_end)
    - [`ggml_vk_submit`](#ggml_vk_submit)
    - [`ggml_vk_buffer_read`](#ggml_vk_buffer_read)
    - [`ggml_fp16_to_fp32`](../ggml.c.driver.md#ggml_fp16_to_fp32)
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)
    - [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer)


---
### ggml\_vk\_test\_dequant\_matmul<!-- {{#callable:ggml_vk_test_dequant_matmul}} -->
The `ggml_vk_test_dequant_matmul` function performs a dequantized matrix multiplication test using Vulkan, measuring performance and accuracy.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the Vulkan device context.
    - `m`: The number of rows in the first matrix.
    - `n`: The number of columns in the second matrix.
    - `k`: The number of columns in the first matrix and rows in the second matrix.
    - `batch`: The number of matrices to process in parallel.
    - `num_it`: The number of iterations to run the matrix multiplication.
    - `split_k`: The split factor for the K dimension during matrix multiplication.
    - `shader_size`: An integer indicating the size of the shader to use.
    - `quant`: The quantization type used for the input matrices.
    - `mmq`: A boolean flag indicating whether to use multi-query quantization.
- **Control Flow**:
    - Logs the function parameters for debugging purposes.
    - Calculates the number of elements for input and output matrices based on the dimensions and batch size.
    - Selects the appropriate Vulkan pipeline based on the quantization type and whether multi-query quantization is enabled.
    - Allocates memory for input and output matrices and their quantized versions.
    - Initializes input matrices with random values and quantizes the first matrix.
    - Requests descriptor sets for the Vulkan pipeline and prepares for execution.
    - Writes the quantized input and second matrix to Vulkan buffers.
    - Creates a Vulkan context and executes the matrix multiplication in a loop for the specified number of iterations.
    - Measures the execution time and reads the output from the Vulkan buffer.
    - Computes the average error between the Vulkan output and the expected output using GGML functions.
    - Logs performance metrics and any discrepancies in the results.
- **Output**: The function does not return a value but logs performance metrics, including execution time, TFLOPS, and average error between computed and expected results.
- **Functions called**:
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)
    - [`ggml_vk_align_size`](#ggml_vk_align_size)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`ggml_vk_create_buffer_check`](#ggml_vk_create_buffer_check)
    - [`ggml_vk_quantize_data`](#ggml_vk_quantize_data)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer)
    - [`ggml_pipeline_allocate_descriptor_sets`](#ggml_pipeline_allocate_descriptor_sets)
    - [`ggml_vk_buffer_write`](#ggml_vk_buffer_write)
    - [`ggml_vk_create_context`](#ggml_vk_create_context)
    - [`ggml_vk_ctx_begin`](#ggml_vk_ctx_begin)
    - [`ggml_vk_quantize_q8_1`](#ggml_vk_quantize_q8_1)
    - [`ggml_vk_matmul`](#ggml_vk_matmul)
    - [`ggml_vk_ctx_end`](#ggml_vk_ctx_end)
    - [`ggml_vk_submit`](#ggml_vk_submit)
    - [`ggml_vk_buffer_read`](#ggml_vk_buffer_read)
    - [`ggml_init`](../ggml.c.driver.md#ggml_init)
    - [`ggml_new_tensor_3d`](../ggml.c.driver.md#ggml_new_tensor_3d)
    - [`ggml_mul_mat`](../ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_new_graph`](../ggml.c.driver.md#ggml_new_graph)
    - [`ggml_build_forward_expand`](../ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_graph_compute_with_ctx`](../ggml-cpu/ggml-cpu.c.driver.md#ggml_graph_compute_with_ctx)
    - [`ggml_free`](../ggml.c.driver.md#ggml_free)
    - [`ggml_vk_print_matrix_area`](#ggml_vk_print_matrix_area)


---
### ggml\_vk\_preallocate\_buffers<!-- {{#callable:ggml_vk_preallocate_buffers}} -->
Preallocates Vulkan buffers for a given context based on specified sizes.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains Vulkan context and buffer size information.
- **Control Flow**:
    - The function checks if Vulkan tests are enabled using a preprocessor directive.
    - If tests are enabled, it initializes a vector of sizes and performs multiple matrix multiplication tests with different parameters.
    - The function then checks if the preallocated buffers for `x`, `y`, and `split_k` are either null or insufficient in size.
    - If any buffer is insufficient, it logs the memory allocation request, destroys the existing buffer if it exists, and creates a new buffer of the required size.
- **Output**: The function does not return a value; it modifies the Vulkan context by allocating or resizing buffers as necessary.
- **Functions called**:
    - [`ggml_vk_test_dequant_matmul`](#ggml_vk_test_dequant_matmul)
    - [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer)
    - [`ggml_vk_create_buffer_device`](#ggml_vk_create_buffer_device)


---
### ggml\_vk\_build\_graph<!-- {{#callable:ggml_vk_build_graph}} -->
The `ggml_vk_build_graph` function constructs a Vulkan compute graph for a given tensor operation, handling various tensor operations and managing the compute context.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_backend_vk_context` structure that holds the Vulkan backend context.
    - `node`: A pointer to the `ggml_tensor` structure representing the current tensor node to be processed.
    - `node_idx`: An integer index representing the position of the current node in the tensor graph.
    - `node_begin`: A pointer to the starting tensor node for the computation.
    - `node_idx_begin`: An integer index representing the starting position in the tensor graph.
    - `dryrun`: A boolean flag indicating whether to perform a dry run without executing the operations.
    - `last_node`: A boolean flag indicating if the current node is the last node in the computation.
    - `almost_ready`: A boolean flag indicating if the computation is almost ready to be executed.
    - `submit`: A boolean flag indicating whether to submit the computation for execution.
- **Control Flow**:
    - The function first checks if the `node` is empty or lacks a buffer, returning false if so.
    - It logs the function call and initializes the semaphore index.
    - The function retrieves source tensors from the `node` and checks the operation type using a switch statement.
    - If the operation is not supported or is a no-op, it returns false.
    - For supported operations, it either creates a new compute context or reuses an existing one based on the `dryrun` flag.
    - If `dryrun` is true, it handles specific operations by requesting pipeline descriptor sets and returns false.
    - For each supported operation, it calls the corresponding Vulkan function to build the graph.
    - If `dryrun` is false and the `submit` or `last_node` flags are set, it ends the compute context and submits the computation for execution.
    - Finally, it returns true to indicate successful graph building.
- **Output**: The function returns a boolean value indicating the success or failure of the graph building process.
- **Functions called**:
    - [`ggml_is_empty`](../ggml.c.driver.md#ggml_is_empty)
    - [`ggml_op_name`](../ggml.c.driver.md#ggml_op_name)
    - [`ggml_get_unary_op`](../ggml.c.driver.md#ggml_get_unary_op)
    - [`ggml_vk_create_context`](#ggml_vk_create_context)
    - [`ggml_vk_ctx_begin`](#ggml_vk_ctx_begin)
    - [`ggml_vk_op_get_pipeline`](#ggml_vk_op_get_pipeline)
    - [`ggml_pipeline_request_descriptor_sets`](#ggml_pipeline_request_descriptor_sets)
    - [`ggml_vk_repeat`](#ggml_vk_repeat)
    - [`ggml_vk_repeat_back`](#ggml_vk_repeat_back)
    - [`ggml_vk_acc`](#ggml_vk_acc)
    - [`ggml_vk_get_rows`](#ggml_vk_get_rows)
    - [`ggml_vk_add`](#ggml_vk_add)
    - [`ggml_vk_sub`](#ggml_vk_sub)
    - [`ggml_vk_mul`](#ggml_vk_mul)
    - [`ggml_vk_div`](#ggml_vk_div)
    - [`ggml_vk_concat`](#ggml_vk_concat)
    - [`ggml_vk_upscale`](#ggml_vk_upscale)
    - [`ggml_vk_scale`](#ggml_vk_scale)
    - [`ggml_vk_sqr`](#ggml_vk_sqr)
    - [`ggml_vk_sin`](#ggml_vk_sin)
    - [`ggml_vk_cos`](#ggml_vk_cos)
    - [`ggml_vk_clamp`](#ggml_vk_clamp)
    - [`ggml_vk_pad`](#ggml_vk_pad)
    - [`ggml_vk_cpy`](#ggml_vk_cpy)
    - [`ggml_vk_silu_back`](#ggml_vk_silu_back)
    - [`ggml_vk_norm`](#ggml_vk_norm)
    - [`ggml_vk_group_norm`](#ggml_vk_group_norm)
    - [`ggml_vk_rms_norm`](#ggml_vk_rms_norm)
    - [`ggml_vk_rms_norm_back`](#ggml_vk_rms_norm_back)
    - [`ggml_vk_l2_norm`](#ggml_vk_l2_norm)
    - [`ggml_vk_unary`](#ggml_vk_unary)
    - [`ggml_vk_diag_mask_inf`](#ggml_vk_diag_mask_inf)
    - [`ggml_vk_soft_max`](#ggml_vk_soft_max)
    - [`ggml_vk_soft_max_back`](#ggml_vk_soft_max_back)
    - [`ggml_vk_rope`](#ggml_vk_rope)
    - [`ggml_vk_argsort`](#ggml_vk_argsort)
    - [`ggml_vk_sum`](#ggml_vk_sum)
    - [`ggml_vk_sum_rows`](#ggml_vk_sum_rows)
    - [`ggml_vk_argmax`](#ggml_vk_argmax)
    - [`ggml_vk_count_equal`](#ggml_vk_count_equal)
    - [`ggml_vk_im2col`](#ggml_vk_im2col)
    - [`ggml_vk_timestep_embedding`](#ggml_vk_timestep_embedding)
    - [`ggml_vk_conv_transpose_1d`](#ggml_vk_conv_transpose_1d)
    - [`ggml_vk_pool_2d`](#ggml_vk_pool_2d)
    - [`ggml_vk_conv_2d_dw`](#ggml_vk_conv_2d_dw)
    - [`ggml_vk_leaky_relu`](#ggml_vk_leaky_relu)
    - [`ggml_vk_mul_mat`](#ggml_vk_mul_mat)
    - [`ggml_vk_mul_mat_id`](#ggml_vk_mul_mat_id)
    - [`ggml_vk_flash_attn`](#ggml_vk_flash_attn)
    - [`ggml_vk_rwkv_wkv6`](#ggml_vk_rwkv_wkv6)
    - [`ggml_vk_rwkv_wkv7`](#ggml_vk_rwkv_wkv7)
    - [`ggml_vk_opt_step_adamw`](#ggml_vk_opt_step_adamw)
    - [`ggml_vk_ctx_end`](#ggml_vk_ctx_end)
    - [`ggml_vk_compute_forward`](#ggml_vk_compute_forward)
    - [`ggml_unary_op_name`](../ggml.c.driver.md#ggml_unary_op_name)


---
### ggml\_vk\_graph\_cleanup<!-- {{#callable:ggml_vk_graph_cleanup}} -->
The `ggml_vk_graph_cleanup` function cleans up various resources associated with a Vulkan graphics context.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains the Vulkan context and associated resources to be cleaned up.
- **Control Flow**:
    - Logs the entry into the function using `VK_LOG_DEBUG`.
    - Iterates over `temp_buffers` in the context and frees each buffer using [`ggml_vk_pool_free`](#ggml_vk_pool_free).
    - Clears the `temp_buffers` list after freeing the buffers.
    - Iterates over the `pipeline_descriptor_set_requirements` to clean up each pipeline if it is not expired.
    - Calls [`ggml_pipeline_cleanup`](#ggml_pipeline_cleanup) for each valid pipeline.
    - Cleans up the compute and transfer queues by calling [`ggml_vk_queue_cleanup`](#ggml_vk_queue_cleanup).
    - Destroys each semaphore in `gc.semaphores` and clears the list.
    - Destroys each semaphore in `gc.tl_semaphores` and clears the list.
    - Resets the semaphore index and event index to zero.
    - Resets each event in `gc.events` using the Vulkan device's `resetEvent` method.
    - Clears the `tensor_ctxs` and `gc.contexts` lists.
    - Clears the `pipeline_descriptor_set_requirements` in the device context.
- **Output**: The function does not return a value; it performs cleanup operations to free resources and reset states within the provided Vulkan context.
- **Functions called**:
    - [`ggml_vk_pool_free`](#ggml_vk_pool_free)
    - [`ggml_pipeline_cleanup`](#ggml_pipeline_cleanup)
    - [`ggml_vk_queue_cleanup`](#ggml_vk_queue_cleanup)


---
### ggml\_vk\_cleanup<!-- {{#callable:ggml_vk_cleanup}} -->
Cleans up Vulkan resources associated with the given context.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_vk_context` structure that contains Vulkan context and resource information.
- **Control Flow**:
    - Logs the cleanup operation with the context's name.
    - Calls [`ggml_vk_graph_cleanup`](#ggml_vk_graph_cleanup) to clean up any graph-related resources.
    - Destroys preallocated Vulkan buffers for `prealloc_x`, `prealloc_y`, and `prealloc_split_k`.
    - Iterates over the `buffer_pool` and destroys each buffer using [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer).
    - Resets the preallocated sizes for `prealloc_size_x`, `prealloc_size_y`, and `prealloc_size_split_k` to zero.
    - Iterates over the events in the garbage collection structure and destroys each Vulkan event.
    - Clears the list of events in the garbage collection structure.
    - Destroys the Vulkan fences associated with the context.
- **Output**: This function does not return a value; it performs cleanup operations on Vulkan resources.
- **Functions called**:
    - [`ggml_vk_graph_cleanup`](#ggml_vk_graph_cleanup)
    - [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer)


---
### ggml\_vk\_get\_device\_count<!-- {{#callable:ggml_vk_get_device_count}} -->
Retrieves the count of Vulkan devices initialized in the current Vulkan instance.
- **Inputs**: None
- **Control Flow**:
    - Calls the [`ggml_vk_instance_init`](#ggml_vk_instance_init) function to ensure the Vulkan instance is properly initialized.
    - Returns the size of the `device_indices` vector from the `vk_instance`, which holds the indices of the available Vulkan devices.
- **Output**: An integer representing the number of Vulkan devices available in the initialized Vulkan instance.
- **Functions called**:
    - [`ggml_vk_instance_init`](#ggml_vk_instance_init)


---
### ggml\_vk\_get\_device\_description<!-- {{#callable:ggml_vk_get_device_description}} -->
Retrieves the description of a specified Vulkan physical device.
- **Inputs**:
    - `device`: An integer representing the index of the Vulkan physical device to query.
    - `description`: A pointer to a character array where the device description will be stored.
    - `description_size`: The size of the character array to ensure the description does not exceed this limit.
- **Control Flow**:
    - Initializes the Vulkan instance by calling `ggml_vk_instance_init()`.
    - Enumerates all available physical devices using `vk_instance.instance.enumeratePhysicalDevices()` and stores them in a vector.
    - Retrieves the properties of the specified physical device using `getProperties()` method.
    - Formats the device name into the provided `description` buffer using `snprintf()`.
- **Output**: The function does not return a value; instead, it populates the `description` buffer with the name of the specified Vulkan physical device.
- **Functions called**:
    - [`ggml_vk_instance_init`](#ggml_vk_instance_init)


---
### ggml\_backend\_buffer\_is\_vk<!-- {{#callable:ggml_backend_buffer_is_vk}} -->
Checks if the given `ggml_backend_buffer_t` buffer is of Vulkan buffer type.
- **Inputs**:
    - `buffer`: A `ggml_backend_buffer_t` type representing the buffer to be checked.
- **Control Flow**:
    - The function accesses the `buft` member of the `buffer` to retrieve its interface.
    - It compares the name of the buffer's interface with the predefined Vulkan buffer type name.
- **Output**: Returns a boolean value indicating whether the buffer is of Vulkan type.


---
### ggml\_backend\_vk\_buffer\_free\_buffer<!-- {{#callable:ggml_backend_vk_buffer_free_buffer}} -->
Frees a Vulkan buffer and its associated context.
- **Inputs**:
    - `buffer`: A pointer to a `ggml_backend_buffer_t` structure that contains the Vulkan buffer and its context.
- **Control Flow**:
    - Logs the memory freeing operation using `VK_LOG_MEMORY`.
    - Retrieves the Vulkan buffer context from the provided `buffer` argument.
    - Calls [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer) to free the Vulkan buffer associated with the context.
    - Deletes the context object to free its memory.
- **Output**: This function does not return a value; it performs memory deallocation.
- **Functions called**:
    - [`ggml_vk_destroy_buffer`](#ggml_vk_destroy_buffer)


---
### ggml\_backend\_vk\_buffer\_get\_base<!-- {{#callable:ggml_backend_vk_buffer_get_base}} -->
Retrieves the base pointer of a Vulkan buffer.
- **Inputs**:
    - `buffer`: An instance of `ggml_backend_buffer_t` representing the Vulkan buffer from which the base pointer is to be retrieved.
- **Control Flow**:
    - The function immediately returns the value of `vk_ptr_base`.
    - The input parameter `buffer` is marked as unused, indicating it is not utilized in the function's logic.
- **Output**: Returns a pointer to the base of the Vulkan buffer.


---
### ggml\_backend\_vk\_buffer\_init\_tensor<!-- {{#callable:ggml_backend_vk_buffer_init_tensor}} -->
Initializes a Vulkan buffer for a given tensor.
- **Inputs**:
    - `buffer`: A pointer to a `ggml_backend_buffer_t` structure representing the Vulkan buffer to be initialized.
    - `tensor`: A pointer to a `ggml_tensor` structure that represents the tensor to be associated with the Vulkan buffer.
- **Control Flow**:
    - Logs the function call with the provided `buffer` and `tensor` details.
    - Checks if the `tensor` has a source view (`view_src`); if it does, it asserts that the buffer type of the source view matches the buffer type of the provided `buffer`.
- **Output**: Returns `GGML_STATUS_SUCCESS` indicating that the buffer initialization for the tensor was successful.


---
### ggml\_backend\_vk\_buffer\_memset\_tensor<!-- {{#callable:ggml_backend_vk_buffer_memset_tensor}} -->
The `ggml_backend_vk_buffer_memset_tensor` function sets a specified range of bytes in a Vulkan buffer to a given value.
- **Inputs**:
    - `buffer`: A `ggml_backend_buffer_t` structure representing the Vulkan buffer context.
    - `tensor`: A pointer to a `ggml_tensor` structure that specifies the tensor to be modified.
    - `value`: An 8-bit unsigned integer value that will be used to set the specified range in the buffer.
    - `offset`: A size_t value indicating the starting point in the tensor's buffer where the setting begins.
    - `size`: A size_t value representing the number of bytes to set in the buffer.
- **Control Flow**:
    - The function logs the input parameters for debugging purposes using `VK_LOG_DEBUG`.
    - It retrieves the Vulkan buffer context from the provided `buffer` argument.
    - The function calculates a 32-bit value by replicating the 8-bit `value` across all four bytes.
    - It calls [`ggml_vk_buffer_memset`](#ggml_vk_buffer_memset) to set the specified range in the Vulkan buffer, using the calculated offset and size.
- **Output**: The function does not return a value; it modifies the Vulkan buffer directly.
- **Functions called**:
    - [`ggml_vk_buffer_memset`](#ggml_vk_buffer_memset)
    - [`vk_tensor_offset`](#vk_tensor_offset)


---
### ggml\_backend\_vk\_buffer\_set\_tensor<!-- {{#callable:ggml_backend_vk_buffer_set_tensor}} -->
Sets the data of a specified tensor in a Vulkan buffer.
- **Inputs**:
    - `buffer`: A `ggml_backend_buffer_t` representing the Vulkan buffer context.
    - `tensor`: A pointer to a `ggml_tensor` structure that specifies the tensor to be updated.
    - `data`: A pointer to the data that will be written to the tensor.
    - `offset`: A size_t value indicating the offset within the tensor where the data will be written.
    - `size`: A size_t value representing the number of bytes to write to the tensor.
- **Control Flow**:
    - Logs the function call with the provided parameters for debugging purposes.
    - Retrieves the Vulkan buffer context from the provided `buffer` argument.
    - Calculates the correct offset in the Vulkan buffer using the tensor's offset and the provided `offset` argument.
    - Calls [`ggml_vk_buffer_write`](#ggml_vk_buffer_write) to write the specified `data` to the calculated offset in the Vulkan buffer.
- **Output**: The function does not return a value; it performs a write operation to the Vulkan buffer.
- **Functions called**:
    - [`ggml_vk_buffer_write`](#ggml_vk_buffer_write)
    - [`vk_tensor_offset`](#vk_tensor_offset)


---
### ggml\_backend\_vk\_buffer\_get\_tensor<!-- {{#callable:ggml_backend_vk_buffer_get_tensor}} -->
The `ggml_backend_vk_buffer_get_tensor` function retrieves a tensor's data from a Vulkan buffer.
- **Inputs**:
    - `buffer`: A `ggml_backend_buffer_t` structure representing the Vulkan backend buffer context.
    - `tensor`: A pointer to a `ggml_tensor` structure that specifies the tensor to retrieve data from.
    - `data`: A pointer to the memory location where the retrieved tensor data will be stored.
    - `offset`: A size_t value indicating the offset from the start of the tensor data.
    - `size`: A size_t value specifying the number of bytes to read from the tensor.
- **Control Flow**:
    - The function begins by logging the input parameters for debugging purposes.
    - It retrieves the Vulkan buffer context from the provided `buffer` argument.
    - The Vulkan buffer is accessed from the buffer context.
    - The function then calls [`ggml_vk_buffer_read`](#ggml_vk_buffer_read) to read the specified amount of data from the Vulkan buffer into the provided `data` pointer, using the calculated offset.
- **Output**: The function does not return a value; it directly modifies the memory pointed to by the `data` argument with the retrieved tensor data.
- **Functions called**:
    - [`ggml_vk_buffer_read`](#ggml_vk_buffer_read)
    - [`vk_tensor_offset`](#vk_tensor_offset)


---
### ggml\_backend\_vk\_buffer\_cpy\_tensor<!-- {{#callable:ggml_backend_vk_buffer_cpy_tensor}} -->
Copies data from a source tensor buffer to a destination tensor buffer if the source buffer is a Vulkan buffer.
- **Inputs**:
    - `buffer`: A `ggml_backend_buffer_t` representing the backend buffer context.
    - `src`: A pointer to a constant `ggml_tensor` representing the source tensor.
    - `dst`: A pointer to a `ggml_tensor` representing the destination tensor.
- **Control Flow**:
    - Checks if the source tensor's buffer is a Vulkan buffer using [`ggml_backend_buffer_is_vk`](#ggml_backend_buffer_is_vk).
    - If true, retrieves the Vulkan buffer contexts for both source and destination tensors.
    - Extracts the Vulkan device buffers from the contexts.
    - Calls [`ggml_vk_buffer_copy`](#ggml_vk_buffer_copy) to perform the actual data copy from the source buffer to the destination buffer, adjusting for tensor offsets.
    - Returns true to indicate a successful copy operation.
    - If the source buffer is not a Vulkan buffer, returns false.
- **Output**: Returns a boolean value indicating whether the copy operation was successful.
- **Functions called**:
    - [`ggml_backend_buffer_is_vk`](#ggml_backend_buffer_is_vk)
    - [`ggml_vk_buffer_copy`](#ggml_vk_buffer_copy)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)


---
### ggml\_backend\_vk\_buffer\_clear<!-- {{#callable:ggml_backend_vk_buffer_clear}} -->
Clears a Vulkan buffer by setting its memory to a specified value.
- **Inputs**:
    - `buffer`: A `ggml_backend_buffer_t` structure representing the buffer to be cleared.
    - `value`: A `uint8_t` value that specifies the value to set the buffer's memory to.
- **Control Flow**:
    - The function retrieves the Vulkan buffer context from the provided `buffer` argument.
    - It then calls [`ggml_vk_buffer_memset`](#ggml_vk_buffer_memset) to set the memory of the Vulkan buffer to the specified `value` for the entire size of the buffer.
- **Output**: This function does not return a value; it modifies the memory of the specified Vulkan buffer directly.
- **Functions called**:
    - [`ggml_vk_buffer_memset`](#ggml_vk_buffer_memset)


---
### ggml\_backend\_vk\_buffer\_type\_name<!-- {{#callable:ggml_backend_vk_buffer_type_name}} -->
Returns the name of the Vulkan buffer type associated with the given buffer type.
- **Inputs**:
    - `buft`: A pointer to a `ggml_backend_buffer_type_t` structure representing the buffer type.
- **Control Flow**:
    - The function casts the `context` member of the `buft` structure to a pointer of type `ggml_backend_vk_buffer_type_context`.
    - It then accesses the `name` member of the context and returns its C-style string representation using `c_str()`.
- **Output**: A pointer to a constant character string representing the name of the Vulkan buffer type.


---
### ggml\_backend\_vk\_buffer\_type\_alloc\_buffer<!-- {{#callable:ggml_backend_vk_buffer_type_alloc_buffer}} -->
Allocates a Vulkan buffer of a specified size and initializes a backend buffer context.
- **Inputs**:
    - `buft`: A `ggml_backend_buffer_type_t` representing the type of buffer to allocate.
    - `size`: A `size_t` indicating the size of the buffer to be allocated.
- **Control Flow**:
    - Logs the memory allocation request with the specified size.
    - Retrieves the Vulkan buffer type context from the provided buffer type.
    - Attempts to create a Vulkan device buffer using the specified size.
    - Catches any `vk::SystemError` exceptions that may occur during buffer creation and returns nullptr in case of an error.
    - If successful, creates a new `ggml_backend_vk_buffer_context` using the device and the created buffer.
    - Initializes and returns a `ggml_backend_buffer_t` using the buffer type, buffer interface, and the new buffer context.
- **Output**: Returns a `ggml_backend_buffer_t` initialized with the new buffer context, or nullptr if the allocation fails.
- **Functions called**:
    - [`ggml_vk_create_buffer_device`](#ggml_vk_create_buffer_device)


---
### ggml\_backend\_vk\_buffer\_type\_get\_alignment<!-- {{#callable:ggml_backend_vk_buffer_type_get_alignment}} -->
Retrieves the minimum storage buffer offset alignment for a given Vulkan buffer type.
- **Inputs**:
    - `buft`: A pointer to a `ggml_backend_buffer_type_t` structure representing the Vulkan buffer type.
- **Control Flow**:
    - The function casts the `context` member of the `buft` structure to a `ggml_backend_vk_buffer_type_context` pointer.
    - It accesses the `device` member of the context to retrieve the Vulkan device properties.
    - Finally, it returns the `minStorageBufferOffsetAlignment` value from the device properties.
- **Output**: Returns a `size_t` value representing the minimum storage buffer offset alignment for the specified Vulkan buffer type.


---
### ggml\_backend\_vk\_buffer\_type\_get\_max\_size<!-- {{#callable:ggml_backend_vk_buffer_type_get_max_size}} -->
Retrieves the maximum size of a buffer type from the Vulkan backend context.
- **Inputs**:
    - `buft`: A pointer to a `ggml_backend_buffer_type_t` structure representing the buffer type whose maximum size is to be retrieved.
- **Control Flow**:
    - The function casts the `context` member of the `buft` structure to a pointer of type `ggml_backend_vk_buffer_type_context`.
    - It accesses the `device` member of the context and retrieves the `suballocation_block_size` which represents the maximum size of the buffer.
- **Output**: Returns a `size_t` value representing the maximum size of the buffer type as defined by the Vulkan backend context.


---
### ggml\_backend\_vk\_buffer\_type\_get\_alloc\_size<!-- {{#callable:ggml_backend_vk_buffer_type_get_alloc_size}} -->
Calculates the allocation size for a given tensor in bytes.
- **Inputs**:
    - `buft`: An enumeration value of type `ggml_backend_buffer_type_t` representing the buffer type, which is unused in the function.
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor for which the allocation size is to be calculated.
- **Control Flow**:
    - The function directly calls `ggml_nbytes(tensor)` to compute the size in bytes required for the tensor.
    - The input parameter `buft` is declared but not used within the function, indicating it may be for future use or for interface consistency.
- **Output**: Returns the size in bytes required to allocate memory for the specified tensor.
- **Functions called**:
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)


---
### ggml\_backend\_vk\_buffer\_type<!-- {{#callable:ggml_backend_vk_buffer_type}} -->
The `ggml_backend_vk_buffer_type` function retrieves the buffer type associated with a specified Vulkan device.
- **Inputs**:
    - `dev_num`: An integer representing the device number for which the buffer type is to be retrieved.
- **Control Flow**:
    - The function begins by initializing the Vulkan instance through `ggml_vk_instance_init()`.
    - A debug log is generated to indicate the invocation of the function with the provided device number.
    - The Vulkan device corresponding to the given device number is obtained using `ggml_vk_get_device(dev_num)`.
    - Finally, the function returns a pointer to the buffer type associated with the retrieved Vulkan device.
- **Output**: The function returns a pointer to the `buffer_type` member of the Vulkan device structure.
- **Functions called**:
    - [`ggml_vk_instance_init`](#ggml_vk_instance_init)
    - [`ggml_vk_get_device`](#ggml_vk_get_device)


---
### ggml\_backend\_vk\_host\_buffer\_type\_name<!-- {{#callable:ggml_backend_vk_host_buffer_type_name}} -->
Returns the name of the Vulkan host buffer type.
- **Inputs**:
    - `buft`: An enumeration value of type `ggml_backend_buffer_type_t` representing the buffer type.
- **Control Flow**:
    - The function immediately returns a constant string that represents the Vulkan host buffer type name.
    - The input parameter `buft` is unused in the function body.
- **Output**: A constant string representing the Vulkan host buffer type name, specifically formatted as 'GGML_VK_NAME_Host'.


---
### ggml\_backend\_vk\_host\_buffer\_name<!-- {{#callable:ggml_backend_vk_host_buffer_name}} -->
Returns the name of the host buffer for Vulkan backend.
- **Inputs**:
    - `buffer`: An instance of `ggml_backend_buffer_t` representing the Vulkan backend buffer.
- **Control Flow**:
    - The function immediately returns a constant string that represents the host buffer name.
    - The input parameter `buffer` is marked as unused, indicating it has no effect on the function's output.
- **Output**: A constant string representing the name of the host buffer, specifically 'GGML_VK_NAME_Host'.


---
### ggml\_backend\_vk\_host\_buffer\_free\_buffer<!-- {{#callable:ggml_backend_vk_host_buffer_free_buffer}} -->
Frees a Vulkan host buffer associated with the given `ggml_backend_buffer_t`.
- **Inputs**:
    - `buffer`: A pointer to a `ggml_backend_buffer_t` structure representing the Vulkan host buffer to be freed.
- **Control Flow**:
    - Logs the memory freeing operation using `VK_LOG_MEMORY`.
    - Calls [`ggml_vk_host_free`](#ggml_vk_host_free) to deallocate the memory associated with the buffer's context.
- **Output**: This function does not return a value; it performs a memory deallocation operation.
- **Functions called**:
    - [`ggml_vk_host_free`](#ggml_vk_host_free)


---
### ggml\_backend\_vk\_host\_buffer\_type\_alloc\_buffer<!-- {{#callable:ggml_backend_vk_host_buffer_type_alloc_buffer}} -->
Allocates a Vulkan host buffer of a specified size, with a fallback to CPU buffer allocation in case of failure.
- **Inputs**:
    - `buft`: An enumeration value of type `ggml_backend_buffer_type_t` that specifies the type of buffer to allocate.
    - `size`: A `size_t` value representing the requested size of the buffer to be allocated.
- **Control Flow**:
    - Logs the memory allocation request with the specified size.
    - Increases the requested size by 32 bytes to accommodate additional overhead.
    - Attempts to allocate memory using [`ggml_vk_host_malloc`](#ggml_vk_host_malloc) and catches any `vk::SystemError` exceptions.
    - If memory allocation fails, logs a warning and falls back to allocating a CPU buffer using `ggml_backend_buft_alloc_buffer`.
    - If successful, creates a `ggml_backend_buffer_t` from the allocated pointer and sets its buffer type and free function.
- **Output**: Returns a `ggml_backend_buffer_t` structure representing the allocated Vulkan host buffer, or a CPU buffer if the Vulkan allocation fails.
- **Functions called**:
    - [`ggml_vk_host_malloc`](#ggml_vk_host_malloc)


---
### ggml\_backend\_vk\_host\_buffer\_type\_get\_alignment<!-- {{#callable:ggml_backend_vk_host_buffer_type_get_alignment}} -->
Retrieves the minimum memory map alignment for Vulkan device buffers.
- **Inputs**:
    - `buft`: An enumeration value of type `ggml_backend_buffer_type_t` representing the type of buffer.
- **Control Flow**:
    - The function directly accesses the `minMemoryMapAlignment` property of the first Vulkan device in the `vk_instance.devices` array.
    - The input parameter `buft` is declared but not used in the function, indicating it may be for future use or for interface consistency.
- **Output**: Returns a `size_t` value representing the minimum memory map alignment required for Vulkan buffers.


---
### ggml\_backend\_vk\_host\_buffer\_type<!-- {{#callable:ggml_backend_vk_host_buffer_type}} -->
The `ggml_backend_vk_host_buffer_type` function initializes and returns a pointer to a static `ggml_backend_buffer_type` structure configured for Vulkan host buffer management.
- **Inputs**: None
- **Control Flow**:
    - A static instance of `ggml_backend_buffer_type` is defined and initialized with function pointers and device information.
    - The function ensures that the Vulkan device is initialized by calling `ggml_vk_instance_init()`.
    - It retrieves the Vulkan device using `ggml_vk_get_device(0)` to ensure device 0 is ready for use.
- **Output**: The function returns a pointer to the static `ggml_backend_vk_buffer_type_host` structure, which contains the interface and device information for Vulkan host buffer operations.
- **Functions called**:
    - [`ggml_backend_vk_reg`](#ggml_backend_vk_reg)
    - [`ggml_vk_instance_init`](#ggml_vk_instance_init)
    - [`ggml_vk_get_device`](#ggml_vk_get_device)


---
### ggml\_backend\_vk\_name<!-- {{#callable:ggml_backend_vk_name}} -->
Retrieves the Vulkan backend name from the given backend context.
- **Inputs**:
    - `backend`: A pointer to a `ggml_backend_t` structure that contains the Vulkan backend context.
- **Control Flow**:
    - The function casts the `context` member of the `backend` structure to a `ggml_backend_vk_context` pointer.
    - It accesses the `name` member of the `ggml_backend_vk_context` and returns its C-style string representation.
- **Output**: Returns a pointer to a constant character string representing the name of the Vulkan backend.


---
### ggml\_backend\_vk\_free<!-- {{#callable:ggml_backend_vk_free}} -->
This function frees the resources associated with a Vulkan backend.
- **Inputs**:
    - `backend`: A pointer to a `ggml_backend_t` structure representing the Vulkan backend to be freed.
- **Control Flow**:
    - The function begins by casting the `context` member of the `backend` to a `ggml_backend_vk_context` pointer.
    - It logs a debug message indicating the name of the Vulkan context being freed.
    - The [`ggml_vk_cleanup`](#ggml_vk_cleanup) function is called to perform any necessary cleanup operations for the Vulkan context.
    - Finally, the function deletes the Vulkan context and the backend itself to free the allocated memory.
- **Output**: The function does not return a value; it performs cleanup and deallocates memory.
- **Functions called**:
    - [`ggml_vk_cleanup`](#ggml_vk_cleanup)


---
### ggml\_backend\_vk\_get\_default\_buffer\_type<!-- {{#callable:ggml_backend_vk_get_default_buffer_type}} -->
Retrieves the default buffer type from the Vulkan backend context.
- **Inputs**:
    - `backend`: A pointer to a `ggml_backend_t` structure representing the Vulkan backend.
- **Control Flow**:
    - The function casts the `context` member of the `backend` structure to a `ggml_backend_vk_context` pointer.
    - It accesses the `device` member of the context and retrieves the `buffer_type`.
- **Output**: Returns a pointer to the default buffer type associated with the Vulkan device.


---
### ggml\_backend\_vk\_set\_tensor\_async<!-- {{#callable:ggml_backend_vk_set_tensor_async}} -->
Sets tensor data asynchronously in a Vulkan backend.
- **Inputs**:
    - `backend`: A pointer to the `ggml_backend_t` structure representing the Vulkan backend.
    - `tensor`: A pointer to the `ggml_tensor` structure that specifies the tensor to be updated.
    - `data`: A pointer to the data that will be written to the tensor.
    - `offset`: The offset in the tensor where the data writing should begin.
    - `size`: The size of the data to be written to the tensor.
- **Control Flow**:
    - Logs the size of the data being set for debugging purposes.
    - Retrieves the Vulkan backend context from the provided backend.
    - Asserts that the tensor's buffer type is supported by the Vulkan backend.
    - Retrieves the buffer context associated with the tensor's buffer.
    - Checks if the transfer context is expired; if so, creates a new transfer context and begins it.
    - Locks the existing transfer context if it is still valid.
    - Retrieves the Vulkan buffer associated with the tensor's buffer context.
    - Calls [`ggml_vk_buffer_write_async`](#ggml_vk_buffer_write_async) to write the data to the specified offset in the tensor asynchronously.
- **Output**: This function does not return a value; it performs an asynchronous operation to set the tensor data.
- **Functions called**:
    - [`ggml_backend_vk_get_default_buffer_type`](#ggml_backend_vk_get_default_buffer_type)
    - [`ggml_backend_vk_host_buffer_type`](#ggml_backend_vk_host_buffer_type)
    - [`ggml_vk_create_context`](#ggml_vk_create_context)
    - [`ggml_vk_ctx_begin`](#ggml_vk_ctx_begin)
    - [`ggml_vk_buffer_write_async`](#ggml_vk_buffer_write_async)
    - [`vk_tensor_offset`](#vk_tensor_offset)


---
### ggml\_backend\_vk\_get\_tensor\_async<!-- {{#callable:ggml_backend_vk_get_tensor_async}} -->
The `ggml_backend_vk_get_tensor_async` function asynchronously retrieves a portion of a tensor's data from a Vulkan backend.
- **Inputs**:
    - `backend`: A `ggml_backend_t` structure representing the Vulkan backend context.
    - `tensor`: A pointer to a `ggml_tensor` structure that specifies the tensor from which data is to be retrieved.
    - `data`: A pointer to the memory location where the retrieved tensor data will be stored.
    - `offset`: A size_t value indicating the offset in the tensor from which to start reading data.
    - `size`: A size_t value specifying the number of bytes to read from the tensor.
- **Control Flow**:
    - Logs the size of the data to be retrieved using `VK_LOG_DEBUG`.
    - Asserts that the tensor's buffer type is either the default buffer type or a host buffer type.
    - Retrieves the buffer context associated with the tensor's buffer.
    - Checks if the transfer context is expired; if so, it creates a new transfer context and begins the Vulkan context.
    - If the transfer context is valid, it locks the existing transfer context.
    - Retrieves the device buffer from the buffer context.
    - Calls [`ggml_vk_buffer_read_async`](#ggml_vk_buffer_read_async) to read the specified portion of the tensor data asynchronously.
- **Output**: The function does not return a value; instead, it initiates an asynchronous operation to read data into the provided memory location.
- **Functions called**:
    - [`ggml_backend_vk_get_default_buffer_type`](#ggml_backend_vk_get_default_buffer_type)
    - [`ggml_backend_vk_host_buffer_type`](#ggml_backend_vk_host_buffer_type)
    - [`ggml_vk_create_context`](#ggml_vk_create_context)
    - [`ggml_vk_ctx_begin`](#ggml_vk_ctx_begin)
    - [`ggml_vk_buffer_read_async`](#ggml_vk_buffer_read_async)
    - [`vk_tensor_offset`](#vk_tensor_offset)


---
### ggml\_backend\_vk\_cpy\_tensor\_async<!-- {{#callable:ggml_backend_vk_cpy_tensor_async}} -->
The `ggml_backend_vk_cpy_tensor_async` function asynchronously copies data from a source tensor to a destination tensor using Vulkan backend if certain conditions are met.
- **Inputs**:
    - `backend`: A handle to the backend context which contains Vulkan-related information.
    - `src`: A pointer to the source `ggml_tensor` from which data will be copied.
    - `dst`: A pointer to the destination `ggml_tensor` where data will be copied to.
- **Control Flow**:
    - The function starts by logging a debug message indicating its invocation.
    - It retrieves the Vulkan context from the provided `backend` argument.
    - It checks if the destination tensor's buffer type is compatible with the Vulkan backend and if the source tensor's buffer is a Vulkan buffer.
    - If the conditions are met, it retrieves the buffer contexts for both source and destination tensors.
    - It checks if the transfer context has expired; if so, it creates a new transfer context and begins it.
    - It retrieves the device buffers for both source and destination tensors.
    - It calls [`ggml_vk_buffer_copy_async`](#ggml_vk_buffer_copy_async) to perform the asynchronous copy operation from the source buffer to the destination buffer.
    - If the copy operation is initiated, the function returns true; otherwise, it returns false.
- **Output**: The function returns a boolean value indicating whether the asynchronous copy operation was successfully initiated.
- **Functions called**:
    - [`ggml_backend_vk_get_default_buffer_type`](#ggml_backend_vk_get_default_buffer_type)
    - [`ggml_backend_vk_host_buffer_type`](#ggml_backend_vk_host_buffer_type)
    - [`ggml_backend_buffer_is_vk`](#ggml_backend_buffer_is_vk)
    - [`ggml_vk_create_context`](#ggml_vk_create_context)
    - [`ggml_vk_ctx_begin`](#ggml_vk_ctx_begin)
    - [`ggml_vk_buffer_copy_async`](#ggml_vk_buffer_copy_async)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)


---
### ggml\_backend\_vk\_synchronize<!-- {{#callable:ggml_backend_vk_synchronize}} -->
The `ggml_backend_vk_synchronize` function synchronizes memory transfers in a Vulkan backend context.
- **Inputs**:
    - `backend`: A pointer to a `ggml_backend_t` structure that contains the Vulkan backend context.
- **Control Flow**:
    - Logs the entry into the function using `VK_LOG_DEBUG`.
    - Retrieves the Vulkan context from the `backend` structure.
    - Checks if the `transfer_ctx` is expired; if so, the function returns early.
    - Locks the `transfer_ctx` for safe access.
    - Ends the Vulkan context using [`ggml_vk_ctx_end`](#ggml_vk_ctx_end).
    - Iterates over the `in_memcpys` vector to perform memory copies from source to destination.
    - Submits the Vulkan context for processing with [`ggml_vk_submit`](#ggml_vk_submit) and waits for the fence to complete using [`ggml_vk_wait_for_fence`](#ggml_vk_wait_for_fence).
    - Iterates over the `out_memcpys` vector to perform memory copies from source to destination.
    - Resets the `transfer_ctx` to release resources.
- **Output**: The function does not return a value; it performs synchronization and memory transfer operations within the Vulkan backend context.
- **Functions called**:
    - [`ggml_vk_ctx_end`](#ggml_vk_ctx_end)
    - [`ggml_vk_submit`](#ggml_vk_submit)
    - [`ggml_vk_wait_for_fence`](#ggml_vk_wait_for_fence)


---
### ggml\_vk\_is\_empty<!-- {{#callable:ggml_vk_is_empty}} -->
Checks if a given `ggml_tensor` is empty or has a specific operation type.
- **Inputs**:
    - `node`: A pointer to a `ggml_tensor` structure that is being checked for emptiness.
- **Control Flow**:
    - The function first calls `ggml_is_empty(node)` to check if the tensor is empty.
    - It then checks if the operation type of the tensor (`node->op`) is one of the specified types: `GGML_OP_NONE`, `GGML_OP_RESHAPE`, `GGML_OP_TRANSPOSE`, `GGML_OP_VIEW`, or `GGML_OP_PERMUTE`.
    - If any of these conditions are true, the function returns true; otherwise, it returns false.
- **Output**: Returns a boolean value indicating whether the `ggml_tensor` is empty or has one of the specified operation types.
- **Functions called**:
    - [`ggml_is_empty`](../ggml.c.driver.md#ggml_is_empty)


---
### ggml\_backend\_vk\_graph\_compute<!-- {{#callable:ggml_backend_vk_graph_compute}} -->
Executes a Vulkan-based computation graph for a given backend.
- **Inputs**:
    - `backend`: A pointer to the `ggml_backend_t` structure representing the backend context.
    - `cgraph`: A pointer to the `ggml_cgraph` structure containing the computation graph with nodes to be processed.
- **Control Flow**:
    - Logs the number of nodes in the computation graph.
    - Initializes the Vulkan context and checks if shaders need to be loaded.
    - Preallocates buffers and allocates descriptor sets for the Vulkan device.
    - Iterates through the nodes of the computation graph to build the graph and accumulate matrix multiplication byte sizes.
    - Handles the submission of nodes to the GPU based on accumulated workload and node count.
    - If performance logging is enabled, manages query pools and timestamps for performance measurement.
    - Cleans up the Vulkan graph context after execution.
- **Output**: Returns `GGML_STATUS_SUCCESS` upon successful execution of the computation graph.
- **Functions called**:
    - [`ggml_vk_build_graph`](#ggml_vk_build_graph)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_vk_preallocate_buffers`](#ggml_vk_preallocate_buffers)
    - [`ggml_pipeline_allocate_descriptor_sets`](#ggml_pipeline_allocate_descriptor_sets)
    - [`ggml_vk_is_empty`](#ggml_vk_is_empty)
    - [`ggml_vk_create_context`](#ggml_vk_create_context)
    - [`ggml_vk_ctx_begin`](#ggml_vk_ctx_begin)
    - [`ggml_vk_ctx_end`](#ggml_vk_ctx_end)
    - [`ggml_vk_submit`](#ggml_vk_submit)
    - [`ggml_vk_graph_cleanup`](#ggml_vk_graph_cleanup)


---
### ggml\_backend\_vk\_guid<!-- {{#callable:ggml_backend_vk_guid}} -->
Returns a static GUID for the Vulkan backend.
- **Inputs**: None
- **Control Flow**:
    - The function defines a static variable `guid` initialized with a specific byte array.
    - The function returns a pointer to the static `guid` variable.
- **Output**: Returns a pointer to a static `ggml_guid_t` structure containing the Vulkan backend GUID.


---
### ggml\_backend\_vk\_init<!-- {{#callable:ggml_backend_vk_init}} -->
Initializes a Vulkan backend context for a specified device number.
- **Inputs**:
    - `dev_num`: A size_t representing the device number to initialize the Vulkan backend for.
- **Control Flow**:
    - Logs the initialization call with the specified device number using `VK_LOG_DEBUG`.
    - Creates a new instance of `ggml_backend_vk_context` to hold the Vulkan context.
    - Calls [`ggml_vk_init`](#ggml_vk_init) to initialize the Vulkan context with the created context and device number.
    - Creates a new `ggml_backend` structure, populating its fields with a GUID, interface, device registration, and the initialized context.
    - Returns the newly created Vulkan backend structure.
- **Output**: Returns a `ggml_backend_t` which is a pointer to the newly initialized Vulkan backend structure.
- **Functions called**:
    - [`ggml_vk_init`](#ggml_vk_init)
    - [`ggml_backend_vk_guid`](#ggml_backend_vk_guid)
    - [`ggml_backend_vk_reg`](#ggml_backend_vk_reg)


---
### ggml\_backend\_is\_vk<!-- {{#callable:ggml_backend_is_vk}} -->
Checks if the given `ggml_backend_t` is a valid Vulkan backend.
- **Inputs**:
    - `backend`: A pointer to a `ggml_backend_t` structure representing the backend to be checked.
- **Control Flow**:
    - The function first checks if the `backend` pointer is not NULL.
    - If `backend` is not NULL, it then calls [`ggml_guid_matches`](../ggml.c.driver.md#ggml_guid_matches) to compare the `guid` of the backend with the Vulkan backend GUID obtained from `ggml_backend_vk_guid()`.
- **Output**: Returns true if the `backend` is valid and matches the Vulkan GUID; otherwise, returns false.
- **Functions called**:
    - [`ggml_guid_matches`](../ggml.c.driver.md#ggml_guid_matches)
    - [`ggml_backend_vk_guid`](#ggml_backend_vk_guid)


---
### ggml\_backend\_vk\_get\_device\_count<!-- {{#callable:ggml_backend_vk_get_device_count}} -->
This function retrieves the count of Vulkan devices available.
- **Inputs**: None
- **Control Flow**:
    - The function directly calls another function `ggml_vk_get_device_count()` to obtain the device count.
    - It returns the result of the `ggml_vk_get_device_count()` function without any additional processing.
- **Output**: The output is an integer representing the number of Vulkan devices available.
- **Functions called**:
    - [`ggml_vk_get_device_count`](#ggml_vk_get_device_count)


---
### ggml\_backend\_vk\_get\_device\_description<!-- {{#callable:ggml_backend_vk_get_device_description}} -->
Retrieves the description of a specified Vulkan device.
- **Inputs**:
    - `device`: An integer representing the index of the Vulkan device whose description is to be retrieved.
    - `description`: A pointer to a character array where the device description will be stored.
    - `description_size`: A size_t value indicating the maximum size of the description buffer.
- **Control Flow**:
    - The function first asserts that the provided `device` index is valid by checking it against the size of `vk_instance.device_indices`.
    - If the assertion passes, it retrieves the actual device index from `vk_instance.device_indices` using the provided `device` index.
    - Finally, it calls the [`ggml_vk_get_device_description`](#ggml_vk_get_device_description) function with the retrieved device index, the description buffer, and its size.
- **Output**: The function does not return a value; instead, it populates the provided `description` buffer with the Vulkan device's description.
- **Functions called**:
    - [`ggml_vk_get_device_description`](#ggml_vk_get_device_description)


---
### ggml\_backend\_vk\_get\_device\_memory<!-- {{#callable:ggml_backend_vk_get_device_memory}} -->
Retrieves the total and free device memory size for a specified Vulkan physical device.
- **Inputs**:
    - `device`: An integer representing the index of the Vulkan physical device.
    - `free`: A pointer to a size_t variable where the free memory size will be stored.
    - `total`: A pointer to a size_t variable where the total memory size will be stored.
- **Control Flow**:
    - The function asserts that the provided device index is valid by checking it against the size of the device indices vector.
    - It retrieves the Vulkan physical device corresponding to the specified device index.
    - The function then obtains the memory properties of the physical device.
    - It iterates through the memory heaps to find a heap that is marked as device local.
    - Once a device local heap is found, it assigns the total and free memory sizes to the provided pointers and exits the loop.
- **Output**: The function does not return a value; instead, it updates the values pointed to by the `free` and `total` pointers with the corresponding memory sizes.


---
### ggml\_backend\_vk\_device\_get\_name<!-- {{#callable:ggml_backend_vk_device_get_name}} -->
Retrieves the name of a Vulkan device from its context.
- **Inputs**:
    - `dev`: A pointer to a `ggml_backend_dev_t` structure representing the Vulkan device.
- **Control Flow**:
    - The function casts the `context` member of the `dev` structure to a `ggml_backend_vk_device_context` pointer.
    - It accesses the `name` member of the `ggml_backend_vk_device_context` structure and returns its C-style string representation.
- **Output**: Returns a pointer to a constant character string representing the name of the Vulkan device.


---
### ggml\_backend\_vk\_device\_get\_description<!-- {{#callable:ggml_backend_vk_device_get_description}} -->
Retrieves the description of a Vulkan device from its context.
- **Inputs**:
    - `dev`: A pointer to a `ggml_backend_dev_t` structure representing the Vulkan device.
- **Control Flow**:
    - The function casts the `context` member of the `dev` structure to a `ggml_backend_vk_device_context` pointer.
    - It accesses the `description` member of the `ggml_backend_vk_device_context` and returns its C-style string representation.
- **Output**: A C-style string containing the description of the Vulkan device.


---
### ggml\_backend\_vk\_device\_get\_memory<!-- {{#callable:ggml_backend_vk_device_get_memory}} -->
Retrieves the memory information of a Vulkan device.
- **Inputs**:
    - `device`: A handle to the Vulkan device from which memory information is to be retrieved.
    - `free`: A pointer to a size_t variable where the amount of free memory will be stored.
    - `total`: A pointer to a size_t variable where the total amount of memory will be stored.
- **Control Flow**:
    - The function casts the `device` argument to a `ggml_backend_vk_device_context` type to access the Vulkan device context.
    - It then calls the [`ggml_backend_vk_get_device_memory`](#ggml_backend_vk_get_device_memory) function, passing the Vulkan device and the pointers to `free` and `total` to retrieve the memory information.
- **Output**: The function does not return a value; instead, it populates the `free` and `total` pointers with the respective memory values.
- **Functions called**:
    - [`ggml_backend_vk_get_device_memory`](#ggml_backend_vk_get_device_memory)


---
### ggml\_backend\_vk\_device\_get\_buffer\_type<!-- {{#callable:ggml_backend_vk_device_get_buffer_type}} -->
Retrieves the buffer type associated with a Vulkan device context.
- **Inputs**:
    - `dev`: A pointer to a `ggml_backend_dev_t` structure representing the device from which the buffer type is to be retrieved.
- **Control Flow**:
    - The function casts the `context` member of the `dev` structure to a `ggml_backend_vk_device_context` pointer.
    - It then calls the [`ggml_backend_vk_buffer_type`](#ggml_backend_vk_buffer_type) function, passing the `device` member of the context to retrieve the buffer type.
- **Output**: Returns a `ggml_backend_buffer_type_t` value representing the type of buffer associated with the Vulkan device.
- **Functions called**:
    - [`ggml_backend_vk_buffer_type`](#ggml_backend_vk_buffer_type)


---
### ggml\_backend\_vk\_device\_get\_host\_buffer\_type<!-- {{#callable:ggml_backend_vk_device_get_host_buffer_type}} -->
Retrieves the host buffer type for a Vulkan backend device.
- **Inputs**:
    - `dev`: An instance of `ggml_backend_dev_t` representing the Vulkan backend device.
- **Control Flow**:
    - The function starts by marking the input parameter `dev` as unused to avoid compiler warnings.
    - It then calls the function `ggml_backend_vk_host_buffer_type()` to obtain the host buffer type.
- **Output**: Returns a value of type `ggml_backend_buffer_type_t` that indicates the type of host buffer used by the Vulkan backend.
- **Functions called**:
    - [`ggml_backend_vk_host_buffer_type`](#ggml_backend_vk_host_buffer_type)


---
### ggml\_backend\_vk\_device\_get\_type<!-- {{#callable:ggml_backend_vk_device_get_type}} -->
Returns the type of the specified device as a GPU.
- **Inputs**:
    - `dev`: An instance of `ggml_backend_dev_t` representing the device whose type is to be determined.
- **Control Flow**:
    - The function takes a single input parameter `dev` but does not use it in the function body.
    - It directly returns the constant value `GGML_BACKEND_DEVICE_TYPE_GPU`.
- **Output**: The function outputs an enumeration value indicating that the device type is a GPU.


---
### ggml\_backend\_vk\_device\_get\_props<!-- {{#callable:ggml_backend_vk_device_get_props}} -->
Retrieves properties of a Vulkan device and populates a provided structure with the device's details.
- **Inputs**:
    - `dev`: An instance of `ggml_backend_dev_t` representing the Vulkan device whose properties are to be retrieved.
    - `props`: A pointer to a `struct ggml_backend_dev_props` where the device properties will be stored.
- **Control Flow**:
    - Calls [`ggml_backend_vk_device_get_name`](#ggml_backend_vk_device_get_name) to get the device name and assigns it to `props->name`.
    - Calls [`ggml_backend_vk_device_get_description`](#ggml_backend_vk_device_get_description) to get the device description and assigns it to `props->description`.
    - Calls [`ggml_backend_vk_device_get_type`](#ggml_backend_vk_device_get_type) to get the device type and assigns it to `props->type`.
    - Calls [`ggml_backend_vk_device_get_memory`](#ggml_backend_vk_device_get_memory) to retrieve the free and total memory of the device, storing the results in `props->memory_free` and `props->memory_total`.
    - Initializes the `props->caps` structure with specific capabilities of the device.
- **Output**: The function does not return a value; instead, it populates the `props` structure with the Vulkan device's properties.
- **Functions called**:
    - [`ggml_backend_vk_device_get_name`](#ggml_backend_vk_device_get_name)
    - [`ggml_backend_vk_device_get_description`](#ggml_backend_vk_device_get_description)
    - [`ggml_backend_vk_device_get_type`](#ggml_backend_vk_device_get_type)
    - [`ggml_backend_vk_device_get_memory`](#ggml_backend_vk_device_get_memory)


---
### ggml\_backend\_vk\_device\_init<!-- {{#callable:ggml_backend_vk_device_init}} -->
Initializes a Vulkan backend device using the provided device context.
- **Inputs**:
    - `dev`: A pointer to a `ggml_backend_dev_t` structure representing the device to be initialized.
    - `params`: A string containing parameters for initialization, which is unused in this function.
- **Control Flow**:
    - The function begins by marking the `params` argument as unused to avoid compiler warnings.
    - It retrieves the Vulkan device context from the `dev` structure.
    - Finally, it calls the [`ggml_backend_vk_init`](#ggml_backend_vk_init) function, passing the Vulkan device from the context to initialize the backend.
- **Output**: Returns a `ggml_backend_t` type that represents the initialized Vulkan backend.
- **Functions called**:
    - [`ggml_backend_vk_init`](#ggml_backend_vk_init)


---
### ggml\_backend\_vk\_device\_supports\_op<!-- {{#callable:ggml_backend_vk_device_supports_op}} -->
Determines if a Vulkan device supports a specific operation for a given tensor.
- **Inputs**:
    - `dev`: A `ggml_backend_dev_t` representing the Vulkan device context.
    - `op`: A pointer to a `ggml_tensor` structure that describes the operation to be performed.
- **Control Flow**:
    - The function begins by checking the type of operation specified in the `op` tensor.
    - For unary operations, it checks if the source tensor is contiguous and of compatible types.
    - For matrix multiplication operations, it verifies the types of the source tensors and checks for memory constraints.
    - For attention operations, it checks the dimensions and types of the source tensors to ensure compatibility.
    - For various other operations, it checks specific conditions related to tensor types and contiguity.
    - If any condition fails, the function returns false; otherwise, it returns true.
- **Output**: Returns a boolean value indicating whether the specified operation is supported by the Vulkan device for the given tensor.
- **Functions called**:
    - [`ggml_get_unary_op`](../ggml.c.driver.md#ggml_get_unary_op)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_vk_get_device`](#ggml_vk_get_device)
    - [`ggml_vk_dim01_contiguous`](#ggml_vk_dim01_contiguous)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)


---
### ggml\_backend\_vk\_device\_supports\_buft<!-- {{#callable:ggml_backend_vk_device_supports_buft}} -->
Checks if a Vulkan device supports a specific buffer type.
- **Inputs**:
    - `dev`: A handle to the Vulkan backend device.
    - `buft`: A handle to the buffer type being checked.
- **Control Flow**:
    - The function first checks if the name of the buffer type matches the expected Vulkan buffer type name.
    - If the names do not match, it immediately returns false.
    - It retrieves the Vulkan device context from the provided device handle.
    - It retrieves the buffer type context from the provided buffer type handle.
    - Finally, it compares the device index of the buffer type context with the device index of the Vulkan device context and returns the result.
- **Output**: Returns true if the Vulkan device supports the specified buffer type, otherwise returns false.


---
### ggml\_backend\_vk\_device\_offload\_op<!-- {{#callable:ggml_backend_vk_device_offload_op}} -->
Evaluates whether a given tensor operation can be offloaded to a Vulkan device based on its dimensions and operation type.
- **Inputs**:
    - `dev`: An instance of `ggml_backend_dev_t` representing the Vulkan device used for offloading operations.
    - `op`: A pointer to a `ggml_tensor` structure that contains the operation details, including its dimensions and type.
- **Control Flow**:
    - Defines a constant `min_batch_size` set to 32, which is the minimum batch size required for offloading.
    - Checks if the second dimension of the tensor (`op->ne[1]`) is greater than or equal to `min_batch_size` and that the operation type is not `GGML_OP_GET_ROWS`.
    - Alternatively, checks if the third dimension of the tensor (`op->ne[2]`) is greater than or equal to `min_batch_size` and that the operation type is `GGML_OP_MUL_MAT_ID`.
    - Returns true if either of the above conditions is satisfied, indicating that the operation can be offloaded; otherwise, returns false.
- **Output**: Returns a boolean value indicating whether the tensor operation can be offloaded to the Vulkan device based on the specified conditions.


---
### ggml\_backend\_vk\_reg\_get\_name<!-- {{#callable:ggml_backend_vk_reg_get_name}} -->
Returns the name of the Vulkan backend.
- **Inputs**:
    - `reg`: An enumeration value of type `ggml_backend_reg_t` representing a specific backend registration.
- **Control Flow**:
    - The function takes a single input parameter `reg` but does not use it within the function body.
    - It returns a constant string defined by `GGML_VK_NAME`.
- **Output**: A constant character pointer to the name of the Vulkan backend.


---
### ggml\_backend\_vk\_reg\_get\_device\_count<!-- {{#callable:ggml_backend_vk_reg_get_device_count}} -->
Retrieves the count of Vulkan devices registered in the backend.
- **Inputs**:
    - `reg`: An instance of `ggml_backend_reg_t` representing the backend registration, which is unused in this function.
- **Control Flow**:
    - The function starts by marking the `reg` parameter as unused to avoid compiler warnings.
    - It then calls the `ggml_backend_vk_get_device_count()` function to retrieve the number of Vulkan devices.
- **Output**: Returns the number of Vulkan devices as a `size_t` value.
- **Functions called**:
    - [`ggml_backend_vk_get_device_count`](#ggml_backend_vk_get_device_count)


---
### ggml\_backend\_vk\_reg\_get\_device<!-- {{#callable:ggml_backend_vk_reg_get_device}} -->
Retrieves a Vulkan backend device from a static list, initializing the list if it hasn't been initialized yet.
- **Inputs**:
    - `reg`: A `ggml_backend_reg_t` type representing the backend registration context.
    - `device`: A `size_t` representing the index of the device to retrieve from the static list.
- **Control Flow**:
    - A static vector `devices` is used to store the initialized Vulkan backend devices.
    - A static boolean `initialized` tracks whether the devices have been initialized.
    - A mutex is used to ensure thread safety during the initialization process.
    - If `initialized` is false, the function enters a critical section where it initializes the devices by iterating over the number of available Vulkan devices.
    - For each device, a new `ggml_backend_vk_device_context` is created, and its properties are set based on the device index and description.
    - Each initialized device is then added to the `devices` vector.
    - After initialization, `initialized` is set to true to prevent re-initialization.
    - The function asserts that the requested device index is valid and returns the corresponding device from the `devices` vector.
- **Output**: Returns a `ggml_backend_dev_t` representing the requested Vulkan backend device.
- **Functions called**:
    - [`ggml_backend_vk_get_device_count`](#ggml_backend_vk_get_device_count)
    - [`ggml_backend_vk_get_device_description`](#ggml_backend_vk_get_device_description)


---
### ggml\_backend\_vk\_reg<!-- {{#callable:ggml_backend_vk_reg}} -->
The `ggml_backend_vk_reg` function initializes a Vulkan backend and returns a registration structure.
- **Inputs**: None
- **Control Flow**:
    - A static variable `reg` of type `ggml_backend_reg` is defined and initialized with API version, interface, and a null context.
    - The function attempts to initialize the Vulkan instance by calling `ggml_vk_instance_init()`.
    - If the initialization is successful, the function returns a pointer to the static `reg` variable.
    - If a `vk::SystemError` exception is thrown during initialization, an error message is logged and the function returns a null pointer.
- **Output**: The function returns a pointer to a `ggml_backend_reg` structure if successful, or a null pointer if an error occurs.
- **Functions called**:
    - [`ggml_vk_instance_init`](#ggml_vk_instance_init)


---
### ggml\_vk\_instance\_validation\_ext\_available<!-- {{#callable:ggml_vk_instance_validation_ext_available}} -->
Checks if the Vulkan instance extension `VK_KHR_portability_enumeration` is available.
- **Inputs**:
    - `instance_extensions`: A vector of `vk::ExtensionProperties` representing the available Vulkan instance extensions.
- **Control Flow**:
    - The function checks if the `GGML_VULKAN_VALIDATE` preprocessor directive is defined.
    - If defined, it iterates over the `instance_extensions` vector to look for the `VK_KHR_portability_enumeration` extension.
    - If the extension is found, the function returns true immediately.
    - If the extension is not found, a warning message is printed to standard error.
    - If `GGML_VULKAN_VALIDATE` is not defined, the function skips the checks and returns false.
- **Output**: Returns true if the `VK_KHR_portability_enumeration` extension is found; otherwise, returns false.


---
### ggml\_vk\_instance\_portability\_enumeration\_ext\_available<!-- {{#callable:ggml_vk_instance_portability_enumeration_ext_available}} -->
Checks if the Vulkan portability enumeration extension is available for instance extensions on Apple platforms.
- **Inputs**:
    - `instance_extensions`: A vector of `vk::ExtensionProperties` representing the available instance extensions.
- **Control Flow**:
    - The function first checks if the code is being compiled on an Apple platform using the `__APPLE__` preprocessor directive.
    - If on Apple, it initializes a boolean variable `portability_enumeration_ext` to false.
    - It then iterates over each `properties` in the `instance_extensions` vector to check if the extension name matches 'VK_KHR_portability_enumeration'.
    - If a match is found, the function immediately returns true, indicating the extension is available.
    - If no match is found after the loop, a warning message is printed to standard error indicating the extension was not found.
    - Finally, the function returns false if the extension is not available.
- **Output**: Returns a boolean value: true if the Vulkan portability enumeration extension is available, false otherwise.


---
### ggml\_vk\_khr\_cooperative\_matrix\_support<!-- {{#callable:ggml_vk_khr_cooperative_matrix_support}} -->
Determines if cooperative matrix support is available based on the physical device properties and driver properties.
- **Inputs**:
    - `props`: A constant reference to a `vk::PhysicalDeviceProperties` object that contains properties of the physical device.
    - `driver_props`: A constant reference to a `vk::PhysicalDeviceDriverProperties` object that contains properties of the driver.
    - `arch`: An enumeration value of type `vk_device_architecture` representing the architecture of the device.
- **Control Flow**:
    - The function begins by checking the `vendorID` of the physical device properties.
    - If the vendor is Intel, it checks if the architecture is `INTEL_XE2` and returns true only for that architecture.
    - If the vendor is AMD, it checks the driver ID; if it is either proprietary or open-source, it returns true only for the `AMD_RDNA3` architecture.
    - For any other vendor, the function defaults to returning true, indicating support.
- **Output**: Returns a boolean value indicating whether cooperative matrix support is available for the specified device architecture.


---
### ggml\_vk\_print\_graph\_origin<!-- {{#callable:ggml_vk_print_graph_origin}} -->
The `ggml_vk_print_graph_origin` function recursively prints the operation names of a computation graph represented by `ggml_tensor` objects.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the current tensor in the computation graph.
    - `done`: A reference to a vector of pointers to `ggml_tensor` structures that have already been processed to avoid cycles.
    - `level`: An integer representing the current depth level in the recursion, defaulting to 0.
- **Control Flow**:
    - The function first checks if the current `tensor` has already been processed (exists in `done`) or if the recursion level exceeds 10, in which case it returns early.
    - It prints indentation based on the current `level` to visually represent the depth of the tensor in the graph.
    - The operation name of the current `tensor` is printed along with a boolean indicating if it has extra data (gpu).
    - The current `tensor` is added to the `done` vector to mark it as processed.
    - The function then iterates over the source tensors (`src`) of the current `tensor`, recursively calling itself for each non-null source tensor, increasing the `level` by 1.
- **Output**: The function does not return a value; instead, it outputs the operation names and GPU status of the tensors to the standard error stream.
- **Functions called**:
    - [`ggml_op_name`](../ggml.c.driver.md#ggml_op_name)


---
### ggml\_vk\_print\_tensor<!-- {{#callable:ggml_vk_print_tensor}} -->
Prints detailed information about a `ggml_tensor`, including its data, type, dimensions, and source tensors.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure containing the tensor data and metadata.
    - `name`: A string representing the name to be printed alongside the tensor information.
- **Control Flow**:
    - Checks if the tensor is stored in GPU memory by verifying if the buffer is not null and is a Vulkan buffer.
    - If the tensor is on the GPU, allocates memory for the tensor data and reads the data from the GPU buffer.
    - Prints the tensor's name, operation type, dimensions, and byte sizes to standard error.
    - If the tensor has source tensors, prints their details as well.
    - Calls [`ggml_vk_print_tensor_area`](#ggml_vk_print_tensor_area) to print a specific area of the tensor data.
    - Calls [`ggml_vk_print_graph_origin`](#ggml_vk_print_graph_origin) to print the origin of the tensor in the computation graph.
    - If the tensor was on the GPU, frees the allocated memory for the tensor data.
- **Output**: The function does not return a value; it outputs the tensor information to the standard error stream.
- **Functions called**:
    - [`ggml_backend_buffer_is_vk`](#ggml_backend_buffer_is_vk)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_vk_buffer_read`](#ggml_vk_buffer_read)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_op_name`](../ggml.c.driver.md#ggml_op_name)
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)
    - [`ggml_vk_print_tensor_area`](#ggml_vk_print_tensor_area)
    - [`ggml_vk_print_graph_origin`](#ggml_vk_print_graph_origin)


---
### ggml\_vk\_check\_results\_0<!-- {{#callable:ggml_vk_check_results_0}} -->
Checks and processes the results of a `ggml_tensor`, performing various operations based on its type and managing memory for tensor data.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that contains the operation type and source tensors for processing.
- **Control Flow**:
    - The function first checks if the operation of the tensor is `GGML_OP_TRANSPOSE`, in which case it returns immediately.
    - It increments a global `check_counter` and checks if it should proceed based on the value of `vk_output_tensor` and `vk_skip_checks`.
    - If conditions are met, it initializes a new `ggml_context` and prepares to clone the source tensors.
    - It iterates over the source tensors, cloning them and managing their memory based on their buffer type (host or Vulkan).
    - Depending on the operation type of the tensor, it calls the appropriate function to perform the operation and create a new tensor clone.
    - After processing, it builds a computation graph and executes it within the context.
    - Finally, it copies the result data to a buffer, cleans up allocated memory, and logs the end of the function.
- **Output**: The function does not return a value but modifies global state and allocates memory for the result tensor, which is copied to a separate buffer for further use.
- **Functions called**:
    - [`ggml_init`](../ggml.c.driver.md#ggml_init)
    - [`ggml_dup_tensor`](../ggml.c.driver.md#ggml_dup_tensor)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_backend_buffer_is_host`](../ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
    - [`ggml_backend_buffer_is_vk`](#ggml_backend_buffer_is_vk)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_vk_dim01_contiguous`](#ggml_vk_dim01_contiguous)
    - [`ggml_vk_buffer_read`](#ggml_vk_buffer_read)
    - [`ggml_vk_print_tensor`](#ggml_vk_print_tensor)
    - [`ggml_flash_attn_ext`](../ggml.c.driver.md#ggml_flash_attn_ext)
    - [`ggml_mul_mat`](../ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_mul_mat_id`](../ggml.c.driver.md#ggml_mul_mat_id)
    - [`ggml_sub`](../ggml.c.driver.md#ggml_sub)
    - [`ggml_mul`](../ggml.c.driver.md#ggml_mul)
    - [`ggml_div`](../ggml.c.driver.md#ggml_div)
    - [`ggml_concat`](../ggml.c.driver.md#ggml_concat)
    - [`ggml_upscale_ext`](../ggml.c.driver.md#ggml_upscale_ext)
    - [`ggml_scale`](../ggml.c.driver.md#ggml_scale)
    - [`ggml_sqr`](../ggml.c.driver.md#ggml_sqr)
    - [`ggml_sin`](../ggml.c.driver.md#ggml_sin)
    - [`ggml_cos`](../ggml.c.driver.md#ggml_cos)
    - [`ggml_pad`](../ggml.c.driver.md#ggml_pad)
    - [`ggml_repeat`](../ggml.c.driver.md#ggml_repeat)
    - [`ggml_repeat_back`](../ggml.c.driver.md#ggml_repeat_back)
    - [`ggml_add`](../ggml.c.driver.md#ggml_add)
    - [`ggml_acc`](../ggml.c.driver.md#ggml_acc)
    - [`ggml_norm`](../ggml.c.driver.md#ggml_norm)
    - [`ggml_group_norm`](../ggml.c.driver.md#ggml_group_norm)
    - [`ggml_rms_norm`](../ggml.c.driver.md#ggml_rms_norm)
    - [`ggml_rms_norm_back`](../ggml.c.driver.md#ggml_rms_norm_back)
    - [`ggml_silu_back`](../ggml.c.driver.md#ggml_silu_back)
    - [`ggml_l2_norm`](../ggml.c.driver.md#ggml_l2_norm)
    - [`ggml_soft_max_ext`](../ggml.c.driver.md#ggml_soft_max_ext)
    - [`ggml_soft_max`](../ggml.c.driver.md#ggml_soft_max)
    - [`ggml_soft_max_ext_back`](../ggml.c.driver.md#ggml_soft_max_ext_back)
    - [`ggml_diag_mask_inf`](../ggml.c.driver.md#ggml_diag_mask_inf)
    - [`ggml_rope_multi`](../ggml.c.driver.md#ggml_rope_multi)
    - [`ggml_rope_ext`](../ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_get_unary_op`](../ggml.c.driver.md#ggml_get_unary_op)
    - [`ggml_silu`](../ggml.c.driver.md#ggml_silu)
    - [`ggml_gelu`](../ggml.c.driver.md#ggml_gelu)
    - [`ggml_gelu_quick`](../ggml.c.driver.md#ggml_gelu_quick)
    - [`ggml_relu`](../ggml.c.driver.md#ggml_relu)
    - [`ggml_tanh`](../ggml.c.driver.md#ggml_tanh)
    - [`ggml_sigmoid`](../ggml.c.driver.md#ggml_sigmoid)
    - [`ggml_op_name`](../ggml.c.driver.md#ggml_op_name)
    - [`ggml_dup`](../ggml.c.driver.md#ggml_dup)
    - [`ggml_cpy`](../ggml.c.driver.md#ggml_cpy)
    - [`ggml_cont_4d`](../ggml.c.driver.md#ggml_cont_4d)
    - [`ggml_reshape_4d`](../ggml.c.driver.md#ggml_reshape_4d)
    - [`ggml_view_4d`](../ggml.c.driver.md#ggml_view_4d)
    - [`ggml_permute`](../ggml.c.driver.md#ggml_permute)
    - [`ggml_transpose`](../ggml.c.driver.md#ggml_transpose)
    - [`ggml_get_rows`](../ggml.c.driver.md#ggml_get_rows)
    - [`ggml_argsort`](../ggml.c.driver.md#ggml_argsort)
    - [`ggml_sum`](../ggml.c.driver.md#ggml_sum)
    - [`ggml_sum_rows`](../ggml.c.driver.md#ggml_sum_rows)
    - [`ggml_argmax`](../ggml.c.driver.md#ggml_argmax)
    - [`ggml_count_equal`](../ggml.c.driver.md#ggml_count_equal)
    - [`ggml_im2col`](../ggml.c.driver.md#ggml_im2col)
    - [`ggml_timestep_embedding`](../ggml.c.driver.md#ggml_timestep_embedding)
    - [`ggml_conv_transpose_1d`](../ggml.c.driver.md#ggml_conv_transpose_1d)
    - [`ggml_pool_2d`](../ggml.c.driver.md#ggml_pool_2d)
    - [`ggml_leaky_relu`](../ggml.c.driver.md#ggml_leaky_relu)
    - [`ggml_rwkv_wkv6`](../ggml.c.driver.md#ggml_rwkv_wkv6)
    - [`ggml_rwkv_wkv7`](../ggml.c.driver.md#ggml_rwkv_wkv7)
    - [`ggml_opt_step_adamw`](../ggml.c.driver.md#ggml_opt_step_adamw)
    - [`ggml_new_graph`](../ggml.c.driver.md#ggml_new_graph)
    - [`ggml_build_forward_expand`](../ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_graph_compute_with_ctx`](../ggml-cpu/ggml-cpu.c.driver.md#ggml_graph_compute_with_ctx)
    - [`ggml_free`](../ggml.c.driver.md#ggml_free)


---
### ggml\_vk\_check\_results\_1<!-- {{#callable:ggml_vk_check_results_1}} -->
Checks the results of a tensor operation against expected values and logs discrepancies.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor whose results are to be checked.
- **Control Flow**:
    - The function first checks if the tensor operation is a transpose; if so, it returns immediately.
    - It verifies if the output tensor and check counter conditions are met; if not, it returns.
    - Debug logging is performed to indicate the start of the check.
    - The function retrieves source tensors and allocates memory for tensor data if the backend buffer is Vulkan.
    - It iterates over the tensor dimensions to compare computed results with expected values, logging errors if discrepancies are found.
    - If the average error exceeds a threshold or is NaN, detailed error information is logged, and the function aborts.
    - Finally, it cleans up allocated resources and logs the end of the function.
- **Output**: The function does not return a value but logs errors and debug information to standard error output.
- **Functions called**:
    - [`ggml_backend_buffer_is_vk`](#ggml_backend_buffer_is_vk)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`vk_tensor_offset`](#vk_tensor_offset)
    - [`ggml_vk_buffer_read`](#ggml_vk_buffer_read)
    - [`ggml_fp16_to_fp32`](../ggml.c.driver.md#ggml_fp16_to_fp32)
    - [`ggml_type_name`](../ggml.c.driver.md#ggml_type_name)
    - [`ggml_op_name`](../ggml.c.driver.md#ggml_op_name)
    - [`ggml_vk_print_tensor_area`](#ggml_vk_print_tensor_area)
    - [`ggml_vk_print_graph_origin`](#ggml_vk_print_graph_origin)


