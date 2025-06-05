# Purpose
This Python code file is designed to manage and interact with server processes, specifically for machine learning models hosted on a server. It provides a comprehensive set of functionalities to start, stop, and communicate with these server processes, which are configured to run various machine learning models. The `ServerProcess` class is central to this functionality, encapsulating the configuration and lifecycle management of a server instance. It includes methods to start the server with specific configurations, make HTTP requests to the server, and handle streaming responses. The code also defines several server presets through the `ServerPreset` class, which provides pre-configured server instances for different models, such as "tinyllama2" and "bert_bge_small".

Additionally, the file includes utility functions for parallel execution of functions ([`parallel_function_calls`](#cpp/tools/server/tests/utilsparallel_function_calls)), regex matching ([`match_regex`](#cpp/tools/server/tests/utilsmatch_regex)), file downloading ([`download_file`](#cpp/tools/server/tests/utilsdownload_file)), and checking environment configurations ([`is_slow_test_allowed`](#cpp/tools/server/tests/utilsis_slow_test_allowed)). These utilities support the main server management functionality by providing necessary auxiliary operations. The code is structured to be used as a library, with the potential for integration into larger systems that require dynamic server management and interaction with machine learning models. The presence of environment variable checks and default configurations suggests that the code is designed to be flexible and adaptable to different deployment environments.
# Imports and Dependencies

---
- `subprocess`
- `os`
- `re`
- `json`
- `sys`
- `requests`
- `time`
- `concurrent.futures.ThreadPoolExecutor`
- `concurrent.futures.as_completed`
- `typing.Any`
- `typing.Callable`
- `typing.ContextManager`
- `typing.Iterable`
- `typing.Iterator`
- `typing.List`
- `typing.Literal`
- `typing.Tuple`
- `typing.Set`
- `re.RegexFlag`
- `wget`


# Global Variables

---
### DEFAULT\_HTTP\_TIMEOUT
- **Type**: `int`
- **Description**: `DEFAULT_HTTP_TIMEOUT` is an integer variable that sets the default timeout duration for HTTP requests in seconds. It is initially set to 12 seconds, but can be overridden to 30 seconds if certain environment variables (`LLAMA_SANITIZE` or `GITHUB_ACTION`) are present.
- **Use**: This variable is used to specify the default timeout for HTTP requests made by the server process.


---
### server\_instances
- **Type**: `Set[ServerProcess]`
- **Description**: The `server_instances` variable is a global set that holds instances of the `ServerProcess` class. It is initialized as an empty set and is used to keep track of all active server processes that are started within the application.
- **Use**: This variable is used to manage and track the lifecycle of server processes, allowing for operations such as starting, stopping, and monitoring these processes.


# Classes

---
### ServerResponse<!-- {{#class:llama.cpp/tools/server/tests/utils.ServerResponse}} -->
- **Members**:
    - `headers`: A dictionary containing the headers of the server response.
    - `status_code`: An integer representing the HTTP status code of the response.
    - `body`: The body of the response, which can be a dictionary or any other data type.
- **Description**: The `ServerResponse` class is a simple data structure used to encapsulate the details of a server's response, including headers, status code, and body content. It is designed to store the essential components of an HTTP response, allowing for easy access and manipulation of response data within the application.


---
### ServerProcess<!-- {{#class:llama.cpp/tools/server/tests/utils.ServerProcess}} -->
- **Members**:
    - `debug`: Indicates if the server is running in debug mode.
    - `server_port`: Specifies the port on which the server will run.
    - `server_host`: Defines the host address for the server.
    - `model_hf_repo`: Specifies the Hugging Face repository for the model.
    - `model_hf_file`: Indicates the file path for the model in the Hugging Face repository.
    - `model_alias`: Provides an alias for the model.
    - `temperature`: Sets the temperature for model predictions.
    - `seed`: Defines the seed for random number generation.
    - `model_url`: Specifies the URL for the model.
    - `model_file`: Indicates the file path for the model.
    - `model_draft`: Specifies the draft version of the model.
    - `n_threads`: Defines the number of threads to use.
    - `n_gpu_layer`: Specifies the number of GPU layers to use.
    - `n_batch`: Sets the batch size for processing.
    - `n_ubatch`: Defines the micro-batch size for processing.
    - `n_ctx`: Specifies the context size for the model.
    - `n_ga`: Indicates the number of group attention layers.
    - `n_ga_w`: Defines the width of group attention layers.
    - `n_predict`: Sets the number of predictions to make.
    - `n_prompts`: Indicates the number of prompts to use.
    - `slot_save_path`: Specifies the path to save slot data.
    - `id_slot`: Defines the ID for the slot.
    - `cache_prompt`: Indicates if prompts should be cached.
    - `n_slots`: Specifies the number of slots to use.
    - `ctk`: Defines the context key.
    - `ctv`: Specifies the context value.
    - `fa`: Indicates if feature aggregation is enabled.
    - `server_continuous_batching`: Indicates if continuous batching is enabled on the server.
    - `server_embeddings`: Specifies if embeddings are enabled on the server.
    - `server_reranking`: Indicates if reranking is enabled on the server.
    - `server_metrics`: Specifies if metrics are enabled on the server.
    - `server_slots`: Indicates if slots are enabled on the server.
    - `pooling`: Defines the pooling method to use.
    - `draft`: Specifies the draft level for the model.
    - `api_key`: Indicates the API key for authentication.
    - `lora_files`: Lists the LoRA files to use.
    - `disable_ctx_shift`: Indicates if context shift is disabled.
    - `draft_min`: Specifies the minimum draft level.
    - `draft_max`: Defines the maximum draft level.
    - `no_webui`: Indicates if the web UI is disabled.
    - `jinja`: Specifies if Jinja templating is enabled.
    - `reasoning_format`: Defines the format for reasoning.
    - `reasoning_budget`: Specifies the budget for reasoning.
    - `chat_template`: Indicates the chat template to use.
    - `chat_template_file`: Specifies the file path for the chat template.
    - `server_path`: Defines the path to the server executable.
    - `mmproj_url`: Specifies the URL for the mmproj file.
    - `process`: Holds the subprocess for the server process.
- **Description**: The `ServerProcess` class is designed to manage the configuration and execution of a server process, particularly for machine learning models hosted on a server. It includes a wide range of configuration options such as server host and port, model repository and file paths, and various operational parameters like batch size, number of threads, and GPU layers. The class also supports advanced features like continuous batching, embeddings, reranking, and metrics collection. It provides methods to start and stop the server, make HTTP requests, and handle streaming data, making it a comprehensive solution for managing server-based model deployments.
- **Methods**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.__init__`](#ServerProcess__init__)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.stop`](#ServerProcessstop)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_stream_request`](#ServerProcessmake_stream_request)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_any_request`](#ServerProcessmake_any_request)

**Methods**

---
#### ServerProcess\.\_\_init\_\_<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerProcess.__init__}} -->
The `__init__` method initializes a `ServerProcess` instance by setting certain attributes based on environment variables.
- **Inputs**: None
- **Control Flow**:
    - Checks if the environment variable 'N_GPU_LAYERS' is set; if so, assigns its integer value to `self.n_gpu_layer`.
    - Checks if the environment variable 'DEBUG' is set; if so, sets `self.debug` to `True`.
    - Checks if the environment variable 'PORT' is set; if so, assigns its integer value to `self.server_port`.
- **Output**: This method does not return any value; it initializes instance attributes based on environment variables.
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)  (Base Class)


---
#### ServerProcess\.start<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerProcess.start}} -->
The `start` method initializes and starts a server process with specified configurations and waits for it to become ready within a given timeout period.
- **Inputs**:
    - `timeout_seconds`: An optional integer specifying the maximum number of seconds to wait for the server to start, defaulting to `DEFAULT_HTTP_TIMEOUT`.
- **Control Flow**:
    - Determine the server executable path based on `self.server_path`, environment variables, or default paths depending on the operating system.
    - Construct a list of server arguments using various instance attributes such as `server_host`, `server_port`, `temperature`, `seed`, and others if they are set.
    - Print the command used to start the server for debugging purposes.
    - Set process creation flags for Windows systems to run the server in a detached process without a console window.
    - Start the server process using `subprocess.Popen` with the constructed command and environment variables, and add the process to `server_instances`.
    - Print the process ID of the started server and the current process ID for reference.
    - Enter a loop to repeatedly check the server's health endpoint until it responds with a status code of 200 or the timeout is reached.
    - If the server process terminates unexpectedly, raise a `RuntimeError`.
    - If the server does not start within the specified timeout, raise a `TimeoutError`.
- **Output**: The method does not return any value but raises exceptions if the server fails to start or dies unexpectedly.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](#ServerProcessmake_request)
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)  (Base Class)


---
#### ServerProcess\.stop<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerProcess.stop}} -->
The `stop` method terminates the server process associated with the `ServerProcess` instance and removes it from the global `server_instances` set.
- **Inputs**: None
- **Control Flow**:
    - Checks if the current instance is in the `server_instances` set and removes it if present.
    - Checks if the `process` attribute is not `None`, indicating an active server process.
    - If a process exists, it prints a message indicating the server is stopping, kills the process, and sets the `process` attribute to `None`.
- **Output**: The method does not return any value (returns `None`).
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)  (Base Class)


---
#### ServerProcess\.make\_request<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerProcess.make_request}} -->
The `make_request` method sends HTTP requests to a server and returns the server's response encapsulated in a [`ServerResponse`](#cpp/tools/server/tests/utilsServerResponse) object.
- **Inputs**:
    - `method`: A string representing the HTTP method to use for the request, such as 'GET', 'POST', or 'OPTIONS'.
    - `path`: A string representing the path of the URL to which the request is sent.
    - `data`: An optional dictionary or any data type to be sent as the JSON payload in the request body, defaulting to None.
    - `headers`: An optional dictionary of HTTP headers to include in the request, defaulting to None.
    - `timeout`: An optional float representing the timeout for the request in seconds, defaulting to None.
- **Control Flow**:
    - Constructs the full URL using the server host, port, and provided path.
    - Initializes a boolean `parse_body` to determine if the response body should be parsed as JSON.
    - Checks the HTTP method: if 'GET', sends a GET request and sets `parse_body` to True; if 'POST', sends a POST request with JSON data and sets `parse_body` to True; if 'OPTIONS', sends an OPTIONS request; otherwise, raises a ValueError for unimplemented methods.
    - Creates a [`ServerResponse`](#cpp/tools/server/tests/utilsServerResponse) object to store the response headers, status code, and optionally the JSON-parsed body if `parse_body` is True.
    - Prints the server response body in a formatted JSON string.
    - Returns the [`ServerResponse`](#cpp/tools/server/tests/utilsServerResponse) object.
- **Output**: A [`ServerResponse`](#cpp/tools/server/tests/utilsServerResponse) object containing the response headers, status code, and optionally the JSON-parsed body if applicable.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerResponse`](#cpp/tools/server/tests/utilsServerResponse)
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)  (Base Class)


---
#### ServerProcess\.make\_stream\_request<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerProcess.make_stream_request}} -->
The `make_stream_request` method sends a POST request to a specified server path and yields JSON-decoded data from the server's streamed response lines.
- **Inputs**:
    - `method`: A string specifying the HTTP method to use, which must be 'POST'.
    - `path`: A string representing the server path to which the request is made.
    - `data`: An optional dictionary containing the JSON data to be sent with the request.
    - `headers`: An optional dictionary of HTTP headers to include in the request.
- **Control Flow**:
    - Constructs the URL using the server host and port from the instance and the provided path.
    - Checks if the method is 'POST'; if not, raises a ValueError for unimplemented methods.
    - Sends a POST request to the constructed URL with the provided headers and data, enabling streaming of the response.
    - Iterates over each line in the streamed response, decoding it from bytes to a UTF-8 string.
    - Checks if the line contains '[DONE]' and breaks the loop if found, indicating the end of the stream.
    - If a line starts with 'data: ', it extracts the JSON data from the line, prints it, and yields it as a dictionary.
- **Output**: An iterator that yields dictionaries containing JSON-decoded data from the server's response.
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)  (Base Class)


---
#### ServerProcess\.make\_any\_request<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerProcess.make_any_request}} -->
The `make_any_request` method sends HTTP requests to a server, handling both streaming and non-streaming responses, and processes the response data accordingly.
- **Inputs**:
    - `method`: A string representing the HTTP method to use for the request, such as 'GET' or 'POST'.
    - `path`: A string representing the path of the URL to which the request is sent.
    - `data`: An optional dictionary containing the data to be sent with the request; it may include a 'stream' key to indicate if the request should be streamed.
    - `headers`: An optional dictionary of HTTP headers to include in the request.
    - `timeout`: An optional float specifying the timeout duration for the request.
- **Control Flow**:
    - Check if the 'stream' key in the data dictionary is set to True.
    - If streaming, initialize lists for content, reasoning content, and tool calls, and variables for counting parts.
    - Iterate over chunks from the [`make_stream_request`](#ServerProcessmake_stream_request) method, processing each chunk's 'choices' and appending data to the respective lists.
    - If not streaming, use the [`make_request`](#ServerProcessmake_request) method to send a standard HTTP request and check the response status code.
    - Return a structured dictionary containing the processed response data if streaming, or the response body if not.
- **Output**: A dictionary containing the processed response data, structured differently based on whether the request was streamed or not.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_stream_request`](#ServerProcessmake_stream_request)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](#ServerProcessmake_request)
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)  (Base Class)



---
### ServerPreset<!-- {{#class:llama.cpp/tools/server/tests/utils.ServerPreset}} -->
- **Description**: The `ServerPreset` class provides a collection of static methods that configure and return instances of the `ServerProcess` class with predefined settings for various server configurations. Each method sets specific parameters such as model repository, file, alias, context size, batch size, and other server options, allowing for easy instantiation of server processes tailored to different use cases, such as embedding, reranking, or infill tasks.
- **Methods**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama2`](#ServerPresettinyllama2)
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.bert_bge_small`](#ServerPresetbert_bge_small)
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.bert_bge_small_with_fa`](#ServerPresetbert_bge_small_with_fa)
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama_infill`](#ServerPresettinyllama_infill)
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.stories15m_moe`](#ServerPresetstories15m_moe)
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.jina_reranker_tiny`](#ServerPresetjina_reranker_tiny)
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinygemma3`](#ServerPresettinygemma3)

**Methods**

---
#### ServerPreset\.tinyllama2<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama2}} -->
The `tinyllama2` method initializes and returns a [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance configured for the 'tinyllama-2' model.
- **Decorators**: `@staticmethod`
- **Inputs**: None
- **Control Flow**:
    - A new instance of [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) is created.
    - The `model_hf_repo` attribute is set to 'ggml-org/models'.
    - The `model_hf_file` attribute is set to 'tinyllamas/stories260K.gguf'.
    - The `model_alias` attribute is set to 'tinyllama-2'.
    - The `n_ctx` attribute is set to 512.
    - The `n_batch` attribute is set to 32.
    - The `n_slots` attribute is set to 2.
    - The `n_predict` attribute is set to 64.
    - The `seed` attribute is set to 42.
    - The configured [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance is returned.
- **Output**: A [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance configured with specific model parameters for 'tinyllama-2'.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerPreset`](#cpp/tools/server/tests/utilsServerPreset)  (Base Class)


---
#### ServerPreset\.bert\_bge\_small<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerPreset.bert_bge_small}} -->
The `bert_bge_small` method configures and returns a [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance with specific settings for the BERT BGE small model.
- **Decorators**: `@staticmethod`
- **Inputs**: None
- **Control Flow**:
    - A new [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance is created.
    - The `model_hf_repo` attribute is set to 'ggml-org/models'.
    - The `model_hf_file` attribute is set to 'bert-bge-small/ggml-model-f16.gguf'.
    - The `model_alias` attribute is set to 'bert-bge-small'.
    - The `n_ctx` attribute is set to 512, defining the context size.
    - The `n_batch` and `n_ubatch` attributes are both set to 128, defining batch sizes.
    - The `n_slots` attribute is set to 2, defining the number of slots.
    - The `seed` attribute is set to 42, ensuring reproducibility.
    - The `server_embeddings` attribute is set to True, enabling server embeddings.
    - The configured [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance is returned.
- **Output**: A configured [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance for the BERT BGE small model.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerPreset`](#cpp/tools/server/tests/utilsServerPreset)  (Base Class)


---
#### ServerPreset\.bert\_bge\_small\_with\_fa<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerPreset.bert_bge_small_with_fa}} -->
The `bert_bge_small_with_fa` method configures and returns a [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance with specific settings for the BERT BGE small model, including enabling the FA (Feature Augmentation) option.
- **Decorators**: `@staticmethod`
- **Inputs**: None
- **Control Flow**:
    - A new [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance is created.
    - The `model_hf_repo` attribute is set to 'ggml-org/models'.
    - The `model_hf_file` attribute is set to 'bert-bge-small/ggml-model-f16.gguf'.
    - The `model_alias` attribute is set to 'bert-bge-small'.
    - The `n_ctx` attribute is set to 1024, indicating the context size.
    - The `n_batch` and `n_ubatch` attributes are both set to 300, indicating batch sizes.
    - The `n_slots` attribute is set to 2, indicating the number of slots.
    - The `fa` attribute is set to `True`, enabling feature augmentation.
    - The `seed` attribute is set to 42, providing a seed for random operations.
    - The `server_embeddings` attribute is set to `True`, enabling server embeddings.
    - The configured [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance is returned.
- **Output**: A configured [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance with specific settings for the BERT BGE small model with feature augmentation enabled.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerPreset`](#cpp/tools/server/tests/utilsServerPreset)  (Base Class)


---
#### ServerPreset\.tinyllama\_infill<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama_infill}} -->
The `tinyllama_infill` method initializes and returns a [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) configured for the 'tinyllama-infill' model with specific parameters.
- **Decorators**: `@staticmethod`
- **Inputs**: None
- **Control Flow**:
    - A new instance of [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) is created.
    - The `model_hf_repo` attribute is set to 'ggml-org/models'.
    - The `model_hf_file` attribute is set to 'tinyllamas/stories260K-infill.gguf'.
    - The `model_alias` attribute is set to 'tinyllama-infill'.
    - The `n_ctx` attribute is set to 2048, defining the context size.
    - The `n_batch` attribute is set to 1024, defining the batch size.
    - The `n_slots` attribute is set to 1, indicating the number of slots.
    - The `n_predict` attribute is set to 64, defining the number of predictions.
    - The `temperature` attribute is set to 0.0, affecting randomness in predictions.
    - The `seed` attribute is set to 42, ensuring reproducibility.
    - The configured [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance is returned.
- **Output**: A [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance configured with specific model and server parameters for the 'tinyllama-infill' model.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerPreset`](#cpp/tools/server/tests/utilsServerPreset)  (Base Class)


---
#### ServerPreset\.stories15m\_moe<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerPreset.stories15m_moe}} -->
The `stories15m_moe` method configures and returns a [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance with specific settings for the 'stories15M_MOE' model.
- **Decorators**: `@staticmethod`
- **Inputs**: None
- **Control Flow**:
    - A new [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance is created.
    - The `model_hf_repo` attribute is set to 'ggml-org/stories15M_MOE'.
    - The `model_hf_file` attribute is set to 'stories15M_MOE-F16.gguf'.
    - The `model_alias` attribute is set to 'stories15m-moe'.
    - The `n_ctx` attribute is set to 2048.
    - The `n_batch` attribute is set to 1024.
    - The `n_slots` attribute is set to 1.
    - The `n_predict` attribute is set to 64.
    - The `temperature` attribute is set to 0.0.
    - The `seed` attribute is set to 42.
    - The configured [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance is returned.
- **Output**: A [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance configured with specific model and server settings.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerPreset`](#cpp/tools/server/tests/utilsServerPreset)  (Base Class)


---
#### ServerPreset\.jina\_reranker\_tiny<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerPreset.jina_reranker_tiny}} -->
The `jina_reranker_tiny` method initializes and returns a [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) configured for reranking using a specific model.
- **Decorators**: `@staticmethod`
- **Inputs**: None
- **Control Flow**:
    - A new instance of [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) is created.
    - The `model_hf_repo` attribute is set to 'ggml-org/models'.
    - The `model_hf_file` attribute is set to 'jina-reranker-v1-tiny-en/ggml-model-f16.gguf'.
    - The `model_alias` attribute is set to 'jina-reranker'.
    - The `n_ctx` attribute is set to 512.
    - The `n_batch` attribute is set to 512.
    - The `n_slots` attribute is set to 1.
    - The `seed` attribute is set to 42.
    - The `server_reranking` attribute is set to True.
    - The configured [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance is returned.
- **Output**: A [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance configured for reranking with specific model settings.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerPreset`](#cpp/tools/server/tests/utilsServerPreset)  (Base Class)


---
#### ServerPreset\.tinygemma3<!-- {{#callable:llama.cpp/tools/server/tests/utils.ServerPreset.tinygemma3}} -->
The `tinygemma3` method initializes and returns a [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance configured for the 'tinygemma3' model with specific parameters.
- **Decorators**: `@staticmethod`
- **Inputs**: None
- **Control Flow**:
    - A new [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance is created and assigned to the variable `server`.
    - The `model_hf_repo` attribute of `server` is set to 'ggml-org/tinygemma3-GGUF'.
    - The `model_hf_file` attribute of `server` is set to 'tinygemma3-Q8_0.gguf'.
    - The `mmproj_url` attribute of `server` is set to a specific URL pointing to the 'mmproj-tinygemma3.gguf' file.
    - The `model_alias` attribute of `server` is set to 'tinygemma3'.
    - The `n_ctx` attribute of `server` is set to 1024, defining the context size.
    - The `n_batch` attribute of `server` is set to 32, defining the batch size.
    - The `n_slots` attribute of `server` is set to 2, defining the number of slots.
    - The `n_predict` attribute of `server` is set to 4, defining the number of predictions.
    - The `seed` attribute of `server` is set to 42, ensuring reproducibility.
    - The configured `server` instance is returned.
- **Output**: The method returns a [`ServerProcess`](#cpp/tools/server/tests/utilsServerProcess) instance configured with specific attributes for the 'tinygemma3' model.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess`](#cpp/tools/server/tests/utilsServerProcess)
- **See also**: [`llama.cpp/tools/server/tests/utils.ServerPreset`](#cpp/tools/server/tests/utilsServerPreset)  (Base Class)



# Functions

---
### parallel\_function\_calls<!-- {{#callable:llama.cpp/tools/server/tests/utils.parallel_function_calls}} -->
The `parallel_function_calls` function executes multiple functions concurrently and returns their results in the order they were called.
- **Inputs**:
    - `function_list`: A list of tuples, where each tuple contains a callable function and a tuple of arguments to be passed to that function.
- **Control Flow**:
    - Initialize a list `results` with `None` values, having the same length as `function_list`, to store the results of each function call.
    - Initialize an empty list `exceptions` to store any exceptions that occur during function execution.
    - Define a nested function `worker` that takes an index, a function, and its arguments, executes the function with the provided arguments, and stores the result in the `results` list at the given index. If an exception occurs, it appends the exception details to the `exceptions` list.
    - Create a `ThreadPoolExecutor` to manage concurrent execution of the functions.
    - Iterate over `function_list`, submitting each function and its arguments to the executor, and store the resulting futures in a list `futures`.
    - Use `as_completed` to wait for all futures to complete, ensuring all functions have finished executing.
    - Check if any exceptions were recorded; if so, print the details of each exception.
    - Return the `results` list containing the results of all function calls.
- **Output**: A list of results from the executed functions, in the same order as the input `function_list`. If any function raises an exception, the corresponding result will remain `None`, and the exception details will be printed.


---
### match\_regex<!-- {{#callable:llama.cpp/tools/server/tests/utils.match_regex}} -->
The `match_regex` function checks if a given regular expression pattern matches any part of a provided text string, ignoring case, and considering multiline and dotall modes.
- **Inputs**:
    - `regex`: A string representing the regular expression pattern to be matched against the text.
    - `text`: A string representing the text in which to search for the pattern.
- **Control Flow**:
    - The function compiles the provided regex pattern with flags for ignoring case, multiline, and dotall modes.
    - It then searches the compiled pattern within the provided text.
    - The function returns True if a match is found, otherwise it returns False.
- **Output**: A boolean value indicating whether the regex pattern matches any part of the text.


---
### download\_file<!-- {{#callable:llama.cpp/tools/server/tests/utils.download_file}} -->
The `download_file` function downloads a file from a given URL to a specified local path, avoiding re-download if the file already exists.
- **Inputs**:
    - `url`: A string representing the URL of the file to be downloaded.
    - `output_file_path`: An optional string representing the local path where the file should be saved; defaults to a path in the './tmp/' directory if not provided.
- **Control Flow**:
    - Extract the file name from the URL by splitting the URL string and taking the last segment.
    - Determine the output file path: use './tmp/' directory with the file name if `output_file_path` is not provided, otherwise use `output_file_path`.
    - Check if the file already exists at the determined output path using `os.path.exists`.
    - If the file does not exist, print a message indicating the start of the download, use `wget.download` to download the file to the output path, and print a completion message.
    - If the file already exists, print a message indicating that the file already exists at the output path.
- **Output**: The function returns a string representing the local path where the file is saved.


---
### is\_slow\_test\_allowed<!-- {{#callable:llama.cpp/tools/server/tests/utils.is_slow_test_allowed}} -->
The function checks if slow tests are allowed based on environment variables.
- **Inputs**: None
- **Control Flow**:
    - The function retrieves the value of the 'SLOW_TESTS' environment variable.
    - It checks if the value is either '1' or 'ON'.
    - The function returns True if either condition is met, otherwise it returns False.
- **Output**: A boolean value indicating whether slow tests are allowed based on the environment variable.


