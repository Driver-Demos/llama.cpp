# Purpose
This code is a performance testing script written for the k6 load testing tool, designed to evaluate the efficiency and behavior of a server handling chat completions. The script imports several k6 modules, including those for server-sent events (SSE), metrics, and execution control, indicating its purpose is to simulate and measure server performance under load. The script is configured to interact with a server URL, which can be specified via environment variables, and it uses a dataset of conversation prompts to generate requests to the server. The script is structured to handle a specific number of prompts, model configurations, and token limits, all of which can be customized through environment variables.

The script defines a setup function to log the benchmark configuration and uses a `SharedArray` to load and filter conversation data from a JSON file. This data is processed to ensure it meets certain criteria, such as having a minimum number of conversation turns and token counts, before being used in the test. The script also sets up various metrics, such as trends and counters, to track the number of tokens processed and the rate of completions, providing detailed insights into the server's performance.

The main function of the script executes a load test by sending POST requests to the server with conversation prompts and model configurations. It uses server-sent events to handle responses and updates metrics based on the server's performance, such as the time taken to emit the first token and the rate of token processing. The script includes checks to ensure successful completions and uses thresholds to determine when to abort the test if performance criteria are not met. This setup allows for comprehensive performance analysis of the server's ability to handle chat completions under simulated user load.
# Imports and Dependencies

---
- `k6/x/sse`
- `k6`
- `k6/data`
- `k6/metrics`
- `k6/execution`


# Global Variables

---
### server\_url
- **Type**: `string`
- **Description**: The `server_url` variable is a global string that holds the URL of the server endpoint used for chat completions. It is initialized with the value of the environment variable `SERVER_BENCH_URL` if it exists, otherwise it defaults to 'http://localhost:8080/v1'. This allows the script to dynamically adjust the server endpoint based on the environment configuration.
- **Use**: This variable is used to construct the URL for making HTTP requests to the server for chat completions.


---
### n\_prompt
- **Type**: `number`
- **Description**: The `n_prompt` variable is a global constant that determines the number of prompts to be used from the dataset for benchmarking purposes. It is calculated based on the environment variable `SERVER_BENCH_N_PROMPTS`, or defaults to a calculated value of 480 if the environment variable is not set.
- **Use**: This variable is used to limit the number of prompts processed from the dataset during the benchmarking setup.


---
### model
- **Type**: `string`
- **Description**: The `model` variable is a global string variable that holds the name of the model to be requested during the server chat completions. It is set based on the environment variable `SERVER_BENCH_MODEL_ALIAS`, defaulting to 'my-model' if the environment variable is not provided.
- **Use**: This variable is used to specify which model should be used in the payload for server chat completions.


---
### dataset\_path
- **Type**: `string`
- **Description**: The `dataset_path` variable is a string that specifies the file path to the dataset used in the script. It defaults to './ShareGPT_V3_unfiltered_cleaned_split.json' if the environment variable `SERVER_BENCH_DATASET` is not set.
- **Use**: This variable is used to load and parse the dataset file for processing and filtering conversations in the script.


---
### max\_tokens
- **Type**: `number`
- **Description**: The `max_tokens` variable is a global constant that defines the maximum number of tokens that can be predicted in a single request. It is set to a default value of 512, but can be overridden by the environment variable `SERVER_BENCH_MAX_TOKENS`. This variable is crucial for controlling the length of the generated text in the AI model's response.
- **Use**: This variable is used to limit the number of tokens generated in the AI model's response during a chat completion request.


---
### n\_prompt\_tokens
- **Type**: `number`
- **Description**: The `n_prompt_tokens` variable is a global constant that defines the maximum number of tokens allowed in a prompt. It is set to a default value of 1024 tokens, but can be overridden by the environment variable `SERVER_BENCH_MAX_PROMPT_TOKENS`. This variable is used to filter out conversations that exceed the specified token limit, ensuring that only prompts within the allowed token range are processed.
- **Use**: This variable is used to filter and limit the number of tokens in a prompt during data processing.


---
### n\_ctx\_slot
- **Type**: `number`
- **Description**: The `n_ctx_slot` variable is a global constant that defines the maximum number of tokens allowed in a context slot for a given operation. It is set to a default value of 2048 tokens unless overridden by the environment variable `SERVER_BENCH_MAX_CONTEXT`. This variable is crucial for ensuring that the total number of tokens in a prompt and its completion does not exceed the specified limit, thereby maintaining the efficiency and performance of the system.
- **Use**: `n_ctx_slot` is used to filter out sequences that exceed the maximum allowed token count in the context slot during data processing.


---
### data
- **Type**: `SharedArray`
- **Description**: The `data` variable is a global instance of `SharedArray` that holds a filtered and processed list of conversation data extracted from a JSON dataset. It is initialized with a name 'conversations' and a function that loads and processes the dataset from the specified `dataset_path`. The processing involves filtering conversations to ensure they have at least two turns, start with a human message, and meet certain token length criteria.
- **Use**: This variable is used to provide a shared dataset of conversation prompts and their token counts for use in performance testing scenarios.


---
### llamacpp\_prompt\_tokens
- **Type**: `Trend`
- **Description**: The `llamacpp_prompt_tokens` variable is an instance of the `Trend` class from the `k6/metrics` module. It is used to track and record the number of prompt tokens processed during the execution of the load test. This metric helps in analyzing the performance and efficiency of the prompt processing in the system being tested.
- **Use**: This variable is used to add and record the number of prompt tokens processed in each test iteration, allowing for performance analysis over time.


---
### llamacpp\_completion\_tokens
- **Type**: `Trend`
- **Description**: The `llamacpp_completion_tokens` variable is a global instance of the `Trend` class from the `k6/metrics` module. It is used to track and record the number of completion tokens generated during the execution of a chat completion request in a performance test. This variable helps in analyzing the performance and efficiency of the model in generating completion tokens.
- **Use**: This variable is used to add and record the number of completion tokens generated during each chat completion event in the performance test.


---
### llamacpp\_tokens\_second
- **Type**: `Trend`
- **Description**: The `llamacpp_tokens_second` variable is a global instance of the `Trend` class from the `k6/metrics` module. It is used to track and record the rate of completion tokens processed per second during the execution of a performance test.
- **Use**: This variable is used to measure and analyze the performance of token processing in terms of tokens per second, providing insights into the efficiency of the system being tested.


---
### llamacpp\_prompt\_processing\_second
- **Type**: `Trend`
- **Description**: The `llamacpp_prompt_processing_second` is a global variable of type `Trend` used to track the rate of prompt processing in terms of tokens per second. It is part of the performance metrics collected during the execution of a load test for a chat completion server.
- **Use**: This variable is used to record and analyze the number of prompt tokens processed per second during the test execution.


---
### llamacpp\_emit\_first\_token\_second
- **Type**: `Trend`
- **Description**: The `llamacpp_emit_first_token_second` is a global variable of type `Trend` used to track the time taken to emit the first token in a chat completion process. It records the duration from the start of the request to the point when the first token is emitted, measured in seconds.
- **Use**: This variable is used to monitor and analyze the performance of the system in terms of latency for emitting the first token during chat completions.


---
### llamacpp\_prompt\_tokens\_total\_counter
- **Type**: `Counter`
- **Description**: The `llamacpp_prompt_tokens_total_counter` is a global variable of type `Counter` that tracks the cumulative number of prompt tokens processed during the execution of the script. It is used to aggregate the total count of prompt tokens across all iterations of the test.
- **Use**: This variable is used to accumulate the total number of prompt tokens processed, which helps in analyzing the overall token usage during the test.


---
### llamacpp\_completion\_tokens\_total\_counter
- **Type**: `Counter`
- **Description**: The `llamacpp_completion_tokens_total_counter` is a global variable defined as a `Counter` metric in the k6 performance testing script. It is used to keep track of the total number of completion tokens processed during the execution of the test. This counter helps in monitoring and analyzing the performance of the AI model by providing insights into the volume of completion tokens generated over time.
- **Use**: This variable is used to accumulate the total count of completion tokens processed during the test execution.


---
### llamacpp\_completions\_truncated\_rate
- **Type**: `Rate`
- **Description**: The `llamacpp_completions_truncated_rate` is a global variable of type `Rate` used to track the proportion of AI model completions that are truncated due to reaching a maximum length limit. This metric helps in understanding how often the model's output is cut short, which can be critical for performance evaluation and optimization.
- **Use**: This variable is used to record and evaluate the rate at which AI model completions are truncated, influencing test thresholds and potentially aborting tests if the rate exceeds a specified limit.


---
### llamacpp\_completions\_stop\_rate
- **Type**: `Rate`
- **Description**: The `llamacpp_completions_stop_rate` is a global variable of type `Rate` used to track the rate at which chat completions are stopped due to a specific stop condition. This variable is part of the performance metrics collected during the execution of a load test using the k6 tool.
- **Use**: This variable is used to record and analyze the frequency of chat completions that end with a 'stop' finish reason during the test.


---
### options
- **Type**: `object`
- **Description**: The `options` variable is a global configuration object for the k6 load testing script. It defines the thresholds for performance metrics, such as the rate of truncated completions, and sets the duration and number of virtual users (VUs) for the test. This configuration helps control the execution parameters and performance criteria for the load test.
- **Use**: This variable is used to configure the load test parameters and performance thresholds in the k6 script.


# Functions

---
### setup
The `setup` function logs the configuration details for a benchmark test, including server URL, number of prompts, model name, dataset path, and maximum tokens.
- **Inputs**: None
- **Control Flow**:
    - The function uses `console.info` to log a formatted string containing the benchmark configuration details.
    - The logged details include `server_url`, `n_prompt`, `model`, `dataset_path`, and `max_tokens`.
- **Output**: The function does not return any value; it only logs information to the console.


