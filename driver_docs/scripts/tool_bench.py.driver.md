# Purpose
The provided Python script is a benchmarking tool designed to evaluate the performance of different server configurations, specifically focusing on llama-server and ollama. It is structured as a command-line application using the Typer library, which facilitates the creation of user-friendly command-line interfaces. The script's primary functionality is to execute a series of predefined tests multiple times across various server backends and temperature settings, then aggregate and visualize the results as a success rate heatmap. This visualization helps in comparing the performance of different models and server configurations under varying conditions.

The script is organized into two main commands: [`run`](#cpp/scripts/tool_benchrun) and [`plot`](#cpp/scripts/tool_benchplot). The [`run`](#cpp/scripts/tool_benchrun) command executes the tests, collecting data on success ratios, execution times, and other metrics, and outputs this data to a JSON file. It supports various options to customize the test parameters, such as the number of iterations, temperature settings, and model configurations. The [`plot`](#cpp/scripts/tool_benchplot) command reads the JSON output files, processes the data, and generates a heatmap using the Seaborn and Matplotlib libraries to visually represent the success ratios of the tests. The script also includes logging for error handling and debugging, ensuring that users are informed of any issues during execution. Overall, this script serves as a comprehensive tool for performance benchmarking and analysis of server-based machine learning models.
# Imports and Dependencies

---
- `contextlib.contextmanager`
- `pathlib.Path`
- `re`
- `statistics.mean`
- `statistics.median`
- `typing.Annotated`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `typing.Tuple`
- `atexit`
- `json`
- `logging`
- `matplotlib.pyplot`
- `numpy`
- `pandas`
- `seaborn`
- `subprocess`
- `sys`
- `time`
- `typer`
- `tools.server.tests.utils.ServerProcess`
- `tools.server.tests.unit.test_tool_call.TIMEOUT_SERVER_START`
- `tools.server.tests.unit.test_tool_call.do_test_calc_result`
- `tools.server.tests.unit.test_tool_call.do_test_hello_world`
- `tools.server.tests.unit.test_tool_call.do_test_weather`


# Global Variables

---
### logger
- **Type**: `logging.Logger`
- **Description**: The `logger` variable is an instance of a `Logger` object obtained from the Python `logging` module. It is configured to log messages with a specific format and at the INFO level, which includes timestamps, log levels, and messages.
- **Use**: This variable is used throughout the script to log informational messages, warnings, and errors, providing insights into the script's execution and any issues encountered.


---
### app
- **Type**: `typer.Typer`
- **Description**: The `app` variable is an instance of the `typer.Typer` class, which is part of the Typer library used for building command-line interface (CLI) applications in Python. This instance is used to define and manage the CLI commands and options for the script.
- **Use**: The `app` variable is used to register and manage CLI commands such as `plot` and `run`, allowing the script to be executed with different command-line arguments and options.


# Functions

---
### scoped\_server<!-- {{#callable:llama.cpp/scripts/tool_bench.scoped_server}} -->
The `scoped_server` function manages the lifecycle of a `ServerProcess` instance, ensuring it is stopped when the context is exited or the program terminates.
- **Decorators**: `@contextmanager`
- **Inputs**:
    - `sp`: An instance of `ServerProcess` that represents the server process to be managed.
- **Control Flow**:
    - Defines an inner function [`stop`](../tools/server/tests/utils.py.driver.md#ServerProcessstop) that stops the server process if it is not `None` and sets `sp` to `None`.
    - Registers the [`stop`](../tools/server/tests/utils.py.driver.md#ServerProcessstop) function to be called upon program exit using `atexit.register`.
    - Yields the `ServerProcess` instance `sp` to the context block.
    - Calls the [`stop`](../tools/server/tests/utils.py.driver.md#ServerProcessstop) function after the context block is exited to ensure the server is stopped.
- **Output**: The function yields the `ServerProcess` instance `sp` to the context block, allowing the caller to interact with the server process within the context.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.stop`](../tools/server/tests/utils.py.driver.md#ServerProcessstop)


---
### plot<!-- {{#callable:llama.cpp/scripts/tool_bench.plot}} -->
The `plot` function processes JSONL files to extract and filter data, then visualizes the success ratios of tool call benchmarks as a heatmap, optionally saving the plot to a file.
- **Decorators**: `@app.command`
- **Inputs**:
    - `files`: A list of Path objects representing the JSONL files to be processed.
    - `output`: An optional Path object specifying the file path to save the plot image; if not provided, the plot is displayed.
    - `test_regex`: An optional string for filtering records by matching the 'test' field using a regular expression.
    - `server_regex`: An optional string for filtering records by matching the 'server_name' field using a regular expression.
- **Control Flow**:
    - Initialize an empty list `lines` to store valid JSON records.
    - Iterate over each file in `files`, checking if the file exists; log an error and continue if not.
    - Read the file content, log the file size, and split it into lines.
    - For each line, strip whitespace and attempt to parse it as JSON; log a warning if parsing fails.
    - If no valid data is loaded, raise an exception.
    - Initialize several collections to store unique values and a dictionary `data_dict` to map data points to success ratios.
    - Iterate over each record in `lines`, extracting fields and calculating total counts; filter records using `test_regex` and `server_regex` if provided.
    - Populate `data_dict` with success ratios and update collections with unique values.
    - Log a warning if total counts are inconsistent across records.
    - Sort the collected temperatures, tests, and server names.
    - Log the number of processed lines, valid data points, and unique values for models, temperatures, tests, and servers.
    - Create a matrix and index for the heatmap, iterating over models and temperatures to populate rows with success ratios from `data_dict`.
    - Create a DataFrame from the matrix and index, and plot a heatmap using seaborn.
    - Set plot titles, labels, and adjust layout; save the plot to `output` if specified, otherwise display it.
- **Output**: A heatmap plot visualizing success ratios, either displayed or saved to a specified file.


---
### run<!-- {{#callable:llama.cpp/scripts/tool_bench.run}} -->
The `run` function executes a series of tests on specified models using different server configurations and writes the results to a JSON file.
- **Decorators**: `@app.command`
- **Inputs**:
    - `output`: A Path object representing the output JSON file where results will be stored.
    - `model`: An optional string specifying the name of the model to test, server agnostic.
    - `hf`: An optional string for the GGUF huggingface model repo id to test with llama-server.
    - `chat_template`: An optional string to override the chat template for llama-server.
    - `chat_template_file`: An optional string to override the chat template file for llama-server.
    - `ollama`: An optional string for the Ollama model tag to test.
    - `llama_baseline`: An optional string for the llama-server baseline binary path to use as baseline.
    - `n`: An integer specifying the number of times to run each test, default is 10.
    - `temp`: An optional list of floats representing the set of temperatures to test.
    - `top_p`: An optional float for the top_p parameter.
    - `top_k`: An optional integer for the top_k parameter.
    - `ctk`: An optional string for the ctk parameter.
    - `ctv`: An optional string for the ctv parameter.
    - `fa`: An optional boolean for the fa parameter.
    - `seed`: An optional integer for the random seed.
    - `port`: An integer specifying the llama-server port, default is 8084.
    - `force`: A boolean indicating whether to force overwrite of the output file.
    - `append`: A boolean indicating whether to append to the output file.
    - `test_hello_world`: A boolean indicating whether to run the hello world test, default is True.
    - `test_weather`: A boolean indicating whether to run the weather test, default is True.
    - `test_calc_result`: A boolean indicating whether to run the calc result test, default is False.
- **Control Flow**:
    - Initialize n_predict and n_ctx variables for prediction and context size.
    - Determine the model name based on provided hf or ollama if model is not specified.
    - Assert that the output file does not exist unless force or append is specified.
    - Open the output file in append or write mode based on the append flag.
    - Define an inner function `run` to execute tests on a server process with specified parameters.
    - Set up request_kwargs with temperature, top_p, top_k, and seed if provided.
    - Define tests to run based on test_hello_world, test_weather, and test_calc_result flags.
    - Iterate over each test, executing it n times, and log success or failure with timing.
    - Write test results to the output file in JSON format.
    - Iterate over temperatures and server configurations, setting up and starting server processes.
    - Run tests on each server configuration using the inner `run` function.
- **Output**: The function writes the results of the tests, including success ratios and timing statistics, to the specified output JSON file.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/unit/test_tool_call.do_test_hello_world`](../tools/server/tests/unit/test_tool_call.py.driver.md#cpp/tools/server/tests/unit/test_tool_calldo_test_hello_world)
    - [`llama.cpp/tools/server/tests/unit/test_tool_call.do_test_weather`](../tools/server/tests/unit/test_tool_call.py.driver.md#cpp/tools/server/tests/unit/test_tool_calldo_test_weather)
    - [`llama.cpp/tools/server/tests/unit/test_tool_call.do_test_calc_result`](../tools/server/tests/unit/test_tool_call.py.driver.md#cpp/tools/server/tests/unit/test_tool_calldo_test_calc_result)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess`](../tools/server/tests/utils.py.driver.md#cpp/tools/server/tests/utilsServerProcess)
    - [`llama.cpp/scripts/tool_bench.scoped_server`](#cpp/scripts/tool_benchscoped_server)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../tools/server/tests/utils.py.driver.md#ServerProcessstart)


