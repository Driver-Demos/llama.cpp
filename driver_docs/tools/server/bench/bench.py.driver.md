# Purpose
This Python script is designed to automate the process of benchmarking a server's performance, specifically in the context of running a machine learning model hosted on a server. The script is structured as a command-line tool, utilizing the `argparse` module to handle various input parameters that define the benchmarking scenario, such as server details, model specifications, and benchmarking configurations. The script's primary function is to start a server process, execute a benchmarking scenario using the `k6` tool, and then collect and process performance metrics. These metrics are gathered from both the server's output and Prometheus, a monitoring tool, and are subsequently formatted and saved for further analysis.

The script is comprehensive, incorporating several technical components to ensure robust functionality. It includes subprocess management to handle server and benchmarking tool execution, threading for concurrent log handling, and network operations to verify server readiness and health. Additionally, it uses data visualization libraries like `matplotlib` to generate plots of the collected metrics, which are saved as images and in a text-based format for redundancy. The script is intended to be executed as a standalone program, as indicated by the `if __name__ == '__main__':` block, and it outputs results in a format suitable for integration with continuous integration systems, such as GitHub Actions, by writing environment variables to a file.
# Imports and Dependencies

---
- `__future__.annotations`
- `argparse`
- `json`
- `os`
- `re`
- `signal`
- `socket`
- `subprocess`
- `sys`
- `threading`
- `time`
- `traceback`
- `contextlib.closing`
- `datetime.datetime`
- `matplotlib`
- `matplotlib.dates`
- `matplotlib.pyplot`
- `requests`
- `statistics.mean`


# Functions

---
### main<!-- {{#callable:llama.cpp/tools/server/bench/bench.main}} -->
The `main` function initializes and runs a server benchmark scenario, processes the results, and generates performance metrics and visualizations.
- **Inputs**:
    - `args_in`: A list of strings representing command-line arguments, or None to use default arguments.
- **Control Flow**:
    - Initialize an argument parser and define various command-line arguments required for the benchmark scenario.
    - Parse the input arguments using the argument parser.
    - Record the start time of the benchmark process.
    - Attempt to start the server using the parsed arguments and handle any exceptions by printing the error and exiting.
    - Initialize variables for iterations and data to store benchmark results.
    - Attempt to start the benchmark process and parse the results from a JSON file, rounding numerical values and writing them to an environment file.
    - Handle any exceptions during the benchmark process by printing the error.
    - If the server process is running, attempt to gracefully shut it down using appropriate signals, and force-kill if necessary.
    - Check if the server is still listening and wait until it stops.
    - Generate a title and xlabel for the benchmark results based on the input arguments.
    - Check if a Prometheus server is listening and, if so, query for specific metrics, save them as JSON, and generate plots and Mermaid diagrams for visualization.
    - Compile benchmark results into a dictionary and write them to an environment file.
- **Output**: The function does not return any value; it performs operations related to starting a server, running benchmarks, and generating output files and visualizations.
- **Functions called**:
    - [`llama.cpp/tools/server/bench/bench.start_server`](#cpp/tools/server/bench/benchstart_server)
    - [`llama.cpp/tools/server/bench/bench.start_benchmark`](#cpp/tools/server/bench/benchstart_benchmark)
    - [`llama.cpp/tools/server/bench/bench.escape_metric_name`](#cpp/tools/server/bench/benchescape_metric_name)
    - [`llama.cpp/tools/server/bench/bench.is_server_listening`](#cpp/tools/server/bench/benchis_server_listening)


---
### start\_benchmark<!-- {{#callable:llama.cpp/tools/server/bench/bench.start_benchmark}} -->
The `start_benchmark` function initiates a performance benchmark using the k6 tool with specified parameters and handles execution errors.
- **Inputs**:
    - `args`: An object containing various benchmark parameters such as scenario, duration, number of prompts, parallelism, and others required for configuring the k6 benchmark run.
- **Control Flow**:
    - Set the default path for the k6 binary to './k6'.
    - Check if 'BENCH_K6_BIN_PATH' is set in the environment variables and update the k6 path if it is.
    - Initialize a list `k6_args` with basic k6 run parameters including scenario, no-color, and connection reuse options.
    - Extend `k6_args` with additional parameters for duration, iterations, virtual users, summary export, and output format.
    - Construct a command string `args` with environment variables and the k6 command including all arguments.
    - Print the constructed command to the console for logging purposes.
    - Execute the k6 command using `subprocess.run` with shell execution, redirecting stdout and stderr to the system's standard output and error streams.
    - Check the return code of the subprocess; if non-zero, raise an exception indicating the k6 run failed.
- **Output**: The function does not return a value but raises an exception if the k6 benchmark execution fails.


---
### start\_server<!-- {{#callable:llama.cpp/tools/server/bench/bench.start_server}} -->
The `start_server` function initiates a server process and waits for it to be fully operational by checking its listening and readiness status.
- **Inputs**:
    - `args`: An object containing server configuration parameters such as host, port, and other server-related settings.
- **Control Flow**:
    - Call [`start_server_background`](#cpp/tools/server/bench/benchstart_server_background) with `args` to initiate the server process in the background.
    - Initialize `attempts` to 0 and set `max_attempts` to 600, doubling it if running in a GitHub Actions environment.
    - Enter a loop to check if the server is listening using [`is_server_listening`](#cpp/tools/server/bench/benchis_server_listening); increment `attempts` and assert failure if `max_attempts` is exceeded.
    - Print a waiting message and sleep for 0.5 seconds if the server is not yet listening.
    - Reset `attempts` to 0 and enter another loop to check if the server is ready using [`is_server_ready`](#cpp/tools/server/bench/benchis_server_ready); increment `attempts` and assert failure if `max_attempts` is exceeded.
    - Print a waiting message and sleep for 0.5 seconds if the server is not yet ready.
    - Print a success message indicating the server is started and ready.
- **Output**: Returns the `server_process` object representing the running server process.
- **Functions called**:
    - [`llama.cpp/tools/server/bench/bench.start_server_background`](#cpp/tools/server/bench/benchstart_server_background)
    - [`llama.cpp/tools/server/bench/bench.is_server_listening`](#cpp/tools/server/bench/benchis_server_listening)
    - [`llama.cpp/tools/server/bench/bench.is_server_ready`](#cpp/tools/server/bench/benchis_server_ready)


---
### start\_server\_background<!-- {{#callable:llama.cpp/tools/server/bench/bench.start_server_background}} -->
The `start_server_background` function initializes and starts a server process in the background with specified arguments and logs its output to the console.
- **Inputs**:
    - `args`: An object containing various server configuration parameters such as host, port, Hugging Face repository and file, GPU layers, context size, parallel slots, batch sizes, and maximum tokens.
- **Control Flow**:
    - Determine the server executable path, defaulting to '../../../build/bin/llama-server' unless overridden by the 'LLAMA_SERVER_BIN_PATH' environment variable.
    - Construct a list of server arguments using the provided `args` object, including host, port, Hugging Face repository and file, GPU layers, context size, parallel slots, batch sizes, and other server options.
    - Convert all arguments to strings and print the command used to start the server.
    - Start the server process using `subprocess.Popen` with the constructed arguments, capturing both stdout and stderr.
    - Define a helper function `server_log` to read and print lines from the server's output streams.
    - Create and start two threads to handle logging of the server's stdout and stderr to the console.
    - Return the `server_process` object representing the running server process.
- **Output**: The function returns a `subprocess.Popen` object representing the server process that was started.


---
### is\_server\_listening<!-- {{#callable:llama.cpp/tools/server/bench/bench.is_server_listening}} -->
The function `is_server_listening` checks if a server is listening on a specified host and port.
- **Inputs**:
    - `server_fqdn`: The fully qualified domain name (FQDN) of the server to check.
    - `server_port`: The port number on which the server is expected to be listening.
- **Control Flow**:
    - A socket is created using the `socket.AF_INET` and `socket.SOCK_STREAM` parameters to specify an IPv4 TCP connection.
    - The socket is used to attempt a connection to the specified server FQDN and port using `connect_ex`, which returns 0 if the connection is successful.
    - A boolean variable `_is_server_listening` is set to `True` if the connection result is 0, indicating the server is listening.
    - If the server is listening, a message is printed to the console indicating the server is listening on the specified FQDN and port.
    - The function returns the boolean value `_is_server_listening` indicating whether the server is listening.
- **Output**: A boolean value indicating whether the server is listening on the specified FQDN and port.


---
### is\_server\_ready<!-- {{#callable:llama.cpp/tools/server/bench/bench.is_server_ready}} -->
The `is_server_ready` function checks if a server is ready by sending an HTTP GET request to its health endpoint and returns True if the server responds with a status code of 200.
- **Inputs**:
    - `server_fqdn`: The fully qualified domain name of the server to check.
    - `server_port`: The port number on which the server is expected to be running.
- **Control Flow**:
    - Constructs a URL using the provided server FQDN and port, appending '/health' to form the health check endpoint.
    - Sends an HTTP GET request to the constructed URL using the `requests` library.
    - Checks the status code of the response from the server.
    - Returns True if the status code is 200, indicating the server is ready; otherwise, returns False.
- **Output**: A boolean value indicating whether the server is ready (True if the server responds with a status code of 200, otherwise False).


---
### escape\_metric\_name<!-- {{#callable:llama.cpp/tools/server/bench/bench.escape_metric_name}} -->
The `escape_metric_name` function converts a given metric name to uppercase and replaces any non-alphanumeric characters with underscores.
- **Inputs**:
    - `metric_name`: A string representing the metric name that needs to be escaped.
- **Control Flow**:
    - Convert the input `metric_name` to uppercase using `metric_name.upper()`.
    - Use `re.sub` to replace any character in the uppercase string that is not an uppercase letter or digit with an underscore ('_').
- **Output**: A string where all non-alphanumeric characters in the input metric name are replaced with underscores, and all letters are converted to uppercase.


