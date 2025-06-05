# Purpose
This Bash script is designed to automate the process of building and benchmarking various machine learning models using a tool called `tool_bench.py`. It provides a narrow functionality focused on setting up the environment, verifying necessary binaries and cache directories, and executing a series of benchmark tests on different models, such as Qwen, Llama, Mistral, Hermes, and others. The script first ensures that the `llama-server` binary is available and that the cache directory is set, then it constructs a set of arguments for the benchmarking tool, including various temperature settings for model evaluation. It runs the benchmarking tool for each specified model, outputting results in JSONL format, and subsequently generates plots from these results. This script is not an executable or a library but rather a utility script intended to be run in a command-line environment to facilitate model performance evaluation.
# Global Variables

---
### LLAMA\_CACHE
- **Type**: `string`
- **Description**: The `LLAMA_CACHE` variable is a global environment variable that specifies the directory path where the cache for the llama application is stored. It defaults to `$HOME/Library/Caches/llama.cpp` if not explicitly set by the user.
- **Use**: This variable is used to determine the location of the llama cache directory, ensuring that the application can access cached data during execution.


---
### LLAMA\_SERVER\_BIN\_PATH
- **Type**: `string`
- **Description**: `LLAMA_SERVER_BIN_PATH` is a global environment variable that stores the path to the llama-server binary executable. It is set to the path `$PWD/build/bin/llama-server`, which is the expected location of the compiled llama-server binary after the build process.
- **Use**: This variable is used to verify the existence and executability of the llama-server binary before proceeding with further script operations.


---
### ARGS
- **Type**: `array`
- **Description**: The `ARGS` variable is a global array that contains a list of command-line arguments used to configure the execution of the `tool_bench.py` script. It includes a baseline path to the `llama-server` executable, a fixed number of iterations (`--n 30`), and a series of temperature settings (`--temp`) that are used to adjust the behavior of the script. The array also appends any additional arguments passed to the script (`"$@"`).
- **Use**: This variable is used to pass a consistent set of parameters to multiple invocations of the `tool_bench.py` script, ensuring that each run is configured with the same baseline and temperature settings.


