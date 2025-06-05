# Purpose
This Bash script, `compare-commits.sh`, is designed to automate the process of comparing the performance of two different Git commits using a benchmarking tool called `llama-bench`. It provides a narrow functionality focused on performance comparison by executing a series of commands to build and run benchmarks for two specified commits. The script checks for necessary dependencies, sets up build configurations (including optional CUDA support), and manages the build process in a temporary directory. It then runs the benchmark for each commit, storing results in an SQLite database, and finally uses another script, `compare-llama-bench.py`, to analyze and compare the results. This script is intended to be executed directly and is part of a larger suite of tools for performance analysis in a development environment.
# Global Variables

---
### bench\_args
- **Type**: `string`
- **Description**: The `bench_args` variable is a string that captures all command-line arguments passed to the script starting from the third argument onward. It is used to pass additional arguments to the `llama-bench` command within the script.
- **Use**: This variable is used to append any additional arguments provided by the user to the `llama-bench` command during execution.


---
### dir
- **Type**: `string`
- **Description**: The `dir` variable is a global string variable that specifies the directory name where the build process will take place. It is set to the value 'build-bench', which is used as the directory for building the project using CMake.
- **Use**: This variable is used to define the directory path for CMake to build the project and store the build artifacts.


# Functions

---
### run
The `run` function builds and executes a benchmarking tool, storing results in a SQLite database.
- **Inputs**:
    - `None`: The function does not take any direct input arguments; it uses global variables and environment settings.
- **Control Flow**:
    - Remove the existing build directory `build-bench` if it exists.
    - Run CMake to configure the build system in the `build-bench` directory with any specified options.
    - Build the `llama-bench` target using the configured build system.
    - Execute the `llama-bench` binary with specified arguments, outputting results in SQL format and storing them in `llama-bench.sqlite`.
- **Output**: The function does not return a value; it performs actions that result in the creation of a SQLite database file containing benchmark results.


