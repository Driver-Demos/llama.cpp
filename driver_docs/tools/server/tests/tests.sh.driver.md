# Purpose
This Bash script is designed to facilitate the execution of test suites using the `pytest` framework, with a particular focus on handling both regular and slow tests. It is a utility script that ensures the execution environment is correctly set up by navigating to the script's directory and setting strict error handling with `set -eu`. The script checks for an environment variable `SLOW_TESTS` to determine whether to run slow tests, and if so, it pre-fetches necessary models using a Python script to prevent timeouts. The script then runs `pytest` with appropriate flags based on the presence of command-line arguments and the `SLOW_TESTS` variable. This script provides narrow functionality, specifically tailored for managing test execution conditions in a development or continuous integration environment.
# Global Variables

---
### SCRIPT\_DIR
- **Type**: `string`
- **Description**: `SCRIPT_DIR` is a string variable that stores the absolute path of the directory where the script is located. It is determined by using the `dirname` command on the script's source path and converting it to an absolute path with `pwd`. This ensures that any subsequent commands or scripts executed from this script are run in the correct directory context.
- **Use**: This variable is used to change the current working directory to the script's directory, ensuring that all relative paths in the script are resolved correctly.


