# Purpose
The `check-requirements.sh` script is a Bash script designed to verify the dependencies for Python scripts, specifically those named with the pattern `convert*.py`. It provides a narrow functionality focused on setting up isolated Python virtual environments (venvs) for each script, installing the required packages, and checking for import errors to ensure that all dependencies are correctly specified. This script is intended to be executed in a development environment, as it is IO intensive and can generate a significant amount of temporary data, recommending the use of a tmpfs or ramdisk for frequent runs. It includes options for specifying a working directory and controlling cleanup behavior, making it a utility script rather than a library or configuration file. The script also enforces best practices by checking that exact package versions are not pinned in the requirements files, promoting the use of version specifiers like `~=` instead.
# Global Variables

---
### do\_cleanup
- **Type**: `integer`
- **Description**: The `do_cleanup` variable is a global integer variable that determines whether the script should perform cleanup operations after execution. It is set to 1 by default, indicating that cleanup should occur, but can be set to 0 if the 'nocleanup' argument is provided, disabling automatic cleanup of temporary files and directories created by the script.
- **Use**: This variable is used to control whether the cleanup function is triggered upon script exit, based on its value.


---
### this
- **Type**: `string`
- **Description**: The `this` variable is a global string variable that stores the absolute path of the current script being executed. It is initialized using the `realpath` command, which resolves the full path of the script file (`$0`).
- **Use**: This variable is used to ensure that operations such as directory changes and script checks are performed relative to the script's location.


---
### reqs\_dir
- **Type**: `string`
- **Description**: The `reqs_dir` variable is a global string variable that holds the name of the directory where the requirements files are stored. It is used to construct paths to specific requirements files for each Python script being checked.
- **Use**: This variable is used to locate and access the requirements files needed for setting up virtual environments and checking for import errors in Python scripts.


---
### tmp\_dir
- **Type**: `string`
- **Description**: The `tmp_dir` variable is a global variable that holds the path to a temporary directory used as the base for setting up virtual environments (venvs) during the execution of the script. It defaults to `/tmp` if no directory is provided as an argument to the script. The script checks if the specified directory is writable and exists, and if not, it terminates with an error.
- **Use**: This variable is used to determine the base directory for creating temporary working directories for virtual environments.


---
### workdir
- **Type**: `string`
- **Description**: The `workdir` variable is a string that holds the path to a temporary directory created using the `mktemp` command. This directory is used as a working space for setting up virtual environments (venvs) during the execution of the script.
- **Use**: This variable is used to store the path of the temporary directory where virtual environments are created and managed for checking Python script requirements.


---
### ignore\_eq\_eq
- **Type**: `string`
- **Description**: The `ignore_eq_eq` variable is a string that contains the value 'check_requirements: ignore "=="'. It is used as a marker or comment to indicate that certain lines in requirements files should not be checked for exact version pinning using the '==' operator.
- **Use**: This variable is used to filter out lines in requirements files that should not trigger a fatal error when exact version pinning is detected.


---
### all\_venv
- **Type**: `string`
- **Description**: The `all_venv` variable is a string that holds the path to a virtual environment directory created within the script. This directory is named 'all-venv' and is located within the `workdir` directory, which is a temporary working directory created for the script's execution.
- **Use**: This variable is used to store the path of the virtual environment where the requirements from `requirements.txt` are installed and checked.


# Functions

---
### log
The `log` function outputs a formatted log message to standard error with a specified log level and message.
- **Inputs**:
    - `level`: A string representing the log level (e.g., DEBUG, INFO, FATAL).
    - `msg`: A string containing the message to be logged.
- **Control Flow**:
    - The function takes two arguments: 'level' and 'msg'.
    - It uses the `printf` command to format and print the log message to standard error (file descriptor 2).
    - The message is prefixed with the log level and followed by the message content.
- **Output**: The function does not return any value; it outputs a formatted log message to standard error.


---
### debug
The `debug` function logs a debug-level message using the `log` function.
- **Inputs**:
    - `$@`: A variable number of arguments that represent the message to be logged.
- **Control Flow**:
    - The function calls the `log` function with 'DEBUG' as the log level and passes all received arguments to it.
- **Output**: The function does not return any value; it outputs a debug message to standard error.


---
### info
The `info` function logs informational messages to standard error with a specific format.
- **Inputs**:
    - `level`: The log level, which is a string indicating the severity or type of message, in this case, 'INFO'.
    - `msg`: The message to be logged, which is a string containing the information to be displayed.
- **Control Flow**:
    - The function calls the `log` function, passing 'INFO' as the log level and the provided message.
    - The `log` function formats the message and prints it to standard error.
- **Output**: The function does not return any value; it outputs the formatted log message to standard error.


---
### fatal
The `fatal` function logs a fatal error message and terminates the script execution.
- **Inputs**:
    - `$@`: A variable number of arguments representing the message to be logged as a fatal error.
- **Control Flow**:
    - The function calls the `log` function with 'FATAL' as the log level and the provided message.
    - The function then calls `exit 1` to terminate the script with a non-zero status, indicating an error.
- **Output**: The function does not return a value; it exits the script with a status code of 1.


---
### cleanup
The `cleanup` function removes a specified working directory if it exists and is writable, providing feedback on the removal process.
- **Inputs**:
    - `workdir`: A global variable representing the working directory to be removed, which must be set, exist, and be writable for the function to proceed.
- **Control Flow**:
    - Check if the global variable `workdir` is set, the directory exists, and is writable.
    - If the conditions are met, log the start of the removal process.
    - Use `rm -rfv` to remove the directory, printing a dot every 750 lines of output to indicate progress.
    - Log the completion of the removal process.
- **Output**: The function does not return a value but logs messages indicating the progress and completion of the directory removal process.


---
### check\_requirements
The `check_requirements` function installs the Python packages specified in a given requirements file using pip.
- **Inputs**:
    - `reqs`: A string representing the path to the requirements file that needs to be checked and installed.
- **Control Flow**:
    - Log the beginning of the check for the given requirements file.
    - Use pip to install the packages listed in the requirements file, suppressing version check messages.
    - Log the successful completion of the requirements check.
- **Output**: The function does not return any value; it logs messages to indicate the status of the requirements check.


---
### check\_convert\_script
The `check_convert_script` function verifies that a Python script can be imported without errors by setting up a virtual environment, installing its requirements, and attempting to import the script.
- **Inputs**:
    - `py`: The path to the Python script to be checked, e.g., './convert_hf_to_gguf.py'.
- **Control Flow**:
    - Extract the script name from the provided path and remove the '.py' extension to get the base name.
    - Log the start of the check for the given script.
    - Construct the path to the requirements file specific to the script using the base name.
    - Check if the requirements file exists and is readable; if not, log a fatal error and exit.
    - Verify that the script's requirements file is listed in the top-level 'requirements.txt'; if not, log a fatal error and exit.
    - Create a new virtual environment in the working directory specific to the script.
    - Activate the virtual environment and install the requirements from the script's requirements file.
    - Attempt to import the script using Python's import machinery to check for import errors.
    - If cleanup is enabled, remove the virtual environment after the check.
    - Log that the script imports successfully if no errors occur.
- **Output**: The function does not return a value but logs messages indicating the success or failure of the import check for the specified Python script.


