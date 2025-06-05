# Purpose
This Bash script is an executable file designed to facilitate the debugging and execution of CTest programs within a continuous integration (CI) environment. It provides a narrow functionality focused on setting up a build environment, compiling test binaries, and running or debugging specific tests based on user input. The script includes features for checking necessary dependencies, parsing command-line options, and handling positional arguments to filter and select tests using regular expressions. It also supports running tests in gdb mode for debugging purposes. The script is structured to ensure that it operates within a Git repository, resetting the folder context and managing build directories, and it provides user feedback through color-coded messages to indicate the status of operations and test results.
# Global Variables

---
### PROG
- **Type**: `string`
- **Description**: The `PROG` variable is a string that holds the name of the script being executed. It is derived from the special parameter `$0`, which contains the path of the script, and the `${0##*/}` expression extracts just the filename from this path.
- **Use**: This variable is used to display the script's name in help messages and error outputs, providing context to the user about which script is running.


---
### build\_dir
- **Type**: `string`
- **Description**: The `build_dir` variable is a global string variable that specifies the name of the directory where the build process will take place. It is initialized with the value "build-ci-debug".
- **Use**: This variable is used to define the directory path for building and compiling test binaries, ensuring that all build-related files are organized in a specific directory.


---
### red
- **Type**: `string`
- **Description**: The `red` variable is a string that holds the terminal escape sequence for setting the text color to red. It is defined using the `tput` command, which retrieves the escape sequence for the specified color (in this case, color code 1 for red) from the terminal's capabilities database.
- **Use**: This variable is used to change the text color to red in the terminal output, typically to indicate errors or failed tests.


---
### green
- **Type**: `string`
- **Description**: The `green` variable is a string that stores the terminal escape sequence for setting the text color to green using the `tput` command. This is achieved by calling `tput setaf 2`, where `2` corresponds to the color green in the terminal's color palette.
- **Use**: This variable is used to change the text color to green in the terminal output, typically to indicate a successful operation or a passing test.


---
### yellow
- **Type**: `string`
- **Description**: The `yellow` variable is a string that stores the terminal escape sequence for setting the text color to yellow using the `tput` command. This is achieved by calling `tput setaf 3`, where `3` is the ANSI color code for yellow.
- **Use**: This variable is used to change the text color to yellow in the terminal output for better visual distinction.


---
### blue
- **Type**: `string`
- **Description**: The `blue` variable is a string that stores the terminal escape sequence for setting the text color to blue using the `tput` command. This is achieved by calling `tput setaf 4`, where `4` is the color code for blue in the terminal's color palette.
- **Use**: This variable is used to change the text color to blue in the terminal output for better visual distinction of messages.


---
### magenta
- **Type**: `string`
- **Description**: The `magenta` variable is a string that stores the terminal escape sequence for setting the foreground text color to magenta using the `tput` command. This is achieved by calling `tput setaf 5`, where `5` corresponds to the magenta color code in the terminal's color palette.
- **Use**: This variable is used to format output text in the terminal with a magenta color for better visual distinction.


---
### cyan
- **Type**: `string`
- **Description**: The `cyan` variable is a string that stores the terminal escape sequence for setting the text color to cyan using the `tput` command. This is achieved by calling `tput setaf 6`, where `6` corresponds to the cyan color in the terminal's color palette.
- **Use**: This variable is used to format output text in cyan color in the terminal.


---
### normal
- **Type**: `string`
- **Description**: The `normal` variable is a string that stores the terminal escape sequence to reset text formatting to the default style. It is set using the `tput sgr0` command, which is a standard way to reset terminal text attributes such as color, boldness, and underline.
- **Use**: This variable is used to reset the terminal text formatting to default after printing colored or styled text.


# Functions

---
### print\_full\_help
The `print_full_help` function displays a help message detailing the usage, options, and arguments for a script that debugs specific ctest programs.
- **Inputs**: None
- **Control Flow**:
    - The function uses a here document (EOF) to print a multi-line help message to the standard output.
    - The help message includes usage instructions, options, arguments, and examples for running the script.
- **Output**: The function outputs a formatted help message to the standard output.


---
### abort
The `abort` function prints an error message and usage instructions to standard error and then exits the script with a status code of 1.
- **Inputs**:
    - `$1`: A string containing the error message to be displayed.
- **Control Flow**:
    - The function prints an error message prefixed with 'Error: ' followed by the content of the first argument to standard error.
    - It then prints a usage message to standard error, which includes the script's usage pattern and a reference to the help option.
    - Finally, the function exits the script with a status code of 1, indicating an error.
- **Output**: The function does not return a value; it terminates the script with an exit status of 1.


---
### check\_dependency
The `check_dependency` function verifies the presence of a specified command-line tool and aborts execution with an error message if the tool is not found.
- **Inputs**:
    - `$1`: The name of the command-line tool to check for, passed as a string.
- **Control Flow**:
    - The function uses the `command -v` command to check if the specified tool is available in the system's PATH.
    - If the tool is not found, the function calls the `abort` function with an error message indicating the missing tool and exits the script.
- **Output**: The function does not return any value; it either continues execution if the tool is found or aborts the script if the tool is missing.


