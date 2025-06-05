# Purpose
This Bash script is designed to download and process a specified number of essays from a given RSS feed, specifically from Paul Graham's essays. It provides a narrow functionality focused on fetching and formatting text data from web pages. The script checks for the availability of necessary command-line tools such as `curl`, `html2text`, `tail`, and `sed`, ensuring they are installed before proceeding. It takes a single argument, `n`, which determines how many essays to download, and outputs the processed text into a file named `pg.txt`, with additional individual and cumulative files for each essay. The script includes a usage function to guide users on how to execute it correctly and provides a token count reference for different values of `n`.
# Functions

---
### usage
The `usage` function displays instructions on how to use the script, including the expected input and the resulting number of tokens for different input values.
- **Inputs**: None
- **Control Flow**:
    - The function prints a usage message indicating how to run the script with the required argument 'n'.
    - It provides a table showing the number of tokens in the resulting 'pg.txt' file for various values of 'n'.
    - The function then exits with a status code of 1.
- **Output**: The function does not return a value but prints usage instructions to the console and exits the script with a status code of 1.


---
### has\_cmd
The `has_cmd` function checks if a specified command is available and executable on the system, exiting with an error message if it is not.
- **Inputs**:
    - `$1`: The name of the command to check for availability and executability.
- **Control Flow**:
    - The function uses the `command -v` to check if the command specified by `$1` is available in the system's PATH and is executable.
    - If the command is not found or is not executable, the function outputs an error message to standard error indicating the command is not available.
    - The function then exits with a status code of 1 to indicate failure.
- **Output**: The function does not return a value but exits the script with an error message if the command is not available.


