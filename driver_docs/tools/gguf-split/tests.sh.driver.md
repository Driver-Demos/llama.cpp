# Purpose
This Bash script is designed to automate the process of handling machine learning model files, specifically for splitting and merging GGUF (Generic Graphical User Format) model files. It provides a narrow functionality focused on testing and managing model file operations, such as splitting a model into smaller parts based on tensor count or file size, and then merging them back. The script is not an executable or a library but rather a utility script intended to be run from the command line, requiring paths to a build binary and optionally a temporary folder as arguments. It includes steps to download a model, perform various split and merge operations, and verify the integrity of these operations by loading the resulting model files. The script also ensures a clean working environment by removing any temporary files created during its execution.
# Global Variables

---
### TMP\_DIR
- **Type**: `string`
- **Description**: The `TMP_DIR` variable is a global string variable that specifies the directory path used for temporary files during the script's execution. It is set based on the second command-line argument provided to the script, or defaults to '/tmp' if no second argument is given.
- **Use**: This variable is used to define the base path for temporary working directories and files created and manipulated by the script.


---
### SPLIT
- **Type**: `string`
- **Description**: The `SPLIT` variable is a string that holds the path to the `llama-gguf-split` executable, which is derived from the first command-line argument provided to the script. This path is used to execute the `llama-gguf-split` tool for splitting and merging model files in the script.
- **Use**: This variable is used to construct and execute commands for splitting and merging model files using the `llama-gguf-split` tool.


---
### MAIN
- **Type**: `string`
- **Description**: The `MAIN` variable is a global string variable that stores the path to the `llama-cli` executable. This path is constructed using the first command-line argument provided to the script, which is expected to be the path to the build binary directory. The `llama-cli` executable is used for testing the loading of sharded and merged models.
- **Use**: `MAIN` is used to execute the `llama-cli` command with specific options to test the loading of models at various stages of the script.


---
### WORK\_PATH
- **Type**: `string`
- **Description**: The `WORK_PATH` variable is a string that represents the path to a temporary working directory used for storing intermediate files during the execution of the script. It is constructed by appending '/gguf-split' to the `TMP_DIR`, which is either provided as a command-line argument or defaults to '/tmp'. This directory is used to store split and merged model files during the script's operations.
- **Use**: `WORK_PATH` is used to define the directory where temporary files related to model splitting and merging are stored and managed.


---
### ROOT\_DIR
- **Type**: `string`
- **Description**: The `ROOT_DIR` variable is a string that holds the absolute path to the root directory of the project. It is calculated using the `realpath` command, which resolves the absolute path of the directory two levels up from the script's location.
- **Use**: This variable is used to construct paths to other scripts or resources within the project, ensuring that the script can access necessary files regardless of the current working directory.


