# Purpose
This Bash script is an executable file designed to automate the process of handling machine learning models, specifically for splitting, quantizing, and testing them. It provides a narrow functionality focused on managing model files, likely in a development or testing environment. The script takes a path to a build binary and an optional path to a temporary folder as arguments, setting up a workspace for processing. It performs several operations: downloading a model, splitting it into smaller parts, quantizing the split model with and without keeping the split, and testing the quantized models to ensure they load correctly. The script also includes cleanup steps to remove temporary files, ensuring that the workspace is reset for future runs.
# Global Variables

---
### TMP\_DIR
- **Type**: `string`
- **Description**: The `TMP_DIR` variable is a global string variable that specifies the directory path used for temporary files during the script's execution. It is set based on the second command-line argument provided to the script, or defaults to '/tmp' if no second argument is given.
- **Use**: This variable is used to define the base path for temporary working directories and files created during the script's operations.


---
### SPLIT
- **Type**: `string`
- **Description**: The `SPLIT` variable is a string that holds the path to the `llama-gguf-split` executable. It is constructed by appending `/llama-gguf-split` to the first command-line argument, which is expected to be the path to the build binary directory. This variable is used to specify the location of the `llama-gguf-split` tool, which is responsible for splitting model files as part of the script's operations.
- **Use**: This variable is used to execute the `llama-gguf-split` command for splitting model files during the script's execution.


---
### QUANTIZE
- **Type**: `string`
- **Description**: The `QUANTIZE` variable is a string that holds the path to the 'llama-quantize' binary within the specified build directory. It is used to reference the executable responsible for quantizing models in the script.
- **Use**: This variable is used to execute the 'llama-quantize' binary with specific options for model quantization.


---
### MAIN
- **Type**: `string`
- **Description**: The `MAIN` variable is a global string variable that stores the path to the `llama-cli` executable. It is constructed by appending `/llama-cli` to the path provided as the first argument to the script.
- **Use**: This variable is used to execute the `llama-cli` command with specific options for testing the loading of requanted models.


---
### WORK\_PATH
- **Type**: `string`
- **Description**: The `WORK_PATH` variable is a string that represents the directory path where temporary files related to the quantization process are stored. It is constructed by appending '/quantize' to the `TMP_DIR`, which is either provided as a command-line argument or defaults to '/tmp'. This path is used throughout the script to store intermediate and final files generated during the model processing steps.
- **Use**: This variable is used to define the working directory for storing temporary files during the model quantization and testing processes.


---
### ROOT\_DIR
- **Type**: `string`
- **Description**: The `ROOT_DIR` variable is a string that holds the absolute path to the root directory of the project. It is calculated using the `realpath` command, which resolves the absolute path of the directory two levels up from the script's location.
- **Use**: This variable is used to construct paths to scripts and resources within the project, ensuring that the script can access necessary files regardless of the current working directory.


