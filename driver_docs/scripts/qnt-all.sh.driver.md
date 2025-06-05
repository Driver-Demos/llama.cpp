# Purpose
This Bash script is designed to automate the quantization of machine learning models, providing a narrow but specific functionality. It is an executable script that takes a model name as a required argument and optionally accepts a list of quantization types and additional arguments. The script sets up a directory for output results and iterates over the specified quantization types, executing a quantization command for each type. The results of these operations are logged into separate text files within the output directory. The script ensures robust execution by using options like `set -o pipefail` and `set -e` to handle errors effectively.
# Global Variables

---
### qnt
- **Type**: `array`
- **Description**: The `qnt` variable is a global array that holds a list of quantization levels or identifiers, such as 'q8_0', 'q6_k', 'q5_k', etc. These identifiers are likely used to specify different quantization configurations for a model processing task.
- **Use**: This variable is used to iterate over each quantization level in a loop, applying a quantization process to a model file and saving the results.


---
### args
- **Type**: `string`
- **Description**: The `args` variable is a global string variable that is initialized as an empty string. It is intended to store additional command-line arguments that may be passed to the script.
- **Use**: This variable is used to append any extra arguments provided by the user to the command executed within the loop, allowing for flexible customization of the script's behavior.


---
### model
- **Type**: `string`
- **Description**: The `model` variable is a global string variable that stores the first command-line argument passed to the script. It represents the name or identifier of a model that the script will process.
- **Use**: This variable is used to construct file paths and directory names for storing results and accessing model files during the script's execution.


---
### out
- **Type**: `string`
- **Description**: The `out` variable is a string that represents the path to a directory where the results of the script's operations will be stored. It is constructed using a relative path that includes a 'tmp' directory and a subdirectory named 'results-' followed by the value of the `model` variable, which is the first argument passed to the script.
- **Use**: This variable is used to specify the output directory for storing the quantization results of different model configurations.


