# Purpose
This Bash script is an executable file designed to automate the process of evaluating the perplexity of language models using different quantization levels. It provides a narrow functionality focused on running a specific command-line tool (`llama-perplexity`) with various configurations. The script accepts a model name as a mandatory argument and optionally allows the user to specify quantization levels and additional arguments. It sets up a directory for storing results and iterates over the specified quantization levels, executing the perplexity evaluation for each and saving the output to a designated results directory. This script is useful for users who need to perform batch processing of model evaluations with varying parameters, streamlining the process and ensuring consistent output management.
# Global Variables

---
### qnt
- **Type**: `array`
- **Description**: The variable `qnt` is an array that holds a list of quantization levels or types, such as 'f16', 'q8_0', 'q6_k', etc. These values represent different quantization schemes that can be applied to a model during processing.
- **Use**: This variable is used to iterate over different quantization types when executing a command in a loop, allowing the script to process each quantization type separately.


---
### args
- **Type**: `string`
- **Description**: The `args` variable is a global string variable that holds default command-line arguments for a script. It is initialized with the value "-ngl 999 -t 8", which are likely options for a command executed later in the script.
- **Use**: This variable is used to pass default or user-specified command-line arguments to the `llama-perplexity` command within the script.


