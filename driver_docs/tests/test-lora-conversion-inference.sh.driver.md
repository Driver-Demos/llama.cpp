# Purpose
This Bash script is designed to automate the process of testing machine learning models by converting model files and running inference tests. It provides a narrow functionality focused on handling specific models for causal language modeling, as indicated by the model names in the `params` array. The script first checks for the existence of a specified repository, cloning it if necessary, and then iterates over a list of models to perform a series of operations. These operations include converting model files from one format to another, running inference tests using different configurations, and verifying the output against expected results. The script is not an executable in the traditional sense but rather a utility script intended to be run in a command-line environment to facilitate model testing and validation.
# Imports and Dependencies

---
- `git`
- `python`
- `python3`


# Global Variables

---
### params
- **Type**: `array`
- **Description**: The `params` variable is a global array that contains a list of model names and their corresponding hidden sizes. Each element in the array is a string that combines a model name with a hidden size, separated by a space. This array is used to iterate over different models and their configurations for testing purposes.
- **Use**: This variable is used to iterate over each model and its hidden size in a loop, passing them as arguments to the `run_conversion_and_inference_lora` function for processing.


---
### MODELS\_REPO
- **Type**: `string`
- **Description**: The `MODELS_REPO` variable is a string that specifies the name of the directory where the Hugging Face repository will be cloned. In this script, it is set to 'lora-tests', which is used as the local directory name for storing the cloned repository.
- **Use**: This variable is used to define the directory name for cloning the Hugging Face repository and is referenced in various parts of the script to access files and directories within the cloned repository.


---
### MODELS\_REPO\_URL
- **Type**: `string`
- **Description**: The `MODELS_REPO_URL` is a string variable that holds the URL to a specific repository on Hugging Face, constructed using the base URL 'https://huggingface.co/ggml-org/' and the repository name stored in `MODELS_REPO`. This URL is used to clone the repository if it does not already exist locally.
- **Use**: This variable is used to specify the location of the repository to be cloned for model testing and conversion operations.


---
### COMMIT
- **Type**: `string`
- **Description**: The `COMMIT` variable is a string that holds the specific commit hash `c26d5fb85b4070a9e9c4e65d132c783b98086890`. This hash uniquely identifies a particular state of the code in the `lora-tests` repository on Hugging Face.
- **Use**: This variable is used to fetch and reset the repository to a specific commit state, ensuring that the code operates on a known version of the repository.


---
### results
- **Type**: `array`
- **Description**: The `results` variable is a global array that is used to store the output results of running tests on different models. Each element in the array contains formatted strings that summarize the results of the tests for a specific model, including outputs for base, Lora hot, and Lora merged configurations.
- **Use**: This variable is used to accumulate and store the results of model tests, which are later printed as a summary.


---
### EXPECTED\_BASE\_FULL
- **Type**: `string`
- **Description**: `EXPECTED_BASE_FULL` is a global variable that stores the entire content of the file `pale_blue_dot.txt` located in the `data` directory of the cloned `lora-tests` repository. This file is expected to contain a reference text used for comparison in the script.
- **Use**: This variable is used to verify that the output of the model inference matches the expected starting substring of the reference text.


---
### EXPECTED\_LORA\_FULL
- **Type**: `string`
- **Description**: `EXPECTED_LORA_FULL` is a global variable that stores the full content of the 'bohemian_rhapsody.txt' file located in the 'data' directory of the cloned repository. This variable is used to hold the expected output string for comparison during inference tests.
- **Use**: It is used to verify that the output from the inference process matches the expected content of the 'bohemian_rhapsody.txt' file.


---
### EXPECTED\_BASE\_FIRST\_WORD
- **Type**: `string`
- **Description**: `EXPECTED_BASE_FIRST_WORD` is a global variable that stores the first word extracted from the contents of a file located at `$MODELS_REPO/data/pale_blue_dot.txt`. This file is expected to contain a text from which the first word is extracted using the `get_first_word` function.
- **Use**: This variable is used as a prompt input for the `llama-cli` command during inference tests to ensure the output starts with the expected string.


---
### EXPECTED\_LORA\_FIRST\_WORD
- **Type**: `string`
- **Description**: `EXPECTED_LORA_FIRST_WORD` is a global variable that stores the first word extracted from the contents of the file `bohemian_rhapsody.txt` located in the `data` directory of the cloned repository. This file is expected to contain the full text of the song 'Bohemian Rhapsody', and the variable holds the first word of this text.
- **Use**: This variable is used as a prompt input for running inference tests with the `llama-cli` tool to verify the output against expected results.


# Functions

---
### trim\_leading\_whitespace
The `trim_leading_whitespace` function removes leading whitespace from a given input string.
- **Inputs**:
    - `input_string`: A string from which leading whitespace is to be removed.
- **Control Flow**:
    - The function takes a single argument, `input_string`, which is expected to be a string.
    - It uses parameter expansion to remove leading whitespace from `input_string`.
    - The function echoes the modified string without leading whitespace.
- **Output**: A string with leading whitespace removed from the input string.


---
### extract\_starting\_substring
The `extract_starting_substring` function extracts a substring from the beginning of a reference string that matches the length of a target string.
- **Inputs**:
    - `reference_string`: The string from which the starting substring will be extracted.
    - `target_string`: The string whose length determines the length of the substring to extract from the reference string.
- **Control Flow**:
    - Calculate the length of the target string and store it in `target_length`.
    - Use substring expansion to extract the starting substring from `reference_string` with the length of `target_length`.
    - Output the extracted substring.
- **Output**: A substring from the beginning of `reference_string` that has the same length as `target_string`.


---
### get\_first\_word
The `get_first_word` function extracts and returns the first word from a given input string.
- **Inputs**:
    - `input_string`: A string from which the first word is to be extracted.
- **Control Flow**:
    - The function takes a single input argument, `input_string`.
    - It uses the `read` command with a here-string to split the input string into words, assigning the first word to the variable `first_word`.
    - The function then echoes the value of `first_word`, effectively returning it.
- **Output**: The first word of the input string.


---
### run\_conversion\_and\_inference\_lora
The `run_conversion_and_inference_lora` function performs model conversion and inference tests for a given model name and hidden size, verifying the outputs against expected results.
- **Inputs**:
    - `model_name`: The name of the model to be tested, such as 'Gemma2ForCausalLM'.
    - `hidden_size`: The hidden size parameter for the model, such as '64'.
- **Control Flow**:
    - Prints a header indicating the start of tests for the specified model and hidden size.
    - Runs a Python script `convert_hf_to_gguf.py` to convert safetensors to gguf format for the base model.
    - Runs a Python script `convert_lora_to_gguf.py` to convert LoRA model to gguf format.
    - Executes `llama-export-lora` to merge the base and LoRA models into a single gguf file.
    - Runs `llama-cli` to perform inference without LoRA, capturing the output.
    - Runs `llama-cli` to perform inference with hot LoRA, capturing the output.
    - Runs `llama-cli` to perform inference with merged LoRA, capturing the output.
    - Trims leading whitespace from the outputs of the inference runs.
    - Extracts the starting substring from the expected full strings to compare with the outputs.
    - Checks if the outputs match the expected starting substrings, exiting with an error if they do not match.
    - Stores the results of the tests in a results array.
    - Prints a success message if all tests pass for the given model and hidden size.
- **Output**: The function does not return a value but appends formatted test results to a global `results` array and prints success or error messages to the console.


