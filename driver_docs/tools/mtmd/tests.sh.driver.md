# Purpose
This Bash script is designed to automate the testing of various machine learning models, specifically vision and audio models, by executing a series of tests and logging the results. It provides a narrow functionality focused on setting up the environment, determining which models to test based on input arguments, and running tests using a specific binary (`llama-mtmd-cli`). The script is not an executable in the traditional sense but rather a utility script that orchestrates the testing process by preparing directories, handling model selection, and executing tests with different configurations. It includes conditional logic to handle "big" and "huge" model tests, allowing for scalability based on the machine's capabilities. The results of each test are logged, and the script provides a summary of the outcomes, indicating success or failure for each model tested.
# Global Variables

---
### SCRIPT\_DIR
- **Type**: `string`
- **Description**: `SCRIPT_DIR` is a string variable that stores the absolute path of the directory where the script is located. It is determined by using the `dirname` command on the script's source path and converting it to an absolute path using `pwd`. This ensures that any subsequent operations in the script are performed relative to the script's directory.
- **Use**: `SCRIPT_DIR` is used to change the current working directory to the script's directory and to construct paths for output directories and input files.


---
### PROJ\_ROOT
- **Type**: `string`
- **Description**: The `PROJ_ROOT` variable is a string that represents the root directory of the project. It is defined by navigating two levels up from the current script directory (`SCRIPT_DIR`).
- **Use**: This variable is used to change the working directory to the project's root and to construct paths for executing binaries and saving output logs.


---
### RUN\_BIG\_TESTS
- **Type**: `boolean`
- **Description**: The `RUN_BIG_TESTS` variable is a boolean flag used to determine whether to include tests for big models in the script execution. It is initially set to `false` and is toggled to `true` if the script is run with the argument 'big', indicating that the user wants to include big model tests.
- **Use**: This variable is used to conditionally add big model tests to the test suite when the script is executed with the 'big' argument.


---
### RUN\_HUGE\_TESTS
- **Type**: `boolean`
- **Description**: The `RUN_HUGE_TESTS` variable is a boolean flag used to determine whether tests involving huge models should be executed. It is set to `true` if the script is run with the argument 'huge', indicating that both big and huge models should be included in the test suite.
- **Use**: This variable is used to conditionally add huge models to the test suite when executing the script.


---
### arr\_prefix
- **Type**: `array`
- **Description**: The `arr_prefix` variable is a global array that stores string prefixes used to categorize test cases based on their type, such as '[vision]' or '[audio]'. These prefixes are added to the array when specific test functions, like `add_test_vision` or `add_test_audio`, are called.
- **Use**: This variable is used to prepend a category label to the test results, indicating the type of test being executed.


---
### arr\_hf
- **Type**: `array`
- **Description**: The `arr_hf` variable is a global array that stores strings representing the identifiers of various models used in the testing script. These identifiers are typically in the format of a model name followed by a version or configuration string, such as 'ggml-org/SmolVLM-500M-Instruct-GGUF:Q8_0'. The array is populated by the `add_test_vision` and `add_test_audio` functions, which append model identifiers to `arr_hf` based on the tests being configured.
- **Use**: `arr_hf` is used to store and iterate over model identifiers for running tests with different models in the script.


---
### arr\_tmpl
- **Type**: `array`
- **Description**: The `arr_tmpl` variable is a global array that stores chat templates associated with different test cases. It is used to hold template strings that are optionally passed to the `add_test_vision` and `add_test_audio` functions, which append these templates to the array. The templates are used in the execution of tests to provide specific chat configurations for the models being tested.
- **Use**: This variable is used to store and manage chat templates for different test cases, which are then utilized during the execution of model tests to configure chat interactions.


---
### arr\_file
- **Type**: `array`
- **Description**: The `arr_file` variable is a global array that stores the filenames of test input files used in the script. It is populated with specific filenames, such as 'test-1.jpeg' for vision tests and 'test-2.mp3' for audio tests, through the `add_test_vision` and `add_test_audio` functions.
- **Use**: This variable is used to provide the input file paths for the tests executed by the script, allowing the test functions to access the correct files for processing.


---
### arr\_res
- **Type**: `array`
- **Description**: The `arr_res` variable is a global array that stores the results of running tests on various models. Each element in the array is a string that indicates whether a test passed or failed, along with the model and binary used in the test.
- **Use**: This variable is used to accumulate and store the results of each test execution, which are then printed out at the end of the script.


# Functions

---
### add\_test\_vision
The `add_test_vision` function appends vision model test configurations to several arrays for later execution.
- **Inputs**:
    - `hf`: A string representing the Hugging Face model identifier to be tested.
    - `tmpl`: An optional string representing the chat template to be used, defaulting to an empty string if not provided.
- **Control Flow**:
    - The function takes two parameters: `hf` and an optional `tmpl`.
    - It appends the string '[vision]' to the `arr_prefix` array.
    - It appends the `hf` parameter to the `arr_hf` array.
    - It appends the `tmpl` parameter to the `arr_tmpl` array, defaulting to an empty string if not provided.
    - It appends the string 'test-1.jpeg' to the `arr_file` array.
- **Output**: The function does not return a value; it modifies global arrays by adding new elements to them.


---
### add\_test\_audio
The `add_test_audio` function adds an audio test configuration to a set of arrays for later processing in a testing script.
- **Inputs**:
    - `hf`: A string representing the Hugging Face model identifier to be used for the audio test.
- **Control Flow**:
    - The function takes a single argument `hf` which is the Hugging Face model identifier.
    - It appends the string '[audio] ' to the `arr_prefix` array.
    - It appends the `hf` argument to the `arr_hf` array.
    - It appends an empty string to the `arr_tmpl` array, indicating no chat template is needed for audio tests.
    - It appends the string 'test-2.mp3' to the `arr_file` array, specifying the audio file to be used in the test.
- **Output**: The function does not return any value; it modifies global arrays by adding new elements to them.


