# Purpose
This Python file is a test suite designed to validate the functionality of a server that utilizes LoRA (Low-Rank Adaptation) models for text generation. The code leverages the `pytest` framework to define and execute tests that ensure the server behaves as expected when different LoRA configurations are applied. The primary focus is on testing the server's ability to generate text in different styles, such as bedtime stories or Shakespearean text, by adjusting the LoRA scale parameter. The tests are structured to verify that the server can handle requests with varying LoRA scales, both in isolation and in parallel, ensuring that the server's response content matches expected patterns.

The file includes several key components: a fixture to set up the server environment, parameterized tests to check the server's response to different LoRA scales, and a test for handling requests with a larger model. The tests involve making HTTP POST requests to the server's endpoints and asserting that the responses meet specific criteria, such as status codes and content patterns. Additionally, the file includes a test marked to be skipped unless slow tests are allowed, which tests the server's performance with a larger model. This test suite is intended to be run as part of a continuous integration process to ensure the robustness and reliability of the server's text generation capabilities.
# Imports and Dependencies

---
- `pytest`
- `utils.*`


# Global Variables

---
### server
- **Type**: `ServerPreset`
- **Description**: The `server` variable is an instance of the `ServerPreset` class, specifically initialized with the `stories15m_moe` preset. This suggests that the server is configured to handle specific tasks or models related to the 'stories15m_moe' configuration, which likely involves machine learning or AI model operations.
- **Use**: The `server` variable is used to manage and execute requests related to AI model operations, including handling LoRA adapters and processing text generation tasks.


---
### LORA\_FILE\_URL
- **Type**: `string`
- **Description**: `LORA_FILE_URL` is a string variable that holds the URL to a specific file hosted on Hugging Face's platform. This file is a LoRA (Low-Rank Adaptation) model file used for modifying the behavior of a language model.
- **Use**: This variable is used to download the LoRA model file, which is then applied to a server to alter its text generation capabilities.


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_lora.create_server}} -->
The `create_server` function is a pytest fixture that initializes a global server instance with a specific configuration and downloads a required file for testing.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture` with `scope="module"` and `autouse=True`, meaning it runs automatically once per module.
    - It declares the `server` variable as global to modify the module-level `server` variable.
    - It initializes the `server` variable using `ServerPreset.stories15m_moe()`.
    - It sets the `lora_files` attribute of the `server` to a list containing a file downloaded from `LORA_FILE_URL`.
- **Output**: The function does not return any value; it modifies the global `server` variable.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.stories15m_moe`](../utils.py.driver.md#ServerPresetstories15m_moe)
    - [`llama.cpp/tools/server/tests/utils.download_file`](../utils.py.driver.md#cpp/tools/server/tests/utilsdownload_file)


---
### test\_lora<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_lora.test_lora}} -->
The `test_lora` function tests the behavior of a server model with different LoRA scales to ensure it generates expected text content.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `scale`: A float representing the scale of the LoRA adapter to be applied to the model.
    - `re_content`: A string containing a regular expression pattern that the model's output should match.
- **Control Flow**:
    - The function starts the global server instance.
    - It sends a POST request to the '/lora-adapters' endpoint with the specified LoRA scale.
    - It asserts that the response status code from the '/lora-adapters' request is 200, indicating success.
    - It sends another POST request to the '/completion' endpoint with a fixed prompt 'Look in thy glass'.
    - It asserts that the response status code from the '/completion' request is 200, indicating success.
    - It checks if the content of the response body matches the expected regular expression pattern `re_content`.
- **Output**: The function does not return any value; it uses assertions to validate the server's behavior and output.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_lora\_per\_request<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_lora.test_lora_per_request}} -->
The `test_lora_per_request` function tests the server's ability to handle multiple parallel requests with different LoRA scales, ensuring each request is processed correctly and matches expected output patterns.
- **Inputs**: None
- **Control Flow**:
    - Set the global server's number of slots to 4 and start the server.
    - Define a prompt and a list of LoRA configurations with different scales and expected regex patterns.
    - Create a list of tasks, each representing a server request with a specific LoRA configuration, to be executed in parallel.
    - Execute the tasks in parallel using [`parallel_function_calls`](../utils.py.driver.md#cpp/tools/server/tests/utilsparallel_function_calls).
    - Assert that all responses have a status code of 200, indicating success.
    - For each response, assert that the content matches the expected regex pattern associated with its LoRA configuration.
- **Output**: The function does not return any value but asserts that all server responses are successful and match expected content patterns.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.parallel_function_calls`](../utils.py.driver.md#cpp/tools/server/tests/utilsparallel_function_calls)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_with\_big\_model<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_lora.test_with_big_model}} -->
The `test_with_big_model` function tests a large language model's response to a prompt under different LoRA scales, ensuring the model's behavior aligns with expected outputs.
- **Decorators**: `@pytest.mark.skipif`
- **Inputs**: None
- **Control Flow**:
    - Initialize a [`ServerProcess`](../utils.py.driver.md#cpp/tools/server/tests/utilsServerProcess) instance and configure it with model details and parameters such as number of slots, context size, prediction length, temperature, and seed.
    - Download and set LoRA files for the server to use.
    - Start the server with a timeout of 600 seconds.
    - Define a prompt and a list of LoRA configurations with expected responses for each configuration.
    - Create tasks to send requests to the server using different LoRA scales in parallel, each with the same prompt.
    - Execute the tasks in parallel and collect the results.
    - Assert that all responses have a status code of 200, indicating successful requests.
    - Verify that each response contains the expected content based on the LoRA configuration.
- **Output**: The function does not return any value but asserts that the model's responses match expected outputs for each LoRA configuration.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.is_slow_test_allowed`](../utils.py.driver.md#cpp/tools/server/tests/utilsis_slow_test_allowed)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess`](../utils.py.driver.md#cpp/tools/server/tests/utilsServerProcess)
    - [`llama.cpp/tools/server/tests/utils.download_file`](../utils.py.driver.md#cpp/tools/server/tests/utilsdownload_file)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.parallel_function_calls`](../utils.py.driver.md#cpp/tools/server/tests/utilsparallel_function_calls)


