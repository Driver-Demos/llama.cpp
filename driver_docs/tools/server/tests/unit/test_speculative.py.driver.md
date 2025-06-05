# Purpose
This Python code file is a test suite designed to validate the functionality of a server model, specifically focusing on its behavior with and without a draft model, as well as its handling of various configuration parameters. The code uses the `pytest` framework to define and execute a series of tests that assess the server's response to different scenarios. The server is configured using a preset model, and the tests ensure that the server can handle requests correctly, maintain consistent output across different draft configurations, and manage context and slot settings effectively. The tests also include parallel request handling to verify the server's capability to process multiple requests simultaneously.

The file imports utility functions and uses a fixture to set up the server environment before running the tests. Key components include the [`create_server`](#cpp/tools/server/tests/unit/test_speculativecreate_server) function, which initializes the server with a specific model draft, and several test functions that check the server's response to various input parameters and configurations. The tests cover scenarios such as disabling the draft model, varying draft minimum and maximum values, ensuring context limits are not exceeded, and verifying the server's ability to handle context shifts and parallel requests. This code is intended to be part of a larger testing framework, ensuring the robustness and reliability of the server model's functionality.
# Imports and Dependencies

---
- `pytest`
- `utils.*`


# Global Variables

---
### server
- **Type**: `ServerPreset`
- **Description**: The `server` variable is an instance of the `ServerPreset` class, specifically initialized with the `stories15m_moe` preset. This preset likely configures the server with specific model parameters and settings for handling requests related to story generation or similar tasks.
- **Use**: The `server` variable is used to manage and execute server operations, including starting, stopping, and handling requests for text completion tasks.


---
### MODEL\_DRAFT\_FILE\_URL
- **Type**: `str`
- **Description**: `MODEL_DRAFT_FILE_URL` is a string variable that holds the URL to a draft model file hosted on Hugging Face's repository. This URL points to a specific model file named `stories15M-q4_0.gguf` within the `tinyllamas` directory.
- **Use**: This variable is used to download the draft model file for the server setup in the `create_server` function.


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_speculative.create_server}} -->
The `create_server` function initializes a global server instance with specific default settings for a model draft and draft range.
- **Inputs**: None
- **Control Flow**:
    - The function declares the `server` variable as global, allowing it to modify the global `server` instance.
    - It initializes the `server` variable using the `ServerPreset.stories15m_moe()` method, which presumably sets up a server with a specific configuration.
    - The function sets the `model_draft` attribute of the `server` to the result of `download_file(MODEL_DRAFT_FILE_URL)`, which downloads a file from a specified URL.
    - It sets the `draft_min` attribute of the `server` to 4, establishing a minimum draft value.
    - It sets the `draft_max` attribute of the `server` to 8, establishing a maximum draft value.
- **Output**: The function does not return any value; it modifies the global `server` instance.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.stories15m_moe`](../utils.py.driver.md#ServerPresetstories15m_moe)
    - [`llama.cpp/tools/server/tests/utils.download_file`](../utils.py.driver.md#cpp/tools/server/tests/utilsdownload_file)


---
### fixture\_create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_speculative.fixture_create_server}} -->
The `fixture_create_server` function is a pytest fixture that initializes and returns a server instance for testing purposes.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture` with a scope of 'module' and is set to run automatically before any tests in the module.
    - The function calls the [`create_server`](#cpp/tools/server/tests/unit/test_speculativecreate_server) function, which initializes a global `server` object with default settings.
- **Output**: The function returns the result of the [`create_server`](#cpp/tools/server/tests/unit/test_speculativecreate_server) function, which is a server instance.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/unit/test_speculative.create_server`](#cpp/tools/server/tests/unit/test_speculativecreate_server)


---
### test\_with\_and\_without\_draft<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_speculative.test_with_and_without_draft}} -->
The function `test_with_and_without_draft` tests the server's response consistency with and without a draft model enabled.
- **Inputs**: None
- **Control Flow**:
    - The function begins by setting the global `server`'s `model_draft` attribute to `None` to disable the draft model.
    - The server is started, and a POST request is made to the `/completion` endpoint with specific data parameters.
    - The response status code is asserted to be 200, and the content of the response is stored in `content_no_draft`.
    - The server is stopped, and a new server instance is created with the draft model enabled by calling `create_server()`.
    - The server is started again, and the same POST request is made to the `/completion` endpoint.
    - The response status code is asserted to be 200, and the content of the response is stored in `content_draft`.
    - Finally, the function asserts that the content from the response with the draft model (`content_draft`) is equal to the content without the draft model (`content_no_draft`).
- **Output**: The function does not return any value; it asserts the equality of server responses with and without the draft model.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.stop`](../utils.py.driver.md#ServerProcessstop)
    - [`llama.cpp/tools/server/tests/unit/test_speculative.create_server`](#cpp/tools/server/tests/unit/test_speculativecreate_server)


---
### test\_different\_draft\_min\_draft\_max<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_speculative.test_different_draft_min_draft_max}} -->
The function tests the server's response consistency when varying the draft_min and draft_max parameters.
- **Inputs**: None
- **Control Flow**:
    - Initialize a list of test values for draft_min and draft_max pairs.
    - Set last_content to None to store the content from the previous iteration.
    - Iterate over each pair of draft_min and draft_max values in the test_values list.
    - For each pair, stop the server, set the server's draft_min and draft_max to the current pair, and restart the server.
    - Make a POST request to the server with a fixed prompt and parameters, and assert that the response status code is 200.
    - If last_content is not None, assert that the current response content matches last_content.
    - Update last_content with the current response content.
- **Output**: The function does not return any value; it asserts the consistency of server responses.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.stop`](../utils.py.driver.md#ServerProcessstop)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_slot\_ctx\_not\_exceeded<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_speculative.test_slot_ctx_not_exceeded}} -->
The function `test_slot_ctx_not_exceeded` tests that the server can handle a request with a large prompt without exceeding the context limit.
- **Inputs**: None
- **Control Flow**:
    - Sets the global server's context size (`n_ctx`) to 64.
    - Starts the server.
    - Makes a POST request to the `/completion` endpoint with a prompt repeated 56 times and specific parameters for temperature, top_k, and speculative.p_min.
    - Asserts that the response status code is 200, indicating success.
    - Asserts that the response body contains content, ensuring the request was processed correctly.
- **Output**: The function does not return any value; it uses assertions to validate the server's behavior.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_with\_ctx\_shift<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_speculative.test_with_ctx_shift}} -->
The function `test_with_ctx_shift` tests the server's ability to handle a specific context size and verify the response properties when making a POST request to the '/completion' endpoint.
- **Inputs**: None
- **Control Flow**:
    - Sets the global server's context size (`n_ctx`) to 64.
    - Starts the server.
    - Makes a POST request to the '/completion' endpoint with specific data including a repeated 'Hello' prompt and prediction parameters.
    - Asserts that the response status code is 200, indicating a successful request.
    - Checks that the response body contains content and that the number of tokens predicted is 64.
    - Verifies that the response indicates the content was truncated.
- **Output**: The function does not return any value; it performs assertions to validate the server's response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_multi\_requests\_parallel<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_speculative.test_multi_requests_parallel}} -->
The function `test_multi_requests_parallel` tests the server's ability to handle multiple parallel requests with varying slot configurations.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `n_slots`: The number of slots to configure on the server for handling requests.
    - `n_requests`: The number of parallel requests to be made to the server.
- **Control Flow**:
    - The function is parameterized to run with different combinations of `n_slots` and `n_requests`.
    - The server's slot configuration is set to `n_slots`.
    - The server is started to handle requests.
    - A list of tasks is created, each representing a request to the server with a predefined payload.
    - The [`parallel_function_calls`](../utils.py.driver.md#cpp/tools/server/tests/utilsparallel_function_calls) function is called with the list of tasks to execute them in parallel.
    - The results of the parallel requests are iterated over to assert that each response has a status code of 200 and the content matches a specific regex pattern.
- **Output**: The function does not return any value; it asserts the correctness of server responses to parallel requests.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.parallel_function_calls`](../utils.py.driver.md#cpp/tools/server/tests/utilsparallel_function_calls)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


