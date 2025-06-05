# Purpose
This Python script is a collection of test cases using the `pytest` framework to verify the functionality of a server instance, specifically one configured with the `tinyllama2` preset. The script provides narrow functionality focused on testing various server endpoints and configurations, such as health checks, properties, model loading, slot management, and web UI availability. It uses fixtures to set up the server environment and includes multiple test functions that start the server, make HTTP requests to different endpoints, and assert expected outcomes. The tests ensure that the server behaves correctly under different configurations, such as with or without web UI, and when handling model data and slots. This script is essential for validating the server's reliability and correctness in handling requests and configurations.
# Imports and Dependencies

---
- `pytest`
- `requests`
- `utils.*`


# Global Variables

---
### server
- **Type**: `ServerPreset`
- **Description**: The `server` variable is an instance of the `ServerPreset` class, specifically initialized with the `tinyllama2` preset. This suggests that it is configured with predefined settings suitable for a 'tinyllama2' server environment, which likely includes specific model configurations and server properties.
- **Use**: This variable is used to manage and interact with a server instance throughout the test suite, allowing for operations such as starting the server, making requests, and verifying server responses.


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_basic.create_server}} -->
The `create_server` function is a pytest fixture that initializes a global server instance using the `ServerPreset.tinyllama2()` configuration for use in module-scoped tests.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture` with `scope='module'` and `autouse=True`, meaning it runs automatically once per module before any tests are executed.
    - The function sets a global variable `server` to an instance of `ServerPreset.tinyllama2()`.
- **Output**: The function does not return any value; it sets up a global server instance for testing purposes.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama2`](../utils.py.driver.md#ServerPresettinyllama2)


---
### test\_server\_start\_simple<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_basic.test_server_start_simple}} -->
The function `test_server_start_simple` tests if the server can start and respond with a 200 status code to a health check request.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the global `server` object.
    - It calls the [`start`](../utils.py.driver.md#ServerProcessstart) method on the `server` to initiate it.
    - A GET request is made to the `/health` endpoint of the server using `server.make_request`.
    - The response status code is asserted to be 200, indicating a successful health check.
- **Output**: The function does not return any value; it asserts the server's health check response status code is 200.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_server\_props<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_basic.test_server_props}} -->
The `test_server_props` function tests the server's properties endpoint to ensure it returns the correct model path, total slots, and default generation settings.
- **Inputs**: None
- **Control Flow**:
    - The function starts the server using `server.start()`.
    - It makes a GET request to the `/props` endpoint using `server.make_request`.
    - The function asserts that the response status code is 200, indicating a successful request.
    - It checks that the model path in the response body contains the string '.gguf'.
    - The function verifies that the total slots in the response body match the server's `n_slots`.
    - It retrieves the `default_generation_settings` from the response body and performs several assertions:
    - It asserts that `server.n_ctx` and `server.n_slots` are not `None`.
    - It checks that `default_generation_settings['n_ctx']` equals `server.n_ctx / server.n_slots`.
    - It asserts that `default_generation_settings['params']['seed']` matches `server.seed`.
- **Output**: The function does not return any value; it raises an assertion error if any of the checks fail.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_server\_models<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_basic.test_server_models}} -->
The `test_server_models` function tests the server's ability to correctly return model information via a GET request to the '/models' endpoint.
- **Inputs**: None
- **Control Flow**:
    - The function begins by declaring the `server` variable as global to ensure it uses the global server instance.
    - The server is started using `server.start()`.
    - A GET request is made to the '/models' endpoint using `server.make_request`.
    - The response status code is asserted to be 200, indicating a successful request.
    - The length of the 'data' array in the response body is asserted to be 1, ensuring only one model is returned.
    - The 'id' of the first model in the 'data' array is asserted to match `server.model_alias`, verifying the correct model is returned.
- **Output**: The function does not return any value; it uses assertions to validate the server's response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_server\_slots<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_basic.test_server_slots}} -->
The `test_server_slots` function tests the server's response to the `/slots` endpoint with and without the slots feature enabled.
- **Inputs**: None
- **Control Flow**:
    - Set `server.server_slots` to `False` and start the server.
    - Make a GET request to the `/slots` endpoint and assert that the response status code is 501, indicating the feature is not supported.
    - Check that the response body contains an error message.
    - Stop the server.
    - Set `server.server_slots` to `True`, set `server.n_slots` to 2, and start the server again.
    - Make a GET request to the `/slots` endpoint and assert that the response status code is 200, indicating success.
    - Assert that the length of the response body matches `server.n_slots`.
    - Verify that `server.n_ctx` and `server.n_slots` are not `None`.
    - Check that the `n_ctx` value in the response body is equal to `server.n_ctx` divided by `server.n_slots`.
    - Ensure that the response body contains a `params` field with a `seed` value matching `server.seed`.
- **Output**: The function does not return any value; it uses assertions to validate server behavior.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.stop`](../utils.py.driver.md#ServerProcessstop)


---
### test\_load\_split\_model<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_basic.test_load_split_model}} -->
The `test_load_split_model` function tests the server's ability to load a specific split model and generate a completion response.
- **Inputs**: None
- **Control Flow**:
    - Set the server's model repository to 'ggml-org/models'.
    - Set the server's model file to 'tinyllamas/split/stories15M-q8_0-00001-of-00003.gguf'.
    - Set the server's model alias to 'tinyllama-split'.
    - Start the server.
    - Make a POST request to the '/completion' endpoint with specific data parameters.
    - Assert that the response status code is 200, indicating success.
    - Assert that the response body content matches the regex pattern '(little|girl)+' to verify the model's output.
- **Output**: The function does not return any value; it uses assertions to validate the server's behavior and model output.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_no\_webui<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_basic.test_no_webui}} -->
The `test_no_webui` function tests the server's behavior with and without the web UI enabled.
- **Inputs**: None
- **Control Flow**:
    - The function starts the server with the default settings where the web UI is enabled.
    - It constructs a URL using the server's host and port and sends a GET request to this URL.
    - The function asserts that the response status code is 200 and that the response text contains HTML content, indicating the web UI is accessible.
    - The server is stopped.
    - The server is then configured to disable the web UI by setting `server.no_webui` to `True`.
    - The server is restarted, and another GET request is sent to the same URL.
    - The function asserts that the response status code is 404, indicating the web UI is not accessible.
- **Output**: The function does not return any value; it uses assertions to validate the server's behavior.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.stop`](../utils.py.driver.md#ServerProcessstop)


