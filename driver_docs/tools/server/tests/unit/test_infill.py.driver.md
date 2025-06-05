# Purpose
This Python file is a test suite designed to validate the functionality of a server that provides code infilling capabilities. The tests are implemented using the `pytest` framework, which is a popular testing tool in Python for writing simple as well as scalable test cases. The server is instantiated using a preset configuration, `ServerPreset.tinyllama_infill()`, which suggests that it is configured to handle specific infilling tasks, likely related to code completion or generation. The tests focus on making HTTP POST requests to the server's `/infill` endpoint with various input data, including code snippets and additional file content, to verify that the server responds correctly and performs the expected infilling operations.

The file contains several test functions, each targeting different aspects of the server's functionality. The [`test_infill_without_input_extra`](#cpp/tools/server/tests/unit/test_infilltest_infill_without_input_extra) and [`test_infill_with_input_extra`](#cpp/tools/server/tests/unit/test_infilltest_infill_with_input_extra) functions test the server's ability to handle requests with and without additional file content, respectively. The [`test_invalid_input_extra_req`](#cpp/tools/server/tests/unit/test_infilltest_invalid_input_extra_req) function uses parameterization to test the server's response to invalid input data, ensuring that it returns appropriate error messages. Additionally, the [`test_with_qwen_model`](#cpp/tools/server/tests/unit/test_infilltest_with_qwen_model) function is a conditional test that runs only if slow tests are allowed, and it tests the server's behavior with a specific model configuration. This test suite is crucial for ensuring the reliability and correctness of the server's infilling feature, making it an essential component of the software's quality assurance process.
# Imports and Dependencies

---
- `pytest`
- `utils.*`


# Global Variables

---
### server
- **Type**: `ServerPreset`
- **Description**: The `server` variable is an instance of the `ServerPreset` class, specifically initialized with the `tinyllama_infill` preset. This suggests that it is configured to handle specific server operations related to infilling tasks, likely involving code completion or text generation.
- **Use**: The `server` variable is used to start a server instance and make HTTP POST requests to the `/infill` endpoint for testing purposes.


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_infill.create_server}} -->
The `create_server` function is a pytest fixture that initializes a global server instance using the `ServerPreset.tinyllama_infill` configuration.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture` with `scope="module"` and `autouse=True`, meaning it runs automatically once per module.
    - It sets the global variable `server` to an instance of `ServerPreset.tinyllama_infill`.
- **Output**: The function does not return any value; it modifies the global `server` variable.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama_infill`](../utils.py.driver.md#ServerPresettinyllama_infill)


---
### test\_infill\_without\_input\_extra<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_infill.test_infill_without_input_extra}} -->
The function `test_infill_without_input_extra` tests the server's ability to handle a POST request to the '/infill' endpoint without additional input data and verifies the response content.
- **Inputs**: None
- **Control Flow**:
    - The function starts by declaring the `server` as a global variable and starting it using `server.start()`.
    - A POST request is made to the '/infill' endpoint of the server with specific data including `input_prefix`, `prompt`, and `input_suffix`.
    - The response status code is asserted to be 200, indicating a successful request.
    - The response body content is checked against a regular expression pattern to ensure it matches expected content.
- **Output**: The function does not return any value but asserts the correctness of the server response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_infill\_with\_input\_extra<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_infill.test_infill_with_input_extra}} -->
The function `test_infill_with_input_extra` tests the server's ability to handle a POST request with additional input data for code infilling.
- **Inputs**: None
- **Control Flow**:
    - The function starts by declaring the `server` as a global variable and starting it.
    - It makes a POST request to the `/infill` endpoint of the server with a data payload that includes `input_extra`, `input_prefix`, `prompt`, and `input_suffix`.
    - The `input_extra` contains a filename and text, simulating additional input data for the infill operation.
    - The function asserts that the response status code is 200, indicating a successful request.
    - It then checks if the response body content matches a specific regex pattern `(Dad|excited|park)+`.
- **Output**: The function does not return any value but asserts the correctness of the server's response to the request.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_invalid\_input\_extra\_req<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_infill.test_invalid_input_extra_req}} -->
The function `test_invalid_input_extra_req` tests the server's response to invalid `input_extra` data by ensuring it returns a 400 status code and an error message.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `input_extra`: A dictionary representing additional input data to be tested, which is parameterized with various invalid configurations.
- **Control Flow**:
    - The function is decorated with `@pytest.mark.parametrize` to run the test multiple times with different `input_extra` values.
    - The server is started using `server.start()`.
    - A POST request is made to the `/infill` endpoint with the `input_extra` data included in the request body.
    - The function asserts that the response status code is 400, indicating a bad request.
    - The function also asserts that the response body contains an 'error' key, confirming that an error message is returned.
- **Output**: The function does not return any value; it asserts conditions to validate the server's response to invalid input.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_with\_qwen\_model<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_infill.test_with_qwen_model}} -->
The function `test_with_qwen_model` tests the server's ability to handle a specific model configuration and verify the response content for a code infill request.
- **Decorators**: `@pytest.mark.skipif`
- **Inputs**: None
- **Control Flow**:
    - The function begins by setting the global `server`'s model file to `None` and configuring it with a specific Hugging Face repository and file for the Qwen model.
    - The server is started with a timeout of 600 seconds.
    - A POST request is made to the `/infill` endpoint with specific data including `input_extra`, `input_prefix`, `prompt`, and `input_suffix`.
    - The response status code is asserted to be 200, indicating a successful request.
    - The response body content is asserted to match the expected C code snippet.
- **Output**: The function does not return any value but asserts the correctness of the server's response to a model-specific infill request.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.is_slow_test_allowed`](../utils.py.driver.md#cpp/tools/server/tests/utilsis_slow_test_allowed)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


