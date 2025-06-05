# Purpose
This Python file is a test suite designed to validate the functionality of a server, specifically focusing on context management and text completion capabilities. The code uses the `pytest` framework to define and execute a series of tests that assess how the server handles text prompts of varying lengths and configurations. The server is instantiated using a preset configuration (`ServerPreset.tinyllama2()`), and the tests are structured to evaluate both the enabling and disabling of context shifting—a feature that allows the server to manage and generate text when the context size is exceeded.

The file includes several test functions, each targeting specific scenarios: [`test_ctx_shift_enabled`](#cpp/tools/server/tests/unit/test_ctx_shifttest_ctx_shift_enabled) checks the server's ability to handle a long prompt with context shifting enabled, ensuring that the prompt is truncated correctly and additional tokens are generated. [`test_ctx_shift_disabled_short_prompt`](#cpp/tools/server/tests/unit/test_ctx_shifttest_ctx_shift_disabled_short_prompt) and [`test_ctx_shift_disabled_long_prompt`](#cpp/tools/server/tests/unit/test_ctx_shifttest_ctx_shift_disabled_long_prompt) assess the server's behavior when context shifting is disabled, with the former focusing on short prompts and the latter on long prompts that exceed the context size. Additionally, [`test_ctx_shift_disabled_stream`](#cpp/tools/server/tests/unit/test_ctx_shifttest_ctx_shift_disabled_stream) evaluates the server's streaming capabilities under the same conditions. The tests make HTTP POST requests to the server's completion endpoint and assert the correctness of the server's responses, including status codes, token counts, and error messages. This file is intended to be run as part of a test suite to ensure the server's text completion features work as expected under different configurations.
# Imports and Dependencies

---
- `pytest`
- `utils.*`


# Global Variables

---
### server
- **Type**: `ServerPreset`
- **Description**: The `server` variable is an instance of the `ServerPreset` class, specifically initialized with the `tinyllama2` preset. This suggests that it is configured with specific settings or parameters that are predefined in the `tinyllama2` preset.
- **Use**: The `server` variable is used to manage and execute server operations, such as handling requests and managing context shifts, within the test functions.


---
### LONG\_TEXT
- **Type**: `str`
- **Description**: `LONG_TEXT` is a string variable that contains a multi-line Lorem Ipsum text. Lorem Ipsum is a placeholder text commonly used to demonstrate the visual form of a document or a typeface without relying on meaningful content.
- **Use**: This variable is used as a prompt in server requests to test the server's handling of text input.


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_ctx_shift.create_server}} -->
The `create_server` function is a pytest fixture that initializes a global server instance with specific context and slot settings for testing purposes.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture` with `scope="module"` and `autouse=True`, meaning it runs automatically once per module.
    - A global variable `server` is set to a new instance of `ServerPreset.tinyllama2()`.
    - The `server` instance's `n_ctx` attribute is set to 256, and `n_slots` is set to 2.
- **Output**: The function does not return any value; it sets up a global server instance for use in tests.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama2`](../utils.py.driver.md#ServerPresettinyllama2)


---
### test\_ctx\_shift\_enabled<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_ctx_shift.test_ctx_shift_enabled}} -->
The function `test_ctx_shift_enabled` tests the server's ability to handle context shifting when generating text completions with a long prompt.
- **Inputs**: None
- **Control Flow**:
    - The function starts by initializing the global `server` object and starting the server.
    - A POST request is made to the `/completion` endpoint with a specified number of tokens to predict (`n_predict`) and a long prompt (`LONG_TEXT`).
    - The response from the server is checked to ensure the status code is 200, indicating a successful request.
    - Assertions are made to verify that the prompt was truncated to 109 tokens, 64 tokens were predicted, and the prompt was indeed truncated.
- **Output**: The function does not return any value; it uses assertions to validate the server's behavior.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_ctx\_shift\_disabled\_short\_prompt<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_ctx_shift.test_ctx_shift_disabled_short_prompt}} -->
The function `test_ctx_shift_disabled_short_prompt` tests the behavior of a server's text completion feature when context shifting is disabled, using a short prompt.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `n_predict`: An integer indicating the number of tokens to predict.
    - `n_token_output`: An integer representing the expected number of tokens in the output.
    - `truncated`: A boolean indicating whether the output is expected to be truncated.
- **Control Flow**:
    - The function is parameterized to run with different sets of inputs: (64, 64, False) and (-1, 120, True).
    - The global `server` object is configured to disable context shifting and set `n_predict` to -1.
    - The server is started, and a POST request is made to the '/completion' endpoint with the specified `n_predict` and a short prompt 'Hi how are you'.
    - The response is checked to ensure the status code is 200, indicating a successful request.
    - Assertions are made to verify that the number of predicted tokens matches `n_token_output` and that the truncation status matches `truncated`.
- **Output**: The function does not return a value; it asserts conditions to validate server behavior.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_ctx\_shift\_disabled\_long\_prompt<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_ctx_shift.test_ctx_shift_disabled_long_prompt}} -->
The function `test_ctx_shift_disabled_long_prompt` tests the server's response when context shifting is disabled and a long prompt is used, expecting an error due to context size limitations.
- **Inputs**: None
- **Control Flow**:
    - The global `server` object is accessed and its `disable_ctx_shift` attribute is set to `True` to disable context shifting.
    - The server is started using `server.start()`.
    - A POST request is made to the `/completion` endpoint with a payload containing `n_predict` set to 64 and a long prompt `LONG_TEXT`.
    - The response is checked to ensure the status code is not 200, indicating an error occurred.
    - The response body is checked to confirm it contains an 'error' key.
    - The error message is verified to contain the phrase 'exceeds the available context size'.
- **Output**: The function does not return any value; it asserts conditions to validate the server's behavior under specific test conditions.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_ctx\_shift\_disabled\_stream<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_ctx_shift.test_ctx_shift_disabled_stream}} -->
The function `test_ctx_shift_disabled_stream` tests the server's behavior when context shifting is disabled and a streaming request is made.
- **Inputs**: None
- **Control Flow**:
    - The global `server` object is accessed and its `disable_ctx_shift` attribute is set to `True`.
    - The server is started using `server.start()`.
    - A streaming request is made to the server with a POST method to the endpoint `/v1/completions`, with data specifying `n_predict` as 256, a `prompt` of 'Once', and `stream` set to `True`.
    - An empty string `content` is initialized to accumulate text from the response.
    - The response `res` is iterated over, and for each `data` item, the first choice is accessed.
    - If the `finish_reason` of the choice is 'length', it asserts that `content` is not empty.
    - If the `finish_reason` is `None`, it appends the `text` from the choice to `content`.
- **Output**: The function does not return any value; it performs assertions to validate the server's response behavior.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_stream_request`](../utils.py.driver.md#ServerProcessmake_stream_request)


