# Purpose
This Python file is a test suite designed to validate the functionality and security of an API server, specifically focusing on authentication, endpoint accessibility, and CORS (Cross-Origin Resource Sharing) configurations. The code utilizes the `pytest` framework to define and execute a series of tests that ensure the server behaves as expected under various conditions. The server is instantiated using a preset configuration, and a test API key is used to simulate authenticated requests. The tests cover scenarios such as accessing public endpoints, handling incorrect and correct API keys, and verifying CORS headers for different origins and methods.

The file includes several key components: a fixture to set up the server environment, parameterized tests to check multiple endpoints and API key scenarios, and a test that integrates with the OpenAI library to simulate a chat completion request. The tests are designed to ensure that the server correctly handles authentication errors, returns appropriate status codes, and includes necessary headers in its responses. This suite is intended to be run as part of a continuous integration process to maintain the reliability and security of the API server.
# Imports and Dependencies

---
- `pytest`
- `openai.OpenAI`
- `utils.*`


# Global Variables

---
### server
- **Type**: `ServerPreset`
- **Description**: The `server` variable is an instance of the `ServerPreset` class, specifically initialized with the `tinyllama2` preset. This suggests that it is configured with a specific set of parameters or settings suitable for a particular server environment or use case.
- **Use**: The `server` variable is used to start a server instance and make HTTP requests to test various endpoints and functionalities, such as authentication and CORS handling, in the provided test suite.


---
### TEST\_API\_KEY
- **Type**: `string`
- **Description**: `TEST_API_KEY` is a global variable that holds a string representing a secret API key used for authentication purposes in the test suite. It is a placeholder key used to simulate authentication in various test cases.
- **Use**: This variable is used to authenticate requests to the server in test cases, ensuring that the server responds correctly to both valid and invalid API keys.


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_security.create_server}} -->
The `create_server` function is a pytest fixture that initializes a global server instance with a specific preset and assigns it a test API key for use in module-scoped tests.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture` with `scope='module'` and `autouse=True`, meaning it runs automatically once per module before any tests are executed.
    - It declares the `server` variable as global to modify the module-level `server` variable.
    - The function initializes the `server` variable using `ServerPreset.tinyllama2()`, which presumably sets up a server with a specific configuration.
    - The function assigns the `TEST_API_KEY` to the `server.api_key` attribute, configuring the server to use this API key for authentication.
- **Output**: The function does not return any value; it sets up a global server instance for testing purposes.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama2`](../utils.py.driver.md#ServerPresettinyllama2)


---
### test\_access\_public\_endpoint<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_security.test_access_public_endpoint}} -->
The `test_access_public_endpoint` function tests if public endpoints of a server are accessible and return a successful response without errors.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `endpoint`: A string representing the endpoint to be tested, either '/health' or '/models'.
- **Control Flow**:
    - The function starts the server using `server.start()`.
    - It makes a GET request to the specified endpoint using `server.make_request()`.
    - The function asserts that the response status code is 200, indicating success.
    - It also asserts that the response body does not contain the word 'error', ensuring no errors are present in the response.
- **Output**: The function does not return any value; it uses assertions to validate the server's response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_incorrect\_api\_key<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_security.test_incorrect_api_key}} -->
The `test_incorrect_api_key` function tests the server's response to requests made with either a missing or invalid API key, ensuring it returns an authentication error.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `api_key`: A string representing the API key to be tested, which can be either `None` or an 'invalid-key'.
- **Control Flow**:
    - The function is decorated with `@pytest.mark.parametrize` to run the test with two different `api_key` values: `None` and 'invalid-key'.
    - The server is started using `server.start()`.
    - A POST request is made to the `/completions` endpoint with a prompt in the data and an Authorization header containing the API key if provided.
    - The response is checked to ensure the status code is 401, indicating unauthorized access.
    - The response body is checked to confirm it contains an 'error' key.
    - The error type in the response body is verified to be 'authentication_error'.
- **Output**: The function does not return a value; it asserts conditions to validate the server's response to incorrect API keys.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_correct\_api\_key<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_security.test_correct_api_key}} -->
The function `test_correct_api_key` tests if a server correctly processes a request with a valid API key by checking the response status and content.
- **Inputs**: None
- **Control Flow**:
    - The function starts the server using `server.start()`.
    - It makes a POST request to the `/completions` endpoint with a prompt and a valid API key in the headers.
    - The function asserts that the response status code is 200, indicating success.
    - It checks that there is no 'error' in the response body.
    - It verifies that 'content' is present in the response body.
- **Output**: The function does not return any value; it uses assertions to validate the server's response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_openai\_library\_correct\_api\_key<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_security.test_openai_library_correct_api_key}} -->
The function `test_openai_library_correct_api_key` tests the OpenAI library's ability to create a chat completion using a correct API key.
- **Inputs**: None
- **Control Flow**:
    - The function starts by accessing the global `server` variable and starting the server.
    - An `OpenAI` client is instantiated with the correct `TEST_API_KEY` and the server's base URL.
    - A chat completion request is made using the `client` with a predefined model and message sequence.
    - The function asserts that the response contains exactly one choice, indicating a successful completion.
- **Output**: The function does not return any value but asserts that the response from the OpenAI client contains exactly one choice, ensuring the API key is correct and the request is successful.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)


---
### test\_cors\_options<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_security.test_cors_options}} -->
The `test_cors_options` function tests the server's CORS (Cross-Origin Resource Sharing) configuration by sending OPTIONS requests and verifying the response headers.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `origin`: A string representing the origin of the request, used to test different CORS configurations.
    - `cors_header`: A string representing the expected CORS header in the response.
    - `cors_header_value`: A string representing the expected value of the CORS header in the response.
- **Control Flow**:
    - The function starts the server using `server.start()`.
    - It makes an OPTIONS request to the `/completions` endpoint with specified headers including `Origin`, `Access-Control-Request-Method`, and `Access-Control-Request-Headers`.
    - The function asserts that the response status code is 200, indicating a successful request.
    - It checks that the expected CORS header is present in the response headers.
    - It verifies that the value of the CORS header matches the expected value.
- **Output**: The function does not return any value; it uses assertions to validate the server's CORS configuration.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


