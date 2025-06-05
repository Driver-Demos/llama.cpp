# Purpose
This Python script is a test suite using the `pytest` framework to validate the functionality of a server process, specifically focusing on image recognition capabilities. It imports necessary modules, including `pytest`, utility functions, and libraries for handling HTTP requests and base64 encoding. The script defines a fixture to automatically create and start a server instance using a predefined server preset. It then uses parameterized tests to send various prompts and image URLs to the server's `/chat/completions` endpoint, checking for expected outcomes such as successful responses and correct content recognition. The tests cover scenarios with valid and invalid image URLs, including base64-encoded images, and verify the server's ability to handle different types of input and return appropriate responses.
# Imports and Dependencies

---
- `pytest`
- `utils.*`
- `base64`
- `requests`


# Global Variables

---
### server
- **Type**: `ServerProcess`
- **Description**: The `server` variable is an instance of the `ServerProcess` class, which is likely used to manage a server process for testing purposes. It is initialized in the `create_server` fixture, which is automatically used in the test suite to set up the server environment before tests are run.
- **Use**: This variable is used to start the server and make requests to it during the execution of test cases.


---
### IMG\_URL\_0
- **Type**: `str`
- **Description**: `IMG_URL_0` is a string variable that holds the URL of an image hosted on the Hugging Face platform. The image is located at the path `/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png` and is likely used for testing purposes in the context of the code.
- **Use**: This variable is used to fetch the image from the specified URL for testing the vision chat completion functionality.


---
### IMG\_URL\_1
- **Type**: `str`
- **Description**: `IMG_URL_1` is a string variable that holds the URL of an image hosted on the Hugging Face platform. The image is identified by the filename `91_cat.png`, suggesting it is an image of a cat.
- **Use**: This variable is used in test cases to provide a URL for image-based prompts in a vision chat completion test.


---
### response
- **Type**: `requests.models.Response`
- **Description**: The `response` variable is an instance of the `Response` object returned by the `requests.get()` method, which is used to send a GET request to the specified URL (`IMG_URL_0`). This variable holds the server's response to the HTTP request, including the status code, headers, and content of the response.
- **Use**: The `response` variable is used to check the status of the HTTP request and to access the content of the image for further processing.


---
### IMG\_BASE64\_0
- **Type**: `str`
- **Description**: `IMG_BASE64_0` is a string that represents the base64-encoded version of an image fetched from a specified URL. It is prefixed with 'data:image/png;base64,' to indicate that it is a base64-encoded PNG image.
- **Use**: This variable is used to store and provide a base64-encoded image string for testing purposes in the `test_vision_chat_completion` function.


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_vision_api.create_server}} -->
The `create_server` function is a pytest fixture that initializes a global server instance using the `ServerPreset.tinygemma3()` method.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture(autouse=True)`, which means it will automatically run before each test function in the module.
    - It sets a global variable `server` to an instance of `ServerPreset.tinygemma3()`.
- **Output**: The function does not return any value; it sets a global variable `server`.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinygemma3`](../utils.py.driver.md#ServerPresettinygemma3)


---
### test\_vision\_chat\_completion<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_vision_api.test_vision_chat_completion}} -->
The `test_vision_chat_completion` function tests the server's ability to process image-based chat completions using various image URLs and prompts.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `prompt`: A string representing the text prompt to be sent to the server.
    - `image_url`: A string representing the URL or base64 encoded string of the image to be sent to the server.
    - `success`: A boolean indicating whether the request is expected to succeed.
    - `re_content`: A regular expression pattern to match against the server's response content if the request is successful.
- **Control Flow**:
    - The server is started with a timeout of 60 seconds to allow for model loading.
    - If the `image_url` is 'IMG_BASE64_0', it is replaced with the actual base64 encoded image string.
    - A POST request is made to the server with the prompt and image URL as part of the message data.
    - If `success` is True, the function asserts that the response status code is 200, the role in the response is 'assistant', and the content matches the provided regular expression.
    - If `success` is False, the function asserts that the response status code is not 200.
- **Output**: The function does not return a value but asserts the correctness of the server's response based on the input parameters.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


