# Purpose
This Python file is a comprehensive test suite designed to validate the functionality of a chat completion server, likely part of a larger system involving AI-driven text generation or conversation models. The file uses the `pytest` framework to define a series of test cases that assess various aspects of the server's behavior, including its ability to handle different input parameters, response formats, and streaming capabilities. The tests are organized using `pytest` fixtures and parameterized test functions, which allow for the systematic testing of multiple scenarios with varying inputs and expected outcomes.

Key components of the file include the use of a `ServerProcess` object, which is presumably a mock or a test instance of the server being tested. The tests cover a wide range of functionalities, such as handling different chat templates, applying JSON schemas, and processing grammars. Additionally, the file includes tests for the server's integration with the OpenAI library, ensuring that the server can correctly process requests and return expected results. The tests also verify the server's ability to handle invalid requests gracefully and to provide detailed logging information, such as log probabilities, which are crucial for debugging and performance analysis. Overall, this file serves as a critical component in ensuring the reliability and correctness of the chat completion server's implementation.
# Imports and Dependencies

---
- `pytest`
- `openai.OpenAI`
- `utils.*`


# Global Variables

---
### server
- **Type**: `ServerProcess`
- **Description**: The `server` variable is a global instance of the `ServerProcess` class, which is used to manage and interact with a server process for testing purposes. It is initialized in the `create_server` fixture with a specific server preset configuration, `ServerPreset.tinyllama2()`, and is used throughout the test functions to start the server and make requests to it.
- **Use**: This variable is used to manage the lifecycle of a server process, including starting the server and making HTTP requests to test chat completion functionalities.


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.create_server}} -->
The `create_server` function is a pytest fixture that initializes a global server variable with a specific server preset configuration.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture` with `autouse=True`, meaning it will automatically run before each test function in the module.
    - It sets the global variable `server` to an instance of `ServerPreset.tinyllama2()`.
- **Output**: The function does not return any value; it modifies the global `server` variable.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama2`](../utils.py.driver.md#ServerPresettinyllama2)


---
### test\_chat\_completion<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.test_chat_completion}} -->
The `test_chat_completion` function tests the chat completion endpoint of a server by sending various parameterized requests and validating the responses against expected values.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `model`: The model to be used for the chat completion, or None to use the server's default model.
    - `system_prompt`: The initial system message content for the chat.
    - `user_prompt`: The user's message content for the chat, which can be a string or a list of message objects.
    - `max_tokens`: The maximum number of tokens allowed for the completion.
    - `re_content`: A regular expression pattern to match against the content of the assistant's message.
    - `n_prompt`: The expected number of tokens used in the prompt.
    - `n_predicted`: The expected number of tokens predicted in the completion.
    - `finish_reason`: The expected reason for the completion to finish, such as 'length'.
    - `jinja`: A boolean indicating whether Jinja templating is used.
    - `chat_template`: The chat template to be used, or None if no template is specified.
- **Control Flow**:
    - Set the server's Jinja and chat template settings based on the input parameters.
    - Start the server to handle requests.
    - Make a POST request to the '/chat/completions' endpoint with the specified model, max_tokens, and messages.
    - Assert that the response status code is 200, indicating a successful request.
    - Verify that the completion ID in the response body contains 'cmpl'.
    - Check that the system fingerprint in the response body starts with 'b'.
    - Ensure the model in the response matches the input model or the server's default model alias if the input model is None.
    - Validate that the number of prompt and completion tokens in the response match the expected values.
    - Extract the first choice from the response and verify its role is 'assistant'.
    - Use a regular expression to check that the assistant's message content matches the expected pattern.
    - Confirm that the finish reason in the response matches the expected finish reason.
- **Output**: The function does not return a value; it uses assertions to validate the server's response against expected outcomes.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_chat\_completion\_stream<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.test_chat_completion_stream}} -->
The `test_chat_completion_stream` function tests the streaming chat completion functionality of a server by sending requests and validating the responses against expected parameters.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `system_prompt`: A string representing the system's initial message or prompt.
    - `user_prompt`: A string representing the user's message or prompt.
    - `max_tokens`: An integer specifying the maximum number of tokens for the completion.
    - `re_content`: A regular expression pattern to match against the content of the completion.
    - `n_prompt`: An integer representing the expected number of prompt tokens used.
    - `n_predicted`: An integer representing the expected number of completion tokens predicted.
    - `finish_reason`: A string indicating the expected reason for the completion to finish, such as 'length'.
- **Control Flow**:
    - The server's model alias is set to None and the server is started.
    - A streaming request is made to the server with the specified parameters, including system and user prompts, and max tokens.
    - An empty string `content` and a variable `last_cmpl_id` are initialized to track the completion content and ID.
    - The response is iterated over, and for the first item, it checks that the role is 'assistant' and content is None.
    - For subsequent items, it ensures no role is present in the delta and checks the system fingerprint and model name.
    - The completion ID is verified to be consistent across all events in the stream.
    - If the finish reason is 'stop' or 'length', it checks the prompt and completion token counts, absence of content in delta, and matches the content against the regex pattern.
    - If the finish reason is None, it appends the delta content to the `content` string.
- **Output**: The function does not return any value; it asserts various conditions to validate the streaming chat completion process.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_stream_request`](../utils.py.driver.md#ServerProcessmake_stream_request)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_chat\_completion\_with\_openai\_library<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.test_chat_completion_with_openai_library}} -->
The function tests the chat completion feature of the OpenAI library by sending a request to a mock server and verifying the response.
- **Inputs**: None
- **Control Flow**:
    - The function starts by accessing a global server instance and starting it.
    - An OpenAI client is instantiated with a dummy API key and a base URL pointing to the mock server.
    - A chat completion request is made using the client with specified parameters such as model, messages, max_tokens, seed, and temperature.
    - The response is checked to ensure the system fingerprint is not None and starts with 'b'.
    - The function asserts that the finish reason for the first choice is 'length'.
    - It verifies that the content of the message in the first choice is not None.
    - A regex match is performed on the message content to ensure it matches the pattern '(Suddenly)+'
- **Output**: The function does not return any value; it performs assertions to validate the response from the chat completion request.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_chat\_template<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.test_chat_template}} -->
The function `test_chat_template` tests the server's chat completion endpoint with a specific chat template and verifies the response for expected verbose output.
- **Inputs**: None
- **Control Flow**:
    - Sets the global `server`'s `chat_template` to 'llama3' and enables debug mode to include verbose output in the response.
    - Starts the server to prepare it for handling requests.
    - Makes a POST request to the server's '/chat/completions' endpoint with a payload containing a system and user message, and a maximum token limit of 8.
    - Asserts that the response status code is 200, indicating a successful request.
    - Checks that the response body contains a '__verbose' key, ensuring verbose output is present.
    - Verifies that the '__verbose' prompt in the response body matches the expected formatted string.
- **Output**: The function does not return any value; it performs assertions to validate the server's response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_apply\_chat\_template<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.test_apply_chat_template}} -->
The function `test_apply_chat_template` tests the application of a chat template to a set of messages and verifies the response format.
- **Inputs**: None
- **Control Flow**:
    - Sets the global `server`'s `chat_template` to 'command-r'.
    - Starts the server using `server.start()`.
    - Makes a POST request to the '/apply-template' endpoint with a payload containing system and user messages.
    - Asserts that the response status code is 200, indicating success.
    - Checks that the response body contains a 'prompt' key.
    - Verifies that the 'prompt' in the response body matches the expected formatted string.
- **Output**: The function does not return any value but asserts the correctness of the server's response to applying a chat template.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_completion\_with\_response\_format<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.test_completion_with_response_format}} -->
The `test_completion_with_response_format` function tests the server's response to various response formats and checks if the response content matches expected patterns or handles errors appropriately.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `response_format`: A dictionary specifying the expected response format for the server's response.
    - `n_predicted`: An integer representing the maximum number of tokens expected in the response.
    - `re_content`: A string or None, representing the regex pattern to match against the response content or None if an error is expected.
- **Control Flow**:
    - The server is started using `server.start()`.
    - A POST request is made to the `/chat/completions` endpoint with the specified `response_format`, `n_predicted`, and a predefined message structure.
    - If `re_content` is not None, the function asserts that the response status code is 200 and that the response content matches the regex pattern `re_content`.
    - If `re_content` is None, the function asserts that the response status code is not 200 and that the response body contains an error.
- **Output**: The function does not return any value but uses assertions to validate the server's response against expected outcomes.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_completion\_with\_json\_schema<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.test_completion_with_json_schema}} -->
The function `test_completion_with_json_schema` tests the server's ability to handle chat completions with a specified JSON schema and verifies the response content against a regular expression.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `jinja`: A boolean indicating whether Jinja templating is enabled on the server.
    - `json_schema`: A dictionary representing the JSON schema to be used in the request.
    - `n_predicted`: An integer specifying the maximum number of tokens to predict in the response.
    - `re_content`: A string containing a regular expression to match against the content of the response message.
- **Control Flow**:
    - The function is decorated with `@pytest.mark.parametrize` to run the test with different sets of input parameters.
    - The server's Jinja setting is configured based on the `jinja` parameter.
    - The server is started using `server.start()`.
    - A POST request is made to the `/chat/completions` endpoint with the specified `json_schema`, `max_tokens`, and message content.
    - The response status code is asserted to be 200, indicating a successful request.
    - The content of the first choice in the response is matched against the `re_content` regular expression to verify correctness.
- **Output**: The function does not return any value; it raises an assertion error if the response does not meet the expected conditions.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_completion\_with\_grammar<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.test_completion_with_grammar}} -->
The function `test_completion_with_grammar` tests the server's ability to generate chat completions using a specified grammar and checks if the output matches a given regular expression pattern.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `jinja`: A boolean indicating whether the server should use Jinja templates.
    - `grammar`: A string representing the grammar to be used for generating chat completions.
    - `n_predicted`: An integer specifying the maximum number of tokens to predict in the completion.
    - `re_content`: A string representing the regular expression pattern that the completion content should match.
- **Control Flow**:
    - Set the server's Jinja configuration based on the `jinja` parameter.
    - Start the server to handle requests.
    - Make a POST request to the server's `/chat/completions` endpoint with the specified `max_tokens`, `messages`, and `grammar`.
    - Check if the server's response status code is 200, indicating a successful request.
    - Extract the first choice from the response body and verify that its content matches the `re_content` regular expression pattern.
- **Output**: The function does not return any value but asserts that the server's response is successful and matches the expected pattern.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_invalid\_chat\_completion\_req<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.test_invalid_chat_completion_req}} -->
The function `test_invalid_chat_completion_req` tests the server's response to invalid chat completion requests by sending various malformed message formats and asserting that the server returns an error status code and an error message in the response body.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `messages`: A parameterized input representing different invalid message formats to be tested against the server's chat completion endpoint.
- **Control Flow**:
    - The function is decorated with `@pytest.mark.parametrize` to run the test multiple times with different `messages` inputs.
    - The server is started using `server.start()` before making any requests.
    - A POST request is made to the `/chat/completions` endpoint with the `messages` data.
    - The response status code is checked to ensure it is either 400 or 500, indicating a client or server error.
    - The response body is checked to ensure it contains an 'error' key, indicating an error message is present.
- **Output**: The function does not return any value but asserts that the server responds with an error status code and an error message in the response body for invalid message inputs.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_chat\_completion\_with\_timings\_per\_token<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.test_chat_completion_with_timings_per_token}} -->
The function `test_chat_completion_with_timings_per_token` tests the server's ability to handle chat completions with token-level timing information in a streaming context.
- **Inputs**: None
- **Control Flow**:
    - The function starts by initializing the global `server` and starting it.
    - A streaming request is made to the server with specific parameters including `max_tokens`, `messages`, `stream`, and `timings_per_token`.
    - The response is iterated over, and for the first item, it checks that the content is `None`, the role is `assistant`, and that there are no timings present.
    - For subsequent items, it asserts that the role is not present, timings are included, and specific timing metrics (`prompt_per_second`, `predicted_per_second`, `predicted_n`) are present and valid.
    - It ensures that the `predicted_n` value does not exceed 10.
- **Output**: The function does not return any value; it uses assertions to validate the expected behavior of the server's response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_stream_request`](../utils.py.driver.md#ServerProcessmake_stream_request)


---
### test\_logprobs<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.test_logprobs}} -->
The `test_logprobs` function tests the log probability outputs of a chat completion request using the OpenAI API.
- **Inputs**: None
- **Control Flow**:
    - Starts the global server instance.
    - Creates an OpenAI client with a dummy API key and connects to the server.
    - Sends a chat completion request with specific parameters including model, temperature, messages, max_tokens, logprobs, and top_logprobs.
    - Stores the output text from the response.
    - Asserts that the logprobs and their content are not None.
    - Iterates over each token in the logprobs content, appending the token to an aggregated text string.
    - Asserts that each token's log probability is less than or equal to 0.0, and that the token's bytes and top_logprobs are not None and have a length greater than 0.
    - Asserts that the aggregated text matches the output text.
- **Output**: The function does not return any value but performs assertions to validate the log probability outputs of the chat completion response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)


---
### test\_logprobs\_stream<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_chat_completion.test_logprobs_stream}} -->
The `test_logprobs_stream` function tests the streaming log probabilities feature of the OpenAI API's chat completion endpoint.
- **Inputs**: None
- **Control Flow**:
    - Starts the global server instance.
    - Creates an OpenAI client with a dummy API key and connects to the server.
    - Sends a chat completion request with specific parameters including streaming and log probabilities enabled.
    - Iterates over the streaming response data.
    - For the first response, checks that the content is None and the role is 'assistant'.
    - For subsequent responses, checks that the role is None and processes the content if available.
    - Aggregates the content from log probabilities and checks various assertions on the log probability data.
    - Asserts that the aggregated text matches the output text.
- **Output**: The function does not return any value; it performs assertions to validate the behavior of the streaming log probabilities feature.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)


