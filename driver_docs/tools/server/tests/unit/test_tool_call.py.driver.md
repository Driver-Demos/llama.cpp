# Purpose
This Python script is a comprehensive test suite designed to validate the functionality of a server process that handles chat completions and tool calls. It uses the `pytest` framework to define a series of test cases that assess the server's ability to process requests involving different tools, such as a Python code interpreter, a weather information retriever, and a basic test function. The script is structured to automatically set up a server instance using a pytest fixture, configure it with various parameters, and execute a range of tests that simulate different scenarios, including tool-assisted completions and direct chat interactions.

The script defines several test functions, each decorated with `pytest.mark.parametrize` to run the tests with multiple configurations, such as different models, tools, and streaming modes. The tests are organized to cover both fast and slow execution paths, with some tests marked as slow to indicate they require more time to complete. The script also includes utility functions to perform specific test actions, such as sending requests to the server and verifying the responses. The primary focus of the tests is to ensure that the server correctly handles tool calls, generates expected outputs, and adheres to specified constraints, such as response timeouts and content formats.
# Imports and Dependencies

---
- `pytest`
- `pathlib.Path`
- `sys`
- `utils.*`
- `enum.Enum`


# Global Variables

---
### path
- **Type**: `Path`
- **Description**: The `path` variable is a `Path` object that represents the directory path of the grandparent directory of the current file. It is resolved to an absolute path using the `resolve()` method and the `parents[1]` attribute is used to navigate two levels up from the current file's directory.
- **Use**: This variable is used to insert the grandparent directory into the system path (`sys.path`) to ensure that modules from that directory can be imported.


---
### server
- **Type**: `ServerProcess`
- **Description**: The `server` variable is a global instance of the `ServerProcess` class, which is used to manage and interact with a server process in the testing environment. It is configured with various settings such as model alias, server port, and number of slots, and is used throughout the test functions to perform server-related operations.
- **Use**: This variable is used to configure and control the server process for running tests, including starting the server, making requests, and handling server responses.


---
### TIMEOUT\_SERVER\_START
- **Type**: `int`
- **Description**: `TIMEOUT_SERVER_START` is a global integer variable that represents the timeout duration for starting a server, set to 15 minutes (15*60 seconds).
- **Use**: This variable is used to specify the maximum time allowed for the server to start before a timeout occurs.


---
### TIMEOUT\_HTTP\_REQUEST
- **Type**: `int`
- **Description**: `TIMEOUT_HTTP_REQUEST` is a global variable set to the integer value 60. It represents a timeout duration for HTTP requests in seconds.
- **Use**: This variable is used to specify the maximum time to wait for an HTTP request to complete before timing out.


---
### TEST\_TOOL
- **Type**: `dict`
- **Description**: `TEST_TOOL` is a dictionary that defines a tool with a type of 'function'. It specifies a function named 'test' with a parameter 'success' that is a boolean and must be true. The dictionary structure is used to define the function's metadata and expected parameters.
- **Use**: This variable is used to define a tool for testing purposes, ensuring that the 'success' parameter is always true when the function is called.


---
### PYTHON\_TOOL
- **Type**: `dict`
- **Description**: The `PYTHON_TOOL` variable is a dictionary that defines a function tool for executing Python code within an IPython interpreter. It specifies the function's name as 'python', a description of its purpose, and the parameters it requires, which include a single 'code' parameter of type string.
- **Use**: This variable is used to define a tool that can execute Python code snippets and return the result after a specified timeout.


---
### WEATHER\_TOOL
- **Type**: `dict`
- **Description**: The `WEATHER_TOOL` variable is a dictionary that defines a function tool for retrieving the current weather in a specified location. It includes the function name `get_current_weather`, a description of its purpose, and a parameter `location` which is required and must be a string representing the city and country/state.
- **Use**: This variable is used to specify the structure and requirements for a tool that can be called to obtain current weather information for a given location.


# Classes

---
### CompletionMode<!-- {{#class:llama.cpp/tools/server/tests/unit/test_tool_call.CompletionMode}} -->
- **Members**:
    - `NORMAL`: Represents the normal completion mode.
    - `STREAMED`: Represents the streamed completion mode.
- **Description**: The `CompletionMode` class is an enumeration that defines two modes of operation for completions: `NORMAL` and `STREAMED`. These modes are used to specify how a completion process should be handled, either in a standard manner or as a continuous stream.
- **Inherits From**:
    - `Enum`


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.create_server}} -->
The `create_server` function is a pytest fixture that initializes a global server instance with specific configurations for testing purposes.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture(autouse=True)`, which means it will automatically run before each test function in the module.
    - A global variable `server` is initialized using `ServerPreset.tinyllama2()`.
    - The `server` object is configured with a model alias, server port, and number of slots.
- **Output**: The function does not return any value; it sets up a global server instance for use in tests.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama2`](../utils.py.driver.md#ServerPresettinyllama2)


---
### do\_test\_completion\_with\_required\_tool\_tiny<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.do_test_completion_with_required_tool_tiny}} -->
The function `do_test_completion_with_required_tool_tiny` tests the server's ability to handle a chat completion request with a required tool and validates the tool call and its arguments.
- **Inputs**:
    - `server`: An instance of `ServerProcess` used to make requests to the server.
    - `tool`: A dictionary representing the tool to be used in the request, containing details like type and function name.
    - `argument_key`: A string or None, representing the key expected in the tool's function arguments.
    - `n_predict`: An integer specifying the maximum number of tokens to predict in the response.
    - `kwargs`: Additional keyword arguments to be included in the request data.
- **Control Flow**:
    - Make a POST request to the server at the endpoint '/v1/chat/completions' with specified data including messages, tool choice, and tools.
    - Extract the first choice from the response body and retrieve the tool calls from the message.
    - Assert that there is exactly one tool call in the message.
    - Assert that the message content is either None or an empty string.
    - Determine the expected function name based on the tool type and assert it matches the function name in the tool call.
    - Retrieve the function arguments from the tool call and assert it is a string.
    - If `argument_key` is not None, parse the arguments as JSON and assert that `argument_key` is present in the arguments.
- **Output**: The function does not return any value; it raises assertions if any validation fails.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_any_request`](../utils.py.driver.md#ServerProcessmake_any_request)


---
### test\_completion\_with\_required\_tool\_tiny\_fast<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.test_completion_with_required_tool_tiny_fast}} -->
The function `test_completion_with_required_tool_tiny_fast` tests the completion functionality of a server using various templates and tools, ensuring the correct tool is called with the expected arguments.
- **Decorators**: `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `template_name`: A string representing the name of the template to be used for the test.
    - `tool`: A dictionary representing the tool to be used in the test, which includes its type and function details.
    - `argument_key`: A string or None, representing the key expected in the tool's arguments.
    - `stream`: An instance of `CompletionMode` enum, indicating whether the completion should be normal or streamed.
- **Control Flow**:
    - The function is decorated with `@pytest.mark.parametrize` to run the test with different combinations of `stream`, `template_name`, `tool`, and `argument_key` parameters.
    - The server's configuration is set up with a specific template file and prediction settings.
    - The server is started with a timeout defined by `TIMEOUT_SERVER_START`.
    - The function [`do_test_completion_with_required_tool_tiny`](#cpp/tools/server/tests/unit/test_tool_calldo_test_completion_with_required_tool_tiny) is called with the server and test parameters to perform the actual test.
    - The test checks if the server correctly calls the specified tool with the expected arguments.
- **Output**: The function does not return any value; it performs assertions to validate the test conditions.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/unit/test_tool_call.do_test_completion_with_required_tool_tiny`](#cpp/tools/server/tests/unit/test_tool_calldo_test_completion_with_required_tool_tiny)


---
### test\_completion\_with\_required\_tool\_tiny\_slow<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.test_completion_with_required_tool_tiny_slow}} -->
The function `test_completion_with_required_tool_tiny_slow` tests the completion of chat templates using specified tools and parameters in a slow execution mode.
- **Decorators**: `@pytest.mark.slow`, `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `template_name`: A string representing the name of the chat template to be used.
    - `tool`: A dictionary representing the tool to be used in the test, which includes its type and function details.
    - `argument_key`: A string or None, representing the key expected in the tool's arguments.
    - `stream`: An instance of `CompletionMode` enum indicating whether the completion should be normal or streamed.
- **Control Flow**:
    - The function is decorated with `@pytest.mark.slow` to indicate it is a slow test and uses `@pytest.mark.parametrize` to run the test with different combinations of `stream`, `template_name`, `tool`, and `argument_key`.
    - It sets a global `server` variable and initializes `n_predict` to 512.
    - The server's Jinja template processing is enabled, and the number of predictions is set to `n_predict`.
    - The server's chat template file is set based on the `template_name` provided.
    - The server is started with a timeout defined by `TIMEOUT_SERVER_START`.
    - The function [`do_test_completion_with_required_tool_tiny`](#cpp/tools/server/tests/unit/test_tool_calldo_test_completion_with_required_tool_tiny) is called with the server, tool, argument_key, n_predict, and a boolean indicating if the stream mode is `STREAMED`.
- **Output**: The function does not return any value; it performs assertions to validate the behavior of the server's completion process with the specified tools and parameters.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/unit/test_tool_call.do_test_completion_with_required_tool_tiny`](#cpp/tools/server/tests/unit/test_tool_calldo_test_completion_with_required_tool_tiny)


---
### test\_completion\_with\_required\_tool\_real\_model<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.test_completion_with_required_tool_real_model}} -->
The function `test_completion_with_required_tool_real_model` tests the server's ability to handle chat completions with required tool calls using various model configurations and parameters.
- **Decorators**: `@pytest.mark.slow`, `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `tool`: A dictionary representing the tool to be used in the test, which includes its type and function details.
    - `argument_key`: A string or None, representing the key expected in the tool's function arguments.
    - `hf_repo`: A string representing the Hugging Face repository identifier for the model to be used.
    - `template_override`: A string, tuple, or None, representing the template override for the chat template file or content.
    - `stream`: An instance of `CompletionMode` enum indicating whether the completion should be streamed or not.
- **Control Flow**:
    - The function sets up the server with specific configurations such as `n_predict`, `n_ctx`, and `model_hf_repo`.
    - It checks if `template_override` is a tuple or string to set the server's chat template file or content accordingly.
    - The server is started with a specified timeout.
    - A POST request is made to the server to initiate a chat completion with specified parameters including `max_tokens`, `messages`, `tool_choice`, `tools`, `parallel_tool_calls`, `stream`, `temperature`, `top_k`, and `top_p`.
    - The response is checked to ensure there is exactly one tool call in the message.
    - The function name in the tool call is verified to match the expected function name based on the tool type.
    - The function arguments are checked to ensure they are a string and contain the `argument_key` if it is not None.
- **Output**: The function does not return any value but asserts various conditions to validate the server's handling of tool calls in chat completions.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_any_request`](../utils.py.driver.md#ServerProcessmake_any_request)


---
### do\_test\_completion\_without\_tool\_call<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.do_test_completion_without_tool_call}} -->
The function `do_test_completion_without_tool_call` tests a server's ability to generate chat completions without invoking any tool calls.
- **Inputs**:
    - `server`: An instance of `ServerProcess` used to make HTTP requests.
    - `n_predict`: An integer specifying the maximum number of tokens to predict in the completion.
    - `tools`: A list of dictionaries representing tools that could be used in the completion, or an empty list if no tools are to be used.
    - `tool_choice`: A string or None indicating the tool choice strategy, or 'none' if no tool should be used.
    - `**kwargs`: Additional keyword arguments to be passed to the server request.
- **Control Flow**:
    - The function constructs a request body with parameters including `max_tokens`, `messages`, `tools`, and `tool_choice`.
    - It sends a POST request to the server at the endpoint `/v1/chat/completions` with the constructed data and a timeout setting.
    - The response body is parsed to extract the first choice from the `choices` list.
    - An assertion checks that the `tool_calls` field in the choice's message is `None`, ensuring no tool was called.
- **Output**: The function does not return a value but asserts that no tool calls are present in the server's response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_any_request`](../utils.py.driver.md#ServerProcessmake_any_request)


---
### test\_completion\_without\_tool\_call\_fast<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.test_completion_without_tool_call_fast}} -->
The `test_completion_without_tool_call_fast` function tests the server's ability to handle chat completions without invoking any tools, using various templates and configurations.
- **Decorators**: `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `template_name`: A string representing the name of the template to be used for the chat completion.
    - `n_predict`: An integer specifying the number of tokens to predict.
    - `tools`: A list of dictionaries representing the tools available for the chat completion, which can be empty.
    - `tool_choice`: A string or None indicating the tool choice strategy, which can be 'none' or None.
    - `stream`: An instance of the CompletionMode enum indicating whether the completion should be streamed or not.
- **Control Flow**:
    - The function sets the global `server`'s `n_predict` attribute to the provided `n_predict` value.
    - It enables Jinja templating on the server by setting `server.jinja` to True.
    - The server's chat template file is set based on the `template_name` provided.
    - The server is started with a timeout defined by `TIMEOUT_SERVER_START`.
    - The function [`do_test_completion_without_tool_call`](#cpp/tools/server/tests/unit/test_tool_calldo_test_completion_without_tool_call) is called with the server and the provided parameters, including a boolean indicating if streaming is enabled.
- **Output**: The function does not return any value; it performs assertions to validate the server's behavior during the test.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/unit/test_tool_call.do_test_completion_without_tool_call`](#cpp/tools/server/tests/unit/test_tool_calldo_test_completion_without_tool_call)


---
### test\_completion\_without\_tool\_call\_slow<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.test_completion_without_tool_call_slow}} -->
The function `test_completion_without_tool_call_slow` tests the server's ability to handle chat completions without invoking any tools, using various templates and configurations.
- **Decorators**: `@pytest.mark.slow`, `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `template_name`: A string representing the name of the template file to be used for chat completions.
    - `n_predict`: An integer specifying the maximum number of tokens to predict during the chat completion.
    - `tools`: A list of dictionaries representing tools that could be used during the chat completion, though they are not expected to be called in this test.
    - `tool_choice`: A string or None, indicating the choice of tool to be used, if any.
    - `stream`: An instance of `CompletionMode` enum, indicating whether the completion should be streamed or not.
- **Control Flow**:
    - The function is decorated with `@pytest.mark.slow` to indicate it is a slow test and uses `@pytest.mark.parametrize` to run the test with different combinations of parameters.
    - The server's `n_predict` attribute is set to the provided `n_predict` value.
    - The server's `jinja` attribute is set to `True`, enabling Jinja templating.
    - The server's `chat_template_file` is set to the path of the template file corresponding to `template_name`.
    - The server is started with a timeout defined by `TIMEOUT_SERVER_START`.
    - The function [`do_test_completion_without_tool_call`](#cpp/tools/server/tests/unit/test_tool_calldo_test_completion_without_tool_call) is called with the server and the provided parameters, including a boolean indicating if streaming is enabled.
- **Output**: The function does not return any value; it performs assertions to validate the server's behavior during the test.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/unit/test_tool_call.do_test_completion_without_tool_call`](#cpp/tools/server/tests/unit/test_tool_calldo_test_completion_without_tool_call)


---
### test\_weather<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.test_weather}} -->
The `test_weather` function tests the weather tool functionality by configuring a server with various model and template settings, then verifying the tool call for weather information in Istanbul.
- **Decorators**: `@pytest.mark.slow`, `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `hf_repo`: A string representing the Hugging Face repository identifier for the model to be tested.
    - `template_override`: An optional string or tuple that specifies a template override for the chat template, which can be a string or a tuple of strings.
    - `stream`: An instance of the `CompletionMode` enum indicating whether the completion should be normal or streamed.
- **Control Flow**:
    - Set global server configurations such as `jinja`, `n_ctx`, and `n_predict`.
    - Assign the `hf_repo` to `server.model_hf_repo` and set `server.model_hf_file` to `None`.
    - Check if `template_override` is a tuple; if so, construct the `server.chat_template_file` path and assert its existence.
    - If `template_override` is a string, assign it to `server.chat_template`.
    - Start the server with a specified timeout using `server.start()`.
    - Call [`do_test_weather`](#cpp/tools/server/tests/unit/test_tool_calldo_test_weather) with the server and streaming configuration to perform the actual test.
- **Output**: The function does not return a value but asserts the correct behavior of the weather tool call, ensuring it is invoked with the expected parameters.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/unit/test_tool_call.do_test_weather`](#cpp/tools/server/tests/unit/test_tool_calldo_test_weather)


---
### do\_test\_weather<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.do_test_weather}} -->
The `do_test_weather` function tests the weather tool call functionality by sending a request to a server and verifying the response for expected tool call details and arguments.
- **Inputs**:
    - `server`: An instance of `ServerProcess` used to make requests to the server.
    - `**kwargs`: Additional keyword arguments that can be passed to the request.
- **Control Flow**:
    - A POST request is made to the server at the endpoint '/v1/chat/completions' with a predefined message asking for the weather in Istanbul and the WEATHER_TOOL included in the request data.
    - The response body is parsed to extract the first choice from the 'choices' list.
    - The function checks if there is exactly one tool call in the 'tool_calls' field of the message in the choice.
    - The function verifies that the tool call's function name matches the expected weather tool function name.
    - The function checks that the 'location' argument is present in the tool call's function arguments and that it is a string.
    - The function asserts that the location matches the expected format for Istanbul.
- **Output**: The function does not return any value; it raises assertions if any checks fail, indicating a test failure.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_any_request`](../utils.py.driver.md#ServerProcessmake_any_request)


---
### test\_calc\_result<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.test_calc_result}} -->
The `test_calc_result` function is a parameterized test that configures a server with specific model and template settings, then verifies the server's response to a calculation request, ensuring it matches expected results.
- **Decorators**: `@pytest.mark.slow`, `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `result_override`: A string or None, used to override the expected result pattern for validation.
    - `n_predict`: An integer specifying the number of tokens to predict.
    - `hf_repo`: A string representing the Hugging Face repository identifier for the model to be used.
    - `template_override`: A string, tuple, or None, specifying a template override for the server's chat template configuration.
    - `stream`: An instance of CompletionMode, indicating whether the completion should be streamed or not.
- **Control Flow**:
    - The function sets global server configurations such as `jinja`, `n_ctx`, `n_predict`, and `model_hf_repo`.
    - It checks if `template_override` is a tuple or a string to configure the server's chat template file or template directly.
    - The server is started with a specified timeout using `server.start(timeout_seconds=TIMEOUT_SERVER_START)`.
    - The function [`do_test_calc_result`](#cpp/tools/server/tests/unit/test_tool_calldo_test_calc_result) is called with the server and other parameters to perform the actual test.
- **Output**: The function does not return a value; it performs assertions to validate the server's response against expected results.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/unit/test_tool_call.do_test_calc_result`](#cpp/tools/server/tests/unit/test_tool_calldo_test_calc_result)


---
### do\_test\_calc\_result<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.do_test_calc_result}} -->
The `do_test_calc_result` function tests the server's ability to handle a calculation request and validate the response content against expected results.
- **Inputs**:
    - `server`: An instance of ServerProcess used to make HTTP requests.
    - `result_override`: A string or None, used to override the expected result pattern for validation.
    - `n_predict`: An integer specifying the maximum number of tokens to predict.
    - `kwargs`: Additional keyword arguments passed to the server request.
- **Control Flow**:
    - Make a POST request to the server with a specific data payload including a calculation request for the sine of 30 degrees.
    - Retrieve the first choice from the server's response body.
    - Check if there are any tool calls in the response message and assert that there should be none.
    - Retrieve the content from the response message and assert that it is not None.
    - If result_override is provided, assert that the content matches the result_override pattern.
    - If result_override is not provided, assert that the content matches a default pattern indicating the expected result.
- **Output**: The function does not return any value; it raises assertions if the server response does not meet expectations.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_any_request`](../utils.py.driver.md#ServerProcessmake_any_request)


---
### test\_thoughts<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.test_thoughts}} -->
The `test_thoughts` function is a parameterized test that verifies the behavior of a server's chat completion endpoint, ensuring it returns expected content and reasoning content based on various input configurations.
- **Decorators**: `@pytest.mark.slow`, `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `n_predict`: An integer specifying the maximum number of tokens to predict in the response.
    - `reasoning_format`: A string or None indicating the format of reasoning, either 'deepseek' or 'none'.
    - `expect_content`: A string or None representing the expected regex pattern for the content of the response.
    - `expect_reasoning_content`: A string or None representing the expected regex pattern for the reasoning content of the response.
    - `hf_repo`: A string specifying the Hugging Face repository to use for the model.
    - `template_override`: A string, tuple, or None indicating a template override for the chat template file.
    - `stream`: An instance of CompletionMode indicating whether the response should be streamed or normal.
- **Control Flow**:
    - Set global server's reasoning_format, jinja, n_ctx, n_predict, model_hf_repo, and model_hf_file attributes based on input parameters.
    - Check if template_override is a tuple or string and set server's chat_template_file or chat_template accordingly.
    - Start the server with a specified timeout.
    - Make a POST request to the server's chat completions endpoint with specified data, including max_tokens, messages, and stream settings.
    - Extract the first choice from the response body and assert that no tool calls are present in the message.
    - Check if expect_content is None and assert the content is None or empty, otherwise match the content against expect_content regex.
    - Check if expect_reasoning_content is None and assert the reasoning content is None, otherwise match the reasoning content against expect_reasoning_content regex.
- **Output**: The function does not return any value but asserts the correctness of the server's response content and reasoning content.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_any_request`](../utils.py.driver.md#ServerProcessmake_any_request)


---
### test\_hello\_world<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.test_hello_world}} -->
The `test_hello_world` function is a parameterized test that configures a server to test the 'hello world' functionality using various model configurations and template overrides.
- **Decorators**: `@pytest.mark.slow`, `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `hf_repo`: A string representing the Hugging Face repository identifier for the model to be tested.
    - `template_override`: An optional string or tuple that specifies a template override for the chat template, which can be a string or a tuple of strings.
    - `stream`: An instance of the `CompletionMode` enum indicating whether the completion should be streamed or not.
- **Control Flow**:
    - The function sets global server configurations such as `jinja`, `n_ctx`, `n_predict`, and `model_hf_repo` based on the input parameters.
    - If `template_override` is a tuple, it constructs a file path for the chat template and checks if the file exists, asserting if it does not.
    - If `template_override` is a string, it assigns it directly to `server.chat_template`.
    - The server is started with a specified timeout using `server.start(timeout_seconds=TIMEOUT_SERVER_START)`.
    - The function [`do_test_hello_world`](#cpp/tools/server/tests/unit/test_tool_calldo_test_hello_world) is called with the server and streaming configuration to perform the actual test.
- **Output**: The function does not return any value; it performs assertions to validate the test conditions.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/unit/test_tool_call.do_test_hello_world`](#cpp/tools/server/tests/unit/test_tool_calldo_test_hello_world)


---
### do\_test\_hello\_world<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tool_call.do_test_hello_world}} -->
The `do_test_hello_world` function tests if a server process can correctly generate a Python 'Hello, World!' script using a tool call.
- **Inputs**:
    - `server`: An instance of `ServerProcess` that represents the server to which the request is made.
    - `kwargs`: Additional keyword arguments that can be passed to customize the request.
- **Control Flow**:
    - The function sends a POST request to the server at the endpoint '/v1/chat/completions' with a specific data payload.
    - The payload includes a system message indicating the agent's role and a user message requesting a Python 'Hello, World!' script.
    - The request specifies the use of the `PYTHON_TOOL` and includes any additional keyword arguments provided.
    - The response body is parsed to extract the first choice from the 'choices' list.
    - The function checks if there is exactly one tool call in the choice's message.
    - It asserts that the tool call's function name matches the expected name from `PYTHON_TOOL`.
    - The function retrieves and parses the 'arguments' from the tool call to ensure it contains a 'code' key.
    - It verifies that the 'code' is a string and matches a regular expression pattern for a 'Hello, World!' print statement.
- **Output**: The function does not return a value; it uses assertions to validate the expected behavior and will raise an AssertionError if any check fails.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_any_request`](../utils.py.driver.md#ServerProcessmake_any_request)


