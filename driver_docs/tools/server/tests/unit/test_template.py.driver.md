# Purpose
This Python script is a test suite designed to validate the functionality of a server process that applies chat templates to user messages. It uses the `pytest` framework to define and execute a series of tests, each of which checks different aspects of the server's behavior when processing chat templates. The script includes three main test functions: [`test_reasoning_budget`](#cpp/tools/server/tests/unit/test_templatetest_reasoning_budget), [`test_date_inside_prompt`](#cpp/tools/server/tests/unit/test_templatetest_date_inside_prompt), and [`test_add_generation_prompt`](#cpp/tools/server/tests/unit/test_templatetest_add_generation_prompt). Each function uses the `pytest.mark.parametrize` decorator to run the tests with various combinations of input parameters, such as different template names, reasoning budgets, and tool configurations. The tests ensure that the server correctly applies the specified templates and generates the expected output prompts, including verifying the presence of specific strings or date formats in the generated prompts.

The script is structured to automatically set up a server instance using a fixture, [`create_server`](#cpp/tools/server/tests/unit/test_templatecreate_server), which configures the server with specific settings such as model alias, server port, and the number of slots. The server is then used in each test to make HTTP POST requests to the `/apply-template` endpoint, simulating user interactions. The responses are checked for correct status codes and expected content, ensuring that the server's template application logic functions as intended. This test suite is crucial for maintaining the reliability and correctness of the server's template processing capabilities, particularly in scenarios involving complex template logic and dynamic content generation.
# Imports and Dependencies

---
- `pytest`
- `pathlib.Path`
- `sys`
- `unit.test_tool_call.TEST_TOOL`
- `datetime`
- `utils.*`


# Global Variables

---
### path
- **Type**: `Path`
- **Description**: The `path` variable is a `Path` object that represents the directory path of the grandparent directory of the current file. It is resolved to an absolute path using the `resolve()` method and then the `parents[1]` attribute is used to navigate two levels up from the current file's directory.
- **Use**: This variable is used to insert the grandparent directory into the system path (`sys.path`) to ensure that modules from that directory can be imported.


---
### server
- **Type**: `ServerProcess`
- **Description**: The `server` variable is an instance of the `ServerProcess` class, which is used to manage and interact with a server process in the test environment. It is configured with specific settings such as model alias, server port, and number of slots, and is used to apply templates and make requests during testing.
- **Use**: This variable is used to configure and start a server process for testing purposes, allowing the execution of test cases that involve server interactions.


---
### TIMEOUT\_SERVER\_START
- **Type**: `int`
- **Description**: `TIMEOUT_SERVER_START` is an integer variable that represents the timeout duration for starting a server, set to 15 minutes (15 multiplied by 60 seconds).
- **Use**: This variable is used to specify the maximum time allowed for the server to start before timing out.


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_template.create_server}} -->
The `create_server` function is a pytest fixture that initializes a global server instance with specific configuration settings for testing purposes.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture(autouse=True)`, which means it will automatically run before each test function in the module.
    - A global variable `server` is initialized using `ServerPreset.tinyllama2()`, which presumably sets up a server with a predefined configuration.
    - The `model_alias` attribute of the server is set to 'tinyllama-2'.
    - The `server_port` attribute is set to 8081, indicating the port on which the server will listen.
    - The `n_slots` attribute is set to 1, which likely configures the number of concurrent slots or connections the server can handle.
- **Output**: The function does not return any value; it sets up a global server instance for use in tests.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama2`](../utils.py.driver.md#ServerPresettinyllama2)


---
### test\_reasoning\_budget<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_template.test_reasoning_budget}} -->
The `test_reasoning_budget` function tests the server's response to different reasoning budgets and template names, ensuring the generated prompt ends with the expected string.
- **Decorators**: `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `template_name`: A string representing the name of the template to be used for generating the prompt.
    - `reasoning_budget`: An integer or None indicating the reasoning budget to be set on the server.
    - `expected_end`: A string that the generated prompt is expected to end with.
    - `tools`: A list of dictionaries representing tools to be included in the request, which can be None, an empty list, or a list containing a test tool.
- **Control Flow**:
    - The function is decorated with `@pytest.mark.parametrize` to run multiple test cases with different combinations of `template_name`, `reasoning_budget`, and `expected_end`.
    - The server's `jinja` attribute is set to True, and the `reasoning_budget` is set to the provided value.
    - The server's `chat_template_file` is set based on the `template_name` provided.
    - The server is started with a timeout defined by `TIMEOUT_SERVER_START`.
    - A POST request is made to the server's `/apply-template` endpoint with a message asking 'What is today?' and the specified `tools`.
    - The response status code is checked to ensure it is 200, indicating a successful request.
    - The prompt from the response body is retrieved and checked to ensure it ends with the `expected_end` string.
    - An assertion error is raised if the prompt does not end with the expected string, providing a detailed error message.
- **Output**: The function does not return any value but raises an assertion error if the prompt does not end with the expected string, indicating a test failure.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_date\_inside\_prompt<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_template.test_date_inside_prompt}} -->
The `test_date_inside_prompt` function tests if the current date is correctly included in the prompt generated by a server using different template formats.
- **Decorators**: `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `template_name`: A string representing the name of the template file to be used for generating the prompt.
    - `format`: A string specifying the date format to be used for checking the inclusion of today's date in the prompt.
    - `tools`: A list of dictionaries representing tools to be included in the request, which can be None, an empty list, or a list containing a test tool.
- **Control Flow**:
    - The function is decorated with `@pytest.mark.parametrize` to run the test with different combinations of `template_name`, `format`, and `tools`.
    - The server's Jinja template processing is enabled, and the chat template file is set based on the `template_name` parameter.
    - The server is started with a specified timeout.
    - A POST request is made to the server's `/apply-template` endpoint with a message asking for today's date and the specified tools.
    - The response status code is asserted to be 200, indicating a successful request.
    - The prompt from the response body is extracted.
    - The current date is formatted according to the `format` parameter and checked for inclusion in the prompt.
    - An assertion is made to ensure the formatted date is present in the prompt, raising an error if it is not.
- **Output**: The function does not return any value; it asserts the presence of the formatted current date in the server-generated prompt.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_add\_generation\_prompt<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_template.test_add_generation_prompt}} -->
The function `test_add_generation_prompt` tests whether a generation prompt is correctly added to or excluded from a server-generated prompt based on a boolean flag.
- **Decorators**: `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `template_name`: A string representing the name of the template to be used for generating the prompt.
    - `expected_generation_prompt`: A string representing the expected generation prompt that should be present or absent in the server response.
    - `add_generation_prompt`: A boolean flag indicating whether the generation prompt should be added to the server-generated prompt.
- **Control Flow**:
    - The function is decorated with `@pytest.mark.parametrize` to run tests with different combinations of `add_generation_prompt` and `template_name, expected_generation_prompt` values.
    - The server is configured with Jinja templating and the specified chat template file based on `template_name`.
    - The server is started with a timeout defined by `TIMEOUT_SERVER_START`.
    - A POST request is made to the server's `/apply-template` endpoint with a message and the `add_generation_prompt` flag.
    - The response status code is checked to ensure it is 200, indicating a successful request.
    - The prompt from the response body is extracted.
    - If `add_generation_prompt` is True, the function asserts that `expected_generation_prompt` is present in the prompt.
    - If `add_generation_prompt` is False, the function asserts that `expected_generation_prompt` is not present in the prompt.
- **Output**: The function does not return a value; it uses assertions to validate the presence or absence of the generation prompt in the server response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


