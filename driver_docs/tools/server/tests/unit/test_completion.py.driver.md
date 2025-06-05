# Purpose
This Python file is a comprehensive test suite designed to validate the functionality of a server that handles text completion requests, likely for a language model. The code utilizes the `pytest` framework to define a series of test cases that ensure the server's completion endpoint behaves as expected under various conditions. The tests cover a wide range of scenarios, including standard and streaming completions, consistency of results with the same seed, variability with different seeds, handling of batch sizes, and the server's ability to process requests in parallel slots. Additionally, the tests verify the server's integration with the OpenAI library, ensuring that it can handle requests formatted for OpenAI's API.

The file is structured to test both the server's core functionality and its edge cases, such as handling long prompts, caching behavior, and response field customization. It also includes tests for probabilistic outputs, ensuring that the server can return token probabilities and handle post-sampling probabilities correctly. The use of parameterized tests allows for efficient testing of multiple input variations, and the inclusion of a test to check the server's response to canceled requests ensures robustness in real-world usage scenarios. Overall, this file serves as a critical component in maintaining the reliability and accuracy of the server's text completion capabilities.
# Imports and Dependencies

---
- `pytest`
- `requests`
- `time`
- `openai.OpenAI`
- `utils.*`


# Global Variables

---
### server
- **Type**: `ServerPreset`
- **Description**: The `server` variable is an instance of the `ServerPreset` class, specifically initialized with the `tinyllama2` preset. This suggests that it is configured with a specific set of parameters or settings defined by the `tinyllama2` preset, which is likely tailored for a particular use case or performance profile.
- **Use**: This variable is used to manage and execute server operations, such as starting the server and handling requests for text completion tasks.


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.create_server}} -->
The `create_server` function is a pytest fixture that initializes a global server instance using the `ServerPreset.tinyllama2()` configuration.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture` with `scope="module"` and `autouse=True`, meaning it will automatically run once per module before any tests are executed.
    - It sets the global variable `server` to an instance of `ServerPreset.tinyllama2()`.
- **Output**: The function does not return any value; it sets up a global server instance for use in tests.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama2`](../utils.py.driver.md#ServerPresettinyllama2)


---
### test\_completion<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_completion}} -->
The `test_completion` function tests the server's completion endpoint by sending requests with various parameters and validating the response against expected values.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `prompt`: A string representing the initial text prompt for the completion request.
    - `n_predict`: An integer specifying the number of tokens to predict in the completion.
    - `re_content`: A regular expression pattern to match against the content of the response.
    - `n_prompt`: An integer representing the expected number of tokens in the prompt as returned in the response timings.
    - `n_predicted`: An integer representing the expected number of predicted tokens as returned in the response timings.
    - `truncated`: A boolean indicating whether the response content is expected to be truncated.
    - `return_tokens`: A boolean indicating whether the response should include token information.
- **Control Flow**:
    - The server is started using `server.start()`.
    - A POST request is made to the `/completion` endpoint with the provided `prompt`, `n_predict`, and `return_tokens` parameters.
    - The response status code is asserted to be 200, indicating a successful request.
    - The response body is checked to ensure the `prompt_n`, `predicted_n`, and `truncated` fields match the expected values (`n_prompt`, `n_predicted`, and `truncated`).
    - The type of `has_new_line` in the response body is asserted to be a boolean.
    - The `content` of the response is matched against the `re_content` regular expression pattern.
    - If `return_tokens` is True, the response is checked to ensure it contains a non-empty list of tokens, all of which are integers.
    - If `return_tokens` is False, the response is checked to ensure the `tokens` list is empty.
- **Output**: The function does not return any value; it uses assertions to validate the response from the server.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_completion\_stream<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_completion_stream}} -->
The `test_completion_stream` function tests the streaming completion functionality of a server by sending a prompt and verifying the response against expected values.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `prompt`: A string representing the initial text prompt to be sent to the server for completion.
    - `n_predict`: An integer specifying the number of tokens to predict in the completion.
    - `re_content`: A regular expression string used to validate the content of the completion.
    - `n_prompt`: An integer representing the expected number of tokens in the prompt.
    - `n_predicted`: An integer representing the expected number of predicted tokens.
    - `truncated`: A boolean indicating whether the completion is expected to be truncated.
- **Control Flow**:
    - The server is started using `server.start()`.
    - A streaming request is made to the server with the given prompt and `n_predict` value, and the response is iterated over.
    - For each data item in the response, it checks if the 'stop' key is present and is a boolean.
    - If 'stop' is true, it asserts various conditions on the data, such as the number of prompt and predicted tokens, whether the response was truncated, and the presence of generation settings.
    - It also checks that the `n_predict` in generation settings matches the minimum of the input `n_predict` and the server's `n_predict`, and that the server's seed is used.
    - The content is validated against the provided regular expression using [`match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex).
    - If 'stop' is false, it asserts that there are tokens present and that they are all integers, then appends the content to a cumulative string.
- **Output**: The function does not return any value; it uses assertions to validate the server's response against expected conditions.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_stream_request`](../utils.py.driver.md#ServerProcessmake_stream_request)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_completion\_stream\_vs\_non\_stream<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_completion_stream_vs_non_stream}} -->
The function `test_completion_stream_vs_non_stream` tests the equivalence of content generated by streaming and non-streaming completion requests to a server.
- **Inputs**: None
- **Control Flow**:
    - The function starts by initializing the global `server` and starting it with `server.start()`.
    - It makes a streaming request to the server using `server.make_stream_request` with a specific prompt and `n_predict` value, storing the result in `res_stream`.
    - It makes a non-streaming request to the server using `server.make_request` with the same prompt and `n_predict` value, storing the result in `res_non_stream`.
    - An empty string `content_stream` is initialized to accumulate content from the streaming response.
    - A loop iterates over `res_stream`, appending each piece of content to `content_stream`.
    - An assertion checks that the accumulated `content_stream` is equal to the content in `res_non_stream.body['content']`.
- **Output**: The function does not return any value but asserts that the content from the streaming and non-streaming requests are identical.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_stream_request`](../utils.py.driver.md#ServerProcessmake_stream_request)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_completion\_with\_openai\_library<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_completion_with_openai_library}} -->
The function `test_completion_with_openai_library` tests the OpenAI library's ability to generate text completions using a mock server.
- **Inputs**: None
- **Control Flow**:
    - The function starts by declaring the `server` as a global variable and starting it.
    - An `OpenAI` client is instantiated with a dummy API key and a base URL pointing to the mock server.
    - A text completion request is made using the `client.completions.create` method with specified parameters such as model, prompt, and max_tokens.
    - Assertions are made to verify that the response contains a non-null system fingerprint starting with 'b', the finish reason is 'length', the text is not null, and the text matches a specified regex pattern.
- **Output**: The function does not return any value; it performs assertions to validate the behavior of the OpenAI library's completion functionality.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_completion\_stream\_with\_openai\_library<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_completion_stream_with_openai_library}} -->
The function tests the streaming completion feature of the OpenAI library by generating text based on a given prompt and verifying the output against a regex pattern.
- **Inputs**: None
- **Control Flow**:
    - The global server is started to handle requests.
    - An OpenAI client is instantiated with a dummy API key and a base URL pointing to the server.
    - A streaming completion request is made using the OpenAI client with a specified model, prompt, and maximum tokens.
    - An empty string 'output_text' is initialized to accumulate the text from the streaming response.
    - The function iterates over the streaming response data, extracting the first choice from each data chunk.
    - If the finish reason of the choice is None, it asserts that the choice text is not None and appends the text to 'output_text'.
    - Finally, the accumulated 'output_text' is checked against a regex pattern to ensure it matches the expected content.
- **Output**: The function does not return any value but asserts that the generated text matches a specified regex pattern.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_completion\_stream\_with\_openai\_library\_stops<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_completion_stream_with_openai_library_stops}} -->
The function tests the OpenAI library's ability to stream text completions with specific stop sequences and verifies the output against a regex pattern.
- **Decorators**: `@pytest.mark.slow`
- **Inputs**: None
- **Control Flow**:
    - The function sets the global server's model repository and file to specific values.
    - The server is started using the `server.start()` method.
    - An OpenAI client is instantiated with a dummy API key and a base URL pointing to the server.
    - A completion request is made using the OpenAI client with a specific prompt, stop sequences, and streaming enabled.
    - The function iterates over the streamed response data, appending text to `output_text` if the finish reason is None.
    - The final output text is asserted against a regex pattern to ensure it matches the expected format.
- **Output**: The function does not return any value but asserts that the output text matches a specific regex pattern, raising an assertion error if it does not.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.match_regex`](../utils.py.driver.md#cpp/tools/server/tests/utilsmatch_regex)


---
### test\_consistent\_result\_same\_seed<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_consistent_result_same_seed}} -->
The function `test_consistent_result_same_seed` tests that the server consistently returns the same result for the same seed value across multiple requests.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `n_slots`: The number of slots to be set on the server, which is used to parameterize the test.
- **Control Flow**:
    - The function sets the global server's `n_slots` attribute to the provided `n_slots` parameter.
    - The server is started using `server.start()`.
    - A loop runs four times, making a POST request to the server's `/completion` endpoint with a fixed prompt, seed, temperature, and cache_prompt setting.
    - The response from each request is compared to the previous response to ensure the content is consistent when the seed is the same.
    - The last response is stored in `last_res` for comparison in the next iteration.
- **Output**: The function does not return any value; it asserts that the content of the server's response is consistent across multiple requests with the same seed.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_different\_result\_different\_seed<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_different_result_different_seed}} -->
The function `test_different_result_different_seed` tests that different random seeds produce different results in a server's completion request.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `n_slots`: The number of slots to be set on the server, which can be either 1 or 2.
- **Control Flow**:
    - The server's number of slots is set to the input parameter `n_slots`.
    - The server is started using `server.start()`.
    - A loop iterates over a range of seeds from 0 to 3.
    - For each seed, a POST request is made to the server's `/completion` endpoint with a fixed prompt, the current seed, a temperature of 1.0, and `cache_prompt` set to False.
    - The response content is compared to the last response content to ensure they are different, asserting that different seeds produce different results.
    - The last response is updated to the current response for the next iteration.
- **Output**: The function does not return any value; it asserts that different seeds produce different completion results.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_consistent\_result\_different\_batch\_size<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_consistent_result_different_batch_size}} -->
The function `test_consistent_result_different_batch_size` tests if the server returns consistent results for different batch sizes when generating text completions with a fixed seed and temperature.
- **Decorators**: `@pytest.mark.parametrize`, `@pytest.mark.parametrize`
- **Inputs**:
    - `n_batch`: An integer representing the batch size to be tested.
    - `temperature`: A float representing the temperature setting for the text generation.
- **Control Flow**:
    - The function is decorated with `@pytest.mark.parametrize` to run the test with different values of `n_batch` and `temperature`.
    - The server's batch size is set to the current `n_batch` value, and the server is started.
    - A loop runs four times to make requests to the server for text completion with a fixed prompt, seed, and temperature.
    - For each request, the response content is compared to the previous response to ensure consistency.
    - If the content of the current response does not match the last response, an assertion error is raised.
- **Output**: The function does not return any value but asserts that the content of the server's response is consistent across multiple requests with the same parameters.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_cache\_vs\_nocache\_prompt<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_cache_vs_nocache_prompt}} -->
The `test_cache_vs_nocache_prompt` function tests whether the content generated by a server request with caching enabled matches the content generated without caching.
- **Decorators**: `@pytest.mark.skip`
- **Inputs**: None
- **Control Flow**:
    - The function starts the global server instance.
    - It makes a POST request to the '/completion' endpoint with caching enabled and stores the response.
    - It makes another POST request to the '/completion' endpoint with caching disabled and stores the response.
    - It asserts that the content of the response with caching is equal to the content of the response without caching.
- **Output**: The function does not return any value; it asserts the equality of the content in the two responses.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_nocache\_long\_input\_prompt<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_nocache_long_input_prompt}} -->
The function `test_nocache_long_input_prompt` tests the server's ability to handle a long input prompt without caching.
- **Inputs**: None
- **Control Flow**:
    - The function starts the server using `server.start()`.
    - It makes a POST request to the `/completion` endpoint with a long prompt, a seed value, a temperature setting, and caching disabled.
    - The function asserts that the response status code is 200, indicating a successful request.
- **Output**: The function does not return any value; it asserts the success of the request by checking the response status code.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_completion\_with\_tokens\_input<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_completion_with_tokens_input}} -->
The function `test_completion_with_tokens_input` tests the server's ability to handle completion requests using tokenized input, mixed input types, and sequences of tokens and strings.
- **Inputs**: None
- **Control Flow**:
    - The server's temperature is set to 0.0 and the server is started.
    - A prompt string is defined and tokenized by making a POST request to the '/tokenize' endpoint with the prompt string and a flag to add special tokens.
    - The response is checked to ensure a successful status code and the tokens are extracted from the response body.
    - A single completion request is made using the tokens, and the response is checked for a successful status code and that the content is a string.
    - A batch completion request is made using a list of the same tokens twice, and the response is checked for a successful status code, that the response body is a list, and that both elements in the list have the same content.
    - A mixed input completion request is made using a list containing both the tokens and the original prompt string, and the response is checked similarly to the batch completion request.
    - A mixed sequence completion request is made using a list of integers, the prompt string, and more integers, and the response is checked for a successful status code and that the content is a string.
- **Output**: The function does not return any value; it uses assertions to validate the server's responses.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_completion\_parallel\_slots<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_completion_parallel_slots}} -->
The `test_completion_parallel_slots` function tests the server's ability to handle multiple completion requests in parallel across different numbers of slots.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `n_slots`: The number of slots available on the server for processing requests.
    - `n_requests`: The number of completion requests to be made to the server.
- **Control Flow**:
    - Set the server's number of slots to `n_slots` and start the server with a temperature of 0.0.
    - Define a list of prompt and regex pairs to be used for the requests.
    - Define a helper function `check_slots_status` to verify if all slots are busy based on the number of requests and slots.
    - Create a list of tasks, each task being a request to the server with a prompt from the predefined list, and append the `check_slots_status` function as a task.
    - Execute the tasks in parallel using [`parallel_function_calls`](../utils.py.driver.md#cpp/tools/server/tests/utilsparallel_function_calls).
    - Iterate over the results of the requests to assert that each response has a status code of 200, the content is a string, and its length is greater than 10.
- **Output**: The function does not return any value but asserts the correctness of server responses and slot utilization during parallel request processing.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)
    - [`llama.cpp/tools/server/tests/utils.parallel_function_calls`](../utils.py.driver.md#cpp/tools/server/tests/utilsparallel_function_calls)


---
### test\_completion\_response\_fields<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_completion_response_fields}} -->
The `test_completion_response_fields` function tests the server's response to a completion request, ensuring it returns the expected fields and content based on the provided parameters.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `prompt`: A string representing the initial text prompt for the completion request.
    - `n_predict`: An integer specifying the number of tokens to predict in the completion.
    - `response_fields`: A list of strings indicating which fields should be included in the response.
- **Control Flow**:
    - The function starts the server using `server.start()`.
    - It makes a POST request to the `/completion` endpoint with the provided `prompt`, `n_predict`, and `response_fields`.
    - It asserts that the response status code is 200, indicating a successful request.
    - It checks that the 'content' field is present in the response body and that it contains data.
    - If `response_fields` is not empty, it verifies that the response includes the specified fields and that their values match the expected values.
    - If `response_fields` is empty, it asserts that the response body contains data and includes the 'generation_settings' field.
- **Output**: The function does not return any value; it uses assertions to validate the server's response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_n\_probs<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_n_probs}} -->
The `test_n_probs` function tests the server's ability to handle a completion request with specific probability settings and validates the response structure and content.
- **Inputs**: None
- **Control Flow**:
    - The function starts by initializing the global `server` and starting it.
    - A POST request is made to the server's `/completion` endpoint with specific data including a prompt, number of probabilities (`n_probs`), temperature, and number of predictions (`n_predict`).
    - The response status code is asserted to be 200, indicating a successful request.
    - The function checks that the response body contains a `completion_probabilities` field with a length equal to `n_predict`.
    - For each token in `completion_probabilities`, several assertions are made to ensure the presence and validity of fields like `id`, `token`, `logprob`, `bytes`, and `top_logprobs`.
    - Each `top_logprobs` entry is further validated for the presence and correctness of fields `id`, `token`, `logprob`, and `bytes`.
- **Output**: The function does not return any value; it uses assertions to validate the response from the server.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_n\_probs\_stream<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_n_probs_stream}} -->
The `test_n_probs_stream` function tests the streaming response of a server for completion probabilities, ensuring the response structure and data types are correct.
- **Inputs**: None
- **Control Flow**:
    - The function starts by initializing the global `server` and calling `server.start()` to begin the server process.
    - It makes a streaming request to the server using `server.make_stream_request` with specific parameters including `prompt`, `n_probs`, `temperature`, `n_predict`, and `stream`.
    - The function iterates over the streaming response `res`.
    - For each data item in the response, it checks if the `stop` field is `False`.
    - If `stop` is `False`, it asserts the presence of `completion_probabilities` in the data and verifies its length is 1.
    - For each token in `completion_probabilities`, it asserts the presence and validity of fields `id`, `token`, `logprob`, `bytes`, and `top_logprobs`.
    - It further iterates over `top_logprobs` to assert the presence and validity of fields `id`, `token`, `logprob`, and `bytes`.
- **Output**: The function does not return any value; it uses assertions to validate the response data structure and content.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_stream_request`](../utils.py.driver.md#ServerProcessmake_stream_request)


---
### test\_n\_probs\_post\_sampling<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_n_probs_post_sampling}} -->
The function `test_n_probs_post_sampling` tests the server's ability to handle a POST request for text completion with post-sampling probabilities and verifies the structure and content of the response.
- **Inputs**: None
- **Control Flow**:
    - The function starts by initializing the global `server` and starting it.
    - A POST request is made to the server at the `/completion` endpoint with specific data parameters including `prompt`, `n_probs`, `temperature`, `n_predict`, and `post_sampling_probs`.
    - The response status code is asserted to be 200, indicating a successful request.
    - The function checks that the response body contains a `completion_probabilities` field with a length of 5.
    - For each token in `completion_probabilities`, several assertions are made to verify the presence and validity of fields such as `id`, `token`, `prob`, `bytes`, and `top_probs`.
    - Each `top_probs` entry is further validated for fields `id`, `token`, `prob`, and `bytes`.
    - Finally, it asserts that at least one probability in `top_probs` is exactly 1.0, ensuring the model outputs a token with certainty.
- **Output**: The function does not return any value; it raises an assertion error if any of the checks fail.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_cancel\_request<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_completion.test_cancel_request}} -->
The `test_cancel_request` function tests the server's ability to handle a request cancellation and ensure that the server slot is freed up after the cancellation.
- **Inputs**: None
- **Control Flow**:
    - Set global server parameters: `n_ctx` to 4096, `n_predict` to -1, `n_slots` to 1, and `server_slots` to True.
    - Start the server using `server.start()`.
    - Attempt to make a POST request to the `/completion` endpoint with a prompt, setting a very short timeout of 0.1 seconds.
    - Catch the `requests.exceptions.ReadTimeout` exception, which is expected due to the short timeout, and pass it.
    - Wait for 1 second to ensure the server has time to process the cancellation and free up the slot.
    - Make a GET request to the `/slots` endpoint to verify that the slot is no longer processing any requests.
- **Output**: The function does not return any value but asserts that the server slot is free after the request cancellation.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


