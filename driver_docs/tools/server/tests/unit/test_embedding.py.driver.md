# Purpose
This Python file is a test suite designed to validate the functionality of a server that provides text embedding services. The code uses the `pytest` framework to define a series of test cases that ensure the server's embedding API behaves as expected under various conditions. The tests cover a range of scenarios, including single and multiple input embeddings, handling of different input types, and the server's response to invalid or edge-case inputs. The tests also verify the server's ability to handle different pooling strategies and encoding formats, such as base64, and ensure that the embeddings are correctly normalized or not, depending on the configuration.

The file imports necessary modules and sets up a server instance using a predefined server preset. It includes tests that interact with the server's API endpoints, specifically the `/v1/embeddings` endpoint, to send POST requests with different payloads. The tests assert the correctness of the server's responses, including status codes, the presence and structure of embedding data, and the usage statistics of tokens. Additionally, the file includes tests that integrate with the OpenAI library to ensure compatibility and correct behavior when using the library's client to interact with the server. Overall, this file serves as a comprehensive validation tool for developers to ensure the robustness and reliability of the embedding service.
# Imports and Dependencies

---
- `base64`
- `struct`
- `pytest`
- `openai.OpenAI`
- `utils.*`


# Global Variables

---
### server
- **Type**: `ServerPreset`
- **Description**: The `server` variable is an instance of a server preset, specifically initialized with the `bert_bge_small` configuration. This configuration likely sets up a server with specific parameters and capabilities related to the BERT model for generating embeddings.
- **Use**: This variable is used to handle requests for generating embeddings in various test cases, simulating a server environment for testing purposes.


---
### EPSILON
- **Type**: `float`
- **Description**: EPSILON is a small floating-point number set to 0.001. It is used as a threshold for numerical comparisons to account for floating-point precision errors.
- **Use**: EPSILON is used in assertions to check if the difference between calculated values and expected values is within an acceptable range, ensuring numerical stability in tests.


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.create_server}} -->
The `create_server` function is a pytest fixture that initializes a global server instance using a predefined server preset.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture` with `scope="module"` and `autouse=True`, meaning it will automatically run once per module before any tests are executed.
    - It sets the global variable `server` to an instance of `ServerPreset.bert_bge_small()`.
- **Output**: The function does not return any value; it sets a global variable `server`.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.bert_bge_small`](../utils.py.driver.md#ServerPresetbert_bge_small)


---
### test\_embedding\_single<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_embedding_single}} -->
The `test_embedding_single` function tests the server's ability to generate a single embedding from a given input string and verifies the normalization of the resulting embedding vector.
- **Inputs**: None
- **Control Flow**:
    - The function sets the global server's pooling attribute to 'last'.
    - The server is started using `server.start()`.
    - A POST request is made to the server at the endpoint '/v1/embeddings' with a specific input string.
    - The response status code is asserted to be 200, indicating a successful request.
    - The function checks that the response contains exactly one data item with an 'embedding' key.
    - It verifies that the length of the embedding is greater than 1.
    - The function ensures that the embedding vector is normalized by checking that the sum of the squares of its components is approximately 1, within a small epsilon margin.
- **Output**: The function does not return any value; it raises an assertion error if any of the checks fail.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_embedding\_multiple<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_embedding_multiple}} -->
The function `test_embedding_multiple` tests the server's ability to handle multiple input strings for generating embeddings and verifies the response structure and content.
- **Inputs**: None
- **Control Flow**:
    - Sets the global server's pooling attribute to 'last'.
    - Starts the server.
    - Makes a POST request to the '/v1/embeddings' endpoint with multiple input strings.
    - Asserts that the response status code is 200, indicating success.
    - Asserts that the response contains four data entries, one for each input string.
    - Iterates over each data entry in the response to check that each contains an 'embedding' key and that the embedding is non-empty.
- **Output**: The function does not return any value; it uses assertions to validate the server's response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_embedding\_multiple\_with\_fa<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_embedding_multiple_with_fa}} -->
The function `test_embedding_multiple_with_fa` tests the embedding generation for multiple inputs using a server preset with a specific configuration that includes a feature for handling context sizes that are multiples of 256.
- **Inputs**: None
- **Control Flow**:
    - Initialize a server using the `ServerPreset.bert_bge_small_with_fa()` configuration.
    - Set the server's pooling method to 'last'.
    - Start the server to begin accepting requests.
    - Make a POST request to the server's `/v1/embeddings` endpoint with a data payload containing four strings, each repeated a number of times to create different context sizes.
    - Check that the server responds with a status code of 200, indicating success.
    - Verify that the response contains four data entries, each with an 'embedding' key.
    - Ensure that each 'embedding' in the response has a length greater than 1.
- **Output**: The function does not return any value; it asserts the correctness of the server's response to the embedding request.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.bert_bge_small_with_fa`](../utils.py.driver.md#ServerPresetbert_bge_small_with_fa)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_embedding\_mixed\_input<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_embedding_mixed_input}} -->
The `test_embedding_mixed_input` function tests the server's ability to handle various types of input for generating embeddings, ensuring correct response structure and content based on whether the input is a single or multiple prompts.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `input`: The input data to be sent to the server for embedding, which can be a string, a list of numbers, or a list of mixed types.
    - `is_multi_prompt`: A boolean indicating whether the input is expected to be treated as multiple prompts (True) or a single prompt (False).
- **Control Flow**:
    - The server is started using `server.start()`.
    - A POST request is made to the `/v1/embeddings` endpoint with the input data.
    - The response status code is asserted to be 200, indicating a successful request.
    - The response body is checked to ensure it contains the expected 'embedding' data.
    - If `is_multi_prompt` is True, the function asserts that the number of embeddings matches the number of input prompts and that each embedding is valid.
    - If `is_multi_prompt` is False, the function asserts that a single embedding is returned and is valid.
- **Output**: The function does not return any value; it uses assertions to validate the server's response to the input data.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_embedding\_pooling\_none<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_embedding_pooling_none}} -->
The function `test_embedding_pooling_none` tests the server's ability to handle embedding requests with 'none' pooling and verifies the non-normalization of the resulting embedding vectors.
- **Inputs**: None
- **Control Flow**:
    - Set the global server's pooling attribute to 'none'.
    - Start the server.
    - Make a POST request to the '/embeddings' endpoint with the input 'hello hello hello'.
    - Assert that the response status code is 200, indicating success.
    - Check that the response body contains an 'embedding' key and that the length of the embedding is 5, accounting for 3 text tokens and 2 special tokens.
    - Iterate over each vector in the embedding and assert that the sum of squares of its components deviates from 1 by more than EPSILON, confirming non-normalization.
- **Output**: The function does not return any value; it uses assertions to validate the expected behavior of the server's embedding response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_embedding\_pooling\_none\_oai<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_embedding_pooling_none_oai}} -->
The function `test_embedding_pooling_none_oai` tests the server's response when an unsupported pooling type 'none' is used for the '/v1/embeddings' endpoint.
- **Inputs**: None
- **Control Flow**:
    - Set the global server's pooling attribute to 'none'.
    - Start the server.
    - Make a POST request to the '/v1/embeddings' endpoint with input data 'hello hello hello'.
    - Assert that the response status code is 400, indicating a bad request.
    - Assert that the response body contains an 'error' key, indicating an error message is present.
- **Output**: The function does not return any value; it asserts conditions to validate the server's behavior.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_embedding\_openai\_library\_single<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_embedding_openai_library_single}} -->
The function `test_embedding_openai_library_single` tests the creation of a single text embedding using the OpenAI library with a mock server.
- **Inputs**: None
- **Control Flow**:
    - Set the global server's pooling attribute to 'last'.
    - Start the server.
    - Create an OpenAI client with a dummy API key and the server's base URL.
    - Use the client to create an embedding for the input text 'I believe the meaning of life is' using the model 'text-embedding-3-small'.
    - Assert that the response contains exactly one data item.
    - Assert that the embedding in the response data has more than one element.
- **Output**: The function does not return any value; it performs assertions to validate the behavior of the embedding creation process.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)


---
### test\_embedding\_openai\_library\_multiple<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_embedding_openai_library_multiple}} -->
The function tests the OpenAI library's ability to create embeddings for multiple input strings using a specified model.
- **Inputs**: None
- **Control Flow**:
    - Sets the global server's pooling attribute to 'last'.
    - Starts the server.
    - Initializes an OpenAI client with a dummy API key and the server's base URL.
    - Calls the client's embeddings.create method with a specified model and a list of input strings.
    - Asserts that the response contains four data entries, one for each input string.
    - Iterates over each data entry to assert that the embedding length is greater than one.
- **Output**: The function does not return any value; it performs assertions to validate the behavior of the OpenAI library's embedding creation.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)


---
### test\_embedding\_error\_prompt\_too\_long<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_embedding_error_prompt_too_long}} -->
The function `test_embedding_error_prompt_too_long` tests the server's response to an excessively long input prompt for embeddings, ensuring it returns an error.
- **Inputs**: None
- **Control Flow**:
    - Set the global server's pooling attribute to 'last'.
    - Start the server.
    - Make a POST request to the '/v1/embeddings' endpoint with a very long input string (repeated 512 times).
    - Assert that the response status code is not 200, indicating an error occurred.
    - Assert that the error message in the response body contains the phrase 'too large'.
- **Output**: The function does not return any value; it asserts conditions to validate the server's error handling for long input prompts.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_same\_prompt\_give\_same\_result<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_same_prompt_give_same_result}} -->
The function `test_same_prompt_give_same_result` verifies that identical input prompts produce identical embedding results from the server.
- **Inputs**: None
- **Control Flow**:
    - Set the server's pooling method to 'last' and start the server.
    - Make a POST request to the server's '/v1/embeddings' endpoint with five identical input prompts.
    - Assert that the response status code is 200, indicating a successful request.
    - Assert that the length of the response data is 5, matching the number of input prompts.
    - Iterate over the embeddings in the response data, comparing each to the first embedding to ensure they are identical within a small margin of error defined by EPSILON.
- **Output**: The function does not return any value; it uses assertions to validate that the server's response meets the expected conditions.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_embedding\_usage\_single<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_embedding_usage_single}} -->
The `test_embedding_usage_single` function tests the usage statistics of a single embedding request to ensure the prompt tokens match the expected number of tokens and total tokens.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**:
    - `content`: A string input representing the content to be embedded.
    - `n_tokens`: An integer representing the expected number of tokens in the content.
- **Control Flow**:
    - The function starts the global server instance.
    - It makes a POST request to the '/v1/embeddings' endpoint with the input content.
    - It asserts that the response status code is 200, indicating a successful request.
    - It checks that the number of prompt tokens in the response matches the total tokens.
    - It verifies that the number of prompt tokens equals the expected number of tokens, `n_tokens`.
- **Output**: The function does not return any value; it uses assertions to validate the expected behavior of the embedding service.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_embedding\_usage\_multiple<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_embedding_usage_multiple}} -->
The function `test_embedding_usage_multiple` tests the usage statistics of embedding requests with multiple identical inputs to ensure correct token counting.
- **Decorators**: `@pytest.mark.parametrize`
- **Inputs**: None
- **Control Flow**:
    - The function starts the global server instance.
    - It makes a POST request to the '/v1/embeddings' endpoint with two identical input strings.
    - It asserts that the response status code is 200, indicating a successful request.
    - It checks that the number of prompt tokens equals the total tokens in the response body.
    - It verifies that the number of prompt tokens is twice the number of tokens in a single input string, which is 9.
- **Output**: The function does not return any value; it uses assertions to validate the expected behavior of the server's response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_embedding\_openai\_library\_base64<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_embedding.test_embedding_openai_library_base64}} -->
The function tests the OpenAI library's ability to generate and verify base64-encoded embeddings.
- **Inputs**: None
- **Control Flow**:
    - The server is started to handle requests.
    - A test input string is defined for generating embeddings.
    - A POST request is made to the server to get the embedding in the default format, and the result is stored.
    - Another POST request is made to get the embedding in base64 format.
    - Assertions are made to ensure the response status is 200 and the data structure is correct.
    - The base64-encoded embedding is decoded and converted back to a float array.
    - Assertions verify that the decoded float array matches the original embedding vector within a small error margin.
- **Output**: The function does not return any value; it uses assertions to validate the behavior of the embedding generation and encoding process.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


