# Purpose
This Python file is a test suite using the `pytest` framework to validate the functionality of a server, specifically focusing on its tokenization and detokenization capabilities. The code defines a module-scoped fixture to initialize a server instance using a preset configuration (`tinyllama2`) and includes three test functions. Each test function starts the server and sends HTTP POST requests to endpoints for tokenizing and detokenizing text, verifying the server's responses for correctness. The tests check various aspects of the server's tokenization process, such as handling special tokens and processing text with Unicode and emoji characters. This code provides narrow functionality, focusing solely on testing the server's text processing features.
# Imports and Dependencies

---
- `pytest`
- `utils.*`


# Global Variables

---
### server
- **Type**: `ServerPreset`
- **Description**: The `server` variable is an instance of the `ServerPreset` class, specifically initialized with the `tinyllama2` preset. This suggests that it is configured with a specific set of parameters or settings defined by the `tinyllama2` preset, which is likely tailored for certain server operations or testing scenarios.
- **Use**: The `server` variable is used to perform server operations such as starting the server and making HTTP requests for tokenization and detokenization in the test functions.


# Functions

---
### create\_server<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tokenize.create_server}} -->
The `create_server` function is a pytest fixture that initializes a global server instance using the `ServerPreset.tinyllama2()` configuration for use in module-scoped tests.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is decorated with `@pytest.fixture` with `scope="module"` and `autouse=True`, meaning it runs automatically once per module before any tests are executed.
    - The function sets a global variable `server` to an instance of `ServerPreset.tinyllama2()`.
- **Output**: The function does not return any value; it sets a global variable `server`.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerPreset.tinyllama2`](../utils.py.driver.md#ServerPresettinyllama2)


---
### test\_tokenize\_detokenize<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tokenize.test_tokenize_detokenize}} -->
The `test_tokenize_detokenize` function tests the server's ability to tokenize and detokenize a given string, ensuring the operations are performed correctly and the original content is preserved.
- **Inputs**: None
- **Control Flow**:
    - The function starts by initializing the global `server` and starting it.
    - It defines a string `content` with the value 'What is the capital of France ?'.
    - A POST request is made to the `/tokenize` endpoint with the `content` as data, and the response is stored in `res_tok`.
    - The function asserts that the status code of `res_tok` is 200, indicating a successful request.
    - It checks that the number of tokens in `res_tok.body['tokens']` is greater than 5, ensuring the content was tokenized into multiple tokens.
    - A POST request is made to the `/detokenize` endpoint with the tokens from `res_tok` as data, and the response is stored in `res_detok`.
    - The function asserts that the status code of `res_detok` is 200, indicating a successful request.
    - It verifies that the detokenized content in `res_detok.body['content']` matches the original `content` after stripping whitespace.
- **Output**: The function does not return any value; it uses assertions to validate the correctness of the tokenization and detokenization processes.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_tokenize\_with\_bos<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tokenize.test_tokenize_with_bos}} -->
The function `test_tokenize_with_bos` tests the server's ability to tokenize a given string with a special beginning-of-sequence token.
- **Inputs**: None
- **Control Flow**:
    - The function starts by accessing the global `server` object and starting the server.
    - It defines a string `content` with the sentence 'What is the capital of France ?'.
    - A variable `bosId` is set to 1, representing the expected beginning-of-sequence token ID.
    - A POST request is made to the server's `/tokenize` endpoint with the `content` and a flag `add_special` set to True, indicating that special tokens should be added.
    - The function asserts that the response status code is 200, indicating a successful request.
    - It further asserts that the first token in the response body is equal to `bosId`, verifying that the beginning-of-sequence token is correctly added.
- **Output**: The function does not return any value; it uses assertions to validate the expected behavior of the server's tokenization process.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


---
### test\_tokenize\_with\_pieces<!-- {{#callable:llama.cpp/tools/server/tests/unit/test_tokenize.test_tokenize_with_pieces}} -->
The function `test_tokenize_with_pieces` tests the server's ability to tokenize a string into tokens with associated pieces, ensuring each token has a valid ID and non-empty piece.
- **Inputs**: None
- **Control Flow**:
    - The function starts the server using `server.start()`.
    - It defines a string `content` containing text with unicode and emoji characters.
    - A POST request is made to the `/tokenize` endpoint with the `content` and `with_pieces` set to `True`.
    - The function asserts that the response status code is 200, indicating a successful request.
    - It iterates over each token in the response body, asserting that each token contains an 'id' field with a value greater than 0 and a 'piece' field with a non-empty value.
- **Output**: The function does not return any value; it uses assertions to validate the server's response.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.start`](../utils.py.driver.md#ServerProcessstart)
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.make_request`](../utils.py.driver.md#ServerProcessmake_request)


