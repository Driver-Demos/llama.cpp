# Purpose
This Python script is designed to perform asynchronous HTTP POST requests to a local server and compute the cosine similarity between the resulting embeddings. The script uses the `asyncio` library to manage asynchronous operations, allowing it to send multiple HTTP requests concurrently to a specified URL endpoint (`http://127.0.0.1:6900/embedding`). The `requests` library is used to handle the HTTP requests, while the `numpy` library is employed for numerical operations, specifically to compute the cosine similarity between the embeddings received from the server.

The script defines an asynchronous function [`requests_post_async`](#cpp/examples/server_embdrequests_post_async) to wrap the synchronous `requests.post` method, enabling it to be used in an asynchronous context. The [`main`](#cpp/examples/server_embdmain) function orchestrates the sending of `n` requests, where `n` is set to 8, and collects the responses. Each response is expected to contain an "embedding" in JSON format, which is then printed and stored in a list. After gathering all embeddings, the script calculates the cosine similarity between each pair of embeddings using `numpy` operations and prints the results. This script is a standalone utility, not intended to be imported as a library, and it does not define any public APIs or external interfaces. Its primary purpose is to demonstrate asynchronous HTTP requests and the computation of cosine similarity between vectors.
# Imports and Dependencies

---
- `asyncio`
- `asyncio.threads`
- `requests`
- `numpy`


# Global Variables

---
### n
- **Type**: `int`
- **Description**: The variable `n` is an integer set to the value 8. It represents the number of asynchronous HTTP POST requests to be made to a specified URL in the `main` function.
- **Use**: `n` is used to determine the number of iterations for creating asynchronous requests and for computing pairwise cosine similarities.


---
### result
- **Type**: `list`
- **Description**: The variable `result` is a global list that is initially empty. It is used to store embeddings obtained from asynchronous HTTP POST requests to a specified model URL. Each embedding is appended to this list as the responses are processed.
- **Use**: This variable is used to store and later access the embeddings for computing cosine similarity between them.


# Functions

---
### requests\_post\_async<!-- {{#callable:llama.cpp/examples/server_embd.requests_post_async}} -->
The `requests_post_async` function asynchronously sends a POST request using the `requests` library in a separate thread.
- **Inputs**:
    - `*args`: Positional arguments to be passed to the `requests.post` function.
    - `**kwargs`: Keyword arguments to be passed to the `requests.post` function.
- **Control Flow**:
    - The function uses `asyncio.threads.to_thread` to run the `requests.post` function in a separate thread, allowing it to be non-blocking in an asynchronous context.
    - The `await` keyword is used to pause the coroutine until the thread completes the POST request.
- **Output**: The function returns the result of the `requests.post` call, which is a `requests.Response` object, wrapped in an awaitable coroutine.


---
### main<!-- {{#callable:llama.cpp/examples/server_embd.main}} -->
The `main` function asynchronously sends multiple POST requests to a specified URL to retrieve embeddings and stores them in a global list.
- **Inputs**: None
- **Control Flow**:
    - Define the URL for the model endpoint as 'http://127.0.0.1:6900'.
    - Use `asyncio.gather` to concurrently send `n` POST requests to the '/embedding' endpoint with a JSON payload containing a repeated character string.
    - For each response received, extract the 'embedding' from the JSON response.
    - Print the last 8 elements of each embedding to the console.
    - Append each embedding to the global `result` list.
- **Output**: The function does not return any value; it prints parts of the embeddings and appends them to a global list `result`.
- **Functions called**:
    - [`llama.cpp/examples/server_embd.requests_post_async`](#cpp/examples/server_embdrequests_post_async)


