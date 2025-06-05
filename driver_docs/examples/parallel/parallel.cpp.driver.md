# Purpose
This C++ source code file implements a basic application that simulates a server handling multiple client requests in parallel. The primary functionality of this code is to simulate interactions between users and an AI assistant, where clients submit requests to the server, and these requests are processed concurrently. The code is structured to handle multiple clients, each represented by a `client` struct, which manages the state and data associated with each client request, including input prompts, responses, and timing information.

The code includes several key components: it initializes a server environment using the `llama` library, processes client requests by tokenizing and decoding input prompts, and generates responses using a sampling mechanism. The application is designed to handle a configurable number of parallel client requests, with the ability to load prompts from an external file or use built-in defaults. The code also includes logging functionality to track the progress and performance of the request processing, including metrics such as token processing speed and cache misses. This file is intended to be compiled and executed as a standalone application, as indicated by the presence of a [`main`](#main) function, and it does not define public APIs or external interfaces for use by other software components.
# Imports and Dependencies

---
- `arg.h`
- `common.h`
- `sampling.h`
- `log.h`
- `llama.h`
- `cmath`
- `cstdio`
- `string`
- `vector`
- `ctime`
- `algorithm`


# Global Variables

---
### k\_system
- **Type**: `std::string`
- **Description**: The variable `k_system` is a static global string that contains a predefined transcript of a dialog between a User and an Assistant. The dialog showcases the Assistant's ability to provide helpful, precise, and immediate responses to the User's requests, such as recommending a restaurant or providing information about Richard Feynman.
- **Use**: This variable is used as a system prompt in a simulated server application to initialize or guide interactions between clients and the server.


---
### k\_questions
- **Type**: `std::vector<std::string>`
- **Description**: The `k_questions` variable is a static vector of strings that contains a list of trivia questions. These questions cover a variety of topics, including geography, history, science, and popular culture.
- **Use**: This variable is used to provide a set of predefined questions that can be randomly selected and included in client prompts during the simulation of parallel requests.


---
### k\_answers
- **Type**: `std::vector<std::string>`
- **Description**: `k_answers` is a static global variable that holds a vector of strings, each string being an answer to a corresponding question in the `k_questions` vector. The answers cover a variety of general knowledge topics, such as geography, history, science, and culture.
- **Use**: This variable is used to provide predefined answers to questions, likely in a simulated dialog or quiz application.


---
### k\_prompts
- **Type**: `std::vector<std::string>`
- **Description**: The `k_prompts` variable is a static vector of strings that contains a set of predefined prompts. These prompts are questions or requests that can be used to simulate user input in a dialog with an assistant.
- **Use**: This variable is used to provide random prompts to clients in the simulation, allowing them to generate responses based on these predefined questions or requests.


# Data Structures

---
### client<!-- {{#data_structure:client}} -->
- **Type**: `struct`
- **Members**:
    - `id`: An integer representing the unique identifier for the client.
    - `seq_id`: A llama_seq_id representing the sequence identifier, initialized to -1.
    - `sampled`: A llama_token representing the last sampled token for the client.
    - `t_start_prompt`: A 64-bit integer representing the start time of the prompt.
    - `t_start_gen`: A 64-bit integer representing the start time of the generation.
    - `n_past`: An integer representing the number of past tokens processed.
    - `n_prompt`: An integer representing the number of tokens in the prompt.
    - `n_decoded`: An integer representing the number of tokens decoded.
    - `i_batch`: An integer representing the index of the batch, initialized to -1.
    - `input`: A string representing the input provided by the client.
    - `prompt`: A string representing the full prompt including system and user inputs.
    - `response`: A string representing the response generated for the client.
    - `smpl`: A pointer to a common_sampler structure, used for sampling tokens.
- **Description**: The `client` struct is designed to represent a client in a server simulation, managing the state and data associated with each client's request and response cycle. It includes fields for tracking unique identifiers, sequence and batch indices, timing information, and the input/output data for each client. Additionally, it manages a pointer to a `common_sampler` for token sampling, ensuring that resources are properly freed upon destruction.
- **Member Functions**:
    - [`client::~client`](#clientclient)

**Methods**

---
#### client::\~client<!-- {{#callable:client::~client}} -->
The `~client` destructor function releases resources associated with the `smpl` pointer if it is not null.
- **Inputs**: None
- **Control Flow**:
    - Check if the `smpl` pointer is not null.
    - If `smpl` is not null, call `common_sampler_free(smpl)` to free the associated resources.
- **Output**: The function does not return any value as it is a destructor.
- **Functions called**:
    - [`common_sampler_free`](../../common/sampling.cpp.driver.md#common_sampler_free)
- **See also**: [`client`](#client)  (Data Structure)



# Functions

---
### trim<!-- {{#callable:trim}} -->
The `trim` function removes leading and trailing whitespace from a given string.
- **Inputs**:
    - `str`: A constant reference to a `std::string` from which leading and trailing whitespace will be removed.
- **Control Flow**:
    - Initialize `start` to 0 and `end` to the size of the input string `str`.
    - Iterate from the beginning of the string, incrementing `start` while the current character is a whitespace and `start` is less than `end`.
    - Iterate from the end of the string, decrementing `end` while the character before `end` is a whitespace and `end` is greater than `start`.
    - Return a substring of `str` starting from `start` and of length `end - start`.
- **Output**: A `std::string` that is a copy of the input string `str` with leading and trailing whitespace removed.


---
### print\_date\_time<!-- {{#callable:print_date_time}} -->
The `print_date_time` function logs the current date and time in a specific format to the console.
- **Inputs**: None
- **Control Flow**:
    - Retrieve the current time using `std::time(nullptr)` and store it in `current_time`.
    - Convert `current_time` to local time using `std::localtime` and store the result in `local_time`.
    - Define a character buffer `buffer` of size 80 to hold the formatted date and time string.
    - Use `strftime` to format `local_time` into a string with the format `%Y-%m-%d %H:%M:%S` and store it in `buffer`.
    - Log a newline character using `LOG_INF`.
    - Log the formatted date and time string with a specific color code using `LOG_INF`.
    - Log another newline character using `LOG_INF`.
- **Output**: The function does not return any value; it outputs the formatted date and time to the console using logging functions.


---
### split\_string<!-- {{#callable:split_string}} -->
The `split_string` function splits a given string into a vector of substrings based on a specified delimiter.
- **Inputs**:
    - `input`: A constant reference to a `std::string` that represents the string to be split.
    - `delimiter`: A `char` that specifies the character used to split the input string into tokens.
- **Control Flow**:
    - Initialize an empty vector of strings named `tokens` to store the resulting substrings.
    - Create an `std::istringstream` object named `stream` initialized with the input string to facilitate reading tokens.
    - Declare a `std::string` variable named `token` to temporarily hold each substring extracted from the input string.
    - Use a `while` loop with `std::getline` to read each substring from the `stream` using the specified delimiter, storing each substring in `token`.
    - Within the loop, append each `token` to the `tokens` vector.
    - Return the `tokens` vector containing all the substrings.
- **Output**: A `std::vector<std::string>` containing the substrings of the input string, split by the specified delimiter.


---
### main<!-- {{#callable:main}} -->
The `main` function simulates a server handling multiple client requests in parallel, processing them through a language model and generating responses.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize random seed with a fixed value for reproducibility.
    - Parse command-line arguments into a `common_params` structure and check for successful parsing.
    - Initialize the backend and NUMA settings for the language model.
    - Load the language model and its context using the parsed parameters.
    - Check if external prompts are provided; if not, use built-in defaults.
    - Initialize a vector of `client` objects to simulate multiple clients.
    - Tokenize the system prompt and prepare it for processing.
    - Enter a loop to process client requests, handling token batching and decoding.
    - For each client, construct a prompt, tokenize it, and add it to the batch for processing.
    - Decode the batch of tokens, handling cache misses and adjusting batch size as needed.
    - For each client, sample tokens, generate responses, and log performance metrics.
    - Continue processing until all sequences are completed and the batch is empty.
    - Log final performance metrics and clean up resources before exiting.
- **Output**: The function returns an integer status code, where 0 indicates successful execution and 1 indicates an error occurred during processing.
- **Functions called**:
    - [`common_params_parse`](../../common/arg.cpp.driver.md#common_params_parse)
    - [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_numa_init`](../../src/llama.cpp.driver.md#llama_numa_init)
    - [`split_string`](#split_string)
    - [`common_sampler_init`](../../common/sampling.cpp.driver.md#common_sampler_init)
    - [`common_sampler_reset`](../../common/sampling.cpp.driver.md#common_sampler_reset)
    - [`common_sampler_sample`](../../common/sampling.cpp.driver.md#common_sampler_sample)
    - [`common_sampler_accept`](../../common/sampling.cpp.driver.md#common_sampler_accept)
    - [`print_date_time`](#print_date_time)
    - [`llama_backend_free`](../../src/llama.cpp.driver.md#llama_backend_free)


