# Purpose
This Bash script is designed to facilitate a chat interaction between a user and an artificial intelligence assistant by leveraging an API for natural language processing. It provides a narrow functionality focused on handling user input, formatting prompts, and managing the conversation flow with the AI. The script includes functions for trimming whitespace, formatting prompts, tokenizing input, and managing chat completions by interacting with an external API. It uses `curl` to send requests to the API and processes the responses using Python for real-time output. The script is intended to be executed directly in a terminal, as it continuously reads user input, processes it, and appends the AI's responses to the chat history.
# Imports and Dependencies

---
- `curl`
- `jq`
- `python`


# Global Variables

---
### API\_URL
- **Type**: `string`
- **Description**: `API_URL` is a global variable that holds the URL endpoint for the API server. It defaults to 'http://127.0.0.1:8080' if not set externally, allowing for local testing or development.
- **Use**: This variable is used to construct the URL for API requests in the `tokenize` and `chat_completion` functions.


---
### CHAT
- **Type**: `array`
- **Description**: The `CHAT` variable is an array that stores a sequence of dialogue exchanges between a human user and an artificial intelligence assistant. It is initialized with two strings representing a greeting from the human and a response from the assistant.
- **Use**: This variable is used to maintain the context of the conversation by appending new user inputs and AI responses to the array.


---
### INSTRUCTION
- **Type**: `string`
- **Description**: The `INSTRUCTION` variable is a string that provides a description of a chat interaction between a human and an AI assistant. It specifies that the assistant should provide helpful, detailed, and polite answers to the human's questions.
- **Use**: This variable is used as part of the prompt formatting in the `format_prompt` function to guide the AI's responses during the chat.


---
### N\_KEEP
- **Type**: `integer`
- **Description**: `N_KEEP` is a global variable that stores the number of tokens in the initial system instruction prompt. It is calculated by tokenizing the instruction string and counting the resulting tokens.
- **Use**: This variable is used to specify the number of tokens to retain in the chat completion request payload.


# Functions

---
### trim
The `trim` function removes leading and trailing whitespace from a given string.
- **Inputs**:
    - `$1`: The string from which leading and trailing whitespace should be removed.
- **Control Flow**:
    - Enable extended pattern matching features with `shopt -s extglob`.
    - Remove leading whitespace from the input string using parameter expansion and set the result as the positional parameter.
    - Remove trailing whitespace from the modified string using parameter expansion and print the result.
- **Output**: The function outputs the input string with all leading and trailing whitespace removed.


---
### trim\_trailing
The `trim_trailing` function removes trailing whitespace from a given string.
- **Inputs**:
    - `$1`: The input string from which trailing whitespace is to be removed.
- **Control Flow**:
    - Enable extended pattern matching features with `shopt -s extglob`.
    - Use parameter expansion to remove trailing whitespace from the input string.
    - Print the modified string without trailing whitespace.
- **Output**: The function outputs the input string with trailing whitespace removed.


---
### format\_prompt
The `format_prompt` function formats a chat prompt by appending a new instruction to the last message in a chat array or initializing a new prompt if the chat array is empty.
- **Inputs**:
    - `$1`: A string representing the new instruction to be appended to the chat prompt.
- **Control Flow**:
    - Check if the CHAT array is empty.
    - If CHAT is empty, output a formatted string with the system instruction enclosed in special tags.
    - If CHAT is not empty, calculate the index of the last element in the CHAT array.
    - Output the last message from the CHAT array followed by the new instruction enclosed in special tags.
- **Output**: A formatted string that either contains the system instruction or the last chat message followed by the new instruction.


---
### tokenize
The `tokenize` function sends a POST request to a specified API endpoint to tokenize a given string and returns the list of tokens.
- **Inputs**:
    - `$1`: A string input that represents the content to be tokenized.
- **Control Flow**:
    - The function uses `curl` to send a POST request to the API endpoint `${API_URL}/tokenize`.
    - The request includes a JSON payload with the input string as the value of the `content` key.
    - The response from the API is piped into `jq` to extract and output the list of tokens from the `.tokens[]` array.
- **Output**: A list of tokens extracted from the input string, printed to the standard output.


---
### chat\_completion
The `chat_completion` function generates a response from an AI assistant based on a user's input question and appends the interaction to a chat history.
- **Inputs**:
    - `$1`: The user's input question to which the AI assistant should respond.
- **Control Flow**:
    - The function starts by formatting the prompt using the `format_prompt` function and trimming any trailing spaces.
    - It constructs a JSON payload with the prompt and other parameters like temperature, top_k, top_p, n_keep, n_predict, stop, and stream.
    - A temporary file is created to store the output from a Python script that processes the response from the AI model.
    - A POST request is made to the API endpoint `/completion` with the JSON payload, and the response is read line by line.
    - The Python script extracts the 'content' from each line of the response, appends it to the answer, and writes the final answer to the temporary file.
    - The temporary file is read to get the final answer, which is then trimmed and added to the `CHAT` array along with the user's question.
    - The temporary file is deleted after reading the answer.
- **Output**: The function does not return a value but updates the `CHAT` array with the user's question and the AI's response.


