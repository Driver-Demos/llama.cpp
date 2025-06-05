# Purpose
This Bash script is designed to facilitate a chat interaction between a user and an artificial intelligence assistant, providing a narrow functionality focused on simulating a conversational exchange. It acts as an executable script that continuously prompts the user for input, processes the input to generate a formatted prompt, and sends it to an API for tokenization and completion. The script uses functions to trim whitespace, format prompts, and handle API requests, leveraging `curl` and `jq` to interact with a local or specified API endpoint. The conversation is maintained in a loop, appending each user query and AI response to a chat history, demonstrating its purpose as a simple, interactive chat client for testing or demonstration purposes.
# Imports and Dependencies

---
- `curl`
- `jq`


# Global Variables

---
### API\_URL
- **Type**: `string`
- **Description**: The `API_URL` variable is a global string variable that holds the URL endpoint for the API server. It defaults to 'http://127.0.0.1:8080' if not set externally, allowing for flexibility in specifying different API server locations.
- **Use**: This variable is used to construct the URL for API requests made by the script, such as tokenization and chat completion.


---
### CHAT
- **Type**: `array`
- **Description**: The `CHAT` variable is an array that stores a sequence of strings representing a conversation between a human and an artificial intelligence assistant. Each element in the array alternates between a human's question and the assistant's response.
- **Use**: This variable is used to maintain the context of the conversation by appending new questions and responses as the chat progresses.


---
### INSTRUCTION
- **Type**: `string`
- **Description**: The `INSTRUCTION` variable is a string that provides a brief description of a conversation between a human and an AI assistant. It sets the context for the interaction, indicating that the assistant is expected to provide helpful, detailed, and polite responses to the human's inquiries.
- **Use**: This variable is used as part of the prompt formatting in the `format_prompt` function to guide the AI's responses during the chat session.


---
### N\_KEEP
- **Type**: `integer`
- **Description**: `N_KEEP` is a global variable that stores the number of tokens in the `INSTRUCTION` string. It is calculated by tokenizing the `INSTRUCTION` string using the `tokenize` function and counting the resulting tokens with `wc -l`. This value represents the number of tokens that should be retained in the prompt data structure when making a request to the chat completion API.
- **Use**: `N_KEEP` is used in the `chat_completion` function to specify the number of tokens to keep in the prompt data when sending a request to the API.


# Functions

---
### trim
The `trim` function removes leading and trailing whitespace from a given string.
- **Inputs**:
    - `$1`: The string from which leading and trailing whitespace should be removed.
- **Control Flow**:
    - Enable extended pattern matching features with `shopt -s extglob`.
    - Remove leading whitespace from the input string using parameter expansion and assign it back to the positional parameter.
    - Remove trailing whitespace from the modified string and print the result.
- **Output**: The function outputs the input string with leading and trailing whitespace removed.


---
### trim\_trailing
The `trim_trailing` function removes trailing whitespace from a given string.
- **Inputs**:
    - `$1`: The input string from which trailing whitespace is to be removed.
- **Control Flow**:
    - Enable extended pattern matching features with `shopt -s extglob`.
    - Use `printf` to output the input string with trailing whitespace removed using pattern substitution.
- **Output**: The function outputs the input string with all trailing whitespace characters removed.


---
### format\_prompt
The `format_prompt` function constructs a formatted string for a chat prompt by combining a predefined instruction with a sequence of human and assistant dialogue lines.
- **Inputs**:
    - `$1`: The latest assistant response to be included in the formatted prompt.
- **Control Flow**:
    - The function begins by echoing the `INSTRUCTION` variable, which contains a description of the chat context.
    - It then uses `printf` to format each pair of human and assistant dialogue lines from the `CHAT` array, appending the latest assistant response passed as `$1`.
    - The formatted string is output as a single line without a trailing newline.
- **Output**: A formatted string representing the chat prompt, including the instruction and dialogue lines.


---
### tokenize
The `tokenize` function sends a POST request to a specified API endpoint to tokenize a given text input and returns the list of tokens.
- **Inputs**:
    - `$1`: A string representing the text content to be tokenized.
- **Control Flow**:
    - The function uses `curl` to send a POST request to the API endpoint `${API_URL}/tokenize`.
    - The request includes a JSON payload with the text content to be tokenized, formatted using `jq`.
    - The response from the API is piped into `jq` to extract and output the list of tokens from the `.tokens[]` array.
- **Output**: A list of tokens extracted from the input text, printed to the standard output.


---
### chat\_completion
The `chat_completion` function generates a response from an AI assistant based on a given user question and updates the chat history.
- **Inputs**:
    - `$1`: The user's question or input to the AI assistant.
- **Control Flow**:
    - The function starts by formatting the prompt using the `format_prompt` function, which includes the instruction and chat history.
    - The prompt is trimmed of trailing spaces using `trim_trailing`.
    - The prompt is then converted into a JSON object with various parameters for the AI model, such as temperature, top_k, top_p, and others.
    - A POST request is made to the `/completion` endpoint of the API with the JSON data to get the AI's response.
    - The response is read line by line, extracting content prefixed with 'data:', and appending it to the `ANSWER` variable.
    - The final answer is printed and added to the `CHAT` array, updating the chat history with the user's question and the AI's response.
- **Output**: The function outputs the AI assistant's response to the user's question and updates the chat history with the new interaction.


