# Purpose
This source code file is a Node.js script that facilitates a chat interaction between a human user and an artificial intelligence (AI) assistant. The script is designed to read user input from the command line, process it, and generate responses using a remote AI service. The script imports several Node.js modules, such as `readline` for handling input/output operations, `process` for accessing command-line arguments, and `fs` for file system operations. It also imports a custom module, `SchemaConverter`, which is used to convert JSON schema files into a grammar format that can be utilized by the AI service.

The script processes command-line arguments to determine the paths to JSON schema files and grammar files, which are then read and processed. The `SchemaConverter` is used to resolve references within the JSON schema and format it into a grammar that the AI service can use. The script sets up a chat context with predefined exchanges between a human and the assistant, which is used to format prompts for the AI service. The `format_prompt` function constructs a prompt by combining the instruction and chat history with the user's question.

The script defines two asynchronous functions, `tokenize` and `chat_completion`, which interact with a local API service running on `http://127.0.0.1:8080`. The `tokenize` function sends a request to tokenize the instruction text, while `chat_completion` sends a request to generate a response to the user's question. The AI service's response is streamed back to the user, and the chat history is updated accordingly. The script continuously prompts the user for input and processes each question in a loop, providing a dynamic and interactive chat experience.
# Imports and Dependencies

---
- `node:readline`
- `node:process`
- `node:fs`
- `./public_legacy/json-schema-to-grammar.mjs`


# Global Variables

---
### args
- **Type**: `array`
- **Description**: The `args` variable is an array that contains command-line arguments passed to the Node.js process, excluding the first two default arguments (the path to the Node.js executable and the path to the script file).
- **Use**: This variable is used to extract specific command-line options and arguments for configuring the behavior of the script, such as file paths and flags.


---
### grammarJsonSchemaFile
- **Type**: `string | undefined`
- **Description**: The `grammarJsonSchemaFile` variable is a global variable that holds the file path of a JSON schema file specified by the user through command-line arguments. It is extracted from the command-line arguments using the `--grammar-json-schema` flag.
- **Use**: This variable is used to read and parse the JSON schema file, which is then converted into a grammar format for further processing.


---
### no\_cached\_prompt
- **Type**: `string`
- **Description**: The `no_cached_prompt` variable is a global string variable that is initialized by searching the command-line arguments for the presence of the `--no-cache-prompt` flag. If the flag is found, it assigns the corresponding argument value to `no_cached_prompt`; otherwise, it defaults to the string "false".
- **Use**: This variable is used to determine whether the prompt should be cached during the chat completion process by checking if its value is "false".


---
### grammarFile
- **Type**: `string`
- **Description**: The `grammarFile` variable is a string that holds the file path to a grammar file specified by the user through command-line arguments. It is extracted from the command-line arguments by searching for the argument that follows the `--grammar` flag.
- **Use**: This variable is used to read the contents of the specified grammar file into the `grammar` variable if the file path is provided.


---
### grammarJsonSchemaPropOrder
- **Type**: `string`
- **Description**: The `grammarJsonSchemaPropOrder` variable is a global variable that stores a string representing the order of properties in a JSON schema grammar. This string is expected to be a comma-separated list of property names, which is then used to create an object mapping each property to its index in the list. This mapping is used to maintain a specific order of properties when converting a JSON schema to a grammar format.
- **Use**: This variable is used to determine the order of properties when converting a JSON schema to a grammar format.


---
### propOrder
- **Type**: `object`
- **Description**: The `propOrder` variable is a global object that maps property names to their respective order indices, based on a comma-separated list provided as a command-line argument. If the `--grammar-json-schema-prop-order` argument is not provided, `propOrder` defaults to an empty object.
- **Use**: This variable is used to specify the order of properties when converting a JSON schema to a grammar format using the `SchemaConverter`.


---
### grammar
- **Type**: `null`
- **Description**: The `grammar` variable is a global variable that is initially set to `null`. It is intended to hold the grammar data, which can be derived from either a JSON schema file or a direct grammar file, depending on the command-line arguments provided. The grammar data is used in the `chat_completion` function to influence the behavior of the AI assistant's response generation.
- **Use**: This variable is used to store and provide grammar data for the AI assistant's response generation process.


---
### slot\_id
- **Type**: `number`
- **Description**: The `slot_id` variable is a global variable initialized to -1. It is used to track the slot identifier for cached prompts in the chat completion process.
- **Use**: This variable is used to store and update the slot identifier received from the server during the chat completion process, allowing for the management of cached prompts.


---
### API\_URL
- **Type**: `string`
- **Description**: `API_URL` is a string variable that holds the URL of the local server endpoint, specifically 'http://127.0.0.1:8080'. This URL is used to make HTTP requests to the server for various operations such as tokenization and chat completion.
- **Use**: This variable is used as the base URL for making HTTP requests to the server in the `tokenize` and `chat_completion` functions.


---
### chat
- **Type**: `Array<Object>`
- **Description**: The `chat` variable is an array of objects, where each object represents a dialogue exchange between a human and an artificial intelligence assistant. Each object contains two properties: `human`, which holds the human's input as a string, and `assistant`, which contains the assistant's response as a string.
- **Use**: This variable is used to store the history of interactions between the human and the assistant, which is then utilized to format prompts for generating new responses.


---
### instruction
- **Type**: `string`
- **Description**: The `instruction` variable is a string that provides a brief description of the context for a chat interaction between a human and an AI assistant. It sets the tone and expectations for the assistant's responses, emphasizing that they should be helpful, detailed, and polite.
- **Use**: This variable is used as part of the prompt formatting in the `format_prompt` function to guide the AI's responses during chat interactions.


---
### n\_keep
- **Type**: `number`
- **Description**: The `n_keep` variable is a global variable that stores the length of the tokenized `instruction` string. It is calculated by calling the `tokenize` function with the `instruction` string and obtaining the length of the resulting token array.
- **Use**: This variable is used to specify the number of tokens to retain in the prompt during the chat completion process.


---
### rl
- **Type**: `readline.Interface`
- **Description**: The `rl` variable is an instance of the `readline.Interface` class, which is part of the Node.js `readline` module. This interface provides an abstraction for reading data from a Readable stream, such as `stdin`, one line at a time.
- **Use**: The `rl` variable is used to create an interactive command-line interface that reads user input from the standard input stream (`stdin`) and outputs to the standard output stream (`stdout`).


# Functions

---
### format\_prompt
The `format_prompt` function constructs a formatted string for a chat prompt by combining a predefined instruction with a series of human-assistant exchanges and a new question.
- **Inputs**:
    - `question`: A string representing the question posed by the human to the assistant.
- **Control Flow**:
    - The function starts by defining a constant `instruction` which describes the nature of the chat between a human and an AI assistant.
    - It then maps over the `chat` array, which contains objects with `human` and `assistant` properties, to create a formatted string for each exchange.
    - Each exchange is formatted as '### Human: <human's message>\n### Assistant: <assistant's message>'.
    - The formatted exchanges are joined together with newline characters.
    - The function appends the new question to the formatted exchanges, prefixed by '### Human: ', and ends with '### Assistant:'.
    - Finally, the function returns the complete formatted string.
- **Output**: A string that represents the complete chat prompt, including the instruction, previous exchanges, and the new question.


---
### tokenize
The `tokenize` function sends a POST request to a specified API endpoint to tokenize a given content string and returns the tokens if the request is successful.
- **Inputs**:
    - `content`: A string representing the content to be tokenized.
- **Control Flow**:
    - The function sends a POST request to the API endpoint `${API_URL}/tokenize` with the content string as the body in JSON format.
    - It checks if the response from the API is successful (i.e., `result.ok` is true).
    - If the response is not successful, it returns an empty array.
    - If the response is successful, it parses the JSON response to extract and return the `tokens` array.
- **Output**: An array of tokens extracted from the content string, or an empty array if the API request fails.


---
### chat\_completion
The `chat_completion` function sends a formatted prompt to an AI model for generating a conversational response and updates the chat history with the response.
- **Inputs**:
    - `question`: A string representing the human's question to be answered by the AI assistant.
- **Control Flow**:
    - The function begins by sending a POST request to the API endpoint '/completion' with a JSON body containing the formatted prompt and various parameters for the AI model.
    - The function checks if the API response is not OK, in which case it returns immediately without further processing.
    - If the response is OK, it initializes an empty string `answer` to accumulate the AI's response.
    - The function iterates over the streamed chunks of the response body, converting each chunk from a buffer to a UTF-8 string.
    - For each chunk, if it starts with 'data: ', it parses the JSON message, updates the `slot_id`, appends the message content to `answer`, and writes the content to the standard output.
    - If the message indicates a stop condition, it checks if the message was truncated and removes the oldest chat entry if so, then breaks the loop.
    - After processing all chunks, it writes a newline to the standard output and appends the new question and answer pair to the chat history.
- **Output**: The function does not return a value; instead, it updates the global `chat` array with the new question and response pair and writes the response to the standard output.


---
### readlineQuestion
The `readlineQuestion` function prompts the user with a query and returns their input as a promise.
- **Inputs**:
    - `rl`: An instance of readline.Interface used to read input from the standard input.
    - `query`: A string representing the question or prompt to display to the user.
    - `options`: Optional parameter that can be used to specify additional options for the readline question method.
- **Control Flow**:
    - The function returns a new Promise.
    - The `rl.question` method is called with the provided query and options.
    - The `resolve` function of the Promise is passed as the callback to `rl.question`, which will be called with the user's input.
- **Output**: A Promise that resolves with the user's input as a string.


