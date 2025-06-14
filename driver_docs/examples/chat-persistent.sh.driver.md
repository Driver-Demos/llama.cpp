# Purpose
This Bash script is designed to manage and facilitate interactions with a language model, specifically the "llama-cli" model, by setting up the necessary environment and handling prompt caching and context management. It provides a narrow functionality focused on preparing and maintaining the context for chat sessions, ensuring that user and AI interactions are logged and managed efficiently. The script checks for required environment variables, sets default values for model and prompt templates, and manages prompt files and caches to optimize the language model's performance. It also includes mechanisms to handle context size limitations by rotating prompts and updating caches in the background, ensuring seamless user interaction with the AI. This script is not an executable or a library but rather a utility script intended to be run in a command-line environment to support AI chat functionalities.
# Global Variables

---
### MODEL
- **Type**: `string`
- **Description**: The `MODEL` variable is a global string variable that specifies the file path to the machine learning model used by the script. It defaults to the path `./models/llama-13b/ggml-model-q4_0.gguf` if not set externally.
- **Use**: This variable is used to define the model file path for the `llama-cli` command, which is executed within the script to process prompts and generate responses.


---
### PROMPT\_TEMPLATE
- **Type**: `string`
- **Description**: The `PROMPT_TEMPLATE` variable is a global string variable that specifies the file path to a text file containing the template for chat prompts. It is initialized with a default value of `./prompts/chat.txt`, but can be overridden by an environment variable of the same name.
- **Use**: This variable is used to locate and load the chat prompt template file, which is then processed and customized with user-specific information before being used in the chat application.


---
### USER\_NAME
- **Type**: `string`
- **Description**: The `USER_NAME` variable is a global string variable that holds the name of the user interacting with the script. It is initialized with a default value of 'User' if not set externally.
- **Use**: This variable is used to personalize the interaction by replacing placeholders in prompt templates and to identify user input in the chat log.


---
### AI\_NAME
- **Type**: `string`
- **Description**: The `AI_NAME` variable is a global string variable that holds the name of the AI, which defaults to 'ChatLLaMa' if not explicitly set. It is used to personalize the AI's identity in interactions, replacing placeholders in prompt templates with the AI's name.
- **Use**: This variable is used to replace the `[[AI_NAME]]` placeholder in prompt templates to personalize the AI's responses.


---
### DATE\_TIME
- **Type**: `string`
- **Description**: The `DATE_TIME` variable is a string that stores the current time in the format of hours and minutes (HH:MM). It is initialized using the `date` command with the `+%H:%M` format specifier, which extracts the current time from the system clock.
- **Use**: This variable is used to replace the `[[DATE_TIME]]` placeholder in the prompt template file, ensuring that the current time is included in the generated prompt.


---
### DATE\_YEAR
- **Type**: `string`
- **Description**: The `DATE_YEAR` variable is a global string variable that stores the current year in a four-digit format (e.g., '2023'). It is initialized using the `date` command with the `+%Y` format specifier, which extracts the year from the system's current date.
- **Use**: This variable is used to dynamically insert the current year into text templates or logs, ensuring that the year is always up-to-date.


---
### LOG
- **Type**: `string`
- **Description**: The `LOG` variable is a string that specifies the file path for the main log file used by the script. It is constructed by appending '/main.log' to the directory path stored in the `CHAT_SAVE_DIR` variable.
- **Use**: This variable is used to store log messages generated by the script, capturing output and errors from the `llama-cli` command and other script operations.


---
### LOG\_BG
- **Type**: `string`
- **Description**: `LOG_BG` is a global variable that holds the file path to the background log file, `main-bg.log`, located in the directory specified by `CHAT_SAVE_DIR`. This log file is used to store output from background processes related to the chat application.
- **Use**: `LOG_BG` is used to redirect the output of background processes to a specific log file for later review or debugging.


---
### CUR\_PROMPT\_FILE
- **Type**: `string`
- **Description**: `CUR_PROMPT_FILE` is a global variable that stores the file path to the current prompt text file used in the chat application. This file is located in the directory specified by `CHAT_SAVE_DIR` and is named `current-prompt.txt`. It is used to store the current state of the chat prompt, including user and AI interactions.
- **Use**: This variable is used to read from and write to the current prompt file, ensuring the chat application has the latest prompt data for processing and generating responses.


---
### CUR\_PROMPT\_CACHE
- **Type**: `string`
- **Description**: `CUR_PROMPT_CACHE` is a global variable that stores the file path to the current prompt cache file, which is used to store cached data for the current session of the chat application. This file is located in the directory specified by `CHAT_SAVE_DIR` and is named `current-cache.bin`. The cache file is used to optimize the performance of the chat application by storing precomputed data that can be reused in subsequent operations.
- **Use**: This variable is used to specify the location of the current prompt cache file, which is utilized by the chat application to store and retrieve cached data for the current session.


---
### NEXT\_PROMPT\_FILE
- **Type**: `string`
- **Description**: `NEXT_PROMPT_FILE` is a global variable that stores the file path for the next prompt text file used in the chat application. It is initialized with a path that combines the `CHAT_SAVE_DIR` directory with the filename `next-prompt.txt`. This file is used to store the next set of prompts that will be used in the chat session.
- **Use**: This variable is used to manage and store the next set of prompts for the chat application, ensuring a smooth transition between prompt files during chat sessions.


---
### NEXT\_PROMPT\_CACHE
- **Type**: `string`
- **Description**: `NEXT_PROMPT_CACHE` is a global variable that holds the file path to the cache file for the next prompt in a chat session. It is initialized with the path `${CHAT_SAVE_DIR}/next-cache.bin`, which indicates that it is stored in the directory specified by `CHAT_SAVE_DIR`.
- **Use**: This variable is used to store and manage the cache for the next prompt, ensuring that the chat session can continue smoothly by preloading necessary data.


---
### SESSION\_AND\_SAMPLE\_PATTERN
- **Type**: `string`
- **Description**: The `SESSION_AND_SAMPLE_PATTERN` is a string variable that contains a regular expression pattern. This pattern is used to match specific log messages related to session file matches and sampling times in milliseconds. The pattern is designed to extract numerical data from log entries that follow a specific format.
- **Use**: This variable is used to search and extract relevant log information from the output of the `llama-cli` command, specifically focusing on session and sampling metrics.


---
### SED\_DELETE\_MESSAGES
- **Type**: `string`
- **Description**: The `SED_DELETE_MESSAGES` variable is a string that contains a sed command pattern used to delete lines from a file. It is constructed to match lines that start with either the `USER_NAME`, `AI_NAME`, or ellipsis (`...`) and delete from that point to the end of the file.
- **Use**: This variable is used in a sed command to process and clean up prompt files by removing specific lines based on the defined pattern.


---
### CTX\_SIZE
- **Type**: `integer`
- **Description**: `CTX_SIZE` is a global variable that defines the maximum size of the context window for the chat application. It is set to 2048, which likely represents the number of tokens or characters that can be processed in a single context window.
- **Use**: This variable is used to determine the context size for the chat model, affecting how much text can be processed or generated in a single session.


---
### CTX\_ROTATE\_POINT
- **Type**: `integer`
- **Description**: `CTX_ROTATE_POINT` is a global variable that calculates a threshold value based on the context size (`CTX_SIZE`). It is set to 60% of the `CTX_SIZE`, which is 2048, resulting in a value of 1228. This value is used to determine when to rotate or swap prompts in the chat application.
- **Use**: This variable is used to decide when to append new input lines to the next prompt file, ensuring that the context does not exceed a certain limit.


---
### OPTS
- **Type**: `array`
- **Description**: The `OPTS` variable is an array that holds command-line options for the `llama-cli` command. It includes options for specifying the model path, context size, and the number of tokens to repeat from the last context, along with any additional arguments passed to the script.
- **Use**: This variable is used to configure and pass necessary options to the `llama-cli` command for processing prompts and generating responses.


# Functions

---
### skip\_bytes
The `skip_bytes` function reads and discards a specified number of bytes from the input before printing the remaining input.
- **Inputs**:
    - `$1`: The number of bytes to skip from the input.
- **Control Flow**:
    - Reads and discards the first N bytes from the input, where N is specified by the argument $1.
    - Continues to read the input byte by byte, printing each byte until the end of the input is reached.
- **Output**: The function outputs the input data starting from the byte immediately after the skipped bytes.


