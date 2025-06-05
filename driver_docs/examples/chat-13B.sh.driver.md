# Purpose
This Bash script is designed to execute a command-line interface for a chatbot application, likely named "ChatLLaMa," by setting up the necessary environment and parameters for its operation. It provides a narrow functionality focused on configuring and running the chatbot with specific user-defined or default settings, such as model file paths, prompt templates, user and AI names, and generation options. The script dynamically generates a prompt file by replacing placeholders with actual values like the user's name, AI's name, and current date and time. It then executes the chatbot with these configurations, allowing for interactive sessions with specified options like the number of prediction tokens and threading. This script is not a standalone executable but rather a utility to facilitate the setup and execution of the chatbot application.
# Global Variables

---
### MODEL
- **Type**: `string`
- **Description**: The `MODEL` variable is a global string variable that specifies the file path to the machine learning model binary used by the script. It defaults to './models/13B/ggml-model-q4_0.bin' if not set externally.
- **Use**: This variable is used to define the model file path for the AI application, allowing the script to load and utilize the specified model for processing.


---
### PROMPT\_TEMPLATE
- **Type**: `string`
- **Description**: The `PROMPT_TEMPLATE` variable is a global string variable that holds the file path to a text file containing the prompt template for the chat application. It defaults to './prompts/chat.txt' if not set by the user.
- **Use**: This variable is used to specify the location of the prompt template file, which is then processed to replace placeholders with actual values before being used in the chat application.


---
### USER\_NAME
- **Type**: `string`
- **Description**: The `USER_NAME` variable is a global string variable that holds the name of the user interacting with the script. It is initialized with a default value of 'USER' if not set externally.
- **Use**: This variable is used to personalize the interaction by replacing placeholders in the prompt template with the user's name.


---
### AI\_NAME
- **Type**: `string`
- **Description**: The `AI_NAME` variable is a global string variable that holds the name of the AI model used in the script. It defaults to 'ChatLLaMa' if not set by the user.
- **Use**: This variable is used to replace placeholders in the prompt template with the AI's name.


---
### N\_THREAD
- **Type**: `string`
- **Description**: The `N_THREAD` variable is a global string variable that specifies the number of CPU cores to be used by the script. It is set to a default value of '8', but can be overridden by an environment variable of the same name.
- **Use**: This variable is used to determine the number of threads allocated for processing tasks in the script, optimizing performance based on available CPU resources.


---
### N\_PREDICTS
- **Type**: `string`
- **Description**: The `N_PREDICTS` variable is a global string variable that specifies the number of tokens to predict during the execution of the script. It is set to a default value of 2048, which is larger than the typical default to allow for longer interactions.
- **Use**: This variable is used to determine the number of tokens the AI model should predict when generating responses.


---
### GEN\_OPTIONS
- **Type**: `string`
- **Description**: The `GEN_OPTIONS` variable is a string that contains default command-line options for the `llama-cli` tool. These options include settings for context size, temperature, top-k sampling, top-p sampling, repeat penalty, and batch size, which are used to control the behavior of the language model during text generation.
- **Use**: This variable is used to specify default generation parameters for the `llama-cli` command, which can be overridden by command-line arguments.


---
### DATE\_TIME
- **Type**: `string`
- **Description**: The `DATE_TIME` variable is a string that stores the current time in the format of hours and minutes (HH:MM). It is set using the `date` command with the `+%H:%M` format specifier, which extracts the current hour and minute from the system's date and time settings.
- **Use**: This variable is used to replace the placeholder `[[DATE_TIME]]` in the prompt template file with the current time.


---
### DATE\_YEAR
- **Type**: `string`
- **Description**: The variable `DATE_YEAR` is a global variable that stores the current year as a string. It is initialized using the `date` command with the `+%Y` format specifier, which extracts the year from the system's date and time settings.
- **Use**: This variable is used to replace the placeholder `[[DATE_YEAR]]` in the prompt template file with the current year.


---
### PROMPT\_FILE
- **Type**: `string`
- **Description**: The `PROMPT_FILE` variable is a string that holds the path to a temporary file created using the `mktemp` command. This file is used to store a modified version of the prompt template, where placeholders for user name, AI name, date, and time are replaced with actual values.
- **Use**: `PROMPT_FILE` is used to pass the path of the modified prompt template to the `llama-cli` command for generating AI interactions.


