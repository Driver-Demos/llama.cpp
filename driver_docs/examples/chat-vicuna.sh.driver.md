# Purpose
This Bash script is designed to execute a command-line interface for a language model, specifically configured to run a chat application using a pre-trained model file. It provides a narrow functionality focused on setting up and executing a chat session between a human user and an AI assistant. The script sets various environment variables and options, such as the model file path, prompt template, and generation options, which can be overridden by command-line arguments. It dynamically generates a prompt file by replacing placeholders with actual values like user and AI names, and the current date and time. The script then runs the `llama-cli` executable with the specified options, facilitating an interactive chat session with the AI model.
# Global Variables

---
### MODEL
- **Type**: `string`
- **Description**: The `MODEL` variable is a string that specifies the file path to the machine learning model binary used by the script. It defaults to './models/ggml-vic13b-uncensored-q5_0.bin' if not set externally.
- **Use**: This variable is used to define the model file path for the `llama-cli` command, which is executed later in the script.


---
### PROMPT\_TEMPLATE
- **Type**: `string`
- **Description**: The `PROMPT_TEMPLATE` variable is a global string variable that specifies the file path to a text file containing the prompt template for the chat application. It defaults to './prompts/chat.txt' if not set externally.
- **Use**: This variable is used to determine the source file for the prompt template, which is then processed to replace placeholders with actual values before being used in the chat application.


---
### USER\_NAME
- **Type**: `string`
- **Description**: The `USER_NAME` variable is a global string variable that holds the value '### Human'. It is used as a placeholder for the user's name in the prompt template.
- **Use**: This variable is used to replace the placeholder `[[USER_NAME]]` in the prompt template with the string '### Human'.


---
### AI\_NAME
- **Type**: `string`
- **Description**: The `AI_NAME` variable is a global string variable that holds the name identifier for the AI in the chat interaction. It is set to the value '### Assistant', which is used to replace placeholders in the prompt template file.
- **Use**: This variable is used to personalize the AI's identity in the chat prompt by replacing the placeholder `[[AI_NAME]]` in the prompt template with its value.


---
### N\_THREAD
- **Type**: `string`
- **Description**: The `N_THREAD` variable is a global string variable that specifies the number of CPU cores to be used by the script. It is set to a default value of '8', but can be overridden by setting the `N_THREAD` environment variable before running the script.
- **Use**: This variable is used to determine the number of threads allocated for processing tasks in the script, optimizing performance based on available CPU resources.


---
### N\_PREDICTS
- **Type**: `string`
- **Description**: `N_PREDICTS` is a global variable that specifies the number of tokens to predict during a model interaction. It is set to a default value of 2048, allowing for a longer interaction than the typical default.
- **Use**: This variable is used to determine the length of the token prediction sequence in the AI model's execution.


---
### GEN\_OPTIONS
- **Type**: `string`
- **Description**: The `GEN_OPTIONS` variable is a string that contains default command-line options for the llama-cli tool. These options include settings for context size, temperature, top-k sampling, top-p sampling, repeat penalty, and batch size, which are used to control the behavior of the language model during text generation.
- **Use**: This variable is used to specify default generation options for the llama-cli command, which can be overridden by command-line arguments.


---
### DATE\_TIME
- **Type**: `string`
- **Description**: The `DATE_TIME` variable is a global string variable that stores the current time in the format of hours and minutes (HH:MM). It is initialized using the `date` command with the `+%H:%M` format specifier, which extracts the current hour and minute from the system's date and time settings.
- **Use**: This variable is used to replace the placeholder `[[DATE_TIME]]` in the prompt template file with the actual current time.


---
### DATE\_YEAR
- **Type**: `string`
- **Description**: The variable `DATE_YEAR` is a global variable that stores the current year as a string. It is initialized using the `date` command with the `+%Y` format specifier, which extracts the year from the system's date.
- **Use**: This variable is used to replace the placeholder `[[DATE_YEAR]]` in the prompt template file with the current year.


---
### PROMPT\_FILE
- **Type**: `string`
- **Description**: `PROMPT_FILE` is a global variable that stores the path to a temporary file created using the `mktemp` command. This file is used to store a modified version of the prompt template, where placeholders for user name, AI name, date, and time are replaced with actual values.
- **Use**: This variable is used to pass the path of the modified prompt file to the `llama-cli` command for generating AI interactions.


