# Purpose
This Bash script is designed to execute a command-line interface (CLI) for an AI model, specifically a conversational AI assistant named Miku. The script sets up environment variables and configuration options for running the AI model, such as the model file path, AI and user names, context size, and prediction parameters. It provides a narrow functionality focused on initializing and running a specific AI model with customizable settings, making it a utility script rather than a broad application. The script includes options for threading and various generation parameters, allowing users to adjust the AI's behavior and performance. It is intended to be executed directly, serving as a setup and execution script for the AI model, facilitating an interactive text-based conversation between the user and the AI.
# Global Variables

---
### AI\_NAME
- **Type**: `string`
- **Description**: `AI_NAME` is a global variable that holds the name of the AI assistant, defaulting to 'Miku' if not set by the user. It is used to personalize interactions and responses from the AI, making the conversation more engaging and relatable.
- **Use**: This variable is used to insert the AI's name into prompts and responses, creating a personalized conversational experience.


---
### MODEL
- **Type**: `string`
- **Description**: The `MODEL` variable is a global string variable that specifies the file path to the AI model binary used by the script. It defaults to './models/llama-2-7b-chat.ggmlv3.q4_K_M.bin' if not set externally.
- **Use**: This variable is used to define the model file path for the AI assistant, which is passed as an argument to the `llama-cli` command.


---
### USER\_NAME
- **Type**: `string`
- **Description**: The `USER_NAME` variable is a global string variable that holds the name of the user interacting with the AI assistant. It is initialized with a default value of 'Anon' if not set externally, allowing for customization of the user's name in the conversation.
- **Use**: This variable is used to personalize the interaction between the user and the AI by incorporating the user's name into the conversation prompts and responses.


---
### CTX\_SIZE
- **Type**: `string`
- **Description**: `CTX_SIZE` is a global variable that defines the context size for the AI model's generation process. It is set to a default value of 4096, which can be overridden by an environment variable if specified. The context size determines how much of the conversation history the AI model considers when generating responses.
- **Use**: This variable is used to set the `--ctx_size` option in the `GEN_OPTIONS` array, which configures the AI model's context size during execution.


---
### N\_PREDICTS
- **Type**: `string`
- **Description**: The `N_PREDICTS` variable is a global string variable that sets the default number of predictions or outputs the AI model should generate during its execution. It is initialized with a default value of 4096, which can be overridden by an environment variable of the same name if set.
- **Use**: This variable is used to specify the number of predictions the AI model should generate, influencing the length or extent of the output produced by the model.


---
### GEN\_OPTIONS
- **Type**: `array`
- **Description**: `GEN_OPTIONS` is a global array variable that holds a set of command-line options for configuring the behavior of the `llama-cli` command. These options include parameters such as batch size, context size, repeat penalty, temperature, and mirostat settings, which are used to control the AI model's generation process.
- **Use**: This variable is used to store and pass configuration options to the `llama-cli` command, allowing for customization of the AI model's execution.


