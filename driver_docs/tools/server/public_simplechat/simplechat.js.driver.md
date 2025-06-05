# Purpose
This JavaScript file implements a web-based frontend logic for a chat and completions system, designed to interact with a language model API. The code is structured around several classes, each serving a specific role in managing chat sessions and API interactions. The `SimpleChat` class is central to this functionality, managing individual chat sessions, storing chat messages, and handling the construction and parsing of API requests and responses. It supports both streaming and one-shot response modes, allowing for dynamic interaction with the language model. The `ApiEP` class defines the API endpoints and constructs URLs for API requests, while the `Roles` class categorizes different types of participants in the chat, such as the system, user, and assistant.

The `MultiChatUI` class manages the user interface elements, facilitating the creation and switching of chat sessions. It handles user inputs, updates the chat display, and manages session-specific data. This class also provides mechanisms for users to interact with the chat system, such as submitting queries and setting system prompts. The `Me` class acts as a configuration manager, storing global settings and options for API requests, such as the base URL, headers, and request options. It also provides methods to display and modify these settings through the user interface.

Overall, this file provides a comprehensive framework for a chat application that interfaces with a language model API. It includes functionality for managing multiple chat sessions, constructing and sending API requests, handling responses, and providing a user-friendly interface for interaction. The code is modular, with clear separation of concerns, making it adaptable for different use cases involving chat and completion tasks with language models.
# Imports and Dependencies

---
- `./datautils.mjs`
- `./ui.mjs`


# Global Variables

---
### gUsageMsg
- **Type**: `string`
- **Description**: The `gUsageMsg` variable is a global string variable that contains HTML content. This HTML content is structured to provide usage instructions for a web-based chat interface, detailing how users can interact with the system, such as using shift+enter for new lines and entering queries for the AI assistant.
- **Use**: This variable is used to display usage instructions in the chat interface when no chat messages are present.


---
### gMe
- **Type**: `Me`
- **Description**: The `gMe` variable is an instance of the `Me` class, which encapsulates the configuration and management of a multi-chat user interface for interacting with a language model API. It holds various settings such as the base URL for API requests, default chat session IDs, and options for handling chat completions and streaming responses.
- **Use**: This variable is used to initialize and manage the chat interface, configure API request options, and handle user interactions with the chat system.


# Data Structures

---
### Roles
- **Type**: `class`
- **Members**:
    - `System`: A static member representing the 'system' role.
    - `User`: A static member representing the 'user' role.
    - `Assistant`: A static member representing the 'assistant' role.
- **Description**: The `Roles` class is a simple data structure that defines three static string constants: `System`, `User`, and `Assistant`. These constants are used to represent different roles in a chat or completion system, allowing for easy identification and management of messages or actions associated with each role. The class serves as a centralized definition for these role identifiers, ensuring consistency across the application.


---
### ApiEP
- **Type**: `class`
- **Members**:
    - `Type`: An object containing constants for different API endpoint types, specifically 'Chat' and 'Completion'.
    - `UrlSuffix`: An object mapping API endpoint types to their respective URL suffixes.
    - `Url`: A static method that constructs a full URL by appending the appropriate URL suffix to a given base URL.
- **Description**: The `ApiEP` class is a utility class designed to manage API endpoint types and their corresponding URL suffixes for a web application. It provides constants for different types of API endpoints, such as 'Chat' and 'Completion', and maps these types to their respective URL suffixes. The class also includes a static method, `Url`, which constructs a complete URL by appending the appropriate suffix to a given base URL, ensuring that the base URL does not end with a trailing slash. This class is essential for managing and constructing URLs for API requests in the application.


---
### SimpleChat
- **Type**: `class`
- **Members**:
    - `chatId`: A unique identifier for the chat session.
    - `xchat`: An array of chat messages, each with a role and content.
    - `iLastSys`: Index of the last system message in the chat history.
    - `latestResponse`: Stores the latest response from the AI model.
- **Description**: The `SimpleChat` class is designed to manage chat sessions with a large language model (LLM) web service. It maintains a chat history (`xchat`) that includes messages from different roles (system, user, assistant) and tracks the last system message index (`iLastSys`). The class provides methods to add messages, clear chat history, save and load chat sessions from local storage, and handle responses from the AI model. It also supports generating request payloads for different API endpoints and managing the display of chat messages in a web UI.


---
### MultiChatUI
- **Type**: `class`
- **Members**:
    - `simpleChats`: An object that stores instances of SimpleChat, indexed by chat session IDs.
    - `curChatId`: A string representing the current active chat session ID.
    - `elInSystem`: An HTMLInputElement for system input.
    - `elDivChat`: An HTMLDivElement for displaying chat content.
    - `elBtnUser`: An HTMLButtonElement for user interaction.
    - `elInUser`: An HTMLInputElement for user input.
    - `elDivHeading`: An HTMLSelectElement for heading display.
    - `elDivSessions`: An HTMLDivElement for displaying chat sessions.
    - `elBtnSettings`: An HTMLButtonElement for accessing settings.
- **Description**: The MultiChatUI class is responsible for managing multiple chat sessions in a web-based chat application. It maintains a collection of SimpleChat instances, each representing a chat session, and provides methods to handle user interactions, such as submitting queries and switching between chat sessions. The class also manages the UI elements associated with the chat interface, ensuring that user inputs and chat displays are correctly handled and updated. It validates the presence of necessary HTML elements and sets up event listeners for user actions, facilitating a seamless chat experience.


---
### Me
- **Type**: `class`
- **Members**:
    - `baseURL`: The base URL for the API endpoint.
    - `defaultChatIds`: An array of default chat session identifiers.
    - `multiChat`: An instance of the MultiChatUI class for managing multiple chat sessions.
    - `bStream`: A boolean indicating if streaming mode is enabled.
    - `bCompletionFreshChatAlways`: A boolean indicating if a fresh chat should be started for each completion.
    - `bCompletionInsertStandardRolePrefix`: A boolean indicating if a standard role prefix should be inserted in completions.
    - `bTrimGarbage`: A boolean indicating if garbage should be trimmed from responses.
    - `iRecentUserMsgCnt`: An integer representing the count of recent user messages to include in context.
    - `sRecentUserMsgCnt`: An object mapping string keys to integer values for recent user message count options.
    - `apiEP`: The current API endpoint type, either 'Chat' or 'Completion'.
    - `headers`: An object containing HTTP headers for API requests.
    - `apiRequestOptions`: An object containing options for API requests, such as model and temperature.
- **Description**: The Me class is a configuration and management class for a web-based chat application that interacts with a language model API. It holds various settings and options for API requests, manages multiple chat sessions through the MultiChatUI instance, and provides methods to configure and display settings. The class is designed to facilitate interaction with a language model by setting up chat sessions, handling API requests, and managing user interface elements.


# Functions

---
### Url
The `Url` function constructs a complete URL by appending an API endpoint suffix to a given base URL.
- **Inputs**:
    - `baseUrl`: A string representing the base URL to which the API endpoint suffix will be appended.
    - `apiEP`: A string representing the API endpoint type, which determines the suffix to append to the base URL.
- **Control Flow**:
    - Check if the baseUrl ends with a "/" and remove it if present.
    - Retrieve the appropriate URL suffix from the `UrlSuffix` object using the `apiEP` key.
    - Concatenate the baseUrl with the retrieved URL suffix to form the complete URL.
    - Return the constructed URL.
- **Output**: A string representing the complete URL formed by appending the appropriate API endpoint suffix to the base URL.


---
### clear
The `clear` function resets the chat history and the index of the last system message in a `SimpleChat` instance.
- **Inputs**: None
- **Control Flow**:
    - The function sets the `xchat` array to an empty array, effectively clearing all chat messages.
    - The function sets `iLastSys` to -1, indicating that there is no system message in the chat history.
- **Output**: The function does not return any value; it modifies the state of the `SimpleChat` instance.


---
### ods\_key
The `ods_key` function generates a unique key for storing and retrieving chat data from local storage based on the chat ID.
- **Inputs**: None
- **Control Flow**:
    - The function constructs a string by concatenating the prefix 'SimpleChat-' with the `chatId` property of the `SimpleChat` instance.
    - It returns the constructed string as the unique key.
- **Output**: A string representing the unique key for the chat session, formatted as 'SimpleChat-{chatId}'.


---
### save
The `save` function stores the current state of a `SimpleChat` instance in the browser's local storage.
- **Inputs**:
    - `None`: The function does not take any input arguments.
- **Control Flow**:
    - Create an object `ods` containing the current `iLastSys` and `xchat` properties of the `SimpleChat` instance.
    - Convert the `ods` object to a JSON string.
    - Store the JSON string in local storage using a key derived from the `ods_key` method.
- **Output**: The function does not return any value; it performs a side effect by saving data to local storage.


---
### load
The `load` function retrieves and restores a previously saved chat session from local storage into the current `SimpleChat` instance.
- **Inputs**:
    - `None`: The function does not take any parameters.
- **Control Flow**:
    - Retrieve the saved chat session data from local storage using the chat's unique key.
    - Check if the retrieved data is null; if so, exit the function.
    - Parse the retrieved JSON string into an object of type `SimpleChatODS`.
    - Update the `iLastSys` and `xchat` properties of the current `SimpleChat` instance with the parsed data.
- **Output**: The function does not return any value; it updates the state of the `SimpleChat` instance.


---
### recent\_chat
The `recent_chat` function retrieves recent chat messages from a chat history, either returning the full history or a limited number of recent user messages up to the last system prompt.
- **Inputs**:
    - `iRecentUserMsgCnt`: A number indicating how many recent user messages to include; if negative, the full chat history is returned.
- **Control Flow**:
    - Check if `iRecentUserMsgCnt` is less than 0, and if so, return the full chat history `this.xchat`.
    - If `iRecentUserMsgCnt` is 0, log a warning about no user messages being sent.
    - Initialize an empty array `rchat` to store the recent chat messages.
    - Retrieve the latest system message using `get_system_latest` and add it to `rchat` if it exists.
    - Iterate backwards through `this.xchat` from the end to the last system message, counting user messages until `iRecentUserMsgCnt` is reached.
    - Add each message from the starting point to the end of `this.xchat` to `rchat`, excluding system messages.
    - Return the `rchat` array containing the recent chat messages.
- **Output**: An array of chat messages, either the full history or a limited set of recent messages based on the input parameter.


---
### append\_response
The `append_response` function appends a given string content to the `latestResponse` property of the `SimpleChat` class.
- **Inputs**:
    - `content`: A string representing the content to be appended to the `latestResponse` property.
- **Control Flow**:
    - The function takes a single string argument `content`.
    - It appends this `content` to the `latestResponse` property of the `SimpleChat` instance.
- **Output**: The function does not return any value; it modifies the `latestResponse` property in place.


---
### add
The `add` function adds a new chat message to the `xchat` array with a specified role and content, and updates the last system message index if the role is 'system'.
- **Inputs**:
    - `role`: A string representing the role of the message, such as 'system', 'user', or 'assistant'.
    - `content`: A string representing the content of the message to be added, which can also be undefined or null.
- **Control Flow**:
    - Check if the content is undefined, null, or an empty string; if so, return false.
    - Push a new object with the role and content into the `xchat` array.
    - If the role is 'system', update `iLastSys` to the index of the new message.
    - Call the `save` method to persist the current state of the chat.
    - Return true to indicate the message was successfully added.
- **Output**: A boolean value indicating whether the message was successfully added to the chat.


---
### show
The `show` function displays chat messages in a specified HTML div element, optionally clearing previous content, and scrolls the last message into view.
- **Inputs**:
    - `div`: An HTMLDivElement where the chat messages will be displayed.
    - `bClear`: A boolean indicating whether to clear the div's existing content before displaying new messages; defaults to true.
- **Control Flow**:
    - If `bClear` is true, the function clears the contents of the `div` by replacing its children.
    - The function retrieves recent chat messages using `recent_chat` with a global message count limit.
    - For each message, it creates a paragraph element with the message's role and content, appends it to the `div`, and assigns a class name based on the role.
    - If there are messages, the last message is scrolled into view.
    - If no messages are present and `bClear` is true, the function sets the `div`'s inner HTML to a usage message and calls setup and info display functions.
- **Output**: The function returns the last appended HTML element if messages are displayed, otherwise it returns undefined.


---
### fetch\_headers
The `fetch_headers` function sets up HTTP headers for a request, including an optional Authorization header, based on global settings.
- **Inputs**:
    - `apiEP`: A string representing the API endpoint type, which is not directly used in this function but is part of the function signature.
- **Control Flow**:
    - Initialize a new Headers object.
    - Iterate over the global headers object `gMe.headers`.
    - For each header, check if the key is 'Authorization' and if its value is empty; if so, skip adding it to the headers.
    - Append each header key-value pair to the Headers object.
- **Output**: Returns a Headers object containing the HTTP headers to be used in a request.


---
### request\_jsonstr\_extend
The `request_jsonstr_extend` function extends a given JSON object with additional fields from a global configuration and optionally adds a stream flag, then converts it to a JSON string.
- **Inputs**:
    - `obj`: An object representing the JSON data to be extended and converted to a string.
- **Control Flow**:
    - Iterate over the global `gMe.apiRequestOptions` object.
    - For each key-value pair in `gMe.apiRequestOptions`, add the pair to the input object `obj`.
    - Check if the global `gMe.bStream` flag is true; if so, add a `stream` field with the value `true` to `obj`.
    - Convert the modified `obj` to a JSON string using `JSON.stringify` and return it.
- **Output**: A JSON string representation of the extended input object.


---
### request\_messages\_jsonstr
The `request_messages_jsonstr` function generates a JSON string representation of recent chat messages for use with chat/completions API endpoints.
- **Inputs**: None
- **Control Flow**:
    - Create a request object containing recent chat messages by calling `recent_chat` with the global `iRecentUserMsgCnt` value.
    - Extend the request object with additional fields and options from a global object using `request_jsonstr_extend`.
    - Convert the extended request object to a JSON string and return it.
- **Output**: A JSON string representing the recent chat messages, suitable for sending to chat/completions API endpoints.


---
### request\_prompt\_jsonstr
The `request_prompt_jsonstr` function generates a JSON string representation of recent chat messages formatted for a completions API endpoint, optionally including role prefixes.
- **Inputs**:
    - `bInsertStandardRolePrefix`: A boolean indicating whether to insert a standard role prefix (e.g., "<THE_ROLE>: ") before each role's message in the prompt.
- **Control Flow**:
    - Initialize an empty string `prompt` and a counter `iCnt`.
    - Iterate over recent chat messages obtained from `recent_chat` method.
    - For each message, increment the counter `iCnt` and append a newline to `prompt` if `iCnt` is greater than 1.
    - If `bInsertStandardRolePrefix` is true, append the role and a colon to `prompt`.
    - Append the message content to `prompt`.
    - Create a request object `req` with the `prompt` as its property.
    - Return the JSON string representation of the request object using `request_jsonstr_extend`.
- **Output**: A JSON string representing the request object, formatted for a completions API endpoint, with the prompt constructed from recent chat messages.


---
### request\_jsonstr
The `request_jsonstr` function returns a JSON string representation of a request object suitable for either chat or completion API endpoints based on the specified endpoint type.
- **Inputs**:
    - `apiEP`: A string representing the API endpoint type, which can be either 'chat' or 'completion'.
- **Control Flow**:
    - Check if the provided apiEP is 'chat'.
    - If apiEP is 'chat', call and return the result of `request_messages_jsonstr()`.
    - If apiEP is not 'chat', call and return the result of `request_prompt_jsonstr()` with the argument `gMe.bCompletionInsertStandardRolePrefix`.
- **Output**: A JSON string representing the request object for the specified API endpoint.


---
### response\_extract
The `response_extract` function extracts the AI model's response from an HTTP response body based on the specified API endpoint type.
- **Inputs**:
    - `respBody`: The response body from the HTTP request, which contains the AI model's response data.
    - `apiEP`: A string indicating the type of API endpoint, either 'chat' or 'completion', which determines how the response is extracted.
- **Control Flow**:
    - Initialize an empty string 'assistant' to store the extracted response.
    - Check if the 'apiEP' is of type 'Chat'.
    - If 'apiEP' is 'Chat', extract the assistant's response from 'respBody' using the path ['choices'][0]['message']['content'].
    - If 'apiEP' is not 'Chat', attempt to extract the response using the path ['choices'][0]['text'].
    - If the above extraction fails, use a fallback to extract the response from 'respBody' using the path ['content'].
    - Return the extracted 'assistant' response.
- **Output**: A string containing the extracted response from the AI model, based on the specified API endpoint type.


---
### response\_extract\_stream
The `response_extract_stream` function extracts the AI model's response from a streaming HTTP response body based on the specified API endpoint type.
- **Inputs**:
    - `respBody`: The response body from the HTTP response, which contains the AI model's output data.
    - `apiEP`: A string indicating the type of API endpoint, either 'chat' or 'completion', which determines how the response is extracted.
- **Control Flow**:
    - Initialize an empty string 'assistant' to store the extracted response.
    - Check if the API endpoint type is 'chat'.
    - If 'chat', check if the 'finish_reason' in the response body is not 'stop'.
    - If not 'stop', extract the 'content' from the 'delta' field of the first choice in the response body and assign it to 'assistant'.
    - If the API endpoint type is not 'chat', attempt to extract the 'text' from the first choice in the response body.
    - If extracting 'text' fails, fall back to extracting 'content' from the response body.
- **Output**: Returns a string containing the extracted content from the AI model's response.


---
### add\_system\_begin
The `add_system_begin` function allows setting a system prompt at the beginning of a chat session, ensuring it is the first message and cannot be changed once set.
- **Inputs**:
    - `sysPrompt`: A string representing the system prompt to be added at the beginning of the chat.
    - `msgTag`: A string used for logging or error messages to identify the context of the operation.
- **Control Flow**:
    - Check if the chat history (xchat) is empty.
    - If empty and sysPrompt is not empty, add the system prompt to the chat and return true.
    - If not empty, check if the first message is a system prompt.
    - If the first message is not a system prompt, log an error and return false.
    - If the first message is a system prompt but differs from sysPrompt, log an error and return false.
    - Return false if no conditions for adding the system prompt are met.
- **Output**: Returns a boolean indicating whether the system prompt was successfully added at the beginning of the chat.


---
### add\_system\_anytime
The `add_system_anytime` function allows setting a system prompt at any point in the chat history, adding it to the chat if it differs from the last system prompt.
- **Inputs**:
    - `sysPrompt`: A string representing the system prompt to be added to the chat.
    - `msgTag`: A string used for logging or error messages to identify the context of the operation.
- **Control Flow**:
    - Check if the sysPrompt is non-empty; if empty, return false.
    - If no system prompt has been added before (iLastSys < 0), add the sysPrompt as a new system message and return the result of the add operation.
    - Retrieve the last system prompt from the chat history.
    - If the last system prompt differs from the new sysPrompt, add the new sysPrompt as a system message and return the result of the add operation.
    - If the sysPrompt is the same as the last system prompt, return false.
- **Output**: Returns a boolean indicating whether the system prompt was successfully added to the chat.


---
### get\_system\_latest
The `get_system_latest` function retrieves the most recent system prompt from the chat history.
- **Inputs**: None
- **Control Flow**:
    - Check if `iLastSys` is -1, indicating no system prompt exists, and return an empty string if true.
    - Retrieve the content of the system prompt at the index `iLastSys` in the `xchat` array.
    - Return the retrieved system prompt content.
- **Output**: The function returns the latest system prompt as a string, or an empty string if no system prompt is present.


---
### handle\_response\_multipart
The `handle_response_multipart` function processes a multipart response from a server or AI model, updating the UI with the response content as it becomes available.
- **Inputs**:
    - `resp`: A `Response` object representing the HTTP response from the server or AI model.
    - `apiEP`: A string indicating the API endpoint type, either 'chat' or 'completion'.
    - `elDiv`: An `HTMLDivElement` where the response content will be displayed.
- **Control Flow**:
    - Create a new paragraph element and append it to `elDiv` to display the response.
    - Check if the response has a body; if not, throw an error.
    - Initialize a `TextDecoder` for UTF-8 and a `ReadableStreamDefaultReader` to read the response body.
    - Initialize an empty string `latestResponse` and a `NewLines` object to handle line-by-line processing.
    - Enter a loop to read chunks from the response body until done.
    - Decode each chunk and append it to `xLines` for line processing.
    - Process each line from `xLines`, ignoring empty lines and lines not starting with 'data:'.
    - Parse JSON from lines starting with 'data:', extract content using `response_extract_stream`, and append it to `latestResponse`.
    - Update the paragraph element's text with `latestResponse` and ensure it is visible.
    - Break the loop if the response is fully read.
- **Output**: Returns the full response content as a string after processing all parts.


---
### handle\_response\_oneshot
The `handle_response_oneshot` function processes a single response from a server or AI model and extracts the assistant's response content based on the specified API endpoint type.
- **Inputs**:
    - `resp`: A `Response` object representing the HTTP response received from the server or AI model.
    - `apiEP`: A string indicating the type of API endpoint, either 'chat' or 'completion', which determines how the response content is extracted.
- **Control Flow**:
    - The function awaits the JSON body of the response using `resp.json()`.
    - It logs the response body for debugging purposes.
    - It calls the `response_extract` method with the response body and API endpoint type to extract the assistant's response content.
    - The extracted content is returned as the function's output.
- **Output**: The function returns a string containing the assistant's response extracted from the response body.


---
### handle\_response
The `handle_response` function processes the server's response, either in oneshot or streaming mode, and optionally trims any garbage from the end of the response before adding it to the chat history.
- **Inputs**:
    - `resp`: The HTTP response object received from the server.
    - `apiEP`: A string indicating the API endpoint type, either 'chat' or 'completion'.
    - `elDiv`: The HTMLDivElement where the response will be displayed.
- **Control Flow**:
    - Initialize an object `theResp` to store the assistant's response and any trimmed content.
    - Check if streaming mode is enabled (`gMe.bStream`).
    - If streaming mode is enabled, call `handle_response_multipart` to process the response in parts and handle any errors by storing the latest response and rethrowing the error.
    - If streaming mode is not enabled, call `handle_response_oneshot` to process the response in one go.
    - If garbage trimming is enabled (`gMe.bTrimGarbage`), trim the garbage from the end of the assistant's response and store the trimmed content separately.
    - Add the processed assistant's response to the chat history.
    - Return the `theResp` object containing the assistant's response and any trimmed content.
- **Output**: An object containing the assistant's response and any trimmed content from the end of the response.


---
### validate\_element
The `validate_element` function checks if a given HTML element is not null and logs its ID and name, or throws an error if it is null.
- **Inputs**:
    - `el`: An `HTMLElement` or `null` that represents the element to be validated.
    - `msgTag`: A `string` that serves as a tag for logging or error messages, indicating the context or identifier of the element being validated.
- **Control Flow**:
    - Check if the `el` parameter is `null`.
    - If `el` is `null`, throw an error with a message indicating the element is missing, using `msgTag` for context.
    - If `el` is not `null`, log a debug message with the element's ID and name, using `msgTag` for context.
- **Output**: The function does not return any value; it either logs a debug message or throws an error.


---
### ui\_reset\_userinput
The `ui_reset_userinput` function clears, enables, and sets focus to the user input field in the chat UI.
- **Inputs**: None
- **Control Flow**:
    - The function sets the value of the user input field (`elInUser`) to an empty string, effectively clearing any existing input.
    - It then enables the user input field by setting its `disabled` property to `false`.
    - Finally, it sets the focus to the user input field, allowing the user to start typing immediately.
- **Output**: The function does not return any value; it performs UI updates directly.


---
### setup\_ui
The `setup_ui` function initializes the user interface for a chat application, setting up event listeners and handling session switching.
- **Inputs**:
    - `defaultChatId`: A string representing the default chat session ID to be used initially.
    - `bSwitchSession`: A boolean indicating whether to switch to the specified default chat session immediately.
- **Control Flow**:
    - Set the current chat ID to the provided defaultChatId.
    - If bSwitchSession is true, call handle_session_switch to switch to the specified chat session.
    - Add a click event listener to the settings button to display settings when clicked.
    - Add a click event listener to the user button to handle user input submission when clicked, with error handling for failed submissions.
    - Add a keyup event listener to the user input field to handle Enter key presses for submitting input or inserting new lines with Shift+Enter.
    - Add a keyup event listener to the system input field to handle Enter key presses for setting the system prompt or inserting new lines with Shift+Enter.
- **Output**: The function does not return any value; it sets up the UI and event listeners for the chat application.


---
### new\_chat\_session
The `new_chat_session` function initializes a new chat session with a given chat ID and optionally switches the UI to this new session.
- **Inputs**:
    - `chatId`: A string representing the unique identifier for the new chat session.
    - `bSwitchSession`: A boolean indicating whether to switch the UI to the new chat session immediately after creation; defaults to false.
- **Control Flow**:
    - Create a new instance of `SimpleChat` with the provided `chatId` and store it in the `simpleChats` object.
    - Check if `bSwitchSession` is true; if so, call `handle_session_switch` with the `chatId` to switch the UI to the new session.
- **Output**: The function does not return any value; it modifies the state of the `MultiChatUI` instance by adding a new chat session.


---
### handle\_user\_submit
The `handle_user_submit` function processes a user's query submission for a specified chat session, handling both chat and completion modes, and updates the UI accordingly.
- **Inputs**:
    - `chatId`: A string representing the unique identifier of the chat session.
    - `apiEP`: A string indicating the API endpoint type, either 'chat' or 'completion'.
- **Control Flow**:
    - Retrieve the SimpleChat instance associated with the given chatId.
    - If in completion mode and configured to start fresh, clear the chat history.
    - Add the current system prompt to the chat if applicable.
    - Retrieve the user input from the UI and add it to the chat as a user message.
    - Display the updated chat in the UI.
    - Construct the request URL using the base URL and API endpoint type.
    - Prepare the request body as a JSON string based on the chat history and API endpoint type.
    - Disable the user input field and set its value to 'working...'.
    - Send a POST request to the constructed URL with the prepared headers and body.
    - Handle the server's response using the appropriate method based on the streaming configuration.
    - If the current chat session is still active, update the UI with the assistant's response and any trimmed content.
    - Reset the user input field to be ready for new input.
- **Output**: The function does not return a value but updates the UI with the assistant's response and manages the chat session state.


---
### show\_sessions
The `show_sessions` function displays buttons for creating new chat sessions and for existing chat sessions, updating the UI to highlight the selected session.
- **Inputs**:
    - `elDiv`: An optional HTMLDivElement where the session buttons will be displayed; if not provided, it defaults to `this.elDivSessions`.
- **Control Flow**:
    - Check if `elDiv` is provided; if not, use `this.elDivSessions`.
    - Clear the contents of `elDiv`.
    - Create a button for starting a new chat session and append it to `elDiv`.
    - Iterate over existing chat session IDs and create a button for each, appending them to `elDiv`.
    - Highlight the button corresponding to the current chat session.
- **Output**: The function does not return a value; it updates the UI by adding buttons to the specified or default div element.


---
### create\_session\_btn
The `create_session_btn` function creates a button for a chat session and appends it to a specified HTML element, allowing users to switch between chat sessions.
- **Inputs**:
    - `elDiv`: An HTMLDivElement where the session button will be appended.
    - `cid`: A string representing the chat session ID for which the button is being created.
- **Control Flow**:
    - Create a button element with the label set to the chat session ID (cid).
    - Attach a click event listener to the button that handles session switching.
    - Append the button to the specified HTMLDivElement (elDiv).
    - Return the created button element.
- **Output**: The function returns the created HTMLButtonElement for the chat session.


---
### handle\_session\_switch
The `handle_session_switch` function switches the UI to a specified chat session and updates the current chat session ID.
- **Inputs**:
    - `chatId`: A string representing the ID of the chat session to switch to.
- **Control Flow**:
    - Retrieve the `SimpleChat` instance associated with the provided `chatId` from the `simpleChats` object.
    - Check if the `SimpleChat` instance exists; if not, log an error and return.
    - Set the system input field's value to the latest system prompt from the chat session.
    - Clear the user input field and focus on it.
    - Display the chat session's messages in the chat display area.
    - Update the `curChatId` to the provided `chatId`.
    - Log an informational message indicating the session switch.
- **Output**: The function does not return a value; it updates the UI and internal state to reflect the new chat session.


---
### debug\_disable
The `debug_disable` function disables the console's debug logging by mapping `console.debug` to an empty function.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - The function stores the current `console.debug` function in `this.console_debug` for potential future use.
    - It then reassigns `console.debug` to an empty function, effectively disabling any debug logging.
- **Output**: The function does not return any value or output.


---
### setup\_load
The `setup_load` function sets up the UI to allow loading a previously saved chat session if available.
- **Inputs**:
    - `div`: An HTMLDivElement where the load UI will be appended.
    - `chat`: An instance of the SimpleChat class representing the chat session to be potentially loaded.
- **Control Flow**:
    - Check if the chat session key exists in localStorage.
    - If the key exists, append HTML content to the div to indicate the option to restore a saved chat session.
    - Create a button that, when clicked, loads the saved chat session from localStorage and updates the UI accordingly.
- **Output**: The function does not return a value; it modifies the DOM to include a button for loading a saved chat session if available.


---
### show\_info
The `show_info` function displays the current settings and configuration parameters in a specified HTMLDivElement, with an option to show all settings or just a subset.
- **Inputs**:
    - `elDiv`: An HTMLDivElement where the settings information will be displayed.
    - `bAll`: A boolean flag indicating whether to show all settings (true) or just a subset (false).
- **Control Flow**:
    - Create a paragraph element with the text 'Settings (devel-tools-console document[gMe])' and append it to elDiv.
    - If bAll is true, append additional paragraphs to elDiv displaying various settings such as baseURL, Authorization, bStream, bTrimGarbage, ApiEndPoint, iRecentUserMsgCnt, bCompletionFreshChatAlways, and bCompletionInsertStandardRolePrefix.
    - Regardless of bAll, append paragraphs to elDiv displaying apiRequestOptions and headers as JSON strings.
- **Output**: The function does not return a value; it modifies the DOM by appending elements to the provided elDiv.


---
### show\_settings\_apirequestoptions
The `show_settings_apirequestoptions` function dynamically creates and displays UI input elements for configuring API request options based on their data types.
- **Inputs**:
    - `elDiv`: An HTMLDivElement where the settings UI elements will be appended.
- **Control Flow**:
    - Create a fieldset element and a legend element with the text 'ApiRequestOptions'.
    - Append the legend to the fieldset and the fieldset to the provided div element.
    - Iterate over each key-value pair in the `apiRequestOptions` object.
    - Determine the type of each value and create an appropriate input element (text, number, or boolean button) using helper functions from the `ui` module.
    - Append each created input element to the fieldset.
- **Output**: The function does not return a value; it modifies the DOM by appending input elements to the provided div element.


---
### show\_settings
The `show_settings` function displays a user interface for configuring various settings related to API requests and chat behavior.
- **Inputs**:
    - `elDiv`: An HTMLDivElement where the settings UI will be rendered.
- **Control Flow**:
    - Create an input field for setting the base URL and append it to the provided div.
    - Create an input field for setting the Authorization header and append it to the provided div.
    - Create a boolean button for toggling streaming mode and append it to the provided div.
    - Create a boolean button for toggling garbage trimming and append it to the provided div.
    - Call `show_settings_apirequestoptions` to display input fields for API request options and append them to the provided div.
    - Create a select dropdown for choosing the API endpoint type and append it to the provided div.
    - Create a select dropdown for choosing the chat history context and append it to the provided div.
    - Create a boolean button for toggling fresh chat mode for completions and append it to the provided div.
    - Create a boolean button for toggling the insertion of standard role prefixes in completions and append it to the provided div.
- **Output**: The function does not return any value; it modifies the DOM by appending settings UI elements to the provided div.


---
### startme
The `startme` function initializes the chat application by setting up global configurations, disabling debug logs, creating default chat sessions, and configuring the user interface.
- **Inputs**: None
- **Control Flow**:
    - Log a message indicating the start of the function.
    - Instantiate a new `Me` object and assign it to the global variable `gMe`.
    - Disable console debug logging by calling `gMe.debug_disable()`.
    - Attach the `gMe` and `du` objects to the `document` for global access.
    - Iterate over `gMe.defaultChatIds` to create new chat sessions for each default chat ID using `gMe.multiChat.new_chat_session()`.
    - Set up the user interface for the first default chat session by calling `gMe.multiChat.setup_ui()` with the first default chat ID and `true` to switch to it.
    - Display available chat sessions by calling `gMe.multiChat.show_sessions()`.
- **Output**: The function does not return any value; it sets up the initial state and UI for the chat application.


