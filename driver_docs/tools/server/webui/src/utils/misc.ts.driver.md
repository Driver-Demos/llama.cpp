# Purpose
This source code file provides a collection of utility functions and type definitions that facilitate various operations related to message processing, server communication, and client-side interactions. The file imports several types from a local `types` module, which suggests that it is part of a larger codebase where these types are used to ensure type safety and consistency across the application. The code includes functions for type checking (`isString`, `isBoolean`, `isNumeric`), string manipulation (`escapeAttr`), and handling server-sent events (SSE) through the `getSSEStreamAsync` function, which processes streaming data from a server response.

The file also contains functions for manipulating and normalizing message data, such as `normalizeMsgsForAPI`, which formats messages for API consumption by filtering and structuring content based on message roles and types. Another function, `filterThoughtFromMsgs`, is designed to process messages by removing specific content tags, which is particularly recommended for a system referred to as "DeepsSeek-R1". Additionally, utility functions like `copyStr` for clipboard operations, `classNames` for generating class strings from object maps, and `throttle` for rate-limiting function execution are included, indicating a focus on enhancing client-side functionality.

Furthermore, the file includes a function `getServerProps` that fetches server properties from a specified base URL, optionally using an API key for authentication. This function demonstrates the file's role in facilitating server communication, likely as part of a client-server architecture. Overall, the file serves as a utility module that provides essential functions for handling data, interacting with APIs, and managing client-side operations, making it a versatile component within the broader application.
# Imports and Dependencies

---
- `textlinestream`
- `./types`
- `@sec-ant/readable-stream/ponyfill/asyncIterator`


# Global Variables

---
### isString
- **Type**: `function`
- **Description**: The `isString` variable is a function that checks if a given input can be treated as a string by attempting to call the `toLowerCase` method on it. This method is typically available on string objects, so the function returns a boolean indicating whether the input behaves like a string in this context.
- **Use**: This function is used to determine if a variable can be considered a string based on its ability to invoke the `toLowerCase` method.


---
### isBoolean
- **Type**: `function`
- **Description**: The `isBoolean` variable is a function that checks if a given input is strictly a boolean value, either `true` or `false`. It is a utility function used to determine the boolean nature of a variable by comparing it against the boolean literals.
- **Use**: This function is used to verify if a variable is a boolean by checking if it is strictly equal to `true` or `false`.


---
### isNumeric
- **Type**: `function`
- **Description**: The `isNumeric` variable is a function that determines if a given input is numeric. It checks that the input is not a string, is not NaN, and is not a boolean value.
- **Use**: This function is used to validate whether a given input can be considered a numeric value, excluding strings and boolean values.


---
### escapeAttr
- **Type**: `function`
- **Description**: The `escapeAttr` variable is a function that takes a string as input and returns a new string with certain characters replaced by their HTML entity equivalents. Specifically, it replaces the '>' character with '&gt;' and the '"' character with '&quot;'. This is typically used to sanitize strings for safe inclusion in HTML attributes, preventing potential security vulnerabilities such as cross-site scripting (XSS) attacks.
- **Use**: This variable is used to sanitize strings by replacing certain characters with their HTML entity equivalents for safe inclusion in HTML attributes.


---
### copyStr
- **Type**: `function`
- **Description**: The `copyStr` variable is a function that copies a given string to the clipboard. It utilizes the Navigator clipboard API if available and the context is secure, otherwise it falls back to using a hidden textarea element to perform the copy operation.
- **Use**: This function is used to copy text to the clipboard, ensuring compatibility across different environments by checking for secure context and using a fallback method if necessary.


---
### delay
- **Type**: `function`
- **Description**: The `delay` variable is a function that returns a Promise which resolves after a specified number of milliseconds. It uses the `setTimeout` function to create a delay before resolving the Promise.
- **Use**: This variable is used to introduce a delay in asynchronous operations by pausing execution for a specified duration.


---
### throttle
- **Type**: `function`
- **Description**: The `throttle` variable is a higher-order function that limits the rate at which a given callback function can be executed. It takes a callback function and a delay in milliseconds as arguments, and returns a new function that ensures the callback is not called more frequently than the specified delay.
- **Use**: This function is used to control the execution frequency of a callback, preventing it from being called repeatedly within a short time span.


---
### cleanCurrentUrl
- **Type**: `function`
- **Description**: The `cleanCurrentUrl` function is a utility function designed to modify the current URL by removing specified query parameters. It takes an array of query parameter names as input and deletes these parameters from the URL's search parameters, then updates the browser's history state with the modified URL.
- **Use**: This function is used to clean up the current URL by removing unnecessary query parameters, which can be useful for maintaining cleaner URLs or for privacy reasons.


---
### getServerProps
- **Type**: `function`
- **Description**: `getServerProps` is an asynchronous function that fetches server properties from a specified base URL. It optionally accepts an API key for authorization and returns a promise that resolves to `LlamaCppServerProps`, which contains the server properties.
- **Use**: This function is used to retrieve server properties by making an HTTP GET request to the `/props` endpoint of the specified base URL, optionally using an API key for authentication.


# Functions

---
### isString
The `isString` function checks if a given input has a `toLowerCase` method, indicating it is a string.
- **Inputs**:
    - `x`: The input value to be checked if it is a string.
- **Control Flow**:
    - The function takes an input `x`.
    - It checks if `x` has a `toLowerCase` method by attempting to convert it to a boolean using `!!x.toLowerCase`.
    - If `x` has a `toLowerCase` method, it returns `true`; otherwise, it returns `false`.
- **Output**: A boolean value indicating whether the input is a string (i.e., has a `toLowerCase` method).


---
### isBoolean
The `isBoolean` function checks if a given input is strictly a boolean value, either `true` or `false`.
- **Inputs**:
    - `x`: The input value to be checked, which can be of any type.
- **Control Flow**:
    - The function evaluates whether the input `x` is strictly equal to `true` or `false`.
- **Output**: A boolean value: `true` if the input is either `true` or `false`, otherwise `false`.


---
### isNumeric
The `isNumeric` function checks if a given input is a numeric value, excluding strings and boolean values.
- **Inputs**:
    - `n`: The input value to be checked for being numeric.
- **Control Flow**:
    - The function first checks if the input is a string using the `isString` function.
    - If the input is not a string, it then checks if the input is not a number using `isNaN`.
    - If the input is not a boolean value using the `isBoolean` function, the function returns true, indicating the input is numeric.
- **Output**: Returns a boolean value: `true` if the input is numeric and not a string or boolean, otherwise `false`.


---
### escapeAttr
The `escapeAttr` function replaces certain characters in a string with their corresponding HTML entity codes to prevent HTML attribute injection.
- **Inputs**:
    - `str`: A string that may contain characters needing to be escaped for safe HTML attribute usage.
- **Control Flow**:
    - The function takes a string input.
    - It uses the `replace` method to substitute occurrences of '>' with '&gt;'.
    - It further replaces occurrences of '"' with '&quot;'.
    - The modified string is returned.
- **Output**: A string with '>' and '"' characters replaced by '&gt;' and '&quot;', respectively, to ensure safe usage in HTML attributes.


---
### getSSEStreamAsync
The `getSSEStreamAsync` function processes a server-sent events (SSE) stream from a fetch response, yielding parsed JSON data or throwing errors as necessary.
- **Inputs**:
    - `fetchResponse`: A Response object from a fetch call, expected to contain a readable stream in its body.
- **Control Flow**:
    - Check if the `fetchResponse` has a body; if not, throw an error indicating the response body is empty.
    - Pipe the response body through a `TextDecoderStream` and a `TextLineStream` to process the stream line by line.
    - Iterate over each line of the stream using an async iterator.
    - For each line, check if it starts with 'data:' and does not end with '[DONE]'; if so, parse the line as JSON and yield the resulting data.
    - If a line starts with 'error:', parse it as JSON and throw an error with the message from the parsed data.
- **Output**: Yields parsed JSON data from the SSE stream or throws an error if an error message is encountered.


---
### copyStr
The `copyStr` function copies a given string to the clipboard using the Clipboard API or a fallback method if the API is unavailable.
- **Inputs**:
    - `textToCopy`: A string representing the text that needs to be copied to the clipboard.
- **Control Flow**:
    - Check if the `navigator.clipboard` API is available and the context is secure (HTTPS).
    - If available, use `navigator.clipboard.writeText` to copy the text directly to the clipboard.
    - If not available, create a hidden textarea element, set its value to the text to be copied, and append it to the document body.
    - Select the text within the textarea and execute the `document.execCommand('copy')` to copy the text.
    - Remove the textarea from the document.
- **Output**: The function does not return any value; it performs a side effect of copying text to the clipboard.


---
### normalizeMsgsForAPI
The `normalizeMsgsForAPI` function processes an array of `Message` objects to filter out redundant fields and format additional content for API consumption.
- **Inputs**:
    - `messages`: An array of `Message` objects, each containing a role, content, and optional extra data.
- **Control Flow**:
    - Iterate over each `Message` object in the `messages` array.
    - Check if the message role is 'user' and if it contains extra data.
    - If the message does not meet the above condition, return an `APIMessage` object with the role and content of the message.
    - If the message contains extra data, initialize an empty array `contentArr` to store formatted content parts.
    - Iterate over each item in the `extra` array of the message.
    - For each `extra` item, check its type and push a corresponding `APIMessageContentPart` object to `contentArr`.
    - If the `extra` type is 'context', add a text part with the extra content.
    - If the `extra` type is 'textFile', add a text part with the file name and content.
    - If the `extra` type is 'imageFile', add an image URL part with the base64 URL.
    - If the `extra` type is 'audioFile', add an input audio part with the base64 data and format.
    - Throw an error if the `extra` type is unknown.
    - After processing all `extra` items, append the original message content as a text part to `contentArr`.
    - Return an `APIMessage` object with the role and the `contentArr` as the content.
- **Output**: An array of `APIMessage` objects, each containing a role and a formatted content array suitable for API use.


---
### filterThoughtFromMsgs
The `filterThoughtFromMsgs` function processes an array of APIMessage objects, specifically filtering out content between <think> and </think> tags for messages with the role 'assistant'.
- **Inputs**:
    - `messages`: An array of APIMessage objects, where each message has a role and content.
- **Control Flow**:
    - Iterate over each message in the input array.
    - Check if the message role is 'assistant'.
    - If the role is 'assistant', split the content string by '</think>' and take the last part, trimming any whitespace.
    - Return the modified message with the filtered content if applicable, otherwise return the message unchanged.
- **Output**: An array of APIMessage objects with 'assistant' messages having content filtered to exclude text between <think> and </think> tags.


---
### classNames
The `classNames` function generates a space-separated string of class names based on a given object where the keys are class names and the values are booleans indicating whether the class should be included.
- **Inputs**:
    - `classes`: An object where keys are class names (strings) and values are booleans indicating whether the class should be included in the output string.
- **Control Flow**:
    - Convert the input object into an array of its entries (key-value pairs).
    - Filter the entries to include only those where the value is true.
    - Map the filtered entries to extract the keys (class names).
    - Join the resulting array of class names into a single string, separated by spaces.
- **Output**: A string containing the class names whose corresponding values in the input object are true, separated by spaces.


---
### delay
The `delay` function creates a promise that resolves after a specified number of milliseconds.
- **Inputs**:
    - `ms`: A number representing the delay duration in milliseconds.
- **Control Flow**:
    - The function returns a new Promise.
    - The Promise uses the `setTimeout` function to delay the resolution.
    - The `setTimeout` is set to resolve the Promise after the specified `ms` milliseconds.
- **Output**: A Promise that resolves after the specified delay in milliseconds.


---
### throttle
The `throttle` function limits the execution of a callback function to once every specified delay period, preventing it from being called too frequently.
- **Inputs**:
    - `callback`: A function to be executed after the throttle delay period.
    - `delay`: A number representing the delay period in milliseconds between allowed executions of the callback function.
- **Control Flow**:
    - Initialize a boolean variable `isWaiting` to `false` to track if the function is in a waiting state.
    - Return a new function that takes any number of arguments `args`.
    - Check if `isWaiting` is `true`; if so, return immediately without executing the callback.
    - If `isWaiting` is `false`, execute the callback function with the provided arguments `args`.
    - Set `isWaiting` to `true` to prevent further execution of the callback until the delay period has passed.
    - Use `setTimeout` to reset `isWaiting` to `false` after the specified delay period, allowing the callback to be executed again.
- **Output**: Returns a throttled version of the input callback function that can only be executed once per specified delay period.


---
### cleanCurrentUrl
The `cleanCurrentUrl` function removes specified query parameters from the current URL and updates the browser's history state without reloading the page.
- **Inputs**:
    - `removeQueryParams`: An array of strings representing the query parameter names to be removed from the current URL.
- **Control Flow**:
    - Create a new URL object from the current window location.
    - Iterate over each query parameter name in the `removeQueryParams` array.
    - For each parameter, delete it from the URL's search parameters.
    - Update the browser's history state with the modified URL using `window.history.replaceState`.
- **Output**: The function does not return any value; it modifies the browser's URL and history state in place.


---
### getServerProps
The `getServerProps` function fetches server properties from a specified base URL, optionally using an API key for authorization.
- **Inputs**:
    - `baseUrl`: A string representing the base URL from which to fetch server properties.
    - `apiKey`: An optional string representing the API key used for authorization in the request headers.
- **Control Flow**:
    - The function attempts to fetch server properties from the endpoint `${baseUrl}/props` using the Fetch API.
    - It sets the 'Content-Type' header to 'application/json' and includes an 'Authorization' header if an `apiKey` is provided.
    - If the response is not OK (i.e., the status is not in the range 200-299), it throws an error indicating failure to fetch server properties.
    - If the response is successful, it parses the JSON response body and returns it as `LlamaCppServerProps`.
    - If any error occurs during the fetch or JSON parsing, it logs the error to the console and rethrows the error.
- **Output**: The function returns a promise that resolves to an object of type `LlamaCppServerProps`, representing the server properties.


