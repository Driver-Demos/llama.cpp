# Purpose
This code file serves as a configuration module for a software application, providing a set of default configuration constants and their descriptions. It is not an executable file but rather a library intended to be imported and used elsewhere in the application. The file defines a broad range of configuration options, such as API keys, system messages, and various parameters for controlling text generation behavior, including sampling methods and penalties. Additionally, it includes a list of supported themes from the daisyui library and identifies which configuration keys are numeric. The presence of environment-specific variables, like `isDev`, suggests that this module is designed to adapt to different deployment environments, such as development or production.
# Imports and Dependencies

---
- `daisyui/theme/object`
- `./utils/misc`


# Global Variables

---
### isDev
- **Type**: `boolean`
- **Description**: The `isDev` variable is a boolean that determines if the current environment mode is set to 'development'. It is derived from the `import.meta.env.MODE` property, which is a part of the environment metadata provided by the build tool.
- **Use**: This variable is used to conditionally execute code or configure settings that are specific to a development environment.


---
### BASE\_URL
- **Type**: `string`
- **Description**: The `BASE_URL` variable is a global constant that holds the base URL of the current document. It is derived from the `document.baseURI` and is converted to a string after ensuring it does not end with a trailing slash.
- **Use**: This variable is used to provide a consistent base URL for constructing relative URLs throughout the application.


---
### CONFIG\_DEFAULT
- **Type**: `object`
- **Description**: `CONFIG_DEFAULT` is a global configuration object that holds default settings for various parameters used in the application. It includes settings for API keys, system messages, token handling, sampling methods, and experimental features. The object is designed to be single-level and avoids nested objects to maintain simplicity and prevent breaking changes.
- **Use**: This variable is used to provide default configuration values for the application, ensuring consistent behavior and allowing for easy adjustments of settings.


---
### CONFIG\_INFO
- **Type**: `Record<string, string>`
- **Description**: CONFIG_INFO is a global variable that serves as a record mapping configuration keys to their respective descriptions. It provides detailed explanations for each configuration parameter used in the application, such as 'apiKey', 'systemMessage', and various sampling parameters. This record helps users understand the purpose and effect of each configuration setting.
- **Use**: This variable is used to provide descriptive information about each configuration parameter, aiding in the understanding and proper usage of the application's settings.


---
### CONFIG\_NUMERIC\_KEYS
- **Type**: `Array<string>`
- **Description**: `CONFIG_NUMERIC_KEYS` is an array that contains the keys from the `CONFIG_DEFAULT` object which have numeric values. This includes configuration parameters such as `temperature`, `top_k`, `top_p`, and others that are used to control various aspects of the system's behavior.
- **Use**: This variable is used to identify and manage configuration keys that are expected to have numeric values, facilitating operations that require numeric data types.


---
### THEMES
- **Type**: `Array<string>`
- **Description**: The `THEMES` variable is a global array that contains a list of theme names supported by the DaisyUI library. It ensures that the 'light' and 'dark' themes are always at the beginning of the list, followed by other themes available in the DaisyUI theme object, excluding 'light' and 'dark'. This allows for a flexible and dynamic way to manage and access the available themes in the application.
- **Use**: This variable is used to store and provide access to the list of available themes for the application, ensuring 'light' and 'dark' are prioritized.


