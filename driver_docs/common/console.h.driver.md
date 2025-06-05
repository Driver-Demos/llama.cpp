# Purpose
This code is a C++ header file that provides a narrow set of functionalities related to console input and output operations. It defines a namespace `console` which encapsulates an enumeration `display_t` for different display modes such as `reset`, `prompt`, `user_input`, and `error`. The header declares several functions: `init` for initializing the console with options for simple or advanced I/O, `cleanup` for resource deallocation or resetting the console state, `set_display` for setting the current display mode, and `readline` for reading input from the console, with support for multiline input. This header is intended to be included in other C++ source files to facilitate console-based interactions.
# Imports and Dependencies

---
- `string`


# Data Structures

---
### display\_t<!-- {{#data_structure:console::display_t}} -->
- **Type**: `enum`
- **Members**:
    - `reset`: Represents the reset state with a value of 0.
    - `prompt`: Represents the prompt state.
    - `user_input`: Represents the user input state.
    - `error`: Represents the error state.
- **Description**: The `display_t` enum defines a set of named constants used to represent different display states in the console namespace, including reset, prompt, user input, and error states. Each enumerator corresponds to a specific state that can be used to control the display behavior in console applications.


