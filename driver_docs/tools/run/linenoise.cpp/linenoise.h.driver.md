# Purpose
The provided code is a C++ header file for the "linenoise" library, a lightweight line editing library designed to offer essential line editing functionalities without the complexity and size of larger libraries. This file defines the public API for the library, allowing users to integrate line editing capabilities into their applications. The library provides both blocking and non-blocking APIs for line editing, enabling developers to choose the most suitable approach for their needs. It includes structures such as `linenoiseState` and `linenoiseCompletions` to manage the state of line editing and handle command completions, respectively.

The header file outlines various functions for line editing, history management, and completion handling. It includes functions to start and stop line editing, manage command history, and set callbacks for command completion and hints. Additionally, it provides utilities for screen management and character encoding, enhancing the library's flexibility and adaptability to different terminal environments. The use of `extern "C"` ensures compatibility with C++ compilers, allowing the library to be used in both C and C++ projects. Overall, this header file serves as the interface for the linenoise library, offering a concise and efficient solution for line editing in terminal-based applications.
# Imports and Dependencies

---
- `stddef.h`
- `stdlib.h`


# Data Structures

---
### linenoiseState<!-- {{#data_structure:linenoiseState}} -->
- **Type**: `struct`
- **Members**:
    - `in_completion`: Indicates if the user is in completion mode after pressing TAB.
    - `completion_idx`: Holds the index of the next completion suggestion.
    - `ifd`: File descriptor for terminal standard input.
    - `ofd`: File descriptor for terminal standard output.
    - `buf`: Buffer for the edited line.
    - `buflen`: Size of the edited line buffer.
    - `prompt`: Prompt string to display to the user.
    - `plen`: Length of the prompt string.
    - `pos`: Current cursor position in the line.
    - `oldcolpos`: Previous cursor column position for refresh purposes.
    - `len`: Current length of the edited line.
    - `cols`: Number of columns in the terminal.
    - `oldrows`: Number of rows used by the last refreshed line in multiline mode.
    - `history_index`: Index of the history entry currently being edited.
- **Description**: The `linenoiseState` struct is a data structure used to maintain the state of a line editing session in the linenoise library. It contains various fields that track the current editing context, such as whether the user is in completion mode, the current buffer and its size, the prompt and its length, cursor position, and terminal dimensions. This struct is essential for managing the interactive line editing process, including handling user input, displaying prompts, and managing command history.


---
### linenoiseCompletions<!-- {{#data_structure:linenoiseCompletions}} -->
- **Type**: `struct`
- **Members**:
    - `len`: Stores the number of completions currently held in the structure.
    - `cvec`: A pointer to an array of C-style strings representing the completions.
    - `to_free`: Indicates whether the memory allocated for completions should be freed upon destruction.
- **Description**: The `linenoiseCompletions` struct is designed to manage a collection of string completions for a line editing library. It maintains a dynamic array of C-style strings (`cvec`) that represent possible completions, and a count (`len`) of how many completions are stored. The `to_free` boolean flag determines if the memory allocated for these completions should be released when the struct is destroyed, ensuring proper memory management and preventing leaks.
- **Member Functions**:
    - [`linenoiseCompletions::~linenoiseCompletions`](#linenoiseCompletionslinenoiseCompletions)

**Methods**

---
#### linenoiseCompletions::\~linenoiseCompletions<!-- {{#callable:linenoiseCompletions::~linenoiseCompletions}} -->
The destructor `~linenoiseCompletions` is responsible for freeing memory allocated for the completion vector in the `linenoiseCompletions` structure if the `to_free` flag is set to true.
- **Inputs**: None
- **Control Flow**:
    - Check if the `to_free` flag is false; if so, return immediately without freeing memory.
    - Iterate over each element in the `cvec` array up to `len`, freeing each individual string.
    - Free the `cvec` array itself.
- **Output**: The function does not return any value; it performs cleanup by freeing allocated memory.
- **See also**: [`linenoiseCompletions`](#linenoiseCompletions)  (Data Structure)



