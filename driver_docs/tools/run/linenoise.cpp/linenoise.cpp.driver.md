# Purpose
The provided C++ source code is part of a library that implements a line-editing interface for terminal applications, similar to the GNU Readline library. This code is specifically designed for UNIX-like systems and provides functionality for reading and editing lines of text in a terminal, with features such as history management, line completion, and support for multi-line input. The code is structured to handle terminal input and output, manage terminal modes, and process user input efficiently.

Key components of the code include functions for handling terminal escape sequences, managing input history, and providing line-editing capabilities such as moving the cursor, deleting characters, and inserting text. The code also includes support for UTF-8 encoding, allowing it to handle a wide range of characters and symbols. Additionally, the code defines a set of callback functions for custom completion and hinting, enabling developers to extend the library's functionality. The code is designed to be integrated into other applications, providing a robust and flexible interface for terminal-based user input.
# Imports and Dependencies

---
- `linenoise.h`
- `ctype.h`
- `errno.h`
- `poll.h`
- `stdio.h`
- `string.h`
- `sys/file.h`
- `sys/ioctl.h`
- `sys/stat.h`
- `sys/types.h`
- `termios.h`
- `unistd.h`
- `memory`
- `string`
- `vector`


# Global Variables

---
### unsupported\_term
- **Type**: `std::vector<const char *>`
- **Description**: The `unsupported_term` variable is a static vector of constant character pointers, initialized with the strings "dumb", "cons25", and "emacs". This vector is used to store terminal names that are considered unsupported by the program.
- **Use**: This variable is used to check if the current terminal is in the list of unsupported terminals by comparing the terminal name against the entries in this vector.


---
### completionCallback
- **Type**: `linenoiseCompletionCallback*`
- **Description**: The `completionCallback` is a static global variable of type `linenoiseCompletionCallback*`, which is a pointer to a function used for handling command-line completion in the linenoise library. It is initialized to `NULL`, indicating that no completion function is set by default.
- **Use**: This variable is used to store a pointer to a user-defined function that provides completion suggestions when the user presses the tab key during input.


---
### hintsCallback
- **Type**: `linenoiseHintsCallback*`
- **Description**: The `hintsCallback` is a static global variable of type `linenoiseHintsCallback*`, which is a pointer to a function that provides hints to the user during line editing. It is initialized to `NULL`, indicating that no hints function is set by default.
- **Use**: This variable is used to store a callback function that generates hints to be displayed to the user during line editing.


---
### freeHintsCallback
- **Type**: `linenoiseFreeHintsCallback*`
- **Description**: The `freeHintsCallback` is a static global pointer to a function of type `linenoiseFreeHintsCallback`. It is initialized to `NULL`, indicating that no function is currently assigned to it.
- **Use**: This variable is used to store a callback function that is responsible for freeing memory allocated for hints in the linenoise library.


---
### linenoiseNoTTY
- **Type**: `char*`
- **Description**: The `linenoiseNoTTY` function is a static function that returns a pointer to a character array (string). It is designed to handle input when the standard input is not a TTY (terminal), such as when input is redirected from a file or pipe.
- **Use**: This function is used to read input from non-interactive sources, allowing the program to handle input from files or pipes instead of a terminal.


---
### orig\_termios
- **Type**: ``struct termios``
- **Description**: The `orig_termios` variable is a static instance of the `struct termios` data structure. It is used to store the original terminal settings so that they can be restored when the program exits.
- **Use**: This variable is used to save the current terminal settings before modifying them, allowing the program to restore the original settings upon exit.


---
### maskmode
- **Type**: `int`
- **Description**: The `maskmode` variable is a static integer that determines whether input should be masked with asterisks ("***") instead of being displayed as typed. This is typically used for password input to hide the characters being entered.
- **Use**: It is used to toggle the display of input characters as asterisks for password fields.


---
### rawmode
- **Type**: `int`
- **Description**: The `rawmode` variable is a static integer initialized to 0, used to indicate whether the terminal is in raw mode. It is utilized by the `atexit()` function to determine if the terminal settings need to be restored upon program exit.
- **Use**: `rawmode` is used to track the state of the terminal mode, ensuring it is restored to its original settings when the program exits.


---
### mlmode
- **Type**: `int`
- **Description**: The `mlmode` variable is a static integer that indicates whether multi-line mode is enabled or not. By default, it is set to 0, which means single-line mode is active.
- **Use**: This variable is used to determine if the application should operate in multi-line mode, affecting how input is processed and displayed.


---
### atexit\_registered
- **Type**: `int`
- **Description**: The `atexit_registered` variable is a static integer initialized to 0, used to track whether the `atexit` function has been registered. This ensures that the `atexit` function is registered only once during the program's execution.
- **Use**: It is used to prevent multiple registrations of the `atexit` function by checking its value before registering.


---
### history\_max\_len
- **Type**: `int`
- **Description**: The `history_max_len` variable is a static integer that defines the maximum number of entries that can be stored in the command history of the linenoise library. It is initialized to the value of `LINENOISE_DEFAULT_HISTORY_MAX_LEN`, which is set to 100.
- **Use**: This variable is used to limit the number of command history entries that can be stored, ensuring that the history does not exceed a predefined length.


---
### history\_len
- **Type**: `int`
- **Description**: The `history_len` variable is a static integer that keeps track of the number of entries currently stored in the command history. It is initialized to zero, indicating that there are no entries in the history at the start.
- **Use**: This variable is used to manage and track the number of command history entries in the linenoise library.


---
### history
- **Type**: `char **`
- **Description**: The `history` variable is a static global pointer to a pointer of characters, initialized to `NULL`. It is used to store the command history in a dynamically allocated array of strings.
- **Use**: This variable is used to keep track of the command history in the linenoise library, allowing users to navigate through previously entered commands.


---
### wideCharTable
- **Type**: ``unsigned long``
- **Description**: The `wideCharTable` is a static constant two-dimensional array of unsigned long integers, where each element is a pair of hexadecimal values representing Unicode code point ranges. These ranges correspond to wide characters, which typically occupy more space in a terminal or text display.
- **Use**: This variable is used to determine if a given Unicode code point is a wide character by checking if it falls within any of the specified ranges.


---
### wideCharTableSize
- **Type**: ``size_t``
- **Description**: The variable `wideCharTableSize` is a constant of type `size_t` that holds the number of elements in the `wideCharTable` array. This array is a static constant array of unsigned long pairs, representing ranges of Unicode code points that are considered wide characters.
- **Use**: This variable is used to determine the number of wide character ranges in the `wideCharTable` for functions that need to check if a character is wide.


---
### combiningCharTable
- **Type**: ``const unsigned long[]``
- **Description**: The `combiningCharTable` is a static constant array of unsigned long integers that contains Unicode code points representing combining characters. Combining characters are used in Unicode to modify the appearance of the preceding base character, such as adding accents or other diacritical marks.
- **Use**: This array is used to check if a given Unicode code point is a combining character by comparing it against the entries in the table.


---
### combiningCharTableSize
- **Type**: `unsigned long`
- **Description**: The variable `combiningCharTableSize` is a constant of type `unsigned long` that holds the size of the `combiningCharTable` array. It is calculated by dividing the total size of the `combiningCharTable` by the size of a single element in the array.
- **Use**: This variable is used to determine the number of elements in the `combiningCharTable` array, which is likely used for operations involving combining characters in Unicode processing.


---
### prevCharLen
- **Type**: `linenoisePrevCharLen *`
- **Description**: The `prevCharLen` is a static global pointer variable of type `linenoisePrevCharLen *`, which is a function pointer type. It is initialized to point to the function `defaultPrevCharLen`. This function is responsible for determining the length of the previous UTF-8 character in a buffer.
- **Use**: `prevCharLen` is used to calculate the length of the previous character in a UTF-8 encoded string, aiding in text processing and cursor movement operations.


---
### nextCharLen
- **Type**: `linenoiseNextCharLen *`
- **Description**: The `nextCharLen` is a static global pointer variable of type `linenoiseNextCharLen *`, which is initialized to point to the function `defaultNextCharLen`. This function is responsible for determining the length of the next grapheme in a UTF-8 encoded string.
- **Use**: It is used to calculate the length of the next character in a buffer, considering UTF-8 encoding, for text processing in the linenoise library.


---
### readCode
- **Type**: `linenoiseReadCode*`
- **Description**: The `readCode` variable is a static pointer to a function of type `linenoiseReadCode`, which is set to point to the `defaultReadCode` function. This function is responsible for reading a Unicode character from a file descriptor and converting it to a code point.
- **Use**: This variable is used to read and interpret input characters from the terminal, allowing for custom input handling in the linenoise library.


---
### linenoiseEditMore
- **Type**: `const char*`
- **Description**: `linenoiseEditMore` is a constant character pointer that holds a string message indicating a misuse of the API. The message is used to inform the user that when `linenoiseEditFeed()` is called and returns `linenoiseEditMore`, it means the user is still editing the line.
- **Use**: This variable is used as a return value to signal that line editing is still in progress.


# Data Structures

---
### KEY\_ACTION<!-- {{#data_structure:KEY_ACTION}} -->
- **Type**: `enum`
- **Members**:
    - `KEY_NULL`: Represents a null key action with a value of 0.
    - `CTRL_A`: Represents the Ctrl+A key action with a value of 1.
    - `CTRL_B`: Represents the Ctrl+B key action with a value of 2.
    - `CTRL_C`: Represents the Ctrl+C key action with a value of 3.
    - `CTRL_D`: Represents the Ctrl+D key action with a value of 4.
    - `CTRL_E`: Represents the Ctrl+E key action with a value of 5.
    - `CTRL_F`: Represents the Ctrl+F key action with a value of 6.
    - `CTRL_H`: Represents the Ctrl+H key action with a value of 8.
    - `TAB`: Represents the Tab key action with a value of 9.
    - `CTRL_K`: Represents the Ctrl+K key action with a value of 11.
    - `CTRL_L`: Represents the Ctrl+L key action with a value of 12.
    - `ENTER`: Represents the Enter key action with a value of 13.
    - `CTRL_N`: Represents the Ctrl+N key action with a value of 14.
    - `CTRL_P`: Represents the Ctrl+P key action with a value of 16.
    - `CTRL_T`: Represents the Ctrl+T key action with a value of 20.
    - `CTRL_U`: Represents the Ctrl+U key action with a value of 21.
    - `CTRL_W`: Represents the Ctrl+W key action with a value of 23.
    - `ESC`: Represents the Escape key action with a value of 27.
    - `BACKSPACE`: Represents the Backspace key action with a value of 127.
- **Description**: The `KEY_ACTION` enum defines a set of constants representing various keyboard actions, primarily control key combinations and special keys, each associated with a unique integer value. This enumeration is used to map specific key actions to their corresponding integer codes, facilitating the handling of keyboard input in applications.


---
### File<!-- {{#data_structure:File}} -->
- **Type**: `class`
- **Members**:
    - `file`: A pointer to a FILE object, initialized to nullptr, used to manage file operations.
    - `fd`: An integer file descriptor, initialized to -1, used for file locking operations.
- **Description**: The `File` class is a utility for managing file operations in C++. It encapsulates a file pointer (`FILE *`) and a file descriptor (`int`) to handle file opening, locking, and closing. The class provides methods to open a file with a specified mode, lock the file to prevent concurrent access, and ensure proper cleanup by unlocking and closing the file in its destructor. This class abstracts the complexities of file handling and locking, making it easier to manage file resources safely and efficiently.
- **Member Functions**:
    - [`File::open`](../run.cpp.driver.md#Fileopen)
    - [`File::lock`](../run.cpp.driver.md#Filelock)
    - [`File::to_string`](../run.cpp.driver.md#Fileto_string)
    - [`File::~File`](../run.cpp.driver.md#FileFile)
    - [`File::open`](#Fileopen)
    - [`File::lock`](#Filelock)
    - [`File::~File`](#FileFile)

**Methods**

---
#### File::open<!-- {{#callable:File::open}} -->
The `open` method in the `File` class opens a file with the specified filename and mode, and returns a pointer to the `FILE` object.
- **Inputs**:
    - `filename`: A `std::string` representing the name of the file to be opened.
    - `mode`: A `const char*` representing the mode in which to open the file, such as "r" for read, "w" for write, etc.
- **Control Flow**:
    - The method calls `fopen` with the `filename` converted to a C-style string using `c_str()` and the specified `mode`.
    - The result of `fopen`, which is a pointer to a `FILE` object, is assigned to the `file` member variable of the `File` class.
    - The method returns the `file` pointer.
- **Output**: A pointer to a `FILE` object representing the opened file, or `nullptr` if the file could not be opened.
- **See also**: [`File`](#File)  (Data Structure)


---
#### File::lock<!-- {{#callable:File::lock}} -->
The `lock` function attempts to acquire an exclusive, non-blocking lock on a file associated with the `File` class instance.
- **Inputs**: None
- **Control Flow**:
    - Check if the `file` member is not null, indicating that a file is open.
    - Retrieve the file descriptor `fd` from the `file` using `fileno`.
    - Attempt to lock the file using `flock` with `LOCK_EX | LOCK_NB` flags for exclusive, non-blocking lock.
    - If `flock` fails, set `fd` to -1 and return 1 to indicate failure.
    - Return 0 to indicate success if the lock is acquired.
- **Output**: Returns 0 if the lock is successfully acquired, otherwise returns 1 if the lock cannot be acquired.
- **See also**: [`File`](#File)  (Data Structure)


---
#### File::\~File<!-- {{#callable:File::~File}} -->
The destructor `~File()` releases any file lock and closes the file if they are open.
- **Inputs**: None
- **Control Flow**:
    - Check if the file descriptor `fd` is valid (i.e., `fd >= 0`).
    - If valid, release the file lock using `flock(fd, LOCK_UN)`.
    - Check if the file pointer `file` is not null.
    - If not null, close the file using `fclose(file)`.
- **Output**: The function does not return any value as it is a destructor.
- **See also**: [`File`](#File)  (Data Structure)



---
### ESC\_TYPE<!-- {{#data_structure:ESC_TYPE}} -->
- **Type**: `enum`
- **Members**:
    - `ESC_NULL`: Represents a null or no-operation escape sequence.
    - `ESC_DELETE`: Represents the delete key escape sequence.
    - `ESC_UP`: Represents the up arrow key escape sequence.
    - `ESC_DOWN`: Represents the down arrow key escape sequence.
    - `ESC_RIGHT`: Represents the right arrow key escape sequence.
    - `ESC_LEFT`: Represents the left arrow key escape sequence.
    - `ESC_HOME`: Represents the home key escape sequence.
    - `ESC_END`: Represents the end key escape sequence.
- **Description**: The `ESC_TYPE` enum defines a set of constants representing various escape sequences used for handling keyboard input in terminal applications. Each constant corresponds to a specific key or action, such as arrow keys or the delete key, allowing the program to interpret and respond to user input appropriately.


# Functions

---
### lndebug<!-- {{#callable:lndebug}} -->
The `lndebug` function logs formatted debug messages to a specified file.
- **Inputs**:
    - `fmt`: A format string that specifies how subsequent arguments are converted for output.
    - `...`: A variable number of arguments that are formatted according to the `fmt` string.
- **Control Flow**:
    - The function checks if the static `file` object has been initialized; if not, it opens the log file in append mode.
    - If the file is successfully opened, it initializes a `va_list` to handle the variable arguments.
    - The formatted output is written to the file using `vfprintf`, and the file is flushed to ensure all data is written.
- **Output**: The function does not return a value; it writes formatted debug messages directly to the log file.


---
### prevUtf8CodePointLen<!-- {{#callable:prevUtf8CodePointLen}} -->
Calculates the length of the previous UTF-8 code point in a given buffer.
- **Inputs**:
    - `buf`: A pointer to a character buffer containing UTF-8 encoded data.
    - `pos`: An integer representing the current position in the buffer from which to calculate the previous UTF-8 code point length.
- **Control Flow**:
    - The function initializes an integer `end` to the value of `pos` and decrements `pos` by 1.
    - It enters a while loop that continues as long as `pos` is non-negative and the byte at `buf[pos]` is a continuation byte (i.e., it has the format 10xxxxxx).
    - Within the loop, `pos` is decremented to move backwards through the buffer until a non-continuation byte is found.
    - Finally, the function returns the difference between `end` and `pos`, which represents the length of the previous UTF-8 code point.
- **Output**: Returns the length of the previous UTF-8 code point as a size_t value.


---
### utf8BytesToCodePoint<!-- {{#callable:utf8BytesToCodePoint}} -->
Converts a sequence of UTF-8 encoded bytes into a Unicode code point.
- **Inputs**:
    - `buf`: A pointer to the buffer containing the UTF-8 encoded bytes.
    - `len`: The length of the buffer in bytes.
    - `cp`: A pointer to an integer where the resulting Unicode code point will be stored.
- **Control Flow**:
    - Checks if the length of the buffer is greater than zero.
    - Examines the first byte to determine the number of bytes in the UTF-8 character.
    - If the first byte indicates a single-byte character (0xxxxxxx), it assigns the byte to the code point and returns 1.
    - If the first byte indicates a two-byte character (110xxxxx), it checks if there are at least two bytes available, then combines the first two bytes to form the code point and returns 2.
    - If the first byte indicates a three-byte character (1110xxxx), it checks for three bytes, combines them, and returns 3.
    - If the first byte indicates a four-byte character (11110xxx), it checks for four bytes, combines them, and returns 4.
    - If none of the conditions are met, it returns 0, indicating an invalid or incomplete UTF-8 sequence.
- **Output**: Returns the number of bytes consumed from the buffer to form the code point, or 0 if the input is invalid.


---
### isWideChar<!-- {{#callable:isWideChar}} -->
Checks if a given Unicode code point is a wide character.
- **Inputs**:
    - `cp`: An unsigned long representing the Unicode code point to be checked.
- **Control Flow**:
    - Iterates through a predefined table of wide character ranges.
    - For each range, checks if the input code point is within the range.
    - Returns true if the code point is found within any range, otherwise returns false.
- **Output**: Returns a boolean value indicating whether the code point is a wide character.


---
### isCombiningChar<!-- {{#callable:isCombiningChar}} -->
Checks if a given Unicode code point is a combining character.
- **Inputs**:
    - `cp`: An unsigned long representing the Unicode code point to be checked.
- **Control Flow**:
    - Iterates through the `combiningCharTable` array, which contains known combining character codes.
    - For each code in the table, it checks if the code is greater than the input code point `cp`, returning false if so.
    - If a match is found (i.e., the code equals `cp`), it returns true.
    - If no matches are found after checking all entries, it returns false.
- **Output**: Returns true if `cp` is a combining character, otherwise returns false.


---
### defaultPrevCharLen<!-- {{#callable:defaultPrevCharLen}} -->
Calculates the length of the previous character in a UTF-8 encoded string, considering combining characters.
- **Inputs**:
    - `buf`: A pointer to a character array (C-style string) containing the UTF-8 encoded text.
    - `buf_len`: The length of the buffer, although it is not used in the function.
    - `pos`: The current position in the buffer from which to calculate the previous character length.
    - `col_len`: A pointer to a size_t variable where the width of the character (1 or 2) will be stored if not NULL.
- **Control Flow**:
    - Initializes a variable `end` to store the original position.
    - Enters a loop that continues as long as `pos` is greater than 0.
    - Calculates the length of the previous UTF-8 code point using [`prevUtf8CodePointLen`](#prevUtf8CodePointLen).
    - Decreases `pos` by the length of the previous code point.
    - Converts the bytes at the new position to a Unicode code point using [`utf8BytesToCodePoint`](#utf8BytesToCodePoint).
    - Checks if the code point is not a combining character.
    - If `col_len` is not NULL, sets it to 2 if the character is wide, otherwise sets it to 1.
    - Returns the difference between `end` and the new `pos` to indicate the length of the character.
- **Output**: Returns the length of the previous character in bytes, or 0 if no valid character is found.
- **Functions called**:
    - [`prevUtf8CodePointLen`](#prevUtf8CodePointLen)
    - [`utf8BytesToCodePoint`](#utf8BytesToCodePoint)
    - [`isCombiningChar`](#isCombiningChar)
    - [`isWideChar`](#isWideChar)


---
### defaultNextCharLen<!-- {{#callable:defaultNextCharLen}} -->
Calculates the length of the next character in a UTF-8 encoded string, considering combining characters.
- **Inputs**:
    - `buf`: A pointer to a character buffer containing the UTF-8 encoded string.
    - `buf_len`: The length of the buffer in bytes.
    - `pos`: The current position in the buffer from which to start calculating the next character length.
    - `col_len`: A pointer to a size_t variable where the column length of the character will be stored, if not NULL.
- **Control Flow**:
    - Initializes a variable `beg` to the current position `pos`.
    - Calls [`utf8BytesToCodePoint`](#utf8BytesToCodePoint) to get the length of the character at the current position and store its code point in `cp`.
    - Checks if the character is a combining character using [`isCombiningChar`](#isCombiningChar). If it is, the function returns 0.
    - If `col_len` is not NULL, it sets it to 2 if the character is wide, otherwise to 1.
    - Increments `pos` by the length of the character.
    - Enters a loop that continues until `pos` is less than `buf_len`.
    - In each iteration, it retrieves the next character's length and code point.
    - If the character is not a combining character, it returns the difference between `pos` and `beg`.
    - If all characters are combining characters, it returns the difference between `pos` and `beg` after the loop.
- **Output**: Returns the length of the next character in bytes, or the number of bytes processed if all characters are combining characters.
- **Functions called**:
    - [`utf8BytesToCodePoint`](#utf8BytesToCodePoint)
    - [`isCombiningChar`](#isCombiningChar)
    - [`isWideChar`](#isWideChar)


---
### defaultReadCode<!-- {{#callable:defaultReadCode}} -->
Reads a UTF-8 encoded character from a file descriptor into a buffer and converts it to a Unicode code point.
- **Inputs**:
    - `fd`: An integer file descriptor from which to read the UTF-8 encoded character.
    - `buf`: A character buffer where the read bytes will be stored.
    - `buf_len`: The length of the buffer, which must be at least 1.
    - `cp`: A pointer to an integer where the resulting Unicode code point will be stored.
- **Control Flow**:
    - Checks if the buffer length is less than 1, returning -1 if true.
    - Reads one byte from the file descriptor into the buffer.
    - If the read is unsuccessful (nread <= 0), it returns the number of bytes read.
    - Determines the number of bytes to read based on the first byte's value, which indicates the UTF-8 character length.
    - Reads additional bytes into the buffer if necessary, based on the first byte's value.
    - If the first byte does not match any valid UTF-8 prefix, it returns -1.
    - Calls [`utf8BytesToCodePoint`](#utf8BytesToCodePoint) to convert the UTF-8 bytes in the buffer to a Unicode code point and returns the result.
- **Output**: Returns the number of bytes read from the file descriptor, or -1 on error.
- **Functions called**:
    - [`utf8BytesToCodePoint`](#utf8BytesToCodePoint)


---
### linenoiseSetEncodingFunctions<!-- {{#callable:linenoiseSetEncodingFunctions}} -->
Sets custom encoding functions for handling character lengths and reading input in the `linenoise` library.
- **Inputs**:
    - `prevCharLenFunc`: A pointer to a function that calculates the length of the previous character in a UTF-8 encoded string.
    - `nextCharLenFunc`: A pointer to a function that calculates the length of the next character in a UTF-8 encoded string.
    - `readCodeFunc`: A pointer to a function that reads a Unicode character from a file descriptor.
- **Control Flow**:
    - The function assigns the provided function pointers to the static variables `prevCharLen`, `nextCharLen`, and `readCode`.
    - No conditional logic or loops are present, making the function straightforward and direct.
- **Output**: The function does not return a value; it modifies the internal state of the `linenoise` library by setting the encoding functions.


---
### linenoiseMaskModeEnable<!-- {{#callable:linenoiseMaskModeEnable}} -->
Enables mask mode for the terminal input, displaying asterisks instead of the actual input characters.
- **Inputs**: None
- **Control Flow**:
    - The function sets the global variable `maskmode` to 1, indicating that mask mode is enabled.
- **Output**: The function does not return any value; it modifies the state of the `maskmode` variable.


---
### linenoiseMaskModeDisable<!-- {{#callable:linenoiseMaskModeDisable}} -->
Disables the mask mode for input, allowing the actual characters to be displayed instead of asterisks.
- **Inputs**: None
- **Control Flow**:
    - The function sets the global variable `maskmode` to 0, which indicates that mask mode is disabled.
- **Output**: The function does not return any value; it modifies the state of the `maskmode` variable.


---
### linenoiseSetMultiLine<!-- {{#callable:linenoiseSetMultiLine}} -->
Sets the multi-line mode for the linenoise input.
- **Inputs**:
    - `ml`: An integer value indicating whether to enable (1) or disable (0) multi-line mode.
- **Control Flow**:
    - The function directly assigns the input integer 'ml' to the static variable 'mlmode'.
    - No conditional logic or loops are present in the function.
- **Output**: The function does not return a value; it modifies the internal state of the linenoise library.


---
### isUnsupportedTerm<!-- {{#callable:isUnsupportedTerm}} -->
Checks if the current terminal type is unsupported based on the TERM environment variable.
- **Inputs**: None
- **Control Flow**:
    - Retrieves the value of the TERM environment variable using `getenv`.
    - If the TERM variable is NULL, the function returns 0, indicating no unsupported terminal.
    - Iterates through the `unsupported_term` vector to check if the TERM matches any unsupported types using `strcasecmp`.
    - If a match is found, the function returns 1, indicating an unsupported terminal.
    - If no match is found after checking all entries, the function returns 0.
- **Output**: Returns 1 if the terminal type is unsupported, 0 otherwise.


---
### enableRawMode<!-- {{#callable:enableRawMode}} -->
Enables raw mode for terminal input, modifying terminal settings to allow for unprocessed input.
- **Inputs**:
    - `fd`: The file descriptor for the terminal to be set to raw mode.
- **Control Flow**:
    - Checks if the standard input is a terminal using `isatty`.
    - Registers a cleanup function with `atexit` if not already registered.
    - Retrieves the current terminal attributes using `tcgetattr`.
    - Modifies the terminal attributes to set raw mode by adjusting input, output, control, and local modes.
    - Sets the minimum number of bytes and timer for reading input.
    - Applies the modified attributes using `tcsetattr`.
    - Handles errors by jumping to the fatal label, setting `errno` to `ENOTTY` and returning -1.
- **Output**: Returns 0 on success, or -1 on failure with `errno` set to indicate the error.


---
### disableRawMode<!-- {{#callable:disableRawMode}} -->
Disables raw mode for the terminal by restoring the original terminal settings.
- **Inputs**:
    - `fd`: An integer file descriptor representing the terminal to which the raw mode settings apply.
- **Control Flow**:
    - The function first checks if the `rawmode` flag is set to indicate that raw mode is currently enabled.
    - If `rawmode` is true, it attempts to restore the terminal settings using `tcsetattr` with the original terminal attributes stored in `orig_termios`.
    - If the call to `tcsetattr` is successful (returns -1), it sets `rawmode` to 0, indicating that raw mode has been disabled.
- **Output**: The function does not return a value; it modifies the state of the terminal by restoring its original settings.


---
### getCursorPosition<!-- {{#callable:getCursorPosition}} -->
The `getCursorPosition` function retrieves the current cursor position in a terminal by sending an escape sequence and reading the response.
- **Inputs**:
    - `ifd`: The input file descriptor from which to read the terminal response.
    - `ofd`: The output file descriptor to which the escape sequence is written.
- **Control Flow**:
    - The function first sends the escape sequence '[6n' to the terminal to request the cursor position.
    - It then reads the response character by character until it receives the character 'R' or fills the buffer.
    - After reading, it checks if the response starts with the expected escape sequence format.
    - If the format is correct, it parses the row and column values from the response.
    - Finally, it returns the column number or -1 if any error occurs during the process.
- **Output**: The function returns the column position of the cursor on success, or -1 if an error occurs.


---
### getColumns<!-- {{#callable:getColumns}} -->
The `getColumns` function retrieves the number of columns in the terminal window.
- **Inputs**:
    - `ifd`: The input file descriptor used to read from the terminal.
    - `ofd`: The output file descriptor used to write to the terminal.
- **Control Flow**:
    - The function first attempts to get the terminal size using the `ioctl` system call with `TIOCGWINSZ`.
    - If `ioctl` fails or returns zero columns, it tries to determine the column size by moving the cursor to the right margin and checking the cursor position.
    - It saves the current cursor position, moves to the right margin, and retrieves the column position.
    - If successful, it restores the cursor to its original position and returns the number of columns.
    - If all attempts fail, it defaults to returning 80 columns.
- **Output**: Returns the number of columns in the terminal, or 80 if it cannot be determined.
- **Functions called**:
    - [`getCursorPosition`](#getCursorPosition)


---
### linenoiseClearScreen<!-- {{#callable:linenoiseClearScreen}} -->
Clears the terminal screen and moves the cursor to the home position.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the write operation to standard output is successful.
    - If the write operation fails, it does nothing further to avoid warnings.
- **Output**: The function does not return any value; it performs an action on the terminal.


---
### linenoiseBeep<!-- {{#callable:linenoiseBeep}} -->
This function sends a beep sound to the standard error output.
- **Inputs**: None
- **Control Flow**:
    - The function uses `fprintf` to write the ASCII bell character (represented by `` or ``) to the standard error stream.
    - It then calls `fflush` to ensure that the output buffer is flushed, making the beep sound immediate.
- **Output**: The function does not return any value; it produces a side effect of generating a beep sound in the terminal.


---
### refreshLineWithCompletion<!-- {{#callable:refreshLineWithCompletion}} -->
The `refreshLineWithCompletion` function updates the displayed line in the terminal with a completion suggestion based on the current input.
- **Inputs**:
    - `ls`: A pointer to a `linenoiseState` structure that holds the current state of the line being edited.
    - `lc`: A pointer to a `linenoiseCompletions` structure that contains the list of possible completions; if NULL, a completion callback is invoked to populate it.
    - `flags`: An integer representing flags that control the refresh behavior, such as whether to clean the old prompt or rewrite it.
- **Control Flow**:
    - If the `lc` parameter is NULL, the function calls a completion callback to populate a local completion table.
    - If the current completion index is valid (less than the number of completions), it saves the current state of `ls`, updates it with the selected completion, and refreshes the line display.
    - If the completion index is out of bounds, it simply refreshes the line without any completion.
    - If the local completion table was created within the function, it ensures that it is not freed after use.
- **Output**: The function does not return a value; it directly modifies the terminal display to show the updated line with the completion suggestion.
- **Functions called**:
    - [`refreshLineWithFlags`](#refreshLineWithFlags)


---
### readEscapeSequence<!-- {{#callable:readEscapeSequence}} -->
Reads an escape sequence from the terminal input and returns the corresponding action type.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the state of the line editing session, including file descriptors for input and output.
- **Control Flow**:
    - Initializes a `pollfd` structure to check for input availability on the file descriptor associated with the `linenoiseState`.
    - Calls `poll` with a timeout of 1 millisecond to check if there is data to read.
    - If no data is available or an error occurs, returns `ESC_NULL`.
    - Reads the first byte of the escape sequence and checks for errors.
    - Reads the second byte of the escape sequence and checks for errors.
    - If the first byte is '[', checks the second byte for specific control characters and handles extended sequences if necessary.
    - If the first byte is 'O', checks for specific control characters.
    - Returns the corresponding `ESC_TYPE` based on the recognized escape sequence.
- **Output**: Returns an `ESC_TYPE` value indicating the action associated with the escape sequence, such as cursor movement or deletion, or `ESC_NULL` if the sequence is unrecognized or an error occurred.


---
### completeLine<!-- {{#callable:completeLine}} -->
The `completeLine` function handles tab completion and updates the input buffer based on user key presses.
- **Inputs**:
    - `struct linenoiseState * ls`: Pointer to the state structure containing the current input buffer and completion state.
    - `int keypressed`: The key that was pressed by the user, represented as an integer.
    - `ESC_TYPE esc_type`: An enumeration indicating the type of escape sequence, if any, that was detected.
- **Control Flow**:
    - Calls the `completionCallback` to populate the `linenoiseCompletions` structure with possible completions based on the current buffer.
    - If no completions are found, it beeps and resets the completion state.
    - If the TAB key is pressed, it toggles the completion mode and cycles through the available completions.
    - If the ESC key is pressed and no escape type is detected, it refreshes the original buffer.
    - If a valid completion index is set, it updates the buffer with the selected completion and refreshes the display.
- **Output**: Returns the last read character, which may be modified based on the completion logic.
- **Functions called**:
    - [`linenoiseBeep`](#linenoiseBeep)
    - [`refreshLine`](#refreshLine)
    - [`refreshLineWithCompletion`](#refreshLineWithCompletion)


---
### linenoiseSetCompletionCallback<!-- {{#callable:linenoiseSetCompletionCallback}} -->
Sets the completion callback function for the linenoise library.
- **Inputs**:
    - `fn`: A pointer to a function of type `linenoiseCompletionCallback` that will be called for tab-completion.
- **Control Flow**:
    - The function directly assigns the provided callback function pointer `fn` to the static variable `completionCallback`.
    - No conditional logic or loops are present, making the function straightforward and efficient.
- **Output**: This function does not return a value; it modifies the state of the `completionCallback` variable.


---
### linenoiseSetHintsCallback<!-- {{#callable:linenoiseSetHintsCallback}} -->
Sets the hints callback function for the linenoise library.
- **Inputs**:
    - `fn`: A pointer to a function of type `linenoiseHintsCallback` that will be called to provide hints based on user input.
- **Control Flow**:
    - The function assigns the provided callback function pointer `fn` to the global variable `hintsCallback`.
    - No additional logic or conditions are present in the function.
- **Output**: The function does not return a value; it simply sets the hints callback for future use.


---
### linenoiseSetFreeHintsCallback<!-- {{#callable:linenoiseSetFreeHintsCallback}} -->
Sets a callback function to free hints returned by the hints callback.
- **Inputs**:
    - `fn`: A pointer to a function of type `linenoiseFreeHintsCallback` that will be called to free hints.
- **Control Flow**:
    - The function directly assigns the provided callback function pointer to the global variable `freeHintsCallback`.
    - No conditional logic or loops are present in the function.
- **Output**: The function does not return a value; it simply updates the global `freeHintsCallback` variable.


---
### linenoiseAddCompletion<!-- {{#callable:linenoiseAddCompletion}} -->
Adds a new completion string to the `linenoiseCompletions` structure.
- **Inputs**:
    - `lc`: A pointer to a `linenoiseCompletions` structure that holds the current list of completions.
    - `str`: A pointer to a null-terminated string representing the completion to be added.
- **Control Flow**:
    - Calculates the length of the input string `str` using `strlen`.
    - Allocates memory for a copy of the string using `std::make_unique`.
    - Checks if the memory allocation was successful; if not, the function returns early.
    - Copies the string into the allocated memory using `memcpy`.
    - Reallocates the `cvec` array in the `linenoiseCompletions` structure to accommodate the new completion.
    - Checks if the reallocation was successful; if not, the function returns early.
    - Updates the `cvec` pointer in the `linenoiseCompletions` structure and increments the count of completions.
- **Output**: The function does not return a value; it modifies the `linenoiseCompletions` structure in place.


---
### columnPos<!-- {{#callable:columnPos}} -->
Calculates the column position in a buffer up to a specified position.
- **Inputs**:
    - `buf`: A pointer to a character buffer containing the text to analyze.
    - `buf_len`: The length of the buffer in bytes.
    - `pos`: The position in the buffer up to which the column position is calculated.
- **Control Flow**:
    - Initializes a variable `ret` to store the column count and `off` to track the current offset.
    - Enters a while loop that continues until `off` is less than `pos`.
    - Within the loop, it calls `nextCharLen` to determine the length of the next character and its column width.
    - Updates `off` by adding the length of the character and increments `ret` by the column width.
    - Finally, returns the total column count stored in `ret`.
- **Output**: Returns the total number of columns from the start of the buffer to the specified position.


---
### refreshShowHints<!-- {{#callable:refreshShowHints}} -->
The `refreshShowHints` function updates a string with hints based on the current input state and terminal width.
- **Inputs**:
    - `ab`: A reference to a `std::string` that will be modified to include the hint text.
    - `l`: A pointer to a `linenoiseState` structure that contains the current state of the input line.
    - `pcollen`: An integer representing the length of the prompt text, used to calculate available space for hints.
- **Control Flow**:
    - Calculates the total column length by adding the prompt length and the current input length.
    - Checks if a hints callback is set and if there is enough space in the terminal to display hints.
    - Retrieves the hint text from the callback, along with its formatting options (color and bold).
    - Determines the maximum length of the hint that can be displayed based on available space.
    - Formats the hint with ANSI escape codes if color or bold options are specified.
    - Appends the formatted hint to the `ab` string, followed by the hint text itself.
    - Calls the free hints callback to release any allocated memory for the hint.
- **Output**: The function modifies the `ab` string to include the hint text, formatted according to the specified color and boldness, if applicable.
- **Functions called**:
    - [`columnPos`](#columnPos)


---
### isAnsiEscape<!-- {{#callable:isAnsiEscape}} -->
Checks if a given buffer contains an ANSI escape sequence and returns its length.
- **Inputs**:
    - `buf`: A pointer to a character array (C-string) that contains the data to be checked for ANSI escape sequences.
    - `buf_len`: The length of the buffer `buf`, indicating how many characters to check.
    - `len`: A pointer to a size_t variable where the length of the detected ANSI escape sequence will be stored.
- **Control Flow**:
    - The function first checks if the buffer length is greater than 2 and if the first two characters match the ANSI escape sequence prefix '\033['.
    - If the initial check passes, it enters a loop to iterate through the buffer starting from the third character.
    - Within the loop, it checks each character against a set of known ANSI escape sequence terminators (A, B, C, D, E, F, G, H, J, K, S, T, f, m).
    - If a match is found, it updates the length pointer and returns 1, indicating a valid ANSI escape sequence was found.
    - If no valid sequence is found by the end of the buffer, the function returns 0.
- **Output**: Returns 1 if an ANSI escape sequence is found, otherwise returns 0.


---
### promptTextColumnLen<!-- {{#callable:promptTextColumnLen}} -->
Calculates the column length of a prompt text while ignoring ANSI escape sequences.
- **Inputs**:
    - `prompt`: A pointer to a null-terminated string representing the prompt text.
    - `plen`: The length of the prompt text, excluding the null terminator.
- **Control Flow**:
    - Initializes a buffer to store characters from the prompt text.
    - Iterates through the prompt text using an offset until the end of the specified length.
    - Checks if the current character is part of an ANSI escape sequence using the [`isAnsiEscape`](#isAnsiEscape) function.
    - If an escape sequence is detected, it skips the length of the escape sequence.
    - If not, it adds the current character to the buffer and increments the buffer length.
    - Finally, it calculates and returns the total column position using the [`columnPos`](#columnPos) function.
- **Output**: Returns the total number of columns occupied by the prompt text, accounting for any ANSI escape sequences.
- **Functions called**:
    - [`isAnsiEscape`](#isAnsiEscape)
    - [`columnPos`](#columnPos)


---
### refreshSingleLine<!-- {{#callable:refreshSingleLine}} -->
The `refreshSingleLine` function updates the display of a single line in the terminal based on the current state of the input buffer and cursor position.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line editing, including the prompt, buffer, cursor position, and terminal settings.
    - `flags`: An integer representing flags that dictate the behavior of the refresh operation, such as whether to clean the old prompt or rewrite the prompt.
- **Control Flow**:
    - Calculate the length of the prompt text and the current buffer content.
    - Adjust the buffer content and cursor position if they exceed the terminal width.
    - Construct a string to represent the updated line, including the prompt and current buffer content.
    - If the `REFRESH_WRITE` flag is set, append the prompt and buffer content to the output string.
    - Erase the line to the right of the cursor position.
    - If the `REFRESH_WRITE` flag is set, move the cursor back to its original position.
    - Write the constructed string to the terminal output file descriptor.
- **Output**: The function does not return a value; instead, it directly writes the updated line to the terminal output.
- **Functions called**:
    - [`promptTextColumnLen`](#promptTextColumnLen)
    - [`columnPos`](#columnPos)
    - [`refreshShowHints`](#refreshShowHints)


---
### columnPosForMultiLine<!-- {{#callable:columnPosForMultiLine}} -->
Calculates the column position for a given character position in a multi-line buffer, taking into account the width of characters and the specified number of columns.
- **Inputs**:
    - `buf`: A pointer to a character buffer containing the text to be processed.
    - `buf_len`: The length of the buffer, indicating how many characters are available for processing.
    - `pos`: The position in the buffer for which the column position is to be calculated.
    - `cols`: The total number of columns available for display in the terminal.
    - `ini_pos`: The initial column position from which to start the calculation.
- **Control Flow**:
    - Initialize the return value `ret` and the current column width `colwid` with `ini_pos`.
    - Iterate through the buffer until the end of the buffer is reached.
    - For each character, determine its length using `nextCharLen` and calculate the difference between the current column width plus the character length and the total columns.
    - If the difference is positive, increment `ret` by the difference and set `colwid` to the current character's width.
    - If the difference is zero, reset `colwid` to zero.
    - If the difference is negative, add the character's width to `colwid`.
    - Break the loop if the current offset reaches or exceeds `pos`.
    - Increment the offset by the length of the character processed and add the character's width to `ret`.
- **Output**: Returns the calculated column position as a size_t value.


---
### refreshMultiLine<!-- {{#callable:refreshMultiLine}} -->
The `refreshMultiLine` function updates the display of a multi-line input buffer in a terminal, handling cursor positioning and line clearing.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the input line, including the buffer, cursor position, and prompt.
    - `flags`: An integer representing flags that dictate the refresh behavior, such as whether to clean the screen or write the current buffer.
- **Control Flow**:
    - Calculate the length of the prompt and the current buffer to determine how many rows are needed for display.
    - If the `REFRESH_CLEAN` flag is set, clear the previous lines used by the input buffer.
    - If the `REFRESH_ALL` flag is set, clear the top line of the terminal.
    - Calculate the new cursor position based on the current buffer and prompt.
    - If the `REFRESH_WRITE` flag is set, write the prompt and the current buffer content to the terminal, handling masking for sensitive input.
    - Adjust the cursor position based on the new calculated position and the number of rows.
- **Output**: The function does not return a value but writes the updated display content directly to the terminal.
- **Functions called**:
    - [`promptTextColumnLen`](#promptTextColumnLen)
    - [`columnPosForMultiLine`](#columnPosForMultiLine)
    - [`refreshShowHints`](#refreshShowHints)


---
### refreshLineWithFlags<!-- {{#callable:refreshLineWithFlags}} -->
Refreshes the display of the current line in either single or multi-line mode based on the provided flags.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line being edited.
    - `flags`: An integer representing flags that dictate how the line should be refreshed, such as whether to clean the old prompt or rewrite the prompt.
- **Control Flow**:
    - Checks if multi-line mode is enabled by evaluating the `mlmode` variable.
    - If multi-line mode is enabled, it calls the [`refreshMultiLine`](#refreshMultiLine) function with the provided `linenoiseState` and flags.
    - If multi-line mode is not enabled, it calls the [`refreshSingleLine`](#refreshSingleLine) function with the same parameters.
- **Output**: The function does not return a value; it directly modifies the terminal display to reflect the current state of the line being edited.
- **Functions called**:
    - [`refreshMultiLine`](#refreshMultiLine)
    - [`refreshSingleLine`](#refreshSingleLine)


---
### refreshLine<!-- {{#callable:refreshLine}} -->
The `refreshLine` function updates the display of the current line in the terminal by calling [`refreshLineWithFlags`](#refreshLineWithFlags) with the `REFRESH_ALL` flag.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line being edited, including the buffer, cursor position, and other relevant information.
- **Control Flow**:
    - The function directly calls [`refreshLineWithFlags`](#refreshLineWithFlags) with the provided `linenoiseState` pointer and the `REFRESH_ALL` flag.
    - The `REFRESH_ALL` flag indicates that both the old prompt should be cleaned and the new prompt should be written to the terminal.
- **Output**: The function does not return a value; it performs an action to refresh the terminal display based on the current state of the line.
- **Functions called**:
    - [`refreshLineWithFlags`](#refreshLineWithFlags)


---
### linenoiseHide<!-- {{#callable:linenoiseHide}} -->
The `linenoiseHide` function clears the current line in the terminal based on the mode of operation.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the state of the line editing.
- **Control Flow**:
    - Checks if `mlmode` (multi-line mode) is enabled.
    - If `mlmode` is true, it calls [`refreshMultiLine`](#refreshMultiLine) to clear the current line.
    - If `mlmode` is false, it calls [`refreshSingleLine`](#refreshSingleLine) to clear the current line.
- **Output**: The function does not return a value; it performs an action to refresh the terminal display by clearing the current line.
- **Functions called**:
    - [`refreshMultiLine`](#refreshMultiLine)
    - [`refreshSingleLine`](#refreshSingleLine)


---
### linenoiseShow<!-- {{#callable:linenoiseShow}} -->
The `linenoiseShow` function refreshes the display of the current line being edited, either showing completion suggestions or the line itself based on the state of the completion.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line editing session, including the buffer, cursor position, and completion status.
- **Control Flow**:
    - The function first checks if the `in_completion` flag in the `linenoiseState` structure is set.
    - If `in_completion` is true, it calls [`refreshLineWithCompletion`](#refreshLineWithCompletion) to display the current line with completion suggestions.
    - If `in_completion` is false, it calls [`refreshLineWithFlags`](#refreshLineWithFlags) to refresh the line display without completion.
- **Output**: The function does not return a value; it directly modifies the terminal display to show the current line or completion suggestions.
- **Functions called**:
    - [`refreshLineWithCompletion`](#refreshLineWithCompletion)
    - [`refreshLineWithFlags`](#refreshLineWithFlags)


---
### linenoiseEditInsert<!-- {{#callable:linenoiseEditInsert}} -->
Inserts a string into the current position of the editing buffer and updates the display accordingly.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line editing session.
    - `cbuf`: A pointer to a character buffer containing the string to be inserted into the editing buffer.
    - `clen`: An integer representing the length of the string to be inserted.
- **Control Flow**:
    - Checks if the total length of the current buffer plus the length of the new string does not exceed the maximum buffer length.
    - If the insertion position is at the end of the current buffer, it directly copies the new string into the buffer and updates the position and length.
    - If the insertion position is not at the end, it shifts the existing content to make space for the new string, then inserts the new string.
    - After insertion, it checks if a full line refresh is necessary based on the current editing mode and whether hints are being displayed.
    - If a full refresh is not needed, it writes the new string or a placeholder character to the output directly.
- **Output**: Returns 0 on success, or -1 if there was an error writing to the terminal.
- **Functions called**:
    - [`promptTextColumnLen`](#promptTextColumnLen)
    - [`columnPos`](#columnPos)
    - [`refreshLine`](#refreshLine)


---
### linenoiseEditMoveLeft<!-- {{#callable:linenoiseEditMoveLeft}} -->
Moves the cursor one character to the left in the input buffer.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the input line, including the position of the cursor.
- **Control Flow**:
    - Checks if the current cursor position (`l->pos`) is greater than 0.
    - If true, it calculates the length of the previous character using `prevCharLen` and decreases the cursor position by that length.
    - Calls [`refreshLine`](#refreshLine) to update the display of the input line.
- **Output**: The function does not return a value; it modifies the cursor position in the `linenoiseState` structure and refreshes the displayed line.
- **Functions called**:
    - [`refreshLine`](#refreshLine)


---
### linenoiseEditMoveRight<!-- {{#callable:linenoiseEditMoveRight}} -->
Moves the cursor one character to the right in the input buffer if it is not already at the end.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line editing, including the position of the cursor and the length of the input buffer.
- **Control Flow**:
    - Checks if the current cursor position (`l->pos`) is not equal to the length of the input buffer (`l->len`).
    - If the cursor is not at the end, it calculates the length of the next character using the `nextCharLen` function.
    - Updates the cursor position by adding the length of the next character to `l->pos`.
    - Calls the [`refreshLine`](#refreshLine) function to update the display of the input line.
- **Output**: The function does not return a value; it modifies the state of the `linenoiseState` structure and refreshes the line display.
- **Functions called**:
    - [`refreshLine`](#refreshLine)


---
### linenoiseEditMoveHome<!-- {{#callable:linenoiseEditMoveHome}} -->
Moves the cursor to the beginning of the current line in the `linenoiseState` structure.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line editing session.
- **Control Flow**:
    - Checks if the current cursor position (`l->pos`) is not already at the beginning of the line (0).
    - If the cursor is not at the beginning, it sets the cursor position to 0.
    - Calls the [`refreshLine`](#refreshLine) function to update the display of the line after moving the cursor.
- **Output**: The function does not return a value; it modifies the state of the `linenoiseState` structure and refreshes the line display.
- **Functions called**:
    - [`refreshLine`](#refreshLine)


---
### linenoiseEditMoveEnd<!-- {{#callable:linenoiseEditMoveEnd}} -->
Moves the cursor to the end of the current line in the `linenoiseState` structure.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line editing session.
- **Control Flow**:
    - Checks if the current cursor position (`l->pos`) is not equal to the length of the input line (`l->len`).
    - If the cursor is not at the end, it sets the cursor position to the end of the line (`l->pos = l->len`).
    - Calls the [`refreshLine`](#refreshLine) function to update the display of the line in the terminal.
- **Output**: The function does not return a value; it modifies the state of the `linenoiseState` structure and refreshes the line display.
- **Functions called**:
    - [`refreshLine`](#refreshLine)


---
### linenoiseEditHistoryNext<!-- {{#callable:linenoiseEditHistoryNext}} -->
Substitutes the currently edited line with the next or previous history entry.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line editing.
    - `dir`: An integer indicating the direction of history navigation; it can be either `LINENOISE_HISTORY_PREV` for previous or `LINENOISE_HISTORY_NEXT` for next.
- **Control Flow**:
    - Checks if the history length is greater than 1 to ensure there are entries to navigate.
    - Frees the current history entry at the index specified by `history_index` and saves the current buffer content into the history.
    - Updates the `history_index` based on the direction specified by `dir`.
    - Bounds checks the `history_index` to ensure it does not go out of range.
    - Copies the new history entry into the buffer and updates the length and position of the buffer.
    - Calls `refreshLine(l)` to update the display with the new buffer content.
- **Output**: The function does not return a value; it modifies the state of the `linenoiseState` structure and updates the displayed line accordingly.
- **Functions called**:
    - [`refreshLine`](#refreshLine)


---
### linenoiseEditDelete<!-- {{#callable:linenoiseEditDelete}} -->
Deletes the character at the current cursor position in the input buffer.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the input line, including the buffer, its length, and the cursor position.
- **Control Flow**:
    - Checks if the length of the buffer is greater than 0 and if the cursor position is within the bounds of the buffer.
    - Calculates the length of the character to be deleted using `nextCharLen`.
    - Uses `memmove` to shift the remaining characters in the buffer to the left, effectively removing the character at the cursor position.
    - Decreases the length of the buffer and null-terminates the string.
    - Calls [`refreshLine`](#refreshLine) to update the display with the modified buffer.
- **Output**: The function does not return a value; it modifies the input buffer in place and updates the display.
- **Functions called**:
    - [`refreshLine`](#refreshLine)


---
### linenoiseEditBackspace<!-- {{#callable:linenoiseEditBackspace}} -->
Handles the backspace operation in the linenoise text editor, removing the character before the cursor.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the text editor, including the buffer, cursor position, and length of the input.
- **Control Flow**:
    - Checks if the cursor position (`l->pos`) is greater than 0 and if the length of the buffer (`l->len`) is greater than 0.
    - Calculates the length of the previous character using `prevCharLen` function.
    - Moves the remaining characters in the buffer to the left to overwrite the character being deleted.
    - Updates the cursor position and the length of the buffer accordingly.
    - Sets the last character of the buffer to null terminator to maintain a valid string.
    - Calls [`refreshLine`](#refreshLine) to update the display of the current line.
- **Output**: The function does not return a value; it modifies the state of the `linenoiseState` structure and updates the display.
- **Functions called**:
    - [`refreshLine`](#refreshLine)


---
### linenoiseEditDeletePrevWord<!-- {{#callable:linenoiseEditDeletePrevWord}} -->
Deletes the previous word from the current cursor position in the input buffer.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the input line, including the buffer, cursor position, and length.
- **Control Flow**:
    - Stores the current cursor position in `old_pos`.
    - Moves the cursor back past any trailing spaces to find the start of the previous word.
    - Continues moving the cursor back until it reaches the start of the previous word.
    - Calculates the difference in position (`diff`) between the old position and the new position.
    - Uses `memmove` to shift the remaining characters in the buffer to the left, effectively deleting the previous word.
    - Updates the length of the buffer to reflect the deletion.
    - Calls [`refreshLine`](#refreshLine) to update the display with the modified buffer.
- **Output**: The function does not return a value; it modifies the input buffer directly and refreshes the display to reflect the changes.
- **Functions called**:
    - [`refreshLine`](#refreshLine)


---
### linenoiseEditStart<!-- {{#callable:linenoiseEditStart}} -->
Initializes the `linenoiseState` structure for line editing, sets up terminal modes, and displays the prompt.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the state of the line editing session.
    - `stdin_fd`: The file descriptor for standard input; if set to -1, defaults to `STDIN_FILENO`.
    - `stdout_fd`: The file descriptor for standard output; if set to -1, defaults to `STDOUT_FILENO`.
    - `buf`: A character buffer where the edited line will be stored.
    - `buflen`: The size of the buffer `buf`.
    - `prompt`: A string that will be displayed as the prompt for the user.
- **Control Flow**:
    - The function initializes the `linenoiseState` structure with the provided parameters.
    - It checks if the input file descriptor is valid and sets it to raw mode.
    - The number of columns in the terminal is determined.
    - If the input is not a TTY, the function exits early.
    - The prompt is written to the output file descriptor.
- **Output**: Returns 0 on success, -1 if an error occurs during initialization or writing the prompt.
- **Functions called**:
    - [`enableRawMode`](#enableRawMode)
    - [`getColumns`](#getColumns)
    - [`linenoiseHistoryAdd`](#linenoiseHistoryAdd)


---
### handleEnterKey<!-- {{#callable:handleEnterKey}} -->
Handles the Enter key press in the linenoise input system, updating history and refreshing the input line.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the input line.
- **Control Flow**:
    - Decrements the history length and frees the last entry in the history.
    - If multi-line mode is enabled, moves the cursor to the end of the line.
    - If a hints callback is set, it temporarily disables it to refresh the line without hints, then restores the callback.
    - Returns a duplicate of the current input buffer as a string.
- **Output**: Returns a dynamically allocated string containing the current input line, which should be freed by the caller.
- **Functions called**:
    - [`linenoiseEditMoveEnd`](#linenoiseEditMoveEnd)
    - [`refreshLine`](#refreshLine)


---
### handleCtrlCKey<!-- {{#callable:handleCtrlCKey}} -->
Handles the Ctrl+C key press by setting errno to EAGAIN and returning NULL.
- **Inputs**: None
- **Control Flow**:
    - Sets the global variable `errno` to `EAGAIN` to indicate a temporary error.
    - Returns `NULL` to signal that no valid output is produced.
- **Output**: Returns `NULL`, indicating that the operation was interrupted by Ctrl+C.


---
### handleCtrlDKey<!-- {{#callable:handleCtrlDKey}} -->
Handles the Ctrl+D key input in a line editing context, either deleting a character or signaling end-of-file.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line editing session.
- **Control Flow**:
    - Checks if the length of the current input (`l->len`) is greater than 0.
    - If it is, it calls `linenoiseEditDelete(l)` to delete the character at the cursor position and returns `linenoiseEditMore` to indicate that editing can continue.
    - If the length is 0, it decrements the global `history_len`, frees the last entry in the history, sets `errno` to `ENOENT`, and returns NULL to indicate end-of-file.
- **Output**: Returns a pointer to a string indicating the next action (either a continuation of editing or NULL if end-of-file is reached).
- **Functions called**:
    - [`linenoiseEditDelete`](#linenoiseEditDelete)


---
### handleCtrlTKey<!-- {{#callable:handleCtrlTKey}} -->
Handles the Ctrl+T key press by swapping the previous and current characters in the input buffer.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the input line.
- **Control Flow**:
    - Checks if the cursor position `l->pos` is within valid bounds (greater than 0 and less than the length of the buffer `l->len`).
    - Calculates the length of the previous character using `prevCharLen` and the length of the current character using `nextCharLen`.
    - Creates a string `prev_char` to hold the previous character and copies it from the buffer.
    - Moves the current character to the position of the previous character, effectively deleting it from its original position.
    - Inserts the previous character at the current position, effectively swapping the two characters.
    - Updates the cursor position `l->pos` based on the lengths of the previous and current characters.
    - Calls `refreshLine(l)` to update the display with the new buffer content.
- **Output**: This function does not return a value; it modifies the state of the `linenoiseState` structure and updates the terminal display.
- **Functions called**:
    - [`refreshLine`](#refreshLine)


---
### handleEscapeSequence<!-- {{#callable:handleEscapeSequence}} -->
Handles various escape sequences for line editing in a terminal.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line editing.
    - `esc_type`: An integer representing the type of escape sequence received, which determines the action to be taken.
- **Control Flow**:
    - The function uses a `switch` statement to determine the action based on the value of `esc_type`.
    - If `esc_type` is `ESC_NULL`, no action is taken.
    - For `ESC_DELETE`, it calls [`linenoiseEditDelete`](#linenoiseEditDelete) to delete the character at the cursor position.
    - For `ESC_UP`, it calls [`linenoiseEditHistoryNext`](#linenoiseEditHistoryNext) to navigate to the previous history entry.
    - For `ESC_DOWN`, it calls [`linenoiseEditHistoryNext`](#linenoiseEditHistoryNext) to navigate to the next history entry.
    - For `ESC_RIGHT`, it calls [`linenoiseEditMoveRight`](#linenoiseEditMoveRight) to move the cursor one position to the right.
    - For `ESC_LEFT`, it calls [`linenoiseEditMoveLeft`](#linenoiseEditMoveLeft) to move the cursor one position to the left.
    - For `ESC_HOME`, it calls [`linenoiseEditMoveHome`](#linenoiseEditMoveHome) to move the cursor to the beginning of the line.
    - For `ESC_END`, it calls [`linenoiseEditMoveEnd`](#linenoiseEditMoveEnd) to move the cursor to the end of the line.
- **Output**: The function does not return a value; it modifies the state of the `linenoiseState` structure based on the escape sequence processed.
- **Functions called**:
    - [`linenoiseEditDelete`](#linenoiseEditDelete)
    - [`linenoiseEditHistoryNext`](#linenoiseEditHistoryNext)
    - [`linenoiseEditMoveRight`](#linenoiseEditMoveRight)
    - [`linenoiseEditMoveLeft`](#linenoiseEditMoveLeft)
    - [`linenoiseEditMoveHome`](#linenoiseEditMoveHome)
    - [`linenoiseEditMoveEnd`](#linenoiseEditMoveEnd)


---
### handleCtrlUKey<!-- {{#callable:handleCtrlUKey}} -->
Handles the Ctrl+U key press by clearing the input buffer and resetting the cursor position.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the input line, including the buffer and cursor position.
- **Control Flow**:
    - Sets the first character of the buffer to the null terminator, effectively clearing it.
    - Resets the cursor position (`pos`) and the length of the buffer (`len`) to zero.
    - Calls the [`refreshLine`](#refreshLine) function to update the display with the cleared line.
- **Output**: The function does not return a value; it modifies the state of the `linenoiseState` structure and refreshes the line display.
- **Functions called**:
    - [`refreshLine`](#refreshLine)


---
### handleCtrlKKey<!-- {{#callable:handleCtrlKKey}} -->
Handles the Ctrl+K key press by truncating the current input buffer at the cursor position.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the input line.
- **Control Flow**:
    - Sets the character at the current cursor position in the buffer to the null terminator, effectively truncating the string.
    - Updates the length of the buffer to the current cursor position.
    - Calls the [`refreshLine`](#refreshLine) function to update the display with the modified buffer.
- **Output**: The function does not return a value; it modifies the state of the input buffer directly and refreshes the display.
- **Functions called**:
    - [`refreshLine`](#refreshLine)


---
### processInputCharacter<!-- {{#callable:processInputCharacter}} -->
Processes a single input character and performs the corresponding action based on the character received.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line editing.
    - `c`: An integer representing the input character code.
    - `cbuf`: A character buffer that may contain additional characters to be inserted.
    - `nread`: An integer indicating the number of characters read into `cbuf`.
    - `esc_type`: An integer representing the type of escape sequence, if any.
- **Control Flow**:
    - The function uses a switch statement to determine the action based on the value of `c`.
    - If `c` is `ENTER`, it calls [`handleEnterKey`](#handleEnterKey) and returns its result.
    - If `c` is `CTRL_C`, it calls [`handleCtrlCKey`](#handleCtrlCKey) and returns its result.
    - For backspace and `CTRL_H`, it calls [`linenoiseEditBackspace`](#linenoiseEditBackspace) to remove the character before the cursor.
    - If `c` is `CTRL_D`, it calls [`handleCtrlDKey`](#handleCtrlDKey) to delete the character at the cursor or signal end-of-file.
    - For `CTRL_T`, it calls [`handleCtrlTKey`](#handleCtrlTKey) to swap characters around the cursor.
    - For cursor movement commands (`CTRL_B`, `CTRL_F`, `CTRL_P`, `CTRL_N`, `CTRL_A`, `CTRL_E`), it calls the respective movement functions.
    - If `c` is `ESC`, it calls [`handleEscapeSequence`](#handleEscapeSequence) to process escape sequences.
    - For any other character, it attempts to insert it into the buffer using [`linenoiseEditInsert`](#linenoiseEditInsert).
    - If `c` is `CTRL_U`, it clears the entire line, and if `c` is `CTRL_K`, it deletes from the cursor to the end of the line.
    - Finally, it returns `linenoiseEditMore` to indicate that more input is expected.
- **Output**: Returns a pointer to a string containing the edited line if the input was completed, or NULL if an error occurred or if the user pressed CTRL+C.
- **Functions called**:
    - [`handleEnterKey`](#handleEnterKey)
    - [`handleCtrlCKey`](#handleCtrlCKey)
    - [`linenoiseEditBackspace`](#linenoiseEditBackspace)
    - [`handleCtrlDKey`](#handleCtrlDKey)
    - [`handleCtrlTKey`](#handleCtrlTKey)
    - [`linenoiseEditMoveLeft`](#linenoiseEditMoveLeft)
    - [`linenoiseEditMoveRight`](#linenoiseEditMoveRight)
    - [`linenoiseEditHistoryNext`](#linenoiseEditHistoryNext)
    - [`handleEscapeSequence`](#handleEscapeSequence)
    - [`linenoiseEditInsert`](#linenoiseEditInsert)
    - [`handleCtrlUKey`](#handleCtrlUKey)
    - [`handleCtrlKKey`](#handleCtrlKKey)
    - [`linenoiseEditMoveHome`](#linenoiseEditMoveHome)
    - [`linenoiseEditMoveEnd`](#linenoiseEditMoveEnd)
    - [`linenoiseClearScreen`](#linenoiseClearScreen)
    - [`refreshLine`](#refreshLine)
    - [`linenoiseEditDeletePrevWord`](#linenoiseEditDeletePrevWord)


---
### linenoiseEditFeed<!-- {{#callable:linenoiseEditFeed}} -->
Processes input characters for line editing in a terminal environment.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the current state of the line editing session.
- **Control Flow**:
    - Checks if the input file descriptor is a TTY; if not, it calls `linenoiseNoTTY()` to handle input without character limits.
    - Reads a character from the input using `readCode()` and stores it in `cbuf`, while also capturing the character in `c`.
    - If the character read is an escape character (ESC), it calls `readEscapeSequence()` to determine the type of escape sequence.
    - If the input is in completion mode or the TAB key is pressed, it calls `completeLine()` to handle autocompletion.
    - If the completion returns 0, it indicates that more input is expected, and the function returns `linenoiseEditMore`.
    - Finally, it processes the input character using `processInputCharacter()` and returns the result.
- **Output**: Returns a pointer to the edited line if the editing session is complete, or NULL if an error occurs. If the user is still editing, it returns `linenoiseEditMore`.
- **Functions called**:
    - [`linenoiseNoTTY`](#linenoiseNoTTY)
    - [`readEscapeSequence`](#readEscapeSequence)
    - [`completeLine`](#completeLine)
    - [`processInputCharacter`](#processInputCharacter)


---
### linenoiseEditStop<!-- {{#callable:linenoiseEditStop}} -->
Stops the line editing session and restores the terminal to its normal mode.
- **Inputs**:
    - `l`: A pointer to a `linenoiseState` structure that holds the state of the line editing session.
- **Control Flow**:
    - Checks if the input file descriptor (`ifd`) of the `linenoiseState` structure is associated with a terminal using `isatty`.
    - If it is not a terminal, the function returns immediately without performing any actions.
    - If it is a terminal, the function calls [`disableRawMode`](#disableRawMode) to restore the terminal settings to normal.
    - Finally, it prints a newline character to the output.
- **Output**: The function does not return a value; it performs actions to stop the editing session and restore terminal settings.
- **Functions called**:
    - [`disableRawMode`](#disableRawMode)


---
### linenoiseBlockingEdit<!-- {{#callable:linenoiseBlockingEdit}} -->
The `linenoiseBlockingEdit` function facilitates a blocking line editing session, allowing users to input text interactively in a terminal.
- **Inputs**:
    - `stdin_fd`: The file descriptor for standard input, typically set to STDIN_FILENO.
    - `stdout_fd`: The file descriptor for standard output, typically set to STDOUT_FILENO.
    - `buf`: A pointer to a character buffer where the edited line will be stored.
    - `buflen`: The size of the buffer pointed to by `buf`, which must be greater than zero.
    - `prompt`: A string that will be displayed as a prompt to the user.
- **Control Flow**:
    - The function first checks if the provided buffer length is zero, returning NULL and setting errno to EINVAL if true.
    - It initializes the editing state by calling [`linenoiseEditStart`](#linenoiseEditStart), passing the relevant parameters.
    - A loop is entered where [`linenoiseEditFeed`](#linenoiseEditFeed) is called repeatedly to process user input until it returns a value other than `linenoiseEditMore`.
    - After exiting the loop, [`linenoiseEditStop`](#linenoiseEditStop) is called to restore the terminal state.
- **Output**: The function returns a pointer to the edited line if the user completes the input, or NULL if an error occurs or if the user pressed Ctrl-D.
- **Functions called**:
    - [`linenoiseEditStart`](#linenoiseEditStart)
    - [`linenoiseEditFeed`](#linenoiseEditFeed)
    - [`linenoiseEditStop`](#linenoiseEditStop)


---
### linenoisePrintKeyCodes<!-- {{#callable:linenoisePrintKeyCodes}} -->
The `linenoisePrintKeyCodes` function enables a debugging mode to display key scan codes as they are pressed.
- **Inputs**: None
- **Control Flow**:
    - Prints a welcome message indicating the debugging mode and instructions to exit.
    - Enters raw mode for the terminal to capture key presses without additional processing.
    - Initializes a character array `quit` to track the last few characters typed.
    - Enters an infinite loop to read single characters from standard input.
    - If a character is read, it shifts the `quit` array left and adds the new character to the end.
    - Checks if the last four characters in `quit` match 'quit', and breaks the loop if they do.
    - Prints the character pressed along with its hexadecimal and decimal values.
    - Flushes the output to ensure the printed information appears immediately.
    - Exits the loop and disables raw mode when 'quit' is typed.
- **Output**: The function does not return a value; it outputs the scan codes of pressed keys to the terminal until 'quit' is typed.
- **Functions called**:
    - [`enableRawMode`](#enableRawMode)
    - [`disableRawMode`](#disableRawMode)


---
### linenoiseNoTTY<!-- {{#callable:linenoiseNoTTY}} -->
Reads a line of input from standard input without terminal control.
- **Inputs**: None
- **Control Flow**:
    - Initializes a buffer for the input line and sets its length and maximum length.
    - Enters an infinite loop to read characters from standard input.
    - Doubles the buffer size when the current length reaches the maximum length.
    - Reads a character from standard input and checks if it is EOF or a newline.
    - If EOF is encountered and no characters have been read, frees the buffer and returns NULL.
    - If a newline is encountered, null-terminates the string and returns the buffer.
    - If a valid character is read, it is added to the buffer and the length is incremented.
- **Output**: Returns a dynamically allocated string containing the input line, or NULL on failure.


---
### linenoise<!-- {{#callable:linenoise}} -->
The `linenoise` function provides a simple line editing interface that handles user input from the terminal, supporting both interactive and non-interactive modes.
- **Inputs**:
    - `prompt`: A string that serves as a prompt displayed to the user before input is taken.
- **Control Flow**:
    - Checks if the standard input is a terminal using `isatty(STDIN_FILENO)`.
    - If not a terminal, it calls `linenoiseNoTTY()` to read input without size limits.
    - If the terminal is unsupported, it prints the prompt, reads a line using `fgets`, and removes trailing newline characters.
    - If the terminal is supported, it calls `linenoiseBlockingEdit()` to handle interactive line editing.
- **Output**: Returns a pointer to a dynamically allocated string containing the user input, or NULL on error.
- **Functions called**:
    - [`linenoiseNoTTY`](#linenoiseNoTTY)
    - [`isUnsupportedTerm`](#isUnsupportedTerm)
    - [`linenoiseBlockingEdit`](#linenoiseBlockingEdit)


---
### linenoiseFree<!-- {{#callable:linenoiseFree}} -->
Frees the memory allocated for a given pointer, ensuring it is not the special `linenoiseEditMore` pointer.
- **Inputs**:
    - `ptr`: A pointer to the memory that needs to be freed.
- **Control Flow**:
    - Checks if the input pointer `ptr` is equal to the special pointer `linenoiseEditMore`.
    - If they are equal, the function returns immediately to prevent misuse.
    - If they are not equal, the function calls `free(ptr)` to deallocate the memory.
- **Output**: The function does not return a value; it performs a memory deallocation operation.


---
### freeHistory<!-- {{#callable:freeHistory}} -->
Frees the memory allocated for the command history.
- **Inputs**: None
- **Control Flow**:
    - Checks if the `history` pointer is not null.
    - Iterates over the `history` array using a loop from 0 to `history_len`.
    - Frees each entry in the `history` array.
    - Finally, frees the `history` array itself.
- **Output**: The function does not return a value; it performs memory cleanup.


---
### linenoiseAtExit<!-- {{#callable:linenoiseAtExit}} -->
The `linenoiseAtExit` function is responsible for cleaning up resources and restoring terminal settings when the program exits.
- **Inputs**: None
- **Control Flow**:
    - Calls [`disableRawMode`](#disableRawMode) to restore the terminal to its original state.
    - Calls [`freeHistory`](#freeHistory) to deallocate any memory used for command history.
- **Output**: This function does not return a value; it performs cleanup operations.
- **Functions called**:
    - [`disableRawMode`](#disableRawMode)
    - [`freeHistory`](#freeHistory)


---
### linenoiseHistoryAdd<!-- {{#callable:linenoiseHistoryAdd}} -->
Adds a new line to the command history, managing duplicates and maximum length.
- **Inputs**:
    - `line`: A pointer to a constant character string representing the line to be added to the history.
- **Control Flow**:
    - Checks if the maximum history length is set to zero, returning 0 if true.
    - Initializes the history array if it is NULL, allocating memory for it.
    - Checks if the last entry in the history is the same as the new line, returning 0 if they are identical.
    - Duplicates the input line and checks for successful memory allocation.
    - If the history is full, it frees the oldest entry and shifts the remaining entries to make space.
    - Adds the new line to the history and increments the history length.
- **Output**: Returns 1 on successful addition of the line to the history, or 0 if the operation fails.


---
### linenoiseHistorySetMaxLen<!-- {{#callable:linenoiseHistorySetMaxLen}} -->
Sets the maximum length of the command history in the linenoise library.
- **Inputs**:
    - `len`: An integer representing the new maximum length for the history.
- **Control Flow**:
    - If `len` is less than 1, the function returns 0 immediately.
    - If the `history` array is not null, it proceeds to allocate a new array of pointers with the size of `len`.
    - If the allocation fails, it returns 0.
    - It checks if the new length is less than the current history length; if so, it frees the excess history entries.
    - The function then copies the relevant entries from the old history to the new one.
    - After updating the `history` pointer, it sets the new maximum length and adjusts the current length if necessary.
- **Output**: Returns 1 on success, indicating the maximum length has been set successfully.


---
### linenoiseHistorySave<!-- {{#callable:linenoiseHistorySave}} -->
Saves the command history to a specified file.
- **Inputs**:
    - `filename`: A pointer to a constant character string representing the name of the file where the history will be saved.
- **Control Flow**:
    - Sets the file permission mask to restrict access to the file being created.
    - Attempts to open the specified file in write mode.
    - Restores the original file permission mask after attempting to open the file.
    - Checks if the file was successfully opened; if not, returns -1.
    - Sets the file permissions to allow read and write access for the user.
    - Iterates through the command history and writes each entry to the file, followed by a newline.
    - Returns 0 upon successful completion of the save operation.
- **Output**: Returns 0 on success, or -1 if an error occurs during file operations.


---
### linenoiseHistoryLoad<!-- {{#callable:linenoiseHistoryLoad}} -->
Loads command history from a specified file into the history buffer.
- **Inputs**:
    - `filename`: A pointer to a constant character string representing the name of the file from which to load the history.
- **Control Flow**:
    - Attempts to open the specified file in read mode.
    - If the file cannot be opened, returns -1.
    - Reads each line from the file until EOF is reached.
    - For each line read, it removes any trailing newline or carriage return characters.
    - Adds the cleaned line to the history using [`linenoiseHistoryAdd`](#linenoiseHistoryAdd).
- **Output**: Returns 0 on success, or -1 if an error occurs (e.g., file cannot be opened).
- **Functions called**:
    - [`linenoiseHistoryAdd`](#linenoiseHistoryAdd)


