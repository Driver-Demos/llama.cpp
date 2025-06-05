# Purpose
This C++ source code file provides a comprehensive console input/output management system, designed to handle both Windows and POSIX (Unix-like) environments. The code is structured to facilitate advanced console interactions, such as handling UTF-8 input/output, managing console display settings, and supporting both simple and advanced input modes. It includes platform-specific initialization and cleanup routines to configure console settings appropriately, such as enabling virtual terminal processing on Windows and adjusting terminal attributes on POSIX systems. The code also defines ANSI color codes for text formatting, allowing for colored console output.

The file is organized into several key components: initialization and cleanup functions, display and input/output management, and utility functions for handling UTF-8 encoding and cursor manipulation. The [`init`](#consoleinit) and [`cleanup`](#consolecleanup) functions set up and restore console states, respectively, while the [`set_display`](#consoleset_display) function manages text color changes based on the current display state. Input handling is split into simple and advanced modes, with the [`readline`](#consolereadline) function serving as the main interface for reading user input. This code is intended to be part of a larger application, providing a robust and flexible console interface that can be integrated into other projects. It does not define a public API or external interfaces but rather serves as an internal utility for managing console interactions.
# Imports and Dependencies

---
- `console.h`
- `vector`
- `iostream`
- `windows.h`
- `fcntl.h`
- `io.h`
- `climits`
- `sys/ioctl.h`
- `unistd.h`
- `wchar.h`
- `stdio.h`
- `stdlib.h`
- `signal.h`
- `termios.h`


# Functions

---
### init<!-- {{#callable:console::init}} -->
The `init` function initializes console settings for input and output based on the operating system and user preferences for simple I/O and advanced display features.
- **Inputs**:
    - `use_simple_io`: A boolean flag indicating whether to use simple input/output settings.
    - `use_advanced_display`: A boolean flag indicating whether to enable advanced display features.
- **Control Flow**:
    - Set the global variables `advanced_display` and `simple_io` based on the input parameters.
    - For Windows systems, obtain the standard output handle and check its validity; if invalid, attempt to use the standard error handle.
    - If a valid console handle is found, attempt to enable virtual terminal processing for advanced display; if unsuccessful, disable advanced display.
    - Set the console output codepage to UTF-8.
    - Obtain the standard input handle and configure it for UTF-16 input; adjust line input and echo settings based on `simple_io`.
    - For POSIX systems, if `simple_io` is not used, modify terminal settings to disable canonical mode and echo, and open `/dev/tty` for output.
    - Set the locale to the user's default settings.
- **Output**: The function does not return a value; it modifies global state and console settings.


---
### cleanup<!-- {{#callable:console::cleanup}} -->
The `cleanup` function resets the console display and restores terminal settings on POSIX systems.
- **Inputs**: None
- **Control Flow**:
    - The function calls `set_display(reset)` to reset the console display settings.
    - On non-Windows systems, it checks if `simple_io` is false, indicating advanced I/O settings were used.
    - If `tty` is not null, it closes the `tty` file and sets `out` back to `stdout`.
    - It restores the terminal attributes to their initial state using `tcsetattr`.
- **Output**: The function does not return any value.
- **Functions called**:
    - [`console::set_display`](#consoleset_display)


---
### set\_display<!-- {{#callable:console::set_display}} -->
The `set_display` function changes the console's text color based on the specified display mode if advanced display is enabled and the current display mode is different from the specified one.
- **Inputs**:
    - `display`: An enumeration value of type `display_t` that specifies the desired display mode, such as `reset`, `prompt`, `user_input`, or `error`.
- **Control Flow**:
    - Check if `advanced_display` is true and `current_display` is not equal to the input `display`.
    - Flush the standard output stream to ensure all previous output is displayed.
    - Use a switch statement to determine the ANSI color code to apply based on the `display` value.
    - Update the `current_display` to the new `display` value.
    - Flush the output stream `out` to apply the changes immediately.
- **Output**: The function does not return a value; it modifies the console's display settings as a side effect.


---
### getchar32<!-- {{#callable:console::getchar32}} -->
The `getchar32` function reads a single UTF-32 character from the console input, handling surrogate pairs on both Windows and non-Windows systems.
- **Inputs**: None
- **Control Flow**:
    - On Windows, it uses `ReadConsoleInputW` to read input events from the console, checking for key events and handling surrogate pairs to form a complete UTF-32 character.
    - If a high surrogate is detected, it waits for a low surrogate to form a complete character; otherwise, it returns the character directly.
    - On non-Windows systems, it uses `getwchar` to read a wide character and checks for surrogate pairs if `WCHAR_MAX` is 0xFFFF, returning a replacement character for invalid pairs.
    - The function loops until a valid character is read or an end-of-file condition is encountered.
- **Output**: The function returns a `char32_t` representing the UTF-32 character read from the console, or `WEOF` if an error occurs or end-of-file is reached.


---
### pop\_cursor<!-- {{#callable:console::pop_cursor}} -->
The `pop_cursor` function moves the console cursor one position back, either by adjusting the cursor position directly on Windows or by outputting a backspace character on other systems.
- **Inputs**: None
- **Control Flow**:
    - Check if the code is running on a Windows system using the preprocessor directive `#if defined(_WIN32)`.
    - If on Windows, check if `hConsole` is not `NULL`.
    - Retrieve the current console screen buffer information using `GetConsoleScreenBufferInfo`.
    - Calculate the new cursor position: if the cursor is at the start of a line (`X == 0`), move it to the end of the previous line; otherwise, move it one position to the left.
    - Set the new cursor position using `SetConsoleCursorPosition`.
    - If not on Windows, output a backspace character using `putc('\b', out)`.
- **Output**: The function does not return any value; it modifies the console cursor position as a side effect.


---
### estimateWidth<!-- {{#callable:console::estimateWidth}} -->
The `estimateWidth` function estimates the display width of a Unicode codepoint, returning a fixed width of 1 on Windows and using `wcwidth` on other platforms.
- **Inputs**:
    - `codepoint`: A Unicode codepoint represented as a `char32_t` type, whose display width is to be estimated.
- **Control Flow**:
    - The function checks if the code is being compiled on a Windows platform using the preprocessor directive `#if defined(_WIN32)`.
    - If on Windows, the function ignores the `codepoint` input and returns a fixed width of 1.
    - If not on Windows, the function calls `wcwidth(codepoint)` to determine the display width of the codepoint.
- **Output**: An integer representing the estimated display width of the given Unicode codepoint.


---
### put\_codepoint<!-- {{#callable:console::put_codepoint}} -->
The `put_codepoint` function writes a UTF-8 encoded codepoint to the console and calculates the width of the codepoint on the console, handling both Windows and POSIX systems.
- **Inputs**:
    - `utf8_codepoint`: A pointer to a UTF-8 encoded character sequence representing the codepoint to be written to the console.
    - `length`: The length of the UTF-8 encoded character sequence.
    - `expectedWidth`: The expected width of the codepoint on the console, used as a fallback if actual width cannot be determined.
- **Control Flow**:
    - On Windows, retrieve the initial cursor position using `GetConsoleScreenBufferInfo` and write the codepoint to the console using `WriteConsole`.
    - Check if the cursor is in the last column and adjust the cursor position if necessary by writing a space and backspace.
    - Calculate the width of the codepoint by comparing the new cursor position with the initial position, adjusting for wrapping if necessary.
    - On POSIX systems, if `expectedWidth` is non-negative or `tty` is null, write the codepoint using `fwrite` and return `expectedWidth`.
    - If `expectedWidth` is negative, query the cursor position before and after writing the codepoint using ANSI escape sequences and `fscanf`.
    - Calculate the width by comparing the cursor positions, adjusting for wrapping using `ioctl` if necessary.
- **Output**: The function returns the actual width of the codepoint on the console, or `expectedWidth` if the actual width cannot be determined.


---
### replace\_last<!-- {{#callable:console::replace_last}} -->
The `replace_last` function replaces the last character in the console output with a specified character, handling both Windows and non-Windows systems differently.
- **Inputs**:
    - `ch`: A character to replace the last character in the console output.
- **Control Flow**:
    - Check if the code is being compiled on a Windows system using the `_WIN32` macro.
    - If on Windows, call `pop_cursor()` to move the cursor back one position, then use `put_codepoint()` to write the new character at the current cursor position.
    - If not on Windows, use `fprintf()` to write a backspace character followed by the new character to the output stream.
- **Output**: The function does not return a value; it performs an action on the console output.
- **Functions called**:
    - [`console::pop_cursor`](#consolepop_cursor)
    - [`console::put_codepoint`](#consoleput_codepoint)


---
### append\_utf8<!-- {{#callable:console::append_utf8}} -->
The `append_utf8` function encodes a Unicode code point into its UTF-8 representation and appends it to a given string.
- **Inputs**:
    - `ch`: A Unicode code point represented as a `char32_t`.
    - `out`: A reference to a `std::string` where the UTF-8 encoded result will be appended.
- **Control Flow**:
    - Check if the code point `ch` is less than or equal to 0x7F, and if so, append it directly as a single byte to `out`.
    - If `ch` is less than or equal to 0x7FF, encode it as a two-byte UTF-8 sequence and append both bytes to `out`.
    - If `ch` is less than or equal to 0xFFFF, encode it as a three-byte UTF-8 sequence and append all three bytes to `out`.
    - If `ch` is less than or equal to 0x10FFFF, encode it as a four-byte UTF-8 sequence and append all four bytes to `out`.
    - If `ch` is greater than 0x10FFFF, it is considered an invalid Unicode code point and no action is taken.
- **Output**: The function does not return a value; it modifies the `out` string by appending the UTF-8 encoded bytes.


---
### pop\_back\_utf8\_char<!-- {{#callable:console::pop_back_utf8_char}} -->
The `pop_back_utf8_char` function removes the last UTF-8 character from a given string.
- **Inputs**:
    - `line`: A reference to a `std::string` from which the last UTF-8 character will be removed.
- **Control Flow**:
    - Check if the string `line` is empty; if so, return immediately.
    - Initialize `pos` to the last index of the string `line`.
    - Iterate up to 3 times or until `pos` is greater than 0, decrementing `pos` each time, to find the start of the last UTF-8 character by checking if the byte does not have the form `10xxxxxx`.
    - Erase the character starting from the position `pos` to the end of the string.
- **Output**: The function modifies the input string `line` in place by removing its last UTF-8 character, and it does not return any value.


---
### readline\_advanced<!-- {{#callable:console::readline_advanced}} -->
The `readline_advanced` function reads a line of input from the console, handling special characters, escape sequences, and multiline input, while updating the display accordingly.
- **Inputs**:
    - `line`: A reference to a string where the input line will be stored.
    - `multiline_input`: A boolean indicating whether multiline input is allowed.
- **Control Flow**:
    - Flushes the output stream if it is not stdout.
    - Clears the input line and initializes variables for tracking character widths, special character status, and end-of-stream status.
    - Enters a loop to read characters using [`getchar32`](#consolegetchar32) until a newline, carriage return, or end-of-stream is encountered.
    - Handles special characters by setting the display and replacing the last character if necessary.
    - Processes escape sequences by discarding them until a terminating character is found.
    - Handles backspace by removing characters from the line and adjusting the cursor position accordingly.
    - Appends regular characters to the line, calculates their display width, and updates the widths vector.
    - Checks for special characters '\' or '/' at the end of the line to toggle special character status.
    - After exiting the loop, determines if more input is expected based on the multiline_input flag and special character status.
    - Flushes the output stream and returns whether more input is expected.
- **Output**: Returns a boolean indicating whether more input is expected (true if multiline input is ongoing, false otherwise).
- **Functions called**:
    - [`console::getchar32`](#consolegetchar32)
    - [`console::set_display`](#consoleset_display)
    - [`console::replace_last`](#consolereplace_last)
    - [`console::pop_cursor`](#consolepop_cursor)
    - [`console::pop_back_utf8_char`](#consolepop_back_utf8_char)
    - [`console::append_utf8`](#consoleappend_utf8)
    - [`console::put_codepoint`](#consoleput_codepoint)
    - [`console::estimateWidth`](#consoleestimateWidth)


---
### readline\_simple<!-- {{#callable:console::readline_simple}} -->
The `readline_simple` function reads a line of input from the console, handling special characters for multiline input and returning a boolean indicating whether to continue reading.
- **Inputs**:
    - `line`: A reference to a string where the input line will be stored.
    - `multiline_input`: A boolean flag indicating whether multiline input is currently enabled.
- **Control Flow**:
    - On Windows, reads a wide string from standard input and converts it to a UTF-8 encoded string.
    - On non-Windows systems, reads a line directly from standard input into the provided string.
    - Checks if the last character of the line is '/' or '\'.
    - If the last character is '/', it removes it and returns false to stop reading.
    - If the last character is '\', it removes it and toggles the multiline_input flag.
    - Appends a newline character to the line.
    - Returns the value of multiline_input to determine if reading should continue.
- **Output**: Returns a boolean indicating whether to continue reading input based on the multiline_input flag and the presence of special characters.


---
### readline<!-- {{#callable:console::readline}} -->
The `readline` function reads a line of input from the console, using either a simple or advanced method based on the `simple_io` flag.
- **Inputs**:
    - `line`: A reference to a `std::string` where the input line will be stored.
    - `multiline_input`: A boolean indicating whether multiline input is allowed.
- **Control Flow**:
    - The function sets the display mode to `user_input` using [`set_display`](#consoleset_display).
    - It checks the `simple_io` flag to determine which input method to use.
    - If `simple_io` is true, it calls [`readline_simple`](#consolereadline_simple) to read the input line.
    - If `simple_io` is false, it calls [`readline_advanced`](#consolereadline_advanced) to read the input line.
- **Output**: Returns a boolean indicating whether more input is expected, based on the `multiline_input` flag and the input method used.
- **Functions called**:
    - [`console::set_display`](#consoleset_display)
    - [`console::readline_simple`](#consolereadline_simple)
    - [`console::readline_advanced`](#consolereadline_advanced)


