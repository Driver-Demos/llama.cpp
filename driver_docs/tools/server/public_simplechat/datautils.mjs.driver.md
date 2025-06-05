# Purpose
This source code file provides a set of utility functions and a class designed to handle and manipulate strings, specifically focusing on removing repetitive or "garbage" text from the end of strings. The primary functionality is encapsulated in several functions that implement different strategies for identifying and trimming unwanted repeated text patterns. These strategies include substring-based repeat matching and character histogram analysis, which are used to detect and remove repetitive sequences that may degrade the quality of text data, particularly in the context of local language models with limited context size.

The file defines several functions, such as `trim_repeat_garbage_at_end`, `trim_repeat_garbage_at_end_loop`, `trim_hist_garbage_at_end`, and `trim_hist_garbage_at_end_loop`, each implementing a specific logic for garbage trimming. The `trim_repeat_garbage_at_end` function uses a progressively larger substring matching approach, while `trim_hist_garbage_at_end` employs a histogram-driven method to identify and trim garbage text. These functions are designed to be used iteratively, as seen in the `trim_garbage_at_end` function, which combines both methods to ensure comprehensive garbage removal.

Additionally, the file includes a `NewLines` class, which provides a utility for managing and manipulating lines of text. This class allows for the addition and appending of lines, as well as the shifting of lines from the beginning of the list, with options to handle full or partial lines. The class is designed to facilitate the construction and management of text data, supporting operations that are common in text processing tasks. Overall, the file offers a focused set of tools for improving text data quality by removing redundant or unwanted text patterns.
# Data Structures

---
### NewLines
- **Type**: `class`
- **Members**:
    - `lines`: An array of strings that stores the lines.
- **Description**: The `NewLines` class is a utility for managing a collection of text lines, allowing for the addition and manipulation of lines of text. It provides methods to append new lines or parts of lines to the existing collection and to retrieve and remove the earliest line from the collection. The class is designed to handle both complete lines and partial lines, ensuring that lines are correctly appended and managed within the internal array.


# Functions

---
### trim\_repeat\_garbage\_at\_end
The `trim_repeat_garbage_at_end` function removes perfectly matching repeating substrings from the end of a string to eliminate garbage text.
- **Inputs**:
    - `sIn`: The input string from which repeating garbage at the end is to be trimmed.
    - `maxSubL`: The maximum length of the substring to consider for repeat matching, defaulting to 10.
    - `maxMatchLenThreshold`: The minimum length of the repeating substring required to consider it as garbage, defaulting to 40.
- **Control Flow**:
    - Initialize an array `rCnt` to keep track of repeat counts for each substring length.
    - Set `maxMatchLen` to `maxSubL` and `iMML` to -1 to track the maximum match length and its corresponding substring length.
    - Iterate over possible substring lengths from 1 to `maxSubL`.
    - For each substring length, extract the reference substring from the end of the input string.
    - Iterate backwards through the string in steps of the current substring length, comparing each substring to the reference substring.
    - If a mismatch is found, calculate the current match length and update `maxMatchLen` and `iMML` if the current match length is greater.
    - If no mismatch is found, increment the repeat count for the current substring length.
    - After iterating through all substring lengths, check if a valid match was found (i.e., `iMML` is not -1 and `maxMatchLen` is greater than `maxMatchLenThreshold`).
    - If a valid match is found, trim the input string by removing the matched garbage and return the trimmed result.
    - If no valid match is found, return the original string indicating no trimming was done.
- **Output**: An object containing a boolean `trimmed` indicating if trimming occurred, and `data` which is the resulting string after trimming.


---
### trim\_repeat\_garbage\_at\_end\_loop
The function `trim_repeat_garbage_at_end_loop` iteratively trims repeating garbage at the end of a string by using a substring-based matching approach, with optional character skipping to handle variations.
- **Inputs**:
    - `sIn`: The input string from which repeating garbage at the end is to be trimmed.
    - `maxSubL`: The maximum length of the substring to consider for repeat matching.
    - `maxMatchLenThreshold`: The minimum length of the repeating pattern required to consider it as garbage.
    - `skipMax`: The maximum number of times to attempt skipping a character at the end if trimming is not successful, defaulting to 16.
- **Control Flow**:
    - Initialize `sCur` with the input string `sIn`, `sSaved` as an empty string, and `iTry` as 0.
    - Enter an infinite loop to repeatedly attempt trimming the string using `trim_repeat_garbage_at_end`.
    - If the trimming is unsuccessful (`got.trimmed` is false), check if this is the first attempt (`iTry == 0`) and save the current data to `sSaved`.
    - Increment the `iTry` counter and check if it has reached `skipMax`; if so, return `sSaved`.
    - If not, remove the last character from `got.data` and continue the loop.
    - If trimming is successful, reset `iTry` to 0 and update `sCur` with the trimmed data.
    - Continue the loop until a successful trim is no longer possible or `skipMax` is reached.
- **Output**: Returns the trimmed string if successful, or the last saved untrimmed string if the maximum number of skips is reached.


---
### trim\_hist\_garbage\_at\_end
The `trim_hist_garbage_at_end` function attempts to remove garbage text at the end of a string using a histogram-based approach to identify and trim repeated character patterns.
- **Inputs**:
    - `sIn`: The input string from which garbage text at the end is to be trimmed.
    - `maxType`: The maximum allowed count for any type of character (alphabet, numeral, or other) in the histogram.
    - `maxUniq`: The maximum number of unique characters allowed in the histogram.
    - `maxMatchLenThreshold`: The maximum length of the substring to consider for matching and trimming.
- **Control Flow**:
    - Check if the input string length is less than the maxMatchLenThreshold; if so, return the original string as no trimming is possible.
    - Initialize counters for different character types (alphabet, numeral, other) and a histogram to track character frequencies.
    - Iterate over the last maxMatchLenThreshold characters of the string to populate the histogram and count unique characters.
    - If the number of unique characters exceeds maxUniq, break the loop.
    - Check if any character type exceeds maxType; if so, return the original string as no trimming is possible.
    - Iterate over the string from the end to check if characters are within the histogram; if a character is not found in the histogram, trim the string up to that point.
    - If all characters are within the histogram, trim the string completely.
- **Output**: An object containing a boolean `trimmed` indicating if trimming occurred, and `data` which is the resulting string after trimming.


---
### trim\_hist\_garbage\_at\_end\_loop
The function `trim_hist_garbage_at_end_loop` repeatedly trims repeating garbage at the end of a string using a histogram-driven approach until no more garbage can be trimmed.
- **Inputs**:
    - `sIn`: The input string from which garbage at the end needs to be trimmed.
    - `maxType`: The maximum number of each character type (alphabet, numeral, other) allowed in the histogram for trimming.
    - `maxUniq`: The maximum number of unique characters allowed in the histogram for trimming.
    - `maxMatchLenThreshold`: The maximum length of the substring to consider for matching and trimming.
- **Control Flow**:
    - Initialize `sCur` with the input string `sIn`.
    - Enter an infinite loop to repeatedly attempt trimming.
    - Call `trim_hist_garbage_at_end` with the current string and parameters to attempt trimming.
    - If the result indicates no trimming was done (`got.trimmed` is false), return the current string as no more garbage can be trimmed.
    - If trimming was successful, update `sCur` with the trimmed string and continue the loop.
- **Output**: The function returns the trimmed string with garbage removed from the end, or the original string if no garbage could be trimmed.


---
### trim\_garbage\_at\_end
The `trim_garbage_at_end` function attempts to remove repeating and histogram-based garbage from the end of a string using two different trimming strategies.
- **Inputs**:
    - `sIn`: The input string from which garbage at the end needs to be trimmed.
- **Control Flow**:
    - Initialize `sCur` with the input string `sIn`.
    - Iterate twice to apply both trimming strategies.
    - First, apply `trim_hist_garbage_at_end_loop` to `sCur` with parameters for maximum type, unique characters, and match length threshold.
    - Then, apply `trim_repeat_garbage_at_end_loop` to the result with parameters for maximum substring length, match length threshold, and skip maximum.
    - Return the final trimmed string `sCur`.
- **Output**: The function returns a string with garbage trimmed from the end, if any was found and removed.


---
### add\_append
The `add_append` function processes a string by splitting it into lines and appending them to an existing list of lines, ensuring proper handling of newline characters.
- **Inputs**:
    - `sLines`: A string containing lines of text, potentially with newline characters, to be added to the existing list of lines.
- **Control Flow**:
    - Split the input string `sLines` into an array of lines using the newline character as a delimiter.
    - Iterate over each line in the resulting array.
    - For each line, check if it is not the last line in the array or if the original string ends with a newline, and append a newline character if necessary.
    - If the current line is the first line in the array, check if the last line in the existing list does not end with a newline; if so, append the current line to the last line in the list.
    - If the current line is not appended to the last line, add it as a new line to the list.
- **Output**: The function does not return a value; it modifies the `lines` property of the `NewLines` class instance by adding the processed lines to it.


---
### shift
The `shift` function removes and returns the earliest line from an array of lines, optionally ensuring it ends with a newline character.
- **Inputs**:
    - `bFullWithNewLineOnly`: A boolean flag indicating whether only full lines (those ending with a newline character) should be returned.
- **Control Flow**:
    - Retrieve the first line from the `lines` array.
    - Check if the line is undefined; if so, return undefined.
    - If the line does not end with a newline character and `bFullWithNewLineOnly` is true, return undefined.
    - Remove and return the first line from the `lines` array.
- **Output**: Returns the earliest line from the `lines` array if it meets the specified conditions, otherwise returns undefined.


