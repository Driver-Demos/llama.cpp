# Purpose
This Bash script is designed to synchronize changes from the `ggml` repository to the `llama.cpp` project, providing a narrow functionality focused on version control and patch management. It is a utility script, not an executable or library, and is intended to be run from the command line within the `llama.cpp` directory. The script identifies new commits in the `ggml` repository since the last synchronization, generates patches for these commits, and applies them to the `llama.cpp` project, while allowing the user to skip specific commits and adjust the context lines for the patches. It also updates a record of the last synchronized commit to ensure continuity in future synchronizations.
# Global Variables

---
### sd
- **Type**: `string`
- **Description**: The variable `sd` is a string that holds the directory path of the script's location. It is determined by using the `dirname` command on `$0`, which represents the script's name or path as it was invoked.
- **Use**: This variable is used to change the current working directory to the parent directory of the script's location.


---
### SRC\_LLAMA
- **Type**: `string`
- **Description**: The `SRC_LLAMA` variable is a string that holds the absolute path to the current working directory where the script is executed. It is determined by using the `pwd` command after changing the directory to the parent of the script's location.
- **Use**: This variable is used to reference the base directory of the `llama.cpp` project throughout the script, particularly for file path operations and validations.


---
### SRC\_GGML
- **Type**: `string`
- **Description**: `SRC_GGML` is a global variable that stores the absolute path to the 'ggml' directory, which is located one level up from the current working directory. It is determined by changing the directory to '../ggml' and then using the `pwd` command to get the full path.
- **Use**: This variable is used to verify the existence of the 'ggml' directory and to navigate into it for further operations such as logging and patching.


---
### lc
- **Type**: `string`
- **Description**: The variable `lc` is a string that stores the last commit hash from the `sync-ggml.last` file located in the `scripts` directory of the `llama.cpp` project. This hash represents the last synchronized commit from the `ggml` repository.
- **Use**: It is used to determine the range of new commits in the `ggml` repository that need to be synchronized with the `llama.cpp` project.


---
### to\_skip
- **Type**: `string`
- **Description**: The `to_skip` variable is a global string variable initialized to an empty string. It is used to store a list of commit hashes that should be skipped during the synchronization process of ggml changes to llama.cpp.
- **Use**: This variable is used to determine which commits should be ignored when generating patches for synchronization.


---
### ctx
- **Type**: `string`
- **Description**: The `ctx` variable is a global string variable that specifies the number of lines of context to include in git patches. It is initially set to "8" by default, meaning that 8 lines of context will be included in the patches. This variable can be modified by the user through the command line option `-C` to specify a different number of context lines.
- **Use**: The `ctx` variable is used to determine the number of lines of context included in git patches when using the `git format-patch` and `git am` commands.


