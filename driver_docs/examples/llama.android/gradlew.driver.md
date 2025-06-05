# Purpose
This script is a startup script for Gradle on UNIX-like systems, designed to initialize and execute the Gradle build tool. It provides a narrow functionality focused on setting up the environment and executing the Gradle Wrapper, which is a mechanism to ensure a specific version of Gradle is used for a project. The script determines the appropriate Java command to use, sets up necessary environment variables like `CLASSPATH`, and handles platform-specific configurations, such as path conversions for Cygwin or MSYS environments. It also manages JVM options and ensures that the maximum file descriptor limit is set appropriately. This script is typically used as an executable to launch Gradle builds in a consistent and controlled manner across different UNIX-like operating systems.
# Global Variables

---
### PRG
- **Type**: `string`
- **Description**: The `PRG` variable is initialized with the value of `$0`, which represents the name of the script being executed. It is used to resolve the actual path of the script, even if it is a symbolic link, by following the links until the real path is found.
- **Use**: `PRG` is used to determine the script's directory path, which is essential for setting the `APP_HOME` variable to the correct location.


---
### SAVED
- **Type**: `string`
- **Description**: The `SAVED` variable is a string that stores the current working directory path at the time of its assignment. It is defined using the command substitution syntax to capture the output of the `pwd` command, which prints the current directory.
- **Use**: This variable is used to save the current directory path so that the script can return to it after changing directories to determine the application's home directory.


---
### APP\_HOME
- **Type**: `string`
- **Description**: `APP_HOME` is a global variable that stores the absolute path of the directory where the script is located. It is determined by resolving any symbolic links and navigating to the directory of the script, then capturing the current working directory path.
- **Use**: This variable is used to set the base directory for the application, which is essential for locating resources and executing commands relative to the script's location.


---
### APP\_NAME
- **Type**: `string`
- **Description**: The `APP_NAME` variable is a global string variable that holds the name of the application, which is set to "Gradle". This variable is used to identify the application in various contexts, such as when specifying how the application appears in the dock on macOS systems.
- **Use**: `APP_NAME` is used to set the application name for display purposes, particularly in macOS dock options.


---
### APP\_BASE\_NAME
- **Type**: `string`
- **Description**: `APP_BASE_NAME` is a global variable that holds the base name of the script being executed. It is determined by using the `basename` command on the script's path, which extracts the file name from the full path.
- **Use**: This variable is used to set the application name in the Java command options, specifically for the `-Dorg.gradle.appname` property.


---
### DEFAULT\_JVM\_OPTS
- **Type**: `string`
- **Description**: The `DEFAULT_JVM_OPTS` variable is a string that contains default Java Virtual Machine (JVM) options for running the Gradle application. In this script, it is set to include options for setting the maximum and minimum heap size to 64 megabytes each (`"-Xmx64m" "-Xms64m"`).
- **Use**: This variable is used to specify default JVM options that are applied when the Gradle application is started, unless overridden by user-specified options.


---
### MAX\_FD
- **Type**: `string`
- **Description**: The `MAX_FD` variable is a global string variable that is used to determine the maximum number of file descriptors that can be opened by the script. It is initially set to the string 'maximum', which indicates that the script should attempt to use the highest possible limit for file descriptors.
- **Use**: This variable is used to set the file descriptor limit using the `ulimit` command, allowing the script to handle more open files if the system permits.


---
### cygwin
- **Type**: `boolean`
- **Description**: The `cygwin` variable is a boolean flag used to determine if the script is running in a Cygwin environment. Cygwin is a POSIX-compatible environment that runs on Windows, allowing Unix-like applications to be compiled and run on Windows systems. This variable is initially set to `false` and is set to `true` if the `uname` command returns a string starting with 'CYGWIN', indicating that the script is being executed in a Cygwin environment.
- **Use**: The `cygwin` variable is used to conditionally execute code specific to Cygwin, such as converting paths to Windows format before running Java.


---
### msys
- **Type**: `boolean`
- **Description**: The `msys` variable is a boolean flag used to determine if the script is running in an MSYS environment, which is a collection of GNU utilities such as bash, make, and gawk, to allow building of applications and deployment of applications that use the Windows API. It is set to `true` if the script detects that it is running on an MSYS system, specifically when the output of the `uname` command starts with 'MINGW'. Otherwise, it remains `false`. This variable is part of the OS-specific support section of the script, which also includes flags for Cygwin, Darwin, and NonStop environments.
- **Use**: The `msys` variable is used to conditionally execute code that is specific to the MSYS environment, such as converting paths to Windows format before running Java.


---
### darwin
- **Type**: `boolean`
- **Description**: The `darwin` variable is a boolean flag used to determine if the operating system is Darwin-based, which is the core of macOS. It is set to `true` if the script detects that it is running on a Darwin system, otherwise it remains `false`. This detection is done by checking the output of the `uname` command.
- **Use**: This variable is used to apply specific configurations or options for macOS systems within the script.


---
### nonstop
- **Type**: `boolean`
- **Description**: The `nonstop` variable is a boolean flag that is used to determine if the script is running on a NonStop operating system. It is initially set to `false` and is updated to `true` if the script detects that it is running on a NonStop system by checking the output of the `uname` command.
- **Use**: This variable is used to conditionally execute code specific to NonStop systems, such as skipping the increase of maximum file descriptors.


---
### CLASSPATH
- **Type**: `string`
- **Description**: The `CLASSPATH` variable is a string that specifies the path to the Gradle wrapper JAR file, which is located at `$APP_HOME/gradle/wrapper/gradle-wrapper.jar`. This path is used by the Java Virtual Machine (JVM) to locate the necessary classes and resources needed to execute the Gradle wrapper.
- **Use**: `CLASSPATH` is used to set the classpath for the JVM, ensuring it can find and load the Gradle wrapper classes during execution.


---
### JAVACMD
- **Type**: `string`
- **Description**: The `JAVACMD` variable is a string that holds the path to the Java executable used to start the Java Virtual Machine (JVM). It is determined based on the `JAVA_HOME` environment variable, defaulting to 'java' if `JAVA_HOME` is not set. This variable ensures that the correct Java executable is used to run the Gradle wrapper script.
- **Use**: `JAVACMD` is used to execute the Java command with the appropriate JVM options and classpath settings for running the Gradle wrapper.


---
### MAX\_FD\_LIMIT
- **Type**: `integer`
- **Description**: `MAX_FD_LIMIT` is a global variable that stores the hard limit on the number of file descriptors that can be opened by the process. It is determined by the `ulimit -H -n` command, which retrieves the maximum number of file descriptors allowed by the system.
- **Use**: This variable is used to set the `MAX_FD` variable to the system's maximum file descriptor limit if `MAX_FD` is set to 'maximum' or 'max', and to attempt to increase the file descriptor limit for the process.


---
### ROOTDIRSRAW
- **Type**: `string`
- **Description**: `ROOTDIRSRAW` is a string variable that stores the list of directories found at the root level of the filesystem. It is generated by executing the `find` command with options to follow symbolic links and limit the search to the first level of directories.
- **Use**: This variable is used to build a pattern for converting Unix-style paths to Windows-style paths in Cygwin or MSYS environments.


---
### SEP
- **Type**: `string`
- **Description**: The `SEP` variable is a string used as a separator in the script. It is initially set to an empty string and is used to concatenate directory paths with a separator character in the script.
- **Use**: `SEP` is used to build a pattern for arguments to be converted via `cygpath` by concatenating directory paths with a separator.


---
### ROOTDIRS
- **Type**: `string`
- **Description**: The `ROOTDIRS` variable is a string that concatenates directory paths found at the root level of the filesystem. It is constructed by iterating over the directories found by the `find` command and appending them to the `ROOTDIRS` variable, separated by a pipe (`|`) character.
- **Use**: `ROOTDIRS` is used to build a pattern for converting Unix-style paths to Windows-style paths in Cygwin or MSYS environments.


---
### OURCYGPATTERN
- **Type**: `string`
- **Description**: `OURCYGPATTERN` is a string variable that holds a regular expression pattern used to match directory paths in Cygwin or MSYS environments. It is constructed by iterating over the root directories found in the system and concatenating them into a single pattern. Additionally, it can be extended with a user-defined pattern specified by the `GRADLE_CYGPATTERN` variable.
- **Use**: This variable is used to identify and convert arguments that match specific directory paths into a format compatible with Cygwin or MSYS.


---
### APP\_ARGS
- **Type**: `string`
- **Description**: `APP_ARGS` is a global variable that stores the command-line arguments passed to the script, formatted for safe use in shell commands. It is created by the `save` function, which escapes each argument to prevent issues with special characters in shell environments.
- **Use**: `APP_ARGS` is used to safely pass user-provided command-line arguments to the Java command executed by the script.


# Functions

---
### warn
The `warn` function outputs a given message to the standard output.
- **Inputs**:
    - `$*`: A variable number of arguments representing the message to be printed.
- **Control Flow**:
    - The function uses the `echo` command to print all arguments passed to it.
    - The message is printed as a single line to the standard output.
- **Output**: The function outputs the provided message to the standard output.


---
### die
The `die` function prints an error message and terminates the script with an exit status of 1.
- **Inputs**:
    - `$*`: A variable number of arguments representing the error message to be printed.
- **Control Flow**:
    - The function starts by printing a blank line.
    - It then prints the error message passed as arguments to the function.
    - Another blank line is printed after the error message.
    - Finally, the script is terminated with an exit status of 1 using the `exit` command.
- **Output**: The function does not return any value; it terminates the script execution with an exit status of 1.


---
### save
The `save` function escapes and formats command-line arguments for safe usage in shell scripts.
- **Inputs**:
    - `i`: A list of command-line arguments to be processed and escaped.
- **Control Flow**:
    - Iterate over each argument provided to the function.
    - For each argument, use `printf` to print the argument followed by a newline.
    - Pipe the output to `sed` to escape single quotes and wrap the argument in single quotes, appending a backslash and newline at the end.
    - Print a space after processing all arguments.
- **Output**: A string of escaped and formatted command-line arguments, each wrapped in single quotes and separated by spaces.


