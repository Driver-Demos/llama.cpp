# Purpose
The provided content is a markdown file that serves as a guide for debugging tests within a software codebase. It offers detailed instructions on using a specific script, `debug-test.sh`, to run, execute, or debug tests efficiently, thereby shortening the feedback loop during development. The file provides a narrow functionality focused on test execution and debugging, detailing steps such as setting up the build environment, compiling test binaries, and using the GNU Debugger (GDB) for in-depth analysis. It categorizes the process into clear steps, each with a specific purpose, such as resetting the folder context, compiling in debug mode, and identifying test commands for debugging. This documentation is crucial for developers working on the codebase, as it streamlines the testing process and aids in quickly identifying and resolving issues within the software.
# Content Summary
The provided content is a detailed guide for developers on how to efficiently run, execute, or debug specific tests within a software codebase using a script named `debug-test.sh`. This script is located in the `scripts` folder and is designed to streamline the testing process by allowing developers to focus on individual tests, thereby keeping the feedback loop short.

Key functionalities of the `debug-test.sh` script include:

1. **Running Specific Tests**: Developers can execute a specific test by providing a test name or a regular expression (REGEX) that matches the test name. The script can also take an optional test number to directly run a specific test instance.

2. **Debugging with GDB**: The script supports debugging in GDB by using the `-g` flag. This allows developers to enter an interactive debugging session where they can set breakpoints and inspect the test execution in detail.

3. **Help and Options**: The script provides a help option (`-h`) to display usage instructions and available options, ensuring that developers can easily understand how to utilize the script's capabilities.

The document also outlines the internal workings of the script, broken down into several steps:

- **Step 1: Reset and Setup Folder Context**: It involves creating a new build directory (`build-ci-debug`) to ensure a clean build environment.

- **Step 2: Setup Build Environment and Compile Test Binaries**: This step involves configuring the build environment using CMake with specific flags for debugging and compiling the test binaries.

- **Step 3: Find Tests Matching REGEX**: The script uses `ctest` to list all available tests that match the provided REGEX, with options for verbose output and showing test commands without executing them.

- **Step 4: Identify Test Command for Debugging**: Developers can extract the test command and arguments needed for debugging from the `ctest` output.

- **Step 5: Run GDB on Test Command**: Finally, developers can initiate a GDB session using the test command identified in the previous step, allowing for in-depth debugging of the test.

Overall, this guide provides a comprehensive approach to running and debugging tests, emphasizing efficiency and ease of use for developers working within the codebase.
