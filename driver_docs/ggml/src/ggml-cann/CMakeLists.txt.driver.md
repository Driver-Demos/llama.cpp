# Purpose
This file is a CMake configuration script used to set up the build environment for software that relies on the CANN (Compute Architecture for Neural Networks) toolkit, which is typically used for AI and machine learning applications on Ascend hardware. The script performs several key functions: it checks and sets the installation directory for CANN, auto-detects the System on Chip (SoC) type and version, and configures the necessary compile options and library paths based on the detected hardware. The script ensures compatibility with supported platforms (Linux, x86-64, and arm64) and sets up include directories and libraries required for building the software. This configuration is crucial for ensuring that the software is correctly compiled and linked against the appropriate hardware-specific libraries, thereby enabling optimal performance on Ascend devices.
# Content Summary
This configuration file is a CMake script designed to set up the build environment for a software project that utilizes the CANN (Compute Architecture for Neural Networks) toolkit. The script performs several key functions to ensure the correct configuration and compilation of the project.

1. **CANN Installation Directory Configuration**: The script checks if the `CANN_INSTALL_DIR` is set. If not, and if the environment variable `ASCEND_TOOLKIT_HOME` is defined, it assigns `CANN_INSTALL_DIR` to the value of `ASCEND_TOOLKIT_HOME`. This ensures that the toolkit's installation path is correctly set, which is crucial for locating necessary headers and libraries.

2. **Automatic Detection of SoC Type and Version**: The script includes a function `detect_ascend_soc_type` that attempts to automatically detect the System on Chip (SoC) type and version using the `npu-smi` command. If detection fails, the build process is aborted, prompting the user to manually specify the SoC type or verify the device's operational status. The detected SoC version is then used to set the `SOC_TYPE` variable.

3. **SoC Compile Option Construction**: The script constructs a compile option string based on the detected SoC type, formatted as `ASCEND_<Soc_Major_SN>`, where `<Soc_Major_SN>` is derived from the SoC version. This compile option is used to define preprocessor directives during compilation.

4. **Platform and Architecture Checks**: The script verifies that the build is being performed on a supported Unix platform and architecture (x86-64 or arm64). If the platform or architecture is unsupported, the build process is halted with an error message.

5. **Include Directories and Libraries Setup**: The script sets up the include directories and libraries required for the CANN toolkit. It specifies paths to include directories and appends necessary libraries (`ascendcl`, `nnopbase`, `opapi`, `acl_op_compiler`) to the `CANN_LIBRARIES` list.

6. **Source Files and Compilation**: The script uses `file(GLOB ...)` to gather C++ source files and adds them to a backend library named `ggml-cann`. It links this library with the specified CANN libraries and includes directories, ensuring that all necessary components are available during the build process.

7. **Error Handling and Messaging**: Throughout the script, informative status messages are printed to provide feedback on the configuration process. If critical steps fail, such as the inability to find `CANN_INSTALL_DIR`, the script terminates with a fatal error message, guiding the user to resolve the issue.

Overall, this CMake script is essential for configuring the build environment for projects that depend on the CANN toolkit, ensuring that all necessary components are correctly set up and that the build process is executed on a compatible platform and architecture.
