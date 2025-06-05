# Purpose
This file is a spec file used for building and packaging software into an RPM (Red Hat Package Manager) format, specifically for RPM-based Linux distributions. It provides detailed instructions for compiling the "llama.cpp" project, which is a CPU-based inference engine for Meta's LLaMA models, using C/C++ without GPU support. The file includes metadata such as the package name, version, release information, and dependencies required for building and running the software. It also contains script sections for preparing the build environment, compiling the source code, and installing the resulting binaries and configuration files. The file is crucial for automating the packaging process, ensuring that the software can be easily distributed and installed on compatible systems, and it includes service configuration for systemd to manage the software as a service.
# Content Summary
This file is a specification for building a Source RPM (SRPM) package for the `llama.cpp` project, which facilitates CPU inference of Meta's LLaMA models using C/C++ without GPU support. The SRPM is designed for RPM-based Linux distributions, and the file outlines the necessary steps and dependencies for building and packaging the software.

Key technical details include:

1. **Metadata and Versioning**: The package is named `llama.cpp`, and its version is dynamically set using the current date in the `YYYYMMDD` format. This approach is used due to the current tagging system based on hashes, which do not sort alphabetically. The release number is set to `1`, and the package is licensed under the MIT License.

2. **Source and Dependencies**: The source code is fetched from the `llama.cpp` GitHub repository. The build process requires several development tools and libraries, including `coreutils`, `make`, `gcc-c++`, `git`, and `libstdc++-devel`. The runtime requires `libstdc++`.

3. **Build and Installation**: The build process is executed using the `make` command with parallel jobs. The installation phase involves copying the built binaries (`llama-cli`, `llama-server`, and `llama-simple`) to the appropriate binary directory. Additionally, a systemd service file (`llama.service`) is created to manage the `llama-server` as a service, and a configuration file (`/etc/sysconfig/llama`) is set up to specify runtime arguments.

4. **Service Configuration**: The systemd service is configured to start the `llama-server` with specified environment variables, and it is set to not restart automatically. The service is dependent on several system targets, ensuring it starts after essential services are available.

5. **Package Lifecycle Scripts**: The file includes placeholders for pre-installation, post-installation, pre-uninstallation, and post-uninstallation scripts, although they are currently empty.

6. **Changelog**: A section is reserved for maintaining a changelog, although it is currently empty.

Overall, this file provides a comprehensive guide for building and packaging the `llama.cpp` project as an RPM package, detailing the necessary dependencies, build instructions, and service configuration for deployment on RPM-based systems.
