# Purpose
The provided content is a YAML configuration file for a bug report template used in a software codebase, specifically for issues related to model evaluation in the llama.cpp project. This file is designed to standardize the process of reporting bugs by guiding users through a structured form that collects essential information about the issue. The template includes various fields such as software version, operating system, hardware details, and specific models used, ensuring that all necessary data is captured to facilitate efficient troubleshooting. The file's content is organized into several components, each with a specific purpose, such as text areas for detailed descriptions and dropdowns for selecting affected systems and backends. This structured approach helps maintain consistency in bug reports, making it easier for developers to identify and address issues within the codebase.
# Content Summary
This file is a bug report template designed for issues related to model evaluation in the llama.cpp project. It is structured to guide users in providing detailed and relevant information when reporting bugs, particularly those involving incorrect model evaluation results or crashes during model evaluation. The template is organized into several sections, each requiring specific information to help developers diagnose and address the reported issues effectively.

Key sections include:

1. **Version Information**: Users must specify the version of the software they are using, which can be obtained using the `--version` command with the `llama-cli` binary. This information is crucial for identifying whether the bug is version-specific.

2. **Operating Systems**: Users are required to indicate which operating systems are affected by the bug. The template provides options for Linux, Mac, Windows, BSD, and an option for other systems, allowing for multiple selections.

3. **GGML Backends**: This section requires users to specify which GGML backends are affected. Options include AMX, BLAS, CPU, CUDA, HIP, Kompute, Metal, Musa, RPC, SYCL, and Vulkan, with the ability to select multiple backends.

4. **Hardware Details**: Users must describe the CPUs and GPUs they are using, which helps in identifying hardware-specific issues.

5. **Model Information**: Users can provide details about the models and their quantization levels that were in use when the bug occurred. This section is optional but can be valuable for reproducing the issue.

6. **Problem Description and Reproduction Steps**: A detailed summary of the problem and the steps to reproduce it is required. Users are encouraged to include specific hardware, compile flags, or command line arguments that may be relevant.

7. **First Bad Commit**: If the bug was not present in earlier versions, users are encouraged to identify the first commit where the bug appeared, potentially using a git bisect.

8. **Relevant Log Output**: Users must provide any relevant log output, including commands entered and any generated text. This information is automatically formatted into code for clarity.

Overall, this template is designed to ensure that bug reports are comprehensive and structured, facilitating efficient troubleshooting and resolution by the development team.
