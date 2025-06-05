# Purpose
The provided content is a GitHub issue template file, specifically designed for reporting compilation bugs related to the `llama.cpp` project. This file is used to standardize the information collected from users when they encounter issues during the compilation process, ensuring that developers receive all necessary details to diagnose and address the problem efficiently. The template includes several sections, such as markdown instructions, text areas for specific information like the Git commit, operating systems affected, GGML backends, problem description, and compile commands. Each section is equipped with attributes and validations to guide users in providing comprehensive and relevant data. The relevance of this file to the codebase lies in its role in facilitating effective communication between users and developers, streamlining the bug reporting process, and ultimately contributing to the maintenance and improvement of the software.
# Content Summary
This file is a GitHub issue template designed for reporting compilation bugs specifically related to the `llama.cpp` project. It provides a structured format for users to submit detailed bug reports, ensuring that all necessary information is collected to diagnose and resolve compilation issues effectively.

Key components of the template include:

- **Metadata**: The issue is labeled as a "Bug (compilation)" with a description indicating that it pertains to problems encountered during the compilation of `llama.cpp`. The title prefix "Compile bug: " is used to standardize issue titles.

- **Labels**: The issue is automatically tagged with "bug-unconfirmed" and "compilation" to categorize and prioritize it appropriately.

- **Body Structure**: The template is divided into several sections, each requiring specific information:
  - **Markdown Section**: Provides initial instructions, advising users to disable ccache (`-DGGML_CCACHE=OFF`) to verify if the issue persists and suggests clearing the ccache if it resolves the problem.
  - **Git Commit**: A required field where users must specify the commit hash they are attempting to compile, ensuring the issue is linked to a specific code state.
  - **Operating Systems**: A dropdown menu where users select affected operating systems, supporting multiple selections to identify cross-platform issues.
  - **GGML Backends**: Another dropdown for users to indicate which GGML backends are affected, allowing multiple selections to pinpoint backend-specific problems.
  - **Problem Description & Steps to Reproduce**: A required text area for users to describe the issue and provide reproduction steps, with emphasis on identifying problematic compile flags.
  - **First Bad Commit**: An optional field for users to specify the first commit where the bug appeared, encouraging the use of `git bisect` to trace the bug's origin.
  - **Compile Command**: A required field for users to input the exact command used for compilation, formatted as shell code for clarity.
  - **Relevant Log Output**: A required section for users to paste log outputs, also formatted as shell code, to provide context and error details.

This template ensures comprehensive data collection, facilitating efficient troubleshooting and resolution of compilation issues in the `llama.cpp` project.
