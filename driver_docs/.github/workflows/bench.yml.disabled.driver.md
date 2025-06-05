# Purpose
The provided file is a GitHub Actions workflow configuration file written in YAML. It is designed to automate the benchmarking process for a software project, specifically targeting the `llama.cpp` server. The workflow is triggered by various events such as manual dispatch, pushes to the master branch, and pull request updates, and it can also be scheduled to run at specific times. The file defines a job named `bench-server-baseline` that runs on specified Azure GPU instances, setting up the environment, installing necessary dependencies, and executing a series of steps to benchmark the server's performance. The results, including performance metrics and visualizations, are uploaded as artifacts and optionally commented on pull requests. This file is crucial for continuous integration and performance testing within the codebase, ensuring that changes do not degrade the server's performance.
# Content Summary
This configuration file is a GitHub Actions workflow designed for benchmarking the `llama.cpp` server. The workflow is named "Benchmark" and is currently disabled due to issues, as noted in the comments. It is triggered by several events: manual dispatch via `workflow_dispatch`, pushes to the `master` branch, and specific pull request activities. Additionally, it is scheduled to run daily at 2:04 AM UTC.

Key inputs for the workflow include the Azure GPU series to be used, the commit SHA1 to build, and the duration of the benchmark, with a default duration of 10 minutes. The workflow is configured to run on specific paths and branches, ensuring that only relevant changes trigger the benchmark.

The workflow defines a single job, `bench-server-baseline`, which runs on the `Standard_NC4as_T4_v3` environment. It uses a matrix strategy to test different model and file type combinations, specifically `phi-2` with `q4_0`, `q8_0`, and `f16` file types. The job includes several steps:

1. **Clone the Repository**: Uses the `actions/checkout` action to clone the repository at the specified commit or branch.
2. **Set Up Python Environment**: Installs a Python virtual environment and the necessary dependencies from `requirements.txt`.
3. **Install Prometheus**: Downloads and runs Prometheus for monitoring, ensuring it is active before proceeding.
4. **Set Up Go Environment**: Installs Go version 1.21.
5. **Install k6 and xk6-sse**: Builds the k6 load testing tool with the xk6-sse extension.
6. **Build the Server**: Uses CMake to configure and build the `llama-server` with specific options for CUDA and server settings.
7. **Download Dataset**: Retrieves a dataset from Hugging Face for benchmarking purposes.
8. **Run Server Benchmark**: Executes the benchmark script with various parameters, including model path, parallel users, and batch sizes.
9. **Upload Artifacts**: Saves benchmark results, including images and logs, as artifacts.
10. **Update Commit Status**: Uses the `Sibz/github-status-action` to update the commit status on GitHub.
11. **Upload Benchmark Images**: Attempts to upload images to Imgur, with error handling for potential failures.
12. **Extract Mermaid Diagrams**: Extracts and sets environment variables for mermaid diagrams used in the benchmark results.
13. **Extract Image URLs**: Retrieves URLs of uploaded images for use in comments.
14. **Comment on Pull Requests**: Adds a detailed comment to pull requests with benchmark results, including performance metrics and visualizations.

The workflow is designed to provide comprehensive benchmarking of the `llama.cpp` server, with detailed reporting and visualization of performance metrics. It is a critical tool for developers to assess the impact of changes on server performance and ensure optimal operation across different configurations.
