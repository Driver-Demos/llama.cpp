# Purpose
This document is a comprehensive guide for configuring and utilizing Docker images related to the LLaMA model project, specifically for the `llama.cpp` application. It outlines the prerequisites for using Docker, such as having Docker installed and setting up directories for model storage. The document details various Docker images available, each tailored for different functionalities and hardware support, including CUDA, ROCm, MUSA, and Intel SYCL, which cater to different GPU architectures. It provides instructions for running these images, including commands for downloading, converting, and optimizing models, as well as running them in different modes (full, light, server). Additionally, it includes guidance on building Docker images locally with specific configurations for CUDA and MUSA environments, emphasizing the need for appropriate toolkit installations and runtime settings. This file is crucial for developers and users who need to deploy and manage LLaMA models in containerized environments, ensuring compatibility and performance across various hardware platforms.
# Content Summary
This document provides comprehensive instructions for utilizing Docker images associated with the LLaMA project, specifically focusing on the conversion and optimization of LLaMA models into ggml format and their deployment. The document outlines the prerequisites, available Docker images, usage instructions, and guidelines for building Docker images locally with various GPU support.

### Prerequisites
Before using the Docker images, Docker must be installed and running on the system. Additionally, a directory should be created to store large models and intermediate files, such as `/llama/models`.

### Docker Images
The project offers several Docker images hosted on GitHub Container Registry (ghcr.io), each tailored for different functionalities and platforms:

1. **Standard Images:**
   - `ghcr.io/ggml-org/llama.cpp:full`: Contains the main executable and tools for model conversion and 4-bit quantization. Supports `linux/amd64` and `linux/arm64`.
   - `ghcr.io/ggml-org/llama.cpp:light`: Includes only the main executable. Supports `linux/amd64` and `linux/arm64`.
   - `ghcr.io/ggml-org/llama.cpp:server`: Contains only the server executable. Supports `linux/amd64` and `linux/arm64`.

2. **GPU-Enabled Images:**
   - Variants of the above images are available with CUDA, ROCm, MUSA, and SYCL support, each compiled for specific GPU architectures. These images are not tested by CI beyond being built and require local building for custom configurations.

### Usage Instructions
The document provides command-line examples for running the Docker images. The `--all-in-one` command is recommended for downloading, converting, and optimizing models using the full Docker image. The usage examples demonstrate how to run models with different image types (full, light, server) and include options for specifying model paths, prompts, and output configurations.

### Building Docker Locally
Instructions are provided for building Docker images locally with CUDA and MUSA support. The process involves using specific Dockerfiles and setting environment variables such as `CUDA_VERSION` and `MUSA_VERSION`. The local build process allows customization of the Docker images to match the host's GPU environment and architecture.

### GPU-Specific Instructions
For CUDA, the document advises ensuring the `nvidia-container-toolkit` is installed, while for MUSA, it suggests setting `mthreads` as the default Docker runtime. Usage examples for locally built images include the `--gpus` flag and `--n-gpu-layers` option to leverage GPU capabilities.

Overall, this document serves as a detailed guide for developers to effectively utilize and customize Docker images for the LLaMA project, ensuring compatibility with various hardware configurations and optimizing model deployment workflows.
