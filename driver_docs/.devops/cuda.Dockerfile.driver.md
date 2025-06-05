# Purpose
The provided content is a Dockerfile, which is used to automate the creation of Docker images for software deployment. This file is specifically designed to configure and build Docker images that utilize NVIDIA's CUDA for GPU-accelerated computing, targeting different environments such as development, runtime, and specific application modes like full, light, and server. The Dockerfile defines multiple stages, including a build stage that compiles the application with CUDA support and a base stage that sets up the runtime environment. It also includes specific configurations for different deployment scenarios, such as a full application setup, a command-line interface (CLI) only setup, and a server setup, each with its own entry point. The relevance of this file to a codebase is significant as it encapsulates the environment setup, dependencies, and build instructions necessary to ensure consistent and reproducible application deployment across various systems.
# Content Summary
This Dockerfile is designed to build and configure a multi-stage Docker image for a CUDA-based application. It is structured to create different variants of the application container, each tailored for specific use cases: full, light (CLI only), and server.

1. **Base Image Configuration**: The Dockerfile begins by defining several build arguments, including `UBUNTU_VERSION` and `CUDA_VERSION`, which specify the Ubuntu and CUDA versions to be used. These are used to construct the base images for both development (`BASE_CUDA_DEV_CONTAINER`) and runtime (`BASE_CUDA_RUN_CONTAINER`) stages, leveraging NVIDIA's CUDA images.

2. **Build Stage**: The `build` stage uses the development container to compile the application. It installs necessary build tools and dependencies such as `build-essential`, `cmake`, `python3`, and others. The application source code is copied into the container, and a conditional build process is executed using CMake, which allows for specifying CUDA architectures if needed. The compiled shared libraries (`*.so`) are then copied to a designated `/app/lib` directory.

3. **Base Runtime Image**: The `base` stage is derived from the runtime container and is optimized by cleaning up unnecessary files to reduce image size. It installs minimal runtime dependencies like `libgomp1` and `curl`.

4. **Full Variant**: The `full` stage extends the `base` image, copying the complete application, including Python scripts and dependencies. It installs additional tools such as `git` and `python3-pip`, and uses `pip` to install Python dependencies listed in `requirements.txt`. The entry point for this variant is set to a script located at `/app/tools.sh`.

5. **Light Variant**: The `light` stage is a minimalistic version that only includes the command-line interface (`llama-cli`). It is intended for environments where only CLI functionality is required. The entry point is set to the CLI executable.

6. **Server Variant**: The `server` stage is configured to run the server component of the application (`llama-server`). It sets an environment variable `LLAMA_ARG_HOST` to bind the server to all network interfaces. A health check is defined to ensure the server is running correctly by making HTTP requests to a health endpoint. The entry point is set to the server executable.

Overall, this Dockerfile is designed to efficiently build and deploy a CUDA-enabled application with flexibility for different deployment scenarios, ensuring that each variant is optimized for its intended use case.
