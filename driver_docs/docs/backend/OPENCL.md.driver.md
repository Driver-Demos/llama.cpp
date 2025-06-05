# Purpose
This document is a markdown file that serves as a comprehensive guide for configuring and building the llama.cpp project with OpenCL support. It provides detailed instructions and information on setting up the environment for different operating systems and hardware, specifically targeting Qualcomm Adreno GPUs and certain Intel GPUs. The file outlines the supported operating systems and hardware, details the data types supported by the OpenCL backend, and provides step-by-step instructions for preparing models and building the project on Android and Windows 11 Arm64 platforms. It also includes CMake options for configuring the build process and highlights known issues and future tasks (TODOs) for further development. This file is crucial for developers looking to leverage OpenCL for parallel programming in the llama.cpp project, ensuring compatibility and optimized performance across various platforms and devices.
# Content Summary
This document provides a comprehensive guide for configuring and building the llama.cpp project with OpenCL support, specifically targeting Qualcomm Adreno GPUs and certain Intel GPUs. The document is structured into several sections, each addressing different aspects of the setup and build process.

### Key Sections and Details:

1. **Background**: The document begins by explaining OpenCL, an open standard for cross-platform parallel programming, and its application in programming GPUs. It highlights the integration of OpenCL with llama.cpp, primarily targeting Qualcomm Adreno GPUs, with some support for Intel GPUs.

2. **Operating System and Hardware Support**: It lists supported operating systems (Android, Windows 11 Arm64, and Linux) and verified hardware, specifically Adreno GPUs like Adreno 750, 830, and X85, indicating compatibility and support status.

3. **DataType Supports**: The document specifies supported data types for quantization, namely `Q4_0` and `Q6_K`, with `Q4_0` being optimized for performance.

4. **Model Preparation**: Instructions are provided for preparing and quantizing models using the `llama-quantize` tool, with a focus on achieving optimal performance on Adreno GPUs using `Q4_0` quantization.

5. **CMake Options**: The document outlines CMake options for the OpenCL backend, such as embedding OpenCL kernels and using Adreno-optimized kernels, with default values provided.

6. **Platform-Specific Instructions**:
   - **Android**: Detailed steps for setting up the environment, including installing the Android NDK, OpenCL headers, and libraries, followed by building llama.cpp using CMake and Ninja.
   - **Windows 11 Arm64**: Instructions for setting up the environment on a Snapdragon X Elite device, including installing necessary tools and building llama.cpp with OpenCL support.

7. **Known Issues and TODOs**: The document notes that the OpenCL backend does not currently support Adreno 6xx GPUs and lists future tasks such as optimizing `Q6_K` and supporting `Q4_K`.

This document is essential for developers aiming to build and run llama.cpp with OpenCL support on specified hardware and operating systems, providing detailed setup and build instructions tailored to each platform.
