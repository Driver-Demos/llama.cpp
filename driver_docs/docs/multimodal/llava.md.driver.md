# Purpose
The provided content is a detailed documentation file for configuring and using the LLaVA (Large Language and Vision Assistant) models, specifically versions 1.5 and 1.6. This file serves as a guide for users to set up and run these models, which are hosted on Hugging Face, a popular platform for machine learning models. The document outlines the steps to build and execute the `llama-mtmd-cli` binary, which is used to interact with the models, and provides instructions for converting model components into the GGUF format, a specific data format used for these models. It includes commands for cloning necessary repositories, installing dependencies, and performing model surgery to split and convert model components. The file also provides usage notes, such as recommended settings for temperature and GPU offloading, and explains how to identify which version of the model is being used based on token counts. This documentation is crucial for developers and researchers working with LLaVA models, as it ensures proper setup and execution within a codebase.
# Content Summary
The provided document is a comprehensive guide for developers working with the LLaVA (Large Language and Vision Assistant) models, specifically versions 1.5 and 1.6. It outlines the steps necessary to set up and utilize these models, which are hosted on Hugging Face, a popular platform for machine learning models.

### Key Technical Details:

1. **Model Variants and Availability**: The document specifies that the implementation supports LLaVA versions 1.5 and 1.6, with links to their respective repositories on Hugging Face. Pre-converted models for version 1.5 (7b and 13b) and a variety of prepared GGUF models for version 1.6 (7b-34b) are available.

2. **Usage Instructions**: Developers are instructed to build the `llama-mtmd-cli` binary to interact with the models. A sample command is provided to demonstrate how to run the CLI with specific model files and configurations, such as using a lower temperature setting for improved quality and enabling GPU offloading.

3. **LLaVA 1.5 Setup**:
   - **Cloning and Installation**: Instructions are provided for cloning the LLaVA and CLIP models and installing necessary Python packages.
   - **Model Conversion**: The process involves using scripts like `llava_surgery.py` and `convert_image_encoder_to_gguf.py` to split and convert model components into the GGUF format.

4. **LLaVA 1.6 Setup**:
   - **Cloning and Installation**: Similar to version 1.5, with additional steps for handling the model's components.
   - **Model Conversion**: The document details the use of `llava_surgery_v2.py` for model surgery, handling both PyTorch and safetensor models, and converting components to GGUF format.
   - **Running the CLI**: Instructions for executing the CLI with the 1.6 model version are provided, along with notes on context requirements and benefits of batched prompt processing.

5. **Chat Template**: Both LLaVA versions require the `vicuna` chat template, which can be activated by adding `--chat-template vicuna` to the command line.

6. **Mode Identification**: The document explains how to identify whether the CLI is running in LLaVA 1.5 or 1.6 mode based on the number of tokens used in image embeddings and prompts.

7. **Additional Notes**: The document includes troubleshooting tips for handling language model conversion incompatibilities and provides a Python script snippet for exporting and converting models using the `transformers` library.

This guide is essential for developers aiming to work with LLaVA models, providing detailed instructions for setup, conversion, and execution, ensuring efficient utilization of these advanced language and vision models.
