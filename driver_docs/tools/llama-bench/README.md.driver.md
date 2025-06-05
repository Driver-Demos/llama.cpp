# Purpose
The provided content is a documentation file for a performance testing tool named "llama-bench," which is part of the llama.cpp project. This file serves as a comprehensive guide for users to understand how to utilize the llama-bench tool to conduct performance tests on different models, focusing on text generation and prompt processing. The document outlines the syntax for using the tool, including various command-line options that allow users to customize test parameters such as model selection, batch sizes, number of threads, and GPU layer offloading. It also provides examples of different test scenarios and explains the output formats available, including Markdown, CSV, JSON, JSONL, and SQL, which are crucial for analyzing and storing test results. This file is essential for developers and researchers who need to benchmark and optimize the performance of models within the llama.cpp framework, offering both narrow functionality in terms of specific test configurations and broad applicability through its support for multiple output formats and test scenarios.
# Content Summary
The provided content is a detailed documentation for `llama-bench`, a performance testing tool for the `llama.cpp` project. This tool is designed to evaluate the performance of different models and configurations by running various tests and outputting the results in multiple formats. The document is structured with a table of contents, syntax description, examples, and output format specifications.

### Key Functional Details:

1. **Syntax and Options**: 
   - The tool is invoked using `llama-bench [options]`, where options include parameters for NUMA mode, repetitions, priority, delay, output format, verbosity, and progress indicators.
   - Test parameters allow specification of model files, prompt and generation sizes, batch sizes, cache types, thread numbers, CPU and GPU configurations, and more. These parameters can be specified multiple times or as ranges to run comprehensive tests.

2. **Test Types**:
   - `llama-bench` supports three primary test types: prompt processing (`pp`), text generation (`tg`), and a combination of both (`pg`). Each test type can be configured with various parameters to assess performance under different conditions.

3. **Repetition and Averaging**:
   - Tests can be repeated multiple times as specified by the `-r` option, with results averaged to provide metrics such as average tokens per second and standard deviation.

4. **Output Formats**:
   - The tool supports multiple output formats including Markdown, CSV, JSON, JSONL, and SQL. The default output is Markdown, but users can specify their preferred format using the `-o` option. Each format is suitable for different use cases, such as human-readable reports or data import into databases.

5. **Examples**:
   - The documentation provides examples of how to use `llama-bench` for different scenarios, such as testing text generation with various models, processing prompts with different batch sizes, using different numbers of threads, and offloading layers to the GPU. These examples illustrate the flexibility and configurability of the tool.

6. **Detailed Output**:
   - The output includes comprehensive details about the test environment, model specifications, and performance metrics. For instance, JSON and JSONL formats provide detailed records of each test, including build information, hardware details, and performance statistics.

Overall, `llama-bench` is a versatile tool for benchmarking the performance of models in the `llama.cpp` framework, offering extensive configurability and detailed output to aid in performance analysis and optimization.
