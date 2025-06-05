# Purpose
The provided content is a documentation file for integrating the LLGuidance library into the llama.cpp project. LLGuidance is a library designed for constrained decoding in Large Language Models, supporting JSON Schemas and context-free grammars using a Lark-like syntax. This file provides detailed instructions on how to build llama.cpp with LLGuidance support, requiring the Rust compiler and `cargo` tool, and explains the interface changes when LLGuidance is enabled. It highlights the performance benefits of using LLGuidance, particularly in token masking, and discusses its adherence to JSON Schema specifications. The document also addresses the reasons for not reusing the GBNF format, emphasizing the efficiency of using a lexer-parser split, and provides guidance on error handling and conversion of existing grammars to the LLGuidance format. This documentation is crucial for developers looking to enhance llama.cpp with advanced constrained decoding capabilities using LLGuidance.
# Content Summary
The provided content is a detailed documentation excerpt for integrating LLGuidance support into the llama.cpp project. LLGuidance is a library designed for constrained decoding in Large Language Models (LLMs), supporting JSON Schemas and context-free grammars (CFGs) using a variant of Lark syntax. It is noted for its speed and comprehensive JSON Schema coverage, though it requires the Rust compiler, adding complexity to the build process.

### Building and Configuration
To enable LLGuidance, developers must build llama.cpp with the `LLAMA_LLGUIDANCE` option using CMake and Make, or an equivalent command for Windows. This setup necessitates the installation of the Rust compiler and the `cargo` tool.

### Interface and Usage
The integration does not introduce new command-line arguments or changes to existing parameters. When enabled, grammars prefixed with `%llguidance` are processed by LLGuidance, and JSON Schema requests are also directed to it. Existing GBNF grammars can be converted to the LLGuidance format using the `gbnf_to_lark.py` script.

### Performance
LLGuidance is optimized for performance, with token mask computation for a llama3 tokenizer averaging 50μs of CPU time. The documentation highlights the efficiency of the lexer/parser split and various optimizations that contribute to this performance.

### JSON Schema Support
LLGuidance closely follows the JSON Schema specification, with specific behaviors such as `additionalProperties` defaulting to `true` and maintaining the order of properties. Unsupported schemas trigger error messages, ensuring no silent failures.

### Lexical and Parsing Considerations
The documentation explains why the GBNF format is not reused, emphasizing the importance of a lexer in processing, which is faster and more efficient. The distinction between lexemes and CFG symbols is crucial, with Lark syntax using uppercase for lexemes and lowercase for CFG symbols. The `gbnf_to_lark.py` script assists in this conversion.

### Error Handling
Current error handling involves printing errors to `stderr` while continuing generation, with potential improvements anticipated in the future.

Overall, this documentation provides comprehensive guidance for developers looking to integrate and utilize LLGuidance within the llama.cpp framework, detailing the build process, interface changes, performance metrics, and error handling strategies.
