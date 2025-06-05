# Purpose
The provided content is documentation for a command-line interface (CLI) tool named `llama-gguf-hash`, which is designed to hash GGUF files to detect differences at both the model and tensor levels. This tool is particularly useful for ensuring the integrity and consistency of tensor data, even when metadata within the GGUF key-value store is updated. The file outlines various command-line options for different hashing algorithms, such as xxh64, sha1, sha256, and UUID, allowing users to choose the level of security and speed they require. The documentation is structured to cater to different stakeholders, including maintainers, model creators, and model users, by providing functionality for detecting tensor inconsistencies, generating consistent UUIDs, and ensuring tensor layer integrity. This file is crucial for developers and users within a codebase that relies on GGUF files, as it provides a mechanism to verify and validate the integrity of tensor data, which is essential for maintaining the reliability and accuracy of machine learning models.
# Content Summary
The `llama-gguf-hash` is a command-line interface (CLI) tool designed to hash GGUF files, allowing users to detect differences at both the model and tensor levels. This tool is particularly useful for ensuring the integrity of tensor data, even when metadata within the GGUF key-value store is updated. It provides several hashing options, including xxh64 (default), sha1, sha256, and UUIDv5, each serving different purposes and user needs.

Key command-line options include:
- `--help`: Displays the help message.
- `--xxh64`, `--sha1`, `--sha256`, `--uuid`: Selects the hashing algorithm.
- `--all`: Uses all available hash algorithms.
- `--no-layer`: Excludes per-layer hashing.
- `-c`, `--check <manifest>`: Verifies the GGUF file against a provided manifest.

The tool is designed to hash GGUF tensor payloads on a per-tensor layer basis, in addition to hashing the entire tensor model. This dual-level hashing allows for initial checks of the entire tensor layer, with the option to narrow down inconsistencies to specific tensor layers if needed. This feature is particularly beneficial for maintainers during development and automated testing, as it helps quickly identify faulty tensor layers.

For model creators, the tool offers optional consistent UUID generation based on model tensor content, which is useful for database keys. For model users, the tool provides assurance of tensor layer integrity, even if metadata is updated, by using secure hashing algorithms like sha256.

The default behavior of the program, if no arguments are provided, is to use the xxhash's xxh32 mode due to its speed, making it suitable for automated tests. However, the tool also supports xxh64 and xxh128, with xxh64 being the preferred choice for 64-bit systems.

The documentation provides examples of how to compile the tool using CMake and make, as well as how to generate and verify manifests. The manifest generation includes multiple hash types and per-tensor layer hashes, excluding UUIDs, which are IDs rather than hashes. Verification can be performed using the highest security strength hash by default, or users can specify a faster hash like xxh64.

The tool relies on micro C libraries for hashing, installed via the clib C package manager, including libraries for xxHash, sha1, and sha256. This ensures that the tool is lightweight and efficient, suitable for integration into various development workflows.
