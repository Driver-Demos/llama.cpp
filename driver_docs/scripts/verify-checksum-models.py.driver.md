# Purpose
This Python script is designed to verify the integrity of files by comparing their SHA256 checksums against a predefined list. The script reads a file named "SHA256SUMS" located in the parent directory of the script's directory, which contains lines of expected hash values paired with corresponding filenames. For each file listed, the script calculates its SHA256 checksum using the `hashlib` library and compares it to the expected hash. The results of these comparisons are stored in a list, indicating whether each file's checksum is valid or if the file is missing. The script then outputs these results in a tabular format, providing a clear overview of the integrity status of each file.

The script is structured as a standalone utility, utilizing logging to inform users of its progress and any errors encountered, such as a missing hash list file. The primary technical components include the [`sha256sum`](#cpp/scripts/verify-checksum-modelssha256sum) function, which efficiently computes the checksum of a file in chunks to handle large files, and the main logic that processes the hash list and performs the integrity checks. This script is intended to be executed directly and does not define any public APIs or external interfaces, focusing solely on the task of file integrity verification within a specific directory structure.
# Imports and Dependencies

---
- `logging`
- `os`
- `hashlib`


# Global Variables

---
### logger
- **Type**: `logging.Logger`
- **Description**: The `logger` variable is an instance of a `Logger` object from the Python `logging` module, configured to handle logging messages for the application. It is initialized with the name 'verify-checksum-models', which is used to identify the source of the log messages.
- **Use**: This variable is used to log error and informational messages related to the verification of file checksums.


---
### llama\_path
- **Type**: `str`
- **Description**: The `llama_path` variable is a string that represents the absolute path to the parent directory of the script's directory. It is constructed using the `os.path.abspath` and `os.path.join` functions to navigate to the parent directory of the current file's directory.
- **Use**: This variable is used to define the base directory path for accessing files, such as the hash list file, within the parent directory of the script.


---
### hash\_list\_file
- **Type**: `str`
- **Description**: The `hash_list_file` variable is a string that represents the file path to a file named 'SHA256SUMS' located in the parent directory of the script's directory. It is constructed using the `os.path.join` function to concatenate the `llama_path` with the filename 'SHA256SUMS'.
- **Use**: This variable is used to specify the location of the hash list file, which contains SHA256 checksums and filenames, for verifying file integrity.


---
### results
- **Type**: `list`
- **Description**: The `results` variable is a list that stores dictionaries containing the filename, validity of the checksum, and file missing status for each file processed. It is used to accumulate the results of the checksum verification process for each file listed in the hash list file.
- **Use**: This variable is used to store and later display the results of the checksum verification for each file.


# Functions

---
### sha256sum<!-- {{#callable:llama.cpp/scripts/verify-checksum-models.sha256sum}} -->
The `sha256sum` function calculates the SHA-256 checksum of a given file.
- **Inputs**:
    - `file`: The path to the file for which the SHA-256 checksum is to be calculated.
- **Control Flow**:
    - Initialize a block size of 16 MB for reading the file in chunks.
    - Create a bytearray of the specified block size to hold file data temporarily.
    - Initialize a SHA-256 hash object using `hashlib.sha256()`.
    - Create a memory view of the bytearray to facilitate efficient data reading.
    - Open the specified file in binary read mode with no buffering.
    - Enter a loop to read the file into the memory view in chunks of the block size.
    - For each chunk read, update the hash object with the data read.
    - Break the loop when no more data can be read from the file.
    - Return the hexadecimal digest of the hash object as the final checksum.
- **Output**: A string representing the hexadecimal SHA-256 checksum of the file.


