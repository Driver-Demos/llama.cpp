## Folders
- **[unit](tests/unit.driver.md)**: The `unit` folder in the `llama.cpp` codebase contains a comprehensive suite of unit tests for various functionalities of the `llama.cpp` server, including server operations, chat and completion features, context shifting, embedding, infill, LoRA models, reranking, security, slot management, speculative completion, template application, tokenization, tool calls, and vision API processing.

## Files
- **[.gitignore](tests/.gitignore.driver.md)**: The `.gitignore` file in `llama.cpp/tools/server/tests/` specifies that the `.venv` directory and `tmp` files should be ignored by version control.
- **[conftest.py](tests/conftest.py.driver.md)**: The `conftest.py` file in the `llama.cpp` codebase defines a pytest fixture that automatically stops all server instances after each test is executed.
- **[pytest.ini](tests/pytest.ini.driver.md)**: The `pytest.ini` file in the `llama.cpp` codebase configures pytest to include markers for identifying slow tests and serial tests.
- **[README.md](tests/README.md.driver.md)**: The `README.md` file in the `llama.cpp/tools/server/tests` directory provides instructions for setting up and running Python-based server tests using `pytest`, targeting GitHub workflows job runners with specific configurations and environment variable options.
- **[requirements.txt](tests/requirements.txt.driver.md)**: The `requirements.txt` file in the `llama.cpp` codebase specifies the dependencies needed for testing the server tools, including libraries such as `aiohttp`, `pytest`, `huggingface_hub`, `numpy`, `openai`, `prometheus-client`, `requests`, and `wget`.
- **[tests.sh](tests/tests.sh.driver.md)**: The `tests.sh` file is a shell script for running tests in the `llama.cpp` codebase, with options to include or exclude slow tests based on the `SLOW_TESTS` environment variable.
- **[utils.py](tests/utils.py.driver.md)**: The `utils.py` file in the `llama.cpp` codebase provides utility functions and classes for managing server processes, making HTTP requests, downloading files, and running functions in parallel, specifically for testing purposes.
