# Purpose
This code is a pytest fixture designed to manage server instances during testing, providing narrow functionality specifically for test setup and teardown. It automatically executes before and after each test, ensuring that any server instances are stopped after a test completes. The fixture uses a `yield` statement to separate the pre-test and post-test actions, where the pre-test action is intentionally left empty, and the post-test action iterates over a copy of the `server_instances` set to stop each server. This approach prevents issues related to modifying the set during iteration, ensuring a clean test environment for each test case.
# Imports and Dependencies

---
- `pytest`
- `utils.*`


# Functions

---
### stop\_server\_after\_each\_test<!-- {{#callable:llama.cpp/tools/server/tests/conftest.stop_server_after_each_test}} -->
The function `stop_server_after_each_test` is a pytest fixture that stops all server instances after each test is executed.
- **Decorators**: `@pytest.fixture`
- **Inputs**: None
- **Control Flow**:
    - The function is a pytest fixture with `autouse=True`, meaning it is automatically used by each test without needing to be explicitly included.
    - The function does nothing before each test, as indicated by the `yield` statement.
    - After each test, it creates a copy of the `server_instances` set to avoid issues with modifying the set during iteration.
    - It iterates over the copied set of server instances and calls the [`stop`](utils.py.driver.md#ServerProcessstop) method on each server to stop them.
- **Output**: There is no direct output from this function; it performs an action (stopping servers) as a side effect after each test.
- **Functions called**:
    - [`llama.cpp/tools/server/tests/utils.ServerProcess.stop`](utils.py.driver.md#ServerProcessstop)


