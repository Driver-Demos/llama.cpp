# Purpose
The provided Kotlin source code defines a class named `LLamaAndroid` within the `android.llama.cpp` package. This class serves as a bridge between Android applications and native C++ libraries, specifically for handling machine learning models. The class is designed to manage the lifecycle of a machine learning model, including loading, executing, and unloading the model. It utilizes Kotlin coroutines to manage asynchronous operations, ensuring that tasks such as model loading and execution are performed on a dedicated background thread, thus preventing blocking the main UI thread.

The `LLamaAndroid` class encapsulates several key functionalities. It provides methods to load a machine learning model from a specified file path, execute a benchmarking process on the loaded model, and send messages to the model for processing. The class uses a `ThreadLocal` state management system to track whether a model is currently loaded, and it ensures that resources are properly allocated and freed. The class also defines several external native functions, which are likely implemented in a C++ library, to perform operations such as loading models, initializing contexts, and executing model-specific tasks.

Additionally, the class implements a singleton pattern through a companion object, ensuring that only one instance of `LLamaAndroid` exists at any time. This design choice is crucial for managing resources efficiently and preventing conflicts in accessing the native library. The class also includes a coroutine dispatcher to handle operations on a dedicated thread, enhancing performance and stability by isolating native code execution from the main application thread. Overall, `LLamaAndroid` provides a structured and efficient interface for integrating native machine learning capabilities into Android applications.
# Imports and Dependencies

---
- `android.util.Log`
- `kotlinx.coroutines.CoroutineDispatcher`
- `kotlinx.coroutines.asCoroutineDispatcher`
- `kotlinx.coroutines.flow.Flow`
- `kotlinx.coroutines.flow.flow`
- `kotlinx.coroutines.flow.flowOn`
- `kotlinx.coroutines.withContext`
- `java.util.concurrent.Executors`
- `kotlin.concurrent.thread`


# Data Structures

---
### LLamaAndroid
- **Type**: `class`
- **Members**:
    - `tag`: A string representing the class name for logging purposes.
    - `threadLocalState`: A ThreadLocal variable holding the current state of the LLamaAndroid instance.
    - `runLoop`: A CoroutineDispatcher for executing tasks on a dedicated thread.
    - `nlen`: An integer representing the maximum length for certain operations.
    - `external functions`: A set of external functions interfacing with native code for model operations.
    - `bench`: A suspend function to benchmark the loaded model.
    - `load`: A suspend function to load a model from a specified path.
    - `send`: A function returning a Flow of strings for processing messages.
    - `unload`: A suspend function to unload the model and free resources.
    - `IntVar`: A private class for managing an integer with thread-safe increment operations.
    - `State`: A sealed interface representing the state of the LLamaAndroid instance.
    - `_instance`: A private static instance of LLamaAndroid enforcing a singleton pattern.
    - `instance`: A static method to access the singleton instance of LLamaAndroid.
- **Description**: The LLamaAndroid class is a singleton data structure designed to interface with native code for loading, managing, and processing machine learning models on Android. It uses a dedicated thread for executing native operations and provides coroutine-based functions for loading models, sending messages, and benchmarking. The class maintains its state using a ThreadLocal variable and ensures thread safety through synchronized operations. It includes external functions for native interactions and provides a flow-based API for message processing.


---
### IntVar
- **Type**: `class`
- **Members**:
    - `value`: A volatile integer field that holds the current value of the IntVar.
- **Description**: The `IntVar` class is a simple data structure designed to encapsulate an integer value with thread-safe operations. It provides a volatile integer field `value` to store the current integer and ensures thread safety by synchronizing the increment operation through the `inc` method. This class is used within the `LLamaAndroid` class to manage integer values that may be accessed or modified concurrently.


---
### State
- **Type**: `sealed interface`
- **Members**:
    - `Idle`: Represents the state when no model is loaded.
    - `Loaded`: Holds the state when a model is loaded, including model, context, batch, and sampler identifiers.
- **Description**: The `State` sealed interface represents the state of the LLamaAndroid instance, with two possible states: `Idle` and `Loaded`. The `Idle` state indicates that no model is currently loaded, while the `Loaded` state contains information about the loaded model, including its model, context, batch, and sampler identifiers. This structure is used to manage the lifecycle of the model within the LLamaAndroid class, ensuring that resources are properly allocated and freed as needed.


---
### State\.Idle
- **Type**: `sealed interface`
- **Members**:
    - `Idle`: Represents the state where no model is loaded and the system is idle.
- **Description**: The `State.Idle` is a sealed interface member of the `State` interface in the `LLamaAndroid` class, representing a state where no model is currently loaded, and the system is not performing any operations. It is used to manage the state transitions within the `LLamaAndroid` class, particularly to check if the system is ready to load a new model or perform other operations.


---
### State\.Loaded
- **Type**: `data class`
- **Members**:
    - `model`: A Long representing the loaded model identifier.
    - `context`: A Long representing the context associated with the model.
    - `batch`: A Long representing the batch identifier for processing.
    - `sampler`: A Long representing the sampler identifier used in processing.
- **Description**: The `State.Loaded` data class is part of a sealed interface `State` within the `LLamaAndroid` class, representing a state where a model is fully loaded and ready for operations. It holds four Long type members: `model`, `context`, `batch`, and `sampler`, which are identifiers for the loaded model, its context, the batch for processing, and the sampler used, respectively. This state is used to manage and track the resources and identifiers necessary for executing operations on the loaded model.


# Functions

---
### bench
The `bench` function benchmarks a loaded model using specified parameters and returns the result as a string.
- **Inputs**:
    - `pp`: An integer representing the number of parallel processes to use during benchmarking.
    - `tg`: An integer representing the number of target tokens for the benchmark.
    - `pl`: An integer representing the prompt length for the benchmark.
    - `nr`: An optional integer representing the number of repetitions for the benchmark, defaulting to 1.
- **Control Flow**:
    - The function is executed within a coroutine context using the `runLoop` dispatcher.
    - It retrieves the current state from `threadLocalState`.
    - If the state is `State.Loaded`, it logs the current state and calls the `bench_model` external function with the provided parameters and the model, context, and batch from the state.
    - If the state is not `State.Loaded`, it throws an `IllegalStateException` indicating that no model is loaded.
- **Output**: A string containing the result of the benchmark operation.


---
### load
The `load` function loads a machine learning model from a specified file path and initializes necessary resources for its operation.
- **Inputs**:
    - `pathToModel`: A string representing the file path to the model that needs to be loaded.
- **Control Flow**:
    - The function is executed within a coroutine context using the `runLoop` dispatcher.
    - It checks the current state of the `threadLocalState`; if it is `State.Idle`, it proceeds to load the model.
    - The `load_model` function is called with `pathToModel` to load the model, and if it returns 0L, an exception is thrown indicating failure.
    - A new context is created using `new_context` with the loaded model, and if it returns 0L, an exception is thrown indicating failure.
    - A new batch is created using `new_batch`, and if it returns 0L, an exception is thrown indicating failure.
    - A new sampler is created using `new_sampler`, and if it returns 0L, an exception is thrown indicating failure.
    - If all resources are successfully initialized, the state is updated to `State.Loaded` with the model, context, batch, and sampler.
    - If the state is not `State.Idle`, an exception is thrown indicating that a model is already loaded.
- **Output**: The function does not return any value but updates the `threadLocalState` to `State.Loaded` with the initialized resources if successful.


---
### send
The `send` function generates a flow of strings by processing a message through a loaded model context, optionally formatting it as chat, and emitting results until completion.
- **Inputs**:
    - `message`: A string representing the message to be processed by the model.
    - `formatChat`: A boolean flag indicating whether the message should be formatted as chat; defaults to false.
- **Control Flow**:
    - The function starts a flow block to emit strings.
    - It checks the current state from `threadLocalState` to ensure a model is loaded.
    - If the state is `State.Loaded`, it initializes a completion process using `completion_init` with the provided message and formatChat flag.
    - A loop runs while `ncur.value` is less than or equal to `nlen`, calling `completion_loop` to get the next string from the model.
    - If `completion_loop` returns a non-null string, it is emitted; otherwise, the loop breaks.
    - After the loop, `kv_cache_clear` is called to clear the context's cache.
- **Output**: A `Flow<String>` that emits strings generated by the model based on the input message.


---
### unload
The `unload` function releases resources associated with a loaded model and resets the state to idle.
- **Inputs**: None
- **Control Flow**:
    - The function is executed within the context of the `runLoop` coroutine dispatcher.
    - It checks the current state using `threadLocalState.get()`.
    - If the state is `State.Loaded`, it proceeds to free resources by calling `free_context`, `free_model`, `free_batch`, and `free_sampler` with the respective identifiers from the state.
    - After freeing resources, it sets the `threadLocalState` to `State.Idle`.
    - If the state is not `State.Loaded`, the function performs no operations.
- **Output**: The function does not return any value; it performs resource cleanup and state transition.


---
### instance
The `instance` function provides a singleton instance of the `LLamaAndroid` class.
- **Inputs**: None
- **Control Flow**:
    - The function accesses a private static variable `_instance` which holds a single instance of `LLamaAndroid`.
    - It returns this instance to the caller.
- **Output**: A singleton instance of the `LLamaAndroid` class.


