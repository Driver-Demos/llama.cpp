# Purpose
This Swift source code file defines a model management and processing system for machine learning models, specifically focusing on handling models related to the "Llama" context. The file contains a `Model` struct and a `LlamaState` class. The `Model` struct is a simple data structure that holds information about a model, including its unique identifier, name, URL, filename, and status. This struct is used to represent both downloaded and undownloaded models, facilitating the management of model states.

The `LlamaState` class is an observable object that manages the state and operations related to these models. It includes properties to track message logs, cache status, and lists of downloaded and undownloaded models. The class provides functionality to load models from disk, manage default models, and perform operations such as loading, benchmarking, and clearing models. The class uses Swift's concurrency features, such as `@MainActor` and `async`, to handle asynchronous operations, ensuring that UI updates and model processing are performed efficiently and without blocking the main thread.

The file is structured to support a broader application that involves model downloading, loading, and processing, likely in a machine learning or AI context. It does not define a public API or external interface directly but provides essential functionality for managing and interacting with models within the application. The use of `LlamaContext` suggests integration with a specific machine learning framework or library, although the details of `LlamaContext` are not provided in this file.
# Imports and Dependencies

---
- `Foundation`
- `Identifiable`
- `MainActor`
- `ObservableObject`
- `Published`
- `Bundle`
- `FileManager`
- `URL`
- `DispatchTime`
- `Task`


# Data Structures

---
### Model
- **Type**: `struct`
- **Members**:
    - `id`: A unique identifier for the model, generated using UUID.
    - `name`: The name of the model.
    - `url`: The URL from which the model can be downloaded.
    - `filename`: The filename of the model.
    - `status`: An optional status indicating the download state of the model.
- **Description**: The `Model` struct is a data structure that represents a machine learning model with attributes for identification, naming, download URL, filename, and an optional status to indicate its download state. It is used within the `LlamaState` class to manage and track models that are either downloaded or available for download.


---
### LlamaState
- **Type**: `class`
- **Members**:
    - `messageLog`: A published string that logs messages and updates.
    - `cacheCleared`: A published boolean indicating if the cache has been cleared.
    - `downloadedModels`: A published array of Model objects representing downloaded models.
    - `undownloadedModels`: A published array of Model objects representing models that are not downloaded.
    - `NS_PER_S`: A constant representing the number of nanoseconds per second.
    - `llamaContext`: An optional private variable of type LlamaContext used for managing model context.
    - `defaultModelUrl`: A computed property that returns the URL of the default model resource.
    - `defaultModels`: A private array of Model objects representing default models available for download.
- **Description**: LlamaState is a class that manages the state of a machine learning model environment, including handling downloaded and undownloaded models, logging messages, and interacting with a LlamaContext for model operations. It provides functionality to load models from disk, manage default models, and perform operations such as model completion and benchmarking. The class uses Swift's @Published property wrapper to allow for reactive updates to its properties, making it suitable for use in SwiftUI applications.


# Functions

---
### loadModelsFromDisk
The `loadModelsFromDisk` function loads model files from the user's documents directory and appends them to the `downloadedModels` list.
- **Inputs**: None
- **Control Flow**:
    - The function attempts to retrieve the URL of the user's documents directory using `getDocumentsDirectory()`.
    - It uses `FileManager` to list all files in the documents directory, skipping hidden files and subdirectories.
    - For each file found, it extracts the file name without the extension to use as the model name.
    - A new `Model` instance is created for each file with the extracted name, the file's name, and a status of 'downloaded'.
    - Each new `Model` instance is appended to the `downloadedModels` list.
    - If an error occurs during this process, it is caught and printed to the console.
- **Output**: The function does not return a value; it updates the `downloadedModels` list with models found on disk.


---
### loadDefaultModels
The `loadDefaultModels` function attempts to load a default model from a specified URL and updates the list of undownloaded models if they are not found on disk.
- **Inputs**: None
- **Control Flow**:
    - The function attempts to load a model using the `loadModel` function with the `defaultModelUrl` as the parameter.
    - If an error occurs during model loading, it appends an error message to `messageLog`.
    - The function iterates over each model in the `defaultModels` list.
    - For each model, it constructs a file URL using the model's filename and checks if the file exists in the documents directory.
    - If the file does not exist, it updates the model's status to 'download' and appends it to the `undownloadedModels` list.
- **Output**: The function does not return any value; it updates the `undownloadedModels` list and `messageLog` as side effects.


---
### getDocumentsDirectory
The `getDocumentsDirectory` function retrieves the URL of the user's document directory in the file system.
- **Inputs**: None
- **Control Flow**:
    - The function calls `FileManager.default.urls` with parameters `.documentDirectory` and `.userDomainMask` to get an array of URLs for the document directory.
    - It returns the first URL from the array, which corresponds to the user's document directory.
- **Output**: The function returns a `URL` object representing the path to the user's document directory.


---
### loadModel
The `loadModel` function attempts to load a model from a given URL and updates the model's status if successful.
- **Inputs**:
    - `modelUrl`: An optional URL of the model to be loaded.
- **Control Flow**:
    - Check if the `modelUrl` is not nil.
    - If `modelUrl` is valid, append a loading message to `messageLog`.
    - Attempt to create a `LlamaContext` using the model's path.
    - If successful, append a success message to `messageLog` and update the model's status to 'downloaded'.
    - If `modelUrl` is nil, append a message to `messageLog` indicating that a model should be loaded from the list.
- **Output**: The function does not return a value but updates the `messageLog` and modifies the `downloadedModels` and `undownloadedModels` lists.


---
### updateDownloadedModels
The `updateDownloadedModels` function removes a model from the list of undownloaded models based on its name.
- **Inputs**:
    - `modelName`: A string representing the name of the model to be updated.
    - `status`: A string representing the new status of the model, although it is not used in the function logic.
- **Control Flow**:
    - The function iterates over the `undownloadedModels` list.
    - It removes any model from the `undownloadedModels` list whose name matches the `modelName` provided as input.
- **Output**: The function does not return any value; it modifies the `undownloadedModels` list in place.


---
### complete
The `complete` function asynchronously initializes and executes a text completion process using a LlamaContext, updating the message log with the results and performance metrics.
- **Inputs**:
    - `text`: A string representing the initial text input for the completion process.
- **Control Flow**:
    - Check if `llamaContext` is not nil; if nil, return immediately.
    - Record the start time for the completion process.
    - Initialize the completion process with the provided text using `llamaContext.completion_init`.
    - Calculate the time taken for the heat-up phase and update the message log with the initial text.
    - Start a detached task to perform the completion loop until `llamaContext.is_done` returns true.
    - Within the loop, retrieve results from `llamaContext.completion_loop` and append them to the message log.
    - After completion, calculate the total generation time and tokens per second.
    - Clear the `llamaContext` and update the message log with completion details and performance metrics.
- **Output**: The function does not return a value but updates the `messageLog` with the completion results and performance metrics.


---
### bench
The `bench` function performs a benchmarking test on the current model context to measure its performance and logs the results.
- **Inputs**: None
- **Control Flow**:
    - Check if `llamaContext` is not nil; if it is nil, return immediately.
    - Append a new line and 'Running benchmark...' to `messageLog`.
    - Retrieve and append model information to `messageLog`.
    - Record the start time using `DispatchTime.now().uptimeNanoseconds`.
    - Call `llamaContext.bench` with parameters `pp: 8, tg: 4, pl: 1` to perform a heat-up benchmark.
    - Record the end time and calculate the heat-up time in seconds.
    - Append the heat-up time to `messageLog`.
    - If the heat-up time exceeds 5 seconds, append a message to `messageLog` indicating the benchmark is aborted and return.
    - Call `llamaContext.bench` with parameters `pp: 512, tg: 128, pl: 1, nr: 3` to perform the main benchmark.
    - Append the benchmark result to `messageLog`.
- **Output**: The function does not return a value but updates the `messageLog` with benchmark results and performance metrics.


---
### clear
The `clear` function asynchronously clears the current LlamaContext and resets the message log.
- **Inputs**: None
- **Control Flow**:
    - Check if `llamaContext` is not nil; if it is nil, return immediately.
    - Call the `clear` method on `llamaContext` asynchronously.
    - Reset the `messageLog` to an empty string.
- **Output**: The function does not return any value.


