# Purpose
The provided code is a Kotlin file that defines a data class `Downloadable` within the package `com.example.llama`. This file is designed to facilitate the downloading of files using Android's `DownloadManager` and provides a user interface component using Jetpack Compose. The `Downloadable` class encapsulates the details of a downloadable item, including its name, source URI, and destination file. It also defines a companion object that contains a sealed interface `State` to represent the various states of a download process, such as `Ready`, `Downloading`, `Downloaded`, and `Error`.

A key feature of this file is the `Button` composable function, which is responsible for rendering a button in the UI that allows users to initiate, monitor, and complete the download process. The function uses Jetpack Compose's state management to track the download status and progress. It employs coroutines to handle asynchronous operations, such as querying the download manager and updating the download progress. The `onClick` function within the `Button` composable manages the download lifecycle, handling different states and updating the UI accordingly.

Overall, this file provides a focused functionality for managing downloads in an Android application, integrating both backend logic for handling downloads and frontend components for user interaction. It leverages Android's `DownloadManager` for download operations and Jetpack Compose for UI rendering, making it a cohesive component for applications that require file downloading capabilities.
# Imports and Dependencies

---
- `android.app.DownloadManager`
- `android.net.Uri`
- `android.util.Log`
- `androidx.compose.material3.Button`
- `androidx.compose.material3.Text`
- `androidx.compose.runtime.Composable`
- `androidx.compose.runtime.getValue`
- `androidx.compose.runtime.mutableDoubleStateOf`
- `androidx.compose.runtime.mutableStateOf`
- `androidx.compose.runtime.remember`
- `androidx.compose.runtime.rememberCoroutineScope`
- `androidx.compose.runtime.setValue`
- `androidx.core.database.getLongOrNull`
- `androidx.core.net.toUri`
- `kotlinx.coroutines.delay`
- `kotlinx.coroutines.launch`
- `java.io.File`


# Data Structures

---
### Downloadable
- **Type**: `data class`
- **Members**:
    - `name`: The name of the downloadable item.
    - `source`: The URI from which the item will be downloaded.
    - `destination`: The file location where the downloaded item will be saved.
- **Description**: The `Downloadable` data class represents an item that can be downloaded, encapsulating its name, source URI, and destination file. It includes a companion object with a nested sealed interface `State` to represent various states of the download process, such as `Ready`, `Downloading`, `Downloaded`, and `Error`. Additionally, it provides a composable `Button` function to manage the download process, updating the UI based on the current state and progress of the download.


# Functions

---
### Button
The `Button` function is a composable UI component that manages the download state of a file, updating its display and behavior based on the current download status.
- **Inputs**:
    - `viewModel`: An instance of `MainViewModel` used to interact with the application's data and logic.
    - `dm`: An instance of `DownloadManager` used to handle the download requests.
    - `item`: A `Downloadable` object representing the file to be downloaded, including its name, source URI, and destination file path.
- **Control Flow**:
    - Initialize the `status` state to `Downloaded` if the file exists at the destination, otherwise to `Ready`.
    - Initialize the `progress` state to track download progress.
    - Define a coroutine scope for asynchronous operations.
    - Define a suspend function `waitForDownload` to monitor the download progress and update the status accordingly.
    - Define an `onClick` function to handle button clicks based on the current `status`.
    - If the status is `Downloaded`, load the file using the `viewModel`.
    - If the status is `Downloading`, launch a coroutine to wait for the download to complete.
    - If the status is `Ready` or `Error`, delete any existing file at the destination, create a download request, enqueue it, and update the status to `Downloading`.
    - Render a `Button` with an `onClick` handler and a text label that changes based on the current `status`.
- **Output**: A composable `Button` UI element that displays different text and behavior based on the download state of the `Downloadable` item.


---
### waitForDownload
The `waitForDownload` function monitors the progress of a download initiated by the Android DownloadManager and updates the download state accordingly.
- **Inputs**:
    - `result`: An instance of the `Downloading` state containing the download ID.
    - `item`: A `Downloadable` object representing the file being downloaded, including its name, source URI, and destination file.
- **Control Flow**:
    - The function enters an infinite loop to continuously check the download status.
    - It queries the DownloadManager using the download ID from the `result` parameter.
    - If the query returns null, logs an error and returns an `Error` state.
    - If the cursor cannot move to the first row or has no rows, logs an info message and returns a `Ready` state, indicating the download might have been canceled.
    - Retrieves the number of bytes downloaded so far and the total size of the file from the cursor.
    - Closes the cursor after retrieving the necessary data.
    - Checks if the number of bytes downloaded equals the total size, indicating the download is complete, and returns a `Downloaded` state with the `item`.
    - Calculates the download progress as a percentage and updates the `progress` variable.
    - Delays the loop for 1000 milliseconds (1 second) before checking the download status again.
- **Output**: Returns a `State` object representing the current state of the download, which can be `Downloaded`, `Ready`, or `Error`.


---
### onClick
The `onClick` function manages the download process of a file, updating its status and handling user interactions based on the current download state.
- **Inputs**:
    - `None`: The function does not take any direct parameters, but it operates on the `status` and `item` variables within its scope.
- **Control Flow**:
    - The function checks the current `status` of the download.
    - If the status is `Downloaded`, it calls `viewModel.load()` with the file path.
    - If the status is `Downloading`, it launches a coroutine to wait for the download to complete using `waitForDownload()`.
    - If the status is `Ready` or `Error`, it deletes any existing file at the destination, creates a new download request, logs the action, enqueues the download, updates the status to `Downloading`, and recursively calls `onClick()` to handle the new state.
- **Output**: The function does not return a value; it updates the `status` variable and triggers side effects such as starting a download or loading a file.


