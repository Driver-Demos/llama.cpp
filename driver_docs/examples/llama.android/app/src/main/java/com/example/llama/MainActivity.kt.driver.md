# Purpose
The provided code is an Android application file written in Kotlin, specifically defining the `MainActivity` class, which serves as the entry point for the application. This file is responsible for setting up the main user interface and managing interactions with system services such as the `ActivityManager`, `DownloadManager`, and `ClipboardManager`. The `MainActivity` class extends `ComponentActivity`, indicating that it is a component of the Android app's activity lifecycle. It initializes system services and a `MainViewModel` to handle data and business logic, and it configures a strict mode policy to detect leaked closable objects, which is a common practice to ensure resource management and application stability.

The `onCreate` method in `MainActivity` is crucial as it sets up the user interface using Jetpack Compose, a modern toolkit for building native Android UI. It defines a list of downloadable models, each represented by a `Downloadable` object, which includes a name, a URI for downloading, and a file path for storage. The method also logs the device's current memory status and the path to the external files directory, providing insights into the app's resource usage. The `setContent` function is used to apply the `LlamaAndroidTheme` and render the `MainCompose` composable function, which defines the layout and behavior of the app's main screen.

The `MainCompose` function is a composable that structures the app's UI using a column layout. It includes a scrollable list of messages, an input field for user messages, and a row of buttons for sending, benchmarking, clearing messages, and copying text to the clipboard. Additionally, it iterates over the list of downloadable models, providing a button for each to initiate downloads via the `DownloadManager`. This composable function leverages Jetpack Compose's declarative UI paradigm to create a responsive and interactive user interface, integrating closely with the `MainViewModel` to manage state and handle user interactions.
# Imports and Dependencies

---
- `android.app.ActivityManager`
- `android.app.DownloadManager`
- `android.content.ClipData`
- `android.content.ClipboardManager`
- `android.net.Uri`
- `android.os.Bundle`
- `android.os.StrictMode`
- `android.os.StrictMode.VmPolicy`
- `android.text.format.Formatter`
- `androidx.activity.ComponentActivity`
- `androidx.activity.compose.setContent`
- `androidx.activity.viewModels`
- `androidx.compose.foundation.layout.Box`
- `androidx.compose.foundation.layout.Column`
- `androidx.compose.foundation.layout.Row`
- `androidx.compose.foundation.layout.fillMaxSize`
- `androidx.compose.foundation.layout.padding`
- `androidx.compose.foundation.lazy.LazyColumn`
- `androidx.compose.foundation.lazy.items`
- `androidx.compose.foundation.lazy.rememberLazyListState`
- `androidx.compose.material3.Button`
- `androidx.compose.material3.LocalContentColor`
- `androidx.compose.material3.MaterialTheme`
- `androidx.compose.material3.OutlinedTextField`
- `androidx.compose.material3.Surface`
- `androidx.compose.material3.Text`
- `androidx.compose.runtime.Composable`
- `androidx.compose.ui.Modifier`
- `androidx.compose.ui.unit.dp`
- `androidx.core.content.getSystemService`
- `com.example.llama.ui.theme.LlamaAndroidTheme`
- `java.io.File`


# Functions

---
### availableMemory
The `availableMemory` function retrieves the current memory status of the device using the `ActivityManager` service.
- **Inputs**: None
- **Control Flow**:
    - Create a new `ActivityManager.MemoryInfo` object.
    - Use the `activityManager` to populate the `MemoryInfo` object with the current memory status by calling `getMemoryInfo(memoryInfo)`.
    - Return the populated `MemoryInfo` object.
- **Output**: The function returns an `ActivityManager.MemoryInfo` object containing the current memory status of the device.


---
### MainCompose
The `MainCompose` function is a composable UI component that displays a list of messages, an input field, and buttons for user interaction, while also providing download options for a list of models.
- **Inputs**:
    - `viewModel`: An instance of `MainViewModel` that manages the state and logic for the UI, including messages and actions.
    - `clipboard`: An instance of `ClipboardManager` used to manage clipboard operations, such as copying text.
    - `dm`: An instance of `DownloadManager` used to handle download operations for the models.
    - `models`: A list of `Downloadable` objects representing models that can be downloaded, each containing a name, URI, and file location.
- **Control Flow**:
    - The function starts by creating a `Column` layout to organize UI components vertically.
    - A `LazyColumn` is used within a `Box` to display a scrollable list of messages from the `viewModel`.
    - An `OutlinedTextField` is provided for user input, with its value bound to `viewModel.message` and updates handled by `viewModel.updateMessage`.
    - A `Row` contains several `Button` components for sending messages, benchmarking, clearing messages, and copying messages to the clipboard.
    - Another `Column` iterates over the `models` list, creating a `Downloadable.Button` for each model to facilitate downloading.
- **Output**: The function does not return any value; it defines a UI layout and behavior using Jetpack Compose.


