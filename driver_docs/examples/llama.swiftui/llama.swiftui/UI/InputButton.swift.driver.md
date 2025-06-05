# Purpose
The provided SwiftUI code defines a user interface component named `InputButton`, which is a `View` struct designed to facilitate the downloading and management of custom model files. This component is part of a broader application that likely deals with machine learning models, as indicated by the use of terms like "model" and "downloadedModels". The primary functionality of this component is to allow users to input a URL link to a quantized model file, download it, and manage its state within the application.

Key technical components of this code include the use of SwiftUI's state management features, such as `@ObservedObject` and `@State`, to track the download status, input link, and filename. The `download` function is central to the component's functionality, handling the extraction of model information from the URL, initiating the download task using `URLSession`, and updating the application's state based on the download progress and completion. The code also includes error handling for download failures and file operations, ensuring robustness in various scenarios.

The `InputButton` component is designed to be interactive, providing users with buttons to initiate downloads, cancel ongoing downloads, and load downloaded models. The user interface dynamically updates based on the download status, offering a seamless experience for managing model files. This component is likely part of a larger application that involves model management, and it integrates with an external state object, `LlamaState`, to maintain a list of downloaded models and handle cache-related operations.
# Imports and Dependencies

---
- `SwiftUI`
- `URLSession`
- `FileManager`
- `NSKeyValueObservation`
- `HTTPURLResponse`


# Data Structures

---
### InputButton
- **Type**: `struct`
- **Members**:
    - `llamaState`: An observed object of type LlamaState that tracks the state of the llama model.
    - `inputLink`: A private state variable that holds the input link for downloading a model.
    - `status`: A private state variable that indicates the current status of the download process.
    - `filename`: A private state variable that stores the filename of the downloaded model.
    - `downloadTask`: A private state variable that holds the URLSessionDownloadTask for the download process.
    - `progress`: A private state variable that tracks the progress of the download as a double.
    - `observation`: A private state variable that observes the download task's progress.
- **Description**: The InputButton struct is a SwiftUI View that provides a user interface for downloading and managing custom models. It includes state variables to manage the download link, status, filename, and progress of the download task. The struct also defines methods for extracting model information from a link, obtaining file URLs, and handling the download process, including error handling and updating the llamaState with the downloaded model. The user interface consists of text fields and buttons to initiate, cancel, and load downloads, with dynamic behavior based on the current download status.


# Functions

---
### extractModelInfo
The `extractModelInfo` function extracts the model name and filename from a given URL string.
- **Inputs**:
    - `link`: A string representing the URL from which to extract model information.
- **Control Flow**:
    - Attempt to create a URL object from the input string.
    - Extract the last path component of the URL and split it by '.' to get the base name.
    - Split the base name by '-' and drop the last component to derive the model name.
    - Decode any percent-encoded characters in the model name and filename.
    - Return a tuple containing the model name and filename if all steps are successful, otherwise return nil.
- **Output**: A tuple containing the model name and filename if extraction is successful, otherwise nil.


---
### getFileURL
The `getFileURL` function constructs a file URL for a given filename within the user's document directory.
- **Inputs**:
    - `filename`: A string representing the name of the file for which the URL is to be constructed.
- **Control Flow**:
    - Access the user's document directory using `FileManager.default.urls` with the `.documentDirectory` and `.userDomainMask` options.
    - Append the provided `filename` to the document directory path using `appendingPathComponent`.
- **Output**: A `URL` object representing the full path to the specified file within the user's document directory.


---
### download
The `download` function initiates a download task for a model file from a given URL, manages the download progress, and updates the application state upon completion.
- **Inputs**:
    - `self`: The instance of the `InputButton` view, which contains state variables like `inputLink`, `status`, `filename`, `downloadTask`, `progress`, and `observation`.
- **Control Flow**:
    - Extract model information (model name and filename) from the `inputLink` using `extractModelInfo` function.
    - If extraction fails, the function returns early without proceeding.
    - Set the `filename` state variable and update the `status` to 'downloading'.
    - Create a download task using `URLSession.shared.downloadTask` with the URL from `inputLink`.
    - In the download task's completion handler, check for errors and validate the HTTP response status code.
    - If successful, copy the downloaded file from the temporary URL to the destination file URL in the document directory.
    - Update the `llamaState` to reflect the new downloaded model and change the `status` to 'downloaded'.
    - Observe the download progress and update the `progress` state variable accordingly.
    - Start the download task by calling `resume` on it.
- **Output**: The function does not return a value but updates the state variables `status`, `filename`, and `progress`, and modifies the `llamaState` to include the newly downloaded model.


---
### body
The `body` function defines the user interface and interaction logic for downloading and managing a custom model file in a SwiftUI view.
- **Inputs**: None
- **Control Flow**:
    - The function returns a SwiftUI `View` composed of a `VStack` containing a `HStack` and conditional buttons based on the `status` state.
    - The `HStack` includes a `TextField` for inputting a download link and a `Cancel` button to cancel any ongoing download task.
    - If the `status` is 'download', a 'Download Custom Model' button is displayed, which triggers the `download` function when pressed.
    - If the `status` is 'downloading', a button showing the download progress percentage is displayed, allowing the user to cancel the download.
    - If the `status` is 'downloaded', a 'Load Custom Model' button is displayed, which attempts to load the model from the file system.
    - The `onDisappear` modifier cancels any ongoing download task when the view disappears.
    - The `onChange` modifier observes changes to `llamaState.cacheCleared` and updates the `status` and file existence check accordingly.
- **Output**: The function outputs a SwiftUI `View` that provides a user interface for downloading and managing a custom model file.


