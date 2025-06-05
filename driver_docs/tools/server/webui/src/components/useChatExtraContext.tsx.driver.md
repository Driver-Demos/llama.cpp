# Purpose
This source code file defines a custom React hook, `useChatExtraContext`, which provides functionality for handling the upload and processing of various file types within a chat application. The hook manages a collection of extra context items, such as images, audio, text, and PDF files, which can be added, removed, or cleared. The primary purpose of this hook is to facilitate the conversion of these files into a format suitable for further processing or display, such as converting images and audio files to base64, converting PDFs to text or images, and ensuring text files are not binary.

The file imports several libraries and utilities, including React's `useState` for state management, `pdfjs-dist` for PDF processing, and `react-hot-toast` for displaying notifications. It defines an interface, `ChatExtraContextApi`, which outlines the API provided by the hook, including methods for adding, removing, and clearing items, as well as handling file uploads. The `onFileAdded` function is a key component, responsible for processing uploaded files based on their MIME types, converting them as necessary, and adding them to the context.

Additionally, the file includes several utility functions to assist with file processing. These include `getFileAsBase64` and `getFileAsBuffer` for reading files, `convertPDFToText` and `convertPDFToImage` for handling PDF files, and `isLikelyNotBinary` for determining if a file is text-based. The file also contains a function to convert SVG images to PNG format using the Canvas API. Overall, this code provides a comprehensive solution for managing and processing file uploads in a chat application, ensuring compatibility with various file types and formats.
# Imports and Dependencies

---
- `useState`
- `MessageExtra`
- `toast`
- `useAppContext`
- `pdfjs`
- `pdfjsWorkerSrc`
- `TextContent`
- `TextItem`


# Functions

---
### useChatExtraContext
The `useChatExtraContext` function provides a hook for managing and processing extra context items, such as files, in a chat application.
- **Inputs**: None
- **Control Flow**:
    - Initialize `items` state to manage the list of extra context items.
    - Define `addItems` function to append new items to the existing list.
    - Define `removeItem` function to remove an item by its index from the list.
    - Define `clearItems` function to clear all items from the list.
    - Check if the server supports vision modality using `isSupportVision`.
    - Define `onFileAdded` function to handle file uploads, processing each file based on its MIME type.
    - For image files, convert to base64 and optionally convert SVG to PNG if necessary.
    - For audio files, ensure they are mp3 or wav, then convert to base64.
    - For PDF files, convert to text or images based on configuration and server support.
    - For other text files, read as text and check if they are not binary before adding.
    - Handle errors during file processing and display appropriate error messages using `toast`.
    - Return an object containing the current items and functions to manipulate them.
- **Output**: An object implementing the `ChatExtraContextApi` interface, which includes the current list of items and functions to add, remove, clear, and process files.


---
### getFileAsBase64
The `getFileAsBase64` function reads a file and converts it to a Base64 encoded string, optionally including the data URL prefix.
- **Inputs**:
    - `file`: A `File` object representing the file to be converted to Base64.
    - `outputUrl`: A boolean indicating whether to include the data URL prefix in the output; defaults to `true`.
- **Control Flow**:
    - Create a new `Promise` to handle asynchronous file reading.
    - Instantiate a `FileReader` object to read the file.
    - Set up an `onload` event handler for the `FileReader` to process the file once it is read.
    - In the `onload` handler, check if the file reading was successful by verifying `event.target?.result`.
    - If successful, convert the result to a string and optionally remove the data URL prefix if `outputUrl` is `false`.
    - Resolve the promise with the processed Base64 string.
    - If the file reading fails, reject the promise with an error message.
    - Use `FileReader.readAsDataURL` to read the file as a data URL.
- **Output**: A `Promise` that resolves to a Base64 encoded string of the file, optionally without the data URL prefix.


---
### getFileAsBuffer
The `getFileAsBuffer` function reads a file and returns its content as an ArrayBuffer.
- **Inputs**:
    - `file`: A File object representing the file to be read.
- **Control Flow**:
    - Create a new Promise to handle asynchronous file reading.
    - Instantiate a FileReader object to read the file.
    - Set up an onload event handler for the FileReader to resolve the promise with the file's content as an ArrayBuffer when reading is complete.
    - Set up an onerror event handler to reject the promise with an error if reading fails.
    - Use the FileReader to read the file as an ArrayBuffer.
- **Output**: An ArrayBuffer containing the file's content, or an error if the file reading fails.


---
### convertPDFToText
The `convertPDFToText` function extracts and returns the text content from a PDF file.
- **Inputs**:
    - `file`: A `File` object representing the PDF file to be converted to text.
- **Control Flow**:
    - The function first reads the PDF file as an ArrayBuffer using `getFileAsBuffer`.
    - It then loads the PDF document using `pdfjs.getDocument` and retrieves the number of pages in the document.
    - For each page, it retrieves the text content using `getTextContent` and stores the promises in an array.
    - After all text content promises are resolved, it extracts the text strings from each `TextItem` in the `TextContent`.
    - Finally, it joins all the text strings with newline characters and returns the resulting string.
- **Output**: A `Promise` that resolves to a string containing the concatenated text content of the PDF file.


---
### convertPDFToImage
The `convertPDFToImage` function converts each page of a PDF file into a base64-encoded image using a canvas rendering approach.
- **Inputs**:
    - `file`: A `File` object representing the PDF file to be converted into images.
- **Control Flow**:
    - Retrieve the PDF file as an ArrayBuffer using `getFileAsBuffer`.
    - Load the PDF document using `pdfjs.getDocument` with the ArrayBuffer.
    - Initialize an array `pages` to store promises for each page's image conversion.
    - Iterate over each page of the PDF document.
    - For each page, create a canvas element and set its dimensions based on the page's viewport.
    - Get the 2D rendering context of the canvas and render the page onto the canvas.
    - Convert the rendered canvas to a base64-encoded data URL and store the promise in the `pages` array.
    - Return a promise that resolves when all pages have been converted to images, resulting in an array of base64-encoded image strings.
- **Output**: An array of strings, each representing a base64-encoded image of a page from the PDF file.


---
### isLikelyNotBinary
The `isLikelyNotBinary` function determines if a given string is likely not binary by analyzing its character content.
- **Inputs**:
    - `str`: A string that is to be analyzed to determine if it is likely not binary.
- **Control Flow**:
    - Initialize options for analysis, including prefix length, suspicious character threshold ratio, and maximum null bytes allowed.
    - Check if the input string is empty or effectively empty after considering the prefix length, and return true if so.
    - Iterate over the first 10KB of the string or the entire string if it's shorter, counting suspicious characters and null bytes.
    - Consider characters as suspicious if they are Unicode Replacement Characters, Null Bytes, or certain C0 Control Characters, excluding common text control characters.
    - Return false if the number of null bytes exceeds the maximum allowed.
    - Calculate the ratio of suspicious characters to the sample length and return true if it is within the allowed threshold, otherwise return false.
- **Output**: A boolean value indicating whether the string is likely not binary (true) or likely binary (false).


---
### svgBase64UrlToPngDataURL
The function `svgBase64UrlToPngDataURL` converts a Base64URL encoded SVG string into a PNG Data URL using the browser's Canvas API.
- **Inputs**:
    - `base64UrlSvg`: A string representing the Base64URL encoded SVG image.
- **Control Flow**:
    - Create a new `Image` object.
    - Set the `onload` event handler for the image to draw it onto a canvas once loaded.
    - Create a canvas element and get its 2D rendering context.
    - Set the canvas dimensions to the natural dimensions of the image or default to 300x300 if not available.
    - Fill the canvas with a white background color.
    - Draw the loaded SVG image onto the canvas.
    - Convert the canvas content to a PNG Data URL and resolve the promise with this URL.
    - Set the `onerror` event handler to reject the promise if the image fails to load.
    - Set the `src` of the image to the provided Base64URL encoded SVG string.
- **Output**: A promise that resolves to a string containing the PNG Data URL of the converted image.


