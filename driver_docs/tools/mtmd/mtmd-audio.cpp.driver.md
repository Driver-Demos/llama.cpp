# Purpose
This C++ source code file is part of an audio processing library, specifically designed to preprocess audio data for further analysis or machine learning tasks. The primary functionality of this file is to compute the log mel spectrogram of audio samples, which is a crucial step in many audio processing pipelines, including speech recognition and audio classification. The code includes implementations of the Discrete Fourier Transform (DFT) and the Fast Fourier Transform (FFT), which are used to convert time-domain audio signals into frequency-domain representations. These transformations are optimized using precomputed sine and cosine values stored in a global cache, which enhances computational efficiency.

The file is organized into several key components, including the `whisper_preprocessor` namespace, which contains functions and structures for audio preprocessing. The [`whisper_global_cache`](#whisper_preprocessor(anonymous)::whisper_global_cache::whisper_global_cache) structure precomputes and stores values for sine, cosine, and Hann window functions, which are used in the FFT calculations. The [`log_mel_spectrogram`](#whisper_preprocessorlog_mel_spectrogram) function is a central part of the file, responsible for generating the mel spectrogram by applying the FFT to windowed audio samples and then mapping the frequency components to the mel scale using precomputed filter banks. The file also includes a function to split the resulting mel spectrogram into smaller chunks, which is useful for processing large audio files in manageable segments. Additionally, the file defines a namespace `whisper_precalc_filters` that provides precomputed mel filter banks, which are essential for converting frequency-domain data into the mel scale. Overall, this file provides a focused and efficient implementation of audio preprocessing techniques, suitable for integration into larger audio analysis systems.
# Imports and Dependencies

---
- `mtmd-audio.h`
- `cmath`
- `cstdint`
- `cstring`
- `thread`
- `vector`
- `fstream`
- `algorithm`


# Data Structures

---
### whisper\_global\_cache<!-- {{#data_structure:whisper_preprocessor::(anonymous)::whisper_global_cache}} -->
- **Type**: `struct`
- **Members**:
    - `sin_vals`: An array of precomputed sine values used in FFT operations.
    - `cos_vals`: An array of precomputed cosine values used in FFT operations.
    - `hann_window`: An array representing the Hann window used in signal processing.
- **Description**: The `whisper_global_cache` struct is designed to optimize Fast Fourier Transform (FFT) operations by storing precomputed sine and cosine values, as well as a Hann window. This caching mechanism reduces the computational overhead associated with repeatedly calculating these values during FFT operations, thereby improving performance. The struct initializes these arrays upon construction and provides methods to fill them with the appropriate values.
- **Member Functions**:
    - [`whisper_preprocessor::(anonymous)::whisper_global_cache::whisper_global_cache`](#whisper_preprocessor(anonymous)::whisper_global_cache::whisper_global_cache)
    - [`whisper_preprocessor::(anonymous)::whisper_global_cache::fill_sin_cos_table`](#whisper_preprocessor(anonymous)::whisper_global_cache::fill_sin_cos_table)
    - [`whisper_preprocessor::(anonymous)::whisper_global_cache::fill_hann_window`](#whisper_preprocessor(anonymous)::whisper_global_cache::fill_hann_window)

**Methods**

---
#### whisper\_global\_cache::whisper\_global\_cache<!-- {{#callable:whisper_preprocessor::(anonymous)::whisper_global_cache::whisper_global_cache}} -->
The `whisper_global_cache` constructor initializes precomputed sine, cosine, and Hann window values for efficient FFT operations.
- **Inputs**: None
- **Control Flow**:
    - The constructor `whisper_global_cache` is called when an instance of the `whisper_global_cache` struct is created.
    - It calls the [`fill_sin_cos_table`](#whisper_preprocessor(anonymous)::whisper_global_cache::fill_sin_cos_table) method to populate the `sin_vals` and `cos_vals` arrays with precomputed sine and cosine values for efficient FFT operations.
    - It calls the [`fill_hann_window`](#whisper_preprocessor(anonymous)::whisper_global_cache::fill_hann_window) method to populate the `hann_window` array with precomputed Hann window values, which are used in signal processing to reduce spectral leakage.
- **Output**: The constructor does not return any value; it initializes the struct's member arrays with precomputed values.
- **Functions called**:
    - [`whisper_preprocessor::(anonymous)::whisper_global_cache::fill_sin_cos_table`](#whisper_preprocessor(anonymous)::whisper_global_cache::fill_sin_cos_table)
    - [`whisper_preprocessor::(anonymous)::whisper_global_cache::fill_hann_window`](#whisper_preprocessor(anonymous)::whisper_global_cache::fill_hann_window)
- **See also**: [`whisper_preprocessor::(anonymous)::whisper_global_cache`](#whisper_preprocessor(anonymous)::whisper_global_cache)  (Data Structure)


---
#### whisper\_global\_cache::fill\_sin\_cos\_table<!-- {{#callable:whisper_preprocessor::(anonymous)::whisper_global_cache::fill_sin_cos_table}} -->
The `fill_sin_cos_table` function precomputes sine and cosine values for a range of angles and stores them in arrays for efficient reuse.
- **Inputs**: None
- **Control Flow**:
    - The function iterates over a range from 0 to `SIN_COS_N_COUNT`.
    - For each iteration, it calculates an angle `theta` based on the current index and the total count.
    - It computes the sine and cosine of `theta` using `sinf` and `cosf` functions, respectively.
    - The computed sine and cosine values are stored in the `sin_vals` and `cos_vals` arrays at the current index.
- **Output**: The function does not return a value; it populates the `sin_vals` and `cos_vals` arrays with precomputed sine and cosine values.
- **See also**: [`whisper_preprocessor::(anonymous)::whisper_global_cache`](#whisper_preprocessor(anonymous)::whisper_global_cache)  (Data Structure)


---
#### whisper\_global\_cache::fill\_hann\_window<!-- {{#callable:whisper_preprocessor::(anonymous)::whisper_global_cache::fill_hann_window}} -->
The `fill_hann_window` function populates an array with values of a Hann window, which is used in signal processing to reduce spectral leakage.
- **Inputs**:
    - `length`: The number of elements to fill in the output array, representing the size of the Hann window.
    - `periodic`: A boolean flag indicating whether the window should be periodic (true) or symmetric (false).
    - `output`: A pointer to a float array where the Hann window values will be stored.
- **Control Flow**:
    - Initialize an offset variable to -1.
    - If the `periodic` flag is true, set the offset to 0.
    - Iterate over the range from 0 to `length`, calculating the Hann window value for each index `i` using the formula `0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)))` and store it in the `output` array.
- **Output**: The function does not return a value; it modifies the `output` array in place to contain the calculated Hann window values.
- **See also**: [`whisper_preprocessor::(anonymous)::whisper_global_cache`](#whisper_preprocessor(anonymous)::whisper_global_cache)  (Data Structure)



# Functions

---
### dft<!-- {{#callable:whisper_preprocessor::dft}} -->
The `dft` function performs a naive Discrete Fourier Transform (DFT) on a real-valued input array, producing a complex-valued output array.
- **Inputs**:
    - `in`: A pointer to a float array representing the real-valued input signal of length N.
    - `N`: An integer representing the number of samples in the input signal.
    - `out`: A pointer to a float array where the complex-valued output will be stored, with a length of 2*N to accommodate real and imaginary parts.
- **Control Flow**:
    - Calculate the step size for accessing precomputed sine and cosine values based on the input size N.
    - Iterate over each frequency bin k from 0 to N-1.
    - For each k, initialize real (re) and imaginary (im) components to zero.
    - Iterate over each sample n from 0 to N-1 to compute the DFT for the k-th frequency bin.
    - Calculate the index for accessing precomputed sine and cosine values using the formula (k * n * sin_cos_step) % SIN_COS_N_COUNT.
    - Accumulate the real part by adding the product of the input sample and the cosine value at the calculated index.
    - Accumulate the imaginary part by subtracting the product of the input sample and the sine value at the calculated index.
    - Store the computed real and imaginary parts in the output array at positions k*2 and k*2+1, respectively.
- **Output**: The function outputs a complex-valued array stored in the `out` parameter, where each pair of consecutive elements represents the real and imaginary parts of the DFT result for each frequency bin.


---
### fft<!-- {{#callable:whisper_preprocessor::fft}} -->
The `fft` function implements a recursive Cooley-Tukey Fast Fourier Transform (FFT) algorithm to convert a real-valued input array into a complex-valued frequency domain representation.
- **Inputs**:
    - `in`: A pointer to a float array representing the real-valued input signal.
    - `N`: An integer representing the number of elements in the input array, which must be a power of two.
    - `out`: A pointer to a float array where the complex-valued output will be stored, with a size of at least 2*N to accommodate real and imaginary parts.
- **Control Flow**:
    - Check if N is 1, in which case the output is simply the input with an imaginary part of 0, and return.
    - Calculate half_N as N/2 and check if N is odd; if so, use the naive Discrete Fourier Transform (DFT) instead and return.
    - Separate the input into even and odd indexed elements, storing them in temporary arrays.
    - Recursively call the fft function on the even and odd arrays to compute their FFTs.
    - Use precomputed sine and cosine values from a global cache to combine the results of the even and odd FFTs into the final output array, using the Cooley-Tukey FFT algorithm.
- **Output**: The function outputs a complex-valued array stored in the `out` parameter, representing the frequency domain transformation of the input signal.
- **Functions called**:
    - [`whisper_preprocessor::dft`](#whisper_preprocessordft)


---
### log\_mel\_spectrogram\_worker\_thread<!-- {{#callable:whisper_preprocessor::log_mel_spectrogram_worker_thread}} -->
The `log_mel_spectrogram_worker_thread` function computes the log-mel spectrogram for a segment of audio samples using a Hann window and FFT, and stores the result in a shared `whisper_mel` object.
- **Inputs**:
    - `ith`: The index of the current thread, used to determine which segment of the audio samples this thread will process.
    - `hann`: A pointer to an array containing the Hann window coefficients used to window the audio samples.
    - `samples`: A reference to a vector of audio samples to be processed.
    - `n_samples`: The total number of audio samples in the `samples` vector.
    - `frame_size`: The size of each frame to be processed, in samples.
    - `frame_step`: The step size between consecutive frames, in samples.
    - `n_threads`: The total number of threads being used to process the audio samples.
    - `filters`: A `whisper_filters` object containing the mel filter bank data used to convert FFT results to mel spectrogram values.
    - `mel`: A reference to a `whisper_mel` object where the computed log-mel spectrogram will be stored.
- **Control Flow**:
    - Initialize `fft_in` and `fft_out` vectors for FFT input and output, respectively.
    - Assert that `n_fft` is equal to `1 + (frame_size / 2)` to ensure correct FFT size.
    - Iterate over frames of audio samples, starting from the index `i` and incrementing by `n_threads` to distribute work across threads.
    - For each frame, calculate the offset and apply the Hann window to the samples, storing the result in `fft_in`.
    - If the frame is shorter than `frame_size`, pad the remaining `fft_in` with zeros.
    - Perform FFT on `fft_in` and store the result in `fft_out`.
    - Calculate the squared modulus of the complex FFT output and store it back in `fft_out`.
    - For each mel filter, compute the weighted sum of the FFT output using the filter coefficients, take the logarithm, and store the result in the `mel` data array.
    - If no valid frames are processed, fill the `mel` data array with a default log value.
- **Output**: The function outputs the computed log-mel spectrogram values into the `mel` data array, which is part of the `whisper_mel` object passed by reference.
- **Functions called**:
    - [`whisper_preprocessor::fft`](#whisper_preprocessorfft)


---
### log\_mel\_spectrogram<!-- {{#callable:whisper_preprocessor::log_mel_spectrogram}} -->
The `log_mel_spectrogram` function computes the log-mel spectrogram of audio samples using multi-threading and returns the result in a `whisper_mel` structure.
- **Inputs**:
    - `samples`: A pointer to an array of audio samples represented as floats.
    - `n_samples`: The number of audio samples in the `samples` array.
    - `sample_rate`: The sample rate of the audio, though it is not used in the function.
    - `frame_size`: The size of each frame for the spectrogram computation, must match `WHISPER_N_FFT`.
    - `frame_step`: The step size between frames for the spectrogram computation.
    - `n_mel`: The number of mel bands to use in the spectrogram.
    - `n_threads`: The number of threads to use for parallel processing.
    - `filters`: A `whisper_filters` object containing the mel filter banks.
    - `debug`: A boolean flag indicating whether to output the spectrogram data to a JSON file for debugging.
    - `mel`: A reference to a `whisper_mel` object where the computed spectrogram will be stored.
- **Control Flow**:
    - Assert that the frame size matches `WHISPER_N_FFT` and retrieve the Hann window from the global cache.
    - Calculate padding lengths and initialize a padded samples vector with zero-padding and reflective padding.
    - Set the number of mel bands and calculate the number of frames for the spectrogram, resizing the `mel` data vector accordingly.
    - Create worker threads to process the spectrogram computation in parallel, each calling [`log_mel_spectrogram_worker_thread`](#whisper_preprocessorlog_mel_spectrogram_worker_thread).
    - Join all worker threads after processing is complete.
    - Find the maximum value in the `mel` data, adjust it by subtracting 8.0, and clamp all values below this maximum to the adjusted maximum.
    - Normalize the `mel` data by scaling each value.
    - If `debug` is true, output the `mel` data to a JSON file named `log_mel_spectrogram.json`.
    - Return `true` to indicate successful computation.
- **Output**: A boolean value `true` indicating the successful computation of the log-mel spectrogram.
- **Functions called**:
    - [`whisper_preprocessor::log_mel_spectrogram_worker_thread`](#whisper_preprocessorlog_mel_spectrogram_worker_thread)


---
### preprocess\_audio<!-- {{#callable:whisper_preprocessor::preprocess_audio}} -->
The `preprocess_audio` function processes audio samples into mel spectrogram chunks suitable for further analysis or processing.
- **Inputs**:
    - `samples`: A pointer to an array of audio samples represented as floats.
    - `n_samples`: The number of audio samples in the `samples` array.
    - `filters`: A `whisper_filters` object containing filter parameters for mel spectrogram calculation.
    - `output`: A reference to a vector of `whisper_mel` objects where the processed mel spectrogram chunks will be stored.
- **Control Flow**:
    - Check if `n_samples` is zero; if so, return false indicating no processing is done.
    - Call [`log_mel_spectrogram`](#whisper_preprocessorlog_mel_spectrogram) to compute the full mel spectrogram from the audio samples.
    - If [`log_mel_spectrogram`](#whisper_preprocessorlog_mel_spectrogram) fails, return false.
    - Define a constant `frames_per_chunk` as 3000 and assert that the full mel spectrogram length is greater than this value.
    - Iterate over the full mel spectrogram in chunks of `frames_per_chunk` size.
    - For each chunk, create a `whisper_mel` object, copy the relevant data from the full spectrogram, and add it to the `output` vector.
    - Return true indicating successful processing.
- **Output**: Returns a boolean value indicating whether the audio preprocessing was successful (true) or not (false).
- **Functions called**:
    - [`whisper_preprocessor::log_mel_spectrogram`](#whisper_preprocessorlog_mel_spectrogram)


---
### get\_128\_bins<!-- {{#callable:whisper_precalc_filters::get_128_bins}} -->
The `get_128_bins` function initializes and returns a `whisper_filters` object with pre-calculated mel filter bank data for 128 mel bins and 201 FFT points.
- **Inputs**: None
- **Control Flow**:
    - Initialize a `whisper_filters` object named `filters`.
    - Set `filters.n_mel` to 128 and `filters.n_fft` to 201.
    - Create a `std::vector` named `data` with size `filters.n_mel * filters.n_fft`, initialized to 0.0f.
    - Assign specific pre-calculated values to certain indices of the `data` vector.
    - Iterate over each element in `data` and divide it by 1000.0f to scale down the values.
    - Assign the modified `data` vector to `filters.data`.
    - Return the `filters` object.
- **Output**: A `whisper_filters` object containing the pre-calculated mel filter bank data.


