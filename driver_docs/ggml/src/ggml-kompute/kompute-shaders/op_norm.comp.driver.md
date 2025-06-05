# Purpose
This code is a GLSL compute shader designed to perform normalization operations on input data, specifically calculating the mean and variance of a tensor and then normalizing the data. The shader is written for OpenGL version 4.5 and utilizes the compute shader capabilities to perform parallel computations on the GPU. The shader processes data in chunks defined by the `local_size_x` layout, which is set to 256, indicating that each workgroup will handle 256 elements concurrently.

The shader reads input data from a buffer (`tensorIn`) and writes the processed output to another buffer (`tensorOut`). It uses push constants to receive parameters such as offsets and sizes, which control the data processing. The shader first calculates the mean of the input data by summing the elements in parallel and then reducing the sum to a single value. This mean is then used to recenter the data by subtracting it from each element. Subsequently, the shader calculates the variance of the recentered data, again using parallel summation and reduction. Finally, it normalizes the data by scaling it with the inverse square root of the variance plus a small epsilon value to prevent division by zero.

The shader is a specialized component intended for use in applications that require efficient data normalization, such as machine learning or image processing tasks. It leverages the parallel processing power of the GPU to handle large datasets efficiently, making it suitable for real-time applications where performance is critical. The use of shared memory and synchronization barriers ensures that the parallel computations are correctly coordinated across the workgroup.
# Functions

---
### main
The `main` function performs mean and variance normalization on input data using parallel computation in a compute shader.
- **Inputs**:
    - `in_[]`: A read-only buffer containing the input float data to be normalized.
    - `out_[]`: A buffer where the normalized output float data will be stored.
    - `pcs`: A structure of push constants containing offsets, sizes, and an epsilon value for numerical stability.
- **Control Flow**:
    - Initialize a local sum array to zero for parallel computation.
    - Calculate the mean by summing input values in parallel and reducing the sum across the workgroup.
    - Broadcast the mean to all threads and recenter the input data by subtracting the mean from each element.
    - Calculate the variance by summing the squared differences of recentered data in parallel and reducing the sum.
    - Broadcast the variance to all threads and compute the scaling factor using the variance and epsilon.
    - Normalize the recentered data by multiplying with the scaling factor and store the result in the output buffer.
- **Output**: The function outputs the normalized data into the `out_[]` buffer, with each element adjusted to have zero mean and unit variance.


