# Purpose
This code is a GLSL (OpenGL Shading Language) compute shader designed to perform a specific mathematical operation on input data stored in a buffer and write the results to an output buffer. The shader is intended to be executed on a GPU, leveraging parallel processing capabilities to efficiently handle large datasets. The primary function of this shader is to apply the Gaussian Error Linear Unit (GELU) activation function, a common operation in neural networks, to elements of an input tensor and store the results in an output tensor. The shader is structured to process data in chunks, with each invocation of the `main` function handling a segment of the input data.

The shader begins by including a common component file, "common.comp," which likely contains shared definitions or constants such as `SQRT_2_OVER_PI` and `GELU_COEF_A` used in the GELU calculation. It defines two buffer objects, `tensorIn` and `tensorOut`, which are bound to specific binding points and are used to read input data and write output data, respectively. The use of `restrict readonly` and `restrict writeonly` qualifiers indicates that these buffers are accessed in a manner that optimizes memory usage and access patterns. The shader also utilizes push constants, a mechanism for passing small amounts of data to the shader, to specify offsets for reading from and writing to the buffers.

The `main` function calculates a base index using the workgroup ID and processes a fixed number of elements (eight in this case) in a loop. For each element, it reads a value from the input buffer, applies the GELU function, and writes the result to the output buffer. The use of `clamp` ensures that the intermediate values remain within a specified range, preventing potential numerical issues. This shader is a specialized component within a larger graphics or compute pipeline, focusing on efficiently applying a specific mathematical transformation to data in parallel.
# Functions

---
### main
The `main` function processes input tensor data using a Gaussian Error Linear Unit (GELU) activation function and writes the results to an output tensor.
- **Inputs**:
    - `tensorIn`: A buffer containing the input tensor data, accessed as a read-only array of floats.
    - `tensorOut`: A buffer for the output tensor data, accessed as a write-only array of floats.
    - `pcs`: A uniform structure containing push constants `inOff` and `outOff`, which are offsets for input and output data arrays.
- **Control Flow**:
    - Calculate the base index for the current workgroup using `gl_WorkGroupID.x` multiplied by 8.
    - Iterate over a loop from 0 to 7 to process 8 elements per workgroup.
    - For each element, calculate the index `i` by adding the loop counter `x` to `baseIndex`.
    - Retrieve the input value `y` from the input buffer `in_` at the position `i + pcs.inOff`.
    - Compute the GELU activation function on `y` using a formula involving `tanh` and `clamp`.
    - Store the result of the GELU computation into the output buffer `out_` at the position `i + pcs.outOff`.
- **Output**: The function does not return a value but writes the processed data to the `tensorOut` buffer.


