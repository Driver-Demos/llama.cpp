# Purpose
This source code file contains two OpenCL kernel functions, `kernel_upscale` and `kernel_upscale_bilinear`, which are designed to perform image upscaling operations on a GPU. The primary purpose of these kernels is to transform a source image into a larger destination image by mapping and interpolating pixel values. The `kernel_upscale` function performs a straightforward nearest-neighbor upscaling, where each pixel in the destination image is directly mapped from a corresponding pixel in the source image based on scaling factors. This is achieved by calculating the source pixel indices using the provided scaling factors and copying the pixel value to the destination.

The `kernel_upscale_bilinear` function, on the other hand, implements bilinear interpolation for upscaling. This method involves calculating the destination pixel value as a weighted average of the four nearest pixels in the source image. The weights are determined by the fractional distances of the destination pixel's coordinates from the source pixels, allowing for smoother transitions and less pixelation compared to nearest-neighbor scaling. The function computes these weights and uses them to interpolate the pixel values, resulting in a more visually appealing upscaled image.

Both kernels are designed to be executed in parallel across multiple threads on a GPU, leveraging OpenCL's capabilities to handle large-scale data processing efficiently. The kernels use global memory pointers to access the source and destination image data, and they calculate the appropriate memory offsets to read and write pixel values. The use of scaling factors and dimension parameters allows these kernels to be flexible and applicable to various image sizes and scaling requirements.
# Functions

---
### kernel\_upscale
The `kernel_upscale` function performs nearest-neighbor upscaling of a source image to a destination image using specified scaling factors.
- **Inputs**:
    - `p_src0`: A pointer to the source image data.
    - `off_src0`: An offset in bytes to the start of the source image data.
    - `p_dst`: A pointer to the destination image data.
    - `off_dst`: An offset in bytes to the start of the destination image data.
    - `nb00`: The byte stride for the first dimension of the source image.
    - `nb01`: The byte stride for the second dimension of the source image.
    - `nb02`: The byte stride for the third dimension of the source image.
    - `nb03`: The byte stride for the fourth dimension of the source image.
    - `ne10`: The number of elements in the first dimension of the destination image.
    - `ne11`: The number of elements in the second dimension of the destination image.
    - `ne12`: The number of elements in the third dimension of the destination image.
    - `ne13`: The number of elements in the fourth dimension of the destination image.
    - `sf0`: The scaling factor for the first dimension.
    - `sf1`: The scaling factor for the second dimension.
    - `sf2`: The scaling factor for the third dimension.
    - `sf3`: The scaling factor for the fourth dimension.
- **Control Flow**:
    - Calculate the base pointers for the source and destination images using the provided offsets.
    - Determine the global index of the current work item using `get_global_id(0)`.
    - Calculate the total number of elements in the destination image.
    - Check if the current index is out of bounds for the destination image and return if so.
    - Compute the indices for each dimension of the destination image based on the current index.
    - Calculate the corresponding source indices by dividing the destination indices by the scaling factors.
    - Compute the offset to the source element using the source indices and byte strides.
    - Retrieve the source element value and assign it to the corresponding position in the destination image.
- **Output**: The function does not return a value; it writes the upscaled image data to the destination pointer.


---
### kernel\_upscale\_bilinear
The `kernel_upscale_bilinear` function performs bilinear interpolation to upscale a source image to a destination image using specified scaling factors.
- **Inputs**:
    - `p_src0`: A pointer to the source image data.
    - `off_src0`: An offset in bytes to the start of the source image data.
    - `p_dst`: A pointer to the destination image data.
    - `off_dst`: An offset in bytes to the start of the destination image data.
    - `nb00`: The byte stride between elements in the first dimension of the source image.
    - `nb01`: The byte stride between elements in the second dimension of the source image.
    - `nb02`: The byte stride between elements in the third dimension of the source image.
    - `nb03`: The byte stride between elements in the fourth dimension of the source image.
    - `ne00_src`: The number of elements in the first dimension of the source image.
    - `ne01_src`: The number of elements in the second dimension of the source image.
    - `ne10_dst`: The number of elements in the first dimension of the destination image.
    - `ne11_dst`: The number of elements in the second dimension of the destination image.
    - `ne12_dst`: The number of elements in the third dimension of the destination image.
    - `ne13_dst`: The number of elements in the fourth dimension of the destination image.
    - `sf0`: The scaling factor for the first dimension.
    - `sf1`: The scaling factor for the second dimension.
    - `sf2`: The scaling factor for the third dimension.
    - `sf3`: The scaling factor for the fourth dimension.
- **Control Flow**:
    - Calculate the global index of the current thread and the total number of elements in the destination image.
    - Check if the current index is out of bounds for the destination image; if so, return immediately.
    - Compute the destination indices for each dimension based on the current index.
    - Calculate the corresponding source indices for the third and fourth dimensions using the scaling factors.
    - Compute the floating-point source coordinates for the first and second dimensions, adjusting for pixel offset.
    - Determine the integer source indices for bilinear interpolation, ensuring they are within valid bounds.
    - Calculate the interpolation weights (dx and dy) based on the fractional part of the source coordinates.
    - Retrieve the four neighboring source pixel values needed for bilinear interpolation.
    - Perform bilinear interpolation using the four source pixel values and the interpolation weights.
    - Store the interpolated result in the destination image at the current index.
- **Output**: The function outputs the upscaled image data in the destination buffer, with each element computed using bilinear interpolation from the source image.


