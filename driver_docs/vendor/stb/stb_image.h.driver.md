# Purpose
The provided code is part of the `stb_image` library, a widely-used single-header library in C/C++ for loading images. It is designed to decode a variety of image formats, such as JPEG, PNG, BMP, PSD, TGA, GIF, HDR, PIC, and PNM, into a simple pixel array, making it particularly popular among game developers and software engineers who require a straightforward image loading solution. The library offers a simple API with functions like `stbi_load`, `stbi_load_from_memory`, and `stbi_load_from_callbacks`, and supports loading images with different bit depths and transformations, such as vertical flipping and iPhone PNG conversion. It is structured to be included in a single C or C++ file using the `#define STB_IMAGE_IMPLEMENTATION` directive, allowing for easy integration without external dependencies. The code also provides internal functions for handling specific image formats, robust error handling, and memory management, ensuring a flexible and efficient image loading process tailored to various use cases.
# Imports and Dependencies

---
- `stdio.h`
- `stdlib.h`
- `stdarg.h`
- `stddef.h`
- `string.h`
- `limits.h`
- `math.h`
- `assert.h`
- `stdint.h`
- `emmintrin.h`
- `intrin.h`
- `arm_neon.h`


# Data Structures

---
### stbi\_io\_callbacks
- **Type**: `struct`
- **Members**:
    - `read`: A function pointer that reads data into a buffer and returns the number of bytes read.
    - `skip`: A function pointer that skips a specified number of bytes in the data stream.
    - `eof`: A function pointer that checks if the end of the data stream has been reached.
- **Description**: The `stbi_io_callbacks` structure is designed to provide a customizable interface for reading data from various sources. It contains three function pointers: `read`, `skip`, and `eof`, which allow the user to define how data is read, how to skip bytes, and how to determine if the end of the data stream has been reached, respectively. This structure is particularly useful in scenarios where data is not read from a standard file, such as reading from memory buffers or network streams.


---
### stbi\_\_context
- **Type**: `struct`
- **Members**:
    - `img_x`: The width of the image in pixels.
    - `img_y`: The height of the image in pixels.
    - `img_n`: The number of components in the image.
    - `img_out_n`: The number of output components for the image.
    - `io`: A structure containing callback functions for I/O operations.
    - `io_user_data`: A pointer to user-defined data for I/O operations.
    - `read_from_callbacks`: A flag indicating if the image is read from callbacks.
    - `buflen`: The length of the buffer used for reading data.
    - `buffer_start`: A buffer to store the initial data read from the image.
    - `callback_already_read`: The number of bytes already read by the callback.
    - `img_buffer`: A pointer to the current position in the image buffer.
    - `img_buffer_end`: A pointer to the end of the image buffer.
    - `img_buffer_original`: A pointer to the start of the original image buffer.
    - `img_buffer_original_end`: A pointer to the end of the original image buffer.
- **Description**: The `stbi__context` structure is used to manage the state and data necessary for reading and processing image files in the STB image library. It contains information about the image dimensions, the number of components, and the I/O callbacks for reading data. The structure also includes buffers and pointers to manage the image data being processed, allowing for efficient reading and manipulation of image files.


---
### stbi\_\_result\_info
- **Type**: `struct`
- **Members**:
    - `bits_per_channel`: Specifies the number of bits used to represent each channel in the image.
    - `num_channels`: Indicates the number of color channels present in the image.
    - `channel_order`: Defines the order in which the channels are stored or processed.
- **Description**: The `stbi__result_info` structure is used to store metadata about an image's channel configuration, including the bit depth per channel, the number of channels, and the order of these channels. This information is crucial for correctly interpreting and processing image data, especially when dealing with different image formats that may have varying channel arrangements and bit depths.


---
### stbi\_\_huffman
- **Type**: `struct`
- **Members**:
    - `fast`: An array used for fast lookup of Huffman codes with a size determined by FAST_BITS.
    - `code`: An array storing the Huffman codes for each symbol.
    - `values`: An array holding the values corresponding to each Huffman code.
    - `size`: An array indicating the size of each Huffman code.
    - `maxcode`: An array storing the maximum code for each bit length.
    - `delta`: An array used to calculate the difference between the first symbol and the first code for each bit length.
- **Description**: The `stbi__huffman` structure is designed to represent a Huffman coding table, which is used in data compression algorithms to efficiently encode and decode data. It contains arrays for fast lookup, code storage, value mapping, and size tracking of Huffman codes, as well as arrays for managing maximum codes and calculating differences in symbol and code indices. This structure is optimized for performance, as indicated by the use of a fast lookup table and the specific arrangement of its members.


---
### stbi\_\_jpeg
- **Type**: `struct`
- **Members**:
    - `s`: Pointer to the stbi__context structure, which holds the context for image decoding.
    - `huff_dc`: Array of Huffman tables for DC coefficients.
    - `huff_ac`: Array of Huffman tables for AC coefficients.
    - `dequant`: 2D array for dequantization tables, one for each component.
    - `fast_ac`: Array for fast AC coefficient lookup.
    - `img_h_max`: Maximum horizontal sampling factor.
    - `img_v_max`: Maximum vertical sampling factor.
    - `img_mcu_x`: Width of the image in MCU (Minimum Coded Unit) blocks.
    - `img_mcu_y`: Height of the image in MCU blocks.
    - `img_mcu_w`: Width of the MCU block.
    - `img_mcu_h`: Height of the MCU block.
    - `img_comp`: Array of structures defining each JPEG image component.
    - `code_buffer`: Buffer for entropy-coded data.
    - `code_bits`: Number of valid bits in the code buffer.
    - `marker`: JPEG marker seen while filling the entropy buffer.
    - `nomore`: Flag indicating if a marker was seen, requiring decoding to stop.
    - `progressive`: Flag indicating if the JPEG is progressive.
    - `spec_start`: Start of spectral selection.
    - `spec_end`: End of spectral selection.
    - `succ_high`: Successive approximation high bit position.
    - `succ_low`: Successive approximation low bit position.
    - `eob_run`: End-of-block run length for progressive JPEGs.
    - `jfif`: Flag indicating if the JPEG is in JFIF format.
    - `app14_color_transform`: Adobe APP14 color transform tag.
    - `rgb`: Flag indicating if the image is in RGB format.
    - `scan_n`: Number of components in the current scan.
    - `order`: Order of components in the scan.
    - `restart_interval`: Interval for JPEG restart markers.
    - `todo`: Number of blocks to process in the current scan.
    - `idct_block_kernel`: Function pointer for the IDCT block processing kernel.
    - `YCbCr_to_RGB_kernel`: Function pointer for converting YCbCr to RGB.
    - `resample_row_hv_2_kernel`: Function pointer for resampling rows with horizontal and vertical scaling.
- **Description**: The `stbi__jpeg` structure is a comprehensive data structure used in the decoding of JPEG images. It contains various fields and arrays to manage the decoding process, including Huffman tables for DC and AC coefficients, dequantization tables, and component definitions for the JPEG image. The structure also includes fields for handling progressive JPEGs, such as spectral selection and successive approximation, as well as function pointers for key operations like inverse discrete cosine transform (IDCT) and color space conversion. Additionally, it manages the entropy-coded data buffer and tracks the state of the decoding process, including markers and restart intervals.


---
### stbi\_\_resample
- **Type**: `struct`
- **Members**:
    - `resample`: A function pointer to a resampling function.
    - `line0`: A pointer to the first line of image data.
    - `line1`: A pointer to the second line of image data.
    - `hs`: The horizontal expansion factor.
    - `vs`: The vertical expansion factor.
    - `w_lores`: The number of horizontal pixels before expansion.
    - `ystep`: Tracks the progress through vertical expansion.
    - `ypos`: Indicates the current pre-expansion row.
- **Description**: The `stbi__resample` structure is used in image processing to manage the resampling of image data. It contains a function pointer for the resampling operation, pointers to two lines of image data, and several integer fields that track the expansion factors and the current position within the image data. This structure is essential for handling the transformation of image data from a lower resolution to a higher resolution by expanding the image along both horizontal and vertical axes.


---
### stbi\_\_zhuffman
- **Type**: `struct`
- **Members**:
    - `fast`: An array used for fast lookup of Huffman codes, indexed by the first few bits of the code.
    - `firstcode`: An array storing the first code of each bit length.
    - `maxcode`: An array storing the maximum code for each bit length.
    - `firstsymbol`: An array storing the first symbol of each bit length.
    - `size`: An array storing the size of each symbol in bits.
    - `value`: An array storing the value of each symbol.
- **Description**: The `stbi__zhuffman` structure is used to represent a Huffman coding table, which is essential for decoding compressed data in formats like PNG or JPEG. It contains arrays for fast lookup of Huffman codes, as well as arrays to store the first code, maximum code, first symbol, size, and value of each symbol, facilitating efficient decoding of variable-length codes.


---
### stbi\_\_zbuf
- **Type**: `struct`
- **Members**:
    - `zbuffer`: Pointer to the start of the input buffer.
    - `zbuffer_end`: Pointer to the end of the input buffer.
    - `num_bits`: Number of bits currently in the code buffer.
    - `hit_zeof_once`: Flag indicating if the end of the input buffer was reached once.
    - `code_buffer`: Buffer holding bits of the current code being processed.
    - `zout`: Pointer to the current position in the output buffer.
    - `zout_start`: Pointer to the start of the output buffer.
    - `zout_end`: Pointer to the end of the output buffer.
    - `z_expandable`: Flag indicating if the output buffer can be expanded.
    - `z_length`: Huffman coding structure for lengths.
    - `z_distance`: Huffman coding structure for distances.
- **Description**: The `stbi__zbuf` structure is used in the context of decompression, specifically for handling zlib-compressed data. It manages both input and output buffers, tracks the state of the decompression process, and uses Huffman coding structures to decode lengths and distances. The structure includes pointers to the input and output buffers, a code buffer for bit manipulation, and flags to handle buffer expansion and end-of-file conditions.


---
### stbi\_\_pngchunk
- **Type**: `struct`
- **Members**:
    - `length`: Represents the length of the PNG chunk data.
    - `type`: Indicates the type of the PNG chunk.
- **Description**: The `stbi__pngchunk` structure is used to represent a chunk in a PNG file, which is a part of the file's data stream. Each chunk in a PNG file has a specific format, consisting of a length field and a type field, which this structure captures. The `length` field specifies the size of the data contained in the chunk, while the `type` field identifies the kind of data or operation the chunk represents. This structure is essential for parsing and processing PNG files, as it allows for the identification and handling of different types of chunks within the file.


---
### stbi\_\_png
- **Type**: `struct`
- **Members**:
    - `s`: A pointer to an stbi__context structure, which holds the context for image loading.
    - `idata`: A pointer to an unsigned char array that stores the image data.
    - `expanded`: A pointer to an unsigned char array that holds the expanded image data after decompression.
    - `out`: A pointer to an unsigned char array that contains the final output image data.
    - `depth`: An integer representing the bit depth of the image.
- **Description**: The `stbi__png` structure is used in the context of loading and processing PNG images. It contains pointers to various stages of image data, including the raw image data (`idata`), the expanded data after decompression (`expanded`), and the final output data (`out`). The `s` member is a pointer to an `stbi__context` structure, which provides the necessary context for image loading operations. The `depth` member indicates the bit depth of the image, which is crucial for correctly interpreting the image data.


---
### stbi\_\_bmp\_data
- **Type**: `struct`
- **Members**:
    - `bpp`: Represents the bits per pixel in the BMP image.
    - `offset`: Indicates the offset where the BMP image data starts.
    - `hsz`: Specifies the size of the BMP header.
    - `mr`: Mask for the red channel in the BMP image.
    - `mg`: Mask for the green channel in the BMP image.
    - `mb`: Mask for the blue channel in the BMP image.
    - `ma`: Mask for the alpha channel in the BMP image.
    - `all_a`: Indicates if all alpha values are set in the BMP image.
    - `extra_read`: Tracks additional bytes read beyond the BMP header.
- **Description**: The `stbi__bmp_data` structure is used to store metadata and configuration information for BMP image processing. It includes fields for bits per pixel, data offset, header size, and color channel masks, which are essential for interpreting the BMP image data correctly. The structure also includes a field to track any extra bytes read, which can be useful for handling non-standard BMP files.


---
### stbi\_\_pic\_packet
- **Type**: `struct`
- **Members**:
    - `size`: Represents the size of the packet in terms of data units.
    - `type`: Indicates the type of the packet, possibly defining its role or format.
    - `channel`: Specifies the channel information, which could relate to color or data channels.
- **Description**: The `stbi__pic_packet` structure is a compact data structure used to represent a packet of image data, typically in the context of image processing or manipulation. It contains three fields: `size`, `type`, and `channel`, each represented by an unsigned character (`stbi_uc`). These fields collectively describe the packet's size, its type, and the channel information, which are essential for handling image data efficiently in various operations.


---
### stbi\_\_gif\_lzw
- **Type**: `struct`
- **Members**:
    - `prefix`: A 16-bit integer used to store the prefix code in the LZW decompression algorithm.
    - `first`: An unsigned 8-bit character representing the first character of the string in the LZW table.
    - `suffix`: An unsigned 8-bit character representing the last character of the string in the LZW table.
- **Description**: The `stbi__gif_lzw` structure is used in the LZW (Lempel-Ziv-Welch) decompression algorithm, which is commonly employed in GIF image decoding. It holds information about a code in the LZW table, including its prefix code, the first character of the string it represents, and the suffix character. This structure is essential for reconstructing the original data from compressed GIF files.


---
### stbi\_\_gif
- **Type**: `struct`
- **Members**:
    - `w`: Width of the GIF image.
    - `h`: Height of the GIF image.
    - `out`: Pointer to the output buffer, always containing 4 components.
    - `background`: Pointer to the current background as interpreted by the GIF.
    - `history`: Pointer to the history of the GIF frames.
    - `flags`: Flags related to the GIF image.
    - `bgindex`: Index of the background color in the color table.
    - `ratio`: Aspect ratio of the GIF image.
    - `transparent`: Index of the transparent color in the color table.
    - `eflags`: Extended flags for additional GIF properties.
    - `pal`: Global color palette with 256 colors, each having 4 components.
    - `lpal`: Local color palette with 256 colors, each having 4 components.
    - `codes`: Array of LZW codes used for GIF decompression.
    - `color_table`: Pointer to the current color table being used.
    - `parse`: State of the parsing process.
    - `step`: Current step in the GIF processing.
    - `lflags`: Local flags for the current frame.
    - `start_x`: Starting x-coordinate for the current frame.
    - `start_y`: Starting y-coordinate for the current frame.
    - `max_x`: Maximum x-coordinate for the current frame.
    - `max_y`: Maximum y-coordinate for the current frame.
    - `cur_x`: Current x-coordinate being processed.
    - `cur_y`: Current y-coordinate being processed.
    - `line_size`: Size of a line in the output buffer.
    - `delay`: Delay time between frames in the GIF.
- **Description**: The `stbi__gif` structure is used to represent and manage the state of a GIF image during decoding. It contains various fields to store the dimensions of the image, pointers to buffers for output, background, and history, as well as color palettes for both global and local contexts. The structure also includes fields for managing the LZW compression codes, parsing state, and frame-specific information such as starting coordinates, maximum dimensions, and delay between frames. This comprehensive structure is essential for handling the complexities of GIF image decoding, ensuring that each frame is processed correctly and efficiently.


# Functions

---
### stbi\_\_cpuid3<!-- {{#callable:stbi__cpuid3}} -->
Retrieves the CPU features by executing the `cpuid` instruction and returns the value of the `edx` register.
- **Inputs**: None
- **Control Flow**:
    - The function begins by declaring an integer variable `res` to store the result.
    - It uses inline assembly to set the `eax` register to 1, which is a request for CPU feature information.
    - The `cpuid` instruction is executed, which populates the `eax`, `ebx`, `ecx`, and `edx` registers with information about the CPU.
    - The value in the `edx` register, which contains specific feature flags, is moved into the `res` variable.
    - Finally, the function returns the value stored in `res`.
- **Output**: The function returns an integer representing the contents of the `edx` register after executing the `cpuid` instruction, which indicates the supported CPU features.


---
### stbi\_\_sse2\_available<!-- {{#callable:stbi__sse2_available}} -->
Checks if SSE2 instructions are available for use.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value of 1, indicating that SSE2 instructions are available.
- **Output**: Returns an integer value of 1, signifying that SSE2 instructions can be utilized.


---
### stbi\_\_start\_mem<!-- {{#callable:stbi__start_mem}} -->
Initializes the `stbi__context` structure for reading image data from a memory buffer.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that will be initialized.
    - `buffer`: A pointer to the memory buffer containing the image data.
    - `len`: An integer representing the length of the memory buffer.
- **Control Flow**:
    - Sets the `read` function pointer in the `stbi__context` to NULL, indicating no custom read function is used.
    - Initializes the `read_from_callbacks` flag to 0, indicating that reading is not done from callbacks.
    - Sets the `callback_already_read` flag to 0, indicating that no data has been read from callbacks yet.
    - Assigns the `img_buffer` and `img_buffer_original` pointers to the provided `buffer`, allowing access to the image data.
    - Calculates and sets the `img_buffer_end` and `img_buffer_original_end` pointers to the end of the buffer based on the provided length.
- **Output**: This function does not return a value; it modifies the `stbi__context` structure to prepare it for reading image data from the specified memory buffer.


---
### stbi\_\_start\_callbacks<!-- {{#callable:stbi__start_callbacks}} -->
Initializes the `stbi__context` structure with provided I/O callbacks and user data.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that will be initialized.
    - `c`: A pointer to the `stbi_io_callbacks` structure containing the I/O callback functions.
    - `user`: A pointer to user-defined data that will be associated with the callbacks.
- **Control Flow**:
    - Assigns the provided I/O callbacks to the `io` field of the `stbi__context` structure.
    - Stores the user data pointer in the `io_user_data` field of the `stbi__context` structure.
    - Sets the buffer length to the size of the `buffer_start` array.
    - Indicates that reading will be performed from callbacks by setting `read_from_callbacks` to 1.
    - Resets the `callback_already_read` flag to 0.
    - Initializes the image buffer pointers to point to the start of the buffer.
    - Calls [`stbi__refill_buffer`](#stbi__refill_buffer) to fill the buffer with data from the callbacks.
    - Sets the end pointer of the original image buffer to the end of the filled buffer.
- **Output**: The function does not return a value; it modifies the state of the `stbi__context` structure directly.
- **Functions called**:
    - [`stbi__refill_buffer`](#stbi__refill_buffer)


---
### stbi\_\_stdio\_read<!-- {{#callable:stbi__stdio_read}} -->
Reads a specified number of bytes from a file stream into a buffer.
- **Inputs**:
    - `user`: A pointer to a `FILE` stream from which data will be read.
    - `data`: A pointer to a buffer where the read data will be stored.
    - `size`: The number of bytes to read from the file stream.
- **Control Flow**:
    - The function calls `fread` to read `size` bytes from the file stream pointed to by `user`.
    - The data read is stored in the buffer pointed to by `data`.
    - The return value of `fread`, which indicates the number of items successfully read, is cast to an `int` and returned.
- **Output**: Returns the number of bytes read from the file stream, or 0 if no bytes were read.


---
### stbi\_\_stdio\_skip<!-- {{#callable:stbi__stdio_skip}} -->
The `stbi__stdio_skip` function skips a specified number of bytes in a file stream.
- **Inputs**:
    - `user`: A pointer to a `FILE` object representing the file stream to be manipulated.
    - `n`: An integer representing the number of bytes to skip in the file stream.
- **Control Flow**:
    - The function uses `fseek` to move the file position indicator forward by `n` bytes from the current position.
    - It reads the next byte from the file using `fgetc` to ensure that the end-of-file (EOF) flag is reset.
    - If the byte read is not EOF, it uses `ungetc` to push the byte back onto the stream, allowing it to be read again later.
- **Output**: The function does not return a value; it modifies the file stream's position and state.


---
### stbi\_\_stdio\_eof<!-- {{#callable:stbi__stdio_eof}} -->
Checks if the end-of-file or an error has occurred on a given file stream.
- **Inputs**:
    - `user`: A pointer to a `FILE` object representing the file stream to check.
- **Control Flow**:
    - The function casts the `user` pointer to a `FILE*` type.
    - It calls the `feof` function to check if the end-of-file indicator is set for the file stream.
    - It calls the `ferror` function to check if an error has occurred on the file stream.
    - The function returns the logical OR of the results from `feof` and `ferror`.
- **Output**: Returns a non-zero value if either the end-of-file has been reached or an error has occurred; otherwise, it returns zero.


---
### stbi\_\_start\_file<!-- {{#callable:stbi__start_file}} -->
Initializes a file context for the STB image library using standard I/O callbacks.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure that holds the state of the image decoding process.
    - `f`: A pointer to a `FILE` object representing the file to be processed.
- **Control Flow**:
    - Calls the [`stbi__start_callbacks`](#stbi__start_callbacks) function to set up the context for reading from the specified file.
    - Passes the `stbi__stdio_callbacks` structure to handle standard I/O operations.
- **Output**: This function does not return a value; it modifies the state of the `stbi__context` to prepare it for reading image data from the specified file.
- **Functions called**:
    - [`stbi__start_callbacks`](#stbi__start_callbacks)


---
### stbi\_\_rewind<!-- {{#callable:stbi__rewind}} -->
Rewinds the image buffer to its original position in the `stbi__context` structure.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the image buffer and its original position.
- **Control Flow**:
    - Sets the `img_buffer` of the context `s` to the original image buffer position stored in `img_buffer_original`.
    - Sets the `img_buffer_end` of the context `s` to the original end position stored in `img_buffer_original_end`.
- **Output**: This function does not return a value; it modifies the state of the `stbi__context` by resetting the image buffer pointers.


---
### stbi\_failure\_reason<!-- {{#callable:stbi_failure_reason}} -->
Returns the reason for the last failure encountered by the STB image library.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the global variable `stbi__g_failure_reason`.
- **Output**: A pointer to a string that describes the reason for the last failure, or NULL if no failure has occurred.


---
### stbi\_\_err<!-- {{#callable:stbi__err}} -->
Sets the global failure reason string and returns zero.
- **Inputs**:
    - `str`: A pointer to a string that describes the failure reason.
- **Control Flow**:
    - Assigns the input string `str` to the global variable `stbi__g_failure_reason`.
    - Returns the integer value 0.
- **Output**: Always returns 0, indicating a standard return value for error handling.


---
### stbi\_\_malloc<!-- {{#callable:stbi__malloc}} -->
Allocates memory of the specified size using the `STBI_MALLOC` macro.
- **Inputs**:
    - `size`: The number of bytes to allocate in memory.
- **Control Flow**:
    - The function takes a single input parameter `size` which specifies the amount of memory to allocate.
    - It calls the `STBI_MALLOC` macro with the `size` parameter to perform the memory allocation.
    - The result of the `STBI_MALLOC` call is returned directly.
- **Output**: Returns a pointer to the allocated memory block, or NULL if the allocation fails.


---
### stbi\_\_addsizes\_valid<!-- {{#callable:stbi__addsizes_valid}} -->
Checks if the sum of two integers `a` and `b` is valid without causing an overflow.
- **Inputs**:
    - `a`: An integer value that is the first operand in the addition.
    - `b`: An integer value that is the second operand in the addition, which must be non-negative to avoid overflow.
- **Control Flow**:
    - First, the function checks if `b` is negative; if it is, the function immediately returns 0, indicating an invalid sum.
    - Next, it verifies that `a` is less than or equal to `INT_MAX - b`, which ensures that adding `a` and `b` will not exceed the maximum integer value.
- **Output**: Returns 1 if the sum of `a` and `b` is valid (i.e., does not overflow), otherwise returns 0.


---
### stbi\_\_mul2sizes\_valid<!-- {{#callable:stbi__mul2sizes_valid}} -->
Checks if multiplying two integers will not cause an overflow.
- **Inputs**:
    - `a`: The first integer to be multiplied.
    - `b`: The second integer to be multiplied.
- **Control Flow**:
    - The function first checks if either `a` or `b` is negative; if so, it returns 0, indicating invalid sizes.
    - Next, it checks if `b` is zero; if true, it returns 1, as multiplying by zero is always safe.
    - Finally, it checks if `a` is less than or equal to `INT_MAX / b` to determine if the multiplication will overflow, returning the result of this check.
- **Output**: Returns 0 if the multiplication is invalid (due to negative values), 1 if safe (multiplying by zero), or 1 if multiplication is safe without overflow.


---
### stbi\_\_mad2sizes\_valid<!-- {{#callable:stbi__mad2sizes_valid}} -->
Validates the sizes of two dimensions and an additional size using multiplication and addition checks.
- **Inputs**:
    - `a`: The first dimension size to validate.
    - `b`: The second dimension size to validate.
    - `add`: An additional size to validate after multiplication.
- **Control Flow**:
    - Calls [`stbi__mul2sizes_valid`](#stbi__mul2sizes_valid) with inputs `a` and `b` to check if the multiplication of the two sizes is valid.
    - Calls [`stbi__addsizes_valid`](#stbi__addsizes_valid) with the product of `a` and `b` and the `add` parameter to check if the addition is valid.
    - Returns the logical AND of the two validation results.
- **Output**: Returns 1 (true) if both size validations are successful, otherwise returns 0 (false).
- **Functions called**:
    - [`stbi__mul2sizes_valid`](#stbi__mul2sizes_valid)
    - [`stbi__addsizes_valid`](#stbi__addsizes_valid)


---
### stbi\_\_mad3sizes\_valid<!-- {{#callable:stbi__mad3sizes_valid}} -->
The `stbi__mad3sizes_valid` function checks the validity of size calculations involving three dimensions and an additional size.
- **Inputs**:
    - `a`: The first dimension size.
    - `b`: The second dimension size.
    - `c`: The third dimension size.
    - `add`: An additional size to be added to the product of the three dimensions.
- **Control Flow**:
    - The function calls [`stbi__mul2sizes_valid`](#stbi__mul2sizes_valid) with `a` and `b` to validate the multiplication of the first two dimensions.
    - It then calls [`stbi__mul2sizes_valid`](#stbi__mul2sizes_valid) again with the product of `a` and `b`, and `c` to validate the multiplication of the result with the third dimension.
    - Finally, it calls [`stbi__addsizes_valid`](#stbi__addsizes_valid) with the product of `a`, `b`, and `c`, and `add` to validate the addition of the additional size.
- **Output**: The function returns a boolean value (1 for true, 0 for false) indicating whether all size calculations are valid.
- **Functions called**:
    - [`stbi__mul2sizes_valid`](#stbi__mul2sizes_valid)
    - [`stbi__addsizes_valid`](#stbi__addsizes_valid)


---
### stbi\_\_mad4sizes\_valid<!-- {{#callable:stbi__mad4sizes_valid}} -->
The `stbi__mad4sizes_valid` function checks the validity of size calculations involving four dimensions and an additional value.
- **Inputs**:
    - `a`: The first dimension size.
    - `b`: The second dimension size.
    - `c`: The third dimension size.
    - `d`: The fourth dimension size.
    - `add`: An additional value to be added to the product of the dimensions.
- **Control Flow**:
    - The function calls [`stbi__mul2sizes_valid`](#stbi__mul2sizes_valid) to validate the multiplication of the first two dimensions, `a` and `b`.
    - Then it checks the validity of the product of `a` and `b` with the third dimension `c`.
    - Next, it validates the product of `a`, `b`, and `c` with the fourth dimension `d`.
    - Finally, it checks if the product of all four dimensions can be added to `add` without overflow using [`stbi__addsizes_valid`](#stbi__addsizes_valid).
    - The function returns true (non-zero) if all checks pass, otherwise it returns false (zero).
- **Output**: The output is an integer value, where a non-zero value indicates that all size calculations are valid, and zero indicates at least one calculation is invalid.
- **Functions called**:
    - [`stbi__mul2sizes_valid`](#stbi__mul2sizes_valid)
    - [`stbi__addsizes_valid`](#stbi__addsizes_valid)


---
### stbi\_\_malloc\_mad2<!-- {{#callable:stbi__malloc_mad2}} -->
Allocates memory based on the product of two integers and an additional size if the sizes are valid.
- **Inputs**:
    - `a`: An integer representing one dimension of the size to allocate.
    - `b`: An integer representing the other dimension of the size to allocate.
    - `add`: An integer representing additional memory to allocate beyond the product of a and b.
- **Control Flow**:
    - The function first checks if the sizes calculated from a, b, and add are valid using the [`stbi__mad2sizes_valid`](#stbi__mad2sizes_valid) function.
    - If the sizes are not valid, the function returns NULL, indicating failure to allocate memory.
    - If the sizes are valid, it proceeds to allocate memory using the [`stbi__malloc`](#stbi__malloc) function, calculating the total size as the product of a and b plus add.
- **Output**: Returns a pointer to the allocated memory if successful, or NULL if the sizes are invalid.
- **Functions called**:
    - [`stbi__mad2sizes_valid`](#stbi__mad2sizes_valid)
    - [`stbi__malloc`](#stbi__malloc)


---
### stbi\_\_malloc\_mad3<!-- {{#callable:stbi__malloc_mad3}} -->
Allocates memory based on the product of three dimensions and an additional size if the dimensions are valid.
- **Inputs**:
    - `a`: The first dimension size used for memory allocation.
    - `b`: The second dimension size used for memory allocation.
    - `c`: The third dimension size used for memory allocation.
    - `add`: An additional size to be added to the total memory allocation.
- **Control Flow**:
    - Checks if the dimensions a, b, c, and add are valid using the [`stbi__mad3sizes_valid`](#stbi__mad3sizes_valid) function.
    - If the dimensions are not valid, the function returns NULL.
    - If the dimensions are valid, it calculates the total size required for allocation as a*b*c + add.
    - Calls [`stbi__malloc`](#stbi__malloc) to allocate the calculated memory size.
- **Output**: Returns a pointer to the allocated memory if the dimensions are valid; otherwise, returns NULL.
- **Functions called**:
    - [`stbi__mad3sizes_valid`](#stbi__mad3sizes_valid)
    - [`stbi__malloc`](#stbi__malloc)


---
### stbi\_\_malloc\_mad4<!-- {{#callable:stbi__malloc_mad4}} -->
Allocates memory for a multi-dimensional array based on the specified dimensions and an additional size.
- **Inputs**:
    - `a`: The size of the first dimension of the array.
    - `b`: The size of the second dimension of the array.
    - `c`: The size of the third dimension of the array.
    - `d`: The size of the fourth dimension of the array.
    - `add`: An additional size to be added to the total memory allocation.
- **Control Flow**:
    - Checks if the dimensions and additional size are valid using the [`stbi__mad4sizes_valid`](#stbi__mad4sizes_valid) function.
    - If the dimensions are invalid, returns NULL to indicate failure.
    - If valid, calculates the total size required for the allocation and calls [`stbi__malloc`](#stbi__malloc) to allocate the memory.
- **Output**: Returns a pointer to the allocated memory if successful, or NULL if the dimensions are invalid.
- **Functions called**:
    - [`stbi__mad4sizes_valid`](#stbi__mad4sizes_valid)
    - [`stbi__malloc`](#stbi__malloc)


---
### stbi\_\_addints\_valid<!-- {{#callable:stbi__addints_valid}} -->
The `stbi__addints_valid` function checks if the addition of two integers `a` and `b` can be performed without causing an overflow.
- **Inputs**:
    - `a`: An integer value to be added.
    - `b`: Another integer value to be added.
- **Control Flow**:
    - The function first checks if `a` and `b` have different signs; if they do, it returns 1 indicating no overflow is possible.
    - If both `a` and `b` are negative, it checks if `a` is greater than or equal to `INT_MIN - b` to ensure the sum does not underflow.
    - If both `a` and `b` are non-negative, it checks if `a` is less than or equal to `INT_MAX - b` to ensure the sum does not overflow.
- **Output**: Returns 1 if the addition of `a` and `b` is valid (no overflow), otherwise returns 0.


---
### stbi\_\_mul2shorts\_valid<!-- {{#callable:stbi__mul2shorts_valid}} -->
Checks if the multiplication of two integers `a` and `b` is valid without causing overflow for short integers.
- **Inputs**:
    - `a`: An integer value that is one of the operands in the multiplication.
    - `b`: An integer value that is the other operand in the multiplication.
- **Control Flow**:
    - The function first checks if `b` is 0 or -1, returning 1 if true, as multiplication by 0 results in 0 and -1 is checked to prevent overflow.
    - If both `a` and `b` have the same sign (both positive or both negative), it checks if `a` is less than or equal to `SHRT_MAX / b` to ensure the product does not exceed the maximum short integer value.
    - If `b` is negative, it checks if `a` is less than or equal to `SHRT_MIN / b` to ensure the product does not fall below the minimum short integer value.
    - If none of the above conditions are met, it checks if `a` is greater than or equal to `SHRT_MIN / b` to validate the multiplication.
- **Output**: Returns 1 if the multiplication of `a` and `b` is valid without causing overflow for short integers, otherwise returns 0.


---
### stbi\_image\_free<!-- {{#callable:stbi_image_free}} -->
Frees the memory allocated for an image loaded by `stbi_load`.
- **Inputs**:
    - `retval_from_stbi_load`: A pointer to the memory block that was allocated for the image data.
- **Control Flow**:
    - The function calls `STBI_FREE` to deallocate the memory pointed to by `retval_from_stbi_load`.
- **Output**: This function does not return a value; it simply frees the allocated memory.


---
### stbi\_set\_flip\_vertically\_on\_load<!-- {{#callable:stbi_set_flip_vertically_on_load}} -->
Sets a global flag to determine whether images should be flipped vertically upon loading.
- **Inputs**:
    - `flag_true_if_should_flip`: An integer flag where a non-zero value indicates that images should be flipped vertically on load, and zero indicates no flipping.
- **Control Flow**:
    - The function assigns the value of `flag_true_if_should_flip` to the global variable `stbi__vertically_flip_on_load_global`.
    - No conditional logic or loops are present; the function performs a direct assignment.
- **Output**: The function does not return a value; it modifies a global state that affects subsequent image loading behavior.


---
### stbi\_set\_flip\_vertically\_on\_load\_thread<!-- {{#callable:stbi_set_flip_vertically_on_load_thread}} -->
Sets a flag to determine whether images should be flipped vertically when loaded in a thread.
- **Inputs**:
    - `flag_true_if_should_flip`: An integer flag where a non-zero value indicates that images should be flipped vertically upon loading.
- **Control Flow**:
    - The function assigns the value of `flag_true_if_should_flip` to the global variable `stbi__vertically_flip_on_load_local`.
    - It then sets the global variable `stbi__vertically_flip_on_load_set` to 1, indicating that the flip setting has been configured.
- **Output**: The function does not return a value; it modifies global state variables to control the vertical flipping behavior of image loading.


---
### stbi\_\_load\_main<!-- {{#callable:stbi__load_main}} -->
The `stbi__load_main` function loads an image from a given context and returns a pointer to the image data based on its format.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the image data to be loaded.
    - `x`: A pointer to an integer where the width of the loaded image will be stored.
    - `y`: A pointer to an integer where the height of the loaded image will be stored.
    - `comp`: A pointer to an integer where the number of components in the loaded image will be stored.
    - `req_comp`: An integer specifying the number of components requested in the output image.
    - `ri`: A pointer to a `stbi__result_info` structure that will be filled with information about the loaded image.
    - `bpc`: An integer representing bits per channel, used for certain image formats.
- **Control Flow**:
    - The function initializes the `ri` structure to ensure it has default values.
    - It checks the image format by testing the header of the input data using various format-specific test functions.
    - If a matching format is found, it calls the corresponding load function for that format, passing the necessary parameters.
    - If no known format is detected, it returns an error indicating that the image type is unknown or corrupt.
- **Output**: The function returns a pointer to the loaded image data, or an error message if the image type is not recognized.
- **Functions called**:
    - [`stbi__png_test`](#stbi__png_test)
    - [`stbi__png_load`](#stbi__png_load)
    - [`stbi__bmp_test`](#stbi__bmp_test)
    - [`stbi__bmp_load`](#stbi__bmp_load)
    - [`stbi__gif_test`](#stbi__gif_test)
    - [`stbi__gif_load`](#stbi__gif_load)
    - [`stbi__psd_test`](#stbi__psd_test)
    - [`stbi__psd_load`](#stbi__psd_load)
    - [`stbi__pic_test`](#stbi__pic_test)
    - [`stbi__pic_load`](#stbi__pic_load)
    - [`stbi__jpeg_test`](#stbi__jpeg_test)
    - [`stbi__jpeg_load`](#stbi__jpeg_load)
    - [`stbi__pnm_test`](#stbi__pnm_test)
    - [`stbi__pnm_load`](#stbi__pnm_load)
    - [`stbi__hdr_test`](#stbi__hdr_test)
    - [`stbi__hdr_load`](#stbi__hdr_load)
    - [`stbi__hdr_to_ldr`](#stbi__hdr_to_ldr)
    - [`stbi__tga_test`](#stbi__tga_test)
    - [`stbi__tga_load`](#stbi__tga_load)


---
### stbi\_\_convert\_16\_to\_8<!-- {{#callable:stbi__convert_16_to_8}} -->
Converts a 16-bit per channel image to an 8-bit per channel format.
- **Inputs**:
    - `orig`: A pointer to an array of 16-bit unsigned integers representing the original image data.
    - `w`: An integer representing the width of the image.
    - `h`: An integer representing the height of the image.
    - `channels`: An integer representing the number of color channels in the image.
- **Control Flow**:
    - Calculates the total number of pixels in the image by multiplying width, height, and channels.
    - Allocates memory for the new 8-bit image data; if allocation fails, returns an error.
    - Iterates over each pixel in the original 16-bit image, converting each value to 8-bit by shifting and masking.
    - Frees the original 16-bit image data to prevent memory leaks.
    - Returns the pointer to the newly created 8-bit image data.
- **Output**: Returns a pointer to an array of 8-bit unsigned integers representing the converted image data.
- **Functions called**:
    - [`stbi__malloc`](#stbi__malloc)


---
### stbi\_\_convert\_8\_to\_16<!-- {{#callable:stbi__convert_8_to_16}} -->
Converts an 8-bit image to a 16-bit image by replicating each byte into two bytes.
- **Inputs**:
    - `orig`: A pointer to the original image data in 8-bit format.
    - `w`: The width of the image.
    - `h`: The height of the image.
    - `channels`: The number of color channels in the image (e.g., 1 for grayscale, 3 for RGB).
- **Control Flow**:
    - Calculates the total number of pixels in the image by multiplying width, height, and channels.
    - Allocates memory for the new 16-bit image data, which is twice the size of the original data.
    - Checks if memory allocation was successful; if not, returns an error.
    - Iterates over each pixel in the original image, replicating each 8-bit value into a 16-bit value by shifting and adding.
    - Frees the original 8-bit image data to prevent memory leaks.
    - Returns the pointer to the newly created 16-bit image data.
- **Output**: A pointer to the newly allocated 16-bit image data, or an error if memory allocation fails.
- **Functions called**:
    - [`stbi__malloc`](#stbi__malloc)


---
### stbi\_\_vertical\_flip<!-- {{#callable:stbi__vertical_flip}} -->
The `stbi__vertical_flip` function flips an image vertically by swapping its rows.
- **Inputs**:
    - `image`: A pointer to the image data that needs to be flipped.
    - `w`: The width of the image in pixels.
    - `h`: The height of the image in pixels.
    - `bytes_per_pixel`: The number of bytes used to represent each pixel.
- **Control Flow**:
    - Calculate the number of bytes per row based on the image width and bytes per pixel.
    - Iterate over the first half of the rows of the image.
    - For each row, calculate the pointers for the current row and its corresponding row from the bottom.
    - Swap the pixel data between the two rows using a temporary buffer to handle the data safely.
- **Output**: The function modifies the image data in place, resulting in a vertically flipped image.


---
### stbi\_\_vertical\_flip\_slices<!-- {{#callable:stbi__vertical_flip_slices}} -->
Flips the image slices vertically in place.
- **Inputs**:
    - `image`: A pointer to the image data that will be modified.
    - `w`: The width of each image slice.
    - `h`: The height of each image slice.
    - `z`: The number of slices in the image.
    - `bytes_per_pixel`: The number of bytes used to represent each pixel.
- **Control Flow**:
    - Calculates the size of each slice based on width, height, and bytes per pixel.
    - Iterates over each slice from 0 to z-1.
    - Calls the [`stbi__vertical_flip`](#stbi__vertical_flip) function for each slice to perform the vertical flip.
    - Advances the pointer to the next slice after each flip.
- **Output**: The function modifies the image data in place, resulting in the slices being flipped vertically.
- **Functions called**:
    - [`stbi__vertical_flip`](#stbi__vertical_flip)


---
### stbi\_\_load\_and\_postprocess\_8bit<!-- {{#callable:stbi__load_and_postprocess_8bit}} -->
Loads an image from a given context and processes it to ensure it is in 8-bit format.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the image data to be loaded.
    - `x`: A pointer to an integer where the width of the loaded image will be stored.
    - `y`: A pointer to an integer where the height of the loaded image will be stored.
    - `comp`: A pointer to an integer that indicates the number of color components in the loaded image.
    - `req_comp`: An integer that specifies the number of color components requested by the user.
- **Control Flow**:
    - Calls [`stbi__load_main`](#stbi__load_main) to load the image data and retrieve its information.
    - Checks if the result of the loading process is NULL, returning NULL if it is.
    - Asserts that the bits per channel of the loaded image is either 8 or 16.
    - If the bits per channel is 16, converts the image data to 8-bit format using [`stbi__convert_16_to_8`](#stbi__convert_16_to_8).
    - Checks if vertical flipping is enabled and applies it to the image data if necessary.
    - Returns the processed image data as an unsigned char pointer.
- **Output**: Returns a pointer to the loaded and processed image data in 8-bit format, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__load_main`](#stbi__load_main)
    - [`stbi__convert_16_to_8`](#stbi__convert_16_to_8)
    - [`stbi__vertical_flip`](#stbi__vertical_flip)


---
### stbi\_\_load\_and\_postprocess\_16bit<!-- {{#callable:stbi__load_and_postprocess_16bit}} -->
Loads an image in 16-bit format and processes it, including conversion from 8-bit if necessary.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that contains the image data.
    - `x`: A pointer to an integer where the width of the loaded image will be stored.
    - `y`: A pointer to an integer where the height of the loaded image will be stored.
    - `comp`: A pointer to an integer that indicates the number of components in the image (e.g., RGB, RGBA).
    - `req_comp`: An integer that specifies the requested number of components for the output image.
- **Control Flow**:
    - Calls [`stbi__load_main`](#stbi__load_main) to load the image data and store the result in `result`.
    - Checks if `result` is NULL, indicating a failure to load the image, and returns NULL if so.
    - Asserts that the bits per channel in `ri` are either 8 or 16.
    - If the bits per channel are not 16, converts the image from 8-bit to 16-bit using [`stbi__convert_8_to_16`](#stbi__convert_8_to_16).
    - Checks the `stbi__vertically_flip_on_load` flag and vertically flips the image if necessary using [`stbi__vertical_flip`](#stbi__vertical_flip).
    - Returns the processed image data as a pointer to `stbi__uint16`.
- **Output**: Returns a pointer to the loaded and processed image data in 16-bit format, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__load_main`](#stbi__load_main)
    - [`stbi__convert_8_to_16`](#stbi__convert_8_to_16)
    - [`stbi__vertical_flip`](#stbi__vertical_flip)


---
### stbi\_\_float\_postprocess<!-- {{#callable:stbi__float_postprocess}} -->
Processes the float image data by optionally flipping it vertically based on a global setting.
- **Inputs**:
    - `result`: A pointer to the float array that holds the image data to be processed.
    - `x`: A pointer to an integer that represents the width of the image.
    - `y`: A pointer to an integer that represents the height of the image.
    - `comp`: A pointer to an integer that indicates the number of color components in the image.
    - `req_comp`: An integer that specifies the requested number of components; if zero, the function uses the value pointed to by `comp`.
- **Control Flow**:
    - The function first checks if the global variable `stbi__vertically_flip_on_load` is true and if the `result` pointer is not NULL.
    - If both conditions are met, it determines the number of channels to use for flipping, either from `req_comp` or `*comp`.
    - It then calls the [`stbi__vertical_flip`](#stbi__vertical_flip) function to flip the image data vertically.
- **Output**: The function does not return a value; it modifies the image data in place if flipping is required.
- **Functions called**:
    - [`stbi__vertical_flip`](#stbi__vertical_flip)


---
### stbi\_convert\_wchar\_to\_utf8<!-- {{#callable:stbi_convert_wchar_to_utf8}} -->
Converts a wide character string to a UTF-8 encoded string.
- **Inputs**:
    - `buffer`: A pointer to a character array where the UTF-8 encoded string will be stored.
    - `bufferlen`: The size of the buffer in bytes, indicating the maximum number of characters that can be written.
    - `input`: A pointer to the wide character string (wchar_t) that needs to be converted to UTF-8.
- **Control Flow**:
    - The function calls `WideCharToMultiByte`, which is a Windows API function responsible for converting wide character strings to multi-byte character strings.
    - The function specifies the code page as 65001, which corresponds to UTF-8.
    - The `input` string is passed with a length of -1, indicating that it should be processed until a null terminator is encountered.
    - The result of the conversion is returned as an integer, which indicates the number of bytes written to the buffer.
- **Output**: Returns the number of bytes written to the `buffer` on success, or 0 if the conversion fails.


---
### stbi\_\_fopen<!-- {{#callable:stbi__fopen}} -->
The `stbi__fopen` function opens a file with the specified filename and mode, handling different character encodings based on the platform.
- **Inputs**:
    - `filename`: A constant character pointer representing the name of the file to be opened.
    - `mode`: A constant character pointer representing the mode in which the file should be opened (e.g., read, write).
- **Control Flow**:
    - The function checks if the platform is Windows and if UTF-8 support is enabled.
    - If so, it converts the `filename` and `mode` from UTF-8 to wide character format using `MultiByteToWideChar`.
    - It then attempts to open the file using `_wfopen_s` if the Microsoft compiler version is 1400 or higher, or `_wfopen` otherwise.
    - If the platform is not Windows or UTF-8 is not supported, it checks if the Microsoft compiler version is 1400 or higher to use `fopen_s`, or falls back to `fopen`.
    - Finally, it returns the file pointer or NULL if the file could not be opened.
- **Output**: The function returns a pointer to the opened `FILE` stream, or NULL if the file could not be opened.


---
### stbi\_load<!-- {{#callable:stbi_load}} -->
The `stbi_load` function loads an image from a specified file and returns a pointer to the image data.
- **Inputs**:
    - `filename`: A constant character pointer representing the path to the image file to be loaded.
    - `x`: A pointer to an integer where the width of the loaded image will be stored.
    - `y`: A pointer to an integer where the height of the loaded image will be stored.
    - `comp`: A pointer to an integer where the number of components in the loaded image will be stored.
    - `req_comp`: An integer specifying the desired number of components per pixel (e.g., 3 for RGB, 4 for RGBA).
- **Control Flow**:
    - The function attempts to open the specified file in binary read mode using [`stbi__fopen`](#stbi__fopen).
    - If the file cannot be opened, it returns an error message using `stbi__errpuc`.
    - If the file is successfully opened, it calls [`stbi_load_from_file`](#stbi_load_from_file) to load the image data, passing the file pointer and pointers for width, height, and component information.
    - After loading the image, the function closes the file using `fclose`.
    - Finally, it returns the pointer to the loaded image data.
- **Output**: Returns a pointer to the loaded image data as an array of unsigned characters, or an error message if the file could not be opened.
- **Functions called**:
    - [`stbi__fopen`](#stbi__fopen)
    - [`stbi_load_from_file`](#stbi_load_from_file)


---
### stbi\_load\_from\_file<!-- {{#callable:stbi_load_from_file}} -->
Loads an image from a file and returns a pointer to the image data.
- **Inputs**:
    - `FILE *f`: A pointer to a `FILE` object that represents the file from which the image will be loaded.
    - `int *x`: A pointer to an integer where the width of the loaded image will be stored.
    - `int *y`: A pointer to an integer where the height of the loaded image will be stored.
    - `int *comp`: A pointer to an integer where the number of color components in the loaded image will be stored.
    - `int req_comp`: An integer specifying the number of color components requested for the output image.
- **Control Flow**:
    - Initializes a `stbi__context` structure for reading the image data from the file.
    - Calls [`stbi__load_and_postprocess_8bit`](#stbi__load_and_postprocess_8bit) to load the image data and process it, storing the result in `result`.
    - If the image data is successfully loaded (i.e., `result` is not NULL), it seeks back in the file to reset the file pointer to the position before the image data was read.
- **Output**: Returns a pointer to the loaded image data in 8-bit format, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__start_file`](#stbi__start_file)
    - [`stbi__load_and_postprocess_8bit`](#stbi__load_and_postprocess_8bit)


---
### stbi\_load\_16<!-- {{#callable:stbi_load_16}} -->
`stbi_load_16` loads a 16-bit image from a specified file.
- **Inputs**:
    - `filename`: A string representing the path to the image file to be loaded.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored.
    - `req_comp`: An integer specifying the number of color components requested in the output.
- **Control Flow**:
    - Open the specified file in binary read mode using [`stbi__fopen`](#stbi__fopen).
    - Check if the file was successfully opened; if not, return an error message.
    - Call `stbi_load_from_file_16` to load the image data from the file, passing the file pointer and pointers for width, height, and components.
    - Close the file after loading the image data.
    - Return the loaded image data.
- **Output**: Returns a pointer to the loaded 16-bit image data, or an error message if the file could not be opened.
- **Functions called**:
    - [`stbi__fopen`](#stbi__fopen)


---
### stbi\_load\_16\_from\_memory<!-- {{#callable:stbi_load_16_from_memory}} -->
Loads a 16-bit image from a memory buffer and processes it.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing the image data.
    - `len`: The length of the memory buffer in bytes.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `channels_in_file`: A pointer to an integer where the number of channels in the image file will be stored.
    - `desired_channels`: The number of channels that the user wants in the output image.
- **Control Flow**:
    - Initializes a `stbi__context` structure to manage the image loading process.
    - Calls [`stbi__start_mem`](#stbi__start_mem) to set up the context with the provided buffer and its length.
    - Invokes [`stbi__load_and_postprocess_16bit`](#stbi__load_and_postprocess_16bit) to load the image data and perform any necessary post-processing.
- **Output**: Returns a pointer to the loaded 16-bit image data, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__start_mem`](#stbi__start_mem)
    - [`stbi__load_and_postprocess_16bit`](#stbi__load_and_postprocess_16bit)


---
### stbi\_load\_16\_from\_callbacks<!-- {{#callable:stbi_load_16_from_callbacks}} -->
Loads a 16-bit image from provided callbacks and returns a pointer to the image data.
- **Inputs**:
    - `clbk`: A pointer to a `stbi_io_callbacks` structure that contains callback functions for reading image data.
    - `user`: A user-defined pointer that is passed to the callback functions for context.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `channels_in_file`: A pointer to an integer where the number of channels in the image file will be stored.
    - `desired_channels`: An integer specifying the number of channels the user wants in the output image.
- **Control Flow**:
    - Initializes a `stbi__context` structure to manage the image loading process.
    - Calls [`stbi__start_callbacks`](#stbi__start_callbacks) to set up the context with the provided callbacks and user data.
    - Invokes [`stbi__load_and_postprocess_16bit`](#stbi__load_and_postprocess_16bit) to load the image data and process it according to the specified parameters.
- **Output**: Returns a pointer to the loaded 16-bit image data, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__start_callbacks`](#stbi__start_callbacks)
    - [`stbi__load_and_postprocess_16bit`](#stbi__load_and_postprocess_16bit)


---
### stbi\_load\_from\_memory<!-- {{#callable:stbi_load_from_memory}} -->
Loads an image from a memory buffer and returns a pointer to the image data.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing the image data.
    - `len`: The length of the memory buffer in bytes.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored.
    - `req_comp`: The number of color components requested (e.g., 3 for RGB, 4 for RGBA).
- **Control Flow**:
    - Initializes a `stbi__context` structure to manage the image loading process.
    - Calls [`stbi__start_mem`](#stbi__start_mem) to set up the context with the provided buffer and length.
    - Invokes [`stbi__load_and_postprocess_8bit`](#stbi__load_and_postprocess_8bit) to load the image data and process it, passing the context and pointers for width, height, and components.
- **Output**: Returns a pointer to the loaded image data in 8-bit format, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__start_mem`](#stbi__start_mem)
    - [`stbi__load_and_postprocess_8bit`](#stbi__load_and_postprocess_8bit)


---
### stbi\_load\_from\_callbacks<!-- {{#callable:stbi_load_from_callbacks}} -->
Loads an image from a set of callbacks and returns a pointer to the image data.
- **Inputs**:
    - `clbk`: A pointer to a `stbi_io_callbacks` structure that contains callback functions for reading data.
    - `user`: A user-defined pointer that is passed to the callback functions.
    - `x`: A pointer to an integer where the width of the loaded image will be stored.
    - `y`: A pointer to an integer where the height of the loaded image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the loaded image will be stored.
    - `req_comp`: An integer specifying the number of color components requested in the output image.
- **Control Flow**:
    - Initializes a `stbi__context` structure to manage the image loading process.
    - Calls [`stbi__start_callbacks`](#stbi__start_callbacks) to set up the context with the provided callbacks and user data.
    - Invokes [`stbi__load_and_postprocess_8bit`](#stbi__load_and_postprocess_8bit) to load the image data and process it according to the specified requirements.
- **Output**: Returns a pointer to the loaded image data in 8-bit format, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__start_callbacks`](#stbi__start_callbacks)
    - [`stbi__load_and_postprocess_8bit`](#stbi__load_and_postprocess_8bit)


---
### stbi\_load\_gif\_from\_memory<!-- {{#callable:stbi_load_gif_from_memory}} -->
Loads a GIF image from a memory buffer and optionally flips it vertically.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing the GIF image data.
    - `len`: The length of the memory buffer in bytes.
    - `delays`: A pointer to an integer array that will hold the delays for each frame in the GIF.
    - `x`: A pointer to an integer that will receive the width of the loaded image.
    - `y`: A pointer to an integer that will receive the height of the loaded image.
    - `z`: A pointer to an integer that will receive the number of channels in the loaded image.
    - `comp`: A pointer to an integer that will receive the number of components in the loaded image.
    - `req_comp`: The number of components requested for the output image.
- **Control Flow**:
    - Initializes a context `s` for reading the GIF data from the provided memory buffer.
    - Calls [`stbi__load_gif_main`](#stbi__load_gif_main) to load the GIF image and retrieve the pixel data along with frame delays and dimensions.
    - Checks if vertical flipping is enabled; if so, calls [`stbi__vertical_flip_slices`](#stbi__vertical_flip_slices) to flip the image data vertically.
    - Returns the pointer to the loaded image data.
- **Output**: Returns a pointer to the loaded image data in memory, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__start_mem`](#stbi__start_mem)
    - [`stbi__load_gif_main`](#stbi__load_gif_main)
    - [`stbi__vertical_flip_slices`](#stbi__vertical_flip_slices)


---
### stbi\_\_loadf\_main<!-- {{#callable:stbi__loadf_main}} -->
Loads an image in either HDR or LDR format and processes it accordingly.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the image data to be loaded.
    - `x`: A pointer to an integer where the width of the loaded image will be stored.
    - `y`: A pointer to an integer where the height of the loaded image will be stored.
    - `comp`: A pointer to an integer that indicates the number of color components in the loaded image.
    - `req_comp`: An integer that specifies the requested number of color components for the output image.
- **Control Flow**:
    - The function first checks if the image is in HDR format using [`stbi__hdr_test`](#stbi__hdr_test).
    - If the image is HDR, it calls [`stbi__hdr_load`](#stbi__hdr_load) to load the HDR data and processes it with [`stbi__float_postprocess`](#stbi__float_postprocess).
    - If the image is not HDR, it falls back to loading the image as an 8-bit LDR format using [`stbi__load_and_postprocess_8bit`](#stbi__load_and_postprocess_8bit).
    - If the LDR data is successfully loaded, it converts it to HDR format using [`stbi__ldr_to_hdr`](#stbi__ldr_to_hdr).
    - If neither loading method is successful, it returns an error message indicating an unknown image type.
- **Output**: Returns a pointer to a float array containing the image data in HDR format, or NULL if the image could not be loaded, along with an error message.
- **Functions called**:
    - [`stbi__hdr_test`](#stbi__hdr_test)
    - [`stbi__hdr_load`](#stbi__hdr_load)
    - [`stbi__float_postprocess`](#stbi__float_postprocess)
    - [`stbi__load_and_postprocess_8bit`](#stbi__load_and_postprocess_8bit)
    - [`stbi__ldr_to_hdr`](#stbi__ldr_to_hdr)


---
### stbi\_loadf\_from\_memory<!-- {{#callable:stbi_loadf_from_memory}} -->
Loads an image from a memory buffer and returns a pointer to the image data as floating-point values.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing the image data.
    - `len`: The length of the memory buffer in bytes.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of components in the image will be stored.
    - `req_comp`: The number of components requested (e.g., 3 for RGB, 4 for RGBA).
- **Control Flow**:
    - Initializes a `stbi__context` structure to manage the image loading process.
    - Calls [`stbi__start_mem`](#stbi__start_mem) to set up the context with the provided memory buffer and its length.
    - Delegates the actual loading of the image data to the [`stbi__loadf_main`](#stbi__loadf_main) function, passing the context and pointers for width, height, components, and requested components.
- **Output**: Returns a pointer to the loaded image data as an array of floating-point values, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__start_mem`](#stbi__start_mem)
    - [`stbi__loadf_main`](#stbi__loadf_main)


---
### stbi\_loadf\_from\_callbacks<!-- {{#callable:stbi_loadf_from_callbacks}} -->
Loads an image as floating-point values from a set of callbacks.
- **Inputs**:
    - `clbk`: A pointer to a `stbi_io_callbacks` structure that contains callback functions for reading data.
    - `user`: A user-defined pointer that is passed to the callback functions.
    - `x`: A pointer to an integer where the width of the loaded image will be stored.
    - `y`: A pointer to an integer where the height of the loaded image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the loaded image will be stored.
    - `req_comp`: An integer specifying the number of color components requested in the output image.
- **Control Flow**:
    - Initializes a `stbi__context` structure to manage the image loading process.
    - Calls [`stbi__start_callbacks`](#stbi__start_callbacks) to set up the context with the provided callbacks and user data.
    - Delegates the actual image loading to [`stbi__loadf_main`](#stbi__loadf_main), passing the context and pointers for width, height, components, and requested components.
- **Output**: Returns a pointer to the loaded image data as floating-point values, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__start_callbacks`](#stbi__start_callbacks)
    - [`stbi__loadf_main`](#stbi__loadf_main)


---
### stbi\_loadf<!-- {{#callable:stbi_loadf}} -->
`stbi_loadf` loads an image from a file and returns its pixel data as a float array.
- **Inputs**:
    - `filename`: A string representing the path to the image file to be loaded.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored.
    - `req_comp`: An integer specifying the number of color components requested (e.g., 3 for RGB, 4 for RGBA).
- **Control Flow**:
    - The function attempts to open the specified file in binary read mode using [`stbi__fopen`](#stbi__fopen).
    - If the file cannot be opened, it returns an error message using `stbi__errpf`.
    - If the file is successfully opened, it calls [`stbi_loadf_from_file`](#stbi_loadf_from_file) to load the image data, passing the file pointer and pointers for width, height, and components.
    - After loading the image, it closes the file using `fclose`.
    - Finally, it returns the pointer to the loaded image data.
- **Output**: Returns a pointer to a float array containing the pixel data of the image, or NULL if an error occurred.
- **Functions called**:
    - [`stbi__fopen`](#stbi__fopen)
    - [`stbi_loadf_from_file`](#stbi_loadf_from_file)


---
### stbi\_loadf\_from\_file<!-- {{#callable:stbi_loadf_from_file}} -->
Loads an image from a file and returns its pixel data as a float array.
- **Inputs**:
    - `f`: A pointer to a `FILE` object that represents the file from which the image will be loaded.
    - `x`: A pointer to an integer where the width of the loaded image will be stored.
    - `y`: A pointer to an integer where the height of the loaded image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the loaded image will be stored.
    - `req_comp`: An integer specifying the number of color components requested (e.g., 3 for RGB, 4 for RGBA).
- **Control Flow**:
    - Initializes a `stbi__context` structure to manage the image loading process.
    - Calls [`stbi__start_file`](#stbi__start_file) to prepare the context for reading from the specified file.
    - Delegates the actual loading of the image data to the [`stbi__loadf_main`](#stbi__loadf_main) function, passing the context and pointers for width, height, and component information.
- **Output**: Returns a pointer to a float array containing the pixel data of the loaded image, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__start_file`](#stbi__start_file)
    - [`stbi__loadf_main`](#stbi__loadf_main)


---
### stbi\_is\_hdr\_from\_memory<!-- {{#callable:stbi_is_hdr_from_memory}} -->
Determines if a given memory buffer contains HDR image data.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer that potentially contains HDR image data.
    - `len`: The length of the memory buffer in bytes.
- **Control Flow**:
    - Checks if HDR support is enabled using the `STBI_NO_HDR` preprocessor directive.
    - If HDR support is enabled, initializes a context `s` for reading from memory using [`stbi__start_mem`](#stbi__start_mem).
    - Calls [`stbi__hdr_test`](#stbi__hdr_test) with the context to determine if the buffer contains HDR data and returns the result.
    - If HDR support is not enabled, the function ignores the input parameters and returns 0.
- **Output**: Returns 1 if the buffer contains HDR data, 0 if it does not, or if HDR support is disabled.
- **Functions called**:
    - [`stbi__start_mem`](#stbi__start_mem)
    - [`stbi__hdr_test`](#stbi__hdr_test)


---
### stbi\_is\_hdr<!-- {{#callable:stbi_is_hdr}} -->
Determines if a file is an HDR image by checking its header.
- **Inputs**:
    - `filename`: A constant character pointer representing the path to the file to be checked.
- **Control Flow**:
    - The function attempts to open the file specified by `filename` in binary read mode.
    - If the file is successfully opened, it calls the [`stbi_is_hdr_from_file`](#stbi_is_hdr_from_file) function to check if the file is an HDR image.
    - After checking, it closes the file to free resources.
    - The result of the HDR check is returned; if the file could not be opened, it returns 0.
- **Output**: Returns a non-zero value if the file is an HDR image, and 0 if it is not or if the file could not be opened.
- **Functions called**:
    - [`stbi__fopen`](#stbi__fopen)
    - [`stbi_is_hdr_from_file`](#stbi_is_hdr_from_file)


---
### stbi\_is\_hdr\_from\_file<!-- {{#callable:stbi_is_hdr_from_file}} -->
Determines if a file contains HDR (High Dynamic Range) image data.
- **Inputs**:
    - `f`: A pointer to a `FILE` object representing the file to be checked for HDR content.
- **Control Flow**:
    - Checks if HDR support is enabled using the `STBI_NO_HDR` preprocessor directive.
    - If HDR support is enabled, it saves the current file position using `ftell`.
    - Initializes a `stbi__context` structure and starts reading the file with [`stbi__start_file`](#stbi__start_file).
    - Calls [`stbi__hdr_test`](#stbi__hdr_test) to check if the file contains HDR data.
    - Restores the original file position using `fseek`.
    - Returns the result of the HDR test.
    - If HDR support is not enabled, it ignores the input file and returns 0.
- **Output**: Returns 1 if the file contains HDR data, 0 if it does not, or if HDR support is disabled.
- **Functions called**:
    - [`stbi__start_file`](#stbi__start_file)
    - [`stbi__hdr_test`](#stbi__hdr_test)


---
### stbi\_is\_hdr\_from\_callbacks<!-- {{#callable:stbi_is_hdr_from_callbacks}} -->
Determines if a file is in HDR format using provided callbacks.
- **Inputs**:
    - `clbk`: A pointer to a `stbi_io_callbacks` structure that contains callback functions for reading data.
    - `user`: A pointer to user-defined data that is passed to the callback functions.
- **Control Flow**:
    - Checks if HDR support is enabled using the `STBI_NO_HDR` preprocessor directive.
    - If HDR support is enabled, initializes a `stbi__context` structure and starts the callbacks with [`stbi__start_callbacks`](#stbi__start_callbacks).
    - Calls [`stbi__hdr_test`](#stbi__hdr_test) to check if the data read from the callbacks is in HDR format and returns the result.
    - If HDR support is not enabled, the function ignores the input parameters and returns 0.
- **Output**: Returns 1 if the data is in HDR format, 0 if it is not, or if HDR support is disabled.
- **Functions called**:
    - [`stbi__start_callbacks`](#stbi__start_callbacks)
    - [`stbi__hdr_test`](#stbi__hdr_test)


---
### stbi\_ldr\_to\_hdr\_gamma<!-- {{#callable:stbi_ldr_to_hdr_gamma}} -->
Sets the gamma correction value for converting LDR to HDR images.
- **Inputs**:
    - `gamma`: A floating-point value representing the gamma correction factor to be applied during the conversion from low dynamic range (LDR) to high dynamic range (HDR).
- **Control Flow**:
    - The function directly assigns the input `gamma` value to the global variable `stbi__l2h_gamma`.
    - There are no conditional statements or loops; the function performs a single operation.
- **Output**: The function does not return a value; it modifies a global variable to store the gamma correction factor.


---
### stbi\_ldr\_to\_hdr\_scale<!-- {{#callable:stbi_ldr_to_hdr_scale}} -->
Sets the global scale factor for converting LDR (Low Dynamic Range) images to HDR (High Dynamic Range) images.
- **Inputs**:
    - `scale`: A float value representing the scale factor to be applied when converting LDR images to HDR.
- **Control Flow**:
    - The function directly assigns the input `scale` to the global variable `stbi__l2h_scale`.
    - There are no conditional statements or loops; the function performs a single operation.
- **Output**: The function does not return a value; it modifies the global variable `stbi__l2h_scale` to affect future LDR to HDR conversions.


---
### stbi\_hdr\_to\_ldr\_gamma<!-- {{#callable:stbi_hdr_to_ldr_gamma}} -->
Sets the inverse of the provided gamma value for HDR to LDR conversion.
- **Inputs**:
    - `gamma`: A float value representing the gamma correction factor used in the conversion from HDR to LDR.
- **Control Flow**:
    - Calculates the inverse of the input `gamma` value.
    - Assigns the calculated inverse to the global variable `stbi__h2l_gamma_i`.
- **Output**: This function does not return a value; it modifies a global variable to store the inverse gamma value.


---
### stbi\_hdr\_to\_ldr\_scale<!-- {{#callable:stbi_hdr_to_ldr_scale}} -->
The `stbi_hdr_to_ldr_scale` function sets a global scaling factor for converting HDR values to LDR.
- **Inputs**:
    - `scale`: A float value representing the scaling factor for converting HDR to LDR.
- **Control Flow**:
    - The function takes a single input argument, `scale`.
    - It calculates the inverse of the `scale` value and assigns it to the global variable `stbi__h2l_scale_i`.
- **Output**: The function does not return a value; it modifies a global variable to be used in HDR to LDR conversions.


---
### stbi\_\_refill\_buffer<!-- {{#callable:stbi__refill_buffer}} -->
Refills the buffer in the `stbi__context` structure by reading data from a user-defined input source.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the state of the buffer and the input source.
- **Control Flow**:
    - Calls the `read` function from the `io` member of the `stbi__context` structure to fill the buffer with data.
    - Updates the `callback_already_read` member to reflect how much data has already been read from the image buffer.
    - Checks if the read operation returned 0, indicating the end of the file.
    - If at the end of the file, sets the `img_buffer` to point to the start of the buffer and marks the end of the buffer as one byte beyond the start, initializing it to zero.
    - If data was read successfully, updates the `img_buffer` to point to the start of the buffer and sets `img_buffer_end` to the end of the newly read data.
- **Output**: The function does not return a value; it modifies the `stbi__context` structure in place to reflect the new state of the buffer after attempting to refill it.


---
### stbi\_\_get8<!-- {{#callable:stbi__get8}} -->
The `stbi__get8` function retrieves a single byte from a buffer or refills the buffer if necessary.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the buffer and its boundaries.
- **Control Flow**:
    - The function first checks if the current position in the buffer is less than the end of the buffer.
    - If there are bytes available, it returns the current byte and increments the buffer pointer.
    - If the buffer is exhausted and reading from callbacks is enabled, it calls [`stbi__refill_buffer`](#stbi__refill_buffer) to refill the buffer.
    - After refilling, it returns the next byte from the buffer.
    - If neither condition is met, it returns 0, indicating no byte could be retrieved.
- **Output**: Returns the next byte from the buffer or 0 if no byte is available.
- **Functions called**:
    - [`stbi__refill_buffer`](#stbi__refill_buffer)


---
### stbi\_\_at\_eof<!-- {{#callable:stbi__at_eof}} -->
Checks if the end of the file has been reached in a given context.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains information about the current state of the file being read.
- **Control Flow**:
    - The function first checks if the `read` function pointer in the `io` structure of `s` is set.
    - If `read` is set, it calls the `eof` function with `io_user_data` to determine if the end of the file has been reached; if not, it returns 0.
    - If the `read_from_callbacks` is 0, it indicates a special case where only a null character is at the end, and the function returns 1.
    - If the above conditions are not met, it checks if the current position in the image buffer has reached the end by comparing `img_buffer` with `img_buffer_end`.
- **Output**: Returns 1 if the end of the file is reached, 0 otherwise.


---
### stbi\_\_skip<!-- {{#callable:stbi__skip}} -->
The `stbi__skip` function advances the read position in an image buffer by a specified number of bytes.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the current state of the image buffer.
    - `n`: An integer representing the number of bytes to skip in the image buffer.
- **Control Flow**:
    - If `n` is zero, the function returns immediately as no skipping is needed.
    - If `n` is negative, the function sets the image buffer pointer to the end of the buffer and returns.
    - If `n` is positive and there is a read function defined, it checks the length of the remaining buffer.
    - If the remaining buffer length is less than `n`, it sets the image buffer pointer to the end and calls the skip function to skip the remaining bytes.
    - If there are enough bytes in the buffer, it simply advances the image buffer pointer by `n` bytes.
- **Output**: The function does not return a value; it modifies the `img_buffer` pointer within the `stbi__context` structure to reflect the new read position.


---
### stbi\_\_getn<!-- {{#callable:stbi__getn}} -->
The `stbi__getn` function reads `n` bytes of data into a buffer from a specified context, handling both buffered and direct I/O.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the state and I/O functions for reading data.
    - `buffer`: A pointer to a buffer where the read data will be stored.
    - `n`: An integer representing the number of bytes to read into the buffer.
- **Control Flow**:
    - The function first checks if the `read` function is defined in the `io` structure of the context.
    - If the available data in the buffer (`img_buffer`) is less than `n`, it copies the available data to the `buffer` and attempts to read the remaining bytes using the `read` function.
    - If the read operation is successful (i.e., the number of bytes read equals the remaining bytes needed), it updates the `img_buffer` pointer and returns success.
    - If there is enough data in the buffer, it directly copies `n` bytes from `img_buffer` to `buffer`, updates the `img_buffer` pointer, and returns success.
    - If neither condition is met, it returns failure.
- **Output**: The function returns 1 on success (indicating that `n` bytes were successfully read), or 0 on failure (indicating that not enough data was available to read).


---
### stbi\_\_get16be<!-- {{#callable:stbi__get16be}} -->
Reads two bytes from a `stbi__context` and combines them into a single 16-bit integer in big-endian order.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the state for reading data.
- **Control Flow**:
    - Calls `stbi__get8(s)` to read the first byte and stores it in the variable `z`.
    - Calls `stbi__get8(s)` again to read the second byte.
    - Combines the two bytes into a single 16-bit integer by shifting the first byte left by 8 bits and adding the second byte.
- **Output**: Returns a 16-bit integer constructed from two bytes read from the `stbi__context`, with the first byte being the most significant byte.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_get32be<!-- {{#callable:stbi__get32be}} -->
Reads a 32-bit unsigned integer from a given context in big-endian format.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the data stream from which the 32-bit integer is read.
- **Control Flow**:
    - Calls the [`stbi__get16be`](#stbi__get16be) function to read the first 16 bits (2 bytes) from the context `s` and stores it in the variable `z`.
    - Calls the [`stbi__get16be`](#stbi__get16be) function again to read the next 16 bits (2 bytes) from the context `s`.
    - Combines the two 16-bit values by shifting the first value `z` left by 16 bits and adding the second 16-bit value, effectively constructing a 32-bit integer.
- **Output**: Returns a `stbi__uint32` value representing the 32-bit unsigned integer read from the context, in big-endian order.
- **Functions called**:
    - [`stbi__get16be`](#stbi__get16be)


---
### stbi\_\_get16le<!-- {{#callable:stbi__get16le}} -->
Reads two little-endian bytes from the `stbi__context` and combines them into a single 16-bit integer.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the data stream from which bytes are read.
- **Control Flow**:
    - Calls the [`stbi__get8`](#stbi__get8) function to read the first byte and stores it in the variable `z`.
    - Calls the [`stbi__get8`](#stbi__get8) function again to read the second byte, shifts it left by 8 bits, and adds it to `z` to form a 16-bit integer.
- **Output**: Returns a 16-bit integer constructed from two bytes read in little-endian order.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_get32le<!-- {{#callable:stbi__get32le}} -->
The `stbi__get32le` function reads a 32-bit little-endian integer from a given context.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the data source from which the integer is read.
- **Control Flow**:
    - The function first calls [`stbi__get16le`](#stbi__get16le) to read the lower 16 bits of the 32-bit integer.
    - It then calls [`stbi__get16le`](#stbi__get16le) again to read the next 16 bits, which are shifted left by 16 bits to form the upper part of the integer.
    - The two 16-bit values are combined using addition to form the final 32-bit integer.
- **Output**: The function returns a `stbi__uint32` value representing the 32-bit integer read from the context in little-endian format.
- **Functions called**:
    - [`stbi__get16le`](#stbi__get16le)


---
### stbi\_\_compute\_y<!-- {{#callable:stbi__compute_y}} -->
Computes the luminance (Y) value from RGB color components using a weighted sum.
- **Inputs**:
    - `r`: The red component of the color, represented as an integer.
    - `g`: The green component of the color, represented as an integer.
    - `b`: The blue component of the color, represented as an integer.
- **Control Flow**:
    - The function takes three integer inputs representing the red, green, and blue components of a color.
    - It calculates the luminance by applying specific weights to each color component: 77 for red, 150 for green, and 29 for blue.
    - The weighted sum is then right-shifted by 8 bits to scale down the result.
    - Finally, the result is cast to an `stbi_uc` type before being returned.
- **Output**: Returns the computed luminance value as an `stbi_uc`, which is an unsigned character type.


---
### stbi\_\_convert\_format<!-- {{#callable:stbi__convert_format}} -->
Converts an image from one color format to another based on the specified number of components.
- **Inputs**:
    - `data`: Pointer to the input image data in the original format.
    - `img_n`: The number of components in the input image (e.g., 1 for grayscale, 3 for RGB, 4 for RGBA).
    - `req_comp`: The desired number of components in the output image.
    - `x`: The width of the image.
    - `y`: The height of the image.
- **Control Flow**:
    - Checks if the requested number of components is the same as the input; if so, returns the original data.
    - Asserts that the requested components are within the valid range (1 to 4).
    - Allocates memory for the new image data based on the requested components and dimensions.
    - If memory allocation fails, frees the original data and returns an error message.
    - Iterates over each scanline of the image, converting pixel data from the original format to the requested format using a switch statement.
    - Uses macros to handle different combinations of input and output component counts efficiently.
    - Frees the original image data after conversion and returns the newly allocated image data.
- **Output**: Returns a pointer to the newly allocated image data in the requested format, or an error message if the conversion fails.
- **Functions called**:
    - [`stbi__malloc_mad3`](#stbi__malloc_mad3)
    - [`stbi__compute_y`](#stbi__compute_y)


---
### stbi\_\_compute\_y\_16<!-- {{#callable:stbi__compute_y_16}} -->
Computes the luminance (Y) value from RGB components using a specific weighted formula.
- **Inputs**:
    - `r`: The red component of the RGB color, represented as an integer.
    - `g`: The green component of the RGB color, represented as an integer.
    - `b`: The blue component of the RGB color, represented as an integer.
- **Control Flow**:
    - The function takes three integer inputs representing the red, green, and blue components of a color.
    - It calculates the luminance using the formula: (r*77 + g*150 + b*29) >> 8.
    - The result is cast to a `stbi__uint16` type before being returned.
- **Output**: Returns the computed luminance value as a `stbi__uint16`, which represents the brightness of the color.


---
### stbi\_\_convert\_format16<!-- {{#callable:stbi__convert_format16}} -->
Converts an image from one format to another with a specified number of components.
- **Inputs**:
    - `data`: Pointer to the input image data in 16-bit unsigned integer format.
    - `img_n`: The number of components in the input image (1 to 4).
    - `req_comp`: The desired number of components in the output image (1 to 4).
    - `x`: The width of the image.
    - `y`: The height of the image.
- **Control Flow**:
    - Checks if the requested number of components equals the input number of components; if so, returns the original data.
    - Asserts that the requested components are within the valid range (1 to 4).
    - Allocates memory for the new image data based on the requested components and image dimensions; if allocation fails, frees the original data and returns an error.
    - Iterates over each scanline of the image, converting the pixel data from the source format to the requested format using a switch statement based on the combination of input and output components.
    - Uses macros to define conversion cases for different combinations of input and output components, handling each case accordingly.
    - Frees the original data after conversion and returns the pointer to the newly allocated image data.
- **Output**: Returns a pointer to the newly allocated image data in the requested format, or an error message if the conversion fails.
- **Functions called**:
    - [`stbi__malloc`](#stbi__malloc)
    - [`stbi__compute_y_16`](#stbi__compute_y_16)


---
### stbi\_\_ldr\_to\_hdr<!-- {{#callable:stbi__ldr_to_hdr}} -->
Converts low dynamic range (LDR) image data to high dynamic range (HDR) format.
- **Inputs**:
    - `data`: A pointer to the LDR image data in the form of an array of unsigned characters.
    - `x`: The width of the image in pixels.
    - `y`: The height of the image in pixels.
    - `comp`: The number of color components in the image (e.g., 3 for RGB, 4 for RGBA).
- **Control Flow**:
    - Checks if the input `data` is NULL; if so, returns NULL.
    - Allocates memory for the output HDR data based on the dimensions and components of the image.
    - If memory allocation fails, frees the input `data` and returns an error message.
    - Determines the number of non-alpha components based on the `comp` value.
    - Iterates over each pixel to convert the LDR values to HDR using a gamma correction and scaling factor.
    - If there is an alpha component, it copies the alpha values directly from the input data to the output.
- **Output**: Returns a pointer to the newly allocated array containing the HDR image data, or NULL if the input data was NULL or memory allocation failed.
- **Functions called**:
    - [`stbi__malloc_mad4`](#stbi__malloc_mad4)


---
### stbi\_\_hdr\_to\_ldr<!-- {{#callable:stbi__hdr_to_ldr}} -->
Converts HDR image data to LDR format by scaling and clamping pixel values.
- **Inputs**:
    - `data`: A pointer to an array of floating-point HDR pixel data.
    - `x`: The width of the image in pixels.
    - `y`: The height of the image in pixels.
    - `comp`: The number of color components per pixel (e.g., 3 for RGB, 4 for RGBA).
- **Control Flow**:
    - Checks if the input `data` is NULL; if so, returns NULL.
    - Allocates memory for the output image based on the dimensions and components.
    - If memory allocation fails, frees the input data and returns an error.
    - Determines the number of non-alpha components based on the `comp` value.
    - Iterates over each pixel in the image, processing each color component.
    - For each component, scales the HDR value, applies gamma correction, and clamps it to the range [0, 255].
    - Handles the alpha component separately if present.
    - Frees the input `data` after processing and returns the output image.
- **Output**: Returns a pointer to an array of `stbi_uc` containing the converted LDR pixel data, or NULL if an error occurred.
- **Functions called**:
    - [`stbi__malloc_mad3`](#stbi__malloc_mad3)


---
### stbi\_\_build\_huffman<!-- {{#callable:stbi__build_huffman}} -->
The `stbi__build_huffman` function constructs Huffman coding tables from symbol size counts for JPEG compression.
- **Inputs**:
    - `h`: A pointer to a `stbi__huffman` structure that will hold the generated Huffman tables.
    - `count`: An array of integers representing the number of symbols for each possible size (1 to 16 bits).
- **Control Flow**:
    - Iterates over the `count` array to build a size list for each symbol, checking for errors if the size list exceeds expected limits.
    - Computes actual Huffman symbols based on the size list, ensuring that the generated codes do not exceed the maximum allowed for their bit length.
    - Constructs a fast lookup table for decoding, marking entries with 255 for symbols that are not accelerated.
- **Output**: Returns 1 on success, indicating that the Huffman tables were built successfully, or an error message if the input data is corrupt.
- **Functions called**:
    - [`stbi__err`](#stbi__err)


---
### stbi\_\_build\_fast\_ac<!-- {{#callable:stbi__build_fast_ac}} -->
The `stbi__build_fast_ac` function populates a fast access table for Huffman coding based on provided Huffman data.
- **Inputs**:
    - `fast_ac`: A pointer to an array of `stbi__int16` that will be filled with fast access values.
    - `h`: A pointer to a `stbi__huffman` structure containing Huffman coding information, including fast access values, size, and actual values.
- **Control Flow**:
    - The function iterates over a range defined by `FAST_BITS` to process each potential Huffman code.
    - For each index `i`, it retrieves the corresponding fast value from the `h->fast` array.
    - If the fast value is less than 255, it extracts the run length, magnitude bits, and length from the `h->values` and `h->size` arrays.
    - If the magnitude bits are non-zero and the total length fits within `FAST_BITS`, it calculates a key `k` based on the index and length.
    - The function checks if `k` can be represented within the range of -128 to 127, and if so, it stores a combined value in the `fast_ac` array.
- **Output**: The function does not return a value but modifies the `fast_ac` array in place, storing encoded values that facilitate fast decoding of Huffman codes.


---
### stbi\_\_grow\_buffer\_unsafe<!-- {{#callable:stbi__grow_buffer_unsafe}} -->
The `stbi__grow_buffer_unsafe` function expands the buffer of a JPEG decoder by reading bytes from the input stream.
- **Inputs**:
    - `j`: A pointer to a `stbi__jpeg` structure that contains the state of the JPEG decoder, including the input stream and buffer.
- **Control Flow**:
    - The function enters a do-while loop that continues until the `code_bits` exceeds 24.
    - Within the loop, it checks if the `nomore` flag is set; if not, it reads a byte from the input stream using [`stbi__get8`](#stbi__get8).
    - If the byte read is 0xff, it reads another byte to check for marker bytes, consuming any additional 0xff bytes encountered.
    - If a non-zero byte is found after the 0xff bytes, it sets the `marker` in the `stbi__jpeg` structure and sets `nomore` to 1, then exits the function.
    - If the byte is not 0xff, it shifts the byte into the `code_buffer` and increments the `code_bits` by 8.
- **Output**: The function does not return a value; it modifies the state of the `stbi__jpeg` structure, particularly the `code_buffer`, `code_bits`, and potentially the `marker` and `nomore` fields.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_jpeg\_huff\_decode<!-- {{#callable:stbi__jpeg_huff_decode}} -->
Decodes a Huffman-encoded symbol from a JPEG stream using a provided Huffman table.
- **Inputs**:
    - `j`: A pointer to a `stbi__jpeg` structure that contains the JPEG decoding state, including the code buffer and the number of valid bits.
    - `h`: A pointer to a `stbi__huffman` structure that contains the Huffman table used for decoding.
- **Control Flow**:
    - Checks if the number of valid bits in the code buffer is less than 16; if so, it calls [`stbi__grow_buffer_unsafe`](#stbi__grow_buffer_unsafe) to ensure enough data is available.
    - Extracts the top `FAST_BITS` bits from the code buffer to determine the symbol ID and checks if it corresponds to a fast lookup in the Huffman table.
    - If the fast lookup is successful (k < 255), it verifies the size of the Huffman code and updates the code buffer and bits accordingly.
    - If the fast lookup fails, it shifts the code buffer to check against the maximum codes in the Huffman table, determining the length of the code needed.
    - If the required code length exceeds the available bits, it returns -1 indicating an error.
    - Once the correct code length is determined, it extracts the symbol ID and checks for validity before updating the code buffer and returning the decoded symbol.
- **Output**: Returns the decoded symbol value from the Huffman table, or -1 if an error occurs during decoding.
- **Functions called**:
    - [`stbi__grow_buffer_unsafe`](#stbi__grow_buffer_unsafe)


---
### stbi\_\_extend\_receive<!-- {{#callable:stbi__extend_receive}} -->
Extracts a signed integer from a JPEG stream based on the specified number of bits.
- **Inputs**:
    - `j`: A pointer to a `stbi__jpeg` structure that contains the current state of the JPEG decoding process, including the code buffer and the number of bits available.
    - `n`: An integer representing the number of bits to extract from the code buffer.
- **Control Flow**:
    - Checks if the number of available bits in `j->code_bits` is less than `n`, and if so, calls `stbi__grow_buffer_unsafe(j)` to attempt to increase the buffer size.
    - If there are still not enough bits after attempting to grow the buffer, the function returns 0 to indicate failure.
    - Extracts the sign bit from the most significant bit (MSB) of `j->code_buffer` to determine if the result should be positive or negative.
    - Rotates the `j->code_buffer` to the right by `n` bits to align the desired bits for extraction.
    - Clears the bits in `j->code_buffer` that have been extracted and updates the number of available bits accordingly.
    - Returns the extracted value adjusted by the bias for signed integers based on the sign bit.
- **Output**: Returns the extracted signed integer value based on the specified number of bits, or 0 if there were not enough bits available.
- **Functions called**:
    - [`stbi__grow_buffer_unsafe`](#stbi__grow_buffer_unsafe)


---
### stbi\_\_jpeg\_get\_bits<!-- {{#callable:stbi__jpeg_get_bits}} -->
Extracts a specified number of bits from a JPEG stream.
- **Inputs**:
    - `j`: A pointer to a `stbi__jpeg` structure that contains the current state of the JPEG decoder, including the bit buffer and the number of bits available.
    - `n`: An integer representing the number of bits to extract from the JPEG stream.
- **Control Flow**:
    - Checks if the number of available bits (`j->code_bits`) is less than the requested number of bits (`n`); if so, it calls `stbi__grow_buffer_unsafe(j)` to attempt to increase the buffer size.
    - If there are still not enough bits available after attempting to grow the buffer, the function returns 0, indicating that it cannot provide the requested bits.
    - If enough bits are available, it performs a left rotation on the `j->code_buffer` to align the bits and masks out the bits that have been extracted.
    - Updates the `j->code_buffer` to remove the extracted bits and decrements the `j->code_bits` by the number of bits extracted.
    - Finally, it returns the extracted bits.
- **Output**: Returns the extracted bits as an integer, or 0 if there were not enough bits available in the stream.
- **Functions called**:
    - [`stbi__grow_buffer_unsafe`](#stbi__grow_buffer_unsafe)


---
### stbi\_\_jpeg\_get\_bit<!-- {{#callable:stbi__jpeg_get_bit}} -->
Retrieves the next bit from the JPEG bitstream.
- **Inputs**:
    - `j`: A pointer to a `stbi__jpeg` structure that contains the current state of the JPEG decoder, including the bitstream buffer and the number of bits available.
- **Control Flow**:
    - Checks if `code_bits` in the `stbi__jpeg` structure is less than 1; if so, it calls [`stbi__grow_buffer_unsafe`](#stbi__grow_buffer_unsafe) to attempt to refill the buffer.
    - If `code_bits` is still less than 1 after attempting to grow the buffer, it returns 0, indicating that there are no bits left to read.
    - Stores the current `code_buffer` value in `k`, shifts the `code_buffer` left by one bit, and decrements `code_bits` by one.
    - Returns the most significant bit of `k` by performing a bitwise AND operation with `0x80000000`.
- **Output**: Returns the value of the most significant bit (MSB) of the current `code_buffer`, which is either 0 or 1, depending on the bit read from the stream.
- **Functions called**:
    - [`stbi__grow_buffer_unsafe`](#stbi__grow_buffer_unsafe)


---
### stbi\_\_jpeg\_decode\_block<!-- {{#callable:stbi__jpeg_decode_block}} -->
Decodes a JPEG block by processing DC and AC coefficients using Huffman coding.
- **Inputs**:
    - `j`: A pointer to the `stbi__jpeg` structure containing JPEG decoding state.
    - `data`: An array of 64 shorts to store the decoded coefficients.
    - `hdc`: A pointer to the `stbi__huffman` structure for decoding DC coefficients.
    - `hac`: A pointer to the `stbi__huffman` structure for decoding AC coefficients.
    - `fac`: An array of `stbi__int16` used for fast AC decoding.
    - `b`: An integer representing the component index for the current block.
    - `dequant`: An array of `stbi__uint16` used for dequantization of coefficients.
- **Control Flow**:
    - Checks if there are enough bits in the buffer; if not, it calls [`stbi__grow_buffer_unsafe`](#stbi__grow_buffer_unsafe) to expand the buffer.
    - Decodes the DC coefficient using [`stbi__jpeg_huff_decode`](#stbi__jpeg_huff_decode) and checks for errors.
    - Initializes the `data` array to zero to prepare for storing coefficients.
    - Calculates the DC value and updates the predictor, checking for validity.
    - Decodes the AC coefficients in a loop until all 64 coefficients are processed or an end condition is met.
    - For each AC coefficient, it checks if it can use the fast path or needs to decode using Huffman coding.
    - Handles the case for zero-length runs and updates the `data` array accordingly.
- **Output**: Returns 1 on successful decoding of the block, or an error code if decoding fails.
- **Functions called**:
    - [`stbi__grow_buffer_unsafe`](#stbi__grow_buffer_unsafe)
    - [`stbi__jpeg_huff_decode`](#stbi__jpeg_huff_decode)
    - [`stbi__err`](#stbi__err)
    - [`stbi__extend_receive`](#stbi__extend_receive)
    - [`stbi__addints_valid`](#stbi__addints_valid)
    - [`stbi__mul2shorts_valid`](#stbi__mul2shorts_valid)


---
### stbi\_\_jpeg\_decode\_block\_prog\_dc<!-- {{#callable:stbi__jpeg_decode_block_prog_dc}} -->
Decodes the DC coefficient of a JPEG image block during progressive decoding.
- **Inputs**:
    - `j`: A pointer to the `stbi__jpeg` structure containing JPEG decoding state and parameters.
    - `data`: An array of 64 shorts where the decoded DC coefficient will be stored.
    - `hdc`: A pointer to the `stbi__huffman` structure used for Huffman decoding of DC coefficients.
    - `b`: An integer representing the index of the current image component being processed.
- **Control Flow**:
    - Checks if `spec_end` is non-zero, returning an error if true, indicating that DC and AC coefficients cannot be merged.
    - Ensures that there are enough bits in the buffer for decoding; if not, it calls [`stbi__grow_buffer_unsafe`](#stbi__grow_buffer_unsafe) to expand the buffer.
    - If `succ_high` is zero, it performs the first scan for the DC coefficient, initializing the `data` array to zero and decoding the DC coefficient using Huffman decoding.
    - Validates the decoded value and updates the DC predictor for the specified component, ensuring that the delta is valid.
    - If `succ_high` is not zero, it performs a refinement scan, potentially modifying the DC coefficient based on the next bit read from the stream.
- **Output**: Returns 1 on successful decoding of the DC coefficient, or an error code if any validation fails.
- **Functions called**:
    - [`stbi__err`](#stbi__err)
    - [`stbi__grow_buffer_unsafe`](#stbi__grow_buffer_unsafe)
    - [`stbi__jpeg_huff_decode`](#stbi__jpeg_huff_decode)
    - [`stbi__extend_receive`](#stbi__extend_receive)
    - [`stbi__addints_valid`](#stbi__addints_valid)
    - [`stbi__mul2shorts_valid`](#stbi__mul2shorts_valid)
    - [`stbi__jpeg_get_bit`](#stbi__jpeg_get_bit)


---
### stbi\_\_jpeg\_decode\_block\_prog\_ac<!-- {{#callable:stbi__jpeg_decode_block_prog_ac}} -->
Decodes the AC coefficients of a JPEG image block using Huffman coding.
- **Inputs**:
    - `j`: A pointer to a `stbi__jpeg` structure containing JPEG decoding state information.
    - `data`: An array of 64 `short` integers to store the decoded AC coefficients.
    - `hac`: A pointer to a `stbi__huffman` structure that contains Huffman coding tables for AC coefficients.
    - `fac`: An array of `stbi__int16` integers used for fast AC decoding.
- **Control Flow**:
    - Checks if `spec_start` is zero, returning an error if true.
    - If `succ_high` is zero, it processes the AC coefficients in a normal scan.
    - Handles end-of-block (EOB) runs and decodes coefficients using either a fast path or standard Huffman decoding.
    - If `succ_high` is not zero, it performs a refinement scan for the AC coefficients, adjusting their values based on the current state.
    - The function continues decoding until all coefficients in the specified range are processed.
- **Output**: Returns 1 on successful decoding of the block, or an error code if a decoding error occurs.
- **Functions called**:
    - [`stbi__err`](#stbi__err)
    - [`stbi__grow_buffer_unsafe`](#stbi__grow_buffer_unsafe)
    - [`stbi__jpeg_huff_decode`](#stbi__jpeg_huff_decode)
    - [`stbi__jpeg_get_bits`](#stbi__jpeg_get_bits)
    - [`stbi__extend_receive`](#stbi__extend_receive)
    - [`stbi__jpeg_get_bit`](#stbi__jpeg_get_bit)


---
### stbi\_\_clamp<!-- {{#callable:stbi__clamp}} -->
Clamps an integer value to the range of 0 to 255.
- **Inputs**:
    - `x`: An integer value that needs to be clamped.
- **Control Flow**:
    - The function first checks if the unsigned version of `x` is greater than 255.
    - If `x` is less than 0, it returns 0.
    - If `x` is greater than 255, it returns 255.
    - If `x` is within the range of 0 to 255, it casts `x` to `stbi_uc` and returns it.
- **Output**: Returns a `stbi_uc` value that is clamped between 0 and 255.


---
### stbi\_\_idct\_block<!-- {{#callable:stbi__idct_block}} -->
Performs the inverse discrete cosine transform (IDCT) on an 8x8 block of DCT coefficients and outputs the resulting pixel values.
- **Inputs**:
    - `out`: A pointer to the output array where the resulting pixel values will be stored.
    - `out_stride`: The stride (number of bytes per row) of the output array.
    - `data`: An array of 64 short integers representing the DCT coefficients to be transformed.
- **Control Flow**:
    - Iterates over each column of the 8x8 block to process the DCT coefficients.
    - Checks if all coefficients in the column are zero; if so, it sets the corresponding output values to a scaled constant derived from the first coefficient.
    - If not all coefficients are zero, it calls the `STBI__IDCT_1D` macro to perform a 1D IDCT on the coefficients.
    - After processing the columns, it iterates over the rows to apply the IDCT again using the results from the first pass.
    - Finally, it clamps the output values to ensure they fall within the valid range for pixel values (0 to 255).
- **Output**: The function modifies the output array in place, filling it with pixel values derived from the inverse DCT transformation of the input coefficients.
- **Functions called**:
    - [`stbi__clamp`](#stbi__clamp)


---
### stbi\_\_idct\_simd<!-- {{#callable:stbi__idct_simd}} -->
Performs the inverse discrete cosine transform (IDCT) using SIMD instructions on a block of data.
- **Inputs**:
    - `out`: Pointer to the output buffer where the transformed data will be stored.
    - `out_stride`: The stride (number of bytes) between rows in the output buffer.
    - `data`: An array of 64 short integers representing the input DCT coefficients.
- **Control Flow**:
    - Initializes SIMD vectors with predefined constants for the IDCT calculations.
    - Defines several macros for performing operations such as multiplication, addition, and subtraction on SIMD vectors.
    - Calculates the even and odd parts of the IDCT using the defined macros and SIMD operations.
    - Combines results from the even and odd calculations to produce the final output values.
    - Applies the butterfly operation to rearrange and combine the results into the output buffer.
- **Output**: The function does not return a value; instead, it writes the transformed data directly to the output buffer specified by the `out` parameter.


---
### stbi\_\_get\_marker<!-- {{#callable:stbi__get_marker}} -->
The `stbi__get_marker` function retrieves the next JPEG marker from a given JPEG stream.
- **Inputs**:
    - `j`: A pointer to a `stbi__jpeg` structure that contains the JPEG stream and the current marker state.
- **Control Flow**:
    - The function first checks if the current marker in the `stbi__jpeg` structure is not `STBI__MARKER_none`.
    - If a valid marker exists, it retrieves the marker, resets it to `STBI__MARKER_none`, and returns the marker value.
    - If no valid marker exists, it reads a byte from the stream using [`stbi__get8`](#stbi__get8).
    - If the byte read is not `0xff`, it returns `STBI__MARKER_none` indicating no valid marker was found.
    - If the byte is `0xff`, it enters a loop to consume any subsequent `0xff` bytes, reading until a non-`0xff` byte is found.
    - Finally, it returns the non-`0xff` byte as the next marker.
- **Output**: The function returns the next valid JPEG marker byte or `STBI__MARKER_none` if no valid marker is found.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_jpeg\_reset<!-- {{#callable:stbi__jpeg_reset}} -->
Resets the state of a `stbi__jpeg` structure to its initial conditions.
- **Inputs**:
    - `j`: A pointer to a `stbi__jpeg` structure that holds the state of the JPEG decoder.
- **Control Flow**:
    - Sets the `code_bits` and `code_buffer` fields of the `stbi__jpeg` structure to 0, indicating no bits are currently buffered.
    - Resets the `nomore` field to 0, which likely indicates that there are more data to process.
    - Initializes the DC predictor values for all image components to 0, preparing for new image data.
    - Sets the `marker` field to `STBI__MARKER_none`, indicating that no markers are currently being processed.
    - Determines the `todo` field based on the `restart_interval`, defaulting to a large value if no interval is set, which controls the number of Minimum Coded Units (MCUs) to process.
    - Resets the `eob_run` field to 0, which likely tracks the number of end-of-block codes encountered.
- **Output**: The function does not return a value; it modifies the state of the `stbi__jpeg` structure directly.


---
### stbi\_\_parse\_entropy\_coded\_data<!-- {{#callable:stbi__parse_entropy_coded_data}} -->
Parses entropy-coded data from a JPEG image, handling both progressive and non-progressive scans.
- **Inputs**:
    - `z`: A pointer to a `stbi__jpeg` structure that contains the JPEG image data and state.
- **Control Flow**:
    - The function begins by resetting the JPEG state using `stbi__jpeg_reset(z)`.
    - It checks if the JPEG is not progressive and if it has a single scan component.
    - For non-interleaved data, it calculates the width and height in blocks and processes each block in a nested loop.
    - For interleaved data, it processes multiple components in a nested loop, handling each component's blocks sequentially.
    - During processing, it decodes each block using [`stbi__jpeg_decode_block`](#stbi__jpeg_decode_block) or [`stbi__jpeg_decode_block_prog_dc`](#stbi__jpeg_decode_block_prog_dc)/[`stbi__jpeg_decode_block_prog_ac`](#stbi__jpeg_decode_block_prog_ac) based on the scan type.
    - After processing each MCU (Minimum Coded Unit), it checks if the restart interval has been reached and handles it accordingly.
    - If the JPEG is progressive, it follows a similar structure but uses different decoding functions for progressive data.
- **Output**: Returns 1 on successful parsing of the data, or 0 if an error occurs during decoding.
- **Functions called**:
    - [`stbi__jpeg_reset`](#stbi__jpeg_reset)
    - [`stbi__jpeg_decode_block`](#stbi__jpeg_decode_block)
    - [`stbi__grow_buffer_unsafe`](#stbi__grow_buffer_unsafe)
    - [`stbi__jpeg_decode_block_prog_dc`](#stbi__jpeg_decode_block_prog_dc)
    - [`stbi__jpeg_decode_block_prog_ac`](#stbi__jpeg_decode_block_prog_ac)


---
### stbi\_\_jpeg\_dequantize<!-- {{#callable:stbi__jpeg_dequantize}} -->
The `stbi__jpeg_dequantize` function scales an array of quantized JPEG coefficients by a corresponding array of dequantization values.
- **Inputs**:
    - `data`: A pointer to an array of `short` integers representing quantized JPEG coefficients.
    - `dequant`: A pointer to an array of `stbi__uint16` integers representing dequantization values.
- **Control Flow**:
    - The function initializes a loop counter `i` to iterate from 0 to 63.
    - Within the loop, each element of the `data` array is multiplied by the corresponding element in the `dequant` array.
- **Output**: The function does not return a value; it modifies the `data` array in place.


---
### stbi\_\_jpeg\_finish<!-- {{#callable:stbi__jpeg_finish}} -->
The `stbi__jpeg_finish` function processes the final steps of JPEG decoding for progressive images.
- **Inputs**:
    - `z`: A pointer to a `stbi__jpeg` structure that contains information about the JPEG image being processed, including its components and decoding state.
- **Control Flow**:
    - The function first checks if the JPEG image is progressive by evaluating `z->progressive`.
    - If the image is progressive, it enters a loop that iterates over each image component (`img_n`).
    - For each component, it calculates the width (`w`) and height (`h`) in blocks of 8 pixels.
    - Nested loops iterate over the height and width of the component, processing each 8x8 block.
    - Within the innermost loop, it retrieves the coefficient data for the current block, dequantizes it using [`stbi__jpeg_dequantize`](#stbi__jpeg_dequantize), and applies the inverse discrete cosine transform (IDCT) using `z->idct_block_kernel`.
- **Output**: The function does not return a value; instead, it modifies the `data` field of each image component in the `stbi__jpeg` structure to store the decoded pixel data.
- **Functions called**:
    - [`stbi__jpeg_dequantize`](#stbi__jpeg_dequantize)


---
### stbi\_\_process\_marker<!-- {{#callable:stbi__process_marker}} -->
Processes a JPEG marker and updates the JPEG state accordingly.
- **Inputs**:
    - `z`: A pointer to a `stbi__jpeg` structure that holds the JPEG state and data.
    - `m`: An integer representing the marker type to be processed.
- **Control Flow**:
    - The function begins by checking the value of the marker `m` using a switch statement.
    - If the marker is `STBI__MARKER_none`, it returns an error indicating that a marker was expected.
    - For the DRI marker (0xDD), it checks the length and updates the restart interval.
    - For the DQT marker (0xDB), it processes the quantization table, validating types and reading values into the `dequant` array.
    - For the DHT marker (0xC4), it processes the Huffman table, validating the header and reading sizes and values into the appropriate structures.
    - If the marker is in the range for APP blocks or comments, it checks the length and processes specific APP segments like JFIF and Adobe.
    - If none of the cases match, it returns an error for an unknown marker.
- **Output**: Returns 1 on successful processing of the marker, 0 on failure, or an error message indicating the type of corruption encountered.
- **Functions called**:
    - [`stbi__err`](#stbi__err)
    - [`stbi__get16be`](#stbi__get16be)
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__build_huffman`](#stbi__build_huffman)
    - [`stbi__build_fast_ac`](#stbi__build_fast_ac)
    - [`stbi__skip`](#stbi__skip)


---
### stbi\_\_process\_scan\_header<!-- {{#callable:stbi__process_scan_header}} -->
Processes the scan header of a JPEG image and validates its components.
- **Inputs**:
    - `z`: A pointer to a `stbi__jpeg` structure that contains the JPEG image data and state.
- **Control Flow**:
    - Reads the length of the scan header (`Ls`) and the number of components in the scan (`scan_n`).
    - Validates that `scan_n` is within the acceptable range and matches the expected length of the scan header.
    - Iterates over each component in the scan to retrieve its ID and quantization table information, validating against the image components.
    - Checks if the component's Huffman table indices are valid.
    - Reads and validates the spectral selection and successively high/low values, ensuring they conform to JPEG standards based on whether the image is progressive or not.
- **Output**: Returns 1 on success, or 0 if an error occurs, with an error message indicating the type of corruption.
- **Functions called**:
    - [`stbi__get16be`](#stbi__get16be)
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__err`](#stbi__err)


---
### stbi\_\_free\_jpeg\_components<!-- {{#callable:stbi__free_jpeg_components}} -->
The `stbi__free_jpeg_components` function frees allocated memory for JPEG component data in a given `stbi__jpeg` structure.
- **Inputs**:
    - `z`: A pointer to an `stbi__jpeg` structure containing JPEG component data.
    - `ncomp`: An integer representing the number of components to free.
    - `why`: An integer that is returned as the output, typically used for error reporting or logging.
- **Control Flow**:
    - The function iterates over each component in the `img_comp` array of the `stbi__jpeg` structure up to `ncomp` times.
    - For each component, it checks if `raw_data`, `raw_coeff`, or `linebuf` pointers are not NULL.
    - If any of these pointers are not NULL, it frees the associated memory using `STBI_FREE` and sets the pointers to NULL or zero to avoid dangling references.
    - Finally, the function returns the value of `why` after processing all components.
- **Output**: The function returns the integer value passed as `why`, which may be used for further processing or logging.


---
### stbi\_\_process\_frame\_header<!-- {{#callable:stbi__process_frame_header}} -->
Processes the frame header of a JPEG image, validating its parameters and preparing for decoding.
- **Inputs**:
    - `z`: A pointer to a `stbi__jpeg` structure that contains the JPEG context and image components.
    - `scan`: An integer indicating the scan type, which determines how the image data will be processed.
- **Control Flow**:
    - Initializes local variables and retrieves the length of the frame header using [`stbi__get16be`](#stbi__get16be).
    - Checks the validity of the frame header length and the bit depth, returning errors for invalid values.
    - Retrieves the image dimensions (height and width) and checks for zero or excessively large dimensions.
    - Validates the number of color components and initializes component data structures.
    - Checks if the frame header length matches the expected size based on the number of components.
    - Processes each color component to retrieve its ID, horizontal and vertical sampling factors, and quantization table index, validating each value.
    - If the scan type is not `STBI__SCAN_load`, the function returns early.
    - Validates the maximum horizontal and vertical sampling factors and computes the maximum MCU dimensions.
    - Calculates the number of effective pixels for each component and allocates memory for the raw data, handling potential memory allocation failures.
    - If the image is progressive, allocates additional memory for coefficient data.
- **Output**: Returns 1 on success, or an error code if any validation fails or memory allocation issues occur.
- **Functions called**:
    - [`stbi__get16be`](#stbi__get16be)
    - [`stbi__err`](#stbi__err)
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__mad3sizes_valid`](#stbi__mad3sizes_valid)
    - [`stbi__malloc_mad2`](#stbi__malloc_mad2)
    - [`stbi__free_jpeg_components`](#stbi__free_jpeg_components)
    - [`stbi__malloc_mad3`](#stbi__malloc_mad3)


---
### stbi\_\_decode\_jpeg\_header<!-- {{#callable:stbi__decode_jpeg_header}} -->
Decodes the JPEG header from a given JPEG structure and handles various markers.
- **Inputs**:
    - `z`: A pointer to a `stbi__jpeg` structure that contains the JPEG data and state.
    - `scan`: An integer indicating the type of scan to perform, which can affect how the header is processed.
- **Control Flow**:
    - Initializes the `jfif` and `app14_color_transform` fields of the `stbi__jpeg` structure.
    - Retrieves the first marker using [`stbi__get_marker`](#stbi__get_marker) and checks if it is the Start of Image (SOI) marker.
    - If the marker is not SOI, an error is returned indicating a corrupt JPEG.
    - If the scan type is `STBI__SCAN_type`, the function returns 1 immediately.
    - Enters a loop to retrieve markers until a Start of Frame (SOF) marker is found.
    - Processes each marker using [`stbi__process_marker`](#stbi__process_marker), returning 0 if any processing fails.
    - Handles potential padding by checking for end-of-file conditions.
    - Once a SOF marker is found, it checks if the JPEG is progressive and processes the frame header.
    - Returns 1 if the header is successfully decoded.
- **Output**: Returns 1 on successful decoding of the JPEG header, or 0 if an error occurs during processing.
- **Functions called**:
    - [`stbi__get_marker`](#stbi__get_marker)
    - [`stbi__err`](#stbi__err)
    - [`stbi__process_marker`](#stbi__process_marker)
    - [`stbi__at_eof`](#stbi__at_eof)
    - [`stbi__process_frame_header`](#stbi__process_frame_header)


---
### stbi\_\_skip\_jpeg\_junk\_at\_end<!-- {{#callable:stbi__skip_jpeg_junk_at_end}} -->
The `stbi__skip_jpeg_junk_at_end` function skips over junk data at the end of a JPEG stream until it encounters a valid marker or reaches the end of the stream.
- **Inputs**:
    - `j`: A pointer to a `stbi__jpeg` structure that contains the JPEG stream and its associated state.
- **Control Flow**:
    - The function enters a loop that continues until the end of the JPEG stream is reached.
    - Within the loop, it reads a byte from the stream.
    - If the byte is `0xff`, it indicates a potential marker, and the function continues to read subsequent bytes.
    - If a byte other than `0x00` or `0xff` is encountered, it is treated as a valid marker, and the function returns this byte.
    - If the end of the stream is reached while looking for a marker, the function returns `STBI__MARKER_none`.
- **Output**: The function returns either a valid marker byte if found, or `STBI__MARKER_none` if no valid marker is detected before reaching the end of the stream.
- **Functions called**:
    - [`stbi__at_eof`](#stbi__at_eof)
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_decode\_jpeg\_image<!-- {{#callable:stbi__decode_jpeg_image}} -->
Decodes a JPEG image by processing its header and scan data.
- **Inputs**:
    - `j`: A pointer to a `stbi__jpeg` structure that contains the JPEG image data and state.
- **Control Flow**:
    - Initializes the raw data and coefficients for each of the four image components to NULL.
    - Sets the restart interval to 0.
    - Calls [`stbi__decode_jpeg_header`](#stbi__decode_jpeg_header) to read the JPEG header; if it fails, returns 0.
    - Enters a loop to process markers until the End of Image (EOI) marker is encountered.
    - If a Start of Scan (SOS) marker is found, processes the scan header and entropy-coded data.
    - Handles the DNL marker by checking the length and height values.
    - Processes other markers using [`stbi__process_marker`](#stbi__process_marker).
    - If the image is progressive, calls [`stbi__jpeg_finish`](#stbi__jpeg_finish) to finalize the decoding.
- **Output**: Returns 1 on successful decoding of the JPEG image, or 0 if an error occurs during processing.
- **Functions called**:
    - [`stbi__decode_jpeg_header`](#stbi__decode_jpeg_header)
    - [`stbi__get_marker`](#stbi__get_marker)
    - [`stbi__process_scan_header`](#stbi__process_scan_header)
    - [`stbi__parse_entropy_coded_data`](#stbi__parse_entropy_coded_data)
    - [`stbi__skip_jpeg_junk_at_end`](#stbi__skip_jpeg_junk_at_end)
    - [`stbi__get16be`](#stbi__get16be)
    - [`stbi__err`](#stbi__err)
    - [`stbi__process_marker`](#stbi__process_marker)
    - [`stbi__jpeg_finish`](#stbi__jpeg_finish)


---
### resample\_row\_1<!-- {{#callable:resample_row_1}} -->
The `resample_row_1` function returns the `in_near` input without performing any resampling.
- **Inputs**:
    - `out`: A pointer to the output buffer where the resampled data would be stored, but is unused in this function.
    - `in_near`: A pointer to the input buffer containing the near pixel data that will be returned.
    - `in_far`: A pointer to the input buffer containing the far pixel data, which is unused in this function.
    - `w`: An integer representing the width of the image, which is not utilized in this function.
    - `hs`: An integer representing the height scale factor, which is also not utilized in this function.
- **Control Flow**:
    - The function begins by marking the `out`, `in_far`, `w`, and `hs` parameters as unused, indicating that they are not needed for the function's operation.
    - The function directly returns the `in_near` pointer without any modifications or processing.
- **Output**: The output is a pointer to the `in_near` input buffer, effectively returning the same data without any changes.


---
### stbi\_\_resample\_row\_v\_2<!-- {{#callable:stbi__resample_row_v_2}} -->
The `stbi__resample_row_v_2` function generates a vertically resampled row of pixel data by averaging input samples.
- **Inputs**:
    - `out`: A pointer to the output array where the resampled pixel data will be stored.
    - `in_near`: A pointer to the input array containing pixel data from the near source.
    - `in_far`: A pointer to the input array containing pixel data from the far source.
    - `w`: An integer representing the width of the input arrays.
    - `hs`: An integer that is not used in the function, likely intended for future use or compatibility.
- **Control Flow**:
    - The function starts by declaring a loop variable `i` and marks `hs` as unused.
    - A for loop iterates from 0 to `w`, processing each pixel in the input arrays.
    - Within the loop, each output pixel is calculated as a weighted average of the corresponding pixels from `in_near` and `in_far` using the formula: `out[i] = stbi__div4(3*in_near[i] + in_far[i] + 2)`.
    - The calculated value is stored in the `out` array.
- **Output**: The function returns a pointer to the `out` array containing the resampled pixel data.


---
### stbi\_\_resample\_row\_h\_2<!-- {{#callable:stbi__resample_row_h_2}} -->
The `stbi__resample_row_h_2` function performs horizontal resampling of an input image row, generating two output samples for each input sample.
- **Inputs**:
    - `out`: A pointer to the output array where the resampled values will be stored.
    - `in_near`: A pointer to the input array containing the original pixel values to be resampled.
    - `in_far`: A pointer to an unused input array, included for compatibility but not utilized in the function.
    - `w`: An integer representing the width of the input array, indicating the number of samples.
    - `hs`: An integer representing the height scale, included for compatibility but not utilized in the function.
- **Control Flow**:
    - The function first checks if the width `w` is 1; if so, it copies the single input sample to both output positions and returns.
    - If `w` is greater than 1, it initializes the first two output samples based on the first input sample.
    - It then enters a loop that iterates from the second input sample to the second-to-last sample, calculating two output samples for each input sample using a weighted average of neighboring input samples.
    - After the loop, it sets the last two output samples based on the last input sample and the second-to-last input sample.
- **Output**: The function returns a pointer to the output array containing the resampled pixel values, with each input sample producing two output samples.


---
### stbi\_\_resample\_row\_hv\_2<!-- {{#callable:stbi__resample_row_hv_2}} -->
The `stbi__resample_row_hv_2` function performs horizontal and vertical resampling of an input image row to generate a higher resolution output.
- **Inputs**:
    - `out`: A pointer to the output array where the resampled pixel values will be stored.
    - `in_near`: A pointer to the input array containing pixel values from the near row.
    - `in_far`: A pointer to the input array containing pixel values from the far row.
    - `w`: An integer representing the width of the input row.
    - `hs`: An integer that is not used in the function but is included for compatibility.
- **Control Flow**:
    - If the width `w` is 1, the function calculates a single output pixel value based on the input values and returns immediately.
    - For wider inputs, the function initializes a variable `t1` to compute the first output pixel and enters a loop to process each input pixel.
    - Within the loop, it updates the previous pixel value `t0`, computes the new pixel value `t1`, and generates two output pixels for each input pixel using weighted averages.
    - Finally, it computes the last output pixel after the loop and returns the output array.
- **Output**: The function returns a pointer to the output array containing the resampled pixel values, which are twice the width of the input.


---
### stbi\_\_resample\_row\_hv\_2\_simd<!-- {{#callable:stbi__resample_row_hv_2_simd}} -->
The `stbi__resample_row_hv_2_simd` function performs high-quality resampling of an input image row using SIMD instructions.
- **Inputs**:
    - `out`: A pointer to the output buffer where the resampled pixel data will be stored.
    - `in_near`: A pointer to the input buffer containing pixel data from the 'near' row.
    - `in_far`: A pointer to the input buffer containing pixel data from the 'far' row.
    - `w`: An integer representing the width of the input row.
    - `hs`: An integer that is not used in the function but is included for compatibility.
- **Control Flow**:
    - If the width `w` is 1, the function calculates a single output pixel based on the input values and returns immediately.
    - For widths greater than 1, the function processes groups of 8 pixels using SIMD instructions for performance.
    - It performs vertical filtering on the input pixel values to compute the current row's pixel values.
    - The function then applies horizontal filtering using the current, previous, and next pixel values to compute the final output pixels.
    - After processing the main block of pixels, it handles any remaining pixels individually.
    - Finally, it sets the last pixel in the output buffer and returns the output pointer.
- **Output**: The function returns a pointer to the output buffer containing the resampled pixel data.


---
### stbi\_\_resample\_row\_generic<!-- {{#callable:stbi__resample_row_generic}} -->
The `stbi__resample_row_generic` function performs nearest-neighbor resampling of a row of pixel data.
- **Inputs**:
    - `out`: A pointer to the output buffer where the resampled pixel data will be stored.
    - `in_near`: A pointer to the input buffer containing the original pixel data to be resampled.
    - `in_far`: A pointer to an unused input buffer, which is not utilized in this function.
    - `w`: An integer representing the width of the input pixel data.
    - `hs`: An integer representing the height scale factor for resampling.
- **Control Flow**:
    - The function begins by declaring two integer variables, `i` and `j`, for iteration.
    - The `STBI_NOTUSED(in_far)` macro is called to indicate that the `in_far` parameter is intentionally unused.
    - A nested loop iterates over the width (`w`) and height scale factor (`hs`), where the outer loop iterates over each pixel in the width and the inner loop iterates over the height scale factor.
    - Within the inner loop, the output buffer is populated by assigning the value from `in_near[i]` to `out[i*hs+j]`, effectively replicating the pixel value for the specified height scale.
- **Output**: The function returns a pointer to the output buffer `out`, which contains the resampled pixel data.


---
### stbi\_\_YCbCr\_to\_RGB\_row<!-- {{#callable:stbi__YCbCr_to_RGB_row}} -->
Converts YCbCr color values to RGB format for a specified number of pixels.
- **Inputs**:
    - `out`: Pointer to the output buffer where the RGB values will be stored.
    - `y`: Pointer to the Y (luminance) component array.
    - `pcb`: Pointer to the Cb (chrominance blue) component array.
    - `pcr`: Pointer to the Cr (chrominance red) component array.
    - `count`: The number of pixels to convert.
    - `step`: The step size for writing to the output buffer.
- **Control Flow**:
    - Iterates over each pixel for the specified count.
    - Calculates the fixed-point representation of the Y value with rounding.
    - Computes the R, G, and B values using the Y, Cb, and Cr components.
    - Clamps the R, G, and B values to the range [0, 255].
    - Stores the computed RGB values in the output buffer, followed by an alpha value of 255.
- **Output**: The function does not return a value; instead, it writes the converted RGB values directly to the output buffer.


---
### stbi\_\_YCbCr\_to\_RGB\_simd<!-- {{#callable:stbi__YCbCr_to_RGB_simd}} -->
Converts YCbCr color values to RGB format using SIMD optimizations when available.
- **Inputs**:
    - `out`: Pointer to the output buffer where the RGB values will be stored.
    - `y`: Pointer to the input Y channel values.
    - `pcb`: Pointer to the input Cb channel values.
    - `pcr`: Pointer to the input Cr channel values.
    - `count`: The number of pixels to process.
    - `step`: The step size for the output buffer, typically 3 or 4.
- **Control Flow**:
    - Checks if SIMD optimizations are enabled (SSE2 or NEON) and if the step size is 4.
    - If using SSE2, processes 8 pixels at a time using SIMD instructions for loading, transforming, and storing RGB values.
    - If using NEON, similarly processes 8 pixels at a time with ARM-specific SIMD instructions.
    - For remaining pixels (if count is not a multiple of 8), processes each pixel individually using standard calculations.
- **Output**: The function outputs the converted RGB values into the provided output buffer, interleaved with an optional alpha channel.


---
### stbi\_\_setup\_jpeg<!-- {{#callable:stbi__setup_jpeg}} -->
The `stbi__setup_jpeg` function initializes function pointers for JPEG processing based on available SIMD optimizations.
- **Inputs**:
    - `j`: A pointer to a `stbi__jpeg` structure that holds function pointers for JPEG processing.
- **Control Flow**:
    - The function sets default function pointers for IDCT, YCbCr to RGB conversion, and resampling.
    - If `STBI_SSE2` is defined and SSE2 is available, it overrides the default function pointers with SIMD-optimized versions.
    - If `STBI_NEON` is defined, it sets the function pointers to SIMD-optimized versions regardless of SSE2 availability.
- **Output**: The function does not return a value; it modifies the `stbi__jpeg` structure in place to use the appropriate function pointers for JPEG processing.
- **Functions called**:
    - [`stbi__sse2_available`](#stbi__sse2_available)


---
### stbi\_\_cleanup\_jpeg<!-- {{#callable:stbi__cleanup_jpeg}} -->
Cleans up JPEG components associated with a given `stbi__jpeg` structure.
- **Inputs**:
    - `j`: A pointer to an `stbi__jpeg` structure that contains information about the JPEG image and its components.
- **Control Flow**:
    - The function calls [`stbi__free_jpeg_components`](#stbi__free_jpeg_components) to free the JPEG components.
    - It passes the number of image channels (`img_n`) and a zero value as parameters to the cleanup function.
- **Output**: This function does not return a value; it performs cleanup by freeing allocated resources.
- **Functions called**:
    - [`stbi__free_jpeg_components`](#stbi__free_jpeg_components)


---
### stbi\_\_blinn\_8x8<!-- {{#callable:stbi__blinn_8x8}} -->
Calculates a modified average of two input values using a specific formula.
- **Inputs**:
    - `x`: An 8-bit unsigned integer representing the first input value.
    - `y`: An 8-bit unsigned integer representing the second input value.
- **Control Flow**:
    - The function computes the product of `x` and `y`, adds 128 to it, and stores the result in `t`.
    - It then performs a right shift operation on `t` to adjust the value and returns the final result as an 8-bit unsigned integer.
- **Output**: Returns an 8-bit unsigned integer that represents the modified average of the inputs `x` and `y`.


---
### load\_jpeg\_image<!-- {{#callable:load_jpeg_image}} -->
Loads a JPEG image, decodes it, and converts it to a specified color format.
- **Inputs**:
    - `z`: A pointer to a `stbi__jpeg` structure that contains JPEG image data and decoding state.
    - `out_x`: A pointer to an integer where the output image width will be stored.
    - `out_y`: A pointer to an integer where the output image height will be stored.
    - `comp`: A pointer to an integer where the number of color components in the output image will be stored.
    - `req_comp`: An integer specifying the desired number of color components in the output image (1 to 4).
- **Control Flow**:
    - Initializes the number of image components to zero for safety.
    - Validates the `req_comp` input to ensure it is within the acceptable range (0 to 4).
    - Attempts to decode the JPEG image; if it fails, cleans up and returns NULL.
    - Determines the actual number of components to generate based on `req_comp` and the original image's component count.
    - Checks if there are components to decode; if not, cleans up and returns NULL.
    - Allocates memory for line buffers and sets up resampling parameters for each component.
    - Allocates memory for the output image based on the desired number of components and image dimensions.
    - Resamples and converts the image data from YCbCr to the requested color format, handling different cases for RGB and CMYK.
    - Cleans up resources and sets the output dimensions and component count before returning the decoded image.
- **Output**: Returns a pointer to the decoded image data in the specified format, or NULL if an error occurs.
- **Functions called**:
    - [`stbi__decode_jpeg_image`](#stbi__decode_jpeg_image)
    - [`stbi__cleanup_jpeg`](#stbi__cleanup_jpeg)
    - [`stbi__malloc`](#stbi__malloc)
    - [`stbi__malloc_mad3`](#stbi__malloc_mad3)
    - [`stbi__blinn_8x8`](#stbi__blinn_8x8)
    - [`stbi__compute_y`](#stbi__compute_y)


---
### stbi\_\_jpeg\_load<!-- {{#callable:stbi__jpeg_load}} -->
Loads a JPEG image from a given context and returns the pixel data.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the JPEG image data.
    - `x`: A pointer to an integer where the width of the loaded image will be stored.
    - `y`: A pointer to an integer where the height of the loaded image will be stored.
    - `comp`: A pointer to an integer that specifies the number of color components in the output image.
    - `req_comp`: An integer that indicates the desired number of color components to return.
    - `ri`: A pointer to a `stbi__result_info` structure that is not used in this function.
- **Control Flow**:
    - Allocates memory for a `stbi__jpeg` structure and checks for successful allocation.
    - Initializes the allocated `stbi__jpeg` structure to zero.
    - Sets the context pointer in the `stbi__jpeg` structure.
    - Calls [`stbi__setup_jpeg`](#stbi__setup_jpeg) to prepare the JPEG structure for loading.
    - Calls [`load_jpeg_image`](#load_jpeg_image) to actually load the image data and store it in the `result` variable.
    - Frees the allocated `stbi__jpeg` structure.
    - Returns the loaded image data.
- **Output**: Returns a pointer to the loaded image data in memory, or an error message if memory allocation fails.
- **Functions called**:
    - [`stbi__malloc`](#stbi__malloc)
    - [`stbi__setup_jpeg`](#stbi__setup_jpeg)
    - [`load_jpeg_image`](#load_jpeg_image)


---
### stbi\_\_jpeg\_test<!-- {{#callable:stbi__jpeg_test}} -->
The `stbi__jpeg_test` function tests if a JPEG image can be decoded from a given context.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the input data for the JPEG image.
- **Control Flow**:
    - Allocates memory for a `stbi__jpeg` structure and checks for successful allocation.
    - Initializes the allocated `stbi__jpeg` structure to zero.
    - Sets the `s` member of the `stbi__jpeg` structure to the provided context pointer.
    - Calls [`stbi__setup_jpeg`](#stbi__setup_jpeg) to prepare the JPEG structure for decoding.
    - Attempts to decode the JPEG header using [`stbi__decode_jpeg_header`](#stbi__decode_jpeg_header) and stores the result.
    - Rewinds the input context to the beginning.
    - Frees the allocated `stbi__jpeg` structure.
    - Returns the result of the JPEG header decoding.
- **Output**: Returns an integer indicating the success or failure of the JPEG header decoding process.
- **Functions called**:
    - [`stbi__malloc`](#stbi__malloc)
    - [`stbi__err`](#stbi__err)
    - [`stbi__setup_jpeg`](#stbi__setup_jpeg)
    - [`stbi__decode_jpeg_header`](#stbi__decode_jpeg_header)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_jpeg\_info\_raw<!-- {{#callable:stbi__jpeg_info_raw}} -->
Extracts JPEG image dimensions and component count from a `stbi__jpeg` structure.
- **Inputs**:
    - `j`: A pointer to a `stbi__jpeg` structure containing JPEG image data.
    - `x`: A pointer to an integer where the image width will be stored.
    - `y`: A pointer to an integer where the image height will be stored.
    - `comp`: A pointer to an integer where the number of color components will be stored.
- **Control Flow**:
    - Calls [`stbi__decode_jpeg_header`](#stbi__decode_jpeg_header) to decode the JPEG header and check for success.
    - If the header decoding fails, rewinds the stream and returns 0.
    - If the header decoding is successful, assigns the image width to `*x`, height to `*y`, and sets `*comp` to 3 if there are 3 or more components, otherwise sets it to 1.
    - Returns 1 to indicate successful extraction of image information.
- **Output**: Returns 1 if the JPEG header is successfully decoded and image information is extracted; otherwise, returns 0.
- **Functions called**:
    - [`stbi__decode_jpeg_header`](#stbi__decode_jpeg_header)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_jpeg\_info<!-- {{#callable:stbi__jpeg_info}} -->
The `stbi__jpeg_info` function retrieves JPEG image information such as dimensions and component count.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the JPEG image data.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored.
- **Control Flow**:
    - Allocates memory for a `stbi__jpeg` structure and checks for successful allocation.
    - If memory allocation fails, it returns an error message indicating out of memory.
    - Initializes the allocated `stbi__jpeg` structure to zero.
    - Sets the `s` member of the `stbi__jpeg` structure to the provided context pointer `s`.
    - Calls the [`stbi__jpeg_info_raw`](#stbi__jpeg_info_raw) function to retrieve the image information, passing the `stbi__jpeg` structure and pointers for width, height, and component count.
    - Frees the allocated memory for the `stbi__jpeg` structure after retrieving the information.
    - Returns the result obtained from the [`stbi__jpeg_info_raw`](#stbi__jpeg_info_raw) function.
- **Output**: Returns an integer indicating success or failure of the operation, with the image dimensions and component count stored in the provided pointers.
- **Functions called**:
    - [`stbi__malloc`](#stbi__malloc)
    - [`stbi__err`](#stbi__err)
    - [`stbi__jpeg_info_raw`](#stbi__jpeg_info_raw)


---
### stbi\_\_bitreverse16<!-- {{#callable:stbi__bitreverse16}} -->
Reverses the bits of a 16-bit integer.
- **Inputs**:
    - `n`: A 16-bit integer whose bits are to be reversed.
- **Control Flow**:
    - The function first swaps the bits in pairs (1-bit groups) using a mask of `0xAAAA` and `0x5555`.
    - Next, it swaps the bits in nibbles (2-bit groups) using a mask of `0xCCCC` and `0x3333`.
    - Then, it swaps the bits in bytes (4-bit groups) using a mask of `0xF0F0` and `0x0F0F`.
    - Finally, it swaps the high byte with the low byte using a mask of `0xFF00` and `0x00FF`.
- **Output**: Returns the 16-bit integer with its bits reversed.


---
### stbi\_\_bit\_reverse<!-- {{#callable:stbi__bit_reverse}} -->
Reverses the bits of an integer value up to 16 bits.
- **Inputs**:
    - `v`: The integer value whose bits are to be reversed.
    - `bits`: The number of bits to consider for the reversal, must be less than or equal to 16.
- **Control Flow**:
    - The function asserts that the number of bits is less than or equal to 16 using `STBI_ASSERT`.
    - It calls the helper function [`stbi__bitreverse16`](#stbi__bitreverse16) to reverse the bits of the integer `v`.
    - The result of the bit reversal is then right-shifted by (16 - bits) to discard the unnecessary higher bits.
- **Output**: Returns the bit-reversed integer value, adjusted for the specified number of bits.
- **Functions called**:
    - [`stbi__bitreverse16`](#stbi__bitreverse16)


---
### stbi\_\_zbuild\_huffman<!-- {{#callable:stbi__zbuild_huffman}} -->
The `stbi__zbuild_huffman` function constructs a Huffman coding table based on the provided size list.
- **Inputs**:
    - `z`: A pointer to a `stbi__zhuffman` structure that will hold the generated Huffman codes.
    - `sizelist`: An array of `stbi_uc` values representing the sizes of the symbols to be encoded.
    - `num`: An integer representing the number of symbols in the `sizelist`.
- **Control Flow**:
    - Initializes the `sizes` array to count occurrences of each symbol size and clears the `fast` array in the `z` structure.
    - Counts the number of symbols for each size and checks for validity against the DEFLATE specification.
    - Generates the `next_code` and `maxcode` arrays for each symbol size, ensuring that the generated codes do not exceed the allowed limits.
    - Populates the `size` and `value` arrays in the `z` structure with the corresponding sizes and values for each symbol.
    - If the symbol size is less than or equal to `STBI__ZFAST_BITS`, it fills the `fast` lookup table with the appropriate values.
- **Output**: Returns 1 on success, indicating that the Huffman table has been successfully built.
- **Functions called**:
    - [`stbi__err`](#stbi__err)
    - [`stbi__bit_reverse`](#stbi__bit_reverse)


---
### stbi\_\_zeof<!-- {{#callable:stbi__zeof}} -->
Checks if the end of the buffer has been reached in a decompression context.
- **Inputs**:
    - `z`: A pointer to a `stbi__zbuf` structure that contains the current position and end of the buffer.
- **Control Flow**:
    - The function compares the current position of the buffer (`z->zbuffer`) with the end of the buffer (`z->zbuffer_end`).
    - If the current position is greater than or equal to the end position, it indicates that the end of the buffer has been reached.
- **Output**: Returns an integer value: 1 if the end of the buffer is reached, otherwise 0.


---
### stbi\_\_zget8<!-- {{#callable:stbi__zget8}} -->
Retrieves a single byte from the `zbuffer` of a `stbi__zbuf` structure, returning 0 if the end of the buffer is reached.
- **Inputs**:
    - `z`: A pointer to a `stbi__zbuf` structure that contains the compressed data buffer.
- **Control Flow**:
    - Checks if the end of the buffer is reached using the [`stbi__zeof`](#stbi__zeof) function.
    - If the end of the buffer is reached, returns 0.
    - If not at the end, retrieves the current byte from `z->zbuffer` and increments the buffer pointer.
- **Output**: Returns a single byte from the `zbuffer`, or 0 if the end of the buffer has been reached.
- **Functions called**:
    - [`stbi__zeof`](#stbi__zeof)


---
### stbi\_\_fill\_bits<!-- {{#callable:stbi__fill_bits}} -->
The `stbi__fill_bits` function fills the bit buffer for a given `stbi__zbuf` structure by reading bytes from the input stream.
- **Inputs**:
    - `z`: A pointer to a `stbi__zbuf` structure that contains the current state of the bit buffer and the input stream.
- **Control Flow**:
    - The function enters a loop that continues as long as the number of bits in the buffer is less than or equal to 24.
    - Inside the loop, it checks if the `code_buffer` has reached or exceeded the maximum value for the current number of bits; if so, it sets the `zbuffer` to `zbuffer_end` to indicate EOF and exits the function.
    - If the buffer is not full, it reads a byte from the input stream using [`stbi__zget8`](#stbi__zget8), shifts it left by the current number of bits, and adds it to the `code_buffer` while also increasing the `num_bits` by 8.
- **Output**: The function does not return a value; it modifies the `stbi__zbuf` structure in place, updating the `code_buffer` and `num_bits` fields.
- **Functions called**:
    - [`stbi__zget8`](#stbi__zget8)


---
### stbi\_\_zreceive<!-- {{#callable:stbi__zreceive}} -->
The `stbi__zreceive` function retrieves a specified number of bits from a bit buffer.
- **Inputs**:
    - `z`: A pointer to a `stbi__zbuf` structure that contains the bit buffer and related metadata.
    - `n`: An integer specifying the number of bits to retrieve from the buffer.
- **Control Flow**:
    - The function first checks if the number of bits available in the buffer (`z->num_bits`) is less than the requested number of bits (`n`).
    - If there are not enough bits, it calls the [`stbi__fill_bits`](#stbi__fill_bits) function to refill the buffer.
    - It then retrieves the specified number of bits from the `code_buffer` using a bitwise AND operation with a mask created by shifting 1 left by `n` and subtracting 1.
    - The `code_buffer` is then right-shifted by `n` to remove the bits that have been read.
    - Finally, the number of bits available (`num_bits`) is decremented by `n` and the retrieved bits are returned.
- **Output**: The function returns an unsigned integer representing the bits retrieved from the buffer.
- **Functions called**:
    - [`stbi__fill_bits`](#stbi__fill_bits)


---
### stbi\_\_zhuffman\_decode\_slowpath<!-- {{#callable:stbi__zhuffman_decode_slowpath}} -->
Decodes a Huffman code using a slow path method when fast decoding is not possible.
- **Inputs**:
    - `a`: A pointer to a `stbi__zbuf` structure that contains the current state of the bit buffer.
    - `z`: A pointer to a `stbi__zhuffman` structure that holds the Huffman coding information including maximum codes and symbol values.
- **Control Flow**:
    - The function begins by reversing the bits in the `code_buffer` of the `stbi__zbuf` structure to prepare for decoding.
    - It then enters a loop to determine the size of the Huffman code by comparing the reversed code against the maximum codes in the `stbi__zhuffman` structure.
    - If the size exceeds 16, it returns -1 indicating an invalid code.
    - The function calculates the symbol index based on the size of the code and checks for data corruption by validating the symbol size.
    - If all checks pass, it updates the bit buffer and returns the decoded symbol value.
- **Output**: Returns the decoded symbol value from the Huffman table, or -1 if an error occurs during decoding.
- **Functions called**:
    - [`stbi__bit_reverse`](#stbi__bit_reverse)


---
### stbi\_\_zhuffman\_decode<!-- {{#callable:stbi__zhuffman_decode}} -->
`stbi__zhuffman_decode` decodes a Huffman-encoded value from a bit stream.
- **Inputs**:
    - `a`: A pointer to a `stbi__zbuf` structure that contains the current state of the bit stream.
    - `z`: A pointer to a `stbi__zhuffman` structure that holds the Huffman decoding table.
- **Control Flow**:
    - The function first checks if the number of bits in the buffer is less than 16.
    - If it is, it checks for end-of-file (EOF) conditions; if EOF is encountered for the first time, it adds 16 implicit zero bits to the buffer.
    - If EOF is encountered again without consuming the extra bits, it returns -1 indicating an error.
    - If not at EOF, it calls [`stbi__fill_bits`](#stbi__fill_bits) to refill the bit buffer.
    - The function then attempts to decode a value using the fast lookup table in the `z` structure.
    - If a valid value is found, it updates the bit buffer and returns the decoded value.
    - If no valid value is found, it calls [`stbi__zhuffman_decode_slowpath`](#stbi__zhuffman_decode_slowpath) to handle the decoding in a slower manner.
- **Output**: Returns the decoded Huffman value as an integer, or -1 if an error occurs due to premature EOF.
- **Functions called**:
    - [`stbi__zeof`](#stbi__zeof)
    - [`stbi__fill_bits`](#stbi__fill_bits)
    - [`stbi__zhuffman_decode_slowpath`](#stbi__zhuffman_decode_slowpath)


---
### stbi\_\_zexpand<!-- {{#callable:stbi__zexpand}} -->
The `stbi__zexpand` function expands the output buffer for decompressed data in a PNG image.
- **Inputs**:
    - `z`: A pointer to a `stbi__zbuf` structure that contains the current state of the decompression process.
    - `zout`: A pointer to the character array that will hold the expanded output data.
    - `n`: An integer representing the number of bytes to allocate in the output buffer.
- **Control Flow**:
    - The function first checks if the output buffer is expandable; if not, it returns an error.
    - It calculates the current position and the limits of the output buffer.
    - If the requested expansion size exceeds the current limit, it doubles the limit until it can accommodate the new size or reaches a maximum threshold.
    - The function attempts to reallocate the output buffer using `STBI_REALLOC_SIZED` and checks for successful allocation.
    - If successful, it updates the pointers in the `stbi__zbuf` structure to reflect the new buffer state.
- **Output**: The function returns 1 on successful expansion of the output buffer, or an error code if memory allocation fails.
- **Functions called**:
    - [`stbi__err`](#stbi__err)


---
### stbi\_\_parse\_huffman\_block<!-- {{#callable:stbi__parse_huffman_block}} -->
`stbi__parse_huffman_block` decodes a Huffman-encoded block of data and outputs the decompressed bytes.
- **Inputs**:
    - `a`: A pointer to a `stbi__zbuf` structure that contains the compressed data and buffers for output.
- **Control Flow**:
    - The function enters an infinite loop to continuously decode Huffman codes until a termination condition is met.
    - It first attempts to decode a Huffman code using [`stbi__zhuffman_decode`](#stbi__zhuffman_decode) and checks if the result is a literal byte, a special end code, or a length-distance pair.
    - If a literal byte is decoded, it checks if there is space in the output buffer and writes the byte to the output.
    - If the end code (256) is encountered, it checks for malformed input and returns success.
    - For length-distance pairs, it retrieves the length and distance values, checks for validity, and copies the corresponding bytes from the output buffer based on the distance.
    - The function handles buffer expansion if the output buffer is full, ensuring that there is enough space for the decoded data.
- **Output**: Returns 1 on successful decoding of a block, 0 on failure, or an error code if the input data is corrupt.
- **Functions called**:
    - [`stbi__zhuffman_decode`](#stbi__zhuffman_decode)
    - [`stbi__err`](#stbi__err)
    - [`stbi__zexpand`](#stbi__zexpand)
    - [`stbi__zreceive`](#stbi__zreceive)


---
### stbi\_\_compute\_huffman\_codes<!-- {{#callable:stbi__compute_huffman_codes}} -->
Computes Huffman codes for a given `stbi__zbuf` structure based on the provided lengths and builds the corresponding Huffman trees.
- **Inputs**:
    - `a`: A pointer to a `stbi__zbuf` structure that contains the compressed data and state for decoding.
- **Control Flow**:
    - Initializes a static array `length_dezigzag` to map lengths for Huffman code lengths.
    - Receives the number of literal codes (`hlit`), distance codes (`hdist`), and code length symbols (`hclen`) from the input buffer.
    - Initializes an array `codelength_sizes` to store the sizes of the code lengths and fills it based on received values.
    - Builds a Huffman tree for code lengths using [`stbi__zbuild_huffman`](#stbi__zbuild_huffman).
    - Enters a loop to decode the actual lengths of the Huffman codes until the total number of codes (`ntot`) is reached.
    - Handles special cases for lengths 16, 17, and 18 which require additional bits to determine the number of times to repeat a previous length.
    - Checks for errors during decoding and ensures the total number of lengths matches `ntot`.
    - Builds the final Huffman trees for lengths and distances using the decoded lengths.
- **Output**: Returns 1 on success, indicating that the Huffman codes were successfully computed and built; returns 0 on failure, indicating an error in the decoding process.
- **Functions called**:
    - [`stbi__zreceive`](#stbi__zreceive)
    - [`stbi__zbuild_huffman`](#stbi__zbuild_huffman)
    - [`stbi__zhuffman_decode`](#stbi__zhuffman_decode)
    - [`stbi__err`](#stbi__err)


---
### stbi\_\_parse\_uncompressed\_block<!-- {{#callable:stbi__parse_uncompressed_block}} -->
Parses an uncompressed block of data from a zlib-compressed stream.
- **Inputs**:
    - `a`: A pointer to a `stbi__zbuf` structure that contains the state of the decompression process, including buffers and bit counts.
- **Control Flow**:
    - Checks if there are leftover bits in `num_bits` and discards them if necessary.
    - Drains the bit-packed data into the `header` array until `num_bits` is exhausted.
    - If `num_bits` is negative after draining, returns an error indicating a corrupt PNG.
    - Fills the remaining `header` bytes using [`stbi__zget8`](#stbi__zget8) until the header is fully populated.
    - Calculates the length of the uncompressed data and its complement, checking for consistency.
    - Validates that the read operation does not exceed the buffer limits.
    - Expands the output buffer if necessary and copies the uncompressed data from `zbuffer` to `zout`.
    - Returns 1 on success or 0 if expansion fails.
- **Output**: Returns 1 if the uncompressed block is successfully parsed and copied; otherwise, returns 0 on failure.
- **Functions called**:
    - [`stbi__zreceive`](#stbi__zreceive)
    - [`stbi__err`](#stbi__err)
    - [`stbi__zget8`](#stbi__zget8)
    - [`stbi__zexpand`](#stbi__zexpand)


---
### stbi\_\_parse\_zlib\_header<!-- {{#callable:stbi__parse_zlib_header}} -->
`stbi__parse_zlib_header` validates the header of a zlib compressed stream.
- **Inputs**:
    - `a`: A pointer to a `stbi__zbuf` structure that contains the zlib compressed data.
- **Control Flow**:
    - The function reads the first byte of the zlib header to determine the compression method and flags.
    - It checks if the end of the buffer is reached, returning an error if so.
    - It validates the header checksum against the zlib specification.
    - It checks for the presence of a preset dictionary, which is not allowed for PNG files.
    - It ensures that the compression method is DEFLATE (cm == 8), returning an error if it is not.
    - If all checks pass, the function returns 1, indicating a valid zlib header.
- **Output**: Returns 1 if the zlib header is valid; otherwise, it returns an error code from [`stbi__err`](#stbi__err) indicating the specific issue.
- **Functions called**:
    - [`stbi__zget8`](#stbi__zget8)
    - [`stbi__zeof`](#stbi__zeof)
    - [`stbi__err`](#stbi__err)


---
### stbi\_\_parse\_zlib<!-- {{#callable:stbi__parse_zlib}} -->
Parses a zlib compressed data stream and extracts the uncompressed data.
- **Inputs**:
    - `a`: A pointer to a `stbi__zbuf` structure that contains the state and data for the zlib parsing.
    - `parse_header`: An integer flag indicating whether to parse the zlib header before processing the data.
- **Control Flow**:
    - If `parse_header` is true, the function first attempts to parse the zlib header using [`stbi__parse_zlib_header`](#stbi__parse_zlib_header). If this fails, it returns 0.
    - The function initializes the bit buffer and other state variables.
    - It enters a loop that continues until the final block of data is reached, indicated by the `final` variable.
    - Within the loop, it reads the `final` and `type` values using [`stbi__zreceive`](#stbi__zreceive).
    - If `type` is 0, it processes an uncompressed block using [`stbi__parse_uncompressed_block`](#stbi__parse_uncompressed_block). If this fails, it returns 0.
    - If `type` is 3, it indicates an error condition, and the function returns 0.
    - If `type` is 1, it builds fixed Huffman codes for lengths and distances using [`stbi__zbuild_huffman`](#stbi__zbuild_huffman). If this fails, it returns 0.
    - For other types, it computes Huffman codes using [`stbi__compute_huffman_codes`](#stbi__compute_huffman_codes) and checks for success.
    - Finally, it parses a Huffman block using [`stbi__parse_huffman_block`](#stbi__parse_huffman_block). If this fails, it returns 0.
- **Output**: Returns 1 if the parsing is successful and all data is processed, or 0 if an error occurs during parsing.
- **Functions called**:
    - [`stbi__parse_zlib_header`](#stbi__parse_zlib_header)
    - [`stbi__zreceive`](#stbi__zreceive)
    - [`stbi__parse_uncompressed_block`](#stbi__parse_uncompressed_block)
    - [`stbi__zbuild_huffman`](#stbi__zbuild_huffman)
    - [`stbi__compute_huffman_codes`](#stbi__compute_huffman_codes)
    - [`stbi__parse_huffman_block`](#stbi__parse_huffman_block)


---
### stbi\_\_do\_zlib<!-- {{#callable:stbi__do_zlib}} -->
The `stbi__do_zlib` function initializes a zlib buffer structure and calls a parser function to process zlib data.
- **Inputs**:
    - `a`: A pointer to a `stbi__zbuf` structure that holds the state and output buffer for zlib decompression.
    - `obuf`: A pointer to the output buffer where decompressed data will be stored.
    - `olen`: An integer representing the size of the output buffer.
    - `exp`: An integer indicating whether the output buffer is expandable.
    - `parse_header`: An integer flag that determines whether to parse the zlib header.
- **Control Flow**:
    - The function sets the starting point, current position, and end point of the output buffer in the `stbi__zbuf` structure.
    - It assigns the expandability flag to the `stbi__zbuf` structure.
    - Finally, it calls the [`stbi__parse_zlib`](#stbi__parse_zlib) function to handle the actual zlib data parsing.
- **Output**: Returns the result of the [`stbi__parse_zlib`](#stbi__parse_zlib) function, which indicates the success or failure of the zlib parsing operation.
- **Functions called**:
    - [`stbi__parse_zlib`](#stbi__parse_zlib)


---
### stbi\_zlib\_decode\_malloc\_guesssize<!-- {{#callable:stbi_zlib_decode_malloc_guesssize}} -->
Decodes a zlib-compressed buffer into a dynamically allocated memory block, estimating the size of the output.
- **Inputs**:
    - `buffer`: A pointer to the zlib-compressed data to be decoded.
    - `len`: The length of the compressed data in bytes.
    - `initial_size`: The initial size of the memory block to allocate for the decoded data.
    - `outlen`: A pointer to an integer where the length of the decoded data will be stored.
- **Control Flow**:
    - Allocates memory for the output buffer using [`stbi__malloc`](#stbi__malloc) with the specified `initial_size`.
    - Checks if the memory allocation was successful; if not, returns NULL.
    - Initializes a `stbi__zbuf` structure with the input buffer and its length.
    - Calls [`stbi__do_zlib`](#stbi__do_zlib) to perform the actual decompression, passing the initialized buffer and allocated memory.
    - If decompression is successful, updates the output length if `outlen` is not NULL and returns the pointer to the decompressed data.
    - If decompression fails, frees the allocated memory and returns NULL.
- **Output**: Returns a pointer to the dynamically allocated memory containing the decompressed data, or NULL if the decompression fails.
- **Functions called**:
    - [`stbi__malloc`](#stbi__malloc)
    - [`stbi__do_zlib`](#stbi__do_zlib)


---
### stbi\_zlib\_decode\_malloc<!-- {{#callable:stbi_zlib_decode_malloc}} -->
Decodes a zlib-compressed buffer into a dynamically allocated memory block.
- **Inputs**:
    - `buffer`: A pointer to the zlib-compressed data to be decoded.
    - `len`: The length of the compressed data in bytes.
    - `outlen`: A pointer to an integer where the length of the decoded data will be stored.
- **Control Flow**:
    - The function calls [`stbi_zlib_decode_malloc_guesssize`](#stbi_zlib_decode_malloc_guesssize) with the provided buffer and length.
    - It uses a default guess size of 16384 bytes for the output buffer.
    - The result of the decoding operation is returned directly.
- **Output**: Returns a pointer to the dynamically allocated memory containing the decoded data, or NULL if decoding fails.
- **Functions called**:
    - [`stbi_zlib_decode_malloc_guesssize`](#stbi_zlib_decode_malloc_guesssize)


---
### stbi\_zlib\_decode\_malloc\_guesssize\_headerflag<!-- {{#callable:stbi_zlib_decode_malloc_guesssize_headerflag}} -->
Decodes a zlib-compressed buffer into a dynamically allocated memory block, estimating the size based on an initial guess.
- **Inputs**:
    - `buffer`: A pointer to the compressed data buffer that needs to be decoded.
    - `len`: The length of the compressed data buffer.
    - `initial_size`: An initial size hint for the memory allocation of the output buffer.
    - `outlen`: A pointer to an integer where the output length will be stored if decoding is successful.
    - `parse_header`: A flag indicating whether to parse the zlib header.
- **Control Flow**:
    - Allocates memory for the output buffer using [`stbi__malloc`](#stbi__malloc) with the specified `initial_size`.
    - Checks if the memory allocation was successful; if not, returns NULL.
    - Initializes a `stbi__zbuf` structure with the input buffer and its boundaries.
    - Calls [`stbi__do_zlib`](#stbi__do_zlib) to perform the actual zlib decoding, passing the initialized buffer and output parameters.
    - If decoding is successful, updates the output length if `outlen` is not NULL and returns the pointer to the decoded data.
    - If decoding fails, frees the allocated output buffer and returns NULL.
- **Output**: Returns a pointer to the dynamically allocated memory containing the decoded data if successful, or NULL if decoding fails.
- **Functions called**:
    - [`stbi__malloc`](#stbi__malloc)
    - [`stbi__do_zlib`](#stbi__do_zlib)


---
### stbi\_zlib\_decode\_buffer<!-- {{#callable:stbi_zlib_decode_buffer}} -->
Decodes a zlib-compressed buffer into an output buffer.
- **Inputs**:
    - `obuffer`: A pointer to the output buffer where the decoded data will be stored.
    - `olen`: The size of the output buffer in bytes.
    - `ibuffer`: A pointer to the input buffer containing the zlib-compressed data.
    - `ilen`: The size of the input buffer in bytes.
- **Control Flow**:
    - Initializes a `stbi__zbuf` structure with the input buffer and its end pointer.
    - Calls the [`stbi__do_zlib`](#stbi__do_zlib) function to perform the actual decompression, passing the initialized structure and output buffer.
    - If decompression is successful, calculates the number of bytes written to the output buffer and returns that value.
    - If decompression fails, returns -1 to indicate an error.
- **Output**: Returns the number of bytes written to the output buffer on success, or -1 if an error occurred during decompression.
- **Functions called**:
    - [`stbi__do_zlib`](#stbi__do_zlib)


---
### stbi\_zlib\_decode\_noheader\_malloc<!-- {{#callable:stbi_zlib_decode_noheader_malloc}} -->
Decodes a zlib-compressed buffer without a header and returns a dynamically allocated buffer containing the decompressed data.
- **Inputs**:
    - `buffer`: A pointer to the zlib-compressed data to be decoded.
    - `len`: The length of the compressed data in bytes.
    - `outlen`: A pointer to an integer where the length of the decompressed data will be stored.
- **Control Flow**:
    - Allocates a buffer of 16384 bytes for the decompressed data using [`stbi__malloc`](#stbi__malloc).
    - Checks if the memory allocation was successful; if not, returns NULL.
    - Initializes a `stbi__zbuf` structure with the input buffer and its end.
    - Calls [`stbi__do_zlib`](#stbi__do_zlib) to perform the decompression; if successful, stores the decompressed length in `outlen` if it is not NULL and returns the start of the decompressed data.
    - If decompression fails, frees the allocated memory and returns NULL.
- **Output**: Returns a pointer to the dynamically allocated buffer containing the decompressed data, or NULL if the decompression fails.
- **Functions called**:
    - [`stbi__malloc`](#stbi__malloc)
    - [`stbi__do_zlib`](#stbi__do_zlib)


---
### stbi\_zlib\_decode\_noheader\_buffer<!-- {{#callable:stbi_zlib_decode_noheader_buffer}} -->
Decodes a zlib-compressed buffer without a header into an output buffer.
- **Inputs**:
    - `obuffer`: A pointer to the output buffer where the decoded data will be stored.
    - `olen`: The size of the output buffer in bytes.
    - `ibuffer`: A pointer to the input buffer containing the zlib-compressed data.
    - `ilen`: The size of the input buffer in bytes.
- **Control Flow**:
    - Initializes a `stbi__zbuf` structure with the input buffer and its end pointer.
    - Calls the [`stbi__do_zlib`](#stbi__do_zlib) function to perform the decompression, passing the initialized structure and output buffer.
    - If decompression is successful, calculates the number of bytes written to the output buffer and returns that value.
    - If decompression fails, returns -1 to indicate an error.
- **Output**: Returns the number of bytes written to the output buffer if successful, or -1 if an error occurred during decompression.
- **Functions called**:
    - [`stbi__do_zlib`](#stbi__do_zlib)


---
### stbi\_\_get\_chunk\_header<!-- {{#callable:stbi__get_chunk_header}} -->
Retrieves the header of a PNG chunk from a given context.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the PNG data stream.
- **Control Flow**:
    - Calls [`stbi__get32be`](#stbi__get32be) to read the first 4 bytes from the context `s` and assigns it to `c.length`, which represents the length of the chunk.
    - Calls [`stbi__get32be`](#stbi__get32be) again to read the next 4 bytes from the context `s` and assigns it to `c.type`, which indicates the type of the chunk.
    - Returns the populated `stbi__pngchunk` structure `c` containing the chunk's length and type.
- **Output**: Returns a `stbi__pngchunk` structure containing the length and type of the PNG chunk.
- **Functions called**:
    - [`stbi__get32be`](#stbi__get32be)


---
### stbi\_\_check\_png\_header<!-- {{#callable:stbi__check_png_header}} -->
Checks if the provided stream is a valid PNG header.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the input stream to be checked.
- **Control Flow**:
    - The function defines a static array `png_sig` containing the expected PNG signature bytes.
    - A loop iterates over each byte of the expected signature.
    - For each byte, it retrieves a byte from the input stream using `stbi__get8(s)` and compares it to the corresponding byte in `png_sig`.
    - If any byte does not match, the function returns an error message indicating that the signature is invalid.
    - If all bytes match, the function returns 1, indicating a valid PNG header.
- **Output**: Returns 1 if the header is valid, or an error message if the header does not match the PNG signature.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__err`](#stbi__err)


---
### stbi\_\_paeth<!-- {{#callable:stbi__paeth}} -->
Calculates the Paeth predictor value based on three input integers.
- **Inputs**:
    - `a`: The first integer input used in the predictor calculation.
    - `b`: The second integer input used in the predictor calculation.
    - `c`: The third integer input used in the predictor calculation.
- **Control Flow**:
    - Calculates a threshold value based on the inputs `a`, `b`, and `c`.
    - Determines the lower and higher values between `a` and `b`.
    - Evaluates conditions to select between `lo`, `hi`, and `c` to compute the predictor value.
    - Returns the final computed predictor value.
- **Output**: Returns an integer that represents the predicted value based on the Paeth algorithm.


---
### stbi\_\_create\_png\_alpha\_expand8<!-- {{#callable:stbi__create_png_alpha_expand8}} -->
Expands an 8-bit PNG image with alpha channel by duplicating pixel data.
- **Inputs**:
    - `dest`: Pointer to the destination buffer where the expanded image data will be stored.
    - `src`: Pointer to the source buffer containing the original image data.
    - `x`: The width of the image, representing the number of pixels.
    - `img_n`: The number of channels in the source image, either 1 for grayscale or 3 for RGB.
- **Control Flow**:
    - The function checks the number of channels in the image (`img_n`).
    - If `img_n` is 1, it processes the image as a grayscale image, expanding each pixel to two bytes in the destination buffer.
    - If `img_n` is 3, it processes the image as an RGB image, expanding each pixel to four bytes in the destination buffer, adding an alpha channel with a value of 255.
    - The processing is done in reverse order to handle cases where the source and destination buffers are the same.
- **Output**: The function does not return a value; it modifies the `dest` buffer in place to contain the expanded image data.


---
### stbi\_\_create\_png\_image\_raw<!-- {{#callable:stbi__create_png_image_raw}} -->
Creates a raw PNG image from the provided raw pixel data with specified parameters.
- **Inputs**:
    - `a`: Pointer to a `stbi__png` structure containing PNG context and output buffer.
    - `raw`: Pointer to the raw pixel data to be processed.
    - `raw_len`: Length of the raw pixel data.
    - `out_n`: Number of output channels (e.g., RGB, RGBA).
    - `x`: Width of the image in pixels.
    - `y`: Height of the image in pixels.
    - `depth`: Bit depth of the image (e.g., 8 or 16 bits per channel).
    - `color`: Color type indicator (e.g., grayscale or color).
- **Control Flow**:
    - Calculate the number of bytes per pixel based on the depth.
    - Allocate memory for the output image buffer and check for allocation success.
    - Validate the dimensions and sizes of the image against the raw data length.
    - Allocate a buffer for filtering scan lines.
    - Iterate over each row of the image, applying the specified filter type to the raw data.
    - Handle different filtering methods (none, sub, up, avg, paeth) based on the filter type.
    - Expand the pixel data based on the bit depth and output channel requirements.
    - Free the filter buffer and return success or failure based on the processing results.
- **Output**: Returns 1 on success, indicating the image was created successfully, or 0 on failure, indicating an error occurred during processing.
- **Functions called**:
    - [`stbi__malloc_mad3`](#stbi__malloc_mad3)
    - [`stbi__err`](#stbi__err)
    - [`stbi__mad3sizes_valid`](#stbi__mad3sizes_valid)
    - [`stbi__mad2sizes_valid`](#stbi__mad2sizes_valid)
    - [`stbi__malloc_mad2`](#stbi__malloc_mad2)
    - [`stbi__paeth`](#stbi__paeth)
    - [`stbi__create_png_alpha_expand8`](#stbi__create_png_alpha_expand8)


---
### stbi\_\_create\_png\_image<!-- {{#callable:stbi__create_png_image}} -->
Creates a PNG image from raw image data, handling interlacing if specified.
- **Inputs**:
    - `a`: A pointer to a `stbi__png` structure that holds PNG image information.
    - `image_data`: A pointer to the raw image data to be processed.
    - `image_data_len`: The length of the raw image data.
    - `out_n`: The number of output channels (e.g., RGB, RGBA).
    - `depth`: The bit depth of the image (1, 2, 4, 8, or 16 bits per channel).
    - `color`: The color type of the image (e.g., grayscale, RGB, palette).
    - `interlaced`: A flag indicating whether the image is interlaced.
- **Control Flow**:
    - Calculates the number of bytes per pixel based on the depth and the number of output channels.
    - If the image is not interlaced, it directly calls [`stbi__create_png_image_raw`](#stbi__create_png_image_raw) to create the image.
    - If the image is interlaced, it allocates memory for the final image and checks for memory allocation failure.
    - Iterates through the 7 passes of the interlaced image, calculating the dimensions for each pass.
    - For each pass, it calls [`stbi__create_png_image_raw`](#stbi__create_png_image_raw) to process the image data and checks for success.
    - Copies the processed image data into the final image buffer according to the interlacing scheme.
    - Frees the temporary output buffer and updates the image data pointer and length.
- **Output**: Returns 1 on success, indicating the PNG image was created successfully, or 0 on failure.
- **Functions called**:
    - [`stbi__create_png_image_raw`](#stbi__create_png_image_raw)
    - [`stbi__malloc_mad3`](#stbi__malloc_mad3)
    - [`stbi__err`](#stbi__err)


---
### stbi\_\_compute\_transparency<!-- {{#callable:stbi__compute_transparency}} -->
The `stbi__compute_transparency` function computes transparency values for an image based on specified color values.
- **Inputs**:
    - `z`: A pointer to a `stbi__png` structure containing image data and context.
    - `tc`: An array of three `stbi_uc` values representing the target color for transparency.
    - `out_n`: An integer indicating the number of channels in the output (2 for grayscale with alpha, 4 for RGBA).
- **Control Flow**:
    - The function asserts that `out_n` is either 2 or 4, ensuring valid output channel configurations.
    - It calculates the total number of pixels in the image using `s->img_x` and `s->img_y`.
    - If `out_n` is 2, it iterates through each pixel, setting the alpha channel to 0 if the color matches `tc[0]`, otherwise setting it to 255.
    - If `out_n` is 4, it checks each pixel's RGB values against the target color `tc`, setting the alpha channel to 0 for matching pixels.
- **Output**: The function returns 1 to indicate successful completion of the transparency computation.


---
### stbi\_\_compute\_transparency16<!-- {{#callable:stbi__compute_transparency16}} -->
Computes transparency for a PNG image based on color values.
- **Inputs**:
    - `z`: A pointer to a `stbi__png` structure containing image data and context.
    - `tc`: An array of three `stbi__uint16` values representing the target color for transparency.
    - `out_n`: An integer indicating the number of channels in the output (2 for grayscale, 4 for RGBA).
- **Control Flow**:
    - The function asserts that `out_n` is either 2 or 4.
    - If `out_n` is 2, it iterates over each pixel, setting the second channel to 0 if the first channel matches the target color, otherwise setting it to 65535.
    - If `out_n` is 4, it checks if the first three channels match the target color and sets the fourth channel to 0 if they do.
    - The loop continues until all pixels are processed.
- **Output**: Returns 1 to indicate successful computation of transparency.


---
### stbi\_\_expand\_png\_palette<!-- {{#callable:stbi__expand_png_palette}} -->
The `stbi__expand_png_palette` function expands a PNG image palette into a full RGBA or RGB format based on the original indexed image.
- **Inputs**:
    - `a`: A pointer to a `stbi__png` structure containing image metadata and output buffer.
    - `palette`: A pointer to an array of color values representing the palette.
    - `len`: An integer representing the length of the palette, which is unused in this function.
    - `pal_img_n`: An integer indicating the number of channels in the output image (3 for RGB, 4 for RGBA).
- **Control Flow**:
    - Calculates the total number of pixels in the image using the dimensions stored in `a->s`.
    - Allocates memory for the output image based on the pixel count and the number of channels specified by `pal_img_n`.
    - Checks if memory allocation was successful; if not, returns an error.
    - Iterates over each pixel in the original image, using the original indexed values to index into the palette and fill the output buffer with the corresponding color values.
    - If `pal_img_n` is 3, it writes 3 bytes (RGB) per pixel; if 4, it writes 4 bytes (RGBA) per pixel.
    - Frees the original output buffer in `a->out` and updates it to point to the newly allocated output buffer.
    - The unused `len` parameter is marked as unused to avoid compiler warnings.
- **Output**: Returns 1 on success, indicating that the palette expansion was completed successfully.
- **Functions called**:
    - [`stbi__malloc_mad2`](#stbi__malloc_mad2)
    - [`stbi__err`](#stbi__err)


---
### stbi\_set\_unpremultiply\_on\_load<!-- {{#callable:stbi_set_unpremultiply_on_load}} -->
Sets a global flag to determine whether to unpremultiply pixel colors on load.
- **Inputs**:
    - `flag_true_if_should_unpremultiply`: An integer flag where a non-zero value indicates that unpremultiplication should occur when loading images.
- **Control Flow**:
    - The function directly assigns the input flag to a global variable `stbi__unpremultiply_on_load_global`.
    - No conditional logic or loops are present; the function performs a single operation.
- **Output**: The function does not return a value; it modifies a global state that affects subsequent image loading operations.


---
### stbi\_convert\_iphone\_png\_to\_rgb<!-- {{#callable:stbi_convert_iphone_png_to_rgb}} -->
Sets a global flag to indicate whether iPhone PNG images should be converted to RGB format.
- **Inputs**:
    - `flag_true_if_should_convert`: An integer flag where a non-zero value indicates that iPhone PNG images should be converted to RGB format.
- **Control Flow**:
    - The function assigns the input flag value to a global variable `stbi__de_iphone_flag_global`.
    - No conditional logic or loops are present; the function performs a direct assignment.
- **Output**: The function does not return a value; it modifies a global state based on the input flag.


---
### stbi\_set\_unpremultiply\_on\_load\_thread<!-- {{#callable:stbi_set_unpremultiply_on_load_thread}} -->
Sets a flag to determine whether to unpremultiply pixel colors during image loading in a separate thread.
- **Inputs**:
    - `flag_true_if_should_unpremultiply`: An integer flag where a non-zero value indicates that pixel colors should be unpremultiplied during loading.
- **Control Flow**:
    - The function assigns the value of `flag_true_if_should_unpremultiply` to the global variable `stbi__unpremultiply_on_load_local`.
    - It then sets the global variable `stbi__unpremultiply_on_load_set` to 1, indicating that the unpremultiply setting has been configured.
- **Output**: The function does not return a value; it modifies global state variables to affect future image loading behavior.


---
### stbi\_convert\_iphone\_png\_to\_rgb\_thread<!-- {{#callable:stbi_convert_iphone_png_to_rgb_thread}} -->
Sets flags to indicate whether to convert iPhone PNG images to RGB format.
- **Inputs**:
    - `flag_true_if_should_convert`: An integer flag that indicates whether the conversion should be performed (non-zero for true, zero for false).
- **Control Flow**:
    - The function assigns the value of `flag_true_if_should_convert` to the global variable `stbi__de_iphone_flag_local`.
    - It sets the global variable `stbi__de_iphone_flag_set` to 1, indicating that the conversion flag has been set.
- **Output**: The function does not return a value; it modifies global state to control the behavior of image processing functions.


---
### stbi\_\_de\_iphone<!-- {{#callable:stbi__de_iphone}} -->
The `stbi__de_iphone` function converts pixel data from BGR format to RGB format, optionally unpremultiplying alpha values.
- **Inputs**:
    - `z`: A pointer to a `stbi__png` structure that contains image data and context.
- **Control Flow**:
    - The function retrieves the image context from the `stbi__png` structure.
    - It calculates the total number of pixels based on the image dimensions.
    - If the output format is 3 channels (BGR), it swaps the first and third channels to convert to RGB.
    - If the output format is 4 channels (BGRA), it checks if unpremultiplication is enabled.
    - If unpremultiplication is enabled, it adjusts the RGB values based on the alpha channel before converting from BGR to RGB.
    - If unpremultiplication is not enabled, it simply swaps the channels from BGR to RGB.
- **Output**: The function modifies the pixel data in place, converting it from BGR to RGB format, with optional unpremultiplication of alpha values for 4-channel images.


---
### stbi\_\_parse\_png\_file<!-- {{#callable:stbi__parse_png_file}} -->
Parses a PNG file and extracts image data based on the PNG specification.
- **Inputs**:
    - `z`: A pointer to a `stbi__png` structure that holds the state and data for the PNG being parsed.
    - `scan`: An integer indicating the scan mode, which determines the stage of parsing (e.g., header only or full load).
    - `req_comp`: An integer specifying the number of components requested in the output image.
- **Control Flow**:
    - Checks the PNG header for validity; if invalid, returns 0.
    - If the scan mode is `STBI__SCAN_type`, it returns 1 immediately.
    - Enters an infinite loop to process chunks until the end of the PNG file is reached.
    - Handles different chunk types (`IHDR`, `PLTE`, `tRNS`, `IDAT`, `IEND`) with specific logic for each.
    - Validates chunk lengths and types, returning errors for any inconsistencies or corrupt data.
    - Allocates memory for image data as needed, especially for `IDAT` chunks, and reads the image data.
    - On encountering the `IEND` chunk, it processes the image data, applies transparency if necessary, and prepares the final output.
- **Output**: Returns 1 on successful parsing and image data extraction, or 0 on failure, with error messages set in case of issues.
- **Functions called**:
    - [`stbi__check_png_header`](#stbi__check_png_header)
    - [`stbi__get_chunk_header`](#stbi__get_chunk_header)
    - [`stbi__skip`](#stbi__skip)
    - [`stbi__err`](#stbi__err)
    - [`stbi__get32be`](#stbi__get32be)
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__get16be`](#stbi__get16be)
    - [`stbi__getn`](#stbi__getn)
    - [`stbi_zlib_decode_malloc_guesssize_headerflag`](#stbi_zlib_decode_malloc_guesssize_headerflag)
    - [`stbi__create_png_image`](#stbi__create_png_image)
    - [`stbi__compute_transparency16`](#stbi__compute_transparency16)
    - [`stbi__compute_transparency`](#stbi__compute_transparency)
    - [`stbi__de_iphone`](#stbi__de_iphone)
    - [`stbi__expand_png_palette`](#stbi__expand_png_palette)


---
### stbi\_\_do\_png<!-- {{#callable:stbi__do_png}} -->
Processes a PNG image and returns the pixel data based on the requested components.
- **Inputs**:
    - `p`: A pointer to a `stbi__png` structure that contains the PNG image data and metadata.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `n`: A pointer to an integer where the number of components in the image will be stored.
    - `req_comp`: An integer specifying the number of color components requested (1 to 4).
    - `ri`: A pointer to a `stbi__result_info` structure to store information about the result.
- **Control Flow**:
    - Checks if `req_comp` is within the valid range (0 to 4), returning an error if not.
    - Calls [`stbi__parse_png_file`](#stbi__parse_png_file) to parse the PNG file; if successful, it checks the color depth.
    - Sets the `bits_per_channel` in `ri` based on the depth of the image.
    - If `req_comp` is specified and differs from the output components, it converts the image format accordingly.
    - Stores the image dimensions in `x` and `y`, and the number of components in `n` if `n` is not NULL.
    - Frees any allocated memory for image data before returning the result.
- **Output**: Returns a pointer to the pixel data of the image, or NULL if an error occurred during processing.
- **Functions called**:
    - [`stbi__parse_png_file`](#stbi__parse_png_file)
    - [`stbi__convert_format`](#stbi__convert_format)
    - [`stbi__convert_format16`](#stbi__convert_format16)


---
### stbi\_\_png\_load<!-- {{#callable:stbi__png_load}} -->
Loads a PNG image from a given context and returns a pointer to the image data.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the image data to be loaded.
    - `x`: A pointer to an integer where the width of the loaded image will be stored.
    - `y`: A pointer to an integer where the height of the loaded image will be stored.
    - `comp`: A pointer to an integer that will hold the number of color components in the loaded image.
    - `req_comp`: An integer specifying the number of color components requested in the output image.
    - `ri`: A pointer to a `stbi__result_info` structure that will be filled with information about the result of the loading process.
- **Control Flow**:
    - Initializes a `stbi__png` structure with the provided context `s`.
    - Calls the [`stbi__do_png`](#stbi__do_png) function with the initialized `stbi__png` structure and the provided parameters to load the image.
- **Output**: Returns a pointer to the loaded image data, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__do_png`](#stbi__do_png)


---
### stbi\_\_png\_test<!-- {{#callable:stbi__png_test}} -->
Tests if the provided context contains a valid PNG header.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the PNG file context to be tested.
- **Control Flow**:
    - Calls the [`stbi__check_png_header`](#stbi__check_png_header) function to verify if the context contains a valid PNG header.
    - Rewinds the context back to the beginning using [`stbi__rewind`](#stbi__rewind).
    - Returns the result of the header check.
- **Output**: Returns an integer indicating the result of the PNG header check, where a non-zero value indicates a valid PNG header and zero indicates an invalid header.
- **Functions called**:
    - [`stbi__check_png_header`](#stbi__check_png_header)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_png\_info\_raw<!-- {{#callable:stbi__png_info_raw}} -->
The `stbi__png_info_raw` function retrieves the dimensions and color components of a PNG image.
- **Inputs**:
    - `p`: A pointer to a `stbi__png` structure that contains the PNG image data.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components will be stored.
- **Control Flow**:
    - The function first checks if the PNG file can be parsed by calling [`stbi__parse_png_file`](#stbi__parse_png_file) with the header scan option.
    - If parsing fails, it rewinds the input stream and returns 0, indicating failure.
    - If parsing is successful, it assigns the image width, height, and number of components to the provided pointers if they are not NULL.
    - Finally, it returns 1 to indicate success.
- **Output**: The function returns 1 on success, indicating that the image information was successfully retrieved, or 0 on failure.
- **Functions called**:
    - [`stbi__parse_png_file`](#stbi__parse_png_file)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_png\_info<!-- {{#callable:stbi__png_info}} -->
The `stbi__png_info` function retrieves PNG image information such as dimensions and color components.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the PNG image data.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components will be stored.
- **Control Flow**:
    - A `stbi__png` structure is initialized with the context pointer `s`.
    - The function calls [`stbi__png_info_raw`](#stbi__png_info_raw), passing the initialized `stbi__png` structure and the pointers for width, height, and color components.
- **Output**: The function returns the result of [`stbi__png_info_raw`](#stbi__png_info_raw), which indicates success or failure in retrieving the image information.
- **Functions called**:
    - [`stbi__png_info_raw`](#stbi__png_info_raw)


---
### stbi\_\_png\_is16<!-- {{#callable:stbi__png_is16}} -->
Checks if the PNG image in the given context is a 16-bit depth image.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the PNG image context.
- **Control Flow**:
    - Creates a `stbi__png` structure and assigns the input context `s` to it.
    - Calls the [`stbi__png_info_raw`](#stbi__png_info_raw) function to retrieve PNG image information; if it fails, returns 0.
    - Checks if the depth of the image is 16; if not, rewinds the context and returns 0.
    - If the depth is 16, returns 1.
- **Output**: Returns 1 if the image is 16-bit depth, otherwise returns 0.
- **Functions called**:
    - [`stbi__png_info_raw`](#stbi__png_info_raw)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_bmp\_test\_raw<!-- {{#callable:stbi__bmp_test_raw}} -->
Checks if the provided context contains a valid BMP header.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the BMP file context.
- **Control Flow**:
    - The function first checks if the first byte read from the context is 'B'; if not, it returns 0.
    - It then checks if the second byte is 'M'; if not, it returns 0.
    - The function discards the next four values from the BMP header: filesize, two reserved fields, and data offset.
    - It reads the size of the DIB header and checks if it matches one of the valid BMP header sizes (12, 40, 56, 108, or 124).
    - The function returns 1 if the size is valid, otherwise it returns 0.
- **Output**: Returns 1 if the BMP header is valid, otherwise returns 0.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__get32le`](#stbi__get32le)
    - [`stbi__get16le`](#stbi__get16le)


---
### stbi\_\_bmp\_test<!-- {{#callable:stbi__bmp_test}} -->
Tests if the given BMP file is valid.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the BMP file context to be tested.
- **Control Flow**:
    - The function calls [`stbi__bmp_test_raw`](#stbi__bmp_test_raw) with the context `s` to perform the actual BMP validity test.
    - It then rewinds the context `s` using [`stbi__rewind`](#stbi__rewind) to reset the read position.
    - The result of the validity test is returned.
- **Output**: Returns an integer indicating the validity of the BMP file; a non-zero value indicates a valid BMP file, while zero indicates an invalid BMP file.
- **Functions called**:
    - [`stbi__bmp_test_raw`](#stbi__bmp_test_raw)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_high\_bit<!-- {{#callable:stbi__high_bit}} -->
Determines the position of the highest set bit in an unsigned integer.
- **Inputs**:
    - `z`: An unsigned integer whose highest set bit position is to be determined.
- **Control Flow**:
    - Checks if the input `z` is zero; if so, returns -1 indicating no bits are set.
    - If `z` is greater than or equal to 0x10000, increments `n` by 16 and right shifts `z` by 16 bits.
    - If `z` is greater than or equal to 0x00100, increments `n` by 8 and right shifts `z` by 8 bits.
    - If `z` is greater than or equal to 0x00010, increments `n` by 4 and right shifts `z` by 4 bits.
    - If `z` is greater than or equal to 0x00004, increments `n` by 2 and right shifts `z` by 2 bits.
    - If `z` is greater than or equal to 0x00002, increments `n` by 1 without shifting `z` further.
- **Output**: Returns the position of the highest set bit in `z`, or -1 if `z` is zero.


---
### stbi\_\_bitcount<!-- {{#callable:stbi__bitcount}} -->
Calculates the number of set bits (1s) in a 32-bit unsigned integer.
- **Inputs**:
    - `a`: An unsigned integer whose set bits are to be counted.
- **Control Flow**:
    - The function uses bitwise operations to count the number of set bits in the input integer `a`.
    - It first pairs bits and sums them, then groups them into larger sets, progressively reducing the number of bits to count.
    - Finally, it returns the total count of set bits, masked to fit within 8 bits.
- **Output**: Returns an integer representing the number of set bits (1s) in the input integer `a`, constrained to a maximum of 255.


---
### stbi\_\_shiftsigned<!-- {{#callable:stbi__shiftsigned}} -->
The `stbi__shiftsigned` function shifts a given unsigned integer value either left or right based on the specified shift amount and then scales it according to the number of bits specified.
- **Inputs**:
    - `v`: An unsigned integer value that will be shifted and scaled.
    - `shift`: An integer indicating the number of positions to shift the value; negative values indicate left shifts, while positive values indicate right shifts.
    - `bits`: An integer representing the number of bits to consider for scaling the value, which must be between 0 and 8.
- **Control Flow**:
    - The function initializes two static arrays: `mul_table` for multiplication factors and `shift_table` for shift amounts based on the `bits` parameter.
    - It checks if the `shift` is negative; if so, it left shifts `v` by the absolute value of `shift`, otherwise it right shifts `v` by `shift`.
    - The function asserts that the shifted value `v` is less than 256 to ensure it fits within a byte.
    - It then right shifts `v` by (8 - `bits`) to adjust the value based on the specified bit width.
    - Another assertion checks that `bits` is within the valid range of 0 to 8.
    - Finally, the function returns the scaled value by multiplying `v` with the appropriate factor from `mul_table` and then right shifting the result by the corresponding amount from `shift_table`.
- **Output**: The function returns an integer result that represents the scaled and shifted value of the input `v` based on the specified `shift` and `bits`.


---
### stbi\_\_bmp\_set\_mask\_defaults<!-- {{#callable:stbi__bmp_set_mask_defaults}} -->
Sets default color masks for BMP image data based on bits per pixel and compression type.
- **Inputs**:
    - `info`: A pointer to a `stbi__bmp_data` structure that contains information about the BMP image, including bits per pixel (bpp) and color masks.
    - `compress`: An integer indicating the compression type of the BMP image, where 0 indicates no compression and 3 indicates BI_BITFIELDS.
- **Control Flow**:
    - Checks if the `compress` value is 3; if so, it returns 1 immediately, indicating that masks should not be overridden.
    - If `compress` is 0, it checks the `bpp` value in the `info` structure.
    - For `bpp` equal to 16, it sets the red, green, and blue masks to specific bit-shifted values.
    - For `bpp` equal to 32, it sets the masks for red, green, blue, and alpha channels to their respective maximum values and initializes `all_a` to 0.
    - If `bpp` is neither 16 nor 32, it sets all masks to 0, indicating default values.
    - If `compress` is not 0 or 3, it returns 0, indicating an error.
- **Output**: Returns 1 on success (masks set or not overridden) or 0 on error (invalid compression type).


---
### stbi\_\_bmp\_parse\_header<!-- {{#callable:stbi__bmp_parse_header}} -->
Parses the header of a BMP file and populates the provided `stbi__bmp_data` structure with relevant information.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the BMP file data.
    - `info`: A pointer to a `stbi__bmp_data` structure where the parsed header information will be stored.
- **Control Flow**:
    - Checks if the first two bytes of the file are 'B' and 'M' to confirm it's a BMP file; if not, returns an error.
    - Discards the file size and reserved fields from the BMP header.
    - Reads the offset to the pixel data and the header size, storing them in the `info` structure.
    - Validates the header size against known BMP header sizes (12, 40, 56, 108, 124); returns an error for unknown sizes.
    - Reads image dimensions based on the header size and checks for a valid color depth.
    - Handles compression types and additional fields based on the header size, returning errors for unsupported types.
    - For V4/V5 headers, reads additional fields and discards unnecessary data as specified by the BMP format.
- **Output**: Returns a pointer to 1 on success, or an error message if the BMP header is invalid or unsupported.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__get32le`](#stbi__get32le)
    - [`stbi__get16le`](#stbi__get16le)
    - [`stbi__bmp_set_mask_defaults`](#stbi__bmp_set_mask_defaults)


---
### stbi\_\_bmp\_load<!-- {{#callable:stbi__bmp_load}} -->
Loads a BMP image from a given context and returns the pixel data.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that contains the image data and state.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored.
    - `req_comp`: An integer specifying the desired number of color components in the output (e.g., 3 for RGB, 4 for RGBA).
    - `ri`: A pointer to a `stbi__result_info` structure, which is not used in this function.
- **Control Flow**:
    - The function begins by parsing the BMP header using [`stbi__bmp_parse_header`](#stbi__bmp_parse_header) and checks for errors.
    - It verifies the image dimensions against a maximum limit and initializes color masks based on the header information.
    - If the image uses a palette, it reads the palette data and prepares for pixel data extraction.
    - The function handles different bits per pixel (bpp) cases, including 1, 4, 8, 16, 24, and 32 bpp, processing pixel data accordingly.
    - It checks for the presence of an alpha channel and adjusts the output format based on the requested components.
    - If the image is vertically flipped, it reverses the pixel rows.
    - Finally, it returns the pixel data along with the image dimensions and component count.
- **Output**: Returns a pointer to the pixel data of the loaded BMP image, or NULL if an error occurs.
- **Functions called**:
    - [`stbi__bmp_parse_header`](#stbi__bmp_parse_header)
    - [`stbi__skip`](#stbi__skip)
    - [`stbi__mad3sizes_valid`](#stbi__mad3sizes_valid)
    - [`stbi__malloc_mad3`](#stbi__malloc_mad3)
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__high_bit`](#stbi__high_bit)
    - [`stbi__bitcount`](#stbi__bitcount)
    - [`stbi__get16le`](#stbi__get16le)
    - [`stbi__get32le`](#stbi__get32le)
    - [`stbi__shiftsigned`](#stbi__shiftsigned)
    - [`stbi__convert_format`](#stbi__convert_format)


---
### stbi\_\_tga\_get\_comp<!-- {{#callable:stbi__tga_get_comp}} -->
Determines the component format based on the bits per pixel and whether the image is greyscale.
- **Inputs**:
    - `bits_per_pixel`: An integer representing the number of bits used per pixel in the image.
    - `is_grey`: An integer indicating whether the image is greyscale (non-zero if true).
    - `is_rgb16`: A pointer to an integer that will be set to 1 if the image is RGB16 format, otherwise set to 0.
- **Control Flow**:
    - The function first checks if the `is_rgb16` pointer is not null and initializes it to 0.
    - It then uses a switch statement to evaluate the `bits_per_pixel` value.
    - For 8 bits, it returns `STBI_grey` indicating a greyscale image.
    - For 16 bits, it checks if the image is greyscale and returns `STBI_grey_alpha` if true, otherwise it falls through to the next case.
    - For 15 bits, it sets `*is_rgb16` to 1 if the image is RGB16 and returns `STBI_rgb`.
    - For 24 and 32 bits, it returns the number of color channels (3 or 4 respectively).
    - If `bits_per_pixel` does not match any case, it returns 0.
- **Output**: Returns an integer representing the component format: `STBI_grey`, `STBI_grey_alpha`, `STBI_rgb`, or the number of color channels, or 0 for unsupported formats.


---
### stbi\_\_tga\_info<!-- {{#callable:stbi__tga_info}} -->
Extracts the width, height, and component count of a TGA image from a given context.
- **Inputs**:
    - `s`: Pointer to a `stbi__context` structure that contains the TGA image data.
    - `x`: Pointer to an integer where the width of the image will be stored.
    - `y`: Pointer to an integer where the height of the image will be stored.
    - `comp`: Pointer to an integer where the number of components per pixel will be stored.
- **Control Flow**:
    - Discards the first byte (offset) of the TGA header.
    - Reads the colormap type and checks if it is valid (0 or 1).
    - If the colormap type is invalid, rewinds the context and returns 0.
    - Reads the image type and checks if it is valid based on the colormap type.
    - If the image type is invalid, rewinds the context and returns 0.
    - If the image is colormapped, checks the bits per palette color entry for validity.
    - If the image is not colormapped, checks the bits per pixel for validity.
    - Reads the width and height of the image and checks if they are greater than 0.
    - Determines the number of components based on the bits per pixel and image type.
    - If any checks fail, rewinds the context and returns 0.
    - If all checks pass, assigns the width, height, and component count to the provided pointers and returns 1.
- **Output**: Returns 1 if the TGA image information is successfully extracted, otherwise returns 0.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__rewind`](#stbi__rewind)
    - [`stbi__skip`](#stbi__skip)
    - [`stbi__get16le`](#stbi__get16le)
    - [`stbi__tga_get_comp`](#stbi__tga_get_comp)


---
### stbi\_\_tga\_test<!-- {{#callable:stbi__tga_test}} -->
The `stbi__tga_test` function validates the header of a TGA image file to determine if it is a supported format.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the state of the file being read.
- **Control Flow**:
    - The function starts by initializing a result variable `res` to 0 and reads the first byte to discard the offset.
    - It retrieves the color type and checks if it is greater than 1, which would indicate an unsupported format.
    - If the color type is 1 (colormapped), it checks the image type and ensures it is either 1 or 9, then verifies the bits per palette color entry.
    - For normal images (color type 0), it checks that the image type is valid (2, 3, 10, or 11) and skips the colormap specification.
    - The function then checks the width and height of the image to ensure they are greater than 0.
    - It validates the bits per pixel based on the color type and ensures they are within acceptable values.
    - If all checks pass, it sets `res` to 1, indicating a valid TGA format; otherwise, it jumps to the error handling section.
    - In the error handling section, the function rewinds the context and returns the result.
- **Output**: The function returns 1 if the TGA header is valid and supported, or 0 if it is not.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__skip`](#stbi__skip)
    - [`stbi__get16le`](#stbi__get16le)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_tga\_read\_rgb16<!-- {{#callable:stbi__tga_read_rgb16}} -->
Reads a 16-bit RGB value from a TGA image and converts it to an 8-bit RGB format.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the state of the TGA image being read.
    - `out`: A pointer to an array of `stbi_uc` where the converted RGB values will be stored.
- **Control Flow**:
    - Reads a 16-bit pixel value from the TGA image using [`stbi__get16le`](#stbi__get16le).
    - Extracts the red, green, and blue components from the 16-bit value using bitwise operations.
    - Converts the 5-bit color values to 8-bit by scaling them to the range of 0-255.
    - Stores the resulting RGB values in the `out` array in the order of red, green, and blue.
- **Output**: The function does not return a value; instead, it populates the `out` array with the converted RGB values.
- **Functions called**:
    - [`stbi__get16le`](#stbi__get16le)


---
### stbi\_\_tga\_load<!-- {{#callable:stbi__tga_load}} -->
Loads a TGA image from a given context and returns the pixel data.
- **Inputs**:
    - `s`: Pointer to the `stbi__context` structure that contains the image data.
    - `x`: Pointer to an integer where the width of the image will be stored.
    - `y`: Pointer to an integer where the height of the image will be stored.
    - `comp`: Pointer to an integer where the number of color components will be stored.
    - `req_comp`: The requested number of color components for the output image.
    - `ri`: Pointer to a `stbi__result_info` structure for additional result information (unused in this function).
- **Control Flow**:
    - Reads the TGA header to extract image properties such as width, height, and color depth.
    - Checks for image size limits and returns an error if exceeded.
    - Determines if the image is RLE compressed and adjusts the image type accordingly.
    - Allocates memory for the image data based on width, height, and color components.
    - Handles indexed color images by loading the palette if necessary.
    - Reads pixel data either directly or through RLE compression, depending on the image type.
    - Inverts the image data if specified in the header.
    - Swaps RGB components if the image is not in RGB16 format.
    - Converts the image data to the requested number of components if different from the original.
- **Output**: Returns a pointer to the loaded image data in the specified format, or NULL if an error occurs.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__get16le`](#stbi__get16le)
    - [`stbi__tga_get_comp`](#stbi__tga_get_comp)
    - [`stbi__mad3sizes_valid`](#stbi__mad3sizes_valid)
    - [`stbi__malloc_mad3`](#stbi__malloc_mad3)
    - [`stbi__skip`](#stbi__skip)
    - [`stbi__getn`](#stbi__getn)
    - [`stbi__malloc_mad2`](#stbi__malloc_mad2)
    - [`stbi__tga_read_rgb16`](#stbi__tga_read_rgb16)
    - [`stbi__convert_format`](#stbi__convert_format)


---
### stbi\_\_psd\_test<!-- {{#callable:stbi__psd_test}} -->
Tests if the provided stream is a valid PSD (Photoshop Document) file by checking its signature.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the input stream to be tested.
- **Control Flow**:
    - The function reads a 4-byte big-endian integer from the stream using [`stbi__get32be`](#stbi__get32be).
    - It checks if the read value matches the PSD file signature (0x38425053).
    - The stream is then rewound to its original position using [`stbi__rewind`](#stbi__rewind).
    - The function returns 1 if the signature matches, otherwise it returns 0.
- **Output**: Returns an integer: 1 if the stream is a valid PSD file, 0 otherwise.
- **Functions called**:
    - [`stbi__get32be`](#stbi__get32be)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_psd\_decode\_rle<!-- {{#callable:stbi__psd_decode_rle}} -->
Decodes RLE (Run-Length Encoded) data from a PSD file into a pixel buffer.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the state for reading the PSD data.
    - `p`: A pointer to a buffer where the decoded pixel data will be stored.
    - `pixelCount`: The total number of pixels to decode.
- **Control Flow**:
    - Initializes `count` to track the number of pixels decoded and enters a loop that continues until all pixels are processed.
    - Within the loop, retrieves the next byte (`len`) to determine how many bytes to read or replicate.
    - If `len` is 128, it performs a no-operation (no action).
    - If `len` is less than 128, it reads `len + 1` bytes directly from the source and writes them to the output buffer.
    - If `len` is greater than 128, it calculates the number of bytes to replicate from the next byte and fills the output buffer with that value.
    - Checks for corrupt data by ensuring that the number of bytes to read does not exceed the remaining pixels.
- **Output**: Returns 1 on successful decoding of all pixels, or 0 if there was corrupt data encountered during the process.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_psd\_load<!-- {{#callable:stbi__psd_load}} -->
Loads a PSD image from a given context and returns the pixel data.
- **Inputs**:
    - `s`: Pointer to the `stbi__context` structure that contains the image data.
    - `x`: Pointer to an integer where the width of the image will be stored.
    - `y`: Pointer to an integer where the height of the image will be stored.
    - `comp`: Pointer to an integer where the number of color components will be stored.
    - `req_comp`: The number of color components requested for the output image.
    - `ri`: Pointer to a `stbi__result_info` structure to store additional result information.
    - `bpc`: Bits per channel for the output image.
- **Control Flow**:
    - Checks the PSD file identifier and version, returning an error if invalid.
    - Reads and validates the number of channels, image dimensions, bit depth, and color mode.
    - Determines if the image data is compressed and allocates memory for the output image accordingly.
    - Handles RLE decompression if the image is compressed, or reads raw pixel data if not.
    - Processes the alpha channel to remove a white matte effect if applicable.
    - Converts the image data to the requested format if necessary and updates output parameters.
- **Output**: Returns a pointer to the loaded image data in RGBA format, or an error message if the loading fails.
- **Functions called**:
    - [`stbi__get32be`](#stbi__get32be)
    - [`stbi__get16be`](#stbi__get16be)
    - [`stbi__skip`](#stbi__skip)
    - [`stbi__mad3sizes_valid`](#stbi__mad3sizes_valid)
    - [`stbi__malloc_mad3`](#stbi__malloc_mad3)
    - [`stbi__malloc`](#stbi__malloc)
    - [`stbi__psd_decode_rle`](#stbi__psd_decode_rle)
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__convert_format16`](#stbi__convert_format16)
    - [`stbi__convert_format`](#stbi__convert_format)


---
### stbi\_\_pic\_is4<!-- {{#callable:stbi__pic_is4}} -->
Checks if the first four bytes read from a given context match a specified string.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the current reading context.
    - `str`: A pointer to a string of at least four characters that the function will compare against.
- **Control Flow**:
    - The function initializes a loop that iterates four times, corresponding to the four bytes to be checked.
    - In each iteration, it reads a byte from the context using `stbi__get8(s)` and compares it to the corresponding byte in the input string `str`.
    - If any byte does not match, the function immediately returns 0, indicating a mismatch.
    - If all four bytes match, the function returns 1, indicating a successful match.
- **Output**: Returns 1 if the first four bytes from the context match the provided string, otherwise returns 0.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_pic\_test\_core<!-- {{#callable:stbi__pic_test_core}} -->
The `stbi__pic_test_core` function checks if a given stream is a valid PICT image format.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the image data stream to be tested.
- **Control Flow**:
    - The function first checks if the first four bytes of the stream match the expected signature for a PICT file using [`stbi__pic_is4`](#stbi__pic_is4).
    - If the signature check fails, the function returns 0, indicating that the stream is not a valid PICT file.
    - If the signature is valid, the function reads the next 84 bytes from the stream using [`stbi__get8`](#stbi__get8).
    - The function then checks if the next four bytes match the 'PICT' identifier using [`stbi__pic_is4`](#stbi__pic_is4) again.
    - If this second check fails, the function returns 0, indicating the stream is not a valid PICT file.
    - If both checks pass, the function returns 1, indicating a valid PICT file.
- **Output**: The function returns 1 if the stream is a valid PICT image format, and 0 otherwise.
- **Functions called**:
    - [`stbi__pic_is4`](#stbi__pic_is4)
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_readval<!-- {{#callable:stbi__readval}} -->
Reads up to 4 values from a given channel in a file context and stores them in a destination array.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure representing the file context from which values are read.
    - `channel`: An integer representing the channel mask to determine which values to read.
    - `dest`: A pointer to an array of `stbi_uc` where the read values will be stored.
- **Control Flow**:
    - Initializes a mask variable to 0x80 (binary 10000000) to check each of the 4 possible channels.
    - Iterates over 4 bits (from 0 to 3) to check if each corresponding channel is active in the `channel` mask.
    - For each active channel, checks if the end of the file is reached using `stbi__at_eof(s)`, returning an error if true.
    - If not at EOF, reads a value from the file using `stbi__get8(s)` and stores it in the `dest` array.
- **Output**: Returns a pointer to the `dest` array containing the read values, or an error if the file is too short.
- **Functions called**:
    - [`stbi__at_eof`](#stbi__at_eof)
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_copyval<!-- {{#callable:stbi__copyval}} -->
Copies values from the `src` array to the `dest` array based on the specified `channel` mask.
- **Inputs**:
    - `channel`: An integer representing a bitmask that indicates which channels to copy.
    - `dest`: A pointer to an array where the copied values will be stored.
    - `src`: A pointer to an array from which values will be copied.
- **Control Flow**:
    - Initializes a mask variable to 0x80 (binary 10000000) to check the highest bit of the `channel`.
    - Iterates over four possible channels (0 to 3) using a for loop.
    - In each iteration, checks if the current channel bit is set in `channel` using a bitwise AND operation with `mask`.
    - If the bit is set, copies the corresponding value from `src` to `dest`.
- **Output**: The function does not return a value; it modifies the `dest` array in place based on the specified channels.


---
### stbi\_\_pic\_load\_core<!-- {{#callable:stbi__pic_load_core}} -->
Loads image data from a given context into a result buffer, handling various packet types and compression methods.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the image data to be loaded.
    - `width`: An integer representing the width of the image.
    - `height`: An integer representing the height of the image.
    - `comp`: A pointer to an integer where the number of components (e.g., RGB or RGBA) will be stored.
    - `result`: A pointer to a buffer where the loaded image data will be stored.
- **Control Flow**:
    - Initializes variables for tracking the number of packets and the active components.
    - Enters a loop to read packets from the context until no more packets are chained.
    - Checks for errors in packet format and reads the channel information.
    - Determines if the image has an alpha channel based on the active components.
    - Iterates over each row of the image height to process the packets.
    - For each packet, it handles different compression types: uncompressed, pure RLE, and mixed RLE.
    - For uncompressed packets, it reads pixel values directly into the result buffer.
    - For pure RLE packets, it reads a count and value, then fills the result buffer with repeated values.
    - For mixed RLE packets, it distinguishes between repeated and raw pixel data based on the count.
- **Output**: Returns a pointer to the result buffer containing the loaded image data, or NULL in case of an error.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__at_eof`](#stbi__at_eof)
    - [`stbi__readval`](#stbi__readval)
    - [`stbi__copyval`](#stbi__copyval)
    - [`stbi__get16be`](#stbi__get16be)


---
### stbi\_\_pic\_load<!-- {{#callable:stbi__pic_load}} -->
`stbi__pic_load` loads a PIC image from a given context and returns a pointer to the image data.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that contains the image data.
    - `px`: A pointer to an integer where the width of the image will be stored.
    - `py`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer that specifies the number of components per pixel; if NULL, an internal variable is used.
    - `req_comp`: An integer that specifies the requested number of components per pixel.
    - `ri`: A pointer to a `stbi__result_info` structure, which is unused in this function.
- **Control Flow**:
    - The function begins by checking if `comp` is NULL and assigns it to an internal variable if so.
    - It reads and discards the first 92 bytes of the image data.
    - The width (`x`) and height (`y`) of the image are read from the context.
    - Checks are performed to ensure that the dimensions do not exceed predefined limits and that the file is not prematurely ended.
    - Memory is allocated for the image data buffer, initialized to opaque white, and checked for successful allocation.
    - The core loading function [`stbi__pic_load_core`](#stbi__pic_load_core) is called to load the image data into the buffer.
    - If loading fails, the allocated memory is freed and a NULL pointer is returned.
    - The width and height are stored in the provided pointers, and the image data is converted to the requested format if necessary.
    - Finally, the function returns a pointer to the loaded image data.
- **Output**: Returns a pointer to the loaded image data in the requested format, or NULL if an error occurs.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__get16be`](#stbi__get16be)
    - [`stbi__at_eof`](#stbi__at_eof)
    - [`stbi__mad3sizes_valid`](#stbi__mad3sizes_valid)
    - [`stbi__get32be`](#stbi__get32be)
    - [`stbi__malloc_mad3`](#stbi__malloc_mad3)
    - [`stbi__pic_load_core`](#stbi__pic_load_core)
    - [`stbi__convert_format`](#stbi__convert_format)


---
### stbi\_\_pic\_test<!-- {{#callable:stbi__pic_test}} -->
The `stbi__pic_test` function tests an image by invoking a core testing function and then rewinding the context.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the image context to be tested.
- **Control Flow**:
    - The function calls [`stbi__pic_test_core`](#stbi__pic_test_core) with the context `s` to perform the actual image test, storing the result in `r`.
    - It then calls [`stbi__rewind`](#stbi__rewind) to reset the context `s` to its initial state before returning the result.
- **Output**: The function returns an integer value `r`, which indicates the result of the image test performed by [`stbi__pic_test_core`](#stbi__pic_test_core).
- **Functions called**:
    - [`stbi__pic_test_core`](#stbi__pic_test_core)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_gif\_test\_raw<!-- {{#callable:stbi__gif_test_raw}} -->
The `stbi__gif_test_raw` function checks if the provided data stream corresponds to a valid GIF file header.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the data stream to be tested.
- **Control Flow**:
    - The function reads the first four bytes from the data stream and checks if they match the ASCII values for 'G', 'I', 'F', and '8'.
    - If any of these checks fail, the function returns 0, indicating that the data does not represent a valid GIF file.
    - The function then reads the next byte and checks if it is either '9' or '7', which are valid versions of the GIF format.
    - If this check fails, the function returns 0.
    - Finally, the function checks if the next byte is 'a', which is part of the GIF file specification.
    - If all checks pass, the function returns 1, indicating a valid GIF header.
- **Output**: The function returns 1 if the data stream is a valid GIF header, and 0 otherwise.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_gif\_test<!-- {{#callable:stbi__gif_test}} -->
Tests if the given context contains a valid GIF image.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the input stream to be tested.
- **Control Flow**:
    - Calls the [`stbi__gif_test_raw`](#stbi__gif_test_raw) function to perform the actual test on the input stream.
    - Rewinds the input stream to its original position using [`stbi__rewind`](#stbi__rewind).
    - Returns the result of the GIF test.
- **Output**: Returns an integer indicating whether the input stream contains a valid GIF image (non-zero for valid, zero for invalid).
- **Functions called**:
    - [`stbi__gif_test_raw`](#stbi__gif_test_raw)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_gif\_parse\_colortable<!-- {{#callable:stbi__gif_parse_colortable}} -->
Parses a color table from a GIF file and populates a palette array with RGBA values.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that provides access to the GIF file data.
    - `pal`: A 2D array of `stbi_uc` (unsigned char) to hold the color palette, where each entry contains 4 values (RGBA).
    - `num_entries`: An integer representing the number of color entries to read from the GIF file.
    - `transp`: An integer indicating the index of the transparent color in the palette.
- **Control Flow**:
    - A for loop iterates from 0 to `num_entries`, processing each color entry.
    - Within the loop, the function reads three color values (red, green, blue) from the GIF file using [`stbi__get8`](#stbi__get8).
    - The alpha value is set to 0 (transparent) if the current index matches `transp`, otherwise it is set to 255 (opaque).
- **Output**: The function does not return a value; instead, it modifies the `pal` array in place to store the parsed color values.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_gif\_header<!-- {{#callable:stbi__gif_header}} -->
The `stbi__gif_header` function reads and validates the header of a GIF file, extracting relevant information into a provided `stbi__gif` structure.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the current state of the file being read.
    - `g`: A pointer to a `stbi__gif` structure where the extracted GIF header information will be stored.
    - `comp`: A pointer to an integer that will be set to the number of color components (3 or 4) if not NULL.
    - `is_info`: An integer flag indicating whether to return early after reading the header information.
- **Control Flow**:
    - The function first checks if the first few bytes of the file match the GIF signature ('GIF8').
    - If the signature is invalid, it returns an error indicating a corrupt GIF.
    - It then checks the version of the GIF, ensuring it is either '87a' or '89a'.
    - If the version is invalid, it returns an error.
    - The function reads the width and height of the GIF, as well as other header flags and properties.
    - It checks if the width or height exceeds a predefined maximum dimension, returning an error if so.
    - If the `comp` pointer is not NULL, it sets the value to 4, indicating the potential for an alpha channel.
    - If `is_info` is true, the function returns early after reading the header.
    - If the GIF has a color table (indicated by a specific flag), it calls [`stbi__gif_parse_colortable`](#stbi__gif_parse_colortable) to parse the color table.
- **Output**: The function returns 1 on success, indicating that the header was read and processed correctly, or an error code if the GIF is invalid.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__err`](#stbi__err)
    - [`stbi__get16le`](#stbi__get16le)
    - [`stbi__gif_parse_colortable`](#stbi__gif_parse_colortable)


---
### stbi\_\_gif\_info\_raw<!-- {{#callable:stbi__gif_info_raw}} -->
The `stbi__gif_info_raw` function retrieves the width and height of a GIF image from a given context.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the GIF image data.
    - `x`: A pointer to an integer where the width of the GIF will be stored.
    - `y`: A pointer to an integer where the height of the GIF will be stored.
    - `comp`: A pointer to an integer where the number of color components will be stored.
- **Control Flow**:
    - Allocates memory for a `stbi__gif` structure and checks for successful allocation.
    - Calls [`stbi__gif_header`](#stbi__gif_header) to read the GIF header and populate the `stbi__gif` structure; if it fails, frees the allocated memory and rewinds the context.
    - If the header is successfully read, assigns the width and height to the provided pointers if they are not NULL.
    - Frees the allocated `stbi__gif` structure and returns 1 to indicate success.
- **Output**: Returns 1 if the GIF header is successfully read and the dimensions are retrieved, otherwise returns 0.
- **Functions called**:
    - [`stbi__malloc`](#stbi__malloc)
    - [`stbi__err`](#stbi__err)
    - [`stbi__gif_header`](#stbi__gif_header)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_out\_gif\_code<!-- {{#callable:stbi__out_gif_code}} -->
The `stbi__out_gif_code` function decodes and outputs a GIF code to a pixel buffer while managing the current position in the image.
- **Inputs**:
    - `g`: A pointer to a `stbi__gif` structure that contains GIF decoding state, including the output buffer, color table, and current position.
    - `code`: A `stbi__uint16` value representing the GIF code to decode and output.
- **Control Flow**:
    - The function first checks if the prefix of the given `code` is valid (non-negative) and recursively calls itself to decode the prefix.
    - It then checks if the current vertical position (`cur_y`) has reached the maximum height (`max_y`); if so, it exits the function.
    - The function calculates the current index in the output buffer based on the current x and y coordinates.
    - It marks the current position in the history to indicate that it has been processed.
    - The color corresponding to the suffix of the `code` is retrieved from the color table, and if the alpha value is greater than 128 (not transparent), the color is written to the output buffer.
    - The current x position is then incremented by 4 to move to the next pixel position.
    - If the current x position exceeds the maximum width (`max_x`), it resets `cur_x` to the starting x position and increments `cur_y` by the step size.
    - Finally, it checks if `cur_y` exceeds `max_y` and adjusts the parsing state if necessary.
- **Output**: The function does not return a value but modifies the output buffer in the `stbi__gif` structure to reflect the decoded pixel data.


---
### stbi\_\_process\_gif\_raster<!-- {{#callable:stbi__process_gif_raster}} -->
Processes a GIF raster image using LZW decompression.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the state of the GIF data stream.
    - `g`: A pointer to a `stbi__gif` structure that holds the GIF image data and decompression state.
- **Control Flow**:
    - Reads the LZW code size from the input stream and checks if it is valid (<= 12).
    - Initializes the LZW dictionary with clear codes and sets up variables for processing the GIF raster.
    - Enters an infinite loop to read and process LZW codes until the end of the stream is reached.
    - Handles the case for starting a new block of data when there are not enough valid bits.
    - Processes the codes read from the stream, handling clear codes, end of stream codes, and valid LZW codes.
    - Updates the LZW dictionary dynamically as new codes are encountered, ensuring it does not exceed the maximum size.
    - Returns error messages for illegal codes or corrupt data as necessary.
- **Output**: Returns a pointer to the output image data if successful, or NULL if an error occurs during processing.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__skip`](#stbi__skip)
    - [`stbi__out_gif_code`](#stbi__out_gif_code)


---
### stbi\_\_gif\_load\_next<!-- {{#callable:stbi__gif_load_next}} -->
Loads the next frame of a GIF image, handling background colors and disposal methods.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that contains the state of the GIF being processed.
    - `g`: A pointer to the `stbi__gif` structure that holds GIF-specific data, including dimensions and color tables.
    - `comp`: A pointer to an integer that indicates the number of color components in the output image.
    - `req_comp`: An integer specifying the requested number of color components for the output image.
    - `two_back`: A pointer to the pixel data of the previous frame, used for disposal methods.
- **Control Flow**:
    - Checks if this is the first frame; if so, initializes output buffers and sets the background color.
    - If not the first frame, determines the disposal method based on the graphic control extension flags.
    - Handles the disposal of the previous frame's pixels based on the determined method.
    - Clears the history of affected pixels from the previous frame.
    - Enters a loop to read GIF data, processing image descriptors, comments, and termination codes.
    - Processes image descriptors to extract dimensions and color tables, and calls [`stbi__process_gif_raster`](#stbi__process_gif_raster) to handle pixel data.
- **Output**: Returns a pointer to the pixel data of the loaded frame, or NULL on failure.
- **Functions called**:
    - [`stbi__gif_header`](#stbi__gif_header)
    - [`stbi__mad3sizes_valid`](#stbi__mad3sizes_valid)
    - [`stbi__malloc`](#stbi__malloc)
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__get16le`](#stbi__get16le)
    - [`stbi__gif_parse_colortable`](#stbi__gif_parse_colortable)
    - [`stbi__process_gif_raster`](#stbi__process_gif_raster)
    - [`stbi__skip`](#stbi__skip)


---
### stbi\_\_load\_gif\_main\_outofmem<!-- {{#callable:stbi__load_gif_main_outofmem}} -->
Cleans up memory allocated for GIF loading and returns an error message.
- **Inputs**:
    - `g`: A pointer to a `stbi__gif` structure that contains GIF loading data.
    - `out`: A pointer to a buffer that may hold image data, which will be freed if not NULL.
    - `delays`: A pointer to an array of integer delays, which will be freed if it points to allocated memory.
- **Control Flow**:
    - The function starts by freeing the memory allocated for the `out`, `history`, and `background` fields of the `stbi__gif` structure.
    - It checks if the `out` pointer is not NULL, and if so, it frees the memory pointed to by `out`.
    - Next, it checks if `delays` is not NULL and if the pointer it points to is not NULL, and if both conditions are met, it frees the memory for the delays array.
    - Finally, the function returns an error message indicating an out-of-memory condition.
- **Output**: Returns a pointer to an error message string indicating that an out-of-memory condition has occurred.


---
### stbi\_\_load\_gif\_main<!-- {{#callable:stbi__load_gif_main}} -->
`stbi__load_gif_main` loads an animated GIF from a given context and returns the pixel data along with frame delays.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that contains the GIF data.
    - `delays`: A pointer to an integer pointer that will hold the delays for each frame of the GIF.
    - `x`: A pointer to an integer that will be set to the width of the GIF.
    - `y`: A pointer to an integer that will be set to the height of the GIF.
    - `z`: A pointer to an integer that will be set to the number of frames (layers) in the GIF.
    - `comp`: A pointer to an integer that specifies the desired number of color components in the output.
    - `req_comp`: An integer that specifies the required number of color components to return.
- **Control Flow**:
    - The function first checks if the input context `s` contains a valid GIF using [`stbi__gif_test`](#stbi__gif_test).
    - If valid, it initializes necessary variables and enters a loop to load each frame of the GIF using [`stbi__gif_load_next`](#stbi__gif_load_next).
    - For each frame loaded, it reallocates memory for the output buffer and the delays array if they are provided.
    - It copies the pixel data of the current frame into the output buffer and stores the delay for that frame.
    - The loop continues until all frames are loaded, indicated by a return value of 0 from [`stbi__gif_load_next`](#stbi__gif_load_next).
    - After loading all frames, it frees temporary buffers and performs a final conversion of the pixel format if required.
    - Finally, it sets the number of layers and returns the pointer to the output pixel data.
- **Output**: Returns a pointer to the pixel data of the loaded GIF, or an error message if the input is not a GIF.
- **Functions called**:
    - [`stbi__gif_test`](#stbi__gif_test)
    - [`stbi__gif_load_next`](#stbi__gif_load_next)
    - [`stbi__load_gif_main_outofmem`](#stbi__load_gif_main_outofmem)
    - [`stbi__malloc`](#stbi__malloc)
    - [`stbi__convert_format`](#stbi__convert_format)


---
### stbi\_\_gif\_load<!-- {{#callable:stbi__gif_load}} -->
Loads a GIF image from a given context and returns a pointer to the image data.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that contains the GIF data to be loaded.
    - `x`: A pointer to an integer where the width of the loaded image will be stored.
    - `y`: A pointer to an integer where the height of the loaded image will be stored.
    - `comp`: A pointer to an integer that indicates the number of color components in the image.
    - `req_comp`: An integer that specifies the desired number of color components for the output image.
    - `ri`: A pointer to a `stbi__result_info` structure that is not used in this function.
- **Control Flow**:
    - Initializes a `stbi__gif` structure and sets it to zero.
    - Calls [`stbi__gif_load_next`](#stbi__gif_load_next) to load the next frame of the GIF image.
    - Checks if the returned pointer is equal to the context pointer, indicating the end of the GIF; if so, sets the pointer to zero.
    - If the image data is successfully loaded, updates the width and height pointers, and converts the image format if required.
    - If loading fails but an output buffer was allocated, it frees the allocated buffer.
    - Frees any buffers used for loading multiple frames, including history and background buffers.
    - Returns the pointer to the loaded image data or NULL if loading failed.
- **Output**: Returns a pointer to the loaded image data in the specified format, or NULL if the loading fails.
- **Functions called**:
    - [`stbi__gif_load_next`](#stbi__gif_load_next)
    - [`stbi__convert_format`](#stbi__convert_format)


---
### stbi\_\_gif\_info<!-- {{#callable:stbi__gif_info}} -->
The `stbi__gif_info` function retrieves GIF image information by delegating the task to [`stbi__gif_info_raw`](#stbi__gif_info_raw).
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the state and data for reading the GIF image.
    - `x`: A pointer to an integer where the width of the GIF image will be stored.
    - `y`: A pointer to an integer where the height of the GIF image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the GIF image will be stored.
- **Control Flow**:
    - The function calls [`stbi__gif_info_raw`](#stbi__gif_info_raw), passing the same parameters it received.
    - It does not perform any additional processing or error handling.
- **Output**: The function returns the result of the [`stbi__gif_info_raw`](#stbi__gif_info_raw) function, which indicates success or failure in retrieving the GIF image information.
- **Functions called**:
    - [`stbi__gif_info_raw`](#stbi__gif_info_raw)


---
### stbi\_\_hdr\_test\_core<!-- {{#callable:stbi__hdr_test_core}} -->
The `stbi__hdr_test_core` function checks if the bytes read from a given context match a specified signature.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the current reading context.
    - `signature`: A string containing the expected byte sequence to compare against.
- **Control Flow**:
    - The function initializes a loop to iterate over each character in the `signature` string.
    - For each character, it reads a byte from the `stbi__context` using the [`stbi__get8`](#stbi__get8) function.
    - If any byte read does not match the corresponding character in the `signature`, the function returns 0, indicating a mismatch.
    - If all bytes match, the function rewinds the context using [`stbi__rewind`](#stbi__rewind) and returns 1, indicating a successful match.
- **Output**: The function returns 1 if the bytes match the signature, and 0 if they do not.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_hdr\_test<!-- {{#callable:stbi__hdr_test}} -->
The `stbi__hdr_test` function checks if the provided context contains valid HDR image data by testing for specific header strings.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the state of the image data being tested.
- **Control Flow**:
    - The function first calls [`stbi__hdr_test_core`](#stbi__hdr_test_core) with the string '#?RADIANCE\n' to check for the presence of the Radiance HDR format.
    - If the first test fails (returns 0), it rewinds the context using [`stbi__rewind`](#stbi__rewind) and then tests for the string '#?RGBE\n' to check for the RGBE format.
    - After each test, the context is rewound to ensure that the tests are performed from the beginning of the data.
- **Output**: The function returns an integer value: a non-zero value if a valid HDR header is found, and 0 if neither header is present.
- **Functions called**:
    - [`stbi__hdr_test_core`](#stbi__hdr_test_core)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_hdr\_gettoken<!-- {{#callable:stbi__hdr_gettoken}} -->
Retrieves a token from a header context until a newline or end of file is reached.
- **Inputs**:
    - `z`: A pointer to a `stbi__context` structure that represents the current state of the header being read.
    - `buffer`: A character array where the retrieved token will be stored.
- **Control Flow**:
    - The function initializes a length counter `len` to 0 and reads the first character using [`stbi__get8`](#stbi__get8).
    - It enters a loop that continues until the end of the file is reached or a newline character is encountered.
    - Within the loop, the current character is stored in the `buffer`, and the length counter is incremented.
    - If the length of the token reaches the maximum buffer length minus one, it flushes the remaining characters until a newline is found.
    - After exiting the loop, it null-terminates the `buffer` and returns it.
- **Output**: Returns the `buffer` containing the retrieved token, null-terminated.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__at_eof`](#stbi__at_eof)


---
### stbi\_\_hdr\_convert<!-- {{#callable:stbi__hdr_convert}} -->
Converts HDR input pixel values to a specified output format based on the required number of components.
- **Inputs**:
    - `output`: A pointer to an array where the converted pixel values will be stored.
    - `input`: A pointer to an array containing the HDR pixel values to be converted.
    - `req_comp`: An integer specifying the number of components required in the output (1, 2, 3, or 4).
- **Control Flow**:
    - Checks if the fourth component of the input is non-zero to determine if conversion is needed.
    - Calculates the exponent based on the fourth component of the input using `ldexp`.
    - If `req_comp` is less than or equal to 2, computes the average of the first three input components and scales it by the exponent.
    - If `req_comp` is greater than 2, scales each of the first three input components individually by the exponent.
    - Sets the second output component to 1 if `req_comp` is 2, or the fourth output component to 1 if `req_comp` is 4.
    - If the fourth component of the input is zero, sets the output based on the value of `req_comp` to handle cases of no valid input.
- **Output**: The function does not return a value but populates the `output` array with the converted pixel values based on the specified number of components.


---
### stbi\_\_hdr\_load<!-- {{#callable:stbi__hdr_load}} -->
Loads an HDR image from a given context and returns the pixel data as a float array.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that contains the image data.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components will be stored.
    - `req_comp`: An integer specifying the number of color components requested (e.g., 3 for RGB).
    - `ri`: A pointer to a `stbi__result_info` structure, which is unused in this function.
- **Control Flow**:
    - Checks if the HDR file starts with a valid identifier ('#?RADIANCE' or '#?RGBE').
    - Parses the header to validate the format and extract the image dimensions.
    - Allocates memory for the HDR data based on the width, height, and requested components.
    - Reads the image data either as flat data or RLE-encoded data depending on the width.
    - Converts the read data into the appropriate format and stores it in the allocated memory.
- **Output**: Returns a pointer to a float array containing the HDR image data, or an error if the loading fails.
- **Functions called**:
    - [`stbi__hdr_gettoken`](#stbi__hdr_gettoken)
    - [`stbi__mad4sizes_valid`](#stbi__mad4sizes_valid)
    - [`stbi__malloc_mad4`](#stbi__malloc_mad4)
    - [`stbi__getn`](#stbi__getn)
    - [`stbi__hdr_convert`](#stbi__hdr_convert)
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__malloc_mad2`](#stbi__malloc_mad2)


---
### stbi\_\_hdr\_info<!-- {{#callable:stbi__hdr_info}} -->
The `stbi__hdr_info` function extracts image dimensions and component count from a High Dynamic Range (HDR) image file.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the HDR image context.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components will be stored.
- **Control Flow**:
    - The function initializes a buffer and checks if the pointers for width, height, and components are NULL, assigning them to a dummy variable if they are.
    - It tests if the HDR context is valid using [`stbi__hdr_test`](#stbi__hdr_test), and if not, it rewinds the context and returns 0.
    - It enters a loop to read tokens from the HDR file, checking for the 'FORMAT=32-bit_rle_rgbe' token to validate the file format.
    - If the format is valid, it reads the next token to check for the height, ensuring it starts with '-Y '.
    - It then reads the width, ensuring it starts with '+X ', and finally sets the number of components to 3.
    - If any checks fail, it rewinds the context and returns 0; otherwise, it returns 1.
- **Output**: The function returns 1 if the HDR information is successfully extracted, or 0 if there is an error or the format is invalid.
- **Functions called**:
    - [`stbi__hdr_test`](#stbi__hdr_test)
    - [`stbi__rewind`](#stbi__rewind)
    - [`stbi__hdr_gettoken`](#stbi__hdr_gettoken)


---
### stbi\_\_bmp\_info<!-- {{#callable:stbi__bmp_info}} -->
The `stbi__bmp_info` function retrieves the dimensions and color component information of a BMP image.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the BMP image data.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components will be stored.
- **Control Flow**:
    - The function initializes a `stbi__bmp_data` structure to hold BMP header information.
    - It calls [`stbi__bmp_parse_header`](#stbi__bmp_parse_header) to parse the BMP header and retrieve image information.
    - If the header parsing fails (returns NULL), the function rewinds the context and returns 0.
    - If the parsing is successful, it assigns the image width and height to the provided pointers if they are not NULL.
    - It determines the number of color components based on the bits per pixel (bpp) and the alpha mask, storing the result in the `comp` pointer if it is not NULL.
    - Finally, the function returns 1 to indicate success.
- **Output**: The function returns 1 on success and 0 on failure, indicating whether the image information was successfully retrieved.
- **Functions called**:
    - [`stbi__bmp_parse_header`](#stbi__bmp_parse_header)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_psd\_info<!-- {{#callable:stbi__psd_info}} -->
The `stbi__psd_info` function extracts and validates information from a PSD (Photoshop Document) file, including its dimensions and channel count.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the PSD file context.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components will be stored.
- **Control Flow**:
    - The function checks if the pointers `x`, `y`, and `comp` are NULL and assigns them to a dummy variable if they are.
    - It reads the first 4 bytes to verify the PSD signature (0x38425053); if it doesn't match, it rewinds the context and returns 0.
    - It checks the version number (should be 1); if not, it rewinds and returns 0.
    - The function skips 6 bytes and reads the channel count, validating it to be between 0 and 16; if invalid, it rewinds and returns 0.
    - It reads the height and width of the image, storing them in the provided pointers.
    - It checks the depth of the image (should be either 8 or 16); if not, it rewinds and returns 0.
    - It verifies that the color mode is 3; if not, it rewinds and returns 0.
    - If all checks pass, it sets `comp` to 4 (indicating RGBA) and returns 1.
- **Output**: The function returns 1 if the PSD file is valid and the information is successfully extracted; otherwise, it returns 0.
- **Functions called**:
    - [`stbi__get32be`](#stbi__get32be)
    - [`stbi__rewind`](#stbi__rewind)
    - [`stbi__get16be`](#stbi__get16be)
    - [`stbi__skip`](#stbi__skip)


---
### stbi\_\_psd\_is16<!-- {{#callable:stbi__psd_is16}} -->
The `stbi__psd_is16` function checks if a given PSD file context represents a 16-bit per channel image.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the PSD file context.
- **Control Flow**:
    - The function first checks if the file signature matches the PSD format by reading 4 bytes and comparing it to the constant 0x38425053.
    - If the signature is incorrect, it rewinds the context and returns 0.
    - Next, it checks if the version of the PSD file is 1 by reading 2 bytes.
    - If the version is not 1, it rewinds the context and returns 0.
    - The function skips 6 bytes to move to the channel count, which is read next.
    - It checks if the channel count is within the valid range (0 to 16); if not, it rewinds and returns 0.
    - The function reads and ignores two 32-bit values, which are not used in the validation process.
    - Finally, it checks if the depth is exactly 16 bits; if not, it rewinds and returns 0.
    - If all checks pass, the function returns 1, indicating that the PSD file is a valid 16-bit per channel image.
- **Output**: The function returns 1 if the PSD file is a valid 16-bit per channel image, and 0 otherwise.
- **Functions called**:
    - [`stbi__get32be`](#stbi__get32be)
    - [`stbi__rewind`](#stbi__rewind)
    - [`stbi__get16be`](#stbi__get16be)
    - [`stbi__skip`](#stbi__skip)


---
### stbi\_\_pic\_info<!-- {{#callable:stbi__pic_info}} -->
The `stbi__pic_info` function retrieves image dimensions and component information from a given image context.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure representing the image data.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components will be stored.
- **Control Flow**:
    - Checks if the pointers `x`, `y`, or `comp` are NULL and assigns them to a dummy variable if they are.
    - Validates the image format by checking a specific signature; if invalid, it rewinds the context and returns 0.
    - Skips the first 88 bytes of the image data to reach the dimensions.
    - Retrieves the width and height of the image; if end of file is reached or dimensions are invalid, it rewinds and returns 0.
    - Skips an additional 8 bytes before processing packets.
    - Enters a loop to read packets until no more chained packets are found, checking for EOF and validating packet size.
    - Determines the number of components based on the channel information from the packets.
- **Output**: Returns 1 on success, indicating that the image information was successfully retrieved, or 0 on failure.
- **Functions called**:
    - [`stbi__pic_is4`](#stbi__pic_is4)
    - [`stbi__rewind`](#stbi__rewind)
    - [`stbi__skip`](#stbi__skip)
    - [`stbi__get16be`](#stbi__get16be)
    - [`stbi__at_eof`](#stbi__at_eof)
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_pnm\_test<!-- {{#callable:stbi__pnm_test}} -->
The `stbi__pnm_test` function checks if the input stream corresponds to a valid PNM (Portable Any Map) format.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure representing the input stream to be tested.
- **Control Flow**:
    - The function reads two bytes from the input stream: the first byte is expected to be 'P' and the second byte should be either '5' or '6'.
    - If the first byte is not 'P' or the second byte is neither '5' nor '6', the function rewinds the input stream and returns 0, indicating an invalid format.
    - If both conditions are satisfied, the function returns 1, indicating a valid PNM format.
- **Output**: The function returns 1 if the input stream is a valid PNM format (either P5 or P6), and 0 otherwise.
- **Functions called**:
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__rewind`](#stbi__rewind)


---
### stbi\_\_pnm\_load<!-- {{#callable:stbi__pnm_load}} -->
`stbi__pnm_load` loads a PNM image from a given context and returns a pointer to the image data.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that contains the image data.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored.
    - `req_comp`: An integer specifying the requested number of color components.
    - `ri`: A pointer to a `stbi__result_info` structure to store additional result information.
- **Control Flow**:
    - The function retrieves image dimensions and color components using [`stbi__pnm_info`](#stbi__pnm_info).
    - It checks for image size limits and returns an error if exceeded.
    - It allocates memory for the image data based on the dimensions and color components.
    - It reads the image data into the allocated memory using [`stbi__getn`](#stbi__getn).
    - If a different number of components is requested, it converts the image format accordingly.
    - The function returns the pointer to the image data or an error if any step fails.
- **Output**: Returns a pointer to the loaded image data or NULL if an error occurs.
- **Functions called**:
    - [`stbi__pnm_info`](#stbi__pnm_info)
    - [`stbi__mad4sizes_valid`](#stbi__mad4sizes_valid)
    - [`stbi__malloc_mad4`](#stbi__malloc_mad4)
    - [`stbi__getn`](#stbi__getn)
    - [`stbi__convert_format16`](#stbi__convert_format16)
    - [`stbi__convert_format`](#stbi__convert_format)


---
### stbi\_\_pnm\_isspace<!-- {{#callable:stbi__pnm_isspace}} -->
Determines if a given character is a whitespace character.
- **Inputs**:
    - `c`: A character to be checked for whitespace.
- **Control Flow**:
    - The function evaluates the input character `c` against a set of predefined whitespace characters.
    - It returns 1 (true) if `c` matches any of the whitespace characters, otherwise it returns 0 (false).
- **Output**: Returns an integer value: 1 if the character is a whitespace character, and 0 otherwise.


---
### stbi\_\_pnm\_skip\_whitespace<!-- {{#callable:stbi__pnm_skip_whitespace}} -->
The `stbi__pnm_skip_whitespace` function skips over whitespace and comments in a PNM (Portable Any Map) file.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the current state of the file being read.
    - `c`: A pointer to a character variable that is used to store the current character being read from the file.
- **Control Flow**:
    - The function enters an infinite loop that continues until a break condition is met.
    - Within the loop, it first checks if the end of the file is not reached and if the current character is whitespace using [`stbi__pnm_isspace`](#stbi__pnm_isspace), and if so, it reads the next character using [`stbi__get8`](#stbi__get8).
    - If the end of the file is reached or the current character is not a comment indicator ('#'), the loop breaks.
    - If the current character is a comment indicator, it enters another loop that continues reading characters until it encounters a newline ('
') or carriage return (''), effectively skipping the comment.
- **Output**: The function does not return a value; it modifies the character pointed to by `c` to the next relevant character after skipping whitespace and comments.
- **Functions called**:
    - [`stbi__at_eof`](#stbi__at_eof)
    - [`stbi__pnm_isspace`](#stbi__pnm_isspace)
    - [`stbi__get8`](#stbi__get8)


---
### stbi\_\_pnm\_isdigit<!-- {{#callable:stbi__pnm_isdigit}} -->
Checks if a character is a digit (0-9).
- **Inputs**:
    - `c`: A character to be checked if it is a digit.
- **Control Flow**:
    - Evaluates whether the character `c` falls within the ASCII range of '0' to '9'.
    - Returns 1 (true) if `c` is a digit, otherwise returns 0 (false).
- **Output**: Returns an integer: 1 if the character is a digit, 0 otherwise.


---
### stbi\_\_pnm\_getinteger<!-- {{#callable:stbi__pnm_getinteger}} -->
Parses an integer from a PNM (Portable Anymap) file context until a non-digit character is encountered.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure representing the current state of the PNM file being read.
    - `c`: A pointer to a character that is expected to point to the first digit of the integer to be parsed.
- **Control Flow**:
    - Initializes an integer `value` to 0 to accumulate the parsed integer.
    - Enters a loop that continues until the end of the file is reached or a non-digit character is encountered.
    - In each iteration, it updates `value` by multiplying the current `value` by 10 and adding the integer value of the current character.
    - Retrieves the next character from the file and checks if it is a digit.
    - Checks for integer overflow conditions, returning an error if the parsed value exceeds the limits of a 32-bit integer.
- **Output**: Returns the parsed integer value if successful, or triggers an error message if an overflow occurs.
- **Functions called**:
    - [`stbi__at_eof`](#stbi__at_eof)
    - [`stbi__pnm_isdigit`](#stbi__pnm_isdigit)
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__err`](#stbi__err)


---
### stbi\_\_pnm\_info<!-- {{#callable:stbi__pnm_info}} -->
The `stbi__pnm_info` function extracts image dimensions and component information from a PNM (Portable Any Map) file header.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the PNM file context.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components will be stored.
- **Control Flow**:
    - The function checks if the pointers `x`, `y`, or `comp` are NULL and assigns them to a dummy variable if they are.
    - It rewinds the file context to the beginning to read the header.
    - The function reads the identifier characters to determine if the file is a valid PNM format (either 'P5' for grayscale or 'P6' for color).
    - If the identifier is invalid, it rewinds the context and returns 0.
    - It sets the number of components based on the identifier: 1 for 'P5' and 3 for 'P6'.
    - The function reads the width and height of the image, ensuring they are valid (non-zero).
    - It reads the maximum value from the header and checks if it exceeds 65535, returning an error if it does.
    - Finally, it returns 16 if the max value is greater than 255, otherwise it returns 8.
- **Output**: The function returns the bit depth of the image (8 or 16) or 0 if the header is invalid.
- **Functions called**:
    - [`stbi__rewind`](#stbi__rewind)
    - [`stbi__get8`](#stbi__get8)
    - [`stbi__pnm_skip_whitespace`](#stbi__pnm_skip_whitespace)
    - [`stbi__pnm_getinteger`](#stbi__pnm_getinteger)
    - [`stbi__err`](#stbi__err)


---
### stbi\_\_pnm\_is16<!-- {{#callable:stbi__pnm_is16}} -->
Determines if the PNM image format is 16-bit.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the image data and state.
- **Control Flow**:
    - Calls the [`stbi__pnm_info`](#stbi__pnm_info) function with the context pointer `s` to check the bit depth of the PNM image.
    - If the returned value is 16, the function returns 1, indicating that the image is 16-bit.
    - If the returned value is not 16, the function returns 0, indicating that the image is not 16-bit.
- **Output**: Returns 1 if the PNM image is 16-bit, otherwise returns 0.
- **Functions called**:
    - [`stbi__pnm_info`](#stbi__pnm_info)


---
### stbi\_\_info\_main<!-- {{#callable:stbi__info_main}} -->
The `stbi__info_main` function retrieves image information such as dimensions and component count from various supported image formats.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that contains the image data to be analyzed.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored.
- **Control Flow**:
    - The function first checks if JPEG information can be retrieved using [`stbi__jpeg_info`](#stbi__jpeg_info), and if successful, returns 1.
    - It continues to check for PNG, GIF, BMP, PSD, PIC, PNM, and HDR formats in a similar manner, returning 1 if any of these formats successfully provide information.
    - Finally, it checks for TGA format, which is tested last due to its less reliable nature.
    - If none of the format checks succeed, the function returns an error message indicating that the image type is unknown or corrupt.
- **Output**: The function returns 1 if image information is successfully retrieved from any supported format, or an error message if the image type is unknown or corrupt.
- **Functions called**:
    - [`stbi__jpeg_info`](#stbi__jpeg_info)
    - [`stbi__png_info`](#stbi__png_info)
    - [`stbi__gif_info`](#stbi__gif_info)
    - [`stbi__bmp_info`](#stbi__bmp_info)
    - [`stbi__psd_info`](#stbi__psd_info)
    - [`stbi__pic_info`](#stbi__pic_info)
    - [`stbi__pnm_info`](#stbi__pnm_info)
    - [`stbi__hdr_info`](#stbi__hdr_info)
    - [`stbi__tga_info`](#stbi__tga_info)
    - [`stbi__err`](#stbi__err)


---
### stbi\_\_is\_16\_main<!-- {{#callable:stbi__is_16_main}} -->
Determines if the input stream contains a 16-bit image format.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the input stream.
- **Control Flow**:
    - Checks if the input stream is a 16-bit PNG image using [`stbi__png_is16`](#stbi__png_is16) and returns 1 if true.
    - Checks if the input stream is a 16-bit PSD image using [`stbi__psd_is16`](#stbi__psd_is16) and returns 1 if true.
    - Checks if the input stream is a 16-bit PNM image using [`stbi__pnm_is16`](#stbi__pnm_is16) and returns 1 if true.
    - If none of the above checks are true, returns 0.
- **Output**: Returns 1 if the input stream is identified as a 16-bit image format (PNG, PSD, or PNM), otherwise returns 0.
- **Functions called**:
    - [`stbi__png_is16`](#stbi__png_is16)
    - [`stbi__psd_is16`](#stbi__psd_is16)
    - [`stbi__pnm_is16`](#stbi__pnm_is16)


---
### stbi\_info<!-- {{#callable:stbi_info}} -->
The `stbi_info` function retrieves image dimensions and component count from a specified image file.
- **Inputs**:
    - `filename`: A constant character pointer representing the path to the image file to be opened.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored.
- **Control Flow**:
    - The function attempts to open the specified file in binary read mode using [`stbi__fopen`](#stbi__fopen).
    - If the file cannot be opened, it returns an error message indicating the failure.
    - If the file is successfully opened, it calls [`stbi_info_from_file`](#stbi_info_from_file) to extract the image dimensions and component count.
    - After retrieving the information, the file is closed using `fclose`.
    - Finally, the result from [`stbi_info_from_file`](#stbi_info_from_file) is returned.
- **Output**: The function returns an integer indicating success or failure, with the dimensions and component count stored in the provided pointers.
- **Functions called**:
    - [`stbi__fopen`](#stbi__fopen)
    - [`stbi__err`](#stbi__err)
    - [`stbi_info_from_file`](#stbi_info_from_file)


---
### stbi\_info\_from\_file<!-- {{#callable:stbi_info_from_file}} -->
Retrieves image dimensions and component count from a file.
- **Inputs**:
    - `f`: A pointer to a `FILE` object representing the image file from which to extract information.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored.
- **Control Flow**:
    - The function starts by saving the current position in the file using `ftell`.
    - It initializes a `stbi__context` structure for reading the file.
    - The function then calls [`stbi__info_main`](#stbi__info_main) to extract the image's width, height, and component count, passing the context and pointers to the output variables.
    - Finally, it restores the file position using `fseek` before returning the result of the information extraction.
- **Output**: Returns an integer indicating success (non-zero) or failure (zero) of the information retrieval process.
- **Functions called**:
    - [`stbi__start_file`](#stbi__start_file)
    - [`stbi__info_main`](#stbi__info_main)


---
### stbi\_is\_16\_bit<!-- {{#callable:stbi_is_16_bit}} -->
The `stbi_is_16_bit` function checks if a specified image file is in 16-bit format.
- **Inputs**:
    - `filename`: A constant character pointer representing the path to the image file to be checked.
- **Control Flow**:
    - The function attempts to open the file specified by `filename` in binary read mode.
    - If the file cannot be opened, it returns an error message indicating the failure.
    - If the file is successfully opened, it calls the [`stbi_is_16_bit_from_file`](#stbi_is_16_bit_from_file) function to determine if the file is in 16-bit format.
    - After checking, it closes the file and returns the result of the 16-bit check.
- **Output**: The function returns an integer indicating whether the file is in 16-bit format, or an error code if the file could not be opened.
- **Functions called**:
    - [`stbi__fopen`](#stbi__fopen)
    - [`stbi__err`](#stbi__err)
    - [`stbi_is_16_bit_from_file`](#stbi_is_16_bit_from_file)


---
### stbi\_is\_16\_bit\_from\_file<!-- {{#callable:stbi_is_16_bit_from_file}} -->
Determines if the image file pointed to by the given file pointer is a 16-bit image.
- **Inputs**:
    - `f`: A pointer to a `FILE` object that represents the image file to be checked.
- **Control Flow**:
    - The function retrieves the current position in the file using `ftell`.
    - It initializes a `stbi__context` structure for reading the file.
    - The function checks if the file is a 16-bit image by calling [`stbi__is_16_main`](#stbi__is_16_main) with the context.
    - Finally, it restores the original file position using `fseek` before returning the result.
- **Output**: Returns a non-zero value if the file is a 16-bit image, and zero otherwise.
- **Functions called**:
    - [`stbi__start_file`](#stbi__start_file)
    - [`stbi__is_16_main`](#stbi__is_16_main)


---
### stbi\_info\_from\_memory<!-- {{#callable:stbi_info_from_memory}} -->
Retrieves image information such as dimensions and component count from a memory buffer.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing the image data.
    - `len`: The length of the memory buffer in bytes.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored.
- **Control Flow**:
    - Initializes a `stbi__context` structure to manage the image data.
    - Calls [`stbi__start_mem`](#stbi__start_mem) to set up the context with the provided buffer and length.
    - Invokes [`stbi__info_main`](#stbi__info_main) to extract the image information, passing the context and pointers for width, height, and component count.
- **Output**: Returns the number of components in the image if successful, or 0 if there was an error.
- **Functions called**:
    - [`stbi__start_mem`](#stbi__start_mem)
    - [`stbi__info_main`](#stbi__info_main)


---
### stbi\_info\_from\_callbacks<!-- {{#callable:stbi_info_from_callbacks}} -->
The `stbi_info_from_callbacks` function retrieves image information such as width, height, and number of components from a source defined by callbacks.
- **Inputs**:
    - `c`: A pointer to a `stbi_io_callbacks` structure that contains callback functions for reading data.
    - `user`: A pointer to user-defined data that is passed to the callback functions.
    - `x`: A pointer to an integer where the width of the image will be stored.
    - `y`: A pointer to an integer where the height of the image will be stored.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored.
- **Control Flow**:
    - The function initializes a `stbi__context` structure to manage the image data.
    - It calls [`stbi__start_callbacks`](#stbi__start_callbacks) to set up the context with the provided callbacks and user data.
    - Finally, it invokes [`stbi__info_main`](#stbi__info_main) to extract the image information and returns the result.
- **Output**: The function returns an integer indicating success or failure of the operation, while also populating the provided pointers with the image's width, height, and number of components.
- **Functions called**:
    - [`stbi__start_callbacks`](#stbi__start_callbacks)
    - [`stbi__info_main`](#stbi__info_main)


---
### stbi\_is\_16\_bit\_from\_memory<!-- {{#callable:stbi_is_16_bit_from_memory}} -->
Determines if the provided memory buffer contains 16-bit image data.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing image data.
    - `len`: The length of the memory buffer in bytes.
- **Control Flow**:
    - Initializes a `stbi__context` structure to manage the memory buffer.
    - Calls [`stbi__start_mem`](#stbi__start_mem) to set up the context with the provided buffer and length.
    - Invokes [`stbi__is_16_main`](#stbi__is_16_main) to check if the image data in the context is 16-bit.
- **Output**: Returns a non-zero value if the image data is 16-bit, otherwise returns 0.
- **Functions called**:
    - [`stbi__start_mem`](#stbi__start_mem)
    - [`stbi__is_16_main`](#stbi__is_16_main)


---
### stbi\_is\_16\_bit\_from\_callbacks<!-- {{#callable:stbi_is_16_bit_from_callbacks}} -->
Determines if the image data from the provided callbacks is in 16-bit format.
- **Inputs**:
    - `c`: A pointer to `stbi_io_callbacks` structure that contains callback functions for reading image data.
    - `user`: A pointer to user-defined data that is passed to the callback functions.
- **Control Flow**:
    - Initializes a `stbi__context` structure to manage the image data processing.
    - Calls [`stbi__start_callbacks`](#stbi__start_callbacks) to set up the context with the provided callbacks and user data.
    - Invokes [`stbi__is_16_main`](#stbi__is_16_main) to check if the image data is in 16-bit format and returns the result.
- **Output**: Returns an integer indicating whether the image data is in 16-bit format (non-zero for true, zero for false).
- **Functions called**:
    - [`stbi__start_callbacks`](#stbi__start_callbacks)
    - [`stbi__is_16_main`](#stbi__is_16_main)


# Function Declarations (Public API)

---
### stbi\_load\_from\_memory<!-- {{#callable_declaration:stbi_load_from_memory}} -->
Loads an image from a memory buffer.
- **Description**: This function is used to load an image from a memory buffer, which is particularly useful for applications that need to process images stored in memory rather than on disk. It should be called with a valid buffer containing image data and the length of that data. The function will decode the image and provide its dimensions and color components through the provided pointers. It is important to ensure that the buffer is not null and that the length is greater than zero. The function may return a null pointer if the image cannot be loaded, which indicates an error in the provided data.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing the image data. Must not be null and should point to valid image data.
    - `len`: The length of the buffer in bytes. Must be greater than zero.
    - `x`: A pointer to an integer where the width of the loaded image will be stored. Can be null if the width is not needed.
    - `y`: A pointer to an integer where the height of the loaded image will be stored. Can be null if the height is not needed.
    - `comp`: A pointer to an integer where the number of color components in the loaded image will be stored. Can be null if this information is not needed.
    - `req_comp`: The number of color components requested for the output image. Valid values are typically 1, 3, or 4, corresponding to grayscale, RGB, or RGBA formats.
- **Output**: Returns a pointer to the loaded image data in 8-bit format. If the image cannot be loaded, it returns null.
- **See also**: [`stbi_load_from_memory`](#stbi_load_from_memory)  (Implementation)


---
### stbi\_load\_from\_callbacks<!-- {{#callable_declaration:stbi_load_from_callbacks}} -->
Loads an image from callbacks.
- **Description**: This function is used to load an image from a series of callbacks that provide the image data. It is essential to call this function when you need to load images dynamically from sources such as memory or network streams, rather than from a file. Before calling, ensure that the `clbk` parameter is properly initialized and that the `user` pointer is set to any user-defined data you want to pass to the callbacks. The function will populate the width and height of the image in the `x` and `y` parameters, respectively, and can also return the number of color components in the image through the `comp` parameter. If the image loading fails, the function will return `NULL`, and the values pointed to by `x`, `y`, and `comp` will remain unchanged.
- **Inputs**:
    - `clbk`: A pointer to an `stbi_io_callbacks` structure that defines the callbacks for reading the image data. Must not be null.
    - `user`: A pointer to user-defined data that will be passed to the callbacks. Can be null if no user data is needed.
    - `x`: A pointer to an integer where the width of the loaded image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the loaded image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of color components in the loaded image will be stored. Can be null if this information is not needed.
    - `req_comp`: An integer specifying the desired number of color components for the output image. Valid values are 1, 2, 3, or 4, corresponding to grayscale, grayscale with alpha, RGB, or RGBA respectively.
- **Output**: Returns a pointer to the loaded image data in 8-bit format. If the loading fails, it returns `NULL`.
- **See also**: [`stbi_load_from_callbacks`](#stbi_load_from_callbacks)  (Implementation)


---
### stbi\_load<!-- {{#callable_declaration:stbi_load}} -->
Loads an image from a file.
- **Description**: This function is used to load an image from a specified file, allowing the user to retrieve the image's dimensions and color components. It must be called with a valid filename pointing to an image file, and the pointers for width, height, and components must be valid and writable. If the file cannot be opened, the function will return a null pointer and set an error message. The `req_comp` parameter allows the user to specify the desired number of color components in the output image, which can be useful for converting images to a specific format.
- **Inputs**:
    - `filename`: A string representing the path to the image file to be loaded. Must not be null and should point to a valid image file.
    - `x`: A pointer to an integer where the width of the image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored. Can be null if this information is not needed.
    - `req_comp`: An integer specifying the number of color components requested in the output image. Valid values are typically 1, 2, 3, or 4, corresponding to grayscale, grayscale with alpha, RGB, or RGBA respectively.
- **Output**: Returns a pointer to the loaded image data as an array of `stbi_uc` (unsigned char). If the image loading fails, it returns a null pointer.
- **See also**: [`stbi_load`](#stbi_load)  (Implementation)


---
### stbi\_load\_from\_file<!-- {{#callable_declaration:stbi_load_from_file}} -->
Loads an image from a file.
- **Description**: This function is used to load an image from a specified file stream, allowing the user to retrieve the image's dimensions and color components. It should be called with a valid file pointer that has been opened for reading. The function will populate the provided pointers for width and height with the image's dimensions, and it can also return the number of color components based on the user's request. If the image loading is successful, the caller is responsible for freeing the returned image data. If the file pointer is invalid or the image cannot be loaded, the function will return `NULL`.
- **Inputs**:
    - `f`: A pointer to a `FILE` object that represents the open file stream from which the image will be loaded. Must not be null and must be opened in a mode that allows reading.
    - `x`: A pointer to an integer where the width of the loaded image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the loaded image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of color components of the loaded image will be stored. Can be null if this information is not needed.
    - `req_comp`: An integer specifying the number of color components requested (e.g., 3 for RGB, 4 for RGBA). Valid values are typically 0 to 4, where 0 means to use the original number of components.
- **Output**: Returns a pointer to the loaded image data as an array of `stbi_uc` (unsigned char). If the image loading fails, it returns `NULL`.
- **See also**: [`stbi_load_from_file`](#stbi_load_from_file)  (Implementation)


---
### stbi\_load\_gif\_from\_memory<!-- {{#callable_declaration:stbi_load_gif_from_memory}} -->
Loads a GIF image from memory.
- **Description**: This function is used to load a GIF image from a memory buffer, which is particularly useful for applications that need to handle GIFs dynamically. It should be called with a valid memory buffer containing the GIF data and the length of that buffer. The function will populate the provided pointers with the image dimensions and other relevant information. It is important to ensure that the `buffer` is not null and that `len` is greater than zero. The function may also modify the output parameters to reflect the image's width, height, and color components. If the GIF is successfully loaded, the function returns a pointer to the image data; otherwise, it returns null.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing the GIF data. Must not be null and must point to a valid memory region.
    - `len`: The length of the buffer in bytes. Must be greater than zero.
    - `delays`: A pointer to an integer array that will be populated with the delays for each frame in the GIF. Can be null if delays are not needed.
    - `x`: A pointer to an integer where the width of the loaded image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the loaded image will be stored. Must not be null.
    - `z`: A pointer to an integer where the number of color channels will be stored. Must not be null.
    - `comp`: A pointer to an integer that specifies the desired number of color components in the output. Must not be null.
    - `req_comp`: An integer specifying the number of color components requested. Valid values are typically 1, 2, 3, or 4.
- **Output**: Returns a pointer to the loaded image data. If the loading fails, it returns null.
- **See also**: [`stbi_load_gif_from_memory`](#stbi_load_gif_from_memory)  (Implementation)


---
### stbi\_convert\_wchar\_to\_utf8<!-- {{#callable_declaration:stbi_convert_wchar_to_utf8}} -->
Converts a wide character string to a UTF-8 encoded string.
- **Description**: This function is used to convert a wide character string (`input`) into a UTF-8 encoded string stored in `buffer`. It is essential to ensure that `buffer` has sufficient space to hold the resulting UTF-8 string, as specified by `bufferlen`. The function should be called when there is a need to handle wide character strings in a UTF-8 format, which is commonly required for compatibility with various text processing systems. If the conversion fails, the function will return a value less than or equal to zero, indicating an error.
- **Inputs**:
    - `buffer`: A pointer to a character array where the resulting UTF-8 string will be stored. Must not be null and should have enough space to accommodate the converted string.
    - `bufferlen`: The size of the `buffer` in bytes. This value must be greater than zero to ensure that there is space for the output.
    - `input`: A pointer to the wide character string to be converted. This parameter can be null, in which case the function will return an error.
- **Output**: Returns the number of bytes written to `buffer`, or a value less than or equal to zero if the conversion fails.
- **See also**: [`stbi_convert_wchar_to_utf8`](#stbi_convert_wchar_to_utf8)  (Implementation)


---
### stbi\_load\_16\_from\_memory<!-- {{#callable_declaration:stbi_load_16_from_memory}} -->
Loads a 16-bit image from memory.
- **Description**: This function is used to load a 16-bit image from a memory buffer, which is particularly useful for applications that need to process high-quality images. It should be called with a valid memory buffer containing image data, and the length of the buffer must be specified. The function will populate the provided pointers for width (`x`), height (`y`), and the number of channels in the file (`channels_in_file`). The `desired_channels` parameter allows the caller to specify how many channels they want in the output image, which can be useful for converting images to a specific format. It is important to ensure that the buffer is not null and that the length is greater than zero; otherwise, the function may return a null pointer, indicating failure to load the image.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing the image data. Must not be null and should point to a valid memory region with image data.
    - `len`: The length of the memory buffer in bytes. Must be greater than zero to ensure there is data to read.
    - `x`: A pointer to an integer where the width of the image will be stored. Caller retains ownership and must ensure it is not null.
    - `y`: A pointer to an integer where the height of the image will be stored. Caller retains ownership and must ensure it is not null.
    - `channels_in_file`: A pointer to an integer where the number of channels in the image file will be stored. Caller retains ownership and must ensure it is not null.
    - `desired_channels`: An integer specifying the number of channels to request in the output image. Valid values are typically 1 (grayscale), 3 (RGB), or 4 (RGBA).
- **Output**: Returns a pointer to the loaded image data in 16-bit format. If the image cannot be loaded, it returns null.
- **See also**: [`stbi_load_16_from_memory`](#stbi_load_16_from_memory)  (Implementation)


---
### stbi\_load\_16\_from\_callbacks<!-- {{#callable_declaration:stbi_load_16_from_callbacks}} -->
Loads a 16-bit image from callbacks.
- **Description**: This function is used to load a 16-bit image from a source defined by the provided callback functions. It is essential to call this function when you need to read image data from a custom source, such as a file or network stream, using the specified `stbi_io_callbacks`. Before calling, ensure that the `x`, `y`, and `channels_in_file` pointers are valid and can store the dimensions and channel information of the loaded image. The function will populate these pointers with the image's width, height, and the number of channels present in the file. If the image loading is successful, it returns a pointer to the loaded image data; otherwise, it returns `NULL` to indicate an error.
- **Inputs**:
    - `clbk`: A pointer to a `stbi_io_callbacks` structure that defines the input source for the image data. Must not be null.
    - `user`: A user-defined pointer that is passed to the callback functions. Can be null if not needed.
    - `x`: A pointer to an integer where the width of the loaded image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the loaded image will be stored. Must not be null.
    - `channels_in_file`: A pointer to an integer where the number of channels in the loaded image will be stored. Can be null if this information is not needed.
    - `desired_channels`: An integer specifying the number of channels desired in the output image. Valid values are typically 1, 2, 3, or 4, corresponding to grayscale, grayscale with alpha, RGB, or RGBA respectively.
- **Output**: Returns a pointer to the loaded 16-bit image data on success, or `NULL` if an error occurs during loading.
- **See also**: [`stbi_load_16_from_callbacks`](#stbi_load_16_from_callbacks)  (Implementation)


---
### stbi\_load\_16<!-- {{#callable_declaration:stbi_load_16}} -->
Loads a 16-bit image from a file.
- **Description**: This function is used to load a 16-bit image from a specified file. It should be called with a valid filename, and it will populate the provided pointers for width, height, and component count of the image. The function will attempt to open the file in binary mode, and if it fails, it will return an error. It is important to ensure that the pointers for width, height, and component count are not null if you wish to retrieve those values. The function also allows specifying the desired number of components to return, which can be useful for converting the image to a specific format.
- **Inputs**:
    - `filename`: A string representing the path to the image file. Must not be null and should point to a valid file.
    - `x`: A pointer to an integer where the width of the image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of components in the image will be stored. Can be null if this information is not needed.
    - `req_comp`: An integer specifying the number of components to request in the output image. Valid values are typically 1, 2, 3, or 4, corresponding to grayscale, grayscale with alpha, RGB, or RGBA respectively.
- **Output**: Returns a pointer to the loaded image data as an array of 16-bit unsigned integers. If the image cannot be loaded, it returns a null pointer and sets an error message.
- **See also**: [`stbi_load_16`](#stbi_load_16)  (Implementation)


---
### stbi\_loadf\_from\_memory<!-- {{#callable_declaration:stbi_loadf_from_memory}} -->
Loads an image from memory and returns a pointer to the pixel data.
- **Description**: This function is used to load an image from a memory buffer, which is useful for applications that need to process images without relying on file I/O. It should be called with a valid memory buffer containing image data and the length of that buffer. The function will populate the provided pointers for width (`x`), height (`y`), and color components (`comp`) of the image. It is important to ensure that the buffer contains valid image data in a supported format. If the image cannot be loaded, the function will return `NULL`, and the values pointed to by `x`, `y`, and `comp` may be modified to reflect the dimensions and components of the image that was attempted to be loaded.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing the image data. Must not be null.
    - `len`: The length of the memory buffer in bytes. Must be greater than zero.
    - `x`: A pointer to an integer where the width of the image will be stored. Caller retains ownership.
    - `y`: A pointer to an integer where the height of the image will be stored. Caller retains ownership.
    - `comp`: A pointer to an integer where the number of color components will be stored. Caller retains ownership.
    - `req_comp`: The number of color components requested in the output (e.g., 3 for RGB, 4 for RGBA). Can be 0 to request the default number of components.
- **Output**: Returns a pointer to the pixel data of the loaded image. If the image cannot be loaded, returns `NULL`.
- **See also**: [`stbi_loadf_from_memory`](#stbi_loadf_from_memory)  (Implementation)


---
### stbi\_loadf\_from\_callbacks<!-- {{#callable_declaration:stbi_loadf_from_callbacks}} -->
Loads an image from callbacks.
- **Description**: This function is used to load an image from a custom input source defined by the provided callbacks. It is essential to call this function when you need to read image data from a non-standard source, such as a network stream or a custom file format. Before calling, ensure that the `clbk` parameter is properly initialized and points to valid callback functions. The function will populate the width and height of the image in the `x` and `y` parameters, respectively, and can also return the number of color components in `comp`. If the image cannot be loaded, the function will return `NULL`.
- **Inputs**:
    - `clbk`: A pointer to an `stbi_io_callbacks` structure that contains the callback functions for reading the image data. Must not be null.
    - `user`: A pointer to user-defined data that will be passed to the callback functions. Can be null if not needed.
    - `x`: A pointer to an integer where the width of the loaded image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the loaded image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of color components will be stored. Can be null if the component count is not needed.
    - `req_comp`: An integer specifying the desired number of color components for the output image. Valid values are typically 1, 3, or 4, corresponding to grayscale, RGB, and RGBA formats.
- **Output**: Returns a pointer to the loaded image data as an array of floats, or `NULL` if the image could not be loaded.
- **See also**: [`stbi_loadf_from_callbacks`](#stbi_loadf_from_callbacks)  (Implementation)


---
### stbi\_loadf<!-- {{#callable_declaration:stbi_loadf}} -->
Loads an image from a file and returns its pixel data as floating-point values.
- **Description**: This function is used to load an image from a specified file, returning the pixel data in a format suitable for further processing. It is essential to call this function with a valid filename pointing to an image file. The function will attempt to open the file in binary mode, and if it fails, it will return an error. The dimensions of the image can be retrieved through the provided pointers, and the function allows for specifying the desired number of color components to be returned. It is important to ensure that the pointers for width, height, and components are valid and can be modified by the function.
- **Inputs**:
    - `filename`: A string representing the path to the image file. Must not be null and should point to a valid image file.
    - `x`: A pointer to an integer where the width of the image will be stored. Caller retains ownership and must ensure it is not null.
    - `y`: A pointer to an integer where the height of the image will be stored. Caller retains ownership and must ensure it is not null.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored. Caller retains ownership and must ensure it is not null.
    - `req_comp`: An integer specifying the number of color components to request. Valid values are typically 1, 2, 3, or 4, corresponding to grayscale, grayscale with alpha, RGB, or RGBA respectively.
- **Output**: Returns a pointer to the pixel data of the image as an array of floating-point values. If the image cannot be loaded, it returns a null pointer.
- **See also**: [`stbi_loadf`](#stbi_loadf)  (Implementation)


---
### stbi\_loadf\_from\_file<!-- {{#callable_declaration:stbi_loadf_from_file}} -->
Loads an image from a file as floating-point values.
- **Description**: This function is used to load an image from a specified file stream into memory, returning the image data as an array of floating-point values. It is essential to call this function with a valid file pointer that has been opened for reading. The parameters `x` and `y` will be populated with the dimensions of the loaded image, while `comp` will indicate the number of color components in the image. The `req_comp` parameter allows the caller to specify the desired number of components in the output (e.g., 1 for grayscale, 3 for RGB, 4 for RGBA). If the image cannot be loaded, the function will return `NULL`, and the values pointed to by `x`, `y`, and `comp` will remain unchanged.
- **Inputs**:
    - `f`: A pointer to a `FILE` object that represents the open file stream from which the image will be loaded. Must not be null and must be opened in a mode that allows reading.
    - `x`: A pointer to an integer where the width of the loaded image will be stored. Caller retains ownership and must ensure it is not null.
    - `y`: A pointer to an integer where the height of the loaded image will be stored. Caller retains ownership and must ensure it is not null.
    - `comp`: A pointer to an integer where the number of color components in the loaded image will be stored. Caller retains ownership and must ensure it is not null.
    - `req_comp`: An integer specifying the number of color components desired in the output. Valid values are typically 1, 3, or 4. If the value is invalid, the function will attempt to load the image with its original number of components.
- **Output**: Returns a pointer to the loaded image data as an array of floating-point values. If the image loading fails, it returns `NULL`.
- **See also**: [`stbi_loadf_from_file`](#stbi_loadf_from_file)  (Implementation)


---
### stbi\_hdr\_to\_ldr\_gamma<!-- {{#callable_declaration:stbi_hdr_to_ldr_gamma}} -->
Sets the gamma correction value for HDR to LDR conversion.
- **Description**: This function is used to specify the gamma correction factor that will be applied during the conversion of high dynamic range (HDR) images to low dynamic range (LDR) images. It should be called before any HDR to LDR conversion functions to ensure that the specified gamma value is used. The gamma value must be a positive float; passing a non-positive value will result in undefined behavior. This function modifies a global state that affects subsequent image processing.
- **Inputs**:
    - `gamma`: A float representing the gamma correction factor. Must be a positive value. Passing a non-positive value may lead to undefined behavior.
- **Output**: None
- **See also**: [`stbi_hdr_to_ldr_gamma`](#stbi_hdr_to_ldr_gamma)  (Implementation)


---
### stbi\_hdr\_to\_ldr\_scale<!-- {{#callable_declaration:stbi_hdr_to_ldr_scale}} -->
Sets the scale factor for converting HDR to LDR.
- **Description**: This function is used to define the scale factor that will be applied when converting high dynamic range (HDR) images to low dynamic range (LDR) images. It should be called before any HDR to LDR conversion functions to ensure that the specified scale is applied correctly. The scale value must be a positive float; passing a non-positive value will result in undefined behavior. It is important to ensure that the scale is set appropriately to avoid issues during image processing.
- **Inputs**:
    - `scale`: A positive float representing the scale factor for HDR to LDR conversion. Must not be zero or negative; otherwise, the behavior is undefined.
- **Output**: None
- **See also**: [`stbi_hdr_to_ldr_scale`](#stbi_hdr_to_ldr_scale)  (Implementation)


---
### stbi\_ldr\_to\_hdr\_gamma<!-- {{#callable_declaration:stbi_ldr_to_hdr_gamma}} -->
Sets the gamma correction value for converting LDR to HDR.
- **Description**: This function is used to specify the gamma correction value that will be applied when converting low dynamic range (LDR) images to high dynamic range (HDR) format. It should be called before any LDR to HDR conversion functions to ensure that the specified gamma value is used. The gamma value affects the brightness and contrast of the resulting HDR image, so it is important to choose an appropriate value based on the source image characteristics. There are no specific constraints on the gamma value, but it is typically a positive float.
- **Inputs**:
    - `gamma`: A float representing the gamma correction value. It should be a positive number, as negative or zero values may lead to undefined behavior. The caller retains ownership of the value, and it must be set before any LDR to HDR conversion is performed.
- **Output**: None
- **See also**: [`stbi_ldr_to_hdr_gamma`](#stbi_ldr_to_hdr_gamma)  (Implementation)


---
### stbi\_ldr\_to\_hdr\_scale<!-- {{#callable_declaration:stbi_ldr_to_hdr_scale}} -->
Sets the scale factor for converting LDR to HDR.
- **Description**: This function is used to define the scale factor that will be applied when converting low dynamic range (LDR) images to high dynamic range (HDR) format. It should be called before any LDR to HDR conversion functions to ensure that the specified scale is applied correctly. The scale factor can be any floating-point value, including zero or negative values, which may affect the resulting HDR image. It is important to note that calling this function with an invalid scale may lead to unexpected results in the conversion process.
- **Inputs**:
    - `scale`: A floating-point value representing the scale factor for LDR to HDR conversion. It can be any valid float, including zero or negative values. The caller retains ownership of the value, and it must be set before any conversion functions are called.
- **Output**: None
- **See also**: [`stbi_ldr_to_hdr_scale`](#stbi_ldr_to_hdr_scale)  (Implementation)


---
### stbi\_is\_hdr\_from\_callbacks<!-- {{#callable_declaration:stbi_is_hdr_from_callbacks}} -->
Determines if the data from callbacks represents an HDR image.
- **Description**: This function is used to check if the data provided through the specified callbacks corresponds to a High Dynamic Range (HDR) image. It should be called when you need to ascertain the format of image data before processing it further. The `clbk` parameter must point to a valid `stbi_io_callbacks` structure, which defines the methods for reading the image data. The `user` parameter is a user-defined pointer that can be used to pass additional context to the callbacks. If the HDR functionality is disabled during compilation, the function will ignore the parameters and return 0, indicating that it cannot determine the image type.
- **Inputs**:
    - `clbk`: A pointer to a `stbi_io_callbacks` structure that defines the methods for reading image data. Must not be null.
    - `user`: A user-defined pointer that can be used to pass additional context to the callbacks. Can be null.
- **Output**: Returns a non-zero value if the data represents an HDR image; otherwise, it returns 0.
- **See also**: [`stbi_is_hdr_from_callbacks`](#stbi_is_hdr_from_callbacks)  (Implementation)


---
### stbi\_is\_hdr\_from\_memory<!-- {{#callable_declaration:stbi_is_hdr_from_memory}} -->
Determines if the provided memory buffer contains HDR image data.
- **Description**: This function is used to check whether a given memory buffer contains High Dynamic Range (HDR) image data. It should be called with a valid buffer and its length, and it is particularly useful for applications that need to differentiate between HDR and non-HDR images before processing. The function will return a non-zero value if the buffer contains HDR data, and zero otherwise. It is important to ensure that the buffer is not null and that the length is greater than zero to avoid undefined behavior.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing image data. Must not be null.
    - `len`: The length of the buffer in bytes. Must be greater than zero.
- **Output**: Returns a non-zero value if the buffer contains HDR image data, and zero if it does not.
- **See also**: [`stbi_is_hdr_from_memory`](#stbi_is_hdr_from_memory)  (Implementation)


---
### stbi\_is\_hdr<!-- {{#callable_declaration:stbi_is_hdr}} -->
Checks if a file is an HDR image.
- **Description**: This function is used to determine whether a specified file is in the High Dynamic Range (HDR) image format. It should be called with a valid filename that points to a file on the filesystem. The function attempts to open the file in binary read mode, and if successful, it checks the file's format. If the file cannot be opened, the function will return 0, indicating that it could not determine the HDR status. It is important to ensure that the filename provided is valid and accessible; otherwise, the function will not be able to perform its check.
- **Inputs**:
    - `filename`: A pointer to a null-terminated string representing the path to the file. Must not be null and should point to a valid file path. If the file does not exist or cannot be opened, the function will return 0.
- **Output**: Returns a non-zero value if the file is an HDR image, and 0 if it is not or if the file could not be opened.
- **See also**: [`stbi_is_hdr`](#stbi_is_hdr)  (Implementation)


---
### stbi\_is\_hdr\_from\_file<!-- {{#callable_declaration:stbi_is_hdr_from_file}} -->
Determines if a file contains HDR image data.
- **Description**: This function is used to check if the specified file stream contains High Dynamic Range (HDR) image data. It should be called with a valid file pointer that has been opened for reading. The function will seek to the current position in the file, perform the HDR check, and then return to the original position. If the file pointer is null or if HDR support is disabled during compilation, the function will return 0, indicating that the file does not contain HDR data.
- **Inputs**:
    - `f`: A pointer to a `FILE` object that represents the file to be checked. Must not be null and should be opened in read mode. If the pointer is null or if HDR support is not enabled, the function will return 0.
- **Output**: Returns 1 if the file contains HDR image data, or 0 if it does not or if HDR support is disabled.
- **See also**: [`stbi_is_hdr_from_file`](#stbi_is_hdr_from_file)  (Implementation)


---
### stbi\_failure\_reason<!-- {{#callable_declaration:stbi_failure_reason}} -->
Retrieves the reason for the last failure.
- **Description**: This function is used to obtain a human-readable string that describes the reason for the last failure encountered by the library. It is particularly useful for debugging purposes, allowing developers to understand what went wrong during the last operation. This function can be called at any time after a failure has occurred, and it will return a valid string pointer as long as a failure reason has been set. If no failure has occurred, the returned string may be `NULL`.
- **Inputs**: None
- **Output**: Returns a pointer to a string that describes the failure reason. If no failure has occurred, the return value may be `NULL`.
- **See also**: [`stbi_failure_reason`](#stbi_failure_reason)  (Implementation)


---
### stbi\_image\_free<!-- {{#callable_declaration:stbi_image_free}} -->
Frees memory allocated for an image.
- **Description**: This function is used to release memory that was previously allocated for an image by the `stbi_load` function. It is important to call this function to avoid memory leaks after you are done using the image data. The caller must ensure that the pointer passed to this function is valid and was obtained from a successful call to `stbi_load`. Passing a null pointer is safe and will have no effect.
- **Inputs**:
    - `retval_from_stbi_load`: A pointer to the image data that needs to be freed. This pointer must be a valid memory address returned by `stbi_load`. If the pointer is null, the function will safely do nothing.
- **Output**: None
- **See also**: [`stbi_image_free`](#stbi_image_free)  (Implementation)


---
### stbi\_info\_from\_memory<!-- {{#callable_declaration:stbi_info_from_memory}} -->
Retrieves image information from a memory buffer.
- **Description**: This function is used to extract image dimensions and color components from an image stored in memory. It should be called with a valid memory buffer containing image data, and the length of that buffer. The function populates the provided pointers with the width and height of the image, as well as the number of color components. It is important to ensure that the pointers for width, height, and components are not null before calling this function. If the provided buffer is invalid or does not contain a recognizable image format, the function will return an error code.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing the image data. Must not be null.
    - `len`: The length of the memory buffer in bytes. Must be greater than zero.
    - `x`: A pointer to an integer where the width of the image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of color components will be stored. Must not be null.
- **Output**: Returns a non-zero value if the image information was successfully retrieved, or zero if an error occurred.
- **See also**: [`stbi_info_from_memory`](#stbi_info_from_memory)  (Implementation)


---
### stbi\_info\_from\_callbacks<!-- {{#callable_declaration:stbi_info_from_callbacks}} -->
Retrieves image information from a callback-based input.
- **Description**: This function is used to obtain the dimensions and color components of an image from a source that provides data through callbacks. It should be called when you need to read image metadata without loading the entire image into memory. The `stbi_io_callbacks` structure must be properly initialized and passed to the function, along with a user-defined pointer for context. The function will populate the provided pointers for width, height, and number of components with the corresponding values from the image. It is important to ensure that the pointers for width, height, and components are not null, as the function expects valid memory addresses to write the output.
- **Inputs**:
    - `c`: A pointer to a `stbi_io_callbacks` structure that defines the input callbacks for reading the image data. Must not be null.
    - `user`: A user-defined pointer that is passed to the callbacks. Can be null if not needed.
    - `x`: A pointer to an integer where the width of the image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of color components will be stored. Must not be null.
- **Output**: Returns a non-zero value if the image information was successfully retrieved, or zero if an error occurred.
- **See also**: [`stbi_info_from_callbacks`](#stbi_info_from_callbacks)  (Implementation)


---
### stbi\_is\_16\_bit\_from\_memory<!-- {{#callable_declaration:stbi_is_16_bit_from_memory}} -->
Determines if the data in memory represents a 16-bit image.
- **Description**: This function is used to check whether the provided memory buffer contains a 16-bit image format. It should be called with a valid buffer and its length, which must be greater than zero. The function will analyze the data in the buffer and return a non-zero value if it identifies the format as 16-bit, or zero otherwise. It is important to ensure that the buffer is not null and that the length is appropriate to avoid undefined behavior.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer containing image data. Must not be null.
    - `len`: The length of the buffer in bytes. Must be greater than zero.
- **Output**: Returns a non-zero value if the buffer contains a 16-bit image format; otherwise, it returns zero.
- **See also**: [`stbi_is_16_bit_from_memory`](#stbi_is_16_bit_from_memory)  (Implementation)


---
### stbi\_is\_16\_bit\_from\_callbacks<!-- {{#callable_declaration:stbi_is_16_bit_from_callbacks}} -->
Determines if the image data is 16-bit from the provided callbacks.
- **Description**: This function is used to check if the image data being read through the provided callbacks is in 16-bit format. It should be called when you need to ascertain the bit depth of the image before processing it further. The function requires a valid `stbi_io_callbacks` structure to be passed, which contains the necessary callbacks for reading the image data. It is important to ensure that the callbacks are properly set up and that the user data is valid, as invalid inputs may lead to undefined behavior.
- **Inputs**:
    - `c`: A pointer to a `stbi_io_callbacks` structure that contains the callbacks for reading the image data. Must not be null.
    - `user`: A pointer to user-defined data that will be passed to the callbacks. This can be null if no user data is needed.
- **Output**: Returns a non-zero value if the image data is in 16-bit format, and zero otherwise.
- **See also**: [`stbi_is_16_bit_from_callbacks`](#stbi_is_16_bit_from_callbacks)  (Implementation)


---
### stbi\_info<!-- {{#callable_declaration:stbi_info}} -->
Retrieves image information from a file.
- **Description**: This function is used to obtain the dimensions and color components of an image file specified by its filename. It must be called with a valid filename pointing to an image file, and the caller should provide pointers for the width, height, and number of color components, which will be populated with the corresponding values. If the file cannot be opened, the function will return an error code, and the output parameters will remain unchanged. It is important to ensure that the provided pointers are valid and that the filename points to a readable image file.
- **Inputs**:
    - `filename`: A string representing the path to the image file. Must not be null and should point to a valid image file.
    - `x`: A pointer to an integer where the width of the image will be stored. Caller retains ownership and must ensure it is not null.
    - `y`: A pointer to an integer where the height of the image will be stored. Caller retains ownership and must ensure it is not null.
    - `comp`: A pointer to an integer where the number of color components will be stored. Caller retains ownership and must ensure it is not null.
- **Output**: Returns a non-zero value on success, indicating that the image information was successfully retrieved. If the function fails to open the file, it returns a specific error code.
- **See also**: [`stbi_info`](#stbi_info)  (Implementation)


---
### stbi\_info\_from\_file<!-- {{#callable_declaration:stbi_info_from_file}} -->
Retrieves image information from a file.
- **Description**: This function is used to obtain the dimensions and number of components of an image from a file stream. It should be called with a valid file pointer that has been opened for reading. The function will read the image data to extract the width, height, and color components, which are then stored in the provided pointers. It is important to ensure that the pointers for width, height, and components are not null before calling this function. If the file does not contain valid image data, the function will return an error code, and the output parameters will remain unchanged.
- **Inputs**:
    - `f`: A pointer to a `FILE` object that represents the open file stream from which to read the image data. The file must be opened in a mode that allows reading. Must not be null.
    - `x`: A pointer to an integer where the width of the image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of color components in the image will be stored. Must not be null.
- **Output**: Returns a non-zero value if the image information was successfully retrieved; otherwise, it returns zero to indicate an error.
- **See also**: [`stbi_info_from_file`](#stbi_info_from_file)  (Implementation)


---
### stbi\_is\_16\_bit<!-- {{#callable_declaration:stbi_is_16_bit}} -->
Checks if a file is a 16-bit image.
- **Description**: This function is used to determine whether the specified image file is in a 16-bit format. It should be called with a valid filename that points to an image file. The function attempts to open the file in binary mode and will return an error if the file cannot be opened. It is important to ensure that the filename provided is correct and accessible, as failure to do so will result in an error message being generated. The function will close the file after checking its format.
- **Inputs**:
    - `filename`: A pointer to a null-terminated string representing the path to the image file. Must not be null and should point to a valid file path. If the file cannot be opened, the function will return an error.
- **Output**: Returns a non-zero value if the file is a 16-bit image, or zero if it is not.
- **See also**: [`stbi_is_16_bit`](#stbi_is_16_bit)  (Implementation)


---
### stbi\_is\_16\_bit\_from\_file<!-- {{#callable_declaration:stbi_is_16_bit_from_file}} -->
Determines if a file contains 16-bit image data.
- **Description**: This function is used to check whether the image data in a specified file is in 16-bit format. It should be called with a valid file pointer that has been opened for reading. The function will seek to the beginning of the file to read the necessary data and then return to the original position in the file. It is important to ensure that the file pointer is not null and points to a valid file; otherwise, the behavior is undefined.
- **Inputs**:
    - `f`: A pointer to a `FILE` object that represents the file to be checked. Must not be null and should be opened for reading. If the file pointer is invalid or the file cannot be read, the function's behavior is undefined.
- **Output**: Returns a non-zero value if the file contains 16-bit image data, and zero otherwise.
- **See also**: [`stbi_is_16_bit_from_file`](#stbi_is_16_bit_from_file)  (Implementation)


---
### stbi\_set\_unpremultiply\_on\_load<!-- {{#callable_declaration:stbi_set_unpremultiply_on_load}} -->
Sets the unpremultiply flag for image loading.
- **Description**: This function configures whether images should be unpremultiplied upon loading. It should be called before any image loading functions to ensure that the desired behavior is applied consistently. The flag indicates whether the alpha channel should be processed to remove premultiplication, which can be important for accurate color representation in images with transparency. It is essential to set this flag according to the needs of your application, as it affects how images are rendered after loading.
- **Inputs**:
    - `flag_true_if_should_unpremultiply`: An integer flag that determines if unpremultiplication should occur (non-zero for true, zero for false). Must be a valid integer value. The caller retains ownership of this value, and it should be set before loading any images to take effect.
- **Output**: None
- **See also**: [`stbi_set_unpremultiply_on_load`](#stbi_set_unpremultiply_on_load)  (Implementation)


---
### stbi\_convert\_iphone\_png\_to\_rgb<!-- {{#callable_declaration:stbi_convert_iphone_png_to_rgb}} -->
Sets the global flag for converting iPhone PNG images to RGB.
- **Description**: This function is used to specify whether iPhone PNG images should be converted to RGB format when they are loaded. It should be called before loading any images to ensure that the conversion behavior is set according to the user's needs. The function modifies a global setting, which affects all subsequent image loading operations. It is important to note that the flag should be set appropriately based on the desired output format, as it will influence how images are processed.
- **Inputs**:
    - `flag_true_if_should_convert`: An integer flag indicating whether to convert iPhone PNG images to RGB. A non-zero value enables conversion, while zero disables it. The caller retains ownership of this value, and it must be a valid integer. Invalid values are handled by simply setting the flag accordingly, with no additional error handling.
- **Output**: None
- **See also**: [`stbi_convert_iphone_png_to_rgb`](#stbi_convert_iphone_png_to_rgb)  (Implementation)


---
### stbi\_set\_flip\_vertically\_on\_load<!-- {{#callable_declaration:stbi_set_flip_vertically_on_load}} -->
Sets the vertical flip behavior for image loading.
- **Description**: This function configures whether images should be flipped vertically upon loading. It should be called before any image loading functions to ensure that the desired flipping behavior is applied. The `flag_true_if_should_flip` parameter determines if the images will be flipped; setting it to a non-zero value enables flipping, while zero disables it. It is important to note that this setting affects all subsequent image loads until it is changed again.
- **Inputs**:
    - `flag_true_if_should_flip`: An integer flag indicating the vertical flip behavior. A non-zero value enables vertical flipping, while zero disables it. The caller retains ownership of this value, and it must be a valid integer. Invalid values are handled by simply treating them as zero.
- **Output**: None
- **See also**: [`stbi_set_flip_vertically_on_load`](#stbi_set_flip_vertically_on_load)  (Implementation)


---
### stbi\_set\_unpremultiply\_on\_load\_thread<!-- {{#callable_declaration:stbi_set_unpremultiply_on_load_thread}} -->
Sets the unpremultiply flag for image loading.
- **Description**: This function configures whether images should be unpremultiplied when loaded in a separate thread. It should be called before any image loading operations are performed to ensure that the setting is applied correctly. The flag indicates whether the unpremultiply operation should be applied, which can affect the appearance of images with alpha transparency. It is important to note that this setting is global and will affect all subsequent image loads until changed again.
- **Inputs**:
    - `flag_true_if_should_unpremultiply`: A boolean flag indicating whether to unpremultiply the loaded images. It accepts values of 0 (false) or 1 (true). The caller retains ownership of this value, and it must be set before any image loading occurs. Invalid values outside of this range will be ignored.
- **Output**: None
- **See also**: [`stbi_set_unpremultiply_on_load_thread`](#stbi_set_unpremultiply_on_load_thread)  (Implementation)


---
### stbi\_convert\_iphone\_png\_to\_rgb\_thread<!-- {{#callable_declaration:stbi_convert_iphone_png_to_rgb_thread}} -->
Sets the conversion flag for iPhone PNG images.
- **Description**: This function is used to specify whether iPhone PNG images should be converted to RGB format during decoding. It should be called before any image decoding functions that may process iPhone PNG files. The conversion is controlled by the `flag_true_if_should_convert` parameter, which determines if the conversion should take place. It is important to note that this function does not return a value and does not modify any input parameters.
- **Inputs**:
    - `flag_true_if_should_convert`: A boolean flag indicating whether to convert iPhone PNG images to RGB format. It accepts any integer value, where a non-zero value indicates conversion should occur, and zero indicates no conversion. The caller retains ownership of this value, and it must be set before decoding images.
- **Output**: None
- **See also**: [`stbi_convert_iphone_png_to_rgb_thread`](#stbi_convert_iphone_png_to_rgb_thread)  (Implementation)


---
### stbi\_set\_flip\_vertically\_on\_load\_thread<!-- {{#callable_declaration:stbi_set_flip_vertically_on_load_thread}} -->
Sets the vertical flip behavior for image loading.
- **Description**: This function configures whether images should be flipped vertically when loaded. It should be called before loading any images to ensure the desired flipping behavior is applied. The function modifies a global setting that affects all subsequent image loads, making it useful for adjusting the orientation of images based on the application's requirements.
- **Inputs**:
    - `flag_true_if_should_flip`: An integer flag indicating whether images should be flipped vertically (non-zero value for true, zero for false). Must be a valid integer. The caller retains ownership of this value, and it can be set to any integer; however, only the zero or non-zero state is considered.
- **Output**: None
- **See also**: [`stbi_set_flip_vertically_on_load_thread`](#stbi_set_flip_vertically_on_load_thread)  (Implementation)


---
### stbi\_zlib\_decode\_malloc\_guesssize<!-- {{#callable_declaration:stbi_zlib_decode_malloc_guesssize}} -->
Decodes a zlib-compressed buffer into a dynamically allocated memory block.
- **Description**: This function is used to decode a zlib-compressed data buffer, allocating memory for the output based on an initial size provided by the caller. It is essential to call this function with a valid compressed buffer and its length, and it will allocate memory to hold the decompressed data. If the decompression is successful, the size of the decompressed data can be retrieved through the `outlen` parameter. If the function fails to allocate memory or the decompression fails, it will return `NULL` and the `outlen` will not be modified. The caller is responsible for freeing the allocated memory when it is no longer needed.
- **Inputs**:
    - `buffer`: A pointer to the zlib-compressed data buffer. Must not be null.
    - `len`: The length of the compressed data buffer in bytes. Must be a positive integer.
    - `initial_size`: The initial size for the allocated output buffer. Must be a positive integer.
    - `outlen`: A pointer to an integer where the size of the decompressed data will be stored. Can be null if the size is not needed.
- **Output**: Returns a pointer to the dynamically allocated memory containing the decompressed data. If the decompression fails or memory allocation fails, it returns `NULL`.
- **See also**: [`stbi_zlib_decode_malloc_guesssize`](#stbi_zlib_decode_malloc_guesssize)  (Implementation)


---
### stbi\_zlib\_decode\_malloc\_guesssize\_headerflag<!-- {{#callable_declaration:stbi_zlib_decode_malloc_guesssize_headerflag}} -->
Decodes a zlib-compressed buffer into a dynamically allocated memory block.
- **Description**: This function is used to decode a zlib-compressed data buffer, allocating memory for the output based on an initial size provided by the caller. It is essential to call this function with a valid compressed buffer and its length, and it can optionally parse the zlib header if specified. The caller must ensure that the `initial_size` is sufficient to hold the decompressed data, as the function will allocate memory accordingly. If the decoding is successful, the output length can be retrieved through the `outlen` parameter. If the function fails, it will return `NULL`, and any allocated memory will be freed.
- **Inputs**:
    - `buffer`: A pointer to the zlib-compressed data buffer. Must not be null.
    - `len`: The length of the compressed data buffer in bytes. Must be greater than zero.
    - `initial_size`: The initial size for the allocated output buffer. Must be greater than zero.
    - `outlen`: A pointer to an integer where the output length will be stored. Can be null if the length is not needed.
    - `parse_header`: An integer flag indicating whether to parse the zlib header. Valid values are 0 (do not parse) or 1 (parse).
- **Output**: Returns a pointer to the dynamically allocated buffer containing the decompressed data, or `NULL` if the decoding fails. If successful, the length of the decompressed data is stored in the variable pointed to by `outlen`, if it is not null.
- **See also**: [`stbi_zlib_decode_malloc_guesssize_headerflag`](#stbi_zlib_decode_malloc_guesssize_headerflag)  (Implementation)


---
### stbi\_zlib\_decode\_malloc<!-- {{#callable_declaration:stbi_zlib_decode_malloc}} -->
Decodes a zlib-compressed buffer into a newly allocated memory block.
- **Description**: This function is used to decode a zlib-compressed data buffer into a dynamically allocated memory block. It is essential to call this function with a valid compressed data buffer and its length. The function will allocate memory for the decompressed data, which the caller is responsible for freeing. If the input buffer is invalid or the decompression fails, the function may return a null pointer, and the output length will be set to zero. It is important to ensure that the `outlen` pointer is not null, as it will be used to store the size of the decompressed data.
- **Inputs**:
    - `buffer`: A pointer to the zlib-compressed data buffer. Must not be null.
    - `len`: The length of the compressed data buffer in bytes. Must be a non-negative integer.
    - `outlen`: A pointer to an integer where the function will store the length of the decompressed data. Must not be null.
- **Output**: Returns a pointer to the newly allocated memory containing the decompressed data. If the decompression fails, returns null.
- **See also**: [`stbi_zlib_decode_malloc`](#stbi_zlib_decode_malloc)  (Implementation)


---
### stbi\_zlib\_decode\_buffer<!-- {{#callable_declaration:stbi_zlib_decode_buffer}} -->
Decodes a zlib-compressed buffer.
- **Description**: This function is used to decode data that has been compressed using the zlib format. It should be called when you have a valid zlib-compressed input buffer and you want to obtain the decompressed output. The output buffer must be large enough to hold the decompressed data, and the function will return the size of the decompressed data if successful. If the input buffer is invalid or the output buffer is insufficient, the function will return -1, indicating a failure. It is important to ensure that the input buffer is not null and that the output buffer is allocated with a size greater than zero.
- **Inputs**:
    - `obuffer`: A pointer to the output buffer where the decompressed data will be stored. Must not be null and must have sufficient size to hold the decompressed data.
    - `olen`: The size of the output buffer in bytes. Must be greater than zero.
    - `ibuffer`: A pointer to the input buffer containing the zlib-compressed data. Must not be null.
    - `ilen`: The size of the input buffer in bytes. Must be greater than zero.
- **Output**: Returns the size of the decompressed data in bytes if successful, or -1 if an error occurs.
- **See also**: [`stbi_zlib_decode_buffer`](#stbi_zlib_decode_buffer)  (Implementation)


---
### stbi\_zlib\_decode\_noheader\_malloc<!-- {{#callable_declaration:stbi_zlib_decode_noheader_malloc}} -->
Allocates memory and decodes a zlib-compressed buffer.
- **Description**: This function is used to decode a zlib-compressed data buffer without a header. It should be called when you have a valid zlib-compressed input and need to retrieve the decompressed data. The function allocates memory for the output, which must be freed by the caller. If the input buffer is invalid or the decompression fails, the function will return `NULL`. The `outlen` parameter can be provided to receive the size of the decompressed data, which is useful for managing the allocated memory.
- **Inputs**:
    - `buffer`: A pointer to the zlib-compressed data. Must not be null and should point to a valid memory region containing the compressed data.
    - `len`: The length of the compressed data in bytes. Must be a positive integer.
    - `outlen`: A pointer to an integer where the size of the decompressed data will be stored. Can be null if the size is not needed.
- **Output**: Returns a pointer to the newly allocated memory containing the decompressed data. If decompression fails or if memory allocation fails, returns `NULL`.
- **See also**: [`stbi_zlib_decode_noheader_malloc`](#stbi_zlib_decode_noheader_malloc)  (Implementation)


---
### stbi\_zlib\_decode\_noheader\_buffer<!-- {{#callable_declaration:stbi_zlib_decode_noheader_buffer}} -->
Decodes a zlib-compressed buffer without a header.
- **Description**: This function is used to decode a zlib-compressed data buffer into an output buffer. It should be called when you have a zlib-compressed input buffer and you want to obtain the decompressed data. The output buffer must be large enough to hold the decompressed data, and the function will return the number of bytes written to the output buffer. If the input buffer is invalid or if the decompression fails, the function will return -1, indicating an error. It is important to ensure that the input length is greater than zero and that the output buffer is not null.
- **Inputs**:
    - `obuffer`: A pointer to the output buffer where the decompressed data will be written. Must not be null and must be large enough to hold the decompressed data.
    - `olen`: The size of the output buffer in bytes. Must be greater than zero to ensure there is space for the decompressed data.
    - `ibuffer`: A pointer to the input buffer containing the zlib-compressed data. Must not be null and must point to a valid memory region.
    - `ilen`: The size of the input buffer in bytes. Must be greater than zero to indicate that there is data to decompress.
- **Output**: Returns the number of bytes written to the output buffer upon successful decompression, or -1 if an error occurs.
- **See also**: [`stbi_zlib_decode_noheader_buffer`](#stbi_zlib_decode_noheader_buffer)  (Implementation)


---
### stbi\_\_refill\_buffer<!-- {{#callable_declaration:stbi__refill_buffer}} -->
Refills the buffer with data from a source.
- **Description**: This function is used to refill the internal buffer with data from a specified input source. It should be called when the current buffer is exhausted and more data is needed for processing. The function reads data into the buffer and updates the pointers that track the start and end of the image data. If the end of the input source is reached, it ensures that the buffer is safely terminated. It is important to ensure that the context has been properly initialized before calling this function.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure that holds the state of the buffer and the input source. Must not be null.
- **Output**: None
- **See also**: [`stbi__refill_buffer`](#stbi__refill_buffer)  (Implementation)


---
### stbi\_\_jpeg\_test<!-- {{#callable_declaration:stbi__jpeg_test}} -->
Tests if the provided context contains a JPEG image.
- **Description**: This function is used to determine if the data in the provided context represents a JPEG image. It should be called with a valid `stbi__context` that has been properly initialized and points to a data source. The function allocates memory for internal processing, so it is important to ensure that the system has sufficient memory available. If the context does not contain a JPEG image, the function will return a negative value. The function also resets the context to its initial state after the test is performed.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure that must be initialized and point to a valid data source. Must not be null. If the context is invalid or not initialized, the function may return an error.
- **Output**: Returns a non-negative value if the context contains a JPEG image; otherwise, it returns a negative value indicating that the data is not a JPEG.
- **See also**: [`stbi__jpeg_test`](#stbi__jpeg_test)  (Implementation)


---
### stbi\_\_jpeg\_load<!-- {{#callable_declaration:stbi__jpeg_load}} -->
Loads a JPEG image from a given context.
- **Description**: This function is used to load a JPEG image from the provided context, returning the image data in a dynamically allocated buffer. It should be called when you need to decode a JPEG image, and it will fill in the dimensions and color components of the image if the corresponding pointers are provided. The function allocates memory for internal processing, which is freed before returning the image data. If memory allocation fails, it returns a null pointer and sets an error message. Ensure that the context is properly initialized before calling this function.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that contains the JPEG data to be loaded. Must not be null.
    - `x`: A pointer to an integer where the width of the loaded image will be stored. Can be null if the width is not needed.
    - `y`: A pointer to an integer where the height of the loaded image will be stored. Can be null if the height is not needed.
    - `comp`: A pointer to an integer where the number of color components in the loaded image will be stored. Can be null if the component count is not needed.
    - `req_comp`: An integer specifying the number of color components requested in the output image. Valid values are typically 1, 3, or 4, corresponding to grayscale, RGB, and RGBA formats respectively.
    - `ri`: A pointer to a `stbi__result_info` structure for additional result information. This parameter is not used in the current implementation and can be passed as null.
- **Output**: Returns a pointer to the loaded image data in a dynamically allocated buffer. If the image cannot be loaded due to an error, it returns null.
- **See also**: [`stbi__jpeg_load`](#stbi__jpeg_load)  (Implementation)


---
### stbi\_\_jpeg\_info<!-- {{#callable_declaration:stbi__jpeg_info}} -->
Retrieves JPEG image information.
- **Description**: This function is used to obtain the dimensions and number of components of a JPEG image from a given context. It should be called after initializing the context with a valid JPEG image. The function allocates memory for internal processing, and if memory allocation fails, it will return an error. The caller must ensure that the provided pointers for width, height, and component count are valid and can be modified.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that contains the JPEG image data. Must not be null and should point to a valid initialized context.
    - `x`: A pointer to an integer where the width of the image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of color components will be stored. Must not be null.
- **Output**: Returns 1 on success, indicating that the image information was successfully retrieved, or 0 on failure, indicating an error occurred.
- **See also**: [`stbi__jpeg_info`](#stbi__jpeg_info)  (Implementation)


---
### stbi\_\_png\_test<!-- {{#callable_declaration:stbi__png_test}} -->
Tests if the provided context contains a valid PNG header.
- **Description**: This function is used to determine whether the data in the provided context represents a valid PNG image. It should be called when you need to verify the format of the data before attempting to decode it. The function rewinds the context after checking the header, allowing for subsequent reads from the beginning. It is important to ensure that the context is properly initialized and points to a valid data source before calling this function.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the data source. Must not be null and should point to a valid context containing data to be checked.
- **Output**: Returns a non-zero value if the header is valid, indicating that the data is a PNG image; otherwise, it returns zero.
- **See also**: [`stbi__png_test`](#stbi__png_test)  (Implementation)


---
### stbi\_\_png\_load<!-- {{#callable_declaration:stbi__png_load}} -->
Loads a PNG image from a given context.
- **Description**: This function is used to load a PNG image from a specified context, providing the dimensions and color components of the image. It should be called after initializing the context and before accessing the image data. The function will populate the provided pointers for width, height, and color components based on the loaded image. It is important to ensure that the context is valid and points to a PNG image; otherwise, the function may not behave as expected.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that contains the image data. Must not be null and should point to a valid PNG image context.
    - `x`: A pointer to an integer where the width of the image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of color components will be stored. Can be null if this information is not needed.
    - `req_comp`: An integer specifying the number of color components requested. Valid values are typically 1, 3, or 4, corresponding to grayscale, RGB, or RGBA formats.
    - `ri`: A pointer to a `stbi__result_info` structure that will receive information about the result of the loading operation. Can be null if this information is not needed.
- **Output**: Returns a pointer to the loaded image data in memory. If the loading fails, it returns null.
- **See also**: [`stbi__png_load`](#stbi__png_load)  (Implementation)


---
### stbi\_\_png\_info<!-- {{#callable_declaration:stbi__png_info}} -->
Retrieves PNG image information.
- **Description**: This function is used to obtain the dimensions and color components of a PNG image from a given context. It should be called after ensuring that the context is properly initialized and points to a valid PNG image. The function will populate the provided pointers with the width, height, and number of color components of the image. If the context does not contain valid PNG data, the function will return an error code, and the output parameters will remain unchanged.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure that represents the PNG image data. Must not be null and must point to a valid PNG image.
    - `x`: A pointer to an integer where the width of the image will be stored. Caller retains ownership and must ensure it is not null.
    - `y`: A pointer to an integer where the height of the image will be stored. Caller retains ownership and must ensure it is not null.
    - `comp`: A pointer to an integer where the number of color components will be stored. Caller retains ownership and must ensure it is not null.
- **Output**: Returns 1 if the image information was successfully retrieved, or 0 if there was an error.
- **See also**: [`stbi__png_info`](#stbi__png_info)  (Implementation)


---
### stbi\_\_png\_is16<!-- {{#callable_declaration:stbi__png_is16}} -->
Checks if the PNG image is in 16-bit depth.
- **Description**: This function is used to determine whether a PNG image, represented by a `stbi__context`, has a color depth of 16 bits per channel. It should be called after initializing the context with a valid PNG image. If the image is not in 16-bit depth, the function rewinds the context to its original position. The function will return a non-zero value if the image is indeed 16-bit, and zero otherwise.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` that represents the PNG image. Must not be null and should point to a valid PNG image context.
- **Output**: Returns 1 if the image is in 16-bit depth, otherwise returns 0.
- **See also**: [`stbi__png_is16`](#stbi__png_is16)  (Implementation)


---
### stbi\_\_bmp\_test<!-- {{#callable_declaration:stbi__bmp_test}} -->
Tests if the input stream is a BMP image.
- **Description**: This function is used to determine whether the data in the provided stream corresponds to a BMP image format. It should be called with a valid `stbi__context` that has been properly initialized and points to a stream containing image data. The function rewinds the stream after performing the test, ensuring that subsequent reads will start from the beginning of the data. It is important to note that the function does not modify the stream's content, but it does reset the read position, which may affect any ongoing operations on the stream.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure that represents the input stream. This must not be null and should point to a valid context that has been initialized to read image data.
- **Output**: Returns a non-zero value if the stream is identified as a BMP image; otherwise, it returns zero.
- **See also**: [`stbi__bmp_test`](#stbi__bmp_test)  (Implementation)


---
### stbi\_\_bmp\_load<!-- {{#callable_declaration:stbi__bmp_load}} -->
Loads a BMP image from a given context.
- **Description**: This function is used to load a BMP image from a specified context, extracting its pixel data and providing the dimensions of the image. It should be called after initializing the context and before accessing the image data. The function handles various BMP formats and can return an image with a specified number of components. If the image dimensions exceed the maximum allowed size or if the BMP header is malformed, the function will return an error. Additionally, it can flip the image vertically based on the BMP format.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that contains the image data. Must not be null.
    - `x`: A pointer to an integer where the width of the image will be stored. Caller retains ownership.
    - `y`: A pointer to an integer where the height of the image will be stored. Caller retains ownership.
    - `comp`: A pointer to an integer where the number of components in the image will be stored. Can be null if not needed.
    - `req_comp`: An integer specifying the desired number of components in the output image. Valid values are 1, 3, or 4. If set to 0, the function will use the default number of components.
    - `ri`: A pointer to a `stbi__result_info` structure for additional result information. This parameter is currently unused and can be passed as null.
- **Output**: Returns a pointer to the loaded image data in the specified format. If the loading fails, it returns null and sets an appropriate error message.
- **See also**: [`stbi__bmp_load`](#stbi__bmp_load)  (Implementation)


---
### stbi\_\_bmp\_info<!-- {{#callable_declaration:stbi__bmp_info}} -->
Retrieves image dimensions and component count from a BMP file.
- **Description**: This function is used to extract the width, height, and number of color components from a BMP image file. It should be called after initializing the `stbi__context` with a valid BMP file. The function will parse the BMP header to obtain the necessary information. If the header is invalid or cannot be parsed, the function will return 0, and the output parameters will remain unchanged. If the header is valid, the dimensions and component count will be stored in the provided pointers, which must not be null.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the BMP file context. Must not be null.
    - `x`: A pointer to an integer where the width of the image will be stored. Can be null if the width is not needed.
    - `y`: A pointer to an integer where the height of the image will be stored. Can be null if the height is not needed.
    - `comp`: A pointer to an integer where the number of color components will be stored. Can be null if the component count is not needed.
- **Output**: Returns 1 if the image information was successfully retrieved, or 0 if there was an error parsing the BMP header.
- **See also**: [`stbi__bmp_info`](#stbi__bmp_info)  (Implementation)


---
### stbi\_\_tga\_test<!-- {{#callable_declaration:stbi__tga_test}} -->
Tests if the provided context contains a valid TGA image.
- **Description**: This function is used to verify whether the data in the provided context represents a valid TGA (Targa) image file. It should be called with a properly initialized `stbi__context` that points to the beginning of the TGA data. The function checks various properties of the TGA header, including color type, image type, and dimensions. If any of the checks fail, the function will return 0, indicating an invalid TGA image. It is important to note that the function will rewind the context to its original position before returning, ensuring that subsequent reads can occur without issues.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure that contains the TGA image data. This pointer must not be null and should point to a valid TGA file. If the context is invalid or does not contain a TGA image, the function will return 0.
- **Output**: Returns 1 if the TGA image is valid, otherwise returns 0.
- **See also**: [`stbi__tga_test`](#stbi__tga_test)  (Implementation)


---
### stbi\_\_tga\_load<!-- {{#callable_declaration:stbi__tga_load}} -->
Loads a TGA image from a given context.
- **Description**: This function is used to load a TGA image from a specified context, extracting its width, height, and color components. It should be called when you need to read TGA image data, and it expects the context to be properly initialized. The function handles various TGA formats, including indexed and RLE-compressed images. It is important to ensure that the provided pointers for width, height, and components are valid and can be modified. If the image dimensions exceed predefined limits or if memory allocation fails, the function will return an error. Additionally, the function may modify the pixel format based on the requested components.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` from which the TGA image will be read. Must not be null.
    - `x`: A pointer to an integer where the width of the image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of color components will be stored. Can be null if not needed.
    - `req_comp`: An integer specifying the desired number of color components in the output image. Valid values are typically 1, 3, or 4.
    - `ri`: A pointer to a `stbi__result_info` structure for additional result information. Can be null.
- **Output**: Returns a pointer to the loaded image data in the specified format, or null if an error occurred.
- **See also**: [`stbi__tga_load`](#stbi__tga_load)  (Implementation)


---
### stbi\_\_tga\_info<!-- {{#callable_declaration:stbi__tga_info}} -->
Retrieves information about a TGA image.
- **Description**: This function is used to extract the width, height, and number of components of a TGA image from a given context. It should be called after ensuring that the context is properly initialized and points to a valid TGA file. The function performs various checks on the TGA file format, including the colormap type and image type, and will return an error if the format is unsupported or if the dimensions are invalid. If the provided pointers for width, height, or components are not null, they will be populated with the corresponding values from the TGA file. It is important to handle the return value correctly, as a return of 0 indicates an error in reading the TGA file.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the TGA file context. Must not be null and should point to a valid TGA file.
    - `x`: A pointer to an integer where the width of the image will be stored. Can be null if the width is not needed.
    - `y`: A pointer to an integer where the height of the image will be stored. Can be null if the height is not needed.
    - `comp`: A pointer to an integer where the number of components (color channels) will be stored. Can be null if the component count is not needed.
- **Output**: Returns 1 if the information was successfully retrieved, or 0 if there was an error in reading the TGA file.
- **See also**: [`stbi__tga_info`](#stbi__tga_info)  (Implementation)


---
### stbi\_\_psd\_test<!-- {{#callable_declaration:stbi__psd_test}} -->
Tests if the given context is a valid PSD file.
- **Description**: This function is used to determine whether the provided context represents a valid Photoshop Document (PSD) file. It should be called when you need to verify the file type before attempting to read or process the data. The function rewinds the context after checking, ensuring that subsequent reads can occur from the beginning of the file. It is important to ensure that the context is properly initialized and points to a valid data source before calling this function.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure representing the file context. Must not be null and should point to a valid initialized context. If the context is invalid or uninitialized, the behavior is undefined.
- **Output**: Returns a non-zero value if the context is a valid PSD file, and zero otherwise.
- **See also**: [`stbi__psd_test`](#stbi__psd_test)  (Implementation)


---
### stbi\_\_psd\_load<!-- {{#callable_declaration:stbi__psd_load}} -->
Loads a PSD image from the given context.
- **Description**: This function is used to load a Photoshop Document (PSD) image from a specified context. It should be called when you need to read a PSD file and extract its pixel data. The function expects the context to be properly initialized and positioned at the start of a valid PSD file. It performs various checks to ensure the file format is correct, including verifying the PSD signature, version, channel count, dimensions, bit depth, and color mode. If any of these checks fail, the function will return an error. The output image data is allocated dynamically, and the caller is responsible for freeing this memory. The function also handles the conversion of the image data to the requested number of components, if specified.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure representing the PSD file to be loaded. Must not be null.
    - `x`: A pointer to an integer where the width of the loaded image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the loaded image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of components in the loaded image will be stored. Can be null if the component count is not needed.
    - `req_comp`: An integer specifying the desired number of components in the output image. Valid values are 0 (default), 3 (RGB), or 4 (RGBA). If 0, the function will return the image in its original format.
    - `ri`: A pointer to a `stbi__result_info` structure where additional information about the result will be stored. Must not be null.
    - `bpc`: An integer specifying the bits per channel for the output image. Valid values are 8 or 16. This affects how the pixel data is interpreted.
- **Output**: Returns a pointer to the loaded image data in the format specified by `req_comp`. If the image cannot be loaded due to an error, it returns a null pointer.
- **See also**: [`stbi__psd_load`](#stbi__psd_load)  (Implementation)


---
### stbi\_\_psd\_info<!-- {{#callable_declaration:stbi__psd_info}} -->
Retrieves information about a PSD image.
- **Description**: This function is used to extract the dimensions and component count of a PSD (Photoshop Document) image from a given context. It should be called after ensuring that the context is properly initialized and points to a valid PSD file. The function will populate the provided pointers with the image's width, height, and number of components. If the PSD file is invalid or does not meet the expected format, the function will reset the context and return 0, indicating failure. It is important to note that the function expects the image to have a specific structure, including a maximum of 16 channels and a depth of either 8 or 16 bits.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure that represents the PSD file context. Must not be null.
    - `x`: A pointer to an integer where the width of the image will be stored. Can be null; if so, a dummy variable will be used.
    - `y`: A pointer to an integer where the height of the image will be stored. Can be null; if so, a dummy variable will be used.
    - `comp`: A pointer to an integer where the number of components will be stored. Can be null; if so, a dummy variable will be used.
- **Output**: Returns 1 on success, indicating that the information was successfully retrieved and the output parameters have been populated. Returns 0 on failure, indicating that the PSD file is invalid or does not conform to the expected format.
- **See also**: [`stbi__psd_info`](#stbi__psd_info)  (Implementation)


---
### stbi\_\_psd\_is16<!-- {{#callable_declaration:stbi__psd_is16}} -->
Checks if the PSD file is a 16-bit format.
- **Description**: This function is used to determine if a given PSD (Photoshop Document) file is in 16-bit format. It should be called after initializing the `stbi__context` with the PSD file data. The function reads the necessary headers and checks the channel count and bit depth to confirm the format. If the file does not meet the expected criteria, such as an incorrect signature, unsupported channel count, or bit depth, the function will rewind the context to its original position and return 0. A return value of 1 indicates that the file is indeed a 16-bit PSD.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure that represents the PSD file. Must not be null and should be properly initialized with the PSD file data.
- **Output**: Returns 1 if the PSD file is a valid 16-bit format, otherwise returns 0.
- **See also**: [`stbi__psd_is16`](#stbi__psd_is16)  (Implementation)


---
### stbi\_\_hdr\_test<!-- {{#callable_declaration:stbi__hdr_test}} -->
Tests if the input stream contains a valid HDR image.
- **Description**: This function is used to determine if the provided input stream contains a valid High Dynamic Range (HDR) image format. It should be called with a properly initialized `stbi__context` that points to the image data. The function checks for two specific HDR headers, and it rewinds the stream after each check to ensure that the context is ready for subsequent operations. If the input stream does not contain a valid HDR image, the function will return a failure indicator.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure that represents the input stream. This must not be null and should be properly initialized to point to the image data.
- **Output**: Returns a non-zero value if the input stream contains a valid HDR image format; otherwise, it returns zero.
- **See also**: [`stbi__hdr_test`](#stbi__hdr_test)  (Implementation)


---
### stbi\_\_hdr\_load<!-- {{#callable_declaration:stbi__hdr_load}} -->
Loads an HDR image from a given context.
- **Description**: This function is used to load High Dynamic Range (HDR) images from a specified context. It should be called when you need to read HDR image data, and it will populate the width and height of the image through the provided pointers. The function expects the context to be properly initialized and positioned at the start of a valid HDR image. If the image format is unsupported or if the dimensions exceed predefined limits, the function will return an error. Additionally, it allocates memory for the image data, which the caller is responsible for freeing after use.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` from which the HDR image will be read. Must not be null.
    - `x`: A pointer to an integer where the width of the image will be stored. Caller retains ownership.
    - `y`: A pointer to an integer where the height of the image will be stored. Caller retains ownership.
    - `comp`: A pointer to an integer where the number of color components will be stored. Can be null if not needed.
    - `req_comp`: An integer specifying the number of components requested. If set to 0, defaults to 3 (RGB).
    - `ri`: A pointer to `stbi__result_info` for additional result information. Not used in this function.
- **Output**: Returns a pointer to a float array containing the HDR image data. The caller is responsible for freeing this memory. If an error occurs, it returns a null pointer.
- **See also**: [`stbi__hdr_load`](#stbi__hdr_load)  (Implementation)


---
### stbi\_\_hdr\_info<!-- {{#callable_declaration:stbi__hdr_info}} -->
Retrieves image dimensions and component count from an HDR file.
- **Description**: This function is used to extract the width, height, and number of color components from an HDR image file. It should be called after ensuring that the provided `stbi__context` is valid and points to a properly initialized HDR file. The function will modify the values pointed to by `x`, `y`, and `comp` to reflect the image's dimensions and component count. If the HDR file is not valid or does not conform to the expected format, the function will reset the context and return 0, indicating failure. It is important to note that if any of the output parameters are `NULL`, they will be assigned a dummy value instead.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure representing the HDR file. Must not be null.
    - `x`: A pointer to an integer where the width of the image will be stored. Can be null; if so, a dummy value will be used.
    - `y`: A pointer to an integer where the height of the image will be stored. Can be null; if so, a dummy value will be used.
    - `comp`: A pointer to an integer where the number of color components will be stored. Can be null; if so, a dummy value will be used.
- **Output**: Returns 1 on success, indicating that the dimensions and component count have been successfully retrieved. Returns 0 on failure, indicating that the HDR file is invalid or not in the expected format.
- **See also**: [`stbi__hdr_info`](#stbi__hdr_info)  (Implementation)


---
### stbi\_\_pic\_test<!-- {{#callable_declaration:stbi__pic_test}} -->
Tests if the input stream is a valid picture.
- **Description**: This function is used to determine whether the provided input stream represents a valid image format. It should be called with a properly initialized `stbi__context` that points to the image data. The function rewinds the input stream after performing the test, allowing for subsequent reads from the beginning of the stream. It is important to ensure that the context is valid and properly set up before calling this function, as invalid contexts may lead to undefined behavior.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure that represents the input stream. Must not be null and should be initialized to point to valid image data.
- **Output**: Returns a non-zero value if the input stream is a valid picture format; otherwise, it returns zero.
- **See also**: [`stbi__pic_test`](#stbi__pic_test)  (Implementation)


---
### stbi\_\_pic\_load<!-- {{#callable_declaration:stbi__pic_load}} -->
Loads a PIC image from the specified context.
- **Description**: This function is used to load a PIC image from a given context, extracting its dimensions and pixel data. It should be called when you need to read a PIC image file, and it will populate the provided pointers with the image's width and height. The function checks for various error conditions, such as excessively large dimensions or memory allocation failures, and will return an error if any issues are encountered. It is important to ensure that the context is valid and that the pointers for width and height are not null before calling this function.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure representing the image context. Must not be null.
    - `px`: A pointer to an integer where the width of the image will be stored. Must not be null.
    - `py`: A pointer to an integer where the height of the image will be stored. Must not be null.
    - `comp`: A pointer to an integer that will hold the number of components in the image. If null, an internal variable will be used. Caller retains ownership.
    - `req_comp`: An integer specifying the desired number of components in the output image. Valid values are typically 1, 3, or 4. If set to 0, the function will use the original number of components.
    - `ri`: A pointer to a `stbi__result_info` structure for additional result information. This parameter is unused and can be passed as null.
- **Output**: Returns a pointer to the loaded image data in the specified format, or null if an error occurred. The output image data is allocated dynamically and must be freed by the caller.
- **See also**: [`stbi__pic_load`](#stbi__pic_load)  (Implementation)


---
### stbi\_\_pic\_info<!-- {{#callable_declaration:stbi__pic_info}} -->
Retrieves image dimensions and component count from a picture.
- **Description**: This function is used to extract the width, height, and number of color components from an image file. It should be called after ensuring that the image context has been properly initialized. The function checks for specific image format signatures and reads the necessary data to populate the output parameters. If the image format is invalid or if the dimensions exceed certain limits, the function will reset the context and return an error. It is important to provide valid pointers for the output parameters to receive the image dimensions and component count.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` structure representing the image context. Must not be null.
    - `x`: A pointer to an integer where the width of the image will be stored. Can be null; if so, a dummy variable will be used.
    - `y`: A pointer to an integer where the height of the image will be stored. Can be null; if so, a dummy variable will be used.
    - `comp`: A pointer to an integer where the number of color components will be stored. Can be null; if so, a dummy variable will be used.
- **Output**: Returns 1 on success, indicating that the dimensions and component count have been successfully retrieved. Returns 0 on failure, indicating that the image format is invalid or that an error occurred during reading.
- **See also**: [`stbi__pic_info`](#stbi__pic_info)  (Implementation)


---
### stbi\_\_gif\_test<!-- {{#callable_declaration:stbi__gif_test}} -->
Tests if the input stream is a valid GIF image.
- **Description**: This function should be called to determine if the data in the provided context represents a valid GIF image format. It is essential to ensure that the input stream is correctly positioned at the beginning of the GIF data before calling this function, as it will rewind the stream after testing. If the input stream does not contain valid GIF data, the function will return an error code, allowing the caller to handle the invalid format appropriately.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure representing the input stream. Must not be null and should point to a valid context that has been initialized with image data.
- **Output**: Returns a non-zero value if the input stream is a valid GIF image; otherwise, it returns zero.
- **See also**: [`stbi__gif_test`](#stbi__gif_test)  (Implementation)


---
### stbi\_\_gif\_load<!-- {{#callable_declaration:stbi__gif_load}} -->
Loads a GIF image from the specified context.
- **Description**: This function is used to load a GIF image from a given context, extracting its width and height, and optionally converting the image to a specified number of components. It should be called when you need to read a GIF file, and it will populate the provided width and height pointers with the dimensions of the loaded image. The function handles multiple frames and will free any allocated buffers if an error occurs during loading. It is important to ensure that the context is properly initialized before calling this function.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` from which the GIF image will be loaded. Must not be null.
    - `x`: A pointer to an integer where the width of the loaded image will be stored. Caller retains ownership.
    - `y`: A pointer to an integer where the height of the loaded image will be stored. Caller retains ownership.
    - `comp`: A pointer to an integer that specifies the number of components in the output image. Must not be null.
    - `req_comp`: An integer specifying the desired number of components in the output image. Valid values are 1, 2, 3, or 4. If set to 0, the function will use the original number of components.
    - `ri`: A pointer to a `stbi__result_info` structure for additional result information. This parameter is unused and can be passed as null.
- **Output**: Returns a pointer to the loaded image data in the specified format, or null if the loading fails. If the loading is successful, the width and height of the image are stored in the provided pointers.
- **See also**: [`stbi__gif_load`](#stbi__gif_load)  (Implementation)


---
### stbi\_\_load\_gif\_main<!-- {{#callable_declaration:stbi__load_gif_main}} -->
Loads an animated GIF and retrieves its frames.
- **Description**: This function is used to load an animated GIF from a given context, extracting its frames and their respective delays. It should be called when you need to process GIF images, and it expects the context to be properly initialized. The function allocates memory for the output frames and delays, which must be freed by the caller. If the GIF is invalid or not of the correct type, an error is returned. The function handles multiple frames and provides the width, height, and number of frames in the output parameters.
- **Inputs**:
    - `s`: A pointer to the `stbi__context` from which the GIF will be loaded. Must not be null.
    - `delays`: A pointer to an integer pointer where the delays for each frame will be stored. Can be null if delays are not needed.
    - `x`: A pointer to an integer where the width of the GIF will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the GIF will be stored. Must not be null.
    - `z`: A pointer to an integer where the number of frames (layers) in the GIF will be stored. Must not be null.
    - `comp`: A pointer to an integer that specifies the desired number of color components in the output. Must not be null.
    - `req_comp`: An integer indicating the requested number of components (e.g., 3 for RGB, 4 for RGBA). Valid values are typically 1, 3, or 4.
- **Output**: Returns a pointer to the loaded image data, which contains the frames of the GIF. If the GIF is invalid or an error occurs, it returns a null pointer.
- **See also**: [`stbi__load_gif_main`](#stbi__load_gif_main)  (Implementation)


---
### stbi\_\_gif\_info<!-- {{#callable_declaration:stbi__gif_info}} -->
Retrieves GIF image information.
- **Description**: This function is used to obtain the dimensions and color component information of a GIF image from a given context. It should be called after initializing the context with a valid GIF image. The function populates the provided pointers with the width and height of the image, as well as the number of color components. It is important to ensure that the pointers passed for width, height, and components are not null, as the function expects valid memory addresses to write the information. If the context does not point to a valid GIF image, the function may return an error.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure that represents the GIF image context. Must not be null and must point to a valid initialized context.
    - `x`: A pointer to an integer where the width of the GIF image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the GIF image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of color components will be stored. Must not be null.
- **Output**: Returns a non-zero value on success, indicating that the image information was successfully retrieved. Returns zero if there was an error in retrieving the information.
- **See also**: [`stbi__gif_info`](#stbi__gif_info)  (Implementation)


---
### stbi\_\_pnm\_test<!-- {{#callable_declaration:stbi__pnm_test}} -->
Tests if the input stream is a valid PNM image.
- **Description**: This function is used to determine if the provided input stream corresponds to a valid PNM (Portable Any Map) image format, specifically P5 (grayscale) or P6 (RGB). It should be called with a valid `stbi__context` that has been properly initialized to read from an image file. If the first two characters of the stream do not match the expected PNM format identifiers, the function will rewind the stream to its original position. This allows for subsequent reads without losing the initial state of the stream.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure that represents the input stream. This must not be null and should be initialized to point to a valid image file.
- **Output**: Returns 1 if the input stream is a valid PNM image (either P5 or P6), and 0 otherwise.
- **See also**: [`stbi__pnm_test`](#stbi__pnm_test)  (Implementation)


---
### stbi\_\_pnm\_load<!-- {{#callable_declaration:stbi__pnm_load}} -->
Loads a PNM image from a given context.
- **Description**: This function is used to load a PNM (Portable Any Map) image from the provided context. It must be called with a valid `stbi__context` that has been properly initialized. The function retrieves the image dimensions and number of components, storing them in the provided pointers. It is important to ensure that the image dimensions do not exceed predefined limits, as this may indicate a corrupt image. The function also handles memory allocation for the image data and can convert the image to a requested number of components if specified. If any errors occur during loading, such as memory allocation failure or truncated data, the function will return a null pointer.
- **Inputs**:
    - `s`: A pointer to an `stbi__context` structure that contains the image data. Must not be null.
    - `x`: A pointer to an integer where the width of the image will be stored. Must not be null.
    - `y`: A pointer to an integer where the height of the image will be stored. Must not be null.
    - `comp`: A pointer to an integer where the number of components in the image will be stored. Can be null if not needed.
    - `req_comp`: An integer specifying the number of components requested in the output image. Valid values are typically 1, 3, or 4.
    - `ri`: A pointer to an `stbi__result_info` structure where additional result information will be stored. Must not be null.
- **Output**: Returns a pointer to the loaded image data in memory, or null if an error occurred during loading.
- **See also**: [`stbi__pnm_load`](#stbi__pnm_load)  (Implementation)


---
### stbi\_\_pnm\_info<!-- {{#callable_declaration:stbi__pnm_info}} -->
Retrieves image information from a PNM file.
- **Description**: This function is used to extract the width, height, and number of components from a PNM (Portable Anymap) image file. It should be called after initializing the `stbi__context` with a valid PNM file. The function reads the image header to determine the dimensions and color depth, returning an error if the header is invalid or if the dimensions are zero or overflow. The caller must ensure that the pointers for width, height, and components are valid, as the function will write the results to these locations.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the PNM file. Must not be null.
    - `x`: A pointer to an integer where the width of the image will be stored. Can be null; if so, a dummy variable will be used.
    - `y`: A pointer to an integer where the height of the image will be stored. Can be null; if so, a dummy variable will be used.
    - `comp`: A pointer to an integer where the number of color components will be stored. Can be null; if so, a dummy variable will be used.
- **Output**: Returns the bit depth of the image (8 or 16) if successful, or 0 if there was an error in reading the header or if the dimensions are invalid.
- **See also**: [`stbi__pnm_info`](#stbi__pnm_info)  (Implementation)


---
### stbi\_\_pnm\_is16<!-- {{#callable_declaration:stbi__pnm_is16}} -->
Checks if the PNM image is in 16-bit format.
- **Description**: This function is used to determine whether a PNM (Portable Any Map) image is stored in a 16-bit format. It should be called after initializing the `stbi__context` with a valid PNM image. The function checks the image format and returns a boolean value indicating if the image is indeed 16-bit. If the image is not properly initialized or is not a PNM image, the behavior is undefined.
- **Inputs**:
    - `s`: A pointer to a `stbi__context` structure that represents the image context. Must not be null and should point to a valid initialized context containing a PNM image.
- **Output**: Returns 1 if the image is in 16-bit format, and 0 otherwise.
- **See also**: [`stbi__pnm_is16`](#stbi__pnm_is16)  (Implementation)


---
### stbi\_\_ldr\_to\_hdr<!-- {{#callable_declaration:stbi__ldr_to_hdr}} -->
Converts an 8-bit per channel image to a floating-point HDR format.
- **Description**: This function is used to convert image data from an 8-bit per channel format to a high dynamic range (HDR) format represented as floating-point values. It should be called with valid image data, width, height, and the number of components per pixel. The function allocates memory for the output; if memory allocation fails, it frees the input data and returns an error. The input data must not be null, and the function handles the conversion for both RGB and RGBA formats, ensuring that the alpha channel is preserved if present.
- **Inputs**:
    - `data`: Pointer to the input image data in 8-bit per channel format. Must not be null. The function takes ownership of this pointer and will free it after processing.
    - `x`: The width of the image in pixels. Must be a positive integer.
    - `y`: The height of the image in pixels. Must be a positive integer.
    - `comp`: The number of components per pixel (e.g., 3 for RGB, 4 for RGBA). Must be a positive integer.
- **Output**: Returns a pointer to the newly allocated array of floating-point values representing the HDR image. If memory allocation fails or if the input data is null, it returns null.
- **See also**: [`stbi__ldr_to_hdr`](#stbi__ldr_to_hdr)  (Implementation)


---
### stbi\_\_hdr\_to\_ldr<!-- {{#callable_declaration:stbi__hdr_to_ldr}} -->
Converts HDR image data to LDR format.
- **Description**: This function is used to convert high dynamic range (HDR) image data into low dynamic range (LDR) format, which is suitable for display. It should be called with valid HDR data, along with the dimensions of the image and the number of components per pixel. The function allocates memory for the output image, which must be freed by the caller after use. If the input data is null or if memory allocation fails, the function will return null. The function also handles clamping of pixel values to ensure they remain within the valid range for LDR.
- **Inputs**:
    - `data`: A pointer to the HDR image data, which must not be null. The data should contain floating-point values representing pixel colors.
    - `x`: The width of the image in pixels, must be a positive integer.
    - `y`: The height of the image in pixels, must be a positive integer.
    - `comp`: The number of color components per pixel (e.g., 3 for RGB, 4 for RGBA). Must be a positive integer.
- **Output**: Returns a pointer to the converted LDR image data. If the conversion fails or if the input data is null, returns null.
- **See also**: [`stbi__hdr_to_ldr`](#stbi__hdr_to_ldr)  (Implementation)


