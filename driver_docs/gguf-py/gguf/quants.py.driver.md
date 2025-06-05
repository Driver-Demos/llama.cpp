# Purpose
This Python code file is a comprehensive library designed for handling various quantization and dequantization processes for tensors, specifically using the GGML (Generalized Graphical Model Library) quantization types. The file defines a series of classes and functions that facilitate the conversion of data between different numerical precisions, such as floating-point and quantized formats, which are essential for optimizing storage and computation in machine learning models. The core functionality revolves around the `__Quant` abstract base class, which provides a framework for implementing specific quantization strategies, and its numerous subclasses, each representing a distinct quantization type (e.g., `BF16`, `Q4_0`, `Q5_1`, etc.).

The file includes utility functions like [`quant_shape_to_byte_shape`](#cpp/gguf-py/gguf/quantsquant_shape_to_byte_shape) and [`quant_shape_from_byte_shape`](#cpp/gguf-py/gguf/quantsquant_shape_from_byte_shape) to manage tensor shape transformations during quantization. The [`_apply_over_grouped_rows`](#cpp/gguf-py/gguf/quants_apply_over_grouped_rows) function optimizes operations over tensor rows, enhancing performance by processing multiple rows simultaneously. The [`quantize`](#cpp/gguf-py/gguf/quantsquantize) and [`dequantize`](#cpp/gguf-py/gguf/quantsdequantize) functions serve as the primary interfaces for converting data to and from quantized formats, leveraging the specific methods defined in each subclass. The code is structured to be imported as a module, providing a public API for quantization operations, and it integrates with NumPy for efficient numerical computations. The use of abstract methods and class-level attributes ensures that each quantization type is implemented with the necessary specificity while maintaining a consistent interface across different types.
# Imports and Dependencies

---
- `__future__.annotations`
- `abc.ABC`
- `abc.abstractmethod`
- `typing.Any`
- `typing.Callable`
- `typing.Sequence`
- `math.log2`
- `math.ceil`
- `numpy.typing.DTypeLike`
- `.constants.GGML_QUANT_SIZES`
- `.constants.GGMLQuantizationType`
- `.constants.QK_K`
- `.lazy.LazyNumpyTensor`
- `numpy`


# Global Variables

---
### \_type\_traits
- **Type**: `dict[GGMLQuantizationType, type[__Quant]]`
- **Description**: The `_type_traits` variable is a dictionary that maps `GGMLQuantizationType` keys to corresponding `__Quant` class types. It is initially defined as an empty dictionary and is populated dynamically as subclasses of `__Quant` are defined.
- **Use**: This variable is used to store and retrieve the appropriate quantization class type for a given quantization type, facilitating the quantization and dequantization processes.


# Classes

---
### QuantError<!-- {{#class:llama.cpp/gguf-py/gguf/quants.QuantError}} -->
- **Description**: The `QuantError` class is a custom exception that inherits from Python's built-in `Exception` class. It is used to signal errors specific to quantization processes within the codebase, providing a mechanism to handle quantization-related exceptions separately from other types of exceptions.
- **Inherits From**:
    - `Exception`


---
### \_\_Quant<!-- {{#class:llama.cpp/gguf-py/gguf/quants.__Quant}} -->
- **Decorators**: `@ABC`
- **Members**:
    - `qtype`: Specifies the quantization type for the class.
    - `block_size`: Defines the size of each block for quantization.
    - `type_size`: Indicates the size of the data type used in quantization.
    - `grid`: Holds the quantization grid as a numpy array, initially set to None.
    - `grid_shape`: Specifies the shape of the quantization grid.
    - `grid_map`: Maps quantization values to specific grid indices.
    - `grid_hex`: Stores the hexadecimal representation of the quantization grid, initially set to None.
- **Description**: The `__Quant` class is an abstract base class designed for handling quantization and dequantization of data using various quantization types defined by `GGMLQuantizationType`. It provides a framework for subclasses to implement specific quantization and dequantization methods for different data types and block sizes. The class manages a quantization grid and its associated properties, and it ensures that subclasses properly initialize and utilize these resources. The class also includes methods for converting data shapes between quantized and dequantized forms, and it supports lazy evaluation for quantization operations.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__init__`](#__Quant__init__)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__init_subclass__`](#__Quant__init_subclass__)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.init_grid`](#__Quantinit_grid)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.quantize_blocks`](#__Quantquantize_blocks)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.dequantize_blocks`](#__Quantdequantize_blocks)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.quantize_rows`](#__Quantquantize_rows)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.dequantize_rows`](#__Quantdequantize_rows)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__shape_to_bytes`](#__Quant__shape_to_bytes)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__shape_from_bytes`](#__Quant__shape_from_bytes)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__quantize_array`](#__Quant__quantize_array)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__dequantize_array`](#__Quant__dequantize_array)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__quantize_lazy`](#__Quant__quantize_lazy)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__dequantize_lazy`](#__Quant__dequantize_lazy)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.can_quantize`](#__Quantcan_quantize)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.quantize`](#__Quantquantize)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.dequantize`](#__Quantdequantize)
- **Inherits From**:
    - `ABC`

**Methods**

---
#### \_\_Quant\.\_\_init\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.__init__}} -->
The `__init__` method in the `__Quant` class raises a `TypeError` to prevent instantiation of the class.
- **Inputs**: None
- **Control Flow**:
    - The method immediately raises a `TypeError` with a specific message.
- **Output**: The method raises a `TypeError` with the message "Quant conversion classes can't have instances".
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.\_\_init\_subclass\_\_<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.__init_subclass__}} -->
The `__init_subclass__` method initializes subclass-specific attributes and functions for quantization and dequantization based on a given quantization type.
- **Inputs**:
    - `cls`: The class object that is being initialized as a subclass.
    - `qtype`: An instance of `GGMLQuantizationType` that specifies the quantization type for the subclass.
- **Control Flow**:
    - Assigns the `qtype` to the class attribute `cls.qtype`.
    - Retrieves `block_size` and `type_size` from `GGML_QUANT_SIZES` using `qtype` and assigns them to class attributes `cls.block_size` and `cls.type_size`.
    - Wraps the `__quantize_array` method using `LazyNumpyTensor._wrap_fn` and assigns it to `cls.__quantize_lazy`.
    - Wraps the `__dequantize_array` method using `LazyNumpyTensor._wrap_fn` and assigns it to `cls.__dequantize_lazy`.
    - Asserts that `qtype` is not already in the `_type_traits` dictionary.
    - Adds the class to the `_type_traits` dictionary with `qtype` as the key.
- **Output**: The method does not return any value; it modifies class-level attributes and registers the class in a global dictionary.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyBase._wrap_fn`](lazy.py.driver.md#LazyBase_wrap_fn)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.init\_grid<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.init_grid}} -->
The `init_grid` method initializes the `grid` attribute by decoding and transforming the `grid_hex` data using the `grid_map` and reshaping it according to `grid_shape`.
- **Decorators**: `@classmethod`
- **Inputs**: None
- **Control Flow**:
    - Check if `cls.grid` is not `None` or `cls.grid_hex` is `None`, and return if either condition is true.
    - Calculate `bits_per_elem` as the ceiling of the logarithm base 2 of the length of `cls.grid_map`.
    - Assert that `bits_per_elem` is not zero, using `cls.qtype.name` for the assertion message.
    - Calculate `elems_per_byte` as the integer division of 8 by `bits_per_elem`.
    - Convert `cls.grid_hex` from a buffer to a NumPy array of type `uint8`.
    - Reshape the `grid` array to have two columns, effectively splitting each byte into two nibbles.
    - Decode the hexadecimal characters by adjusting values greater than 0x40 and applying a bitwise AND with 0x0F, then shift the nibbles to their correct positions.
    - Combine the nibbles into single byte values using bitwise OR operations.
    - Unpack the grid values by reshaping and right-shifting them according to `elems_per_byte`.
    - Apply a bitwise AND operation to extract the relevant bits for each element.
    - Use `np.take_along_axis` to map the grid values to their corresponding float values in `grid_map`.
    - Reshape the final grid to match the specified `grid_shape` and assign it to `cls.grid`.
- **Output**: The method does not return any value; it modifies the class attribute `cls.grid`.
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.quantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.quantize_blocks}} -->
The `quantize_blocks` method is an abstract class method intended to quantize blocks of data represented as a NumPy array.
- **Decorators**: `@classmethod`, `@abstractmethod`
- **Inputs**:
    - `cls`: The class object that is calling the method, typically a subclass of `__Quant`.
    - `blocks`: A NumPy array representing the blocks of data to be quantized.
- **Control Flow**:
    - The method is defined as an abstract method, meaning it must be implemented by any subclass of `__Quant` that inherits it.
    - The method raises a `NotImplementedError`, indicating that it is a placeholder for subclasses to provide specific quantization logic.
- **Output**: The method is expected to return a NumPy array representing the quantized blocks, but as an abstract method, it does not provide an implementation.
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.dequantize_blocks}} -->
The `dequantize_blocks` method is an abstract class method intended to convert quantized data blocks back into their original floating-point representation.
- **Decorators**: `@classmethod`, `@abstractmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized data blocks to be dequantized.
- **Control Flow**:
    - The method is defined as an abstract method, meaning it must be implemented by any subclass inheriting from the parent class.
    - The method raises a NotImplementedError, indicating that it serves as a placeholder for subclasses to provide specific dequantization logic.
- **Output**: The method is expected to return a numpy ndarray containing the dequantized floating-point data, but as an abstract method, it does not provide an implementation.
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.quantize\_rows<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.quantize_rows}} -->
The `quantize_rows` method converts a 2D numpy array of floating-point numbers into a quantized format using a specified block size and type size.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `rows`: A 2D numpy array of floating-point numbers to be quantized.
- **Control Flow**:
    - Convert the input `rows` to a numpy array of type `float32` without copying the data.
    - Determine the shape of the input `rows` and calculate the number of blocks by dividing the total number of elements by the class's `block_size`.
    - Reshape the `rows` into blocks of size `block_size`.
    - Call the [`quantize_blocks`](#__Quantquantize_blocks) method to quantize these blocks.
    - Assert that the resulting blocks have a data type of `uint8` and that the last dimension of the blocks matches the class's `type_size`.
    - Reshape the quantized blocks back to the original shape converted to bytes using the [`__shape_to_bytes`](#__Quant__shape_to_bytes) method.
- **Output**: A numpy array of quantized data with a shape determined by the [`__shape_to_bytes`](#__Quant__shape_to_bytes) method.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.quantize_blocks`](#__Quantquantize_blocks)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__shape_to_bytes`](#__Quant__shape_to_bytes)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.dequantize\_rows<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.dequantize_rows}} -->
The `dequantize_rows` method converts quantized data in rows back to its original floating-point representation.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `rows`: A numpy ndarray representing quantized data, expected to be in a specific byte format.
- **Control Flow**:
    - The method first views the input `rows` as an array of type `np.uint8`.
    - It calculates the shape of the input `rows` and determines the number of blocks by dividing the total size of `rows` by `cls.type_size`.
    - The `rows` are reshaped into blocks of size `cls.type_size`.
    - The method calls `cls.dequantize_blocks` to convert these blocks from quantized to floating-point format.
    - It asserts that the resulting blocks have a data type of `np.float32` and that the last dimension of the blocks matches `cls.block_size`.
    - Finally, it reshapes the dequantized blocks back to the original shape using `cls.__shape_from_bytes` and returns the result.
- **Output**: A numpy ndarray of type `np.float32` representing the dequantized data, reshaped to match the original input shape.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.dequantize_blocks`](#__Quantdequantize_blocks)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__shape_from_bytes`](#__Quant__shape_from_bytes)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.\_\_shape\_to\_bytes<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.__shape_to_bytes}} -->
The `__shape_to_bytes` method converts a given shape into a byte shape based on the quantization type of the class.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `shape`: A sequence of integers representing the dimensions of a tensor.
- **Control Flow**:
    - The method calls the [`quant_shape_to_byte_shape`](#cpp/gguf-py/gguf/quantsquant_shape_to_byte_shape) function, passing the input `shape` and the class's `qtype` as arguments.
- **Output**: A tuple of integers representing the byte shape of the input tensor.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/quants.quant_shape_to_byte_shape`](#cpp/gguf-py/gguf/quantsquant_shape_to_byte_shape)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.\_\_shape\_from\_bytes<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.__shape_from_bytes}} -->
The `__shape_from_bytes` method converts a byte-based shape into a quantized shape using the class's quantization type.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `shape`: A sequence of integers representing the shape of the data in bytes.
- **Control Flow**:
    - The method calls the [`quant_shape_from_byte_shape`](#cpp/gguf-py/gguf/quantsquant_shape_from_byte_shape) function, passing the `shape` and the class's `qtype` as arguments.
- **Output**: The method returns a tuple of integers representing the quantized shape derived from the byte-based shape.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/quants.quant_shape_from_byte_shape`](#cpp/gguf-py/gguf/quantsquant_shape_from_byte_shape)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.\_\_quantize\_array<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.__quantize_array}} -->
The `__quantize_array` method quantizes a given numpy array using a class-specific quantization process and returns the quantized array as an unsigned 8-bit integer numpy array.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `array`: A numpy ndarray that is to be quantized.
- **Control Flow**:
    - The method is a class method, meaning it operates on the class rather than an instance of the class.
    - It calls the [`_apply_over_grouped_rows`](#cpp/gguf-py/gguf/quants_apply_over_grouped_rows) function, passing `cls.quantize_rows` as the function to apply over the rows of the input array.
    - The input array is reshaped into groups of rows, and the `quantize_rows` method is applied to each group.
    - The output type is specified as `np.uint8`, indicating that the quantized data will be in 8-bit unsigned integer format.
    - The output shape is determined by calling `cls.__shape_to_bytes` with the shape of the input array, which converts the shape to the appropriate byte shape for quantization.
- **Output**: A numpy ndarray of type `np.uint8` representing the quantized version of the input array.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/quants._apply_over_grouped_rows`](#cpp/gguf-py/gguf/quants_apply_over_grouped_rows)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__shape_to_bytes`](#__Quant__shape_to_bytes)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.\_\_dequantize\_array<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.__dequantize_array}} -->
The `__dequantize_array` method dequantizes a given numpy array by applying a dequantization function over grouped rows of the array.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `array`: A numpy ndarray that represents the quantized data to be dequantized.
- **Control Flow**:
    - The method begins by calling `cls.init_grid()` to ensure the grid is initialized.
    - It then calls [`_apply_over_grouped_rows`](#cpp/gguf-py/gguf/quants_apply_over_grouped_rows) with `cls.dequantize_rows` as the function to apply, `arr=array` as the input array, `otype=np.float32` as the output data type, and `oshape=cls.__shape_from_bytes(array.shape)` as the output shape.
    - The [`_apply_over_grouped_rows`](#cpp/gguf-py/gguf/quants_apply_over_grouped_rows) function processes the array in groups of rows, applying the `dequantize_rows` function to each group and reshaping the result to the specified output shape.
- **Output**: A numpy ndarray of type `np.float32` that represents the dequantized data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.init_grid`](#__Quantinit_grid)
    - [`llama.cpp/gguf-py/gguf/quants._apply_over_grouped_rows`](#cpp/gguf-py/gguf/quants_apply_over_grouped_rows)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__shape_from_bytes`](#__Quant__shape_from_bytes)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.\_\_quantize\_lazy<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.__quantize_lazy}} -->
The `__quantize_lazy` method is a placeholder for quantizing a `LazyNumpyTensor` within the `__Quant` class.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `lazy_tensor`: A `LazyNumpyTensor` object that is intended to be quantized.
- **Control Flow**:
    - The method is defined but not implemented, indicated by the `pass` statement.
- **Output**: The method does not return any value or output as it is not implemented.
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.\_\_dequantize\_lazy<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.__dequantize_lazy}} -->
The `__dequantize_lazy` method is a placeholder for dequantizing a `LazyNumpyTensor` using a class method.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `lazy_tensor`: A `LazyNumpyTensor` object that is intended to be dequantized.
- **Control Flow**:
    - The method is defined as a class method, indicated by the `@classmethod` decorator.
    - The method currently contains no implementation (indicated by `pass`).
- **Output**: The method is expected to return any type of result (`Any`), but currently, it does not return anything due to the lack of implementation.
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.can\_quantize<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.can_quantize}} -->
The `can_quantize` method checks if a given tensor can be quantized based on the block size of the quantization type.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `tensor`: A tensor which can be either a NumPy ndarray or a LazyNumpyTensor, representing the data to be checked for quantization compatibility.
- **Control Flow**:
    - The method checks if the last dimension of the tensor's shape is divisible by the class's block size.
- **Output**: A boolean value indicating whether the tensor can be quantized (True) or not (False).
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.quantize<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.quantize}} -->
The `quantize` method quantizes a given tensor into a specified format using class-specific quantization logic.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `tensor`: A tensor of type `np.ndarray` or `LazyNumpyTensor` that needs to be quantized.
- **Control Flow**:
    - Check if the tensor can be quantized using `cls.can_quantize(tensor)`; if not, raise a [`QuantError`](#cpp/gguf-py/gguf/quantsQuantError) with a descriptive message.
    - Determine if the tensor is an instance of `LazyNumpyTensor`.
    - If the tensor is a `LazyNumpyTensor`, call `cls.__quantize_lazy(tensor)` to perform lazy quantization.
    - If the tensor is a regular `np.ndarray`, call `cls.__quantize_array(tensor)` to perform array quantization.
- **Output**: Returns a quantized tensor as an `np.ndarray`.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.can_quantize`](#__Quantcan_quantize)
    - [`llama.cpp/gguf-py/gguf/quants.QuantError`](#cpp/gguf-py/gguf/quantsQuantError)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__quantize_lazy`](#__Quant__quantize_lazy)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__quantize_array`](#__Quant__quantize_array)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)


---
#### \_\_Quant\.dequantize<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.__Quant.dequantize}} -->
The `dequantize` method converts a quantized tensor back to its original floating-point representation, handling both regular and lazy tensors.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `tensor`: A tensor that can be either a numpy ndarray or a LazyNumpyTensor, representing the quantized data to be dequantized.
- **Control Flow**:
    - Check if the input tensor is an instance of LazyNumpyTensor.
    - If it is a LazyNumpyTensor, call the private method [`__dequantize_lazy`](#__Quant__dequantize_lazy) to handle the dequantization.
    - If it is not a LazyNumpyTensor, call the private method [`__dequantize_array`](#__Quant__dequantize_array) to handle the dequantization.
- **Output**: Returns a numpy ndarray representing the dequantized floating-point data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__dequantize_lazy`](#__Quant__dequantize_lazy)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.__dequantize_array`](#__Quant__dequantize_array)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)  (Base Class)



---
### BF16<!-- {{#class:llama.cpp/gguf-py/gguf/quants.BF16}} -->
- **Description**: The `BF16` class is a specialized quantization class that inherits from the `__Quant` abstract base class, specifically implementing the BF16 quantization type from the `GGMLQuantizationType` enumeration. It provides class methods for quantizing and dequantizing blocks of data, converting between full precision and BF16 formats. The `quantize_blocks` method rounds to the nearest even and handles NaN values, while the `dequantize_blocks` method converts BF16 data back to full precision. This class is part of a larger framework for handling various quantization types, facilitating efficient data storage and processing.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.BF16.quantize_blocks`](#BF16quantize_blocks)
    - [`llama.cpp/gguf-py/gguf/quants.BF16.dequantize_blocks`](#BF16dequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### BF16\.quantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.BF16.quantize_blocks}} -->
The `quantize_blocks` method converts a numpy array of blocks from 32-bit floating point to 16-bit bfloat16 format, handling NaN values and rounding to the nearest even number.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy array of blocks to be quantized, expected to be in 32-bit floating point format.
- **Control Flow**:
    - The method first views the input blocks as 32-bit unsigned integers.
    - It then forces NaN values to quiet by checking if the absolute value of each element is greater than the maximum representable float value and adjusting accordingly.
    - The method rounds each element to the nearest even number by adding a bias and shifting right by 16 bits.
    - Finally, it converts the result to 16-bit unsigned integers and views it as 8-bit unsigned integers before returning.
- **Output**: A numpy array of the quantized blocks in 8-bit unsigned integer format, representing the original data in bfloat16 format.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.BF16`](#cpp/gguf-py/gguf/quantsBF16)  (Base Class)


---
#### BF16\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.BF16.dequantize_blocks}} -->
The `dequantize_blocks` method converts quantized blocks of data back into their original floating-point representation.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray containing quantized data blocks that need to be dequantized.
- **Control Flow**:
    - The method first views the input `blocks` as 16-bit integers using `np.int16`.
    - It then converts these integers to 32-bit integers using `np.int32`.
    - The 32-bit integers are left-shifted by 16 bits to adjust their scale.
    - Finally, the shifted integers are viewed as 32-bit floating-point numbers using `np.float32` and returned.
- **Output**: A numpy ndarray of 32-bit floating-point numbers representing the dequantized data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.BF16`](#cpp/gguf-py/gguf/quantsBF16)  (Base Class)



---
### Q4\_0<!-- {{#class:llama.cpp/gguf-py/gguf/quants.Q4_0}} -->
- **Description**: The `Q4_0` class is a specialized quantization class that inherits from the abstract base class `__Quant` and is associated with the `GGMLQuantizationType.Q4_0` quantization type. It provides class methods for quantizing and dequantizing blocks of data using a specific quantization scheme. The quantization process involves determining the maximum absolute value in each block, computing a scaling factor, and then encoding the data into a compact form. The dequantization process reverses this encoding to approximate the original data. This class is part of a larger framework for handling different quantization types, enabling efficient storage and processing of numerical data.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.Q4_0.quantize_blocks`](#Q4_0quantize_blocks)
    - [`llama.cpp/gguf-py/gguf/quants.Q4_0.dequantize_blocks`](#Q4_0dequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### Q4\_0\.quantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q4_0.quantize_blocks}} -->
The `quantize_blocks` method quantizes an array of blocks by scaling and encoding them into a compact format using a specific quantization scheme.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the blocks of data to be quantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Find the index of the maximum absolute value in each block and use it to extract the maximum value for each block.
    - Calculate the divisor `d` by dividing the maximum value by -8.
    - Handle division by zero by setting the inverse divisor `id` to zero where `d` is zero, otherwise compute the inverse of `d`.
    - Quantize the blocks by scaling with `id`, adding an offset, truncating, and clipping the values to fit within a 4-bit range (0 to 15).
    - Reshape the quantized values and combine them into a single byte per pair of values using bitwise operations.
    - Convert the divisor `d` to a 16-bit float and then view it as an 8-bit unsigned integer.
    - Concatenate the quantized values and the converted divisor to form the final output array.
- **Output**: A numpy ndarray containing the quantized representation of the input blocks, with each block encoded into a compact format.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q4_0`](#cpp/gguf-py/gguf/quantsQ4_0)  (Base Class)


---
#### Q4\_0\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q4_0.dequantize_blocks}} -->
The `dequantize_blocks` method converts quantized data blocks back into floating-point values using specific dequantization logic.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized data blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into two parts: `d` and `qs`, where `d` contains the first two columns and `qs` contains the rest.
    - Convert `d` from a view of `np.float16` to `np.float32`.
    - Reshape `qs` and perform bitwise operations to extract and adjust quantization values.
    - Combine `d` and `qs` to compute the final dequantized floating-point values.
- **Output**: A numpy ndarray of dequantized floating-point values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q4_0`](#cpp/gguf-py/gguf/quantsQ4_0)  (Base Class)



---
### Q4\_1<!-- {{#class:llama.cpp/gguf-py/gguf/quants.Q4_1}} -->
- **Description**: The `Q4_1` class is a specialized quantization class that inherits from the `__Quant` abstract base class, specifically designed for handling quantization and dequantization of data blocks using the `GGMLQuantizationType.Q4_1` type. It provides class methods to quantize and dequantize blocks of data, converting them between floating-point and quantized representations. The quantization process involves scaling the data to fit within a 4-bit range, while dequantization reverses this process to approximate the original data values.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.Q4_1.quantize_blocks`](#Q4_1quantize_blocks)
    - [`llama.cpp/gguf-py/gguf/quants.Q4_1.dequantize_blocks`](#Q4_1dequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### Q4\_1\.quantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q4_1.quantize_blocks}} -->
The `quantize_blocks` method quantizes an array of blocks into a compressed format using a specific quantization scheme.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the blocks of data to be quantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Calculate the maximum and minimum values for each block along the last axis.
    - Compute the quantization step size `d` as the difference between max and min divided by 15.
    - Handle division by zero by setting the inverse of `d` to zero where `d` is zero, otherwise compute the inverse.
    - Quantize the blocks by scaling and truncating the values, then convert them to uint8 and clip them to the range [0, 15].
    - Reshape the quantized values and combine them into a single byte per pair of values using bitwise operations.
    - Convert `d` and `min` to float16 and then to uint8 for storage.
    - Concatenate the quantized step sizes, minimum values, and quantized blocks into a single output array.
- **Output**: A numpy ndarray containing the quantized representation of the input blocks, including quantization step sizes, minimum values, and quantized data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q4_1`](#cpp/gguf-py/gguf/quantsQ4_1)  (Base Class)


---
#### Q4\_1\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q4_1.dequantize_blocks}} -->
The `dequantize_blocks` method converts quantized data blocks back into their original floating-point representation using specific dequantization logic.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray containing quantized data blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into three parts: `d`, `m`, and `qs` using numpy's `hsplit` function.
    - Convert `d` and `m` from uint8 to float32 by first viewing them as float16 and then casting to float32.
    - Reshape `qs` and perform bitwise operations to extract the quantized values, then convert them to float32.
    - Compute the dequantized values by multiplying `d` with `qs` and adding `m`.
- **Output**: A numpy ndarray of dequantized floating-point values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q4_1`](#cpp/gguf-py/gguf/quantsQ4_1)  (Base Class)



---
### Q5\_0<!-- {{#class:llama.cpp/gguf-py/gguf/quants.Q5_0}} -->
- **Description**: The `Q5_0` class is a specialized quantization class that inherits from `__Quant` and is associated with the `GGMLQuantizationType.Q5_0` quantization type. It provides class methods for quantizing and dequantizing blocks of data using a specific quantization scheme that involves scaling, rounding, and bit manipulation to compress and decompress data efficiently. The class is designed to handle data in blocks, performing operations such as finding maximum values, scaling, and packing bits to achieve the desired quantization effect.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.Q5_0.quantize_blocks`](#Q5_0quantize_blocks)
    - [`llama.cpp/gguf-py/gguf/quants.Q5_0.dequantize_blocks`](#Q5_0dequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### Q5\_0\.quantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q5_0.quantize_blocks}} -->
The `quantize_blocks` method quantizes an array of blocks into a compressed format using a specific quantization scheme.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the blocks of data to be quantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Find the index of the maximum absolute value in each block and use it to extract the maximum value.
    - Calculate the divisor `d` as the maximum value divided by -16.
    - Handle division by zero by setting the inverse divisor `id` to zero where `d` is zero, otherwise compute the reciprocal of `d`.
    - Quantize the blocks by multiplying with `id`, adding 16.5, truncating, and clipping the result to the range [0, 31].
    - Reshape and combine the quantized values into a compact format using bitwise operations.
    - Pack the higher bits of the quantized values into a separate array using `np.packbits`.
    - Convert the divisor `d` to a float16 view and then to uint8.
    - Concatenate the processed arrays to form the final quantized output.
- **Output**: A numpy ndarray containing the quantized representation of the input blocks.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q5_0`](#cpp/gguf-py/gguf/quantsQ5_0)  (Base Class)


---
#### Q5\_0\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q5_0.dequantize_blocks}} -->
The `dequantize_blocks` method converts quantized data blocks back into floating-point values using specific bit manipulations and transformations.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized data blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into three parts: `d`, `qh`, and `qs`.
    - Convert `d` from a view of float16 to float32 for further calculations.
    - Convert `qh` from a view of uint32 and perform bitwise operations to extract high bits.
    - Reshape and shift `qs` to extract low bits and combine them with high bits to form `qs`.
    - Adjust `qs` by subtracting 16 to center the values around zero.
    - Multiply `d` with the adjusted `qs` to produce the final dequantized output.
- **Output**: A numpy ndarray of dequantized floating-point values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q5_0`](#cpp/gguf-py/gguf/quantsQ5_0)  (Base Class)



---
### Q5\_1<!-- {{#class:llama.cpp/gguf-py/gguf/quants.Q5_1}} -->
- **Description**: The `Q5_1` class is a specialized quantization class that inherits from the `__Quant` abstract base class, specifically implementing the quantization and dequantization processes for the `GGMLQuantizationType.Q5_1` type. It provides class methods to quantize and dequantize blocks of data using a 5-bit quantization scheme, where the quantization process involves scaling the data to fit within a 5-bit range and the dequantization process reconstructs the original data from the quantized form. This class is part of a larger framework for handling different quantization types, each with its own specific implementation.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.Q5_1.quantize_blocks`](#Q5_1quantize_blocks)
    - [`llama.cpp/gguf-py/gguf/quants.Q5_1.dequantize_blocks`](#Q5_1dequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### Q5\_1\.quantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q5_1.quantize_blocks}} -->
The `quantize_blocks` method quantizes an array of blocks into a compressed format using a specific quantization scheme.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the blocks of data to be quantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Calculate the maximum and minimum values for each block along the last axis.
    - Compute the quantization step size `d` as the difference between max and min divided by 31.
    - Handle division by zero by setting the inverse of `d` to zero where `d` is zero, otherwise compute the inverse of `d`.
    - Quantize the blocks by scaling and truncating the values, then convert them to uint8 and clip them to the range [0, 31].
    - Reshape the quantized values and combine them into a single array using bitwise operations.
    - Pack the higher bits of the quantized values into a separate array using `np.packbits`.
    - Convert the quantization step size `d` and minimum values to float16 and view them as uint8.
    - Concatenate the quantization step size, minimum values, packed higher bits, and combined quantized values into a single output array.
- **Output**: A numpy ndarray containing the quantized representation of the input blocks.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q5_1`](#cpp/gguf-py/gguf/quantsQ5_1)  (Base Class)


---
#### Q5\_1\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q5_1.dequantize_blocks}} -->
The `dequantize_blocks` method converts quantized data blocks back into their original floating-point representation using specific dequantization logic.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray containing quantized data blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into components: `d`, `m`, `qh`, and `qs`.
    - Convert `d` and `m` from float16 to float32, and `qh` to uint32.
    - Reshape and bit-shift `qh` to extract high bits, and `qs` to extract low bits.
    - Combine high and low bits to form the dequantized `qs` values.
    - Compute the final dequantized output by multiplying `d` with `qs` and adding `m`.
- **Output**: A numpy ndarray of dequantized floating-point data.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q5_1`](#cpp/gguf-py/gguf/quantsQ5_1)  (Base Class)



---
### Q8\_0<!-- {{#class:llama.cpp/gguf-py/gguf/quants.Q8_0}} -->
- **Description**: The `Q8_0` class is a specialized implementation of the `__Quant` abstract base class, designed to perform quantization and dequantization of data blocks using the Q8_0 quantization type. It provides methods to convert blocks of data into a quantized format and back, ensuring bit-exact results with a reference implementation. The class leverages numpy operations to efficiently handle the conversion of data to and from a compressed format, optimizing storage and computation for specific quantization needs.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.Q8_0.quantize_blocks`](#Q8_0quantize_blocks)
    - [`llama.cpp/gguf-py/gguf/quants.Q8_0.dequantize_blocks`](#Q8_0dequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### Q8\_0\.quantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q8_0.quantize_blocks}} -->
The `quantize_blocks` method quantizes an array of blocks using a specific quantization scheme, producing a bit-exact result as the reference implementation.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `cls`: The class reference, used to call class methods and access class variables.
    - `blocks`: A numpy ndarray representing the blocks of data to be quantized.
- **Control Flow**:
    - Calculate the maximum absolute value of each block and divide by 127 to get the scaling factor `d` for each block.
    - Use numpy's error state management to handle division by zero, setting `id` to 0 where `d` is zero, otherwise compute the inverse of `d`.
    - Multiply the blocks by `id` and round the result using a custom rounding function [`np_roundf`](#cpp/gguf-py/gguf/quantsnp_roundf).
    - Convert `d` to a float16 type and then view it as uint8, and convert the quantized blocks `qs` to int8 and then view as uint8.
    - Concatenate the scaled `d` and quantized `qs` arrays along the second axis to form the final quantized output.
- **Output**: A numpy ndarray containing the quantized data, with the scaling factors and quantized values concatenated.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/quants.np_roundf`](#cpp/gguf-py/gguf/quantsnp_roundf)
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q8_0`](#cpp/gguf-py/gguf/quantsQ8_0)  (Base Class)


---
#### Q8\_0\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q8_0.dequantize_blocks}} -->
The `dequantize_blocks` method converts quantized data blocks back to their original floating-point representation using specific scaling factors.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray containing quantized data blocks, where each block consists of a pair of scaling factors and quantized values.
- **Control Flow**:
    - Split the input `blocks` array into two parts: `d` for scaling factors and `x` for quantized values, using the first two columns for `d` and the rest for `x`.
    - Convert the `d` array from its current view to `np.float16` and then cast it to `np.float32` for higher precision.
    - Convert the `x` array from its current view to `np.int8` and then cast it to `np.float32` for computation.
    - Multiply the `x` array by the `d` array to dequantize the values, resulting in the original floating-point representation.
- **Output**: A numpy ndarray of dequantized floating-point values, representing the original data before quantization.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q8_0`](#cpp/gguf-py/gguf/quantsQ8_0)  (Base Class)



---
### Q2\_K<!-- {{#class:llama.cpp/gguf-py/gguf/quants.Q2_K}} -->
- **Description**: The `Q2_K` class is a specialized subclass of `__Quant` designed for handling dequantization of data blocks using the `GGMLQuantizationType.Q2_K` quantization type. It provides a class method `dequantize_blocks` that processes an array of quantized data blocks, extracting and transforming the data into a dequantized format using specific bit manipulation and scaling techniques. This class is part of a larger framework for quantization and dequantization of data, likely used in machine learning or data compression contexts.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.Q2_K.dequantize_blocks`](#Q2_Kdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### Q2\_K\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q2_K.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific scaling and transformation operations.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into scales, quantized values (qs), and two additional components, d and dmin, using numpy's hsplit function.
    - Convert d and dmin from float16 to float32 for higher precision calculations.
    - Calculate the dequantization factors (dl and ml) by applying bitwise operations on scales and multiplying with d and dmin, respectively.
    - Create a shift array to assist in bit manipulation of the quantized values.
    - Transform the quantized values (qs) by reshaping, shifting, and masking to extract relevant bits, then convert to float32.
    - Apply the dequantization formula by multiplying the transformed quantized values with dl and subtracting ml.
    - Reshape the final dequantized values to the desired output shape.
- **Output**: A numpy ndarray of dequantized floating-point values, reshaped to the appropriate dimensions.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q2_K`](#cpp/gguf-py/gguf/quantsQ2_K)  (Base Class)



---
### Q3\_K<!-- {{#class:llama.cpp/gguf-py/gguf/quants.Q3_K}} -->
- **Description**: The `Q3_K` class is a specialized subclass of `__Quant` designed for handling quantization and dequantization processes specific to the `GGMLQuantizationType.Q3_K` type. It provides a class method `dequantize_blocks` that processes blocks of quantized data, extracting and transforming them into a dequantized format using specific bit manipulation and scaling techniques. This class is part of a larger framework for managing different quantization types, each with its own unique handling and processing methods.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.Q3_K.dequantize_blocks`](#Q3_Kdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### Q3\_K\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q3_K.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific bit manipulation and scaling techniques.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into `hmask`, `qs`, `scales`, and `d` using numpy's `hsplit` function.
    - Convert `d` from a view of `np.float16` to `np.float32`.
    - Unpack the `scales` into `lscales` and `hscales` using bitwise operations and shifts to extract 6-bit scale values.
    - Combine `lscales` and `hscales` to form the final `scales` array, adjusting the scale values by subtracting 32 and converting to `np.float32`.
    - Compute `dl` by multiplying `d` with `scales` and reshaping the result.
    - Extract and manipulate `qs` and `hmask` to compute `ql` and `qh` using bitwise operations and shifts.
    - Adjust `qh` by XORing with 1 to handle bitmask offsets.
    - Calculate the final dequantized values by multiplying `dl` with `q` and reshaping the result to the desired output shape.
- **Output**: A numpy ndarray of dequantized floating-point values with the shape determined by the number of blocks and the constant `QK_K`.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q3_K`](#cpp/gguf-py/gguf/quantsQ3_K)  (Base Class)



---
### Q4\_K<!-- {{#class:llama.cpp/gguf-py/gguf/quants.Q4_K}} -->
- **Members**:
    - `K_SCALE_SIZE`: A constant integer value set to 12, representing the size of the scale.
- **Description**: The `Q4_K` class is a specialized quantization class that extends the `__Quant` abstract base class, specifically designed for handling Q4_K type quantization as defined by the `GGMLQuantizationType`. It includes a constant `K_SCALE_SIZE` and provides static and class methods for dequantizing blocks of data and extracting scale and minimum values from scales. The class is part of a larger framework for quantization and dequantization of data, leveraging numpy for efficient numerical operations.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.Q4_K.get_scale_min`](#Q4_Kget_scale_min)
    - [`llama.cpp/gguf-py/gguf/quants.Q4_K.dequantize_blocks`](#Q4_Kdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### Q4\_K\.get\_scale\_min<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q4_K.get_scale_min}} -->
The `get_scale_min` method processes a numpy array of scales to extract and return two arrays representing scale and minimum values.
- **Decorators**: `@staticmethod`
- **Inputs**:
    - `scales`: A numpy ndarray representing scales, expected to be reshaped and processed to extract scale and minimum values.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input scales array.
    - Convert the scales array to an unsigned 8-bit integer view for bit manipulation.
    - Reshape the scales array into a 3D array with dimensions (n_blocks, 3, 4).
    - Split the reshaped scales array into three parts: d, m, and m_d, along the second axis.
    - Compute the scale values by combining bitwise operations on d and m_d arrays.
    - Compute the minimum values by combining bitwise operations on m and m_d arrays.
    - Reshape the computed scale and minimum arrays to have dimensions (n_blocks, 8).
- **Output**: A tuple containing two numpy ndarrays: the first for scale values and the second for minimum values, both reshaped to (n_blocks, 8).
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q4_K`](#cpp/gguf-py/gguf/quantsQ4_K)  (Base Class)


---
#### Q4\_K\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q4_K.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific scale and minimum values.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into components: `d`, `dmin`, `scales`, and `qs`.
    - Convert `d` and `dmin` from float16 to float32 for higher precision.
    - Retrieve scale and minimum values using the [`get_scale_min`](#Q4_Kget_scale_min) method from the `Q4_K` class.
    - Scale `d` and `dmin` using the retrieved scale and minimum values, reshaping them appropriately.
    - Process `qs` to extract quantized values, adjusting bit positions and converting to float32.
    - Compute the final dequantized values by multiplying the scaled `d` with `qs` and subtracting the scaled `dmin`.
    - Reshape the result to match the expected output dimensions.
- **Output**: A numpy ndarray of dequantized floating-point values with dimensions based on the number of input blocks and the constant `QK_K`.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
    - [`llama.cpp/gguf-py/gguf/quants.Q4_K.get_scale_min`](#Q4_Kget_scale_min)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q4_K`](#cpp/gguf-py/gguf/quantsQ4_K)  (Base Class)



---
### Q5\_K<!-- {{#class:llama.cpp/gguf-py/gguf/quants.Q5_K}} -->
- **Description**: The `Q5_K` class is a specialized quantization class that extends the `__Quant` abstract base class, specifically implementing the dequantization process for the Q5_K quantization type. It utilizes numpy operations to manipulate and transform blocks of data, converting them from a quantized format back to a floating-point representation. The class is part of a larger framework for handling various quantization types, leveraging the `GGMLQuantizationType` enumeration to define its specific quantization behavior.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.Q5_K.dequantize_blocks`](#Q5_Kdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### Q5\_K\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q5_K.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific scaling and bit manipulation techniques.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into components: `d`, `dmin`, `scales`, `qh`, and `qs` using numpy's `hsplit`.
    - Convert `d` and `dmin` from float16 to float32 for precision.
    - Retrieve scale and minimum values using the [`get_scale_min`](#Q4_Kget_scale_min) method from the `Q4_K` class.
    - Compute scaled `d` and `dmin` values using the retrieved scales and minimums.
    - Reshape and bit-shift `qs` and `qh` to extract low and high quantization bits, respectively.
    - Combine the low and high quantization bits to form the complete quantized values `q`.
    - Compute the final dequantized result by applying the scale and offset to the quantized values and reshape the result to the desired output shape.
- **Output**: A numpy ndarray of dequantized floating-point values with the shape determined by the number of blocks and the constant `QK_K`.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
    - [`llama.cpp/gguf-py/gguf/quants.Q4_K.get_scale_min`](#Q4_Kget_scale_min)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q5_K`](#cpp/gguf-py/gguf/quantsQ5_K)  (Base Class)



---
### Q6\_K<!-- {{#class:llama.cpp/gguf-py/gguf/quants.Q6_K}} -->
- **Description**: The `Q6_K` class is a specialized quantization class that inherits from the abstract base class `__Quant`. It is designed to handle the dequantization of data blocks using a specific quantization type, `GGMLQuantizationType.Q6_K`. The class provides a class method `dequantize_blocks` that processes an array of quantized blocks, extracting and transforming the data using bit manipulation and scaling operations to return a dequantized numpy array. This class is part of a larger framework for handling various quantization types, each with its own specific implementation for quantization and dequantization.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.Q6_K.dequantize_blocks`](#Q6_Kdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### Q6\_K\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.Q6_K.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific bit manipulation and scaling techniques.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into four parts: ql, qh, scales, and d using numpy's hsplit function.
    - Convert scales from int8 to float32 and d from float16 to float32, then scale d by scales and reshape it.
    - Perform bitwise operations on ql and qh to extract and combine bits, then reshape and adjust them to form q.
    - Combine ql and qh to form q, adjust its range, and convert it to float32.
    - Multiply the scaled d with q and reshape the result to the desired output shape.
- **Output**: A numpy ndarray of dequantized floating-point values with the same number of blocks as the input.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.Q6_K`](#cpp/gguf-py/gguf/quantsQ6_K)  (Base Class)



---
### TQ1\_0<!-- {{#class:llama.cpp/gguf-py/gguf/quants.TQ1_0}} -->
- **Description**: The `TQ1_0` class is a specialized quantization class that inherits from the abstract base class `__Quant`. It is designed to handle the quantization and dequantization of data blocks using a specific quantization type, `GGMLQuantizationType.TQ1_0`. The class provides class methods to quantize and dequantize blocks of data, transforming them into a more compact form for efficient storage and processing, and then back to their original form. The quantization process involves scaling and rounding operations, while dequantization reverses these transformations to approximate the original data.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.TQ1_0.quantize_blocks`](#TQ1_0quantize_blocks)
    - [`llama.cpp/gguf-py/gguf/quants.TQ1_0.dequantize_blocks`](#TQ1_0dequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### TQ1\_0\.quantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.TQ1_0.quantize_blocks}} -->
The `quantize_blocks` method quantizes an array of blocks by normalizing, rounding, and compressing the data into a smaller representation.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the blocks of data to be quantized.
- **Control Flow**:
    - Calculate the number of blocks from the shape of the input array.
    - Determine the maximum absolute value in each block to use for normalization.
    - Compute the inverse of the maximum values, handling division by zero by setting the inverse to zero where the maximum is zero.
    - Multiply the blocks by the inverse of the maximum values and round the result using a custom rounding function.
    - Convert the rounded values to an unsigned 8-bit integer format with an offset of 1.
    - Split the quantized data into three parts: `qs0`, `qs1`, and `qh`, each representing different sections of the data.
    - Reshape and scale each part using predefined scaling factors, then sum along specific axes to compress the data further.
    - Concatenate the compressed parts back together and apply a final scaling and offset operation to fit the data into an 8-bit format.
    - Convert the maximum values to a 16-bit float and then to an 8-bit unsigned integer view.
    - Concatenate the quantized data with the scaled maximum values to form the final output.
- **Output**: A numpy ndarray containing the quantized representation of the input blocks, including the quantized data and the scaled maximum values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/quants.np_roundf`](#cpp/gguf-py/gguf/quantsnp_roundf)
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.TQ1_0`](#cpp/gguf-py/gguf/quantsTQ1_0)  (Base Class)


---
#### TQ1\_0\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.TQ1_0.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific transformations and scaling factors.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into quantized values (`qs`) and the rest, which includes high bits (`qh`) and scaling factors (`d`).
    - Convert the scaling factors `d` from half-precision to single-precision floating-point format.
    - Separate the quantized values `qs` into two parts, `qs0` and `qs1`, and reshape them for further processing.
    - Multiply `qs0`, `qs1`, and `qh` by specific arrays to transform them into a higher precision format.
    - Concatenate the transformed `qs0`, `qs1`, and `qh` to form a complete quantized value array `qs`.
    - Adjust the quantized values by scaling and shifting to convert them into signed integers.
    - Multiply the adjusted quantized values by the scaling factors `d` to obtain the final dequantized floating-point values.
- **Output**: A numpy ndarray of dequantized floating-point values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.TQ1_0`](#cpp/gguf-py/gguf/quantsTQ1_0)  (Base Class)



---
### TQ2\_0<!-- {{#class:llama.cpp/gguf-py/gguf/quants.TQ2_0}} -->
- **Description**: The `TQ2_0` class is a specialized quantization class that inherits from `__Quant` and is associated with the `GGMLQuantizationType.TQ2_0` quantization type. It provides class methods for quantizing and dequantizing blocks of data using a specific quantization scheme. The quantization process involves scaling and rounding the data, while dequantization reverses this process to approximate the original data. This class is part of a larger framework for handling different quantization types, enabling efficient storage and processing of numerical data.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.TQ2_0.quantize_blocks`](#TQ2_0quantize_blocks)
    - [`llama.cpp/gguf-py/gguf/quants.TQ2_0.dequantize_blocks`](#TQ2_0dequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### TQ2\_0\.quantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.TQ2_0.quantize_blocks}} -->
The `quantize_blocks` method quantizes an array of blocks by scaling and rounding the values, then compresses the data into a more compact form.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the blocks of data to be quantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Calculate the maximum absolute value for each block to use as a scaling factor.
    - Compute the inverse of the scaling factor, handling division by zero by setting the inverse to zero where the scaling factor is zero.
    - Scale and round the blocks using the computed inverse scaling factor.
    - Convert the scaled values to an 8-bit unsigned integer format, adjusting the range by adding 1.
    - Reshape the quantized values and apply bitwise shifts to pack them into a more compact form.
    - Combine the packed values using bitwise OR operations to further compress the data.
    - Reshape the compressed data back into a 2D array format.
    - Convert the scaling factors to a 16-bit float and then to an 8-bit unsigned integer view.
    - Concatenate the compressed quantized values and the scaling factors along the last axis to form the final output.
- **Output**: A numpy ndarray containing the quantized and compressed representation of the input blocks, with scaling factors appended.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/quants.np_roundf`](#cpp/gguf-py/gguf/quantsnp_roundf)
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.TQ2_0`](#cpp/gguf-py/gguf/quantsTQ2_0)  (Base Class)


---
#### TQ2\_0\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.TQ2_0.dequantize_blocks}} -->
The `dequantize_blocks` method converts quantized data blocks back into their original floating-point representation using specific dequantization logic.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized data blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into quantized values (`qs`) and dequantization factors (`d`) using numpy's `hsplit`.
    - Convert the dequantization factors from a view of `np.float16` to `np.float32`.
    - Reshape and bit-shift the quantized values to extract the original quantization levels.
    - Adjust the quantization levels by subtracting 1 to center them around zero.
    - Multiply the adjusted quantization levels by the dequantization factors to obtain the dequantized floating-point values.
- **Output**: A numpy ndarray of dequantized floating-point values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.TQ2_0`](#cpp/gguf-py/gguf/quantsTQ2_0)  (Base Class)



---
### IQ2\_XXS<!-- {{#class:llama.cpp/gguf-py/gguf/quants.IQ2_XXS}} -->
- **Members**:
    - `ksigns`: A byte sequence used for sign manipulation in dequantization.
    - `grid_shape`: A tuple representing the shape of the grid used in quantization.
    - `grid_map`: A tuple mapping specific byte values to indices in the grid.
    - `grid_hex`: A byte sequence representing the hexadecimal values of the grid.
- **Description**: The IQ2_XXS class is a specialized quantization class that extends the __Quant abstract base class, specifically designed for handling the IQ2_XXS quantization type. It defines specific byte sequences and grid configurations used in the quantization and dequantization processes. The class provides a method to dequantize blocks of data, utilizing a predefined grid and sign manipulation to accurately convert quantized data back to its original form. This class is part of a larger framework for handling various quantization types, each with its own unique configuration and processing logic.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.IQ2_XXS.dequantize_blocks`](#IQ2_XXSdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### IQ2\_XXS\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.IQ2_XXS.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific transformations and lookups.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into two parts: `d` and `qs`.
    - Convert `d` from float16 to float32 for further calculations.
    - Reshape `qs` to a specific format for processing.
    - Calculate `db` by applying a transformation involving `d` and a bit-shifted version of `qs`.
    - Reshape `db` to a specific 4D shape for further operations.
    - Extract sign indices from `qs` and unpack the bits using bitwise operations and a predefined `ksigns` array.
    - Use the unpacked sign bits to determine the sign of each element, converting them to either 1 or -1.
    - Ensure the class-level `grid` is initialized and not None.
    - Use the `grid` to transform `qs` into a specific format for dequantization.
    - Multiply `db`, `grid`, and `signs` to produce the final dequantized output.
    - Reshape the final result to a 2D array with the same number of blocks as the input.
- **Output**: A numpy ndarray of dequantized floating-point values, reshaped to a 2D array.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.IQ2_XXS`](#cpp/gguf-py/gguf/quantsIQ2_XXS)  (Base Class)



---
### IQ2\_XS<!-- {{#class:llama.cpp/gguf-py/gguf/quants.IQ2_XS}} -->
- **Members**:
    - `grid_shape`: Defines the shape of the grid as a tuple of two integers.
    - `grid_map`: Maps specific byte values to integers for grid representation.
    - `grid_hex`: Contains the hexadecimal representation of the grid data.
- **Description**: The IQ2_XS class is a specialized quantization class that extends the __Quant abstract base class, specifically designed for handling IQ2_XS type quantization. It defines a grid with a specific shape and mapping, where each byte is packed into 2 bits, and provides a method for dequantizing blocks of data. The class uses a grid defined by a hexadecimal string and a mapping to convert byte values into quantized values, facilitating efficient data storage and retrieval in quantized form.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.IQ2_XS.dequantize_blocks`](#IQ2_XSdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### IQ2\_XS\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.IQ2_XS.dequantize_blocks}} -->
The `dequantize_blocks` method converts quantized data blocks into dequantized floating-point values using specific scaling and sign adjustments.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized data blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into three parts: `d`, `qs`, and `scales`.
    - Convert `d` from half-precision to single-precision floating-point format.
    - Convert `qs` to unsigned 16-bit integers.
    - Reshape and adjust `scales` using bitwise operations to extract scale factors.
    - Compute the dequantized base `db` using `d` and `scales`, and reshape it for further operations.
    - Extract sign information from a predefined byte buffer, adjust it using bitwise operations, and reshape it.
    - Ensure the class-level `grid` is initialized and not `None`.
    - Use `qs` to index into the `grid` to obtain grid values, and reshape them.
    - Combine `db`, `grid`, and `signs` to compute the final dequantized values, and reshape the result to the desired output shape.
- **Output**: A numpy ndarray containing the dequantized floating-point values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.IQ2_XS`](#cpp/gguf-py/gguf/quantsIQ2_XS)  (Base Class)



---
### IQ2\_S<!-- {{#class:llama.cpp/gguf-py/gguf/quants.IQ2_S}} -->
- **Members**:
    - `grid_shape`: Defines the shape of the grid used for quantization.
    - `grid_map`: Maps specific byte values to quantization indices.
    - `grid_hex`: Contains the hexadecimal representation of the quantization grid.
- **Description**: The IQ2_S class is a specialized quantization class that extends the __Quant abstract base class, specifically designed for the IQ2_S quantization type. It defines a grid for quantization where each byte is packed into 2 bits, mapping specific byte values to indices. The class provides a method to dequantize blocks of data, converting them from a quantized format back to a floating-point representation using the defined grid and mapping. This class is part of a larger framework for handling various quantization types, facilitating efficient storage and processing of numerical data.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.IQ2_S.dequantize_blocks`](#IQ2_Sdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### IQ2\_S\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.IQ2_S.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific unpacking and scaling operations.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into components: `d`, `qs`, `signs`, `qh`, and `scales` using numpy's `hsplit`.
    - Convert `d` from float16 to float32 for further calculations.
    - Reshape and adjust `scales` to extract scale values, then compute `db` as a scaled version of `d`.
    - Unpack the sign bits from `signs`, converting them into a sign multiplier array.
    - Unpack and combine `qh` and `qs` to form a complete quantized value array.
    - Ensure the class grid is initialized and not None, then use it to map quantized values to a grid of floating-point values.
    - Multiply the scaled `d` values (`db`) by the grid and sign multipliers to produce the final dequantized output.
- **Output**: A numpy ndarray of dequantized floating-point values, reshaped to match the original block structure.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.IQ2_S`](#cpp/gguf-py/gguf/quantsIQ2_S)  (Base Class)



---
### IQ3\_XXS<!-- {{#class:llama.cpp/gguf-py/gguf/quants.IQ3_XXS}} -->
- **Members**:
    - `grid_shape`: Defines the shape of the grid used in quantization.
    - `grid_map`: Maps specific values to grid indices for quantization.
    - `grid_hex`: Contains hexadecimal data representing the grid for quantization.
- **Description**: The IQ3_XXS class is a specialized quantization class that extends the __Quant abstract base class, specifically designed for the IQ3_XXS quantization type. It defines a grid used for quantization with a specific shape, map, and hexadecimal representation. The class provides a method to dequantize blocks of data, converting them from a quantized format back to a floating-point representation using the defined grid and scale values. This class is part of a larger framework for handling various quantization types, each with its own specific implementation details.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.IQ3_XXS.dequantize_blocks`](#IQ3_XXSdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### IQ3\_XXS\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.IQ3_XXS.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific scaling and sign adjustments.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into components: `d`, `qs`, and `scales`.
    - Convert `d` from float16 to float32 and `scales` to uint32.
    - Calculate `db` by scaling `d` with a factor derived from `scales`.
    - Reshape `db` to a specific shape for further processing.
    - Extract sign indices from `scales` and unpack the bits to determine the sign of each element.
    - Use a predefined `ksigns` array to map the sign indices to actual sign values.
    - Ensure the class-level `grid` is initialized and not None.
    - Use the `qs` values to index into the `grid` to get the dequantization grid values.
    - Combine `db`, `grid`, and `signs` to compute the final dequantized values.
    - Return the reshaped dequantized array.
- **Output**: A numpy ndarray of dequantized floating-point values, reshaped to match the number of input blocks.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.IQ3_XXS`](#cpp/gguf-py/gguf/quantsIQ3_XXS)  (Base Class)



---
### IQ3\_S<!-- {{#class:llama.cpp/gguf-py/gguf/quants.IQ3_S}} -->
- **Members**:
    - `grid_shape`: Defines the shape of the grid as a tuple of two integers.
    - `grid_map`: Maps specific integer values to grid indices.
    - `grid_hex`: Contains a byte string representing the hexadecimal values for the grid.
- **Description**: The IQ3_S class is a specialized quantization class that inherits from the abstract base class __Quant, designed for handling IQ3_S type quantization as defined by the GGMLQuantizationType enumeration. It defines specific grid parameters such as grid_shape, grid_map, and grid_hex, which are used in the quantization and dequantization processes. The class provides a method to dequantize blocks of data, converting them from a quantized format back to a floating-point representation, utilizing the defined grid and scale parameters.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.IQ3_S.dequantize_blocks`](#IQ3_Sdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### IQ3\_S\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.IQ3_S.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific grid and scale transformations.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into components: `d`, `qs`, `qh`, `signs`, and `scales`.
    - Convert `d` from float16 to float32 for precision.
    - Reshape and adjust `scales` using bitwise operations to extract scale factors.
    - Compute `db` by multiplying `d` with the adjusted scales and reshape it for further operations.
    - Unpack the sign bits from `signs`, convert them to float32, and reshape them.
    - Unpack `qh` bits, combine with `qs` to form a complete quantized value.
    - Ensure the class grid is initialized and not None.
    - Use the grid to map quantized values to their dequantized counterparts.
    - Return the final dequantized array by combining `db`, `grid`, and `signs`.
- **Output**: A numpy ndarray of dequantized floating-point values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.IQ3_S`](#cpp/gguf-py/gguf/quantsIQ3_S)  (Base Class)



---
### IQ1\_S<!-- {{#class:llama.cpp/gguf-py/gguf/quants.IQ1_S}} -->
- **Members**:
    - `grid_shape`: Defines the shape of the grid used for quantization.
    - `grid_map`: Maps quantization values to grid indices.
    - `grid_hex`: Hexadecimal representation of the quantization grid.
    - `delta`: A constant value used in dequantization calculations.
- **Description**: The IQ1_S class is a specialized quantization class that extends the abstract __Quant class, specifically implementing the IQ1_S quantization type. It defines a grid for quantization with a specific shape and mapping, and provides a method for dequantizing blocks of data. The class uses a grid packed into 2 bits per byte, mapping values -1, 0, and 1 to indices 0, 1, and 2, respectively. It also includes a delta value used in the dequantization process to adjust the quantized values.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.IQ1_S.dequantize_blocks`](#IQ1_Sdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### IQ1\_S\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.IQ1_S.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific transformations and a predefined grid.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into components: `d`, `qs`, and `qh`.
    - Convert `d` from float16 to float32 and `qh` to uint16.
    - Calculate `dl` by scaling `d` with a factor derived from `qh`.
    - Reshape `dl` and compute `delta` based on the sign bit in `qh`.
    - Adjust `qh` and combine it with `qs` to form a new index for the grid.
    - Ensure the grid is initialized and not None.
    - Use the indices to extract values from the grid and reshape the grid data.
    - Compute the final dequantized values by combining `dl`, `grid`, and `delta`.
- **Output**: A numpy ndarray of dequantized floating-point values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.IQ1_S`](#cpp/gguf-py/gguf/quantsIQ1_S)  (Base Class)



---
### IQ1\_M<!-- {{#class:llama.cpp/gguf-py/gguf/quants.IQ1_M}} -->
- **Members**:
    - `grid_shape`: Inherits the grid shape from IQ1_S.
    - `grid_map`: Inherits the grid map from IQ1_S.
    - `grid_hex`: Inherits the grid hex from IQ1_S.
    - `delta`: Inherits the delta value from IQ1_S.
- **Description**: The IQ1_M class is a specialized quantization class that extends the __Quant abstract base class, specifically designed for the GGMLQuantizationType.IQ1_M quantization type. It inherits several properties from the IQ1_S class, such as grid_shape, grid_map, grid_hex, and delta, which define the quantization grid and scaling factors. The class is unique in its handling of f16 scales, which are stored in multiple parts, and provides a method to dequantize blocks of data by reconstructing the original floating-point values from the quantized representation.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.IQ1_M.dequantize_blocks`](#IQ1_Mdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### IQ1\_M\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.IQ1_M.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific unpacking and scaling operations.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing quantized blocks of data to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into quantized values (`qs`) and the rest, which is further split into quantized high bits (`qh`) and scales.
    - Unpack the scales from multiple bytes and convert them to a floating-point representation (`d`).
    - Calculate the dequantization scale (`dl`) using the unpacked scales and a specific formula.
    - Adjust the quantized high bits (`qh`) and combine them with the quantized values (`qs`) to form a complete quantized representation.
    - Determine the delta values based on the high bits and apply them to the grid values.
    - Ensure the grid is initialized and use it to map the quantized values to their corresponding floating-point values.
    - Return the dequantized floating-point array after applying the scale and delta adjustments.
- **Output**: A numpy ndarray of dequantized floating-point values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.IQ1_M`](#cpp/gguf-py/gguf/quantsIQ1_M)  (Base Class)



---
### IQ4\_NL<!-- {{#class:llama.cpp/gguf-py/gguf/quants.IQ4_NL}} -->
- **Members**:
    - `kvalues`: A tuple of integer values used for dequantization.
- **Description**: The `IQ4_NL` class is a specialized quantization class that inherits from `__Quant` and is associated with the `GGMLQuantizationType.IQ4_NL` quantization type. It defines a specific set of k-values used in the dequantization process, which are applied to transform quantized data back into a more interpretable form. The class provides a method to dequantize blocks of data using these k-values, facilitating the conversion of quantized data into floating-point representations.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.IQ4_NL.dequantize_blocks`](#IQ4_NLdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### IQ4\_NL\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.IQ4_NL.dequantize_blocks}} -->
The `dequantize_blocks` method converts quantized data blocks back into floating-point values using predefined k-values and scaling factors.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized data blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into two parts: `d` and `qs`.
    - Convert `d` from float16 to float32 for precision.
    - Reshape `qs` and perform bitwise operations to extract quantization indices.
    - Map these indices to k-values using numpy's `take_along_axis` function.
    - Multiply the dequantized `qs` values with `d` to get the final dequantized output.
- **Output**: A numpy ndarray of dequantized floating-point values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.IQ4_NL`](#cpp/gguf-py/gguf/quantsIQ4_NL)  (Base Class)



---
### IQ4\_XS<!-- {{#class:llama.cpp/gguf-py/gguf/quants.IQ4_XS}} -->
- **Description**: The `IQ4_XS` class is a specialized quantization class that inherits from `__Quant` and is associated with the `GGMLQuantizationType.IQ4_XS` quantization type. It provides a method for dequantizing blocks of data, which involves splitting the input blocks into components, processing scales and quantized values, and then reconstructing the original data using these components. The class leverages numpy operations for efficient data manipulation and is part of a larger framework for handling different quantization types.
- **Methods**:
    - [`llama.cpp/gguf-py/gguf/quants.IQ4_XS.dequantize_blocks`](#IQ4_XSdequantize_blocks)
- **Inherits From**:
    - [`llama.cpp/gguf-py/gguf/quants.__Quant`](#cpp/gguf-py/gguf/quants__Quant)

**Methods**

---
#### IQ4\_XS\.dequantize\_blocks<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.IQ4_XS.dequantize_blocks}} -->
The `dequantize_blocks` method dequantizes a given array of quantized blocks into a floating-point representation using specific scaling and transformation operations.
- **Decorators**: `@classmethod`
- **Inputs**:
    - `blocks`: A numpy ndarray representing the quantized blocks to be dequantized.
- **Control Flow**:
    - Determine the number of blocks from the shape of the input array.
    - Split the input blocks into components: `d`, `scales_h`, `scales_l`, and `qs`.
    - Convert `d` from float16 to float32 and `scales_h` from uint16.
    - Reshape and bit-shift `scales_l` and `scales_h` to extract scale values.
    - Combine `scales_l` and `scales_h` to form the final scales, adjusting for offset.
    - Compute `dl` by multiplying `d` with the scales and reshaping.
    - Reshape and bit-shift `qs` to extract quantized values.
    - Map quantized values to `kvalues` using `np.take_along_axis` and convert to float32.
    - Multiply `dl` with the transformed `qs` to get the dequantized result.
    - Return the reshaped dequantized result.
- **Output**: A numpy ndarray of dequantized floating-point values.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
- **See also**: [`llama.cpp/gguf-py/gguf/quants.IQ4_XS`](#cpp/gguf-py/gguf/quantsIQ4_XS)  (Base Class)



# Functions

---
### quant\_shape\_to\_byte\_shape<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.quant_shape_to_byte_shape}} -->
The function `quant_shape_to_byte_shape` converts a quantized tensor shape to its corresponding byte shape based on the quantization type.
- **Inputs**:
    - `shape`: A sequence of integers representing the dimensions of the quantized tensor.
    - `quant_type`: An instance of `GGMLQuantizationType` that specifies the type of quantization applied to the tensor.
- **Control Flow**:
    - Retrieve the block size and type size from the `GGML_QUANT_SIZES` dictionary using the `quant_type` as the key.
    - Check if the last dimension of the `shape` is a multiple of the block size; if not, raise a `ValueError`.
    - Return a new shape tuple where the last dimension is adjusted by dividing it by the block size and multiplying by the type size.
- **Output**: A tuple of integers representing the byte shape of the quantized tensor.


---
### quant\_shape\_from\_byte\_shape<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.quant_shape_from_byte_shape}} -->
The function `quant_shape_from_byte_shape` converts a tensor's byte shape to its quantized shape based on the specified quantization type.
- **Inputs**:
    - `shape`: A sequence of integers representing the dimensions of the tensor in bytes.
    - `quant_type`: An instance of `GGMLQuantizationType` that specifies the quantization type to be used.
- **Control Flow**:
    - Retrieve the block size and type size from the `GGML_QUANT_SIZES` dictionary using the provided `quant_type`.
    - Check if the last dimension of `shape` is a multiple of `type_size`. If not, raise a `ValueError`.
    - Return a new shape tuple where the last dimension is adjusted by dividing by `type_size` and multiplying by `block_size`.
- **Output**: A tuple of integers representing the quantized shape of the tensor.


---
### \_apply\_over\_grouped\_rows<!-- {{#callable:llama.cpp/gguf-py/gguf/quants._apply_over_grouped_rows}} -->
The function `_apply_over_grouped_rows` applies a given function to groups of rows in a numpy array and reshapes the result to a specified output shape.
- **Inputs**:
    - `func`: A callable function that takes a numpy array as input and returns a numpy array.
    - `arr`: A numpy array to which the function will be applied.
    - `otype`: The desired data type of the output array.
    - `oshape`: A tuple representing the desired shape of the output array.
- **Control Flow**:
    - The input array `arr` is reshaped into a 2D array `rows` where each row corresponds to the last dimension of `arr`.
    - The total size of the output array is calculated by multiplying the dimensions in `oshape`.
    - An empty numpy array `out` is created with the calculated size and specified data type `otype`.
    - The number of groups is determined by dividing the number of rows by 16, defaulting to 1 if the result is zero.
    - The input function `func` is applied to each group of rows, and the results are concatenated into the `out` array.
    - The concatenated result is reshaped to the specified `oshape` and returned.
- **Output**: A numpy array with the specified output shape `oshape`, containing the results of applying `func` to the input array `arr`.


---
### np\_roundf<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.np_roundf}} -->
The `np_roundf` function rounds elements of a numpy array away from zero.
- **Inputs**:
    - `n`: A numpy ndarray containing the elements to be rounded.
- **Control Flow**:
    - Calculate the absolute value of the input array `n` and store it in `a`.
    - Compute the floored values of `a` and store them in `floored`.
    - Calculate `b` by adding `floored` to the floored values of `2 * (a - floored)`.
    - Return the product of the sign of `n` and `b`, effectively rounding each element of `n` away from zero.
- **Output**: A numpy ndarray with the same shape as `n`, containing the rounded values.


---
### quantize<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.quantize}} -->
The [`quantize`](#__Quantquantize) function converts a numpy array to a specified quantization type, either by changing its data type or using a custom quantization method.
- **Inputs**:
    - `data`: A numpy array (`np.ndarray`) that needs to be quantized.
    - `qtype`: An instance of `GGMLQuantizationType` that specifies the desired quantization type.
- **Control Flow**:
    - Check if `qtype` is `GGMLQuantizationType.F32`; if true, convert `data` to `np.float32` without copying and return.
    - Check if `qtype` is `GGMLQuantizationType.F16`; if true, convert `data` to `np.float16` without copying and return.
    - Attempt to retrieve a quantization method from `_type_traits` using `qtype`; if found, use it to quantize `data` and return the result.
    - If no matching quantization method is found, raise a `NotImplementedError` indicating the quantization type is not implemented.
- **Output**: Returns a numpy array (`np.ndarray`) that has been quantized according to the specified `qtype`.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.quantize`](#__Quantquantize)


---
### dequantize<!-- {{#callable:llama.cpp/gguf-py/gguf/quants.dequantize}} -->
The [`dequantize`](#__Quantdequantize) function converts quantized data back to its original floating-point representation based on the specified quantization type.
- **Inputs**:
    - `data`: A numpy ndarray containing the quantized data to be dequantized.
    - `qtype`: An instance of GGMLQuantizationType indicating the type of quantization used on the data.
- **Control Flow**:
    - Check if the quantization type is F32; if so, return the data viewed as np.float32.
    - Check if the quantization type is F16; if so, return the data viewed as np.float16 and then cast to np.float32.
    - Check if the quantization type exists in the _type_traits dictionary; if so, use the corresponding dequantize method to dequantize the data.
    - If none of the above conditions are met, raise a NotImplementedError indicating that dequantization for the given type is not implemented.
- **Output**: Returns a numpy ndarray of the dequantized data in np.float32 format.
- **Functions called**:
    - [`llama.cpp/gguf-py/gguf/lazy.LazyNumpyTensor.astype`](lazy.py.driver.md#LazyNumpyTensorastype)
    - [`llama.cpp/gguf-py/gguf/quants.__Quant.dequantize`](#__Quantdequantize)


