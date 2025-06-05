# Purpose
The provided C++ source code defines a comprehensive template engine within the `minja` namespace, which is designed to parse, evaluate, and render templates similar to those used in web frameworks like Jinja2 for Python. The code is structured around several key components, including classes for handling expressions, template nodes, and context management. The [`Value`](#ValueValue) class is central to the system, providing a flexible representation of data that can mimic Python-like behavior, supporting operations on arrays, objects, and primitive types. This class is heavily utilized throughout the code to manage data within templates.

The code also includes a [`Parser`](#ParserParser) class that tokenizes and parses template strings into a series of [`TemplateToken`](#TemplateTokenTemplateToken) objects, which are then transformed into a tree of [`TemplateNode`](#TemplateNodeTemplateNode) objects. These nodes represent different parts of the template, such as text, expressions, and control structures like loops and conditionals. The [`Context`](#ContextContext) class manages variable scopes and built-in functions, allowing templates to access and manipulate data dynamically. The code is designed to be extensible, with support for custom filters and functions, making it suitable for a wide range of templating tasks. The use of the `nlohmann::json` library facilitates JSON-like data manipulation, enhancing the engine's capabilities in handling complex data structures.
# Imports and Dependencies

---
- `algorithm`
- `cctype`
- `cstddef`
- `cstdint`
- `cmath`
- `exception`
- `functional`
- `iostream`
- `iterator`
- `limits`
- `map`
- `memory`
- `regex`
- `sstream`
- `string`
- `stdexcept`
- `unordered_map`
- `unordered_set`
- `utility`
- `vector`
- `nlohmann/json.hpp`


# Data Structures

---
### Options<!-- {{#data_structure:minja::Options}} -->
- **Type**: `struct`
- **Members**:
    - `trim_blocks`: Indicates whether to remove the first newline after a block.
    - `lstrip_blocks`: Indicates whether to remove leading whitespace on the line of the block.
    - `keep_trailing_newline`: Indicates whether to keep the last newline.
- **Description**: The `Options` struct is used to configure the behavior of a templating engine, specifically regarding whitespace handling around blocks in templates. It contains three boolean fields that control whether to trim blocks, strip leading whitespace, and keep trailing newlines.


---
### Value<!-- {{#data_structure:minja::Value}} -->
- **Type**: `class`
- **Members**:
    - `array_`: A shared pointer to an array of `Value` objects.
    - `object_`: A shared pointer to an object that maps `json` keys to `Value` objects.
    - `callable_`: A shared pointer to a callable function that returns a `Value`.
    - `primitive_`: A `json` object representing a primitive value.
- **Description**: The `Value` class is a versatile data structure that can represent various types of data, including arrays, objects, callable functions, and primitive values. It provides methods for manipulating and accessing these data types, allowing for operations similar to those found in dynamic languages like Python. The class supports features such as type checking, serialization to JSON, and dynamic method invocation, making it suitable for use in contexts where flexible data representation is required.
- **Member Functions**:
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::dump_string`](#Valuedump_string)
    - [`minja::Value::dump`](#Valuedump)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::keys`](#Valuekeys)
    - [`minja::Value::size`](#Valuesize)
    - [`minja::Value::array`](#Valuearray)
    - [`minja::Value::object`](#Valueobject)
    - [`minja::Value::callable`](#Valuecallable)
    - [`minja::Value::insert`](#Valueinsert)
    - [`minja::Value::push_back`](#Valuepush_back)
    - [`minja::Value::pop`](#Valuepop)
    - [`minja::Value::get`](#Valueget)
    - [`minja::Value::set`](#Valueset)
    - [`minja::Value::call`](#Valuecall)
    - [`minja::Value::is_object`](#Valueis_object)
    - [`minja::Value::is_array`](#Valueis_array)
    - [`minja::Value::is_callable`](#Valueis_callable)
    - [`minja::Value::is_null`](#Valueis_null)
    - [`minja::Value::is_boolean`](#Valueis_boolean)
    - [`minja::Value::is_number_integer`](#Valueis_number_integer)
    - [`minja::Value::is_number_float`](#Valueis_number_float)
    - [`minja::Value::is_number`](#Valueis_number)
    - [`minja::Value::is_string`](#Valueis_string)
    - [`minja::Value::is_iterable`](#Valueis_iterable)
    - [`minja::Value::is_primitive`](#Valueis_primitive)
    - [`minja::Value::is_hashable`](#Valueis_hashable)
    - [`minja::Value::empty`](#Valueempty)
    - [`minja::Value::for_each`](#Valuefor_each)
    - [`minja::Value::to_bool`](#Valueto_bool)
    - [`minja::Value::to_int`](#Valueto_int)
    - [`minja::Value::operator<`](#Valueoperator<)
    - [`minja::Value::operator>=`](#Valueoperator>=)
    - [`minja::Value::operator>`](#Valueoperator>)
    - [`minja::Value::operator<=`](#Valueoperator<=)
    - [`minja::Value::operator==`](#Valueoperator==)
    - [`minja::Value::operator!=`](#Valueoperator!=)
    - [`minja::Value::contains`](#Valuecontains)
    - [`minja::Value::contains`](#Valuecontains)
    - [`minja::Value::contains`](#Valuecontains)
    - [`minja::Value::erase`](#Valueerase)
    - [`minja::Value::erase`](#Valueerase)
    - [`minja::Value::at`](#Valueat)
    - [`minja::Value::at`](#Valueat)
    - [`minja::Value::at`](#Valueat)
    - [`minja::Value::at`](#Valueat)
    - [`minja::Value::get`](#Valueget)
    - [`minja::Value::get`](#Valueget)
    - [`minja::Value::dump`](#Valuedump)
    - [`minja::Value::operator-`](#Valueoperator-)
    - [`minja::Value::to_str`](#Valueto_str)
    - [`minja::Value::operator+`](#Valueoperator+)
    - [`minja::Value::operator-`](#Valueoperator-)
    - [`minja::Value::operator*`](#Valueoperator*)
    - [`minja::Value::operator/`](#Valueoperator/)
    - [`minja::Value::operator%`](#Valueoperator%)
- **Inherits From**:
    - `std::enable_shared_from_this<Value>`

**Methods**

---
#### Value::Value<!-- {{#callable:minja::Value::Value}} -->
Constructs a `Value` object from either an array or an object.
- **Inputs**:
    - `array`: A shared pointer to an `ArrayType`, which is a vector of `Value` objects.
    - `object`: A shared pointer to an `ObjectType`, which is an ordered map containing `json` keys and `Value` objects.
- **Control Flow**:
    - The constructor initializes the `Value` object based on the type of input provided.
    - If an `ArrayType` is provided, it initializes the `array_` member variable.
    - If an `ObjectType` is provided, it initializes the `object_` member variable.
- **Output**: The constructor does not return a value but initializes the `Value` object with the provided array or object.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::Value<!-- {{#callable:minja::Value::Value}} -->
Constructs a `Value` object from either an `ObjectType`, `CallableType`, or an `ArrayType`.
- **Inputs**:
    - `object`: A shared pointer to an `ObjectType`, which is an ordered map containing primitive keys.
    - `callable`: A shared pointer to a `CallableType`, which is a function that takes a context and arguments.
- **Control Flow**:
    - The constructor initializes the `object_` member with the provided `object` if it is of type `ObjectType`.
    - If a `callable` is provided, it initializes `object_` with a new `ObjectType` and sets the `callable_` member to the provided callable.
- **Output**: The constructor does not return a value but initializes a `Value` object with the specified properties.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::Value<!-- {{#callable:minja::Value::Value}} -->
Constructs a `Value` object that holds a callable function.
- **Inputs**:
    - `callable`: A shared pointer to a `CallableType`, which is a function that takes a shared pointer to a `Context` and an `ArgumentsValue` reference, and returns a `Value`.
- **Control Flow**:
    - The constructor initializes the `object_` member with a new shared pointer to an `ObjectType`, which is an ordered map.
    - It assigns the provided `callable` argument to the `callable_` member.
- **Output**: The constructor does not return a value but initializes a `Value` object with the specified callable.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::dump\_string<!-- {{#callable:minja::Value::dump_string}} -->
The `dump_string` function formats a JSON string value for output, handling escape sequences and custom quote characters.
- **Inputs**:
    - `primitive`: A `json` object expected to be a string.
    - `out`: An `std::ostringstream` object where the formatted string will be written.
    - `string_quote`: A character used to quote the string in the output, defaulting to single quote.
- **Control Flow**:
    - The function first checks if the `primitive` is a string; if not, it throws a runtime error.
    - It then retrieves the string representation of `primitive` using `dump()`.
    - If the `string_quote` is a double quote or if the string contains a single quote, it directly writes the string to `out`.
    - If the `string_quote` is a single quote, it begins writing the string with the quote, escaping any occurrences of the quote within the string.
    - Finally, it closes the string with the same quote character.
- **Output**: The function outputs a formatted string representation of the JSON string value to the provided `std::ostringstream`.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::dump<!-- {{#callable:minja::Value::dump}} -->
The `dump` method serializes the `Value` object into a string representation, optionally formatted as JSON.
- **Inputs**:
    - `out`: An `std::ostringstream` reference where the serialized output will be written.
    - `indent`: An integer specifying the number of spaces to use for indentation; defaults to -1 (no indentation).
    - `level`: An integer representing the current depth level for indentation; defaults to 0.
    - `to_json`: A boolean flag indicating whether to format the output as JSON; defaults to false.
- **Control Flow**:
    - The method begins by defining two lambda functions: `print_indent` for handling indentation and `print_sub_sep` for printing separators between elements.
    - It checks if the `Value` is null, an array, an object, callable, boolean, string, or a primitive type, and processes each case accordingly.
    - For arrays, it iterates through the elements, recursively calling `dump` on each element.
    - For objects, it iterates through key-value pairs, dumping each key and value, handling string keys differently.
    - If the `Value` is callable, it throws an exception, as callable types cannot be serialized.
    - For boolean and string types, it outputs their respective string representations.
    - Finally, it outputs the primitive value's string representation.
- **Output**: The output is a string representation of the `Value` object, formatted according to the specified options (JSON or plain text).
- **Functions called**:
    - [`minja::Value::is_null`](#Valueis_null)
    - [`minja::Value::dump_string`](#Valuedump_string)
    - [`minja::Value::is_boolean`](#Valueis_boolean)
    - [`minja::Value::is_string`](#Valueis_string)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::Value<!-- {{#callable:minja::Value::Value}} -->
Constructs a `Value` object, which can represent various data types including boolean.
- **Inputs**:
    - `v`: A boolean value used to initialize the `primitive_` member of the `Value` class.
- **Control Flow**:
    - The constructor initializes the `primitive_` member with the provided boolean value.
    - If no arguments are provided, the default constructor initializes an empty `Value` object.
- **Output**: The constructor does not return a value but initializes a `Value` object that can represent a boolean.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::Value<!-- {{#callable:minja::Value::Value}} -->
Constructs a `Value` object from a boolean or an integer.
- **Inputs**:
    - `v`: A boolean value used to initialize the `Value` object.
    - `v`: An integer value used to initialize the `Value` object.
- **Control Flow**:
    - The constructor initializes the `primitive_` member variable with the provided boolean or integer value.
    - If a boolean is passed, it sets `primitive_` to that boolean value.
    - If an integer is passed, it sets `primitive_` to that integer value.
- **Output**: The constructor does not return a value but initializes a `Value` object with the specified primitive type.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::Value<!-- {{#callable:minja::Value::Value}} -->
Constructs a `Value` object from either an `int64_t` or a `double`.
- **Inputs**:
    - `v`: An integer value of type `int64_t` to initialize the `Value` object.
    - `v`: A floating-point value of type `double` to initialize the `Value` object.
- **Control Flow**:
    - The constructor initializes the `primitive_` member variable with the provided value.
    - There are two overloads of the constructor, one for `int64_t` and another for `double`, allowing for different types of numeric initialization.
- **Output**: The constructor does not return a value but initializes a `Value` object with the specified numeric type.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::Value<!-- {{#callable:minja::Value::Value}} -->
Constructs a `Value` object from a double or a null pointer.
- **Inputs**:
    - `v`: A constant reference to a double value used to initialize the `primitive_` member.
    - `nullptr`: A null pointer used to create a `Value` object representing a null value.
- **Control Flow**:
    - The constructor initializes the `primitive_` member with the provided double value.
    - If a null pointer is passed, it initializes the `Value` object to represent a null value.
- **Output**: The constructor does not return a value but initializes a `Value` object.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::Value<!-- {{#callable:minja::Value::Value}} -->
Constructs a `Value` object from various types including `nullptr`, `std::string`, and others.
- **Inputs**:
    - `v`: A `std::string` representing a primitive value.
    - `nullptr_t`: A `nullptr_t` type used to create a null `Value`.
- **Control Flow**:
    - The constructor checks the type of the input argument.
    - If the input is a `std::string`, it initializes the `primitive_` member with the string value.
    - If the input is a `nullptr_t`, it initializes the `Value` as a null object.
- **Output**: The constructor initializes a `Value` object, either as a string primitive or as a null value.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::Value<!-- {{#callable:minja::Value::Value}} -->
Constructs a `Value` object from a string or character pointer.
- **Inputs**:
    - `v`: A constant reference to a `std::string` that initializes the `primitive_` member.
    - `v`: A pointer to a character array (C-style string) that is converted to a `std::string` and initializes the `primitive_` member.
- **Control Flow**:
    - The constructor initializes the `primitive_` member variable with the provided string.
    - If a `const char*` is provided, it is converted to a `std::string` before assignment.
- **Output**: The constructor does not return a value but initializes a `Value` object with the specified primitive value.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::Value<!-- {{#callable:minja::Value::Value}} -->
Constructs a `Value` object from a C-style string.
- **Inputs**:
    - `v`: A pointer to a null-terminated C-style string that will be converted to a `std::string`.
- **Control Flow**:
    - The constructor initializes the `primitive_` member variable with a `std::string` created from the input C-style string.
    - The `std::string` constructor handles the conversion and memory management of the string.
- **Output**: This constructor does not return a value but initializes a `Value` object with the string representation of the input.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::Value<!-- {{#callable:minja::Value::Value}} -->
Constructs a [`Value`](#ValueValue) object from a `json` input, initializing it as an object, array, or primitive based on the type of the input.
- **Inputs**:
    - `v`: A `json` object that can be an object, array, or primitive value, which determines how the [`Value`](#ValueValue) instance is initialized.
- **Control Flow**:
    - Checks if the input `json` is an object using `v.is_object()`. If true, it creates a shared pointer to an `ObjectType` and populates it with key-value pairs from the `json` object.
    - If the input is an array (checked using `v.is_array()`), it creates a shared pointer to an `ArrayType` and populates it with [`Value`](#ValueValue) instances created from each item in the `json` array.
    - If the input is neither an object nor an array, it assigns the primitive value directly to the `primitive_` member.
- **Output**: The function does not return a value; instead, it initializes the [`Value`](#ValueValue) object with the appropriate type (object, array, or primitive) based on the input `json`.
- **Functions called**:
    - [`minja::Value::Value`](#ValueValue)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::keys<!-- {{#callable:minja::Value::keys}} -->
The `keys` method retrieves the keys of the `Value` object if it is an object.
- **Inputs**: None
- **Control Flow**:
    - Checks if the `object_` member is null; if it is, throws a runtime error indicating that the value is not an object.
    - Initializes an empty vector `res` to store the keys.
    - Iterates over each item in the `object_`, pushing the first element of each key-value pair into the `res` vector.
    - Returns the populated `res` vector containing the keys.
- **Output**: Returns a vector of `Value` objects representing the keys of the `Value` object.
- **Functions called**:
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::size<!-- {{#callable:minja::Value::size}} -->
The `size` method returns the size of the `Value` object, which can be an array, object, or string.
- **Inputs**: None
- **Control Flow**:
    - The method first checks if the `Value` is an object using the `is_object()` method; if true, it returns the size of the object.
    - If the `Value` is not an object, it checks if it is an array using the `is_array()` method; if true, it returns the size of the array.
    - If the `Value` is not an array, it checks if it is a string using the `is_string()` method; if true, it returns the length of the string.
    - If none of the above conditions are met, it throws a runtime error indicating that the value is neither an array nor an object.
- **Output**: The output is a `size_t` representing the size of the `Value` if it is an object, array, or the length of a string; otherwise, an exception is thrown.
- **Functions called**:
    - [`minja::Value::is_object`](#Valueis_object)
    - [`minja::Value::is_array`](#Valueis_array)
    - [`minja::Value::is_string`](#Valueis_string)
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::array<!-- {{#callable:minja::Value::array}} -->
Creates a [`Value`](#ValueValue) object representing an array initialized with the provided values.
- **Inputs**:
    - `values`: A vector of [`Value`](#ValueValue) objects to be included in the array, defaulting to an empty vector if not provided.
- **Control Flow**:
    - A shared pointer to an `ArrayType` is created to hold the array elements.
    - A for loop iterates over each item in the `values` vector.
    - Each item is pushed back into the `ArrayType` shared pointer.
    - Finally, a [`Value`](#ValueValue) object is returned, encapsulating the shared pointer to the array.
- **Output**: Returns a [`Value`](#ValueValue) object that contains a shared pointer to the newly created array.
- **Functions called**:
    - [`minja::Value::Value`](#ValueValue)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::object<!-- {{#callable:minja::Value::object}} -->
Creates a [`Value`](#ValueValue) object from a shared pointer to an `ObjectType`, defaulting to a new `ObjectType` if none is provided.
- **Inputs**:
    - `object`: A shared pointer to an `ObjectType`, which is an ordered map of JSON keys to [`Value`](#ValueValue) objects. If not provided, a new `ObjectType` is created.
- **Control Flow**:
    - The function checks if the `object` parameter is provided.
    - If `object` is not provided, it creates a new instance of `ObjectType` using `std::make_shared<ObjectType>()`.
    - Finally, it returns a [`Value`](#ValueValue) object initialized with the provided or newly created `object`.
- **Output**: Returns a [`Value`](#ValueValue) object that encapsulates the provided or newly created `ObjectType`.
- **Functions called**:
    - [`minja::Value::Value`](#ValueValue)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::callable<!-- {{#callable:minja::Value::callable}} -->
Creates a [`Value`](#ValueValue) object that encapsulates a callable function.
- **Inputs**:
    - `callable`: A reference to a `CallableType`, which is a function that takes a `Context` pointer and `ArgumentsValue` reference, returning a [`Value`](#ValueValue).
- **Control Flow**:
    - The function takes a `CallableType` as input.
    - It creates a shared pointer to the `CallableType` using `std::make_shared`.
    - It then constructs a [`Value`](#ValueValue) object using this shared pointer.
- **Output**: Returns a [`Value`](#ValueValue) object that contains the callable function.
- **Functions called**:
    - [`minja::Value::Value`](#ValueValue)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::insert<!-- {{#callable:minja::Value::insert}} -->
Inserts a `Value` object into an array at a specified index.
- **Inputs**:
    - `index`: The position in the array where the new value will be inserted.
    - `v`: The `Value` object to be inserted into the array.
- **Control Flow**:
    - Checks if the `array_` member is null; if it is, a runtime error is thrown indicating that the value is not an array.
    - If the array is valid, the `insert` method of the underlying array is called to insert the value at the specified index.
- **Output**: The function does not return a value; it modifies the array in place.
- **Functions called**:
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::push\_back<!-- {{#callable:minja::Value::push_back}} -->
Adds a new `Value` to the end of the `array_` if it is a valid array.
- **Inputs**:
    - `v`: A constant reference to a `Value` object that is to be added to the array.
- **Control Flow**:
    - Checks if `array_` is null, indicating that the `Value` is not an array.
    - If `array_` is null, throws a runtime error with a message that includes the result of the `dump()` method.
    - If `array_` is valid, calls the `push_back` method on `array_` to add the new `Value`.
- **Output**: The function does not return a value; it modifies the internal state of the `Value` object by adding a new element to the array.
- **Functions called**:
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::pop<!-- {{#callable:minja::Value::pop}} -->
Removes and returns the last element from an array or a key-value pair from an object based on the provided index.
- **Inputs**:
    - `index`: A `Value` object representing the index of the element to be removed from an array or the key of the element to be removed from an object. If null, the last element of the array is removed.
- **Control Flow**:
    - Checks if the current `Value` instance is an array.
    - If the array is empty, throws a runtime error.
    - If the `index` is null, retrieves and removes the last element from the array.
    - If the `index` is not null, checks if it is an integer; if not, throws a runtime error.
    - Converts the index to an integer and checks if it is within the valid range of the array.
    - If valid, retrieves the element at the specified index and removes it from the array.
    - If the current `Value` instance is an object, checks if the `index` is hashable and retrieves the corresponding value if the key exists, otherwise throws a runtime error.
    - If the `Value` is neither an array nor an object, throws a runtime error.
- **Output**: Returns the removed `Value` object, which can be either an element from an array or a value associated with a key in an object.
- **Functions called**:
    - [`minja::Value::is_array`](#Valueis_array)
    - [`minja::Value::is_object`](#Valueis_object)
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::get<!-- {{#callable:minja::Value::get}} -->
Retrieves a [`Value`](#ValueValue) from either an array or an object based on the provided key.
- **Inputs**:
    - `key`: A [`Value`](#ValueValue) representing the key used to access an element in the array or object.
- **Control Flow**:
    - Checks if the [`Value`](#ValueValue) instance has an associated array (`array_`).
    - If an array is present, it verifies if the `key` is an integer; if not, it returns a default [`Value`](#ValueValue).
    - If the `key` is valid, it calculates the index and retrieves the corresponding element from the array.
    - If no array is present, it checks for an associated object (`object_`).
    - If an object is present, it verifies if the `key` is hashable; if not, it throws an error.
    - It then attempts to find the `key` in the object; if not found, it returns a default [`Value`](#ValueValue).
    - If the `key` is found, it returns the associated value from the object.
    - If neither an array nor an object is present, it returns a default [`Value`](#ValueValue).
- **Output**: Returns the [`Value`](#ValueValue) associated with the provided key, or a default [`Value`](#ValueValue) if the key is not found or if the instance is neither an array nor an object.
- **Functions called**:
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::set<!-- {{#callable:minja::Value::set}} -->
Sets a key-value pair in a `Value` object if it is an object and the key is hashable.
- **Inputs**:
    - `key`: A `Value` object representing the key to be set in the object.
    - `value`: A `Value` object representing the value to be associated with the key.
- **Control Flow**:
    - Checks if the `Value` object is not null and is indeed an object; if not, throws a runtime error.
    - Checks if the `key` is hashable; if not, throws a runtime error.
    - Sets the `value` in the `object_` using the `key`'s primitive representation.
- **Output**: No return value; the function modifies the internal state of the `Value` object.
- **Functions called**:
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::call<!-- {{#callable:minja::Value::call}} -->
Invokes a callable object with the provided context and arguments.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the execution context for the callable.
    - `args`: A reference to an `ArgumentsValue` object containing the arguments to be passed to the callable.
- **Control Flow**:
    - Checks if the `callable_` member is null; if it is, throws a runtime error indicating that the value is not callable.
    - If `callable_` is valid, it dereferences the callable and invokes it with the provided `context` and `args`.
- **Output**: Returns the result of invoking the callable, which is of type `Value`.
- **Functions called**:
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::is\_object<!-- {{#callable:minja::Value::is_object}} -->
Determines if the `Value` instance represents an object.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the `object_` member variable is not null.
    - It uses the double negation operator (!!) to convert the pointer to a boolean value.
- **Output**: Returns true if the `Value` instance is an object, otherwise returns false.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::is\_array<!-- {{#callable:minja::Value::is_array}} -->
The `is_array` method checks if the `Value` instance contains an array.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `Value` class.
- **Control Flow**:
    - The method uses the logical NOT operator to convert the pointer `array_` to a boolean value.
    - It returns true if `array_` is not null, indicating that the instance represents an array.
- **Output**: Returns a boolean value indicating whether the `Value` instance is an array.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::is\_callable<!-- {{#callable:minja::Value::is_callable}} -->
Checks if the `Value` instance has a callable associated with it.
- **Inputs**: None
- **Control Flow**:
    - The function evaluates the `callable_` member variable.
    - It returns true if `callable_` is not null, otherwise it returns false.
- **Output**: Returns a boolean indicating whether the `Value` instance is callable.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::is\_null<!-- {{#callable:minja::Value::is_null}} -->
Checks if the `Value` instance is null.
- **Inputs**: None
- **Control Flow**:
    - The function evaluates four conditions: whether `object_` is null, whether `array_` is null, whether `primitive_` is null, and whether `callable_` is null.
    - If all four conditions are true, the function returns true, indicating that the `Value` instance is null.
- **Output**: Returns a boolean value indicating whether the `Value` instance is null.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::is\_boolean<!-- {{#callable:minja::Value::is_boolean}} -->
Determines if the `primitive_` member of the `Value` class is of boolean type.
- **Inputs**: None
- **Control Flow**:
    - The function directly calls the `is_boolean` method of the `primitive_` member.
    - It returns the result of that method call.
- **Output**: Returns a boolean value indicating whether `primitive_` is a boolean type.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::is\_number\_integer<!-- {{#callable:minja::Value::is_number_integer}} -->
The `is_number_integer` method checks if the `primitive_` member of the `Value` class represents an integer.
- **Inputs**: None
- **Control Flow**:
    - The method directly calls the `is_number_integer` method on the `primitive_` member.
    - It returns the result of that method call.
- **Output**: The output is a boolean value indicating whether the `primitive_` is an integer.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::is\_number\_float<!-- {{#callable:minja::Value::is_number_float}} -->
Checks if the `primitive_` member of the `Value` class is a floating-point number.
- **Inputs**: None
- **Control Flow**:
    - The function directly calls the `is_number_float` method of the `primitive_` member.
    - It returns the result of that method call.
- **Output**: Returns a boolean value indicating whether `primitive_` is a floating-point number.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::is\_number<!-- {{#callable:minja::Value::is_number}} -->
The `is_number` function checks if the `primitive_` member of the `Value` class is of a numeric type.
- **Inputs**: None
- **Control Flow**:
    - The function directly calls the `is_number` method on the `primitive_` member.
    - It returns the result of that method call.
- **Output**: The output is a boolean value indicating whether the `primitive_` is a number.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::is\_string<!-- {{#callable:minja::Value::is_string}} -->
The `is_string` method checks if the `Value` instance holds a string type.
- **Inputs**: None
- **Control Flow**:
    - The method directly calls the `is_string` method on the `primitive_` member variable.
    - It returns the result of the `is_string` method, which indicates whether the `primitive_` is of string type.
- **Output**: The output is a boolean value indicating whether the `Value` instance represents a string.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::is\_iterable<!-- {{#callable:minja::Value::is_iterable}} -->
Determines if the `Value` instance is iterable.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the `Value` instance is an array, object, or string by calling the respective methods.
    - It returns true if any of the checks for array, object, or string are true, otherwise it returns false.
- **Output**: Returns a boolean indicating whether the `Value` instance is iterable.
- **Functions called**:
    - [`minja::Value::is_array`](#Valueis_array)
    - [`minja::Value::is_object`](#Valueis_object)
    - [`minja::Value::is_string`](#Valueis_string)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::is\_primitive<!-- {{#callable:minja::Value::is_primitive}} -->
Determines if the `Value` instance is a primitive type.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `Value` class.
- **Control Flow**:
    - The function checks if the `array_`, `object_`, and `callable_` members are all null.
    - If all three members are null, the function returns true, indicating that the instance is a primitive type.
    - If any of the members are not null, it returns false, indicating that the instance is not a primitive type.
- **Output**: Returns a boolean value: true if the instance is a primitive type, false otherwise.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::is\_hashable<!-- {{#callable:minja::Value::is_hashable}} -->
The `is_hashable` method checks if the `Value` instance is a primitive type.
- **Inputs**: None
- **Control Flow**:
    - The method calls `is_primitive()` to determine if the current instance is a primitive type.
- **Output**: Returns a boolean value indicating whether the `Value` instance is hashable (i.e., if it is a primitive type).
- **Functions called**:
    - [`minja::Value::is_primitive`](#Valueis_primitive)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::empty<!-- {{#callable:minja::Value::empty}} -->
Checks if the `Value` instance is empty.
- **Inputs**:
    - `this`: A constant reference to the current `Value` instance.
- **Control Flow**:
    - First, it checks if the `Value` instance is null by calling `is_null()`. If it is null, a runtime error is thrown.
    - Next, it checks if the `Value` instance is a string using `is_string()`, and if so, it returns the result of calling `empty()` on the string.
    - If the instance is an array (checked using `is_array()`), it returns the result of calling `empty()` on the array.
    - If the instance is an object (checked using `is_object()`), it returns the result of calling `empty()` on the object.
    - If none of the above conditions are met, it returns false, indicating that the `Value` is not empty.
- **Output**: Returns a boolean indicating whether the `Value` instance is empty (true) or not (false).
- **Functions called**:
    - [`minja::Value::is_null`](#Valueis_null)
    - [`minja::Value::is_string`](#Valueis_string)
    - [`minja::Value::is_array`](#Valueis_array)
    - [`minja::Value::is_object`](#Valueis_object)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::for\_each<!-- {{#callable:minja::Value::for_each}} -->
The `for_each` method iterates over the elements of a [`Value`](#ValueValue) object, applying a provided callback function to each element.
- **Inputs**:
    - `callback`: A callable function that takes a reference to a [`Value`](#ValueValue) object and performs an operation on it.
- **Control Flow**:
    - Checks if the [`Value`](#ValueValue) is null and throws a runtime error if it is.
    - If the [`Value`](#ValueValue) is an array, iterates through each item in the array and applies the callback function.
    - If the [`Value`](#ValueValue) is an object, iterates through each key-value pair, creating a [`Value`](#ValueValue) for the key and applying the callback function.
    - If the [`Value`](#ValueValue) is a string, iterates through each character, creating a [`Value`](#ValueValue) for each character and applying the callback function.
    - If the [`Value`](#ValueValue) is none of the above types, throws a runtime error indicating that the value is not iterable.
- **Output**: The method does not return a value; it executes the callback function for each element in the [`Value`](#ValueValue).
- **Functions called**:
    - [`minja::Value::is_null`](#Valueis_null)
    - [`minja::Value::is_string`](#Valueis_string)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::to\_bool<!-- {{#callable:minja::Value::to_bool}} -->
Converts various types of `Value` instances to their boolean representation.
- **Inputs**:
    - `this`: A constant reference to the current `Value` instance.
- **Control Flow**:
    - Checks if the `Value` instance is null; if so, returns false.
    - Checks if the `Value` instance is a boolean; if so, returns its boolean value.
    - Checks if the `Value` instance is a number; if so, returns true if the number is not zero.
    - Checks if the `Value` instance is a string; if so, returns true if the string is not empty.
    - Checks if the `Value` instance is an array; if so, returns true if the array is not empty.
    - If none of the above conditions are met, returns true.
- **Output**: Returns a boolean value representing the truthiness of the `Value` instance.
- **Functions called**:
    - [`minja::Value::is_null`](#Valueis_null)
    - [`minja::Value::is_boolean`](#Valueis_boolean)
    - [`minja::Value::is_number`](#Valueis_number)
    - [`minja::Value::is_string`](#Valueis_string)
    - [`minja::Value::is_array`](#Valueis_array)
    - [`minja::Value::empty`](#Valueempty)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::to\_int<!-- {{#callable:minja::Value::to_int}} -->
Converts various types of `Value` to an `int64_t` representation.
- **Inputs**:
    - `this`: A constant reference to the current `Value` object.
- **Control Flow**:
    - Checks if the `Value` is null; if so, returns 0.
    - Checks if the `Value` is a boolean; if so, returns 1 for true and 0 for false.
    - Checks if the `Value` is a number; if so, converts it to `int64_t`.
    - Checks if the `Value` is a string; attempts to convert it to `int64_t` using `std::stol`, returning 0 on failure.
    - If none of the above conditions are met, returns 0.
- **Output**: Returns an `int64_t` representation of the `Value`, or 0 if conversion is not possible.
- **Functions called**:
    - [`minja::Value::is_null`](#Valueis_null)
    - [`minja::Value::is_boolean`](#Valueis_boolean)
    - [`minja::Value::is_number`](#Valueis_number)
    - [`minja::Value::is_string`](#Valueis_string)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::operator<<!-- {{#callable:minja::Value::operator<}} -->
Compares two `Value` objects to determine if the current object is less than the other.
- **Inputs**:
    - `other`: A constant reference to another `Value` object to compare against.
- **Control Flow**:
    - Checks if the current object is null and throws a runtime error if it is.
    - Checks if both the current object and the `other` object are numbers, and if so, compares their double values.
    - Checks if both the current object and the `other` object are strings, and if so, compares their string values.
    - Throws a runtime error if the types of the two objects are incompatible for comparison.
- **Output**: Returns a boolean indicating whether the current `Value` is less than the `other` `Value`.
- **Functions called**:
    - [`minja::Value::is_null`](#Valueis_null)
    - [`minja::Value::is_number`](#Valueis_number)
    - [`minja::Value::is_string`](#Valueis_string)
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::operator>=<!-- {{#callable:minja::Value::operator>=}} -->
Compares two `Value` objects for greater than or equal relationship.
- **Inputs**:
    - `other`: A constant reference to another `Value` object to compare against.
- **Control Flow**:
    - The function calls the less-than operator (`<`) to determine if the current object is less than the `other` object.
    - If the current object is less than `other`, the function returns false; otherwise, it returns true.
- **Output**: Returns a boolean value indicating whether the current `Value` object is greater than or equal to the `other` object.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::operator><!-- {{#callable:minja::Value::operator>}} -->
Compares two `Value` objects and returns true if the current object is greater than the other.
- **Inputs**:
    - `other`: A constant reference to another `Value` object to compare against.
- **Control Flow**:
    - Checks if the current object is null and throws a runtime error if it is.
    - Checks if both objects are numbers and compares their double values if true.
    - Checks if both objects are strings and compares their string values if true.
    - Throws a runtime error if the types of the objects are incompatible for comparison.
- **Output**: Returns a boolean value indicating whether the current `Value` object is greater than the `other` `Value` object.
- **Functions called**:
    - [`minja::Value::is_null`](#Valueis_null)
    - [`minja::Value::is_number`](#Valueis_number)
    - [`minja::Value::is_string`](#Valueis_string)
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::operator<=<!-- {{#callable:minja::Value::operator<=}} -->
The `operator<=` function compares the current `Value` object with another `Value` object and returns true if the current object is less than or equal to the other.
- **Inputs**:
    - `other`: A constant reference to another `Value` object to compare against.
- **Control Flow**:
    - The function calls the `operator>` to check if the current object is greater than the `other` object.
    - It negates the result of the comparison to determine if the current object is less than or equal to the `other` object.
- **Output**: Returns a boolean value indicating whether the current `Value` object is less than or equal to the `other` object.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::operator==<!-- {{#callable:minja::Value::operator==}} -->
Compares two `Value` objects for equality based on their types and contents.
- **Inputs**:
    - `other`: Another `Value` object to compare against.
- **Control Flow**:
    - First, check if either `Value` is callable; if so, compare their callable pointers.
    - If both are arrays, check their sizes and compare each element for equality.
    - If both are objects, check their sizes and compare each key-value pair for equality.
    - If neither is an array or object, compare their primitive values directly.
- **Output**: Returns `true` if the two `Value` objects are equal, otherwise returns `false`.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::operator\!=<!-- {{#callable:minja::Value::operator!=}} -->
The `operator!=` function checks if the current `Value` object is not equal to another `Value` object.
- **Inputs**:
    - `other`: A constant reference to another `Value` object to compare against.
- **Control Flow**:
    - The function calls the equality operator `operator==` to check if the current object is equal to the `other` object.
    - The result of the equality check is negated using the logical NOT operator `!`.
- **Output**: Returns a boolean value indicating whether the current `Value` object is not equal to the `other` object.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::contains<!-- {{#callable:minja::Value::contains}} -->
Checks if a given key or value exists in the `Value` object.
- **Inputs**:
    - `key`: A `const char*` representing the key to check for existence.
    - `key`: A `const std::string&` representing the key to check for existence.
    - `value`: A `const Value&` representing the value to check for existence.
- **Control Flow**:
    - If the `Value` object is an array, the function iterates through its elements to check if any match the provided value.
    - If the `Value` object is an object, it checks if the provided key exists in the object's key-value pairs.
    - If the `Value` object is neither an array nor an object, an exception is thrown indicating that the operation is invalid.
- **Output**: Returns a boolean indicating whether the key or value exists in the `Value` object.
- **Functions called**:
    - [`minja::Value::contains`](#Valuecontains)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::contains<!-- {{#callable:minja::Value::contains}} -->
Checks if a given key or value exists in the `Value` object.
- **Inputs**:
    - `key`: A `std::string` representing the key to check for existence in the object.
    - `value`: A `Value` object to check for existence in the array.
- **Control Flow**:
    - The function first checks if the `Value` object is an array or an object.
    - If it is an array, it iterates through the elements to check if the specified `value` exists.
    - If it is an object, it checks if the specified `key` exists in the object's key set.
    - If neither an array nor an object is present, it throws a runtime error indicating the invalid call.
- **Output**: Returns a boolean indicating whether the specified key or value exists in the `Value` object.
- **Functions called**:
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::contains<!-- {{#callable:minja::Value::contains}} -->
Checks if a given `Value` is contained within the current `Value` object, which can be either an array or an object.
- **Inputs**:
    - `value`: A constant reference to a `Value` object that is being checked for containment.
- **Control Flow**:
    - First, the function checks if the current `Value` is null, throwing an exception if it is.
    - If the current `Value` is an array, it iterates through each item in the array, returning true if any item matches the input `value`.
    - If the current `Value` is an object, it checks if the input `value` is hashable and then checks if it exists as a key in the object.
    - If the current `Value` is neither an array nor an object, it throws an exception indicating that the operation is invalid.
- **Output**: Returns true if the `value` is found within the current `Value` object; otherwise, returns false.
- **Functions called**:
    - [`minja::Value::is_null`](#Valueis_null)
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::erase<!-- {{#callable:minja::Value::erase}} -->
The `erase` method removes an element from an array at a specified index.
- **Inputs**:
    - `index`: A `size_t` representing the index of the element to be removed from the array.
- **Control Flow**:
    - The method first checks if the `array_` member is null; if it is, a runtime error is thrown indicating that the value is not an array.
    - If the `array_` is valid, the method proceeds to erase the element at the specified `index` using the `erase` method of the underlying vector.
- **Output**: The method does not return a value; it modifies the internal state of the `Value` object by removing the specified element from the array.
- **Functions called**:
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::erase<!-- {{#callable:minja::Value::erase}} -->
The `erase` method removes a key from an object if it exists.
- **Inputs**:
    - `key`: A `std::string` representing the key to be removed from the object.
- **Control Flow**:
    - The method first checks if the `object_` member is null, indicating that the current `Value` instance is not an object.
    - If `object_` is null, it throws a `std::runtime_error` with a message that includes the result of the `dump()` method.
    - If `object_` is valid, it calls the `erase` method on the `object_` to remove the specified key.
- **Output**: The method does not return a value; it modifies the internal state of the object by removing the specified key.
- **Functions called**:
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::at<!-- {{#callable:minja::Value::at}} -->
The `at` method retrieves a reference to a `Value` object at a specified index or key, allowing both const and non-const access.
- **Inputs**:
    - `index`: A `Value` object representing the index or key to access the desired element from the `Value` object.
- **Control Flow**:
    - The method first checks if the `index` is hashable; if not, it throws an error.
    - If the `Value` object is an array, it retrieves the element at the specified index.
    - If the `Value` object is an object, it retrieves the value associated with the specified key.
    - If the `Value` object is neither an array nor an object, it throws an error.
- **Output**: Returns a reference to the `Value` object located at the specified index or key, allowing for both read and write access.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::at<!-- {{#callable:minja::Value::at}} -->
The `at` method retrieves a reference to a `Value` object from an array or object based on the provided index or key.
- **Inputs**:
    - `index`: A `Value` object representing the index or key used to access the desired element in the array or object.
- **Control Flow**:
    - The method first checks if the `index` is hashable; if not, it throws a runtime error.
    - If the `Value` is an array, it retrieves the element at the specified index using the `array_` pointer.
    - If the `Value` is an object, it retrieves the element associated with the specified key using the `object_` pointer.
    - If the `Value` is neither an array nor an object, it throws a runtime error.
- **Output**: Returns a reference to the `Value` object located at the specified index or key, or throws an error if the access is invalid.
- **Functions called**:
    - [`minja::Value::dump`](#Valuedump)
    - [`minja::Value::is_array`](#Valueis_array)
    - [`minja::Value::is_object`](#Valueis_object)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::at<!-- {{#callable:minja::Value::at}} -->
Returns a reference to the element at the specified index in the `Value` object, allowing both const and non-const access.
- **Inputs**:
    - `index`: The index of the element to access, specified as a `size_t`.
- **Control Flow**:
    - The function first attempts to cast the current object to a non-const version using `const_cast`.
    - It then calls the non-const version of the `at` method with the provided index.
- **Output**: Returns a reference to the `Value` object at the specified index.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::at<!-- {{#callable:minja::Value::at}} -->
The `at` function retrieves a reference to a `Value` at a specified index.
- **Inputs**:
    - `index`: A `size_t` representing the index of the element to retrieve from the `Value`.
- **Control Flow**:
    - Checks if the `Value` is null and throws a runtime error if it is.
    - Checks if the `Value` is an array and retrieves the element at the specified index if true.
    - Checks if the `Value` is an object and retrieves the element at the specified index if true.
    - Throws a runtime error if the `Value` is neither an array nor an object.
- **Output**: Returns a reference to the `Value` at the specified index if the `Value` is an array or object; otherwise, throws an error.
- **Functions called**:
    - [`minja::Value::is_null`](#Valueis_null)
    - [`minja::Value::is_array`](#Valueis_array)
    - [`minja::Value::is_object`](#Valueis_object)
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::get<!-- {{#callable:minja::Value::get}} -->
Retrieves the value associated with a specified key from an object, returning a default value if the key does not exist.
- **Inputs**:
    - `key`: A `std::string` representing the key whose associated value is to be retrieved.
    - `default_value`: A value of type `T` that will be returned if the specified key does not exist in the object.
- **Control Flow**:
    - The function first checks if the object contains the specified `key` using the [`contains`](#Valuecontains) method.
    - If the key is not found, it returns the provided `default_value`.
    - If the key is found, it retrieves the associated value using the [`at`](#Valueat) method and returns it after calling `get<T>()` to convert it to the desired type.
- **Output**: Returns the value associated with the specified `key` if it exists, otherwise returns the provided `default_value`.
- **Functions called**:
    - [`minja::Value::contains`](#Valuecontains)
    - [`minja::Value::at`](#Valueat)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::get<!-- {{#callable:minja::Value::get}} -->
Retrieves a value of type `T` from a `Value` object if it is a primitive type.
- **Inputs**:
    - `T`: The type to which the value should be converted.
- **Control Flow**:
    - Checks if the `Value` object is a primitive type using the `is_primitive()` method.
    - If it is a primitive type, it retrieves the value of type `T` using the `get<T>()` method of the `primitive_` member.
    - If it is not a primitive type, it throws a `std::runtime_error` with a message indicating that the operation is not defined for the current value type.
- **Output**: Returns the value of type `T` if the `Value` object is a primitive type; otherwise, throws an exception.
- **Functions called**:
    - [`minja::Value::is_primitive`](#Valueis_primitive)
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::dump<!-- {{#callable:minja::Value::dump}} -->
The [`dump`](#Valuedump) function generates a string representation of the `Value` object.
- **Inputs**:
    - `indent`: An integer specifying the number of spaces to use for indentation; defaults to -1, which means no indentation.
    - `to_json`: A boolean flag indicating whether to format the output as JSON; defaults to false.
- **Control Flow**:
    - Creates a `std::ostringstream` object named `out` to hold the output string.
    - Calls the private [`dump`](#Valuedump) method, passing the `out` stream, `indent`, a level counter initialized to 0, and the `to_json` flag.
    - Returns the string representation of the output stream by calling `out.str()`.
- **Output**: Returns a string that represents the `Value` object, formatted according to the specified indentation and JSON options.
- **Functions called**:
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::operator\-<!-- {{#callable:minja::Value::operator-}} -->
The `operator-` function negates the value of a `Value` object, returning the negative of its integer or double representation.
- **Inputs**:
    - `this`: A constant reference to the current `Value` object on which the negation is performed.
- **Control Flow**:
    - The function first checks if the current `Value` object represents an integer using the `is_number_integer()` method.
    - If it is an integer, it retrieves the integer value using `get<int64_t>()` and negates it.
    - If it is not an integer, it retrieves the double value using `get<double>()` and negates it.
- **Output**: The function returns a new `Value` object that represents the negated value, either as an integer or a double.
- **Functions called**:
    - [`minja::Value::is_number_integer`](#Valueis_number_integer)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::to\_str<!-- {{#callable:minja::Value::to_str}} -->
Converts a `Value` object to its string representation.
- **Inputs**: None
- **Control Flow**:
    - Checks if the `Value` is a string and returns it directly if true.
    - Checks if the `Value` is an integer and converts it to a string using `std::to_string`.
    - Checks if the `Value` is a float and converts it to a string using `std::to_string`.
    - Checks if the `Value` is a boolean and returns 'True' or 'False' based on its value.
    - Checks if the `Value` is null and returns 'None'.
    - If none of the above conditions are met, it calls the `dump()` method to get a string representation.
- **Output**: Returns a `std::string` that represents the `Value` object in a human-readable format.
- **Functions called**:
    - [`minja::Value::is_string`](#Valueis_string)
    - [`minja::Value::is_number_integer`](#Valueis_number_integer)
    - [`minja::Value::is_number_float`](#Valueis_number_float)
    - [`minja::Value::is_boolean`](#Valueis_boolean)
    - [`minja::Value::is_null`](#Valueis_null)
    - [`minja::Value::dump`](#Valuedump)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::operator\+<!-- {{#callable:minja::Value::operator+}} -->
Implements the addition operator for the `Value` class, allowing for the addition of different types of values.
- **Inputs**:
    - `rhs`: A constant reference to another `Value` object that is to be added to the current object.
- **Control Flow**:
    - Checks if either the current object or the `rhs` object is a string; if so, it converts both to strings and concatenates them.
    - If both objects are integers, it retrieves their integer values and adds them.
    - If both objects are arrays, it creates a new array and appends the elements of both arrays to it.
    - If none of the above conditions are met, it retrieves the double values of both objects and adds them.
- **Output**: Returns a new `Value` object that represents the result of the addition operation, which can be a string, integer, array, or double depending on the types of the operands.
- **Functions called**:
    - [`minja::Value::is_string`](#Valueis_string)
    - [`minja::Value::to_str`](#Valueto_str)
    - [`minja::Value::is_number_integer`](#Valueis_number_integer)
    - [`minja::Value::is_array`](#Valueis_array)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::operator\-<!-- {{#callable:minja::Value::operator-}} -->
The `operator-` function performs subtraction between two `Value` objects, handling both integer and floating-point types.
- **Inputs**:
    - `rhs`: A constant reference to another `Value` object that will be subtracted from the current object.
- **Control Flow**:
    - The function first checks if both the current object and the `rhs` object are integers using the `is_number_integer()` method.
    - If both are integers, it retrieves their integer values using the `get<int64_t>()` method and performs the subtraction.
    - If either of the objects is not an integer, it retrieves their double values using the `get<double>()` method and performs the subtraction.
- **Output**: The function returns a new `Value` object that represents the result of the subtraction.
- **Functions called**:
    - [`minja::Value::is_number_integer`](#Valueis_number_integer)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::operator\*<!-- {{#callable:minja::Value::operator*}} -->
Implements the multiplication operator for the `Value` class, allowing for different behaviors based on the types of the operands.
- **Inputs**:
    - `rhs`: A constant reference to another `Value` object that is the right-hand side operand in the multiplication operation.
- **Control Flow**:
    - Checks if the current object is a string and the right-hand side is an integer; if so, it repeats the string `n` times, where `n` is the integer value.
    - If both the current object and the right-hand side are integers, it performs integer multiplication.
    - If neither of the above conditions are met, it defaults to multiplying the double representations of both values.
- **Output**: Returns a new `Value` object that represents the result of the multiplication operation, which can be a string, integer, or double depending on the input types.
- **Functions called**:
    - [`minja::Value::is_string`](#Valueis_string)
    - [`minja::Value::to_str`](#Valueto_str)
    - [`minja::Value::is_number_integer`](#Valueis_number_integer)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::operator/<!-- {{#callable:minja::Value::operator/}} -->
Performs division between two `Value` objects, handling both integer and floating-point types.
- **Inputs**:
    - `rhs`: The right-hand side `Value` object to divide by.
- **Control Flow**:
    - Checks if both the current object and `rhs` are integers using `is_number_integer()`.
    - If both are integers, it retrieves their integer values using `get<int64_t>()` and performs integer division.
    - If either is not an integer, it retrieves their double values using `get<double>()` and performs floating-point division.
- **Output**: Returns a new `Value` object containing the result of the division.
- **Functions called**:
    - [`minja::Value::is_number_integer`](#Valueis_number_integer)
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)


---
#### Value::operator%<!-- {{#callable:minja::Value::operator%}} -->
The `operator%` function computes the modulus of the current `Value` object with another `Value` object.
- **Inputs**:
    - `rhs`: A constant reference to another `Value` object that represents the right-hand side operand for the modulus operation.
- **Control Flow**:
    - The function retrieves the integer value of the current `Value` object using `get<int64_t>()`.
    - It then retrieves the integer value of the `rhs` object using `rhs.get<int64_t>()`.
    - The modulus operation is performed using the `%` operator on the two integer values.
- **Output**: The result of the modulus operation, which is a new `Value` object containing the result of the operation.
- **See also**: [`minja::Value`](#minjaValue)  (Data Structure)



---
### ArgumentsValue<!-- {{#data_structure:minja::ArgumentsValue}} -->
- **Type**: `struct`
- **Members**:
    - `args`: A vector containing positional arguments.
    - `kwargs`: A vector of pairs containing named arguments and their corresponding values.
- **Description**: The `ArgumentsValue` struct is designed to hold both positional and named arguments, allowing for flexible argument passing in function calls. It contains a vector for positional arguments (`args`) and a vector of key-value pairs for named arguments (`kwargs`). The struct also provides methods to check for named arguments, retrieve their values, and validate the number of arguments provided.
- **Member Functions**:
    - [`minja::ArgumentsValue::has_named`](#ArgumentsValuehas_named)
    - [`minja::ArgumentsValue::get_named`](#ArgumentsValueget_named)
    - [`minja::ArgumentsValue::empty`](#ArgumentsValueempty)
    - [`minja::ArgumentsValue::expectArgs`](#ArgumentsValueexpectArgs)

**Methods**

---
#### ArgumentsValue::has\_named<!-- {{#callable:minja::ArgumentsValue::has_named}} -->
Checks if a named keyword argument exists in the `kwargs` vector.
- **Inputs**:
    - `name`: A constant reference to a `std::string` representing the name of the keyword argument to check.
- **Control Flow**:
    - Iterates over each pair in the `kwargs` vector.
    - Compares the first element of each pair (the key) with the provided `name`.
    - Returns true immediately if a match is found.
    - If no match is found after checking all pairs, returns false.
- **Output**: Returns a boolean value indicating whether the specified keyword argument exists in the `kwargs` vector.
- **See also**: [`minja::ArgumentsValue`](#minjaArgumentsValue)  (Data Structure)


---
#### ArgumentsValue::get\_named<!-- {{#callable:minja::ArgumentsValue::get_named}} -->
Retrieves a [`Value`](#ValueValue) associated with a given named key from the `kwargs` vector.
- **Inputs**:
    - `name`: A constant reference to a `std::string` representing the key to search for in the `kwargs`.
- **Control Flow**:
    - Iterates over each key-value pair in the `kwargs` vector.
    - Checks if the current key matches the provided `name`.
    - If a match is found, returns the corresponding [`Value`](#ValueValue).
    - If no match is found after the loop, returns a default-constructed [`Value`](#ValueValue).
- **Output**: Returns the [`Value`](#ValueValue) associated with the specified `name` if found; otherwise, returns a default-constructed [`Value`](#ValueValue).
- **Functions called**:
    - [`minja::Value::Value`](#ValueValue)
- **See also**: [`minja::ArgumentsValue`](#minjaArgumentsValue)  (Data Structure)


---
#### ArgumentsValue::empty<!-- {{#callable:minja::ArgumentsValue::empty}} -->
Checks if both positional and keyword arguments are empty.
- **Inputs**: None
- **Control Flow**:
    - The function evaluates the emptiness of the `args` vector.
    - It also checks the emptiness of the `kwargs` vector.
    - The function returns true only if both vectors are empty.
- **Output**: Returns a boolean value indicating whether both `args` and `kwargs` are empty.
- **See also**: [`minja::ArgumentsValue`](#minjaArgumentsValue)  (Data Structure)


---
#### ArgumentsValue::expectArgs<!-- {{#callable:minja::ArgumentsValue::expectArgs}} -->
Validates the number of positional and keyword arguments passed to a method against specified constraints.
- **Inputs**:
    - `method_name`: The name of the method being validated, used in the error message if validation fails.
    - `pos_count`: A pair representing the minimum and maximum number of positional arguments allowed.
    - `kw_count`: A pair representing the minimum and maximum number of keyword arguments allowed.
- **Control Flow**:
    - Checks if the number of positional arguments in `args` is less than the minimum or greater than the maximum specified in `pos_count`.
    - Checks if the number of keyword arguments in `kwargs` is less than the minimum or greater than the maximum specified in `kw_count`.
    - If any of the above checks fail, constructs an error message detailing the expected argument counts and throws a `std::runtime_error`.
- **Output**: The function does not return a value; it throws an exception if the argument validation fails.
- **See also**: [`minja::ArgumentsValue`](#minjaArgumentsValue)  (Data Structure)



---
### Context<!-- {{#data_structure:minja::Context}} -->
- **Type**: `class`
- **Members**:
    - `values_`: Stores the values associated with the context.
    - `parent_`: Holds a shared pointer to the parent context.
- **Description**: The `Context` class represents a scoped environment for variable storage, allowing for hierarchical contexts through parent-child relationships. It manages a collection of `Value` objects, which can be accessed and modified, and supports retrieval of values from parent contexts if they are not found in the current context.
- **Member Functions**:
    - [`minja::Context::Context`](#ContextContext)
    - [`minja::Context::~Context`](#ContextContext)
    - [`minja::Context::keys`](#Contextkeys)
    - [`minja::Context::get`](#Contextget)
    - [`minja::Context::at`](#Contextat)
    - [`minja::Context::contains`](#Contextcontains)
    - [`minja::Context::set`](#Contextset)
    - [`minja::Context::builtins`](#Contextbuiltins)
    - [`minja::Context::make`](#Contextmake)
- **Inherits From**:
    - `std::enable_shared_from_this<Context>`

**Methods**

---
#### Context::Context<!-- {{#callable:minja::Context::Context}} -->
Constructs a `Context` object with specified values and an optional parent context.
- **Inputs**:
    - `values`: An rvalue reference to a `Value` object that is expected to be an object.
    - `parent`: An optional shared pointer to a parent `Context`, defaulting to nullptr.
- **Control Flow**:
    - The constructor initializes the member variable `values_` by moving the `values` argument.
    - It also initializes the `parent_` member variable with the provided `parent` argument.
    - If the `values_` is not an object, a runtime error is thrown with a descriptive message.
- **Output**: The constructor does not return a value but initializes a new instance of the `Context` class.
- **See also**: [`minja::Context`](#minjaContext)  (Data Structure)


---
#### Context::\~Context<!-- {{#callable:minja::Context::~Context}} -->
The `~Context` destructor is a virtual destructor that allows for proper cleanup of derived classes.
- **Inputs**: None
- **Control Flow**:
    - The destructor does not contain any specific logic and simply allows for the cleanup of resources when an object of a derived class is destroyed.
- **Output**: The function does not return any value as it is a destructor.
- **See also**: [`minja::Context`](#minjaContext)  (Data Structure)


---
#### Context::keys<!-- {{#callable:minja::Context::keys}} -->
The `keys` method retrieves all the keys from the `values_` object.
- **Inputs**: None
- **Control Flow**:
    - The method directly calls the `keys` method on the `values_` object.
    - It does not contain any conditional statements or loops.
- **Output**: Returns a vector of `Value` objects representing the keys of the `values_` object.
- **See also**: [`minja::Context`](#minjaContext)  (Data Structure)


---
#### Context::get<!-- {{#callable:minja::Context::get}} -->
Retrieves a [`Value`](#ValueValue) associated with a given `key` from the current `Context`, checking parent contexts if necessary.
- **Inputs**:
    - `key`: A constant reference to a [`Value`](#ValueValue) object representing the key to look up in the current context.
- **Control Flow**:
    - Checks if the `key` exists in the `values_` member of the current `Context` instance.
    - If the `key` is found, the corresponding [`Value`](#ValueValue) is returned.
    - If the `key` is not found and a parent context exists, the method recursively calls `get` on the parent context.
    - If the `key` is not found in both the current and parent contexts, a default [`Value`](#ValueValue) (likely representing null or an empty state) is returned.
- **Output**: Returns a [`Value`](#ValueValue) object associated with the provided `key`, or a default [`Value`](#ValueValue) if the key is not found.
- **Functions called**:
    - [`minja::Value::Value`](#ValueValue)
- **See also**: [`minja::Context`](#minjaContext)  (Data Structure)


---
#### Context::at<!-- {{#callable:minja::Context::at}} -->
Retrieves a reference to a `Value` associated with a given `key` in the current `Context`, searching parent contexts if necessary.
- **Inputs**:
    - `key`: A constant reference to a `Value` that serves as the key to look up in the current context's `values_`.
- **Control Flow**:
    - Checks if the `key` exists in the current context's `values_` using the `contains` method.
    - If the `key` is found, it retrieves and returns the corresponding `Value` using the `at` method.
    - If the `key` is not found and a parent context exists, it recursively calls `at` on the parent context.
    - If the `key` is not found in both the current and parent contexts, it throws a `std::runtime_error` indicating that the variable is undefined.
- **Output**: Returns a reference to the `Value` associated with the specified `key` if found; otherwise, throws an exception.
- **See also**: [`minja::Context`](#minjaContext)  (Data Structure)


---
#### Context::contains<!-- {{#callable:minja::Context::contains}} -->
Checks if a given key exists in the current context or its parent.
- **Inputs**:
    - `key`: A `Value` object representing the key to check for existence.
- **Control Flow**:
    - First, it checks if the `key` exists in the `values_` member of the current `Context` instance.
    - If the `key` is found, it returns true.
    - If the `key` is not found and a `parent_` context exists, it recursively calls `contains` on the parent context.
    - If the `key` is not found in both the current and parent contexts, it returns false.
- **Output**: Returns a boolean indicating whether the `key` exists in the current context or any of its parent contexts.
- **See also**: [`minja::Context`](#minjaContext)  (Data Structure)


---
#### Context::set<!-- {{#callable:minja::Context::set}} -->
Sets a key-value pair in the `values_` object of the `Context` class.
- **Inputs**:
    - `key`: A `Value` object representing the key to be set.
    - `value`: A `Value` object representing the value to be associated with the key.
- **Control Flow**:
    - The method directly calls the `set` method of the `values_` object, passing the `key` and `value` arguments.
    - No additional control flow or error handling is implemented within this method.
- **Output**: This method does not return a value; it modifies the internal state of the `Context` object by setting the specified key-value pair.
- **See also**: [`minja::Context`](#minjaContext)  (Data Structure)


---
#### Context::builtins<!-- {{#callable:minja::Context::builtins}} -->
The `builtins` method initializes a shared `Context` object with a set of built-in functions.
- **Inputs**: None
- **Control Flow**:
    - Creates a [`Value`](#ValueValue) object to hold global functions.
    - Defines several built-in functions such as `raise_exception`, `tojson`, `items`, `last`, `trim`, and others using [`simple_function`](#minjasimple_function).
    - Each function is added to the `globals` object with a specific name.
    - The method returns a new `Context` object containing the `globals`.
- **Output**: Returns a shared pointer to a `Context` object that contains the defined built-in functions.
- **Functions called**:
    - [`minja::simple_function`](#minjasimple_function)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::strip`](#minjastrip)
    - [`minja::html_escape`](#minjahtml_escape)
- **See also**: [`minja::Context`](#minjaContext)  (Data Structure)


---
#### Context::make<!-- {{#callable:minja::Context::make}} -->
Creates a new `Context` instance with specified values and an optional parent context.
- **Inputs**:
    - `values`: An rvalue reference to a `Value` object that holds the context values, which can be null.
    - `parent`: A shared pointer to a `Context` object that serves as the parent context, defaulting to `builtins()` if not provided.
- **Control Flow**:
    - Checks if the `values` parameter is null.
    - If `values` is null, it initializes the context with an empty object using `Value::object()`.
    - If `values` is not null, it moves the `values` into the new `Context` instance.
    - The new `Context` instance is created with the specified `values` and `parent`.
- **Output**: Returns a shared pointer to the newly created `Context` instance.
- **See also**: [`minja::Context`](#minjaContext)  (Data Structure)



---
### Location<!-- {{#data_structure:minja::Location}} -->
- **Type**: `struct`
- **Members**:
    - `source`: A shared pointer to a string representing the source of the location.
    - `pos`: A size_t representing the position within the source.
- **Description**: The `Location` struct is designed to hold information about a specific position in a source string, including a shared pointer to the source string itself and the position index within that string.


---
### Expression<!-- {{#data_structure:minja::Expression}} -->
- **Type**: `class`
- **Members**:
    - `location`: Stores the location information of the expression.
- **Description**: The `Expression` class serves as an abstract base class for various types of expressions in a templating engine, providing a structure for evaluating expressions within a given context and handling errors during evaluation.
- **Member Functions**:
    - [`minja::Expression::Expression`](#ExpressionExpression)
    - [`minja::Expression::~Expression`](#ExpressionExpression)
    - [`minja::Expression::evaluate`](#Expressionevaluate)

**Methods**

---
#### Expression::Expression<!-- {{#callable:minja::Expression::Expression}} -->
Constructs an `Expression` object with a specified `Location`.
- **Inputs**:
    - `location`: A constant reference to a `Location` object that specifies the source location of the expression.
- **Control Flow**:
    - The constructor initializes the `location` member variable with the provided `Location` argument.
    - The destructor is declared as default, indicating that it will perform the default cleanup for the class.
- **Output**: This function does not return a value; it initializes an instance of the `Expression` class.
- **See also**: [`minja::Expression`](#minjaExpression)  (Data Structure)


---
#### Expression::\~Expression<!-- {{#callable:minja::Expression::~Expression}} -->
The `~Expression` destructor is a virtual destructor that allows derived classes to be properly cleaned up when an object is deleted through a base class pointer.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as `virtual`, ensuring that the correct destructor for derived classes is called when an object is deleted through a base class pointer.
    - The destructor has a default implementation, which means it does not perform any specific cleanup actions beyond what is automatically handled by the compiler.
- **Output**: The function does not return any value, as it is a destructor.
- **See also**: [`minja::Expression`](#minjaExpression)  (Data Structure)


---
#### Expression::evaluate<!-- {{#callable:minja::Expression::evaluate}} -->
Evaluates an expression in a given context, handling exceptions and providing error location details.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment in which the expression is evaluated.
- **Control Flow**:
    - The function attempts to call the [`do_evaluate`](#VariableExprdo_evaluate) method with the provided `context`.
    - If an exception is thrown during the evaluation, it catches the exception and constructs an error message.
    - If the `location.source` is available, it appends additional error location details to the message.
    - Finally, it throws a `std::runtime_error` with the constructed error message.
- **Output**: Returns a `Value` object that represents the result of the evaluated expression.
- **Functions called**:
    - [`minja::VariableExpr::do_evaluate`](#VariableExprdo_evaluate)
    - [`minja::error_location_suffix`](#minjaerror_location_suffix)
- **See also**: [`minja::Expression`](#minjaExpression)  (Data Structure)



---
### VariableExpr<!-- {{#data_structure:minja::VariableExpr}} -->
- **Type**: `class`
- **Members**:
    - `name`: Stores the name of the variable as a string.
- **Description**: The `VariableExpr` class represents an expression that evaluates to the value of a variable identified by its name, which is stored as a string. It inherits from the `Expression` class and provides functionality to evaluate the variable's value within a given context.
- **Member Functions**:
    - [`minja::VariableExpr::VariableExpr`](#VariableExprVariableExpr)
    - [`minja::VariableExpr::get_name`](#VariableExprget_name)
    - [`minja::VariableExpr::do_evaluate`](#VariableExprdo_evaluate)
- **Inherits From**:
    - [`minja::Expression::Expression`](#ExpressionExpression)

**Methods**

---
#### VariableExpr::VariableExpr<!-- {{#callable:minja::VariableExpr::VariableExpr}} -->
The `VariableExpr` class represents an expression that retrieves the value of a variable from a given context.
- **Inputs**:
    - `loc`: A `Location` object that indicates the source location of the variable expression.
    - `n`: A `std::string` representing the name of the variable.
- **Control Flow**:
    - The constructor initializes the base `Expression` class with the provided location and sets the variable name.
    - The `get_name` method returns the name of the variable stored in the `name` member variable.
- **Output**: The `get_name` method outputs the name of the variable as a `std::string`.
- **See also**: [`minja::VariableExpr`](#minjaVariableExpr)  (Data Structure)


---
#### VariableExpr::get\_name<!-- {{#callable:minja::VariableExpr::get_name}} -->
The `get_name` function retrieves the name of the variable represented by the `VariableExpr` class.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the private member variable `name` of the `VariableExpr` instance.
- **Output**: The output is a `std::string` representing the name of the variable.
- **See also**: [`minja::VariableExpr`](#minjaVariableExpr)  (Data Structure)


---
#### VariableExpr::do\_evaluate<!-- {{#callable:minja::VariableExpr::do_evaluate}} -->
Evaluates a variable expression by retrieving its value from the provided context.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that holds variable values.
- **Control Flow**:
    - Checks if the `context` contains the variable name.
    - If the variable name is not found, returns a default [`Value`](#ValueValue) object.
    - If found, retrieves and returns the value associated with the variable name from the context.
- **Output**: Returns a [`Value`](#ValueValue) object representing the variable's value from the context, or a default [`Value`](#ValueValue) if the variable is not found.
- **Functions called**:
    - [`minja::Value::Value`](#ValueValue)
- **See also**: [`minja::VariableExpr`](#minjaVariableExpr)  (Data Structure)



---
### SpaceHandling<!-- {{#data_structure:minja::SpaceHandling}} -->
- **Type**: `enum`
- **Members**:
    - `Keep`: Indicates that whitespace should be preserved.
    - `Strip`: Indicates that leading and trailing whitespace should be removed.
    - `StripSpaces`: Indicates that only spaces should be stripped.
    - `StripNewline`: Indicates that newline characters should be stripped.
- **Description**: The `SpaceHandling` enum defines constants that specify how whitespace should be handled in a template processing context, allowing for options to keep, strip, or selectively remove whitespace characters.


---
### TemplateToken<!-- {{#data_structure:minja::TemplateToken}} -->
- **Type**: `class`
- **Members**:
    - `type`: The type of the `TemplateToken`, indicating its category.
    - `location`: The `Location` object representing the position of the token in the source.
    - `pre_space`: Indicates how to handle whitespace before the token.
    - `post_space`: Indicates how to handle whitespace after the token.
- **Description**: The `TemplateToken` class serves as a base for various types of tokens used in template parsing, encapsulating the token's type, its location in the source, and whitespace handling options before and after the token.
- **Member Functions**:
    - [`minja::TemplateToken::typeToString`](#TemplateTokentypeToString)
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)
    - [`minja::TemplateToken::~TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### TemplateToken::typeToString<!-- {{#callable:minja::TemplateToken::typeToString}} -->
Converts a `Type` enum value to its corresponding string representation.
- **Inputs**:
    - `t`: An enum value of type `Type` representing a specific token type.
- **Control Flow**:
    - The function uses a `switch` statement to match the input `Type` against predefined cases.
    - For each case, it returns a corresponding string literal that represents the type.
    - If the input type does not match any case, it defaults to returning the string 'Unknown'.
- **Output**: Returns a `std::string` that represents the name of the token type corresponding to the input `Type`.
- **See also**: [`minja::TemplateToken`](#minjaTemplateToken)  (Data Structure)


---
#### TemplateToken::TemplateToken<!-- {{#callable:minja::TemplateToken::TemplateToken}} -->
Constructs a `TemplateToken` object with specified type, location, and space handling.
- **Inputs**:
    - `type`: An enumerated type indicating the kind of template token (e.g., Text, Expression, If, etc.).
    - `location`: A `Location` object that specifies the source and position of the token.
    - `pre`: A `SpaceHandling` enum value that determines how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value that determines how to handle whitespace after the token.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `location`, `pre_space`, and `post_space` with the provided arguments.
    - The constructor does not contain any conditional logic or loops, as it simply assigns values to the member variables.
- **Output**: The constructor does not return a value, but it initializes a `TemplateToken` object with the specified properties.
- **See also**: [`minja::TemplateToken`](#minjaTemplateToken)  (Data Structure)


---
#### TemplateToken::\~TemplateToken<!-- {{#callable:minja::TemplateToken::~TemplateToken}} -->
The `~TemplateToken` is a virtual destructor for the `TemplateToken` class, ensuring proper cleanup of derived classes.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual, allowing derived classes to override it.
    - The default implementation is called, which handles any necessary cleanup for the base class.
- **Output**: This function does not return a value; it ensures that resources are released when an object of a derived class is destroyed.
- **See also**: [`minja::TemplateToken`](#minjaTemplateToken)  (Data Structure)



---
### Type<!-- {{#data_structure:minja::TemplateToken::Type}} -->
- **Type**: `enum class`
- **Members**:
    - `Text`: Represents a text token.
    - `Expression`: Represents an expression token.
    - `If`: Represents the start of an if block.
    - `Else`: Represents an else block.
    - `Elif`: Represents an else-if block.
    - `EndIf`: Represents the end of an if block.
    - `For`: Represents the start of a for loop.
    - `EndFor`: Represents the end of a for loop.
    - `Generation`: Represents the start of a generation block.
    - `EndGeneration`: Represents the end of a generation block.
    - `Set`: Represents the start of a set block.
    - `EndSet`: Represents the end of a set block.
    - `Comment`: Represents a comment token.
    - `Macro`: Represents the start of a macro definition.
    - `EndMacro`: Represents the end of a macro definition.
    - `Filter`: Represents the start of a filter block.
    - `EndFilter`: Represents the end of a filter block.
    - `Break`: Represents a break statement.
    - `Continue`: Represents a continue statement.
- **Description**: The `Type` enum class defines various token types used in a templating engine, including control flow tokens (like `If`, `Else`, `For`), block delimiters (like `EndIf`, `EndFor`), and other constructs (like `Comment`, `Macro`, `Filter`). Each enumerator represents a specific type of token that can be encountered while parsing templates.


---
### TextTemplateToken<!-- {{#data_structure:minja::TextTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `text`: Holds the text content of the token.
- **Description**: The `TextTemplateToken` struct is a specialized type of `TemplateToken` that represents a text segment within a template, encapsulating the text content along with its associated location and space handling properties.
- **Member Functions**:
    - [`minja::TextTemplateToken::TextTemplateToken`](#TextTemplateTokenTextTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### TextTemplateToken::TextTemplateToken<!-- {{#callable:minja::TextTemplateToken::TextTemplateToken}} -->
Constructs a `TextTemplateToken` representing a text segment in a template.
- **Inputs**:
    - `loc`: A `Location` object indicating the position of the token in the source.
    - `pre`: A `SpaceHandling` enum value indicating how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value indicating how to handle whitespace after the token.
    - `t`: A `std::string` containing the text content of the token.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type set to `Text`, and passes the `loc`, `pre`, and `post` parameters.
    - The `text` member variable is initialized with the provided string `t`.
- **Output**: This function does not return a value; it initializes an instance of `TextTemplateToken`.
- **See also**: [`minja::TextTemplateToken`](#minjaTextTemplateToken)  (Data Structure)



---
### ExpressionTemplateToken<!-- {{#data_structure:minja::ExpressionTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `expr`: A shared pointer to an `Expression` object representing the expression associated with this token.
- **Description**: The `ExpressionTemplateToken` struct is a specialized type of `TemplateToken` that encapsulates an expression to be evaluated within a template context, storing the expression as a shared pointer for efficient memory management.
- **Member Functions**:
    - [`minja::ExpressionTemplateToken::ExpressionTemplateToken`](#ExpressionTemplateTokenExpressionTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### ExpressionTemplateToken::ExpressionTemplateToken<!-- {{#callable:minja::ExpressionTemplateToken::ExpressionTemplateToken}} -->
Constructs an `ExpressionTemplateToken` which represents an expression in a template.
- **Inputs**:
    - `loc`: A `Location` object that indicates the position of the token in the source template.
    - `pre`: A `SpaceHandling` enum value that specifies how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value that specifies how to handle whitespace after the token.
    - `e`: A rvalue reference to a `std::shared_ptr<Expression>` representing the expression associated with this token.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type set to `Expression`, along with the provided location and space handling parameters.
    - The expression `e` is moved into the member variable `expr`, which is a shared pointer to an `Expression`.
- **Output**: This constructor does not return a value but initializes an instance of `ExpressionTemplateToken`.
- **See also**: [`minja::ExpressionTemplateToken`](#minjaExpressionTemplateToken)  (Data Structure)



---
### IfTemplateToken<!-- {{#data_structure:minja::IfTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `condition`: A shared pointer to an `Expression` representing the condition for the if statement.
- **Description**: The `IfTemplateToken` struct inherits from `TemplateToken` and is used to represent an 'if' statement in a template. It contains a condition that is evaluated to determine whether the associated block of template code should be executed.
- **Member Functions**:
    - [`minja::IfTemplateToken::IfTemplateToken`](#IfTemplateTokenIfTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### IfTemplateToken::IfTemplateToken<!-- {{#callable:minja::IfTemplateToken::IfTemplateToken}} -->
Constructs an `IfTemplateToken` with a specified condition and location.
- **Inputs**:
    - `loc`: A `Location` object representing the position in the source code.
    - `pre`: A `SpaceHandling` enum value indicating how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value indicating how to handle whitespace after the token.
    - `c`: A rvalue reference to a `std::shared_ptr<Expression>` representing the condition for the if statement.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type set to `If`, along with the provided location and space handling parameters.
    - The `condition` member variable is initialized by moving the provided expression pointer.
- **Output**: This constructor does not return a value but initializes an instance of `IfTemplateToken`.
- **See also**: [`minja::IfTemplateToken`](#minjaIfTemplateToken)  (Data Structure)



---
### ElifTemplateToken<!-- {{#data_structure:minja::ElifTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `condition`: A shared pointer to an `Expression` representing the condition for the elif statement.
- **Description**: The `ElifTemplateToken` struct is a specialized type of `TemplateToken` that represents an 'elif' clause in a template, containing a condition that determines whether the associated block of code should be executed based on the evaluation of that condition.
- **Member Functions**:
    - [`minja::ElifTemplateToken::ElifTemplateToken`](#ElifTemplateTokenElifTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### ElifTemplateToken::ElifTemplateToken<!-- {{#callable:minja::ElifTemplateToken::ElifTemplateToken}} -->
Constructs an `ElifTemplateToken` with a specified location, space handling options, and a condition expression.
- **Inputs**:
    - `loc`: A `Location` object representing the position in the source code where this token is defined.
    - `pre`: A `SpaceHandling` enum value indicating how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value indicating how to handle whitespace after the token.
    - `c`: A `std::shared_ptr<Expression>` representing the condition that this `elif` token evaluates.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type `Elif`, location, and space handling options.
    - The condition expression is moved into the member variable `condition`.
- **Output**: This function does not return a value; it initializes an instance of `ElifTemplateToken`.
- **See also**: [`minja::ElifTemplateToken`](#minjaElifTemplateToken)  (Data Structure)



---
### ElseTemplateToken<!-- {{#data_structure:minja::ElseTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `location`: Indicates the location in the source code where this token is defined.
    - `pre_space`: Specifies how to handle whitespace before the token.
    - `post_space`: Specifies how to handle whitespace after the token.
- **Description**: The `ElseTemplateToken` is a specialized type of `TemplateToken` that represents an 'else' clause in a template, inheriting properties such as location and whitespace handling from its base class.
- **Member Functions**:
    - [`minja::ElseTemplateToken::ElseTemplateToken`](#ElseTemplateTokenElseTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### ElseTemplateToken::ElseTemplateToken<!-- {{#callable:minja::ElseTemplateToken::ElseTemplateToken}} -->
Constructs an `ElseTemplateToken` which represents an 'else' block in a template.
- **Inputs**:
    - `loc`: A `Location` object that indicates the position of the token in the source template.
    - `pre`: A `SpaceHandling` enum value that specifies how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value that specifies how to handle whitespace after the token.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type set to `Type::Else`, along with the provided location and space handling parameters.
    - No additional logic is executed within the constructor, as it directly forwards the parameters to the base class.
- **Output**: An instance of `ElseTemplateToken` that can be used to represent an 'else' block in a template processing context.
- **See also**: [`minja::ElseTemplateToken`](#minjaElseTemplateToken)  (Data Structure)



---
### EndIfTemplateToken<!-- {{#data_structure:minja::EndIfTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `location`: Indicates the location in the source code where this token is defined.
    - `pre_space`: Specifies how to handle whitespace before the token.
    - `post_space`: Specifies how to handle whitespace after the token.
- **Description**: The `EndIfTemplateToken` is a specialized type of `TemplateToken` that represents the end of an 'if' block in a templating language, encapsulating information about its location and whitespace handling.
- **Member Functions**:
    - [`minja::EndIfTemplateToken::EndIfTemplateToken`](#EndIfTemplateTokenEndIfTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### EndIfTemplateToken::EndIfTemplateToken<!-- {{#callable:minja::EndIfTemplateToken::EndIfTemplateToken}} -->
Constructs an `EndIfTemplateToken` which signifies the end of an if block in a template.
- **Inputs**:
    - `loc`: A `Location` object that indicates the position of the token in the source template.
    - `pre`: A `SpaceHandling` enum value that determines how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value that determines how to handle whitespace after the token.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type set to `Type::EndIf`, along with the provided location and space handling parameters.
    - No additional logic is present in the constructor, as it directly calls the base class constructor.
- **Output**: An instance of `EndIfTemplateToken` that encapsulates the end of an if block in a template.
- **See also**: [`minja::EndIfTemplateToken`](#minjaEndIfTemplateToken)  (Data Structure)



---
### MacroTemplateToken<!-- {{#data_structure:minja::MacroTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `name`: A shared pointer to a `VariableExpr` representing the name of the macro.
    - `params`: A collection of parameters for the macro, represented as `Expression::Parameters`.
- **Description**: The `MacroTemplateToken` struct represents a macro in a templating system, inheriting from `TemplateToken`. It contains a name, which is a shared pointer to a `VariableExpr`, and a set of parameters that define the inputs for the macro. This structure is used to facilitate the definition and invocation of macros within templates, allowing for dynamic content generation based on the provided parameters.
- **Member Functions**:
    - [`minja::MacroTemplateToken::MacroTemplateToken`](#MacroTemplateTokenMacroTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### MacroTemplateToken::MacroTemplateToken<!-- {{#callable:minja::MacroTemplateToken::MacroTemplateToken}} -->
Constructs a `MacroTemplateToken` representing a macro with a name and parameters.
- **Inputs**:
    - `loc`: A `Location` object indicating the position of the macro in the source.
    - `pre`: A `SpaceHandling` enum value indicating how to handle whitespace before the macro.
    - `post`: A `SpaceHandling` enum value indicating how to handle whitespace after the macro.
    - `n`: A `std::shared_ptr<VariableExpr>` representing the name of the macro.
    - `p`: An `Expression::Parameters` object containing the parameters for the macro.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type set to `Macro`, location, and space handling options.
    - The `name` member is initialized by moving the provided `std::shared_ptr<VariableExpr>`.
    - The `params` member is initialized by moving the provided `Expression::Parameters`.
- **Output**: This function does not return a value; it initializes a `MacroTemplateToken` object.
- **See also**: [`minja::MacroTemplateToken`](#minjaMacroTemplateToken)  (Data Structure)



---
### EndMacroTemplateToken<!-- {{#data_structure:minja::EndMacroTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `location`: Indicates the location in the source code where this token is defined.
    - `pre_space`: Specifies how to handle whitespace before this token.
    - `post_space`: Specifies how to handle whitespace after this token.
- **Description**: The `EndMacroTemplateToken` is a specialized type of `TemplateToken` that signifies the end of a macro definition in a templating system, inheriting properties related to its location and whitespace handling from its base class.
- **Member Functions**:
    - [`minja::EndMacroTemplateToken::EndMacroTemplateToken`](#EndMacroTemplateTokenEndMacroTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### EndMacroTemplateToken::EndMacroTemplateToken<!-- {{#callable:minja::EndMacroTemplateToken::EndMacroTemplateToken}} -->
Constructs an `EndMacroTemplateToken` representing the end of a macro in a template.
- **Inputs**:
    - `loc`: A `Location` object indicating the position in the source code where this token is defined.
    - `pre`: A `SpaceHandling` enum value indicating how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value indicating how to handle whitespace after the token.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type set to `Type::EndMacro`.
    - It passes the `loc`, `pre`, and `post` parameters to the `TemplateToken` constructor.
- **Output**: This function does not return a value; it constructs an instance of `EndMacroTemplateToken`.
- **See also**: [`minja::EndMacroTemplateToken`](#minjaEndMacroTemplateToken)  (Data Structure)



---
### FilterTemplateToken<!-- {{#data_structure:minja::FilterTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `filter`: A shared pointer to an `Expression` representing the filter expression.
- **Description**: The `FilterTemplateToken` struct is a specialized type of `TemplateToken` that encapsulates a filter expression, allowing for the application of filters in template processing. It inherits from `TemplateToken` and includes a member that holds a reference to an `Expression`, which defines the filtering logic to be applied.
- **Member Functions**:
    - [`minja::FilterTemplateToken::FilterTemplateToken`](#FilterTemplateTokenFilterTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### FilterTemplateToken::FilterTemplateToken<!-- {{#callable:minja::FilterTemplateToken::FilterTemplateToken}} -->
The `FilterTemplateToken` constructor initializes a filter template token with a specified location, space handling options, and a filter expression.
- **Inputs**:
    - `loc`: A `Location` object representing the position of the token in the source template.
    - `pre`: A `SpaceHandling` enum value indicating how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value indicating how to handle whitespace after the token.
    - `filter`: A `std::shared_ptr<Expression>` representing the filter expression associated with this token.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type `Filter`, location, and space handling options.
    - It then moves the provided `filter` expression into the member variable `filter`.
- **Output**: The constructor does not return a value; it initializes an instance of `FilterTemplateToken`.
- **See also**: [`minja::FilterTemplateToken`](#minjaFilterTemplateToken)  (Data Structure)



---
### EndFilterTemplateToken<!-- {{#data_structure:minja::EndFilterTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `location`: Indicates the location in the source code.
    - `pre_space`: Specifies how to handle whitespace before the token.
    - `post_space`: Specifies how to handle whitespace after the token.
- **Description**: The `EndFilterTemplateToken` struct represents a token that marks the end of a filter block in a template, inheriting from `TemplateToken` and carrying location and whitespace handling information.
- **Member Functions**:
    - [`minja::EndFilterTemplateToken::EndFilterTemplateToken`](#EndFilterTemplateTokenEndFilterTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### EndFilterTemplateToken::EndFilterTemplateToken<!-- {{#callable:minja::EndFilterTemplateToken::EndFilterTemplateToken}} -->
Constructs an `EndFilterTemplateToken` which signifies the end of a filter block in a template.
- **Inputs**:
    - `loc`: A `Location` object that indicates the position of the token in the source template.
    - `pre`: A `SpaceHandling` enum value that specifies how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value that specifies how to handle whitespace after the token.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type set to `Type::EndFilter`.
    - It passes the `loc`, `pre`, and `post` parameters to the `TemplateToken` constructor.
- **Output**: This function does not return a value; it constructs an instance of `EndFilterTemplateToken`.
- **See also**: [`minja::EndFilterTemplateToken`](#minjaEndFilterTemplateToken)  (Data Structure)



---
### ForTemplateToken<!-- {{#data_structure:minja::ForTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `var_names`: A vector of variable names used in the for loop.
    - `iterable`: A shared pointer to an `Expression` representing the iterable object.
    - `condition`: A shared pointer to an `Expression` representing the loop condition.
    - `recursive`: A boolean indicating if the loop is recursive.
- **Description**: The `ForTemplateToken` struct represents a for loop in a template, containing variable names, an iterable expression, an optional condition expression, and a flag indicating whether the loop is recursive.
- **Member Functions**:
    - [`minja::ForTemplateToken::ForTemplateToken`](#ForTemplateTokenForTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### ForTemplateToken::ForTemplateToken<!-- {{#callable:minja::ForTemplateToken::ForTemplateToken}} -->
Constructs a `ForTemplateToken` object representing a for-loop in a template.
- **Inputs**:
    - `loc`: A `Location` object indicating the position of the token in the source.
    - `pre`: A `SpaceHandling` enum value indicating how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value indicating how to handle whitespace after the token.
    - `vns`: A vector of strings representing variable names to be used in the loop.
    - `iter`: A `shared_ptr` to an `Expression` representing the iterable object.
    - `c`: A `shared_ptr` to an `Expression` representing the condition for the loop.
    - `r`: A boolean indicating whether the loop is recursive.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type set to `For` and the provided location and space handling parameters.
    - It initializes the member variables `var_names`, `iterable`, `condition`, and `recursive` with the corresponding input parameters.
- **Output**: No output is returned; the function constructs an instance of `ForTemplateToken`.
- **See also**: [`minja::ForTemplateToken`](#minjaForTemplateToken)  (Data Structure)



---
### EndForTemplateToken<!-- {{#data_structure:minja::EndForTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `location`: Indicates the location in the source code where this token is defined.
    - `pre_space`: Specifies how to handle whitespace before the token.
    - `post_space`: Specifies how to handle whitespace after the token.
- **Description**: The `EndForTemplateToken` is a specialized type of `TemplateToken` that signifies the end of a 'for' loop in a template, encapsulating information about its location and whitespace handling.
- **Member Functions**:
    - [`minja::EndForTemplateToken::EndForTemplateToken`](#EndForTemplateTokenEndForTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### EndForTemplateToken::EndForTemplateToken<!-- {{#callable:minja::EndForTemplateToken::EndForTemplateToken}} -->
Constructs an `EndForTemplateToken` representing the end of a for loop in a template.
- **Inputs**:
    - `loc`: A `Location` object indicating the position in the source code where this token is defined.
    - `pre`: A `SpaceHandling` enum value indicating how to handle whitespace before this token.
    - `post`: A `SpaceHandling` enum value indicating how to handle whitespace after this token.
- **Control Flow**:
    - The constructor of `EndForTemplateToken` initializes its base class `TemplateToken` with the type set to `Type::EndFor`.
    - It passes the `loc`, `pre`, and `post` parameters to the `TemplateToken` constructor.
- **Output**: This function does not return a value; it initializes an instance of `EndForTemplateToken`.
- **See also**: [`minja::EndForTemplateToken`](#minjaEndForTemplateToken)  (Data Structure)



---
### GenerationTemplateToken<!-- {{#data_structure:minja::GenerationTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `location`: Indicates the location in the source code.
    - `pre_space`: Specifies how to handle whitespace before the token.
    - `post_space`: Specifies how to handle whitespace after the token.
- **Description**: The `GenerationTemplateToken` is a specialized type of `TemplateToken` that represents a generation block in a template, encapsulating its location and whitespace handling preferences.
- **Member Functions**:
    - [`minja::GenerationTemplateToken::GenerationTemplateToken`](#GenerationTemplateTokenGenerationTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### GenerationTemplateToken::GenerationTemplateToken<!-- {{#callable:minja::GenerationTemplateToken::GenerationTemplateToken}} -->
Constructs a `GenerationTemplateToken` which is a type of `TemplateToken` used for generation in templates.
- **Inputs**:
    - `loc`: A constant reference to a `Location` object that indicates the position of the token in the source.
    - `pre`: A `SpaceHandling` enum value that specifies how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value that specifies how to handle whitespace after the token.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type set to `Type::Generation`, passing along the `loc`, `pre`, and `post` parameters.
- **Output**: This function does not return a value; it initializes an instance of `GenerationTemplateToken`.
- **See also**: [`minja::GenerationTemplateToken`](#minjaGenerationTemplateToken)  (Data Structure)



---
### EndGenerationTemplateToken<!-- {{#data_structure:minja::EndGenerationTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `location`: Indicates the location in the source code.
    - `pre_space`: Specifies how to handle whitespace before the token.
    - `post_space`: Specifies how to handle whitespace after the token.
- **Description**: The `EndGenerationTemplateToken` is a specialized type of `TemplateToken` that signifies the end of a generation block in a template, inheriting properties such as location and space handling from its base class.
- **Member Functions**:
    - [`minja::EndGenerationTemplateToken::EndGenerationTemplateToken`](#EndGenerationTemplateTokenEndGenerationTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### EndGenerationTemplateToken::EndGenerationTemplateToken<!-- {{#callable:minja::EndGenerationTemplateToken::EndGenerationTemplateToken}} -->
Constructs an `EndGenerationTemplateToken` which signifies the end of a generation block in a template.
- **Inputs**:
    - `loc`: A `Location` object that indicates the position in the source code where this token is defined.
    - `pre`: A `SpaceHandling` enum value that specifies how to handle whitespace before this token.
    - `post`: A `SpaceHandling` enum value that specifies how to handle whitespace after this token.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type `EndGeneration`, passing along the location and space handling parameters.
    - No additional logic is present in the constructor, as it directly calls the base class constructor.
- **Output**: This function does not return a value; it constructs an instance of `EndGenerationTemplateToken`.
- **See also**: [`minja::EndGenerationTemplateToken`](#minjaEndGenerationTemplateToken)  (Data Structure)



---
### SetTemplateToken<!-- {{#data_structure:minja::SetTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `ns`: A string representing the namespace.
    - `var_names`: A vector of strings containing variable names.
    - `value`: A shared pointer to an `Expression` representing the value to be set.
- **Description**: The `SetTemplateToken` struct is a specialized type of `TemplateToken` that encapsulates a namespace, a list of variable names, and an expression value, allowing for the setting of variables within a specific context in a templating system.
- **Member Functions**:
    - [`minja::SetTemplateToken::SetTemplateToken`](#SetTemplateTokenSetTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### SetTemplateToken::SetTemplateToken<!-- {{#callable:minja::SetTemplateToken::SetTemplateToken}} -->
Constructs a `SetTemplateToken` object to represent a template token that sets variables.
- **Inputs**:
    - `loc`: A `Location` object representing the position in the source code where this token is defined.
    - `pre`: A `SpaceHandling` enum value indicating how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value indicating how to handle whitespace after the token.
    - `ns`: A string representing the namespace for the variable being set.
    - `vns`: A vector of strings representing the variable names to be set.
    - `v`: A `shared_ptr` to an `Expression` that represents the value to be assigned to the variable.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type `Set`, location, and space handling options.
    - It then initializes the member variables `ns`, `var_names`, and `value` with the provided arguments.
- **Output**: This function does not return a value; it initializes an instance of `SetTemplateToken`.
- **See also**: [`minja::SetTemplateToken`](#minjaSetTemplateToken)  (Data Structure)



---
### EndSetTemplateToken<!-- {{#data_structure:minja::EndSetTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `location`: Indicates the location in the source code.
    - `pre_space`: Specifies how to handle whitespace before the token.
    - `post_space`: Specifies how to handle whitespace after the token.
- **Description**: The `EndSetTemplateToken` struct represents a token that marks the end of a 'set' block in a template, inheriting from `TemplateToken` and carrying location and whitespace handling information.
- **Member Functions**:
    - [`minja::EndSetTemplateToken::EndSetTemplateToken`](#EndSetTemplateTokenEndSetTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### EndSetTemplateToken::EndSetTemplateToken<!-- {{#callable:minja::EndSetTemplateToken::EndSetTemplateToken}} -->
Constructs an `EndSetTemplateToken` which signifies the end of a 'set' block in a template.
- **Inputs**:
    - `loc`: A `Location` object that indicates the position in the source code where this token is defined.
    - `pre`: A `SpaceHandling` enum value that specifies how to handle whitespace before this token.
    - `post`: A `SpaceHandling` enum value that specifies how to handle whitespace after this token.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type `Type::EndSet`, passing the location and space handling parameters.
    - No additional logic is executed in this constructor; it simply sets up the token.
- **Output**: This function does not return a value; it constructs an instance of `EndSetTemplateToken`.
- **See also**: [`minja::EndSetTemplateToken`](#minjaEndSetTemplateToken)  (Data Structure)



---
### CommentTemplateToken<!-- {{#data_structure:minja::CommentTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `text`: A string that holds the comment text.
- **Description**: The `CommentTemplateToken` struct is a specialized type of `TemplateToken` that represents a comment in a template, containing a `text` field to store the comment's content.
- **Member Functions**:
    - [`minja::CommentTemplateToken::CommentTemplateToken`](#CommentTemplateTokenCommentTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### CommentTemplateToken::CommentTemplateToken<!-- {{#callable:minja::CommentTemplateToken::CommentTemplateToken}} -->
Constructs a `CommentTemplateToken` with specified location, space handling options, and text.
- **Inputs**:
    - `loc`: A `Location` object representing the position of the token in the source.
    - `pre`: A `SpaceHandling` enum value indicating how to handle whitespace before the token.
    - `post`: A `SpaceHandling` enum value indicating how to handle whitespace after the token.
    - `t`: A `std::string` containing the text of the comment.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type set to `Comment`, along with the provided location and space handling options.
    - The `text` member variable is initialized with the provided comment text.
- **Output**: This function does not return a value; it initializes an instance of `CommentTemplateToken`.
- **See also**: [`minja::CommentTemplateToken`](#minjaCommentTemplateToken)  (Data Structure)



---
### LoopControlType<!-- {{#data_structure:minja::LoopControlType}} -->
- **Type**: `enum class`
- **Members**:
    - `Break`: Represents a control statement to exit a loop.
    - `Continue`: Represents a control statement to skip the current iteration of a loop.
- **Description**: The `LoopControlType` enum class defines two constants, `Break` and `Continue`, which are used to control the flow of loops in programming, allowing for the termination of a loop or skipping to the next iteration, respectively.


---
### LoopControlException<!-- {{#data_structure:minja::LoopControlException}} -->
- **Type**: `class`
- **Members**:
    - `control_type`: Indicates the type of loop control (break or continue).
- **Description**: The `LoopControlException` class is a custom exception derived from `std::runtime_error`, designed to handle errors related to loop control statements such as 'break' and 'continue' in a template rendering context. It includes a member `control_type` that specifies whether the exception is for a break or continue operation, and it provides constructors to initialize the exception with a message and the control type.
- **Member Functions**:
    - [`minja::LoopControlException::LoopControlException`](#LoopControlExceptionLoopControlException)
    - [`minja::LoopControlException::LoopControlException`](#LoopControlExceptionLoopControlException)
- **Inherits From**:
    - `std::runtime_error`

**Methods**

---
#### LoopControlException::LoopControlException<!-- {{#callable:minja::LoopControlException::LoopControlException}} -->
The `LoopControlException` class is a custom exception that indicates control flow issues related to loop constructs.
- **Inputs**:
    - `message`: A string that describes the error message associated with the exception.
    - `control_type`: An enumeration value of type `LoopControlType` that indicates whether the control flow is a 'break' or 'continue'.
- **Control Flow**:
    - The constructor initializes the base class `std::runtime_error` with the provided message.
    - The `control_type` member variable is set to the provided `control_type` argument.
- **Output**: The constructor does not return a value but initializes an instance of `LoopControlException` with a specific error message and control type.
- **See also**: [`minja::LoopControlException`](#minjaLoopControlException)  (Data Structure)


---
#### LoopControlException::LoopControlException<!-- {{#callable:minja::LoopControlException::LoopControlException}} -->
Constructs a `LoopControlException` with a message based on the specified `LoopControlType`.
- **Inputs**:
    - `control_type`: An enumeration value of type `LoopControlType` that indicates whether the exception is for a 'continue' or 'break' control flow.
- **Control Flow**:
    - The constructor checks the value of `control_type`.
    - If `control_type` is `LoopControlType::Continue`, it sets the error message to 'continue outside of a loop'.
    - If `control_type` is `LoopControlType::Break`, it sets the error message to 'break outside of a loop'.
    - The constructed message is passed to the base class `std::runtime_error` constructor.
- **Output**: An instance of `LoopControlException` initialized with a specific error message indicating the type of loop control violation.
- **See also**: [`minja::LoopControlException`](#minjaLoopControlException)  (Data Structure)



---
### LoopControlTemplateToken<!-- {{#data_structure:minja::LoopControlTemplateToken}} -->
- **Type**: `struct`
- **Members**:
    - `control_type`: Specifies the type of loop control, either 'Break' or 'Continue'.
- **Description**: The `LoopControlTemplateToken` struct is a specialized type of `TemplateToken` that represents control statements within loops, specifically for breaking or continuing the loop execution based on the `control_type`.
- **Member Functions**:
    - [`minja::LoopControlTemplateToken::LoopControlTemplateToken`](#LoopControlTemplateTokenLoopControlTemplateToken)
- **Inherits From**:
    - [`minja::TemplateToken::TemplateToken`](#TemplateTokenTemplateToken)

**Methods**

---
#### LoopControlTemplateToken::LoopControlTemplateToken<!-- {{#callable:minja::LoopControlTemplateToken::LoopControlTemplateToken}} -->
Constructs a `LoopControlTemplateToken` representing a loop control statement (break or continue) in a template.
- **Inputs**:
    - `loc`: A `Location` object that indicates the position in the source code where this token is defined.
    - `pre`: A `SpaceHandling` enum value that specifies how to handle whitespace before this token.
    - `post`: A `SpaceHandling` enum value that specifies how to handle whitespace after this token.
    - `control_type`: A `LoopControlType` enum value that indicates whether this token represents a break or continue statement.
- **Control Flow**:
    - The constructor initializes the base class `TemplateToken` with the type set to `Type::Break`, the provided location, and the specified space handling options.
    - The `control_type` member variable is initialized with the provided `control_type` argument.
- **Output**: This function does not return a value; it initializes an instance of `LoopControlTemplateToken`.
- **See also**: [`minja::LoopControlTemplateToken`](#minjaLoopControlTemplateToken)  (Data Structure)



---
### TemplateNode<!-- {{#data_structure:minja::TemplateNode}} -->
- **Type**: `class`
- **Members**:
    - `location_`: Stores the location information of the template node.
- **Description**: The `TemplateNode` class serves as an abstract base class for all template nodes in a templating system, providing a structure for rendering templates based on their location and defining a pure virtual function `do_render` that must be implemented by derived classes to specify how the node should be rendered.
- **Member Functions**:
    - [`minja::TemplateNode::TemplateNode`](#TemplateNodeTemplateNode)
    - [`minja::TemplateNode::render`](#TemplateNoderender)
    - [`minja::TemplateNode::location`](#TemplateNodelocation)
    - [`minja::TemplateNode::~TemplateNode`](#TemplateNodeTemplateNode)
    - [`minja::TemplateNode::render`](#TemplateNoderender)

**Methods**

---
#### TemplateNode::TemplateNode<!-- {{#callable:minja::TemplateNode::TemplateNode}} -->
The `render` method of the `TemplateNode` class generates output by invoking a rendering function and handles exceptions that may arise during the rendering process.
- **Inputs**:
    - `out`: An `std::ostringstream` object where the rendered output will be written.
    - `context`: A shared pointer to a `Context` object that provides the necessary context for rendering.
- **Control Flow**:
    - The method begins by attempting to call the `do_render` method, which is a pure virtual function that must be implemented by derived classes to perform the actual rendering.
    - If a `LoopControlException` is caught, it constructs an error message that includes the location of the error and rethrows the exception.
    - If any other `std::exception` is caught, it similarly constructs an error message and rethrows it as a `std::runtime_error`.
- **Output**: The method does not return a value; instead, it writes the rendered output to the provided `std::ostringstream` object.
- **See also**: [`minja::TemplateNode`](#minjaTemplateNode)  (Data Structure)


---
#### TemplateNode::render<!-- {{#callable:minja::TemplateNode::render}} -->
The `render` method of the `TemplateNode` class generates output by invoking the [`do_render`](#SequenceNodedo_render) method and handles exceptions that may arise during rendering.
- **Inputs**:
    - `out`: An `std::ostringstream` reference where the rendered output will be written.
    - `context`: A shared pointer to a `Context` object that provides the necessary context for rendering.
- **Control Flow**:
    - The method begins by attempting to call the [`do_render`](#SequenceNodedo_render) method, passing the `out` and `context` parameters.
    - If a [`LoopControlException`](#LoopControlExceptionLoopControlException) is caught, it constructs an error message that includes the exception's message and the location of the error, then rethrows the exception.
    - If any other `std::exception` is caught, it similarly constructs an error message and rethrows it as a `std::runtime_error`.
- **Output**: The method does not return a value; instead, it writes the rendered output to the `out` stream and may throw exceptions if rendering fails.
- **Functions called**:
    - [`minja::SequenceNode::do_render`](#SequenceNodedo_render)
    - [`minja::error_location_suffix`](#minjaerror_location_suffix)
    - [`minja::LoopControlException::LoopControlException`](#LoopControlExceptionLoopControlException)
- **See also**: [`minja::TemplateNode`](#minjaTemplateNode)  (Data Structure)


---
#### TemplateNode::location<!-- {{#callable:minja::TemplateNode::location}} -->
Returns a constant reference to the `Location` object associated with the `TemplateNode`.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the member variable `location_` which is of type `Location`.
- **Output**: A constant reference to a `Location` object, which contains information about the source and position of the template node.
- **See also**: [`minja::TemplateNode`](#minjaTemplateNode)  (Data Structure)


---
#### TemplateNode::\~TemplateNode<!-- {{#callable:minja::TemplateNode::~TemplateNode}} -->
The `render` method of the `TemplateNode` class generates a string representation of the template node by rendering its content into a provided output stream.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the necessary data and environment for rendering the template.
- **Control Flow**:
    - The method initializes an output string stream (`std::ostringstream out`).
    - It calls the `render` method that takes an output stream and the context, which is responsible for the actual rendering logic.
    - If the rendering process throws a `LoopControlException`, it catches the exception, constructs an error message that includes the location of the error, and rethrows the exception.
    - If any other exception is caught, it constructs a generic error message and rethrows it.
- **Output**: The method returns a string containing the rendered output of the template node.
- **See also**: [`minja::TemplateNode`](#minjaTemplateNode)  (Data Structure)


---
#### TemplateNode::render<!-- {{#callable:minja::TemplateNode::render}} -->
The [`render`](#TemplateNoderender) function generates a string representation of the template by rendering it into a string stream.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the necessary data for rendering the template.
- **Control Flow**:
    - Creates an output string stream (`out`) to hold the rendered content.
    - Calls the overloaded [`render`](#TemplateNoderender) method with the output stream and context.
    - Returns the string representation of the output stream.
- **Output**: Returns a `std::string` containing the rendered template output.
- **Functions called**:
    - [`minja::TemplateNode::render`](#TemplateNoderender)
- **See also**: [`minja::TemplateNode`](#minjaTemplateNode)  (Data Structure)



---
### SequenceNode<!-- {{#data_structure:minja::SequenceNode}} -->
- **Type**: `class`
- **Members**:
    - `children`: A vector of shared pointers to `TemplateNode` objects representing the child nodes.
- **Description**: The `SequenceNode` class is a derived class of `TemplateNode` that represents a sequence of child template nodes, allowing for the rendering of multiple child nodes in a specified order.
- **Member Functions**:
    - [`minja::SequenceNode::SequenceNode`](#SequenceNodeSequenceNode)
    - [`minja::SequenceNode::do_render`](#SequenceNodedo_render)
- **Inherits From**:
    - [`minja::TemplateNode::TemplateNode`](#TemplateNodeTemplateNode)

**Methods**

---
#### SequenceNode::SequenceNode<!-- {{#callable:minja::SequenceNode::SequenceNode}} -->
The `SequenceNode` class represents a sequence of child `TemplateNode` objects and renders them in order.
- **Inputs**:
    - `loc`: A `Location` object that indicates the source location of the node.
    - `c`: A vector of `shared_ptr<TemplateNode>` representing the child nodes to be rendered.
- **Control Flow**:
    - The constructor initializes the base class `TemplateNode` with the provided location and moves the child nodes into the `children` member.
    - The `do_render` method iterates over each child node in the `children` vector and calls their `render` method, passing the output stream and context.
- **Output**: The `do_render` method produces output by rendering each child node in sequence to the provided output stream.
- **See also**: [`minja::SequenceNode`](#minjaSequenceNode)  (Data Structure)


---
#### SequenceNode::do\_render<!-- {{#callable:minja::SequenceNode::do_render}} -->
The `do_render` method of the `SequenceNode` class renders all child nodes to the provided output stream.
- **Inputs**:
    - `out`: An output stream of type `std::ostringstream` where the rendered output will be written.
    - `context`: A shared pointer to a `Context` object that provides the necessary context for rendering.
- **Control Flow**:
    - Iterates over each child node in the `children` vector.
    - Calls the `render` method of each child node, passing the output stream and context as arguments.
- **Output**: This method does not return a value; it directly writes the rendered output to the provided output stream.
- **See also**: [`minja::SequenceNode`](#minjaSequenceNode)  (Data Structure)



---
### TextNode<!-- {{#data_structure:minja::TextNode}} -->
- **Type**: `class`
- **Members**:
    - `text`: Holds the text content of the `TextNode`.
- **Description**: The `TextNode` class is a derived class of `TemplateNode` that represents a node containing plain text in a template. It has a single member, `text`, which stores the text to be rendered. The `do_render` method outputs the text to the provided output stream.
- **Member Functions**:
    - [`minja::TextNode::TextNode`](#TextNodeTextNode)
    - [`minja::TextNode::do_render`](#TextNodedo_render)
- **Inherits From**:
    - [`minja::TemplateNode::TemplateNode`](#TemplateNodeTemplateNode)

**Methods**

---
#### TextNode::TextNode<!-- {{#callable:minja::TextNode::TextNode}} -->
The `TextNode` class represents a node that renders a static text string.
- **Inputs**:
    - `loc`: A `Location` object that indicates the position of the node in the source template.
    - `t`: A `std::string` containing the text to be rendered by the node.
- **Control Flow**:
    - The constructor initializes the `TextNode` by calling the base class `TemplateNode` constructor with the provided `Location`.
    - The `do_render` method appends the `text` member to the provided output stream `out`.
- **Output**: The `do_render` method does not return a value; it writes the text to the output stream.
- **See also**: [`minja::TextNode`](#minjaTextNode)  (Data Structure)


---
#### TextNode::do\_render<!-- {{#callable:minja::TextNode::do_render}} -->
Renders the text of a `TextNode` to an output stream.
- **Inputs**:
    - `out`: An output stream (`std::ostringstream`) where the text will be rendered.
    - `context`: A shared pointer to a `Context` object, which is not used in this method.
- **Control Flow**:
    - The method directly writes the `text` member variable to the output stream `out`.
    - No conditional logic or loops are present in this method.
- **Output**: The method does not return a value; it modifies the output stream directly.
- **See also**: [`minja::TextNode`](#minjaTextNode)  (Data Structure)



---
### ExpressionNode<!-- {{#data_structure:minja::ExpressionNode}} -->
- **Type**: `class`
- **Members**:
    - `expr`: A shared pointer to an `Expression` object that represents the expression to be evaluated.
- **Description**: The `ExpressionNode` class is a derived class of `TemplateNode` that encapsulates an expression and provides functionality to render the result of evaluating that expression within a given context.
- **Member Functions**:
    - [`minja::ExpressionNode::ExpressionNode`](#ExpressionNodeExpressionNode)
    - [`minja::ExpressionNode::do_render`](#ExpressionNodedo_render)
- **Inherits From**:
    - [`minja::TemplateNode::TemplateNode`](#TemplateNodeTemplateNode)

**Methods**

---
#### ExpressionNode::ExpressionNode<!-- {{#callable:minja::ExpressionNode::ExpressionNode}} -->
The `do_render` method of the `ExpressionNode` class renders the result of evaluating an expression into an output stream.
- **Inputs**:
    - `out`: An output stream (`std::ostringstream`) where the rendered result will be written.
    - `context`: A shared pointer to a `Context` object that provides the necessary context for evaluating the expression.
- **Control Flow**:
    - The method first checks if the `expr` member is null, throwing a runtime error if it is.
    - It then evaluates the expression using the provided `context`, storing the result.
    - Depending on the type of the result (string, boolean, or other), it formats the output accordingly and writes it to the `out` stream.
- **Output**: The method does not return a value; instead, it writes the rendered output directly to the provided output stream.
- **See also**: [`minja::ExpressionNode`](#minjaExpressionNode)  (Data Structure)


---
#### ExpressionNode::do\_render<!-- {{#callable:minja::ExpressionNode::do_render}} -->
The `do_render` method evaluates an expression and writes the result to an output stream.
- **Inputs**:
    - `out`: An output stream (`std::ostringstream`) where the result of the expression evaluation will be written.
    - `context`: A shared pointer to a `Context` object that provides the environment for evaluating the expression.
- **Control Flow**:
    - Checks if the `expr` member is null and throws a runtime error if it is.
    - Evaluates the expression using the provided `context`.
    - Checks the type of the evaluation result: if it's a string, it writes it directly to the output; if it's a boolean, it writes 'True' or 'False'; if it's neither and not null, it dumps the result as JSON.
- **Output**: The method does not return a value; instead, it writes the evaluated result to the provided output stream.
- **See also**: [`minja::ExpressionNode`](#minjaExpressionNode)  (Data Structure)



---
### IfNode<!-- {{#data_structure:minja::IfNode}} -->
- **Type**: `class`
- **Members**:
    - `cascade`: A vector of pairs, each containing a shared pointer to an `Expression` and a shared pointer to a `TemplateNode`.
- **Description**: The `IfNode` class is a derived class of `TemplateNode` that represents a conditional structure in a template, allowing for multiple branches of execution based on evaluated expressions. It contains a `cascade` member that holds pairs of conditions and corresponding template nodes, enabling the rendering of the appropriate template based on the evaluation of the conditions.
- **Member Functions**:
    - [`minja::IfNode::IfNode`](#IfNodeIfNode)
    - [`minja::IfNode::do_render`](#IfNodedo_render)
- **Inherits From**:
    - [`minja::TemplateNode::TemplateNode`](#TemplateNodeTemplateNode)

**Methods**

---
#### IfNode::IfNode<!-- {{#callable:minja::IfNode::IfNode}} -->
The `do_render` method of the `IfNode` class evaluates a series of conditional expressions and renders the corresponding template node based on the first true condition.
- **Inputs**:
    - `out`: An output stream where the rendered content will be written.
    - `context`: A shared pointer to a `Context` object that provides the necessary context for evaluating expressions.
- **Control Flow**:
    - Iterates over each branch in the `cascade` vector, which contains pairs of expressions and template nodes.
    - For each branch, it evaluates the expression using the provided `context`.
    - If the expression evaluates to true, it checks if the corresponding template node is not null and then calls its `render` method to output the result.
    - If no expressions evaluate to true, the method completes without rendering any output.
- **Output**: The output is the rendered content of the first template node whose corresponding expression evaluates to true, written to the provided output stream.
- **See also**: [`minja::IfNode`](#minjaIfNode)  (Data Structure)


---
#### IfNode::do\_render<!-- {{#callable:minja::IfNode::do_render}} -->
The `do_render` method evaluates conditions in a cascade and renders the corresponding template if the condition is true.
- **Inputs**:
    - `out`: A reference to a `std::ostringstream` object where the rendered output will be written.
    - `context`: A shared pointer to a `Context` object that provides the necessary data for evaluating conditions.
- **Control Flow**:
    - Iterates over each branch in the `cascade` vector, which contains pairs of expressions and template nodes.
    - For each branch, it checks if the first element (an expression) is not null and evaluates it using the provided `context`.
    - If the evaluation returns true, it checks if the second element (the template node) is not null; if it is null, an exception is thrown.
    - If the second element is valid, it calls its `render` method, passing the `out` and `context`, and then exits the function.
- **Output**: The function does not return a value; it writes the rendered output directly to the `out` stream.
- **See also**: [`minja::IfNode`](#minjaIfNode)  (Data Structure)



---
### LoopControlNode<!-- {{#data_structure:minja::LoopControlNode}} -->
- **Type**: `class`
- **Members**:
    - `control_type_`: Specifies the type of loop control, either break or continue.
- **Description**: The `LoopControlNode` class is a specialized type of `TemplateNode` that represents control flow statements in a template, specifically for managing loop control with types such as break and continue. It throws a `LoopControlException` when rendered, indicating the type of control action to be taken.
- **Member Functions**:
    - [`minja::LoopControlNode::LoopControlNode`](#LoopControlNodeLoopControlNode)
    - [`minja::LoopControlNode::do_render`](#LoopControlNodedo_render)
- **Inherits From**:
    - [`minja::TemplateNode::TemplateNode`](#TemplateNodeTemplateNode)

**Methods**

---
#### LoopControlNode::LoopControlNode<!-- {{#callable:minja::LoopControlNode::LoopControlNode}} -->
The `LoopControlNode` class represents a control structure for handling loop control statements such as break and continue.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes the `LoopControlNode` with a specified `Location` and `LoopControlType`.
    - The `do_render` method throws a `LoopControlException` with the specified control type, indicating a break or continue action.
- **Output**: The output is an exception of type `LoopControlException`, which indicates the type of loop control action (break or continue) that occurred.
- **See also**: [`minja::LoopControlNode`](#minjaLoopControlNode)  (Data Structure)


---
#### LoopControlNode::do\_render<!-- {{#callable:minja::LoopControlNode::do_render}} -->
The `do_render` method in the `LoopControlNode` class throws a [`LoopControlException`](#LoopControlExceptionLoopControlException) with the specified control type.
- **Inputs**:
    - `out`: An `std::ostringstream` reference used for outputting rendered content.
    - `context`: A shared pointer to a `Context` object that provides the execution context.
- **Control Flow**:
    - The method immediately throws a [`LoopControlException`](#LoopControlExceptionLoopControlException) without performing any operations.
    - The exception is constructed using the `control_type_` member variable, which indicates the type of loop control (break or continue).
- **Output**: The method does not return a value; it throws an exception instead.
- **Functions called**:
    - [`minja::LoopControlException::LoopControlException`](#LoopControlExceptionLoopControlException)
- **See also**: [`minja::LoopControlNode`](#minjaLoopControlNode)  (Data Structure)



---
### ForNode<!-- {{#data_structure:minja::ForNode}} -->
- **Type**: `class`
- **Members**:
    - `var_names`: A vector of variable names used in the for loop.
    - `iterable`: A shared pointer to an `Expression` representing the iterable object.
    - `condition`: A shared pointer to an `Expression` representing the loop condition.
    - `body`: A shared pointer to a `TemplateNode` representing the body of the loop.
    - `recursive`: A boolean indicating if the loop is recursive.
    - `else_body`: A shared pointer to a `TemplateNode` representing the body to execute if the loop is empty.
- **Description**: The `ForNode` class represents a for loop structure in a templating engine, allowing iteration over a specified iterable with optional conditions and recursive behavior. It contains fields for variable names, the iterable expression, a condition expression, the loop body, and an optional else body to handle cases when the iterable is empty.
- **Member Functions**:
    - [`minja::ForNode::ForNode`](#ForNodeForNode)
    - [`minja::ForNode::do_render`](#ForNodedo_render)
- **Inherits From**:
    - [`minja::TemplateNode::TemplateNode`](#TemplateNodeTemplateNode)

**Methods**

---
#### ForNode::ForNode<!-- {{#callable:minja::ForNode::ForNode}} -->
The `ForNode` constructor initializes a for-loop template node with variable names, an iterable, an optional condition, a body, and an optional else body.
- **Inputs**:
    - `loc`: A `Location` object representing the source location of the node.
    - `var_names`: A vector of strings representing the names of the variables to be used in the loop.
    - `iterable`: A shared pointer to an `Expression` that evaluates to the iterable object.
    - `condition`: A shared pointer to an `Expression` that evaluates to a boolean condition for the loop.
    - `body`: A shared pointer to a `TemplateNode` representing the body of the loop.
    - `recursive`: A boolean indicating whether the loop should be recursive.
    - `else_body`: A shared pointer to a `TemplateNode` representing the body to execute if the iterable is empty.
- **Control Flow**:
    - The constructor initializes the base class `TemplateNode` with the provided location.
    - It stores the variable names, iterable, condition, body, recursive flag, and else body as member variables.
    - The constructor does not contain any control flow logic as it is primarily for initialization.
- **Output**: The constructor does not return a value; it initializes an instance of `ForNode`.
- **See also**: [`minja::ForNode`](#minjaForNode)  (Data Structure)


---
#### ForNode::do\_render<!-- {{#callable:minja::ForNode::do_render}} -->
Renders a for loop in a template context.
- **Inputs**:
    - `out`: An output stream where the rendered result will be written.
    - `context`: A shared pointer to the context containing variables and functions available during rendering.
- **Control Flow**:
    - Checks if the `iterable` and `body` are not null, throwing an error if either is null.
    - Evaluates the `iterable` expression to get the iterable value.
    - Defines a recursive function `visit` to process each item in the iterable.
    - Filters items based on the provided `condition` and stores them in `filtered_items`.
    - If `filtered_items` is empty, renders the `else_body` if it exists.
    - If `filtered_items` is not empty, sets up a loop context and iterates over the items, rendering the `body` for each item.
    - Handles loop control exceptions to allow for breaking or continuing the loop.
- **Output**: The function does not return a value; it writes the rendered output directly to the provided output stream.
- **Functions called**:
    - [`minja::destructuring_assign`](#minjadestructuring_assign)
    - [`minja::Value::Value`](#ValueValue)
- **See also**: [`minja::ForNode`](#minjaForNode)  (Data Structure)



---
### MacroNode<!-- {{#data_structure:minja::MacroNode}} -->
- **Type**: `class`
- **Members**:
    - `name`: A shared pointer to a `VariableExpr` representing the name of the macro.
    - `params`: A collection of parameters for the macro, represented as a vector of pairs.
    - `body`: A shared pointer to a `TemplateNode` representing the body of the macro.
    - `named_param_positions`: A map that stores the positions of named parameters for quick access.
- **Description**: The `MacroNode` class represents a macro in a templating system, encapsulating its name, parameters, and body. It inherits from `TemplateNode` and is responsible for rendering the macro with the provided context and arguments. The class maintains a mapping of named parameter positions to facilitate the handling of both positional and keyword arguments during macro invocation.
- **Member Functions**:
    - [`minja::MacroNode::MacroNode`](#MacroNodeMacroNode)
    - [`minja::MacroNode::do_render`](#MacroNodedo_render)
- **Inherits From**:
    - [`minja::TemplateNode::TemplateNode`](#TemplateNodeTemplateNode)

**Methods**

---
#### MacroNode::MacroNode<!-- {{#callable:minja::MacroNode::MacroNode}} -->
Constructs a `MacroNode` that represents a macro with a name, parameters, and a body.
- **Inputs**:
    - `loc`: A `Location` object representing the source location of the macro.
    - `n`: A `std::shared_ptr<VariableExpr>` representing the name of the macro.
    - `p`: An `Expression::Parameters` object containing the parameters for the macro.
    - `b`: A `std::shared_ptr<TemplateNode>` representing the body of the macro.
- **Control Flow**:
    - The constructor initializes the base class `TemplateNode` with the provided location.
    - It moves the name, parameters, and body into the member variables.
    - A loop iterates over the parameters to populate the `named_param_positions` map with the names and their respective indices.
- **Output**: The constructor does not return a value but initializes a `MacroNode` object.
- **See also**: [`minja::MacroNode`](#minjaMacroNode)  (Data Structure)


---
#### MacroNode::do\_render<!-- {{#callable:minja::MacroNode::do_render}} -->
The `do_render` method in the `MacroNode` class renders a macro by setting up its context and handling positional and keyword arguments.
- **Inputs**:
    - `out`: An output stream (`std::ostringstream`) where the rendered result will be written.
    - `macro_context`: A shared pointer to a `Context` object that provides the context for rendering the macro.
- **Control Flow**:
    - Checks if the `name` and `body` members of the `MacroNode` are not null, throwing an exception if either is null.
    - Creates a callable function that takes a context and arguments, which sets the parameters in the context based on the provided arguments.
    - Iterates over positional arguments, ensuring the correct number of arguments is provided and setting them in the context.
    - Processes keyword arguments, checking for unknown parameter names and setting them in the context.
    - Sets default values for any parameters that were not provided in the arguments.
    - Calls the `render` method of the `body` with the updated context and returns the result.
- **Output**: The output is the result of rendering the `body` of the macro with the provided context, which is returned as a `Value`.
- **See also**: [`minja::MacroNode`](#minjaMacroNode)  (Data Structure)



---
### FilterNode<!-- {{#data_structure:minja::FilterNode}} -->
- **Type**: `class`
- **Members**:
    - `filter`: A shared pointer to an `Expression` that represents the filter to be applied.
    - `body`: A shared pointer to a `TemplateNode` that represents the body of the template to be rendered.
- **Description**: The `FilterNode` class is a specialized type of `TemplateNode` that applies a filter to its body content during rendering. It holds a filter expression and a body template node, and when rendered, it evaluates the filter and applies it to the rendered output of the body.
- **Member Functions**:
    - [`minja::FilterNode::FilterNode`](#FilterNodeFilterNode)
    - [`minja::FilterNode::do_render`](#FilterNodedo_render)
- **Inherits From**:
    - [`minja::TemplateNode::TemplateNode`](#TemplateNodeTemplateNode)

**Methods**

---
#### FilterNode::FilterNode<!-- {{#callable:minja::FilterNode::FilterNode}} -->
Constructs a `FilterNode` that applies a filter to a body of template content.
- **Inputs**:
    - `loc`: A `Location` object representing the source location of the node.
    - `f`: A `std::shared_ptr<Expression>` representing the filter expression to be applied.
    - `b`: A `std::shared_ptr<TemplateNode>` representing the body of the template to which the filter will be applied.
- **Control Flow**:
    - The constructor initializes the base class `TemplateNode` with the provided location.
    - It moves the filter and body expressions into the member variables `filter` and `body`.
- **Output**: This constructor does not return a value but initializes a `FilterNode` object.
- **See also**: [`minja::FilterNode`](#minjaFilterNode)  (Data Structure)


---
#### FilterNode::do\_render<!-- {{#callable:minja::FilterNode::do_render}} -->
The `do_render` method in the `FilterNode` class applies a filter to the rendered output of a template body.
- **Inputs**:
    - `out`: An `std::ostringstream` reference where the rendered output will be written.
    - `context`: A shared pointer to a `Context` object that provides the necessary context for evaluating the filter and rendering the body.
- **Control Flow**:
    - Checks if the `filter` member is null and throws a runtime error if it is.
    - Checks if the `body` member is null and throws a runtime error if it is.
    - Evaluates the `filter` expression using the provided `context`.
    - Checks if the evaluated `filter_value` is callable; if not, throws a runtime error.
    - Renders the `body` using the provided `context`.
    - Creates an `ArgumentsValue` object with the rendered body as an argument for the filter.
    - Calls the `filter_value` with the context and the arguments, and writes the result to the `out` stream.
- **Output**: The output is the result of the filter applied to the rendered body, which is written to the `out` stream.
- **Functions called**:
    - [`minja::Value::Value`](#ValueValue)
- **See also**: [`minja::FilterNode`](#minjaFilterNode)  (Data Structure)



---
### SetNode<!-- {{#data_structure:minja::SetNode}} -->
- **Type**: `class`
- **Members**:
    - `ns`: A string representing the namespace.
    - `var_names`: A vector of strings containing variable names.
    - `value`: A shared pointer to an `Expression` representing the value to be set.
- **Description**: The `SetNode` class is a specialized type of `TemplateNode` that is responsible for setting variables in a specified namespace or directly in the context. It contains a namespace identifier, a list of variable names, and an expression that evaluates to the value to be assigned. The `do_render` method implements the logic for assigning the evaluated value to the specified variable(s) in the context, handling both namespaced and non-namespaced assignments.
- **Member Functions**:
    - [`minja::SetNode::SetNode`](#SetNodeSetNode)
    - [`minja::SetNode::do_render`](#SetNodedo_render)
- **Inherits From**:
    - [`minja::TemplateNode::TemplateNode`](#TemplateNodeTemplateNode)

**Methods**

---
#### SetNode::SetNode<!-- {{#callable:minja::SetNode::SetNode}} -->
The `SetNode` class method `do_render` assigns a value to a variable or a property in a specified namespace.
- **Inputs**:
    - `out`: An output stream where the rendered result will be written.
    - `context`: A shared pointer to a `Context` object that provides access to variables and namespaces.
- **Control Flow**:
    - The method first checks if the `value` is null, throwing an exception if it is.
    - If the `ns` (namespace) is not empty, it checks that there is exactly one variable name in `var_names`.
    - It retrieves the namespace value from the context and checks if it is an object.
    - The method then evaluates the `value` expression and sets the evaluated value to the specified variable in the namespace.
    - If the `ns` is empty, it evaluates the `value` and performs destructuring assignment to the variables in `var_names`.
- **Output**: The method does not return a value; it modifies the context by setting variables based on the evaluated expression.
- **See also**: [`minja::SetNode`](#minjaSetNode)  (Data Structure)


---
#### SetNode::do\_render<!-- {{#callable:minja::SetNode::do_render}} -->
The `do_render` method in the `SetNode` class evaluates an expression and assigns its result to a variable in a specified namespace or directly in the context.
- **Inputs**:
    - `out`: An output stream (`std::ostringstream`) where the rendered result can be written.
    - `context`: A shared pointer to a `Context` object that provides access to variables and their values.
- **Control Flow**:
    - Checks if the `value` member is null and throws a runtime error if it is.
    - If the `ns` (namespace) member is not empty, it checks that there is exactly one variable name in `var_names` and retrieves the corresponding value from the context.
    - If the retrieved namespace value is not an object, it throws a runtime error.
    - Sets the evaluated value of `this->value` into the namespace under the specified variable name.
    - If `ns` is empty, it evaluates `value` and performs destructuring assignment to the variables in `var_names`.
- **Output**: The method does not return a value; it modifies the context by setting a variable or variables based on the evaluated expression.
- **Functions called**:
    - [`minja::destructuring_assign`](#minjadestructuring_assign)
- **See also**: [`minja::SetNode`](#minjaSetNode)  (Data Structure)



---
### SetTemplateNode<!-- {{#data_structure:minja::SetTemplateNode}} -->
- **Type**: `class`
- **Members**:
    - `name`: A string representing the name associated with the `SetTemplateNode`.
    - `template_value`: A shared pointer to a `TemplateNode` that holds the value to be set.
- **Description**: The `SetTemplateNode` class is a specialized type of `TemplateNode` that is responsible for setting a value in a given context. It takes a name and a template value, and during rendering, it evaluates the template value and assigns it to the specified name in the context.
- **Member Functions**:
    - [`minja::SetTemplateNode::SetTemplateNode`](#SetTemplateNodeSetTemplateNode)
    - [`minja::SetTemplateNode::do_render`](#SetTemplateNodedo_render)
- **Inherits From**:
    - [`minja::TemplateNode::TemplateNode`](#TemplateNodeTemplateNode)

**Methods**

---
#### SetTemplateNode::SetTemplateNode<!-- {{#callable:minja::SetTemplateNode::SetTemplateNode}} -->
The `SetTemplateNode` class sets a variable in the context with the rendered value of a template.
- **Inputs**:
    - `loc`: A `Location` object representing the source location of the template node.
    - `name`: A `std::string` representing the name of the variable to be set in the context.
    - `tv`: A `std::shared_ptr<TemplateNode>` pointing to the template node that will be rendered to obtain the value.
- **Control Flow**:
    - The constructor initializes the `SetTemplateNode` with the provided location, variable name, and template value.
    - The `do_render` method checks if `template_value` is null and throws an exception if it is.
    - It then renders the `template_value` using the provided context to obtain a `Value` object.
    - Finally, it sets the rendered value in the context under the specified variable name.
- **Output**: The output is the setting of a variable in the context with the rendered value of the template node.
- **See also**: [`minja::SetTemplateNode`](#minjaSetTemplateNode)  (Data Structure)


---
#### SetTemplateNode::do\_render<!-- {{#callable:minja::SetTemplateNode::do_render}} -->
The `do_render` method sets a value in the provided `context` using a template value.
- **Inputs**:
    - `out`: An output stream of type `std::ostringstream` where the rendered output can be written.
    - `context`: A shared pointer to a `Context` object that holds the variables and their values.
- **Control Flow**:
    - The method first checks if `template_value` is null, throwing a runtime error if it is.
    - It then calls the `render` method on `template_value`, passing the `context`, to obtain a `Value` object.
    - Finally, it sets the obtained `Value` in the `context` using the `name` member variable.
- **Output**: The method does not return a value; it modifies the `context` by setting a new value.
- **See also**: [`minja::SetTemplateNode`](#minjaSetTemplateNode)  (Data Structure)



---
### IfExpr<!-- {{#data_structure:minja::IfExpr}} -->
- **Type**: `class`
- **Members**:
    - `condition`: A shared pointer to an `Expression` representing the condition of the if statement.
    - `then_expr`: A shared pointer to an `Expression` representing the expression to evaluate if the condition is true.
    - `else_expr`: A shared pointer to an `Expression` representing the expression to evaluate if the condition is false.
- **Description**: The `IfExpr` class represents a conditional expression in a template engine, encapsulating a condition and two possible expressions (then and else) to evaluate based on the truthiness of the condition.
- **Member Functions**:
    - [`minja::IfExpr::IfExpr`](#IfExprIfExpr)
    - [`minja::IfExpr::do_evaluate`](#IfExprdo_evaluate)
- **Inherits From**:
    - [`minja::Expression::Expression`](#ExpressionExpression)

**Methods**

---
#### IfExpr::IfExpr<!-- {{#callable:minja::IfExpr::IfExpr}} -->
The `IfExpr` class represents a conditional expression that evaluates a condition and returns one of two expressions based on the result.
- **Inputs**:
    - `loc`: A `Location` object that indicates the source location of the expression.
    - `c`: A `std::shared_ptr<Expression>` representing the condition to evaluate.
    - `t`: A `std::shared_ptr<Expression>` representing the expression to evaluate if the condition is true.
    - `e`: A `std::shared_ptr<Expression>` representing the expression to evaluate if the condition is false.
- **Control Flow**:
    - The method first checks if the `condition`, `then_expr`, or `else_expr` are null, throwing an exception if any are.
    - It evaluates the `condition` expression using the provided `context`.
    - If the result of the `condition` is true, it evaluates and returns the result of `then_expr`.
    - If the `condition` is false and `else_expr` is not null, it evaluates and returns the result of `else_expr`.
    - If both `then_expr` and `else_expr` are null, it returns a null value.
- **Output**: Returns a `Value` that is the result of evaluating either `then_expr` or `else_expr`, depending on the evaluation of the `condition`.
- **See also**: [`minja::IfExpr`](#minjaIfExpr)  (Data Structure)


---
#### IfExpr::do\_evaluate<!-- {{#callable:minja::IfExpr::do_evaluate}} -->
Evaluates an `IfExpr` based on its condition and returns the corresponding expression value.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for evaluating the expressions.
- **Control Flow**:
    - Checks if the `condition` is null and throws a runtime error if it is.
    - Checks if the `then_expr` is null and throws a runtime error if it is.
    - Evaluates the `condition` expression using the provided `context` and checks if it evaluates to true.
    - If the condition is true, evaluates and returns the value of `then_expr` using the context.
    - If the condition is false and `else_expr` is not null, evaluates and returns the value of `else_expr` using the context.
    - If both the condition is false and `else_expr` is null, returns a null value.
- **Output**: Returns a `Value` object representing the result of evaluating either the `then_expr` or `else_expr`, or null if both are not applicable.
- **See also**: [`minja::IfExpr`](#minjaIfExpr)  (Data Structure)



---
### LiteralExpr<!-- {{#data_structure:minja::LiteralExpr}} -->
- **Type**: `class`
- **Members**:
    - `value`: Stores the literal value of the expression.
- **Description**: The `LiteralExpr` class represents a literal expression in the expression evaluation context, encapsulating a `Value` that holds the actual data to be evaluated.
- **Member Functions**:
    - [`minja::LiteralExpr::LiteralExpr`](#LiteralExprLiteralExpr)
    - [`minja::LiteralExpr::do_evaluate`](#LiteralExprdo_evaluate)
- **Inherits From**:
    - [`minja::Expression::Expression`](#ExpressionExpression)

**Methods**

---
#### LiteralExpr::LiteralExpr<!-- {{#callable:minja::LiteralExpr::LiteralExpr}} -->
The `LiteralExpr` class represents a literal value in an expression.
- **Inputs**:
    - `loc`: A `Location` object that indicates the position of the literal in the source code.
    - `v`: A `Value` object that holds the literal value.
- **Control Flow**:
    - The constructor initializes the base `Expression` class with the provided location.
    - The `do_evaluate` method returns the stored `value` when called.
- **Output**: The output is a `Value` object representing the literal value stored in the `LiteralExpr` instance.
- **See also**: [`minja::LiteralExpr`](#minjaLiteralExpr)  (Data Structure)


---
#### LiteralExpr::do\_evaluate<!-- {{#callable:minja::LiteralExpr::do_evaluate}} -->
Evaluates and returns the stored `Value` of a `LiteralExpr`.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements as it directly returns the member variable `value`.
- **Output**: Returns the `Value` object stored in the `LiteralExpr` instance.
- **See also**: [`minja::LiteralExpr`](#minjaLiteralExpr)  (Data Structure)



---
### ArrayExpr<!-- {{#data_structure:minja::ArrayExpr}} -->
- **Type**: `class`
- **Members**:
    - `elements`: A vector of shared pointers to `Expression` objects representing the elements of the array.
- **Description**: The `ArrayExpr` class represents an array expression in the context of an expression evaluation framework, containing a collection of `Expression` objects that can be evaluated to produce an array value.
- **Member Functions**:
    - [`minja::ArrayExpr::ArrayExpr`](#ArrayExprArrayExpr)
    - [`minja::ArrayExpr::do_evaluate`](#ArrayExprdo_evaluate)
- **Inherits From**:
    - [`minja::Expression::Expression`](#ExpressionExpression)

**Methods**

---
#### ArrayExpr::ArrayExpr<!-- {{#callable:minja::ArrayExpr::ArrayExpr}} -->
The `do_evaluate` method of the `ArrayExpr` class evaluates an array expression by evaluating each of its elements and returning the results as an array.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for evaluating the expressions.
- **Control Flow**:
    - Initializes a `Value` object to hold the result as an array.
    - Iterates over each expression in the `elements` vector.
    - Checks if the current expression is null and throws an error if it is.
    - Evaluates each expression using the provided `context` and appends the result to the `result` array.
    - Returns the populated `result` array after all elements have been evaluated.
- **Output**: Returns a `Value` object representing an array containing the evaluated results of each expression in the `elements` vector.
- **See also**: [`minja::ArrayExpr`](#minjaArrayExpr)  (Data Structure)


---
#### ArrayExpr::do\_evaluate<!-- {{#callable:minja::ArrayExpr::do_evaluate}} -->
Evaluates an array of expressions and returns their results as a `Value` array.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for evaluating the expressions.
- **Control Flow**:
    - Initializes an empty `Value` array to store results.
    - Iterates over each expression in the `elements` vector.
    - Checks if the current expression is null; if so, throws a runtime error.
    - Evaluates the current expression using the provided `context` and appends the result to the `result` array.
    - Returns the populated `result` array after all expressions have been evaluated.
- **Output**: Returns a `Value` object representing an array containing the results of evaluating each expression in the `elements` vector.
- **See also**: [`minja::ArrayExpr`](#minjaArrayExpr)  (Data Structure)



---
### DictExpr<!-- {{#data_structure:minja::DictExpr}} -->
- **Type**: `class`
- **Members**:
    - `elements`: A vector of pairs containing shared pointers to `Expression` objects representing key-value pairs.
- **Description**: The `DictExpr` class represents a dictionary-like expression in a templating language, allowing for the evaluation of key-value pairs where both keys and values are expressions that can be evaluated in a given context.
- **Member Functions**:
    - [`minja::DictExpr::DictExpr`](#DictExprDictExpr)
    - [`minja::DictExpr::do_evaluate`](#DictExprdo_evaluate)
- **Inherits From**:
    - [`minja::Expression::Expression`](#ExpressionExpression)

**Methods**

---
#### DictExpr::DictExpr<!-- {{#callable:minja::DictExpr::DictExpr}} -->
The `DictExpr` class represents a dictionary expression that evaluates to a key-value mapping.
- **Inputs**:
    - `loc`: A `Location` object that indicates the source location of the expression.
    - `e`: A vector of pairs, where each pair consists of a shared pointer to an `Expression` representing a key and a shared pointer to an `Expression` representing a value.
- **Control Flow**:
    - The constructor initializes the base `Expression` class with the provided location and moves the elements into the `elements` member variable.
    - The `do_evaluate` method initializes an empty `Value` object to store the result.
    - It iterates over each key-value pair in the `elements` vector.
    - For each pair, it checks if the key or value is null, throwing an exception if so.
    - It evaluates both the key and value expressions in the context provided and sets them in the result object.
    - Finally, it returns the populated result object.
- **Output**: The output is a `Value` object representing the dictionary constructed from the evaluated key-value pairs.
- **See also**: [`minja::DictExpr`](#minjaDictExpr)  (Data Structure)


---
#### DictExpr::do\_evaluate<!-- {{#callable:minja::DictExpr::do_evaluate}} -->
Evaluates a dictionary expression by computing the keys and values in the given context.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for evaluating the dictionary keys and values.
- **Control Flow**:
    - Initializes an empty `Value` object to store the result.
    - Iterates over each key-value pair in the `elements` vector.
    - Checks if the key is null and throws a runtime error if it is.
    - Checks if the value is null and throws a runtime error if it is.
    - Evaluates the key and value expressions using the provided context and sets them in the result object.
    - Returns the populated result object after processing all key-value pairs.
- **Output**: Returns a `Value` object representing the evaluated dictionary, with keys and values computed based on the provided context.
- **See also**: [`minja::DictExpr`](#minjaDictExpr)  (Data Structure)



---
### SliceExpr<!-- {{#data_structure:minja::SliceExpr}} -->
- **Type**: `class`
- **Members**:
    - `start`: A shared pointer to an `Expression` representing the start of the slice.
    - `end`: A shared pointer to an `Expression` representing the end of the slice.
    - `step`: A shared pointer to an `Expression` representing the step of the slice, which is optional.
- **Description**: The `SliceExpr` class represents a slice operation in an expression context, allowing for the specification of a start, end, and optional step for slicing collections or strings.
- **Member Functions**:
    - [`minja::SliceExpr::SliceExpr`](#SliceExprSliceExpr)
    - [`minja::SliceExpr::do_evaluate`](#SliceExprdo_evaluate)
- **Inherits From**:
    - [`minja::Expression::Expression`](#ExpressionExpression)

**Methods**

---
#### SliceExpr::SliceExpr<!-- {{#callable:minja::SliceExpr::SliceExpr}} -->
The `SliceExpr` class represents a slicing expression with start, end, and optional step parameters.
- **Inputs**:
    - `loc`: A `Location` object that indicates the position of the expression in the source code.
    - `s`: A `shared_ptr` to an `Expression` representing the start of the slice.
    - `e`: A `shared_ptr` to an `Expression` representing the end of the slice.
    - `st`: An optional `shared_ptr` to an `Expression` representing the step of the slice, defaulting to `nullptr`.
- **Control Flow**:
    - The constructor initializes the `SliceExpr` object by moving the provided `shared_ptr` expressions for start, end, and step into member variables.
    - The `do_evaluate` method is defined but throws a runtime error indicating that the slicing functionality is not implemented.
- **Output**: The output of the `do_evaluate` method is a runtime error stating that the slicing expression is not implemented.
- **See also**: [`minja::SliceExpr`](#minjaSliceExpr)  (Data Structure)


---
#### SliceExpr::do\_evaluate<!-- {{#callable:minja::SliceExpr::do_evaluate}} -->
The `do_evaluate` method in the `SliceExpr` class throws a runtime error indicating that the slice expression is not implemented.
- **Inputs**: None
- **Control Flow**:
    - The method immediately throws a `std::runtime_error` with the message 'SliceExpr not implemented'.
- **Output**: The method does not return a value; instead, it raises an exception indicating that the functionality is not yet implemented.
- **See also**: [`minja::SliceExpr`](#minjaSliceExpr)  (Data Structure)



---
### SubscriptExpr<!-- {{#data_structure:minja::SubscriptExpr}} -->
- **Type**: `class`
- **Members**:
    - `base`: A shared pointer to the base expression being indexed.
    - `index`: A shared pointer to the expression representing the index.
- **Description**: The `SubscriptExpr` class represents an expression that accesses an element of a collection (like an array or a string) using an index. It inherits from the `Expression` class and contains two members: `base`, which is the collection being accessed, and `index`, which specifies the index of the element to retrieve. The class provides functionality to evaluate the expression, handling both direct indexing and slicing operations.
- **Member Functions**:
    - [`minja::SubscriptExpr::SubscriptExpr`](#SubscriptExprSubscriptExpr)
    - [`minja::SubscriptExpr::do_evaluate`](#SubscriptExprdo_evaluate)
- **Inherits From**:
    - [`minja::Expression::Expression`](#ExpressionExpression)

**Methods**

---
#### SubscriptExpr::SubscriptExpr<!-- {{#callable:minja::SubscriptExpr::SubscriptExpr}} -->
The `do_evaluate` method of the `SubscriptExpr` class evaluates a subscript expression, allowing access to elements of arrays or strings based on an index or slice.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for evaluating the expression.
- **Control Flow**:
    - Checks if the `base` expression is null and throws an error if it is.
    - Checks if the `index` expression is null and throws an error if it is.
    - Evaluates the `base` expression to get the target value.
    - Checks if the `index` is a `SliceExpr` and handles slicing logic if true.
    - If the index is not a slice, it evaluates the index expression to get the index value.
    - Handles different types of target values (string or array) and performs the appropriate subscript operation.
    - Throws errors for unsupported types or null values.
- **Output**: Returns a `Value` object representing the result of the subscript operation, which can be a single element from an array or string, or a sliced portion of a string or array.
- **See also**: [`minja::SubscriptExpr`](#minjaSubscriptExpr)  (Data Structure)


---
#### SubscriptExpr::do\_evaluate<!-- {{#callable:minja::SubscriptExpr::do_evaluate}} -->
Evaluates a subscript expression, allowing for both direct indexing and slicing of arrays or strings.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for evaluation.
- **Control Flow**:
    - Checks if `base` and `index` are not null, throwing an error if either is null.
    - Evaluates the `base` expression to get the target value.
    - If `index` is a `SliceExpr`, it calculates the start, end, and step values for slicing.
    - Handles string and array types differently when slicing, returning the appropriate substring or subarray.
    - If `index` is not a `SliceExpr`, it evaluates the `index` expression and retrieves the corresponding value from the target value.
- **Output**: Returns a `Value` representing the result of the subscript operation, which can be a substring, subarray, or a specific element from the target value.
- **See also**: [`minja::SubscriptExpr`](#minjaSubscriptExpr)  (Data Structure)



---
### UnaryOpExpr<!-- {{#data_structure:minja::UnaryOpExpr}} -->
- **Type**: `class`
- **Members**:
    - `expr`: A shared pointer to an `Expression` that this unary operation is applied to.
    - `op`: An enumeration value representing the type of unary operation.
- **Description**: The `UnaryOpExpr` class represents a unary operation in an expression tree, allowing operations such as negation or logical NOT to be performed on a single operand, which is encapsulated in the `expr` member. The operation type is specified by the `op` member, which can take values from the `Op` enumeration, indicating the specific unary operation to be executed during evaluation.
- **Member Functions**:
    - [`minja::UnaryOpExpr::UnaryOpExpr`](#UnaryOpExprUnaryOpExpr)
    - [`minja::UnaryOpExpr::do_evaluate`](#UnaryOpExprdo_evaluate)
- **Inherits From**:
    - [`minja::Expression::Expression`](#ExpressionExpression)

**Methods**

---
#### UnaryOpExpr::UnaryOpExpr<!-- {{#callable:minja::UnaryOpExpr::UnaryOpExpr}} -->
Evaluates a unary operation on an expression.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for evaluation.
- **Control Flow**:
    - Checks if the `expr` member is null and throws a runtime error if it is.
    - Evaluates the `expr` member using the provided `context`.
    - Uses a switch statement to determine the operation specified by `op` and returns the result based on the operation.
- **Output**: Returns a `Value` object that represents the result of the unary operation.
- **See also**: [`minja::UnaryOpExpr`](#minjaUnaryOpExpr)  (Data Structure)


---
#### UnaryOpExpr::do\_evaluate<!-- {{#callable:minja::UnaryOpExpr::do_evaluate}} -->
Evaluates a unary operation on an expression.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for the evaluation.
- **Control Flow**:
    - Checks if the `expr` member is null and throws a runtime error if it is.
    - Evaluates the `expr` using the provided `context`.
    - Uses a switch statement to determine the operation specified by `op`.
    - Returns the evaluated result based on the operation: returns the value for `Plus`, negates the value for `Minus`, applies logical negation for `LogicalNot`, and throws an error for unsupported operations.
- **Output**: Returns a `Value` object representing the result of the unary operation.
- **See also**: [`minja::UnaryOpExpr`](#minjaUnaryOpExpr)  (Data Structure)



---
### Op<!-- {{#data_structure:minja::BinaryOpExpr::Op}} -->
- **Type**: `enum class`
- **Members**:
    - `StrConcat`: Represents string concatenation operation.
    - `Add`: Represents addition operation.
    - `Sub`: Represents subtraction operation.
    - `Mul`: Represents multiplication operation.
    - `MulMul`: Represents exponentiation operation.
    - `Div`: Represents division operation.
    - `DivDiv`: Represents integer division operation.
    - `Mod`: Represents modulus operation.
    - `Eq`: Represents equality comparison.
    - `Ne`: Represents inequality comparison.
    - `Lt`: Represents less than comparison.
    - `Gt`: Represents greater than comparison.
    - `Le`: Represents less than or equal to comparison.
    - `Ge`: Represents greater than or equal to comparison.
    - `And`: Represents logical AND operation.
    - `Or`: Represents logical OR operation.
    - `In`: Represents membership test.
    - `NotIn`: Represents non-membership test.
    - `Is`: Represents identity test.
    - `IsNot`: Represents non-identity test.
- **Description**: The `Op` enum class defines a set of operations that can be performed in an expression context, including arithmetic operations, comparison operations, and logical operations, facilitating the evaluation of expressions in a structured manner.


---
### BinaryOpExpr<!-- {{#data_structure:minja::BinaryOpExpr}} -->
- **Type**: `class`
- **Members**:
    - `left`: A shared pointer to the left operand of the binary operation.
    - `right`: A shared pointer to the right operand of the binary operation.
    - `op`: An enumeration value representing the type of binary operation.
- **Description**: The `BinaryOpExpr` class represents a binary operation expression in an expression tree, encapsulating two operands (`left` and `right`) and the operation type (`op`) which can be one of several predefined operations such as addition, subtraction, or logical operations.
- **Member Functions**:
    - [`minja::BinaryOpExpr::BinaryOpExpr`](#BinaryOpExprBinaryOpExpr)
    - [`minja::BinaryOpExpr::do_evaluate`](#BinaryOpExprdo_evaluate)
- **Inherits From**:
    - [`minja::Expression::Expression`](#ExpressionExpression)

**Methods**

---
#### BinaryOpExpr::BinaryOpExpr<!-- {{#callable:minja::BinaryOpExpr::BinaryOpExpr}} -->
The `BinaryOpExpr` class represents a binary operation expression that evaluates two operands with a specified operator.
- **Inputs**:
    - `loc`: A `Location` object that indicates the source location of the expression.
    - `l`: A `std::shared_ptr<Expression>` representing the left operand of the binary operation.
    - `r`: A `std::shared_ptr<Expression>` representing the right operand of the binary operation.
    - `o`: An `Op` enum value that specifies the type of binary operation to perform.
- **Control Flow**:
    - The constructor initializes the `BinaryOpExpr` with the provided location, left operand, right operand, and operator.
    - The `do_evaluate` method first checks if the left and right operands are valid (not null).
    - It evaluates the left operand and stores the result.
    - Depending on the operator, it performs the corresponding operation using the evaluated left and right operands.
    - For operators like 'is' and 'is not', it checks the type of the right operand against the left operand's type.
    - For logical operators 'and' and 'or', it short-circuits evaluation based on the result of the left operand.
    - If the left operand is callable, it creates a callable that evaluates the right operand with the result of the left operand.
- **Output**: The output is a `Value` object that represents the result of the binary operation.
- **See also**: [`minja::BinaryOpExpr`](#minjaBinaryOpExpr)  (Data Structure)


---
#### BinaryOpExpr::do\_evaluate<!-- {{#callable:minja::BinaryOpExpr::do_evaluate}} -->
Evaluates a binary operation between two expressions based on the specified operator.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for the evaluation.
- **Control Flow**:
    - Checks if the `left` or `right` expressions are null and throws an exception if they are.
    - Evaluates the `left` expression using the provided `context`.
    - Defines a lambda function `do_eval` to evaluate the binary operation based on the operator `op`.
    - Handles special cases for the `Is` and `IsNot` operators, checking the type of the right expression.
    - Handles logical operations (`And`, `Or`) by evaluating the left expression and conditionally evaluating the right expression.
    - For other operators, evaluates the right expression and performs the corresponding operation based on the operator type.
    - If the left expression is callable, it returns a callable value that evaluates the left expression with arguments.
- **Output**: Returns a [`Value`](#ValueValue) object representing the result of the evaluated binary operation.
- **Functions called**:
    - [`minja::Value::Value`](#ValueValue)
- **See also**: [`minja::BinaryOpExpr`](#minjaBinaryOpExpr)  (Data Structure)



---
### ArgumentsExpression<!-- {{#data_structure:minja::ArgumentsExpression}} -->
- **Type**: `struct`
- **Members**:
    - `args`: A vector of shared pointers to `Expression` representing positional arguments.
    - `kwargs`: A vector of pairs containing a string and a shared pointer to `Expression` representing keyword arguments.
- **Description**: The `ArgumentsExpression` struct is designed to hold a collection of arguments for function calls, encapsulating both positional arguments (`args`) and keyword arguments (`kwargs`). It provides a method to evaluate these arguments in a given context, handling special cases such as argument expansion for arrays and dictionaries.
- **Member Functions**:
    - [`minja::ArgumentsExpression::evaluate`](#ArgumentsExpressionevaluate)

**Methods**

---
#### ArgumentsExpression::evaluate<!-- {{#callable:minja::ArgumentsExpression::evaluate}} -->
Evaluates a collection of arguments, handling expansion for arrays and dictionaries.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for evaluation.
- **Control Flow**:
    - Iterates over each argument in the `args` vector.
    - Checks if the argument is a `UnaryOpExpr` and handles it based on its operation type.
    - For `Expansion`, evaluates the expression and checks if it is an array, then expands its elements into `vargs.args`.
    - For `ExpansionDict`, evaluates the expression and checks if it is an object, then expands its key-value pairs into `vargs.kwargs`.
    - For non-unary expressions, evaluates the argument and adds it to `vargs.args`.
    - After processing positional arguments, iterates over `kwargs` to evaluate and add them to `vargs.kwargs`.
- **Output**: Returns an `ArgumentsValue` object containing evaluated positional and keyword arguments.
- **See also**: [`minja::ArgumentsExpression`](#minjaArgumentsExpression)  (Data Structure)



---
### MethodCallExpr<!-- {{#data_structure:minja::MethodCallExpr}} -->
- **Type**: `class`
- **Members**:
    - `object`: A shared pointer to an `Expression` representing the object on which the method is called.
    - `method`: A shared pointer to a `VariableExpr` representing the method being called.
    - `args`: An `ArgumentsExpression` object representing the arguments passed to the method.
- **Description**: The `MethodCallExpr` class represents a method call expression in the expression evaluation context, encapsulating the target object, the method to be invoked, and the arguments to be passed to that method.
- **Member Functions**:
    - [`minja::MethodCallExpr::MethodCallExpr`](#MethodCallExprMethodCallExpr)
    - [`minja::MethodCallExpr::do_evaluate`](#MethodCallExprdo_evaluate)
- **Inherits From**:
    - [`minja::Expression::Expression`](#ExpressionExpression)

**Methods**

---
#### MethodCallExpr::MethodCallExpr<!-- {{#callable:minja::MethodCallExpr::MethodCallExpr}} -->
The `MethodCallExpr` class represents a method call expression that evaluates a method on an object with given arguments.
- **Inputs**:
    - `loc`: A `Location` object representing the source location of the method call.
    - `obj`: A `std::shared_ptr<Expression>` representing the object on which the method is called.
    - `m`: A `std::shared_ptr<VariableExpr>` representing the method being called.
    - `a`: An `ArgumentsExpression` object representing the arguments passed to the method.
- **Control Flow**:
    - The method first checks if the `object` or `method` is null, throwing an exception if either is.
    - It evaluates the `object` and the `args` to get the actual values to work with.
    - If the evaluated object is null, it throws an error indicating that a method cannot be called on a null object.
    - If the object is an array, it checks the method name and performs the corresponding array operation (e.g., append, pop, insert).
    - If the object is an object, it checks for method names like 'items', 'pop', or 'get' and performs the corresponding operations.
    - If the object is a string, it checks for string methods like 'strip', 'split', 'capitalize', etc., and performs the corresponding operations.
    - If the method name does not match any known methods, it throws an error indicating an unknown method.
- **Output**: The method returns a `Value` object representing the result of the method call, or throws an exception if an error occurs.
- **See also**: [`minja::MethodCallExpr`](#minjaMethodCallExpr)  (Data Structure)


---
#### MethodCallExpr::do\_evaluate<!-- {{#callable:minja::MethodCallExpr::do_evaluate}} -->
Evaluates a method call expression on a given object with specified arguments.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for evaluation.
- **Control Flow**:
    - Checks if the `object` and `method` members are not null, throwing an exception if either is null.
    - Evaluates the `object` to get the target on which the method is called.
    - Evaluates the `args` to get the arguments for the method call.
    - Checks if the evaluated object is null, throwing an exception if it is.
    - If the object is an array, it checks the method name and performs the corresponding array operation (append, pop, insert).
    - If the object is an object, it checks the method name and performs the corresponding object operation (items, pop, get, or call a callable property).
    - If the object is a string, it checks the method name and performs the corresponding string operation (strip, lstrip, rstrip, split, capitalize, endswith, startswith, title).
    - If the method name does not match any known methods, it throws an exception for an unknown method.
- **Output**: Returns a [`Value`](#ValueValue) object representing the result of the method call, or an empty [`Value`](#ValueValue) for methods that do not return a value.
- **Functions called**:
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::strip`](#minjastrip)
    - [`minja::split`](#minjasplit)
    - [`minja::capitalize`](#minjacapitalize)
- **See also**: [`minja::MethodCallExpr`](#minjaMethodCallExpr)  (Data Structure)



---
### CallExpr<!-- {{#data_structure:minja::CallExpr}} -->
- **Type**: `class`
- **Members**:
    - `object`: A shared pointer to an `Expression` representing the object being called.
    - `args`: An `ArgumentsExpression` containing the arguments for the function call.
- **Description**: The `CallExpr` class represents a function call expression in the expression evaluation context, encapsulating the callable object and its arguments, and providing a mechanism to evaluate the call within a given context.
- **Member Functions**:
    - [`minja::CallExpr::CallExpr`](#CallExprCallExpr)
    - [`minja::CallExpr::do_evaluate`](#CallExprdo_evaluate)
- **Inherits From**:
    - [`minja::Expression::Expression`](#ExpressionExpression)

**Methods**

---
#### CallExpr::CallExpr<!-- {{#callable:minja::CallExpr::CallExpr}} -->
The `CallExpr` class represents a function call expression in the template engine.
- **Inputs**:
    - `loc`: A `Location` object that indicates the position of the expression in the source code.
    - `obj`: A `std::shared_ptr<Expression>` representing the object being called.
    - `a`: An `ArgumentsExpression` object containing the arguments for the function call.
- **Control Flow**:
    - The method first checks if the `object` is null, throwing an error if it is.
    - It evaluates the `object` to get the callable entity.
    - If the evaluated object is not callable, it throws an error.
    - The method then evaluates the arguments provided in `args`.
    - Finally, it calls the evaluated object with the evaluated arguments and returns the result.
- **Output**: Returns a `Value` object that is the result of invoking the callable object with the provided arguments.
- **See also**: [`minja::CallExpr`](#minjaCallExpr)  (Data Structure)


---
#### CallExpr::do\_evaluate<!-- {{#callable:minja::CallExpr::do_evaluate}} -->
Evaluates a function call expression by invoking the callable object with the provided arguments.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for the evaluation.
- **Control Flow**:
    - Checks if the `object` member is null and throws a runtime error if it is.
    - Evaluates the `object` to get the callable function.
    - Checks if the evaluated object is callable and throws a runtime error if it is not.
    - Evaluates the `args` to get the arguments for the function call.
    - Calls the evaluated object with the evaluated arguments and returns the result.
- **Output**: Returns a `Value` object that is the result of invoking the callable with the evaluated arguments.
- **See also**: [`minja::CallExpr`](#minjaCallExpr)  (Data Structure)



---
### FilterExpr<!-- {{#data_structure:minja::FilterExpr}} -->
- **Type**: `class`
- **Members**:
    - `parts`: A vector of shared pointers to `Expression` objects that represent the components of the filter expression.
- **Description**: The `FilterExpr` class is a specialized type of `Expression` that evaluates a sequence of expressions, applying filters in a chain-like manner. It stores its components in a vector and processes them in order, where the result of each expression can be passed as an argument to the next one. This allows for complex filtering logic to be constructed dynamically.
- **Member Functions**:
    - [`minja::FilterExpr::FilterExpr`](#FilterExprFilterExpr)
    - [`minja::FilterExpr::do_evaluate`](#FilterExprdo_evaluate)
    - [`minja::FilterExpr::prepend`](#FilterExprprepend)
- **Inherits From**:
    - [`minja::Expression::Expression`](#ExpressionExpression)

**Methods**

---
#### FilterExpr::FilterExpr<!-- {{#callable:minja::FilterExpr::FilterExpr}} -->
The `do_evaluate` method of the `FilterExpr` class evaluates a sequence of expressions, applying each subsequent expression as a filter to the result of the previous one.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for evaluating the expressions.
- **Control Flow**:
    - The method initializes a `Value` object to store the result of the evaluation.
    - It iterates over each expression in the `parts` vector.
    - For the first expression, it evaluates it directly and stores the result.
    - For subsequent expressions, it checks if the expression is a `CallExpr` and evaluates it accordingly, passing the previous result as an argument.
    - If the expression is not a `CallExpr`, it evaluates it as a callable and applies the previous result as an argument.
- **Output**: The method returns a `Value` object that represents the final result after applying all the filters in sequence.
- **See also**: [`minja::FilterExpr`](#minjaFilterExpr)  (Data Structure)


---
#### FilterExpr::do\_evaluate<!-- {{#callable:minja::FilterExpr::do_evaluate}} -->
Evaluates a sequence of expressions in a `FilterExpr` context.
- **Inputs**:
    - `context`: A shared pointer to a `Context` object that provides the environment for evaluating the expressions.
- **Control Flow**:
    - Initializes a `Value` object to store the result and a boolean flag `first` to track the first iteration.
    - Iterates over each `part` in the `parts` vector.
    - Throws an exception if any `part` is null.
    - If it's the first part, evaluates it directly and assigns the result to `result`.
    - For subsequent parts, checks if the part is a `CallExpr` and evaluates the target and arguments accordingly.
    - Calls the evaluated target with the accumulated result and new arguments, updating `result`.
- **Output**: Returns the final evaluated `Value` after processing all parts.
- **See also**: [`minja::FilterExpr`](#minjaFilterExpr)  (Data Structure)


---
#### FilterExpr::prepend<!-- {{#callable:minja::FilterExpr::prepend}} -->
Prepends a new `Expression` to the beginning of the `parts` vector in the `FilterExpr` class.
- **Inputs**:
    - `e`: A rvalue reference to a `std::shared_ptr<Expression>` that represents the expression to be prepended.
- **Control Flow**:
    - The function uses `std::move` to transfer ownership of the `std::shared_ptr<Expression>` to avoid unnecessary copies.
    - The `std::shared_ptr<Expression>` is inserted at the beginning of the `parts` vector using the `insert` method.
- **Output**: This function does not return a value; it modifies the internal state of the `FilterExpr` instance by adding a new expression to its `parts` vector.
- **See also**: [`minja::FilterExpr`](#minjaFilterExpr)  (Data Structure)



---
### Parser<!-- {{#data_structure:minja::Parser}} -->
- **Type**: `class`
- **Members**:
    - `template_str`: A shared pointer to the template string being parsed.
    - `start`: An iterator pointing to the start of the template string.
    - `end`: An iterator pointing to the end of the template string.
    - `it`: An iterator used for parsing through the template string.
    - `options`: An `Options` struct that holds parsing options.
- **Description**: The `Parser` class is responsible for parsing a template string and converting it into a structured representation, handling various expressions, control structures, and whitespace management according to specified options.
- **Member Functions**:
    - [`minja::Parser::Parser`](#ParserParser)
    - [`minja::Parser::consumeSpaces`](#ParserconsumeSpaces)
    - [`minja::Parser::parseString`](#ParserparseString)
    - [`minja::Parser::parseNumber`](#ParserparseNumber)
    - [`minja::Parser::parseConstant`](#ParserparseConstant)
    - [`minja::Parser::peekSymbols`](#ParserpeekSymbols)
    - [`minja::Parser::consumeTokenGroups`](#ParserconsumeTokenGroups)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::parseExpression`](#ParserparseExpression)
    - [`minja::Parser::get_location`](#Parserget_location)
    - [`minja::Parser::parseIfExpression`](#ParserparseIfExpression)
    - [`minja::Parser::parseLogicalOr`](#ParserparseLogicalOr)
    - [`minja::Parser::parseLogicalNot`](#ParserparseLogicalNot)
    - [`minja::Parser::parseLogicalAnd`](#ParserparseLogicalAnd)
    - [`minja::Parser::parseLogicalCompare`](#ParserparseLogicalCompare)
    - [`minja::Parser::parseParameters`](#ParserparseParameters)
    - [`minja::Parser::parseCallArgs`](#ParserparseCallArgs)
    - [`minja::Parser::parseIdentifier`](#ParserparseIdentifier)
    - [`minja::Parser::parseStringConcat`](#ParserparseStringConcat)
    - [`minja::Parser::parseMathPow`](#ParserparseMathPow)
    - [`minja::Parser::parseMathPlusMinus`](#ParserparseMathPlusMinus)
    - [`minja::Parser::parseMathMulDiv`](#ParserparseMathMulDiv)
    - [`minja::Parser::call_func`](#Parsercall_func)
    - [`minja::Parser::parseMathUnaryPlusMinus`](#ParserparseMathUnaryPlusMinus)
    - [`minja::Parser::parseExpansion`](#ParserparseExpansion)
    - [`minja::Parser::parseValueExpression`](#ParserparseValueExpression)
    - [`minja::Parser::parseBracedExpressionOrArray`](#ParserparseBracedExpressionOrArray)
    - [`minja::Parser::parseArray`](#ParserparseArray)
    - [`minja::Parser::parseDictionary`](#ParserparseDictionary)
    - [`minja::Parser::parsePreSpace`](#ParserparsePreSpace)
    - [`minja::Parser::parsePostSpace`](#ParserparsePostSpace)
    - [`minja::Parser::parseVarNames`](#ParserparseVarNames)
    - [`minja::Parser::unexpected`](#Parserunexpected)
    - [`minja::Parser::unterminated`](#Parserunterminated)
    - [`minja::Parser::tokenize`](#Parsertokenize)
    - [`minja::Parser::parseTemplate`](#ParserparseTemplate)
    - [`minja::Parser::parse`](#Parserparse)

**Methods**

---
#### Parser::Parser<!-- {{#callable:minja::Parser::Parser}} -->
The `Parser` constructor initializes a parser instance with a template string and options.
- **Inputs**:
    - `template_str`: A shared pointer to a string that represents the template to be parsed.
    - `options`: An `Options` structure that contains configuration settings for the parser.
- **Control Flow**:
    - Checks if the `template_str` is null and throws a runtime error if it is.
    - Initializes the `start`, `it`, and `end` iterators to the beginning and end of the `template_str`.
- **Output**: The constructor does not return a value but initializes the `Parser` object.
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::consumeSpaces<!-- {{#callable:minja::Parser::consumeSpaces}} -->
The `consumeSpaces` function advances the iterator past any whitespace characters in the input string.
- **Inputs**:
    - `space_handling`: An optional parameter of type `SpaceHandling` that determines how whitespace should be handled, defaulting to `SpaceHandling::Strip`.
- **Control Flow**:
    - The function checks if the `space_handling` parameter is set to `SpaceHandling::Strip`.
    - If it is, a while loop iterates through the input string, incrementing the iterator `it` until it reaches a non-whitespace character or the end of the string.
- **Output**: The function always returns `true`, indicating that the operation was successful.
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseString<!-- {{#callable:minja::Parser::parseString}} -->
Parses a string from the input template, handling escape sequences and quotes.
- **Inputs**:
    - `this->template_str`: A shared pointer to the template string being parsed.
    - `it`: An iterator pointing to the current character in the template string.
    - `end`: An iterator pointing to the end of the template string.
- **Control Flow**:
    - The function first calls `consumeSpaces()` to skip any leading whitespace.
    - It checks if the iterator has reached the end of the string; if so, it returns nullptr.
    - If the current character is a double quote, it calls the `doParse` lambda with '"' as the argument.
    - If the current character is a single quote, it calls the `doParse` lambda with '\'' as the argument.
    - If neither condition is met, it returns nullptr.
- **Output**: Returns a unique pointer to a string containing the parsed content, or nullptr if parsing fails.
- **Functions called**:
    - [`minja::Parser::consumeSpaces`](#ParserconsumeSpaces)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseNumber<!-- {{#callable:minja::Parser::parseNumber}} -->
Parses a number from a string and returns it as a JSON object.
- **Inputs**:
    - `it`: A reference to a character iterator that points to the current position in the input string.
    - `end`: A constant iterator that marks the end of the input string.
- **Control Flow**:
    - The function starts by saving the current position of the iterator.
    - It consumes any leading whitespace in the input string.
    - It checks for an optional sign ('-' or '+') and advances the iterator if found.
    - It enters a loop to read digits, a decimal point, or an exponent indicator ('e' or 'E').
    - If multiple decimal points or exponents are found, an error is thrown.
    - If no valid number characters are found, the iterator is reset and an empty JSON object is returned.
    - The valid number string is parsed into a JSON object, and any parsing errors are caught and rethrown with a descriptive message.
- **Output**: Returns a JSON object representing the parsed number, or an empty JSON object if parsing fails.
- **Functions called**:
    - [`minja::Parser::consumeSpaces`](#ParserconsumeSpaces)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseConstant<!-- {{#callable:minja::Parser::parseConstant}} -->
Parses a constant value from the input string.
- **Inputs**:
    - `it`: An iterator pointing to the current position in the input string.
    - `end`: An iterator pointing to the end of the input string.
- **Control Flow**:
    - The function starts by saving the current position of the iterator.
    - It consumes any whitespace characters before processing the next token.
    - If the iterator has reached the end of the input, it returns nullptr.
    - If the next character is a quote (single or double), it attempts to parse a string.
    - If a string is successfully parsed, it returns a shared pointer to a new `Value` containing the string.
    - It then checks for boolean and None keywords using a regex pattern.
    - If a valid token is found, it creates a `Value` based on the token's type (true, false, or None).
    - If no valid token is found, it attempts to parse a number.
    - If a number is successfully parsed, it returns a shared pointer to a new `Value` containing the number.
    - If no valid constant is found, it resets the iterator to the saved position and returns nullptr.
- **Output**: Returns a shared pointer to a `Value` object representing the parsed constant, or nullptr if no constant is found.
- **Functions called**:
    - [`minja::Parser::consumeSpaces`](#ParserconsumeSpaces)
    - [`minja::Parser::parseString`](#ParserparseString)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::parseNumber`](#ParserparseNumber)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::peekSymbols<!-- {{#callable:minja::Parser::peekSymbols}} -->
The `peekSymbols` function checks if any of the provided symbols match the current position in a string.
- **Inputs**:
    - `symbols`: A vector of strings representing the symbols to check against the current position in the string.
- **Control Flow**:
    - Iterates over each symbol in the `symbols` vector.
    - For each symbol, it checks if the remaining characters in the string are at least as many as the length of the symbol.
    - If the substring from the current iterator position matches the symbol, it returns true.
    - If no symbols match, it returns false after the loop.
- **Output**: Returns a boolean value indicating whether any of the specified symbols match the current position in the string.
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::consumeTokenGroups<!-- {{#callable:minja::Parser::consumeTokenGroups}} -->
The `consumeTokenGroups` function extracts and returns a vector of matched token groups from the input string based on a provided regular expression.
- **Inputs**:
    - `regex`: A `std::regex` object that defines the pattern to match against the input string.
    - `space_handling`: An optional `SpaceHandling` enum value that determines how to handle whitespace before matching (default is `SpaceHandling::Strip`).
- **Control Flow**:
    - The function starts by saving the current position of the iterator `it` to `start`.
    - It calls [`consumeSpaces`](#ParserconsumeSpaces) to skip any leading whitespace based on the `space_handling` parameter.
    - A `std::smatch` object `match` is declared to hold the results of the regex search.
    - If a match is found at the current position of `it`, the iterator is advanced by the length of the matched string.
    - A vector `ret` is created to store the matched groups, which are extracted from `match` and pushed into `ret`.
    - The function returns the vector `ret` containing the matched token groups.
    - If no match is found, the iterator `it` is reset to the original position stored in `start`, and an empty vector is returned.
- **Output**: Returns a vector of strings containing the matched token groups, or an empty vector if no match is found.
- **Functions called**:
    - [`minja::Parser::consumeSpaces`](#ParserconsumeSpaces)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::consumeToken<!-- {{#callable:minja::Parser::consumeToken}} -->
The `consumeToken` function extracts a substring from the input string that matches a given regular expression, optionally skipping whitespace.
- **Inputs**:
    - `regex`: A `std::regex` object that defines the pattern to match against the input string.
    - `space_handling`: An optional `SpaceHandling` enum value that determines how to handle whitespace before the token.
- **Control Flow**:
    - The function starts by saving the current position of the iterator `it` to `start`.
    - It calls [`consumeSpaces`](#ParserconsumeSpaces) to skip any whitespace based on the `space_handling` parameter.
    - A `std::smatch` object `match` is declared to hold the results of the regex search.
    - If a match is found at the current position (i.e., `match.position() == 0`), the iterator `it` is advanced by the length of the matched string, and the matched string is returned.
    - If no match is found, the iterator `it` is reset to the original position stored in `start`, and an empty string is returned.
- **Output**: Returns the matched substring if found; otherwise, returns an empty string.
- **Functions called**:
    - [`minja::Parser::consumeSpaces`](#ParserconsumeSpaces)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::consumeToken<!-- {{#callable:minja::Parser::consumeToken}} -->
The `consumeToken` function checks for a specific token in the input string and consumes it if found.
- **Inputs**:
    - `token`: A `std::string` representing the token to be consumed.
    - `space_handling`: An optional `SpaceHandling` enum value that determines how to handle spaces before the token.
- **Control Flow**:
    - The function starts by saving the current iterator position in `start`.
    - It calls [`consumeSpaces`](#ParserconsumeSpaces) to skip any leading whitespace based on the `space_handling` parameter.
    - It checks if there are enough characters left in the input to match the token size and if the substring matches the token.
    - If a match is found, it advances the iterator by the length of the token and returns the token.
    - If no match is found, it resets the iterator to the saved position and returns an empty string.
- **Output**: Returns the matched token as a `std::string` if found, otherwise returns an empty string.
- **Functions called**:
    - [`minja::Parser::consumeSpaces`](#ParserconsumeSpaces)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseExpression<!-- {{#callable:minja::Parser::parseExpression}} -->
Parses an expression from the input string, allowing for conditional 'if' expressions.
- **Inputs**:
    - `allow_if_expr`: A boolean flag indicating whether 'if' expressions are allowed in the parsing.
- **Control Flow**:
    - Calls `parseLogicalOr()` to parse the left-hand side of the expression.
    - Checks if the end of the input has been reached; if so, returns the parsed left expression.
    - If 'if' expressions are allowed, checks for the presence of the 'if' token using a regex.
    - If the 'if' token is found, calls `parseIfExpression()` to parse the condition and the else expression.
    - Returns a new `IfExpr` object constructed with the parsed condition, left expression, and else expression.
- **Output**: Returns a shared pointer to an `Expression` object, which can be either a simple expression or an `IfExpr` if an 'if' expression was parsed.
- **Functions called**:
    - [`minja::Parser::parseLogicalOr`](#ParserparseLogicalOr)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::get_location`](#Parserget_location)
    - [`minja::Parser::parseIfExpression`](#ParserparseIfExpression)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::get\_location<!-- {{#callable:minja::Parser::get_location}} -->
The `get_location` method retrieves the current parsing location within the template string.
- **Inputs**: None
- **Control Flow**:
    - The method calculates the distance between the `start` iterator and the current iterator `it` to determine the current position.
    - It constructs and returns a `Location` object containing the template string and the calculated position.
- **Output**: Returns a `Location` object that includes the template string and the current position in the string.
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseIfExpression<!-- {{#callable:minja::Parser::parseIfExpression}} -->
Parses an 'if' expression and optionally an 'else' expression.
- **Inputs**:
    - `none`: No input arguments are required for this function.
- **Control Flow**:
    - Calls `parseLogicalOr()` to parse the condition of the 'if' expression.
    - If the condition is not valid, throws a runtime error.
    - Checks for the presence of an 'else' token using a regex.
    - If an 'else' token is found, calls `parseExpression()` to parse the 'else' expression.
    - If the 'else' expression is not valid, throws a runtime error.
    - Returns a pair containing the parsed condition and the parsed 'else' expression (if any).
- **Output**: Returns a pair of shared pointers to `Expression` objects, where the first is the condition and the second is the 'else' expression (which may be null).
- **Functions called**:
    - [`minja::Parser::parseLogicalOr`](#ParserparseLogicalOr)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::parseExpression`](#ParserparseExpression)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseLogicalOr<!-- {{#callable:minja::Parser::parseLogicalOr}} -->
Parses a logical OR expression from the input.
- **Inputs**:
    - `none`: This function does not take any input arguments.
- **Control Flow**:
    - Calls `parseLogicalAnd()` to parse the left-hand side of the logical OR expression.
    - If the left-hand side is not valid, throws a runtime error indicating the expected left side.
    - Uses a regex to match the 'or' token and enters a loop to parse additional logical AND expressions.
    - For each 'or' token found, it calls `parseLogicalAnd()` to parse the right-hand side.
    - If the right-hand side is not valid, throws a runtime error indicating the expected right side.
    - Creates a new `BinaryOpExpr` for each 'or' operation, combining the left and right expressions.
- **Output**: Returns a shared pointer to an `Expression` representing the logical OR operation.
- **Functions called**:
    - [`minja::Parser::parseLogicalAnd`](#ParserparseLogicalAnd)
    - [`minja::Parser::get_location`](#Parserget_location)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseLogicalNot<!-- {{#callable:minja::Parser::parseLogicalNot}} -->
Parses a logical NOT expression in a template.
- **Inputs**:
    - `none`: This function does not take any input parameters.
- **Control Flow**:
    - The function begins by defining a static regex pattern to match the 'not' keyword.
    - It retrieves the current location in the template for error reporting.
    - If the 'not' token is consumed successfully, it recursively calls `parseLogicalNot` to parse the subsequent expression.
    - If the recursive call does not return a valid expression, an exception is thrown indicating an expected expression after 'not'.
    - If the 'not' token is not found, it calls [`parseLogicalCompare`](#ParserparseLogicalCompare) to parse a different type of logical expression.
- **Output**: Returns a shared pointer to an `UnaryOpExpr` representing the logical NOT operation or a shared pointer to an expression from [`parseLogicalCompare`](#ParserparseLogicalCompare).
- **Functions called**:
    - [`minja::Parser::get_location`](#Parserget_location)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::parseLogicalCompare`](#ParserparseLogicalCompare)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseLogicalAnd<!-- {{#callable:minja::Parser::parseLogicalAnd}} -->
Parses a logical AND expression from the input.
- **Inputs**:
    - `none`: This function does not take any input arguments.
- **Control Flow**:
    - Calls `parseLogicalNot()` to parse the left-hand side of the logical AND expression.
    - If the left-hand side is not valid, throws a runtime error indicating the expected left side.
    - Uses a regex to match the 'and' token and enters a loop to parse additional right-hand sides.
    - For each 'and' token found, calls `parseLogicalNot()` to parse the right-hand side.
    - If the right-hand side is not valid, throws a runtime error indicating the expected right side.
    - Creates a new `BinaryOpExpr` object combining the left and right expressions with the AND operation.
    - Continues looping until no more 'and' tokens are found.
- **Output**: Returns a shared pointer to an `Expression` representing the logical AND operation, which may consist of multiple `BinaryOpExpr` instances.
- **Functions called**:
    - [`minja::Parser::parseLogicalNot`](#ParserparseLogicalNot)
    - [`minja::Parser::get_location`](#Parserget_location)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseLogicalCompare<!-- {{#callable:minja::Parser::parseLogicalCompare}} -->
Parses logical comparison expressions, handling various operators and operands.
- **Inputs**:
    - `none`: The function does not take any explicit input parameters.
- **Control Flow**:
    - Calls `parseStringConcat()` to get the left operand of the comparison.
    - Checks if the left operand is valid; if not, throws an error.
    - Defines regex patterns for comparison operators and the 'not' keyword.
    - Enters a loop to consume comparison operators and parse right operands.
    - Handles the special case for the 'is' operator, checking for negation and parsing an identifier.
    - For each valid operator, creates a `BinaryOpExpr` with the left and right operands and the operator.
    - Returns the final expression after processing all operators.
- **Output**: Returns a shared pointer to an `Expression` representing the logical comparison.
- **Functions called**:
    - [`minja::Parser::parseStringConcat`](#ParserparseStringConcat)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::get_location`](#Parserget_location)
    - [`minja::Parser::parseIdentifier`](#ParserparseIdentifier)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseParameters<!-- {{#callable:minja::Parser::parseParameters}} -->
Parses a list of parameters from a template string, handling both positional and named arguments.
- **Inputs**:
    - `none`: The function does not take any input parameters directly, but it processes the template string being parsed.
- **Control Flow**:
    - The function begins by consuming any leading whitespace.
    - It checks for an opening parenthesis '(' and throws an error if not found.
    - It initializes an empty result container for parameters.
    - It enters a loop to parse parameters until the end of the input is reached.
    - Within the loop, it checks for a closing parenthesis ')' to terminate the parameter list.
    - It attempts to parse an expression, which can be a positional argument or a named argument.
    - If the parsed expression is a variable, it checks for an '=' token to determine if it's a named argument.
    - If it's a named argument, it parses the corresponding value expression.
    - If a comma ',' is found, it continues to the next parameter; otherwise, it checks for a closing parenthesis.
    - If the closing parenthesis is not found when expected, it throws an error.
- **Output**: Returns a vector of pairs, where each pair consists of a parameter name (or an empty string for positional parameters) and the corresponding expression.
- **Functions called**:
    - [`minja::Parser::consumeSpaces`](#ParserconsumeSpaces)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::parseExpression`](#ParserparseExpression)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseCallArgs<!-- {{#callable:minja::Parser::parseCallArgs}} -->
Parses a set of call arguments from a template string.
- **Inputs**:
    - `none`: No explicit input arguments; the function reads from the internal iterator.
- **Control Flow**:
    - The function starts by consuming any leading whitespace.
    - It checks for an opening parenthesis '(' and throws an error if not found.
    - It initializes an `ArgumentsExpression` object to store parsed arguments.
    - It enters a loop to parse expressions until the end of the input is reached.
    - If a closing parenthesis ')' is encountered, the loop exits and returns the result.
    - For each expression, it checks if it is a variable and if it is followed by an '=' sign for named arguments.
    - If a named argument is found, it parses the value expression and adds it to the `kwargs` map.
    - If not, it adds the expression to the `args` vector.
    - If a comma ',' is encountered, it continues parsing for more arguments.
    - If a closing parenthesis ')' is not found after the last argument, it throws an error.
- **Output**: Returns an `ArgumentsExpression` object containing positional and keyword arguments parsed from the input.
- **Functions called**:
    - [`minja::Parser::consumeSpaces`](#ParserconsumeSpaces)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::parseExpression`](#ParserparseExpression)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseIdentifier<!-- {{#callable:minja::Parser::parseIdentifier}} -->
Parses an identifier from the input string.
- **Inputs**:
    - `none`: This function does not take any input arguments.
- **Control Flow**:
    - Defines a static regex pattern to match valid identifiers, excluding certain keywords.
    - Retrieves the current location in the input string.
    - Attempts to consume a token that matches the identifier regex.
    - If no identifier is found, returns nullptr.
    - If an identifier is found, creates and returns a shared pointer to a new `VariableExpr` object initialized with the location and identifier.
- **Output**: Returns a shared pointer to a `VariableExpr` object representing the parsed identifier, or nullptr if no valid identifier was found.
- **Functions called**:
    - [`minja::Parser::get_location`](#Parserget_location)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseStringConcat<!-- {{#callable:minja::Parser::parseStringConcat}} -->
Parses a string concatenation expression, combining two expressions with a string concatenation operator.
- **Inputs**:
    - `left`: The left-hand side expression parsed from a mathematical power expression.
    - `right`: The right-hand side expression parsed from a logical AND expression, if a concatenation operator is found.
- **Control Flow**:
    - Calls `parseMathPow()` to parse the left-hand side expression.
    - Checks if the left expression is valid; if not, throws a runtime error.
    - Uses a regex to check for the string concatenation operator '~'.
    - If the operator is found, calls `parseLogicalAnd()` to parse the right-hand side expression.
    - Checks if the right expression is valid; if not, throws a runtime error.
    - Creates a new `BinaryOpExpr` combining the left and right expressions with the string concatenation operator.
- **Output**: Returns a shared pointer to an `Expression` representing the concatenated string expression, or the left expression if no concatenation operator is found.
- **Functions called**:
    - [`minja::Parser::parseMathPow`](#ParserparseMathPow)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::parseLogicalAnd`](#ParserparseLogicalAnd)
    - [`minja::Parser::get_location`](#Parserget_location)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseMathPow<!-- {{#callable:minja::Parser::parseMathPow}} -->
Parses mathematical power expressions in the form of 'base ** exponent'.
- **Inputs**:
    - `none`: This function does not take any input parameters directly.
- **Control Flow**:
    - Calls `parseMathPlusMinus()` to parse the left operand of the power expression.
    - If the left operand is not valid, it throws a runtime error indicating the expected left side.
    - Enters a loop to check for the '**' token indicating a power operation.
    - For each '**' token found, it calls `parseMathPlusMinus()` again to parse the right operand.
    - If the right operand is not valid, it throws a runtime error indicating the expected right side.
    - Creates a new `BinaryOpExpr` object representing the power operation with the left and right operands.
- **Output**: Returns a shared pointer to an `Expression` representing the parsed power expression.
- **Functions called**:
    - [`minja::Parser::parseMathPlusMinus`](#ParserparseMathPlusMinus)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::get_location`](#Parserget_location)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseMathPlusMinus<!-- {{#callable:minja::Parser::parseMathPlusMinus}} -->
Parses mathematical expressions involving addition and subtraction.
- **Inputs**:
    - `none`: This function does not take any input parameters.
- **Control Flow**:
    - Calls `parseMathMulDiv()` to parse the left operand of the expression.
    - If the left operand is not valid, throws a runtime error indicating the expected left side.
    - Enters a loop to consume tokens that match the plus or minus regex pattern.
    - For each operator token consumed, calls `parseMathMulDiv()` to parse the right operand.
    - If the right operand is not valid, throws a runtime error indicating the expected right side.
    - Creates a new `BinaryOpExpr` object with the left and right operands and the operator, updating the left operand for the next iteration.
- **Output**: Returns a shared pointer to an `Expression` representing the parsed mathematical expression.
- **Functions called**:
    - [`minja::Parser::parseMathMulDiv`](#ParserparseMathMulDiv)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::get_location`](#Parserget_location)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseMathMulDiv<!-- {{#callable:minja::Parser::parseMathMulDiv}} -->
Parses mathematical multiplication and division expressions.
- **Inputs**:
    - `none`: This function does not take any explicit input arguments.
- **Control Flow**:
    - Calls `parseMathUnaryPlusMinus()` to parse the left operand of the multiplication/division expression.
    - Checks if the left operand is valid; if not, throws a runtime error.
    - Uses a regex to identify multiplication/division operators (e.g., '*', '**', '/', '//', '%').
    - Enters a loop to consume tokens matching the multiplication/division operators.
    - For each operator found, calls `parseMathUnaryPlusMinus()` again to parse the right operand.
    - Checks if the right operand is valid; if not, throws a runtime error.
    - Creates a `BinaryOpExpr` object to represent the operation and updates the left operand.
    - After processing all operators, checks for a filter token ('|') to handle additional expressions.
    - If a filter token is found, recursively calls `parseMathMulDiv()` to parse the subsequent expression.
    - If the subsequent expression is a `FilterExpr`, prepends the left operand to it; otherwise, creates a new `FilterExpr`.
- **Output**: Returns a shared pointer to an `Expression` representing the parsed multiplication/division expression.
- **Functions called**:
    - [`minja::Parser::parseMathUnaryPlusMinus`](#ParserparseMathUnaryPlusMinus)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::get_location`](#Parserget_location)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::call\_func<!-- {{#callable:minja::Parser::call_func}} -->
Creates a `CallExpr` object representing a function call with the specified name and arguments.
- **Inputs**:
    - `name`: A `std::string` representing the name of the function to be called.
    - `args`: An `ArgumentsExpression` object containing the arguments to be passed to the function.
- **Control Flow**:
    - The function begins by calling `get_location()` to retrieve the current location in the source code.
    - It then creates a `VariableExpr` object using the provided `name` and the location.
    - Finally, it constructs a `CallExpr` object using the location, the created `VariableExpr`, and the provided `args`, and returns it.
- **Output**: Returns a `std::shared_ptr<Expression>` pointing to the newly created `CallExpr` object.
- **Functions called**:
    - [`minja::Parser::get_location`](#Parserget_location)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseMathUnaryPlusMinus<!-- {{#callable:minja::Parser::parseMathUnaryPlusMinus}} -->
Parses a unary plus or minus operator followed by an expression.
- **Inputs**:
    - `none`: This function does not take any input parameters.
- **Control Flow**:
    - A static regex pattern is defined to match unary plus or minus tokens.
    - The function attempts to consume a token matching the unary plus or minus pattern.
    - It then calls `parseExpansion()` to parse the subsequent expression.
    - If the expression is not valid, a runtime error is thrown.
    - If a valid unary operator is found, a `UnaryOpExpr` object is created with the parsed expression and the operator.
    - If no unary operator is found, the parsed expression is returned directly.
- **Output**: Returns a shared pointer to an `Expression` object representing the unary operation or the parsed expression if no unary operator is present.
- **Functions called**:
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::parseExpansion`](#ParserparseExpansion)
    - [`minja::Parser::get_location`](#Parserget_location)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseExpansion<!-- {{#callable:minja::Parser::parseExpansion}} -->
Parses an expansion expression, which may include an operator and a value expression.
- **Inputs**:
    - `op_str`: A string representing the operator for the expansion, which can be either '*' or '**'.
    - `expr`: A shared pointer to an `Expression` that represents the value expression being expanded.
- **Control Flow**:
    - A static regex `expansion_tok` is defined to match the expansion operator.
    - The function calls [`consumeToken`](#ParserconsumeToken) with the regex to get the operator string.
    - It then calls [`parseValueExpression`](#ParserparseValueExpression) to parse the value expression.
    - If the operator string is empty, it returns the parsed expression.
    - If the expression is null, it throws a runtime error indicating an expected expression.
    - Finally, it creates and returns a shared pointer to a `UnaryOpExpr` with the parsed expression and the appropriate operator.
- **Output**: Returns a shared pointer to a `UnaryOpExpr` representing the expansion operation, or throws an error if the expression is invalid.
- **Functions called**:
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::parseValueExpression`](#ParserparseValueExpression)
    - [`minja::Parser::get_location`](#Parserget_location)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseValueExpression<!-- {{#callable:minja::Parser::parseValueExpression}} -->
Parses a value expression from a template string.
- **Inputs**: None
- **Control Flow**:
    - Defines a lambda function `parseValue` to handle the parsing of different value types.
    - Attempts to parse a constant value, returning a `LiteralExpr` if successful.
    - Checks for a 'null' token and returns a `LiteralExpr` with a null value if found.
    - Tries to parse an identifier, returning it if successful.
    - Attempts to parse a braced expression or array, returning it if successful.
    - Attempts to parse an array, returning it if successful.
    - Attempts to parse a dictionary, returning it if successful.
    - Throws a runtime error if no valid value expression is found.
    - Processes additional subscript or method call expressions if present.
    - Handles slicing and indexing for arrays and dictionaries.
    - Returns the final parsed value expression.
- **Output**: Returns a shared pointer to an `Expression` representing the parsed value expression.
- **Functions called**:
    - [`minja::Parser::get_location`](#Parserget_location)
    - [`minja::Parser::parseConstant`](#ParserparseConstant)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Value::Value`](#ValueValue)
    - [`minja::Parser::parseIdentifier`](#ParserparseIdentifier)
    - [`minja::Parser::parseBracedExpressionOrArray`](#ParserparseBracedExpressionOrArray)
    - [`minja::Parser::parseArray`](#ParserparseArray)
    - [`minja::Parser::parseDictionary`](#ParserparseDictionary)
    - [`minja::Parser::consumeSpaces`](#ParserconsumeSpaces)
    - [`minja::Parser::peekSymbols`](#ParserpeekSymbols)
    - [`minja::Parser::parseExpression`](#ParserparseExpression)
    - [`minja::Parser::parseCallArgs`](#ParserparseCallArgs)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseBracedExpressionOrArray<!-- {{#callable:minja::Parser::parseBracedExpressionOrArray}} -->
Parses a braced expression or an array from the input.
- **Inputs**:
    - `none`: No input arguments are required for this function.
- **Control Flow**:
    - The function starts by checking for an opening parenthesis '(' and returns nullptr if not found.
    - It then attempts to parse an expression inside the parentheses.
    - If the expression is not valid, it throws a runtime error.
    - If a closing parenthesis ')' is found immediately after the expression, it returns the parsed expression.
    - If a closing parenthesis is not found, it initializes a vector to hold multiple expressions.
    - It enters a loop to parse additional expressions separated by commas, throwing an error if a comma is expected but not found.
    - If a closing parenthesis is found after parsing additional expressions, it returns an ArrayExpr containing the parsed expressions.
    - If the loop ends without finding a closing parenthesis, it throws a runtime error.
- **Output**: Returns a shared pointer to an `Expression` representing the parsed braced expression or an `ArrayExpr` if multiple expressions were parsed.
- **Functions called**:
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::parseExpression`](#ParserparseExpression)
    - [`minja::Parser::get_location`](#Parserget_location)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseArray<!-- {{#callable:minja::Parser::parseArray}} -->
Parses an array expression from the input string.
- **Inputs**:
    - `none`: No input arguments are required.
- **Control Flow**:
    - Checks for the opening bracket '[' and returns nullptr if not found.
    - Initializes an empty vector to hold the elements of the array.
    - If a closing bracket ']' is found immediately after the opening bracket, returns an empty array.
    - Parses the first expression and adds it to the elements vector.
    - Enters a loop to parse additional expressions separated by commas.
    - If a comma is found, it attempts to parse the next expression and adds it to the elements vector.
    - If a closing bracket ']' is found, it returns the constructed array.
    - Throws an error if neither a comma nor a closing bracket is found.
    - Throws an error if the closing bracket is not found by the end of the input.
- **Output**: Returns a shared pointer to an `ArrayExpr` containing the parsed elements, or throws an error if the input is malformed.
- **Functions called**:
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::get_location`](#Parserget_location)
    - [`minja::Parser::parseExpression`](#ParserparseExpression)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseDictionary<!-- {{#callable:minja::Parser::parseDictionary}} -->
Parses a dictionary from a string representation, returning a `DictExpr` object.
- **Inputs**:
    - `none`: No input arguments are required.
- **Control Flow**:
    - Checks for the opening brace '{' and returns nullptr if not found.
    - Initializes an empty vector to hold key-value pairs.
    - If a closing brace '}' is found immediately after '{', returns an empty dictionary.
    - Defines a lambda function to parse key-value pairs.
    - Calls the lambda to parse the first key-value pair.
    - Enters a loop to parse additional key-value pairs until the end of the input is reached.
    - Handles commas and closing braces appropriately, throwing errors for unexpected tokens.
- **Output**: Returns a shared pointer to a `DictExpr` object containing the parsed key-value pairs.
- **Functions called**:
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::get_location`](#Parserget_location)
    - [`minja::Parser::parseExpression`](#ParserparseExpression)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parsePreSpace<!-- {{#callable:minja::Parser::parsePreSpace}} -->
The `parsePreSpace` function determines how to handle whitespace based on the input string.
- **Inputs**:
    - `s`: A constant reference to a string that is checked to determine the whitespace handling.
- **Control Flow**:
    - The function checks if the input string `s` is equal to '-' to decide the whitespace handling.
    - If `s` is '-', it returns `SpaceHandling::Strip`, otherwise it returns `SpaceHandling::Keep`.
- **Output**: Returns a value of type `SpaceHandling` indicating whether to strip whitespace or keep it.
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parsePostSpace<!-- {{#callable:minja::Parser::parsePostSpace}} -->
The `parsePostSpace` function determines how to handle whitespace after a specific string.
- **Inputs**:
    - `s`: A constant reference to a string that is evaluated to determine the whitespace handling.
- **Control Flow**:
    - The function checks if the input string `s` is equal to '-' using a simple equality comparison.
    - If the condition is true, it returns `SpaceHandling::Strip`, indicating that whitespace should be stripped.
    - If the condition is false, it returns `SpaceHandling::Keep`, indicating that whitespace should be preserved.
- **Output**: The function returns a value of type `SpaceHandling`, which indicates whether to strip or keep whitespace.
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseVarNames<!-- {{#callable:minja::Parser::parseVarNames}} -->
Parses variable names from a string and returns them as a vector.
- **Inputs**: None
- **Control Flow**:
    - A static regex `varnames_regex` is defined to match variable names in a specific format.
    - The function calls [`consumeTokenGroups`](#ParserconsumeTokenGroups) with the regex to extract potential variable names.
    - If no variable names are found, a runtime error is thrown.
    - The matched variable names are processed by splitting them on commas and stripping whitespace.
    - Each cleaned variable name is added to the `varnames` vector.
    - Finally, the `varnames` vector is returned.
- **Output**: Returns a vector of strings containing the parsed variable names.
- **Functions called**:
    - [`minja::Parser::consumeTokenGroups`](#ParserconsumeTokenGroups)
    - [`minja::strip`](#minjastrip)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::unexpected<!-- {{#callable:minja::Parser::unexpected}} -->
Generates a `std::runtime_error` indicating an unexpected `TemplateToken`.
- **Inputs**:
    - `token`: A constant reference to a `TemplateToken` object that contains information about the unexpected token.
- **Control Flow**:
    - The function constructs an error message by concatenating the string 'Unexpected ' with the string representation of the token type obtained from `TemplateToken::typeToString(token.type)`.
    - It appends additional information about the location of the token in the template string using the [`error_location_suffix`](#minjaerror_location_suffix) function.
- **Output**: Returns a `std::runtime_error` object initialized with the constructed error message.
- **Functions called**:
    - [`minja::error_location_suffix`](#minjaerror_location_suffix)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::unterminated<!-- {{#callable:minja::Parser::unterminated}} -->
The `unterminated` function generates a runtime error message indicating that a template token is not properly terminated.
- **Inputs**:
    - `token`: A constant reference to a `TemplateToken` object that contains information about the token type and its location.
- **Control Flow**:
    - The function constructs a runtime error message by concatenating the string 'Unterminated ' with the type of the provided `token` using `TemplateToken::typeToString`.
    - It appends additional information about the location of the error by calling [`error_location_suffix`](#minjaerror_location_suffix), which uses the `template_str` and the position of the token.
- **Output**: The function returns a `std::runtime_error` object containing the constructed error message.
- **Functions called**:
    - [`minja::error_location_suffix`](#minjaerror_location_suffix)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::tokenize<!-- {{#callable:minja::Parser::tokenize}} -->
The `tokenize` function processes a template string and generates a vector of tokens representing comments, expressions, and control structures.
- **Inputs**: None
- **Control Flow**:
    - The function initializes several static regex patterns to identify different types of tokens in the template string.
    - It enters a loop that continues until the end of the template string is reached.
    - Within the loop, it attempts to match the current position in the string against the defined regex patterns.
    - If a comment token is found, it extracts the content and creates a `CommentTemplateToken`.
    - If an expression opening token is found, it parses the expression and creates an `ExpressionTemplateToken`.
    - If a block opening token is found, it identifies the block type and processes it accordingly, creating the appropriate block token.
    - If no tokens match, it treats the remaining text as a `TextTemplateToken`.
    - The function handles exceptions by throwing runtime errors with context about the error location.
- **Output**: The function returns a vector of unique pointers to `TemplateToken` objects, representing the parsed tokens from the template string.
- **Functions called**:
    - [`minja::Parser::get_location`](#Parserget_location)
    - [`minja::Parser::consumeTokenGroups`](#ParserconsumeTokenGroups)
    - [`minja::Parser::parsePreSpace`](#ParserparsePreSpace)
    - [`minja::Parser::parsePostSpace`](#ParserparsePostSpace)
    - [`minja::Parser::parseExpression`](#ParserparseExpression)
    - [`minja::Parser::consumeToken`](#ParserconsumeToken)
    - [`minja::Parser::parseVarNames`](#ParserparseVarNames)
    - [`minja::Parser::parseIdentifier`](#ParserparseIdentifier)
    - [`minja::Parser::parseParameters`](#ParserparseParameters)
    - [`minja::error_location_suffix`](#minjaerror_location_suffix)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parseTemplate<!-- {{#callable:minja::Parser::parseTemplate}} -->
Parses a sequence of template tokens into a structured representation of template nodes.
- **Inputs**:
    - `begin`: An iterator pointing to the start of the template tokens.
    - `it`: An iterator that will be advanced through the template tokens.
    - `end`: An iterator pointing to the end of the template tokens.
    - `fully`: A boolean flag indicating whether to check for unexpected tokens at the end.
- **Control Flow**:
    - Iterates through the tokens until the end is reached.
    - For each token, checks its type and processes it accordingly (e.g., If, For, Text, Expression).
    - Handles nested structures for If and For tokens, including Else and Elif branches.
    - Throws exceptions for unterminated blocks or unexpected tokens.
    - Collects parsed nodes into a vector and returns a single node or a sequence node based on the number of children.
- **Output**: Returns a shared pointer to a `TemplateNode`, representing the parsed structure of the template.
- **Functions called**:
    - [`minja::Parser::unterminated`](#Parserunterminated)
    - [`minja::Parser::unexpected`](#Parserunexpected)
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)


---
#### Parser::parse<!-- {{#callable:minja::Parser::parse}} -->
Parses a template string into a `TemplateNode` structure using specified options.
- **Inputs**:
    - `template_str`: A string representing the template to be parsed.
    - `options`: An `Options` structure that contains parsing options such as trimming blocks and handling newlines.
- **Control Flow**:
    - Creates a `Parser` instance initialized with the normalized template string and options.
    - Calls the `tokenize` method of the `Parser` to generate a list of tokens from the template string.
    - Defines iterators for the beginning and end of the token list.
    - Invokes the `parseTemplate` method of the `Parser` to construct a `TemplateNode` from the tokens, passing the iterators.
    - Returns the constructed `TemplateNode`.
- **Output**: Returns a shared pointer to a `TemplateNode` that represents the parsed structure of the template.
- **See also**: [`minja::Parser`](#minjaParser)  (Data Structure)



---
### expression\_parsing\_error<!-- {{#data_structure:minja::Parser::expression_parsing_error}} -->
- **Type**: `class`
- **Members**:
    - `it`: An iterator pointing to the location of the parsing error.
- **Description**: The `expression_parsing_error` class is a custom exception derived from `std::runtime_error`, designed to represent errors encountered during the parsing of expressions, encapsulating an error message and the specific character position in the input where the error occurred.
- **Member Functions**:
    - [`minja::Parser::expression_parsing_error::expression_parsing_error`](#expression_parsing_errorexpression_parsing_error)
    - [`minja::Parser::expression_parsing_error::get_pos`](#expression_parsing_errorget_pos)
- **Inherits From**:
    - `std::runtime_error`

**Methods**

---
#### expression\_parsing\_error::expression\_parsing\_error<!-- {{#callable:minja::Parser::expression_parsing_error::expression_parsing_error}} -->
The `expression_parsing_error` class is a custom exception that captures an error message and the position in the input where the error occurred.
- **Inputs**:
    - `message`: A string containing the error message that describes the parsing error.
    - `it`: A `CharIterator` that points to the location in the input where the error occurred.
- **Control Flow**:
    - The constructor initializes the base class `std::runtime_error` with the provided error message.
    - It also stores the `CharIterator` to track the position of the error in the input.
- **Output**: The constructor does not return a value but initializes an instance of `expression_parsing_error` that can be thrown as an exception.
- **See also**: [`minja::Parser::expression_parsing_error`](#minjaParser::expression_parsing_error)  (Data Structure)


---
#### expression\_parsing\_error::get\_pos<!-- {{#callable:minja::Parser::expression_parsing_error::get_pos}} -->
Calculates the position of an iterator relative to a given starting iterator.
- **Inputs**:
    - `begin`: A `CharIterator` representing the starting point from which the distance to the current iterator is calculated.
- **Control Flow**:
    - The function uses `std::distance` to compute the number of elements between the `begin` iterator and the member iterator `it`.
    - It returns the computed distance as a size_t value.
- **Output**: Returns a `size_t` representing the number of elements between the `begin` iterator and the current iterator `it`.
- **See also**: [`minja::Parser::expression_parsing_error`](#minjaParser::expression_parsing_error)  (Data Structure)



# Functions

---
### normalize\_newlines<!-- {{#callable:minja::normalize_newlines}} -->
The `normalize_newlines` function converts Windows-style newlines (\r\n) to Unix-style newlines (\n) in a given string on Windows systems.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that represents the input string whose newlines are to be normalized.
- **Control Flow**:
    - Check if the code is being compiled on a Windows system using the `_WIN32` preprocessor directive.
    - If on Windows, define a static `std::regex` to match Windows-style newlines (`\r\n`).
    - Use `std::regex_replace` to replace all occurrences of `\r\n` with `\n` in the input string `s`.
    - Return the modified string with Unix-style newlines.
    - If not on Windows, simply return the input string `s` unchanged.
- **Output**: A `std::string` with all Windows-style newlines replaced by Unix-style newlines on Windows systems, or the original string on other systems.


---
### operator\(\)<!-- {{#callable:std::operator()}} -->
The `operator()` function is a hash function for `minja::Value` objects that ensures the value is hashable and then returns its hash.
- **Inputs**:
    - `v`: A constant reference to a `minja::Value` object that is to be hashed.
- **Control Flow**:
    - Check if the input `minja::Value` object `v` is hashable using `v.is_hashable()`.
    - If `v` is not hashable, throw a `std::runtime_error` with a message indicating the type is unsupported for hashing.
    - If `v` is hashable, convert it to a JSON object using `v.get<json>()`.
    - Compute and return the hash of the JSON object using `std::hash<json>()`.
- **Output**: Returns a `size_t` value representing the hash of the input `minja::Value` object.


---
### error\_location\_suffix<!-- {{#callable:minja::error_location_suffix}} -->
The `error_location_suffix` function generates a string indicating the line and column of a specified position in a source string, along with the surrounding lines for context.
- **Inputs**:
    - `source`: A constant reference to a `std::string` representing the source text in which the error location is to be identified.
    - `pos`: A `size_t` representing the position in the source string for which the error location is to be determined.
- **Control Flow**:
    - Define a lambda function `get_line` to extract a specific line from the source string.
    - Initialize iterators `start` and `end` to the beginning and end of the source string, respectively.
    - Calculate the line number by counting newline characters from the start to the position `pos`.
    - Calculate the total number of lines in the source string.
    - Determine the column number by finding the position of the last newline before `pos`.
    - Create an output stream `out` to build the error location string.
    - Append the line and column information to the output stream.
    - If the error is not on the first line, append the previous line to the output stream.
    - Append the current line and a caret (`^`) indicating the error position to the output stream.
    - If the error is not on the last line, append the next line to the output stream.
- **Output**: Returns a `std::string` containing the error location information, including the line and column numbers, and the relevant lines of text with a caret pointing to the error position.


---
### destructuring\_assign<!-- {{#callable:minja::destructuring_assign}} -->
The `destructuring_assign` function assigns values from a `Value` object to variables in a context, supporting both single and multiple variable assignments.
- **Inputs**:
    - `var_names`: A vector of strings representing the names of the variables to which values will be assigned.
    - `context`: A shared pointer to a `Context` object where the variables will be set.
    - `item`: A `Value` object containing the data to be assigned to the variables.
- **Control Flow**:
    - Check if `var_names` contains only one element.
    - If true, create a `Value` object with the single variable name and set it in the context with the `item`.
    - If `var_names` contains more than one element, check if `item` is an array and its size matches `var_names`.
    - If the check fails, throw a runtime error indicating a mismatch between variables and items.
    - If the check passes, iterate over `var_names` and assign each corresponding element from `item` to the context.
- **Output**: The function does not return a value; it modifies the `context` by setting variables.


---
### strip<!-- {{#callable:minja::strip}} -->
The `strip` function removes specified leading and trailing characters from a string.
- **Inputs**:
    - `s`: The input string from which characters are to be stripped.
    - `chars`: A string containing characters to be removed from the input string; defaults to whitespace characters if not provided.
    - `left`: A boolean indicating whether to strip characters from the left side of the string; defaults to true.
    - `right`: A boolean indicating whether to strip characters from the right side of the string; defaults to true.
- **Control Flow**:
    - Determine the set of characters to strip, defaulting to whitespace if `chars` is empty.
    - Find the first character not in the charset from the left if `left` is true, otherwise start from the beginning.
    - Return an empty string if no such character is found (i.e., the string is entirely composed of characters to be stripped).
    - Find the last character not in the charset from the right if `right` is true, otherwise use the last character of the string.
    - Return the substring from the first non-stripped character to the last non-stripped character.
- **Output**: A new string with the specified characters removed from the beginning and/or end.


---
### split<!-- {{#callable:minja::split}} -->
The `split` function divides a given string into a vector of substrings based on a specified separator.
- **Inputs**:
    - `s`: A constant reference to the input string that needs to be split.
    - `sep`: A constant reference to the separator string used to determine where to split the input string.
- **Control Flow**:
    - Initialize an empty vector `result` to store the substrings.
    - Set `start` to 0 and find the first occurrence of `sep` in `s`, storing the position in `end`.
    - Enter a loop that continues as long as `end` is not `std::string::npos`.
    - Within the loop, extract the substring from `start` to `end` and add it to `result`.
    - Update `start` to the position after the current `sep` and find the next occurrence of `sep` starting from `start`.
    - After the loop, add the remaining part of the string (from `start` to the end of `s`) to `result`.
- **Output**: A vector of strings, where each element is a substring of the input string `s` split by the separator `sep`.


---
### capitalize<!-- {{#callable:minja::capitalize}} -->
The `capitalize` function converts the first character of a given string to uppercase.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that represents the input string to be capitalized.
- **Control Flow**:
    - Check if the input string `s` is empty; if so, return it as is.
    - Create a copy of the input string `s` named `result`.
    - Convert the first character of `result` to uppercase using `std::toupper`.
    - Return the modified `result` string.
- **Output**: A `std::string` with the first character capitalized, or the original string if it was empty.


---
### html\_escape<!-- {{#callable:minja::html_escape}} -->
The `html_escape` function converts special HTML characters in a string to their corresponding HTML entities.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that represents the input string to be escaped.
- **Control Flow**:
    - Initialize an empty `std::string` named `result` and reserve space equal to the size of the input string `s`.
    - Iterate over each character `c` in the input string `s`.
    - For each character, use a `switch` statement to check if it is a special HTML character (`&`, `<`, `>`, `"`, or `'`).
    - If the character is a special HTML character, append its corresponding HTML entity to `result`.
    - If the character is not a special HTML character, append the character itself to `result`.
    - After processing all characters, return the `result` string.
- **Output**: A `std::string` containing the escaped version of the input string, where special HTML characters are replaced with their corresponding HTML entities.


---
### simple\_function<!-- {{#callable:minja::simple_function}} -->
The `simple_function` creates a callable `Value` that wraps a function with named and positional arguments, ensuring correct argument mapping and validation.
- **Inputs**:
    - `fn_name`: A string representing the name of the function for error messages.
    - `params`: A vector of strings representing the names of the expected parameters for the function.
    - `fn`: A function object that takes a shared pointer to a `Context` and a `Value` representing the arguments, and returns a `Value`.
- **Control Flow**:
    - Initialize a map `named_positions` to store the index of each parameter name in `params`.
    - Iterate over `params` to populate `named_positions` with each parameter's index.
    - Return a callable `Value` that captures the `fn_name`, `params`, and `fn`.
    - Inside the callable, create an empty `Value` object `args_obj` to hold the arguments.
    - Create a vector `provided_args` to track which arguments have been provided.
    - Iterate over positional arguments in `args.args`, mapping them to `params` and setting them in `args_obj`.
    - Throw a runtime error if there are more positional arguments than expected.
    - Iterate over keyword arguments in `args.kwargs`, checking if each name exists in `named_positions`.
    - Throw a runtime error if a keyword argument name is not recognized.
    - Set the keyword arguments in `args_obj` and mark them as provided in `provided_args`.
    - Call the provided function `fn` with the `context` and `args_obj`, returning its result.
- **Output**: A `Value` object that is callable, wrapping the provided function with argument handling logic.


---
### Expression<!-- {{#callable:minja::Expression::Expression}} -->
The `Expression` constructor initializes an `Expression` object with a given `Location`.
- **Inputs**:
    - `location`: A `Location` object that specifies the source and position of the expression.
- **Control Flow**:
    - The constructor takes a `Location` object as an argument.
    - It initializes the `location` member variable of the `Expression` class with the provided `Location` object.
- **Output**: An `Expression` object is created and initialized with the specified location.


---
### TemplateToken<!-- {{#callable:minja::TemplateToken::TemplateToken}} -->
The `TemplateToken` constructor initializes a `TemplateToken` object with a specified type, location, and space handling options for pre and post spaces.
- **Inputs**:
    - `type`: An enumeration value of type `TemplateToken::Type` that specifies the type of the template token.
    - `location`: A `Location` object that indicates the source and position of the token within the template.
    - `pre`: A `SpaceHandling` enumeration value that specifies how spaces before the token should be handled.
    - `post`: A `SpaceHandling` enumeration value that specifies how spaces after the token should be handled.
- **Control Flow**:
    - The constructor initializes the `type` member with the provided `type` argument.
    - The `location` member is initialized with the provided `location` argument.
    - The `pre_space` member is set to the provided `pre` argument.
    - The `post_space` member is set to the provided `post` argument.
- **Output**: The function does not return any value as it is a constructor.


---
### TemplateNode<!-- {{#callable:minja::TemplateNode::TemplateNode}} -->
The `TemplateNode` class constructor initializes a template node with a given location, and its `render` method outputs the rendered content of the node using a provided context.
- **Inputs**:
    - `location`: A `Location` object representing the source and position of the template node.
    - `out`: A reference to an `std::ostringstream` object where the rendered output will be written.
    - `context`: A shared pointer to a `Context` object that provides the necessary context for rendering the template node.
- **Control Flow**:
    - The constructor initializes the `location_` member variable with the provided `location` argument.
    - The `render` method calls the `do_render` method, which is a pure virtual function meant to be implemented by derived classes, passing `out` and `context` as arguments.
    - The `render` method catches any `LoopControlException` or `std::exception` thrown during rendering, appends error location information if available, and rethrows the exception.
- **Output**: The `render` method outputs the rendered content to the `out` stream, and the constructor does not produce any output.


