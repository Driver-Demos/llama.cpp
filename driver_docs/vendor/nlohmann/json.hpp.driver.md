# Purpose
The provided C++ code is part of the "JSON for Modern C++" library, version 3.12.0, authored by Niels Lohmann, designed to facilitate comprehensive JSON handling in C++ applications. This header-only library offers a robust framework for parsing, serializing, and manipulating JSON data, supporting both DOM and SAX style parsing, and handling various binary formats like CBOR, MessagePack, UBJSON, BJData, and BSON. The core component, the `basic_json` class template, provides extensive functionalities for JSON operations, including type-safe handling of JSON data types, JSON Pointer and Patch operations, and binary serialization/deserialization. The code includes utility functions, type definitions, and custom exception handling to ensure efficient and flexible JSON processing, while maintaining compatibility across different C++ standards. Designed for easy integration, the library provides a public API with modern C++ features, making it a valuable tool for developers working with JSON data in diverse applications.
# Imports and Dependencies

---
- `algorithm`
- `cstddef`
- `functional`
- `initializer_list`
- `iosfwd`
- `iterator`
- `memory`
- `string`
- `utility`
- `vector`
- `array`
- `forward_list`
- `map`
- `optional`
- `tuple`
- `type_traits`
- `unordered_map`
- `valarray`
- `exception`
- `numeric`
- `stdexcept`
- `cstdint`
- `stdint.h`
- `version`
- `cstdlib`
- `cassert`
- `compare`
- `limits`
- `cstring`
- `experimental/filesystem`
- `filesystem`
- `ranges`
- `cmath`
- `cstdio`
- `bit`
- `istream`
- `clocale`
- `cctype`
- `cerrno`
- `ios`
- `ostream`
- `iomanip`
- `any`
- `string_view`


# Data Structures

---
### make\_void<!-- {{#data_structure:namespace::make_void}} -->
- **Type**: `struct`
- **Members**:
    - `type`: Defines a type alias for `void`.
- **Description**: The `make_void` struct is a template that provides a way to create a type alias for `void`, which can be useful in template metaprogramming to simplify type traits and enable SFINAE (Substitution Failure Is Not An Error) techniques.


---
### nonesuch<!-- {{#data_structure:NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch}} -->
- **Type**: `struct`
- **Members**:
    - `constructor`: The constructor is deleted to prevent instantiation.
    - `destructor`: The destructor is deleted to prevent destruction.
    - `copy constructor`: The copy constructor is deleted to prevent copying.
    - `move constructor`: The move constructor is deleted to prevent moving.
    - `copy assignment operator`: The copy assignment operator is deleted to prevent assignment.
    - `move assignment operator`: The move assignment operator is deleted to prevent assignment.
- **Description**: The `nonesuch` struct is a utility type designed to be non-instantiable and non-copyable, effectively serving as a placeholder or marker type in template metaprogramming, ensuring that certain types cannot be created or manipulated.
- **Member Functions**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch::nonesuch`](#nonesuchnonesuch)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch::~nonesuch`](#nonesuchnonesuch)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch::nonesuch`](#nonesuchnonesuch)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch::nonesuch`](#nonesuchnonesuch)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch::operator=`](#nonesuchoperator=)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch::operator=`](#nonesuchoperator=)

**Methods**

---
#### nonesuch::\~nonesuch<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch::~nonesuch}} -->
The `nonesuch` class is a deleted type that prevents instantiation and copying.
- **Inputs**: None
- **Control Flow**:
    - The constructor `nonesuch()` is deleted, preventing object creation.
    - The destructor `~nonesuch()` is deleted, ensuring no destruction can occur.
    - The copy constructor `nonesuch(nonesuch const&)` is deleted, preventing copying of objects.
    - The move constructor `nonesuch(nonesuch const&&)` is deleted, preventing moving of objects.
    - The copy assignment operator `operator=(nonesuch const&)` is deleted, preventing assignment from one object to another.
    - The move assignment operator `operator=(nonesuch&&)` is deleted, preventing assignment from a temporary object.
- **Output**: The `nonesuch` class does not produce any output as it cannot be instantiated or used in any way.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch`](#NLOHMANN_JSON_NAMESPACE_BEGINnonesuch)  (Data Structure)


---
#### nonesuch::nonesuch<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch::nonesuch}} -->
The `nonesuch` struct is designed to be a non-instantiable type by deleting its constructors and assignment operators.
- **Inputs**: None
- **Control Flow**:
    - The `nonesuch` struct has a default constructor that is deleted, preventing instantiation.
    - The destructor is also deleted, ensuring that no instances can be destructed.
    - Copy constructor and move constructor are both deleted, preventing copying or moving of `nonesuch` instances.
    - Copy assignment operator and move assignment operator are deleted, preventing assignment of `nonesuch` instances.
- **Output**: The `nonesuch` struct does not produce any output as it cannot be instantiated or used in any meaningful way.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch`](#NLOHMANN_JSON_NAMESPACE_BEGINnonesuch)  (Data Structure)


---
#### nonesuch::nonesuch<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch::nonesuch}} -->
The `nonesuch` struct is designed to prevent instantiation and copying.
- **Inputs**:
    - `nonesuch const&&`: A rvalue reference to a `nonesuch` object, which is deleted to prevent moving.
- **Control Flow**:
    - The constructor `nonesuch()` is deleted, preventing any instantiation of the struct.
    - The destructor `~nonesuch()` is deleted, ensuring that no instances can be destructed.
    - The copy constructor `nonesuch(nonesuch const&)` is deleted, preventing copying of `nonesuch` objects.
    - The move constructor `nonesuch(nonesuch const&&)` is deleted, preventing moving of `nonesuch` objects.
    - The copy assignment operator `operator=(nonesuch const&)` is deleted, preventing assignment from one `nonesuch` object to another.
    - The move assignment operator `operator=(nonesuch&&)` is deleted, preventing assignment from a temporary `nonesuch` object.
- **Output**: The `nonesuch` struct does not produce any output as it cannot be instantiated or used in any way.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch`](#NLOHMANN_JSON_NAMESPACE_BEGINnonesuch)  (Data Structure)


---
#### nonesuch::operator=<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch::operator=}} -->
The `operator=` for the `nonesuch` struct is deleted to prevent assignment operations.
- **Inputs**:
    - `nonesuch const&`: A constant reference to a `nonesuch` object, which is not allowed for assignment.
    - `nonesuch&&`: An rvalue reference to a `nonesuch` object, which is also not allowed for assignment.
- **Control Flow**:
    - The function is marked as `delete`, which means it cannot be called or used.
    - Any attempt to use this operator will result in a compile-time error, effectively preventing assignment.
- **Output**: The function does not produce any output as it is deleted and cannot be invoked.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch`](#NLOHMANN_JSON_NAMESPACE_BEGINnonesuch)  (Data Structure)


---
#### nonesuch::operator=<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch::operator=}} -->
The `operator=` function is deleted to prevent assignment of `nonesuch` objects.
- **Inputs**:
    - `nonesuch&&`: An rvalue reference to a `nonesuch` object, which is not usable due to the deletion of this operator.
- **Control Flow**:
    - The function is marked as deleted, which means it cannot be called or used in any context.
    - Any attempt to use this operator will result in a compile-time error.
- **Output**: The function does not produce any output as it is deleted and cannot be invoked.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::nonesuch`](#NLOHMANN_JSON_NAMESPACE_BEGINnonesuch)  (Data Structure)



---
### detector<!-- {{#data_structure:NLOHMANN_JSON_NAMESPACE_BEGIN::detector}} -->
- **Type**: `struct`
- **Members**:
    - `value_t`: A type alias representing a false type.
    - `type`: A type alias for the default type provided as a template parameter.
- **Description**: The `detector` struct is a template-based utility designed to facilitate type detection in C++. It provides a default type and a value type that indicates the success or failure of a type trait operation, defaulting to `std::false_type`.


---
### is\_detected\_lazy<!-- {{#data_structure:NLOHMANN_JSON_NAMESPACE_BEGIN::is_detected_lazy}} -->
- **Type**: `struct`
- **Members**:
    - `Op`: A template parameter representing a callable type.
    - `Args`: A variadic template parameter pack representing the arguments to be passed to the callable type.
- **Description**: `is_detected_lazy` is a template struct that inherits from `is_detected`, allowing for lazy evaluation of whether a given operation can be applied to a set of arguments, enabling compile-time checks for the validity of the operation.
- **Inherits From**:
    - `is_detected`


---
### value\_t<!-- {{#data_structure:namespace::value_t}} -->
- **Type**: `enum class`
- **Members**:
    - `null`: Represents a null value.
    - `object`: Represents an object, which is an unordered set of name/value pairs.
    - `array`: Represents an array, which is an ordered collection of values.
    - `string`: Represents a string value.
    - `boolean`: Represents a boolean value.
    - `number_integer`: Represents a signed integer number value.
    - `number_unsigned`: Represents an unsigned integer number value.
    - `number_float`: Represents a floating-point number value.
    - `binary`: Represents a binary array, which is an ordered collection of bytes.
    - `discarded`: Indicates a value that has been discarded by the parser callback function.
- **Description**: `value_t` is an enumeration that defines a set of possible types for a value, including null, various data structures like objects and arrays, primitive types like strings and booleans, numeric types, and a binary type, as well as a special state for discarded values.


---
### position\_t<!-- {{#data_structure:namespace::position_t}} -->
- **Type**: `struct`
- **Members**:
    - `chars_read_total`: The total number of characters read.
    - `chars_read_current_line`: The number of characters read in the current line.
    - `lines_read`: The number of lines read.
- **Description**: The `position_t` structure is designed to track the reading progress of a text input, maintaining counts of total characters read, characters read in the current line, and the number of lines read, facilitating efficient management of reading operations.


---
### integer\_sequence<!-- {{#data_structure:namespace::integer_sequence}} -->
- **Type**: `struct`
- **Members**:
    - `value_type`: Defines the type of the values in the sequence.
    - `size`: A static constexpr function that returns the number of elements in the sequence.
- **Description**: The `integer_sequence` is a template struct that represents a compile-time sequence of integers, allowing for type-safe manipulation of integer values and providing a way to retrieve the size of the sequence.
- **Member Functions**:
    - [`namespace::integer_sequence::size`](#integer_sequencesize)

**Methods**

---
#### integer\_sequence::size<!-- {{#callable:namespace::integer_sequence::size}} -->
The `size` function returns the number of elements in the `integer_sequence`.
- **Inputs**:
    - `T`: The type of the elements in the sequence.
    - `Ints`: A parameter pack representing the integer values in the sequence.
- **Control Flow**:
    - The function uses the `sizeof...` operator to count the number of elements in the parameter pack `Ints`.
    - It returns this count as a `std::size_t` value.
- **Output**: The output is a `std::size_t` representing the number of integers in the `integer_sequence`.
- **See also**: [`namespace::integer_sequence`](#namespaceinteger_sequence)  (Data Structure)



---
### Gen<!-- {{#data_structure:namespace::utility_internal::Gen}} -->
- **Type**: `struct Gen`
- **Members**:
    - `type`: Defines a type that extends the `Gen` structure recursively based on the template parameters.
- **Description**: The `Gen` structure is a template-based recursive data structure that generates a type based on the provided type `T` and size `N`. It utilizes template metaprogramming to create a new type by recursively dividing `N` by 2 and applying an `Extend` type transformation, allowing for complex type manipulations at compile time.


---
### priority\_tag<!-- {{#data_structure:namespace::priority_tag}} -->
- **Type**: `struct`
- **Members**:
    - `N`: A template parameter that defines the priority level.
- **Description**: The `priority_tag` is a variadic template struct that creates a chain of inheritance based on the template parameter `N`, allowing for a compile-time representation of priority levels, where each specialization of `priority_tag<N>` inherits from `priority_tag<N-1>` until it reaches the base case of `priority_tag<0>`, which is an empty struct.
- **Inherits From**:
    - [`namespace::priority_tag`](#namespacepriority_tag)


---
### static\_const<!-- {{#data_structure:namespace::static_const}} -->
- **Type**: `struct`
- **Members**:
    - `value`: A static constant of type `T` initialized to a default value.
- **Description**: The `static_const` struct is a template that defines a static constant member `value` of type `T`, which is initialized to its default value at compile time, allowing for the creation of type-safe constant values.


---
### iterator\_types<!-- {{#data_structure:detail::iterator_types}} -->
- **Type**: `struct`
- **Description**: `iterator_types` is a template struct designed to define a type trait for iterators, allowing for specialization based on the iterator type provided as a template parameter.


---
### iterator\_traits<!-- {{#data_structure:detail::iterator_traits}} -->
- **Type**: `struct`
- **Members**:
    - `T`: The type of the iterator.
    - `void`: A default template parameter used for SFINAE (Substitution Failure Is Not An Error).
- **Description**: `iterator_traits` is a template struct designed to provide type traits for iterators, allowing for the extraction of properties and types associated with a given iterator type.


---
### is\_basic\_json<!-- {{#data_structure:namespace::is_basic_json}} -->
- **Type**: `struct`
- **Members**:
    - `is_basic_json`: A template struct that inherits from `std::false_type`.
- **Description**: `is_basic_json` is a template struct designed to serve as a type trait that defaults to `false`, indicating that a type is not a basic JSON type. This struct can be specialized for specific types to provide a mechanism for type checking in template metaprogramming.
- **Inherits From**:
    - `std::false_type`


---
### is\_basic\_json\_context<!-- {{#data_structure:namespace::is_basic_json_context}} -->
- **Type**: `struct`
- **Members**:
    - `value`: A boolean value indicating if `BasicJsonContext` is a valid JSON context.
- **Description**: The `is_basic_json_context` is a template `struct` that determines if a given type, `BasicJsonContext`, is a valid JSON context by checking if it is either a type derived from `BasicJson` or if it is a `nullptr_t`. This is achieved using `std::integral_constant` to provide a compile-time boolean value based on the type traits of `BasicJsonContext`.
- **Inherits From**:
    - `std::integral_constant < bool,
    is_basic_json<typename std::remove_cv<typename std::remove_pointer<BasicJsonContext>::type>::type>::value
    || std::is_same<BasicJsonContext, std::nullptr_t>::value >`


---
### is\_json\_ref<!-- {{#data_structure:namespace::is_json_ref}} -->
- **Type**: `struct`
- **Members**:
    - `is_json_ref`: A template struct that inherits from `std::false_type`.
- **Description**: `is_json_ref` is a template struct designed to serve as a base type that always evaluates to `false`, indicating that a given type is not a JSON reference.
- **Inherits From**:
    - `std::false_type`


---
### has\_from\_json<!-- {{#data_structure:namespace::has_from_json}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: The type used for JSON representation.
    - `T`: The type that is being checked for the `from_json` function.
    - `void`: A default template parameter that allows for SFINAE.
- **Description**: `has_from_json` is a template `struct` designed to determine if a specific type `T` has a `from_json` function defined for it, utilizing SFINAE (Substitution Failure Is Not An Error) to provide a compile-time boolean value.
- **Inherits From**:
    - `std::false_type`


---
### is\_getable<!-- {{#data_structure:namespace::is_getable}} -->
- **Type**: `struct`
- **Members**:
    - `value`: A static constant boolean indicating if a `get_template_function` can be detected for the given types.
- **Description**: The `is_getable` struct is a template that determines if a specific template function, `get_template_function`, can be applied to a given `BasicJsonType` and type `T`. It uses a type trait mechanism to evaluate this at compile time, providing a boolean value that indicates the result.


---
### has\_non\_default\_from\_json<!-- {{#data_structure:namespace::has_non_default_from_json}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: A template parameter representing the type of JSON being used.
    - `T`: A template parameter representing the type to be checked for a non-default `from_json` function.
    - `void`: A default template parameter used for SFINAE (Substitution Failure Is Not An Error) purposes.
- **Description**: `has_non_default_from_json` is a template `struct` designed to determine if a specific type `T` has a non-default implementation of a `from_json` function for the given `BasicJsonType`, defaulting to `false_type` if not.
- **Inherits From**:
    - `std::false_type`


---
### has\_to\_json<!-- {{#data_structure:namespace::has_to_json}} -->
- **Type**: `struct`
- **Description**: `has_to_json` is a template struct that is designed to determine if a type `T` has a `to_json` function, defaulting to `false` if not.
- **Inherits From**:
    - `std::false_type`


---
### has\_key\_compare<!-- {{#data_structure:namespace::has_key_compare}} -->
- **Type**: `struct`
- **Members**:
    - `T`: The template type parameter used to define the structure.
- **Description**: `has_key_compare` is a template `struct` that determines if a type `T` has a key comparison function by utilizing the `is_detected` utility with a `detect_key_compare` type trait.
- **Inherits From**:
    - `std::integral_constant<bool, is_detected<detect_key_compare, T>::value>`


---
### actual\_object\_comparator<!-- {{#data_structure:namespace::actual_object_comparator}} -->
- **Type**: `struct`
- **Members**:
    - `object_t`: Defines a type alias for the object type of the `BasicJsonType`.
    - `object_comparator_t`: Defines a type alias for the default object comparator of the `BasicJsonType`.
    - `type`: Defines a type alias that conditionally selects a key comparison type based on the presence of a key comparator.
- **Description**: The `actual_object_comparator` is a template struct that provides type aliases for handling object types and comparators in a JSON-like structure, allowing for flexible comparison strategies based on the characteristics of the provided `BasicJsonType`.


---
### char\_traits<!-- {{#data_structure:namespace::char_traits}} -->
- **Type**: `struct `char_traits` is a template struct.`
- **Members**:
    - `T`: The template parameter representing the character type.
- **Description**: The `char_traits` struct is a template specialization that inherits from `std::char_traits<T>`, allowing for the customization of character traits for different character types in C++.
- **Inherits From**:
    - `std::char_traits<T>`


---
### conjunction<!-- {{#data_structure:namespace::conjunction}} -->
- **Type**: `struct`
- **Members**:
    - `B`: A template parameter that represents a boolean type.
- **Description**: `conjunction` is a variadic template struct that evaluates to `std::true_type` if all provided template parameters are true types, and to the type of the last parameter if only one is provided, effectively implementing a logical AND operation for type traits.
- **Inherits From**:
    - `std::true_type`


---
### negation<!-- {{#data_structure:namespace::negation}} -->
- **Type**: `struct`
- **Members**:
    - `B`: A template parameter representing a boolean type.
- **Description**: The `negation` struct is a template that inherits from `std::integral_constant`, providing a compile-time boolean value that is the logical negation of the boolean value represented by the template parameter `B`.
- **Inherits From**:
    - `std::integral_constant < bool, !B::value >`


---
### is\_default\_constructible<!-- {{#data_structure:namespace::is_default_constructible}} -->
- **Type**: `struct`
- **Members**:
    - `T`: The type parameter for which default constructibility is being checked.
- **Description**: The `is_default_constructible` struct is a template that inherits from `std::is_default_constructible`, allowing it to determine if a type `T` can be default constructed, providing a compile-time boolean value.
- **Inherits From**:
    - `std::is_default_constructible<T>`


---
### is\_constructible<!-- {{#data_structure:namespace::is_constructible}} -->
- **Type**: `struct`
- **Members**:
    - `T`: The type that is being checked for constructibility.
    - `Args`: The types of arguments that are passed to the constructor of type `T`.
- **Description**: The `is_constructible` struct is a template that determines if an object of type `T` can be constructed using the provided argument types `Args...`, leveraging the functionality of `std::is_constructible` from the standard library.
- **Inherits From**:
    - `std::is_constructible<T, Args...>`


---
### is\_iterator\_traits<!-- {{#data_structure:namespace::is_iterator_traits}} -->
- **Type**: `struct`
- **Members**:
    - `T`: The type parameter that represents the type being checked for iterator traits.
    - `void`: A default template parameter that allows specialization of the `is_iterator_traits` struct.
- **Description**: `is_iterator_traits` is a template struct that serves as a base case for determining whether a type `T` has iterator traits, defaulting to `std::false_type` to indicate that `T` does not possess these traits unless specialized.
- **Inherits From**:
    - `std::false_type`


---
### is\_range<!-- {{#data_structure:namespace::is_range}} -->
- **Type**: `struct`
- **Members**:
    - `value`: A compile-time constant indicating whether the type `T` is a range.
    - `is_iterator_begin`: A compile-time constant that checks if the type has valid iterator traits.
    - `iterator`: A type alias for the iterator type detected from the type `T`.
    - `sentinel`: A type alias for the sentinel type detected from the type `T`.
- **Description**: The `is_range` struct is a template that determines if a given type `T` can be treated as a range by checking for the presence of valid iterator and sentinel types, utilizing type traits and SFINAE (Substitution Failure Is Not An Error) principles.


---
### is\_complete\_type<!-- {{#data_structure:namespace::is_complete_type}} -->
- **Type**: `struct`
- **Members**:
    - `T`: A template parameter representing the type to be checked.
    - `std::false_type`: Inherits from `std::false_type`, indicating that the type is not complete.
- **Description**: `is_complete_type` is a template struct that serves as a type trait to determine if a given type `T` is complete, defaulting to `false` for incomplete types.
- **Inherits From**:
    - `std::false_type`


---
### is\_compatible\_object\_type\_impl<!-- {{#data_structure:namespace::is_compatible_object_type_impl}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: A template parameter representing a basic JSON type.
    - `CompatibleObjectType`: A template parameter representing a compatible object type.
- **Description**: `is_compatible_object_type_impl` is a template specialization of a struct that inherits from `std::false_type`, indicating that by default, the combination of `BasicJsonType` and `CompatibleObjectType` is not considered compatible.
- **Inherits From**:
    - `std::false_type`


---
### is\_compatible\_object\_type<!-- {{#data_structure:namespace::is_compatible_object_type}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: A template parameter representing the basic JSON type.
    - `CompatibleObjectType`: A template parameter representing the compatible object type.
- **Description**: The `is_compatible_object_type` is a template `struct` that inherits from `is_compatible_object_type_impl`, allowing for compile-time checks to determine if a given `BasicJsonType` is compatible with a specified `CompatibleObjectType`.
- **Inherits From**:
    - [`namespace::is_compatible_object_type_impl`](#namespaceis_compatible_object_type_impl)


---
### is\_constructible\_object\_type\_impl<!-- {{#data_structure:namespace::is_constructible_object_type_impl}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: A template parameter representing the type of JSON.
    - `ConstructibleObjectType`: A template parameter representing the type intended to be constructed.
- **Description**: `is_constructible_object_type_impl` is a template specialization of a struct that inherits from `std::false_type`, indicating that the specified `ConstructibleObjectType` cannot be constructed from the given `BasicJsonType`.
- **Inherits From**:
    - `std::false_type`


---
### is\_constructible\_object\_type<!-- {{#data_structure:namespace::is_constructible_object_type}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: A template parameter representing the type of the basic JSON object.
    - `ConstructibleObjectType`: A template parameter representing the type that is being checked for constructibility.
- **Description**: The `is_constructible_object_type` is a template `struct` that inherits from `is_constructible_object_type_impl`, facilitating the determination of whether a given `ConstructibleObjectType` can be constructed from a `BasicJsonType`, thus enabling type-checking in template metaprogramming.
- **Inherits From**:
    - [`namespace::is_constructible_object_type_impl`](#namespaceis_constructible_object_type_impl)


---
### is\_compatible\_string\_type<!-- {{#data_structure:namespace::is_compatible_string_type}} -->
- **Type**: `struct`
- **Members**:
    - `value`: A static constexpr boolean value indicating if `CompatibleStringType` can be constructed from `BasicJsonType::string_t`.
- **Description**: The `is_compatible_string_type` is a template struct that determines if a given `CompatibleStringType` can be used to construct the string type defined within a `BasicJsonType`. It utilizes the `is_constructible` type trait to evaluate the compatibility at compile time, providing a boolean value that can be used for type checks in template metaprogramming.


---
### is\_constructible\_string\_type<!-- {{#data_structure:namespace::is_constructible_string_type}} -->
- **Type**: `struct`
- **Members**:
    - `laundered_type`: A type alias that resolves to the type of `ConstructibleStringType` or its laundered version depending on the compiler.
    - `value`: A compile-time constant that indicates whether `ConstructibleStringType` can be constructed from `BasicJsonType::string_t`.
- **Description**: The `is_constructible_string_type` struct is a template that checks if a given type, `ConstructibleStringType`, can be constructed from a specific string type defined in `BasicJsonType`. It uses type traits and conditional compilation to ensure compatibility across different compilers, particularly addressing issues with the Intel Compiler. The struct provides a type alias for a potentially modified version of `ConstructibleStringType` and a static constant that evaluates to true or false based on the constructibility conditions.


---
### is\_compatible\_array\_type\_impl<!-- {{#data_structure:namespace::is_compatible_array_type_impl}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: A template parameter representing the basic JSON type.
    - `CompatibleArrayType`: A template parameter representing a compatible array type.
- **Description**: `is_compatible_array_type_impl` is a template specialization of a `struct` that inherits from `std::false_type`, indicating that by default, the combination of `BasicJsonType` and `CompatibleArrayType` is not considered compatible for array types.
- **Inherits From**:
    - `std::false_type`


---
### is\_compatible\_array\_type<!-- {{#data_structure:namespace::is_compatible_array_type}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: A template parameter representing the basic JSON type.
    - `CompatibleArrayType`: A template parameter representing the compatible array type.
- **Description**: The `is_compatible_array_type` is a template `struct` that inherits from `is_compatible_array_type_impl`, allowing for type trait checks to determine compatibility between a specified `BasicJsonType` and a `CompatibleArrayType`.
- **Inherits From**:
    - [`namespace::is_compatible_array_type_impl`](#namespaceis_compatible_array_type_impl)


---
### is\_constructible\_array\_type\_impl<!-- {{#data_structure:namespace::is_constructible_array_type_impl}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: A template parameter representing the basic JSON type.
    - `ConstructibleArrayType`: A template parameter representing the type intended to be checked for constructibility.
- **Description**: `is_constructible_array_type_impl` is a template specialization of a struct that inherits from `std::false_type`, indicating that the specified `ConstructibleArrayType` cannot be constructed from the `BasicJsonType`.
- **Inherits From**:
    - `std::false_type`


---
### is\_constructible\_array\_type<!-- {{#data_structure:namespace::is_constructible_array_type}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: A template parameter representing the type of JSON data.
    - `ConstructibleArrayType`: A template parameter representing the type intended to be constructed as an array.
- **Description**: The `is_constructible_array_type` is a template `struct` that inherits from `is_constructible_array_type_impl`, designed to determine if a specific type can be constructed as an array from a given JSON type.
- **Inherits From**:
    - [`namespace::is_constructible_array_type_impl`](#namespaceis_constructible_array_type_impl)


---
### is\_compatible\_integer\_type\_impl<!-- {{#data_structure:namespace::is_compatible_integer_type_impl}} -->
- **Type**: `struct`
- **Members**:
    - `RealIntegerType`: A template parameter representing a real integer type.
    - `CompatibleNumberIntegerType`: A template parameter representing a compatible number integer type.
- **Description**: `is_compatible_integer_type_impl` is a template `struct` that inherits from `std::false_type`, indicating that by default, the combination of `RealIntegerType` and `CompatibleNumberIntegerType` is not considered compatible unless specialized.
- **Inherits From**:
    - `std::false_type`


---
### is\_compatible\_integer\_type<!-- {{#data_structure:namespace::is_compatible_integer_type}} -->
- **Type**: `struct`
- **Members**:
    - `RealIntegerType`: The first template parameter representing a real integer type.
    - `CompatibleNumberIntegerType`: The second template parameter representing a compatible number integer type.
- **Description**: The `is_compatible_integer_type` is a template `struct` that determines compatibility between two integer types, `RealIntegerType` and `CompatibleNumberIntegerType`, by inheriting from another implementation struct, `is_compatible_integer_type_impl`, which likely contains the logic for the compatibility check.
- **Inherits From**:
    - [`namespace::is_compatible_integer_type_impl`](#namespaceis_compatible_integer_type_impl)


---
### is\_compatible\_type\_impl<!-- {{#data_structure:namespace::is_compatible_type_impl}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: A template parameter representing the basic JSON type.
    - `CompatibleType`: A template parameter representing a type that may be compatible.
    - `void`: A default template parameter that is unused in this context.
- **Description**: `is_compatible_type_impl` is a template specialization of a struct that inherits from `std::false_type`, indicating that by default, the specified `CompatibleType` is not compatible with the `BasicJsonType`.
- **Inherits From**:
    - `std::false_type`


---
### is\_compatible\_type<!-- {{#data_structure:namespace::is_compatible_type}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: A template parameter representing the basic JSON type.
    - `CompatibleType`: A template parameter representing a type that may be compatible with the basic JSON type.
- **Description**: The `is_compatible_type` struct is a template that inherits from `is_compatible_type_impl`, allowing for compile-time checks to determine if a given `CompatibleType` is compatible with a specified `BasicJsonType`, facilitating type safety in JSON handling.
- **Inherits From**:
    - [`namespace::is_compatible_type_impl`](#namespaceis_compatible_type_impl)


---
### is\_constructible\_tuple<!-- {{#data_structure:namespace::is_constructible_tuple}} -->
- **Type**: `struct`
- **Members**:
    - `T1`: The first type parameter used to determine constructibility.
    - `T2`: The second type parameter used to determine constructibility.
- **Description**: `is_constructible_tuple` is a template `struct` that inherits from `std::false_type`, indicating that a tuple cannot be constructed from the specified types `T1` and `T2`.
- **Inherits From**:
    - `std::false_type`


---
### is\_json\_iterator\_of<!-- {{#data_structure:namespace::is_json_iterator_of}} -->
- **Type**: `struct`
- **Members**:
    - `BasicJsonType`: A template parameter representing the type of the JSON object.
    - `T`: A template parameter representing the type of the iterator.
- **Description**: `is_json_iterator_of` is a template `struct` that inherits from `std::false_type`, indicating that by default, it is not a JSON iterator for the specified types `BasicJsonType` and `T`. This structure is likely intended to be specialized for specific types to provide a mechanism for type traits in template metaprogramming.
- **Inherits From**:
    - `std::false_type`


---
### is\_specialization\_of<!-- {{#data_structure:namespace::is_specialization_of}} -->
- **Type**: `struct`
- **Members**:
    - `Primary`: A template parameter representing a primary template.
    - `T`: A type parameter that is used in conjunction with the primary template.
- **Description**: `is_specialization_of` is a template `struct` that inherits from `std::false_type`, designed to determine if a given type `T` is a specialization of a template `Primary`. This structure serves as a base for further specialization, allowing for compile-time type checks in template metaprogramming.
- **Inherits From**:
    - `std::false_type`


---
### is\_comparable<!-- {{#data_structure:namespace::is_comparable}} -->
- **Type**: `struct`
- **Members**:
    - `Compare`: A template parameter representing the comparison type.
    - `A`: A template parameter representing the first type to compare.
    - `B`: A template parameter representing the second type to compare.
    - `void`: A default template parameter used for SFINAE (Substitution Failure Is Not An Error).
- **Description**: `is_comparable` is a template `struct` that determines if two types, `A` and `B`, can be compared using a specified comparison type `Compare`, defaulting to `std::false_type` to indicate that comparison is not possible unless specialized.
- **Inherits From**:
    - `std::false_type`


---
### is\_ordered\_map<!-- {{#data_structure:namespace::is_ordered_map}} -->
- **Type**: `struct`
- **Members**:
    - `one`: A type alias for `char`.
    - `two`: A nested struct containing a character array of size 2.
    - `value`: A compile-time constant that indicates if the type `T` has a `capacity` member function.
- **Description**: The `is_ordered_map` struct is a type trait designed to determine if a given type `T` behaves like an ordered map by checking for the presence of a `capacity` member function. It utilizes SFINAE (Substitution Failure Is Not An Error) to differentiate between types that have this member function and those that do not, returning a compile-time boolean value through the `value` member.


---
### two<!-- {{#data_structure:namespace::is_ordered_map::two}} -->
- **Type**: `struct`
- **Members**:
    - `x`: An array of two characters.
- **Description**: The `two` struct is a simple data structure that contains a single member, `x`, which is an array designed to hold exactly two characters, providing a compact representation for storing small fixed-size character data.


---
### is\_c\_string<!-- {{#data_structure:namespace::is_c_string}} -->
- **Type**: `struct`
- **Members**:
    - `T`: The template type parameter used to determine if it is a C-style string.
- **Description**: `is_c_string` is a template `struct` that inherits from `bool_constant`, evaluating to `true` if the type `T` is a C-style string, based on the implementation defined in `impl::is_c_string<T>()`.
- **Inherits From**:
    - `bool_constant`


---
### is\_transparent<!-- {{#data_structure:namespace::is_transparent}} -->
- **Type**: `struct`
- **Members**:
    - `T`: The template type parameter used to determine transparency.
- **Description**: The `is_transparent` struct is a template that inherits from `bool_constant`, evaluating to true or false based on whether the type `T` is considered transparent according to the `impl::is_transparent` function.
- **Inherits From**:
    - `bool_constant`


---
### parse\_error<!-- {{#data_structure:parse_error}} -->
- **Type**: `class `parse_error``
- **Members**:
    - `byte`: The byte index of the last read character in the input file.
- **Description**: The `parse_error` class is a custom exception type that inherits from `exception`, designed to represent errors encountered during parsing operations, encapsulating details such as the error's byte index and a descriptive message.
- **Member Functions**:
    - [`parse_error::create`](#parse_errorcreate)
    - [`parse_error::create`](#parse_errorcreate)
    - [`parse_error::parse_error`](#parse_errorparse_error)
    - [`parse_error::position_string`](#parse_errorposition_string)
- **Inherits From**:
    - [`namespace::exception`](#namespaceexception)

**Methods**

---
#### parse\_error::create<!-- {{#callable:parse_error::create}} -->
Creates a `parse_error` object with detailed error information.
- **Inputs**:
    - `id_`: An integer representing the ID of the exception.
    - `pos`: A `position_t` object indicating the position where the error occurred.
    - `what_arg`: A string providing an explanatory message about the error.
    - `context`: A context object of type `BasicJsonContext` used for diagnostics.
- **Control Flow**:
    - Concatenates various strings to create a detailed error message.
    - The error message includes the exception name, position information, diagnostics from the context, and the explanatory string.
    - Returns a new `parse_error` object initialized with the ID, total characters read, and the constructed error message.
- **Output**: Returns a `parse_error` object that encapsulates the error details.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)
    - [`parse_error::position_string`](#parse_errorposition_string)
- **See also**: [`parse_error`](#parse_error)  (Data Structure)


---
#### parse\_error::create<!-- {{#callable:parse_error::create}} -->
Creates a `parse_error` object with a detailed error message.
- **Inputs**:
    - `id_`: An integer representing the ID of the exception.
    - `byte_`: A size_t value indicating the byte position where the error occurred.
    - `what_arg`: A string providing additional context or explanation for the error.
    - `context`: A context object of type `BasicJsonContext` used for diagnostics.
- **Control Flow**:
    - Concatenates various strings to form a detailed error message.
    - The error message includes the exception name, the byte position (if not zero), diagnostics from the context, and the explanatory string.
    - Returns a new `parse_error` object initialized with the provided ID, byte position, and constructed error message.
- **Output**: Returns a `parse_error` object that encapsulates the error details.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`parse_error`](#parse_error)  (Data Structure)


---
#### parse\_error::parse\_error<!-- {{#callable:parse_error::parse_error}} -->
Constructs a `parse_error` exception with an error ID, byte position, and explanatory message.
- **Inputs**:
    - `id_`: An integer representing the unique identifier of the exception.
    - `byte_`: A size_t value indicating the byte position where the parse error occurred.
    - `what_arg`: A C-style string providing an explanatory message for the error.
- **Control Flow**:
    - The constructor initializes the base class `exception` with the provided `id_` and `what_arg`.
    - It then assigns the `byte_` value to the member variable `byte`, which stores the byte index of the parse error.
- **Output**: The constructor does not return a value but initializes a `parse_error` object that can be used to represent a parsing error in a JSON context.
- **See also**: [`parse_error`](#parse_error)  (Data Structure)


---
#### parse\_error::position\_string<!-- {{#callable:parse_error::position_string}} -->
Generates a formatted string indicating the line and column position of a parsing error.
- **Inputs**:
    - `pos`: A constant reference to a `position_t` structure that contains the current line and character read information.
- **Control Flow**:
    - The function uses the [`concat`](#namespaceconcat) function to build a string.
    - It retrieves the line number by adding 1 to `pos.lines_read` to account for zero-based indexing.
    - It retrieves the current column number from `pos.chars_read_current_line`.
- **Output**: Returns a formatted string that specifies the line and column where the error occurred.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`parse_error`](#parse_error)  (Data Structure)



---
### invalid\_iterator<!-- {{#data_structure:invalid_iterator}} -->
- **Type**: `class`
- **Members**:
    - `id_`: An integer representing the identifier for the invalid iterator.
    - `what_arg`: A string that describes the error associated with the invalid iterator.
- **Description**: The `invalid_iterator` class is a custom exception derived from the standard `exception` class, designed to handle errors related to invalid iterators in a JSON context, encapsulating an error identifier and a descriptive message.
- **Member Functions**:
    - [`invalid_iterator::create`](#invalid_iteratorcreate)
- **Inherits From**:
    - [`namespace::exception`](#namespaceexception)

**Methods**

---
#### invalid\_iterator::create<!-- {{#callable:invalid_iterator::create}} -->
The `create` function constructs an `invalid_iterator` exception with a formatted error message.
- **Inputs**:
    - `id_`: An integer identifier for the exception.
    - `what_arg`: A string providing additional context about the exception.
    - `context`: A context object of type `BasicJsonContext` used for diagnostics.
- **Control Flow**:
    - The function begins by concatenating the name of the exception, its diagnostics, and the additional argument into a single string.
    - It uses the [`concat`](#namespaceconcat) function to combine these components into a formatted message.
    - Finally, it returns a new instance of `invalid_iterator` initialized with the identifier and the formatted message.
- **Output**: The function outputs an instance of `invalid_iterator` initialized with the provided identifier and a detailed error message.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`invalid_iterator`](#invalid_iterator)  (Data Structure)



---
### type\_error<!-- {{#data_structure:type_error}} -->
- **Type**: `class`
- **Members**:
    - `id_`: An integer representing the error ID.
    - `what_arg`: A C-style string that describes the error.
- **Description**: The `type_error` class is a custom exception type derived from the standard `exception` class, designed to represent errors related to type mismatches in JSON contexts, encapsulating an error ID and a descriptive message.
- **Member Functions**:
    - [`type_error::create`](#type_errorcreate)
- **Inherits From**:
    - [`namespace::exception`](#namespaceexception)

**Methods**

---
#### type\_error::create<!-- {{#callable:type_error::create}} -->
Creates a `type_error` exception with a formatted message based on the provided parameters.
- **Inputs**:
    - `id_`: An integer identifier for the type of error.
    - `what_arg`: A string providing additional context or description of the error.
    - `context`: A context object of type `BasicJsonContext` used to gather diagnostic information.
- **Control Flow**:
    - The function begins by concatenating the error name, diagnostics from the context, and the additional error description into a single string.
    - It then constructs and returns a `type_error` object using the concatenated string and the provided error ID.
- **Output**: Returns a `type_error` object initialized with the error ID and the formatted error message.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`type_error`](#type_error)  (Data Structure)



---
### out\_of\_range<!-- {{#data_structure:out_of_range}} -->
- **Type**: `class`
- **Members**:
    - `id_`: An integer representing the error ID.
    - `what_arg`: A C-style string that describes the error.
- **Description**: The `out_of_range` class is a custom exception derived from the standard `exception` class, designed to handle out-of-range errors specifically in JSON contexts, encapsulating an error ID and a descriptive message.
- **Member Functions**:
    - [`out_of_range::create`](#out_of_rangecreate)
- **Inherits From**:
    - [`namespace::exception`](#namespaceexception)

**Methods**

---
#### out\_of\_range::create<!-- {{#callable:out_of_range::create}} -->
The `create` function constructs an `out_of_range` exception with a detailed message.
- **Inputs**:
    - `id_`: An integer identifier for the exception.
    - `what_arg`: A string providing additional context about the exception.
    - `context`: A context object of type `BasicJsonContext` used for diagnostics.
- **Control Flow**:
    - The function begins by concatenating the exception name, diagnostics from the context, and the additional argument into a single string.
    - It then constructs and returns an `out_of_range` object using the concatenated string and the provided identifier.
- **Output**: The function returns an `out_of_range` object initialized with the given identifier and a detailed message.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`out_of_range`](#out_of_range)  (Data Structure)



---
### other\_error<!-- {{#data_structure:other_error}} -->
- **Type**: `class`
- **Members**:
    - `id_`: An integer representing the error identifier.
    - `what_arg`: A character pointer that holds the error message.
- **Description**: The `other_error` class is a custom exception type that inherits from the standard `exception` class, designed to encapsulate error information with a specific identifier and a descriptive message.
- **Member Functions**:
    - [`other_error::create`](#other_errorcreate)
- **Inherits From**:
    - [`namespace::exception`](#namespaceexception)

**Methods**

---
#### other\_error::create<!-- {{#callable:other_error::create}} -->
The `create` function constructs an `other_error` object using an error ID, a message string, and a JSON context.
- **Inputs**:
    - `id_`: An integer representing the error ID.
    - `what_arg`: A string containing additional information about the error.
    - `context`: A context object of type `BasicJsonContext` used for diagnostics.
- **Control Flow**:
    - The function begins by concatenating the error name, diagnostics from the context, and the additional message into a single string `w`.
    - It then creates and returns an `other_error` object initialized with the error ID and the concatenated message.
- **Output**: The function outputs an `other_error` object that encapsulates the error ID and a detailed error message.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`other_error`](#other_error)  (Data Structure)



---
### identity\_tag<!-- {{#data_structure:namespace::identity_tag}} -->
- **Type**: `struct`
- **Description**: The `identity_tag` is a template struct that serves as a marker type, allowing for type differentiation without carrying any data.


---
### from\_json\_fn<!-- {{#data_structure:namespace::from_json_fn}} -->
- **Type**: `struct`
- **Members**:
    - `operator()`: A templated function call operator that deserializes a JSON object into a given value.
- **Description**: The `from_json_fn` struct is designed to facilitate the deserialization of JSON data into C++ objects by providing a templated function call operator that invokes the `from_json` function, ensuring type safety and forwarding of the value.
- **Member Functions**:
    - [`namespace::from_json_fn::operator()`](#from_json_fnoperator())

**Methods**

---
#### from\_json\_fn::operator\(\)<!-- {{#callable:namespace::from_json_fn::operator()}} -->
The `operator()` function in the `from_json_fn` struct converts a JSON object into a specified type using the [`from_json`](#from_json) function.
- **Inputs**:
    - `j`: A constant reference to a JSON object of type `BasicJsonType` that contains the data to be converted.
    - `val`: A forwarding reference of type `T` that represents the variable where the converted data will be stored.
- **Control Flow**:
    - The function first checks if the [`from_json`](#from_json) function can be called with the provided arguments using [`noexcept`](#noexcept).
    - It then calls the [`from_json`](#from_json) function with the JSON object `j` and the forwarded variable `val`.
    - The result of the [`from_json`](#from_json) call is returned directly.
- **Output**: The output is the result of the [`from_json`](#from_json) function, which is the converted value of type `T` derived from the JSON object `j`.
- **Functions called**:
    - [`noexcept`](#noexcept)
    - [`from_json`](#from_json)
- **See also**: [`namespace::from_json_fn`](#namespacefrom_json_fn)  (Data Structure)



---
### iteration\_proxy\_value<!-- {{#data_structure:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value}} -->
- **Type**: `class`
- **Members**:
    - `anchor`: The iterator that serves as the core element of the proxy.
    - `array_index`: An index used for arrays to create key names.
    - `array_index_last`: Stores the last stringified array index for comparison.
    - `array_index_str`: A string representation of the current array index.
    - `empty_str`: An empty string used for returning references for primitive values.
- **Description**: The `iteration_proxy_value` class is a template-based iterator proxy designed to facilitate iteration over collections, providing a mechanism to access both keys and values while maintaining an index for array elements. It encapsulates an iterator and manages the conversion of array indices to string representations, allowing for seamless integration with range-based for loops and other iterator-based constructs.
- **Member Functions**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator=`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator=)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator=`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator=)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::~iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::~iteration_proxy_value)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator*`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator*)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator++`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator++)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator++`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator++)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator==`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator==)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator!=`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator!=)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::key`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::key)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::value)

**Methods**

---
#### iteration\_proxy\_value::iteration\_proxy\_value<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value}} -->
Constructs an `iteration_proxy_value` object that encapsulates an iterator and an optional array index.
- **Inputs**:
    - `it`: An iterator of type `IteratorType` that the `iteration_proxy_value` will encapsulate.
    - `array_index_`: An optional size_t representing the index of the array, defaulting to 0.
- **Control Flow**:
    - The constructor initializes the `anchor` member with the provided iterator `it` using move semantics.
    - The `array_index` member is initialized with the provided `array_index_` value.
    - If the constructor is called with no arguments, it defaults to initializing an empty `iteration_proxy_value`.
- **Output**: The constructor does not return a value but initializes an instance of `iteration_proxy_value` with the specified iterator and array index.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value)  (Data Structure)


---
#### iteration\_proxy\_value::iteration\_proxy\_value<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value}} -->
Constructs an `iteration_proxy_value` object with a given iterator and an optional array index.
- **Inputs**:
    - `it`: An iterator of type `IteratorType` that will be used to traverse a collection.
    - `array_index_`: An optional `std::size_t` representing the index in an array, defaulting to 0.
- **Control Flow**:
    - The constructor initializes the `anchor` member with the provided iterator using move semantics.
    - The `array_index` member is initialized with the provided array index value.
- **Output**: The constructor does not return a value but initializes an instance of `iteration_proxy_value` with the specified iterator and array index.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value)  (Data Structure)


---
#### iteration\_proxy\_value::iteration\_proxy\_value<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value}} -->
The `iteration_proxy_value` class provides a proxy for iterating over a collection, encapsulating an iterator and managing array indices.
- **Inputs**:
    - `it`: An iterator of type `IteratorType` used to initialize the proxy.
    - `array_index_`: An optional size_t representing the index of the current element in an array, defaulting to 0.
- **Control Flow**:
    - The constructor initializes the `anchor` with the provided iterator and sets the `array_index`.
    - The class supports copy and move semantics through defaulted copy and move constructors and assignment operators.
    - The `operator*` returns a reference to the current proxy value.
    - The `operator++` increments the iterator and the array index, while the post-increment operator returns a copy of the current state before incrementing.
    - The equality and inequality operators compare the underlying iterators for equality.
    - The `key` method determines the key based on the type of the object pointed to by the iterator, handling arrays, objects, and primitive types appropriately.
    - The `value` method retrieves the value from the iterator.
- **Output**: The class does not produce a direct output but provides methods to access the key and value of the current iterator position.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value)  (Data Structure)


---
#### iteration\_proxy\_value::operator=<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator=}} -->
The `operator=` method assigns the value of one `iteration_proxy_value` instance to another, utilizing the default assignment operator behavior.
- **Inputs**:
    - `other`: A constant reference to another `iteration_proxy_value` instance from which the current instance will be assigned.
- **Control Flow**:
    - The method uses the default assignment operator behavior provided by the compiler, which performs a member-wise assignment of the data members from the `other` instance to the current instance.
    - No additional logic or checks are performed within this method, as it relies on the default behavior.
- **Output**: The method returns a reference to the current instance of `iteration_proxy_value`, allowing for chained assignments.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value)  (Data Structure)


---
#### iteration\_proxy\_value::iteration\_proxy\_value<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value}} -->
Moves the state of one `iteration_proxy_value` instance to another.
- **Inputs**:
    - `other`: An rvalue reference to another `iteration_proxy_value` instance, which is to be moved.
- **Control Flow**:
    - The function is a move constructor that initializes a new instance of `iteration_proxy_value` by transferring resources from the provided rvalue reference.
    - It uses the default move constructor behavior, which is specified to be noexcept if the contained types are also noexcept movable.
- **Output**: The function does not return a value; it constructs a new `iteration_proxy_value` instance by moving the state from the provided instance.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value)  (Data Structure)


---
#### iteration\_proxy\_value::operator=<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator=}} -->
Moves the state of one `iteration_proxy_value` instance to another.
- **Inputs**:
    - `other`: An rvalue reference to another `iteration_proxy_value` instance, which is to be moved.
- **Control Flow**:
    - The function is defined as a default move assignment operator, which means it will automatically handle the move semantics for the member variables of the class.
    - It checks if the move assignment is noexcept for both `IteratorType` and `string_type` to ensure that it can be safely used in contexts that require no exceptions.
- **Output**: Returns a reference to the current instance after moving the state from the provided `other` instance.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value)  (Data Structure)


---
#### iteration\_proxy\_value::operator\*<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator*}} -->
The `operator*` function returns a constant reference to the current instance of `iteration_proxy_value`.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple dereference operator that does not perform any complex logic.
    - It directly returns a reference to the current instance (`*this`), allowing the object to be used in contexts where dereferencing is required.
- **Output**: The output is a constant reference to the current `iteration_proxy_value` instance, enabling access to its members without copying.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value)  (Data Structure)


---
#### iteration\_proxy\_value::operator\+\+<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator++}} -->
The `operator++(int)` function increments the state of an [`iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value) object while returning a copy of its previous state.
- **Inputs**:
    - `this`: A reference to the current [`iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value) object on which the post-increment operation is performed.
- **Control Flow**:
    - A temporary [`iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value) object `tmp` is created, initialized with the current state of `anchor` and `array_index`.
    - The `anchor` iterator is incremented to point to the next element.
    - The `array_index` is also incremented to reflect the new position.
    - The function returns the temporary object `tmp`, which holds the previous state before the increment.
- **Output**: The function returns an [`iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value) object representing the state of the iterator before the increment operation.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::iteration_proxy_value)
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value)  (Data Structure)


---
#### iteration\_proxy\_value::operator==<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator==}} -->
The `operator==` function checks for equality between two `iteration_proxy_value` objects based on their `anchor` member.
- **Inputs**:
    - `o`: An instance of `iteration_proxy_value` to compare against the current instance.
- **Control Flow**:
    - The function directly compares the `anchor` member of the current instance with the `anchor` member of the input instance `o`.
    - It returns the result of the comparison, which is a boolean value indicating whether the two `anchor` members are equal.
- **Output**: The function returns a boolean value: `true` if the `anchor` members of both `iteration_proxy_value` instances are equal, and `false` otherwise.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value)  (Data Structure)


---
#### iteration\_proxy\_value::operator\!=<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::operator!=}} -->
The `operator!=` function checks if two `iteration_proxy_value` objects are not equal by comparing their underlying iterators.
- **Inputs**:
    - `o`: An instance of `iteration_proxy_value` to compare against the current instance.
- **Control Flow**:
    - The function directly compares the `anchor` member of the current instance with the `anchor` member of the input instance `o`.
    - It returns the result of the inequality comparison.
- **Output**: Returns a boolean value indicating whether the `anchor` of the current instance is not equal to the `anchor` of the provided instance `o`.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value)  (Data Structure)


---
#### iteration\_proxy\_value::key<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::key}} -->
The `key` function retrieves the key associated with the current element of an iterator, handling different types of JSON values.
- **Inputs**:
    - `this`: A constant reference to the current instance of `iteration_proxy_value`, which contains the iterator and its state.
- **Control Flow**:
    - The function first asserts that the `anchor.m_object` is not null using `JSON_ASSERT`.
    - It then checks the type of the object pointed to by `anchor.m_object` using a switch statement.
    - If the type is `array`, it checks if the current `array_index` is different from `array_index_last` and updates `array_index_str` accordingly.
    - If the type is `object`, it retrieves the key from the `anchor` object.
    - For all other types (including primitive types), it returns an empty string.
- **Output**: The function returns a constant reference to a string representing the key, which can be an array index, an object key, or an empty string for primitive types.
- **Functions called**:
    - [`(anonymous)::namespace::int_to_string`](#(anonymous)::namespace::int_to_string)
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value)  (Data Structure)


---
#### iteration\_proxy\_value::value<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::value}} -->
Returns the value referenced by the `anchor` iterator.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `iteration_proxy_value` class.
- **Control Flow**:
    - The function directly accesses the `value()` method of the `anchor` iterator.
    - No conditional logic or loops are present in this function.
- **Output**: Returns a reference to the value associated with the current position of the `anchor` iterator.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value)  (Data Structure)



---
### iteration\_proxy<!-- {{#data_structure:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy}} -->
- **Type**: `class`
- **Members**:
    - `container`: A pointer to the container that is being iterated over.
- **Description**: The `iteration_proxy` class is a template that provides a way to create an iterator-like interface for a given container type, allowing for range-based for loops by defining `begin` and `end` methods that return iterators.
- **Member Functions**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::iteration_proxy)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::iteration_proxy)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::iteration_proxy)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::operator=`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::operator=)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::iteration_proxy)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::operator=`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::operator=)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::~iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::~iteration_proxy)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::begin`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::begin)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::end`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::end)

**Methods**

---
#### iteration\_proxy::iteration\_proxy<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::iteration_proxy}} -->
The `iteration_proxy` constructor initializes an instance of the `iteration_proxy` class, which is designed to facilitate iteration over a container.
- **Inputs**:
    - `cont`: A reference to a container of type `IteratorType`, which is used to initialize the `container` pointer.
- **Control Flow**:
    - The constructor initializes the `container` member variable to point to the provided container reference.
    - If no container reference is provided, the default constructor initializes `container` to nullptr.
- **Output**: The constructor does not return a value but initializes the `iteration_proxy` object for use in iterating over the specified container.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy)  (Data Structure)


---
#### iteration\_proxy::iteration\_proxy<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::iteration_proxy}} -->
Constructs an `iteration_proxy` from a reference to a container.
- **Inputs**:
    - `cont`: A reference to a container of type `IteratorType`, which will be used for iteration.
- **Control Flow**:
    - The constructor initializes the private member `container` with the address of the provided reference `cont`.
    - The `noexcept` specifier indicates that this constructor does not throw exceptions.
- **Output**: The constructor does not return a value but initializes an instance of `iteration_proxy` that can be used to iterate over the specified container.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy)  (Data Structure)


---
#### iteration\_proxy::iteration\_proxy<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::iteration_proxy}} -->
The `iteration_proxy` class provides a wrapper for iterating over a container using range-based for loops.
- **Inputs**:
    - `cont`: A reference to a container of type `IteratorType`, which is used to initialize the `iteration_proxy`.
- **Control Flow**:
    - The constructor initializes the `container` pointer to the address of the provided container reference.
    - The class provides default copy and move constructors and assignment operators, allowing for standard behavior when copying or moving `iteration_proxy` instances.
    - The `begin()` method returns an iterator to the start of the container, while the `end()` method returns an iterator to the end of the container.
- **Output**: The output of the `begin()` and `end()` methods are `iteration_proxy_value<IteratorType>` objects that represent the start and end iterators of the container, respectively.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy)  (Data Structure)


---
#### iteration\_proxy::operator=<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::operator=}} -->
The `operator=` function assigns one `iteration_proxy` object to another, utilizing the default assignment operator behavior.
- **Inputs**:
    - `other`: A constant reference to another `iteration_proxy` object that is being assigned to the current object.
- **Control Flow**:
    - The function uses the default assignment operator provided by the compiler, which performs a member-wise assignment of the `iteration_proxy` object.
    - No additional logic is implemented in this function, as it relies on the default behavior.
- **Output**: The function returns a reference to the current `iteration_proxy` object after the assignment operation.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy)  (Data Structure)


---
#### iteration\_proxy::iteration\_proxy<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::iteration_proxy}} -->
Moves the state of one `iteration_proxy` object to another.
- **Inputs**:
    - `other`: An rvalue reference to another `iteration_proxy` object, which is being moved.
- **Control Flow**:
    - The function uses the default move constructor provided by the compiler, which transfers ownership of resources from the source object to the current object.
    - No additional logic is implemented in this function, as it relies on the default behavior.
- **Output**: The function does not return a value; it modifies the current object to take ownership of the resources from the `other` object.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy)  (Data Structure)


---
#### iteration\_proxy::operator=<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::operator=}} -->
Moves the state of one `iteration_proxy` object to another, effectively transferring ownership of the underlying container.
- **Inputs**:
    - `other`: An rvalue reference to another `iteration_proxy` object from which the state will be moved.
- **Control Flow**:
    - The function is defined as a default move assignment operator, which means it will automatically handle the transfer of resources from the source object to the current object.
    - No explicit resource management or state checks are performed, as the default behavior is sufficient for the class's needs.
- **Output**: Returns a reference to the current `iteration_proxy` object after transferring the state.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy)  (Data Structure)


---
#### iteration\_proxy::begin<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::begin}} -->
Returns an `iteration_proxy_value` representing the beginning of the container's iteration.
- **Inputs**: None
- **Control Flow**:
    - The function is a const method, ensuring it does not modify the state of the `iteration_proxy` instance.
    - It directly accesses the `begin()` method of the `container` pointer, which is expected to be a valid container type.
    - An `iteration_proxy_value` is constructed using the result of `container->begin()`.
- **Output**: Returns an `iteration_proxy_value<IteratorType>` that encapsulates the beginning iterator of the container.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy)  (Data Structure)


---
#### iteration\_proxy::end<!-- {{#callable:(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::end}} -->
Returns an `iteration_proxy_value` representing the end of the container being iterated.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `end()` method of the container pointed to by `container`.
    - It constructs and returns an `iteration_proxy_value<IteratorType>` using the result of the `end()` method.
- **Output**: An `iteration_proxy_value<IteratorType>` that signifies the end of the iteration over the container.
- **See also**: [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy)  (Data Structure)



---
### to\_json\_fn<!-- {{#data_structure:(anonymous)::namespace::to_json_fn}} -->
- **Type**: `struct`
- **Members**:
    - `operator()`: A templated function operator that converts a value to JSON format.
- **Description**: The `to_json_fn` struct defines a function object that facilitates the conversion of various types to JSON format by utilizing a templated `operator()`, which calls a `to_json` function with the provided JSON object and value.
- **Member Functions**:
    - [`(anonymous)::namespace::to_json_fn::operator()`](#(anonymous)::namespace::to_json_fn::operator())

**Methods**

---
#### to\_json\_fn::operator\(\)<!-- {{#callable:(anonymous)::namespace::to_json_fn::operator()}} -->
The `operator()` method in the `to_json_fn` struct converts a given value into a JSON representation and stores it in the provided JSON object.
- **Inputs**:
    - `j`: A reference to a JSON object of type `BasicJsonType` where the converted value will be stored.
    - `val`: A value of any type `T` that is to be converted into JSON format.
- **Control Flow**:
    - The method uses `std::forward` to perfectly forward the value `val` to the [`to_json`](#(anonymous)::namespace::to_json) function, preserving its value category.
    - It checks if the [`to_json`](#(anonymous)::namespace::to_json) function can be called with the provided arguments using [`noexcept`](#noexcept) to ensure that no exceptions will be thrown during the conversion.
- **Output**: The output is the result of the [`to_json`](#(anonymous)::namespace::to_json) function, which is the JSON representation of the input value `val` stored in the JSON object `j`.
- **Functions called**:
    - [`noexcept`](#noexcept)
    - [`(anonymous)::namespace::to_json`](#(anonymous)::namespace::to_json)
- **See also**: [`(anonymous)::namespace::to_json_fn`](#(anonymous)::namespace::to_json_fn)  (Data Structure)



---
### input\_format\_t<!-- {{#data_structure:(anonymous)::namespace::input_format_t}} -->
- **Type**: `enum class`
- **Members**:
    - `json`: Represents the JSON input format.
    - `cbor`: Represents the CBOR (Concise Binary Object Representation) input format.
    - `msgpack`: Represents the MessagePack input format.
    - `ubjson`: Represents the UBJSON (Universal Binary JSON) input format.
    - `bson`: Represents the BSON (Binary JSON) input format.
    - `bjdata`: Represents the BJData (Binary JSON Data) input format.
- **Description**: The `input_format_t` is an enumeration that defines a set of constants representing various input data formats, including JSON, CBOR, MessagePack, UBJSON, BSON, and BJData, allowing for type-safe handling of different serialization formats in applications.


---
### input\_stream\_adapter<!-- {{#data_structure:(anonymous)::namespace::input_stream_adapter}} -->
- **Type**: `class `input_stream_adapter``
- **Members**:
    - `is`: Pointer to the associated input stream.
    - `sb`: Pointer to the associated stream buffer.
- **Description**: The `input_stream_adapter` class is designed to facilitate reading from an `std::istream` by providing a streamlined interface to access characters and elements from the underlying stream buffer, while managing the stream's state.
- **Member Functions**:
    - [`(anonymous)::namespace::input_stream_adapter::~input_stream_adapter`](#(anonymous)::namespace::input_stream_adapter::~input_stream_adapter)
    - [`(anonymous)::namespace::input_stream_adapter::input_stream_adapter`](#(anonymous)::namespace::input_stream_adapter::input_stream_adapter)
    - [`(anonymous)::namespace::input_stream_adapter::input_stream_adapter`](#(anonymous)::namespace::input_stream_adapter::input_stream_adapter)
    - [`(anonymous)::namespace::input_stream_adapter::operator=`](#(anonymous)::namespace::input_stream_adapter::operator=)
    - [`(anonymous)::namespace::input_stream_adapter::operator=`](#(anonymous)::namespace::input_stream_adapter::operator=)
    - [`(anonymous)::namespace::input_stream_adapter::input_stream_adapter`](#(anonymous)::namespace::input_stream_adapter::input_stream_adapter)
    - [`(anonymous)::namespace::input_stream_adapter::get_character`](#(anonymous)::namespace::input_stream_adapter::get_character)
    - [`(anonymous)::namespace::input_stream_adapter::get_elements`](#(anonymous)::namespace::input_stream_adapter::get_elements)

**Methods**

---
#### input\_stream\_adapter::\~input\_stream\_adapter<!-- {{#callable:(anonymous)::namespace::input_stream_adapter::~input_stream_adapter}} -->
The `~input_stream_adapter` destructor clears the stream flags of the associated input stream, retaining only the end-of-file state.
- **Inputs**: None
- **Control Flow**:
    - Checks if the member pointer `is` (the associated input stream) is not null.
    - If `is` is valid, it clears the stream's state flags, preserving only the end-of-file flag.
- **Output**: The function does not return a value; it modifies the state of the associated input stream by clearing its flags.
- **See also**: [`(anonymous)::namespace::input_stream_adapter`](#(anonymous)::namespace::input_stream_adapter)  (Data Structure)


---
#### input\_stream\_adapter::input\_stream\_adapter<!-- {{#callable:(anonymous)::namespace::input_stream_adapter::input_stream_adapter}} -->
The `input_stream_adapter` constructor initializes an adapter for an input stream, linking it to a given `std::istream`.
- **Inputs**:
    - `i`: A reference to a `std::istream` object that the adapter will use for input operations.
- **Control Flow**:
    - The constructor initializes the member variable `is` to point to the provided input stream `i`.
    - It also initializes the member variable `sb` with the stream buffer of the input stream, allowing for direct buffer operations.
- **Output**: The constructor does not return a value; it sets up the internal state of the `input_stream_adapter` object for subsequent input operations.
- **See also**: [`(anonymous)::namespace::input_stream_adapter`](#(anonymous)::namespace::input_stream_adapter)  (Data Structure)


---
#### input\_stream\_adapter::input\_stream\_adapter<!-- {{#callable:(anonymous)::namespace::input_stream_adapter::input_stream_adapter}} -->
The `input_stream_adapter` class provides an interface to adapt an `std::istream` for character and element retrieval.
- **Inputs**:
    - `i`: A reference to an `std::istream` object that the adapter will use for input operations.
- **Control Flow**:
    - The constructor initializes the adapter with a reference to the provided `std::istream` and its associated `std::streambuf`.
    - The destructor clears the stream flags of the associated input stream, preserving only the end-of-file state.
    - The move constructor transfers ownership of the stream and buffer pointers from the source object to the new object, nullifying the source's pointers.
    - The `get_character` method retrieves the next character from the stream buffer and sets the end-of-file flag if the end is reached.
    - The `get_elements` method reads a specified number of elements from the stream buffer into a destination array and sets the end-of-file flag if fewer elements are read than requested.
- **Output**: The class does not produce a direct output but provides methods to retrieve characters and elements from the associated input stream.
- **See also**: [`(anonymous)::namespace::input_stream_adapter`](#(anonymous)::namespace::input_stream_adapter)  (Data Structure)


---
#### input\_stream\_adapter::operator=<!-- {{#callable:(anonymous)::namespace::input_stream_adapter::operator=}} -->
The `operator=` function is deleted to prevent assignment of `input_stream_adapter` instances.
- **Inputs**: None
- **Control Flow**:
    - The function is marked as deleted, which means it cannot be called or used in any context.
    - This prevents copying or moving of `input_stream_adapter` objects, ensuring that the class maintains unique ownership of its resources.
- **Output**: The function does not produce an output as it is deleted and cannot be invoked.
- **See also**: [`(anonymous)::namespace::input_stream_adapter`](#(anonymous)::namespace::input_stream_adapter)  (Data Structure)


---
#### input\_stream\_adapter::input\_stream\_adapter<!-- {{#callable:(anonymous)::namespace::input_stream_adapter::input_stream_adapter}} -->
Moves the resources from one `input_stream_adapter` instance to another, leaving the source instance in a valid but empty state.
- **Inputs**:
    - `rhs`: An rvalue reference to another `input_stream_adapter` instance from which resources are being moved.
- **Control Flow**:
    - Initializes the current instance's `is` and `sb` members with the corresponding members from the `rhs` instance.
    - Sets the `is` and `sb` members of the `rhs` instance to `nullptr`, effectively transferring ownership of the resources.
- **Output**: The function does not return a value; it modifies the state of the current instance and the `rhs` instance.
- **See also**: [`(anonymous)::namespace::input_stream_adapter`](#(anonymous)::namespace::input_stream_adapter)  (Data Structure)


---
#### input\_stream\_adapter::get\_character<!-- {{#callable:(anonymous)::namespace::input_stream_adapter::get_character}} -->
`get_character` retrieves the next character from the associated input stream buffer.
- **Inputs**: None
- **Control Flow**:
    - The function calls `sb->sbumpc()` to read the next character from the stream buffer.
    - It checks if the result is equal to `std::char_traits<char>::eof()` to determine if the end of the stream has been reached.
    - If the end of the stream is reached, it manually sets the end-of-file state on the associated input stream using `is->clear()`.
- **Output**: The function returns the next character as an `int_type`, or `std::char_traits<char>::eof()` if the end of the stream is reached.
- **See also**: [`(anonymous)::namespace::input_stream_adapter`](#(anonymous)::namespace::input_stream_adapter)  (Data Structure)


---
#### input\_stream\_adapter::get\_elements<!-- {{#callable:(anonymous)::namespace::input_stream_adapter::get_elements}} -->
The `get_elements` function reads a specified number of elements from an input stream into a destination buffer.
- **Inputs**:
    - `dest`: A pointer to the destination buffer where the read elements will be stored.
    - `count`: The number of elements of type `T` to read from the input stream, defaulting to 1.
- **Control Flow**:
    - The function calls `sgetn` on the stream buffer to read `count` elements of type `T` into the `dest` buffer.
    - The result of the read operation is stored in `res`, which indicates the number of bytes successfully read.
    - If the number of bytes read is less than expected (i.e., less than `count * sizeof(T)`), the function sets the end-of-file (EOF) flag on the input stream.
- **Output**: The function returns the number of bytes actually read from the input stream into the destination buffer.
- **See also**: [`(anonymous)::namespace::input_stream_adapter`](#(anonymous)::namespace::input_stream_adapter)  (Data Structure)



---
### iterator\_input\_adapter<!-- {{#data_structure:(anonymous)::namespace::iterator_input_adapter}} -->
- **Type**: `class `iterator_input_adapter``
- **Members**:
    - `current`: An iterator representing the current position in the input range.
    - `end`: An iterator representing the end of the input range.
- **Description**: The `iterator_input_adapter` class is a template that adapts an input iterator to provide a character-based interface, allowing for the retrieval of characters and elements from a specified range defined by two iterators, `current` and `end`.
- **Member Functions**:
    - [`(anonymous)::namespace::iterator_input_adapter::iterator_input_adapter`](#(anonymous)::namespace::iterator_input_adapter::iterator_input_adapter)
    - [`(anonymous)::namespace::iterator_input_adapter::get_character`](#(anonymous)::namespace::iterator_input_adapter::get_character)
    - [`(anonymous)::namespace::iterator_input_adapter::get_elements`](#(anonymous)::namespace::iterator_input_adapter::get_elements)
    - [`(anonymous)::namespace::iterator_input_adapter::empty`](#(anonymous)::namespace::iterator_input_adapter::empty)

**Methods**

---
#### iterator\_input\_adapter::iterator\_input\_adapter<!-- {{#callable:(anonymous)::namespace::iterator_input_adapter::iterator_input_adapter}} -->
The `iterator_input_adapter` constructor initializes an adapter for a range of elements defined by two iterators.
- **Inputs**:
    - `first`: The starting iterator of the range to be adapted.
    - `last`: The ending iterator of the range to be adapted.
- **Control Flow**:
    - The constructor uses `std::move` to transfer ownership of the iterators `first` and `last` to the member variables `current` and `end` respectively.
    - No additional control flow or logic is present in the constructor, as it simply initializes the member variables.
- **Output**: The constructor does not return a value; it initializes an instance of `iterator_input_adapter` with the provided iterators.
- **See also**: [`(anonymous)::namespace::iterator_input_adapter`](#(anonymous)::namespace::iterator_input_adapter)  (Data Structure)


---
#### iterator\_input\_adapter::get\_character<!-- {{#callable:(anonymous)::namespace::iterator_input_adapter::get_character}} -->
The `get_character` function retrieves the next character from an iterator-based input range or returns an end-of-file indicator if the end is reached.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the `current` iterator is not equal to the `end` iterator using a likely branch hint.
    - If there are more characters to read, it converts the character pointed to by `current` to an integer type and advances the `current` iterator by one position.
    - If the end of the range is reached, it returns the end-of-file indicator.
- **Output**: The function returns the integer representation of the next character or an end-of-file indicator if there are no more characters to read.
- **See also**: [`(anonymous)::namespace::iterator_input_adapter`](#(anonymous)::namespace::iterator_input_adapter)  (Data Structure)


---
#### iterator\_input\_adapter::get\_elements<!-- {{#callable:(anonymous)::namespace::iterator_input_adapter::get_elements}} -->
The `get_elements` function reads a specified number of elements from an iterator and stores them in a destination buffer.
- **Inputs**:
    - `dest`: A pointer to the destination buffer where the elements will be stored.
    - `count`: The number of elements of type `T` to read from the iterator, defaulting to 1.
- **Control Flow**:
    - The function begins by casting the `dest` pointer to a `char*` type for byte-wise access.
    - A loop iterates from 0 to `count * sizeof(T)`, attempting to read each byte from the iterator.
    - Within the loop, it checks if the current iterator position is not at the end; if so, it reads the current element, advances the iterator, and stores the byte in the destination buffer.
    - If the end of the iterator is reached before reading the specified number of bytes, the function returns the number of bytes successfully read.
    - If the loop completes without reaching the end, it returns the total number of bytes requested, which is `count * sizeof(T)`.
- **Output**: The function returns the number of bytes read from the iterator, which can be less than the requested size if the end of the iterator is reached.
- **See also**: [`(anonymous)::namespace::iterator_input_adapter`](#(anonymous)::namespace::iterator_input_adapter)  (Data Structure)


---
#### iterator\_input\_adapter::empty<!-- {{#callable:(anonymous)::namespace::iterator_input_adapter::empty}} -->
Checks if the iterator input adapter is empty by comparing the current iterator position to the end position.
- **Inputs**: None
- **Control Flow**:
    - The function compares the `current` iterator with the `end` iterator.
    - If `current` is equal to `end`, it indicates that there are no more elements to iterate over.
- **Output**: Returns a boolean value: true if the iterator is at the end (empty), false otherwise.
- **See also**: [`(anonymous)::namespace::iterator_input_adapter`](#(anonymous)::namespace::iterator_input_adapter)  (Data Structure)



---
### wide\_string\_input\_adapter<!-- {{#data_structure:(anonymous)::namespace::wide_string_input_adapter}} -->
- **Type**: `class wide_string_input_adapter`
- **Members**:
    - `base_adapter`: An instance of the `BaseInputAdapter` type used for adapting input.
    - `utf8_bytes`: An array that holds UTF-8 byte values.
    - `utf8_bytes_index`: An index tracking the position of the next valid byte in the `utf8_bytes` array.
    - `utf8_bytes_filled`: A count of the number of valid bytes currently filled in the `utf8_bytes` array.
- **Description**: The `wide_string_input_adapter` class is a template-based input adapter designed to handle wide character strings by utilizing a base input adapter of type `BaseInputAdapter`. It manages a buffer of UTF-8 bytes and provides mechanisms to retrieve characters while ensuring that the buffer is filled as needed. This class is particularly useful for adapting wide string inputs into a format that can be processed as UTF-8.
- **Member Functions**:
    - [`(anonymous)::namespace::wide_string_input_adapter::wide_string_input_adapter`](#(anonymous)::namespace::wide_string_input_adapter::wide_string_input_adapter)
    - [`(anonymous)::namespace::wide_string_input_adapter::get_character`](#(anonymous)::namespace::wide_string_input_adapter::get_character)
    - [`(anonymous)::namespace::wide_string_input_adapter::get_elements`](#(anonymous)::namespace::wide_string_input_adapter::get_elements)
    - [`(anonymous)::namespace::wide_string_input_adapter::fill_buffer`](#(anonymous)::namespace::wide_string_input_adapter::fill_buffer)

**Methods**

---
#### wide\_string\_input\_adapter::wide\_string\_input\_adapter<!-- {{#callable:(anonymous)::namespace::wide_string_input_adapter::wide_string_input_adapter}} -->
The `wide_string_input_adapter` class is designed to adapt a base input for reading wide strings by managing a buffer of UTF-8 bytes.
- **Inputs**:
    - `base`: An instance of `BaseInputAdapter` that serves as the underlying input source for the adapter.
- **Control Flow**:
    - The constructor initializes the `base_adapter` member with the provided `base` input adapter.
    - The `get_character` method checks if the buffer needs to be filled; if so, it calls `fill_buffer` to populate the buffer with UTF-8 bytes.
    - Assertions are made to ensure that the buffer has been filled and that the index is within valid bounds before returning the next character.
    - The `get_elements` method throws a parse error if called, as wide string types cannot be interpreted as binary data.
- **Output**: The output of the `get_character` method is the next valid UTF-8 byte from the buffer, while the `get_elements` method does not produce a valid output but instead raises an error.
- **See also**: [`(anonymous)::namespace::wide_string_input_adapter`](#(anonymous)::namespace::wide_string_input_adapter)  (Data Structure)


---
#### wide\_string\_input\_adapter::get\_character<!-- {{#callable:(anonymous)::namespace::wide_string_input_adapter::get_character}} -->
The `get_character` function retrieves the next character from a UTF-8 byte buffer, filling the buffer if necessary.
- **Inputs**:
    - `none`: The function does not take any input arguments.
- **Control Flow**:
    - Checks if the current index `utf8_bytes_index` is equal to the number of filled bytes `utf8_bytes_filled` to determine if the buffer needs to be filled.
    - If the buffer needs to be filled, it calls the `fill_buffer` method to populate the `utf8_bytes` array.
    - Asserts that the buffer has been filled and that the index is reset to zero after filling.
    - Asserts that there are valid bytes in the buffer and that the current index is within bounds.
    - Returns the character at the current index and increments the index for the next call.
- **Output**: The function returns the next character as an `int_type` from the `utf8_bytes` array.
- **See also**: [`(anonymous)::namespace::wide_string_input_adapter`](#(anonymous)::namespace::wide_string_input_adapter)  (Data Structure)


---
#### wide\_string\_input\_adapter::get\_elements<!-- {{#callable:(anonymous)::namespace::wide_string_input_adapter::get_elements}} -->
The `get_elements` function throws a parse error indicating that wide string types cannot be interpreted as binary data.
- **Inputs**:
    - `dest`: A pointer to the destination where elements would be stored, though it is not used in this function.
    - `count`: A size_t value representing the number of elements to retrieve, defaulting to 1, but it is not utilized in this function.
- **Control Flow**:
    - The function immediately throws a `parse_error` exception with a specific error message.
    - No conditional logic or loops are present, as the function's sole purpose is to signal an error.
- **Output**: The function does not return a value; instead, it raises an exception to indicate an error in processing wide string types.
- **See also**: [`(anonymous)::namespace::wide_string_input_adapter`](#(anonymous)::namespace::wide_string_input_adapter)  (Data Structure)


---
#### wide\_string\_input\_adapter::fill\_buffer<!-- {{#callable:(anonymous)::namespace::wide_string_input_adapter::fill_buffer}} -->
The `fill_buffer` function populates a buffer with UTF-8 bytes using a helper class.
- **Inputs**:
    - `BaseInputAdapter`: The base input adapter used to read data.
    - `T`: A template parameter that determines the size of the wide character type.
- **Control Flow**:
    - The function calls `wide_string_input_helper<BaseInputAdapter, T>::fill_buffer` to fill the `utf8_bytes` buffer.
    - It uses the `base_adapter` to read data into the buffer, updating the `utf8_bytes_filled` and `utf8_bytes_index` accordingly.
- **Output**: The function does not return a value; it modifies the internal state of the `utf8_bytes` buffer and its associated indices.
- **See also**: [`(anonymous)::namespace::wide_string_input_adapter`](#(anonymous)::namespace::wide_string_input_adapter)  (Data Structure)



---
### iterator\_input\_adapter\_factory<!-- {{#data_structure:(anonymous)::namespace::iterator_input_adapter_factory}} -->
- **Type**: `struct`
- **Members**:
    - `iterator_type`: Defines the type of the iterator.
    - `char_type`: Defines the character type based on the iterator's value type.
    - `adapter_type`: Defines the type of the input adapter associated with the iterator.
- **Description**: The `iterator_input_adapter_factory` is a template struct designed to facilitate the creation of an `iterator_input_adapter` for a given iterator type, encapsulating the logic to derive the iterator's value type and providing a static method to instantiate the adapter.
- **Member Functions**:
    - [`(anonymous)::namespace::iterator_input_adapter_factory::create`](#(anonymous)::namespace::iterator_input_adapter_factory::create)

**Methods**

---
#### iterator\_input\_adapter\_factory::create<!-- {{#callable:(anonymous)::namespace::iterator_input_adapter_factory::create}} -->
The `create` function constructs an `adapter_type` object using two iterators.
- **Inputs**:
    - `first`: An iterator of type `IteratorType` representing the beginning of a range.
    - `last`: An iterator of type `IteratorType` representing the end of a range.
- **Control Flow**:
    - The function takes two iterators as input parameters.
    - It uses `std::move` to efficiently transfer ownership of the iterators to the `adapter_type` constructor.
    - Finally, it returns a newly constructed `adapter_type` object.
- **Output**: The function returns an instance of `adapter_type`, which is initialized with the provided iterators.
- **See also**: [`(anonymous)::namespace::iterator_input_adapter_factory`](#(anonymous)::namespace::iterator_input_adapter_factory)  (Data Structure)



---
### is\_iterator\_of\_multibyte<!-- {{#data_structure:(anonymous)::namespace::is_iterator_of_multibyte}} -->
- **Type**: `struct`
- **Members**:
    - `value_type`: Defines the type of value that the iterator `T` points to.
    - `value`: A compile-time constant that indicates if the `value_type` is a multibyte type.
- **Description**: The `is_iterator_of_multibyte` struct is a template that determines whether the type pointed to by an iterator `T` is a multibyte type, based on the size of its `value_type`. It utilizes `std::iterator_traits` to extract the `value_type` and defines a constant `value` that evaluates to true if the size of `value_type` is greater than one byte.


---
### container\_input\_adapter\_factory<!-- {{#data_structure:(anonymous)::namespace::container_input_adapter_factory_impl::container_input_adapter_factory}} -->
- **Type**: `struct`
- **Description**: `container_input_adapter_factory` is a template struct designed to facilitate the creation of input adapters for various container types, allowing for flexible and type-safe handling of different container implementations.


---
### span\_input\_adapter<!-- {{#data_structure:(anonymous)::namespace::span_input_adapter}} -->
- **Type**: `class`
- **Members**:
    - `ia`: An instance of `contiguous_bytes_input_adapter` that holds the input data.
- **Description**: The `span_input_adapter` class is designed to facilitate the adaptation of input data from either a pointer to a byte array or a range defined by two iterators, encapsulating this data within a `contiguous_bytes_input_adapter` instance for further processing.
- **Member Functions**:
    - [`(anonymous)::namespace::span_input_adapter::span_input_adapter`](#(anonymous)::namespace::span_input_adapter::span_input_adapter)
    - [`(anonymous)::namespace::span_input_adapter::span_input_adapter`](#(anonymous)::namespace::span_input_adapter::span_input_adapter)
    - [`(anonymous)::namespace::span_input_adapter::get`](#(anonymous)::namespace::span_input_adapter::get)

**Methods**

---
#### span\_input\_adapter::span\_input\_adapter<!-- {{#callable:(anonymous)::namespace::span_input_adapter::span_input_adapter}} -->
Constructs a `span_input_adapter` from a pointer to a byte array and its length.
- **Inputs**:
    - `b`: A pointer to a byte array, which must point to an integral type of size 1.
    - `l`: The length of the byte array, specified as a size_t.
- **Control Flow**:
    - The function checks if the type of `b` is a pointer to an integral type of size 1 using SFINAE (Substitution Failure Is Not An Error).
    - If the conditions are met, it initializes the member `ia` by casting the pointer `b` to a `const char*` and creating a range from `b` to `b + l`.
- **Output**: The constructor does not return a value but initializes the `span_input_adapter` object with a contiguous byte range.
- **See also**: [`(anonymous)::namespace::span_input_adapter`](#(anonymous)::namespace::span_input_adapter)  (Data Structure)


---
#### span\_input\_adapter::span\_input\_adapter<!-- {{#callable:(anonymous)::namespace::span_input_adapter::span_input_adapter}} -->
The `span_input_adapter` constructor initializes an instance using a range defined by two random access iterators.
- **Inputs**:
    - `first`: An iterator representing the beginning of the range.
    - `last`: An iterator representing the end of the range.
- **Control Flow**:
    - The constructor template is enabled only if the `IteratorType` is a random access iterator.
    - It initializes the member `ia` by calling the [`input_adapter`](#(anonymous)::namespace::input_adapter) constructor with the provided iterators.
- **Output**: The constructor does not return a value but initializes the `span_input_adapter` object with an [`input_adapter`](#(anonymous)::namespace::input_adapter) that encapsulates the range defined by the iterators.
- **Functions called**:
    - [`(anonymous)::namespace::input_adapter`](#(anonymous)::namespace::input_adapter)
- **See also**: [`(anonymous)::namespace::span_input_adapter`](#(anonymous)::namespace::span_input_adapter)  (Data Structure)


---
#### span\_input\_adapter::get<!-- {{#callable:(anonymous)::namespace::span_input_adapter::get}} -->
The `get` function returns an rvalue reference to the `ia` member of the `span_input_adapter` class.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the result of `std::move(ia)`, which casts `ia` to an rvalue reference.
    - No conditional statements or loops are present, making the control flow straightforward.
- **Output**: The output is an rvalue reference to a `contiguous_bytes_input_adapter`, allowing for efficient transfer of ownership of the `ia` member.
- **See also**: [`(anonymous)::namespace::span_input_adapter`](#(anonymous)::namespace::span_input_adapter)  (Data Structure)



---
### lexer\_base<!-- {{#data_structure:(anonymous)::namespace::lexer_base}} -->
- **Type**: `class`
- **Members**:
    - `token_type`: An enumeration representing various token types for the parser.
- **Description**: The `lexer_base` class template defines a base lexer that utilizes an enumeration `token_type` to categorize different types of tokens that can be encountered during parsing, such as literals, structural characters, and error indicators.
- **Member Functions**:
    - [`(anonymous)::namespace::lexer_base::token_type_name`](#(anonymous)::namespace::lexer_base::token_type_name)

**Methods**

---
#### lexer\_base::token\_type\_name<!-- {{#callable:(anonymous)::namespace::lexer_base::token_type_name}} -->
The `token_type_name` function returns a string representation of a given `token_type` enumeration value.
- **Inputs**:
    - `t`: An enumeration value of type `token_type` representing different types of tokens.
- **Control Flow**:
    - The function uses a `switch` statement to evaluate the input token type `t`.
    - For each case in the `switch`, it returns a corresponding string that describes the token type.
    - If the token type does not match any defined cases, it defaults to returning 'unknown token'.
- **Output**: The function outputs a constant string that describes the type of the token represented by the input `token_type` value.
- **See also**: [`(anonymous)::namespace::lexer_base`](#(anonymous)::namespace::lexer_base)  (Data Structure)



---
### token\_type<!-- {{#data_structure:(anonymous)::namespace::lexer_base::token_type}} -->
- **Type**: `enum class`
- **Members**:
    - `uninitialized`: Indicates the scanner is uninitialized.
    - `literal_true`: Represents the `true` literal.
    - `literal_false`: Represents the `false` literal.
    - `literal_null`: Represents the `null` literal.
    - `value_string`: Represents a string, with actual value accessed via `get_string()`.
    - `value_unsigned`: Represents an unsigned integer, with actual value accessed via `get_number_unsigned()`.
    - `value_integer`: Represents a signed integer, with actual value accessed via `get_number_integer()`.
    - `value_float`: Represents a floating point number, with actual value accessed via `get_number_float()`.
    - `begin_array`: Represents the character for array begin `[`.
    - `begin_object`: Represents the character for object begin `{`.
    - `end_array`: Represents the character for array end `]`.
    - `end_object`: Represents the character for object end `}`.
    - `name_separator`: Represents the name separator `:`.
    - `value_separator`: Represents the value separator `,`.
    - `parse_error`: Indicates a parse error.
    - `end_of_input`: Indicates the end of the input buffer.
    - `literal_or_value`: Indicates a literal or the beginning of a value, used only for diagnostics.
- **Description**: The `token_type` enum class defines a set of constants representing various types of tokens that can be encountered during the parsing of input, including literals, values, structural characters, and error indicators.


---
### lexer<!-- {{#data_structure:(anonymous)::namespace::lexer}} -->
- **Type**: `class`
- **Members**:
    - `ia`: An instance of the input adapter used for reading input.
    - `ignore_comments`: A boolean flag indicating whether comments should be ignored.
    - `decimal_point_char`: A character representing the locale-dependent decimal point.
- **Description**: The `lexer` class is a template-based lexical analyzer designed to parse JSON-like input, utilizing an input adapter for reading characters and providing functionality to handle various token types, including numbers, strings, and comments, while managing locale-specific details such as decimal points.
- **Member Functions**:
    - [`(anonymous)::namespace::lexer::lexer`](#(anonymous)::namespace::lexer::lexer)
    - [`(anonymous)::namespace::lexer::lexer`](#(anonymous)::namespace::lexer::lexer)
    - [`(anonymous)::namespace::lexer::lexer`](#(anonymous)::namespace::lexer::lexer)
    - [`(anonymous)::namespace::lexer::operator=`](#(anonymous)::namespace::lexer::operator=)
    - [`(anonymous)::namespace::lexer::operator=`](#(anonymous)::namespace::lexer::operator=)
    - [`(anonymous)::namespace::lexer::~lexer`](#(anonymous)::namespace::lexer::~lexer)
    - [`(anonymous)::namespace::lexer::get_decimal_point`](#(anonymous)::namespace::lexer::get_decimal_point)
    - [`(anonymous)::namespace::lexer::get_codepoint`](#(anonymous)::namespace::lexer::get_codepoint)
    - [`(anonymous)::namespace::lexer::next_byte_in_range`](#(anonymous)::namespace::lexer::next_byte_in_range)
    - [`(anonymous)::namespace::lexer::scan_string`](#(anonymous)::namespace::lexer::scan_string)
    - [`(anonymous)::namespace::lexer::scan_comment`](#(anonymous)::namespace::lexer::scan_comment)
    - [`(anonymous)::namespace::lexer::strtof`](#(anonymous)::namespace::lexer::strtof)
    - [`(anonymous)::namespace::lexer::strtof`](#(anonymous)::namespace::lexer::strtof)
    - [`(anonymous)::namespace::lexer::strtof`](#(anonymous)::namespace::lexer::strtof)
    - [`(anonymous)::namespace::lexer::scan_number`](#(anonymous)::namespace::lexer::scan_number)
- **Inherits From**:
    - [`(anonymous)::namespace::lexer_base`](#(anonymous)::namespace::lexer_base)

**Methods**

---
#### lexer::lexer<!-- {{#callable:(anonymous)::namespace::lexer::lexer}} -->
Constructs a `lexer` object with an input adapter and an option to ignore comments.
- **Inputs**:
    - `adapter`: An rvalue reference to an `InputAdapterType` object that provides the input for the lexer.
    - `ignore_comments_`: A boolean flag indicating whether to ignore comments in the input; defaults to false.
- **Control Flow**:
    - The constructor initializes the member variables `ia`, `ignore_comments`, and `decimal_point_char`.
    - The `adapter` is moved into the member variable `ia`, ensuring efficient transfer of resources.
    - The `ignore_comments` variable is set based on the provided argument.
    - The `decimal_point_char` is initialized by calling the static method `get_decimal_point()` to retrieve the locale-specific decimal point character.
- **Output**: The constructor does not return a value but initializes the `lexer` object for further use in parsing input.
- **Functions called**:
    - [`(anonymous)::namespace::lexer::get_decimal_point`](#(anonymous)::namespace::lexer::get_decimal_point)
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::lexer<!-- {{#callable:(anonymous)::namespace::lexer::lexer}} -->
The `lexer` class constructor is deleted for copy operations and defaulted for move operations.
- **Inputs**: None
- **Control Flow**:
    - The copy constructor is deleted to prevent copying of `lexer` instances due to pointer members.
    - The move constructor is defaulted, allowing efficient transfer of resources from one `lexer` instance to another.
- **Output**: The constructor does not return a value but ensures that `lexer` instances cannot be copied, only moved.
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::lexer<!-- {{#callable:(anonymous)::namespace::lexer::lexer}} -->
The `lexer` class implements a move constructor that allows for the efficient transfer of resources from one `lexer` instance to another.
- **Inputs**: None
- **Control Flow**:
    - The move constructor is defined using the default keyword, which means it will perform a member-wise move of the resources.
    - The move constructor is marked as noexcept, indicating that it will not throw exceptions during its execution.
- **Output**: The output of the move constructor is an instance of the `lexer` class that has taken ownership of the resources from the moved-from instance.
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::operator=<!-- {{#callable:(anonymous)::namespace::lexer::operator=}} -->
Moves the state of one `lexer` instance to another, effectively transferring resources.
- **Inputs**:
    - `other`: An rvalue reference to another `lexer` instance from which the state will be moved.
- **Control Flow**:
    - The function is defined as a default move assignment operator, which means it uses the compiler-generated implementation.
    - It transfers the resources from the `other` instance to the current instance, leaving `other` in a valid but unspecified state.
- **Output**: Returns a reference to the current `lexer` instance after the move operation.
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::\~lexer<!-- {{#callable:(anonymous)::namespace::lexer::~lexer}} -->
The `~lexer` function is a default destructor for the `lexer` class.
- **Inputs**: None
- **Control Flow**:
    - The destructor does not contain any specific logic and relies on the default behavior provided by the compiler.
    - It is implicitly defined to clean up resources when an instance of the `lexer` class is destroyed.
- **Output**: The function does not return any value; it simply ensures that the destructor of the base class and any member variables are properly cleaned up.
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::get\_decimal\_point<!-- {{#callable:(anonymous)::namespace::lexer::get_decimal_point}} -->
Returns the locale-dependent decimal point character.
- **Inputs**:
    - `none`: This function does not take any input arguments.
- **Control Flow**:
    - Calls `localeconv()` to retrieve the current locale's numeric formatting information.
    - Asserts that the returned pointer from `localeconv()` is not null.
    - Checks if the `decimal_point` field of the locale structure is null; if so, it returns '.' as the default decimal point.
    - If `decimal_point` is not null, it dereferences the pointer and returns the character it points to.
- **Output**: Returns a `char` representing the decimal point character for the current locale, or '.' if the locale does not specify one.
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::get\_codepoint<!-- {{#callable:(anonymous)::namespace::lexer::get_codepoint}} -->
Extracts a Unicode codepoint from a sequence of four hexadecimal digits following a '\u' escape.
- **Inputs**:
    - `current`: A character representing the current input character being processed, expected to be 'u' when this function is called.
- **Control Flow**:
    - The function begins by asserting that the current character is 'u'.
    - It initializes a variable `codepoint` to 0.
    - A loop iterates over a predefined set of bit-shift factors (12, 8, 4, 0) to process four hexadecimal digits.
    - Within the loop, the function calls `get()` to read the next character.
    - It checks if the character is a valid hexadecimal digit ('0'-'9', 'A'-'F', 'a'-'f') and updates the `codepoint` accordingly.
    - If an invalid character is encountered, the function returns -1.
    - After processing all four digits, it asserts that the resulting `codepoint` is within the valid range (0x0000 to 0xFFFF) and returns the `codepoint`.
- **Output**: Returns the calculated Unicode codepoint as an integer in the range 0x0000 to 0xFFFF, or -1 if an error occurs (e.g., invalid character or EOF).
- **Functions called**:
    - [`get`](#get)
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::next\_byte\_in\_range<!-- {{#callable:(anonymous)::namespace::lexer::next_byte_in_range}} -->
Checks if the current byte is within specified ranges and adds it if valid.
- **Inputs**:
    - `ranges`: An initializer list of `char_int_type` representing pairs of inclusive lower and upper bounds.
- **Control Flow**:
    - Asserts that the size of `ranges` is either 2, 4, or 6, ensuring valid input.
    - Adds the current byte to an internal structure.
    - Iterates through the `ranges`, checking if the current byte falls within each specified range.
    - If the current byte is within a range, it is added; otherwise, an error message is set and the function returns false.
    - If all checks pass, the function returns true.
- **Output**: Returns true if the current byte is within the specified ranges; otherwise, returns false and sets an error message.
- **Functions called**:
    - [`add`](#add)
    - [`get`](#get)
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::scan\_string<!-- {{#callable:(anonymous)::namespace::lexer::scan_string}} -->
Scans a JSON string literal, handling escape sequences and Unicode characters.
- **Inputs**:
    - `none`: This function does not take any input parameters directly, but relies on the internal state of the lexer class.
- **Control Flow**:
    - The function begins by resetting the token buffer and asserting that the current character is an opening quote.
    - It enters a loop where it reads characters one by one until it encounters a closing quote or an error.
    - If an EOF is encountered before a closing quote, it sets an error message and returns a parse error token.
    - If a closing quote is found, it returns a token indicating a valid string.
    - If a backslash is encountered, it processes escape sequences, including handling Unicode escapes.
    - For each character read, it checks for valid control characters and handles them appropriately, returning errors for invalid sequences.
    - The function continues until it either successfully reads a complete string or encounters an error.
- **Output**: Returns a token type indicating whether the string was successfully scanned (token_type::value_string) or if a parse error occurred (token_type::parse_error).
- **Functions called**:
    - [`reset`](#reset)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`add`](#add)
    - [`(anonymous)::namespace::lexer::get_codepoint`](#(anonymous)::namespace::lexer::get_codepoint)
    - [`(anonymous)::namespace::lexer::next_byte_in_range`](#(anonymous)::namespace::lexer::next_byte_in_range)
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::scan\_comment<!-- {{#callable:(anonymous)::namespace::lexer::scan_comment}} -->
Scans and processes comments in a source code, handling both single-line and multi-line comments.
- **Inputs**:
    - `none`: The function does not take any input parameters.
- **Control Flow**:
    - The function begins by calling `get()` to retrieve the next character.
    - If the character is '/', it enters a loop to skip characters until a newline or EOF is encountered, returning true when done.
    - If the character is '*', it enters a loop to skip characters until '*/' is found, returning true if found, or returning false with an error message if EOF is reached without finding '*/'.
    - If the character is neither '/' nor '*', it sets an error message indicating an invalid comment and returns false.
- **Output**: Returns true if the comment is successfully scanned, or false if there is an error (e.g., missing closing '*/' for multi-line comments or invalid comment syntax).
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`unget`](#unget)
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::strtof<!-- {{#callable:(anonymous)::namespace::lexer::strtof}} -->
Converts a C-style string to a floating-point number and updates the reference to the result.
- **Inputs**:
    - `f`: A reference to a `float` variable where the converted value will be stored.
    - `str`: A pointer to a null-terminated C-style string that represents the floating-point number to be converted.
    - `endptr`: A pointer to a pointer that will be set to point to the character after the last character used in the conversion.
- **Control Flow**:
    - The function calls `std::strtof`, which performs the actual conversion from the string to a float.
    - The result of the conversion is assigned to the reference variable `f`.
    - If the conversion is successful, `endptr` will point to the character following the last character used in the conversion.
- **Output**: The function does not return a value; instead, it modifies the reference `f` to hold the converted float value.
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::strtof<!-- {{#callable:(anonymous)::namespace::lexer::strtof}} -->
Converts a string to a double-precision floating-point number.
- **Inputs**:
    - `f`: A reference to a `double` variable where the converted value will be stored.
    - `str`: A pointer to a null-terminated string containing the representation of a floating-point number.
    - `endptr`: A pointer to a pointer that will be set to point to the character after the last character used in the conversion.
- **Control Flow**:
    - The function calls `std::strtod` to perform the conversion from string to double.
    - The result of the conversion is assigned to the reference variable `f`.
- **Output**: The function does not return a value; instead, it modifies the `f` variable to hold the converted double value.
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::strtof<!-- {{#callable:(anonymous)::namespace::lexer::strtof}} -->
Converts a string to a `long double` value using `std::strtold`.
- **Inputs**:
    - `f`: A reference to a `long double` variable where the converted value will be stored.
    - `str`: A pointer to a null-terminated string containing the representation of the floating-point number.
    - `endptr`: A pointer to a pointer that will be set to point to the character after the last character used in the conversion.
- **Control Flow**:
    - The function calls `std::strtold` to perform the conversion from string to `long double`.
    - The result of the conversion is assigned to the variable referenced by `f`.
- **Output**: The function does not return a value; instead, it modifies the `long double` referenced by `f` and updates `endptr` to point to the next character in the string after the number.
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)


---
#### lexer::scan\_number<!-- {{#callable:(anonymous)::namespace::lexer::scan_number}} -->
Scans and parses a number from the input stream, returning its type.
- **Inputs**:
    - `current`: The current character being processed from the input stream.
- **Control Flow**:
    - The function uses a state machine implemented with `goto` statements to handle different parsing states.
    - It starts by resetting the `token_buffer` and determining the initial state based on the first character.
    - Depending on the character, it transitions to different states for parsing negative numbers, zeros, digits, decimals, and exponents.
    - Each state processes the input character and may transition to another state or return an error if the input is invalid.
    - The function concludes by attempting to convert the accumulated string in `token_buffer` to an integer or float, returning the appropriate token type.
- **Output**: Returns `token_type::value_unsigned`, `token_type::value_integer`, or `token_type::value_float` if the number is successfully parsed, or `token_type::parse_error` if an error occurs.
- **Functions called**:
    - [`reset`](#reset)
    - [`add`](#add)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`unget`](#unget)
    - [`(anonymous)::namespace::lexer::strtof`](#(anonymous)::namespace::lexer::strtof)
- **See also**: [`(anonymous)::namespace::lexer`](#(anonymous)::namespace::lexer)  (Data Structure)



---
### json\_sax\_dom\_parser<!-- {{#data_structure:detail::json_sax_dom_parser}} -->
- **Type**: `class`
- **Members**:
    - `root`: A reference to the parsed JSON value.
    - `ref_stack`: A stack to model the hierarchy of JSON values.
    - `object_element`: A pointer to hold the reference for the next object element.
    - `errored`: A boolean indicating whether a syntax error occurred.
    - `allow_exceptions`: A boolean indicating whether to throw exceptions on errors.
    - `m_lexer_ref`: A reference to the lexer for obtaining the current position.
- **Description**: The `json_sax_dom_parser` class is a template-based SAX parser for JSON data that constructs a DOM-like representation of the parsed JSON structure. It utilizes a stack to manage the hierarchy of JSON objects and arrays, allowing for the dynamic handling of various JSON value types such as null, boolean, numbers, strings, and binary data. The class also provides options for error handling, including the ability to throw exceptions on parse errors, and maintains diagnostic positions for better error reporting.
- **Member Functions**:
    - [`detail::json_sax_dom_parser::json_sax_dom_parser`](#json_sax_dom_parserjson_sax_dom_parser)
    - [`detail::json_sax_dom_parser::json_sax_dom_parser`](#json_sax_dom_parserjson_sax_dom_parser)
    - [`detail::json_sax_dom_parser::json_sax_dom_parser`](#json_sax_dom_parserjson_sax_dom_parser)
    - [`detail::json_sax_dom_parser::operator=`](#json_sax_dom_parseroperator=)
    - [`detail::json_sax_dom_parser::operator=`](#json_sax_dom_parseroperator=)
    - [`detail::json_sax_dom_parser::~json_sax_dom_parser`](#json_sax_dom_parserjson_sax_dom_parser)
    - [`detail::json_sax_dom_parser::null`](#json_sax_dom_parsernull)
    - [`detail::json_sax_dom_parser::boolean`](#json_sax_dom_parserboolean)
    - [`detail::json_sax_dom_parser::number_integer`](#json_sax_dom_parsernumber_integer)
    - [`detail::json_sax_dom_parser::number_unsigned`](#json_sax_dom_parsernumber_unsigned)
    - [`detail::json_sax_dom_parser::number_float`](#json_sax_dom_parsernumber_float)
    - [`detail::json_sax_dom_parser::string`](#json_sax_dom_parserstring)
    - [`detail::json_sax_dom_parser::binary`](#json_sax_dom_parserbinary)
    - [`detail::json_sax_dom_parser::start_object`](#json_sax_dom_parserstart_object)
    - [`detail::json_sax_dom_parser::key`](#json_sax_dom_parserkey)
    - [`detail::json_sax_dom_parser::end_object`](#json_sax_dom_parserend_object)
    - [`detail::json_sax_dom_parser::start_array`](#json_sax_dom_parserstart_array)
    - [`detail::json_sax_dom_parser::end_array`](#json_sax_dom_parserend_array)
    - [`detail::json_sax_dom_parser::parse_error`](#json_sax_dom_parserparse_error)
    - [`detail::json_sax_dom_parser::is_errored`](#json_sax_dom_parseris_errored)
    - [`detail::json_sax_dom_parser::handle_diagnostic_positions_for_json_value`](#json_sax_dom_parserhandle_diagnostic_positions_for_json_value)

**Methods**

---
#### json\_sax\_dom\_parser::json\_sax\_dom\_parser<!-- {{#callable:detail::json_sax_dom_parser::json_sax_dom_parser}} -->
The `json_sax_dom_parser` constructor initializes a SAX parser for JSON data, allowing for the manipulation of a JSON value during parsing.
- **Inputs**:
    - `r`: A reference to a `BasicJsonType` object that will be manipulated while parsing.
    - `allow_exceptions_`: A boolean flag indicating whether to throw exceptions on parse errors (default is true).
    - `lexer_`: An optional pointer to a `lexer_t` object used for tracking the current position in the input.
- **Control Flow**:
    - The constructor initializes member variables with the provided arguments.
    - It sets the `root` to the reference of the JSON object to be parsed.
    - It configures whether exceptions should be thrown on errors based on the `allow_exceptions_` parameter.
    - If a lexer is provided, it is stored for later use in tracking positions during parsing.
- **Output**: The constructor does not return a value but sets up the `json_sax_dom_parser` instance for parsing JSON data.
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::json\_sax\_dom\_parser<!-- {{#callable:detail::json_sax_dom_parser::json_sax_dom_parser}} -->
The `json_sax_dom_parser` class is designed to parse JSON data into a DOM structure using a SAX (Simple API for XML) approach.
- **Inputs**:
    - `BasicJsonType& r`: A reference to a JSON value that will be manipulated during parsing.
    - `const bool allow_exceptions_`: A boolean flag indicating whether to throw exceptions on parse errors.
    - `lexer_t* lexer_`: An optional pointer to a lexer used for tracking the current position in the input.
- **Control Flow**:
    - The constructor initializes the parser with a reference to a JSON value and sets the exception handling behavior.
    - The class is designed to be move-only, preventing copying to ensure unique ownership of the parser state.
    - Various methods handle different JSON value types (null, boolean, number, string, binary) by calling `handle_value` to store the parsed value.
    - Methods for starting and ending objects and arrays manage the reference stack to maintain the hierarchy of parsed values.
    - Error handling is performed in the `parse_error` method, which can throw exceptions based on the `allow_exceptions` flag.
- **Output**: The output is a structured representation of the parsed JSON data, stored in the `root` member of the class, which can be accessed after parsing.
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::json\_sax\_dom\_parser<!-- {{#callable:detail::json_sax_dom_parser::json_sax_dom_parser}} -->
`json_sax_dom_parser` is a class designed to parse JSON data using a SAX (Simple API for XML) style approach, allowing for the construction of a DOM (Document Object Model) representation of the JSON.
- **Inputs**:
    - `r`: A reference to a `BasicJsonType` object that will be manipulated during the parsing process.
    - `allow_exceptions_`: A boolean flag indicating whether to throw exceptions on parse errors.
    - `lexer_`: An optional pointer to a `lexer_t` object used for tracking the current position in the input.
- **Control Flow**:
    - The constructor initializes the parser with a reference to a JSON value, a flag for exception handling, and an optional lexer reference.
    - The class is designed to be move-only, preventing copying of parser instances.
    - Various methods handle different JSON value types (null, boolean, number, string, binary) by calling `handle_value` to process and store the parsed values.
    - The parser maintains a stack (`ref_stack`) to manage the hierarchy of JSON objects and arrays being parsed.
    - Methods for starting and ending objects and arrays manage the stack and ensure that the structure of the JSON is correctly represented.
    - Error handling is performed through the `parse_error` method, which can throw exceptions based on the `allow_exceptions` flag.
- **Output**: The output of the parsing process is a constructed JSON value stored in the `root` member of the class, which represents the entire JSON structure parsed from the input.
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::\~json\_sax\_dom\_parser<!-- {{#callable:detail::json_sax_dom_parser::~json_sax_dom_parser}} -->
The `~json_sax_dom_parser` is a default destructor for the `json_sax_dom_parser` class, which is responsible for parsing JSON data into a DOM structure.
- **Inputs**: None
- **Control Flow**:
    - The destructor is implicitly defined as default, meaning it will automatically clean up resources when an instance of `json_sax_dom_parser` goes out of scope.
    - No specific control flow logic is implemented in the destructor, as it relies on the default behavior provided by the compiler.
- **Output**: The output of the destructor is the successful deallocation of resources associated with the `json_sax_dom_parser` instance, ensuring proper cleanup without any return value.
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::null<!-- {{#callable:detail::json_sax_dom_parser::null}} -->
The `null` function handles the parsing of a JSON null value.
- **Inputs**:
    - `none`: This function does not take any input arguments.
- **Control Flow**:
    - Calls the [`handle_value`](#json_sax_dom_callback_parserhandle_value) function with a `nullptr` to represent a JSON null value.
    - Returns `true` to indicate successful handling of the null value.
- **Output**: The function returns a boolean value `true`, indicating that the null value was successfully processed.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::boolean<!-- {{#callable:detail::json_sax_dom_parser::boolean}} -->
Processes a boolean value and returns true.
- **Inputs**:
    - `val`: A boolean value that is to be processed.
- **Control Flow**:
    - Calls the [`handle_value`](#json_sax_dom_callback_parserhandle_value) function with the input boolean value `val`.
    - Returns true after processing the value.
- **Output**: Always returns true, indicating successful processing of the boolean value.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::number\_integer<!-- {{#callable:detail::json_sax_dom_parser::number_integer}} -->
The `number_integer` function processes an integer value and handles it using the [`handle_value`](#json_sax_dom_callback_parserhandle_value) method.
- **Inputs**:
    - `val`: An integer value of type `number_integer_t` that is to be processed.
- **Control Flow**:
    - The function calls `handle_value(val)` to process the input integer value.
    - After processing, it returns `true` to indicate successful handling.
- **Output**: The function returns a boolean value, which is always `true` in this implementation, indicating that the integer value was handled successfully.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::number\_unsigned<!-- {{#callable:detail::json_sax_dom_parser::number_unsigned}} -->
The `number_unsigned` function processes an unsigned number value and returns true.
- **Inputs**:
    - `val`: An unsigned number of type `number_unsigned_t` that is to be handled.
- **Control Flow**:
    - The function calls `handle_value(val)` to process the input value.
    - After processing, it returns true, indicating successful handling.
- **Output**: The function returns a boolean value, which is always true in this implementation.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::number\_float<!-- {{#callable:detail::json_sax_dom_parser::number_float}} -->
The `number_float` function processes a floating-point number value and handles it appropriately.
- **Inputs**:
    - `val`: A floating-point number of type `number_float_t` that is to be processed.
    - `unused`: A constant reference to a string of type `string_t`, which is not used in the function.
- **Control Flow**:
    - The function calls `handle_value(val)` to process the floating-point number.
    - After processing, it returns `true` to indicate successful handling.
- **Output**: The function returns a boolean value `true`, indicating that the floating-point number was successfully processed.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::string<!-- {{#callable:detail::json_sax_dom_parser::string}} -->
The `string` function processes a string value by passing it to the [`handle_value`](#json_sax_dom_callback_parserhandle_value) method.
- **Inputs**:
    - `val`: A reference to a string of type `string_t` that is to be processed.
- **Control Flow**:
    - The function calls [`handle_value`](#json_sax_dom_callback_parserhandle_value) with the provided string reference `val`.
    - After processing, it returns `true` to indicate successful handling.
- **Output**: The function returns a boolean value, which is always `true` in this implementation.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::binary<!-- {{#callable:detail::json_sax_dom_parser::binary}} -->
The `binary` function processes a binary value by moving it and handling it through a dedicated function.
- **Inputs**:
    - `val`: A reference to a `binary_t` type value that is to be processed.
- **Control Flow**:
    - The function calls [`handle_value`](#json_sax_dom_callback_parserhandle_value) with the moved `val`, which transfers ownership of the binary data.
    - After processing the value, the function returns `true` to indicate successful handling.
- **Output**: The function returns a boolean value, specifically `true`, indicating that the binary value was successfully processed.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::start\_object<!-- {{#callable:detail::json_sax_dom_parser::start_object}} -->
The `start_object` function initializes a new JSON object in the parser and manages its size constraints.
- **Inputs**:
    - `len`: A `std::size_t` representing the expected length of the JSON object.
- **Control Flow**:
    - Pushes a new object onto the `ref_stack` using [`handle_value`](#json_sax_dom_callback_parserhandle_value).
    - If `JSON_DIAGNOSTIC_POSITIONS` is defined, it sets the start position of the object based on the lexer position.
    - Checks if the provided length exceeds the maximum size allowed for the object, throwing an exception if it does.
- **Output**: Returns `true` to indicate successful initialization of the object.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::key<!-- {{#callable:detail::json_sax_dom_parser::key}} -->
The `key` function adds a null value at a specified key in the current JSON object and stores a reference to that key for later use.
- **Inputs**:
    - `val`: A reference to a string (`string_t&`) that represents the key to be added to the current JSON object.
- **Control Flow**:
    - The function first checks if the `ref_stack` is not empty to ensure there is a current context for adding a key.
    - It then asserts that the last element in the `ref_stack` is an object, confirming that keys can be added.
    - The function uses the key provided in `val` to add a null value to the current JSON object and stores a reference to this key in `object_element`.
- **Output**: The function returns `true` to indicate that the key has been successfully added.
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::end\_object<!-- {{#callable:detail::json_sax_dom_parser::end_object}} -->
The `end_object` function finalizes the parsing of a JSON object by updating its end position and removing it from the reference stack.
- **Inputs**:
    - `none`: The function does not take any input arguments.
- **Control Flow**:
    - The function asserts that the reference stack is not empty and that the top element is an object.
    - If diagnostic positions are enabled, it updates the end position of the current object using the lexer reference.
    - It calls `set_parents` on the current object to update its parent references.
    - Finally, it removes the current object from the reference stack.
- **Output**: The function returns a boolean value `true`, indicating successful completion of the operation.
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::start\_array<!-- {{#callable:detail::json_sax_dom_parser::start_array}} -->
This function initiates the parsing of a JSON array and checks for size constraints.
- **Inputs**:
    - `len`: The expected length of the array to be parsed.
- **Control Flow**:
    - Pushes a new array value onto the `ref_stack` using [`handle_value`](#json_sax_dom_callback_parserhandle_value).
    - If `JSON_DIAGNOSTIC_POSITIONS` is defined, it sets the start position of the array based on the current lexer position.
    - Checks if the provided length exceeds the maximum size allowed for the array, throwing an exception if it does.
- **Output**: Returns true to indicate successful initiation of the array parsing.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::end\_array<!-- {{#callable:detail::json_sax_dom_parser::end_array}} -->
The `end_array` function finalizes the parsing of a JSON array by updating its end position and removing it from the reference stack.
- **Inputs**:
    - `none`: The function does not take any input parameters.
- **Control Flow**:
    - The function first asserts that the reference stack is not empty and that the last element in the stack is an array.
    - If diagnostic positions are enabled, it updates the end position of the array based on the current lexer position.
    - It then calls `set_parents` on the last element of the stack to update its parent references.
    - Finally, it removes the last element from the reference stack and returns true.
- **Output**: The function returns a boolean value, always true, indicating successful completion of the array parsing.
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::parse\_error<!-- {{#callable:detail::json_sax_dom_parser::parse_error}} -->
Handles parsing errors by setting an error flag and optionally throwing an exception.
- **Inputs**:
    - `unused_size`: A size_t parameter that is unused in the function.
    - `unused_string`: A string parameter that is unused in the function.
    - `ex`: An exception object that contains information about the parsing error.
- **Control Flow**:
    - Sets the `errored` flag to true to indicate that a parsing error has occurred.
    - Uses `static_cast<void>(ex)` to suppress unused variable warnings for the exception parameter.
    - Checks the `allow_exceptions` flag; if true, it throws the exception using `JSON_THROW(ex)`.
    - Returns false to indicate that the parsing operation was unsuccessful.
- **Output**: Returns a boolean value, specifically false, indicating that a parsing error occurred.
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::is\_errored<!-- {{#callable:detail::json_sax_dom_parser::is_errored}} -->
Checks if a syntax error occurred during JSON parsing.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the member variable `errored`, which indicates if an error has occurred.
- **Output**: Returns a boolean value: true if an error occurred during parsing, false otherwise.
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)


---
#### json\_sax\_dom\_parser::handle\_diagnostic\_positions\_for\_json\_value<!-- {{#callable:detail::json_sax_dom_parser::handle_diagnostic_positions_for_json_value}} -->
Handles the diagnostic positions for a JSON value during parsing.
- **Inputs**:
    - `v`: A reference to a `BasicJsonType` object representing the JSON value whose diagnostic positions are being set.
- **Control Flow**:
    - Checks if the lexer reference `m_lexer_ref` is valid.
    - Sets the `end_position` of the JSON value `v` to the current position of the lexer.
    - Uses a switch statement to determine the type of the JSON value and calculates the `start_position` based on its type.
    - Handles specific cases for boolean, null, string, binary, and numeric types to set the correct start position.
    - For object and array types, it skips setting positions as they are handled elsewhere.
    - Includes a default case that asserts false, indicating an unexpected type.
- **Output**: The function does not return a value but modifies the `start_position` and `end_position` attributes of the input JSON value `v`.
- **See also**: [`detail::json_sax_dom_parser`](#detailjson_sax_dom_parser)  (Data Structure)



---
### json\_sax\_dom\_callback\_parser<!-- {{#data_structure:detail::json_sax_dom_callback_parser}} -->
- **Type**: `class`
- **Members**:
    - `root`: The parsed JSON value.
    - `ref_stack`: Stack to model the hierarchy of values.
    - `keep_stack`: Stack to manage which values to keep.
    - `key_keep_stack`: Stack to manage which object keys to keep.
    - `object_element`: Helper to hold the reference for the next object element.
    - `errored`: Indicates whether a syntax error occurred.
    - `callback`: Callback function for handling parsing events.
    - `allow_exceptions`: Flag to determine if exceptions should be thrown on errors.
    - `discarded`: A discarded value for the callback.
    - `m_lexer_ref`: Reference to the lexer for obtaining the current position.
- **Description**: The `json_sax_dom_callback_parser` class is a template-based SAX parser designed to parse JSON data into a DOM-like structure, utilizing callback functions to handle various parsing events such as the start and end of objects and arrays, as well as individual values. It maintains a stack to track the hierarchy of parsed values and manages which values should be retained or discarded based on the callback's responses. The class also provides mechanisms for error handling and diagnostic position tracking during parsing.
- **Member Functions**:
    - [`detail::json_sax_dom_callback_parser::json_sax_dom_callback_parser`](#json_sax_dom_callback_parserjson_sax_dom_callback_parser)
    - [`detail::json_sax_dom_callback_parser::json_sax_dom_callback_parser`](#json_sax_dom_callback_parserjson_sax_dom_callback_parser)
    - [`detail::json_sax_dom_callback_parser::json_sax_dom_callback_parser`](#json_sax_dom_callback_parserjson_sax_dom_callback_parser)
    - [`detail::json_sax_dom_callback_parser::operator=`](#json_sax_dom_callback_parseroperator=)
    - [`detail::json_sax_dom_callback_parser::operator=`](#json_sax_dom_callback_parseroperator=)
    - [`detail::json_sax_dom_callback_parser::~json_sax_dom_callback_parser`](#json_sax_dom_callback_parserjson_sax_dom_callback_parser)
    - [`detail::json_sax_dom_callback_parser::null`](#json_sax_dom_callback_parsernull)
    - [`detail::json_sax_dom_callback_parser::boolean`](#json_sax_dom_callback_parserboolean)
    - [`detail::json_sax_dom_callback_parser::number_integer`](#json_sax_dom_callback_parsernumber_integer)
    - [`detail::json_sax_dom_callback_parser::number_unsigned`](#json_sax_dom_callback_parsernumber_unsigned)
    - [`detail::json_sax_dom_callback_parser::number_float`](#json_sax_dom_callback_parsernumber_float)
    - [`detail::json_sax_dom_callback_parser::string`](#json_sax_dom_callback_parserstring)
    - [`detail::json_sax_dom_callback_parser::binary`](#json_sax_dom_callback_parserbinary)
    - [`detail::json_sax_dom_callback_parser::start_object`](#json_sax_dom_callback_parserstart_object)
    - [`detail::json_sax_dom_callback_parser::key`](#json_sax_dom_callback_parserkey)
    - [`detail::json_sax_dom_callback_parser::end_object`](#json_sax_dom_callback_parserend_object)
    - [`detail::json_sax_dom_callback_parser::start_array`](#json_sax_dom_callback_parserstart_array)
    - [`detail::json_sax_dom_callback_parser::end_array`](#json_sax_dom_callback_parserend_array)
    - [`detail::json_sax_dom_callback_parser::parse_error`](#json_sax_dom_callback_parserparse_error)
    - [`detail::json_sax_dom_callback_parser::is_errored`](#json_sax_dom_callback_parseris_errored)
    - [`detail::json_sax_dom_callback_parser::handle_diagnostic_positions_for_json_value`](#json_sax_dom_callback_parserhandle_diagnostic_positions_for_json_value)
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)

**Methods**

---
#### json\_sax\_dom\_callback\_parser::json\_sax\_dom\_callback\_parser<!-- {{#callable:detail::json_sax_dom_callback_parser::json_sax_dom_callback_parser}} -->
Constructs a `json_sax_dom_callback_parser` object to parse JSON data using a SAX-style callback mechanism.
- **Inputs**:
    - `r`: A reference to a `BasicJsonType` object that will hold the parsed JSON data.
    - `cb`: A callback function of type `parser_callback_t` that is invoked during parsing events.
    - `allow_exceptions_`: A boolean flag indicating whether exceptions should be thrown on parsing errors (default is true).
    - `lexer_`: An optional pointer to a `lexer_t` object used to track the current position in the input stream.
- **Control Flow**:
    - The constructor initializes the member variables with the provided arguments.
    - It pushes a boolean value `true` onto the `keep_stack` to indicate that the first value should be kept.
    - The `callback` function is set up to handle parsing events as they occur.
- **Output**: The constructor does not return a value but initializes the parser object for subsequent JSON parsing operations.
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::json\_sax\_dom\_callback\_parser<!-- {{#callable:detail::json_sax_dom_callback_parser::json_sax_dom_callback_parser}} -->
`json_sax_dom_callback_parser` is a move-only class designed to parse JSON data using a SAX (Simple API for XML) style callback mechanism.
- **Inputs**:
    - `BasicJsonType& r`: A reference to a `BasicJsonType` object that will hold the parsed JSON value.
    - `parser_callback_t cb`: A callback function that is invoked during parsing to handle various events such as object start, key, and value.
    - `const bool allow_exceptions_`: A boolean flag indicating whether exceptions should be thrown on parsing errors.
    - `lexer_t* lexer_`: An optional pointer to a lexer object used to track the current position in the input.
- **Control Flow**:
    - The constructor initializes the parser with the provided JSON root, callback, exception handling flag, and lexer reference.
    - The class is designed to be move-only, disabling copy construction and assignment.
    - Various methods handle different JSON types (null, boolean, number, string, binary) by calling `handle_value` to process the value.
    - The `start_object` and `start_array` methods manage the beginning of objects and arrays, checking the callback and maintaining the reference stack.
    - The `key` method processes keys in objects, invoking the callback and managing the storage of values.
    - The `end_object` and `end_array` methods finalize the parsing of objects and arrays, invoking the callback and managing the reference stack.
    - The `parse_error` method handles errors during parsing, optionally throwing exceptions based on the `allow_exceptions` flag.
- **Output**: The output is a parsed JSON structure represented by the `BasicJsonType` object, which is built incrementally as the parser processes the input.
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::json\_sax\_dom\_callback\_parser<!-- {{#callable:detail::json_sax_dom_callback_parser::json_sax_dom_callback_parser}} -->
`json_sax_dom_callback_parser` is a move-only class designed to parse JSON data using a SAX (Simple API for XML) style callback mechanism.
- **Inputs**:
    - `r`: A reference to a `BasicJsonType` object that will hold the parsed JSON value.
    - `cb`: A callback function of type `parser_callback_t` that is invoked during parsing events.
    - `allow_exceptions_`: A boolean flag indicating whether exceptions should be thrown on parse errors.
    - `lexer_`: An optional pointer to a `lexer_t` object used to track the current position in the input.
- **Control Flow**:
    - The constructor initializes the parser with the provided JSON object, callback, exception handling flag, and lexer reference.
    - The class is designed to be move-only, disabling copy construction and assignment.
    - Various parsing methods (e.g., `null`, `boolean`, `number_integer`, etc.) handle different JSON value types by calling `handle_value`.
    - The `start_object` and `start_array` methods manage the beginning of JSON objects and arrays, respectively, checking limits and invoking the callback.
    - The `key` method processes keys in JSON objects, determining whether to keep the associated value based on the callback result.
    - The `end_object` and `end_array` methods finalize the parsing of objects and arrays, invoking the callback and managing the reference stack.
    - Error handling is performed in the `parse_error` method, which can throw exceptions based on the `allow_exceptions` flag.
- **Output**: The output is a parsed JSON structure stored in the `root` member of type `BasicJsonType`, with the ability to manage and discard values based on callback responses.
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::null<!-- {{#callable:detail::json_sax_dom_callback_parser::null}} -->
The `null` function handles a null value in a JSON parsing context.
- **Inputs**:
    - `none`: This function does not take any input arguments.
- **Control Flow**:
    - Calls the [`handle_value`](#json_sax_dom_callback_parserhandle_value) function with a `nullptr` to represent a null value.
    - Returns `true` to indicate successful handling of the null value.
- **Output**: Returns a boolean value `true`, indicating that the null value was successfully processed.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::boolean<!-- {{#callable:detail::json_sax_dom_callback_parser::boolean}} -->
Processes a boolean value and invokes a handler.
- **Inputs**:
    - `val`: A boolean value to be processed.
- **Control Flow**:
    - The function calls `handle_value(val)` to process the input boolean value.
    - After processing, it returns true unconditionally.
- **Output**: Always returns true, indicating successful processing of the boolean value.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::number\_integer<!-- {{#callable:detail::json_sax_dom_callback_parser::number_integer}} -->
Processes an integer value and handles it accordingly.
- **Inputs**:
    - `val`: An integer value of type `number_integer_t` to be processed.
- **Control Flow**:
    - Calls the [`handle_value`](#json_sax_dom_callback_parserhandle_value) function with the provided integer value `val`.
    - Returns true after processing the value.
- **Output**: Returns a boolean value indicating the success of the operation, which is always true in this case.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::number\_unsigned<!-- {{#callable:detail::json_sax_dom_callback_parser::number_unsigned}} -->
The `number_unsigned` function processes an unsigned number value and returns true.
- **Inputs**:
    - `val`: An unsigned number of type `number_unsigned_t` that is to be processed.
- **Control Flow**:
    - The function calls [`handle_value`](#json_sax_dom_callback_parserhandle_value) with the input value `val`.
    - After processing the value, the function returns true.
- **Output**: The function always returns true, indicating successful processing of the input value.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::number\_float<!-- {{#callable:detail::json_sax_dom_callback_parser::number_float}} -->
The `number_float` function processes a floating-point number value and invokes a handler function.
- **Inputs**:
    - `val`: A floating-point number of type `number_float_t` that is to be processed.
    - `unused`: A constant reference to a string of type `string_t`, which is not used in the function.
- **Control Flow**:
    - The function calls [`handle_value`](#json_sax_dom_callback_parserhandle_value) with the input `val` to process the floating-point number.
    - After processing, it returns `true` to indicate successful handling.
- **Output**: The function returns a boolean value, which is always `true` in this implementation, indicating that the floating-point number was handled successfully.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::string<!-- {{#callable:detail::json_sax_dom_callback_parser::string}} -->
Handles a string value during JSON parsing.
- **Inputs**:
    - `val`: A reference to a `string_t` object that represents the string value to be handled.
- **Control Flow**:
    - Calls the [`handle_value`](#json_sax_dom_callback_parserhandle_value) function with the provided string value.
    - Returns true to indicate successful handling of the string value.
- **Output**: Returns a boolean value indicating the success of the operation, which is always true in this case.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::binary<!-- {{#callable:detail::json_sax_dom_callback_parser::binary}} -->
Processes a binary value by handling it and returning a success status.
- **Inputs**:
    - `val`: A reference to a `binary_t` object that represents the binary data to be processed.
- **Control Flow**:
    - The function calls [`handle_value`](#json_sax_dom_callback_parserhandle_value) with the `val` argument moved, which processes the binary data.
    - After processing, the function returns true, indicating successful handling of the binary value.
- **Output**: Returns a boolean value, always true in this case, indicating that the binary value was successfully processed.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::start\_object<!-- {{#callable:detail::json_sax_dom_callback_parser::start_object}} -->
The `start_object` function initiates the parsing of a JSON object, checks for size limits, and manages the callback for object start events.
- **Inputs**:
    - `len`: A `std::size_t` representing the expected length of the JSON object being parsed.
- **Control Flow**:
    - The function first invokes a callback to check if parsing should continue for the object start event.
    - It then calls [`handle_value`](#json_sax_dom_callback_parserhandle_value) to create a new JSON object and pushes it onto the reference stack.
    - If the reference stack is not empty, it checks if the lexer is available to set the start position of the object.
    - The function checks if the provided length exceeds the maximum allowed size for the object, throwing an exception if it does.
- **Output**: The function returns a boolean value indicating the success of the operation, which is always `true` in this implementation.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::key<!-- {{#callable:detail::json_sax_dom_callback_parser::key}} -->
The `key` function processes a key in a JSON object and determines whether to keep it based on a callback.
- **Inputs**:
    - `val`: A reference to a string representing the key to be processed.
- **Control Flow**:
    - Creates a `BasicJsonType` object from the input key string `val`.
    - Calls the `callback` function with the current stack size and the key event, storing the result in `keep`.
    - Pushes the `keep` value onto the `key_keep_stack` to track whether the key should be retained.
    - If `keep` is true and the last reference in `ref_stack` is valid, assigns a discarded value to the object element at the specified key.
    - Returns true to indicate successful processing.
- **Output**: Always returns true, indicating that the key processing was completed.
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::end\_object<!-- {{#callable:detail::json_sax_dom_callback_parser::end_object}} -->
Ends the current JSON object and handles its finalization.
- **Inputs**:
    - `none`: This function does not take any input arguments.
- **Control Flow**:
    - Checks if the last reference in `ref_stack` is valid.
    - Calls the `callback` function with the current object end event; if it returns false, the object is marked as discarded.
    - If the object is kept, it sets the end position of the object if a lexer reference is available.
    - Calls `set_parents()` on the current object to update its parent references.
    - Pops the last element from both `ref_stack` and `keep_stack` to finalize the current object.
    - If the previous object in `ref_stack` is structured, it iterates through its elements to remove any discarded values.
- **Output**: Returns true to indicate successful completion of the object end processing.
- **Functions called**:
    - [`detail::json_sax_dom_parser::handle_diagnostic_positions_for_json_value`](#json_sax_dom_parserhandle_diagnostic_positions_for_json_value)
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::start\_array<!-- {{#callable:detail::json_sax_dom_callback_parser::start_array}} -->
The `start_array` function initializes the parsing of a JSON array, handling callbacks and size constraints.
- **Inputs**:
    - `len`: The expected length of the array to be parsed.
- **Control Flow**:
    - Calls the `callback` function to determine if parsing should continue, passing the current stack size and an event type indicating the start of an array.
    - Pushes the result of the callback onto the `keep_stack` to track whether to keep the array.
    - Calls [`handle_value`](#json_sax_dom_callback_parserhandle_value) to create a new JSON array value and pushes it onto the `ref_stack`.
    - If the last element in `ref_stack` is valid, it checks if the lexer has read the first character of the array to set the correct start position.
    - Checks if the provided length exceeds the maximum size of the array, throwing an exception if it does.
- **Output**: Returns true to indicate that the array parsing has started successfully.
- **Functions called**:
    - [`detail::json_sax_dom_callback_parser::handle_value`](#json_sax_dom_callback_parserhandle_value)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::end\_array<!-- {{#callable:detail::json_sax_dom_callback_parser::end_array}} -->
The `end_array` function finalizes the parsing of a JSON array, invoking a callback and managing the state of the reference stack.
- **Inputs**:
    - `none`: The function does not take any input parameters.
- **Control Flow**:
    - Checks if the last element in the `ref_stack` is valid.
    - Invokes the `callback` function with the current array's end event and updates the `keep` variable based on the callback's return value.
    - If `keep` is true, updates the end position of the array in the lexer reference and sets parent references.
    - If `keep` is false, marks the array as discarded and handles diagnostic positions for the discarded array.
    - Pops the last element from both `ref_stack` and `keep_stack`.
    - If the array was discarded, removes the last element from the parent array.
- **Output**: The function returns a boolean value `true`, indicating successful completion of the array parsing process.
- **Functions called**:
    - [`detail::json_sax_dom_parser::handle_diagnostic_positions_for_json_value`](#json_sax_dom_parserhandle_diagnostic_positions_for_json_value)
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::parse\_error<!-- {{#callable:detail::json_sax_dom_callback_parser::parse_error}} -->
Handles parsing errors by setting an error flag and optionally throwing an exception.
- **Inputs**:
    - `unused`: A size_t parameter that is unused in the function.
    - `unused`: A string parameter that is unused in the function.
    - `ex`: An exception object that contains information about the parsing error.
- **Control Flow**:
    - Sets the `errored` flag to true to indicate that a parsing error has occurred.
    - Uses `static_cast<void>(ex)` to suppress unused variable warnings for the exception parameter.
    - Checks the `allow_exceptions` flag; if true, it throws the exception using `JSON_THROW(ex)`.
    - Returns false to indicate that parsing has failed.
- **Output**: Returns a boolean value, specifically false, indicating that an error occurred during parsing.
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::is\_errored<!-- {{#callable:detail::json_sax_dom_callback_parser::is_errored}} -->
The `is_errored` function checks if a syntax error has occurred during JSON parsing.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `json_sax_dom_callback_parser` class.
- **Control Flow**:
    - The function directly returns the value of the member variable `errored`.
- **Output**: Returns a boolean indicating whether a syntax error has occurred (true if an error has occurred, false otherwise).
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::handle\_diagnostic\_positions\_for\_json\_value<!-- {{#callable:detail::json_sax_dom_callback_parser::handle_diagnostic_positions_for_json_value}} -->
Handles the diagnostic positions for a JSON value based on the lexer state.
- **Inputs**:
    - `v`: A reference to a `BasicJsonType` object representing the JSON value whose diagnostic positions are to be set.
- **Control Flow**:
    - Checks if the lexer reference `m_lexer_ref` is valid.
    - Sets the `end_position` of the JSON value `v` to the current position of the lexer.
    - Uses a switch statement to determine the type of the JSON value and sets the `start_position` accordingly.
    - Handles specific cases for boolean, null, string, binary, and numeric types to calculate their start positions based on their string representations.
    - For `discarded` type, sets both `start_position` and `end_position` to `std::string::npos`.
    - For `object` and `array` types, skips setting positions as they are handled elsewhere.
- **Output**: The function does not return a value; it modifies the `start_position` and `end_position` attributes of the input JSON value `v`.
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)


---
#### json\_sax\_dom\_callback\_parser::handle\_value<!-- {{#callable:detail::json_sax_dom_callback_parser::handle_value}} -->
Handles the addition of a value to a JSON structure, managing its storage based on the current parsing context.
- **Inputs**:
    - `v`: The value to be added to the JSON structure, which can be of any type that is convertible to `BasicJsonType`.
    - `skip_callback`: A boolean flag indicating whether to skip invoking the callback function during the handling of the value.
- **Control Flow**:
    - Checks if the `keep_stack` is empty to ensure that values are only handled when appropriate.
    - If the last entry in `keep_stack` is false, the function returns early, indicating the value should not be kept.
    - Creates a new `BasicJsonType` instance from the input value `v`.
    - Invokes the callback function to determine if the value should be kept, based on the current parsing context.
    - If the `ref_stack` is empty, the new value is set as the root of the JSON structure.
    - If the `ref_stack` is not empty, it checks if the last entry is an array or object and adds the value accordingly.
    - For objects, it checks if the current key should store the value and updates the corresponding entry.
- **Output**: Returns a pair consisting of a boolean indicating whether the value was kept and a pointer to the stored value in the JSON structure, or nullptr if the value was discarded.
- **Functions called**:
    - [`detail::json_sax_dom_parser::handle_diagnostic_positions_for_json_value`](#json_sax_dom_parserhandle_diagnostic_positions_for_json_value)
- **See also**: [`detail::json_sax_dom_callback_parser`](#detailjson_sax_dom_callback_parser)  (Data Structure)



---
### json\_sax\_acceptor<!-- {{#data_structure:detail::json_sax_acceptor}} -->
- **Type**: `class`
- **Members**:
    - `number_integer_t`: Type alias for signed integer numbers.
    - `number_unsigned_t`: Type alias for unsigned integer numbers.
    - `number_float_t`: Type alias for floating-point numbers.
    - `string_t`: Type alias for string types.
    - `binary_t`: Type alias for binary data types.
- **Description**: The `json_sax_acceptor` class template provides a SAX (Simple API for XML) style interface for processing JSON data, allowing users to define how to handle various JSON types such as null, boolean, numbers, strings, and binary data, as well as the structure of JSON objects and arrays.
- **Member Functions**:
    - [`detail::json_sax_acceptor::null`](#json_sax_acceptornull)
    - [`detail::json_sax_acceptor::boolean`](#json_sax_acceptorboolean)
    - [`detail::json_sax_acceptor::number_integer`](#json_sax_acceptornumber_integer)
    - [`detail::json_sax_acceptor::number_unsigned`](#json_sax_acceptornumber_unsigned)
    - [`detail::json_sax_acceptor::number_float`](#json_sax_acceptornumber_float)
    - [`detail::json_sax_acceptor::string`](#json_sax_acceptorstring)
    - [`detail::json_sax_acceptor::binary`](#json_sax_acceptorbinary)
    - [`detail::json_sax_acceptor::start_object`](#json_sax_acceptorstart_object)
    - [`detail::json_sax_acceptor::key`](#json_sax_acceptorkey)
    - [`detail::json_sax_acceptor::end_object`](#json_sax_acceptorend_object)
    - [`detail::json_sax_acceptor::start_array`](#json_sax_acceptorstart_array)
    - [`detail::json_sax_acceptor::end_array`](#json_sax_acceptorend_array)
    - [`detail::json_sax_acceptor::parse_error`](#json_sax_acceptorparse_error)

**Methods**

---
#### json\_sax\_acceptor::null<!-- {{#callable:detail::json_sax_acceptor::null}} -->
The `null` function always returns true.
- **Inputs**: None
- **Control Flow**:
    - The function executes a single return statement.
    - No conditional logic or loops are present.
- **Output**: The output is a boolean value, specifically `true`.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)


---
#### json\_sax\_acceptor::boolean<!-- {{#callable:detail::json_sax_acceptor::boolean}} -->
The `boolean` function always returns `true` regardless of its input.
- **Inputs**:
    - `unused`: A boolean input parameter that is not utilized within the function.
- **Control Flow**:
    - The function takes a single boolean parameter named `unused` but does not use it in any computations.
    - It directly returns the boolean value `true` without any conditions or additional logic.
- **Output**: The function outputs a boolean value, specifically `true`.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)


---
#### json\_sax\_acceptor::number\_integer<!-- {{#callable:detail::json_sax_acceptor::number_integer}} -->
The `number_integer` function always returns true, indicating successful handling of an integer number.
- **Inputs**:
    - `number_integer_t`: An unused parameter of type `number_integer_t`, which is defined as part of the `BasicJsonType` template.
- **Control Flow**:
    - The function takes a single parameter but does not utilize it in any way.
    - It directly returns the boolean value `true` without any conditions or computations.
- **Output**: The function outputs a boolean value `true`, indicating a successful operation.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)


---
#### json\_sax\_acceptor::number\_unsigned<!-- {{#callable:detail::json_sax_acceptor::number_unsigned}} -->
The `number_unsigned` function always returns true regardless of its input.
- **Inputs**:
    - `number_unsigned_t`: An unused parameter of type `number_unsigned_t`, which is defined in the template parameter `BasicJsonType`.
- **Control Flow**:
    - The function takes a single parameter but does not utilize it in any way.
    - It directly returns the boolean value `true`.
- **Output**: The function outputs a boolean value `true`.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)


---
#### json\_sax\_acceptor::number\_float<!-- {{#callable:detail::json_sax_acceptor::number_float}} -->
The `number_float` function always returns true, regardless of its input parameters.
- **Inputs**:
    - `number_float_t`: An unused parameter of type `number_float_t`, which is defined in the `BasicJsonType` template.
    - `string_t`: An unused parameter of type `string_t`, which is also defined in the `BasicJsonType` template.
- **Control Flow**:
    - The function takes two parameters but does not utilize them in any way.
    - It directly returns a boolean value of true without any conditions or computations.
- **Output**: The output is a boolean value, specifically true, indicating a successful operation or acceptance of the input.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)


---
#### json\_sax\_acceptor::string<!-- {{#callable:detail::json_sax_acceptor::string}} -->
The `string` function in the `json_sax_acceptor` class always returns true regardless of the input.
- **Inputs**:
    - `string_t& /*unused*/`: A reference to a string type defined by the template parameter `BasicJsonType`, which is not used in the function.
- **Control Flow**:
    - The function does not perform any operations on the input argument.
    - It directly returns a boolean value of true.
- **Output**: The function outputs a boolean value, specifically `true`, indicating successful handling of the string input.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)


---
#### json\_sax\_acceptor::binary<!-- {{#callable:detail::json_sax_acceptor::binary}} -->
The `binary` function always returns true regardless of its input.
- **Inputs**:
    - `binary_t& /*unused*/`: A reference to a binary type, which is not utilized in the function.
- **Control Flow**:
    - The function does not perform any operations on the input argument.
    - It directly returns a boolean value of true.
- **Output**: The function outputs a boolean value, specifically true.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)


---
#### json\_sax\_acceptor::start\_object<!-- {{#callable:detail::json_sax_acceptor::start_object}} -->
The `start_object` function indicates the beginning of a JSON object and always returns true.
- **Inputs**:
    - `unused`: An optional size parameter that defaults to `detail::unknown_size()` but is not utilized within the function.
- **Control Flow**:
    - The function does not contain any conditional statements or loops.
    - It directly returns a boolean value of true, indicating successful execution.
- **Output**: The function outputs a boolean value, specifically true, indicating that the start of a JSON object has been successfully processed.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)


---
#### json\_sax\_acceptor::key<!-- {{#callable:detail::json_sax_acceptor::key}} -->
The `key` function always returns true, indicating successful processing of a key in a JSON-like structure.
- **Inputs**:
    - `string_t& /*unused*/`: A reference to a string type that is not used within the function.
- **Control Flow**:
    - The function does not contain any conditional statements or loops.
    - It directly returns a boolean value of true.
- **Output**: The function outputs a boolean value, specifically `true`, indicating that the key processing was successful.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)


---
#### json\_sax\_acceptor::end\_object<!-- {{#callable:detail::json_sax_acceptor::end_object}} -->
The `end_object` function indicates the end of a JSON object parsing.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any conditional statements or loops.
    - It directly returns a boolean value of true.
- **Output**: The function outputs a boolean value, specifically true, indicating successful completion of the end object operation.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)


---
#### json\_sax\_acceptor::start\_array<!-- {{#callable:detail::json_sax_acceptor::start_array}} -->
The `start_array` function indicates the beginning of an array in a JSON parsing context.
- **Inputs**:
    - `unused`: An optional size parameter that defaults to `detail::unknown_size()`, which is not utilized in the function.
- **Control Flow**:
    - The function does not contain any conditional statements or loops.
    - It directly returns a boolean value of `true`.
- **Output**: The function outputs a boolean value `true`, indicating successful initiation of an array.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)


---
#### json\_sax\_acceptor::end\_array<!-- {{#callable:detail::json_sax_acceptor::end_array}} -->
The `end_array` function indicates the end of an array in a JSON parsing context.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any conditional statements or loops.
    - It directly returns a boolean value of `true`.
- **Output**: The function outputs a boolean value, specifically `true`, indicating successful completion of the operation.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)


---
#### json\_sax\_acceptor::parse\_error<!-- {{#callable:detail::json_sax_acceptor::parse_error}} -->
The `parse_error` function always returns false, indicating that an error in parsing has occurred.
- **Inputs**:
    - `unused`: A size_t parameter that is not used in the function.
    - `unused`: A string parameter that is not used in the function.
    - `unused`: A `detail::exception` parameter that is not used in the function.
- **Control Flow**:
    - The function does not perform any operations or checks on the input parameters.
    - It directly returns a boolean value of false.
- **Output**: The output is a boolean value, specifically false, indicating a parsing error.
- **See also**: [`detail::json_sax_acceptor`](#detailjson_sax_acceptor)  (Data Structure)



---
### is\_sax<!-- {{#data_structure:namespace::is_sax}} -->
- **Type**: `struct`
- **Members**:
    - `value`: A compile-time constant that indicates whether the `SAX` type meets the required interface for a SAX parser.
    - `number_integer_t`: An alias for the integer type defined in `BasicJsonType`.
    - `number_unsigned_t`: An alias for the unsigned integer type defined in `BasicJsonType`.
    - `number_float_t`: An alias for the floating-point type defined in `BasicJsonType`.
    - `string_t`: An alias for the string type defined in `BasicJsonType`.
    - `binary_t`: An alias for the binary type defined in `BasicJsonType`.
    - `exception_t`: An alias for the exception type defined in `BasicJsonType`.
- **Description**: The `is_sax` struct is a type trait that checks if a given `SAX` type conforms to the expected interface for a SAX parser, ensuring it provides the necessary functions for handling various JSON data types as defined by the `BasicJsonType`. It uses static assertions and type aliases to enforce these requirements at compile time.


---
### is\_sax\_static\_asserts<!-- {{#data_structure:namespace::is_sax_static_asserts}} -->
- **Type**: `struct`
- **Members**:
    - `number_integer_t`: Type alias for the integer type defined in BasicJsonType.
    - `number_unsigned_t`: Type alias for the unsigned integer type defined in BasicJsonType.
    - `number_float_t`: Type alias for the floating-point type defined in BasicJsonType.
    - `string_t`: Type alias for the string type defined in BasicJsonType.
    - `binary_t`: Type alias for the binary type defined in BasicJsonType.
    - `exception_t`: Type alias for the exception type defined in BasicJsonType.
- **Description**: The `is_sax_static_asserts` struct is a template-based static assertion mechanism that validates the presence and correctness of various functions in a SAX (Simple API for XML) parser, ensuring that the provided `SAX` type adheres to the expected interface and that the `BasicJsonType` is a valid JSON type.


---
### cbor\_tag\_handler\_t<!-- {{#data_structure:NLOHMANN_JSON_NAMESPACE_BEGIN::cbor_tag_handler_t}} -->
- **Type**: `enum class cbor_tag_handler_t`
- **Members**:
    - `error`: Indicates that a parse_error exception should be thrown when a tag is encountered.
    - `ignore`: Specifies that tags should be ignored during processing.
    - `store`: Indicates that tags should be stored as a binary type.
- **Description**: `cbor_tag_handler_t` is an enumeration that defines how to handle tags in CBOR (Concise Binary Object Representation) processing, providing options to throw an error, ignore the tags, or store them as binary data.


---
### binary\_reader<!-- {{#data_structure:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader}} -->
- **Type**: `class template for reading binary data formats.`
- **Members**:
    - `ia`: input adapter used for reading data.
    - `current`: the current character being read.
    - `chars_read`: the number of characters read from the input.
    - `is_little_endian`: indicates if the system uses little-endian byte order.
    - `input_format`: the format of the input data being read.
    - `sax`: pointer to the SAX parser for processing parsed data.
- **Description**: The `binary_reader` class template is designed to read and parse various binary data formats such as BSON, CBOR, MessagePack, and UBJSON. It utilizes an input adapter to read data and a SAX parser to handle parsed events. The class is move-only and provides functionality to handle different input formats, ensuring proper parsing and error handling for each format. The internal state includes the current character being read, the number of characters read, and the input format, allowing for flexible and efficient data processing.
- **Member Functions**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::binary_reader`](#binary_readerbinary_reader)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::binary_reader`](#binary_readerbinary_reader)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::binary_reader`](#binary_readerbinary_reader)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::operator=`](#binary_readeroperator=)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::operator=`](#binary_readeroperator=)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::~binary_reader`](#binary_readerbinary_reader)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_internal`](#binary_readerparse_bson_internal)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_bson_cstr`](#binary_readerget_bson_cstr)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_bson_string`](#binary_readerget_bson_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_bson_binary`](#binary_readerget_bson_binary)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_element_internal`](#binary_readerparse_bson_element_internal)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_element_list`](#binary_readerparse_bson_element_list)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_array`](#binary_readerparse_bson_array)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_cbor_internal`](#binary_readerparse_cbor_internal)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_string`](#binary_readerget_cbor_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_binary`](#binary_readerget_cbor_binary)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_array`](#binary_readerget_cbor_array)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_object`](#binary_readerget_cbor_object)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_msgpack_internal`](#binary_readerparse_msgpack_internal)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_string`](#binary_readerget_msgpack_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_binary`](#binary_readerget_msgpack_binary)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_array`](#binary_readerget_msgpack_array)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_object`](#binary_readerget_msgpack_object)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_ubjson_internal`](#binary_readerparse_ubjson_internal)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_string`](#binary_readerget_ubjson_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_ndarray_size`](#binary_readerget_ubjson_ndarray_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_size_value`](#binary_readerget_ubjson_size_value)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_size_type`](#binary_readerget_ubjson_size_type)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_value`](#binary_readerget_ubjson_value)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_array`](#binary_readerget_ubjson_array)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_object`](#binary_readerget_ubjson_object)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_high_precision_number`](#binary_readerget_ubjson_high_precision_number)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get`](#binary_readerget)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_to`](#binary_readerget_to)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ignore_noop`](#binary_readerget_ignore_noop)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::byte_swap`](#binary_readerbyte_swap)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_number`](#binary_readerget_number)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_string`](#binary_readerget_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_binary`](#binary_readerget_binary)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_token_string`](#binary_readerget_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)

**Methods**

---
#### binary\_reader::binary\_reader<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::binary_reader}} -->
Constructs a `binary_reader` object that initializes an input adapter and specifies the input format.
- **Inputs**:
    - `adapter`: An rvalue reference to an `InputAdapterType` that serves as the input source for reading data.
    - `format`: An optional parameter of type `input_format_t` that specifies the format of the input data, defaulting to `json`.
- **Control Flow**:
    - The constructor uses `std::move` to transfer ownership of the `adapter` to the member variable `ia`.
    - It initializes the `input_format` member variable with the provided `format` argument.
    - A static assertion is performed to ensure that the `SAX` type and `BasicJsonType` are compatible.
- **Output**: The constructor does not return a value but initializes the `binary_reader` object for subsequent use in reading binary data.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::binary\_reader<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::binary_reader}} -->
The `sax_parse` method of the `binary_reader` class parses binary data formats and invokes a SAX parser for processing.
- **Inputs**:
    - `format`: The binary format to parse, specified as an `input_format_t` enumeration.
    - `sax_`: A pointer to a SAX event processor that handles parsed data.
    - `strict`: A boolean indicating whether to enforce strict parsing rules.
    - `tag_handler`: A handler that defines how to treat CBOR tags.
- **Control Flow**:
    - The method begins by assigning the SAX processor to the member variable `sax`.
    - It then uses a switch statement to determine the parsing logic based on the specified `format`.
    - For each format (BSON, CBOR, MessagePack, UBJSON, and BJData), it calls the corresponding internal parsing function.
    - If parsing is successful and `strict` mode is enabled, it checks for any remaining input to ensure it has reached EOF.
    - If any unexpected input is found, it triggers a parse error through the SAX processor.
- **Output**: Returns a boolean indicating whether the parsing was successful.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::binary\_reader<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::binary_reader}} -->
The `binary_reader` class is designed to read and parse binary data formats such as BSON, CBOR, MessagePack, and UBJSON.
- **Inputs**:
    - `adapter`: An input adapter of type `InputAdapterType` that provides the data to be read.
    - `format`: An optional parameter of type `input_format_t` that specifies the format of the input data, defaulting to JSON.
- **Control Flow**:
    - The constructor initializes the input adapter and format, and performs static assertions on the SAX parser.
    - The class is designed to be move-only, with copy constructors and assignment operators deleted to prevent copying.
    - The `sax_parse` method determines the input format and calls the appropriate parsing function based on the format.
    - For strict parsing, it checks if the end of input is reached after successful parsing.
- **Output**: The output of the parsing functions is a boolean indicating whether the parsing was successful, with parsed data being passed to the SAX parser.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::operator=<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::operator=}} -->
The `operator=` function in the `binary_reader` class is deleted for copy assignment and defaulted for move assignment.
- **Inputs**:
    - `other`: A rvalue reference to another `binary_reader` instance, which is being moved.
- **Control Flow**:
    - The copy assignment operator is deleted to prevent copying of `binary_reader` instances.
    - The move assignment operator is defaulted, allowing the transfer of resources from one `binary_reader` instance to another.
- **Output**: Returns a reference to the current instance of `binary_reader` after moving the resources.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::\~binary\_reader<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::~binary_reader}} -->
The `~binary_reader` is a default destructor for the `binary_reader` class, which is responsible for cleaning up resources when an instance of the class is destroyed.
- **Inputs**: None
- **Control Flow**:
    - The destructor does not contain any specific logic and relies on the default behavior provided by the compiler.
    - It ensures that any resources allocated by the `binary_reader` class are properly released when an instance is destroyed.
- **Output**: The function does not return any value, as it is a destructor.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::parse\_bson\_internal<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_internal}} -->
Parses a BSON document and passes its elements to a SAX parser.
- **Inputs**: None
- **Control Flow**:
    - Calls `get_number` to read the size of the BSON document.
    - Checks if the SAX parser can start a new object; if not, returns false.
    - Calls [`parse_bson_element_list`](#binary_readerparse_bson_element_list) to parse the elements of the BSON document.
    - Checks if the SAX parser can end the object; if not, returns false.
- **Output**: Returns true if the BSON document was successfully parsed and passed to the SAX parser, otherwise returns false.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_element_list`](#binary_readerparse_bson_element_list)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_bson\_cstr<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_bson_cstr}} -->
Parses a C-style string from BSON input until a null terminator is encountered.
- **Inputs**:
    - `result`: A reference to a string variable where the parsed C-style string will be stored.
- **Control Flow**:
    - The function enters an infinite loop to read characters from the input.
    - It calls `get()` to retrieve the next character.
    - If the end of the input is unexpectedly reached before a null terminator is found, it returns false.
    - If a null character (0x00) is encountered, it breaks the loop and returns true.
    - Otherwise, the character is appended to the `result` string.
- **Output**: Returns true if the string was successfully parsed, or false if an unexpected end of input occurred.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_bson\_string<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_bson_string}} -->
Parses a BSON string of a specified length and stores it in the provided result variable.
- **Inputs**:
    - `len`: The length of the string to be read, including the null terminator.
    - `result`: A reference to a string variable where the parsed string will be stored.
- **Control Flow**:
    - Checks if the provided length is less than 1; if so, it triggers a parse error.
    - Calls [`get_string`](#get_string) to read the string data from the BSON input, adjusting the length to account for the null terminator.
    - Checks if the next character read is not EOF to ensure the string was fully read.
- **Output**: Returns true if the string was successfully parsed, false otherwise.
- **Functions called**:
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`namespace::concat`](#namespaceconcat)
    - [`get_string`](#get_string)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_bson\_binary<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_bson_binary}} -->
Parses a BSON binary value of a specified length and stores it in a result object.
- **Inputs**:
    - `len`: The length of the binary data to be read, which must be non-negative.
    - `result`: A reference to a `binary_t` object where the parsed binary data will be stored.
- **Control Flow**:
    - Checks if the provided length `len` is negative; if so, it triggers a parse error.
    - Reads the binary subtype from the input and sets it in the `result` object.
    - Calls the [`get_binary`](#binary_readerget_binary) function to read the actual binary data of the specified length into the `result` object.
- **Output**: Returns a boolean indicating whether the binary data was successfully parsed and stored.
- **Functions called**:
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`namespace::concat`](#namespaceconcat)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_binary`](#binary_readerget_binary)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::parse\_bson\_element\_internal<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_element_internal}} -->
Parses a BSON element based on its type and forwards the parsed data to a SAX parser.
- **Inputs**:
    - `element_type`: The BSON element type, represented as a `char_int_type`, which determines how to parse the element.
    - `element_type_parse_position`: The position in the input stream where the `element_type` was read, used for error reporting.
- **Control Flow**:
    - The function begins by evaluating the `element_type` using a switch statement.
    - For each case corresponding to a BSON type (e.g., double, string, object, array, binary, boolean, null, int32, int64, uint64), it performs specific parsing actions.
    - If the type is unsupported, it constructs an error message and calls the SAX parser's `parse_error` method.
    - For supported types, it retrieves the necessary data (like length for strings and binaries) and invokes the appropriate SAX methods to handle the parsed data.
- **Output**: Returns a boolean indicating whether the parsing was successful, with true for success and false for failure.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_bson_string`](#binary_readerget_bson_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_internal`](#binary_readerparse_bson_internal)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_array`](#binary_readerparse_bson_array)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_bson_binary`](#binary_readerget_bson_binary)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::parse\_bson\_element\_list<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_element_list}} -->
Parses a list of BSON elements, handling both object and array formats.
- **Inputs**:
    - `is_array`: A boolean indicating whether the element list being read is an array (true) or an object (false).
- **Control Flow**:
    - The function enters a loop that continues until there are no more elements to read.
    - It retrieves the type of the next BSON element using the `get()` function.
    - If the end of the input is unexpectedly reached, it returns false.
    - It reads a BSON C-style string key using `get_bson_cstr(key)`.
    - If not reading an array, it calls `sax->key(key)` to process the key.
    - It calls `parse_bson_element_internal(element_type, element_type_parse_position)` to parse the element based on its type.
    - After processing each element, it clears the key string for the next iteration.
- **Output**: Returns true if the entire list of BSON elements was successfully parsed; otherwise, returns false.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_bson_cstr`](#binary_readerget_bson_cstr)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_element_internal`](#binary_readerparse_bson_element_internal)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::parse\_bson\_array<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_array}} -->
Parses a BSON array and passes its elements to a SAX parser.
- **Inputs**:
    - `none`: The function does not take any input parameters.
- **Control Flow**:
    - Reads the size of the BSON document using `get_number`.
    - Checks if the SAX parser can start an array using `sax->start_array`.
    - If starting the array fails, the function returns false.
    - Calls [`parse_bson_element_list`](#binary_readerparse_bson_element_list) to parse the elements of the array.
    - If parsing the element list fails, the function returns false.
    - Ends the array using `sax->end_array` and returns true.
- **Output**: Returns true if the BSON array was successfully parsed and passed to the SAX parser; otherwise, returns false.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_bson_element_list`](#binary_readerparse_bson_element_list)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::parse\_cbor\_internal<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_cbor_internal}} -->
Parses CBOR (Concise Binary Object Representation) data and dispatches the parsed values to a SAX event handler.
- **Inputs**:
    - `get_char`: A boolean indicating whether to retrieve a new character from the input.
    - `tag_handler`: An enumeration that specifies how to handle CBOR tags.
- **Control Flow**:
    - The function begins by checking if it should get a new character or use the current one.
    - It then uses a switch statement to handle different CBOR data types based on the character value.
    - For each case, it processes the corresponding data type, such as integers, binary data, strings, arrays, and maps.
    - If the character indicates an end-of-file (EOF), it calls a function to handle unexpected EOF.
    - For each recognized data type, it calls specific functions to read the data and dispatch it to the SAX handler.
    - If an unrecognized character is encountered, it triggers a parse error.
- **Output**: Returns a boolean indicating whether the parsing was successful.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_number`](#binary_readerget_number)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_binary`](#binary_readerget_cbor_binary)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_string`](#binary_readerget_cbor_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_array`](#binary_readerget_cbor_array)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_object`](#binary_readerget_cbor_object)
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_cbor\_string<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_string}} -->
Parses a CBOR encoded UTF-8 string and stores it in the provided result variable.
- **Inputs**:
    - `result`: A reference to a string variable where the parsed CBOR string will be stored.
- **Control Flow**:
    - Checks for unexpected end of file (EOF) for the CBOR input format.
    - Switches on the value of `current` to determine the type of string length specification.
    - Handles fixed-length strings (0x60 to 0x77) by calling [`get_string`](#get_string) with the appropriate length.
    - Handles one-byte (0x78), two-byte (0x79), four-byte (0x7A), and eight-byte (0x7B) length specifications by reading the length and then calling [`get_string`](#get_string).
    - Handles indefinite length strings (0x7F) by repeatedly calling `get_cbor_string` until the end of the string is reached (indicated by 0xFF).
    - If an invalid byte is encountered, it triggers a parse error with a detailed message.
- **Output**: Returns true if the string was successfully parsed and stored in `result`, otherwise returns false.
- **Functions called**:
    - [`get_string`](#get_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_number`](#binary_readerget_number)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_cbor\_binary<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_binary}} -->
The `get_cbor_binary` function reads binary data from a CBOR input format and stores it in a provided `binary_t` result.
- **Inputs**:
    - `result`: A reference to a `binary_t` object where the read binary data will be stored.
- **Control Flow**:
    - The function first checks for unexpected end-of-file conditions using `unexpect_eof`.
    - It then uses a switch statement to handle different cases based on the value of `current`, which indicates the type of binary data to read.
    - For binary data types 0x40 to 0x57, it calls [`get_binary`](#binary_readerget_binary) with the appropriate length derived from `current`.
    - For types 0x58, 0x59, 0x5A, and 0x5B, it reads the length as a one-byte, two-byte, four-byte, or eight-byte integer respectively, and then calls [`get_binary`](#binary_readerget_binary).
    - For type 0x5F, it enters a loop to read indefinite-length binary data until it encounters the end marker (0xFF), recursively calling `get_cbor_binary` for each chunk.
    - If an invalid type is encountered, it triggers a parse error using `sax->parse_error`.
- **Output**: The function returns a boolean indicating success (true) or failure (false) in reading the binary data.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_binary`](#binary_readerget_binary)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_number`](#binary_readerget_number)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_cbor\_array<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_array}} -->
The `get_cbor_array` function reads a CBOR array from the input and processes its elements using a SAX parser.
- **Inputs**:
    - `len`: The length of the array to be read, or `detail::unknown_size()` for an indefinite length array.
    - `tag_handler`: A handler that specifies how to treat CBOR tags during parsing.
- **Control Flow**:
    - The function first attempts to signal the start of an array using `sax->start_array(len)`, returning false if it fails.
    - If the length is known (not `detail::unknown_size()`), it enters a loop to read each element, calling [`parse_cbor_internal`](#binary_readerparse_cbor_internal) for each element.
    - If the length is unknown, it continues reading elements until it encounters the end-of-array marker (0xFF), calling [`parse_cbor_internal`](#binary_readerparse_cbor_internal) for each element.
    - Finally, it signals the end of the array using `sax->end_array()`.
- **Output**: Returns true if the array was successfully read and processed; otherwise, returns false.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_cbor_internal`](#binary_readerparse_cbor_internal)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_cbor\_object<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_object}} -->
The `get_cbor_object` function parses a CBOR object from the input stream and processes its key-value pairs.
- **Inputs**:
    - `len`: The length of the object to be parsed, or detail::unknown_size() for an indefinite length.
    - `tag_handler`: A handler that defines how to treat CBOR tags during parsing.
- **Control Flow**:
    - The function starts by calling `sax->start_object(len)` to signal the beginning of an object parsing.
    - If the length is not zero, it checks if the length is known; if known, it iterates `len` times to read key-value pairs.
    - For each key, it retrieves the key string using `get_cbor_string(key)` and processes it with `sax->key(key)`.
    - Then, it calls `parse_cbor_internal(true, tag_handler)` to parse the corresponding value for the key.
    - If the length is unknown, it continues reading until it encounters the end marker (0xFF), processing each key-value pair similarly.
    - Finally, it calls `sax->end_object()` to signal the end of the object parsing.
- **Output**: Returns true if the object was successfully parsed, false otherwise.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_cbor_string`](#binary_readerget_cbor_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_cbor_internal`](#binary_readerparse_cbor_internal)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::parse\_msgpack\_internal<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_msgpack_internal}} -->
Parses a MessagePack formatted input and dispatches the parsed values to a SAX event handler.
- **Inputs**: None
- **Control Flow**:
    - The function begins by reading the next byte from the input using the `get()` function.
    - It checks if the byte corresponds to an EOF (end of file) condition, returning an error if so.
    - The function then uses a switch statement to handle different byte values, each representing a different type of MessagePack data.
    - For positive fix integers (0x00 to 0x7F), it directly calls the SAX handler to process the unsigned number.
    - For fix maps (0x80 to 0x8F), it calls `get_msgpack_object()` to parse the object.
    - For fix arrays (0x90 to 0x9F), it calls `get_msgpack_array()` to parse the array.
    - For fix strings (0xA0 to 0xBF and extended string types 0xD9, 0xDA, 0xDB), it calls `get_msgpack_string()` to parse the string.
    - For nil (0xC0), it calls the SAX handler to process a null value.
    - For boolean values (0xC2 for false and 0xC3 for true), it calls the SAX handler to process the boolean.
    - For binary and extended types (0xC4 to 0xC9 and 0xD4 to 0xD8), it calls `get_msgpack_binary()` to parse the binary data.
    - For float values (0xCA for float32 and 0xCB for float64), it reads the number and calls the SAX handler to process the float.
    - For unsigned integers (0xCC to 0xCF) and signed integers (0xD0 to 0xD3), it reads the respective number and calls the SAX handler.
    - For arrays and maps with 16 or 32-bit lengths (0xDC to 0xDF), it reads the length and calls the respective parsing functions.
    - If the byte does not match any expected values, it triggers a parse error.
- **Output**: Returns true if the parsing was successful, otherwise returns false and triggers an error through the SAX handler.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_object`](#binary_readerget_msgpack_object)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_array`](#binary_readerget_msgpack_array)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_string`](#binary_readerget_msgpack_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_binary`](#binary_readerget_msgpack_binary)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_number`](#binary_readerget_number)
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_msgpack\_string<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_string}} -->
Parses a MessagePack string from the input and stores it in the provided result variable.
- **Inputs**:
    - `result`: A reference to a string variable where the parsed MessagePack string will be stored.
- **Control Flow**:
    - The function first checks for unexpected end-of-file conditions using `unexpect_eof`.
    - It then uses a switch statement to determine the type of string based on the value of `current`.
    - For fixstr types (0xA0 to 0xBF), it calls [`get_string`](#get_string) with the appropriate length derived from `current`.
    - For str 8 (0xD9), str 16 (0xDA), and str 32 (0xDB), it first reads the length using [`get_number`](#binary_readerget_number) and then calls [`get_string`](#get_string).
    - If the `current` value does not match any expected types, it retrieves the last token and triggers a parse error.
- **Output**: Returns true if the string was successfully parsed and stored in `result`, otherwise returns false.
- **Functions called**:
    - [`get_string`](#get_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_number`](#binary_readerget_number)
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_msgpack\_binary<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_binary}} -->
The `get_msgpack_binary` function reads binary data from a MessagePack format and populates a `binary_t` object with the data.
- **Inputs**:
    - `result`: A reference to a `binary_t` object where the read binary data will be stored.
- **Control Flow**:
    - The function starts by defining a lambda function `assign_and_return_true` that sets the subtype of the `result` and returns true.
    - It then checks the value of `current` to determine the type of binary data to read, using a switch statement.
    - For each case (0xC4, 0xC5, 0xC6, etc.), it reads the length of the binary data and then calls [`get_binary`](#binary_readerget_binary) to populate the `result`.
    - In cases where the subtype is needed (0xC7, 0xC8, 0xC9, 0xD4, etc.), it reads the subtype after reading the length and assigns it using the lambda function.
    - If `current` does not match any expected values, the function returns false.
- **Output**: Returns true if the binary data was successfully read and stored in `result`, otherwise returns false.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_number`](#binary_readerget_number)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_binary`](#binary_readerget_binary)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_msgpack\_array<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_array}} -->
The `get_msgpack_array` function reads a MessagePack array of a specified length and processes each element using a SAX parser.
- **Inputs**:
    - `len`: The length of the MessagePack array to be read.
- **Control Flow**:
    - The function first checks if the SAX parser can start an array of the specified length using `sax->start_array(len)`.
    - If starting the array fails, the function returns false.
    - It then enters a loop that iterates `len` times, calling `parse_msgpack_internal()` for each element.
    - If any call to `parse_msgpack_internal()` fails, the function returns false.
    - Finally, it calls `sax->end_array()` to signal the end of the array and returns its result.
- **Output**: The function returns true if the array was successfully read and processed; otherwise, it returns false.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_msgpack_internal`](#binary_readerparse_msgpack_internal)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_msgpack\_object<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_object}} -->
Parses a MessagePack object and passes its key-value pairs to a SAX parser.
- **Inputs**:
    - `len`: The number of key-value pairs in the MessagePack object to be parsed.
- **Control Flow**:
    - Checks if the SAX parser can start a new object with the specified length.
    - Iterates over the number of key-value pairs specified by 'len'.
    - For each iteration, retrieves the next byte, reads a string key, and checks if the SAX parser can accept the key.
    - Parses the corresponding value for the key using 'parse_msgpack_internal()'.
    - Clears the key string after processing each key-value pair.
    - Ends the object in the SAX parser after all key-value pairs have been processed.
- **Output**: Returns true if the object was successfully parsed and passed to the SAX parser; otherwise, returns false.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_msgpack_string`](#binary_readerget_msgpack_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_msgpack_internal`](#binary_readerparse_msgpack_internal)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::parse\_ubjson\_internal<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_ubjson_internal}} -->
Parses a UBJSON value from the input stream.
- **Inputs**:
    - `get_char`: A boolean flag indicating whether to retrieve a new character from the input (true) or use the last read character (false). Defaults to true.
- **Control Flow**:
    - The function calls [`get_ubjson_value`](#binary_readerget_ubjson_value) with either the result of `get_ignore_noop()` (if `get_char` is true) or the current character (if `get_char` is false).
    - The [`get_ubjson_value`](#binary_readerget_ubjson_value) function processes the character to determine the type of value to parse (e.g., boolean, null, number, string, array, or object).
- **Output**: Returns a boolean indicating whether a valid UBJSON value was successfully parsed and passed to the SAX parser.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_value`](#binary_readerget_ubjson_value)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ignore_noop`](#binary_readerget_ignore_noop)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_ubjson\_string<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_string}} -->
The `get_ubjson_string` function reads a UBJSON string from the input and stores it in the provided result variable.
- **Inputs**:
    - `result`: A reference to a string variable where the read string will be stored.
    - `get_char`: A boolean flag indicating whether to read a new character from the input (default is true).
- **Control Flow**:
    - If `get_char` is true, the function reads the next character from the input.
    - The function checks for unexpected end-of-file conditions.
    - Based on the value of the current character, it determines the length of the string to read using different cases ('U', 'i', 'I', 'l', 'L', 'u', 'm', 'M').
    - For each case, it reads the corresponding length and then reads the string of that length into `result`.
    - If the current character does not match any expected type, it constructs an error message indicating the expected types and returns a parse error.
- **Output**: Returns true if the string was successfully read and stored in `result`, otherwise returns false.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_number`](#binary_readerget_number)
    - [`get_string`](#get_string)
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_ubjson\_ndarray\_size<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_ndarray_size}} -->
The `get_ubjson_ndarray_size` function reads the size of a UBJSON ND array and populates a vector with its dimensions.
- **Inputs**:
    - `dim`: A reference to a vector of size_t where the dimensions of the ND array will be stored.
- **Control Flow**:
    - The function initializes a pair to hold size and type information, a variable for dimension length, and a flag to indicate if an ND array is present.
    - It calls [`get_ubjson_size_type`](#binary_readerget_ubjson_size_type) to retrieve the size and type of the array, returning false if this fails.
    - If the size is not `npos`, it checks the type; if the type is not 'N', it enters a loop to read the size values for each dimension and pushes them into the `dim` vector.
    - If the size is `npos`, it enters a while loop that continues until it encounters a closing bracket ']', reading size values and pushing them into the `dim` vector.
    - The function returns true if all operations are successful.
- **Output**: Returns true if the ND array size was successfully read and stored in the `dim` vector; otherwise, it returns false.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_size_type`](#binary_readerget_ubjson_size_type)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_size_value`](#binary_readerget_ubjson_size_value)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ignore_noop`](#binary_readerget_ignore_noop)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_ubjson\_size\_value<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_size_value}} -->
Determines the size value for UBJSON data based on a given prefix.
- **Inputs**:
    - `result`: A reference to a `std::size_t` variable where the determined size will be stored.
    - `is_ndarray`: A reference to a boolean indicating if the current context is within an ndarray.
    - `prefix`: An optional character indicating the type of size to read; defaults to 0.
- **Control Flow**:
    - If `prefix` is 0, it retrieves the next character using `get_ignore_noop()`.
    - A switch statement processes the `prefix` character to determine the size type.
    - For each case ('U', 'i', 'I', 'l', 'L', 'u', 'm', 'M', '['), it reads the corresponding number and checks for validity.
    - If the size is negative or out of range, it triggers a parse error.
    - If the prefix is '[', it handles the case for ndarray sizes, checking dimensions and returning appropriate results.
- **Output**: Returns a boolean indicating success or failure of the size determination process.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ignore_noop`](#binary_readerget_ignore_noop)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_number`](#binary_readerget_number)
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_ndarray_size`](#binary_readerget_ubjson_ndarray_size)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_ubjson\_size\_type<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_size_type}} -->
Determines the size and type of a UBJSON container.
- **Inputs**:
    - `result`: A reference to a `std::pair<std::size_t, char_int_type>` that will store the size and type of the container.
    - `inside_ndarray`: A boolean flag indicating whether the parser is currently inside an ND array dimensional vector.
- **Control Flow**:
    - Initializes the size in `result.first` to `npos` and type in `result.second` to 0.
    - Calls `get_ignore_noop()` to skip any 'N' characters.
    - Checks if the current character is '$' or '#' to determine the type of container.
    - If the current character is '$', retrieves the type and checks for errors related to optimized type markers.
    - If the current character is '#', retrieves the size value directly.
    - Handles errors related to unexpected end of input and invalid type specifications.
    - Returns true if the size and type determination is successful.
- **Output**: Returns a boolean indicating whether the size and type determination was completed successfully.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ignore_noop`](#binary_readerget_ignore_noop)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`namespace::concat`](#namespaceconcat)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_size_value`](#binary_readerget_ubjson_size_value)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_ubjson\_value<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_value}} -->
The `get_ubjson_value` function interprets a UBJSON value based on a given prefix character.
- **Inputs**:
    - `prefix`: A character that indicates the type of value to be parsed (e.g., 'T' for true, 'F' for false, etc.).
- **Control Flow**:
    - The function starts by checking the value of `prefix` using a switch statement.
    - For each case, it handles different types of UBJSON values such as boolean, null, integers, floats, strings, arrays, and objects.
    - If the prefix indicates a numeric type, it reads the corresponding number and passes it to the SAX parser.
    - If the prefix is not recognized, it triggers a parse error.
- **Output**: Returns a boolean indicating whether the parsing of the value was successful or not.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_number`](#binary_readerget_number)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_high_precision_number`](#binary_readerget_ubjson_high_precision_number)
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`namespace::concat`](#namespaceconcat)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_string`](#binary_readerget_ubjson_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_array`](#binary_readerget_ubjson_array)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_object`](#binary_readerget_ubjson_object)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_ubjson\_array<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_array}} -->
The `get_ubjson_array` function reads and parses an array from UBJSON or BJData formatted input.
- **Inputs**:
    - `size_and_type`: A pair containing the size of the array and its type, determined by the [`get_ubjson_size_type`](#binary_readerget_ubjson_size_type) function.
- **Control Flow**:
    - The function first retrieves the size and type of the array using [`get_ubjson_size_type`](#binary_readerget_ubjson_size_type).
    - If the input format is BJData and the type indicates an ndarray, it processes the array as a JData annotated array.
    - It checks for specific type markers and handles them accordingly, including starting and ending the array.
    - If the type is binary, it reads the binary data instead.
    - If the size is known, it iterates through the expected number of elements, calling [`get_ubjson_value`](#binary_readerget_ubjson_value) for each.
    - If the size is unknown, it continues reading until the end of the array is reached.
- **Output**: Returns true if the array was successfully parsed and processed, otherwise returns false.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_size_type`](#binary_readerget_ubjson_size_type)
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_value`](#binary_readerget_ubjson_value)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_binary`](#binary_readerget_binary)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_ubjson_internal`](#binary_readerparse_ubjson_internal)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ignore_noop`](#binary_readerget_ignore_noop)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_ubjson\_object<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_object}} -->
The `get_ubjson_object` function parses a UBJSON object from the input stream and passes the parsed data to a SAX event processor.
- **Inputs**: None
- **Control Flow**:
    - The function begins by retrieving the size and type of the UBJSON object using [`get_ubjson_size_type`](#binary_readerget_ubjson_size_type).
    - If the input format is `bjdata` and the size indicates an ND-array, an error is raised.
    - If a valid size is obtained, the function starts the object parsing by calling `sax->start_object`.
    - If the size type is not zero, it enters a loop to read key-value pairs, using [`get_ubjson_string`](#binary_readerget_ubjson_string) to read keys and [`get_ubjson_value`](#binary_readerget_ubjson_value) to read values.
    - If the size type is zero, it enters a loop to read key-value pairs until the closing brace is encountered.
    - Finally, it calls `sax->end_object` to signal the end of the object.
- **Output**: The function returns a boolean indicating whether the parsing was successful.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_size_type`](#binary_readerget_ubjson_size_type)
    - [`get_token_string`](#get_token_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_string`](#binary_readerget_ubjson_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_value`](#binary_readerget_ubjson_value)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::parse_ubjson_internal`](#binary_readerparse_ubjson_internal)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ignore_noop`](#binary_readerget_ignore_noop)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_ubjson\_high\_precision\_number<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_high_precision_number}} -->
Parses a high precision number from UBJSON format.
- **Inputs**:
    - `none`: The function does not take any input parameters directly.
- **Control Flow**:
    - The function first retrieves the size of the number string using [`get_ubjson_size_value`](#binary_readerget_ubjson_size_value).
    - If the size retrieval fails, it returns false.
    - It then reads the number string character by character into a vector.
    - If an unexpected end of file occurs while reading the number string, it returns false.
    - The number string is parsed using a lexer to determine its type (integer, unsigned, or float).
    - If there are remaining tokens after parsing, it triggers a parse error.
    - Based on the parsed type, it calls the appropriate SAX method to handle the number.
- **Output**: Returns true if the number was successfully parsed and handled; otherwise, it returns false.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ubjson_size_value`](#binary_readerget_ubjson_size_value)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get}} -->
The `get` function retrieves the next character from the input adapter and increments the count of characters read.
- **Inputs**: None
- **Control Flow**:
    - The function increments the `chars_read` counter to track the number of characters read.
    - It calls the `get_character` method of the input adapter `ia` to retrieve the next character.
    - The retrieved character is assigned to the `current` variable.
- **Output**: The function returns the character read from the input, which is of type `char_int_type`.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_to<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_to}} -->
Reads data into a specified destination variable from an input source, handling potential end-of-file conditions.
- **Inputs**:
    - `dest`: A reference to the variable of type T where the read data will be stored.
    - `format`: An enumeration value of type `input_format_t` that indicates the format of the input data.
    - `context`: A string providing context for error messages, indicating where the read operation is taking place.
- **Control Flow**:
    - Calls `ia.get_elements(&dest)` to read data into `dest` and updates the `chars_read` counter.
    - Checks if the number of characters read is less than the size of T, indicating a potential end-of-file condition.
    - If an end-of-file condition is detected, increments `chars_read`, reports an error using `sax->parse_error`, and returns false.
    - If the read operation is successful, returns true.
- **Output**: Returns a boolean indicating whether the read operation was successful.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_ignore\_noop<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_ignore_noop}} -->
The `get_ignore_noop` function retrieves characters from the input until a character other than 'N' is found.
- **Inputs**: None
- **Control Flow**:
    - The function enters a do-while loop that continuously calls the `get()` function to read the next character.
    - The loop continues as long as the current character is 'N'.
- **Output**: The function returns the first character that is not 'N', or the EOF character if the end of input is reached.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::byte\_swap<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::byte_swap}} -->
The `byte_swap` function swaps the byte order of a given number, handling both integral types and single-byte cases.
- **Inputs**:
    - `number`: A reference to a variable of type `NumberType`, which can be any integral type or a type that can be byteswapped.
- **Control Flow**:
    - The function first determines the size of the input `number` using `sizeof`.
    - If the size is 1 byte, the function returns immediately as no swapping is needed.
    - If the type is integral and the `std::byteswap` function is available, it uses that to swap the bytes.
    - If `std::byteswap` is not available, it manually swaps the bytes by iterating through half of the size of the number and swapping the corresponding bytes.
- **Output**: The function does not return a value; it modifies the input `number` in place to reflect its byte-swapped version.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_number<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_number}} -->
The `get_number` function reads a number from the input in a specified format and handles endianness.
- **Inputs**:
    - `format`: The format of the input data, specified as an `input_format_t` enumeration.
    - `result`: A reference to a variable of type `NumberType` where the read number will be stored.
- **Control Flow**:
    - The function first attempts to read the number from the input using the [`get_to`](#binary_readerget_to) function.
    - If reading fails, it returns false.
    - It then checks if the system's endianness matches the expected endianness for the input format.
    - If they do not match, it calls [`byte_swap`](#binary_readerbyte_swap) to adjust the byte order of the result.
    - Finally, it returns true indicating the number was successfully read.
- **Output**: Returns a boolean indicating whether the number was successfully read from the input.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_to`](#binary_readerget_to)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::byte_swap`](#binary_readerbyte_swap)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_string<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_string}} -->
The `get_string` function reads a specified number of characters from an input source and appends them to a result string.
- **Inputs**:
    - `format`: An enumeration value of type `input_format_t` that specifies the format of the input being read.
    - `len`: A numeric value of type `NumberType` that indicates the number of characters to read.
    - `result`: A reference to a string of type `string_t` where the read characters will be appended.
- **Control Flow**:
    - The function initializes a boolean variable `success` to true.
    - A for loop iterates from 0 to `len`, reading one character at a time from the input source.
    - Within the loop, the `get()` function is called to read the next character.
    - If the end of the input is reached unexpectedly (checked by `unexpect_eof`), `success` is set to false and the loop breaks.
    - Each successfully read character is appended to the `result` string.
- **Output**: The function returns a boolean value indicating whether all characters were successfully read and appended to the result string.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_binary<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_binary}} -->
The `get_binary` function reads a specified number of bytes from an input source and stores them in a binary array.
- **Inputs**:
    - `format`: An enumeration value of type `input_format_t` that specifies the format of the input data.
    - `len`: A value of type `NumberType` that indicates the number of bytes to read.
    - `result`: A reference to a `binary_t` object where the read bytes will be stored.
- **Control Flow**:
    - The function initializes a boolean variable `success` to true to track the success of the read operation.
    - A for loop iterates from 0 to `len`, reading one byte at a time from the input source.
    - Within the loop, the `get()` function is called to read the next byte.
    - If the end of the input is unexpectedly reached (checked by `unexpect_eof`), `success` is set to false and the loop is exited.
    - The read byte is cast to `std::uint8_t` and pushed back into the `result` binary array.
- **Output**: The function returns a boolean value indicating whether the read operation was successful.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::get\_token\_string<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::get_token_string}} -->
Returns a hexadecimal string representation of the current character.
- **Inputs**: None
- **Control Flow**:
    - Creates a character array of size 3 to hold the hexadecimal representation.
    - Uses `std::snprintf` to format the `current` character as a two-digit hexadecimal string into the character array.
    - Returns the character array as a `std::string`.
- **Output**: A `std::string` containing the hexadecimal representation of the `current` character.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)


---
#### binary\_reader::exception\_message<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message}} -->
Generates a detailed error message for syntax errors based on the input format.
- **Inputs**:
    - `format`: An enumeration value of type `input_format_t` indicating the format of the input data (e.g., CBOR, MessagePack, etc.).
    - `detail`: A string containing specific details about the error encountered during parsing.
    - `context`: A string providing additional context about where the error occurred.
- **Control Flow**:
    - Initializes the error message with a base string indicating a syntax error.
    - Uses a switch statement to append the specific format type to the error message based on the `format` input.
    - If the format is not recognized, it triggers an assertion failure.
    - Concatenates the error message with the `context` and `detail` strings before returning.
- **Output**: Returns a formatted string that describes the syntax error, including the format type, context, and specific details.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_reader)  (Data Structure)



---
### parse\_event\_t<!-- {{#data_structure:namespace::parse_event_t}} -->
- **Type**: `enum class parse_event_t`
- **Members**:
    - `object_start`: Indicates the start of processing a JSON object.
    - `object_end`: Indicates the end of processing a JSON object.
    - `array_start`: Indicates the start of processing a JSON array.
    - `array_end`: Indicates the end of processing a JSON array.
    - `key`: Indicates that a key of a value in an object has been read.
    - `value`: Indicates that a JSON value has been fully read.
- **Description**: `parse_event_t` is an enumeration that defines various events that can occur during the parsing of JSON data, allowing the parser to track its state as it processes objects and arrays.


---
### parser<!-- {{#data_structure:namespace::parser}} -->
- **Type**: `class`
- **Members**:
    - `callback`: A callback function used for handling parsed JSON values.
    - `last_token`: The type of the last read token from the lexer.
    - `m_lexer`: An instance of the lexer used to read input.
    - `allow_exceptions`: A flag indicating whether to throw exceptions on errors.
- **Description**: The `parser` class is a template-based JSON parser that utilizes an input adapter and a lexer to read and parse JSON data. It provides functionality to parse JSON values, handle errors, and manage the parsing state through a callback mechanism. The class is designed to be flexible and can be configured to allow or disallow exceptions during parsing, making it suitable for various use cases in JSON processing.
- **Member Functions**:
    - [`namespace::parser::parser`](#parserparser)
    - [`namespace::parser::parse`](#parserparse)
    - [`namespace::parser::accept`](#parseraccept)
    - [`namespace::parser::sax_parse_internal`](#parsersax_parse_internal)
    - [`namespace::parser::get_token`](#parserget_token)
    - [`namespace::parser::exception_message`](#parserexception_message)

**Methods**

---
#### parser::parser<!-- {{#callable:namespace::parser::parser}} -->
Constructs a `parser` object that initializes a lexer and prepares to parse JSON input.
- **Inputs**:
    - `adapter`: An `InputAdapterType` that provides the input data for parsing.
    - `cb`: An optional callback function of type `parser_callback_t<BasicJsonType>` for handling parsed data.
    - `allow_exceptions_`: A boolean flag indicating whether exceptions should be thrown on errors.
    - `skip_comments`: A boolean flag indicating whether comments in the input should be skipped.
- **Control Flow**:
    - The constructor initializes the `callback`, `m_lexer`, and `allow_exceptions` member variables.
    - It then calls the `get_token()` method to read the first token from the input.
- **Output**: The constructor does not return a value but initializes the `parser` object for subsequent parsing operations.
- **Functions called**:
    - [`namespace::parser::get_token`](#parserget_token)
- **See also**: [`namespace::parser`](#namespaceparser)  (Data Structure)


---
#### parser::parse<!-- {{#callable:namespace::parser::parse}} -->
Parses JSON input and populates the result based on the specified strictness and callback.
- **Inputs**:
    - `strict`: A boolean indicating whether the parser should enforce strict parsing rules, requiring the input to be completely read.
    - `result`: A reference to a `BasicJsonType` object that will hold the parsed JSON value.
- **Control Flow**:
    - Checks if a callback function is provided; if so, it uses `json_sax_dom_callback_parser` to handle parsing.
    - Calls [`sax_parse_internal`](#parsersax_parse_internal) to perform the actual parsing.
    - In strict mode, verifies that the end of input is reached; if not, triggers a parse error.
    - Handles errors by checking if the parser encountered any issues and sets the result to discarded if so.
    - If the result is discarded, it sets the result to null.
    - If no callback is provided, it uses `json_sax_dom_parser` for parsing, following similar steps as above.
- **Output**: The function does not return a value but modifies the `result` parameter to contain the parsed JSON value or indicates an error state.
- **Functions called**:
    - [`namespace::parser::sax_parse_internal`](#parsersax_parse_internal)
    - [`namespace::parser::get_token`](#parserget_token)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
- **See also**: [`namespace::parser`](#namespaceparser)  (Data Structure)


---
#### parser::accept<!-- {{#callable:namespace::parser::accept}} -->
The `accept` function checks if the input is a valid JSON text.
- **Inputs**:
    - `strict`: A boolean flag indicating whether the last token is expected to be EOF (end of file). Default is true.
- **Control Flow**:
    - Creates an instance of `json_sax_acceptor<BasicJsonType>` to handle the parsing.
    - Calls the [`sax_parse`](#sax_parse) function with the `sax_acceptor` and the `strict` parameter.
    - Returns the result of the [`sax_parse`](#sax_parse) function, which indicates if the input is valid JSON.
- **Output**: Returns a boolean value indicating whether the input is a proper JSON text.
- **Functions called**:
    - [`sax_parse`](#sax_parse)
- **See also**: [`namespace::parser`](#namespaceparser)  (Data Structure)


---
#### parser::sax\_parse\_internal<!-- {{#callable:namespace::parser::sax_parse_internal}} -->
The `sax_parse_internal` function processes a JSON input stream using a SAX (Simple API for XML) style parser.
- **Inputs**:
    - `sax`: A pointer to a SAX handler that defines methods for handling different JSON token types.
- **Control Flow**:
    - The function enters an infinite loop to continuously parse tokens until the end of the input is reached.
    - It checks the type of the last token and processes it accordingly, handling JSON objects, arrays, and primitive values.
    - For each object or array, it pushes the current state onto a stack to track the hierarchy.
    - It uses a switch statement to handle different token types, invoking the appropriate SAX handler methods.
    - If an unexpected token is encountered, it triggers a parse error using the SAX handler.
    - The function continues parsing until the stack is empty, indicating that the entire JSON structure has been processed.
- **Output**: Returns true if the parsing was successful and the entire JSON structure was processed; otherwise, it returns false in case of an error.
- **Functions called**:
    - [`namespace::parser::get_token`](#parserget_token)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_reader::exception_message`](#binary_readerexception_message)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`namespace::parser`](#namespaceparser)  (Data Structure)


---
#### parser::get\_token<!-- {{#callable:namespace::parser::get_token}} -->
The `get_token` function retrieves the next token from the lexer and updates the `last_token` member variable.
- **Inputs**:
    - `none`: This function does not take any input arguments.
- **Control Flow**:
    - The function calls the `scan` method of the `m_lexer` object to obtain the next token.
    - The result of the `scan` method is assigned to the `last_token` member variable.
    - The function returns the value of `last_token` after it has been updated.
- **Output**: The function returns the type of the token retrieved from the lexer, which is stored in the `last_token` variable.
- **See also**: [`namespace::parser`](#namespaceparser)  (Data Structure)


---
#### parser::exception\_message<!-- {{#callable:namespace::parser::exception_message}} -->
Generates a detailed error message for syntax errors encountered during parsing.
- **Inputs**:
    - `expected`: The expected `token_type` that the parser was anticipating.
    - `context`: A string providing context about where the error occurred during parsing.
- **Control Flow**:
    - Initializes the error message with 'syntax error'.
    - If the `context` string is not empty, appends it to the error message.
    - Appends a hyphen to the error message.
    - Checks if the last token was a `parse_error` and appends the lexer error message if true.
    - If the last token is not a parse error, appends the unexpected token type to the error message.
    - If the expected token is not `uninitialized`, appends the expected token type to the error message.
- **Output**: Returns a formatted string containing the complete error message detailing the syntax error.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`namespace::parser`](#namespaceparser)  (Data Structure)



---
### primitive\_iterator\_t<!-- {{#data_structure:namespace::primitive_iterator_t}} -->
- **Type**: `class`
- **Members**:
    - `m_it`: Holds the current position of the iterator as a signed integer.
    - `begin_value`: Static constant representing the beginning value of the iterator.
    - `end_value`: Static constant representing the end value of the iterator.
- **Description**: The `primitive_iterator_t` class represents a simple iterator that tracks its position using a signed integer type. It provides functionality to set the iterator to a defined beginning or end, check its current state, and perform arithmetic operations such as incrementing or decrementing the iterator. This class is designed to facilitate iteration over a range of values while maintaining a clear distinction between the start and end of the iteration.
- **Member Functions**:
    - [`namespace::primitive_iterator_t::get_value`](#primitive_iterator_tget_value)
    - [`namespace::primitive_iterator_t::set_begin`](#primitive_iterator_tset_begin)
    - [`namespace::primitive_iterator_t::set_end`](#primitive_iterator_tset_end)
    - [`namespace::primitive_iterator_t::is_begin`](#primitive_iterator_tis_begin)
    - [`namespace::primitive_iterator_t::is_end`](#primitive_iterator_tis_end)
    - [`namespace::primitive_iterator_t::operator==`](#primitive_iterator_toperator==)
    - [`namespace::primitive_iterator_t::operator<`](#primitive_iterator_toperator<)
    - [`namespace::primitive_iterator_t::operator+`](#primitive_iterator_toperator+)
    - [`namespace::primitive_iterator_t::operator-`](#primitive_iterator_toperator-)
    - [`namespace::primitive_iterator_t::operator++`](#primitive_iterator_toperator++)
    - [`namespace::primitive_iterator_t::operator++`](#primitive_iterator_toperator++)
    - [`namespace::primitive_iterator_t::operator--`](#primitive_iterator_toperator--)
    - [`namespace::primitive_iterator_t::operator--`](#primitive_iterator_toperator--)
    - [`namespace::primitive_iterator_t::operator+=`](#primitive_iterator_toperator+=)
    - [`namespace::primitive_iterator_t::operator-=`](#primitive_iterator_toperator-=)

**Methods**

---
#### primitive\_iterator\_t::get\_value<!-- {{#callable:namespace::primitive_iterator_t::get_value}} -->
The `get_value` function retrieves the current value of the iterator as a signed integer.
- **Inputs**:
    - `this`: A constant reference to the current instance of `primitive_iterator_t`, which is implied in member functions.
- **Control Flow**:
    - The function directly accesses the private member variable `m_it` of the class.
    - No conditional statements or loops are present, as it simply returns the value of `m_it`.
- **Output**: The function returns the current value of the iterator, represented as a `difference_type`, which is a signed integer type.
- **See also**: [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)  (Data Structure)


---
#### primitive\_iterator\_t::set\_begin<!-- {{#callable:namespace::primitive_iterator_t::set_begin}} -->
The `set_begin` function resets the iterator to a predefined starting value.
- **Inputs**: None
- **Control Flow**:
    - The function directly assigns the value of `begin_value` to the member variable `m_it`.
- **Output**: The function does not return a value; it modifies the internal state of the `primitive_iterator_t` object by setting the iterator to the beginning.
- **See also**: [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)  (Data Structure)


---
#### primitive\_iterator\_t::set\_end<!-- {{#callable:namespace::primitive_iterator_t::set_end}} -->
The `set_end` function sets the iterator to a predefined end value.
- **Inputs**: None
- **Control Flow**:
    - The function directly assigns the `end_value` to the member variable `m_it`.
- **Output**: The function does not return a value; it modifies the internal state of the `primitive_iterator_t` class by setting the iterator to the end position.
- **See also**: [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)  (Data Structure)


---
#### primitive\_iterator\_t::is\_begin<!-- {{#callable:namespace::primitive_iterator_t::is_begin}} -->
Checks if the iterator is at the beginning position.
- **Inputs**: None
- **Control Flow**:
    - The function compares the member variable `m_it` with the static constant `begin_value`.
    - If `m_it` is equal to `begin_value`, the function returns true, indicating the iterator is at the beginning.
- **Output**: Returns a boolean value: true if the iterator is at the beginning, false otherwise.
- **See also**: [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)  (Data Structure)


---
#### primitive\_iterator\_t::is\_end<!-- {{#callable:namespace::primitive_iterator_t::is_end}} -->
Checks if the iterator is at the end position.
- **Inputs**: None
- **Control Flow**:
    - The function compares the current iterator position `m_it` with the predefined `end_value`.
    - If `m_it` is equal to `end_value`, the function returns true, indicating the iterator is at the end.
- **Output**: Returns a boolean value: true if the iterator is at the end position, false otherwise.
- **See also**: [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)  (Data Structure)


---
#### primitive\_iterator\_t::operator==<!-- {{#callable:namespace::primitive_iterator_t::operator==}} -->
Compares two `primitive_iterator_t` objects for equality based on their internal iterator values.
- **Inputs**:
    - `lhs`: The left-hand side `primitive_iterator_t` object to compare.
    - `rhs`: The right-hand side `primitive_iterator_t` object to compare.
- **Control Flow**:
    - The function checks if the internal iterator value `m_it` of the left-hand side object `lhs` is equal to that of the right-hand side object `rhs`.
    - It returns the result of the comparison as a boolean value.
- **Output**: Returns `true` if both `primitive_iterator_t` objects have the same internal iterator value, otherwise returns `false`.
- **See also**: [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)  (Data Structure)


---
#### primitive\_iterator\_t::operator<<!-- {{#callable:namespace::primitive_iterator_t::operator<}} -->
Compares two `primitive_iterator_t` objects to determine if the first is less than the second.
- **Inputs**:
    - `lhs`: The left-hand side `primitive_iterator_t` object to compare.
    - `rhs`: The right-hand side `primitive_iterator_t` object to compare.
- **Control Flow**:
    - The function directly accesses the private member `m_it` of both `lhs` and `rhs`.
    - It performs a comparison using the less-than operator (`<`) on the `m_it` values of the two iterators.
- **Output**: Returns a boolean value indicating whether the `m_it` value of `lhs` is less than that of `rhs`.
- **See also**: [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)  (Data Structure)


---
#### primitive\_iterator\_t::operator\+<!-- {{#callable:namespace::primitive_iterator_t::operator+}} -->
The `operator+` function allows for the addition of a specified integer offset to the current iterator position, returning a new `primitive_iterator_t` instance.
- **Inputs**:
    - `n`: An integer value of type `difference_type` that represents the offset to be added to the current iterator position.
- **Control Flow**:
    - Creates a copy of the current iterator instance named `result`.
    - Applies the `+=` operator to `result` with the input offset `n`.
    - Returns the modified `result` iterator.
- **Output**: Returns a new `primitive_iterator_t` object that represents the iterator position after adding the specified offset.
- **See also**: [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)  (Data Structure)


---
#### primitive\_iterator\_t::operator\+\+<!-- {{#callable:namespace::primitive_iterator_t::operator++}} -->
The `operator++(int)` function increments the iterator and returns the previous state of the iterator.
- **Inputs**:
    - `int`: An integer parameter that indicates the post-increment operation.
- **Control Flow**:
    - The function creates a copy of the current state of the iterator (`result`).
    - It increments the internal iterator value (`m_it`) by one.
    - Finally, it returns the copy of the iterator before the increment.
- **Output**: The function returns a `primitive_iterator_t` object representing the state of the iterator before it was incremented.
- **See also**: [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)  (Data Structure)


---
#### primitive\_iterator\_t::operator\-\-<!-- {{#callable:namespace::primitive_iterator_t::operator--}} -->
The `operator--` function implements the post-decrement operation for the `primitive_iterator_t` class.
- **Inputs**:
    - `int`: An integer value that indicates the post-decrement operation, but is not used in the function.
- **Control Flow**:
    - The function creates a copy of the current iterator state and stores it in `result`.
    - It then decrements the internal iterator value `m_it` by one.
    - Finally, it returns the original state of the iterator before the decrement.
- **Output**: The function returns a `primitive_iterator_t` object representing the state of the iterator before the decrement operation.
- **See also**: [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)  (Data Structure)


---
#### primitive\_iterator\_t::operator\-=<!-- {{#callable:namespace::primitive_iterator_t::operator-=}} -->
The `operator-=` function decrements the internal iterator value by a specified difference.
- **Inputs**:
    - `n`: A signed integer representing the amount by which to decrement the iterator.
- **Control Flow**:
    - The function directly modifies the internal state of the `primitive_iterator_t` instance by subtracting `n` from `m_it`.
    - It then returns a reference to the current instance of `primitive_iterator_t`.
- **Output**: The function returns a reference to the modified `primitive_iterator_t` object, allowing for chained operations.
- **See also**: [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)  (Data Structure)



---
### internal\_iterator<!-- {{#data_structure:NLOHMANN_JSON_NAMESPACE_BEGIN::internal_iterator}} -->
- **Type**: `struct`
- **Members**:
    - `object_iterator`: An iterator for traversing JSON objects.
    - `array_iterator`: An iterator for traversing JSON arrays.
    - `primitive_iterator`: A generic iterator for traversing all other JSON types.
- **Description**: The `internal_iterator` struct is designed to facilitate iteration over different types of JSON data structures, including objects, arrays, and other primitive types, by providing specific iterators for each type.


---
### iter\_impl<!-- {{#data_structure:namespace::set_end::iter_impl}} -->
- **Type**: `class`
- **Members**:
    - `iterator_category`: Defines the category of the iterator as bidirectional.
    - `value_type`: Specifies the type of values when the iterator is dereferenced.
    - `difference_type`: Represents the type used for differences between iterators.
    - `pointer`: Defines a pointer type to the value being iterated over.
    - `reference`: Defines a reference type to the value being iterated over.
- **Description**: The `iter_impl` class is a custom iterator designed to work with a `BasicJsonType`, providing bidirectional iteration capabilities over JSON objects and arrays. It includes type definitions for various iterator properties and ensures compatibility with both const and non-const instances of `BasicJsonType`. The class employs static assertions to validate that it meets the requirements of a bidirectional iterator and to ensure that it only operates on valid JSON types.
- **Member Functions**:
    - [`namespace::set_end::iter_impl::iter_impl`](#iter_impliter_impl)
    - [`namespace::set_end::iter_impl::~iter_impl`](#iter_impliter_impl)
    - [`namespace::set_end::iter_impl::iter_impl`](#iter_impliter_impl)
    - [`namespace::set_end::iter_impl::operator=`](#iter_imploperator=)
    - [`namespace::set_end::iter_impl::iter_impl`](#iter_impliter_impl)
    - [`namespace::set_end::iter_impl::iter_impl`](#iter_impliter_impl)
    - [`namespace::set_end::iter_impl::operator=`](#iter_imploperator=)
    - [`namespace::set_end::iter_impl::iter_impl`](#iter_impliter_impl)
    - [`namespace::set_end::iter_impl::operator=`](#iter_imploperator=)

**Methods**

---
#### iter\_impl::iter\_impl<!-- {{#callable:namespace::set_end::iter_impl::iter_impl}} -->
The `iter_impl` class serves as an iterator for a JSON-like data structure, allowing traversal of its elements.
- **Inputs**:
    - `object`: A pointer to a JSON object that the iterator will traverse, which must not be null.
- **Control Flow**:
    - The constructor initializes the iterator based on the type of the JSON object it points to.
    - It uses a switch statement to determine the type of the JSON data (object, array, or primitive) and initializes the appropriate iterator.
    - If the JSON object is of type object or array, it initializes the respective iterators; otherwise, it initializes a primitive iterator.
- **Output**: The output is an initialized iterator that can be used to traverse the elements of the specified JSON object.
- **See also**: [`namespace::set_end::iter_impl`](#set_enditer_impl)  (Data Structure)


---
#### iter\_impl::iter\_impl<!-- {{#callable:namespace::set_end::iter_impl::iter_impl}} -->
The `iter_impl` class provides move semantics for iterators used with JSON objects, allowing efficient transfer of ownership.
- **Inputs**:
    - `other`: An rvalue reference to another `iter_impl` instance, used for move construction or move assignment.
- **Control Flow**:
    - The move constructor and move assignment operator are defined as default, allowing the compiler to generate the necessary code for transferring resources.
    - No additional logic is implemented in these methods, as they rely on the default behavior provided by the compiler.
- **Output**: The output of the move operations is the current instance of `iter_impl`, with its resources transferred from the source instance.
- **See also**: [`namespace::set_end::iter_impl`](#set_enditer_impl)  (Data Structure)


---
#### iter\_impl::iter\_impl<!-- {{#callable:namespace::set_end::iter_impl::iter_impl}} -->
The `iter_impl` constructor initializes an iterator based on the type of a JSON object.
- **Inputs**:
    - `object`: A pointer to a JSON object that the iterator will operate on, which must not be null.
- **Control Flow**:
    - The constructor asserts that the input `object` is not null using `JSON_ASSERT`.
    - It checks the type of the JSON object using a switch statement on `m_object->m_data.m_type`.
    - If the type is `object`, it initializes `m_it.object_iterator`.
    - If the type is `array`, it initializes `m_it.array_iterator`.
    - For all other types (including `null`, `string`, `boolean`, `number`, etc.), it initializes `m_it.primitive_iterator`.
- **Output**: The constructor does not return a value but initializes the internal state of the `iter_impl` object based on the type of the provided JSON object.
- **Functions called**:
    - [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)
- **See also**: [`namespace::set_end::iter_impl`](#set_enditer_impl)  (Data Structure)


---
#### iter\_impl::iter\_impl<!-- {{#callable:namespace::set_end::iter_impl::iter_impl}} -->
The `iter_impl` copy constructor creates a new iterator instance by copying the state from another constant iterator.
- **Inputs**:
    - `other`: A constant reference to another `iter_impl` object from which to copy the state.
- **Control Flow**:
    - The constructor initializes the new iterator's member variables `m_object` and `m_it` using the corresponding members from the `other` iterator.
    - No additional logic is performed, as this is a straightforward member-wise copy.
- **Output**: The function does not return a value; it initializes a new instance of `iter_impl` with the copied state from the provided constant iterator.
- **See also**: [`namespace::set_end::iter_impl`](#set_enditer_impl)  (Data Structure)


---
#### iter\_impl::iter\_impl<!-- {{#callable:namespace::set_end::iter_impl::iter_impl}} -->
The `iter_impl` copy constructor creates a new iterator instance by copying the state of another iterator.
- **Inputs**:
    - `other`: A constant reference to another `iter_impl` object from which to copy the state.
- **Control Flow**:
    - The constructor initializes the new iterator's member variables `m_object` and `m_it` using the corresponding members from the `other` iterator.
    - No additional logic is performed, as this is a straightforward member-wise copy.
- **Output**: The function does not return a value; it initializes a new instance of `iter_impl` with the copied state from the `other` iterator.
- **See also**: [`namespace::set_end::iter_impl`](#set_enditer_impl)  (Data Structure)



---
### json\_reverse\_iterator<!-- {{#data_structure:namespace::json_reverse_iterator}} -->
- **Type**: `class`
- **Members**:
    - `difference_type`: Defines the type used for the difference between iterators.
    - `base_iterator`: A type alias for the base class `std::reverse_iterator<Base>`.
    - `reference`: Defines the reference type for the pointed-to element in the iterator.
- **Description**: The `json_reverse_iterator` class is a template that extends the functionality of `std::reverse_iterator`, allowing for reverse iteration over a collection while providing additional methods to access keys and values of the elements being iterated. It supports various iterator operations such as incrementing, decrementing, and accessing elements, making it suitable for use with JSON-like data structures.
- **Member Functions**:
    - [`namespace::json_reverse_iterator::json_reverse_iterator`](#json_reverse_iteratorjson_reverse_iterator)
    - [`namespace::json_reverse_iterator::json_reverse_iterator`](#json_reverse_iteratorjson_reverse_iterator)
    - [`namespace::json_reverse_iterator::operator++`](#json_reverse_iteratoroperator++)
    - [`namespace::json_reverse_iterator::operator++`](#json_reverse_iteratoroperator++)
    - [`namespace::json_reverse_iterator::operator--`](#json_reverse_iteratoroperator--)
    - [`namespace::json_reverse_iterator::operator--`](#json_reverse_iteratoroperator--)
    - [`namespace::json_reverse_iterator::operator+=`](#json_reverse_iteratoroperator+=)
    - [`namespace::json_reverse_iterator::operator+`](#json_reverse_iteratoroperator+)
    - [`namespace::json_reverse_iterator::operator-`](#json_reverse_iteratoroperator-)
    - [`namespace::json_reverse_iterator::operator-`](#json_reverse_iteratoroperator-)
    - [`namespace::json_reverse_iterator::operator[]`](llama.cpp/vendor/nlohmann/json.hpp#callable:namespace::json_reverse_iterator::operator[])
    - [`namespace::json_reverse_iterator::key`](#json_reverse_iteratorkey)
    - [`namespace::json_reverse_iterator::value`](#json_reverse_iteratorvalue)
- **Inherits From**:
    - `std::reverse_iterator<Base>`

**Methods**

---
#### json\_reverse\_iterator::json\_reverse\_iterator<!-- {{#callable:namespace::json_reverse_iterator::json_reverse_iterator}} -->
Constructs a `json_reverse_iterator` from a given base iterator.
- **Inputs**:
    - `it`: An iterator of the type defined by `base_iterator`, which serves as the base for the reverse iterator.
- **Control Flow**:
    - The constructor initializes the `json_reverse_iterator` by calling the constructor of its base class `std::reverse_iterator` with the provided iterator.
    - The `noexcept` specifier indicates that this constructor does not throw exceptions.
- **Output**: This constructor does not return a value but initializes a `json_reverse_iterator` object that can be used to iterate over a collection in reverse order.
- **See also**: [`namespace::json_reverse_iterator`](#namespacejson_reverse_iterator)  (Data Structure)


---
#### json\_reverse\_iterator::json\_reverse\_iterator<!-- {{#callable:namespace::json_reverse_iterator::json_reverse_iterator}} -->
The `json_reverse_iterator` constructor initializes a reverse iterator from a given base iterator.
- **Inputs**:
    - `it`: A constant reference to a `base_iterator` object, which is used to initialize the reverse iterator.
- **Control Flow**:
    - The constructor takes a `base_iterator` as an argument.
    - It calls the base class constructor of `std::reverse_iterator` with the provided iterator.
- **Output**: The constructor does not return a value but initializes the `json_reverse_iterator` object to point to the reverse of the elements in the provided base iterator.
- **See also**: [`namespace::json_reverse_iterator`](#namespacejson_reverse_iterator)  (Data Structure)


---
#### json\_reverse\_iterator::operator\+\+<!-- {{#callable:namespace::json_reverse_iterator::operator++}} -->
The `operator++(int)` method implements the post-increment operation for the `json_reverse_iterator` class.
- **Inputs**:
    - `int`: An integer argument that indicates the post-increment operation; it is not used in the function body but is required for the post-increment signature.
- **Control Flow**:
    - The method calls the base class's `operator++(1)` to perform the increment operation.
    - The result of the base class's increment operation is cast to `json_reverse_iterator` and returned.
- **Output**: The method returns a new `json_reverse_iterator` object that represents the state of the iterator after the post-increment operation.
- **See also**: [`namespace::json_reverse_iterator`](#namespacejson_reverse_iterator)  (Data Structure)


---
#### json\_reverse\_iterator::operator\+\+<!-- {{#callable:namespace::json_reverse_iterator::operator++}} -->
Implements the pre-increment operator for the `json_reverse_iterator` class, allowing the iterator to move to the previous element in a reverse iteration.
- **Inputs**:
    - `this`: A reference to the current instance of `json_reverse_iterator` that is being incremented.
- **Control Flow**:
    - Calls the pre-increment operator of the base class `std::reverse_iterator` to move the iterator to the previous element.
    - Returns a reference to the current instance of `json_reverse_iterator` after the increment operation.
- **Output**: Returns a reference to the updated `json_reverse_iterator` instance, allowing for chaining of increment operations.
- **See also**: [`namespace::json_reverse_iterator`](#namespacejson_reverse_iterator)  (Data Structure)


---
#### json\_reverse\_iterator::operator\-\-<!-- {{#callable:namespace::json_reverse_iterator::operator--}} -->
The `operator--(int)` method in the `json_reverse_iterator` class implements the post-decrement operation for reverse iterators.
- **Inputs**:
    - `this`: A reference to the current instance of `json_reverse_iterator` on which the post-decrement operation is being performed.
- **Control Flow**:
    - The method calls the base class's post-decrement operator (`base_iterator::operator--(1)`), which decrements the iterator by one position.
    - The result of the base class operation is then cast to `json_reverse_iterator` and returned.
- **Output**: The method returns a new `json_reverse_iterator` that points to the element before the current position, effectively representing the state of the iterator before the decrement.
- **See also**: [`namespace::json_reverse_iterator`](#namespacejson_reverse_iterator)  (Data Structure)


---
#### json\_reverse\_iterator::operator\-\-<!-- {{#callable:namespace::json_reverse_iterator::operator--}} -->
The `operator--` function implements the pre-decrement operation for the `json_reverse_iterator` class.
- **Inputs**:
    - `this`: A reference to the current instance of `json_reverse_iterator` on which the pre-decrement operation is performed.
- **Control Flow**:
    - The function calls the pre-decrement operator of the base class `base_iterator`.
    - It uses `static_cast` to convert the result back to a reference of `json_reverse_iterator`.
- **Output**: The function returns a reference to the current `json_reverse_iterator` instance after decrementing its position.
- **See also**: [`namespace::json_reverse_iterator`](#namespacejson_reverse_iterator)  (Data Structure)


---
#### json\_reverse\_iterator::operator\+=<!-- {{#callable:namespace::json_reverse_iterator::operator+=}} -->
The `operator+=` function adds a specified integer offset to a `json_reverse_iterator` instance.
- **Inputs**:
    - `i`: An integer value of type `difference_type` that represents the offset to be added to the iterator.
- **Control Flow**:
    - The function calls the base class's `operator+=` method, which is defined in `std::reverse_iterator`, passing the integer offset `i` as an argument.
    - The result of the base class's operation is cast back to a reference of type `json_reverse_iterator` and returned.
- **Output**: Returns a reference to the updated `json_reverse_iterator` after adding the specified offset.
- **See also**: [`namespace::json_reverse_iterator`](#namespacejson_reverse_iterator)  (Data Structure)


---
#### json\_reverse\_iterator::operator\+<!-- {{#callable:namespace::json_reverse_iterator::operator+}} -->
The `operator+` function in the `json_reverse_iterator` class allows for the addition of a specified integer offset to the current iterator position, returning a new `json_reverse_iterator` instance.
- **Inputs**:
    - `i`: An integer value of type `difference_type` representing the offset to be added to the current iterator position.
- **Control Flow**:
    - The function calls the base class's `operator+` method with the provided offset `i`.
    - The result of the base class's `operator+` is cast to `json_reverse_iterator` and returned.
- **Output**: The function outputs a new `json_reverse_iterator` that points to the position obtained by adding the offset `i` to the current iterator.
- **See also**: [`namespace::json_reverse_iterator`](#namespacejson_reverse_iterator)  (Data Structure)


---
#### json\_reverse\_iterator::operator\-<!-- {{#callable:namespace::json_reverse_iterator::operator-}} -->
The `operator-` function in the `json_reverse_iterator` class allows for the creation of a new reverse iterator by subtracting a specified difference from the current iterator position.
- **Inputs**:
    - `i`: A value of type `difference_type` that specifies the number of positions to subtract from the current iterator.
- **Control Flow**:
    - The function calls the base class's `operator-` method with the input parameter `i`.
    - The result of the base class's operation is cast to the type `json_reverse_iterator`.
- **Output**: The function returns a new `json_reverse_iterator` that points to the position obtained by subtracting `i` from the current iterator.
- **See also**: [`namespace::json_reverse_iterator`](#namespacejson_reverse_iterator)  (Data Structure)


---
#### json\_reverse\_iterator::operator\-<!-- {{#callable:namespace::json_reverse_iterator::operator-}} -->
Calculates the difference in positions between two `json_reverse_iterator` instances.
- **Inputs**:
    - `other`: Another `json_reverse_iterator` instance to compare against.
- **Control Flow**:
    - Converts the current `json_reverse_iterator` instance and the `other` instance to their corresponding base iterators.
    - Calculates the difference between the two base iterators using the subtraction operator.
- **Output**: Returns the difference as a `difference_type`, which indicates the number of elements between the two iterators.
- **See also**: [`namespace::json_reverse_iterator`](#namespacejson_reverse_iterator)  (Data Structure)


---
#### json\_reverse\_iterator::key<!-- {{#callable:namespace::json_reverse_iterator::key}} -->
Returns the key of the element pointed to by the reverse iterator.
- **Inputs**: None
- **Control Flow**:
    - The function first decrements the base iterator using `--this->base()` to access the previous element in the reverse iteration.
    - It then calls the `key()` method on the decremented iterator to retrieve the key of the current element.
- **Output**: The output is the key of the element that the reverse iterator currently points to, with the type determined by the `key()` method of the base iterator's element.
- **See also**: [`namespace::json_reverse_iterator`](#namespacejson_reverse_iterator)  (Data Structure)


---
#### json\_reverse\_iterator::value<!-- {{#callable:namespace::json_reverse_iterator::value}} -->
Returns a reference to the value pointed to by the reverse iterator.
- **Inputs**: None
- **Control Flow**:
    - Decrements the base iterator by one using `--this->base()` to point to the previous element.
    - Dereferences the decremented iterator to obtain the value it points to using `it.operator*()`.
- **Output**: A reference to the value of the element that the reverse iterator currently points to.
- **See also**: [`namespace::json_reverse_iterator`](#namespacejson_reverse_iterator)  (Data Structure)



---
### json\_default\_base<!-- {{#data_structure:namespace::json_default_base}} -->
- **Type**: `struct`
- **Description**: `json_default_base` is an empty structure that serves as a base type for other JSON-related structures, potentially providing a common interface or functionality.


---
### json\_ref<!-- {{#data_structure:namespace::json_ref}} -->
- **Type**: `class`
- **Members**:
    - `owned_value`: Holds a mutable instance of the JSON value.
    - `value_ref`: Points to a constant reference of a JSON value.
- **Description**: The `json_ref` class template is designed to manage JSON values, allowing for both ownership and reference semantics. It can hold a JSON value either by owning it or by referencing an existing value, enabling efficient memory management and flexibility in handling JSON data. The class supports move semantics, ensuring that owned values can be transferred without unnecessary copies, while also providing a mechanism to access the underlying value through dereferencing.
- **Member Functions**:
    - [`namespace::json_ref::json_ref`](#json_refjson_ref)
    - [`namespace::json_ref::json_ref`](#json_refjson_ref)
    - [`namespace::json_ref::json_ref`](#json_refjson_ref)
    - [`namespace::json_ref::json_ref`](#json_refjson_ref)
    - [`namespace::json_ref::json_ref`](#json_refjson_ref)
    - [`namespace::json_ref::json_ref`](#json_refjson_ref)
    - [`namespace::json_ref::operator=`](#json_refoperator=)
    - [`namespace::json_ref::operator=`](#json_refoperator=)
    - [`namespace::json_ref::~json_ref`](#json_refjson_ref)
    - [`namespace::json_ref::moved_or_copied`](#json_refmoved_or_copied)
    - [`namespace::json_ref::operator*`](#json_refoperator*)
    - [`namespace::json_ref::operator->`](#json_refoperator->)

**Methods**

---
#### json\_ref::json\_ref<!-- {{#callable:namespace::json_ref::json_ref}} -->
The `json_ref` constructor initializes an instance by moving a value into an owned member variable.
- **Inputs**:
    - `value`: An rvalue reference of type `value_type`, which is the type of the JSON value being managed.
- **Control Flow**:
    - The constructor uses `std::move` to transfer ownership of the provided `value` to the member variable `owned_value`.
    - This ensures that the `json_ref` instance takes control of the resource represented by `value`, avoiding unnecessary copies.
- **Output**: The constructor does not return a value but initializes the `json_ref` object with the moved `owned_value`.
- **See also**: [`namespace::json_ref`](#namespacejson_ref)  (Data Structure)


---
#### json\_ref::json\_ref<!-- {{#callable:namespace::json_ref::json_ref}} -->
The `json_ref` constructor initializes a reference to a constant value of type `value_type`.
- **Inputs**:
    - `value`: A constant reference to a value of type `value_type`, which is the type of the JSON object being referenced.
- **Control Flow**:
    - The constructor initializes the member variable `value_ref` with the address of the provided constant reference `value`.
    - No additional control flow or logic is present in this constructor, as it directly assigns the input to a member variable.
- **Output**: This constructor does not return a value but initializes an instance of `json_ref` that holds a reference to the provided constant value.
- **See also**: [`namespace::json_ref`](#namespacejson_ref)  (Data Structure)


---
#### json\_ref::json\_ref<!-- {{#callable:namespace::json_ref::json_ref}} -->
The `json_ref` constructor initializes an instance of `json_ref` using an initializer list of `json_ref` objects.
- **Inputs**:
    - `init`: An initializer list of `json_ref` objects used to initialize the `owned_value` member.
- **Control Flow**:
    - The constructor takes an initializer list as an argument.
    - It directly initializes the `owned_value` member with the provided initializer list.
- **Output**: The constructor does not return a value but initializes the `owned_value` member of the `json_ref` instance.
- **See also**: [`namespace::json_ref`](#namespacejson_ref)  (Data Structure)


---
#### json\_ref::json\_ref<!-- {{#callable:namespace::json_ref::json_ref}} -->
Constructs a `json_ref` object by forwarding arguments to initialize its owned value.
- **Inputs**:
    - `Args`: Variadic template arguments that are forwarded to construct the `owned_value`.
- **Control Flow**:
    - The function template checks if the `value_type` can be constructed with the provided arguments using `std::is_constructible`.
    - If the check passes, it initializes the `owned_value` member by forwarding the arguments using `std::forward`.
- **Output**: The function does not return a value; it initializes the `json_ref` object with an owned value.
- **See also**: [`namespace::json_ref`](#namespacejson_ref)  (Data Structure)


---
#### json\_ref::json\_ref<!-- {{#callable:namespace::json_ref::json_ref}} -->
The `json_ref` move constructor allows for the efficient transfer of ownership of a `json_ref` object without copying.
- **Inputs**:
    - `json_ref&&`: An rvalue reference to another `json_ref` object, which is being moved.
- **Control Flow**:
    - The move constructor is marked as `noexcept`, indicating that it does not throw exceptions.
    - The default implementation of the move constructor is used, which transfers ownership of the internal state from the source object to the new object.
- **Output**: The output is a new `json_ref` object that takes ownership of the resources from the moved-from object, leaving the moved-from object in a valid but unspecified state.
- **See also**: [`namespace::json_ref`](#namespacejson_ref)  (Data Structure)


---
#### json\_ref::json\_ref<!-- {{#callable:namespace::json_ref::json_ref}} -->
The `json_ref` class is designed to manage a JSON-like object, allowing for both ownership and reference semantics.
- **Inputs**:
    - `value`: An rvalue reference to a `BasicJsonType` object, which is moved into the `owned_value` member.
    - `value`: A constant reference to a `BasicJsonType` object, which is stored as a reference in `value_ref`.
    - `init`: An initializer list of `json_ref` objects used to initialize `owned_value`.
    - `Args`: A variadic template parameter pack that allows for constructing `owned_value` with any arguments that can construct a `BasicJsonType`.
- **Control Flow**:
    - The constructor initializes `owned_value` if an rvalue is provided, or sets `value_ref` if a const reference is provided.
    - If an initializer list is passed, it initializes `owned_value` with the list.
    - The class is designed to be movable only, preventing copy construction and assignment by deleting the corresponding methods.
    - The `moved_or_copied` method checks if `value_ref` is null; if so, it returns the moved `owned_value`, otherwise it dereferences `value_ref`.
- **Output**: The output of the `json_ref` class is a managed JSON-like object that can either own its value or reference an existing one, depending on how it was constructed.
- **See also**: [`namespace::json_ref`](#namespacejson_ref)  (Data Structure)


---
#### json\_ref::operator=<!-- {{#callable:namespace::json_ref::operator=}} -->
The `operator=` function is deleted to prevent assignment of `json_ref` objects.
- **Inputs**: None
- **Control Flow**:
    - The first `operator=` overload for copying is deleted, which prevents copying of `json_ref` instances.
    - The second `operator=` overload for moving is also deleted, which prevents moving of `json_ref` instances.
- **Output**: The function does not produce an output as it is deleted and cannot be called.
- **See also**: [`namespace::json_ref`](#namespacejson_ref)  (Data Structure)


---
#### json\_ref::\~json\_ref<!-- {{#callable:namespace::json_ref::~json_ref}} -->
The `json_ref` constructor initializes a `json_ref` object, either by taking ownership of a value or by referencing an existing value.
- **Inputs**:
    - `value`: An rvalue reference of type `value_type` to take ownership of.
    - `value`: A const lvalue reference of type `value_type` to create a reference to an existing value.
    - `init`: An initializer list of `json_ref` objects to initialize the owned value.
    - `Args`: A variadic template parameter pack that allows constructing `value_type` with any number of arguments.
- **Control Flow**:
    - The constructor checks the type of input provided to determine how to initialize the `json_ref` object.
    - If an rvalue is provided, it initializes `owned_value` using `std::move`.
    - If a const lvalue is provided, it sets `value_ref` to point to the provided value.
    - If an initializer list is provided, it initializes `owned_value` with the list.
    - If a variadic argument list is provided, it constructs `owned_value` using the forwarded arguments.
- **Output**: The constructor does not return a value but initializes the internal state of the `json_ref` object, allowing it to manage either an owned value or a reference to an existing value.
- **See also**: [`namespace::json_ref`](#namespacejson_ref)  (Data Structure)


---
#### json\_ref::moved\_or\_copied<!-- {{#callable:namespace::json_ref::moved_or_copied}} -->
The `moved_or_copied` function returns either a moved value from `owned_value` or a dereferenced value from `value_ref`.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `json_ref` class.
- **Control Flow**:
    - The function checks if `value_ref` is a null pointer.
    - If `value_ref` is null, it returns a moved version of `owned_value`.
    - If `value_ref` is not null, it returns the value pointed to by `value_ref`.
- **Output**: The output is of type `value_type`, which is either a moved instance of `owned_value` or a copy of the value pointed to by `value_ref`.
- **See also**: [`namespace::json_ref`](#namespacejson_ref)  (Data Structure)


---
#### json\_ref::operator\-><!-- {{#callable:namespace::json_ref::operator->}} -->
The `operator->` function returns a pointer to the underlying value of the `json_ref` object.
- **Inputs**: None
- **Control Flow**:
    - The function dereferences the current `json_ref` object using the `operator*` to obtain a reference to the value.
    - It then takes the address of that reference to return a pointer to the value.
- **Output**: The output is a pointer of type `value_type const*`, which points to the underlying value managed by the `json_ref` object.
- **See also**: [`namespace::json_ref`](#namespacejson_ref)  (Data Structure)



---
### output\_adapter\_protocol<!-- {{#data_structure:namespace::output_adapter_protocol}} -->
- **Type**: `struct`
- **Members**:
    - `write_character`: A pure virtual function that writes a single character of type `CharType`.
    - `write_characters`: A pure virtual function that writes a sequence of characters of type `CharType`.
- **Description**: The `output_adapter_protocol` is a templated struct that defines an interface for output adapters, requiring implementations to provide methods for writing single and multiple characters of a specified type, `CharType`. This struct serves as a base for creating various output adapters that can handle different character types, ensuring a consistent interface for character output.
- **Member Functions**:
    - [`namespace::output_adapter_protocol::~output_adapter_protocol`](#output_adapter_protocoloutput_adapter_protocol)
    - [`namespace::output_adapter_protocol::output_adapter_protocol`](#output_adapter_protocoloutput_adapter_protocol)
    - [`namespace::output_adapter_protocol::output_adapter_protocol`](#output_adapter_protocoloutput_adapter_protocol)
    - [`namespace::output_adapter_protocol::output_adapter_protocol`](#output_adapter_protocoloutput_adapter_protocol)
    - [`namespace::output_adapter_protocol::operator=`](#output_adapter_protocoloperator=)
    - [`namespace::output_adapter_protocol::operator=`](#output_adapter_protocoloperator=)

**Methods**

---
#### output\_adapter\_protocol::\~output\_adapter\_protocol<!-- {{#callable:namespace::output_adapter_protocol::~output_adapter_protocol}} -->
The `~output_adapter_protocol` is a virtual destructor for the `output_adapter_protocol` class, ensuring proper cleanup of derived classes.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual, allowing derived classes to override it and ensuring that the correct destructor is called for derived objects.
    - The default implementation is provided, which will be used if no derived class provides its own destructor.
- **Output**: The function does not return a value; it is responsible for cleaning up resources when an object of a derived class is destroyed.
- **See also**: [`namespace::output_adapter_protocol`](#namespaceoutput_adapter_protocol)  (Data Structure)


---
#### output\_adapter\_protocol::output\_adapter\_protocol<!-- {{#callable:namespace::output_adapter_protocol::output_adapter_protocol}} -->
`output_adapter_protocol` is a templated abstract base class that defines an interface for writing characters.
- **Inputs**:
    - `CharType`: A template parameter representing the character type that the output adapter will handle.
- **Control Flow**:
    - The class contains pure virtual functions `write_character` and `write_characters`, which must be implemented by derived classes.
    - The default constructor and copy constructor are provided, allowing for the creation and copying of `output_adapter_protocol` objects, although the class itself is intended to be abstract.
- **Output**: The class does not produce output directly; instead, it serves as a base for other classes that will implement the character writing functionality.
- **See also**: [`namespace::output_adapter_protocol`](#namespaceoutput_adapter_protocol)  (Data Structure)


---
#### output\_adapter\_protocol::output\_adapter\_protocol<!-- {{#callable:namespace::output_adapter_protocol::output_adapter_protocol}} -->
The `output_adapter_protocol` class is an abstract base class that defines a protocol for output adapters, allowing for writing single characters and strings.
- **Inputs**:
    - `CharType`: A template parameter representing the character type that the output adapter will handle.
- **Control Flow**:
    - The class defines pure virtual functions `write_character` and `write_characters`, which must be implemented by derived classes.
    - The class provides default implementations for the copy constructor, move constructor, copy assignment operator, and move assignment operator.
- **Output**: The class does not produce a direct output but serves as a base for other classes that implement the output functionality.
- **See also**: [`namespace::output_adapter_protocol`](#namespaceoutput_adapter_protocol)  (Data Structure)


---
#### output\_adapter\_protocol::output\_adapter\_protocol<!-- {{#callable:namespace::output_adapter_protocol::output_adapter_protocol}} -->
The `output_adapter_protocol` class provides a default move constructor and a copy assignment operator for managing output adapters.
- **Inputs**:
    - `output_adapter_protocol&&`: An rvalue reference to an `output_adapter_protocol` object, used for move construction.
    - `const output_adapter_protocol&`: A constant reference to an `output_adapter_protocol` object, used for copy assignment.
- **Control Flow**:
    - The move constructor utilizes the default move semantics provided by the compiler, allowing for efficient transfer of resources.
    - The copy assignment operator also uses the default behavior, which performs a member-wise copy of the object's data.
- **Output**: The output of these operations is the constructed or assigned `output_adapter_protocol` object, which maintains the integrity of the original object's state.
- **See also**: [`namespace::output_adapter_protocol`](#namespaceoutput_adapter_protocol)  (Data Structure)


---
#### output\_adapter\_protocol::operator=<!-- {{#callable:namespace::output_adapter_protocol::operator=}} -->
Moves the state of one `output_adapter_protocol` instance to another using move assignment.
- **Inputs**:
    - `other`: An rvalue reference to another `output_adapter_protocol` instance whose resources are to be moved.
- **Control Flow**:
    - The function is defined as a default move assignment operator, which means it uses the compiler-generated implementation.
    - It transfers ownership of resources from the source instance to the current instance without performing a deep copy.
    - The `noexcept` specifier indicates that this operation is guaranteed not to throw exceptions.
- **Output**: Returns a reference to the current instance after the move assignment operation.
- **See also**: [`namespace::output_adapter_protocol`](#namespaceoutput_adapter_protocol)  (Data Structure)



---
### output\_vector\_adapter<!-- {{#data_structure:namespace::output_vector_adapter}} -->
- **Type**: `class`
- **Members**:
    - `v`: A reference to a `std::vector` that stores characters.
- **Description**: The `output_vector_adapter` class is a template-based adapter that allows writing characters to a specified `std::vector`, enabling the storage of characters in a dynamic array format.
- **Member Functions**:
    - [`namespace::output_vector_adapter::output_vector_adapter`](#output_vector_adapteroutput_vector_adapter)
    - [`namespace::output_vector_adapter::write_character`](#output_vector_adapterwrite_character)
- **Inherits From**:
    - [`namespace::output_adapter_protocol::output_adapter_protocol`](#output_adapter_protocoloutput_adapter_protocol)

**Methods**

---
#### output\_vector\_adapter::output\_vector\_adapter<!-- {{#callable:namespace::output_vector_adapter::output_vector_adapter}} -->
The `output_vector_adapter` constructor initializes an instance by referencing a provided vector.
- **Inputs**:
    - `vec`: A reference to a `std::vector` of type `CharType` and allocator `AllocatorType` that will be used to store output characters.
- **Control Flow**:
    - The constructor takes a reference to a vector and initializes the member variable `v` with it.
    - No additional control flow or logic is present in the constructor.
- **Output**: The constructor does not return a value but sets up the `output_vector_adapter` instance to use the provided vector for character output.
- **See also**: [`namespace::output_vector_adapter`](#namespaceoutput_vector_adapter)  (Data Structure)


---
#### output\_vector\_adapter::write\_character<!-- {{#callable:namespace::output_vector_adapter::write_character}} -->
The `write_character` function appends a character to a vector.
- **Inputs**:
    - `c`: A character of type `CharType` that is to be added to the vector.
- **Control Flow**:
    - The function directly calls the `push_back` method on the vector `v` to add the character `c`.
- **Output**: The function does not return a value; it modifies the internal vector by adding the character.
- **See also**: [`namespace::output_vector_adapter`](#namespaceoutput_vector_adapter)  (Data Structure)



---
### output\_stream\_adapter<!-- {{#data_structure:namespace::output_stream_adapter}} -->
- **Type**: `class`
- **Members**:
    - `stream`: A reference to a basic output stream of type `CharType`.
- **Description**: The `output_stream_adapter` class is a template-based adapter that facilitates writing characters to a specified output stream, implementing the `output_adapter_protocol`. It allows for writing single characters as well as arrays of characters to the output stream.
- **Member Functions**:
    - [`namespace::output_stream_adapter::output_stream_adapter`](#output_stream_adapteroutput_stream_adapter)
    - [`namespace::output_stream_adapter::write_character`](#output_stream_adapterwrite_character)
- **Inherits From**:
    - [`namespace::output_adapter_protocol::output_adapter_protocol`](#output_adapter_protocoloutput_adapter_protocol)

**Methods**

---
#### output\_stream\_adapter::output\_stream\_adapter<!-- {{#callable:namespace::output_stream_adapter::output_stream_adapter}} -->
Constructs an `output_stream_adapter` that wraps a given output stream.
- **Inputs**:
    - `s`: A reference to a `std::basic_ostream<CharType>` object that the adapter will use for output operations.
- **Control Flow**:
    - The constructor initializes the member variable `stream` with the provided output stream reference.
    - No additional logic or control flow is present in the constructor.
- **Output**: The constructor does not return a value but initializes the `output_stream_adapter` object for subsequent output operations.
- **See also**: [`namespace::output_stream_adapter`](#namespaceoutput_stream_adapter)  (Data Structure)


---
#### output\_stream\_adapter::write\_character<!-- {{#callable:namespace::output_stream_adapter::write_character}} -->
The `write_character` function writes a single character to the associated output stream.
- **Inputs**:
    - `c`: The character of type `CharType` to be written to the output stream.
- **Control Flow**:
    - The function directly calls the `put` method on the `stream` object, which is a reference to a `std::basic_ostream<CharType>`.
    - There are no conditional statements or loops; the function performs a single operation.
- **Output**: The function does not return a value; it performs an output operation by writing the character to the stream.
- **See also**: [`namespace::output_stream_adapter`](#namespaceoutput_stream_adapter)  (Data Structure)



---
### output\_string\_adapter<!-- {{#data_structure:namespace::output_string_adapter}} -->
- **Type**: `class`
- **Members**:
    - `str`: A reference to a string of type `StringType` that stores the output characters.
- **Description**: The `output_string_adapter` class is a template-based output adapter that allows writing characters to a specified string, enabling the accumulation of output in a string format.
- **Member Functions**:
    - [`namespace::output_string_adapter::output_string_adapter`](#output_string_adapteroutput_string_adapter)
    - [`namespace::output_string_adapter::write_character`](#output_string_adapterwrite_character)
- **Inherits From**:
    - [`namespace::output_adapter_protocol::output_adapter_protocol`](#output_adapter_protocoloutput_adapter_protocol)

**Methods**

---
#### output\_string\_adapter::output\_string\_adapter<!-- {{#callable:namespace::output_string_adapter::output_string_adapter}} -->
The `output_string_adapter` constructor initializes an instance by referencing a given string.
- **Inputs**:
    - `s`: A reference to a string of type `StringType` that will be adapted for output.
- **Control Flow**:
    - The constructor takes a reference to a string and initializes the member variable `str` with it.
    - No additional logic or control flow is present in the constructor.
- **Output**: The constructor does not return a value but sets up the `output_string_adapter` to modify the provided string.
- **See also**: [`namespace::output_string_adapter`](#namespaceoutput_string_adapter)  (Data Structure)


---
#### output\_string\_adapter::write\_character<!-- {{#callable:namespace::output_string_adapter::write_character}} -->
The `write_character` function appends a single character to a string.
- **Inputs**:
    - `c`: The character of type `CharType` to be appended to the string.
- **Control Flow**:
    - The function directly calls `push_back` on the member variable `str` to add the character `c`.
    - There are no conditional statements or loops in this function.
- **Output**: The function does not return a value; it modifies the internal string by appending the character.
- **See also**: [`namespace::output_string_adapter`](#namespaceoutput_string_adapter)  (Data Structure)



---
### output\_adapter<!-- {{#data_structure:namespace::output_adapter}} -->
- **Type**: `class`
- **Members**:
    - `oa`: A shared pointer to an output adapter type that handles different output mechanisms.
- **Description**: The `output_adapter` class is a template-based class designed to facilitate output operations by wrapping various output mechanisms such as vectors, streams, and strings, allowing for flexible and type-safe output handling.
- **Member Functions**:
    - [`namespace::output_adapter::output_adapter`](#output_adapteroutput_adapter)
    - [`namespace::output_adapter::output_adapter`](#output_adapteroutput_adapter)
    - [`namespace::output_adapter::output_adapter`](#output_adapteroutput_adapter)

**Methods**

---
#### output\_adapter::output\_adapter<!-- {{#callable:namespace::output_adapter::output_adapter}} -->
The `output_adapter` constructor initializes an `output_adapter` instance using a shared pointer to an `output_vector_adapter` that wraps a provided vector.
- **Inputs**:
    - `vec`: A reference to a `std::vector` of type `CharType`, which is used to initialize the internal `output_vector_adapter`.
- **Control Flow**:
    - The constructor takes a reference to a vector as an argument.
    - It creates a shared pointer to an `output_vector_adapter` using the provided vector.
    - The shared pointer is assigned to the private member `oa`.
- **Output**: The constructor does not return a value but initializes the `output_adapter` instance with a shared pointer to an `output_vector_adapter`.
- **See also**: [`namespace::output_adapter`](#namespaceoutput_adapter)  (Data Structure)


---
#### output\_adapter::output\_adapter<!-- {{#callable:namespace::output_adapter::output_adapter}} -->
The `output_adapter` constructor initializes an output adapter for a given output stream.
- **Inputs**:
    - `s`: A reference to a `std::basic_ostream<CharType>` object that represents the output stream to be adapted.
- **Control Flow**:
    - The constructor takes a reference to an output stream as an argument.
    - It creates a shared pointer to an `output_stream_adapter<CharType>` initialized with the provided output stream.
    - The shared pointer is assigned to the private member `oa`.
- **Output**: The constructor does not return a value but initializes the `output_adapter` instance with an output stream adapter.
- **See also**: [`namespace::output_adapter`](#namespaceoutput_adapter)  (Data Structure)


---
#### output\_adapter::output\_adapter<!-- {{#callable:namespace::output_adapter::output_adapter}} -->
The `output_adapter` constructor initializes an output adapter for a given string reference.
- **Inputs**:
    - `s`: A reference to a string of type `StringType` that will be adapted for output.
- **Control Flow**:
    - The constructor takes a reference to a string and creates a shared pointer to an `output_string_adapter` initialized with that string.
    - The `output_string_adapter` is a template class that handles the specifics of outputting the string.
- **Output**: The constructor does not return a value but initializes the internal state of the `output_adapter` object with a shared pointer to the `output_string_adapter`.
- **See also**: [`namespace::output_adapter`](#namespaceoutput_adapter)  (Data Structure)



---
### bjdata\_version\_t<!-- {{#data_structure:NLOHMANN_JSON_NAMESPACE_BEGIN::bjdata_version_t}} -->
- **Type**: `enum class`
- **Members**:
    - `draft2`: Represents the draft version 2 of the bjdata format.
    - `draft3`: Represents the draft version 3 of the bjdata format.
- **Description**: `bjdata_version_t` is an enumeration class that defines the different versions of the bjdata format, specifically `draft2` and `draft3`, allowing for type-safe representation of these versions in code.


---
### binary\_writer<!-- {{#data_structure:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer}} -->
- **Type**: `class`
- **Members**:
    - `oa`: An output adapter used for writing serialized data.
- **Description**: The `binary_writer` class is a template class designed to serialize JSON values into various binary formats such as BSON, CBOR, MessagePack, and UBJSON. It utilizes an output adapter to handle the writing of serialized data, ensuring that the data is correctly formatted according to the specified binary format. The class provides methods for writing different types of JSON values, including objects, arrays, strings, numbers, and booleans, while managing the necessary type conversions and formatting required for each binary format.
- **Member Functions**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::binary_writer`](#binary_writerbinary_writer)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson`](#binary_writerwrite_bson)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_cbor`](#binary_writerwrite_cbor)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_msgpack`](#binary_writerwrite_msgpack)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_ubjson`](#binary_writerwrite_ubjson)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_entry_header_size`](#binary_writercalc_bson_entry_header_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_entry_header`](#binary_writerwrite_bson_entry_header)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_boolean`](#binary_writerwrite_bson_boolean)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_double`](#binary_writerwrite_bson_double)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_string_size`](#binary_writercalc_bson_string_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_string`](#binary_writerwrite_bson_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_null`](#binary_writerwrite_bson_null)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_integer_size`](#binary_writercalc_bson_integer_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_integer`](#binary_writerwrite_bson_integer)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_unsigned_size`](#binary_writercalc_bson_unsigned_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_unsigned`](#binary_writerwrite_bson_unsigned)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_object_entry`](#binary_writerwrite_bson_object_entry)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_array_size`](#binary_writercalc_bson_array_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_binary_size`](#binary_writercalc_bson_binary_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_array`](#binary_writerwrite_bson_array)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_binary`](#binary_writerwrite_bson_binary)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_element_size`](#binary_writercalc_bson_element_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_element`](#binary_writerwrite_bson_element)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_object_size`](#binary_writercalc_bson_object_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_object`](#binary_writerwrite_bson_object)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_cbor_float_prefix`](#binary_writerget_cbor_float_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_cbor_float_prefix`](#binary_writerget_cbor_float_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_msgpack_float_prefix`](#binary_writerget_msgpack_float_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_msgpack_float_prefix`](#binary_writerget_msgpack_float_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number_with_ubjson_prefix`](#binary_writerwrite_number_with_ubjson_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number_with_ubjson_prefix`](#binary_writerwrite_number_with_ubjson_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number_with_ubjson_prefix`](#binary_writerwrite_number_with_ubjson_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::ubjson_prefix`](#binary_writerubjson_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_ubjson_float_prefix`](#binary_writerget_ubjson_float_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_ubjson_float_prefix`](#binary_writerget_ubjson_float_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bjdata_ndarray`](#binary_writerwrite_bjdata_ndarray)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number`](#binary_writerwrite_number)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_compact_float`](#binary_writerwrite_compact_float)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::to_char_type`](#binary_writerto_char_type)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::to_char_type`](#binary_writerto_char_type)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::to_char_type`](#binary_writerto_char_type)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::to_char_type`](#binary_writerto_char_type)

**Methods**

---
#### binary\_writer::binary\_writer<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::binary_writer}} -->
Constructs a `binary_writer` object with a specified output adapter.
- **Inputs**:
    - `adapter`: An `output_adapter_t<CharType>` instance that specifies the output destination for the binary data.
- **Control Flow**:
    - The constructor initializes the member variable `oa` with the provided adapter using `std::move`.
    - It asserts that the adapter is valid using `JSON_ASSERT`.
- **Output**: The constructor does not return a value; it initializes the `binary_writer` instance.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson}} -->
The `write_bson` function serializes a JSON object to BSON format.
- **Inputs**:
    - `j`: A constant reference to a `BasicJsonType` object representing the JSON value to be serialized.
- **Control Flow**:
    - The function checks the type of the input JSON value `j` using a switch statement.
    - If the type is `value_t::object`, it calls the [`write_bson_object`](#binary_writerwrite_bson_object) method to serialize the object.
    - For all other types (including `null`, `array`, `string`, `boolean`, `number_integer`, `number_unsigned`, `number_float`, `binary`, and `discarded`), it throws a `type_error` indicating that only objects can be serialized to BSON.
- **Output**: The function does not return a value; it writes the BSON representation of the JSON object to an output adapter.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_object`](#binary_writerwrite_bson_object)
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_cbor<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_cbor}} -->
The `write_cbor` function serializes a JSON value into the Concise Binary Object Representation (CBOR) format.
- **Inputs**:
    - `j`: A constant reference to a `BasicJsonType` object representing the JSON value to be serialized.
- **Control Flow**:
    - The function begins by checking the type of the JSON value `j` using a switch statement.
    - For each case (e.g., null, boolean, integer, float, string, array, binary, object), it handles serialization differently based on the type.
    - For null, it writes a specific byte (0xF6) to indicate a null value.
    - For booleans, it writes either 0xF5 (true) or 0xF4 (false).
    - For integers, it checks if the number is positive or negative and writes the appropriate byte prefix followed by the number.
    - For floats, it checks for NaN and Infinity, writing specific bytes for these cases, or calls [`write_compact_float`](#binary_writerwrite_compact_float) for regular floats.
    - For strings, it writes the length of the string followed by the string data.
    - For arrays, it writes the size of the array and recursively calls `write_cbor` for each element.
    - For binary data, it writes the subtype if present, followed by the size and the data itself.
    - For objects, it writes the size and recursively calls `write_cbor` for each key-value pair.
- **Output**: The function does not return a value; instead, it writes the serialized CBOR data directly to an output adapter.
- **Functions called**:
    - [`namespace::to_char_type`](#namespaceto_char_type)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number`](#binary_writerwrite_number)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_compact_float`](#binary_writerwrite_compact_float)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_msgpack<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_msgpack}} -->
Serializes a JSON value into MessagePack format.
- **Inputs**:
    - `j`: A constant reference to a `BasicJsonType` object representing the JSON value to be serialized.
- **Control Flow**:
    - The function begins by checking the type of the input JSON value `j` using a switch statement.
    - For each case (e.g., null, boolean, number, string, array, binary, object), it handles serialization differently based on the type.
    - For null, it writes a specific byte (0xC0) to indicate a nil value.
    - For booleans, it writes 0xC3 for true and 0xC2 for false.
    - For integers, it checks if the number is positive or negative and writes the appropriate byte prefix followed by the number itself.
    - For floating-point numbers, it calls [`write_compact_float`](#binary_writerwrite_compact_float) to handle serialization.
    - For strings, it writes the length and then the string data.
    - For arrays, it writes the size and recursively calls `write_msgpack` for each element.
    - For binary data, it checks for subtypes and writes the appropriate control byte and data.
    - For objects, it writes the size and recursively calls `write_msgpack` for each key-value pair.
- **Output**: The function does not return a value; it writes the serialized MessagePack data directly to an output adapter.
- **Functions called**:
    - [`namespace::to_char_type`](#namespaceto_char_type)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number`](#binary_writerwrite_number)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_compact_float`](#binary_writerwrite_compact_float)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_ubjson<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_ubjson}} -->
The `write_ubjson` function serializes a JSON value into the UBJSON format.
- **Inputs**:
    - `j`: A constant reference to a `BasicJsonType` object representing the JSON value to serialize.
    - `use_count`: A boolean indicating whether to include the count of elements in arrays and objects.
    - `use_type`: A boolean indicating whether to include type prefixes for elements.
    - `add_prefix`: A boolean indicating whether to add a prefix character for the serialized value.
    - `use_bjdata`: A boolean indicating whether to write in BJData format.
    - `bjdata_version`: An enumeration value indicating the version of BJData to use.
- **Control Flow**:
    - The function begins by checking the type of the JSON value `j` using a switch statement.
    - For each case (null, boolean, number, string, array, binary, object), it handles serialization differently.
    - If the type is null, it writes 'Z' if `add_prefix` is true.
    - For booleans, it writes 'T' for true and 'F' for false if `add_prefix` is true.
    - For numbers, it calls [`write_number_with_ubjson_prefix`](#binary_writerwrite_number_with_ubjson_prefix) to handle serialization with appropriate prefixes.
    - For strings, it writes 'S' if `add_prefix` is true, followed by the string length and the string itself.
    - For arrays, it writes '[' if `add_prefix` is true, checks for type consistency, and serializes each element recursively.
    - For binary data, it writes '[' if `add_prefix` is true, handles type prefixes, and serializes the binary data.
    - For objects, it checks for BJData ndarray format, writes '{' if `add_prefix` is true, and serializes each key-value pair.
- **Output**: The function does not return a value; it writes the serialized UBJSON data directly to an output adapter.
- **Functions called**:
    - [`namespace::to_char_type`](#namespaceto_char_type)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number_with_ubjson_prefix`](#binary_writerwrite_number_with_ubjson_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::ubjson_prefix`](#binary_writerubjson_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bjdata_ndarray`](#binary_writerwrite_bjdata_ndarray)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::calc\_bson\_entry\_header\_size<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_entry_header_size}} -->
Calculates the size of a BSON entry header based on the name and JSON value.
- **Inputs**:
    - `name`: A string representing the name of the BSON entry.
    - `j`: A `BasicJsonType` object representing the JSON value associated with the BSON entry.
- **Control Flow**:
    - The function first checks if the `name` contains a null character (U+0000) using `find`.
    - If a null character is found, it throws an `out_of_range` exception indicating that BSON keys cannot contain null characters.
    - If no null character is found, it calculates the size of the BSON entry header as 1 (for the ID) plus the size of the `name` and an additional 1 for the null terminator.
- **Output**: Returns the total size of the BSON entry header as a `std::size_t` value.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson\_entry\_header<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_entry_header}} -->
Writes the BSON entry header consisting of an element type and a name.
- **Inputs**:
    - `name`: A string representing the name of the BSON entry.
    - `element_type`: An 8-bit unsigned integer representing the type of the BSON element.
- **Control Flow**:
    - The function first writes the `element_type` as a character to the output adapter.
    - Then, it writes the `name` string to the output, including a null terminator.
- **Output**: The function does not return a value; it writes the BSON entry header directly to the output adapter.
- **Functions called**:
    - [`namespace::to_char_type`](#namespaceto_char_type)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson\_boolean<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_boolean}} -->
Writes a BSON boolean element with a specified name.
- **Inputs**:
    - `name`: The name of the BSON element, represented as a string.
    - `value`: The boolean value to be written, which can be either true or false.
- **Control Flow**:
    - Calls [`write_bson_entry_header`](#binary_writerwrite_bson_entry_header) with the provided name and a fixed element type identifier for boolean (0x08).
    - Writes a character representing the boolean value, where true is represented by 0x01 and false by 0x00.
- **Output**: This function does not return a value; it writes the BSON boolean element directly to the output adapter.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_entry_header`](#binary_writerwrite_bson_entry_header)
    - [`namespace::to_char_type`](#namespaceto_char_type)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson\_double<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_double}} -->
Writes a BSON element with a double value.
- **Inputs**:
    - `name`: The name of the BSON element, represented as a string.
    - `value`: The double value to be written to the BSON document.
- **Control Flow**:
    - Calls [`write_bson_entry_header`](#binary_writerwrite_bson_entry_header) to write the entry header for the BSON element, specifying the element type as 0x01 (double).
    - Calls `write_number<double>` to write the actual double value to the output.
- **Output**: This function does not return a value; it writes the BSON representation of the double value directly to the output adapter.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_entry_header`](#binary_writerwrite_bson_entry_header)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::calc\_bson\_string\_size<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_string_size}} -->
Calculates the size of a BSON-encoded string based on its length.
- **Inputs**:
    - `value`: A constant reference to a string of type `string_t`, representing the string whose BSON size is to be calculated.
- **Control Flow**:
    - The function begins by calculating the size of a BSON string, which includes the size of a 32-bit integer (for the string length) and the size of the string itself.
    - It adds 1 to account for the null terminator that is required in BSON strings.
- **Output**: Returns a `std::size_t` value representing the total size required to store the BSON-encoded string, including the length prefix and null terminator.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson\_string<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_string}} -->
Writes a BSON string element with a specified name and value.
- **Inputs**:
    - `name`: The name of the BSON string element, represented as a `string_t`.
    - `value`: The string value to be written, also represented as a `string_t`.
- **Control Flow**:
    - Calls [`write_bson_entry_header`](#binary_writerwrite_bson_entry_header) to write the entry header for the BSON string, specifying the element type as 0x02 (string).
    - Calculates the size of the string value plus a null terminator and writes this size as a 32-bit integer using `write_number`.
    - Writes the actual string value followed by a null terminator using `oa->write_characters`.
- **Output**: This function does not return a value; it writes the BSON string element directly to the output adapter.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_entry_header`](#binary_writerwrite_bson_entry_header)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson\_null<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_null}} -->
Writes a BSON null element with a specified name.
- **Inputs**:
    - `name`: A string representing the name of the BSON null element.
- **Control Flow**:
    - Calls [`write_bson_entry_header`](#binary_writerwrite_bson_entry_header) with the provided name and the BSON type identifier for null (0x0A).
- **Output**: This function does not return a value; it writes the BSON null element directly to the output adapter.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_entry_header`](#binary_writerwrite_bson_entry_header)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::calc\_bson\_integer\_size<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_integer_size}} -->
Calculates the size of a BSON-encoded integer based on its value.
- **Inputs**:
    - `value`: A 64-bit signed integer whose size in BSON format is to be determined.
- **Control Flow**:
    - The function checks if the input `value` falls within the range of a 32-bit signed integer.
    - If `value` is within the range, it returns the size of a 32-bit integer.
    - If `value` exceeds the range, it returns the size of a 64-bit integer.
- **Output**: Returns the size in bytes required to store the integer in BSON format, either 4 bytes for a 32-bit integer or 8 bytes for a 64-bit integer.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson\_integer<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_integer}} -->
Writes a BSON integer element with a specified name and value.
- **Inputs**:
    - `name`: The name associated with the BSON integer element.
    - `value`: The integer value to be written, which can be either a 32-bit or 64-bit integer.
- **Control Flow**:
    - Checks if the provided `value` fits within the range of a 32-bit signed integer.
    - If it does, it writes a BSON entry header for a 32-bit integer and writes the value as a 32-bit integer.
    - If it does not fit, it writes a BSON entry header for a 64-bit integer and writes the value as a 64-bit integer.
- **Output**: No return value; the function writes the BSON integer directly to the output.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_entry_header`](#binary_writerwrite_bson_entry_header)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::calc\_bson\_unsigned\_size<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_unsigned_size}} -->
Calculates the size required to store an unsigned integer in BSON format.
- **Inputs**:
    - `value`: A `std::uint64_t` representing the unsigned integer whose BSON size is to be calculated.
- **Control Flow**:
    - The function checks if the input `value` is less than or equal to the maximum value of a signed 32-bit integer.
    - If true, it returns the size of a 32-bit integer (4 bytes).
    - If false, it returns the size of a 64-bit integer (8 bytes).
- **Output**: Returns a `std::size_t` indicating the number of bytes required to store the unsigned integer in BSON format, either 4 or 8 bytes.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson\_unsigned<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_unsigned}} -->
Writes a BSON element with an unsigned integer value.
- **Inputs**:
    - `name`: The name associated with the BSON element.
    - `j`: A `BasicJsonType` object containing the unsigned integer value to be written.
- **Control Flow**:
    - Checks if the unsigned integer value in `j` can fit in a 32-bit signed integer.
    - If it fits, writes a BSON entry header for a 32-bit integer and writes the value as a 32-bit signed integer.
    - If it does not fit in 32 bits, checks if it can fit in a 64-bit signed integer.
    - If it fits in 64 bits, writes a BSON entry header for a 64-bit integer and writes the value as a 64-bit signed integer.
    - If it exceeds 64 bits, writes a BSON entry header for a 64-bit unsigned integer and writes the value as a 64-bit unsigned integer.
- **Output**: The function does not return a value; it writes the BSON representation of the unsigned integer to the output.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_entry_header`](#binary_writerwrite_bson_entry_header)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson\_object\_entry<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_object_entry}} -->
Writes a BSON object entry with a specified name and value.
- **Inputs**:
    - `name`: The name of the BSON object entry, represented as a string.
    - `value`: The value of the BSON object entry, represented as an object of type BasicJsonType::object_t.
- **Control Flow**:
    - Calls [`write_bson_entry_header`](#binary_writerwrite_bson_entry_header) with the provided name and the BSON type identifier for an object (0x03).
    - Calls [`write_bson_object`](#binary_writerwrite_bson_object) to serialize the actual object value.
- **Output**: This function does not return a value; it writes the BSON object entry directly to the output adapter.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_entry_header`](#binary_writerwrite_bson_entry_header)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_object`](#binary_writerwrite_bson_object)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::calc\_bson\_array\_size<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_array_size}} -->
Calculates the size of a BSON array based on its elements.
- **Inputs**:
    - `value`: A constant reference to an array of type `BasicJsonType::array_t`, representing the BSON array whose size is to be calculated.
- **Control Flow**:
    - Initializes an index counter `array_index` to zero.
    - Uses `std::accumulate` to iterate over each element in the array, calculating the total size of the embedded documents by calling [`calc_bson_element_size`](#binary_writercalc_bson_element_size) for each element, while incrementing the `array_index` for each element.
    - Returns the total size, which includes the size of the array header (4 bytes for the size of the array), the size of the embedded documents, and an additional byte for the null terminator.
- **Output**: Returns a `std::size_t` representing the total size required to store the BSON array, including the size of the array header and the sizes of its elements.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_element_size`](#binary_writercalc_bson_element_size)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::calc\_bson\_binary\_size<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_binary_size}} -->
Calculates the size of a BSON binary value based on its size and a fixed header size.
- **Inputs**:
    - `value`: A reference to a binary value of type `BasicJsonType::binary_t`, which contains the binary data whose size is to be calculated.
- **Control Flow**:
    - The function begins by determining the size of a BSON binary value.
    - It adds the size of a 32-bit integer header (4 bytes) to the size of the binary data.
    - Finally, it adds 1 byte for the null terminator.
- **Output**: Returns the total size required to store the BSON binary value, which includes the size of the header, the size of the binary data, and the null terminator.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson\_array<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_array}} -->
Writes a BSON array element to an output adapter.
- **Inputs**:
    - `name`: The name associated with the BSON array element.
    - `value`: The array of values to be written in BSON format.
- **Control Flow**:
    - Calls [`write_bson_entry_header`](#binary_writerwrite_bson_entry_header) to write the header for the BSON array, specifying the type as array (0x04).
    - Calculates the size of the array using [`calc_bson_array_size`](#binary_writercalc_bson_array_size) and writes this size as a 32-bit integer.
    - Iterates over each element in the `value` array, writing each element to the output using [`write_bson_element`](#binary_writerwrite_bson_element), with the index as the name.
    - Writes a null character (0x00) to signify the end of the BSON array.
- **Output**: The function does not return a value; it writes the BSON representation of the array directly to the output adapter.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_entry_header`](#binary_writerwrite_bson_entry_header)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_array_size`](#binary_writercalc_bson_array_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_element`](#binary_writerwrite_bson_element)
    - [`namespace::to_char_type`](#namespaceto_char_type)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson\_binary<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_binary}} -->
Writes a BSON binary element with a specified name and value.
- **Inputs**:
    - `name`: The name associated with the BSON binary element, represented as a string.
    - `value`: The binary data to be written, encapsulated in a `binary_t` object.
- **Control Flow**:
    - Calls [`write_bson_entry_header`](#binary_writerwrite_bson_entry_header) to write the entry header for the binary element, specifying the element type as 0x05.
    - Writes the size of the binary data as a 32-bit integer using [`write_number`](#binary_writerwrite_number), ensuring the size is in little-endian format.
    - Checks if the binary data has a subtype; if so, writes the subtype as an 8-bit integer, otherwise writes 0x00.
    - Writes the actual binary data to the output adapter using `oa->write_characters`, converting the data pointer to the appropriate character type.
- **Output**: The function does not return a value; it writes the BSON binary element directly to the output adapter.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_entry_header`](#binary_writerwrite_bson_entry_header)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number`](#binary_writerwrite_number)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::calc\_bson\_element\_size<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_element_size}} -->
Calculates the size of a BSON element based on its name and JSON value.
- **Inputs**:
    - `name`: A string representing the name of the BSON element.
    - `j`: A `BasicJsonType` object representing the JSON value whose size is to be calculated.
- **Control Flow**:
    - The function first calculates the header size using [`calc_bson_entry_header_size`](#binary_writercalc_bson_entry_header_size).
    - It then checks the type of the JSON value `j` using a switch statement.
    - For each case (object, array, binary, boolean, float, integer, unsigned, string, null), it calculates the size accordingly by adding the header size to the size of the specific type.
    - If the type is 'discarded' or unrecognized, it asserts false, indicating an error.
- **Output**: Returns the total size of the BSON element, including the header and the size of the value.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_entry_header_size`](#binary_writercalc_bson_entry_header_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_object_size`](#binary_writercalc_bson_object_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_array_size`](#binary_writercalc_bson_array_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_binary_size`](#binary_writercalc_bson_binary_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_integer_size`](#binary_writercalc_bson_integer_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_unsigned_size`](#binary_writercalc_bson_unsigned_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_string_size`](#binary_writercalc_bson_string_size)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson\_element<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_element}} -->
The `write_bson_element` function serializes a JSON element to BSON format based on its type.
- **Inputs**:
    - `name`: A string representing the name of the BSON element.
    - `j`: A `BasicJsonType` object representing the JSON value to be serialized.
- **Control Flow**:
    - The function begins by checking the type of the JSON value `j` using a switch statement.
    - For each case corresponding to a specific JSON type (object, array, binary, boolean, number, string, null), it calls the appropriate helper function to handle the serialization.
    - If the type is `discarded` or an unrecognized type, it triggers an assertion failure.
- **Output**: The function does not return a value; instead, it writes the serialized BSON data directly to an output adapter.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_object_entry`](#binary_writerwrite_bson_object_entry)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_array`](#binary_writerwrite_bson_array)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_binary`](#binary_writerwrite_bson_binary)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_boolean`](#binary_writerwrite_bson_boolean)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_double`](#binary_writerwrite_bson_double)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_integer`](#binary_writerwrite_bson_integer)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_unsigned`](#binary_writerwrite_bson_unsigned)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_string`](#binary_writerwrite_bson_string)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_null`](#binary_writerwrite_bson_null)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::calc\_bson\_object\_size<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_object_size}} -->
Calculates the size of a BSON object based on its elements.
- **Inputs**:
    - `value`: A constant reference to an object of type `BasicJsonType::object_t`, representing the JSON object whose BSON size is to be calculated.
- **Control Flow**:
    - The function initializes a variable `document_size` by using `std::accumulate` to iterate over each element in the `value` object.
    - For each element, it calls the helper function [`calc_bson_element_size`](#binary_writercalc_bson_element_size) with the element's key and value to compute the size of that element.
    - The total size is computed by adding the size of the BSON document header (4 bytes for the size itself), the accumulated size of all elements, and an additional byte for the null terminator.
- **Output**: Returns a `std::size_t` representing the total size required to serialize the BSON representation of the given JSON object.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_element_size`](#binary_writercalc_bson_element_size)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bson\_object<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_object}} -->
The `write_bson_object` function serializes a BSON object by writing its size and each of its elements to an output adapter.
- **Inputs**:
    - `value`: A constant reference to an object of type `BasicJsonType::object_t`, representing the JSON object to be serialized.
- **Control Flow**:
    - The function first calculates the size of the BSON object using [`calc_bson_object_size`](#binary_writercalc_bson_object_size) and writes this size as a 32-bit integer.
    - It then iterates over each key-value pair in the object, calling [`write_bson_element`](#binary_writerwrite_bson_element) to serialize each element.
    - Finally, it writes a null character (0x00) to indicate the end of the BSON object.
- **Output**: The function does not return a value; it writes the serialized BSON data directly to the output adapter.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::calc_bson_object_size`](#binary_writercalc_bson_object_size)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bson_element`](#binary_writerwrite_bson_element)
    - [`namespace::to_char_type`](#namespaceto_char_type)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::get\_cbor\_float\_prefix<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_cbor_float_prefix}} -->
The `get_cbor_float_prefix` function returns the CBOR prefix for a single-precision float.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as a static constexpr function, meaning it can be evaluated at compile time.
    - It takes a single unused parameter of type `float`, which is ignored in the function body.
    - The function directly returns a character representation of the CBOR prefix for a single-precision float, which is 0xFA.
- **Output**: The output is a `CharType` representing the CBOR prefix for a single-precision float, specifically the value 0xFA.
- **Functions called**:
    - [`namespace::to_char_type`](#namespaceto_char_type)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::get\_cbor\_float\_prefix<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_cbor_float_prefix}} -->
Returns the CBOR prefix for a double-precision float.
- **Inputs**:
    - `unused`: A double value that is not used in the function.
- **Control Flow**:
    - The function directly returns a character representation of the CBOR prefix for double-precision floats.
    - No conditional logic or loops are present in the function.
- **Output**: Returns a `CharType` representing the CBOR prefix for a double-precision float, specifically the byte 0xFB.
- **Functions called**:
    - [`namespace::to_char_type`](#namespaceto_char_type)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::get\_msgpack\_float\_prefix<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_msgpack_float_prefix}} -->
Returns the MessagePack prefix for a 32-bit float.
- **Inputs**:
    - `unused`: A float value that is not used in the function.
- **Control Flow**:
    - The function directly returns a character representation of the MessagePack prefix for a 32-bit float, which is 0xCA.
- **Output**: Returns a `CharType` representing the MessagePack prefix for a 32-bit float.
- **Functions called**:
    - [`namespace::to_char_type`](#namespaceto_char_type)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::get\_msgpack\_float\_prefix<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_msgpack_float_prefix}} -->
Returns the MessagePack prefix for a double-precision floating point.
- **Inputs**:
    - `unused`: A double value that is not used in the function.
- **Control Flow**:
    - The function directly returns a character representation of the MessagePack prefix for a double-precision float (0xCB).
- **Output**: Returns a `CharType` representing the MessagePack prefix for a 64-bit float.
- **Functions called**:
    - [`namespace::to_char_type`](#namespaceto_char_type)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_number\_with\_ubjson\_prefix<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number_with_ubjson_prefix}} -->
Writes a floating-point number with an optional UBJSON prefix.
- **Inputs**:
    - `n`: The floating-point number to be written.
    - `add_prefix`: A boolean indicating whether to add a UBJSON prefix.
    - `use_bjdata`: A boolean indicating whether to use BJData format.
- **Control Flow**:
    - If 'add_prefix' is true, the function writes the appropriate UBJSON prefix for the floating-point number.
    - The function then calls 'write_number' to write the actual number, passing the 'use_bjdata' flag.
- **Output**: The function does not return a value; it writes the number directly to the output adapter.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_ubjson_float_prefix`](#binary_writerget_ubjson_float_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number`](#binary_writerwrite_number)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_number\_with\_ubjson\_prefix<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number_with_ubjson_prefix}} -->
Writes an unsigned integer with a UBJSON prefix to an output adapter.
- **Inputs**:
    - `n`: An unsigned integer value to be written.
    - `add_prefix`: A boolean indicating whether to add a type prefix before writing the number.
    - `use_bjdata`: A boolean indicating whether to use BJData format for writing.
- **Control Flow**:
    - Checks the value of `n` against various limits to determine its type.
    - If `n` is within the range of `int8`, it writes the prefix 'i' and calls [`write_number`](#binary_writerwrite_number) with `n` cast to `uint8_t`.
    - If `n` is within the range of `uint8`, it writes the prefix 'U' and calls [`write_number`](#binary_writerwrite_number) with `n` cast to `uint8_t`.
    - If `n` is within the range of `int16`, it writes the prefix 'I' and calls [`write_number`](#binary_writerwrite_number) with `n` cast to `int16_t`.
    - If `n` is within the range of `uint16` and `use_bjdata` is true, it writes the prefix 'u' and calls [`write_number`](#binary_writerwrite_number) with `n` cast to `uint16_t`.
    - If `n` is within the range of `int32`, it writes the prefix 'l' and calls [`write_number`](#binary_writerwrite_number) with `n` cast to `int32_t`.
    - If `n` is within the range of `uint32` and `use_bjdata` is true, it writes the prefix 'm' and calls [`write_number`](#binary_writerwrite_number) with `n` cast to `uint32_t`.
    - If `n` is within the range of `int64`, it writes the prefix 'L' and calls [`write_number`](#binary_writerwrite_number) with `n` cast to `int64_t`.
    - If `n` is within the range of `uint64` and `use_bjdata` is true, it writes the prefix 'M' and calls [`write_number`](#binary_writerwrite_number) with `n` cast to `uint64_t`.
    - If `n` exceeds all defined ranges, it writes the prefix 'H' for high-precision numbers, converts `n` to a JSON string, and writes its size and characters.
- **Output**: The function does not return a value; it writes the number directly to the output adapter.
- **Functions called**:
    - [`namespace::to_char_type`](#namespaceto_char_type)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number`](#binary_writerwrite_number)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number_with_ubjson_prefix`](#binary_writerwrite_number_with_ubjson_prefix)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_number\_with\_ubjson\_prefix<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number_with_ubjson_prefix}} -->
Writes a signed integer with an optional UBJSON prefix.
- **Inputs**:
    - `n`: The signed integer value to be written.
    - `add_prefix`: A boolean indicating whether to add a prefix character.
    - `use_bjdata`: A boolean indicating whether to use BJData format.
- **Control Flow**:
    - Checks if the input number falls within the range of various integer types (int8, uint8, int16, uint16, int32, uint32, int64, uint64).
    - For each range, if 'add_prefix' is true, it writes the corresponding prefix character to the output adapter.
    - If the number exceeds the defined ranges, it writes a high-precision prefix and serializes the number as a string.
- **Output**: The function does not return a value; it writes the number directly to the output adapter.
- **Functions called**:
    - [`namespace::to_char_type`](#namespaceto_char_type)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number`](#binary_writerwrite_number)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number_with_ubjson_prefix`](#binary_writerwrite_number_with_ubjson_prefix)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::ubjson\_prefix<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::ubjson_prefix}} -->
Determines the UBJSON prefix character for a given JSON value based on its type and value.
- **Inputs**:
    - `j`: A constant reference to a `BasicJsonType` object representing the JSON value whose prefix is to be determined.
    - `use_bjdata`: A boolean flag indicating whether to use BJData format for certain numeric types.
- **Control Flow**:
    - The function starts by checking the type of the JSON value `j` using a switch statement.
    - For each case (e.g., null, boolean, number types, string, array, object), it determines the appropriate prefix character based on the value's type and, for numbers, its range.
    - For integer types, it checks against various limits to decide whether to return a specific character for int8, uint8, int16, etc.
    - For floating-point numbers, it calls the [`get_ubjson_float_prefix`](#binary_writerget_ubjson_float_prefix) function to get the appropriate prefix.
    - If the type is not recognized or is discarded, it returns 'N' as the prefix.
- **Output**: Returns a character representing the UBJSON prefix for the given JSON value, which indicates its type and precision.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_ubjson_float_prefix`](#binary_writerget_ubjson_float_prefix)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::get\_ubjson\_float\_prefix<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_ubjson_float_prefix}} -->
Returns the UBJSON prefix character for a 32-bit float.
- **Inputs**:
    - `unused`: A `float` parameter that is not used in the function body.
- **Control Flow**:
    - The function is defined as a static constexpr, meaning it can be evaluated at compile time.
    - It directly returns the character 'd', which represents a 32-bit float in UBJSON format.
- **Output**: A `CharType` character representing the UBJSON prefix for a 32-bit float, specifically 'd'.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::get\_ubjson\_float\_prefix<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_ubjson_float_prefix}} -->
Returns the UBJSON prefix character for a double-precision floating point number.
- **Inputs**:
    - `unused`: A double value that is not used in the function body.
- **Control Flow**:
    - The function is defined as a static constexpr, indicating it can be evaluated at compile time.
    - It takes a single parameter of type double, which is not utilized within the function.
    - The function directly returns the character 'D', which represents a double-precision float in UBJSON format.
- **Output**: The function outputs a character 'D', indicating that the data type is a double-precision floating point number in UBJSON format.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_bjdata\_ndarray<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_bjdata_ndarray}} -->
Writes a BJData ndarray from a JSON object.
- **Inputs**:
    - `value`: A JSON object containing the array type, size, and data.
    - `use_count`: A boolean indicating whether to use count prefixes.
    - `use_type`: A boolean indicating whether to use type prefixes.
    - `bjdata_version`: The version of BJData format to use.
- **Control Flow**:
    - Defines a mapping of data types to their corresponding character representations.
    - Checks if the array type is valid by looking it up in the mapping.
    - Calculates the total length of the array data based on the size information.
    - Validates that the size of the array data matches the expected length.
    - Writes the opening character for the array, followed by the type and size information.
    - Iterates over the array data and writes each element according to its type.
- **Output**: Returns false if the writing is successful, true if there is an error in type or size.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_ubjson`](#binary_writerwrite_ubjson)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number`](#binary_writerwrite_number)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_number<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number}} -->
Writes a number to an output adapter, handling endianness.
- **Inputs**:
    - `n`: The number of type `NumberType` to be written.
    - `OutputIsLittleEndian`: A boolean indicating if the output should be in little-endian format.
- **Control Flow**:
    - Creates an array `vec` of size equal to `NumberType` to hold the bytes of the number.
    - Copies the bytes of the number `n` into the array `vec` using `std::memcpy`.
    - Checks if the system's endianness differs from the desired output endianness.
    - If they differ, reverses the byte order of `vec` using `std::reverse`.
    - Writes the bytes from `vec` to the output adapter using `oa->write_characters`.
- **Output**: The function does not return a value; it writes the number directly to the output adapter.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::write\_compact\_float<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_compact_float}} -->
Writes a compact representation of a floating-point number to an output adapter.
- **Inputs**:
    - `n`: The floating-point number to be written, of type `number_float_t`.
    - `format`: The format in which to write the number, specified by `detail::input_format_t`.
- **Control Flow**:
    - The function checks if the input number `n` can be represented as a `float` without loss of precision.
    - If `n` is within the range of `float` and can be accurately cast to `float`, it writes a prefix for the specified format (CBOR or MessagePack) followed by the number as a `float`.
    - If `n` cannot be represented as a `float`, it writes a prefix for the specified format followed by the number as a `double`.
- **Output**: The function does not return a value; it writes the compact representation of the number directly to the output adapter.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_cbor_float_prefix`](#binary_writerget_cbor_float_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::get_msgpack_float_prefix`](#binary_writerget_msgpack_float_prefix)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::write_number`](#binary_writerwrite_number)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::to\_char\_type<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::to_char_type}} -->
Converts a `std::uint8_t` value to a specified character type (`CharType`) based on the signedness of the character type.
- **Inputs**:
    - `x`: A `std::uint8_t` value that is to be converted to the character type.
- **Control Flow**:
    - The function checks if `CharType` is signed and if `char` is also signed.
    - If both conditions are true, it uses a reinterpret cast to convert the address of `x` to a pointer to `char` and dereferences it to return the character.
    - If `CharType` is signed but `char` is unsigned, it performs a static assertion to ensure the sizes match and uses `std::memcpy` to copy the value of `x` into a `CharType` variable before returning it.
    - If `CharType` is unsigned, it simply returns `x` as it is.
- **Output**: Returns the converted value of type `CharType` that corresponds to the input `std::uint8_t` value.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::to\_char\_type<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::to_char_type}} -->
Converts a `std::uint8_t` value to a specified character type (`CharType`) while ensuring type compatibility and size constraints.
- **Inputs**:
    - `x`: A `std::uint8_t` value that is to be converted to the specified character type.
- **Control Flow**:
    - The function begins by asserting that the size of `CharType` is equal to the size of `std::uint8_t` using `static_assert`.
    - It also checks that `CharType` is a trivial type, ensuring that it can be safely copied using `std::memcpy`.
    - A local variable `result` of type `CharType` is declared.
    - The value of `x` is copied into `result` using `std::memcpy`.
    - Finally, the function returns the converted value stored in `result`.
- **Output**: Returns a value of type `CharType` that represents the converted value of the input `std::uint8_t`.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::to\_char\_type<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::to_char_type}} -->
Converts a `std::uint8_t` value to a specified character type (`CharType`) based on its signedness.
- **Inputs**:
    - `x`: A `std::uint8_t` value that is to be converted to the character type.
- **Control Flow**:
    - The function checks if `CharType` is signed or unsigned.
    - If `CharType` is signed and `char` is also signed, it directly returns the reinterpretation of `x` as `char`.
    - If `CharType` is signed but `char` is unsigned, it performs a static assertion to ensure size compatibility and uses `std::memcpy` to copy the value of `x` into a `CharType` variable before returning it.
    - If `CharType` is unsigned, it simply returns `x` as it is.
- **Output**: Returns the converted value of type `CharType`.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)


---
#### binary\_writer::to\_char\_type<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer::to_char_type}} -->
Converts a `uint8_t` value to a specified character type.
- **Inputs**:
    - `x`: An input value of type `InputCharType` that is to be converted to `CharType`.
- **Control Flow**:
    - The function checks if the `CharType` is signed and if `InputCharType` is the same as `char`.
    - If the conditions are met, it returns the input value `x` directly.
    - If `CharType` is unsigned, it performs a static assertion to ensure the size matches `std::uint8_t` and uses `std::memcpy` to convert the value.
- **Output**: Returns the converted value of type `CharType`.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::binary_writer`](#NLOHMANN_JSON_NAMESPACE_BEGINbinary_writer)  (Data Structure)



---
### diyfp<!-- {{#data_structure:namespace::dtoa_impl::diyfp}} -->
- **Type**: `struct`
- **Members**:
    - `kPrecision`: A static constant representing the precision of the diyfp structure.
    - `f`: A 64-bit unsigned integer representing the significand.
    - `e`: An integer representing the exponent.
- **Description**: The `diyfp` structure represents a fixed-point number with a significand and an exponent, allowing for precise arithmetic operations while maintaining a specified precision. It includes fields for the significand (`f`) and exponent (`e`), and provides static methods for arithmetic operations such as subtraction, multiplication, and normalization.
- **Member Functions**:
    - [`namespace::dtoa_impl::diyfp::diyfp`](#diyfpdiyfp)
    - [`namespace::dtoa_impl::diyfp::sub`](#diyfpsub)
    - [`namespace::dtoa_impl::diyfp::mul`](#diyfpmul)
    - [`namespace::dtoa_impl::diyfp::normalize`](#diyfpnormalize)
    - [`namespace::dtoa_impl::diyfp::normalize_to`](#diyfpnormalize_to)

**Methods**

---
#### diyfp::diyfp<!-- {{#callable:namespace::dtoa_impl::diyfp::diyfp}} -->
Constructs a `diyfp` object representing a floating-point number with a significand `f` and an exponent `e`.
- **Inputs**:
    - `f_`: A `std::uint64_t` representing the significand of the floating-point number.
    - `e_`: An `int` representing the exponent of the floating-point number.
- **Control Flow**:
    - The constructor initializes the member variable `f` with the value of `f_`.
    - The constructor initializes the member variable `e` with the value of `e_`.
- **Output**: The constructor does not return a value but initializes a `diyfp` object with the specified significand and exponent.
- **See also**: [`namespace::dtoa_impl::diyfp`](#dtoa_impldiyfp)  (Data Structure)


---
#### diyfp::sub<!-- {{#callable:namespace::dtoa_impl::diyfp::sub}} -->
The `sub` function performs subtraction of two `diyfp` numbers, ensuring they have the same exponent and that the first number is greater than or equal to the second.
- **Inputs**:
    - `x`: The first `diyfp` number from which the second `diyfp` number will be subtracted.
    - `y`: The second `diyfp` number that will be subtracted from the first `diyfp` number.
- **Control Flow**:
    - The function first asserts that the exponents of `x` and `y` are equal using `JSON_ASSERT(x.e == y.e)`.
    - It then asserts that the significand of `x` is greater than or equal to that of `y` using `JSON_ASSERT(x.f >= y.f)`.
    - Finally, it returns a new `diyfp` object with the significand calculated as `x.f - y.f` and the exponent as `x.e`.
- **Output**: The output is a new `diyfp` object representing the result of the subtraction, with the significand being the difference of the significands of `x` and `y`, and the exponent being the same as that of `x`.
- **See also**: [`namespace::dtoa_impl::diyfp`](#dtoa_impldiyfp)  (Data Structure)


---
#### diyfp::mul<!-- {{#callable:namespace::dtoa_impl::diyfp::mul}} -->
Multiplies two `diyfp` numbers and returns the result as a new `diyfp` instance.
- **Inputs**:
    - `x`: The first `diyfp` number to be multiplied, containing a significand `f` and an exponent `e`.
    - `y`: The second `diyfp` number to be multiplied, also containing a significand `f` and an exponent `e`.
- **Control Flow**:
    - The function begins by asserting that the precision is set to 64 bits.
    - It extracts the lower and upper 32 bits of the significands `f` from both `x` and `y`.
    - It computes the products of the lower and upper parts of the significands, resulting in four partial products: `p0`, `p1`, `p2`, and `p3`.
    - The function calculates the high part of the product `Q` by summing the relevant parts of the partial products.
    - It rounds `Q` to account for precision, adjusting it upwards if necessary.
    - Finally, it computes the high part of the final product `h` and returns a new `diyfp` instance with `h` and the combined exponent.
- **Output**: Returns a new `diyfp` instance representing the product of `x` and `y`, with the significand rounded and the exponent adjusted accordingly.
- **See also**: [`namespace::dtoa_impl::diyfp`](#dtoa_impldiyfp)  (Data Structure)


---
#### diyfp::normalize<!-- {{#callable:namespace::dtoa_impl::diyfp::normalize}} -->
Normalizes a `diyfp` number by shifting its significand left until it is at least 2^(kPrecision-1).
- **Inputs**:
    - `x`: A `diyfp` structure representing a floating-point number with a significand `f` and an exponent `e`.
- **Control Flow**:
    - The function asserts that the significand `f` of `x` is not zero.
    - It enters a while loop that continues as long as the most significant bit of `f` is zero.
    - Within the loop, `f` is left-shifted by one bit, and the exponent `e` is decremented by one for each shift.
    - The loop effectively shifts the significand until it is at least 2^(kPrecision-1).
- **Output**: Returns the normalized `diyfp` structure with an updated significand and exponent.
- **See also**: [`namespace::dtoa_impl::diyfp`](#dtoa_impldiyfp)  (Data Structure)


---
#### diyfp::normalize\_to<!-- {{#callable:namespace::dtoa_impl::diyfp::normalize_to}} -->
Normalizes a `diyfp` number to a specified target exponent.
- **Inputs**:
    - `x`: A `diyfp` object representing the number to be normalized, which consists of a significand and an exponent.
    - `target_exponent`: An integer representing the exponent to which the `diyfp` number should be normalized.
- **Control Flow**:
    - Calculates the difference `delta` between the current exponent of `x` and the `target_exponent`.
    - Asserts that `delta` is non-negative, ensuring that the normalization does not require increasing the exponent.
    - Asserts that shifting the significand `x.f` left by `delta` bits and then right by `delta` bits results in the original significand, ensuring no bits are lost.
    - Returns a new `diyfp` object with the significand shifted left by `delta` and the exponent set to `target_exponent`.
- **Output**: A new `diyfp` object with the significand adjusted to match the `target_exponent` while maintaining its value.
- **See also**: [`namespace::dtoa_impl::diyfp`](#dtoa_impldiyfp)  (Data Structure)



---
### boundaries<!-- {{#data_structure:namespace::dtoa_impl::boundaries}} -->
- **Type**: `struct`
- **Members**:
    - `w`: Represents the width boundary as a `diyfp` type.
    - `minus`: Represents the lower boundary as a `diyfp` type.
    - `plus`: Represents the upper boundary as a `diyfp` type.
- **Description**: The `boundaries` struct is designed to encapsulate three boundary values, specifically a width and two limits (lower and upper), all represented as `diyfp` types, which likely denote a custom floating-point representation.


---
### cached\_power<!-- {{#data_structure:namespace::dtoa_impl::cached_power}} -->
- **Type**: `struct`
- **Members**:
    - `f`: The base value used in the power calculation.
    - `e`: The exponent representing the power of two.
    - `k`: An integer that approximates the logarithmic scale of the value.
- **Description**: The `cached_power` struct is designed to efficiently represent a power calculation in the form of `c = f * 2^e`, where `f` is a base value, `e` is the exponent, and `k` provides a logarithmic approximation of the value, facilitating quick computations in scenarios where powers of two are frequently used.


---
### error\_handler\_t<!-- {{#data_structure:NLOHMANN_JSON_NAMESPACE_BEGIN::error_handler_t}} -->
- **Type**: `enum class`
- **Members**:
    - `strict`: Indicates that a `type_error` exception should be thrown for invalid UTF-8.
    - `replace`: Specifies that invalid UTF-8 sequences should be replaced with U+FFFD.
    - `ignore`: Denotes that invalid UTF-8 sequences should be ignored.
- **Description**: The `error_handler_t` is an enumeration that defines three strategies for handling invalid UTF-8 sequences: `strict`, which throws an exception; `replace`, which substitutes invalid sequences with a replacement character; and `ignore`, which disregards them entirely.


---
### serializer<!-- {{#data_structure:NLOHMANN_JSON_NAMESPACE_BEGIN::serializer}} -->
- **Type**: `class`
- **Members**:
    - `o`: Output adapter for serialization.
    - `loc`: Locale information for number formatting.
    - `thousands_sep`: Character used as the thousands separator.
    - `decimal_point`: Character used as the decimal point.
    - `indent_char`: Character used for indentation.
    - `indent_string`: String used for indentation.
    - `error_handler`: Handler for error management during serialization.
- **Description**: The `serializer` class is designed to convert various data types into a serialized format, typically for JSON output. It utilizes an output adapter to write serialized data, manages locale-specific formatting for numbers, and provides options for pretty-printing and error handling during the serialization process. The class is equipped with methods to handle different data types, including objects, arrays, strings, and binary data, ensuring that the output is correctly formatted and escaped as necessary.
- **Member Functions**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::serializer`](#serializerserializer)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::serializer`](#serializerserializer)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::operator=`](#serializeroperator=)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::serializer`](#serializerserializer)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::operator=`](#serializeroperator=)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::~serializer`](#serializerserializer)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::dump`](#serializerdump)
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::switch`](#serializerswitch)

**Methods**

---
#### serializer::serializer<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::serializer}} -->
Constructs a `serializer` object for serializing JSON data with specified output adapter, indentation character, and error handling.
- **Inputs**:
    - `s`: An output adapter of type `output_adapter_t<char>` used to write serialized data.
    - `ichar`: A character used for indentation in the serialized output.
    - `error_handler_`: An optional parameter of type `error_handler_t` that defines the behavior on decoding errors, defaulting to strict error handling.
- **Control Flow**:
    - The constructor initializes member variables using the provided parameters and locale information.
    - It uses `std::move` to transfer ownership of the output adapter.
    - It retrieves the locale's thousands separator and decimal point, handling cases where they may be null.
    - The indentation string is initialized with a size of 512, filled with the specified indentation character.
- **Output**: The constructor does not return a value but initializes the `serializer` object for subsequent serialization operations.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer`](#NLOHMANN_JSON_NAMESPACE_BEGINserializer)  (Data Structure)


---
#### serializer::serializer<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::serializer}} -->
The `serializer` class is designed to serialize JSON data structures into a specified output format, with options for pretty-printing and ASCII escaping.
- **Inputs**:
    - `s`: An output adapter of type `output_adapter_t<char>` that specifies where the serialized output will be written.
    - `ichar`: A character used for indentation in the serialized output.
    - `error_handler_`: An optional parameter of type `error_handler_t` that defines how to handle decoding errors, defaulting to strict error handling.
- **Control Flow**:
    - The constructor initializes the output stream, locale settings for number formatting, indentation character, and error handling strategy.
    - Copy and move constructors and assignment operators are deleted to prevent copying of the `serializer` instances, which may contain pointer members.
    - The destructor is defaulted, indicating that the class manages its resources automatically.
- **Output**: The constructor does not return a value but initializes the `serializer` object for subsequent serialization operations.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer`](#NLOHMANN_JSON_NAMESPACE_BEGINserializer)  (Data Structure)


---
#### serializer::serializer<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::serializer}} -->
The `serializer` class has deleted move constructor and move assignment operator to prevent moving instances.
- **Inputs**:
    - `serializer&&`: A rvalue reference to a `serializer` object, which is not allowed to be moved.
- **Control Flow**:
    - The move constructor is deleted, preventing the use of move semantics for `serializer` instances.
    - The move assignment operator is also deleted, ensuring that `serializer` instances cannot be assigned via move.
- **Output**: The function does not produce an output as it is deleted; attempting to use it will result in a compilation error.
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer`](#NLOHMANN_JSON_NAMESPACE_BEGINserializer)  (Data Structure)


---
#### serializer::dump<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::dump}} -->
The `dump` function serializes a `BasicJsonType` object into a specified output format, supporting both pretty-printed and compact representations.
- **Inputs**:
    - `val`: A constant reference to a `BasicJsonType` object that contains the data to be serialized.
    - `pretty_print`: A boolean flag indicating whether the output should be formatted with indentation and newlines for readability.
    - `ensure_ascii`: A boolean flag that, if true, ensures all non-ASCII characters are escaped in the output.
    - `indent_step`: An unsigned integer that specifies the number of spaces to use for each indentation level.
    - `current_indent`: An optional unsigned integer that indicates the current indentation level, defaulting to 0.
- **Control Flow**:
    - The function begins by checking the type of the `val` object using a switch statement.
    - For `object` types, it checks if the object is empty and writes '{}' if true; otherwise, it iterates through the object's key-value pairs, recursively calling `dump` for each value.
    - If `pretty_print` is true, it adds indentation and newlines for better readability.
    - For `array` types, it similarly checks for emptiness and iterates through the array elements, applying the same recursive logic.
    - For `string`, `binary`, `boolean`, `number`, `discarded`, and `null` types, it directly writes the appropriate representation to the output.
    - The function handles each type distinctly, ensuring proper formatting based on the `pretty_print` flag.
- **Output**: The function outputs a serialized string representation of the `BasicJsonType` object to the specified output stream, formatted according to the provided parameters.
- **Functions called**:
    - [`dump_integer`](#dump_integer)
    - [`dump_float`](#dump_float)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer`](#NLOHMANN_JSON_NAMESPACE_BEGINserializer)  (Data Structure)


---
#### serializer::switch<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::serializer::switch}} -->
The `switch` statement processes the result of decoding a UTF-8 byte stream, handling valid code points, invalid bytes, and incomplete multi-byte sequences.
- **Inputs**:
    - `state`: The current state of the UTF-8 decoder, indicating whether it has accepted or rejected a byte.
    - `codepoint`: The decoded Unicode code point from the byte stream.
    - `byte`: The current byte being processed from the input stream.
- **Control Flow**:
    - The outer `switch` statement evaluates the result of the `decode` function, which can return `UTF8_ACCEPT`, `UTF8_REJECT`, or an incomplete state.
    - If `UTF8_ACCEPT` is returned, a nested `switch` statement processes the `codepoint`, handling specific control characters and escaping them appropriately.
    - If `UTF8_REJECT` is returned, the behavior depends on the `error_handler`, which can either throw an error, ignore the byte, or replace it with a replacement character.
    - If the state indicates an incomplete multi-byte sequence, the function increments the `undumped_chars` counter and may copy the byte directly to the buffer if `ensure_ascii` is false.
- **Output**: The function modifies the `string_buffer` to include escaped characters or replacement characters as necessary, and manages the writing of the buffer to the output stream when it reaches capacity.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)
    - [`hex_bytes`](#hex_bytes)
- **See also**: [`NLOHMANN_JSON_NAMESPACE_BEGIN::serializer`](#NLOHMANN_JSON_NAMESPACE_BEGINserializer)  (Data Structure)



---
### json\_value<!-- {{#data_structure:json_value}} -->
- **Type**: `union`
- **Members**:
    - `object`: Pointer to an `object_t` representing a JSON object.
    - `array`: Pointer to an `array_t` representing a JSON array.
    - `string`: Pointer to a `string_t` representing a JSON string.
    - `binary`: Pointer to a `binary_t` representing binary data.
    - `boolean`: A `boolean_t` representing a JSON boolean value.
    - `number_integer`: A `number_integer_t` representing a JSON integer value.
    - `number_unsigned`: A `number_unsigned_t` representing a JSON unsigned integer value.
    - `number_float`: A `number_float_t` representing a JSON floating-point value.
- **Description**: The `json_value` union is a versatile data structure designed to hold various types of JSON data, including objects, arrays, strings, binary data, booleans, and numeric values. Each member of the union is a pointer to a specific type, allowing for efficient memory usage by storing only the necessary data type at any given time. The union provides constructors for initializing its members with different JSON types, ensuring that the appropriate type is created and managed. This structure is essential for representing JSON data in a type-safe manner, facilitating operations such as serialization and deserialization.
- **Member Functions**:
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::json_value`](#json_valuejson_value)
    - [`json_value::destroy`](#json_valuedestroy)

**Methods**

---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
The `json_value` constructor initializes a `json_value` union instance to a default state.
- **Inputs**:
    - `none`: This constructor does not take any input arguments.
- **Control Flow**:
    - The constructor is defined as a default constructor, which means it initializes the union members to their default values.
    - No specific logic or branching occurs within this constructor as it simply sets up the union for future use.
- **Output**: The output is an instance of the `json_value` union, initialized to a default state, which can later hold different types of JSON values.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object representing a boolean.
- **Inputs**:
    - `v`: A boolean value (`boolean_t`) that initializes the `json_value` to represent either true or false.
- **Control Flow**:
    - The constructor initializes the `boolean` member of the `json_value` union with the provided boolean value `v`.
    - No additional control flow or logic is present in this constructor, as it directly assigns the input value to the member.
- **Output**: The constructor does not return a value but initializes a `json_value` instance that holds the boolean value.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object representing an integer number.
- **Inputs**:
    - `v`: An integer value of type `number_integer_t` to initialize the `json_value`.
- **Control Flow**:
    - The constructor initializes the `number_integer` member of the `json_value` union with the provided integer value `v`.
    - The constructor is marked as `noexcept`, indicating that it does not throw exceptions.
- **Output**: This constructor does not return a value; it initializes a `json_value` instance representing the integer.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object representing an unsigned integer.
- **Inputs**:
    - `v`: An unsigned integer value of type `number_unsigned_t` to initialize the `json_value`.
- **Control Flow**:
    - The constructor initializes the `number_unsigned` member of the `json_value` union with the provided unsigned integer value.
    - The `noexcept` specifier indicates that this constructor does not throw exceptions.
- **Output**: The constructor does not return a value but initializes a `json_value` instance that can represent an unsigned integer.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object representing a floating-point number.
- **Inputs**:
    - `v`: A floating-point number of type `number_float_t` to initialize the `json_value`.
- **Control Flow**:
    - The constructor initializes the `number_float` member of the `json_value` union with the provided floating-point value `v`.
    - The `noexcept` specifier indicates that this constructor does not throw exceptions.
- **Output**: This constructor does not return a value but initializes a `json_value` instance that holds a floating-point number.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object based on the specified `value_t` type.
- **Inputs**:
    - `t`: An enumeration of type `value_t` that specifies the type of JSON value to create (e.g., object, array, string, etc.).
- **Control Flow**:
    - The function uses a `switch` statement to determine the type of JSON value to create based on the input `t`.
    - For each case in the switch, it initializes the corresponding member of the `json_value` union with a newly created instance of the specified type.
    - If the type is `null` or `discarded`, it sets the `object` pointer to `nullptr` to avoid warnings.
    - In the case of `discarded`, it also checks if `t` is `null` and throws an error if so.
- **Output**: The function does not return a value; it initializes the `json_value` union member based on the provided type.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object that holds a string.
- **Inputs**:
    - `value`: A constant reference to a `string_t` object that represents the string value to be stored in the `json_value`.
- **Control Flow**:
    - The constructor initializes the `string` member of the `json_value` union by calling the `create<string_t>(value)` function.
    - The `create` function is expected to allocate memory and construct a `string_t` object using the provided `value`.
- **Output**: The constructor does not return a value but initializes the `json_value` instance to hold the specified string.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object from an rvalue reference to a string.
- **Inputs**:
    - `value`: An rvalue reference to a `string_t` object that represents the string value to be stored in the `json_value`.
- **Control Flow**:
    - The constructor initializes the `string` member of the `json_value` union by calling the `create<string_t>` function with the moved `value`.
    - The `std::move` function is used to efficiently transfer ownership of the `value` to the newly created `string_t` object.
- **Output**: The constructor does not return a value but initializes the `string` member of the `json_value` union to point to a newly created `string_t` object containing the provided string.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object that holds a copy of the provided `object_t`.
- **Inputs**:
    - `value`: A constant reference to an `object_t` which is used to initialize the `json_value`.
- **Control Flow**:
    - The constructor initializes the `object` member of the `json_value` union by calling the `create` function with the provided `value`.
    - The `create` function is expected to allocate memory and create a copy of the `object_t`.
- **Output**: The constructor does not return a value but initializes the `json_value` instance to hold a copy of the provided `object_t`.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object by moving an `object_t` value into it.
- **Inputs**:
    - `value`: An rvalue reference to an `object_t` that is being moved into the `json_value`.
- **Control Flow**:
    - The constructor initializes the `object` member of the `json_value` union by calling the `create` function with the moved `value`.
    - The `std::move` function is used to efficiently transfer ownership of the `value` to the `json_value` instance, avoiding unnecessary copies.
- **Output**: The constructor does not return a value but initializes the `json_value` instance with the provided `object_t`.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object that holds an array initialized with the provided `array_t` value.
- **Inputs**:
    - `value`: An `array_t` reference that is used to initialize the internal array pointer of the `json_value`.
- **Control Flow**:
    - The constructor initializes the `array` member of the `json_value` union by calling the `create` function with the provided `value`.
    - The `create` function is expected to allocate memory and construct an `array_t` object based on the input.
- **Output**: The constructor does not return a value but initializes the `json_value` instance to hold an array.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object that holds an array by moving the provided array.
- **Inputs**:
    - `value`: An rvalue reference to an `array_t` object that is to be moved into the `json_value`.
- **Control Flow**:
    - The constructor initializes the `array` member of the `json_value` union by calling the `create` function with the moved `value`.
    - The `std::move` function is used to efficiently transfer ownership of the `array_t` object, avoiding unnecessary copies.
- **Output**: The constructor does not return a value but initializes the `json_value` instance to hold the specified array.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object that holds a binary data type.
- **Inputs**:
    - `value`: A constant reference to a container type that holds binary data, used to initialize the `binary` member of the `json_value` union.
- **Control Flow**:
    - The constructor initializes the `binary` member of the `json_value` union by calling the `create<binary_t>(value)` function, which presumably allocates and initializes a `binary_t` object using the provided `value`.
- **Output**: The constructor does not return a value but initializes the `json_value` instance to hold binary data.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object that holds a binary data type.
- **Inputs**:
    - `value`: An rvalue reference to a container type that holds binary data, which will be moved into the `json_value`.
- **Control Flow**:
    - The constructor initializes the `binary` member of the `json_value` union.
    - It calls the `create<binary_t>(std::move(value))` function to allocate and initialize the binary data.
- **Output**: The constructor does not return a value but initializes the `json_value` instance to hold the provided binary data.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object that holds a binary value.
- **Inputs**:
    - `value`: A constant reference to a `binary_t` object that represents the binary data to be stored in the `json_value`.
- **Control Flow**:
    - The constructor initializes the `binary` member of the `json_value` union.
    - It calls the `create` function template with `binary_t` type, passing the input `value` to allocate and initialize the binary data.
- **Output**: The constructor does not return a value but initializes the `json_value` instance to hold the provided binary data.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::json\_value<!-- {{#callable:json_value::json_value}} -->
Constructs a `json_value` object that holds a `binary_t` value.
- **Inputs**:
    - `value`: An rvalue reference to a `binary_t` object that is to be moved into the `json_value`.
- **Control Flow**:
    - The constructor initializes the `binary` member of the `json_value` union.
    - It uses the `create` function to allocate and initialize a new `binary_t` object with the moved value.
- **Output**: This constructor does not return a value but initializes the `json_value` instance to hold the provided `binary_t`.
- **See also**: [`json_value`](#json_value)  (Data Structure)


---
#### json\_value::destroy<!-- {{#callable:json_value::destroy}} -->
The `destroy` function deallocates resources associated with a `json_value` based on its type.
- **Inputs**:
    - `t`: An enumeration value of type `value_t` that indicates the type of the `json_value` to be destroyed.
- **Control Flow**:
    - The function first checks if the `json_value` of type `t` is initialized; if not, it returns early.
    - If the type is `array` or `object`, it flattens the structure into a stack for processing.
    - It moves the top-level items of the `array` or `object` into the stack.
    - While the stack is not empty, it processes each item, moving any child arrays or objects back onto the stack.
    - After processing all children, it safely destructs the current item.
    - Finally, based on the type `t`, it calls the appropriate allocator to destroy and deallocate the memory for the `json_value`.
- **Output**: The function does not return a value; it performs destruction and deallocation of resources associated with the `json_value`.
- **See also**: [`json_value`](#json_value)  (Data Structure)



---
### data<!-- {{#data_structure:data}} -->
- **Type**: `struct`
- **Members**:
    - `m_type`: Indicates the type of the current element.
    - `m_value`: Holds the value of the current element, which can vary based on the type.
- **Description**: The `data` structure is designed to represent a value of varying types, encapsulating both the type of the value through `m_type` and the actual value itself through `m_value`. It supports initialization with different constructors, allowing for the creation of `data` instances that can represent either a single value or an array of values, while managing memory and type safety.
- **Member Functions**:
    - [`data::data`](#datadata)
    - [`data::data`](#datadata)
    - [`data::data`](#datadata)
    - [`data::data`](#datadata)
    - [`data::data`](#datadata)
    - [`data::operator=`](#dataoperator=)
    - [`data::operator=`](#dataoperator=)
    - [`data::~data`](#datadata)

**Methods**

---
#### data::data<!-- {{#callable:data::data}} -->
Constructs a `data` object by initializing its type and value with the provided argument.
- **Inputs**:
    - `v`: An instance of `value_t` that specifies the type of the current element and initializes the corresponding value.
- **Control Flow**:
    - The constructor initializes the member variable `m_type` with the provided value `v`.
    - Simultaneously, it initializes the member variable `m_value` with the same value `v`, which is expected to be of type `json_value`.
- **Output**: The constructor does not return a value but creates a `data` object with its type and value set to the provided argument.
- **See also**: [`data`](#data)  (Data Structure)


---
#### data::data<!-- {{#callable:data::data}} -->
Constructs a `data` object representing an array initialized with a specified count and value.
- **Inputs**:
    - `cnt`: The number of elements in the array to be created.
    - `val`: A `basic_json` object that serves as the initial value for each element in the array.
- **Control Flow**:
    - The constructor initializes the member variable `m_type` to indicate that the current element is of type `array`.
    - It then calls the `create` function to allocate and initialize an array of size `cnt`, with each element set to the value `val`, and assigns this array to the member variable `m_value.array`.
- **Output**: The constructor does not return a value but initializes a `data` object with an array type and a specified number of elements, each initialized to the provided value.
- **See also**: [`data`](#data)  (Data Structure)


---
#### data::data<!-- {{#callable:data::data}} -->
The `data` constructor initializes a `data` object with a specified type and value.
- **Inputs**:
    - `v`: An enumeration value of type `value_t` that specifies the type of the current element.
    - `cnt`: A size type representing the number of elements in an array.
    - `val`: A constant reference to a `basic_json` object used to initialize the array elements.
- **Control Flow**:
    - The constructor initializes the member variable `m_type` with the provided `value_t` type.
    - If the constructor is called with `cnt` and `val`, it sets `m_type` to `value_t::array` and initializes `m_value.array` using the `create` function.
    - The default constructor initializes `m_type` to `value_t::null` and `m_value` to a default-constructed `json_value`.
- **Output**: The constructor does not return a value but initializes the `data` object with the specified type and value.
- **See also**: [`data`](#data)  (Data Structure)


---
#### data::data<!-- {{#callable:data::data}} -->
The `data` function is a move constructor for the `data` class that allows for the transfer of resources from a temporary `data` object.
- **Inputs**:
    - `data&&`: An rvalue reference to a `data` object, which allows the function to take ownership of the resources from a temporary object.
- **Control Flow**:
    - The function uses the default move constructor behavior, which transfers the internal state of the source object to the new object.
    - No additional logic is implemented in this function, as it relies on the compiler-generated move constructor.
- **Output**: The output is a new `data` object that has taken ownership of the resources from the original temporary `data` object, leaving the original in a valid but unspecified state.
- **See also**: [`data`](#data)  (Data Structure)


---
#### data::data<!-- {{#callable:data::data}} -->
The `data` constructor initializes a `data` object with a specified type and value.
- **Inputs**:
    - `v`: An enumeration value of type `value_t` that specifies the type of the current element.
    - `cnt`: A size type representing the number of elements in an array.
    - `val`: A `basic_json` object that serves as the value for each element in the array.
- **Control Flow**:
    - The constructor initializes the member variable `m_type` with the provided type `v`.
    - If the constructor is called with `cnt` and `val`, it sets `m_type` to `value_t::array` and initializes `m_value.array` using the `create` function with `cnt` and `val`.
- **Output**: The constructor does not return a value but initializes a `data` object with the specified type and value.
- **See also**: [`data`](#data)  (Data Structure)


---
#### data::operator=<!-- {{#callable:data::operator=}} -->
The `operator=` function is deleted for both move and copy assignment in the `data` structure, preventing assignment operations.
- **Inputs**:
    - `data&&`: A rvalue reference to a `data` object for move assignment, which is deleted.
    - `const data&`: A constant reference to a `data` object for copy assignment, which is also deleted.
- **Control Flow**:
    - The function is marked as `delete`, which means it cannot be called or used in any context.
    - No actual assignment logic is implemented since the function is deleted, effectively preventing any assignment operations.
- **Output**: The function does not produce any output as it is deleted and cannot be invoked.
- **See also**: [`data`](#data)  (Data Structure)


---
#### data::\~data<!-- {{#callable:data::~data}} -->
The `~data` destructor is responsible for cleaning up resources associated with a `data` object.
- **Inputs**: None
- **Control Flow**:
    - The destructor is invoked when a `data` object goes out of scope or is explicitly deleted.
    - It calls the `destroy` method on `m_value`, passing the current `m_type` to properly release resources.
- **Output**: The function does not return a value; it performs cleanup operations to free resources associated with the `data` object.
- **See also**: [`data`](#data)  (Data Structure)



---
### patch\_operations<!-- {{#data_structure:patch_inplace::patch_operations}} -->
- **Type**: `enum class`
- **Members**:
    - `add`: Represents an operation to add an element.
    - `remove`: Represents an operation to remove an element.
    - `replace`: Represents an operation to replace an existing element.
    - `move`: Represents an operation to move an element from one location to another.
    - `copy`: Represents an operation to copy an element.
    - `test`: Represents an operation to test a condition without modifying the element.
    - `invalid`: Represents an invalid operation.
- **Description**: The `patch_operations` enum class defines a set of operations that can be performed on elements in a data structure, including adding, removing, replacing, moving, copying, and testing elements, as well as an invalid operation to handle erroneous cases.


# Functions

---
### from\_json<!-- {{#callable:(anonymous)::(anonymous)::adl_serializer::from_json}} -->
The `from_json` function deserializes a JSON object into a specified C++ type.
- **Inputs**:
    - `j`: A JSON object of type `BasicJsonType` that is to be deserialized.
- **Control Flow**:
    - The function uses perfect forwarding to pass the JSON object `j` to the `nlohmann::from_json` function.
    - It utilizes a `detail::identity_tag` to deduce the target type for deserialization.
    - The function is marked as [`noexcept`](#noexcept), indicating it does not throw exceptions.
- **Output**: The output is the result of the deserialization process, which is of type `TargetType`.
- **Functions called**:
    - [`noexcept`](#noexcept)


---
### operator<<!-- {{#callable:operator<}} -->
Compares two `ScalarType` values to determine if the left-hand side is less than the right-hand side.
- **Inputs**:
    - `lhs`: The left-hand side operand of type `ScalarType` to be compared.
    - `rhs`: The right-hand side operand of type `const_reference`, which is the value to compare against.
- **Control Flow**:
    - The function is defined as a friend function, allowing it to access private members of the class if necessary.
    - It constructs a [`basic_json`](#basic_json) object from the `lhs` operand.
    - It then uses the less-than operator (`<`) to compare the [`basic_json`](#basic_json) representation of `lhs` with `rhs`.
- **Output**: Returns a boolean value indicating whether `lhs` is less than `rhs`.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### replace\_substring<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::replace_substring}} -->
Replaces all occurrences of a substring in a given string with another substring.
- **Inputs**:
    - `s`: A reference to the string in which the replacements will be made.
    - `f`: The substring that will be searched for in the string `s`.
    - `t`: The substring that will replace occurrences of `f` in the string `s`.
- **Control Flow**:
    - The function first asserts that the substring `f` is not empty using `JSON_ASSERT`.
    - It then enters a loop where it searches for the first occurrence of `f` in `s` using `s.find(f)`.
    - If `f` is found (i.e., `pos` is not equal to `StringType::npos`), it replaces the found substring with `t` using `s.replace(pos, f.size(), t)`.
    - After replacing, it updates `pos` to find the next occurrence of `f` starting from the position after the newly inserted substring `t`.
    - The loop continues until no more occurrences of `f` are found in `s`.
- **Output**: The function does not return a value; it modifies the input string `s` in place.


---
### escape<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::escape}} -->
The `escape` function replaces specific characters in a string with escape sequences.
- **Inputs**:
    - `s`: A string of type `StringType` that contains the characters to be escaped.
- **Control Flow**:
    - The function first calls [`replace_substring`](#NLOHMANN_JSON_NAMESPACE_BEGINreplace_substring) to replace all occurrences of the character '~' with the escape sequence '~0'.
    - Next, it calls [`replace_substring`](#NLOHMANN_JSON_NAMESPACE_BEGINreplace_substring) again to replace all occurrences of the character '/' with the escape sequence '~1'.
    - Finally, the modified string `s` is returned.
- **Output**: The function returns the modified string with the specified characters replaced by their escape sequences.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::replace_substring`](#NLOHMANN_JSON_NAMESPACE_BEGINreplace_substring)


---
### unescape<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::unescape}} -->
The `unescape` function replaces specific escape sequences in a string with their corresponding characters.
- **Inputs**:
    - `s`: A reference to a string of type `StringType` that will be modified in place to replace escape sequences.
- **Control Flow**:
    - The function calls [`replace_substring`](#NLOHMANN_JSON_NAMESPACE_BEGINreplace_substring) to replace all occurrences of the escape sequence `~1` with the character `/` in the string `s`.
    - It then calls [`replace_substring`](#NLOHMANN_JSON_NAMESPACE_BEGINreplace_substring) again to replace all occurrences of the escape sequence `~0` with the character `~` in the string `s`.
- **Output**: The function does not return a value; it modifies the input string `s` directly by replacing the specified escape sequences.
- **Functions called**:
    - [`NLOHMANN_JSON_NAMESPACE_BEGIN::replace_substring`](#NLOHMANN_JSON_NAMESPACE_BEGINreplace_substring)


---
### make\_array<!-- {{#callable:namespace::make_array}} -->
`make_array` creates a `std::array` of a specified type `T` initialized with a variable number of arguments.
- **Inputs**:
    - `T`: The type of the elements that will be stored in the `std::array`.
    - `Args`: A variadic template parameter pack that represents the arguments used to initialize the array.
- **Control Flow**:
    - The function template accepts a type `T` and a variable number of arguments of any type.
    - It constructs a `std::array` of type `T` with a size equal to the number of arguments passed.
    - Each argument is forwarded and cast to type `T` using `static_cast`.
- **Output**: The function returns a `std::array<T, sizeof...(Args)>` containing the elements initialized with the provided arguments.


---
### to\_int\_type<!-- {{#callable:namespace::to_int_type}} -->
Converts a character of type `char_type` to an integer type `int_type`.
- **Inputs**:
    - `c`: A character of type `char_type` that is to be converted to an integer type.
- **Control Flow**:
    - The function directly converts the input character `c` to an integer type using `static_cast`.
    - There are no conditional statements or loops; the conversion is straightforward and executed in a single step.
- **Output**: Returns the converted value of type `int_type` that corresponds to the input character.


---
### to\_char\_type<!-- {{#callable:namespace::to_char_type}} -->
Converts an integer type to a character type using static casting.
- **Inputs**:
    - `i`: An integer of type `int_type` that is to be converted to `char_type`.
- **Control Flow**:
    - The function takes a single integer input `i`.
    - It uses `static_cast` to convert the integer `i` to the corresponding `char_type`.
    - The function returns the converted value.
- **Output**: The function outputs a value of type `char_type` that represents the character equivalent of the input integer.


---
### eof<!-- {{#callable:namespace::eof}} -->
Returns the end-of-file indicator for character types.
- **Inputs**: None
- **Control Flow**:
    - The function is marked as `noexcept`, indicating it does not throw exceptions.
    - It uses `std::char_traits<char>::eof()` to retrieve the end-of-file indicator for character types.
    - The result is cast to `int_type` before being returned.
- **Output**: Returns an `int_type` value that represents the end-of-file indicator.


---
### conditional\_static\_cast<!-- {{#callable:namespace::conditional_static_cast}} -->
Performs a `static_cast` from type `U` to type `T` only if `T` and `U` are the same type.
- **Inputs**:
    - `value`: An input value of type `U` that is to be conditionally cast to type `T`.
- **Control Flow**:
    - The function uses SFINAE (Substitution Failure Is Not An Error) to enable the function only if `T` and `U` are the same type.
    - If `T` and `U` are the same, the function returns the input `value` as type `T`.
- **Output**: Returns the input `value` cast to type `T` if `T` and `U` are the same type; otherwise, the function is not instantiated.


---
### test<!-- {{#callable:namespace::test}} -->
The `test` function is a constexpr function that always returns true regardless of the input.
- **Inputs**:
    - `val`: A parameter of type T that is not used in the function body.
- **Control Flow**:
    - The function does not contain any conditional statements or loops.
    - It directly returns the boolean value true.
- **Output**: The output is a boolean value, specifically true, indicating a constant result regardless of the input.


---
### value\_in\_range\_of<!-- {{#callable:namespace::value_in_range_of}} -->
Checks if a value is within a specified range defined by the type `OfType`.
- **Inputs**:
    - `val`: The value of type `T` that needs to be checked against the range.
- **Control Flow**:
    - Calls the `value_in_range_of_impl1` template specialization to perform the actual range check.
    - Returns the result of the `test` method from the `value_in_range_of_impl1` class.
- **Output**: Returns a boolean indicating whether the value `val` is within the range defined by `OfType`.


---
### is\_c\_string<!-- {{#callable:namespace::impl::is_c_string}} -->
Determines if a type `T` is a C-style string.
- **Inputs**:
    - `T`: A type parameter that is evaluated to check if it represents a C-style string.
- **Control Flow**:
    - The function first removes any array extent from `T` to get `TUnExt`.
    - It then removes any const or volatile qualifiers from `TUnExt` to get `TUnCVExt`.
    - Next, it removes any pointer from `T` to get `TUnPtr`.
    - It again removes const or volatile qualifiers from `TUnPtr` to get `TUnCVPtr`.
    - Finally, it checks if `T` is an array of characters or a pointer to characters, returning true if either condition is met.
- **Output**: Returns a boolean value indicating whether `T` is a C-style string (either an array of `char` or a pointer to `char`).


---
### is\_transparent<!-- {{#callable:namespace::impl::is_transparent}} -->
The `is_transparent` function template checks if a type `T` is considered transparent by evaluating a detection mechanism.
- **Inputs**:
    - `T`: A type parameter that is being checked for transparency.
- **Control Flow**:
    - The function uses a template parameter `T` to perform a compile-time check.
    - It calls `is_detected` with `detect_is_transparent` and `T` to determine if `T` meets the criteria for being transparent.
    - The result of the detection is accessed via the `value` member of the `is_detected` type.
- **Output**: The function returns a boolean value indicating whether the type `T` is transparent based on the detection mechanism.


---
### concat\_length<!-- {{#callable:namespace::concat_length}} -->
Calculates the total length of a string and a variadic number of additional strings.
- **Inputs**:
    - `str`: The first string whose length will be included in the total.
    - `rest`: A variadic list of additional strings whose lengths will also be included in the total.
- **Control Flow**:
    - The function starts by calculating the size of the first string `str`.
    - It then recursively calls itself with the remaining strings in `rest` until no more strings are left, at which point it returns 0.
- **Output**: Returns the total length of all input strings combined.
- **Functions called**:
    - [`namespace::concat_length`](#namespaceconcat_length)


---
### concat\_into<!-- {{#callable:namespace::concat_into}} -->
The [`concat_into`](#namespaceconcat_into) function appends data from a series of arguments to an output string.
- **Inputs**:
    - `out`: A reference to an output string of type `OutStringType` where the concatenated result will be stored.
    - `arg`: The first argument of type `Arg` whose data will be appended to the output string.
    - `rest`: A variadic list of additional arguments of any type that will also be appended to the output string.
- **Control Flow**:
    - The function first appends the data from the `arg` argument to the `out` string using the `append` method.
    - It then recursively calls itself with the output string and the remaining arguments in `rest`, effectively concatenating all provided arguments.
- **Output**: The function does not return a value; instead, it modifies the `out` string in place by appending the data from the provided arguments.
- **Functions called**:
    - [`namespace::concat_into`](#namespaceconcat_into)


---
### concat<!-- {{#callable:namespace::concat}} -->
The `concat` function constructs a string by concatenating multiple input arguments.
- **Inputs**:
    - `OutStringType`: A template parameter that specifies the type of the output string, defaulting to `std::string`.
    - `Args`: A variadic template parameter pack that accepts multiple arguments of any type to be concatenated.
- **Control Flow**:
    - An output string of type `OutStringType` is initialized.
    - The `reserve` method is called on the output string to allocate enough memory based on the total length of the concatenated arguments, calculated by `concat_length(args...)`.
    - The [`concat_into`](#namespaceconcat_into) function is invoked to perform the actual concatenation of the input arguments into the output string.
    - Finally, the constructed string is returned.
- **Output**: The function returns a string of type `OutStringType` that contains the concatenated result of all input arguments.
- **Functions called**:
    - [`namespace::concat_length`](#namespaceconcat_length)
    - [`namespace::concat_into`](#namespaceconcat_into)


---
### what<!-- {{#callable:namespace::what}} -->
Returns a C-style string representation of the exception.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the result of calling `what()` on the member variable `m`.
- **Output**: A pointer to a constant character string that describes the exception.


---
### exception<!-- {{#callable:namespace::exception}} -->
Constructs an `exception` object with a specified error ID and message.
- **Inputs**:
    - `id_`: An integer representing the error ID associated with the exception.
    - `what_arg`: A C-style string (const char*) that contains the error message.
- **Control Flow**:
    - Initializes the member variable `id` with the value of `id_`.
    - Initializes the member variable `m` with the value of `what_arg`.
- **Output**: This constructor does not return a value but initializes an instance of the `exception` class.


---
### name<!-- {{#callable:namespace::name}} -->
Constructs a formatted string for JSON exceptions using the provided error name and ID.
- **Inputs**:
    - `ename`: A constant reference to a `std::string` representing the name of the error.
    - `id_`: An integer representing the error ID.
- **Control Flow**:
    - The function takes two parameters: `ename` and `id_`.
    - It calls the [`concat`](#namespaceconcat) function to create a formatted string.
    - The formatted string includes the prefix '[json.exception.', the error name, the error ID converted to a string, and a closing bracket followed by a space.
- **Output**: Returns a `std::string` that combines the error name and ID into a specific format for JSON exception handling.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)


---
### diagnostics<!-- {{#callable:namespace::diagnostics}} -->
Generates a diagnostic string representing the path to a given JSON leaf element.
- **Inputs**:
    - `leaf_element`: A pointer to a `BasicJsonType` object representing the JSON leaf element whose diagnostics are to be generated.
- **Control Flow**:
    - Checks if JSON diagnostics are enabled using the `#if JSON_DIAGNOSTICS` preprocessor directive.
    - Initializes an empty vector `tokens` to store the path components.
    - Iterates through the parent elements of `leaf_element` until it reaches a null parent.
    - For each parent, determines its type (array or object) and finds the index or key of the current element, adding it to `tokens`.
    - If `tokens` is empty after the loop, returns an empty string.
    - Reverses the `tokens` vector and concatenates its elements into a single string, formatted with slashes and escaped characters.
    - Returns the concatenated string along with the byte positions of the `leaf_element`.
- **Output**: A string representing the path to the `leaf_element` in the JSON structure, formatted with slashes and including byte positions, or an empty string if no path is found.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)
    - [`namespace::get_byte_positions`](#namespaceget_byte_positions)


---
### get\_byte\_positions<!-- {{#callable:namespace::get_byte_positions}} -->
The `get_byte_positions` function is a template function that takes a pointer to a JSON element and returns an empty string.
- **Inputs**:
    - `leaf_element`: A pointer to a `BasicJsonType` object representing a JSON element.
- **Control Flow**:
    - The function begins by casting the `leaf_element` pointer to void, effectively ignoring it.
    - The function then proceeds to return an empty string.
- **Output**: The function outputs an empty string, indicating that no byte positions are computed or returned.


---
### get\_arithmetic\_value<!-- {{#callable:namespace::get_arithmetic_value}} -->
The `get_arithmetic_value` function extracts a numeric value from a JSON object and assigns it to a specified arithmetic type.
- **Inputs**:
    - `j`: A constant reference to a JSON object of type `BasicJsonType` from which the numeric value is to be extracted.
    - `val`: A reference to a variable of type `ArithmeticType` where the extracted numeric value will be stored.
- **Control Flow**:
    - The function begins by checking the type of the JSON object `j` using a switch statement.
    - If `j` is of type `number_unsigned`, `number_integer`, or `number_float`, the corresponding numeric value is retrieved and cast to `ArithmeticType` before being assigned to `val`.
    - If `j` is of any other type (null, object, array, string, boolean, binary, or discarded), a `type_error` exception is thrown indicating that the type must be a number.
- **Output**: The function does not return a value; instead, it modifies the `val` parameter to hold the extracted numeric value from the JSON object.
- **Functions called**:
    - [`namespace::concat`](#namespaceconcat)


---
### from\_json\_array\_impl<!-- {{#callable:namespace::from_json_array_impl}} -->
The `from_json_array_impl` function populates a given array from a JSON array by transforming each element.
- **Inputs**:
    - `j`: A constant reference to a JSON object representing an array.
    - `arr`: A reference to an array that will be populated with values extracted from the JSON array.
- **Control Flow**:
    - The function begins by declaring a local variable `ret` of type `ConstructibleArrayType` to hold the transformed values.
    - It uses `std::transform` to iterate over each element in the JSON array `j`, applying a lambda function to extract the value of each element.
    - The lambda function calls `get` on each JSON element to convert it to the appropriate type for the array.
    - Finally, the contents of `ret` are moved into `arr`, effectively populating the provided array with the transformed values.
- **Output**: The function does not return a value; instead, it modifies the input array `arr` to contain the values extracted from the JSON array.


---
### from\_json\_inplace\_array\_impl<!-- {{#callable:namespace::from_json_inplace_array_impl}} -->
Converts a JSON array into a `std::array` of a specified type by extracting elements from the JSON.
- **Inputs**:
    - `j`: A JSON object that contains an array from which elements will be extracted.
    - `identity_tag<std::array<T, sizeof...(Idx)>>`: A tag used to specify the type of the output array, which is unused in the function body.
    - `index_sequence<Idx...>`: A compile-time sequence of indices used to access elements in the JSON array.
- **Control Flow**:
    - The function uses a parameter pack expansion to iterate over the indices provided by `Idx...`.
    - For each index, it accesses the corresponding element in the JSON array using `at(Idx)` and retrieves its value using `get<T>()`.
    - The results are collected into a `std::array` and returned.
- **Output**: Returns a `std::array<T, sizeof...(Idx)>` containing the elements extracted from the JSON array.


---
### from\_json\_tuple\_impl\_base<!-- {{#callable:namespace::from_json_tuple_impl_base}} -->
This function template returns an empty `std::tuple`.
- **Inputs**:
    - `BasicJsonType& /*unused*/`: A reference to a JSON type, which is unused in this implementation.
    - `index_sequence<> /*unused*/`: An empty index sequence, which is also unused in this implementation.
- **Control Flow**:
    - This function does not contain any control flow statements as it directly returns an empty tuple.
    - The function parameters are not utilized in the body of the function.
- **Output**: The function outputs an empty `std::tuple`, indicating no data is processed or returned.


---
### from\_json\_tuple\_impl<!-- {{#callable:namespace::from_json_tuple_impl}} -->
The `from_json_tuple_impl` function deserializes a JSON object into a tuple.
- **Inputs**:
    - `j`: A JSON object of type `BasicJsonType` that contains the data to be deserialized.
    - `t`: A reference to a tuple of type `std::tuple<Args...>` where the deserialized values will be stored.
    - `priority_tag<3>`: A tag used for function overloading, indicating the priority of this implementation.
- **Control Flow**:
    - The function calls `from_json_tuple_impl_base` with the JSON object and an index sequence generated for the tuple's types.
    - The result of the base function call is assigned to the tuple reference `t`, effectively populating it with the deserialized values.
- **Output**: The function does not return a value; it modifies the tuple `t` in place with the deserialized data from the JSON object.


---
### int\_to\_string<!-- {{#callable:(anonymous)::namespace::int_to_string}} -->
Converts a given integer value to a string representation and assigns it to the target variable.
- **Inputs**:
    - `target`: A reference to a variable of type `StringType` where the string representation of the integer will be stored.
    - `value`: An unsigned integer (`std::size_t`) that represents the integer value to be converted to a string.
- **Control Flow**:
    - Uses the `std::to_string` function to convert the integer value to a string.
    - Assigns the resulting string to the `target` variable.
- **Output**: The function does not return a value; instead, it modifies the `target` variable to hold the string representation of the input integer.
- **Functions called**:
    - [`(anonymous)::namespace::to_string`](#(anonymous)::namespace::to_string)


---
### to\_string<!-- {{#callable:std::string::to_string}} -->
Converts a `NLOHMANN_BASIC_JSON_TPL` object to its string representation.
- **Inputs**:
    - `j`: A constant reference to a `NLOHMANN_BASIC_JSON_TPL` object that is to be converted to a string.
- **Control Flow**:
    - The function takes a `NLOHMANN_BASIC_JSON_TPL` object as input.
    - It calls the `dump()` method on the input object to generate a JSON string representation.
    - The resulting string is returned as the output of the function.
- **Output**: A `std::string` containing the JSON representation of the input `NLOHMANN_BASIC_JSON_TPL` object.


---
### get<!-- {{#callable:get}} -->
The `get` function is a template function that retrieves a pointer of a specified type from a `basic_json_t` object.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the provided `PointerType` is a pointer type using `std::is_pointer`.
    - If the check passes, it delegates the call to another function `get_ptr<PointerType>()` to retrieve the pointer.
- **Output**: The output is the result of the `get_ptr<PointerType>()` function, which is expected to be a pointer of the specified type.


---
### construct<!-- {{#callable:(anonymous)::namespace::construct}} -->
The `construct` function initializes a `BasicJsonType` object with data from a compatible object type.
- **Inputs**:
    - `j`: A reference to a `BasicJsonType` object that will be constructed or modified.
    - `obj`: A constant reference to a compatible object type that provides data to initialize the `BasicJsonType` object.
- **Control Flow**:
    - The function first destroys any existing data in the `BasicJsonType` object `j` by calling `destroy` on its current value.
    - It then sets the type of `j` to `value_t::object`, indicating that it will now hold an object.
    - Next, it creates a new object of type `typename BasicJsonType::object_t` using the range defined by the beginning and end of `obj`, and assigns it to `j.m_data.m_value.object`.
    - After the object is created, the function calls `set_parents` to establish parent-child relationships in the JSON structure.
    - Finally, it calls `assert_invariant` to ensure that the internal state of `j` is valid.
- **Output**: The function does not return a value; it modifies the `BasicJsonType` object `j` in place to reflect the new data from `obj`.


---
### to\_json<!-- {{#callable:(anonymous)::(anonymous)::adl_serializer::to_json}} -->
Converts a value of type `TargetType` to a JSON representation and stores it in the provided `BasicJsonType` object.
- **Inputs**:
    - `j`: A reference to a `BasicJsonType` object where the JSON representation will be stored.
    - `val`: A value of type `TargetType` that will be converted to JSON.
- **Control Flow**:
    - The function uses the [`noexcept`](#noexcept) specifier to ensure that it does not throw exceptions if the conversion to JSON is not possible.
    - It calls the `::nlohmann::to_json` function, passing the `BasicJsonType` reference and the value to be converted, using `std::forward` to maintain the value's type.
- **Output**: The function does not return a value; it modifies the `BasicJsonType` object `j` in place to contain the JSON representation of `val`.
- **Functions called**:
    - [`noexcept`](#noexcept)


---
### to\_json\_tuple\_impl<!-- {{#callable:(anonymous)::namespace::to_json_tuple_impl}} -->
Initializes a JSON object as an empty array based on the provided JSON type.
- **Inputs**:
    - `j`: A reference to a JSON object of type `BasicJsonType` that will be initialized.
    - `Tuple`: A tuple type that is not used in the function body, serving as a placeholder.
    - `index_sequence<>`: An unused parameter that allows for potential specialization or overloads based on index sequences.
- **Control Flow**:
    - The function uses a type alias `array_t` to refer to the array type defined within `BasicJsonType`.
    - It assigns an empty array to the JSON object `j`.
- **Output**: The function does not return a value; instead, it modifies the input JSON object `j` to be an empty array.


---
### operator\!=<!-- {{#callable:operator!=}} -->
Compares two `ScalarType` values for inequality.
- **Inputs**:
    - `lhs`: The left-hand side operand of type `ScalarType` to be compared.
    - `rhs`: The right-hand side operand of type `const_reference` to be compared.
- **Control Flow**:
    - The function calls the [`basic_json`](#basic_json) constructor with `lhs` to create a [`basic_json`](#basic_json) object.
    - It then compares the created [`basic_json`](#basic_json) object with `rhs` using the `!=` operator.
- **Output**: Returns a boolean value indicating whether the two operands are not equal.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### set\_subtype<!-- {{#callable:(anonymous)::set_subtype}} -->
Sets the subtype of an object and marks it as having a subtype.
- **Inputs**:
    - `subtype_`: The new subtype value to be assigned to the object's subtype.
- **Control Flow**:
    - The function directly assigns the provided `subtype_` to the member variable `m_subtype`.
    - It then sets the member variable `m_has_subtype` to true, indicating that the object now has a valid subtype.
- **Output**: This function does not return a value; it modifies the internal state of the object.


---
### subtype<!-- {{#callable:(anonymous)::subtype}} -->
Returns the `m_subtype` if it exists, otherwise returns a default value.
- **Inputs**:
    - `none`: This function does not take any input arguments.
- **Control Flow**:
    - Checks the boolean member variable `m_has_subtype` to determine if a subtype exists.
    - If `m_has_subtype` is true, it returns the value of `m_subtype`.
    - If `m_has_subtype` is false, it returns a default value of `-1` cast to `subtype_type`.
- **Output**: Returns a value of type `subtype_type`, which is either the subtype or a default value indicating absence.


---
### has\_subtype<!-- {{#callable:(anonymous)::has_subtype}} -->
Checks if the object has a subtype.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the member variable `m_has_subtype`.
- **Output**: Returns a boolean value indicating whether the object has a subtype.


---
### clear\_subtype<!-- {{#callable:(anonymous)::clear_subtype}} -->
Resets the `m_subtype` member variable to zero and sets `m_has_subtype` to false.
- **Inputs**: None
- **Control Flow**:
    - The function directly assigns the value 0 to the member variable `m_subtype`.
    - It then sets the member variable `m_has_subtype` to false.
- **Output**: This function does not return a value; it modifies the state of the object by resetting specific member variables.


---
### combine<!-- {{#callable:(anonymous)::namespace::combine}} -->
Combines two hash values using a specific algorithm to produce a new hash value.
- **Inputs**:
    - `seed`: The initial hash value that will be modified.
    - `h`: The hash value to be combined with the seed.
- **Control Flow**:
    - The function performs a bitwise XOR operation between the `seed` and a computed value derived from `h` and the current `seed`.
    - The computed value includes a constant (0x9e3779b9), the left shift of `seed` by 6 bits, and the right shift of `seed` by 2 bits.
    - Finally, the modified `seed` is returned as the new hash value.
- **Output**: Returns a new hash value that is a combination of the input `seed` and `h`.


---
### hash<!-- {{#callable:(anonymous)::namespace::hash}} -->
Computes a hash value for a given JSON object based on its type and contents.
- **Inputs**:
    - `j`: A constant reference to a JSON object of type `BasicJsonType` whose hash value is to be computed.
- **Control Flow**:
    - The function begins by determining the type of the JSON object `j` and storing it in the variable `type`.
    - A switch statement is used to handle different JSON value types, including null, object, array, string, boolean, integer, unsigned, float, and binary.
    - For null and discarded types, it combines the type with a seed of 0 and returns the result.
    - For objects, it initializes a seed with the type and size of the object, iterates over its items, hashes each key and value, and combines these hashes into the seed.
    - For arrays, it initializes a seed with the type and size, iterates over each element, hashes them, and combines the results.
    - For strings, booleans, integers, unsigned integers, and floats, it hashes the respective value and combines it with the type.
    - For binary types, it combines the type with the size of the binary data, hashes its subtype, and iterates over each byte to combine their hashes.
    - If an unsupported type is encountered, an assertion failure is triggered.
- **Output**: Returns a `std::size_t` hash value that uniquely represents the contents and type of the JSON object.
- **Functions called**:
    - [`(anonymous)::namespace::combine`](#(anonymous)::namespace::combine)


---
### file\_input\_adapter<!-- {{#callable:(anonymous)::namespace::file_input_adapter}} -->
The `file_input_adapter` class has a move constructor defined and deletes the copy assignment operator.
- **Inputs**: None
- **Control Flow**:
    - The move constructor allows for the transfer of resources from one `file_input_adapter` instance to another without copying.
    - The copy assignment operator is explicitly deleted to prevent copying of `file_input_adapter` instances.
- **Output**: The function does not produce a traditional output; instead, it modifies the state of the object being constructed or assigned.


---
### get\_character<!-- {{#callable:(anonymous)::namespace::get_character}} -->
Reads a character from a file and returns it as an integer type.
- **Inputs**: None
- **Control Flow**:
    - Calls `std::fgetc` with `m_file` as the argument to read the next character from the file.
    - Returns the result of `std::fgetc`, which is the character read as an `int_type`.
- **Output**: Returns the character read from the file as an `int_type`, which can represent either a valid character or EOF (end-of-file) if the end of the file is reached.


---
### get\_elements<!-- {{#callable:(anonymous)::namespace::get_elements}} -->
Reads a specified number of elements of type `T` from a file into a destination buffer.
- **Inputs**:
    - `dest`: A pointer to the destination buffer where the read elements will be stored.
    - `count`: The number of elements of type `T` to read from the file, defaulting to 1.
- **Control Flow**:
    - Calls `fread` to read data from the file associated with `m_file` into the buffer pointed to by `dest`.
    - Calculates the total number of bytes to read by multiplying the size of type `T` by `count`.
- **Output**: Returns the number of elements successfully read from the file.


---
### fill\_buffer<!-- {{#callable:(anonymous)::namespace::fill_buffer}} -->
The `fill_buffer` function encodes a character from a `BaseInputAdapter` into UTF-8 format and fills a provided byte array with the encoded bytes.
- **Inputs**:
    - `input`: A reference to a `BaseInputAdapter` object that provides the character to be encoded.
    - `utf8_bytes`: An array of `std::char_traits<char>::int_type` with a size of 4, which will hold the UTF-8 encoded bytes.
    - `utf8_bytes_index`: A reference to a size_t variable that tracks the current index in the `utf8_bytes` array.
    - `utf8_bytes_filled`: A reference to a size_t variable that indicates how many bytes have been filled in the `utf8_bytes` array.
- **Control Flow**:
    - The function initializes `utf8_bytes_index` to 0.
    - It checks if the `input` is empty; if so, it sets the first byte of `utf8_bytes` to EOF and updates `utf8_bytes_filled` to 1.
    - If `input` is not empty, it retrieves the current character using `input.get_character()`.
    - Depending on the value of the character, it encodes it into UTF-8 format, filling the `utf8_bytes` array and updating `utf8_bytes_filled` accordingly.
    - For characters less than 0x80, it fills one byte; for characters between 0x80 and 0x7FF, it fills two bytes; for characters between 0x800 and 0xD7FF, it fills three bytes; and for characters above 0xFFFF, it fills four bytes.
- **Output**: The function does not return a value but modifies the `utf8_bytes` array and updates `utf8_bytes_filled` to indicate how many bytes were written.


---
### create<!-- {{#callable:create}} -->
Creates and initializes an object of type T using a provided allocator and arguments.
- **Inputs**:
    - `args`: Variadic template arguments used to initialize the object of type T.
- **Control Flow**:
    - An allocator of type T is instantiated to manage memory allocation.
    - A custom deleter is defined to deallocate the object when it is no longer needed.
    - Memory for one object of type T is allocated using the allocator.
    - The object is constructed in the allocated memory using the provided arguments.
    - An assertion checks that the object was successfully created and is not null.
    - The ownership of the object is released from the unique pointer and returned.
- **Output**: Returns a pointer to the newly created and initialized object of type T.


---
### input\_adapter<!-- {{#callable:(anonymous)::namespace::input_adapter}} -->
The [`input_adapter`](#(anonymous)::namespace::input_adapter) function template converts a C-style array into an appropriate range for further processing.
- **Inputs**:
    - `array`: A reference to a C-style array of type T with size N.
- **Control Flow**:
    - The function takes a reference to a C-style array and deduces its size.
    - It calls itself recursively with the start and end pointers of the array, effectively creating a range.
- **Output**: The output is the result of the second call to [`input_adapter`](#(anonymous)::namespace::input_adapter), which is expected to be a range type that represents the elements of the array.
- **Functions called**:
    - [`(anonymous)::namespace::input_adapter`](#(anonymous)::namespace::input_adapter)


---
### reset<!-- {{#callable:reset}} -->
Resets the state of the token buffer and related variables in preparation for new input.
- **Inputs**: None
- **Control Flow**:
    - Clears the `token_buffer` to remove any existing tokens.
    - Clears the `token_string` to reset the string holding the current token.
    - Sets `decimal_point_position` to `std::string::npos`, indicating no decimal point is currently set.
    - Adds the current character (`current`) to the `token_string` after converting it using `char_traits<char_type>::to_char_type`.
- **Output**: This function does not return a value; it modifies the internal state of the object.


---
### unget<!-- {{#callable:unget}} -->
The `unget` function reverts the last character read from a stream, adjusting the read position and the associated counters.
- **Inputs**: None
- **Control Flow**:
    - Sets the `next_unget` flag to true to indicate that a character has been ungotten.
    - Decrements the total number of characters read (`chars_read_total`).
    - Checks if the current character being ungotten is a newline; if so, decrements the number of lines read (`lines_read`).
    - If not a newline, decrements the count of characters read in the current line (`chars_read_current_line`).
    - If the current character is not EOF, asserts that the `token_string` is not empty and removes the last character from `token_string`.
- **Output**: The function does not return a value but modifies the internal state of the reading position and the `token_string`.


---
### add<!-- {{#callable:add}} -->
The `add` function appends a character or integer type to a token buffer.
- **Inputs**:
    - `c`: A character or integer type that will be converted and added to the token buffer.
- **Control Flow**:
    - The function takes a single argument `c` of type `char_int_type`.
    - The argument `c` is cast to the appropriate value type of `string_t`.
    - The cast value is then pushed back into the `token_buffer`.
- **Output**: The function does not return a value; it modifies the `token_buffer` by adding the new element.


---
### get\_number\_unsigned<!-- {{#callable:get_number_unsigned}} -->
Returns the value of the member variable `value_unsigned`.
- **Inputs**: None
- **Control Flow**:
    - The function is marked as `constexpr`, allowing it to be evaluated at compile time.
    - It is a `const` member function, indicating it does not modify the state of the object.
    - The function directly returns the value of the member variable `value_unsigned`.
- **Output**: The output is of type `number_unsigned_t`, representing the unsigned number stored in the member variable `value_unsigned`.


---
### get\_number\_float<!-- {{#callable:get_number_float}} -->
Returns the value of the member variable `value_float`.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the member variable `value_float` without any conditions or loops.
- **Output**: The function outputs the current value of `value_float`, which is of type `number_float_t`.


---
### get\_string<!-- {{#callable:get_string}} -->
The `get_string` function modifies a specific character in a string buffer to represent a decimal point and returns the modified string buffer.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the `decimal_point_char` is not a '.' and if the `decimal_point_position` is valid (not `std::string::npos`).
    - If both conditions are met, it replaces the character at `decimal_point_position` in `token_buffer` with a '.' character.
- **Output**: The function returns a reference to the modified `token_buffer`, which is of type `string_t`.


---
### get\_position<!-- {{#callable:get_position}} -->
Returns the current position stored in the `position` member.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the `position` member variable.
- **Output**: The function outputs the current value of the `position` member, which is of type `position_t`.


---
### get\_token\_string<!-- {{#callable:get_token_string}} -->
The `get_token_string` function returns a string representation of `token_string`, escaping control characters.
- **Inputs**:
    - `none`: The function does not take any input arguments.
- **Control Flow**:
    - The function initializes an empty string `result` to store the final output.
    - It iterates over each character `c` in the member variable `token_string`.
    - For each character, it checks if the character is a control character (ASCII value less than or equal to 31).
    - If it is a control character, it formats the character as a Unicode escape sequence and appends it to `result`.
    - If it is not a control character, it appends the character directly to `result`.
    - Finally, the function returns the constructed `result` string.
- **Output**: The output is a `std::string` that contains the original `token_string` with control characters replaced by their Unicode escape sequences.


---
### get\_error\_message<!-- {{#callable:get_error_message}} -->
Retrieves the error message stored in the class.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the member variable `error_message`.
- **Output**: Returns a pointer to a constant character string representing the error message.


---
### skip\_bom<!-- {{#callable:skip_bom}} -->
The `skip_bom` function checks for the presence of a Byte Order Mark (BOM) at the beginning of a stream.
- **Inputs**:
    - `none`: The function does not take any input arguments.
- **Control Flow**:
    - The function first checks if the current character is the first byte of the BOM (0xEF).
    - If it is, it then checks the next two bytes to confirm the complete BOM (0xBB and 0xBF).
    - If the BOM is fully parsed, it returns true; otherwise, it returns false.
    - If the first character is not part of the BOM, it calls `unget()` to put the character back for later processing and returns true.
- **Output**: The function returns a boolean value: true if the BOM is skipped or not present, and false if the BOM is detected but not fully parsed.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)
    - [`unget`](#unget)


---
### skip\_whitespace<!-- {{#callable:skip_whitespace}} -->
The `skip_whitespace` function reads characters from the input until it encounters a non-whitespace character.
- **Inputs**: None
- **Control Flow**:
    - The function enters a `do-while` loop that continuously calls the `get()` function to read the next character.
    - The loop continues as long as the `current` character is a whitespace character, which includes spaces, tabs, newlines, and carriage returns.
- **Output**: The function does not return a value; it modifies the state by advancing the input position past any leading whitespace characters.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::get)


---
### scan<!-- {{#callable:scan}} -->
The `scan` function processes input characters to identify and return their corresponding token types while handling whitespace, comments, and literals.
- **Inputs**: None
- **Control Flow**:
    - The function begins by checking if the byte order mark (BOM) is valid and skips it if necessary.
    - Whitespace characters are ignored before processing the next character.
    - If comments are enabled, the function enters a loop to skip over comments, calling [`scan_comment`](#(anonymous)::namespace::lexer::scan_comment) to handle them.
    - A switch statement is used to determine the type of token based on the current character, handling structural characters, literals, strings, numbers, and end of input.
    - If an invalid character is encountered, an error message is set and a parse error token is returned.
- **Output**: The function returns a `token_type` enumeration value representing the type of token identified, such as `begin_array`, `end_object`, `literal_true`, or `parse_error`.
- **Functions called**:
    - [`skip_bom`](#skip_bom)
    - [`skip_whitespace`](#skip_whitespace)
    - [`(anonymous)::namespace::lexer::scan_comment`](#(anonymous)::namespace::lexer::scan_comment)
    - [`(anonymous)::namespace::lexer::scan_string`](#(anonymous)::namespace::lexer::scan_string)
    - [`(anonymous)::namespace::lexer::scan_number`](#(anonymous)::namespace::lexer::scan_number)


---
### json\_sax<!-- {{#callable:json_sax}} -->
The `json_sax` function is a move constructor that enables the transfer of resources from one `json_sax` object to another.
- **Inputs**:
    - `json_sax&&`: An rvalue reference to a `json_sax` object, allowing the transfer of ownership of resources.
- **Control Flow**:
    - The function utilizes the default move constructor behavior, which efficiently transfers the internal state of the source object to the new object.
    - No additional logic is implemented, as the default behavior suffices for resource management.
- **Output**: The function does not return a value; it constructs a new `json_sax` object by moving the resources from the provided rvalue reference.


---
### \~json\_sax<!-- {{#callable:~json_sax}} -->
Destructor for the `json_sax` class.
- **Inputs**: None
- **Control Flow**:
    - The function is a virtual destructor, which means it can be overridden in derived classes.
    - It does not contain any additional logic or control flow statements.
- **Output**: The function does not return a value; it cleans up resources when an object of `json_sax` or its derived classes is destroyed.


---
### unknown\_size<!-- {{#callable:detail::unknown_size}} -->
Returns the maximum value representable by `std::size_t`.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the maximum value of `std::size_t` using `std::numeric_limits` without any conditional logic or loops.
- **Output**: The output is a `std::size_t` value representing the maximum possible size that can be held by the `std::size_t` type.


---
### little\_endianness<!-- {{#callable:NLOHMANN_JSON_NAMESPACE_BEGIN::little_endianness}} -->
Determines if the system architecture is little-endian.
- **Inputs**:
    - `num`: An integer value, defaulting to 1, used to check the endianness.
- **Control Flow**:
    - Uses a pointer cast to reinterpret the address of `num` as a pointer to a `char`.
    - Checks if the first byte of the integer representation is equal to 1, indicating little-endian format.
- **Output**: Returns a boolean value: true if the system is little-endian, false otherwise.


---
### set\_end<!-- {{#callable:namespace::set_end}} -->
The `set_end` function sets the iterator to a position past the last value of a JSON object or array.
- **Inputs**:
    - `none`: The function does not take any input arguments.
- **Control Flow**:
    - The function asserts that the iterator's associated JSON object is not null.
    - It checks the type of the JSON object (object, array, or primitive) using a switch statement.
    - If the type is an object, it sets the object iterator to the end of the object's internal data.
    - If the type is an array, it sets the array iterator to the end of the array's internal data.
    - For all other types (including null, string, boolean, number, binary, and discarded), it calls `set_end` on the primitive iterator.
- **Output**: The function does not return a value; it modifies the internal state of the iterator to point past the last element.
- **Functions called**:
    - [`namespace::primitive_iterator_t`](#namespaceprimitive_iterator_t)
    - [`namespace::primitive_iterator_t::set_begin`](#primitive_iterator_tset_begin)


---
### operator\-><!-- {{#callable:operator->}} -->
The `operator->` function provides access to the underlying value of a JSON object or array, returning a pointer to the appropriate data type based on the current state of the iterator.
- **Inputs**: None
- **Control Flow**:
    - The function first asserts that `m_object` is not null using `JSON_ASSERT`.
    - It then checks the type of the data contained in `m_object` using a switch statement.
    - If the type is `object`, it asserts that the object iterator is valid and returns a pointer to the second element of the current object iterator.
    - If the type is `array`, it asserts that the array iterator is valid and returns a pointer to the current element of the array iterator.
    - For primitive types (null, string, boolean, number types, binary, discarded), it checks if the primitive iterator is at the beginning.
    - If the primitive iterator is at the beginning, it returns a pointer to `m_object`; otherwise, it throws an `invalid_iterator` exception.
- **Output**: The function returns a pointer to the value represented by `m_object`, which can be of various types depending on the state of the iterator and the type of the JSON data.


---
### operator\+\+<!-- {{#callable:operator++}} -->
Increments the iterator based on the type of the JSON object it points to.
- **Inputs**: None
- **Control Flow**:
    - The function first asserts that `m_object` is not null using `JSON_ASSERT`.
    - It then checks the type of the JSON object using a `switch` statement on `m_object->m_data.m_type`.
    - If the type is `object`, it advances the `object_iterator` by one.
    - If the type is `array`, it advances the `array_iterator` by one.
    - For all other types (including `null`, `string`, `boolean`, `number`, `binary`, and `discarded`), it increments the `primitive_iterator`.
- **Output**: Returns a reference to the current instance of `iter_impl` after incrementing the appropriate iterator.


---
### operator\-\-<!-- {{#callable:operator--}} -->
Decrements the iterator based on the type of the underlying JSON object.
- **Inputs**:
    - `this`: A reference to the current instance of `iter_impl`, which contains the iterator state.
- **Control Flow**:
    - Checks if `m_object` is not null using `JSON_ASSERT`.
    - Switches on the type of the JSON object (`m_data.m_type`):
    - If the type is `object`, decrements the `object_iterator` by one.
    - If the type is `array`, decrements the `array_iterator` by one.
    - For all other types (including `null`, `string`, `boolean`, `number`, etc.), decrements the `primitive_iterator` by one.
    - Returns a reference to the current instance of `iter_impl`.
- **Output**: Returns a reference to the updated `iter_impl` instance after decrementing the appropriate iterator.


---
### operator==<!-- {{#callable:operator==}} -->
The `operator==` function compares a `ScalarType` object with a `const_reference` object for equality.
- **Inputs**:
    - `lhs`: A `ScalarType` object representing the left-hand side of the equality comparison.
    - `rhs`: A `const_reference` object representing the right-hand side of the equality comparison.
- **Control Flow**:
    - The function is defined as a friend function, allowing it to access private members of the class if necessary.
    - It constructs a [`basic_json`](#basic_json) object from the `lhs` parameter.
    - It then compares the constructed [`basic_json`](#basic_json) object with the `rhs` parameter using the `==` operator.
- **Output**: The function returns a boolean value indicating whether the two objects are equal.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### operator<=<!-- {{#callable:operator<=}} -->
The `operator<=` function compares a `ScalarType` with a `const_reference` using the [`basic_json`](#basic_json) class.
- **Inputs**:
    - `lhs`: A value of type `ScalarType` that serves as the left-hand side operand in the comparison.
    - `rhs`: A reference to a constant value of type `const_reference` that serves as the right-hand side operand in the comparison.
- **Control Flow**:
    - The function is defined as a friend function, allowing it to access private members of the class it belongs to.
    - It constructs a [`basic_json`](#basic_json) object using the `lhs` operand.
    - It then performs the less than or equal to comparison (`<=`) between the constructed [`basic_json`](#basic_json) object and the `rhs` operand.
- **Output**: The function returns a boolean value indicating whether `lhs` is less than or equal to `rhs`.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### operator><!-- {{#callable:operator>}} -->
The `operator>` function compares a `ScalarType` object with a `const_reference` object to determine if the former is greater than the latter.
- **Inputs**:
    - `lhs`: A `ScalarType` object representing the left-hand side operand in the comparison.
    - `rhs`: A `const_reference` object representing the right-hand side operand in the comparison.
- **Control Flow**:
    - The function begins by converting the `lhs` operand into a [`basic_json`](#basic_json) object.
    - It then uses the `>` operator to compare the newly created [`basic_json`](#basic_json) object with the `rhs` operand.
    - The result of the comparison (a boolean value) is returned.
- **Output**: A boolean value indicating whether the `lhs` is greater than the `rhs`.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### operator>=<!-- {{#callable:operator>=}} -->
The `operator>=` function compares a `ScalarType` with a `const_reference` and returns a boolean indicating if the left-hand side is greater than or equal to the right-hand side.
- **Inputs**:
    - `lhs`: A value of type `ScalarType` representing the left-hand side operand in the comparison.
    - `rhs`: A reference to a constant value of type `const_reference` representing the right-hand side operand in the comparison.
- **Control Flow**:
    - The function calls the [`basic_json`](#basic_json) constructor to convert `lhs` into a [`basic_json`](#basic_json) object.
    - It then uses the `>=` operator of [`basic_json`](#basic_json) to compare the newly created object with `rhs`.
    - The result of the comparison is returned as a boolean value.
- **Output**: The function outputs a boolean value indicating whether `lhs` is greater than or equal to `rhs`.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### operator\+=<!-- {{#callable:operator+=}} -->
Overloads the `+=` operator to append elements from an initializer list to the current object.
- **Inputs**:
    - `init`: An `initializer_list_t` representing a list of elements to be appended to the current object.
- **Control Flow**:
    - Calls the [`push_back`](#push_back) method to add the elements from the initializer list to the current object.
    - Returns a reference to the current object after the elements have been added.
- **Output**: Returns a reference to the current object (`*this`) after appending the elements from the initializer list.
- **Functions called**:
    - [`push_back`](#push_back)


---
### operator\-=<!-- {{#callable:operator-=}} -->
This function overloads the `-=` operator to decrement an iterator by a specified number of positions.
- **Inputs**:
    - `i`: A `difference_type` representing the number of positions to decrement the iterator.
- **Control Flow**:
    - The function calls the `operator+=` method with the negated value of `i` to achieve the decrement operation.
- **Output**: Returns a reference to the modified `iter_impl` object after decrementing its position.


---
### operator\+<!-- {{#callable:operator+}} -->
The `operator+` function adds a specified integer offset to an `iter_impl` object.
- **Inputs**:
    - `i`: An integer value representing the offset to be added to the iterator.
    - `it`: A constant reference to an `iter_impl` object to which the offset will be applied.
- **Control Flow**:
    - A copy of the `iter_impl` object `it` is created and stored in `result`.
    - The integer offset `i` is added to `result` using the `+=` operator.
    - The modified `result` is returned.
- **Output**: The function returns a new `iter_impl` object that represents the original iterator offset by the specified integer value.


---
### operator\-<!-- {{#callable:operator-}} -->
Calculates the difference between two iterators based on the type of the underlying JSON object.
- **Inputs**:
    - `other`: An instance of `iter_impl` representing the iterator to be subtracted from the current iterator.
- **Control Flow**:
    - Asserts that the member variable `m_object` is not null using `JSON_ASSERT`.
    - Checks the type of the JSON object pointed to by `m_object` using a switch statement.
    - If the type is `object`, throws an `invalid_iterator` exception indicating that offsets cannot be used with object iterators.
    - If the type is `array`, calculates the difference between the two array iterators.
    - For all other types (including `null`, `string`, `boolean`, `number`, `binary`, and `discarded`), calculates the difference using the primitive iterators.
- **Output**: Returns a `difference_type` representing the calculated difference between the two iterators.


---
### operator\[\]<!-- {{#callable:operator[]}} -->
This function overloads the `operator[]` to retrieve a value from a JSON object using a JSON pointer.
- **Inputs**:
    - `ptr`: A constant reference to a `nlohmann::json_pointer<BasicJsonType>` object that specifies the path to the desired value in the JSON structure.
- **Control Flow**:
    - The function directly calls the `get_unchecked` method on the `ptr` object, passing `this` as an argument.
    - No conditional statements or loops are present; the function's logic is straightforward and relies on the `get_unchecked` method to perform the retrieval.
- **Output**: Returns a constant reference to the value located at the specified JSON pointer within the JSON object.


---
### key<!-- {{#callable:key}} -->
Returns the key of the current object in an iterator if the underlying object is indeed an object.
- **Inputs**: None
- **Control Flow**:
    - The function first asserts that `m_object` is not a null pointer using `JSON_ASSERT`.
    - It checks if `m_object` is an object using `is_object()` method.
    - If `m_object` is an object, it returns the key from the object iterator.
    - If `m_object` is not an object, it throws an `invalid_iterator` exception with an error message.
- **Output**: Returns a constant reference to the key of the current object in the iterator if valid; otherwise, it throws an exception.


---
### value<!-- {{#callable:value}} -->
This function retrieves a value from a JSON pointer, providing a default value if the pointer does not point to a valid value.
- **Inputs**:
    - `ptr`: A constant reference to a `nlohmann::json_pointer<BasicJsonType>` object that specifies the location of the value in the JSON structure.
    - `default_value`: An rvalue reference of type `ValueType` that serves as the default value to return if the specified JSON pointer does not yield a valid value.
- **Control Flow**:
    - The function first converts the JSON pointer `ptr` to its corresponding value type using `ptr.convert()`.
    - It then calls another overloaded [`value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::value) function, passing the converted pointer and the forwarded `default_value`.
- **Output**: The function returns a value of type `ReturnType`, which is either the value found at the specified JSON pointer or the provided `default_value` if the pointer is invalid.
- **Functions called**:
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::value`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy_value::value)


---
### json\_pointer<!-- {{#callable:json_pointer}} -->
Constructs a `json_pointer` object from a string representation.
- **Inputs**:
    - `s`: A string representing the JSON pointer, defaulting to an empty string if not provided.
- **Control Flow**:
    - The constructor initializes the `json_pointer` object using the provided string.
    - It calls the [`split`](#split) function to convert the string into a list of reference tokens.
- **Output**: The constructor does not return a value but initializes the `reference_tokens` member with the result of the [`split`](#split) function.
- **Functions called**:
    - [`split`](#split)


---
### operator<<<!-- {{#callable:operator<<}} -->
Overloads the `<<` operator to serialize a `basic_json` object to an output stream.
- **Inputs**:
    - `o`: An output stream (`std::ostream`) where the `basic_json` object will be serialized.
    - `j`: A constant reference to a `basic_json` object that contains the data to be serialized.
- **Control Flow**:
    - Checks if the width of the output stream is greater than zero to determine if pretty printing is enabled.
    - Sets the indentation level based on the output stream's width if pretty printing is enabled, otherwise sets it to zero.
    - Resets the width of the output stream to zero to avoid affecting subsequent output operations.
    - Creates a `serializer` object initialized with an output adapter for the stream and the current fill character.
    - Calls the `dump` method of the `serializer` to perform the actual serialization of the `basic_json` object with the specified pretty print and indentation settings.
- **Output**: Returns a reference to the output stream after the serialization of the `basic_json` object is complete.


---
### operator/=<!-- {{#callable:operator/=}} -->
Overloads the `/=` operator to append a string representation of an array index to a `json_pointer`.
- **Inputs**:
    - `array_idx`: A `std::size_t` representing the index of an array to be appended to the `json_pointer`.
- **Control Flow**:
    - The function calls itself recursively with the string representation of the `array_idx` converted using `std::to_string`.
    - The result of the recursive call is returned, effectively modifying the current `json_pointer` instance.
- **Output**: Returns a reference to the modified `json_pointer` instance after appending the string representation of the array index.


---
### operator/<!-- {{#callable:json_pointer::operator/}} -->
Overloads the division operator to append an array index to a [`json_pointer`](#json_pointer).
- **Inputs**:
    - `lhs`: A constant reference to a [`json_pointer`](#json_pointer) object representing the left-hand side of the division operation.
    - `array_idx`: A `std::size_t` value representing the index of the array to be appended to the [`json_pointer`](#json_pointer).
- **Control Flow**:
    - The function takes two parameters: a [`json_pointer`](#json_pointer) reference and a size_t index.
    - It creates a new [`json_pointer`](#json_pointer) object by copying the `lhs` parameter.
    - The copied [`json_pointer`](#json_pointer) is then modified by appending the `array_idx` using the overloaded `/=` operator.
- **Output**: Returns a new [`json_pointer`](#json_pointer) that represents the original pointer with the specified array index appended.
- **Functions called**:
    - [`json_pointer`](#json_pointer)


---
### parent\_pointer<!-- {{#callable:parent_pointer}} -->
Returns the parent `json_pointer` of the current pointer, or itself if it is empty.
- **Inputs**: None
- **Control Flow**:
    - Checks if the current `json_pointer` is empty using the `empty()` method.
    - If it is empty, returns the current `json_pointer` (i.e., `*this`).
    - If it is not empty, creates a copy of the current `json_pointer` and removes the last component using `pop_back()`.
    - Returns the modified `json_pointer` which represents the parent.
- **Output**: A `json_pointer` object representing the parent of the current pointer, or the current pointer itself if it is empty.
- **Functions called**:
    - [`(anonymous)::namespace::iterator_input_adapter::empty`](#(anonymous)::namespace::iterator_input_adapter::empty)


---
### pop\_back<!-- {{#callable:pop_back}} -->
Removes the last element from the `reference_tokens` container if it is not empty.
- **Inputs**: None
- **Control Flow**:
    - Checks if the `reference_tokens` container is empty using the `empty()` method.
    - If the container is empty, it throws an `out_of_range` exception with a specific error message.
    - If the container is not empty, it proceeds to remove the last element from `reference_tokens` using `pop_back()`.
- **Output**: This function does not return a value; it modifies the state of the `reference_tokens` container by removing its last element.
- **Functions called**:
    - [`(anonymous)::namespace::iterator_input_adapter::empty`](#(anonymous)::namespace::iterator_input_adapter::empty)


---
### back<!-- {{#callable:back}} -->
Returns a reference to the last element of a constant container.
- **Inputs**:
    - `this`: A constant reference to the container from which the last element is being accessed.
- **Control Flow**:
    - Calls `cend()` to get an iterator to the end of the container.
    - Decrements the iterator to point to the last element.
    - Dereferences the iterator to return the last element.
- **Output**: A constant reference to the last element of the container.
- **Functions called**:
    - [`cend`](#cend)


---
### push\_back<!-- {{#callable:push_back}} -->
Adds elements from an initializer list to a JSON object.
- **Inputs**:
    - `init`: An initializer list containing key-value pairs to be added to the JSON object.
- **Control Flow**:
    - Checks if the current object is a valid JSON object and if the initializer list contains exactly two elements, with the first being a string.
    - If the conditions are met, it extracts the key from the first element and the value from the second element, then calls [`push_back`](#push_back) to add them to the JSON object.
    - If the conditions are not met, it converts the initializer list into a [`basic_json`](#basic_json) object and calls [`push_back`](#push_back) to add it.
- **Output**: The function does not return a value; it modifies the JSON object by adding the specified key-value pairs.
- **Functions called**:
    - [`is_object`](#is_object)
    - [`push_back`](#push_back)
    - [`basic_json`](#basic_json)


---
### empty<!-- {{#callable:empty}} -->
The `empty` function checks if the current object is empty based on its type.
- **Inputs**: None
- **Control Flow**:
    - The function begins by checking the type of the current object's data using a `switch` statement.
    - If the type is `null`, it returns `true` indicating the object is empty.
    - If the type is `array`, it calls the `empty` method of the associated `array_t` object to determine emptiness.
    - If the type is `object`, it calls the `empty` method of the associated `object_t` object.
    - For all other types (including `string`, `boolean`, `number_integer`, `number_unsigned`, `number_float`, `binary`, and `discarded`), it returns `false`, indicating the object is not empty.
- **Output**: The function returns a boolean value: `true` if the object is empty, and `false` otherwise.


---
### get\_and\_create<!-- {{#callable:get_and_create}} -->
The `get_and_create` function traverses a JSON structure based on reference tokens and creates new entries as needed.
- **Inputs**:
    - `j`: A reference to a JSON object of type `BasicJsonType` that will be modified or traversed.
- **Control Flow**:
    - The function initializes a pointer `result` to the input JSON reference `j`.
    - It iterates over each `reference_token` in the `reference_tokens` collection.
    - For each token, it checks the type of the current JSON value pointed to by `result`.
    - If the type is `null`, it either starts a new array or a new object based on the token value.
    - If the type is `object`, it creates a new entry in the object using the token.
    - If the type is `array`, it creates a new entry in the array using the token as an index.
    - If the type is a primitive value (string, boolean, number, etc.), it throws a type error indicating an invalid operation.
- **Output**: The function returns a reference to the modified or newly created JSON value.


---
### get\_unchecked<!-- {{#callable:get_unchecked}} -->
The `get_unchecked` function retrieves a value from a JSON structure using a series of reference tokens without performing bounds checking.
- **Inputs**:
    - `ptr`: A pointer to a `BasicJsonType` object representing the starting point of the JSON structure from which values are to be accessed.
- **Control Flow**:
    - The function iterates over each `reference_token` in the `reference_tokens` collection.
    - For each token, it checks the type of the JSON object pointed to by `ptr` using a switch statement.
    - If the type is `object`, it accesses the value associated with the `reference_token` using unchecked access.
    - If the type is `array`, it checks if the `reference_token` is '-', throwing an exception if it is, otherwise it accesses the array element using the token as an index.
    - For all other types, it throws an exception indicating that the reference token could not be resolved.
- **Output**: The function returns a constant reference to the value of type `BasicJsonType` that was accessed using the reference tokens.


---
### get\_checked<!-- {{#callable:get_checked}} -->
The `get_checked` function retrieves a value from a JSON structure based on a series of reference tokens, ensuring that each access is valid.
- **Inputs**:
    - `ptr`: A pointer to a `BasicJsonType` object representing the starting point of the JSON structure from which values are to be retrieved.
- **Control Flow**:
    - The function iterates over each `reference_token` in the `reference_tokens` collection.
    - For each token, it checks the type of the JSON object pointed to by `ptr` using a switch statement.
    - If the type is `object`, it updates `ptr` to point to the value associated with the current `reference_token` using the `at` method, which performs a range check.
    - If the type is `array`, it checks if the `reference_token` is '-', which is invalid and throws an exception if true; otherwise, it updates `ptr` using the `at` method with the converted array index.
    - For all other types (including `null`, `string`, `boolean`, etc.), it throws an exception indicating that the reference token is unresolved.
- **Output**: The function returns a constant reference to the `BasicJsonType` object pointed to by `ptr` after successfully resolving all reference tokens.


---
### contains<!-- {{#callable:contains}} -->
Checks if a given JSON pointer is contained within the current JSON object.
- **Inputs**:
    - `ptr`: A constant reference to a `json_pointer` object that points to a specific location within a JSON structure.
- **Control Flow**:
    - The function directly calls the `contains` method of the `json_pointer` class, passing the current object (`this`) as an argument.
    - The result of the `contains` method is returned as a boolean value.
- **Output**: Returns true if the JSON pointer `ptr` points to a location within the current JSON object; otherwise, returns false.


---
### split<!-- {{#callable:split}} -->
Splits a JSON pointer string into its individual reference tokens.
- **Inputs**:
    - `reference_string`: A constant reference to a string representing a JSON pointer, which must start with a '/' character.
- **Control Flow**:
    - Checks if the `reference_string` is empty; if so, returns an empty result vector.
    - Validates that the `reference_string` starts with a '/' character; if not, throws a parse error.
    - Iterates through the `reference_string` to find and extract tokens between slashes.
    - For each extracted token, checks for proper escaping of the '~' character and throws an error if the escaping is incorrect.
    - Calls a helper function `detail::unescape` to process the token before adding it to the result vector.
- **Output**: Returns a vector of strings, each representing a reference token extracted from the `reference_string`.


---
### convert<!-- {{#callable:convert}} -->
The `convert` function transfers ownership of `reference_tokens` to a new `json_pointer` object.
- **Inputs**:
    - `this`: An rvalue reference to the current object, allowing the function to modify and transfer resources.
- **Control Flow**:
    - A new `json_pointer<string_t>` object named `result` is created.
    - The `reference_tokens` member of the current object is moved into `result.reference_tokens`, transferring ownership.
    - The `result` object is returned.
- **Output**: The function outputs a `json_pointer<string_t>` object that contains the moved `reference_tokens`.


---
### operator<=><!-- {{#callable:operator<=>}} -->
This function overloads the spaceship operator (<=>) for comparing a [`basic_json`](#basic_json) object with a scalar type.
- **Inputs**:
    - `rhs`: A scalar value of type `ScalarType` that is to be compared with the current [`basic_json`](#basic_json) object.
- **Control Flow**:
    - The function begins by invoking the spaceship operator on the current object (`*this`) and a new [`basic_json`](#basic_json) object constructed from the `rhs` scalar value.
    - The result of the comparison is returned directly.
- **Output**: The output is a `std::partial_ordering` value that indicates the result of the comparison between the [`basic_json`](#basic_json) object and the scalar value, which can be less than, equal to, or greater than.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### reinterpret\_bits<!-- {{#callable:namespace::dtoa_impl::reinterpret_bits}} -->
Reinterprets the bits of a `Source` type variable as a `Target` type variable, ensuring both types have the same size.
- **Inputs**:
    - `source`: An instance of the `Source` type whose bits are to be reinterpreted.
- **Control Flow**:
    - The function begins with a static assertion to ensure that the sizes of `Target` and `Source` are equal, preventing potential runtime errors due to size mismatch.
    - A `Target` type variable named `target` is declared.
    - The `std::memcpy` function is used to copy the raw bytes from the `source` variable into the `target` variable.
    - The function returns the `target` variable, which now contains the bit representation of the `source` variable.
- **Output**: An instance of the `Target` type that contains the bitwise representation of the input `source` variable.


---
### compute\_boundaries<!-- {{#callable:namespace::dtoa_impl::compute_boundaries}} -->
The `compute_boundaries` function calculates the floating-point boundaries for a given finite positive value.
- **Inputs**:
    - `value`: A finite positive floating-point number of type `FloatType`.
- **Control Flow**:
    - The function asserts that the input `value` is finite and greater than zero.
    - It checks if the floating-point representation is denormal or normalized and computes the corresponding [`diyfp`](#diyfpdiyfp) representation.
    - It calculates the floating-point predecessor (`v-`) and successor (`v+`) based on the value's characteristics.
    - It computes the midpoints `m-` and `m+` which represent the boundaries around the value.
    - Finally, it normalizes the boundaries and returns them along with the normalized value.
- **Output**: The function returns a `boundaries` structure containing the normalized value, the lower boundary (`w_minus`), and the upper boundary (`w_plus`).
- **Functions called**:
    - [`namespace::dtoa_impl::diyfp::diyfp`](#diyfpdiyfp)


---
### get\_cached\_power\_for\_binary\_exponent<!-- {{#callable:namespace::dtoa_impl::get_cached_power_for_binary_exponent}} -->
Retrieves a cached power of two based on a given binary exponent.
- **Inputs**:
    - `e`: An integer representing the binary exponent for which the cached power is to be retrieved.
- **Control Flow**:
    - The function asserts that the input exponent `e` is within the range of -1500 to 1500.
    - It calculates an intermediate value `f` based on the input exponent `e` and a constant `kAlpha`.
    - Using `f`, it computes an index for the cached power array by adjusting `f` and normalizing it with respect to the minimum decimal exponent and step size.
    - The function retrieves the cached power from the static array `kCachedPowers` using the calculated index.
    - It asserts that the retrieved cached power meets certain conditions related to the constants `kAlpha` and `kGamma` before returning it.
- **Output**: Returns a `cached_power` structure that contains the cached power of two corresponding to the input exponent `e`.


---
### find\_largest\_pow10<!-- {{#callable:namespace::dtoa_impl::find_largest_pow10}} -->
Determines the largest power of 10 less than or equal to a given number.
- **Inputs**:
    - `n`: A 32-bit unsigned integer representing the number to compare against powers of 10.
    - `pow10`: A reference to a 32-bit unsigned integer that will store the largest power of 10 found.
- **Control Flow**:
    - The function checks if `n` is greater than or equal to 1,000,000,000 and sets `pow10` to 1,000,000,000 if true, returning 10.
    - If the first condition is false, it checks if `n` is greater than or equal to 100,000,000, setting `pow10` to 100,000,000 and returning 9.
    - This pattern continues with decreasing powers of 10 (10^7, 10^6, etc.) until it checks if `n` is greater than or equal to 10.
    - If none of the conditions are met, it sets `pow10` to 1 and returns 1.
- **Output**: Returns an integer representing the exponent of the largest power of 10 that is less than or equal to `n`, while also updating `pow10` with the actual power of 10.


---
### grisu2\_round<!-- {{#callable:namespace::dtoa_impl::grisu2_round}} -->
The `grisu2_round` function adjusts a character buffer representing a decimal number by decrementing its last character to round the number down based on specified parameters.
- **Inputs**:
    - `buf`: A pointer to a character array that holds the decimal representation of a number.
    - `len`: An integer representing the length of the `buf` array.
    - `dist`: A `std::uint64_t` value representing the target distance for rounding.
    - `delta`: A `std::uint64_t` value representing the maximum allowable distance for rounding.
    - `rest`: A `std::uint64_t` value representing the current remainder in the rounding process.
    - `ten_k`: A `std::uint64_t` value representing the unit-in-the-last-place in the decimal representation.
- **Control Flow**:
    - The function begins by asserting that the input values meet certain conditions to prevent invalid operations.
    - It enters a while loop that continues as long as the `rest` is less than `dist`, the difference between `delta` and `rest` is greater than or equal to `ten_k`, and the conditions for rounding are satisfied.
    - Within the loop, it decrements the last character in `buf` and updates `rest` by adding `ten_k` until the rounding conditions are no longer met.
- **Output**: The function does not return a value; instead, it modifies the `buf` array in place to reflect the rounded decimal representation.


---
### grisu2\_digit\_gen<!-- {{#callable:namespace::dtoa_impl::grisu2_digit_gen}} -->
Generates the decimal digits of a floating-point number within a specified range.
- **Inputs**:
    - `buffer`: A character array where the generated decimal digits will be stored.
    - `length`: An integer reference that tracks the current length of the generated digits in the buffer.
    - `decimal_exponent`: An integer reference that will be updated to reflect the exponent of the decimal representation.
    - `M_minus`: A `diyfp` structure representing the lower bound of the range.
    - `w`: A `diyfp` structure representing the value to be converted.
    - `M_plus`: A `diyfp` structure representing the upper bound of the range.
- **Control Flow**:
    - The function starts by asserting the validity of the input parameters.
    - It calculates the differences between `M_plus`, `M_minus`, and `w` to determine the range for digit generation.
    - The integral part of `M_plus` is split into two parts, `p1` and `p2`, for processing.
    - Digits of the integral part are generated from `p1` until the generated value is within the specified range.
    - If the generated digits are sufficient, the function may round the result and return.
    - If not, the function continues to generate the fractional part from `p2` until the required precision is achieved.
    - Finally, the function rounds the result to ensure it accurately represents the original value.
- **Output**: The function does not return a value but modifies the `buffer`, `length`, and `decimal_exponent` to represent the decimal digits of the floating-point number within the specified range.
- **Functions called**:
    - [`namespace::dtoa_impl::find_largest_pow10`](#dtoa_implfind_largest_pow10)
    - [`namespace::dtoa_impl::grisu2_round`](#dtoa_implgrisu2_round)


---
### grisu2<!-- {{#callable:namespace::dtoa_impl::grisu2}} -->
The `grisu2` function generates a string representation of a floating-point number using the Grisu2 algorithm.
- **Inputs**:
    - `buf`: A character buffer where the resulting string representation of the floating-point number will be stored.
    - `len`: An integer reference that will hold the length of the resulting string.
    - `decimal_exponent`: An integer reference that will store the exponent of the decimal representation.
    - `m_minus`: A `diyfp` structure representing the lower bound of the range for the floating-point number.
    - `v`: A `diyfp` structure representing the value to be converted to a string.
    - `m_plus`: A `diyfp` structure representing the upper bound of the range for the floating-point number.
- **Control Flow**:
    - The function begins by asserting that the exponents of `m_plus`, `m_minus`, and `v` are equal.
    - It retrieves a cached power for the binary exponent of `m_plus` to scale the values.
    - The function scales `v`, `m_minus`, and `m_plus` using the cached power to ensure their exponents are within a specific range.
    - It calculates the bounds `M_minus` and `M_plus` to account for rounding inaccuracies.
    - Finally, it calls the [`grisu2_digit_gen`](#dtoa_implgrisu2_digit_gen) function to generate the string representation of the number using the calculated bounds.
- **Output**: The function does not return a value but modifies the `buf`, `len`, and `decimal_exponent` parameters to provide the string representation and its properties.
- **Functions called**:
    - [`namespace::dtoa_impl::get_cached_power_for_binary_exponent`](#dtoa_implget_cached_power_for_binary_exponent)
    - [`namespace::dtoa_impl::grisu2_digit_gen`](#dtoa_implgrisu2_digit_gen)


---
### append\_exponent<!-- {{#callable:namespace::dtoa_impl::append_exponent}} -->
The `append_exponent` function appends a formatted exponent to a character buffer.
- **Inputs**:
    - `buf`: A pointer to a character buffer where the formatted exponent will be appended.
    - `e`: An integer representing the exponent value to be formatted and appended.
- **Control Flow**:
    - The function asserts that the exponent `e` is within the range of -1000 to 1000.
    - If `e` is negative, it negates `e` and appends a '-' character to the buffer; otherwise, it appends a '+' character.
    - The absolute value of `e` is cast to a `std::uint32_t` for further processing.
    - Depending on the value of `k` (the absolute value of `e`), the function appends the appropriate number of digits to the buffer, ensuring at least two digits are printed.
- **Output**: Returns a pointer to the next position in the buffer after the appended exponent.


---
### format\_buffer<!-- {{#callable:namespace::dtoa_impl::format_buffer}} -->
Formats a buffer to represent a floating-point number in scientific notation.
- **Inputs**:
    - `buf`: A pointer to a character array where the formatted number will be stored.
    - `len`: The length of the buffer `buf`.
    - `decimal_exponent`: The exponent indicating the position of the decimal point.
    - `min_exp`: The minimum exponent value, which must be less than 0.
    - `max_exp`: The maximum exponent value, which must be greater than 0.
- **Control Flow**:
    - The function starts by asserting that `min_exp` is less than 0 and `max_exp` is greater than 0.
    - It calculates `k` as the length of the buffer and `n` as the sum of `len` and `decimal_exponent`.
    - If `k` is less than or equal to `n` and `n` is less than or equal to `max_exp`, it pads the buffer with zeros and places the decimal point.
    - If `n` is positive and less than or equal to `max_exp`, it moves the digits to insert the decimal point at the correct position.
    - If `n` is between `min_exp` and 0, it adjusts the buffer to represent a number in the form of 0.[digits].
    - If `k` is 1, it adjusts the buffer for a single digit; otherwise, it moves the digits to insert a decimal point.
    - Finally, it appends the exponent part to the buffer and returns a pointer to the end of the formatted string.
- **Output**: Returns a pointer to the end of the formatted string in the buffer.
- **Functions called**:
    - [`namespace::dtoa_impl::append_exponent`](#dtoa_implappend_exponent)


---
### hex\_bytes<!-- {{#callable:hex_bytes}} -->
Converts a single byte into its hexadecimal string representation.
- **Inputs**:
    - `byte`: An 8-bit unsigned integer (uint8_t) representing the byte to be converted.
- **Control Flow**:
    - Initializes a string `result` with a default value of 'FF'.
    - Defines a constant character array `nibble_to_hex` that maps nibble values to their hexadecimal character representations.
    - Calculates the first hexadecimal character by dividing the byte by 16 and indexing into `nibble_to_hex`.
    - Calculates the second hexadecimal character by taking the byte modulo 16 and indexing into `nibble_to_hex`.
    - Returns the resulting hexadecimal string.
- **Output**: A string representing the hexadecimal value of the input byte, formatted as two uppercase characters.


---
### is\_negative\_number<!-- {{#callable:is_negative_number}} -->
This function checks if a given unsigned number is negative, which it always returns false.
- **Inputs**:
    - `NumberType`: An unsigned number type that is passed to the function, but not used in the logic.
- **Control Flow**:
    - The function uses a template to accept any unsigned number type as input.
    - It employs SFINAE (Substitution Failure Is Not An Error) to ensure that only unsigned types can be passed to the function.
    - The function body contains a single return statement that always returns false, indicating that unsigned numbers cannot be negative.
- **Output**: The function returns a boolean value, which is always false, as unsigned numbers cannot be negative.


---
### dump\_integer<!-- {{#callable:dump_integer}} -->
The `dump_integer` function converts an integral number to its string representation and writes it to an output stream.
- **Inputs**:
    - `x`: An integral number of type `NumberType`, which can be various integral types or specific number types.
- **Control Flow**:
    - The function first checks if the input `x` is zero and writes '0' to the output if true.
    - It initializes a pointer to a buffer for storing the string representation of the number.
    - If `x` is negative, it stores a '-' sign and calculates the absolute value, adjusting the character count accordingly.
    - The function then calculates the number of digits in the absolute value of `x`.
    - It ensures that the buffer has enough space for the characters and a null terminator.
    - The function converts the number to its string representation in reverse order to avoid needing to reverse the string later.
    - It uses a pre-defined array `digits_to_99` to efficiently convert numbers to characters.
    - Finally, it writes the resulting string from the buffer to the output stream.
- **Output**: The function does not return a value; instead, it writes the string representation of the integer to the output stream.
- **Functions called**:
    - [`is_negative_number`](#is_negative_number)
    - [`remove_sign`](#remove_sign)


---
### dump\_float<!-- {{#callable:dump_float}} -->
The `dump_float` function converts a floating-point number to a string representation, handling formatting and writing it to an output stream.
- **Inputs**:
    - `x`: The floating-point number of type `number_float_t` to be converted and dumped.
    - `is_ieee_single_or_double`: A `std::false_type` indicating that the function is not dealing with IEEE single or double precision formats.
- **Control Flow**:
    - Defines a constant `d` representing the maximum number of digits for the float conversion.
    - Uses `std::snprintf` to convert the float `x` into a string stored in `number_buffer`, checking for errors in the process.
    - If a thousands separator is defined, it removes it from the string representation.
    - If a custom decimal point is defined and is not '.', it replaces it with '.' in the string.
    - Writes the formatted string to the output stream `o`.
    - Checks if the number is integer-like (i.e., does not contain '.' or 'e') and appends '.0' if it is.
- **Output**: The function does not return a value but writes the formatted string representation of the float to the output stream, potentially appending '.0' if the number is integer-like.


---
### decode<!-- {{#callable:decode}} -->
Decodes a single UTF-8 byte and updates the state and code point accordingly.
- **Inputs**:
    - `state`: A reference to a `std::uint8_t` representing the current state of the UTF-8 decoding process.
    - `codep`: A reference to a `std::uint32_t` that holds the current code point being constructed from the UTF-8 bytes.
    - `byte`: A `std::uint8_t` representing the next byte to decode in the UTF-8 sequence.
- **Control Flow**:
    - The function begins by asserting that the `byte` is within the bounds of the `utf8d` array.
    - It retrieves the type of the byte from the `utf8d` array based on its value.
    - If the current `state` is not `UTF8_ACCEPT`, it updates `codep` by shifting the existing code point and adding the new byte.
    - If the state is `UTF8_ACCEPT`, it initializes `codep` with the byte masked by the type.
    - The function calculates an index to access the `utf8d` array based on the current `state` and the type of the byte.
    - It asserts that the calculated index is valid and updates the `state` with the value from the `utf8d` array at that index.
    - Finally, it returns the updated `state`.
- **Output**: Returns the updated state of the UTF-8 decoding process as a `std::uint8_t`.


---
### remove\_sign<!-- {{#callable:remove_sign}} -->
The `remove_sign` function converts a negative signed integer to its corresponding positive unsigned integer representation.
- **Inputs**:
    - `x`: A signed integer of type `number_integer_t` that is expected to be negative.
- **Control Flow**:
    - The function first asserts that the input `x` is negative and within the valid range of `number_integer_t`.
    - It then calculates the positive equivalent of the negative integer by negating `x`, adding 1, and casting the result to `number_unsigned_t`.
- **Output**: The function returns a value of type `number_unsigned_t`, which is the positive representation of the input negative integer.


---
### ordered\_map<!-- {{#callable:ordered_map}} -->
Constructs an `ordered_map` using an initializer list and an optional allocator.
- **Inputs**:
    - `init`: An initializer list of `value_type` elements used to initialize the `ordered_map`.
    - `alloc`: An optional allocator of type `Allocator` used for memory management, defaulting to a default-constructed `Allocator`.
- **Control Flow**:
    - Calls the constructor of the base class `Container` with the provided initializer list and allocator.
- **Output**: This function does not return a value; it initializes an instance of `ordered_map`.


---
### emplace<!-- {{#callable:emplace}} -->
The `emplace` function inserts a new element into a JSON object or array, transforming a null value into an object if necessary.
- **Inputs**:
    - `Args&& ... args`: Variadic template arguments that represent the key-value pairs to be inserted into the JSON object.
- **Control Flow**:
    - The function first checks if the current object is neither null nor an object; if so, it throws a `type_error` exception.
    - If the object is null, it transforms it into an object by setting its type and value accordingly.
    - It then calls the `emplace` method on the internal object to add the new element, using perfect forwarding for the arguments.
    - The parent of the newly added element is set using [`set_parent`](#set_parent).
    - An iterator is created pointing to the newly added element, and the function prepares to return this iterator along with a boolean indicating success.
- **Output**: The function returns a pair consisting of an iterator pointing to the newly added element and a boolean indicating whether the insertion took place successfully.
- **Functions called**:
    - [`is_null`](#is_null)
    - [`is_object`](#is_object)
    - [`type_name`](#type_name)
    - [`set_parent`](#set_parent)
    - [`(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::begin`](#(anonymous)::NLOHMANN_JSON_NAMESPACE_BEGIN::iteration_proxy::begin)


---
### at<!-- {{#callable:at}} -->
Retrieves a constant reference to a JSON value at the specified pointer.
- **Inputs**:
    - `ptr`: A constant reference to a `nlohmann::json_pointer` object that specifies the location of the desired JSON value.
- **Control Flow**:
    - Calls the `get_checked` method on the `ptr` object, passing `this` as an argument to ensure the pointer is valid and to retrieve the corresponding JSON value.
- **Output**: Returns a constant reference to the JSON value located at the pointer specified by `ptr`.


---
### erase<!-- {{#callable:erase}} -->
The `erase` function removes an element from an array at a specified index.
- **Inputs**:
    - `idx`: The index of the element to be removed from the array.
- **Control Flow**:
    - The function first checks if the current object is an array using `is_array()`.
    - If it is an array, it checks if the provided index `idx` is within the valid range.
    - If the index is out of range, it throws an `out_of_range` exception.
    - If the index is valid, it erases the element at the specified index from the array.
    - If the object is not an array, it throws a `type_error` exception.
- **Output**: The function does not return a value; it modifies the internal state of the object by removing an element from the array.
- **Functions called**:
    - [`is_array`](#is_array)
    - [`namespace::integer_sequence::size`](#integer_sequencesize)
    - [`type_name`](#type_name)


---
### count<!-- {{#callable:count}} -->
Counts the occurrences of a specified key in a JSON object.
- **Inputs**:
    - `key`: The key whose occurrences are to be counted in the JSON object.
- **Control Flow**:
    - Checks if the current instance is a JSON object using the `is_object()` method.
    - If it is an object, it calls the `count` method on the internal object representation with the provided key.
    - If it is not an object, it returns 0.
- **Output**: Returns the number of occurrences of the specified key in the JSON object, or 0 if the instance is not an object.
- **Functions called**:
    - [`is_object`](#is_object)


---
### find<!-- {{#callable:find}} -->
The `find` function searches for a specified key in a JSON object and returns an iterator to the found element.
- **Inputs**:
    - `key`: The key to search for in the JSON object, which can be of any type that is usable as a basic JSON key.
- **Control Flow**:
    - The function first initializes an iterator `result` to the end of the container.
    - It checks if the current instance is an object using the `is_object()` method.
    - If it is an object, it attempts to find the key in the underlying data structure using the `find` method of the object iterator.
    - The result of the find operation is stored in `result.m_it.object_iterator`.
- **Output**: The function returns an iterator pointing to the found key-value pair in the JSON object, or to the end of the container if the key is not found.
- **Functions called**:
    - [`cend`](#cend)
    - [`is_object`](#is_object)


---
### insert<!-- {{#callable:insert}} -->
Inserts a range of elements into a JSON object from specified iterators.
- **Inputs**:
    - `first`: An iterator pointing to the beginning of the range of elements to be inserted.
    - `last`: An iterator pointing to the end of the range of elements to be inserted.
- **Control Flow**:
    - Checks if the current object is a JSON object; if not, throws a type error.
    - Validates that the `first` and `last` iterators belong to the same JSON object; if not, throws an invalid iterator error.
    - Ensures that the iterators point to valid objects; if not, throws an invalid iterator error.
    - If all checks pass, inserts the elements from the range defined by `first` to `last` into the current JSON object's internal data structure.
    - Calls `set_parents()` to update the parent references of the newly inserted elements.
- **Output**: The function does not return a value; it modifies the internal state of the JSON object by inserting new elements.
- **Functions called**:
    - [`is_object`](#is_object)
    - [`type_name`](#type_name)
    - [`set_parents`](#set_parents)


---
### parser<!-- {{#callable:parser}} -->
Creates and returns a `parser` object configured with the provided input adapter and optional parameters.
- **Inputs**:
    - `adapter`: An instance of `InputAdapterType` that serves as the input source for the parser.
    - `cb`: An optional callback function of type `detail::parser_callback_t<basic_json>` that can be invoked during parsing.
    - `allow_exceptions`: A boolean flag indicating whether exceptions should be allowed during parsing; defaults to true.
    - `ignore_comments`: A boolean flag indicating whether comments should be ignored during parsing; defaults to false.
- **Control Flow**:
    - The function begins by creating a new instance of `parser` using the provided `adapter` and optional parameters.
    - It uses `std::move` to efficiently transfer ownership of the `adapter` and `cb` arguments to the new `parser` instance.
    - The function then returns the newly created `parser` object.
- **Output**: Returns a `parser` object of type `::nlohmann::detail::parser<basic_json, InputAdapterType>` that is configured with the specified input adapter and options.


---
### get\_allocator<!-- {{#callable:get_allocator}} -->
Returns a default-constructed instance of the allocator type.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any conditional statements or loops.
    - It directly returns a new instance of `allocator_type`.
- **Output**: The output is a default-constructed object of type `allocator_type`, which is typically used for memory allocation in C++ standard library containers.


---
### meta<!-- {{#callable:meta}} -->
The `meta` function generates a JSON object containing metadata about the JSON library.
- **Inputs**: None
- **Control Flow**:
    - A `basic_json` object named `result` is initialized to store the metadata.
    - The copyright, name, and URL of the library are assigned to the `result` object.
    - The version information is constructed using the major, minor, and patch version numbers and stored in the `result` object.
    - The platform is determined using preprocessor directives and assigned to the `result` object.
    - The compiler information is also determined using preprocessor directives and stored in the `result` object, including the compiler family and version.
    - The C++ standard version used by the compiler is added to the `result` object.
    - Finally, the populated `result` object is returned.
- **Output**: The function returns a `basic_json` object containing structured metadata, including copyright, library name, URL, version, platform, compiler family, and C++ standard version.


---
### JSON\_CATCH<!-- {{#callable:at::JSON_CATCH}} -->
`JSON_CATCH` handles `std::out_of_range` exceptions by throwing a more informative exception.
- **Inputs**:
    - `std::out_of_range&`: A reference to an `std::out_of_range` exception that is caught.
- **Control Flow**:
    - The function catches an `std::out_of_range` exception.
    - It constructs a new exception message using `detail::concat` to provide context about the error.
    - The new exception is thrown using `JSON_THROW`, which likely handles JSON-specific error reporting.
- **Output**: The function does not return a value but throws a new exception that provides a detailed error message regarding the out-of-range access.


---
### set\_parents<!-- {{#callable:set_parents}} -->
Sets the `m_parent` member of a specified number of elements in a range to the current object.
- **Inputs**:
    - `it`: An iterator pointing to the beginning of the range where the `m_parent` member will be set.
    - `count_set_parents`: The number of elements in the range to which the `m_parent` member will be assigned.
- **Control Flow**:
    - If `JSON_DIAGNOSTICS` is defined, a loop iterates from 0 to `count_set_parents`, setting the `m_parent` member of each element in the range to the current object (`this`).
    - If `JSON_DIAGNOSTICS` is not defined, the `count_set_parents` parameter is ignored, and no action is taken.
- **Output**: Returns the original iterator `it` after potentially modifying the `m_parent` members of the specified range.


---
### set\_parent<!-- {{#callable:set_parent}} -->
The `set_parent` function assigns the current object as the parent of a specified reference while handling capacity checks and diagnostics.
- **Inputs**:
    - `j`: A reference to an object whose parent is to be set.
    - `old_capacity`: An optional parameter representing the previous capacity of the object, defaulting to an unknown size.
- **Control Flow**:
    - If diagnostics are enabled, the function first checks if the `old_capacity` is provided and if the current type is an array.
    - If the current array's capacity does not match `old_capacity`, it calls `set_parents()` to update all parent references and returns the reference `j`.
    - If the internal structure is an ordered map, it also calls `set_parents()` and returns `j`.
    - If diagnostics are not enabled, the function simply ignores the inputs and returns `j`.
- **Output**: The function returns the reference `j`, which now has its parent set to the current object.
- **Functions called**:
    - [`type`](#type)
    - [`set_parents`](#set_parents)


---
### basic\_json<!-- {{#callable:basic_json}} -->
The `basic_json` constructor moves the resources from another `basic_json` object while ensuring the integrity of the data.
- **Inputs**:
    - `other`: An rvalue reference to another `basic_json` object from which resources will be moved.
- **Control Flow**:
    - The constructor initializes the base class `json_base_class_t` using the forwarded `other` object.
    - It moves the `m_data` member from `other` to the new instance.
    - If `JSON_DIAGNOSTIC_POSITIONS` is defined, it also moves the `start_position` and `end_position` from `other`.
    - The function checks the validity of the `other` object by calling `assert_invariant` with `false` to ensure it is in a valid state.
    - It then invalidates the payload of `other` by setting its type to `null` and clearing its value.
    - If `JSON_DIAGNOSTIC_POSITIONS` is defined, it resets the `start_position` and `end_position` of `other` to `std::string::npos`.
    - Finally, it calls [`set_parents`](#set_parents) to update parent references and `assert_invariant` to ensure the new object is valid.
- **Output**: The constructor does not return a value but constructs a new `basic_json` object that has taken ownership of the resources from `other`, leaving `other` in a valid but empty state.
- **Functions called**:
    - [`set_parents`](#set_parents)


---
### binary<!-- {{#callable:binary}} -->
Creates a [`basic_json`](#basic_json) object representing binary data initialized with a given container and subtype.
- **Inputs**:
    - `init`: A rvalue reference to a container of binary data used to initialize the binary value.
    - `subtype`: A subtype identifier that specifies the type of binary data being represented.
- **Control Flow**:
    - A new [`basic_json`](#basic_json) object `res` is instantiated.
    - The type of `res` is set to `value_t::binary` to indicate that it holds binary data.
    - The binary data is constructed using the provided `init` and `subtype`, and assigned to `res.m_data.m_value`.
    - Finally, the `res` object is returned.
- **Output**: Returns a [`basic_json`](#basic_json) object that encapsulates the binary data initialized with the specified container and subtype.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### array<!-- {{#callable:array}} -->
Creates a [`basic_json`](#basic_json) object of type array initialized with an optional initializer list.
- **Inputs**:
    - `init`: An optional `initializer_list_t` that provides initial values for the array.
- **Control Flow**:
    - The function checks if an initializer list is provided; if not, it defaults to an empty list.
    - It then calls the constructor of [`basic_json`](#basic_json) with the initializer list, a boolean flag set to false, and the type `value_t::array`.
- **Output**: Returns a [`basic_json`](#basic_json) object representing an array initialized with the provided values.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### object<!-- {{#callable:object}} -->
Creates a [`basic_json`](#basic_json) object initialized with an optional initializer list.
- **Inputs**:
    - `init`: An optional `initializer_list_t` that allows for the initialization of the [`basic_json`](#basic_json) object with a list of values.
- **Control Flow**:
    - The function begins by checking if an initializer list is provided; if not, it defaults to an empty list.
    - It then calls the constructor of [`basic_json`](#basic_json) with the initializer list, a boolean value set to false, and a type identifier `value_t::object` to specify the type of JSON object being created.
- **Output**: Returns a [`basic_json`](#basic_json) object that is initialized based on the provided initializer list or as an empty object if no list is given.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### operator=<!-- {{#callable:operator=}} -->
The `operator=` function assigns the contents of one `basic_json` object to another using move semantics.
- **Inputs**:
    - `other`: A `basic_json` object that is passed by value, which will be used to assign its contents to the current object.
- **Control Flow**:
    - The function first checks the validity of the `other` object by calling `assert_invariant()`.
    - It then uses `std::swap` to exchange the type and value of the current object's data with that of `other`.
    - If `JSON_DIAGNOSTIC_POSITIONS` is defined, it also swaps the `start_position` and `end_position` attributes.
    - The base class's move assignment operator is called to handle any additional assignment logic.
    - Finally, it calls `set_parents()` to update parent references and asserts the invariant of the current object before returning a reference to itself.
- **Output**: Returns a reference to the current `basic_json` object after the assignment operation.
- **Functions called**:
    - [`swap`](#swap)
    - [`set_parents`](#set_parents)


---
### type<!-- {{#callable:type}} -->
Returns the type of the data stored in the object.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the member variable `m_data.m_type` to retrieve the type.
    - No conditional statements or loops are present, making the function straightforward.
- **Output**: Returns a value of type `value_t` representing the type of the data.


---
### is\_primitive<!-- {{#callable:is_primitive}} -->
Determines if the object is of a primitive type.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the object is null by calling `is_null()`.
    - It checks if the object is a string by calling `is_string()`.
    - It checks if the object is a boolean by calling `is_boolean()`.
    - It checks if the object is a number by calling `is_number()`.
    - It checks if the object is binary by calling `is_binary()`.
    - If any of these checks return true, the function returns true; otherwise, it returns false.
- **Output**: Returns a boolean value indicating whether the object is a primitive type (null, string, boolean, number, or binary).
- **Functions called**:
    - [`is_null`](#is_null)
    - [`is_string`](#is_string)
    - [`is_boolean`](#is_boolean)
    - [`is_number`](#is_number)
    - [`is_binary`](#is_binary)


---
### is\_structured<!-- {{#callable:is_structured}} -->
Determines if the current instance is either an array or an object.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the current instance is an array by calling the `is_array()` method.
    - If the instance is not an array, it then checks if it is an object by calling the `is_object()` method.
    - The function returns true if either `is_array()` or `is_object()` returns true; otherwise, it returns false.
- **Output**: Returns a boolean value indicating whether the current instance is structured as an array or an object.
- **Functions called**:
    - [`is_array`](#is_array)
    - [`is_object`](#is_object)


---
### is\_null<!-- {{#callable:is_null}} -->
Checks if the internal data type is null.
- **Inputs**: None
- **Control Flow**:
    - The function checks the value of `m_data.m_type` against `value_t::null`.
    - If `m_data.m_type` is equal to `value_t::null`, the function returns `true`; otherwise, it returns `false`.
- **Output**: Returns a boolean value indicating whether the internal data type is null.


---
### is\_boolean<!-- {{#callable:is_boolean}} -->
The `is_boolean` function checks if the type of the stored data is boolean.
- **Inputs**:
    - `this`: A constant reference to the current object instance, which contains the data to be checked.
- **Control Flow**:
    - The function accesses the member variable `m_data.m_type` of the current object instance.
    - It compares `m_data.m_type` to the enumerated value `value_t::boolean`.
    - The function returns `true` if the type is boolean, otherwise it returns `false`.
- **Output**: The function returns a boolean value indicating whether the type of the stored data is boolean.


---
### is\_number<!-- {{#callable:is_number}} -->
Determines if the current object represents a numeric value, either as an integer or a float.
- **Inputs**: None
- **Control Flow**:
    - Calls the `is_number_integer()` method to check if the object is an integer.
    - Calls the `is_number_float()` method to check if the object is a float.
    - Returns true if either of the above checks is true, otherwise returns false.
- **Output**: Returns a boolean value indicating whether the object is a numeric type (integer or float).
- **Functions called**:
    - [`is_number_integer`](#is_number_integer)
    - [`is_number_float`](#is_number_float)


---
### is\_number\_integer<!-- {{#callable:is_number_integer}} -->
Checks if the stored data type is an integer or an unsigned integer.
- **Inputs**: None
- **Control Flow**:
    - Evaluates the type of `m_data` to determine if it is either `number_integer` or `number_unsigned`.
    - Returns a boolean value based on the evaluation.
- **Output**: Returns `true` if the data type is an integer or unsigned integer, otherwise returns `false`.


---
### is\_number\_unsigned<!-- {{#callable:is_number_unsigned}} -->
The `is_number_unsigned` function checks if the type of the stored value is `number_unsigned`.
- **Inputs**: None
- **Control Flow**:
    - The function evaluates the member variable `m_data.m_type`.
    - It compares `m_data.m_type` to the enumeration value `value_t::number_unsigned`.
    - The function returns a boolean value based on the comparison result.
- **Output**: The function returns `true` if the type is `number_unsigned`, otherwise it returns `false`.


---
### is\_number\_float<!-- {{#callable:is_number_float}} -->
Checks if the type of the stored data is a floating-point number.
- **Inputs**: None
- **Control Flow**:
    - The function checks the member variable `m_data.m_type` against the enumeration `value_t::number_float`.
    - If the type matches, it returns `true`; otherwise, it returns `false`.
- **Output**: Returns a boolean value indicating whether the stored data is of type floating-point number.


---
### is\_object<!-- {{#callable:is_object}} -->
The `is_object` function checks if the current instance's type is an object.
- **Inputs**: None
- **Control Flow**:
    - The function evaluates the member variable `m_data.m_type`.
    - It compares `m_data.m_type` to the enumerated value `value_t::object`.
- **Output**: The function returns a boolean value indicating whether the type of the current instance is an object.


---
### is\_array<!-- {{#callable:is_array}} -->
The `is_array` function checks if the type of the data is an array.
- **Inputs**: None
- **Control Flow**:
    - The function evaluates the member variable `m_data.m_type`.
    - It compares `m_data.m_type` to the enumerated value `value_t::array`.
- **Output**: The function returns a boolean value indicating whether the type is an array.


---
### is\_string<!-- {{#callable:is_string}} -->
Checks if the current object's type is a string.
- **Inputs**: None
- **Control Flow**:
    - The function evaluates the member variable `m_data.m_type`.
    - It compares `m_data.m_type` to the enumerated value `value_t::string`.
- **Output**: Returns a boolean value indicating whether the type is a string.


---
### is\_binary<!-- {{#callable:is_binary}} -->
The `is_binary` function checks if the type of the data is binary.
- **Inputs**: None
- **Control Flow**:
    - The function evaluates the condition `m_data.m_type == value_t::binary`.
    - It returns a boolean value based on the evaluation of the condition.
- **Output**: The output is a boolean value indicating whether the data type is binary.


---
### is\_discarded<!-- {{#callable:is_discarded}} -->
The `is_discarded` function checks if the current object's type is marked as discarded.
- **Inputs**: None
- **Control Flow**:
    - The function evaluates the member variable `m_data.m_type`.
    - It compares `m_data.m_type` to the enumerated value `value_t::discarded`.
- **Output**: The function returns a boolean value indicating whether the type is `discarded`.


---
### get\_impl\_ptr<!-- {{#callable:get_impl_ptr}} -->
Returns a pointer to the binary data if the current object is binary.
- **Inputs**:
    - `unused`: A pointer to a `binary_t` object that is not used in the function.
- **Control Flow**:
    - The function checks if the current object is binary by calling the `is_binary()` method.
    - If `is_binary()` returns true, it returns the pointer to the binary data stored in `m_data.m_value.binary`.
    - If `is_binary()` returns false, it returns a null pointer.
- **Output**: A pointer to the binary data if the object is binary; otherwise, a null pointer.
- **Functions called**:
    - [`is_binary`](#is_binary)


---
### get\_ref\_impl<!-- {{#callable:get_ref_impl}} -->
`get_ref_impl` retrieves a reference of a specified type from an object if the pointer is valid.
- **Inputs**:
    - `obj`: A reference to an object of type `ThisType` from which a pointer is obtained.
- **Control Flow**:
    - The function first calls `get_ptr<>` on the `obj` to obtain a pointer of type `ReferenceType`.
    - It checks if the obtained pointer is not null using `JSON_HEDLEY_LIKELY` for optimization.
    - If the pointer is valid, it dereferences the pointer and returns the reference.
    - If the pointer is null, it throws a `type_error` exception with a message indicating the type mismatch.
- **Output**: Returns a reference of type `ReferenceType` if the pointer is valid; otherwise, it throws an exception.


---
### get\_ptr<!-- {{#callable:get_ptr}} -->
The `get_ptr` function retrieves a pointer from a `basic_json_t` object, ensuring that the pointer type is a constant pointer.
- **Inputs**: None
- **Control Flow**:
    - The function uses template specialization to ensure that the `PointerType` is a pointer and points to a constant type.
    - It calls the [`get_impl_ptr`](#get_impl_ptr) method of the `basic_json_t` class, passing a null pointer of the specified `PointerType`.
- **Output**: The output is the result of the [`get_impl_ptr`](#get_impl_ptr) method, which is expected to return a pointer type based on the implementation of `basic_json_t`.
- **Functions called**:
    - [`get_impl_ptr`](#get_impl_ptr)


---
### noexcept<!-- {{#callable:noexcept}} -->
The `get_impl` function retrieves a `ValueType` object by deserializing JSON data.
- **Inputs**:
    - `detail::priority_tag<0>`: A tag used for function overloading, indicating the priority of this implementation; it is unused in the function body.
- **Control Flow**:
    - The function begins by creating a default instance of `ValueType` named `ret`.
    - It then calls the static method `from_json` of `JSONSerializer<ValueType>`, passing the current object (assumed to be in JSON format) and the `ret` instance to populate it with deserialized data.
    - Finally, the populated `ret` instance is returned.
- **Output**: The function outputs a `ValueType` object that has been populated with data deserialized from JSON.


---
### get\_impl<!-- {{#callable:get_impl}} -->
The `get_impl` function is a template method that retrieves a pointer of a specified type from a `basic_json_t` object.
- **Inputs**:
    - `PointerType`: A template parameter representing the type of pointer to be retrieved.
    - `detail::priority_tag<4>`: A tag used to enable this function only for pointer types, which is unused in the function body.
- **Control Flow**:
    - The function checks if `PointerType` is a pointer type using SFINAE (Substitution Failure Is Not An Error) with `enable_if_t`.
    - If the condition is satisfied, it proceeds to call the `get_ptr` method of the `basic_json_t` class, passing the `PointerType` as a template argument.
    - The function does not perform any additional logic or checks and directly returns the result of the `get_ptr` call.
- **Output**: The output is the result of the `get_ptr<PointerType>()` call, which is expected to be a pointer of the specified type retrieved from the `basic_json_t` object.


---
### get\_to<!-- {{#callable:get_to}} -->
The `get_to` function deserializes a JSON object into a provided array.
- **Inputs**:
    - `v`: A reference to an array of type `T` with size `N`, which will be populated with data from the JSON object.
- **Control Flow**:
    - The function first calls `JSONSerializer<Array>::from_json` to deserialize the JSON data into the provided array `v`.
    - It uses `std::declval` to ensure that the JSON deserialization is valid and noexcept.
    - Finally, it returns the populated array `v`.
- **Output**: The function returns the reference to the array `v` after it has been populated with data from the JSON object.
- **Functions called**:
    - [`noexcept`](#noexcept)


---
### get\_ref<!-- {{#callable:get_ref}} -->
The `get_ref` function is a template method that retrieves a reference of a specified type from the current object, ensuring that the type is a const reference.
- **Inputs**: None
- **Control Flow**:
    - The function uses SFINAE (Substitution Failure Is Not An Error) to ensure that it can only be instantiated if `ReferenceType` is a const reference type.
    - If the conditions are met, it delegates the call to another function `get_ref_impl`, passing the current object as an argument.
- **Output**: The function returns a reference of type `ReferenceType` obtained from the `get_ref_impl` function.


---
### get\_binary<!-- {{#callable:get_binary}} -->
Retrieves a reference to a `binary_t` object if the current type is binary.
- **Inputs**: None
- **Control Flow**:
    - Checks if the current object type is binary using the `is_binary()` method.
    - If the type is not binary, throws a `type_error` exception with a detailed message.
    - If the type is binary, retrieves and returns a reference to the `binary_t` object using `get_ptr()`.
- **Output**: Returns a constant reference to a `binary_t` object.
- **Functions called**:
    - [`is_binary`](#is_binary)
    - [`type_name`](#type_name)


---
### JSON\_INTERNAL\_CATCH<!-- {{#callable:value::JSON_INTERNAL_CATCH}} -->
The `JSON_INTERNAL_CATCH` function handles an `out_of_range` exception by returning a forwarded default value.
- **Inputs**:
    - `default_value`: The default value to be returned when an `out_of_range` exception is caught.
- **Control Flow**:
    - The function is designed to catch an `out_of_range` exception.
    - Upon catching the exception, it immediately returns the `default_value` using `std::forward`.
- **Output**: The function outputs the forwarded `default_value` of type `ValueType`.


---
### front<!-- {{#callable:front}} -->
The `front` function returns a constant reference to the first element of a container.
- **Inputs**: None
- **Control Flow**:
    - The function calls `cbegin()` to obtain an iterator pointing to the beginning of the container.
    - The dereference operator `*` is used to access the value at the iterator's position, which is the first element.
- **Output**: The output is a constant reference to the first element of the container, allowing read-only access to that element.
- **Functions called**:
    - [`cbegin`](#cbegin)


---
### erase\_internal<!-- {{#callable:erase_internal}} -->
The `erase_internal` function removes an entry from a JSON object based on a specified key.
- **Inputs**:
    - `key`: The key of the entry to be removed from the JSON object, which can be of any type that is not supported by the `erase_with_key_type` function.
- **Control Flow**:
    - The function first checks if the current JSON instance is an object; if not, it throws a `type_error` exception.
    - It then attempts to find the specified `key` in the object's internal data structure.
    - If the key is found, the corresponding entry is erased from the object, and the function returns 1.
    - If the key is not found, the function returns 0.
- **Output**: The function returns the number of entries erased, which is either 1 if the key was found and removed, or 0 if the key was not present.
- **Functions called**:
    - [`is_object`](#is_object)
    - [`type_name`](#type_name)


---
### begin<!-- {{#callable:begin}} -->
Returns a constant iterator to the beginning of the container.
- **Inputs**: None
- **Control Flow**:
    - Calls the `cbegin()` method to retrieve a constant iterator.
    - Returns the result of the `cbegin()` method.
- **Output**: A constant iterator pointing to the first element of the container.
- **Functions called**:
    - [`cbegin`](#cbegin)


---
### cbegin<!-- {{#callable:cbegin}} -->
Returns a `const_iterator` pointing to the beginning of the container.
- **Inputs**:
    - `this`: A constant reference to the current object, representing the container.
- **Control Flow**:
    - Creates a `const_iterator` object named `result` initialized with the current container.
    - Calls the `set_begin()` method on the `result` iterator to set its position to the beginning of the container.
    - Returns the `result` iterator.
- **Output**: A `const_iterator` that points to the first element of the container.


---
### end<!-- {{#callable:end}} -->
Returns a constant iterator to the end of the container.
- **Inputs**: None
- **Control Flow**:
    - Calls the `cend()` method to retrieve a constant iterator.
    - Returns the result of the `cend()` method.
- **Output**: A constant iterator pointing to the past-the-end element of the container.
- **Functions called**:
    - [`cend`](#cend)


---
### cend<!-- {{#callable:cend}} -->
Returns a `const_iterator` pointing to the end of the container.
- **Inputs**:
    - `this`: A constant reference to the current object, representing the container.
- **Control Flow**:
    - Creates a `const_iterator` object initialized with the current container.
    - Calls the `set_end()` method on the `const_iterator` to set its position to the end of the container.
    - Returns the configured `const_iterator`.
- **Output**: A `const_iterator` that points to the end of the container, allowing for iteration over the container in a read-only manner.


---
### rbegin<!-- {{#callable:rbegin}} -->
Returns a constant reverse iterator to the beginning of the container.
- **Inputs**: None
- **Control Flow**:
    - The function directly calls another function `crbegin()`.
    - No conditional statements or loops are present in this function.
- **Output**: Returns a `const_reverse_iterator` that points to the last element of the container, allowing for reverse iteration.
- **Functions called**:
    - [`crbegin`](#crbegin)


---
### rend<!-- {{#callable:rend}} -->
Returns a constant reverse iterator to the end of the range.
- **Inputs**: None
- **Control Flow**:
    - Calls the `crend()` method to obtain a constant reverse iterator.
- **Output**: A constant reverse iterator pointing to the theoretical element preceding the first element of the range.
- **Functions called**:
    - [`crend`](#crend)


---
### crbegin<!-- {{#callable:crbegin}} -->
Returns a `const_reverse_iterator` pointing to the end of the container.
- **Inputs**: None
- **Control Flow**:
    - The function directly calls `cend()` to obtain an iterator to the end of the container.
    - It then constructs and returns a `const_reverse_iterator` initialized with the result of `cend()`.
- **Output**: A `const_reverse_iterator` that allows iteration over the container in reverse order, starting from the end.
- **Functions called**:
    - [`cend`](#cend)


---
### crend<!-- {{#callable:crend}} -->
The `crend` function returns a `const_reverse_iterator` pointing to the theoretical element preceding the first element of a constant range.
- **Inputs**: None
- **Control Flow**:
    - The function calls `cbegin()` to obtain a `const_iterator` pointing to the first element of the range.
    - It then constructs a `const_reverse_iterator` using the iterator obtained from `cbegin()`.
- **Output**: The output is a `const_reverse_iterator` that allows iteration over the elements of the container in reverse order, starting from the end.
- **Functions called**:
    - [`cbegin`](#cbegin)


---
### iterator\_wrapper<!-- {{#callable:iterator_wrapper}} -->
The `iterator_wrapper` function returns an `iteration_proxy` object initialized with the items of the provided reference.
- **Inputs**:
    - `ref`: A constant reference to an object of type that contains an `items()` method, which is expected to return an iterable collection.
- **Control Flow**:
    - The function directly calls the `items()` method on the `ref` object.
    - The result of the `items()` method call is returned as an `iteration_proxy`.
- **Output**: The output is an `iteration_proxy` object that allows iteration over the items contained in the `ref` object.


---
### items<!-- {{#callable:items}} -->
Returns an `iteration_proxy` object initialized with the current instance.
- **Inputs**:
    - `this`: A constant reference to the current instance of the class, used to initialize the `iteration_proxy`.
- **Control Flow**:
    - Directly returns an `iteration_proxy` object created with the current instance of the class.
- **Output**: An `iteration_proxy<const_iterator>` object that allows iteration over the elements of the current instance.


---
### size<!-- {{#callable:size}} -->
The `size` function returns the size of the stored value based on its type.
- **Inputs**: None
- **Control Flow**:
    - The function begins by checking the type of the stored value using a `switch` statement on `m_data.m_type`.
    - If the type is `null`, it returns 0, indicating that null values are considered empty.
    - If the type is `array`, it calls the `size` method of the `array_t` class to get the size of the array.
    - If the type is `object`, it calls the `size` method of the `object_t` class to get the size of the object.
    - For types such as `string`, `boolean`, `number_integer`, `number_unsigned`, `number_float`, `binary`, and `discarded`, it returns 1, indicating that these types are considered to have a size of 1.
- **Output**: The output is a `size_type` value representing the size of the stored value, which can be 0, the size of an array or object, or 1 for other types.


---
### max\_size<!-- {{#callable:max_size}} -->
Returns the maximum size of the contained value based on its type.
- **Inputs**: None
- **Control Flow**:
    - The function checks the type of the contained value using a switch statement.
    - If the type is `array`, it delegates the call to the `max_size()` method of the `array_t` class.
    - If the type is `object`, it delegates the call to the `max_size()` method of the `object_t` class.
    - For all other types (`null`, `string`, `boolean`, `number_integer`, `number_unsigned`, `number_float`, `binary`, `discarded`), it returns the current size of the value using the `size()` method.
- **Output**: Returns a `size_type` representing the maximum size of the value, which varies depending on the type of the contained value.
- **Functions called**:
    - [`namespace::integer_sequence::size`](#integer_sequencesize)


---
### clear<!-- {{#callable:clear}} -->
The `clear` function resets the value of a member variable based on its type.
- **Inputs**: None
- **Control Flow**:
    - The function begins by checking the type of `m_data.m_type` using a `switch` statement.
    - For each case corresponding to a specific type (integer, unsigned, float, boolean, string, binary, array, object), it resets the associated value to its default state.
    - If the type is `null`, `discarded`, or any unrecognized type, no action is taken.
- **Output**: The function does not return a value; it modifies the internal state of the object by clearing or resetting the value of `m_data`.


---
### emplace\_back<!-- {{#callable:emplace_back}} -->
The `emplace_back` function adds a new element to a JSON array or transforms a null object into an array before adding the element.
- **Inputs**:
    - `Args&& ... args`: A variadic template parameter that allows passing any number of arguments to construct the new element in place.
- **Control Flow**:
    - The function first checks if the current object is neither null nor an array; if so, it throws a `type_error` exception.
    - If the object is null, it transforms it into an array by setting the appropriate type and value.
    - The function then retrieves the current capacity of the array before adding the new element using perfect forwarding.
    - Finally, it returns a reference to the newly added element while also setting its parent.
- **Output**: The function returns a reference to the newly added element in the array.
- **Functions called**:
    - [`is_null`](#is_null)
    - [`is_array`](#is_array)
    - [`type_name`](#type_name)
    - [`set_parent`](#set_parent)


---
### insert\_iterator<!-- {{#callable:insert_iterator}} -->
Inserts one or more elements at a specified position in an array and returns an iterator to the newly inserted elements.
- **Inputs**:
    - `pos`: A `const_iterator` representing the position in the array where new elements will be inserted.
    - `args`: A variadic template parameter pack that allows passing one or more elements to be inserted into the array.
- **Control Flow**:
    - The function begins by creating a new `iterator` object named `result` initialized with the current object context.
    - It asserts that the array within `m_data.m_value` is not null using `JSON_ASSERT`.
    - The position for insertion is calculated using `std::distance` to find the index of `pos` in the array.
    - The elements specified in `args` are inserted into the array at the calculated position using `insert`.
    - The iterator `result` is updated to point to the beginning of the newly inserted elements.
    - The [`set_parents`](#set_parents) function is called to update any necessary parent references after the insertion.
    - Finally, the updated `result` iterator is returned.
- **Output**: Returns an `iterator` pointing to the first of the newly inserted elements in the array.
- **Functions called**:
    - [`set_parents`](#set_parents)


---
### update<!-- {{#callable:update}} -->
Updates the current JSON object with key-value pairs from a specified range of iterators.
- **Inputs**:
    - `first`: An iterator pointing to the beginning of the range of key-value pairs to be updated.
    - `last`: An iterator pointing to the end of the range of key-value pairs to be updated.
    - `merge_objects`: A boolean flag indicating whether to merge objects if a key already exists in the current object.
- **Control Flow**:
    - Checks if the current object is null and initializes it as an empty object if true.
    - Throws a type error if the current object is not of type object.
    - Validates that the provided iterators belong to the same JSON object.
    - Throws an invalid iterator error if the iterators do not fit.
    - Checks if the first iterator's object is indeed an object, throwing a type error if not.
    - Iterates over the range defined by the iterators, updating or merging key-value pairs into the current object.
- **Output**: The function does not return a value; it modifies the current JSON object in place.
- **Functions called**:
    - [`is_null`](#is_null)
    - [`is_object`](#is_object)
    - [`type_name`](#type_name)


---
### swap<!-- {{#callable:namespace::swap}} -->
Swaps the contents of two `nlohmann::NLOHMANN_BASIC_JSON_TPL` objects.
- **Inputs**:
    - `j1`: The first `nlohmann::NLOHMANN_BASIC_JSON_TPL` object whose contents will be swapped.
    - `j2`: The second `nlohmann::NLOHMANN_BASIC_JSON_TPL` object whose contents will be swapped with the first.
- **Control Flow**:
    - The function calls the `swap` method of the first object `j1`, passing the second object `j2` as an argument.
    - The actual swapping of contents is handled by the `swap` method of the `nlohmann::NLOHMANN_BASIC_JSON_TPL` class.
- **Output**: This function does not return a value; it modifies the two input objects in place by swapping their contents.


---
### if<!-- {{#callable:if}} -->
This function checks if two values are unordered and returns a specific result if they are.
- **Inputs**:
    - `lhs`: The left-hand side value to be compared.
    - `rhs`: The right-hand side value to be compared.
- **Control Flow**:
    - The function first checks if the condition `compares_unordered(lhs, rhs)` is true.
    - If true, it executes the return statement with `unordered_result`.
- **Output**: The function outputs `unordered_result` when the two values are determined to be unordered.


---
### compares\_unordered<!-- {{#callable:compares_unordered}} -->
This function delegates the comparison of two objects for equality, allowing for an optional inverse comparison.
- **Inputs**:
    - `rhs`: The object of the same type as the current instance to compare against.
    - `inverse`: A boolean flag indicating whether to perform the comparison in inverse mode; defaults to false.
- **Control Flow**:
    - The function calls another overloaded version of `compares_unordered`, passing the current object (`*this`), the `rhs` object, and the `inverse` flag.
    - No additional logic is present in this function, as it solely serves as a wrapper to facilitate the comparison.
- **Output**: Returns a boolean value indicating whether the two objects are considered equal based on the comparison logic defined in the called `compares_unordered` function.


---
### parse<!-- {{#callable:parse}} -->
Parses input data using a parser and returns a `basic_json` object.
- **Inputs**:
    - `i`: An rvalue reference to a `detail::span_input_adapter` that provides the input data to be parsed.
    - `cb`: An optional callback function of type `parser_callback_t` that can be used during parsing.
    - `allow_exceptions`: A boolean flag indicating whether exceptions should be allowed during parsing.
    - `ignore_comments`: A boolean flag indicating whether comments in the input should be ignored.
- **Control Flow**:
    - The function initializes a `basic_json` object named `result` to store the parsed output.
    - It creates a [`parser`](#parserparser) object using the input adapter and the provided callback, along with the flags for exceptions and comments.
    - The `parse` method of the [`parser`](#parserparser) object is called with `true` to indicate that parsing should begin, and the `result` object is passed to store the parsed data.
    - Finally, the function returns the `result` object containing the parsed JSON data.
- **Output**: Returns a `basic_json` object that contains the parsed representation of the input data.
- **Functions called**:
    - [`namespace::parser::parser`](#parserparser)


---
### accept<!-- {{#callable:accept}} -->
The `accept` function processes an input adapter and determines if it meets certain acceptance criteria.
- **Inputs**:
    - `i`: A `detail::span_input_adapter` object that provides access to the input data.
    - `ignore_comments`: A boolean flag indicating whether to ignore comments during parsing, defaulting to false.
- **Control Flow**:
    - The function calls `i.get()` to retrieve the underlying input data from the `span_input_adapter`.
    - It then creates a [`parser`](#parserparser) object with the retrieved input, passing `nullptr` for an unspecified parameter, `false` for another parameter, and the `ignore_comments` flag.
    - Finally, it invokes the `accept` method on the [`parser`](#parserparser) object with `true` as an argument, returning the result.
- **Output**: The function returns a boolean value indicating whether the input data was accepted by the parser.
- **Functions called**:
    - [`namespace::parser::parser`](#parserparser)


---
### sax\_parse<!-- {{#callable:sax_parse}} -->
Parses input data using SAX based on the specified format, either JSON or binary.
- **Inputs**:
    - `i`: A `detail::span_input_adapter` object that provides access to the input data.
    - `sax`: A pointer to a `SAX` object that will handle the parsed data.
    - `format`: An optional `input_format_t` enum value indicating the format of the input data, defaulting to JSON.
    - `strict`: A boolean flag indicating whether to enforce strict parsing rules, defaulting to true.
    - `ignore_comments`: A boolean flag indicating whether to ignore comments in the input, defaulting to false.
- **Control Flow**:
    - The function retrieves the input data from the `span_input_adapter` using the `get()` method.
    - It checks the `format` argument to determine whether to parse the input as JSON or binary.
    - If the format is JSON, it creates a [`parser`](#parserparser) object and calls its `sax_parse` method with the provided `sax` and `strict` parameters.
    - If the format is binary, it creates a `detail::binary_reader` object and calls its `sax_parse` method with the same parameters.
- **Output**: Returns a boolean indicating the success or failure of the parsing operation.
- **Functions called**:
    - [`namespace::parser::parser`](#parserparser)


---
### type\_name<!-- {{#callable:type_name}} -->
The `type_name` function returns a string representation of the type of data stored in the object.
- **Inputs**: None
- **Control Flow**:
    - The function uses a `switch` statement to evaluate the value of `m_data.m_type`.
    - For each case corresponding to a specific `value_t` type, it returns a string literal that describes the type.
    - If none of the specified cases match, it defaults to returning 'number'.
- **Output**: The function outputs a constant character pointer to a string that indicates the type of the data, such as 'null', 'object', 'array', 'string', 'boolean', 'binary', 'discarded', or 'number'.


---
### end\_pos<!-- {{#callable:end_pos}} -->
Returns the value of the member variable `end_position`.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the member variable `end_position` without any conditions or loops.
- **Output**: The output is a `std::size_t` representing the value of `end_position`.


---
### to\_cbor<!-- {{#callable:to_cbor}} -->
Converts a `basic_json` object to CBOR format using a binary writer.
- **Inputs**:
    - `j`: A constant reference to a `basic_json` object that contains the data to be converted.
    - `o`: An output adapter of type `detail::output_adapter<char>` that specifies where the CBOR data will be written.
- **Control Flow**:
    - The function initializes a `binary_writer<char>` with the provided output adapter `o`.
    - It then calls the `write_cbor` method of the `binary_writer` instance, passing the `basic_json` object `j` to perform the conversion.
- **Output**: The function does not return a value; instead, it writes the CBOR representation of the `basic_json` object directly to the output adapter.


---
### to\_msgpack<!-- {{#callable:to_msgpack}} -->
Converts a `basic_json` object to MessagePack format using a specified output adapter.
- **Inputs**:
    - `j`: A constant reference to a `basic_json` object that contains the data to be converted.
    - `o`: An output adapter of type `detail::output_adapter<char>` that specifies where the MessagePack data will be written.
- **Control Flow**:
    - The function calls the `write_msgpack` method of a `binary_writer` instance, passing the `basic_json` object and the output adapter.
    - The `binary_writer` is instantiated with the output adapter, which allows it to write the serialized data directly to the specified output.
- **Output**: The function does not return a value; instead, it writes the serialized MessagePack data directly to the output adapter provided.


---
### to\_ubjson<!-- {{#callable:to_ubjson}} -->
Converts a `basic_json` object to UBJSON format using a specified output adapter.
- **Inputs**:
    - `j`: A constant reference to a `basic_json` object that contains the data to be converted.
    - `o`: An output adapter of type `detail::output_adapter<char>` that specifies where the UBJSON data will be written.
    - `use_size`: A boolean flag indicating whether to include size information in the output.
    - `use_type`: A boolean flag indicating whether to include type information in the output.
- **Control Flow**:
    - The function calls the `write_ubjson` method of a `binary_writer<char>` instance, passing the input parameters.
    - The `write_ubjson` method handles the actual conversion of the `basic_json` object to UBJSON format.
- **Output**: The function does not return a value; instead, it writes the UBJSON representation of the `basic_json` object to the specified output adapter.


---
### to\_bjdata<!-- {{#callable:to_bjdata}} -->
Converts a `basic_json` object to BJData format using a specified output adapter.
- **Inputs**:
    - `j`: A constant reference to a `basic_json` object that represents the JSON data to be converted.
    - `o`: An output adapter of type `detail::output_adapter<char>` that handles the output of the converted data.
    - `use_size`: A boolean flag indicating whether to include size information in the output.
    - `use_type`: A boolean flag indicating whether to include type information in the output.
    - `version`: An enumeration value of type `bjdata_version_t` that specifies the version of BJData format to use.
- **Control Flow**:
    - The function calls `binary_writer<char>(o)` to create a binary writer instance using the provided output adapter.
    - It then invokes the `write_ubjson` method on the binary writer, passing the JSON object and the specified flags and version.
- **Output**: The function does not return a value; instead, it writes the converted BJData directly to the output adapter.


---
### to\_bson<!-- {{#callable:to_bson}} -->
Converts a `basic_json` object to BSON format using a specified output adapter.
- **Inputs**:
    - `j`: A constant reference to a `basic_json` object that contains the data to be converted.
    - `o`: An output adapter of type `detail::output_adapter<char>` that specifies where the BSON data will be written.
- **Control Flow**:
    - The function calls `binary_writer<char>(o)` to create a binary writer instance that is associated with the provided output adapter.
    - It then invokes the `write_bson` method on the binary writer, passing the `basic_json` object `j` to perform the conversion to BSON format.
- **Output**: The function does not return a value; instead, it writes the BSON representation of the `basic_json` object directly to the output adapter.


---
### from\_cbor<!-- {{#callable:from_cbor}} -->
Converts a CBOR encoded input into a [`basic_json`](#basic_json) object.
- **Inputs**:
    - `i`: A `detail::span_input_adapter` that provides the CBOR encoded input data.
    - `strict`: A boolean flag indicating whether to enforce strict parsing rules.
    - `allow_exceptions`: A boolean flag that determines if exceptions should be allowed during parsing.
    - `tag_handler`: A handler for processing CBOR tags, defaulting to an error handler.
- **Control Flow**:
    - The function initializes a [`basic_json`](#basic_json) object named `result` to store the parsed output.
    - It retrieves the underlying input data from the `span_input_adapter` using `i.get()`.
    - A `detail::json_sax_dom_parser` is instantiated with `result` and the `allow_exceptions` flag.
    - The `binary_reader` is created with the input data and is used to parse the CBOR input using the `sax_parse` method.
    - If parsing is successful, the `result` is returned; otherwise, a discarded [`basic_json`](#basic_json) object is returned.
- **Output**: Returns a [`basic_json`](#basic_json) object containing the parsed data or a discarded object if parsing fails.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### from\_msgpack<!-- {{#callable:from_msgpack}} -->
Converts a MessagePack binary format input into a [`basic_json`](#basic_json) object.
- **Inputs**:
    - `i`: A `detail::span_input_adapter` that provides the input data in MessagePack format.
    - `strict`: A boolean flag indicating whether to enforce strict parsing rules.
    - `allow_exceptions`: A boolean flag that determines if exceptions are allowed during parsing.
- **Control Flow**:
    - The function initializes a [`basic_json`](#basic_json) object named `result` to store the parsed output.
    - It retrieves the underlying input data from the `span_input_adapter` using `i.get()`.
    - A `detail::json_sax_dom_parser` is instantiated with `result` and the `allow_exceptions` flag to handle the parsing process.
    - The `binary_reader` is created with the input data and is configured to parse the data in MessagePack format using the `sax_parse` method.
    - The result of the parsing operation is stored in the boolean variable `res`.
    - If parsing is successful (`res` is true`), the function returns the populated `result`; otherwise, it returns a discarded [`basic_json`](#basic_json) object.
- **Output**: Returns a [`basic_json`](#basic_json) object containing the parsed data from the MessagePack input, or a discarded [`basic_json`](#basic_json) object if parsing fails.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### from\_ubjson<!-- {{#callable:from_ubjson}} -->
Converts input data from UBJSON format to a [`basic_json`](#basic_json) object.
- **Inputs**:
    - `i`: A `detail::span_input_adapter` object that provides the input data in UBJSON format.
    - `strict`: A boolean flag indicating whether to enforce strict parsing rules (default is true).
    - `allow_exceptions`: A boolean flag that determines if exceptions should be allowed during parsing (default is true).
- **Control Flow**:
    - The function initializes a [`basic_json`](#basic_json) object named `result` to store the parsed output.
    - It retrieves the input data from the `span_input_adapter` using the `get()` method.
    - A `detail::json_sax_dom_parser` is instantiated with the `result` and the `allow_exceptions` flag.
    - The function then creates a `binary_reader` object to parse the input data in UBJSON format, invoking the `sax_parse` method on it.
    - The result of the parsing operation is checked; if successful, the `result` is returned, otherwise a discarded [`basic_json`](#basic_json) object is returned.
- **Output**: Returns a [`basic_json`](#basic_json) object containing the parsed data from the UBJSON input, or a discarded [`basic_json`](#basic_json) object if parsing fails.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### from\_bjdata<!-- {{#callable:from_bjdata}} -->
Converts a range of binary JSON data into a [`basic_json`](#basic_json) object.
- **Inputs**:
    - `first`: An iterator pointing to the beginning of the binary JSON data.
    - `last`: An iterator pointing to the end of the binary JSON data.
    - `strict`: A boolean flag indicating whether to enforce strict parsing rules.
    - `allow_exceptions`: A boolean flag indicating whether to allow exceptions during parsing.
- **Control Flow**:
    - Initializes a [`basic_json`](#basic_json) object named `result` to store the parsed output.
    - Creates an input adapter from the provided iterators to facilitate reading the binary JSON data.
    - Instantiates a SAX parser (`json_sax_dom_parser`) with the `result` object and the exception handling flag.
    - Calls the `sax_parse` method of a `binary_reader` to parse the binary JSON data using the input adapter and the SAX parser.
    - Checks the result of the parsing operation; if successful, returns the populated `result`, otherwise returns a discarded [`basic_json`](#basic_json) object.
- **Output**: Returns a [`basic_json`](#basic_json) object containing the parsed data if successful, or a discarded [`basic_json`](#basic_json) object if parsing fails.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### from\_bson<!-- {{#callable:from_bson}} -->
Converts BSON data into a [`basic_json`](#basic_json) object using a SAX parser.
- **Inputs**:
    - `i`: A `detail::span_input_adapter` object that provides the BSON data to be parsed.
    - `strict`: A boolean flag indicating whether to enforce strict parsing rules (default is true).
    - `allow_exceptions`: A boolean flag that determines if exceptions are allowed during parsing (default is true).
- **Control Flow**:
    - The function initializes a [`basic_json`](#basic_json) object named `result` to store the parsed output.
    - It retrieves the underlying input data from the `span_input_adapter` using `i.get()`.
    - A SAX parser (`detail::json_sax_dom_parser`) is instantiated with the `result` and the `allow_exceptions` flag.
    - The function then attempts to parse the BSON data using `binary_reader` and the SAX parser, passing the strictness flag.
    - If the parsing is successful, the `result` is returned; otherwise, a discarded [`basic_json`](#basic_json) object is returned.
- **Output**: Returns a [`basic_json`](#basic_json) object containing the parsed BSON data, or a discarded [`basic_json`](#basic_json) if parsing fails.
- **Functions called**:
    - [`basic_json`](#basic_json)


---
### flatten<!-- {{#callable:flatten}} -->
The `flatten` function creates a flattened representation of a JSON object.
- **Inputs**: None
- **Control Flow**:
    - A new `basic_json` object named `result` is initialized with the type `object`.
    - The `json_pointer::flatten` function is called with an empty string as the first argument, the current object (`*this`) as the second argument, and `result` as the third argument to perform the flattening operation.
    - The flattened `result` is returned.
- **Output**: The output is a `basic_json` object that contains the flattened representation of the original JSON object.


---
### unflatten<!-- {{#callable:unflatten}} -->
The `unflatten` function converts a flattened JSON object back into its original hierarchical structure.
- **Inputs**:
    - `this`: A constant reference to the current instance of the class, which is expected to be a flattened JSON object.
- **Control Flow**:
    - The function calls the static method `unflatten` from the `json_pointer` class, passing the current instance (`*this`) as an argument.
    - The result of the `unflatten` method is returned directly.
- **Output**: The output is a `basic_json` object that represents the original hierarchical structure of the JSON data.


---
### patch\_inplace<!-- {{#callable:patch_inplace}} -->
Applies a JSON Patch to the current JSON object, modifying it in place based on specified operations.
- **Inputs**:
    - `json_patch`: A `basic_json` object representing an array of JSON Patch operations to be applied.
- **Control Flow**:
    - Checks if the `json_patch` is an array; throws an error if not.
    - Iterates over each operation in the `json_patch` array.
    - For each operation, retrieves the operation type and the target path.
    - Based on the operation type (add, remove, replace, move, copy, test), calls the corresponding helper function to perform the operation.
    - Handles errors for invalid operations and checks for the existence of paths as required by the JSON Patch specification.
- **Output**: The function modifies the current JSON object in place according to the operations defined in the `json_patch`, with no return value.
- **Functions called**:
    - [`namespace::primitive_iterator_t::get_value`](#primitive_iterator_tget_value)


---
### patch<!-- {{#callable:patch}} -->
The `patch` function applies a JSON patch to the current JSON object and returns a new modified JSON object.
- **Inputs**:
    - `json_patch`: A `basic_json` object representing the JSON patch to be applied to the current JSON object.
- **Control Flow**:
    - The function begins by creating a copy of the current JSON object, referred to as `result`.
    - It then calls the `patch_inplace` method on `result`, passing the `json_patch` to modify `result` directly.
    - Finally, the modified `result` is returned.
- **Output**: The output is a `basic_json` object that represents the original JSON object after the specified patch has been applied.


---
### diff<!-- {{#callable:diff}} -->
Calculates the differences between two JSON objects or arrays and returns a JSON patch.
- **Inputs**:
    - `source`: The original `basic_json` object to compare against.
    - `target`: The `basic_json` object that represents the new state.
    - `path`: A string representing the JSON path to the current element being compared, defaulting to an empty string.
- **Control Flow**:
    - Initializes an empty JSON array `result` to store the differences.
    - Checks if `source` and `target` are equal; if so, returns an empty patch.
    - If the types of `source` and `target` differ, adds a 'replace' operation to `result`.
    - If both are arrays, iterates through their elements, recursively calling `diff` for common indices, and handles remaining elements by adding 'remove' or 'add' operations.
    - If both are objects, iterates through the keys of `source`, checking for existence in `target`, and adds 'remove' or 'add' operations as necessary.
    - For primitive types, adds a 'replace' operation to `result`.
- **Output**: Returns a `basic_json` array containing the operations needed to transform `source` into `target`.
- **Functions called**:
    - [`object`](#object)


---
### merge\_patch<!-- {{#callable:merge_patch}} -->
The `merge_patch` function applies a JSON patch to the current JSON object, modifying it based on the provided patch.
- **Inputs**:
    - `apply_patch`: A `basic_json` object representing the patch to be applied, which can either be a JSON object or a value.
- **Control Flow**:
    - The function first checks if `apply_patch` is a JSON object.
    - If `apply_patch` is an object and the current object is not, it initializes the current object as an empty JSON object.
    - It then iterates over each key-value pair in `apply_patch`.
    - For each key-value pair, if the value is null, it removes the key from the current object.
    - If the value is not null, it recursively calls `merge_patch` on the current object's value associated with the key.
    - If `apply_patch` is not an object, it directly assigns `apply_patch` to the current object.
- **Output**: The function modifies the current JSON object in place, resulting in a merged object that reflects the changes specified by the `apply_patch`.
- **Functions called**:
    - [`is_object`](#is_object)
    - [`object`](#object)
    - [`erase`](#erase)


---
### operator "" \_json<!-- {{#callable:operator "" _json}} -->
Parses a JSON string literal into a `nlohmann::json` object.
- **Inputs**:
    - `s`: A pointer to a character array representing the JSON string.
    - `n`: The size of the JSON string, indicating how many characters to parse.
- **Control Flow**:
    - The function calls `nlohmann::json::parse` with the start and end pointers derived from the input string and its size.
    - The `parse` function processes the string and constructs a `nlohmann::json` object from it.
- **Output**: Returns a `nlohmann::json` object that represents the parsed JSON data.


---
### operator "" \_json\_pointer<!-- {{#callable:operator "" _json_pointer}} -->
This function is a user-defined literal operator that converts a string literal into a `json_pointer` object.
- **Inputs**:
    - `s`: A pointer to a character array (string literal) that represents the JSON pointer.
    - `n`: The size of the string literal, indicating how many characters to consider from the pointer.
- **Control Flow**:
    - The function takes two parameters: a character pointer `s` and a size `n`.
    - It constructs a `std::string` from the character pointer `s` with the specified size `n`.
    - It then initializes and returns a `nlohmann::json::json_pointer` object using the constructed string.
- **Output**: The function returns a `nlohmann::json::json_pointer` object that represents the JSON pointer derived from the input string.


---
### operator\(\)<!-- {{#callable:namespace::operator()}} -->
This function overloads the `operator()` to compare two `value_t` objects and returns a boolean indicating if the left-hand side is less than the right-hand side.
- **Inputs**:
    - `lhs`: The left-hand side operand of type `::nlohmann::detail::value_t` to be compared.
    - `rhs`: The right-hand side operand of type `::nlohmann::detail::value_t` to be compared.
- **Control Flow**:
    - The function checks if the macro `JSON_HAS_THREE_WAY_COMPARISON` is defined.
    - If defined, it uses the spaceship operator `<=>` to perform the comparison and checks if the result indicates that `lhs` is less than `rhs`.
    - If not defined, it falls back to using the `operator<` to perform the comparison.
- **Output**: Returns a boolean value: `true` if `lhs` is less than `rhs`, otherwise `false`.


---
### output\_adapter\_protocol<!-- {{#callable:namespace::output_adapter_protocol::output_adapter_protocol}} -->
Default constructor and copy constructor for the `output_adapter_protocol` class.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements as it is a default constructor.
    - It initializes an instance of the `output_adapter_protocol` class with default values.
- **Output**: The function does not return any value; it constructs an instance of the `output_adapter_protocol` class.


