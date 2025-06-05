# Purpose
The provided C++ code is a comprehensive header file for the `httplib` library, designed to facilitate HTTP communication in C++ applications by offering both client and server functionalities. It supports various HTTP methods (GET, POST, PUT, DELETE, etc.), handles HTTP headers, manages connections, and processes HTTP requests and responses, with SSL/TLS support for secure communications. Key components include server and client classes, request and response structures, and utility functions for tasks like URL encoding and MIME type determination. The `ClientImpl` class specifically focuses on HTTP client functionality, providing methods for sending requests, handling responses, managing connections, and supporting features like proxy settings and authentication. The library defines public APIs for setting up HTTP servers, sending requests, and handling responses, designed to be flexible, extensible, and optimized for performance, making it suitable for a wide range of applications requiring HTTP communication.
# Imports and Dependencies

---
- `io.h`
- `winsock2.h`
- `ws2tcpip.h`
- `afunix.h`
- `arpa/inet.h`
- `ifaddrs.h`
- `strings.h`
- `net/if.h`
- `netdb.h`
- `netinet/in.h`
- `resolv.h`
- `csignal`
- `netinet/tcp.h`
- `poll.h`
- `pthread.h`
- `sys/mman.h`
- `sys/socket.h`
- `sys/un.h`
- `unistd.h`
- `algorithm`
- `array`
- `atomic`
- `cassert`
- `cctype`
- `climits`
- `condition_variable`
- `cstring`
- `errno.h`
- `exception`
- `fcntl.h`
- `functional`
- `iomanip`
- `iostream`
- `list`
- `map`
- `memory`
- `mutex`
- `random`
- `regex`
- `set`
- `sstream`
- `string`
- `sys/stat.h`
- `thread`
- `unordered_map`
- `unordered_set`
- `utility`
- `wincrypt.h`
- `TargetConditionals.h`
- `CoreFoundation/CoreFoundation.h`
- `Security/Security.h`
- `openssl/err.h`
- `openssl/evp.h`
- `openssl/ssl.h`
- `openssl/x509v3.h`
- `openssl/applink.c`
- `zlib.h`
- `brotli/decode.h`
- `brotli/encode.h`
- `zstd.h`


# Data Structures

---
### equal\_to<!-- {{#data_structure:detail::case_ignore::equal_to}} -->
- **Type**: `struct`
- **Description**: The `equal_to` struct is a function object that provides a mechanism to compare two `std::string` objects for equality. It overloads the function call operator to take two constant references to `std::string` and returns a boolean indicating whether the two strings are equal. This struct is typically used in contexts where a comparison function is required, such as in associative containers or algorithms that need to determine equality between elements.
- **Member Functions**:
    - [`detail::case_ignore::equal_to::operator()`](#equal_tooperator())

**Methods**

---
#### equal\_to::operator\(\)<!-- {{#callable:detail::case_ignore::equal_to::operator()}} -->
The `operator()` function in the `equal_to` struct checks if two strings are equal using the [`equal`](#case_ignoreequal) function.
- **Inputs**:
    - `a`: The first string to be compared.
    - `b`: The second string to be compared.
- **Control Flow**:
    - The function takes two string arguments, `a` and `b`.
    - It calls the [`equal`](#case_ignoreequal) function with `a` and `b` as arguments to determine if they are equal.
    - The result of the [`equal`](#case_ignoreequal) function call is returned as the output of the `operator()` function.
- **Output**: A boolean value indicating whether the two input strings are equal.
- **Functions called**:
    - [`detail::case_ignore::equal`](#case_ignoreequal)
- **See also**: [`detail::case_ignore::equal_to`](#case_ignoreequal_to)  (Data Structure)



---
### hash<!-- {{#data_structure:detail::case_ignore::hash}} -->
- **Type**: `struct`
- **Description**: The `hash` struct is a custom hash function object designed to compute a hash value for a given string. It overrides the function call operator to provide a hash value by invoking the `hash_core` method, which recursively processes each character of the string, applying a transformation that prevents overflow by unsetting the high bits and incorporating each character into the hash using a combination of bitwise operations and multiplication. This struct is useful for creating hash values that can be used in hash-based data structures like hash tables.
- **Member Functions**:
    - [`detail::case_ignore::hash::operator()`](#hashoperator())
    - [`detail::case_ignore::hash::hash_core`](#hashhash_core)

**Methods**

---
#### hash::operator\(\)<!-- {{#callable:detail::case_ignore::hash::operator()}} -->
The `operator()` function computes a hash value for a given string using a recursive hash function.
- **Inputs**:
    - `key`: A constant reference to a `std::string` that represents the input string to be hashed.
- **Control Flow**:
    - The function `operator()` is called with a `std::string` argument `key`.
    - It calls the [`hash_core`](#hashhash_core) function, passing the data pointer of the string, the size of the string, and an initial hash value of 0.
    - The [`hash_core`](#hashhash_core) function recursively processes each character of the string, updating the hash value by unsetting the 6 high bits to prevent overflow and applying a hash transformation using multiplication and XOR operations.
- **Output**: Returns a `size_t` value representing the computed hash of the input string.
- **Functions called**:
    - [`detail::case_ignore::hash::hash_core`](#hashhash_core)
- **See also**: [`detail::case_ignore::hash`](#case_ignorehash)  (Data Structure)


---
#### hash::hash\_core<!-- {{#callable:detail::case_ignore::hash::hash_core}} -->
The `hash_core` function recursively computes a hash value for a given character string by processing each character, converting it to lowercase, and applying a specific hash formula.
- **Inputs**:
    - `s`: A pointer to the character array (string) to be hashed.
    - `l`: The length of the string to be processed.
    - `h`: The current hash value, initially set to 0.
- **Control Flow**:
    - The function checks if the length `l` is zero; if so, it returns the current hash value `h`.
    - If `l` is not zero, the function calls itself recursively with the next character in the string (`s + 1`), decrements the length (`l - 1`), and computes a new hash value.
    - The new hash value is calculated by unsetting the 6 high bits of `h` to prevent overflow, multiplying `h` by 33, and XORing it with the lowercase version of the current character.
- **Output**: The function returns a `size_t` value representing the computed hash of the input string.
- **Functions called**:
    - [`detail::case_ignore::to_lower`](#case_ignoreto_lower)
- **See also**: [`detail::case_ignore::hash`](#case_ignorehash)  (Data Structure)



---
### scope\_exit<!-- {{#data_structure:detail::scope_exit}} -->
- **Type**: `struct`
- **Members**:
    - `exit_function`: A std::function that stores the function to be executed upon destruction of the scope_exit object.
    - `execute_on_destruction`: A boolean flag indicating whether the exit_function should be executed when the scope_exit object is destroyed.
- **Description**: The `scope_exit` struct is a utility that ensures a specified function is executed when the object goes out of scope, typically used for resource management and cleanup tasks. It holds a function object, `exit_function`, which is called upon destruction if the `execute_on_destruction` flag is true. The struct is non-copyable but movable, and provides a `release` method to prevent the function from being executed upon destruction.
- **Member Functions**:
    - [`detail::scope_exit::scope_exit`](#scope_exitscope_exit)
    - [`detail::scope_exit::scope_exit`](#scope_exitscope_exit)
    - [`detail::scope_exit::~scope_exit`](#scope_exitscope_exit)
    - [`detail::scope_exit::release`](#scope_exitrelease)
    - [`detail::scope_exit::scope_exit`](#scope_exitscope_exit)
    - [`detail::scope_exit::operator=`](#scope_exitoperator=)
    - [`detail::scope_exit::operator=`](#scope_exitoperator=)

**Methods**

---
#### scope\_exit::scope\_exit<!-- {{#callable:detail::scope_exit::scope_exit}} -->
The `scope_exit` constructor initializes an instance with a function to be executed upon the object's destruction.
- **Inputs**:
    - `f`: A rvalue reference to a `std::function<void(void)>` that represents the function to be executed when the `scope_exit` object is destroyed.
- **Control Flow**:
    - The constructor takes an rvalue reference to a `std::function<void(void)>` as an argument.
    - It initializes the `exit_function` member by moving the provided function into it.
    - The `execute_on_destruction` member is set to `true`, indicating that the function should be executed when the object is destroyed.
- **Output**: An instance of the `scope_exit` struct is created, initialized to execute the provided function upon destruction.
- **See also**: [`detail::scope_exit`](#detailscope_exit)  (Data Structure)


---
#### scope\_exit::scope\_exit<!-- {{#callable:detail::scope_exit::scope_exit}} -->
The `scope_exit` move constructor transfers ownership of the exit function and its execution flag from one `scope_exit` object to another, ensuring the original object no longer executes the function on destruction.
- **Inputs**:
    - `rhs`: A rvalue reference to another `scope_exit` object from which resources are being moved.
- **Control Flow**:
    - The move constructor initializes the `exit_function` member by moving it from the `rhs` object.
    - The `execute_on_destruction` flag is copied from the `rhs` object to the new object.
    - The `release` method is called on the `rhs` object to set its `execute_on_destruction` flag to false, preventing it from executing the function on destruction.
- **Output**: A new `scope_exit` object with ownership of the exit function and execution flag from the `rhs` object.
- **See also**: [`detail::scope_exit`](#detailscope_exit)  (Data Structure)


---
#### scope\_exit::\~scope\_exit<!-- {{#callable:detail::scope_exit::~scope_exit}} -->
The destructor of the `scope_exit` struct executes a stored function if the `execute_on_destruction` flag is true.
- **Inputs**: None
- **Control Flow**:
    - The destructor checks if the `execute_on_destruction` flag is true.
    - If true, it calls the `exit_function`.
- **Output**: The destructor does not return any value; it performs an action based on the state of the object.
- **See also**: [`detail::scope_exit`](#detailscope_exit)  (Data Structure)


---
#### scope\_exit::release<!-- {{#callable:detail::scope_exit::release}} -->
The `release` function sets the `execute_on_destruction` flag to false, preventing the execution of the stored function upon object destruction.
- **Inputs**: None
- **Control Flow**:
    - The function directly sets the `execute_on_destruction` member variable to false.
- **Output**: The function does not return any value.
- **See also**: [`detail::scope_exit`](#detailscope_exit)  (Data Structure)


---
#### scope\_exit::scope\_exit<!-- {{#callable:detail::scope_exit::scope_exit}} -->
The `scope_exit` struct is a RAII (Resource Acquisition Is Initialization) utility that ensures a specified function is executed when the scope is exited, unless explicitly released.
- **Inputs**:
    - `f`: A callable object (function) that takes no arguments and returns void, which will be executed upon the destruction of the `scope_exit` object unless released.
- **Control Flow**:
    - The constructor initializes the `exit_function` with the provided function and sets `execute_on_destruction` to true.
    - The move constructor transfers ownership of the `exit_function` and `execute_on_destruction` flag from the source object, then calls `release` on the source to prevent double execution.
    - The destructor checks if `execute_on_destruction` is true, and if so, calls the `exit_function`.
    - The `release` method sets `execute_on_destruction` to false, preventing the `exit_function` from being called upon destruction.
- **Output**: The `scope_exit` struct does not produce a direct output, but it ensures the execution of a specified function upon destruction unless released.
- **See also**: [`detail::scope_exit`](#detailscope_exit)  (Data Structure)


---
#### scope\_exit::operator=<!-- {{#callable:detail::scope_exit::operator=}} -->
The assignment operators for the `scope_exit` struct are deleted to prevent copying or moving of `scope_exit` instances.
- **Inputs**: None
- **Control Flow**:
    - The copy assignment operator `operator=(const scope_exit &)` is deleted, preventing copying of `scope_exit` instances.
    - The move assignment operator `operator=(scope_exit &&)` is also deleted, preventing moving of `scope_exit` instances.
- **Output**: There is no output as these operators are deleted, meaning assignment is not allowed for `scope_exit` instances.
- **See also**: [`detail::scope_exit`](#detailscope_exit)  (Data Structure)


---
#### scope\_exit::operator=<!-- {{#callable:detail::scope_exit::operator=}} -->
The move assignment operator for the `scope_exit` struct is explicitly deleted to prevent move assignment.
- **Inputs**: None
- **Control Flow**:
    - The move assignment operator is declared but immediately deleted, indicating that move assignment is not allowed for instances of `scope_exit`.
- **Output**: There is no output as the function is deleted and cannot be used.
- **See also**: [`detail::scope_exit`](#detailscope_exit)  (Data Structure)



---
### SSLVerifierResponse<!-- {{#data_structure:SSLVerifierResponse}} -->
- **Type**: `enum`
- **Members**:
    - `NoDecisionMade`: Indicates that no decision has been made and the built-in certificate verifier should be used.
    - `CertificateAccepted`: Indicates that the connection certificate is verified and accepted.
    - `CertificateRejected`: Indicates that the connection certificate was processed but is rejected.
- **Description**: The `SSLVerifierResponse` enum represents the possible outcomes of an SSL certificate verification process. It provides three states: `NoDecisionMade` for when no decision has been reached and the default verifier should be used, `CertificateAccepted` for when the certificate is verified and accepted, and `CertificateRejected` for when the certificate is processed but rejected. This enum is typically used in SSL/TLS communication to determine the result of a certificate verification step.


---
### StatusCode<!-- {{#data_structure:StatusCode}} -->
- **Type**: `enum`
- **Members**:
    - `Continue_100`: Represents the HTTP status code 100, indicating that the initial part of a request has been received and has not yet been rejected by the server.
    - `SwitchingProtocol_101`: Represents the HTTP status code 101, indicating that the server is switching protocols as requested by the client.
    - `Processing_102`: Represents the HTTP status code 102, indicating that the server has received and is processing the request, but no response is available yet.
    - `EarlyHints_103`: Represents the HTTP status code 103, used to return some response headers before final HTTP message.
    - `OK_200`: Represents the HTTP status code 200, indicating that the request has succeeded.
    - `Created_201`: Represents the HTTP status code 201, indicating that the request has been fulfilled and has resulted in one or more new resources being created.
    - `Accepted_202`: Represents the HTTP status code 202, indicating that the request has been accepted for processing, but the processing has not been completed.
    - `NonAuthoritativeInformation_203`: Represents the HTTP status code 203, indicating that the request was successful but the enclosed payload has been modified from that of the origin server's 200 (OK) response.
    - `NoContent_204`: Represents the HTTP status code 204, indicating that the server has successfully fulfilled the request and there is no additional content to send in the response payload body.
    - `ResetContent_205`: Represents the HTTP status code 205, indicating that the server has fulfilled the request and desires that the user agent reset the 'document view' which caused the request to be sent.
    - `PartialContent_206`: Represents the HTTP status code 206, indicating that the server is delivering only part of the resource due to a range header sent by the client.
    - `MultiStatus_207`: Represents the HTTP status code 207, providing status for multiple independent operations.
    - `AlreadyReported_208`: Represents the HTTP status code 208, used inside a DAV: propstat response element to avoid enumerating the internal members of multiple bindings to the same collection repeatedly.
    - `IMUsed_226`: Represents the HTTP status code 226, indicating that the server has fulfilled a GET request for the resource, and the response is a representation of the result of one or more instance-manipulations applied to the current instance.
    - `MultipleChoices_300`: Represents the HTTP status code 300, indicating multiple options for the resource from which the client may choose.
    - `MovedPermanently_301`: Represents the HTTP status code 301, indicating that the resource requested has been permanently moved to a new URL.
    - `Found_302`: Represents the HTTP status code 302, indicating that the resource requested has been temporarily moved to a different URL.
    - `SeeOther_303`: Represents the HTTP status code 303, indicating that the response to the request can be found under another URI using a GET method.
    - `NotModified_304`: Represents the HTTP status code 304, indicating that the resource has not been modified since the version specified by the request headers.
    - `UseProxy_305`: Represents the HTTP status code 305, indicating that the requested resource must be accessed through the proxy given by the Location field.
    - `unused_306`: Represents the HTTP status code 306, which is no longer used but reserved for future use.
    - `TemporaryRedirect_307`: Represents the HTTP status code 307, indicating that the resource requested has been temporarily moved to a different URL and the client should use the same method for the request.
    - `PermanentRedirect_308`: Represents the HTTP status code 308, indicating that the resource requested has been permanently moved to a new URL and the client should use the same method for the request.
    - `BadRequest_400`: Represents the HTTP status code 400, indicating that the server cannot or will not process the request due to a client error.
    - `Unauthorized_401`: Represents the HTTP status code 401, indicating that the request has not been applied because it lacks valid authentication credentials for the target resource.
    - `PaymentRequired_402`: Represents the HTTP status code 402, reserved for future use.
    - `Forbidden_403`: Represents the HTTP status code 403, indicating that the server understood the request but refuses to authorize it.
    - `NotFound_404`: Represents the HTTP status code 404, indicating that the server cannot find the requested resource.
    - `MethodNotAllowed_405`: Represents the HTTP status code 405, indicating that the request method is known by the server but is not supported by the target resource.
    - `NotAcceptable_406`: Represents the HTTP status code 406, indicating that the server cannot produce a response matching the list of acceptable values defined in the request's proactive content negotiation headers.
    - `ProxyAuthenticationRequired_407`: Represents the HTTP status code 407, indicating that the client must first authenticate itself with the proxy.
    - `RequestTimeout_408`: Represents the HTTP status code 408, indicating that the server did not receive a complete request message within the time that it was prepared to wait.
    - `Conflict_409`: Represents the HTTP status code 409, indicating that the request could not be completed due to a conflict with the current state of the target resource.
    - `Gone_410`: Represents the HTTP status code 410, indicating that access to the target resource is no longer available at the origin server and that this condition is likely to be permanent.
    - `LengthRequired_411`: Represents the HTTP status code 411, indicating that the server refuses to accept the request without a defined Content-Length.
    - `PreconditionFailed_412`: Represents the HTTP status code 412, indicating that one or more conditions given in the request header fields evaluated to false when tested on the server.
    - `PayloadTooLarge_413`: Represents the HTTP status code 413, indicating that the server is refusing to process a request because the request payload is larger than the server is willing or able to process.
    - `UriTooLong_414`: Represents the HTTP status code 414, indicating that the server is refusing to service the request because the request-target is longer than the server is willing to interpret.
    - `UnsupportedMediaType_415`: Represents the HTTP status code 415, indicating that the origin server is refusing to service the request because the payload is in a format not supported by the target resource for this method.
    - `RangeNotSatisfiable_416`: Represents the HTTP status code 416, indicating that none of the ranges in the request's Range header field overlap the current extent of the selected resource or that the set of ranges requested has been rejected due to invalid ranges.
    - `ExpectationFailed_417`: Represents the HTTP status code 417, indicating that the expectation given in the request's Expect header field could not be met by at least one of the inbound servers.
    - `ImATeapot_418`: Represents the HTTP status code 418, a humorous response code indicating that the server refuses to brew coffee because it is, permanently, a teapot.
    - `MisdirectedRequest_421`: Represents the HTTP status code 421, indicating that the request was directed at a server that is not able to produce a response.
    - `UnprocessableContent_422`: Represents the HTTP status code 422, indicating that the server understands the content type of the request entity, and the syntax of the request entity is correct, but it was unable to process the contained instructions.
    - `Locked_423`: Represents the HTTP status code 423, indicating that the resource that is being accessed is locked.
    - `FailedDependency_424`: Represents the HTTP status code 424, indicating that the method could not be performed on the resource because the requested action depended on another action and that action failed.
    - `TooEarly_425`: Represents the HTTP status code 425, indicating that the server is unwilling to risk processing a request that might be replayed.
    - `UpgradeRequired_426`: Represents the HTTP status code 426, indicating that the client should switch to a different protocol.
    - `PreconditionRequired_428`: Represents the HTTP status code 428, indicating that the origin server requires the request to be conditional.
    - `TooManyRequests_429`: Represents the HTTP status code 429, indicating that the user has sent too many requests in a given amount of time.
    - `RequestHeaderFieldsTooLarge_431`: Represents the HTTP status code 431, indicating that the server is unwilling to process the request because its header fields are too large.
    - `UnavailableForLegalReasons_451`: Represents the HTTP status code 451, indicating that the server is denying access to the resource as a consequence of a legal demand.
    - `InternalServerError_500`: Represents the HTTP status code 500, indicating that the server encountered an unexpected condition that prevented it from fulfilling the request.
    - `NotImplemented_501`: Represents the HTTP status code 501, indicating that the server does not support the functionality required to fulfill the request.
    - `BadGateway_502`: Represents the HTTP status code 502, indicating that the server, while acting as a gateway or proxy, received an invalid response from the upstream server.
    - `ServiceUnavailable_503`: Represents the HTTP status code 503, indicating that the server is currently unable to handle the request due to temporary overloading or maintenance of the server.
    - `GatewayTimeout_504`: Represents the HTTP status code 504, indicating that the server, while acting as a gateway or proxy, did not receive a timely response from the upstream server.
    - `HttpVersionNotSupported_505`: Represents the HTTP status code 505, indicating that the server does not support the HTTP protocol version that was used in the request message.
    - `VariantAlsoNegotiates_506`: Represents the HTTP status code 506, indicating that the server has an internal configuration error: the chosen variant resource is configured to engage in transparent content negotiation itself, and is therefore not a proper end point in the negotiation process.
    - `InsufficientStorage_507`: Represents the HTTP status code 507, indicating that the server is unable to store the representation needed to complete the request.
    - `LoopDetected_508`: Represents the HTTP status code 508, indicating that the server detected an infinite loop while processing a request with "Depth: infinity".
    - `NotExtended_510`: Represents the HTTP status code 510, indicating that further extensions to the request are required for the server to fulfill it.
    - `NetworkAuthenticationRequired_511`: Represents the HTTP status code 511, indicating that the client needs to authenticate to gain network access.
- **Description**: The `StatusCode` enum defines a comprehensive set of HTTP status codes, categorizing them into information responses, successful responses, redirection messages, client error responses, and server error responses. Each enumerator in the `StatusCode` enum corresponds to a specific HTTP status code, providing a clear and structured way to handle HTTP responses in a program. This enum is useful for developers to manage and interpret HTTP status codes in a standardized manner, facilitating error handling and response management in web applications.


---
### MultipartFormData<!-- {{#data_structure:MultipartFormData}} -->
- **Type**: `struct`
- **Members**:
    - `name`: The name of the form field.
    - `content`: The content or value associated with the form field.
    - `filename`: The name of the file being uploaded, if applicable.
    - `content_type`: The MIME type of the content, indicating the nature and format of the file.
- **Description**: The `MultipartFormData` struct is designed to represent a single part of a multipart form data submission, typically used in HTTP requests for file uploads. It contains fields for the name of the form field, the content or value of the field, the filename if a file is being uploaded, and the content type to specify the MIME type of the data. This struct is often used in conjunction with a vector of `MultipartFormData` items to handle multiple parts of a form submission.


---
### DataSink<!-- {{#data_structure:DataSink}} -->
- **Type**: `class`
- **Members**:
    - `write`: A function object that writes data to the sink, returning a boolean indicating success.
    - `is_writable`: A function object that checks if the sink is writable, returning a boolean.
    - `done`: A function object that signals the completion of data writing.
    - `done_with_trailer`: A function object that signals completion with additional trailer headers.
    - `os`: An output stream associated with the data sink.
    - `sb_`: A custom stream buffer used by the output stream to handle data writing.
- **Description**: The `DataSink` class is designed to handle data output operations, providing a flexible interface for writing data, checking writability, and signaling completion. It encapsulates a custom stream buffer (`data_sink_streambuf`) to manage the data flow through an associated output stream (`os`). The class is non-copyable and non-movable, ensuring that each instance maintains its unique state and behavior. The `DataSink` class is particularly useful in scenarios where data needs to be processed or transmitted in a controlled manner, with hooks for handling completion and additional headers.
- **Member Functions**:
    - [`DataSink::DataSink`](#DataSinkDataSink)
    - [`DataSink::DataSink`](#DataSinkDataSink)
    - [`DataSink::operator=`](#DataSinkoperator=)
    - [`DataSink::DataSink`](#DataSinkDataSink)
    - [`DataSink::operator=`](#DataSinkoperator=)

**Methods**

---
#### DataSink::DataSink<!-- {{#callable:DataSink::DataSink}} -->
The `DataSink` constructor initializes a `DataSink` object with a custom stream buffer for output streaming.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes the `os` member with a pointer to the `sb_` member, which is an instance of the `data_sink_streambuf` class.
    - The `sb_` member is initialized with a reference to the `DataSink` object itself.
- **Output**: The constructor does not return any value as it is a constructor for the `DataSink` class.
- **See also**: [`DataSink`](#DataSink)  (Data Structure)


---
#### DataSink::DataSink<!-- {{#callable:DataSink::DataSink}} -->
The `DataSink` constructor initializes a `DataSink` object with a custom stream buffer and deletes copy and move operations to prevent copying or moving of the object.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes the `os` member with a reference to the `sb_` member, which is an instance of the nested `data_sink_streambuf` class.
    - The `data_sink_streambuf` is initialized with a reference to the `DataSink` object itself.
    - Copy constructor and copy assignment operator are deleted to prevent copying of `DataSink` objects.
    - Move constructor and move assignment operator are also deleted to prevent moving of `DataSink` objects.
- **Output**: A `DataSink` object with a custom stream buffer and disabled copy and move operations.
- **See also**: [`DataSink`](#DataSink)  (Data Structure)


---
#### DataSink::operator=<!-- {{#callable:DataSink::operator=}} -->
The assignment operator for the DataSink class is deleted to prevent copying of DataSink objects.
- **Inputs**: None
- **Control Flow**:
    - The assignment operator is explicitly deleted, which means any attempt to copy-assign a DataSink object will result in a compile-time error.
- **Output**: There is no output as the function is deleted and cannot be used.
- **See also**: [`DataSink`](#DataSink)  (Data Structure)


---
#### DataSink::DataSink<!-- {{#callable:DataSink::DataSink}} -->
The `DataSink` class is a non-copyable and non-movable utility for writing data streams using custom write, is_writable, and done functions.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes the `os` member with a custom stream buffer `sb_` that is linked to the `DataSink` instance.
    - The class deletes the copy constructor, copy assignment operator, move constructor, and move assignment operator to prevent copying and moving of `DataSink` instances.
    - The `data_sink_streambuf` class, a private member of `DataSink`, overrides the `xsputn` method to write data using the `DataSink`'s `write` function.
- **Output**: An instance of `DataSink` provides an output stream `os` that can be used to write data using the specified `write` function.
- **See also**: [`DataSink`](#DataSink)  (Data Structure)


---
#### DataSink::operator=<!-- {{#callable:DataSink::operator=}} -->
The move assignment operator for the DataSink class is explicitly deleted, preventing move assignment of DataSink objects.
- **Inputs**: None
- **Control Flow**:
    - The function is declared as deleted, meaning it cannot be used or called.
    - This prevents any move assignment operation on instances of the DataSink class.
- **Output**: There is no output as the function is deleted and cannot be invoked.
- **See also**: [`DataSink`](#DataSink)  (Data Structure)



---
### data\_sink\_streambuf<!-- {{#data_structure:DataSink::data_sink_streambuf}} -->
- **Type**: `class`
- **Members**:
    - `sink_`: A reference to a DataSink object used for writing data.
- **Description**: The `data_sink_streambuf` class is a custom stream buffer that extends `std::streambuf` to facilitate writing data to a `DataSink` object. It overrides the `xsputn` function to write a specified number of characters from a given buffer to the associated `DataSink`. The class is designed to be used as a final class, meaning it cannot be further derived. The primary purpose of this class is to integrate with the C++ stream library, allowing data to be seamlessly directed to a `DataSink`.
- **Member Functions**:
    - [`DataSink::data_sink_streambuf::data_sink_streambuf`](#data_sink_streambufdata_sink_streambuf)
    - [`DataSink::data_sink_streambuf::xsputn`](#data_sink_streambufxsputn)
- **Inherits From**:
    - `std::streambuf`

**Methods**

---
#### data\_sink\_streambuf::data\_sink\_streambuf<!-- {{#callable:DataSink::data_sink_streambuf::data_sink_streambuf}} -->
The `data_sink_streambuf` constructor initializes a stream buffer with a reference to a `DataSink` object.
- **Inputs**:
    - `sink`: A reference to a `DataSink` object that the stream buffer will use for writing data.
- **Control Flow**:
    - The constructor takes a reference to a `DataSink` object as an argument.
    - It initializes the private member `sink_` with the provided `DataSink` reference.
- **Output**: There is no output from the constructor; it initializes the object state.
- **See also**: [`DataSink::data_sink_streambuf`](#DataSink::data_sink_streambuf)  (Data Structure)


---
#### data\_sink\_streambuf::xsputn<!-- {{#callable:DataSink::data_sink_streambuf::xsputn}} -->
The `xsputn` function writes a specified number of characters from a given character array to a data sink and returns the number of characters written.
- **Inputs**:
    - `s`: A pointer to a character array containing the data to be written to the sink.
    - `n`: The number of characters to write from the character array to the sink.
- **Control Flow**:
    - The function calls the `write` method of the `sink_` object, passing the character array `s` and the number of characters `n` cast to a `size_t`.
    - The function returns the number of characters `n` that were intended to be written.
- **Output**: The function returns the number of characters `n` that were written to the sink.
- **See also**: [`DataSink::data_sink_streambuf`](#DataSink::data_sink_streambuf)  (Data Structure)



---
### MultipartFormDataProvider<!-- {{#data_structure:MultipartFormDataProvider}} -->
- **Type**: `struct`
- **Members**:
    - `name`: A string representing the name of the form data part.
    - `provider`: An instance of ContentProviderWithoutLength that provides the content for the form data part.
    - `filename`: A string representing the filename associated with the form data part.
    - `content_type`: A string indicating the MIME type of the form data part.
- **Description**: The `MultipartFormDataProvider` struct is designed to represent a single part of multipart form data, typically used in HTTP requests to upload files or submit form data. Each instance of this struct holds information about a specific part, including its name, the content provider, the filename, and the content type. This struct is often used in conjunction with a vector of `MultipartFormDataProvider` instances, allowing for the representation of multiple parts in a multipart form submission.


---
### ContentReader<!-- {{#data_structure:ContentReader}} -->
- **Type**: `class`
- **Members**:
    - `reader_`: A function object that processes content using a ContentReceiver.
    - `multipart_reader_`: A function object that processes multipart content using a MultipartContentHeader and a ContentReceiver.
- **Description**: The ContentReader class is designed to handle content reading operations, supporting both single and multipart content through function objects. It encapsulates two main function objects, Reader and MultipartReader, which are used to process content and multipart content respectively. The class provides operator overloads to facilitate the invocation of these function objects, allowing for flexible content processing strategies.
- **Member Functions**:
    - [`ContentReader::ContentReader`](#ContentReaderContentReader)
    - [`ContentReader::operator()`](#ContentReaderoperator())
    - [`ContentReader::operator()`](#ContentReaderoperator())

**Methods**

---
#### ContentReader::ContentReader<!-- {{#callable:ContentReader::ContentReader}} -->
The ContentReader constructor initializes a ContentReader object with a Reader and a MultipartReader by moving them into member variables.
- **Inputs**:
    - `reader`: A Reader function, which is a std::function that takes a ContentReceiver and returns a bool.
    - `multipart_reader`: A MultipartReader function, which is a std::function that takes a MultipartContentHeader and a ContentReceiver, and returns a bool.
- **Control Flow**:
    - The constructor initializes the member variable reader_ by moving the provided reader argument into it.
    - The constructor initializes the member variable multipart_reader_ by moving the provided multipart_reader argument into it.
- **Output**: The constructor does not return any value as it is used to initialize an object of the ContentReader class.
- **See also**: [`ContentReader`](#ContentReader)  (Data Structure)


---
#### ContentReader::operator\(\)<!-- {{#callable:ContentReader::operator()}} -->
The `operator()` function invokes a stored `MultipartReader` function with a `MultipartContentHeader` and `ContentReceiver` as arguments.
- **Inputs**:
    - `header`: A `MultipartContentHeader` object that contains metadata or information about the multipart content.
    - `receiver`: A `ContentReceiver` object that is used to handle or process the content.
- **Control Flow**:
    - The function takes two parameters: `header` and `receiver`.
    - It calls the `multipart_reader_` function, passing the `header` and `receiver` as arguments using `std::move` to transfer ownership.
    - The result of the `multipart_reader_` function call is returned as the output of the `operator()` function.
- **Output**: A boolean value indicating the success or failure of the `multipart_reader_` function call.
- **See also**: [`ContentReader`](#ContentReader)  (Data Structure)


---
#### ContentReader::operator\(\)<!-- {{#callable:ContentReader::operator()}} -->
The `operator()` function invokes the `reader_` function with a `ContentReceiver` argument and returns its boolean result.
- **Inputs**:
    - `receiver`: A `ContentReceiver` object that is passed to the `reader_` function.
- **Control Flow**:
    - The function takes a `ContentReceiver` object as an argument.
    - It calls the `reader_` function, passing the `receiver` as an argument using `std::move` to enable move semantics.
    - The result of the `reader_` function call is returned as the output of the operator.
- **Output**: A boolean value that is the result of the `reader_` function call.
- **See also**: [`ContentReader`](#ContentReader)  (Data Structure)



---
### Request<!-- {{#data_structure:Request}} -->
- **Type**: `struct`
- **Members**:
    - `method`: Stores the HTTP method of the request as a string.
    - `path`: Holds the path of the request URL as a string.
    - `params`: Contains the query parameters of the request.
    - `headers`: Stores the headers of the request.
    - `body`: Holds the body content of the request as a string.
    - `remote_addr`: Stores the remote address of the client as a string.
    - `remote_port`: Holds the remote port number of the client, defaulting to -1.
    - `local_addr`: Contains the local address of the server as a string.
    - `local_port`: Holds the local port number of the server, defaulting to -1.
    - `version`: Stores the HTTP version of the request as a string.
    - `target`: Holds the target of the request as a string.
    - `files`: Contains a map of multipart form data files.
    - `ranges`: Stores the byte ranges requested in the HTTP request.
    - `matches`: Holds match results for the request.
    - `path_params`: Contains a map of path parameters extracted from the request URL.
    - `is_connection_closed`: A function to determine if the connection is closed, defaulting to always true.
    - `response_handler`: Handles the response for the client.
    - `content_receiver`: Receives content with progress tracking.
    - `progress`: Tracks the progress of the request.
    - `ssl`: Points to the SSL structure if OpenSSL support is enabled.
    - `redirect_count_`: Tracks the number of redirects, initialized to a maximum count.
    - `content_length_`: Stores the length of the content in the request.
    - `content_provider_`: Provides the content for the request.
    - `is_chunked_content_provider_`: Indicates if the content provider is chunked.
    - `authorization_count_`: Tracks the number of authorization attempts.
    - `start_time_`: Records the start time of the request as a steady clock time point.
- **Description**: The `Request` struct is a comprehensive data structure designed to encapsulate all the necessary information for handling HTTP requests in a client-server architecture. It includes fields for storing HTTP method, path, headers, body, and parameters, as well as network-related information such as remote and local addresses and ports. The struct also supports multipart form data, byte ranges, and path parameters, and provides mechanisms for handling SSL connections, tracking request progress, and managing redirects and authorizations. Additionally, it includes several utility functions for accessing and manipulating headers, parameters, and files, making it a versatile tool for HTTP request management.
- **Member Functions**:
    - [`Request::get_header_value_u64`](#Requestget_header_value_u64)
    - [`Request::has_header`](#Requesthas_header)
    - [`Request::get_header_value`](#Requestget_header_value)
    - [`Request::get_header_value_count`](#Requestget_header_value_count)
    - [`Request::set_header`](#Requestset_header)
    - [`Request::has_param`](#Requesthas_param)
    - [`Request::get_param_value`](#Requestget_param_value)
    - [`Request::get_param_value_count`](#Requestget_param_value_count)
    - [`Request::is_multipart_form_data`](#Requestis_multipart_form_data)
    - [`Request::has_file`](#Requesthas_file)
    - [`Request::get_file_value`](#Requestget_file_value)
    - [`Request::get_file_values`](#Requestget_file_values)

**Methods**

---
#### Request::get\_header\_value\_u64<!-- {{#callable:Request::get_header_value_u64}} -->
The `get_header_value_u64` function retrieves a 64-bit unsigned integer value from the headers of a `Request` object, using a specified key, with a default value and an optional index.
- **Inputs**:
    - `key`: A string representing the key of the header whose value is to be retrieved.
    - `def`: A 64-bit unsigned integer that serves as the default value if the header is not found.
    - `id`: An optional size_t index specifying which occurrence of the header to retrieve if there are multiple headers with the same key.
- **Control Flow**:
    - The function calls `detail::get_header_value_u64` with the `headers` member of the `Request` object, along with the provided `key`, `def`, and `id` arguments.
    - The function directly returns the result of the `detail::get_header_value_u64` call.
- **Output**: A 64-bit unsigned integer representing the value of the specified header, or the default value if the header is not found.
- **See also**: [`Request`](#Request)  (Data Structure)


---
#### Request::has\_header<!-- {{#callable:Request::has_header}} -->
The `has_header` function checks if a specific header key exists in the request's headers.
- **Inputs**:
    - `key`: A string representing the header key to be checked for existence in the request's headers.
- **Control Flow**:
    - The function calls `detail::has_header` with the `headers` member of the `Request` object and the provided `key` as arguments.
    - The result of `detail::has_header` is returned, indicating whether the header key exists.
- **Output**: A boolean value indicating whether the specified header key exists in the request's headers.
- **See also**: [`Request`](#Request)  (Data Structure)


---
#### Request::get\_header\_value<!-- {{#callable:Request::get_header_value}} -->
The `get_header_value` function retrieves the value of a specified header from a request, returning a default value if the header is not found.
- **Inputs**:
    - `key`: A string representing the name of the header to retrieve.
    - `def`: A C-style string representing the default value to return if the header is not found.
    - `id`: A size_t representing the index of the header value to retrieve if multiple values exist for the same header.
- **Control Flow**:
    - The function calls `detail::get_header_value` with the request's headers, the specified key, default value, and index.
    - The function returns the result of the `detail::get_header_value` call.
- **Output**: A string containing the value of the specified header or the default value if the header is not found.
- **See also**: [`Request`](#Request)  (Data Structure)


---
#### Request::get\_header\_value\_count<!-- {{#callable:Request::get_header_value_count}} -->
The `get_header_value_count` function returns the number of values associated with a specific header key in the `headers` map of a `Request` object.
- **Inputs**:
    - `key`: A constant reference to a `std::string` representing the header key whose values are to be counted.
- **Control Flow**:
    - The function uses the `equal_range` method on the `headers` map to get a range of elements that match the given key.
    - It calculates the distance between the first and second iterators of the range, which represents the number of elements with the specified key.
    - The distance is cast to a `size_t` type and returned as the result.
- **Output**: The function returns a `size_t` value representing the count of header values associated with the specified key.
- **See also**: [`Request`](#Request)  (Data Structure)


---
#### Request::set\_header<!-- {{#callable:Request::set_header}} -->
The `set_header` function adds a key-value pair to the headers of a `Request` object if both the key and value are valid according to specific criteria.
- **Inputs**:
    - `key`: A string representing the header name to be added to the request.
    - `val`: A string representing the header value to be associated with the header name.
- **Control Flow**:
    - Check if the provided key is a valid field name using `detail::fields::is_field_name(key)`.
    - Check if the provided value is a valid field value using `detail::fields::is_field_value(val)`.
    - If both checks pass, insert the key-value pair into the `headers` map of the `Request` object.
- **Output**: The function does not return a value; it modifies the `headers` map of the `Request` object in place.
- **See also**: [`Request`](#Request)  (Data Structure)


---
#### Request::has\_param<!-- {{#callable:Request::has_param}} -->
The `has_param` function checks if a specific parameter key exists in the request's parameters.
- **Inputs**:
    - `key`: A constant reference to a string representing the key of the parameter to check for existence in the request's parameters.
- **Control Flow**:
    - The function attempts to find the specified key in the `params` member of the `Request` structure.
    - It returns `true` if the key is found, indicating the parameter exists, otherwise it returns `false`.
- **Output**: A boolean value indicating whether the specified parameter key exists in the request's parameters.
- **See also**: [`Request`](#Request)  (Data Structure)


---
#### Request::get\_param\_value<!-- {{#callable:Request::get_param_value}} -->
The `get_param_value` function retrieves the value of a parameter from a request by its key and index.
- **Inputs**:
    - `key`: A string representing the key of the parameter to retrieve.
    - `id`: A size_t index specifying which occurrence of the parameter to retrieve if multiple exist.
- **Control Flow**:
    - The function uses `params.equal_range(key)` to find the range of elements with the specified key in the `params` multimap.
    - An iterator `it` is initialized to the first element in the range.
    - The iterator `it` is advanced by `id` positions using `std::advance`.
    - If the iterator `it` is not equal to the end of the range, the function returns the value associated with the iterator's current position.
    - If the iterator `it` equals the end of the range, the function returns an empty string.
- **Output**: A string representing the value of the parameter at the specified index, or an empty string if the index is out of range.
- **See also**: [`Request`](#Request)  (Data Structure)


---
#### Request::get\_param\_value\_count<!-- {{#callable:Request::get_param_value_count}} -->
The `get_param_value_count` function returns the number of values associated with a given parameter key in the request's parameters.
- **Inputs**:
    - `key`: A constant reference to a string representing the parameter key whose value count is to be retrieved.
- **Control Flow**:
    - The function uses the `equal_range` method on the `params` member to get a range of iterators that match the given key.
    - It calculates the distance between the first and second iterators of the range using `std::distance`, which gives the count of values associated with the key.
    - The result is cast to `size_t` and returned.
- **Output**: The function returns a `size_t` representing the number of values associated with the specified parameter key.
- **See also**: [`Request`](#Request)  (Data Structure)


---
#### Request::is\_multipart\_form\_data<!-- {{#callable:Request::is_multipart_form_data}} -->
The `is_multipart_form_data` function checks if the request's 'Content-Type' header indicates a 'multipart/form-data' type.
- **Inputs**: None
- **Control Flow**:
    - Retrieve the 'Content-Type' header value using the [`get_header_value`](#get_header_value) method.
    - Check if the 'Content-Type' header starts with 'multipart/form-data' using `rfind`.
    - Return true if 'Content-Type' starts with 'multipart/form-data', otherwise return false.
- **Output**: A boolean value indicating whether the request's 'Content-Type' is 'multipart/form-data'.
- **Functions called**:
    - [`get_header_value`](#get_header_value)
- **See also**: [`Request`](#Request)  (Data Structure)


---
#### Request::has\_file<!-- {{#callable:Request::has_file}} -->
The `has_file` function checks if a file with a specified key exists in the `files` map of a `Request` object.
- **Inputs**:
    - `key`: A string representing the key of the file to check for existence in the `files` map.
- **Control Flow**:
    - The function uses the `find` method of the `files` map to search for the specified key.
    - It compares the result of `find` with `files.end()` to determine if the key exists in the map.
- **Output**: A boolean value indicating whether the file with the specified key exists (`true`) or not (`false`).
- **See also**: [`Request`](#Request)  (Data Structure)


---
#### Request::get\_file\_value<!-- {{#callable:Request::get_file_value}} -->
The `get_file_value` function retrieves a [`MultipartFormData`](#MultipartFormData) object associated with a given key from the `files` map in a `Request` object.
- **Inputs**:
    - `key`: A constant reference to a `std::string` representing the key used to search for the corresponding [`MultipartFormData`](#MultipartFormData) in the `files` map.
- **Control Flow**:
    - The function searches for the provided key in the `files` map using the `find` method.
    - If the key is found, the function returns the associated [`MultipartFormData`](#MultipartFormData) object.
    - If the key is not found, the function returns a default-constructed [`MultipartFormData`](#MultipartFormData) object.
- **Output**: A [`MultipartFormData`](#MultipartFormData) object corresponding to the provided key if found, otherwise a default-constructed [`MultipartFormData`](#MultipartFormData) object.
- **Functions called**:
    - [`MultipartFormData`](#MultipartFormData)
- **See also**: [`Request`](#Request)  (Data Structure)


---
#### Request::get\_file\_values<!-- {{#callable:Request::get_file_values}} -->
The `get_file_values` function retrieves all multipart form data entries associated with a given key from a `Request` object.
- **Inputs**:
    - `key`: A string representing the key for which associated multipart form data entries are to be retrieved.
- **Control Flow**:
    - Initialize an empty vector `values` to store the multipart form data entries.
    - Use the `equal_range` method on the `files` map to get a range of iterators for entries matching the given key.
    - Iterate over the range of iterators from `rng.first` to `rng.second`.
    - For each iterator in the range, push the `second` element (the multipart form data) into the `values` vector.
    - Return the `values` vector containing all the multipart form data entries associated with the key.
- **Output**: A vector of `MultipartFormData` objects containing all entries associated with the specified key.
- **See also**: [`Request`](#Request)  (Data Structure)



---
### Response<!-- {{#data_structure:Response}} -->
- **Type**: `struct`
- **Members**:
    - `version`: Stores the HTTP version of the response.
    - `status`: Holds the HTTP status code, defaulting to -1.
    - `reason`: Contains the reason phrase associated with the status code.
    - `headers`: Represents the HTTP headers of the response.
    - `body`: Holds the body content of the HTTP response.
    - `location`: Specifies the redirect location URL.
    - `content_length_`: Stores the length of the content.
    - `content_provider_`: Holds the content provider function.
    - `content_provider_resource_releaser_`: Manages the release of resources used by the content provider.
    - `is_chunked_content_provider_`: Indicates if the content provider is chunked.
    - `content_provider_success_`: Tracks the success status of the content provider.
    - `file_content_path_`: Stores the file path for file-based content.
    - `file_content_content_type_`: Holds the content type for file-based content.
- **Description**: The `Response` struct is a comprehensive representation of an HTTP response, encapsulating various components such as the HTTP version, status code, reason phrase, headers, and body content. It also manages redirection through the `location` field and supports content provision via different methods, including direct content, file-based content, and chunked content. The struct includes private members to handle content length, content provider functions, and resource management, ensuring efficient handling of HTTP response data.
- **Member Functions**:
    - [`Response::Response`](#ResponseResponse)
    - [`Response::Response`](#ResponseResponse)
    - [`Response::operator=`](#Responseoperator=)
    - [`Response::Response`](#ResponseResponse)
    - [`Response::operator=`](#Responseoperator=)
    - [`Response::~Response`](#ResponseResponse)
    - [`Response::get_header_value_u64`](#Responseget_header_value_u64)
    - [`Response::has_header`](#Responsehas_header)
    - [`Response::get_header_value`](#Responseget_header_value)
    - [`Response::get_header_value_count`](#Responseget_header_value_count)
    - [`Response::set_header`](#Responseset_header)
    - [`Response::set_redirect`](#Responseset_redirect)
    - [`Response::set_content`](#Responseset_content)
    - [`Response::set_content`](#Responseset_content)
    - [`Response::set_content`](#Responseset_content)
    - [`Response::set_content_provider`](#Responseset_content_provider)
    - [`Response::set_content_provider`](#Responseset_content_provider)
    - [`Response::set_chunked_content_provider`](#Responseset_chunked_content_provider)
    - [`Response::set_file_content`](#Responseset_file_content)
    - [`Response::set_file_content`](#Responseset_file_content)

**Methods**

---
#### Response::Response<!-- {{#callable:Response::Response}} -->
The `Response` constructor initializes a new `Response` object with default values or copies an existing `Response` object.
- **Inputs**: None
- **Control Flow**:
    - The default constructor `Response()` initializes a new `Response` object with default values for its members.
    - The copy constructor `Response(const Response &)` creates a new `Response` object by copying the values from an existing `Response` object.
- **Output**: A new `Response` object is created, either with default values or as a copy of an existing object.
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::Response<!-- {{#callable:Response::Response}} -->
The `Response` copy constructor and copy assignment operator are defaulted, allowing for the default behavior of copying all member variables from one `Response` object to another.
- **Inputs**: None
- **Control Flow**:
    - The `Response(const Response &) = default;` line indicates that the copy constructor is defaulted, meaning it will perform a shallow copy of all member variables from the source object to the new object.
    - The `Response &operator=(const Response &) = default;` line indicates that the copy assignment operator is defaulted, meaning it will perform a shallow copy of all member variables from the source object to the target object when assigning one `Response` object to another.
- **Output**: A new `Response` object with member variables copied from the source `Response` object.
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::operator=<!-- {{#callable:Response::operator=}} -->
The `operator=` function is a default copy assignment operator for the `Response` struct, allowing one `Response` object to be assigned to another.
- **Inputs**:
    - `const Response &`: A constant reference to another `Response` object from which data will be copied.
- **Control Flow**:
    - The function is defined as `= default`, which means it uses the compiler-generated default implementation for copying data from the source `Response` object to the target `Response` object.
- **Output**: A reference to the `Response` object that has been assigned the values from another `Response` object.
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::Response<!-- {{#callable:Response::Response}} -->
The move assignment operator for the `Response` class is defined to use the default implementation, allowing for efficient transfer of resources from a temporary `Response` object to another `Response` object.
- **Inputs**: None
- **Control Flow**:
    - The move assignment operator is defined using `= default;`, which means it will use the compiler-generated default implementation.
    - This operator will efficiently transfer resources from a temporary `Response` object to another `Response` object, leaving the temporary object in a valid but unspecified state.
- **Output**: A `Response` object that has been assigned the resources of the temporary `Response` object.
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::operator=<!-- {{#callable:Response::operator=}} -->
The move assignment operator for the `Response` struct is defined as the default move assignment operator, allowing for efficient transfer of resources from a temporary `Response` object to another `Response` object.
- **Inputs**:
    - `Response &&`: A temporary `Response` object from which resources will be moved.
- **Control Flow**:
    - The move assignment operator is defined as `default`, which means it will automatically move all non-static data members from the source object to the target object.
    - The default move assignment operator will transfer ownership of resources, such as dynamically allocated memory, from the source object to the target object, leaving the source object in a valid but unspecified state.
- **Output**: The function returns a reference to the `Response` object that has been assigned the values from the temporary `Response` object.
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::\~Response<!-- {{#callable:Response::~Response}} -->
The destructor of the `Response` struct releases resources associated with the content provider if a resource releaser is set.
- **Inputs**: None
- **Control Flow**:
    - Check if `content_provider_resource_releaser_` is not null.
    - If it is not null, call `content_provider_resource_releaser_` with `content_provider_success_` as the argument.
- **Output**: The destructor does not return any value; it ensures that resources are released properly if a content provider resource releaser is set.
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::get\_header\_value\_u64<!-- {{#callable:Response::get_header_value_u64}} -->
The `get_header_value_u64` function retrieves a 64-bit unsigned integer value from the headers of a `Response` object, using a specified key, with a default value and an optional index.
- **Inputs**:
    - `key`: A string representing the key of the header whose value is to be retrieved.
    - `def`: A 64-bit unsigned integer representing the default value to return if the header is not found.
    - `id`: A size_t index specifying which occurrence of the header to retrieve if there are multiple headers with the same key.
- **Control Flow**:
    - The function calls `detail::get_header_value_u64` with the `headers`, `key`, `def`, and `id` as arguments.
    - The function returns the result of the `detail::get_header_value_u64` call.
- **Output**: A 64-bit unsigned integer representing the value of the specified header, or the default value if the header is not found.
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::has\_header<!-- {{#callable:Response::has_header}} -->
The `has_header` function checks if a specific header key exists in the `headers` map of a `Response` object.
- **Inputs**:
    - `key`: A constant reference to a `std::string` representing the header key to be checked for existence in the `headers` map.
- **Control Flow**:
    - The function attempts to find the specified `key` in the `headers` map using the `find` method.
    - It compares the result of `find` with `headers.end()` to determine if the key exists.
    - If the key is found, the function returns `true`; otherwise, it returns `false`.
- **Output**: A boolean value indicating whether the specified header key exists in the `headers` map.
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::get\_header\_value<!-- {{#callable:Response::get_header_value}} -->
The `get_header_value` function retrieves the value of a specified header from the `Response` object's headers, returning a default value if the header is not found.
- **Inputs**:
    - `key`: A string representing the name of the header whose value is to be retrieved.
    - `def`: A C-style string representing the default value to return if the header is not found; defaults to an empty string.
    - `id`: A size_t representing the index of the header value to retrieve if multiple values exist; defaults to 0.
- **Control Flow**:
    - The function calls `detail::get_header_value` with the `headers`, `key`, `def`, and `id` as arguments.
    - The `detail::get_header_value` function performs the actual retrieval of the header value from the `headers` map.
- **Output**: A `std::string` containing the value of the specified header or the default value if the header is not found.
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::get\_header\_value\_count<!-- {{#callable:Response::get_header_value_count}} -->
The `get_header_value_count` function returns the number of values associated with a specific header key in the `Response` object's headers.
- **Inputs**:
    - `key`: A constant reference to a `std::string` representing the header key whose values are to be counted.
- **Control Flow**:
    - The function uses the `equal_range` method on the `headers` member to get a range of iterators that match the specified key.
    - It calculates the distance between the first and second iterators of the range using `std::distance`, which gives the count of values associated with the key.
    - The result is cast to `size_t` and returned.
- **Output**: The function returns a `size_t` representing the number of values associated with the specified header key.
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::set\_header<!-- {{#callable:Response::set_header}} -->
The `set_header` function adds a key-value pair to the headers of a `Response` object if both the key and value are valid HTTP header fields.
- **Inputs**:
    - `key`: A string representing the header field name to be set.
    - `val`: A string representing the header field value to be set.
- **Control Flow**:
    - Check if the provided key is a valid HTTP header field name using `detail::fields::is_field_name(key)`.
    - Check if the provided value is a valid HTTP header field value using `detail::fields::is_field_value(val)`.
    - If both checks pass, insert the key-value pair into the `headers` map of the `Response` object.
- **Output**: The function does not return a value; it modifies the `headers` member of the `Response` object.
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::set\_redirect<!-- {{#callable:Response::set_redirect}} -->
The `set_redirect` function sets the HTTP response to redirect to a specified URL with a given status code, defaulting to 302 if the provided status code is not a valid redirection code.
- **Inputs**:
    - `url`: A string representing the URL to which the response should redirect.
    - `stat`: An integer representing the HTTP status code for the redirection, defaulting to 302 if not specified.
- **Control Flow**:
    - Check if the provided URL is a valid field value using `detail::fields::is_field_value(url)`.
    - If the URL is valid, set the 'Location' header to the provided URL using `set_header("Location", url)`.
    - Check if the provided status code is between 300 and 399 (inclusive).
    - If the status code is valid, set the response status to the provided status code.
    - If the status code is not valid, set the response status to 302 (Found).
- **Output**: The function does not return a value; it modifies the `Response` object by setting the 'Location' header and the status code.
- **Functions called**:
    - [`Request::set_header`](#Requestset_header)
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::set\_content<!-- {{#callable:Response::set_content}} -->
The `set_content` function sets the body of the response and updates the 'Content-Type' header with the specified content type.
- **Inputs**:
    - `s`: A pointer to a character array representing the content to be set in the response body.
    - `n`: The size of the content to be set, indicating how many characters from the array should be used.
    - `content_type`: A string representing the MIME type of the content being set.
- **Control Flow**:
    - Assigns the content pointed to by `s` with length `n` to the `body` of the response.
    - Finds and removes any existing 'Content-Type' headers from the `headers` map.
    - Sets a new 'Content-Type' header with the provided `content_type`.
- **Output**: This function does not return any value; it modifies the `body` and `headers` of the `Response` object.
- **Functions called**:
    - [`Request::set_header`](#Requestset_header)
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::set\_content<!-- {{#callable:Response::set_content}} -->
The [`set_content`](#Responseset_content) function sets the content of a `Response` object using a string and a specified content type.
- **Inputs**:
    - `s`: A constant reference to a `std::string` representing the content to be set.
    - `content_type`: A constant reference to a `std::string` representing the MIME type of the content.
- **Control Flow**:
    - The function calls another overloaded [`set_content`](#Responseset_content) method, passing the data pointer and size of the string `s`, along with the `content_type`.
- **Output**: This function does not return a value; it modifies the state of the `Response` object by setting its content.
- **Functions called**:
    - [`Response::set_content`](#Responseset_content)
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::set\_content<!-- {{#callable:Response::set_content}} -->
The `set_content` method sets the body of the response and updates the 'Content-Type' header.
- **Inputs**:
    - `s`: A rvalue reference to a `std::string` that represents the content to be set as the body of the response.
    - `content_type`: A constant reference to a `std::string` that specifies the MIME type of the content.
- **Control Flow**:
    - The method assigns the provided string `s` to the `body` member of the `Response` object using `std::move` to efficiently transfer ownership.
    - It retrieves the range of headers with the key 'Content-Type' using `headers.equal_range`.
    - The method erases all existing 'Content-Type' headers within the retrieved range from the `headers` map.
    - It sets a new 'Content-Type' header with the provided `content_type` value by calling [`set_header`](#Requestset_header).
- **Output**: This method does not return any value; it modifies the `Response` object in place.
- **Functions called**:
    - [`Request::set_header`](#Requestset_header)
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::set\_content\_provider<!-- {{#callable:Response::set_content_provider}} -->
The `set_content_provider` function sets up a content provider for a `Response` object, specifying the content length, type, and resource management.
- **Inputs**:
    - `in_length`: The length of the content to be provided, specified as a `size_t`.
    - `content_type`: A `std::string` representing the MIME type of the content.
    - `provider`: A `ContentProvider` object that supplies the content.
    - `resource_releaser`: A `ContentProviderResourceReleaser` object that manages the release of resources associated with the content provider.
- **Control Flow**:
    - Set the 'Content-Type' header of the response to the provided `content_type`.
    - Assign the `in_length` to the `content_length_` member variable.
    - If `in_length` is greater than 0, move the `provider` into the `content_provider_` member variable.
    - Move the `resource_releaser` into the `content_provider_resource_releaser_` member variable.
    - Set `is_chunked_content_provider_` to `false`.
- **Output**: The function does not return a value; it modifies the state of the `Response` object by setting its content provider and related properties.
- **Functions called**:
    - [`Request::set_header`](#Requestset_header)
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::set\_content\_provider<!-- {{#callable:Response::set_content_provider}} -->
The `set_content_provider` method configures a `Response` object to use a content provider without a specified length, setting the content type and managing resource release.
- **Inputs**:
    - `content_type`: A string representing the MIME type of the content to be provided.
    - `provider`: A `ContentProviderWithoutLength` function or object that supplies the content data.
    - `resource_releaser`: A `ContentProviderResourceReleaser` function or object responsible for releasing resources associated with the content provider.
- **Control Flow**:
    - Set the 'Content-Type' header of the response using the provided `content_type`.
    - Initialize `content_length_` to 0, indicating that the content length is unspecified.
    - Assign the `content_provider_` member to a `ContentProviderAdapter` constructed with the given `provider`, adapting it for internal use.
    - Assign the `content_provider_resource_releaser_` member to the provided `resource_releaser`, ensuring resources are released appropriately.
    - Set `is_chunked_content_provider_` to false, indicating that the content is not chunked.
- **Output**: The function does not return a value; it modifies the state of the `Response` object.
- **Functions called**:
    - [`Request::set_header`](#Requestset_header)
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::set\_chunked\_content\_provider<!-- {{#callable:Response::set_chunked_content_provider}} -->
The `set_chunked_content_provider` function configures a `Response` object to use a chunked content provider with a specified content type and optional resource releaser.
- **Inputs**:
    - `content_type`: A string representing the MIME type of the content to be set in the response.
    - `provider`: A `ContentProviderWithoutLength` function that provides the content in chunks.
    - `resource_releaser`: An optional `ContentProviderResourceReleaser` function to release resources after the content has been provided.
- **Control Flow**:
    - Set the 'Content-Type' header of the response to the specified `content_type`.
    - Initialize `content_length_` to 0, indicating that the content length is not predetermined.
    - Wrap the provided `provider` in a `ContentProviderAdapter` and assign it to `content_provider_`.
    - Assign the `resource_releaser` to `content_provider_resource_releaser_`.
    - Set `is_chunked_content_provider_` to true, indicating that the response uses chunked transfer encoding.
- **Output**: The function does not return a value; it modifies the state of the `Response` object to use a chunked content provider.
- **Functions called**:
    - [`Request::set_header`](#Requestset_header)
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::set\_file\_content<!-- {{#callable:Response::set_file_content}} -->
The `set_file_content` function sets the file path and content type for a file to be included in the HTTP response.
- **Inputs**:
    - `path`: A string representing the file path to be set for the response.
    - `content_type`: A string representing the MIME type of the file content.
- **Control Flow**:
    - Assigns the provided `path` to the private member `file_content_path_`.
    - Assigns the provided `content_type` to the private member `file_content_content_type_`.
- **Output**: This function does not return any value.
- **See also**: [`Response`](#Response)  (Data Structure)


---
#### Response::set\_file\_content<!-- {{#callable:Response::set_file_content}} -->
The `set_file_content` function sets the file content path for a `Response` object.
- **Inputs**:
    - `path`: A constant reference to a string representing the file path to be set as the file content path.
- **Control Flow**:
    - Assigns the input `path` to the private member variable `file_content_path_`.
- **Output**: This function does not return any value.
- **See also**: [`Response`](#Response)  (Data Structure)



---
### Stream<!-- {{#data_structure:Stream}} -->
- **Type**: `class`
- **Description**: The `Stream` class is an abstract base class that defines a common interface for stream-like objects, which can be used for reading from and writing to data streams. It includes pure virtual functions for checking readability, waiting for readability and writability, reading and writing data, retrieving remote and local IP addresses and ports, accessing the underlying socket, and getting the duration of the stream. The class also provides overloaded `write` methods for writing data from a character pointer or a string. As an abstract class, `Stream` is intended to be subclassed, with derived classes implementing the specific behavior of these operations.
- **Member Functions**:
    - [`Stream::~Stream`](#StreamStream)
    - [`Stream::write`](#Streamwrite)
    - [`Stream::write`](#Streamwrite)

**Methods**

---
#### Stream::\~Stream<!-- {{#callable:Stream::~Stream}} -->
The `~Stream` function is a virtual destructor for the `Stream` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual to allow derived class destructors to be called when an object is deleted through a base class pointer.
    - The destructor is defined as `default`, indicating that the compiler should generate the default implementation.
- **Output**: There is no output from the destructor itself; it ensures proper resource deallocation for derived classes.
- **See also**: [`Stream`](#Stream)  (Data Structure)


---
#### Stream::write<!-- {{#callable:Stream::write}} -->
The [`write`](#Streamwrite) function writes a null-terminated string to a stream by calling another overloaded [`write`](#Streamwrite) method with the string and its length.
- **Inputs**:
    - `ptr`: A pointer to a null-terminated string that is to be written to the stream.
- **Control Flow**:
    - The function calculates the length of the string using `strlen(ptr)`.
    - It calls the overloaded [`write`](#Streamwrite) method with the string pointer and its length as arguments.
- **Output**: The function returns the result of the overloaded [`write`](#Streamwrite) method, which is of type `ssize_t`, indicating the number of bytes written or an error code.
- **Functions called**:
    - [`Stream::write`](#Streamwrite)
- **See also**: [`Stream`](#Stream)  (Data Structure)


---
#### Stream::write<!-- {{#callable:Stream::write}} -->
The [`write`](#Streamwrite) method writes the contents of a given string to the stream by calling another [`write`](#Streamwrite) method with the string's data and size.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that contains the data to be written to the stream.
- **Control Flow**:
    - The method takes a `std::string` as input.
    - It calls another [`write`](#Streamwrite) method, passing the string's data pointer and its size as arguments.
    - The method returns the result of the called [`write`](#Streamwrite) method.
- **Output**: The method returns a `ssize_t` value, which is typically the number of bytes written, or a negative value if an error occurs.
- **Functions called**:
    - [`Stream::write`](#Streamwrite)
- **See also**: [`Stream`](#Stream)  (Data Structure)



---
### TaskQueue<!-- {{#data_structure:TaskQueue}} -->
- **Type**: `class`
- **Description**: The `TaskQueue` class is an abstract base class that defines an interface for a task queue system. It provides a virtual destructor and three virtual methods: `enqueue`, which is intended to add a task to the queue; `shutdown`, which is meant to handle the shutdown process of the queue; and `on_idle`, which can be overridden to define behavior when the queue is idle. The class does not contain any data members, indicating that it serves as a pure interface for derived classes to implement specific task queue functionalities.
- **Member Functions**:
    - [`TaskQueue::TaskQueue`](#TaskQueueTaskQueue)
    - [`TaskQueue::~TaskQueue`](#TaskQueueTaskQueue)
    - [`TaskQueue::on_idle`](#TaskQueueon_idle)

**Methods**

---
#### TaskQueue::TaskQueue<!-- {{#callable:TaskQueue::TaskQueue}} -->
The TaskQueue class provides an interface for managing a queue of tasks, with methods to enqueue tasks, shut down the queue, and handle idle states.
- **Inputs**: None
- **Control Flow**:
    - The constructor `TaskQueue()` is defined as default, meaning it performs no special initialization.
    - The destructor `~TaskQueue()` is also default, indicating no special cleanup is required.
    - The class declares a pure virtual method `enqueue` which takes a `std::function<void()>` as an argument, indicating that derived classes must implement this method to add tasks to the queue.
    - Another pure virtual method `shutdown` is declared, requiring derived classes to implement logic for shutting down the task queue.
    - An optional virtual method `on_idle` is provided, which can be overridden by derived classes to define behavior when the queue is idle.
- **Output**: The class itself does not produce any output, but it defines an interface for task queue management that derived classes must implement.
- **See also**: [`TaskQueue`](#TaskQueue)  (Data Structure)


---
#### TaskQueue::\~TaskQueue<!-- {{#callable:TaskQueue::~TaskQueue}} -->
The `~TaskQueue` function is a virtual destructor for the `TaskQueue` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual to allow derived class destructors to be called correctly when an object is deleted through a base class pointer.
    - The destructor is defined as `default`, indicating that the compiler should generate the default implementation for it.
- **Output**: There is no explicit output from the destructor; it ensures proper resource cleanup when a `TaskQueue` object or its derived class object is destroyed.
- **See also**: [`TaskQueue`](#TaskQueue)  (Data Structure)


---
#### TaskQueue::on\_idle<!-- {{#callable:TaskQueue::on_idle}} -->
The `on_idle` method is a virtual function in the `TaskQueue` class that is intended to be overridden to define behavior when the task queue is idle.
- **Inputs**: None
- **Control Flow**:
    - The `on_idle` method is defined as a virtual function, allowing derived classes to override it.
    - The method is empty, indicating no default behavior is provided for when the task queue is idle.
- **Output**: The method does not return any value or output.
- **See also**: [`TaskQueue`](#TaskQueue)  (Data Structure)



---
### ThreadPool<!-- {{#data_structure:ThreadPool}} -->
- **Type**: `class`
- **Members**:
    - `threads_`: A vector that holds the worker threads of the thread pool.
    - `jobs_`: A list that stores the queued tasks to be executed by the thread pool.
    - `shutdown_`: A boolean flag indicating whether the thread pool is in the process of shutting down.
    - `max_queued_requests_`: A size_t value representing the maximum number of tasks that can be queued.
    - `cond_`: A condition variable used to synchronize access to the task queue.
    - `mutex_`: A mutex used to protect access to shared resources within the thread pool.
- **Description**: The `ThreadPool` class is a thread management utility that inherits from `TaskQueue` and manages a pool of worker threads to execute tasks concurrently. It allows tasks to be enqueued for execution, with an optional limit on the number of queued tasks. The class handles synchronization using a mutex and condition variable, and provides a mechanism to gracefully shut down the pool by joining all threads. The `worker` struct within the class defines the behavior of each thread, which continuously processes tasks from the queue until a shutdown is initiated.
- **Member Functions**:
    - [`ThreadPool::ThreadPool`](#ThreadPoolThreadPool)
    - [`ThreadPool::ThreadPool`](#ThreadPoolThreadPool)
    - [`ThreadPool::~ThreadPool`](#ThreadPoolThreadPool)
    - [`ThreadPool::enqueue`](#ThreadPoolenqueue)
    - [`ThreadPool::shutdown`](#ThreadPoolshutdown)
- **Inherits From**:
    - [`TaskQueue::TaskQueue`](#TaskQueueTaskQueue)

**Methods**

---
#### ThreadPool::ThreadPool<!-- {{#callable:ThreadPool::ThreadPool}} -->
The `ThreadPool` constructor initializes a thread pool with a specified number of worker threads and an optional maximum queue size for tasks.
- **Inputs**:
    - `n`: The number of worker threads to create in the thread pool.
    - `mqr`: The maximum number of queued requests allowed; defaults to 0, which means no limit.
- **Control Flow**:
    - Initialize the `shutdown_` flag to `false` and set `max_queued_requests_` to the provided `mqr` value.
    - Enter a loop that runs `n` times, each time creating a new [`worker`](#workerworker) thread and adding it to the `threads_` vector.
    - Decrement `n` until it reaches zero, indicating all threads have been created.
- **Output**: A `ThreadPool` object with `n` worker threads and a task queue with an optional maximum size.
- **Functions called**:
    - [`ThreadPool::worker::worker`](#workerworker)
- **See also**: [`ThreadPool`](#ThreadPool)  (Data Structure)


---
#### ThreadPool::ThreadPool<!-- {{#callable:ThreadPool::ThreadPool}} -->
The `ThreadPool` constructor initializes a thread pool with a specified number of worker threads and an optional maximum queue size for tasks.
- **Inputs**:
    - `n`: The number of worker threads to create in the thread pool.
    - `mqr`: The maximum number of queued requests allowed; defaults to 0, which means no limit.
- **Control Flow**:
    - The constructor initializes the `shutdown_` flag to `false` and sets `max_queued_requests_` to the provided value `mqr`.
    - A loop runs `n` times, each time creating a new `worker` thread and adding it to the `threads_` vector.
    - Each `worker` thread is initialized with a reference to the `ThreadPool` instance.
- **Output**: The constructor does not return a value; it initializes the `ThreadPool` object.
- **See also**: [`ThreadPool`](#ThreadPool)  (Data Structure)


---
#### ThreadPool::\~ThreadPool<!-- {{#callable:ThreadPool::~ThreadPool}} -->
The destructor `~ThreadPool()` is a default destructor for the `ThreadPool` class, which is responsible for cleaning up resources when a `ThreadPool` object is destroyed.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as `override = default`, indicating it uses the compiler-generated default destructor.
    - It does not contain any custom logic or resource deallocation code, relying on the default behavior to clean up resources.
- **Output**: The destructor does not return any value or output, as it is responsible for cleanup during object destruction.
- **See also**: [`ThreadPool`](#ThreadPool)  (Data Structure)


---
#### ThreadPool::enqueue<!-- {{#callable:ThreadPool::enqueue}} -->
The `enqueue` function adds a new task to the job queue of a `ThreadPool` if the queue is not full, and notifies a worker thread to process it.
- **Inputs**:
    - `fn`: A `std::function<void()>` representing the task to be added to the job queue.
- **Control Flow**:
    - Acquire a unique lock on the mutex to ensure thread-safe access to the job queue.
    - Check if the maximum number of queued requests is greater than zero and if the current size of the job queue is greater than or equal to this maximum.
    - If the queue is full, return `false` to indicate the task was not added.
    - If the queue is not full, move the task into the job queue.
    - Release the lock and notify one worker thread that a new task is available.
    - Return `true` to indicate the task was successfully added to the queue.
- **Output**: A boolean value indicating whether the task was successfully added to the job queue (`true`) or not (`false`).
- **See also**: [`ThreadPool`](#ThreadPool)  (Data Structure)


---
#### ThreadPool::shutdown<!-- {{#callable:ThreadPool::shutdown}} -->
The `shutdown` function stops all worker threads in the `ThreadPool` by setting a shutdown flag, notifying all threads, and then joining them.
- **Inputs**: None
- **Control Flow**:
    - Acquire a unique lock on the mutex to ensure thread-safe access to shared resources.
    - Set the `shutdown_` flag to `true` to indicate that the thread pool is shutting down.
    - Notify all threads waiting on the condition variable `cond_` to wake them up.
    - Iterate over each thread in the `threads_` vector and call `join()` on them to wait for their completion.
- **Output**: The function does not return any value.
- **See also**: [`ThreadPool`](#ThreadPool)  (Data Structure)



---
### worker<!-- {{#data_structure:ThreadPool::worker}} -->
- **Type**: `struct`
- **Members**:
    - `pool_`: A reference to a ThreadPool object that the worker is associated with.
- **Description**: The `worker` struct is designed to operate within a thread pool, managing the execution of tasks from a shared job queue. It contains a single member, `pool_`, which is a reference to a `ThreadPool` object. The `worker` struct's primary function is to continuously fetch and execute tasks from the `ThreadPool`'s job queue until a shutdown condition is met. It uses a mutex and condition variable to safely access and modify the shared job queue, ensuring thread-safe operations. The struct also includes a mechanism to handle OpenSSL thread cleanup if certain conditions are met.
- **Member Functions**:
    - [`ThreadPool::worker::worker`](#workerworker)
    - [`ThreadPool::worker::operator()`](#workeroperator())

**Methods**

---
#### worker::worker<!-- {{#callable:ThreadPool::worker::worker}} -->
The `worker` function is an operator overload that continuously processes jobs from a thread pool until shutdown is signaled.
- **Inputs**:
    - `None`: The function does not take any direct input parameters when called as it is an operator overload.
- **Control Flow**:
    - The function enters an infinite loop to continuously process jobs.
    - A unique lock is acquired on the thread pool's mutex to ensure thread-safe access to shared resources.
    - The function waits on a condition variable until there are jobs available or a shutdown is signaled.
    - If a shutdown is signaled and there are no jobs left, the loop breaks, ending the worker's execution.
    - The next job is retrieved from the front of the job queue and removed from the queue.
    - An assertion checks that the job function is valid before executing it.
    - The job function is executed.
    - If OpenSSL support is enabled and certain conditions are met, the OpenSSL thread cleanup function is called.
- **Output**: The function does not return any value; it continuously processes jobs until a shutdown is signaled.
- **See also**: [`ThreadPool::worker`](#ThreadPool::worker)  (Data Structure)


---
#### worker::operator\(\)<!-- {{#callable:ThreadPool::worker::operator()}} -->
The `operator()` function in the `worker` struct continuously processes jobs from a thread pool until shutdown is signaled.
- **Inputs**: None
- **Control Flow**:
    - Enter an infinite loop to continuously process jobs.
    - Acquire a lock on the thread pool's mutex to ensure thread-safe access to shared resources.
    - Wait on the condition variable until there are jobs to process or a shutdown is signaled.
    - If shutdown is signaled and there are no jobs left, break out of the loop.
    - Retrieve the next job from the front of the job queue and remove it from the queue.
    - Assert that the job function is valid and then execute it.
    - If OpenSSL support is enabled and certain conditions are met, call `OPENSSL_thread_stop()` to clean up OpenSSL thread resources.
- **Output**: The function does not return any value; it processes jobs from the thread pool until shutdown.
- **See also**: [`ThreadPool::worker`](#ThreadPool::worker)  (Data Structure)



---
### MatcherBase<!-- {{#data_structure:detail::MatcherBase}} -->
- **Type**: `class`
- **Description**: The `MatcherBase` class is an abstract base class that provides an interface for matching request paths. It contains a pure virtual function `match` that must be implemented by derived classes to perform the matching logic on a `Request` object. The class also includes a virtual destructor, ensuring proper cleanup of derived class objects when deleted through a base class pointer.
- **Member Functions**:
    - [`detail::MatcherBase::~MatcherBase`](#MatcherBaseMatcherBase)

**Methods**

---
#### MatcherBase::\~MatcherBase<!-- {{#callable:detail::MatcherBase::~MatcherBase}} -->
The `~MatcherBase` function is a virtual destructor for the `MatcherBase` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The function is declared as virtual, which allows derived class destructors to be called correctly when an object is deleted through a base class pointer.
    - The function is defined as default, indicating that the compiler should generate the default implementation for the destructor.
- **Output**: The function does not produce any output as it is a destructor.
- **See also**: [`detail::MatcherBase`](#detailMatcherBase)  (Data Structure)



---
### PathParamsMatcher<!-- {{#data_structure:detail::PathParamsMatcher}} -->
- **Type**: `class`
- **Members**:
    - `static_fragments_`: A vector containing static path fragments to match against, excluding the '/' after path parameters.
    - `param_names_`: A vector storing the names of path parameters to be used as keys in the Request::path_params map.
- **Description**: The `PathParamsMatcher` class is a specialized matcher that extends `MatcherBase` to handle path parameter matching in URLs. It uses a pattern to identify and match static path fragments and path parameters, which are stored in `static_fragments_` and `param_names_` respectively. The class is designed to treat segment separators as the end of path parameter capture and does not handle query parameters, as they are parsed before path matching. This class is crucial for routing requests based on URL paths in web applications.
- **Member Functions**:
    - [`detail::PathParamsMatcher::PathParamsMatcher`](#PathParamsMatcherPathParamsMatcher)
    - [`detail::PathParamsMatcher::match`](#PathParamsMatchermatch)
- **Inherits From**:
    - [`detail::MatcherBase`](#detailMatcherBase)

**Methods**

---
#### PathParamsMatcher::PathParamsMatcher<!-- {{#callable:detail::PathParamsMatcher::PathParamsMatcher}} -->
The `PathParamsMatcher` constructor initializes a matcher object by parsing a given URL pattern to identify static path fragments and path parameter names, ensuring parameter names are unique.
- **Inputs**:
    - `pattern`: A string representing the URL pattern to be parsed for static fragments and path parameters.
- **Control Flow**:
    - Initialize a constant marker string `/:` to identify path parameters in the pattern.
    - Set `last_param_end` to track the end position of the last path parameter substring.
    - If exceptions are enabled, initialize a set to track unique parameter names.
    - Enter a loop to find occurrences of the marker in the pattern starting from `last_param_end`.
    - If no marker is found, break the loop.
    - Extract and store the static fragment preceding the marker in `static_fragments_`.
    - Determine the start position of the parameter name after the marker.
    - Find the next separator or end of the pattern to determine the end of the parameter name.
    - Extract the parameter name and check for uniqueness if exceptions are enabled, throwing an error if a duplicate is found.
    - Store the parameter name in `param_names_`.
    - Update `last_param_end` to the position after the separator.
    - After the loop, if there is any remaining static fragment, store it in `static_fragments_`.
- **Output**: The constructor does not return a value; it initializes the `PathParamsMatcher` object with parsed static fragments and parameter names.
- **Functions called**:
    - [`detail::str_len`](#detailstr_len)
- **See also**: [`detail::PathParamsMatcher`](#detailPathParamsMatcher)  (Data Structure)


---
#### PathParamsMatcher::match<!-- {{#callable:detail::PathParamsMatcher::match}} -->
The `match` function checks if a given request path matches a predefined pattern and extracts path parameters into the request object.
- **Inputs**:
    - `request`: A reference to a Request object that contains the path to be matched and will store the extracted path parameters.
- **Control Flow**:
    - Initialize the `matches` member of the request to an empty `std::smatch` object and clear the `path_params` map.
    - Reserve space in `path_params` for the number of parameter names.
    - Iterate over each static fragment in `static_fragments_`.
    - Check if the current fragment can fit in the remaining part of the request path; return false if it cannot.
    - Compare the current fragment with the corresponding part of the request path using `strncmp`; return false if they do not match.
    - Update the starting position to the end of the matched fragment.
    - If there are more static fragments than parameter names, continue to the next iteration.
    - Find the next separator in the request path to determine the end of the current path parameter value.
    - Extract the path parameter value and store it in `path_params` using the corresponding parameter name.
    - Update the starting position to just after the separator.
    - Return false if the path is longer than the pattern; otherwise, return true.
- **Output**: A boolean value indicating whether the request path matches the pattern and all path parameters have been successfully extracted.
- **See also**: [`detail::PathParamsMatcher`](#detailPathParamsMatcher)  (Data Structure)



---
### RegexMatcher<!-- {{#data_structure:detail::RegexMatcher}} -->
- **Type**: `class`
- **Members**:
    - `regex_`: A private member of type std::regex that stores the compiled regular expression pattern.
- **Description**: The `RegexMatcher` class is a specialized matcher that inherits from `MatcherBase` and is designed to perform regular expression matching on requests. It encapsulates a `std::regex` object, which is initialized with a pattern provided at construction. The class provides a `match` function that overrides a base class method to determine if a given request matches the stored regular expression pattern.
- **Member Functions**:
    - [`detail::RegexMatcher::RegexMatcher`](#RegexMatcherRegexMatcher)
    - [`detail::RegexMatcher::match`](#RegexMatchermatch)
- **Inherits From**:
    - [`detail::MatcherBase`](#detailMatcherBase)

**Methods**

---
#### RegexMatcher::RegexMatcher<!-- {{#callable:detail::RegexMatcher::RegexMatcher}} -->
The `RegexMatcher` constructor initializes a `RegexMatcher` object with a given regex pattern.
- **Inputs**:
    - `pattern`: A constant reference to a `std::string` representing the regex pattern to be used for matching.
- **Control Flow**:
    - The constructor takes a `std::string` reference as an argument.
    - It initializes the private member `regex_` with the provided pattern using the `std::regex` constructor.
- **Output**: There is no output from the constructor; it initializes the `regex_` member variable.
- **See also**: [`detail::RegexMatcher`](#detailRegexMatcher)  (Data Structure)


---
#### RegexMatcher::match<!-- {{#callable:detail::RegexMatcher::match}} -->
The `match` function checks if a request's path matches a predefined regex pattern and clears any existing path parameters.
- **Inputs**:
    - `request`: A reference to a `Request` object that contains the path to be matched and stores the match results.
- **Control Flow**:
    - Clear the `path_params` of the `request` object to remove any previous path parameters.
    - Use `std::regex_match` to check if the `request.path` matches the `regex_` pattern and store the results in `request.matches`.
- **Output**: A boolean value indicating whether the `request.path` matches the regex pattern.
- **See also**: [`detail::RegexMatcher`](#detailRegexMatcher)  (Data Structure)



---
### Server<!-- {{#data_structure:Server}} -->
- **Type**: `class`
- **Members**:
    - `svr_sock_`: An atomic socket variable used to store the server socket.
    - `keep_alive_max_count_`: Stores the maximum number of keep-alive requests allowed.
    - `keep_alive_timeout_sec_`: Stores the keep-alive timeout in seconds.
    - `keep_alive_timeout_usec_`: Stores the keep-alive timeout in microseconds.
    - `read_timeout_sec_`: Stores the read timeout in seconds.
    - `read_timeout_usec_`: Stores the read timeout in microseconds.
    - `write_timeout_sec_`: Stores the write timeout in seconds.
    - `write_timeout_usec_`: Stores the write timeout in microseconds.
    - `idle_interval_sec_`: Stores the idle interval in seconds.
    - `idle_interval_usec_`: Stores the idle interval in microseconds.
    - `payload_max_length_`: Stores the maximum length of the payload.
    - `is_running_`: An atomic boolean indicating if the server is currently running.
    - `is_decommissioned`: An atomic boolean indicating if the server has been decommissioned.
    - `base_dirs_`: A vector of MountPointEntry structs representing base directories and their mount points.
    - `file_extension_and_mimetype_map_`: A map associating file extensions with their MIME types.
    - `default_file_mimetype_`: Stores the default MIME type for files.
    - `file_request_handler_`: A handler for file requests.
    - `get_handlers_`: A collection of handlers for GET requests.
    - `post_handlers_`: A collection of handlers for POST requests.
    - `post_handlers_for_content_reader_`: A collection of handlers for POST requests that include a content reader.
    - `put_handlers_`: A collection of handlers for PUT requests.
    - `put_handlers_for_content_reader_`: A collection of handlers for PUT requests that include a content reader.
    - `patch_handlers_`: A collection of handlers for PATCH requests.
    - `patch_handlers_for_content_reader_`: A collection of handlers for PATCH requests that include a content reader.
    - `delete_handlers_`: A collection of handlers for DELETE requests.
    - `delete_handlers_for_content_reader_`: A collection of handlers for DELETE requests that include a content reader.
    - `options_handlers_`: A collection of handlers for OPTIONS requests.
    - `error_handler_`: A handler for errors that occur during request processing.
    - `exception_handler_`: A handler for exceptions that occur during request processing.
    - `pre_routing_handler_`: A handler executed before routing the request.
    - `post_routing_handler_`: A handler executed after routing the request.
    - `expect_100_continue_handler_`: A handler for managing 'Expect: 100-continue' headers.
    - `logger_`: A logger for recording server events.
    - `address_family_`: Stores the address family for the server socket.
    - `tcp_nodelay_`: A boolean indicating if TCP_NODELAY is enabled.
    - `ipv6_v6only_`: A boolean indicating if the server is restricted to IPv6 only.
    - `socket_options_`: Stores socket options for the server.
    - `default_headers_`: Stores default headers to be included in responses.
    - `header_writer_`: A function for writing headers to a stream.
- **Description**: The `Server` class is a comprehensive HTTP server implementation that provides a wide range of functionalities for handling HTTP requests and responses. It supports various HTTP methods such as GET, POST, PUT, PATCH, DELETE, and OPTIONS, each with customizable handlers. The class includes mechanisms for setting up file serving, error handling, and exception management. It also allows configuration of server parameters like timeouts, socket options, and MIME type mappings. The server can manage multiple connections with keep-alive support and is designed to be extensible with custom handlers for pre-routing, post-routing, and specific HTTP headers. The `Server` class is equipped with logging capabilities and can be configured to operate with different network protocols and address families.
- **Member Functions**:
    - [`Server::set_error_handler`](#Serverset_error_handler)
    - [`Server::set_read_timeout`](#Serverset_read_timeout)
    - [`Server::set_write_timeout`](#Serverset_write_timeout)
    - [`Server::set_idle_interval`](#Serverset_idle_interval)
    - [`Server::Server`](#ServerServer)
    - [`Server::~Server`](#ServerServer)
    - [`Server::make_matcher`](#Servermake_matcher)
    - [`Server::Get`](#ServerGet)
    - [`Server::Post`](#ServerPost)
    - [`Server::Post`](#ServerPost)
    - [`Server::Put`](#ServerPut)
    - [`Server::Put`](#ServerPut)
    - [`Server::Patch`](#ServerPatch)
    - [`Server::Patch`](#ServerPatch)
    - [`Server::Delete`](#ServerDelete)
    - [`Server::Delete`](#ServerDelete)
    - [`Server::Options`](#ServerOptions)
    - [`Server::set_base_dir`](#Serverset_base_dir)
    - [`Server::set_mount_point`](#Serverset_mount_point)
    - [`Server::remove_mount_point`](#Serverremove_mount_point)
    - [`Server::set_file_extension_and_mimetype_mapping`](#Serverset_file_extension_and_mimetype_mapping)
    - [`Server::set_default_file_mimetype`](#Serverset_default_file_mimetype)
    - [`Server::set_file_request_handler`](#Serverset_file_request_handler)
    - [`Server::set_error_handler_core`](#Serverset_error_handler_core)
    - [`Server::set_error_handler_core`](#Serverset_error_handler_core)
    - [`Server::set_exception_handler`](#Serverset_exception_handler)
    - [`Server::set_pre_routing_handler`](#Serverset_pre_routing_handler)
    - [`Server::set_post_routing_handler`](#Serverset_post_routing_handler)
    - [`Server::set_logger`](#Serverset_logger)
    - [`Server::set_expect_100_continue_handler`](#Serverset_expect_100_continue_handler)
    - [`Server::set_address_family`](#Serverset_address_family)
    - [`Server::set_tcp_nodelay`](#Serverset_tcp_nodelay)
    - [`Server::set_ipv6_v6only`](#Serverset_ipv6_v6only)
    - [`Server::set_socket_options`](#Serverset_socket_options)
    - [`Server::set_default_headers`](#Serverset_default_headers)
    - [`Server::set_header_writer`](#Serverset_header_writer)
    - [`Server::set_keep_alive_max_count`](#Serverset_keep_alive_max_count)
    - [`Server::set_keep_alive_timeout`](#Serverset_keep_alive_timeout)
    - [`Server::set_read_timeout`](#Serverset_read_timeout)
    - [`Server::set_write_timeout`](#Serverset_write_timeout)
    - [`Server::set_idle_interval`](#Serverset_idle_interval)
    - [`Server::set_payload_max_length`](#Serverset_payload_max_length)
    - [`Server::bind_to_port`](#Serverbind_to_port)
    - [`Server::bind_to_any_port`](#Serverbind_to_any_port)
    - [`Server::listen_after_bind`](#Serverlisten_after_bind)
    - [`Server::listen`](#Serverlisten)
    - [`Server::is_running`](#Serveris_running)
    - [`Server::wait_until_ready`](#Serverwait_until_ready)
    - [`Server::stop`](#Serverstop)
    - [`Server::decommission`](#Serverdecommission)
    - [`Server::parse_request_line`](#Serverparse_request_line)
    - [`Server::write_response`](#Serverwrite_response)
    - [`Server::write_response_with_content`](#Serverwrite_response_with_content)
    - [`Server::write_response_core`](#Serverwrite_response_core)
    - [`Server::write_content_with_provider`](#Serverwrite_content_with_provider)
    - [`Server::read_content`](#Serverread_content)
    - [`Server::read_content_with_content_receiver`](#Serverread_content_with_content_receiver)
    - [`Server::read_content_core`](#Serverread_content_core)
    - [`Server::handle_file_request`](#Serverhandle_file_request)
    - [`Server::create_server_socket`](#Servercreate_server_socket)
    - [`Server::bind_internal`](#Serverbind_internal)
    - [`Server::routing`](#Serverrouting)
    - [`Server::dispatch_request`](#Serverdispatch_request)
    - [`Server::apply_ranges`](#Serverapply_ranges)
    - [`Server::dispatch_request_for_content_reader`](#Serverdispatch_request_for_content_reader)
    - [`Server::process_request`](#Serverprocess_request)
    - [`Server::is_valid`](#Serveris_valid)
    - [`Server::process_and_close_socket`](#Serverprocess_and_close_socket)

**Methods**

---
#### Server::set\_error\_handler<!-- {{#callable:Server::set_error_handler}} -->
Sets a custom error handler for the `Server`.
- **Inputs**:
    - `handler`: A callable object (function, lambda, etc.) that defines how to handle errors, which can either be a simple handler or one that returns a `HandlerResponse`.
- **Control Flow**:
    - The function forwards the `handler` to another function [`set_error_handler_core`](#Serverset_error_handler_core).
    - It uses `std::is_convertible` to determine if the provided `handler` can be treated as a `HandlerWithResponse`.
    - Depending on the result of the conversion check, it calls the appropriate overload of [`set_error_handler_core`](#Serverset_error_handler_core).
- **Output**: Returns a reference to the `Server` instance, allowing for method chaining.
- **Functions called**:
    - [`Server::set_error_handler_core`](#Serverset_error_handler_core)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_read\_timeout<!-- {{#callable:Server::set_read_timeout}} -->
Sets the read timeout for the `Server` instance.
- **Inputs**:
    - `duration`: A `std::chrono::duration` object representing the desired read timeout duration.
- **Control Flow**:
    - Calls the `detail::duration_to_sec_and_usec` function to convert the provided duration into seconds and microseconds.
    - Uses a lambda function to pass the converted seconds and microseconds to the [`set_read_timeout`](#ClientImplset_read_timeout) method of the `Server` class.
- **Output**: Returns a reference to the current `Server` instance, allowing for method chaining.
- **Functions called**:
    - [`ClientImpl::set_read_timeout`](#ClientImplset_read_timeout)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_write\_timeout<!-- {{#callable:Server::set_write_timeout}} -->
Sets the write timeout for the `Server` instance using a specified duration.
- **Inputs**:
    - `duration`: A `std::chrono::duration` object representing the desired write timeout duration.
- **Control Flow**:
    - The function calls `detail::duration_to_sec_and_usec`, passing the `duration` and a lambda function.
    - The lambda function captures the seconds and microseconds and calls the [`set_write_timeout`](#ClientImplset_write_timeout) method with these values.
    - Finally, it returns a reference to the current `Server` instance.
- **Output**: Returns a reference to the current `Server` instance, allowing for method chaining.
- **Functions called**:
    - [`ClientImpl::set_write_timeout`](#ClientImplset_write_timeout)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_idle\_interval<!-- {{#callable:Server::set_idle_interval}} -->
Sets the idle interval for the `Server` instance using a specified duration.
- **Inputs**:
    - `duration`: A `std::chrono::duration` object representing the idle interval to be set, which can be specified in various time units.
- **Control Flow**:
    - The function calls `detail::duration_to_sec_and_usec`, passing the `duration` and a lambda function.
    - The lambda function captures the seconds and microseconds from the duration and calls the [`set_idle_interval`](#Serverset_idle_interval) method with these values.
    - Finally, it returns a reference to the current `Server` instance.
- **Output**: Returns a reference to the `Server` instance, allowing for method chaining.
- **Functions called**:
    - [`Server::set_idle_interval`](#Serverset_idle_interval)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::Server<!-- {{#callable:Server::Server}} -->
Constructs a `Server` instance and initializes a thread pool for handling tasks.
- **Inputs**:
    - `none`: The constructor does not take any input arguments.
- **Control Flow**:
    - Initializes the `new_task_queue` member with a lambda function that creates a new `ThreadPool` instance with a predefined thread count.
    - On non-Windows platforms, it sets the signal handler for `SIGPIPE` to ignore the signal.
- **Output**: The constructor does not return a value but initializes the `Server` object for further use.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::make\_matcher<!-- {{#callable:Server::make_matcher}} -->
Creates a matcher based on the provided pattern, either as a path parameter matcher or a regex matcher.
- **Inputs**:
    - `pattern`: A string representing the pattern used to create the matcher.
- **Control Flow**:
    - Checks if the `pattern` contains the substring '/:'.
    - If found, it creates and returns a `PathParamsMatcher` using the `pattern`.
    - If not found, it creates and returns a `RegexMatcher` using the `pattern`.
- **Output**: Returns a unique pointer to a `MatcherBase` object, which is either a `PathParamsMatcher` or a `RegexMatcher` depending on the input pattern.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::Get<!-- {{#callable:Server::Get}} -->
Registers a GET request handler with a specified URL pattern.
- **Inputs**:
    - `pattern`: A string representing the URL pattern that the handler will respond to.
    - `handler`: A function that takes a `Request` and a `Response` object, defining the logic to execute when the pattern is matched.
- **Control Flow**:
    - The function creates a matcher for the provided `pattern` using the [`make_matcher`](#Servermake_matcher) function.
    - It then stores the matcher and the associated `handler` in the `get_handlers_` vector.
    - Finally, it returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling further configuration or chaining of method calls.
- **Functions called**:
    - [`Server::make_matcher`](#Servermake_matcher)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::Post<!-- {{#callable:Server::Post}} -->
Registers a `POST` request handler for a specified URL pattern.
- **Inputs**:
    - `pattern`: A string representing the URL pattern that the handler will respond to.
    - `handler`: A function that takes a `Request` and a `Response` object, defining the logic to execute when the pattern is matched.
- **Control Flow**:
    - The function uses [`make_matcher`](#Servermake_matcher) to create a matcher object based on the provided `pattern`.
    - The matcher and the `handler` are then stored in the `post_handlers_` vector as a pair.
    - Finally, the function returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling further configuration or method chaining.
- **Functions called**:
    - [`Server::make_matcher`](#Servermake_matcher)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::Post<!-- {{#callable:Server::Post}} -->
Registers a POST request handler with a specified pattern and content reader.
- **Inputs**:
    - `pattern`: A string representing the URL pattern that the handler will respond to.
    - `handler`: A function that takes a `Request`, `Response`, and a `ContentReader`, which processes the request and generates a response.
- **Control Flow**:
    - The function creates a matcher for the provided `pattern` using [`make_matcher`](#Servermake_matcher).
    - It then stores the matcher and the `handler` in the `post_handlers_for_content_reader_` vector.
    - Finally, it returns a reference to the current `Server` instance to allow for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling method chaining for further configuration.
- **Functions called**:
    - [`Server::make_matcher`](#Servermake_matcher)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::Put<!-- {{#callable:Server::Put}} -->
Registers a `PUT` request handler for a specified URL pattern.
- **Inputs**:
    - `pattern`: A string representing the URL pattern that the handler will respond to.
    - `handler`: A function that takes a `Request` and a `Response` object, defining the logic to execute when the pattern is matched.
- **Control Flow**:
    - The method uses [`make_matcher`](#Servermake_matcher) to create a matcher object for the provided `pattern`.
    - The matcher and the `handler` are then stored in the `put_handlers_` vector as a pair.
    - The method returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling fluent interface for further configuration.
- **Functions called**:
    - [`Server::make_matcher`](#Servermake_matcher)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::Put<!-- {{#callable:Server::Put}} -->
Registers a `PUT` request handler for a specified URL pattern.
- **Inputs**:
    - `pattern`: A string representing the URL pattern that the handler will respond to.
    - `handler`: A function that processes the request and response, including a content reader.
- **Control Flow**:
    - The function creates a matcher for the provided `pattern` using [`make_matcher`](#Servermake_matcher).
    - It then stores the matcher and the `handler` in the `put_handlers_for_content_reader_` vector.
    - Finally, it returns a reference to the current `Server` instance.
- **Output**: Returns a reference to the `Server` instance, allowing for method chaining.
- **Functions called**:
    - [`Server::make_matcher`](#Servermake_matcher)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::Patch<!-- {{#callable:Server::Patch}} -->
Registers a `PATCH` request handler for a specified URL pattern.
- **Inputs**:
    - `pattern`: A string representing the URL pattern that the handler will respond to.
    - `handler`: A function that takes a `Request` and a `Response` object, defining the logic to execute when the pattern is matched.
- **Control Flow**:
    - The function uses [`make_matcher`](#Servermake_matcher) to create a matcher object based on the provided `pattern`.
    - The matcher and the `handler` are then stored in the `patch_handlers_` vector as a pair.
    - The function returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` object, enabling method chaining for further configuration.
- **Functions called**:
    - [`Server::make_matcher`](#Servermake_matcher)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::Patch<!-- {{#callable:Server::Patch}} -->
Registers a PATCH request handler with a specified URL pattern.
- **Inputs**:
    - `pattern`: A string representing the URL pattern that the handler will respond to.
    - `handler`: A function that takes a `Request`, `Response`, and a `ContentReader`, which processes the request and generates a response.
- **Control Flow**:
    - The function creates a matcher for the provided `pattern` using [`make_matcher`](#Servermake_matcher).
    - It then stores the matcher and the `handler` in the `patch_handlers_for_content_reader_` vector.
    - Finally, it returns a reference to the current `Server` instance to allow for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling method chaining for further configuration.
- **Functions called**:
    - [`Server::make_matcher`](#Servermake_matcher)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::Delete<!-- {{#callable:Server::Delete}} -->
Registers a DELETE request handler for a specified URL pattern.
- **Inputs**:
    - `pattern`: A string representing the URL pattern that the DELETE handler will respond to.
    - `handler`: A function that takes a `Request` and a `Response` object, defining the logic to execute when a DELETE request matches the pattern.
- **Control Flow**:
    - The function creates a matcher for the provided `pattern` using the [`make_matcher`](#Servermake_matcher) method.
    - It then stores the matcher and the `handler` in the `delete_handlers_` vector, which holds all DELETE request handlers.
    - Finally, it returns a reference to the current `Server` instance to allow for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling method chaining for further configuration.
- **Functions called**:
    - [`Server::make_matcher`](#Servermake_matcher)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::Delete<!-- {{#callable:Server::Delete}} -->
Registers a DELETE request handler for a specified URL pattern.
- **Inputs**:
    - `pattern`: A string representing the URL pattern that the DELETE handler will respond to.
    - `handler`: A function that takes a `Request`, `Response`, and a `ContentReader`, which will be invoked when a DELETE request matches the specified pattern.
- **Control Flow**:
    - The function creates a matcher for the provided `pattern` using the [`make_matcher`](#Servermake_matcher) function.
    - It then stores the matcher and the `handler` in the `delete_handlers_for_content_reader_` vector.
    - Finally, it returns a reference to the current `Server` instance to allow for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling method chaining for further configuration.
- **Functions called**:
    - [`Server::make_matcher`](#Servermake_matcher)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::Options<!-- {{#callable:Server::Options}} -->
Registers an `Options` HTTP method handler with a specified pattern.
- **Inputs**:
    - `pattern`: A string representing the URL pattern that the handler will respond to.
    - `handler`: A function that takes a `Request` and a `Response` object, defining the logic to execute when the pattern matches.
- **Control Flow**:
    - The function uses [`make_matcher`](#Servermake_matcher) to create a matcher object based on the provided `pattern`.
    - The matcher and the `handler` are then stored in the `options_handlers_` vector.
    - Finally, the function returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling further configuration or method chaining.
- **Functions called**:
    - [`Server::make_matcher`](#Servermake_matcher)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_base\_dir<!-- {{#callable:Server::set_base_dir}} -->
Sets the base directory for a specified mount point in the `Server`.
- **Inputs**:
    - `dir`: A `std::string` representing the directory path to be set as the base directory.
    - `mount_point`: A `std::string` representing the mount point to which the base directory will be associated.
- **Control Flow**:
    - Calls the [`set_mount_point`](#Serverset_mount_point) method with the provided `mount_point` and `dir` arguments.
    - Returns the result of the [`set_mount_point`](#Serverset_mount_point) method, which indicates success or failure.
- **Output**: Returns a `bool` indicating whether the base directory was successfully set for the specified mount point.
- **Functions called**:
    - [`Server::set_mount_point`](#Serverset_mount_point)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_mount\_point<!-- {{#callable:Server::set_mount_point}} -->
Sets a mount point for a specified directory in the server.
- **Inputs**:
    - `mount_point`: A string representing the mount point to be set.
    - `dir`: A string representing the directory to be associated with the mount point.
    - `headers`: An object of type Headers containing any additional headers to be associated with the mount point.
- **Control Flow**:
    - Creates a `FileStat` object to check if the provided directory exists and is a directory.
    - If the directory is valid, it checks if the mount point is not empty and starts with a '/' character.
    - If both conditions are satisfied, it adds the mount point, directory, and headers to the `base_dirs_` vector.
    - Returns true if the mount point was successfully set; otherwise, returns false.
- **Output**: Returns a boolean indicating the success or failure of setting the mount point.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::remove\_mount\_point<!-- {{#callable:Server::remove_mount_point}} -->
Removes a specified mount point from the server's base directories.
- **Inputs**:
    - `mount_point`: A constant reference to a string representing the mount point to be removed.
- **Control Flow**:
    - Iterates through the `base_dirs_` vector to find a matching `mount_point`.
    - If a match is found, the corresponding entry is erased from `base_dirs_`.
    - Returns true if the mount point was successfully removed, otherwise returns false.
- **Output**: Returns a boolean indicating whether the removal of the mount point was successful.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_file\_extension\_and\_mimetype\_mapping<!-- {{#callable:Server::set_file_extension_and_mimetype_mapping}} -->
Sets the mapping between a file extension and its corresponding MIME type.
- **Inputs**:
    - `ext`: A string representing the file extension to be mapped.
    - `mime`: A string representing the MIME type associated with the given file extension.
- **Control Flow**:
    - The function assigns the provided MIME type to the specified file extension in the `file_extension_and_mimetype_map_` member variable.
    - It then returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling further configuration through method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_default\_file\_mimetype<!-- {{#callable:Server::set_default_file_mimetype}} -->
Sets the default MIME type for files served by the `Server`.
- **Inputs**:
    - `mime`: A `std::string` representing the MIME type to be set as the default for files.
- **Control Flow**:
    - Assigns the provided `mime` string to the member variable `default_file_mimetype_`.
    - Returns a reference to the current `Server` instance to allow for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_file\_request\_handler<!-- {{#callable:Server::set_file_request_handler}} -->
Sets a custom file request handler for the `Server`.
- **Inputs**:
    - `handler`: A `Handler` function that takes a `Request` and a `Response` and processes file requests.
- **Control Flow**:
    - The function assigns the provided `handler` to the member variable `file_request_handler_`.
    - It uses `std::move` to efficiently transfer ownership of the `handler` to avoid unnecessary copies.
    - Finally, it returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_error\_handler\_core<!-- {{#callable:Server::set_error_handler_core}} -->
Sets the error handler for the `Server` instance.
- **Inputs**:
    - `handler`: A callable that takes a `Request` and a `Response` and returns a `HandlerResponse`, which defines how to handle errors.
    - `std::true_type`: A type trait used to indicate that the provided handler is convertible to `HandlerWithResponse`.
- **Control Flow**:
    - The function assigns the provided `handler` to the member variable `error_handler_` after moving it.
    - It returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling fluent interface style for further configuration.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_error\_handler\_core<!-- {{#callable:Server::set_error_handler_core}} -->
Sets a custom error handler for the `Server` instance.
- **Inputs**:
    - `handler`: A callable that takes a `Request` and a `Response` object, which will be invoked when an error occurs.
    - `std::false_type`: A type trait used to indicate that the provided handler is not convertible to a `HandlerWithResponse`.
- **Control Flow**:
    - The function assigns a lambda function to the `error_handler_` member variable of the `Server` class.
    - This lambda captures the provided `handler` and defines its behavior when invoked with a `Request` and `Response`.
    - The lambda calls the captured `handler` and returns `HandlerResponse::Handled` to indicate that the error has been processed.
- **Output**: Returns a reference to the current `Server` instance, allowing for method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_exception\_handler<!-- {{#callable:Server::set_exception_handler}} -->
Sets a custom exception handler for the `Server`.
- **Inputs**:
    - `handler`: A callable object that takes a `Request`, a `Response`, and an `std::exception_ptr`, which will be invoked when an exception occurs during request processing.
- **Control Flow**:
    - The function assigns the provided `handler` to the member variable `exception_handler_` using `std::move` to efficiently transfer ownership.
    - It then returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling fluent interface style method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_pre\_routing\_handler<!-- {{#callable:Server::set_pre_routing_handler}} -->
Sets a pre-routing handler for the `Server`.
- **Inputs**:
    - `handler`: A `HandlerWithResponse` function that processes requests before routing.
- **Control Flow**:
    - The function takes a `HandlerWithResponse` as an argument.
    - It moves the provided handler into the `pre_routing_handler_` member variable.
    - The function returns a reference to the current `Server` instance.
- **Output**: Returns a reference to the `Server` instance, allowing for method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_post\_routing\_handler<!-- {{#callable:Server::set_post_routing_handler}} -->
Sets a post-routing handler for the `Server`.
- **Inputs**:
    - `handler`: A `Handler` function that takes a `Request` and a `Response` and is invoked after routing.
- **Control Flow**:
    - The function assigns the provided `handler` to the member variable `post_routing_handler_`.
    - It uses `std::move` to efficiently transfer ownership of the `handler` to avoid unnecessary copies.
    - Finally, it returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_logger<!-- {{#callable:Server::set_logger}} -->
Sets the `logger_` member of the `Server` class to the provided `Logger` instance.
- **Inputs**:
    - `logger`: An instance of `Logger` that will be assigned to the `logger_` member variable of the `Server` class.
- **Control Flow**:
    - The function uses `std::move` to efficiently transfer ownership of the `logger` argument to the `logger_` member variable.
    - It returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the current `Server` instance, enabling fluent interface style method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_expect\_100\_continue\_handler<!-- {{#callable:Server::set_expect_100_continue_handler}} -->
Sets a custom handler for processing HTTP 100-Continue expectations.
- **Inputs**:
    - `handler`: A callable object that takes a `Request` and a `Response` and returns an integer, used to handle the 100-Continue HTTP status.
- **Control Flow**:
    - The function takes the provided `handler` and moves it into the member variable `expect_100_continue_handler_`.
    - It then returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance to facilitate method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_address\_family<!-- {{#callable:Server::set_address_family}} -->
Sets the address family for the `Server` instance.
- **Inputs**:
    - `family`: An integer representing the address family, typically defined by constants such as AF_INET for IPv4 or AF_INET6 for IPv6.
- **Control Flow**:
    - The function assigns the provided `family` value to the member variable `address_family_`.
    - It then returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the current `Server` instance, enabling fluent interface style method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_tcp\_nodelay<!-- {{#callable:Server::set_tcp_nodelay}} -->
Sets the TCP_NODELAY option for the server socket.
- **Inputs**:
    - `on`: A boolean value indicating whether to enable (true) or disable (false) the TCP_NODELAY option.
- **Control Flow**:
    - The function assigns the value of the input parameter 'on' to the member variable 'tcp_nodelay_'.
    - It then returns a reference to the current instance of the `Server` class, allowing for method chaining.
- **Output**: Returns a reference to the current `Server` instance.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_ipv6\_v6only<!-- {{#callable:Server::set_ipv6_v6only}} -->
Sets the IPv6 socket option to either allow or disallow dual-stack mode.
- **Inputs**:
    - `on`: A boolean value indicating whether to enable (true) or disable (false) the IPv6 `V6ONLY` option.
- **Control Flow**:
    - The function assigns the value of the input parameter `on` to the member variable `ipv6_v6only_`.
    - It then returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling method chaining for further configuration.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_socket\_options<!-- {{#callable:Server::set_socket_options}} -->
Sets the socket options for the `Server` instance.
- **Inputs**:
    - `socket_options`: An instance of `SocketOptions` that contains the configuration settings for the socket.
- **Control Flow**:
    - The function takes the `socket_options` parameter and moves it into the member variable `socket_options_`.
    - It then returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling fluent interface style method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_default\_headers<!-- {{#callable:Server::set_default_headers}} -->
Sets the default headers for the `Server` instance.
- **Inputs**:
    - `headers`: An instance of `Headers` containing the default headers to be set for the server.
- **Control Flow**:
    - The function uses `std::move` to transfer ownership of the `headers` argument to the `default_headers_` member variable.
    - It then returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling method chaining for further configuration.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_header\_writer<!-- {{#callable:Server::set_header_writer}} -->
Sets a custom header writer function for the `Server`.
- **Inputs**:
    - `writer`: A `std::function` that takes a `Stream` and `Headers` as parameters and returns a `ssize_t`, representing the custom logic for writing headers.
- **Control Flow**:
    - The function assigns the provided `writer` function to the member variable `header_writer_`.
    - It then returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling fluent interface style method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_keep\_alive\_max\_count<!-- {{#callable:Server::set_keep_alive_max_count}} -->
Sets the maximum number of keep-alive connections for the `Server`.
- **Inputs**:
    - `count`: A `size_t` value representing the maximum number of keep-alive connections allowed.
- **Control Flow**:
    - The function assigns the provided `count` value to the private member variable `keep_alive_max_count_`.
    - It then returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling method chaining for further configuration.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_keep\_alive\_timeout<!-- {{#callable:Server::set_keep_alive_timeout}} -->
Sets the keep-alive timeout duration for the server.
- **Inputs**:
    - `sec`: A `time_t` value representing the number of seconds to set as the keep-alive timeout.
- **Control Flow**:
    - The function assigns the input value `sec` to the member variable `keep_alive_timeout_sec_`.
    - It then returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the `Server` instance, enabling fluent interface style method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_read\_timeout<!-- {{#callable:Server::set_read_timeout}} -->
Sets the read timeout for the `Server` instance.
- **Inputs**:
    - `sec`: The number of seconds for the read timeout.
    - `usec`: The number of microseconds for the read timeout.
- **Control Flow**:
    - Assigns the value of `sec` to the member variable `read_timeout_sec_`.
    - Assigns the value of `usec` to the member variable `read_timeout_usec_`.
    - Returns a reference to the current `Server` instance.
- **Output**: Returns a reference to the `Server` instance, allowing for method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_write\_timeout<!-- {{#callable:Server::set_write_timeout}} -->
Sets the write timeout for the `Server` instance.
- **Inputs**:
    - `sec`: The number of seconds for the write timeout.
    - `usec`: The number of microseconds for the write timeout.
- **Control Flow**:
    - Assigns the value of `sec` to the member variable `write_timeout_sec_`.
    - Assigns the value of `usec` to the member variable `write_timeout_usec_`.
    - Returns a reference to the current `Server` instance.
- **Output**: Returns a reference to the `Server` instance, allowing for method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_idle\_interval<!-- {{#callable:Server::set_idle_interval}} -->
Sets the idle interval for the `Server` instance.
- **Inputs**:
    - `sec`: The number of seconds for the idle interval.
    - `usec`: The number of microseconds for the idle interval.
- **Control Flow**:
    - The function assigns the value of `sec` to the member variable `idle_interval_sec_`.
    - It assigns the value of `usec` to the member variable `idle_interval_usec_`.
    - Finally, it returns a reference to the current `Server` instance.
- **Output**: Returns a reference to the `Server` instance, allowing for method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::set\_payload\_max\_length<!-- {{#callable:Server::set_payload_max_length}} -->
Sets the maximum allowable length for the payload in the `Server` class.
- **Inputs**:
    - `length`: A `size_t` value representing the maximum length of the payload that the server will accept.
- **Control Flow**:
    - The function assigns the provided `length` value to the member variable `payload_max_length_`.
    - It then returns a reference to the current `Server` instance, allowing for method chaining.
- **Output**: Returns a reference to the current `Server` instance, enabling fluent interface style method chaining.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::bind\_to\_port<!-- {{#callable:Server::bind_to_port}} -->
Binds the server to a specified host and port, updating the server's state if the binding fails.
- **Inputs**:
    - `host`: A string representing the hostname or IP address to bind the server.
    - `port`: An integer representing the port number to bind the server.
    - `socket_flags`: An integer representing optional socket flags for binding.
- **Control Flow**:
    - Calls the [`bind_internal`](#Serverbind_internal) method with the provided `host`, `port`, and `socket_flags` to attempt binding.
    - Checks the return value of [`bind_internal`](#Serverbind_internal); if it is -1, sets the `is_decommissioned` flag to true.
    - Returns true if the binding was successful (return value >= 0), otherwise returns false.
- **Output**: Returns a boolean indicating whether the binding was successful.
- **Functions called**:
    - [`Server::bind_internal`](#Serverbind_internal)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::bind\_to\_any\_port<!-- {{#callable:Server::bind_to_any_port}} -->
Binds a server socket to any available port on the specified host.
- **Inputs**:
    - `host`: A string representing the hostname or IP address to which the server socket will be bound.
    - `socket_flags`: An integer representing various socket options that can be applied during the binding process.
- **Control Flow**:
    - Calls the [`bind_internal`](#Serverbind_internal) method with the specified `host`, a port value of 0 (indicating any available port), and the provided `socket_flags`.
    - Checks the return value of [`bind_internal`](#Serverbind_internal); if it equals -1, it sets the `is_decommissioned` flag to true, indicating that the server is no longer operational.
- **Output**: Returns the result of the [`bind_internal`](#Serverbind_internal) call, which is an integer indicating the success or failure of the binding operation.
- **Functions called**:
    - [`Server::bind_internal`](#Serverbind_internal)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::listen\_after\_bind<!-- {{#callable:Server::listen_after_bind}} -->
This function invokes the `listen_internal` method to initiate listening for incoming connections after the server has been bound to a port.
- **Inputs**: None
- **Control Flow**:
    - The function directly calls the `listen_internal` method.
    - It returns the boolean result of the `listen_internal` method, indicating success or failure.
- **Output**: The function returns a boolean value that indicates whether the server is successfully listening for incoming connections.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::listen<!-- {{#callable:Server::listen}} -->
The `listen` function binds the server to a specified host and port, and prepares it to accept incoming connections.
- **Inputs**:
    - `host`: A string representing the hostname or IP address to which the server will bind.
    - `port`: An integer representing the port number on which the server will listen for incoming connections.
    - `socket_flags`: An integer representing optional socket flags that can modify the behavior of the socket.
- **Control Flow**:
    - The function first calls [`bind_to_port`](#Serverbind_to_port) with the provided `host`, `port`, and `socket_flags` to bind the server socket to the specified address.
    - If the binding is successful (i.e., [`bind_to_port`](#Serverbind_to_port) returns true), it then calls `listen_internal` to start listening for incoming connections.
    - The function returns true only if both [`bind_to_port`](#Serverbind_to_port) and `listen_internal` succeed.
- **Output**: The function returns a boolean value indicating whether the server was successfully bound to the specified host and port and is ready to accept connections.
- **Functions called**:
    - [`Server::bind_to_port`](#Serverbind_to_port)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::is\_running<!-- {{#callable:Server::is_running}} -->
Checks if the `Server` instance is currently running.
- **Inputs**:
    - `this`: A constant reference to the `Server` instance on which the method is called.
- **Control Flow**:
    - The method directly returns the value of the private member variable `is_running_`.
- **Output**: Returns a boolean value indicating whether the server is running (`true`) or not (`false`).
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::wait\_until\_ready<!-- {{#callable:Server::wait_until_ready}} -->
The `wait_until_ready` function blocks the current thread until the server is either running or has been decommissioned.
- **Inputs**: None
- **Control Flow**:
    - The function enters a while loop that continues as long as the server is not running and has not been decommissioned.
    - Within the loop, the thread sleeps for 1 millisecond to avoid busy-waiting.
- **Output**: The function does not return any value; it simply ensures that the server is ready to accept requests by blocking until the server state changes.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::stop<!-- {{#callable:Server::stop}} -->
Stops the server by shutting down and closing the server socket if it is currently running.
- **Inputs**: None
- **Control Flow**:
    - Checks if the server is currently running using the `is_running_` flag.
    - If the server is running, it asserts that the server socket is valid (not `INVALID_SOCKET`).
    - Exchanges the current server socket with `INVALID_SOCKET` atomically, effectively marking it as closed.
    - Calls `detail::shutdown_socket(sock)` to gracefully shut down the socket.
    - Calls `detail::close_socket(sock)` to release the resources associated with the socket.
    - Sets the `is_decommissioned` flag to false, indicating that the server is not decommissioned.
- **Output**: The function does not return a value; it modifies the state of the server by stopping it and closing the socket.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::decommission<!-- {{#callable:Server::decommission}} -->
Marks the `Server` instance as decommissioned.
- **Inputs**: None
- **Control Flow**:
    - The function sets the `is_decommissioned` member variable of the `Server` class to `true`.
- **Output**: The function does not return any value; it modifies the internal state of the `Server` instance.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::parse\_request\_line<!-- {{#callable:Server::parse_request_line}} -->
Parses an HTTP request line from a string and populates a `Request` object.
- **Inputs**:
    - `s`: A pointer to a C-style string representing the raw HTTP request line.
    - `req`: A reference to a `Request` object that will be populated with the parsed method, target, and version.
- **Control Flow**:
    - Checks if the input string `s` has a valid length and ends with CRLF (\r\n); if not, returns false.
    - Uses a lambda function to split the string into components (method, target, version) and populates the `req` object.
    - Validates that exactly three components were parsed; if not, returns false.
    - Checks if the parsed method is one of the allowed HTTP methods; if not, returns false.
    - Validates that the HTTP version is either 'HTTP/1.1' or 'HTTP/1.0'; if not, returns false.
    - Removes any URL fragment from the target by erasing the part after a '#' character.
    - Parses the query string from the target and populates the `params` field of the `req` object.
- **Output**: Returns true if the request line was successfully parsed and validated; otherwise, returns false.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::write\_response<!-- {{#callable:Server::write_response}} -->
The `write_response` function prepares and sends an HTTP response to a client stream.
- **Inputs**:
    - `strm`: A reference to a `Stream` object representing the output stream to which the response will be written.
    - `close_connection`: A boolean indicating whether the connection should be closed after the response is sent.
    - `req`: A reference to a `Request` object containing the details of the incoming HTTP request.
    - `res`: A reference to a `Response` object that will be populated with the response data to be sent.
- **Control Flow**:
    - The function first clears any existing range specifications in the `req` object to prevent incorrect application to the response content.
    - It then calls the [`write_response_core`](#Serverwrite_response_core) function, passing along the stream, connection status, request, response, and a flag indicating that range application is not needed.
- **Output**: The function returns a boolean indicating the success or failure of the response writing process.
- **Functions called**:
    - [`Server::write_response_core`](#Serverwrite_response_core)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::write\_response\_with\_content<!-- {{#callable:Server::write_response_with_content}} -->
Writes a response to a stream with content based on the provided request and response objects.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where the response will be written.
    - `close_connection`: A boolean indicating whether to close the connection after the response is sent.
    - `req`: A constant reference to a `Request` object containing the details of the incoming request.
    - `res`: A reference to a `Response` object that will be populated with the response data.
- **Control Flow**:
    - The function calls [`write_response_core`](#Serverwrite_response_core) with the provided parameters and an additional boolean argument set to true.
    - The [`write_response_core`](#Serverwrite_response_core) function handles the actual writing of the response, including any necessary processing based on the request and response objects.
- **Output**: Returns a boolean indicating the success or failure of the response writing operation.
- **Functions called**:
    - [`Server::write_response_core`](#Serverwrite_response_core)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::write\_response\_core<!-- {{#callable:Server::write_response_core}} -->
The `write_response_core` function constructs and sends an HTTP response based on the provided request and response objects.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where the response will be written.
    - `close_connection`: A boolean indicating whether the connection should be closed after the response.
    - `req`: A constant reference to a `Request` object representing the incoming HTTP request.
    - `res`: A reference to a `Response` object that will be populated with the response data.
    - `need_apply_ranges`: A boolean indicating whether range headers need to be applied to the response.
- **Control Flow**:
    - The function asserts that the response status is valid (not -1).
    - If the response status indicates an error (400 or higher) and an error handler is set, it invokes the error handler.
    - If `need_apply_ranges` is true, it applies range headers to the response.
    - It prepares additional headers based on the connection status and request headers.
    - It sets the 'Content-Type' header if the response body is not empty and no 'Content-Type' is already set.
    - If the response body is empty and no content length is specified, it sets 'Content-Length' to 0.
    - If the request method is 'HEAD', it sets the 'Accept-Ranges' header if not already present.
    - If a post-routing handler is set, it invokes it with the request and response.
    - It writes the response line and headers to a buffer stream.
    - It writes the response body or content using a content provider if the request method is not 'HEAD'.
    - Finally, it logs the request and response if a logger is set.
- **Output**: Returns a boolean indicating the success of the response writing process.
- **Functions called**:
    - [`Server::apply_ranges`](#Serverapply_ranges)
    - [`Server::write_content_with_provider`](#Serverwrite_content_with_provider)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::write\_content\_with\_provider<!-- {{#callable:Server::write_content_with_provider}} -->
Writes content to a stream based on the request and response parameters.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where the content will be written.
    - `req`: A reference to a `Request` object containing the client's request data.
    - `res`: A reference to a `Response` object that holds the server's response data.
    - `boundary`: A string representing the boundary used for multipart responses.
    - `content_type`: A string indicating the MIME type of the content being sent.
- **Control Flow**:
    - Checks if the response content length is greater than zero.
    - If there are no ranges specified in the request, it writes the entire content using `detail::write_content`.
    - If a single range is specified, it calculates the offset and length and writes that range.
    - If multiple ranges are specified, it calls `detail::write_multipart_ranges_data` to handle the multipart response.
    - If the content length is zero, it checks if the content provider is chunked.
    - If chunked, it determines the encoding type and creates the appropriate compressor before writing the chunked content.
    - If not chunked, it writes the content without length using `detail::write_content_without_length`.
- **Output**: Returns a boolean indicating the success or failure of the content writing operation.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::read\_content<!-- {{#callable:Server::read_content}} -->
Reads content from a stream and populates a `Request` and `Response` object.
- **Inputs**:
    - `strm`: A reference to a `Stream` object from which the content is read.
    - `req`: A reference to a `Request` object that will be populated with the read content.
    - `res`: A reference to a `Response` object that will be modified based on the read operation.
- **Control Flow**:
    - Calls [`read_content_core`](#Serverread_content_core) with the provided stream, request, and response, along with three lambda functions for handling regular data, multipart file data, and multipart content.
    - The first lambda appends regular data to `req.body` if it does not exceed the maximum size.
    - The second lambda handles multipart file uploads, ensuring the number of files does not exceed a predefined limit and storing them in `req.files`.
    - The third lambda appends data to the content of the current file being processed.
    - After reading, it checks the `Content-Type` header of the request to determine if it is URL-encoded and parses it if necessary.
    - If any of the conditions fail during reading or parsing, the function returns false.
- **Output**: Returns a boolean indicating the success or failure of the content reading and processing operation.
- **Functions called**:
    - [`Server::read_content_core`](#Serverread_content_core)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::read\_content\_with\_content\_receiver<!-- {{#callable:Server::read_content_with_content_receiver}} -->
This function reads content from a stream and processes it using specified content receivers.
- **Inputs**:
    - `strm`: A reference to a `Stream` object from which content is read.
    - `req`: A reference to a `Request` object that contains the details of the incoming request.
    - `res`: A reference to a `Response` object that will be populated with the response data.
    - `receiver`: A `ContentReceiver` function that processes the received content.
    - `multipart_header`: A `MultipartContentHeader` object that contains metadata for multipart content.
    - `multipart_receiver`: A `ContentReceiver` function for handling multipart content.
- **Control Flow**:
    - The function calls [`read_content_core`](#Serverread_content_core), passing all its parameters after moving the `receiver`, `multipart_header`, and `multipart_receiver` to optimize performance.
    - The [`read_content_core`](#Serverread_content_core) function is responsible for the actual reading and processing of the content from the stream.
- **Output**: The function returns a boolean indicating the success or failure of the content reading and processing operation.
- **Functions called**:
    - [`Server::read_content_core`](#Serverread_content_core)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::read\_content\_core<!-- {{#callable:Server::read_content_core}} -->
Processes the content of an incoming HTTP request, handling both multipart and regular data.
- **Inputs**:
    - `strm`: A reference to a `Stream` object representing the input stream from which the content is read.
    - `req`: A reference to a `Request` object containing the details of the incoming HTTP request.
    - `res`: A reference to a `Response` object used to set the response status and headers.
    - `receiver`: A `ContentReceiver` function that processes the received content.
    - `multipart_header`: A `MultipartContentHeader` object that contains metadata for multipart content.
    - `multipart_receiver`: A `ContentReceiver` function specifically for handling multipart content.
- **Control Flow**:
    - Checks if the request is of type multipart form data.
    - If multipart, it retrieves the content type and parses the boundary; if parsing fails, it sets the response status to Bad Request and returns false.
    - Sets up a lambda function to handle the parsing of multipart data using `multipart_form_data_parser`.
    - If the request method is DELETE and lacks a Content-Length header, it returns true immediately.
    - Calls `detail::read_content` to read the content from the stream, using the appropriate receiver function.
    - If the request is multipart, it validates the multipart data after reading; if invalid, it sets the response status to Bad Request and returns false.
    - Returns true if all operations are successful.
- **Output**: Returns a boolean indicating the success or failure of the content reading process.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::handle\_file\_request<!-- {{#callable:Server::handle_file_request}} -->
Handles file requests by checking the request path against mounted directories and serving the appropriate file or redirecting.
- **Inputs**:
    - `req`: A constant reference to a `Request` object that contains the details of the incoming request, including the requested path.
    - `res`: A reference to a `Response` object that will be populated with the response data to be sent back to the client.
    - `head`: A boolean flag indicating whether the request is an HTTP HEAD request, which should not return a message body.
- **Control Flow**:
    - Iterates over each entry in the `base_dirs_` vector to find a matching mount point for the request path.
    - If a match is found, it constructs a sub-path and checks if it is valid using `detail::is_valid_path`.
    - If valid, it constructs the full file path and checks if it is a directory or a file using `detail::FileStat`.
    - If it is a directory, it sets a redirect response to the sub-path with a 301 status code.
    - If it is a file, it sets the appropriate headers, prepares the content provider using memory mapping, and optionally calls a file request handler if not a HEAD request.
    - Returns true if a file or directory was successfully handled, otherwise continues checking other entries.
- **Output**: Returns a boolean indicating whether the file request was successfully handled (true) or not (false).
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::create\_server\_socket<!-- {{#callable:Server::create_server_socket}} -->
Creates a server socket and binds it to the specified host and port.
- **Inputs**:
    - `host`: A string representing the hostname or IP address to bind the server socket.
    - `port`: An integer specifying the port number on which the server will listen.
    - `socket_flags`: An integer representing various socket options (e.g., non-blocking mode).
    - `socket_options`: An instance of `SocketOptions` containing additional socket configuration.
- **Control Flow**:
    - Calls `detail::create_socket` to create a socket with the specified parameters.
    - Within `create_socket`, a lambda function is defined to handle the binding and listening of the socket.
    - The lambda attempts to bind the socket to the address specified by `ai` and checks for errors.
    - If binding is successful, it then calls `listen` on the socket to start listening for incoming connections.
    - Returns true if both binding and listening are successful, otherwise returns false.
- **Output**: Returns a `socket_t` representing the created server socket.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::bind\_internal<!-- {{#callable:Server::bind_internal}} -->
Binds a server to a specified host and port, returning the port number if successful.
- **Inputs**:
    - `host`: A string representing the hostname or IP address to bind the server.
    - `port`: An integer representing the port number to bind the server to.
    - `socket_flags`: An integer representing flags for socket options.
- **Control Flow**:
    - Checks if the server is decommissioned; if so, returns -1.
    - Validates the server state; if invalid, returns -1.
    - Attempts to create a server socket using [`create_server_socket`](#Servercreate_server_socket) with the provided host, port, and socket flags.
    - If the socket creation fails (returns `INVALID_SOCKET`), returns -1.
    - If the port is 0, retrieves the actual port number assigned to the socket using `getsockname`.
    - Checks the address family of the socket and returns the appropriate port number for IPv4 or IPv6.
    - If the address family is neither, returns -1.
    - If the port is not 0, simply returns the provided port number.
- **Output**: Returns the port number that the server is bound to, or -1 if an error occurs during the binding process.
- **Functions called**:
    - [`gzip_decompressor::is_valid`](#gzip_decompressoris_valid)
    - [`Server::create_server_socket`](#Servercreate_server_socket)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::routing<!-- {{#callable:Server::routing}} -->
The `routing` function processes incoming HTTP requests and dispatches them to the appropriate handler based on the request method.
- **Inputs**:
    - `req`: A reference to a `Request` object representing the incoming HTTP request.
    - `res`: A reference to a `Response` object used to construct the HTTP response.
    - `strm`: A reference to a `Stream` object used for reading the request content.
- **Control Flow**:
    - The function first checks if a pre-routing handler is set and if it handles the request, returning true if so.
    - It then checks if the request method is either 'GET' or 'HEAD' and attempts to handle file requests.
    - If the request expects content, it initializes a `ContentReader` and dispatches the request based on the HTTP method (POST, PUT, PATCH, DELETE).
    - If the request method does not match any specific handlers, it defaults to a regular handler based on the method.
    - If no handlers are found for the request method, it sets the response status to 400 Bad Request and returns false.
- **Output**: The function returns a boolean indicating whether the request was successfully handled.
- **Functions called**:
    - [`Server::handle_file_request`](#Serverhandle_file_request)
    - [`Server::read_content_with_content_receiver`](#Serverread_content_with_content_receiver)
    - [`Server::dispatch_request_for_content_reader`](#Serverdispatch_request_for_content_reader)
    - [`read_content`](#read_content)
    - [`Server::dispatch_request`](#Serverdispatch_request)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::dispatch\_request<!-- {{#callable:Server::dispatch_request}} -->
Dispatches a `Request` to the appropriate handler based on matching criteria.
- **Inputs**:
    - `req`: A reference to a `Request` object that contains the details of the incoming request.
    - `res`: A reference to a `Response` object that will be populated with the response data.
    - `handlers`: A collection of handler functions paired with matchers that determine how to process the request.
- **Control Flow**:
    - Iterates over each handler in the `handlers` collection.
    - For each handler, retrieves the `matcher` and `handler` function.
    - Checks if the `matcher` successfully matches the incoming `req`.
    - If a match is found, invokes the corresponding `handler` with `req` and `res`, and returns true.
    - If no matches are found after iterating through all handlers, returns false.
- **Output**: Returns a boolean indicating whether a matching handler was found and executed.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::apply\_ranges<!-- {{#callable:Server::apply_ranges}} -->
The `apply_ranges` function processes HTTP range requests and modifies the response accordingly.
- **Inputs**:
    - `req`: A constant reference to a `Request` object containing the client's request data, including any specified ranges.
    - `res`: A reference to a `Response` object that will be modified to include the appropriate headers and body based on the request.
    - `content_type`: A reference to a string that will be set to the content type of the response.
    - `boundary`: A reference to a string that will be set to the boundary used for multipart responses.
- **Control Flow**:
    - Checks if the request contains multiple ranges and if the response status is 206 Partial Content.
    - If so, retrieves and erases the existing Content-Type header, generates a new boundary, and sets the Content-Type to multipart/byteranges.
    - Determines the encoding type based on the request and response.
    - If the response body is empty, calculates the content length based on the request ranges and sets the appropriate headers.
    - If the response body is not empty, processes the ranges to either return a single range or multiple ranges, updating the response body accordingly.
    - If an encoding type is determined, applies the corresponding compression and updates the response headers.
    - Finally, sets the Content-Length header based on the size of the response body.
- **Output**: The function modifies the `Response` object to include the correct headers and body based on the range requests, and sets the content type and length appropriately.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::dispatch\_request\_for\_content\_reader<!-- {{#callable:Server::dispatch_request_for_content_reader}} -->
Dispatches a request to the appropriate content reader handler based on the request matcher.
- **Inputs**:
    - `req`: A reference to a `Request` object representing the incoming request.
    - `res`: A reference to a `Response` object where the response will be written.
    - `content_reader`: A `ContentReader` object used to read the content associated with the request.
    - `handlers`: A collection of pairs containing matchers and their corresponding handlers for content reading.
- **Control Flow**:
    - Iterates over each pair in the `handlers` collection.
    - For each pair, retrieves the `matcher` and `handler`.
    - Checks if the `matcher` matches the incoming `req`.
    - If a match is found, invokes the `handler` with `req`, `res`, and `content_reader`, then returns true.
    - If no match is found after iterating through all handlers, returns false.
- **Output**: Returns a boolean indicating whether a matching handler was found and executed.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::process\_request<!-- {{#callable:Server::process_request}} -->
Processes an incoming HTTP request, handling parsing, routing, and response generation.
- **Inputs**:
    - `strm`: A reference to a `Stream` object representing the connection stream for the request.
    - `remote_addr`: A string representing the remote address of the client making the request.
    - `remote_port`: An integer representing the remote port of the client.
    - `local_addr`: A string representing the local address of the server handling the request.
    - `local_port`: An integer representing the local port of the server.
    - `close_connection`: A boolean indicating whether the connection should be closed after the response.
    - `connection_closed`: A reference to a boolean that will be set to true if the connection is closed.
    - `setup_request`: A function that takes a `Request` reference for additional setup before processing.
- **Control Flow**:
    - Initializes a buffer and a line reader for the incoming stream.
    - Checks if the connection has been closed by the client; if so, returns false.
    - Parses the request line and headers; if parsing fails, sends a 400 Bad Request response.
    - Checks if the request URI exceeds the maximum length; if so, sends a 414 URI Too Long response.
    - Determines if the connection should be closed based on the 'Connection' header.
    - Sets remote and local address/port information in the request object.
    - Handles 'Range' headers and validates them; if invalid, sends a 416 Range Not Satisfiable response.
    - Calls the `setup_request` function if provided to allow for additional request setup.
    - Handles 'Expect: 100-continue' header and responds accordingly.
    - Sets up a closure to check if the connection is closed based on the socket state.
    - Routes the request to the appropriate handler; catches exceptions and handles them if they occur.
    - Generates a response based on the routing result, serving file content if applicable, or sending a 404 Not Found response.
- **Output**: Returns a boolean indicating the success of the request processing, with the response written to the stream.
- **Functions called**:
    - [`Server::parse_request_line`](#Serverparse_request_line)
    - [`Server::write_response`](#Serverwrite_response)
    - [`Server::routing`](#Serverrouting)
    - [`Server::write_response_with_content`](#Serverwrite_response_with_content)
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::is\_valid<!-- {{#callable:Server::is_valid}} -->
Checks if the `Server` instance is valid.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `Server` class.
- **Control Flow**:
    - The function directly returns a boolean value without any conditional checks or loops.
    - It always returns `true`, indicating that the server is considered valid.
- **Output**: Returns a boolean value, which is always `true` in this implementation.
- **See also**: [`Server`](#Server)  (Data Structure)


---
#### Server::process\_and\_close\_socket<!-- {{#callable:Server::process_and_close_socket}} -->
Processes a server socket by handling requests and then closes the socket.
- **Inputs**:
    - `sock`: A socket identifier of type `socket_t` representing the server socket to be processed.
- **Control Flow**:
    - Retrieves the remote IP address and port associated with the socket using `detail::get_remote_ip_and_port`.
    - Retrieves the local IP address and port associated with the socket using `detail::get_local_ip_and_port`.
    - Processes the server socket by calling `detail::process_server_socket`, passing in various parameters including a lambda function that handles the request processing.
    - Shuts down the socket using `detail::shutdown_socket` to ensure no further communication can occur.
    - Closes the socket using `detail::close_socket` to release the resources associated with it.
    - Returns the result of the socket processing operation.
- **Output**: Returns a boolean indicating the success or failure of the socket processing operation.
- **Functions called**:
    - [`Server::process_request`](#Serverprocess_request)
- **See also**: [`Server`](#Server)  (Data Structure)



---
### HandlerResponse<!-- {{#data_structure:Server::HandlerResponse}} -->
- **Type**: `enum`
- **Members**:
    - `Handled`: Represents a state where the handler has successfully processed the request.
    - `Unhandled`: Represents a state where the handler did not process the request.
- **Description**: The `HandlerResponse` enum class defines two possible states for a handler's response: `Handled` and `Unhandled`. This enum is used to indicate whether a particular request or event has been processed by a handler or not, providing a clear and concise way to manage control flow based on the handler's outcome.


---
### MountPointEntry<!-- {{#data_structure:Server::MountPointEntry}} -->
- **Type**: `struct`
- **Members**:
    - `mount_point`: A string representing the mount point path.
    - `base_dir`: A string representing the base directory path.
    - `headers`: An instance of the Headers type, presumably containing HTTP headers or similar metadata.
- **Description**: The `MountPointEntry` struct is designed to represent a mapping between a mount point and its corresponding base directory, along with associated headers. It is likely used in a context where directories are mounted or mapped to specific paths, and additional metadata in the form of headers is required for each mapping. This struct is part of a collection, as indicated by the `std::vector<MountPointEntry> base_dirs_;`, suggesting that multiple such mappings can be managed together.


---
### Error<!-- {{#data_structure:Error}} -->
- **Type**: `enum`
- **Members**:
    - `Success`: Indicates a successful operation.
    - `Unknown`: Represents an unknown error.
    - `Connection`: Indicates a connection error.
    - `BindIPAddress`: Represents an error binding to an IP address.
    - `Read`: Indicates an error during a read operation.
    - `Write`: Indicates an error during a write operation.
    - `ExceedRedirectCount`: Represents an error when the redirect count is exceeded.
    - `Canceled`: Indicates an operation was canceled.
    - `SSLConnection`: Represents an error in SSL connection.
    - `SSLLoadingCerts`: Indicates an error loading SSL certificates.
    - `SSLServerVerification`: Represents an error in SSL server verification.
    - `SSLServerHostnameVerification`: Indicates an error in SSL server hostname verification.
    - `UnsupportedMultipartBoundaryChars`: Represents an error with unsupported multipart boundary characters.
    - `Compression`: Indicates an error related to compression.
    - `ConnectionTimeout`: Represents a connection timeout error.
    - `ProxyConnection`: Indicates an error with proxy connection.
    - `SSLPeerCouldBeClosed_`: For internal use only, possibly indicates a closed SSL peer.
- **Description**: The `Error` enum class defines a set of error codes that represent various error conditions that can occur in a network or SSL context. Each enumerator corresponds to a specific type of error, such as connection issues, SSL verification failures, or read/write errors. The enum provides a structured way to handle and identify different error states within the application.


---
### Result<!-- {{#data_structure:Result}} -->
- **Type**: `class`
- **Members**:
    - `res_`: A unique pointer to a Response object, representing the result of an operation.
    - `err_`: An Error object indicating the error status of the operation, defaulting to Error::Unknown.
    - `request_headers_`: A Headers object containing the headers of the request associated with the result.
- **Description**: The `Result` class encapsulates the outcome of an operation, holding a `Response` object if the operation was successful, an `Error` object to indicate any error that occurred, and the request headers associated with the operation. It provides various operators and methods to access the response, check for errors, and retrieve request header information, making it a versatile utility for handling operation results in a structured manner.
- **Member Functions**:
    - [`Result::Result`](#ResultResult)
    - [`Result::Result`](#ResultResult)
    - [`Result::operator==`](#Resultoperator==)
    - [`Result::operator!=`](#Resultoperator!=)
    - [`Result::value`](#Resultvalue)
    - [`Result::value`](#Resultvalue)
    - [`Result::operator*`](#Resultoperator*)
    - [`Result::operator*`](#Resultoperator*)
    - [`Result::operator->`](#Resultoperator->)
    - [`Result::operator->`](#Resultoperator->)
    - [`Result::error`](#Resulterror)
    - [`Result::get_request_header_value_u64`](#Resultget_request_header_value_u64)
    - [`Result::has_request_header`](#Resulthas_request_header)
    - [`Result::get_request_header_value`](#Resultget_request_header_value)
    - [`Result::get_request_header_value_count`](#Resultget_request_header_value_count)

**Methods**

---
#### Result::Result<!-- {{#callable:Result::Result}} -->
The `Result` constructor initializes a `Result` object with a response, an error, and optional request headers.
- **Inputs**:
    - `res`: A unique pointer to a `Response` object, which is moved into the `Result` object.
    - `err`: An `Error` object representing the error state of the `Result`.
    - `request_headers`: An optional `Headers` object representing the request headers, which is moved into the `Result` object; defaults to an empty `Headers` object if not provided.
- **Control Flow**:
    - The constructor initializes the `res_` member by moving the `res` argument into it.
    - The `err_` member is initialized with the `err` argument.
    - The `request_headers_` member is initialized by moving the `request_headers` argument into it, defaulting to an empty `Headers` object if not provided.
- **Output**: A `Result` object is constructed with the provided response, error, and request headers.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::Result<!-- {{#callable:Result::Result}} -->
The `Result` constructor initializes a `Result` object with a response, an error, and optional request headers.
- **Inputs**:
    - `res`: A unique pointer to a `Response` object, representing the response data.
    - `err`: An `Error` object, representing the error state of the result.
    - `request_headers`: An optional `Headers` object, representing the request headers, defaulting to an empty `Headers` object if not provided.
- **Control Flow**:
    - The constructor initializes the `res_` member by moving the `res` argument into it.
    - The `err_` member is initialized with the `err` argument.
    - The `request_headers_` member is initialized by moving the `request_headers` argument into it.
- **Output**: The constructor does not return a value; it initializes the `Result` object.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::operator==<!-- {{#callable:Result::operator==}} -->
The `operator==` function checks if the `res_` member of the `Result` class is `nullptr`, indicating the absence of a valid `Response` object.
- **Inputs**:
    - `std::nullptr_t`: A null pointer constant used to compare against the `res_` member of the `Result` class.
- **Control Flow**:
    - The function compares the `res_` member, which is a `std::unique_ptr<Response>`, to `nullptr`.
    - If `res_` is `nullptr`, the function returns `true`, indicating that the `Result` object does not contain a valid `Response`.
    - If `res_` is not `nullptr`, the function returns `false`, indicating that the `Result` object contains a valid `Response`.
- **Output**: A boolean value indicating whether the `res_` member is `nullptr`.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::operator\!=<!-- {{#callable:Result::operator!=}} -->
The `operator!=` function checks if the `res_` member of the `Result` class is not equal to `nullptr`.
- **Inputs**:
    - `std::nullptr_t`: A null pointer constant used to compare against the `res_` member.
- **Control Flow**:
    - The function compares the `res_` member variable with `nullptr`.
    - It returns `true` if `res_` is not `nullptr`, otherwise it returns `false`.
- **Output**: A boolean value indicating whether the `res_` member is not `nullptr`.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::value<!-- {{#callable:Result::value}} -->
The `value` function returns a reference to the `Response` object stored within the `Result` class.
- **Inputs**: None
- **Control Flow**:
    - The function checks the `res_` member variable, which is a `std::unique_ptr<Response>`, and dereferences it to return a reference to the `Response` object it points to.
    - There are two overloads of the `value` function: one that returns a const reference and one that returns a non-const reference, allowing both read-only and mutable access to the `Response` object.
- **Output**: A reference to the `Response` object stored in the `res_` member variable of the `Result` class.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::value<!-- {{#callable:Result::value}} -->
The `value` function returns a reference to the `Response` object stored in the `Result` class.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private member `res_`, which is a `std::unique_ptr<Response>`.
    - It dereferences `res_` to obtain the `Response` object it points to.
    - The function returns a reference to this `Response` object.
- **Output**: A reference to the `Response` object stored in the `Result` class.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::operator\*<!-- {{#callable:Result::operator*}} -->
The `operator*` function provides access to the `Response` object managed by the `Result` class, either as a constant or mutable reference.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the `Result` object contains a valid `Response` by dereferencing the `res_` pointer.
    - If the `Result` object is valid, it returns a reference to the `Response` object.
    - There are two overloads: one for constant access and one for mutable access.
- **Output**: A reference to the `Response` object contained within the `Result` class.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::operator\*<!-- {{#callable:Result::operator*}} -->
The non-const `operator*` function returns a reference to the `Response` object managed by the `Result` class.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private member `res_`, which is a `std::unique_ptr<Response>`.
    - It dereferences `res_` to obtain a reference to the `Response` object it manages.
    - The function returns this reference.
- **Output**: A non-const reference to the `Response` object managed by the `Result` class.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::operator\-><!-- {{#callable:Result::operator->}} -->
The `operator->` provides access to the `Response` object managed by the `Result` class through a pointer interface.
- **Inputs**: None
- **Control Flow**:
    - The `operator->` is overloaded twice, once for const access and once for non-const access.
    - Both overloads return the raw pointer to the `Response` object managed by the `std::unique_ptr` `res_`.
- **Output**: A raw pointer to the `Response` object managed by the `Result` class.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::operator\-><!-- {{#callable:Result::operator->}} -->
The `operator->` function provides access to the `Response` object managed by the `Result` class through a pointer interface.
- **Inputs**: None
- **Control Flow**:
    - The function returns the raw pointer to the `Response` object by calling `get()` on the `std::unique_ptr` member `res_`.
- **Output**: A raw pointer to the `Response` object managed by the `std::unique_ptr` `res_`.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::error<!-- {{#callable:Result::error}} -->
The `error` function returns the error state of a `Result` object.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple getter that returns the private member `err_`.
- **Output**: The function returns an `Error` object, which represents the error state of the `Result`.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::get\_request\_header\_value\_u64<!-- {{#callable:Result::get_request_header_value_u64}} -->
The `get_request_header_value_u64` function retrieves a 64-bit unsigned integer value from the request headers using a specified key and index, returning a default value if the key is not found.
- **Inputs**:
    - `key`: A string representing the key of the header whose value is to be retrieved.
    - `def`: A 64-bit unsigned integer that serves as the default value to return if the header key is not found.
    - `id`: A size_t index specifying which occurrence of the header to retrieve if there are multiple headers with the same key.
- **Control Flow**:
    - The function calls `detail::get_header_value_u64` with the `request_headers_`, `key`, `def`, and `id` as arguments.
    - The function returns the result of the `detail::get_header_value_u64` call, which is the header value as a 64-bit unsigned integer or the default value if the key is not found.
- **Output**: A 64-bit unsigned integer representing the value of the specified request header or the default value if the header is not found.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::has\_request\_header<!-- {{#callable:Result::has_request_header}} -->
The `has_request_header` function checks if a specific request header key exists in the `request_headers_` map of the `Result` class.
- **Inputs**:
    - `key`: A constant reference to a string representing the key of the request header to check for existence.
- **Control Flow**:
    - The function uses the `find` method of the `request_headers_` map to search for the provided `key`.
    - It compares the result of `find` with `request_headers_.end()` to determine if the key exists in the map.
- **Output**: Returns a boolean value: `true` if the key exists in the `request_headers_` map, otherwise `false`.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::get\_request\_header\_value<!-- {{#callable:Result::get_request_header_value}} -->
The `get_request_header_value` function retrieves the value of a specified request header from a collection of headers, returning a default value if the header is not found.
- **Inputs**:
    - `key`: A string representing the name of the header to retrieve.
    - `def`: A C-style string representing the default value to return if the header is not found.
    - `id`: A size_t index specifying which occurrence of the header to retrieve if there are multiple headers with the same name.
- **Control Flow**:
    - The function calls `detail::get_header_value` with the `request_headers_`, `key`, `def`, and `id` as arguments.
    - The `detail::get_header_value` function is responsible for the actual retrieval of the header value from the `request_headers_`.
- **Output**: A `std::string` containing the value of the specified request header, or the default value if the header is not found.
- **See also**: [`Result`](#Result)  (Data Structure)


---
#### Result::get\_request\_header\_value\_count<!-- {{#callable:Result::get_request_header_value_count}} -->
The `get_request_header_value_count` function returns the number of values associated with a specific request header key in the `Result` class.
- **Inputs**:
    - `key`: A constant reference to a string representing the header key whose values are to be counted.
- **Control Flow**:
    - The function uses the `equal_range` method on the `request_headers_` member to get a range of iterators that match the specified key.
    - It calculates the distance between the first and second iterators of the range using `std::distance`.
    - The result of the distance calculation is cast to `size_t` and returned as the count of header values.
- **Output**: The function returns a `size_t` representing the number of values associated with the specified header key.
- **See also**: [`Result`](#Result)  (Data Structure)



---
### ClientImpl<!-- {{#data_structure:ClientImpl}} -->
- **Type**: `class`
- **Members**:
    - `host_`: Stores the host address as a string.
    - `port_`: Stores the port number as an integer.
    - `host_and_port_`: Combines host and port into a single string.
    - `socket_`: Represents the current open socket.
    - `socket_mutex_`: Mutex for synchronizing access to the socket.
    - `request_mutex_`: Recursive mutex for synchronizing request operations.
    - `socket_requests_in_flight_`: Tracks the number of requests currently using the socket.
    - `socket_requests_are_from_thread_`: Stores the thread ID of the thread using the socket.
    - `socket_should_be_closed_when_request_is_done_`: Indicates if the socket should be closed after a request is completed.
    - `addr_map_`: Maps hostnames to IP addresses.
    - `default_headers_`: Stores default headers to be used in requests.
    - `header_writer_`: Function for writing headers to a stream.
    - `client_cert_path_`: Path to the client certificate file.
    - `client_key_path_`: Path to the client key file.
    - `connection_timeout_sec_`: Connection timeout in seconds.
    - `connection_timeout_usec_`: Connection timeout in microseconds.
    - `read_timeout_sec_`: Read timeout in seconds.
    - `read_timeout_usec_`: Read timeout in microseconds.
    - `write_timeout_sec_`: Write timeout in seconds.
    - `write_timeout_usec_`: Write timeout in microseconds.
    - `max_timeout_msec_`: Maximum timeout in milliseconds.
    - `basic_auth_username_`: Username for basic authentication.
    - `basic_auth_password_`: Password for basic authentication.
    - `bearer_token_auth_token_`: Token for bearer token authentication.
    - `keep_alive_`: Indicates if keep-alive is enabled.
    - `follow_location_`: Indicates if the client should follow redirects.
    - `url_encode_`: Indicates if URL encoding is enabled.
    - `address_family_`: Specifies the address family for the socket.
    - `tcp_nodelay_`: Indicates if TCP_NODELAY is enabled.
    - `ipv6_v6only_`: Indicates if IPv6-only mode is enabled.
    - `socket_options_`: Stores socket options.
    - `compress_`: Indicates if compression is enabled.
    - `decompress_`: Indicates if decompression is enabled.
    - `interface_`: Specifies the network interface to use.
    - `proxy_host_`: Stores the proxy host address.
    - `proxy_port_`: Stores the proxy port number.
    - `proxy_basic_auth_username_`: Username for proxy basic authentication.
    - `proxy_basic_auth_password_`: Password for proxy basic authentication.
    - `proxy_bearer_token_auth_token_`: Token for proxy bearer token authentication.
    - `logger_`: Logger for logging client operations.
- **Description**: The `ClientImpl` class is a comprehensive HTTP client implementation that provides functionality for making HTTP requests such as GET, POST, PUT, PATCH, DELETE, and OPTIONS. It supports various configurations including setting timeouts, authentication methods, proxy settings, and socket options. The class manages socket connections and provides mechanisms for handling request and response streams, including support for SSL/TLS if enabled. It is designed to be flexible and extensible, allowing for detailed control over HTTP communication and connection management.
- **Member Functions**:
    - [`ClientImpl::set_connection_timeout`](#ClientImplset_connection_timeout)
    - [`ClientImpl::set_read_timeout`](#ClientImplset_read_timeout)
    - [`ClientImpl::set_write_timeout`](#ClientImplset_write_timeout)
    - [`ClientImpl::set_max_timeout`](#ClientImplset_max_timeout)
    - [`ClientImpl::ClientImpl`](#ClientImplClientImpl)
    - [`ClientImpl::ClientImpl`](#ClientImplClientImpl)
    - [`ClientImpl::ClientImpl`](#ClientImplClientImpl)
    - [`ClientImpl::~ClientImpl`](#ClientImplClientImpl)
    - [`ClientImpl::is_valid`](#ClientImplis_valid)
    - [`ClientImpl::copy_settings`](#ClientImplcopy_settings)
    - [`ClientImpl::create_client_socket`](#ClientImplcreate_client_socket)
    - [`ClientImpl::create_and_connect_socket`](#ClientImplcreate_and_connect_socket)
    - [`ClientImpl::shutdown_ssl`](#ClientImplshutdown_ssl)
    - [`ClientImpl::shutdown_socket`](#ClientImplshutdown_socket)
    - [`ClientImpl::close_socket`](#ClientImplclose_socket)
    - [`ClientImpl::read_response_line`](#ClientImplread_response_line)
    - [`ClientImpl::send`](#ClientImplsend)
    - [`ClientImpl::send_`](#ClientImplsend_)
    - [`ClientImpl::send`](#ClientImplsend)
    - [`ClientImpl::send_`](#ClientImplsend_)
    - [`ClientImpl::handle_request`](#ClientImplhandle_request)
    - [`ClientImpl::redirect`](#ClientImplredirect)
    - [`ClientImpl::write_content_with_provider`](#ClientImplwrite_content_with_provider)
    - [`ClientImpl::write_request`](#ClientImplwrite_request)
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
    - [`ClientImpl::adjust_host_string`](#ClientImpladjust_host_string)
    - [`ClientImpl::process_request`](#ClientImplprocess_request)
    - [`ClientImpl::get_multipart_content_provider`](#ClientImplget_multipart_content_provider)
    - [`ClientImpl::process_socket`](#ClientImplprocess_socket)
    - [`ClientImpl::is_ssl`](#ClientImplis_ssl)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Get`](#ClientImplGet)
    - [`ClientImpl::Head`](#ClientImplHead)
    - [`ClientImpl::Head`](#ClientImplHead)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Post`](#ClientImplPost)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Put`](#ClientImplPut)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Patch`](#ClientImplPatch)
    - [`ClientImpl::Delete`](#ClientImplDelete)
    - [`ClientImpl::Delete`](#ClientImplDelete)
    - [`ClientImpl::Delete`](#ClientImplDelete)
    - [`ClientImpl::Delete`](#ClientImplDelete)
    - [`ClientImpl::Delete`](#ClientImplDelete)
    - [`ClientImpl::Delete`](#ClientImplDelete)
    - [`ClientImpl::Delete`](#ClientImplDelete)
    - [`ClientImpl::Delete`](#ClientImplDelete)
    - [`ClientImpl::Delete`](#ClientImplDelete)
    - [`ClientImpl::Delete`](#ClientImplDelete)
    - [`ClientImpl::Options`](#ClientImplOptions)
    - [`ClientImpl::Options`](#ClientImplOptions)
    - [`ClientImpl::stop`](#ClientImplstop)
    - [`ClientImpl::host`](#ClientImplhost)
    - [`ClientImpl::port`](#ClientImplport)
    - [`ClientImpl::is_socket_open`](#ClientImplis_socket_open)
    - [`ClientImpl::socket`](#ClientImplsocket)
    - [`ClientImpl::set_connection_timeout`](#ClientImplset_connection_timeout)
    - [`ClientImpl::set_read_timeout`](#ClientImplset_read_timeout)
    - [`ClientImpl::set_write_timeout`](#ClientImplset_write_timeout)
    - [`ClientImpl::set_max_timeout`](#ClientImplset_max_timeout)
    - [`ClientImpl::set_basic_auth`](#ClientImplset_basic_auth)
    - [`ClientImpl::set_bearer_token_auth`](#ClientImplset_bearer_token_auth)
    - [`ClientImpl::set_digest_auth`](#ClientImplset_digest_auth)
    - [`ClientImpl::set_keep_alive`](#ClientImplset_keep_alive)
    - [`ClientImpl::set_follow_location`](#ClientImplset_follow_location)
    - [`ClientImpl::set_url_encode`](#ClientImplset_url_encode)
    - [`ClientImpl::set_hostname_addr_map`](#ClientImplset_hostname_addr_map)
    - [`ClientImpl::set_default_headers`](#ClientImplset_default_headers)
    - [`ClientImpl::set_header_writer`](#ClientImplset_header_writer)
    - [`ClientImpl::set_address_family`](#ClientImplset_address_family)
    - [`ClientImpl::set_tcp_nodelay`](#ClientImplset_tcp_nodelay)
    - [`ClientImpl::set_ipv6_v6only`](#ClientImplset_ipv6_v6only)
    - [`ClientImpl::set_socket_options`](#ClientImplset_socket_options)
    - [`ClientImpl::set_compress`](#ClientImplset_compress)
    - [`ClientImpl::set_decompress`](#ClientImplset_decompress)
    - [`ClientImpl::set_interface`](#ClientImplset_interface)
    - [`ClientImpl::set_proxy`](#ClientImplset_proxy)
    - [`ClientImpl::set_proxy_basic_auth`](#ClientImplset_proxy_basic_auth)
    - [`ClientImpl::set_proxy_bearer_token_auth`](#ClientImplset_proxy_bearer_token_auth)
    - [`ClientImpl::set_proxy_digest_auth`](#ClientImplset_proxy_digest_auth)
    - [`ClientImpl::set_ca_cert_path`](#ClientImplset_ca_cert_path)
    - [`ClientImpl::set_ca_cert_store`](#ClientImplset_ca_cert_store)
    - [`ClientImpl::create_ca_cert_store`](#ClientImplcreate_ca_cert_store)
    - [`ClientImpl::enable_server_certificate_verification`](#ClientImplenable_server_certificate_verification)
    - [`ClientImpl::enable_server_hostname_verification`](#ClientImplenable_server_hostname_verification)
    - [`ClientImpl::set_server_certificate_verifier`](#ClientImplset_server_certificate_verifier)
    - [`ClientImpl::set_logger`](#ClientImplset_logger)

**Methods**

---
#### ClientImpl::set\_connection\_timeout<!-- {{#callable:ClientImpl::set_connection_timeout}} -->
Sets the connection timeout for the `ClientImpl` instance.
- **Inputs**:
    - `duration`: A `std::chrono::duration` object representing the timeout duration, which can be specified in various time units.
- **Control Flow**:
    - The function calls `detail::duration_to_sec_and_usec` with the provided `duration` and a lambda function.
    - The lambda function takes two parameters, `sec` and `usec`, which represent seconds and microseconds respectively.
    - Inside the lambda, it calls the [`set_connection_timeout`](#Clientset_connection_timeout) method with the extracted `sec` and `usec` values.
- **Output**: This function does not return a value; it sets the connection timeout for the client.
- **Functions called**:
    - [`Client::set_connection_timeout`](#Clientset_connection_timeout)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_read\_timeout<!-- {{#callable:ClientImpl::set_read_timeout}} -->
Sets the read timeout for the client using a specified duration.
- **Inputs**:
    - `duration`: A `std::chrono::duration` object representing the timeout duration.
- **Control Flow**:
    - Calls `detail::duration_to_sec_and_usec` to convert the `duration` into seconds and microseconds.
    - Uses a lambda function to pass the converted seconds and microseconds to the overloaded [`set_read_timeout`](#Serverset_read_timeout) method.
- **Output**: This function does not return a value; it sets the read timeout for the client.
- **Functions called**:
    - [`Server::set_read_timeout`](#Serverset_read_timeout)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_write\_timeout<!-- {{#callable:ClientImpl::set_write_timeout}} -->
Sets the write timeout for the client.
- **Inputs**:
    - `duration`: A `std::chrono::duration` object representing the desired write timeout duration.
- **Control Flow**:
    - Calls `detail::duration_to_sec_and_usec` with the provided `duration`.
    - The `duration_to_sec_and_usec` function converts the duration into seconds and microseconds, and then invokes a lambda function.
    - The lambda function calls [`set_write_timeout`](#Serverset_write_timeout) with the converted seconds and microseconds.
- **Output**: This function does not return a value; it sets the write timeout for the client based on the provided duration.
- **Functions called**:
    - [`Server::set_write_timeout`](#Serverset_write_timeout)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_max\_timeout<!-- {{#callable:ClientImpl::set_max_timeout}} -->
Sets the maximum timeout for the client in milliseconds.
- **Inputs**:
    - `duration`: A `std::chrono::duration` object representing the timeout duration.
- **Control Flow**:
    - Converts the input `duration` to milliseconds using `std::chrono::duration_cast`.
    - Calls the overloaded [`set_max_timeout`](#Clientset_max_timeout) method with the converted milliseconds.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` object by setting the maximum timeout.
- **Functions called**:
    - [`Client::set_max_timeout`](#Clientset_max_timeout)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::ClientImpl<!-- {{#callable:ClientImpl::ClientImpl}} -->
Constructs a `ClientImpl` object with a specified host and a default port of 80.
- **Inputs**:
    - `host`: A constant reference to a string representing the hostname or IP address of the server to connect to.
- **Control Flow**:
    - The constructor initializes the `ClientImpl` object by calling another constructor of the same class with the specified `host`, a default `port` of 80, and two empty strings for `client_cert_path` and `client_key_path`.
- **Output**: This constructor does not return a value; it initializes the `ClientImpl` instance with the provided host and default values for other parameters.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::ClientImpl<!-- {{#callable:ClientImpl::ClientImpl}} -->
Constructs a `ClientImpl` object with specified host and port.
- **Inputs**:
    - `host`: A constant reference to a `std::string` representing the hostname or IP address of the server.
    - `port`: An integer representing the port number on which the server is listening.
- **Control Flow**:
    - The constructor initializes a `ClientImpl` object by delegating to another constructor of the same class.
    - It passes the `host` and `port` parameters along with two empty `std::string` arguments for client certificate and key paths.
- **Output**: This constructor does not return a value; it initializes the `ClientImpl` instance.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::ClientImpl<!-- {{#callable:ClientImpl::ClientImpl}} -->
Constructs a `ClientImpl` object with specified host, port, and optional client certificate and key paths.
- **Inputs**:
    - `host`: A string representing the hostname or IP address of the server.
    - `port`: An integer representing the port number to connect to.
    - `client_cert_path`: A string representing the file path to the client's certificate.
    - `client_key_path`: A string representing the file path to the client's private key.
- **Control Flow**:
    - The constructor initializes member variables using the provided parameters.
    - The `host_` member is set by escaping the abstract namespace for Unix domain sockets.
    - The `host_and_port_` member is constructed by adjusting the host string and appending the port number.
- **Output**: The constructor does not return a value but initializes an instance of `ClientImpl` with the specified configuration.
- **Functions called**:
    - [`ClientImpl::adjust_host_string`](#ClientImpladjust_host_string)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::\~ClientImpl<!-- {{#callable:ClientImpl::~ClientImpl}} -->
The `ClientImpl` destructor waits for all in-flight requests to complete before shutting down and closing the socket.
- **Inputs**: None
- **Control Flow**:
    - The destructor initializes a retry count to 10 and enters a loop to check if there are any requests in flight.
    - Within the loop, it locks the `socket_mutex_` to safely check the `socket_requests_in_flight_` counter.
    - If there are no requests in flight, it breaks out of the loop; otherwise, it sleeps for 1 millisecond before retrying.
    - After exiting the loop, it locks the `socket_mutex_` again to perform socket shutdown and closure operations.
    - The [`shutdown_socket`](#shutdown_socket) and [`close_socket`](#close_socket) functions are called to properly terminate the socket connection.
- **Output**: The function does not return a value; it performs cleanup operations to ensure that the socket is properly shut down and closed.
- **Functions called**:
    - [`shutdown_socket`](#shutdown_socket)
    - [`close_socket`](#close_socket)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::is\_valid<!-- {{#callable:ClientImpl::is_valid}} -->
The `is_valid` method of the `ClientImpl` class checks if the client instance is valid.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as an inline method, which means it is intended to be small and efficient.
    - It directly returns a boolean value of `true`, indicating that the client is always considered valid.
- **Output**: The output is a boolean value, specifically `true`, indicating that the client instance is valid.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::copy\_settings<!-- {{#callable:ClientImpl::copy_settings}} -->
Copies the settings from one `ClientImpl` instance to another.
- **Inputs**:
    - `rhs`: A constant reference to another `ClientImpl` instance from which settings will be copied.
- **Control Flow**:
    - The function accesses each setting in the `rhs` instance and assigns it to the corresponding member variable of the current instance.
    - It uses direct member access to copy values, ensuring that all relevant settings are duplicated.
    - Conditional compilation directives are used to include or exclude certain settings based on whether OpenSSL support is enabled.
- **Output**: The function does not return a value; it modifies the current instance's member variables to match those of the `rhs` instance.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::create\_client\_socket<!-- {{#callable:ClientImpl::create_client_socket}} -->
Creates a client socket for connecting to a server, optionally using a proxy.
- **Inputs**:
    - `error`: An `Error` reference that will be populated with error information if socket creation fails.
- **Control Flow**:
    - Checks if a proxy host and port are specified; if so, it calls `detail::create_client_socket` with proxy parameters.
    - If no proxy is used, it checks if a custom IP address is mapped for the specified host and uses it if available.
    - Finally, it calls `detail::create_client_socket` with the host, port, and other socket options.
- **Output**: Returns a `socket_t` representing the created client socket.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::create\_and\_connect\_socket<!-- {{#callable:ClientImpl::create_and_connect_socket}} -->
Creates a client socket and connects it, updating the provided `Socket` object.
- **Inputs**:
    - `socket`: A reference to a `Socket` object that will be updated with the newly created socket.
    - `error`: A reference to an `Error` object that will capture any error that occurs during socket creation.
- **Control Flow**:
    - Calls [`create_client_socket`](#create_client_socket) to attempt to create a new socket, passing the `error` reference to capture any issues.
    - Checks if the returned socket is `INVALID_SOCKET`; if so, it returns false indicating failure.
    - If the socket is valid, it assigns the socket to the `socket.sock` member and returns true.
- **Output**: Returns a boolean indicating the success of the socket creation and connection process.
- **Functions called**:
    - [`create_client_socket`](#create_client_socket)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::shutdown\_ssl<!-- {{#callable:ClientImpl::shutdown_ssl}} -->
The `shutdown_ssl` function ensures that SSL shutdown is performed safely by asserting that no requests are in flight from other threads.
- **Inputs**:
    - `socket`: A reference to a `Socket` object that represents the connection to be shut down.
    - `shutdown_gracefully`: A boolean flag indicating whether the shutdown should be performed gracefully.
- **Control Flow**:
    - The function checks if there are any requests currently in flight using the `socket_requests_in_flight_` member variable.
    - If there are requests in flight, it asserts that those requests are from the same thread as the current one using `socket_requests_are_from_thread_` and `std::this_thread::get_id()`.
- **Output**: The function does not return a value; it asserts conditions to ensure thread safety during SSL shutdown.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::shutdown\_socket<!-- {{#callable:ClientImpl::shutdown_socket}} -->
The `shutdown_socket` function safely shuts down a socket connection if it is valid.
- **Inputs**:
    - `socket`: A reference to a `Socket` object that represents the socket connection to be shut down.
- **Control Flow**:
    - The function first checks if the socket's `sock` member is equal to `INVALID_SOCKET`.
    - If the socket is invalid, the function returns immediately without performing any action.
    - If the socket is valid, it calls the `detail::shutdown_socket` function, passing the socket's `sock` member to properly shut down the connection.
- **Output**: The function does not return a value; it performs an action to shut down the socket connection.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::close\_socket<!-- {{#callable:ClientImpl::close_socket}} -->
Closes a socket and ensures it is safe to do so by checking for active requests and SSL status.
- **Inputs**:
    - `socket`: A reference to a `Socket` object that represents the socket to be closed.
- **Control Flow**:
    - Asserts that there are no requests in flight or that they are from the current thread to prevent race conditions.
    - Checks if the socket is already invalid; if so, it returns immediately.
    - Calls a helper function `detail::close_socket` to perform the actual closing of the socket.
    - Sets the socket's identifier to `INVALID_SOCKET` after closing.
- **Output**: This function does not return a value; it modifies the state of the provided `Socket` by closing it and marking it as invalid.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::read\_response\_line<!-- {{#callable:ClientImpl::read_response_line}} -->
Reads a response line from a stream and populates the response object with the HTTP version, status code, and reason phrase.
- **Inputs**:
    - `strm`: A reference to a `Stream` object from which the response line is read.
    - `req`: A reference to a `Request` object that contains the HTTP request details.
    - `res`: A reference to a `Response` object that will be populated with the parsed response data.
- **Control Flow**:
    - Initializes a buffer and a line reader to read from the provided stream.
    - Checks if a line can be read; if not, returns false.
    - Defines a regex pattern to match the HTTP response line based on whether line terminators are allowed.
    - Uses regex to match the read line; if it fails, checks if the request method is 'CONNECT'.
    - If the regex matches, extracts the HTTP version, status code, and reason phrase into the response object.
    - Handles '100 Continue' responses by reading additional lines until a different status is encountered.
- **Output**: Returns a boolean indicating the success of reading and parsing the response line.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::send<!-- {{#callable:ClientImpl::send}} -->
The `send` method in `ClientImpl` sends a request and handles potential SSL errors.
- **Inputs**:
    - `req`: A reference to a `Request` object that contains the details of the request to be sent.
    - `res`: A reference to a `Response` object that will be populated with the response from the server.
    - `error`: A reference to an `Error` object that will capture any errors that occur during the send operation.
- **Control Flow**:
    - The method begins by acquiring a lock on `request_mutex_` to ensure thread safety during the request sending process.
    - It then calls the private method [`send_`](#ClientImplsend_) with the provided `req`, `res`, and `error` to attempt sending the request.
    - If the `error` indicates that the SSL peer could be closed, it asserts that the first send attempt failed and retries sending the request.
    - Finally, it returns the result of the send operation, indicating success or failure.
- **Output**: The method returns a boolean value indicating whether the request was successfully sent or not.
- **Functions called**:
    - [`ClientImpl::send_`](#ClientImplsend_)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::send\_<!-- {{#callable:ClientImpl::send_}} -->
The `send_` function handles sending a request over a socket, managing connection states and ensuring thread safety.
- **Inputs**:
    - `req`: A reference to a `Request` object that contains the details of the HTTP request to be sent.
    - `res`: A reference to a `Response` object that will be populated with the server's response.
    - `error`: A reference to an `Error` object that will capture any errors that occur during the request process.
- **Control Flow**:
    - The function begins by acquiring a lock on `socket_mutex_` to ensure thread safety.
    - It checks if the socket is open and alive, and if not, attempts to create and connect a new socket.
    - If SSL support is enabled, it checks the SSL connection state and initializes it if necessary.
    - The function increments the count of requests in flight and marks the current thread as the owner of the socket.
    - Default headers are added to the request if they are not already present.
    - A scope exit guard is set up to ensure that resources are cleaned up after the request is processed.
    - The actual request handling is done by calling [`process_socket`](#ClientImplprocess_socket), which processes the request and response.
    - Finally, the function returns a boolean indicating the success or failure of the request.
- **Output**: Returns a boolean value indicating whether the request was successfully sent and processed.
- **Functions called**:
    - [`ClientImpl::is_ssl`](#ClientImplis_ssl)
    - [`ClientImpl::shutdown_ssl`](#ClientImplshutdown_ssl)
    - [`shutdown_socket`](#shutdown_socket)
    - [`close_socket`](#close_socket)
    - [`ClientImpl::create_and_connect_socket`](#ClientImplcreate_and_connect_socket)
    - [`ClientImpl::process_socket`](#ClientImplprocess_socket)
    - [`ClientImpl::handle_request`](#ClientImplhandle_request)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::send<!-- {{#callable:ClientImpl::send}} -->
Sends a `Request` object and returns the result.
- **Inputs**:
    - `req`: A constant reference to a `Request` object that contains the details of the request to be sent.
- **Control Flow**:
    - Creates a copy of the input `Request` object named `req2`.
    - Calls the private method [`send_`](#ClientImplsend_) with `req2` moved to it, which handles the actual sending of the request.
- **Output**: Returns a `Result` object that encapsulates the outcome of the send operation.
- **Functions called**:
    - [`ClientImpl::send_`](#ClientImplsend_)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::send\_<!-- {{#callable:ClientImpl::send_}} -->
The `send_` function sends a `Request` and returns a `Result` containing the response or error.
- **Inputs**:
    - `req`: An rvalue reference to a `Request` object that contains the details of the request to be sent.
- **Control Flow**:
    - A unique pointer to a `Response` object is created to hold the response from the server.
    - An `Error` variable is initialized to `Error::Success` to track any errors during the send operation.
    - The [`send`](#ClientImplsend) method is called with the request, response, and error variables, which attempts to send the request and populate the response.
    - The function returns a `Result` object that contains the response if the send operation was successful, or a null pointer if it failed, along with the error status and the original request headers.
- **Output**: A `Result` object that encapsulates the response from the server, any error that occurred during the send operation, and the headers from the original request.
- **Functions called**:
    - [`ClientImpl::send`](#ClientImplsend)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::handle\_request<!-- {{#callable:ClientImpl::handle_request}} -->
Handles an HTTP request by processing it, managing SSL connections, and handling redirects and authentication.
- **Inputs**:
    - `strm`: A reference to a `Stream` object used for reading and writing data.
    - `req`: A reference to a `Request` object representing the HTTP request to be processed.
    - `res`: A reference to a `Response` object where the HTTP response will be stored.
    - `close_connection`: A boolean indicating whether the connection should be closed after the request.
    - `error`: A reference to an `Error` object that will be set if an error occurs during processing.
- **Control Flow**:
    - Checks if the request path is empty and sets an error if it is.
    - Saves the original request for potential redirection.
    - If not using SSL and a proxy is configured, modifies the request path to route through the proxy and processes it.
    - Processes the request directly if no proxy is used.
    - Checks the response headers to determine if the connection should be closed and safely shuts down the socket if necessary.
    - Handles HTTP redirects if the response status indicates a redirection and following is enabled.
    - Handles authentication challenges for unauthorized responses, potentially retrying the request with updated credentials.
- **Output**: Returns a boolean indicating the success or failure of the request handling process.
- **Functions called**:
    - [`ClientImpl::is_ssl`](#ClientImplis_ssl)
    - [`Server::process_request`](#Serverprocess_request)
    - [`ClientImpl::shutdown_ssl`](#ClientImplshutdown_ssl)
    - [`shutdown_socket`](#shutdown_socket)
    - [`close_socket`](#close_socket)
    - [`redirect`](#redirect)
    - [`ClientImpl::send`](#ClientImplsend)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::redirect<!-- {{#callable:ClientImpl::redirect}} -->
The `redirect` function handles HTTP redirection by processing the response's location header and managing the redirect logic based on the current request and response.
- **Inputs**:
    - `req`: A reference to a `Request` object representing the current HTTP request.
    - `res`: A reference to a `Response` object containing the HTTP response that may include a redirection.
    - `error`: A reference to an `Error` object that will be populated with error information if the redirection fails.
- **Control Flow**:
    - The function first checks if the redirect count in `req` is zero, setting an error and returning false if so.
    - It retrieves the 'location' header from `res` and checks if it is empty, returning false if it is.
    - A regex is used to parse the location URL, and if it fails to match, the function returns false.
    - The scheme, host, port, path, and query components of the URL are extracted and default values are assigned if necessary.
    - If the next scheme, host, and port match the current request's, it calls a detail function to perform the redirect.
    - If the next scheme is HTTPS, it creates an `SSLClient` and performs the redirect; otherwise, it creates a new `ClientImpl` for HTTP and performs the redirect.
- **Output**: The function returns a boolean indicating the success or failure of the redirection process.
- **Functions called**:
    - [`ClientImpl::is_ssl`](#ClientImplis_ssl)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::write\_content\_with\_provider<!-- {{#callable:ClientImpl::write_content_with_provider}} -->
The `write_content_with_provider` function writes content to a stream using a specified content provider, supporting both chunked and non-chunked content.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where the content will be written.
    - `req`: A reference to a `Request` object that contains information about the content provider and whether the content is chunked.
    - `error`: A reference to an `Error` object that will be populated with error information if the operation fails.
- **Control Flow**:
    - The function defines a lambda `is_shutting_down` that always returns false, indicating that the shutdown state is not active.
    - It checks if the `Request` object indicates chunked content by evaluating `req.is_chunked_content_provider_`.
    - If chunked content is indicated, it creates a compressor (either gzip or no compression) based on the `compress_` flag.
    - The function then calls `detail::write_content_chunked` to write the content in chunks, passing the stream, content provider, shutdown state, compressor, and error reference.
    - If the content is not chunked, it calls `detail::write_content` to write the entire content at once, using the specified content length and the same shutdown state and error reference.
- **Output**: Returns a boolean indicating the success or failure of the content writing operation.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::write\_request<!-- {{#callable:ClientImpl::write_request}} -->
The `write_request` function prepares and sends an HTTP request over a specified stream.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where the request will be written.
    - `req`: A reference to a `Request` object containing the details of the HTTP request.
    - `close_connection`: A boolean indicating whether to close the connection after the request.
    - `error`: A reference to an `Error` object that will be set if an error occurs during the request.
- **Control Flow**:
    - Checks if the connection should be closed and sets the 'Connection' header accordingly.
    - Sets the 'Host' header based on whether the request is SSL and the port number.
    - Sets default headers like 'Accept', 'Accept-Encoding', and 'User-Agent' if they are not already present.
    - Determines the 'Content-Length' header based on the request body or content provider.
    - Handles basic and bearer token authentication by setting the appropriate headers if credentials are provided.
    - Writes the request line and headers to a buffer stream.
    - Writes the request body to the stream, either directly or through a content provider.
- **Output**: Returns a boolean indicating the success of the write operation; if false, the `error` object is set to indicate the type of error encountered.
- **Functions called**:
    - [`ClientImpl::is_ssl`](#ClientImplis_ssl)
    - [`make_basic_authentication_header`](#make_basic_authentication_header)
    - [`make_bearer_token_authentication_header`](#make_bearer_token_authentication_header)
    - [`append_query_params`](#append_query_params)
    - [`Server::write_content_with_provider`](#Serverwrite_content_with_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::send\_with\_content\_provider<!-- {{#callable:ClientImpl::send_with_content_provider}} -->
Sends a request with a content provider, optionally compressing the body.
- **Inputs**:
    - `req`: A reference to a `Request` object that contains the details of the HTTP request.
    - `body`: A pointer to a character array representing the body of the request.
    - `content_length`: The size of the content in bytes.
    - `content_provider`: A function that provides content for the request.
    - `content_provider_without_length`: A function that provides content without a specified length.
    - `content_type`: A string representing the MIME type of the content.
    - `error`: An `Error` object that captures any errors that occur during the request.
- **Control Flow**:
    - Checks if the `content_type` is not empty and sets it in the request headers.
    - If compression is enabled, it sets the `Content-Encoding` header to 'gzip'.
    - If compression is enabled and a content provider is provided, it initializes a `gzip_compressor` and processes the content in chunks, compressing it as it is written to the request body.
    - If no content provider is provided, it directly assigns the body to the request.
    - If no compression is enabled, it checks the type of content provider and sets the request's content length and provider accordingly.
    - Finally, it creates a `Response` object and sends the request, returning the response or null if an error occurred.
- **Output**: Returns a unique pointer to a `Response` object containing the server's response to the request, or null if an error occurred.
- **Functions called**:
    - [`ClientImpl::send`](#ClientImplsend)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::send\_with\_content\_provider<!-- {{#callable:ClientImpl::send_with_content_provider}} -->
Sends an HTTP request with a specified content provider and returns the result.
- **Inputs**:
    - `method`: The HTTP method (e.g., GET, POST) to be used for the request.
    - `path`: The endpoint path for the request.
    - `headers`: A collection of HTTP headers to include in the request.
    - `body`: A pointer to the body content of the request.
    - `content_length`: The length of the body content.
    - `content_provider`: A function or callable that provides the content for the request.
    - `content_provider_without_length`: A function or callable that provides content without specifying its length.
    - `content_type`: The MIME type of the content being sent.
    - `progress`: A callback function to report progress of the request.
- **Control Flow**:
    - A `Request` object is created and populated with the method, path, headers, and progress.
    - If a maximum timeout is set, the current time is recorded as the start time.
    - An `Error` object is initialized to track any errors during the request.
    - The function [`send_with_content_provider`](#ClientImplsend_with_content_provider) is called with the populated request and other parameters.
    - The result of the send operation and any errors are captured and returned.
- **Output**: Returns a `Result` object containing the response from the server, any error that occurred, and the headers of the request.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::adjust\_host\_string<!-- {{#callable:ClientImpl::adjust_host_string}} -->
Adjusts the given host string to ensure it is properly formatted for use in network requests.
- **Inputs**:
    - `host`: A constant reference to a string representing the host address that needs to be adjusted.
- **Control Flow**:
    - Checks if the `host` string contains a colon (':') character.
    - If a colon is found, it wraps the `host` in square brackets ('[' and ']') and returns the modified string.
    - If no colon is found, it simply returns the original `host` string.
- **Output**: Returns a string that is either the original `host` or the `host` wrapped in square brackets if it contained a colon.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::process\_request<!-- {{#callable:ClientImpl::process_request}} -->
Processes an HTTP request by sending it, reading the response, and handling the response body and headers.
- **Inputs**:
    - `strm`: A reference to a `Stream` object used for sending the request and receiving the response.
    - `req`: A reference to a `Request` object that contains the details of the HTTP request to be sent.
    - `res`: A reference to a `Response` object that will be populated with the response data from the server.
    - `close_connection`: A boolean indicating whether the connection should be closed after the request is processed.
    - `error`: A reference to an `Error` object that will be set if any error occurs during the request processing.
- **Control Flow**:
    - The function first attempts to send the request using [`write_request`](#ClientImplwrite_request). If this fails, it returns false.
    - If SSL support is enabled and the connection is SSL, it checks if the proxy is enabled and whether the SSL peer can be closed.
    - Next, it reads the response line and headers from the stream. If this fails, it sets an error and returns false.
    - If the response status is not '204 No Content' and the request method is not 'HEAD' or 'CONNECT', it processes the response body.
    - It checks for redirection and invokes the response handler if provided, handling cancellation if necessary.
    - If a content receiver is specified, it uses it to process the response body; otherwise, it appends the body to the response object.
    - It also handles progress reporting if a progress function is provided.
    - Finally, if logging is enabled, it logs the request and response before returning true.
- **Output**: Returns a boolean indicating the success or failure of the request processing.
- **Functions called**:
    - [`ClientImpl::write_request`](#ClientImplwrite_request)
    - [`ClientImpl::is_ssl`](#ClientImplis_ssl)
    - [`ClientImpl::read_response_line`](#ClientImplread_response_line)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::get\_multipart\_content\_provider<!-- {{#callable:ClientImpl::get_multipart_content_provider}} -->
Creates a content provider for multipart form data.
- **Inputs**:
    - `boundary`: A string that defines the boundary used to separate parts in the multipart form data.
    - `items`: A collection of `MultipartFormDataItems` representing the form data items.
    - `provider_items`: A collection of `MultipartFormDataProviderItems` that provide the data for each part.
- **Control Flow**:
    - Initializes `cur_item` and `cur_start` to track the current item and its start position.
    - Returns a lambda function that takes an offset and a `DataSink` to write the serialized multipart data.
    - If the offset is zero and there are items, it serializes the initial part of the multipart data.
    - If there are more provider items, it serializes the beginning of the current item and writes data to the sink.
    - If the current item has no more data, it serializes the end of the current item and increments `cur_item`.
    - If all items have been processed, it serializes the finish of the multipart data and marks the sink as done.
- **Output**: Returns a `ContentProviderWithoutLength` that can be used to stream multipart form data.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::process\_socket<!-- {{#callable:ClientImpl::process_socket}} -->
Processes a socket by invoking a callback function with a stream.
- **Inputs**:
    - `socket`: A `Socket` object representing the socket to be processed.
    - `start_time`: A `std::chrono::time_point` indicating the start time for processing.
    - `callback`: A `std::function` that takes a `Stream` reference and returns a boolean, used for processing the stream.
- **Control Flow**:
    - Calls the `detail::process_client_socket` function with the socket's file descriptor and various timeout parameters.
    - Passes the `start_time` and the `callback` function to `process_client_socket` for further processing.
- **Output**: Returns a boolean indicating the success or failure of the socket processing.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::is\_ssl<!-- {{#callable:ClientImpl::is_ssl}} -->
Determines if the `ClientImpl` instance is using SSL.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple inline method that directly returns a boolean value.
    - It does not contain any conditional statements or loops.
- **Output**: Returns a boolean value, specifically 'false', indicating that SSL is not being used.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
The [`Get`](#ServerGet) function in the `ClientImpl` class initiates a GET request to the specified path with default headers and progress tracking.
- **Inputs**:
    - `path`: A `std::string` representing the endpoint path to which the GET request is sent.
- **Control Flow**:
    - The function calls another overloaded version of [`Get`](#ServerGet), passing the `path` along with default constructed `Headers` and `Progress` objects.
    - The actual request handling and response processing is delegated to the more complex [`Get`](#ServerGet) method that takes additional parameters.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, which may include the response data or error information.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
This function initiates a GET request to a specified path with default headers and a progress callback.
- **Inputs**:
    - `path`: A string representing the endpoint path to which the GET request is sent.
    - `progress`: A `Progress` object that allows tracking the progress of the request.
- **Control Flow**:
    - The function calls another overloaded version of [`Get`](#ServerGet) with the provided `path`, default headers (an empty `Headers` object), and the `progress` object.
    - The actual logic for handling the GET request is encapsulated in the overloaded [`Get`](#ServerGet) method that is invoked.
- **Output**: The function returns a `Result` object that contains the outcome of the GET request, which may include the response data or error information.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
This function initiates a GET request to a specified path with optional headers.
- **Inputs**:
    - `path`: A string representing the endpoint path for the GET request.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
- **Control Flow**:
    - The function calls another overloaded version of [`Get`](#ServerGet) with the same `path` and `headers`, and a default `Progress` object.
    - The `Progress` object is likely used to track the progress of the request, although its specific implementation is not detailed in this snippet.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, which may include response data or error information.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
Executes an HTTP GET request with the specified path, headers, and progress.
- **Inputs**:
    - `path`: A string representing the URL path to which the GET request is sent.
    - `headers`: An object containing key-value pairs representing the HTTP headers to include in the request.
    - `progress`: A callable that is invoked to report the progress of the request.
- **Control Flow**:
    - Creates a `Request` object and sets its method to 'GET'.
    - Assigns the provided `path`, `headers`, and `progress` to the corresponding fields in the `Request` object.
    - If `max_timeout_msec_` is greater than 0, records the current time as the start time of the request.
    - Calls the [`send_`](#ClientImplsend_) method with the constructed `Request` object to execute the GET request.
- **Output**: Returns a `Result` object that contains the outcome of the GET request, which may include the response data or an error.
- **Functions called**:
    - [`ClientImpl::send_`](#ClientImplsend_)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
This function initiates a GET request to a specified path using a content receiver.
- **Inputs**:
    - `path`: A string representing the endpoint path for the GET request.
    - `content_receiver`: A callable that will receive the content of the response.
- **Control Flow**:
    - The function calls another overloaded [`Get`](#ServerGet) method with the provided `path`, an empty `Headers` object, and the `content_receiver` moved into the call.
    - The other parameters in the call to [`Get`](#ServerGet) are set to `nullptr`, indicating that no additional headers or progress tracking is used.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including any response data or error information.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
The [`Get`](#ServerGet) method in `ClientImpl` initiates a GET request to a specified path using a content receiver and progress callback.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the GET request is sent.
    - `content_receiver`: A `ContentReceiver` function that processes the content received from the GET request.
    - `progress`: A `Progress` callback that is invoked to report the progress of the request.
- **Control Flow**:
    - The method calls another overloaded version of [`Get`](#ServerGet), passing the `path`, an empty `Headers` object, a `nullptr` for the response handler, and the moved `content_receiver` and `progress` arguments.
    - This allows for a more flexible handling of the GET request by utilizing the existing overloaded methods that can handle additional parameters.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, which may include the response data or an error.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
The [`Get`](#ServerGet) method in `ClientImpl` initiates a GET request to a specified path with optional headers and a content receiver.
- **Inputs**:
    - `path`: A `std::string` representing the endpoint path for the GET request.
    - `headers`: A `Headers` object containing any additional headers to be sent with the request.
    - `content_receiver`: A `ContentReceiver` function or object that will handle the content received from the GET request.
- **Control Flow**:
    - The method calls another overloaded version of [`Get`](#ServerGet), passing the `path`, `headers`, and `content_receiver` while providing `nullptr` for the other parameters.
    - This allows for a more flexible handling of the request by delegating to a more complex implementation of the [`Get`](#ServerGet) method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including success or failure information.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
This function initiates a GET request to a specified path with optional headers, a content receiver, and progress tracking.
- **Inputs**:
    - `path`: A string representing the endpoint path for the GET request.
    - `headers`: An object of type `Headers` containing any additional headers to be sent with the request.
    - `content_receiver`: A callable that will receive the content of the response.
    - `progress`: A callable to track the progress of the request.
- **Control Flow**:
    - The function calls another overloaded version of [`Get`](#ServerGet), passing the same `path` and `headers`, while providing `nullptr` for the response handler.
    - It moves the `content_receiver` and `progress` arguments to the called function, allowing for efficient resource management.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including success or failure information.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
This function initiates a GET request to a specified path using a response handler and a content receiver.
- **Inputs**:
    - `path`: A string representing the endpoint path for the GET request.
    - `response_handler`: A callable that processes the response received from the GET request.
    - `content_receiver`: A callable that receives the content of the response.
- **Control Flow**:
    - The function calls another overloaded [`Get`](#ServerGet) method with the provided `path`, default headers, and the moved `response_handler` and `content_receiver`.
    - The `nullptr` is passed for the progress parameter, indicating that no progress tracking is required for this request.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including success or failure information.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
The [`Get`](#ServerGet) method in `ClientImpl` initiates an HTTP GET request with specified headers, a response handler, and a content receiver.
- **Inputs**:
    - `path`: A `std::string` representing the URL path for the GET request.
    - `headers`: A `Headers` object containing key-value pairs for HTTP headers to be sent with the request.
    - `response_handler`: A `ResponseHandler` function that processes the response received from the server.
    - `content_receiver`: A `ContentReceiver` function that handles the content received in the response.
- **Control Flow**:
    - The method calls another overloaded [`Get`](#ServerGet) method, passing the same `path` and `headers`, along with moved versions of `response_handler` and `content_receiver`.
    - The last argument is set to `nullptr`, indicating that no progress handler is provided.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including success or error information.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
The [`Get`](#ServerGet) function in `ClientImpl` initiates a GET request to a specified path using default headers and provided response handling mechanisms.
- **Inputs**:
    - `path`: A `std::string` representing the endpoint path for the GET request.
    - `response_handler`: A `ResponseHandler` function that processes the response received from the GET request.
    - `content_receiver`: A `ContentReceiver` function that handles the content received from the GET request.
    - `progress`: A `Progress` function that tracks the progress of the GET request.
- **Control Flow**:
    - The function calls another overloaded [`Get`](#ServerGet) method with the provided `path`, default headers, and moved arguments for `response_handler`, `content_receiver`, and `progress`.
    - The use of `std::move` indicates that the function takes ownership of the `response_handler`, `content_receiver`, and `progress` arguments, allowing for efficient resource management.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including success or failure information.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
Executes an HTTP GET request with specified parameters.
- **Inputs**:
    - `path`: A string representing the URL path to which the GET request is sent.
    - `headers`: An object containing key-value pairs representing HTTP headers to be included in the request.
    - `response_handler`: A callback function that processes the HTTP response received from the server.
    - `content_receiver`: A callback function that handles the content received in the response.
    - `progress`: A callback function that reports the progress of the request.
- **Control Flow**:
    - A `Request` object is created and populated with the HTTP method set to 'GET', the specified path, headers, and callback functions.
    - If a maximum timeout is set, the current time is recorded as the start time of the request.
    - The [`send_`](#ClientImplsend_) method is called with the constructed request, which handles the actual sending of the request and returns the result.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including any response data or error information.
- **Functions called**:
    - [`ClientImpl::send_`](#ClientImplsend_)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
This function retrieves a resource from a specified path, optionally appending query parameters and handling headers.
- **Inputs**:
    - `path`: A string representing the resource path to retrieve.
    - `params`: A `Params` object containing query parameters to be appended to the path.
    - `headers`: A `Headers` object containing any additional headers to include in the request.
    - `progress`: A `Progress` callback function to report progress of the request.
- **Control Flow**:
    - The function first checks if the `params` object is empty.
    - If `params` is empty, it calls another overload of [`Get`](#ServerGet) with just the `path` and `headers`.
    - If `params` is not empty, it constructs a new path by appending the query parameters to the original `path` using [`append_query_params`](#append_query_params).
    - Finally, it calls another overload of [`Get`](#ServerGet) with the new path, `headers`, and the `progress` callback.
- **Output**: Returns a `Result` object representing the outcome of the GET request, which may include the response data or an error.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
    - [`append_query_params`](#append_query_params)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
This function initiates a GET request to a specified path with optional parameters, headers, content receiver, and progress tracking.
- **Inputs**:
    - `path`: A string representing the endpoint path for the GET request.
    - `params`: An object of type `Params` containing query parameters to be included in the request.
    - `headers`: An object of type `Headers` representing additional headers to be sent with the request.
    - `content_receiver`: A callable object of type `ContentReceiver` that will handle the response content.
    - `progress`: An object of type `Progress` that can be used to track the progress of the request.
- **Control Flow**:
    - The function calls another overloaded version of [`Get`](#ServerGet), passing the same parameters but with a null pointer for the response handler.
    - It uses `std::move` to efficiently transfer ownership of `content_receiver` and `progress` to the called function.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including success or failure information.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Get<!-- {{#callable:ClientImpl::Get}} -->
The [`Get`](#ServerGet) method in `ClientImpl` performs an HTTP GET request, optionally appending query parameters to the request path.
- **Inputs**:
    - `path`: A `std::string` representing the URL path for the GET request.
    - `params`: A `Params` object containing query parameters to be appended to the URL.
    - `headers`: A `Headers` object containing any additional headers to include in the request.
    - `response_handler`: A `ResponseHandler` function to handle the response from the server.
    - `content_receiver`: A `ContentReceiver` function to receive the content of the response.
    - `progress`: A `Progress` function to report the progress of the request.
- **Control Flow**:
    - The function first checks if the `params` object is empty.
    - If `params` is empty, it calls another overload of [`Get`](#ServerGet) with the original `path` and the provided headers and handlers.
    - If `params` is not empty, it appends the query parameters to the `path` using the [`append_query_params`](#append_query_params) function.
    - Finally, it calls the same overload of [`Get`](#ServerGet) with the modified `path_with_query` and the provided headers and handlers.
- **Output**: The function returns a `Result` object representing the outcome of the GET request, which may include the response data or an error.
- **Functions called**:
    - [`Server::Get`](#ServerGet)
    - [`append_query_params`](#append_query_params)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Head<!-- {{#callable:ClientImpl::Head}} -->
The [`Head`](#ClientImplHead) function sends an HTTP HEAD request to the specified path.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the HEAD request is sent.
- **Control Flow**:
    - The function calls another overloaded [`Head`](#ClientImplHead) method with the provided `path` and an empty `Headers` object.
    - This allows the function to utilize the existing logic for handling HEAD requests with default headers.
- **Output**: Returns a `Result` object that contains the response from the HEAD request.
- **Functions called**:
    - [`ClientImpl::Head`](#ClientImplHead)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Head<!-- {{#callable:ClientImpl::Head}} -->
The `Head` method sends an HTTP HEAD request to the specified path with optional headers.
- **Inputs**:
    - `path`: A string representing the URL path to which the HEAD request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
- **Control Flow**:
    - A `Request` object is created and initialized with the method set to 'HEAD', the provided path, and headers.
    - If `max_timeout_msec_` is greater than 0, the current time is recorded as the start time of the request.
    - The [`send_`](#ClientImplsend_) method is called with the constructed request, and the result is returned.
- **Output**: The output is a `Result` object that contains the response from the server after the HEAD request is processed.
- **Functions called**:
    - [`ClientImpl::send_`](#ClientImplsend_)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
The [`Post`](#ServerPost) function in the `ClientImpl` class initiates a POST request to a specified path without a body or headers.
- **Inputs**:
    - `path`: A `std::string` representing the endpoint path to which the POST request is sent.
- **Control Flow**:
    - The function calls another overloaded version of [`Post`](#ServerPost), passing the `path` along with two empty `std::string` arguments for body and content type.
    - The actual logic for handling the POST request is delegated to the overloaded [`Post`](#ServerPost) method that accepts more parameters.
- **Output**: The function returns a `Result` object that encapsulates the outcome of the POST request, which may include the response status and data.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
The [`Post`](#ServerPost) function in the `ClientImpl` class initiates an HTTP POST request to a specified path with optional headers.
- **Inputs**:
    - `path`: A `std::string` representing the endpoint path to which the POST request is sent.
    - `headers`: A `Headers` object containing key-value pairs representing the HTTP headers to be included in the request.
- **Control Flow**:
    - The function calls another overloaded version of [`Post`](#ServerPost), passing the `path`, `headers`, and default values for the body and content length.
    - The default values for the body and content length are set to `nullptr` and `0`, respectively, indicating that no body content is sent with the request.
- **Output**: The function returns a `Result` object that encapsulates the outcome of the POST request, which may include the response status, headers, and body.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
Sends a POST request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the endpoint path to which the POST request is sent.
    - `body`: A pointer to a character array containing the body of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The function calls another overloaded [`Post`](#ServerPost) method, passing the provided parameters along with an empty `Headers` object and a null pointer for the progress parameter.
    - The actual sending of the POST request and handling of the response is managed by the called [`Post`](#ServerPost) method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, which may include success or error information.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
Sends a POST request to a specified path with given headers and body content.
- **Inputs**:
    - `path`: A string representing the endpoint path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for the HTTP headers.
    - `body`: A pointer to a character array representing the body content of the POST request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method 'POST' and the provided parameters.
    - The [`send_with_content_provider`](#ClientImplsend_with_content_provider) function handles the actual sending of the request and returns the result.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, including success or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
Sends a POST request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the endpoint path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for HTTP headers.
    - `body`: A pointer to a character array representing the body content of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the body content.
    - `progress`: A `Progress` object used to track the progress of the request.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method 'POST' and the provided parameters.
    - It passes the `path`, `headers`, `body`, `content_length`, and `content_type` to the [`send_with_content_provider`](#ClientImplsend_with_content_provider) function.
    - The `progress` parameter is also forwarded to track the request's progress.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, including success or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
This function sends a POST request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the endpoint path to which the POST request is sent.
    - `body`: A string containing the data to be sent in the body of the POST request.
    - `content_type`: A string indicating the MIME type of the content being sent.
- **Control Flow**:
    - The function calls another overloaded [`Post`](#ServerPost) method with the provided `path`, an empty `Headers` object, `body`, and `content_type`.
    - The [`Post`](#ServerPost) method that is invoked handles the actual sending of the request and processing of the response.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, including success or failure information.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
The [`Post`](#ServerPost) method sends a POST request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the endpoint path to which the POST request is sent.
    - `body`: A string containing the body of the POST request.
    - `content_type`: A string indicating the MIME type of the content being sent.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls another overloaded version of [`Post`](#ServerPost), passing the `path`, an empty `Headers` object, `body`, `content_type`, and `progress`.
    - The actual sending of the POST request is handled by the called [`Post`](#ServerPost) method, which is responsible for constructing the request and processing the response.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, which may include the response data or error information.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
Sends a POST request to a specified path with given headers and body content.
- **Inputs**:
    - `path`: The endpoint path to which the POST request is sent.
    - `headers`: A collection of HTTP headers to include in the request.
    - `body`: The content of the request body to be sent.
    - `content_type`: The MIME type of the content being sent in the body.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method 'POST' and the provided parameters.
    - It passes the `body.data()` and `body.size()` to specify the content of the request.
    - The function does not perform any additional processing or error handling; it relies on [`send_with_content_provider`](#ClientImplsend_with_content_provider) to handle those aspects.
- **Output**: Returns a `Result` object that contains the outcome of the POST request, which may include the response or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
Sends a POST request to a specified path with given headers and body content.
- **Inputs**:
    - `path`: A string representing the endpoint path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for the HTTP headers.
    - `body`: A string containing the body content to be sent in the POST request.
    - `content_type`: A string specifying the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method 'POST' and the provided parameters.
    - The [`send_with_content_provider`](#ClientImplsend_with_content_provider) function handles the actual sending of the request and manages the connection.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, including any response data or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
The [`Post`](#ServerPost) function sends a POST request to a specified path with given parameters.
- **Inputs**:
    - `path`: A `std::string` representing the endpoint path to which the POST request is sent.
    - `params`: A `Params` object containing the parameters to be included in the POST request.
- **Control Flow**:
    - The function calls another overload of [`Post`](#ServerPost) with the specified `path`, an empty `Headers` object, and the provided `params`.
    - This allows for a simplified interface for sending POST requests without needing to specify headers.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, which may include response data or error information.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
The [`Post`](#ServerPost) method in the `ClientImpl` class sends a POST request to a specified path with a given content length and content provider.
- **Inputs**:
    - `path`: A `std::string` representing the endpoint path to which the POST request is sent.
    - `content_length`: A `size_t` indicating the length of the content being sent in the request.
    - `content_provider`: A `ContentProvider` function or object that provides the content to be sent in the request body.
    - `content_type`: A `std::string` specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls another overloaded version of [`Post`](#ServerPost), passing the `path`, an empty `Headers` object, `content_length`, the moved `content_provider`, and `content_type`.
    - The actual sending of the POST request is handled by the called [`Post`](#ServerPost) method, which is responsible for constructing the request and managing the underlying network operations.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the POST request, which may include success or error information.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
The [`Post`](#ServerPost) method sends a POST request to a specified path using a content provider without a specified length.
- **Inputs**:
    - `path`: A `std::string` representing the endpoint path to which the POST request is sent.
    - `content_provider`: A `ContentProviderWithoutLength` function that provides the content to be sent in the POST request.
    - `content_type`: A `std::string` indicating the MIME type of the content being sent.
- **Control Flow**:
    - The method calls another overloaded [`Post`](#ServerPost) method with the provided `path`, an empty `Headers` object, the `content_provider` moved to avoid copying, and the `content_type`.
    - The `std::move` is used to efficiently transfer ownership of the `content_provider` to the called method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, including any response or error information.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
Sends a POST request to a specified path with given headers and content.
- **Inputs**:
    - `path`: The endpoint path to which the POST request is sent.
    - `headers`: A collection of HTTP headers to include in the request.
    - `content_length`: The length of the content being sent in the request.
    - `content_provider`: A callable that provides the content to be sent.
    - `content_type`: The MIME type of the content being sent.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method 'POST' and the provided parameters.
    - It passes the `path`, `headers`, `content_length`, and `content_provider` to the [`send_with_content_provider`](#ClientImplsend_with_content_provider) function.
    - The function does not perform any additional logic or error handling; it relies on the called function to handle those aspects.
- **Output**: Returns a `Result` object that contains the outcome of the POST request.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
Sends a POST request to a specified path with optional headers and a content provider.
- **Inputs**:
    - `path`: The endpoint path to which the POST request is sent.
    - `headers`: A collection of HTTP headers to include in the request.
    - `content_provider`: A callable that provides the content to be sent in the request body.
    - `content_type`: The MIME type of the content being sent.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method set to 'POST'.
    - It passes the provided `path`, `headers`, and `content_provider` along with a null body and content length of zero.
    - The `content_type` is also passed to specify the type of content being sent.
- **Output**: Returns a `Result` object that contains the outcome of the POST request.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
The [`Post`](#ServerPost) method sends a POST request to a specified path with given headers and parameters, converting the parameters to a query string.
- **Inputs**:
    - `path`: A `std::string` representing the endpoint path to which the POST request is sent.
    - `headers`: A `Headers` object containing key-value pairs representing the HTTP headers to be included in the request.
    - `params`: A `Params` object containing key-value pairs representing the parameters to be sent in the body of the POST request.
- **Control Flow**:
    - The method first calls `detail::params_to_query_str(params)` to convert the `params` into a query string format.
    - It then calls another overloaded [`Post`](#ServerPost) method with the specified `path`, `headers`, the generated `query`, and a content type of 'application/x-www-form-urlencoded'.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, including any response data or error information.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
Sends a POST request to a specified path with given headers and parameters.
- **Inputs**:
    - `path`: A string representing the endpoint path to which the POST request is sent.
    - `headers`: An object representing the headers to be included in the POST request.
    - `params`: An object containing parameters to be sent in the body of the POST request.
    - `progress`: A callback function to report progress of the request.
- **Control Flow**:
    - Converts the `params` object into a query string format using the `detail::params_to_query_str` function.
    - Calls another overloaded [`Post`](#ServerPost) method with the constructed query string, along with the provided `path`, `headers`, and a content type of 'application/x-www-form-urlencoded'.
- **Output**: Returns a `Result` object that contains the outcome of the POST request.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
Sends a POST request to the specified path with multipart form data.
- **Inputs**:
    - `path`: A string representing the endpoint path to which the POST request is sent.
    - `items`: A `MultipartFormDataItems` object containing the data to be sent in the POST request.
- **Control Flow**:
    - The function calls another overloaded [`Post`](#ServerPost) method, passing the `path`, an empty `Headers` object, and the `items`.
    - The actual sending of the POST request and handling of the response is managed by the called [`Post`](#ServerPost) method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, including success or error information.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
The [`Post`](#ServerPost) method sends a multipart form data POST request to a specified path with given headers and items.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for the HTTP headers.
    - `items`: An object of type `MultipartFormDataItems` representing the form data items to be included in the request body.
- **Control Flow**:
    - The method begins by generating a multipart data boundary using `detail::make_multipart_data_boundary()`.
    - It then serializes the content type for the multipart form data using `detail::serialize_multipart_formdata_get_content_type(boundary)`.
    - Next, it serializes the actual body of the multipart form data with `detail::serialize_multipart_formdata(items, boundary)`.
    - Finally, it calls another overloaded [`Post`](#ServerPost) method with the path, headers, serialized body, and content type to execute the request.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the POST request, which may include the response status, headers, and body.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
The [`Post`](#ServerPost) method sends a multipart form data POST request to a specified path with given headers and items.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing the headers to be included in the request.
    - `items`: A collection of multipart form data items to be sent in the request.
    - `boundary`: A string representing the boundary used to separate different parts of the multipart form data.
- **Control Flow**:
    - The function first checks if the provided `boundary` string contains valid multipart boundary characters using `detail::is_multipart_boundary_chars_valid`.
    - If the boundary is invalid, it returns a `Result` object indicating an error with the type `Error::UnsupportedMultipartBoundaryChars`.
    - If the boundary is valid, it serializes the multipart form data into a body string using `detail::serialize_multipart_formdata`.
    - It also retrieves the content type for the multipart data using `detail::serialize_multipart_formdata_get_content_type`.
    - Finally, it calls another overloaded [`Post`](#ServerPost) method with the path, headers, serialized body, and content type to execute the actual POST request.
- **Output**: The function returns a `Result` object that contains the outcome of the POST request, which may include a response or an error.
- **Functions called**:
    - [`Server::Post`](#ServerPost)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Post<!-- {{#callable:ClientImpl::Post}} -->
Sends a POST request with multipart form data to a specified path.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An object containing HTTP headers to be included in the request.
    - `items`: A collection of multipart form data items to be sent in the request.
    - `provider_items`: A collection of multipart form data provider items for dynamic content generation.
- **Control Flow**:
    - Generates a multipart data boundary using `detail::make_multipart_data_boundary()`.
    - Serializes the content type for the multipart form data using `detail::serialize_multipart_formdata_get_content_type()`.
    - Calls `send_with_content_provider()` with the HTTP method 'POST', the specified path, headers, and the generated content provider.
- **Output**: Returns a `Result` object representing the outcome of the POST request.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
    - [`ClientImpl::get_multipart_content_provider`](#ClientImplget_multipart_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
This function initiates a `PUT` request to a specified path with default empty body and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the `PUT` request is sent.
- **Control Flow**:
    - The function calls another overloaded [`Put`](#ServerPut) method with the specified `path` and two empty strings as arguments.
    - The empty strings represent the body and content type for the request, indicating that no data is being sent with this `PUT` request.
- **Output**: The function returns a `Result` object that encapsulates the outcome of the `PUT` request.
- **Functions called**:
    - [`Server::Put`](#ServerPut)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
The [`Put`](#ServerPut) method in `ClientImpl` sends a PUT request to a specified path with a given body, content length, and content type.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PUT request is sent.
    - `body`: A pointer to a `char` array containing the body of the request.
    - `content_length`: A `size_t` indicating the length of the body content.
    - `content_type`: A `std::string` specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls another overloaded [`Put`](#ServerPut) method with the same `path`, an empty `Headers` object, and the provided `body`, `content_length`, and `content_type`.
    - The actual sending of the request and handling of the response is delegated to the overloaded [`Put`](#ServerPut) method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PUT request, which may include success or error information.
- **Functions called**:
    - [`Server::Put`](#ServerPut)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
Sends a PUT request to a specified path with given headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for the HTTP headers.
    - `body`: A pointer to a character array representing the body content of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the body content.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method 'PUT' and the provided parameters.
    - The [`send_with_content_provider`](#ClientImplsend_with_content_provider) function handles the actual sending of the request and returns a `Result` object.
- **Output**: Returns a `Result` object that contains the outcome of the PUT request, including any response data or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
Sends a PUT request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the endpoint path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `body`: A pointer to a character array representing the body content to be sent with the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the body content.
    - `progress`: A `Progress` object used to track the progress of the request.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method 'PUT' and the provided parameters.
    - The [`send_with_content_provider`](#ClientImplsend_with_content_provider) function handles the actual sending of the request and returns the result.
- **Output**: Returns a `Result` object that contains the outcome of the PUT request, including any response data or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
The [`Put`](#ServerPut) method sends a PUT request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PUT request is sent.
    - `body`: A `std::string` containing the data to be sent in the body of the PUT request.
    - `content_type`: A `std::string` specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls another overloaded [`Put`](#ServerPut) method with the same `path`, an empty `Headers` object, `body`, and `content_type`.
    - The actual sending of the request and handling of the response is managed by the called [`Put`](#ServerPut) method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PUT request, including success or error information.
- **Functions called**:
    - [`Server::Put`](#ServerPut)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
The [`Put`](#ServerPut) method in `ClientImpl` sends a PUT request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `body`: A string containing the data to be sent in the body of the PUT request.
    - `content_type`: A string indicating the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls another overloaded [`Put`](#ServerPut) method with the same `path`, an empty `Headers` object, and the provided `body`, `content_type`, and `progress`.
    - This allows for a more complex implementation of the [`Put`](#ServerPut) method to handle the actual sending of the request.
- **Output**: Returns a `Result` object that contains the outcome of the PUT request, which may include success or error information.
- **Functions called**:
    - [`Server::Put`](#ServerPut)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
Sends a PUT request to a specified path with given headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for the HTTP headers.
    - `body`: A string containing the body content to be sent with the PUT request.
    - `content_type`: A string specifying the MIME type of the body content.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method 'PUT' and the provided parameters.
    - The `body.data()` and `body.size()` are used to pass the body content and its length to the [`send_with_content_provider`](#ClientImplsend_with_content_provider) function.
    - The function does not contain any conditional logic or loops; it directly returns the result of the [`send_with_content_provider`](#ClientImplsend_with_content_provider) call.
- **Output**: Returns a `Result` object that contains the outcome of the PUT request, which may include response data or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
Sends a PUT request to a specified path with given headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for the HTTP headers.
    - `body`: A string containing the body content to be sent with the PUT request.
    - `content_type`: A string specifying the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method 'PUT' and the provided parameters.
    - The [`send_with_content_provider`](#ClientImplsend_with_content_provider) function handles the actual sending of the request and returns the result.
- **Output**: Returns a `Result` object that contains the outcome of the PUT request, including any response data or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
The [`Put`](#ServerPut) method in `ClientImpl` sends a PUT request to a specified path with a given content length and content provider.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `content_length`: A size_t value indicating the length of the content being sent.
    - `content_provider`: A `ContentProvider` function or object that provides the content to be sent in the request.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls another overloaded [`Put`](#ServerPut) method with the same `path`, an empty `Headers` object, the `content_length`, the `content_provider` moved to avoid copying, and the `content_type`.
    - This allows for a more flexible implementation of the PUT request while keeping the interface clean.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the PUT request, which may include success or error information.
- **Functions called**:
    - [`Server::Put`](#ServerPut)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
This function sends a PUT request to a specified path using a content provider without length.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `content_provider`: A `ContentProviderWithoutLength` function that provides the content to be sent in the PUT request.
    - `content_type`: A string indicating the MIME type of the content being sent.
- **Control Flow**:
    - The function calls another overloaded [`Put`](#ServerPut) method with the provided `path`, an empty `Headers` object, the `content_provider`, and the `content_type`.
    - The `std::move` is used to transfer ownership of the `content_provider` to the called [`Put`](#ServerPut) method, ensuring efficient resource management.
- **Output**: Returns a `Result` object that indicates the outcome of the PUT request.
- **Functions called**:
    - [`Server::Put`](#ServerPut)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
Sends a PUT request to a specified path with provided headers and content.
- **Inputs**:
    - `path`: The URL path to which the PUT request is sent.
    - `headers`: A collection of HTTP headers to include in the request.
    - `content_length`: The length of the content being sent in the request.
    - `content_provider`: A callable that provides the content to be sent.
    - `content_type`: The MIME type of the content being sent.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method 'PUT' and the provided parameters.
    - The [`send_with_content_provider`](#ClientImplsend_with_content_provider) function handles the actual sending of the request and manages the content transfer.
- **Output**: Returns a `Result` object that contains the outcome of the PUT request.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
Sends a PUT request to a specified path with provided headers and content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for the HTTP headers.
    - `content_provider`: A `ContentProviderWithoutLength` function that provides the content to be sent in the request.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method set to 'PUT' and passes the provided arguments.
    - The [`send_with_content_provider`](#ClientImplsend_with_content_provider) function handles the actual sending of the request, including managing the connection and processing the response.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PUT request, including any response data or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
This function invokes another [`Put`](#ServerPut) method with default headers to send data to a specified path.
- **Inputs**:
    - `path`: A string representing the endpoint path where the data will be sent.
    - `params`: An object of type `Params` containing the parameters to be sent with the request.
- **Control Flow**:
    - The function calls another overloaded version of [`Put`](#ServerPut) with the provided `path`, default headers, and `params`.
    - It does not perform any additional logic or error handling within its body.
- **Output**: Returns a `Result` object that represents the outcome of the [`Put`](#ServerPut) operation.
- **Functions called**:
    - [`Server::Put`](#ServerPut)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
Sends a PUT request to a specified path with given headers and parameters.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for the HTTP headers.
    - `params`: An object of type `Params` containing parameters to be included in the request.
- **Control Flow**:
    - Converts the `params` object into a query string using the `detail::params_to_query_str` function.
    - Calls the overloaded [`Put`](#ServerPut) method with the specified `path`, `headers`, the generated query string, and a default content type of 'application/x-www-form-urlencoded'.
- **Output**: Returns a `Result` object representing the outcome of the PUT request.
- **Functions called**:
    - [`Server::Put`](#ServerPut)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
The [`Put`](#ServerPut) method sends a PUT request to a specified path with given headers and parameters.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing the headers to be included in the request.
    - `params`: An object of type `Params` containing the parameters to be sent with the request.
    - `progress`: A `Progress` callback function to track the progress of the request.
- **Control Flow**:
    - The method first converts the `params` object into a query string using the `detail::params_to_query_str` function.
    - It then calls another overloaded [`Put`](#ServerPut) method, passing the `path`, `headers`, the generated query string, a content type of 'application/x-www-form-urlencoded', and the `progress` callback.
- **Output**: The method returns a `Result` object that represents the outcome of the PUT request.
- **Functions called**:
    - [`Server::Put`](#ServerPut)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
The [`Put`](#ServerPut) method sends a PUT request to a specified path with multipart form data.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PUT request is sent.
    - `items`: A `MultipartFormDataItems` object containing the data to be sent in the PUT request.
- **Control Flow**:
    - The method calls another overload of [`Put`](#ServerPut) with the provided `path`, an empty `Headers` object, and the `items`.
    - The actual sending of the request and handling of the response is managed by the called [`Put`](#ServerPut) method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PUT request, including any response data or error information.
- **Functions called**:
    - [`Server::Put`](#ServerPut)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
Sends a PUT request with multipart form data to a specified path.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing the headers to be included in the request.
    - `items`: An object of type `MultipartFormDataItems` representing the multipart form data to be sent in the request.
- **Control Flow**:
    - Generates a multipart data boundary using `detail::make_multipart_data_boundary()`.
    - Serializes the content type for the multipart form data using `detail::serialize_multipart_formdata_get_content_type(boundary)`.
    - Serializes the multipart form data into a body string using `detail::serialize_multipart_formdata(items, boundary)`.
    - Calls the overloaded [`Put`](#ServerPut) method with the path, headers, serialized body, and content type.
- **Output**: Returns a `Result` object representing the outcome of the PUT request.
- **Functions called**:
    - [`Server::Put`](#ServerPut)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
The [`Put`](#ServerPut) method sends a multipart form data request to a specified path with given headers.
- **Inputs**:
    - `path`: A string representing the URL path to which the request is sent.
    - `headers`: An object of type `Headers` containing the headers to be included in the request.
    - `items`: A collection of multipart form data items to be sent in the request.
    - `boundary`: A string representing the boundary used to separate parts in the multipart form data.
- **Control Flow**:
    - The function first checks if the provided `boundary` string contains valid characters using `detail::is_multipart_boundary_chars_valid`.
    - If the boundary is invalid, it returns a `Result` indicating an error with unsupported boundary characters.
    - If the boundary is valid, it serializes the multipart form data into a body string using `detail::serialize_multipart_formdata`.
    - It also retrieves the content type for the multipart data using `detail::serialize_multipart_formdata_get_content_type`.
    - Finally, it calls another overloaded [`Put`](#ServerPut) method with the path, headers, serialized body, and content type to send the request.
- **Output**: The output is a `Result` object that indicates the success or failure of the PUT request, containing either the response data or an error.
- **Functions called**:
    - [`Server::Put`](#ServerPut)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Put<!-- {{#callable:ClientImpl::Put}} -->
The `Put` method sends a multipart/form-data HTTP PUT request to a specified path with given headers and data items.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for HTTP headers to be included in the request.
    - `items`: A collection of `MultipartFormDataItems` representing the form data to be sent in the request.
    - `provider_items`: A collection of `MultipartFormDataProviderItems` that provide additional data for the multipart form.
- **Control Flow**:
    - The method begins by generating a multipart data boundary using `detail::make_multipart_data_boundary()`.
    - It then serializes the content type for the multipart form data using `detail::serialize_multipart_formdata_get_content_type()` with the generated boundary.
    - Finally, it calls `send_with_content_provider()` with the HTTP method 'PUT', the specified path, headers, and the content provider generated by `get_multipart_content_provider()`.
- **Output**: The method returns a `Result` object that represents the outcome of the HTTP request, which may include success or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
    - [`ClientImpl::get_multipart_content_provider`](#ClientImplget_multipart_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
This function calls another [`Patch`](#ServerPatch) method with default empty string parameters.
- **Inputs**:
    - `path`: A string representing the path to which the patch request is to be sent.
- **Control Flow**:
    - The function [`Patch`](#ServerPatch) is invoked with the provided `path` and two additional empty string parameters.
    - The result of the inner [`Patch`](#ServerPatch) method call is returned directly.
- **Output**: The output is a `Result` object that represents the outcome of the patch operation.
- **Functions called**:
    - [`Server::Patch`](#ServerPatch)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
The [`Patch`](#ServerPatch) method in `ClientImpl` sends a PATCH request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the PATCH request is sent.
    - `body`: A pointer to a character array containing the body of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls another overloaded [`Patch`](#ServerPatch) function with the same `path`, an empty `Headers` object, and the provided `body`, `content_length`, and `content_type`.
    - The actual sending of the PATCH request is handled by the called [`Patch`](#ServerPatch) function, which is expected to manage the request and return a `Result`.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the PATCH request, which may include success or error information.
- **Functions called**:
    - [`Server::Patch`](#ServerPatch)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
The [`Patch`](#ServerPatch) method in `ClientImpl` sends a PATCH request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the PATCH request is sent.
    - `body`: A pointer to a character array containing the body of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the content being sent.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls another overloaded [`Patch`](#ServerPatch) method with the provided parameters, along with an empty `Headers` object.
    - The call to the overloaded [`Patch`](#ServerPatch) method handles the actual sending of the PATCH request.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the PATCH request.
- **Functions called**:
    - [`Server::Patch`](#ServerPatch)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
The [`Patch`](#ServerPatch) method in the `ClientImpl` class initiates a PATCH request to a specified path with given headers and body content.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `headers`: A `Headers` object containing key-value pairs for HTTP headers to be included in the request.
    - `body`: A pointer to a `char` array representing the body content of the PATCH request.
    - `content_length`: A `size_t` indicating the length of the body content.
    - `content_type`: A `std::string` specifying the MIME type of the body content.
- **Control Flow**:
    - The method calls another overloaded [`Patch`](#ServerPatch) method with the same parameters plus an additional `nullptr` argument, which likely represents an optional progress callback or similar functionality.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PATCH request, which may include success or error information.
- **Functions called**:
    - [`Server::Patch`](#ServerPatch)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
Sends a PATCH request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PATCH request is sent.
    - `headers`: An object representing the headers to be included in the request.
    - `body`: A pointer to a character array containing the body content of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string representing the MIME type of the body content.
    - `progress`: A Progress object used to track the progress of the request.
- **Control Flow**:
    - The function calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method 'PATCH' and the provided parameters.
    - The [`send_with_content_provider`](#ClientImplsend_with_content_provider) function handles the actual sending of the request and returns the result.
- **Output**: Returns a `Result` object that contains the outcome of the PATCH request.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
The [`Patch`](#ServerPatch) method sends a PATCH request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `body`: A `std::string` containing the data to be sent in the body of the PATCH request.
    - `content_type`: A `std::string` specifying the MIME type of the body content.
- **Control Flow**:
    - The method calls another overloaded [`Patch`](#ServerPatch) function with the provided `path`, an empty `Headers` object, `body`, and `content_type`.
    - The actual sending of the PATCH request and handling of the response is managed by the called [`Patch`](#ServerPatch) function.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PATCH request, which may include success or error information.
- **Functions called**:
    - [`Server::Patch`](#ServerPatch)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
The [`Patch`](#ServerPatch) method in the `ClientImpl` class sends a PATCH request to a specified path with a given body and content type, while also allowing for progress tracking.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `body`: A `std::string` containing the data to be sent in the body of the PATCH request.
    - `content_type`: A `std::string` specifying the MIME type of the content being sent in the request body.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls another overloaded [`Patch`](#ServerPatch) method, passing the `path`, an empty `Headers` object, `body`, `content_type`, and `progress` as arguments.
    - The actual sending of the PATCH request and handling of the response is delegated to the overloaded [`Patch`](#ServerPatch) method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PATCH request, which may include the response data or error information.
- **Functions called**:
    - [`Server::Patch`](#ServerPatch)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
The [`Patch`](#ServerPatch) method in `ClientImpl` initiates a PATCH request to a specified path with given headers and body.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `headers`: A `Headers` object containing key-value pairs for the HTTP headers to be included in the request.
    - `body`: A `std::string` containing the body of the PATCH request.
    - `content_type`: A `std::string` specifying the MIME type of the body content.
- **Control Flow**:
    - The method calls another overloaded [`Patch`](#ServerPatch) function with the same parameters, passing `nullptr` for the last argument.
    - This indicates that the method is likely designed to provide a simplified interface for making PATCH requests without needing to specify additional parameters.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the PATCH request, which may include success or error information.
- **Functions called**:
    - [`Server::Patch`](#ServerPatch)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
The `Patch` method sends a PATCH request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PATCH request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `body`: A string containing the body content to be sent with the PATCH request.
    - `content_type`: A string specifying the content type of the body being sent.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method 'PATCH' and the provided parameters.
    - The [`send_with_content_provider`](#ClientImplsend_with_content_provider) function handles the actual sending of the request and returns the result.
- **Output**: Returns a `Result` object that contains the outcome of the PATCH request, including any response data or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
The [`Patch`](#ServerPatch) method in the `ClientImpl` class sends a PATCH request to a specified path with a given content length and content provider.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `content_length`: A `size_t` indicating the length of the content being sent in the request.
    - `content_provider`: A `ContentProvider` function or object that provides the content to be sent in the PATCH request.
    - `content_type`: A `std::string` specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls another overload of [`Patch`](#ServerPatch), passing the `path`, an empty `Headers` object, `content_length`, the moved `content_provider`, and `content_type`.
    - The `std::move` is used to efficiently transfer ownership of the `content_provider` to the called function.
- **Output**: The method returns a `Result` object that represents the outcome of the PATCH request, which may include success or error information.
- **Functions called**:
    - [`Server::Patch`](#ServerPatch)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
The [`Patch`](#ServerPatch) method in the `ClientImpl` class sends a PATCH request to a specified path using a content provider and content type.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `content_provider`: A `ContentProviderWithoutLength` function that provides the content to be sent in the PATCH request.
    - `content_type`: A `std::string` indicating the MIME type of the content being sent.
- **Control Flow**:
    - The method calls another overloaded [`Patch`](#ServerPatch) method with the provided `path`, an empty `Headers` object, the `content_provider` moved to avoid copying, and the `content_type`.
    - The actual sending of the PATCH request and handling of the response is managed by the called [`Patch`](#ServerPatch) method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PATCH request, which may include success or error information.
- **Functions called**:
    - [`Server::Patch`](#ServerPatch)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
The `Patch` method sends a PATCH request to a specified path with optional headers and content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PATCH request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `content_length`: A size_t value indicating the length of the content being sent.
    - `content_provider`: A `ContentProvider` function that provides the content to be sent in the request.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method set to 'PATCH' and passes all the input parameters.
    - The [`send_with_content_provider`](#ClientImplsend_with_content_provider) function handles the actual sending of the request and returns the result.
- **Output**: Returns a `Result` object that contains the outcome of the PATCH request, including any response data or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Patch<!-- {{#callable:ClientImpl::Patch}} -->
The `Patch` method sends a PATCH request to a specified path with optional headers and content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PATCH request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `content_provider`: A callable that provides the content to be sent in the request body, without specifying its length.
    - `content_type`: A string indicating the MIME type of the content being sent.
- **Control Flow**:
    - The method calls [`send_with_content_provider`](#ClientImplsend_with_content_provider) with the HTTP method set to 'PATCH' and passes along the provided parameters.
    - The [`send_with_content_provider`](#ClientImplsend_with_content_provider) function handles the actual sending of the request and manages the content provider.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PATCH request, including any response data or error information.
- **Functions called**:
    - [`ClientImpl::send_with_content_provider`](#ClientImplsend_with_content_provider)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Delete<!-- {{#callable:ClientImpl::Delete}} -->
The [`Delete`](#ServerDelete) function in the `ClientImpl` class initiates a DELETE request to a specified path without additional headers or body content.
- **Inputs**:
    - `path`: A `std::string` representing the resource path to which the DELETE request will be sent.
- **Control Flow**:
    - The function calls another overloaded version of [`Delete`](#ServerDelete), passing the `path` along with default values for headers and body.
    - The default values for headers and body are initialized as empty instances of `Headers` and `std::string`, respectively.
- **Output**: The function returns a `Result` object that encapsulates the outcome of the DELETE request, which may include success or error information.
- **Functions called**:
    - [`Server::Delete`](#ServerDelete)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Delete<!-- {{#callable:ClientImpl::Delete}} -->
This function initiates a [`Delete`](#ServerDelete) request to a specified path with optional headers.
- **Inputs**:
    - `path`: A `std::string` representing the resource path to be deleted.
    - `headers`: A `Headers` object containing any additional headers to be sent with the request.
- **Control Flow**:
    - The function calls another overloaded version of [`Delete`](#ServerDelete), passing the `path`, `headers`, and two empty `std::string` arguments.
    - The empty strings likely represent optional body content or content type that are not needed for this specific overload.
- **Output**: Returns a `Result` object that encapsulates the outcome of the delete operation, which may include success or error information.
- **Functions called**:
    - [`Server::Delete`](#ServerDelete)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Delete<!-- {{#callable:ClientImpl::Delete}} -->
The [`Delete`](#ServerDelete) function in `ClientImpl` initiates a DELETE request to a specified path with optional body content and content type.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the DELETE request is sent.
    - `body`: A pointer to a `char` array representing the body content to be sent with the DELETE request.
    - `content_length`: A `size_t` indicating the length of the body content.
    - `content_type`: A `std::string` specifying the MIME type of the body content.
- **Control Flow**:
    - The function calls another overloaded version of [`Delete`](#ServerDelete), passing the `path`, an empty `Headers` object, and the other parameters.
    - This allows for a consistent interface while potentially handling additional headers in the other overload.
- **Output**: Returns a `Result` object that encapsulates the outcome of the DELETE request, which may include success or error information.
- **Functions called**:
    - [`Server::Delete`](#ServerDelete)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Delete<!-- {{#callable:ClientImpl::Delete}} -->
This function initiates a [`Delete`](#ServerDelete) request to a specified path with optional body content and headers.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the [`Delete`](#ServerDelete) request is sent.
    - `body`: A pointer to a `char` array representing the body content to be sent with the request, which can be null.
    - `content_length`: A `size_t` indicating the length of the body content.
    - `content_type`: A `std::string` specifying the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The function calls another overloaded version of [`Delete`](#ServerDelete), passing the `path`, an empty `Headers` object, and the other parameters.
    - This allows for a consistent handling of the [`Delete`](#ServerDelete) request while providing flexibility in specifying headers.
- **Output**: Returns a `Result` object that encapsulates the outcome of the [`Delete`](#ServerDelete) request, which may include success or error information.
- **Functions called**:
    - [`Server::Delete`](#ServerDelete)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Delete<!-- {{#callable:ClientImpl::Delete}} -->
The [`Delete`](#ServerDelete) function in the `ClientImpl` class initiates a DELETE HTTP request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the DELETE request will be sent.
    - `headers`: A `Headers` object containing any additional HTTP headers to include in the request.
    - `body`: A pointer to a `char` array representing the body content to be sent with the DELETE request.
    - `content_length`: A `size_t` value indicating the length of the body content.
    - `content_type`: A `std::string` specifying the MIME type of the body content.
- **Control Flow**:
    - The function calls another overloaded version of [`Delete`](#ServerDelete), passing all its parameters along with a `nullptr` for an additional argument.
    - This design allows for a more flexible implementation of the DELETE request, potentially handling different scenarios based on the presence of the additional argument.
- **Output**: The function returns a `Result` object that encapsulates the outcome of the DELETE request, which may include success or error information.
- **Functions called**:
    - [`Server::Delete`](#ServerDelete)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Delete<!-- {{#callable:ClientImpl::Delete}} -->
The `Delete` method sends an HTTP DELETE request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the DELETE request is sent.
    - `headers`: An object containing HTTP headers to be included in the request.
    - `body`: A pointer to a character array representing the body content of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the body content.
    - `progress`: A callback function to report progress of the request.
- **Control Flow**:
    - A `Request` object is created and initialized with the DELETE method, headers, path, and progress callback.
    - If a maximum timeout is set, the current time is recorded as the start time of the request.
    - If the `content_type` is not empty, it is set as a header in the request.
    - The body of the request is assigned from the provided character array and its length.
    - The request is sent using the [`send_`](#ClientImplsend_) method, which handles the actual transmission of the request.
- **Output**: The function returns a `Result` object that encapsulates the outcome of the DELETE request, including any response data or error information.
- **Functions called**:
    - [`ClientImpl::send_`](#ClientImplsend_)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Delete<!-- {{#callable:ClientImpl::Delete}} -->
This function sends a DELETE request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the DELETE request is sent.
    - `body`: A string containing the body of the DELETE request.
    - `content_type`: A string indicating the content type of the body being sent.
- **Control Flow**:
    - The function calls another overloaded [`Delete`](#ServerDelete) method with the provided `path`, an empty `Headers` object, the data from `body`, its size, and the `content_type`.
    - The [`Delete`](#ServerDelete) method is expected to handle the actual HTTP DELETE request processing.
- **Output**: Returns a `Result` object that encapsulates the outcome of the DELETE request, which may include success or error information.
- **Functions called**:
    - [`Server::Delete`](#ServerDelete)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Delete<!-- {{#callable:ClientImpl::Delete}} -->
The [`Delete`](#ServerDelete) method in the `ClientImpl` class initiates a DELETE request to a specified path with optional body content and content type.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the DELETE request is sent.
    - `body`: A `std::string` containing the body content to be sent with the DELETE request.
    - `content_type`: A `std::string` specifying the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls another overloaded [`Delete`](#ServerDelete) function, passing the `path`, an empty `Headers` object, the data pointer of the `body`, the size of the `body`, the `content_type`, and the `progress` object.
    - The actual logic for handling the DELETE request is encapsulated in the called [`Delete`](#ServerDelete) function, which is responsible for processing the request and returning the result.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the DELETE request, which may include success or error information.
- **Functions called**:
    - [`Server::Delete`](#ServerDelete)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Delete<!-- {{#callable:ClientImpl::Delete}} -->
This function invokes a more detailed [`Delete`](#ServerDelete) method to perform an HTTP DELETE request with a specified path, headers, body, and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the DELETE request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to be included in the request.
    - `body`: A string containing the body of the request, which may be empty for a DELETE operation.
    - `content_type`: A string indicating the MIME type of the body content.
- **Control Flow**:
    - The function calls another overloaded [`Delete`](#ServerDelete) method, passing the `path`, `headers`, a pointer to the body data, the size of the body, and the `content_type`.
    - The parameters are prepared to match the expected types of the more detailed [`Delete`](#ServerDelete) method.
- **Output**: The function returns a `Result` object that encapsulates the outcome of the DELETE request, which may include success or error information.
- **Functions called**:
    - [`Server::Delete`](#ServerDelete)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Delete<!-- {{#callable:ClientImpl::Delete}} -->
The [`Delete`](#ServerDelete) method in `ClientImpl` initiates a DELETE request to a specified path with optional headers and body.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the DELETE request is sent.
    - `headers`: A `Headers` object containing any additional headers to include in the request.
    - `body`: A `std::string` representing the body content to be sent with the DELETE request.
    - `content_type`: A `std::string` indicating the MIME type of the body content.
    - `progress`: A `Progress` object used to track the progress of the request.
- **Control Flow**:
    - The method calls another overloaded [`Delete`](#ServerDelete) function, passing the `path`, `headers`, and the body data along with its size and content type.
    - The body is accessed using `body.data()` to get a pointer to the character data and `body.size()` to get the length of the body.
- **Output**: Returns a `Result` object that encapsulates the outcome of the DELETE request, which may include success or error information.
- **Functions called**:
    - [`Server::Delete`](#ServerDelete)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Options<!-- {{#callable:ClientImpl::Options}} -->
This function calls another [`Options`](#ServerOptions) method with a specified path and default headers.
- **Inputs**:
    - `path`: A constant reference to a `std::string` representing the URL path for which the HTTP OPTIONS request is made.
- **Control Flow**:
    - The function directly invokes the overloaded [`Options`](#ServerOptions) method of the `ClientImpl` class, passing the provided `path` and a default constructed `Headers` object.
    - No additional logic or error handling is performed within this function.
- **Output**: The function returns the result of the [`Options`](#ServerOptions) method call, which is of type `Result`, representing the outcome of the HTTP OPTIONS request.
- **Functions called**:
    - [`Server::Options`](#ServerOptions)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::Options<!-- {{#callable:ClientImpl::Options}} -->
Sends an HTTP OPTIONS request to a specified path with optional headers.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the OPTIONS request is sent.
    - `headers`: A `Headers` object containing any additional headers to include in the request.
- **Control Flow**:
    - Creates a `Request` object and sets its method to 'OPTIONS'.
    - Assigns the provided `path` and `headers` to the request object.
    - If `max_timeout_msec_` is greater than 0, records the current time as the start time of the request.
    - Calls the [`send_`](#ClientImplsend_) method with the constructed request to send the OPTIONS request.
- **Output**: Returns a `Result` object that contains the response from the server after sending the OPTIONS request.
- **Functions called**:
    - [`ClientImpl::send_`](#ClientImplsend_)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::stop<!-- {{#callable:ClientImpl::stop}} -->
Stops the client by shutting down the socket and SSL connections.
- **Inputs**: None
- **Control Flow**:
    - Acquires a lock on `socket_mutex_` to ensure thread safety.
    - Checks if there are ongoing socket requests by evaluating `socket_requests_in_flight_`.
    - If there are ongoing requests, it calls [`shutdown_socket`](#shutdown_socket) to stop further read/write operations on the socket.
    - Sets the flag `socket_should_be_closed_when_request_is_done_` to true to indicate that the socket should be closed after current requests are completed.
    - If there are no ongoing requests, it proceeds to shut down SSL and the socket, and finally closes the socket.
- **Output**: The function does not return a value; it performs operations to safely shut down the socket and SSL connections.
- **Functions called**:
    - [`shutdown_socket`](#shutdown_socket)
    - [`ClientImpl::shutdown_ssl`](#ClientImplshutdown_ssl)
    - [`close_socket`](#close_socket)
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::host<!-- {{#callable:ClientImpl::host}} -->
Returns the host string associated with the `ClientImpl` instance.
- **Inputs**: None
- **Control Flow**:
    - The function is marked as `inline`, indicating that it is a simple function that may be expanded in place to improve performance.
    - It accesses the private member variable `host_` of the `ClientImpl` class, which stores the host string.
    - The function returns the value of `host_` as a `std::string`.
- **Output**: The output is a `std::string` representing the host associated with the `ClientImpl` instance.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::port<!-- {{#callable:ClientImpl::port}} -->
Returns the port number associated with the `ClientImpl` instance.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the private member variable `port_` of the `ClientImpl` class.
    - It returns the value of `port_` without any additional logic or conditions.
- **Output**: An integer representing the port number that the `ClientImpl` instance is configured to use.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::is\_socket\_open<!-- {{#callable:ClientImpl::is_socket_open}} -->
Checks if the socket is currently open.
- **Inputs**: None
- **Control Flow**:
    - Acquires a lock on `socket_mutex_` to ensure thread safety.
    - Calls the `is_open()` method on the `socket_` member to determine if the socket is open.
- **Output**: Returns a boolean value indicating whether the socket is open (true) or not (false).
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::socket<!-- {{#callable:ClientImpl::socket}} -->
Returns the socket associated with the `ClientImpl` instance.
- **Inputs**:
    - `this`: A constant reference to the `ClientImpl` instance, which provides access to the member variable `socket_`.
- **Control Flow**:
    - The function directly accesses the `sock` member of the `socket_` structure, which is a member of the `ClientImpl` class.
    - No conditional logic or loops are present; the function simply returns the value.
- **Output**: Returns a `socket_t` type representing the socket associated with the `ClientImpl` instance.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_connection\_timeout<!-- {{#callable:ClientImpl::set_connection_timeout}} -->
Sets the connection timeout for the `ClientImpl` instance.
- **Inputs**:
    - `sec`: The number of seconds to set for the connection timeout.
    - `usec`: The number of microseconds to set for the connection timeout.
- **Control Flow**:
    - The function assigns the value of `sec` to the member variable `connection_timeout_sec_`.
    - The function assigns the value of `usec` to the member variable `connection_timeout_usec_`.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by setting the connection timeout.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_read\_timeout<!-- {{#callable:ClientImpl::set_read_timeout}} -->
Sets the read timeout for the `ClientImpl` instance.
- **Inputs**:
    - `sec`: The number of seconds for the read timeout.
    - `usec`: The number of microseconds for the read timeout.
- **Control Flow**:
    - The function directly assigns the provided `sec` value to the member variable `read_timeout_sec_`.
    - The function directly assigns the provided `usec` value to the member variable `read_timeout_usec_`.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by setting the read timeout.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_write\_timeout<!-- {{#callable:ClientImpl::set_write_timeout}} -->
Sets the write timeout for the `ClientImpl` instance.
- **Inputs**:
    - `sec`: The number of seconds for the write timeout.
    - `usec`: The number of microseconds for the write timeout.
- **Control Flow**:
    - Assigns the value of `sec` to the member variable `write_timeout_sec_`.
    - Assigns the value of `usec` to the member variable `write_timeout_usec_`.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by setting the write timeout.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_max\_timeout<!-- {{#callable:ClientImpl::set_max_timeout}} -->
Sets the maximum timeout value for the `ClientImpl` instance.
- **Inputs**:
    - `msec`: A `time_t` value representing the maximum timeout in milliseconds.
- **Control Flow**:
    - The function directly assigns the input value `msec` to the member variable `max_timeout_msec_`.
    - No conditional statements or loops are present, making this a straightforward assignment operation.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by updating the `max_timeout_msec_` member variable.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_basic\_auth<!-- {{#callable:ClientImpl::set_basic_auth}} -->
Sets the basic authentication credentials for the client.
- **Inputs**:
    - `username`: A `std::string` representing the username for basic authentication.
    - `password`: A `std::string` representing the password for basic authentication.
- **Control Flow**:
    - Assigns the provided `username` to the member variable `basic_auth_username_`.
    - Assigns the provided `password` to the member variable `basic_auth_password_`.
- **Output**: This function does not return a value; it updates the internal state of the `ClientImpl` object with the provided authentication credentials.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_bearer\_token\_auth<!-- {{#callable:ClientImpl::set_bearer_token_auth}} -->
Sets the bearer token for authentication in the `ClientImpl` class.
- **Inputs**:
    - `token`: A constant reference to a string representing the bearer token used for authentication.
- **Control Flow**:
    - The function directly assigns the provided `token` to the member variable `bearer_token_auth_token_`.
    - There are no conditional statements or loops; the function executes a single assignment operation.
- **Output**: The function does not return a value; it modifies the internal state of the `ClientImpl` instance by updating the bearer token.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_digest\_auth<!-- {{#callable:ClientImpl::set_digest_auth}} -->
Sets the username and password for digest authentication.
- **Inputs**:
    - `username`: A string representing the username for digest authentication.
    - `password`: A string representing the password for digest authentication.
- **Control Flow**:
    - The function directly assigns the provided `username` to the member variable `digest_auth_username_`.
    - It also assigns the provided `password` to the member variable `digest_auth_password_`.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` class by storing the provided username and password.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_keep\_alive<!-- {{#callable:ClientImpl::set_keep_alive}} -->
Sets the keep-alive status for the client connection.
- **Inputs**:
    - `on`: A boolean value indicating whether to enable (true) or disable (false) keep-alive for the client connection.
- **Control Flow**:
    - The function directly assigns the input boolean value 'on' to the member variable 'keep_alive_'.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by updating the 'keep_alive_' member variable.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_follow\_location<!-- {{#callable:ClientImpl::set_follow_location}} -->
Sets the `follow_location_` flag to enable or disable following HTTP redirects.
- **Inputs**:
    - `on`: A boolean value indicating whether to enable (`true`) or disable (`false`) the following of HTTP redirects.
- **Control Flow**:
    - The function directly assigns the input boolean value `on` to the member variable `follow_location_`.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by updating the `follow_location_` member variable.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_url\_encode<!-- {{#callable:ClientImpl::set_url_encode}} -->
Sets the URL encoding state for the `ClientImpl` instance.
- **Inputs**:
    - `on`: A boolean value indicating whether URL encoding should be enabled (true) or disabled (false).
- **Control Flow**:
    - The function directly assigns the input boolean value to the member variable `url_encode_`.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_hostname\_addr\_map<!-- {{#callable:ClientImpl::set_hostname_addr_map}} -->
Sets the hostname to address mapping for the `ClientImpl` instance.
- **Inputs**:
    - `addr_map`: A `std::map` that associates hostname strings with their corresponding address strings.
- **Control Flow**:
    - The function takes a `std::map<std::string, std::string>` as an argument.
    - It uses `std::move` to transfer ownership of the input map to the member variable `addr_map_`.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by updating the `addr_map_` member variable.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_default\_headers<!-- {{#callable:ClientImpl::set_default_headers}} -->
Sets the default headers for the `ClientImpl` instance.
- **Inputs**:
    - `headers`: An instance of `Headers` that contains the default headers to be set.
- **Control Flow**:
    - The function uses `std::move` to transfer ownership of the `headers` argument to the member variable `default_headers_`.
    - No conditional logic or loops are present; the function executes a single assignment operation.
- **Output**: The function does not return a value; it modifies the internal state of the `ClientImpl` instance by updating its `default_headers_` member.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_header\_writer<!-- {{#callable:ClientImpl::set_header_writer}} -->
Sets a custom header writer function for the `ClientImpl` class.
- **Inputs**:
    - `writer`: A `std::function` that takes a `Stream` and `Headers` as parameters and returns a `ssize_t`, representing the custom function to write headers.
- **Control Flow**:
    - The function assigns the provided `writer` function to the member variable `header_writer_`.
    - No additional logic or control flow is present; the function executes a single assignment operation.
- **Output**: The function does not return a value; it modifies the internal state of the `ClientImpl` instance by setting the `header_writer_`.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_address\_family<!-- {{#callable:ClientImpl::set_address_family}} -->
Sets the address family for the `ClientImpl` instance.
- **Inputs**:
    - `family`: An integer representing the address family to be set, such as `AF_INET` for IPv4 or `AF_INET6` for IPv6.
- **Control Flow**:
    - The function directly assigns the input `family` to the member variable `address_family_`.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by updating the `address_family_` member variable.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_tcp\_nodelay<!-- {{#callable:ClientImpl::set_tcp_nodelay}} -->
Sets the TCP_NODELAY option for the client socket.
- **Inputs**:
    - `on`: A boolean value indicating whether to enable (true) or disable (false) the TCP_NODELAY option.
- **Control Flow**:
    - The function directly assigns the input boolean value to the member variable `tcp_nodelay_`.
    - There are no conditional statements or loops; the function executes a single assignment operation.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` class by updating the `tcp_nodelay_` member variable.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_ipv6\_v6only<!-- {{#callable:ClientImpl::set_ipv6_v6only}} -->
Sets the IPv6 socket option to either enable or disable the use of IPv6-only mode.
- **Inputs**:
    - `on`: A boolean value indicating whether to enable (true) or disable (false) IPv6-only mode.
- **Control Flow**:
    - The function directly assigns the input boolean value 'on' to the member variable 'ipv6_v6only_'.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` class by setting the `ipv6_v6only_` member variable.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_socket\_options<!-- {{#callable:ClientImpl::set_socket_options}} -->
Sets the socket options for the `ClientImpl` instance.
- **Inputs**:
    - `socket_options`: An instance of `SocketOptions` that contains the configuration settings for the socket.
- **Control Flow**:
    - The function takes the `socket_options` parameter and moves it into the member variable `socket_options_`.
    - No additional logic or error handling is performed within this function.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by updating its socket options.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_compress<!-- {{#callable:ClientImpl::set_compress}} -->
Sets the compression flag for the `ClientImpl` instance.
- **Inputs**:
    - `on`: A boolean value indicating whether compression should be enabled (true) or disabled (false).
- **Control Flow**:
    - The function directly assigns the input boolean value `on` to the member variable `compress_`.
    - There are no conditional statements or loops; the function executes a single assignment operation.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by setting the `compress_` member variable.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_decompress<!-- {{#callable:ClientImpl::set_decompress}} -->
Sets the `decompress_` member variable to the specified boolean value.
- **Inputs**:
    - `on`: A boolean value indicating whether decompression should be enabled (true) or disabled (false).
- **Control Flow**:
    - The function directly assigns the input boolean value to the member variable `decompress_`.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` class.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_interface<!-- {{#callable:ClientImpl::set_interface}} -->
Sets the network interface for the `ClientImpl` instance.
- **Inputs**:
    - `intf`: A constant reference to a string representing the network interface to be set.
- **Control Flow**:
    - The function directly assigns the input string `intf` to the member variable `interface_`.
    - There are no conditional statements or loops; the function executes a single assignment operation.
- **Output**: The function does not return a value; it modifies the internal state of the `ClientImpl` instance by updating the `interface_` member variable.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_proxy<!-- {{#callable:ClientImpl::set_proxy}} -->
Sets the proxy host and port for the `ClientImpl` instance.
- **Inputs**:
    - `host`: A string representing the hostname or IP address of the proxy server.
    - `port`: An integer representing the port number on which the proxy server is listening.
- **Control Flow**:
    - The function directly assigns the provided `host` to the member variable `proxy_host_`.
    - It assigns the provided `port` to the member variable `proxy_port_`.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by setting the proxy host and port.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_proxy\_basic\_auth<!-- {{#callable:ClientImpl::set_proxy_basic_auth}} -->
Sets the basic authentication credentials for the proxy.
- **Inputs**:
    - `username`: A `std::string` representing the username for proxy authentication.
    - `password`: A `std::string` representing the password for proxy authentication.
- **Control Flow**:
    - The function directly assigns the provided `username` to the member variable `proxy_basic_auth_username_`.
    - It then assigns the provided `password` to the member variable `proxy_basic_auth_password_`.
- **Output**: This function does not return a value; it updates the internal state of the `ClientImpl` object with the provided authentication credentials.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_proxy\_bearer\_token\_auth<!-- {{#callable:ClientImpl::set_proxy_bearer_token_auth}} -->
Sets the proxy bearer token for authentication.
- **Inputs**:
    - `token`: A string representing the bearer token used for proxy authentication.
- **Control Flow**:
    - The function directly assigns the provided `token` to the member variable `proxy_bearer_token_auth_token_`.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by setting the proxy bearer token.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_proxy\_digest\_auth<!-- {{#callable:ClientImpl::set_proxy_digest_auth}} -->
Sets the proxy digest authentication credentials for the `ClientImpl` instance.
- **Inputs**:
    - `username`: A `std::string` representing the username for proxy digest authentication.
    - `password`: A `std::string` representing the password for proxy digest authentication.
- **Control Flow**:
    - The function directly assigns the provided `username` to the member variable `proxy_digest_auth_username_`.
    - It then assigns the provided `password` to the member variable `proxy_digest_auth_password_`.
- **Output**: This function does not return a value; it updates the internal state of the `ClientImpl` instance with the provided authentication credentials.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_ca\_cert\_path<!-- {{#callable:ClientImpl::set_ca_cert_path}} -->
Sets the paths for the CA certificate file and directory in the `ClientImpl` class.
- **Inputs**:
    - `ca_cert_file_path`: A string representing the file path to the CA certificate.
    - `ca_cert_dir_path`: A string representing the directory path for CA certificates, defaulting to an empty string.
- **Control Flow**:
    - The function assigns the provided `ca_cert_file_path` to the member variable `ca_cert_file_path_`.
    - It assigns the provided `ca_cert_dir_path` to the member variable `ca_cert_dir_path_`.
- **Output**: This function does not return a value; it updates the internal state of the `ClientImpl` instance with the specified certificate paths.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_ca\_cert\_store<!-- {{#callable:ClientImpl::set_ca_cert_store}} -->
Sets the CA certificate store for the `ClientImpl` instance.
- **Inputs**:
    - `ca_cert_store`: A pointer to an `X509_STORE` structure that represents the CA certificate store to be set.
- **Control Flow**:
    - Checks if the provided `ca_cert_store` is not null and is different from the current `ca_cert_store_`.
    - If the conditions are met, updates the member variable `ca_cert_store_` with the new `ca_cert_store`.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::create\_ca\_cert\_store<!-- {{#callable:ClientImpl::create_ca_cert_store}} -->
Creates an `X509_STORE` from a given CA certificate in PEM format.
- **Inputs**:
    - `ca_cert`: A pointer to a character array containing the CA certificate in PEM format.
    - `size`: The size of the `ca_cert` character array.
- **Control Flow**:
    - Creates a memory buffer using `BIO_new_mem_buf` to hold the CA certificate.
    - If the memory buffer creation fails, the function returns nullptr.
    - Reads the PEM-encoded certificate information from the memory buffer using `PEM_X509_INFO_read_bio`.
    - If reading the certificate information fails, the function returns nullptr.
    - Creates a new `X509_STORE` instance.
    - Iterates over the certificate information, adding each certificate and CRL to the `X509_STORE` if they exist.
    - Frees the certificate information stack using `sk_X509_INFO_pop_free`.
    - Returns the populated `X509_STORE`.
- **Output**: Returns a pointer to an `X509_STORE` containing the certificates and CRLs from the provided CA certificate, or nullptr if an error occurs.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::enable\_server\_certificate\_verification<!-- {{#callable:ClientImpl::enable_server_certificate_verification}} -->
Enables or disables server certificate verification for the `ClientImpl` instance.
- **Inputs**:
    - `enabled`: A boolean value indicating whether server certificate verification should be enabled (true) or disabled (false).
- **Control Flow**:
    - The function directly assigns the value of the input parameter `enabled` to the member variable `server_certificate_verification_`.
    - No conditional logic or loops are present, making this a straightforward setter function.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by updating the `server_certificate_verification_` member variable.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::enable\_server\_hostname\_verification<!-- {{#callable:ClientImpl::enable_server_hostname_verification}} -->
Enables or disables server hostname verification for secure connections.
- **Inputs**:
    - `enabled`: A boolean value indicating whether to enable (true) or disable (false) server hostname verification.
- **Control Flow**:
    - The function directly assigns the value of the input parameter `enabled` to the member variable `server_hostname_verification_`.
    - No conditional logic or loops are present; the function executes a single assignment operation.
- **Output**: The function does not return a value; it modifies the internal state of the `ClientImpl` class by setting the `server_hostname_verification_` member variable.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_server\_certificate\_verifier<!-- {{#callable:ClientImpl::set_server_certificate_verifier}} -->
Sets a custom server certificate verifier for SSL connections.
- **Inputs**:
    - `verifier`: A callable function that takes an `SSL*` pointer and returns an `SSLVerifierResponse`, used to verify server certificates.
- **Control Flow**:
    - The function assigns the provided `verifier` function to the member variable `server_certificate_verifier_`.
    - No additional logic or control flow is present; the function is a simple setter.
- **Output**: The function does not return a value; it modifies the internal state of the `ClientImpl` class by setting the server certificate verifier.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)


---
#### ClientImpl::set\_logger<!-- {{#callable:ClientImpl::set_logger}} -->
Sets the `logger_` member variable of the `ClientImpl` class to the provided `logger`.
- **Inputs**:
    - `logger`: An instance of `Logger` that will be assigned to the `logger_` member variable.
- **Control Flow**:
    - The function uses `std::move` to efficiently transfer ownership of the `logger` argument to `logger_`.
    - No conditional logic or loops are present; the function directly assigns the moved logger to the member variable.
- **Output**: This function does not return a value; it modifies the internal state of the `ClientImpl` instance by setting the `logger_` member.
- **See also**: [`ClientImpl`](#ClientImpl)  (Data Structure)



---
### Socket<!-- {{#data_structure:ClientImpl::Socket}} -->
- **Type**: `struct`
- **Members**:
    - `sock`: Holds the socket descriptor, initialized to INVALID_SOCKET.
    - `ssl`: Pointer to an SSL structure, initialized to nullptr, used when OpenSSL support is enabled.
- **Description**: The `Socket` struct is a simple data structure designed to encapsulate a network socket descriptor and, optionally, an SSL pointer for secure connections when OpenSSL support is enabled. It provides a basic mechanism to check if the socket is open through the `is_open` method, which verifies if the socket descriptor is valid.
- **Member Functions**:
    - [`ClientImpl::Socket::is_open`](#Socketis_open)

**Methods**

---
#### Socket::is\_open<!-- {{#callable:ClientImpl::Socket::is_open}} -->
The `is_open` function checks if the socket is currently open by comparing it to the `INVALID_SOCKET` constant.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the member variable `sock` is not equal to `INVALID_SOCKET`.
    - If `sock` is not equal to `INVALID_SOCKET`, the function returns `true`, indicating the socket is open.
    - If `sock` is equal to `INVALID_SOCKET`, the function returns `false`, indicating the socket is not open.
- **Output**: A boolean value indicating whether the socket is open (`true`) or not (`false`).
- **See also**: [`ClientImpl::Socket`](#ClientImpl::Socket)  (Data Structure)



---
### Client<!-- {{#data_structure:Client}} -->
- **Type**: `class`
- **Members**:
    - `cli_`: A unique pointer to a ClientImpl object, which likely handles the underlying implementation details of the Client class.
    - `is_ssl_`: A boolean flag indicating whether SSL is supported or enabled for the client.
- **Description**: The `Client` class is a comprehensive HTTP client interface designed to facilitate various HTTP operations such as GET, POST, PUT, PATCH, DELETE, and OPTIONS. It provides multiple constructors to support different configurations, including SSL support and client certificate authentication. The class offers a wide range of methods to customize HTTP requests, manage connection settings, and handle authentication. It also supports setting timeouts, socket options, and proxy configurations. The `Client` class is designed to be flexible and extendable, making it suitable for a variety of HTTP communication needs.
- **Member Functions**:
    - [`Client::Client`](#ClientClient)
    - [`Client::operator=`](#Clientoperator=)
    - [`Client::set_connection_timeout`](#Clientset_connection_timeout)
    - [`Client::set_read_timeout`](#Clientset_read_timeout)
    - [`Client::set_write_timeout`](#Clientset_write_timeout)
    - [`Client::set_max_timeout`](#Clientset_max_timeout)
    - [`Client::Client`](#ClientClient)
    - [`Client::Client`](#ClientClient)
    - [`Client::Client`](#ClientClient)
    - [`Client::~Client`](#ClientClient)
    - [`Client::is_valid`](#Clientis_valid)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Get`](#ClientGet)
    - [`Client::Head`](#ClientHead)
    - [`Client::Head`](#ClientHead)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Post`](#ClientPost)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Put`](#ClientPut)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Patch`](#ClientPatch)
    - [`Client::Delete`](#ClientDelete)
    - [`Client::Delete`](#ClientDelete)
    - [`Client::Delete`](#ClientDelete)
    - [`Client::Delete`](#ClientDelete)
    - [`Client::Delete`](#ClientDelete)
    - [`Client::Delete`](#ClientDelete)
    - [`Client::Delete`](#ClientDelete)
    - [`Client::Delete`](#ClientDelete)
    - [`Client::Delete`](#ClientDelete)
    - [`Client::Delete`](#ClientDelete)
    - [`Client::Options`](#ClientOptions)
    - [`Client::Options`](#ClientOptions)
    - [`Client::send`](#Clientsend)
    - [`Client::send`](#Clientsend)
    - [`Client::stop`](#Clientstop)
    - [`Client::host`](#Clienthost)
    - [`Client::port`](#Clientport)
    - [`Client::is_socket_open`](#Clientis_socket_open)
    - [`Client::socket`](#Clientsocket)
    - [`Client::set_hostname_addr_map`](#Clientset_hostname_addr_map)
    - [`Client::set_default_headers`](#Clientset_default_headers)
    - [`Client::set_header_writer`](#Clientset_header_writer)
    - [`Client::set_address_family`](#Clientset_address_family)
    - [`Client::set_tcp_nodelay`](#Clientset_tcp_nodelay)
    - [`Client::set_socket_options`](#Clientset_socket_options)
    - [`Client::set_connection_timeout`](#Clientset_connection_timeout)
    - [`Client::set_read_timeout`](#Clientset_read_timeout)
    - [`Client::set_write_timeout`](#Clientset_write_timeout)
    - [`Client::set_basic_auth`](#Clientset_basic_auth)
    - [`Client::set_bearer_token_auth`](#Clientset_bearer_token_auth)
    - [`Client::set_digest_auth`](#Clientset_digest_auth)
    - [`Client::set_keep_alive`](#Clientset_keep_alive)
    - [`Client::set_follow_location`](#Clientset_follow_location)
    - [`Client::set_url_encode`](#Clientset_url_encode)
    - [`Client::set_compress`](#Clientset_compress)
    - [`Client::set_decompress`](#Clientset_decompress)
    - [`Client::set_interface`](#Clientset_interface)
    - [`Client::set_proxy`](#Clientset_proxy)
    - [`Client::set_proxy_basic_auth`](#Clientset_proxy_basic_auth)
    - [`Client::set_proxy_bearer_token_auth`](#Clientset_proxy_bearer_token_auth)
    - [`Client::set_proxy_digest_auth`](#Clientset_proxy_digest_auth)
    - [`Client::enable_server_certificate_verification`](#Clientenable_server_certificate_verification)
    - [`Client::enable_server_hostname_verification`](#Clientenable_server_hostname_verification)
    - [`Client::set_server_certificate_verifier`](#Clientset_server_certificate_verifier)
    - [`Client::set_logger`](#Clientset_logger)
    - [`Client::set_ca_cert_path`](#Clientset_ca_cert_path)
    - [`Client::set_ca_cert_store`](#Clientset_ca_cert_store)
    - [`Client::load_ca_cert_store`](#Clientload_ca_cert_store)
    - [`Client::get_openssl_verify_result`](#Clientget_openssl_verify_result)
    - [`Client::ssl_context`](#Clientssl_context)

**Methods**

---
#### Client::Client<!-- {{#callable:Client::Client}} -->
The `Client` class provides a mechanism for handling HTTP requests and responses, including support for SSL.
- **Inputs**:
    - `scheme_host_port`: A string representing the scheme, host, and port for the client.
    - `client_cert_path`: A string representing the file path to the client certificate.
    - `client_key_path`: A string representing the file path to the client key.
    - `host`: A string representing the host for the HTTP connection.
    - `port`: An integer representing the port for the HTTP connection.
- **Control Flow**:
    - The `Client` class has a default move constructor and move assignment operator, allowing for efficient transfer of resources.
    - The constructor initializes the client with the provided parameters, setting up the necessary configurations for HTTP communication.
    - The destructor cleans up resources associated with the `Client` instance.
- **Output**: The `Client` class does not return a value directly; instead, it manages the state and behavior of HTTP requests and responses.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::operator=<!-- {{#callable:Client::operator=}} -->
The `operator=` function is a move assignment operator for the `Client` class, allowing the transfer of resources from one `Client` instance to another.
- **Inputs**:
    - `other`: An rvalue reference to another `Client` instance from which resources will be moved.
- **Control Flow**:
    - The function uses the default move assignment operator provided by the compiler, which efficiently transfers ownership of resources from the `other` instance to the current instance.
    - No explicit resource management or cleanup is performed in this function, as it relies on the default behavior.
- **Output**: Returns a reference to the current `Client` instance after the move assignment operation.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_connection\_timeout<!-- {{#callable:Client::set_connection_timeout}} -->
Sets the connection timeout for the `Client` instance.
- **Inputs**:
    - `duration`: A `std::chrono::duration` object representing the timeout duration to be set for the connection.
- **Control Flow**:
    - The function calls the `set_connection_timeout` method of the `cli_` member, passing the `duration` argument.
    - No additional control flow or logic is present in this function.
- **Output**: This function does not return a value; it modifies the state of the `Client` instance by setting the connection timeout.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_read\_timeout<!-- {{#callable:Client::set_read_timeout}} -->
Sets the read timeout duration for the `Client` instance.
- **Inputs**:
    - `duration`: A `std::chrono::duration` object representing the timeout duration for read operations.
- **Control Flow**:
    - The function calls the `set_read_timeout` method on the `cli_` member, passing the `duration` argument.
    - No additional control flow or logic is present; it directly delegates the operation to the underlying implementation.
- **Output**: This function does not return a value; it modifies the state of the `Client` instance by setting the read timeout.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_write\_timeout<!-- {{#callable:Client::set_write_timeout}} -->
Sets the write timeout duration for the `Client` instance.
- **Inputs**:
    - `duration`: A `std::chrono::duration` object representing the timeout duration for write operations.
- **Control Flow**:
    - The function calls the `set_write_timeout` method on the `cli_` member, passing the `duration` argument.
    - No additional control flow or logic is present; the function directly delegates the operation to the `cli_` object.
- **Output**: This function does not return a value; it modifies the state of the `cli_` object to set the write timeout.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_max\_timeout<!-- {{#callable:Client::set_max_timeout}} -->
Sets the maximum timeout duration for the `Client` instance.
- **Inputs**:
    - `duration`: A `std::chrono::duration` object representing the maximum timeout duration to be set.
- **Control Flow**:
    - The function calls the `set_max_timeout` method on the `cli_` member, passing the `duration` argument.
    - No additional logic or error handling is performed within this function.
- **Output**: This function does not return a value; it modifies the state of the `Client` instance by setting the maximum timeout.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Client<!-- {{#callable:Client::Client}} -->
Constructs a `Client` object using a scheme, host, and port.
- **Inputs**:
    - `scheme_host_port`: A string representing the scheme, host, and port for the client connection.
- **Control Flow**:
    - The constructor initializes a `Client` object by delegating to another constructor of the same class.
    - It passes the `scheme_host_port` as the first argument, while providing two empty strings for the `client_cert_path` and `client_key_path` parameters.
- **Output**: The constructor does not return a value but initializes a `Client` instance with the specified parameters.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Client<!-- {{#callable:Client::Client}} -->
Constructs a `Client` object using the specified host and port.
- **Inputs**:
    - `host`: A constant reference to a string representing the hostname or IP address of the server.
    - `port`: An integer representing the port number on which the server is listening.
- **Control Flow**:
    - The constructor initializes the `Client` object by creating a unique pointer to a `ClientImpl` instance.
    - The `ClientImpl` is constructed with the provided `host` and `port` parameters.
- **Output**: This constructor does not return a value; it initializes the `Client` object.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Client<!-- {{#callable:Client::Client}} -->
Constructs a `Client` object with specified host, port, client certificate, and client key.
- **Inputs**:
    - `host`: A string representing the hostname or IP address of the server.
    - `port`: An integer representing the port number to connect to.
    - `client_cert_path`: A string representing the file path to the client's certificate.
    - `client_key_path`: A string representing the file path to the client's private key.
- **Control Flow**:
    - The constructor initializes the `Client` object by creating a unique pointer to a `ClientImpl` instance.
    - It passes the `host`, `port`, `client_cert_path`, and `client_key_path` to the `ClientImpl` constructor.
- **Output**: The constructor does not return a value but initializes the `Client` object for further use.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::\~Client<!-- {{#callable:Client::~Client}} -->
The `Client` destructor is a default destructor that cleans up resources when a `Client` object is destroyed.
- **Inputs**:
    - `none`: The destructor does not take any input arguments.
- **Control Flow**:
    - The destructor is defined as default, meaning it will automatically handle the destruction of member variables and resources without additional user-defined logic.
- **Output**: The destructor does not return any value; it simply ensures that the `Client` object is properly destroyed and resources are released.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::is\_valid<!-- {{#callable:Client::is_valid}} -->
Checks if the `Client` instance is valid by verifying that the internal client pointer is not null and that the internal client's validity check passes.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `Client` class.
- **Control Flow**:
    - The function first checks if the internal pointer `cli_` is not null.
    - If `cli_` is not null, it then calls the `is_valid()` method on the `cli_` object to determine its validity.
    - The result of the validity check is returned as a boolean value.
- **Output**: Returns a boolean value indicating whether the `Client` instance is valid, which is true if both the internal pointer is not null and the internal client is valid.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
The `Get` method in the `Client` class retrieves data from a specified path, optionally including custom headers.
- **Inputs**:
    - `path`: A `std::string` representing the URL path from which to retrieve data.
    - `headers`: An optional `Headers` object containing custom HTTP headers to include in the request.
- **Control Flow**:
    - The method calls another `Get` method on the `cli_` member, passing the `path` and `headers` arguments.
    - The `cli_` member is expected to be an instance of a class that handles the actual HTTP request and response.
- **Output**: Returns a `Result` object that encapsulates the outcome of the HTTP GET request, including any data retrieved or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
The `Get` method in the `Client` class sends an HTTP GET request to a specified path with optional headers.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the GET request is sent.
    - `headers`: A `Headers` object containing key-value pairs for HTTP headers to be included in the request.
- **Control Flow**:
    - The method calls the `Get` method of the `cli_` member, which is a pointer to the implementation of the client, passing the `path` and `headers` as arguments.
    - The result of the `cli_->Get` call is returned directly as the output of the `Get` method.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the GET request, which may include the response data or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
Executes an HTTP GET request to the specified path using the provided progress callback.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the GET request is sent.
    - `progress`: A `Progress` object that is used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Get` method of the `cli_` member, which is a pointer to the implementation of the client.
    - The `path` argument is passed directly to the `Get` method, while the `progress` argument is moved to avoid unnecessary copies.
- **Output**: Returns a `Result` object that contains the outcome of the GET request, which may include the response data or an error.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
The `Get` method in the `Client` class initiates an HTTP GET request to a specified path with optional headers and progress tracking.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the GET request is sent.
    - `headers`: A `Headers` object containing key-value pairs for HTTP headers to be included in the request.
    - `progress`: A `Progress` object that allows tracking the progress of the request.
- **Control Flow**:
    - The method calls the `Get` function of the `cli_` member, which is a pointer to an implementation of the client, passing along the `path`, `headers`, and a moved `progress` object.
    - The `std::move(progress)` is used to efficiently transfer ownership of the `progress` object to the called function, avoiding unnecessary copies.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the GET request, which may include the response data or an error status.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
The `Get` method in the `Client` class initiates an HTTP GET request to the specified path using a provided content receiver.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the GET request is sent.
    - `content_receiver`: A `ContentReceiver` object that handles the incoming content from the GET request.
- **Control Flow**:
    - The method calls the `Get` function of the `cli_` member, which is a pointer to the implementation of the client, passing the `path` and the moved `content_receiver` as arguments.
    - The `std::move` function is used to transfer ownership of the `content_receiver` to the `cli_`'s `Get` method, allowing for efficient resource management.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the GET request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
The `Get` method in the `Client` class initiates an HTTP GET request to a specified path with optional headers and a content receiver.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the GET request is sent.
    - `headers`: A `Headers` object containing key-value pairs for HTTP headers to be included in the request.
    - `content_receiver`: A `ContentReceiver` function or object that processes the content received from the GET request.
- **Control Flow**:
    - The method calls the `Get` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the `path`, `headers`, and a moved `content_receiver`.
    - The `std::move` is used to transfer ownership of the `content_receiver` to the `cli_`'s `Get` method, allowing for efficient resource management.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, which may include the response data or an error status.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
The `Get` method in the `Client` class initiates an HTTP GET request to a specified path using a content receiver and progress handler.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the GET request is sent.
    - `content_receiver`: A `ContentReceiver` object that handles the content received from the GET request.
    - `progress`: A `Progress` object that tracks the progress of the GET request.
- **Control Flow**:
    - The method calls the `Get` function of the `cli_` member, which is a pointer to the implementation of the client.
    - It passes the `path`, `content_receiver`, and `progress` arguments to the `cli_`'s `Get` method.
    - The `std::move` function is used to efficiently transfer ownership of `content_receiver` and `progress` to the `cli_`'s method.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the GET request, including success or failure information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
Executes an HTTP GET request using the specified path, headers, content receiver, and progress callback.
- **Inputs**:
    - `path`: A string representing the URL path to which the GET request is sent.
    - `headers`: An object of type `Headers` containing any additional HTTP headers to include in the request.
    - `content_receiver`: A callable that receives the content of the response.
    - `progress`: A callable that is invoked to report the progress of the request.
- **Control Flow**:
    - The function calls the `Get` method of the `cli_` member, which is an instance of `ClientImpl`, passing along the provided arguments.
    - The `std::move` function is used to efficiently transfer ownership of `content_receiver` and `progress` to the `cli_`'s `Get` method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including any response data or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
Executes an HTTP GET request using the specified `path`, `ResponseHandler`, and `ContentReceiver`.
- **Inputs**:
    - `path`: A `std::string` representing the URL path for the GET request.
    - `response_handler`: A `ResponseHandler` function that processes the response received from the server.
    - `content_receiver`: A `ContentReceiver` function that handles the content received from the server.
- **Control Flow**:
    - Calls the `Get` method of the `cli_` member, passing the `path`, `response_handler`, and `content_receiver` as arguments.
    - Utilizes `std::move` to efficiently transfer ownership of `response_handler` and `content_receiver` to the `cli_`'s `Get` method.
- **Output**: Returns a `Result` object that indicates the outcome of the GET request, including success or failure information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
The `Get` method in the `Client` class initiates an HTTP GET request with specified headers, a response handler, and a content receiver.
- **Inputs**:
    - `path`: A `std::string` representing the URL path for the GET request.
    - `headers`: A `Headers` object containing key-value pairs for HTTP headers to be sent with the request.
    - `response_handler`: A `ResponseHandler` function that processes the response received from the server.
    - `content_receiver`: A `ContentReceiver` function that handles the content received from the server.
- **Control Flow**:
    - The method calls the `Get` function of the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - The `std::move` function is used to efficiently transfer ownership of the `response_handler` and `content_receiver` to the `cli_`'s `Get` method.
- **Output**: Returns a `Result` object that indicates the outcome of the GET request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
The `Get` method in the `Client` class initiates an HTTP GET request using the specified path and various optional parameters.
- **Inputs**:
    - `path`: A `std::string` representing the URL path for the GET request.
    - `response_handler`: A `ResponseHandler` function that processes the response received from the server.
    - `content_receiver`: A `ContentReceiver` function that handles the content received from the server.
    - `progress`: A `Progress` function that tracks the progress of the request.
- **Control Flow**:
    - The method calls the `Get` function of the `cli_` member, which is a pointer to the implementation of the client.
    - It passes the `path`, `response_handler`, `content_receiver`, and `progress` parameters to the `cli_`'s `Get` method.
    - The parameters are moved to optimize performance, avoiding unnecessary copies.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
The `Get` method in the `Client` class initiates an HTTP GET request with specified parameters.
- **Inputs**:
    - `path`: A `std::string` representing the URL path for the GET request.
    - `headers`: A `Headers` object containing key-value pairs for HTTP headers to be sent with the request.
    - `response_handler`: A `ResponseHandler` function that processes the response received from the server.
    - `content_receiver`: A `ContentReceiver` function that handles the content received in the response.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Get` function of the `cli_` member, which is an instance of `ClientImpl`, passing along the provided parameters.
    - The parameters are moved to avoid unnecessary copies, optimizing performance.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
The `Get` method in the `Client` class performs an HTTP GET request with specified parameters, headers, and progress tracking.
- **Inputs**:
    - `path`: A `std::string` representing the URL path for the GET request.
    - `params`: A `Params` object containing query parameters to be included in the GET request.
    - `headers`: A `Headers` object containing HTTP headers to be sent with the request.
    - `progress`: A `Progress` callback function to track the progress of the request.
- **Control Flow**:
    - The method calls the `Get` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - The `std::move(progress)` is used to efficiently transfer ownership of the `progress` callback to the `cli_`'s `Get` method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including any response data or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
The `Get` method in the `Client` class initiates an HTTP GET request with specified parameters, headers, and optional content handling.
- **Inputs**:
    - `path`: A `std::string` representing the URL path for the GET request.
    - `params`: A `Params` object containing query parameters to be included in the request.
    - `headers`: A `Headers` object containing HTTP headers to be sent with the request.
    - `content_receiver`: A `ContentReceiver` function or object that handles the response content.
    - `progress`: A `Progress` function or object that tracks the progress of the request.
- **Control Flow**:
    - The method calls the `Get` function of the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - The `std::move` function is used to efficiently transfer ownership of `content_receiver` and `progress` to the `cli_`'s `Get` method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Get<!-- {{#callable:Client::Get}} -->
The `Get` method in the `Client` class initiates an HTTP GET request with specified parameters, headers, and handlers.
- **Inputs**:
    - `path`: A `std::string` representing the URL path for the GET request.
    - `params`: A `Params` object containing query parameters to be included in the request.
    - `headers`: A `Headers` object containing HTTP headers to be sent with the request.
    - `response_handler`: A `ResponseHandler` function that processes the response received from the server.
    - `content_receiver`: A `ContentReceiver` function that handles the content received in the response.
    - `progress`: A `Progress` function that tracks the progress of the request.
- **Control Flow**:
    - The method calls the `Get` method of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes all the input parameters to the `cli_`'s `Get` method, utilizing `std::move` for the handler and receiver functions to optimize performance.
- **Output**: Returns a `Result` object that encapsulates the outcome of the GET request, including success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Head<!-- {{#callable:Client::Head}} -->
The `Head` method sends an HTTP HEAD request to the specified path.
- **Inputs**:
    - `path`: A string representing the URL path to which the HEAD request is sent.
    - `headers`: An optional `Headers` object containing additional HTTP headers to include in the request.
- **Control Flow**:
    - The method `Head` is called with a single argument `path` or with `path` and `headers`.
    - In both cases, it delegates the actual request handling to the `cli_` member's `Head` method, which is expected to perform the HTTP request.
- **Output**: Returns a `Result` object that encapsulates the outcome of the HTTP HEAD request, including status and any response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Head<!-- {{#callable:Client::Head}} -->
The `Head` method sends an HTTP HEAD request to the specified path with optional headers.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the HEAD request is sent.
    - `headers`: A `Headers` object containing any additional HTTP headers to include in the request.
- **Control Flow**:
    - The method calls the `Head` function of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes the `path` and `headers` arguments directly to this `Head` function.
- **Output**: Returns a `Result` object that contains the response from the HEAD request.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends an HTTP POST request to a specified path, optionally including headers.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An optional `Headers` object containing key-value pairs for HTTP headers to include in the request.
- **Control Flow**:
    - The method first checks if the `headers` argument is provided; if not, it calls the single-argument version of `Post`.
    - If `headers` are provided, it forwards the `path` and `headers` to the underlying implementation's `Post` method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, including any response data or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends a POST request to a specified path with optional headers.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the POST request is sent.
    - `headers`: A `Headers` object containing key-value pairs for HTTP headers to be included in the request.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client, passing the `path` and `headers` as arguments.
    - The result of the `cli_->Post` call is returned directly.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, including any response data or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` function sends an HTTP POST request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `body`: A pointer to a character array containing the body of the POST request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The function calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - The result of the `Post` call is returned directly to the caller.
- **Output**: The function returns a `Result` object that encapsulates the outcome of the POST request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends an HTTP POST request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `body`: A pointer to a character array representing the body content of the POST request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the body content.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client, passing all the input parameters.
    - The `Post` method of `cli_` handles the actual sending of the HTTP request and returns the result.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, including success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends an HTTP POST request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for HTTP headers.
    - `body`: A pointer to a character array representing the body content of the POST request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client, passing all the input parameters.
    - The `cli_` member is expected to handle the actual HTTP request and return a `Result` object.
- **Output**: Returns a `Result` object that contains the outcome of the POST request, which may include the response status, headers, and body.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends an HTTP POST request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the POST request is sent.
    - `body`: A `std::string` containing the data to be sent in the body of the POST request.
    - `content_type`: A `std::string` specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the `path`, `body`, and `content_type` arguments.
    - The result of the `cli_->Post` call is returned directly as the output of this method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the POST request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends an HTTP POST request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `body`: A string containing the data to be sent in the body of the POST request.
    - `content_type`: A string indicating the MIME type of the content being sent.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - The `cli_` member is expected to handle the actual HTTP request and return a `Result` object.
- **Output**: Returns a `Result` object that contains the outcome of the POST request, which may include the response data or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends an HTTP POST request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for HTTP headers to include in the request.
    - `body`: A string containing the body content to be sent with the POST request.
    - `content_type`: A string indicating the MIME type of the body content.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - The `Post` method of `cli_` handles the actual sending of the HTTP request and returns the result.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the POST request, which may include the response status, headers, and body.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
Sends an HTTP POST request to the specified path with the given headers, body, content type, and progress callback.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for the HTTP headers to be included in the request.
    - `body`: A string containing the body of the POST request.
    - `content_type`: A string specifying the MIME type of the content being sent in the body.
    - `progress`: A `Progress` callback function that can be used to track the progress of the request.
- **Control Flow**:
    - The function calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client, passing all the input parameters.
    - The `Post` method of `cli_` handles the actual sending of the HTTP request and returns the result.
- **Output**: Returns a `Result` object that contains the outcome of the POST request, which may include the response data or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends a POST request to a specified path with a given content length, content provider, and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `content_length`: A size_t value indicating the length of the content being sent.
    - `content_provider`: A `ContentProvider` object that supplies the content to be sent in the request.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to a `ClientImpl` instance.
    - It passes the `path`, `content_length`, `content_provider` (moved to avoid copying), and `content_type` to the `cli_`'s `Post` method.
    - The result of the `cli_`'s `Post` method is returned directly.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the POST request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends a POST request to a specified path using a content provider without a specified length.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `content_provider`: A `ContentProviderWithoutLength` object that provides the content to be sent in the POST request.
    - `content_type`: A string indicating the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, passing the `path`, `content_provider`, and `content_type` as arguments.
    - The `std::move` function is used to efficiently transfer ownership of the `content_provider` to the `cli_`'s `Post` method.
- **Output**: Returns a `Result` object that represents the outcome of the POST request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends an HTTP POST request to a specified path with optional headers, content length, and a content provider.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `content_length`: A size_t value indicating the length of the content being sent.
    - `content_provider`: A `ContentProvider` object that provides the content to be sent in the request.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client, passing all the input parameters.
    - The `std::move` function is used to efficiently transfer ownership of the `content_provider` to the `cli_`'s `Post` method.
- **Output**: Returns a `Result` object that represents the outcome of the POST request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends an HTTP POST request to a specified path with optional headers and a content provider.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An object representing the HTTP headers to include in the request.
    - `content_provider`: A `ContentProviderWithoutLength` object that provides the content to be sent in the request body.
    - `content_type`: A string indicating the MIME type of the content being sent.
- **Control Flow**:
    - The method calls another `Post` method on the `cli_` member, passing the provided arguments.
    - The `std::move` function is used to efficiently transfer ownership of the `content_provider` to the called method.
- **Output**: Returns a `Result` object that represents the outcome of the POST request.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends a POST request to a specified path with given parameters.
- **Inputs**:
    - `path`: A string representing the endpoint to which the POST request is sent.
    - `params`: An object of type `Params` containing the parameters to be included in the POST request.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to the implementation of the client.
    - It passes the `path` and `params` arguments directly to the `cli_`'s `Post` method.
- **Output**: Returns a `Result` object that contains the outcome of the POST request.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method in the `Client` class sends a POST request to a specified path with optional headers and parameters.
- **Inputs**:
    - `path`: A `std::string` representing the endpoint to which the POST request is sent.
    - `headers`: An instance of `Headers` containing key-value pairs for HTTP headers to be included in the request.
    - `params`: An instance of `Params` that holds additional parameters to be sent with the request.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the `path`, `headers`, and `params`.
    - The `cli_` member is expected to handle the actual HTTP request and return a `Result` object.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the POST request, which may include the response data, status code, and any errors encountered.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends a POST request to a specified path with optional headers, parameters, and progress tracking.
- **Inputs**:
    - `path`: A string representing the endpoint to which the POST request is sent.
    - `headers`: An object representing the HTTP headers to include in the request.
    - `params`: An object containing parameters to be sent with the request.
    - `progress`: A callback function to track the progress of the request.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is an instance of `ClientImpl`, passing along the provided arguments.
    - The `cli_` object handles the actual HTTP request and response processing.
- **Output**: Returns a `Result` object that contains the outcome of the POST request, including any response data or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` function sends a POST request to a specified path with multipart form data.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `items`: A collection of multipart form data items to be included in the POST request.
- **Control Flow**:
    - The function calls the `Post` method of the `cli_` member, which is a pointer to the implementation of the client.
    - It passes the `path` and `items` arguments directly to the `cli_`'s `Post` method.
- **Output**: Returns a `Result` object that contains the outcome of the POST request.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends a POST request to a specified path with optional headers and multipart form data.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for the HTTP headers to be included in the request.
    - `items`: An object of type `MultipartFormDataItems` representing the multipart form data to be sent in the request.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the `path`, `headers`, and `items`.
    - The `cli_` member is expected to handle the actual HTTP request and return a `Result` object.
- **Output**: The method returns a `Result` object that contains the outcome of the POST request, which may include the response status, headers, and body.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends a POST request to a specified path with optional headers, multipart form data items, and a boundary string.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for the HTTP headers to be included in the request.
    - `items`: An object of type `MultipartFormDataItems` representing the form data to be sent in the request.
    - `boundary`: A string that defines the boundary used in the multipart form data.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - The `cli_` member is expected to handle the actual HTTP request and return a `Result` object.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the POST request, including any response data or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Post<!-- {{#callable:Client::Post}} -->
The `Post` method sends a POST request to a specified path with optional headers and multipart form data.
- **Inputs**:
    - `path`: A string representing the URL path to which the POST request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for HTTP headers.
    - `items`: An object of type `MultipartFormDataItems` representing the form data to be sent.
    - `provider_items`: An object of type `MultipartFormDataProviderItems` providing additional multipart form data.
- **Control Flow**:
    - The method calls the `Post` method of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes the `path`, `headers`, `items`, and `provider_items` arguments directly to the `cli_`'s `Post` method.
    - The result of the `cli_`'s `Post` method is returned as the output of this method.
- **Output**: The output is a `Result` object that contains the response from the POST request, which may include status information and data returned from the server.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `body`: A pointer to a character array representing the body of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the body content.
- **Control Flow**:
    - The method calls the `Put` function of the `cli_` member, which is a pointer to the implementation of the client.
    - It passes the provided `path` and other parameters to the `cli_`'s `Put` method.
    - The control flow is straightforward as it directly forwards the parameters without additional logic.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PUT request.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `body`: A pointer to a character array containing the body of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is an instance of `ClientImpl`, passing along the provided arguments.
    - The `Put` method of `cli_` handles the actual sending of the PUT request and returns a `Result` object.
- **Output**: The method returns a `Result` object that indicates the outcome of the PUT request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends an HTTP PUT request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `body`: A pointer to a character array representing the body content of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the body content.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is an instance of `ClientImpl`, passing all the input parameters.
    - The result of the `cli_->Put` call is returned directly as the output of this method.
- **Output**: The output is a `Result` object that encapsulates the response from the HTTP PUT request, which may include status information and any returned data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends an HTTP PUT request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `body`: A pointer to a character array representing the body content of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is a pointer to an implementation of the client, passing all the input parameters.
    - The `cli_` member handles the actual HTTP request and returns a `Result` object.
- **Output**: The method returns a `Result` object that contains the outcome of the PUT request, which may include status information and response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PUT request is sent.
    - `body`: A `std::string` containing the data to be sent in the body of the PUT request.
    - `content_type`: A `std::string` specifying the MIME type of the body content.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is a pointer to a `ClientImpl` instance.
    - It passes the `path`, `body`, and `content_type` arguments directly to the `cli_`'s `Put` method.
    - The result of the `cli_`'s `Put` method is returned as the output of this method.
- **Output**: Returns a `Result` object that contains the outcome of the PUT request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `body`: A string containing the data to be sent in the body of the PUT request.
    - `content_type`: A string indicating the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes the `path`, `body`, `content_type`, and `progress` parameters directly to the `cli_`'s `Put` method.
    - The result of the `cli_`'s `Put` method is returned as the output of this method.
- **Output**: Returns a `Result` object that contains the outcome of the PUT request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends an HTTP PUT request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing key-value pairs for the HTTP headers to be included in the request.
    - `body`: A string containing the body content to be sent with the PUT request.
    - `content_type`: A string specifying the MIME type of the body content.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - The result of the `cli_->Put` call is returned directly as the output of this method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PUT request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends an HTTP PUT request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `body`: A string containing the body content to be sent with the PUT request.
    - `content_type`: A string specifying the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is a pointer to an implementation of the client, passing all the input parameters.
    - The `cli_` member handles the actual HTTP request and returns a `Result` object.
- **Output**: The method returns a `Result` object that contains the outcome of the PUT request, which may include status information and any response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path with a given content length and content provider.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `content_length`: A size_t value indicating the length of the content being sent.
    - `content_provider`: A `ContentProvider` object that supplies the content to be sent in the request.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is an instance of `ClientImpl`, passing along the provided arguments.
    - The `std::move` function is used to efficiently transfer ownership of the `content_provider` to the `cli_`'s `Put` method.
- **Output**: Returns a `Result` object that indicates the outcome of the PUT request.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path using a content provider without a specified length.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `content_provider`: A `ContentProviderWithoutLength` object that provides the content to be sent in the PUT request.
    - `content_type`: A string indicating the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, passing the `path`, `content_provider`, and `content_type` as arguments.
    - The `std::move` function is used to efficiently transfer ownership of the `content_provider` to the `cli_`'s `Put` method.
- **Output**: Returns a `Result` object that indicates the outcome of the PUT request.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path with given headers, content length, content provider, and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing the HTTP headers to be included in the request.
    - `content_length`: A size_t value indicating the length of the content being sent.
    - `content_provider`: A `ContentProvider` object that provides the content to be sent in the request.
    - `content_type`: A string representing the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is a pointer to an implementation of the client, passing all the input parameters.
    - The `std::move` function is used to efficiently transfer ownership of the `content_provider` to the `cli_`'s `Put` method.
- **Output**: Returns a `Result` object that represents the outcome of the PUT request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method in the `Client` class sends a PUT request to a specified path with optional headers and a content provider.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `content_provider`: A `ContentProviderWithoutLength` object that provides the content to be sent in the request.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls another `Put` method on the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - The `std::move` function is used to efficiently transfer ownership of the `content_provider` to the called method.
- **Output**: Returns a `Result` object that indicates the outcome of the PUT request.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method in the `Client` class sends a PUT request to a specified path with given parameters.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PUT request is sent.
    - `params`: A `Params` object containing the parameters to be included in the PUT request.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is a pointer to the implementation of the client, passing the `path` and `params` as arguments.
    - The result of the `cli_->Put` call is returned directly.
- **Output**: The output is a `Result` object that encapsulates the response from the PUT request, which may include status information and any data returned by the server.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path with optional headers and parameters.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `params`: An object of type `Params` that may contain additional parameters for the request.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the `path`, `headers`, and `params`.
    - The result of the `cli_->Put` call is returned directly.
- **Output**: The output is a `Result` object that encapsulates the response from the PUT request, which may include status information and any data returned by the server.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path with optional headers, parameters, and progress tracking.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PUT request is sent.
    - `headers`: A `Headers` object containing key-value pairs for HTTP headers to be included in the request.
    - `params`: A `Params` object representing additional parameters to be sent with the request.
    - `progress`: A `Progress` callback function that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - The `cli_` member is expected to handle the actual HTTP request and return a `Result` object.
- **Output**: Returns a `Result` object that contains the outcome of the PUT request, which may include status information and response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path with multipart form data.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `items`: A `MultipartFormDataItems` object containing the data to be sent in the request.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is a pointer to the implementation of the client.
    - It passes the `path` and `items` arguments directly to the `cli_`'s `Put` method.
- **Output**: Returns a `Result` object that represents the outcome of the PUT request.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path with given headers and multipart form data items.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing the HTTP headers to be included in the request.
    - `items`: An object of type `MultipartFormDataItems` representing the multipart form data to be sent in the request.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the `path`, `headers`, and `items`.
    - The `cli_` member is expected to handle the actual HTTP request and return a `Result` object.
- **Output**: The method returns a `Result` object that contains the outcome of the PUT request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path with optional headers and multipart form data.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `items`: A collection of multipart form data items to be included in the request.
    - `boundary`: A string that defines the boundary used in the multipart form data.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is an instance of `ClientImpl`, passing along the provided arguments.
    - The `Put` method of `cli_` handles the actual sending of the PUT request and returns the result.
- **Output**: Returns a `Result` object that contains the outcome of the PUT request, which may include status information and response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Put<!-- {{#callable:Client::Put}} -->
The `Put` method sends a PUT request to a specified path with optional headers and multipart form data.
- **Inputs**:
    - `path`: A string representing the URL path to which the PUT request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `items`: A collection of `MultipartFormDataItems` representing the form data to be sent in the request.
    - `provider_items`: A collection of `MultipartFormDataProviderItems` that provide additional data for the multipart form.
- **Control Flow**:
    - The method calls the `Put` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - The `cli_` object handles the actual sending of the request and returns the result.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the PUT request, including any response data or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method in the `Client` class sends a PATCH request to a specified path.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `body`: A pointer to a character array representing the body of the PATCH request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A `std::string` specifying the MIME type of the body content.
    - `progress`: An optional `Progress` callback to monitor the progress of the request.
- **Control Flow**:
    - The method calls the `Patch` method of the `cli_` member, which is a pointer to the implementation of the client.
    - It passes the `path` and other parameters to the `cli_`'s `Patch` method, which handles the actual HTTP request.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PATCH request, including success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method sends a PATCH request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `body`: A pointer to a `char` array containing the body of the request.
    - `content_length`: A `size_t` indicating the length of the body content.
    - `content_type`: A `std::string` specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Patch` method of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes the same parameters received by the `Patch` method to the `cli_`'s `Patch` method.
    - The result of the `cli_`'s `Patch` method is returned directly.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PATCH request, which may include status information and response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method sends an HTTP PATCH request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the PATCH request is sent.
    - `body`: A pointer to a character array containing the body of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the content being sent.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Patch` method of the `cli_` member, which is an instance of `ClientImpl`, passing all the input parameters.
    - The `Patch` method of `ClientImpl` is expected to handle the actual HTTP request and return a `Result` object.
- **Output**: The method returns a `Result` object that contains the outcome of the PATCH request, which may include status information and any response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method sends an HTTP PATCH request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `headers`: A `Headers` object containing any additional HTTP headers to include in the request.
    - `body`: A pointer to a `char` array representing the body content of the PATCH request.
    - `content_length`: A `size_t` indicating the length of the body content.
    - `content_type`: A `std::string` specifying the MIME type of the body content.
- **Control Flow**:
    - The method calls the `Patch` method of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes all the input parameters directly to the `cli_`'s `Patch` method.
    - The result of the `cli_`'s `Patch` method is returned as the output of this method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PATCH request, which may include status information and response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method sends an HTTP PATCH request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PATCH request is sent.
    - `headers`: An object of type `Headers` containing any additional HTTP headers to include in the request.
    - `body`: A pointer to a character array representing the body content of the PATCH request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Patch` method of the `cli_` member, which is a pointer to an implementation of the client, passing all the input parameters.
    - The `cli_` member is expected to handle the actual HTTP request and return a `Result` object.
- **Output**: The method returns a `Result` object that contains the outcome of the PATCH request, which may include the response status, headers, and body.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method sends a PATCH request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `body`: A `std::string` containing the data to be sent in the body of the PATCH request.
    - `content_type`: A `std::string` specifying the MIME type of the content being sent in the request.
- **Control Flow**:
    - The method directly calls the `Patch` method of the `cli_` member, which is an instance of `ClientImpl`, passing along the provided arguments.
    - No additional logic or error handling is implemented within this method; it simply acts as a forwarding function.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PATCH request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method sends a PATCH request to a specified path with a given body and content type.
- **Inputs**:
    - `path`: A string representing the URL path to which the PATCH request is sent.
    - `body`: A string containing the data to be sent in the body of the PATCH request.
    - `content_type`: A string indicating the media type of the resource being sent.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Patch` method of the `cli_` member, which is an instance of `ClientImpl`, passing along the provided arguments.
    - The `Patch` method of `cli_` handles the actual sending of the PATCH request and returns the result.
- **Output**: Returns a `Result` object that contains the outcome of the PATCH request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method sends an HTTP PATCH request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `headers`: A `Headers` object containing any additional HTTP headers to include in the request.
    - `body`: A `std::string` containing the body of the PATCH request, which may include data to be updated.
    - `content_type`: A `std::string` specifying the media type of the body content, such as 'application/json'.
- **Control Flow**:
    - The method calls the `Patch` method of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes the `path`, `headers`, `body`, and `content_type` arguments directly to the `cli_`'s `Patch` method.
    - The result of the `cli_`'s `Patch` method is returned as the output of this method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the PATCH request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method sends a PATCH request to a specified path with optional headers and body content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PATCH request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `body`: A string containing the body of the PATCH request.
    - `content_type`: A string specifying the content type of the body being sent.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Patch` method of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes all the input parameters directly to the `cli_`'s `Patch` method.
    - The result of the `cli_`'s `Patch` method is returned as the output of this method.
- **Output**: The output is a `Result` object that contains the response from the PATCH request, which may include status information and any returned data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method sends a PATCH request to a specified path with a given content length and content provider.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `content_length`: A `size_t` indicating the length of the content being sent.
    - `content_provider`: A `ContentProvider` object that supplies the content to be sent in the request.
    - `content_type`: A `std::string` specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Patch` method of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes the `path`, `content_length`, a moved `content_provider`, and `content_type` to the `cli_`'s `Patch` method.
    - The result of the `cli_`'s `Patch` method is returned directly.
- **Output**: Returns a `Result` object that represents the outcome of the PATCH request.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method sends a PATCH request to a specified path using a content provider.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the PATCH request is sent.
    - `content_provider`: A `ContentProviderWithoutLength` function that provides the content to be sent in the PATCH request.
    - `content_type`: A `std::string` indicating the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Patch` method of the `cli_` member, which is a pointer to an implementation of the client.
    - It uses `std::move` to transfer ownership of the `content_provider` to the `cli_`'s `Patch` method, ensuring efficient resource management.
- **Output**: Returns a `Result` object that indicates the outcome of the PATCH request.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method sends an HTTP PATCH request to a specified path with optional headers and content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PATCH request is sent.
    - `headers`: An object of type `Headers` containing any additional HTTP headers to include in the request.
    - `content_length`: A size_t value indicating the length of the content being sent.
    - `content_provider`: A `ContentProvider` function or object that provides the content to be sent in the request.
    - `content_type`: A string specifying the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Patch` method of the `cli_` member, which is a pointer to an implementation of the client, passing all the input parameters.
    - The `std::move` function is used to efficiently transfer ownership of the `content_provider` to the `cli_`'s `Patch` method.
- **Output**: Returns a `Result` object that represents the outcome of the PATCH request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Patch<!-- {{#callable:Client::Patch}} -->
The `Patch` method sends an HTTP PATCH request to a specified path with optional headers and content.
- **Inputs**:
    - `path`: A string representing the URL path to which the PATCH request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `content_provider`: A `ContentProviderWithoutLength` function that provides the content to be sent in the request body.
    - `content_type`: A string indicating the MIME type of the content being sent.
- **Control Flow**:
    - The method calls the `Patch` method of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes the `path`, `headers`, a moved version of `content_provider`, and `content_type` to the `cli_`'s `Patch` method.
    - The result of the `cli_`'s `Patch` method is returned directly.
- **Output**: The output is a `Result` object that represents the outcome of the PATCH request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Delete<!-- {{#callable:Client::Delete}} -->
The `Delete` method in the `Client` class sends a DELETE request to a specified path.
- **Inputs**:
    - `path`: A string representing the resource path to which the DELETE request is sent.
    - `headers`: An optional `Headers` object containing additional HTTP headers to include in the request.
- **Control Flow**:
    - The method calls the `Delete` method of the `cli_` member, which is an instance of `ClientImpl`, passing the `path` and optionally `headers`.
    - The `Delete` method of `cli_` handles the actual HTTP DELETE request and returns a `Result` object.
- **Output**: The method returns a `Result` object that contains the outcome of the DELETE request, which may include status information and any response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Delete<!-- {{#callable:Client::Delete}} -->
The `Delete` method in the `Client` class sends an HTTP DELETE request to a specified path.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the DELETE request is sent.
    - `headers`: A `Headers` object containing any additional HTTP headers to include in the request.
- **Control Flow**:
    - The method calls the `Delete` function of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes the `path` and `headers` arguments directly to the `cli_`'s `Delete` method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the DELETE request, including any response data or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Delete<!-- {{#callable:Client::Delete}} -->
The `Delete` function in the `Client` class sends a DELETE request to a specified path with an optional body and content type.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the DELETE request is sent.
    - `body`: A pointer to a `char` array representing the body of the DELETE request, which can be null if no body is needed.
    - `content_length`: A `size_t` indicating the length of the body content.
    - `content_type`: A `std::string` specifying the MIME type of the body content.
- **Control Flow**:
    - The function calls the `Delete` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - It does not contain any additional logic or error handling; it simply forwards the parameters to the underlying implementation.
- **Output**: Returns a `Result` object that represents the outcome of the DELETE request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Delete<!-- {{#callable:Client::Delete}} -->
The `Delete` method in the `Client` class sends a DELETE request to a specified path with optional body content and headers.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the DELETE request is sent.
    - `body`: A pointer to a character array representing the body of the request, which can be null.
    - `content_length`: A `size_t` indicating the length of the body content.
    - `content_type`: A `std::string` specifying the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Delete` method of the `cli_` member, which is a pointer to an implementation of the client, passing all the input parameters.
    - The `cli_` member handles the actual HTTP DELETE request and returns the result.
- **Output**: Returns a `Result` object that contains the outcome of the DELETE request, which may include status information and any response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Delete<!-- {{#callable:Client::Delete}} -->
This function sends a DELETE request to a specified path with optional headers and body.
- **Inputs**:
    - `path`: A string representing the URL path to which the DELETE request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `body`: A pointer to a character array representing the body of the request, which can be null.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string representing the MIME type of the body content.
- **Control Flow**:
    - The function directly calls the `Delete` method of the `cli_` member, which is an instance of `ClientImpl`.
    - It passes all the input parameters to the `Delete` method of `cli_` without any modification.
- **Output**: Returns a `Result` object that encapsulates the outcome of the DELETE request, including success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Delete<!-- {{#callable:Client::Delete}} -->
The `Delete` method in the `Client` class sends an HTTP DELETE request to a specified path.
- **Inputs**:
    - `path`: A string representing the URL path to which the DELETE request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `body`: A pointer to a character array representing the body of the request.
    - `content_length`: A size_t value indicating the length of the body content.
    - `content_type`: A string specifying the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls another `Delete` method on the `cli_` member, which is a pointer to an implementation of the client.
    - It passes all the input parameters directly to the `cli_`'s `Delete` method.
    - The result of the `cli_`'s `Delete` method is returned as the output of this method.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the DELETE request, which may include status information and any response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Delete<!-- {{#callable:Client::Delete}} -->
The `Delete` method in the `Client` class sends a DELETE request to a specified path with an optional body and content type.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the DELETE request is sent.
    - `body`: A `std::string` containing the body of the DELETE request, which can be empty if not needed.
    - `content_type`: A `std::string` specifying the content type of the body being sent with the DELETE request.
- **Control Flow**:
    - The method calls the `Delete` method of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes the `path`, `body`, and `content_type` arguments directly to the `cli_`'s `Delete` method.
    - The result of the `cli_`'s `Delete` method is returned as the output of this method.
- **Output**: The method returns a `Result` object that encapsulates the outcome of the DELETE request, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Delete<!-- {{#callable:Client::Delete}} -->
The `Delete` method in the `Client` class sends a DELETE request to a specified path with optional body content and headers.
- **Inputs**:
    - `path`: A string representing the URL path to which the DELETE request is sent.
    - `body`: A string containing the body content to be sent with the DELETE request.
    - `content_type`: A string indicating the MIME type of the body content.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The method calls the `Delete` method of the `cli_` member, which is an instance of `ClientImpl`, passing along the provided arguments.
    - The `Delete` method of `cli_` handles the actual HTTP DELETE request and returns a `Result` object.
- **Output**: The method returns a `Result` object that contains the outcome of the DELETE request, which may include status information and any response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Delete<!-- {{#callable:Client::Delete}} -->
The `Delete` method in the `Client` class sends a DELETE request to a specified path with optional headers and body.
- **Inputs**:
    - `path`: A string representing the URL path to which the DELETE request is sent.
    - `headers`: An object of type `Headers` containing any additional headers to include in the request.
    - `body`: A string representing the body of the request, which can be empty if not needed.
    - `content_type`: A string indicating the content type of the body, used to inform the server how to interpret the body.
- **Control Flow**:
    - The method calls the `Delete` method of the `cli_` member, which is a pointer to an implementation of the client, passing along the provided arguments.
    - The result of the `cli_->Delete` call is returned directly, which encapsulates the outcome of the DELETE request.
- **Output**: The output is a `Result` object that indicates the success or failure of the DELETE request, along with any relevant response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Delete<!-- {{#callable:Client::Delete}} -->
Executes an HTTP DELETE request using the specified parameters.
- **Inputs**:
    - `path`: A string representing the URL path to which the DELETE request is sent.
    - `headers`: An object of type `Headers` containing any additional HTTP headers to include in the request.
    - `body`: A string representing the body content to be sent with the DELETE request.
    - `content_type`: A string indicating the content type of the body being sent.
    - `progress`: A `Progress` object that can be used to track the progress of the request.
- **Control Flow**:
    - The function calls the `Delete` method of the `cli_` member, which is a pointer to an implementation of the client, passing all the input parameters.
    - The `Delete` method of `cli_` handles the actual HTTP DELETE request and returns a `Result` object.
- **Output**: Returns a `Result` object that contains the outcome of the DELETE request, which may include status information and any response data.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Options<!-- {{#callable:Client::Options}} -->
The `Options` method in the `Client` class retrieves HTTP OPTIONS for a specified path.
- **Inputs**:
    - `path`: A `std::string` representing the URL path for which the OPTIONS request is made.
    - `headers`: An optional `Headers` object containing additional HTTP headers to include in the request.
- **Control Flow**:
    - The method calls the `Options` method of the `cli_` member, which is a pointer to the implementation of the client.
    - It passes the `path` and optionally the `headers` to the `cli_`'s `Options` method.
- **Output**: Returns a `Result` object that encapsulates the outcome of the OPTIONS request, including any response data or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::Options<!-- {{#callable:Client::Options}} -->
This function sends an HTTP OPTIONS request to a specified path with optional headers.
- **Inputs**:
    - `path`: A `std::string` representing the URL path to which the OPTIONS request is sent.
    - `headers`: A `Headers` object containing any additional HTTP headers to include in the request.
- **Control Flow**:
    - The function calls the `Options` method of the `cli_` member, which is a pointer to an implementation of the client.
    - It passes the `path` and `headers` arguments directly to this method.
- **Output**: Returns a `Result` object that encapsulates the response from the OPTIONS request.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::send<!-- {{#callable:Client::send}} -->
The `send` method in the `Client` class forwards a `Request` object to the underlying client implementation and retrieves the corresponding `Response` and `Error`.
- **Inputs**:
    - `req`: A reference to a `Request` object that contains the details of the request to be sent.
    - `res`: A reference to a `Response` object that will be populated with the response data from the server.
    - `error`: A reference to an `Error` object that will capture any errors that occur during the sending process.
- **Control Flow**:
    - The method calls the `send` method of the `cli_` member, which is a pointer to the underlying client implementation.
    - It passes the `req`, `res`, and `error` references to the `cli_`'s `send` method.
    - The return value of the `cli_`'s `send` method is returned directly by the `Client::send` method.
- **Output**: The method returns a boolean value indicating the success or failure of the send operation, as determined by the underlying client implementation.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::send<!-- {{#callable:Client::send}} -->
Sends a `Request` object using the underlying client implementation.
- **Inputs**:
    - `req`: A constant reference to a `Request` object that contains the details of the request to be sent.
- **Control Flow**:
    - The function calls the `send` method of the `cli_` member, which is a pointer to the underlying client implementation.
    - It passes the `req` object to the `cli_`'s `send` method and returns the result.
- **Output**: Returns a `Result` object that represents the outcome of the send operation, which may include success or error information.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::stop<!-- {{#callable:Client::stop}} -->
Stops the client by invoking the `stop` method on its internal implementation.
- **Inputs**: None
- **Control Flow**:
    - The method directly calls the `stop` method on the `cli_` member, which is a pointer to the implementation of the client.
- **Output**: This function does not return a value; it performs an action to stop the client.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::host<!-- {{#callable:Client::host}} -->
The `host` method retrieves the host string from the underlying client implementation.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `Client` class.
- **Control Flow**:
    - The method directly calls the `host` method of the `cli_` member, which is a pointer to the `ClientImpl` class.
    - It returns the result of the `cli_->host()` call, which is expected to be a string representing the host.
- **Output**: The output is a `std::string` that represents the host associated with the client.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::port<!-- {{#callable:Client::port}} -->
Returns the port number associated with the `Client` instance.
- **Inputs**:
    - `this`: A constant reference to the `Client` instance, which is implied and not explicitly passed as an argument.
- **Control Flow**:
    - The function directly calls the `port()` method of the `cli_` member, which is a pointer to the implementation of the client.
    - No conditional logic or loops are present; the function simply returns the result of the `cli_->port()` call.
- **Output**: An integer representing the port number that the `Client` instance is using.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::is\_socket\_open<!-- {{#callable:Client::is_socket_open}} -->
Checks if the socket associated with the `Client` instance is open.
- **Inputs**:
    - `this`: A constant reference to the `Client` instance, which contains the socket to be checked.
- **Control Flow**:
    - The function directly calls the `is_socket_open` method of the `cli_` member, which is a pointer to the implementation of the client.
    - The result of the `cli_->is_socket_open()` call is returned as the output of the function.
- **Output**: Returns a size_t value indicating the status of the socket; typically, a non-zero value indicates that the socket is open.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::socket<!-- {{#callable:Client::socket}} -->
Returns the socket associated with the `Client` instance.
- **Inputs**:
    - `this`: A constant reference to the `Client` instance, which is implied and not explicitly passed as an argument.
- **Control Flow**:
    - The function directly calls the `socket()` method on the `cli_` member, which is a pointer to the implementation of the client.
    - No conditional logic or loops are present; the function simply returns the result of the `socket()` call.
- **Output**: Returns a `socket_t` type, which represents the socket associated with the `Client` instance.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_hostname\_addr\_map<!-- {{#callable:Client::set_hostname_addr_map}} -->
Sets a mapping of hostnames to addresses in the `Client` instance.
- **Inputs**:
    - `addr_map`: A `std::map` where the keys are hostnames (as `std::string`) and the values are their corresponding addresses (also as `std::string`).
- **Control Flow**:
    - The function takes a `std::map<std::string, std::string>` as an argument.
    - It calls the `set_hostname_addr_map` method of the `cli_` member, passing the `addr_map` after moving it to avoid unnecessary copies.
- **Output**: This function does not return a value; it modifies the internal state of the `Client` instance by updating the hostname-address mapping.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_default\_headers<!-- {{#callable:Client::set_default_headers}} -->
Sets the default HTTP headers for the `Client` instance.
- **Inputs**:
    - `headers`: An instance of `Headers` containing the default headers to be set for the HTTP client.
- **Control Flow**:
    - The function calls the `set_default_headers` method on the `cli_` member, passing the `headers` argument after moving it.
    - The use of `std::move` indicates that the ownership of the `headers` object is transferred to the `cli_` object, optimizing performance by avoiding unnecessary copies.
- **Output**: This function does not return a value; it modifies the state of the `Client` instance by setting its default headers.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_header\_writer<!-- {{#callable:Client::set_header_writer}} -->
Sets a custom header writer function for the `Client`.
- **Inputs**:
    - `writer`: A `std::function` that takes a `Stream` and `Headers` as parameters and returns a `ssize_t`, representing the custom logic for writing headers.
- **Control Flow**:
    - The function takes a single argument, `writer`, which is a callable function.
    - It then calls the `set_header_writer` method on the `cli_` member, passing the `writer` function to it.
- **Output**: This function does not return a value; it modifies the internal state of the `Client` by setting the header writer.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_address\_family<!-- {{#callable:Client::set_address_family}} -->
Sets the address family for the `Client` instance.
- **Inputs**:
    - `family`: An integer representing the address family to be set, such as AF_INET for IPv4 or AF_INET6 for IPv6.
- **Control Flow**:
    - The method calls the `set_address_family` method of the `cli_` member, which is a pointer to the implementation of the client.
    - The input parameter `family` is directly passed to the `cli_` method without any additional processing.
- **Output**: This function does not return a value; it modifies the state of the `Client` instance by setting the address family.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_tcp\_nodelay<!-- {{#callable:Client::set_tcp_nodelay}} -->
Sets the TCP_NODELAY option for the client socket.
- **Inputs**:
    - `on`: A boolean value indicating whether to enable (true) or disable (false) the TCP_NODELAY option.
- **Control Flow**:
    - The function calls the `set_tcp_nodelay` method on the `cli_` member, which is a pointer to the implementation of the client.
    - The value of the `on` parameter is passed directly to the `set_tcp_nodelay` method of the `cli_` object.
- **Output**: This function does not return a value; it modifies the state of the underlying socket by enabling or disabling the TCP_NODELAY option.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_socket\_options<!-- {{#callable:Client::set_socket_options}} -->
Sets the socket options for the `Client` instance.
- **Inputs**:
    - `socket_options`: An instance of `SocketOptions` that contains the configuration settings for the socket.
- **Control Flow**:
    - The method calls `set_socket_options` on the `cli_` member, which is a pointer to the implementation of the `Client` class.
    - The `socket_options` argument is moved to avoid unnecessary copying, ensuring efficient transfer of resources.
- **Output**: This function does not return a value; it modifies the internal state of the `Client` instance by applying the provided socket options.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_connection\_timeout<!-- {{#callable:Client::set_connection_timeout}} -->
Sets the connection timeout for the `Client` instance.
- **Inputs**:
    - `sec`: The number of seconds to set as the connection timeout.
    - `usec`: The number of microseconds to set as the connection timeout.
- **Control Flow**:
    - The function calls the `set_connection_timeout` method of the `cli_` member, passing the `sec` and `usec` parameters.
    - No additional logic or error handling is performed within this function.
- **Output**: This function does not return a value; it modifies the connection timeout settings of the `Client` instance.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_read\_timeout<!-- {{#callable:Client::set_read_timeout}} -->
Sets the read timeout for the client.
- **Inputs**:
    - `sec`: The number of seconds for the read timeout.
    - `usec`: The number of microseconds for the read timeout.
- **Control Flow**:
    - Calls the `set_read_timeout` method of the `cli_` member, passing the `sec` and `usec` parameters.
- **Output**: This function does not return a value; it configures the read timeout for the client.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_write\_timeout<!-- {{#callable:Client::set_write_timeout}} -->
Sets the write timeout for the `Client` instance.
- **Inputs**:
    - `sec`: The number of seconds for the write timeout.
    - `usec`: The number of microseconds for the write timeout.
- **Control Flow**:
    - Calls the `set_write_timeout` method of the `cli_` member, passing the `sec` and `usec` parameters.
    - The `cli_` member is expected to be an instance of a class that implements the actual timeout logic.
- **Output**: This function does not return a value; it configures the write timeout for the client.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_basic\_auth<!-- {{#callable:Client::set_basic_auth}} -->
Sets the basic authentication credentials for the `Client` instance.
- **Inputs**:
    - `username`: A `std::string` representing the username for basic authentication.
    - `password`: A `std::string` representing the password for basic authentication.
- **Control Flow**:
    - The function calls the `set_basic_auth` method of the `cli_` member, passing the `username` and `password` arguments.
    - No additional logic or control flow is present in this function.
- **Output**: This function does not return a value; it configures the `Client` instance to use basic authentication with the provided credentials.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_bearer\_token\_auth<!-- {{#callable:Client::set_bearer_token_auth}} -->
Sets the bearer token for authentication in the HTTP client.
- **Inputs**:
    - `token`: A constant reference to a string representing the bearer token used for authentication.
- **Control Flow**:
    - The function calls the `set_bearer_token_auth` method of the `cli_` member, passing the provided `token` as an argument.
    - No additional logic or control flow is present; the function directly delegates the operation to the underlying client implementation.
- **Output**: This function does not return a value; it modifies the state of the `cli_` object to include the specified bearer token for future HTTP requests.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_digest\_auth<!-- {{#callable:Client::set_digest_auth}} -->
Sets the digest authentication credentials for the HTTP client.
- **Inputs**:
    - `username`: A string representing the username for digest authentication.
    - `password`: A string representing the password for digest authentication.
- **Control Flow**:
    - The function directly calls the `set_digest_auth` method of the `cli_` member, passing the `username` and `password` arguments.
    - There are no conditional statements or loops; the function simply forwards the parameters to another method.
- **Output**: This function does not return a value; it configures the authentication settings of the HTTP client.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_keep\_alive<!-- {{#callable:Client::set_keep_alive}} -->
Sets the keep-alive option for the HTTP client.
- **Inputs**:
    - `on`: A boolean value indicating whether to enable (true) or disable (false) the keep-alive feature.
- **Control Flow**:
    - The function calls the `set_keep_alive` method on the `cli_` member, which is a pointer to the implementation of the client.
    - The value of the `on` parameter is passed directly to the `set_keep_alive` method of the underlying client implementation.
- **Output**: This function does not return a value; it modifies the state of the HTTP client to either enable or disable keep-alive connections.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_follow\_location<!-- {{#callable:Client::set_follow_location}} -->
Sets the follow location option for the HTTP client.
- **Inputs**:
    - `on`: A boolean value indicating whether to enable (true) or disable (false) the follow location feature.
- **Control Flow**:
    - The function directly calls the `set_follow_location` method of the `cli_` member, which is an instance of `ClientImpl`.
    - The value of the `on` parameter is passed to the `set_follow_location` method of `cli_`.
- **Output**: This function does not return a value; it modifies the state of the `cli_` object to reflect the follow location setting.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_url\_encode<!-- {{#callable:Client::set_url_encode}} -->
Sets the URL encoding option for the client.
- **Inputs**:
    - `on`: A boolean value indicating whether to enable (true) or disable (false) URL encoding.
- **Control Flow**:
    - The function directly calls the `set_url_encode` method on the `cli_` member, which is an instance of `ClientImpl`.
    - The value of the `on` parameter is passed to the `set_url_encode` method of `cli_`.
- **Output**: This function does not return a value; it modifies the state of the `ClientImpl` instance to reflect the URL encoding setting.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_compress<!-- {{#callable:Client::set_compress}} -->
Sets the compression option for the client.
- **Inputs**:
    - `on`: A boolean value indicating whether to enable (`true`) or disable (`false`) compression.
- **Control Flow**:
    - The function calls the `set_compress` method on the `cli_` member, passing the `on` parameter to it.
- **Output**: This function does not return a value; it modifies the state of the `cli_` object to enable or disable compression.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_decompress<!-- {{#callable:Client::set_decompress}} -->
Sets the decompression option for the HTTP client.
- **Inputs**:
    - `on`: A boolean value indicating whether to enable or disable decompression.
- **Control Flow**:
    - The function calls the `set_decompress` method on the `cli_` member, passing the boolean `on` as an argument.
- **Output**: This function does not return a value; it modifies the state of the `Client` instance by setting the decompression option.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_interface<!-- {{#callable:Client::set_interface}} -->
Sets the network interface for the `Client` instance.
- **Inputs**:
    - `intf`: A constant reference to a string representing the network interface to be set.
- **Control Flow**:
    - The function calls the `set_interface` method on the `cli_` member, passing the `intf` argument.
- **Output**: This function does not return a value; it modifies the state of the `Client` instance by setting the specified network interface.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_proxy<!-- {{#callable:Client::set_proxy}} -->
Sets the proxy server address and port for the `Client`.
- **Inputs**:
    - `host`: A string representing the hostname or IP address of the proxy server.
    - `port`: An integer representing the port number on which the proxy server is listening.
- **Control Flow**:
    - The function calls the `set_proxy` method of the `cli_` member, passing the `host` and `port` arguments.
    - No additional logic or error handling is implemented within this function.
- **Output**: This function does not return a value; it configures the proxy settings for the `Client` instance.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_proxy\_basic\_auth<!-- {{#callable:Client::set_proxy_basic_auth}} -->
Sets the basic authentication credentials for the proxy.
- **Inputs**:
    - `username`: A string representing the username for proxy authentication.
    - `password`: A string representing the password for proxy authentication.
- **Control Flow**:
    - The function calls the `set_proxy_basic_auth` method of the `cli_` member, passing the provided `username` and `password` as arguments.
    - No additional logic or error handling is implemented within this function.
- **Output**: This function does not return a value; it configures the proxy authentication settings.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_proxy\_bearer\_token\_auth<!-- {{#callable:Client::set_proxy_bearer_token_auth}} -->
Sets the bearer token for proxy authentication in the `Client` class.
- **Inputs**:
    - `token`: A constant reference to a `std::string` representing the bearer token used for proxy authentication.
- **Control Flow**:
    - The function calls the `set_proxy_bearer_token_auth` method on the `cli_` member, passing the provided `token` as an argument.
    - This method is expected to handle the actual implementation of setting the bearer token for proxy authentication.
- **Output**: This function does not return a value; it modifies the state of the `cli_` object to include the specified bearer token for proxy authentication.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_proxy\_digest\_auth<!-- {{#callable:Client::set_proxy_digest_auth}} -->
Sets the proxy digest authentication credentials for the `Client`.
- **Inputs**:
    - `username`: The username for proxy authentication.
    - `password`: The password for proxy authentication.
- **Control Flow**:
    - The function directly calls the `set_proxy_digest_auth` method of the `cli_` member, passing the `username` and `password` arguments.
    - No additional logic or control flow is present in this inline function.
- **Output**: This function does not return a value; it configures the proxy authentication settings within the `Client` instance.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::enable\_server\_certificate\_verification<!-- {{#callable:Client::enable_server_certificate_verification}} -->
Enables or disables server certificate verification for the client.
- **Inputs**:
    - `enabled`: A boolean value indicating whether server certificate verification should be enabled (true) or disabled (false).
- **Control Flow**:
    - The function calls the `enable_server_certificate_verification` method on the `cli_` member, passing the `enabled` argument.
    - The `cli_` member is expected to be an instance of a class that implements the actual logic for enabling or disabling server certificate verification.
- **Output**: This function does not return a value; it modifies the state of the client to either enable or disable server certificate verification.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::enable\_server\_hostname\_verification<!-- {{#callable:Client::enable_server_hostname_verification}} -->
Enables or disables server hostname verification for the `Client`.
- **Inputs**:
    - `enabled`: A boolean value indicating whether to enable (true) or disable (false) server hostname verification.
- **Control Flow**:
    - The function calls the `enable_server_hostname_verification` method on the `cli_` member, passing the `enabled` argument.
    - The `cli_` member is expected to be an instance of a class that implements the actual logic for enabling or disabling hostname verification.
- **Output**: This function does not return a value; it modifies the state of the `Client` instance's underlying implementation regarding hostname verification.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_server\_certificate\_verifier<!-- {{#callable:Client::set_server_certificate_verifier}} -->
Sets a custom server certificate verifier for SSL connections.
- **Inputs**:
    - `verifier`: A callable function that takes an `SSL*` pointer and returns an `SSLVerifierResponse`, used to verify server certificates.
- **Control Flow**:
    - The function takes a single argument, `verifier`, which is a function that will be used to verify SSL certificates.
    - It calls the `set_server_certificate_verifier` method on the `cli_` member, passing the `verifier` function to it.
- **Output**: This function does not return a value; it configures the client to use the provided certificate verifier for SSL connections.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_logger<!-- {{#callable:Client::set_logger}} -->
Sets the `logger` for the `Client` instance.
- **Inputs**:
    - `logger`: An instance of `Logger` that will be set for the `Client`.
- **Control Flow**:
    - The function calls the `set_logger` method on the `cli_` member of the `Client` class.
    - The `logger` argument is moved into the `set_logger` method, which allows for efficient transfer of resources.
- **Output**: This function does not return a value; it modifies the state of the `Client` instance by setting its logger.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_ca\_cert\_path<!-- {{#callable:Client::set_ca_cert_path}} -->
Sets the paths for the CA certificate file and directory in the `Client` class.
- **Inputs**:
    - `ca_cert_file_path`: A string representing the file path to the CA certificate.
    - `ca_cert_dir_path`: A string representing the directory path for CA certificates.
- **Control Flow**:
    - The function calls the `set_ca_cert_path` method of the `cli_` member, passing the provided file and directory paths.
    - No additional logic or error handling is implemented within this function.
- **Output**: This function does not return a value; it modifies the state of the `cli_` object by setting the CA certificate paths.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::set\_ca\_cert\_store<!-- {{#callable:Client::set_ca_cert_store}} -->
Sets the Certificate Authority (CA) certificate store for the client.
- **Inputs**:
    - `ca_cert_store`: A pointer to an `X509_STORE` structure that contains the CA certificates.
- **Control Flow**:
    - The function first checks if the client is configured to use SSL by evaluating the `is_ssl_` member variable.
    - If SSL is enabled, it casts the `cli_` pointer to `SSLClient` and calls its `set_ca_cert_store` method with the provided `ca_cert_store`.
    - If SSL is not enabled, it directly calls the `set_ca_cert_store` method on the `cli_` pointer.
- **Output**: This function does not return a value; it modifies the state of the client by setting the CA certificate store.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::load\_ca\_cert\_store<!-- {{#callable:Client::load_ca_cert_store}} -->
Loads a CA certificate store from a given certificate string and its size.
- **Inputs**:
    - `ca_cert`: A pointer to a character array containing the CA certificate data.
    - `size`: The size of the CA certificate data in bytes.
- **Control Flow**:
    - Calls the `create_ca_cert_store` method on the `cli_` member, passing the `ca_cert` and `size` as arguments.
    - The result of `create_ca_cert_store` is then passed to the [`set_ca_cert_store`](#ClientImplset_ca_cert_store) method.
- **Output**: This function does not return a value; it sets the CA certificate store for the client.
- **Functions called**:
    - [`ClientImpl::set_ca_cert_store`](#ClientImplset_ca_cert_store)
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::get\_openssl\_verify\_result<!-- {{#callable:Client::get_openssl_verify_result}} -->
Retrieves the OpenSSL verification result if SSL is enabled.
- **Inputs**:
    - `none`: This function does not take any input arguments.
- **Control Flow**:
    - The function first checks if SSL is enabled by evaluating the `is_ssl_` member variable.
    - If SSL is enabled, it casts the `cli_` pointer to `SSLClient` and calls its `get_openssl_verify_result()` method.
    - If SSL is not enabled, it returns -1, indicating that no valid verification result is available.
- **Output**: Returns the OpenSSL verification result as a long integer, or -1 if SSL is not enabled.
- **See also**: [`Client`](#Client)  (Data Structure)


---
#### Client::ssl\_context<!-- {{#callable:Client::ssl_context}} -->
Returns the SSL context if the client is configured for SSL.
- **Inputs**: None
- **Control Flow**:
    - Checks if the `is_ssl_` member variable is true.
    - If true, it retrieves the SSL context from the `cli_` member, which is cast to `SSLClient`.
    - If false, it returns a null pointer.
- **Output**: Returns a pointer to an `SSL_CTX` object if SSL is enabled, otherwise returns nullptr.
- **See also**: [`Client`](#Client)  (Data Structure)



---
### SSLServer<!-- {{#data_structure:SSLServer}} -->
- **Type**: `class`
- **Members**:
    - `ctx_`: A pointer to the SSL context (SSL_CTX) used for managing SSL/TLS operations.
    - `ctx_mutex_`: A mutex used to ensure thread-safe access to the SSL context.
- **Description**: The `SSLServer` class is a specialized server class that extends the `Server` class to provide SSL/TLS capabilities. It manages SSL/TLS operations using an SSL context (`SSL_CTX`) and ensures thread-safe access to this context with a mutex. The class offers multiple constructors for initializing the server with certificate paths, certificate objects, or a custom SSL context setup callback. It includes methods to check the server's validity, retrieve the SSL context, and update certificates. The class is designed to handle SSL/TLS connections securely and efficiently.
- **Member Functions**:
    - [`SSLServer::SSLServer`](#SSLServerSSLServer)
    - [`SSLServer::SSLServer`](#SSLServerSSLServer)
    - [`SSLServer::SSLServer`](#SSLServerSSLServer)
    - [`SSLServer::~SSLServer`](#SSLServerSSLServer)
    - [`SSLServer::is_valid`](#SSLServeris_valid)
    - [`SSLServer::ssl_context`](#SSLServerssl_context)
    - [`SSLServer::update_certs`](#SSLServerupdate_certs)
    - [`SSLServer::process_and_close_socket`](#SSLServerprocess_and_close_socket)
- **Inherits From**:
    - [`Server::Server`](#ServerServer)

**Methods**

---
#### SSLServer::SSLServer<!-- {{#callable:SSLServer::SSLServer}} -->
The `SSLServer` constructor initializes an SSL server context with specified certificate and key paths, and optionally sets up client certificate verification.
- **Inputs**:
    - `cert_path`: Path to the server's certificate file.
    - `private_key_path`: Path to the server's private key file.
    - `client_ca_cert_file_path`: Optional path to a file containing client CA certificates.
    - `client_ca_cert_dir_path`: Optional path to a directory containing client CA certificates.
    - `private_key_password`: Optional password for the server's private key.
- **Control Flow**:
    - Create a new SSL context using `TLS_server_method()` and assign it to `ctx_`.
    - If the context is successfully created, set options to disable compression and session resumption on renegotiation.
    - Set the minimum protocol version to TLS 1.2.
    - If a private key password is provided and not empty, set it as the default password callback user data for the context.
    - Attempt to load the server's certificate and private key from the specified paths; if unsuccessful, free the context and set `ctx_` to `nullptr`.
    - If client CA certificate paths are provided, load them and set the context to verify peer certificates.
- **Output**: The constructor initializes the `ctx_` member variable with a configured SSL context or sets it to `nullptr` if initialization fails.
- **See also**: [`SSLServer`](#SSLServer)  (Data Structure)


---
#### SSLServer::SSLServer<!-- {{#callable:SSLServer::SSLServer}} -->
The `SSLServer` constructor initializes an SSL server context with specified certificate, private key, and optional client CA certificate store, setting various SSL options and protocols.
- **Inputs**:
    - `cert`: A pointer to an X509 certificate used for the server's SSL context.
    - `private_key`: A pointer to an EVP_PKEY structure representing the server's private key.
    - `client_ca_cert_store`: An optional pointer to an X509_STORE containing client CA certificates for verifying client certificates.
- **Control Flow**:
    - Create a new SSL context using `TLS_server_method()` and assign it to `ctx_`.
    - Check if `ctx_` is successfully created; if not, the constructor does nothing further.
    - Set SSL context options to disable compression and session resumption on renegotiation.
    - Set the minimum protocol version to TLS 1.2.
    - Attempt to use the provided certificate and private key in the SSL context; if unsuccessful, free the context and set `ctx_` to `nullptr`.
    - If a client CA certificate store is provided, set it in the SSL context and configure the context to verify peer certificates and fail if no peer certificate is provided.
- **Output**: The constructor does not return a value, but it initializes the `ctx_` member variable of the `SSLServer` class with a configured SSL context or sets it to `nullptr` if initialization fails.
- **See also**: [`SSLServer`](#SSLServer)  (Data Structure)


---
#### SSLServer::SSLServer<!-- {{#callable:SSLServer::SSLServer}} -->
The SSLServer constructor initializes an SSL context using a provided callback function to configure it, and cleans up if the configuration fails.
- **Inputs**:
    - `setup_ssl_ctx_callback`: A callback function that takes a reference to an SSL_CTX object and returns a boolean indicating whether the SSL context was successfully set up.
- **Control Flow**:
    - Create a new SSL context using TLS_method().
    - Check if the SSL context was successfully created.
    - Invoke the setup_ssl_ctx_callback with the created SSL context.
    - If the callback returns false, indicating failure to set up the SSL context, free the SSL context and set the context pointer to nullptr.
- **Output**: The function does not return a value, but it initializes the ctx_ member variable of the SSLServer class to a valid SSL context or nullptr if setup fails.
- **See also**: [`SSLServer`](#SSLServer)  (Data Structure)


---
#### SSLServer::\~SSLServer<!-- {{#callable:SSLServer::~SSLServer}} -->
The destructor for the SSLServer class releases the SSL context if it has been initialized.
- **Inputs**: None
- **Control Flow**:
    - Check if the SSL context (ctx_) is not null.
    - If ctx_ is not null, call SSL_CTX_free(ctx_) to release the SSL context.
- **Output**: This destructor does not return any value.
- **See also**: [`SSLServer`](#SSLServer)  (Data Structure)


---
#### SSLServer::is\_valid<!-- {{#callable:SSLServer::is_valid}} -->
The `is_valid` function checks if the SSL context (`ctx_`) is initialized and valid.
- **Inputs**: None
- **Control Flow**:
    - The function returns the value of `ctx_`, which is a pointer to an `SSL_CTX` object.
    - If `ctx_` is non-null, the function returns `true`, indicating the SSL context is valid.
    - If `ctx_` is null, the function returns `false`, indicating the SSL context is not valid.
- **Output**: A boolean value indicating whether the SSL context (`ctx_`) is valid (true) or not (false).
- **See also**: [`SSLServer`](#SSLServer)  (Data Structure)


---
#### SSLServer::ssl\_context<!-- {{#callable:SSLServer::ssl_context}} -->
The `ssl_context` method returns the SSL context associated with the `SSLServer` instance.
- **Inputs**: None
- **Control Flow**:
    - The method is defined as `inline`, indicating it is a small function that is expanded in place where it is called.
    - It is a constant method, meaning it does not modify the state of the `SSLServer` instance.
    - The method simply returns the private member `ctx_`, which is a pointer to an `SSL_CTX` object.
- **Output**: A pointer to an `SSL_CTX` object representing the SSL context of the server.
- **See also**: [`SSLServer`](#SSLServer)  (Data Structure)


---
#### SSLServer::update\_certs<!-- {{#callable:SSLServer::update_certs}} -->
The `update_certs` function updates the SSL context with a new certificate, private key, and optionally a client CA certificate store, ensuring thread safety with a mutex lock.
- **Inputs**:
    - `cert`: A pointer to an X509 certificate to be used by the SSL context.
    - `private_key`: A pointer to an EVP_PKEY structure representing the private key associated with the certificate.
    - `client_ca_cert_store`: A pointer to an X509_STORE structure representing the client CA certificate store, which can be null.
- **Control Flow**:
    - Acquire a lock on the mutex `ctx_mutex_` to ensure thread safety while updating the SSL context.
    - Use `SSL_CTX_use_certificate` to set the provided certificate (`cert`) in the SSL context (`ctx_`).
    - Use `SSL_CTX_use_PrivateKey` to set the provided private key (`private_key`) in the SSL context (`ctx_`).
    - Check if `client_ca_cert_store` is not null; if so, use `SSL_CTX_set_cert_store` to set the client CA certificate store in the SSL context (`ctx_`).
- **Output**: The function does not return any value.
- **See also**: [`SSLServer`](#SSLServer)  (Data Structure)


---
#### SSLServer::process\_and\_close\_socket<!-- {{#callable:SSLServer::process_and_close_socket}} -->
The `process_and_close_socket` function handles SSL connection processing for a given socket, including establishing the connection, processing requests, and closing the socket.
- **Inputs**:
    - `sock`: A socket descriptor representing the connection to be processed.
- **Control Flow**:
    - Create a new SSL object for the socket using `detail::ssl_new` with a non-blocking SSL accept operation.
    - Initialize a boolean `ret` to false to track the success of the operation.
    - If the SSL object is successfully created, retrieve the remote and local IP addresses and ports using `detail::get_remote_ip_and_port` and `detail::get_local_ip_and_port`.
    - Process the server socket with SSL using `detail::process_server_socket_ssl`, passing in various timeout and keep-alive parameters, and a lambda function to process requests.
    - Set `ret` to the result of `detail::process_server_socket_ssl`.
    - Delete the SSL object using `detail::ssl_delete`, with a graceful shutdown if `ret` is true.
    - Shutdown and close the socket using `detail::shutdown_socket` and `detail::close_socket`.
- **Output**: A boolean value indicating whether the socket processing was successful.
- **Functions called**:
    - [`Server::process_request`](#Serverprocess_request)
- **See also**: [`SSLServer`](#SSLServer)  (Data Structure)



---
### SSLClient<!-- {{#data_structure:SSLClient}} -->
- **Type**: `class`
- **Members**:
    - `ctx_`: A pointer to the SSL context used for managing SSL/TLS connections.
    - `ctx_mutex_`: A mutex to ensure thread-safe operations on the SSL context.
    - `initialize_cert_`: A flag to ensure the certificate initialization is done only once.
    - `host_components_`: A vector storing components of the host for SSL verification purposes.
    - `verify_result_`: Stores the result of the SSL certificate verification process.
- **Description**: The `SSLClient` class is a specialized client implementation that extends `ClientImpl` to provide SSL/TLS support for secure network communications. It manages SSL contexts, handles certificate verification, and ensures secure connections through various SSL/TLS protocols. The class includes mechanisms for loading certificates, verifying host identities, and managing SSL connections with thread safety. It is designed to be used in environments where secure communication is critical, leveraging OpenSSL for cryptographic operations.
- **Member Functions**:
    - [`SSLClient::SSLClient`](#SSLClientSSLClient)
    - [`SSLClient::SSLClient`](#SSLClientSSLClient)
    - [`SSLClient::SSLClient`](#SSLClientSSLClient)
    - [`SSLClient::SSLClient`](#SSLClientSSLClient)
    - [`SSLClient::~SSLClient`](#SSLClientSSLClient)
    - [`SSLClient::is_valid`](#SSLClientis_valid)
    - [`SSLClient::set_ca_cert_store`](#SSLClientset_ca_cert_store)
    - [`SSLClient::load_ca_cert_store`](#SSLClientload_ca_cert_store)
    - [`SSLClient::get_openssl_verify_result`](#SSLClientget_openssl_verify_result)
    - [`SSLClient::ssl_context`](#SSLClientssl_context)
    - [`SSLClient::create_and_connect_socket`](#SSLClientcreate_and_connect_socket)
    - [`SSLClient::connect_with_proxy`](#SSLClientconnect_with_proxy)
    - [`SSLClient::load_certs`](#SSLClientload_certs)
    - [`SSLClient::initialize_ssl`](#SSLClientinitialize_ssl)
    - [`SSLClient::shutdown_ssl`](#SSLClientshutdown_ssl)
    - [`SSLClient::shutdown_ssl_impl`](#SSLClientshutdown_ssl_impl)
    - [`SSLClient::process_socket`](#SSLClientprocess_socket)
    - [`SSLClient::is_ssl`](#SSLClientis_ssl)
    - [`SSLClient::verify_host`](#SSLClientverify_host)
    - [`SSLClient::verify_host_with_subject_alt_name`](#SSLClientverify_host_with_subject_alt_name)
    - [`SSLClient::verify_host_with_common_name`](#SSLClientverify_host_with_common_name)
    - [`SSLClient::check_host_name`](#SSLClientcheck_host_name)
- **Inherits From**:
    - [`ClientImpl::ClientImpl`](#ClientImplClientImpl)

**Methods**

---
#### SSLClient::SSLClient<!-- {{#callable:SSLClient::SSLClient}} -->
The `SSLClient` constructor initializes an SSLClient object with a specified host, defaulting to port 443 and empty client certificate and key paths.
- **Inputs**:
    - `host`: A string representing the hostname of the server to connect to.
- **Control Flow**:
    - The constructor is called with a single argument, `host`.
    - It delegates the initialization to another overloaded constructor of `SSLClient` with default values for port (443), client certificate path, and client key path.
- **Output**: An instance of the `SSLClient` class is created and initialized with the specified host and default parameters for port and certificate paths.
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::SSLClient<!-- {{#callable:SSLClient::SSLClient}} -->
The `SSLClient` constructor initializes an `SSLClient` object with a specified host and port, delegating to another constructor with additional default parameters for client certificate and key paths.
- **Inputs**:
    - `host`: A string representing the hostname of the server to connect to.
    - `port`: An integer representing the port number to connect to on the server.
- **Control Flow**:
    - The constructor is called with a host and port as arguments.
    - It delegates the initialization to another `SSLClient` constructor, passing the host, port, and default empty strings for client certificate and key paths.
- **Output**: An `SSLClient` object is initialized with the specified host and port, using default values for client certificate and key paths.
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::SSLClient<!-- {{#callable:SSLClient::SSLClient}} -->
The `SSLClient` constructor initializes an SSL client with specified host, port, and optional client certificate and key paths, setting up the SSL context and handling certificate loading.
- **Inputs**:
    - `host`: A string representing the hostname of the server to connect to.
    - `port`: An integer representing the port number to connect to.
    - `client_cert_path`: A string representing the file path to the client's certificate.
    - `client_key_path`: A string representing the file path to the client's private key.
    - `private_key_password`: A string representing the password for the client's private key, if any.
- **Control Flow**:
    - The constructor initializes the base class `ClientImpl` with the provided host, port, client certificate path, and client key path.
    - An SSL context (`ctx_`) is created using `TLS_client_method()`.
    - The minimum protocol version for the SSL context is set to TLS 1.2.
    - The host string is split into components using a custom delimiter function and stored in `host_components_`.
    - If both client certificate and key paths are provided, the private key password is set as a callback user data if it is not empty.
    - The client certificate and private key are loaded into the SSL context; if loading fails, the SSL context is freed and set to `nullptr`.
- **Output**: The constructor does not return a value but initializes the `SSLClient` object with an SSL context and host components, ready for SSL connections.
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::SSLClient<!-- {{#callable:SSLClient::SSLClient}} -->
The `SSLClient` constructor initializes an SSL client with a specified host, port, client certificate, client key, and optional private key password, setting up the SSL context and handling certificate and key usage.
- **Inputs**:
    - `host`: A string representing the hostname of the server to connect to.
    - `port`: An integer representing the port number to connect to on the server.
    - `client_cert`: A pointer to an X509 structure representing the client's certificate.
    - `client_key`: A pointer to an EVP_PKEY structure representing the client's private key.
    - `private_key_password`: A string representing the password for the client's private key, if any.
- **Control Flow**:
    - The constructor initializes the base class `ClientImpl` with the provided host and port.
    - An SSL context is created using `TLS_client_method()`.
    - The host string is split into components using a delimiter '.' and stored in `host_components_`.
    - If both `client_cert` and `client_key` are provided, the function checks if a private key password is provided and sets it as the default password callback user data for the SSL context.
    - The function attempts to use the provided client certificate and private key with the SSL context.
    - If either the certificate or key cannot be used, the SSL context is freed and set to `nullptr`.
- **Output**: The function does not return a value, but it initializes the SSL context and sets up the client certificate and key for the SSL client.
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::\~SSLClient<!-- {{#callable:SSLClient::~SSLClient}} -->
The destructor for the `SSLClient` class ensures proper cleanup of SSL resources by freeing the SSL context and shutting down the SSL connection.
- **Inputs**: None
- **Control Flow**:
    - Check if `ctx_` is not null, and if so, free the SSL context using `SSL_CTX_free(ctx_)`.
    - Call `shutdown_ssl_impl(socket_, true)` to ensure the SSL connection is properly shut down, preventing resource leaks.
- **Output**: This destructor does not return any value as it is responsible for cleanup operations when an `SSLClient` object is destroyed.
- **Functions called**:
    - [`SSLClient::shutdown_ssl_impl`](#SSLClientshutdown_ssl_impl)
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::is\_valid<!-- {{#callable:SSLClient::is_valid}} -->
The `is_valid` function checks if the SSL context (`ctx_`) is initialized and valid for the `SSLClient` instance.
- **Inputs**: None
- **Control Flow**:
    - The function returns the value of the `ctx_` member variable, which is a pointer to an `SSL_CTX` object.
    - If `ctx_` is non-null, the function returns `true`, indicating the SSL context is valid; otherwise, it returns `false`.
- **Output**: A boolean value indicating whether the SSL context (`ctx_`) is valid.
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::set\_ca\_cert\_store<!-- {{#callable:SSLClient::set_ca_cert_store}} -->
The `set_ca_cert_store` function sets a new CA certificate store for the SSL context if it differs from the current one, or frees the store if the context is not initialized.
- **Inputs**:
    - `ca_cert_store`: A pointer to an X509_STORE object representing the CA certificate store to be set.
- **Control Flow**:
    - Check if the `ca_cert_store` is not null.
    - If `ctx_` (the SSL context) is initialized, check if the current certificate store is different from `ca_cert_store`.
    - If the current certificate store is different, set the new `ca_cert_store` using `SSL_CTX_set_cert_store`.
    - If `ctx_` is not initialized, free the `ca_cert_store` using `X509_STORE_free`.
- **Output**: The function does not return a value; it modifies the SSL context's certificate store or frees the provided store.
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::load\_ca\_cert\_store<!-- {{#callable:SSLClient::load_ca_cert_store}} -->
The `load_ca_cert_store` function loads a CA certificate into the SSL client's certificate store.
- **Inputs**:
    - `ca_cert`: A pointer to a character array containing the CA certificate data.
    - `size`: The size of the CA certificate data in bytes.
- **Control Flow**:
    - The function calls `ClientImpl::create_ca_cert_store` with `ca_cert` and `size` to create a CA certificate store.
    - The resulting CA certificate store is then set using the [`set_ca_cert_store`](#ClientImplset_ca_cert_store) method.
- **Output**: This function does not return a value; it modifies the internal state of the `SSLClient` by setting its CA certificate store.
- **Functions called**:
    - [`ClientImpl::set_ca_cert_store`](#ClientImplset_ca_cert_store)
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::get\_openssl\_verify\_result<!-- {{#callable:SSLClient::get_openssl_verify_result}} -->
The `get_openssl_verify_result` function returns the result of the OpenSSL verification process for an SSL connection.
- **Inputs**: None
- **Control Flow**:
    - The function is a simple inline method of the `SSLClient` class.
    - It directly returns the value of the private member variable `verify_result_`.
- **Output**: The function returns a `long` integer representing the OpenSSL verification result.
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::ssl\_context<!-- {{#callable:SSLClient::ssl_context}} -->
The `ssl_context` method returns the SSL context associated with the `SSLClient` instance.
- **Inputs**: None
- **Control Flow**:
    - The method is defined as `inline`, indicating it is a small function that is expanded in place where it is called.
    - It is a constant method, meaning it does not modify the state of the `SSLClient` object.
    - The method simply returns the private member `ctx_`, which is a pointer to an `SSL_CTX` object.
- **Output**: A pointer to an `SSL_CTX` object representing the SSL context.
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::create\_and\_connect\_socket<!-- {{#callable:SSLClient::create_and_connect_socket}} -->
The `create_and_connect_socket` function attempts to create and connect a socket if the SSLClient is in a valid state.
- **Inputs**:
    - `socket`: A reference to a Socket object that will be created and connected.
    - `error`: A reference to an Error object that will store any error information if the operation fails.
- **Control Flow**:
    - Check if the SSLClient instance is in a valid state using the `is_valid()` method.
    - If valid, call the `create_and_connect_socket` method from the `ClientImpl` base class, passing the `socket` and `error` references.
    - Return the result of the `ClientImpl::create_and_connect_socket` call.
- **Output**: A boolean value indicating whether the socket was successfully created and connected.
- **Functions called**:
    - [`gzip_decompressor::is_valid`](#gzip_decompressoris_valid)
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::connect\_with\_proxy<!-- {{#callable:SSLClient::connect_with_proxy}} -->
The `connect_with_proxy` function attempts to establish a connection through a proxy server, handling proxy authentication if required, and returns whether the connection was successful.
- **Inputs**:
    - `socket`: A reference to a `Socket` object representing the network socket to be used for the connection.
    - `start_time`: A `std::chrono::time_point` representing the start time of the connection attempt, used for timeout calculations.
    - `res`: A reference to a [`Response`](#ResponseResponse) object where the response from the proxy server will be stored.
    - `success`: A reference to a boolean that will be set to true if the connection is successful, or false otherwise.
    - `error`: A reference to an `Error` object that will be set to indicate the type of error if the connection fails.
- **Control Flow**:
    - Initialize `success` to true and create a [`Response`](#ResponseResponse) object `proxy_res` to store the proxy server's response.
    - Attempt to process the client socket with a CONNECT request to the proxy server using the `process_client_socket` function.
    - If the initial connection attempt fails, shut down and close the socket, set `success` to false, and return false.
    - If the proxy server requires authentication (status 407), check for stored proxy credentials and attempt to authenticate using digest authentication.
    - If authentication is required and fails, shut down and close the socket, set `success` to false, and return false.
    - If the proxy server's response status is not 200 (OK), set the error to `ProxyConnection`, move the proxy response to `res`, shut down and close the socket, and return false.
    - If all checks pass and the connection is successful, return true.
- **Output**: Returns a boolean indicating whether the connection through the proxy was successful, with additional output through the `res`, `success`, and `error` parameters.
- **Functions called**:
    - [`Server::process_request`](#Serverprocess_request)
    - [`ClientImpl::shutdown_ssl`](#ClientImplshutdown_ssl)
    - [`shutdown_socket`](#shutdown_socket)
    - [`close_socket`](#close_socket)
    - [`Response::Response`](#ResponseResponse)
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::load\_certs<!-- {{#callable:SSLClient::load_certs}} -->
The `load_certs` function attempts to load SSL certificates for the `SSLClient` either from specified file paths or system defaults, ensuring thread safety and one-time initialization.
- **Inputs**: None
- **Control Flow**:
    - The function starts by setting a boolean variable `ret` to `true`, indicating success by default.
    - It uses `std::call_once` with a lambda to ensure the certificate loading logic is executed only once, even in multithreaded scenarios.
    - Within the lambda, a `std::lock_guard` is used to lock `ctx_mutex_` for thread safety while accessing shared resources.
    - The function checks if `ca_cert_file_path_` is not empty; if so, it attempts to load certificates from the file path using `SSL_CTX_load_verify_locations`. If this fails, `ret` is set to `false`.
    - If `ca_cert_file_path_` is empty but `ca_cert_dir_path_` is not, it attempts to load certificates from the directory path, similarly setting `ret` to `false` on failure.
    - If neither path is provided, it attempts to load system certificates based on the operating system, using platform-specific functions for Windows or macOS, or defaults to `SSL_CTX_set_default_verify_paths` if these fail.
    - The function returns the value of `ret`, indicating whether the certificate loading was successful.
- **Output**: A boolean value indicating whether the certificate loading was successful (`true`) or not (`false`).
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::initialize\_ssl<!-- {{#callable:SSLClient::initialize_ssl}} -->
The `initialize_ssl` function sets up an SSL connection for a given socket, handling certificate verification and error management.
- **Inputs**:
    - `socket`: A reference to a `Socket` object that represents the network socket to be secured with SSL.
    - `error`: A reference to an `Error` object that will be set if any SSL initialization errors occur.
- **Control Flow**:
    - Call `detail::ssl_new` to create a new SSL object with the provided socket, context, and mutex, and define two lambda functions for SSL setup and hostname setting.
    - In the first lambda, check if server certificate verification is enabled; if so, attempt to load certificates and set verification options.
    - Attempt to establish a non-blocking SSL connection using `detail::ssl_connect_or_accept_nonblocking`; if it fails, set the error to `Error::SSLConnection` and return false.
    - If server certificate verification is enabled, check the verification status using a custom verifier if available, or default to OpenSSL's verification result.
    - If verification fails, set the error to `Error::SSLServerVerification` and return false.
    - If hostname verification is enabled, verify the server's hostname against the certificate; if it fails, set the error to `Error::SSLServerHostnameVerification` and return false.
    - In the second lambda, set the TLS SNI (Server Name Indication) using the host name, with a conditional compilation for BoringSSL.
    - If the SSL object is successfully created, assign it to the socket's SSL member and return true.
    - If SSL object creation fails, shut down and close the socket, then return false.
- **Output**: A boolean value indicating whether the SSL initialization was successful (true) or not (false).
- **Functions called**:
    - [`SSLClient::load_certs`](#SSLClientload_certs)
    - [`SSLClient::verify_host`](#SSLClientverify_host)
    - [`shutdown_socket`](#shutdown_socket)
    - [`close_socket`](#close_socket)
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::shutdown\_ssl<!-- {{#callable:SSLClient::shutdown_ssl}} -->
The `shutdown_ssl` function initiates the shutdown process of an SSL connection on a given socket, with an option for graceful shutdown.
- **Inputs**:
    - `socket`: A reference to a `Socket` object representing the network connection to be shut down.
    - `shutdown_gracefully`: A boolean flag indicating whether the shutdown should be performed gracefully.
- **Control Flow**:
    - The function calls [`shutdown_ssl_impl`](#SSLClientshutdown_ssl_impl) with the provided `socket` and `shutdown_gracefully` arguments to perform the actual shutdown operation.
- **Output**: This function does not return any value.
- **Functions called**:
    - [`SSLClient::shutdown_ssl_impl`](#SSLClientshutdown_ssl_impl)
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::shutdown\_ssl\_impl<!-- {{#callable:SSLClient::shutdown_ssl_impl}} -->
The `shutdown_ssl_impl` function safely shuts down an SSL connection associated with a given socket, optionally performing a graceful shutdown.
- **Inputs**:
    - `socket`: A reference to a `Socket` object that contains the SSL connection to be shut down.
    - `shutdown_gracefully`: A boolean flag indicating whether the SSL connection should be shut down gracefully.
- **Control Flow**:
    - Check if the socket is invalid (i.e., `socket.sock` is `INVALID_SOCKET`); if so, assert that `socket.ssl` is `nullptr` and return immediately.
    - If `socket.ssl` is not `nullptr`, call `detail::ssl_delete` to delete the SSL connection, passing the context mutex, the SSL object, the socket, and the `shutdown_gracefully` flag.
    - Set `socket.ssl` to `nullptr` to indicate that the SSL connection has been shut down.
    - Assert that `socket.ssl` is `nullptr` to ensure the SSL connection has been properly cleaned up.
- **Output**: The function does not return any value; it modifies the state of the `Socket` object by shutting down its SSL connection.
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::process\_socket<!-- {{#callable:SSLClient::process_socket}} -->
The `process_socket` function processes an SSL socket connection using a callback function and manages timeouts.
- **Inputs**:
    - `socket`: A `Socket` object that contains the SSL connection to be processed.
    - `start_time`: A `std::chrono::time_point` representing the start time of the operation, used for timeout calculations.
    - `callback`: A `std::function` that takes a `Stream` reference and returns a `bool`, used to process the stream data.
- **Control Flow**:
    - The function asserts that the socket has an SSL connection.
    - It calls the `detail::process_client_socket_ssl` function, passing the SSL connection, socket, timeout settings, start time, and the callback function.
    - The `detail::process_client_socket_ssl` function handles the actual processing of the SSL socket, including managing read and write timeouts and invoking the callback.
- **Output**: Returns a `bool` indicating the success or failure of processing the SSL socket.
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::is\_ssl<!-- {{#callable:SSLClient::is_ssl}} -->
The `is_ssl` function in the `SSLClient` class always returns `true`, indicating that the client is using SSL.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as an inline method within the `SSLClient` class.
    - It overrides a virtual method from the `ClientImpl` base class.
    - The function simply returns the boolean value `true`.
- **Output**: A boolean value `true`, indicating that the client is using SSL.
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::verify\_host<!-- {{#callable:SSLClient::verify_host}} -->
The `verify_host` function checks the validity of a server's certificate by verifying its host identity using either the subject alternative name or the common name.
- **Inputs**:
    - `server_cert`: A pointer to an X509 structure representing the server's certificate to be verified.
- **Control Flow**:
    - The function first attempts to verify the host using the subject alternative name by calling `verify_host_with_subject_alt_name(server_cert)`.
    - If the verification using the subject alternative name fails, it then attempts to verify the host using the common name by calling `verify_host_with_common_name(server_cert)`.
    - The function returns true if either of the verification methods succeeds, otherwise it returns false.
- **Output**: A boolean value indicating whether the server's certificate is verified successfully against the host identity.
- **Functions called**:
    - [`SSLClient::verify_host_with_subject_alt_name`](#SSLClientverify_host_with_subject_alt_name)
    - [`SSLClient::verify_host_with_common_name`](#SSLClientverify_host_with_common_name)
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::verify\_host\_with\_subject\_alt\_name<!-- {{#callable:SSLClient::verify_host_with_subject_alt_name}} -->
The `verify_host_with_subject_alt_name` function checks if the host matches any of the subject alternative names in the server's SSL certificate.
- **Inputs**:
    - `server_cert`: A pointer to an X509 structure representing the server's SSL certificate.
- **Control Flow**:
    - Initialize the return value `ret` to false and set the default type to `GEN_DNS`.
    - Check if the host is an IPv6 or IPv4 address using `inet_pton`, and set `type` to `GEN_IPADD` and `addr_len` accordingly.
    - Retrieve the subject alternative names from the server certificate using `X509_get_ext_d2i`.
    - If alternative names are present, iterate over them to check if any match the host.
    - For each alternative name, if its type matches the determined `type`, compare it with the host name or IP address.
    - If a match is found, set `dsn_matched` or `ip_matched` to true.
    - If either `dsn_matched` or `ip_matched` is true, set `ret` to true.
    - Free the memory allocated for the alternative names and return the result `ret`.
- **Output**: A boolean value indicating whether the host matches any of the subject alternative names in the server's certificate.
- **Functions called**:
    - [`SSLClient::check_host_name`](#SSLClientcheck_host_name)
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::verify\_host\_with\_common\_name<!-- {{#callable:SSLClient::verify_host_with_common_name}} -->
The `verify_host_with_common_name` function checks if the common name in the server's SSL certificate matches the expected host name.
- **Inputs**:
    - `server_cert`: A pointer to an X509 structure representing the server's SSL certificate.
- **Control Flow**:
    - Retrieve the subject name from the server's certificate using `X509_get_subject_name`.
    - Check if the subject name is not null.
    - If the subject name is valid, extract the common name using `X509_NAME_get_text_by_NID` with the NID_commonName identifier.
    - Check if the common name was successfully extracted (name_len != -1).
    - If successful, call [`check_host_name`](#SSLClientcheck_host_name) with the extracted common name and its length to verify the host name.
    - Return false if any step fails or if the common name does not match.
- **Output**: A boolean value indicating whether the common name in the server's certificate matches the expected host name.
- **Functions called**:
    - [`SSLClient::check_host_name`](#SSLClientcheck_host_name)
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)


---
#### SSLClient::check\_host\_name<!-- {{#callable:SSLClient::check_host_name}} -->
The `check_host_name` function verifies if a given host name matches a specified pattern, supporting exact and wildcard matches.
- **Inputs**:
    - `pattern`: A C-style string representing the host name pattern to match against.
    - `pattern_len`: The length of the pattern string.
- **Control Flow**:
    - Check if the host name size matches the pattern length and if the host name is exactly equal to the pattern; return true if both conditions are met.
    - Split the pattern into components using '.' as a delimiter and store them in a vector.
    - Compare the size of the host components with the pattern components; return false if they differ.
    - Iterate over the host components and corresponding pattern components, checking for exact matches or wildcard matches (where the pattern component is '*').
    - For partial wildcard matches, check if the pattern component ends with '*' and matches the beginning of the host component; return false if no match is found.
    - Return true if all components match according to the rules.
- **Output**: A boolean value indicating whether the host name matches the pattern.
- **See also**: [`SSLClient`](#SSLClient)  (Data Structure)



---
### FileStat<!-- {{#data_structure:detail::FileStat}} -->
- **Type**: `struct`
- **Members**:
    - `st_`: A platform-dependent structure that holds file status information.
    - `ret_`: An integer initialized to -1, likely used to store the return status of file operations.
- **Description**: The `FileStat` struct is designed to encapsulate file status information, providing methods to determine if a given path is a file or a directory. It uses a platform-dependent structure (`_stat` on Windows and `stat` on other systems) to store the file status details, and an integer `ret_` to likely store the result of file operations. The struct is initialized with a file path and provides methods to check the type of the file system object at that path.


---
### EncodingType<!-- {{#data_structure:detail::EncodingType}} -->
- **Type**: `enum`
- **Members**:
    - `None`: Represents no encoding, with a value of 0.
    - `Gzip`: Represents Gzip encoding.
    - `Brotli`: Represents Brotli encoding.
    - `Zstd`: Represents Zstandard (Zstd) encoding.
- **Description**: The `EncodingType` enum class defines a set of constants representing different types of data compression encodings. It includes options for no encoding (`None`), as well as three popular compression algorithms: Gzip, Brotli, and Zstandard (Zstd). This enum can be used to specify the encoding type in applications that handle data compression and decompression.


---
### BufferStream<!-- {{#data_structure:detail::BufferStream}} -->
- **Type**: `class`
- **Members**:
    - `buffer`: A private member variable that stores the data in a string format.
    - `position`: A private member variable that tracks the current position in the buffer, initialized to 0.
- **Description**: The `BufferStream` class is a final class that inherits from the `Stream` class, providing an implementation for handling buffered data streams. It includes methods for reading from and writing to the buffer, as well as retrieving network-related information such as IP addresses and ports. The class maintains a private buffer and a position index to manage the current read/write location within the buffer.
- **Member Functions**:
    - [`detail::BufferStream::BufferStream`](#BufferStreamBufferStream)
    - [`detail::BufferStream::~BufferStream`](#BufferStreamBufferStream)
    - [`detail::BufferStream::is_readable`](#BufferStreamis_readable)
    - [`detail::BufferStream::wait_readable`](#BufferStreamwait_readable)
    - [`detail::BufferStream::wait_writable`](#BufferStreamwait_writable)
    - [`detail::BufferStream::read`](#BufferStreamread)
    - [`detail::BufferStream::write`](#BufferStreamwrite)
    - [`detail::BufferStream::get_remote_ip_and_port`](#BufferStreamget_remote_ip_and_port)
    - [`detail::BufferStream::get_local_ip_and_port`](#BufferStreamget_local_ip_and_port)
    - [`detail::BufferStream::socket`](#BufferStreamsocket)
    - [`detail::BufferStream::duration`](#BufferStreamduration)
    - [`detail::BufferStream::get_buffer`](#BufferStreamget_buffer)
- **Inherits From**:
    - [`Stream`](#Stream)

**Methods**

---
#### BufferStream::BufferStream<!-- {{#callable:detail::BufferStream::BufferStream}} -->
The `BufferStream` class is a final implementation of the `Stream` interface, providing buffered read and write operations along with network-related functionalities.
- **Inputs**: None
- **Control Flow**:
    - The constructor `BufferStream()` is defined as default, meaning it initializes the object with default values.
    - The destructor `~BufferStream()` is also defined as default, ensuring proper cleanup when the object is destroyed.
    - The class overrides several methods from the `Stream` interface, including `is_readable`, `wait_readable`, `wait_writable`, `read`, `write`, `get_remote_ip_and_port`, `get_local_ip_and_port`, `socket`, and `duration`, each providing specific functionality related to stream operations.
    - The class maintains a private buffer as a `std::string` and a position index to track the current read/write position within the buffer.
    - The method `get_buffer()` returns a constant reference to the internal buffer, allowing external access to the buffer's content without modification.
- **Output**: The class does not produce a direct output but provides various methods to interact with the buffer and network-related functionalities.
- **See also**: [`detail::BufferStream`](#detailBufferStream)  (Data Structure)


---
#### BufferStream::\~BufferStream<!-- {{#callable:detail::BufferStream::~BufferStream}} -->
The `~BufferStream` function is the destructor for the `BufferStream` class, ensuring proper cleanup when an instance is destroyed.
- **Inputs**: None
- **Control Flow**:
    - The destructor `~BufferStream` is defined with the `override` keyword, indicating it overrides a virtual destructor from the base class `Stream`.
    - The destructor is marked as `default`, meaning it uses the compiler-generated default implementation for cleanup.
- **Output**: There is no explicit output from the destructor, as it is responsible for cleanup and resource deallocation when a `BufferStream` object is destroyed.
- **See also**: [`detail::BufferStream`](#detailBufferStream)  (Data Structure)


---
#### BufferStream::is\_readable<!-- {{#callable:detail::BufferStream::is_readable}} -->
The `is_readable` function in the `BufferStream` class always returns `true`, indicating that the buffer is always considered readable.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as `inline`, suggesting it is intended to be expanded in place where it is called, for performance reasons.
    - The function simply returns the boolean value `true`.
- **Output**: The function returns a boolean value `true`, indicating the buffer is always readable.
- **See also**: [`detail::BufferStream`](#detailBufferStream)  (Data Structure)


---
#### BufferStream::wait\_readable<!-- {{#callable:detail::BufferStream::wait_readable}} -->
The `wait_readable` function in the `BufferStream` class always returns `true`, indicating that the buffer is always ready to be read.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as `inline`, suggesting it is intended to be expanded in place to reduce function call overhead.
    - The function is marked as `const`, indicating it does not modify any member variables of the `BufferStream` class.
    - The function simply returns the boolean value `true`.
- **Output**: The function returns a boolean value `true`, indicating that the buffer is always considered readable.
- **See also**: [`detail::BufferStream`](#detailBufferStream)  (Data Structure)


---
#### BufferStream::wait\_writable<!-- {{#callable:detail::BufferStream::wait_writable}} -->
The `wait_writable` function in the `BufferStream` class always returns true, indicating that the buffer is always ready for writing.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as `inline`, suggesting it is intended to be small and efficient.
    - The function simply returns the boolean value `true`.
- **Output**: A boolean value `true`, indicating that the buffer is writable.
- **See also**: [`detail::BufferStream`](#detailBufferStream)  (Data Structure)


---
#### BufferStream::read<!-- {{#callable:detail::BufferStream::read}} -->
The `read` function reads a specified number of bytes from the buffer into a provided character array and updates the read position.
- **Inputs**:
    - `ptr`: A pointer to a character array where the read bytes will be stored.
    - `size`: The number of bytes to read from the buffer.
- **Control Flow**:
    - The function checks the compiler version to determine which method to use for copying data from the buffer.
    - If the compiler is Microsoft Visual C++ and the version is less than 1910, it uses `_Copy_s` to copy data from the buffer to `ptr`.
    - Otherwise, it uses the `copy` method to copy data from the buffer to `ptr`.
    - The function updates the `position` member variable by adding the number of bytes read.
    - Finally, it returns the number of bytes read as a signed size type.
- **Output**: The function returns the number of bytes successfully read from the buffer as a `ssize_t`.
- **See also**: [`detail::BufferStream`](#detailBufferStream)  (Data Structure)


---
#### BufferStream::write<!-- {{#callable:detail::BufferStream::write}} -->
The `write` function appends data from a given character pointer to the internal buffer and returns the size of the data written.
- **Inputs**:
    - `ptr`: A pointer to the character array containing the data to be written to the buffer.
    - `size`: The number of bytes to write from the character array to the buffer.
- **Control Flow**:
    - The function appends the data pointed to by `ptr` with the specified `size` to the internal `buffer` string.
    - The function then returns the `size` cast to `ssize_t`.
- **Output**: The function returns the number of bytes written, cast to `ssize_t`.
- **See also**: [`detail::BufferStream`](#detailBufferStream)  (Data Structure)


---
#### BufferStream::get\_remote\_ip\_and\_port<!-- {{#callable:detail::BufferStream::get_remote_ip_and_port}} -->
The `get_remote_ip_and_port` function is a placeholder method in the `BufferStream` class intended to retrieve the remote IP address and port number, but it currently has no implementation.
- **Inputs**:
    - `ip`: A reference to a `std::string` where the remote IP address is expected to be stored.
    - `port`: A reference to an `int` where the remote port number is expected to be stored.
- **Control Flow**:
    - The function is defined as an inline method within the `BufferStream` class.
    - It takes two parameters by reference, a `std::string` for the IP and an `int` for the port.
    - The function body is empty, indicating no operations are performed and no values are assigned to the input parameters.
- **Output**: There is no output or effect from this function as it is currently unimplemented.
- **See also**: [`detail::BufferStream`](#detailBufferStream)  (Data Structure)


---
#### BufferStream::get\_local\_ip\_and\_port<!-- {{#callable:detail::BufferStream::get_local_ip_and_port}} -->
The `get_local_ip_and_port` function is a placeholder method in the `BufferStream` class intended to retrieve the local IP address and port, but it currently has no implementation.
- **Inputs**:
    - `ip`: A reference to a string where the local IP address is expected to be stored.
    - `port`: A reference to an integer where the local port number is expected to be stored.
- **Control Flow**:
    - The function is defined as an inline method within the `BufferStream` class.
    - It takes two parameters by reference, a string for the IP and an integer for the port.
    - The function body is empty, indicating it does not perform any operations or modify the input parameters.
- **Output**: There is no output or modification to the input parameters as the function body is empty.
- **See also**: [`detail::BufferStream`](#detailBufferStream)  (Data Structure)


---
#### BufferStream::socket<!-- {{#callable:detail::BufferStream::socket}} -->
The `socket` method in the `BufferStream` class returns a default socket identifier of zero.
- **Inputs**: None
- **Control Flow**:
    - The method is defined as `inline`, suggesting it is intended to be expanded in place where it is called, rather than being invoked as a separate function call.
    - The method is marked as `const`, indicating it does not modify any member variables of the `BufferStream` class.
    - The method simply returns the integer value `0`, which is likely a placeholder or default value for a socket identifier.
- **Output**: The method returns a `socket_t` type, which is an alias for an integer representing a socket identifier, specifically returning the value `0`.
- **See also**: [`detail::BufferStream`](#detailBufferStream)  (Data Structure)


---
#### BufferStream::duration<!-- {{#callable:detail::BufferStream::duration}} -->
The `duration` method of the `BufferStream` class returns a constant value of 0.
- **Inputs**: None
- **Control Flow**:
    - The method is defined as `inline`, suggesting it is intended to be expanded in place where it is called, rather than being invoked through a function call.
    - The method is marked as `const`, indicating it does not modify any member variables of the `BufferStream` class.
    - The method simply returns the integer value 0.
- **Output**: The method returns a `time_t` value, which is always 0.
- **See also**: [`detail::BufferStream`](#detailBufferStream)  (Data Structure)


---
#### BufferStream::get\_buffer<!-- {{#callable:detail::BufferStream::get_buffer}} -->
The `get_buffer` function returns a constant reference to the internal buffer string of the `BufferStream` class.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as an inline member of the `BufferStream` class.
    - It returns a constant reference to the private member variable `buffer`.
- **Output**: A constant reference to the `buffer` string member of the `BufferStream` class.
- **See also**: [`detail::BufferStream`](#detailBufferStream)  (Data Structure)



---
### compressor<!-- {{#data_structure:detail::compressor}} -->
- **Type**: `class`
- **Description**: The `compressor` class is an abstract base class that defines an interface for compression operations. It includes a virtual destructor and a pure virtual function `compress`, which must be implemented by derived classes. The `compress` function takes a data buffer, its length, a boolean indicating if it is the last chunk, and a callback function to handle the compressed data. The `Callback` type is defined as a function that takes a character pointer and a size, returning a boolean, which allows for flexible handling of the compressed data.
- **Member Functions**:
    - [`detail::compressor::~compressor`](#compressorcompressor)

**Methods**

---
#### compressor::\~compressor<!-- {{#callable:detail::compressor::~compressor}} -->
The destructor `~compressor` is a virtual default destructor for the `compressor` class.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual, ensuring that the destructor of the derived class is called when an object is deleted through a pointer to the base class.
    - The destructor is defined as default, indicating that the compiler should generate the default implementation for the destructor.
- **Output**: There is no output from the destructor itself; it ensures proper cleanup of resources when a `compressor` object or its derived class object is destroyed.
- **See also**: [`detail::compressor`](#detailcompressor)  (Data Structure)



---
### decompressor<!-- {{#data_structure:detail::decompressor}} -->
- **Type**: `class`
- **Description**: The `decompressor` class is an abstract base class designed for handling decompression tasks. It defines a virtual destructor and two pure virtual functions: `is_valid`, which checks the validity of the decompressor, and `decompress`, which takes data and a callback function to perform the decompression process. The class also defines a `Callback` type, which is a function that processes decompressed data. This class serves as a blueprint for specific decompressor implementations that will provide concrete behavior for these virtual functions.
- **Member Functions**:
    - [`detail::decompressor::~decompressor`](#decompressordecompressor)

**Methods**

---
#### decompressor::\~decompressor<!-- {{#callable:detail::decompressor::~decompressor}} -->
The `~decompressor` function is a virtual destructor for the `decompressor` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The destructor is declared as virtual to allow derived class destructors to be called when an object is deleted through a base class pointer.
    - The destructor is defined as `default`, indicating that the compiler should generate the default implementation for the destructor.
- **Output**: There is no output from the destructor itself; it ensures proper resource cleanup when an object of a derived class is destroyed.
- **See also**: [`detail::decompressor`](#detaildecompressor)  (Data Structure)



---
### nocompressor<!-- {{#data_structure:detail::nocompressor}} -->
- **Type**: `class`
- **Description**: The `nocompressor` class is a final class that inherits from the `compressor` class and provides an implementation for the `compress` method. It is designed to handle data compression operations, although the specific implementation details are not provided in the given code. The class does not have any member variables, indicating that it may rely on the base class or external resources for its functionality.
- **Member Functions**:
    - [`detail::nocompressor::~nocompressor`](#nocompressornocompressor)
- **Inherits From**:
    - [`detail::compressor`](#detailcompressor)

**Methods**

---
#### nocompressor::\~nocompressor<!-- {{#callable:detail::nocompressor::~nocompressor}} -->
The destructor `~nocompressor` is a default destructor for the `nocompressor` class, which overrides the base class `compressor`'s destructor.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as `override`, indicating it overrides a virtual destructor in the base class `compressor`.
    - The destructor is marked as `default`, meaning it uses the compiler-generated default implementation.
- **Output**: There is no output from a destructor; it is used to clean up resources when an object is destroyed.
- **See also**: [`detail::nocompressor`](#detailnocompressor)  (Data Structure)



---
### gzip\_compressor<!-- {{#data_structure:detail::gzip_compressor}} -->
- **Type**: `class`
- **Members**:
    - `is_valid_`: A boolean flag indicating whether the compressor is in a valid state.
    - `strm_`: An instance of z_stream used for managing the compression stream.
- **Description**: The `gzip_compressor` class is a specialized compressor that inherits from a base `compressor` class, designed to handle gzip compression tasks. It includes a constructor and a destructor, and overrides a `compress` method to perform the actual compression of data. The class maintains an internal state with a boolean `is_valid_` to track the validity of the compressor, and a `z_stream` object `strm_` to manage the compression stream, ensuring efficient data processing.
- **Inherits From**:
    - [`detail::compressor`](#detailcompressor)


---
### gzip\_decompressor<!-- {{#data_structure:detail::gzip_decompressor}} -->
- **Type**: `class`
- **Members**:
    - `is_valid_`: A private boolean member indicating the validity of the decompressor.
    - `strm_`: A private member of type z_stream used for managing the decompression stream.
- **Description**: The `gzip_decompressor` class is a final class derived from the `decompressor` base class, designed to handle the decompression of gzip-compressed data. It provides methods to check the validity of the decompressor and to perform the decompression operation using a callback mechanism. The class maintains an internal state with a boolean flag `is_valid_` to track its validity and a `z_stream` object `strm_` to manage the decompression process.
- **Inherits From**:
    - [`detail::decompressor`](#detaildecompressor)


---
### brotli\_compressor<!-- {{#data_structure:detail::brotli_compressor}} -->
- **Type**: `class`
- **Members**:
    - `state_`: A pointer to a BrotliEncoderState object, used to maintain the state of the Brotli compression process.
- **Description**: The `brotli_compressor` class is a specialized compressor that implements the Brotli compression algorithm. It inherits from a base class `compressor` and provides functionality to compress data using the Brotli library. The class maintains a private member `state_`, which is a pointer to a `BrotliEncoderState` object, essential for managing the state during the compression process. The class offers a public method `compress` to perform the compression operation, taking input data and a callback function to handle the compressed output.
- **Inherits From**:
    - [`detail::compressor`](#detailcompressor)


---
### brotli\_decompressor<!-- {{#data_structure:detail::brotli_decompressor}} -->
- **Type**: `class`
- **Members**:
    - `decoder_r`: Holds the result of the Brotli decoding process.
    - `decoder_s`: Pointer to the Brotli decoder state used during decompression.
- **Description**: The `brotli_decompressor` class is a specialized decompressor that implements the decompression functionality using the Brotli algorithm. It inherits from a base class `decompressor` and provides methods to check the validity of the decompressor and to perform the decompression operation. The class maintains internal state through a `BrotliDecoderState` pointer and stores the result of the decoding process in a `BrotliDecoderResult` variable.
- **Inherits From**:
    - [`detail::decompressor`](#detaildecompressor)


---
### zstd\_compressor<!-- {{#data_structure:detail::zstd_compressor}} -->
- **Type**: `class`
- **Members**:
    - `ctx_`: A pointer to a ZSTD_CCtx object, used for managing the compression context.
- **Description**: The `zstd_compressor` class is a specialized compressor that inherits from a base `compressor` class, designed to handle data compression using the Zstandard (ZSTD) algorithm. It encapsulates a ZSTD_CCtx pointer, which is essential for maintaining the state and configuration of the compression process. The class provides a constructor and destructor for managing resources, and it overrides a `compress` method to perform the actual compression operation, taking input data and a callback function as parameters.
- **Inherits From**:
    - [`detail::compressor`](#detailcompressor)


---
### zstd\_decompressor<!-- {{#data_structure:detail::zstd_decompressor}} -->
- **Type**: `class`
- **Members**:
    - `ctx_`: A pointer to a ZSTD_DCtx object, used for managing the decompression context.
- **Description**: The `zstd_decompressor` class is a specialized decompressor that inherits from a base `decompressor` class, designed to handle decompression tasks using the Zstandard (ZSTD) algorithm. It includes a private member `ctx_`, which is a pointer to a ZSTD_DCtx object, essential for maintaining the decompression context. The class provides methods to check the validity of the decompressor and to perform the decompression operation on given data, utilizing a callback mechanism for processing the decompressed output.
- **Inherits From**:
    - [`detail::decompressor`](#detaildecompressor)


---
### stream\_line\_reader<!-- {{#data_structure:detail::stream_line_reader}} -->
- **Type**: `class`
- **Members**:
    - `strm_`: A reference to a Stream object used for reading lines.
    - `fixed_buffer_`: A pointer to a fixed-size character buffer for storing line data.
    - `fixed_buffer_size_`: A constant size of the fixed buffer.
    - `fixed_buffer_used_size_`: Tracks the amount of the fixed buffer currently used.
    - `growable_buffer_`: A string buffer that can grow dynamically to accommodate larger lines.
- **Description**: The `stream_line_reader` class is designed to read lines from a stream, utilizing both a fixed-size buffer and a growable buffer to handle varying line lengths. It maintains a reference to a Stream object for input operations and uses a fixed buffer for efficiency, while a growable buffer is used to handle cases where the line exceeds the fixed buffer's capacity. This class provides methods to access the current line, check its size, and determine if it ends with a CRLF sequence.


---
### mmap<!-- {{#data_structure:detail::mmap}} -->
- **Type**: `class`
- **Members**:
    - `hFile_`: A handle to the file on Windows systems.
    - `hMapping_`: A handle to the file mapping on Windows systems.
    - `fd_`: A file descriptor for the file on non-Windows systems.
    - `size_`: The size of the mapped file.
    - `addr_`: A pointer to the memory-mapped file data.
    - `is_open_empty_file`: A flag indicating if the file is open and empty.
- **Description**: The `mmap` class provides an abstraction for memory-mapped file operations, allowing files to be mapped into memory for efficient access. It supports both Windows and non-Windows systems by using platform-specific handles or file descriptors. The class manages the opening and closing of files, checks if a file is open, and provides access to the file's size and data. It encapsulates the complexity of handling memory-mapped files, making it easier to work with file data as if it were in memory.


---
### SocketStream<!-- {{#data_structure:SocketStream}} -->
- **Type**: `class`
- **Members**:
    - `sock_`: Stores the socket descriptor for the connection.
    - `read_timeout_sec_`: Specifies the read timeout in seconds.
    - `read_timeout_usec_`: Specifies the read timeout in microseconds.
    - `write_timeout_sec_`: Specifies the write timeout in seconds.
    - `write_timeout_usec_`: Specifies the write timeout in microseconds.
    - `max_timeout_msec_`: Defines the maximum timeout in milliseconds.
    - `start_time_`: Records the start time of the socket stream operation.
    - `read_buff_`: Holds the buffer for reading data from the socket.
    - `read_buff_off_`: Tracks the current offset in the read buffer.
    - `read_buff_content_size_`: Indicates the size of the content in the read buffer.
    - `read_buff_size_`: Defines the fixed size of the read buffer.
- **Description**: The `SocketStream` class is a final class derived from `Stream` that manages socket-based I/O operations with specific timeout settings for reading and writing. It encapsulates a socket descriptor and provides functionality to read from and write to the socket, while handling timeouts and buffering internally. The class also offers methods to retrieve remote and local IP addresses and ports, and it maintains a read buffer to optimize data handling.
- **Inherits From**:
    - [`Stream`](#Stream)


---
### SSLSocketStream<!-- {{#data_structure:SSLSocketStream}} -->
- **Type**: `class`
- **Members**:
    - `sock_`: A socket descriptor used for network communication.
    - `ssl_`: A pointer to an SSL structure for managing SSL/TLS connections.
    - `read_timeout_sec_`: The read timeout in seconds.
    - `read_timeout_usec_`: The read timeout in microseconds.
    - `write_timeout_sec_`: The write timeout in seconds.
    - `write_timeout_usec_`: The write timeout in microseconds.
    - `max_timeout_msec_`: The maximum timeout in milliseconds.
    - `start_time_`: The start time point for measuring duration.
- **Description**: The `SSLSocketStream` class is a specialized stream that handles SSL/TLS encrypted communication over a network socket. It extends the `Stream` class and provides functionality for reading from and writing to a socket with SSL encryption, as well as managing timeouts for these operations. The class maintains internal state such as the socket descriptor, SSL context, and various timeout settings to ensure secure and efficient data transmission.
- **Member Functions**:
    - [`SSLSocketStream::SSLSocketStream`](#SSLSocketStreamSSLSocketStream)
    - [`SSLSocketStream::~SSLSocketStream`](#SSLSocketStreamSSLSocketStream)
    - [`SSLSocketStream::is_readable`](#SSLSocketStreamis_readable)
    - [`SSLSocketStream::wait_readable`](#SSLSocketStreamwait_readable)
    - [`SSLSocketStream::wait_writable`](#SSLSocketStreamwait_writable)
    - [`SSLSocketStream::write`](#SSLSocketStreamwrite)
    - [`SSLSocketStream::get_remote_ip_and_port`](#SSLSocketStreamget_remote_ip_and_port)
    - [`SSLSocketStream::get_local_ip_and_port`](#SSLSocketStreamget_local_ip_and_port)
    - [`SSLSocketStream::socket`](#SSLSocketStreamsocket)
    - [`SSLSocketStream::duration`](#SSLSocketStreamduration)
- **Inherits From**:
    - [`Stream`](#Stream)

**Methods**

---
#### SSLSocketStream::SSLSocketStream<!-- {{#callable:SSLSocketStream::SSLSocketStream}} -->
The SSLSocketStream constructor initializes an SSL socket stream with specified socket, SSL context, read and write timeouts, and a start time, while clearing the SSL auto-retry mode.
- **Inputs**:
    - `sock`: A socket descriptor of type `socket_t` representing the network socket.
    - `ssl`: A pointer to an SSL object used for managing the SSL/TLS connection.
    - `read_timeout_sec`: The read timeout in seconds, of type `time_t`, for the socket operations.
    - `read_timeout_usec`: The read timeout in microseconds, of type `time_t`, for the socket operations.
    - `write_timeout_sec`: The write timeout in seconds, of type `time_t`, for the socket operations.
    - `write_timeout_usec`: The write timeout in microseconds, of type `time_t`, for the socket operations.
    - `max_timeout_msec`: The maximum timeout in milliseconds, of type `time_t`, for the socket operations.
    - `start_time`: A `std::chrono::time_point` representing the start time of the socket stream.
- **Control Flow**:
    - Initialize member variables with the provided arguments.
    - Call `SSL_clear_mode` on the `ssl` object to clear the `SSL_MODE_AUTO_RETRY` mode.
- **Output**: The function does not return a value as it is a constructor.
- **See also**: [`SSLSocketStream`](#SSLSocketStream)  (Data Structure)


---
#### SSLSocketStream::\~SSLSocketStream<!-- {{#callable:SSLSocketStream::~SSLSocketStream}} -->
The destructor for the SSLSocketStream class is defined as the default destructor.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined inline and marked as default, indicating that the compiler will generate the default implementation for it.
    - No custom cleanup or resource management logic is implemented in this destructor.
- **Output**: The destructor does not return any value, as is typical for destructors.
- **See also**: [`SSLSocketStream`](#SSLSocketStream)  (Data Structure)


---
#### SSLSocketStream::is\_readable<!-- {{#callable:SSLSocketStream::is_readable}} -->
The `is_readable` function checks if there is any pending data to be read from the SSL connection.
- **Inputs**: None
- **Control Flow**:
    - The function calls `SSL_pending` with the `ssl_` member variable to determine the number of bytes available to read.
    - It returns `true` if the number of pending bytes is greater than zero, indicating that the stream is readable.
- **Output**: A boolean value indicating whether there is data available to read from the SSL connection.
- **See also**: [`SSLSocketStream`](#SSLSocketStream)  (Data Structure)


---
#### SSLSocketStream::wait\_readable<!-- {{#callable:SSLSocketStream::wait_readable}} -->
The `wait_readable` function checks if the SSL socket stream is ready for reading within a specified timeout period.
- **Inputs**:
    - `None`: This function does not take any input parameters.
- **Control Flow**:
    - Check if `max_timeout_msec_` is less than or equal to 0.
    - If true, call [`select_read`](#select_read) with `sock_`, `read_timeout_sec_`, and `read_timeout_usec_` and return whether the result is greater than 0.
    - If false, calculate the actual timeout using [`calc_actual_timeout`](#detailcalc_actual_timeout) with `max_timeout_msec_`, `duration()`, `read_timeout_sec_`, and `read_timeout_usec_`.
    - Call [`select_read`](#select_read) with `sock_`, the calculated `read_timeout_sec`, and `read_timeout_usec`, and return whether the result is greater than 0.
- **Output**: Returns a boolean indicating whether the socket is ready for reading within the specified timeout.
- **Functions called**:
    - [`select_read`](#select_read)
    - [`detail::calc_actual_timeout`](#detailcalc_actual_timeout)
    - [`detail::SocketStream::duration`](#SocketStreamduration)
- **See also**: [`SSLSocketStream`](#SSLSocketStream)  (Data Structure)


---
#### SSLSocketStream::wait\_writable<!-- {{#callable:SSLSocketStream::wait_writable}} -->
The `wait_writable` function checks if the SSL socket is ready for writing by verifying the socket's writability, its aliveness, and the SSL peer's connection status.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - Call [`select_write`](#select_write) with the socket and write timeout values to check if the socket is writable.
    - Check if the result of [`select_write`](#select_write) is greater than 0, indicating the socket is ready for writing.
    - Verify that the socket is still alive using [`is_socket_alive`](#is_socket_alive).
    - Ensure that the SSL peer connection is not closed using [`is_ssl_peer_could_be_closed`](#parse_range_headeris_ssl_peer_could_be_closed).
    - Return true if all conditions are met, otherwise return false.
- **Output**: A boolean value indicating whether the socket is writable, alive, and the SSL peer connection is open.
- **Functions called**:
    - [`select_write`](#select_write)
    - [`is_socket_alive`](#is_socket_alive)
    - [`parse_range_header::is_ssl_peer_could_be_closed`](#parse_range_headeris_ssl_peer_could_be_closed)
- **See also**: [`SSLSocketStream`](#SSLSocketStream)  (Data Structure)


---
#### SSLSocketStream::write<!-- {{#callable:SSLSocketStream::write}} -->
The `write` function attempts to write data to an SSL socket, handling potential write errors and retrying if necessary.
- **Inputs**:
    - `ptr`: A pointer to the data buffer that needs to be written to the SSL socket.
    - `size`: The size of the data buffer pointed to by `ptr`.
- **Control Flow**:
    - Check if the socket is writable using `wait_writable()`; if not, return -1.
    - Determine the maximum size that can be handled by `SSL_write` based on the minimum of `size` and the maximum integer value.
    - Attempt to write data to the SSL socket using `SSL_write`; if successful, return the number of bytes written.
    - If `SSL_write` returns an error, retrieve the error code using `SSL_get_error`.
    - On Windows, retry writing if the error is `SSL_ERROR_WANT_WRITE` or a timeout error; on other systems, retry only if the error is `SSL_ERROR_WANT_WRITE`.
    - Retry writing up to 1000 times, waiting for the socket to become writable and sleeping for 10 microseconds between attempts.
    - If writing is successful during retries, return the number of bytes written; otherwise, return -1.
- **Output**: Returns the number of bytes successfully written to the SSL socket, or -1 if writing fails.
- **Functions called**:
    - [`detail::SocketStream::wait_writable`](#SocketStreamwait_writable)
- **See also**: [`SSLSocketStream`](#SSLSocketStream)  (Data Structure)


---
#### SSLSocketStream::get\_remote\_ip\_and\_port<!-- {{#callable:SSLSocketStream::get_remote_ip_and_port}} -->
The `get_remote_ip_and_port` method retrieves the remote IP address and port number associated with the SSL socket stream.
- **Inputs**:
    - `ip`: A reference to a string where the remote IP address will be stored.
    - `port`: A reference to an integer where the remote port number will be stored.
- **Control Flow**:
    - The method calls the `detail::get_remote_ip_and_port` function, passing the socket descriptor `sock_`, and the references `ip` and `port` to store the results.
- **Output**: The method does not return a value but modifies the `ip` and `port` arguments to contain the remote IP address and port number.
- **See also**: [`SSLSocketStream`](#SSLSocketStream)  (Data Structure)


---
#### SSLSocketStream::get\_local\_ip\_and\_port<!-- {{#callable:SSLSocketStream::get_local_ip_and_port}} -->
The `get_local_ip_and_port` method retrieves the local IP address and port number associated with the socket in the `SSLSocketStream` class.
- **Inputs**:
    - `ip`: A reference to a string where the local IP address will be stored.
    - `port`: A reference to an integer where the local port number will be stored.
- **Control Flow**:
    - The method calls the `detail::get_local_ip_and_port` function, passing the socket (`sock_`), and the references to `ip` and `port` to retrieve the local IP and port.
- **Output**: The method does not return a value but modifies the `ip` and `port` arguments to contain the local IP address and port number.
- **See also**: [`SSLSocketStream`](#SSLSocketStream)  (Data Structure)


---
#### SSLSocketStream::socket<!-- {{#callable:SSLSocketStream::socket}} -->
The `socket` method returns the socket descriptor associated with the `SSLSocketStream` object.
- **Inputs**: None
- **Control Flow**:
    - The method is defined as `inline`, meaning it is a small function that is expanded in place where it is called, rather than being invoked through the usual function call mechanism.
    - The method simply returns the private member variable `sock_`, which is of type `socket_t`.
- **Output**: The method returns a `socket_t` type, which is the socket descriptor of the `SSLSocketStream`.
- **See also**: [`SSLSocketStream`](#SSLSocketStream)  (Data Structure)


---
#### SSLSocketStream::duration<!-- {{#callable:SSLSocketStream::duration}} -->
The `duration` function calculates the elapsed time in milliseconds since the `SSLSocketStream` object was created.
- **Inputs**: None
- **Control Flow**:
    - The function retrieves the current time using `std::chrono::steady_clock::now()`.
    - It calculates the difference between the current time and the `start_time_` member variable, which represents the time when the `SSLSocketStream` object was created.
    - The time difference is cast to milliseconds using `std::chrono::duration_cast<std::chrono::milliseconds>`.
    - The function returns the count of milliseconds as a `time_t` value.
- **Output**: The function returns the elapsed time in milliseconds as a `time_t` value.
- **See also**: [`SSLSocketStream`](#SSLSocketStream)  (Data Structure)



---
### MultipartFormDataParser<!-- {{#data_structure:parse_range_header::MultipartFormDataParser}} -->
- **Type**: `class`
- **Members**:
    - `dash_`: A constant string representing the boundary prefix '--'.
    - `crlf_`: A constant string representing the carriage return and line feed '\r\n'.
    - `boundary_`: A string storing the boundary used to separate parts in the multipart data.
    - `dash_boundary_crlf_`: A string combining the boundary with '--' and '\r\n'.
    - `crlf_dash_boundary_`: A string combining '\r\n' with '--' and the boundary.
    - `state_`: An integer representing the current state of the parser.
    - `is_valid_`: A boolean indicating whether the parsed data is valid.
    - `file_`: An instance of MultipartFormData holding information about the current file being processed.
    - `buf_`: A string used as a buffer to store incoming data.
    - `buf_spos_`: A size_t indicating the start position of valid data in the buffer.
    - `buf_epos_`: A size_t indicating the end position of valid data in the buffer.
- **Description**: The `MultipartFormDataParser` class is designed to parse multipart form data, typically used in HTTP requests to upload files. It manages the parsing state and processes headers and content of each part, storing relevant information in a `MultipartFormData` instance. The class uses a buffer to handle incoming data and employs boundary strings to identify and separate different parts of the multipart data. It provides methods to set the boundary, validate the parsed data, and parse the data using callbacks for content and headers.
- **Member Functions**:
    - [`parse_range_header::MultipartFormDataParser::MultipartFormDataParser`](#MultipartFormDataParserMultipartFormDataParser)
    - [`parse_range_header::MultipartFormDataParser::set_boundary`](#MultipartFormDataParserset_boundary)
    - [`parse_range_header::MultipartFormDataParser::is_valid`](#MultipartFormDataParseris_valid)
    - [`parse_range_header::MultipartFormDataParser::parse`](#MultipartFormDataParserparse)
    - [`parse_range_header::MultipartFormDataParser::clear_file_info`](#MultipartFormDataParserclear_file_info)
    - [`parse_range_header::MultipartFormDataParser::start_with_case_ignore`](#MultipartFormDataParserstart_with_case_ignore)
    - [`parse_range_header::MultipartFormDataParser::start_with`](#MultipartFormDataParserstart_with)
    - [`parse_range_header::MultipartFormDataParser::buf_size`](#MultipartFormDataParserbuf_size)
    - [`parse_range_header::MultipartFormDataParser::buf_data`](#MultipartFormDataParserbuf_data)
    - [`parse_range_header::MultipartFormDataParser::buf_head`](#MultipartFormDataParserbuf_head)
    - [`parse_range_header::MultipartFormDataParser::buf_start_with`](#MultipartFormDataParserbuf_start_with)
    - [`parse_range_header::MultipartFormDataParser::buf_find`](#MultipartFormDataParserbuf_find)
    - [`parse_range_header::MultipartFormDataParser::buf_append`](#MultipartFormDataParserbuf_append)
    - [`parse_range_header::MultipartFormDataParser::buf_erase`](#MultipartFormDataParserbuf_erase)

**Methods**

---
#### MultipartFormDataParser::MultipartFormDataParser<!-- {{#callable:parse_range_header::MultipartFormDataParser::MultipartFormDataParser}} -->
The `MultipartFormDataParser` constructor initializes a parser object for handling multipart form data without any initial setup.
- **Inputs**: None
- **Control Flow**:
    - The constructor is defined as `default`, meaning it does not perform any specific initialization beyond what is automatically provided by the compiler.
- **Output**: An instance of the `MultipartFormDataParser` class is created with default initialization.
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::set\_boundary<!-- {{#callable:parse_range_header::MultipartFormDataParser::set_boundary}} -->
The `set_boundary` function sets the boundary string for parsing multipart form data and updates related boundary markers used in the parsing process.
- **Inputs**:
    - `boundary`: A string representing the boundary used to separate parts in multipart form data, passed as an rvalue reference to allow for efficient move semantics.
- **Control Flow**:
    - Assigns the input boundary string to the member variable `boundary_`.
    - Concatenates the `dash_`, `boundary_`, and `crlf_` strings to form `dash_boundary_crlf_`.
    - Concatenates the `crlf_`, `dash_`, and `boundary_` strings to form `crlf_dash_boundary_`.
- **Output**: This function does not return any value; it modifies the internal state of the `MultipartFormDataParser` object by setting boundary-related member variables.
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::is\_valid<!-- {{#callable:parse_range_header::MultipartFormDataParser::is_valid}} -->
The `is_valid` function checks and returns the validity status of the `MultipartFormDataParser` object.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the private member variable `is_valid_`.
- **Output**: A boolean value indicating whether the `MultipartFormDataParser` object is in a valid state.
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::parse<!-- {{#callable:parse_range_header::MultipartFormDataParser::parse}} -->
The `parse` function processes a buffer of multipart form data, extracting headers and content, and invoking callbacks for each part.
- **Inputs**:
    - `buf`: A pointer to a character array containing the multipart form data to be parsed.
    - `n`: The size of the buffer `buf`.
    - `content_callback`: A callback function to handle the content of each part of the multipart data.
    - `header_callback`: A callback function to handle the headers of each part of the multipart data.
- **Control Flow**:
    - Append the buffer data to an internal buffer using [`buf_append`](#MultipartFormDataParserbuf_append).
    - Enter a loop that continues while there is data in the buffer.
    - Switch on the current state to determine the parsing phase (initial boundary, new entry, headers, body, boundary).
    - In state 0, check for the initial boundary and transition to state 1 if found.
    - In state 1, clear file information and transition to state 2.
    - In state 2, parse headers, invoking `header_callback` and transitioning to state 3 if successful.
    - In state 3, parse the body content, invoking `content_callback` and transitioning to state 4 if successful.
    - In state 4, check for the boundary and transition back to state 1 or end parsing if the final boundary is found.
    - Return `true` if parsing completes successfully, or `false` if an error occurs.
- **Output**: Returns a boolean indicating whether the parsing was successful (`true`) or encountered an error (`false`).
- **Functions called**:
    - [`parse_range_header::MultipartFormDataParser::buf_append`](#MultipartFormDataParserbuf_append)
    - [`parse_range_header::MultipartFormDataParser::buf_size`](#MultipartFormDataParserbuf_size)
    - [`parse_range_header::MultipartFormDataParser::buf_erase`](#MultipartFormDataParserbuf_erase)
    - [`parse_range_header::MultipartFormDataParser::buf_find`](#MultipartFormDataParserbuf_find)
    - [`parse_range_header::MultipartFormDataParser::buf_start_with`](#MultipartFormDataParserbuf_start_with)
    - [`parse_range_header::MultipartFormDataParser::clear_file_info`](#MultipartFormDataParserclear_file_info)
    - [`parse_range_header::MultipartFormDataParser::buf_head`](#MultipartFormDataParserbuf_head)
    - [`parse_header`](#parse_header)
    - [`parse_range_header::MultipartFormDataParser::start_with_case_ignore`](#MultipartFormDataParserstart_with_case_ignore)
    - [`trim_copy`](#trim_copy)
    - [`detail::str_len`](#detailstr_len)
    - [`parse_disposition_params`](#parse_disposition_params)
    - [`decode_url`](#decode_url)
    - [`parse_range_header::MultipartFormDataParser::buf_data`](#MultipartFormDataParserbuf_data)
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::clear\_file\_info<!-- {{#callable:parse_range_header::MultipartFormDataParser::clear_file_info}} -->
The `clear_file_info` function resets the `name`, `filename`, and `content_type` fields of the `file_` object to empty strings.
- **Inputs**: None
- **Control Flow**:
    - Accesses the `file_` object, which is an instance of `MultipartFormData`.
    - Clears the `name` field of the `file_` object by calling `clear()` on it.
    - Clears the `filename` field of the `file_` object by calling `clear()` on it.
    - Clears the `content_type` field of the `file_` object by calling `clear()` on it.
- **Output**: This function does not return any value; it modifies the state of the `file_` object by clearing its fields.
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::start\_with\_case\_ignore<!-- {{#callable:parse_range_header::MultipartFormDataParser::start_with_case_ignore}} -->
The `start_with_case_ignore` function checks if a given string `a` starts with another string `b`, ignoring case differences.
- **Inputs**:
    - `a`: A reference to a `std::string` that represents the string to be checked.
    - `b`: A pointer to a constant character array (C-style string) that represents the prefix to check against.
- **Control Flow**:
    - Calculate the length of the string `b` using `strlen`.
    - Check if the length of `a` is less than the length of `b`; if so, return `false`.
    - Iterate over each character in `b` and compare it with the corresponding character in `a` after converting both to lowercase using `case_ignore::to_lower`.
    - If any character comparison fails, return `false`.
    - If all characters match, return `true`.
- **Output**: Returns a boolean value: `true` if the string `a` starts with the string `b` (case-insensitive), otherwise `false`.
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::start\_with<!-- {{#callable:parse_range_header::MultipartFormDataParser::start_with}} -->
The `start_with` function checks if a substring of string `a` starting at position `spos` and ending at `epos` matches the string `b`.
- **Inputs**:
    - `a`: The main string to be checked against.
    - `spos`: The starting position in string `a` from where the comparison begins.
    - `epos`: The ending position in string `a` up to which the comparison is made.
    - `b`: The string to compare with the substring of `a`.
- **Control Flow**:
    - Check if the length of the substring from `spos` to `epos` is less than the size of `b`; if so, return false.
    - Iterate over each character in `b` and compare it with the corresponding character in the substring of `a` starting from `spos`.
    - If any character does not match, return false.
    - If all characters match, return true.
- **Output**: Returns a boolean value indicating whether the substring of `a` matches `b`.
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::buf\_size<!-- {{#callable:parse_range_header::MultipartFormDataParser::buf_size}} -->
The `buf_size` function calculates the current size of the buffer by subtracting the start position from the end position.
- **Inputs**: None
- **Control Flow**:
    - The function calculates the buffer size by subtracting `buf_spos_` from `buf_epos_`.
- **Output**: The function returns a `size_t` value representing the size of the buffer.
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::buf\_data<!-- {{#callable:parse_range_header::MultipartFormDataParser::buf_data}} -->
The `buf_data` function returns a pointer to the current start position of the buffer within the `MultipartFormDataParser` class.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the buffer `buf_` using the current start position `buf_spos_`.
    - It returns a pointer to the character at the position `buf_spos_` in the buffer `buf_`.
- **Output**: A `const char*` pointer to the current start position of the buffer.
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::buf\_head<!-- {{#callable:parse_range_header::MultipartFormDataParser::buf_head}} -->
The `buf_head` function returns a substring of the buffer starting from the current buffer start position with a specified length.
- **Inputs**:
    - `l`: The length of the substring to be returned from the buffer.
- **Control Flow**:
    - The function calls the `substr` method on the `buf_` string, starting from `buf_spos_` and with a length of `l`.
- **Output**: A `std::string` containing the specified portion of the buffer.
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::buf\_start\_with<!-- {{#callable:parse_range_header::MultipartFormDataParser::buf_start_with}} -->
The `buf_start_with` function checks if the buffer, defined by `buf_`, `buf_spos_`, and `buf_epos_`, starts with a given string `s`.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that represents the string to check against the start of the buffer.
- **Control Flow**:
    - The function calls the [`start_with`](#MultipartFormDataParserstart_with) method, passing the buffer `buf_`, the start position `buf_spos_`, the end position `buf_epos_`, and the string `s` as arguments.
    - The [`start_with`](#MultipartFormDataParserstart_with) method checks if the substring of `buf_` from `buf_spos_` to `buf_epos_` starts with the string `s`.
- **Output**: Returns a boolean value: `true` if the buffer starts with the string `s`, otherwise `false`.
- **Functions called**:
    - [`parse_range_header::MultipartFormDataParser::start_with`](#MultipartFormDataParserstart_with)
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::buf\_find<!-- {{#callable:parse_range_header::MultipartFormDataParser::buf_find}} -->
The `buf_find` function searches for the first occurrence of a given substring within a buffer and returns its position relative to the buffer's start position.
- **Inputs**:
    - `s`: A constant reference to a `std::string` representing the substring to search for within the buffer.
- **Control Flow**:
    - Initialize `c` with the first character of the input string `s`.
    - Set `off` to the current start position of the buffer, `buf_spos_`.
    - Enter a loop that continues while `off` is less than the end position of the buffer, `buf_epos_`.
    - Within the loop, initialize `pos` to `off` and enter another loop to find the first occurrence of `c` in the buffer starting from `pos`.
    - If `pos` reaches `buf_epos_`, return the size of the buffer, indicating the substring was not found.
    - If `buf_[pos]` matches `c`, break out of the inner loop.
    - Calculate `remaining_size` as the difference between `buf_epos_` and `pos`.
    - If the size of `s` is greater than `remaining_size`, return the buffer size, indicating the substring cannot fit.
    - Check if the buffer starting at `pos` matches the string `s` using the [`start_with`](#MultipartFormDataParserstart_with) function.
    - If a match is found, return the position `pos` adjusted by `buf_spos_`.
    - Increment `off` to `pos + 1` and continue the outer loop.
    - If the loop completes without finding the substring, return the buffer size.
- **Output**: Returns a `size_t` representing the position of the first occurrence of the substring `s` within the buffer, or the buffer size if the substring is not found.
- **Functions called**:
    - [`parse_range_header::MultipartFormDataParser::buf_size`](#MultipartFormDataParserbuf_size)
    - [`parse_range_header::MultipartFormDataParser::start_with`](#MultipartFormDataParserstart_with)
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::buf\_append<!-- {{#callable:parse_range_header::MultipartFormDataParser::buf_append}} -->
The `buf_append` function appends a given data buffer to an internal buffer, managing buffer size and position indices.
- **Inputs**:
    - `data`: A pointer to the character array (const char*) that contains the data to be appended to the buffer.
    - `n`: The size (size_t) of the data to be appended, indicating the number of characters to append from the data array.
- **Control Flow**:
    - Calculate the remaining size of the buffer using `buf_size()`.
    - If there is remaining data and the start position `buf_spos_` is greater than 0, shift the existing data to the beginning of the buffer.
    - Reset `buf_spos_` to 0 and update `buf_epos_` to the remaining size.
    - Check if the current buffer size plus the new data size exceeds the buffer's capacity, and resize the buffer if necessary.
    - Append the new data to the buffer starting from `buf_epos_`.
    - Update `buf_epos_` to reflect the new end position after appending the data.
- **Output**: The function does not return a value; it modifies the internal buffer state by appending new data to it.
- **Functions called**:
    - [`parse_range_header::MultipartFormDataParser::buf_size`](#MultipartFormDataParserbuf_size)
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)


---
#### MultipartFormDataParser::buf\_erase<!-- {{#callable:parse_range_header::MultipartFormDataParser::buf_erase}} -->
The `buf_erase` function advances the starting position of a buffer by a specified size, effectively removing that portion from the buffer.
- **Inputs**:
    - `size`: The number of bytes to advance the buffer's starting position by, effectively erasing this many bytes from the beginning of the buffer.
- **Control Flow**:
    - The function takes a single input parameter, `size`, which indicates how many bytes to erase from the start of the buffer.
    - It increments the `buf_spos_` member variable by the value of `size`, effectively moving the starting position of the buffer forward by `size` bytes.
- **Output**: The function does not return any value; it modifies the internal state of the buffer by updating the `buf_spos_` member variable.
- **See also**: [`parse_range_header::MultipartFormDataParser`](#parse_range_headerMultipartFormDataParser)  (Data Structure)



---
### WSInit<!-- {{#data_structure:parse_range_header::WSInit}} -->
- **Type**: `class`
- **Members**:
    - `is_valid_`: A boolean flag indicating whether the WSAStartup was successful.
- **Description**: The `WSInit` class is a utility class designed to manage the initialization and cleanup of the Windows Sockets API. It encapsulates the process of starting up the Winsock library using `WSAStartup` and ensures that `WSACleanup` is called if the initialization was successful. The class maintains a boolean member `is_valid_` to track the success of the initialization process, ensuring that resources are properly managed and released.
- **Member Functions**:
    - [`parse_range_header::WSInit::WSInit`](#WSInitWSInit)
    - [`parse_range_header::WSInit::~WSInit`](#WSInitWSInit)

**Methods**

---
#### WSInit::WSInit<!-- {{#callable:parse_range_header::WSInit::WSInit}} -->
The WSInit constructor initializes the Windows Sockets API and sets a validity flag based on the success of the initialization.
- **Inputs**: None
- **Control Flow**:
    - Declare a WSADATA object named wsaData.
    - Call WSAStartup with version 0x0002 and the address of wsaData.
    - If WSAStartup returns 0, set the is_valid_ member variable to true.
- **Output**: The constructor does not return a value, but it sets the is_valid_ member variable to true if the Windows Sockets API is successfully initialized.
- **See also**: [`parse_range_header::WSInit`](#parse_range_headerWSInit)  (Data Structure)


---
#### WSInit::\~WSInit<!-- {{#callable:parse_range_header::WSInit::~WSInit}} -->
The destructor ~WSInit() cleans up the Windows Sockets API if the initialization was successful.
- **Inputs**: None
- **Control Flow**:
    - Check if the member variable is_valid_ is true.
    - If true, call WSACleanup() to clean up the Windows Sockets API.
- **Output**: This function does not return any value as it is a destructor.
- **See also**: [`parse_range_header::WSInit`](#parse_range_headerWSInit)  (Data Structure)



---
### ContentProviderAdapter<!-- {{#data_structure:parse_range_header::ContentProviderAdapter}} -->
- **Type**: `class`
- **Members**:
    - `content_provider_`: A private member of type ContentProviderWithoutLength that stores the content provider instance.
- **Description**: The ContentProviderAdapter class is a wrapper around a ContentProviderWithoutLength object, allowing it to be used with a specific interface that requires a callable object. It provides an operator() method that forwards calls to the underlying content provider, facilitating interaction with a DataSink object. This class is designed to adapt content providers that do not inherently support length information, enabling their use in contexts where such an interface is required.
- **Member Functions**:
    - [`parse_range_header::ContentProviderAdapter::ContentProviderAdapter`](#ContentProviderAdapterContentProviderAdapter)
    - [`parse_range_header::ContentProviderAdapter::operator()`](#ContentProviderAdapteroperator())

**Methods**

---
#### ContentProviderAdapter::ContentProviderAdapter<!-- {{#callable:parse_range_header::ContentProviderAdapter::ContentProviderAdapter}} -->
The `ContentProviderAdapter` constructor initializes an adapter object with a `ContentProviderWithoutLength` instance.
- **Inputs**:
    - `content_provider`: An rvalue reference to a `ContentProviderWithoutLength` object, which is used to initialize the adapter.
- **Control Flow**:
    - The constructor takes an rvalue reference to a `ContentProviderWithoutLength` object as its parameter.
    - It initializes the member variable `content_provider_` with the provided `content_provider` argument.
- **Output**: An instance of `ContentProviderAdapter` is created with the `content_provider_` member initialized.
- **See also**: [`parse_range_header::ContentProviderAdapter`](#parse_range_headerContentProviderAdapter)  (Data Structure)


---
#### ContentProviderAdapter::operator\(\)<!-- {{#callable:parse_range_header::ContentProviderAdapter::operator()}} -->
The `operator()` function in `ContentProviderAdapter` invokes the `content_provider_` with a given offset and sink, returning its boolean result.
- **Inputs**:
    - `offset`: A `size_t` value representing the offset to be used when invoking the content provider.
    - `(unnamed)`: An unnamed `size_t` parameter that is not used in the function.
    - `sink`: A reference to a `DataSink` object that is passed to the content provider.
- **Control Flow**:
    - The function directly calls the `content_provider_` member with the `offset` and `sink` parameters.
    - The result of the `content_provider_` call is returned as the output of the function.
- **Output**: A boolean value that is the result of the `content_provider_` function call.
- **See also**: [`parse_range_header::ContentProviderAdapter`](#parse_range_headerContentProviderAdapter)  (Data Structure)



# Functions

---
### make\_unique<!-- {{#callable:detail::make_unique}} -->
Creates a `std::unique_ptr` to an array of type `T` with a specified size.
- **Inputs**:
    - `n`: The number of elements in the array to be allocated.
- **Control Flow**:
    - Checks if the type `T` is an array using `std::is_array<T>::value` to ensure the function is only instantiated for array types.
    - Uses `std::remove_extent<T>::type` to determine the underlying type of the array elements.
    - Allocates an array of `RT` (the underlying type) of size `n` using `new` and wraps it in a `std::unique_ptr`.
- **Output**: Returns a `std::unique_ptr<T>` that manages the allocated array of type `T`.


---
### to\_lower<!-- {{#callable:detail::case_ignore::to_lower}} -->
Converts an integer character code to its lowercase equivalent using a lookup table.
- **Inputs**:
    - `c`: An integer representing the character code to be converted to lowercase.
- **Control Flow**:
    - The function defines a static lookup table of size 256 that maps each character code to its lowercase equivalent.
    - The input character code `c` is cast to an unsigned char and used as an index to access the lookup table.
    - The function returns the value from the lookup table corresponding to the input character code.
- **Output**: Returns the lowercase equivalent of the input character code as an unsigned char.


---
### equal<!-- {{#callable:detail::case_ignore::equal}} -->
Compares two strings for equality, ignoring case differences.
- **Inputs**:
    - `a`: The first string to compare.
    - `b`: The second string to compare.
- **Control Flow**:
    - First, the function checks if the sizes of the two strings `a` and `b` are equal.
    - If the sizes are equal, it proceeds to compare the characters of both strings using `std::equal`.
    - The comparison of characters is done using a lambda function that converts each character to lowercase before comparing them.
- **Output**: Returns true if both strings are of the same length and all corresponding characters are equal when case is ignored; otherwise, returns false.
- **Functions called**:
    - [`detail::case_ignore::to_lower`](#case_ignoreto_lower)


---
### duration\_to\_sec\_and\_usec<!-- {{#callable:detail::duration_to_sec_and_usec}} -->
Converts a duration into seconds and microseconds and passes the result to a callback function.
- **Inputs**:
    - `duration`: A duration object that can be converted to seconds and microseconds.
    - `callback`: A callable object (function, lambda, etc.) that takes two parameters: seconds and microseconds.
- **Control Flow**:
    - The function begins by casting the input `duration` to seconds using `std::chrono::duration_cast`.
    - It then calculates the remaining microseconds by subtracting the seconds portion from the original duration and casting the result to microseconds.
    - Finally, it invokes the `callback` function with the calculated seconds and microseconds as arguments.
- **Output**: The function does not return a value; instead, it outputs the seconds and microseconds through the provided callback function.


---
### str\_len<!-- {{#callable:detail::str_len}} -->
Calculates the length of a string literal excluding the null terminator.
- **Inputs**:
    - `char (&)[N]`: A reference to a character array (string literal) of size N.
- **Control Flow**:
    - The function template takes a reference to a character array as its parameter.
    - It computes the length of the array by subtracting 1 from the size N to exclude the null terminator.
- **Output**: Returns the length of the string as a constant expression of type size_t.


---
### is\_numeric<!-- {{#callable:detail::is_numeric}} -->
Checks if a given string represents a numeric value.
- **Inputs**:
    - `str`: A constant reference to a `std::string` that is to be checked for numeric content.
- **Control Flow**:
    - The function first checks if the input string `str` is not empty.
    - If the string is not empty, it then uses `std::all_of` to verify that every character in the string is a digit by applying the `::isdigit` function.
- **Output**: Returns `true` if the string is non-empty and all characters are digits; otherwise, returns `false`.


---
### get\_header\_value\_u64<!-- {{#callable:detail::get_header_value_u64}} -->
Retrieves a 64-bit unsigned integer header value from the provided headers, using a default value if the key is not found.
- **Inputs**:
    - `headers`: A constant reference to a `Headers` object containing key-value pairs.
    - `key`: A `std::string` representing the key for which the header value is to be retrieved.
    - `def`: A `uint64_t` default value to return if the specified key is not found.
    - `id`: A `size_t` identifier used in the retrieval process.
- **Control Flow**:
    - The function initializes a boolean variable `dummy` to false.
    - It calls another overloaded function [`get_header_value_u64`](#detailget_header_value_u64) with the same parameters plus the `dummy` variable.
- **Output**: Returns a `uint64_t` value corresponding to the header associated with the specified key, or the default value if the key is absent.
- **Functions called**:
    - [`detail::get_header_value_u64`](#detailget_header_value_u64)


---
### set\_socket\_opt\_impl<!-- {{#callable:detail::set_socket_opt_impl}} -->
Sets a socket option using the `setsockopt` system call.
- **Inputs**:
    - `sock`: The socket descriptor on which the option is to be set.
    - `level`: The protocol level at which the option resides.
    - `optname`: The name of the option to be set.
    - `optval`: A pointer to the buffer containing the option value.
    - `optlen`: The size of the option value buffer.
- **Control Flow**:
    - Calls the `setsockopt` function with the provided parameters to set the specified socket option.
    - On Windows, the `optval` is cast to a `const char*` before passing to `setsockopt`.
    - Returns true if `setsockopt` returns 0, indicating success; otherwise, returns false.
- **Output**: Returns a boolean indicating whether the socket option was successfully set.


---
### set\_socket\_opt<!-- {{#callable:detail::set_socket_opt}} -->
Sets a socket option by calling an implementation function with the specified parameters.
- **Inputs**:
    - `sock`: The socket descriptor for which the option is being set.
    - `level`: The protocol level at which the option resides.
    - `optname`: The name of the option to be set.
    - `optval`: The value to set for the specified option.
- **Control Flow**:
    - The function directly calls [`set_socket_opt_impl`](#detailset_socket_opt_impl) with the provided parameters.
    - It passes the address of `optval` and its size to the implementation function.
- **Output**: Returns a boolean indicating the success or failure of setting the socket option.
- **Functions called**:
    - [`detail::set_socket_opt_impl`](#detailset_socket_opt_impl)


---
### set\_socket\_opt\_time<!-- {{#callable:detail::set_socket_opt_time}} -->
Sets the socket option for timeout using specified seconds and microseconds.
- **Inputs**:
    - `sock`: The socket descriptor for which the option is being set.
    - `level`: The level at which the option is defined, typically `SOL_SOCKET` or a protocol level.
    - `optname`: The specific option name to be set, such as `SO_RCVTIMEO` or `SO_SNDTIMEO`.
    - `sec`: The timeout duration in seconds.
    - `usec`: The timeout duration in microseconds.
- **Control Flow**:
    - Checks if the platform is Windows using the `_WIN32` preprocessor directive.
    - If on Windows, calculates the timeout in milliseconds by converting seconds to milliseconds and adding the microseconds converted to milliseconds.
    - If not on Windows, initializes a `timeval` structure with seconds and microseconds.
    - Calls the [`set_socket_opt_impl`](#detailset_socket_opt_impl) function with the socket, level, option name, and a pointer to the timeout structure along with its size.
- **Output**: Returns a boolean indicating the success or failure of setting the socket option.
- **Functions called**:
    - [`detail::set_socket_opt_impl`](#detailset_socket_opt_impl)


---
### default\_socket\_options<!-- {{#callable:default_socket_options}} -->
Sets default socket options for the given socket.
- **Inputs**:
    - `sock`: A socket of type `socket_t` for which the default options are being set.
- **Control Flow**:
    - Calls `detail::set_socket_opt` to configure socket options.
    - Uses conditional compilation to determine whether to set `SO_REUSEPORT` or `SO_REUSEADDR` based on the availability of `SO_REUSEPORT`.
- **Output**: This function does not return a value; it configures the socket options directly.


---
### status\_message<!-- {{#callable:status_message}} -->
The `status_message` function returns a string representation of an HTTP status code.
- **Inputs**:
    - `status`: An integer representing the HTTP status code.
- **Control Flow**:
    - The function uses a `switch` statement to evaluate the `status` input.
    - For each case that matches a specific HTTP status code, it returns the corresponding status message as a string.
    - If the `status` does not match any defined cases, it defaults to returning 'Internal Server Error' for the `500` status code.
- **Output**: The function outputs a constant string that describes the HTTP status associated with the provided status code.


---
### get\_bearer\_token\_auth<!-- {{#callable:get_bearer_token_auth}} -->
Extracts the bearer token from the 'Authorization' header of a given request.
- **Inputs**:
    - `req`: A constant reference to a `Request` object that contains HTTP headers.
- **Control Flow**:
    - Checks if the `Request` object has an 'Authorization' header.
    - If the header exists, it calculates the length of the 'Bearer ' prefix.
    - Extracts the value of the 'Authorization' header and removes the 'Bearer ' prefix from it.
    - If the header does not exist, returns an empty string.
- **Output**: Returns the bearer token as a string if present, otherwise returns an empty string.


---
### to\_string<!-- {{#callable:to_string}} -->
Converts an `Error` enum value to its corresponding string representation.
- **Inputs**:
    - `error`: An `Error` enum value representing the type of error that occurred.
- **Control Flow**:
    - The function uses a `switch` statement to evaluate the `error` input.
    - For each case in the `switch`, it returns a specific string message corresponding to the `Error` value.
    - If the `error` does not match any defined cases, it falls through to the default case.
    - If no valid case is matched, it returns 'Invalid'.
- **Output**: A `std::string` that describes the error represented by the `Error` enum value.


---
### operator<<<!-- {{#callable:operator<<}} -->
Overloads the `<<` operator to enable outputting an `Error` object to an output stream.
- **Inputs**:
    - `os`: A reference to an `std::ostream` object where the `Error` object will be output.
    - `obj`: A constant reference to an `Error` object that is to be converted to a string and output.
- **Control Flow**:
    - Calls `to_string(obj)` to convert the `Error` object to its string representation and outputs it to the stream.
    - Outputs the underlying integer value of the `Error` object, enclosed in parentheses, to the stream.
    - Returns the modified output stream to allow for chaining of output operations.
- **Output**: Returns a reference to the modified `std::ostream` object, allowing for further output operations to be performed on the same stream.
- **Functions called**:
    - [`to_string`](#to_string)


---
### u8string\_to\_wstring<!-- {{#callable:detail::u8string_to_wstring}} -->
Converts a UTF-8 encoded C-style string to a wide string (UTF-16) format.
- **Inputs**:
    - `s`: A pointer to a null-terminated UTF-8 encoded C-style string.
- **Control Flow**:
    - Calculates the length of the input string `s` using `strlen`.
    - Determines the required size for the wide string using `MultiByteToWideChar` with a null output to get the size.
    - If the required size is greater than zero, resizes the wide string `ws` to the calculated size.
    - Calls `MultiByteToWideChar` again to perform the actual conversion from UTF-8 to wide string, storing the result in `ws`.
    - If the conversion does not fill the entire allocated size, the wide string `ws` is cleared.
- **Output**: Returns a `std::wstring` containing the converted wide string, or an empty string if the conversion fails.


---
### is\_token\_char<!-- {{#callable:detail::fields::is_token_char}} -->
Determines if a character is considered a valid token character.
- **Inputs**:
    - `c`: A character to be checked for validity as a token character.
- **Control Flow**:
    - The function evaluates the character `c` against a series of conditions.
    - It checks if `c` is alphanumeric using `std::isalnum(c)`.
    - If `c` is not alphanumeric, it checks if `c` matches any of the specified special characters.
- **Output**: Returns `true` if `c` is a valid token character, otherwise returns `false`.


---
### is\_token<!-- {{#callable:detail::fields::is_token}} -->
Checks if a given string consists entirely of valid token characters.
- **Inputs**:
    - `s`: A constant reference to a string that is to be checked for token validity.
- **Control Flow**:
    - The function first checks if the input string `s` is empty; if it is, the function returns false.
    - It then iterates over each character in the string `s` and checks if it is a valid token character using the [`is_token_char`](#fieldsis_token_char) function.
    - If any character is found that is not a valid token character, the function returns false immediately.
    - If all characters are valid, the function returns true after completing the iteration.
- **Output**: Returns a boolean value indicating whether the string `s` is a valid token (true) or not (false).
- **Functions called**:
    - [`detail::fields::is_token_char`](#fieldsis_token_char)


---
### is\_field\_name<!-- {{#callable:detail::fields::is_field_name}} -->
Checks if the provided string is a valid field name by verifying it as a token.
- **Inputs**:
    - `s`: A constant reference to a `std::string` representing the field name to be validated.
- **Control Flow**:
    - The function directly calls the [`is_token`](#fieldsis_token) function with the input string `s`.
    - The result of the [`is_token`](#fieldsis_token) function is returned as the output of `is_field_name`.
- **Output**: Returns a boolean value indicating whether the input string `s` is a valid field name (true) or not (false).
- **Functions called**:
    - [`detail::fields::is_token`](#fieldsis_token)


---
### is\_vchar<!-- {{#callable:detail::fields::is_vchar}} -->
Determines if a given character is a visible ASCII character.
- **Inputs**:
    - `c`: A character of type `char` that is to be checked for visibility.
- **Control Flow**:
    - The function checks if the input character `c` falls within the ASCII range of visible characters, which is from 33 to 126 inclusive.
    - It returns `true` if `c` is within this range, otherwise it returns `false`.
- **Output**: Returns a boolean value indicating whether the character `c` is a visible ASCII character.


---
### is\_obs\_text<!-- {{#callable:detail::fields::is_obs_text}} -->
Determines if a character is an observable text character based on its ASCII value.
- **Inputs**:
    - `c`: A character of type `char` that is to be evaluated.
- **Control Flow**:
    - The function uses a static cast to convert the character `c` to an `unsigned char`.
    - It then checks if the value of the character is greater than or equal to 128.
- **Output**: Returns a boolean value: `true` if the character is an observable text character (ASCII value >= 128), otherwise `false`.


---
### is\_field\_vchar<!-- {{#callable:detail::fields::is_field_vchar}} -->
Determines if a character is a valid vchar or obs-text character.
- **Inputs**:
    - `c`: A character to be checked for validity as a vchar or obs-text.
- **Control Flow**:
    - Calls the [`is_vchar`](#fieldsis_vchar) function to check if the character is a valid vchar.
    - Calls the [`is_obs_text`](#fieldsis_obs_text) function to check if the character is a valid obs-text character.
    - Returns true if either check is true, otherwise returns false.
- **Output**: Returns a boolean value indicating whether the character `c` is a valid vchar or obs-text character.
- **Functions called**:
    - [`detail::fields::is_vchar`](#fieldsis_vchar)
    - [`detail::fields::is_obs_text`](#fieldsis_obs_text)


---
### is\_field\_content<!-- {{#callable:detail::fields::is_field_content}} -->
Determines if a given string consists of valid field content based on specific character rules.
- **Inputs**:
    - `s`: A constant reference to a string that represents the content to be validated.
- **Control Flow**:
    - Checks if the string `s` is empty, returning true if it is.
    - If `s` has a size of 1, it checks if the single character is a valid field character using [`is_field_vchar`](#fieldsis_field_vchar) function.
    - If `s` has a size of 2, it checks both characters for validity using [`is_field_vchar`](#fieldsis_field_vchar) function.
    - For strings longer than 2 characters, it initializes an index `i` and checks the first character for validity.
    - It then enters a loop to check each character in the string (except the last one) to see if it is either a space, a tab, or a valid field character.
    - Finally, it checks the last character of the string for validity using [`is_field_vchar`](#fieldsis_field_vchar) function.
- **Output**: Returns a boolean value indicating whether the string `s` contains valid field content according to the defined rules.
- **Functions called**:
    - [`detail::fields::is_field_vchar`](#fieldsis_field_vchar)


---
### is\_field\_value<!-- {{#callable:detail::fields::is_field_value}} -->
Checks if the provided string is a valid field value by delegating to the [`is_field_content`](#fieldsis_field_content) function.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that represents the value to be checked.
- **Control Flow**:
    - The function directly calls the [`is_field_content`](#fieldsis_field_content) function with the input string `s`.
    - The result of the [`is_field_content`](#fieldsis_field_content) function call is returned as the output of `is_field_value`.
- **Output**: Returns a boolean value indicating whether the input string `s` is considered a valid field value based on the logic defined in [`is_field_content`](#fieldsis_field_content).
- **Functions called**:
    - [`detail::fields::is_field_content`](#fieldsis_field_content)


---
### is\_hex<!-- {{#callable:is_hex}} -->
Determines if a character is a valid hexadecimal digit and converts it to its integer value.
- **Inputs**:
    - `c`: A character to be checked if it is a hexadecimal digit.
    - `v`: An integer reference that will store the converted value of the hexadecimal digit if `c` is valid.
- **Control Flow**:
    - Checks if the character `c` is a digit (0-9) and assigns its integer value to `v`.
    - If `c` is an uppercase letter (A-F), it calculates the corresponding integer value and assigns it to `v`.
    - If `c` is a lowercase letter (a-f), it similarly calculates the integer value and assigns it to `v`.
    - If none of the conditions are met, the function returns false.
- **Output**: Returns true if `c` is a valid hexadecimal digit and sets `v` to its integer value; otherwise, returns false.


---
### from\_hex\_to\_i<!-- {{#callable:from_hex_to_i}} -->
Converts a substring of a hexadecimal string to an integer.
- **Inputs**:
    - `s`: A constant reference to a string containing the hexadecimal representation.
    - `i`: The starting index in the string `s` from which to begin conversion.
    - `cnt`: The number of hexadecimal characters to convert.
    - `val`: A reference to an integer where the converted value will be stored.
- **Control Flow**:
    - The function first checks if the starting index `i` is out of bounds of the string `s`. If it is, the function returns false immediately.
    - The function initializes `val` to 0 to prepare for the conversion process.
    - A loop iterates `cnt` times, processing one character at a time from the string `s` starting at index `i`.
    - Within the loop, it checks if the current character is valid (not null). If it is null, the function returns false.
    - The function calls [`is_hex`](#is_hex) to determine if the current character is a valid hexadecimal digit and retrieves its integer value in `v`.
    - If the character is valid, it updates `val` by multiplying the current value by 16 and adding the integer value of the hexadecimal digit.
    - If any character is invalid, the function returns false after the loop ends.
- **Output**: Returns true if the conversion was successful, otherwise returns false.
- **Functions called**:
    - [`is_hex`](#is_hex)


---
### from\_i\_to\_hex<!-- {{#callable:from_i_to_hex}} -->
Converts a non-negative integer to its hexadecimal string representation.
- **Inputs**:
    - `n`: A non-negative integer of type `size_t` that is to be converted to a hexadecimal string.
- **Control Flow**:
    - The function initializes a static character set containing hexadecimal digits.
    - It enters a `do-while` loop that continues as long as `n` is greater than zero.
    - Within the loop, it appends the hexadecimal character corresponding to the last 4 bits of `n` to the result string `ret`.
    - The integer `n` is right-shifted by 4 bits to process the next set of 4 bits in the subsequent iteration.
- **Output**: Returns a string representing the hexadecimal format of the input integer `n`.


---
### to\_utf8<!-- {{#callable:to_utf8}} -->
Converts a Unicode code point to its UTF-8 representation and stores it in a buffer.
- **Inputs**:
    - `code`: An integer representing a Unicode code point to be converted.
    - `buff`: A pointer to a character array where the UTF-8 encoded bytes will be stored.
- **Control Flow**:
    - Checks if the `code` is less than 0x0080, and if so, stores the single byte representation in `buff` and returns 1.
    - If `code` is between 0x0080 and 0x0800, it calculates and stores the two-byte UTF-8 representation in `buff` and returns 2.
    - For `code` values between 0x0800 and 0xD800, it stores the three-byte UTF-8 representation in `buff` and returns 3.
    - If `code` is between 0xD800 and 0xE000, it recognizes this as an invalid range and returns 0.
    - For `code` values between 0xE000 and 0x10000, it stores the three-byte UTF-8 representation in `buff` and returns 3.
    - If `code` is between 0x10000 and 0x110000, it stores the four-byte UTF-8 representation in `buff` and returns 4.
    - If none of the conditions are met, it returns 0, indicating an invalid code point.
- **Output**: Returns the number of bytes written to the buffer, which can be 1, 2, 3, or 4 for valid code points, or 0 for invalid code points.


---
### base64\_encode<!-- {{#callable:base64_encode}} -->
Encodes a given string into its Base64 representation.
- **Inputs**:
    - `in`: A constant reference to a string that contains the data to be encoded.
- **Control Flow**:
    - A static lookup string is defined to map 6-bit values to Base64 characters.
    - An output string is initialized and reserved with the size of the input string.
    - Two variables, `val` and `valb`, are initialized to hold the current value and bit count respectively.
    - The function iterates over each character in the input string, updating `val` by shifting its bits and adding the ASCII value of the character.
    - When `valb` is 6 or more, the function extracts the next 6 bits from `val`, maps it to a Base64 character using the lookup string, and appends it to the output string.
    - After processing all characters, if there are remaining bits in `valb`, it encodes the last bits into a Base64 character.
    - Finally, the output string is padded with '=' characters to ensure its length is a multiple of 4.
- **Output**: Returns a Base64 encoded string representation of the input string.


---
### is\_valid\_path<!-- {{#callable:is_valid_path}} -->
Checks if a given file path is valid by analyzing its components.
- **Inputs**:
    - `path`: A constant reference to a string representing the file path to be validated.
- **Control Flow**:
    - Initializes two size_t variables, `level` to track the depth of directory traversal and `i` to iterate through the characters of the path.
    - Skips any leading slashes in the path.
    - Enters a loop that continues until the end of the path is reached, processing each component of the path separated by slashes.
    - For each component, checks for invalid characters (null character or backslash) and calculates the length of the component.
    - Uses assertions to ensure that each component has a non-zero length.
    - Handles special cases for the current directory (`.`) and parent directory (`..`), adjusting the `level` accordingly.
    - Skips any trailing slashes after processing each component.
    - Returns true if the path is valid after processing all components.
- **Output**: Returns a boolean value indicating whether the path is valid (true) or not (false).


---
### FileStat<!-- {{#callable:FileStat::FileStat}} -->
Constructs a `FileStat` object by retrieving file status information for a given file path.
- **Inputs**:
    - `path`: A constant reference to a `std::string` representing the file path for which the status is to be retrieved.
- **Control Flow**:
    - Checks if the platform is Windows using a preprocessor directive.
    - If on Windows, converts the UTF-8 file path to a wide string and calls `_wstat` to retrieve the file status.
    - If not on Windows, directly calls `stat` with the file path to retrieve the file status.
- **Output**: The function does not return a value but initializes the `FileStat` object with the file status information in the `st_` member variable.
- **Functions called**:
    - [`detail::u8string_to_wstring`](#detailu8string_to_wstring)


---
### is\_file<!-- {{#callable:FileStat::is_file}} -->
Determines if the file represented by the `FileStat` object is a regular file.
- **Inputs**: None
- **Control Flow**:
    - Checks if the member variable `ret_` is greater than or equal to 0, indicating a successful file status retrieval.
    - Uses the macro `S_ISREG` to check if the file mode stored in `st_.st_mode` indicates a regular file.
- **Output**: Returns a boolean value: true if the file is a regular file, false otherwise.


---
### is\_dir<!-- {{#callable:FileStat::is_dir}} -->
Determines if the file represented by the `FileStat` object is a directory.
- **Inputs**:
    - `this`: A constant reference to the `FileStat` object on which the method is called.
- **Control Flow**:
    - The method checks if the `ret_` member variable is greater than or equal to zero, indicating a successful file status retrieval.
    - It then checks the `st_mode` member of the `st_` structure to determine if the file type is a directory using the `S_ISDIR` macro.
- **Output**: Returns a boolean value: true if the file is a directory, false otherwise.


---
### encode\_query\_param<!-- {{#callable:encode_query_param}} -->
Encodes a string by replacing non-alphanumeric characters with their percent-encoded representation.
- **Inputs**:
    - `value`: The input string that needs to be encoded for use in a URL query parameter.
- **Control Flow**:
    - Initializes a `std::ostringstream` object to build the encoded string.
    - Iterates over each character in the input string `value`.
    - Checks if the character is alphanumeric or one of the allowed special characters.
    - If the character is allowed, it is appended directly to the output stream.
    - If the character is not allowed, it is percent-encoded by converting it to its hexadecimal representation and prepending a '%' character.
    - The hexadecimal representation is formatted to be uppercase and two digits wide.
- **Output**: Returns the encoded string as a `std::string`, which can safely be used in a URL query.


---
### encode\_url<!-- {{#callable:encode_url}} -->
Encodes a given string into a URL-encoded format.
- **Inputs**:
    - `s`: A constant reference to a string that needs to be URL-encoded.
- **Control Flow**:
    - The function initializes an empty string `result` and reserves space based on the size of the input string `s`.
    - It iterates over each character in the input string `s` using a for loop.
    - For each character, a switch statement checks for specific characters that need to be replaced with their URL-encoded equivalents.
    - If a character matches one of the cases (like space or newline), it appends the corresponding encoded string to `result`.
    - For characters not explicitly handled, it checks if the character is a non-ASCII character (greater than 0x80) and encodes it in hexadecimal format.
    - If the character is ASCII, it is appended directly to `result`.
- **Output**: Returns a string that represents the URL-encoded version of the input string `s`.


---
### decode\_url<!-- {{#callable:decode_url}} -->
Decodes a URL-encoded string by converting percent-encoded characters and optionally converting plus signs to spaces.
- **Inputs**:
    - `s`: The URL-encoded string to be decoded.
    - `convert_plus_to_space`: A boolean flag indicating whether to convert plus signs ('+') to spaces (' ').
- **Control Flow**:
    - Iterates through each character in the input string `s`.
    - Checks for percent-encoded characters starting with '%'.
    - If a 'u' follows '%', it processes a Unicode code point using [`from_hex_to_i`](#from_hex_to_i) and converts it to UTF-8.
    - If two hexadecimal digits follow '%', it converts them to a character.
    - If `convert_plus_to_space` is true and a '+' is encountered, it appends a space to the result.
    - Otherwise, it appends the current character to the result.
- **Output**: Returns the decoded string with all percent-encoded characters and optional plus signs converted.
- **Functions called**:
    - [`from_hex_to_i`](#from_hex_to_i)
    - [`to_utf8`](#to_utf8)


---
### file\_extension<!-- {{#callable:file_extension}} -->
Extracts the file extension from a given file path.
- **Inputs**:
    - `path`: A constant reference to a string representing the file path from which the extension is to be extracted.
- **Control Flow**:
    - A regular expression is defined to match a file extension pattern at the end of the string.
    - The function checks if the provided `path` matches the regular expression using `std::regex_search`.
    - If a match is found, the captured group (the file extension) is returned as a string.
    - If no match is found, an empty string is returned.
- **Output**: Returns the file extension as a string if found; otherwise, returns an empty string.


---
### is\_space\_or\_tab<!-- {{#callable:is_space_or_tab}} -->
Determines if a given character is a space or a tab.
- **Inputs**:
    - `c`: A character to be checked if it is a space or a tab.
- **Control Flow**:
    - The function evaluates the input character `c` against two conditions: whether it is equal to a space character (' ') or a tab character ('\t').
    - The result of the evaluation is returned as a boolean value.
- **Output**: Returns `true` if the character is a space or a tab, otherwise returns `false`.


---
### trim<!-- {{#callable:trim}} -->
The `trim` function calculates the new left and right indices for a range of characters, removing leading and trailing whitespace or tab characters.
- **Inputs**:
    - `b`: A pointer to the beginning of the character array to be trimmed.
    - `e`: A pointer to the end of the character array, marking the boundary for trimming.
    - `left`: An initial index indicating the starting position for trimming from the left.
    - `right`: An initial index indicating the starting position for trimming from the right.
- **Control Flow**:
    - The first `while` loop increments the `left` index as long as it points to a whitespace or tab character and does not exceed the end pointer `e`.
    - The second `while` loop decrements the `right` index as long as it points to a whitespace or tab character and is greater than zero.
- **Output**: The function returns a `std::pair<size_t, size_t>` containing the updated `left` and `right` indices after trimming.
- **Functions called**:
    - [`is_space_or_tab`](#is_space_or_tab)


---
### trim\_copy<!-- {{#callable:trim_copy}} -->
`trim_copy` returns a new string that is a trimmed copy of the input string.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that represents the input string to be trimmed.
- **Control Flow**:
    - The function calls [`trim`](#trim) with the data pointers of the input string and its size to determine the range of characters that are not whitespace.
    - It uses the result from [`trim`](#trim) to create a substring of the original string, effectively removing leading and trailing whitespace.
- **Output**: A new `std::string` that contains the trimmed version of the input string.
- **Functions called**:
    - [`trim`](#trim)


---
### trim\_double\_quotes\_copy<!-- {{#callable:trim_double_quotes_copy}} -->
Removes leading and trailing double quotes from a string if present.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that may contain double quotes at the beginning and end.
- **Control Flow**:
    - Checks if the length of the string `s` is at least 2 characters.
    - Verifies if the first character of `s` is a double quote (`'"'`) and the last character is also a double quote.
    - If both conditions are true, returns a substring of `s` that excludes the first and last characters.
    - If the conditions are not met, returns the original string `s` unchanged.
- **Output**: Returns a new `std::string` that is the input string `s` without leading and trailing double quotes, or the original string if no quotes are present.


---
### divide<!-- {{#callable:divide}} -->
This function overloads the [`divide`](#divide) function to accept a `std::string` and a character delimiter, invoking a callback function with the string data.
- **Inputs**:
    - `str`: A constant reference to a `std::string` that contains the data to be divided.
    - `d`: A character that serves as the delimiter for dividing the string.
    - `fn`: A callable function object that takes two pointers to characters and their respective sizes as parameters.
- **Control Flow**:
    - The function first retrieves the underlying character data and size of the `std::string` using `str.data()` and `str.size()`.
    - It then calls another overloaded version of [`divide`](#divide), passing the character data, size, delimiter, and the callback function `fn` after moving it.
- **Output**: This function does not return a value; instead, it invokes the provided callback function with the divided segments of the string.
- **Functions called**:
    - [`divide`](#divide)


---
### split<!-- {{#callable:split}} -->
The `split` function divides a string into segments based on a specified delimiter and applies a callback function to each segment.
- **Inputs**:
    - `b`: A pointer to the beginning of the string to be split.
    - `e`: A pointer to the end of the string; if null, the string is assumed to be null-terminated.
    - `d`: The delimiter character used to split the string.
    - `m`: The maximum number of segments to be created.
    - `fn`: A callback function that takes two pointers as arguments, which will be called for each segment found.
- **Control Flow**:
    - The function initializes indices and a count for segments.
    - It enters a loop that continues until the end of the string is reached, checking either the end pointer or for a null terminator.
    - Within the loop, it checks if the current character matches the delimiter and if the segment count is less than the maximum allowed.
    - If a delimiter is found, it calls the [`trim`](#trim) function to get the start and end indices of the segment, and if valid, invokes the callback function with the segment.
    - The beginning index is updated to the character after the delimiter, and the segment count is incremented.
    - After the loop, it processes any remaining segment from the last delimiter to the end of the string.
- **Output**: The function does not return a value; instead, it calls the provided callback function for each segment of the string that is found.
- **Functions called**:
    - [`trim`](#trim)


---
### stream\_line\_reader<!-- {{#callable:stream_line_reader::stream_line_reader}} -->
Constructs a `stream_line_reader` object with a specified stream and a fixed buffer.
- **Inputs**:
    - `strm`: A reference to a `Stream` object that will be used for reading lines.
    - `fixed_buffer`: A pointer to a character array that serves as a fixed buffer for reading data.
    - `fixed_buffer_size`: A size_t value representing the size of the fixed buffer.
- **Control Flow**:
    - The constructor initializes the member variables `strm_`, `fixed_buffer_`, and `fixed_buffer_size_` with the provided arguments.
    - No complex control flow or logic is present in this constructor; it simply assigns values.
- **Output**: The constructor does not return a value but initializes a `stream_line_reader` object for subsequent use.


---
### ptr<!-- {{#callable:stream_line_reader::ptr}} -->
Returns a pointer to the current data in the buffer, either from a fixed buffer or a growable buffer.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `stream_line_reader` class.
- **Control Flow**:
    - Checks if the `growable_buffer_` is empty.
    - If it is empty, returns a pointer to `fixed_buffer_`.
    - If it is not empty, returns a pointer to the data in `growable_buffer_`.
- **Output**: A pointer to a constant character array representing the current buffer data.


---
### size<!-- {{#callable:mmap::size}} -->
Returns the size of the `mmap` object.
- **Inputs**:
    - `this`: A constant reference to the `mmap` object from which the size is being retrieved.
- **Control Flow**:
    - The function directly accesses the private member variable `size_` of the `mmap` class.
    - It returns the value of `size_` without any additional computation or checks.
- **Output**: The function outputs the size of the `mmap` object as a `size_t` value.


---
### end\_with\_crlf<!-- {{#callable:stream_line_reader::end_with_crlf}} -->
Checks if the last two characters of the stream are a carriage return followed by a newline.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `stream_line_reader` class.
- **Control Flow**:
    - Calculates the pointer to the end of the stream by adding the current size to the base pointer.
    - Checks if the size of the stream is at least 2 characters.
    - Verifies if the character two positions before the end is a carriage return ('\r') and the last character is a newline ('\n').
- **Output**: Returns a boolean value indicating whether the stream ends with a CRLF sequence.
- **Functions called**:
    - [`stream_line_reader::ptr`](#stream_line_readerptr)
    - [`stream_line_reader::size`](#stream_line_readersize)


---
### getline<!-- {{#callable:stream_line_reader::getline}} -->
Reads a line from the input stream and stores it in a buffer, handling line termination based on configuration.
- **Inputs**:
    - `this`: A reference to the `stream_line_reader` object that contains the input stream and buffer.
- **Control Flow**:
    - Initializes the buffer size and clears the growable buffer.
    - Checks if the current line length exceeds a predefined maximum length, returning false if it does.
    - Enters a loop to read bytes one at a time from the input stream.
    - Handles different cases for the number of bytes read: negative values indicate an error, zero indicates end-of-stream, and positive values are appended to the buffer.
    - Checks for line termination based on the configuration: either on newline characters or carriage return followed by newline.
    - Exits the loop when a line terminator is found or the end of the stream is reached.
- **Output**: Returns true if a line was successfully read and false if an error occurred or the end of the stream was reached without reading any bytes.
- **Functions called**:
    - [`stream_line_reader::size`](#stream_line_readersize)
    - [`stream_line_reader::append`](#stream_line_readerappend)


---
### append<!-- {{#callable:stream_line_reader::append}} -->
Appends a character to a buffer, either a fixed-size buffer or a growable buffer.
- **Inputs**:
    - `c`: The character to be appended to the buffer.
- **Control Flow**:
    - Checks if the current size of the fixed buffer is less than its maximum size minus one.
    - If there is space in the fixed buffer, appends the character and null-terminates the buffer.
    - If the fixed buffer is full and the growable buffer is empty, it copies the contents of the fixed buffer to the growable buffer.
    - Finally, appends the character to the growable buffer.
- **Output**: The function does not return a value; it modifies the internal state of the `stream_line_reader` object by updating the buffer.


---
### mmap<!-- {{#callable:mmap::mmap}} -->
The `mmap` constructor initializes an instance by opening a file specified by the given path.
- **Inputs**:
    - `path`: A pointer to a constant character string representing the file path to be opened.
- **Control Flow**:
    - The constructor calls the [`open`](#mmapopen) method with the provided `path` argument to open the specified file.
- **Output**: The constructor does not return a value; it initializes the `mmap` object by opening the specified file.
- **Functions called**:
    - [`mmap::open`](#mmapopen)


---
### \~mmap<!-- {{#callable:mmap::~mmap}} -->
Destructor for the `mmap` class that ensures resources are properly released.
- **Inputs**: None
- **Control Flow**:
    - The destructor `~mmap` is called when an object of the `mmap` class goes out of scope or is explicitly deleted.
    - It invokes the `close()` method to release any resources associated with the `mmap` object.
- **Output**: This function does not return a value; it performs cleanup by closing resources.
- **Functions called**:
    - [`mmap::close`](#mmapclose)


---
### open<!-- {{#callable:mmap::open}} -->
The `open` function attempts to open a file specified by the given path and maps it into memory.
- **Inputs**:
    - `path`: A pointer to a constant character string representing the file path to be opened.
- **Control Flow**:
    - The function first calls `close()` to ensure any previously opened file is closed.
    - It checks if the platform is Windows or not, and executes the corresponding file opening logic.
    - For Windows, it converts the `path` to a wide string and attempts to create a file handle using `CreateFile2` or `CreateFileW`.
    - If the file handle is invalid, it returns false.
    - It retrieves the file size using `GetFileSizeEx` and checks if it exceeds the maximum size for `size_`.
    - It creates a file mapping object using `CreateFileMappingFromApp` or `CreateFileMappingW`.
    - If the mapping is NULL and the file size is zero, it marks the file as empty and returns true.
    - If the mapping is NULL, it closes the file and returns false.
    - It maps the file into memory using `MapViewOfFileFromApp` or `MapViewOfFile`.
    - If the mapping fails and the file size is zero, it marks the file as empty and returns false.
    - Finally, if all operations succeed, it returns true.
- **Output**: Returns a boolean value indicating whether the file was successfully opened and mapped into memory.
- **Functions called**:
    - [`mmap::close`](#mmapclose)
    - [`detail::u8string_to_wstring`](#detailu8string_to_wstring)


---
### is\_open<!-- {{#callable:mmap::is_open}} -->
Checks if the memory-mapped file is open.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `mmap` class.
- **Control Flow**:
    - Evaluates the boolean condition `is_open_empty_file` to determine if the file is considered open when empty.
    - Returns `true` if `is_open_empty_file` is true; otherwise, checks if `addr_` is not a null pointer and returns the result.
- **Output**: Returns a boolean value indicating whether the memory-mapped file is open.


---
### data<!-- {{#callable:mmap::data}} -->
Returns a pointer to the data of the memory-mapped file or an empty string if the file is not open.
- **Inputs**:
    - `is_open_empty_file`: A boolean indicating whether the memory-mapped file is open and empty.
    - `addr_`: A pointer to the memory address of the mapped file data.
- **Control Flow**:
    - The function checks the value of `is_open_empty_file`.
    - If `is_open_empty_file` is true, it returns an empty string.
    - Otherwise, it returns a pointer cast from `addr_` to a `const char *`.
- **Output**: A pointer to the data of the memory-mapped file or an empty string if the file is not open.


---
### close<!-- {{#callable:mmap::close}} -->
Closes the memory-mapped file and releases associated resources.
- **Inputs**:
    - `None`: The function does not take any input arguments.
- **Control Flow**:
    - Checks if the memory address `addr_` is not null and unmaps the view of the file if it is valid.
    - Sets `addr_` to nullptr after unmapping.
    - On Windows, checks if the handle `hMapping_` is valid and closes it, setting it to NULL.
    - Checks if the file handle `hFile_` is valid and closes it, setting it to INVALID_HANDLE_VALUE.
    - On non-Windows systems, checks if the file descriptor `fd_` is valid and closes it, setting it to -1.
    - Resets the size of the memory-mapped file to 0.
- **Output**: The function does not return a value; it performs cleanup operations on the resources associated with the memory-mapped file.


---
### close\_socket<!-- {{#callable:close_socket}} -->
Closes a socket connection based on the operating system.
- **Inputs**:
    - `sock`: A socket descriptor of type `socket_t` that represents the socket to be closed.
- **Control Flow**:
    - Checks if the operating system is Windows using the `_WIN32` preprocessor directive.
    - If on Windows, it calls the `closesocket` function to close the socket.
    - If not on Windows, it calls the [`close`](#mmapclose) function to close the socket.
- **Output**: Returns an integer indicating the success or failure of the socket closure operation.
- **Functions called**:
    - [`mmap::close`](#mmapclose)


---
### handle\_EINTR<!-- {{#callable:handle_EINTR}} -->
The `handle_EINTR` function repeatedly calls a provided function until it succeeds or returns an error that is not due to an interrupted system call.
- **Inputs**:
    - `fn`: A callable object (function, lambda, etc.) that returns a `ssize_t` value, which is the result of the operation being performed.
- **Control Flow**:
    - The function initializes a variable `res` to store the result of the callable `fn`.
    - It enters an infinite loop where it calls `fn()` and assigns the result to `res`.
    - If `res` is negative and `errno` is set to `EINTR`, indicating that the call was interrupted, it sleeps for 1 microsecond and continues the loop.
    - If `res` is non-negative or `errno` is not `EINTR`, it breaks out of the loop.
- **Output**: The function returns the result of the last successful call to `fn`, which is of type `ssize_t`.


---
### read\_socket<!-- {{#callable:read_socket}} -->
Reads data from a socket using the `recv` function while handling interruptions.
- **Inputs**:
    - `sock`: A socket descriptor of type `socket_t` from which data will be read.
    - `ptr`: A pointer to a buffer where the received data will be stored.
    - `size`: The maximum number of bytes to read from the socket.
    - `flags`: Flags that modify the behavior of the `recv` function.
- **Control Flow**:
    - The function begins by calling [`handle_EINTR`](#handle_EINTR), which is a utility to handle interruptions during the `recv` call.
    - Inside [`handle_EINTR`](#handle_EINTR), a lambda function is defined that encapsulates the call to `recv`.
    - The `recv` function is called with the provided socket, buffer pointer, size, and flags, with platform-specific adjustments for Windows.
    - The result of the `recv` call is returned from the lambda function, which is then returned by [`handle_EINTR`](#handle_EINTR).
- **Output**: Returns the number of bytes received on success, or -1 on error, with the error being handled by [`handle_EINTR`](#handle_EINTR).
- **Functions called**:
    - [`handle_EINTR`](#handle_EINTR)


---
### send\_socket<!-- {{#callable:send_socket}} -->
Sends data over a socket while handling interrupted system calls.
- **Inputs**:
    - `sock`: A socket descriptor of type `socket_t` representing the socket to which data will be sent.
    - `ptr`: A pointer to the data that needs to be sent.
    - `size`: The size of the data to be sent, specified in bytes.
    - `flags`: Flags that modify the behavior of the `send` function.
- **Control Flow**:
    - The function begins by calling [`handle_EINTR`](#handle_EINTR), which is a utility designed to handle interrupted system calls.
    - Inside [`handle_EINTR`](#handle_EINTR), a lambda function is defined that encapsulates the call to the `send` function.
    - The `send` function is invoked with the provided socket, data pointer, size, and flags, with a conditional compilation directive to handle Windows-specific type casting.
    - The result of the `send` function call is returned from the lambda, which is then returned by [`handle_EINTR`](#handle_EINTR).
- **Output**: Returns the number of bytes sent on success, or -1 on error, with the error code set appropriately.
- **Functions called**:
    - [`handle_EINTR`](#handle_EINTR)


---
### poll\_wrapper<!-- {{#callable:poll_wrapper}} -->
The `poll_wrapper` function provides a cross-platform interface for polling file descriptors.
- **Inputs**:
    - `fds`: A pointer to an array of `pollfd` structures that specify the file descriptors to be monitored.
    - `nfds`: The number of file descriptors in the `fds` array.
    - `timeout`: The maximum time, in milliseconds, to wait for an event on the file descriptors.
- **Control Flow**:
    - The function checks if the code is being compiled on a Windows platform using the `_WIN32` preprocessor directive.
    - If on Windows, it calls the `WSAPoll` function to poll the file descriptors.
    - If not on Windows, it calls the standard `poll` function to perform the same operation.
- **Output**: The function returns the number of file descriptors that have events pending, or -1 if an error occurred.


---
### select\_impl<!-- {{#callable:select_impl}} -->
The `select_impl` function monitors a socket for readiness to read or write, based on the specified timeout.
- **Inputs**:
    - `sock`: A socket descriptor of type `socket_t` that is being monitored.
    - `sec`: A `time_t` value representing the number of seconds for the timeout.
    - `usec`: A `time_t` value representing the number of microseconds for the timeout.
- **Control Flow**:
    - A `pollfd` structure is initialized with the socket descriptor and the event type (either `POLLIN` for reading or `POLLOUT` for writing).
    - The timeout is calculated in milliseconds by converting seconds to milliseconds and adding the microseconds converted to milliseconds.
    - The function calls [`handle_EINTR`](#handle_EINTR), which wraps the [`poll_wrapper`](#poll_wrapper) function to handle interruptions, passing the `pollfd` structure and the timeout.
- **Output**: Returns the number of file descriptors that are ready for the requested operation, or -1 if an error occurs.
- **Functions called**:
    - [`handle_EINTR`](#handle_EINTR)
    - [`poll_wrapper`](#poll_wrapper)


---
### select\_read<!-- {{#callable:select_read}} -->
The `select_read` function invokes `select_impl` to perform a read operation on a specified socket with a timeout.
- **Inputs**:
    - `sock`: A `socket_t` type representing the socket on which the read operation is to be performed.
    - `sec`: A `time_t` value representing the number of seconds to wait for data to become available.
    - `usec`: A `time_t` value representing the number of microseconds to wait for data to become available.
- **Control Flow**:
    - The function directly calls `select_impl` with the parameters passed to `select_read`.
    - The `select_impl` function is templated and is expected to handle the actual logic of selecting the socket for reading.
- **Output**: The function returns the result of the `select_impl` call, which is expected to be of type `ssize_t`, indicating the number of bytes read or an error code.


---
### select\_write<!-- {{#callable:select_write}} -->
The `select_write` function invokes `select_impl` to manage socket write operations with specified timeout.
- **Inputs**:
    - `sock`: A `socket_t` type representing the socket on which the write operation is to be performed.
    - `sec`: A `time_t` value representing the number of seconds to wait before timing out.
    - `usec`: A `time_t` value representing the number of microseconds to wait before timing out.
- **Control Flow**:
    - The function directly calls `select_impl` with the provided socket and timeout parameters.
    - The `select_impl` function is templated and is called with a `false` argument, indicating a specific behavior for write operations.
- **Output**: The function returns the result of the `select_impl` call, which is of type `ssize_t`, indicating the status of the write operation.


---
### wait\_until\_socket\_is\_ready<!-- {{#callable:wait_until_socket_is_ready}} -->
The `wait_until_socket_is_ready` function checks if a socket is ready for reading or writing within a specified timeout period.
- **Inputs**:
    - `sock`: A socket descriptor of type `socket_t` that represents the socket to be monitored.
    - `sec`: A `time_t` value representing the number of seconds to wait before timing out.
    - `usec`: A `time_t` value representing the number of microseconds to wait before timing out.
- **Control Flow**:
    - A `pollfd` structure is initialized to monitor the specified socket for both reading and writing events.
    - The timeout for the poll operation is calculated in milliseconds from the provided seconds and microseconds.
    - The [`poll_wrapper`](#poll_wrapper) function is called to check the socket's readiness, handling any interruptions from system calls.
    - If the poll result indicates a timeout (0), the function returns `Error::ConnectionTimeout`.
    - If the poll indicates that the socket is ready (greater than 0), it checks for errors using `getsockopt`.
    - If no errors are found, the function returns `Error::Success`; otherwise, it returns `Error::Connection`.
    - If the poll result is neither a timeout nor a successful readiness check, it defaults to returning `Error::Connection`.
- **Output**: The function returns an `Error` enum value indicating the result of the socket readiness check, which can be `Error::Success`, `Error::Connection`, or `Error::ConnectionTimeout`.
- **Functions called**:
    - [`handle_EINTR`](#handle_EINTR)
    - [`poll_wrapper`](#poll_wrapper)


---
### is\_socket\_alive<!-- {{#callable:is_socket_alive}} -->
Checks if a socket is alive by attempting to read from it.
- **Inputs**:
    - `sock`: A socket of type `socket_t` that is being checked for its status.
- **Control Flow**:
    - Calls `detail::select_read` with the socket and zero timeout to check if the socket is ready for reading.
    - If `select_read` returns 0, it indicates the socket is alive, and the function returns true.
    - If `select_read` returns a negative value and `errno` is set to `EBADF`, it indicates the socket is invalid, and the function returns false.
    - If the socket is valid, it attempts to read from the socket using `detail::read_socket` with the `MSG_PEEK` flag to check if there is data available without removing it from the queue.
    - If `read_socket` returns a positive value, it indicates the socket is alive; otherwise, it returns false.
- **Output**: Returns a boolean value indicating whether the socket is alive (true) or not (false).


---
### keep\_alive<!-- {{#callable:keep_alive}} -->
The `keep_alive` function checks if a socket is still active by monitoring its readability within a specified timeout period.
- **Inputs**:
    - `svr_sock`: An atomic reference to a server socket of type `socket_t`, which is used to determine if the server socket is still valid.
    - `sock`: A socket of type `socket_t` that is being monitored for activity.
    - `keep_alive_timeout_sec`: A time duration in seconds that specifies how long to wait before considering the socket inactive.
- **Control Flow**:
    - The function first checks if the socket is readable using [`select_read`](#select_read) and returns true immediately if it is.
    - It initializes a start time and a timeout duration based on the provided `keep_alive_timeout_sec`.
    - A loop is entered where it continuously checks the state of the socket until either the server socket is invalid, an error occurs, or the timeout is reached.
    - If the socket is readable, the function returns true; if the timeout is exceeded, the loop breaks and the function returns false.
- **Output**: The function returns a boolean value: true if the socket is still active and ready for reading, and false if it has timed out or encountered an error.
- **Functions called**:
    - [`select_read`](#select_read)


---
### process\_server\_socket\_core<!-- {{#callable:process_server_socket_core}} -->
Processes a server socket by maintaining a keep-alive mechanism and invoking a callback until conditions are met.
- **Inputs**:
    - `svr_sock`: An atomic reference to the server socket used for connection management.
    - `sock`: The socket identifier for the current connection being processed.
    - `keep_alive_max_count`: The maximum number of keep-alive attempts allowed.
    - `keep_alive_timeout_sec`: The timeout duration in seconds for each keep-alive attempt.
    - `callback`: A callable object that is invoked with the connection status.
- **Control Flow**:
    - Asserts that the maximum keep-alive count is greater than zero.
    - Initializes a return value to false and sets a counter to the maximum keep-alive count.
    - Enters a while loop that continues as long as the count is greater than zero and the keep-alive check is successful.
    - Determines if the current iteration is the last keep-alive attempt.
    - Calls the provided callback with the connection status and checks the return value.
    - Breaks the loop if the callback indicates failure or if the connection is closed.
    - Decrements the keep-alive count after each iteration.
- **Output**: Returns a boolean indicating the success of the keep-alive process and callback execution.
- **Functions called**:
    - [`keep_alive`](#keep_alive)


---
### process\_server\_socket<!-- {{#callable:process_server_socket}} -->
Processes a server socket by invoking a callback with a `SocketStream` and connection state.
- **Inputs**:
    - `svr_sock`: An atomic reference to the server socket of type `socket_t`.
    - `sock`: The socket of type `socket_t` that will be processed.
    - `keep_alive_max_count`: The maximum number of keep-alive messages to send.
    - `keep_alive_timeout_sec`: The timeout duration in seconds for keep-alive messages.
    - `read_timeout_sec`: The read timeout duration in seconds.
    - `read_timeout_usec`: The read timeout duration in microseconds.
    - `write_timeout_sec`: The write timeout duration in seconds.
    - `write_timeout_usec`: The write timeout duration in microseconds.
    - `callback`: A callable object that takes a `SocketStream`, a boolean indicating whether to close the connection, and a reference to a boolean indicating if the connection is closed.
- **Control Flow**:
    - Calls [`process_server_socket_core`](#process_server_socket_core) with the provided socket and timeout parameters.
    - Defines a lambda function that creates a `SocketStream` with the specified read and write timeouts.
    - The lambda function is passed to the `callback`, which processes the `SocketStream` and manages connection state.
- **Output**: Returns a boolean indicating the success or failure of processing the server socket.
- **Functions called**:
    - [`process_server_socket_core`](#process_server_socket_core)


---
### process\_client\_socket<!-- {{#callable:process_client_socket}} -->
Processes a client socket by creating a `SocketStream` and invoking a callback function.
- **Inputs**:
    - `sock`: A socket identifier of type `socket_t` representing the client socket to be processed.
    - `read_timeout_sec`: The read timeout in seconds.
    - `read_timeout_usec`: The read timeout in microseconds.
    - `write_timeout_sec`: The write timeout in seconds.
    - `write_timeout_usec`: The write timeout in microseconds.
    - `max_timeout_msec`: The maximum timeout in milliseconds.
    - `start_time`: A `std::chrono::time_point` representing the start time for the operation.
    - `callback`: A function that takes a `Stream` reference and returns a boolean, used to process the `SocketStream`.
- **Control Flow**:
    - A `SocketStream` object is instantiated with the provided socket and timeout parameters.
    - The `callback` function is called with the `SocketStream` object as an argument.
    - The result of the `callback` function is returned as the output of `process_client_socket`.
- **Output**: Returns a boolean value indicating the success or failure of the callback execution.


---
### shutdown\_socket<!-- {{#callable:shutdown_socket}} -->
This function gracefully shuts down a socket connection.
- **Inputs**:
    - `sock`: A socket descriptor of type `socket_t` that represents the socket to be shut down.
- **Control Flow**:
    - The function checks if the code is being compiled on a Windows platform using the `_WIN32` preprocessor directive.
    - If on Windows, it calls the [`shutdown`](#ThreadPoolshutdown) function with the socket and the `SD_BOTH` option to close both sending and receiving on the socket.
    - If not on Windows, it calls the [`shutdown`](#ThreadPoolshutdown) function with the socket and the `SHUT_RDWR` option, which serves the same purpose of closing both directions.
- **Output**: Returns an integer indicating the success or failure of the shutdown operation, where a return value of 0 typically indicates success.
- **Functions called**:
    - [`ThreadPool::shutdown`](#ThreadPoolshutdown)


---
### escape\_abstract\_namespace\_unix\_domain<!-- {{#callable:escape_abstract_namespace_unix_domain}} -->
Escapes the abstract namespace in a Unix domain socket path by replacing the leading null character with an '@'.
- **Inputs**:
    - `s`: A constant reference to a string that represents the Unix domain socket path.
- **Control Flow**:
    - Checks if the size of the input string `s` is greater than 1 and if the first character is a null character ('\0').
    - If the condition is true, it creates a copy of `s`, replaces the first character with '@', and returns the modified string.
    - If the condition is false, it simply returns the original string `s` unchanged.
- **Output**: Returns a modified string with the leading null character replaced by '@' if applicable, otherwise returns the original string.


---
### unescape\_abstract\_namespace\_unix\_domain<!-- {{#callable:unescape_abstract_namespace_unix_domain}} -->
This function removes the leading '@' character from a string if it is present.
- **Inputs**:
    - `s`: A constant reference to a string that may contain an abstract namespace Unix domain socket representation.
- **Control Flow**:
    - The function first checks if the input string `s` has more than one character and if the first character is '@'.
    - If both conditions are met, it creates a copy of the string, replaces the first character with a null character, and returns this modified string.
    - If the conditions are not met, it simply returns the original string unchanged.
- **Output**: The function returns a string that has the leading '@' character removed if it was present; otherwise, it returns the original string.


---
### create\_socket<!-- {{#callable:create_socket}} -->
`create_socket` creates a socket based on provided parameters and either binds or connects it.
- **Inputs**:
    - `host`: A string representing the hostname to resolve.
    - `ip`: A string representing the IP address to use, if provided.
    - `port`: An integer representing the port number to use.
    - `address_family`: An integer specifying the address family (e.g., AF_INET, AF_INET6, AF_UNIX).
    - `socket_flags`: An integer representing flags for socket creation.
    - `tcp_nodelay`: A boolean indicating whether to set the TCP_NODELAY option.
    - `ipv6_v6only`: A boolean indicating whether to restrict IPv6 sockets to only IPv6 addresses.
    - `socket_options`: A callable that applies additional socket options.
    - `bind_or_connect`: A callable that either binds or connects the socket.
- **Control Flow**:
    - Initializes `addrinfo` structure and sets hints for address resolution.
    - Checks if an IP address is provided; if so, sets the node to the IP and configures hints accordingly.
    - If the address family is AF_UNIX, it creates a UNIX domain socket and handles specific options.
    - If the address family is not AF_UNIX, it resolves the hostname or IP address using `getaddrinfo`.
    - Iterates through the resolved addresses to create a socket for each address.
    - Sets socket options based on the provided parameters (e.g., TCP_NODELAY, IPV6_V6ONLY).
    - Calls the `bind_or_connect` function to either bind or connect the socket, returning the socket if successful.
    - Cleans up resources and returns `INVALID_SOCKET` if no valid socket could be created.
- **Output**: Returns a valid socket descriptor if successful, or `INVALID_SOCKET` if an error occurs during socket creation or binding/connecting.
- **Functions called**:
    - [`unescape_abstract_namespace_unix_domain`](#unescape_abstract_namespace_unix_domain)
    - [`close_socket`](#close_socket)
    - [`detail::set_socket_opt`](#detailset_socket_opt)


---
### set\_nonblocking<!-- {{#callable:set_nonblocking}} -->
Sets the specified socket to non-blocking mode based on the provided boolean flag.
- **Inputs**:
    - `sock`: The socket descriptor to be modified.
    - `nonblocking`: A boolean flag indicating whether to set the socket to non-blocking mode (true) or blocking mode (false).
- **Control Flow**:
    - Checks if the platform is Windows using the `_WIN32` preprocessor directive.
    - If on Windows, it sets the socket to non-blocking mode using `ioctlsocket` with the `FIONBIO` command.
    - If not on Windows, it retrieves the current flags of the socket using `fcntl`, then modifies the flags to set or clear the `O_NONBLOCK` option based on the `nonblocking` parameter.
- **Output**: This function does not return a value; it modifies the state of the specified socket directly.


---
### is\_connection\_error<!-- {{#callable:is_connection_error}} -->
Determines if the last connection attempt resulted in an error.
- **Inputs**: None
- **Control Flow**:
    - Checks if the code is being compiled on a Windows platform using the `_WIN32` preprocessor directive.
    - If on Windows, it retrieves the last socket error using `WSAGetLastError()` and checks if it is not equal to `WSAEWOULDBLOCK`.
    - If not on Windows, it checks the global `errno` variable to see if it is not equal to `EINPROGRESS`.
- **Output**: Returns a boolean value indicating whether a connection error occurred.


---
### bind\_ip\_address<!-- {{#callable:bind_ip_address}} -->
The `bind_ip_address` function attempts to bind a socket to an IP address resolved from a given hostname.
- **Inputs**:
    - `sock`: A socket descriptor of type `socket_t` that will be bound to the resolved IP address.
    - `host`: A string representing the hostname to resolve for binding.
- **Control Flow**:
    - The function initializes a `struct addrinfo` to specify the criteria for address resolution.
    - It calls `getaddrinfo` to resolve the hostname into a linked list of `addrinfo` structures; if it fails, the function returns false.
    - A scope exit mechanism is set up to free the `addrinfo` structures after use.
    - The function iterates over the linked list of `addrinfo` structures, attempting to bind the socket to each address until a successful bind occurs or the list is exhausted.
    - If a successful bind is found, the loop breaks and the function prepares to return true.
- **Output**: The function returns a boolean value indicating whether the socket was successfully bound to an IP address (true) or not (false).


---
### if2ip<!-- {{#callable:if2ip}} -->
Converts a network interface name to its corresponding IP address based on the specified address family.
- **Inputs**:
    - `address_family`: An integer representing the address family (e.g., AF_INET for IPv4, AF_INET6 for IPv6, or AF_UNSPEC for both).
    - `ifn`: A string representing the name of the network interface whose IP address is to be retrieved.
- **Control Flow**:
    - Calls `getifaddrs` to retrieve the list of network interfaces and their addresses.
    - Uses a scope exit mechanism to ensure that the memory allocated for the interface addresses is freed after use.
    - Iterates through each interface address structure (`ifaddrs`) to find a match for the specified interface name (`ifn`).
    - Checks if the address is valid and matches the specified address family.
    - If the address family is IPv4, it converts the address to a string using `inet_ntop` and returns it.
    - If the address family is IPv6, it checks if the address is not link-local and processes it accordingly, returning the address if it is unique-local or otherwise storing it for later return.
- **Output**: Returns a string containing the IP address of the specified network interface, or an empty string if no valid address is found.


---
### create\_client\_socket<!-- {{#callable:create_client_socket}} -->
Creates a client socket and establishes a connection to a specified host and port.
- **Inputs**:
    - `host`: The hostname of the server to connect to.
    - `ip`: The IP address of the server to connect to.
    - `port`: The port number on the server to connect to.
    - `address_family`: The address family (e.g., AF_INET for IPv4, AF_INET6 for IPv6).
    - `tcp_nodelay`: A boolean indicating whether to disable Nagle's algorithm.
    - `ipv6_v6only`: A boolean indicating whether to restrict the socket to IPv6 addresses.
    - `socket_options`: Additional socket options to configure the socket.
    - `connection_timeout_sec`: The timeout duration in seconds for establishing a connection.
    - `connection_timeout_usec`: The timeout duration in microseconds for establishing a connection.
    - `read_timeout_sec`: The timeout duration in seconds for read operations.
    - `read_timeout_usec`: The timeout duration in microseconds for read operations.
    - `write_timeout_sec`: The timeout duration in seconds for write operations.
    - `write_timeout_usec`: The timeout duration in microseconds for write operations.
    - `intf`: The network interface to bind the socket to.
    - `error`: An Error reference to capture any errors that occur during socket creation.
- **Control Flow**:
    - Calls [`create_socket`](#create_socket) to initialize a socket with the provided parameters.
    - If the `intf` parameter is not empty, it attempts to bind the socket to the specified interface.
    - Sets the socket to non-blocking mode before attempting to connect.
    - Attempts to connect to the server using the provided address information.
    - If the connection fails, it checks for connection errors and waits until the socket is ready based on the specified timeout.
    - If the connection is successful, it sets the socket to blocking mode and configures read and write timeouts.
    - Returns the socket if successful, or an error code if the socket creation or connection fails.
- **Output**: Returns the created socket if successful, or an invalid socket value if an error occurred.
- **Functions called**:
    - [`create_socket`](#create_socket)
    - [`if2ip`](#if2ip)
    - [`bind_ip_address`](#bind_ip_address)
    - [`set_nonblocking`](#set_nonblocking)
    - [`is_connection_error`](#is_connection_error)
    - [`wait_until_socket_is_ready`](#wait_until_socket_is_ready)
    - [`detail::set_socket_opt_time`](#detailset_socket_opt_time)


---
### get\_ip\_and\_port<!-- {{#callable:get_ip_and_port}} -->
The `get_ip_and_port` function extracts the IP address and port number from a given socket address structure.
- **Inputs**:
    - `addr`: A constant reference to a `sockaddr_storage` structure that contains the socket address information.
    - `addr_len`: A `socklen_t` value representing the length of the address structure.
    - `ip`: A reference to a `std::string` where the extracted IP address will be stored.
    - `port`: A reference to an `int` where the extracted port number will be stored.
- **Control Flow**:
    - The function first checks the address family of the `addr` parameter to determine if it is IPv4 or IPv6.
    - If the address family is `AF_INET`, it extracts the port number from the `sin_port` field and converts it from network byte order to host byte order using `ntohs`.
    - If the address family is `AF_INET6`, it similarly extracts the port number from the `sin6_port` field.
    - If the address family is neither, the function returns false, indicating failure.
    - Next, it attempts to retrieve the IP address using `getnameinfo`, storing the result in a character array `ipstr`.
    - If `getnameinfo` fails, the function returns false.
    - If successful, the IP address is assigned to the `ip` string, and the function returns true.
- **Output**: The function returns a boolean value: true if the IP address and port were successfully extracted, and false otherwise.


---
### get\_local\_ip\_and\_port<!-- {{#callable:detail::SocketStream::get_local_ip_and_port}} -->
Retrieves the local IP address and port number associated with the socket.
- **Inputs**:
    - `ip`: A reference to a `std::string` where the local IP address will be stored.
    - `port`: A reference to an `int` where the local port number will be stored.
- **Control Flow**:
    - Calls the `detail::get_local_ip_and_port` function, passing the socket descriptor `sock_`, and the references to `ip` and `port`.
    - The function does not contain any additional logic or control structures; it directly forwards the call to the detail function.
- **Output**: The function does not return a value; instead, it modifies the `ip` and `port` parameters to hold the local IP address and port number.


---
### get\_remote\_ip\_and\_port<!-- {{#callable:detail::SocketStream::get_remote_ip_and_port}} -->
Retrieves the remote IP address and port number associated with the socket.
- **Inputs**:
    - `ip`: A reference to a `std::string` where the remote IP address will be stored.
    - `port`: A reference to an `int` where the remote port number will be stored.
- **Control Flow**:
    - The function directly calls `detail::get_remote_ip_and_port` with the socket descriptor `sock_`, and the references to `ip` and `port` as arguments.
    - There are no conditional statements or loops; the function simply forwards the call.
- **Output**: This function does not return a value; instead, it modifies the `ip` and `port` parameters to contain the remote IP address and port number.


---
### str2tag\_core<!-- {{#callable:str2tag_core}} -->
Recursively computes a hash value from a string by processing each character.
- **Inputs**:
    - `s`: A pointer to the character array (C-style string) to be hashed.
    - `l`: The length of the string to be processed.
    - `h`: The current hash value being computed, initialized to a starting value.
- **Control Flow**:
    - Checks if the length 'l' is zero; if so, returns the current hash 'h'.
    - If 'l' is not zero, it recursively calls itself with the next character in the string, decremented length, and a new hash value calculated from the current character.
- **Output**: Returns the computed hash value as an unsigned integer.


---
### str2tag<!-- {{#callable:str2tag}} -->
Converts a string into a tag represented as an unsigned integer.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that contains the input string to be converted into a tag.
- **Control Flow**:
    - The function calls [`str2tag_core`](#str2tag_core) with three arguments: the character data of the string, the size of the string, and an initial value of 0.
    - The [`str2tag_core`](#str2tag_core) function is expected to handle the actual conversion logic and return an unsigned integer representing the tag.
- **Output**: Returns an unsigned integer that represents the tag derived from the input string.
- **Functions called**:
    - [`str2tag_core`](#str2tag_core)


---
### operator""\_t<!-- {{#callable:udl::operator""_t}} -->
This function is a user-defined literal operator that converts a string literal into an unsigned integer tag.
- **Inputs**:
    - `s`: A pointer to a null-terminated character array (string literal) that represents the input string.
    - `l`: The size of the string literal, which indicates the number of characters in the string.
- **Control Flow**:
    - The function calls [`str2tag_core`](#str2tag_core) with the string pointer `s`, its length `l`, and an additional argument `0`.
    - The result of [`str2tag_core`](#str2tag_core) is returned as the output of the operator function.
- **Output**: The function returns an unsigned integer that represents a tag derived from the input string.
- **Functions called**:
    - [`str2tag_core`](#str2tag_core)


---
### find\_content\_type<!-- {{#callable:find_content_type}} -->
Determines the content type of a file based on its extension and user-defined mappings.
- **Inputs**:
    - `path`: A string representing the file path whose content type is to be determined.
    - `user_data`: A map that associates file extensions with specific content types defined by the user.
    - `default_content_type`: A string representing the default content type to return if the file extension is not recognized.
- **Control Flow**:
    - Extracts the file extension from the provided `path` using the [`file_extension`](#file_extension) function.
    - Checks if the extracted extension exists in the `user_data` map; if found, returns the corresponding content type.
    - If the extension is not found in `user_data`, it uses a switch statement to determine the content type based on predefined cases for various file extensions.
    - If the extension does not match any predefined cases, it returns the `default_content_type`.
- **Output**: Returns a string representing the content type of the file, either from the user-defined mappings, a predefined type based on the file extension, or a default content type.
- **Functions called**:
    - [`file_extension`](#file_extension)
    - [`str2tag`](#str2tag)


---
### can\_compress\_content\_type<!-- {{#callable:can_compress_content_type}} -->
Determines if a given content type can be compressed.
- **Inputs**:
    - `content_type`: A constant reference to a string representing the content type to be evaluated.
- **Control Flow**:
    - The function begins by converting the `content_type` string into a tag using the [`str2tag`](#str2tag) function.
    - A switch statement is used to evaluate the `tag` against predefined cases for specific content types.
    - If the `tag` matches any of the specified cases that can be compressed, the function returns true.
    - If the `tag` matches 'text/event-stream', the function returns false.
    - For any other content type, the function checks if the `content_type` starts with 'text/' and returns true if it does, otherwise returns false.
- **Output**: Returns a boolean value indicating whether the specified content type can be compressed.
- **Functions called**:
    - [`str2tag`](#str2tag)


---
### encoding\_type<!-- {{#callable:encoding_type}} -->
Determines the appropriate encoding type based on the request and response headers.
- **Inputs**:
    - `req`: A constant reference to a `Request` object containing the request headers.
    - `res`: A constant reference to a `Response` object containing the response headers.
- **Control Flow**:
    - Checks if the content type in the response can be compressed using `detail::can_compress_content_type`.
    - If compression is not possible, returns `EncodingType::None`.
    - Retrieves the 'Accept-Encoding' header from the request.
    - Checks for the presence of 'br' in the 'Accept-Encoding' header and returns `EncodingType::Brotli` if found.
    - Checks for the presence of 'gzip' in the 'Accept-Encoding' header and returns `EncodingType::Gzip` if found.
    - Checks for the presence of 'zstd' in the 'Accept-Encoding' header and returns `EncodingType::Zstd` if found.
    - If none of the encoding types are found, returns `EncodingType::None`.
- **Output**: Returns an `EncodingType` enumeration value indicating the preferred encoding type based on the request's 'Accept-Encoding' header.


---
### compress<!-- {{#callable:zstd_compressor::compress}} -->
Compresses a given data stream using Zstandard compression and invokes a callback with the compressed data.
- **Inputs**:
    - `data`: A pointer to the input data that needs to be compressed.
    - `data_length`: The length of the input data in bytes.
    - `last`: A boolean flag indicating if this is the last chunk of data to be compressed.
    - `callback`: A callback function that is called with the compressed data buffer and its size.
- **Control Flow**:
    - Initializes a buffer for compressed data and sets the compression mode based on the 'last' flag.
    - Enters a loop to compress the input data using Zstandard's streaming API.
    - Checks for errors during compression; if an error occurs, returns false.
    - Calls the provided callback with the compressed data; if the callback returns false, exits and returns false.
    - Determines if compression is finished based on the 'last' flag and the amount of data processed, continuing until all data is compressed.
- **Output**: Returns true if the compression was successful and all data was processed, otherwise returns false.


---
### gzip\_compressor<!-- {{#callable:gzip_compressor::gzip_compressor}} -->
Initializes a `gzip_compressor` object by setting up the zlib stream for compression.
- **Inputs**: None
- **Control Flow**:
    - The function begins by zeroing out the `strm_` member variable using `std::memset`.
    - It sets the `zalloc`, `zfree`, and `opaque` fields of the `strm_` to `Z_NULL`, indicating no custom memory allocation functions are provided.
    - The function then calls `deflateInit2` to initialize the compression stream with default parameters, checking if the initialization was successful.
- **Output**: The function does not return a value but sets the `is_valid_` member to indicate whether the initialization of the compression stream was successful.


---
### \~gzip\_compressor<!-- {{#callable:gzip_compressor::~gzip_compressor}} -->
The `gzip_compressor` destructor cleans up resources by calling `deflateEnd` on the compression stream.
- **Inputs**: None
- **Control Flow**:
    - The destructor is invoked when an instance of `gzip_compressor` goes out of scope or is explicitly deleted.
    - It calls the `deflateEnd` function, passing the reference to the internal compression stream `strm_` to properly release resources.
- **Output**: The function does not return a value; it ensures that the resources associated with the `gzip_compressor` instance are released.


---
### gzip\_decompressor<!-- {{#callable:gzip_decompressor::gzip_decompressor}} -->
Initializes a `gzip_decompressor` object by setting up the decompression stream.
- **Inputs**: None
- **Control Flow**:
    - The function begins by zeroing out the `strm_` member variable using `std::memset`.
    - It sets the `zalloc`, `zfree`, and `opaque` members of the `strm_` structure to `Z_NULL`, indicating no custom memory allocation functions are provided.
    - The function then calls `inflateInit2` with a window size of 15 and a flag to automatically detect the stream type, checking if the initialization was successful.
- **Output**: The function does not return a value but sets the `is_valid_` member to indicate whether the initialization of the decompression stream was successful.


---
### \~gzip\_decompressor<!-- {{#callable:gzip_decompressor::~gzip_decompressor}} -->
The `gzip_decompressor` destructor cleans up resources by ending the inflation process.
- **Inputs**: None
- **Control Flow**:
    - The destructor calls `inflateEnd` on the `strm_` member to release any resources associated with the inflation process.
- **Output**: The function does not return a value; it performs cleanup operations.


---
### is\_valid<!-- {{#callable:zstd_decompressor::is_valid}} -->
Checks if the decompressor context is valid.
- **Inputs**:
    - `this`: A constant reference to the current instance of the `zstd_decompressor` class.
- **Control Flow**:
    - The function directly evaluates the validity of the `ctx_` member variable.
    - It returns a boolean value based on whether `ctx_` is a non-null pointer.
- **Output**: Returns `true` if the `ctx_` member variable is not null, indicating a valid decompressor context; otherwise, returns `false`.


---
### decompress<!-- {{#callable:zstd_decompressor::decompress}} -->
The `decompress` function decompresses a given data stream using the Zstandard algorithm and processes the output through a callback function.
- **Inputs**:
    - `data`: A pointer to the compressed data that needs to be decompressed.
    - `data_length`: The size of the compressed data in bytes.
    - `callback`: A function that is called with the decompressed data buffer and its size after each decompression step.
- **Control Flow**:
    - An input buffer `ZSTD_inBuffer` is initialized with the compressed data and its length.
    - A loop runs as long as there is remaining data to decompress, where an output buffer `ZSTD_outBuffer` is prepared for decompressed data.
    - The `ZSTD_decompressStream` function is called to decompress a portion of the input data into the output buffer.
    - If an error occurs during decompression, the function returns false.
    - The callback function is invoked with the decompressed data and its size; if it returns false, the function also returns false.
    - The loop continues until all input data is processed, and if successful, the function returns true.
- **Output**: Returns true if the decompression and callback processing are successful; otherwise, returns false.


---
### brotli\_compressor<!-- {{#callable:brotli_compressor::brotli_compressor}} -->
The `brotli_compressor` constructor initializes a new instance of the Brotli encoder.
- **Inputs**:
    - `this`: A pointer to the current instance of the `brotli_compressor` class.
- **Control Flow**:
    - The constructor calls `BrotliEncoderCreateInstance` to create a new encoder instance.
    - The parameters for `BrotliEncoderCreateInstance` are all set to `nullptr`, indicating default behavior.
- **Output**: The constructor does not return a value but initializes the `state_` member with the encoder instance.


---
### \~brotli\_compressor<!-- {{#callable:brotli_compressor::~brotli_compressor}} -->
Destructor for the `brotli_compressor` class that cleans up the Brotli encoder instance.
- **Inputs**: None
- **Control Flow**:
    - The function calls `BrotliEncoderDestroyInstance` with the `state_` member variable to release resources associated with the Brotli encoder.
- **Output**: This function does not return a value; it performs cleanup by destroying the encoder instance.


---
### brotli\_decompressor<!-- {{#callable:brotli_decompressor::brotli_decompressor}} -->
Initializes a `brotli_decompressor` instance by creating a Brotli decoder.
- **Inputs**: None
- **Control Flow**:
    - Calls `BrotliDecoderCreateInstance` to create a decoder instance, passing null for the callbacks.
    - Checks if the decoder instance was created successfully; if not, sets the result to `BROTLI_DECODER_RESULT_ERROR`, otherwise sets it to `BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT`.
- **Output**: The function does not return a value but initializes the state of the `brotli_decompressor` instance with the result of the decoder creation.


---
### \~brotli\_decompressor<!-- {{#callable:brotli_decompressor::~brotli_decompressor}} -->
Destructor for the `brotli_decompressor` class that cleans up the decoder instance.
- **Inputs**:
    - `this`: A pointer to the current instance of the `brotli_decompressor` class.
- **Control Flow**:
    - Checks if the `decoder_s` member variable is not null.
    - If `decoder_s` is valid, it calls `BrotliDecoderDestroyInstance` to free the resources associated with the decoder.
- **Output**: This function does not return a value; it performs cleanup by destroying the decoder instance if it exists.


---
### zstd\_compressor<!-- {{#callable:zstd_compressor::zstd_compressor}} -->
Constructs a `zstd_compressor` object and initializes its compression context with a fast compression level.
- **Inputs**: None
- **Control Flow**:
    - Calls `ZSTD_createCCtx()` to create a new compression context and assigns it to `ctx_`.
    - Sets the compression level of the context to `ZSTD_fast` using `ZSTD_CCtx_setParameter()`.
- **Output**: The function does not return a value; it initializes the internal state of the `zstd_compressor` object.


---
### \~zstd\_compressor<!-- {{#callable:zstd_compressor::~zstd_compressor}} -->
Destructor for the `zstd_compressor` class that frees the compression context.
- **Inputs**: None
- **Control Flow**:
    - The destructor is invoked when an object of the `zstd_compressor` class goes out of scope or is explicitly deleted.
    - It calls the `ZSTD_freeCCtx` function to release the resources associated with the compression context stored in `ctx_`.
- **Output**: This function does not return a value; it performs cleanup by deallocating resources.


---
### zstd\_decompressor<!-- {{#callable:zstd_decompressor::zstd_decompressor}} -->
Constructs a `zstd_decompressor` object and initializes its decompression context.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes the `ctx_` member variable by calling `ZSTD_createDCtx()` to create a new decompression context.
- **Output**: The constructor does not return a value but initializes the internal state of the `zstd_decompressor` object.


---
### has\_header<!-- {{#callable:has_header}} -->
Checks if a specified header key exists in a collection of headers.
- **Inputs**:
    - `headers`: A constant reference to a collection of headers, typically a map or similar structure that stores key-value pairs.
    - `key`: A string representing the header key to search for in the headers collection.
- **Control Flow**:
    - The function uses the `find` method on the `headers` collection to search for the specified `key`.
    - It compares the result of `find` with `end()` to determine if the key exists in the collection.
- **Output**: Returns a boolean value: true if the key exists in the headers, false otherwise.


---
### get\_header\_value<!-- {{#callable:get_header_value}} -->
Retrieves the value associated with a specified key from a collection of headers, returning a default value if the key is not found.
- **Inputs**:
    - `headers`: A `Headers` object that contains key-value pairs representing the headers.
    - `key`: A `std::string` representing the key for which the header value is to be retrieved.
    - `def`: A pointer to a `const char` that serves as the default value to return if the key is not found.
    - `id`: A `size_t` representing the index of the value to retrieve in case of multiple values for the same key.
- **Control Flow**:
    - The function uses `equal_range` to find all entries in `headers` that match the specified `key`.
    - An iterator `it` is initialized to the first element of the range returned by `equal_range`.
    - The iterator is advanced by `id` positions to access the desired value.
    - If the advanced iterator is still within the bounds of the range, the corresponding value is returned as a C-style string.
    - If the iterator exceeds the range, the default value `def` is returned.
- **Output**: Returns a pointer to the C-style string of the header value if found; otherwise, returns the default value.


---
### parse\_header<!-- {{#callable:parse_header}} -->
Parses a header from a given character range and applies a function to the key-value pair if valid.
- **Inputs**:
    - `beg`: A pointer to the beginning of the header string.
    - `end`: A pointer to the end of the header string.
    - `fn`: A function that takes a key and a value as parameters to process the parsed header.
- **Control Flow**:
    - The function first trims trailing spaces and tabs from the end of the header.
    - It then searches for the first colon ':' to separate the key from the value.
    - If the key is not a valid field name, or if the colon is missing, the function returns false.
    - Leading spaces after the colon are skipped before extracting the value.
    - If the key is 'Location' or 'Referer', the value is passed directly to the function; otherwise, the value is URL-decoded before being passed.
- **Output**: Returns true if the header is successfully parsed and processed; otherwise, returns false.
- **Functions called**:
    - [`is_space_or_tab`](#is_space_or_tab)
    - [`decode_url`](#decode_url)


---
### read\_headers<!-- {{#callable:read_headers}} -->
The `read_headers` function reads HTTP headers from a stream and populates a `Headers` object.
- **Inputs**:
    - `strm`: A reference to a `Stream` object from which the headers are read.
    - `headers`: A reference to a `Headers` object where the parsed headers will be stored.
- **Control Flow**:
    - The function initializes a buffer and a line reader for the stream.
    - It enters an infinite loop to read lines from the stream.
    - For each line, it checks if the line ends with CRLF or LF to determine if it is a blank line, which indicates the end of headers.
    - If a blank line is detected, the loop breaks.
    - If the line exceeds the maximum header length, the function returns false.
    - The line terminator is excluded, and the line is parsed into key-value pairs using the [`parse_header`](#parse_header) function.
    - If parsing fails at any point, the function returns false.
- **Output**: The function returns a boolean value: true if headers were successfully read and parsed, or false if an error occurred during reading or parsing.
- **Functions called**:
    - [`parse_header`](#parse_header)


---
### read\_content\_with\_length<!-- {{#callable:read_content_with_length}} -->
Reads a specified number of bytes from a stream and processes them using a callback function while optionally reporting progress.
- **Inputs**:
    - `strm`: A reference to a `Stream` object from which data will be read.
    - `len`: A `uint64_t` value representing the total number of bytes to read from the stream.
    - `progress`: A `Progress` callback function that is called to report the current progress of the read operation.
    - `out`: A `ContentReceiverWithProgress` callback function that processes the data read from the stream.
- **Control Flow**:
    - Initializes a buffer of size `CPPHTTPLIB_RECV_BUFSIZ` to hold the data being read.
    - Enters a loop that continues until the total number of bytes read (`r`) is less than the specified length (`len`).
    - Calculates the remaining bytes to read and determines the number of bytes to read in the current iteration, ensuring it does not exceed the buffer size.
    - Reads data from the stream into the buffer and checks if the read operation was successful; if not, returns false.
    - Calls the `out` function to process the data read; if this function returns false, the read operation is aborted.
    - Updates the total number of bytes read (`r`) by adding the number of bytes successfully read in the current iteration.
    - If a `progress` callback is provided, it is called to report the current progress; if it returns false, the read operation is aborted.
- **Output**: Returns `true` if the specified number of bytes were successfully read and processed; otherwise, returns `false`.


---
### skip\_content\_with\_length<!-- {{#callable:skip_content_with_length}} -->
The `skip_content_with_length` function reads and discards a specified number of bytes from a given stream.
- **Inputs**:
    - `strm`: A reference to a `Stream` object from which data will be read.
    - `len`: A `uint64_t` value representing the number of bytes to skip in the stream.
- **Control Flow**:
    - The function initializes a buffer `buf` of size `CPPHTTPLIB_RECV_BUFSIZ` and a counter `r` to track the number of bytes read.
    - It enters a while loop that continues until the total bytes read `r` is less than the specified length `len`.
    - Within the loop, it calculates the remaining bytes to read and determines the number of bytes to read in the current iteration, ensuring it does not exceed the buffer size.
    - The function attempts to read from the stream into the buffer and checks if the number of bytes read `n` is less than or equal to zero, in which case it exits the function.
    - If bytes are successfully read, it increments the counter `r` by the number of bytes read.
- **Output**: The function does not return a value; it modifies the state of the stream by skipping the specified number of bytes.


---
### read\_content\_without\_length<!-- {{#callable:read_content_without_length}} -->
Reads data from a `Stream` and processes it using a `ContentReceiverWithProgress` callback until no more data is available.
- **Inputs**:
    - `strm`: A reference to a `Stream` object from which data will be read.
    - `out`: A `ContentReceiverWithProgress` callback function that processes the data read from the stream.
- **Control Flow**:
    - An infinite loop is initiated to continuously read data from the `Stream`.
    - Within the loop, data is read into a buffer (`buf`) of size `CPPHTTPLIB_RECV_BUFSIZ`.
    - If the number of bytes read (`n`) is zero, the function returns true, indicating successful completion.
    - If `n` is negative, indicating an error, the function returns false.
    - If data is successfully read, it is passed to the `out` callback along with the number of bytes read and the total bytes read so far.
    - If the `out` callback returns false, the function also returns false, indicating a failure in processing.
- **Output**: Returns true if all data was read and processed successfully, or false if an error occurred during reading or processing.


---
### read\_content\_chunked<!-- {{#callable:read_content_chunked}} -->
The `read_content_chunked` function reads and processes chunked data from a stream, extracting content and headers.
- **Inputs**:
    - `strm`: A reference to a `Stream` object from which chunked data is read.
    - `x`: A reference to an object of type `T` where extracted headers will be stored.
    - `out`: A `ContentReceiverWithProgress` callback used to handle the received content.
- **Control Flow**:
    - Initializes a buffer and a `stream_line_reader` to read lines from the stream.
    - Checks if the first line can be read; if not, returns false.
    - Enters a loop to read chunk sizes in hexadecimal format until a chunk size of zero is encountered.
    - For each chunk, reads the specified length of content and checks for errors.
    - Validates the end of each chunk with a CRLF sequence.
    - After processing all chunks, checks for any remaining headers and processes them until an empty line is found.
- **Output**: Returns true if the chunked content and headers are successfully read and processed; otherwise, returns false.
- **Functions called**:
    - [`read_content_with_length`](#read_content_with_length)
    - [`parse_header`](#parse_header)


---
### is\_chunked\_transfer\_encoding<!-- {{#callable:is_chunked_transfer_encoding}} -->
Determines if the `Transfer-Encoding` header in the provided `Headers` object indicates chunked transfer encoding.
- **Inputs**:
    - `headers`: A constant reference to a `Headers` object that contains HTTP headers.
- **Control Flow**:
    - Calls the [`get_header_value`](#get_header_value) function to retrieve the value of the `Transfer-Encoding` header from the `headers` object.
    - Uses the `case_ignore::equal` function to compare the retrieved header value with the string 'chunked', ignoring case sensitivity.
- **Output**: Returns a boolean value: `true` if the `Transfer-Encoding` header is 'chunked', and `false` otherwise.
- **Functions called**:
    - [`get_header_value`](#get_header_value)


---
### prepare\_content\_receiver<!-- {{#callable:prepare_content_receiver}} -->
The `prepare_content_receiver` function prepares a content receiver for processing data, optionally decompressing it based on the content encoding.
- **Inputs**:
    - `T &x`: A reference to an object of type T that provides access to the content header.
    - `int &status`: A reference to an integer that holds the status code of the operation.
    - `ContentReceiverWithProgress receiver`: A callable that processes the received content.
    - `bool decompress`: A boolean flag indicating whether to decompress the content.
    - `U callback`: A callback function that is called with the prepared content receiver.
- **Control Flow**:
    - The function first checks if decompression is requested by evaluating the `decompress` flag.
    - If decompression is needed, it retrieves the content encoding from the header of `x`.
    - Based on the encoding, it attempts to create an appropriate decompressor (gzip, Brotli, or zstd), checking for support via preprocessor directives.
    - If a valid decompressor is created, it defines a new content receiver that decompresses the data and forwards it to the original `receiver`.
    - If the decompressor is invalid, it sets the status to an internal server error and returns false.
    - If no decompression is needed, it simply forwards the data to the original `receiver` without modification.
    - Finally, it invokes the provided `callback` with the prepared content receiver.
- **Output**: The function returns a boolean indicating the success or failure of the operation, with the status code updated accordingly.


---
### read\_content<!-- {{#callable:read_content}} -->
The `read_content` function reads data from a stream into a specified object while handling various content transfer encodings and managing payload size.
- **Inputs**:
    - `strm`: A reference to a `Stream` object from which content will be read.
    - `x`: A reference to a template type `T` that will hold the read content.
    - `payload_max_length`: A size_t value representing the maximum allowed length of the payload.
    - `status`: An integer reference that will be updated to reflect the status of the read operation.
    - `progress`: A `Progress` object used to track the progress of the content reading.
    - `receiver`: A `ContentReceiverWithProgress` object that will receive the content as it is read.
    - `decompress`: A boolean indicating whether the content should be decompressed.
- **Control Flow**:
    - The function begins by calling [`prepare_content_receiver`](#prepare_content_receiver) to set up the content receiver with the provided parameters.
    - A lambda function is defined to handle the actual reading of content based on the transfer encoding and headers.
    - If the content is chunked, it calls [`read_content_chunked`](#read_content_chunked) to read the data.
    - If there is no 'Content-Length' header, it calls [`read_content_without_length`](#read_content_without_length) to read the data without a specified length.
    - If a 'Content-Length' header is present, it retrieves the length and checks if it exceeds the maximum allowed length.
    - If the length is valid and within limits, it calls [`read_content_with_length`](#read_content_with_length) to read the specified amount of data.
    - If any read operation fails, it updates the `status` variable to indicate the type of error encountered.
- **Output**: Returns a boolean indicating whether the content was successfully read from the stream.
- **Functions called**:
    - [`prepare_content_receiver`](#prepare_content_receiver)
    - [`is_chunked_transfer_encoding`](#is_chunked_transfer_encoding)
    - [`read_content_chunked`](#read_content_chunked)
    - [`has_header`](#has_header)
    - [`read_content_without_length`](#read_content_without_length)
    - [`detail::get_header_value_u64`](#detailget_header_value_u64)
    - [`skip_content_with_length`](#skip_content_with_length)
    - [`read_content_with_length`](#read_content_with_length)


---
### write\_request\_line<!-- {{#callable:write_request_line}} -->
Constructs an HTTP request line and writes it to the provided stream.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where the request line will be written.
    - `method`: A `std::string` representing the HTTP method (e.g., GET, POST).
    - `path`: A `std::string` representing the path of the resource being requested.
- **Control Flow**:
    - Initializes a `std::string` variable `s` with the value of `method`.
    - Appends a space, the `path`, and the HTTP version string ' HTTP/1.1\r\n' to `s`.
    - Calls the `write` method of the `Stream` object `strm`, passing the data and size of the constructed string `s`.
- **Output**: Returns the number of bytes written to the stream as a `ssize_t`.


---
### write\_response\_line<!-- {{#callable:write_response_line}} -->
The `write_response_line` function constructs an HTTP response line and writes it to a given stream.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where the response line will be written.
    - `status`: An integer representing the HTTP status code to be included in the response line.
- **Control Flow**:
    - The function begins by initializing a string `s` with the HTTP version prefix 'HTTP/1.1 '.
    - It appends the `status` code converted to a string to `s`.
    - A space character is added to `s` followed by the status message corresponding to the `status` code, retrieved using `httplib::status_message(status)`.
    - Finally, it appends a carriage return and newline sequence '\r\n' to `s` to complete the HTTP response line.
    - The constructed response line is then written to the provided `Stream` object using its `write` method, which returns the number of bytes written.
- **Output**: The function returns the number of bytes written to the stream, which corresponds to the length of the constructed HTTP response line.


---
### write\_headers<!-- {{#callable:write_headers}} -->
The `write_headers` function writes HTTP headers to a given stream.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where the headers will be written.
    - `headers`: A reference to a `Headers` object, which is a collection of key-value pairs representing the headers.
- **Control Flow**:
    - The function initializes a variable `write_len` to zero to keep track of the total number of bytes written.
    - It iterates over each key-value pair in the `headers` collection.
    - For each header, it constructs a string in the format 'key: value\r\n'.
    - It writes the constructed string to the `strm` using the `write` method and checks for errors.
    - If an error occurs during writing, it returns the error code immediately.
    - After writing all headers, it writes an additional '\r\n' to signify the end of the headers section.
    - Finally, it returns the total number of bytes written.
- **Output**: The function returns the total number of bytes written to the stream, or a negative value if an error occurred during writing.


---
### write\_data<!-- {{#callable:write_data}} -->
The `write_data` function writes a specified number of bytes from a given data buffer to a stream.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where the data will be written.
    - `d`: A pointer to a character array (data buffer) containing the data to be written.
    - `l`: A size_t value representing the total number of bytes to write from the data buffer.
- **Control Flow**:
    - Initializes an `offset` variable to track the number of bytes written.
    - Enters a while loop that continues until all bytes specified by `l` have been written.
    - Within the loop, calls the `write` method of the `Stream` object to write a portion of the data buffer.
    - Checks if the `write` method returns a negative value, indicating an error, and returns false if so.
    - Updates the `offset` by adding the number of bytes successfully written.
    - Exits the loop when all bytes have been written and returns true.
- **Output**: Returns a boolean value indicating the success of the write operation; true if all data was written successfully, false if an error occurred during writing.


---
### write\_content<!-- {{#callable:write_content}} -->
The [`write_content`](#write_content) function writes content to a stream using a specified content provider, offset, and length.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where the content will be written.
    - `content_provider`: A reference to a `ContentProvider` object that supplies the content to be written.
    - `offset`: A `size_t` value indicating the starting point in the content from which to begin writing.
    - `length`: A `size_t` value specifying the number of bytes to write from the content.
    - `is_shutting_down`: A constant reference to a type `T` that indicates whether the operation is shutting down.
- **Control Flow**:
    - The function initializes an `error` variable to `Error::Success`.
    - It then calls another overloaded version of [`write_content`](#write_content), passing along the same parameters along with the `error` variable.
- **Output**: The function returns a boolean value indicating the success or failure of the content writing operation.
- **Functions called**:
    - [`write_content`](#write_content)


---
### write\_content\_without\_length<!-- {{#callable:write_content_without_length}} -->
The `write_content_without_length` function writes data to a stream using a content provider until the data is no longer available or a shutdown condition is met.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where data will be written.
    - `content_provider`: A `ContentProvider` object that supplies data to be written to the stream.
    - `is_shutting_down`: A callable that returns a boolean indicating whether the system is shutting down.
- **Control Flow**:
    - Initialize local variables: `offset`, `data_available`, and `ok`.
    - Define a `data_sink` structure with three lambda functions for writing data, checking writability, and marking completion.
    - Enter a loop that continues while data is available and the system is not shutting down.
    - Within the loop, check if the stream is writable; if not, return false.
    - Call the `content_provider` with the current offset to get data; if it fails, return false.
    - Check the `ok` flag; if it is false, exit the loop and return false.
- **Output**: Returns a boolean indicating the success of the write operation, true if all data was written successfully, false otherwise.
- **Functions called**:
    - [`write_data`](#write_data)


---
### write\_content\_chunked<!-- {{#callable:write_content_chunked}} -->
This function initiates the process of writing content in chunks to a stream using a specified content provider.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where the content will be written.
    - `content_provider`: A reference to a `ContentProvider` object that supplies the content to be written.
    - `is_shutting_down`: A constant reference of type `T` that indicates whether the system is in the process of shutting down.
    - `compressor`: A reference of type `U` that is used for compressing the content before writing.
- **Control Flow**:
    - The function begins by initializing an `Error` variable to `Error::Success`.
    - It then calls another overloaded version of [`write_content_chunked`](#write_content_chunked), passing along the stream, content provider, shutdown status, compressor, and the error variable.
- **Output**: The function returns a boolean indicating the success or failure of the content writing operation.
- **Functions called**:
    - [`write_content_chunked`](#write_content_chunked)


---
### redirect<!-- {{#callable:redirect}} -->
The `redirect` function modifies a `Request` and `Response` object to handle HTTP redirection based on specified parameters.
- **Inputs**:
    - `cli`: A reference to a client object that handles sending requests.
    - `req`: A reference to the original `Request` object that is being modified for redirection.
    - `res`: A reference to the original `Response` object that will be updated based on the new request.
    - `path`: A string representing the new path for the redirected request.
    - `location`: A string representing the location to be set in the response if it is empty.
    - `error`: An `Error` object that captures any errors that occur during the request sending process.
- **Control Flow**:
    - A new `Request` object is created as a copy of the original `req` and its path is updated to the new `path`.
    - The redirect count of the new request is decremented to track the number of redirects remaining.
    - If the response status is 303 (See Other) and the original request method is not GET or HEAD, the method of the new request is changed to GET, and its body and headers are cleared.
    - A new `Response` object is created to store the response from the client after sending the new request.
    - The modified request is sent using the `cli.send` method, and the result is stored in `ret`.
    - If the request is successfully sent (`ret` is true), the original `req` and `res` are updated with the new request and response.
    - If the new response's location is empty, it is set to the provided `location`.
- **Output**: The function returns a boolean indicating whether the request was successfully sent and processed.


---
### params\_to\_query\_str<!-- {{#callable:params_to_query_str}} -->
Converts a `Params` object into a URL query string format.
- **Inputs**:
    - `params`: A constant reference to a `Params` object, which is a collection of key-value pairs representing query parameters.
- **Control Flow**:
    - Initializes an empty string `query` to build the query string.
    - Iterates over each key-value pair in the `params` collection using an iterator.
    - For each key-value pair, if it is not the first element, appends an ampersand '&' to separate query parameters.
    - Appends the key, an equals sign '=', and the encoded value (obtained by calling [`encode_query_param`](#encode_query_param) on the value) to the `query` string.
    - Returns the constructed query string after processing all key-value pairs.
- **Output**: A string representing the URL-encoded query string formed from the input `Params` object.
- **Functions called**:
    - [`encode_query_param`](#encode_query_param)


---
### parse\_query\_text<!-- {{#callable:parse_query_text}} -->
This function serves as a wrapper that calls another [`parse_query_text`](#parse_query_text) function with the data and size of the input string.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that contains the query text to be parsed.
    - `params`: A reference to a `Params` object that will be populated with the parsed query parameters.
- **Control Flow**:
    - The function takes the input string `s` and retrieves its underlying character data using `s.data()`.
    - It also obtains the size of the string using `s.size()`.
    - These values are then passed to another overloaded version of [`parse_query_text`](#parse_query_text) along with the `params` object.
- **Output**: This function does not return a value; instead, it modifies the `params` object in place based on the parsed query text.
- **Functions called**:
    - [`parse_query_text`](#parse_query_text)


---
### parse\_multipart\_boundary<!-- {{#callable:parse_multipart_boundary}} -->
The `parse_multipart_boundary` function extracts the boundary string from a given content type.
- **Inputs**:
    - `content_type`: A constant reference to a string representing the content type from which the boundary will be extracted.
    - `boundary`: A reference to a string where the extracted boundary will be stored.
- **Control Flow**:
    - The function defines a keyword 'boundary=' to search for in the `content_type` string.
    - It uses `find` to locate the position of the boundary keyword in the `content_type` string.
    - If the keyword is not found, the function returns false.
    - If found, it determines the end position of the boundary by searching for the next semicolon.
    - It calculates the beginning position of the boundary string by adding the length of the keyword to the found position.
    - The boundary string is extracted from the `content_type`, trimmed of double quotes, and stored in the `boundary` reference.
    - Finally, the function returns true if the boundary string is not empty.
- **Output**: The function returns a boolean indicating whether the boundary was successfully extracted and stored.
- **Functions called**:
    - [`trim_double_quotes_copy`](#trim_double_quotes_copy)


---
### parse\_disposition\_params<!-- {{#callable:parse_disposition_params}} -->
Parses a semicolon-separated string of key-value pairs and stores them in a `Params` object.
- **Inputs**:
    - `s`: A constant reference to a string containing the semicolon-separated key-value pairs.
    - `params`: A reference to a `Params` object where the parsed key-value pairs will be stored.
- **Control Flow**:
    - Initializes a set `cache` to keep track of already processed key-value pairs to avoid duplicates.
    - Splits the input string `s` by semicolons into individual key-value segments using a lambda function.
    - For each segment, checks if it has already been processed by looking it up in `cache`; if so, it skips further processing.
    - If the segment is new, it inserts it into `cache` and splits it further by the equals sign to separate the key and value.
    - Assigns the key and value from the split operation, ensuring that the first part is treated as the key and the second as the value.
    - Trims double quotes from both the key and value before storing them in the `params` object.
- **Output**: The function does not return a value; instead, it modifies the `params` object by adding the parsed key-value pairs.
- **Functions called**:
    - [`split`](#split)
    - [`trim_double_quotes_copy`](#trim_double_quotes_copy)


---
### parse\_range\_header<!-- {{#callable:parse_range_header::parse_range_header}} -->
Parses a range header string and populates a `Ranges` object with valid byte ranges.
- **Inputs**:
    - `s`: A string representing the range header to be parsed, expected to start with 'bytes='.
    - `ranges`: A reference to a `Ranges` object where valid parsed ranges will be stored.
- **Control Flow**:
    - Checks if the input string `s` starts with 'bytes=' and has a sufficient length.
    - Defines a lambda function `is_valid` to check if a string consists only of digits.
    - Splits the substring after 'bytes=' by commas to handle multiple ranges.
    - For each range, checks for the presence of a '-' character to separate the start and end values.
    - Validates both the start and end values using the `is_valid` function.
    - Converts the valid start and end values from strings to signed long long integers.
    - Checks for logical consistency of the ranges (e.g., start should not be greater than end).
    - If all ranges are valid and at least one range is found, returns true; otherwise, returns false.
- **Output**: Returns a boolean indicating whether the parsing was successful and at least one valid range was found.
- **Functions called**:
    - [`split`](#split)


---
### random\_string<!-- {{#callable:parse_range_header::random_string}} -->
Generates a random alphanumeric string of a specified length.
- **Inputs**:
    - `length`: The desired length of the random string to be generated.
- **Control Flow**:
    - Defines a constant character array `data` containing alphanumeric characters.
    - Initializes a thread-local random number generator using a seed sequence derived from a random device.
    - Creates an empty string `result` to store the generated random characters.
    - Iterates `length` times, appending a randomly selected character from `data` to `result` during each iteration.
    - Returns the generated random string after the loop completes.
- **Output**: A `std::string` containing a randomly generated sequence of alphanumeric characters of the specified length.


---
### make\_multipart\_data\_boundary<!-- {{#callable:parse_range_header::make_multipart_data_boundary}} -->
Generates a unique multipart data boundary string for HTTP requests.
- **Inputs**: None
- **Control Flow**:
    - The function constructs a string by concatenating a fixed prefix with a randomly generated string.
    - It calls the `detail::random_string` function with an argument of 16 to generate the random part of the boundary.
- **Output**: Returns a string that serves as a unique boundary identifier for multipart HTTP data.


---
### is\_multipart\_boundary\_chars\_valid<!-- {{#callable:parse_range_header::is_multipart_boundary_chars_valid}} -->
Checks if the characters in a multipart boundary string are valid.
- **Inputs**:
    - `boundary`: A constant reference to a `std::string` representing the multipart boundary to be validated.
- **Control Flow**:
    - The function initializes a boolean variable `valid` to true.
    - It iterates over each character in the `boundary` string using a for loop.
    - For each character, it checks if the character is alphanumeric or one of the allowed characters ('-' or '_').
    - If an invalid character is found, `valid` is set to false and the loop is exited early.
    - Finally, the function returns the value of `valid`.
- **Output**: Returns a boolean indicating whether all characters in the `boundary` string are valid according to the specified criteria.


---
### serialize\_multipart\_formdata\_item\_begin<!-- {{#callable:parse_range_header::serialize_multipart_formdata_item_begin}} -->
The `serialize_multipart_formdata_item_begin` function generates the initial part of a multipart form-data item for HTTP requests.
- **Inputs**:
    - `item`: A reference to an object of type T that contains the properties 'name', 'filename', and 'content_type' used to construct the form-data header.
    - `boundary`: A string representing the boundary that separates different parts of the multipart form-data.
- **Control Flow**:
    - The function starts by initializing a string `body` with the boundary prefix followed by a carriage return and line feed.
    - It appends the 'Content-Disposition' header to `body`, including the 'name' from the `item`.
    - If the `item.filename` is not empty, it appends the filename to the 'Content-Disposition' header.
    - If the `item.content_type` is not empty, it appends the 'Content-Type' header to `body`.
    - Finally, it adds an additional carriage return and line feed to signify the end of the headers.
- **Output**: The function returns a string that represents the serialized headers for a multipart form-data item, ready to be sent in an HTTP request.


---
### serialize\_multipart\_formdata\_item\_end<!-- {{#callable:parse_range_header::serialize_multipart_formdata_item_end}} -->
This function returns the string representation of the end of a multipart form-data item.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a string literal without any conditional logic or loops.
- **Output**: The output is a string containing the characters '\r\n', which signifies the end of a multipart form-data item.


---
### serialize\_multipart\_formdata\_finish<!-- {{#callable:parse_range_header::serialize_multipart_formdata_finish}} -->
The `serialize_multipart_formdata_finish` function generates the closing boundary string for a multipart form data request.
- **Inputs**:
    - `boundary`: A string representing the boundary used to separate parts in a multipart form data.
- **Control Flow**:
    - The function concatenates the string '--' with the provided `boundary` and appends '--\r\n' to signify the end of the multipart data.
    - The resulting string is then returned as the output.
- **Output**: A string that represents the closing boundary of a multipart form data, formatted as '--<boundary>--\r\n'.


---
### serialize\_multipart\_formdata\_get\_content\_type<!-- {{#callable:parse_range_header::serialize_multipart_formdata_get_content_type}} -->
Generates the content type string for a multipart form-data request with a specified boundary.
- **Inputs**:
    - `boundary`: A string representing the boundary used to separate parts in the multipart form-data.
- **Control Flow**:
    - The function concatenates the string 'multipart/form-data; boundary=' with the provided `boundary` string.
    - It directly returns the resulting string without any conditional logic or loops.
- **Output**: A string that specifies the content type for a multipart form-data request, including the provided boundary.


---
### serialize\_multipart\_formdata<!-- {{#callable:parse_range_header::serialize_multipart_formdata}} -->
The `serialize_multipart_formdata` function constructs a multipart form data string from a collection of items.
- **Inputs**:
    - `items`: A collection of `MultipartFormDataItems` that contains the data to be serialized.
    - `boundary`: A string that serves as the boundary delimiter for the multipart form data.
    - `finish`: A boolean flag indicating whether to append the final boundary to the serialized output; defaults to true.
- **Control Flow**:
    - The function initializes an empty string `body` to accumulate the serialized data.
    - It iterates over each `item` in the `items` collection, appending the serialized beginning, content, and end of each item to `body`.
    - If the `finish` flag is true, it appends the final boundary to `body` using [`serialize_multipart_formdata_finish`](#parse_range_headerserialize_multipart_formdata_finish).
- **Output**: The function returns a string representing the complete serialized multipart form data.
- **Functions called**:
    - [`parse_range_header::serialize_multipart_formdata_item_begin`](#parse_range_headerserialize_multipart_formdata_item_begin)
    - [`parse_range_header::serialize_multipart_formdata_item_end`](#parse_range_headerserialize_multipart_formdata_item_end)
    - [`parse_range_header::serialize_multipart_formdata_finish`](#parse_range_headerserialize_multipart_formdata_finish)


---
### range\_error<!-- {{#callable:parse_range_header::range_error}} -->
Checks if the requested byte ranges in an HTTP request are valid according to specified rules.
- **Inputs**:
    - `req`: A reference to a `Request` object that contains the byte ranges requested by the client.
    - `res`: A reference to a `Response` object that contains the status code and content length of the response.
- **Control Flow**:
    - Checks if the `ranges` in the `req` object are not empty and if the response status is in the 200 range.
    - Calculates the content length from the response, defaulting to the size of the response body if necessary.
    - Validates the number of ranges against a maximum count to prevent denial-of-service attacks.
    - Iterates through each range in the request, adjusting the first and last positions based on the content length.
    - Checks if the ranges are within valid bounds and in ascending order.
    - Counts overlapping ranges and ensures there are no more than two overlapping ranges.
- **Output**: Returns true if any validation checks fail (indicating a range error), otherwise returns false.


---
### get\_range\_offset\_and\_length<!-- {{#callable:parse_range_header::get_range_offset_and_length}} -->
The `get_range_offset_and_length` function calculates the starting offset and length of a specified range within a given content length.
- **Inputs**:
    - `r`: A `Range` object representing the start and end indices of the range.
    - `content_length`: A `size_t` representing the total length of the content.
- **Control Flow**:
    - The function begins by asserting that the first and second elements of the range `r` are not equal to -1.
    - It then checks that the first element of the range is within the bounds of the content length and that it is less than the second element.
    - Finally, it returns a pair containing the first element of the range and the calculated length of the range, which is the difference between the second and first elements plus one.
- **Output**: The function outputs a `std::pair<size_t, size_t>` where the first element is the starting offset of the range and the second element is the length of the range.


---
### make\_content\_range\_header\_field<!-- {{#callable:parse_range_header::make_content_range_header_field}} -->
Generates a Content-Range HTTP header field based on the provided byte range and total content length.
- **Inputs**:
    - `offset_and_length`: A pair of size_t values representing the starting offset and the length of the content range.
    - `content_length`: A size_t value representing the total length of the content.
- **Control Flow**:
    - Extracts the starting offset from the first element of the `offset_and_length` pair.
    - Calculates the ending offset by adding the length to the starting offset and subtracting one.
    - Constructs the Content-Range header string by concatenating the starting offset, ending offset, and total content length.
- **Output**: Returns a string formatted as 'bytes <start>-<end>/<total>', representing the specified byte range and total content length.


---
### process\_multipart\_ranges\_data<!-- {{#callable:parse_range_header::process_multipart_ranges_data}} -->
Processes multipart range data from a request and generates corresponding content.
- **Inputs**:
    - `req`: A `Request` object containing the ranges to be processed.
    - `boundary`: A string representing the boundary used in the multipart data.
    - `content_type`: A string specifying the content type of the data.
    - `content_length`: A size_t representing the total length of the content.
    - `stoken`: A callable that processes string tokens.
    - `ctoken`: A callable that processes content tokens.
    - `content`: A callable that handles the content for each range.
- **Control Flow**:
    - Iterates over each range in the `req.ranges` vector.
    - For each range, it generates the multipart headers including the boundary and content type if provided.
    - Calculates the offset and length for the current range using [`get_range_offset_and_length`](#parse_range_headerget_range_offset_and_length).
    - Creates a `Content-Range` header using the calculated offset and length.
    - Calls the `content` function with the offset and length, returning false if it fails.
    - After processing all ranges, it appends the closing boundary.
- **Output**: Returns a boolean indicating the success of processing all ranges.
- **Functions called**:
    - [`parse_range_header::get_range_offset_and_length`](#parse_range_headerget_range_offset_and_length)
    - [`parse_range_header::make_content_range_header_field`](#parse_range_headermake_content_range_header_field)


---
### make\_multipart\_ranges\_data<!-- {{#callable:parse_range_header::make_multipart_ranges_data}} -->
The `make_multipart_ranges_data` function processes multipart range data from a `Request` and appends the results to a `Response` data string.
- **Inputs**:
    - `req`: A constant reference to a `Request` object containing the multipart data to be processed.
    - `res`: A reference to a `Response` object where the processed data will be appended.
    - `boundary`: A string representing the boundary used to separate different parts of the multipart data.
    - `content_type`: A string indicating the content type of the multipart data.
    - `content_length`: A size_t value representing the total length of the content being processed.
    - `data`: A reference to a string where the processed data will be accumulated.
- **Control Flow**:
    - The function calls [`process_multipart_ranges_data`](#parse_range_headerprocess_multipart_ranges_data), passing in the `req`, `boundary`, `content_type`, and `content_length` as parameters.
    - It provides three lambda functions as callbacks to handle tokens and data segments during the processing.
    - The first two lambda functions append the received tokens directly to the `data` string.
    - The third lambda function checks the validity of the offset and length, then appends the corresponding substring from `res.body` to `data`.
- **Output**: The function does not return a value; instead, it modifies the `data` string by appending processed multipart range data.
- **Functions called**:
    - [`parse_range_header::process_multipart_ranges_data`](#parse_range_headerprocess_multipart_ranges_data)


---
### get\_multipart\_ranges\_data\_length<!-- {{#callable:parse_range_header::get_multipart_ranges_data_length}} -->
Calculates the total length of data in multipart ranges from a given request.
- **Inputs**:
    - `req`: A constant reference to a `Request` object containing the multipart data.
    - `boundary`: A string representing the boundary used to separate parts in the multipart data.
    - `content_type`: A string indicating the content type of the multipart data.
    - `content_length`: A size_t representing the total length of the content.
- **Control Flow**:
    - Initializes a variable `data_length` to zero to keep track of the total length.
    - Calls the [`process_multipart_ranges_data`](#parse_range_headerprocess_multipart_ranges_data) function with the provided parameters and three lambda functions.
    - The first two lambda functions increment `data_length` by the size of each token processed.
    - The third lambda function adds the length of the data segment to `data_length` based on the provided offset and length.
    - Returns the total `data_length` after processing all multipart ranges.
- **Output**: Returns a size_t value representing the total length of the data in the multipart ranges.
- **Functions called**:
    - [`parse_range_header::process_multipart_ranges_data`](#parse_range_headerprocess_multipart_ranges_data)


---
### write\_multipart\_ranges\_data<!-- {{#callable:parse_range_header::write_multipart_ranges_data}} -->
The `write_multipart_ranges_data` function processes and writes multipart range data to a stream.
- **Inputs**:
    - `strm`: A reference to a `Stream` object where the data will be written.
    - `req`: A reference to a `Request` object containing the request data.
    - `res`: A reference to a `Response` object that holds the response data.
    - `boundary`: A string representing the boundary used in multipart data.
    - `content_type`: A string indicating the content type of the data.
    - `content_length`: A size_t value representing the length of the content.
    - `is_shutting_down`: A reference to a type T that indicates if the system is shutting down.
- **Control Flow**:
    - The function calls [`process_multipart_ranges_data`](#parse_range_headerprocess_multipart_ranges_data) with the provided parameters.
    - It uses lambda functions to handle writing tokens to the stream and to write content based on offsets and lengths.
    - The first two lambda functions write tokens directly to the stream using `strm.write(token)`.
    - The third lambda function calls [`write_content`](#write_content) to write specific content from the response provider based on the given offset and length.
- **Output**: The function returns a boolean indicating the success or failure of the multipart data processing.
- **Functions called**:
    - [`parse_range_header::process_multipart_ranges_data`](#parse_range_headerprocess_multipart_ranges_data)
    - [`write_content`](#write_content)


---
### expect\_content<!-- {{#callable:parse_range_header::expect_content}} -->
The `expect_content` function checks if a given HTTP `Request` requires a body based on its method or headers.
- **Inputs**:
    - `req`: A constant reference to a `Request` object that contains information about the HTTP request, including its method and headers.
- **Control Flow**:
    - The function first checks if the HTTP method of the request is one of 'POST', 'PUT', 'PATCH', or 'DELETE', returning true if it is.
    - If the method is not one of the specified types, it checks if the request has a 'Content-Length' header and if its value is greater than 0, returning true if both conditions are met.
    - Next, it checks if the request uses chunked transfer encoding by calling [`is_chunked_transfer_encoding`](#is_chunked_transfer_encoding) with the request headers, returning true if this is the case.
    - If none of the above conditions are satisfied, the function returns false.
- **Output**: The function returns a boolean value: true if the request is expected to have content, and false otherwise.
- **Functions called**:
    - [`is_chunked_transfer_encoding`](#is_chunked_transfer_encoding)


---
### has\_crlf<!-- {{#callable:parse_range_header::has_crlf}} -->
The `has_crlf` function checks if a given string contains any carriage return (``) or newline (`
`) characters.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that is to be checked for carriage return or newline characters.
- **Control Flow**:
    - The function starts by obtaining a pointer to the character array of the input string `s` using `c_str()`.
    - It enters a while loop that continues until the end of the string is reached (indicated by the null terminator).
    - Within the loop, it checks if the current character pointed to by `p` is either a carriage return (``) or a newline (`
`).
    - If either character is found, the function immediately returns `true`.
    - If the loop completes without finding either character, the function returns `false`.
- **Output**: The function returns a boolean value: `true` if the string contains at least one carriage return or newline character, and `false` otherwise.


---
### message\_digest<!-- {{#callable:parse_range_header::message_digest}} -->
Computes the hexadecimal representation of a message digest for a given string using a specified hashing algorithm.
- **Inputs**:
    - `s`: A constant reference to a string that contains the input message to be hashed.
    - `algo`: A pointer to an `EVP_MD` structure that specifies the hashing algorithm to be used.
- **Control Flow**:
    - Creates a unique pointer to an `EVP_MD_CTX` context for managing the digest operation.
    - Initializes the digest context with the specified hashing algorithm using `EVP_DigestInit_ex`.
    - Updates the digest context with the input string using `EVP_DigestUpdate`.
    - Finalizes the digest operation and retrieves the resulting hash using `EVP_DigestFinal_ex`.
    - Converts the hash bytes into a hexadecimal string format using a stringstream.
- **Output**: Returns a string containing the hexadecimal representation of the computed message digest.


---
### MD5<!-- {{#callable:parse_range_header::MD5}} -->
Computes the MD5 hash of a given string.
- **Inputs**:
    - `s`: A constant reference to a string that contains the input data to be hashed.
- **Control Flow**:
    - The function calls [`message_digest`](#parse_range_headermessage_digest) with the input string `s` and the MD5 algorithm provided by `EVP_md5()`.
    - The result of the [`message_digest`](#parse_range_headermessage_digest) function is returned as the output.
- **Output**: Returns a string representing the MD5 hash of the input string.
- **Functions called**:
    - [`parse_range_header::message_digest`](#parse_range_headermessage_digest)


---
### SHA\_256<!-- {{#callable:parse_range_header::SHA_256}} -->
Computes the SHA-256 hash of a given string.
- **Inputs**:
    - `s`: A constant reference to a string input for which the SHA-256 hash is to be computed.
- **Control Flow**:
    - The function calls [`message_digest`](#parse_range_headermessage_digest) with the input string `s` and the SHA-256 algorithm provided by `EVP_sha256()`.
    - The result of the [`message_digest`](#parse_range_headermessage_digest) function, which is the computed hash, is returned as a string.
- **Output**: Returns a string representing the SHA-256 hash of the input string.
- **Functions called**:
    - [`parse_range_header::message_digest`](#parse_range_headermessage_digest)


---
### SHA\_512<!-- {{#callable:parse_range_header::SHA_512}} -->
Computes the SHA-512 hash of the input string.
- **Inputs**:
    - `s`: A constant reference to a string containing the data to be hashed.
- **Control Flow**:
    - The function calls [`message_digest`](#parse_range_headermessage_digest) with the input string `s` and the SHA-512 algorithm provided by `EVP_sha512()`.
    - The result of the [`message_digest`](#parse_range_headermessage_digest) function is returned directly.
- **Output**: A string representing the SHA-512 hash of the input data.
- **Functions called**:
    - [`parse_range_header::message_digest`](#parse_range_headermessage_digest)


---
### make\_digest\_authentication\_header<!-- {{#callable:parse_range_header::make_digest_authentication_header}} -->
Generates a Digest Authentication header for HTTP requests.
- **Inputs**:
    - `req`: A `Request` object containing the HTTP method and path.
    - `auth`: A map containing authentication parameters such as realm, nonce, and optional qop and algorithm.
    - `cnonce_count`: A size_t representing the count of nonce used for the request.
    - `cnonce`: A string representing the client nonce.
    - `username`: A string representing the username for authentication.
    - `password`: A string representing the password for authentication.
    - `is_proxy`: A boolean indicating whether the header is for a proxy or not, defaulting to false.
- **Control Flow**:
    - Initializes a string `nc` by formatting `cnonce_count` as an 8-character hexadecimal string.
    - Checks if 'qop' is present in the `auth` map and sets it to 'auth' or 'auth-int' based on its value.
    - Determines the hashing algorithm to use, defaulting to 'MD5' unless specified otherwise in `auth`.
    - Constructs the A1 and A2 components for the hash based on the username, realm, password, request method, and path.
    - Calculates the response hash based on the selected algorithm, including the nonce, cnonce, and qop if applicable.
    - Constructs the final Digest Authentication header string using the calculated response and other parameters.
    - Returns a pair containing the appropriate authorization key ('Authorization' or 'Proxy-Authorization') and the constructed header.
- **Output**: Returns a pair of strings where the first is the authorization key and the second is the formatted Digest Authentication header.


---
### is\_ssl\_peer\_could\_be\_closed<!-- {{#callable:parse_range_header::is_ssl_peer_could_be_closed}} -->
Determines if an SSL peer connection can be closed.
- **Inputs**:
    - `ssl`: A pointer to an `SSL` structure representing the SSL connection.
    - `sock`: A socket descriptor of type `socket_t` used for the connection.
- **Control Flow**:
    - Sets the socket to non-blocking mode using `detail::set_nonblocking`.
    - Creates a scope exit object to ensure the socket is set back to blocking mode after the function execution.
    - Declares a buffer of size 1 to hold data from the SSL connection.
    - Calls `SSL_peek` to check if there is any data available to read from the SSL connection without removing it from the queue.
    - Checks if `SSL_get_error` returns `SSL_ERROR_ZERO_RETURN`, indicating that the peer has closed the connection.
- **Output**: Returns a boolean value indicating whether the SSL peer can be closed, which is true if there is no data to read and the peer has signaled closure.


---
### load\_system\_certs\_on\_windows<!-- {{#callable:parse_range_header::load_system_certs_on_windows}} -->
Loads system root certificates into an `X509_STORE` on Windows.
- **Inputs**:
    - `store`: A pointer to an `X509_STORE` where the certificates will be added.
- **Control Flow**:
    - Attempts to open the system certificate store for root certificates using `CertOpenSystemStoreW`.
    - If the store cannot be opened, the function returns false.
    - Iterates through each certificate in the opened store using `CertEnumCertificatesInStore`.
    - For each certificate, it decodes the certificate using `d2i_X509`.
    - If decoding is successful, the certificate is added to the `X509_STORE` using `X509_STORE_add_cert`, and the decoded certificate is freed with `X509_free`.
    - The function keeps track of whether at least one certificate was successfully added.
    - After processing all certificates, it frees the certificate context and closes the certificate store.
- **Output**: Returns true if at least one certificate was successfully added to the `X509_STORE`, otherwise returns false.


---
### cf\_object\_ptr\_deleter<!-- {{#callable:parse_range_header::cf_object_ptr_deleter}} -->
Releases a Core Foundation object if the provided pointer is not null.
- **Inputs**:
    - `obj`: A pointer to a Core Foundation object of type `CFTypeRef` that is to be released.
- **Control Flow**:
    - Checks if the `obj` pointer is not null.
    - If `obj` is not null, calls `CFRelease` to release the object.
- **Output**: This function does not return a value; it performs a side effect of releasing the memory associated with the Core Foundation object.


---
### retrieve\_certs\_from\_keychain<!-- {{#callable:parse_range_header::retrieve_certs_from_keychain}} -->
The `retrieve_certs_from_keychain` function retrieves all certificate references from the keychain and stores them in a provided array.
- **Inputs**:
    - `certs`: A reference to a `CFObjectPtr<CFArrayRef>` where the retrieved certificate references will be stored.
- **Control Flow**:
    - Defines an array of keys and corresponding values for the keychain query.
    - Creates a dictionary query using `CFDictionaryCreate` with the specified keys and values.
    - Checks if the query creation was successful; if not, returns false.
    - Calls `SecItemCopyMatching` to execute the query and retrieve matching security items.
    - Validates the result of the query; if it fails or the result is not an array, returns false.
    - If successful, resets the `certs` reference to point to the retrieved array of certificates and returns true.
- **Output**: Returns a boolean indicating the success or failure of the certificate retrieval operation.


---
### retrieve\_root\_certs\_from\_keychain<!-- {{#callable:parse_range_header::retrieve_root_certs_from_keychain}} -->
Retrieves root certificates from the keychain and stores them in a provided array.
- **Inputs**:
    - `certs`: A reference to a `CFObjectPtr<CFArrayRef>` where the retrieved root certificates will be stored.
- **Control Flow**:
    - Calls `SecTrustCopyAnchorCertificates` to retrieve the root security items.
    - Checks if the call to `SecTrustCopyAnchorCertificates` was successful by comparing the return value to `errSecSuccess`.
    - If the call fails, the function returns `false`.
    - If successful, it resets the `certs` reference with the retrieved `root_security_items` and returns `true`.
- **Output**: Returns a boolean indicating the success or failure of the operation.


---
### add\_certs\_to\_x509\_store<!-- {{#callable:parse_range_header::add_certs_to_x509_store}} -->
The `add_certs_to_x509_store` function adds valid certificates from a given `CFArrayRef` to an `X509_STORE`.
- **Inputs**:
    - `certs`: A reference to a `CFArrayRef` containing certificates to be added.
    - `store`: A pointer to an `X509_STORE` where the certificates will be added.
- **Control Flow**:
    - The function initializes a boolean variable `result` to false to track if any certificates were successfully added.
    - It iterates over each certificate in the `CFArrayRef` using a for loop.
    - For each certificate, it checks if the type of the certificate matches the expected type using `SecCertificateGetTypeID`.
    - If the type matches, it attempts to export the certificate data using `SecItemExport`.
    - If the export is successful, it creates a `CFObjectPtr` to manage the certificate data's memory.
    - The function then decodes the certificate data into an `X509` structure using `d2i_X509`.
    - If the decoding is successful, it adds the `X509` certificate to the `X509_STORE` and frees the `X509` structure.
    - The `result` variable is set to true if at least one certificate was added.
- **Output**: The function returns a boolean value indicating whether at least one certificate was successfully added to the `X509_STORE`.


---
### load\_system\_certs\_on\_macos<!-- {{#callable:parse_range_header::load_system_certs_on_macos}} -->
The `load_system_certs_on_macos` function loads system certificates from the macOS keychain into an `X509_STORE`.
- **Inputs**:
    - `store`: A pointer to an `X509_STORE` where the certificates will be added.
- **Control Flow**:
    - The function initializes a boolean variable `result` to false.
    - It attempts to retrieve certificates from the keychain using [`retrieve_certs_from_keychain`](#parse_range_headerretrieve_certs_from_keychain) and checks if the retrieval was successful and if the `certs` array is not null.
    - If successful, it adds the retrieved certificates to the `X509_STORE` using [`add_certs_to_x509_store`](#parse_range_headeradd_certs_to_x509_store) and updates `result`.
    - It then attempts to retrieve root certificates from the keychain in a similar manner.
    - If the root certificates are successfully retrieved, it adds them to the `X509_STORE` and updates `result` to true if either addition was successful.
- **Output**: The function returns a boolean indicating whether any certificates were successfully loaded into the `X509_STORE`.
- **Functions called**:
    - [`parse_range_header::retrieve_certs_from_keychain`](#parse_range_headerretrieve_certs_from_keychain)
    - [`parse_range_header::add_certs_to_x509_store`](#parse_range_headeradd_certs_to_x509_store)
    - [`parse_range_header::retrieve_root_certs_from_keychain`](#parse_range_headerretrieve_root_certs_from_keychain)


---
### parse\_www\_authenticate<!-- {{#callable:parse_range_header::parse_www_authenticate}} -->
Parses the `WWW-Authenticate` or `Proxy-Authenticate` header from a `Response` object and populates a map with the authentication parameters.
- **Inputs**:
    - `res`: A constant reference to a `Response` object that contains the headers to be parsed.
    - `auth`: A reference to a map that will be populated with authentication parameters extracted from the header.
    - `is_proxy`: A boolean indicating whether to parse the `Proxy-Authenticate` header instead of the `WWW-Authenticate` header.
- **Control Flow**:
    - Determines the appropriate authentication header key based on the `is_proxy` flag.
    - Checks if the header exists in the `Response` object.
    - If the header exists, retrieves its value and checks for the presence of a space to separate the authentication type.
    - If the type is 'Basic', the function returns false immediately.
    - If the type is 'Digest', it processes the remaining string using a regular expression to extract key-value pairs.
    - Populates the `auth` map with the extracted key-value pairs from the header.
    - Returns true if the parsing was successful, otherwise returns false.
- **Output**: Returns a boolean indicating whether the parsing was successful (true) or not (false).


---
### hosted\_at<!-- {{#callable:hosted_at}} -->
The `hosted_at` function retrieves the IP addresses associated with a given hostname and stores them in a provided vector.
- **Inputs**:
    - `hostname`: A constant reference to a string representing the hostname to resolve.
    - `addrs`: A reference to a vector of strings where the resolved IP addresses will be stored.
- **Control Flow**:
    - The function initializes a `struct addrinfo` with hints for address resolution, specifying that it should accept any address family and socket type.
    - It calls `getaddrinfo` to resolve the hostname into a linked list of `addrinfo` structures; if this fails, it optionally calls `res_init` on Linux systems and exits the function.
    - A scope exit mechanism is set up to ensure that the memory allocated for the `addrinfo` structures is freed after use.
    - The function iterates through the linked list of `addrinfo` structures, extracting the IP address from each entry using `detail::get_ip_and_port` and adding it to the `addrs` vector if successful.
- **Output**: The function does not return a value; instead, it populates the `addrs` vector with the resolved IP addresses.


---
### append\_query\_params<!-- {{#callable:append_query_params}} -->
The `append_query_params` function appends query parameters to a given URL path.
- **Inputs**:
    - `path`: A constant reference to a string representing the base URL path to which query parameters will be appended.
    - `params`: A reference to a `Params` object containing the query parameters to be converted into a query string.
- **Control Flow**:
    - The function initializes a string `path_with_query` with the value of `path`.
    - It uses a thread-local regular expression to check if the `path` already contains a query string.
    - Based on the presence of a query string, it determines the appropriate delimiter (`&` or `?`) to use for appending the new parameters.
    - The function then appends the delimiter followed by the query string generated from the `params` using `detail::params_to_query_str`.
    - Finally, it returns the modified URL with the appended query parameters.
- **Output**: The function returns a string that represents the original path with the appended query parameters.


---
### make\_range\_header<!-- {{#callable:make_range_header}} -->
Constructs a 'Range' header string from a collection of byte ranges.
- **Inputs**:
    - `ranges`: A collection of pairs representing byte ranges, where each pair contains a start and end value.
- **Control Flow**:
    - Initializes a string `field` with the prefix 'bytes='.
    - Iterates over each range in the `ranges` collection using a for loop.
    - For each range, checks if it is not the first range to append a comma and space.
    - Appends the start value of the range to `field` if it is not -1.
    - Appends a hyphen to `field` to separate the start and end values.
    - Appends the end value of the range to `field` if it is not -1.
    - Increments the index `i` to track the number of ranges processed.
    - Returns a pair containing the header name 'Range' and the constructed `field` string.
- **Output**: A pair of strings where the first element is 'Range' and the second element is the constructed range header string.


---
### make\_basic\_authentication\_header<!-- {{#callable:make_basic_authentication_header}} -->
Generates a basic authentication header for HTTP requests.
- **Inputs**:
    - `username`: The username to be included in the authentication header.
    - `password`: The password associated with the username.
    - `is_proxy`: A boolean indicating whether the header is for a proxy or a standard authorization.
- **Control Flow**:
    - Concatenates the username and password with a colon in between.
    - Encodes the concatenated string using Base64 encoding.
    - Determines the appropriate header key based on the is_proxy flag.
    - Returns a pair containing the header key and the encoded authentication field.
- **Output**: A pair of strings where the first string is the header key ('Authorization' or 'Proxy-Authorization') and the second string is the Base64 encoded authentication field.


---
### make\_bearer\_token\_authentication\_header<!-- {{#callable:make_bearer_token_authentication_header}} -->
Generates a bearer token authentication header as a key-value pair.
- **Inputs**:
    - `token`: A string representing the bearer token used for authentication.
    - `is_proxy`: A boolean flag indicating whether the header is for proxy authorization; defaults to false.
- **Control Flow**:
    - Concatenates 'Bearer ' with the provided `token` to create the authentication field.
    - Determines the key to use based on the `is_proxy` flag, selecting either 'Proxy-Authorization' or 'Authorization'.
    - Returns a pair containing the key and the constructed field.
- **Output**: A pair of strings where the first element is the authorization key and the second element is the bearer token field.


---
### calc\_actual\_timeout<!-- {{#callable:detail::calc_actual_timeout}} -->
Calculates the actual timeout values in seconds and microseconds based on maximum allowed timeout and elapsed duration.
- **Inputs**:
    - `max_timeout_msec`: The maximum allowable timeout in milliseconds.
    - `duration_msec`: The elapsed duration in milliseconds.
    - `timeout_sec`: The desired timeout in seconds.
    - `timeout_usec`: The desired timeout in microseconds.
    - `actual_timeout_sec`: A reference to store the calculated actual timeout in seconds.
    - `actual_timeout_usec`: A reference to store the calculated actual timeout in microseconds.
- **Control Flow**:
    - The function first converts the desired timeout from seconds and microseconds into milliseconds.
    - It then calculates the actual timeout in milliseconds by taking the minimum of the remaining time (max_timeout_msec - duration_msec) and the desired timeout in milliseconds.
    - If the calculated actual timeout is negative, it is set to zero to avoid negative timeout values.
    - Finally, the function converts the actual timeout back into seconds and microseconds and stores them in the provided reference variables.
- **Output**: The function outputs the actual timeout values in seconds and microseconds through the reference parameters.


---
### SocketStream<!-- {{#callable:detail::SocketStream::SocketStream}} -->
The `SocketStream` constructor initializes a `SocketStream` object with specified socket parameters and timeout settings.
- **Inputs**:
    - `sock`: A socket identifier of type `socket_t` used for network communication.
    - `read_timeout_sec`: The number of seconds for the read timeout.
    - `read_timeout_usec`: The number of microseconds for the read timeout.
    - `write_timeout_sec`: The number of seconds for the write timeout.
    - `write_timeout_usec`: The number of microseconds for the write timeout.
    - `max_timeout_msec`: The maximum timeout duration in milliseconds.
    - `start_time`: A `std::chrono::time_point` representing the start time for the socket operations.
- **Control Flow**:
    - The constructor initializes member variables with the provided input parameters.
    - It sets up the socket for communication by storing the socket identifier.
    - Timeout values for reading and writing operations are stored in their respective member variables.
    - The `start_time_` member variable is initialized to track the beginning of socket operations.
    - A read buffer is initialized with a specified size, filled with zeros.
- **Output**: The constructor does not return a value but initializes a `SocketStream` object with the specified parameters.


---
### \~SocketStream<!-- {{#callable:detail::SocketStream::~SocketStream}} -->
The `SocketStream` destructor is defined as default, indicating that it will automatically clean up resources when an instance of `SocketStream` is destroyed.
- **Inputs**:
    - `this`: A pointer to the instance of `SocketStream` that is being destroyed.
- **Control Flow**:
    - Since the destructor is defined as default, there is no custom logic or control flow within it.
    - The destructor will automatically handle the cleanup of any resources allocated by the `SocketStream` class.
- **Output**: The function does not return a value; it performs cleanup operations for the `SocketStream` instance.


---
### is\_readable<!-- {{#callable:detail::SocketStream::is_readable}} -->
Checks if there are unread bytes in the socket's read buffer.
- **Inputs**:
    - `this`: A constant reference to the `SocketStream` object on which the method is called.
- **Control Flow**:
    - Compares the current offset of the read buffer (`read_buff_off_`) with the total size of the content in the read buffer (`read_buff_content_size_`).
    - Returns `true` if there are unread bytes (i.e., `read_buff_off_` is less than `read_buff_content_size_`), otherwise returns `false`.
- **Output**: Returns a boolean value indicating whether there are unread bytes in the read buffer.


---
### wait\_readable<!-- {{#callable:detail::SocketStream::wait_readable}} -->
The `wait_readable` method checks if a socket is ready for reading within a specified timeout.
- **Inputs**:
    - `max_timeout_msec_`: The maximum timeout in milliseconds for waiting for the socket to become readable.
    - `sock_`: The socket file descriptor that is being monitored for readability.
    - `read_timeout_sec_`: The seconds component of the timeout for reading.
    - `read_timeout_usec_`: The microseconds component of the timeout for reading.
- **Control Flow**:
    - The function first checks if `max_timeout_msec_` is less than or equal to zero.
    - If true, it directly calls [`select_read`](#select_read) with the socket and the predefined timeout values.
    - If false, it calculates the actual timeout values in seconds and microseconds using [`calc_actual_timeout`](#detailcalc_actual_timeout).
    - Finally, it calls [`select_read`](#select_read) again with the socket and the calculated timeout values.
- **Output**: The function returns a boolean indicating whether the socket is readable (true) or not (false) based on the result of the [`select_read`](#select_read) function.
- **Functions called**:
    - [`select_read`](#select_read)
    - [`detail::calc_actual_timeout`](#detailcalc_actual_timeout)
    - [`detail::SocketStream::duration`](#SocketStreamduration)


---
### wait\_writable<!-- {{#callable:detail::SocketStream::wait_writable}} -->
Checks if a socket is writable and alive within a specified timeout.
- **Inputs**: None
- **Control Flow**:
    - Calls [`select_write`](#select_write) with the socket, write timeout seconds, and write timeout microseconds to check if the socket is ready for writing.
    - Checks if the return value of [`select_write`](#select_write) is greater than 0, indicating that the socket is writable.
    - Calls [`is_socket_alive`](#is_socket_alive) to verify that the socket is still alive.
- **Output**: Returns a boolean value indicating whether the socket is writable and alive.
- **Functions called**:
    - [`select_write`](#select_write)
    - [`is_socket_alive`](#is_socket_alive)


---
### read<!-- {{#callable:detail::SocketStream::read}} -->
The `read` function reads data from a socket into a buffer, managing internal read states and handling different scenarios based on the available data.
- **Inputs**:
    - `ptr`: A pointer to the buffer where the read data will be stored.
    - `size`: The maximum number of bytes to read from the socket.
- **Control Flow**:
    - The function first adjusts the `size` to ensure it does not exceed platform-specific limits.
    - It checks if there is already data available in the internal buffer (`read_buff_`); if so, it copies the available data to `ptr` and updates the offset.
    - If there is no data in the buffer, it waits for the socket to become readable.
    - Once readable, it attempts to read data from the socket into the internal buffer and then copies the requested amount of data to `ptr`.
    - If the requested size is larger than the internal buffer size, it directly reads from the socket into `ptr`.
- **Output**: The function returns the number of bytes read, or -1 if an error occurs, with special handling for cases where less data is available than requested.
- **Functions called**:
    - [`detail::SocketStream::wait_readable`](#SocketStreamwait_readable)
    - [`read_socket`](#read_socket)


---
### write<!-- {{#callable:detail::SocketStream::write}} -->
The `write` method sends data from a buffer to a socket.
- **Inputs**:
    - `ptr`: A pointer to the buffer containing the data to be sent.
    - `size`: The number of bytes to send from the buffer.
- **Control Flow**:
    - The method first checks if the socket is writable by calling `wait_writable()`, returning -1 if it is not.
    - If the platform is Windows (32-bit), it limits the `size` to the maximum value of an `int`.
    - Finally, it calls [`send_socket`](#send_socket) to send the data from the buffer to the socket.
- **Output**: Returns the number of bytes sent on success, or -1 if the socket is not writable.
- **Functions called**:
    - [`detail::SocketStream::wait_writable`](#SocketStreamwait_writable)
    - [`send_socket`](#send_socket)


---
### socket<!-- {{#callable:detail::SocketStream::socket}} -->
This function returns the socket associated with the `SocketStream` instance.
- **Inputs**:
    - `this`: A constant reference to the `SocketStream` instance from which the socket is being retrieved.
- **Control Flow**:
    - The function is defined as an inline method, indicating it may be expanded in place to optimize performance.
    - It directly returns the value of the member variable `sock_`, which represents the socket.
- **Output**: The output is of type `socket_t`, which is the socket associated with the `SocketStream` instance.


---
### duration<!-- {{#callable:detail::SocketStream::duration}} -->
Calculates the duration in milliseconds since the `start_time_` of the `SocketStream` object.
- **Inputs**: None
- **Control Flow**:
    - Uses `std::chrono::steady_clock::now()` to get the current time.
    - Calculates the difference between the current time and `start_time_`.
    - Converts the duration difference to milliseconds using `std::chrono::duration_cast`.
    - Returns the count of milliseconds as a `time_t` value.
- **Output**: Returns the duration in milliseconds as a `time_t` value.


---
### ssl\_new<!-- {{#callable:ssl_new}} -->
Creates a new `SSL` object and configures it for non-blocking socket communication.
- **Inputs**:
    - `sock`: A socket descriptor of type `socket_t` that will be used for SSL communication.
    - `ctx`: A pointer to an `SSL_CTX` structure that contains the SSL context.
    - `ctx_mutex`: A reference to a `std::mutex` used to synchronize access to the SSL context.
    - `SSL_connect_or_accept`: A callable that either connects or accepts an SSL connection.
    - `setup`: A callable that performs additional setup on the `SSL` object.
- **Control Flow**:
    - Locks the `ctx_mutex` to ensure thread-safe access to the SSL context while creating a new `SSL` object.
    - Checks if the `SSL` object was successfully created.
    - Sets the socket to non-blocking mode and creates a new BIO for the socket.
    - Associates the BIO with the `SSL` object.
    - Calls the `setup` function to perform additional configuration on the `SSL` object.
    - If the setup fails or the connection/accept operation fails, it shuts down the `SSL` object, frees it, and returns nullptr.
    - If successful, it sets the BIO back to blocking mode and returns the configured `SSL` object.
- **Output**: Returns a pointer to the newly created and configured `SSL` object, or nullptr if the creation or setup fails.
- **Functions called**:
    - [`set_nonblocking`](#set_nonblocking)


---
### ssl\_delete<!-- {{#callable:ssl_delete}} -->
The `ssl_delete` function safely shuts down an SSL connection and frees the associated resources.
- **Inputs**:
    - `ctx_mutex`: A reference to a `std::mutex` used to ensure thread-safe access to shared resources.
    - `ssl`: A pointer to an `SSL` structure representing the SSL connection to be closed.
    - `sock`: A socket identifier of type `socket_t`, which is not used if `shutdown_gracefully` is true.
    - `shutdown_gracefully`: A boolean flag indicating whether to perform a graceful shutdown of the SSL connection.
- **Control Flow**:
    - The function first checks if `shutdown_gracefully` is true to determine if it should attempt to gracefully shut down the SSL connection.
    - If graceful shutdown is requested, it calls `SSL_shutdown(ssl)` to send a close notification, and if the first call returns 0, it calls `SSL_shutdown(ssl)` again to ensure the close notification is received.
    - Regardless of the shutdown process, the function then locks the mutex `ctx_mutex` to ensure thread safety while freeing the SSL resources.
    - Finally, it calls `SSL_free(ssl)` to release the memory allocated for the SSL structure.
- **Output**: The function does not return a value; it performs cleanup by freeing the SSL resources and optionally shutting down the connection.


---
### ssl\_connect\_or\_accept\_nonblocking<!-- {{#callable:ssl_connect_or_accept_nonblocking}} -->
The `ssl_connect_or_accept_nonblocking` function attempts to establish an SSL connection or accept an SSL connection in a non-blocking manner.
- **Inputs**:
    - `sock`: A socket descriptor of type `socket_t` used for the SSL connection.
    - `ssl`: A pointer to an `SSL` structure representing the SSL context.
    - `ssl_connect_or_accept`: A callable (function or functor) that performs the SSL connect or accept operation.
    - `timeout_sec`: The timeout duration in seconds for the operation.
    - `timeout_usec`: The timeout duration in microseconds for the operation.
- **Control Flow**:
    - The function enters a loop that continues until a successful SSL connection or acceptance is made (indicated by a return value of 1).
    - Inside the loop, it calls the `ssl_connect_or_accept` function with the `ssl` parameter and checks the result.
    - If the result indicates that the operation needs to wait for reading (SSL_ERROR_WANT_READ), it calls [`select_read`](#select_read) to wait for the socket to be ready for reading.
    - If the result indicates that the operation needs to wait for writing (SSL_ERROR_WANT_WRITE), it calls [`select_write`](#select_write) to wait for the socket to be ready for writing.
    - If any other error occurs, the function returns false, indicating failure.
    - If the operation completes successfully, the function returns true.
- **Output**: The function returns a boolean value: true if the SSL connection or acceptance was successful, and false if it failed or encountered an error.
- **Functions called**:
    - [`select_read`](#select_read)
    - [`select_write`](#select_write)


---
### process\_server\_socket\_ssl<!-- {{#callable:process_server_socket_ssl}} -->
Processes an SSL server socket by invoking a callback with a secure socket stream.
- **Inputs**:
    - `svr_sock`: An atomic reference to the server socket.
    - `ssl`: A pointer to the SSL structure used for secure communication.
    - `sock`: The socket descriptor for the connection.
    - `keep_alive_max_count`: The maximum number of keep-alive messages allowed.
    - `keep_alive_timeout_sec`: The timeout duration for keep-alive messages in seconds.
    - `read_timeout_sec`: The read timeout duration in seconds.
    - `read_timeout_usec`: The read timeout duration in microseconds.
    - `write_timeout_sec`: The write timeout duration in seconds.
    - `write_timeout_usec`: The write timeout duration in microseconds.
    - `callback`: A callable object that processes the secure socket stream.
- **Control Flow**:
    - Calls [`process_server_socket_core`](#process_server_socket_core) with the server socket and connection parameters.
    - Defines a lambda function that creates an `SSLSocketStream` using the provided socket and SSL context.
    - Invokes the provided `callback` with the `SSLSocketStream`, close connection flag, and connection closed reference.
- **Output**: Returns a boolean indicating the success or failure of processing the server socket.
- **Functions called**:
    - [`process_server_socket_core`](#process_server_socket_core)


---
### process\_client\_socket\_ssl<!-- {{#callable:process_client_socket_ssl}} -->
Processes a client socket using SSL and a provided callback function.
- **Inputs**:
    - `ssl`: A pointer to an `SSL` structure representing the SSL context.
    - `sock`: A socket descriptor of type `socket_t` representing the client socket.
    - `read_timeout_sec`: The read timeout in seconds.
    - `read_timeout_usec`: The read timeout in microseconds.
    - `write_timeout_sec`: The write timeout in seconds.
    - `write_timeout_usec`: The write timeout in microseconds.
    - `max_timeout_msec`: The maximum timeout in milliseconds.
    - `start_time`: A `std::chrono::time_point` representing the start time for the operation.
    - `callback`: A callable object (function, lambda, etc.) that takes an `SSLSocketStream` as an argument and returns a boolean.
- **Control Flow**:
    - An `SSLSocketStream` object is instantiated with the provided socket and SSL parameters.
    - The callback function is invoked with the `SSLSocketStream` object as an argument.
    - The result of the callback function is returned as the output of `process_client_socket_ssl`.
- **Output**: Returns a boolean indicating the success or failure of the callback execution.


---
### TaskQueue<!-- {{#callable:TaskQueue::TaskQueue}} -->
`TaskQueue` is a default constructor for a class that manages a queue of tasks.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes an instance of the `TaskQueue` class with default settings.
    - There are no parameters or complex logic involved in the constructor.
- **Output**: The constructor does not return any value; it simply creates an instance of the `TaskQueue` class.


---
### Server<!-- {{#callable:Server::Server}} -->
Constructs a `Server` object and initializes a thread pool for handling tasks.
- **Inputs**:
    - `none`: The constructor does not take any input arguments.
- **Control Flow**:
    - Initializes the `new_task_queue` member with a lambda function that creates a new `ThreadPool` instance with a specified thread count.
    - On non-Windows platforms, it ignores the SIGPIPE signal to prevent the server from crashing when writing to a closed socket.
- **Output**: The constructor does not return a value but initializes the `Server` object and its internal task queue.


---
### ClientImpl<!-- {{#callable:ClientImpl::ClientImpl}} -->
The `ClientImpl` constructor initializes a `ClientImpl` object with a specified host and default values for port and other parameters.
- **Inputs**:
    - `host`: A constant reference to a `std::string` representing the hostname or IP address of the server to connect to.
- **Control Flow**:
    - The constructor uses an initializer list to call another constructor of `ClientImpl` with the specified `host`, a default port of 80, and two empty strings for additional parameters.
- **Output**: This constructor does not return a value; it initializes the `ClientImpl` object.


