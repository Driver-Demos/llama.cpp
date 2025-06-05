# Purpose
This Python script is designed to demonstrate the use of Pydantic models for function calling and structured data handling in conjunction with a language model server, specifically llama-server. The script includes several examples that showcase how to define and use Pydantic models to interact with an AI system, allowing it to perform tasks such as sending messages, performing calculations, creating structured data entries, and executing concurrent function calls. The script is structured as a command-line tool, utilizing argparse for argument parsing, and it includes logging capabilities to provide insights into the operations being performed.

The script defines several Pydantic models, such as `SendMessageToUser`, `Calculator`, and `Book`, each with specific fields and methods to perform their respective tasks. It also includes functions like [`create_completion`](#cpp/examples/pydantic_models_to_grammar_examplescreate_completion) to interact with the llama-server API, and `example_*` functions to demonstrate various use cases, such as remote code execution, calculator operations, structured data creation, and concurrent function execution. The script is intended to be run as a standalone application, as indicated by the [`main`](#cpp/examples/pydantic_models_to_grammar_examplesmain) function, which orchestrates the execution of the examples based on the provided command-line arguments. This setup makes it a versatile tool for exploring the integration of Pydantic models with AI-driven function calls and data processing.
# Imports and Dependencies

---
- `__future__.annotations`
- `argparse`
- `datetime`
- `json`
- `logging`
- `textwrap`
- `sys`
- `enum.Enum`
- `typing.Optional`
- `typing.Union`
- `requests`
- `pydantic.BaseModel`
- `pydantic.Field`
- `pydantic_models_to_grammar.add_run_method_to_dynamic_model`
- `pydantic_models_to_grammar.convert_dictionary_to_pydantic_model`
- `pydantic_models_to_grammar.create_dynamic_model_from_function`
- `pydantic_models_to_grammar.generate_gbnf_grammar_and_documentation`


# Classes

---
### SendMessageToUser<!-- {{#class:llama.cpp/examples/pydantic_models_to_grammar_examples.SendMessageToUser}} -->
- **Decorators**: `@dataclass`
- **Members**:
    - `chain_of_thought`: Your chain of thought while sending the message.
    - `message`: Message you want to send to the user.
- **Description**: The `SendMessageToUser` class is a Pydantic model designed to encapsulate the functionality of sending a message to a user. It includes two fields: `chain_of_thought`, which captures the reasoning or thought process behind sending the message, and `message`, which is the actual content intended for the user. This class is part of a larger system that allows for structured function calls and interactions, particularly in the context of AI-driven applications.
- **Methods**:
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.SendMessageToUser.run`](#SendMessageToUserrun)
- **Inherits From**:
    - `BaseModel`

**Methods**

---
#### SendMessageToUser\.run<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar_examples.SendMessageToUser.run}} -->
The `run` method in the `SendMessageToUser` class prints a message to the console.
- **Inputs**:
    - `self`: An instance of the `SendMessageToUser` class.
- **Control Flow**:
    - The method prints a formatted string to the console that includes the `message` attribute of the `SendMessageToUser` instance.
- **Output**: The method does not return any value; it outputs a message to the console.
- **See also**: [`llama.cpp/examples/pydantic_models_to_grammar_examples.SendMessageToUser`](#cpp/examples/pydantic_models_to_grammar_examplesSendMessageToUser)  (Base Class)



---
### MathOperation<!-- {{#class:llama.cpp/examples/pydantic_models_to_grammar_examples.MathOperation}} -->
- **Description**: The `MathOperation` class is an enumeration that defines four basic mathematical operations: addition, subtraction, multiplication, and division. Each operation is represented as a string value, allowing for easy reference and use in mathematical computations or logic that requires these operations.
- **Inherits From**:
    - `Enum`


---
### Calculator<!-- {{#class:llama.cpp/examples/pydantic_models_to_grammar_examples.Calculator}} -->
- **Members**:
    - `number_one`: First number for the operation, can be an integer or a float.
    - `operation`: Math operation to perform, defined by the MathOperation enum.
    - `number_two`: Second number for the operation, can be an integer or a float.
- **Description**: The Calculator class is a Pydantic model designed to perform basic mathematical operations on two numbers. It supports addition, subtraction, multiplication, and division, as specified by the MathOperation enum. The class ensures that the input numbers are either integers or floats and uses the specified operation to compute the result when the run method is called.
- **Methods**:
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.Calculator.run`](#Calculatorrun)
- **Inherits From**:
    - `BaseModel`

**Methods**

---
#### Calculator\.run<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar_examples.Calculator.run}} -->
The `run` method performs a specified mathematical operation (addition, subtraction, multiplication, or division) on two numbers and returns the result.
- **Inputs**:
    - `self`: An instance of the Calculator class containing the attributes `number_one`, `operation`, and `number_two`.
- **Control Flow**:
    - Check if the operation is `MathOperation.ADD`, and if so, return the sum of `number_one` and `number_two`.
    - Check if the operation is `MathOperation.SUBTRACT`, and if so, return the difference between `number_one` and `number_two`.
    - Check if the operation is `MathOperation.MULTIPLY`, and if so, return the product of `number_one` and `number_two`.
    - Check if the operation is `MathOperation.DIVIDE`, and if so, return the quotient of `number_one` divided by `number_two`.
    - If the operation does not match any of the above, raise a `ValueError` indicating an unknown operation.
- **Output**: The result of the mathematical operation performed on `number_one` and `number_two`, or raises a `ValueError` if the operation is unknown.
- **See also**: [`llama.cpp/examples/pydantic_models_to_grammar_examples.Calculator`](#cpp/examples/pydantic_models_to_grammar_examplesCalculator)  (Base Class)



---
### Category<!-- {{#class:llama.cpp/examples/pydantic_models_to_grammar_examples.Category}} -->
- **Description**: The `Category` class is an enumeration that defines two categories for books: 'Fiction' and 'Non-Fiction'. It is used to categorize books into these two distinct types, providing a simple way to classify and manage book entries based on their content type.
- **Inherits From**:
    - `Enum`


---
### Book<!-- {{#class:llama.cpp/examples/pydantic_models_to_grammar_examples.Book}} -->
- **Members**:
    - `title`: Title of the book.
    - `author`: Author of the book.
    - `published_year`: Publishing year of the book.
    - `keywords`: A list of keywords.
    - `category`: Category of the book.
    - `summary`: Summary of the book.
- **Description**: The `Book` class is a Pydantic model that represents an entry about a book, encapsulating essential details such as the title, author, published year, keywords, category, and a summary. It uses Pydantic's `Field` to enforce data validation and provide descriptions for each attribute, ensuring that the data conforms to expected types and formats. This class is designed to facilitate structured data handling and validation for book-related information in applications.
- **Inherits From**:
    - `BaseModel`


# Functions

---
### create\_completion<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar_examples.create_completion}} -->
The `create_completion` function sends a POST request to the `/completion` API endpoint on a specified server to generate a completion based on a given prompt and grammar, and returns the content of the response.
- **Inputs**:
    - `host`: The server address where the `/completion` API is hosted.
    - `prompt`: The text prompt to be sent to the API for generating a completion.
    - `gbnf_grammar`: The grammar in GBNF format to be used by the API for generating a completion.
- **Control Flow**:
    - Prints the request details including the grammar and prompt with indentation for readability.
    - Sets the request headers to indicate JSON content type.
    - Creates a data dictionary containing the prompt and grammar to be sent in the request body.
    - Sends a POST request to the `/completion` endpoint on the specified host with the headers and data, and parses the JSON response.
    - Asserts that there is no error in the response data, raising an assertion error if an error is present.
    - Logs the result of the API call using the logging module.
    - Extracts the 'content' from the result and prints the model and formatted result content.
    - Returns the extracted content from the API response.
- **Output**: The function returns the 'content' field from the JSON response of the API call, which contains the generated completion.


---
### example\_rce<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar_examples.example_rce}} -->
The `example_rce` function demonstrates a minimal test case where a language model (LLM) calls an arbitrary Python function using JSON format.
- **Inputs**:
    - `host`: The host address of the server where the LLM API is running.
- **Control Flow**:
    - Prints a message indicating the start of the `example_rce` function.
    - Defines a list of tools, which includes the `SendMessageToUser` class.
    - Generates a GBNF grammar and documentation for the tools using [`generate_gbnf_grammar_and_documentation`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_and_documentation).
    - Constructs a system message describing the available functions and their parameters.
    - Creates a user message asking for the result of 42 * 42.
    - Forms a prompt by combining the system and user messages.
    - Calls the [`create_completion`](#cpp/examples/pydantic_models_to_grammar_examplescreate_completion) function with the host, prompt, and GBNF grammar to get a response from the LLM.
    - Parses the LLM's response as JSON data.
    - Maps tool names to their corresponding classes using a dictionary.
    - Retrieves the tool specified in the JSON data and checks if it exists.
    - If the tool is not found, prints an error message and returns 1.
    - If the tool is found, calls its [`run`](#SendMessageToUserrun) method with the parameters from the JSON data.
    - Returns 0 to indicate successful execution.
- **Output**: The function returns 0 if the specified tool is successfully executed, or 1 if an unknown tool is encountered.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar_and_documentation`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_and_documentation)
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.create_completion`](#cpp/examples/pydantic_models_to_grammar_examplescreate_completion)
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.SendMessageToUser.run`](#SendMessageToUserrun)


---
### example\_calculator<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar_examples.example_calculator}} -->
The `example_calculator` function uses a language model to generate a JSON-based function call for a calculation, executes the call using a specified tool, and verifies the result.
- **Inputs**:
    - `host`: The host address of the llama-server to which the function sends requests for completion.
- **Control Flow**:
    - Prints a message indicating the start of the function.
    - Defines a list of tools, including `SendMessageToUser` and `Calculator`.
    - Generates GBNF grammar and documentation using the [`generate_gbnf_grammar_and_documentation`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_and_documentation) function with the tools list.
    - Constructs a system message and a user message for the prompt.
    - Creates a prompt combining the system and user messages and sends it to the [`create_completion`](#cpp/examples/pydantic_models_to_grammar_examplescreate_completion) function to get a completion response.
    - Parses the response into JSON data and compares it with an expected JSON structure.
    - If the JSON data does not match the expected structure, prints a message indicating the result is not as expected.
    - Maps tool names to tool classes and retrieves the tool corresponding to the function specified in the JSON data.
    - If the tool is not found, prints an error message and returns 1.
    - Executes the tool's [`run`](#Calculatorrun) method with the parameters from the JSON data and prints the result.
    - Returns 0 to indicate successful execution.
- **Output**: Returns 0 if the function executes successfully and the tool is found and executed correctly; otherwise, returns 1 if an error occurs.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar_and_documentation`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_and_documentation)
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.create_completion`](#cpp/examples/pydantic_models_to_grammar_examplescreate_completion)
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.Calculator.run`](#Calculatorrun)


---
### example\_struct<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar_examples.example_struct}} -->
The `example_struct` function generates a structured JSON entry for a Book using a language model and pydantic models.
- **Inputs**:
    - `host`: The host address of the server where the completion API is called.
- **Control Flow**:
    - Prints a message indicating the start of the function.
    - Defines a list of tools containing the Book pydantic model.
    - Generates GBNF grammar and documentation for the Book model using [`generate_gbnf_grammar_and_documentation`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_and_documentation).
    - Creates a system message describing the task of creating a JSON dataset entry for a Book.
    - Defines a text input describing a book, which is used as the user input for the language model.
    - Constructs a prompt combining the system message and user input text.
    - Calls [`create_completion`](#cpp/examples/pydantic_models_to_grammar_examplescreate_completion) with the host, prompt, and grammar to get a JSON response from the language model.
    - Parses the JSON response into a Python dictionary.
    - Checks if the keys in the JSON data match the expected keys for a Book entry.
    - If the keys do not match, prints an error message and returns 1.
    - If the keys match, creates a Book object using the JSON data.
    - Prints the Book object and returns 0.
- **Output**: The function returns 0 if the JSON data keys match the expected Book entry keys, otherwise it returns 1.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar_and_documentation`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_and_documentation)
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.create_completion`](#cpp/examples/pydantic_models_to_grammar_examplescreate_completion)
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.Book`](#cpp/examples/pydantic_models_to_grammar_examplesBook)


---
### get\_current\_datetime<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar_examples.get_current_datetime}} -->
The function `get_current_datetime` returns the current date and time formatted according to a specified format or a default format if none is provided.
- **Inputs**:
    - `output_format`: An optional string specifying the format for the date and time output, defaulting to '%Y-%m-%d %H:%M:%S' if not provided.
- **Control Flow**:
    - The function checks if an `output_format` is provided.
    - If `output_format` is not provided, it defaults to '%Y-%m-%d %H:%M:%S'.
    - The function retrieves the current date and time using `datetime.datetime.now()`.
    - The current date and time are formatted using the `strftime` method with the specified or default format.
- **Output**: A string representing the current date and time formatted according to the specified or default format.


---
### get\_current\_weather<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar_examples.get_current_weather}} -->
The `get_current_weather` function returns the current weather for a specified location in a given unit.
- **Inputs**:
    - `location`: A string representing the location for which the weather is requested.
    - `unit`: An object with a `value` attribute representing the unit of temperature (e.g., Celsius or Fahrenheit).
- **Control Flow**:
    - Check if the location contains 'London', and if so, return a JSON string with the location set to 'London' and temperature set to '42' in the specified unit.
    - Check if the location contains 'New York', and if so, return a JSON string with the location set to 'New York' and temperature set to '24' in the specified unit.
    - Check if the location contains 'North Pole', and if so, return a JSON string with the location set to 'North Pole' and temperature set to '-42' in the specified unit.
    - If the location does not match any of the specified cases, return a JSON string with the location set to the input location and temperature set to 'unknown'.
- **Output**: A JSON string representing the weather information for the specified location, including the location name, temperature, and unit.


---
### example\_concurrent<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar_examples.example_concurrent}} -->
The `example_concurrent` function demonstrates parallel function calling using Python functions, pydantic models, and OpenAI-style function definitions to execute tasks like getting the current date and time, weather information, and performing calculations.
- **Inputs**:
    - `host`: The host address of the llama-server to which the function will send requests for completion.
- **Control Flow**:
    - Prints a message indicating the start of the `example_concurrent` function.
    - Defines a function in OpenAI style for getting current weather information, specifying its parameters and description.
    - Converts the OpenAI function definition into a pydantic model using [`convert_dictionary_to_pydantic_model`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammarconvert_dictionary_to_pydantic_model).
    - Adds a run method to the dynamic model for the weather function using [`add_run_method_to_dynamic_model`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammaradd_run_method_to_dynamic_model).
    - Converts a normal Python function (`get_current_datetime`) into a pydantic model using [`create_dynamic_model_from_function`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammarcreate_dynamic_model_from_function).
    - Creates a list of tools including `SendMessageToUser`, `Calculator`, the current datetime model, and the current weather tool model.
    - Generates GBNF grammar and documentation for the tools using [`generate_gbnf_grammar_and_documentation`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_and_documentation).
    - Constructs a system message and a user prompt for the AI assistant to execute.
    - Calls the [`create_completion`](#cpp/examples/pydantic_models_to_grammar_examplescreate_completion) function to get a response from the server based on the prompt and grammar.
    - Parses the JSON response from the server and compares it to an expected result.
    - If the response does not match the expected result, prints an error message and sets a result flag to 1.
    - Maps tool names to their respective classes and iterates over the JSON data to execute each function call using the mapped tools.
    - Prints the result of each function call.
    - Returns the result flag indicating success or failure of the function execution.
- **Output**: The function returns an integer `res`, which is 0 if the function calls executed as expected, or 1 if there was a mismatch in the expected results.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar.convert_dictionary_to_pydantic_model`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammarconvert_dictionary_to_pydantic_model)
    - [`llama.cpp/examples/pydantic_models_to_grammar.add_run_method_to_dynamic_model`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammaradd_run_method_to_dynamic_model)
    - [`llama.cpp/examples/pydantic_models_to_grammar.create_dynamic_model_from_function`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammarcreate_dynamic_model_from_function)
    - [`llama.cpp/examples/pydantic_models_to_grammar.generate_gbnf_grammar_and_documentation`](pydantic_models_to_grammar.py.driver.md#cpp/examples/pydantic_models_to_grammargenerate_gbnf_grammar_and_documentation)
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.create_completion`](#cpp/examples/pydantic_models_to_grammar_examplescreate_completion)
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.SendMessageToUser.run`](#SendMessageToUserrun)


---
### main<!-- {{#callable:llama.cpp/examples/pydantic_models_to_grammar_examples.main}} -->
The `main` function sets up command-line argument parsing, configures logging, and sequentially executes a series of example functions, returning a status code based on their success.
- **Inputs**: None
- **Control Flow**:
    - An `ArgumentParser` is created with a description from the module's docstring.
    - Two command-line arguments are added: `--host` with a default value and `--verbose` as a flag for logging.
    - The parsed arguments are stored in `args`.
    - Logging is configured to `INFO` level if `--verbose` is set, otherwise `ERROR`.
    - A variable `ret` is initialized to 0 to track the return status.
    - The function [`example_rce`](#cpp/examples/pydantic_models_to_grammar_examplesexample_rce) is called with `args.host`, and its return value is OR-ed with `ret`.
    - Similarly, [`example_calculator`](#cpp/examples/pydantic_models_to_grammar_examplesexample_calculator), [`example_struct`](#cpp/examples/pydantic_models_to_grammar_examplesexample_struct), and [`example_concurrent`](#cpp/examples/pydantic_models_to_grammar_examplesexample_concurrent) are called in sequence, each OR-ing their return value with `ret`.
    - The final value of `ret` is returned, indicating the overall success or failure of the executed examples.
- **Output**: The function returns an integer `ret`, which is 0 if all example functions succeed, or a non-zero value if any fail.
- **Functions called**:
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.example_rce`](#cpp/examples/pydantic_models_to_grammar_examplesexample_rce)
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.example_calculator`](#cpp/examples/pydantic_models_to_grammar_examplesexample_calculator)
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.example_struct`](#cpp/examples/pydantic_models_to_grammar_examplesexample_struct)
    - [`llama.cpp/examples/pydantic_models_to_grammar_examples.example_concurrent`](#cpp/examples/pydantic_models_to_grammar_examplesexample_concurrent)


