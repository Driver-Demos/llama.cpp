# Purpose
This source code file is a JavaScript module that provides utility functions for creating and manipulating HTML elements, specifically focusing on user interface components. The module exports several functions that facilitate the creation of buttons, paragraphs, select elements, and input fields, along with their associated event handling. These functions are designed to streamline the process of dynamically generating and configuring HTML elements in a web application, making it easier to manage UI components programmatically.

The module includes functions such as `el_create_button`, `el_create_boolbutton`, `el_create_select`, and `el_create_input`, each of which is responsible for creating a specific type of HTML element and setting up its properties and event listeners. For instance, `el_create_button` creates a button element with a specified ID, callback function, and optional name and inner text, while `el_create_boolbutton` creates a toggle button that switches between two states, represented by different text labels. Additionally, the module provides functions like `el_creatediv_boolbutton` and `el_creatediv_select`, which wrap these elements in a div container, often with a label, to facilitate layout and styling.

Overall, this module serves as a collection of helper functions aimed at simplifying the creation and configuration of interactive HTML elements in a web application. By abstracting common tasks related to UI component setup, it allows developers to focus on higher-level application logic while ensuring consistent and efficient element management.
# Functions

---
### el\_children\_config\_class
The function `el_children_config_class` sets the class of each child element of a given HTMLDivElement based on whether the child's ID matches a specified ID.
- **Inputs**:
    - `elBase`: An HTMLDivElement whose children will have their class names set.
    - `idSelected`: A string representing the ID of the child element that should receive the `classSelected` class.
    - `classSelected`: A string representing the class name to assign to the child element with the matching ID.
    - `classUnSelected`: A string representing the class name to assign to child elements that do not have the matching ID; defaults to an empty string if not provided.
- **Control Flow**:
    - Iterate over each child element of `elBase`.
    - Check if the current child's ID matches `idSelected`.
    - If the IDs match, set the child's class name to `classSelected`.
    - If the IDs do not match, set the child's class name to `classUnSelected`.
- **Output**: The function does not return a value; it modifies the class names of the child elements of `elBase` in place.


---
### el\_create\_button
The `el_create_button` function creates a button element with specified attributes and a click event listener.
- **Inputs**:
    - `id`: A string representing the ID to be assigned to the button element.
    - `callback`: A function to be executed when the button is clicked, with `this` bound to the button element and receiving a `MouseEvent` as an argument.
    - `name`: An optional string representing the name attribute of the button; defaults to the value of `id` if not provided.
    - `innerText`: An optional string representing the text to be displayed inside the button; defaults to the value of `id` if not provided.
- **Control Flow**:
    - Check if `name` is provided; if not, set it to the value of `id`.
    - Check if `innerText` is provided; if not, set it to the value of `id`.
    - Create a new button element using `document.createElement`.
    - Set the button's `id`, `name`, and `innerText` attributes.
    - Add a click event listener to the button that triggers the provided `callback` function.
    - Return the created button element.
- **Output**: The function returns the created HTMLButtonElement with the specified attributes and event listener.


---
### el\_create\_append\_p
The `el_create_append_p` function creates a paragraph element with specified text, optionally assigns it an ID, and appends it to a given parent element if provided.
- **Inputs**:
    - `text`: A string representing the text content to be set for the paragraph element.
    - `elParent`: An optional HTMLElement to which the created paragraph will be appended; defaults to undefined if not provided.
    - `id`: An optional string to set as the ID of the paragraph element; defaults to undefined if not provided.
- **Control Flow**:
    - Create a paragraph element using `document.createElement('p')`.
    - Set the `innerText` of the paragraph to the provided `text`.
    - If an `id` is provided, set it as the ID of the paragraph element.
    - If an `elParent` is provided, append the paragraph element to this parent element.
- **Output**: Returns the created paragraph element.


---
### el\_create\_boolbutton
The `el_create_boolbutton` function creates a button element that toggles between two text values representing boolean states when clicked, and executes a callback function with the new boolean value.
- **Inputs**:
    - `id`: A string representing the ID to be assigned to the button element.
    - `texts`: An object containing two string properties, 'true' and 'false', which represent the text to display for each boolean state.
    - `defaultValue`: A boolean indicating the initial state of the button.
    - `cb`: A callback function that takes a boolean argument, which is called whenever the button is clicked and the state changes.
- **Control Flow**:
    - Create a button element and assign it to the variable `el`.
    - Set a custom property `xbool` on the button to store the current boolean state, initialized to `defaultValue`.
    - Clone the `texts` object and assign it to a custom property `xtexts` on the button.
    - Set the button's inner text to the value in `xtexts` corresponding to the initial boolean state.
    - If an `id` is provided, set it as the button's ID.
    - Add a 'click' event listener to the button that toggles the `xbool` state, updates the button's inner text to reflect the new state, and calls the callback function `cb` with the new boolean value.
- **Output**: Returns the created button element with the specified behavior and properties.


---
### el\_creatediv\_boolbutton
The `el_creatediv_boolbutton` function creates a div element containing a labeled button that toggles between two text values representing a boolean state.
- **Inputs**:
    - `id`: A string representing the unique identifier for the button element.
    - `label`: A string representing the text label associated with the button.
    - `texts`: An object containing two string properties, 'true' and 'false', which define the button's text for each boolean state.
    - `defaultValue`: A boolean indicating the initial state of the button.
    - `cb`: A callback function that is invoked with the new boolean state whenever the button is clicked.
    - `className`: A string representing the CSS class name for the div element, defaulting to 'gridx2'.
- **Control Flow**:
    - Create a div element and set its class name to the provided className or 'gridx2' by default.
    - Create a label element, set its 'for' attribute to the button's id, and set its inner text to the provided label.
    - Append the label to the div element.
    - Invoke the `el_create_boolbutton` function to create a button with the specified id, texts, defaultValue, and callback function.
    - Append the created button to the div element.
    - Return an object containing the div and button elements.
- **Output**: An object containing two properties: 'div', which is the created div element, and 'el', which is the created button element.


---
### el\_create\_select
The `el_create_select` function creates a select HTML element with specified options and a default selection, and sets up a callback for when the selection changes.
- **Inputs**:
    - `id`: A string representing the ID to be assigned to the select element.
    - `options`: An object containing name-value pairs representing the options to be included in the select element.
    - `defaultOption`: The value of the option that should be selected by default.
    - `cb`: A callback function that is called with the name of the selected option when the selection changes.
- **Control Flow**:
    - Create a select HTML element and assign it to the variable `el`.
    - Clone the `options` object and assign it to `el['xoptions']`.
    - Iterate over the keys of the `options` object to create and append option elements to the select element.
    - Set the `selected` attribute of the option element if it matches the `defaultOption`.
    - Assign the provided `id` to the select element's `id` and `name` attributes if `id` is provided.
    - Add an event listener to the select element that triggers on change, logs the change, and calls the callback function with the selected option's value.
- **Output**: Returns the created select HTML element.


---
### el\_creatediv\_select
The `el_creatediv_select` function creates a div element containing a labeled select dropdown with specified options and a callback for selection changes.
- **Inputs**:
    - `id`: A string representing the unique identifier for the select element.
    - `label`: A string or any type representing the label text for the select element.
    - `options`: An object containing key-value pairs where keys are option names and values are option values.
    - `defaultOption`: The value of the option that should be selected by default.
    - `cb`: A callback function that is called with the name of the selected option when the selection changes.
    - `className`: A string representing the CSS class name for the div element, defaulting to 'gridx2'.
- **Control Flow**:
    - Create a div element and set its class name to the provided className.
    - Create a label element, set its 'for' attribute to the id, and set its inner text to the label.
    - Append the label to the div.
    - Call `el_create_select` to create a select element with the provided id, options, defaultOption, and callback.
    - Append the select element to the div.
    - Return an object containing the div and the select element.
- **Output**: An object containing the created div element and the select element.


---
### el\_create\_input
The `el_create_input` function creates an HTML input element with specified attributes and a change event listener that triggers a callback function.
- **Inputs**:
    - `id`: A string representing the ID to be assigned to the input element.
    - `type`: A string specifying the type of the input element (e.g., 'text', 'number').
    - `defaultValue`: The initial value to be set for the input element.
    - `cb`: A callback function that is invoked with the input's value whenever the input's value changes.
- **Control Flow**:
    - Create an input element using `document.createElement('input')`.
    - Set the `type` attribute of the input element to the provided `type` argument.
    - Set the `value` attribute of the input element to the provided `defaultValue` argument.
    - If an `id` is provided, set the `id` attribute of the input element to this value.
    - Add an event listener to the input element that listens for the 'change' event and calls the provided callback function `cb` with the current value of the input element.
- **Output**: Returns the created HTML input element.


---
### el\_creatediv\_input
The `el_creatediv_input` function creates a div element containing a labeled input element with specified attributes and a callback for value changes.
- **Inputs**:
    - `id`: A string representing the unique identifier for the input element.
    - `label`: A string representing the text label for the input element.
    - `type`: A string specifying the type of the input element (e.g., 'text', 'number').
    - `defaultValue`: The initial value to be set for the input element.
    - `cb`: A callback function that is triggered when the input value changes.
    - `className`: A string representing the CSS class name for the div element, defaulting to 'gridx2'.
- **Control Flow**:
    - Create a div element and set its class name to the provided className or 'gridx2' by default.
    - Create a label element, set its 'for' attribute to the input's id, and set its inner text to the provided label.
    - Append the label element to the div.
    - Call `el_create_input` to create an input element with the specified id, type, defaultValue, and callback function.
    - Append the created input element to the div.
    - Return an object containing the div and the input element.
- **Output**: An object containing the created div element and the input element.


