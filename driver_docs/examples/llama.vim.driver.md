# Purpose
This code is a Vim script file designed to integrate LLM-based text completion functionality into the Neovim or Vim text editors using a server instance of llama.cpp. It provides a relatively narrow functionality focused on enhancing text editing by offering predictive text suggestions based on the context around the cursor. The script requires a running llama.cpp server and a FIM-compatible model to function, and it uses curl to communicate with the server. The file includes configuration options for customizing the behavior of the text completion, such as the number of lines to consider as context, the maximum number of tokens to predict, and the frequency of processing queued chunks. Additionally, it defines key mappings for accepting suggestions and includes functions for managing the ring buffer of text chunks used as extra context for the server. This script is intended to be symlinked or copied to the Neovim autoload directory, indicating its role as a plugin or extension to enhance the editor's capabilities.
# Global Variables

---
### s:default\_config
- **Type**: `dictionary`
- **Description**: The `s:default_config` variable is a dictionary that holds the default configuration settings for the LLM-based text completion plugin using llama.cpp. It includes various parameters such as the server endpoint, number of lines for prefix and suffix, maximum number of tokens to predict, and settings for the ring buffer used for extra context.
- **Use**: This variable is used to initialize the `g:llama_config` variable with default settings if no custom configuration is provided by the user.


---
### g:llama\_config
- **Type**: `Dictionary`
- **Description**: `g:llama_config` is a global dictionary variable that holds configuration settings for the llama.vim plugin, which integrates with a llama.cpp server for text completion in Vim or Neovim. It includes parameters such as the server endpoint, number of lines for context, prediction limits, and settings for automatic completion and ring buffer management.
- **Use**: This variable is used to store and retrieve configuration settings for the llama.vim plugin, allowing it to interact with the llama.cpp server for text completion tasks.


# Functions

---
### s:get\_indent
The `s:get_indent` function calculates the indentation level of a given string by counting the number of leading tab characters and adjusting for the tab stop setting.
- **Inputs**:
    - `str`: A string whose leading indentation is to be calculated.
- **Control Flow**:
    - Initialize a counter `l:count` to zero to keep track of the indentation level.
    - Iterate over each character in the input string `a:str`.
    - For each character, check if it is a tab character ('\t').
    - If it is a tab character, increment `l:count` by the value of `&tabstop - 1` to account for the tab width.
    - If a non-tab character is encountered, break out of the loop.
    - Return the final value of `l:count` as the indentation level.
- **Output**: The function returns an integer representing the indentation level of the input string, calculated based on the number of leading tab characters.


---
### s:rand
The `s:rand` function generates a random integer within a specified inclusive range.
- **Inputs**:
    - `i0`: The lower bound of the range (inclusive) from which to generate a random integer.
    - `i1`: The upper bound of the range (inclusive) from which to generate a random integer.
- **Control Flow**:
    - The function takes two arguments, `i0` and `i1`, which define the inclusive range for the random number.
    - It calculates the random number by adding `i0` to the result of `rand() % (i1 - i0 + 1)`, ensuring the result is within the specified range.
- **Output**: An integer that is randomly selected from the inclusive range between `i0` and `i1`.


---
### llama\#init
The `llama#init` function initializes the llama.vim plugin for LLM-based text completion by setting up necessary configurations, mappings, and autocommands.
- **Inputs**: None
- **Control Flow**:
    - Check if 'curl' is executable; if not, display a warning and exit.
    - Initialize various state variables related to cursor position, line content, and ring buffer for extra context.
    - Set up highlight groups for displaying hints and information if supported by the editor.
    - Define an autocommand group 'llama' to handle various events like entering insert mode, cursor movement, buffer changes, and file writes, triggering functions for FIM completion and chunk gathering.
    - Cancel any existing FIM suggestions and start background updates of the ring buffer if configured.
- **Output**: The function does not return any output; it sets up the environment and state for the llama.vim plugin to function.


---
### s:chunk\_sim
The `s:chunk_sim` function calculates the similarity between two text chunks by comparing their lines.
- **Inputs**:
    - `c0`: The first text chunk, represented as a list of lines.
    - `c1`: The second text chunk, also represented as a list of lines.
- **Control Flow**:
    - Initialize variables to store the number of lines in each chunk and a counter for common lines.
    - Iterate over each line in the first chunk.
    - For each line in the first chunk, iterate over each line in the second chunk.
    - If a line from the first chunk matches a line from the second chunk, increment the common line counter and break out of the inner loop.
    - Calculate the similarity score as twice the number of common lines divided by the total number of lines in both chunks.
- **Output**: Returns a floating-point number between 0 and 1 representing the similarity score, where 0 indicates no similarity and 1 indicates high similarity.


---
### s:pick\_chunk
The `s:pick_chunk` function selects a random chunk of text from the provided input and queues it for processing, with options to avoid modified buffers and evict similar chunks.
- **Inputs**:
    - `text`: A list of strings representing lines of text from which a chunk will be selected.
    - `no_mod`: A boolean flag indicating whether to avoid picking chunks from buffers with pending changes.
    - `do_evict`: A boolean flag indicating whether to evict chunks that are very similar to the new one.
- **Control Flow**:
    - Check if chunks should be picked from buffers with pending changes or non-file buffers based on `no_mod` flag.
    - Return immediately if extra context option is disabled or if the text is too small.
    - Determine the range of lines to form a chunk based on `g:llama_config.ring_chunk_size`.
    - Check if the selected chunk already exists in `s:ring_chunks` or `s:ring_queued` and return if it does.
    - Evict queued chunks that are very similar to the new one if `do_evict` is true.
    - Add the new chunk to `s:ring_queued` if it doesn't already exist and manage the queue size.
- **Output**: The function does not return a value; it modifies the `s:ring_queued` list by adding a new chunk if applicable.


---
### s:ring\_update
The `s:ring_update` function periodically processes queued text chunks, moving them to a ring buffer for use as extra context in text completion requests.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - The function starts a timer with a delay specified by `g:llama_config.ring_update_ms` to call itself periodically.
    - It checks if the current mode is normal or if the cursor hasn't moved for a while before proceeding.
    - If there are no queued chunks (`s:ring_queued` is empty), the function returns without doing anything.
    - If the ring buffer (`s:ring_chunks`) is full, it removes the oldest chunk to make space for a new one.
    - It moves the first chunk from the queue (`s:ring_queued`) to the ring buffer (`s:ring_chunks`).
    - The function prepares a JSON request with the current extra context from the ring buffer and sends it asynchronously to the server using a curl command.
- **Output**: The function does not return any value; it updates the state of `s:ring_chunks` and `s:ring_queued` and sends a request to the server.


---
### llama\#fim\_inline
The `llama#fim_inline` function triggers the FIM (Fill-In-the-Middle) completion process for text editing in Vim or Neovim using a llama.cpp server.
- **Inputs**:
    - `is_auto`: A boolean indicating whether the FIM completion is triggered automatically or manually.
- **Control Flow**:
    - The function calls `llama#fim` with the `is_auto` parameter to initiate the FIM process.
    - It returns an empty string to satisfy the expression mapping requirement in Vim.
    - The `llama#fim` function checks if a suggestion is already shown and cancels it if necessary.
    - It prevents sending repeated requests too quickly by using a timer to delay the next request.
    - The function gathers the current cursor position, line content, and context around the cursor for the FIM request.
    - It prepares a JSON request with the gathered context and sends it to the llama.cpp server using an asynchronous curl command.
    - If the cursor has moved significantly, it gathers additional context chunks for future requests.
- **Output**: The function returns an empty string, which is used for expression mapping in Vim to trigger the FIM process.


---
### llama\#fim
The `llama#fim` function sends local and extra context around the cursor to a server for text completion and displays the suggestion in the editor.
- **Inputs**:
    - `is_auto`: A boolean indicating whether the function is triggered automatically or manually.
- **Control Flow**:
    - Check if a suggestion is already shown and cancel if necessary.
    - Cancel any existing FIM operation and check if the request is being sent too quickly.
    - Initialize variables for cursor position, current line, and context lines around the cursor.
    - Prepare the prefix, prompt, and suffix strings from the current buffer content.
    - Prepare extra context data from previously gathered chunks.
    - Create a JSON request with the context and send it to the server using an asynchronous job.
    - If the cursor has moved significantly, gather additional context chunks for future requests.
- **Output**: The function does not return a value but updates the editor with a text completion suggestion based on the server's response.


---
### llama\#fim\_accept
The `llama#fim_accept` function inserts a text suggestion at the cursor location in a Vim buffer, either accepting the entire suggestion or just the first line, and then cancels the suggestion display.
- **Inputs**:
    - `first_line`: A boolean indicating whether to accept only the first line of the suggestion (true) or the entire suggestion (false).
- **Control Flow**:
    - Check if a suggestion can be accepted and if there is content available.
    - If the suggestion can be accepted, insert the suggestion at the current cursor position.
    - If `first_line` is false and there is more than one line in the suggestion, append the remaining lines below the current line.
    - Move the cursor to the end of the accepted text, adjusting based on whether only the first line or the entire suggestion was accepted.
    - Call `llama#fim_cancel` to clear the suggestion display and mappings.
- **Output**: The function does not return a value; it modifies the buffer by inserting the suggestion and updates the cursor position.


---
### llama\#fim\_cancel
The `llama#fim_cancel` function cancels any ongoing FIM (Fill-In-the-Middle) completion process and clears related virtual text and mappings in the editor.
- **Inputs**: None
- **Control Flow**:
    - Sets the `s:hint_shown` variable to `v:false` to indicate that no hint is currently shown.
    - Retrieves the current buffer number using `bufnr('%')`.
    - Checks if the environment supports Neovim's virtual text or Vim's text properties to clear the virtual text accordingly.
    - If using Neovim, it creates a namespace for virtual text and clears it using `nvim_buf_clear_namespace`.
    - If using Vim, it removes text properties of types `s:hlgroup_hint` and `s:hlgroup_info`.
    - Silently unmaps the buffer-specific mappings for `<Tab>`, `<S-Tab>`, and `<Esc>` keys.
- **Output**: The function does not return any value; it performs cleanup operations to cancel FIM completion.


---
### s:on\_move
The `s:on_move` function updates the last cursor movement time and cancels any ongoing FIM (Fill-In-the-Middle) completion process.
- **Inputs**: None
- **Control Flow**:
    - The function sets the variable `s:t_last_move` to the current relative time using `reltime()`, marking the last time the cursor moved.
    - It then calls the `llama#fim_cancel()` function to cancel any ongoing FIM completion process.
- **Output**: The function does not return any value.


---
### s:fim\_on\_stdout
The `s:fim_on_stdout` function processes the output from a server job to display text suggestions in a text editor.
- **Inputs**:
    - `pos_x`: The x-coordinate (column) of the cursor position when the FIM request was made.
    - `pos_y`: The y-coordinate (line) of the cursor position when the FIM request was made.
    - `is_auto`: A boolean indicating whether the FIM completion was triggered automatically.
    - `job_id`: The identifier of the job that produced the output.
    - `data`: The output data from the server job, typically a list of strings.
    - `event`: An optional event parameter, defaulting to null, which may contain additional event information.
- **Control Flow**:
    - Check if the data is empty or if the cursor position has changed since the request was made; if so, return early.
    - Verify that the editor is in insert mode; if not, return early.
    - Parse the server response to extract the content and generation settings.
    - Check for errors in the server response and handle them appropriately.
    - Process the content to remove any redundant or repeated text.
    - Display the processed suggestion as virtual text in the editor, using different methods depending on the editor's capabilities.
    - Set up keyboard shortcuts for accepting the suggestion.
- **Output**: The function does not return a value but updates the editor's state to display text suggestions based on the server's output.


---
### s:fim\_on\_exit
The `s:fim_on_exit` function handles the completion of a job by checking the exit code and resetting the current job state.
- **Inputs**:
    - `job_id`: The identifier of the job that has completed.
    - `exit_code`: The exit code returned by the job, indicating success or failure.
    - `event`: An optional parameter, defaulting to `v:null`, representing the event associated with the job's completion.
- **Control Flow**:
    - Check if the exit code is not zero, indicating a failure.
    - If the job failed, output a message with the exit code.
    - Reset the `s:current_job` variable to `v:null` to indicate no active job.
- **Output**: There is no return value; the function performs side effects by logging a message and resetting the job state.


