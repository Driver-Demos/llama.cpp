# Purpose
The provided C++ code defines a class named `llama_kv_cells_unified`, which is designed to manage metadata about key-value (KV) cells that can be part of multiple sequences simultaneously. This class is part of a broader system, likely related to the "llama" namespace or project, as indicated by the included headers `llama.h` and `llama-cparams.h`. The class provides a collection of methods to manipulate and query the state of these KV cells, including operations to reset, resize, move, copy, and set the state of cells. It also includes functionality to manage sequences within the cells, such as adding, removing, and checking sequences, as well as tracking the positions of sequences.

The class uses several important technical components, such as `std::vector` for dynamic arrays, `std::set` for managing unique elements, and `std::bitset` for efficiently handling sequence occupancy within cells. The class maintains internal state variables like `pos`, `shift`, and `seq` to track the position, shift, and sequence occupancy of each cell, respectively. Additionally, it uses a `std::set` named `used` to keep track of indices of cells that are currently in use. The class provides a narrow, specialized functionality focused on managing the state and sequence information of KV cells, making it a utility component likely intended for use within a larger system rather than as a standalone application. The presence of private helper functions like [`seq_pos_rm`](#llama_kv_cells_unifiedseq_pos_rm) and [`seq_pos_add`](#llama_kv_cells_unifiedseq_pos_add) suggests an emphasis on maintaining internal consistency when updating sequence positions.
# Imports and Dependencies

---
- `llama.h`
- `llama-cparams.h`
- `bitset`
- `cassert`
- `vector`
- `set`


# Data Structures

---
### llama\_kv\_cells\_unified<!-- {{#data_structure:llama_kv_cells_unified}} -->
- **Type**: `class`
- **Members**:
    - `has_shift`: A boolean flag indicating if any shifts have been applied to the positions.
    - `used`: A set of indices representing used cells where pos[i] is not -1.
    - `pos`: A vector storing the positions of cells, with -1 indicating an empty cell.
    - `shift`: A vector accumulating shifts applied to the pos array since the last reset_shift() call.
    - `seq`: A vector of bitsets indicating which sequences occupy each cell.
    - `seq_pos`: An array of sets tracking the positions present for each sequence.
- **Description**: The `llama_kv_cells_unified` class is a data structure designed to manage key-value cells that can be part of multiple sequences simultaneously. It maintains the state of each cell, including its position, any shifts applied, and the sequences it belongs to. The class provides functionality to reset, resize, and manipulate the cells, including moving, copying, and setting their states. It also tracks used cells and manages sequence positions efficiently, allowing for operations like defragmentation and state restoration.
- **Member Functions**:
    - [`llama_kv_cells_unified::reset`](#llama_kv_cells_unifiedreset)
    - [`llama_kv_cells_unified::reset_shift`](#llama_kv_cells_unifiedreset_shift)
    - [`llama_kv_cells_unified::size`](#llama_kv_cells_unifiedsize)
    - [`llama_kv_cells_unified::resize`](#llama_kv_cells_unifiedresize)
    - [`llama_kv_cells_unified::is_empty`](#llama_kv_cells_unifiedis_empty)
    - [`llama_kv_cells_unified::get_used`](#llama_kv_cells_unifiedget_used)
    - [`llama_kv_cells_unified::used_min`](#llama_kv_cells_unifiedused_min)
    - [`llama_kv_cells_unified::used_max_p1`](#llama_kv_cells_unifiedused_max_p1)
    - [`llama_kv_cells_unified::get_has_shift`](#llama_kv_cells_unifiedget_has_shift)
    - [`llama_kv_cells_unified::mv`](#llama_kv_cells_unifiedmv)
    - [`llama_kv_cells_unified::cp`](#llama_kv_cells_unifiedcp)
    - [`llama_kv_cells_unified::set`](#llama_kv_cells_unifiedset)
    - [`llama_kv_cells_unified::rm`](#llama_kv_cells_unifiedrm)
    - [`llama_kv_cells_unified::seq_rm`](#llama_kv_cells_unifiedseq_rm)
    - [`llama_kv_cells_unified::seq_keep`](#llama_kv_cells_unifiedseq_keep)
    - [`llama_kv_cells_unified::seq_count`](#llama_kv_cells_unifiedseq_count)
    - [`llama_kv_cells_unified::seq_has`](#llama_kv_cells_unifiedseq_has)
    - [`llama_kv_cells_unified::seq_add`](#llama_kv_cells_unifiedseq_add)
    - [`llama_kv_cells_unified::seq_get`](#llama_kv_cells_unifiedseq_get)
    - [`llama_kv_cells_unified::seq_pos_min`](#llama_kv_cells_unifiedseq_pos_min)
    - [`llama_kv_cells_unified::seq_pos_max`](#llama_kv_cells_unifiedseq_pos_max)
    - [`llama_kv_cells_unified::pos_get`](#llama_kv_cells_unifiedpos_get)
    - [`llama_kv_cells_unified::get_shift`](#llama_kv_cells_unifiedget_shift)
    - [`llama_kv_cells_unified::pos_in`](#llama_kv_cells_unifiedpos_in)
    - [`llama_kv_cells_unified::pos_set`](#llama_kv_cells_unifiedpos_set)
    - [`llama_kv_cells_unified::pos_add`](#llama_kv_cells_unifiedpos_add)
    - [`llama_kv_cells_unified::pos_div`](#llama_kv_cells_unifiedpos_div)
    - [`llama_kv_cells_unified::seq_pos_rm`](#llama_kv_cells_unifiedseq_pos_rm)
    - [`llama_kv_cells_unified::seq_pos_add`](#llama_kv_cells_unifiedseq_pos_add)

**Methods**

---
#### llama\_kv\_cells\_unified::reset<!-- {{#callable:llama_kv_cells_unified::reset}} -->
The `reset` function reinitializes the state of the `llama_kv_cells_unified` object by resetting positions, shifts, sequences, and clearing used indices and sequence positions.
- **Inputs**: None
- **Control Flow**:
    - Iterate over each element in the `pos` vector, setting each position to -1, each shift to 0, and resetting each sequence.
    - Set the `has_shift` flag to false.
    - Clear the `used` set, which tracks indices of used cells.
    - Iterate over each sequence position set in `seq_pos` and clear them.
- **Output**: The function does not return any value; it modifies the internal state of the `llama_kv_cells_unified` object.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::reset\_shift<!-- {{#callable:llama_kv_cells_unified::reset_shift}} -->
The `reset_shift` function resets the `has_shift` flag to false and sets all elements in the `shift` vector to zero.
- **Inputs**: None
- **Control Flow**:
    - Set the `has_shift` boolean member variable to `false`.
    - Iterate over each element in the `shift` vector using a for loop.
    - Set each element in the `shift` vector to `0`.
- **Output**: The function does not return any value (void).
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::size<!-- {{#callable:llama_kv_cells_unified::size}} -->
The `size` function returns the number of elements in the `pos` vector of the `llama_kv_cells_unified` class.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `pos` vector, which is a member of the `llama_kv_cells_unified` class.
    - It calls the `size()` method on the `pos` vector to get the number of elements it contains.
    - The function returns this size as a `uint32_t` value.
- **Output**: The function returns a `uint32_t` representing the number of elements in the `pos` vector.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::resize<!-- {{#callable:llama_kv_cells_unified::resize}} -->
The `resize` function adjusts the size of the `pos`, `shift`, and `seq` vectors to a specified number of elements and then resets the state of the `llama_kv_cells_unified` object.
- **Inputs**:
    - `n`: A `uint32_t` representing the new size for the `pos`, `shift`, and `seq` vectors.
- **Control Flow**:
    - The function resizes the `pos` vector to have `n` elements.
    - The function resizes the `shift` vector to have `n` elements.
    - The function resizes the `seq` vector to have `n` elements.
    - The function calls the [`reset`](#llama_kv_cells_unifiedreset) method to initialize or clear the state of the object.
- **Output**: This function does not return any value; it modifies the internal state of the `llama_kv_cells_unified` object.
- **Functions called**:
    - [`llama_kv_cells_unified::reset`](#llama_kv_cells_unifiedreset)
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::is\_empty<!-- {{#callable:llama_kv_cells_unified::is_empty}} -->
The `is_empty` function checks if a specific cell in the `pos` vector is empty by verifying if its value is -1.
- **Inputs**:
    - `i`: An unsigned 32-bit integer representing the index of the cell to check in the `pos` vector.
- **Control Flow**:
    - The function asserts that the index `i` is within the bounds of the `pos` vector size.
    - It asserts that the value at `pos[i]` is either -1 or non-negative, ensuring data integrity.
    - The function returns true if `pos[i]` is -1, indicating the cell is empty.
- **Output**: A boolean value indicating whether the cell at index `i` in the `pos` vector is empty (true if empty, false otherwise).
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::get\_used<!-- {{#callable:llama_kv_cells_unified::get_used}} -->
The `get_used` function returns the number of used cells in the `llama_kv_cells_unified` class.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `used` set, which contains indices of used cells.
    - It returns the size of the `used` set, which represents the number of used cells.
- **Output**: The function returns a `uint32_t` representing the number of used cells.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::used\_min<!-- {{#callable:llama_kv_cells_unified::used_min}} -->
The `used_min` function returns the index of the first used cell in the `llama_kv_cells_unified` class, or 0 if no cells are used.
- **Inputs**: None
- **Control Flow**:
    - Check if the `used` set is empty.
    - If `used` is empty, return 0.
    - If `used` is not empty, return the smallest element in the `used` set, which is the index of the first used cell.
- **Output**: The function returns a `uint32_t` representing the index of the first used cell, or 0 if no cells are used.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::used\_max\_p1<!-- {{#callable:llama_kv_cells_unified::used_max_p1}} -->
The `used_max_p1` function returns the index of the last used cell plus one, or zero if no cells are used.
- **Inputs**: None
- **Control Flow**:
    - Check if the `used` set is empty.
    - If `used` is empty, return 0.
    - If `used` is not empty, return the last element in `used` incremented by 1.
- **Output**: The function returns a `uint32_t` representing the index of the last used cell plus one, or zero if no cells are used.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::get\_has\_shift<!-- {{#callable:llama_kv_cells_unified::get_has_shift}} -->
The `get_has_shift` function returns the current state of the `has_shift` flag, indicating whether any positional shifts have been applied to the cells.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the `has_shift` member variable.
- **Output**: A boolean value indicating whether any shifts have been applied to the positions of the cells.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::mv<!-- {{#callable:llama_kv_cells_unified::mv}} -->
The `mv` function moves the data from one cell to another within the `llama_kv_cells_unified` structure, updating the position, shift, and sequence data, and adjusting the set of used indices accordingly.
- **Inputs**:
    - `isrc`: The index of the source cell from which data is to be moved.
    - `idst`: The index of the destination cell to which data is to be moved.
- **Control Flow**:
    - The function begins by asserting that both `isrc` and `idst` are valid indices within the bounds of the `pos` vector.
    - It then copies the `pos`, `shift`, and `seq` data from the source index `isrc` to the destination index `idst`.
    - The source index `isrc` is then reset: `pos[isrc]` is set to -1, `shift[isrc]` is set to 0, and `seq[isrc]` is reset.
    - The `used` set is updated by removing `isrc` and inserting `idst`.
- **Output**: The function does not return any value; it modifies the internal state of the `llama_kv_cells_unified` object.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::cp<!-- {{#callable:llama_kv_cells_unified::cp}} -->
The `cp` function copies a specified range of cells from the current `llama_kv_cells_unified` object into a new `llama_kv_cells_unified` object.
- **Inputs**:
    - `i`: The starting index of the range of cells to copy.
    - `n`: The number of cells to copy from the starting index.
- **Control Flow**:
    - The function asserts that the range [i, i + n) is within the bounds of the current object's `pos` vector.
    - A new `llama_kv_cells_unified` object `res` is created and resized to hold `n` cells.
    - A loop iterates over the range [i, i + n), copying the `pos` and `seq` values from the current object to `res`.
    - The function asserts that the `shift` value for each copied cell is zero.
    - The function returns the newly created `res` object containing the copied cells.
- **Output**: A new `llama_kv_cells_unified` object containing the copied range of cells.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::set<!-- {{#callable:llama_kv_cells_unified::set}} -->
The `set` function updates a range of cells in the current `llama_kv_cells_unified` object with the state from another `llama_kv_cells_unified` object, managing the `used` set and sequence positions accordingly.
- **Inputs**:
    - `i`: The starting index in the current object where the update will begin.
    - `other`: A reference to another `llama_kv_cells_unified` object whose state will be copied into the current object.
- **Control Flow**:
    - Assert that the range [i, i + other.pos.size()) is within the bounds of the current object's `pos` vector.
    - Iterate over each index `j` in the range of `other.pos.size()`.
    - If the current object's position at `i + j` is -1 and `other`'s position at `j` is not -1, insert `i + j` into the `used` set.
    - If the current object's position at `i + j` is not -1 and `other`'s position at `j` is -1, erase `i + j` from the `used` set.
    - If the current object's position at `i + j` is not -1, call `seq_pos_rm(i + j)` to update sequence positions.
    - Copy the position and sequence from `other` to the current object at index `i + j`.
    - If the new position at `i + j` is not -1, call `seq_pos_add(i + j)` to update sequence positions.
    - Assert that the shift at `i + j` is 0.
- **Output**: The function does not return a value; it modifies the state of the current `llama_kv_cells_unified` object.
- **Functions called**:
    - [`llama_kv_cells_unified::seq_pos_rm`](#llama_kv_cells_unifiedseq_pos_rm)
    - [`llama_kv_cells_unified::seq_pos_add`](#llama_kv_cells_unifiedseq_pos_add)
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::rm<!-- {{#callable:llama_kv_cells_unified::rm}} -->
The `rm` function clears a non-empty cell at a specified index in the `llama_kv_cells_unified` data structure.
- **Inputs**:
    - `i`: The index of the cell to be cleared, which must be within the bounds of the `pos` vector and must not be -1.
- **Control Flow**:
    - The function asserts that the index `i` is within the bounds of the `pos` vector and that the cell at index `i` is not empty (i.e., `pos[i]` is not -1).
    - It calls `seq_pos_rm(i)` to remove the cell's position from any sequence position sets.
    - The position at index `i` in the `pos` vector is set to -1, indicating the cell is now empty.
    - The sequence bitset at index `i` is reset, clearing any sequence information associated with the cell.
    - The index `i` is removed from the `used` set, indicating the cell is no longer in use.
- **Output**: The function does not return any value; it modifies the state of the `llama_kv_cells_unified` object by clearing the specified cell.
- **Functions called**:
    - [`llama_kv_cells_unified::seq_pos_rm`](#llama_kv_cells_unifiedseq_pos_rm)
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::seq\_rm<!-- {{#callable:llama_kv_cells_unified::seq_rm}} -->
The `seq_rm` function removes a specific sequence ID from a cell and checks if the cell becomes empty as a result.
- **Inputs**:
    - `i`: The index of the cell in the `pos` vector from which the sequence ID should be removed.
    - `seq_id`: The sequence ID to be removed from the specified cell.
- **Control Flow**:
    - The function begins by asserting that the index `i` is within bounds, the sequence ID `seq_id` is present in the cell, the position at index `i` is not -1, and `seq_id` is non-negative.
    - The sequence ID `seq_id` is reset (removed) from the bitset at index `i`.
    - The position of the sequence ID `seq_id` is erased from the `seq_pos` set corresponding to `seq_id`.
    - The function checks if the bitset at index `i` is now empty (i.e., no sequences are present).
    - If the bitset is empty, the position at index `i` is set to -1, the index `i` is removed from the `used` set, and the function returns `true`.
    - If the bitset is not empty, the function returns `false`.
- **Output**: A boolean value indicating whether the cell at index `i` became empty after removing the sequence ID.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::seq\_keep<!-- {{#callable:llama_kv_cells_unified::seq_keep}} -->
The `seq_keep` function checks if a sequence ID is present in a specified cell, updates the cell's state accordingly, and returns whether the cell becomes empty.
- **Inputs**:
    - `i`: The index of the cell in the `pos` vector to be checked and potentially modified.
    - `seq_id`: The sequence ID to be checked against the cell at index `i`.
- **Control Flow**:
    - Assert that the index `i` is within the bounds of the `pos` vector.
    - Check if the sequence ID `seq_id` is present in the cell at index `i` using `seq[i].test(seq_id)`.
    - If `seq_id` is present, remove the cell's position from `seq_pos`, reset the cell, set the sequence ID again, and insert the position back into `seq_pos` for `seq_id`, then return `false`.
    - If any sequence is present in the cell, remove the cell's position from `seq_pos`, reset the cell, set the position to -1, remove the index from `used`, and return `true`.
    - Assert that the position at index `i` is -1, indicating the cell is empty, and return `false`.
- **Output**: A boolean value indicating whether the cell at index `i` becomes empty after the operation.
- **Functions called**:
    - [`llama_kv_cells_unified::seq_pos_rm`](#llama_kv_cells_unifiedseq_pos_rm)
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::seq\_count<!-- {{#callable:llama_kv_cells_unified::seq_count}} -->
The `seq_count` function returns the number of different sequences present in a specified cell of the `llama_kv_cells_unified` class.
- **Inputs**:
    - `i`: An unsigned 32-bit integer representing the index of the cell in the `pos` vector to be queried.
- **Control Flow**:
    - The function asserts that the index `i` is within the bounds of the `pos` vector size.
    - It asserts that the position at index `i` in the `pos` vector is not -1, indicating the cell is not empty.
    - It returns the count of set bits in the `seq[i]` bitset, which represents the number of sequences in the cell.
- **Output**: An integer representing the number of different sequences in the specified cell.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::seq\_has<!-- {{#callable:llama_kv_cells_unified::seq_has}} -->
The `seq_has` function checks if a specific sequence ID is present in a given cell of the `llama_kv_cells_unified` data structure.
- **Inputs**:
    - `i`: An unsigned 32-bit integer representing the index of the cell to check within the `pos` vector.
    - `seq_id`: A `llama_seq_id` representing the sequence ID to check for presence in the specified cell.
- **Control Flow**:
    - The function asserts that the index `i` is within the bounds of the `pos` vector size.
    - It asserts that the `seq_id` is non-negative.
    - The function returns the result of testing if the sequence ID `seq_id` is present in the bitset `seq[i]`.
- **Output**: A boolean value indicating whether the specified sequence ID is present in the cell at index `i`.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::seq\_add<!-- {{#callable:llama_kv_cells_unified::seq_add}} -->
The `seq_add` function adds a sequence identifier to a specified cell in the `llama_kv_cells_unified` data structure, ensuring the cell is not empty and the sequence is not already present.
- **Inputs**:
    - `i`: An unsigned 32-bit integer representing the index of the cell in the `pos` vector.
    - `seq_id`: A `llama_seq_id` representing the sequence identifier to be added to the cell.
- **Control Flow**:
    - The function asserts that the index `i` is within the bounds of the `pos` vector.
    - It asserts that the cell at index `i` is not empty by checking `pos[i] != -1`.
    - It asserts that the sequence identifier `seq_id` is not already present in the cell by checking `!seq[i].test(seq_id)`.
    - The sequence identifier `seq_id` is added to the bitset `seq[i]` using `seq[i].set(seq_id)`.
    - The position `pos[i]` is inserted into the set `seq_pos[seq_id]` to track the position of the sequence.
- **Output**: The function does not return any value.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::seq\_get<!-- {{#callable:llama_kv_cells_unified::seq_get}} -->
The `seq_get` function retrieves the sequence ID of a cell at a given index if it contains exactly one sequence.
- **Inputs**:
    - `i`: An unsigned 32-bit integer representing the index of the cell in the sequence vector.
- **Control Flow**:
    - The function asserts that the cell at index `i` contains exactly one sequence using `assert(seq[i].count() == 1)`.
    - It iterates over all possible sequence IDs from 0 to `LLAMA_MAX_PARALLEL_SEQUENCES - 1`.
    - For each sequence ID `s`, it checks if the sequence is present in the cell using `seq[i].test(s)`.
    - If a sequence is found, it returns the sequence ID `s`.
    - If no sequence is found after the loop, it returns -1.
- **Output**: The function returns the sequence ID of the cell at index `i` if it contains exactly one sequence, otherwise it returns -1.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::seq\_pos\_min<!-- {{#callable:llama_kv_cells_unified::seq_pos_min}} -->
The `seq_pos_min` function returns the minimum position of a given sequence ID within the cells, or -1 if the sequence is not present.
- **Inputs**:
    - `seq_id`: An integer representing the sequence ID for which the minimum position is to be found.
- **Control Flow**:
    - The function asserts that the `seq_id` is non-negative and less than `LLAMA_MAX_PARALLEL_SEQUENCES`.
    - It checks if the set `seq_pos[seq_id]` is empty; if so, it returns -1.
    - If the set is not empty, it returns the first element of the set, which represents the minimum position.
- **Output**: The function returns a `llama_pos` which is the minimum position of the specified sequence ID, or -1 if the sequence is not present.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::seq\_pos\_max<!-- {{#callable:llama_kv_cells_unified::seq_pos_max}} -->
The `seq_pos_max` function returns the maximum position of a given sequence ID within the `llama_kv_cells_unified` class, or -1 if the sequence is not present.
- **Inputs**:
    - `seq_id`: An integer representing the sequence ID for which the maximum position is to be determined.
- **Control Flow**:
    - The function asserts that `seq_id` is non-negative and less than `LLAMA_MAX_PARALLEL_SEQUENCES`.
    - It checks if the `seq_pos` set for the given `seq_id` is empty.
    - If the set is empty, it returns -1.
    - If the set is not empty, it returns the last element of the set, which represents the maximum position.
- **Output**: The function returns a `llama_pos` value, which is the maximum position of the specified sequence ID, or -1 if the sequence is not present.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::pos\_get<!-- {{#callable:llama_kv_cells_unified::pos_get}} -->
The `pos_get` function retrieves the position value at a specified index from the `pos` vector, ensuring the index is valid and the position is not empty.
- **Inputs**:
    - `i`: A 32-bit unsigned integer representing the index in the `pos` vector from which to retrieve the position value.
- **Control Flow**:
    - The function asserts that the index `i` is within the bounds of the `pos` vector size.
    - It asserts that the position at index `i` is not equal to -1, indicating the cell is not empty.
    - The function returns the position value at index `i` from the `pos` vector.
- **Output**: The function returns a `llama_pos` value, which is the position at the specified index `i` in the `pos` vector.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::get\_shift<!-- {{#callable:llama_kv_cells_unified::get_shift}} -->
The `get_shift` function retrieves the shift value for a specified index in the `shift` vector, ensuring the index is valid and the corresponding position is not empty.
- **Inputs**:
    - `i`: An unsigned 32-bit integer representing the index of the cell for which the shift value is to be retrieved.
- **Control Flow**:
    - The function asserts that the index `i` is within the bounds of the `pos` vector size.
    - It asserts that the position at index `i` is not -1, indicating the cell is not empty.
    - The function returns the shift value at index `i` from the `shift` vector.
- **Output**: The function returns a `llama_pos` value, which is the shift value at the specified index `i`.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::pos\_in<!-- {{#callable:llama_kv_cells_unified::pos_in}} -->
The `pos_in` function checks if a specified position in the `pos` vector is within a given range [p0, p1).
- **Inputs**:
    - `i`: An index into the `pos` vector, representing the position to check.
    - `p0`: The lower bound of the range to check against, inclusive.
    - `p1`: The upper bound of the range to check against, exclusive.
- **Control Flow**:
    - The function asserts that the index `i` is within the bounds of the `pos` vector.
    - It then checks if the value at `pos[i]` is greater than or equal to `p0` and less than `p1`.
- **Output**: Returns `true` if `pos[i]` is within the range [p0, p1), otherwise returns `false`.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::pos\_set<!-- {{#callable:llama_kv_cells_unified::pos_set}} -->
The `pos_set` function sets the position of an empty cell in the `llama_kv_cells_unified` class and marks it as used.
- **Inputs**:
    - `i`: An index of type `uint32_t` representing the position in the `pos` vector to be set.
    - `p`: A `llama_pos` value representing the new position to be assigned to the cell at index `i`.
- **Control Flow**:
    - The function asserts that the index `i` is within the bounds of the `pos` vector.
    - It asserts that the position at index `i` is currently unset (i.e., `pos[i] == -1`).
    - It asserts that the sequence bitset at index `i` is empty (i.e., `seq[i].none()`).
    - The position at index `i` is set to the value `p`.
    - The index `i` is added to the `used` set, marking it as used.
- **Output**: The function does not return any value.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::pos\_add<!-- {{#callable:llama_kv_cells_unified::pos_add}} -->
The `pos_add` function updates the position and shift of a specified cell by adding a given value, and marks the cell as shifted, potentially clearing it if the new position is negative.
- **Inputs**:
    - `i`: The index of the cell in the `pos` vector to be updated.
    - `d`: The value to be added to the position of the specified cell.
- **Control Flow**:
    - Assert that the index `i` is within the bounds of the `pos` vector and that the position at `pos[i]` is not -1.
    - Remove the current position of the cell from the sequence position tracking using `seq_pos_rm(i)`.
    - Add the value `d` to both `pos[i]` and `shift[i]`.
    - Re-add the updated position to the sequence position tracking using `seq_pos_add(i)`.
    - Set `has_shift` to true to indicate that a shift has occurred.
    - Check if the updated position `pos[i]` is negative.
    - If `pos[i]` is negative, remove the position from sequence tracking, reset the sequence bitset for the cell, set `pos[i]` to -1, remove the index from the `used` set, and return true.
    - If `pos[i]` is not negative, return false.
- **Output**: Returns a boolean indicating whether the cell became empty (true if `pos[i]` became negative, false otherwise).
- **Functions called**:
    - [`llama_kv_cells_unified::seq_pos_rm`](#llama_kv_cells_unifiedseq_pos_rm)
    - [`llama_kv_cells_unified::seq_pos_add`](#llama_kv_cells_unifiedseq_pos_add)
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::pos\_div<!-- {{#callable:llama_kv_cells_unified::pos_div}} -->
The `pos_div` function divides the position of a specified cell by a given divisor and updates the shift and sequence position accordingly, marking that a shift has occurred.
- **Inputs**:
    - `i`: An unsigned 32-bit integer representing the index of the cell in the `pos` vector.
    - `d`: An integer divisor by which the position of the cell at index `i` will be divided.
- **Control Flow**:
    - The function asserts that the index `i` is within the bounds of the `pos` vector and that the position at `pos[i]` is not -1, indicating the cell is not empty.
    - It stores the current position `pos[i]` in a temporary variable `p_old`.
    - The function calls `seq_pos_rm(i)` to remove the current position from the sequence position set.
    - The position `pos[i]` is divided by `d`, and the difference between `p_old` and the new `pos[i]` is added to `shift[i]`.
    - The function calls `seq_pos_add(i)` to add the updated position back to the sequence position set.
    - Finally, it sets `has_shift` to true, indicating that a shift has occurred.
- **Output**: The function does not return a value; it modifies the `pos`, `shift`, and `has_shift` members of the `llama_kv_cells_unified` class in place.
- **Functions called**:
    - [`llama_kv_cells_unified::seq_pos_rm`](#llama_kv_cells_unifiedseq_pos_rm)
    - [`llama_kv_cells_unified::seq_pos_add`](#llama_kv_cells_unifiedseq_pos_add)
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::seq\_pos\_rm<!-- {{#callable:llama_kv_cells_unified::seq_pos_rm}} -->
The `seq_pos_rm` function removes the position of a specified cell from all sequence position sets where the cell is currently active.
- **Inputs**:
    - `i`: The index of the cell whose position is to be removed from the sequence position sets.
- **Control Flow**:
    - Iterate over all possible sequences from 0 to LLAMA_MAX_PARALLEL_SEQUENCES.
    - For each sequence, check if the sequence is active in the cell at index `i` using the `test` method on the `seq` bitset.
    - If the sequence is active, remove the position of the cell from the corresponding sequence position set `seq_pos`.
- **Output**: This function does not return any value; it modifies the `seq_pos` data structure in place.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)


---
#### llama\_kv\_cells\_unified::seq\_pos\_add<!-- {{#callable:llama_kv_cells_unified::seq_pos_add}} -->
The `seq_pos_add` function inserts the position of a cell into the sequence position set for each sequence that the cell is part of.
- **Inputs**:
    - `i`: The index of the cell whose position is to be added to the sequence position sets.
- **Control Flow**:
    - Iterate over all possible sequences from 0 to `LLAMA_MAX_PARALLEL_SEQUENCES - 1`.
    - For each sequence, check if the cell at index `i` is part of the sequence using the `test` method on the `seq` bitset.
    - If the cell is part of the sequence, insert its position `pos[i]` into the corresponding sequence position set `seq_pos[s]`.
- **Output**: This function does not return any value; it modifies the `seq_pos` sets in place.
- **See also**: [`llama_kv_cells_unified`](#llama_kv_cells_unified)  (Data Structure)



