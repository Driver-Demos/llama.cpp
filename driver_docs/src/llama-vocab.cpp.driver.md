# Purpose
This C++ source code file is part of a larger software system that deals with tokenization, specifically for natural language processing (NLP) tasks. The file defines various structures and classes to handle different types of tokenizers, such as SentencePiece (SPM), Byte Pair Encoding (BPE), WordPiece (WPM), Unigram (UGM), and RWKV tokenizers. These tokenizers are used to convert text into tokens, which are the basic units of input for machine learning models, particularly in the context of language models like LLaMA.

The file includes several key components: a [`naive_trie`](#naive_trienaive_trie) structure for efficient token lookup, various tokenizer classes ([`llm_tokenizer_spm`](#llm_tokenizer_spmllm_tokenizer_spm), [`llm_tokenizer_bpe`](#llm_tokenizer_bpellm_tokenizer_bpe), etc.) that implement specific tokenization algorithms, and sessions for each tokenizer type that manage the tokenization process. The code also handles special tokens, such as beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens, and provides functionality for detokenizing, which is the process of converting tokens back into human-readable text. The file is designed to be part of a library that can be imported and used in other parts of a software system, providing a public API for tokenization tasks.
# Imports and Dependencies

---
- `llama-vocab.h`
- `ggml.h`
- `gguf.h`
- `llama-impl.h`
- `llama-model-loader.h`
- `unicode.h`
- `algorithm`
- `cassert`
- `cfloat`
- `climits`
- `cstdarg`
- `cstring`
- `forward_list`
- `map`
- `queue`
- `set`
- `unordered_map`
- `cctype`


# Data Structures

---
### naive\_trie<!-- {{#data_structure:naive_trie}} -->
- **Type**: `struct`
- **Members**:
    - `children`: A map that associates characters with child `naive_trie` nodes.
    - `has_value`: A boolean indicating if the current node represents a complete key.
    - `value`: An integer value associated with the key if `has_value` is true.
- **Description**: The `naive_trie` struct is a simple implementation of a trie data structure, which is used to store a dynamic set of strings. Each node in the trie can have multiple children, represented by a map from characters to `naive_trie` nodes. The `has_value` boolean indicates whether a node corresponds to the end of a valid key, and `value` stores an integer associated with that key. This structure allows for efficient insertion and retrieval of strings, supporting operations like inserting keys and finding the longest prefix of a given string.
- **Member Functions**:
    - [`naive_trie::naive_trie`](#naive_trienaive_trie)
    - [`naive_trie::insert`](#naive_trieinsert)
    - [`naive_trie::get_longest_prefix`](#naive_trieget_longest_prefix)
    - [`naive_trie::traverse`](#naive_trietraverse)

**Methods**

---
#### naive\_trie::naive\_trie<!-- {{#callable:naive_trie::naive_trie}} -->
The `naive_trie` constructor initializes a trie node with default values for `has_value` and `value`.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes the `has_value` member to `false`.
    - The constructor initializes the `value` member to `0`.
- **Output**: A `naive_trie` object with `has_value` set to `false` and `value` set to `0`.
- **See also**: [`naive_trie`](#naive_trie)  (Data Structure)


---
#### naive\_trie::insert<!-- {{#callable:naive_trie::insert}} -->
The `insert` function adds a key-value pair to a trie data structure, creating new nodes as necessary.
- **Inputs**:
    - `key`: A pointer to a character array representing the key to be inserted into the trie.
    - `len`: The length of the key, indicating how many characters from the key should be considered.
    - `value`: An optional integer value associated with the key, defaulting to 0 if not provided.
- **Control Flow**:
    - Check if the length of the key is zero; if so, set the current node's value and mark it as having a value, then return.
    - Extract the first character of the key and search for it in the current node's children map.
    - If the character is found, recursively call `insert` on the corresponding child node with the rest of the key.
    - If the character is not found, create a new child node for the character and recursively call `insert` on it with the rest of the key.
- **Output**: The function does not return a value; it modifies the trie in place.
- **Functions called**:
    - [`naive_trie::naive_trie`](#naive_trienaive_trie)
- **See also**: [`naive_trie`](#naive_trie)  (Data Structure)


---
#### naive\_trie::get\_longest\_prefix<!-- {{#callable:naive_trie::get_longest_prefix}} -->
The `get_longest_prefix` function finds the longest prefix of a given key that matches a path in a trie structure, starting from a specified offset.
- **Inputs**:
    - `key`: A pointer to a character array representing the key to search for in the trie.
    - `len`: The length of the key.
    - `offset`: An optional starting position in the key from which to begin the search, defaulting to 0.
- **Control Flow**:
    - Check if the length of the key is zero or if the offset equals the length, returning the key and offset if true.
    - Retrieve the character at the current offset from the key.
    - Search for this character in the children of the current trie node.
    - If the character is found, recursively call `get_longest_prefix` on the corresponding child node with an incremented offset.
    - If the character is not found, return the key and the current offset.
- **Output**: A `std::pair` containing the original key and the length of the longest matching prefix found.
- **See also**: [`naive_trie`](#naive_trie)  (Data Structure)


---
#### naive\_trie::traverse<!-- {{#callable:naive_trie::traverse}} -->
The `traverse` function searches for a child node in a trie structure corresponding to a given character and returns a pointer to it if found, or NULL if not.
- **Inputs**:
    - `c`: A character representing the key to search for in the children map of the current trie node.
- **Control Flow**:
    - The function attempts to find the character `c` in the `children` map of the current `naive_trie` node.
    - If the character `c` is found in the map, the function returns a pointer to the corresponding `naive_trie` child node.
    - If the character `c` is not found, the function returns `NULL`.
- **Output**: A pointer to the `naive_trie` child node corresponding to the character `c`, or `NULL` if no such child exists.
- **See also**: [`naive_trie`](#naive_trie)  (Data Structure)



---
### llm\_tokenizer<!-- {{#data_structure:llm_tokenizer}} -->
- **Type**: `struct`
- **Description**: The `llm_tokenizer` is a base struct for tokenizers, providing a default constructor and a virtual destructor. It serves as a foundational component for more specialized tokenizer implementations, allowing for polymorphic behavior when dealing with different types of tokenizers in a consistent manner.
- **Member Functions**:
    - [`llm_tokenizer::llm_tokenizer`](#llm_tokenizerllm_tokenizer)
    - [`llm_tokenizer::~llm_tokenizer`](#llm_tokenizerllm_tokenizer)

**Methods**

---
#### llm\_tokenizer::llm\_tokenizer<!-- {{#callable:llm_tokenizer::llm_tokenizer}} -->
The `llm_tokenizer` function is a constructor and destructor for the `llm_tokenizer` struct, which serves as a base class for various tokenizer implementations.
- **Inputs**: None
- **Control Flow**:
    - The constructor `llm_tokenizer()` is defined but does nothing, serving as a default constructor.
    - The destructor `~llm_tokenizer()` is virtual and set to default, allowing derived classes to override it if necessary.
- **Output**: There is no output from this function as it is a constructor and destructor for a struct.
- **See also**: [`llm_tokenizer`](#llm_tokenizer)  (Data Structure)


---
#### llm\_tokenizer::\~llm\_tokenizer<!-- {{#callable:llm_tokenizer::~llm_tokenizer}} -->
The destructor `~llm_tokenizer` is a virtual default destructor for the `llm_tokenizer` class.
- **Inputs**: None
- **Control Flow**:
    - The destructor is defined as virtual, allowing for proper cleanup in derived classes when a base class pointer is deleted.
    - The destructor is set to default, indicating that the compiler should generate the default destructor implementation.
- **Output**: There is no explicit output from this destructor; it ensures proper resource cleanup when an `llm_tokenizer` object is destroyed.
- **See also**: [`llm_tokenizer`](#llm_tokenizer)  (Data Structure)



---
### llm\_symbol<!-- {{#data_structure:llm_symbol}} -->
- **Type**: `struct`
- **Members**:
    - `prev`: An integer index pointing to the previous symbol in a sequence.
    - `next`: An integer index pointing to the next symbol in a sequence.
    - `text`: A constant character pointer to the text associated with the symbol.
    - `n`: A size_t value representing the length of the text.
- **Description**: The `llm_symbol` struct is a data structure used to represent a symbol in a linked list or sequence, where each symbol has a reference to its previous and next symbols, a pointer to its associated text, and the length of that text. This structure is likely used in the context of tokenization or text processing, where symbols are linked together to form a sequence or chain.


---
### llm\_bigram\_spm<!-- {{#data_structure:llm_bigram_spm}} -->
- **Type**: `struct`
- **Members**:
    - `left`: Represents the index of the left symbol in a bigram.
    - `right`: Represents the index of the right symbol in a bigram.
    - `score`: Stores the score associated with the bigram.
    - `size`: Indicates the size of the bigram.
- **Description**: The `llm_bigram_spm` struct is designed to represent a bigram in a sentence piece model (SPM) tokenizer. It includes indices for the left and right symbols of the bigram, a score that likely represents the frequency or importance of the bigram, and the size of the bigram. The struct also defines a comparator for ordering bigrams based on their scores and left symbol indices, and it uses a priority queue to manage these bigrams efficiently.


---
### comparator<!-- {{#data_structure:llm_bigram_bpe::comparator}} -->
- **Type**: `struct`
- **Description**: The `comparator` struct is a functor used to compare two `llm_bigram_bpe` objects. It defines an `operator()` that returns `true` if the first bigram has a higher rank than the second, or if they have the same rank, it compares the `left` index to determine the order. This struct is typically used in priority queues or sorting algorithms to maintain a specific order based on the rank and left index of bigrams.
- **Member Functions**:
    - [`llm_bigram_bpe::comparator::operator()`](#comparatoroperator())

**Methods**

---
#### comparator::operator\(\)<!-- {{#callable:llm_bigram_bpe::comparator::operator()}} -->
The `operator()` function in the `comparator` struct compares two `llm_bigram_bpe` objects based on their `rank` and `left` attributes.
- **Inputs**:
    - `l`: An `llm_bigram_bpe` object representing the left operand in the comparison.
    - `r`: An `llm_bigram_bpe` object representing the right operand in the comparison.
- **Control Flow**:
    - The function first compares the `rank` attribute of the two `llm_bigram_bpe` objects.
    - If the `rank` of `l` is greater than the `rank` of `r`, the function returns `true`.
    - If the `rank` of `l` is equal to the `rank` of `r`, it then compares the `left` attribute of the two objects.
    - If the `left` attribute of `l` is greater than the `left` attribute of `r`, the function returns `true`.
    - If neither condition is met, the function returns `false`.
- **Output**: A boolean value indicating whether the `llm_bigram_bpe` object `l` should be considered greater than `r` based on the specified comparison criteria.
- **See also**: [`llm_bigram_bpe::comparator`](#llm_bigram_bpecomparator)  (Data Structure)



---
### llm\_tokenizer\_spm<!-- {{#data_structure:llm_tokenizer_spm}} -->
- **Type**: `struct`
- **Description**: The `llm_tokenizer_spm` is a struct that inherits from the `llm_tokenizer` base struct. It represents a specific type of tokenizer, likely related to SentencePiece (SPM), used in the context of the LLaMA model. The constructor takes a `llama_vocab` object as a parameter, but does not utilize it within the constructor body, indicating that this struct may serve as a placeholder or a base for further specialization in the tokenization process.
- **Member Functions**:
    - [`llm_tokenizer_spm::llm_tokenizer_spm`](#llm_tokenizer_spmllm_tokenizer_spm)
- **Inherits From**:
    - [`llm_tokenizer::llm_tokenizer`](#llm_tokenizerllm_tokenizer)

**Methods**

---
#### llm\_tokenizer\_spm::llm\_tokenizer\_spm<!-- {{#callable:llm_tokenizer_spm::llm_tokenizer_spm}} -->
The `llm_tokenizer_spm` constructor initializes an instance of the `llm_tokenizer_spm` class, which is a derived class of `llm_tokenizer`, using a `llama_vocab` object.
- **Inputs**:
    - `vocab`: A reference to a `llama_vocab` object, which is presumably used for tokenization purposes, although it is not utilized in the constructor body.
- **Control Flow**:
    - The constructor is defined but does not perform any operations or initialize any member variables.
    - The `vocab` parameter is not used within the constructor body.
- **Output**: The constructor does not return any value as it is a constructor for the `llm_tokenizer_spm` class.
- **See also**: [`llm_tokenizer_spm`](#llm_tokenizer_spm)  (Data Structure)



---
### llm\_tokenizer\_spm\_session<!-- {{#data_structure:llm_tokenizer_spm_session}} -->
- **Type**: `struct`
- **Members**:
    - `vocab`: A reference to a llama_vocab object used for tokenization.
    - `symbols`: A vector of llm_symbol objects representing the symbols in the text being tokenized.
    - `work_queue`: A priority queue of llm_bigram_spm objects used to manage bigram substitutions during tokenization.
    - `rev_merge`: A map that tracks merged symbols and their original indices for resegmentation.
- **Description**: The `llm_tokenizer_spm_session` struct is designed to handle the tokenization of text using a SentencePiece Model (SPM) approach. It utilizes a vocabulary reference to convert text into tokens, managing the process through a series of symbols and bigram substitutions. The struct maintains a queue to prioritize and process bigram merges, and a map to track symbol merges for accurate resegmentation. This struct is integral to the tokenization process, ensuring that text is efficiently and accurately converted into a sequence of tokens.
- **Member Functions**:
    - [`llm_tokenizer_spm_session::llm_tokenizer_spm_session`](#llm_tokenizer_spm_sessionllm_tokenizer_spm_session)
    - [`llm_tokenizer_spm_session::tokenize`](#llm_tokenizer_spm_sessiontokenize)
    - [`llm_tokenizer_spm_session::resegment`](#llm_tokenizer_spm_sessionresegment)
    - [`llm_tokenizer_spm_session::try_add_bigram`](#llm_tokenizer_spm_sessiontry_add_bigram)

**Methods**

---
#### llm\_tokenizer\_spm\_session::llm\_tokenizer\_spm\_session<!-- {{#callable:llm_tokenizer_spm_session::llm_tokenizer_spm_session}} -->
The `llm_tokenizer_spm_session` constructor initializes a session for tokenizing text using a given vocabulary.
- **Inputs**:
    - `vocab`: A constant reference to a `llama_vocab` object, which provides the vocabulary used for tokenization.
- **Control Flow**:
    - The constructor initializes the `vocab` member variable with the provided `vocab` argument.
    - No additional logic or control flow is present in the constructor.
- **Output**: The constructor does not return any value; it initializes the object.
- **See also**: [`llm_tokenizer_spm_session`](#llm_tokenizer_spm_session)  (Data Structure)


---
#### llm\_tokenizer\_spm\_session::tokenize<!-- {{#callable:llm_tokenizer_spm_session::tokenize}} -->
The `tokenize` function processes a given text into tokens using a specific tokenization strategy, handling UTF-8 characters and merging frequent bigrams to optimize the tokenization.
- **Inputs**:
    - `text`: A constant reference to a `std::string` representing the input text to be tokenized.
    - `output`: A reference to a `std::vector<llama_token>` where the resulting tokens will be stored.
- **Control Flow**:
    - Initialize index and offset to track position in the text.
    - Iterate over the text, splitting it into UTF-8 characters and storing them as `llm_symbol` objects in a `symbols` vector.
    - Seed a work queue with all possible 2-character tokens (bigrams) from the `symbols` vector.
    - While the work queue is not empty, pop the highest frequency bigram and merge its symbols if they haven't been merged already.
    - Update the `symbols` vector by removing merged symbols and adjusting the chain of symbols.
    - Resegment the remaining symbols into tokens and store them in the `output` vector.
- **Output**: The function outputs the tokens into the provided `output` vector, representing the tokenized form of the input text.
- **Functions called**:
    - [`unicode_len_utf8`](unicode.cpp.driver.md#unicode_len_utf8)
    - [`llm_tokenizer_spm_session::try_add_bigram`](#llm_tokenizer_spm_sessiontry_add_bigram)
    - [`llm_tokenizer_spm_session::resegment`](#llm_tokenizer_spm_sessionresegment)
- **See also**: [`llm_tokenizer_spm_session`](#llm_tokenizer_spm_session)  (Data Structure)


---
#### llm\_tokenizer\_spm\_session::resegment<!-- {{#callable:llm_tokenizer_spm_session::resegment}} -->
The `resegment` function processes a given symbol to convert it into tokens and appends them to the output vector.
- **Inputs**:
    - `symbol`: A reference to an `llm_symbol` object representing a segment of text to be tokenized.
    - `output`: A reference to a vector of `llama_token` where the resulting tokens will be stored.
- **Control Flow**:
    - Convert the text from the `symbol` into a string and attempt to find a corresponding token using the vocabulary's `text_to_token` method.
    - If a valid token is found (not `LLAMA_TOKEN_NULL`), append it to the `output` vector and return.
    - If no valid token is found, check if the text can be split into two parts using the `rev_merge` map.
    - If the text cannot be split, convert each byte of the text into a token using `byte_to_token` and append them to the `output` vector.
    - If the text can be split, recursively call `resegment` on the two parts and append their results to the `output` vector.
- **Output**: The function does not return a value; it modifies the `output` vector by appending tokens.
- **See also**: [`llm_tokenizer_spm_session`](#llm_tokenizer_spm_session)  (Data Structure)


---
#### llm\_tokenizer\_spm\_session::try\_add\_bigram<!-- {{#callable:llm_tokenizer_spm_session::try_add_bigram}} -->
The `try_add_bigram` function attempts to create and add a bigram from two symbols to a work queue if certain conditions are met.
- **Inputs**:
    - `left`: An integer representing the index of the left symbol in the symbols vector.
    - `right`: An integer representing the index of the right symbol in the symbols vector.
- **Control Flow**:
    - Check if either 'left' or 'right' is -1; if so, return immediately.
    - Concatenate the text of the symbols at the 'left' and 'right' indices to form a bigram text.
    - Convert the bigram text to a token using the vocabulary's text_to_token method.
    - If the token is LLAMA_TOKEN_NULL, return immediately.
    - Check if the token is within the valid range of tokens in the vocabulary; if not, return.
    - Retrieve the token data for the token from the vocabulary.
    - Create a llm_bigram_spm object with the left and right indices, the token's score, and the size of the bigram text.
    - Push the bigram object onto the work queue.
    - Store the bigram text in the rev_merge map with a pair of the left and right indices.
- **Output**: The function does not return a value; it modifies the work queue and rev_merge map as side effects.
- **See also**: [`llm_tokenizer_spm_session`](#llm_tokenizer_spm_session)  (Data Structure)



---
### llama\_priority\_queue<!-- {{#data_structure:llama_priority_queue}} -->
- **Type**: `class`
- **Description**: The `llama_priority_queue` is a template class that extends the standard C++ `std::priority_queue` with an additional method `pop_move()`. This method allows for efficient removal and return of the top element by moving it, which can be more efficient than copying. The class template parameters include the type of elements `T`, the underlying container type `Container` (defaulting to `std::vector<T>`), and the comparison function `Compare` (defaulting to `std::less<typename Container::value_type>`). The standard `pop()` method is deleted to prevent its use, encouraging the use of `pop_move()` instead.
- **Member Functions**:
    - [`llama_priority_queue::pop_move`](#llama_priority_queuepop_move)
    - [`llama_priority_queue::pop`](#llama_priority_queuepop)
- **Inherits From**:
    - `std::priority_queue<T, Container, Compare>`

**Methods**

---
#### llama\_priority\_queue::pop\_move<!-- {{#callable:llama_priority_queue::pop_move}} -->
The `pop_move` function removes and returns the top element from a priority queue, ensuring the heap property is maintained.
- **Inputs**: None
- **Control Flow**:
    - Move the front element of the container `c` to a temporary variable `item`.
    - Call `std::pop_heap` to adjust the heap, maintaining the heap property after removing the top element.
    - Remove the last element from the container `c` using `pop_back`.
    - Return the moved element `item`.
- **Output**: The function returns the top element of the priority queue, which is of type `T`.
- **See also**: [`llama_priority_queue`](#llama_priority_queue)  (Data Structure)


---
#### llama\_priority\_queue::pop<!-- {{#callable:llama_priority_queue::pop}} -->
The `pop` function is a deleted member function of the `llama_priority_queue` class, preventing its use.
- **Inputs**: None
- **Control Flow**:
    - The function is marked as `= delete`, which means it is explicitly deleted and cannot be used.
    - This prevents the `pop` function from being called on instances of `llama_priority_queue`.
- **Output**: There is no output as the function is deleted and cannot be invoked.
- **See also**: [`llama_priority_queue`](#llama_priority_queue)  (Data Structure)



---
### llm\_bigram\_bpe<!-- {{#data_structure:llm_bigram_bpe}} -->
- **Type**: `struct`
- **Members**:
    - `left`: Represents the index of the left symbol in a bigram.
    - `right`: Represents the index of the right symbol in a bigram.
    - `text`: Stores the concatenated text of the bigram.
    - `rank`: Indicates the rank of the bigram, used for comparison.
    - `size`: Holds the size of the bigram text.
- **Description**: The `llm_bigram_bpe` struct is designed to represent a bigram in a Byte Pair Encoding (BPE) tokenizer. It includes indices for the left and right symbols of the bigram, the concatenated text of these symbols, and a rank used for prioritizing bigrams during processing. The struct also defines a comparator for sorting or prioritizing bigrams based on their rank and left symbol index. This struct is integral to the BPE tokenization process, where it helps manage and prioritize bigram merges.


---
### llm\_tokenizer\_bpe<!-- {{#data_structure:llm_tokenizer_bpe}} -->
- **Type**: `struct`
- **Members**:
    - `regex_exprs`: A vector of strings containing regular expressions used for tokenization.
- **Description**: The `llm_tokenizer_bpe` struct is a specialized tokenizer that extends the `llm_tokenizer` base class, designed to handle Byte Pair Encoding (BPE) tokenization. It initializes with a vocabulary of type `LLAMA_VOCAB_TYPE_BPE` and sets up a series of regular expressions (`regex_exprs`) based on the pre-tokenization type of the vocabulary. These regular expressions are used to preprocess text for BPE tokenization, allowing the tokenizer to handle various linguistic and formatting nuances across different languages and contexts.
- **Member Functions**:
    - [`llm_tokenizer_bpe::llm_tokenizer_bpe`](#llm_tokenizer_bpellm_tokenizer_bpe)
- **Inherits From**:
    - [`llm_tokenizer::llm_tokenizer`](#llm_tokenizerllm_tokenizer)

**Methods**

---
#### llm\_tokenizer\_bpe::llm\_tokenizer\_bpe<!-- {{#callable:llm_tokenizer_bpe::llm_tokenizer_bpe}} -->
The `llm_tokenizer_bpe` function initializes a BPE tokenizer by setting up regular expressions based on the pre-tokenizer type of the given vocabulary.
- **Inputs**:
    - `vocab`: An instance of `llama_vocab` which contains the vocabulary information, including its type and pre-tokenizer type.
- **Control Flow**:
    - Assert that the vocabulary type is BPE using `GGML_ASSERT`.
    - Use a switch statement to determine the pre-tokenizer type of the vocabulary.
    - For each case in the switch statement, assign a specific set of regular expressions to `regex_exprs` based on the pre-tokenizer type.
    - If the pre-tokenizer type does not match any specific case, use a default set of regular expressions for BPE tokenization.
- **Output**: The function does not return a value; it initializes the `regex_exprs` member variable of the `llm_tokenizer_bpe` class.
- **See also**: [`llm_tokenizer_bpe`](#llm_tokenizer_bpe)  (Data Structure)



---
### llm\_tokenizer\_bpe\_session<!-- {{#data_structure:llm_tokenizer_bpe_session}} -->
- **Type**: `struct`
- **Members**:
    - `vocab`: A reference to a `llama_vocab` object, representing the vocabulary used in the tokenization process.
    - `tokenizer`: A reference to a `llm_tokenizer_bpe` object, representing the tokenizer used for byte pair encoding.
    - `symbols`: A vector of `llm_symbol` objects, used to store symbols during the tokenization process.
    - `symbols_final`: A vector of `llm_symbol` objects, used to store the final symbols after processing.
    - `work_queue`: A priority queue of `llm_bigram_bpe` objects, used to manage bigrams during tokenization.
- **Description**: The `llm_tokenizer_bpe_session` struct is designed to handle a session of byte pair encoding (BPE) tokenization using a given vocabulary and tokenizer. It manages the tokenization process by maintaining a list of symbols and a work queue for bigrams, allowing for efficient merging and token generation. The struct provides methods to append special tokens like BOS (beginning of sequence) and EOS (end of sequence), check for duplicate BOS/EOS tokens, and perform the actual tokenization of input text into a sequence of tokens.
- **Member Functions**:
    - [`llm_tokenizer_bpe_session::llm_tokenizer_bpe_session`](#llm_tokenizer_bpe_sessionllm_tokenizer_bpe_session)
    - [`llm_tokenizer_bpe_session::append`](#llm_tokenizer_bpe_sessionappend)
    - [`llm_tokenizer_bpe_session::append_bos`](#llm_tokenizer_bpe_sessionappend_bos)
    - [`llm_tokenizer_bpe_session::append_eos`](#llm_tokenizer_bpe_sessionappend_eos)
    - [`llm_tokenizer_bpe_session::check_double_bos_eos`](#llm_tokenizer_bpe_sessioncheck_double_bos_eos)
    - [`llm_tokenizer_bpe_session::tokenize`](#llm_tokenizer_bpe_sessiontokenize)
    - [`llm_tokenizer_bpe_session::add_new_bigram`](#llm_tokenizer_bpe_sessionadd_new_bigram)

**Methods**

---
#### llm\_tokenizer\_bpe\_session::llm\_tokenizer\_bpe\_session<!-- {{#callable:llm_tokenizer_bpe_session::llm_tokenizer_bpe_session}} -->
The `llm_tokenizer_bpe_session` constructor initializes a session for BPE tokenization using a given vocabulary and tokenizer.
- **Inputs**:
    - `vocab`: A reference to a `llama_vocab` object that contains the vocabulary information needed for tokenization.
    - `tokenizer`: A reference to a `llm_tokenizer_bpe` object that provides the BPE tokenization logic and regex expressions.
- **Control Flow**:
    - The constructor initializes the `vocab` and `tokenizer` member variables with the provided arguments.
    - No additional logic or control flow is present in this constructor.
- **Output**: This constructor does not return any value; it initializes the object state.
- **See also**: [`llm_tokenizer_bpe_session`](#llm_tokenizer_bpe_session)  (Data Structure)


---
#### llm\_tokenizer\_bpe\_session::append<!-- {{#callable:llm_tokenizer_bpe_session::append}} -->
The `append` function adds a given `llama_token` to the end of a provided vector of `llama_token` objects.
- **Inputs**:
    - `token_id`: A `llama_token` representing the token to be appended to the vector.
    - `output`: A reference to a `std::vector` of `llama_token` where the `token_id` will be appended.
- **Control Flow**:
    - The function takes a `llama_token` and a reference to a vector of `llama_token` as inputs.
    - It uses the `push_back` method of the vector to append the `token_id` to the end of the vector.
- **Output**: The function does not return any value; it modifies the input vector by adding the token to it.
- **See also**: [`llm_tokenizer_bpe_session`](#llm_tokenizer_bpe_session)  (Data Structure)


---
#### llm\_tokenizer\_bpe\_session::append\_bos<!-- {{#callable:llm_tokenizer_bpe_session::append_bos}} -->
The `append_bos` function appends a Beginning of Sequence (BOS) token to a given vector of tokens if the vocabulary settings allow it.
- **Inputs**:
    - `output`: A reference to a vector of `llama_token` where the BOS token will be appended if applicable.
- **Control Flow**:
    - Check if the vocabulary is configured to add a BOS token using `vocab.get_add_bos()`.
    - If true, assert that the BOS token is not null using `GGML_ASSERT(vocab.token_bos() != LLAMA_TOKEN_NULL)`.
    - Append the BOS token to the `output` vector using `output.push_back(vocab.token_bos())`.
    - Return `true` if the BOS token was appended, otherwise return `false`.
- **Output**: Returns a boolean value: `true` if the BOS token was appended to the output vector, `false` otherwise.
- **See also**: [`llm_tokenizer_bpe_session`](#llm_tokenizer_bpe_session)  (Data Structure)


---
#### llm\_tokenizer\_bpe\_session::append\_eos<!-- {{#callable:llm_tokenizer_bpe_session::append_eos}} -->
The `append_eos` function appends an end-of-sequence (EOS) token to a given vector of tokens if the vocabulary settings allow it.
- **Inputs**:
    - `output`: A reference to a vector of `llama_token` where the EOS token will be appended if applicable.
- **Control Flow**:
    - Check if the vocabulary is configured to add an EOS token using `vocab.get_add_eos()`.
    - Assert that the EOS token is not null using `GGML_ASSERT(vocab.token_eos() != LLAMA_TOKEN_NULL)`.
    - If the EOS token is valid, append it to the `output` vector using `output.push_back(vocab.token_eos())`.
    - Return `true` if the EOS token was appended, otherwise return `false`.
- **Output**: Returns a boolean value: `true` if the EOS token was appended to the output vector, `false` otherwise.
- **See also**: [`llm_tokenizer_bpe_session`](#llm_tokenizer_bpe_session)  (Data Structure)


---
#### llm\_tokenizer\_bpe\_session::check\_double\_bos\_eos<!-- {{#callable:llm_tokenizer_bpe_session::check_double_bos_eos}} -->
The `check_double_bos_eos` function checks for and logs warnings if there are duplicate beginning-of-sequence (BOS) or end-of-sequence (EOS) tokens in a given output vector of tokens.
- **Inputs**:
    - `output`: A constant reference to a vector of `llama_token` objects representing the tokenized output to be checked for duplicate BOS or EOS tokens.
- **Control Flow**:
    - Check if the vocabulary is set to add a BOS token and if the output vector has at least two tokens, then verify if the second token is a BOS token.
    - If the above condition is true, log a warning indicating that the prompt starts with two BOS tokens.
    - Check if the vocabulary is set to add an EOS token and if the output vector has at least two tokens, then verify if the second-to-last token is an EOS token.
    - If the above condition is true, log a warning indicating that the prompt ends with two EOS tokens.
- **Output**: This function does not return any value; it performs logging operations if conditions are met.
- **See also**: [`llm_tokenizer_bpe_session`](#llm_tokenizer_bpe_session)  (Data Structure)


---
#### llm\_tokenizer\_bpe\_session::tokenize<!-- {{#callable:llm_tokenizer_bpe_session::tokenize}} -->
The `tokenize` function processes a given text into a sequence of tokens using a byte pair encoding (BPE) approach, considering specific vocabulary rules and merging strategies.
- **Inputs**:
    - `text`: A constant reference to a `std::string` representing the input text to be tokenized.
    - `output`: A reference to a `std::vector<llama_token>` where the resulting tokens will be stored.
- **Control Flow**:
    - Initialize `final_prev_index` to -1 and split the input text into words using [`unicode_regex_split`](unicode.cpp.driver.md#unicode_regex_split) with the tokenizer's regex expressions.
    - Clear the `symbols_final` vector to prepare for storing final symbols.
    - Iterate over each word in the `word_collection`.
    - For each word, initialize a new `work_queue` and clear the `symbols` vector.
    - Set `index` to 0 and `offset` to 0 to track character positions within the word.
    - Check if merges should be ignored and if the word is already a known token; if so, add it directly to `symbols` and update `offset`.
    - While `offset` is less than the word size, create `llm_symbol` objects for each character or character sequence, updating `offset`, `prev`, and `next` indices accordingly, and add them to `symbols`.
    - For each pair of symbols, add a new bigram to the `work_queue` using [`add_new_bigram`](#llm_tokenizer_bpe_sessionadd_new_bigram).
    - While the `work_queue` is not empty, process each bigram by merging symbols if they match the bigram text, updating `prev` and `next` indices, and adding new bigrams for adjacent symbols.
    - After processing bigrams, add non-zero length symbols to `symbols_final`, maintaining correct order with `prev` and `next` indices.
    - Replace `symbols` with `symbols_final` to prepare for token conversion.
    - If `symbols` is not empty, iterate through them, converting each symbol to a token using `vocab.text_to_token`.
    - If a symbol cannot be directly converted to a token, convert each byte of the symbol to a token and add it to `output`.
- **Output**: The function outputs a sequence of tokens stored in the `output` vector, representing the tokenized form of the input text.
- **Functions called**:
    - [`unicode_regex_split`](unicode.cpp.driver.md#unicode_regex_split)
    - [`unicode_len_utf8`](unicode.cpp.driver.md#unicode_len_utf8)
    - [`llm_tokenizer_bpe_session::add_new_bigram`](#llm_tokenizer_bpe_sessionadd_new_bigram)
- **See also**: [`llm_tokenizer_bpe_session`](#llm_tokenizer_bpe_session)  (Data Structure)


---
#### llm\_tokenizer\_bpe\_session::add\_new\_bigram<!-- {{#callable:llm_tokenizer_bpe_session::add_new_bigram}} -->
The `add_new_bigram` function attempts to create and enqueue a new bigram from two given indices if they are valid and the bigram has a valid rank.
- **Inputs**:
    - `left`: An integer representing the index of the left symbol in the symbols vector.
    - `right`: An integer representing the index of the right symbol in the symbols vector.
- **Control Flow**:
    - Check if either 'left' or 'right' is -1; if so, return immediately.
    - Extract the text and size of the left and right symbols using their indices.
    - Find the BPE rank of the combined left and right tokens using the vocabulary's `find_bpe_rank` method.
    - If the rank is less than 0, return immediately.
    - Create a new `llm_bigram_bpe` object and set its properties using the left and right indices, combined text, size, and rank.
    - Push the new bigram onto the work queue.
- **Output**: The function does not return a value; it modifies the work queue by potentially adding a new bigram.
- **See also**: [`llm_tokenizer_bpe_session`](#llm_tokenizer_bpe_session)  (Data Structure)



---
### llm\_tokenizer\_wpm<!-- {{#data_structure:llm_tokenizer_wpm}} -->
- **Type**: `struct`
- **Description**: The `llm_tokenizer_wpm` is a struct that inherits from the `llm_tokenizer` base class. It is a specialized tokenizer for word-piece models, but in this definition, it does not have any additional members or fields beyond its constructor, which takes a `llama_vocab` reference as a parameter. This struct is part of a larger framework for handling different types of tokenizers, but in its current form, it serves as a placeholder or a base for further extensions specific to word-piece tokenization.
- **Member Functions**:
    - [`llm_tokenizer_wpm::llm_tokenizer_wpm`](#llm_tokenizer_wpmllm_tokenizer_wpm)
- **Inherits From**:
    - [`llm_tokenizer::llm_tokenizer`](#llm_tokenizerllm_tokenizer)

**Methods**

---
#### llm\_tokenizer\_wpm::llm\_tokenizer\_wpm<!-- {{#callable:llm_tokenizer_wpm::llm_tokenizer_wpm}} -->
The `llm_tokenizer_wpm` constructor initializes a WPM tokenizer object with a given vocabulary.
- **Inputs**:
    - `vocab`: A reference to a `llama_vocab` object, which contains the vocabulary data needed for the tokenizer.
- **Control Flow**:
    - The constructor takes a `llama_vocab` reference as an argument.
    - It initializes the `llm_tokenizer_wpm` object without performing any additional operations.
- **Output**: An instance of the `llm_tokenizer_wpm` class is created.
- **See also**: [`llm_tokenizer_wpm`](#llm_tokenizer_wpm)  (Data Structure)



---
### llm\_tokenizer\_wpm\_session<!-- {{#data_structure:llm_tokenizer_wpm_session}} -->
- **Type**: `struct`
- **Members**:
    - `vocab`: A reference to a `llama_vocab` object used for tokenization.
- **Description**: The `llm_tokenizer_wpm_session` struct is designed to handle the tokenization process using a WordPiece Model (WPM) approach. It utilizes a reference to a `llama_vocab` object to convert input text into a sequence of tokens. The struct provides a `tokenize` method that processes the input text by normalizing it, splitting it into words, and then finding the longest matching tokens from the vocabulary for each word. If no match is found, it uses an unknown token. The struct also includes a static `preprocess` method to normalize and split the text into words, and a static `is_chinese_char` method to identify Chinese characters.
- **Member Functions**:
    - [`llm_tokenizer_wpm_session::llm_tokenizer_wpm_session`](#llm_tokenizer_wpm_sessionllm_tokenizer_wpm_session)
    - [`llm_tokenizer_wpm_session::tokenize`](#llm_tokenizer_wpm_sessiontokenize)
    - [`llm_tokenizer_wpm_session::preprocess`](#llm_tokenizer_wpm_sessionpreprocess)
    - [`llm_tokenizer_wpm_session::is_chinese_char`](#llm_tokenizer_wpm_sessionis_chinese_char)

**Methods**

---
#### llm\_tokenizer\_wpm\_session::llm\_tokenizer\_wpm\_session<!-- {{#callable:llm_tokenizer_wpm_session::llm_tokenizer_wpm_session}} -->
The `llm_tokenizer_wpm_session` constructor initializes a session for tokenizing text using a given vocabulary.
- **Inputs**:
    - `vocab`: A reference to a `llama_vocab` object that provides the vocabulary for tokenization.
- **Control Flow**:
    - The constructor takes a constant reference to a `llama_vocab` object as its parameter.
    - It initializes the `vocab` member variable with the provided `llama_vocab` reference.
- **Output**: An instance of `llm_tokenizer_wpm_session` is created, ready to tokenize text using the specified vocabulary.
- **See also**: [`llm_tokenizer_wpm_session`](#llm_tokenizer_wpm_session)  (Data Structure)


---
#### llm\_tokenizer\_wpm\_session::tokenize<!-- {{#callable:llm_tokenizer_wpm_session::tokenize}} -->
The `tokenize` function processes a given text into a sequence of tokens using a vocabulary, handling normalization, whitespace splitting, and token matching.
- **Inputs**:
    - `text`: A constant reference to a `std::string` representing the input text to be tokenized.
    - `output`: A reference to a `std::vector<llama_token>` where the resulting tokens will be stored.
- **Control Flow**:
    - The function begins by normalizing the input text and splitting it into words using the [`preprocess`](#llm_tokenizer_wpm_sessionpreprocess) function.
    - It iterates over each word in the resulting list of words.
    - For each word, it checks if the word is empty and skips it if so.
    - A phantom space is prepended to the word, and the function attempts to match the longest possible token from the vocabulary starting from each character position in the word.
    - If a match is found, the token is added to the output vector, and the search continues from the end of the matched token.
    - If no match is found for a character position, the function discards all tokens found for the current word and breaks out of the loop.
    - If no tokens were added for a word, an unknown token is added to the output vector.
- **Output**: The function outputs a sequence of tokens in the `output` vector, representing the tokenized form of the input text.
- **Functions called**:
    - [`llm_tokenizer_wpm_session::preprocess`](#llm_tokenizer_wpm_sessionpreprocess)
- **See also**: [`llm_tokenizer_wpm_session`](#llm_tokenizer_wpm_session)  (Data Structure)


---
#### llm\_tokenizer\_wpm\_session::preprocess<!-- {{#callable:llm_tokenizer_wpm_session::preprocess}} -->
The `preprocess` function normalizes a given UTF-8 text string into a vector of lowercase words, handling whitespace, punctuation, and special characters.
- **Inputs**:
    - `text`: A constant reference to a UTF-8 encoded string that needs to be preprocessed.
- **Control Flow**:
    - Convert the input UTF-8 text into a vector of Unicode code points and normalize them using NFD (Normalization Form D).
    - Initialize a vector of strings `words` with a single empty string to store the processed words.
    - Iterate over each Unicode code point in the normalized vector.
    - For each code point, retrieve its flags to determine its type (e.g., whitespace, control, punctuation).
    - If the code point is whitespace and the last word in `words` is not empty, add a new empty string to `words` to start a new word.
    - Skip code points that are separators, control characters, or invalid (0 or 0xFFFD).
    - Convert the code point to lowercase and then to a UTF-8 string.
    - If the code point is punctuation, a symbol (for ASCII), or a Chinese character, treat it as a single-character word and start a new word in `words`.
    - Otherwise, append the UTF-8 string to the last word in `words`.
    - After processing all code points, remove the last word if it is empty.
    - Return the vector `words` containing the processed words.
- **Output**: A vector of strings, where each string is a processed word from the input text.
- **Functions called**:
    - [`unicode_cpts_normalize_nfd`](unicode.cpp.driver.md#unicode_cpts_normalize_nfd)
    - [`unicode_cpts_from_utf8`](unicode.cpp.driver.md#unicode_cpts_from_utf8)
    - [`unicode_cpt_flags_from_cpt`](unicode.cpp.driver.md#unicode_cpt_flags_from_cpt)
    - [`unicode_cpt_to_utf8`](unicode.cpp.driver.md#unicode_cpt_to_utf8)
    - [`unicode_tolower`](unicode.cpp.driver.md#unicode_tolower)
    - [`llm_tokenizer_wpm_session::is_chinese_char`](#llm_tokenizer_wpm_sessionis_chinese_char)
- **See also**: [`llm_tokenizer_wpm_session`](#llm_tokenizer_wpm_session)  (Data Structure)


---
#### llm\_tokenizer\_wpm\_session::is\_chinese\_char<!-- {{#callable:llm_tokenizer_wpm_session::is_chinese_char}} -->
The `is_chinese_char` function checks if a given Unicode code point corresponds to a Chinese character.
- **Inputs**:
    - `cpt`: A 32-bit unsigned integer representing a Unicode code point.
- **Control Flow**:
    - The function evaluates a series of conditional checks to determine if the input code point falls within any of the specified Unicode ranges for Chinese characters.
    - It returns true if the code point is within any of these ranges, indicating it is a Chinese character.
    - If the code point does not fall within any of these ranges, the function returns false.
- **Output**: A boolean value indicating whether the input code point is a Chinese character.
- **See also**: [`llm_tokenizer_wpm_session`](#llm_tokenizer_wpm_session)  (Data Structure)



---
### llm\_tokenizer\_ugm<!-- {{#data_structure:llm_tokenizer_ugm}} -->
- **Type**: `struct`
- **Members**:
    - `escaped_space`: A constant string representing the escaped space symbol, U+2581.
    - `prefix_replacements`: A pointer to a character array containing null-terminated replacement strings for prefixes matched by the XCDA.
    - `prefix_replacements_size`: The size of the prefix_replacements array.
    - `xcda_array`: A pointer to a uint32_t array containing entries of XOR-compressed compact double array (XCDA).
    - `xcda_array_size`: The size of the xcda_array.
    - `user_defined_token_matcher`: A naive_trie structure for matching user-defined tokens.
    - `min_score`: A float representing the minimum score of normal tokens.
    - `max_score`: A float representing the maximum score of normal tokens.
    - `unknown_token_score_penalty`: A float representing the penalty applied to the score of unknown tokens.
    - `unknown_token_score`: A float representing the score for unknown tokens.
    - `token_matcher`: A naive_trie structure for matching tokens.
- **Description**: The `llm_tokenizer_ugm` struct is a specialized tokenizer that extends the `llm_tokenizer` base class. It is designed to handle tokenization using a precompiled character map and a vocabulary. The struct manages various components such as a compact double array for efficient prefix matching, a trie for user-defined tokens, and score management for tokens. It initializes these components based on the provided vocabulary and precompiled character map, and it calculates scores for tokens, including penalties for unknown tokens.
- **Member Functions**:
    - [`llm_tokenizer_ugm::llm_tokenizer_ugm`](#llm_tokenizer_ugmllm_tokenizer_ugm)
- **Inherits From**:
    - [`llm_tokenizer::llm_tokenizer`](#llm_tokenizerllm_tokenizer)

**Methods**

---
#### llm\_tokenizer\_ugm::llm\_tokenizer\_ugm<!-- {{#callable:llm_tokenizer_ugm::llm_tokenizer_ugm}} -->
The `llm_tokenizer_ugm` function initializes a tokenizer for a unigram language model using a given vocabulary and a precompiled character map.
- **Inputs**:
    - `vocab`: A reference to a `llama_vocab` object that provides token data and methods to check token types.
    - `precompiled_charsmap`: A vector of characters representing a precompiled character map used for token replacement and compression.
- **Control Flow**:
    - Check if `precompiled_charsmap` is not empty.
    - Extract the size of the XOR-compressed compact double array (XCDA) from the first four bytes of `precompiled_charsmap`.
    - Verify that the XCDA blob size does not exceed the bounds of `precompiled_charsmap`.
    - Set `xcda_array` to point to the XCDA entries and calculate its size.
    - Set `prefix_replacements` to point to the remaining bytes of `precompiled_charsmap`.
    - Iterate over each token in the vocabulary using its ID.
    - For normal tokens, update `min_score` and `max_score` based on the token's score.
    - Insert normal, user-defined, and unused tokens into `token_matcher`.
    - Insert user-defined tokens into `user_defined_token_matcher`.
    - Calculate `unknown_token_score` by subtracting `unknown_token_score_penalty` from `min_score`.
- **Output**: The function does not return a value; it initializes the tokenizer's internal state.
- **See also**: [`llm_tokenizer_ugm`](#llm_tokenizer_ugm)  (Data Structure)



---
### llm\_tokenizer\_ugm\_session<!-- {{#data_structure:llm_tokenizer_ugm_session}} -->
- **Type**: `struct`
- **Members**:
    - `vocab`: A reference to a llama_vocab object, representing the vocabulary used for tokenization.
    - `tokenizer`: A reference to a llm_tokenizer_ugm object, representing the tokenizer used for processing text.
- **Description**: The `llm_tokenizer_ugm_session` struct is designed to handle the tokenization of text using a unigram language model, specifically optimized with a Viterbi algorithm-based approach. It utilizes a given vocabulary and tokenizer to process input text, normalizing it and finding the best tokenization by traversing a token trie. The struct is capable of handling unknown tokens by applying a score penalty and backtracking to determine the optimal sequence of tokens. This struct is part of a larger system for text processing and tokenization in natural language processing applications.
- **Member Functions**:
    - [`llm_tokenizer_ugm_session::llm_tokenizer_ugm_session`](#llm_tokenizer_ugm_sessionllm_tokenizer_ugm_session)
    - [`llm_tokenizer_ugm_session::tokenize`](#llm_tokenizer_ugm_sessiontokenize)
    - [`llm_tokenizer_ugm_session::normalize`](#llm_tokenizer_ugm_sessionnormalize)
    - [`llm_tokenizer_ugm_session::normalize_prefix`](#llm_tokenizer_ugm_sessionnormalize_prefix)

**Methods**

---
#### llm\_tokenizer\_ugm\_session::llm\_tokenizer\_ugm\_session<!-- {{#callable:llm_tokenizer_ugm_session::llm_tokenizer_ugm_session}} -->
The `llm_tokenizer_ugm_session` constructor initializes a session for tokenizing text using a unigram language model with a given vocabulary and tokenizer.
- **Inputs**:
    - `vocab`: A reference to a `llama_vocab` object that provides the vocabulary for tokenization.
    - `tokenizer`: A reference to a `llm_tokenizer_ugm` object that provides the tokenizer configuration and data for the session.
- **Control Flow**:
    - The constructor initializes the `vocab` and `tokenizer` member variables with the provided arguments.
    - No additional logic or operations are performed in the constructor.
- **Output**: The constructor does not return any value; it initializes the session object with the provided vocabulary and tokenizer.
- **See also**: [`llm_tokenizer_ugm_session`](#llm_tokenizer_ugm_session)  (Data Structure)


---
#### llm\_tokenizer\_ugm\_session::tokenize<!-- {{#callable:llm_tokenizer_ugm_session::tokenize}} -->
The `tokenize` function processes a given text into a sequence of tokens using a Viterbi-like algorithm for unigram language models, storing the best tokenization results in the output vector.
- **Inputs**:
    - `text`: A constant reference to a `std::string` representing the input text to be tokenized.
    - `output`: A reference to a `std::vector<llama_token>` where the resulting tokens will be stored.
- **Control Flow**:
    - Initialize `output_size` to the current size of the `output` vector for later reversal.
    - Normalize the input `text` and store it in `normalized`; if the normalized text is empty, return immediately.
    - Initialize a vector `tokenization_results` to store the best tokenization results for each position in the input, with initial scores set to `-DBL_MAX` except for the start position, which is set to zero.
    - Iterate over each UTF-8 code point in the normalized text, using a trie to find matching tokens and calculate scores for each potential tokenization.
    - If a valid token is found that matches the entire UTF-8 code point, update the best tokenization result for the current position if the new score is higher.
    - If no valid token is found for a UTF-8 code point, use an unknown token with a penalty score.
    - Backtrack from the end of the input to gather the best tokenization results, merging consecutive unknown tokens into a single token.
    - Reverse the `output` vector from the position `output_size` to the end to correct the order of tokens.
- **Output**: The function does not return a value but modifies the `output` vector to contain the sequence of tokens representing the best tokenization of the input text.
- **Functions called**:
    - [`llm_tokenizer_ugm_session::normalize`](#llm_tokenizer_ugm_sessionnormalize)
    - [`unicode_len_utf8`](unicode.cpp.driver.md#unicode_len_utf8)
- **See also**: [`llm_tokenizer_ugm_session`](#llm_tokenizer_ugm_session)  (Data Structure)


---
#### llm\_tokenizer\_ugm\_session::normalize<!-- {{#callable:llm_tokenizer_ugm_session::normalize}} -->
The `normalize` function processes an input string to produce a normalized version by handling whitespace and character sequences according to specific rules defined by the vocabulary and tokenizer settings.
- **Inputs**:
    - `input`: A constant reference to a `std::string` representing the input text to be normalized.
    - `normalized`: A pointer to a `std::string` where the normalized output will be stored.
- **Control Flow**:
    - Clear the `normalized` string and reserve space for it based on the input size.
    - Determine the space character to use based on the vocabulary's escape whitespace setting.
    - Set flags for whether to prepend or append spaces and whether to merge spaces based on vocabulary settings.
    - Initialize flags for tracking space prepending and non-whitespace processing.
    - Iterate over the input string, processing each character or sequence of characters using [`normalize_prefix`](#llm_tokenizer_ugm_sessionnormalize_prefix).
    - For each character in the normalized result, decide whether to prepend a space, append the character, or handle spaces based on the flags.
    - Update the input offset by the number of characters consumed in the current iteration.
    - If the append space flag is set, append a space to the normalized string at the end.
- **Output**: The function outputs a normalized version of the input string in the `normalized` parameter, with spaces and character sequences adjusted according to the specified rules.
- **Functions called**:
    - [`llm_tokenizer_ugm_session::normalize_prefix`](#llm_tokenizer_ugm_sessionnormalize_prefix)
- **See also**: [`llm_tokenizer_ugm_session`](#llm_tokenizer_ugm_session)  (Data Structure)


---
#### llm\_tokenizer\_ugm\_session::normalize\_prefix<!-- {{#callable:llm_tokenizer_ugm_session::normalize_prefix}} -->
The `normalize_prefix` function normalizes a given input string prefix by matching it against user-defined tokens, an XOR-compressed compact double array (XCDA), or validating it as a UTF-8 sequence, and returns the normalized result.
- **Inputs**:
    - `input`: A constant reference to a `std::string` representing the input string to be normalized.
    - `input_offset`: A `size_t` representing the starting position in the input string from which normalization should begin.
- **Control Flow**:
    - Check if `input_offset` is equal to the size of the input string; if so, return a normalization result with zero length and consumption.
    - Attempt to match the input prefix with a user-defined token using the `user_defined_token_matcher`; if a match is found, return the matched token as the normalization result.
    - Initialize variables to track the longest prefix length and offset.
    - If the XCDA array size is greater than zero, create an `xcda_array_view` and traverse the XCDA to find the longest matching normalized sequence, updating the longest prefix length and offset if a leaf node is reached.
    - If a longest prefix is found, verify the offset is within bounds and return the corresponding replacement sequence.
    - Attempt to validate the input prefix as a UTF-8 sequence; if valid, return the sequence unmodified.
    - If the UTF-8 validation fails, return a replacement character (U+FFFD) for the first byte.
- **Output**: A `normalization_result` struct containing a pointer to the normalized string, the length of the normalized string, and the number of input characters consumed.
- **Functions called**:
    - [`unicode_cpt_from_utf8`](unicode.cpp.driver.md#unicode_cpt_from_utf8)
- **See also**: [`llm_tokenizer_ugm_session`](#llm_tokenizer_ugm_session)  (Data Structure)



---
### normalization\_result<!-- {{#data_structure:llm_tokenizer_ugm_session::normalization_result}} -->
- **Type**: `struct`
- **Members**:
    - `normalized`: A pointer to a constant character array representing the normalized string.
    - `normalized_len`: A size_t variable indicating the length of the normalized string.
    - `consumed_input`: A size_t variable representing the amount of input consumed during normalization.
- **Description**: The `normalization_result` struct is designed to encapsulate the result of a normalization process. It contains a pointer to the normalized string, the length of this string, and the amount of input that was consumed during the normalization. This struct is useful for functions that need to return multiple pieces of information about the normalization process, such as the resulting string and metadata about the operation.


---
### xcda\_array\_view<!-- {{#data_structure:llm_tokenizer_ugm_session::xcda_array_view}} -->
- **Type**: `struct`
- **Members**:
    - `xcda_array`: A pointer to a constant array of 32-bit unsigned integers representing the XCDA data.
    - `xcda_array_size`: The size of the XCDA array, indicating the number of elements it contains.
- **Description**: The `xcda_array_view` struct provides a view into an XOR-compressed compact double array (XCDA), which is a data structure used for efficient string dictionary storage. It allows access to the base, lcheck, leaf, and value components of the packed nodes within the XCDA array. The struct includes methods to retrieve these components by unpacking the bit-packed entries, which are stored as 32-bit integers. This struct is designed to facilitate operations on the XCDA without modifying the underlying data.
- **Member Functions**:
    - [`llm_tokenizer_ugm_session::xcda_array_view::xcda_array_view`](#xcda_array_viewxcda_array_view)
    - [`llm_tokenizer_ugm_session::xcda_array_view::get_base`](#xcda_array_viewget_base)
    - [`llm_tokenizer_ugm_session::xcda_array_view::get_lcheck`](#xcda_array_viewget_lcheck)
    - [`llm_tokenizer_ugm_session::xcda_array_view::get_leaf`](#xcda_array_viewget_leaf)
    - [`llm_tokenizer_ugm_session::xcda_array_view::get_value`](#xcda_array_viewget_value)
    - [`llm_tokenizer_ugm_session::xcda_array_view::get_node`](#xcda_array_viewget_node)

**Methods**

---
#### xcda\_array\_view::xcda\_array\_view<!-- {{#callable:llm_tokenizer_ugm_session::xcda_array_view::xcda_array_view}} -->
The `xcda_array_view` constructor initializes an instance of the `xcda_array_view` struct with a given array and its size.
- **Inputs**:
    - `xcda_array`: A pointer to a constant array of 32-bit unsigned integers representing the XCDA array.
    - `xcda_array_size`: The size of the XCDA array, indicating the number of elements in the array.
- **Control Flow**:
    - The constructor takes two parameters: a pointer to a constant array of 32-bit unsigned integers and its size.
    - It initializes the `xcda_array` and `xcda_array_size` member variables with the provided arguments.
- **Output**: An instance of the `xcda_array_view` struct is created and initialized with the provided array and its size.
- **See also**: [`llm_tokenizer_ugm_session::xcda_array_view`](#llm_tokenizer_ugm_session::xcda_array_view)  (Data Structure)


---
#### xcda\_array\_view::get\_base<!-- {{#callable:llm_tokenizer_ugm_session::xcda_array_view::get_base}} -->
The `get_base` function extracts and returns the BASE value from a packed node at a specified index in an XCDA array.
- **Inputs**:
    - `index`: The index of the node in the XCDA array from which the BASE value is to be extracted.
- **Control Flow**:
    - Retrieve the packed node value from the XCDA array using the [`get_node`](#xcda_array_viewget_node) method with the given index.
    - Shift the packed node value right by 10 bits to isolate the BASE value.
    - Check if the 9th bit of the packed node is set, and if so, shift the BASE value left by 1 bit.
    - Return the calculated BASE value.
- **Output**: The function returns a `uint32_t` representing the BASE value extracted from the packed node.
- **Functions called**:
    - [`llm_tokenizer_ugm_session::xcda_array_view::get_node`](#xcda_array_viewget_node)
- **See also**: [`llm_tokenizer_ugm_session::xcda_array_view`](#llm_tokenizer_ugm_session::xcda_array_view)  (Data Structure)


---
#### xcda\_array\_view::get\_lcheck<!-- {{#callable:llm_tokenizer_ugm_session::xcda_array_view::get_lcheck}} -->
The `get_lcheck` function retrieves the `lcheck` value from a packed node at a specified index in an `xcda_array_view` structure.
- **Inputs**:
    - `index`: The index of the node in the `xcda_array` from which to retrieve the `lcheck` value.
- **Control Flow**:
    - Retrieve the packed node at the specified index using the [`get_node`](#xcda_array_viewget_node) method.
    - Perform a bitwise AND operation on the packed node with the mask `((1U << 31) | 0xff)` to extract the `lcheck` value.
    - Return the extracted `lcheck` value.
- **Output**: A `uint32_t` value representing the `lcheck` value extracted from the packed node.
- **Functions called**:
    - [`llm_tokenizer_ugm_session::xcda_array_view::get_node`](#xcda_array_viewget_node)
- **See also**: [`llm_tokenizer_ugm_session::xcda_array_view`](#llm_tokenizer_ugm_session::xcda_array_view)  (Data Structure)


---
#### xcda\_array\_view::get\_leaf<!-- {{#callable:llm_tokenizer_ugm_session::xcda_array_view::get_leaf}} -->
The `get_leaf` function checks if a node at a given index in an XCDA array is a leaf node by examining a specific bit in the packed node data.
- **Inputs**:
    - `index`: A size_t value representing the index of the node in the XCDA array to be checked.
- **Control Flow**:
    - Retrieve the packed node data from the XCDA array using the [`get_node`](#xcda_array_viewget_node) method with the provided index.
    - Shift the packed node data 8 bits to the right to isolate the leaf bit.
    - Perform a bitwise AND operation with 1 to extract the leaf bit value.
    - Return the result of the bitwise operation as a boolean indicating if the node is a leaf.
- **Output**: A boolean value indicating whether the node at the specified index is a leaf node.
- **Functions called**:
    - [`llm_tokenizer_ugm_session::xcda_array_view::get_node`](#xcda_array_viewget_node)
- **See also**: [`llm_tokenizer_ugm_session::xcda_array_view`](#llm_tokenizer_ugm_session::xcda_array_view)  (Data Structure)


---
#### xcda\_array\_view::get\_value<!-- {{#callable:llm_tokenizer_ugm_session::xcda_array_view::get_value}} -->
The `get_value` function retrieves a 31-bit value from a packed node at a specified index in an `xcda_array_view` structure.
- **Inputs**:
    - `index`: The position in the `xcda_array` from which to retrieve the packed node.
- **Control Flow**:
    - Call `get_node(index)` to retrieve the packed node at the specified index.
    - Apply a bitwise AND operation with `(1U << 31) - 1` to the packed node to extract the lower 31 bits.
- **Output**: A 31-bit unsigned integer value extracted from the packed node at the specified index.
- **Functions called**:
    - [`llm_tokenizer_ugm_session::xcda_array_view::get_node`](#xcda_array_viewget_node)
- **See also**: [`llm_tokenizer_ugm_session::xcda_array_view`](#llm_tokenizer_ugm_session::xcda_array_view)  (Data Structure)


---
#### xcda\_array\_view::get\_node<!-- {{#callable:llm_tokenizer_ugm_session::xcda_array_view::get_node}} -->
The `get_node` function retrieves a node from the `xcda_array` at a specified index, ensuring the index is within bounds.
- **Inputs**:
    - `index`: A `size_t` representing the index of the node to retrieve from the `xcda_array`.
- **Control Flow**:
    - Check if the provided index is greater than `xcda_array_size`.
    - If the index is out of bounds, throw a `std::runtime_error` with a specific error message.
    - If the index is valid, return the node at the specified index from `xcda_array`.
- **Output**: Returns a `uint32_t` representing the node at the specified index in the `xcda_array`.
- **See also**: [`llm_tokenizer_ugm_session::xcda_array_view`](#llm_tokenizer_ugm_session::xcda_array_view)  (Data Structure)



---
### best\_tokenization<!-- {{#data_structure:llm_tokenizer_ugm_session::best_tokenization}} -->
- **Type**: `struct`
- **Members**:
    - `token_id`: Represents the identifier of a token in the tokenization process.
    - `input_offset`: Indicates the offset position in the input where the token starts.
    - `score_sum`: Stores the cumulative score associated with the tokenization up to this point.
- **Description**: The `best_tokenization` struct is designed to encapsulate information about a specific tokenization result, including the token's identifier, its starting position in the input, and the cumulative score of the tokenization path leading to this token. This structure is likely used in processes where multiple tokenization paths are evaluated, and the best path needs to be tracked based on scoring criteria.


---
### llm\_tokenizer\_rwkv<!-- {{#data_structure:llm_tokenizer_rwkv}} -->
- **Type**: `struct`
- **Members**:
    - `token_matcher`: A naive_trie structure used for token matching during tokenization.
- **Description**: The `llm_tokenizer_rwkv` struct is a specialized tokenizer for the RWKV model, inheriting from the `llm_tokenizer` base class. It is designed to handle tokenization by supporting arbitrary byte tokens, which are decoded from a given vocabulary. The struct primarily uses a `naive_trie` data structure to build a lookup for tokenization, allowing it to efficiently match and tokenize input text based on the provided vocabulary.
- **Member Functions**:
    - [`llm_tokenizer_rwkv::llm_tokenizer_rwkv`](#llm_tokenizer_rwkvllm_tokenizer_rwkv)
- **Inherits From**:
    - [`llm_tokenizer::llm_tokenizer`](#llm_tokenizerllm_tokenizer)

**Methods**

---
#### llm\_tokenizer\_rwkv::llm\_tokenizer\_rwkv<!-- {{#callable:llm_tokenizer_rwkv::llm_tokenizer_rwkv}} -->
The `llm_tokenizer_rwkv` constructor initializes a tokenizer for RWKV models by building a trie from the given vocabulary to support tokenization.
- **Inputs**:
    - `vocab`: A reference to a `llama_vocab` object that contains the vocabulary data, including token strings and their associated data.
- **Control Flow**:
    - The constructor begins by iterating over each token in the provided vocabulary using a loop that runs from 0 to `vocab.n_tokens()`.
    - For each token, it retrieves the token data using `vocab.get_token_data(id)`.
    - The token text is then unescaped using the [`llama_unescape_rwkv_token`](#llama_unescape_rwkv_token) function to handle any escape sequences.
    - The unescaped text is inserted into a `naive_trie` data structure, `token_matcher`, along with its token ID, to build a lookup trie for tokenization.
- **Output**: The function does not return a value; it initializes the `token_matcher` trie within the `llm_tokenizer_rwkv` object for later use in tokenization.
- **Functions called**:
    - [`llama_unescape_rwkv_token`](#llama_unescape_rwkv_token)
- **See also**: [`llm_tokenizer_rwkv`](#llm_tokenizer_rwkv)  (Data Structure)



---
### llm\_tokenizer\_rwkv\_session<!-- {{#data_structure:llm_tokenizer_rwkv_session}} -->
- **Type**: `struct`
- **Members**:
    - `vocab`: A reference to a llama_vocab object, representing the vocabulary used for tokenization.
    - `tokenizer`: A reference to a llm_tokenizer_rwkv object, representing the tokenizer used for tokenization.
- **Description**: The `llm_tokenizer_rwkv_session` struct is designed to handle tokenization sessions using the RWKV tokenizer. It holds references to a vocabulary and a tokenizer, which are used to tokenize input text into a sequence of tokens. The struct provides a method `tokenize` that processes a given text, traverses a trie structure to find the longest matching tokens, and outputs the tokenized result into a vector. This struct is essential for managing the state and operations required during a tokenization session with the RWKV tokenizer.
- **Member Functions**:
    - [`llm_tokenizer_rwkv_session::llm_tokenizer_rwkv_session`](#llm_tokenizer_rwkv_sessionllm_tokenizer_rwkv_session)
    - [`llm_tokenizer_rwkv_session::tokenize`](#llm_tokenizer_rwkv_sessiontokenize)

**Methods**

---
#### llm\_tokenizer\_rwkv\_session::llm\_tokenizer\_rwkv\_session<!-- {{#callable:llm_tokenizer_rwkv_session::llm_tokenizer_rwkv_session}} -->
The `llm_tokenizer_rwkv_session` constructor initializes an instance of the `llm_tokenizer_rwkv_session` class with a given vocabulary and tokenizer.
- **Inputs**:
    - `vocab`: A constant reference to a `llama_vocab` object, representing the vocabulary to be used by the tokenizer session.
    - `tokenizer`: A constant reference to a `llm_tokenizer_rwkv` object, representing the tokenizer to be used by the session.
- **Control Flow**:
    - The constructor takes two parameters: a reference to a `llama_vocab` object and a reference to a `llm_tokenizer_rwkv` object.
    - It initializes the `vocab` and `tokenizer` member variables of the `llm_tokenizer_rwkv_session` class with the provided arguments.
- **Output**: There is no output from this constructor as it is used to initialize an object of the `llm_tokenizer_rwkv_session` class.
- **See also**: [`llm_tokenizer_rwkv_session`](#llm_tokenizer_rwkv_session)  (Data Structure)


---
#### llm\_tokenizer\_rwkv\_session::tokenize<!-- {{#callable:llm_tokenizer_rwkv_session::tokenize}} -->
The `tokenize` function processes a given text to identify and extract tokens using a trie-based token matcher, storing the results in an output vector.
- **Inputs**:
    - `text`: A constant reference to a `std::string` representing the input text to be tokenized.
    - `output`: A reference to a `std::vector<llama_token>` where the resulting tokens will be stored.
- **Control Flow**:
    - Initialize `position` to 0 to track the current position in the text.
    - Enter a while loop that continues as long as `position` is less than the size of the text.
    - Use the `token_matcher` to traverse the trie starting from the current character in the text to find a matching token node.
    - If no matching node is found, add an unknown token to the output and increment `position` by 1.
    - If a matching node is found, enter another while loop to traverse the trie further to find the longest matching token.
    - Within the inner loop, if the current node has a value, update `token_id` and `token_length` to store the longest matching token found so far.
    - Continue traversing the trie with the next character in the text until no further matching node is found.
    - Add the longest matching token to the output vector and update `position` to `token_length` to continue processing the text.
- **Output**: The function outputs the tokens found in the input text by appending them to the `output` vector.
- **See also**: [`llm_tokenizer_rwkv_session`](#llm_tokenizer_rwkv_session)  (Data Structure)



---
### FRAGMENT\_BUFFER\_VARIANT\_TYPE<!-- {{#data_structure:FRAGMENT_BUFFER_VARIANT_TYPE}} -->
- **Type**: `enum`
- **Members**:
    - `FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN`: Represents a token variant type in the fragment buffer.
    - `FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT`: Represents a raw text variant type in the fragment buffer.
- **Description**: The `FRAGMENT_BUFFER_VARIANT_TYPE` is an enumeration that defines two possible types for a fragment buffer variant: `FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN` and `FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT`. This enum is used to distinguish between different types of data that can be stored in a fragment buffer, specifically whether the data is a token or raw text.


---
### fragment\_buffer\_variant<!-- {{#data_structure:fragment_buffer_variant}} -->
- **Type**: `struct`
- **Members**:
    - `type`: Indicates the type of the fragment, either a token or raw text.
    - `token`: Holds a llama_token value when the fragment is of type token.
    - `_dummy`: A placeholder string used when the fragment is of type token.
    - `raw_text`: References the raw text string when the fragment is of type raw text.
    - `offset`: Specifies the starting position of the fragment within the raw text.
    - `length`: Indicates the length of the fragment within the raw text.
- **Description**: The `fragment_buffer_variant` struct is designed to represent a fragment of data that can either be a token or a segment of raw text. It uses a type indicator to determine the nature of the fragment and stores relevant data accordingly. When the fragment is a token, it holds a `llama_token` value and uses a dummy string for the raw text reference. Conversely, when the fragment is raw text, it references the actual text and records the offset and length of the fragment within that text. This struct is useful for handling and processing text data that may be in different forms, such as during tokenization or text parsing operations.
- **Member Functions**:
    - [`fragment_buffer_variant::fragment_buffer_variant`](#fragment_buffer_variantfragment_buffer_variant)
    - [`fragment_buffer_variant::fragment_buffer_variant`](#fragment_buffer_variantfragment_buffer_variant)

**Methods**

---
#### fragment\_buffer\_variant::fragment\_buffer\_variant<!-- {{#callable:fragment_buffer_variant::fragment_buffer_variant}} -->
The `fragment_buffer_variant` constructor initializes a `fragment_buffer_variant` object with a given `llama_token` and sets default values for other member variables.
- **Inputs**:
    - `_token`: A `llama_token` that is used to initialize the `token` member of the `fragment_buffer_variant` object.
- **Control Flow**:
    - The constructor is called with a `llama_token` parameter `_token`.
    - The `type` member is initialized to `FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN`.
    - The `token` member is initialized with the provided `_token`.
    - The `raw_text` member is initialized with `_dummy`.
    - The `offset` member is initialized to 0.
    - The `length` member is initialized to 0.
- **Output**: A `fragment_buffer_variant` object initialized with the specified `llama_token` and default values for other members.
- **See also**: [`fragment_buffer_variant`](#fragment_buffer_variant)  (Data Structure)


---
#### fragment\_buffer\_variant::fragment\_buffer\_variant<!-- {{#callable:fragment_buffer_variant::fragment_buffer_variant}} -->
The `fragment_buffer_variant` constructor initializes a `fragment_buffer_variant` object with raw text, offset, and length, while performing assertions to ensure valid input.
- **Inputs**:
    - `_raw_text`: A constant reference to a `std::string` representing the raw text to be stored in the fragment.
    - `_offset`: An `int64_t` representing the starting position of the fragment within the raw text.
    - `_length`: An `int64_t` representing the length of the fragment.
- **Control Flow**:
    - Initialize the `type` member to `FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT`.
    - Set the `token` member to `(llama_token) - 1`.
    - Assign `_raw_text` to the `raw_text` member.
    - Assign `_offset` to the `offset` member.
    - Assign `_length` to the `length` member.
    - Assert that `_offset` is non-negative.
    - Assert that `_length` is at least 1.
    - Assert that the sum of `offset` and `length` does not exceed the length of `raw_text`.
- **Output**: A `fragment_buffer_variant` object initialized with the specified raw text, offset, and length.
- **See also**: [`fragment_buffer_variant`](#fragment_buffer_variant)  (Data Structure)



---
### pair\_hash<!-- {{#data_structure:llama_vocab::impl::pair_hash}} -->
- **Type**: `struct`
- **Description**: The `pair_hash` struct is a custom hash function for `std::pair<std::string, std::string>`. It defines an `operator()` that takes a pair of strings and returns a size_t hash value. The hash is computed by XORing the hash of the first string with the left-shifted hash of the second string. This struct is used as a custom hash function in an `std::unordered_map` to store pairs of strings as keys.
- **Member Functions**:
    - [`llama_vocab::impl::pair_hash::operator()`](#pair_hashoperator())

**Methods**

---
#### pair\_hash::operator\(\)<!-- {{#callable:llama_vocab::impl::pair_hash::operator()}} -->
The `operator()` function in the `pair_hash` struct generates a hash value for a pair of strings by combining the hash of the first string with a left-shifted hash of the second string.
- **Inputs**:
    - `p`: A constant reference to a `std::pair` containing two `std::string` elements, representing the pair of strings to be hashed.
- **Control Flow**:
    - Compute the hash of the first string in the pair using `std::hash<std::string>{}(p.first)`.
    - Compute the hash of the second string in the pair using `std::hash<std::string>{}(p.second)` and left-shift it by 1 bit.
    - Combine the two hash values using the bitwise XOR operator `^`.
- **Output**: Returns a `size_t` value representing the combined hash of the pair of strings.
- **See also**: [`llama_vocab::impl::pair_hash`](#implpair_hash)  (Data Structure)



---
### llama\_vocab<!-- {{#data_structure:llama_vocab}} -->
- **Description**: [See definition](llama-vocab.h.driver.md#llama_vocab)
- **Member Functions**:
    - [`llama_vocab::llama_vocab`](#llama_vocabllama_vocab)
    - [`llama_vocab::~llama_vocab`](#llama_vocabllama_vocab)
    - [`llama_vocab::load`](#llama_vocabload)
    - [`llama_vocab::get_tokenizer_model`](#llama_vocabget_tokenizer_model)
    - [`llama_vocab::get_tokenizer_pre`](#llama_vocabget_tokenizer_pre)
    - [`llama_vocab::get_type`](#llama_vocabget_type)
    - [`llama_vocab::get_pre_type`](#llama_vocabget_pre_type)
    - [`llama_vocab::n_tokens`](#llama_vocabn_tokens)
    - [`llama_vocab::n_token_types`](#llama_vocabn_token_types)
    - [`llama_vocab::type_name`](#llama_vocabtype_name)
    - [`llama_vocab::is_normal`](#llama_vocabis_normal)
    - [`llama_vocab::is_unknown`](#llama_vocabis_unknown)
    - [`llama_vocab::is_control`](#llama_vocabis_control)
    - [`llama_vocab::is_byte`](#llama_vocabis_byte)
    - [`llama_vocab::is_user_defined`](#llama_vocabis_user_defined)
    - [`llama_vocab::is_unused`](#llama_vocabis_unused)
    - [`llama_vocab::is_eog`](#llama_vocabis_eog)
    - [`llama_vocab::token_to_byte`](#llama_vocabtoken_to_byte)
    - [`llama_vocab::byte_to_token`](#llama_vocabbyte_to_token)
    - [`llama_vocab::text_to_token`](#llama_vocabtext_to_token)
    - [`llama_vocab::get_token_data`](#llama_vocabget_token_data)
    - [`llama_vocab::token_get_text`](#llama_vocabtoken_get_text)
    - [`llama_vocab::token_get_score`](#llama_vocabtoken_get_score)
    - [`llama_vocab::token_get_attr`](#llama_vocabtoken_get_attr)
    - [`llama_vocab::token_bos`](#llama_vocabtoken_bos)
    - [`llama_vocab::token_eos`](#llama_vocabtoken_eos)
    - [`llama_vocab::token_eot`](#llama_vocabtoken_eot)
    - [`llama_vocab::token_eom`](#llama_vocabtoken_eom)
    - [`llama_vocab::token_unk`](#llama_vocabtoken_unk)
    - [`llama_vocab::token_sep`](#llama_vocabtoken_sep)
    - [`llama_vocab::token_nl`](#llama_vocabtoken_nl)
    - [`llama_vocab::token_pad`](#llama_vocabtoken_pad)
    - [`llama_vocab::token_prefix`](#llama_vocabtoken_prefix)
    - [`llama_vocab::token_middle`](#llama_vocabtoken_middle)
    - [`llama_vocab::token_suffix`](#llama_vocabtoken_suffix)
    - [`llama_vocab::token_fim_pre`](#llama_vocabtoken_fim_pre)
    - [`llama_vocab::token_fim_suf`](#llama_vocabtoken_fim_suf)
    - [`llama_vocab::token_fim_mid`](#llama_vocabtoken_fim_mid)
    - [`llama_vocab::token_fim_pad`](#llama_vocabtoken_fim_pad)
    - [`llama_vocab::token_fim_rep`](#llama_vocabtoken_fim_rep)
    - [`llama_vocab::token_fim_sep`](#llama_vocabtoken_fim_sep)
    - [`llama_vocab::get_add_space_prefix`](#llama_vocabget_add_space_prefix)
    - [`llama_vocab::get_add_bos`](#llama_vocabget_add_bos)
    - [`llama_vocab::get_add_eos`](#llama_vocabget_add_eos)
    - [`llama_vocab::get_ignore_merges`](#llama_vocabget_ignore_merges)
    - [`llama_vocab::get_clean_spaces`](#llama_vocabget_clean_spaces)
    - [`llama_vocab::get_remove_extra_whitespaces`](#llama_vocabget_remove_extra_whitespaces)
    - [`llama_vocab::get_escape_whitespaces`](#llama_vocabget_escape_whitespaces)
    - [`llama_vocab::get_treat_whitespace_as_suffix`](#llama_vocabget_treat_whitespace_as_suffix)
    - [`llama_vocab::max_token_len`](#llama_vocabmax_token_len)
    - [`llama_vocab::find_bpe_rank`](#llama_vocabfind_bpe_rank)
    - [`llama_vocab::get_bpe_merges`](#llama_vocabget_bpe_merges)
    - [`llama_vocab::get_precompiled_charsmap`](#llama_vocabget_precompiled_charsmap)
    - [`llama_vocab::tokenize`](#llama_vocabtokenize)
    - [`llama_vocab::tokenize`](#llama_vocabtokenize)
    - [`llama_vocab::token_to_piece`](#llama_vocabtoken_to_piece)
    - [`llama_vocab::token_to_piece`](#llama_vocabtoken_to_piece)
    - [`llama_vocab::detokenize`](#llama_vocabdetokenize)
    - [`llama_vocab::detokenize`](#llama_vocabdetokenize)
    - [`llama_vocab::print_info`](#llama_vocabprint_info)

**Methods**

---
#### llama\_vocab::llama\_vocab<!-- {{#callable:llama_vocab::llama_vocab}} -->
The `llama_vocab` constructor initializes a new instance of the `llama_vocab` class, creating a unique implementation object.
- **Inputs**: None
- **Control Flow**:
    - The constructor uses an initializer list to create a new instance of the `impl` struct, passing a reference to the current `llama_vocab` instance.
- **Output**: The constructor does not return a value; it initializes the internal state of the `llama_vocab` object.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::\~llama\_vocab<!-- {{#callable:llama_vocab::~llama_vocab}} -->
The `~llama_vocab` destructor is responsible for cleaning up resources associated with the `llama_vocab` class.
- **Inputs**: None
- **Control Flow**:
    - The destructor does not contain any specific logic or resource management code.
    - It is implicitly defined and will call the destructor of any member variables or base classes.
- **Output**: The function does not return any value as it is a destructor.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::load<!-- {{#callable:llama_vocab::load}} -->
The `load` function initializes the vocabulary by loading model data from a specified loader and key-value store.
- **Inputs**:
    - `ml`: A reference to a `llama_model_loader` object that is responsible for loading the model data.
    - `kv`: A constant reference to an `LLM_KV` object that contains key-value pairs for configuration and data.
- **Control Flow**:
    - The function calls the `load` method of the private implementation (`pimpl`) of the `llama_vocab` class.
    - The `load` method of `pimpl` takes the same parameters, `ml` and `kv`, to perform the loading operation.
- **Output**: The function does not return a value; it modifies the internal state of the `llama_vocab` instance by loading vocabulary data.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_tokenizer\_model<!-- {{#callable:llama_vocab::get_tokenizer_model}} -->
The `get_tokenizer_model` function retrieves the tokenizer model name from the `llama_vocab` class.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `tokenizer_model` member of the `pimpl` pointer, which is an instance of the private implementation of the `llama_vocab` class.
- **Output**: The function returns a `std::string` containing the name of the tokenizer model.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_tokenizer\_pre<!-- {{#callable:llama_vocab::get_tokenizer_pre}} -->
The `get_tokenizer_pre` function retrieves the pre-tokenizer model string from the `llama_vocab` class.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private implementation pointer `pimpl` of the `llama_vocab` class.
    - It returns the value of the `tokenizer_pre` member variable from the `pimpl` structure.
- **Output**: The output is a `std::string` representing the pre-tokenizer model string.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_type<!-- {{#callable:llama_vocab::get_type}} -->
The `get_type` method retrieves the vocabulary type of the `llama_vocab` instance.
- **Inputs**: None
- **Control Flow**:
    - The method directly accesses the `type` member of the `pimpl` pointer, which is an instance of the private implementation of the `llama_vocab` class.
- **Output**: The method returns an enumeration value of type `llama_vocab_type`, indicating the type of vocabulary used.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_pre\_type<!-- {{#callable:llama_vocab::get_pre_type}} -->
The `get_pre_type` method retrieves the pre-tokenization type of the `llama_vocab` instance.
- **Inputs**: None
- **Control Flow**:
    - The method accesses the `pimpl` member of the `llama_vocab` class, which is a pointer to an implementation structure.
    - It returns the `pre_type` member from the `pimpl` structure.
- **Output**: The method returns an enumeration value of type `llama_vocab_pre_type`, indicating the pre-tokenization type.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::n\_tokens<!-- {{#callable:llama_vocab::n_tokens}} -->
The `n_tokens` function returns the number of tokens in the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `pimpl` member of the `llama_vocab` class.
    - It retrieves the size of the `id_to_token` vector from the `pimpl` structure.
- **Output**: The function outputs a `uint32_t` representing the total number of tokens in the vocabulary.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::n\_token\_types<!-- {{#callable:llama_vocab::n_token_types}} -->
The `n_token_types` function returns the number of token types in the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `n_token_types` member of the `pimpl` structure.
    - It casts the value to `uint32_t` before returning.
- **Output**: The output is a `uint32_t` representing the number of token types in the vocabulary.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::type\_name<!-- {{#callable:llama_vocab::type_name}} -->
The `type_name` method returns the type name of the vocabulary used in the `llama_vocab` class.
- **Inputs**: None
- **Control Flow**:
    - The method directly calls the `type_name` method of the `pimpl` member, which is a pointer to an implementation class.
    - The result from the `pimpl->type_name()` call is returned as the output of the method.
- **Output**: The output is a string representing the type name of the vocabulary, which can be one of several predefined types such as 'SPM', 'BPE', 'WPM', etc.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::is\_normal<!-- {{#callable:llama_vocab::is_normal}} -->
The `is_normal` function checks if a given `llama_token` is classified as a normal token.
- **Inputs**:
    - `id`: A `llama_token` identifier that represents the token to be checked.
- **Control Flow**:
    - The function directly calls the `is_normal` method of the `pimpl` member, which is an instance of an implementation class.
    - The result of the `pimpl->is_normal(id)` call is returned as the output of the function.
- **Output**: Returns a boolean value indicating whether the specified `llama_token` is classified as normal.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::is\_unknown<!-- {{#callable:llama_vocab::is_unknown}} -->
The `is_unknown` function checks if a given `llama_token` ID is classified as unknown.
- **Inputs**:
    - `id`: A `llama_token` representing the token ID to be checked.
- **Control Flow**:
    - The function calls the `is_unknown` method of the `pimpl` member, which is a pointer to an implementation class.
    - The result of the `pimpl->is_unknown(id)` call is returned directly.
- **Output**: Returns a boolean value indicating whether the specified token ID is classified as unknown.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::is\_control<!-- {{#callable:llama_vocab::is_control}} -->
The `is_control` function checks if a given `llama_token` is classified as a control token.
- **Inputs**:
    - `id`: A `llama_token` identifier that is being checked for control status.
- **Control Flow**:
    - The function calls the `is_control` method of the `pimpl` member, which is a pointer to an implementation class.
    - The result of the `pimpl->is_control(id)` call is returned directly.
- **Output**: Returns a boolean value indicating whether the specified `llama_token` is a control token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::is\_byte<!-- {{#callable:llama_vocab::is_byte}} -->
The `is_byte` function checks if a given `llama_token` is classified as a byte token.
- **Inputs**:
    - `id`: A `llama_token` identifier that is being checked to determine if it is a byte.
- **Control Flow**:
    - The function directly calls the `is_byte` method of the `pimpl` member, which is an instance of an implementation class.
    - The result of the `pimpl->is_byte(id)` call is returned as the output of the function.
- **Output**: Returns a boolean value indicating whether the provided `llama_token` is classified as a byte token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::is\_user\_defined<!-- {{#callable:llama_vocab::is_user_defined}} -->
The `is_user_defined` function checks if a given `llama_token` is defined by the user.
- **Inputs**:
    - `id`: A `llama_token` identifier that is checked to determine if it is user-defined.
- **Control Flow**:
    - The function directly calls the `is_user_defined` method of the `pimpl` member, which is an instance of an implementation class.
    - The result of the `pimpl->is_user_defined(id)` call is returned as the output of the function.
- **Output**: Returns a boolean value indicating whether the specified `llama_token` is user-defined.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::is\_unused<!-- {{#callable:llama_vocab::is_unused}} -->
The `is_unused` function checks if a given `llama_token` is marked as unused in the vocabulary.
- **Inputs**:
    - `id`: A `llama_token` identifier that represents the token to be checked for its usage status.
- **Control Flow**:
    - The function directly calls the `is_unused` method of the `pimpl` member, which is a pointer to an implementation class.
    - The result of the `pimpl->is_unused(id)` call is returned as the output of the function.
- **Output**: Returns a boolean value indicating whether the specified `llama_token` is unused.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::is\_eog<!-- {{#callable:llama_vocab::is_eog}} -->
The `is_eog` function checks if a given `llama_token` ID corresponds to an end-of-generation token.
- **Inputs**:
    - `id`: A `llama_token` representing the token ID to be checked.
- **Control Flow**:
    - The function directly calls the `is_eog` method of the `pimpl` member, which is an instance of an implementation class.
    - It passes the `id` argument to this method.
- **Output**: Returns a boolean value indicating whether the provided token ID is classified as an end-of-generation token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_to\_byte<!-- {{#callable:llama_vocab::token_to_byte}} -->
The `token_to_byte` function converts a `llama_token` identifier to its corresponding byte representation.
- **Inputs**:
    - `id`: A `llama_token` identifier that represents a specific token in the vocabulary.
- **Control Flow**:
    - The function calls the `token_to_byte` method of the `pimpl` member, which is an instance of an implementation class.
    - The result of the `pimpl->token_to_byte(id)` call is returned directly.
- **Output**: Returns a `uint8_t` value representing the byte corresponding to the given `llama_token` identifier.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::byte\_to\_token<!-- {{#callable:llama_vocab::byte_to_token}} -->
Converts a byte represented as an `uint8_t` to a corresponding token using the vocabulary.
- **Inputs**:
    - `ch`: An 8-bit unsigned integer representing a byte value to be converted to a token.
- **Control Flow**:
    - Asserts that the vocabulary type is not 'none'.
    - Checks the type of vocabulary (SPM, UGM, WPM, BPE) to determine the conversion method.
    - For SPM and UGM types, constructs a hexadecimal string representation of the byte and looks it up in the token-to-id map.
    - If the hexadecimal representation is not found, it attempts to convert the byte directly to a string and look it up.
    - For WPM and BPE types, it converts the byte to a UTF-8 string and looks it up in the token-to-id map.
    - If the vocabulary type is unrecognized, it triggers a fatal error.
- **Output**: Returns the corresponding `llama_token` for the given byte, or triggers an error if the conversion fails.
- **Functions called**:
    - [`llama_vocab::impl::get_type`](#implget_type)
    - [`unicode_byte_to_utf8`](unicode.cpp.driver.md#unicode_byte_to_utf8)
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::text\_to\_token<!-- {{#callable:llama_vocab::text_to_token}} -->
Converts a given text string into its corresponding token ID using a vocabulary mapping.
- **Inputs**:
    - `text`: A constant reference to a string containing the text to be converted into a token.
- **Control Flow**:
    - The function first asserts that the vocabulary type is not 'none' using `GGML_ASSERT`.
    - It then attempts to find the token corresponding to the input text in the `token_to_id` map.
    - If the text is found, the corresponding token ID is returned.
    - If the text is not found, the function returns `LLAMA_TOKEN_NULL`.
- **Output**: Returns the token ID corresponding to the input text if found, otherwise returns `LLAMA_TOKEN_NULL`.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_token\_data<!-- {{#callable:llama_vocab::get_token_data}} -->
The `get_token_data` function retrieves the token data associated with a given token ID.
- **Inputs**:
    - `id`: A `llama_token` representing the ID of the token whose data is to be retrieved.
- **Control Flow**:
    - The function asserts that the vocabulary type is not `LLAMA_VOCAB_TYPE_NONE` using `GGML_ASSERT`.
    - It accesses the `id_to_token` map from the `pimpl` structure to retrieve the token data corresponding to the provided `id`.
- **Output**: Returns a constant reference to a `token_data` structure containing the text, score, and attributes of the specified token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_get\_text<!-- {{#callable:llama_vocab::token_get_text}} -->
The `token_get_text` function retrieves the text representation of a token given its identifier.
- **Inputs**:
    - `id`: A `llama_token` identifier representing the token whose text is to be retrieved.
- **Control Flow**:
    - The function first asserts that the vocabulary type is not `LLAMA_VOCAB_TYPE_NONE` using `GGML_ASSERT`.
    - It then accesses the `id_to_token` map from the `pimpl` structure to find the corresponding token data for the given `id`.
    - Finally, it returns the C-style string representation of the token's text using `c_str()`.
- **Output**: Returns a pointer to a constant character string representing the text of the specified token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_get\_score<!-- {{#callable:llama_vocab::token_get_score}} -->
The `token_get_score` function retrieves the score associated with a specific token ID from the vocabulary.
- **Inputs**:
    - `id`: A `llama_token` representing the token ID for which the score is to be retrieved.
- **Control Flow**:
    - The function first asserts that the vocabulary type is not `LLAMA_VOCAB_TYPE_NONE` using `GGML_ASSERT`.
    - It then accesses the `id_to_token` map from the `pimpl` structure to retrieve the `token_data` associated with the given token ID.
    - Finally, it returns the `score` field from the retrieved `token_data`.
- **Output**: Returns a `float` representing the score of the specified token ID.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_get\_attr<!-- {{#callable:llama_vocab::token_get_attr}} -->
The `token_get_attr` function retrieves the attribute of a specified token from the vocabulary.
- **Inputs**:
    - `id`: A `llama_token` representing the identifier of the token whose attribute is to be retrieved.
- **Control Flow**:
    - The function calls the `token_get_attr` method on the `pimpl` member, passing the `id` as an argument.
    - The result of the `token_get_attr` method is returned directly.
- **Output**: Returns a `llama_token_attr` that represents the attribute of the specified token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_bos<!-- {{#callable:llama_vocab::token_bos}} -->
The `token_bos` function retrieves the beginning-of-sequence token ID from the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private implementation pointer `pimpl` of the `llama_vocab` class.
    - It returns the value of `special_bos_id`, which represents the beginning-of-sequence token ID.
- **Output**: The function returns a `llama_token` representing the ID of the beginning-of-sequence token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_eos<!-- {{#callable:llama_vocab::token_eos}} -->
The `token_eos` function retrieves the special end-of-sequence token ID from the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private implementation pointer `pimpl` of the `llama_vocab` class.
    - It returns the value of `special_eos_id`, which represents the end-of-sequence token ID.
- **Output**: The function returns a `llama_token` representing the end-of-sequence token ID.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_eot<!-- {{#callable:llama_vocab::token_eot}} -->
The `token_eot` function retrieves the special end-of-text token identifier.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private implementation pointer `pimpl` of the `llama_vocab` class.
    - It returns the value of `special_eot_id`, which is a member of the `impl` structure.
- **Output**: The output is of type `llama_token`, representing the identifier for the end-of-text token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_eom<!-- {{#callable:llama_vocab::token_eom}} -->
The `token_eom` function retrieves the special end-of-message token ID from the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `special_eom_id` member of the `pimpl` pointer, which is an instance of the implementation class.
    - There are no conditional statements or loops; the function simply returns the value of `special_eom_id`.
- **Output**: The output is of type `llama_token`, representing the ID of the special end-of-message token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_unk<!-- {{#callable:llama_vocab::token_unk}} -->
The `token_unk` function returns the unique identifier for the unknown token in the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private implementation pointer `pimpl` of the `llama_vocab` class.
    - It retrieves the `special_unk_id`, which represents the unknown token identifier.
- **Output**: The function outputs a `llama_token` representing the identifier for the unknown token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_sep<!-- {{#callable:llama_vocab::token_sep}} -->
The `token_sep` function retrieves the special separator token ID from the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `pimpl` member of the `llama_vocab` class to retrieve the special separator token ID.
    - It directly returns the value of `special_sep_id` without any conditional logic or loops.
- **Output**: The function returns a `llama_token` representing the special separator token ID.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_nl<!-- {{#callable:llama_vocab::token_nl}} -->
The `token_nl` function retrieves the token ID associated with the newline character.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `linefeed_id` member of the `pimpl` structure.
    - No conditional logic or loops are present in this function.
- **Output**: The function returns a `llama_token` representing the newline character.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_pad<!-- {{#callable:llama_vocab::token_pad}} -->
The `token_pad` function retrieves the special padding token ID from the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `pimpl` member of the `llama_vocab` class to retrieve the `special_pad_id`.
- **Output**: The function returns a `llama_token` representing the ID of the padding token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_prefix<!-- {{#callable:llama_vocab::token_prefix}} -->
The `token_prefix` function retrieves the special prefix token ID from the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private implementation pointer `pimpl` of the `llama_vocab` class.
    - It directly returns the value of `special_fim_pre_id`, which is a member of the `impl` structure.
- **Output**: The output is of type `llama_token`, representing the ID of the special prefix token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_middle<!-- {{#callable:llama_vocab::token_middle}} -->
The `token_middle` function retrieves the special token ID for the middle token in the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `special_fim_mid_id` member of the `pimpl` pointer, which is an instance of the implementation class.
    - There are no conditional statements or loops; the function simply returns the value of `special_fim_mid_id`.
- **Output**: The output is of type `llama_token`, representing the ID of the special middle token in the vocabulary.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_suffix<!-- {{#callable:llama_vocab::token_suffix}} -->
The `token_suffix` method retrieves the special suffix token ID from the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The method accesses the private implementation pointer `pimpl` of the `llama_vocab` class.
    - It directly returns the value of `special_fim_suf_id`, which is a member of the `impl` structure.
- **Output**: The output is of type `llama_token`, representing the ID of the special suffix token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_fim\_pre<!-- {{#callable:llama_vocab::token_fim_pre}} -->
The `token_fim_pre` function retrieves the special 'fim pre' token ID from the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `special_fim_pre_id` member of the `pimpl` pointer, which is an instance of the implementation class.
    - No conditional logic or loops are present in this function.
- **Output**: The function returns a `llama_token` representing the ID of the special 'fim pre' token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_fim\_suf<!-- {{#callable:llama_vocab::token_fim_suf}} -->
The `token_fim_suf` function retrieves the special token ID for the suffix in the `llama_vocab` class.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `special_fim_suf_id` member of the `pimpl` pointer, which is an instance of the implementation class.
    - No conditional statements or loops are present, making the function straightforward and efficient.
- **Output**: The function returns a `llama_token` representing the special suffix token ID.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_fim\_mid<!-- {{#callable:llama_vocab::token_fim_mid}} -->
The `token_fim_mid` function retrieves the special token ID for the middle of a FIM (Function Invocation Model) sequence.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the `pimpl` member of the `llama_vocab` class, which is a pointer to an implementation structure.
    - It directly returns the value of `special_fim_mid_id` from the `pimpl` structure.
- **Output**: The output is of type `llama_token`, representing the special token ID for the middle of a FIM sequence.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_fim\_pad<!-- {{#callable:llama_vocab::token_fim_pad}} -->
The `token_fim_pad` function retrieves the special padding token ID from the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `special_fim_pad_id` member of the `pimpl` structure.
    - No conditional logic or loops are present in this function.
- **Output**: Returns the `llama_token` representing the special padding token ID.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_fim\_rep<!-- {{#callable:llama_vocab::token_fim_rep}} -->
The `token_fim_rep` function retrieves the special token ID representing a 'fim' repetition.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private implementation pointer `pimpl` of the `llama_vocab` class.
    - It directly returns the value of `special_fim_rep_id` from the `pimpl` structure.
- **Output**: The function returns a `llama_token` which is the ID of the special 'fim' repetition token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_fim\_sep<!-- {{#callable:llama_vocab::token_fim_sep}} -->
The `token_fim_sep` function retrieves the special file separator token ID from the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `special_fim_sep_id` member of the `pimpl` structure.
    - No conditional logic or loops are present in this function.
- **Output**: The function returns a `llama_token` representing the special file separator token ID.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_add\_space\_prefix<!-- {{#callable:llama_vocab::get_add_space_prefix}} -->
The `get_add_space_prefix` method retrieves the value of the `add_space_prefix` flag from the `pimpl` implementation.
- **Inputs**: None
- **Control Flow**:
    - The method directly accesses the `add_space_prefix` member of the `pimpl` object.
    - No conditional logic or loops are present in this method.
- **Output**: Returns a boolean value indicating whether the `add_space_prefix` feature is enabled.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_add\_bos<!-- {{#callable:llama_vocab::get_add_bos}} -->
The `get_add_bos` method retrieves the boolean flag indicating whether a beginning-of-sequence (BOS) token should be added.
- **Inputs**: None
- **Control Flow**:
    - The method accesses the `pimpl` member of the `llama_vocab` class, which is a pointer to an implementation detail.
    - It returns the value of the `add_bos` member from the `pimpl` structure.
- **Output**: The method returns a boolean value indicating if a BOS token should be added.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_add\_eos<!-- {{#callable:llama_vocab::get_add_eos}} -->
The `get_add_eos` function retrieves the boolean flag indicating whether an end-of-sequence (EOS) token should be added.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private implementation pointer `pimpl`.
    - It returns the value of the `add_eos` member variable from the `pimpl` structure.
- **Output**: The function returns a boolean value indicating if the EOS token should be added.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_ignore\_merges<!-- {{#callable:llama_vocab::get_ignore_merges}} -->
The `get_ignore_merges` function retrieves the value of the `ignore_merges` flag from the `pimpl` implementation.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `ignore_merges` member of the `pimpl` structure.
    - It returns the boolean value stored in `ignore_merges`.
- **Output**: The function returns a boolean indicating whether merges should be ignored during tokenization.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_clean\_spaces<!-- {{#callable:llama_vocab::get_clean_spaces}} -->
The `get_clean_spaces` method retrieves the value of the `clean_spaces` flag from the `pimpl` implementation.
- **Inputs**: None
- **Control Flow**:
    - The method directly accesses the `clean_spaces` member of the `pimpl` object.
    - No conditional logic or loops are present in this method.
- **Output**: The method returns a boolean value indicating whether the `clean_spaces` flag is set to true or false.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_remove\_extra\_whitespaces<!-- {{#callable:llama_vocab::get_remove_extra_whitespaces}} -->
The `get_remove_extra_whitespaces` function retrieves the setting for removing extra whitespaces from the tokenizer.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `remove_extra_whitespaces` member of the `pimpl` object.
    - It returns the value of `remove_extra_whitespaces` as a boolean.
- **Output**: The function returns a boolean indicating whether extra whitespaces should be removed.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_escape\_whitespaces<!-- {{#callable:llama_vocab::get_escape_whitespaces}} -->
The `get_escape_whitespaces` method retrieves the current setting for escaping whitespace characters.
- **Inputs**: None
- **Control Flow**:
    - The method accesses the private implementation pointer `pimpl`.
    - It returns the value of the `escape_whitespaces` member variable from the `pimpl` structure.
- **Output**: The method returns a boolean value indicating whether whitespace characters should be escaped.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_treat\_whitespace\_as\_suffix<!-- {{#callable:llama_vocab::get_treat_whitespace_as_suffix}} -->
The `get_treat_whitespace_as_suffix` function retrieves the boolean setting that indicates whether whitespace should be treated as a suffix.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private implementation pointer `pimpl` of the `llama_vocab` class.
    - It returns the value of the `treat_whitespace_as_suffix` member variable from the `pimpl` structure.
- **Output**: The function returns a boolean value indicating if whitespace is treated as a suffix.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::max\_token\_len<!-- {{#callable:llama_vocab::max_token_len}} -->
The `max_token_len` function returns the maximum length of tokens in the `llama_vocab`.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the private implementation pointer `pimpl` of the `llama_vocab` class.
    - It retrieves the `max_token_len` attribute from the `pimpl` structure and returns it.
- **Output**: The output is an integer representing the maximum length of tokens that can be processed by the vocabulary.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::find\_bpe\_rank<!-- {{#callable:llama_vocab::find_bpe_rank}} -->
The `find_bpe_rank` function retrieves the rank of a specified byte pair encoding (BPE) token pair from a vocabulary.
- **Inputs**:
    - `token_left`: A string representing the left token in the BPE pair.
    - `token_right`: A string representing the right token in the BPE pair.
- **Control Flow**:
    - The function first asserts that both `token_left` and `token_right` do not contain spaces or newline characters.
    - It then attempts to find the rank of the BPE pair formed by `token_left` and `token_right` in the `bpe_ranks` map.
    - If the pair is found, the corresponding rank is returned; otherwise, -1 is returned.
- **Output**: Returns the rank of the BPE pair if found, otherwise returns -1.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_bpe\_merges<!-- {{#callable:llama_vocab::get_bpe_merges}} -->
The `get_bpe_merges` function retrieves the byte pair encoding (BPE) merges from the vocabulary.
- **Inputs**: None
- **Control Flow**:
    - A vector `result` is initialized with the size equal to the number of BPE ranks.
    - A for loop iterates over each pair in `pimpl->bpe_ranks`, where each pair consists of a key (a pair of strings) and a value (an index).
    - For each pair, the first and second elements of the key are concatenated with a space and assigned to the `result` vector at the index specified by the value.
- **Output**: The function returns a vector of strings, where each string represents a BPE merge in the format 'first second'.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::get\_precompiled\_charsmap<!-- {{#callable:llama_vocab::get_precompiled_charsmap}} -->
The `get_precompiled_charsmap` function retrieves the precompiled character mapping as a vector of characters.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `precompiled_charsmap` member of the `pimpl` pointer, which is an instance of a private implementation structure.
- **Output**: Returns a `std::vector<char>` containing the precompiled character mapping.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::tokenize<!-- {{#callable:llama_vocab::tokenize}} -->
The [`tokenize`](#llm_tokenizer_spm_sessiontokenize) function converts a given text into a sequence of tokens, storing them in a provided array while ensuring the number of tokens does not exceed a specified maximum.
- **Inputs**:
    - `text`: A pointer to a character array containing the text to be tokenized.
    - `text_len`: An integer representing the length of the text to be tokenized.
    - `tokens`: A pointer to an array of `llama_token` where the resulting tokens will be stored.
    - `n_tokens_max`: An integer specifying the maximum number of tokens that can be stored in the `tokens` array.
    - `add_special`: A boolean indicating whether to add special tokens (like start or end of sequence) to the tokenized output.
    - `parse_special`: A boolean indicating whether to parse special tokens during tokenization.
- **Control Flow**:
    - The function first calls an overloaded [`tokenize`](#llm_tokenizer_spm_sessiontokenize) method that takes a `std::string` to perform the actual tokenization.
    - It checks if the number of tokens generated exceeds `n_tokens_max`, returning a negative value if it does.
    - If the tokenization is successful, it copies the generated tokens into the provided `tokens` array.
    - Finally, it returns the number of tokens generated.
- **Output**: The function returns the number of tokens generated, or a negative value if the number of tokens exceeds the specified maximum.
- **Functions called**:
    - [`llm_tokenizer_spm_session::tokenize`](#llm_tokenizer_spm_sessiontokenize)
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::token\_to\_piece<!-- {{#callable:llama_vocab::token_to_piece}} -->
The `token_to_piece` function retrieves the string representation of a given token from the vocabulary.
- **Inputs**:
    - `token`: A `llama_token` representing the token whose string representation is to be retrieved.
- **Control Flow**:
    - The function calls the `token_to_piece` method of the `pimpl` member, which is a pointer to an implementation of the vocabulary.
    - The result of the call is returned directly, which is a reference to a string.
- **Output**: The function returns a constant reference to a string that represents the piece corresponding to the input token.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::detokenize<!-- {{#callable:llama_vocab::detokenize}} -->
The `detokenize` function converts a sequence of tokens back into a human-readable string.
- **Inputs**:
    - `tokens`: An array of `llama_token` representing the tokens to be converted back into text.
    - `n_tokens`: The number of tokens in the `tokens` array.
    - `text`: A character buffer where the resulting text will be stored.
    - `text_len_max`: The maximum length of the `text` buffer.
    - `remove_special`: A boolean flag indicating whether to remove special tokens from the output.
    - `unparse_special`: A boolean flag indicating whether to unparse special tokens.
- **Control Flow**:
    - The function first checks if the vocabulary type is none, returning 0 if true.
    - It initializes variables for available space in the output buffer and total characters written.
    - It checks for and removes the beginning of sequence (BOS) token if specified.
    - It checks for and removes the end of sequence (EOS) token if specified.
    - For each token in the input array, it converts the token to its corresponding text representation using `token_to_piece`.
    - It handles whitespace and special character cleaning based on the provided flags.
    - Finally, it returns the total number of characters written to the output buffer.
- **Output**: The function returns the total number of characters written to the `text` buffer, or a negative value if an error occurs.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)


---
#### llama\_vocab::print\_info<!-- {{#callable:llama_vocab::print_info}} -->
The `print_info` method outputs information about the `llama_vocab` instance.
- **Inputs**: None
- **Control Flow**:
    - The method calls the `print_info` method of the `pimpl` member, which is a pointer to an implementation of the `llama_vocab` class.
- **Output**: The output is the result of the `print_info` method from the `pimpl` object, which likely logs or displays information about the vocabulary.
- **See also**: [`llama_vocab`](llama-vocab.h.driver.md#llama_vocab)  (Data Structure)



# Functions

---
### llama\_unescape\_rwkv\_token<!-- {{#callable:llama_unescape_rwkv_token}} -->
The `llama_unescape_rwkv_token` function decodes a string containing escaped characters and hexadecimal values into a vector of bytes.
- **Inputs**:
    - `escaped`: A string containing escaped characters and hexadecimal sequences that need to be decoded.
- **Control Flow**:
    - The function initializes a vector `output` to store the decoded bytes and reserves space based on the size of the input string.
    - It uses a loop to iterate through each character in the input string.
    - If a hexadecimal sequence is being parsed (indicated by `hex_remaining`), it calculates the byte value from the hex characters.
    - If an escape character is detected (indicated by `escaping`), it decodes specific escape sequences (like '\t', '\n', '\r') or starts a new hex sequence.
    - If a backslash is encountered, it sets the `escaping` flag to true for the next character.
    - Finally, it appends the decoded byte or character to the `output` vector.
- **Output**: A vector of `uint8_t` containing the decoded byte values from the input string.


---
### impl<!-- {{#callable:llama_vocab::impl::impl}} -->
The `impl` constructor initializes the `impl` class with a reference to a `llama_vocab` object.
- **Inputs**:
    - `vocab`: A constant reference to a `llama_vocab` object that provides vocabulary data for the implementation.
- **Control Flow**:
    - The constructor initializes the member variable `vocab` with the provided `vocab` argument.
- **Output**: This constructor does not return a value; it initializes the internal state of the `impl` class.


---
### \~impl<!-- {{#callable:llama_vocab::impl::~impl}} -->
The `~impl` function is a default destructor for the `impl` class, which is part of the `llama_vocab` class.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements as it is a default destructor.
    - It is implicitly defined to clean up resources when an instance of the `impl` class is destroyed.
- **Output**: The function does not return any value as it is a destructor.


---
### load<!-- {{#callable:llama_vocab::impl::load}} -->
The `load` function initializes the vocabulary and tokenizer settings from a model loader.
- **Inputs**:
    - `ml`: A reference to a `llama_model_loader` object that provides access to model metadata.
    - `kv`: A constant reference to an `LLM_KV` object that contains key-value pairs for configuration.
- **Control Flow**:
    - The function retrieves the tokenizer model type and other settings from the model loader.
    - Based on the tokenizer model type, it sets the vocabulary type and initializes special token IDs.
    - If the tokenizer model is 'no_vocab' or 'none', it sets the vocabulary type to none and initializes default special tokens.
    - For specific tokenizer models like 'llama', 'bert', 'gpt2', 't5', and 'rwkv', it configures the vocabulary and special tokens accordingly.
    - It reads the vocabulary size and initializes the token-to-ID and ID-to-token mappings.
    - The function also handles special tokens and their attributes, ensuring they are correctly set up for the tokenizer.
- **Output**: The function does not return a value but modifies the internal state of the vocabulary object, setting up the tokenizer and special tokens.
- **Functions called**:
    - [`gguf_find_key`](../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`gguf_get_arr_n`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_n)
    - [`gguf_get_arr_str`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_str)
    - [`gguf_get_arr_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_type)
    - [`gguf_get_arr_data`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_data)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`llama_vocab::impl::init_tokenizer`](#implinit_tokenizer)
    - [`llm_tokenizer_spm_session::tokenize`](#llm_tokenizer_spm_sessiontokenize)
    - [`llama_vocab::impl::token_to_piece_for_cache`](#impltoken_to_piece_for_cache)


---
### get\_type<!-- {{#callable:llama_vocab::impl::get_type}} -->
The `get_type` method retrieves the type of the vocabulary used in the `llama_vocab` class.
- **Inputs**: None
- **Control Flow**:
    - The method directly returns the value of the member variable `type`.
- **Output**: The output is of type `enum llama_vocab_type`, representing the vocabulary type.


---
### type\_name<!-- {{#callable:llama_vocab::impl::type_name}} -->
The `type_name` function returns a string representation of the vocabulary type.
- **Inputs**: None
- **Control Flow**:
    - The function uses a `switch` statement to determine the value of the `type` member variable.
    - For each case corresponding to a specific vocabulary type, it returns a string that describes that type.
    - If the `type` does not match any predefined cases, it defaults to returning 'unknown'.
- **Output**: The output is a string that indicates the type of vocabulary, such as 'no vocab', 'SPM', 'BPE', 'WPM', 'UGM', or 'RWKV'.


---
### is\_normal<!-- {{#callable:llama_vocab::impl::is_normal}} -->
The `is_normal` function checks if a given `llama_token` is classified as a normal token.
- **Inputs**:
    - `id`: A `llama_token` identifier that represents a specific token in the vocabulary.
- **Control Flow**:
    - The function asserts that the vocabulary type is not `LLAMA_VOCAB_TYPE_NONE` using `GGML_ASSERT`.
    - It retrieves the attributes of the token identified by `id` from the `id_to_token` array and checks if the `LLAMA_TOKEN_ATTR_NORMAL` attribute is set.
- **Output**: Returns a boolean value indicating whether the token is classified as normal.


---
### is\_unknown<!-- {{#callable:llama_vocab::impl::is_unknown}} -->
The `is_unknown` function checks if a given `llama_token` is marked as unknown.
- **Inputs**:
    - `id`: A `llama_token` identifier that is being checked for the unknown attribute.
- **Control Flow**:
    - The function asserts that the vocabulary type is not `LLAMA_VOCAB_TYPE_NONE` using `GGML_ASSERT`.
    - It retrieves the attributes of the token corresponding to the provided `id` from the `id_to_token` mapping.
    - The function checks if the `LLAMA_TOKEN_ATTR_UNKNOWN` attribute is set for the token and returns the result.
- **Output**: Returns a boolean indicating whether the token is marked as unknown.


---
### is\_control<!-- {{#callable:llama_vocab::impl::is_control}} -->
The `is_control` function checks if a given `llama_token` is classified as a control token.
- **Inputs**:
    - `id`: A `llama_token` identifier that is checked for control attributes.
- **Control Flow**:
    - The function asserts that the vocabulary type is not `LLAMA_VOCAB_TYPE_NONE` using `GGML_ASSERT`.
    - It retrieves the attributes of the token identified by `id` from the `id_to_token` mapping.
    - The function checks if the `LLAMA_TOKEN_ATTR_CONTROL` attribute is set for the token.
- **Output**: Returns a boolean indicating whether the specified token is a control token.


---
### is\_byte<!-- {{#callable:llama_vocab::impl::is_byte}} -->
The `is_byte` function checks if a given `llama_token` is classified as a byte token.
- **Inputs**:
    - `id`: A `llama_token` identifier that is being checked for the byte attribute.
- **Control Flow**:
    - The function asserts that the vocabulary type is not 'none' using `GGML_ASSERT`.
    - It retrieves the attributes of the token associated with the given `id` from the `id_to_token` mapping.
    - The function checks if the `LLAMA_TOKEN_ATTR_BYTE` attribute is set for the token.
- **Output**: Returns a boolean indicating whether the token is classified as a byte token.


---
### is\_user\_defined<!-- {{#callable:llama_vocab::impl::is_user_defined}} -->
The `is_user_defined` function checks if a given token ID corresponds to a user-defined token.
- **Inputs**:
    - `id`: A `llama_token` representing the token ID to be checked.
- **Control Flow**:
    - The function asserts that the vocabulary type is not 'none' using `GGML_ASSERT`.
    - It retrieves the attributes of the token corresponding to the given ID from the `id_to_token` mapping.
    - It checks if the `LLAMA_TOKEN_ATTR_USER_DEFINED` attribute is set for the token.
- **Output**: Returns a boolean value indicating whether the token is user-defined.


---
### is\_unused<!-- {{#callable:llama_vocab::impl::is_unused}} -->
Checks if a given `llama_token` is marked as unused in the vocabulary.
- **Inputs**:
    - `id`: A `llama_token` identifier that is checked for the unused attribute.
- **Control Flow**:
    - Asserts that the vocabulary type is not `LLAMA_VOCAB_TYPE_NONE` to ensure valid state.
    - Returns true if the `attr` attribute of the token at index `id` contains the `LLAMA_TOKEN_ATTR_UNUSED` flag, indicating that the token is unused.
- **Output**: Returns a boolean value: true if the token is unused, false otherwise.


---
### is\_eog<!-- {{#callable:llama_vocab::impl::is_eog}} -->
Determines if a given `llama_token` ID represents an end-of-generation (EOG) token.
- **Inputs**:
    - `id`: A `llama_token` representing the token ID to be checked.
- **Control Flow**:
    - The function first checks if the `id` is not equal to `LLAMA_TOKEN_NULL`.
    - Then, it checks if the `id` exists in the `special_eog_ids` set.
    - If both conditions are satisfied, it returns true; otherwise, it returns false.
- **Output**: Returns a boolean value indicating whether the provided token ID is an end-of-generation token.


---
### token\_to\_byte<!-- {{#callable:llama_vocab::impl::token_to_byte}} -->
Converts a `llama_token` to its corresponding byte representation.
- **Inputs**:
    - `id`: A `llama_token` identifier that represents a specific token in the vocabulary.
- **Control Flow**:
    - Asserts that the vocabulary type is not `LLAMA_VOCAB_TYPE_NONE`.
    - Asserts that the token `id` is of byte type.
    - Retrieves the token data associated with the given `id`.
    - Switches based on the vocabulary type to determine how to convert the token to a byte.
    - For `LLAMA_VOCAB_TYPE_SPM` and `LLAMA_VOCAB_TYPE_UGM`, extracts a substring from the token's text and converts it from hexadecimal to a byte.
    - For `LLAMA_VOCAB_TYPE_BPE` and `LLAMA_VOCAB_TYPE_WPM`, triggers a fatal error indicating that these types do not support byte conversion.
- **Output**: Returns the byte representation of the token as a `uint8_t` value, or triggers a fatal error for unsupported vocabulary types.
- **Functions called**:
    - [`llama_vocab::impl::get_type`](#implget_type)
    - [`llama_vocab::impl::is_byte`](#implis_byte)


---
### token\_get\_attr<!-- {{#callable:llama_vocab::impl::token_get_attr}} -->
The `token_get_attr` function retrieves the attributes of a token identified by its ID.
- **Inputs**:
    - `id`: A `llama_token` representing the unique identifier of the token whose attributes are to be retrieved.
- **Control Flow**:
    - The function first asserts that the vocabulary type is not `LLAMA_VOCAB_TYPE_NONE` to ensure valid operation.
    - It then accesses the `id_to_token` map using the provided `id` to retrieve the corresponding token data.
    - Finally, it returns the `attr` attribute of the retrieved token data.
- **Output**: Returns a `llama_token_attr` value representing the attributes of the specified token.


---
### init\_tokenizer<!-- {{#callable:llama_vocab::impl::init_tokenizer}} -->
Initializes the tokenizer based on the specified vocabulary type.
- **Inputs**:
    - `type`: An enumeration value of type `llama_vocab_type` that specifies the tokenizer type to initialize.
- **Control Flow**:
    - Logs a debug message indicating the initialization of the tokenizer with the specified type.
    - Uses a switch statement to determine the tokenizer type based on the input `type`.
    - For each case corresponding to a tokenizer type, it creates a unique pointer to the appropriate tokenizer class and assigns it to the `tokenizer` member variable.
    - If the input type does not match any known tokenizer types, it calls `GGML_ABORT` to terminate the program with an error message.
- **Output**: The function does not return a value; it initializes the `tokenizer` member variable of the `llama_vocab::impl` class.


---
### tokenizer\_st\_partition<!-- {{#callable:llama_vocab::impl::tokenizer_st_partition}} -->
The `tokenizer_st_partition` function processes a list of text fragments, inserting special tokens at appropriate positions based on their occurrences.
- **Inputs**:
    - `buffer`: A forward list of `fragment_buffer_variant` objects representing text fragments and tokens to be processed.
    - `parse_special`: A boolean flag indicating whether to parse special tokens or ignore control and unknown tokens.
- **Control Flow**:
    - Iterates over each special token in `cache_special_tokens`.
    - For each special token, retrieves its associated text and attributes.
    - If `parse_special` is false and the token is a control or unknown token, it skips further processing for that token.
    - Iterates through each fragment in the `buffer` to find occurrences of the special token's text.
    - For each occurrence found, it splits the fragment into left and right parts around the special token, inserting them back into the buffer.
    - Handles whitespace trimming based on token attributes (LSTRIP and RSTRIP) when inserting left and right fragments.
- **Output**: The function modifies the input `buffer` in place, inserting special tokens and adjusting the text fragments accordingly.


---
### token\_to\_piece\_for\_cache<!-- {{#callable:llama_vocab::impl::token_to_piece_for_cache}} -->
Converts a `llama_token` to its corresponding string piece representation, utilizing caching for efficiency.
- **Inputs**:
    - `token`: The `llama_token` that needs to be converted to a string piece.
    - `special`: A boolean flag indicating whether to treat the token as a special token.
- **Control Flow**:
    - Initializes an empty string `piece` and resizes it based on its capacity.
    - Calls `vocab.token_to_piece` to fill `piece` with the string representation of the token.
    - If the number of characters returned is negative, it resizes `piece` to the absolute value of the return value and calls `vocab.token_to_piece` again to fill it correctly.
    - If the number of characters is non-negative, it resizes `piece` to the actual number of characters returned.
- **Output**: Returns the string representation of the token as a `std::string`.


---
### llama\_escape\_whitespace<!-- {{#callable:llama_escape_whitespace}} -->
The `llama_escape_whitespace` function replaces all spaces in a given string with a specific Unicode character.
- **Inputs**:
    - `text`: A reference to a `std::string` that contains the text in which whitespace characters will be replaced.
- **Control Flow**:
    - The function calls [`replace_all`](llama-impl.cpp.driver.md#replace_all), which is assumed to be a utility function that performs the replacement operation.
    - The [`replace_all`](llama-impl.cpp.driver.md#replace_all) function takes three parameters: the string to modify, the substring to find (in this case, a space), and the substring to replace it with (the Unicode character for a lower one-eighth block).
- **Output**: The function does not return a value; it modifies the input string `text` in place.
- **Functions called**:
    - [`replace_all`](llama-impl.cpp.driver.md#replace_all)


---
### llama\_unescape\_whitespace<!-- {{#callable:llama_unescape_whitespace}} -->
The `llama_unescape_whitespace` function replaces all occurrences of a specific escaped whitespace character in a string with a regular space.
- **Inputs**:
    - `word`: A reference to a `std::string` that contains the text in which escaped whitespace characters are to be replaced.
- **Control Flow**:
    - The function calls [`replace_all`](llama-impl.cpp.driver.md#replace_all) to search for the specific escaped whitespace character (represented as 'xe2x96x81') in the `word` string.
    - All instances of the escaped character are replaced with a regular space character.
- **Output**: The function does not return a value; it modifies the input string `word` in place.
- **Functions called**:
    - [`replace_all`](llama-impl.cpp.driver.md#replace_all)


---
### llama\_decode\_text<!-- {{#callable:llama_decode_text}} -->
The `llama_decode_text` function decodes a UTF-8 encoded string into a byte representation, handling any invalid characters by appending an unknown byte placeholder.
- **Inputs**:
    - `text`: A constant reference to a `std::string` containing the UTF-8 encoded text to be decoded.
- **Control Flow**:
    - The function initializes an empty `std::string` called `decoded_text` to store the result.
    - It calls [`unicode_cpts_from_utf8`](unicode.cpp.driver.md#unicode_cpts_from_utf8) to convert the input `text` into a sequence of Unicode code points.
    - For each code point, it converts it back to UTF-8 using [`unicode_cpt_to_utf8`](unicode.cpp.driver.md#unicode_cpt_to_utf8).
    - It attempts to convert the UTF-8 string to a byte using [`unicode_utf8_to_byte`](unicode.cpp.driver.md#unicode_utf8_to_byte) and appends it to `decoded_text`.
    - If an `std::out_of_range` exception is thrown, it appends a placeholder indicating an unknown byte, followed by the original text.
- **Output**: Returns a `std::string` containing the decoded byte representation of the input text, with unknown bytes represented as placeholders.
- **Functions called**:
    - [`unicode_cpts_from_utf8`](unicode.cpp.driver.md#unicode_cpts_from_utf8)
    - [`unicode_cpt_to_utf8`](unicode.cpp.driver.md#unicode_cpt_to_utf8)
    - [`unicode_utf8_to_byte`](unicode.cpp.driver.md#unicode_utf8_to_byte)
    - [`format`](llama-impl.cpp.driver.md#format)


---
### tokenize<!-- {{#callable:llama_vocab::impl::tokenize}} -->
The `tokenize` function processes a raw text input and converts it into a vector of tokens based on the specified tokenizer type.
- **Inputs**:
    - `raw_text`: A string containing the text to be tokenized.
    - `add_special`: A boolean indicating whether to add special tokens (like beginning-of-sequence or end-of-sequence tokens) to the output.
    - `parse_special`: A boolean indicating whether to parse special tokens during the tokenization process.
- **Control Flow**:
    - The function first asserts that the tokenizer is initialized, throwing an error if it is not.
    - It initializes an empty vector `output` to store the resulting tokens and a forward list `fragment_buffer` to hold text fragments.
    - If `raw_text` is not empty, it adds the entire text as a fragment to `fragment_buffer` and calls [`tokenizer_st_partition`](#impltokenizer_st_partition) to process it.
    - The function then switches on the tokenizer type, executing different tokenization logic for each type (SPM, BPE, WPM, UGM, RWKV).
    - For each fragment in `fragment_buffer`, it processes raw text or tokens, applying the appropriate tokenization logic and adding tokens to the `output` vector.
    - If `add_special` is true, it adds special tokens (like BOS or EOS) to the output based on the specified conditions.
- **Output**: The function returns a vector of `llama_token` representing the tokenized form of the input text, including any special tokens if specified.
- **Functions called**:
    - [`llama_vocab::impl::tokenizer_st_partition`](#impltokenizer_st_partition)
    - [`llama_vocab::impl::get_type`](#implget_type)
    - [`llama_escape_whitespace`](#llama_escape_whitespace)


---
### token\_to\_piece<!-- {{#callable:llama_vocab::impl::token_to_piece}} -->
The `token_to_piece` function retrieves the string representation of a given token from a cached mapping.
- **Inputs**:
    - `token`: A `llama_token` representing the token whose string representation is to be retrieved.
- **Control Flow**:
    - The function accesses the `cache_token_to_piece` member, which is a vector of strings.
    - It uses the provided `token` as an index to retrieve the corresponding string from the cache.
- **Output**: Returns a constant reference to the string representation of the specified token.


---
### detokenize<!-- {{#callable:llama_vocab::impl::detokenize}} -->
The `detokenize` function converts a sequence of tokens back into a human-readable string.
- **Inputs**:
    - `tokens`: An array of `llama_token` representing the tokens to be converted back to text.
    - `n_tokens`: The number of tokens in the `tokens` array.
    - `text`: A character array where the resulting text will be stored.
    - `text_len_max`: The maximum length of the `text` buffer.
    - `remove_special`: A boolean flag indicating whether to remove special tokens (like BOS and EOS) from the output.
    - `unparse_special`: A boolean flag indicating whether to unparse special tokens into their original forms.
- **Control Flow**:
    - The function first checks if the vocabulary type is valid; if not, it returns 0.
    - It asserts that the tokenizer is initialized.
    - It initializes variables to track available space in the output buffer and the total number of characters written.
    - It checks if the first token is a special BOS token and adjusts the token count and pointer accordingly.
    - It checks if the last token is a special EOS token and decrements the token count if necessary.
    - For each token, it converts the token to its corresponding text piece and appends it to the output buffer, managing spaces as needed.
    - If the total characters exceed the maximum length, it returns a negative value indicating the overflow.
    - If `clean_spaces` is true, it performs additional processing to clean up spaces in the output text.
    - Finally, it returns the total number of characters written to the output buffer, or a negative value if it exceeds the maximum length.
- **Output**: The function returns the total number of characters written to the `text` buffer, or a negative value if the output exceeds the specified maximum length.
- **Functions called**:
    - [`llama_vocab::impl::token_to_piece`](#impltoken_to_piece)


---
### print\_info<!-- {{#callable:llama_vocab::impl::print_info}} -->
The `print_info` method logs detailed information about the vocabulary, including its type, number of tokens, number of merges, and various special tokens.
- **Inputs**: None
- **Control Flow**:
    - Logs the vocabulary type using `LLAMA_LOG_INFO`.
    - Logs the number of tokens in the vocabulary.
    - Logs the number of merges in the BPE ranks.
    - Checks and logs each special token if it is not null, including BOS, EOS, EOT, EOM, UNK, SEP, PAD, MASK, and LF tokens.
    - Iterates through a collection of end-of-generation tokens and logs each.
- **Output**: The method does not return a value; it outputs information to the log.
- **Functions called**:
    - [`llama_vocab::impl::type_name`](#impltype_name)


---
### llama\_vocab\_n\_tokens<!-- {{#callable:llama_vocab_n_tokens}} -->
The `llama_vocab_n_tokens` function returns the number of tokens in the given vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary data.
- **Control Flow**:
    - The function directly accesses the `n_tokens` method of the `vocab` structure.
    - It returns the result of the `n_tokens` method call.
- **Output**: An integer representing the total number of tokens in the vocabulary.


---
### llama\_n\_vocab<!-- {{#callable:llama_n_vocab}} -->
The `llama_n_vocab` function retrieves the number of tokens in a given vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary data.
- **Control Flow**:
    - The function calls [`llama_vocab_n_tokens`](#llama_vocab_n_tokens) with the provided `vocab` pointer to get the number of tokens.
- **Output**: Returns an integer representing the total number of tokens in the vocabulary.
- **Functions called**:
    - [`llama_vocab_n_tokens`](#llama_vocab_n_tokens)


---
### llama\_vocab\_type<!-- {{#callable:llama_vocab_type}} -->
The `llama_vocab_type` function retrieves the vocabulary type of a given `llama_vocab` structure.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure whose vocabulary type is to be retrieved.
- **Control Flow**:
    - The function directly accesses the `get_type()` method of the `llama_vocab` structure to obtain the vocabulary type.
- **Output**: Returns an enumeration value of type `llama_vocab_type` representing the type of the vocabulary.


---
### llama\_vocab\_get\_text<!-- {{#callable:llama_vocab_get_text}} -->
The `llama_vocab_get_text` function retrieves the text representation of a specified token from a vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary data.
    - `token`: A `llama_token` representing the specific token whose text representation is to be retrieved.
- **Control Flow**:
    - The function calls the `token_get_text` method on the `vocab` structure, passing the `token` as an argument.
    - The result of the `token_get_text` method is returned directly.
- **Output**: The function returns a pointer to a constant character string representing the text associated with the specified token.


---
### llama\_vocab\_get\_score<!-- {{#callable:llama_vocab_get_score}} -->
The `llama_vocab_get_score` function retrieves the score associated with a specific token from the vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary data.
    - `token`: A `llama_token` representing the specific token for which the score is to be retrieved.
- **Control Flow**:
    - The function directly calls the `token_get_score` method on the `vocab` structure, passing the `token` as an argument.
    - No conditional statements or loops are present; the function simply returns the result of the method call.
- **Output**: Returns a float value representing the score of the specified token.


---
### llama\_vocab\_get\_attr<!-- {{#callable:llama_vocab_get_attr}} -->
The `llama_vocab_get_attr` function retrieves the attributes of a specified token from the vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary data.
    - `token`: A `llama_token` representing the specific token whose attributes are to be retrieved.
- **Control Flow**:
    - The function calls the `token_get_attr` method of the `vocab` structure, passing the `token` as an argument.
    - The result of the `token_get_attr` method is returned directly.
- **Output**: Returns an enumeration value of type `llama_token_attr` that represents the attributes of the specified token.


---
### llama\_vocab\_is\_eog<!-- {{#callable:llama_vocab_is_eog}} -->
The `llama_vocab_is_eog` function checks if a given token is an end-of-generation (EOG) token in the specified vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
    - `token`: A `llama_token` representing the token to be checked.
- **Control Flow**:
    - The function calls the `is_eog` method of the `vocab` structure, passing the `token` as an argument.
    - The result of the `is_eog` method is returned directly.
- **Output**: Returns a boolean value indicating whether the specified token is an end-of-generation token.


---
### llama\_vocab\_is\_control<!-- {{#callable:llama_vocab_is_control}} -->
The `llama_vocab_is_control` function checks if a given token is classified as a control token in the specified vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
    - `token`: A `llama_token` representing the token to be checked.
- **Control Flow**:
    - The function calls the `is_control` method of the `vocab` structure, passing the `token` as an argument.
    - The result of the `is_control` method is returned directly as the output of the function.
- **Output**: Returns a boolean value indicating whether the specified token is a control token.


---
### llama\_vocab\_bos<!-- {{#callable:llama_vocab_bos}} -->
The `llama_vocab_bos` function retrieves the beginning-of-sequence (BOS) token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information, including special tokens.
- **Control Flow**:
    - The function directly accesses the `token_bos` method of the `vocab` structure.
    - No conditional logic or loops are present; it simply returns the result of the method call.
- **Output**: Returns a `llama_token` representing the beginning-of-sequence token defined in the vocabulary.


---
### llama\_vocab\_eos<!-- {{#callable:llama_vocab_eos}} -->
The `llama_vocab_eos` function retrieves the end-of-sequence token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
- **Control Flow**:
    - The function directly calls the `token_eos` method on the `vocab` structure.
    - No conditional statements or loops are present in the function.
- **Output**: Returns a `llama_token` representing the end-of-sequence token defined in the vocabulary.


---
### llama\_vocab\_eot<!-- {{#callable:llama_vocab_eot}} -->
The `llama_vocab_eot` function retrieves the end-of-text token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
- **Control Flow**:
    - The function directly accesses the `token_eot` method of the `vocab` structure.
    - No conditional statements or loops are present in the function.
- **Output**: Returns the end-of-text token as a `llama_token`.


---
### llama\_vocab\_cls<!-- {{#callable:llama_vocab_cls}} -->
The `llama_vocab_cls` function retrieves the beginning-of-sequence (BOS) token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information, including special tokens.
- **Control Flow**:
    - The function directly accesses the `token_bos` method of the `vocab` structure.
    - No conditional logic or loops are present in this function.
- **Output**: Returns a `llama_token` representing the beginning-of-sequence (BOS) token.


---
### llama\_vocab\_sep<!-- {{#callable:llama_vocab_sep}} -->
The `llama_vocab_sep` function retrieves the separator token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
- **Control Flow**:
    - The function calls the `token_sep` method on the `vocab` object.
    - The return value of `token_sep` is directly returned as the output of the function.
- **Output**: Returns a `llama_token` representing the separator token defined in the vocabulary.


---
### llama\_vocab\_nl<!-- {{#callable:llama_vocab_nl}} -->
The `llama_vocab_nl` function retrieves the newline token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary data.
- **Control Flow**:
    - The function directly accesses the `token_nl` method of the `vocab` structure.
    - No conditional logic or loops are present in this function.
- **Output**: Returns a `llama_token` representing the newline token defined in the vocabulary.


---
### llama\_vocab\_pad<!-- {{#callable:llama_vocab_pad}} -->
The `llama_vocab_pad` function retrieves the padding token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
- **Control Flow**:
    - The function directly accesses the `token_pad` method of the `vocab` structure.
    - No conditional statements or loops are present in the function.
- **Output**: Returns a `llama_token` representing the padding token defined in the vocabulary.


---
### llama\_vocab\_get\_add\_bos<!-- {{#callable:llama_vocab_get_add_bos}} -->
The `llama_vocab_get_add_bos` function retrieves the boolean value indicating whether the beginning-of-sequence (BOS) token should be added.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
- **Control Flow**:
    - The function calls the `get_add_bos` method on the `vocab` structure.
    - The return value of the `get_add_bos` method is directly returned as the output of the function.
- **Output**: Returns a boolean value indicating whether the BOS token should be added.


---
### llama\_vocab\_get\_add\_eos<!-- {{#callable:llama_vocab_get_add_eos}} -->
The `llama_vocab_get_add_eos` function retrieves the flag indicating whether the end-of-sequence (EOS) token should be added.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
- **Control Flow**:
    - The function calls the `get_add_eos` method on the `vocab` structure.
    - The return value of the `get_add_eos` method is directly returned by the `llama_vocab_get_add_eos` function.
- **Output**: Returns a boolean value indicating whether the EOS token should be added.


---
### llama\_vocab\_fim\_pre<!-- {{#callable:llama_vocab_fim_pre}} -->
The `llama_vocab_fim_pre` function retrieves the 'fim pre' token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information and methods.
- **Control Flow**:
    - The function directly calls the `token_fim_pre` method on the `vocab` structure.
    - No conditional statements or loops are present in the function.
- **Output**: Returns a `llama_token` representing the 'fim pre' token from the vocabulary.


---
### llama\_vocab\_fim\_suf<!-- {{#callable:llama_vocab_fim_suf}} -->
The `llama_vocab_fim_suf` function retrieves the suffix token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
- **Control Flow**:
    - The function directly accesses the `token_fim_suf` method of the `vocab` structure.
    - No conditional logic or loops are present; it simply returns the result of the method call.
- **Output**: Returns a `llama_token` representing the suffix token defined in the vocabulary.


---
### llama\_vocab\_fim\_mid<!-- {{#callable:llama_vocab_fim_mid}} -->
The `llama_vocab_fim_mid` function retrieves the middle FIM token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
- **Control Flow**:
    - The function directly accesses the `token_fim_mid` method of the `vocab` structure.
    - It returns the result of the `token_fim_mid` method call.
- **Output**: The function returns a `llama_token` representing the middle FIM token from the vocabulary.


---
### llama\_vocab\_fim\_pad<!-- {{#callable:llama_vocab_fim_pad}} -->
The `llama_vocab_fim_pad` function retrieves the padding token for the vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
- **Control Flow**:
    - The function directly calls the `token_fim_pad` method on the `vocab` structure.
    - No conditional statements or loops are present in the function.
- **Output**: Returns a `llama_token` representing the padding token from the vocabulary.


---
### llama\_vocab\_fim\_rep<!-- {{#callable:llama_vocab_fim_rep}} -->
The `llama_vocab_fim_rep` function retrieves the final representation token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary data.
- **Control Flow**:
    - The function directly calls the `token_fim_rep` method on the `vocab` structure.
    - No conditional statements or loops are present in the function.
- **Output**: Returns a `llama_token` representing the final representation token from the vocabulary.


---
### llama\_vocab\_fim\_sep<!-- {{#callable:llama_vocab_fim_sep}} -->
The `llama_vocab_fim_sep` function retrieves the file separator token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information and methods.
- **Control Flow**:
    - The function directly calls the `token_fim_sep` method on the `vocab` object.
    - No conditional statements or loops are present in the function.
- **Output**: Returns a `llama_token` representing the file separator token defined in the vocabulary.


---
### llama\_token\_get\_text<!-- {{#callable:llama_token_get_text}} -->
The `llama_token_get_text` function retrieves the text representation of a given token from the vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary data.
    - `token`: A `llama_token` representing the token whose text representation is to be retrieved.
- **Control Flow**:
    - The function calls [`llama_vocab_get_text`](#llama_vocab_get_text) with the provided `vocab` and `token`.
    - The result of the [`llama_vocab_get_text`](#llama_vocab_get_text) function is returned directly.
- **Output**: Returns a pointer to a constant character string representing the text of the specified token.
- **Functions called**:
    - [`llama_vocab_get_text`](#llama_vocab_get_text)


---
### llama\_token\_get\_score<!-- {{#callable:llama_token_get_score}} -->
The `llama_token_get_score` function retrieves the score associated with a given token from the vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary data.
    - `token`: A `llama_token` representing the token whose score is to be retrieved.
- **Control Flow**:
    - The function directly calls [`llama_vocab_get_score`](#llama_vocab_get_score) with the provided `vocab` and `token` arguments.
    - The result from [`llama_vocab_get_score`](#llama_vocab_get_score) is returned as the output of `llama_token_get_score`.
- **Output**: Returns a float value representing the score of the specified token from the vocabulary.
- **Functions called**:
    - [`llama_vocab_get_score`](#llama_vocab_get_score)


---
### llama\_token\_get\_attr<!-- {{#callable:llama_token_get_attr}} -->
The `llama_token_get_attr` function retrieves the attribute of a specified token from a given vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary from which the token attribute is to be retrieved.
    - `token`: A `llama_token` representing the specific token whose attribute is being queried.
- **Control Flow**:
    - The function directly calls [`llama_vocab_get_attr`](#llama_vocab_get_attr) with the provided `vocab` and `token` arguments.
    - The result of the [`llama_vocab_get_attr`](#llama_vocab_get_attr) function is returned as the output of `llama_token_get_attr`.
- **Output**: Returns the attribute of the specified token as an enumeration of type `llama_token_attr`.
- **Functions called**:
    - [`llama_vocab_get_attr`](#llama_vocab_get_attr)


---
### llama\_token\_is\_eog<!-- {{#callable:llama_token_is_eog}} -->
The `llama_token_is_eog` function checks if a given token is an end-of-generation (EOG) token.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
    - `token`: A `llama_token` representing the token to be checked.
- **Control Flow**:
    - The function calls [`llama_vocab_is_eog`](#llama_vocab_is_eog) with the provided `vocab` and `token`.
    - The result of [`llama_vocab_is_eog`](#llama_vocab_is_eog) is returned directly.
- **Output**: Returns a boolean value indicating whether the specified token is an end-of-generation token.
- **Functions called**:
    - [`llama_vocab_is_eog`](#llama_vocab_is_eog)


---
### llama\_token\_is\_control<!-- {{#callable:llama_token_is_control}} -->
The `llama_token_is_control` function checks if a given token is a control token in the specified vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary data.
    - `token`: A `llama_token` representing the token to be checked.
- **Control Flow**:
    - The function calls [`llama_vocab_is_control`](#llama_vocab_is_control) with the provided `vocab` and `token`.
    - The result of the [`llama_vocab_is_control`](#llama_vocab_is_control) function is returned directly.
- **Output**: Returns a boolean value indicating whether the specified token is a control token.
- **Functions called**:
    - [`llama_vocab_is_control`](#llama_vocab_is_control)


---
### llama\_token\_bos<!-- {{#callable:llama_token_bos}} -->
The `llama_token_bos` function retrieves the beginning-of-sequence (BOS) token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary from which the BOS token is to be retrieved.
- **Control Flow**:
    - The function directly calls [`llama_vocab_bos`](#llama_vocab_bos) with the provided `vocab` argument.
    - The result of [`llama_vocab_bos`](#llama_vocab_bos) is returned as the output of `llama_token_bos`.
- **Output**: Returns a `llama_token` representing the beginning-of-sequence token from the vocabulary.
- **Functions called**:
    - [`llama_vocab_bos`](#llama_vocab_bos)


---
### llama\_token\_eos<!-- {{#callable:llama_token_eos}} -->
The `llama_token_eos` function retrieves the end-of-sequence token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary from which the end-of-sequence token is to be retrieved.
- **Control Flow**:
    - The function directly calls [`llama_vocab_eos`](#llama_vocab_eos) with the provided `vocab` argument.
    - There are no conditional statements or loops; the function simply returns the result of the called function.
- **Output**: Returns a `llama_token` representing the end-of-sequence token defined in the vocabulary.
- **Functions called**:
    - [`llama_vocab_eos`](#llama_vocab_eos)


---
### llama\_token\_eot<!-- {{#callable:llama_token_eot}} -->
The `llama_token_eot` function retrieves the end-of-text token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary from which the end-of-text token is retrieved.
- **Control Flow**:
    - The function directly calls [`llama_vocab_eot`](#llama_vocab_eot) with the provided `vocab` argument.
    - No conditional statements or loops are present in the function.
- **Output**: Returns a `llama_token` representing the end-of-text token defined in the vocabulary.
- **Functions called**:
    - [`llama_vocab_eot`](#llama_vocab_eot)


---
### llama\_token\_cls<!-- {{#callable:llama_token_cls}} -->
The `llama_token_cls` function retrieves the beginning-of-sequence (BOS) token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary data.
- **Control Flow**:
    - The function directly calls [`llama_vocab_bos`](#llama_vocab_bos) with the provided `vocab` argument to obtain the BOS token.
    - The commented line indicates a previous implementation that returned a class token instead, but it has been replaced to avoid deprecation warnings.
- **Output**: Returns a `llama_token` representing the beginning-of-sequence token from the vocabulary.
- **Functions called**:
    - [`llama_vocab_bos`](#llama_vocab_bos)


---
### llama\_token\_sep<!-- {{#callable:llama_token_sep}} -->
The `llama_token_sep` function retrieves the separator token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary information.
- **Control Flow**:
    - The function calls [`llama_vocab_sep`](#llama_vocab_sep) with the provided `vocab` argument.
    - The result of [`llama_vocab_sep`](#llama_vocab_sep) is returned directly.
- **Output**: Returns a `llama_token` representing the separator token defined in the vocabulary.
- **Functions called**:
    - [`llama_vocab_sep`](#llama_vocab_sep)


---
### llama\_token\_nl<!-- {{#callable:llama_token_nl}} -->
The `llama_token_nl` function retrieves the newline token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary from which the newline token is to be retrieved.
- **Control Flow**:
    - The function directly calls [`llama_vocab_nl`](#llama_vocab_nl) with the provided `vocab` argument.
    - The result of [`llama_vocab_nl`](#llama_vocab_nl) is returned as the output of `llama_token_nl`.
- **Output**: Returns a `llama_token` representing the newline token defined in the vocabulary.
- **Functions called**:
    - [`llama_vocab_nl`](#llama_vocab_nl)


---
### llama\_token\_pad<!-- {{#callable:llama_token_pad}} -->
The `llama_token_pad` function retrieves the padding token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary information.
- **Control Flow**:
    - The function directly calls [`llama_vocab_pad`](#llama_vocab_pad) with the provided `vocab` argument.
    - No conditional statements or loops are present in this function.
- **Output**: Returns a `llama_token` representing the padding token associated with the given vocabulary.
- **Functions called**:
    - [`llama_vocab_pad`](#llama_vocab_pad)


---
### llama\_add\_bos\_token<!-- {{#callable:llama_add_bos_token}} -->
The `llama_add_bos_token` function checks if a beginning-of-sequence (BOS) token should be added based on the vocabulary settings.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary settings, including whether to add a BOS token.
- **Control Flow**:
    - The function calls [`llama_vocab_get_add_bos`](#llama_vocab_get_add_bos) with the provided `vocab` pointer.
    - The return value of [`llama_vocab_get_add_bos`](#llama_vocab_get_add_bos) indicates whether a BOS token should be added.
- **Output**: Returns a boolean value indicating if a BOS token should be added based on the vocabulary settings.
- **Functions called**:
    - [`llama_vocab_get_add_bos`](#llama_vocab_get_add_bos)


---
### llama\_add\_eos\_token<!-- {{#callable:llama_add_eos_token}} -->
The `llama_add_eos_token` function checks if the end-of-sequence (EOS) token should be added based on the vocabulary settings.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary settings, including whether to add an EOS token.
- **Control Flow**:
    - The function calls [`llama_vocab_get_add_eos`](#llama_vocab_get_add_eos) with the provided `vocab` pointer.
    - The return value of [`llama_vocab_get_add_eos`](#llama_vocab_get_add_eos) determines if the EOS token should be added.
- **Output**: Returns a boolean value indicating whether the EOS token should be added based on the vocabulary settings.
- **Functions called**:
    - [`llama_vocab_get_add_eos`](#llama_vocab_get_add_eos)


---
### llama\_token\_fim\_pre<!-- {{#callable:llama_token_fim_pre}} -->
The `llama_token_fim_pre` function retrieves the 'fim pre' token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary from which the 'fim pre' token is to be retrieved.
- **Control Flow**:
    - The function directly calls [`llama_vocab_fim_pre`](#llama_vocab_fim_pre) with the provided `vocab` argument.
    - No conditional statements or loops are present in this function.
- **Output**: Returns a `llama_token` representing the 'fim pre' token from the vocabulary.
- **Functions called**:
    - [`llama_vocab_fim_pre`](#llama_vocab_fim_pre)


---
### llama\_token\_fim\_suf<!-- {{#callable:llama_token_fim_suf}} -->
The `llama_token_fim_suf` function retrieves the final suffix token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary from which the final suffix token is to be retrieved.
- **Control Flow**:
    - The function directly calls [`llama_vocab_fim_suf`](#llama_vocab_fim_suf) with the provided `vocab` argument.
    - No conditional statements or loops are present in this function.
- **Output**: Returns a `llama_token` representing the final suffix token from the vocabulary.
- **Functions called**:
    - [`llama_vocab_fim_suf`](#llama_vocab_fim_suf)


---
### llama\_token\_fim\_mid<!-- {{#callable:llama_token_fim_mid}} -->
The `llama_token_fim_mid` function retrieves the middle special token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary from which the middle token is to be retrieved.
- **Control Flow**:
    - The function directly calls [`llama_vocab_fim_mid`](#llama_vocab_fim_mid) with the provided `vocab` argument.
    - No additional control flow or logic is present; it simply returns the result of the called function.
- **Output**: Returns a `llama_token` representing the middle special token defined in the vocabulary.
- **Functions called**:
    - [`llama_vocab_fim_mid`](#llama_vocab_fim_mid)


---
### llama\_token\_fim\_pad<!-- {{#callable:llama_token_fim_pad}} -->
The `llama_token_fim_pad` function retrieves the special 'fim pad' token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary from which the 'fim pad' token is to be retrieved.
- **Control Flow**:
    - The function directly calls [`llama_vocab_fim_pad`](#llama_vocab_fim_pad) with the provided `vocab` argument.
    - No conditional logic or loops are present in this function.
- **Output**: Returns a `llama_token` representing the 'fim pad' token from the vocabulary.
- **Functions called**:
    - [`llama_vocab_fim_pad`](#llama_vocab_fim_pad)


---
### llama\_token\_fim\_rep<!-- {{#callable:llama_token_fim_rep}} -->
The `llama_token_fim_rep` function retrieves the 'fim_rep' token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary from which the token is to be retrieved.
- **Control Flow**:
    - The function directly calls [`llama_vocab_fim_rep`](#llama_vocab_fim_rep) with the provided `vocab` argument.
    - No conditional statements or loops are present in the function.
- **Output**: Returns a `llama_token` representing the 'fim_rep' token from the vocabulary.
- **Functions called**:
    - [`llama_vocab_fim_rep`](#llama_vocab_fim_rep)


---
### llama\_token\_fim\_sep<!-- {{#callable:llama_token_fim_sep}} -->
The `llama_token_fim_sep` function retrieves the file separator token from the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary from which the file separator token is to be retrieved.
- **Control Flow**:
    - The function directly calls [`llama_vocab_fim_sep`](#llama_vocab_fim_sep) with the provided `vocab` argument.
    - No conditional logic or loops are present in this function.
- **Output**: Returns a `llama_token` representing the file separator token defined in the vocabulary.
- **Functions called**:
    - [`llama_vocab_fim_sep`](#llama_vocab_fim_sep)


---
### llama\_tokenize<!-- {{#callable:llama_tokenize}} -->
The `llama_tokenize` function tokenizes a given text using a specified vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary used for tokenization.
    - `text`: A pointer to a character array (C-string) containing the text to be tokenized.
    - `text_len`: An integer representing the length of the text to be tokenized.
    - `tokens`: A pointer to an array of `llama_token` where the resulting tokens will be stored.
    - `n_tokens_max`: An integer specifying the maximum number of tokens that can be stored in the `tokens` array.
    - `add_special`: A boolean indicating whether to add special tokens (like start or end of sequence tokens) to the output.
    - `parse_special`: A boolean indicating whether to parse special tokens during tokenization.
- **Control Flow**:
    - The function calls the `tokenize` method of the `vocab` structure, passing all the input parameters.
    - The `tokenize` method processes the text and fills the `tokens` array with the resulting tokens.
    - The function returns the number of tokens generated by the tokenization process.
- **Output**: Returns the number of tokens generated, or a negative value if an error occurs.


---
### llama\_token\_to\_piece<!-- {{#callable:llama_token_to_piece}} -->
Converts a `llama_token` to its corresponding piece representation using the provided vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary and its associated methods.
    - `token`: The `llama_token` that needs to be converted to a piece.
    - `buf`: A character buffer where the resulting piece will be stored.
    - `length`: The maximum length of the buffer to prevent overflow.
    - `lstrip`: The number of leading spaces to strip from the piece before copying.
    - `special`: A boolean indicating whether to treat special tokens differently.
- **Control Flow**:
    - The function first checks if the token is a special token and if the `special` flag is not set; if so, it returns 0.
    - It defines a lambda function `_try_copy` to handle copying the piece to the buffer while respecting the `lstrip` parameter.
    - If a cache exists for the token, it attempts to use it to copy the piece directly to the buffer.
    - If the token is valid, it retrieves the corresponding text from the vocabulary and processes it based on the vocabulary type.
    - For SPM, UGM, and WPM types, it handles whitespace and special attributes accordingly, while for BPE, it decodes the text.
    - Finally, it returns the number of characters copied to the buffer or an error code if the buffer was insufficient.
- **Output**: Returns the number of characters written to the buffer, a negative value if there was an error, or zero if the token is not valid.


---
### llama\_detokenize<!-- {{#callable:llama_detokenize}} -->
The `llama_detokenize` function converts a sequence of tokens back into a human-readable string.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains vocabulary and tokenization information.
    - `tokens`: An array of `llama_token` representing the tokens to be detokenized.
    - `n_tokens`: An integer representing the number of tokens in the `tokens` array.
    - `text`: A character array where the resulting detokenized string will be stored.
    - `text_len_max`: An integer specifying the maximum length of the output string.
    - `remove_special`: A boolean flag indicating whether to remove special tokens from the output.
    - `unparse_special`: A boolean flag indicating whether to unparse special tokens.
- **Control Flow**:
    - The function first checks if the vocabulary type is valid and initialized.
    - It then calls the `detokenize` method of the `vocab` object, passing all the input parameters.
    - The `detokenize` method processes the tokens and constructs the output string based on the specified options.
- **Output**: The function returns the number of characters written to the `text` buffer, or a negative value if an error occurs.


---
### llm\_tokenizer<!-- {{#callable:llm_tokenizer::llm_tokenizer}} -->
The `llm_tokenizer` class serves as a base class for various tokenizer implementations.
- **Inputs**: None
- **Control Flow**:
    - The constructor initializes the `llm_tokenizer` object without any specific operations.
    - The destructor is virtual and defined as default, allowing for proper cleanup of derived classes.
- **Output**: The function does not return any value, as it is a constructor for the `llm_tokenizer` class.


