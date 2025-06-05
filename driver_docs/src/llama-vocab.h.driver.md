# Purpose
This C++ header file defines the `llama_vocab` class, which is a comprehensive component for handling vocabulary and tokenization processes in a language model context. The class provides a broad range of functionalities related to token management, including loading vocabulary data, converting between text and tokens, and handling special token types. It includes methods for tokenization and detokenization, allowing for the conversion of raw text into tokens and vice versa. The class also supports various token attributes and types, such as normal, unknown, control, and user-defined tokens, and provides methods to retrieve token data, such as text, score, and attributes.

The `llama_vocab` class is designed to be a central part of a language model's vocabulary management system, likely intended for use in natural language processing applications. It includes methods for handling byte-pair encoding (BPE) merges, managing special tokens like beginning-of-sequence (BOS) and end-of-sequence (EOS), and configuring tokenization behavior with options like adding space prefixes or cleaning spaces. The class uses a private implementation pattern with a `std::unique_ptr` to an `impl` struct, suggesting an encapsulated design that separates interface from implementation details. This file is intended to be included in other parts of a software project, providing a public API for vocabulary and tokenization operations.
# Imports and Dependencies

---
- `llama.h`
- `string`
- `vector`
- `memory`


# Data Structures

---
### llama\_vocab<!-- {{#data_structure:llama_vocab}} -->
- **Type**: `struct`
- **Members**:
    - `token_data`: A nested struct within llama_vocab that holds information about a token, including its text, score, and attributes.
    - `pimpl`: A unique pointer to an implementation struct, used to encapsulate the internal details of llama_vocab.
- **Description**: The `llama_vocab` struct is a comprehensive data structure designed to manage and manipulate a vocabulary of tokens for a language model. It includes a nested `token_data` struct to store individual token information such as text, score, and attributes. The struct provides a wide range of methods for loading vocabulary data, tokenizing and detokenizing text, and accessing token properties. It uses the Pimpl idiom to hide implementation details, ensuring a clean interface and encapsulation. The `llama_vocab` is equipped to handle various token types and special tokens, making it a versatile component in language processing applications.
- **Member Functions**:
    - [`llama_vocab::llama_vocab`](llama-vocab.cpp.driver.md#llama_vocabllama_vocab)
    - [`llama_vocab::~llama_vocab`](llama-vocab.cpp.driver.md#llama_vocabllama_vocab)
    - [`llama_vocab::load`](llama-vocab.cpp.driver.md#llama_vocabload)
    - [`llama_vocab::get_tokenizer_model`](llama-vocab.cpp.driver.md#llama_vocabget_tokenizer_model)
    - [`llama_vocab::get_tokenizer_pre`](llama-vocab.cpp.driver.md#llama_vocabget_tokenizer_pre)
    - [`llama_vocab::get_type`](llama-vocab.cpp.driver.md#llama_vocabget_type)
    - [`llama_vocab::get_pre_type`](llama-vocab.cpp.driver.md#llama_vocabget_pre_type)
    - [`llama_vocab::n_tokens`](llama-vocab.cpp.driver.md#llama_vocabn_tokens)
    - [`llama_vocab::n_token_types`](llama-vocab.cpp.driver.md#llama_vocabn_token_types)
    - [`llama_vocab::type_name`](llama-vocab.cpp.driver.md#llama_vocabtype_name)
    - [`llama_vocab::is_normal`](llama-vocab.cpp.driver.md#llama_vocabis_normal)
    - [`llama_vocab::is_unknown`](llama-vocab.cpp.driver.md#llama_vocabis_unknown)
    - [`llama_vocab::is_control`](llama-vocab.cpp.driver.md#llama_vocabis_control)
    - [`llama_vocab::is_byte`](llama-vocab.cpp.driver.md#llama_vocabis_byte)
    - [`llama_vocab::is_user_defined`](llama-vocab.cpp.driver.md#llama_vocabis_user_defined)
    - [`llama_vocab::is_unused`](llama-vocab.cpp.driver.md#llama_vocabis_unused)
    - [`llama_vocab::is_eog`](llama-vocab.cpp.driver.md#llama_vocabis_eog)
    - [`llama_vocab::token_to_byte`](llama-vocab.cpp.driver.md#llama_vocabtoken_to_byte)
    - [`llama_vocab::byte_to_token`](llama-vocab.cpp.driver.md#llama_vocabbyte_to_token)
    - [`llama_vocab::text_to_token`](llama-vocab.cpp.driver.md#llama_vocabtext_to_token)
    - [`llama_vocab::get_token_data`](llama-vocab.cpp.driver.md#llama_vocabget_token_data)
    - [`llama_vocab::token_get_text`](llama-vocab.cpp.driver.md#llama_vocabtoken_get_text)
    - [`llama_vocab::token_get_score`](llama-vocab.cpp.driver.md#llama_vocabtoken_get_score)
    - [`llama_vocab::token_get_attr`](llama-vocab.cpp.driver.md#llama_vocabtoken_get_attr)
    - [`llama_vocab::token_bos`](llama-vocab.cpp.driver.md#llama_vocabtoken_bos)
    - [`llama_vocab::token_eos`](llama-vocab.cpp.driver.md#llama_vocabtoken_eos)
    - [`llama_vocab::token_eot`](llama-vocab.cpp.driver.md#llama_vocabtoken_eot)
    - [`llama_vocab::token_eom`](llama-vocab.cpp.driver.md#llama_vocabtoken_eom)
    - [`llama_vocab::token_unk`](llama-vocab.cpp.driver.md#llama_vocabtoken_unk)
    - [`llama_vocab::token_sep`](llama-vocab.cpp.driver.md#llama_vocabtoken_sep)
    - [`llama_vocab::token_nl`](llama-vocab.cpp.driver.md#llama_vocabtoken_nl)
    - [`llama_vocab::token_pad`](llama-vocab.cpp.driver.md#llama_vocabtoken_pad)
    - [`llama_vocab::token_prefix`](llama-vocab.cpp.driver.md#llama_vocabtoken_prefix)
    - [`llama_vocab::token_middle`](llama-vocab.cpp.driver.md#llama_vocabtoken_middle)
    - [`llama_vocab::token_suffix`](llama-vocab.cpp.driver.md#llama_vocabtoken_suffix)
    - [`llama_vocab::token_fim_pre`](llama-vocab.cpp.driver.md#llama_vocabtoken_fim_pre)
    - [`llama_vocab::token_fim_suf`](llama-vocab.cpp.driver.md#llama_vocabtoken_fim_suf)
    - [`llama_vocab::token_fim_mid`](llama-vocab.cpp.driver.md#llama_vocabtoken_fim_mid)
    - [`llama_vocab::token_fim_pad`](llama-vocab.cpp.driver.md#llama_vocabtoken_fim_pad)
    - [`llama_vocab::token_fim_rep`](llama-vocab.cpp.driver.md#llama_vocabtoken_fim_rep)
    - [`llama_vocab::token_fim_sep`](llama-vocab.cpp.driver.md#llama_vocabtoken_fim_sep)
    - [`llama_vocab::get_add_space_prefix`](llama-vocab.cpp.driver.md#llama_vocabget_add_space_prefix)
    - [`llama_vocab::get_add_bos`](llama-vocab.cpp.driver.md#llama_vocabget_add_bos)
    - [`llama_vocab::get_add_eos`](llama-vocab.cpp.driver.md#llama_vocabget_add_eos)
    - [`llama_vocab::get_ignore_merges`](llama-vocab.cpp.driver.md#llama_vocabget_ignore_merges)
    - [`llama_vocab::get_clean_spaces`](llama-vocab.cpp.driver.md#llama_vocabget_clean_spaces)
    - [`llama_vocab::get_remove_extra_whitespaces`](llama-vocab.cpp.driver.md#llama_vocabget_remove_extra_whitespaces)
    - [`llama_vocab::get_escape_whitespaces`](llama-vocab.cpp.driver.md#llama_vocabget_escape_whitespaces)
    - [`llama_vocab::get_treat_whitespace_as_suffix`](llama-vocab.cpp.driver.md#llama_vocabget_treat_whitespace_as_suffix)
    - [`llama_vocab::max_token_len`](llama-vocab.cpp.driver.md#llama_vocabmax_token_len)
    - [`llama_vocab::find_bpe_rank`](llama-vocab.cpp.driver.md#llama_vocabfind_bpe_rank)
    - [`llama_vocab::get_bpe_merges`](llama-vocab.cpp.driver.md#llama_vocabget_bpe_merges)
    - [`llama_vocab::get_precompiled_charsmap`](llama-vocab.cpp.driver.md#llama_vocabget_precompiled_charsmap)
    - [`llama_vocab::tokenize`](llama-vocab.cpp.driver.md#llama_vocabtokenize)
    - [`llama_vocab::tokenize`](llama-vocab.cpp.driver.md#llama_vocabtokenize)
    - [`llama_vocab::token_to_piece`](llama-vocab.cpp.driver.md#llama_vocabtoken_to_piece)
    - [`llama_vocab::token_to_piece`](llama-vocab.cpp.driver.md#llama_vocabtoken_to_piece)
    - [`llama_vocab::detokenize`](llama-vocab.cpp.driver.md#llama_vocabdetokenize)
    - [`llama_vocab::detokenize`](llama-vocab.cpp.driver.md#llama_vocabdetokenize)
    - [`llama_vocab::print_info`](llama-vocab.cpp.driver.md#llama_vocabprint_info)


---
### token\_data<!-- {{#data_structure:llama_vocab::token_data}} -->
- **Type**: `struct`
- **Members**:
    - `text`: A string representing the text of the token.
    - `score`: A floating-point value representing the score or weight of the token.
    - `attr`: An attribute of type llama_token_attr associated with the token.
- **Description**: The `token_data` struct is a component of the `llama_vocab` structure, designed to encapsulate information about individual tokens within a vocabulary. It contains three members: `text`, which holds the string representation of the token; `score`, a float that may represent the token's relevance or frequency; and `attr`, which is an attribute of type `llama_token_attr` that provides additional metadata about the token. This struct is used to manage and access token-specific data efficiently within the larger context of the vocabulary management system.


