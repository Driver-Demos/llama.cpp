# Purpose
This C++ header file provides a narrow functionality focused on handling chat templates for a language model (LLM) system. It defines an enumeration `llm_chat_template` that lists various chat template identifiers, which likely correspond to different configurations or styles of chat interactions. The file also declares a forward struct `llama_chat_message` and three functions: `llm_chat_template_from_str`, which converts a string to a corresponding chat template enum; `llm_chat_detect_template`, which likely identifies the template type from a given string; and `llm_chat_apply_template`, which applies a specified chat template to a collection of chat messages, modifying a destination string and possibly adding additional information. This header is intended to be included in other C++ source files, providing a structured way to manage and apply chat templates within a larger application.
# Imports and Dependencies

---
- `string`
- `vector`
- `cstdint`


# Data Structures

---
### llm\_chat\_template<!-- {{#data_structure:llm_chat_template}} -->
- **Type**: `enum`
- **Members**:
    - `LLM_CHAT_TEMPLATE_CHATML`: Represents the CHATML chat template.
    - `LLM_CHAT_TEMPLATE_LLAMA_2`: Represents the LLAMA 2 chat template.
    - `LLM_CHAT_TEMPLATE_LLAMA_2_SYS`: Represents the LLAMA 2 SYS chat template.
    - `LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS`: Represents the LLAMA 2 SYS BOS chat template.
    - `LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP`: Represents the LLAMA 2 SYS STRIP chat template.
    - `LLM_CHAT_TEMPLATE_MISTRAL_V1`: Represents the MISTRAL V1 chat template.
    - `LLM_CHAT_TEMPLATE_MISTRAL_V3`: Represents the MISTRAL V3 chat template.
    - `LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN`: Represents the MISTRAL V3 TEKKEN chat template.
    - `LLM_CHAT_TEMPLATE_MISTRAL_V7`: Represents the MISTRAL V7 chat template.
    - `LLM_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN`: Represents the MISTRAL V7 TEKKEN chat template.
    - `LLM_CHAT_TEMPLATE_PHI_3`: Represents the PHI 3 chat template.
    - `LLM_CHAT_TEMPLATE_PHI_4`: Represents the PHI 4 chat template.
    - `LLM_CHAT_TEMPLATE_FALCON_3`: Represents the FALCON 3 chat template.
    - `LLM_CHAT_TEMPLATE_ZEPHYR`: Represents the ZEPHYR chat template.
    - `LLM_CHAT_TEMPLATE_MONARCH`: Represents the MONARCH chat template.
    - `LLM_CHAT_TEMPLATE_GEMMA`: Represents the GEMMA chat template.
    - `LLM_CHAT_TEMPLATE_ORION`: Represents the ORION chat template.
    - `LLM_CHAT_TEMPLATE_OPENCHAT`: Represents the OPENCHAT chat template.
    - `LLM_CHAT_TEMPLATE_VICUNA`: Represents the VICUNA chat template.
    - `LLM_CHAT_TEMPLATE_VICUNA_ORCA`: Represents the VICUNA ORCA chat template.
    - `LLM_CHAT_TEMPLATE_DEEPSEEK`: Represents the DEEPSEEK chat template.
    - `LLM_CHAT_TEMPLATE_DEEPSEEK_2`: Represents the DEEPSEEK 2 chat template.
    - `LLM_CHAT_TEMPLATE_DEEPSEEK_3`: Represents the DEEPSEEK 3 chat template.
    - `LLM_CHAT_TEMPLATE_COMMAND_R`: Represents the COMMAND R chat template.
    - `LLM_CHAT_TEMPLATE_LLAMA_3`: Represents the LLAMA 3 chat template.
    - `LLM_CHAT_TEMPLATE_CHATGLM_3`: Represents the CHATGLM 3 chat template.
    - `LLM_CHAT_TEMPLATE_CHATGLM_4`: Represents the CHATGLM 4 chat template.
    - `LLM_CHAT_TEMPLATE_GLMEDGE`: Represents the GLMEDGE chat template.
    - `LLM_CHAT_TEMPLATE_MINICPM`: Represents the MINICPM chat template.
    - `LLM_CHAT_TEMPLATE_EXAONE_3`: Represents the EXAONE 3 chat template.
    - `LLM_CHAT_TEMPLATE_RWKV_WORLD`: Represents the RWKV WORLD chat template.
    - `LLM_CHAT_TEMPLATE_GRANITE`: Represents the GRANITE chat template.
    - `LLM_CHAT_TEMPLATE_GIGACHAT`: Represents the GIGACHAT chat template.
    - `LLM_CHAT_TEMPLATE_MEGREZ`: Represents the MEGREZ chat template.
    - `LLM_CHAT_TEMPLATE_YANDEX`: Represents the YANDEX chat template.
    - `LLM_CHAT_TEMPLATE_BAILING`: Represents the BAILING chat template.
    - `LLM_CHAT_TEMPLATE_LLAMA4`: Represents the LLAMA4 chat template.
    - `LLM_CHAT_TEMPLATE_SMOLVLM`: Represents the SMOLVLM chat template.
    - `LLM_CHAT_TEMPLATE_UNKNOWN`: Represents an unknown chat template.
- **Description**: The `llm_chat_template` enum defines a set of constants representing various chat templates used in a chat application or system. Each enumerator corresponds to a specific chat template, which may be associated with different versions or configurations of chat models, such as LLAMA, MISTRAL, PHI, FALCON, and others. This enum is likely used to identify and apply specific chat templates within the system, facilitating the customization and management of chat interactions.


