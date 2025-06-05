# Purpose
This code file defines a set of TypeScript interfaces and types that are used to model a conversation system with branching capabilities. The primary focus of the file is to provide a structured way to represent messages and conversations, allowing for features such as editing past messages and creating new branches in a conversation history. The `Message` interface is central to this system, encapsulating details such as the message's ID, type, timestamp, role, content, and its relationship to other messages through parent and children nodes. This node-based structure facilitates the branching feature, where editing a message can create a new branch while preserving the original conversation flow.

The file also defines several auxiliary interfaces and types to support additional functionalities. The `TimingReport` interface is used to capture timing information related to message prompts and predictions. The `MessageExtra` types allow for the inclusion of additional content types such as text files, image files, audio files, and contextual information within a message. The `APIMessage` type is designed to represent messages in a format suitable for API interactions, supporting various content types like text, image URLs, and audio inputs.

Furthermore, the file includes interfaces for managing conversations and viewing chat sessions. The `Conversation` interface tracks the conversation's ID, last modification timestamp, current message node, and name. The `ViewingChat` interface provides a read-only view of a conversation and its associated messages. Additionally, the file defines types related to a Python interpreter canvas and server properties for a LlamaCpp server, indicating potential integration with external systems or services. Overall, this code file serves as a comprehensive schema for managing and interacting with a branching conversation system.
# Data Structures

---
### TimingReport
- **Type**: `interface`
- **Members**:
    - `prompt_n`: The number of prompts processed.
    - `prompt_ms`: The time taken in milliseconds to process the prompts.
    - `predicted_n`: The number of predictions made.
    - `predicted_ms`: The time taken in milliseconds to make the predictions.
- **Description**: The `TimingReport` interface is a data structure that captures timing metrics related to the processing of prompts and predictions. It includes fields to record the number of prompts and predictions, as well as the time taken in milliseconds for each of these operations. This structure is useful for performance monitoring and analysis in systems that handle conversational data or similar tasks.


---
### Message
- **Type**: `interface`
- **Members**:
    - `id`: A unique identifier for the message.
    - `convId`: The conversation ID to which the message belongs.
    - `type`: Specifies if the message is 'text' or 'root'.
    - `timestamp`: The time when the message was created, represented as a timestamp.
    - `role`: Indicates the role of the message sender, such as 'user', 'assistant', or 'system'.
    - `content`: The actual content of the message.
    - `timings`: Optional timing report associated with the message.
    - `extra`: Optional array of additional message data, such as files or context.
    - `parent`: The ID of the parent message in the conversation tree.
    - `children`: An array of IDs representing the child messages of this message.
- **Description**: The `Message` interface represents a node in a conversation tree, allowing for conversation branching by maintaining parent-child relationships between messages. Each message has a unique ID, belongs to a specific conversation, and can be of type 'text' or 'root'. It records the timestamp of its creation and the role of the sender. The content of the message is stored as a string, and it may include optional timing information and additional data such as files or context. The node-based structure facilitates editing and branching of conversations, where each message can have a parent and multiple children, enabling complex conversation flows.


---
### MessageExtraTextFile
- **Type**: `interface`
- **Members**:
    - `type`: Specifies the type of the extra as 'textFile'.
    - `name`: The name of the text file.
    - `content`: The content of the text file.
- **Description**: The `MessageExtraTextFile` interface is a data structure used to represent additional text file information associated with a message in a conversation. It includes the type of the extra, which is always 'textFile', the name of the text file, and the content of the text file. This structure is part of a larger system that supports conversation branching, allowing users to attach text files to messages within a node-based chat interface.


---
### MessageExtraImageFile
- **Type**: `interface`
- **Members**:
    - `type`: Specifies the type of the extra as 'imageFile'.
    - `name`: The name of the image file.
    - `base64Url`: A base64 encoded URL representing the image file.
- **Description**: The `MessageExtraImageFile` interface is a data structure used to represent an image file associated with a message in a chat application. It includes a type identifier to specify that it is an image file, a name for the file, and a base64 encoded URL that contains the image data. This structure is part of a larger system that supports various types of message extras, allowing for the inclusion of multimedia content in conversations.


---
### MessageExtraAudioFile
- **Type**: `interface`
- **Members**:
    - `type`: Specifies the type of the extra file as 'audioFile'.
    - `name`: The name of the audio file.
    - `base64Data`: The audio file data encoded in base64 format.
    - `mimeType`: The MIME type of the audio file, indicating the format.
- **Description**: The `MessageExtraAudioFile` interface is a data structure used to represent an audio file associated with a message in a conversation. It includes fields for specifying the type of the file, the name of the file, the audio data encoded in base64, and the MIME type of the audio file. This structure is part of a larger system that supports various types of message extras, allowing for the inclusion of audio content in a conversation's message history.


---
### MessageExtraContext
- **Type**: `interface`
- **Members**:
    - `type`: Specifies the type of the context, which is always 'context'.
    - `name`: The name of the context.
    - `content`: The content of the context, represented as a string.
- **Description**: The `MessageExtraContext` interface is a part of the `MessageExtra` type, which represents additional context information associated with a message in a conversation. It is used to store contextual data that can be attached to a message, allowing for richer interactions and more detailed message metadata. This interface includes fields for specifying the type of the context, its name, and the actual content, all of which are strings. The `type` field is fixed to 'context', indicating that this structure is specifically for contextual information.


---
### APIMessageContentPart
- **Type**: `TypeScript Union Type`
- **Members**:
    - `type`: Specifies the type of content, which can be 'text', 'image_url', or 'input_audio'.
    - `text`: Contains the text content when the type is 'text'.
    - `image_url`: Holds an object with a URL string when the type is 'image_url'.
    - `input_audio`: Contains an object with audio data and format when the type is 'input_audio'.
- **Description**: The `APIMessageContentPart` is a TypeScript union type that represents different parts of a message's content in an API. It can be one of three types: 'text', which includes a simple text string; 'image_url', which includes an object with a URL pointing to an image; or 'input_audio', which includes an object containing audio data and its format (either 'wav' or 'mp3'). This structure allows for flexible message content handling, accommodating various media types within a single message framework.


---
### APIMessage
- **Type**: `type`
- **Members**:
    - `role`: Specifies the role of the message, which can be 'user', 'assistant', or 'system'.
    - `content`: Contains the message content, which can be a string or an array of APIMessageContentPart objects.
- **Description**: The APIMessage data structure represents a message in a conversation, encapsulating the role of the sender and the content of the message. The content can be a simple string or a more complex structure consisting of various content parts, such as text, image URLs, or audio inputs. This flexibility allows the APIMessage to support diverse types of communication within a conversation, accommodating different media formats and interaction types.


---
### Conversation
- **Type**: `interface`
- **Members**:
    - `id`: A unique identifier for the conversation, formatted as `conv-{timestamp}`.
    - `lastModified`: A timestamp indicating the last modification time of the conversation.
    - `currNode`: The ID of the current message node being viewed in the conversation.
    - `name`: The name of the conversation.
- **Description**: The `Conversation` interface represents a structured chat session, identified by a unique ID and a name, with a timestamp indicating its last modification. It tracks the current message node being viewed, allowing for dynamic conversation branching and editing, similar to features seen in advanced chat applications like ChatGPT. This structure supports a node-based system where each message can have parent and child nodes, facilitating the creation of conversation branches when messages are edited.


---
### ViewingChat
- **Type**: `interface`
- **Members**:
    - `conv`: A read-only Conversation object representing the current conversation being viewed.
    - `messages`: A read-only array of Message objects representing the messages in the current conversation.
- **Description**: The ViewingChat interface is a data structure that encapsulates the state of a chat view, including the current conversation and its associated messages. It provides a read-only view of the Conversation object, which contains metadata about the conversation, and an array of Message objects, which represent the individual messages within the conversation. This structure is designed to support features like conversation branching, where messages can have parent and child relationships, allowing for complex conversation flows.


---
### PendingMessage
- **Type**: `TypeScript `type``
- **Members**:
    - `id`: A unique identifier for the message.
    - `convId`: The conversation ID to which the message belongs.
    - `type`: Indicates whether the message is 'text' or 'root'.
    - `timestamp`: The time when the message was created, represented as a timestamp.
    - `role`: The role of the message sender, which can be 'user', 'assistant', or 'system'.
    - `timings`: Optional timing report associated with the message.
    - `extra`: Optional array of additional message data, such as files or context.
    - `parent`: The ID of the parent message in the conversation tree.
    - `children`: An array of IDs representing the child messages in the conversation tree.
    - `content`: The content of the message, which can be a string or null.
- **Description**: The `PendingMessage` type is a specialized version of the `Message` interface, used to represent a message that is in the process of being sent or edited. It includes all the properties of a `Message` except for the `content`, which can be either a string or null, indicating that the message content is not yet finalized. This type is useful in scenarios where a message is being composed or modified, allowing for the representation of incomplete message data within a conversation's branching structure.


---
### CanvasPyInterpreter
- **Type**: `interface`
- **Members**:
    - `type`: Specifies the type of the canvas, which is set to CanvasType.PY_INTERPRETER.
    - `content`: Holds the content of the Python interpreter canvas as a string.
- **Description**: The CanvasPyInterpreter is an interface that represents a specific type of canvas used for Python interpretation within a chat application. It is characterized by a type field, which is set to CanvasType.PY_INTERPRETER, indicating its purpose as a Python interpreter canvas. The content field stores the actual content or code that is to be interpreted or displayed within this canvas. This structure is part of a larger system that supports conversation branching and message handling, allowing users to interact with Python code in a conversational context.


---
### LlamaCppServerProps
- **Type**: `interface`
- **Members**:
    - `build_info`: A string containing information about the build.
    - `model_path`: A string specifying the path to the model.
    - `n_ctx`: A number representing the context size.
    - `modalities`: An optional object indicating support for vision and audio modalities.
- **Description**: The `LlamaCppServerProps` interface defines the properties required for configuring a LlamaCpp server instance. It includes essential information such as the build details, model path, and context size, along with optional support for different modalities like vision and audio. This interface is crucial for setting up and managing the server's operational parameters.


