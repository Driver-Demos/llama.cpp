# Purpose
This Python script is designed to manage and manipulate model checkpoints, specifically for models using PyTorch and SafeTensors formats. It provides functionality to load and save model files, identify and extract specific components from model checkpoints, and clean up model files by removing certain components, such as the "vision tower" tensors. The script is structured to handle both PyTorch and SafeTensors formats seamlessly, using helper functions to determine the file type and perform the appropriate loading and saving operations. The script also includes command-line interface (CLI) capabilities, allowing users to specify model paths and options for cleaning vision tower components directly from the command line.

The script's primary focus is on processing model checkpoints related to the LLaVA (Large Language and Vision Assistant) model, as indicated by the references to "llava.clip" and "llava.projector" files. It identifies and extracts tensors related to vision towers, newline components, and multi-modal projectors, storing them separately for further use. The script is intended to be run as a standalone tool, providing a streamlined process for managing model components and preparing them for conversion to other formats, such as LLaMA GGUF files. The use of argparse for CLI setup and the integration of file handling functions make this script a versatile tool for model checkpoint management in machine learning workflows.
# Imports and Dependencies

---
- `argparse`
- `glob`
- `os`
- `torch`
- `safetensors.safe_open`
- `safetensors.torch.save_file`
- `typing.Any`
- `typing.ContextManager`
- `typing.cast`


# Global Variables

---
### ap
- **Type**: `argparse.ArgumentParser`
- **Description**: The variable `ap` is an instance of `argparse.ArgumentParser`, which is used to handle command-line arguments in Python scripts. It is configured to accept a required argument for the model path and an optional flag to clean the vision tower from model files.
- **Use**: This variable is used to parse and manage command-line arguments for the script, enabling users to specify the model path and optional actions.


---
### args
- **Type**: `argparse.Namespace`
- **Description**: The `args` variable is an instance of `argparse.Namespace` that holds the parsed command-line arguments. It is created by calling `ap.parse_args()`, where `ap` is an `ArgumentParser` object configured to accept specific command-line options.
- **Use**: This variable is used to access the command-line arguments provided by the user, such as the model path and the clean-vision-tower flag.


---
### model\_files
- **Type**: `list`
- **Description**: The `model_files` variable is a list that contains the paths to all files within the directory specified by `args.model`. These files are sorted in descending order based on their last modification time. The sorting is achieved using the `sorted` function with a key that utilizes `os.path.getmtime` to determine the modification time of each file.
- **Use**: This variable is used to identify and order model files for further processing, such as cleaning vision tower components or extracting specific tensors.


---
### checkpoint\_paths
- **Type**: `list`
- **Description**: The `checkpoint_paths` variable is a list that contains file paths to model checkpoint files. These paths are filtered from a list of model files, specifically including files that end with '.bin' and contain 'pytorch' in their name, or files that end with '.safetensors' and contain 'model' in their name.
- **Use**: This variable is used to store paths to relevant model checkpoint files for further processing, such as cleaning vision tower tensors or extracting specific components.


---
### newline\_checkpoint\_path
- **Type**: `Optional[str]`
- **Description**: The `newline_checkpoint_path` variable is a global variable that stores the file path of a checkpoint that meets the criteria defined by the `newline_criteria` function. It is determined by iterating over a list of checkpoint paths and checking each one against the `newline_criteria`. If a checkpoint satisfies the criteria and `newline_checkpoint_path` is not already set, it is assigned the path of that checkpoint.
- **Use**: This variable is used to store the path of a checkpoint that contains specific components identified by the `newline_criteria` function.


---
### projector\_checkpoint\_path
- **Type**: `str or None`
- **Description**: The `projector_checkpoint_path` is a global variable that stores the file path of the checkpoint containing the multi-modal projector component of a model. It is determined by the `find_relevant_checkpoints` function, which searches through a list of checkpoint paths to find one that meets the criteria for containing a multi-modal projector.
- **Use**: This variable is used to identify and load the specific checkpoint file that contains the multi-modal projector component for further processing or extraction.


---
### first\_mm\_tensors
- **Type**: `list`
- **Description**: The variable `first_mm_tensors` is a list that is initially empty. It is intended to store the names of tensors that meet certain criteria, specifically those identified as 'newline' tensors from a checkpoint file.
- **Use**: This variable is used to collect and store the names of 'newline' tensors extracted from the first checkpoint file that meets the newline criteria.


---
### first\_checkpoint
- **Type**: `NoneType or dict`
- **Description**: The variable `first_checkpoint` is initially set to `None` and is later assigned a dictionary containing model tensors if a valid `newline_checkpoint_path` is found. This dictionary is populated by loading a model checkpoint file that meets certain criteria.
- **Use**: `first_checkpoint` is used to store the loaded model tensors from the first checkpoint that contains newline components, which are later used to extract specific tensors for further processing.


---
### mm\_tensors
- **Type**: `list`
- **Description**: The `mm_tensors` variable is a global list that is used to store the names of tensors related to the multi-modal projector component of a model. It is populated by extracting relevant tensor names from the last checkpoint file that contains the multi-modal projector component. This list is used to identify and process specific tensors for further operations, such as saving them into a separate file.
- **Use**: This variable is used to store and manage the names of multi-modal projector tensors extracted from a model checkpoint.


---
### last\_checkpoint
- **Type**: `NoneType`
- **Description**: The variable `last_checkpoint` is initialized to `None` and is intended to store the last loaded model checkpoint if a valid projector checkpoint path is found. It is used to hold the data structure returned by the `load_model` function, which loads model data from a specified file path.
- **Use**: This variable is used to store the last model checkpoint data when a projector checkpoint path is identified, allowing further processing of the model's tensors.


---
### projector
- **Type**: `dict`
- **Description**: The `projector` variable is a dictionary that stores tensors related to the multi-modal projector component of a model. It is populated by iterating over the `mm_tensors` and `first_mm_tensors` lists, which contain the names of tensors identified as part of the multi-modal projector from the last and first checkpoints, respectively. Each tensor is converted to a float type before being added to the `projector` dictionary.
- **Use**: This variable is used to store and save the multi-modal projector tensors extracted from model checkpoints.


# Functions

---
### is\_safetensor\_file<!-- {{#callable:llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.is_safetensor_file}} -->
The function `is_safetensor_file` checks if a given file path ends with the '.safetensors' extension.
- **Inputs**:
    - `file_path`: A string representing the path to the file that needs to be checked.
- **Control Flow**:
    - The function uses the `str.endswith` method to check if the `file_path` ends with the '.safetensors' extension.
- **Output**: A boolean value indicating whether the file path ends with '.safetensors'.


---
### load\_model<!-- {{#callable:llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.load_model}} -->
The `load_model` function loads a model from a file, determining whether it is a SafeTensor or PyTorch file, and returns the model data along with its type.
- **Inputs**:
    - `file_path`: The path to the file from which the model should be loaded.
- **Control Flow**:
    - Check if the file is a SafeTensor file using the [`is_safetensor_file`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2is_safetensor_file) function.
    - If it is a SafeTensor file, open the file using `safe_open` and iterate over its keys to clone each tensor into a dictionary, printing the shape of each tensor.
    - Return the dictionary of tensors and the string 'safetensor'.
    - If it is not a SafeTensor file, load the model using `torch.load` with the map location set to CPU.
    - Return the loaded model and the string 'pytorch'.
- **Output**: A tuple containing the loaded model data (either a dictionary of tensors or a PyTorch model) and a string indicating the type of file ('safetensor' or 'pytorch').
- **Functions called**:
    - [`llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.is_safetensor_file`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2is_safetensor_file)


---
### save\_model<!-- {{#callable:llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.save_model}} -->
The `save_model` function saves a given model to a specified file path using either the SafeTensors or PyTorch format based on the provided file type.
- **Inputs**:
    - `model`: The model object to be saved.
    - `file_path`: The file path where the model should be saved.
    - `file_type`: A string indicating the format to save the model in, either 'safetensor' or another format (assumed to be PyTorch).
- **Control Flow**:
    - Check if the `file_type` is 'safetensor'.
    - If `file_type` is 'safetensor', use the `save_file` function from the `safetensors.torch` module to save the model.
    - If `file_type` is not 'safetensor', use the `torch.save` function to save the model in PyTorch format.
- **Output**: The function does not return any value; it performs a side effect of saving the model to the specified file path.


---
### is\_vision\_tower<!-- {{#callable:llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.is_vision_tower}} -->
The `is_vision_tower` function checks if a given weight name corresponds to a vision tower component by examining its prefix.
- **Inputs**:
    - `weight_name`: A string representing the name of a weight, which is checked to determine if it belongs to a vision tower component.
- **Control Flow**:
    - The function checks if the input string `weight_name` starts with any of the specified prefixes: 'model.vision_tower', 'vit.', or 'vision_tower'.
    - It returns `True` if any of these conditions are met, otherwise it returns `False`.
- **Output**: A boolean value indicating whether the `weight_name` starts with any of the specified vision tower prefixes.


---
### is\_newline<!-- {{#callable:llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.is_newline}} -->
The `is_newline` function checks if a given weight name starts with specific prefixes related to 'image_newline'.
- **Inputs**:
    - `weight_name`: A string representing the name of a weight to be checked for specific prefixes.
- **Control Flow**:
    - The function evaluates if the input string `weight_name` starts with either 'model.image_newline' or 'image_newline'.
- **Output**: A boolean value indicating whether the `weight_name` starts with the specified prefixes.


---
### is\_mm\_projector<!-- {{#callable:llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.is_mm_projector}} -->
The function `is_mm_projector` checks if a given weight name corresponds to a multi-modal projector component by evaluating its prefix.
- **Inputs**:
    - `weight_name`: A string representing the name of a weight, which is checked to determine if it belongs to a multi-modal projector component.
- **Control Flow**:
    - The function checks if the input `weight_name` starts with the prefix 'model.mm_projector'.
    - If the first condition is not met, it checks if `weight_name` starts with the prefix 'vision_proj.'.
    - If neither of the first two conditions are met, it checks if `weight_name` starts with the prefix 'multi_modal_projector'.
    - The function returns `True` if any of the above conditions are satisfied, otherwise it returns `False`.
- **Output**: A boolean value indicating whether the `weight_name` matches any of the specified prefixes for a multi-modal projector.


---
### newline\_criteria<!-- {{#callable:llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.newline_criteria}} -->
The `newline_criteria` function checks if any key in a given checkpoint dictionary starts with 'model.image_newline' or 'image_newline'.
- **Inputs**:
    - `checkpoint`: A dictionary representing a model checkpoint, where keys are weight names and values are the corresponding tensors.
- **Control Flow**:
    - The function uses a generator expression to iterate over the keys of the checkpoint dictionary.
    - For each key, it calls the [`is_newline`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2is_newline) function to check if the key starts with 'model.image_newline' or 'image_newline'.
    - The `any` function evaluates the generator expression and returns True if any key satisfies the [`is_newline`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2is_newline) condition, otherwise it returns False.
- **Output**: A boolean value indicating whether any key in the checkpoint dictionary matches the newline criteria.
- **Functions called**:
    - [`llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.is_newline`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2is_newline)


---
### proj\_criteria<!-- {{#callable:llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.proj_criteria}} -->
The `proj_criteria` function checks if any key in a given checkpoint dictionary corresponds to a multi-modal projector component.
- **Inputs**:
    - `checkpoint`: A dictionary representing a model checkpoint, where keys are weight names.
- **Control Flow**:
    - The function iterates over all keys in the `checkpoint` dictionary.
    - For each key, it checks if the key matches the criteria defined by the [`is_mm_projector`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2is_mm_projector) function.
    - The function returns `True` if any key satisfies the [`is_mm_projector`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2is_mm_projector) condition, otherwise it returns `False`.
- **Output**: A boolean value indicating whether any key in the checkpoint matches the multi-modal projector criteria.
- **Functions called**:
    - [`llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.is_mm_projector`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2is_mm_projector)


---
### clean\_vision\_tower\_from\_checkpoint<!-- {{#callable:llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.clean_vision_tower_from_checkpoint}} -->
The function `clean_vision_tower_from_checkpoint` extracts and saves vision tower tensors from a model checkpoint to a separate file, removing them from the original checkpoint.
- **Inputs**:
    - `checkpoint_path`: The file path to the model checkpoint from which vision tower tensors are to be extracted.
- **Control Flow**:
    - Load the model checkpoint from the given path using the [`load_model`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2load_model) function, which determines the file type (either 'pytorch' or 'safetensor').
    - Determine the directory path of the model from the checkpoint path.
    - Identify tensors related to the vision tower by filtering keys in the checkpoint using the [`is_vision_tower`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2is_vision_tower) function.
    - If vision tower tensors are found, construct the path for the 'llava.clip' file in the same directory as the checkpoint.
    - Check if 'llava.clip' already exists; if it does, load its contents, otherwise initialize an empty dictionary for it.
    - Iterate over the identified vision tower tensors, simplify their names, and add them to the 'llava.clip' dictionary if they are not already present.
    - Save the updated 'llava.clip' dictionary back to the file using the [`save_model`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2save_model) function.
    - Remove the vision tower tensors from the original checkpoint.
    - Return `True` if any vision tower tensors were found and processed, otherwise return `False`.
- **Output**: The function returns a boolean value: `True` if vision tower tensors were found and processed, `False` otherwise.
- **Functions called**:
    - [`llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.load_model`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2load_model)
    - [`llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.is_vision_tower`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2is_vision_tower)
    - [`llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.save_model`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2save_model)


---
### find\_relevant\_checkpoints<!-- {{#callable:llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.find_relevant_checkpoints}} -->
The function `find_relevant_checkpoints` identifies and returns the paths of checkpoints that meet specific newline and projector criteria from a list of checkpoint paths.
- **Inputs**:
    - `checkpoint_paths`: A list of file paths to checkpoint files that need to be evaluated.
    - [`newline_criteria`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2newline_criteria): A function that takes a checkpoint and returns a boolean indicating if the checkpoint meets the newline criteria.
    - `projector`: A function that takes a checkpoint and returns a boolean indicating if the checkpoint meets the projector criteria.
- **Control Flow**:
    - Initialize `newline_checkpoint_path` and `projector_checkpoint_path` to `None`.
    - Iterate over each path in `checkpoint_paths`.
    - For each path, load the checkpoint using [`load_model`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2load_model).
    - Check if the checkpoint meets the [`newline_criteria`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2newline_criteria) and if `newline_checkpoint_path` is `None`; if so, set `newline_checkpoint_path` to the current path.
    - Check if the checkpoint meets the `projector` criteria; if so, set `projector_checkpoint_path` to the current path.
    - Return a tuple containing `newline_checkpoint_path` and `projector_checkpoint_path`.
- **Output**: A tuple containing two elements: the path of the first checkpoint that meets the newline criteria and the path of the last checkpoint that meets the projector criteria.
- **Functions called**:
    - [`llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.load_model`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2load_model)
    - [`llama.cpp/tools/mtmd/legacy-models/llava_surgery_v2.newline_criteria`](#cpp/tools/mtmd/legacy-models/llava_surgery_v2newline_criteria)


