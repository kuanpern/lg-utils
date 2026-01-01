import os
import glob
import inspect
from typing import Dict, Any, Optional, Callable

def recusvie_load_files(
    file_extensions: list[str],
    parser: Optional[Callable[[str], Any]] = None,
    root_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Recursively loads files with given extensions into a nested dictionary.
    Optionally parses file content using a custom parser (e.g., YAML, Jinja2, JSON).

    Args:
        file_extensions: List of file extensions to search for (e.g., [".yaml", ".jinja2"]).
        parser: Function to parse file content (e.g., `yaml.safe_load` for YAML).
                If `None`, reads files as plain text.
        root_dir: Root directory to search. Defaults to the script's directory.

    Returns:
        Nested dictionary where keys are path segments and values are parsed file contents.
    """
    # Set root directory (default: script's directory)
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    # Find all matching files
    files = []
    for ext in file_extensions:
        files.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))

    # Initialize output dictionary
    file_registry = {}

    for file in files:
        # Get relative path and remove extension
        relpath = os.path.relpath(file, root_dir)
        relpath_without_ext = os.path.splitext(relpath)[0]
        parts = relpath_without_ext.split(os.sep)

        # Traverse/create nested dictionary
        current_level = file_registry
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

        # Read and optionally parse the file
        with open(file, "r") as f:
            content = f.read()
            current_level[parts[-1]] = parser(content) if parser else content

    return file_registry
# end def


import inspect
from typing import Dict, Any, Callable, Tuple, Optional, Union, List

def get_function_arguments(
    func: Callable
) -> Dict[str, Dict[str, Union[type, Any, None]]]:
    """
    Retrieve the valid arguments of a target function, including their names,
    default values, and type annotations (if available).

    Args:
        func (Callable): The target function to inspect.

    Returns:
        Dict[str, Dict[str, Union[type, Any, None]]]:
            A dictionary where keys are argument names, and values are dictionaries
            containing:
            - 'type': The type annotation (if available), otherwise `None`.
            - 'default': The default value (if available), otherwise `inspect.Parameter.empty`.
            - 'kind': The kind of parameter (e.g., positional, keyword, var-positional, var-keyword).

    Example:
        >>> def example(a: int, b: str = "hello", *args, **kwargs): pass
        >>> get_function_arguments(example)
        {
            'a': {'type': <class 'int'>, 'default': <empty>, 'kind': 'POSITIONAL_OR_KEYWORD'},
            'b': {'type': <class 'str'>, 'default': 'hello', 'kind': 'POSITIONAL_OR_KEYWORD'},
            'args': {'type': None, 'default': <empty>, 'kind': 'VAR_POSITIONAL'},
            'kwargs': {'type': None, 'default': <empty>, 'kind': 'VAR_KEYWORD'}
        }
    """
    signature = inspect.signature(func)
    arguments = {}

    for name, param in signature.parameters.items():
        arguments[name] = {
            'type': param.annotation if param.annotation != inspect.Parameter.empty else None,
            'default': param.default if param.default != inspect.Parameter.empty else None,
            'kind': param.kind.name.lower()
        }

    return arguments
# end def

def get_nested_value(data, keys, default_value=None):
    """Get value from nested dict
    
    Args:
        data (dict): The input dictionary.
        keys (list): A list of keys to access the nested value.
        default_value (any): The default value to return if the key is not found.

    Returns:
        any: The nested value or the default value.
    """
    for key in keys:
        if key in data:
            data = data[key]
        else:
            return default_value
        # end if
    # end for
    return data
# end def

def get_nested_attribute(obj, attribute_path, default_value=None):
    """Get value from nested object attribute    """
    parts = attribute_path.split('.')
    current_obj = obj
    for part in parts:
        if hasattr(current_obj, part):
            current_obj = getattr(current_obj, part)
        else:
            return default_value
    return current_obj
# end def
