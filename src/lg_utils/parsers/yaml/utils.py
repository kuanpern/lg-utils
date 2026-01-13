import re
import yaml
from typing import List, Dict, Any, Optional
import logging
from markdown import Markdown
from io import StringIO

logger = logging.getLogger(__name__)

# --- YAML Parsing Functions ---
def parse_multiline_yaml(yaml_string: str) -> Optional[Dict[str, Any]]:
    """
    Parses a multiline YAML string with improved error handling.
    Args:
        yaml_string: The YAML input string.
    Returns:
        The parsed YAML as a Python dict, or None if parsing fails.
    """
    try:
        data = yaml.safe_load(yaml_string)
        if isinstance(data, dict):
            return data
        logger.warning(f"Warning: YAML segment is not a dictionary (got {type(data).__name__}).")
        return None
    except (yaml.scanner.ScannerError, yaml.parser.ParserError) as e:
        # Combine error logging for YAML parsing issues
        error_message = f"YAML parsing error"
        if hasattr(e, 'problem_mark') and e.problem_mark:
            error_message += f" at line {e.problem_mark.line}, column {e.problem_mark.column}"
        logger.warning(f"{error_message}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing YAML: {e}")
        return None

def extract_yaml_segments(text: str, remove_markdown: bool = False) -> List[Dict[str, Any]]:
    """
    Extracts all valid YAML segments from a given text string.
    Handles YAML in markdown-style code blocks and standalone YAML.
    """
    # Pattern looks for newlines around the content.
    pattern_in_backticks = r'```(?:yaml|YAML)?\n((?:(?!\n```)[\s\S])*?)\n```'
    
    potential_segments = []
    
    # 1. Extract YAML from backticked blocks
    matches_in_backticks = list(re.finditer(pattern_in_backticks, text, re.DOTALL))
    for match in matches_in_backticks:
        segment_content = match.group(1).strip()
        if segment_content:
            potential_segments.append(segment_content)

    # 2. Remove backticked blocks from text to avoid re-matching standalone YAML
    # Iterate in reverse to avoid index issues after removal
    text_stripped = text
    for match in reversed(matches_in_backticks):
        text_stripped = text_stripped[:match.start()] + text_stripped[match.end():]

    # 3. Extract standalone YAML (split by double newline)
    standalone_blocks = re.split(r'\n\s*\n', text_stripped)
    for block in standalone_blocks:
        stripped_block = block.strip()
        # Basic heuristic: ignore blocks that don't look like key-value pairs to reduce noise
        if stripped_block and ":" in stripped_block:
            potential_segments.append(stripped_block)

    valid_segments = []
    # FIX: Removed 'seen_segments' deduplication logic.
    # If the user has two identical config blocks, we want to preserve order
    # so that strategy="last" works correctly.
    for segment in potential_segments:
        processed_segment = segment
        if remove_markdown:
            processed_segment = unmark(processed_segment)
        
        parsed_data = parse_multiline_yaml(processed_segment)
        if parsed_data is not None:
            valid_segments.append(parsed_data)
            
    return valid_segments

# --- Markdown Unmarking Functions ---
def _walk_element(element, stream: StringIO):
    """Recursively writes element's text content to a stream."""
    if element.text:
        stream.write(element.text)
    for sub_element in element:
        _walk_element(sub_element, stream)
    if element.tail:
        stream.write(element.tail)

def _unmark_element(element):
    """Convert a single element to plain text."""
    stream = StringIO()
    _walk_element(element, stream)
    return stream.getvalue()

# Patch Markdown for plain text output
Markdown.output_formats["plain"] = _unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False

def unmark(text: str) -> str:
    """Converts markdown text to plain text."""
    return __md.convert(text)

class YAMLExtractor:
    """Base class for extracting and filtering YAML data."""
    def __init__(self, mandatory_keys: Optional[List[str]] = None, strategy: str = "last"):
        """
        Args:
            mandatory_keys: A list of keys that must be present in a YAML document
                            for it to be considered valid.
            strategy: Determines which YAML document to return if multiple are found
                      and are valid. Can be "first" or "last".
        """
        self.mandatory_keys = mandatory_keys or []
        if strategy not in ["first", "last"]:
            raise ValueError("Strategy must be 'first' or 'last'")
        self.strategy = strategy

    def extract_from_text(self, text: str, exclude_non_mandatory: bool = False) -> List[Dict[str, Any]]:
        """Extracts YAML segments and filters by mandatory keys."""
        candidates = extract_yaml_segments(text)
        
        if not self.mandatory_keys:
            return candidates
        
        # FIX: The validation logic has been corrected.
        # Previously, it raised an error if *any* block was missing keys, regardless of the 'exclude' flag.
        
        if exclude_non_mandatory:
            # Filter mode: remove candidates that don't match, keep the valid ones.
            candidates = [
                candidate for candidate in candidates
                if all(key in candidate for key in self.mandatory_keys)
            ]
        else:
            # Strict mode: All extracted YAML segments MUST contain the mandatory keys.
            # If we found YAML blocks but they are incomplete, this is an error.
            for i, candidate in enumerate(candidates):
                missing_keys = [key for key in self.mandatory_keys if key not in candidate]
                if missing_keys:
                    raise ValueError(f"YAML segment {i+1} is missing mandatory keys: {missing_keys}")
        
        return candidates

    def process(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Processes text, extracts YAML, and returns the selected document
        based on the defined strategy.
        """
        candidates = self.extract_from_text(text.strip(), **kwargs)
        
        if not candidates:
            # This error is raised if no YAML was found, or if all found YAML 
            # was filtered out because of missing keys.
            raise ValueError(
                "No valid YAML found matching requirements. "
                f"Required keys: {self.mandatory_keys}"
            )
            
        return candidates[0] if self.strategy == "first" else candidates[-1]

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        return self.process(*args, **kwargs)