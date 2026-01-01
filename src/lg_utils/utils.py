import logging
from transformers import PreTrainedTokenizer

# --- Text Truncation Function ---
def truncate_text_to_token_limit(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
    ellipsis: str = " ...",
    verbose: bool = False,
    logger: logging.Logger = None

) -> str:
    """
    Truncates input text to fit within a specified token limit using a tokenizer.
    Preserves as much meaningful content as possible while respecting the token constraint.

    Args:
        text: Input text to truncate.
        tokenizer: HuggingFace tokenizer to measure token count.
        max_tokens: Maximum allowed tokens in the output.
        ellipsis: String appended to indicate truncation (default: " ...").
        verbose: If True, prints truncation progress (default: False).
        logger: logger

    Returns:
        Truncated text with ellipsis appended, guaranteed to be <= max_tokens.

    Raises:
        ValueError: If max_tokens is non-positive or text is empty after processing.
    """

    if logging is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    # end if

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    
    stripped_text = text.strip()
    if not stripped_text:
        return stripped_text

    # Initial token count to check if truncation is needed
    try:
        initial_tokens = tokenizer(stripped_text, return_tensors="pt")['input_ids']
    except Exception as e:
        logger.error(f"Tokenizer error: {e}")
        return stripped_text # Return original if tokenizer fails

    if initial_tokens.shape[-1] <= max_tokens:
        return stripped_text

    # Reduce text proportionally based on token count
    words = stripped_text.split()
    if not words:
        return stripped_text # Should not happen if stripped_text is not empty

    # Estimate words to keep, ensuring at least one word remains.
    # Calculate target word count based on ratio of max_tokens to initial_tokens.
    # Ensure target_word_count is at least 1 to prevent empty output.
    estimated_target_words = max(1, int(len(words) * max_tokens / initial_tokens.shape[-1]))
    words_to_keep = words[:estimated_target_words]

    # Iteratively refine by removing words until token limit is met
    current_text = " ".join(words_to_keep)
    while len(tokenizer(current_text, return_tensors="pt")['input_ids']) > max_tokens:
        if len(words_to_keep) <= 1: # Cannot truncate further
            break
        words_to_keep.pop() # Remove the last word
        current_text = " ".join(words_to_keep)
        if verbose:
            logger.info(f"Truncating... Remaining words: {len(words_to_keep)}")

    # Add ellipsis if truncation occurred and there's space
    final_text_with_ellipsis = current_text
    if len(words) > len(words_to_keep): # If actual truncation happened
        try:
            # Check if ellipsis fits
            ellipsis_tokens = len(tokenizer(ellipsis, return_tensors="pt")['input_ids'])
            if len(tokenizer(current_text, return_tensors="pt")['input_ids']) + ellipsis_tokens <= max_tokens:
                final_text_with_ellipsis += ellipsis
            else:
                # If ellipsis doesn't fit, try to truncate current_text more to make space
                while len(tokenizer(final_text_with_ellipsis, return_tensors="pt")['input_ids']) + ellipsis_tokens > max_tokens:
                    if len(words_to_keep) <= 1:
                        break # Cannot make space for ellipsis
                    words_to_keep.pop()
                    final_text_with_ellipsis = " ".join(words_to_keep)
                if len(final_text_with_ellipsis) > 0: # Only add if there's text left
                    final_text_with_ellipsis += ellipsis

        except Exception as e:
            logger.error(f"Tokenizer error while adding ellipsis: {e}")
            # If ellipsis addition fails, return the truncated text without it.

    # Ensure final output adheres to max_tokens. This is a safeguard.
    # If the logic above somehow fails, this re-truncates to the absolute limit.
    final_tokens = tokenizer(final_text_with_ellipsis, return_tensors="pt")['input_ids']
    if final_tokens.shape[-1] > max_tokens:
        return tokenizer.decode(final_tokens[0, :max_tokens], skip_special_tokens=True)
        
    return final_text_with_ellipsis
