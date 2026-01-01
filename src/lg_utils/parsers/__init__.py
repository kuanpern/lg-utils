import json
import yaml
from typing import Type, TypeVar, List
import traceback

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import Runnable
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# Basic yaml content extractor. Keep for reference. We will use the version from yaml_utils
def extract_yaml_content(text: str) -> str:
    """
    STUB: Replace this with your existing YAML extraction function.
    
    Current logic: 
    1. Tries to find ```yaml content ``` blocks.
    2. Fallback: returns the whole text.
    """
    import re
    pattern = re.compile(r"```(?:yaml)?\n(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()

# New implementation of yaml content extractor
from yaml_utils import YAMLExtractor
extract_yaml_content = YAMLExtractor()


class YamlPydanticParser(BaseOutputParser[T]):
    pydantic_model: Type[T]

    def parse(self, text: str) -> T:
        # Extract content
        try:
            # Parse YAML string to Dict
            data = extract_yaml_content(text)
        except yaml.YAMLError as e:
            raise OutputParserException(f"Invalid YAML Syntax: {e}")

        try:
            # Validate Dict against Pydantic
            return self.pydantic_model.model_validate(data)
        except ValidationError as e:
            # Raising OutputParserException is crucial for .with_retry() 
            raise OutputParserException(f"Schema Validation Failed: {e}")

    @property
    def _type(self) -> str:
        return "yaml_pydantic_parser"


# def _prepend_schema_instruction(input_data: Union[str, List[BaseMessage]], schema_json: str) -> List[BaseMessage]:
#     """
#     Helper to inject the Schema instructions into the message history
#     before sending to the LLM.
#     """
#     system_instruction = (
#         "Output strictly in YAML format.\n"
#         "Wrap content in ```yaml ... ``` code blocks.\n"
#         f"Conform to the following schema:\n{schema_json}"
#     )
#     
#     sys_msg = SystemMessage(content=system_instruction)
# 
#     if isinstance(input_data, str):
#         return [sys_msg, HumanMessage(content=input_data)]
#     
#     if isinstance(input_data, list):
#         # We prepend to the existing list of messages
#         return [sys_msg] + input_data
#     
#     raise ValueError("Input must be string or list of messages")


def with_custom_structured_output(self, schema: Type[BaseModel], **kwargs) -> Runnable:
    """
    This method is a wrapper around the `with_structured_output` method,
    but it uses a custom YAML parser instead of the default JSON parser.
    """
    
    # Create the Parser Instance
    parser = YamlPydanticParser(pydantic_model=schema)

    # Create the Chain
    chain = ( self | parser )

    return chain


# Usage example
if __name__ == '__main__':
    import os
    from typing import List
    from pydantic import Field
    from langchain.chat_models import init_chat_model

    # Apply the patch
    BaseChatModel.with_custom_structured_output = with_custom_structured_output

    # 1. Define Schemas
    class Ingredient(BaseModel):
        name: str
        quantity: str
        is_allergen: bool = False

    class Recipe(BaseModel):
        title: str = Field(description="Name of the dish")
        steps: List[str] = Field(description="Step by step guide")
        ingredients: List[Ingredient]
        difficulty: int = Field(description="1 to 10 scale")

    schema_json = json.dumps(Recipe.model_json_schema(), indent=2)

    # 2. Initialize Model
    assert os.getenv('GOOGLE_API_KEY', os.getenv('GEMINI_API_KEY')) is not None, "env var GOOGLE_API_KEY not set"
    llm = init_chat_model(model="gemini-2.0-flash", model_provider="google-genai")

    # 3. Define Chain
    chain = llm.with_custom_structured_output(Recipe).with_retry(stop_after_attempt=3)

    # 4. Invoke
    user_query = f"Give me a dangerous spicy pasta recipe with peanuts. \
    Return in the following schema using YAML (*not* JSON): {schema_json}"
    
    try:
        # This single call handles:
        # 1. Getting YAML from LLM
        # 2. Extracting & Validating
        # 3. Retrying if validation fails
        result = chain.invoke(user_query)
        
        print(f"Recipe: {result.title}")
        print(f"First Ingredient: {result.ingredients[0]}")
        print(type(result)) # <class '__main__.Recipe'>

    except Exception as e:
        # traceback
        traceback.print_exc()