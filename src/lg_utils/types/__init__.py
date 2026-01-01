
import logging
import jinja2
from typing import Dict, Any, Callable, Optional
from langchain.messages import HumanMessage, SystemMessage
from tenacity import (
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    Retrying
)

class StructuredAgent:
    def __init__(
        self, 
        name: str,
        instruction: str, 
        llm: Optional[Any] = None, 
        model_name: Optional[str] = "google_genai:gemini-2.0-flash",
        description: Optional[str] = None, 
        prompt_defaults: Optional[dict] = None,
        post_processor: Optional[Callable] = None, 
        retry_configs: Optional[dict] = None,
        post_process_retry_configs: Optional[dict] = None, 
        jinja2_env: Optional[jinja2.Environment] = None, 
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.name = name

        if llm is None:
            from langchain.chat_models import init_chat_model
            llm = init_chat_model(model_name)
            self.llm = llm
        
        # Setup Jinja2
        self.jinja2_env = jinja2_env or jinja2.Environment(
            loader=jinja2.BaseLoader,
            undefined=jinja2.StrictUndefined,
            autoescape=False
        )
        
        # Pre-compile templates for performance
        self.instruction_tmpl = self.jinja2_env.from_string(instruction)
        self.system_tmpl = self.jinja2_env.from_string(description) if description else None
        
        self.prompt_defaults = prompt_defaults or {}
        self.post_processor = post_processor or (lambda x: {"output": x})

        # Retry Configuration
        default_retry = {'max_attempts': 3, 'wait_base': 0.2, 'wait_min': 0, 'wait_max': 30}
        self.retry_configs = {**default_retry, **(retry_configs or {})}
        self.pp_retry_configs = {**default_retry, **(post_process_retry_configs or {})}

        self._llm_retryer = self._build_retryer(self.retry_configs, Exception, self._log_retry_attempt)
        self._pp_retryer = self._build_retryer(
            self.pp_retry_configs, 
            (ValueError, KeyError, TypeError, RuntimeError), 
            self._log_post_process_retry
        )

    def _build_retryer(self, configs, exc_types, hook):
        return Retrying(
            stop=stop_after_attempt(configs['max_attempts']),
            wait=wait_exponential(exp_base=configs['wait_base'], min=configs['wait_min'], max=configs['wait_max']),
            retry=retry_if_exception_type(exc_types),
            before=hook,
            reraise=True
        )

    def _log_retry_attempt(self, retry_state):
        if retry_state.attempt_number > 1:
            self.logger.warning(f"[{self.name}] LLM attempt {retry_state.attempt_number} failed. Retrying...")

    def _log_post_process_retry(self, retry_state):
        if retry_state.attempt_number > 1:
            self.logger.warning(f"[{self.name}] Logic error/hallucination. Re-invoking LLM (Attempt {retry_state.attempt_number})")

    def _invoke_llm(self, input_messages):
        # We use the retryer as a context manager
        for attempt in self._llm_retryer:
            with attempt:
                response = self.llm.invoke(input_messages)
                return response.content

    def __call__(self, state: Dict[str, Any], runtime: Any = None) -> Dict[str, Any]:
        # 1. Prepare context
        runtime_var = {}
        if runtime and hasattr(runtime, 'context'):
            # More robust way to get attributes
            runtime_var = {k: v for k, v in vars(runtime.context).items() if not k.startswith('_') and isinstance(v, (str, int, float, bool))}
        
        payloads = {**self.prompt_defaults, **state, **runtime_var}
        
        # 2. Render Messages
        input_messages = []
        if self.system_tmpl:
            input_messages.append(SystemMessage(content=self.system_tmpl.render(**payloads)))
        input_messages.append(HumanMessage(content=self.instruction_tmpl.render(**payloads)))

        # 3. Execute with Post-Processing Retries
        try:
            for attempt in self._pp_retryer:
                with attempt:
                    raw_content = self._invoke_llm(input_messages)
                    return self.post_processor(raw_content)
        except Exception as e:
            self.logger.error(f"Processor {self.name} failed after all retries. Last error: {e}")
            raise