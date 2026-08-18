from langchain.chat_models import init_chat_model

class ChatModelManager:
    def __init__(self, configs):
        self._chat_models = {}
        self._configs = configs
        self._supported_models = list(configs.keys())

    def init_chat_model(self, model_name: str, **kwargs):
        """
        Initializes and caches LangChain chat models.

        Args:
            model_name (str): The name of the model to retrieve.
            **kwargs: Additional keyword arguments to pass to init_chat_model.

        Returns:
            langchain.chat_models.base.BaseChatModel: An initialized chat model.
        """
        chat_model = self._chat_models.get(model_name)
        if chat_model is not None:
            return chat_model

        assert model_name in self._supported_models, f"{model_name} is not supported."
        default_kwargs = dict(self._configs.get(model_name, {}))
        combined_kwargs = {"model": "default", **default_kwargs, **kwargs}

        base_urls = combined_kwargs.pop("base_url", None)
        if isinstance(base_urls, list):
            if not base_urls:
                raise ValueError("base_url list cannot be empty")

            models = []
            for url in base_urls:
                model_kwargs = {**combined_kwargs, "base_url": url}
                models.append(init_chat_model(**model_kwargs))

            primary_model = models[0]
            if len(models) > 1:
                chat_model = primary_model.with_fallbacks(models[1:])
            else:
                chat_model = primary_model
        else:
            if base_urls is not None:
                combined_kwargs["base_url"] = base_urls
            chat_model = init_chat_model(**combined_kwargs)

        self._chat_models[model_name] = chat_model
        return chat_model
