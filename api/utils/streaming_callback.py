from langchain_core.callbacks.base import AsyncCallbackHandler

class StreamingCallback(AsyncCallbackHandler):
    """
    Callback handler to handle token by token streaming
    """
    def __init__(self, publish, stage: str):
        self.publish = publish
        self.stage = stage

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.publish(f"debug_{self.stage}_token", token)

    async def on_llm_end(self, response, **kwargs):
        await self.publish(f"debug_{self.stage}_end", "stream_completed")
