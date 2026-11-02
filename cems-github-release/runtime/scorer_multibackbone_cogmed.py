"""Preserve frozen COGMED prompts; explicitly disable Qwen3 thinking at the API."""
import hashlib,os
from pathlib import Path
_source=Path(__file__).with_name('scorer_remote_paper_20261018.py')
_bytes=_source.read_bytes()
if hashlib.sha256(_bytes).hexdigest()!='5f8fc73975fe5e865779aee53f7befb6860d4dc27c634039e28bf708ae2e21a7':
    raise RuntimeError('Frozen scorer source changed')
exec(compile(_bytes,str(_source),'exec'),globals())
_base_client=OpenAICompatibleClient
class OpenAICompatibleClient(_base_client):
    def __init__(self,base_url,model,api_key='EMPTY',timeout=240,chat_template_kwargs=None):
        if chat_template_kwargs is None and os.environ.get('MODEL_DISABLE_THINKING')=='1':
            chat_template_kwargs={'enable_thinking':False}
        super().__init__(base_url,model,api_key,timeout,chat_template_kwargs)
