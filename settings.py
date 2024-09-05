OPENAI_API_KEY = ''
SERPAPI_API_KEY = ''
LANGCHAIN_API_KEY = ''
HUGGINGFACEHUB_API_TOKEN= ''
ZEP_API_URL = ''
ZEP_API_KEY = ''
WOLFRAM_ALPHA_APPID = ''

from local_settings import *


def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or "ID" in key:
            os.environ[key] = value
    return


def get_zep_config() -> dict:
    return {
        'url': ZEP_API_URL,
        'api_key': ZEP_API_KEY,
    }

