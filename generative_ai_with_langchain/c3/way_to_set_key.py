import os

OPENAI_API_KEY = ''

def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or "ID" in key:
            os.environ[key] = value

# when you need to set key, just
# import set_environment from config
# set_environment()