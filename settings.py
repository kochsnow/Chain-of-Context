import pathlib

from dotenv import load_dotenv
import transformers
import datasets
import os

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()

load_dotenv()  # loads the api keys in .env

MAX_LENGTH_GENERATION = 2500  # todo change this
TEMPERATURE = 0.8
TOP_P = 0.95
# OPENAI_API_ENV_KEYS = ['OPENAI_API_KEY', 'OPENAI_API_KEY2', 'OPENAI_API_KEY3', 'OPENAI_API_KEY4', 'OPENAI_API_KEY5',
#                        'OPENAI_API_KEY6']
OPENAI_API_ENV_KEYS = ['OPENAI_API_KEY0', 'OPENAI_API_KEY00', 'OPENAI_API_KEY2', 'OPENAI_API_KEY3', 'OPENAI_API_KEY4', 'OPENAI_API_KEY5', 'OPENAI_API_KEY6',
                       'OPENAI_API_KEY7']

MODEL = 'gpt-4-0613'  # 'gpt-3.5-turbo-16k-0613'  # 'gpt-3.5-turbo'  # hard coded model here
ALLOW_CODE_EXECUTION = True

PROJECT_DIR = (os.path.dirname(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs2")
CACHE_DIR = os.path.join(PROJECT_DIR, ".cache")
RECENT_QUERY_DIR = os.path.join(PROJECT_DIR, "recent_query")

pathlib.Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(RECENT_QUERY_DIR).mkdir(parents=True, exist_ok=True)

SYSTEM_CHAT_INSTRUCTION = """
You are a helpful assistant that carefully follows instructions.
You should complete the user text, continuing from the example format, rather than providing a conversational response.
"""

CONTEXT_PASS_N_SAMPLES = 3
TRANSLATION_PASS_N_SAMPLES = 1

RECENT_QUERY_COUNTER = 0
def get_next_recent_query_counter():
    global RECENT_QUERY_COUNTER
    RECENT_QUERY_COUNTER += 1
    return RECENT_QUERY_COUNTER
