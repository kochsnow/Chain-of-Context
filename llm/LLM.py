import os

import openai
import random
import hashlib
import time

from diskcache import Cache
from warnings import warn
from abc import ABC, abstractmethod
from eval.args import OAIArguments, GenerationArguments
from settings import SYSTEM_CHAT_INSTRUCTION, OPENAI_API_ENV_KEYS, CACHE_DIR, TOP_P, MAX_LENGTH_GENERATION, TEMPERATURE


class BaseLLM(ABC):
    def __init__(self):
        pass

    def __call__(self, prompt, stop):
        return self.get_completion(prompt, stop)

    @abstractmethod
    def get_completion(self, prompt, stop):
        pass


class OAILLM(BaseLLM):
    def __init__(self,
                 chat,
                 model,
                 n_samples,
                 cache_dir=CACHE_DIR,
                 openai_api_env_keys=OPENAI_API_ENV_KEYS,
                 chat_system_instruction=SYSTEM_CHAT_INSTRUCTION,
                 top_p: float=TOP_P,
                 max_length_generation=MAX_LENGTH_GENERATION,
                 temperature=TEMPERATURE
                 ):
        super(OAILLM, self).__init__()

        self.chat = chat
        self.model = model
        self.api_keys = [os.environ[key] for key in openai_api_env_keys]
        self.cache = Cache(cache_dir)

        self.n_samples = n_samples
        self.chat_system_instruction = chat_system_instruction
        self.temperature = temperature
        self.max_length_generation = max_length_generation
        self.top_p = top_p

    def make_request(self, prompt, stop):
        if self.chat:
            response = openai.ChatCompletion.create(
                model=self.model,
                n=self.n_samples,
                messages=[
                    {"role": "system", "content": self.chat_system_instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_length_generation,
                top_p=self.top_p,
                stop=stop,
                stream=False,
            )
        else:
            response = openai.Completion.create(
                engine=self.model,
                n=self.n_samples,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_length_generation,
                top_p=self.top_p,
                stop=stop,
                stream=False,
            )
        return response

    def get_completion(self, prompt, stop, api_key=None, exhausted={}, retry_after=60):
        if self.temperature == 0:
            request_id = "_".join(
                str(x)
                for x in [
                    self.model,
                    self.n_samples,
                    prompt,
                    self.max_length_generation,
                    stop,
                ]
            )
            request_key = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
            if request_key in self.cache:
                print(
                    "Identical OpenAI API request previously executed. Loading response from cache."
                )
                return self.cache[request_key]
        if api_key is None:
            api_key = random.choice(self.api_keys)
        openai.api_key = api_key
        try:
            response = self.make_request(prompt, stop)
        except openai.error.RateLimitError:
            if len(self.api_keys) == 1:
                warn(
                    f"Only one API key was provided, and it has been rate limited. sleeping for {retry_after}s. Please provide more API keys to avoid sleeping."
                )
                time.sleep(retry_after)
                return self.get_completion(prompt, stop, api_key)
            else:
                print(f"Rate limit error; trying again with a different API key.")
                exhausted[api_key] = time.time()
                exhausted = {
                    k: v
                    for k, v in exhausted.items()
                    if (time.time() - v) < retry_after
                }
                if len(exhausted) == len(self.api_keys):
                    print(
                        f"All API keys have been exhausted. sleeping for {retry_after}s then trying again with all keys."
                    )
                    time.sleep(retry_after)
                    exhausted = {}
                try_next = random.choice(
                    [k for k in self.api_keys if k != api_key and k not in exhausted]
                )
                return self.get_completion(prompt, stop, try_next, exhausted)
        except (openai.error.Timeout, openai.error.APIError, openai.error.ServiceUnavailableError) as e:
            print(f"API Error; sleeping for {retry_after}s then trying again.")
            time.sleep(retry_after)
            return self.get_completion(prompt, stop, api_key, exhausted)
        if self.chat:
            response = [c["message"]["content"] for c in response["choices"]]
        else:
            response = [c["text"] for c in response["choices"]]
        if self.temperature == 0:
            print("Temperature is 0, caching OpenAI API response for future use.")
            self.cache[request_key] = response
        return response
