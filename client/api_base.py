import time
from typing import Optional, List, Union, Type, Dict, Any
from abc import ABCMeta, abstractmethod

import requests
import os
from dotenv import load_dotenv

from typing import List, Dict
from openai import OpenAI, OpenAIError

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class ApiConcreteBase(metaclass=ABCMeta):
    def __init__(
            self,
            chat_model,
    ) -> None:
        self._chat_model = chat_model
        self._exception_wait_sec = 5

    def __call__(
            self,
            input: Dict[str, str],
            **kwargs,
    ):
        return self.invoke(input, **kwargs)

    @abstractmethod
    def _invoke_handling(
            self,
            input: Dict[str, str],
            **kwargs,
    ):
        raise NotImplementedError

    def invoke(
            self,
            input: Dict[str, str],
            director_config: Optional[Dict] = None,
            **kwargs,
    ):
        try:
            return self._invoke_handling(input, **kwargs)
        except requests.exceptions.Timeout as e:
            logger.warning(e)
            logger.warning(f"Retry invoke {self._exception_wait_sec} sec after.")
            time.sleep(self._exception_wait_sec)
            return self.invoke(input, director_config, **kwargs)  # retry
        except OpenAIError as e:
            logger.warning(e)
            logger.warning(f"Retry invoke {self._exception_wait_sec} sec after.")
            time.sleep(self._exception_wait_sec)
            return self.invoke(input, director_config, **kwargs)  # retry
        except Exception as e:
            logger.error(type(e).__mro__)
            logger.error(f"Error: {e}.")
            raise


if __name__ == "__main__":
    """
    python -m client.chain_base
    """
    pass
