import time
import logging
import asyncio
from config.settings import settings




logger = logging.getLogger("python-server")




def backoff(func):
    def wrapper(*args, **kwargs):
        for retry in range(settings.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if retry >= settings.MAX_RETRIES - 1: 
                    raise
                wait_time = settings.INITIAL_DELAY * (2 ** retry)
                logger.error(f"{e} occurred. Retrying in{wait_time} seconds...")
                time.sleep(wait_time)

    return wrapper




def async_backoff(func):
    async def wrapper(*args, **kwargs):
        for i in range(settings.MAX_RETRIES):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if i >= settings.MAX_RETRIES - 1:
                    raise
                wait_time = settings.INITIAL_DELAY * (2 ** i)
                logger.error(f"{e} occurred. Retrying in{wait_time} seconds...")
                await asyncio.sleep(wait_time)
    return wrapper
