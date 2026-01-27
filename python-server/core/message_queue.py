import asyncio
from dataclasses import dataclass
import numpy as np
from config.settings import settings

@dataclass
class QueueItem:
    data: np.ndarray
    receive_timestamp: float
    

queue = asyncio.Queue(maxsize=settings.QUEUE_MAX_SIZE) # 1초당 25000개 씩 들어오는 데이터를 설정된 크기까지 저장
# '{"data": {"feature_0": 0.12, "feature_1": 0.34, 