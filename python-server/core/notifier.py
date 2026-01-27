import httpx
import logging
from typing import Dict, Any
from datetime import datetime
from core.backoff import async_backoff
from config.settings import settings
import time

logger = logging.getLogger(__name__)



# test 코드 
# @async_backoff
# async def send_alert(output: float):
    
#     data = {
#         'timestamp': datetime.now().isoformat(),
#         "anomaly_detected" : True
#     }


#     try:
#         async with httpx.AsyncClient(timeout=settings.NESTJS_TIMEOUT) as client:
#             response = await client.post(
#                 f"{settings.NESTJS_URL}{settings.NESTJS_ANOMALY_ENDPOINT}", json=data)
#             response.raise_for_status()

#             logger.info(f"Alert sent successfully", extra= {"ID" : time.time(), "anomaly_score" : output})

#             return response.json()

#     except Exception as e:
#         logger.error(f" error sending alert: {e}")
#         raise e


@async_backoff
async def send_alert(
    anomalous_features: list[dict],
):
    data = {
        "timestamp": datetime.now().isoformat(),
        "anomaly_detected": True,
        "num_anomalous_features": len(anomalous_features),
        "anomalous_features": anomalous_features,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.NESTJS_TIMEOUT) as client:
            response = await client.post(
                f"{settings.NESTJS_URL}{settings.NESTJS_ANOMALY_ENDPOINT}",
                json=data,
            )
            response.raise_for_status()

            logger.info(
                "Alert sent successfully",
            )

            return response.json()

    except Exception as e:
        logger.error(
            "Error sending alert",
        )
        raise


    