import torch
import asyncio
from core.message_queue import queue, QueueItem
import logging
import numpy as np
from core.notifier import send_alert
import time
from processors.resource_check import check_resource_usage
from collections import deque
from config.settings import settings
from concurrent.futures import ThreadPoolExecutor



logger = logging.getLogger(__name__)



async def model_worker(model_manager: "ModelManager", app):

    try:
        model = model_manager.get_model(settings.DEFAULT_MODEL_NAME)
        logger.info(f"Worker loaded with model: {settings.DEFAULT_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Error loading worker: {e}")
        raise RuntimeError("Failed to load worker")


    # 슬라이딩 윈도우를 위한 버퍼 
    window_buffer = deque(maxlen=5)
    window_timestamp = deque(maxlen=5)

    batch_windows = []
    BATCH_SIZE = settings.INFERENCE_BATCH_SIZE


    while True:
        item: QueueItem = await queue.get()


        window_buffer.append(item.data)
        window_timestamp.append(item.receive_timestamp)


        if len(window_buffer) < 5:
            logger.info(f"Window buffer is not full: {len(window_buffer)}")
            continue

        batch_windows.append(
            np.asarray(window_buffer, dtype=np.float32)
        )



        # 배치가 아직 안 찼으면 대기
        if len(batch_windows) < BATCH_SIZE:
            logger.info(f"Batch windows is not full: {len(batch_windows)}")
            continue

        # Lateny 기록 시작점
        if len(batch_windows) == BATCH_SIZE:
            batch_start_time = time.time()


        # 배치 추론 
        inputs = torch.from_numpy(
            np.stack(batch_windows, axis=0)   # [B, 5, 25000]
        ).to(model.device)

        # with torch.no_grad():
        #     outputs = model.predict(inputs) # [B, F]

        loop = asyncio.get_running_loop()
        with torch.no_grad():
            outputs = await loop.run_in_executor(app.state.inference_executor, model.predict, inputs)


        inference_timestamp = time.time()

        
        resource_usage = check_resource_usage()
        THRESHOLD = 0.9
        # outputs: [B, F]


        batch_anomaly_counts = []


        try:
            for b in range(outputs.shape[0]):

                feature_scores = outputs[b]   # [F]

                # feature별 score list (로깅/전송용)
                scores = feature_scores.detach().cpu().tolist()

                anomalous_features = [
                    {"feature_idx": i, "score": float(s)}
                    for i, s in enumerate(scores)
                    if s >= THRESHOLD
                    ]
                
                batch_anomaly_counts.append(len(anomalous_features))



                if anomalous_features:
                    asyncio.create_task(
                        send_alert({
                            "features": anomalous_features,
                        })
                    )
            
            inference_latency_ms = (time.time() - batch_start_time) * 1000
          
            logger.info(
                "Model inference completed",
                extra={
                    "avg_num_anomaly_features_per_BATCH":  sum(batch_anomaly_counts) / len(batch_anomaly_counts),
                    "inference_latency_ms": round(inference_latency_ms, 2),
                    "batch_size": BATCH_SIZE,
                    "device": resource_usage["device"],
                    "memory_percent": resource_usage["memory_percent"],
                    "compute_percent": resource_usage["compute_percent"],
                },
            )


            batch_windows.clear()
        


        except Exception as e:
            logger.error(f"Error in model worker: {e}")
            raise e
