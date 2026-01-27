from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field
from ai.model import ModelManager
import numpy as np
from core.message_queue import queue, QueueItem  
from processors.worker import model_worker
import asyncio
from typing import Dict
import logging
from pythonjsonlogger import jsonlogger
from contextlib import asynccontextmanager
import gzip
import json
from torch import cuda
from core.backoff import *
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor



logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
    json_ensure_ascii=False # JSON 직렬화할때 한글 깨짐 방지
)
handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(handler)

logger = logging.getLogger(__name__) 




class EnqueueRequest(BaseModel):
    data : Dict[str, float] = Field(..., description="25000 features timeseries data / one point") # key-value 쌍 






@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    -Startup  : 모델 사전 로드, worker 시작 
    -Shutdown : worker 종료, GPU 정리 
    """
    # 모델 사전 로드 
    try:
        app.state.model_manager = ModelManager()
        app.state.inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference_thread")
        logger.info("Inference executor initialized successfully")

        # 모델 경로 설정
        base_dir = Path(__file__).parent
        from config.settings import settings
        
        # 디바이스 결정
        if settings.DEFAULT_DEVICE == "auto":
            device = 'cuda' if cuda.is_available() else 'cpu'
        else:
            device = settings.DEFAULT_DEVICE
        
        # 모델 초기화
        model_name = settings.DEFAULT_MODEL_NAME
        logger.info(f"Initializing model: {model_name}")
        
        # 모델 타입에 따라 초기화 인자 설정
        if model_name == "featurewise_ocsvm":
            model_path = base_dir / "ai" / "models" / "featurewise_ocsvm_unified.pth"
            metadata_path = base_dir / "ai" / "models" / "featurewise_ocsvm_metadata.json"
            init_kwargs = {
                "model_path": str(model_path),
                "device": device
            }
            if metadata_path.exists():
                init_kwargs["metadata_path"] = str(metadata_path)

        else:
            # dummy 모델 또는 기타 모델
            init_kwargs = {
                "device": device
            }
        
        app.state.model_manager.initialize(
            model_name = model_name,
            **init_kwargs
        )

        # Worker 시작
        app.state.worker_task = asyncio.create_task(model_worker(app.state.model_manager, app))
        logger.info("Model initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize model")


    yield
    app.state.worker_task.cancel()
    
    try:
        await app.state.worker_task
    except asyncio.CancelledError:
        logger.info("Worker stopped cleanly")

    try:
        if cuda.is_available():
            cuda.empty_cache()
            logger.info("GPU cache cleared successfully")

    except Exception as e:
        logger.error(f"Error clearing GPU cache: {e}")
        pass


    try: 
        if hasattr(app.state, "inference_executor") and app.state.inference_executor:
            app.state.inference_executor.shutdown(wait=True)
            logger.info("Inference executor shutdown successfully")
    except Exception as e:
        logger.error(f"Error shutting down inference executor: {e}")
        


app = FastAPI(lifespan=lifespan)



@app.post("/data-enqueue", status_code=status.HTTP_202_ACCEPTED) # 요청 큐에 데이터 추가
async def enqueue(request: Request): 
    
    # 1. 압축 데이터 받기
    compressed_data = await request.body()
    logger.info(f"Compressed data size: {len(compressed_data)/1024:.2f} KB")

    # 2. 압축 해제 
    try:
        json_bytes = gzip.decompress(compressed_data)
        logger.info(f"Decompressed data size: {len(json_bytes)/1024:.2f} KB")

    except Exception as e:
        logger.error(f"Error decompressing data: {e}")
        raise HTTPException(status_code=400, detail="Invalid compressed data")

    # 3. JSON 파싱 
    try:
        raw_data = json.loads(json_bytes.decode('utf-8'))
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    
    # 4. Pydantic으로 데이터 검증 (타입 검증)
    try:
        validated_data = EnqueueRequest(**raw_data)
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        raise HTTPException(status_code=400, detail="Invalid data")

    # 5. 데이터 변환
    vec = np.array(list(validated_data.data.values()), dtype=np.float32)   # shape = [25000]

    receive_data_timestamp = time.time() # 데이터 수신 시간. 


    # 6. 큐에 추가 (Non-blocking)
    try:
        await queue.put(QueueItem(data=vec, receive_timestamp=receive_data_timestamp))
        logger.info("Successfully put data into queue")
    except asyncio.QueueFull:
        raise HTTPException(status_code=429, detail="Queue is full")

    return {
        "status": "ok",
        "message": "Data enqueued successfully"
    }





@app.get("/health/live", status_code=status.HTTP_200_OK)
async def live():
    return {
        "status": "ok",
        "message": "Service is running"
    }




@app.get("/health/ready", status_code=status.HTTP_200_OK)
async def ready():

    # 모델 변수가 None이 아닌지 확인
    try:
        from config.settings import settings
        model_manager = ModelManager()
        is_ready = model_manager.is_ready(settings.DEFAULT_MODEL_NAME)


    except Exception as e:
        logger.error(f"Error checking model readiness: {e}")
        raise HTTPException(status_code=503, detail="Service Unavailable")
    
    # GPU 사용 가능한지 확인
    gpu_available = cuda.is_available()

    
    return {
        "status": "ok",
        "gpu":{
            "is_available" : gpu_available
        },
        "model":{
            "is_ready" : is_ready
        }
    }
