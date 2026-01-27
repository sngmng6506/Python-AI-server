import torch
import torch.nn as nn
from typing import Optional, Dict, Type, Any, List
import logging
from pathlib import Path
import numpy as np
import json
import time


logger = logging.getLogger(__name__)


# ---- Base Interface ----
class BaseModel:
    def predict(self, *args, **kwargs):
        raise NotImplementedError


# ---- Concrete Model ----
class DummyModel(BaseModel):
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)


    def predict(self, x: torch.Tensor):
        """
        x: [B, T, F]
        return: [B, F]
        """
        x = x.to(self.device)

        # 시간 축(T)만 평균 → feature-wise score
        output = x.mean(dim=1)   # [B, F]

        return output


class LinearOneClassSVM(nn.Module):
    """
    PyTorch 기반 Linear One-Class SVM
    입력: [batch, window_size]
    출력: [batch, 1] (decision score)
    """
    def __init__(self, window_size: int = 5, nu: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.nu = nu
        
        # 선형 레이어: w^T * x - rho
        self.linear = nn.Linear(window_size, 1, bias=True)
        
        # 초기화: 작은 랜덤 값
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, window_size]
        Returns:
            decision_score: [batch, 1] (양수면 정상, 음수면 이상)
        """
        return self.linear(x)


class UnifiedFeatureWiseOCSVM(nn.Module):
    """
    25,000개의 feature별 Linear One-Class SVM을 하나의 모델로 통합
    """
    def __init__(self, window_size: int = 5, n_features: int = 25000, nu: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.nu = nu
        
        # 모든 feature 모델을 ModuleList로 저장
        self.models = nn.ModuleList([
            LinearOneClassSVM(window_size=window_size, nu=nu)
            for _ in range(n_features)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, window_size, n_features]
        Returns:
            anomaly_scores: [batch, n_features] - 각 feature별 이상 점수
        """
        batch_size, window_size, n_features = x.shape
        
        # 각 feature별 이상 점수 계산
        feature_scores = []
        for i in range(n_features):
            feature_data = x[:, :, i]  # [batch, window_size]
            decision_score = self.models[i](feature_data)  # [batch, 1]
            # 음수면 이상이므로 절댓값 사용
            anomaly_score = -decision_score.squeeze(-1)  # [batch]
            feature_scores.append(anomaly_score)
        
        # [n_features, batch] -> [batch, n_features]
        score_matrix = torch.stack(feature_scores, dim=1)
        return score_matrix
    
    def predict_anomaly_score(self, x: torch.Tensor) -> float:
        """
        전체 이상 점수 계산 (모든 feature의 평균)
        """
        score_matrix = self.forward(x)  # [batch, n_features]
        final_score = torch.mean(score_matrix).item()
        return max(0.0, final_score)


class FeatureWiseOCSVMModel(BaseModel):
    """
    25,000개의 feature별 Linear One-Class SVM (통합 모델 버전)
    """
    def __init__(self, model_path: str, metadata_path: str = None, device: str = "cpu"):
        """
        Args:
            model_path: 통합 모델이 저장된 파일 경로 (.pth)
            metadata_path: 메타데이터 JSON 파일 경로
            device: 실행 디바이스 ("cpu" 또는 "cuda")
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 메타데이터 로드
        self.metadata = {}
        if metadata_path:
            metadata_file = Path(metadata_path)
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
        
        self.window_size = self.metadata.get('window_size', 5)
        self.n_features = self.metadata.get('n_features', 25000)
        self.nu = self.metadata.get('nu', 0.1)
        self.device = torch.device(device)
        
        # 통합 모델 로드 (시간 측정)
        logger.info(f"Loading unified feature-wise OCSVM model from {model_path}...")
        start_time = time.time()
        
        # 1. 모델 구조 생성
        logger.info(f"Creating model structure with {self.n_features} features...")
        model_create_start = time.time()
        self.unified_model = UnifiedFeatureWiseOCSVM(
            window_size=self.window_size,
            n_features=self.n_features,
            nu=self.nu
        )
        logger.info(f"Model structure created in {time.time() - model_create_start:.2f}s")
        
        # 2. 파일 크기 확인
        file_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        logger.info(f"Model file size: {file_size_mb:.2f} MB")
        
        # 3. state_dict 로드 (CPU에서 먼저 로드)
        logger.info("Loading state_dict from file...")
        load_start = time.time()
        try:
            state_dict = torch.load(
                self.model_path, 
                map_location='cpu',
                weights_only=False
            )
        except TypeError:
            state_dict = torch.load(
                self.model_path, 
                map_location='cpu'
            )
        logger.info(f"State_dict loaded in {time.time() - load_start:.2f}s")
        logger.info(f"State_dict keys: {len(state_dict)} keys")
        
        # 4. 가중치 적용 (모델은 CPU에 있음)
        logger.info("Loading weights into model (on CPU)...")
        load_weights_start = time.time()
        try:
            missing_keys, unexpected_keys = self.unified_model.load_state_dict(
                state_dict, 
                strict=False
            )
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
        except Exception as e:
            logger.error(f"Error loading state_dict: {e}")
            raise
        logger.info(f"Weights loaded in {time.time() - load_weights_start:.2f}s")
        
        # 5. 메모리 정리
        del state_dict
        
        # 6. 모델을 한 번에 GPU로 이동
        logger.info(f"Moving model to {device}...")
        device_move_start = time.time()
        self.unified_model.to(self.device)
        logger.info(f"Model moved to device in {time.time() - device_move_start:.2f}s")
        
        # 7. GPU 메모리 정리
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 8. 모드 설정
        self.unified_model.eval()
        logger.info("Model set to eval mode")
        
        total_time = time.time() - start_time
        logger.info(f"✅ Loaded unified model with {self.n_features} feature models in {total_time:.2f}s")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature-wise 이상 점수 예측
        x: [batch, window_size, n_features]
        return: [batch, n_features]
        """
        if isinstance(x, torch.Tensor):
            x_tensor = x.to(self.device)
        else:
            x_np = np.array(x, dtype=np.float32)
            x_tensor = torch.FloatTensor(x_np).to(self.device)
        
        _, window_size, n_features = x_tensor.shape
        
        if window_size != self.window_size:
            raise ValueError(f"Expected window_size={self.window_size}, got {window_size}")
        if n_features != self.n_features:
            raise ValueError(f"Expected n_features={self.n_features}, got {n_features}")
        
        with torch.no_grad():
            score_matrix = self.unified_model.forward(x_tensor)
        
        return score_matrix


# ---- Model Registry ----
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "dummy": DummyModel,
    "featurewise_ocsvm": FeatureWiseOCSVMModel,
    # 모델들 추가될 위치 
}

def get_model_cls(model_name: str) -> Type[BaseModel]:
    if model_name not in MODEL_REGISTRY:
        raise RuntimeError(f"Model '{model_name}' not found in registry.")
    return MODEL_REGISTRY[model_name]


# ---- Model Manager ----
class ModelManager:
    _instance: Optional["ModelManager"] = None
    DEFAULT_MODEL_NAME = "dummy"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models: Dict[str, BaseModel] = {}
        return cls._instance

    def initialize(
        self,
        model_name: Optional[str] = None,
        **kwargs: Any
    ) -> BaseModel:
        
        if model_name is None:
            model_name = self.DEFAULT_MODEL_NAME

        model_cls = get_model_cls(model_name)
        logger.info(f"Initializing model '{model_name}' with class '{model_cls.__name__}'...")
        logger.debug(f"Model initialization kwargs: {list(kwargs.keys())}")

        self._models[model_name] = model_cls(**kwargs)
        return self._models[model_name]


    def get_model(self, name: str) -> BaseModel:
        if name not in self._models:
            raise RuntimeError(f"Model '{name}' not initialized.")
        return self._models[name]

    def is_ready(self, name: str) -> bool:
        return name in self._models
