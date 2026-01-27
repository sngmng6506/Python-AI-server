import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.onnx
from pathlib import Path
import zipfile
import json

# Utils import
sys.path.append(os.path.join(os.path.dirname(__file__), 'usad'))
from utils import get_default_device, to_device


class TimeSeriesWindowDataset(Dataset):
    """시계열 데이터를 윈도우로 변환하는 Dataset"""
    def __init__(self, data, window_size=5):
        self.window_size = window_size
        self.n_features = data.shape[1]
        
        windows = []
        for i in range(len(data) - window_size + 1):
            window = data[i:i+window_size]  # [window_size, n_features]
            windows.append(window)
        
        self.windows = np.array(windows, dtype=np.float32)
        print(f"윈도우 생성 완료: {len(self.windows)}개 윈도우, shape: {self.windows[0].shape}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx])



class TimeSeriesCNN(nn.Module):
    """
    시간축에만 1D CNN 적용 (센서별 독립)
    [batch, window_size, n_features] -> [batch, n_features]
    """
    def __init__(self, n_features=25000, window_size=5):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        
        # 🔥 센서별 독립 Time CNN
        self.conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=window_size,
            stride=1,
            padding=0,
            groups=n_features      # ✅ 핵심
        )
        # self.bn = nn.BatchNorm1d(n_features)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: [batch, window_size, n_features]
        
        # Conv1D는 (B, C, T)
        x = x.permute(0, 2, 1)        # [batch, n_features, window_size]
        
        x = self.conv(x)              # [batch, n_features, 1]
        # x = self.bn(x)
        x = self.relu(x)
        
        x = x.squeeze(-1)             # [batch, n_features]
        return x



class SimpleAutoencoder(nn.Module):
    """
    기본적인 Autoencoder
    시간축에 1D CNN 적용 후 Autoencoder
    입력: [batch, window_size, n_features]
    출력: [batch, n_features] - 센서별 이상치 점수
    """
    def __init__(self, window_size=5, n_features=25000, latent_size=100):
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features
        
        # 시간축에 1D CNN 적용
        self.time_cnn = TimeSeriesCNN(n_features=n_features, window_size=window_size)
        
        # Autoencoder: 센서 차원에 대해 적용
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, n_features // 4),
            nn.ReLU(),
            nn.Linear(n_features // 4, n_features // 8),
            nn.ReLU(),
            nn.Linear(n_features // 8, latent_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, n_features // 8),
            nn.ReLU(),
            nn.Linear(n_features // 8, n_features // 4),
            nn.ReLU(),
            nn.Linear(n_features // 4, n_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, window_size, n_features]
        Returns:
            sensor_scores: [batch, n_features] - 각 센서별 이상치 점수
        """
        # 1. 시간축에 CNN 적용: [batch, window_size, n_features] -> [batch, n_features]
        time_compressed = self.time_cnn(x)
        
        # 2. Autoencoder: [batch, n_features] -> [batch, n_features]
        z = self.encoder(time_compressed)  # [batch, latent_size]
        reconstructed = self.decoder(z)  # [batch, n_features]
        
        # 3. 각 센서별 재구성 오차 계산
        sensor_scores = (time_compressed - reconstructed) ** 2  # [batch, n_features]
        
        return sensor_scores
    
    def training_step(self, batch):
        """학습용 forward pass"""
        sensor_scores = self.forward(batch)
        # 전체 센서의 평균 오차를 loss로 사용
        loss = sensor_scores.mean()
        return loss
    
    def validation_step(self, batch):
        """검증용 forward pass"""
        with torch.no_grad():
            sensor_scores = self.forward(batch)
            loss = sensor_scores.mean()
        return {'val_loss': loss}


def load_data(data_path):
    """데이터 로드"""
    print(f"데이터 로드 중: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    
    file_size = os.path.getsize(data_path)
    if file_size == 0:
        raise ValueError(f"데이터 파일이 비어있습니다: {data_path}")
    
    print(f"파일 크기: {file_size / (1024**2):.2f} MB")
    
    try:
        data = np.load(data_path, allow_pickle=False)
        
        if 'X' not in data or 'y' not in data:
            raise KeyError("NPZ 파일에 'X' 또는 'y' 키가 없습니다.")
        
        X = data['X']
        y = data['y']
        
        if len(X) != len(y):
            raise ValueError(f"X와 y의 길이가 일치하지 않습니다: X={len(X)}, y={len(y)}")
        
        print(f"데이터 shape: X={X.shape}, y={y.shape}")
        print(f"정상: {np.sum(y == 0)}, 이상: {np.sum(y == 1)}")
        
        return X, y
        
    except zipfile.BadZipFile as e:
        print(f"\n❌ NPZ 파일이 손상되었습니다: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 데이터 로드 중 에러 발생: {e}")
        raise


def prepare_training_data(X, y, window_size=5, train_ratio=0.8):
    """학습 데이터 준비 (정상 데이터만 사용)"""
    normal_indices = np.where(y == 0)[0]
    X_normal = X[normal_indices]
    
    print(f"\n정상 데이터만 사용: {len(X_normal)} 시점")
    
    split_idx = int(len(X_normal) * train_ratio)
    X_train = X_normal[:split_idx]
    X_val = X_normal[split_idx:]
    
    print(f"Train: {len(X_train)} 시점, Val: {len(X_val)} 시점")
    
    train_dataset = TimeSeriesWindowDataset(X_train, window_size=window_size)
    val_dataset = TimeSeriesWindowDataset(X_val, window_size=window_size)
    
    return train_dataset, val_dataset


def print_gpu_memory(stage=""):
    """GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU 메모리 {stage}: 할당={allocated:.2f} GB, 예약={reserved:.2f} GB, 최대={max_allocated:.2f} GB")


def train_autoencoder(
    train_dataset,
    val_dataset,
    window_size=5,
    n_features=25000,
    latent_size=100,
    epochs=50,
    batch_size=1,
    learning_rate=1e-3
):
    """Autoencoder 모델 학습"""
    device = get_default_device()
    print(f"\n사용 디바이스: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 총 메모리: {total_memory:.2f} GB")
        print_gpu_memory("(초기)")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=False,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=False,
        num_workers=0
    )
    
    model = SimpleAutoencoder(
        window_size=window_size,
        n_features=n_features,
        latent_size=latent_size
    )
    model = model.to(device)
    
    print_gpu_memory("(모델 생성 후)")
    
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024**2)
    
    print(f"\n모델 구조:")
    print(f"  입력: [batch, {window_size}, {n_features}]")
    print(f"  Time CNN: 시간축 압축 [{window_size} -> 1] -> [batch, {n_features}]")
    print(f"  Encoder: {n_features} -> {n_features // 4} -> {n_features // 8} -> {latent_size}")
    print(f"  Decoder: {latent_size} -> {n_features // 8} -> {n_features // 4} -> {n_features}")
    print(f"  출력: [batch, {n_features}] - 센서별 이상치 점수")
    print(f"  총 파라미터: {total_params:,} ({model_size_mb:.2f} MB)")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = []
    total_batches = len(train_loader)
    print(f"\n학습 시작 (Epochs: {epochs}, Batch Size: {batch_size})...")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss_sum += loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % max(1, total_batches // 10) == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                avg_loss = epoch_loss_sum / batch_count
                print(f"  Epoch [{epoch+1}/{epochs}] [{batch_idx+1}/{total_batches}] "
                      f"({progress:.1f}%) - Loss: {avg_loss:.4f}")
        
        model.eval()
        val_outputs = []
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)
                val_outputs.append(model.validation_step(batch))
        
        val_losses = [x['val_loss'] for x in val_outputs]
        epoch_val_loss = torch.stack(val_losses).mean().item()
        history.append({'val_loss': epoch_val_loss})
        
        print(f"  Epoch [{epoch+1}/{epochs}] 완료 - "
              f"Train Loss: {epoch_loss_sum/batch_count:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}")
        print("-" * 60)
    
    print_gpu_memory("(학습 완료 후)")
    return model, history


def convert_to_onnx(model, output_path, window_size=5, n_features=25000):
    """ONNX 변환"""
    print(f"\nONNX 변환 중...")
    
    model.eval()
    model = model.cpu()
    
    dummy_input = torch.randn(1, window_size, n_features, dtype=torch.float32)
    print(f"입력 shape: {dummy_input.shape}")
    
    with torch.no_grad():
        try:
            _ = model(dummy_input)
            print("✓ 모델 검증 완료")
        except Exception as e:
            print(f"⚠️ 모델 검증 중 경고: {e}")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['sensor_scores'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'sensor_scores': {0: 'batch_size'}
            },
            opset_version=13,
            do_constant_folding=True,
            verbose=False
        )
        
        file_size = os.path.getsize(output_path) / (1024**2)
        print(f"ONNX 모델 저장 완료: {output_path}")
        print(f"파일 크기: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        raise


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("기본 Autoencoder 모델 학습 (센서별 이상치 출력)")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "data" / "timeseries_test.npz"
    model_dir = base_dir / "ai" / "models"
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # 하이퍼파라미터
    WINDOW_SIZE = 5
    N_FEATURES = 25000
    LATENT_SIZE = 100
    EPOCHS = 3
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    
    # 데이터 로드
    X, y = load_data(data_path)
    train_dataset, val_dataset = prepare_training_data(X, y, window_size=WINDOW_SIZE)
    
    # 모델 학습
    model, history = train_autoencoder(
        train_dataset,
        val_dataset,
        window_size=WINDOW_SIZE,
        n_features=N_FEATURES,
        latent_size=LATENT_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # PyTorch 모델 저장
    pytorch_model_path = model_dir / "autoencoder_model.pth"
    torch.save(model.state_dict(), pytorch_model_path)
    print(f"\nPyTorch 모델 저장: {pytorch_model_path}")
    
    # ONNX 변환
    onnx_model_path = model_dir / "autoencoder_model.onnx"
    convert_to_onnx(
        model,
        onnx_model_path,
        window_size=WINDOW_SIZE,
        n_features=N_FEATURES
    )
    
    # 메타데이터 저장
    metadata = {
        "model_type": "SimpleAutoencoder",
        "window_size": WINDOW_SIZE,
        "n_features": N_FEATURES,
        "latent_size": LATENT_SIZE,
        "input_shape": [WINDOW_SIZE, N_FEATURES],
        "output_shape": [N_FEATURES],
        "pytorch_model": str(pytorch_model_path),
        "onnx_model": str(onnx_model_path),
        "training_history": {
            "final_val_loss": float(history[-1]['val_loss'])
        }
    }
    
    metadata_path = model_dir / "autoencoder_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n메타데이터 저장: {metadata_path}")
    print("\n" + "=" * 60)
    print("학습 및 변환 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
