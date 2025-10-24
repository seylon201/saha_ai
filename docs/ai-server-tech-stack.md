# AI 서버 기술 스택 구성 가이드

**프로젝트**: 장림 유수지 LSTM 수위 예측 모델
**작성일**: 2025-10-16
**목적**: AI 모델 개발 및 운영을 위한 서버 환경 구축

---

## 목차
1. [시스템 요구사항](#1-시스템-요구사항)
2. [핵심 기술 스택](#2-핵심-기술-스택)
3. [Phase별 추가 스택](#3-phase별-추가-기술-스택)
4. [서버 구축 절차](#4-서버-구축-절차)
5. [최적화 설정](#5-성능-최적화-설정)
6. [보안 설정](#6-보안-설정)
7. [체크리스트](#7-구축-체크리스트)
8. [예상 비용](#8-예상-비용)

---

## 1. 시스템 요구사항

### 1.1 하드웨어 권장사양

#### 최소 사양 (개발 환경)
```
CPU: 4코어 이상
RAM: 16GB
GPU: NVIDIA GPU 6GB+ VRAM (GTX 1660 이상)
저장소: SSD 256GB
네트워크: 100Mbps
```

#### 권장 사양 (프로덕션 환경)
```
CPU: 8코어 이상 (Intel Xeon / AMD EPYC)
RAM: 32GB 이상 (64GB 권장)
GPU: NVIDIA GPU 8GB+ VRAM
      - RTX 3060 (12GB) - 개발/소규모
      - RTX 4090 (24GB) - 대규모 학습
      - A4000 (16GB) - 프로덕션
      - T4 (16GB) - 클라우드
저장소: NVMe SSD 500GB 이상 (1TB 권장)
네트워크: 1Gbps 이상
```

### 1.2 운영체제

#### 권장 OS
```
Ubuntu 22.04 LTS (가장 권장)
Ubuntu 20.04 LTS
```

#### 지원 가능 OS
```
CentOS 8 / Rocky Linux 8
Red Hat Enterprise Linux 8
Windows Server 2019/2022 (비권장 - Linux 우선)
```

---

## 2. 핵심 기술 스택

### 2.1 Python 환경

#### Python 버전
```bash
Python 3.10.x (권장)
또는
Python 3.8 ~ 3.11

# TensorFlow 2.13 호환성
- Python 3.8: ✓
- Python 3.9: ✓
- Python 3.10: ✓ (권장)
- Python 3.11: ✓
- Python 3.12: ✗ (미지원)
```

#### 패키지 관리
```bash
pip 23.x
virtualenv 20.x
또는
conda 23.x (Anaconda/Miniconda)
```

### 2.2 딥러닝 프레임워크

#### GPU 지원 라이브러리
```bash
# NVIDIA GPU 필수 구성
CUDA Toolkit 11.8
cuDNN 8.6.0

# 호환성 확인
TensorFlow 2.13.0 → CUDA 11.8 + cuDNN 8.6
TensorFlow 2.15.0 → CUDA 12.2 + cuDNN 8.9
```

#### TensorFlow 설치
```bash
# GPU 버전 (권장)
pip install tensorflow-gpu==2.13.0

# CPU 버전 (GPU 없는 경우)
pip install tensorflow==2.13.0
```

### 2.3 데이터 처리 라이브러리

```bash
# 필수 패키지
pandas==2.0.3         # 시계열 데이터 처리
numpy==1.24.3         # 수치 연산
openpyxl==3.1.2       # Excel 파일 읽기
scikit-learn==1.3.0   # 전처리, 평가 지표, 스케일링
```

**주요 사용 기능**:
- `pandas`: CSV/Excel 로딩, 시계열 리샘플링, 결측치 처리
- `numpy`: 배열 연산, 정규화
- `scikit-learn`: MinMaxScaler, train_test_split, 평가 지표

### 2.4 시각화 라이브러리

```bash
matplotlib==3.7.2     # 기본 그래프 (학습 곡선, 예측 결과)
seaborn==0.12.2       # 통계 시각화 (상관관계, 분포)
plotly==5.18.0        # 인터랙티브 그래프 (선택사항)
```

### 2.5 개발 도구

```bash
jupyter==1.0.0        # Jupyter Notebook
ipython==8.18.0       # 대화형 Python 셸
jupyterlab==4.0.9     # Jupyter Lab (선택)
```

---

## 3. Phase별 추가 기술 스택

### 3.1 Phase 1: 기본 LSTM 모델 (현재)

**필요 스택**: 위 핵심 기술 스택만으로 충분

```txt
tensorflow-gpu==2.13.0
pandas==2.0.3
numpy==1.24.3
openpyxl==3.1.2
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
```

### 3.2 Phase 2: 실시간 데이터 처리

#### 스트리밍 데이터 처리
```bash
# 메시지 큐
kafka-python==2.0.2   # Apache Kafka 클라이언트
pika==1.3.2           # RabbitMQ (대안)

# 캐싱 및 실시간 데이터 저장
redis==5.0.0          # Redis (인메모리 DB)
hiredis==2.2.3        # Redis 고성능 파서

# 비동기 처리
asyncio               # Python 내장
aiohttp==3.9.0        # 비동기 HTTP 클라이언트
```

#### 데이터베이스
```bash
# 관계형 DB
psycopg2-binary==2.9.9    # PostgreSQL
SQLAlchemy==2.0.23        # ORM

# NoSQL / 시계열 DB
pymongo==4.6.0            # MongoDB
influxdb-client==1.38.0   # InfluxDB (시계열 전용)
```

### 3.3 Phase 3: API 서버 (모델 서빙)

#### FastAPI 스택 (권장)
```bash
fastapi==0.104.0          # 고성능 웹 프레임워크
uvicorn[standard]==0.24.0 # ASGI 서버
pydantic==2.5.0           # 데이터 검증 및 시리얼라이제이션
python-multipart==0.0.6   # 파일 업로드 지원
```

**장점**:
- 자동 API 문서 생성 (Swagger UI)
- 고성능 (비동기 처리)
- 타입 힌팅 및 검증

#### Flask 스택 (대안)
```bash
flask==3.0.0              # 웹 프레임워크
gunicorn==21.2.0          # WSGI 서버
flask-cors==4.0.0         # CORS 지원
```

#### 모델 서빙 전용
```bash
# TensorFlow Serving (Docker 기반)
tensorflow-serving-api==2.13.0

# ONNX 변환 (추론 속도 최적화)
onnx==1.15.0
onnxruntime-gpu==1.16.0   # GPU 지원
```

### 3.4 Phase 4: 모니터링 & MLOps

#### 실험 추적 및 모델 관리
```bash
mlflow==2.9.0             # 실험 관리, 모델 레지스트리
tensorboard==2.13.0       # TensorFlow 시각화
wandb==0.16.0             # Weights & Biases (대안)
```

#### 시스템 모니터링
```bash
prometheus-client==0.19.0 # 메트릭 수집
psutil==5.9.6             # 시스템 리소스 모니터링
py-cpuinfo==9.0.0         # CPU 정보

# 별도 설치 필요
# Grafana - 대시보드 시각화
# Prometheus - 메트릭 DB
```

#### 로깅
```bash
loguru==0.7.2             # 고급 로깅 라이브러리
python-json-logger==2.0.7 # JSON 형식 로그
```

### 3.5 Phase 5: 배포 & 오케스트레이션

#### 컨테이너화
```bash
# Docker
Docker 24.0+
docker-compose 2.x

# Dockerfile 예시
FROM tensorflow/tensorflow:2.13.0-gpu
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "scripts/train_model.py"]
```

#### 오케스트레이션 (대규모)
```bash
# Kubernetes
kubectl 1.28+
helm 3.x

# 대안
Docker Swarm
Apache Airflow (워크플로우 관리)
```

#### CI/CD
```bash
# GitLab CI / GitHub Actions
# Jenkins (자체 호스팅)
```

---

## 4. 서버 구축 절차

### Step 1: 기본 환경 설정

#### Ubuntu 서버 초기 설정
```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 개발 도구
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    tree

# 시스템 정보 확인
lscpu                 # CPU 정보
free -h               # 메모리 정보
df -h                 # 디스크 정보
lspci | grep -i nvidia # GPU 확인
```

### Step 2: Python 설치

#### Python 3.10 설치 (Ubuntu 22.04)
```bash
# Python 3.10 설치
sudo apt install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip

# 기본 Python 설정
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# pip 업그레이드
python -m pip install --upgrade pip setuptools wheel
```

#### Conda 설치 (대안)
```bash
# Miniconda 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 환경 생성
conda create -n jangrim python=3.10 -y
conda activate jangrim
```

### Step 3: NVIDIA GPU 드라이버 & CUDA

#### GPU 드라이버 설치
```bash
# NVIDIA 드라이버 확인
nvidia-smi

# 드라이버가 없으면 설치
sudo apt install -y nvidia-driver-535
sudo reboot

# 재부팅 후 확인
nvidia-smi
```

#### CUDA Toolkit 11.8 설치
```bash
# CUDA 11.8 다운로드 및 설치
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

# 환경변수 설정
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 확인
nvcc --version
```

#### cuDNN 8.6 설치
```bash
# cuDNN 다운로드 (NVIDIA 개발자 계정 필요)
# https://developer.nvidia.com/cudnn

# cuDNN 설치
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.6.0.163/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y libcudnn8 libcudnn8-dev
```

### Step 4: 프로젝트 환경 구축

#### 프로젝트 디렉토리 설정
```bash
# 프로젝트 폴더 생성
sudo mkdir -p /opt/jangrim-lstm
sudo chown $USER:$USER /opt/jangrim-lstm
cd /opt/jangrim-lstm

# Git 클론 (또는 파일 복사)
git clone <repository-url> .
# 또는
scp -r local-project-dir/* user@server:/opt/jangrim-lstm/
```

#### Python 가상환경 생성
```bash
# venv 생성
python3.10 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# pip 업그레이드
pip install --upgrade pip setuptools wheel
```

#### 패키지 설치
```bash
# requirements.txt 설치
pip install -r requirements.txt

# GPU 확인
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### Step 5: Jupyter 원격 접속 설정

#### Jupyter 설정
```bash
# Jupyter 설정 파일 생성
jupyter notebook --generate-config

# 비밀번호 설정
python -c "from notebook.auth import passwd; print(passwd())"
# 출력된 해시값 복사

# 설정 파일 수정
vim ~/.jupyter/jupyter_notebook_config.py
```

#### jupyter_notebook_config.py 내용
```python
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.password = 'sha1:...'  # 위에서 복사한 해시값
c.NotebookApp.allow_root = True
c.NotebookApp.notebook_dir = '/opt/jangrim-lstm/notebooks'
```

#### Jupyter 서비스 등록 (systemd)
```bash
# 서비스 파일 생성
sudo vim /etc/systemd/system/jupyter.service
```

```ini
[Unit]
Description=Jupyter Notebook Server
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/opt/jangrim-lstm
ExecStart=/opt/jangrim-lstm/venv/bin/jupyter notebook
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 시작
sudo systemctl daemon-reload
sudo systemctl enable jupyter
sudo systemctl start jupyter
sudo systemctl status jupyter
```

### Step 6: Docker 설치 (선택사항)

#### Docker 설치
```bash
# Docker 공식 설치 스크립트
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER
newgrp docker

# 확인
docker --version
docker run hello-world
```

#### NVIDIA Container Toolkit (GPU 지원)
```bash
# NVIDIA Docker 저장소 추가
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 설치
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Docker 재시작
sudo systemctl restart docker

# GPU 테스트
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#### Docker Compose 설치
```bash
# Docker Compose V2 (플러그인)
sudo apt install -y docker-compose-plugin

# 확인
docker compose version
```

---

## 5. 성능 최적화 설정

### 5.1 TensorFlow GPU 최적화

#### src/config.py에 추가
```python
import tensorflow as tf
import os

def setup_tensorflow():
    """TensorFlow GPU 최적화 설정"""

    # GPU 메모리 동적 할당 (OOM 방지)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU {len(gpus)}개 감지 - 메모리 동적 할당 활성화")
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")

    # Mixed Precision Training (학습 속도 2배 향상)
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("✓ Mixed Precision (FP16) 활성화")

    # 멀티스레딩 최적화
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(8)

    # TensorFlow 로그 레벨 설정
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFO, WARNING만 표시

    return gpus

# 사용 예시
if __name__ == "__main__":
    setup_tensorflow()
```

### 5.2 데이터 로딩 최적화

```python
# tf.data API 사용
def create_dataset(X, y, batch_size=32, shuffle=True):
    """고성능 데이터 파이프라인"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # 백그라운드 로딩

    return dataset
```

### 5.3 시스템 리소스 모니터링

#### 모니터링 스크립트
```python
import psutil
import GPUtil

def monitor_resources():
    """시스템 리소스 모니터링"""
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)

    # 메모리
    mem = psutil.virtual_memory()
    mem_percent = mem.percent

    # GPU
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.memoryUtil*100:.1f}% | {gpu.temperature}°C")

    print(f"CPU: {cpu_percent}% | RAM: {mem_percent}%")

# requirements.txt에 추가
# psutil==5.9.6
# gputil==1.4.0
```

---

## 6. 보안 설정

### 6.1 방화벽 설정 (UFW)

```bash
# UFW 설치 및 활성화
sudo apt install -y ufw

# 기본 정책 설정
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 포트 허용
sudo ufw allow 22/tcp         # SSH
sudo ufw allow 8888/tcp       # Jupyter Notebook
sudo ufw allow 8000/tcp       # FastAPI (필요시)
sudo ufw allow 5000/tcp       # Flask (필요시)

# 특정 IP만 허용 (보안 강화)
sudo ufw allow from 192.168.1.0/24 to any port 8888

# 방화벽 활성화
sudo ufw enable

# 상태 확인
sudo ufw status verbose
```

### 6.2 SSH 보안 강화

```bash
# SSH 설정 파일 수정
sudo vim /etc/ssh/sshd_config
```

```ini
# 루트 로그인 비활성화
PermitRootLogin no

# 패스워드 인증 비활성화 (키 인증만 허용)
PasswordAuthentication no
PubkeyAuthentication yes

# 포트 변경 (선택)
Port 2222
```

```bash
# SSH 재시작
sudo systemctl restart sshd
```

### 6.3 환경변수 관리

#### .env 파일 생성
```bash
# 프로젝트 루트에 .env 생성
cat > /opt/jangrim-lstm/.env << EOF
# 프로젝트 경로
PYTHONPATH=/opt/jangrim-lstm

# TensorFlow 설정
TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=0

# 데이터베이스 (Phase 2)
# DB_HOST=localhost
# DB_PORT=5432
# DB_USER=jangrim
# DB_PASSWORD=secure_password
# DB_NAME=jangrim_db

# Redis (Phase 2)
# REDIS_HOST=localhost
# REDIS_PORT=6379

# API 설정 (Phase 3)
# API_HOST=0.0.0.0
# API_PORT=8000
# API_SECRET_KEY=your-secret-key

# 기상청 API (Phase 4)
# WEATHER_API_KEY=your-api-key
EOF

# .env 파일 권한 설정
chmod 600 .env
```

#### python-dotenv 사용
```python
# src/config.py
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 환경변수 사용
PYTHONPATH = os.getenv('PYTHONPATH')
CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')
```

### 6.4 정기 백업

#### 백업 스크립트
```bash
#!/bin/bash
# /opt/jangrim-lstm/scripts/backup.sh

BACKUP_DIR="/backup/jangrim-lstm"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 백업 폴더 생성
mkdir -p $BACKUP_DIR

# 모델 파일 백업
tar -czf $BACKUP_DIR/models_$TIMESTAMP.tar.gz \
    /opt/jangrim-lstm/models/*.keras \
    /opt/jangrim-lstm/models/*.npy

# 데이터 백업 (선택)
# tar -czf $BACKUP_DIR/data_$TIMESTAMP.tar.gz /opt/jangrim-lstm/data/processed/

# 오래된 백업 삭제 (30일 이상)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "백업 완료: $TIMESTAMP"
```

#### Cron 등록
```bash
# 매일 새벽 2시 백업
crontab -e
```

```cron
0 2 * * * /opt/jangrim-lstm/scripts/backup.sh >> /var/log/jangrim-backup.log 2>&1
```

---

## 7. 구축 체크리스트

### Phase 1: 개발 환경 (현재)

#### 시스템 기본
- [ ] Ubuntu 22.04 LTS 설치
- [ ] 시스템 업데이트 (`apt update && apt upgrade`)
- [ ] 필수 패키지 설치 (`build-essential`, `git`, 등)

#### Python 환경
- [ ] Python 3.10 설치
- [ ] pip 업그레이드
- [ ] 가상환경 생성 (`venv` 또는 `conda`)

#### GPU 환경 (GPU 서버만)
- [ ] NVIDIA 드라이버 설치
- [ ] CUDA 11.8 설치
- [ ] cuDNN 8.6 설치
- [ ] GPU 인식 확인 (`nvidia-smi`)

#### 프로젝트 설정
- [ ] 프로젝트 파일 복사/클론
- [ ] requirements.txt 설치
- [ ] TensorFlow GPU 동작 확인

#### 개발 도구
- [ ] Jupyter Notebook 설정
- [ ] 원격 접속 설정 (선택)
- [ ] VSCode Remote SSH 설정 (선택)

### Phase 2: 실시간 처리

- [ ] Redis 설치 및 설정
- [ ] PostgreSQL 설치 및 설정
- [ ] Kafka 설치 (필요시)
- [ ] 데이터베이스 스키마 생성

### Phase 3: 프로덕션 환경

- [ ] FastAPI 서버 구축
- [ ] Nginx 리버스 프록시 설정
- [ ] SSL 인증서 설치 (Let's Encrypt)
- [ ] Docker 컨테이너화
- [ ] Docker Compose 오케스트레이션

### Phase 4: MLOps

- [ ] MLflow 서버 구축
- [ ] Prometheus 설치
- [ ] Grafana 대시보드 설정
- [ ] 로깅 시스템 구축

### Phase 5: 보안 & 운영

- [ ] 방화벽 설정
- [ ] SSH 보안 강화
- [ ] 백업 자동화
- [ ] 모니터링 알림 설정

---

## 8. 예상 비용

### 8.1 클라우드 서버 (월 비용)

#### AWS EC2

| 인스턴스 타입 | vCPU | RAM | GPU | 스토리지 | 월 비용 (USD) | 적합도 |
|--------------|------|-----|-----|----------|--------------|--------|
| g4dn.xlarge | 4 | 16GB | T4 (16GB) | 125GB SSD | ~$350 | 개발/소규모 |
| g4dn.2xlarge | 8 | 32GB | T4 (16GB) | 225GB SSD | ~$600 | 프로덕션 |
| p3.2xlarge | 8 | 61GB | V100 (16GB) | 100GB SSD | ~$900 | 대규모 학습 |

#### Google Cloud Platform (GCP)

| 인스턴스 타입 | vCPU | RAM | GPU | 월 비용 (USD) | 적합도 |
|--------------|------|-----|-----|--------------|--------|
| n1-standard-4 + T4 | 4 | 15GB | T4 | ~$300 | 개발 |
| n1-standard-8 + T4 | 8 | 30GB | T4 | ~$500 | 프로덕션 |
| n1-highmem-8 + V100 | 8 | 52GB | V100 | ~$1,200 | 대규모 |

#### Microsoft Azure

| 인스턴스 타입 | vCPU | RAM | GPU | 월 비용 (USD) | 적합도 |
|--------------|------|-----|-----|--------------|--------|
| NC4as T4 v3 | 4 | 28GB | T4 | ~$400 | 개발 |
| NC6s v3 | 6 | 112GB | V100 | ~$900 | 프로덕션 |

**참고**:
- 예약 인스턴스 사용 시 30-50% 할인
- Spot/Preemptible 인스턴스 사용 시 최대 90% 할인 (단, 중단 가능)

### 8.2 온프레미스 서버 (초기 구축 비용)

#### 기본 GPU 서버

| 구성요소 | 사양 | 가격 (USD) |
|---------|------|-----------|
| CPU | AMD Ryzen 9 5900X (12코어) | $400 |
| RAM | 64GB DDR4 | $200 |
| GPU | RTX 4090 24GB | $1,600 |
| 메인보드 | X570 | $200 |
| SSD | 1TB NVMe | $100 |
| 전원 | 1000W 80+ Gold | $150 |
| 케이스 | ATX | $100 |
| **합계** | | **$2,750** |

#### 프로덕션 서버

| 구성요소 | 사양 | 가격 (USD) |
|---------|------|-----------|
| CPU | AMD EPYC 7402P (24코어) | $1,000 |
| RAM | 128GB ECC DDR4 | $600 |
| GPU | RTX A6000 48GB (x2) | $8,000 |
| 메인보드 | Server Motherboard | $500 |
| SSD | 2TB NVMe (x2) | $400 |
| 전원 | 1600W Redundant | $400 |
| 케이스 | 4U Rack | $300 |
| **합계** | | **$11,200** |

**운영 비용 (월)**:
- 전기료: ~$50-100 (GPU 서버 24시간 가동)
- 인터넷: ~$100 (전용선)
- 냉각/관리: ~$50

### 8.3 비용 비교 (1년 기준)

| 옵션 | 초기 비용 | 월 비용 | 1년 총 비용 | 장점 | 단점 |
|------|----------|--------|------------|------|------|
| **클라우드 (AWS g4dn.xlarge)** | $0 | $350 | $4,200 | 관리 편리, 확장 쉬움 | 장기적으로 비쌈 |
| **온프레미스 (기본)** | $2,750 | $100 | $3,950 | 장기적으로 저렴 | 초기 투자, 관리 필요 |
| **온프레미스 (프로덕션)** | $11,200 | $150 | $13,000 | 고성능, 독립성 | 높은 초기 비용 |
| **하이브리드** | $2,750 | $200 | $5,150 | 개발은 로컬, 학습은 클라우드 | 복잡한 관리 |

**권장사항**:
- **스타트업/실험 단계**: 클라우드 (Spot 인스턴스)
- **프로토타입 완성 후**: 온프레미스 (기본 GPU 서버)
- **프로덕션 배포**: 온프레미스 + 클라우드 백업

---

## 9. 최종 권장 구성

### 9.1 개발 환경 (현재 단계)

```yaml
OS: Ubuntu 22.04 LTS
Python: 3.10
GPU: NVIDIA RTX 3060 (12GB) 또는 T4
RAM: 32GB
Storage: 500GB SSD

핵심 패키지:
  - tensorflow-gpu==2.13.0
  - pandas==2.0.3
  - numpy==1.24.3
  - matplotlib==3.7.2
  - jupyter==1.0.0

예상 비용:
  - 클라우드: $300-400/월
  - 온프레미스: $2,000-3,000 초기
```

### 9.2 프로덕션 환경 (최종 목표)

```yaml
OS: Ubuntu 22.04 LTS
Python: 3.10
GPU: NVIDIA RTX 4090 (24GB) x2
RAM: 64GB
Storage: 1TB NVMe SSD

추가 구성:
  - Docker + Docker Compose
  - FastAPI 서버
  - PostgreSQL + Redis
  - Nginx 리버스 프록시
  - Prometheus + Grafana
  - MLflow

예상 비용:
  - 클라우드: $800-1,200/월
  - 온프레미스: $6,000-8,000 초기
```

---

## 10. 참고 자료

### 공식 문서
- TensorFlow GPU 설정: https://www.tensorflow.org/install/gpu
- CUDA 설치 가이드: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- FastAPI 문서: https://fastapi.tiangolo.com/
- Docker 문서: https://docs.docker.com/

### 커뮤니티
- TensorFlow Forum: https://discuss.tensorflow.org/
- Stack Overflow: https://stackoverflow.com/questions/tagged/tensorflow

### 벤치마크
- GPU 성능 비교: https://www.techpowerup.com/gpu-specs/
- MLPerf 벤치마크: https://mlcommons.org/benchmarks/

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 | 작성자 |
|------|------|-----------|--------|
| 2025-10-16 | 1.0 | 초안 작성 | - |

---

**문서 상태**: ✅ 완료
**다음 업데이트**: Phase 2 진입 시 실시간 처리 스택 상세화
