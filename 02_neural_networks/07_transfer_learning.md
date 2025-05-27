# 전이 학습 (Transfer Learning)과 사전 학습 모델 활용

## 1. 전이 학습의 개념 및 중요성

### 1.1 전이 학습의 정의

- **전이 학습(Transfer Learning)이란**:
  - 이미 **대규모 데이터셋(예: ImageNet)으로 학습된 모델의 지식(가중치, 구조 등)을 가져와서 새로운, 관련된 작업(Task)에 적용**하는 기법
  - 학습된 특징(Feature)과 패턴을 재활용하여 새로운 문제 해결 방식

### 1.2 전이 학습의 중요성

- **자원 효율성**:
  - 대규모 데이터셋으로 모델을 처음부터 학습시키는 것은 많은 시간, 계산 자원, 인력 필요
  - 사전 학습된 모델 활용 시 **더 적은 데이터로, 더 빠르게, 더 좋은 성능** 달성 가능

- **지식 재활용**:
  - 이미지의 일반적 특징(선, 모서리, 질감 등)은 다양한 시각 작업에 공통적으로 필요
  - 이러한 기본 특징 추출 능력을 재활용하면 새 작업에서 효율적 학습 가능

### 1.3 전이 학습이 효과적인 상황

- **데이터 부족**: 목표 작업의 레이블링된 데이터가 제한적일 때
- **계산 자원 제약**: GPU 등 고성능 컴퓨팅 자원이 제한적일 때 
- **빠른 개발 필요**: 제한된 시간 내 모델 개발이 필요한 경우
- **유사 도메인**: 사전 학습된 모델의 원본 데이터와 목표 작업의 데이터가 유사한 경우

## 2. 전이 학습의 주요 접근 방식

### 2.1 특징 추출기(Feature Extractor)로 사용

- **개념**: 
  - 사전 학습된 모델의 컨볼루션 베이스(Convolutional Base, Backbone) 부분만 가져와 특징 추출
  - 그 위에 새로운 분류기(Classifier, Head)를 추가하여 학습

- **구현 방식**:
  1. 사전 학습 모델에서 분류 레이어(Top/Head) 제거
  2. 남은 베이스 네트워크의 가중치를 고정(Freeze)
  3. 새로운 분류 레이어 추가
  4. 새로운 데이터셋으로 새 분류 레이어만 학습

- **장점**:
  - 빠른 학습 시간 (적은 수의 파라미터만 학습)
  - 적은 학습 데이터로도 효과적

- **단점**:
  - 기존 특징에 의존하므로 새 작업에 특화된 특징 학습 제한

```python
# TensorFlow에서 특징 추출기로 사용하는 예시
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',  # ImageNet 사전 학습 가중치 사용
    include_top=False,   # 분류기(Top) 제외
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # 기본 모델 가중치 고정

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

### 2.2 미세 조정(Fine-tuning)

- **개념**:
  - 사전 학습된 모델의 일부 또는 전체 레이어 가중치를 새로운 데이터셋에 맞게 재학습
  - 보통 모델의 뒷부분(상위 레이어)부터 점진적으로 앞부분까지 조정

- **구현 방식**:
  1. 사전 학습 모델에서 분류 레이어 제거 또는 교체
  2. 새 분류 레이어 학습 (베이스 모델은 고정)
  3. 베이스 모델의 상위 레이어 일부 또는 전체를 해제하고 낮은 학습률로 재학습

- **장점**:
  - 목표 작업에 더 특화된 특징 학습 가능
  - 일반적으로 특징 추출보다 높은 성능

- **단점**:
  - 더 많은 계산 자원과 학습 데이터 필요
  - 과적합 위험성 (특히 데이터가 적을 때)

```python
# TensorFlow에서 미세 조정을 수행하는 예시
# 1단계: 새 분류 레이어만 먼저 학습
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 첫 단계 학습 (분류 레이어만)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=5)

# 2단계: 상위 레이어 미세 조정
base_model.trainable = True
# 하위 100개 레이어는 고정하고 상위 레이어만 미세 조정
for layer in base_model.layers[:100]:
    layer.trainable = False

# 낮은 학습률로 미세 조정
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, epochs=10)
```

## 3. 딥러닝 프레임워크의 사전 학습 모델 제공 방식

### 3.1 소스 코드와 함께 제공 방식

- **제공 형태**:
  - 모델의 전체 구조(소스 코드)와 사전 학습된 가중치를 함께 제공
  - TensorFlow: `tf.keras.applications`
  - PyTorch: `torchvision.models`

- **장점**:
  - 모델 내부 구조를 상세히 확인 가능
  - 레이어 추가, 제거, 수정 등 **자유로운 커스터마이징** 가능
  - 학습 데이터, 가중치 로드 여부 등을 선택적으로 활용 가능
  - 학습 및 연구 목적에 적합

### 3.2 백본(Backbone)/모델 허브(Model Hub) 제공 방식

- **제공 형태**:
  - 모델의 소스 코드 직접 노출 없이, 학습된 모델(주로 백본) 로드 기능 제공
  - TensorFlow Hub, Hugging Face 등 플랫폼 활용

- **장점**:
  - 간소화된 코드로 강력한 사전 학습 모델 빠르게 활용
  - 백본 뒤에 새로운 레이어 추가하여 특정 작업에 맞게 파인튜닝 용이
  - 모델 제공자가 핵심 기술을 숨기면서도 기능 공개 가능 (블랙박스 활용)

## 4. TensorFlow를 이용한 사전 학습 모델 활용

### 4.1 사용 가능한 모델 탐색

- **`tf.keras.applications` 모듈**:
  ```python
  import tensorflow as tf
  print(dir(tf.keras.applications))
  ```
  - 제공 모델: `ConvNeXt`, `DenseNet`, `EfficientNet`, `InceptionResNetV2`, `MobileNet`, `ResNet`, `VGG`, `Xception` 등
  - 모바일 환경용 경량화 모델(예: `MobileNet`) 포함

- **KerasCV**:
  ```python
  # 설치
  !pip install -U keras-cv
  
  # 임포트
  import keras_cv
  ```
  - TensorFlow 생태계의 컴퓨터 비전 관련 최신 모델 제공
  - 최신 레이어, 손실 함수, 데이터 증강 기법 포함

### 4.2 모델 로드 및 활용

1. **모델 및 전처리 함수 임포트**:
   ```python
   from tensorflow.keras.applications import MobileNetV2
   from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
   ```

2. **모델 인스턴스 생성**:
   ```python
   # 전체 모델 로드
   model = MobileNetV2(weights='imagenet')
   
   # 백본만 로드 (분류기 제외)
   backbone = MobileNetV2(weights='imagenet', include_top=False)
   ```
   - `include_top=False`: 모델의 마지막 분류기 부분 제외

3. **데이터 전처리 (중요!)**:
   ```python
   # 이미지 전처리
   processed_img = preprocess_input(img)
   ```
   - 사전 학습된 모델은 특정 방식으로 전처리된 입력 데이터 필요
   - 적절한 전처리 없이는 성능 크게 저하될 수 있음

4. **소스 코드 확인**:
   ```python
   # 모델 정의 확인
   tf.keras.applications.MobileNetV2?
   ```
   - 모델 이해 및 필요 시 수정을 위해 소스 코드 확인 중요

## 5. PyTorch를 이용한 사전 학습 모델 활용

### 5.1 TorchVision 모델

- **사용 가능한 모델 탐색**:
  ```python
  import torch
  import torchvision
  print(dir(torchvision.models))
  ```
  - 제공 모델: `alexnet`, `resnet50`, `vgg16` 등

- **모델 로드 및 활용**:
  ```python
  from torchvision.models import resnet50, ResNet50_Weights
  
  # 최신 API - 가중치 객체 사용
  weights = ResNet50_Weights.DEFAULT  # 또는 특정 버전: ResNet50_Weights.IMAGENET1K_V2
  model = resnet50(weights=weights)
  
  # 가중치 없이 랜덤 초기화
  model_random = resnet50(weights=None)
  
  # 모델 구조 확인
  print(model)
  ```

- **데이터 전처리**:
  ```python
  # 가중치에 맞는 전처리 변환 가져오기
  preprocess = weights.transforms()
  
  # 이미지에 전처리 적용
  img_transformed = preprocess(img)
  ```

- **추론 모드 설정**:
  ```python
  # 추론(평가) 모드로 설정
  model.eval()
  ```
  - 드롭아웃, 배치 정규화 등의 레이어가 올바르게 동작하도록 설정

### 5.2 PyTorch Hub

- **Hub를 통한 모델 로드**:
  ```python
  # TorchVision 설치 없이도 모델 로드 가능
  model = torch.hub.load('pytorch/vision', 'resnet50', weights='DEFAULT')
  ```

## 6. 전이 학습의 실전 패턴 및 권장 사항

### 6.1 과적합 방지 전략

- **데이터 증강(Data Augmentation)** 적극 활용
- **드롭아웃(Dropout)** 및 **배치 정규화(Batch Normalization)** 적용
- 낮은 학습률 사용 (특히 미세 조정 시)
- **점진적 학습 (Progressive Learning)**
  1. 먼저 새 분류기만 학습
  2. 그 후 상위 레이어부터 점진적으로 미세 조정

### 6.2 최적의 모델 선택 기준

- **데이터셋의 크기와 유사성**:
  - 데이터가 적고 원본과 유사: 특징 추출 방식
  - 데이터가 많고 원본과 다름: 미세 조정 방식

- **모델의 크기와 복잡도**:
  - 계산 자원 제약 상황: MobileNet, EfficientNet 등 경량 모델
  - 최고 성능 필요: ResNet, DenseNet 등 대형 모델

- **전이 학습의 깊이**:
  - 데이터셋이 작고 원본과 유사: 상위 레이어 일부만 미세 조정
  - 데이터셋이 크고 원본과 크게 다름: 더 많은 레이어 미세 조정

### 6.3 프레임워크 선택 시 고려사항

- **TensorFlow**:
  - 생산 환경 배포에 강점
  - TensorFlow Serving, TensorFlow Lite 등 배포 도구 다양
  - Keras API의 높은 추상화 수준으로 빠른 개발

- **PyTorch**:
  - 동적 계산 그래프로 디버깅 용이
  - 연구 및 실험에 유연함
  - 낮은 수준의 제어 가능

## 7. 대표적인 사전 학습 모델 소개

### 7.1 컴퓨터 비전 모델

- **VGG (Visual Geometry Group)**:
  - 단순하고 균일한 구조로 이해하기 쉬움
  - 3x3 컨볼루션 필터 일관적 사용

- **ResNet (Residual Network)**:
  - 잔차 연결(Residual Connection)로 깊은 네트워크 학습 안정화
  - 다양한 크기(ResNet-18/34/50/101/152) 제공

- **EfficientNet**:
  - 네트워크 깊이, 너비, 해상도 균형 조정으로 효율성 극대화
  - 모바일부터 서버까지 다양한 환경에 적합한 변형 제공

- **MobileNet**:
  - 모바일 기기 등 제한된 자원에서 실행 가능한 경량 모델
  - 깊이별 분리 컨볼루션(Depthwise Separable Convolution) 사용

### 7.2 자연어 처리 모델

- **BERT (Bidirectional Encoder Representations from Transformers)**:
  - 양방향 트랜스포머 기반의 언어 모델
  - 문맥을 고려한 단어 임베딩 제공

- **GPT (Generative Pre-trained Transformer)**:
  - 자기회귀(Autoregressive) 방식의 생성형 언어 모델
  - 텍스트 생성, 요약, 번역 등에 활용

## 핵심 요약

- **전이 학습**은 사전 학습된 모델의 지식을 재활용하여 **적은 데이터와 계산 자원으로 효율적인 모델 개발** 가능
- 주요 접근법: **특징 추출**(Feature Extraction)과 **미세 조정**(Fine-tuning)
- 딥러닝 프레임워크는 **소스 코드 제공** 방식과 **모델 허브** 방식으로 사전 학습 모델 지원
- **데이터 전처리**는 사전 학습 모델 활용 시 매우 중요한 요소
- 모델 선택과 학습 전략은 **데이터셋 특성과 활용 목적**에 따라 결정
- 현대 딥러닝 응용에서 전이 학습은 선택이 아닌 필수적 기법
