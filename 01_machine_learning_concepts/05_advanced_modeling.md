# 표현 학습과 고급 모델링 기법

## 1. 표현 학습 (Representation Learning)

### 1.1 표현 학습의 개념

- **정의**: 
  - 머신러닝 알고리즘이 **유용하고 의미 있는 표현(Representation)**을 데이터로부터 스스로 학습하는 방법론
  - 단순 예측을 넘어 데이터의 본질적인 구조와 패턴을 포착하는 과정

- **유용한 표현의 특성**:
  - **해석 가능성(Interpretability)**: 의미 있는 해석이 가능한 특징
  - **잠재 특징(Latent Features) 포함**: 데이터에 숨겨진 구조적 특성
  - **전이 가능성(Transferability)**: 다른 작업에도 활용 가능한 일반화된 특징

- **딥러닝과 표현 학습**:
  - 딥러닝 모델(특히 심층 신경망)은 본질적으로 표현 학습 모델로 간주 가능
  - 입력 데이터를 여러 계층을 거치며 다른 부분 공간(Subspace)으로 투영(Project)하여 정보 인코딩
  - 학습된 표현은 후속 작업(Downstream Task)을 위한 선형 분류기 등의 입력으로 활용

### 1.2 심층 아키텍처의 장점

- **특징 재사용 (Re-use of Features)**: 
  - 낮은 계층에서 학습된 특징들이 높은 계층에서 재사용되어 효율적인 학습 가능
  - 이미지의 경우: 에지 → 질감 → 부분 → 객체 전체 등의 계층적 특징 학습

- **추상적 특징 학습 (Abstract Features)**: 
  - 계층이 깊어질수록 더 추상적이고 복잡한 특징 학습
  - 추상적 특징은 입력 데이터의 지역적 변화(Local Changes)에 덜 민감하고 강인한(Invariant) 특성 보유

### 1.3 표현의 종류

- **지역적 표현 (Local Representation)**: 
  - **원-핫 벡터 (One-hot Vector)**: 단 하나의 요소만 활성화되고 나머지는 0인 벡터
  - 각 개체를 독립적으로 표현하며, 유사성 개념이 존재하지 않음
  - 예시: 단어 사전에서 각 단어를 고유한 원-핫 벡터로 표현

- **분산 표현 (Distributed Representation)**:
  - **마이크로 특징 (Micro-features)**: 정보를 여러 뉴런/차원에 분산하여 표현
  - **저차원 임베딩 (Low-dimensional Embeddings)**: 이산적 변수를 저차원의 밀집된 연속 벡터로 표현
  - 특징: 밀집(Dense), 간결(Compact), 유사 데이터에 대한 일반화 성능 우수
  - 예시: 워드 임베딩(Word2Vec, GloVe), 이미지 특징 벡터

- **준-지역적 표현 (Semi-local Representation)**:
  - **멀티-핫 벡터 (Multi-hot Vector)**: 여러 요소가 동시에 활성화될 수 있는 특징 기반 표현
  - 예시: BoW(Bag of Words) 모델에서 문서의 단어 빈도 표현

## 2. 메트릭 학습 (Metric Learning)

### 2.1 메트릭(거리 함수)의 개념

- **메트릭 (Metric = Distance, 거리)**:
  - 집합 내 모든 원소 쌍 사이의 "거리"를 정량화하는 함수
  - 이를 통해 유사도(Similarity)를 측정 가능

- **메트릭 함수의 조건**: 함수 $f(x, y)$는 다음 속성 만족해야 함
  1. **비음수성 (Non-negativity)**: $f(x, y) \ge 0$
  2. **식별 불가능한 대상의 동일성**: $f(x, y) = 0 \iff x = y$
  3. **대칭성 (Symmetry)**: $f(x, y) = f(y, x)$
  4. **삼각 부등식 (Triangle Inequality)**: $f(x, z) \le f(x, y) + f(y, z)$

### 2.2 메트릭 학습의 종류

- **사전 정의된 메트릭 (Pre-defined Metrics)**:
  - 데이터에 대한 지식 없이 완전히 명시된 거리 함수
  - 예: 유클리드 거리 (Euclidean Distance): $f(x, y) = \sqrt{(x - y)^T(x - y)}$

- **학습된 메트릭 (Learned Metrics)**:
  - 데이터에 대한 지식을 통해서만 정의될 수 있는 거리 함수
  - 예: 마할라노비스 거리 (Mahalanobis Distance): $f(x, y) = \sqrt{(x - y)^T M (x - y)}$
    - $M$은 데이터로부터 추정되는 행렬(일반적으로 공분산 행렬의 역행렬 $\Sigma^{-1}$)
    - 데이터의 표준편차와 상관관계를 고려하여 유클리드 거리에 가중치 부여 (Chandra, 1936)

- **학습된 메트릭의 학습 방식**:
  - **비지도 메트릭 학습**: 레이블이 없는 데이터 사용
  - **지도 메트릭 학습**: 레이블이 있는 데이터 사용 (예: LDA - 선형 판별 분석)

### 2.3 딥러닝에서의 메트릭 학습

- **접근 방식**:
  - 딥러닝 모델(특히 CNN)을 사용하여 입력 데이터를 효과적인 임베딩 공간으로 매핑
  - 이 공간에서 좋은 거리 함수(메트릭)를 학습

- **응용 분야**:
  - 얼굴 인식/검증(Face Identification/Verification)
  - 이미지 검색(Image Retrieval)
  - 유사 아이템 추천(Similar Item Recommendation)

## 3. 메트릭 학습을 위한 손실 함수

### 3.1 메트릭 학습의 목표

- 같은 클래스에 속하는 샘플들은 가깝게, 다른 클래스에 속하는 샘플들은 멀리 떨어지도록 임베딩 공간을 학습

### 3.2 유클리드 거리 기반 손실 함수

- **대조 손실 (Contrastive Loss)**:
  - **개념**: Positive pair(같은 클래스) 간 거리는 작아지도록, Negative pair(다른 클래스) 간 거리는 특정 마진($\alpha$) 이상으로 커지도록 학습
  - **수식**: $L(A, B) = y \cdot d(A, B)^2 + (1 - y) \cdot \max(0, \alpha - d(A, B))^2$
    - $y$: 두 샘플이 같은 클래스면 1, 다른 클래스면 0
    - $d(A, B)$: 두 샘플 간 거리
  - **논문**: "Learning a similarity metric discriminatively, with application to face verification"
  - **문제점**: 마진 파라미터 설정 어려움, 모든 Negative sample에 동일 마진 적용으로 학습 효율 저하

- **삼중항 손실 (Triplet Loss)**:
  - **개념**: Anchor(기준 샘플), Positive(Anchor와 같은 클래스), Negative(Anchor와 다른 클래스) 세 개의 샘플 사용
  - **목표**: Anchor-Positive 간 거리($d(A,P)$)가 Anchor-Negative 간 거리($d(A,N)$)보다 특정 마진($\alpha$)만큼 작아지도록 학습
  - **수식**: $L(A, P, N) = \max(0, d(A,P) - d(A,N) + \alpha)$
  - **논문**: "Facenet: A unified embedding for face recognition and clustering"
  - **특징**: 절대적 거리 대신 **상대적 거리 차이**를 학습
  - **샘플 선택 중요성**: 구분하기 어려운 샘플(Hard Negative/Positive Mining) 선택 시 효과적 학습

- **센터 손실 (Center Loss)**:
  - **개념**: 각 클래스마다 중심(Center)을 두고, 같은 클래스 내 샘플들이 해당 클래스 중심으로 모이도록 학습
  - **수식**: $L_C = \frac{1}{2} \sum_{i=1}^{m} \|x_i - c_{y_i}\|_2^2$
    - $c_{y_i}$: 샘플 $x_i$의 클래스 레이블 $y_i$에 해당하는 클래스 중심
  - **논문**: "A discriminative feature learning approach for deep face recognition"
  - **단점**: 많은 GPU 메모리 필요, 각 ID(클래스)에 대한 개별적 학습 필요 가능성

### 3.3 각도/코사인 마진 기반 손실

- **SphereFace, CosFace, ArcFace**:
  - 유클리드 거리 대신 각도 또는 코사인 유사도에 마진을 적용하여 클래스 간 분별력 향상
  - **ArcFace**: 각도 공간에서 가산적 마진(Additive Angular Margin)을 적용
  - **논문**: "Arcface: Additive angular margin loss for deep face recognition", "Deep Hypersphere Embedding for Face Recognition"
  - **장점**: 크기 불변성(Scale Invariance)으로 인해 더 안정적인 학습

## 4. 유사도 학습 (Similarity Learning)

### 4.1 유사도 학습의 개념

- **정의**: 
  - 두 객체가 얼마나 유사하거나 관련되어 있는지를 측정하는 **유사도 함수(Similarity Function)** 학습
  - 주로 지도 학습 방식 사용

- **전통적인 매칭 기법의 한계**:
  - 특징 표현(Feature Representation)과 메트릭(Metric)이 함께 학습되지 않음
  - 예: SIFT/HOG 특징 추출 후 L1-norm으로 유사도 계산

### 4.2 샴 네트워크 (Siamese Network)

- **구조**: 
  - 동일한 구조와 가중치를 공유하는 두 개 이상의 CNN(또는 다른 신경망)으로 구성
  - 각 네트워크는 입력 객체(예: 이미지)를 받아 특징 벡터(임베딩) 추출
  - 추출된 특징 벡터들 사이의 거리(또는 유사도) 계산 및 손실 함수 기반 학습

- **응용 분야**: 
  - One-shot Image Recognition
  - Feature Point Descriptor 학습
  - 얼굴 인식(Face Recognition)
  - 서명 검증(Signature Verification)

- **샴 네트워크 변형**:
  - **Triplet Network**: 세 개의 입력을 동시에 비교(기준 샘플, 유사 샘플, 비유사 샘플)
  - **논문**: "Learning to compare image patches via convolutional neural networks"

### 4.3 샴 네트워크 학습 방법

- **가중치 업데이트**:
  - 두 (또는 그 이상) 스트림을 독립적으로 업데이트한 후 가중치를 평균내는 방식
  - RNN 학습과 유사점

- **데이터 증강**:
  - Random crops, Image flipping 등을 통해 학습 효과 향상

- **역전파 과정**:
  - 각 스트림의 출력에 대한 손실의 편미분($\partial l/\partial D(x_1)$, $\partial l/\partial D(x_2)$)을 계산하여 가중치 업데이트

## 5. CNN에서의 특징 맵 (Convolutional Feature Maps)

### 5.1 객체 탐지의 기본 요소

- **객체 탐지 (Object Detection) = "무엇(What)" + "어디에(Where)"**:
  - **인식(Recognition)**: 이미지 내 어떤 객체가 있는지 ("What")
  - **지역화(Localization)**: 그 객체가 이미지의 어디에 있는지 ("Where")

### 5.2 컨볼루션 특징 맵의 역할

- **컨볼루션(Convolutional)**: 슬라이딩 윈도우 연산
- **맵(Map)**: "어디에" 정보를 명시적으로 인코딩
- **특징(Feature)**: "무엇을" 인코딩 (암묵적으로 "어디에" 정보도 포함)

### 5.3 컨볼루션 레이어의 특성

- **지역적 연결성 (Locally Connected)**:
  - 필터가 이미지의 특정 영역에만 연결
  - 필터의 위치가 지역화 정보 제공

- **가중치 공유 (Spatially Shared Weights)**:
  - 이동 불변성(Translation-Invariant) 제공
  - 동일한 패턴이 이미지의 다른 위치에 나타나도 동일한 응답 생성

- **임의 크기 입력 처리**:
  - 입력 이미지 크기에 비례하는 출력 크기 생성

### 5.4 특징 맵의 이해

- **직관적 해석**:
  - 특정 채널의 특징 맵은 특정 패턴(예: 원형 물체, 특정 모양)에 강하게 반응
  - 특징 맵의 특정 위치에서의 반응은 해당 위치에 특정 패턴 존재 의미

- **시각화 기법**:
  - Deconvolution (또는 Transposed Convolution)을 통한 특징 맵 시각화
  - 네트워크가 무엇을 학습하는지 이해하는 데 도움 (Zeiler & Fergus, ECCV 2014)

- **수용 영역 (Receptive Field)**:
  - 특징 맵의 한 픽셀에 영향을 미치는 입력 이미지 상의 영역
  - 레이어가 깊어질수록 수용 영역 확대

### 5.5 Region-based CNN Features

- **R-CNN**:
  - 객체 후보 영역(Region Proposal)을 약 2000개 생성
  - 각 영역마다 CNN을 통과시켜 특징 추출 및 분류
  - 매우 느린 처리 속도

- **SPP-Net & Fast R-CNN**:
  - 전체 이미지에 대해 CNN을 한 번만 통과시켜 컨볼루션 특징 맵 획득
  - 특징 맵 위에서 후보 영역에 해당하는 부분 추출, 고정된 크기 특징 벡터 생성
  - **SPP (Spatial Pyramid Pooling) Layer**: 다양한 크기의 특징 맵 영역을 고정된 수의 빈(bin)으로 풀링
  - **RoI (Region of Interest) Pooling**: SPP의 단일 레벨 버전(예: 7x7로 풀링)
  - R-CNN보다 훨씬 빠른 처리 속도

### 5.6 단일 스케일 vs. 다중 스케일 특징 맵

- **이미지 피라미드 (Image Pyramid)**:
  - 입력 이미지를 여러 스케일로 리사이즈하여 각각 특징 맵 계산
  - 정확도는 높지만 처리 속도 느림

- **딥러닝 기반 단일 스케일 접근법**:
  - ImageNet 사전 학습 효과로 단일 스케일에서도 좋은 성능
  - 속도와 정확도 간 좋은 트레이드오프 제공
  - 최고 정확도를 위해서는 여전히 다중 스케일 필요

## 6. 특징 맵으로부터 영역 제안 (Region Proposal from Feature Maps)

### 6.1 Faster R-CNN의 접근법

- **기존 영역 제안 방식의 병목**:
  - Selective Search (2초/이미지), EdgeBoxes (0.2초/이미지) 등은 CNN 연산보다 느림

- **Faster R-CNN (Ren et al., NIPS 2015)**:
  - 영역 제안 과정도 CNN 특징 맵을 공유하여 실시간 객체 탐지 목표
  - **RPN (Region Proposal Network)**:
    - 컨볼루션 특징 맵 위를 슬라이딩 윈도우 방식으로 탐색
    - 각 위치에서 미리 정의된 앵커 박스 기준으로 객체 여부와 박스 위치 조정값 예측
    - 완전 컨볼루션 네트워크 구조, 탐지 네트워크와 특징 맵 공유로 End-to-End 학습 가능

- **앵커 박스 (Anchors)**:
  - 사전 정의된 참조 박스들
  - 위치 이동에 불변하며, 다양한 크기와 비율 보유
  - 단일 스케일 특징 맵에서 다중 스케일 예측 가능

### 6.2 Faster R-CNN의 효율성

- **특징 맵 공유의 이점**:
  - RPN과 탐지 네트워크가 특징 맵을 공유하여 매우 빠른 속도 달성
  - VGG-16 기준 약 200ms/이미지 처리 속도
  - 높은 정확도와 실시간성 동시 확보

## 7. Kaiming He의 연구와 딥러닝 발전

### 7.1 Kaiming He의 연구 성과

- **주요 연구**: 
  - ResNet, Mask R-CNN 등 컴퓨터 비전 분야 핵심 연구 주도
  - CVPR, ICCV 등 최고 학회에서 다수 논문 발표

### 7.2 "Revolution of Depth"

- **깊이의 혁명**: 
  - 딥러닝 모델의 깊이(레이어 수) 증가에 따른 혁신적 성능 향상
  - **발전 추이**: 
    - AlexNet (8 layers)
    - VGGNet (19 layers)
    - GoogLeNet (22 layers)
    - ResNet (152 layers)

### 7.3 He 초기화 (He Initialization)

- **초기화 방법**: 
  - `tf.keras.initializers.he_normal`, `tf.keras.initializers.he_uniform`

- **특징**:
  - ReLU 계열 활성화 함수에 적합한 가중치 초기화 방법
  - 학습 초기 그래디언트 소실/폭주 문제 완화
  - 깊은 네트워크 학습 안정화

## 핵심 요약

- **표현 학습**은 데이터로부터 유용하고 의미 있는 표현을 자동으로 학습하는 방법론으로, 딥러닝의 핵심 원리 중 하나
- **메트릭 학습**은 데이터 간 거리/유사도를 효과적으로 측정하는 함수를 학습하며, 얼굴 인식 등 다양한 분야에 활용
- **대조 손실, 삼중항 손실, 센터 손실** 등 다양한 손실 함수가 메트릭 학습에 활용되며, 각각 고유한 장단점 보유
- **샴 네트워크**는 가중치를 공유하는 동일한 구조의 네트워크를 통해 객체 간 유사도를 학습하는 효과적인 방법
- **CNN 특징 맵**은 "무엇"과 "어디에" 정보를 모두 인코딩하며, 객체 탐지에 핵심적인 역할 수행
- **Faster R-CNN**은 특징 맵 공유와 앵커 박스 개념을 통해 빠르고 정확한 객체 탐지를 실현
- **Kaiming He**의 깊이 혁명과 초기화 방법은 현대 딥러닝 발전에 중요한 기여
