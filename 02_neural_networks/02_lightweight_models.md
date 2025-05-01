# 최신 경량 딥러닝 모델 동향

## 1. 딥러닝과 AI 시대의 도래

- **CNN (Convolutional Neural Network):** 이미지 인식 분야의 혁신을 가져온 핵심 기술 (LeNet, 1989). Convolution layer를 주요 구성 요소로 사용
- **AlexNet (2012):** GPU를 활용한 Deep CNN 학습 성공. ReLU, Dropout 등 새로운 기법 도입. ImageNet 분류 대회(ILSVRC)에서 압도적인 성능 향상을 보이며 AI 시대 개막의 신호탄이 됨
- **현재:** AI 기술은 물체 검출(Object Detection), 이미지 분할(Image Segmentation) 등 다양한 분야로 확장되어 실생활에 적용되고 있음

## 2. 딥러닝 모델 학습 개괄 (Backbone 중심)

- **Backbone (백본 모델):** 대규모 데이터셋(예: ImageNet)으로 미리 학습된(Pretrained) 딥러닝 모델. 주로 **특징 추출기(Feature Extractor)** 역할 수행 (VGG16, ResNet 등)
- **Transfer Learning (전이 학습):** 백본 모델의 학습된 가중치(Weight)를 가져와서, 새로운 목표 작업(Target Task)의 데이터셋으로 학습을 시작하는 기법
    - **핵심:** 남이 잘 만들어 놓은 특징 추출 능력을 빌려 쓰는 것
    - **구현:** 가져온 백본의 일부 레이어는 고정(Freezing)하고, 새로운 데이터에 맞게 추가된 뒷부분 레이어만 학습시키거나, 전체 레이어를 미세 조정
- **Finetuning (미세 조정):** Transfer Learning 과정에서 백본의 가중치를 고정하거나 아주 작은 학습률로 업데이트하면서, 목표 작업에 맞게 모델을 **미세하게 조정**하는 과정

## 3. 서비스 적용 및 주요 이슈

- **실 서비스 적용 워크플로우:**
    1. 대규모 데이터셋 확보 (e.g., ImageNet)
    2. 딥러닝 모델(백본) 선택 및 Pretrain (또는 공개된 Pretrained 모델 사용)
    3. 백본 성능 평가 (e.g., ImageNet 분류 성능)
    4. 목표 데이터셋으로 Finetuning
    5. 목표 작업 성능 평가 (e.g., Object Detection mAP)
- **주요 고려 사항:**
    - **데이터셋:** 어떤 대규모 데이터로 Pretrain 할 것인가? (ImageNet이 가장 보편적)
    - **모델 선택:** 어떤 백본 모델을 사용할 것인가? (Model-zoo 활용 또는 신규 개발)
    - **백본 성능:** Pretrained 백본의 성능이 Target Task 성능에 큰 영향을 미침
    - **트레이닝 세팅:** Optimizer, Learning rate, Weight decay, Epoch, Augmentation 등 학습 방법을 정밀하게 최적화해야 함

## 4. 최신 가벼운(Lightweight) 딥러닝 모델 동향 및 이슈

- **개발 배경:** 모바일 등 제한된 환경에서의 효율적 구동 필요성 증대
- **핵심 기술:** **Depthwise Separable Convolution** (MobileNetV1에서 첫 등장) - Regular Convolution 대비 파라미터 수와 연산량(FLOPs) 대폭 감소
- **유명 모델:** 
    - SqueezeNet
    - MobileNet (V1/V2/V3)
    - ShuffleNet (V1/V2)
    - MNasNet, FBNet
    - EfficientNet 등
- **이슈:**
    - **성능 재현의 어려움:** 논문에서 중요 세팅 미공개, 필요한 GPU 수 등 환경적 요인으로 리포트된 성능 재현 어려움
    - **Finetuning 성능 저하:** ImageNet 분류 성능은 좋으나, Detection 등 다른 Task로 Finetuning 시 성능이 기대만큼 나오지 않는 경우 발생
    - **실 서비스 고려사항:**
        - **성능:** 백본 성능 & Finetuning 후 Task 성능 모두 중요
        - **속도:** FLOPs가 낮다고 항상 빠른 것은 아님. 플랫폼/하드웨어(GPU/CPU)에 따라 특정 연산자 속도 다름
        - **모델 크기:** 파라미터 수. GPU 메모리 점유율 및 배치 사이즈 결정에 영향

## 5. 새로운 가벼운 모델 개발

- **설계 배경:** 
    - 기존 경량 모델들의 Finetuning 성능 저하 문제 해결
    - NAS 기반 모델의 ImageNet 과적합 문제 해결

- **설계 목적:**
    - 구조적 문제 해결
    - ImageNet 분류 SOTA 달성 (경량 모델 중)
    - CPU 추론 성능 확보 (GPU에서도 준수)
    - **Finetuning 성능 극대화 (Transferability)**
    - NAS 등 자동 모델 탐색 방법의 Baseline 제공

- **설계 방법 ("모델 깎기"):**
    - 성능 병목 분석 후 모델 축소 (FLOPs/속도 확보)
    - **가설:** Finetuning 성능 저하는 지나친 FLOPs 효율화 추구로 인한 **초기 레이어의 특징(feature) 수 부족** 때문
    - **개선:** 후반 레이어 특징 수를 줄여 초기 레이어 특징 수 보강

- **개발 프로세스:**
    1. 성능 재현
    2. 비교 모델 재현
    3. 신 모델 연구(설계/실험 반복)
    4. 트레이닝 방법 최적화
    5. 추가 모듈 검토
    6. Finetuning 성능 검증

- **세부 기술:**
    - MobileNet 기반 구조
    - ResNet 방식 Skip Connection
    - SE-Net 사용
    - 속도 저하 연산자 배제 (Pooling, Upsample 등)

- **성능 결과:**
    - **ImageNet:** 기존 경량 모델보다 **우수한 Top-1/Top-5 정확도** 달성
    - **Object Detection (SSD-lite):** 최신 SOTA 경량 백본들보다 **월등히 좋은 mAP** 달성
    - **Scene Text Detection:** 기존 모델 대비 파라미터 **1/9 수준**으로 줄이고도 **동일한 성능** 유지

## 6. 추가 팁 및 결론

- **가벼운 모델 트레이닝 유의점:**
    - SGD Optimizer 선호 (ADAM류보다 분류 성능 우수). Nesterov + Momentum 추천
    - Learning rate, Weight decay 등 파라미터 민감 → 스터디 필요
    - 학습 곡선(Training Curve) 반드시 저장/확인
    - Data Augmentation 강도 조절 필요 (약하게)

- **모델 성능 재현:** 최적의 트레이닝 세팅을 찾으면 논문보다 더 좋은 성능을 얻을 수도 있음
- **일반화 성능 향상:** CutMix (ICCV 2019) 등 강력한 Regularization 기법 도입 고려

## 7. 모델 추천

- **무거운 모델 (성능 중시):**
    - VGG16
    - ResNet50-SE
    - ResNeXt101-SE+FPN
    - Xception 계열
    - EfficientNet-B4 이상

- **가벼운 모델 (속도/효율 중시):**
    - ResNet-18
    - Xception(축소형)
    - MobileNetV1
    - 최신 경량화 모델

---

## 핵심 요약

- 모바일 등 실 서비스 환경에서는 **성능, 속도, 모델 크기**를 모두 고려한 **가벼운(Lightweight) 딥러닝 모델**이 필수적
- **Depthwise Separable Convolution** 등의 기술로 경량화가 이루어졌지만, 기존 모델들은 **성능 재현성**이나 **Finetuning 성능**에 이슈가 있었음
- 성공적인 모델 개발 및 적용을 위해서는 단순히 모델 구조뿐만 아니라 **데이터셋 선택, Pretraining의 중요성, Finetuning 성능 검증, 트레이닝 세팅 최적화** 등 전 과정을 체계적으로 이해하고 관리하는 것이 중요
