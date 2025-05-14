# TensorFlow와 Keras 기초

## 1. TensorFlow와 Keras의 이해

### 1.1 TensorFlow의 설계 철학: 프로그레시브 디스클로저 (Progressive Disclosure)

- **개념:** 사용자의 숙련도에 맞춰 기능의 복잡도를 점진적으로 드러내는 방식
- **적용 방식:**
  - **초보자:** 몇 줄의 코드로 간단하게 모델 구현 가능
  - **중급자:** 함수형 API를 통해 복잡한 모델 구조 설계
  - **고급자:** 로우 레벨 API로 세밀한 제어 및 커스터마이징 가능
- **다양한 API 조합:**
  - 모델링 방법: Sequential, Functional, Subclassing
  - 학습 방법: `fit`, `train_on_batch`, `tf.GradientTape` 활용
  - 커스터마이징: 사용자 정의 층, 손실 함수, 옵티마이저 등

### 1.2 Keras의 역사와 역할

- **Theano 시대:** 초창기 딥러닝 프레임워크로, 사용이 복잡하고 어려움
- **Keras의 등장:** 이론과 코드 구현 간 간극을 줄이기 위한 고수준 API로 개발
  - 초기에는 Theano, TensorFlow 등 다양한 백엔드 지원
- **관련 프레임워크 발전:**
  - **Torch/PyTorch:** Lua 기반 Torch에서 Python 기반 PyTorch로 발전
  - **Chainer:** 동적 계산 그래프 개념 도입, Keras와 PyTorch에 영향
- **TensorFlow와의 통합:** TensorFlow의 핵심 API로 Keras 통합
- **Keras 3.0:** 백엔드 독립성 강화 (TensorFlow, PyTorch, JAX 등 지원)

### 1.3 TensorFlow와 PyTorch 비교

- **TensorFlow + Keras:**
  - 다양한 API 레벨과 접근 방식 제공 (약 18~20가지 조합 가능)
  - 시스템화(배포, 서빙)에 강점
  - C++, Java 등으로 모델 포팅 지원
- **PyTorch:**
  - 상대적으로 단순한 모델링 및 학습 방식 (약 2가지 주요 방식)
  - 초기 학습 곡선이 완만하고 직관적
  - 연구와 실험에 유연한 구조

## 2. Keras를 이용한 모델 구축

### 2.1 Sequential API 기초

```python
import tensorflow as tf

# Sequential 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 입력층 (이미지를 1차원 벡터로 변환)
    tf.keras.layers.Dense(128, activation='relu'),  # 은닉층 (128개 뉴런, ReLU 활성화)
    tf.keras.layers.Dense(10)                       # 출력층 (10개 클래스)
])
```

- `tf.keras.Sequential`은 `tf.keras.models.Sequential`의 약칭(alias)
- 레이어를 순차적으로 쌓아 간단한 모델을 구성하는 방법
- 데이터가 일방향으로 흐르는 단순한 구조에 적합

### 2.2 레이어의 동작 원리

- **레이어 객체 생성과 호출:**
  ```python
  flatten_layer = tf.keras.layers.Flatten(input_shape=(28, 28))
  flattened_data = flatten_layer(input_data)  # __call__ 메서드 실행
  ```

- **모델의 함수적 동작:**
  ```python
  # 모델 전체를 함수처럼 사용하여 순전파 수행
  output = model(input_data)  # TensorFlow 텐서 반환
  ```

- **예측 수행:**
  ```python
  # predict 메서드를 사용한 예측 (배치 처리 포함)
  predictions = model.predict(input_data)  # NumPy 배열 반환
  ```

## 3. 모델 구성 요소 상세

### 3.1 Dense 레이어 (완전 연결 레이어)

```python
tf.keras.layers.Dense(
    units,                    # 출력 뉴런 수
    activation=None,          # 활성화 함수
    use_bias=True,            # 편향 사용 여부
    kernel_initializer='glorot_uniform',  # 가중치 초기화 방법
    bias_initializer='zeros',  # 편향 초기화 방법
    kernel_regularizer=None,  # 가중치 정규화
    bias_regularizer=None,    # 편향 정규화
    activity_regularizer=None  # 활성화 값 정규화
)
```

### 3.2 모델링 시 고려 사항

1. **볼륨 (Volume):** 
   - 레이어 종류, 개수, 뉴런 수 등 모델의 전체적인 크기
   - 문제의 복잡도에 맞게 적절한 크기 설정 필요

2. **옵션 (Options):**
   - 각 레이어의 하이퍼파라미터 (활성화 함수, 정규화 등)
   - 성능 향상을 위한 세부 옵션 조정

### 3.3 활성화 함수 지정 방식

- **문자열 지정 방식 (간편):**
  ```python
  tf.keras.layers.Dense(128, activation='relu')
  ```

- **함수 객체 직접 지정 (명시적):**
  ```python
  tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)
  ```

- **`model.summary()`:** 모델 구조와 파라미터 수 요약 출력

## 4. 모델 컴파일 (학습 설정)

### 4.1 컴파일 기본 개념

```python
model.compile(
    optimizer='adam',  # 최적화 알고리즘
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 손실 함수
    metrics=['accuracy']  # 평가 지표
)
```

- 학습을 시작하기 전 **학습 프로세스를 설정**하는 단계
- TensorFlow가 계산 그래프를 생성하고 최적화하는 과정

### 4.2 주요 인자 상세

#### 옵티마이저 (Optimizer)

- **문자열 지정 (기본 설정):**
  ```python
  optimizer='adam'
  ```

- **객체 지정 (세부 설정):**
  ```python
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
  ```

#### 손실 함수 (Loss Function)

- **문자열 지정:**
  ```python
  loss='sparse_categorical_crossentropy'
  ```

- **객체 지정:**
  ```python
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  ```

- **`from_logits` 매개변수:**
  - `True`: 모델 출력이 로짓(활성화 함수 적용 전) 형태일 때
  - `False`: 모델 출력이 확률(소프트맥스 적용 후) 형태일 때

#### 메트릭스 (Metrics)

- 학습 및 평가 과정을 모니터링할 지표 설정
  ```python
  metrics=['accuracy', 'precision', 'recall']
  ```

## 5. 모델 학습 및 평가

### 5.1 `fit` 메서드 기본 사용법

```python
history = model.fit(
    X_train, y_train,         # 학습 데이터 (입력 및 타겟)
    epochs=10,                # 전체 데이터셋 반복 학습 횟수
    batch_size=32,            # 그래디언트 업데이트 당 샘플 수
    validation_split=0.2,     # 검증 데이터 비율 (또는 validation_data=(X_val, y_val))
    callbacks=[...]           # 콜백 함수들 (조기 종료, 체크포인트 등)
)
```

### 5.2 학습 과정

1. 데이터를 배치 단위로 나누기
2. 각 배치마다 순전파 수행
3. 손실 계산
4. 역전파를 통한 그래디언트 계산
5. 가중치 업데이트
6. 전체 과정을 지정된 에포크만큼 반복

### 5.3 학습 결과 (History 객체)

```python
# 학습 곡선 시각화
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.grid(True)
plt.show()
```

### 5.4 모델 평가

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"테스트 정확도: {test_accuracy:.4f}")
```

## 6. 소프트맥스와 로짓 (from_logits) 이해하기

### 6.1 이론과 실제 구현의 차이

- **이론적 구현:**
  ```python
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation='softmax')  # 소프트맥스 명시적 사용
  ])
  
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
  )
  ```

- **수치적으로 안정적인 구현:**
  ```python
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(10)  # 활성화 함수 미지정 (로짓 출력)
  ])
  
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  )
  ```

### 6.2 예측 시 소프트맥스 적용

```python
# 로짓 출력을 사용하는 모델에서 예측 시 소프트맥스 적용
probability_model = tf.keras.Sequential([
    model,  # 기존 모델 (로짓 출력)
    tf.keras.layers.Softmax()  # 소프트맥스 레이어 추가
])

predictions = probability_model.predict(X_test)
```

## 7. 로우 레벨 커스터마이징 (GradientTape)

```python
# 커스텀 학습 루프 예시
@tf.function  # 계산 그래프 컴파일로 성능 향상
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_function(targets, predictions)
    
    # 모델 변수에 대한 그래디언트 계산
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # 옵티마이저를 사용하여 변수 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss
```

## 8. Fashion MNIST 실습 예제

### 8.1 데이터 준비

```python
# 데이터 로드
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 데이터 전처리 (정규화)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 클래스 이름 설정
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

### 8.2 모델 구축 및 학습

```python
# 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)  # 로짓 출력
])

# 컴파일
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 학습
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.1
)
```

### 8.3 모델 평가 및 예측

```python
# 테스트셋 평가
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\n테스트 정확도: {test_acc:.4f}')

# 예측 수행
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(X_test)
```

## 핵심 요약

- TensorFlow와 Keras는 **프로그레시브 디스클로저** 원칙에 따라 다양한 수준의 API 제공
- `Sequential` 모델은 간단한 구조의 모델을 쉽게 구현할 수 있게 함
- 모델 구축 후 **컴파일**을 통해 최적화 방법, 손실 함수, 평가 지표 설정
- `fit` 메서드로 모델 학습, `evaluate`로 평가, `predict`로 예측 수행
- 소프트맥스 활성화 함수는 `from_logits=True` 옵션과 함께 수치적 안정성을 고려하여 적용
- 로우 레벨 커스터마이징을 위해 `tf.GradientTape` 사용 가능
