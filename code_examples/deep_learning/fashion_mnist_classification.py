import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 준비 (Data Preparation)
print("Fashion MNIST 데이터셋 로딩...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 클래스 이름 정의 (시각화 및 해석에 도움)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 데이터 탐색
print(f"학습 데이터 형태: {X_train.shape}")
print(f"테스트 데이터 형태: {X_test.shape}")
print(f"학습 레이블 형태: {y_train.shape}")
print(f"테스트 레이블 형태: {y_test.shape}")

# 이미지 시각화 (첫 9개 이미지)
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.tight_layout()
plt.savefig('fashion_mnist_samples.png')  # 이미지 저장
plt.close()

# 레이블 분포 확인
unique_labels, counts = np.unique(y_train, return_counts=True)
print("레이블 분포:")
for label, count in zip(unique_labels, counts):
    print(f"{class_names[label]}: {count}개")

# 2. 데이터 전처리 (Data Preprocessing)
# 픽셀 값 정규화 (0-255 → 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

print("데이터 정규화 완료: 픽셀값을 0-1 범위로 조정")

# 3. 모델 구축 (Model Building)
print("\n모델 구축 중...")
model = tf.keras.Sequential([
    # 입력층: 2D 이미지(28x28)를 1D 벡터(784)로 변환
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # 은닉층: 128개 뉴런과 ReLU 활성화 함수
    tf.keras.layers.Dense(128, activation='relu'),
    
    # 출력층: 10개 뉴런 (클래스 수와 동일)
    # 활성화 함수 없음 - from_logits=True 옵션 사용 예정
    tf.keras.layers.Dense(10)
])

# 모델 요약 출력
model.summary()

# 4. 모델 컴파일 (Model Compilation)
print("\n모델 컴파일 중...")
model.compile(
    optimizer='adam',
    # 모델이 logits를 출력하므로 from_logits=True 설정
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 5. 모델 학습 (Model Training)
print("\n모델 학습 시작...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,  # 학습 데이터의 10%를 검증 데이터로 사용
    verbose=1
)

# 6. 모델 평가 (Model Evaluation)
print("\n테스트 데이터로 모델 평가 중...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n테스트 정확도: {test_acc:.4f}")

# 7. 학습 곡선 시각화 (Learning Curves)
plt.figure(figsize=(12, 4))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('fashion_mnist_learning_curves.png')  # 학습 곡선 저장
plt.close()

# 8. 예측 및 시각화 (Prediction & Visualization)
# 예측 확률 계산 (logits → probabilities)
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

predictions = probability_model.predict(X_test)

# 테스트 이미지 15개에 대한 예측 결과 시각화
plt.figure(figsize=(15, 10))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')
    
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    
    # 맞으면 초록색, 틀리면 빨간색으로 표시
    color = 'green' if predicted_label == true_label else 'red'
    
    plt.title(f"예측: {class_names[predicted_label]}\n실제: {class_names[true_label]}", 
              color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig('fashion_mnist_predictions.png')  # 예측 결과 저장
plt.close()

print("\n실습 완료! 결과 이미지가 저장되었습니다.")

"""
TensorFlow(Keras)와 PyTorch 비교

TensorFlow (Keras Sequential API):
--------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

PyTorch (Class 기반):
-------------------
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        
주요 차이점:
- TensorFlow Keras: 선언적 방식으로 간결함
- PyTorch: 명령형 방식으로 유연함, Python 클래스와 비슷한 구조
- 두 프레임워크 모두 계산 그래프를 생성하지만 접근 방식이 다름
"""
