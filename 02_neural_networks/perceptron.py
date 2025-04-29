import numpy as np
import matplotlib.pyplot as plt

# AND 게이트 데이터셋 준비
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 입력
Y = np.array([[0], [0], [0], [1]])              # 출력 (AND 연산 결과)

# 데이터 로더 함수 정의
def DataLoader(x, y, batch_size=None, shuffle=False):
    n_data = len(x)
    indices = np.arange(n_data)
    
    if shuffle:
        np.random.shuffle(indices)
    
    if batch_size is None:
        batch_size = n_data
    
    for i in range(0, n_data, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield x[batch_indices], y[batch_indices]

# 모델 초기화
n_in = 2   # 입력 크기
n_out = 1  # 출력 크기
lr = 0.1   # 학습률

# 가중치와 편향 랜덤 초기화
W = np.random.randn(n_out, n_in)  # 출력 크기 x 입력 크기
B = np.zeros([n_out])             # 출력 크기

# 손실 함수 (제곱 오차)와 그 미분 정의
def se(y, pred_y):
    return ((y - pred_y) ** 2).sum() / 2

def dse1(y, pred_y):
    return (pred_y - y)  # 실제 구현에서는 부호를 반대로 사용

# 학습 과정 기록을 위한 변수
history = []

# 학습 시작
epochs = 10000
batch_size = 1  # 미니배치 크기 설정 (SGD)

for epoch in range(epochs):
    epoch_loss = 0
    
    # 데이터 로딩 및 학습
    for x, y in DataLoader(X, Y, batch_size=batch_size, shuffle=True):
        # 순전파
        pred_y = x @ W.T + B
        
        # 손실 계산
        loss = se(y, pred_y)
        epoch_loss += loss
        
        # 역전파 (기울기 계산)
        dloss = dse1(y, pred_y)
        dW = x.T @ dloss
        dB = dloss.sum()
        
        # 가중치와 편향 업데이트
        W -= lr * dW
        B -= lr * dB
    
    # 손실 기록
    history.append(epoch_loss)
    
    # 학습 과정 출력 (1000 에포크마다)
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# 학습된 모델 확인
print("학습된 가중치:", W)
print("학습된 편향:", B)

# 모델 테스트
test_input = np.array([1, 1])
prediction = test_input @ W.T + B
print(f"입력 [1, 1]에 대한 예측값: {prediction[0]:.3f}")
print(f"임계값(0.5) 적용 결과: {prediction[0] > 0.5}")  # AND 게이트 결과

# 손실 시각화
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title('훈련 손실 변화')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.grid(True)
plt.show()
