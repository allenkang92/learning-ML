import numpy as np
import matplotlib.pyplot as plt

# XOR 게이트 데이터셋 준비
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 입력
Y = np.array([[0], [1], [1], [0]])              # 출력 (XOR 연산 결과)

# 활성화 함수와 도함수 정의
logistic = lambda x: 1 / (1 + np.exp(-x))
dlogistic1 = lambda x: logistic(x) * (1 - logistic(x))  # 입력이 활성화 함수 적용 전 값일 때
dlogistic2 = lambda y: y * (1 - y)                     # 입력이 활성화 함수를 거친 후 값일 때

# 손실 함수 정의
se = lambda y, py: (y - py) ** 2                      # 제곱 오차
mse = lambda y, py: np.mean(se(y, py))                # 평균 제곱 오차

# 하이퍼파라미터 설정
lr = 1e-1                                            # 학습률
epoch = 100000                                       # 학습 반복 횟수
history = []                                         # 손실 기록용

# 모델 구조: 입력층(2) -> 은닉층(2) -> 출력층(1) 
input_size = 2
hidden_size = 3  # 은닉층 뉴런 수 (XOR은 최소 2개 이상 필요)
output_size = 1

# 가중치와 편향 초기화 (He 초기화 적용)
# 입력층 -> 은닉층 가중치와 편향
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)  
B1 = np.zeros((1, hidden_size))

# 은닉층 -> 출력층 가중치와 편향  
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
B2 = np.zeros((1, output_size))

# 학습 시작
for i in range(epoch):
    # 순전파 (Forward Propagation)
    # 은닉층 계산
    Z1 = X @ W1 + B1                 # 입력 * 가중치 + 편향 (선형 변환)
    A1 = logistic(Z1)                # 은닉층 활성화
    
    # 출력층 계산
    Z2 = A1 @ W2 + B2                # 은닉층 출력 * 가중치 + 편향 (선형 변환) 
    A2 = logistic(Z2)                # 출력층 활성화 (최종 예측값)
    
    # 손실 계산 (10000 에포크마다)
    if i % 10000 == 0 or i == epoch - 1:
        loss = mse(Y, A2)            # 평균 제곱 오차 계산
        history.append(loss)
        print(f"에포크 {i}/{epoch}, 손실: {loss:.8f}")
    
    # 역전파 (Backpropagation)
    # 출력층 오차 계산
    dZ2 = A2 - Y                     # 출력 오차 = 예측값 - 실제값 (MSE와 시그모이드 미분 결합)
    
    # 은닉층->출력층 가중치/편향 그래디언트 계산
    dW2 = A1.T @ dZ2                 # A1 전치 * 출력 오차
    dB2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # 은닉층 오차 계산 (출력층 오차를 은닉층으로 역전파)
    dA1 = dZ2 @ W2.T                 # 출력 오차 * W2 전치
    dZ1 = dA1 * dlogistic2(A1)       # 은닉층 활성화 함수의 미분 적용
    
    # 입력층->은닉층 가중치/편향 그래디언트 계산
    dW1 = X.T @ dZ1                  # X 전치 * 은닉층 오차
    dB1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # 가중치와 편향 업데이트 (경사하강법)
    W2 -= lr * dW2
    B2 -= lr * dB2
    W1 -= lr * dW1
    B1 -= lr * dB1

# 학습 결과 확인
# 학습된 모델을 사용한 최종 예측
final_A1 = logistic(X @ W1 + B1)
final_A2 = logistic(final_A1 @ W2 + B2)

print("\n학습된 가중치:")
print("W1:\n", W1)
print("B1:", B1)
print("W2:\n", W2)
print("B2:", B2)

print("\n최종 예측:")
print(final_A2)
print("\n반올림한 예측 (0 또는 1):")
print(np.round(final_A2))
print("실제값:", Y.reshape(-1))
print("정확도:", np.mean(np.round(final_A2) == Y))

# 학습 곡선 시각화
plt.figure(figsize=(10, 5))
plt.plot(range(0, epoch, 10000)[:len(history)], history)
plt.title('MLP를 이용한 XOR 게이트 학습 곡선')
plt.xlabel('에포크')
plt.ylabel('손실 (MSE)')
plt.grid(True)

# 결정 경계 시각화
plt.figure(figsize=(8, 6))
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 그리드의 모든 점에 대한 예측값 계산
grid_points = np.c_[xx.ravel(), yy.ravel()]
hidden_output = logistic(grid_points @ W1 + B1)
Z = logistic(hidden_output @ W2 + B2)
Z = Z.reshape(xx.shape)

# 등고선 그리기
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
plt.colorbar(label='예측 확률')
plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='k')  # 결정 경계 (확률=0.5)

# 훈련 데이터 포인트 그리기
plt.scatter(X[:, 0], X[:, 1], c=Y.reshape(-1), cmap=plt.cm.RdBu, edgecolors='k')
plt.title('MLP로 구현한 XOR 게이트의 결정 경계')
plt.xlabel('입력 1')
plt.ylabel('입력 2')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()
