import numpy as np
import matplotlib.pyplot as plt

# AND 게이트 데이터셋 - 다중 클래스 형태(원-핫 인코딩)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 입력
# 원-핫 인코딩된 타겟: [1,0]은 클래스 0(False), [0,1]은 클래스 1(True) 의미
Y = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])  # AND 연산 결과

# 소프트맥스 함수 정의 (수치적 안정성을 위한 max trick 적용)
def softmax(x):
    if x.ndim == 1:
        x = x - np.max(x)  # 최댓값을 빼서 오버플로우 방지
        return np.exp(x) / np.sum(np.exp(x))
    else:  # x.ndim == 2
        x = x - np.max(x, axis=1, keepdims=True)  # 각 행(샘플)별 최댓값 빼기
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

# 손실 함수 (교차 엔트로피)
CE = lambda truey, predy: -np.sum(truey * np.log(predy + 1e-7))  # 수치 안정성을 위해 작은 값 추가

# 모델 초기화
nIn = 2    # 입력 크기
nOut = 2   # 출력 크기 (클래스 수)
lr = 1e-4  # 학습률

# 가중치와 편향 초기화
W = np.random.randn(nIn, nOut)  # 입력 크기 x 출력 크기
B = np.zeros(nOut)              # 출력 크기

# 소프트맥스와 크로스 엔트로피 역전파를 위한 통합 함수
def SoftmaxWithLoss(truey, predy):
    return predy - truey  # 소프트맥스+크로스엔트로피 조합의 미분 = (예측값 - 실제값)

# 학습 과정 기록을 위한 변수
history = []

# 학습 시작
epochs = 100000

for i in range(epochs):
    # 순전파
    ls = X @ W + B  # 가중합 계산 (선형 계산)
    predY = softmax(ls)  # 소프트맥스 함수 적용 (확률 계산)
    
    # 손실 계산 및 기록 (10000 에포크마다)
    if i % 10000 == 0:
        loss = CE(Y, predY)  # 교차 엔트로피 손실 계산
        history.append(loss)
        print(f"에포크 {i}/{epochs}, 손실: {loss:.4f}")
    
    # 역전파 (기울기 계산)
    dY = SoftmaxWithLoss(Y, predY)  # 소프트맥스+교차엔트로피 미분 = (예측값 - 실제값)
    dW = X.T @ dY  # 가중치 기울기
    dB = np.mean(dY, axis=0)  # 편향 기울기 (배치 평균)
    
    # 가중치와 편향 업데이트 (경사 하강법) - 풀배치 방식
    W -= lr * dW
    B -= lr * dB

# 최종 손실 기록
loss = CE(Y, predY)
history.append(loss)
print(f"최종 손실: {loss:.4f}")

# 학습된 가중치와 편향 확인
print("학습된 가중치:\n", W)
print("학습된 편향:", B)

# 최종 예측 확률 확인
final_pred = softmax(X @ W + B)
print("\n최종 예측 확률:")
print(final_pred)

# 예측 클래스 확인 (가장 높은 확률의 클래스 선택)
pred_class = np.argmax(final_pred, axis=1)
true_class = np.argmax(Y, axis=1)
print("\n예측 클래스:", pred_class)
print("실제 클래스:", true_class)
print("정확도:", np.mean(pred_class == true_class))

# 학습 곡선 시각화
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title('학습 곡선: 소프트맥스 회귀 (AND 게이트)')
plt.xlabel('학습 단계 (10000 에포크 단위)')
plt.ylabel('손실 (Cross-Entropy)')
plt.grid(True)
plt.show()

# 결정 경계 시각화
plt.figure(figsize=(8, 6))
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 그리드의 모든 점에 대한 예측 확률 계산
Z = softmax(np.c_[xx.ravel(), yy.ravel()] @ W + B)
# 클래스 1(True)에 대한 확률만 선택
Z = Z[:, 1].reshape(xx.shape)

# 등고선 그리기
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
plt.colorbar(label='클래스 1의 확률')
plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='k')  # 결정 경계 (확률=0.5)

# 훈련 데이터 포인트 그리기
plt.scatter(X[:, 0], X[:, 1], c=true_class, cmap=plt.cm.RdBu, edgecolors='k')
plt.title('소프트맥스 회귀로 구현한 AND 게이트의 결정 경계')
plt.xlabel('입력 1')
plt.ylabel('입력 2')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()
