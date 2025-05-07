import numpy as np
import matplotlib.pyplot as plt

# 활성화 함수 정의
logistic = lambda x: 1/(1+np.exp(-x))
dlogistic = lambda x: logistic(x)*(1-logistic(x))

tanh1 = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
tanh2 = lambda x: 2*logistic(2*x)-1
dtanh1 = lambda x: 1-tanh1(x)**2
dtanh2 = lambda x: 1-tanh2(x)**2

# 소프트맥스 함수 정의 (수치적 안정성 고려)
def softmax(x):
    # 1차원 벡터인 경우
    if x.ndim == 1:
        x = x - np.max(x)
        y = np.exp(x) / np.sum(np.exp(x))
        return y
    # 2차원 행렬인 경우
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y

# 손실 함수 정의
se = lambda y, py: (y-py)**2  # 제곱 오차 (회귀)
nll = bce = lambda y, py: -(y*np.log(py)+(1-y)*np.log(1-py))  # 이진 교차 엔트로피 (이진 분류)
ce = lambda y, py: -np.sum(y*np.log(py+1e-10))  # 범주형 교차 엔트로피 (다중 클래스 분류)

# 학습 데이터 정의
X = np.array([[0,0], [0,1], [1,0], [1,1]])  # 입력 데이터

# 다중 작업 학습을 위한 목표 데이터
# Y[:, 0]: 합계 (SUM) - 회귀 작업
# Y[:, 1]: AND 논리 연산 - 이진 분류 작업
# Y[:, 2:]: 원-핫 인코딩된 OR/XOR 유사 작업 - 다중 클래스 분류 작업
Y = np.array([
    [0, 0, 1, 0],  # [0,0] -> SUM=0, AND=0, 첫번째 클래스(1,0)
    [1, 0, 0, 1],  # [0,1] -> SUM=1, AND=0, 두번째 클래스(0,1)
    [1, 0, 0, 1],  # [1,0] -> SUM=1, AND=0, 두번째 클래스(0,1)
    [2, 1, 0, 1]   # [1,1] -> SUM=2, AND=1, 두번째 클래스(0,1)
])

# 하이퍼파라미터 설정
lr = 1e-3  # 학습률
epoch = 100000  # 학습 반복 횟수
history = []  # 손실 기록용

# 모델 초기화 (가중치와 편향)
W = np.random.rand(2, 4)  # 입력 2개, 출력 4개
B = np.random.rand(4)     # 출력 4개의 편향

# 학습 시작
for i in range(epoch):
    # 순전파 계산
    Z = X @ W + B  # 선형 결합
    
    # 각 작업에 맞는 활성화 함수 적용
    _Y = np.c_[
        Z[:, 0],            # SUM 작업 - 항등 함수 (선형)
        logistic(Z[:, 1]),  # AND 작업 - 로지스틱 함수
        softmax(Z[:, 2:])   # OR/XOR 유사 작업 - 소프트맥스 함수
    ]
    
    # 손실 계산 및 기록 (1000 에포크마다)
    if i % 1000 == 0:
        # 각 작업별 손실 계산
        loss_sum = np.mean(se(Y[:, 0], _Y[:, 0]))
        loss_and = np.mean(bce(Y[:, 1], _Y[:, 1]))
        loss_class = 0
        for j in range(len(X)):
            loss_class += ce(Y[j, 2:], _Y[j, 2:])
        loss_class /= len(X)
        
        # 전체 손실 = 각 작업별 손실의 합
        total_loss = loss_sum + loss_and + loss_class
        history.append([i, loss_sum, loss_and, loss_class, total_loss])
        
        if i % 10000 == 0:
            print(f"Epoch {i}: Sum Loss={loss_sum:.4f}, AND Loss={loss_and:.4f}, Class Loss={loss_class:.4f}, Total={total_loss:.4f}")
    
    # 역전파 (단순화된 형태)
    # 실제 값과 예측 값의 차이를 손실의 그래디언트로 사용
    loss = (_Y - Y)
    
    # 가중치와 편향 업데이트
    W = W - lr * X.T @ loss
    B = B - lr * np.sum(loss, axis=0)

# 학습 종료
print(f"Final Weights:\n{W}")
print(f"Final Biases: {B}")

# 학습 곡선 시각화
history = np.array(history)
plt.figure(figsize=(12, 6))
plt.plot(history[:, 0], history[:, 1], label='Sum Loss')
plt.plot(history[:, 0], history[:, 2], label='AND Loss')
plt.plot(history[:, 0], history[:, 3], label='Class Loss')
plt.plot(history[:, 0], history[:, 4], label='Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()

# 학습된 모델로 예측
Z = X @ W + B
_Y = np.c_[Z[:, 0], logistic(Z[:, 1]), softmax(Z[:, 2:])]

print("\nPredictions:")
print("SUM Task (Regression):")
print(f"Predicted: {np.round(Z[:, 0])}")
print(f"Actual: {Y[:, 0]}")

print("\nAND Task (Binary Classification):")
print(f"Predicted: {np.round(logistic(Z[:, 1]))}")
print(f"Actual: {Y[:, 1]}")

print("\nOR/XOR-like Task (Multi-class Classification):")
print(f"Predicted:\n{np.round(softmax(Z[:, 2:]))}")
print(f"Actual:\n{Y[:, 2:]}")

# 새로운 입력에 대한 예측
new_input = np.array([[0, 2]])  # 학습 데이터에 없는 입력
Z_new = new_input @ W + B
_Y_new = np.c_[Z_new[:, 0], logistic(Z_new[:, 1]), softmax(Z_new[:, 2:])]

print("\nPrediction for new input [0, 2]:")
print(f"SUM: {Z_new[0, 0]:.2f}")
print(f"AND: {logistic(Z_new[0, 1]):.2f}")
print(f"Class: {softmax(Z_new[0, 2:])}")
