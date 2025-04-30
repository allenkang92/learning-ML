import numpy as np
import matplotlib.pyplot as plt

# AND 게이트 데이터셋
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 입력
Y = np.array([0, 0, 0, 1])                      # 출력 (AND 연산 결과)

# 입력 데이터에 편향을 위한 1 추가 (편향 통합 처리)
D = np.c_[np.ones(len(X)), X]

# 가중치 초기화 (편향 + 입력 2개 = 총 3개 가중치)
W = np.random.rand(3)

# 활성화 함수 (시그모이드)와 그 미분
logistic = lambda x: 1/(1+np.exp(-x))
dlogistic = lambda y: y*(1-y)  # 참고용 (직접 사용되지 않음)

# 손실 함수 (Negative Log Likelihood/Binary Cross-Entropy)
NLL = BCE = lambda y, predy: -y*np.log(predy)-(1-y)*np.log(1-predy)

# 학습 설정
lr = 1e-4  # 학습률
epoch = 100000  # 에포크 수
history = []  # 손실 기록용

# 학습 시작
for i in range(epoch):
    # 순전파
    ls = D @ W  # 가중합 (입력과 가중치의 내적 + 편향)
    predY = logistic(ls)  # 활성화 함수 적용 (예측값)
    
    # 손실 계산 및 기록 (10000 에포크마다)
    if i % 10000 == 0:
        loss = np.mean(BCE(Y, predY))  # 평균 손실 계산
        history.append(loss)
        print(f"에포크 {i}/{epoch}, 손실: {loss:.4f}")
    
    # 역전파 (기울기 계산)
    dLossY = -Y + predY  # 손실 함수와 시그모이드 미분을 합친 기울기 (간소화된 형태)
    dW = D.T  # 입력의 전치행렬
    
    # 가중치 업데이트 (경사 하강법) - 풀배치 방식
    W = W - lr * (dW @ dLossY)

# 최종 손실 기록
loss = np.mean(BCE(Y, predY))
history.append(loss)
print(f"최종 손실: {loss:.4f}")

# 학습된 가중치 확인
print("학습된 가중치:", W)

# 학습 결과 확인
predictions = logistic(D @ W)
print("예측값:", predictions)
print("임계값(0.5) 적용 결과:", predictions > 0.5)
print("실제값:", Y)

# 학습 곡선 시각화
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title('학습 곡선: 로지스틱 회귀 (AND 게이트)')
plt.xlabel('학습 단계 (10000 에포크 단위)')
plt.ylabel('손실 (NLL/BCE)')
plt.grid(True)
plt.show()
