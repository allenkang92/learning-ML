# KNN을 이용한 와인 분류 및 모델 선택 예제

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 데이터 로드
wine = load_wine()
X = wine.data
y = wine.target

# 데이터 분할 (홀드아웃 검증)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 다양한 k값을 가진 KNN 모델 초기화
knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn7 = KNeighborsClassifier(n_neighbors=7)

# 모든 모델 학습
knn3.fit(X_train, y_train)
knn5.fit(X_train, y_train)
knn7.fit(X_train, y_train)

# 모델 평가
score3 = knn3.score(X_test, y_test)
score5 = knn5.score(X_test, y_test)
score7 = knn7.score(X_test, y_test)

print(f"KNN (k=3) 정확도: {score3:.3f}")
print(f"KNN (k=5) 정확도: {score5:.3f}")
print(f"KNN (k=7) 정확도: {score7:.3f}")

# 시각화: 다양한 k값에 따른 정확도
k_range = [3, 5, 7]
scores = [score3, score5, score7]

plt.figure(figsize=(10, 6))
plt.plot(k_range, scores, marker='o', linestyle='-', color='b')
plt.title('k 값에 따른 KNN 모델 정확도')
plt.xlabel('k 값 (이웃 수)')
plt.ylabel('정확도')
plt.grid(True)
plt.show()

# 최적 모델 선택
best_k = k_range[np.argmax(scores)]
print(f"최적의 k 값: {best_k}")

# 가장 좋은 모델을 사용한 예측 예시
best_model = knn5 if best_k == 5 else (knn3 if best_k == 3 else knn7)
# 테스트 데이터의 첫 번째 샘플 예측
sample = X_test[0].reshape(1, -1)
prediction = best_model.predict(sample)
print(f"샘플 예측 결과: {prediction[0]} (와인 품종: {wine.target_names[prediction[0]]})")

