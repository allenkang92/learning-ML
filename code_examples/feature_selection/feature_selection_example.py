import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# 데이터 로드
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names

print(f"특성 개수: {X.shape[1]}")
print(f"샘플 개수: {X.shape[0]}")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터 스케일링 (특성 선택 전 중요)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 기준 모델 (특성 선택 없이)
print("\n===== 기준 모델 (특성 선택 없음) =====")
base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_train_scaled, y_train)
base_pred = base_model.predict(X_test_scaled)
base_accuracy = accuracy_score(y_test, base_pred)
print(f"기준 모델 정확도: {base_accuracy:.4f}")

# 1. 필터 방법 (Filter Method) - SelectKBest + f_classif
print("\n===== 필터 방법 (SelectKBest) =====")
selector = SelectKBest(f_classif, k=10)  # 상위 10개 특성 선택
X_train_filter = selector.fit_transform(X_train_scaled, y_train)
X_test_filter = selector.transform(X_test_scaled)

# 선택된 특성 인덱스 및 이름 확인
selected_features_filter = np.where(selector.get_support())[0]
selected_names_filter = [feature_names[i] for i in selected_features_filter]
print(f"선택된 특성: {selected_names_filter}")

# 필터 방법으로 선택된 특성을 사용한 모델
filter_model = LogisticRegression(max_iter=1000)
filter_model.fit(X_train_filter, y_train)
filter_pred = filter_model.predict(X_test_filter)
filter_accuracy = accuracy_score(y_test, filter_pred)
print(f"필터 방법 정확도: {filter_accuracy:.4f}")

# 2. 래퍼 방법 (Wrapper Method) - RFE (Recursive Feature Elimination)
print("\n===== 래퍼 방법 (RFE) =====")
estimator = LogisticRegression(max_iter=1000)
selector = RFE(estimator, n_features_to_select=10, step=1)
X_train_wrapper = selector.fit_transform(X_train_scaled, y_train)
X_test_wrapper = selector.transform(X_test_scaled)

# 선택된 특성 인덱스 및 이름 확인
selected_features_wrapper = np.where(selector.get_support())[0]
selected_names_wrapper = [feature_names[i] for i in selected_features_wrapper]
print(f"선택된 특성: {selected_names_wrapper}")

# 래퍼 방법으로 선택된 특성을 사용한 모델
wrapper_model = LogisticRegression(max_iter=1000)
wrapper_model.fit(X_train_wrapper, y_train)
wrapper_pred = wrapper_model.predict(X_test_wrapper)
wrapper_accuracy = accuracy_score(y_test, wrapper_pred)
print(f"래퍼 방법 정확도: {wrapper_accuracy:.4f}")

# 3. 임베디드 방법 (Embedded Method) - Random Forest Feature Importance
print("\n===== 임베디드 방법 (Random Forest Feature Importance) =====")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
importances = rf.feature_importances_

# 특성 중요도 순위와 이름 매핑
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# 상위 10개 특성 선택
top_features = feature_importance['feature'][:10].tolist()
top_indices = [list(feature_names).index(feature) for feature in top_features]

# 임베디드 방법으로 선택된 특성을 사용한 모델
X_train_embedded = X_train_scaled[:, top_indices]
X_test_embedded = X_test_scaled[:, top_indices]

print(f"선택된 특성: {top_features}")

embedded_model = LogisticRegression(max_iter=1000)
embedded_model.fit(X_train_embedded, y_train)
embedded_pred = embedded_model.predict(X_test_embedded)
embedded_accuracy = accuracy_score(y_test, embedded_pred)
print(f"임베디드 방법 정확도: {embedded_accuracy:.4f}")

# 결과 시각화 - 특성 선택 방법 비교
methods = ['모든 특성', '필터 방법', '래퍼 방법', '임베디드 방법']
accuracies = [base_accuracy, filter_accuracy, wrapper_accuracy, embedded_accuracy]

plt.figure(figsize=(10, 6))
plt.bar(methods, accuracies, color=['gray', 'blue', 'green', 'orange'])
plt.ylim(0.9, 1.0)  # 작은 차이를 확대하기 위한 y축 범위 조정
plt.title('특성 선택 방법에 따른 모델 정확도 비교')
plt.ylabel('정확도')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 정확도 값 표시
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.005, f'{acc:.4f}', ha='center')

plt.tight_layout()
plt.show()

