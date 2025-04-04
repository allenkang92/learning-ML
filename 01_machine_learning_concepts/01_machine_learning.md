## 머신러닝 모델링 및 분석 개념

### 1. 기본 용어 및 개념 정의

- **Analysis vs. Analytics:**
    - 명확한 구분이 필수적이진 않음. 사용자/분야(경제경영, 통계)마다 다르게 사용.
    - **Analysis:** 데이터 분해 및 통찰력 도출을 위한 조사.
    - **Analytics:** 데이터 기반 의사결정을 위한 광범위한 접근. 프로그래밍 기술 포함하는 경향.
- **모델(Model) / 모형:**
    - 현실 세계의 문제를 해결하기 위한 솔루션(Solution).
    - 컴퓨터가 처리하는 경우 **컴퓨테이셔널 모델(Computational Model)**이라고 함.
- **가설(Hypothesis):**
    - **실제 함수 (**$$f_R(x)$$**):** 우리가 모델링하려는, 알 수 없는 실제 세계의 메커니즘.
    - **가설/모델 (f(x)):** 실제 함수와 유사할 것이라고 믿거나 희망하는 특정 함수/모델. 100% 완벽히 실제를 재현할 수 없기 때문에 '가설'이라고도 함.
    - 컴퓨테이셔널 모델에서는 **Hypothesis = Model**로 통용됨.
- **모델링(Modeling):** 모델(솔루션)을 만드는 과정.
    - 포함 요소: Diagram (구조 시각화), Math (이산수학, 선형대수), Statistics (통계), Coding (구현).

### 2. 모델링의 목표와 최적화

- **목표:** 실제 함수 **(**$$f_R(x)$$**)**와 모델(f(x))의 차이를 0에 가깝게 만드는 것.
    - `f_R(x) - f(x) ≡ 0` 을 목표로 함.
- **한계:** 실제 함수 **(**$$f_R(x)$$**)**는 알 수 없으므로 직접 비교/검증 불가.
- **Loss Function (손실 함수, l(x)):** 모델의 예측과 실제 값(또는 목표) 간의 차이(오류)를 측정하는 대리 함수.
- **최적화(Optimization):** Loss Function의 값을 최소화하는 모델 파라미터를 찾는 과정.
    - 정확히 0으로 만들기 어려우므로 최솟값을 찾아 0에 가깝게 만듦.
    - 예: 경사 하강법(Gradient Descent), 역전파(Backpropagation) 등 사용.

### 3. 모델링 접근 방식: 지식 기반 vs. 데이터 기반

- **지식 기반 (Knowledge-based / First-principle):**
    - 사전에 정의된 규칙(Rule), 논리(Logic), 원칙(Principle)을 기반으로 모델 구축.
    - 예: 전문가 시스템(Expert System) - `IF-THEN` 규칙으로 전문가 지식 표현.
    - **장점:** 원리가 명확할 때 강력함, 해석 용이.
    - **단점:** 현실 세계의 모호함, 예외 처리 어려움, 규칙 구축 힘듦. 데이터 부족 시 대안.
- **데이터 기반 (Data-driven / Machine Learning):**
    - 대량의 데이터에서 패턴과 관계를 학습하여 모델 구축.
    - 예: 통계적 모델링, 머신러닝 알고리즘.
    - **장점:** 복잡한 패턴 학습 가능, 데이터가 많을수록 성능 향상 기대.
    - **단점:** 데이터 품질에 민감("Garbage in Garbage out"), 블랙박스 문제 발생 가능.
- **통합적 접근:** 실제 문제 해결을 위해서는 두 가지 접근 방식을 **상호 보완적**으로 활용하는 것이 중요.

### 4. 머신러닝 모델 분류 기준

1. **General vs. Ad Hoc Models:**
    - **General:** 특정 문제에 국한되지 않고 다양한 데이터에 적용 가능 (e.g., 나이브 베이즈). 데이터만 있으면 적용 가능하나 성능은 보장 못 함. 학습 부담 적음.
    - **Ad Hoc:** 특정 문제 해결에 특화되어 설계된 모델 (e.g., 딥러닝). 성능은 우수할 수 있으나 범용성 낮고, 구조 설계 등 구축 어려움.
2. **Parametric vs. Non-parametric Models:**
    - **Parametric:** 모델 구조(파라미터 개수)가 사전에 고정됨 (e.g., 선형 회귀 `y = ax + b`). 파라미터(a, b)를 데이터로부터 학습. **데이터 모델링** 접근 방식. 가정 검증 필요.
    - **Non-parametric:** 모델 구조가 데이터에 따라 유연하게 결정됨 (e.g., 결정 트리, k-NN). 파라미터 수가 데이터에 따라 변함. **알고리드믹 모델링** 접근 방식. 성능으로 평가. 가정 검증 불가 (블랙박스 경향).
3. **Linear vs. Non-Linear Models:**
    - **Linear:** 선형성(동질성 Homogeneity, 가산성 Additivity) 만족. `h(x) = θ₀ + θ₁x₁ + ... + θnxn` 형태.
        - **장점:** 계산 편리, **해석 용이** (계수(θ)가 변수 중요도 반영), 오버피팅 방지 용이(Robust).
        - **Intrinsically Linear:** 비선형처럼 보이지만 변환(로그, 제곱근 등)을 통해 선형으로 만들 수 있는 모델.
    - **Non-Linear:** 고차 다항식, 로그, 지수 함수 등 포함. 복잡한 패턴 학습 가능하나 해석 어려움.
    - **Trade-off:** 일반적으로 선형 모델은 해석 용이, 비선형 모델은 성능 우수 경향. (Accuracy vs. Interpretability)
4. **Blackbox vs. Descriptive Models:**
    - **Descriptive:** 모델의 의사결정 과정을 설명/해석 가능 (e.g., 선형 회귀, 결정 트리).
    - **Blackbox:** 모델 내부 작동 방식 이해 어려움 (e.g., 딥러닝, 복잡한 앙상블).
    - **XAI (설명 가능한 AI):** 블랙박스 모델의 설명력을 높이려는 연구 분야. AI 공정성(Fairness), 법적/윤리적 문제(자율주행 등) 때문에 중요성 증대.
5. **Generative vs. Discriminative Models:**
    - **Generative (생성 모델):** 각 클래스의 데이터 분포(P(x|y))와 클래스 확률(P(y))을 학습하여 P(y|x) 추론. 데이터 생성 가능. 레이블 없는 데이터 활용 가능. (e.g., 나이브 베이즈, GAN). 구축 어려움.
    - **Discriminative (판별 모델):** 클래스 간의 결정 경계(Decision Boundary) 또는 P(y|x)를 직접 학습. 레이블 데이터 필요. (e.g., 로지스틱 회귀, SVM).
6. **Stochastic vs. Deterministic Models:**
    - **Stochastic (확률적):** 모델 내부에 무작위성/확률적 요소 포함. 같은 입력에도 다른 출력 가능. (e.g., 베이지안 모델, 일부 딥러닝). 확률 분포 이해 중요.
    - **Deterministic (결정론적):** 같은 입력에는 항상 같은 출력.
7. **Flat vs. Hierarchical Models:**
    - **Flat:** 단일 구조.
    - **Hierarchical:** 계층적 구조 가짐. 복잡한 문제 표현에 유리. (e.g., 결정 트리, 딥러닝 레이어).

### 5. 확률 분포의 중요성

- 데이터의 기본 특성을 나타냄 (정규분포, 균일분포, 이항분포 등).
- Stochastic 모델링의 기초.
- 적합한 머신러닝 알고리즘 선택의 기준 (e.g., 가우시안 나이브 베이즈 vs. 베르누이 나이브 베이즈).

### 6. 모델링 프로세스 요약

1. **Task 정의:** 해결하려는 문제 정의 (e.g., 회귀, 분류). 문제의 중요성 인지.
2. **Model/Architecture 선택:** 변수 간 관계를 설명할 모델 구조 선택.
3. **Loss Function 정의:** 모델 예측과 실제 값 간의 불일치(오류)를 정량화.
4. **Regularizer 선택 (선택 사항):** 특정 모델(설명) 선호도 반영, 오버피팅 방지.
5. **Model Fitting (Optimization):** 손실 함수 최소화 및 제약 조건 만족하는 모델 학습 (최적화 알고리즘 사용).

### 7. 머신러닝 시스템 구성 요소

- **Data:** 모델 학습 및 평가의 기반.
- **ML Algorithm:** 데이터를 학습하여 모델을 생성하는 절차/규칙.
- **ML Model:** 학습된 결과물. 입력 데이터에 대한 예측 수행.
    - `ML Algorithm + Data (+ Prediction Algorithm) → ML Model`
    - Scikit-learn에서는 학습된 모델을 `Estimator`라고 부름.

### 8. 회귀 분석 (Regression Analysis)

- **정의:** 종속 변수와 하나 이상의 독립 변수 간의 관계를 추정하는 통계적 프로세스. 예측(Prediction)과 정보 추출(Information)이 목표.
- **Francis Galton:** "평균으로의 회귀(Regression to the mean)" 개념 제시.
- **Linear Regression:**
    - 종속 변수와 독립 변수 간의 선형 관계 가정. `Y = a + bX + u`
    - 최소제곱법(Least Squares) 등으로 최적의 직선/평면 찾음.
    - **장점:** 단순성, 해석 용이성, 빠른 학습 속도. 비즈니스 등 다양한 분야 활용.
    - **종류:**
        - **Simple:** 독립 변수 1개. `y = b₀ + b₁x₁`
        - **Multiple:** 독립 변수 2개 이상. `y = b₀ + b₁x₁ + ... + bnxn`
        - **Multivariate vs. Multivariable:** 용어 구분 필요 (슬라이드 55 참고).
            - Multivariable: 단일 결과(y), 다중 입력 변수(x₁, ..., xk).
            - Multivariate: 다중 결과(Y), 다중 입력 변수(X).
- **Logistic Regression:**
    - 종속 변수가 범주형(주로 이진형)일 때 사용.
    - 선형 회귀 결과를 **로짓(logit) 함수(시그모이드 함수의 역함수)**에 적용하여 확률(0~1) 예측.
        - `logit(P(Y=1|x)) = ln(P/(1-P)) = β₀ + β₁x₁ + ... + βnxn`
        - `P(Y=1|x) = 1 / (1 + e^-(WTx+b))` (Sigmoid 함수)
    - **종류:** Binary, Multinomial, Ordinal.
    - **주의:** 변수가 많을 때 오버피팅 경향. 정규화(Regularization) 필요.
    - **머신러닝 관점:** 손실 함수(Negative Log Likelihood)를 경사 하강법으로 최소화하여 계수 추정 (통계적 최대 가능도 추정과 동일).
- **선형 회귀 vs. 로지스틱 회귀:**
    - **Linear:** 연속형 종속 변수 예측.
    - **Logistic:** 범주형 종속 변수 예측 (확률 출력). 로그 변환(logit)으로 선형화 효과. 샘플 크기 더 많이 요구하는 경향.

### 9. 모델 평가 및 선택

- 모델 평가(Evaluation)와 모델 선택(Selection)은 중요하며, 데이터 크기, 비교 대상(모델 vs. 알고리즘) 등에 따라 적절한 방법(Holdout, Cross-Validation, Nested CV, 통계 검정 등)을 사용해야 함.