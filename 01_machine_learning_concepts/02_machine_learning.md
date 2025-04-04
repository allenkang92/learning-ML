## 머신러닝 모델링 및 분석 개념

### 1. 분석 용어 및 모델 기본 개념

- **Analysis vs. Analytics:**
    - 명확한 구분이 필수적이진 않으며, 분야(경제경영, 통계)나 사람에 따라 다르게 사용될 수 있음.
    - `Analysis`: 데이터 분해 및 통찰력 도출에 초점.
    - `Analytics`: 데이터 기반 의사결정을 위한 포괄적 접근. 종종 프로그래밍 기술까지 포함하는 뉘앙스.
- **컴퓨테이셔널 모델 (Computational Model):**
    - 컴퓨터가 처리할 수 있는 형태의 모델.
    - 반드시 **숫자 데이터**로 구성되어야 함 (**수치화(Quantification)** 필수). 수치화 방식이 성능에 중요.
- **모델 (Model) / 모형:**
    - 현실 세계의 문제를 해결하기 위한 **솔루션(Solution)**.
- **가설 (Hypothesis):**
    - 모델링하려는 실제 현상/함수($$f_R(x)$$, 알 수 없음)에 근사할 것이라 가정하는 특정 함수/규칙 (f(x)).
    - 100% 완벽히 실제를 재현할 수 없으므로 '가설'이라고 부름.
    - 머신러닝(컴퓨테이셔널 모델)에서는 **Hypothesis ≈ Model**로 통용됨.
- **모델링 (Modeling):** 모델(솔루션)을 만드는 과정.
    - 주요 요소: Diagram(구조 시각화), Math(수학), Statistics(통계), Coding(구현).
    - 목표: 실제 함수와 모델 간의 차이(`f_R(x) - f(x)`)를 0에 가깝게 만드는 것 (최적화).
    - **Loss Function (손실 함수):** 실제 함수를 모르므로, 모델 예측과 실제 데이터 간의 오류를 측정하는 대리 함수.
    - **Optimization (최적화):** 손실 함수를 최소화하는 모델 파라미터를 찾는 과정 (e.g., 경사하강법).

### 2. 모델링 접근 방식

- **지식 기반 (Knowledge-based / First-principle):**
    - 사전에 정의된 규칙, 논리, 전문가 지식 기반 (e.g., 전문가 시스템).
    - 1970년대 유행. 데이터 부족 시 대안. 필터링 용도 가능.
    - 단점: 현실의 모호함, 예외 처리 어려움, 규칙 구축 힘듦.
- **데이터 기반 (Data-driven / Machine Learning):**
    - 대량의 데이터에서 패턴과 관계를 학습 (빅데이터 기반).
    - 여러 사례에서 추상적인 규칙/패턴 도출.
    - 현재 예측 모델링에서 가장 성능 좋은 기법으로 간주됨.
- **통합적 접근:** 실제 문제 해결에는 두 방식 모두 활용하는 것이 중요.

### 3. 데이터의 중요성 및 특성

- **데이터의 핵심 역할:** 머신러닝 성능의 가장 중요한 요소 (Data > Algorithm > Hardware(GPU)). 데이터가 많을수록 대표성 증가 (대수의 법칙).
- **데이터 수치화:** 컴퓨테이셔널 모델의 필수 전제. 단위(unit) 변환 등 고려 필요.
- **데이터 종류/측정 수준 (Levels of Measurement):**
    - **Nominal (명목):** 구분만 가능 (같다/다르다). 비교 불가 (e.g., set).
    - **Ordinal (서열):** 순서/비교 가능 (크다/작다). 간격 일정 X (e.g., 문자열 비교).
    - **Interval (등간):** 순서/비교 + 덧셈/뺄셈 가능. 절대 0점 없음 (e.g., 온도).
    - **Ratio (비율):** 순서/비교 + 사칙연산 모두 가능. 절대 0점 있음 (e.g., 키, 무게).
- **데이터 관련 용어 (상황에 따라 다르게 불림):**
    - **Row (행):** data point, sample, example, observation, case, tuple, event, record, instance. (로우를 데이터로 보는 경우가 많음)
    - **Column (열):** feature, attribute, predictor, dimension, covariates, regressor, explanatory, stimulus, independent variable. (컬럼 = 차원 = 뉘앙스)
    - **Target (목표 변수):** label, class, response, regressand, tag, category, dependent variable (y).
- **데이터 품질 및 가정:**
    - **충분성 (Sufficiency):** 학습에 충분한 양 (Learning Curve로 확인).
    - **대표성 (Representativeness):** 데이터가 전체 모집단을 잘 반영하는가 (EDA로 확인).
        - Sampling Noise: 우연에 의한 편향 (데이터가 적을 때).
        - Sampling Bias: 표본 추출 방법 자체의 문제 (데이터가 많아도 발생 가능).
    - **독립성 (Independence):** 데이터 샘플(row) 간, 변수(column) 간 독립적이어야 함 (IID 가정).
        - 로우 간 독립성 위배 예: 중복 데이터 (Duplicate).
        - 컬럼 간 독립성 위배 예: 다중 공선성 (Multicollinearity).
        - 모델에 따라 독립성 가정의 중요도 다름 (딥러닝은 덜 민감한 편).
    - **데이터 품질 (Quality):** 오류(Random/Systematic), 편향(Bias - ethics/fairness) 등 고려. 나쁜 데이터는 수정 필요.

### 4. 차원의 저주 (Curse of Dimensionality)

- **정의:** 데이터의 차원(컬럼 수)이 증가할수록, 같은 비율의 공간을 설명(모델링)하기 위해 필요한 데이터 양(로우 수)이 기하급수적으로 증가하는 문제.
- **영향:**
    - 데이터 부족 시 모델 성능 저하.
    - **과적합 (Overfitting)** 문제 유발: 모델이 학습 데이터에만 너무 잘 맞춰져 새로운 데이터에 대한 예측 성능(일반화 성능)이 떨어지는 현상. 수학적으로는 **분산(Variance)**이 큰 모델.
- **판단:** EDA (`info()`, `describe()`, 시각화)를 통해 컬럼 수 대비 로우 수를 보고 가능성 추정.
- **해결책:** 데이터 양 증가, 차원 축소(Dimensionality Reduction), 특징 선택(Feature Selection), 정규화(Regularization) 등. (마법의 은탄환은 없음).
- **주의:** 컬럼(차원)이 많으면 뉘앙스가 풍부해져 좋을 수도 있지만, 차원의 저주 위험 동반.

### 5. 데이터 탐색 및 전처리 (EDA & Preprocessing)

- **탐색적 데이터 분석 (EDA):**
    - 데이터의 기본적인 특성(분포, 관계, 이상치, 결측치 등) 파악.
    - 가정 검증 (대표성, 정규성 등), 차원의 저주 가능성 확인.
    - 주요 도구:
        - `.info()`: 데이터 타입, 결측치 확인, 컬럼/로우 수 파악 (차원 저주 가능성 1차 확인).
        - `.describe()`: 기술 통계량 확인.
        - `.head()`, `.tail()`: 데이터 정돈 상태(Tidy) 확인.
        - 시각화 (`seaborn`, `matplotlib`):
            - `boxplot`, `boxenplot`, `violinplot`: 분포, 이상치 확인.
            - `pairplot`: 변수 간 관계, 분포 동시 확인 (hue='target' 유용).
        - `.skew()`, `.kurt()`: 왜도, 첨도 확인 (정규성 검토).
- **데이터 전처리:** 머신러닝 모델 학습 전 데이터를 적합한 형태로 변환/정제.
    - **Encoding:** 범주형 데이터를 수치형으로 변환 (One-Hot, Label Encoding 등).
    - **Imputation:** 결측치 처리 (삭제, 평균/중앙값/최빈값 대체, 회귀 대체 등).
    - **Discretization:** 연속형 데이터를 범주형으로 변환.
    - **Feature Scaling:** 변수 간 단위/크기 차이 조정 (Standardization, Normalization).
    - **Dimensionality Reduction / Feature Selection:** 차원의 저주 완화, 모델 단순화 (PCA, LDA, RFE, CFS 등).

### 6. 머신러닝 기본 분류 및 워크플로우

- **학습 유형:**
    - **지도 학습 (Supervised):** 정답(label)이 있는 데이터 사용. 예측/분류 문제. (전통적 ML 핵심)
    - **비지도 학습 (Unsupervised):** 정답 없는 데이터 사용. 군집화, 차원 축소 등. (딥러닝에서 중요)
    - **강화 학습 (Reinforcement):** 환경과의 상호작용 통해 보상 최대화 학습 (Agent).
- **일반적 워크플로우:**
    1. 문제 정의 (Problem Definition)
    2. 데이터 수집 및 식별 (Identify and collect data)
    3. 데이터 탐색 및 준비 (Explore and prepare data - EDA, Preprocessing)
    4. 모델 구축 및 평가 (Build and evaluate model - Training, Validation, Testing)
    5. 결과 소통/적용 (Communicate results / Deploy)

### 7. 모델 평가 및 선택

- **모델 평가 이유:** 일반화 성능 추정, 모델 튜닝(하이퍼파라미터 최적화), 알고리즘 비교/선택.
- **핵심 Trade-off:** 편향(Bias) vs. 분산(Variance).
    - Bias: 모델 가정이 잘못되어 발생하는 오류 (Underfitting).
    - Variance: 데이터 변동에 모델이 민감하게 반응하는 정도 (Overfitting).
- **평가 방법:** Holdout, Cross-Validation(K-fold, LOOCV), Bootstrap 등 (별도 학습 내용).
- **모델 선택:** 목적(정확도, 속도, 해석력 등)에 맞는 최적 모델 선택. Baseline 모델과 비교 중요.
- **No Free Lunch Theorem:** 모든 문제에 대해 항상 최적인 단일 모델/알고리즘은 없음.
- **Occam's Razor:** 가능한 단순한 모델 선호 (해석 용이성 등) vs. 복잡한 모델의 높은 정확도 (Trade-off 고려).

### 8. 라이브러리 및 도구

- **Pandas:** 데이터 조작, 분석 (`DataFrame`, `Series`).
- **Seaborn, Matplotlib:** 데이터 시각화 (EDA).
- **Scikit-learn:** 머신러닝 라이브러리 (알고리즘, 전처리, 평가 도구).
    - 연습용 데이터셋 제공: `load_*`(작음), `fetch_*`(다운로드), `make_*`(랜덤 생성).
- **Notebook (Jupyter, Colab):** 대화형 분석 환경.

**핵심 요약:** 

컴퓨테이셔널 모델은 **수치화된 데이터**를 기반으로 함.
**데이터 품질과 양**이 성능에 가장 중요. 
모델링은 **지식 기반**과 **데이터 기반** 접근 방식을 조합하며, 다양한 **분류 기준**(Parametric/Non-parametric, Linear/Non-linear 등)에 따라 모델 특성이 달라짐. 
**차원의 저주**와 **과적합**은 주요 문제이며, **EDA**와 **전처리**를 통해 데이터를 이해하고 정제하는 과정이 필수적. 
최종 목표는 문제 해결을 위한 **최적의 모델**을 선택하고 평가하는 것이며, 이 과정에서 여러 **Trade-off** (e.g., 정확도 vs. 해석력, 편향 vs. 분산)를 고려해야 할 것