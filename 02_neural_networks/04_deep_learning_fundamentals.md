# 딥러닝/신경망 학습

## 1. 딥러닝의 역사적 흐름 및 주요 사건

- **초기 아이디어 및 통계학적 배경:**
    - **1662년 ~ 1960년:** 확률론, 베이즈 정리, 최소제곱법(회귀) 등 통계학의 발전. (Frequentist, Bayesian, Regression, Statistics)
    - **1990년 이후:** 머신러닝/데이터 마이닝, 데이터 분석, 데이터 과학으로 용어 확장.
- **신경망의 탄생과 발전:**
    - **1943년 (Electronic Brain):** 워렌 맥컬록 & 월터 피츠 - 최초의 뉴런 모델 (Thresholded Logic Unit - TLU), 논리 게이트(AND, OR, NOT) 구현.
        - 고정된 가중치, 학습 불가.
    - **1957년 (Perceptron):** 프랭크 로젠블랫 - 학습 가능한 가중치와 임계값을 가진 퍼셉트론 개발.
    - **1960년 (ADALINE):** 버나드 위드로 & 마르시안 호프 - 적응형 선형 요소, 델타 규칙(Delta Rule) 기반 학습.
    - **1969년 (XOR Problem, 1차 AI 겨울 "AI Winter"):** 마빈 민스키 & 시모어 페퍼트 - 단층 퍼셉트론의 한계(XOR 문제 해결 불가) 지적, AI 연구 암흑기.
    - **1986년 (Multi-layered Perceptron, Backpropagation):** 데이비드 셔멜하트, 제프리 힌튼, 로널드 윌리엄스 - 다층 퍼셉트론과 오류역전파 알고리즘 개발, 비선형 문제 해결 가능성 제시. (Golden Age 시작)
        - **Forward Activity (순전파):** 입력 → 출력 계산
        - **Backward Error (오류역전파):** 출력 오류 → 가중치 업데이트
        - **한계:** 큰 계산량, 지역 최적해(Local Optima), 과적합(Overfitting) 문제.
    - **1980년대 후반 ~ 1990년대 초반 (2차 신경망 겨울 "2nd Neural Winter"):** SVM 등 다른 머신러닝 알고리즘의 부상.
    - **1989년 (CNNs):** 얀 르쿤 - 합성곱 신경망 개발.
    - **1995년 (SVMs):** 코르테스 & 바프닉 - 서포트 벡터 머신.
    - **1997년 (LSTMs):** 호흐라이터 & 슈미트후버 - 장단기 메모리 네트워크 개발.
    - **2006년 (Deep Nets, Pretraining):** 제프리 힌튼 등 - 딥 빌리프 네트워크, 층별 예비학습(Pretraining)으로 깊은 신경망 학습 가능성 제시.
    - **2012년 (AlexNet, GPU Era):** 알렉스 크리제브스키 등 - ImageNet 대회에서 AlexNet 압도적 우승, GPU 활용으로 딥러닝 시대 본격화.

## 2. 뉴런의 생물학적 모델과 인공 뉴런

- **생물학적 뉴런:**
    - 수상돌기(Dendrites), 세포체(Cell body), 축삭(Axon), 축삭 말단(Axon terminals)으로 구성.
    - 입력 신호(자극) 수신 → 전압 축적 → 임계값 초과 시 스파이크 반응(발화).
    - 수상돌기는 복잡한 비선형 연산 수행 가능, 시냅스는 단순 가중치가 아닌 복잡한 동적 시스템.
    - 개별 뉴런은 간단한 연산 능력을 갖지만, 서로 연결되어 고도의 인지/판단 능력 생성.
- **인공 뉴런 (유닛, Unit):**
    - 생물학적 뉴런을 단순화한 모델.
    - 입력(Inputs) $x_i$ → 가중치(Weights) $w_i$ → 가중합(Weighted sum) $\sum w_i x_i + b$ (b: 편향) → 활성화 함수(Activation function) $\phi$ → 출력(Output) $y = \phi(\sum w_i x_i + b)$.
    - 로지스틱 회귀와 유사: $y = \sigma(w^T x + b)$ (여기서 $\sigma$는 시그모이드 함수).

## 3. 퍼셉트론 (Perceptron)

- **구조:** 입력층, 가중치, 편향, 하드 임계값 함수(Hard Thresholding Function) 또는 계단 함수(Step Function) 사용.
    - $y = 1 \text{ if } \sum w_i x_i + b \geq \theta \text{ (임계값), else } 0 \text{ (또는 -1)}$
- **단층 퍼셉트론:** 하나의 뉴런 층으로 구성, 선형 분리 가능한 문제만 해결 가능.

## 4. 논리 게이트 구현 (AND, OR, NOT)

- 단층 퍼셉트론으로 AND, OR, NOT 논리 게이트 구현 가능 (적절한 가중치와 편향 설정).
    - **NOT:** $b=1, w=-2$ (입력 $x_1$)
    - **AND:** $b=-1.5, w_1=1, w_2=1$ (입력 $x_1, x_2$)
    - **OR:** $b=-0.3, w_1=0.5, w_2=0.5$ (입력 $x_1, x_2$)
- 선형 결정 경계(Decision Boundary)를 가짐.

## 5. 학습 규칙 (Learning Rules)

- **헵의 규칙 (Hebbian Learning, 1949)**
    - 도널드 헵 제안. "Cells that fire together, wire together." (함께 발화하는 뉴런은 함께 연결된다)
    - 뉴런 A가 뉴런 B를 반복적으로 흥분시키면, 두 뉴런 간의 연결 강도(시냅스 가중치) 증가.
    - 학습 = 뉴런들 간의 연결을 만드는 활동 (시냅스 가소성).
    - 파블로프의 조건반사 설명 가능.
    - **업데이트 규칙:** $W_{new} = W_{old} + x \cdot y$, $B_{new} = B_{old} + y$ (비지도 학습 방식).
- **퍼셉트론 학습 규칙 (Perceptron Learning Rule, 1957)**
    - "Learning from mistakes." (실수로부터 학습)
    - 예측이 틀렸을 경우, 올바른 예측에 기여했을 입력의 연결 가중치 강화.
    - $z = w^T x$ (실제 출력 계산 전 값), $t$ (목표 값, 기대 출력)
    - **업데이트 조건:** $z \cdot t \leq 0$ (예측 부호와 실제 부호가 다를 때)
    - **업데이트 규칙:** $w \leftarrow w + t \cdot x$ (입력에 비례하여 가중치 조정)
    - 만약 출력 $y \in \{0, 1\}$이고, 활성화 함수가 계단 함수라면, $y \neq t$ 일 때 $w \leftarrow w + \alpha(t-y)x$.
    - Epoch 단위로 모든 학습 데이터에 대해 반복, 가중치 업데이트가 없을 때까지.
- **위드로-호프 학습 규칙 (델타 규칙, Delta Rule)**
    - ADALINE에서 사용.
    - 출력 ($y_{in}$)과 목표 값 ($t$)의 차이(오류)를 최소화하는 방향으로 가중치 업데이트 (최소제곱오차 기반).
    - **업데이트 규칙:** $\Delta W = \alpha \cdot x \cdot (t - y_{in})$ ($\alpha$: 학습률)
    - 선형 회귀의 SGD와 유사.
    - 지도 학습 방식.

## 6. 퍼셉트론 수렴 정리 (Perceptron Convergence Theorem)

- 학습 데이터가 선형적으로 분리 가능(Linearly Separable)하다면, 퍼셉트론 학습 알고리즘은 유한한 단계 내에 해(분류 경계)를 찾는 것을 보장.
- 선형 분리 불가능한 문제(e.g., XOR)에 대해서는 수렴 보장 안 됨.

## 7. XOR 문제와 다층 퍼셉트론 (MLP)

- **XOR 문제 (1969):**
    - 단층 퍼셉트론으로는 해결 불가능 (비선형 문제).
    - 입력 공간에서 선 하나로 두 클래스를 분리할 수 없음.
- **다층 퍼셉트론 (Multi-Layer Perceptron, MLP) 또는 인공 신경망 (Artificial Neural Network, ANN):**
    - 입력층(Input Layer), 은닉층(Hidden Layer), 출력층(Output Layer)으로 구성.
        - **입력층:** 데이터를 받는 부분, 레이어 수에 포함시키지 않음.
        - **은닉층:** 입력층과 출력층 사이에 위치, 외부로 드러나지 않음. 1개 이상 존재 가능.
        - **출력층:** 최종 예측 결과 출력.
    - "2-layer Neural Net" = "1-hidden-layer Neural Net"
    - **완전 연결 (Fully-connected) 또는 밀집 (Dense) 레이어:** 한 층의 모든 뉴런이 다음 층의 모든 뉴런과 연결.
    - **XOR 해결:** 두 개의 단층 퍼셉트론(AND, OR 등)을 조합하여 구성 가능. (예: (X1 AND NOT X2) OR (NOT X1 AND X2))
    - **Feed-forward (순전파):** 입력 → 은닉층 → 출력층으로 정보 전달, 예측 수행.

## 8. 활성화 함수 (Activation Function, 비선형성)

- 신경망에 비선형성을 도입하여 복잡한 패턴 학습 가능하게 함.
    - 선형 변환만 반복하면 결국 하나의 선형 변환과 같아짐.
- **역할:** 입력 신호의 가중합을 받아 출력 신호로 변환.
- **종류:**
    - **계단 함수 (Step Function):** 초기 퍼셉트론 사용, 미분 불가능.
    - **시그모이드 (Sigmoid):** $y = 1/(1+e^{-x})$, 출력 (0, 1), 미분 용이하나 Gradient Vanishing 문제.
        - 미분: $\sigma(x)(1-\sigma(x))$
    - **하이퍼볼릭 탄젠트 (Tanh):** $y = \tanh(x)$, 출력 (-1, 1), 시그모이드보다 Zero-Centered (평균 0) 특성으로 학습 효율 좋음.
        - 미분: $1 - \tanh^2(x)$
        - $\tanh(x) = 2 \cdot \text{sigmoid}(2x) - 1$
    - **ReLU (Rectified Linear Unit):** $y = \max(0, x)$, 계산 효율 높고 Gradient Vanishing 완화, 가장 많이 사용.
        - 단점: Dying ReLU (뉴런이 죽는 현상).
    - **Leaky ReLU:** $y = \max(0.1x, x)$, Dying ReLU 문제 완화.
    - **Softplus, Softsign, ELU, Swish, Mish 등:** 다양한 변형 함수들.
- **고려 사항:**
    - **미분 가능성:** 역전파 위해 중요.
    - **Gradient Vanishing/Exploding:** 기울기가 너무 작거나 커지는 문제.
    - **Zero-Centered:** 출력값의 평균이 0에 가까우면 학습 효율 향상.
    - **계산 비용.**

## 9. Dying ReLU 문제

- ReLU 뉴런의 활성화 값이 0이 되면 해당 뉴런은 더 이상 학습되지 않는 문제.
- **원인:** 높은 학습률, 큰 음수 편향.
- **해결책:** 낮은 학습률 사용, Leaky ReLU 등 변형 함수 사용, 초기화 방법 변경.

## 10. 딥러닝의 특징 및 장점

- **함수 근사 (Function Approximation):**
    - 딥러닝은 복잡한 함수(Function Composition)를 근사하는 방법.
    - 신경망 = 로지스틱 회귀 분류기들의 스택
    - **Universal Approximation Theorem (범용 근사 정리):** 충분히 큰 (넓고 깊은) 비선형 활성화 함수를 가진 신경망은 어떤 연속 함수든 원하는 정확도로 근사 가능.
        - 실제 학습 가능성이나 필요한 유닛 수는 보장하지 않음.
- **계층적 특징 학습 (Hierarchical Feature Learning):**
    - 낮은 수준의 특징(선, 모서리) → 중간 수준의 특징(눈, 코) → 높은 수준의 특징(얼굴 전체)으로 계층적으로 특징 추출 및 학습.
- **분산 표현 (Distributed Representation):**
    - Sparse Representation (One-hot) vs. Distributed Representation (Multi-hot, Micro-features).
    - 정보를 여러 뉴런에 분산하여 표현, 더 효율적이고 일반화 성능 향상.
- **데이터 양에 따른 성능:**
    - 데이터가 많을수록 전통적인 머신러닝 알고리즘보다 뛰어난 성능.
    - 모델 크기(파라미터 수)가 클수록 "기억 용량(Memorization Capacity)" 증가.
- **모델 구조:** 정해진 공식 없음, 실험 통해 결정.
- **벡터화 (Vectorization):**
    - 연산을 벡터 및 행렬 형태로 표현하여 Python 루프보다 훨씬 빠르게 계산 (Numpy, Tensorflow 등 라이브러리 활용).
    - GPU 사용 시 행렬 곱셈 매우 빠름.
    - 코드 간결화, 가독성 향상.
    - $y = Xw + b$
    - $J = \frac{1}{2N} ||y - t||^2$ (평균 제곱 오차)
- **딥러닝의 부흥 요인:**
    - **Big Data Availability:** 방대한 양의 데이터 (페이스북, 월마트, 유튜브, ImageNet 등).
    - **New ML Techniques:** 새로운 알고리즘 및 모델 구조.
    - **GPU Acceleration:** 병렬 처리를 통한 학습 속도 향상 (NVIDIA).

## 11. 학습 과정

1. **가중치 초기화 (Random Initialization):** 모델 파라미터 무작위 초기화.
2. **순전파 (Feed Forward):** 입력 데이터 → 신경망 통과 → 실제 출력 계산.
3. **손실 함수 계산 (Calculate Loss Function):** 실제 출력과 목표 출력(정답) 간의 오차 계산.
4. **오류의 미분 계산 (Calculate Derivative of Error):** 손실 함수를 각 가중치에 대해 미분 (마지막 레이어부터).
5. **역전파 (Backpropagate):** 계산된 그래디언트(미분값)를 뒤쪽 레이어로 전파.
6. **가중치 업데이트 (Update the Weights):** 옵티마이저(경사 하강법 등) 사용하여 가중치 수정.
7. **수렴까지 반복.**

## 12. 손실 함수 (Loss Function) / 비용 함수 (Cost Function) / 목적 함수 (Objective Function)

- **목적 함수:** 최적화 과정에서 평가되는 함수.
- **손실 함수:** 단일 학습 예제에 대한 오차 (Error Function).
- **비용 함수:** 전체 학습 데이터셋에 대한 평균 손실.
- **주요 손실 함수:**
    - **L2 손실 (평균 제곱 오차, MSE):** $MSE = \frac{1}{n} \sum (Y_i - \hat{Y}_i)^2$
        - 큰 오차에 민감 (제곱 효과).
    - **L1 손실 (평균 절대 오차, MAE):** $MAE = \frac{1}{n} \sum |Y_i - \hat{Y}_i|$
        - 이상치(Outlier)에 덜 민감 (Robust).
    - **Huber 손실 (Smooth L1 Loss):** MSE와 MAE의 장점 결합.
        - 작은 오차에는 MSE처럼, 큰 오차에는 MAE처럼 동작 ($\delta$ 하이퍼파라미터로 조절).
    - **이진 교차 엔트로피 (Binary Cross-Entropy, BCE):** 이진 분류 문제에 사용.
        - $L = -[y \log(p) + (1-y) \log(1-p)]$
    - **범주형 교차 엔트로피 (Categorical Cross-Entropy):** 다중 클래스 분류 (Single-label)에 사용.
        - 출력은 Softmax, 타겟은 One-hot 인코딩.
        - $L = -\sum y_i \log(\hat{y}_i)$
    - **다중 레이블 분류 시 BCE:** 각 클래스에 대해 Sigmoid 출력, BCE 적용.
    - **Hinge Loss:** SVM에서 주로 사용 (클래스 라벨 -1, 1).
    - **KL-Divergence:** 두 확률 분포 간의 차이 측정.
- **Logit (로짓):** $\log(\frac{y}{1-y})$, Sigmoid 함수의 역함수. Softmax는 다중 클래스 버전.

## 13. 최적화 알고리즘 (Optimization Algorithms)

- 손실 함수를 최소화하는 가중치를 찾는 과정.
- **경사 하강법 (Gradient Descent):**
    - $W_{t+1} = W_t - \alpha \frac{\partial L}{\partial W_t}$ ($\alpha$: 학습률, $\frac{\partial L}{\partial W_t}$: 그래디언트)
    - **확률적 경사 하강법 (Stochastic Gradient Descent, SGD):** 한 번에 하나의 샘플(또는 미니배치)만 사용하여 그래디언트 계산. 빠르지만 불안정.
    - **미니배치 경사 하강법 (Mini-batch Gradient Descent):** 전체 데이터셋을 작은 묶음(미니배치)으로 나누어 학습. SGD와 배치 GD의 절충안.
- **학습률 (Learning Rate, $\alpha$):**
    - 너무 크면 발산, 너무 작으면 학습 속도 느림.
    - **Learning Rate Schedulers:** Epoch에 따라 학습률 조정 (Adaptive Learning Rates).
- **경사 하강법의 문제점 및 개선 알고리즘:**
    - **Local Minima, Saddle Point:** 최적점이 아닌 곳에 수렴.
    - **느린 수렴 속도 (특히 평탄한 지역)**
    - **진동 (Oscillation):** 골짜기에서 좌우로 심하게 움직임.
    - **모멘텀 (Momentum):** 이전 그래디언트 방향을 현재 업데이트에 반영 (관성 효과). 골짜기 방향으로 가속, 진동 감소.
        - $v_t = \mu v_{t-1} - \alpha \nabla L(W_t)$
        - $W_{t+1} = W_t + v_t$
    - **Adagrad (Adaptive Gradient):** 각 파라미터마다 다른 학습률 적용. 자주 업데이트되는 파라미터는 학습률 감소, 적게 업데이트되는 파라미터는 학습률 증가. 학습 후반에 학습률이 너무 작아지는 단점.
        - `cache += dx**2`
        - `x -= learning_rate * dx / (sqrt(cache) + eps)`
    - **RMSProp (Root Mean Square Propagation):** Adagrad의 단점 보완. 그래디언트 제곱의 지수 이동 평균 사용.
        - `cache = decay_rate * cache + (1 - decay_rate) * dx**2`
    - **Adam (Adaptive Moment Estimation):** RMSProp + Momentum. 현재 가장 많이 사용되는 옵티마이저 중 하나.
        - `m = beta1*m + (1-beta1)*dx` (Momentum)
        - `v = beta2*v + (1-beta2)*(dx**2)` (RMSprop-like)
        - `x -= learning_rate * m_corrected / (sqrt(v_corrected) + eps)` (Bias correction 포함)
- **차등 학습률 (Differential Learning Rates):** 레이어 그룹별로 다른 학습률 설정.

## 14. 역전파 (Backpropagation)

- 다층 신경망에서 손실 함수의 그래디언트를 효율적으로 계산하는 알고리즘.
- **핵심 원리: 연쇄 법칙 (Chain Rule)**
    - $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z} \frac{\partial z}{\partial w}$ (예시)
- **과정:**
    1. **순전파:** 입력 → 출력 계산.
    2. **출력층에서의 그래디언트 계산:** 손실 함수를 출력층 활성화 값으로 미분.
    3. **역전파:** 출력층 → 은닉층 → 입력층 순서로 각 층의 가중치에 대한 그래디언트를 연쇄 법칙을 이용해 계산.
- **계산 그래프 (Computational Graph):**
    - 연산을 노드(변수)와 엣지(연산)로 표현한 그래프.
    - 역전파 과정을 시각화하고 이해하는 데 도움.
    - TensorFlow, PyTorch 등 딥러닝 프레임워크의 기반.

## 15. 자동 미분 (Automatic Differentiation, Autodiff)

- 복잡한 함수에 대한 미분 값을 정확하고 효율적으로 계산하는 기법.
- **종류:**
    - **수치 미분 (Numerical Differentiation):** 근사값 계산, 오차 발생 가능. $f'(x) \approx (f(x+h) - f(x))/h$.
    - **기호 미분 (Symbolic Differentiation):** 수학 공식을 이용해 미분 표현 생성, 표현식이 매우 복잡해질 수 있음.
    - **자동 미분:**
        - **Forward Mode (순방향 모드):** 입력부터 출력까지 미분값 계산. 입력 변수 수에 비례하는 계산량.
        - **Reverse Mode (역방향 모드):** 출력부터 입력까지 미분값 계산 (역전파와 동일 원리). 출력 변수 수에 비례하는 계산량. 딥러닝에서 주로 사용 (손실은 스칼라, 파라미터는 다수).

## 16. 텐서플로우 (TensorFlow)

- 구글에서 개발한 오픈소스 머신러닝/딥러닝 프레임워크.
- **핵심 개념:**
    - **텐서 (Tensor):** 다차원 배열 (Numpy 배열과 유사).
    - **계산 그래프:** 연산의 흐름을 그래프로 정의.
- **텐서플로우 2.x 특징:**
    - **즉시 실행 (Eager Execution) 기본:** Define-by-Run 방식. 코드를 실행하면서 동적으로 그래프 생성. 디버깅 용이.
    - **Keras API 고수준 통합 (`tf.keras`):** 모델 구축, 학습, 평가를 쉽게 할 수 있는 사용자 친화적 API.
        - **Sequential API:** 레이어를 순차적으로 쌓는 간단한 모델.
        - **Functional API:** 다중 입력/출력, 공유 레이어 등 복잡한 모델 정의.
        - **Model Subclassing:** `tf.keras.Model` 클래스 상속하여 완전히 새로운 모델/레이어 정의.
    - **`tf.data`:** 효율적인 데이터 입력 파이프라인 구축.
    - **`tf.GradientTape`:** 자동 미분 수행 (특히 즉시 실행 환경에서).
    - **배포 및 최적화:** TensorFlow Lite (모바일/임베디드), TensorFlow Serving (서버), TensorFlow.js (웹), SavedModel (모델 직렬화).
    - **분산 학습 (Distribution Strategy):** 여러 GPU/TPU에서 모델 학습.
- **OOP + FP (객체지향 프로그래밍 + 함수형 프로그래밍):**
    - Keras 모델 정의 시 클래스 상속(`__init__`, `call`), 함수형 API 조합 등 유연한 프로그래밍 가능.
- **딥러닝 구성 요소:**
    - **Representation (표현):** 데이터를 어떻게 표현하고 모델을 어떻게 구성할 것인가.
    - **Optimization (최적화):** 손실 함수를 최소화하는 파라미터를 어떻게 찾을 것인가.
    - **Evaluation (평가):** 모델의 성능을 어떻게 측정할 것인가.
- **네트워크 프루닝 (Network Pruning):** 불필요한 가중치/뉴런 제거하여 모델 경량화.
- **배치 정규화 (Batch Normalization):** 각 레이어의 입력 분포를 정규화하여 학습 안정화 및 속도 향상 (Internal Covariate Shift 방지).

## 핵심 요약

- 딥러닝은 통계학, 뇌과학 등 다양한 분야의 아이디어에서 출발하여 오랜 역사 속에서 발전해왔습니다. 특히 퍼셉트론, 역전파, GPU의 등장이 중요한 변곡점이었습니다.
- **뉴런과 퍼셉트론:** 생물학적 뉴런을 단순화한 인공 뉴런 모델(퍼셉트론)은 입력, 가중치, 활성화 함수, 출력으로 구성되며, 초기에는 간단한 논리 게이트를 구현하는 데 사용되었습니다.
- **학습 규칙:** 헵의 규칙(연관 학습), 퍼셉트론 학습 규칙(실수 기반 학습), 델타 규칙(오차 최소화) 등 다양한 학습 방식이 제안되었습니다.
- **XOR 문제와 MLP:** 단층 퍼셉트론의 한계를 극복하기 위해 다층 퍼셉트론(MLP)이 등장했고, 이는 비선형 문제 해결의 가능성을 열었습니다.
- **활성화 함수:** 신경망에 비선형성을 부여하여 표현력을 높이는 핵심 요소입니다. (ReLU가 현재 주류)
- **역전파:** MLP의 가중치를 효율적으로 학습시키기 위한 핵심 알고리즘으로, 연쇄 법칙에 기반합니다.
- **최적화:** 경사 하강법을 기본으로 하며, 모멘텀, Adagrad, RMSProp, Adam 등 다양한 개선 알고리즘이 사용됩니다.
- **텐서플로우:** Keras를 중심으로 사용자 친화적인 API를 제공하며, 즉시 실행, 자동 미분, 다양한 배포 옵션 등을 지원하는 강력한 딥러닝 프레임워크입니다.
- **딥러닝의 본질:** "뉴럴 네트워크는 로지스틱 회귀를 스태킹한 것이다"라는 통계학적 관점도 있으며, 결국 "표현(Representation), 최적화(Optimization), 평가(Evaluation)"의 조합으로 이해할 수 있습니다.
