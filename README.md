# 17주차: 클라우드 기반 데이터 시각화 및 딥러닝

## 1. 프로젝트 개요

본 프로젝트는 TensorFlow 및 PyTorch를 활용한 딥러닝 모델 개발을 위한 전체적인 환경 구축 과정을 기록하고, 인공신경망(ANN), 심층 신경망(DNN), 그리고 합성곱 신경망(CNN)의 핵심 이론을 학습하며 실제 모델을 구현하는 것을 목표로 합니다. 아이그로스 학원의 강의 내용을 바탕으로, 심화 학습을 위한 추가 정보를 포함하여 체계적으로 정리합니다.

---

## 2. 딥러닝 개발 환경 구축

Windows 환경에서 WSL2(Windows Subsystem for Linux 2)를 기반으로 TensorFlow와 PyTorch를 위한 전문적인 개발 환경을 구축하는 과정을 단계별로 설명합니다.

### 2.1. WSL2 설치 및 설정

- **목표:** Windows에서 가상 머신보다 가볍고 성능이 뛰어난 Linux 환경을 사용하여 딥러닝 프레임워크를 효율적으로 설치하고 실행합니다.
- **왜 WSL2를 사용할까요?**
  - **성능:** 실제 Linux 커널 위에서 동작하므로 파일 시스템 접근 속도나 연산 성능이 기존 WSL1이나 가상 머신(VM)에 비해 월등히 뛰어납니다. 특히, GPU 연산이 필요한 딥러닝 환경에서 이점을 가집니다.
  - **통합성:** Windows 파일 탐색기에서 `\wsl$` 주소로 Linux 파일에 접근할 수 있고, Windows와 Linux 간의 클립보드 공유 등 통합된 사용 경험을 제공합니다.
- **주요 과정:**
  1. Windows Terminal 또는 PowerShell을 **관리자 권한**으로 실행합니다.
  2. `wsl --install` 명령어를 사용하여 WSL2와 기본 Ubuntu 배포판을 설치합니다.
  3. **[팁]** Microsoft Store에서 `Ubuntu 22.04 LTS` 등 원하는 Linux 배포판을 추가로 설치하여 용도별로 분리할 수 있습니다.
  4. WSL2 실행 후, 사용자 계정 및 암호를 설정하여 초기 설정을 완료합니다.
- **심화 학습:**
  - [Microsoft 공식 WSL 설치 문서](https://learn.microsoft.com/ko-kr/windows/wsl/install)

### 2.2. TensorFlow & PyTorch 가상 환경 구성

- **목표:** `conda`를 이용하여 프로젝트별로 독립된 Python 환경을 구성하여, 라이브러리 버전 충돌을 원천적으로 방지하고 재현 가능한(reproducible) 연구 환경을 만듭니다.
- **왜 가상 환경이 중요한가요?**
  - 프로젝트 A는 TensorFlow 2.5, 프로젝트 B는 2.10을 필요로 할 때 시스템에 하나의 버전만 설치되어 있다면 충돌이 발생합니다. 가상 환경은 각 프로젝트가 자신만의 독립된 라이브러리 공간을 갖게 하여 이 문제를 해결합니다.
- **주요 과정:**
  1. WSL2 터미널에 Anaconda 또는 경량 버전인 Miniconda를 설치합니다.
  2. **TensorFlow 환경 생성:** `conda create -n tf tensorflow-gpu` (NVIDIA GPU 사용 시) 또는 `conda create -n tf tensorflow` (CPU만 사용 시)
  3. **PyTorch 환경 생성:** `conda create -n torch pytorch torchvision torchaudio -c pytorch`
  4. **[팁] 자주 사용하는 Conda 명령어:**
     - `conda env list`: 생성된 가상 환경 목록 확인
     - `conda activate <env_name>`: 가상 환경 활성화
     - `conda deactivate`: 가상 환경 비활성화
     - `conda install <package_name>`: 현재 활성화된 환경에 패키지 설치
- **심화 학습:**
  - [Conda 공식 사용자 가이드](https://conda.io/projects/conda/en/latest/user-guide/index.html)

### 2.3. VS Code와 WSL 연동

- **목표:** Windows의 편리한 GUI를 가진 VS Code를 WSL2의 강력한 Linux 개발 환경과 완벽하게 통합하여, 코드 편집, 디버깅, 터미널 작업을 하나의 툴에서 수행합니다.
- **주요 과정:**
  1. VS Code에 **Remote - WSL** 확장 프로그램을 설치합니다. (필수)
  2. WSL2 터미널에서 프로젝트 폴더로 이동한 뒤, `code .` 명령어를 실행하면 VS Code가 해당 폴더를 기준으로 실행됩니다.
  3. **[핵심 기능]** VS Code의 터미널(Ctrl+`)을 열면 바로 WSL의 bash 쉘이 나타나며, 여기서 `conda activate` 등 모든 Linux 명령어를 사용할 수 있습니다. 또한, VS Code의 디버거를 사용하여 WSL 환경에서 실행되는 Python 코드를 직접 디버깅할 수 있습니다.
- **심화 학습:**
  - [VS Code Remote Development 공식 문서](https://code.visualstudio.com/docs/remote/remote-overview)

### 2.4. Git & GitHub 연동

- **목표:** 프로젝트의 모든 변경 사항을 체계적으로 추적하고, GitHub를 통해 코드를 안전하게 백업하며 다른 사람들과 협업할 수 있는 기반을 마련합니다.
- **주요 과정:**
  1. `git clone <repository_url>`: 원격 GitHub 저장소를 로컬 WSL 환경으로 복제합니다.
  2. **[팁] 기본적인 Git 작업 흐름:**
     - `git status`: 현재 변경 상태 확인
     - `git add <file_name>`: 변경된 파일을 다음 커밋에 포함시킬 준비 (Staging)
     - `git commit -m "커밋 메시지"`: 준비된 파일들을 하나의 의미 있는 변경 단위로 저장 (Commit)
     - `git push`: 로컬에 저장된 커밋들을 원격 GitHub 저장소에 업로드
- **심화 학습:**
  - [Pro Git (한국어 번역)](https://git-scm.com/book/ko/v2) - Git을 가장 깊이 있게 배울 수 있는 책

---

## 3. 인공신경망 이론

### 3.1. 인공신경망(ANN) 및 DNN(Deep Neural Network)

- **퍼셉트론(Perceptron):** 다수의 입력을 받아 하나의 출력을 내보내는 가장 단순한 형태의 인공 뉴런입니다. 선형적인 문제만 해결할 수 있다는 한계가 있습니다.
- **다층 퍼셉트론(MLP):** 입력층과 출력층 사이에 하나 이상의 은닉층(Hidden Layer)을 추가하여 퍼셉트론의 한계를 극복한 모델입니다. 은닉층이 깊어질수록 (2개 이상) **심층 신경망(DNN)**이라고 부릅니다.
- **활성화 함수(Activation Function):** 각 뉴런의 최종 출력 값을 결정하는 함수로, 신경망에 **비선형성(non-linearity)**을 부여하는 핵심적인 역할을 합니다. 만약 활성화 함수가 없다면, 신경망은 깊게 쌓아도 결국 하나의 선형 모델과 같아집니다.
  - **Sigmoid:** 출력을 0과 1 사이로 압축하지만, Gradient Vanishing 문제가 발생할 수 있습니다.
  - **ReLU (Rectified Linear Unit):** 현재 가장 널리 사용되는 활성화 함수로, 연산 속도가 빠르고 Gradient Vanishing 문제를 완화하는 효과가 있습니다.
- **역전파(Backpropagation):** 출력층에서 발생한 오차(Error)를 입력층 방향으로 거꾸로 전파하며 각 뉴런의 가중치(Weight)를 얼마나 수정해야 할지 계산하는 알고리즘입니다.
- **DNN 구현:** TensorFlow/Keras의 `Sequential` API를 사용하면 `model.add(Dense(...))`와 같은 직관적인 코드로 손쉽게 DNN 모델을 구축하고 `model.fit()`으로 학습시킬 수 있습니다.

### 3.2. 합성곱 신경망(CNN, Convolutional Neural Network)

- **CNN 아키텍처:** 인간의 시신경 구조를 모방하여 이미지의 공간적인 특징(spatial feature)을 효과적으로 추출하도록 설계된 신경망입니다.
  - **합성곱 층(Convolutional Layer):** 필터(Filter/Kernel)가 이미지를 순회하며(stride) 특정 패턴(예: 수직선, 곡선, 특정 색상 조합)을 감지하고, 그 결과를 **특징 맵(Feature Map)**으로 만듭니다.
  - **풀링 층(Pooling Layer):** 특징 맵의 해상도를 낮추어(Subsampling) 연산량을 줄이고, 이미지 내에서 객체의 위치가 조금 변하더라도 동일한 특징을 잡아낼 수 있도록 돕습니다. (e.g., Max Pooling, Average Pooling)
  - **완전 연결 층(Fully Connected Layer):** 합성곱과 풀링을 통해 추출된 고차원 특징들을 입력으로 받아, 최종적으로 이미지를 어떤 클래스로 분류할지 결정하는 역할을 합니다. (DNN과 동일한 구조)
- **CNN 모델 구현:** Keras의 `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense` 레이어를 조합하여 CNN 모델을 구축합니다. 이미지 데이터는 일반적으로 (가로, 세로, 채널) 형태의 3차원 텐서(Tensor)로 입력됩니다.

---

## 4. 신경망 구현 실습 (Iris 데이터셋)

`1_신경망의_구조.ipynb` 노트북의 실습 내용을 요약 및 보충 설명합니다. 이 실습은 동일한 Iris 데이터셋을 사용하여 다양한 프레임워크와 API로 신경망을 구현하며 각 방식의 차이점을 비교하는 것을 목표로 합니다.

### 4.1. 데이터 준비 (Scikit-learn)

- `sklearn.datasets.load_iris()`: 붓꽃 품종 분류를 위한 Iris 데이터셋을 불러옵니다.
- `sklearn.model_selection.train_test_split`: 불러온 데이터를 모델 학습에 사용할 훈련(train) 데이터와 모델 성능 평가에 사용할 테스트(test) 데이터로 분리합니다. 이는 모델이 학습하지 않은 데이터에 대해 얼마나 잘 일반화되는지 평가하기 위한 필수적인 과정입니다.

### 4.2. 다양한 프레임워크를 이용한 모델 구현

(내용은 이전과 동일하여 생략)

### 4.3. 모델 평가, 저장 및 로드

(내용은 이전과 동일하여 생략)

---

## 5. 신경망 모델 최적화

`2_신경망모델최적화.ipynb` 노트북의 실습 내용을 요약 및 보충 설명합니다. 모델의 성능을 높이기 위해 사용되는 다양한 최적화 기법들을 학습합니다.

### 5.1. 손실 함수 (Loss Function)

- **역할:** 모델의 예측이 실제 정답과 얼마나 다른지를 나타내는 지표입니다. 딥러닝 모델은 이 손실 함수의 값을 최소화하는 방향으로 가중치를 업데이트하며 학습을 진행합니다.
- **주요 손실 함수:**
  - **회귀(Regression) 문제:**
    - `mean_squared_error` (MSE): 실제 값과 예측 값의 차이의 제곱 평균. 가장 일반적으로 사용됩니다.
  - **분류(Classification) 문제:**
    - `binary_crossentropy`: 클래스가 2개일 때 (예: 합격/불합격) 사용합니다.
    - `categorical_crossentropy`: 클래스가 3개 이상이고, 타겟 변수가 원-핫 인코딩된 경우 사용합니다.
    - `sparse_categorical_crossentropy`: `categorical_crossentropy`와 동일하지만, 타겟 변수가 정수(integer) 형태일 때 사용합니다.

### 5.2. 옵티마이저 (Optimizer)

- **역할:** 손실 함수를 기반으로 계산된 기울기(gradient)를 사용하여, 모델의 가중치를 어떻게 업데이트할지 결정하는 알고리즘입니다. 어떤 옵티마이저를 사용하느냐에 따라 학습 속도와 성능이 크게 달라질 수 있습니다.
- **주요 옵티마이저:**
  - `SGD` (Stochastic Gradient Descent): 가장 기본적인 경사 하강법 알고리즘입니다.
  - `RMSprop`: 학습률을 적응적으로 조절하여 학습 안정성을 높입니다.
  - `Adam`: RMSprop과 Momentum 방식을 결합한 형태로, 현재 가장 널리 사용되는 옵티마이저 중 하나입니다.

### 5.3. 과적합(Overfitting) 방지 기법

- **과적합이란?** 모델이 훈련(train) 데이터에만 너무 치중하여 학습한 나머지, 새로운 데이터(test data)에 대해서는 예측 성능이 떨어지는 현상을 말합니다.
- **주요 해결책:**
  - **가중치 규제 (Weight Regularization):** 가중치의 값이 너무 커지지 않도록 제한하여 모델의 복잡도를 낮춥니다.
    - `L1 규제 (Lasso)`: 가중치의 절댓값에 비례하는 페널티를 부여합니다. 중요하지 않은 특성의 가중치를 0으로 만들어 특성 선택(feature selection) 효과를 가집니다.
    - `L2 규제 (Ridge)`: 가중치의 제곱에 비례하는 페널티를 부여합니다. 가중치의 크기를 전반적으로 작게 유지하여 모델을 부드럽게 만듭니다.
  - **드롭아웃 (Dropout):** 학습 과정에서 각 층의 일부 뉴런을 랜덤하게 비활성화하여, 모델이 특정 뉴런에 과도하게 의존하는 것을 방지합니다.
  - **조기 종료 (Early Stopping):** 검증(validation) 데이터셋의 손실 값이 더 이상 감소하지 않고 증가하기 시작하면 학습을 중단시켜 과적합을 방지합니다.
  - **배치 정규화 (Batch Normalization):** 각 층의 입력 분포를 평균 0, 분산 1로 정규화하여 학습을 안정시키고 속도를 향상시킵니다. 이는 규제와 유사한 효과를 주기도 합니다.

---

_이 문서는 아이그로스 학원의 딥러닝 강의 내용과 개인적인 학습 과정을 기록 및 요약한 것입니다._
