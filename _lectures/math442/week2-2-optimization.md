---
title: "Week2-2: Optimization"
date: 2026-09-17
course: "MATH442"
excerpt: "이번 시간부터는 neural network와 손실함수 최적화로 들어간다. 본격적으로 머신러닝에 관한 내용이 나와 기대가 된다. 이전까지 행렬 연산만 계속 하던게 살짝 머리아팠는데 이제 내가 MATH442를 듣고 싶었던 이유와 직접적으로 관련되는 내용이 수업에 등장한다."
tags: ["optimization", "machine-learning"]
---

이번 시간부터는 neural network와 손실함수 최적화로 들어간다. 본격적으로 머신러닝에 관한 내용이 나와 기대가 된다. 이전까지 행렬 연산만 계속 하던게 살짝 머리아팠는데 이제 내가 MATH442를 듣고 싶었던 이유와 직접적으로 관련되는 내용이 수업에 등장한다.

## Universal Approximation Theorem

$$
\begin{aligned}
&\text{Let } f \in \text{C}(\mathbf{K}, \mathbb{R}^m), \quad \sigma \in \text{C}(\mathbb{R}, \mathbb{R}). \\[4pt]
&\text{Then for any } m, n \in \mathbb{N}, \text{ compact } \mathbf{K} \subseteq \mathbb{R}^n, \text{ and } \epsilon > 0, \\[4pt]
&\text{there exists } k \in \mathbb{N},\ \mathbf{A} \in \mathbb{R}^{k \times n},\ \mathbf{b} \in \mathbb{R}^k,\ \mathbf{C} \in \mathbb{R}^{m \times k} \text{ such that} \\[4pt]
&\qquad \sup_{\vec{x} \in \mathbf{K}} \big\lVert \mathbf{C}\,\sigma(\mathbf{A}\vec{x}+\mathbf{b}) - f(\vec{x}) \big\rVert < \epsilon.
\end{aligned}
$$

우선 함수 $$f, \sigma$$는 위와 같은 집합에서 정의된 연속함수이다. $$\sigma$$는 보통 머신러닝에서 활성화함수(activation function)라고 부르며 sigmoid, ReLU 등의 비선형 함수를 사용한다. Universal Approximation Theorem이 함의하는 것은 neural network이 충분히 큰 $$k$$에 대해서 임의의 함수를 $$\epsilon$$ 만큼의 오차 이하로 근사할 수 있다는 것이다. 한가지 살펴봐야 할 것은 위 수식에서 활성화 함수는 한 번만 등장한다. 그리고 이와 같은 구조는 입력$$(\vec{x})$$→입력 가중치$$(\mathbf{A})$$→활성화함수$$(\sigma)$$→출력 가중치$$(\mathbf{C})$$와 같은 흐름을 한 번만 거쳐서 출력을 생성한다. 이는 현대적인 deep neural network와는 차이가 있는데 deep neural network에서는 활성화 함수의 출력이 다시 가중치와 곱해져 활성화 함수의 입력으로 들어가며 여러 단계를 거친다. Universal Approximation Theorem은 활성화 함수를 한 번만 통과해도 $$k$$가 충분히 크다면 근사가 가능함을 수학적으로 보여줬지만 이는 여러 비효율성 때문에 현대 딥러닝의 구조인 deep neural network로 발전하게 되었다.

## Gradient Descent

UAT 덕분에 우리는 어느 종류의 함수이든 neural network로 근사할 수 있다는 확신을 가지게 되었다. 하지만 이 함수가 가지는 파라미터를 결정하는 것은 또 다른 문제이다. 함수가 존재하는 것은 알지만 무엇인지 모른다면 의미가 없다. 우리가 함수를 얼마나 잘 근사했는지를 평가하는 지표로 주로 loss function을 사용한다. 가장 많이 사용하는 loss function의 형태로 squared-error가 있다.

$$
L(\theta)=\frac{1}{N}\sum^N_{i=1}\big\lvert y(x_i;\theta)-\hat{y}_i\big\rvert^2
$$

여기서 $$\theta$$는 파라미터, $$y(x_i;\theta)$$는 파라미터로 출력한 예측값, $$\hat{y}_i$$는 실제 값, $$N$$은 데이터 수이다. 결국 우리는 이 loss function을 최소화 하도록 하는 파라미터 조합인 $$\theta$$를 찾아야 한다.

$$
\theta^{*}=\operatorname*{argmin}_{\theta}\, L(\theta)
$$

$$\theta$$를 찾기 위해서 보통 gradient descent를 사용한다. 우선 가장 기본적인 형태의 gradient descent부터 살펴보자.

$$
\theta_{k+1}=\theta_k-\rho\,\nabla_\theta L(\theta_k)
$$

우리는 어떤 함수 $$f$$에서 local minimum을 향해 가장 빠르게 감소하는 방향이 $$-\nabla f$$라는 것을 안다. 이를 사용해서, 우리는 loss function이 감소하는 방향으로 파라미터 $$\theta$$를 업데이트 한다. 이때 $$\rho$$는 파라미터를 얼마나 크게 업데이트할지 정하는 learning rate로 해석할 수 있다.

## Stochastic Gradient Descent (SGD)

Gradient descent는 한번 gradient를 계산할 때 모든 data point를 사용해야 한다. 하지만 SGD에서는 batch를 나눠서 그 batch에 해당하는 data point에서만 loss를 계산한다.

$$
\theta_{k+1}=\theta_k-\frac{\rho}{\lvert\mathbf{B}\rvert} \sum_{i\in\mathbf{B}}\nabla_\theta L_i(\theta_k)
$$

이때 reasonable한 SGD가 되려면, batch에 대한 평균 gradient가 전체 데이터에 대한 gradient의 불편추정량(unbiased estimator)이 되어야 한다. 즉 기댓값을 취했을 때 다음이 성립해야 한다.

$$
\mathbb{E}\left[\frac{1}{\lvert\mathbf{B}\rvert} \sum_{i\in\mathbf{B}}\nabla_\theta L_i(\theta)\right]=\frac{1}{N}\sum^N_{i=1}\nabla_\theta L_i(\theta)
$$

SGD는 기존에 gradient descent에서 정의한 loss와 비교하면 적은 수의 data point만 가지고 loss를 계산하기 때문에 연산 속도에서 이점이 생긴다. 또한 gradient descent는 전역 최적해가 아닌 local minimum에 갇힐 수 있는데 SGD는 local minimum은 피할 정도의 적당한 노이즈가 동반되어 전역 최적해에 더 잘 도달한다.
