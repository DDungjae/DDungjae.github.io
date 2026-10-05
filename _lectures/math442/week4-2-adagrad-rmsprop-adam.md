---
title: "Week4-2: AdaGrad,RMSProp, Adam"
date: 2026-10-01
course: "MATH442"
excerpt: "GD, AdaGrad, RMSProp, Adam으로 이어지는 최적화 기법의 발전"
tags: ["optimization", "machine-learning", "statistics"]
---

## AdaGrad Review

Sparse online learning에서 AdaGrad가 GD보다 좋은 성능을 가짐을 수식적으로 보였다. 하지만 AdaGrad 역시 단점을 가진다.

$$
\begin{aligned}
&\theta_{t+1}(i)=\theta_t(i)-\frac{\eta g_t(i)}{\sqrt{v_t(i)}+\epsilon}\text{ where }v_t(i)=\sum^t_{r=1}g_r(i)^2\\
&\theta_t=\begin{pmatrix}\theta_t(1)&\cdots &\theta_t(n)\end{pmatrix}^\top \in \mathbb{R}^n
\end{aligned}
$$

AdaGrad는 이전 그래디언트를 누적해 제곱합을 하면서 sparsity에 따라서 learning rate을 보정하는데 스텝이 진행됨에 따라 분모가 계속 커져 결국 learning rate가 소실될 수 있다. 이는 최적해에 도달하기 전 learning rate가 소실되어 최적해에 도달할 수 없다는 문제를 야기할 수 있다. 이러한 AdaGrad의 단점을 개선한 방법으로 RMSProp이 있다.

## RMSProp

$$
\theta_{t+1}(i)=\theta_t(i)-\frac{\eta g_t(i)}{\sqrt{v_t(i)}+\epsilon}\text{ where }v_t(i)=\beta v_{t-1}(i)+(1-\beta)g_t(i)^2=(1-\beta)\sum^t_{j=1}\beta^{t-j}g_j(i)^2
$$

RMSProp은 AdaGrad와 유사하게 이전 스텝의 그래디언트를 반영해서 learning rate을 조절하지만 AdaGrad가 단순히 제곱합을 사용했다면 RMSProp은 exponential moving average를 사용해 가까운 스텝의 그래디언트를 더 강하게 반영한다. 이 방법은 AdaGrad가 가지고 있던 문제점인 learning rate 소실 문제를 해결한다. AdaGrad에서의 learning rate은 단조감소하는 반면, RMSProp에서의 learning rate에서는 이전 시점 그래디언트에 따라서 증가하기도, 감소하기도 하며 learning rate가 비교적 일정하게 유지되어 소실 문제를 피할 수 있다.

## Adam

RMSProp 보다도 개선이 된 최적화 방법으로 Adam이 있다.

$$
\begin{aligned}
&m_t^i=\beta_1m_{t-1}^i+(1-\beta_1)g_t^i\\
&v_t^i=\beta_2v_{t-1}^i+(1-\beta_2)(g_t^i)^2\\
&\theta_{t+1}^i=\theta_t^i-\eta\frac{m_t^i}{\sqrt{v_t^i}+\epsilon}
\end{aligned}
$$

Adam은 RMSProp에서의 exponential moving average로 learning rate를 조절한다. 추가로 분자를 보면 이전 스텝의 그래디언트도 exponential moving average로 반영해 현재 학습 방향에 반영을 한다. $$m_t$$를 보통 1차 모멘텀, $$v_t$$를 2차 모멘텀이라고 부른다. Adam에 대한 자세한 설명은 이전에 논문 리뷰에서 다룬 적이 있으니 이 부분을 참고하면 될 것 같다.

[https://ddungjae.github.io/posts/adam-stochastic-optimization/](https://ddungjae.github.io/posts/adam-stochastic-optimization/)

## Feature of Loss Function Affecting GD

GD, AdaGrad, RMSProp, Adam으로의 발전은 다양한 형태의 loss function에서 최적화가 가능하게 했다. 그렇다면 가장 기본적인 GD는 loss function에 따라 어떻게 영향을 받을까?

우선 loss function이 $$l_2-\text{smooth}$$ 여서 다음과 같은 성질을 만족한다고 해보자.

$$
\begin{aligned}
&L(\theta+\Delta)\le L(\theta)+g^\top\Delta+\frac{L_2}{2}\lVert\Delta\rVert^2_2\\
&\rightarrow \lVert \nabla L(\theta+\Delta)-\nabla L(\theta) \rVert_2\le L_2\lVert\Delta\rVert_2
\end{aligned}
$$

이때 우리는 GD에서 $$\operatorname*{argmin}_\Delta(g^\top\Delta+\frac{L_2}{2}\lVert\Delta\rVert^2_2)=-\frac{1}{L_2}g$$인 $$\Delta$$를 찾기를 원한다. 즉 이 $$\Delta$$가 바로 GD의 업데이트 방향이다. 다음으로, loss function이 $$l_\infty-\text{smooth}$$라고 생각해보자.

$$
\begin{aligned}
&L(\theta+\Delta)\le L(\theta)+g^\top\Delta+\frac{H_\infty}{2}\lVert\Delta\rVert^2_\infty\\
&\rightarrow \operatorname*{argmin}_\Delta(g^\top\Delta+\frac{H_\infty}{2}\lVert\Delta\rVert^2_\infty)=-\frac{\lVert g\rVert_1}{H_\infty}\text{sign}(g)
\end{aligned}
$$

결국 GD의 업데이트 방향 $$-g/L_2$$가 최적인 것은 loss function의 곡률이 모든 방향에서 비슷하다는 가정 아래에서다. 하지만 실제로는 방향마다 곡률이 크게 다른 경우가 많은데(Week3-1에서 다룬 condition number와 같은 문제), 이럴 때는 $$l_2$$ 기준 대신 $$l_\infty$$ 기준으로 smoothness를 생각하면 최적 업데이트 방향이 그래디언트의 부호만 보고 좌표별로 같은 크기만큼 움직이는 형태가 된다. 이는 AdaGrad, RMSProp, Adam이 좌표마다 다른 크기로 나눠서 learning rate를 맞추는 것과 같은 방향이고, 결국 이 최적화 기법들의 발전은 GD가 가정하는 smoothness가 실제 loss function과 맞지 않을 때 그에 맞는 norm으로 다시 유도한 결과라고 볼 수 있다.
