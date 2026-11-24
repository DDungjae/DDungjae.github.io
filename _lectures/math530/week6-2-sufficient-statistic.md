---
title: "Week 6-2: Sufficient Statistic"
date: 2026-10-15
course: "MATH530"
order: 6.02
excerpt: "(Minimal) Sufficient statistic을 찾는 방법"
---

## Sufficient Statistic in Exponential Family

Exponential family는 이들만의 형태를 통해서 보다 쉽게 sufficient statistic을 얻을 수 있다. Exponential family는 아래와 같이 표현할 수 있음을 이전에 다루었었다.

$$
f_\theta(x)=h(x)c(\theta)\exp\left[\sum_{i=1}^k w_i(\theta)t_i(x)\right]
$$

이때 $$T(\underbar{X})=\left(\sum_{j=1}^n t_1(X_j), \dots, \sum_{j=1}^n t_k(X_j)\right)$$는 파라미터 $$\theta$$에 대한 sufficient statistic이 된다. 이에 대한 증명은 factorization theorem으로 쉽게 보일 수 있다. 이에 대한 예시로 $$X_i\sim\mathcal{N}(\mu, \sigma^2)$$를 살펴보자. 정규 분포의 pdf를 exponential family 형태로 나타내면 다음과 같이 나타난다.

$$
f_\theta(x)=(2\pi)^{-\frac{1}{2}}(\sigma^2)^{-\frac{1}{2}}\exp\left[-\frac{\mu^2}{2\sigma^2}\right]\exp\left[-\frac{1}{2\sigma^2}x^2+\frac{\mu}{\sigma^2}x\right]
$$

위와 같은 형태에서 $$t_1(x)=x^2, t_2(x)=x$$이므로 $$T(\underbar{X})=(\sum X_i^2, \sum X_i)$$는 $$(\mu, \sigma^2)$$에 대한 sufficient statistic이 된다. $$T(\underbar{X})=(\bar{X}, S^2)$$도 우리가 알고 있는 sufficient statistic인데 $$T(\underbar{X})=(\sum X_i^2, \sum X_i)$$에 대한 function으로 나타낼 수 있고 다시 한번 sufficient statistic이 유일하지 않음을 보여준다.

## Minimal Sufficient Statistic

$$
\text{Def) A sufficient statistic }T(\underbar{X})\text{ is a minimal sufficient statistic if, for any other sufficient statistic }T'(\underbar{X})\text{ and for every }\underbar{x}, \underbar{y}\in \mathcal{X},\ T(\underbar{x})=T(\underbar{y})\text{ whenever }T'(\underbar{x})=T'(\underbar{y}).
$$

Minimal sufficient statistic의 정의를 풀어서 보면, $$T'(\underbar{x})=T'(\underbar{y})$$일 때 항상 $$T(\underbar{x})=T(\underbar{y})$$라는 것은 $$T(\underbar{X})$$를 다른 어떤 sufficient statistic $$T'(\underbar{X})$$에 대한 function으로 나타낼 수 있다는 것을 의미한다. 즉, $$T(\underbar{X})=g(T'(\underbar{X}))$$를 만족하는 함수 $$g$$가 존재한다는 것이다. Sufficient statistic은 파라미터 $$\theta$$에 대한 정보를 잃지 않으면서 데이터 $$\underbar{X}$$를 요약(data reduction)하는 것이라고 볼 수 있다. 이때 statistic에 function을 한번 더 취하면 같은 값으로 묶이는 $$\underbar{x}$$들이 많아지면서 데이터가 더 압축되므로, minimal sufficient statistic은 sufficient statistic 중에서 data reduction을 가장 많이 한 statistic이라고 이해할 수 있다.

## How Do We Find a Minimal Sufficient Statistic?

그렇다면 minimal sufficient statistic은 어떻게 찾을 수 있을까? 어느 statistic $$T(\underbar{X})$$에 대해, 모든 sample point $$\underbar{x}, \underbar{y}$$에서 파라미터 $$\theta$$에 무관한 상수 $$c=C(\underbar{x}, \underbar{y})$$가 존재해 모든 $$\theta\in \Theta$$에 대해서 $$f_\theta(\underbar{x})=cf_\theta(\underbar{y})$$일 필요충분조건이 $$T(\underbar{x})=T(\underbar{y})$$라면, $$T(\underbar{X})$$는 minimal sufficient statistic이 된다. 여기서 $$f_\theta(\underbar{x})=cf_\theta(\underbar{y})$$는 두 pdf의 상수배 차이 $$c$$가 $$\theta$$에 의존하지 않는다는 것이다. 즉, $$\theta$$가 어떤 값이든 $$\underbar{x}$$와 $$\underbar{y}$$의 pdf는 항상 같은 비율로만 차이 나기 때문에, 두 sample point는 $$\theta$$에 대해 같은 정보를 준다고 볼 수 있다.

이에 대한 예시로, 앞에서 sufficient statistic을 구했던 $$X_1, \dots, X_n \overset{iid}{\sim}\mathcal{N}(\mu, \sigma^2)$$를 이어서 살펴보자. 앞의 exponential family 형태를 이용하면 sample point $$\underbar{x}$$의 joint pdf는 다음과 같다.

$$
f_\theta(\underbar{x})=(2\pi)^{-\frac{n}{2}}(\sigma^2)^{-\frac{n}{2}}\exp\left[-\frac{n\mu^2}{2\sigma^2}\right]\exp\left[-\frac{1}{2\sigma^2}\sum_{i=1}^n x_i^2+\frac{\mu}{\sigma^2}\sum_{i=1}^n x_i\right]
$$

따라서 두 sample point $$\underbar{x}, \underbar{y}$$에 대한 pdf의 비율은 다음과 같이 나타난다.

$$
\frac{f_\theta(\underbar{x})}{f_\theta(\underbar{y})}=\exp\left[-\frac{1}{2\sigma^2}\left(\sum_{i=1}^n x_i^2-\sum_{i=1}^n y_i^2\right)+\frac{\mu}{\sigma^2}\left(\sum_{i=1}^n x_i-\sum_{i=1}^n y_i\right)\right]
$$

이 비율이 $$(\mu, \sigma^2)$$에 무관하려면 $$\sum x_i^2=\sum y_i^2$$이고 $$\sum x_i=\sum y_i$$이어야 한다. 따라서 $$T(\underbar{X})=(\sum X_i^2, \sum X_i)$$는 $$(\mu, \sigma^2)$$에 대한 minimal sufficient statistic이 된다. 또한 $$(\bar{X}, S^2)$$는 $$(\sum X_i^2, \sum X_i)$$와 일대일 대응 관계이므로 역시 minimal sufficient statistic이 된다.

또 다른 예시로 $$X_i\sim\text{Unif}(\theta, \theta+1)$$를 살펴보자. $$x_{(1)}=\min_i x_i$$, $$x_{(n)}=\max_i x_i$$라 하면 sample point $$\underbar{x}$$의 joint pdf는 다음과 같다.

$$
\begin{aligned}
f_\theta(\underbar{x})&=\prod_{i=1}^n \mathbf{I}_{(\theta<x_i<\theta+1)} \\
&=\mathbf{I}_{(x_{(n)}-1<\theta<x_{(1)})}
\end{aligned}
$$

이때 joint pdf는 $$\theta\in(x_{(n)}-1, x_{(1)})$$일 때 1이고 그 외에는 0이 된다. $$f_\theta(\underbar{x})=cf_\theta(\underbar{y})$$가 모든 $$\theta$$에 대해 성립하려면 두 pdf가 1이 되는 $$\theta$$의 구간이 같아야 하므로 $$x_{(1)}=y_{(1)}$$이고 $$x_{(n)}=y_{(n)}$$이어야 한다. 따라서 $$T(\underbar{X})=(X_{(1)}, X_{(n)})$$는 $$\theta$$에 대한 minimal sufficient statistic이 된다. 이 예시는 파라미터가 하나이더라도 minimal sufficient statistic의 차원은 2가 될 수 있음을 보여준다.
