---
title: "Week 5-2: Delta Method"
date: 2026-10-08
course: "MATH530"
order: 5.02
excerpt: "테일러 정리를 활용한 확률 분포의 수렴의 근사"
tags: ["statistics"]
---

## Delta Method

Delta Method는 1, 2차 테일러 전개를 사용해 확률 분포의 수렴을 보이는 방법이다. 우선 first order delta method를 살펴보자. $$Y_n$$이라는 수열에 대해 $$\sqrt{n}(Y_n-\theta)$$가 $$\mathcal{N}(0, \sigma^2)$$로 converge in distribution 한다고 하자. 이때 함수 $$g$$와 특정 값 $$\theta$$에 대해 $$g'(\theta)$$가 0이 아닌 값으로 존재한다면 $$\sqrt{n}(g(Y_n)-g(\theta))$$는 $$\mathcal{N}(0, \sigma^2[g'(\theta)]^2)$$로 converge in distribution 하게 된다. 이는 1st order Taylor series expansion을 사용해 보일 수 있다.

$$
\sqrt{n}(g(Y_n)-g(\theta))\rightarrow \mathcal{N}(0, \sigma^2[g'(\theta)]^2)
$$

예를 들어, $$\sqrt{n}(\bar{X}_n-\mu)$$가 $$\mathcal{N}(0, \sigma^2)$$로 converge in distribution 한다고 생각해보자. 이때 $$\mu\neq 0$$이고 $$\sqrt{n}(\frac{1}{\bar{X}_n}-\frac{1}{\mu})$$의 수렴을 알고 싶다면 1st order delta method를 적용할 수 있다. $$g(\theta)=\frac{1}{\theta}$$라고 하면 $$g'(\theta)=-\frac{1}{\theta^2}$$가 되고 $$\sqrt{n}(\frac{1}{\bar{X}_n}-\frac{1}{\mu})$$은 $$\mathcal{N}(0, \sigma^2[g'(\mu)]^2)=\mathcal{N}(0, \frac{\sigma^2}{\mu^4})$$로 converge in distribution 하게 된다.

1st order delta method는 1차 미분 계수가 0이 아니어야 사용할 수 있는데 이에 대한 대안으로 2nd order delta method가 있다. $$g'(\theta)=0$$이고 $$g''(\theta)$$가 0이 아닌 값으로 존재한다면 다음이 성립한다.

$$
n(g(Y_n)-g(\theta))\rightarrow \sigma^2\frac{g''(\theta)}{2}\chi^2_1
$$

2nd order delta method는 카이제곱분포를 동반한 분포로 수렴함을 볼 수 있다. 이 또한 Taylor expansion을 사용해 보일 수 있고 정규분포의 제곱이 $$\chi^2_1$$을 따름을 알면 쉽게 유도할 수 있다.
