---
title: "Week3-1: Sampling from Normal Distribution"
date: 2026-09-22
course: "MATH530"
excerpt: "정규분포에서 파생되는 다양한 분포"
tags: ["statistics"]
---

정규분포(normal distribution)은 가장 많이 사용되는 확률 분포 중 하나이고 정규분포에서 파생되는 다양한 분포가 있다.

## t-Distribution

$$X_1, \cdots, X_n\sim N(\mu, \sigma^2)$$이라고 해보자. 우리는 sample mean $$\bar{X}=\frac{1}{n}\sum^n_{i=1} X_i$$에 대해서 $$\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\sim N(0, 1)$$임을 안다. $$\mu, \sigma^2$$가 population mean, population variance라면 sample mean, variance는 다음과 같이 나타난다.

$$
\begin{aligned}
&\bar{X}=\frac{1}{n}\sum^n_{i=1} X_i\\
&S^2=\frac{1}{n-1}\sum^n_{i=1}(X_i-\bar{X})^2
\end{aligned}
$$

이때 t-distribution은 다음과 같이 정의가 된다.

$$
\frac{\bar{X}-\mu}{S/\sqrt{n}}\sim t_{n-1}
$$

현실 문제를 해결할 때 모집단의 표준편차를 알 수 있는 방법이 없을 때 표본 표준편차를 사용해 표준화를 진행해야 하는 경우가 있다. 이때 모집단의 표준편차를 사용해 표준화를 하는 것과 달리 표본 표준편차는 그 자체로 확률 변수이고 이로 인해 t-distribution은 normal-distribution과 비교해 heavy tail을 가지게 된다. t-distribution은 모분산을 모를 때 모평균에 대한 신뢰구간이나 가설검정에 사용된다.

## Chi-Squared Distribution

$$X_1, \cdots, X_n\sim N(\mu_X, \sigma_X^2)$$이라고 해보자. 새로운 분포 $$Y=\sum^n_{i=1}(\frac{X_i-\bar{X}}{\sigma_X})^2$$라고 정의하자. 이는 각 분포를 표준화 한 후 제곱해서 더한 값이다. 이 분포는 degree of freedom이 $$(n-1)$$인 chi-squared distribution을 따르게 된다. 위의 수식은 다음과 같이 변형이 된다.

$$
Y=\sum^n_{i=1}\left(\frac{X_i-\bar{X}}{\sigma_X}\right)^2=\frac{n-1}{\sigma_X^2}\frac{1}{n-1}\sum^n_{i=1}(X_i-\bar{X})^2=\frac{(n-1)S_X^2}{\sigma_X^2}\sim\chi^2_{n-1}
$$

즉 $$\frac{(n-1)S_X^2}{\sigma_X^2}\sim\chi^2_{n-1}$$와 같이 chi-squared distribution을 표현하는 방법도 있음을 알 수 있다. Chi-squared distribution은 모분산에 대한 신뢰구간이나 가설검정에 사용된다.

## F-Distribution

위의 chi-squared distribution에서 이어져서 $$X_1, \cdots, X_n\sim N(\mu_X, \sigma_X^2)$$, $$Y_1, \cdots, Y_m\sim N(\mu_Y, \sigma_Y^2)$$에 대해 파생된 chi-squared distribution을 생각해보자.

$$
\begin{aligned}
&\frac{(n-1)S_X^2}{\sigma_X^2}\sim\chi^2_{n-1}\\
&\frac{(m-1)S_Y^2}{\sigma_Y^2}\sim\chi^2_{m-1}\\
&\frac{S_X^2/\sigma_X^2}{S_Y^2/\sigma_Y^2}\sim\frac{\chi^2_{n-1}/(n-1)}{\chi^2_{m-1}/(m-1)}=F_{n-1, m-1}
\end{aligned}
$$

위와 같은 분포는 F-distribution을 따르게 된다. F-distribution은 두 모집단의 분산을 비교하는 가설검정에 사용된다.
