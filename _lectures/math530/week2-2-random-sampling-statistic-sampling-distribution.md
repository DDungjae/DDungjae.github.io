---
title: "Week2-2: Random Sampling, Statistic, Sampling Distribution"
date: 2026-09-17
course: "MATH530"
excerpt: "대한민국 국민 전체의 키의 평균이나, 표준편차 등을 알고 싶다고 해보자. 현실적으로 5000만명이 넘는 모든 국민에 대해서 키를 측정하는 것은 불가능하다. 우리는 이렇게 population에 대한 정보(통계량)을 알고 싶지만, population 전체에 대한 값을 아는 것이 불가능할…"
---

## Random Sample

대한민국 국민 전체의 키의 평균이나, 표준편차 등을 알고 싶다고 해보자. 현실적으로 5000만명이 넘는 모든 국민에 대해서 키를 측정하는 것은 불가능하다. 우리는 이렇게 population에 대한 정보(통계량)을 알고 싶지만, population 전체에 대한 값을 아는 것이 불가능할 때 sample data로 population에 대한 inference를 수행한다.

$$
\text{Statistical estimation/inference use sample data to learn about parameter }\theta\text{ of the population model.}
$$

여기서 $$\theta$$는 평균, 표준편차 등 다양한 통계량이 될 수 있다. 그렇다면, population을 대표할만한 부분집합을 뽑아야 하는데 우리는 이를 random sample이라고 부른다.

$$
\begin{aligned}
&\text{The random variables }X_1, \dots, X_n\text{ form a random sample }\underline{X}=(X_1, \dots, X_n)\text{ of size }n\text{ from a population with PDF, PMF }f_\theta(x)\text{ if }X_1, \dots, X_n\text{ are mutually independent and each }X_i\text{ has the same PDF, PMF }f_\theta(x).\\
&\text{Alternately, }X_1, \dots, X_n \text{ are called independent and identically distributed random variables with PDF, PMF }f_\theta(x).
\end{aligned}
$$

위의 정의에서 살펴보아야 하는 것은 샘플들이 identically distributed와 mutually independent의 성질을 따라야 한다는 것이다. 이 두 가지 조건은 모든 샘플이 $$f_\theta(x)$$라는 확률 분포를 가지는 population에서 뽑혀야 하며 각 샘플이 서로 영향을 주면 안된다는 것을 요구한다.

## Statistic

$$
\text{A statistic }T(\underline{X})\text{ is a real, vector valued function of the data }\underline{X}.
$$

이전까지는 확률에 대한 여러 가지 성질이 나타났다면 처음으로 statistic이라는 용어와 그 정의가 등장했다. Statistic이란 어느 random sample에 대한 함수라고 말하는데 population이 아니라 random sample에 대한 함수인 점이 중요해 보인다. 우리가 일상적으로 사용하는 "통계"라는 용어는 population에 대해서 지칭하는 경우가 많은데 정의상으로는 random sample에 대한 함숫값이다.

$$
\begin{aligned}
&\bar{X}=\frac{\sum X_i}{n}\\
&S^2=\frac{1}{n-1}\sum (X_i-\bar{X})^2
\end{aligned}
$$

위와 같이 statistic에 속하는 sample mean이나 variance 모두 random sample에 대한 함수 형태임을 볼 수 있다. 그리고 우리는 다음과 같은 정리를 통해 random sample을 활용한 statistic으로부터 population에 대한 정보도 간접적으로 알아낼 수 있다.

$$
\begin{aligned}
&\text{Let }X_1, \dots, X_n\text{ be a random sample from a population with mean }\mu\text{ and variance }\sigma^2<\infty.\\
&\text{Then, E}[\bar{X}]=\mu, \text{Var}[\bar{X}]=\frac{\sigma^2}{n}, \text{E}[S^2]=\sigma^2.
\end{aligned}
$$

이를 보면 sample mean의 기댓값은 population mean과 같고, sample variance의 기댓값은 population variance와 같다는 것을 알 수 있다. 나아가 sample mean의 분산($$\text{Var}[\bar{X}]$$)이 $$\frac{\sigma^2}{n}$$이므로, sample이 충분히 많아질수록 오차가 0으로 줄어들어 표본평균이 실제 모집단의 참값으로 정확하게 수렴한다는 의의를 가진다.

## Estimating Sampling Distribution

우리는 random sample로부터 sample mean의 distribution을 추정해야 하는 경우가 있다. $$Y=\sum X_i, \bar{X}=\frac{Y}{n}$$의 구조로 sample mean을 구하는데 sample mean의 distribution을 구하는 방법에는 여러 가지가 있다.

### Method 1: MGF technique (assuming MGF exists)

$$
\begin{aligned}
&X\sim N(\mu, \sigma^2)\\
&M_X(t)=\text{exp}\{\mu t+\frac{1}{2}\sigma^2t^2\}, t\in \mathbb{R}\\
&M_Y(t)=\sum^n_{i=1}M_{X_i}(t)=\{M_X(t)\}^n=\text{exp}\{n\mu t+\frac{1}{2}n\sigma^2t^2\}\\
&Y\sim N(n\mu, n\sigma^2)\\
&M_{\bar{X}}(t)=\text{E}[e^{t\bar{X}}]=\text{E}[e^{\frac{t}{n}y}]=M_Y(\frac{t}{n})\\
&\bar{X}\sim N(\mu, \frac{\sigma^2}{n})
\end{aligned}
$$

만약 어느 확률 변수의 MGF가 존재한다면 위와 같이 MGF를 사용해서 sample mean을 구할 수 있다. 특히 우리는 이전에 독립인 두 확률 변수에 대해서 확률 변수의 합에 대한 MGF는 두 MGF의 곱으로 표현할 수 있음을 알았다. 이 성질을 사용해서 위의 예시와 같이 정규분포와 같은 함수에서 sample mean에 대한 분포를 쉽게 구할 수 있다.

### Method 2: Convolution Formula (for continuous random variable)

$$
\begin{aligned}
&\text{Suppose }X_1\sim f_{X_1}(x) \text{ and }X_2\sim f_{X_2}(x)\text{ are two continuous random variables and independent.}\\
&\text{Then, the PDF of }Y=X_1+X_2\text{ is }f_Y(y)=\int^\infty_{-\infty}f_{X_1}(w)f_{X_2}(y-w)dw=\int^\infty_{-\infty}f_{X_2}(w)f_{X_1}(y-w)dw
\end{aligned}
$$

위와 같이 convolution을 사용해서도 확률 분포간의 합으로 생성되는 분포를 구할 수 있고 이를 확장해 sample mean의 distribution을 추정할수도 있다. 예를 들어 Cauchy 분포에서 기댓값이 발산하기 때문에 정의가 되지 않지만 위의 방법을 사용하면 sample mean의 분포는 구할 수 있다.

$$
\begin{aligned}
&X_1, X_2\sim \text{Cauchy}(0, 1)\text{ where }f_{0, 1}(x)=\frac{1}{\pi}\frac{1}{1+x^2}\\
&f_Y(y)=f_{X_1+X_2}(y)=\int^\infty_{-\infty}f_{X_1}(w)f_{X_2}(y-w)dw=\int^\infty_{-\infty}\frac{1}{\pi}\frac{1}{1+w^2}\frac{1}{\pi}\frac{1}{1+(y-w)^2}dw=\frac{1}{2\pi}\frac{1}{1+\frac{y^2}{4}}\\
&X_1+X_2\sim \text{Cauchy}(0, 2)\\
&\text{Using induction, then we can show }Y=X_1+\cdots+X_n\sim \text{Cauchy}(0, n)\\
&f_{\bar{X}}(z)=f_Y(nz)n=\frac{1}{\pi}\frac{1}{1+z^2}\\
&\text{Thus, }\bar{X}\sim\text{Cauchy}(0, 1)
\end{aligned}
$$

### Method 3: Distribution of Certain Sum in Exponential Family

$$
\begin{aligned}
&\text{Suppose }X_1, \dots, X_n \text{ is a random sample from an exponential family of distributions.}\\
&\text{ Define }T_1, \dots, T_k \text{ by }T_i(X_1, \dots, X_n)=\sum^n_{j=1}t_i(x_j)\text{ where }(i=1, \dots, k).\\
&\text{Assume }\{w_1(\theta), \dots, w_k(\theta)\} \text{ contains an open subset of }\mathbb{R}^k\text{ and }T_1, \dots, T_k\text{ do not satisfy any linear constraint.}\\
&\text{Then the joint distribution of }T_1, \dots, T_k\text{ is an exponential family of the form }f_{T_1, \dots, T_k}(t_1, \dots, t_k;\theta)=H(t_1, \dots, t_k)[c(\theta)]^n\text{exp}\{\sum w_i(\theta)t_i\}
\end{aligned}
$$

Exponential family에서는 위와 같은 방법을 사용할 수 있다.
