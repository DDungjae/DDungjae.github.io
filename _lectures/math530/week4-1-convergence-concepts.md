---
title: "Week4-1: Convergence concepts"
date: 2026-09-29
course: "MATH530"
excerpt: "확률변수가 수렴한다는 것을 어떻게 표현할 수 있을까?"
tags: ["statistics"]
---

우리는 large sample theory에 따라서 다양한 random variable으로 이루어진 수열이 특정 분포로 수렴하는 것을 보일 수 있다. 이는 random variable이 우리가 정보를 알고 있는 분포로 수렴할 때 보다 쉽게 해당 random variable로 이루어진 수열을 해석할 수 있게 해준다. Random variable로 이루어진 수열에 대한 수렴성은 여러 종류가 있다.

- Convergence in probability
- Convergence in distribution
- Almost sure convergence
- $$L^p$$ convergence

## Convergence in Probability

$$
\begin{aligned}
&\text{A sequence of random variables }Y_1, Y_2, \cdots \text{ converges in probability to a random variable }Y \\
&\text{ if for every }\epsilon\gt0, \lim_{n\rightarrow\infty}\text{P}(\lvert Y_n-Y\rvert\ge\epsilon)=0 \Leftrightarrow \lim_{n\rightarrow\infty}\text{P}(\lvert Y_n-Y\rvert\lt\epsilon)=1
\end{aligned}
$$

Convergence in probability는 random variable로 이루어진 수열이 $$(Y-\epsilon, Y+\epsilon)$$ 범위로 $$n$$이 충분히 클때 1의 확률로 수렴한다는 의미를 가진다. $$\epsilon$$을 사용해서 수렴성을 보이는 것이 일반적인 수열에서의 수렴과 유사한 것으로 보이는데 수렴한다는 것을 확률이 1인 것으로 표현한다는게 특징인것 같다.

## Weak Law of Large Numbers

$$
\begin{aligned}
&\text{Let }X_1, X_2, \cdots, X_n \text{ be a random sample from a population }X \text{ with mean and finite variance }\mu, \sigma^2.\\
&\text{Then }\bar{X_n}=\frac{\sum^n_{i=1}X_i}{n}\xrightarrow{P}\mu.
\end{aligned}
$$

Weak law of large numbers는 표본이 충분히 많을 때 표본평균이 모평균으로 수렴한다는 것을 말해준다. 이를 증명하기 위해서 Chebychev's inequality가 사용된다.

$$
\text{P}(g(x)\ge r)\le \frac{\text{E}g(x)}{r}
$$

이를 사용해서 위의 weak law of large numbers를 증명해보자.

$$
\begin{aligned}
&\text{pf) Given }\epsilon \gt0, \text{ show }\lim_{n\rightarrow \infty}\text{P}(\lvert\bar{X_n}-\mu\rvert\ge \epsilon)=0\\
&0\le\text{P}(\lvert\bar{X_n}-\mu\rvert\ge\epsilon)=\text{P}((\bar{X_n}-\mu)^2\ge\epsilon^2)\le \frac{\text{E}[(\bar{X_n}-\mu)^2]}{\epsilon^2}=\frac{\text{Var}[\bar{X_n}]}{\epsilon^2}=\frac{\sigma^2}{n\epsilon^2}\rightarrow0\text{ as }n\rightarrow\infty.
\end{aligned}
$$

즉 $$\bar{X_n}\approx \mu$$가 충분히 큰 $$n$$에 대해 성립한다. 이렇게 우리는 $$X$$에 대한 함수로 특정 파라미터인 $$\mu$$를 추정할 수 있고 estimator과 파라미터에 대해서 $$T_n\xrightarrow{P}\theta\text{ as } n\rightarrow\infty$$이라면 $$T_n$$을 $$\theta$$에 대한 consistent estimator라고 부른다. 추가적으로 어느 확률 분포가 수렴하면 여기에 함수를 취한 값도 수렴함을 알 수 있다.

$$
\begin{aligned}
&\text{Suppose that }X_n\xrightarrow{P} X\text{ and }h\text{ is a continuous function.}\\
&\text{Then }h(X_n)\xrightarrow{P} h(X)
\end{aligned}
$$

예를 들어 $$S_n^2$$가 $$\sigma^2$$의 consistent estimator이기에 함수를 취한 $$S_n=\sqrt{S_n^2}$$는 $$\sigma$$에 대한 consistent estimator가 된다.

## Convergence in Distribution

Convergence in probability보다 약한 수렴 조건으로 convergence in distribution이 있다.

$$
\begin{aligned}
&\text{A sequence of random variables }X_1, X_2, \cdots\text{ converges in distribution to a random variable }X\\
&\text{if }\lim_{n\rightarrow\infty}F_{X_n}(x)=F_X(x)\text{ at all points }x \text{ where }F_X(x)\text{ is continuous.}
\end{aligned}
$$

Convergence in distribution이 convergence in probability보다 약하다고 말하는 이유는 convergence in probability는 convergence in distribution을 보장하지만 그 역은 성립하지 않기 때문이다.

$$
\begin{aligned}
&\text{pf) Let }t\text{ and }\epsilon\gt0\text{ be given.}\\
&\{X\le t-\epsilon\}\subseteq\{X_n\le t\}\cup\{\lvert X_n-X\rvert\ge\epsilon\}\text{, so }\text{P}(X\le t-\epsilon)\le\text{P}(X_n\le t)+\text{P}(\lvert X_n-X\rvert\ge\epsilon).\\
&\text{Similarly, }\{X_n\le t\}\subseteq\{X\le t+\epsilon\}\cup\{\lvert X_n-X\rvert\ge\epsilon\}\text{, so }\text{P}(X_n\le t)\le\text{P}(X\le t+\epsilon)+\text{P}(\lvert X_n-X\rvert\ge\epsilon).\\
&\text{Let }n\rightarrow\infty\text{. Since }X_n\xrightarrow{P}X\text{, }\text{P}(\lvert X_n-X\rvert\ge\epsilon)\rightarrow0\text{, and so}\\
&F_X(t-\epsilon)\le\liminf_{n\rightarrow \infty}F_{X_n}(t)\le\limsup_{n\rightarrow\infty}F_{X_n}(t)\le F_X(t+\epsilon).\\
&\text{Now, suppose }t\text{ is a continuity point of }F_X\text{. Letting }\epsilon\rightarrow 0\text{, }F_X(t)\le\liminf_{n\rightarrow\infty}F_{X_n}(t)\le\limsup_{n\rightarrow\infty}F_{X_n}(t)\le F_X(t)\Rightarrow\lim_{n\rightarrow \infty} F_{X_n}(t)=F_X(t).
\end{aligned}
$$
