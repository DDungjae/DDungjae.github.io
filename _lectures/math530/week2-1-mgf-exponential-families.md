---
title: "Week2-1: Moment Generating Function, Exponential Families"
date: 2026-09-15
course: "MATH530"
excerpt: "MGF와 moment로 확률 분포 추정하기, 확률 분포를 exponential family 형태로 표현하기"
---

## Distributions of Functions of a Random Variable

우리가 일상적으로 "평균"이라고 말하는 것은 expected value이다. 기댓값이라고 번역이 되는데 말 그대로 집단에서 기대되는 값, 대표되는 값이다.

$$
\text{E}[g(X)]=
\begin{cases}
\displaystyle\int^\infty_{-\infty}g(x)f_X(x)\,dx, & \text{continuous} \\[6pt]
\displaystyle\sum_{x \in \mathcal{X}}g(x)f_X(x), & \text{discrete}
\end{cases}
$$

연속이냐, 이산이냐에 대해서 다르게 표현이 되며 적분이나, 합이 존재할 때 기댓값이 존재한다고 말한다. 예를 들어 코시 분포에서는 기댓값이 존재하지 않는다.

$$
f_X(x)=\frac{1}{\pi}\frac{1}{1+x^2}
$$

$$
\int^\infty_{-\infty}\frac{|x|}{\pi(1+x^2)}\,dx
= 2\int^\infty_{0}\frac{x}{\pi(1+x^2)}\,dx
= \frac{1}{\pi}\log(1+x^2)\Big|^{\infty}_{0}
\quad (\text{diverges})
$$

## Moments and Moment Generating Functions

기댓값은 위와 같이 구할 수도 있지만 moment로도 구할 수 있다.

$$
M_r' = \text{E}[X^r] \quad \text{(raw moment)}
$$

$$
M_r = \text{E}[(X-\mu)^r] \quad \text{(central moment)}
$$

이때 기댓값은 1st moment, 분산(variance)은 2nd central moment인 것을 알 수 있다. 그렇다면, moment이라는 것은 무엇이며 왜 사용하는걸까?

Moment generating function(MGF)는 다음과 같이 나타난다.

$$
M_X(t) = \text{E}[e^{tX}] = \int^{\infty}_{-\infty} e^{tx} f_X(x)\, dx
$$

MGF는 항상 존재하는 것은 아니고 적분값이 존재하도록 하는 $$t$$의 범위 안에서만 존재하게 된다. 그리고 MGF를 미분함을 통해서 moment를 얻을 수 있다.

$$
\left.\frac{d^r}{dt^r}M_X(t)\right|_{t=0} = M_r'
$$

Moment는 분포의 형태를 알려준다. 1~4차 moment는 각각 평균, 분산, 왜도, 첨도와 관련이 있는데 우리는 moment로부터 분포의 모양을 파악할 수 있다. 또한 moment는 확률 분포를 확정할 수 있도록 한다. 다음과 같은 정리를 보자.

Moment가 존재하는 $$F_X(x), F_Y(y)$$라는 두 CDF가 존재한다고 생각해보자.

- If $$X, Y$$ have bounded support, then $$F_X(u)=F_Y(u)$$ for all $$u$$ if and only if $$\text{E}[X^r]=\text{E}[Y^r]$$ for all positive integers $$r$$.
- If the MGFs exist and $$M_X(t)=M_Y(t)$$ for all $$t$$ in some neighborhood of $$0$$, then $$F_X(u)=F_Y(u)$$ for all $$u$$.

위의 두 성질은 유계인 구간에서 두 CDF가 동일할 필요충분조건이 모든 차수의 moment가 동일해야 한다는 것을 보여준다. 또한, 0 근방의 neighborhood에서 두 MGF가 수렴한다면 두 확률 분포가 동일하다는 것을 보여준다. MGF가 수렴한다는 것은

$$
M_X(t) = \sum^{\infty}_{k=0} \frac{\text{E}[X^k]}{k!}\,t^k
$$

가 역시 수렴한다는 것을 제공하며 moment만 알고 있다면 확률 분포를 완전히 재구성할 수 있도록 한다. 위 정리를 바탕으로, $$X$$와 $$Y$$가 서로 독립일 때 아래와 같이 두 확률 변수의 합에 대한 MGF를 쉽게 구할 수 있게 된다.

$$
M_{X+Y}(t) = M_X(t)\,M_Y(t) \qquad (X, Y \text{ independent})
$$

## Exponential Families

$$
f_\theta(x)=h(x)\,c(\theta)\,\exp\!\left\{\sum^k_{i=1}w_i(\theta)\,t_i(x)\right\}
$$

어느 확률 분포에 대한 PDF나 PMF를 위와 같이 $$x, \theta$$에 관해 표현할 수 있다면 이 계열의 확률 분포들을 exponential family라고 부른다. Exponential family에는 normal, gamma, beta, binomial, poisson, negative binomial 등이 포함된다. 특징을 보자면 $$h, t$$는 $$x$$에 관한 함수, $$c, w$$는 $$\theta$$에 관한 함수이며 각각 다른 인자에 영향을 받을 수 없다.

$$
f_\eta(x) = h(x)\,c^{*}(\eta)\,\exp\!\left\{\sum^k_{i=1}\eta_i\,t_i(x)\right\}
= h(x)\,\exp\!\left\{\sum^k_{i=1}\eta_i\,t_i(x) - A(\eta)\right\}
$$

만약 수식이 지수 부분에서 $$\eta$$에 대한 선형 결합으로 이루어져 있다면 위와 같은 형태는 canonical (natural) exponential family라고 부른다. $$\eta$$는 natural parameter라고 부르며 연속함수에서 $$\eta$$는 다음과 같은 집합에 속하게 된다.

$$
\eta \in H = \left\{\, \eta \;\middle|\; \int^\infty_{-\infty} h(x)\,\exp\!\left\{\sum^k_{i=1}\eta_i t_i(x)\right\} dx < \infty \,\right\}
$$

$$\eta$$가 위와 같은 집합에 존재해야 하는 이유는 $$f$$는 여전히 PDF이기 때문에 적분하면 1이 나와야 하고 $$x$$에 대해 적분이 되는 항들이 발산하면 안된다는 조건이다. 위의 집합 $$H$$는 "natural parameter space for exponential family"라고 부른다. 추가적으로 어느 확률분포의 support(domain)을 정의할 때 보통 $$\{x \mid f_\theta(x) > 0\}$$와 같이 표현하는데, 이 support가 $$\theta$$에 따라 달라지는 분포들은 exponential family가 될 수 없다.
