---
title: "Week 5-1: Slutsky’s Theorem"
date: 2026-10-06
course: "MATH530"
order: 5.01
excerpt: "확률 분포의 수렴성의 활용"
tags: ["statistics"]
---

## Slutsky’s Theorem

$$
\begin{aligned}
&\text{If }X_n \text{ converges in distribution to }X\text{ and }Y_n\text{ converges in probability to constant }a, \text{ then}\\
&\text{(i) }X_n+Y_n \text{ converges in distribution to }X+a\\
&\text{(ii) }X_nY_n \text{ converges in distribution to }aX
\end{aligned}
$$

Slutsky theorem은 분포 수렴하는 수열과 확률 수렴하는 수열에 대해서 각 항의 합이나 곱이 분포 수렴한다는 것을 보여준다. 합의 경우에 대한 증명은 아래와 같이 보일 수 있다.

$$
\begin{aligned}
&\text{pf (i)) The CDF of }X\text{ is }F_X\text{, and }F_X\text{ has at most countably many discontinuities.}\\
&\text{Let }C(F_X) = \{x\in\mathbb{R} : F_X\text{ is continuous at }x\}.\\
&\text{Let }F_{X+a}\text{ denote the CDF of }X+a\text{, so}\\
&\qquad F_{X+a}(x) = P(X+a\le x) = P(X\le x-a) = F_X(x-a).\\
&\text{Fix }x\in C(F_{X+a})\text{. Then, }x-a\in C(F_X)\text{.}\\
&\text{What we need to show is }\lim_{n\to\infty}F_{X_n+Y_n}(x) = F_{X+a}(x).\\
&\text{We show first that }\limsup_{n\to\infty}F_{X_n+Y_n}(x)\le F_{X+a}(x)\text{. For }\varepsilon>0,\\
&\qquad\begin{aligned}
F_{X_n+Y_n}(x) &= P(X_n+Y_n\le x)\\
&= P(X_n+Y_n\le x,\ X_n+a-\varepsilon>x) + P(X_n+Y_n\le x,\ X_n+a-\varepsilon\le x)\\
&\le P(\lvert Y_n-a\rvert>\varepsilon) + P(X_n+a-\varepsilon\le x)
\end{aligned}\\
&\text{(On the first event, }X_n+Y_n\le x<X_n+a-\varepsilon\text{, so }Y_n-a+\varepsilon<0\text{, i.e. }\lvert Y_n-a\rvert>\varepsilon\text{.)}\\
&\limsup_{n\to\infty}F_{X_n+Y_n}(x) \le \lim_{n\to\infty}P(\lvert Y_n-a\rvert>\varepsilon) + \lim_{n\to\infty}P(X_n\le x-a+\varepsilon)\\
&\qquad\qquad\qquad\qquad\ \ = 0 + F_X(x-a+\varepsilon)\\
&\text{Let }\varepsilon\to 0\text{ in such a way that }x-a+\varepsilon\in C(F_X)\text{.}\\
&\text{This is possible because }\mathbb{R}\setminus C(F_X)\text{ is countable.}\\
&\Rightarrow \limsup_{n\to\infty}F_{X_n+Y_n}(x) \le 0 + F_X(x-a) = F_{X+a}(x)\\
&\text{Similarly, we can show that }\liminf_{n\to\infty}F_{X_n+Y_n}(x)\ge F_X(x-a)\text{.}\\
&\Rightarrow \lim_{n\to\infty}F_{X_n+Y_n}(x) = F_X(x-a) = F_{X+a}(x)\text{, so }X_n+Y_n \xrightarrow{d} X+a.
\end{aligned}
$$

Slutsky theorem이 적용되는 예시로 다음과 같은 경우를 볼 수 있다. $$S_n^2$$가 sample variance이고, 이때 $$\frac{\bar{X}_n-\mu}{S_n/\sqrt{n}}$$의 limit distribution을 구하고 싶다고 하자. 이 값은 $$\frac{\bar{X}_n-\mu}{S_n/\sqrt{n}}=\frac{\bar{X}_n-\mu}{\sigma/\sqrt{n}}\cdot\frac{\sigma}{S_n}$$으로 나누어 쓸 수 있다. CLT에 의해 $$\frac{\bar{X}_n-\mu}{\sigma/\sqrt{n}}$$은 $$\mathcal{N}(0, 1)$$으로 분포 수렴하고, $$S_n^2$$이 $$\sigma^2$$로 확률 수렴하기 때문에 $$\frac{\sigma}{S_n}$$은 1로 확률 수렴한다. 따라서 Slutsky theorem의 (ii)에 의해 전체는 $$\mathcal{N}(0, 1)$$으로 분포 수렴하게 된다.
