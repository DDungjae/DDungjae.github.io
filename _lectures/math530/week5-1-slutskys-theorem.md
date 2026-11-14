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
