---
title: "Week4-2: Convergence Concepts and CLT"
date: 2026-10-01
course: "MATH530"
excerpt: "확률 분포의 수렴과 중심극한정리"
tags: ["statistics"]
---

지난 시간에 convergence in probability, convergence in distribution을 다루었다. 추가적으로 converges almost surely나 $$L^p$$ convergence와 같은 convergence concepts도 더 알아보도록 하자.

## Converges Almost Surely

$$
\begin{aligned}
&\text{A sequence of random variables }X_1, X_2, \cdots \text{ converges almost surely to a random variable }X\\
&\text{ if }\text{P}(\lim_{n\rightarrow \infty}X_n=X)=1
\end{aligned}
$$

Almost sure convergence는 convergence in probability보다 강한 성질을 가져 converges almost surely는 convergence in probability를 보장한다.

$$
\begin{aligned}
&\text{pf) For any }\epsilon\gt0\text{, define } A_n=\{\lvert X_n-X\rvert \gt\epsilon\}. \\
&0\le\text{P}(\lvert X_n-X\rvert\gt\epsilon)=\text{P}(A_n)\le\text{P}(\cup^\infty_{k=n}A_k)=\text{P}(\lvert X_n-X\rvert\gt\epsilon \text{ infinitely often})=0
\end{aligned}
$$

## $$L^p$$ Convergence

$$
\begin{aligned}
&\text{A sequence of random variables }X_1, X_2, \cdots \text{ converges in }L^p \text{ to a random variable }X\\
&\text{if }\lim_{n\rightarrow \infty}\text{E}(\lvert X_n-X\rvert^p)=0.
\end{aligned}
$$

$$L^p$$ convergence는 converges almost surely와 같은 수준의 강한 수렴성을 보장해 $$L^p$$ convergence이면 converges in probability가 성립한다. 하지만 $$L^p$$ convergence와 converges almost surely는 서로 수렴성을 보장하지 않아 두 수렴 성질은 의미가 살짝 다르다고 볼 수 있다.

## Central Limit Theorem

$$
\begin{aligned}
&\text{Let }X_1, X_2, \cdots \text{be a sequence of i.i.d random variables whose MGFs exist in a neighborhood of 0.}\\
&\text{Let }\text{E}[X_i]=\mu \text{ and } \text{Var}[X_i]=\sigma^2. \\
&\text{Let }\bar{X}_n=\frac{1}{n}\sum^n_{i=1}X_i\text{ be the sample mean. }\\
&\text{Then as }n\rightarrow\infty, \frac{\bar{X}_n-\mu}{\frac{\sigma}{\sqrt{n}}}\text{ converges in distribution to }N(0, 1).
\end{aligned}
$$

Central limit theorem은 우리가 $$X_i$$가 어떤 분포를 따르는지 모르고 단순히 기댓값과 분산만을 알고 있을 때 표본 평균의 분포가 정규분포로 수렴한다는 강력한 정보를 제공한다. 모집단이 어느 분포를 따르는지에 관계 없이 이렇게 표본 평균의 분포에 대한 정보를 알 수 있다는 사실은 이후 신뢰구간 추정이나 가설검정에서 핵심적으로 작용한다. 이에 대한 증명은 Taylor's theorem을 MGF에 적용해서 얻을 수 있다.

$$
\begin{aligned}
&\text{pf) Because the MGFs exist, both }M\text{ and }\sigma^2\text{ are finite. Define }Y_i = \frac{X_i-\mu}{\sigma}\text{, so }\sum Y_i = \frac{n(\bar{X}_n-\mu)}{\sigma} \Rightarrow \frac{\bar{X}_n-\mu}{\sigma/\sqrt{n}} = \frac{1}{\sqrt{n}}\sum Y_i.\\
&M_{\frac{\bar{X}_n-\mu}{\sigma/\sqrt{n}}}(t) = M_{\frac{1}{\sqrt{n}}\sum Y_i}(t) = E\left[\exp\left(\frac{t}{\sqrt{n}}\sum Y_i\right)\right] = M_{\sum Y_i}\left(\frac{t}{\sqrt{n}}\right) = \left\{M_Y\left(\frac{t}{\sqrt{n}}\right)\right\}^n\\
&\text{What we need to show is }\left\{M_Y\left(\frac{t}{\sqrt{n}}\right)\right\}^n \to e^{t^2/2}\text{ (the MGF of }N(0,1)\text{).}\\
&\text{Now, we expand }M_Y\left(\frac{t}{\sqrt{n}}\right)\text{ in a Taylor series around 0.}\\
&M_Y\left(\frac{t}{\sqrt{n}}\right) = \sum_{k=0}^{\infty} M_Y^{(k)}(0)\frac{(t/\sqrt{n})^k}{k!}, \quad \text{where } M_Y^{(k)}(0) = \frac{d^k}{dt^k}M_Y(t)\Big\vert_{t=0}\\
&\text{Next, we use Taylor's Theorem.}\\
&M_Y\left(\frac{t}{\sqrt{n}}\right) = T_2\left(\frac{t}{\sqrt{n}}\right) + R_2\left(\frac{t}{\sqrt{n}}\right) = \sum_{k=0}^{2} M_Y^{(k)}(0)\frac{(t/\sqrt{n})^k}{k!} + R_2\left(\frac{t}{\sqrt{n}}\right)\\
&M_Y\left(\frac{t}{\sqrt{n}}\right) = 1 + \frac{(t/\sqrt{n})^2}{2!} + R_2\left(\frac{t}{\sqrt{n}}\right), \quad R_2\left(\frac{t}{\sqrt{n}}\right) = O\left((t/\sqrt{n})^2\right)\\
&\lim_{n\to\infty}\left(M_Y\left(\frac{t}{\sqrt{n}}\right)\right)^n = \lim_{n\to\infty}\left[1+\frac{(t/\sqrt{n})^2}{2}+R_2\left(\frac{t}{\sqrt{n}}\right)\right]^n = \lim_{n\to\infty}\left[1+\frac{1}{n}\left(\frac{t^2}{2}+nR_2\left(\frac{t}{\sqrt{n}}\right)\right)\right]^n = \lim_{n\to\infty}\left[1+\frac{1}{n}\left(\frac{t^2}{2}+O(1)\right)\right]^n = \exp\left(\frac{t^2}{2}\right)\\
&\text{Therefore, the MGF of }\frac{\bar{X}_n-\mu}{\sigma/\sqrt{n}}\text{ converges to }\exp(t^2/2)\text{, the MGF of }N(0,1)\text{, so }\frac{\bar{X}_n-\mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0,1).
\end{aligned}
$$
