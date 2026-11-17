---
title: "Week 6-1: Data Reduction"
date: 2026-10-13
course: "MATH530"
order: 6.01
excerpt: "적은 자료로도 전체 정보를, 충분통계량과 관련 성질"
---

## Data Reduction

Data reduction은 전체 샘플을 사용하는 대신, 샘플로부터의 함수, 즉 statistic으로 샘플을 설명하고자 하는 방법이다. 예를 들어, 시험에 응시한 전체 학생의 점수를 하나하나 기억하는 것이 아니라 평균을 사용해서 해당 시험을 설명할 수 있다.

## Sufficient Statistic

주어진 통계량 $$T(\underbar{X})$$에 대한 $$X$$의 조건부 분포가 파라미터 $$\theta$$에 무관하게 나타나는 경우 $$T(\underbar{X})$$를 $$\theta$$에 대한 sufficient statistic이라고 부른다. $$T(\underbar{X})$$가 $$\theta$$에 대한 sufficient statistic인 경우 서로 다른 sample point에 대해서 $$T(\underbar{X})=T(\underbar{Y})$$라면 $$\theta$$에 대한 추론은 동일해야 한다. 즉 한번 $$T(X)$$를 알고 난 후에는 원본 데이터 $$X$$를 추가로 더 관찰해도 $$\theta$$에 대한 추가적인 정보는 알 수 없다. 이를 theorem으로 적으면 다음과 같이 나타낼 수 있다.

$$
\text{If }p_\theta(x)\text{ is the joint pdf or pmf of }X\text{ and }q_\theta(t) \text{ is the pdf, pmf of }T(X), \text{ then }T(X)\text{ is a sufficient statistic for }\theta, \text{ if for every }\underbar{x}\text{ in the sample space, }\frac{p_\theta(\underbar{x})}{q_\theta(T(\underbar{x}))}\text{ is constant as a function of }\theta.
$$

이에 대한 예시로 Poisson distribution을 생각해보자. $$X_1, \dots, X_n\sim\text{Pois}(\lambda)$$일 때 $$T(\underbar{X})=\sum^n_{i=1}X_i$$가 $$\lambda$$에 대한 sufficient statistic임을 보이자.

$$
f_\lambda(\underbar{x})=\prod^n_{i=1}f_\lambda(x_i)=\frac{e^{-n\lambda} \lambda^{\sum_{i=1}^nx_i}}{\prod^n_{i=1}(x_i!)}\\q_\lambda(t)=\frac{e^{-n\lambda}(n\lambda)^t}{t!}\\\frac{f_\lambda(\underbar{x})}{q_\lambda(t)}=\frac{e^{-n\lambda}\lambda^{\sum_{i=1}^nx_i}}{\prod^n_{i=1}(x_i!)}\frac{t!}{e^{-n\lambda}(n\lambda)^t}
$$

위의 조건에서 $$\frac{f_\lambda(\underbar{x})}{q_\lambda(t)}$$이 $$\lambda$$에 무관한 상수로 나타나려면 $$t=\sum_{i=1}^nx_i$$가 성립되어야 하고 즉 $$T(\underbar{X})=\sum_{i=1}^nX_i$$는 $$\lambda$$에 대한 sufficient statistic이 된다. $$x_1, \dots, x_n$$과 $$y_1, \dots, y_n$$에 대해서 각 $$x_i, y_i$$의 값을 정확히 알고 있지 않아도 $$\sum_{i=1}^nx_i$$와 $$\sum_{i=1}^ny_i$$의 값이 같다면 두 $$X, Y$$는 $$\lambda$$에 대한 정보를 동일하게 제공한다.

## Factorization Theorem

앞서 sufficient statistic을 판별하기 위해서는 $$T(\underbar{X})$$에 대한 $$\underbar{X}$$의 조건부 분포가 파라미터 $$\theta$$에 무관한지 확인해야 했다. 이를 보다 쉽게 수행하는 방법으로 Factorization theorem이 있다. Factorization theorem은 조건부 분포를 직접 계산하는 대신, $$\underbar{X}$$의 joint pdf 또는 pmf를 특정한 형태로 분해하여 충분성을 판별하는 방법이다. 핵심 아이디어는 파라미터 $$\theta$$에 의존하는 부분이 오직 통계량을 통해서만 표현될 수 있다면, 해당 통계량이 $$\theta$$에 관한 모든 정보를 포함하고 있다는 것이다. 즉, 원본 데이터 중 통계량으로 요약되지 않은 나머지 정보가 $$\theta$$와 무관하다면, 그 통계량은 sufficient statistic이 된다.

$$
\text{Let }f_\theta(x) \text{ denote the joint pdf or pmf of a sample }\underbar{x}. \text{ A statistic }T(\underbar{X})\text{ is sufficient for }\theta \text{ if and only if there exist non-negative functions }g_\theta(t)\text{ and }h(x)\text{ such that }f_\theta(\underbar{x})=g_\theta(T(\underbar{x}))h(\underbar{x})\text{ for all }x\in\mathcal{X} \text{ and }\theta \in \Theta.
$$

이에 대한 예시로 uniform distribution을 생각해보자. $$X_1, \dots, X_n\sim\text{Unif}(0, \theta)$$일 때 factorization theorem을 사용해 sufficient statistic을 찾아보자.

$$
f_\theta(\underbar{x})=\prod_{i=1}^n\frac{1}{\theta}\mathbf{I}_{(0\le x_i\le \theta)}=\frac{1}{\theta^n}\mathbf{I}_{(x_{(n)}\le \theta)}\mathbf{I}_{(x_{(1)}\ge 0)}\\g_\theta(T(\underbar{x}))=\frac{1}{\theta^n}\mathbf{I}_{(x_{(n)}\le \theta)}, h(\underbar{x})=\mathbf{I}_{(x_{(1)}\ge 0)}\text{ or }g_\theta(T(\underbar{x}))=\prod_{i=1}^n\frac{1}{\theta}\mathbf{I}_{(0\le x_i\le \theta)}, h(\underbar{x})=1
$$

위의 경우에서 $$T(\underbar{x})=X_{(n)}$$이나 $$T(\underbar{x})=(X_1, \dots, X_n)$$ 모두 factorization theorem을 통해 나타냈을 때 sufficient statistic임을 알 수 있다. 이를 통해 우리는 sufficient statistic이 유일하지 않음을 알 수 있다.
