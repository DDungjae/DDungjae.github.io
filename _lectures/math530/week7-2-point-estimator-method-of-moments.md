---
title: "Week 7-2: Point Estimator: Method of Moments"
date: 2026-10-22
course: "MATH530"
order: 7.02
excerpt: "값을 모르는 파라미터를 “잘” 찾는 방법"
---

## Point Estimator

Point estimator는 관측값 $$\underbar{X}$$를 통해 모집단의 알려지지 않은 파라미터 $$\theta$$를 추측하기 위해 사용하는 statistic $$T(\underbar{X})$$이다. $$T$$라는 함수는 무수히 많게 존재할 수 있기 때문에 estimator 또한 무수히 많게 존재할 수 있겠지만 파라미터를 “잘” 설명하는 estimator를 찾는 것이 당연히 중요할 것이다. 이를 위한 방법으로 method of moments, maximum likelihood estimator, Bayes estimator와 같은 방법이 있다.

## Method of Moments

Method of moments는 말 그대로 moment를 사용해서 파라미터를 추정하는 것이다. $$\theta=(\theta_1, \dots, \theta_k)$$와 같이 미지의 파라미터 $$k$$개가 존재한다고 해보자. 이때 $$k$$개의 파라미터를 찾기 위해 $$m_r=\mathcal{M}_r', r=1, \dots,k$$와 같이 $$k$$차 모멘트까지 $$k$$개의 식을 세우고 연립방정식을 풀어 $$\theta$$를 샘플 값들로 나타낼 수 있다. 여기서 $$m_r$$은 sample moment로 $$m_r=\frac{1}{n}\sum X_i^r$$로 나타나고 $$\mathcal{M}_r'$$는 $$\theta$$에 대한 함수로 나타나기 때문에 $$\theta$$를 관측한 샘플을 사용해 나타낼 수 있다.

이에 대한 예시로 $$X_i\sim\text{U}(0, \theta)$$를 생각해보자. 1차 모멘트를 계산하면 $$\bar{X}=\mathcal{M}_1'=E_\theta[X]=\frac{\theta}{2}$$이고 여기서 $$\hat{\theta}_\text{MoM}=2\bar{X}$$가 method of moments로 구한 point estimator가 된다.

다음으로 $$X_i\sim\text{Bin}(k, p)$$에서 method of moments를 적용해보자. 이때 파라미터는 $$k, p$$ 두 개이므로 2nd moment까지 구해 연립방정식을 풀어야 한다.

$$
\begin{aligned}
\bar{X}&=E[X]=kp \\
\frac{1}{n}\sum X_i^2&=E[X^2]=Var[X]+(E[X])^2=kp(1-p)+(kp)^2
\end{aligned}
$$

이를 정리하면 $$\hat{p}_\text{MoM}=\frac{\bar{X}}{\hat{k}_\text{MoM}}, \hat{k}_\text{MoM}=\frac{\bar{X}^2}{\bar{X}-\frac{1}{n}\sum(X_i-\bar{X})^2}$$를 method of moments로 얻을 수 있다.
