---
title: "Week 7-1: Complete Statistic"
date: 2026-10-20
course: "MATH530"
order: 7.01
excerpt: "Ancillary Statistic, Complete Statistic"
---

## Ancillary Statistic

Ancillary statistic $$S(\underbar{X})$$는 분포가 파라미터 $$\theta$$에 의존하지 않는 통계량이다. 파라미터 $$\theta$$가 변해도 $$S(\underbar{X})$$의 분포는 변하지 않기 때문에 ancillary statistic 자체로는 파라미터에 대한 정보를 얻을 수 없다.

이에 대한 예시로 앞에서 다룬 $$X_i\sim\text{Unif}(\theta, \theta+1)$$를 이어서 살펴보자. Range statistic $$R=X_{(n)}-X_{(1)}$$의 분포를 구하기 위해 $$M=\frac{X_{(1)}+X_{(n)}}{2}$$을 사용해 변수 변환을 하면 $$\vert{}J\vert{}=1$$이 나오고, $$(X_{(1)}, X_{(n)})$$의 joint pdf로부터 다음을 얻는다.

$$
\begin{aligned}
f_\theta(r, m)&=n(n-1)r^{n-2}\,\mathbf{I}_{(0<r<1,\ \theta+\frac{r}{2}<m<\theta+1-\frac{r}{2})} \\
f_R(r)&=\int_{\theta+\frac{r}{2}}^{\theta+1-\frac{r}{2}} n(n-1)r^{n-2}\,dm=n(n-1)r^{n-2}(1-r),\quad 0<r<1
\end{aligned}
$$

$$R$$의 pdf에 $$\theta$$가 나타나지 않으므로 $$R$$은 ancillary statistic이 된다. 또한 $$R$$은 minimal sufficient statistic $$(X_{(1)}, X_{(n)})$$의 function이므로, minimal sufficient statistic 안에도 $$\theta$$와 무관한 정보가 섞여 있을 수 있음을 보여준다. 사실 이 결과는 $$\text{Unif}(\theta, \theta+1)$$가 location family라는 점에서 바로 얻을 수도 있다. $$f(z)=\mathbf{I}_{(0<z<1)}$$라 하면 $$f_\theta(x)=f(x-\theta)$$이므로, $$Z_i\sim\text{Unif}(0, 1)$$에 대해 $$X_i=\theta+Z_i$$로 쓸 수 있다.

$$
R=X_{(n)}-X_{(1)}=(\theta+Z_{(n)})-(\theta+Z_{(1)})=Z_{(n)}-Z_{(1)}
$$

$$Z_i$$의 분포는 $$\theta$$에 의존하지 않으므로 $$R$$의 분포도 $$\theta$$에 의존하지 않는다. 즉, $$\theta$$는 데이터 전체를 평행이동시킬 뿐이므로, location family에서는 range처럼 관측값들의 차이로만 이루어진 통계량이 ancillary statistic이 된다.

## Completeness

통계량 $$T(\underbar{X})$$에 대한 pdf (or pmf) $$f_\theta(t)$$가 있다고 할 때, 모든 함수 $$g$$에 대하여 $$\text{E}_\theta[g(T)]=0, \forall\theta\in\Theta$$일 때 항상 $$P_\theta(g(T)=0)=1, \forall\theta\in\Theta$$이면 $$T$$를 complete statistic이라고 한다. 즉, $$\theta$$가 어떤 값이든 기댓값이 0이 되는 $$T$$의 함수는 사실상 0밖에 없다는 것이다.

이에 대한 예시로 $$T\sim\text{Bin}(n, p),\ 0 < p < 1$$를 살펴보자. $$\text{E}_p[g(T)]=0, \forall p$$라고 하고 $$r=\frac{p}{1-p}$$로 두면 다음과 같다.

$$
\begin{aligned}
\text{E}_p[g(T)]&=\sum_{t=0}^n g(t)\binom{n}{t}p^t(1-p)^{n-t} \\
&=(1-p)^n\sum_{t=0}^n g(t)\binom{n}{t}r^t=0
\end{aligned}
$$

$$(1-p)^n>0$$이므로 $$\sum_{t=0}^n g(t)\binom{n}{t}r^t=0$$이 모든 $$r\in(0, \infty)$$에 대해 성립해야 한다. 이는 $$r$$에 대한 $$n$$차 다항식이 무수히 많은 점에서 0이라는 것이므로 모든 계수가 0이어야 하고, $$\binom{n}{t}>0$$이므로 $$g(t)=0,\ t=0, 1, \dots, n$$이 된다. 따라서 $$P_p(g(T)=0)=1$$이고 $$T$$는 complete statistic이 된다.

## Basu’s Theorem

Ancillary statistic과 complete statistic 사이에는 유의미한 관계가 있다. 파라미터 $$\theta$$에 대한 complete sufficient statistic $$T(\underbar{X})$$와 ancillary statistic $$S(\underbar{X})$$는 independent하며 이를 Basu’s theorem이라고 한다. Basu’s theorem은 두 통계량의 joint distribution이나 Jacobian을 구할 필요 없이 completeness와 ancillarity만 알면 쉽게 독립성을 판단할 수 있다는 의의를 지닌다.

추가적으로 다음과 같은 theorem도 있다. Minimal sufficient statistic이 존재한다면 모든 complete sufficient statistic은 minimal sufficient statistic이 된다. 또한 서로 다른 두 minimal sufficient statistic $$T, T'$$가 있으면 $$T'=\psi(T)$$를 만족하는 일대일 함수 $$\psi$$가 존재한다.

## Complete and Minimal Sufficient Statistics in Exponential Family

이전에 exponential family에 속하는 분포에 대해서는 sufficient statistic을 쉽게 구할 수 있음을 알았다. Exponential family에서는 complete minimal sufficient statistic 또한 바로 구할 수 있다.

$$
f_\theta(x)=h(x)c(\theta)\exp\left[\sum_{i=1}^k w_i(\theta)t_i(x)\right]
$$

위와 같은 exponential family에서 $$T(\underbar{X})=\left(\sum_{j=1}^n t_1(X_j), \dots, \sum_{j=1}^n t_k(X_j)\right)$$는 파라미터 $$\theta$$에 대한 sufficient statistic이 된다고 했었다. 여기에 조건이 하나만 더 추가되면 이 통계량은 complete sufficient statistic이 된다. 필요한 조건은 $$\{(w_1(\theta), \dots, w_k(\theta)) : \theta\in\Theta\}$$가 $$\mathbb{R}^k$$ 공간에서 open subset을 포함해야 한다는 점이다.

이에 대한 예시로 $$\text{Binomial}(n, \theta)$$를 생각해보자. $$f_\theta(x)=\binom{n}{x}\theta^x(1-\theta)^{n-x}=\binom{n}{x}(1-\theta)^n\exp[x\log{\frac{\theta}{1-\theta}}]$$와 같이 exponential family 형태로 작성 가능하다. 여기서 $$t_1(x)=x$$이고 $$\mathcal{H}=\{\log\frac{\theta}{1-\theta} : \theta \in (0, 1)\}=(-\infty, \infty)$$이므로 $$\mathbb{R}$$에서 open subset을 포함한다. 따라서 위의 theorem에 따라 $$T(\underbar{X})=\sum X_i$$는 complete sufficient statistic이 된다.
