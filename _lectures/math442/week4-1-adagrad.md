---
title: "Week 4-1: AdaGrad"
date: 2026-09-29
course: "MATH442"
excerpt: "AdaGrad가 GD 보다 좋은 성능을 보이는 조건"
tags: ["optimization", "machine-learning", "statistics"]
---

이전에 GD와 SGD를 비교하며 unbiased SGD에 대한 조건과 GD가 수렴을 최대화 하기 위한 learning rate $$\eta$$를 찾는 방법에 대해 다루었고, condition number에 대한 수렴 속도도 살펴보았다. GD와 SGD에서는 고정된 $$\eta$$를 사용했는데 step에 따라서 learning rate가 변하는 방법인 AdaGrad를 이번 시간에는 다룬다.

## AdaGrad

$$
\begin{aligned}
&\theta_{t+1}(i)=\theta_t(i)-\frac{\eta g_t(i)}{\sqrt{v_t(i)}+\epsilon}\text{ where }v_t(i)=\sum^t_{r=1}g_r(i)^2\\
&\theta_t=\begin{pmatrix}\theta_t(1)&\cdots &\theta_t(n)\end{pmatrix}^\top \in \mathbb{R}^n
\end{aligned}
$$

AdaGrad의 수식을 보면 coordinate by coordinate으로 파라미터를 업데이트 하는 것을 볼 수 있다. 또한 $$\eta$$는 고정이지만 분모에 이전까지의 gradient의 제곱의 합이 들어가기 때문에 step이 진행되면서 $$\frac{\eta}{\sqrt{v_t(i)}+\epsilon}$$은 감소하고 즉 learning rate이 감소한다. 이러한 learning rate의 감쇠는 파라미터마다 다른 스케일로 적용된다.

- Coordinate-wise update
- Use history of the gradients

위와 같은 AdaGrad의 특징은 항상 AdaGrad가 GD보다 좋은 성능을 가지게 할까? 항상 그렇다고 볼수는 없다. 예를 들어 다음과 같은 loss function을 생각해보자.

$$
\begin{aligned}
&L(\theta)=\frac{1}{2}\lVert\theta\rVert^2, \nabla L(\theta)=\theta\\
&\text{GD}: \theta_0=\begin{pmatrix}a_1 & \cdots &a_n\end{pmatrix}^\top, \theta_1=\theta_0-\eta\theta_0=\vec{0}\text{ when }\eta=1.
\end{aligned}
$$

GD는 위와 같은 형태에서 단 한 스텝만에 전역 최적해로 도달할 수 있지만 AdaGrad는 불가능하다. 그래서 항상 GD가 좋다, AdaGrad가 좋다라고 말할 수는 없고 loss function의 형태에 따라 적절한 최적화 방법을 선택해야 한다.

## When AdaGrad is better than GD?

그렇다면 AdaGrad가 GD보다 "좋은" 조건은 무엇일까? 특정 최적화 방법이 "좋다"라고 말하려면 수렴 속도가 빠르거나, 오차를 최소화 하거나 등으로 기준을 세워 생각해 볼 수 있다. 결론부터 말하면 AdaGrad가 GD보다 "좋을" 조건은 특정 조건을 만족하는 online learning에서 이다. Online learning이란 학습 과정에서 data point가 한번에 다 주어지는 것이 아니라 순차적으로 주어지며, 그때마다 파라미터를 업데이트하는 방식이다. Data point가 batch 단위로 들어온다고 할 때 batch가 추가될 때마다 파라미터를 다시 업데이트 하게 되며 이때 손실 함수는 시간 의존성을 가진다.

$$
L_t(\theta)=\sum^t_{i=1}\lVert y_i(\theta)-\hat{y}_i\rVert^2
$$

우리는 GD에서 최적화 방법론을 평가하기 위해 $$\lVert\theta_t-\theta^*\rVert_2$$가 얼마나 빠른 속도로 줄어드는지를 계산했지만 online learning에서는 regret이라는 개념을 도입한다.

$$
R_t=\sum^t_{j=1}(L_j(\theta_j)-L_j(\theta^*))
$$

Regret은 각 스텝에서 손실 함수가 최적 상태와 비교했을 때 얼마나 차이가 나는지를 누적해서 더하는 것이라고 말할 수 있다. Regret을 지표로 삼아 특정 조건에서 GD와 AdaGrad의 성능을 비교해보자.

- Assumption 1: Convexity
  - $$L_t(\theta_t)-L_t(\theta^*)\le g_t^\top(\theta_t-\theta^*)\text{ where }g_t=\nabla L_t(\theta_t)$$
- Assumption 2: Bounded domain
- Assumption 3: Bounded gradient
  - $$\lVert g_t\rVert_2\le G$$

수학적으로 bounded compact set에서는 다양한 성질을 사용할 수 있기 때문에 이러한 가정을 설정하는 것 같다. 최적화를 진행하면서 파라미터를 업데이트 하다가 domain을 벗어나는 영역으로 파라미터가 변경될 수 있는데 projection function을 사용해서 이를 후처리 한다.

$$
\begin{aligned}
&\Pi_K(\theta):=\operatorname*{argmin}_{x \in K}\lVert x-\theta \rVert_2\\
&\text{Property}:\lVert\Pi_K(\theta)-\theta^*\rVert_2\le\lVert\theta-\theta^*\rVert_2 \text{ for }\theta^*\in K
\end{aligned}
$$

Projection은 Domain 안의 값 중 업데이트된 파라미터와의 차이를 가장 작게 만드는 파라미터로 대신하는 것이라 생각할 수 있다. 만약 파라미터가 in domain이라면 $$x=\theta$$가 되고 파라미터가 out of domain이라면 차이를 최소화 하도록 $$x$$를 선택할 것이다. Projection function이 가지는 property는 이후 online learning에서 GD와 AdaGrad를 비교할 때 사용된다.

## Bound of Regret

### Online GD

$$
\begin{aligned}
&\text{Parameter update}:\theta_{t+1}=\Pi_K(\theta_t-\eta g_t)\\
&\lVert\theta_{t+1}-\theta^*\rVert^2_2=\lVert\theta^*-\Pi_K(\theta_t-\eta g_t)\rVert^2_2\le\lVert\theta_t-\eta g_t-\theta^*\rVert^2_2=\lVert\theta_t-\theta^*\rVert^2_2-2\eta g_t^\top(\theta_t-\theta^*)+\eta^2\lVert g_t\rVert^2_2\\
&\Rightarrow g_t^\top(\theta_t-\theta^*)\le \frac{\lVert \theta_t-\theta^*\rVert^2_2-\lVert \theta_{t+1}-\theta^*\rVert^2_2}{2\eta}+\frac{\eta}{2}\lVert g_t\rVert^2_2\\
&\Rightarrow R_t \le \sum^t_{j=1} g_j^\top(\theta_j-\theta^*)\le \sum^t_{j=1}\frac{\lVert \theta_j-\theta^*\rVert^2_2-\lVert \theta_{j+1}-\theta^*\rVert^2_2}{2\eta}+\frac{\eta}{2}\sum^t_{j=1}\lVert g_j\rVert^2_2\\
&\text{Using telescoping sum, }R_t \le \sum^t_{j=1} g_j^\top(\theta_j-\theta^*)\le \frac{\lVert \theta_1-\theta^*\rVert^2_2}{2\eta}+\frac{\eta}{2}\sum^t_{j=1}\lVert g_j\rVert^2_2\le\frac{D_2^2}{2\eta}+\frac{\eta}{2}tG^2\\
&\text{We can minimize the right hand side with }\eta^*=\frac{D_2}{G\sqrt{t}}\\
&\Rightarrow R_t\le D_2G\sqrt{t}
\end{aligned}
$$

위의 과정을 통해 우리는 regret의 상한을 구할 수 있다. 이때 gradient가 $$G$$로 bound되어 있기 때문에 $$R_t\le D_2G\sqrt{t}$$로 쓸 수도 있지만 보다 정밀한 bound를 구하기 위해 $$R_t\le D_2\sqrt{\sum^n_{i=1}\sum^t_{j=1}g^2_{j, i}}$$로 남겨두자. Notation을 설명하자면 $$D_2=\sup_{x, y \in K}\lVert x-y\rVert_2$$이고 $$\lVert \theta_1-\theta^*\rVert^2_2\le D_2^2$$로 bound할 때 사용한다. 하지만 $$D_2\le \sqrt{n}D_\infty=\sqrt{n}\sup_{x, y\in K}\lVert x-y\rVert_\infty$$으로 $$D_2$$를 bound할 수 있기 때문에 최종적으로 $$R_t\le D_\infty\sqrt{n}\sqrt{\sum^n_{i=1}\sum^t_{j=1}g^2_{j, i}}$$로 bound가 가능하다.

### Online AdaGrad

위와 유사한 계산 과정을 거치면 online learning에서 AdaGrad의 regret에 대한 bound는 다음과 같이 나타난다.

$$
R_t\le D_\infty \sum^n_{i=1}\sqrt{\sum^t_{j=1}g^2_{j, i}} \le D_\infty\sqrt{k}\sqrt{\sum^n_{i=1}\sum^t_{j=1}g^2_{j, i}}
$$

이때 $$D_\infty=\sup_{x, y\in K}\lVert x-y\rVert_\infty$$이고 $$k$$는 $$\sum^t_{j=1}g^2_{j, i}\neq0$$인 $$i$$의 수이다.

### Online GD vs Online AdaGrad

그렇다면 online learning에서 AdaGrad의 regret의 bound가 GD보다 작을 조건은 무엇일까? Regret의 바운드가 작다는 것은 AdaGrad가 GD보다 더 오차가 적게 모델을 최적화 할 수 있다는 것이고 즉 더 "좋은" 최적화 방법이 된다는 의미이다.

$$
\begin{aligned}
&\text{GD}:R_t\le D_\infty\sqrt{n}\sqrt{\sum^n_{i=1}\sum^t_{j=1}g^2_{j, i}} \\
&\text{AdaGrad}:R_t\le D_\infty\sqrt{k}\sqrt{\sum^n_{i=1}\sum^t_{j=1}g^2_{j, i}}
\end{aligned}
$$

두 bound는 같은 $$\sqrt{\sum^n_{i=1}\sum^t_{j=1}g^2_{j, i}}$$ 항에 각각 $$\sqrt{n}$$, $$\sqrt{k}$$ 가 곱해진 형태이므로, AdaGrad의 bound가 GD보다 작거나 같은 조건은 $$k\le n$$이고, **실질적으로 더 작아지는** 조건은 $$k<n$$이다. 여기서 $$k$$는 전체 $$t$$ 스텝 동안 적어도 한 번은 그래디언트가 0이 아니었던 차원 $$i$$의 개수, 즉 $$\sum^t_{j=1}g^2_{j, i}\neq0$$인 $$i$$의 개수이다. 반대로 어떤 차원 $$i$$의 그래디언트가 전체 스텝에서 항상 0이라면 그 차원은 $$k$$에 포함되지 않는다. $$k=n$$이라면(모든 차원이 한 번이라도 그래디언트를 가진다면) 두 bound가 같아 AdaGrad가 GD보다 나을 것이 없고, $$k$$가 $$n$$보다 작을수록, 즉 전체 학습 과정에서 단 한 번도 등장하지 않는 차원이 많을수록 AdaGrad의 bound가 더 작아진다. 이러한 조건을 sparse online learning이라고 부른다.

직관적으로 봤을 때 이는 자연어 처리에서 흔히 등장하는 상황과 맞닿아 있다. 예를 들어 단어 하나하나를 feature(파라미터의 한 차원)로 사용하는 모델을 생각해보자. "그리고", "을/를" 같이 거의 모든 문장에 등장하는 흔한 단어에 대응되는 차원은 거의 모든 스텝에서 그래디언트가 0이 아니므로(즉 $$n$$에 가까운 "dense"한 차원이므로) GD와 AdaGrad가 비슷하게 동작한다. 반면 특정 주제에서만 가끔 등장하는 희귀한 단어에 대응되는 차원은 전체 $$t$$ 스텝 중 극히 일부에서만 그래디언트가 0이 아니다. 이런 "sparse"한 차원이 많을수록 $$k$$가 $$n$$보다 훨씬 작아지고, AdaGrad의 이점이 커진다.

이러한 특성은 앞서 $$\theta_{t+1}(i)=\theta_t(i)-\frac{\eta g_t(i)}{\sqrt{v_t(i)}+\epsilon}$$와 같은 AdaGrad의 파라미터 업데이트 수식과도 바로 연결지어 이해할 수 있다.

- 그래디언트가 자주 발생하는(dense) 차원: $$v_t(i)$$가 빠르게 커져 학습률이 금방 감쇠한다.
- 그래디언트가 희소하게 발생하는(sparse) 차원: $$v_t(i)$$가 작게 유지되어 학습률이 크게 유지된다.

즉 AdaGrad는 "자주 등장해 이미 충분히 업데이트된" 차원은 천천히, "가끔 등장해 아직 덜 업데이트된" 차원은 빠르게 학습하도록 learning rate를 좌표별로 자동 조절하는 방법이라고 볼 수 있다. 모든 차원에 같은 고정된 learning rate를 적용하는 GD와 달리, 흔한 feature와 희귀한 feature가 섞여 있는 sparse online learning 환경에서 AdaGrad가 특히 유리한 이유가 바로 여기에 있다.
