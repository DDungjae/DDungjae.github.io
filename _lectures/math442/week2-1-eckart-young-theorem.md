---
title: "Week2-1: Eckart-Young Theorem"
date: 2026-09-15
course: "MATH442"
excerpt: "복잡한 행렬을 쉬운 행렬로 쪼개는 방법"
tags: ["linear-algebra"]
---

지난 시간에 행렬을 column 관점에서 바라보는 것과, SVD를 통해서 행렬을 여러개의 $$\text{rank}=1$$ 행렬로 분해하는 방법을 알아보았다. SVD의 의의라고 하면 행렬을 $$\text{rank}=1$$ 행렬의 합으로 나타낼 수 있으며 singular value의 크기에 따라서 원본 행렬을 구성하는 행렬들의 중요도를 판단할 수 있다는 것이다. 여기에서 더 나아가 이번 강의에서는 Eckart-Young Theorem에 대해 다루었다. 이 내용을 하나씩 살펴보도록 하자.

## Eckart-Young Theorem

$$
\begin{aligned}
&\text{For any } \mathbf{B} \text{ such that } \operatorname{rank}(\mathbf{B}) \le k, \\[4pt]
&\qquad \lVert \mathbf{A}-\mathbf{A}_k \rVert \le \lVert \mathbf{A}-\mathbf{B} \rVert,
\quad \text{where } \mathbf{A}_k = \sigma_1 \vec{u}_1 \vec{v}_1^\top + \dots + \sigma_k \vec{u}_k \vec{v}_k^\top. \\[6pt]
&\text{In particular } \lVert \mathbf{A}-\mathbf{A}_k \rVert_2 = \sigma_{k+1},
\qquad \lVert \mathbf{A}-\mathbf{A}_k \rVert_F = \sqrt{\sigma_{k+1}^2 + \dots + \sigma_r^2}.
\end{aligned}
$$

위 Theorem을 간단히 설명하자면 $$\mathbf{A}$$라는 행렬을 가장 잘 근사하는 행렬을 찾고 싶은데 이때 top-k singular value와 그에 대응하는 벡터를 사용해 $$\mathbf{A}$$를 근사하는 것이 그 방법이라고 말할 수 있다. 이전에 SVD를 통해서 singular value의 크기를 통해 SVD시에 생성되는 여러개의 $$\text{rank}=1$$ 행렬 중에서 무엇이 중요한지를 알 수 있다고 했는데 여기에서 그 구체적인 사용 사례를 볼 수 있다. 이 Theorem은 우리가 norm을 어떻게 정의하냐에 따라서 모두 성립한다. 다만 근사 오차의 값 자체는 norm에 따라 달라진다.

- $$\lVert \mathbf{M} \rVert = \max_{\lVert \vec{\mathbf{x}} \rVert_2 = 1} \lVert \mathbf{M}\vec{\mathbf{x}} \rVert$$ (spectral norm)
- $$\lVert \mathbf{M} \rVert_F = \sqrt{\sum \mathbf{M}_{ij}^2}$$ (Frobenius norm)

Eckart-Young Theorem을 사용하면 행렬의 근사나, projection에 대한 다음과 같은 특성도 나타남을 보일 수 있다. $$\mathbf{A}$$라는 행렬이 우선 다음과 같은 벡터들로 구성됨을 생각해보자.

$$
\mathbf{A} = \begin{bmatrix} - & \vec{a}_1^\top & - \\ & \vdots & \\ - & \vec{a}_m^\top & - \end{bmatrix}
$$

위의 $$\mathbf{A}$$라는 행렬을 구성하는 벡터들을 어느 평면에 projection할 때 우리는 원본 벡터와, projection 된 벡터의 크기 차이가 작기를 원한다. 이러한 projection은 다음과 같이 나타난다.

$$
\begin{aligned}
&\sum^m_{i=1} \lVert \vec{a}_i^\top - \vec{a}_i'^\top \rVert^2 \text{ is minimized} \\[4pt]
&\quad \text{if } \vec{a}_i'^\top \text{ is the orthogonal projection of } \vec{a}_i^\top \text{ onto the row space of } \mathbf{A}_k, \\[4pt]
&\quad \text{among all rank } k \text{ hyperplanes.}
\end{aligned}
$$

Eckart-Young Theorem을 증명하면 위와 같은 성질도 자동으로 유도가 된다. 직관적으로 봤을 때 위의 성질은 $$\mathbf{A}$$의 행벡터와, 이를 어느 $$\text{rank}=k$$인 hyperplane에 투영할 때 원본 벡터와 투영된 벡터의 거리가 최소가 되도록 하는 hyperplane을 SVD로 얻을 수 있다는 것이다.

$$
\begin{aligned}
&\text{proof) Let } \mathbf{P}_k = \mathbf{V}_k \mathbf{V}_k^\top, \quad
\mathbf{V}_k = \begin{bmatrix} | & & | \\ \vec{v}_1 & \cdots & \vec{v}_k \\ | & & | \end{bmatrix}. \\[8pt]
&\text{Then } \mathbf{A}\mathbf{P}_k = \begin{bmatrix} - & \vec{a}_1'^\top & - \\ & \vdots & \\ - & \vec{a}_m'^\top & - \end{bmatrix},
\text{ where } \vec{a}_i'^\top \text{ is the orthogonal projection of } \vec{a}_i^\top \text{ onto span}(\vec{v}_1, \dots, \vec{v}_k). \\[8pt]
&\text{Note that } \lVert \mathbf{A} - \mathbf{A}\mathbf{P}_k \rVert_F^2
\text{ is the sum of squared distances between } \vec{a}_i^\top \text{ and } \vec{a}_i'^\top. \\[8pt]
&\text{By the way, } \mathbf{A}\mathbf{P}_k = \mathbf{A}\mathbf{V}_k \mathbf{V}_k^\top
= \Big( \sum^r_{i=1} \sigma_i \vec{u}_i \vec{v}_i^\top \Big)
\begin{bmatrix} | & & | \\ \vec{v}_1 & \cdots & \vec{v}_k \\ | & & | \end{bmatrix}
\begin{bmatrix} - & \vec{v}_1^\top & - \\ & \vdots & \\ - & \vec{v}_k^\top & - \end{bmatrix} \\[6pt]
&\qquad\qquad\ \ = \begin{bmatrix} | & & | \\ \sigma_1 \vec{u}_1 & \cdots & \sigma_k \vec{u}_k \\ | & & | \end{bmatrix}
\begin{bmatrix} - & \vec{v}_1^\top & - \\ & \vdots & \\ - & \vec{v}_k^\top & - \end{bmatrix}
= \mathbf{A}_k. \\[8pt]
&\text{Since } \operatorname{rank}(\mathbf{A}\mathbf{P}) \le k \text{ for any rank } k \text{ projection } \mathbf{P},
\text{ Eckart-Young Theorem says} \\[4pt]
&\qquad \lVert \mathbf{A} - \mathbf{A}_k \rVert_F \le \lVert \mathbf{A} - \mathbf{A}\mathbf{P} \rVert_F .
\end{aligned}
$$

증명 과정은 위와 같다. 우선 우리는 어느 hyperplane에 각 벡터를 투영할 때 원본 벡터와, 투영된 벡터의 차이가 가장 작게 하기 위해서는 orthogonal projection을 해야 함을 안다. 하지만 이는 고정된 hyperplane에 대해서이고, 어느 hyperplane을 선택해야 하는지는 아직 정해지지 않았다. 우리는 이 hyperplane을 Top-k singular value와 그에 대응하는 벡터로 구성하겠다는 것이다.

$$
\begin{aligned}
&\text{Suppose that } \mathbf{A} \text{ is centered, i.e. } (1, \dots, 1)\,\mathbf{A} = (0, \dots, 0). \\[8pt]
&\text{The variance captured by a rank } k \text{ projection, }
\frac{1}{m-1} \lVert \mathbf{A}\mathbf{P} \rVert_F^2,
\text{ is maximized at } \mathbf{P} = \mathbf{P}_k = \mathbf{V}_k \mathbf{V}_k^\top. \\[8pt]
&\text{Note that } \lVert \mathbf{A} \rVert_F^2
= \lVert \mathbf{A}\mathbf{P} \rVert_F^2 + \lVert \mathbf{A} - \mathbf{A}\mathbf{P} \rVert_F^2,
\text{ and } \lVert \mathbf{A} - \mathbf{A}\mathbf{P} \rVert_F^2 \text{ is minimized by } \mathbf{P}_k.
\end{aligned}
$$

원본 행렬의 $$\lVert\mathbf{A}\rVert_F^2$$는 상수 이므로 원본 벡터와 투영된 벡터의 차이인 $$\lVert\mathbf{A}-\mathbf{A}\mathbf{P}\rVert^2_F$$는 투영된 벡터의 크기인 $$\lVert\mathbf{A}\mathbf{P}\rVert^2_F$$가 최대화될 때 최소가 된다. 즉 투영된 벡터의 크기를 최대화하고, 원본 벡터와의 차이가 가장 작도록 하는 $$\mathbf{P}=\mathbf{V}_k\mathbf{V}_k^\top$$가 된다.
