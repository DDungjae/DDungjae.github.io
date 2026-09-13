---
title: "Week1-1: Column View of Matrix Multiplication"
date: 2026-09-10
course: "MATH442"
excerpt: "행렬곱을 열 단위로 바라보는 새로운 시각"
tags: ["linear-algebra"]
---

이번주는 수강신청을 늦게 해서 수업 첫날은 못갔다. 물어보니까 수업 첫 시간에는 진도를 나가기 보다는 강의 개요에 대한 설명을 했다고 하고 본격적인 수업은 시작하지 않았다고 한다. 다행히 수강신청에 성공해서 두 번째 수업부터는 들을 수 있었고 첫주의 내용은 행렬 연산에 대한 새로운 관점이 주를 이루었다. 이전에 응용선형대수 과목을 들으며 행렬 연산, SVD, PCA와 같은 방법은 다 배웠었는데 복습의 느낌이 강했지만 새로운 관점으로 행렬을 바라보는게 몇몇 있었다.

## Column-view of matrix multiplication

$$
\mathbf{V}=\begin{bmatrix}| &  & |\\v_1 & \cdots &v_n \\ |& & |\end{bmatrix}, \mathbf{A}=\begin{bmatrix}a_{11}&a_{12}&\cdots &a_{1m}\\a_{21} & \ddots &&\vdots \\ \vdots&&\ddots&\vdots\\a_{n1}&\cdots&\cdots& a_{nm} \end{bmatrix}
$$

$$
\mathbf{V}\mathbf{A}=\begin{bmatrix}| &  & |\\v_1 & \cdots &v_n \\ |& & |\end{bmatrix}\begin{bmatrix}a_{11}&a_{12}&\cdots &a_{1m}\\a_{21} & \ddots &&\vdots \\ \vdots&&\ddots&\vdots\\a_{n1}&\cdots&\cdots& a_{nm} \end{bmatrix}=\begin{bmatrix}| & | & &|\\\sum^n_{i=1}a_{i1}v_i& \sum^n_{i=1}a_{i2}v_i& \cdots &\sum^n_{i=1}a_{im}v_i \\ |&| & & |\end{bmatrix}
$$

일반적으로 처음에 행렬곱을 배울 때에는 $$p \times n$$와 $$n \times m$$행렬을 곱하면 $$p \times m$$ 행렬이 나오고 이때 $$i$$행과 $$j$$열을 곱한 값이 행렬곱 결과에서의 $$a_{ij}$$의 값이 된다고 배운다. 하지만 위의 결과와 같이 행렬곱을 column view의 형태로 볼 수 있다. 이를 보면 연산 결과로 나오는 행렬의 각 열이 $$\mathbf{V}$$의 열벡터의 선형 결합으로 이루어져 있다. 이로부터 우리는 다음과 같은 성질이 있음을 알 수 있다.

$$
\text{Col}(\mathbf{VA})\subseteq\text{Col}(\mathbf{V})
$$

$$
\text{rank}(\mathbf{VA})\le\text{rank}(\mathbf{V})
$$

이는 새롭게 만들어지는 행렬의 열벡터들이 $$\mathbf{V}$$의 열벡터들과 column space보다 축소된 공간을 구성하는 것으로 해석할 수 있다.

## Real symmetric matrices

- If $$\mathbf{A} \in \mathbb{R}^{n \times n}$$ satisfies $$\mathbf{A}^{\top}=\mathbf{A}$$, then $$\mathbf{A}$$ is symmetric
- Theorem: Every real symmetric matrix $$\mathbf{S}$$ has a form $$\mathbf{S}=\mathbf{Q}\Lambda\mathbf{Q^\top}$$

우리는 대칭행렬 $$\mathbf{A}$$를 위와 같이 대각행렬 $$\Lambda$$와 $$\mathbf{Q}$$로 표현할 수 있다. 이때 $$\mathbf{Q}^\top$$은 orthonormal matrix이다. Orthonormal matrix는 다음과 같이 성질을 가진다고 배웠었다.

$$
q_i^\top q_j=
\begin{cases}
1,&i=j,\\
0,&i\neq j
\end{cases}, {Q^\top Q=I_n}
$$

## SVD (Singular Value Decomposition)

$$
\begin{aligned}
&\text{Let } \mathbf{A} \in \mathbb{R}^{m \times n} \text{ with } r = \operatorname{rank}(\mathbf{A}). \text{ Then } \mathbf{A} \text{ can be written as} \\[6pt]
&\qquad \mathbf{A} = \mathbf{U} \Sigma \mathbf{V}^\top
= \begin{bmatrix} | & & | \\ u_1 & \cdots & u_m \\ | & & | \end{bmatrix}
\begin{bmatrix} \sigma_1 & & & \\ & \ddots & & \\ & & \sigma_r & \\ & & & 0 \end{bmatrix}
\begin{bmatrix} \text{---} & v_1^\top & \text{---} \\ & \vdots & \\ \text{---} & v_n^\top & \text{---} \end{bmatrix}
= \sum_{i=1}^{r} \sigma_i \, u_i v_i^\top
= \sigma_1 u_1 v_1^\top + \sigma_2 u_2 v_2^\top + \cdots + \sigma_r u_r v_r^\top, \\[6pt]
&\text{where } \mathbf{U} \in \mathbb{R}^{m \times m},\quad \mathbf{V} \in \mathbb{R}^{n \times n},\quad \Sigma \in \mathbb{R}^{m \times n}, \\[4pt]
&\mathbf{U}^\top \mathbf{U} = \mathbf{U} \mathbf{U}^\top = \mathbf{I}_m, \qquad \mathbf{V}^\top \mathbf{V} = \mathbf{V} \mathbf{V}^\top = \mathbf{I}_n, \\[4pt]
&\Sigma_{ij} = \begin{cases} \sigma_i, & i = j \le r, \\ 0, & \text{otherwise}, \end{cases} \qquad \sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0. \\[8pt]
&\sigma_1, \ldots, \sigma_r : \text{ singular values of } \mathbf{A} \\
&u_1, \ldots, u_m \ (\text{columns of } \mathbf{U}) : \text{ left singular vectors} \\
&v_1, \ldots, v_n \ (\text{columns of } \mathbf{V}) : \text{ right singular vectors}
\end{aligned}
$$

위의 공식은 SVD(특이값 분해)를 나타낸 것이다. 이때 $$\mathbf{U}$$와 $$\mathbf{V}$$는 orthonormal matrix이고 $$\Sigma$$는 diagonal matrix이다. 이때 대각성분은 0보다 크며 내림차순으로 정렬되어 있다.

$$
\sum_{i=1}^{r} \sigma_i \, u_i v_i^\top
= \sigma_1 u_1 v_1^\top + \sigma_2 u_2 v_2^\top + \cdots + \sigma_r u_r v_r^\top
$$

특이값 분해 후 위와 같이 벡터를 표현한다고 할 때 우리는 다음과 같은 효과를 얻을 수 있다.

- 행렬을 $$\text{rank}=1$$인 행렬의 합으로 표현할 수 있다. → 원본 행렬의 $$\text{rank}$$를 쉽게 파악할 수 있다.
- Singular value가 큰 행렬이 원본 행렬에서 높은 비중을 차지할 것이고 이를 통해 행렬에 포함된 정보를 중요도 순으로 파악할 수 있다.
- Column space, row space, null space를 파악할 수 있다.

$$
\operatorname{Col}(\mathbf{A})
=
\operatorname{span}\{u_1,\ldots,u_r\},\quad
\operatorname{Row}(\mathbf{A})
=
\operatorname{span}\{v_1,\ldots,v_r\},\quad
\operatorname{Null}(\mathbf{A})
=
\operatorname{span}\{v_{r+1},\ldots,v_n\}
$$

이처럼 행렬을 SVD로 표현하는 것은 단순히 행렬 그 자체를 보는 것보다 많은 정보를 제공한다. 그렇다면, 행렬에 대한 SVD를 쉽게 수행할 수 있어야 한다. 우리는 SVD를 위해서 다음과 같은 motivation을 생각해볼 수 있다. 이는 $$\mathbf{U}$$와 $$\mathbf{V}$$가 orthonormal matrix라는 점에서 착안한다.

$$
\text{If }\mathbf{A=U\Sigma V^\top \Rightarrow A^\top A=V\Sigma^\top U^\top U\Sigma V^\top=V\Sigma^\top \Sigma V^\top}
$$

이때 $$v_i$$를 $$\mathbf{A^\top A}$$의 orthonormal eigenvectors라고 하자. 그렇다면 모든 $$i$$에 대해

$$
\mathbf{A^\top A }\vec{v_i} =\lambda_i\vec{v_i}
$$

위의 수식이 성립한다. $$\mathbf{A^\top A}$$가 positive semi-definite matrix이기 때문에 $$\lambda_i$$는 non-negative 하다. 이후부터는 다음과 같은 순서로 SVD 유도가 가능하다.

$$
\begin{aligned}
&\text{Note 1) Positive semi-definite} \\[4pt]
&\text{A symmetric matrix } \mathbf{S} \in \mathbb{R}^{n \times n} \text{ is positive semi-definite (PSD) if} \\
&\qquad x^\top \mathbf{S} x \ge 0 \quad \forall x \in \mathbb{R}^n. \\[4pt]
&\text{Example: } x^\top (\mathbf{A}^\top \mathbf{A}) x = \|\mathbf{A} x\|^2 \ge 0, \text{ so } \mathbf{A}^\top \mathbf{A} \text{ is PSD for any } \mathbf{A}.
\end{aligned}
$$

$$
\begin{aligned}
&\text{Note 2) Eigenvalues of a PSD matrix are non-negative} \\[4pt]
&\mathbf{S} q = \lambda q,\ q \neq 0 \ \Longrightarrow\ q^\top \mathbf{S} q = \lambda \|q\|^2 \ge 0 \ \Longrightarrow\ \lambda \ge 0. \\[4pt]
&\text{Hence } \mathbf{A}^\top \mathbf{A} v_i = \lambda_i v_i \text{ with } \lambda_i \ge 0, \text{ and we can define } \sigma_i := \sqrt{\lambda_i}.
\end{aligned}
$$

$$
\begin{aligned}
&\text{Note 3) Constructing the SVD} \\[4pt]
&\text{1. } \mathbf{A}^\top \mathbf{A} = \mathbf{V} \Lambda \mathbf{V}^\top, \quad \lambda_1 \ge \cdots \ge \lambda_r > 0 = \lambda_{r+1} = \cdots = \lambda_n, \quad \sigma_i := \sqrt{\lambda_i} \\[4pt]
&\text{2. Define } u_i := \frac{\mathbf{A} v_i}{\sigma_i} \quad (i = 1, \ldots, r) \\[4pt]
&\text{3. } u_i^\top u_j = \frac{v_i^\top \mathbf{A}^\top \mathbf{A} v_j}{\sigma_i \sigma_j} = \frac{\lambda_j}{\sigma_i \sigma_j} \, v_i^\top v_j = \delta_{ij} \quad \Longrightarrow\ u_1, \ldots, u_r \text{ are orthonormal} \\[4pt]
&\text{4. For } i > r: \ \|\mathbf{A} v_i\|^2 = \lambda_i = 0 \Rightarrow \mathbf{A} v_i = 0. \text{ Extend } u_1, \ldots, u_r \text{ to an orthonormal basis } u_1, \ldots, u_m \text{ of } \mathbb{R}^m. \\[4pt]
&\text{5. } \mathbf{A} \mathbf{V} = \begin{bmatrix} \sigma_1 u_1 & \cdots & \sigma_r u_r & 0 & \cdots & 0 \end{bmatrix} = \mathbf{U} \Sigma \ \Longrightarrow\ \mathbf{A} = \mathbf{U} \Sigma \mathbf{V}^\top
\end{aligned}
$$
