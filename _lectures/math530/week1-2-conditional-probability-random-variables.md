---
title: "Week1-2: Conditional Probability and Independence, Random Variables, Distribution Functions, Density and Mass Functions"
date: 2026-09-10
course: "MATH530"
excerpt: "확률변수, 밀도함수, 질량함수 등 확률에 대한 다양한 함수"
---

이전 시간에 확률을 수학적으로 엄밀하게 정의하기 위해 $$\sigma\text{-algebra}$$를 도입했고 probability function을 정의하는 the Axioms of Probability를 알아 보았다. 이번 시간에는 이러한 확률의 정의를 가지고 다양한 종류의 확률과 관련 함수를 알아 본다. 어디까지나 첫째 주에는 뭔가 이전에 한번씩 들어본 내용이 반복해서 등장하지만 복습하는 느낌으로 하나씩 보려고 한다.

## Conditional Probability & Independence

조건부 확률과 독립은 많이 들어보아서 익숙하다. 여기서 조건부 확률을 다시 한번 강조하는 이유는 베이즈 정리 때문이라고 생각한다.

- If $$A$$ and $$B$$ are events in $$S$$, and $$0 < P(B)$$, then the conditional probability of $$A$$ given $$B$$, denoted by $$P(A|B)$$, is $$P(A|B)=\frac{P(A\cap B)}{P(B)}$$
- Two events $$A$$ and $$B$$ are statistically independent if $$P(A\cap B)=P(A)P(B)$$

조건부 확률을 사용하는 이유로는 새로운 정보가 추가됨에 따라서 sample space를 다시 설정해야 할 수 있기 때문이다. 기존에 우리가 확률을 sample space $$S$$에서 생각했다면 conditional probability는 sample space를 $$B$$로 축소하는 것으로 볼 수 있다. 당연하게도 새로운 sample space $$B$$와 $$A$$가 disjoint 하다면 $$B$$라는 sample space에서는 $$A$$가 발생할 수 없으므로 conditional probability는 0이 된다. Conditional probability의 식을 변형하면 다음과 같은 형태가 유용하게 쓰이는 경우들도 종종 있다.

- $$P(A\cap B)=P(A|B)P(B)$$
- $$P(A|B)=P(B|A)\frac{P(A)}{P(B)}$$

두 번째 수식은 Bayes Rule의 기본 형태이다. 여기서 $$A$$가 여러 partition으로 구성된다고 했을 때 다음과 같이 확장이 가능하다.

$$
P(A_i|B)=\frac{P(B|A_i)P(A_i)}{\sum^\infty_{j=1}P(B|A_j)P(A_j)}
$$

사실, $$B$$라는 사건이 $$A$$라는 사건에 영향을 주지 않아 $$P(A|B)=P(A)$$일수 있다. 이러한 관계를 Bayes Rule에 적용한다고 했을 때 다음과 같이 변형된다.

$$
P(A|B)=P(A)=P(B|A)\frac{P(A)}{P(B)}=\frac{P(A\cap B)}{P(B)}
$$

즉, $$P(A)P(B)=P(A\cap B)$$이고 이는 $$A$$와 $$B$$가 statistically independent 하다는 정의이다. 쉬운 예시를 들자면 상자에서 공을 두 번 뽑을 때 두 번째 시행에서 첫번 째 시행에서 뽑은 공을 다시 넣었으면 독립이고 아니라면 독립이 아니다. 왜냐하면 후자의 경우 첫 번째 시행에서 뽑은 공의 종류에 따라 두 번째 시행에서의 공이 뽑힐 확률이 영향을 받기 때문이다. 두 개 이상의 사건에 대해서도 각 사건들이 mutually independent하다면 다음과 같은 관계가 성립한다.

$$
P\Big(\bigcap^k_{j=1}A_{i_j}\Big)=\prod^k_{j=1}P(A_{i_j})
$$

## Random Variables

확률 변수(random variable)은 sample space를 real numbers로 mapping하는 함수이다. 이때 sample space가 $$S=\set{s_1, s_2, \dots, s_n}$$이고 random variable의 range가 $$X=\set{x_1, x_2, \dots, x_m}$$일 때 서로 다른 $$s_i, s_j$$가 같은 $$x_k$$에 대응할 수도 있다. 그래서 우리는 random variable에서 probability를 다음과 같이 정의한다.

$$
P_X(X=x_i)=P(\set{s_j\in S:X(s_j)=x_i})
$$

즉 확률을 구하려는 range의 $$x_i$$로 대응되는 모든 domain에서의 $$s_j$$의 확률을 구하는 것이다.

## Distribution Functions

모든 random variable $$X$$에 대해서 우리는 $$X$$에 대한 cumulative distribution function을 정의할 수 있다.

$$
F_X(x)=P_X(X\le x)\text{, for all }x
$$

$$F_X(x)$$는 연속일수도 있고 step function과 같이 불연속일 수 있는데 연속이라면 random variable $$X$$는 연속이고, step function이라면 random variable $$X$$는 discrete하다.

또한 random variable $$X, Y$$는 모든 집합 $$A\in B^1$$에서 $$P(X \in A)=P(Y\in A)$$인 경우 identically distributed하다고 말한다. Identically distributed는 아래 표현과도 동치이다.

$$
F_X(x)=F_Y(x)\text{ for every }x
$$

## Density and Mass Functions

Probability mass function(PMF)은 discrete random variable에서, density function은 continuous random variable에서 정의된다.

- Probability mass function: $$f_X(x)=P(X=x)\text{ for all }x$$

Continuous random variable에서는 $$P(X=x)=0$$이기 때문에 바로 probability density function을 구하기는 어렵다. 그래서 주로 다음과 같이 point probability가 아니라 interval의 probability를 구한 후 fundamental theorem of calculus를 적용한다.

$$
P(X\le x)=F_X(x)=\int^x_{-\infty}f_X(t)\,dt
$$

$$
f_X(x)=\frac{d}{dx}F_X(x)
$$

Continuous random variable에서 $$P(X=x)=0$$ 이기 때문에 다음과 같은 성질도 만족한다.

$$
P(a<X<b)=P(a<X\le b)=P(a \le X<b)=P(a\le X\le b)
$$

$$\sigma\text{-algebra}$$로 확률을 정의하고, sample space를 real number로 대응시키는 random variable을 거쳐 cumulative distribution function으로 누적이나 구간의 확률을 구하는 방법을 알았고 probability density나 mass function으로 특정 포인트나, 구간의 확률을 구할 수 있다. CDF와 PDF의 관계도 fundamental theorem of calculus로 보일 수 있었다. 이제 우리는 다양한 확률 분포에서의 시행의 확률을 정의하고, 또 확률을 구할 수 있게 되었다.
