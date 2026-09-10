---
title: "Week1-1: Set Theory, Basics of Probability Theory"
date: 2026-09-10
course: "MATH530"
excerpt: "확률을 수학적으로 엄밀하게 정의하는 방법"
---

어느 과목을 수강해도 1주차 내용을 들으면 "할만한데?"라는 생각이 든다. 하지만, 1주차에 이런 생각이 든다고 해서 그 과목이 쉬웠던 적은 단 한번도 없다. 수리통계학 과목도 마찬가지다. 1주차에는 이전에 배웠던 내용이 다시 나오기는 했지만, 그만큼 중요하니까 다시 등장했을 것이다. 수리통계학 과목의 특징은 본격적으로 확률과 통계에 해석학 지식(measure theory, set theory 등)을 사용한다는 것이다. 확률과 통계를 수학적으로 보다 엄밀하게 정의하게 되었으며 학부 저학년 때 배우는 기초 확률 및 통계와는 이러한 점에서 차이가 있다.

## Set Theory

우선 간단한 용어 정의부터 하고 시작하자.

- An **experiment** is a procedure that can be infinitely repeated.
- The set, $$S$$, of all possible outcomes of a particular experiment is the **sample space** for the experiment.
- An **event** is any subset of the sample space $$S$$.

우리말로 하면 각각 시행, 표본 공간, 사건이다. 자주 등장하는 예시인 동전 던지기로 보다 자세한 내용을 들여다보자.

- Example: Tossing two coins, head and tail.

$$
S=\set{(HH), (HT), (TH), (TT)}
$$

$$
E_3=\set{(HH), (HT), (TH)}
$$

두 개의 동전을 던지는 시행에서 $$S$$라는 sample space에 대해서 $$E_3$$라는 event가 발생하려면 시행의 결과로 $$E_3$$의 원소인 $$(HH), (HT), (TH)$$가 나오면 된다. 수리통계학에서의 확률론은 해석학에서 다루었던 집합론을 기반으로 논리가 전개되기 때문에 집합 관련 notation을 앞으로 많이 사용하게 될 것이다.

- The event $$E\cup F$$ occurs if $$E$$ or $$F$$ occurs.
- The event $$E\cap F$$ occurs if both $$E$$ and $$F$$ occurs.
- Two events $$E$$ and $$F$$ are **disjoint (mutually exclusive)** if $$E\cap F=\emptyset$$
- The events $$E_1, E_2, \dots$$ are **pairwise disjoint (mutually exclusive)** if $$E_i\cap E_j=\emptyset$$ for all $$i\neq j$$

위와 같이 합집합, 교집합 기호를 통해서 어느 사건이 발생하거나 발생하지 않는 조건을 표기할 수 있으며 사건 간의 교집합이 없으면 mutually exclusive라고 부른다.

- If $$E_1, E_2, \dots$$ are pairwise disjoint and $$\bigcup^\infty_{i=1}E_i=S$$, then the collection $$E_1, E_2 \dots$$ forms a **partition** of $$S$$
- The **complement** of $$E$$, $$E^c$$, occurs if and only if $$E$$ does not occur. $$E^c$$ consists of all outcomes in $$S$$ that are not in $$E$$.

Pairwise disjoint한 사건들의 합집합이 표본 공간을 구성할 때 이 사건들은 표본 공간의 partition을 구성한다. 여집합(complement)의 개념도 역시 등장한다. 위의 내용들은 당연해 보이지만 하나하나 정의를 해 두어야 한다.

## Basics of Probability Theory

다음으로 확률이 드디어 등장한다. 일상적으로 확률은 어느 사건이 일어날 가능성을 수치화 하는 의미로 사용이 되지만 수학적 의미의 확률은 보다 엄밀한 정의를 요구한다. 우선, 확률을 정의하기 위해서 다음과 같은 개념을 도입해야 한다.

$$S$$라는 집합의 부분집합을 모아 놓은 것을 $$\sigma\text{-algebra (Borel field)}$$라고 부르고 이는 다음과 같은 성질을 만족한다.

- $$\emptyset \in B$$
- If $$A \in B$$, then $$A^c\in B$$ (closed under complement)
- If $$A_1, A_2, \dots \in B$$, then $$\bigcup^\infty_{i=1}A_i \in B$$ (closed under countable union)

$$B$$라는 집합은 공집합과, $$S$$ 자기 자신을 항상 포함하며 드모르간의 법칙에 따라서 다음 성질도 만족한다.

$$
\text{If } A_1, A_2, \dots \in B \text{ then } \bigcap^\infty_{i=1}A_i\in B
$$

이러한 $$\sigma\text{-algebra}$$의 성질을 만족하는 집합 $$S$$는 measurable하다. 하지만, 모든 부분집합이 measurable 하지는 않다. 그 반례로 Vitali set이 있다.

Countable set에서의 $$\sigma\text{-algebra}$$는 다음과 같이 표현할 수 있다.

$$
B=\set{\text{all subsets of } S \text{, including } S \text{ itself}}
$$

우리는 $$S$$가 $$n$$개의 원소를 가지고 있다면 그 부분집합의 개수를 $$2^n$$개라고 알고 있고 모든 부분집합의 집합이 $$\text{Borel Field}$$가 된다.

$$
S=\set{1, 2, 3}\text{ then }B=\set{\emptyset, \set{1}, \set{2}, \set{3}, \set{1, 2}, \set{2, 3}, \set{1, 3}, \set{1, 2, 3}}
$$

$$\sigma\text{-algebra}$$를 정의하게 되었으면 이제 probability function을 정의할 수 있다. 표본 공간 $$S$$와 이에 대한 $$\sigma\text{-algebra}$$인 $$B$$에 대해서 probability measure은 $$B$$를 정의역으로 가지는 probability function $$P$$이며 다음과 같은 성질을 만족해야 한다.

- $$0\le P(A)\text{ for all } A \in B$$
- $$P(S)=1$$
- $$\text{If }A_1, A_2, \dots \in B\text{ are pairwise disjoint, }P(\bigcup^\infty_{i=1}A_i)=\sum^\infty_{i=1}P(A_i)$$

위의 세 가지 성질은 the Axioms of Probability 또는 Kolmogorov Axioms라고 불리며 the Axioms of Probability를 만족하는 함수는 Probability function이라고 부른다.

직관적으로 봤을 때 확률이라는 것은 0에서 1 사이의 값을 가져야 하며, 표본 공간에 대한 확률은 1이 되어야 한다. 주사위를 던졌을 때 눈이 7보다 작은 수가 나올 확률이라고 한다면 1이 나오는게 당연하다. 세 번째 성질에서 pairwise disjoint하다는 것은 모든 $$A_i$$끼리 겹치는 부분이 하나도 없다는 것인데 이때 합집합에 대한 확률을 확률의 합으로 나타낼 수 있다고 말할 수 있다.

이렇게 확률이라는 것을 정의하기 위해서 집합론부터 시작해서 $$\sigma\text{-algebra}$$까지 도입해 probability function을 정의했다. 사실 countable additivity가 성립한다는 것을 증명하는 것을 해석학 수업을 들을 때 증명했었는데 꽤나 복잡했던 것으로 기억한다. 아무튼, 이후 통계로 넘어가기 전 확률을 수학적으로 정의해 기반을 다졌다고 생각하면 될 것 같다.
