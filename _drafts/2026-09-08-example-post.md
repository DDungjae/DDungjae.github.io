---
# 글 예시. 이 파일을 _posts/ 로 복사하고 파일 이름의 날짜와 제목을 바꾸면 글이 하나 생깁니다.
# (_drafts/ 안의 파일은 사이트에 올라가지 않습니다)
title: "Example post: how to write here"
excerpt: "목록에 보이는 한 줄 요약. 없으면 본문 첫 문단이 쓰입니다."
categories:
  - paper-review # 큰 분류 (예: paper-review, notes, project-log)
tags:
  - llm # 작은 키워드
  - example
# header:
#   teaser: /assets/images/some-thumbnail.png # 목록에 보일 썸네일 (선택)
---

첫 문단은 요약(excerpt)으로도 쓰이니 짧고 명확하게.

## 소제목은 ## 부터

- 목록
- **굵게**, _기울임_, `코드`

수식은 `$$ ... $$` 로 감싸면 됩니다. $$ \mathcal{L} = -\sum_i y_i \log p_i $$

```python
def hello():
    print("code block")
```

이미지는 `assets/images/` 에 넣고 아래처럼 씁니다.

![설명](/assets/images/profile.jpg){: .align-center width="200"}
