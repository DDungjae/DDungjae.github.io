# DDungjae.github.io

서재원의 개인 사이트. [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) 테마를
GitHub Pages 에서 그대로 빌드합니다. 주소: **https://DDungjae.github.io**

`main` 브랜치에 푸시하면 GitHub 이 1~2분 안에 사이트를 다시 만듭니다. 별도의 빌드 설정은 없습니다.

---

## 어디를 고치면 되나

| 하고 싶은 것 | 고칠 파일 |
| :--- | :--- |
| 소개글, CV (첫 화면) | `index.md` |
| 글 쓰기 | `_posts/YYYY-MM-DD-제목.md` (예시: `_drafts/2026-09-08-example-post.md` 복사) |
| 프로젝트 추가 | `_projects/이름.md` (아래 예시) |
| 이름·한 줄 소개·이메일·링크 (왼쪽 사이드바) | `_config.yml` 의 `author:` |
| 프로필 사진 | `assets/images/profile.jpg` 를 교체 |
| 상단 메뉴 | `_data/navigation.yml` |
| 색 테마 | `_config.yml` 의 `minimal_mistakes_skin` |

`_config.yml` 을 고쳤을 때만 로컬 미리보기 서버를 껐다 켜야 합니다. 나머지는 저장하면 바로 반영됩니다.

## 글 쓰기

`_drafts/2026-09-08-example-post.md` 를 `_posts/` 로 복사한 뒤 파일 이름의 날짜와 제목을 바꿉니다.
파일 이름은 반드시 `YYYY-MM-DD-제목.md` 형식이어야 합니다. 글 주소는 `/posts/제목/` 이 됩니다.

```markdown
---
title: "글 제목"
excerpt: "목록에 보이는 한 줄 요약"
categories: [paper-review]
tags: [llm, vision]
---

본문은 마크다운. 수식은 $$ ... $$, 코드는 ``` 로 감쌉니다.
```

`_drafts/` 안의 파일은 사이트에 올라가지 않으니 초안을 두기에 좋습니다.

## 프로젝트 추가

`_projects/wind-forecast.md` 처럼 파일을 하나 만들면 `/projects/` 페이지에 카드가 생깁니다.

```markdown
---
title: "Wind Power Forecasting (BARAM 2026)"
excerpt: "카드에 보이는 한 줄 설명"
header:
  teaser: /assets/images/wind-teaser.png # 카드 썸네일 (선택)
---

프로젝트 설명. 마크다운 그대로.
```

## 로컬 미리보기

`preview.bat` 을 더블클릭하면 http://localhost:4000 에서 확인할 수 있습니다.
(처음 한 번은 gem 을 받느라 몇 분 걸립니다.) 터미널에서는:

```bat
bundle install
bundle exec jekyll serve --livereload
```

## 참고

- 테마 설정 전체: https://mmistakes.github.io/minimal-mistakes/docs/configuration/
- 글 front matter 옵션 (목차, 썸네일, 갤러리 등): https://mmistakes.github.io/minimal-mistakes/docs/layouts/
