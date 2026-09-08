---
name: notion-import
description: 노션(Notion)에서 내보낸 마크다운(zip 또는 .md)을 이 사이트의 글(_posts) 또는 프로젝트(_projects)로 변환해 올린다. 사용자가 "노션 글 올려줘", "이 zip 변환해줘", "포스트로 만들어줘" 라고 하면 사용.
---

# 노션 글 가져오기

변환 자체는 `tools/notion_to_jekyll.py` 가 합니다. 이 스킬은 그 스크립트를 어떤 순서로 쓰고
무엇을 확인해야 하는지 정리한 것입니다. 사이트는 Jekyll + Minimal Mistakes (remote theme) 입니다.

## 0. 노션 MCP 가 연결되어 있으면 (zip 없이 바로)

`mcp__notion__notion-fetch` 도구가 있으면 zip 을 받을 필요가 없습니다. `$ARGUMENTS` 가 노션 URL 이거나
사용자가 페이지 제목을 말하면 이 길로 갑니다.

1. URL 이 없으면 `mcp__notion__notion-search` 로 제목을 찾아 **어느 페이지인지 사용자에게 보여줍니다**.
2. `mcp__notion__notion-fetch` 로 본문을 받습니다. `<content>` 안이 노션식 마크다운입니다.
3. 짧은 경로의 임시 폴더(예: `%TEMP%\notion-<slug>\`)에 `.md` 파일 하나를 만듭니다. 형식은 zip export 와 같게:
   - 첫 줄 `# <제목>` (노션 제목의 오탈자는 고치되 사용자에게 알림)
   - 빈 줄, 그 다음 속성 줄. fetch 결과 `<properties>` JSON 을 이렇게 옮깁니다 ("Paper Reviews" 데이터베이스 기준):
     `Authors` → `Authors: ...`, `Venue` → `Venue: ...`, `Year` → `Year: 2022`, `Tags` 배열 → `Tags: a, b`,
     `Category` → `Category: ...`, `date:Date:start` → `Date: 2026-09-08`. `Paper` 링크는 본문 첫 줄에
     `[url](url)` 로 넣되 본문에 이미 같은 링크가 있으면 생략. 데이터베이스 밖의 일반 페이지는 속성이 title 뿐이니
     내용을 보고 Tags/Category 를 제안하고 Authors/Venue/Year 는 논문을 확인해 채웁니다.
   - 빈 줄, 그 다음 본문. `<empty-block/>` 같은 태그는 지우고, `![](https://prod-files-secure...)` 이미지의 빈 alt 는 짧은 설명으로 채웁니다.
4. 이미지 URL 은 **5분 뒤 만료**됩니다. fetch 직후 바로 변환합니다:
   ```bash
   python tools/notion_to_jekyll.py "<임시 .md>" --type post --download-images --slug <slug> [--excerpt "..."]
   ```
   만료 경고가 나오면 fetch 를 다시 해서 새 URL 로 반복합니다.
5. 이후는 아래 5번(빌드 확인)부터 동일합니다.

**한 줄 요약(excerpt) 규칙**: 논문 리뷰는 서지 정보 한 줄로 통일합니다. 형식은 `Jason Wei et al., NeurIPS 2022`
(제1저자 et al., 저자 2~3명이면 전부; 학회는 약칭, 저널은 정식 이름, arXiv 만 있으면 `arXiv 2022`).
노션 "Paper Reviews" 데이터베이스의 `Authors`, `Venue`, `Year` 속성을 `.md` 의 속성 줄로 옮기면 스크립트가
이 형식을 자동으로 만듭니다. 속성이 비어 있으면 논문을 확인해 `--excerpt "Jason Wei et al., NeurIPS 2022"` 로 넣고,
**내용을 요약한 문장은 쓰지 않습니다** (사용자 결정).

**노션 쪽 구조** (모두 "Github Posts" 페이지 아래):
- `Paper Reviews` 데이터베이스 (collection 9ea357cf-0a50-4c10-9a65-ba5e881dab9a): 논문 리뷰 한 줄 = 글 하나 → `--type post`
- `Lectures` 데이터베이스 (collection 9473503b-8caf-48fb-9848-26c7171401cf): 한 줄 = 과목 (예: MATH530 페이지).
  과목 페이지 안에 `<과목> Notes` 데이터베이스가 있고 (MATH530 Notes: collection 4da3c0d6-e1dd-4360-b12d-c91300bcc0ce),
  그 한 줄 = 노트 하나 → `--type lecture --course <과목코드>`. 과목 코드는 fetch 결과의 `Course` 속성 (예: MATH530) 을 씁니다.
  과목 페이지 제목은 "MATH530 - Mathematical Statistics" 처럼 긴 이름이니 거기서 가져올 때는 대시 앞의 코드만 씁니다.
  현재 과목: MATH530 (Notes DB collection 4da3c0d6-e1dd-4360-b12d-c91300bcc0ce), MATH442.
  `Week` 속성은 front matter 에 넣지 않습니다 (제목에 이미 들어감).
- 새 과목이 생기면: 노션에 과목 페이지 + Notes DB 를 같은 형태로 만들고, 사이트에는 `_pages/lectures/<과목>.md` 와 navigation.yml 항목을 추가합니다.

## 1. 입력 찾기 (zip 으로 받은 경우)

- 사용자가 경로를 주면 그것을 씁니다. `$ARGUMENTS` 에 경로가 들어올 수 있습니다.
- 경로가 없으면 다운로드 폴더에서 가장 최근 노션 export 를 찾아 **어느 파일인지 사용자에게 보여주고** 씁니다.
  ```bash
  ls -t ~/Downloads/Export-*.zip 2>/dev/null | head -3
  ```
- zip 을 풀 필요는 없습니다. 스크립트가 zip, 폴더, .md 모두 받습니다.

## 2. 글인지 프로젝트인지 정하기

- 논문 리뷰·공부 기록·일지 → `--type post` (기본값). `_posts/YYYY-MM-DD-slug.md`, 주소 `/posts/slug/`
- 결과물 소개(대회, 과제, 사이드 프로젝트) → `--type project`. `_projects/slug.md`, 주소 `/projects/slug/`
- 수업 노트 → `--type lecture --course MATH530`. `_lectures/math530/slug.md`, 주소 `/lectures/math530/slug/`.
  과목 페이지(`_pages/lectures/<과목>.md`)와 `_data/navigation.yml` 의 lectures 목차에 그 과목이 없으면 둘 다 추가합니다
  (MATH530 페이지를 복사해 title/permalink/course 만 바꾸면 됨).
- 사용자의 말에서 분명하지 않으면 제목과 내용을 보고 정하되, 결과를 알릴 때 어느 쪽으로 넣었는지 말합니다.

## 3. 먼저 dry-run

```bash
python tools/notion_to_jekyll.py "<경로>" --type post --dry-run
```

stderr 에 제목·날짜·출력 파일·첨부 목록·경고가, stdout 에 완성된 파일 내용이 나옵니다. 확인할 것:

- **title**: 노션 페이지 제목 그대로. 이상하면 사용자에게 묻지 말고 노션 제목을 유지.
- **date**: 노션의 Date/Created 속성 → 없으면 오늘. 사용자가 날짜를 말했으면 `--date`.
- **slug**: 제목에서 자동 생성. 한글 제목이면 한글 주소가 됩니다. 영어 slug 가 낫다고 판단되면
  `--slug` 로 짧은 영어 이름을 지정 (예: `--slug attention-is-all-you-need`).
- **categories / tags**: 노션 속성이 없으면 비어 있습니다. 내용을 보고 제안하되 지어내지 말고,
  기존 글에서 쓰인 것이 있으면 맞춥니다 (`grep -h "^categories:\|^tags:" _posts/*.md | sort | uniq -c`).
  관례: categories 는 한 개 (`paper-review`, `notes`, `project-log` 등), tags 는 여러 개.
- **excerpt**: 첫 문단이 자동으로 들어갑니다. 요약으로 어색하면 `--excerpt` 로 한 줄 써 줍니다.
- **경고**: "파일을 찾지 못해" 가 나오면 노션 export 에 이미지가 빠진 것. zip 을 다시 받아야 합니다
  ("Include content: Everything" 으로 내보내기).

## 4. 실제 변환

```bash
python tools/notion_to_jekyll.py "<경로>" --type post [--slug ...] [--categories ...] [--tags ...] [--excerpt "..."]
```

이미지는 `assets/images/<slug>/` 로 복사되고, 첫 이미지가 `header.teaser` 로 들어갑니다.
같은 파일이 이미 있으면 멈춥니다. 같은 글을 다시 올리는 것이 맞으면 `--force`.

## 5. 빌드 확인

```bash
export PATH="/c/Ruby40-x64/bin:$PATH" SSL_CERT_FILE="/c/Program Files/Git/mingw64/etc/ssl/certs/ca-bundle.crt"
bundle exec jekyll build 2>&1 | grep -iE "error|warning|done in"
```

- 처음이면 `bundle install` 이 먼저 필요합니다 (`preview.bat` 이 같은 일을 합니다).
- 오류 없이 `done in` 이 나오면 `_site/posts/<slug>/index.html` 이 생겼는지 확인합니다.
- 수식(`$$ ... $$`)은 MathJax 가 브라우저에서 그리므로 빌드 로그로는 알 수 없습니다.
  `\(`, `\[` 가 HTML 에 들어갔는지만 봅니다: `grep -c '\\\\(' _site/posts/<slug>/index.html`

## 6. 결과 알리기 · 푸시

- 만든 파일 경로, 사이트 주소(`https://ddungjae.github.io/posts/<slug>/`), 복사한 이미지 수,
  경고가 있었으면 그 내용을 사용자에게 알립니다.
- 커밋·푸시는 **사용자가 올려달라고 했을 때만** 합니다. 커밋 메시지는 `post: <제목>` 또는 `project: <제목>`.
  푸시 뒤 GitHub Pages 빌드(약 1분)를 기다렸다가 실제 주소가 200 인지 확인합니다.

## 노션 export 의 특징 (스크립트가 처리하는 것)

- 첫 줄 `# 제목`, 그 아래 `Tags: a, b` / `Date: September 8, 2026` 같은 속성 줄
- 이미지 경로가 `페이지이름%20<32자리id>/Untitled.png` 처럼 URL 인코딩되어 있음
- 본문 제목이 `#` 부터 시작 → 페이지 제목과 겹치므로 한 단계 내림
- 인라인 수식이 `$x$` → kramdown 은 `$$x$$` 만 인라인으로 인식하므로 변환
- 콜아웃이 `<aside>💡 ...</aside>` → `{: .notice--info}` 상자
- 코드 블록 언어가 `Plain Text` 처럼 나옴 → `text` 등으로 정리
- 본문에 `{{`/`{%` 가 있으면 Liquid 가 해석하므로 `{% raw %}` 로 감쌈

스크립트가 못 다루는 것이 보이면 (예: 데이터베이스 표, 임베드, 토글 안의 이미지) 변환된 파일을
직접 고치고, 반복될 것 같으면 `tools/notion_to_jekyll.py` 에 처리를 추가합니다.
