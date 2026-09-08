#!/usr/bin/env python3
"""
노션(Notion)에서 "Markdown & CSV" 로 내보낸 글을 이 사이트(Jekyll + Minimal Mistakes)의
글(_posts/) 또는 프로젝트(_projects/) 파일로 바꿉니다.

사용법
    python tools/notion_to_jekyll.py <노션 export .zip | .md 파일 | 폴더> [옵션]

옵션
    --type post|project     post(기본) 는 _posts/, project 는 _projects/ 에 만듭니다
    --date YYYY-MM-DD       글 날짜 (없으면 노션의 Date/Created 속성, 그것도 없으면 오늘)
    --slug SLUG             파일 이름과 주소에 쓸 이름 (없으면 제목에서 만듭니다)
    --categories a,b        카테고리 (없으면 노션의 Category 속성)
    --tags a,b              태그 (없으면 노션의 Tags 속성)
    --excerpt TEXT          목록에 보일 한 줄 요약 (없으면 첫 문단)
    --dry-run               파일을 만들지 않고 결과만 화면에 보여줍니다
    --force                 같은 이름의 파일이 있어도 덮어씁니다
    --site DIR              사이트 폴더 (기본: 이 스크립트가 있는 폴더의 상위)

노션 export 에서 처리하는 것
    - 첫 줄의 "# 제목" → front matter 의 title
    - 제목 바로 아래의 "Tags: ...", "Date: ...", "Category: ..." 같은 속성 줄 → front matter
    - 본문 제목 단계를 한 단계 내림 (# → ##). 페이지 제목이 이미 h1 이기 때문
    - 이미지·첨부파일을 assets/images/<slug>/ 로 복사하고 링크를 고침
    - 노션 인라인 수식 $x$ → kramdown 형식 $$x$$ (블록 수식 $$...$$ 은 그대로)
    - <aside>💡 ...</aside> 콜아웃 → 테마의 notice 상자
    - 코드 블록 언어 이름 정리 (Plain Text → text 등)
    - 본문에 {{ 나 {% 가 있으면 Jekyll 이 해석하지 않도록 {% raw %} 로 감쌈
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
import tempfile
import unicodedata
import urllib.parse
import zipfile
from pathlib import Path

# 노션 속성 줄에서 읽어들일 키 (소문자 비교)
KNOWN_PROPS = {
    "tags", "tag", "category", "categories", "date", "created", "created time",
    "published", "status", "excerpt", "summary", "description", "type", "slug",
    "lang", "language", "last edited time", "author", "url",
    "authors", "venue", "year", "paper", "journal", "conference",
}

NOTION_ID_RE = re.compile(r"\s+[0-9a-f]{32}$")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}

CODE_LANG_MAP = {
    "plain text": "text", "plaintext": "text", "c++": "cpp", "c#": "csharp",
    "objective-c": "objectivec", "shell": "bash", "sh": "bash", "javascript": "javascript",
    "typescript": "typescript", "json": "json", "yaml": "yaml", "markdown": "markdown",
    "latex": "latex", "python": "python", "r": "r", "sql": "sql", "java": "java",
    "kotlin": "kotlin", "go": "go", "rust": "rust", "html": "html", "css": "css",
    "docker": "dockerfile", "dockerfile": "dockerfile", "makefile": "makefile",
    "powershell": "powershell", "bash": "bash", "matlab": "matlab", "julia": "julia",
}


# 윈도우 콘솔(cp949)에서도 한글·이모지가 깨지지 않도록 출력을 UTF-8 로 고정
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# 입력 찾기
# ---------------------------------------------------------------------------
def locate_markdown(source: Path, workdir: Path) -> Path:
    """zip / 폴더 / .md 중 무엇이 오든 변환할 .md 파일 하나를 돌려줍니다."""
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            zf.extractall(workdir)
        root = workdir
    elif source.is_dir():
        root = source
    elif source.suffix.lower() == ".md":
        return source
    else:
        sys.exit(f"[X] 지원하지 않는 입력입니다: {source}")

    candidates = sorted(root.rglob("*.md"), key=lambda p: (len(p.relative_to(root).parts), str(p)))
    if not candidates:
        sys.exit(f"[X] .md 파일을 찾지 못했습니다: {root}")
    top_depth = len(candidates[0].relative_to(root).parts)
    top_level = [p for p in candidates if len(p.relative_to(root).parts) == top_depth]
    if len(top_level) > 1:
        listing = "\n".join(f"    {p}" for p in top_level)
        sys.exit(f"[X] 최상위 .md 파일이 여러 개입니다. 하나를 골라 .md 경로를 직접 넘겨주세요:\n{listing}")
    return top_level[0]


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------
def parse_notion(text: str, fallback_title: str) -> tuple[str, dict[str, str], str]:
    lines = text.lstrip("﻿").splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1

    title = fallback_title
    if i < len(lines) and lines[i].startswith("# "):
        title = lines[i][2:].strip()
        i += 1

    # 노션은 제목 다음에 빈 줄을 하나 두고 속성 줄("Tags: ...")을 늘어놓습니다.
    props: dict[str, str] = {}
    j = i
    while j < len(lines) and not lines[j].strip():
        j += 1
    while j < len(lines):
        m = _prop_line(lines[j])
        if not m:
            break
        props[m[0]] = m[1]
        j += 1
    if props:
        i = j

    body = "\n".join(lines[i:]).strip("\n") + "\n"
    return title, props, body


def _prop_line(line: str):
    m = re.match(r"^([A-Za-z][A-Za-z0-9 _/-]{0,40}):\s+(.*)$", line)
    if m and m.group(1).strip().lower() in KNOWN_PROPS:
        return m.group(1).strip().lower(), m.group(2).strip()
    return None


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    value = value.split("→")[0].split("->")[0].strip()
    value = re.sub(r"\s*\(.*?\)\s*$", "", value)  # 시간대 표기 제거
    for fmt in ("%B %d, %Y %I:%M %p", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d %H:%M",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%Y년 %m월 %d일"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", value)
    if m:
        return dt.date(int(m[1]), int(m[2]), int(m[3]))
    log(f"[!] 날짜를 해석하지 못해 오늘 날짜를 씁니다: {value!r}")
    return None


def split_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in re.split(r"[,;|]", value) if v.strip()]


def slugify(text: str) -> str:
    text = NOTION_ID_RE.sub("", text)
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s-]", "", text)      # 문자·숫자·밑줄·공백·하이픈만 남김 (한글 유지)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text or "untitled"


def clean_asset_name(name: str, used: set[str]) -> str:
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = NOTION_ID_RE.sub("", stem)
    stem = re.sub(r"[^\w-]+", "-", unicodedata.normalize("NFC", stem)).strip("-") or "file"
    candidate = f"{stem}{ext}"
    n = 2
    while candidate in used:
        candidate = f"{stem}-{n}{ext}"
        n += 1
    used.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# 본문 변환
# ---------------------------------------------------------------------------
def _looks_like_file(url: str) -> bool:
    path = urllib.parse.urlparse(url).path
    return Path(path).suffix.lower() in IMAGE_EXTS | {".pdf", ".zip", ".csv", ".txt", ".ipynb"}


def _download(url: str, warnings: list[str]) -> Path | None:
    """노션 API 가 주는 임시(서명된) 이미지 URL 을 내려받아 임시 파일 경로를 돌려줍니다."""
    import urllib.request
    name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name or "file"
    tmp = Path(tempfile.mkdtemp(prefix="notion-img-")) / name
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"이미지를 내려받지 못해 링크를 그대로 둡니다 (URL 만료?): {url[:80]}… ({e})")
        return None
    return tmp


def transform_body(body: str, md_path: Path, asset_dir: Path, asset_url: str,
                   dry_run: bool, download_images: bool = False) -> tuple[str, list[str], list[str]]:
    """본문을 고치고 (새 본문, 복사한 파일 목록, 경고 목록) 을 돌려줍니다."""
    warnings: list[str] = []
    copied: list[str] = []
    used_names: set[str] = set()

    def relocate(target: str) -> str | None:
        if target.startswith(("http://", "https://")):
            if not download_images or not _looks_like_file(target):
                return None  # 외부 링크는 그대로
            src = _download(target, warnings)
            if src is None:
                return None
        elif re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith(("#", "/")):
            return None  # mailto:, 앵커, 절대경로는 그대로
        else:
            rel = urllib.parse.unquote(target.split("#")[0])
            src = (md_path.parent / rel)
            if not src.is_file():
                warnings.append(f"파일을 찾지 못해 링크를 그대로 둡니다: {target}")
                return None
        new_name = clean_asset_name(src.name, used_names)
        if not dry_run:
            asset_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, asset_dir / new_name)
        copied.append(new_name)
        return f"{asset_url}/{new_name}"

    def fix_image(m: re.Match) -> str:
        alt, target = m.group(1), m.group(2)
        new = relocate(target)
        if new is None:
            return m.group(0)
        if alt.strip().lower() in ("", "untitled", "image"):
            alt = Path(urllib.parse.unquote(target)).stem
            alt = NOTION_ID_RE.sub("", alt)
        return f"![{alt}]({new})"

    def fix_link(m: re.Match) -> str:
        text, target = m.group(1), m.group(2)
        new = relocate(target)
        return m.group(0) if new is None else f"[{text}]({new})"

    def fix_math_inline(line: str) -> str:
        # $x$ → $$x$$  (이미 $$ 인 것, \$ 로 이스케이프한 것, 줄 넘어가는 것은 건드리지 않음)
        return re.sub(r"(?<![\$\\])\$(?!\$)([^\$\n]+?)(?<![\$\\])\$(?!\$)", r"$$\1$$", line)

    out: list[str] = []
    in_code = False
    # 본문에 h1(# ) 이 있을 때만 제목 단계를 한 단계 내림. (zip export 는 h1 부터, MCP fetch 는 h2 부터 나옴)
    has_heading = any(re.match(r"^# ", l) for l in body.splitlines())
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = re.match(r"^(\s*)(`{3,}|~{3,})\s*(.*)$", line)
        if fence:
            if not in_code:
                lang = fence.group(3).strip().lower()
                lang = CODE_LANG_MAP.get(lang, re.sub(r"[^a-z0-9+#-]", "", lang))
                out.append(f"{fence.group(1)}{fence.group(2)}{lang}")
                in_code = True
            else:
                out.append(f"{fence.group(1)}{fence.group(2)}")
                in_code = False
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        # <aside> 콜아웃 → notice 상자
        if line.strip() == "<aside>":
            j = i + 1
            inner: list[str] = []
            while j < len(lines) and lines[j].strip() != "</aside>":
                inner.append(lines[j].strip())
                j += 1
            content = " ".join(s for s in inner if s)
            out.append(content)
            out.append("{: .notice--info}")
            i = j + 1
            continue

        if has_heading:
            hm = re.match(r"^(#{1,6}) (.*)$", line)
            if hm:
                level = min(len(hm.group(1)) + 1, 6)
                line = f"{'#' * level} {hm.group(2)}"

        if "$$" not in line:
            line = fix_math_inline(line)
        line = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", fix_image, line)
        line = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)", fix_link, line)
        out.append(line)
        i += 1

    text = "\n".join(out).strip("\n") + "\n"
    if "{{" in text or "{%" in text:
        text = "{% raw %}\n" + text + "{% endraw %}\n"
        warnings.append("본문에 {{ 또는 {% 가 있어 {% raw %} 로 감쌌습니다 (Liquid 오해 방지)")
    return text, copied, warnings


def citation_line(props: dict[str, str]) -> str:
    """논문 리뷰용 한 줄 요약: "Jason Wei et al., NeurIPS 2022".
    Authors / Venue(또는 Journal, Conference) / Year 속성이 있을 때만 만듭니다."""
    authors = props.get("authors") or props.get("author") or ""
    venue = props.get("venue") or props.get("conference") or props.get("journal") or ""
    year = props.get("year") or ""
    if not authors and not venue:
        return ""
    names = [a.strip() for a in re.split(r"[,;]| and ", authors) if a.strip()]
    if len(names) > 3 and not names[0].lower().endswith("et al."):
        authors = f"{names[0]} et al."
    elif names:
        authors = ", ".join(names)
    year = re.sub(r"\.0$", "", year.strip())  # 노션 숫자 속성이 2022.0 으로 올 때
    right = " ".join(x for x in (venue.strip(), year) if x)
    return ", ".join(x for x in (authors, right) if x)


def first_paragraph(body: str, limit: int = 160) -> str:
    body = re.sub(r"^\{% (raw|endraw) %\}\s*$", "", body, flags=re.MULTILINE)
    for block in re.split(r"\n\s*\n", body):
        s = block.strip()
        if not s or s.startswith(("#", "!", "```", "$$", "{%", "|", "<", "{:")):
            continue
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
        s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
        s = re.sub(r"[*_`>#]+", "", s)
        s = re.sub(r"\$\$(.*?)\$\$", r"\1", s)  # 수식은 기호만 벗기고 내용은 남김
        s = " ".join(s.split())
        if s:
            return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"
    return ""


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def yaml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def yaml_list(items: list[str]) -> str:
    return "[" + ", ".join(yaml_str(x) for x in items) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--type", choices=["post", "project"], default="post")
    ap.add_argument("--date")
    ap.add_argument("--slug")
    ap.add_argument("--categories")
    ap.add_argument("--tags")
    ap.add_argument("--excerpt")
    ap.add_argument("--download-images", action="store_true",
                    help="본문의 http(s) 이미지·첨부를 내려받아 assets/ 에 저장 (노션 MCP 로 가져온 글용. URL 이 5분 만에 만료되니 바로 실행)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--site")
    args = ap.parse_args()

    site = Path(args.site).resolve() if args.site else Path(__file__).resolve().parent.parent
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        sys.exit(f"[X] 입력을 찾을 수 없습니다: {source}")

    with tempfile.TemporaryDirectory(prefix="notion-") as tmp:
        md_path = locate_markdown(source, Path(tmp))
        text = md_path.read_text(encoding="utf-8")
        fallback_title = NOTION_ID_RE.sub("", md_path.stem)
        title, props, body = parse_notion(text, fallback_title)

        date = parse_date(args.date) or parse_date(props.get("date") or props.get("published")
                                                   or props.get("created") or props.get("created time")) \
            or dt.date.today()
        slug = slugify(args.slug or props.get("slug") or title)
        categories = split_list(args.categories) or split_list(props.get("category") or props.get("categories"))
        tags = split_list(args.tags) or split_list(props.get("tags") or props.get("tag"))

        if args.type == "post":
            out_path = site / "_posts" / f"{date:%Y-%m-%d}-{slug}.md"
            url = f"/posts/{slug}/"
        else:
            out_path = site / "_projects" / f"{slug}.md"
            url = f"/projects/{slug}/"

        asset_dir = site / "assets" / "images" / slug
        asset_url = f"/assets/images/{slug}"
        new_body, copied, warnings = transform_body(body, md_path, asset_dir, asset_url, args.dry_run,
                                                    download_images=args.download_images)

        excerpt = args.excerpt or props.get("excerpt") or props.get("summary") or props.get("description") \
            or citation_line(props) or first_paragraph(new_body)
        teaser = next((f"{asset_url}/{f}" for f in copied if Path(f).suffix.lower() in IMAGE_EXTS), None)

        fm = ["---", f"title: {yaml_str(title)}"]
        if args.type == "post":
            fm.append(f"date: {date:%Y-%m-%d}")
        if excerpt:
            fm.append(f"excerpt: {yaml_str(excerpt)}")
        if categories:
            fm.append(f"categories: {yaml_list(categories)}")
        if tags:
            fm.append(f"tags: {yaml_list(tags)}")
        if teaser:
            fm += ["header:", f"  teaser: {teaser}"]
        fm.append("---")
        result = "\n".join(fm) + "\n\n" + new_body

        ignored = {k: v for k, v in props.items()
                   if k not in ("tags", "tag", "category", "categories", "date", "published", "created",
                                "created time", "excerpt", "summary", "description", "slug",
                                "authors", "author", "venue", "conference", "journal", "year")}

        log(f"[i] 원본      : {md_path.name}")
        log(f"[i] 제목      : {title}")
        log(f"[i] 종류/날짜 : {args.type} / {date:%Y-%m-%d}")
        log(f"[i] 출력 파일 : {out_path.relative_to(site)}")
        log(f"[i] 주소      : {url}")
        if copied:
            log(f"[i] 첨부 {len(copied)}개 → {asset_dir.relative_to(site)}/ : {', '.join(copied)}")
        if ignored:
            log(f"[i] 쓰지 않은 노션 속성: {ignored}")
        for w in warnings:
            log(f"[!] {w}")

        if args.dry_run:
            print(result)
            log("[i] --dry-run: 파일을 만들지 않았습니다.")
            return

        if out_path.exists() and not args.force:
            sys.exit(f"[X] 이미 있는 파일입니다 (덮어쓰려면 --force): {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result, encoding="utf-8", newline="\n")
        log(f"[OK] 만들었습니다: {out_path}")


if __name__ == "__main__":
    main()
