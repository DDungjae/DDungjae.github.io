source "https://rubygems.org"

# ---------------------------------------------------------------------------
# 이 Gemfile 은 로컬 미리보기(preview.bat)에만 쓰입니다.
# GitHub Pages 는 이 파일을 읽지 않고 자체 빌드 환경(github-pages gem)으로 사이트를 만듭니다.
# github-pages gem 은 Ruby 4.x 를 지원하지 않아서, 로컬에서는 Jekyll 4 + 원격 테마로 대신합니다.
# ---------------------------------------------------------------------------
gem "jekyll", "~> 4.4"
gem "jekyll-remote-theme"
gem "webrick"

group :jekyll_plugins do
  gem "jekyll-paginate"
  gem "jekyll-sitemap"
  gem "jekyll-gist"
  gem "jekyll-feed"
  gem "jemoji"
  gem "jekyll-include-cache"
  gem "jekyll-redirect-from"
end

# 윈도우 전용: 시간대 데이터, 빠른 파일 변경 감지
platforms :windows do
  gem "tzinfo-data"
  gem "wdm", ">= 0.2"
end
