---
# 강의 노트 첫 화면. 과목 목록은 _data/navigation.yml 의 lectures 항목에서 자동으로 가져옵니다.
# (과목마다 description 을 적어 두면 제목 아래 한 줄로 나옵니다)
title: "Lectures"
permalink: /lectures/
layout: archive
author_profile: false
sidebar:
  nav: "lectures"
---

<div class="entries-list">
{% for course in site.data.navigation.lectures[0].children %}
  <div class="list__item">
    <article class="archive__item">
      <h2 class="archive__item-title no_toc">
        <a href="{{ course.url | relative_url }}" rel="permalink">{{ course.title }}</a>
      </h2>
      {% if course.description %}
        <p class="archive__item-excerpt">{{ course.description }}</p>
      {% endif %}
    </article>
  </div>
{% endfor %}
</div>
