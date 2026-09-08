---
# MATH442 과목 페이지. _lectures/math442/ 에 노트 파일을 넣으면 아래에 자동으로 나열됩니다.
title: "MATH442 - Mathematics for AI"
permalink: /lectures/math442/
layout: single
author_profile: false
sidebar:
  nav: "lectures"
course: MATH442
---

2026 Fall

{% assign notes = site.lectures | where: "course", page.course | sort: "date" %}
{% if notes.size == 0 %}
아직 올라온 노트가 없습니다.
{% else %}
<div class="entries-list">
{% for post in notes %}
{% include archive-single.html type="list" %}
{% endfor %}
</div>
{% endif %}
