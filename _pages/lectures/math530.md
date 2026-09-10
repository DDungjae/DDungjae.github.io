---
# MATH530 과목 페이지. _lectures/math530/ 에 노트 파일을 넣으면 아래에 자동으로 나열됩니다.
title: "MATH530 - Mathematical Statistics"
permalink: /lectures/math530/
layout: single
author_profile: false
sidebar:
  nav: "lectures"
course: MATH530
---

2026 Fall · Textbook: Casella & Berger, *Statistical Inference* (2nd ed.)

{% assign notes = site.lectures | where: "course", page.course | sort: "date" | reverse %}
{% if notes.size == 0 %}
아직 올라온 노트가 없습니다.
{% else %}
<div class="entries-list">
{% for post in notes %}
{% include archive-single.html type="list" %}
{% endfor %}
</div>
{% endif %}
