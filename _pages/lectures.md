---
# 강의 노트 첫 화면. 과목 목록은 _data/navigation.yml 의 lectures 항목에서 자동으로 가져옵니다.
title: "Lectures"
permalink: /lectures/
layout: single
author_profile: false
sidebar:
  nav: "lectures"
---

수업을 들으며 정리한 노트입니다. 과목을 고르세요.

{% for course in site.data.navigation.lectures[0].children %}
- [{{ course.title }}]({{ course.url | relative_url }})
{% endfor %}
