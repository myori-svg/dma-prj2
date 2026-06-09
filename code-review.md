# DMA Project #2 — 코드 리뷰 (Team 22)

> 보고서 관련 수정·보완은 `report_review.md` 참조.
> 이 파일은 코드 레벨의 미완료/수정 제안 항목만 다룸.

---

## PART III — CL/clasification.py

### ⚠️ stop_words 미설정 (성능 감점 가능)

- **위치**: [CL/clasification.py:31-44](CL/clasification.py#L31-L44) (NB word), [CL/clasification.py:58-71](CL/clasification.py#L58-L71) (SVM word)
- **현재**: 두 모델 모두 `TfidfVectorizer`에 `stop_words` 파라미터 없음
- **채점 기준**: "없으면 성능 하락 가능"으로 명시된 항목 — hard fail은 아니나 accuracy 점수에 영향 가능
- **수정 제안**:
  ```python
  TfidfVectorizer(
      stop_words='english',   # ← 추가
      strip_accents='unicode',
      lowercase=True,
      ngram_range=(1, 2),
      ...
  )
  ```
- **주의**: NB는 char_wb TfidfVectorizer도 있으므로, stop_words는 word 벡터라이저에만 적용하면 됨 (char_wb에는 불필요)

---

## PART II — SE/QueryResult.py

### 🔵 불필요한 dead import (기능 문제 없음, 코드 품질)

- **위치**: [SE/QueryResult.py:3-4](SE/QueryResult.py#L3-L4)
- **현재**:
  ```python
  from whoosh import scoring       # 3번 줄 — 4번 줄에서 덮어씌워짐
  import CustomScoring as scoring  # 4번 줄 — 실제로 사용되는 모듈
  ```
- **문제**: 3번 줄은 4번 줄에 의해 완전히 덮어씌워지므로 dead code
- **수정 제안**: 3번 줄 삭제
  ```python
  import CustomScoring as scoring
  ```

---

## PART II — SE/CustomScoring.py

### 🔵 cf, dc 변수 미활용 (감점 없음, 개선 여지)

- **위치**: [SE/CustomScoring.py:130-157](SE/CustomScoring.py#L130-L157)
- **현재**: `intappscorer` 인자로 받지만 `cf`(컬렉션 빈도), `dc`(전체 문서 수)를 수식에 미사용
- **채점 기준**: "최소 2개 이상 활용" 조건은 충족(tf, idf, fl, avgfl 4개 사용)이므로 감점 없음
- **개선 여지**: JM smoothing 또는 언어 모델 방식을 추가하면 cf, dc를 자연스럽게 활용 가능
  ```python
  # 예: Jelinek-Mercer smoothing 추가
  lambda_ = 0.1
  p_w_C = cf / (dc * avgfl)  # 컬렉션 확률
  score = (1 - lambda_) * score + lambda_ * idf * log(1 + p_w_C)
  ```

---

## 우선순위 요약

| 우선순위 | 파일 | 항목 | 감점 가능성 |
|---------|------|------|-----------|
| 🔴 높음 | `CL/clasification.py` | stop_words 미설정 | accuracy 점수 영향 |
| 🟢 낮음 | `SE/QueryResult.py` | dead import 정리 | 없음 (코드 품질) |
| 🟢 낮음 | `SE/CustomScoring.py` | cf, dc 미활용 | 없음 (이미 충족) |
