# CLAUDE.md

## 역할

너는 서울대학교 "데이터관리와 분석(406.426B)" 수업의 Project #2 코드 채점 도우미야.
이 폴더에 있는 제출 코드를 읽고, `project2_grading_guide.md`의 체크 항목과 대조해서
PART별로 통과/실패/주의를 판정해줘.

---

## 프로젝트 구조

```
project2/
├── CLAUDE.md                      ← 지금 이 파일
├── project2_grading_guide.md      ← 채점 기준 (반드시 먼저 읽을 것)
├── AA/
│   ├── part1.py
│   ├── DMA_project2_team##_part1_horizontal.pkl
│   └── DMA_project2_team##_part1_association.pkl
├── SE/
│   ├── index/
│   ├── CustomScoring.py
│   └── QueryResult.py
└── CL/
    ├── clasification.py           ← 오타 아님, 의도된 파일명
    ├── DMA_project2_team##_nb.pkl
    └── DMA_project2_team##_svm.pkl
```

---

## 검토 순서

1. `project2_grading_guide.md` 읽기
2. 파일 구조 및 팀번호 확인
3. `AA/part1.py` 읽기 → PART I 채점
4. `SE/CustomScoring.py`, `SE/QueryResult.py` 읽기 → PART II 채점
5. `CL/clasification.py` 읽기 → PART III 채점
6. 전체 결과 요약 출력

---

## 출력 형식

각 PART마다 아래 형식으로 출력해줘:

```
## PART I — 연관 분석

### R1-1 Horizontal Table (10점)
- [✅ PASS] question id가 index로 설정됨
- [✅ PASS] tag 이름이 column명으로 설정됨
- [❌ FAIL] 셀 값이 count 형태 → boolean/0·1이어야 함
- [⚠️ 주의] to_pickle 대신 to_csv 사용 → 감점 가능

### R1-2 Association Rules (20점)
- [✅ PASS] apriori(min_support=0.005, use_colnames=True)
- [✅ PASS] association_rules(metric='lift', min_threshold=2.0)
- [⚠️ 주의] 보고서 확인 필요: lift 임계값 변경 분석 포함 여부
```

---

## 절대 우선 확인 항목

코드를 읽기 전에 반드시 먼저 확인:

1. **팀번호**: 코드 내 `TEAM = 0` 이 실제 팀 번호로 수정되었는가
2. **relevance.txt 참조**: `SE/` 폴더 내 파일에서 `relevance` 문자열이 있는가
   - 있으면 → **PART II 전체 0점** 즉시 경고

---

## 채점 불가 항목 (보고서 필요)

아래는 코드만으로 판단 불가 → "보고서 확인 필요"로 표시:

- R1-2 lift 임계값 변경 분석 (정량·정성 서술)
- PART II 성능 개선 보고서 (scoring 선택 근거)
- 전체 보고서 품질 (10점)
- 발표 (5점)

---

## 참고

- 이론 배경 및 슬라이드 출처는 `project2_grading_guide.md` 하단 표 참고
- 성능 점수(BPREF, accuracy)는 실행 없이 판단 불가 → 구조적 올바름만 확인
- NB 모델은 `MultinomialNB`가 텍스트 분류 표준 (Bernoulli, Gaussian도 허용)
- SVM은 `LinearSVC` 권장이나 `SVC`, `NuSVC`도 허용
