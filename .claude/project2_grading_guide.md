# Project #2 채점 가이드 (Claude Code용)

> 이 문서는 Claude Code가 로컬 제출 코드를 읽고 채점 기준과 대조할 때 사용하는 참고 문서입니다.
> 강의 슬라이드(Ch.8~15), TA04/05/06 자료를 기반으로 작성되었습니다.

---

## 📋 채점 개요

| PART | 항목 | 배점 |
|------|------|------|
| PART I | R1-1 Horizontal Table | 10점 |
| PART I | R1-2 Association Rules + 보고서 | 20점 |
| PART II | 검색 성능 (BPREF) | 30점 |
| PART II | 성능 개선 보고서 | 5점 |
| PART III | Naïve Bayes accuracy | 10점 |
| PART III | SVM accuracy | 10점 |
| 공통 | 보고서 품질 | 10점 |
| 공통 | 발표 | 5점 |

---

## 🔍 검토 순서

1. 파일/폴더 구조 확인
2. PART I (AA 폴더)
3. PART II (SE 폴더)
4. PART III (CL 폴더)

---

## ① 파일·폴더 구조 체크

```
DMA_project2_team##.zip
├── AA/
│   ├── part1.py
│   ├── DMA_project2_team##_part1_horizontal.pkl
│   └── DMA_project2_team##_part1_association.pkl
├── SE/
│   ├── index/               (인덱스 폴더)
│   ├── make_index.py        (신규 인덱스 생성 시에만)
│   ├── CustomScoring.py
│   └── QueryResult.py
├── CL/
│   ├── clasification.py     (오타 주의: classification 아님)
│   ├── DMA_project2_team##_nb.pkl
│   └── DMA_project2_team##_svm.pkl
├── DMA_project2_team##_보고서.pdf
└── DMA_project2_team##_발표자료.pdf
```

### 체크 항목
- [x] 팀 번호가 `TEAM = 0`이 아닌 실제 팀 번호로 수정되어 있는가 (미수정 시 감점)
- [x] 폴더명이 AA / SE / CL로 정확한가
- [x] pkl 파일명에 팀 번호가 정확히 들어가 있는가

---

## ② PART I — 연관 분석 (AA/part1.py)

### R1-1: Horizontal Table (10점)

#### 필수 구현 사항

**데이터 로드**
- `DMA_project_UBR.csv`를 pandas로 로드
- 컬럼 구조: `question`, `tag` 두 개 컬럼

**DataFrame 변환 조건**
- `question id`를 **index**로 설정해야 함
- **tag 이름**들이 **column명**이 되어야 함
- 각 셀: 해당 question에 tag가 있으면 `True` 또는 `1`, 없으면 `False` 또는 `0`
- question = transaction 역할, tag = item 역할

**저장**
- `to_pickle()` 함수 사용
- 파일명: `DMA_project2_team##_part1_horizontal.pkl`

#### 체크 항목
- [x] `pd.read_csv()` 또는 동등한 방식으로 CSV 로드
- [x] question id가 DataFrame index로 설정됨 (`.set_index()` 또는 `index_col=`)
- [x] tag 이름이 column명으로 설정됨 (`pivot_table`, `get_dummies`, `groupby`+`unstack` 등 방식 무관)
- [x] 셀 값이 boolean 또는 0/1 형태 (mlxtend apriori 입력 조건)
- [x] `df.to_pickle('...horizontal.pkl')` 사용

#### 감점 요인
- index 설정 누락
- 셀 값이 0/1이 아닌 count 형태 (mlxtend apriori는 boolean 입력 필요)
- 저장 함수가 `to_pickle`이 아닌 다른 방식 사용

---

### R1-2: Association Rules + 분석 보고서 (20점)

#### 필수 구현 사항

**Frequent Itemset 생성**
```python
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True)
```
- `min_support=0.005` 정확히 일치해야 함
- `use_colnames=True` 설정 (column명을 item 이름으로 사용)

**Association Rules 생성**
```python
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=2.0)
```
- `metric='lift'` 정확히 일치해야 함
- `min_threshold=2.0` 정확히 일치해야 함

**저장**
- `rules.to_pickle('...association.pkl')`
- 파일명: `DMA_project2_team##_part1_association.pkl`

#### 체크 항목
- [x] `apriori()` 함수 사용 (mlxtend)
- [x] `min_support=0.005` 설정
- [x] `use_colnames=True` 설정
- [x] `association_rules()` 함수 사용
- [x] `metric='lift'` 설정
- [x] `min_threshold=2.0` 설정
- [x] 결과 pkl 저장

#### 보고서 확인 항목 (20점 중 보고서 비중 높음)
- [x] lift 임계값 변경 실험: 1.0 / 1.5 / 2.0 / 3.0 / 5.0 / 10.0 수준에서 분석
- [x] 각 임계값에서 추출된 규칙 수를 정량적으로 제시 (표 또는 그래프)
- [x] lift > 1 규칙 해석: 함께 자주 등장하는 tag 쌍과 그 이유
- [x] lift < 1 규칙 해석: 함께 등장하지 않는 tag 쌍과 그 이유
- [x] Q&A 사이트 특성과 연결한 도메인 지식 기반 해석 포함

---

## ③ PART II — 문서 검색 엔진 (SE 폴더)

> 성능 평가(30점)는 BPREF 점수에 의해 자동 계산됨.
> 코드 검토는 구조적 올바름 및 금지 사항 위반 여부에 집중.

### 절대 금지 사항 (위반 시 PART II 전체 0점)
- [x] `relevance.txt` 파일을 어떠한 형태로도 참조하지 않는가
  - 직접 읽기 (`open('relevance.txt')`)
  - import 또는 경로 문자열에 포함
  - 정답 문서 ID를 하드코딩
  - 정답 기반 키워드 추출 후 활용
- [x] 특정 query에 대한 예외 처리 없이 범용적으로 동작하는가

---

### CustomScoring.py 체크

**기본 구조 (TA05 슬라이드 27 기준)**
```python
# intappscorer 함수를 수정하는 방식
# 아래 변수들을 활용 가능:
# tf   : term frequency in current document
# idf  : inverse document frequency
# cf   : term frequency in the collection
# dc   : doc count (전체 문서 수)
# fl   : field length in current document (문서 길이)
# avgfl: average field length across documents
# param: free parameter
```

**BM25 이론 배경 (Ch.12 슬라이드 22 기준)**
- BM25 공식: TF saturation + document length normalization 반영
- 파라미터: k1 (TF 가중치), b (length normalization), k2 (query TF)
- TREC 권장값: k1=1.2, b=0.75

**언어 모델 이론 배경 (Ch.12 슬라이드 29-33)**
- JM smoothing: λ × P(w|D) + (1-λ) × P(w|C)
- Dirichlet smoothing: μ 파라미터로 문서 길이 기반 조정

#### 체크 항목
- [x] `intappscorer` 함수가 실제로 수정되어 있는가 (기본 BM25F 그대로가 아닌가)
- [x] tf, idf, cf, dc, fl, avgfl 중 최소 2개 이상 활용
- [x] 단순 tf × idf 조합 이상의 시도가 있는가
- [x] 보고서에서 설명한 수식과 코드가 일치하는가
- [x] `weighting=scoring.ScoringFunction()` 형태로 CustomScoring이 실제로 적용되는가

---

### QueryResult.py 체크

**기본 구조 (TA05 슬라이드 24-26 기준)**
```python
from whoosh import index
from whoosh.qparser import QueryParser, OrGroup

# searcher에 CustomScoring 적용
with ix.searcher(weighting=...) as searcher:
    # QueryParser 설정
    parser = QueryParser("contents", ix.schema, group=OrGroup.factory(q))
    query = parser.parse(query_text)
    results = searcher.search(query, limit=...)
```

**OrGroup 이론 (TA05 슬라이드 25)**
- 기본 AND 대신 OR 검색으로 recall 향상
- `OrGroup.factory(q)`: 0 < q < 1, q가 작을수록 OR에 가깝고, 클수록 AND에 가까움
- q=0: 단순 OR (모든 term 동일 취급)
- q=0.9: 두 단어 모두 포함하는 문서를 더 선호

#### 체크 항목
- [x] `QueryParser`를 사용하여 텍스트 query를 whoosh query object로 변환
- [x] 기본 AND 대신 `OrGroup` 또는 다른 방식으로 OR 검색 적용 여부
- [x] `OrGroup.factory(q)` 사용 시 q 값이 0 < q < 1 범위인가
- [x] 특정 query에 대한 하드코딩 없이 범용 동작
- [x] `relevance.txt` 참조 없음 (재확인)
- [x] 검색 결과를 document ID 기준으로 반환

---

### 성능 개선 보고서 체크 (5점)
- [x] 기본 BM25F와 구현한 방식의 차이를 명시적으로 설명
- [x] CustomScoring 수식과 파라미터 선택 근거 서술
- [x] QueryResult에서 적용한 query 처리 방법 설명 (OrGroup, weighting 등)
- [x] 실험적 근거 또는 이론적 근거 포함

---

## ④ PART III — 문서 분류 (CL/clasification.py)

> 파일명 주의: `clasification.py` (s 하나, 오타가 의도된 파일명)

### 데이터 로드 (TA06 슬라이드 4-5 기준)

**방법 1: sklearn의 load_files 사용 (일반적 방법)**
```python
from sklearn.datasets import load_files
train_data = load_files(container_path='text/train')
test_data  = load_files(container_path='text/test')
```
- `data`: raw text
- `target`: integer label array
- `target_names`: ['AnnStat', 'Biometrika', 'JASA', 'JMLR']

**방법 2: fetch_20newsgroups 사용 (20 Newsgroups 전용)**
- 이번 과제는 커스텀 데이터셋이므로 `load_files`가 적합

#### 체크 항목
- [x] train/test 분리 로드
- [x] 4개 카테고리 모두 포함: JMLR, AnnStat, JASA, Biometrika
- [x] 추가 데이터 크롤링 또는 파일 수정 없음

---

### Feature Extraction (TA06 슬라이드 7-8, Ch.13 기준)

**CountVectorizer (TA06 슬라이드 7)**
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english', ...)
```
- 옵션: `stop_words`, `ngram_range`, `max_df`, `min_df`, `max_features` 등

**TfidfTransformer (TA06 슬라이드 8)**
```python
from sklearn.feature_extraction.text import TfidfTransformer
# 또는 TfidfVectorizer (CountVectorizer + TfidfTransformer 합친 버전)
```
- Count vector → TF-IDF 변환

#### 체크 항목
- [x] CountVectorizer 또는 TfidfVectorizer 사용
- [ ] stop_words 처리 여부 (없으면 성능 하락 가능) ⚠️ 미설정
- [x] TF-IDF 변환 적용 여부

---

### Naïve Bayes Classifier (TA06 슬라이드 9-10, Ch.13 기준)

**이론 배경 (Ch.13 슬라이드 7-16)**
- Bayes' rule 기반 확률적 분류기
- Multiple Bernoulli: 단어 등장 여부 (0/1)
- Multinomial: 단어 빈도 (TF 기반) → 텍스트 분류에 더 일반적

**구현 (TA06 슬라이드 9-10)**
```python
from sklearn.naive_bayes import MultinomialNB  # 텍스트 분류 권장
# 또는 GaussianNB, BernoulliNB
clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)
y_pred = clf_nb.predict(X_test)
```

**Pipeline 사용 가능 (TA06 슬라이드 14)**
```python
from sklearn.pipeline import Pipeline
pipeline_nb = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
```

#### 체크 항목
- [x] `MultinomialNB`, `GaussianNB`, `BernoulliNB` 중 하나 사용
- [x] `.fit(X_train, y_train)` 학습
- [x] `.predict(X_test)` 예측
- [x] `accuracy_score` 또는 동등한 방법으로 정확도 출력
- [x] `nb.pkl`로 모델 저장: `pickle.dump(model, f)` 또는 `joblib.dump()`
- [x] `clasification.py` 단독 실행으로 accuracy 재현 가능

---

### SVM Classifier (TA06 슬라이드 11-13, Ch.13 기준)

**이론 배경 (Ch.13 슬라이드 17-22)**
- geometric 원리 기반, optimal hyperplane 탐색
- Linear separable: margin 최대화
- Non-separable: slack variable ξ + penalty parameter C 추가
- Multi-class 방식:
  - OVO (one-vs-one): SVC, NuSVC 기본값 → K(K-1)/2 classifier
  - OVR (one-vs-rest): LinearSVC 기본값 → K classifier

**구현 (TA06 슬라이드 11-13)**
```python
from sklearn.svm import SVC, NuSVC, LinearSVC

# LinearSVC 권장 (빠르고, 텍스트 분류에 효과적)
clf_svm = LinearSVC(C=1.0)
clf_svm.fit(X_train, y_train)
y_pred = clf_svm.predict(X_test)
```

**파라미터 (TA06 슬라이드 13)**
- `kernel`: linear, poly, rbf, sigmoid (LinearSVC는 linear 고정)
- `C`: penalty parameter (기본값 1.0)
- `degree`: polynomial kernel 차수
- `gamma`: rbf, poly, sigmoid kernel 계수

#### 체크 항목
- [x] `SVC`, `NuSVC`, `LinearSVC` 중 하나 사용
- [x] `.fit(X_train, y_train)` 학습
- [x] `.predict(X_test)` 예측
- [x] `accuracy_score` 또는 동등한 방법으로 정확도 출력
- [x] `svm.pkl`로 모델 저장
- [x] `clasification.py` 단독 실행으로 accuracy 재현 가능
- [x] (가점 요인) 파라미터 튜닝 시도 여부 (C, kernel, gamma 등)
- [x] (가점 요인) Pipeline 구성 여부

---

## ⑤ 공통 보고서 품질 체크 (10점)

### 구조 확인
- [x] 20페이지 이내
- [x] PART I, II, III 섹션 구분 명확
- [x] 파일명: `DMA_project2_team##_보고서.pdf`

### 내용 깊이
- [x] PART I: lift 임계값 분석이 정량·정성 모두 포함
- [x] PART II: scoring 함수 선택 근거가 이론적으로 뒷받침
- [x] PART III: 모델 선택 및 파라미터 튜닝 결과 서술
- [x] 결과 해석이 단순 수치 나열이 아닌 도메인 해석 포함

---

## ⑥ 검토 시 우선순위

**즉시 확인 (감점/0점 위험)**
1. `TEAM = 0` 미수정 여부
2. `relevance.txt` 참조 여부 (PART II 0점)
3. pkl 파일 존재 및 파일명 정확성

**성능 영향 큰 항목**
1. PART II: CustomScoring이 실제로 BM25F와 다른 로직인가
2. PART II: OrGroup 적용 여부 (AND 검색은 recall 매우 낮음)
3. PART III: stop_words 처리 및 TF-IDF 적용 여부

**보고서 품질**
1. PART I R1-2: lift 임계값 실험의 정량 분석 충실성
2. PART II: scoring 방법 선택 근거의 논리성

---

## 📚 이론 참고 (슬라이드 출처)

| 개념 | 출처 |
|------|------|
| BM25 파라미터 (k1, b, k2) | Ch.12, 슬라이드 22-23 |
| JM smoothing (λ) | Ch.12, 슬라이드 31 |
| Dirichlet smoothing (μ) | Ch.12, 슬라이드 33 |
| KL-divergence ranking | Ch.12, 슬라이드 36-37 |
| PRF (Pseudo Relevance Feedback) | Ch.12, 슬라이드 38-42 |
| BPREF 공식 | Ch.15, 슬라이드 23 |
| NB (Multinomial vs Bernoulli) | Ch.13, 슬라이드 10-16 |
| SVM OVO vs OVR | Ch.13, 슬라이드 22 |
| TF-IDF | Ch.12, 슬라이드 13 |
| OrGroup scaling factor | TA05, 슬라이드 25 |
| intappscorer 변수 목록 | TA05, 슬라이드 27 |
| apriori / association_rules | TA04, 슬라이드 7-10 |
| load_files 사용법 | TA06, 슬라이드 5 |
| Pipeline 구성 | TA06, 슬라이드 14 |
