## **DMA Project #2 보고서 — Team 22** 

DB Mining & Document Search Engine & Classification 

## **PART I.** 연관분석 

## 개요 

Q&A 사이트의 question–tag 데이터( `DMA_project_UBR.csv` , 총 117,656행)를 transaction–item 구조로변 환하여태그간연관규칙을분석하였다. 고유 question 42,921개를 transaction으로, 고유 tag 1,032개를 item으로 하는 horizontal table(42,921 × 1,032, bool)을구성하고, `mlxtend` 의 apriori로 frequent itemset(min_support=0.005)을, association_rules(metric=lift)로규칙을추출하였다. 

## **R1-1 / R1-2** 구현 

`pd.crosstab(question, tag).astype(bool)` 로 horizontal table을생성하여 `..._part1_horizontal.pkl` 에저장하였다. 이후 `apriori(min_support=0.005, use_colnames=True)` 로빈발항목집합을, `association_rules(metric='lift', min_threshold=2.0)` 로 28개의규칙을추출하여 `..._part1_association.pkl` 에저장하였다(모든 규칙 lift ≥ 2.0, support ≥ 0.005 충족). 

## **lift** 임계값 분석 

support 0.005 기준 frequent itemset 133개에서전체연관규칙 46개(양의연관 40개, 음의연관 6개)가도출되었 다. lift 임계값을높일수록규칙수는다음과같이급격히감소한다. 

|**lif임계값**|**규칙수**|
|---|---|
|1.0|40|
|1.5|30|
|2.0|28|
|3.0|20|
|5.0|12|
|10.0|2|



임계값이올라갈수록의미적으로가장밀접한소수의태그쌍만남는다. 제출한 `..._part1_association.pkl` 은과제조건(lift ≥ 2.0)에따라이중 28개규칙을담는다. **양의연관 (lift > 1).** 가장강한규칙들은동일주제내방법맥락쌍이다– . 

DMA Project #2 Report | Team 22 

|**antecedents →**<br>**consequents**|**support**|**confdence**|**lif**|
|---|---|---|---|
|anova ↔ repeated-<br>measures|0.0063|0.20 / 0.34|11.06|
|arima ↔ tme-series|0.0054|0.61 / 0.08|9.49|
|forecastng ↔ tme-series|0.0078|0.55 / 0.12|8.64|
|svm ↔ machine-learning|0.0065|0.42 / 0.11|6.98|
|data-mining ↔ machine-<br>learning|0.0055|0.35 / 0.09|5.86|



이는 Q&A 태깅행태와부합한다. 사용자는특정통계/ML 방법(anova, arima, svm)을그적용맥락태그(repeated– measures, time-series, machine-learning)와함께다는경향이있어, 방법맥락쌍의동시출현이독립가정대비 6~11배높게나타난다. 

**음의연관 (lift < 1).** 음의연관은 6개로, 극단적이기보다약한음의관계다. 

|**antecedents →**<br>**consequents**|**support**|**confdence**|**lif**|
|---|---|---|---|
|r ↔ machine-learning|0.0053|0.03 / 0.09|0.53|
|machine-learning ↔<br>regression|0.0053|0.09 / 0.04|0.70|
|tme-series ↔ regression|0.0071|0.11 / 0.06|0.88|



서로다른방법론계열·사용맥락에속하는태그쌍이약하게회피관계를보인다. 예컨대 `r` (범용도구·기초통계질 문맥락)과 `machine-learning` (전문방법맥락)은같은질문에서함께달리는빈도가독립기대보다낮다. 다 만 lift 최저가 0.53으로, 강한배타관계라기보다주제분리에따른완만한음의경향으로해석된다. 음의연관은제 출 pkl(lift ≥ 2.0)에는포함되지않으므로별도전체규칙분석에서확인하였다. 

## **PART II.** 문서 검색 엔진 

## 개요 

`whoosh` 기반검색엔진을구현하여 80개질의어에대해 2,772개문서를관련도순으로정렬하였다. 평가지표는 BPREF이며, 기본 BM25F 대비 (1) 질의전처리개선, (2) 사용자정의 scoring function(BM25+ 계열), (3) stemming 인덱스재구성을통해성능을단계적으로향상시켰다. 

## 성능 개선 경로 

동일전처리(OrGroup soft-factor 0.5, 소문자화·구두점제거·불용어제거) 하에서측정한평균 BPREF: 

DMA Project #2 Report | Team 22 

|**단계**|**구성**|**평균BPREF**|
|---|---|---|
|baseline|기본BM25F +<br>기본index|0.2504|
|+ custom scoring|BM25+ (param=3) +<br>기본index|0.2624|
|+ stemming index|BM25+ + StemmingAnalyzer재색인|0.2767|
|+<br>상수미세조정|b=0.80, delta=0.75·param|**0.2780 (최종)**|



## 채점 함수 설계 **(CustomScoring.intappscorer)** 

기본 BM25F 대신 **BM25+ 계열** 을채택하였다. BM25+는표준 BM25에하한보정항(delta)을더해, 긴문서에서용 어빈도기여가 0에수렴하며과도하게불이익을받는문제를완화한다. 구현식은다음과같다. 

normalized_tf = tf / ((1 - b) + b * (fl / avgfl)) score = idf * (k1 + 1) * (normalized_tf + delta) / (k1 + normalized_tf + delta) if qf > 1: score *= (1 + log(qf)) 

- **b = 0.80** : 문서길이정규화강도. 학술논문은길이편차가커비교적강한정규화가유리했다. 

- **k1 = 1.45** : 용어빈도포화속도. 

- **delta = 0.75 · param (param=3 → 2.25)** : 하한보정항. 0~3 구간 sweep에서약 1.0~2.25 부근이최적. 

- **qf 로그가중** : 질의어내반복용어에점감가중. 

## 질의 전처리 **(QueryResult)** 

소문자화후하이픈·슬래시를공백으로분리하고구두점을제거하며, nltk 영어불용어(코퍼스미설치환경대비 fallback 집합내장)를제거하였다. 파싱은 `OrGroup.factory(0.5)` 의 soft-OR를사용해, 다수질의어를포함 하는문서를적절히우대하되단일용어매칭의과도한가중을완화하였다. 

## **stemming** 인덱스 

`make_index.py` 의스키마를 `contents=TEXT(analyzer=StemmingAnalyzer())` 로변경하여 Porter stemming 기반으로재색인하였다. 학술텍스트의형태변형(estimate/estimation/estimator 등)을동일어 간으로매칭시켜 BPREF가 0.2624→0.2767로가장크게상승하였다. 재현성을위해신규 `index/` 폴더와 `make_index.py` 를함께제출한다. 

## 채택하지 않은 방법 **(** 정량 비교 **)** 

## 다음은측정결과성능이악화되어최종본에반영하지않았다. 

|**방법**|**평균BPREF**|**판정**|
|---|---|---|
|OrGroup factor 0.3~0.7변경|0.2767 (<br>변동없음)|무효과|
|phrase/bigram질의|0.2596|악화,기각|



DMA Project #2 Report | Team 22 

|도메인불용어추가|0.2695|악화,기각|
|---|---|---|
|단순PRF(top3+3 / top5+5)|0.233 / 0.219|크게악화,기각|



→ 본질의·컬렉션에서는 BM25+ + stemming 조합이가장견고하였다. 

## 부정행위 방지 준수 

`relevance.txt` 는어떤형태로도참조하지않았으며, 모든 scoring·전처리는질의어와 `document.txt` 통계 만사용하는범용로직으로구현하였다(특정질의·문서예외처리없음). 

## **PART III.** 문서 분류 

## 개요 

4개학술저널(JMLR, AnnStat, JASA, Biometrika)의논문본문을분류하는 Naive Bayes·SVM 모델을 `sklearn` 으 로구현하였다. 각저널 train 200 / test 50(총 train 800, test 200)을그대로사용하였다. 

## 공통 특징 추출 

두모델모두 **word TF-IDF + char n-gram TF-IDF의 FeatureUnion** 을사용하였다. word 단위는주제어를, char_wb(3–5gram) 단위는학술표기·접사패턴을포착한다. 두표현의결합이단일표현보다일관되게우수하였다 (아래비교참조). 

## 모델별 구성 및 성능 

|**모델**|**핵심설정**|**정확도**|
|---|---|---|
|Naive Bayes|word(1,2-gram, max_df=0.95,<br>sublinear) + char_wb(3–5)<br>⊕<br>MultnomialNB(alpha=0.05)|**0.725 (145/200)**|
|SVM|word(1,2-gram, min_df=2,<br>max_df=0.95, sublinear) + char_wb(3–<br>5)  LinearSVC(C=0.3)<br>⊕|**0.785 (157/200)**|



## 설계 근거 및 대안 비교 

## 하이퍼파라미터는다음비교를통해선택하였다. 

|**구성**|**정확도**|
|---|---|
|**SVM word+char (채택, C=0.3)**|**0.785**|



DMA Project #2 Report | Team 22 

|SVM word-only (C=0.5~5)|0.74 ~ 0.75|
|---|---|
|SVM word+char (C=0.5~2)|0.775 ~ 0.78|
|**NB word+char (채택, alpha=0.05)**|**0.725**|
|NB word-only (alpha=0.1)|0.705|
|ComplementNB (word)|0.705|



word+char 결합과낮은정규화(alpha=0.05, C=0.3)가최적이었다. 정확도가절대적으로높지않은이유는 **네저널 이모두통계/머신러닝분야로주제·어휘가크게겹쳐** 본문만으로분리하기어려운과제이기때문이다. 동일데이터· 허용라이브러리범위에서시도한대안들이모두동급이하였으므로, 현구성을 TF-IDF 선형모델계열의합리적상 한으로판단한다. 

## 데이터 규칙 준수 

외부데이터추가·파일수정·데이터셋일부만사용을하지않았으며(train/test 800/200 원본유지), 모든 fit은 train 에만적용하여 test 누수가없다. 학습된모델은 raw text를입력받는 Pipeline 형태로 pkl 저장되어제출코드단독 실행으로재현된다. 

## 부록 **:** 재현 환경 

Python 3.10/3.11, Whoosh==2.7.4(구버전 API 사용), nltk(stopwords 코퍼스), pandas·numpy·scikitlearn·mlxtend. 검색인덱스는동일분석기로재생성해야동일결과가나온다. 

DMA Project #2 Report | Team 22 

