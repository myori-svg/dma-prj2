import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
from whoosh import scoring
import CustomScoring as scoring
from nltk.corpus import stopwords


def getSearchEngineResult(query_dict):
    """
    질의어 딕셔너리를 받아 검색 결과를 반환한다.
    - query_dict: {queryID: query_text, ...}
    - 반환값: {queryID: [docID, docID, ...], ...}  (관련도 순)
    """
    result_dict = {}
    ix = index.open_dir("index")

    # ============================================================
    # !! TODO 1 - weighting 모델을 선택하세요 !!
    # 기본값: scoring.BM25F()  (whoosh 기본 BM25F 모델)
    # Custom scoring 사용 시: scoring.ScoringFunction()
    #   -> CustomScoring.py의 intappscorer() 함수가 호출됩니다.
    # ============================================================
    with ix.searcher(weighting=scoring.ScoringFunction(param=3.0)) as searcher:
    # with ix.searcher(weighting=scoring.ScoringFunction()) as searcher:

        # ============================================================
        # !! TODO 2 - 질의어 전처리를 개선하세요 !!
        # 아래는 기본 예시 (OrGroup + stopword 제거)입니다.
        # 쿼리 파싱 방법, 전처리 방법 등을 자유롭게 수정할 수 있습니다.
        # ============================================================
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.5))
        try:
            stopWords = set(stopwords.words('english'))
        except LookupError:
            stopWords = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                'from', 'has', 'have', 'in', 'is', 'it', 'its', 'of',
                'on', 'or', 'that', 'the', 'to', 'was', 'were', 'will',
                'with'
            }

        for qid, q in query_dict.items():
            words = []
            for raw_word in q.lower().replace('-', ' ').replace('/', ' ').split():
                word = raw_word.strip(".,;:!?()[]{}\"'")
                if word and word not in stopWords:
                    words.append(word)
            new_q = ' '.join(words)
            query = parser.parse(new_q or q.lower())

            # !! DO NOT MODIFY - 검색 실행 및 결과 수집 !!
            results = searcher.search(query, limit=None)
            result_dict[qid] = [result.fields()['docID'] for result in results]

    return result_dict
