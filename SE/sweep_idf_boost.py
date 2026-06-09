"""
IDF 기반 자동 term boost 실험 — 희귀 용어에 가중치를 줘서 순위 개선
relevance.txt는 평가용으로만 사용, 쿼리 처리에는 미사용
"""
import numpy as np
from math import log
import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
import CustomScoring as scoring

try:
    from nltk.corpus import stopwords as _sw
    STOP = set(_sw.words('english'))
except LookupError:
    STOP = {'a','an','and','are','as','at','be','by','for','from','has','have',
            'in','is','it','its','of','on','or','that','the','to','was','were','will','with'}


def calc_bpref(result_dict, relevant_dict, query_ids):
    scores = []
    for qid in query_ids:
        rel = relevant_dict[qid]
        rc = nc = s = 0
        for doc in result_dict[qid]:
            if doc in rel:
                rc += 1; s += 1 - min(nc, len(rel)) / len(rel)
            else: nc += 1
            if rc == len(rel): break
        scores.append(s / len(rel))
    return float(np.mean(scores))


def read_queries():
    qd = {}
    with open('doc/query.txt') as f:
        for blk in f.read().split('   /\n'):
            br = blk.find('\n')
            if br > 0: qd[int(blk[:br])] = blk[br+1:].strip()
    return qd


def read_relevance(qids):
    rd = {}
    with open('doc/relevance.txt') as f:
        for line in f:
            items = line.split(); qid, did = int(items[0]), int(items[1])
            if qid in qids: rd.setdefault(qid, []).append(did)
    return rd


def evaluate_with_boost(boost_fn, label):
    """boost_fn(searcher, words) → boosted query string"""
    qd = read_queries()
    rd = read_relevance(set(qd))
    ix = index.open_dir("index")
    result_dict = {}
    with ix.searcher(weighting=scoring.ScoringFunction(param=3.0)) as searcher:
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.5))
        dc = searcher.doc_count_all()
        for qid, q in qd.items():
            words = []
            for raw in q.lower().replace('-', ' ').replace('/', ' ').split():
                w = raw.strip(".,;:!?()[]{}\"'")
                if w and w not in STOP:
                    words.append(w)
            new_q = boost_fn(searcher, words, dc) if words else q.lower()
            query = parser.parse(new_q or q.lower())
            results = searcher.search(query, limit=None)
            result_dict[qid] = [r.fields()['docID'] for r in results]
    bpref = calc_bpref(result_dict, rd, qd.keys())
    print(f"{label:<55} BPREF={bpref:.6f}")
    return bpref


def no_boost(searcher, words, dc):
    return ' '.join(words)


def idf_boost_top1(searcher, words, dc):
    """가장 희귀한 term 1개를 ^2.0 boost"""
    term_df = [(w, searcher.doc_frequency("contents", w)) for w in words]
    if not term_df: return ' '.join(words)
    rarest = min(term_df, key=lambda x: x[1] if x[1] > 0 else 999999)
    if rarest[1] == 0:  # 인덱스에 없는 단어는 boost 불가
        return ' '.join(words)
    return ' '.join(f'{w}^2.0' if w == rarest[0] else w for w in words)


def idf_boost_proportional(searcher, words, dc):
    """모든 term을 IDF에 비례하여 boost (IDF 정규화)"""
    term_df = [(w, max(searcher.doc_frequency("contents", w), 1)) for w in words]
    if not term_df: return ' '.join(words)
    idfs = [(w, log(dc / df + 1)) for w, df in term_df]
    max_idf = max(idf for _, idf in idfs)
    if max_idf == 0: return ' '.join(words)
    boosted = [f'{w}^{idf/max_idf:.2f}' for w, idf in idfs]
    return ' '.join(boosted)


def idf_boost_drop_common(searcher, words, dc, threshold_df=200):
    """df > threshold인 매우 흔한 term 제거 (길이가 3 이상인 쿼리에서만)"""
    if len(words) <= 2:
        return ' '.join(words)
    filtered = [w for w in words if searcher.doc_frequency("contents", w) <= threshold_df]
    return ' '.join(filtered) if filtered else ' '.join(words)


def idf_boost_drop100(searcher, words, dc):
    return idf_boost_drop_common(searcher, words, dc, threshold_df=100)


def idf_boost_drop200(searcher, words, dc):
    return idf_boost_drop_common(searcher, words, dc, threshold_df=200)


def idf_boost_top2_heavy(searcher, words, dc):
    """희귀한 term 2개를 ^3.0으로 강하게 boost"""
    term_df = sorted([(w, searcher.doc_frequency("contents", w)) for w in words],
                     key=lambda x: x[1] if x[1] > 0 else 999999)
    top2 = {w for w, df in term_df[:2] if df > 0}
    return ' '.join(f'{w}^3.0' if w in top2 else w for w in words)


def idf_boost_prop_drop_common(searcher, words, dc):
    """IDF 비례 boost + 매우 흔한 term(df>200) 제거 조합"""
    if len(words) > 2:
        words = [w for w in words if searcher.doc_frequency("contents", w) <= 200 or True]
        keep = [w for w in words if searcher.doc_frequency("contents", w) <= 200]
        if len(keep) >= 2:
            words = keep
    return idf_boost_proportional(searcher, words, dc)


if __name__ == '__main__':
    print("=" * 70)
    print("IDF-based term boost sweep")
    print("=" * 70)
    results = []
    experiments = [
        (no_boost,                "baseline (no boost)"),
        (idf_boost_top1,          "boost rarest term ^2.0"),
        (idf_boost_top2_heavy,    "boost rarest 2 terms ^3.0"),
        (idf_boost_proportional,  "boost all proportional to IDF"),
        (idf_boost_drop100,       "drop common terms (df>100)"),
        (idf_boost_drop200,       "drop common terms (df>200)"),
        (idf_boost_prop_drop_common, "IDF boost + drop common (df>200)"),
    ]
    for fn, label in experiments:
        bpref = evaluate_with_boost(fn, label)
        results.append((bpref, label))

    print("\n── 순위 ──")
    best = max(r[0] for r in results)
    for bpref, label in sorted(results, reverse=True):
        marker = " ← BEST" if bpref == best else ""
        delta = bpref - results[0][0]
        sign = "+" if delta >= 0 else ""
        print(f"  {bpref:.6f} ({sign}{delta:+.6f})  {label}{marker}")
