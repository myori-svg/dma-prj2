"""
BPREF sweep script — bpref-exp 브랜치 실험용
evaluate.py의 평가 로직을 재사용해 여러 scorer를 빠르게 비교.
relevance.txt는 오프라인 측정용으로만 사용, 제출 코드(QueryResult/CustomScoring)에는 미포함.
"""
import numpy as np
from math import log
import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
import CustomScoring as _cs

try:
    from nltk.corpus import stopwords as _sw
    STOP = set(_sw.words('english'))
except LookupError:
    STOP = {'a','an','and','are','as','at','be','by','for','from','has','have',
            'in','is','it','its','of','on','or','that','the','to','was','were','will','with'}


def _preprocess(q):
    words = []
    for raw in q.lower().replace('-', ' ').replace('/', ' ').split():
        w = raw.strip(".,;:!?()[]{}\"'")
        if w and w not in STOP:
            words.append(w)
    return ' '.join(words) or q.lower()


def _bpref(result_dict, relevant_dict, query_ids):
    scores = []
    for qid in query_ids:
        rel = relevant_dict[qid]
        rel_count = len(rel)
        rc = nc = s = 0
        for doc in result_dict[qid]:
            if doc in rel:
                rc += 1
                s += 1 - min(nc, rel_count) / rel_count
            else:
                nc += 1
            if rc == rel_count:
                break
        scores.append(s / rel_count)
    return float(np.mean(scores))


def _read_queries():
    qd = {}
    with open('doc/query.txt') as f:
        for blk in f.read().split('   /\n'):
            br = blk.find('\n')
            qd[int(blk[:br])] = blk[br+1:]
    return qd


def _read_relevance(qids):
    rd = {}
    with open('doc/relevance.txt') as f:
        for line in f:
            items = line.split()
            qid, did = int(items[0]), int(items[1])
            if qid in qids:
                rd.setdefault(qid, []).append(did)
    return rd


def run(scorer_fn, param=3.0, label=""):
    """scorer_fn을 monkey-patch해서 BPREF 측정"""
    orig = _cs.intappscorer
    _cs.intappscorer = scorer_fn
    try:
        qd = _read_queries()
        rd = _read_relevance(set(qd))
        ix = index.open_dir("index")
        result_dict = {}
        with ix.searcher(weighting=_cs.ScoringFunction(param=param)) as searcher:
            parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.5))
            for qid, q in qd.items():
                query = parser.parse(_preprocess(q))
                results = searcher.search(query, limit=None)
                result_dict[qid] = [r.fields()['docID'] for r in results]
        bpref = _bpref(result_dict, rd, qd.keys())
        print(f"{label or scorer_fn.__name__:<45} BPREF={bpref:.6f}")
        return bpref
    finally:
        _cs.intappscorer = orig


# ── 실험 scorer 정의 ────────────────────────────────────────────

def current_bm25plus(tf, idf, cf, qf, dc, fl, avgfl, param):
    """현재 제출 코드 (baseline)"""
    tf = max(float(tf), 0.0); idf = max(float(idf), 0.0)
    fl = max(float(fl), 1.0); avgfl = max(float(avgfl), 1.0)
    qf = max(float(qf), 1.0)
    b, k1, delta = 0.80, 1.45, 0.75 * max(float(param), 0.0)
    ntf = tf / ((1.0 - b) + b * (fl / avgfl))
    s = idf * ((k1 + 1.0) * (ntf + delta) / (k1 + ntf + delta))
    if qf > 1.0:
        s *= 1.0 + log(qf)
    return float(s)


def bm25plus_dirichlet(tf, idf, cf, qf, dc, fl, avgfl, param):
    """BM25+ + Dirichlet collection smoothing 블렌드 (cf, dc 활용)"""
    tf = max(float(tf), 0.0); idf = max(float(idf), 0.0)
    fl = max(float(fl), 1.0); avgfl = max(float(avgfl), 1.0)
    qf = max(float(qf), 1.0); cf = max(float(cf), 1.0); dc = max(float(dc), 1.0)
    b, k1, delta = 0.80, 1.45, 0.75 * max(float(param), 0.0)
    ntf = tf / ((1.0 - b) + b * (fl / avgfl))
    bm25_s = idf * ((k1 + 1.0) * (ntf + delta) / (k1 + ntf + delta))
    mu = 2500.0
    p_c = cf / (dc * avgfl)
    dirichlet_boost = idf * log(1.0 + tf / max(fl * p_c, 1e-10)) if tf > 0 else 0.0
    s = bm25_s + 0.05 * dirichlet_boost
    if qf > 1.0:
        s *= 1.0 + log(qf)
    return float(s)


def dirichlet_lm(tf, idf, cf, qf, dc, fl, avgfl, param):
    """순수 Dirichlet 언어모델 (BM25 없음, cf/dc 주역)"""
    tf = max(float(tf), 0.0); idf = max(float(idf), 0.0)
    fl = max(float(fl), 1.0); avgfl = max(float(avgfl), 1.0)
    cf = max(float(cf), 1.0); dc = max(float(dc), 1.0)
    mu = max(float(param) * 500.0, 500.0)
    p_c = cf / (dc * avgfl)
    p_d = (tf + mu * p_c) / (fl + mu)
    s = idf * log(p_d / p_c) if p_d > p_c else 0.0
    return float(max(s, 0.0))


def bm25plus_b075(tf, idf, cf, qf, dc, fl, avgfl, param):
    """b=0.75 (TREC 권장값) 테스트"""
    tf = max(float(tf), 0.0); idf = max(float(idf), 0.0)
    fl = max(float(fl), 1.0); avgfl = max(float(avgfl), 1.0)
    qf = max(float(qf), 1.0)
    b, k1, delta = 0.75, 1.45, 0.75 * max(float(param), 0.0)
    ntf = tf / ((1.0 - b) + b * (fl / avgfl))
    s = idf * ((k1 + 1.0) * (ntf + delta) / (k1 + ntf + delta))
    if qf > 1.0: s *= 1.0 + log(qf)
    return float(s)


def bm25plus_b070(tf, idf, cf, qf, dc, fl, avgfl, param):
    """b=0.70 테스트"""
    tf = max(float(tf), 0.0); idf = max(float(idf), 0.0)
    fl = max(float(fl), 1.0); avgfl = max(float(avgfl), 1.0)
    qf = max(float(qf), 1.0)
    b, k1, delta = 0.70, 1.45, 0.75 * max(float(param), 0.0)
    ntf = tf / ((1.0 - b) + b * (fl / avgfl))
    s = idf * ((k1 + 1.0) * (ntf + delta) / (k1 + ntf + delta))
    if qf > 1.0: s *= 1.0 + log(qf)
    return float(s)


def bm25plus_b090(tf, idf, cf, qf, dc, fl, avgfl, param):
    """b=0.90 테스트"""
    tf = max(float(tf), 0.0); idf = max(float(idf), 0.0)
    fl = max(float(fl), 1.0); avgfl = max(float(avgfl), 1.0)
    qf = max(float(qf), 1.0)
    b, k1, delta = 0.90, 1.45, 0.75 * max(float(param), 0.0)
    ntf = tf / ((1.0 - b) + b * (fl / avgfl))
    s = idf * ((k1 + 1.0) * (ntf + delta) / (k1 + ntf + delta))
    if qf > 1.0: s *= 1.0 + log(qf)
    return float(s)


def bm25plus_k1_12(tf, idf, cf, qf, dc, fl, avgfl, param):
    """k1=1.2 (BM25 표준) 테스트"""
    tf = max(float(tf), 0.0); idf = max(float(idf), 0.0)
    fl = max(float(fl), 1.0); avgfl = max(float(avgfl), 1.0)
    qf = max(float(qf), 1.0)
    b, k1, delta = 0.80, 1.20, 0.75 * max(float(param), 0.0)
    ntf = tf / ((1.0 - b) + b * (fl / avgfl))
    s = idf * ((k1 + 1.0) * (ntf + delta) / (k1 + ntf + delta))
    if qf > 1.0: s *= 1.0 + log(qf)
    return float(s)


def bm25plus_k1_20(tf, idf, cf, qf, dc, fl, avgfl, param):
    """k1=2.0 테스트"""
    tf = max(float(tf), 0.0); idf = max(float(idf), 0.0)
    fl = max(float(fl), 1.0); avgfl = max(float(avgfl), 1.0)
    qf = max(float(qf), 1.0)
    b, k1, delta = 0.80, 2.00, 0.75 * max(float(param), 0.0)
    ntf = tf / ((1.0 - b) + b * (fl / avgfl))
    s = idf * ((k1 + 1.0) * (ntf + delta) / (k1 + ntf + delta))
    if qf > 1.0: s *= 1.0 + log(qf)
    return float(s)


if __name__ == '__main__':
    print("=" * 60)
    print("BPREF sweep (bpref-exp branch)")
    print("=" * 60)
    results = []
    for fn in [current_bm25plus, bm25plus_b075, bm25plus_b070, bm25plus_b090,
               bm25plus_k1_12, bm25plus_k1_20,
               bm25plus_dirichlet, dirichlet_lm]:
        bpref = run(fn, param=3.0, label=fn.__name__)
        results.append((bpref, fn.__name__))

    print("\n── 순위 ──")
    for bpref, name in sorted(results, reverse=True):
        marker = " ← BEST" if bpref == max(r[0] for r in results) else ""
        print(f"  {bpref:.6f}  {name}{marker}")
