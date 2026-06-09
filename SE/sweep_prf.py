"""
PRF / 쿼리 확장 실험 스크립트 — bpref-exp 브랜치
"""
import numpy as np
import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
import CustomScoring as scoring

try:
    from nltk.corpus import stopwords as _sw
    STOP = set(_sw.words('english'))
except LookupError:
    STOP = {'a','an','and','are','as','at','be','by','for','from','has','have',
            'in','is','it','its','of','on','or','that','the','to','was','were','will','with'}


def preprocess(q):
    words = []
    for raw in q.lower().replace('-', ' ').replace('/', ' ').split():
        w = raw.strip(".,;:!?()[]{}\"'")
        if w and w not in STOP:
            words.append(w)
    return ' '.join(words) or q.lower()


def calc_bpref(result_dict, relevant_dict, query_ids):
    scores = []
    for qid in query_ids:
        rel = relevant_dict[qid]
        rc = nc = s = 0
        for doc in result_dict[qid]:
            if doc in rel:
                rc += 1
                s += 1 - min(nc, len(rel)) / len(rel)
            else:
                nc += 1
            if rc == len(rel):
                break
        scores.append(s / len(rel))
    return float(np.mean(scores))


def read_queries():
    qd = {}
    with open('doc/query.txt') as f:
        for blk in f.read().split('   /\n'):
            br = blk.find('\n')
            qd[int(blk[:br])] = blk[br+1:]
    return qd


def read_relevance(qids):
    rd = {}
    with open('doc/relevance.txt') as f:
        for line in f:
            items = line.split()
            qid, did = int(items[0]), int(items[1])
            if qid in qids:
                rd.setdefault(qid, []).append(did)
    return rd


def evaluate(strategy_fn, label):
    """strategy_fn(searcher, parser, qid, q, new_q) → result list"""
    qd = read_queries()
    rd = read_relevance(set(qd))
    ix = index.open_dir("index")
    result_dict = {}
    with ix.searcher(weighting=scoring.ScoringFunction(param=3.0)) as searcher:
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.5))
        for qid, q in qd.items():
            new_q = preprocess(q)
            result_dict[qid] = strategy_fn(searcher, parser, qid, q, new_q)
    bpref = calc_bpref(result_dict, rd, qd.keys())
    print(f"{label:<50} BPREF={bpref:.6f}")
    return bpref


# ── 전략 정의 ────────────────────────────────────────────────────

def baseline(searcher, parser, qid, q, new_q):
    """현재 코드 — 변경 없음"""
    query = parser.parse(new_q)
    results = searcher.search(query, limit=None)
    return [r.fields()['docID'] for r in results]


def prf_key_terms_top3_n2(searcher, parser, qid, q, new_q):
    """1차 top-3 결과에서 key_terms 2개 추가"""
    query = parser.parse(new_q)
    first = searcher.search(query, limit=3)
    if len(first) > 0:
        try:
            extra = [t for t, _ in first.key_terms("contents", numterms=2)
                     if t not in new_q.split()]
            if extra:
                query = parser.parse(new_q + " " + " ".join(extra))
        except Exception:
            pass
    results = searcher.search(query, limit=None)
    return [r.fields()['docID'] for r in results]


def prf_key_terms_top5_n3(searcher, parser, qid, q, new_q):
    """1차 top-5에서 key_terms 3개 추가"""
    query = parser.parse(new_q)
    first = searcher.search(query, limit=5)
    if len(first) > 0:
        try:
            extra = [t for t, _ in first.key_terms("contents", numterms=3)
                     if t not in new_q.split()]
            if extra:
                query = parser.parse(new_q + " " + " ".join(extra))
        except Exception:
            pass
    results = searcher.search(query, limit=None)
    return [r.fields()['docID'] for r in results]


def prf_key_terms_short_query_only(searcher, parser, qid, q, new_q):
    """질의어 단어 수 ≤ 3일 때만 key_terms 2개 확장 (긴 쿼리는 그대로)"""
    query = parser.parse(new_q)
    if len(new_q.split()) <= 3:
        first = searcher.search(query, limit=5)
        if len(first) > 0:
            try:
                extra = [t for _, t in first.key_terms("contents", numterms=2)
                         if t not in new_q.split()]
                if extra:
                    query = parser.parse(new_q + " " + " ".join(extra))
            except Exception:
                pass
    results = searcher.search(query, limit=None)
    return [r.fields()['docID'] for r in results]


def prf_key_terms_top3_n1(searcher, parser, qid, q, new_q):
    """가장 보수적: top-3에서 key_terms 딱 1개만"""
    query = parser.parse(new_q)
    first = searcher.search(query, limit=3)
    if len(first) > 0:
        try:
            extra = [t for t, _ in first.key_terms("contents", numterms=1)
                     if t not in new_q.split()]
            if extra:
                query = parser.parse(new_q + " " + extra[0])
        except Exception:
            pass
    results = searcher.search(query, limit=None)
    return [r.fields()['docID'] for r in results]


if __name__ == '__main__':
    print("=" * 65)
    print("PRF / 쿼리 확장 sweep (bpref-exp branch)")
    print("=" * 65)
    results = []
    for fn, label in [
        (baseline, "baseline (no PRF)"),
        (prf_key_terms_top3_n1, "PRF: top3 key_terms=1"),
        (prf_key_terms_top3_n2, "PRF: top3 key_terms=2"),
        (prf_key_terms_top5_n3, "PRF: top5 key_terms=3"),
        (prf_key_terms_short_query_only, "PRF: short query only (<=3 words, top5 n=2)"),
    ]:
        bpref = evaluate(fn, label)
        results.append((bpref, label))

    print("\n── 순위 ──")
    for bpref, label in sorted(results, reverse=True):
        marker = " ← BEST" if bpref == max(r[0] for r in results) else ""
        print(f"  {bpref:.6f}  {label}{marker}")
