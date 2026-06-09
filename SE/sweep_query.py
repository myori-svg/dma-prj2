"""
쿼리 처리 방식 변형 실험 — OrGroup factor, 전처리, stopwords 제거 여부
"""
import numpy as np
import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup, AndGroup
import CustomScoring as scoring

try:
    from nltk.corpus import stopwords as _sw
    STOP = set(_sw.words('english'))
except LookupError:
    STOP = {'a','an','and','are','as','at','be','by','for','from','has','have',
            'in','is','it','its','of','on','or','that','the','to','was','were','will','with'}


def preprocess_base(q):
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


def evaluate(label, orgroup_q=0.5, use_stopwords=True, param=3.0, no_hyphen_split=False):
    qd = read_queries()
    rd = read_relevance(set(qd))
    ix = index.open_dir("index")
    result_dict = {}
    with ix.searcher(weighting=scoring.ScoringFunction(param=param)) as searcher:
        parser = QueryParser("contents", schema=ix.schema,
                             group=OrGroup.factory(orgroup_q) if orgroup_q < 1.0 else AndGroup)
        for qid, q in qd.items():
            if no_hyphen_split:
                words = []
                for raw in q.lower().split():
                    w = raw.strip(".,;:!?()[]{}\"'")
                    if w and (not use_stopwords or w not in STOP):
                        words.append(w)
                new_q = ' '.join(words) or q.lower()
            else:
                words = []
                for raw in q.lower().replace('-', ' ').replace('/', ' ').split():
                    w = raw.strip(".,;:!?()[]{}\"'")
                    if w and (not use_stopwords or w not in STOP):
                        words.append(w)
                new_q = ' '.join(words) or q.lower()
            query = parser.parse(new_q)
            results = searcher.search(query, limit=None)
            result_dict[qid] = [r.fields()['docID'] for r in results]
    bpref = calc_bpref(result_dict, rd, qd.keys())
    print(f"{label:<55} BPREF={bpref:.6f}")
    return bpref


if __name__ == '__main__':
    print("=" * 70)
    print("쿼리 처리 방식 sweep")
    print("=" * 70)
    results = []
    configs = [
        ("baseline (OrGroup=0.5, stopwords, hyphen-split)",      dict(orgroup_q=0.5,  use_stopwords=True,  param=3.0)),
        ("OrGroup=0.1 (more OR)",                                 dict(orgroup_q=0.1,  use_stopwords=True,  param=3.0)),
        ("OrGroup=0.3",                                           dict(orgroup_q=0.3,  use_stopwords=True,  param=3.0)),
        ("OrGroup=0.7",                                           dict(orgroup_q=0.7,  use_stopwords=True,  param=3.0)),
        ("OrGroup=0.9 (near-AND)",                                dict(orgroup_q=0.9,  use_stopwords=True,  param=3.0)),
        ("no stopwords removal",                                  dict(orgroup_q=0.5,  use_stopwords=False, param=3.0)),
        ("no hyphen-split (keep 'time-series' as one token)",     dict(orgroup_q=0.5,  use_stopwords=True,  param=3.0, no_hyphen_split=True)),
        ("no stopwords + OrGroup=0.3",                            dict(orgroup_q=0.3,  use_stopwords=False, param=3.0)),
        ("no stopwords + OrGroup=0.7",                            dict(orgroup_q=0.7,  use_stopwords=False, param=3.0)),
    ]
    for label, kwargs in configs:
        bpref = evaluate(label, **kwargs)
        results.append((bpref, label))

    print("\n── 순위 ──")
    best = max(r[0] for r in results)
    for bpref, label in sorted(results, reverse=True):
        marker = " ← BEST" if bpref == best else ""
        print(f"  {bpref:.6f}  {label}{marker}")
