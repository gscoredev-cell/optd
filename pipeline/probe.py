"""Detector (§9.2): logistic regression on one layer's boundary states, trace label on every boundary, layer chosen
by held-out AUROC (20% of original_ids held out), trace score = max over boundaries, tau at FPR <= 10%."""
import collections

import numpy as np

from core import auroc, holdout_by_oid, tau_at_fpr


def _probs(p, feats, li):
    return p["clf"].predict_proba(p["scaler"].transform(feats[:, li].numpy()))[:, 1] if len(feats) else np.zeros(0)


def first_flag(probs, tau):
    """1-based index of the first boundary with score >= tau, else None."""
    hit = np.nonzero(np.asarray(probs) >= tau)[0]
    return int(hit[0]) + 1 if len(hit) else None


def fit(feats, wrong, oids, layers):
    """feats: per trace [n_boundaries, n_layers, d]; wrong: 1 if the trace's strict answer is wrong."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    hold = holdout_by_oid(oids)
    tr = [i for i, o in enumerate(oids) if o not in hold and len(feats[i])]
    te = [i for i, o in enumerate(oids) if o in hold and len(feats[i])]
    y = np.concatenate([[wrong[i]] * len(feats[i]) for i in tr]) if tr else np.zeros(0)
    if len(set(y.tolist())) < 2 or not te:
        return {"layer": None, "tau": float("inf"), "metrics": {"probe_blocker": "single-class or empty training labels"}}
    per_layer = {}
    for li, layer in enumerate(layers):
        X = np.concatenate([feats[i][:, li].numpy() for i in tr])
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X), y)
        p = {"scaler": scaler, "clf": clf}
        score = {i: float(_probs(p, feats[i], li).max()) for i in te}
        a = auroc([score[i] for i in te if wrong[i]], [score[i] for i in te if not wrong[i]])
        per_layer[layer] = (a, li, p)
    layer = max(per_layer, key=lambda l: (np.nan_to_num(per_layer[l][0], nan=-1), l))
    a, li, p = per_layer[layer]
    neg = [float(_probs(p, feats[i], li).max()) for i in te if not wrong[i]]
    pos = [float(_probs(p, feats[i], li).max()) for i in te if wrong[i]]
    tau = tau_at_fpr(neg) if neg else float("inf")
    flags = [first_flag(_probs(p, feats[i], li), tau) for i in te]
    flags = [f for f in flags if f]
    restart = sum(f == 1 for f in flags) / len(flags) if flags else float("nan")
    return {"layer": layer, "li": li, **p, "tau": tau, "metrics": {
        "probe_layer": layer, "probe_auroc": a, "probe_auroc_by_layer": {str(l): v[0] for l, v in per_layer.items()},
        "probe_tau": tau, "probe_fpr": float(np.mean(np.array(neg) >= tau)) if neg else float("nan"),
        "probe_recall": float(np.mean(np.array(pos) >= tau)) if pos else float("nan"),
        "probe_restart_share": restart, "probe_first_flag_hist": dict(sorted(collections.Counter(flags).items())),
        "probe_holdout_traces": len(te), "probe_train_boundaries": int(len(y))}}


def flag(probe, feats):
    """First flagged boundary (1-based) for one trace, or None."""
    if probe["layer"] is None:
        return None
    return first_flag(_probs(probe, feats, probe["li"]), probe["tau"])
