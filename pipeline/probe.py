"""Detector (§9.2): standardized boundary states -> L2 logistic regression; the trace label (primary rule wrong) on
every boundary; trace score = max over boundaries. 20% of original_ids are held out for the gate. On the training ids,
C is chosen per layer set by 5-fold CV grouped by original_id, and the layer set by its best CV AUROC; the held-out
ids only report AUROC / tau (FPR <= 10% on correct traces) / recall."""
import collections

import numpy as np

from core import auroc, group_kfold, holdout_by_oid, tau_at_fpr

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def first_flag(probs, tau):
    """1-based index of the first boundary with score >= tau, else None."""
    hit = np.nonzero(np.asarray(probs) >= tau)[0]
    return int(hit[0]) + 1 if len(hit) else None


def _x(f, cols):
    """Boundary features [n_b, n_layers, d] -> [n_b, len(cols) * d] for the chosen layer columns."""
    return np.asarray(f[:, cols], dtype=np.float32).reshape(len(f), -1) if len(f) else np.zeros((0, 0))


def _fit(feats, wrong, idx, cols, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([_x(feats[i], cols) for i in idx])
    y = np.concatenate([[wrong[i]] * len(feats[i]) for i in idx])
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X, y)


def _score(model, f, cols):
    return model.predict_proba(_x(f, cols))[:, 1] if len(f) else np.zeros(0)


def _auroc(model, feats, wrong, idx, cols):
    s = {i: float(_score(model, feats[i], cols).max()) for i in idx}
    return auroc([s[i] for i in idx if wrong[i]], [s[i] for i in idx if not wrong[i]]), s


def fit(feats, wrong, oids, layers, sets=None):
    """feats: per trace [n_boundaries, len(layers), d]; wrong: 1 if the primary rule says wrong; layers: the layer
    numbers stored in feats; sets: {name: [layer, ...]} to evaluate (default each layer alone + all together)."""
    sets = sets or {**{str(l): [l] for l in layers}, "+".join(map(str, layers)): list(layers)}
    hold = holdout_by_oid(oids)
    tr = [i for i, o in enumerate(oids) if o not in hold and len(feats[i])]
    te = [i for i, o in enumerate(oids) if o in hold and len(feats[i])]
    if len({wrong[i] for i in tr}) < 2 or len({wrong[i] for i in te}) < 2:
        return {"name": None, "tau": float("inf"),
                "metrics": {"probe_blocker": "single-class labels in train or held-out traces", "probe_gate": False}}
    folds = group_kfold([oids[i] for i in tr], k=5)
    table, best = {}, None
    for name, ls in sets.items():
        cols = [layers.index(l) for l in ls]
        for C in C_GRID:
            cv = []
            for ftr, fva in folds:
                a_idx, v_idx = [tr[j] for j in ftr], [tr[j] for j in fva]
                if len({wrong[i] for i in a_idx}) < 2 or len({wrong[i] for i in v_idx}) < 2:
                    continue
                cv.append(_auroc(_fit(feats, wrong, a_idx, cols, C), feats, wrong, v_idx, cols)[0])
            table[f"{name}|C={C:g}"] = {"cv_auroc": float(np.mean(cv)) if cv else float("nan")}
        name_best_c = max(C_GRID, key=lambda C: np.nan_to_num(table[f"{name}|C={C:g}"]["cv_auroc"], nan=-1))
        model = _fit(feats, wrong, tr, cols, name_best_c)
        held, _ = _auroc(model, feats, wrong, te, cols)
        table[f"{name}|C={name_best_c:g}"]["heldout_auroc"] = held
        cv_best = table[f"{name}|C={name_best_c:g}"]["cv_auroc"]
        if best is None or np.nan_to_num(cv_best, nan=-1) > np.nan_to_num(best[0], nan=-1):
            best = (cv_best, name, cols, name_best_c, model)
    cv_best, name, cols, C, model = best
    held, s = _auroc(model, feats, wrong, te, cols)
    neg = [s[i] for i in te if not wrong[i]]
    pos = [s[i] for i in te if wrong[i]]
    tau = tau_at_fpr(neg)
    recall = float(np.mean(np.array(pos) >= tau))
    flags = [f for f in (first_flag(_score(model, feats[i], cols), tau) for i in te) if f]
    restart = sum(f == 1 for f in flags) / len(flags) if flags else float("nan")
    return {"name": name, "cols": cols, "model": model, "tau": tau, "metrics": {
        "probe_layers": name, "probe_C": C, "probe_cv_auroc": cv_best, "probe_auroc": held, "probe_tau": tau,
        "probe_fpr": float(np.mean(np.array(neg) >= tau)), "probe_recall": recall,
        "probe_gate": bool(held >= 0.80 and recall >= 0.40), "probe_restart_share": restart,
        "probe_first_flag_hist": dict(sorted(collections.Counter(flags).items())), "probe_table": table,
        "probe_holdout_traces": len(te), "probe_holdout_wrong": len(pos), "probe_train_traces": len(tr)}}


def flag(probe, feats):
    """First flagged boundary (1-based) for one trace, or None."""
    if probe["name"] is None:
        return None
    return first_flag(_score(probe["model"], feats, probe["cols"]), probe["tau"])
