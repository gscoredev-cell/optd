"""Detector (§9.2): standardized boundary states -> L2 logistic regression. 20% of original_ids are held out for the
gate; on the training ids, C is chosen per layer set by 5-fold CV grouped by original_id, and the layer set by its best
CV AUROC; the held-out ids only report AUROC / tau / FPR / recall.

fit():      trace labels (primary rule wrong) on every boundary; trace score = max over boundaries. tau at FPR <= 10%
            on held-out correct traces, optionally reweighted to the eval's set mix (eval_share).
fit_step(): probe-mc step labels per boundary; step AUROC, first-bad-step accuracy, implied trace AUROC."""
import collections

import numpy as np

from core import auroc, group_kfold, holdout_by_oid, set_weights, tau_at_fpr, tau_at_fpr_weighted

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def first_flag(probs, tau):
    """1-based index of the first boundary with score >= tau, else None."""
    hit = np.nonzero(np.asarray(probs) >= tau)[0]
    return int(hit[0]) + 1 if len(hit) else None


def _x(f, cols):
    """Boundary features [n_b, n_layers, d] -> [n_b, len(cols) * d] for the chosen layer columns."""
    return np.asarray(f[:, cols], dtype=np.float32).reshape(len(f), -1) if len(f) else np.zeros((0, 0))


def _fit(feats, labels, idx, cols, C):
    """labels[i] = per-boundary label array for trace i."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([_x(feats[i], cols) for i in idx])
    y = np.concatenate([labels[i] for i in idx])
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X, y)


def _score(model, f, cols):
    return model.predict_proba(_x(f, cols))[:, 1] if len(f) else np.zeros(0)


def _trace_auroc(model, feats, wrong, idx, cols):
    s = {i: float(_score(model, feats[i], cols).max()) for i in idx}
    return auroc([s[i] for i in idx if wrong[i]], [s[i] for i in idx if not wrong[i]]), s


def _step_auroc(model, feats, labels, idx, cols):
    s = {i: _score(model, feats[i], cols) for i in idx}
    y = np.concatenate([labels[i] for i in idx])
    v = np.concatenate([s[i] for i in idx])
    return auroc(v[y == 1].tolist(), v[y == 0].tolist()), s


def _select(feats, labels, oids, layers, layer_sets, c_grid, tr, score_fn, deadline=None):
    """Grouped 5-fold CV over C per layer set on the training traces; returns table and best (cv, name, cols, C, model).
    Past `deadline` (epoch s) no new (set, C) pair starts, but the first set always gets at least one C."""
    import time
    late = lambda: deadline is not None and time.time() > deadline
    folds = group_kfold([oids[i] for i in tr], k=5)
    table, best = {}, None
    for name, ls in layer_sets.items():
        if best is not None and late():
            break
        cols = [layers.index(l) for l in ls]
        for C in c_grid:
            if f"{name}|C={c_grid[0]:g}" in table and late():
                break
            cv = []
            for ftr, fva in folds:
                a_idx, v_idx = [tr[j] for j in ftr], [tr[j] for j in fva]
                ya, yv = (np.concatenate([labels[i] for i in ix]) for ix in (a_idx, v_idx))
                if len(set(ya.tolist())) < 2 or len(set(yv.tolist())) < 2:
                    continue
                cv.append(score_fn(_fit(feats, labels, a_idx, cols, C), v_idx, cols))
            table[f"{name}|C={C:g}"] = {"cv_auroc": float(np.mean(cv)) if cv else float("nan")}
        done = [C for C in c_grid if f"{name}|C={C:g}" in table]
        c_best = max(done, key=lambda C: np.nan_to_num(table[f"{name}|C={C:g}"]["cv_auroc"], nan=-1))
        cv_best = table[f"{name}|C={c_best:g}"]["cv_auroc"]
        model = _fit(feats, labels, tr, cols, c_best)
        if best is None or np.nan_to_num(cv_best, nan=-1) > np.nan_to_num(best[0], nan=-1):
            best = (cv_best, name, cols, c_best, model)
        table[f"{name}|C={c_best:g}"]["_model"] = (cols, model)
    return table, best


def _default_sets(layers):
    return {**{str(l): [l] for l in layers}, "+".join(map(str, layers)): list(layers)}


def fit(feats, wrong, oids, layers, layer_sets=None, c_grid=C_GRID, trace_sets=None, eval_share=None, holdout=None,
        deadline=None):
    """feats: per trace [n_boundaries, len(layers), d]; wrong: 1 if the primary rule says wrong; layers: the layer
    numbers stored in feats; layer_sets: {name: [layer, ...]}; trace_sets: set name per trace (main/p1/p2/rc) for
    per-set numbers; eval_share: {set: share} to reweight held-out negatives when calibrating tau."""
    layer_sets = layer_sets or _default_sets(layers)
    hold = holdout if holdout is not None else holdout_by_oid(oids)
    tr = [i for i, o in enumerate(oids) if o not in hold and len(feats[i])]
    te = [i for i, o in enumerate(oids) if o in hold and len(feats[i])]
    if len({wrong[i] for i in tr}) < 2 or len({wrong[i] for i in te}) < 2:
        return {"name": None, "tau": float("inf"),
                "metrics": {"probe_blocker": "single-class labels in train or held-out traces", "probe_gate": False}}
    labels = {i: np.full(len(feats[i]), int(wrong[i])) for i in tr + te}
    table, (cv_best, name, cols, C, model) = _select(
        feats, labels, oids, layers, layer_sets, c_grid, tr,
        lambda m, idx, cols: _trace_auroc(m, feats, wrong, idx, cols)[0], deadline)
    for k, v in table.items():   # held-out AUROC of each layer set at its best C
        if "_model" in v:
            c2, m2 = v.pop("_model")
            v["heldout_auroc"] = _trace_auroc(m2, feats, wrong, te, c2)[0]
    held, s = _trace_auroc(model, feats, wrong, te, cols)
    neg_i = [i for i in te if not wrong[i]]
    pos = [s[i] for i in te if wrong[i]]
    sets_te = [trace_sets[i] for i in neg_i] if trace_sets else None
    if eval_share and trace_sets:
        tau = tau_at_fpr_weighted([s[i] for i in neg_i], set_weights(sets_te, eval_share))
    else:
        tau = tau_at_fpr([s[i] for i in neg_i])
    recall = float(np.mean(np.array(pos) >= tau))
    flags = [f for f in (first_flag(_score(model, feats[i], cols), tau) for i in te) if f]
    restart = sum(f == 1 for f in flags) / len(flags) if flags else float("nan")
    m = {"probe_layers": name, "probe_C": C, "probe_cv_auroc": cv_best, "probe_auroc": held, "probe_tau": tau,
         "probe_fpr": float(np.mean(np.array([s[i] for i in neg_i]) >= tau)), "probe_recall": recall,
         "probe_gate": bool(held >= 0.80 and recall >= 0.40), "probe_restart_share": restart,
         "probe_first_flag_hist": dict(sorted(collections.Counter(flags).items())), "probe_table": table,
         "probe_holdout_traces": len(te), "probe_holdout_wrong": len(pos), "probe_train_traces": len(tr),
         "probe_tau_weighted": bool(eval_share and trace_sets)}
    if trace_sets:
        by = {}
        for st in sorted({trace_sets[i] for i in te}):
            idx = [i for i in te if trace_sets[i] == st]
            neg = [s[i] for i in idx if not wrong[i]]
            by[st] = {"n": len(idx), "wrong": sum(wrong[i] for i in idx),
                      "auroc": auroc([s[i] for i in idx if wrong[i]], neg),
                      "fpr": float(np.mean(np.array(neg) >= tau)) if neg else float("nan")}
        m["probe_by_set"] = by
    return {"name": name, "cols": cols, "model": model, "tau": tau, "metrics": m}


def fit_step(feats, step_y, wrong, oids, layers, layer_sets=None, c_grid=C_GRID, deadline=None):
    """probe-mc: step_y[i] = per-boundary 0/1 labels for trace i. Step AUROC on held-out boundaries, tau at boundary
    FPR <= 10%, first-bad-step-within-1 accuracy on held-out traces with a bad step, implied trace AUROC (max)."""
    layer_sets = layer_sets or _default_sets(layers)
    hold = holdout_by_oid(oids)
    tr = [i for i, o in enumerate(oids) if o not in hold and len(feats[i])]
    te = [i for i, o in enumerate(oids) if o in hold and len(feats[i])]
    labels = {i: np.asarray(step_y[i], dtype=int) for i in tr + te}
    ytr = np.concatenate([labels[i] for i in tr]) if tr else np.zeros(0)
    yte = np.concatenate([labels[i] for i in te]) if te else np.zeros(0)
    counts = {"step_pos_train": int(ytr.sum()), "step_pos_heldout": int(yte.sum()),
              "step_boundaries_train": int(len(ytr)), "step_boundaries_heldout": int(len(yte))}
    if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
        return {"name": None, "tau": float("inf"),
                "metrics": {**counts, "step_blocker": "too few positive step labels"}}
    table, (cv_best, name, cols, C, model) = _select(
        feats, labels, oids, layers, layer_sets, c_grid, tr,
        lambda m, idx, cols: _step_auroc(m, feats, labels, idx, cols)[0], deadline)
    for v in table.values():
        if "_model" in v:
            c2, m2 = v.pop("_model")
            v["heldout_auroc"] = _step_auroc(m2, feats, labels, te, c2)[0]
    step_auc, s = _step_auroc(model, feats, labels, te, cols)
    neg = np.concatenate([s[i][labels[i] == 0] for i in te]).tolist()
    tau = tau_at_fpr(neg)
    bad = [i for i in te if labels[i].any()]
    hits = 0
    for i in bad:
        true = int(np.nonzero(labels[i])[0][0]) + 1
        pred = first_flag(s[i], tau)
        hits += pred is not None and abs(pred - true) <= 1
    trace_auc = auroc([float(s[i].max()) for i in te if wrong[i]], [float(s[i].max()) for i in te if not wrong[i]])
    return {"name": name, "cols": cols, "model": model, "tau": tau, "metrics": {
        **counts, "step_layers": name, "step_C": C, "step_cv_auroc": cv_best, "step_auroc": step_auc,
        "step_tau": tau, "step_first_bad_within1": hits / len(bad) if bad else float("nan"),
        "step_heldout_traces_with_bad": len(bad), "step_trace_auroc": trace_auc, "step_table": table}}


POOLS = {"max": np.max, "mean": np.mean, "last": lambda v: v[-1]}


def pair_auroc(sw, sc):
    """Within-pair AUROC: share of pairs whose wrong trace scores above its correct twin (ties count 0.5)."""
    sw, sc = np.asarray(sw, dtype=float), np.asarray(sc, dtype=float)
    return float(np.mean(np.where(sw > sc, 1.0, np.where(sw == sc, 0.5, 0.0)))) if len(sw) else float("nan")


def pair_ci(sw, sc, n=2000, seed=0):
    """95% bootstrap CI of pair_auroc, resampling pairs."""
    sw, sc = np.asarray(sw, dtype=float), np.asarray(sc, dtype=float)
    if not len(sw):
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    bs = [pair_auroc(sw[b], sc[b]) for b in (rng.integers(0, len(sw), len(sw)) for _ in range(n))]
    return [float(x) for x in np.percentile(bs, [2.5, 97.5])]


def pairs(idx, wrong, problem):
    """[(wrong trace, correct trace)] per problem that has both among idx (first of each, in idx order)."""
    w, c = {}, {}
    for i in idx:
        (w if wrong[i] else c).setdefault(problem[i], i)
    return [(w[k], c[k]) for k in w if k in c]


def score(pr, f):
    """Pooled trace score of one trace under a fit_within probe."""
    return float(POOLS[pr["pool"]](_score(pr["model"], f, pr["cols"])))


def fit_within(feats, wrong, problem, groups, layers, layer_sets=None, c_grid=C_GRID, holdout=None, deadline=None,
               fpr=0.10):
    """Within-problem probe: traces come in (wrong, correct) pairs of the same problem. Boundary labels = trace wrong;
    layer set, C and pooling (max/mean/last) chosen by grouped 5-fold CV on the train groups maximizing within-pair
    AUROC; held-out groups report within-pair AUROC (+ bootstrap CI) by pooling and boundary position, pooled trace
    AUROC, and tau at FPR <= fpr on held-out correct traces."""
    import time
    late = lambda: deadline is not None and time.time() > deadline
    layer_sets = layer_sets or _default_sets(layers)
    hold = holdout if holdout is not None else holdout_by_oid(groups)
    tr = [i for i, g in enumerate(groups) if g not in hold and len(feats[i])]
    te = [i for i, g in enumerate(groups) if g in hold and len(feats[i])]
    p_tr, p_te = pairs(tr, wrong, problem), pairs(te, wrong, problem)
    counts = {"within_train_pairs": len(p_tr), "within_heldout_pairs": len(p_te)}
    if len(p_tr) < 5 or not p_te:
        return {"name": None, "metrics": {**counts, "within_blocker": "too few contested pairs"}}
    labels = {i: np.full(len(feats[i]), int(wrong[i])) for i in tr + te}
    folds = group_kfold([groups[i] for i in tr], k=5)
    table, best, set_best = {}, None, {}
    for name, ls in layer_sets.items():
        if best is not None and late():
            break
        cols = [layers.index(l) for l in ls]
        for C in c_grid:
            if f"{name}|C={c_grid[0]:g}" in table and late():
                break
            cv = {p: [] for p in POOLS}
            for ftr, fva in folds:
                a_idx, v_idx = [tr[j] for j in ftr], [tr[j] for j in fva]
                pv = pairs(v_idx, wrong, problem)
                if not pv or len({wrong[i] for i in a_idx}) < 2:
                    continue
                mdl = _fit(feats, labels, a_idx, cols, C)
                s = {i: _score(mdl, feats[i], cols) for p in pv for i in p}
                for pn, pool in POOLS.items():
                    cv[pn].append(pair_auroc([pool(s[a]) for a, _ in pv], [pool(s[b]) for _, b in pv]))
            row = {pn: float(np.mean(v)) if v else float("nan") for pn, v in cv.items()}
            table[f"{name}|C={C:g}"] = {"cv": row}
            top = max(row, key=lambda pn: np.nan_to_num(row[pn], nan=-1))
            if name not in set_best or np.nan_to_num(row[top], nan=-1) > set_best[name][0]:
                set_best[name] = (np.nan_to_num(row[top], nan=-1), C, cols)
            if best is None or np.nan_to_num(row[top], nan=-1) > np.nan_to_num(best[0], nan=-1):
                best = (row[top], name, cols, C, top)
    cv_best, name, cols, C, pool = best
    held_sets = {}
    for nm, (_, c2, cols2) in set_best.items():   # held-out within-pair AUROC of every layer set at its best C
        m2 = _fit(feats, labels, tr, cols2, c2)
        r2 = {i: _score(m2, feats[i], cols2) for i in te}
        held_sets[nm] = {"C": c2, **{pn: pair_auroc([pf(r2[a]) for a, _ in p_te], [pf(r2[b]) for _, b in p_te])
                                     for pn, pf in POOLS.items()}}
        if nm == name:
            model, raw = m2, r2
    pr = {"name": name, "cols": cols, "model": model, "pool": pool, "C": C}
    held = {}
    for pn, pf in POOLS.items():
        sw, sc = [pf(raw[a]) for a, _ in p_te], [pf(raw[b]) for _, b in p_te]
        held[pn] = {"auroc": pair_auroc(sw, sc), "ci": pair_ci(sw, sc)}
    by_pos = {}
    for k in range(1, 13):
        pk = [(a, b) for a, b in p_te if len(raw[a]) >= k and len(raw[b]) >= k]
        if pk:
            by_pos[k] = {"n": len(pk), "auroc": pair_auroc([raw[a][k - 1] for a, _ in pk], [raw[b][k - 1] for _, b in pk])}
    s = {i: float(POOLS[pool](raw[i])) for i in te}
    neg = [s[i] for i in te if not wrong[i]]
    tau = tau_at_fpr(neg, fpr)
    pos = [s[i] for i in te if wrong[i]]
    pr["tau"] = tau
    pr["metrics"] = {
        **counts, "within_layers": name, "within_C": C, "within_pool": pool, "within_cv_auroc": cv_best,
        "within_auroc": held[pool]["auroc"], "within_ci": held[pool]["ci"], "within_by_pool": held,
        "within_by_position": by_pos, "within_table": table, "within_by_layers": held_sets,
        "within_trace_auroc": auroc(pos, neg), "within_tau": tau,
        "within_heldout_fpr": float(np.mean(np.array(neg) >= tau)),
        "within_heldout_recall": float(np.mean(np.array(pos) >= tau))}
    return pr


def flag(probe, feats):
    """First flagged boundary (1-based) for one trace, or None."""
    if probe["name"] is None:
        return None
    return first_flag(_score(probe["model"], feats, probe["cols"]), probe["tau"])
