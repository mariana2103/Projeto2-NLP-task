"""Part 2 (person 2): main model S1 + data augmentation.

Everything is evaluated on person 1's 5 group-aware folds (parte1_base.splits),
with person 1's text preparation, class weights and metrics, so that the
numbers are directly comparable with Part 1.
"""
import json
import random
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from parte1_base import (class_weights, csv_fields, data, measure, prepare_texts, priority,
                         save_evaluation, specialties, splits)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Parte2_resultados"
OUT.mkdir(exist_ok=True)
BT_CACHE = ROOT / "data" / "backtranslation_cache.json"


# ------------------------------------------------------------------ S1 model
def word_vectorizer(field, ngrams=2):
    # same settings as person 1's create_model
    return TfidfVectorizer(lowercase=True, ngram_range=(1, ngrams), sublinear_tf=True,
                           min_df=2 if field == "transcription" else 1,
                           max_features=30000 if field == "transcription" else 10000,
                           token_pattern=r"(?u)\b\w+(?:[./'-]\w+)*\b", dtype=np.float64)


def char_vectorizer(field):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                           max_features=30000 if field == "transcription" else 13000, dtype=np.float64)


def default_weights(representation="both", char_weight=0.5, transcription_char=False):
    """Block weights: {field_word / field_char: weight}. 0 removes the block."""
    w = {}
    for f in csv_fields:
        w[f + "_word"] = 1.0 if representation in ("word", "both") else 0.0
        if representation == "char":
            w[f + "_char"] = 1.0
        elif representation == "both":
            w[f + "_char"] = char_weight if (f != "transcription" or transcription_char) else 0.0
        else:
            w[f + "_char"] = 0.0
    return w


def create_s1_model(weights=None, classifier="lr", c=1.0, keywords="strip", sections=False, ngrams=2):
    weights = weights or default_weights()
    blocks = []
    if keywords == "none":  # empty field: drop its blocks (as person 1 does)
        weights = {n: (0.0 if n.startswith("keywords_") else w) for n, w in weights.items()}
    for name, wt in weights.items():
        if wt <= 0:
            continue
        field, kind = name.rsplit("_", 1)
        vec = word_vectorizer(field, ngrams) if kind == "word" else char_vectorizer(field)
        blocks.append((name, vec, field))
    tfidf = ColumnTransformer(blocks, transformer_weights={n: w for n, w in weights.items() if w > 0},
                              sparse_threshold=1.0, remainder="drop", verbose_feature_names_out=False)
    if classifier == "svm":
        algorithm = LinearSVC(C=c, dual="auto", max_iter=20000, random_state=42)
    else:
        algorithm = LogisticRegression(C=c, solver="lbfgs", max_iter=3000, random_state=42)
    return Pipeline([
        ("cleaning", FunctionTransformer(prepare_texts, kw_args={"keywords": keywords, "sections": sections})),
        ("tfidf", tfidf), ("classifier", algorithm)])


# ------------------------------------------------------------------ augmentation
SECTION_RE = re.compile(r",(?=[A-Z][A-Z /&()\-]{2,}:)")
SMALL_CLASS = 100  # classes with fewer training cases than this are "small"


def crop_sections(text, rng, keep=0.6):
    parts = SECTION_RE.split(text)
    if len(parts) < 3:
        return text
    n = max(2, int(len(parts) * keep))
    start = rng.randint(0, len(parts) - n)
    return ",".join(parts[start:start + n])


class BackTranslator:
    """EN->DE->EN with OPUS-MT, sentence by sentence, cached on disk."""

    def __init__(self):
        self.cache = json.loads(BT_CACHE.read_text()) if BT_CACHE.exists() else {}
        self.models = None

    def _load(self):
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.models = {}
        for pair in [("en", "de"), ("de", "en")]:
            name = f"Helsinki-NLP/opus-mt-{pair[0]}-{pair[1]}"
            self.models[pair] = (MarianTokenizer.from_pretrained(name),
                                 MarianMTModel.from_pretrained(name).to(self.device).eval())

    def _translate(self, texts, pair, bs=32):
        import torch
        tok, model = self.models[pair]
        out = []
        for i in range(0, len(texts), bs):
            batch = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                gen = model.generate(**batch, num_beams=1, do_sample=False, max_new_tokens=256)
            out += tok.batch_decode(gen, skip_special_tokens=True)
        return out

    def __call__(self, texts):
        todo = sorted({t for t in texts if t and t not in self.cache})
        if todo:
            if self.models is None:
                self._load()
            sents, owners = [], []
            for t in todo:
                for s in re.split(r"(?<=[.!?])\s+|,(?=[A-Z])", t):
                    if s.strip():
                        sents.append(s.strip()); owners.append(t)
            back = self._translate(self._translate(sents, ("en", "de")), ("de", "en"))
            joined = {}
            for o, b in zip(owners, back):
                joined.setdefault(o, []).append(b)
            for t in todo:
                self.cache[t] = " ".join(joined.get(t, [t]))
            BT_CACHE.parent.mkdir(exist_ok=True)
            BT_CACHE.write_text(json.dumps(self.cache))
        return [self.cache.get(t, t) for t in texts]


_bt = None


def back_translate_rows(rows):
    global _bt
    _bt = _bt or BackTranslator()
    out = rows.copy()
    for col in ["description", "sample_name", "transcription"]:
        out[col] = _bt(rows[col].tolist())
    return out


def augment(train, mode, seed=42):
    """Return (augmented train, log of the generated variants).

    none   : original training data
    target : small classes only, 1 back-translated + 1 section-cropped variant per case
    all    : every class, 1 section-cropped variant per case with >=3 sections and
             1 keyword-dropout variant for 30% of the cases with keywords
    copy   : control, exact copies of small-class cases, as many as 'target' adds
    Variants keep the group_id of their source, so they never leak across folds.
    """
    if mode == "none":
        return train, pd.DataFrame()
    rng = random.Random(seed)
    counts = train["medical_specialty"].value_counts()
    small = train[train["medical_specialty"].isin(counts[counts < SMALL_CLASS].index)]
    variants = []
    if mode in ("target", "copy"):
        bt = back_translate_rows(small).assign(augmentation="backtranslation")
        crop = small.assign(transcription=[crop_sections(t, rng) for t in small["transcription"]],
                            augmentation="section_crop")
        crop = crop[crop["transcription"] != small.loc[crop.index, "transcription"]]
        if mode == "target":
            variants = [bt, crop]
        else:
            n = len(bt) + len(crop)
            variants = [small.sample(n=n, replace=True, random_state=seed).assign(augmentation="exact_copy")]
    elif mode == "all":
        crop = train.assign(transcription=[crop_sections(t, rng) for t in train["transcription"]],
                            augmentation="section_crop")
        crop = crop[crop["transcription"] != train["transcription"]]
        has_kw = train[train["keywords"].str.strip() != ""]
        kwd = has_kw.sample(frac=0.3, random_state=seed).assign(keywords="", augmentation="keyword_dropout")
        variants = [crop, kwd]
    new = pd.concat(variants)
    log = new.assign(source_row_id=new["row_id"])
    new = new.assign(row_id=-1)  # variants are not real cases
    return pd.concat([train.assign(augmentation="original"), new], ignore_index=True), log


# ------------------------------------------------------------------ evaluation
def _load_cached(name, config):
    """Reuse a finished evaluation (Parte2_resultados/<name>/summary.json) if its config matches."""
    folder = OUT / name
    if not (folder / "summary.json").exists():
        return None
    cfg_file = folder / "config.json"
    if cfg_file.exists():
        if json.loads(cfg_file.read_text()) != json.loads(json.dumps(config)):
            return None
    elif config.get("augmentation", "none") != "none":
        return None
    summary = json.loads((folder / "summary.json").read_text())
    summary = {k: v for k, v in summary.items() if not isinstance(v, dict)}
    summary.setdefault("augmented_cases_per_fold", [0] * len(splits))
    return summary


def evaluate_s1(name, config, verbose=False, use_cache=True):
    """5-fold evaluation on person 1's splits; results saved in Parte2_resultados/<name>/.
    A configuration already evaluated with the same settings is read from disk."""
    if use_cache and (cached := _load_cached(name, config)) is not None:
        return cached
    cfg = dict(config)
    q = cfg.pop("q", 1)
    aug = cfg.pop("augmentation", "none")
    predictions = np.empty(len(data), dtype=object)
    folds = np.full(len(data), -1)
    times, aug_counts = [], []
    for split in splits:
        start = perf_counter()
        train = data[data["row_id"].isin(split["train"])]
        validation = data[data["row_id"].isin(split["evaluation"])]
        assert not set(train["group_id"]) & set(validation["group_id"])
        train, log = augment(train, aug, seed=split["fold"])
        aug_counts.append(len(log))
        model = create_s1_model(**cfg)
        model.fit(train, train["medical_specialty"],
                  classifier__sample_weight=class_weights(train["medical_specialty"].tolist(), q))
        predictions[validation.index] = model.predict(validation)
        folds[validation.index] = split["fold"]
        times.append(perf_counter() - start)
        if verbose:
            print(name, split["fold"] + 1, "/", len(splits), flush=True)
    summary = save_evaluation(OUT / name, data, predictions, folds, times)
    summary["augmented_cases_per_fold"] = aug_counts
    (OUT / name / "config.json").write_text(json.dumps(config, indent=1))
    full = json.loads((OUT / name / "summary.json").read_text())
    full["augmented_cases_per_fold"] = aug_counts
    (OUT / name / "summary.json").write_text(json.dumps(full, indent=2))
    return summary


def run(configs, filename):
    """Evaluate several configurations and save a comparison table."""
    rows = {}
    for name, cfg in configs.items():
        rows[name] = evaluate_s1(name, cfg)
        r = rows[name]
        print(f"{name:42s} acc={r['accuracy']:.3f} macroF1={r['macro_f1']:.3f} minF1={r['min_f1']:.3f} "
              f"({r['seconds']:.0f}s)", flush=True)
    cols = ["accuracy", "macro_f1", "min_f1", "meets_thresholds", "folds_meeting_thresholds", "seconds"]
    table = pd.DataFrame(rows).T
    if any(r["augmented_cases_per_fold"] != [0] * len(splits) for r in rows.values()):
        cols.append("augmented_cases_per_fold")
    table = table[cols]
    table.to_csv(OUT / filename)
    return table


def best_of(table):
    """Person 1's selection criterion: thresholds first, then macro-F1, then accuracy."""
    keys = {n: priority({"meets_thresholds": r.meets_thresholds, "accuracy": r.accuracy,
                         "min_f1": r.min_f1, "macro_f1": r.macro_f1}) for n, r in table.iterrows()}
    return max(keys, key=keys.get)


# ------------------------------------------------------------------ field weights
def field_weight_search(base_config, grid=(0.0, 0.5, 1.0, 2.0), rounds=2):
    """Coordinate ascent over the block weights, judged with person 1's priority()
    on the 5 folds. Returns (best weights, log of every accepted step).

    For speed, every TF-IDF block is fitted once per fold; a weight only rescales
    its block, which is exactly what ColumnTransformer's transformer_weights does,
    so each candidate only needs the classifier to be re-trained."""
    import scipy.sparse as sp
    from sklearn.base import clone

    cfg = dict(base_config)
    weights = dict(cfg.pop("weights"))
    q = cfg.pop("q", 1)
    template = create_s1_model(weights={b: 1.0 for b in weights}, **cfg)
    clean, clf = template.named_steps["cleaning"], template.named_steps["classifier"]

    fold_data = []
    for split in splits:
        train = data[data["row_id"].isin(split["train"])]
        validation = data[data["row_id"].isin(split["evaluation"])]
        t_tr, t_va = clean.transform(train), clean.transform(validation)
        blocks = {}
        for name in weights:
            field, kind = name.rsplit("_", 1)
            if cfg.get("keywords") == "none" and field == "keywords":
                continue
            vec = word_vectorizer(field, cfg.get("ngrams", 2)) if kind == "word" else char_vectorizer(field)
            blocks[name] = (vec.fit_transform(t_tr[field]), vec.transform(t_va[field]))
        sw = class_weights(train["medical_specialty"].tolist(), q)
        fold_data.append((train["medical_specialty"].to_numpy(), validation.index, blocks, sw))

    def score(w):
        pred = np.empty(len(data), dtype=object)
        for y, val_index, blocks, sw in fold_data:
            names = [b for b in blocks if w[b] > 0]
            X_tr = sp.hstack([blocks[b][0] * w[b] for b in names]).tocsr()
            X_va = sp.hstack([blocks[b][1] * w[b] for b in names]).tocsr()
            pred[val_index] = clone(clf).fit(X_tr, y, sample_weight=sw).predict(X_va)
        return measure(data["medical_specialty"], pred)

    best = score(weights)
    steps = [{"step": "start", **{k: best[k] for k in ("accuracy", "macro_f1", "min_f1")}, **weights}]
    print(f"start: acc={best['accuracy']:.3f} macroF1={best['macro_f1']:.3f} minF1={best['min_f1']:.3f}", flush=True)
    for rnd in range(rounds):
        changed = False
        for block in list(weights):
            for w in grid:
                if w == weights[block]:
                    continue
                cand = {**weights, block: w}
                if not any(v > 0 for b, v in cand.items() if any(b == n for n in fold_data[0][2])):
                    continue
                s = score(cand)
                if priority(s) > priority(best):
                    best, weights, changed = s, cand, True
                    steps.append({"step": f"round{rnd}:{block}={w}", **{k: s[k] for k in ("accuracy", "macro_f1", "min_f1")}, **weights})
                    print(f"round {rnd} {block}={w}: acc={s['accuracy']:.3f} macroF1={s['macro_f1']:.3f} minF1={s['min_f1']:.3f}", flush=True)
        if not changed:
            break
    pd.DataFrame(steps).to_csv(OUT / "field_weight_search.csv", index=False)
    return weights, pd.DataFrame(steps)
