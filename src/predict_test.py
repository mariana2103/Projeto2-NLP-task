"""For person 3: load the final S1 model saved by person 2 and write results.txt.

  python predict_test.py --test path/to/test_no_labels.csv --model path/to/s1_final_model.joblib --out results.txt

Defaults work from the project folder (StudentsPack/, Parte2_resultados/) and
from the delivery folder Entrega_Parte2/ (resultados/).

The model already contains person 1's text preparation (keyword-label
stripping etc.), so the raw test file is given to it directly.
"""
import argparse
import sys
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # the pickled pipeline references parte1_base / s1

from parte1_base import load_test, specialties  # noqa: E402

ROOT = HERE.parent


def first_existing(*paths):
    return next((p for p in paths if p.exists()), paths[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, default=first_existing(
        ROOT / "StudentsPack" / "test_no_labels.csv", ROOT / "test_no_labels.csv"))
    ap.add_argument("--model", type=Path, default=first_existing(
        ROOT / "Parte2_resultados" / "s1_final_model.joblib", ROOT / "resultados" / "s1_final_model.joblib"))
    ap.add_argument("--out", type=Path, default=ROOT / "results.txt")
    args = ap.parse_args()

    test = load_test(args.test)
    model = joblib.load(args.model)
    pred = model.predict(test)

    assert len(pred) == len(test) == 398, "one label per test case"
    assert set(pred) <= set(specialties), "only valid specialty names"
    out = args.out
    out.write_text("\n".join(pred) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(pred)} lines, {len(set(pred))} distinct specialties")


if __name__ == "__main__":
    main()
