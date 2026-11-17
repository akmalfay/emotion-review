from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)


def clean_for_model(text: Any) -> str:
    """Match the inference-time cleaning used when the v10 model was exported."""
    text = URL_PATTERN.sub(" ", str(text))
    return " ".join(text.split())


def load_model_bundle(path: str | Path) -> dict:
    """Load a trusted, bundled model artifact created by this project."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact tidak ditemukan: {path}")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", InconsistentVersionWarning)
        bundle = joblib.load(path)

    original_versions = []
    for item in captured:
        if isinstance(item.message, InconsistentVersionWarning):
            original_versions.append(item.message.original_sklearn_version)

    # Perbedaan versi dicatat sebagai peringatan, bukan alasan untuk menghentikan aplikasi.
    # joblib/scikit-learn mungkin masih dapat memuat model, tetapi kompatibilitas lintas versi
    # tidak dijamin. Jika deserialisasi gagal, joblib.load() tetap akan melempar exception.
    compatibility_warning = None
    if original_versions:
        original = sorted({version for version in original_versions})
        loaded_from = ", ".join(original)
        compatibility_warning = (
            f"Model disimpan dengan scikit-learn {loaded_from}, sedangkan lingkungan sekarang "
            f"menggunakan scikit-learn {sklearn.__version__}. Model berhasil dimuat, tetapi "
            "kompatibilitas lintas versi tidak dijamin. Periksa hasil prediksi; untuk reproduksi "
            "yang paling konsisten, gunakan versi scikit-learn saat model dibuat."
        )

    required = {
        "vectorizer", "baseline_model", "ga_selected_model",
        "selected_feature_indices", "selected_feature_names", "class_order",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Isi model bundle tidak lengkap: {sorted(missing)}")

    if compatibility_warning:
        bundle["_version_compatibility_warning"] = compatibility_warning

    return bundle


def load_metadata(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def predict_texts(bundle: dict, texts: list[str], model_choice: str = "GA-selected") -> np.ndarray:
    cleaned = [clean_for_model(text) for text in texts]
    matrix = bundle["vectorizer"].transform(cleaned)

    if model_choice == "Baseline":
        model = bundle["baseline_model"]
        return model.predict(matrix)

    selected_matrix = matrix[:, bundle["selected_feature_indices"]]
    model = bundle["ga_selected_model"]
    return model.predict(selected_matrix)


def predict_both(bundle: dict, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cleaned = [clean_for_model(text) for text in texts]
    matrix = bundle["vectorizer"].transform(cleaned)
    baseline_predictions = bundle["baseline_model"].predict(matrix)
    selected_matrix = matrix[:, bundle["selected_feature_indices"]]
    ga_predictions = bundle["ga_selected_model"].predict(selected_matrix)
    return baseline_predictions, ga_predictions


def build_feature_subset_summary(bundle: dict) -> pd.DataFrame:
    names = np.asarray(bundle["selected_feature_names"], dtype=str)
    kinds = np.where(np.char.find(names, " ") >= 0, "Bigram", "Unigram")
    return pd.DataFrame({"feature": names, "ngram_type": kinds})
