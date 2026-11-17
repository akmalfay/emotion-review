from pathlib import Path
import json

import pandas as pd
import streamlit as st

from model_utils import load_metadata

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts"

st.set_page_config(page_title="Model Evaluation | EmoReview", page_icon="📊", layout="wide")
st.title("📊 Model Evaluation")
st.caption("Hasil final evaluation v10 pada PRDECT-ID.")

if "latest_training_result" in st.session_state:
    result = st.session_state["latest_training_result"]
    st.info("Menampilkan evaluasi dari optimasi yang baru dijalankan pada sesi aplikasi ini.")
    metrics = result["comparison"].copy()
    reports = result["reports"]
    matrices = result["confusion_matrices"]
    selected_count = result["selected_count"]
    feature_count = result["feature_count"]
    reduction = result["reduction_pct"]
else:
    metrics = pd.read_csv(ARTIFACT_DIR / "final_test_metrics.csv")
    reports = {
        "Baseline Linear SVM": pd.read_csv(ARTIFACT_DIR / "baseline_classification_report.csv", index_col=0),
        "GA-selected Linear SVM": pd.read_csv(ARTIFACT_DIR / "ga_classification_report.csv", index_col=0),
    }
    matrices = {
        "Baseline Linear SVM": pd.read_csv(ARTIFACT_DIR / "baseline_confusion_matrix.csv", index_col=0),
        "GA-selected Linear SVM": pd.read_csv(ARTIFACT_DIR / "ga_confusion_matrix.csv", index_col=0),
    }
    selected_count = int(metrics.loc[metrics["model"].str.contains("GA-selected", case=False), "selected_features"].iloc[0])
    feature_count = int(metrics.loc[metrics["model"].str.contains("Baseline", case=False), "selected_features"].iloc[0])
    reduction = 100 * (1 - selected_count / feature_count)
    st.caption("Angka bawaan berasal dari final evaluation v10 pada PRDECT-ID.")
    metadata = load_metadata(ARTIFACT_DIR / "final_evaluation_metadata.json")
    if metadata.get("test_split_previously_evaluated", False):
        st.warning(
            "Test split yang sama telah diperiksa pada evaluasi v9. Karena itu, hasil ini merupakan "
            "evaluasi lanjutan dan bukan holdout independen yang sepenuhnya baru. "
            "Pemilihan konfigurasi dan fitur GA tidak menggunakan test set."
        )

baseline = metrics.loc[metrics["model"].str.contains("Baseline", case=False)].iloc[0]
ga = metrics.loc[metrics["model"].str.contains("GA-selected", case=False)].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline Macro-F1", f"{baseline['macro_f1']:.4f}")
c2.metric("GA Macro-F1", f"{ga['macro_f1']:.4f}", f"{(ga['macro_f1']-baseline['macro_f1'])*100:+.2f} pp")
c3.metric("Fitur terpilih", f"{selected_count:,}")
c4.metric("Reduksi fitur", f"{reduction:.2f}%")

st.markdown("### Perbandingan metrik")
metric_columns = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
comparison = metrics.set_index("model")[metric_columns].T
st.bar_chart(comparison, height=300)
st.dataframe(metrics[["model", *metric_columns, "selected_features", "feature_reduction_pct"]].style.format({
    "accuracy": "{:.4f}",
    "macro_precision": "{:.4f}",
    "macro_recall": "{:.4f}",
    "macro_f1": "{:.4f}",
    "weighted_f1": "{:.4f}",
    "feature_reduction_pct": "{:.2f}%",
}), use_container_width=True, hide_index=True)

st.markdown("### Confusion matrix")
selected_model = st.radio("Pilih model", list(matrices.keys()), horizontal=True)
st.dataframe(matrices[selected_model], use_container_width=True)

st.markdown("### Classification report")
st.dataframe(reports[selected_model], use_container_width=True)

st.info(
    "Interpretasi: reduksi fitur dan performa klasifikasi adalah dua tujuan yang perlu dibaca bersama. "
    "Model GA tidak otomatis lebih baik hanya karena memakai lebih sedikit fitur. Bandingkan Macro-F1, "
    "metrik per kelas, dan ukuran subset sebelum memilih model untuk digunakan."
)
