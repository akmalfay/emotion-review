from pathlib import Path
import pandas as pd
import streamlit as st

from model_utils import load_model_bundle, load_metadata

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

st.set_page_config(
    page_title="EmoReview | GA Feature Selection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("EmoReview")
st.caption("Genetic Algorithm-based Feature Selection & Product Review Emotion Classification")

with st.sidebar:
    st.markdown("# 🧬 EmoReview")
    st.caption("PRDECT-ID · TF-IDF · Genetic Algorithm · Linear SVM")
    st.divider()
    st.markdown("**Navigasi**")
    st.page_link("app.py", label="Dashboard", icon="🏠")
    st.page_link("pages/1_GA_Optimization.py", label="Dataset & GA Optimization", icon="🧬")
    st.page_link("pages/2_Model_Evaluation.py", label="Model Evaluation", icon="📊")
    st.page_link("pages/3_Emotion_Classification.py", label="Emotion Classification", icon="💬")
    st.divider()
    st.caption("Model penelitian dimuat dari artefak final evaluation v10. Jalankan GA hanya untuk eksperimen baru.")

@st.cache_resource(show_spinner="Memuat model penelitian...")
def get_bundle():
    return load_model_bundle(ARTIFACT_DIR / "final_models.joblib")


try:
    bundle = get_bundle()
    metadata = load_metadata(ARTIFACT_DIR / "final_evaluation_metadata.json")
    if bundle.get("_version_compatibility_warning"):
        st.warning(bundle["_version_compatibility_warning"])
except Exception as exc:
    st.error(f"Model penelitian tidak dapat dimuat: {exc}")
    st.stop()

st.markdown("### Ringkasan model bawaan")
st.caption("Artefak bawaan: final evaluation v10 · PRDECT-ID · Linear SVM")
if metadata.get("test_split_previously_evaluated", False):
    st.warning(
        "Transparansi evaluasi: test split yang sama pernah diperiksa pada evaluasi v9. "
        "Hasil v10 merupakan evaluasi lanjutan, bukan pengujian pada holdout independen yang sepenuhnya baru."
    )
st.write(
    "Aplikasi menyediakan model hasil eksperimen sebagai model awal. "
    "Kamu dapat langsung mengklasifikasikan ulasan atau mengunggah dataset berlabel "
    "untuk menjalankan optimasi GA dan membangun model baru."
)

metrics = pd.read_csv(ARTIFACT_DIR / "final_test_metrics.csv")
baseline = metrics.loc[metrics["model"].str.contains("Baseline", case=False)].iloc[0]
ga_model = metrics.loc[metrics["model"].str.contains("GA-selected", case=False)].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fitur kandidat", f"{int(metadata['total_candidate_features']):,}")
c2.metric("Fitur terpilih GA", f"{int(metadata['selected_features']):,}")
c3.metric("Reduksi fitur", f"{metadata['feature_reduction_pct']:.2f}%")
c4.metric("GA test Macro-F1", f"{ga_model['macro_f1']:.4f}")

st.divider()
left, right = st.columns([1.1, 1])
with left:
    st.subheader("Alur kerja sistem")
    st.markdown(
        """
        1. **Dataset berlabel** diunggah pada halaman optimasi.
        2. **TF-IDF** membentuk kandidat unigram dan bigram.
        3. **Genetic Algorithm** mencari subset fitur menggunakan fitness berbasis validation Macro-F1 dan penalti ukuran fitur.
        4. **Linear SVM** dilatih menggunakan subset fitur terbaik.
        5. Model yang tersimpan digunakan untuk **prediksi ulasan baru** tanpa menjalankan GA kembali.
        """
    )
with right:
    st.subheader("Perbandingan model penelitian")
    compare = pd.DataFrame({
        "Model": ["Baseline", "GA-selected"],
        "Test Macro-F1": [baseline["macro_f1"], ga_model["macro_f1"]],
        "Fitur": [int(baseline["selected_features"]), int(ga_model["selected_features"])],
    })
    st.dataframe(compare, hide_index=True, use_container_width=True)
    st.info(
        f"Pada evaluasi penelitian, GA mengurangi fitur sebesar {metadata['feature_reduction_pct']:.2f}%, "
        f"dengan test Macro-F1 {ga_model['macro_f1']:.4f}; baseline memperoleh {baseline['macro_f1']:.4f}. "
        "Aplikasi menampilkan trade-off ini secara transparan."
    )

st.divider()
st.subheader("Mulai dari sini")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Ingin menjalankan algoritma genetika?**")
    st.write("Unggah dataset berlabel, atur konfigurasi, jalankan GA, lalu simpan model hasil optimasi.")
    st.page_link("pages/1_GA_Optimization.py", label="Buka GA Optimization →", icon="🧬")
with col_b:
    st.markdown("**Ingin mencoba klasifikasi?**")
    st.write("Masukkan satu ulasan atau unggah CSV. Prediksi emosi akan dibuat menggunakan model yang sudah tersedia.")
    st.page_link("pages/3_Emotion_Classification.py", label="Buka Emotion Classification →", icon="💬")

st.caption("Catatan: prediksi adalah keluaran model, bukan penetapan emosi yang pasti. Kinerja di luar domain PRDECT-ID belum divalidasi.")
