from pathlib import Path

import pandas as pd
import streamlit as st

from model_utils import load_model_bundle, predict_both, predict_texts

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts"

st.set_page_config(page_title="Emotion Classification | EmoReview", page_icon="💬", layout="wide")
st.title("💬 Emotion Classification")
st.caption("Klasifikasikan satu ulasan atau banyak ulasan secara otomatis menggunakan model yang sudah dilatih.")

@st.cache_resource(show_spinner="Memuat model bawaan...")
def load_default_bundle():
    return load_model_bundle(ARTIFACT_DIR / "final_models.joblib")

if "latest_training_result" in st.session_state:
    bundle = st.session_state["latest_training_result"]["model_bundle"]
    st.success("Menggunakan model hasil optimasi terbaru pada sesi ini.")
else:
    try:
        bundle = load_default_bundle()
        st.caption("Menggunakan model bawaan dari hasil penelitian v9.")
    except Exception as exc:
        st.error(f"Model tidak dapat dimuat: {exc}")
        st.stop()

model_choice = st.radio(
    "Model klasifikasi",
    ["GA-selected", "Baseline"],
    horizontal=True,
    help="GA-selected menggunakan subset fitur pilihan Genetic Algorithm. Baseline menggunakan seluruh fitur TF-IDF.",
)

single_tab, batch_tab = st.tabs(["Single review", "Batch CSV"])
with single_tab:
    review = st.text_area(
        "Masukkan ulasan produk",
        placeholder="Contoh: Barangnya bagus, pengiriman cepat dan sesuai ekspektasi.",
        height=130,
    )
    if st.button("Analyze review", type="primary", disabled=not review.strip(), key="single_predict"):
        prediction = predict_texts(bundle, [review], model_choice=model_choice)[0]
        st.success(f"Predicted emotion: **{prediction}**")
        baseline_pred, ga_pred = predict_both(bundle, [review])
        col1, col2 = st.columns(2)
        col1.metric("Baseline prediction", str(baseline_pred[0]))
        col2.metric("GA-selected prediction", str(ga_pred[0]))
        st.caption("Prediksi adalah keluaran model, bukan penetapan emosi yang pasti.")

with batch_tab:
    uploaded_csv = st.file_uploader("Upload CSV yang berisi ulasan", type=["csv"], key="review_batch_upload")
    if uploaded_csv is not None:
        try:
            uploaded_csv.seek(0)
            batch_data = pd.read_csv(uploaded_csv)
        except Exception as exc:
            st.error(f"CSV tidak dapat dibaca: {exc}")
            st.stop()

        if batch_data.empty:
            st.warning("CSV tidak memiliki baris data.")
            st.stop()
        text_columns = batch_data.select_dtypes(include=["object", "string"]).columns.tolist()
        if not text_columns:
            st.error("Tidak ditemukan kolom teks pada CSV.")
            st.stop()
        default_index = text_columns.index("Customer Review") if "Customer Review" in text_columns else 0
        selected_text_column = st.selectbox("Pilih kolom ulasan", text_columns, index=default_index)
        st.write(f"Jumlah ulasan: **{len(batch_data):,}**")
        st.dataframe(batch_data.head(8), use_container_width=True, hide_index=True)

        if st.button("Classify all reviews", type="primary", key="batch_predict"):
            texts = batch_data[selected_text_column].fillna("").astype(str).tolist()
            progress = st.progress(0, text="Memeriksa ulasan...")
            valid_positions = [index for index, text in enumerate(texts) if text.strip()]
            baseline_predictions = ["Empty review"] * len(texts)
            ga_predictions = ["Empty review"] * len(texts)

            if valid_positions:
                valid_texts = [texts[index] for index in valid_positions]
                progress.progress(0.35, text="Mengubah teks menjadi TF-IDF...")
                valid_baseline, valid_ga = predict_both(bundle, valid_texts)
                for position, baseline_label, ga_label in zip(valid_positions, valid_baseline, valid_ga):
                    baseline_predictions[position] = baseline_label
                    ga_predictions[position] = ga_label

            progress.progress(1.0, text="Klasifikasi selesai")
            output = batch_data.copy()
            output["Baseline_Prediction"] = baseline_predictions
            output["GA_Selected_Prediction"] = ga_predictions
            output["Selected_Model_Prediction"] = ga_predictions if model_choice == "GA-selected" else baseline_predictions
            st.session_state["latest_batch_output"] = output

    if "latest_batch_output" in st.session_state:
        output = st.session_state["latest_batch_output"]
        st.markdown("### Hasil klasifikasi")
        st.dataframe(output.head(100), use_container_width=True, hide_index=True)
        classified_rows = output[output["Selected_Model_Prediction"] != "Empty review"]
        counts = classified_rows["Selected_Model_Prediction"].value_counts().rename_axis("Emotion").reset_index(name="Reviews")
        st.markdown("### Distribusi emosi hasil prediksi")
        st.bar_chart(counts.set_index("Emotion"))
        st.download_button(
            "⬇ Download hasil klasifikasi CSV",
            data=output.to_csv(index=False).encode("utf-8-sig"),
            file_name="emoreview_classification_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.caption("Catatan domain: model bawaan dilatih pada ulasan produk PRDECT-ID. Kinerja pada domain lain belum divalidasi.")
