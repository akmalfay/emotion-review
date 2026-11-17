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

try:
    default_bundle = load_default_bundle()
    if default_bundle.get("_version_compatibility_warning"):
        st.warning(default_bundle["_version_compatibility_warning"])
except Exception as exc:
    st.error(f"Model bawaan tidak dapat dimuat: {exc}")
    st.stop()

# Model bawaan v10 selalu tersedia. Hasil GA terbaru hanya menjadi pilihan tambahan
# dan tidak menimpa atau menghapus artefak model bawaan.
if "latest_training_result" in st.session_state:
    model_source = st.radio(
        "Sumber model",
        ["Model penelitian (PRDECT-ID · final v10)", "Model hasil GA terbaru (dataset yang diunggah)"],
        horizontal=True,
        key="classification_model_source",
    )
else:
    model_source = "Model penelitian (PRDECT-ID · final v10)"
    st.caption("Menggunakan model bawaan dari final evaluation v10.")

if model_source == "Model hasil GA terbaru (dataset yang diunggah)" and "latest_training_result" in st.session_state:
    bundle = st.session_state["latest_training_result"]["model_bundle"]
    st.success("Menggunakan bundle dari run GA terbaru. Baseline pada bundle ini juga dilatih ulang dari dataset yang diunggah.")
    config = st.session_state.get("latest_training_config", {})
    if config:
        st.caption(
            f"Konfigurasi run: population={config['population_size']}, "
            f"crossover={config['crossover_probability']}, mutation={config['mutation_setting']}, "
            f"seed={config['seed']}, λ={config['lambda_size']:.3f}."
        )
else:
    bundle = default_bundle
    st.success("Menggunakan bundle model penelitian PRDECT-ID final v10.")

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

    if uploaded_csv is None:
        # Jangan tampilkan hasil CSV sebelumnya ketika file sudah dihapus.
        st.session_state.pop("latest_batch_output", None)
        st.session_state.pop("latest_batch_source_signature", None)
        st.session_state.pop("latest_batch_file_digest", None)
        st.info("Upload CSV untuk memulai klasifikasi batch.")
    else:
        import hashlib
        from io import BytesIO

        csv_bytes = uploaded_csv.getvalue()
        file_digest = hashlib.sha256(csv_bytes).hexdigest()

        # File baru membatalkan hasil klasifikasi dari file sebelumnya.
        previous_digest = st.session_state.get("latest_batch_file_digest")
        if previous_digest != file_digest:
            st.session_state.pop("latest_batch_output", None)
            st.session_state.pop("latest_batch_source_signature", None)
            st.session_state["latest_batch_file_digest"] = file_digest

        try:
            batch_data = pd.read_csv(BytesIO(csv_bytes))
        except Exception as exc:
            st.session_state.pop("latest_batch_output", None)
            st.session_state.pop("latest_batch_source_signature", None)
            st.error(f"CSV tidak dapat dibaca: {exc}")
            st.stop()

        if batch_data.empty:
            st.session_state.pop("latest_batch_output", None)
            st.session_state.pop("latest_batch_source_signature", None)
            st.warning("CSV tidak memiliki baris data.")
            st.stop()

        text_columns = batch_data.select_dtypes(include=["object", "string"]).columns.tolist()
        if not text_columns:
            st.session_state.pop("latest_batch_output", None)
            st.session_state.pop("latest_batch_source_signature", None)
            st.error("Tidak ditemukan kolom teks pada CSV.")
            st.stop()

        default_index = text_columns.index("Customer Review") if "Customer Review" in text_columns else 0
        selected_text_column = st.selectbox(
            "Pilih kolom ulasan",
            text_columns,
            index=default_index,
            # Key berbasis file agar pilihan kolom tidak terbawa ke CSV yang berbeda.
            key=f"batch_text_column_{file_digest[:12]}",
        )
        source_signature = (file_digest, selected_text_column)

        # Pergantian kolom ulasan juga membatalkan hasil sebelumnya.
        previous_signature = st.session_state.get("latest_batch_source_signature")
        if previous_signature is not None and previous_signature != source_signature:
            st.session_state.pop("latest_batch_output", None)
            st.session_state.pop("latest_batch_source_signature", None)
            st.info("File atau kolom ulasan berubah. Jalankan klasifikasi batch kembali.")

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
            # Simpan kedua prediksi model. Kolom model terpilih dibuat saat render,
            # sehingga selalu mengikuti pilihan radio yang sedang aktif.
            stored_output = batch_data.copy()
            stored_output["Baseline_Prediction"] = baseline_predictions
            stored_output["GA_Selected_Prediction"] = ga_predictions
            st.session_state["latest_batch_output"] = stored_output
            st.session_state["latest_batch_source_signature"] = source_signature

        stored_output = st.session_state.get("latest_batch_output")
        stored_signature = st.session_state.get("latest_batch_source_signature")

        if stored_output is not None and stored_signature == source_signature:
            # Selalu turunkan prediksi terpilih dari pilihan model terkini.
            output = stored_output.copy()
            selected_prediction_column = (
                "GA_Selected_Prediction" if model_choice == "GA-selected" else "Baseline_Prediction"
            )
            output["Selected_Model_Prediction"] = output[selected_prediction_column]

            st.markdown("### Hasil klasifikasi")
            st.dataframe(output.head(100), use_container_width=True, hide_index=True)
            classified_rows = output[output["Selected_Model_Prediction"] != "Empty review"]
            counts = (
                classified_rows["Selected_Model_Prediction"]
                .value_counts()
                .rename_axis("Emotion")
                .reset_index(name="Reviews")
            )
            st.markdown("### Distribusi emosi hasil prediksi")
            st.bar_chart(counts.set_index("Emotion"))
            st.download_button(
                "⬇ Download hasil klasifikasi CSV",
                data=output.to_csv(index=False).encode("utf-8-sig"),
                file_name="emoreview_classification_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Belum ada hasil untuk input ini. Tekan **Classify all reviews** untuk menjalankan klasifikasi.")

st.divider()
st.caption("Catatan domain: model bawaan dilatih pada ulasan produk PRDECT-ID. Kinerja pada domain lain belum divalidasi.")
