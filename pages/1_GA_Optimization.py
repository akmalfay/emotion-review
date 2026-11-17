from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from ml_pipeline import CLASS_ORDER, audit_and_clean_dataset, train_and_evaluate_dataset

st.set_page_config(page_title="GA Optimization | EmoReview", page_icon="🧬", layout="wide")
st.title("🧬 Dataset & GA Optimization")
st.caption("Unggah dataset berlabel, jalankan Genetic Algorithm, dan bangun model klasifikasi dari subset fitur terpilih.")

st.info(
    "Halaman ini menjalankan satu konfigurasi GA, bukan mengulang screening 27 konfigurasi di Google Colab. "
    "Model bawaan penelitian tetap tersedia pada halaman klasifikasi."
)

uploaded_file = st.file_uploader("Upload dataset CSV berlabel", type=["csv"], key="ga_dataset_upload")
if uploaded_file is None:
    st.markdown("#### Format dataset")
    st.code("Customer Review,Emotion\n\"Barang bagus dan sesuai\",Happy\n\"Barang rusak dan mengecewakan\",Sadness", language="csv")
    st.caption("Lima label yang didukung: Anger, Fear, Happy, Love, dan Sadness.")
    st.stop()

try:
    uploaded_file.seek(0)
    dataframe = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"CSV tidak dapat dibaca: {exc}")
    st.stop()

if dataframe.empty:
    st.error("CSV tidak memiliki baris data.")
    st.stop()

st.success(f"CSV berhasil dimuat: {len(dataframe):,} baris, {len(dataframe.columns)} kolom.")
st.dataframe(dataframe.head(10), use_container_width=True, hide_index=True)

columns = dataframe.columns.tolist()
default_text_index = columns.index("Customer Review") if "Customer Review" in columns else 0
default_label_index = columns.index("Emotion") if "Emotion" in columns else min(1, len(columns) - 1)
col_a, col_b = st.columns(2)
with col_a:
    text_column = st.selectbox("Kolom teks ulasan", columns, index=default_text_index)
with col_b:
    label_column = st.selectbox("Kolom label emosi", columns, index=default_label_index)

if text_column == label_column:
    st.warning("Pilih kolom teks dan label yang berbeda.")
    st.stop()

try:
    audit = audit_and_clean_dataset(dataframe, text_column, label_column)
except Exception as exc:
    st.error(f"Validasi dataset gagal: {exc}")
    st.stop()

st.markdown("### Ringkasan audit dataset")
summary = audit.summary
metrics = st.columns(4)
metrics[0].metric("Baris awal", f"{summary['input_rows']:,}")
metrics[1].metric("Baris bersih", f"{summary['final_rows']:,}")
metrics[2].metric("Konflik label", f"{summary['conflict_groups_removed']:,} grup")
metrics[3].metric("Duplikasi tersisa", f"{summary['remaining_duplicate_keys']:,}")
st.dataframe(audit.class_counts, use_container_width=True, hide_index=True)

with st.expander("Lihat aturan pembersihan duplikasi"):
    st.markdown(
        """
        1. Mengubah teks menjadi lowercase dengan `casefold()`.
        2. Menghapus URL.
        3. Mengganti tanda baca Unicode dengan spasi.
        4. Merapikan whitespace.
        5. Menghapus seluruh kelompok teks yang memiliki label berbeda.
        6. Untuk teks duplikat dengan label sama, mempertahankan satu baris.

        Normalisasi ini digunakan untuk membuat duplicate key. Teks ulasan yang dipakai oleh TF-IDF tetap berasal dari kolom ulasan, dengan URL dan whitespace dibersihkan saat pemrosesan model.
        """
    )

st.divider()
st.markdown("### Konfigurasi Genetic Algorithm")
st.caption("Preset awal menggunakan konfigurasi terbaik dari eksperimen penelitian. Parameter dapat disesuaikan untuk eksperimen baru.")

with st.expander("Pengaturan optimasi", expanded=True):
    row1 = st.columns(3)
    population_size = row1[0].selectbox("Population size", [20, 30, 40], index=2)
    generations = row1[1].number_input("Maksimum generasi", min_value=5, max_value=100, value=30, step=5)
    seed = row1[2].number_input("Random seed", min_value=0, max_value=999999, value=2, step=1)
    row2 = st.columns(3)
    crossover_probability = row2[0].select_slider("Crossover probability", options=[0.7, 0.8, 0.9], value=0.9)
    mutation_setting = row2[1].selectbox("Mutation rate", ["1/L", "10/L", "100/L"], index=2)
    lambda_size = row2[2].number_input("Feature penalty (λ)", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.3f")
    st.caption("Preset parameter mengikuti konfigurasi terbaik confirmation v10. Seed default 2 adalah seed kromosom final terpilih; menjalankan ulang GA tetap merupakan eksperimen baru, bukan reproduksi identik dari seluruh proses confirmation. Split seed pada eksperimen penelitian adalah 42.")
    st.caption("Tetap digunakan: Tournament k=3, Uniform Crossover, Bit-flip Mutation, Elitism=1, patience=5, dan min_delta=0.001.")

start_clicked = st.button("▶ Run Genetic Algorithm", type="primary", use_container_width=True)

if start_clicked:
    progress_bar = st.progress(0, text="Menyiapkan dataset dan TF-IDF...")
    latest_status = st.empty()
    history_chart = st.empty()
    progress_rows = []

    def update_progress(current_generation, total_generations, row):
        progress_bar.progress(
            min(current_generation / total_generations, 1.0),
            text=f"Generasi {current_generation}/{total_generations}",
        )
        latest_status.markdown(
            f"**Best fitness:** {row['global_best_fitness']:.6f} &nbsp; · &nbsp; "
            f"**Macro-F1:** {row['global_best_macro_f1']:.6f} &nbsp; · &nbsp; "
            f"**Fitur:** {row['global_best_features']:,} &nbsp; · &nbsp; "
            f"**SVM evaluations:** {row['svm_evaluations_total']:,}"
        )
        progress_rows.append(row.copy())
        chart_data = pd.DataFrame(progress_rows).set_index("generation")[[
            "global_best_fitness", "mean_fitness"
        ]]
        history_chart.line_chart(chart_data, height=240)

    try:
        with st.status("Optimasi sedang berjalan. Proses ini dapat memerlukan waktu.", expanded=True) as status:
            result = train_and_evaluate_dataset(
                dataframe,
                text_column,
                label_column,
                population_size=int(population_size),
                generations=int(generations),
                crossover_probability=float(crossover_probability),
                mutation_setting=mutation_setting,
                seed=int(seed),
                lambda_size=float(lambda_size),
                progress_callback=update_progress,
            )
            status.update(label="Optimasi selesai", state="complete", expanded=False)

        st.session_state["latest_training_result"] = result
        st.session_state["latest_training_config"] = {
            "population_size": int(population_size),
            "generations": int(generations),
            "crossover_probability": float(crossover_probability),
            "mutation_setting": mutation_setting,
            "seed": int(seed),
            "lambda_size": float(lambda_size),
        }
        progress_bar.progress(1.0, text="Selesai")
        st.success("GA selesai. Model baru siap digunakan pada halaman Emotion Classification.")
    except Exception as exc:
        st.error(f"Optimasi gagal: {exc}")
        st.stop()

if "latest_training_result" in st.session_state:
    result = st.session_state["latest_training_result"]
    st.divider()
    st.markdown("## Hasil optimasi terbaru")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Generasi selesai", result["ga_result"]["generations_completed"])
    col2.metric("Evaluasi SVM unik", f"{result['ga_result']['svm_evaluations']:,}")
    col3.metric("Fitur terpilih", f"{result['selected_count']:,}")
    col4.metric("Reduksi fitur", f"{result['reduction_pct']:.2f}%")

    st.markdown("### Perbandingan test set")
    st.dataframe(result["comparison"].style.format({
        "accuracy": "{:.4f}",
        "macro_precision": "{:.4f}",
        "macro_recall": "{:.4f}",
        "macro_f1": "{:.4f}",
        "weighted_f1": "{:.4f}",
        "feature_reduction_pct": "{:.2f}%",
    }), use_container_width=True, hide_index=True)

    st.markdown("### Riwayat generasi")
    st.dataframe(result["ga_result"]["history"], use_container_width=True, hide_index=True)

    st.markdown("### Aktivitas operator GA")
    st.caption("Log ini mencatat proses pembentukan generasi berikutnya: hasil tournament selection, penerapan crossover, perubahan bit akibat mutation, dan elite yang dibawa.")
    operator_history = result["ga_result"]["operator_history"]
    if not operator_history.empty:
        st.dataframe(operator_history, use_container_width=True, hide_index=True)
        operator_chart = operator_history.set_index("generation")[[
            "crossovers_applied", "offspring_with_mutation", "total_bit_flips"
        ]]
        st.line_chart(operator_chart, height=240)

    st.markdown("### Fitur terpilih")
    st.write(f"Model baru memilih {result['selected_count']:,} dari {result['feature_count']:,} fitur kandidat.")
    st.dataframe(result["selected_features"].head(30), use_container_width=True, hide_index=True)

    model_buffer = BytesIO()
    joblib.dump(result["model_bundle"], model_buffer, compress=3)
    model_buffer.seek(0)
    st.download_button(
        "⬇ Download model hasil optimasi (.joblib)",
        data=model_buffer.getvalue(),
        file_name="emoreview_ga_model.joblib",
        mime="application/octet-stream",
        use_container_width=True,
    )
    st.download_button(
        "⬇ Download hasil prediksi test (CSV)",
        data=result["predictions"].to_csv(index=False).encode("utf-8-sig"),
        file_name="emoreview_test_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "⬇ Download riwayat generasi (CSV)",
        data=result["ga_result"]["history"].to_csv(index=False).encode("utf-8-sig"),
        file_name="ga_generation_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "⬇ Download log operator GA (CSV)",
        data=result["ga_result"]["operator_history"].to_csv(index=False).encode("utf-8-sig"),
        file_name="ga_operator_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
