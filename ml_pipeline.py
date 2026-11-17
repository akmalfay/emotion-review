from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from model_utils import clean_for_model

CLASS_ORDER = ["Anger", "Fear", "Happy", "Love", "Sadness"]
DUPLICATE_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)


@dataclass
class DatasetAudit:
    cleaned_data: pd.DataFrame
    summary: dict
    class_counts: pd.DataFrame
    removed_conflicts: pd.DataFrame
    duplicate_groups: int


def normalize_duplicate_text(text: str) -> str:
    """Duplicate key: casefold, remove URL, replace punctuation with spaces, normalize whitespace."""
    text = DUPLICATE_URL_PATTERN.sub(" ", str(text).casefold())
    characters = []
    for char in text:
        if unicodedata.category(char).startswith("P"):
            characters.append(" ")
        else:
            characters.append(char)
    return " ".join("".join(characters).split())


def normalize_label(label: str) -> str:
    normalized = str(label).strip().casefold()
    mapping = {name.casefold(): name for name in CLASS_ORDER}
    return mapping.get(normalized, str(label).strip())


def audit_and_clean_dataset(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> DatasetAudit:
    if text_column == label_column:
        raise ValueError("Kolom teks dan label harus berbeda.")
    if text_column not in dataframe.columns or label_column not in dataframe.columns:
        raise ValueError("Kolom teks atau label tidak ditemukan dalam CSV.")

    data = dataframe[[text_column, label_column]].copy()
    data.columns = ["review", "emotion"]
    data["review"] = data["review"].fillna("").astype(str)
    data["emotion"] = data["emotion"].map(normalize_label)
    data["duplicate_key"] = data["review"].map(normalize_duplicate_text)

    empty_mask = data["duplicate_key"].eq("")
    empty_rows = int(empty_mask.sum())
    data = data.loc[~empty_mask].copy()

    unknown_labels = sorted(set(data["emotion"]) - set(CLASS_ORDER))
    if unknown_labels:
        raise ValueError(
            "Label di luar lima kelas yang didukung ditemukan: "
            + ", ".join(map(str, unknown_labels))
        )

    grouped = data.groupby("duplicate_key", sort=False)["emotion"].nunique()
    conflict_keys = set(grouped[grouped > 1].index)
    conflict_rows = data[data["duplicate_key"].isin(conflict_keys)].copy()

    before_dedup = len(data)
    no_conflicts = data.loc[~data["duplicate_key"].isin(conflict_keys)].copy()
    cleaned = no_conflicts.drop_duplicates(subset="duplicate_key", keep="first").copy()
    cleaned.reset_index(drop=True, inplace=True)

    class_counts = (
        cleaned["emotion"].value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .rename_axis("emotion")
        .reset_index(name="rows")
    )
    missing_classes = class_counts.loc[class_counts["rows"] == 0, "emotion"].tolist()
    if missing_classes:
        raise ValueError(f"Dataset tidak memiliki sampel untuk kelas: {missing_classes}")
    if class_counts["rows"].min() < 10:
        raise ValueError(
            "Setiap kelas membutuhkan sedikitnya 10 baris agar stratified split 70:15:15 dapat dilakukan dengan aman."
        )

    remaining_duplicates = int(cleaned["duplicate_key"].duplicated().sum())
    summary = {
        "input_rows": int(len(dataframe)),
        "empty_rows_removed": empty_rows,
        "rows_before_duplicate_handling": int(before_dedup),
        "conflict_groups_removed": int(len(conflict_keys)),
        "conflict_rows_removed": int(len(conflict_rows)),
        "same_label_duplicate_rows_removed": int(before_dedup - len(conflict_rows) - len(cleaned)),
        "final_rows": int(len(cleaned)),
        "remaining_duplicate_keys": remaining_duplicates,
    }

    return DatasetAudit(
        cleaned_data=cleaned[["review", "emotion"]].copy(),
        summary=summary,
        class_counts=class_counts,
        removed_conflicts=conflict_rows[["review", "emotion"]].copy(),
        duplicate_groups=int(grouped.shape[0]),
    )


def split_dataset(texts: pd.Series, labels: pd.Series, random_state: int = 42):
    x_train, x_temp, y_train, y_temp = train_test_split(
        texts,
        labels,
        test_size=0.30,
        random_state=random_state,
        stratify=labels,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def mutation_probability(setting: str, chromosome_length: int) -> float:
    multipliers = {"1/L": 1, "10/L": 10, "100/L": 100}
    if setting not in multipliers:
        raise ValueError("Mutation harus salah satu dari 1/L, 10/L, atau 100/L.")
    return multipliers[setting] / chromosome_length


def initialize_population(population_size: int, feature_count: int, rng: np.random.Generator):
    densities = [0.10, 0.25, 0.50, 0.75]
    per_density, remainder = divmod(population_size, len(densities))
    schedule = []
    for index, density in enumerate(densities):
        schedule.extend([density] * (per_density + (1 if index < remainder else 0)))
    rng.shuffle(schedule)

    population = []
    for density in schedule:
        count = max(1, round(density * feature_count))
        selected = rng.choice(feature_count, size=count, replace=False)
        chromosome = np.zeros(feature_count, dtype=bool)
        chromosome[selected] = True
        population.append(chromosome)
    return population


def evaluate_chromosome(
    chromosome: np.ndarray,
    x_train,
    y_train: pd.Series,
    x_validation,
    y_validation: pd.Series,
    cache: dict,
    svm_c: float,
    lambda_size: float,
):
    cache_key = np.packbits(chromosome).tobytes()
    if cache_key in cache:
        return cache[cache_key]

    selected_indices = np.flatnonzero(chromosome)
    if len(selected_indices) == 0:
        raise ValueError("Kromosom kosong tidak boleh dievaluasi.")

    classifier = LinearSVC(C=svm_c, max_iter=5000, random_state=42)
    classifier.fit(x_train[:, selected_indices], y_train)
    predictions = classifier.predict(x_validation[:, selected_indices])
    macro_f1 = f1_score(y_validation, predictions, average="macro", zero_division=0)
    size_penalty = lambda_size * (len(selected_indices) / chromosome.size)
    result = {
        "fitness": float(macro_f1 - size_penalty),
        "macro_f1": float(macro_f1),
        "size_penalty": float(size_penalty),
        "selected_features": int(len(selected_indices)),
    }
    cache[cache_key] = result
    return result


def tournament_select(population, fitness_values, rng, tournament_size: int = 3):
    candidate_indices = rng.choice(len(population), size=tournament_size, replace=False)
    winner_index = int(candidate_indices[np.argmax(fitness_values[candidate_indices])])
    return population[winner_index]


def uniform_crossover(parent1, parent2, rng, probability: float):
    if rng.random() >= probability:
        return parent1.copy(), parent2.copy(), False
    swap_mask = rng.random(parent1.size) < 0.5
    child1 = np.where(swap_mask, parent1, parent2)
    child2 = np.where(swap_mask, parent2, parent1)
    return child1.astype(bool), child2.astype(bool), True


def bit_flip_mutation(chromosome, rng, mutation_rate: float):
    child = chromosome.copy()
    flip_mask = rng.random(child.size) < mutation_rate
    original_values = child[flip_mask]
    bits_1_to_0 = int(np.count_nonzero(original_values))
    bits_0_to_1 = int(len(original_values) - bits_1_to_0)
    child[flip_mask] = ~child[flip_mask]
    repair_applied = False
    if not child.any():
        child[int(rng.integers(0, child.size))] = True
        repair_applied = True
    details = {
        "bits_flipped": int(flip_mask.sum()),
        "bits_1_to_0": bits_1_to_0,
        "bits_0_to_1": bits_0_to_1,
        "repair_applied": repair_applied,
        "features_after_mutation": int(child.sum()),
    }
    return child, details


def run_genetic_algorithm(
    x_train,
    y_train,
    x_validation,
    y_validation,
    *,
    population_size: int = 40,
    generations: int = 30,
    crossover_probability: float = 0.9,
    mutation_setting: str = "100/L",
    seed: int = 42,
    tournament_size: int = 3,
    elitism: int = 1,
    lambda_size: float = 0.02,
    patience: int = 5,
    min_delta: float = 0.001,
    svm_c: float = 1.0,
    progress_callback: Callable | None = None,
):
    feature_count = x_train.shape[1]
    if population_size < tournament_size:
        raise ValueError("Population size harus lebih besar atau sama dengan tournament size.")
    if elitism != 1:
        raise ValueError("Versi aplikasi ini menggunakan elitism = 1 sesuai rancangan eksperimen.")

    rng = np.random.default_rng(seed)
    mutation_rate = mutation_probability(mutation_setting, feature_count)
    if mutation_rate > 1.0:
        raise ValueError(
            f"Mutation {mutation_setting} menghasilkan probabilitas {mutation_rate:.4f} (> 1) "
            "karena jumlah fitur terlalu sedikit. Pilih level mutation yang lebih rendah."
        )
    population = initialize_population(population_size, feature_count, rng)
    cache = {}
    history = []
    operator_history = []
    best_chromosome = None
    best_metrics = None
    best_fitness = -np.inf
    stopping_reference = None
    stale_generations = 0
    svm_evaluations = 0
    cache_hits = 0

    for generation in range(1, generations + 1):
        generation_start = time.perf_counter()
        evaluated = []
        for chromosome in population:
            before = len(cache)
            result = evaluate_chromosome(
                chromosome, x_train, y_train, x_validation, y_validation,
                cache, svm_c, lambda_size,
            )
            if len(cache) > before:
                svm_evaluations += 1
            else:
                cache_hits += 1
            evaluated.append(result)

        fitness_values = np.array([item["fitness"] for item in evaluated])
        feature_counts = np.array([item["selected_features"] for item in evaluated])
        best_index = int(np.argmax(fitness_values))
        generation_best = float(fitness_values[best_index])

        if generation_best > best_fitness:
            best_fitness = generation_best
            best_chromosome = population[best_index].copy()
            best_metrics = evaluated[best_index].copy()

        if stopping_reference is None:
            stopping_reference = generation_best
            stale_generations = 0
        elif generation_best > stopping_reference + min_delta:
            stopping_reference = generation_best
            stale_generations = 0
        else:
            stale_generations += 1

        unique_count = len({np.packbits(item).tobytes() for item in population})
        row = {
            "generation": generation,
            "best_generation_fitness": generation_best,
            "global_best_fitness": best_fitness,
            "mean_fitness": float(np.mean(fitness_values)),
            "best_generation_macro_f1": float(evaluated[best_index]["macro_f1"]),
            "global_best_macro_f1": float(best_metrics["macro_f1"]),
            "best_generation_features": int(feature_counts[best_index]),
            "global_best_features": int(best_metrics["selected_features"]),
            "mean_features": float(np.mean(feature_counts)),
            "unique_chromosomes": int(unique_count),
            "mutation_setting": mutation_setting,
            "expected_mutation_flips": float(feature_count * mutation_rate),
            "stale_generations": int(stale_generations),
            "svm_evaluations_total": int(svm_evaluations),
            "cache_hits_total": int(cache_hits),
            "generation_seconds": float(time.perf_counter() - generation_start),
        }
        history.append(row)
        if progress_callback:
            progress_callback(generation, generations, row)

        if generation == generations or stale_generations >= patience:
            break

        elite = best_chromosome.copy()
        next_population = [elite]
        selection_calls = 0
        crossover_pairs = 0
        crossovers_applied = 0
        children_mutated = 0
        total_bit_flips = 0
        total_bits_1_to_0 = 0
        total_bits_0_to_1 = 0
        repair_count = 0
        parent_feature_counts = []
        offspring_features_before = []
        offspring_features_after = []

        while len(next_population) < population_size:
            parent1 = tournament_select(population, fitness_values, rng, tournament_size)
            parent2 = tournament_select(population, fitness_values, rng, tournament_size)
            selection_calls += 2
            parent_feature_counts.extend([int(parent1.sum()), int(parent2.sum())])

            child1, child2, crossover_applied = uniform_crossover(
                parent1, parent2, rng, crossover_probability
            )
            crossover_pairs += 1
            crossovers_applied += int(crossover_applied)

            for child in (child1, child2):
                if len(next_population) >= population_size:
                    break
                offspring_features_before.append(int(child.sum()))
                mutated, mutation_details = bit_flip_mutation(child, rng, mutation_rate)
                next_population.append(mutated)
                offspring_features_after.append(mutation_details["features_after_mutation"])
                children_mutated += int(mutation_details["bits_flipped"] > 0)
                total_bit_flips += mutation_details["bits_flipped"]
                total_bits_1_to_0 += mutation_details["bits_1_to_0"]
                total_bits_0_to_1 += mutation_details["bits_0_to_1"]
                repair_count += int(mutation_details["repair_applied"])

        operator_history.append({
            "generation": generation,
            "selection_calls": selection_calls,
            "mean_selected_parent_features": float(np.mean(parent_feature_counts)) if parent_feature_counts else np.nan,
            "crossover_pairs": crossover_pairs,
            "crossovers_applied": crossovers_applied,
            "crossover_applied_rate": crossovers_applied / crossover_pairs if crossover_pairs else 0.0,
            "offspring_count": len(next_population) - 1,
            "offspring_with_mutation": children_mutated,
            "total_bit_flips": total_bit_flips,
            "bits_1_to_0": total_bits_1_to_0,
            "bits_0_to_1": total_bits_0_to_1,
            "mean_features_before_mutation": float(np.mean(offspring_features_before)) if offspring_features_before else np.nan,
            "mean_features_after_mutation": float(np.mean(offspring_features_after)) if offspring_features_after else np.nan,
            "empty_chromosome_repairs": repair_count,
            "elite_features_carried": int(elite.sum()),
            "elite_fitness_carried": float(best_fitness),
        })
        population = next_population

    return {
        "best_chromosome": best_chromosome,
        "selected_indices": np.flatnonzero(best_chromosome),
        "best_metrics": best_metrics,
        "history": pd.DataFrame(history),
        "operator_history": pd.DataFrame(operator_history),
        "generations_completed": len(history),
        "svm_evaluations": svm_evaluations,
        "cache_hits": cache_hits,
        "mutation_probability": mutation_rate,
        "seed": seed,
    }


def train_and_evaluate_dataset(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
    *,
    population_size: int = 40,
    generations: int = 30,
    crossover_probability: float = 0.9,
    mutation_setting: str = "100/L",
    seed: int = 42,
    lambda_size: float = 0.02,
    progress_callback: Callable | None = None,
):
    audit = audit_and_clean_dataset(dataframe, text_column, label_column)
    clean_data = audit.cleaned_data
    texts = clean_data["review"].astype(str).map(clean_for_model)
    labels = clean_data["emotion"].astype(str)

    x_train_text, x_validation_text, x_test_text, y_train, y_validation, y_test = split_dataset(
        texts, labels, random_state=42
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(x_train_text)
    if x_train.shape[1] == 0:
        raise ValueError("TF-IDF tidak menghasilkan fitur dari teks training.")
    x_validation = vectorizer.transform(x_validation_text)
    x_test = vectorizer.transform(x_test_text)

    ga_result = run_genetic_algorithm(
        x_train,
        y_train,
        x_validation,
        y_validation,
        population_size=population_size,
        generations=generations,
        crossover_probability=crossover_probability,
        mutation_setting=mutation_setting,
        seed=seed,
        lambda_size=lambda_size,
        progress_callback=progress_callback,
    )

    selected_indices = ga_result["selected_indices"]
    selected_count = len(selected_indices)
    feature_count = x_train.shape[1]

    # Final refit: keep the exact candidate vocabulary so chromosome indices stay valid.
    final_vectorizer = TfidfVectorizer(
        vocabulary=vectorizer.vocabulary_,
        ngram_range=(1, 2),
        min_df=1,
        dtype=np.float32,
    )
    x_train_validation_text = pd.concat([x_train_text, x_validation_text])
    y_train_validation = pd.concat([y_train, y_validation])
    x_train_validation = final_vectorizer.fit_transform(x_train_validation_text)
    x_test_final = final_vectorizer.transform(x_test_text)

    baseline_model = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    baseline_model.fit(x_train_validation, y_train_validation)

    ga_model = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    ga_model.fit(x_train_validation[:, selected_indices], y_train_validation)

    baseline_predictions = baseline_model.predict(x_test_final)
    ga_predictions = ga_model.predict(x_test_final[:, selected_indices])

    metrics = []
    reports = {}
    matrices = {}
    for model_name, predictions in [
        ("Baseline Linear SVM", baseline_predictions),
        ("GA-selected Linear SVM", ga_predictions),
    ]:
        metrics.append({
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "macro_precision": precision_score(y_test, predictions, average="macro", zero_division=0),
            "macro_recall": recall_score(y_test, predictions, average="macro", zero_division=0),
            "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "selected_features": feature_count if model_name.startswith("Baseline") else selected_count,
            "feature_reduction_pct": 0.0 if model_name.startswith("Baseline") else 100 * (1 - selected_count / feature_count),
        })
        reports[model_name] = pd.DataFrame(
            classification_report(y_test, predictions, labels=CLASS_ORDER, output_dict=True, zero_division=0)
        ).T
        matrices[model_name] = pd.DataFrame(
            confusion_matrix(y_test, predictions, labels=CLASS_ORDER),
            index=CLASS_ORDER,
            columns=CLASS_ORDER,
        )

    comparison = pd.DataFrame(metrics)
    predictions_frame = pd.DataFrame({
        "review": x_test_text.reset_index(drop=True),
        "true_emotion": y_test.reset_index(drop=True),
        "baseline_prediction": baseline_predictions,
        "ga_prediction": ga_predictions,
    })
    selected_names = final_vectorizer.get_feature_names_out()[selected_indices]

    model_bundle = {
        "vectorizer": final_vectorizer,
        "baseline_model": baseline_model,
        "ga_selected_model": ga_model,
        "selected_feature_indices": selected_indices,
        "selected_feature_names": selected_names,
        "class_order": CLASS_ORDER,
        "text_cleaning": "Remove URLs and normalize whitespace; vectorizer performs lowercase tokenization.",
        "final_configuration": {
            "population_size": population_size,
            "generations_limit": generations,
            "crossover_probability": crossover_probability,
            "mutation": mutation_setting,
            "seed": seed,
            "lambda_size": lambda_size,
        },
    }

    return {
        "audit": audit,
        "ga_result": ga_result,
        "comparison": comparison,
        "reports": reports,
        "confusion_matrices": matrices,
        "predictions": predictions_frame,
        "selected_features": pd.DataFrame({"feature": selected_names}),
        "model_bundle": model_bundle,
        "feature_count": feature_count,
        "selected_count": selected_count,
        "reduction_pct": 100 * (1 - selected_count / feature_count),
        "split_sizes": {
            "train": len(y_train),
            "validation": len(y_validation),
            "test": len(y_test),
        },
    }
