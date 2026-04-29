# -*- coding: utf-8 -*-
"""
Version finale .py basée sur ton notebook original.

Objectif :
1) Garder la même logique et les mêmes datasets :
   - Données synthétiques avec make_blobs
   - Iris Dataset
2) Comparer principalement :
   - KMeans normal sur données originales
   - Autoencoder + KMeans++
3) Ajouter des analyses enrichies :
   - Variation du nombre de clusters K
   - Variation de la dimension latente
   - Variation du nombre de features
   - Variation du nombre d'échantillons
   - Analyse temporelle / temps d'exécution
   - Graphiques de performance

Remarque :
- Le code conserve ton autoencoder PyTorch et ta logique de pipeline.
- Les nouvelles fonctions sont ajoutées après les anciennes, sans changer le principe de base.
"""

# ============================================================
# 0. Imports
# ============================================================

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, make_blobs


# ============================================================
# 1. Device
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisant device: {DEVICE}")


# ============================================================
# 2. Autoencoder original
# ============================================================

class Autoencoder(nn.Module):
    """Autoencoder pour compression et extraction de l'espace latent."""

    def __init__(self, input_dim: int, latent_dim: int = 3):
        super(Autoencoder, self).__init__()

        # ENCODEUR : réduit la dimension
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            # GOULOT D'ÉTRANGLEMENT
            nn.Linear(32, latent_dim)
        )

        # DÉCODEUR : reconstruit les données
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        """Retourne les points dans l'espace latent."""
        return self.encoder(x)

    def decode(self, z):
        """Reconstruit à partir de l'espace latent."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass complet."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


# ============================================================
# 3. Entraînement Autoencoder
# ============================================================

def train_autoencoder(
    X: np.ndarray,
    latent_dim: int = 3,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: bool = True
):
    """
    Entraîne l'autoencoder sur les données numériques.

    Retourne :
    - model : Autoencoder entraîné
    - scaler : StandardScaler utilisé pour l'entraînement
    - losses : liste des losses par epoch
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)

    model = Autoencoder(X.shape[1], latent_dim).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if verbose:
        print("\nEntraînement autoencoder:")
        print(f"     Input: {X.shape[1]} features → Latent: {latent_dim} dims")

    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_tensor))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_tensor), batch_size):
            batch_idx = indices[i:i + batch_size]
            X_batch = X_tensor[batch_idx]

            X_reconstructed, _ = model(X_batch)
            loss = criterion(X_reconstructed, X_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        losses.append(epoch_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"     Epoch {epoch + 1:3d}/{epochs}: Loss = {epoch_loss:.6f}")

    model.eval()
    return model, scaler, losses


def encode_with_autoencoder(X: np.ndarray, model: Autoencoder, scaler: StandardScaler):
    """Encode X avec le même scaler et retourne Z."""
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)

    with torch.no_grad():
        Z = model.encode(X_tensor).cpu().numpy()

    return Z


# ============================================================
# 4. Autoencoder + KMeans++ original
# ============================================================

def predict_centroids_with_autoencoder(
    X: np.ndarray,
    n_clusters: int,
    model: Autoencoder,
    scaler,
    latent_dim: int = 3
):
    """
    Prédire les centroïdes en utilisant l'espace latent.

    Processus :
    1. Standardiser X avec le même scaler
    2. Encoder X → Z
    3. KMeans++ sur Z
    4. Retourner les centroïdes dans l'espace latent
    """

    Z = encode_with_autoencoder(X, model, scaler)

    print("\n[v0] Espace latent:")
    print(f"     Forme: {Z.shape}")
    print(f"     Densité: {Z.shape[0]} points dans {latent_dim}D")

    kmeans_latent = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    kmeans_latent.fit(Z)
    centroid_latent = kmeans_latent.cluster_centers_

    print("\n[v0] Centroïdes trouvés dans l'espace latent:")
    print(f"     Forme: {centroid_latent.shape}")
    print(f"     Inertie latent: {kmeans_latent.inertia_:.2f}")

    return centroid_latent, Z


# ============================================================
# 5. Comparaison originale du notebook
# ============================================================

def compare_methods(X: np.ndarray, n_clusters: int):
    """
    Version originale enrichie légèrement pour stocker les labels.

    Compare :
    1. KMeans classique random
    2. KMeans++
    3. Autoencoder + KMeans++
    """

    print("\n" + "=" * 80)
    print("COMPARAISON: Random vs KMeans++ vs Autoencoder+KMeans++")
    print("=" * 80)

    results = {}

    # === 1. KMeans Random ===
    print("\n[1] KMeans Random:")
    t0 = time.time()

    kmeans_random = KMeans(
        n_clusters=n_clusters,
        init="random",
        n_init=1,
        random_state=42
    )

    labels_random = kmeans_random.fit_predict(X)
    time_random = time.time() - t0

    inertia_random = kmeans_random.inertia_
    sil_random = silhouette_score(X, labels_random)
    dbi_random = davies_bouldin_score(X, labels_random)

    results["Random"] = {
        "inertia": inertia_random,
        "silhouette": sil_random,
        "davies_bouldin": dbi_random,
        "calinski_harabasz": calinski_harabasz_score(X, labels_random),
        "time": time_random,
        "centroids": kmeans_random.cluster_centers_,
        "labels": labels_random
    }

    print(f"     Inertia: {inertia_random:.2f}")
    print(f"     Silhouette: {sil_random:.4f}")
    print(f"     Davies-Bouldin: {dbi_random:.4f}")
    print(f"     Time: {time_random:.4f}s")

    # === 2. KMeans++ ===
    print("\n[2] KMeans++:")
    t0 = time.time()

    kmeans_pp = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels_pp = kmeans_pp.fit_predict(X)
    time_pp = time.time() - t0

    inertia_pp = kmeans_pp.inertia_
    sil_pp = silhouette_score(X, labels_pp)
    dbi_pp = davies_bouldin_score(X, labels_pp)

    results["K-means++"] = {
        "inertia": inertia_pp,
        "silhouette": sil_pp,
        "davies_bouldin": dbi_pp,
        "calinski_harabasz": calinski_harabasz_score(X, labels_pp),
        "time": time_pp,
        "centroids": kmeans_pp.cluster_centers_,
        "labels": labels_pp
    }

    print(f"     Inertia: {inertia_pp:.2f}")
    print(f"     Silhouette: {sil_pp:.4f}")
    print(f"     Davies-Bouldin: {dbi_pp:.4f}")
    print(f"     Time: {time_pp:.4f}s")

    # === 3. Autoencoder + KMeans++ ===
    print("\n[3] Autoencoder + KMeans++:")

    latent_dim = min(3, max(1, X.shape[1] - 1))

    t0 = time.time()
    model, scaler, losses = train_autoencoder(X, latent_dim=latent_dim, epochs=100)
    ae_time = time.time() - t0

    centroid_latent, Z = predict_centroids_with_autoencoder(
        X,
        n_clusters,
        model,
        scaler,
        latent_dim
    )

    t0 = time.time()

    kmeans_ae = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels_ae = kmeans_ae.fit_predict(Z)
    km_ae_time = time.time() - t0

    inertia_ae = np.sum(np.min(cdist(Z, kmeans_ae.cluster_centers_) ** 2, axis=1))
    sil_ae = silhouette_score(Z, labels_ae)
    dbi_ae = davies_bouldin_score(Z, labels_ae)

    results["Autoencoder"] = {
        "inertia": inertia_ae,
        "silhouette": sil_ae,
        "davies_bouldin": dbi_ae,
        "calinski_harabasz": calinski_harabasz_score(Z, labels_ae),
        "time": ae_time + km_ae_time,
        "autoencoder_time": ae_time,
        "kmeans_time": km_ae_time,
        "centroids": kmeans_ae.cluster_centers_,
        "labels": labels_ae,
        "model": model,
        "scaler": scaler,
        "losses": losses,
        "Z": Z
    }

    print(f"     Inertia: {inertia_ae:.2f}")
    print(f"     Silhouette: {sil_ae:.4f}")
    print(f"     Davies-Bouldin: {dbi_ae:.4f}")
    print(f"     Autoencoder time: {ae_time:.4f}s")
    print(f"     KMeans++ time: {km_ae_time:.4f}s")
    print(f"     Total time: {ae_time + km_ae_time:.4f}s")

    # === Résumé ===
    print("\n" + "=" * 80)
    print("RÉSUMÉ:")
    print("=" * 80)

    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  Inertia:          {metrics['inertia']:8.2f}")
        print(f"  Silhouette:       {metrics['silhouette']:8.4f}")
        print(f"  Davies-Bouldin:   {metrics['davies_bouldin']:8.4f}")
        print(f"  Calinski-Harabasz:{metrics['calinski_harabasz']:8.2f}")
        print(f"  Time:             {metrics['time']:8.4f}s")

    return results, Z


# ============================================================
# 6. Visualisation originale
# ============================================================

def visualize_results(
    X: np.ndarray,
    results: dict,
    Z: np.ndarray = None,
    save_path: str = "autoencoder_comparison.png"
):
    """Créer des visualisations comparatives entre plusieurs méthodes de clustering."""

    if X.ndim != 2:
        raise ValueError("X doit être un tableau 2D de forme (n_samples, n_features).")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparaison: KMeans vs KMeans++ vs Autoencoder", fontsize=16)

    methods = ["Random", "K-means++", "Autoencoder"]
    colors = ["red", "blue", "green"]

    for ax, method, color in zip(axes.flat[:3], methods, colors):
        if method not in results:
            ax.set_visible(False)
            continue

        centroids = results[method]["centroids"]
        labels = results[method].get("labels", None)
        inertia = results[method].get("inertia", None)

        if method == "Autoencoder":
            if Z is None:
                ax.set_visible(False)
                continue
            data = Z
        else:
            data = X

        if data.shape[1] != centroids.shape[1]:
            ax.set_title(f"{method}\nDimensions incompatibles")
            ax.axis("off")
            continue

        if data.shape[1] == 2:
            data_2d = data
            centroids_2d = centroids
            xlabel = "Dim 1"
            ylabel = "Dim 2"
        else:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
            centroids_2d = pca.transform(centroids)
            xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
            ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"

        if labels is not None:
            ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, s=20)
        else:
            ax.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5, s=20)

        ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c=color,
            marker="*",
            s=500,
            edgecolors="black",
            linewidth=2
        )

        title = method
        if inertia is not None:
            title += f"\nInertia: {inertia:.2f}"

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # 4e subplot : courbe de loss Autoencoder
    ax = axes.flat[3]

    if "Autoencoder" in results and "losses" in results["Autoencoder"]:
        losses = results["Autoencoder"]["losses"]
        ax.plot(losses)
        ax.set_title("Convergence Autoencoder")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Loss non disponible", ha="center", va="center")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nGraphique sauvegardé : {save_path}")
    plt.show()


# ============================================================
# 7. Nouvelle comparaison demandée :
#    KMeans normal original vs Autoencoder + KMeans++
# ============================================================

def compare_kmeans_normal_vs_autoencoder_kmeanspp(
    X: np.ndarray,
    dataset_name: str,
    k_values=range(2, 11),
    latent_dim: int = 3,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: bool = False
):
    """
    Compare uniquement :
    1. KMeans normal sur les données originales
    2. Autoencoder + KMeans++ sur l'espace latent

    Cette fonction garde la logique du notebook :
    - KMeans normal = init random
    - Autoencoder PyTorch
    - KMeans++ appliqué sur Z latent
    """

    rows = []

    # =========================
    # 1. KMeans normal original
    # =========================
    for k in k_values:
        start = time.time()

        kmeans = KMeans(
            n_clusters=k,
            init="random",
            n_init=1,
            random_state=42
        )

        labels = kmeans.fit_predict(X)
        exec_time = time.time() - start

        rows.append({
            "dataset": dataset_name,
            "method": "KMeans normal",
            "K": k,
            "latent_dim": np.nan,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "inertia": kmeans.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "kmeans_time": exec_time,
            "autoencoder_time": 0.0,
            "total_time": exec_time
        })

    # =========================
    # 2. Autoencoder + KMeans++
    # =========================
    start_ae = time.time()
    model, scaler, losses = train_autoencoder(
        X,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    ae_time = time.time() - start_ae

    Z = encode_with_autoencoder(X, model, scaler)

    for k in k_values:
        start = time.time()

        kmeans_pp = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=42
        )

        labels = kmeans_pp.fit_predict(Z)
        kmeans_time = time.time() - start

        rows.append({
            "dataset": dataset_name,
            "method": "Autoencoder + KMeans++",
            "K": k,
            "latent_dim": latent_dim,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "inertia": kmeans_pp.inertia_,
            "silhouette": silhouette_score(Z, labels),
            "davies_bouldin": davies_bouldin_score(Z, labels),
            "calinski_harabasz": calinski_harabasz_score(Z, labels),
            "kmeans_time": kmeans_time,
            "autoencoder_time": ae_time,
            "total_time": ae_time + kmeans_time
        })

    return pd.DataFrame(rows), Z, losses


# ============================================================
# 8. Étude selon la dimension de l'espace latent
# ============================================================

def study_latent_dimensions(
    X: np.ndarray,
    dataset_name: str,
    latent_dims=(2, 3, 4, 5, 8, 10),
    k_values=range(2, 11),
    epochs: int = 100,
    batch_size: int = 32
):
    """
    Étudie Autoencoder + KMeans++ selon la dimension latente.
    """

    rows = []

    for latent_dim in latent_dims:
        if latent_dim >= X.shape[1]:
            # On autorise quand même, mais généralement l'intérêt est plus clair quand latent_dim < input_dim.
            pass

        print(f"\n[{dataset_name}] Dimension latente = {latent_dim}")

        start_ae = time.time()
        model, scaler, losses = train_autoencoder(
            X,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False
        )
        ae_time = time.time() - start_ae

        Z = encode_with_autoencoder(X, model, scaler)

        for k in k_values:
            start = time.time()

            kmeans_pp = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                random_state=42
            )

            labels = kmeans_pp.fit_predict(Z)
            kmeans_time = time.time() - start

            rows.append({
                "dataset": dataset_name,
                "method": "Autoencoder + KMeans++",
                "latent_dim": latent_dim,
                "K": k,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "inertia": kmeans_pp.inertia_,
                "silhouette": silhouette_score(Z, labels),
                "davies_bouldin": davies_bouldin_score(Z, labels),
                "calinski_harabasz": calinski_harabasz_score(Z, labels),
                "autoencoder_time": ae_time,
                "kmeans_time": kmeans_time,
                "total_time": ae_time + kmeans_time
            })

    return pd.DataFrame(rows)


# ============================================================
# 9. Étude selon le nombre de features
# ============================================================

def study_number_of_features(
    X: np.ndarray,
    dataset_name: str,
    feature_sizes=None,
    k_values=range(2, 11),
    latent_dim: int = 3,
    epochs: int = 100,
    batch_size: int = 32
):
    """
    Compare les deux approches en alternant le nombre de features.

    Important :
    - On prend les premières colonnes de X pour garder la logique simple du notebook.
    - Si ton vrai dataset a un ordre de features important, tu peux remplacer cette sélection.
    """

    if feature_sizes is None:
        max_f = X.shape[1]
        feature_sizes = sorted(set([2, 3, 4, 5, 10, 20, max_f]))
        feature_sizes = [f for f in feature_sizes if 1 < f <= max_f]

    rows = []

    for n_features in feature_sizes:
        if n_features > X.shape[1] or n_features < 2:
            continue

        print(f"\n[{dataset_name}] Nombre de features = {n_features}")

        X_sub = X[:, :n_features]

        df_cmp, _, _ = compare_kmeans_normal_vs_autoencoder_kmeanspp(
            X_sub,
            dataset_name=dataset_name,
            k_values=k_values,
            latent_dim=min(latent_dim, max(1, n_features - 1)),
            epochs=epochs,
            batch_size=batch_size,
            verbose=False
        )

        df_cmp["n_features_tested"] = n_features
        rows.append(df_cmp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ============================================================
# 10. Étude selon le nombre d'échantillons
# ============================================================

def study_number_of_samples(
    X: np.ndarray,
    dataset_name: str,
    sample_sizes=None,
    k_values=range(2, 11),
    latent_dim: int = 3,
    epochs: int = 100,
    batch_size: int = 32,
    random_state: int = 42
):
    """
    Compare les deux approches en alternant le nombre d'échantillons.
    """

    rng = np.random.default_rng(random_state)

    if sample_sizes is None:
        n = X.shape[0]
        sample_sizes = sorted(set([50, 100, 150, 300, 500, 1000, n]))
        sample_sizes = [s for s in sample_sizes if 10 <= s <= n]

    rows = []

    for n_samples in sample_sizes:
        if n_samples > len(X) or n_samples < 10:
            continue

        print(f"\n[{dataset_name}] Nombre d'échantillons = {n_samples}")

        idx = rng.choice(len(X), size=n_samples, replace=False)
        X_sub = X[idx]

        # K ne doit pas dépasser n_samples - 1 pour silhouette_score
        valid_k_values = [k for k in k_values if 2 <= k < n_samples]

        df_cmp, _, _ = compare_kmeans_normal_vs_autoencoder_kmeanspp(
            X_sub,
            dataset_name=dataset_name,
            k_values=valid_k_values,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False
        )

        df_cmp["n_samples_tested"] = n_samples
        rows.append(df_cmp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ============================================================
# 11. Fonctions de visualisation enrichies
# ============================================================

def plot_metric_by_k(df: pd.DataFrame, metric: str, title: str, save_path: str = None):
    """Courbe d'une métrique selon K pour chaque méthode."""

    plt.figure(figsize=(10, 6))

    for method in df["method"].unique():
        sub = df[df["method"] == method]
        grouped = sub.groupby("K")[metric].mean().reset_index()
        plt.plot(grouped["K"], grouped[metric], marker="o", label=method)

    plt.xlabel("Nombre de clusters K")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_time_by_k(df: pd.DataFrame, title: str, save_path: str = None):
    """Courbe du temps d'exécution total selon K."""

    plt.figure(figsize=(10, 6))

    for method in df["method"].unique():
        sub = df[df["method"] == method]
        grouped = sub.groupby("K")["total_time"].mean().reset_index()
        plt.plot(grouped["K"], grouped["total_time"], marker="o", label=method)

    plt.xlabel("Nombre de clusters K")
    plt.ylabel("Temps total d'exécution (s)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_best_per_factor(
    df: pd.DataFrame,
    factor_col: str,
    metric: str,
    title: str,
    save_path: str = None
):
    """
    Pour chaque méthode et chaque valeur du facteur,
    sélectionne la meilleure ligne selon silhouette,
    puis trace metric.
    """

    if df.empty or factor_col not in df.columns:
        print(f"Impossible de tracer {title}: colonne absente ou dataframe vide.")
        return

    best = df.loc[df.groupby(["method", factor_col])["silhouette"].idxmax()]

    plt.figure(figsize=(10, 6))

    for method in best["method"].unique():
        sub = best[best["method"] == method].sort_values(factor_col)
        plt.plot(sub[factor_col], sub[metric], marker="o", label=method)

    plt.xlabel(factor_col)
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_latent_dimension_results(df_latent: pd.DataFrame, dataset_name: str):
    """Visualise l'effet de la dimension latente sur Autoencoder + KMeans++."""

    if df_latent.empty:
        return

    best = df_latent.loc[df_latent.groupby("latent_dim")["silhouette"].idxmax()]
    best = best.sort_values("latent_dim")

    plt.figure(figsize=(10, 6))
    plt.plot(best["latent_dim"], best["silhouette"], marker="o")
    plt.xlabel("Dimension de l'espace latent")
    plt.ylabel("Meilleur Silhouette Score")
    plt.title(f"{dataset_name} - Silhouette selon dimension latente")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{dataset_name}_latent_dim_silhouette.png", dpi=150, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(best["latent_dim"], best["total_time"], marker="o")
    plt.xlabel("Dimension de l'espace latent")
    plt.ylabel("Temps total d'exécution (s)")
    plt.title(f"{dataset_name} - Temps selon dimension latente")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{dataset_name}_latent_dim_time.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Heatmap simple avec matplotlib
    pivot = df_latent.pivot_table(
        index="latent_dim",
        columns="K",
        values="silhouette",
        aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Silhouette Score")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xlabel("K")
    plt.ylabel("Dimension latente")
    plt.title(f"{dataset_name} - Heatmap Silhouette : K vs dimension latente")
    plt.savefig(f"{dataset_name}_heatmap_latent_k.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n[{dataset_name}] Meilleures dimensions latentes:")
    print(best.sort_values("silhouette", ascending=False))


def plot_autoencoder_loss(losses, title: str, save_path: str = None):
    """Trace la convergence de l'autoencoder."""

    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def save_results(df: pd.DataFrame, path: str):
    """Sauvegarde un dataframe en CSV."""
    if df is not None and not df.empty:
        df.to_csv(path, index=False)
        print(f"Résultats sauvegardés : {path}")


# ============================================================
# 12. Pipeline complet sur les mêmes datasets
# ============================================================

def run_full_study_on_dataset(
    X: np.ndarray,
    dataset_name: str,
    true_k: int,
    k_values=range(2, 11),
    latent_dims=(2, 3, 4, 5),
    feature_sizes=None,
    sample_sizes=None,
    epochs: int = 100,
    batch_size: int = 32
):
    """
    Lance toute l'étude comparative sur un dataset donné.
    """

    print("\n" + "#" * 90)
    print(f"ÉTUDE COMPLÈTE : {dataset_name}")
    print("#" * 90)

    # 1. Comparaison originale
    results_original, Z_original = compare_methods(X, n_clusters=true_k)
    visualize_results(
        X,
        results_original,
        Z_original,
        save_path=f"{dataset_name}_original_comparison.png"
    )

    # Loss autoencoder original
    if "Autoencoder" in results_original and "losses" in results_original["Autoencoder"]:
        plot_autoencoder_loss(
            results_original["Autoencoder"]["losses"],
            title=f"{dataset_name} - Convergence Autoencoder",
            save_path=f"{dataset_name}_autoencoder_loss.png"
        )

    # 2. Comparaison demandée selon K
    df_main, Z, losses = compare_kmeans_normal_vs_autoencoder_kmeanspp(
        X,
        dataset_name=dataset_name,
        k_values=k_values,
        latent_dim=min(3, max(1, X.shape[1] - 1)),
        epochs=epochs,
        batch_size=batch_size,
        verbose=False
    )

    save_results(df_main, f"{dataset_name}_main_comparison.csv")

    plot_metric_by_k(
        df_main,
        metric="silhouette",
        title=f"{dataset_name} - KMeans normal vs Autoencoder+KMeans++ : Silhouette",
        save_path=f"{dataset_name}_main_silhouette_by_k.png"
    )

    plot_metric_by_k(
        df_main,
        metric="davies_bouldin",
        title=f"{dataset_name} - KMeans normal vs Autoencoder+KMeans++ : Davies-Bouldin",
        save_path=f"{dataset_name}_main_davies_by_k.png"
    )

    plot_metric_by_k(
        df_main,
        metric="calinski_harabasz",
        title=f"{dataset_name} - KMeans normal vs Autoencoder+KMeans++ : Calinski-Harabasz",
        save_path=f"{dataset_name}_main_calinski_by_k.png"
    )

    plot_time_by_k(
        df_main,
        title=f"{dataset_name} - Analyse temporelle selon K",
        save_path=f"{dataset_name}_main_time_by_k.png"
    )

    # 3. Dimension latente
    df_latent = study_latent_dimensions(
        X,
        dataset_name=dataset_name,
        latent_dims=latent_dims,
        k_values=k_values,
        epochs=epochs,
        batch_size=batch_size
    )

    save_results(df_latent, f"{dataset_name}_latent_dimension_study.csv")
    plot_latent_dimension_results(df_latent, dataset_name)

    # 4. Nombre de features
    df_features = study_number_of_features(
        X,
        dataset_name=dataset_name,
        feature_sizes=feature_sizes,
        k_values=k_values,
        latent_dim=min(3, max(1, X.shape[1] - 1)),
        epochs=epochs,
        batch_size=batch_size
    )

    save_results(df_features, f"{dataset_name}_features_study.csv")

    plot_best_per_factor(
        df_features,
        factor_col="n_features_tested",
        metric="silhouette",
        title=f"{dataset_name} - Meilleur Silhouette selon le nombre de features",
        save_path=f"{dataset_name}_features_silhouette.png"
    )

    plot_best_per_factor(
        df_features,
        factor_col="n_features_tested",
        metric="total_time",
        title=f"{dataset_name} - Temps selon le nombre de features",
        save_path=f"{dataset_name}_features_time.png"
    )

    # 5. Nombre d'échantillons
    df_samples = study_number_of_samples(
        X,
        dataset_name=dataset_name,
        sample_sizes=sample_sizes,
        k_values=k_values,
        latent_dim=min(3, max(1, X.shape[1] - 1)),
        epochs=epochs,
        batch_size=batch_size
    )

    save_results(df_samples, f"{dataset_name}_samples_study.csv")

    plot_best_per_factor(
        df_samples,
        factor_col="n_samples_tested",
        metric="silhouette",
        title=f"{dataset_name} - Meilleur Silhouette selon le nombre d'échantillons",
        save_path=f"{dataset_name}_samples_silhouette.png"
    )

    plot_best_per_factor(
        df_samples,
        factor_col="n_samples_tested",
        metric="total_time",
        title=f"{dataset_name} - Temps selon le nombre d'échantillons",
        save_path=f"{dataset_name}_samples_time.png"
    )

    # 6. Résumé final
    print("\n" + "=" * 90)
    print(f"RÉSUMÉ FINAL : {dataset_name}")
    print("=" * 90)

    print("\nMeilleure configuration principale par méthode:")
    best_main = df_main.loc[df_main.groupby("method")["silhouette"].idxmax()]
    print(best_main.sort_values("silhouette", ascending=False))

    print("\nMeilleure dimension latente:")
    best_latent = df_latent.loc[df_latent["silhouette"].idxmax()]
    print(best_latent)

    if not df_features.empty:
        print("\nMeilleure configuration selon features:")
        best_features = df_features.loc[df_features.groupby("method")["silhouette"].idxmax()]
        print(best_features.sort_values("silhouette", ascending=False))

    if not df_samples.empty:
        print("\nMeilleure configuration selon échantillons:")
        best_samples = df_samples.loc[df_samples.groupby("method")["silhouette"].idxmax()]
        print(best_samples.sort_values("silhouette", ascending=False))

    return {
        "original_results": results_original,
        "df_main": df_main,
        "df_latent": df_latent,
        "df_features": df_features,
        "df_samples": df_samples,
        "Z": Z,
        "losses": losses
    }


# ============================================================
# 13. Main : mêmes datasets que ton notebook
# ============================================================

if __name__ == "__main__":
    print("[v0] Deep Embedded Clustering Demo - Version finale enrichie")
    print("=" * 80)

    # Pour reproductibilité
    np.random.seed(42)
    torch.manual_seed(42)

    # --------------------------------------------------------
    # Test 1 : Données synthétiques
    # Même logique que ton notebook : make_blobs
    # --------------------------------------------------------
    print("\n[TEST 1] Données synthétiques")

    X_synth, y_synth = make_blobs(
        n_samples=300,
        n_features=20,
        centers=4,
        random_state=42,
        cluster_std=0.8
    )

    synth_results = run_full_study_on_dataset(
        X=X_synth,
        dataset_name="synthetic",
        true_k=4,
        k_values=range(2, 11),
        latent_dims=(2, 3, 4, 5),
        feature_sizes=[2, 5, 10, 15, 20],
        sample_sizes=[50, 100, 150, 200, 300],
        epochs=100,
        batch_size=32
    )

    # --------------------------------------------------------
    # Test 2 : Iris Dataset
    # Même dataset que ton notebook
    # --------------------------------------------------------
    print("\n[TEST 2] Iris Dataset")

    iris = load_iris()
    X_iris = iris.data

    iris_results = run_full_study_on_dataset(
        X=X_iris,
        dataset_name="iris",
        true_k=3,
        k_values=range(2, 8),
        latent_dims=(2, 3, 4, 5),
        feature_sizes=[2, 3, 4],
        sample_sizes=[50, 100, 150],
        epochs=100,
        batch_size=32
    )

    print("\n" + "=" * 80)
    print("Demo complétée.")
    print("=" * 80)
