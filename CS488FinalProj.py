# Reads train/test split data, makes Exploratory Data analysis plots, ranks features
# selects the best ones and runs PCA.
import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


try:
    from Configuration import RANDOM_SEED, TARGET, PROCESSED_DIR
except ImportError:
    RANDOM_SEED, TARGET, PROCESSED_DIR = 42, "target", "data/processed"

FIG_DIR = "figures/"
os.makedirs(FIG_DIR, exist_ok=True)

def load_process():
    # Load the splits
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR + "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR + "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR + "y_train.csv")).squeeze("columns")
    return X_train, X_test, y_train 

def plot_histogram(X_train, y_train):
    # Histogram of every feature. blue = no disease. red = disease.
    df = X_train.copy()
    df[TARGET] = y_train.values
    cols = list(X_train.columns)
    n_cols = 4
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows))
    
    for ax, c in zip(axes.ravel(), cols): # Hide unused panels
        for label, color in [(0, "tab:blue"), (1, "tab:red")]:
            ax.hist(df.loc[df[TARGET] == label, c], bins=20, alpha=0.55,
                    color=color, label=f"{TARGET}={label}")
            
        ax.set_title(c)

    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes.ravel()[0].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR + "histogram.png", dpi=200)
    plt.close()

def plot_correlation(X_train, y_train):
    # Correlation heatmaps for training data only.
    df = X_train.copy()
    df[TARGET] = y_train.values
    corr_eff = df.corr(method="pearson", numeric_only=True)

    # Full Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_eff, linecolor="white", linewidths=1, annot=True)
    plt.title("Correlation coefficient heat map for training data")
    plt.tight_layout()
    plt.savefig(FIG_DIR + "correlation_heatmap.png", dpi=200)
    plt.close()

    # Mask upper half (redudant data)
    mask = np.zeros_like(corr_eff)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_eff, linecolor="white", linewidths=1, mask=mask, ax=ax,
                annot=True, fmt=".2f", cmap="coolwarm", center=0)
    ax.set_title("Correlation coefficient heat map lower half")
    plt.tight_layout()
    plt.close()
    return corr_eff

# Pair plot
def plot_pp(X_train, y_train, continuos_cols=None):

    if continuos_cols is None:
        continuos_cols = ["age", "resting bp (s)", "cholestrol",
                          "max heart rate", "oldpeak"]
        # Keep only the names that exist
        continuos_cols = [c for c in continuos_cols if c in X_train.columns]
        df = X_train[continuos_cols].copy()
        df["diagnosis"] = y_train.map({0: "No disease", 1: "Disease"}).values
        g = sns.pairplot(df, hue="diagnosis", diag_kind="hist")
        g.savefig(FIG_DIR + "pairplot.png", dpi=150)
        plt.close("all")

def rank_features(X_train, y_train):
    # Rank features three ways and find the average
    
    corr = X_train.corrwith(y_train).abs() # Linear link to target

    mi = pd.Series(mutual_info_classif(X_train, y_train, random_state=RANDOM_SEED),
                   index=X_test.columns)
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED,
                                n_jobs=-1).fit(X_train, y_train)
    # Tree importance
    rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns)

    table = pd.DataFrame({"abs_corr": corr, "mutual_info": mi,
                          "rf_importance": rf_imp})
    # Rank 1 = best. Lowest average rank = most useful feature
    table["avg_rank"] = table.rank(ascending=False).mean(axis=1)

    return table.sort_values("avg_rank")

def plot_importance(table):
    # Bar chart of Random Forest Importance
    t = table.sort_values("rf_importance")
    plt.figure(figsize=(8, 5))
    plt.barh(t.index, t["rf_importance"], color="tab:green")
    plt.xlabel("Random Forrest Importance")
    plt.title("Feature importance (training data)")
    plt.tight_layout()
    plt.savefig(FIG_DIR + "feature_importance.png", dpi=200)
    plt.close()

def select_features(X_train, y_train, X_test, k=8, corr_threshold=0.85):
    # Remove highly correlated duplicates
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > corr_threshold).any()]
    X_tr = X_train.drop(columns=drop)

    # Rank and keep top k
    table = rank_features(X_tr, y_train)
    selected_cols = list(table.index[:k])

    # Same columns are applied to test so train and test data match
    return X_train[selected_cols], X_test[selected_cols], selected_cols

def run_pca(X_train, X_test, n_components=0.95):

    # Scale, then PCA. For train only then applied to test
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED) # Keep 95% variance
    tr = pca.fit_transform(scaler.transform(X_train))
    te = pca.transform(scaler.transform(X_test))
    names = [f"PC{i + 1}" for i in range (tr.shape[1])]

    # Explained variance plot. Bars = components. Lines = running total
    cum = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(7, 4))
    plt.bar(range(1, len(cum) + 1), pca.explained_variance_ratio_, alpha=0.6,
            label="per component")
    plt.plot(range(1, len(cum) +1), cum, "o-", color="tab:red", label="cumulative")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.legend()
    plt.title("PCA explained variance ratio")
    plt.legend()
    plt.title("PCA explained variance")
    plt.tight_layout()
    plt.savefig(FIG_DIR + "pca_variance.png", dpi=200)
    plt.close()

    return pd.DataFrame(tr, columns=names), pd.DataFrame(te, columns=names)

if __name__ == "__main__":
    
    # Run everything
    X_train, X_test, y_train = load_process()
    plot_histogram(X_train, y_train)
    plot_correlation(X_train, y_train)
    plot_pp(X_train, y_train)

    table = rank_features(X_train, y_train)
    plot_importance(table)
    print("Feature ranking:\n", table.round(3), "\n")

    X_tr_sel, X_te_sel, cols  = select_features(X_train, y_train, X_test)
    print("Selected features:", cols)
    X_tr_pca, X_te_pca = run_pca(X_train, X_test)
    print("PCA components:", X_tr_pca.shape[1])

    # File outputs
    out = PROCESSED_DIR + "features_selected/"
    os.makedirs(out, exist_ok=True)
    X_tr_sel.to_csv(out + "X_train_sel.csv", index=False)
    X_te_sel.to_csv(out + "X_test_sel.csv", index=False)
    X_tr_pca.to_csv(out + "X_train_pca.csv", index=False)
    X_te_pca.to_csv(out + "X_test_pca.csv", index=False)
    table.to_csv(out + "feature_ranking.csv")
