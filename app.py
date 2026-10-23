"""
ML Capstone Studio - a point-and-click GUI for the three capstone notebooks.

    Regression      -> Crop yield prediction      (notebooks/02_Regression_Crop_Yield.ipynb)
    Classification  -> Facial emotion (FER2013)   (notebooks/01_Classification_Emotion.ipynb)
    Clustering      -> Earthquake clusters        (notebooks/03_Clustering_Earthquakes_Corrected.ipynb)

Run:   streamlit run app.py
"""
import io
import uuid
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image, ImageOps
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="colorblind")

st.set_page_config(page_title="ML Capstone Studio", page_icon="🤖", layout="wide")

BASE = Path(__file__).resolve().parent
YIELD_PATH = BASE / "data" / "crop_yield" / "yield_df.csv"
QUAKE_PATH = BASE / "data" / "earthquake" / "database.csv"
SAVED_EMOTION_MODEL = BASE / "models" / "final_emotion_classifier.pkl"
RANDOM_STATE = 42


# =====================================================================================
# Shared helpers
# =====================================================================================
def file_id(upload):
    """Stable identifier for an uploaded file (used as a cache key)."""
    return getattr(upload, "file_id", None) or f"{upload.name}-{upload.size}"


def show_fig(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def df_download(df, label, filename, key):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv", key=key)


def match_columns(df, wanted):
    """Case-insensitive rename so 'latitude' / 'LATITUDE' both match 'Latitude'."""
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {lower[w.lower()]: w for w in wanted if w.lower() in lower and lower[w.lower()] != w}
    return df.rename(columns=rename)


# =====================================================================================
# 1) REGRESSION - crop yield
# =====================================================================================
TARGET = "hg/ha_yield"
RAIN, PEST, TEMP = "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"
REG_CAT = ["Area", "Item"]
REG_NUM = ["Year", RAIN, PEST, TEMP, "rainfall_pesticide_interaction"]
REG_INPUT = ["Area", "Item", "Year", RAIN, PEST, TEMP]
OUTLIER_COLS = [RAIN, PEST, TEMP]

REG_MODELS = {
    "Random Forest (best in notebook)": lambda: RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "KNN": lambda: KNeighborsRegressor(n_neighbors=5, weights="distance"),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Gradient Boosting": lambda: GradientBoostingRegressor(learning_rate=0.2, n_estimators=100, random_state=RANDOM_STATE),
    "Ridge": lambda: Ridge(alpha=1.0),
    "Linear Regression": lambda: LinearRegression(),
}


@st.cache_data(show_spinner=False)
def load_bundled_yield():
    return pd.read_csv(YIELD_PATH)


@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes):
    return pd.read_csv(io.BytesIO(data))


def clean_yield(df):
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore").copy()
    for c in df.select_dtypes("number").columns:
        df[c] = df[c].fillna(df[c].median())
    return df.drop_duplicates().reset_index(drop=True)


def add_interaction(df):
    df = df.copy()
    df["rainfall_pesticide_interaction"] = df[RAIN] * df[PEST]
    return df


@st.cache_resource(show_spinner=False)
def train_regressor(data_id, model_name, _df):
    """Same pipeline as the notebook: clean -> interaction -> stratified split ->
    train-only IQR clipping -> one-hot + scaling -> model."""
    df = add_interaction(clean_yield(_df))
    X, y = df[REG_CAT + REG_NUM], df[TARGET]
    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=bins)
    X_tr, X_te = X_tr.copy(), X_te.copy()

    bounds = {}
    for c in OUTLIER_COLS:
        q1, q3 = X_tr[c].quantile(0.25), X_tr[c].quantile(0.75)
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        bounds[c] = (lo, hi)
        X_tr[c], X_te[c] = X_tr[c].clip(lo, hi), X_te[c].clip(lo, hi)

    pre = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), REG_CAT),
         ("numerical", StandardScaler(), REG_NUM)]
    )
    Xtr_p, Xte_p = pre.fit_transform(X_tr), pre.transform(X_te)
    model = REG_MODELS[model_name]()
    model.fit(Xtr_p, y_tr)
    pred = model.predict(Xte_p)
    return {
        "model_name": model_name, "model": model, "pre": pre, "bounds": bounds,
        "metrics": {"R²": r2_score(y_te, pred), "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                    "MAE": mean_absolute_error(y_te, pred)},
        "y_test": y_te.to_numpy(), "pred": pred,
        "feature_names": pre.get_feature_names_out(),
        "areas": sorted(df["Area"].unique()), "items": sorted(df["Item"].unique()),
        "defaults": df[[RAIN, PEST, TEMP, "Year"]].median().to_dict(),
        "history": df, "n_train": len(X_tr), "n_test": len(X_te),
    }


def reg_predict(bundle, df_in):
    d = add_interaction(df_in[REG_INPUT].copy())
    for c, (lo, hi) in bundle["bounds"].items():
        d[c] = d[c].clip(lo, hi)
    return np.clip(bundle["model"].predict(bundle["pre"].transform(d[REG_CAT + REG_NUM])), 0, None)


@st.cache_resource(show_spinner=False)
def compare_regressors(data_id, _df):
    rows = []
    for name in REG_MODELS:
        m = train_regressor(data_id, name, _df)["metrics"]
        rows.append({"Model": name, **m})
    return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)


def regression_tab():
    st.header("🌾 Regression - Crop Yield Prediction")
    st.caption("Predict crop yield (hg/ha) from country, crop, year, rainfall, pesticides and temperature.")

    # ---- data source -------------------------------------------------------------
    c1, c2 = st.columns([1, 1])
    with c1:
        source = st.radio("Training data", ["Use project dataset (yield_df.csv)", "Browse my own CSV…"],
                          key="reg_source")
    df_raw, data_id = None, None
    with c2:
        if source.startswith("Use"):
            if YIELD_PATH.exists():
                df_raw, data_id = load_bundled_yield(), "bundled-yield"
                st.success(f"Loaded `{YIELD_PATH.name}` - {len(df_raw):,} rows")
            else:
                st.error("data/crop_yield/yield_df.csv not found. Choose 'Browse my own CSV…'.")
        else:
            up = st.file_uploader("Select a CSV from your computer", type=["csv"], key="reg_train_csv")
            st.caption(f"Needs columns: {', '.join(REG_INPUT + [TARGET])}")
            if up is not None:
                df_raw, data_id = read_csv_bytes(up.getvalue()), f"upload-{file_id(up)}"
                missing = [c for c in REG_INPUT + [TARGET] if c not in df_raw.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                    df_raw = None
                else:
                    st.success(f"Loaded {up.name} - {len(df_raw):,} rows")
    if df_raw is None:
        return

    # ---- model + train button ----------------------------------------------------
    m1, m2 = st.columns([2, 1])
    model_name = m1.selectbox("Model", list(REG_MODELS), key="reg_model")
    m2.write("")
    m2.write("")
    if m2.button("🚀 Train / load model", type="primary", key="reg_train_btn"):
        st.session_state["reg_cfg"] = (data_id, model_name)
    cfg = st.session_state.get("reg_cfg")
    if not cfg or cfg[0] != data_id:
        st.info("Pick a model and press **Train / load model** to begin (Random Forest takes ~20-40 s the first time).")
        return
    with st.spinner(f"Training {cfg[1]}…"):
        bundle = train_regressor(cfg[0], cfg[1], df_raw)

    r = bundle["metrics"]
    st.markdown(f"**Active model:** {bundle['model_name']} &nbsp;·&nbsp; trained on {bundle['n_train']:,} rows, "
                f"tested on {bundle['n_test']:,}")
    k1, k2, k3 = st.columns(3)
    k1.metric("R² (test)", f"{r['R²']:.4f}")
    k2.metric("RMSE (hg/ha)", f"{r['RMSE']:,.0f}")
    k3.metric("MAE (hg/ha)", f"{r['MAE']:,.0f}")

    t_pred, t_batch, t_perf, t_data = st.tabs(
        ["🔮 Predict one", "📂 Predict from CSV", "📊 Model performance", "🔎 Explore data"])

    # ---- single prediction -------------------------------------------------------
    with t_pred:
        d = bundle["defaults"]
        a, b, c = st.columns(3)
        area = a.selectbox("Country / Area", bundle["areas"],
                           index=bundle["areas"].index("India") if "India" in bundle["areas"] else 0, key="reg_area")
        item = b.selectbox("Crop", bundle["items"], key="reg_item")
        year = c.number_input("Year", 1960, 2100, int(d["Year"]), key="reg_year")
        a, b, c = st.columns(3)
        rain = a.number_input("Average rainfall (mm / year)", 0.0, 10000.0, float(d[RAIN]), step=50.0, key="reg_rain")
        pest = b.number_input("Pesticides (tonnes)", 0.0, 1_000_000.0, float(d[PEST]), step=100.0, key="reg_pest")
        temp = c.number_input("Average temperature (°C)", -30.0, 50.0, float(d[TEMP]), step=0.5, key="reg_temp")
        if st.button("Predict yield", type="primary", key="reg_predict_btn"):
            row = pd.DataFrame([{"Area": area, "Item": item, "Year": year, RAIN: rain, PEST: pest, TEMP: temp}])
            yhat = float(reg_predict(bundle, row)[0])
            o1, o2 = st.columns(2)
            o1.metric("Predicted yield", f"{yhat:,.0f} hg/ha")
            o2.metric("≈ tonnes per hectare", f"{yhat / 10000:.2f} t/ha")
            hist = bundle["history"]
            hist = hist[(hist["Area"] == area) & (hist["Item"] == item)]
            if len(hist):
                st.caption(f"For reference, the dataset's average for {item} in {area} is "
                           f"{hist[TARGET].mean():,.0f} hg/ha ({len(hist)} records, "
                           f"{int(hist['Year'].min())}-{int(hist['Year'].max())}).")
            else:
                st.caption(f"No historical record of {item} in {area} exists in the training data - treat with caution.")

    # ---- batch prediction --------------------------------------------------------
    with t_batch:
        st.write("Browse a CSV with one row per prediction. Required columns:")
        st.code(", ".join(REG_INPUT))
        tmpl = pd.DataFrame([{"Area": bundle["areas"][0], "Item": bundle["items"][0], "Year": 2020,
                              RAIN: d[RAIN], PEST: d[PEST], TEMP: d[TEMP]}])
        df_download(tmpl, "⬇️ Download a template CSV", "yield_prediction_template.csv", "reg_tmpl")
        up = st.file_uploader("Select your CSV", type=["csv"], key="reg_batch_csv")
        if up is not None:
            new = read_csv_bytes(up.getvalue())
            missing = [c for c in REG_INPUT if c not in new.columns]
            if missing:
                st.error(f"Your file is missing: {missing}")
            else:
                for col in [RAIN, PEST, TEMP, "Year"]:
                    new[col] = pd.to_numeric(new[col], errors="coerce")
                ok = new.dropna(subset=REG_INPUT).copy()
                if len(ok) < len(new):
                    st.warning(f"{len(new) - len(ok)} rows with empty/invalid values were skipped.")
                unknown = sorted(set(ok["Area"]) - set(bundle["areas"])) + sorted(set(ok["Item"]) - set(bundle["items"]))
                if unknown:
                    st.warning(f"Not seen in training (predictions less reliable): {unknown}")
                ok["predicted_yield_hg_per_ha"] = reg_predict(bundle, ok).round(0)
                st.success(f"Predicted {len(ok):,} rows")
                st.dataframe(ok.head(500))
                df_download(ok, "⬇️ Download predictions", "yield_predictions.csv", "reg_dl")
                if TARGET in ok.columns:
                    tgt = pd.to_numeric(ok[TARGET], errors="coerce")
                    if tgt.notna().all():
                        st.info(f"Your file also has actual yields → R² = {r2_score(tgt, ok['predicted_yield_hg_per_ha']):.4f}, "
                                f"MAE = {mean_absolute_error(tgt, ok['predicted_yield_hg_per_ha']):,.0f} hg/ha")

    # ---- performance -------------------------------------------------------------
    with t_perf:
        y_te, pred = bundle["y_test"], bundle["pred"]
        p1, p2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(y_te, pred, alpha=0.35, s=10)
        lim = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
        ax.plot(lim, lim, "r--")
        ax.set(xlabel="Actual yield (hg/ha)", ylabel="Predicted yield (hg/ha)", title="Predicted vs Actual")
        with p1:
            show_fig(fig)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(pred, y_te - pred, alpha=0.35, s=10)
        ax.axhline(0, color="r", ls="--")
        ax.set(xlabel="Predicted yield", ylabel="Residual (actual − predicted)", title="Residuals")
        with p2:
            show_fig(fig)

        model = bundle["model"]
        imp = getattr(model, "feature_importances_", None)
        label = "Feature importance"
        if imp is None and hasattr(model, "coef_"):
            imp, label = np.abs(model.coef_), "|Coefficient| (scaled features)"
        if imp is not None:
            top = pd.DataFrame({"Feature": bundle["feature_names"], label: imp}).nlargest(15, label)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=top, y="Feature", x=label, ax=ax, color="steelblue")
            ax.set_title(f"Top 15 - {label}")
            show_fig(fig)

        st.subheader("Compare all models")
        if st.button("Train & compare every model (≈1-2 min)", key="reg_compare_btn"):
            st.session_state["reg_compare"] = data_id
        if st.session_state.get("reg_compare") == data_id:
            with st.spinner("Training all models…"):
                cmp_df = compare_regressors(data_id, df_raw)
            st.dataframe(cmp_df.style.format({"R²": "{:.4f}", "RMSE": "{:,.0f}", "MAE": "{:,.0f}"}))
            fig, ax = plt.subplots(figsize=(8, 3.5))
            sns.barplot(data=cmp_df, y="Model", x="R²", ax=ax, color="seagreen")
            ax.set_xlim(min(0, cmp_df["R²"].min()), 1)
            show_fig(fig)

    # ---- data explorer -----------------------------------------------------------
    with t_data:
        hist = bundle["history"]
        st.dataframe(hist.head(200))
        e1, e2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(hist[["Year", RAIN, PEST, TEMP, TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation heatmap")
        with e1:
            show_fig(fig)
        avg = hist.groupby("Item")[TARGET].mean().sort_values()
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        avg.plot.barh(ax=ax, color="darkorange")
        ax.set(xlabel="Mean yield (hg/ha)", title="Average yield by crop")
        with e2:
            show_fig(fig)
        ca, cb = st.columns(2)
        area = ca.selectbox("Trend for country", bundle["areas"], key="reg_trend_area")
        crops = cb.multiselect("Crops", bundle["items"], default=bundle["items"][:3], key="reg_trend_crops")
        sub = hist[(hist["Area"] == area) & (hist["Item"].isin(crops))]
        if len(sub):
            fig, ax = plt.subplots(figsize=(9, 3.8))
            sns.lineplot(data=sub, x="Year", y=TARGET, hue="Item", marker="o", ax=ax)
            ax.set_title(f"Yield over time - {area}")
            show_fig(fig)


# =====================================================================================
# 2) CLASSIFICATION - FER2013 emotion
# =====================================================================================
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOJI = {"Angry": "😠", "Disgust": "🤢", "Fear": "😨", "Happy": "😄", "Sad": "😢", "Surprise": "😲", "Neutral": "😐"}
IMG = 48


def _fit_svc():
    return SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_STATE)


CLF_MODELS = {
    "Logistic Regression (fast)": lambda: LogisticRegression(max_iter=300, random_state=RANDOM_STATE),
    "KNN (k=7)": lambda: KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    "Gaussian Naive Bayes": lambda: GaussianNB(),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, early_stopping=False,
                                                                random_state=RANDOM_STATE),
    "MLP Neural Network": lambda: make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=300, early_stopping=True,
                      random_state=RANDOM_STATE)),
    "SVC (uses 8,000 images, slower)": _fit_svc,
}


def parse_pixels(series):
    """'12 34 56 …' strings -> (n, 2304) float32 matrix. Raises ValueError on bad rows."""
    rows = [np.array(s.split(), dtype=np.float32) for s in series.astype(str)]
    bad = [i for i, r in enumerate(rows) if r.size != IMG * IMG]
    if bad:
        raise ValueError(f"{len(bad)} rows do not contain exactly {IMG * IMG} pixel values (first bad row index: {bad[0]}).")
    return np.vstack(rows)


@st.cache_resource(show_spinner=False)
def load_fer_csv(_file, fid):
    return pd.read_csv(_file, usecols=lambda c: c in ("emotion", "pixels"))


def train_classifier(df, model_name, n_max):
    df = df.dropna(subset=["pixels", "emotion"]).drop_duplicates().reset_index(drop=True)
    if len(df) > n_max:
        df, _ = train_test_split(df, train_size=n_max, random_state=RANDOM_STATE, stratify=df["emotion"])
    X = parse_pixels(df["pixels"])
    keep = ~np.any((X < 0) | (X > 255), axis=1)
    X, y = X[keep] / 255.0, df["emotion"].to_numpy()[keep].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    pca = PCA(n_components=100, random_state=RANDOM_STATE)
    P_tr, P_te = pca.fit_transform(X_tr), pca.transform(X_te)
    model = CLF_MODELS[model_name]()
    if model_name.startswith("SVC"):
        model.fit(P_tr[:8000], y_tr[:8000])
    else:
        model.fit(P_tr, y_tr)
    pred = model.predict(P_te)
    labels = list(range(len(EMOTIONS)))
    return {
        "token": uuid.uuid4().hex, "model_name": model_name, "pca": pca, "model": model, "class_names": EMOTIONS,
        "source": "trained in this app", "n_train": len(X_tr), "n_test": len(X_te),
        "accuracy": accuracy_score(y_te, pred), "f1": f1_score(y_te, pred, average="weighted"),
        "cm": confusion_matrix(y_te, pred, labels=labels),
        "report": pd.DataFrame(classification_report(y_te, pred, labels=labels, target_names=EMOTIONS,
                                                     output_dict=True, zero_division=0)).T.round(3),
    }


def clf_predict(bundle, X01):
    """X01: (n, 2304) already scaled to 0-1."""
    P = bundle["pca"].transform(X01)
    pred = bundle["model"].predict(P).astype(int)
    proba = bundle["model"].predict_proba(P) if hasattr(bundle["model"], "predict_proba") else None
    return pred, proba


@st.cache_resource(show_spinner=False)
def predict_fer_csv(fid, token, _df, _bundle):
    X = parse_pixels(_df["pixels"]) / 255.0
    pred, proba = clf_predict(_bundle, X)
    return X, pred, proba


def emotion_figure(img48, title):
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.imshow(img48, cmap="gray")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    return fig


def classification_tab():
    st.header("😀 Classification - Facial Emotion Recognition (FER2013)")
    st.caption("48×48 grayscale faces → PCA (100 components) → classifier → one of 7 emotions.")

    # ---- Step 1: get a model -----------------------------------------------------
    st.subheader("Step 1 - Get a model")
    mode = st.radio("How do you want to get a model?",
                    ["Train a model from fer2013.csv", "Load a saved model (.pkl from the notebook)"],
                    horizontal=True, key="clf_mode")

    if mode.startswith("Train"):
        up = st.file_uploader("Browse for **fer2013.csv** on your computer (columns: emotion, pixels)",
                              type=["csv"], key="clf_train_csv")
        if up is not None:
            with st.spinner("Reading CSV…"):
                df = load_fer_csv(up, file_id(up))
            if "pixels" not in df.columns or "emotion" not in df.columns:
                st.error("This file needs `emotion` and `pixels` columns.")
            else:
                st.success(f"Loaded {up.name}: {len(df):,} images")
                c1, c2 = st.columns([2, 2])
                model_name = c1.selectbox("Classifier", list(CLF_MODELS), key="clf_model")
                n_max = c2.slider("Images to use (fewer = faster)", 2000, int(max(len(df), 2001)),
                                  int(min(len(df), 20000)), step=1000, key="clf_nmax")
                if st.button("🚀 Train model", type="primary", key="clf_train_btn"):
                    try:
                        with st.spinner(f"Training {model_name} on up to {n_max:,} images… (PCA + model)"):
                            st.session_state["clf"] = train_classifier(df, model_name, n_max)
                            st.session_state.pop("clf_pkl", None)
                    except ValueError as e:
                        st.error(str(e))
    else:
        pkl = None
        if SAVED_EMOTION_MODEL.exists():
            st.success(f"Found `{SAVED_EMOTION_MODEL.relative_to(BASE)}` created by the notebook.")
            if st.button("Use this saved model", key="clf_use_saved"):
                pkl = SAVED_EMOTION_MODEL
        up_pkl = st.file_uploader("…or browse for a .pkl file (only open files you created yourself)",
                                  type=["pkl", "joblib"], key="clf_pkl_up")
        if up_pkl is not None:
            pkl = up_pkl
        if pkl is not None:
            try:
                obj = joblib.load(pkl)
                assert "pca" in obj and "model" in obj
                st.session_state["clf"] = {
                    "token": uuid.uuid4().hex, "model_name": type(obj["model"]).__name__, "pca": obj["pca"],
                    "model": obj["model"], "class_names": obj.get("class_names", EMOTIONS),
                    "source": getattr(pkl, "name", str(pkl)), "accuracy": None}
                st.session_state.pop("clf_pkl", None)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read that model file: {e}")

    bundle = st.session_state.get("clf")
    if bundle is None:
        st.info("Train or load a model to unlock Step 2.")
        return

    st.success(f"✅ Active model: **{bundle['model_name']}** ({bundle['source']})")
    if bundle.get("accuracy") is not None:
        k1, k2, k3 = st.columns(3)
        k1.metric("Accuracy (held-out 20%)", f"{bundle['accuracy']:.3f}")
        k2.metric("Weighted F1", f"{bundle['f1']:.3f}")
        k3.metric("Train / test images", f"{bundle['n_train']:,} / {bundle['n_test']:,}")
        with st.expander("Confusion matrix & per-emotion report"):
            fig, ax = plt.subplots(figsize=(6.5, 5))
            sns.heatmap(bundle["cm"], annot=True, fmt="d", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
            ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test set)")
            show_fig(fig)
            st.dataframe(bundle["report"])
        if st.button("💾 Prepare model file for download", key="clf_prep_pkl"):
            buf = io.BytesIO()
            joblib.dump({"pca": bundle["pca"], "model": bundle["model"], "class_names": bundle["class_names"],
                         "note": "Input: 48x48 grayscale pixels flattened, divided by 255, then pca.transform"}, buf)
            st.session_state["clf_pkl"] = buf.getvalue()
        if st.session_state.get("clf_pkl"):
            st.download_button("⬇️ Download model (.pkl)", st.session_state["clf_pkl"],
                               "final_emotion_classifier.pkl", key="clf_dl_pkl")

    # ---- Step 2: predict ---------------------------------------------------------
    st.subheader("Step 2 - Predict emotions")
    t_csv, t_img = st.tabs(["📂 From a CSV file", "🖼️ From a single photo"])
    names = bundle["class_names"]

    with t_csv:
        up = st.file_uploader("Browse for a CSV with a `pixels` column (and optionally `emotion` for scoring)",
                              type=["csv"], key="clf_pred_csv")
        if up is not None:
            df = load_fer_csv(up, "pred-" + file_id(up))
            if "pixels" not in df.columns:
                st.error("The CSV needs a `pixels` column.")
            else:
                try:
                    with st.spinner(f"Predicting {len(df):,} images…"):
                        X, pred, proba = predict_fer_csv(file_id(up), bundle["token"], df, bundle)
                except ValueError as e:
                    st.error(str(e))
                    return
                out = pd.DataFrame({"row": np.arange(len(df)), "predicted_emotion": [names[i] for i in pred]})
                if proba is not None:
                    out["confidence"] = proba.max(axis=1).round(3)
                has_truth = "emotion" in df.columns and df["emotion"].notna().all()
                if has_truth:
                    truth = df["emotion"].astype(int).to_numpy()
                    out["actual_emotion"] = [names[i] for i in truth]
                    out["correct"] = pred == truth
                    c1, c2 = st.columns(2)
                    c1.metric("Accuracy on this file", f"{accuracy_score(truth, pred):.3f}")
                    c2.metric("Weighted F1", f"{f1_score(truth, pred, average='weighted'):.3f}")
                    fig, ax = plt.subplots(figsize=(6.5, 5))
                    sns.heatmap(confusion_matrix(truth, pred, labels=range(len(names))), annot=True, fmt="d",
                                cmap="Blues", xticklabels=names, yticklabels=names, ax=ax)
                    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix - your file")
                    show_fig(fig)
                counts = out["predicted_emotion"].value_counts().reindex(names).fillna(0)
                fig, ax = plt.subplots(figsize=(7, 3))
                counts.plot.bar(ax=ax, color="slateblue")
                ax.set(ylabel="Images", title="Predicted emotion distribution")
                plt.xticks(rotation=30)
                show_fig(fig)
                st.dataframe(out.head(300))
                df_download(out, "⬇️ Download predictions", "emotion_predictions.csv", "clf_dl")

                st.markdown("**Look at an individual image**")
                idx = st.number_input("Row number", 0, len(df) - 1, 0, key="clf_row")
                label = f"{EMOJI.get(names[pred[idx]], '')} {names[pred[idx]]}"
                if has_truth:
                    label += f"  (actual: {names[truth[idx]]})"
                show_fig(emotion_figure(X[idx].reshape(IMG, IMG), label))

    with t_img:
        st.caption("Works best with a close-up, front-facing face. The photo is centre-cropped, converted to "
                   "grayscale and shrunk to 48×48 - the same format the model was trained on.")
        up = st.file_uploader("Browse for a photo", type=["png", "jpg", "jpeg", "bmp", "webp"], key="clf_pred_img")
        if up is not None:
            img = Image.open(up).convert("L")
            small = ImageOps.fit(img, (IMG, IMG))
            X = np.asarray(small, dtype=np.float32).reshape(1, -1) / 255.0
            pred, proba = clf_predict(bundle, X)
            name = names[int(pred[0])]
            c1, c2 = st.columns([1, 2])
            c1.image(img, caption="Your photo", width=220)
            c1.image(small.resize((192, 192), Image.NEAREST), caption="What the model sees (48×48)")
            c2.markdown(f"### {EMOJI.get(name, '')} {name}")
            if proba is not None:
                pdf = pd.Series(proba[0], index=names).sort_values()
                fig, ax = plt.subplots(figsize=(6, 3.2))
                pdf.plot.barh(ax=ax, color="teal")
                ax.set(xlim=(0, 1), xlabel="Probability")
                with c2:
                    show_fig(fig)


# =====================================================================================
# 3) CLUSTERING - earthquakes
# =====================================================================================
QUAKE_FEATURES = ["x", "y", "z", "log_depth"]


@st.cache_data(show_spinner=False)
def load_bundled_quakes():
    return pd.read_csv(QUAKE_PATH)


def prep_quakes(df):
    d = match_columns(df, ["Latitude", "Longitude", "Depth", "Magnitude", "Type"]).copy()
    if "Type" in d.columns:
        d = d[d["Type"] == "Earthquake"]
    for c in ["Latitude", "Longitude", "Depth"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Latitude", "Longitude", "Depth"]).reset_index(drop=True)
    d["Depth"] = d["Depth"].clip(lower=0)
    return add_quake_features(d)


def add_quake_features(d):
    lat, lon = np.radians(d["Latitude"]), np.radians(d["Longitude"])
    d["x"], d["y"], d["z"] = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    d["log_depth"] = np.log1p(d["Depth"])
    return d


def cluster_metrics(X, labels):
    if len(set(labels)) < 2:
        return {"Silhouette (↑)": np.nan, "Davies-Bouldin (↓)": np.nan, "Calinski-Harabasz (↑)": np.nan}
    return {
        "Silhouette (↑)": silhouette_score(X, labels, sample_size=min(6000, len(X)), random_state=RANDOM_STATE),
        "Davies-Bouldin (↓)": davies_bouldin_score(X, labels),
        "Calinski-Harabasz (↑)": calinski_harabasz_score(X, labels),
    }


@st.cache_resource(show_spinner=False)
def fit_clusterer(data_id, algo, k, _df):
    q = prep_quakes(_df)
    idx = np.arange(len(q))
    if len(q) >= 500:
        tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    else:
        tr, te = idx, np.array([], dtype=int)
    scaler = StandardScaler().fit(q.loc[tr, QUAKE_FEATURES])
    Xtr = scaler.transform(q.loc[tr, QUAKE_FEATURES])
    Xte = scaler.transform(q.loc[te, QUAKE_FEATURES]) if len(te) else np.empty((0, 4))

    labels = np.zeros(len(q), dtype=int)
    if algo == "K-Means":
        est = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xtr)
        ltr = est.labels_
    else:  # Agglomerative (Ward) on a 6,000-point sample, extended to all points with 15-NN (as in the notebook)
        rng = np.random.RandomState(RANDOM_STATE)
        s_idx = rng.choice(len(Xtr), size=min(6000, len(Xtr)), replace=False)
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Xtr[s_idx])
        est = KNeighborsClassifier(n_neighbors=15).fit(Xtr[s_idx], agg.labels_)
        ltr = est.predict(Xtr)
        ltr[s_idx] = agg.labels_
    labels[tr] = ltr
    lte = est.predict(Xte) if len(te) else np.array([], dtype=int)
    labels[te] = lte
    q["Cluster"] = labels

    metrics = {"train": cluster_metrics(Xtr, ltr)}
    if len(te):
        metrics["test"] = cluster_metrics(Xte, lte)

    agg_spec = {"events": ("Depth", "size"), "median_lat": ("Latitude", "median"), "median_lon": ("Longitude", "median"),
                "median_depth_km": ("Depth", "median"), "pct_deeper_300km": ("Depth", lambda s: (s > 300).mean() * 100)}
    if "Magnitude" in q.columns:
        agg_spec["mean_magnitude"] = ("Magnitude", "mean")
    profile = q.groupby("Cluster").agg(**agg_spec).round(2)
    return {"q": q, "scaler": scaler, "est": est, "algo": algo, "k": k, "metrics": metrics, "profile": profile,
            "pca": PCA(n_components=2, random_state=RANDOM_STATE).fit(Xtr), "Xall": scaler.transform(q[QUAKE_FEATURES])}


@st.cache_resource(show_spinner=False)
def scan_k(data_id, _df):
    q = prep_quakes(_df)
    X = StandardScaler().fit_transform(q[QUAKE_FEATURES])
    rows = []
    for k in range(2, 16):
        km = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE).fit(X)
        rows.append({"k": k, "Inertia": km.inertia_, **cluster_metrics(X, km.labels_)})
    return pd.DataFrame(rows).set_index("k")


def clustering_tab():
    st.header("🌍 Clustering - Earthquake Zones")
    st.caption("Groups earthquakes by position on the globe (3-D unit-sphere coordinates) and log-depth.")

    c1, c2 = st.columns(2)
    with c1:
        source = st.radio("Data to cluster", ["Use project dataset (database.csv)", "Browse my own CSV…"],
                          key="cl_source")
    df_raw, data_id = None, None
    with c2:
        if source.startswith("Use"):
            if QUAKE_PATH.exists():
                df_raw, data_id = load_bundled_quakes(), "bundled-quakes"
                st.success(f"Loaded `{QUAKE_PATH.name}` - {len(df_raw):,} events")
            else:
                st.error("data/earthquake/database.csv not found. Choose 'Browse my own CSV…'.")
        else:
            up = st.file_uploader("Select a CSV from your computer", type=["csv"], key="cl_csv")
            st.caption("Needs columns: Latitude, Longitude, Depth (optional: Magnitude, Type)")
            if up is not None:
                raw = read_csv_bytes(up.getvalue())
                chk = match_columns(raw, ["Latitude", "Longitude", "Depth"])
                missing = [c for c in ["Latitude", "Longitude", "Depth"] if c not in chk.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    df_raw, data_id = raw, f"upload-{file_id(up)}"
                    st.success(f"Loaded {up.name} - {len(raw):,} rows")
    if df_raw is None:
        return

    s1, s2 = st.columns([1, 2])
    algo = s1.radio("Algorithm", ["K-Means", "Agglomerative (Ward)"], key="cl_algo")
    k = s2.slider("Number of clusters (k)", 2, 15, 9, key="cl_k")
    with st.spinner("Clustering…"):
        res = fit_clusterer(data_id, "K-Means" if algo == "K-Means" else "Agglomerative", k, df_raw)
    q, profile = res["q"], res["profile"]
    st.markdown(f"**{len(q):,} events** clustered into **{k}** groups with {algo}.")

    t_map, t_new, t_k, t_tbl = st.tabs(
        ["🗺️ Cluster map", "📍 Locate a new earthquake", "📈 Choose k", "📋 Cluster profiles & export"])
    cmap = "tab10" if k <= 10 else "tab20"

    with t_map:
        mt = pd.DataFrame(res["metrics"]).T
        st.dataframe(mt.rename(index={"train": "Training points", "test": "Held-out test points"}).style.format("{:.3f}"))
        st.caption("Silhouette is estimated on a random 6,000-point sample for speed.")
        fig, ax = plt.subplots(figsize=(12, 5.5))
        sc = ax.scatter(q["Longitude"], q["Latitude"], c=q["Cluster"], cmap=cmap, s=3, alpha=0.7)
        ax.legend(*sc.legend_elements(), title="Cluster", fontsize=8, loc="lower left", ncol=3)
        ax.set(xlim=(-180, 180), ylim=(-80, 90), xlabel="Longitude (°)", ylabel="Latitude (°)",
               title=f"{algo} clusters by location")
        show_fig(fig)
        P = res["pca"].transform(res["Xall"])
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(P[:, 0], P[:, 1], c=q["Cluster"], cmap=cmap, s=3, alpha=0.6)
        ax.legend(*sc.legend_elements(), title="Cluster", fontsize=8, ncol=2)
        ev = res["pca"].explained_variance_ratio_ * 100
        ax.set(xlabel=f"PC1 ({ev[0]:.0f}%)", ylabel=f"PC2 ({ev[1]:.0f}%)", title="PCA 2-D view")
        show_fig(fig)

    with t_new:
        st.write("Enter an earthquake's position and depth to see which cluster it falls into.")
        a, b, c = st.columns(3)
        lat = a.number_input("Latitude (°)", -90.0, 90.0, 35.0, step=0.5, key="cl_lat")
        lon = b.number_input("Longitude (°)", -180.0, 180.0, 140.0, step=0.5, key="cl_lon")
        dep = c.number_input("Depth (km)", 0.0, 800.0, 30.0, step=5.0, key="cl_dep")
        if st.button("Find cluster", type="primary", key="cl_find"):
            ev = add_quake_features(pd.DataFrame({"Latitude": [lat], "Longitude": [lon], "Depth": [dep]}))
            cid = int(res["est"].predict(res["scaler"].transform(ev[QUAKE_FEATURES]))[0])
            row = profile.loc[cid]
            st.success(f"This event belongs to **Cluster {cid}**")
            desc = (f"Cluster {cid} has {int(row['events']):,} events, centred near "
                    f"({row['median_lat']:.1f}°, {row['median_lon']:.1f}°), median depth {row['median_depth_km']:.0f} km")
            if "mean_magnitude" in row:
                desc += f", mean magnitude {row['mean_magnitude']:.2f}"
            st.write(desc + ".")
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.scatter(q["Longitude"], q["Latitude"], c="lightgrey", s=2)
            m = q["Cluster"] == cid
            ax.scatter(q.loc[m, "Longitude"], q.loc[m, "Latitude"], c="tab:blue", s=4, label=f"Cluster {cid}")
            ax.scatter([lon], [lat], c="red", marker="*", s=300, edgecolor="k", label="Your earthquake", zorder=5)
            ax.set(xlim=(-180, 180), ylim=(-80, 90), xlabel="Longitude (°)", ylabel="Latitude (°)")
            ax.legend()
            show_fig(fig)

    with t_k:
        st.write("Scan k = 2…15 with K-Means to see where the curve bends (elbow) and where the scores peak.")
        if st.button("Run k scan (≈20-60 s)", key="cl_scan_btn"):
            st.session_state["cl_scan"] = data_id
        if st.session_state.get("cl_scan") == data_id:
            with st.spinner("Scanning k…"):
                ks = scan_k(data_id, df_raw)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(ks.index, ks["Inertia"], "o-")
            axes[0].axvline(k, color="red", ls="--", label=f"Selected k = {k}")
            axes[0].set(title="Elbow curve", xlabel="k", ylabel="Inertia")
            axes[0].legend()
            axes[1].plot(ks.index, ks["Silhouette (↑)"], "o-", label="Silhouette (↑)")
            axes[1].plot(ks.index, ks["Davies-Bouldin (↓)"], "s-", label="Davies-Bouldin (↓)")
            axes[1].axvline(k, color="red", ls="--")
            axes[1].set(title="Silhouette & Davies-Bouldin", xlabel="k")
            axes[1].legend()
            show_fig(fig)
            st.dataframe(ks.round(3))

    with t_tbl:
        st.dataframe(profile)
        fig, ax = plt.subplots(figsize=(7, 3))
        profile["events"].plot.bar(ax=ax, color="indianred")
        ax.set(xlabel="Cluster", ylabel="Events", title="Cluster sizes")
        plt.xticks(rotation=0)
        show_fig(fig)
        export_cols = [c for c in ["Date", "Time", "Latitude", "Longitude", "Depth", "Magnitude", "Cluster"] if c in q.columns]
        df_download(q[export_cols], "⬇️ Download events with cluster labels", "clustered_events.csv", "cl_dl")


# =====================================================================================
# Page layout
# =====================================================================================
st.title("🤖 ML Capstone Studio")
st.caption("Regression · Classification · Clustering - choose a tab, browse your own CSV from your computer, and see results instantly.")

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Pick a tab at the top.\n"
        "2. Use the bundled data **or** click *Browse files* to pick a CSV from File Explorer.\n"
        "3. Train / run, then explore the results and download them."
    )
    st.divider()
    st.markdown(
        "**Datasets**\n"
        "- 🌾 `yield_df.csv` - crop yield\n"
        "- 😀 `fer2013.csv` - bring your own (not in the repo)\n"
        "- 🌍 `database.csv` - earthquakes"
    )

tab_reg, tab_clf, tab_clu = st.tabs(["🌾 Regression", "😀 Classification", "🌍 Clustering"])
with tab_reg:
    regression_tab()
with tab_clf:
    classification_tab()
with tab_clu:
    clustering_tab()
