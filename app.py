"""
TP Deep Learning – Application Streamlit complète
====================================================
Partie 1 : census-data2015.csv – Modèles classiques
Partie 2 : bank-full.csv – Réseau de neurones (Keras)
Partie 3 : Fashion MNIST – Deep Learning (CNN, LeNet, etc.)

Par : Hadja Nafiesatou – 21L545 – Info4
Sous la supervision de Stéphane C. K. TÉKOUABOU (PhD & Ing.) - UY1 2025-2026
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import time

# ============================================================
# CONFIGURATION PAGE & THÈME
# ============================================================
st.set_page_config(page_title="TP Deep Learning", page_icon="🧠", layout="wide")

# CSS amélioré
st.markdown("""
<style>
    .stApp {
        background-color: #F8F9FA;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #7B3FF2 !important;
    }
    .stButton > button {
        background-color: #FF8C00;
        color: white;
        border: none;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #E67E22;
        color: white;
    }
    .stMetric .label {
        color: #7B3FF2;
        font-weight: bold;
    }
    .stMetric .value {
        color: #FF8C00;
        font-size: 1.8rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #7B3FF2;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #FF8C00;
    }
    hr {
        border-color: #FF8C00;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stExpander"] details summary p {
        font-weight: bold;
        color: #7B3FF2;
    }
</style>
""", unsafe_allow_html=True)

st.title("TP Deep Learning – Classification & Deep Learning")
st.markdown("### Hadja Nafiesatou – matricule 21L545 – Info4")
st.markdown("---")

# ============================================================
# SIDEBAR – Navigation et infos globales
# ============================================================
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
st.sidebar.title("Navigation")
partie = st.sidebar.radio(
    "**Choisissez la partie**",
    ["Partie 1 – Modèles classiques (census-data2015)",
     "Partie 2 – Réseau de neurones (bank-full)",
     "Partie 3 – Deep Learning (Fashion MNIST)"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Données utilisées**\n\n"
    "- **Partie 1** : `census-data2015.csv` (prédiction du revenu)\n"
    "- **Partie 2** : `bank-full.csv` (souscription bancaire)\n"
    "- **Partie 3** : Fashion MNIST (images de vêtements)\n\n"
    "Les fichiers CSV doivent obligatoirement se trouver dans le même répertoire que `app.py`."
)
st.sidebar.markdown("---")
st.sidebar.caption("TP Deep Learning - Info4 - ENS-UY1 2025-2026")

# ============================================================
# CHARGEMENT DES DONNÉES (avec cache)
# ============================================================
@st.cache_data(show_spinner="Chargement de census-data2015.csv...")
def load_census_data():
    df = pd.read_csv("census-data2015.csv")
    # Nettoyage : suppression des colonnes non pertinentes
    df = df.drop(columns=["CensusTract", "State", "County"], errors="ignore")
    df = df.dropna(subset=["Income"])
    median_income = df["Income"].median()
    df["target"] = (df["Income"] > median_income).astype(int)
    df = df.drop(columns=["Income", "IncomeErr"], errors="ignore")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y, median_income

@st.cache_data(show_spinner="Chargement de bank-full.csv...")
def load_bank_data():
    df = pd.read_csv("bank-full.csv", sep=";")
    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y"])
    return X, y

@st.cache_resource(show_spinner="Téléchargement de Fashion MNIST...")
def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return X_train, y_train, X_test, y_test, labels

# ============================================================
# PARTIE 1 : census-data2015.csv – MODÈLES CLASSIQUES
# ============================================================
if partie.startswith("Partie 1"):
    st.header("Partie 1 – Modèles classiques (scikit-learn)")
    st.markdown("**Objectif :** Prédire si le revenu d'un foyer est supérieur à la médiane (`target=1`) à partir des variables socio-démographiques.")
    
    try:
        X_census, y_census, median_income = load_census_data()
        st.success(f"Données census-data2015 chargées : {X_census.shape[0]} lignes, {X_census.shape[1]} features")
        st.metric("Revenu médian (seuil)", f"{median_income:,.0f} $")
    except FileNotFoundError:
        st.error("Fichier `census-data2015.csv` introuvable. Placez-le dans le répertoire courant.")
        st.stop()
    
    @st.cache_data(show_spinner="Prétraitement des données...")
    def preprocess_census(X, y, test_size=0.2, random_state=42):
        X = X.dropna(axis=1, thresh=0.7*len(X))
        cat_cols = X.select_dtypes(include=["object"]).columns
        X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        feature_names = X_encoded.columns.tolist()
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_names, scaler, X_encoded
    
    X_train, X_test, y_train, y_test, features, scaler, X_encoded = preprocess_census(X_census, y_census)
    
    with st.expander("Analyse exploratoire des données"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Instances totales", f"{len(X_census):,}")
        col2.metric("Features (après encodage)", f"{len(features)}")
        col3.metric("Classe positive (revenu>médiane)", f"{y_census.sum():,} ({y_census.mean():.1%})")
        st.subheader("Aperçu des données brutes")
        st.dataframe(X_census.head(10))
        fig_target, ax_target = plt.subplots()
        sns.countplot(x=y_census, ax=ax_target, palette="Purples")
        ax_target.set_title("Distribution de la cible")
        ax_target.set_xticklabels(["≤ médiane", "> médiane"])
        st.pyplot(fig_target)
        num_orig = X_census.select_dtypes(include=[np.number])
        if num_orig.shape[1] > 1:
            st.subheader("Corrélations (variables numériques)")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            sns.heatmap(num_orig.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
            st.pyplot(fig_corr)
    
    st.subheader("Entraînement des modèles classiques")
    model_choice = st.multiselect(
        "**Choisissez les modèles à comparer**",
        ["Régression logistique", "SVM linéaire", "KNN", "Arbre de décision", "Random Forest", "Gradient Boosting", "AdaBoost", "Bagging (arbres)"],
        default=["Régression logistique", "Random Forest", "Gradient Boosting"]
    )
    
    if st.button("Lancer l'entraînement", type="primary"):
        models_dict = {}
        if "Régression logistique" in model_choice:
            models_dict["LogisticRegression"] = LogisticRegression(max_iter=1000, random_state=42)
        if "SVM linéaire" in model_choice:
            models_dict["SVM linéaire"] = SVC(kernel="linear", probability=True, random_state=42)
        if "KNN" in model_choice:
            models_dict["KNN"] = KNeighborsClassifier(n_neighbors=5)
        if "Arbre de décision" in model_choice:
            models_dict["DecisionTree"] = DecisionTreeClassifier(random_state=42)
        if "Random Forest" in model_choice:
            models_dict["RandomForest"] = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
        if "Gradient Boosting" in model_choice:
            models_dict["GradientBoosting"] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        if "AdaBoost" in model_choice:
            models_dict["AdaBoost"] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42)
        if "Bagging (arbres)" in model_choice:
            models_dict["Bagging"] = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=42)
        
        if not models_dict:
            st.warning("Sélectionnez au moins un modèle.")
        else:
            results = {}
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            progress_bar = st.progress(0)
            for i, (name, clf) in enumerate(models_dict.items()):
                cv_acc = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
                cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
                cv_auc = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
                clf.fit(X_train, y_train)
                if hasattr(clf, "predict_proba"):
                    y_proba = clf.predict_proba(X_test)[:, 1]
                else:
                    y_proba = clf.decision_function(X_test) if hasattr(clf, "decision_function") else clf.predict(X_test)
                y_pred = (y_proba >= 0.5).astype(int) if y_proba.ndim == 1 else clf.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred)
                test_auc = roc_auc_score(y_test, y_proba) if y_proba.ndim == 1 else 0
                results[name] = {
                    "CV Accuracy": f"{cv_acc.mean():.4f} (±{cv_acc.std():.4f})",
                    "CV F1": f"{cv_f1.mean():.4f} (±{cv_f1.std():.4f})",
                    "CV AUC": f"{cv_auc.mean():.4f} (±{cv_auc.std():.4f})",
                    "Test Accuracy": f"{test_acc:.4f}",
                    "Test F1": f"{test_f1:.4f}",
                    "Test AUC": f"{test_auc:.4f}",
                    "classifier": clf,
                    "probas": y_proba
                }
                progress_bar.progress((i+1)/len(models_dict))
            st.success("Entraînement terminé !")
            df_res = pd.DataFrame({
                name: {k: v for k, v in res.items() if k not in ["classifier", "probas"]}
                for name, res in results.items()
            }).T
            st.dataframe(df_res.style.highlight_max(axis=0, color="#d4edda"))
            
            st.subheader("Courbes ROC (test set)")
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            for name, res in results.items():
                if res["probas"] is not None and res["probas"].ndim == 1:
                    fpr, tpr, _ = roc_curve(y_test, res["probas"])
                    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={res['Test AUC']})", linewidth=2)
            ax_roc.plot([0,1],[0,1], 'k--')
            ax_roc.set_xlabel("Taux de faux positifs")
            ax_roc.set_ylabel("Taux de vrais positifs")
            ax_roc.set_title("Courbes ROC comparatives")
            ax_roc.legend(fontsize=8)
            ax_roc.grid(alpha=0.3)
            st.pyplot(fig_roc)
            
            st.subheader("Matrices de confusion")
            cols_cm = st.columns(len(results))
            for col, (name, res) in zip(cols_cm, results.items()):
                with col:
                    y_pred = (res["probas"] >= 0.5).astype(int) if res["probas"].ndim == 1 else res["classifier"].predict(X_test)
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                    ConfusionMatrixDisplay(cm, display_labels=["≤ médiane", "> médiane"]).plot(ax=ax_cm, cmap="Purples")
                    ax_cm.set_title(name, fontsize=10)
                    st.pyplot(fig_cm)
            
            st.subheader("Importance des variables")
            for name, res in results.items():
                if hasattr(res["classifier"], "feature_importances_"):
                    importances = res["classifier"].feature_importances_
                    indices = np.argsort(importances)[-15:]
                    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                    ax_imp.barh([features[i] for i in indices], importances[indices], color="#FF8C00")
                    ax_imp.set_title(f"Top 15 – {name}")
                    ax_imp.set_xlabel("Importance")
                    st.pyplot(fig_imp)
    
    with st.expander("Optimisation d'un arbre de décision (GridSearchCV)"):
        st.markdown("Recherche du meilleur `max_depth` par validation croisée 5-folds.")
        depths = list(range(2, 21))
        dt = DecisionTreeClassifier(random_state=42)
        grid = GridSearchCV(dt, param_grid={"max_depth": depths}, cv=5, scoring="roc_auc")
        grid.fit(X_train, y_train)
        best_depth = grid.best_params_["max_depth"]
        st.success(f"Meilleure profondeur : **{best_depth}** (AUC CV = {grid.best_score_:.4f})")
        fig_depth, ax_depth = plt.subplots(figsize=(8, 4))
        ax_depth.plot(depths, grid.cv_results_["mean_test_score"], marker="o", color="#7B3FF2")
        ax_depth.set_xlabel("max_depth")
        ax_depth.set_ylabel("AUC moyenne (CV)")
        ax_depth.set_title("Validation croisée – Arbre de décision")
        ax_depth.grid(alpha=0.3)
        st.pyplot(fig_depth)
        best_dt = grid.best_estimator_
        y_pred_best = best_dt.predict(X_test)
        st.metric("Accuracy sur test (arbre optimisé)", f"{accuracy_score(y_test, y_pred_best):.4f}")
    
    with st.expander("Bagging & Random Forest – Paramètres avancés"):
        st.markdown("**Bagging / Random Forest :** influence du nombre d'arbres et du nombre de features (`max_features`).")
        B = st.slider("Nombre d'arbres (B)", 10, 300, 100, 10, key="bagging_B")
        p = st.slider("Nombre de features par arbre (`max_features`)", 1, len(features), int(np.sqrt(len(features))), key="rf_p")
        if st.button("Entraîner un Random Forest personnalisé"):
            rf = RandomForestClassifier(n_estimators=B, max_features=p, random_state=42, oob_score=True)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
            oob = rf.oob_score_
            st.metric("Test Accuracy", f"{acc:.4f}")
            st.metric("Test AUC", f"{auc:.4f}")
            st.metric("OOB Score", f"{oob:.4f}")
    
    st.info("Cette partie implémente tous les modèles classiques, la validation croisée, la recherche de paramètres, l'importance des variables et les métriques ROC.")

# ============================================================
# PARTIE 2 : bank-full.csv – RÉSEAU DE NEURONES (KERAS)
# ============================================================
elif partie.startswith("Partie 2"):
    st.header("Partie 2 – Réseau de neurones (TensorFlow/Keras)")
    st.markdown("**Objectif :** Prédire si un client souscrira à un dépôt (`y=1`) à partir des données de la campagne bancaire.")
    
    try:
        X_bank, y_bank = load_bank_data()
        st.success(f"Données bank-full chargées : {X_bank.shape[0]} lignes, {X_bank.shape[1]} features")
    except FileNotFoundError:
        st.error("Fichier `bank-full.csv` introuvable.")
        st.stop()
    
    @st.cache_data(show_spinner="Prétraitement des données bancaires...")
    def preprocess_bank(X, y, test_size=0.2, random_state=42):
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        feature_names = X.columns.tolist()
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_names, scaler
    
    X_train, X_test, y_train, y_test, bank_features, bank_scaler = preprocess_bank(X_bank, y_bank)
    st.write(f"**Dimensions train :** {X_train.shape}, **test :** {X_test.shape}")
    
    with st.sidebar:
        st.subheader("Architecture du réseau")
        n_neurons1 = st.slider("Neurones – couche 1", 16, 256, 64, 8)
        n_neurons2 = st.slider("Neurones – couche 2", 8, 128, 32, 8)
        dropout_rate = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
        epochs = st.slider("Époques", 10, 100, 30, 10)
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
        early_stopping = st.checkbox("Early stopping (patience=5)", value=True)
    
    if st.button("Construire et entraîner le MLP", type="primary"):
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(n_neurons1, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(n_neurons2, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
        st.write("**Résumé du modèle**")
        st.text(model.summary())
        
        callback_list = []
        if early_stopping:
            callback_list.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True))
        
        with st.spinner("Entraînement en cours..."):
            history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs,
                                batch_size=batch_size, callbacks=callback_list, verbose=0)
        
        st.success("Entraînement terminé !")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history.history["loss"], label="Train loss", color="#7B3FF2")
        axes[0].plot(history.history["val_loss"], label="Val loss", color="#FF8C00")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(history.history["auc"], label="Train AUC", color="#7B3FF2")
        axes[1].plot(history.history["val_auc"], label="Val AUC", color="#FF8C00")
        axes[1].set_title("AUC")
        axes[1].legend()
        axes[1].grid(True)
        st.pyplot(fig)
        
        y_proba = model.predict(X_test).flatten()
        y_pred = (y_proba >= 0.5).astype(int)
        test_acc = accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba)
        col1, col2 = st.columns(2)
        col1.metric("Test Accuracy", f"{test_acc:.4f}")
        col2.metric("Test AUC", f"{test_auc:.4f}")
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color="#FF8C00", lw=2, label=f"AUC = {test_auc:.4f}")
        ax_roc.plot([0,1],[0,1], 'k--')
        ax_roc.set_title("Courbe ROC – MLP")
        ax_roc.legend()
        st.pyplot(fig_roc)
        
        model.save("bank_tel_mlp.h5")
        st.success("Modèle sauvegardé sous `bank_tel_mlp.h5`")

# ============================================================
# PARTIE 3 : Fashion MNIST – DEEP LEARNING (CNN)
# ============================================================
elif partie.startswith("Partie 3"):
    st.header("Partie 3 – Deep Learning sur Fashion MNIST")
    st.markdown("**Objectif :** Classifier des images de vêtements en 10 catégories à l'aide de réseaux de neurones convolutifs.")
    
    try:
        X_train, y_train, X_test, y_test, labels = load_fashion_mnist()
        st.success(f"Fashion MNIST chargé : train {X_train.shape[0]}, test {X_test.shape[0]}")
        st.write(f"**Classes :** {', '.join(labels)}")
    except Exception as e:
        st.error(f"Erreur de chargement : {e}. Vérifiez votre connexion Internet.")
        st.stop()
    
    with st.expander("Visualisation d'images"):
        fig, axes = plt.subplots(3, 5, figsize=(10, 6))
        for i, ax in enumerate(axes.flat):
            idx = np.random.randint(0, len(X_train))
            ax.imshow(X_train[idx].squeeze(), cmap="gray")
            ax.set_title(labels[y_train[idx]], fontsize=8)
            ax.axis("off")
        st.pyplot(fig)
    
    st.subheader("Entraînement manuel – Choisissez votre architecture et hyperparamètres")
    arch_choice = st.selectbox("Architecture CNN", ["CNN simple (2 conv + pooling)", "LeNet-5 adapté", "ResNet-like simplifié"])
    epochs_cnn = st.slider("Nombre d'époques", 5, 50, 10, 5)
    batch_size_cnn = st.selectbox("Batch size", [32, 64, 128, 256], index=1)
    use_augmentation = st.checkbox("Utiliser l'augmentation de données (rotation/zoom)", value=False)
    use_early_stopping = st.checkbox("Early stopping (patience=3)", value=True)
    
    if st.button("Lancer l'entraînement profond", type="primary"):
        if arch_choice == "CNN simple (2 conv + pooling)":
            model = keras.Sequential([
                layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(64, (3,3), activation="relu"),
                layers.MaxPooling2D((2,2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(10, activation="softmax")
            ])
        elif arch_choice == "LeNet-5 adapté":
            model = keras.Sequential([
                layers.Conv2D(6, (5,5), activation="tanh", input_shape=(28,28,1)),
                layers.AveragePooling2D((2,2)),
                layers.Conv2D(16, (5,5), activation="tanh"),
                layers.AveragePooling2D((2,2)),
                layers.Flatten(),
                layers.Dense(120, activation="tanh"),
                layers.Dense(84, activation="tanh"),
                layers.Dense(10, activation="softmax")
            ])
        else:  # ResNet-like simplifié
            inputs = keras.Input(shape=(28,28,1))
            x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
            x = layers.BatchNormalization()(x)
            shortcut = x
            x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)
            x = layers.Add()([x, shortcut])
            x = layers.MaxPooling2D((2,2))(x)
            x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            shortcut2 = x
            x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
            x = layers.Add()([x, shortcut2])
            x = layers.MaxPooling2D((2,2))(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(128, activation="relu")(x)
            outputs = layers.Dense(10, activation="softmax")(x)
            model = keras.Model(inputs, outputs)
        
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        st.write("**Résumé du modèle**")
        st.text(model.summary())
        
        if use_augmentation:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1
            )
            train_gen = datagen.flow(X_train, y_train, batch_size=batch_size_cnn)
            validation_data = (X_test, y_test)
            steps_per_epoch = len(X_train) // batch_size_cnn
        else:
            train_gen = (X_train, y_train)
            validation_data = (X_test, y_test)
            steps_per_epoch = None
        
        callback_list = []
        if use_early_stopping:
            callback_list.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True))
        
        with st.spinner("Entraînement en cours (cela peut prendre plusieurs minutes)..."):
            start_time = time.time()
            if use_augmentation:
                history = model.fit(train_gen, steps_per_epoch=steps_per_epoch,
                                    validation_data=validation_data,
                                    epochs=epochs_cnn, callbacks=callback_list, verbose=0)
            else:
                history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                    batch_size=batch_size_cnn, epochs=epochs_cnn,
                                    callbacks=callback_list, verbose=0)
            elapsed = time.time() - start_time
        
        st.success(f"Entraînement terminé en {elapsed:.1f} secondes.")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history.history["loss"], label="Train")
        axes[0].plot(history.history["val_loss"], label="Validation")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(history.history["accuracy"], label="Train")
        axes[1].plot(history.history["val_accuracy"], label="Validation")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True)
        st.pyplot(fig)
        
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        st.metric("Test Accuracy", f"{test_acc:.4f}")
        
        y_pred = np.argmax(model.predict(X_test), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels, ax=ax_cm)
        ax_cm.set_title("Matrice de confusion – Fashion MNIST")
        st.pyplot(fig_cm)
        
        st.subheader("Exemples de prédictions")
        indices = np.random.choice(len(X_test), 12, replace=False)
        fig, axes = plt.subplots(3, 4, figsize=(10, 8))
        for i, idx in enumerate(indices):
            ax = axes[i//4, i%4]
            ax.imshow(X_test[idx].squeeze(), cmap="gray")
            pred = labels[y_pred[idx]]
            true = labels[y_test[idx]]
            color = "green" if pred == true else "red"
            ax.set_title(f"P: {pred}\nV: {true}", fontsize=8, color=color)
            ax.axis("off")
        st.pyplot(fig)
        
        model.save("fashion_mnist_cnn.h5")
        st.success("Modèle sauvegardé sous `fashion_mnist_cnn.h5`")
    
    st.info("Vous pouvez modifier l'architecture et les hyperparamètres pour comparer les performances.")

# ============================================================
# PIED DE PAGE
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption("Réalisé par Hadja Nafiesatou 21L545")
st.sidebar.caption("Sous la supervision de Stéphane C. K. TÉKOUABOU (PhD & Ing.) - UY1 2025-2026")
st.caption("TP Deep Learning – Réalisé avec Streamlit, Scikit-learn, TensorFlow/Keras")