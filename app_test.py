# ============================================================
# app.py
# Application web de prédiction et recommandation académique
# Modèle : scores factoriels + régression logistique
# ============================================================

import os
import sys
import __main__

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# 1. Classe nécessaire pour recharger le modèle sauvegardé
# ============================================================

def varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0

    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)

        u, s, vh = np.linalg.svd(
            np.dot(
                Phi.T,
                Lambda ** 3
                - (gamma / p)
                * np.dot(
                    Lambda,
                    np.diag(np.diag(np.dot(Lambda.T, Lambda)))
                )
            )
        )

        R = np.dot(u, vh)
        d = np.sum(s)

        if d_old != 0 and d / d_old < 1 + tol:
            break

    return np.dot(Phi, R), R


class PCAVarimaxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)

        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(X)

        self.pca_ = PCA(n_components=self.n_components, random_state=42)
        self.pca_.fit(Z)

        loadings = self.pca_.components_.T * np.sqrt(
            self.pca_.explained_variance_
        )

        self.rotated_loadings_, self.rotation_matrix_ = varimax(loadings)

        return self

    def transform(self, X):
        Z = self.scaler_.transform(X)
        pca_scores = self.pca_.transform(Z)
        rotated_scores = pca_scores @ self.rotation_matrix_
        return rotated_scores

    def get_loading_table(self):
        return pd.DataFrame(
            self.rotated_loadings_,
            index=self.feature_names_in_,
            columns=["D1", "D2", "D3", "D4"]
        )


setattr(__main__, "PCAVarimaxTransformer", PCAVarimaxTransformer)
sys.modules["__main__"].PCAVarimaxTransformer = PCAVarimaxTransformer


# ============================================================
# 2. Chargement du modèle
# ============================================================

@st.cache_resource
def load_artifact():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "modele_application_performance.pkl"
    )
    return joblib.load(model_path)


artifact = load_artifact()

model = artifact["model"]
features = artifact["features"]
dimension_names = artifact["dimension_names"]
threshold = artifact["threshold"]
metrics = artifact["metrics"]


# ============================================================
# 3. Configuration de la page
# ============================================================

st.set_page_config(
    page_title="Prédiction de la performance académique",
    page_icon="🎓",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    h1, h2, h3 {
        color: #0f172a;
    }

    div[data-testid="stSelectbox"] label p {
        font-weight: 800 !important;
        color: #111827 !important;
        font-size: 16px !important;
    }

    div[data-testid="stCaptionContainer"] {
        color: #64748b !important;
        font-size: 14px !important;
        margin-top: -8px !important;
        margin-bottom: 10px !important;
    }

    .stButton {
        display: flex;
        justify-content: center;
    }

    .stButton > button {
        background-color: #2495ee !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        width: 60% !important;
        min-height: 58px !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        color: white !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
    }

    .stButton > button:active {
        transform: scale(0.98);
    }

    .prediction-result-box {
        background: linear-gradient(135deg, #2495ee, #1d4ed8);
        color: white;
        border-radius: 16px;
        padding: 22px 28px;
        margin: 18px 0 28px 0;
        font-size: 22px;
        font-weight: 900;
        text-align: center;
        box-shadow: 0 8px 22px rgba(37, 99, 235, 0.30);
        line-height: 1.5;
    }

    .metric-label-custom {
        font-weight: 900;
        font-size: 17px;
        color: #0f172a;
        margin-top: 18px;
        margin-bottom: 4px;
    }

    .metric-value-custom {
        font-size: 42px;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 20px;
    }

    @media screen and (max-width: 768px) {
        .stButton > button {
            width: 100% !important;
            font-size: 16px !important;
        }

        .prediction-result-box {
            font-size: 18px;
            padding: 18px 18px;
        }

        .metric-value-custom {
            font-size: 36px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎓 Système de prédiction et recommandation académique")

st.write(
    "Cette application utilise une analyse factorielle exploratoire et une "
    "régression logistique pour prédire la performance académique d’un étudiant."
)

with st.expander("Performance du modèle utilisé"):
    st.write(pd.DataFrame({
        "Indicateur": list(metrics.keys()),
        "Valeur": [round(v, 3) for v in metrics.values()]
    }))


# ============================================================
# 4. Formulaire de saisie avec signification des valeurs encodées
# ============================================================

st.header("1. Informations de l’étudiant")

st.info(
    "Sélectionnez les réponses correspondant au profil de l’étudiant. "
    "Les valeurs encodées utilisées par le modèle sont conservées automatiquement."
)

OPTIONS_LABELS = {
    "SX": {
        0: "Féminin",
        1: "Masculin",
    },
    "AG": {
        0: "Moins de 20 ans",
        1: "20 – 25 ans",
        2: "Plus de 25 ans",
    },
    "SERIE_BAC": {
        0: "D",
        1: "C",
        2: "Technique",
    },
    "MENT_BAC": {
        0: "Passable",
        1: "Assez Bien",
        2: "Bien",
        3: "Très Bien",
    },
    "MOT_MAST": {
        0: "Par défaut (pas d’autre choix)",
        1: "Par conseil / pression familiale",
        2: "Par débouchés personnels",
        3: "Par intérêt personnel",
    },
    "EMP_ETU": {
        0: "Non",
        1: "Oui",
    },
    "CLUB_ASS": {
        0: "Non",
        1: "Oui",
    },
    "TMP_ETU": {
        0: "Moins de 1h",
        1: "1 à 2h",
        2: "2 à 3h",
        3: "Plus de 3h",
    },
    "TMP_LOIS": {
        0: "Moins de 1h",
        1: "1 à 2h",
        2: "2 à 3h",
        3: "Plus de 3h",
    },
    "COURS_SUP": {
        0: "Non",
        1: "Oui",
    },
    "TRAV_GRP": {
        0: "Non",
        1: "Oui",
    },
    "MAT_REDOUB": {
        0: "Aucune",
        1: "1 ou 2",
        2: "3 ou 4",
        3: "5 ou 6",
        4: "7 ou 8",
        5: "9 ou plus",
    },
    "NB_FR-SR": {
        0: "Aucun",
        1: "1 ou 2",
        2: "3 ou plus",
    },
    "NIV_PERE": {
        0: "Sans instruction",
        1: "Primaire",
        2: "Secondaire",
        3: "Supérieur",
    },
    "NIV_MERE": {
        0: "Sans instruction",
        1: "Primaire",
        2: "Secondaire",
        3: "Supérieur",
    },
    "STAT_PARENT": {
        0: "Divorcée",
        1: "Non légitime",
        2: "Légitime",
    },
    "SIT_ECO": {
        0: "Faible",
        1: "Moyenne",
        2: "Bonne",
        3: "Très bonne",
    },
    "LIEU_RES": {
        0: "Rural",
        1: "Périurbain",
        2: "Urbain",
    },
    "PRI_TACH": {
        0: "Faible",
        1: "Moyenne",
        2: "Forte",
    },
    "INFO": {
        0: "Faible",
        1: "Moyen",
        2: "Avancée",
    },
    "ID_NUM": {
        0: "Faible",
        1: "Moyenne",
        2: "Forte",
        3: "Très forte",
    },
    "NTIC": {
        0: "Faible",
        1: "Moyen",
        2: "Fort",
        3: "Très fort",
    },
    "IA": {
        0: "Débutant",
        1: "Intermédiaire",
        2: "Avancé",
        3: "Expert",
    },
    "COM_SC": {
        0: "Faible",
        1: "Moyenne",
        2: "Forte",
        3: "Très forte",
    },
}

QUESTION_LABELS = {
    "SX": "1. Sexe",
    "AG": "2. Age",
    "SERIE_BAC": "3. Série du Baccalauréat",
    "MENT_BAC": "4. Mention au Baccalauréat",
    "MOT_MAST": "5. Motivation pour le choix du Master",
    "EMP_ETU": "6. Emploi à côté des études",
    "CLUB_ASS": "7. Club ou association étudiante",
    "TMP_ETU": "8. Temps moyen consacré aux études personnelles par jour",
    "TMP_LOIS": "9. Temps moyen consacré aux loisirs et réseaux sociaux par jour",
    "COURS_SUP": "10. Suivi de cours supplémentaire",
    "TRAV_GRP": "11. Travail en groupe",
    "MAT_REDOUB": "12. Nombre de matières redoublées durant le cursus universitaire",
    "NB_FR-SR": "13. Nombre de frères et sœurs",
    "NIV_PERE": "14. Niveau académique du père",
    "NIV_MERE": "15. Niveau académique de la mère",
    "STAT_PARENT": "16. Statut des parents",
    "SIT_ECO": "17. Situation économique de la famille",
    "LIEU_RES": "18. Lieu de résidence familiale",
    "PRI_TACH": "19. Priorisation des tâches",
    "INFO": "20. Informatique et programmation",
    "ID_NUM": "21. Identité numérique",
    "NTIC": "22. Nouvelles Technologies de l’Information et de la Communication",
    "IA": "23. Compétences numériques et IA",
    "COM_SC": "24. Communication scientifique",
}


def select_encoded_variable(variable_code):
    options = list(OPTIONS_LABELS[variable_code].keys())

    selected_value = st.selectbox(
        label=f"**{QUESTION_LABELS[variable_code]}**",
        options=options,
        index=None,
        placeholder="Sélectionnez...",
        format_func=lambda value: OPTIONS_LABELS[variable_code][value],
        key=variable_code
    )

    if selected_value is not None:
        st.caption(f"Valeur encodée : {selected_value}")

    return selected_value


input_data = {}

col1, col2, col3 = st.columns(3)

with col1:
    input_data["SX"] = select_encoded_variable("SX")
    input_data["AG"] = select_encoded_variable("AG")
    input_data["SERIE_BAC"] = select_encoded_variable("SERIE_BAC")
    input_data["MENT_BAC"] = select_encoded_variable("MENT_BAC")
    input_data["MOT_MAST"] = select_encoded_variable("MOT_MAST")
    input_data["EMP_ETU"] = select_encoded_variable("EMP_ETU")
    input_data["CLUB_ASS"] = select_encoded_variable("CLUB_ASS")
    input_data["TMP_ETU"] = select_encoded_variable("TMP_ETU")

with col2:
    input_data["TMP_LOIS"] = select_encoded_variable("TMP_LOIS")
    input_data["COURS_SUP"] = select_encoded_variable("COURS_SUP")
    input_data["TRAV_GRP"] = select_encoded_variable("TRAV_GRP")
    input_data["MAT_REDOUB"] = select_encoded_variable("MAT_REDOUB")
    input_data["NB_FR-SR"] = select_encoded_variable("NB_FR-SR")
    input_data["NIV_PERE"] = select_encoded_variable("NIV_PERE")
    input_data["NIV_MERE"] = select_encoded_variable("NIV_MERE")
    input_data["STAT_PARENT"] = select_encoded_variable("STAT_PARENT")

with col3:
    input_data["SIT_ECO"] = select_encoded_variable("SIT_ECO")
    input_data["LIEU_RES"] = select_encoded_variable("LIEU_RES")
    input_data["PRI_TACH"] = select_encoded_variable("PRI_TACH")
    input_data["INFO"] = select_encoded_variable("INFO")
    input_data["ID_NUM"] = select_encoded_variable("ID_NUM")
    input_data["NTIC"] = select_encoded_variable("NTIC")
    input_data["IA"] = select_encoded_variable("IA")
    input_data["COM_SC"] = select_encoded_variable("COM_SC")


# ============================================================
# 5. Fonction de prédiction
# ============================================================

def predict_student(input_data):
    X_new = pd.DataFrame([input_data])

    try:
        X_new = X_new[features]
    except KeyError as e:
        st.error(f"Colonnes manquantes par rapport au modèle : {e}")
        st.stop()

    predicted_class = int(model.predict(X_new)[0])
    probability_class_1 = float(model.predict_proba(X_new)[0, 1])

    factorial = model.named_steps["factorial"]
    logistic = model.named_steps["logistic"]

    factor_scores = factorial.transform(X_new)

    factor_df = pd.DataFrame(
        factor_scores,
        columns=["D1", "D2", "D3", "D4"]
    )

    coefficients = logistic.coef_[0]
    intercept = logistic.intercept_[0]

    contributions = coefficients * factor_scores[0]

    contribution_df = pd.DataFrame({
        "Dimension": ["D1", "D2", "D3", "D4"],
        "Nom scientifique": [
            dimension_names["D1"],
            dimension_names["D2"],
            dimension_names["D3"],
            dimension_names["D4"]
        ],
        "Score factoriel": factor_scores[0],
        "Coefficient logistique": coefficients,
        "Contribution au logit": contributions
    })

    logit_score = intercept + contributions.sum()

    return {
        "predicted_class": predicted_class,
        "probability_class_1": probability_class_1,
        "factor_df": factor_df,
        "contribution_df": contribution_df,
        "logit_score": logit_score,
        "intercept": intercept
    }


# ============================================================
# 6. Fonction de recommandation
# ============================================================

def generate_recommendations(result):
    predicted_class = result["predicted_class"]
    contribution_df = result["contribution_df"].copy()

    recommendations = []

    if predicted_class == 1:
        recommendations.append({
            "Priorité": "Maintien",
            "Dimension": "Profil global",
            "Recommandation": (
                "Le profil est prédit comme favorable à une performance élevée. "
                "Il est recommandé de maintenir les méthodes actuelles, de continuer "
                "à développer les compétences numériques, et de conserver une organisation régulière du travail."
            )
        })
        return pd.DataFrame(recommendations)

    negative_dims = contribution_df[
        contribution_df["Contribution au logit"] < 0
    ].copy()

    negative_dims["Impact négatif"] = negative_dims[
        "Contribution au logit"
    ].abs()

    negative_dims = negative_dims.sort_values(
        "Impact négatif",
        ascending=False
    )

    if negative_dims.empty:
        recommendations.append({
            "Priorité": "Priorité générale",
            "Dimension": "Capital cognitivo-numérique et académique",
            "Recommandation": (
                "La probabilité reste inférieure au seuil malgré l’absence de contribution fortement négative. "
                "Il est recommandé de renforcer en priorité le capital cognitivo-numérique et académique, "
                "car cette dimension possède le coefficient logistique le plus élevé."
            )
        })
        return pd.DataFrame(recommendations)

    priority_rank = 1

    for _, row in negative_dims.iterrows():
        dim = row["Dimension"]

        if dim == "D1":
            rec = (
                "Renforcer le capital cognitivo-numérique et académique : augmenter le temps d’étude personnel, "
                "améliorer les compétences en IA, NTIC, informatique, communication scientifique, "
                "utiliser des outils de planification et réduire les lacunes liées aux matières redoublées."
            )

        elif dim == "D2":
            rec = (
                "Renforcer le soutien socio-familial et les ressources éducatives : utiliser les ressources "
                "universitaires, demander un tutorat, fréquenter la bibliothèque, solliciter un encadrement pédagogique "
                "et chercher un appui académique lorsque les ressources familiales sont limitées."
            )

        elif dim == "D3":
            rec = (
                "Tenir compte du contexte biographique et résidentiel : mieux planifier les déplacements, "
                "organiser un environnement d’étude stable, anticiper les contraintes familiales ou territoriales, "
                "et demander un accompagnement académique si le contexte personnel limite l’apprentissage."
            )

        elif dim == "D4":
            rec = (
                "Améliorer l’engagement stratégique : clarifier la motivation pour le Master, participer à un groupe "
                "de travail, mieux équilibrer emploi et études, suivre des cours complémentaires si nécessaire, "
                "et prioriser les tâches universitaires."
            )

        else:
            rec = "Recommandation générale : renforcer les habitudes d’apprentissage et demander un accompagnement."

        recommendations.append({
            "Priorité": f"Priorité {priority_rank}",
            "Dimension": row["Nom scientifique"],
            "Recommandation": rec
        })

        priority_rank += 1

    return pd.DataFrame(recommendations)


# ============================================================
# 7. Affichage des résultats
# ============================================================

st.header("2. Prédiction")

predict_clicked = st.button("➤ Prédire la performance académique")

if predict_clicked:
    missing_fields = [
        QUESTION_LABELS[key]
        for key, value in input_data.items()
        if value is None
    ]

    if missing_fields:
        st.warning(
            "Veuillez compléter tous les champs avant de lancer la prédiction."
        )

        with st.expander("Champs non complétés"):
            for field in missing_fields:
                st.write(f"- {field}")

        st.stop()

    result = predict_student(input_data)

    predicted_class = result["predicted_class"]
    probability = result["probability_class_1"]

    st.subheader("Résultat de la prédiction")

    if predicted_class == 1:
        result_label = "Classe prédite : 1 → Performance élevée"
    else:
        result_label = "Classe prédite : 0 → Performance moyenne"

    st.markdown(
        f"""
        <div class="prediction-result-box">
            ➤ {result_label}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="metric-label-custom">
            Probabilité de performance élevée
        </div>
        <div class="metric-value-custom">
            {probability:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="metric-label-custom">
            Score logistique
        </div>
        <div class="metric-value-custom">
            {result["logit_score"]:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Scores factoriels")

    factor_display = result["factor_df"].rename(columns=dimension_names)
    st.dataframe(factor_display)

    st.subheader("Contributions des dimensions à la prédiction")

    contribution_df = result["contribution_df"]
    st.dataframe(contribution_df)

    st.subheader("Recommandations personnalisées")

    rec_df = generate_recommendations(result)
    st.dataframe(rec_df)

    for _, row in rec_df.iterrows():
        st.markdown(f"**{row['Priorité']} — {row['Dimension']}**")
        st.write(row["Recommandation"])

    st.subheader("Interprétation")

    if predicted_class == 0:
        st.write(
            "L’étudiant est prédit en performance moyenne. "
            "Les recommandations sont construites à partir des dimensions dont la contribution au logit est négative. "
            "Ces dimensions diminuent la probabilité d’appartenir à la classe de performance élevée."
        )
    else:
        st.write(
            "L’étudiant est prédit en performance élevée. "
            "Aucune recommandation corrective prioritaire n’est nécessaire, mais le maintien des acquis reste recommandé."
        )


# ============================================================
# 8. Méthodologie mathématique affichée dans l’application
# ============================================================

with st.expander("Méthodologie mathématique utilisée"):
    st.markdown(r"""
    L’application applique la chaîne mathématique suivante :

    **1. Standardisation**

    \[
    Z_{new,j}=\frac{X_{new,j}-\mu_j}{\sigma_j}
    \]

    **2. Projection factorielle**

    \[
    D_{new,k}=\sum_{j=1}^{24}a_{jk}Z_{new,j}
    \]

    **3. Régression logistique**

    \[
    P(Y=1)=\frac{1}{1+\exp[-(\theta_0+\theta_1D_1+\theta_2D_2+\theta_3D_3+\theta_4D_4)]}
    \]

    **4. Classification**

    \[
    \hat{Y}=1 \text{ si } P(Y=1)\geq0.5, \text{ sinon } \hat{Y}=0
    \]

    **5. Contribution des dimensions**

    \[
    C_k=\theta_kD_k
    \]

    Une dimension est considérée comme défavorable si :

    \[
    C_k<0
    \]

    Les recommandations sont priorisées ARAKA IRETO :

    \[
    Impact_k=|C_k|
    \]
    """)