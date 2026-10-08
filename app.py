import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================
# Chargement des modèles
# =========================

model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")


# =========================
# Correspondance des clusters
# =========================

noms_segments = {
    0: "Clients VIP",
    1: "Clients à risque",
    2: "Clients réguliers"
}


# =========================
# Interface
# =========================

st.title("Segmentation des clients")
st.write(
    "Entrez les caractéristiques RFM d'un client "
    "pour déterminer automatiquement son segment."
)


# =========================
# Saisie des données RFM
# =========================

recency = st.number_input(
    "Récence (nombre de jours depuis le dernier achat)",
    min_value=0.0,
    value=30.0,
    step=1.0
)

frequency = st.number_input(
    "Fréquence (nombre d'achats)",
    min_value=1.0,
    value=5.0,
    step=1.0
)

monetary = st.number_input(
    "Montant total dépensé",
    min_value=0.0,
    value=1000.0,
    step=100.0
)


# =========================
# Prédiction
# =========================

if st.button("Prédire le segment"):

    # Création du nouveau client
    nouveau_client = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    # Même transformation que pendant l'entraînement
    nouveau_client_log = np.log1p(nouveau_client)

    # Standardisation
    nouveau_client_scaled = scaler.transform(nouveau_client_log)

    # Prédiction
    cluster_predit = model.predict(nouveau_client_scaled)[0]

    # Nom du segment
    nom_segment = noms_segments[cluster_predit]


    # =========================
    # Affichage du résultat
    # =========================

    st.subheader("Résultat de la prédiction")

    st.success(f"Segment : {nom_segment}")

    st.write(f"**Cluster prédit :** {cluster_predit}")

    st.subheader("Caractéristiques RFM du client")

    st.write(f"**Récence :** {recency} jours")
    st.write(f"**Fréquence :** {frequency} achats")
    st.write(f"**Montant :** {monetary:.2f}")