import streamlit as st
import pandas as pd

from tools import *


def display_homepage() -> None:
    """
    This function displays the homepage of the dashboard.

    Returns
    -------
    None
    """
    st.title(
        "Implémentation d'un dashboard de prédiction du risque de défaut de \
                paiement"
    )

    st.subheader("Bienvenue sur le dashboard de l'application !")

    st.markdown(
        "Vous pouvez utiliser le menu de navigation sur la gauche pour naviguer \
                entre les différentes pages de l'application."
    )

    st.markdown(
        "Vous pouvez également consulter le code source de l'application sur  \
                [Github]()"
    )

    st.markdown(
        "Pour plus d'informations sur le projet, vous pouvez consulter le  \
                [rapport]()"
    )

    st.markdown("par Adnene Guessoum")


def display_about_clients(df: pd.DataFrame) -> None:
    st.title("Analyse exploratoire des données clients:")

    st.write("--------------------------------------------------")
    st.title("Observer les données non-filtrées :")

    informations_data(df)
    visualisation_distribution_target(df)

    agree = st.checkbox("Observer quelques graphiques disponibles ?")
    if agree:
        visualisation_univar(df)

    st.write("--------------------------------------------------")
    st.title("Observer les données filtrées :")

    df_filtered = filter_client(df)

    display_filtered_client_visualisation(df_filtered)

    agree_2 = st.checkbox("Observer les changements ?")
    if agree_2:
        visualisation_univar(df_filtered)


def display_about_model() -> None:
    st.title("Comprendre le modèle de score-crédit:")
    st.markdown("Informations sur le modèle choisie:")

    # nombre de client pour sub_sample:
    nb_input = st.text_input(
        "Combien de clients voulez-vous tirer au sort pour observation ?",
    )

    if nb_input.isdigit():
        with st.spinner("Chargement des caractéristiques globales du modèle..."):
            interpretation_global(int(nb_input))
        st.success("Done!")
    else:
        st.write("Veuillez saisir un nombre (raisonnable).")


def display_predict_page() -> None:
    st.title("Prédire et expliquer le risque de défaut d'un client:")
    st.markdown("Analyse des résultats de prédiction d'offre de crédit:")

    # choix du client:
    id_input = st.text_input(
        "identifiant client:",
    )

    if id_input == "":
        st.write("Veuillez saisir un identifiant.")

    elif int(id_input) not in liste_id:
        st.write("Veuillez vérifier si l'identifiant saisie est correct.")
        st.write(
            "Si oui, veuillez vérifier que les informations clients ont bien \
                    été renseigné. Pour rappel les champs à renseigner sont:"
        )

        st.write(df.columns)

    # identifiant correct:
    elif (
        int(id_input) in liste_id
    ):  # quand un identifiant correct a été saisi on appelle l'API

        API_url = "https://api-flask-scoring-credit.herokuapp.com/" + str(int(id_input))

        with st.spinner("Chargement du score du client..."):
            # Appel de l'API :
            call_api(int(id_input), API_url)

        with st.spinner("Chargement des détails de la prédiction..."):
            interpretation_client(id_input)

    st.success("Done!")
