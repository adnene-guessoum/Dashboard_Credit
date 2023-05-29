"""
    main.py est le fichier principal de l'application. il appelle les fonctions
    d'affichages des différentes pages de l'app streamlit et définis la barre de
    navigation latérale.

    Pour lancer l'app, il suffit de lancer la commande suivante dans le terminal :
    streamlit run main.py

    ou dans ce cas:
    poetry run streamlit run main.py
"""
# Import
import random
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu  # type: ignore

# Local import:
from tools import (
    display_homepage,
    display_about_clients,
    display_about_model,
    display_predict_page,
    load_data,
    cleaning,
)

# load small dataset:
PATH_DF = "output_data/selected_feature_dataset"
df = pd.read_csv(PATH_DF)
# df = df.drop(["Unnamed: 0"], axis = 1)

# Load variable descriptions:
PATH_DESC = "output_data/desc_features.csv"
variables_description = load_data(PATH_DESC)

liste_id = df["ID"].tolist()
data = cleaning(df)

# Dashboard

# Sidebar
with st.sidebar:
    selection = option_menu(
        menu_title="Navigation",
        options=[
            "Home",
            "Comprendre nos clients",
            "Comprendre le modèle",
            "Prédire et expliquer",
        ],
        icons=["house", "book", "bar-chart", "bullseye"],
        menu_icon="cast",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#FF6F61", "font-size": "25px"},
            "nav-link": {
                "font-size": "20px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#6B5B95"},
        },
    )

    st.sidebar.markdown("---")
    st.write("Description des variables :")
    dict_desc = dict(zip(variables_description.Row, variables_description.Description))
    dict_desc["ID"] = dict_desc.pop("SK_ID_CURR")

    option = st.selectbox(
        "Veuillez indiquez la variable à expliquer :", dict_desc.keys()
    )

    st.write(dict_desc[option])

    st.write("quelques id clients disponibles si vous voulez tester l'app :")
    st.write(random.sample(liste_id, 50))

# Pages
if selection == "Home":
    display_homepage()
elif selection == "Comprendre nos clients":
    display_about_clients(df)
elif selection == "Comprendre le modèle":
    display_about_model()
elif selection == "Prédire et expliquer":
    display_predict_page()
else:
    raise ValueError("The selection is not valid.")
