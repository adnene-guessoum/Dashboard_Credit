# Import
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

# Local import:
from tools import display_homepage, display_about_clients, display_about_model, display_predict_page
from tools import load_data, cleaning

# load small dataset:
path_df = 'client/saved_model_data/restricted_dataset'
df = pd.read_csv(path_df)
df = df.drop(["Unnamed: 0"], axis = 1)

#Load variable descriptions:
path_desc = 'client/saved_model_data/desc_features.csv'
variables_description = load_data(path_desc)

liste_id = df['ID'].tolist()
data = cleaning(df)

# Dashboard

# Sidebar
with st.sidebar:
    selection = option_menu(
            menu_title="Navigation",
            options = ["Home",
                       "Comprendre nos clients",
                       "Comprendre le modèle",
                       "Prédire et expliquer"],
            icons = ["house", "book", "bar-chart", "bullseye"],
            menu_icon = "cast",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#FF6F61", "font-size": "25px"},
                "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#6B5B95"},
                })

    st.sidebar.markdown('---')
    st.write('Description des variables :')
    dict_desc = dict(zip(variables_description.Feature, variables_description.description))

    option = st.selectbox(
            'Veuillez indiquez la variable à expliquer :',
            dict_desc.keys())

    st.write(dict_desc[option])

    st.write("liste des id clients disponibles pour la version déployée de l'app (4000 clients sélectionnés au hasard) :")
    st.write(liste_id)

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



