"""
This file is used to create the dashboard for the application.

The dashboard is the main page of the application, and it is where the user
will be able to see the current status of the application, as well as
interact with it.

Created by: Adnene Guessoum
on: 26/05/2023
"""

# Imports
import streamlit as st

# Local imports
from tools import *
from pages import *

# Constants
path_df = 'app/saved_model_data/restricted_dataset'
df = pd.read_csv(path_df)
df = df.drop(["Unnamed: 0"], axis = 1)
liste_id = df['ID'].tolist()
data = cleaning(df)

path_desc = 'app/saved_model_data/desc_features.csv'
desc = load_data(path_desc)

# Functions


def display_sidebar(path_data_description: str) -> None:
    """
    This function displays the sidebar of the dashboard and explanations of the data

    Parameters
    ----------
    path_data_description : str
        The path to the csv file containing the data description.

    Returns
    -------
    None
    """
    st.sidebar.markdown("---")
    st.write("Description des variables :")

    dict_description = describe_data(path_data_description)

    option = st.selectbox(
        "Veuillez indiquez la variable à expliquer :",
        dict_description.keys(),
    )

    st.write(dict_desc[option])


def page_selection(selection: str) -> None:
    """
    This function displays the page selected by the user.

    Parameters
    ----------
    selection : str
        The selected options for the sidebar.

    Returns
    -------
    None
    """
    if selection == "Home":
        display_homepage()
    elif selection == "Comprendre nos clients":
        display_about_clients()
    elif selection == "Comprendre le modèle":
        display_about_model()
    elif selection == "Prédire et expliquer":
        display_predict_page()
    else:
        raise ValueError("The selection is not valid.")

def main() -> None:
    """
    This function is the main function of the application.

    It is used to display the dashboard and the different pages.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    with st.sidebar:
        display_sidebar("saved_model_data/data_description.csv")
        selection = sidebar_options()

    # Display the sidebar and get the user selection
    selection = display_sidebar("saved_model_data/data_description.csv")

    # Display the page selected by the user
    page_selection(selection)

if __name__ == "__main__":
    main()
