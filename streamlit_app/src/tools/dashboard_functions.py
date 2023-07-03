"""
Fonctions pour le dashboard streamlit de visualisation des données
et des résultats du modèle.

à l'inverse d'utils.py, ce fichier est destiné à être utilisé dans
le dashboard streamlit directement et est dépendant de shap et de la
nature particulière du projet.

liste des fonctions:

    - credit_score
    - credit_metric
    - st_shap
    - call_api

    - informations_data

    - visualisation_distribution_target
    - visualisation_univar

    - interpretation_global
    - interpretation_client

    - display_filtered_client_visualisation

    - navigation
    - display_homepage
    - display_about_clients
    - display_about_model
    - display_predict_page

"""
import pickle
import json
from urllib.request import urlopen
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # type: ignore
import seaborn as sns  # type: ignore
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import (  # type: ignore
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.metrics import fbeta_score, precision_score, recall_score
import shap  # type: ignore
from .utils import load_data, cleaning

shap.initjs()


# load dataset:
PATH_DF = "streamlit_app/output_data/selected_feature_dataset"
data = pd.read_csv(PATH_DF)
# df = df.drop(["Unnamed: 0"], axis = 1)

# Load variable descriptions:
PATH_DESC = "streamlit_app/output_data/desc_features.csv"
variables_description = load_data(PATH_DESC)

liste_id = data["ID"].tolist()
data = cleaning(data)

# Load model
PATH_MODEL = "streamlit_app/output_data/rec28052023_model_final.pickle.dat"
with open(PATH_MODEL, "rb") as f:
    model = pickle.load(f)
# data = cleaning(df)

# Load explainer:
FILE_PATH = "streamlit_app/output_data/shap_values"
with open(FILE_PATH, "rb") as f:
    shap_values = pickle.load(f)

# model_explainer = pickle.load(open(FILE_PATH,'rb'))
FILE_PATH = "streamlit_app/output_data/model_explainer_bis"
try:
    with open(FILE_PATH, "rb") as f:
        model_explainer = pickle.load(f)
except (pickle.UnpicklingError, FileNotFoundError, AttributeError, TypeError, Warning):
    ERROR_MESSAGE = "Erreur lors du chargement du modèle"
    st.error(ERROR_MESSAGE)

# Load expected values:
FILE_PATH = "streamlit_app/output_data/expected_values"
with open(FILE_PATH, "rb") as f:
    exp_vals = pickle.load(f)

# Split data
train_df = data[data["TARGET"].notnull()]
test_df = data[data["TARGET"].isnull()]
pred_data = data.drop(["TARGET", "ID"], axis=1)
true_y = train_df["TARGET"]
labels = train_df["ID"]

# Predictions
predictions = model.predict(train_df.drop(["TARGET", "ID"], axis=1))
probas_predictions = model.predict_proba(train_df.drop(["TARGET", "ID"], axis=1))[:, 1]


def credit_score(y_true, y_pred):
    """
    fonction de calcul du score métier pour le projet.

    Parameters
    ----------
    y_true : list de int, numpy array
        vrai valeurs de la prédiction pour la target y.
    y_pred : list de int, numpy array
        valeurs prédites par le modèle pour la target y.

    Returns
    -------
    cs : int
        valeur de la métrique métier à minimiser.

    """

    # false positive
    false_positive = ((y_pred == 1) & (y_true == 0)).sum()

    # false negative
    false_negative = ((y_pred == 0) & (y_true == 1)).sum()

    cred_score = 10 * false_negative + false_positive
    return cred_score


def credit_metric(x_arg, y_arg):
    """
    fonction de calcul du score métier pour le projet.
    """
    return 10 * x_arg + y_arg


def st_shap(plot, height=None):
    """
    fonction d'affichage pour les force plot dans le dashboard streamlit
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def call_api(input_data: int, api_url: str, dataframe: pd.DataFrame):
    """
    This function calls the Flask API with the client Id (input data).

    Parameters
    ----------
    input_data : dict
        The input data for the API.

    Returns
    -------
    response : dict
        The response of the API.
    """

    with st.spinner("Appel de l'API en cours..."):
        with urlopen(api_url) as json_url:
            api_data = json.loads(json_url.read())

    classe_predite = api_data["prediction"]
    if classe_predite == 1:
        etat = "client à risque"
        proba = 1 - api_data["proba"]
    else:
        etat = "client peu risqué"
        proba = 1 - api_data["proba"]

    # affichage de la prédiction
    classe = dataframe[dataframe["ID"] == input_data]["TARGET"].values[0]

    if np.isnan(classe):
        classe_reelle = "pas de données réelles pour ce client (client test)"
    else:
        classe_reelle = (
            str(classe)
            .replace("0.0", "pas de défaut sur le crédit octroyé")
            .replace("1.0", "défaut sur le crédit octroyé")
        )

    chaine_prediction = (
        "Prédiction : **"
        + etat
        + "** avec **"
        + str(round(proba * 100))
        + "%** de\
                risque de défaut "
    )
    chaine_realite = "classe réelle : " + str(classe_reelle)

    st.markdown(chaine_prediction)
    st.markdown(chaine_realite)


def informations_data(datafr):
    """
    Parameters
    ----------
    datafr : panda dataframe
        sur lequel on veut des infos.

    Print
    -------
    shape et head
    """
    st.header("Description de nos données :")

    st.write("-" * 100)
    st.write("nombre total de client : " + str(datafr.shape[0]))
    st.write("nombre total de variables : " + str(datafr.shape[1]))
    st.write("-" * 100)
    st.write("les premiers clients dans nos données :")
    st.dataframe(datafr.head())


def visualisation_distribution_target(datafr):
    """Parameters
    ----------
    datafr : panda dataframe
        sur lequel on veut des infos.

    Returns
    -------
    pie chart plotly de la target y.

    """

    target_distribution = datafr.TARGET.value_counts(dropna=False)
    labels_inner = ["Pas de défaut", "Défaut", "Info non disponible"]

    fig = go.Figure(
        data=[
            go.Pie(
                values=target_distribution,
                labels=labels_inner,
                textinfo="label+percent+value",
            )
        ],
        layout=go.Layout(title="Visualiser le déséquilibre des classes"),
    )

    st.header("Camembert : Distribution de la Target")
    st.plotly_chart(fig)


def visualisation_univar(datafr):
    """Parameters
    ----------
    datafr : panda dataframe
        sur lequel on veut des infos.

    Returns
    -------
    quelques graphs d'analyse univariée des colonnes du df.

    """

    df_cat = datafr[["TARGET", "CODE_GENDER", "NAME_FAMILY_STATUS_Married"]].astype(
        "category"
    )
    df_num = datafr[
        [
            "AMT_CREDIT",
            "DAYS_EMPLOYED",
            "DAYS_LAST_PHONE_CHANGE",
            "PAYMENT_RATE",
        ]
    ]

    st.markdown("Nos trois variables catégorielles: défaut, sexe et statut marital :")
    for i in df_cat.columns:
        fig2 = plt.figure(figsize=(5, 4))
        sns.countplot(x=i, data=df_cat)
        # st.pyplot(fig2)
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png")
        st.image(buf2)

    st.markdown("Les distributions de quelques variables numériques :")
    for j in df_num.columns:
        # fig = plt.figure(figsize = (5,4))
        # sns.displot(df_num[j], color = 'black', rug = True)
        # st.pyplot(fig)
        # buf = BytesIO()
        # fig.savefig(buf, format="png")
        # st.image(buf)

        fig3 = plt.figure(figsize=(5, 4))
        sns.boxplot(df_num[j]).set_title(j)
        # st.pyplot(fig3)
        buf3 = BytesIO()
        fig3.savefig(buf3, format="png")
        st.image(buf3)


# pylint: disable=too-many-locals
def interpretation_global(sample_nb: int):
    """
    Parameters
    ----------
    sample_nb : int
        nombre d'individus à tirer aléatoirement pour visualisation shap.


    Returns
    -------
    Shap summary plot, decision plot, infos sur le modèle en format streamlit
    """

    auc_train_model = roc_auc_score(true_y, probas_predictions)

    st.write("-----------------------------------------------------")
    st.write("AUC pour toutes les données disponibles : " + str(auc_train_model))

    # ROC curve
    fpr_train_gbt, tpr_train_gbt, _ = roc_curve(true_y, probas_predictions)

    st.write("-----------------------------------------------------")
    st.write("Courbe ROC pour toutes les données disponibles")

    fig = plt.figure()
    plt.plot(
        fpr_train_gbt,
        tpr_train_gbt,
        color="blue",
        label=f"ROC curve (area = {auc_train_model:0.2f})",
    )
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), color="red")
    plt.legend(loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True positive Rate")
    plt.title("Credit Default- Gradient Boosting")
    # plt.savefig('roc_curve.png', dpi=300)
    plt.show()
    st.pyplot(fig)

    st.write("---------------------------------------------------")

    st.write("Matrice de confusion simple pour les données disponibles : ")

    true_neg, false_pos, false_neg, true_pos = confusion_matrix(
        true_y, predictions
    ).ravel()
    st.write("positif : client fait défaut")
    st.write("négatif : client ne fait pas défaut")
    st.write(
        "vrai negatif : ",
        true_neg,
        ", faux positif : ",
        false_pos,
        ", faux negatif : ",
        false_neg,
        ", vrai positif : ",
        true_pos,
    )
    st.write("--------------------------------------------------")

    st.write(
        "score crédit global des clients actuels : ",
        credit_metric(false_neg, false_pos),
    )

    fbs_gbt = fbeta_score(true_y, predictions, beta=2)

    st.write("score f2 qui privilégie la détection de faux négatifs : ", fbs_gbt)

    auc_gbt = roc_auc_score(true_y, predictions)

    st.write("aire sous la courbe ROC (bon si proche de 1) : ", auc_gbt)

    st.write(
        'Proportion de prédiction correcte parmis tout ce que le modèle prédit \
                    comme "mauvais clients" (vrais positifs / tous prédits positifs) :',
        precision_score(true_y, predictions),
    )

    st.write(
        "Probabilité de détecter un vrai défaut (vrais positifs détectés / tous \
                    vrais positifs) :",
        recall_score(true_y, predictions),
    )

    st.write(
        'Proportion de prédiction correcte parmis tout ce que le modèle prédit \
                    comme "bon clients" (vrais négatifs détéctés / tous \
                    vrais négatifs) :',
        true_neg / (true_neg + false_pos),
    )

    st.write("-------------------------------------------------------")
    st.write("Explication globale du modèle avec SHAP:")

    # plot 1
    st.write(
        "Classement et résumé global de l'importance des features pour \
                    le modèle d'après leurs influences respectives dans l'octroie \
                    de crédit des clients:"
    )
    fig1 = plt.figure()
    # sum_plot = shap.summary_plot(shap_values, pred_data)
    st.pyplot(fig1)

    # plot 2
    st.write("Visualisation sous forme de bar plot:")
    fig2 = plt.figure()
    shap.summary_plot(shap_values, pred_data, plot_type="bar")
    st.pyplot(fig2)

    # plot 3
    st.write(
        "Description du processus de décision pour le sous-ensemble \
                    aléatoire de clients:"
    )
    sub_sample = pred_data.sample(n=sample_nb)
    shap_values_sub = model_explainer.shap_values(sub_sample)
    fig4 = plt.figure()
    # dec_plot_sample = shap.decision_plot(
    # exp_vals.tolist(), shap_values_sub, features=pred_data, highlight=[1]
    # )
    st.pyplot(fig4)

    # plot 4
    st.write(
        "Influences respectives des features pour la décision d'octroyer \
                    le crédit aux clients du sous ensemble :"
    )

    for i in shap_values_sub:
        st.write(
            "-------------------------------------------------------------------------"
        )
        st.write("client aléatoire " + str(i + 1))
        fig = plt.figure()
        # B_plot = shap.bar_plot(j, pred_data)
        st.pyplot(fig)


# pylint: enable=too-many-locals


def interpretation_client(id_input):
    """
    Fonction qui interpréte le score d'un client en utilisant SHAP'
    Parameters
    -------
    id_input : int
    numéro identification du client pour interprétation de sa prédiction
    par le modèle

    Returns
    -------
    série de graphiques et d'infos clients (SHAP)'

    """

    data_inner = data[data["ID"] == int(id_input)]

    st.write("--------------------------------------------")
    st.write("caractéristiques du client sélectionné :")

    id_target_data = data_inner[["ID", "TARGET"]]
    st.dataframe(id_target_data)

    individual_data = data_inner.drop(["TARGET", "ID"], axis=1)
    st.dataframe(individual_data)

    shap_values_inner = model_explainer.shap_values(individual_data)

    # choix variable:
    st.write("Valeurs des variables pour le client :")
    feature_values = dict(individual_data)

    option = st.selectbox(
        "Veuillez indiquez la variable à chercher : ", feature_values.keys()
    )

    st.write(feature_values[option])

    st.write("----------------------------------------------")
    st.write(
        "Graphiques explicatifs de la prédiction pour le client (variables \
                    ayant le plus significativement contribué à la décision):"
    )
    st.write("----------------------------------------------")

    st.write("Contribution des variables principales à la prédiction pour ce client:")

    fig2 = plt.figure()
    # dec_plot = shap.decision_plot(exp_vals.tolist(), shap_values, features=pred_data)
    st.pyplot(fig2)

    st.write("----------------------------------------------")
    st.write(
        "Contribution des variables les plus imprtantes dans le classement du client :"
    )
    # fig = plt.figure()
    # Insert first SHAP plot here
    f_plot = shap.force_plot(
        model_explainer.expected_value, shap_values_inner, individual_data
    )
    st_shap(f_plot, 150)
    # st.pyplot(fig)

    st.write("----------------------------------------------")
    st.write(
        "Top 7 des variables les plus imprtantes pour ce client (et sens de l'influence)"
    )

    fig3 = plt.figure()
    # B_plot = shap.bar_plot(shap_values[0], pred_data)
    st.pyplot(fig3)


def display_filtered_client_visualisation(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction qui affiche les données filtrées par l'utilisateur
    Parameters
    ----------
    dataframe : pd.dataFrame
    données clients filtrées selon les critères choisis par l'utilisateur
    """
    gender = list(dataframe["CODE_GENDER"].unique())

    marit_status = list(dataframe["NAME_FAMILY_STATUS_Married"].unique())

    amount = dataframe["AMT_CREDIT"]

    st.markdown("Filtres disponibles pour les données :")

    gender_choice = st.multiselect(
        "sexe : F = " + str(1) + " ; M = " + str(0), gender, gender
    )

    marital_choice = st.multiselect(
        "Married = " + str(1) + " ; not Married = " + str(0), marit_status, marit_status
    )

    amount_credit = st.slider(
        "AMT_CREDIT",
        float(amount.min()),
        float(amount.max()),
        (float(amount.min()), float(amount.max())),
        1000.0,
    )

    mask_gender = dataframe["CODE_GENDER"].isin(gender_choice)
    mask_marital = dataframe["NAME_FAMILY_STATUS_Married"].isin(marital_choice)

    # get the parties with a number of members in the range of nb_mbrs
    # HERE: mypy type error "between is not a method of Series[float]"
    mask_amount_credit = dataframe["AMT_CREDIT"].between(  # type: ignore
        amount_credit[0], amount_credit[1]
    )

    print(
        "mask_amount_credit : ", mask_amount_credit, "type : ", type(mask_amount_credit)
    )
    print(type(mask_marital))
    print(type(mask_gender))

    df_filtered = dataframe[mask_gender & mask_marital & mask_amount_credit]

    st.write("----------------------------------------")
    st.write("tableau de données filtrées : ")

    st.write("Description des variables :")

    colonnes = list(df_filtered.columns)

    col_choice = st.multiselect(
        "variables à observer dans les données \
            filtrées :",
        colonnes,
        default=("ID", "TARGET"),
    )

    st.dataframe(df_filtered[col_choice])

    return df_filtered


# Pages
def navigation(dataframe, selection):
    """
    fonction qui permet de naviguer entre les différentes pages de l'app
    """
    if selection == "Home":
        display_homepage()
    elif selection == "Comprendre nos clients":
        display_about_clients(dataframe)
    elif selection == "Comprendre le modèle":
        display_about_model()
    elif selection == "Prédire et expliquer":
        display_predict_page(dataframe)
    else:
        raise ValueError("The selection is not valid.")


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


def display_about_clients(dataframe: pd.DataFrame) -> None:
    """
    Fonction qui affiche la pages des informations sur les données clients
    """
    st.title("Analyse exploratoire des données clients:")

    st.write("--------------------------------------------------")
    st.title("Observer les données non-filtrées :")

    informations_data(dataframe)
    visualisation_distribution_target(dataframe)

    agree = st.checkbox("Observer quelques graphiques disponibles ?")
    if agree:
        visualisation_univar(dataframe)

    st.write("--------------------------------------------------")
    st.title("Observer les données filtrées :")

    df_choice = display_filtered_client_visualisation(dataframe)

    informations_data(df_choice)
    visualisation_distribution_target(df_choice)

    agree_2 = st.checkbox("Observer les changements ?")
    if agree_2:
        visualisation_univar(df_choice)


def display_about_model() -> None:
    """
    Fonction qui affiche la page des informations sur le modèle
    """
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


def display_predict_page(dataframe: pd.DataFrame) -> None:
    """
    Fonction qui affiche la page de prédiction de capacité du client à rembourser
    """
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

        st.write(data.columns)

    # identifiant correct:
    elif (
        int(id_input) in liste_id
    ):  # quand un identifiant correct a été saisi on appelle l'API
        api_url = "https://api-flask-scoring-credit.herokuapp.com/" + str(int(id_input))

        with st.spinner("Chargement du score du client..."):
            # Appel de l'API :
            call_api(int(id_input), api_url, dataframe)

        with st.spinner("Chargement des détails de la prédiction..."):
            interpretation_client(id_input)

    st.success("Done!")
