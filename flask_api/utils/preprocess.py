"""
preprocess et fonctions diverses
"""


def cleaning(data):
    """
    reorders the columns of the dataframe
    """
    # sert à assurer que les ID et Targets
    # sont les deux dernières colonnes
    identifiant = data["ID"]
    target = data["TARGET"]
    clean_df = data.drop(["TARGET", "ID"], axis=1)
    clean_df["ID"] = identifiant
    clean_df["TARGET"] = target
    return clean_df
