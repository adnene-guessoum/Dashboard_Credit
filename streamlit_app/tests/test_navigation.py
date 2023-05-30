"""
    tests pour main.py
"""
import pytest
from tools.dashboard_functions import navigation


def mock_display_homepage():
    """mock de l'affichage de la page d'accueil"""
    print("Displaying homepage")


def mock_display_about_clients(dataframe):
    """mock de la page about_clients"""
    print(f"Displaying about_clients with dataframe : {dataframe}")


def mock_display_about_model():
    """mock de la page about_model"""
    print("Displaying about_model")


def mock_display_predict_page(dataframe):
    """mock de la page predict_page"""
    print(f"Displaying predict_page with dataframe : {dataframe}")


def test_navigation_homepage(capsys, monkeypatch):
    """test de la navigation vers la page d'accueil"""
    selection = "Home"
    dataframe = None

    monkeypatch.setattr(
        "tools.dashboard_functions.display_homepage", mock_display_homepage
    )

    navigation(dataframe, selection)

    captured = capsys.readouterr()
    assert "Displaying homepage" in captured.out


def test_navigation_about_clients(capsys, monkeypatch):
    """test de la navigation vers la page about_clients"""
    selection = "Comprendre nos clients"
    dataframe = None

    monkeypatch.setattr(
        "tools.dashboard_functions.display_about_clients", mock_display_about_clients
    )

    navigation(dataframe, selection)

    captured = capsys.readouterr()
    assert "Displaying about_clients" in captured.out


def test_navigation_about_model(capsys, monkeypatch):
    """test de la navigation vers la page about_model"""
    selection = "Comprendre le modèle"
    dataframe = None

    monkeypatch.setattr(
        "tools.dashboard_functions.display_about_model", mock_display_about_model
    )

    navigation(dataframe, selection)

    captured = capsys.readouterr()
    assert "Displaying about_model" in captured.out


def test_navigation_predict_page(capsys, monkeypatch):
    """test de la navigation vers la page predict_page"""
    selection = "Prédire et expliquer"
    dataframe = None

    monkeypatch.setattr(
        "tools.dashboard_functions.display_predict_page", mock_display_predict_page
    )

    navigation(dataframe, selection)

    captured = capsys.readouterr()
    assert "Displaying predict_page" in captured.out


def test_navigation_invalid_selection():
    """test de la navigation vers une page inexistante"""
    selection = "Invalid Selection"
    dataframe = None

    with pytest.raises(ValueError):
        navigation(dataframe, selection)
