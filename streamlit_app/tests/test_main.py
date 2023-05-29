"""
    tests pour main.py
"""
import pytest
from streamlit_app.tools.dashboard_functions import navigation


@pytest.fixture
def mock_data():
    """
    fixture pour les tests
    """
    return {"key": "value"}


def test_navigation_homepage(data_fixture, capsys):
    """
    test de la fonction navigation
    afficher la page d'accueil du site
    """
    navigation("Home", data_fixture)
    captured = capsys.readouterr()
    assert "display_homepage" in captured.out


def test_navigation_about_clients(data_fixture, capsys):
    """
    test de la fonction navigation
    afficher la page about clients du site
    """
    navigation("Comprendre nos clients", data_fixture)
    captured = capsys.readouterr()
    assert "display_about_clients" in captured.out


def test_navigation_about_model(data_fixture, capsys):
    """
    test de la fonction navigation
    afficher la page about mode du site
    """
    navigation("Comprendre le modèle", data_fixture)
    captured = capsys.readouterr()
    assert "display_about_model" in captured.out


def test_navigation_predict_page(data_fixture, capsys):
    """
    test de la fonction navigation
    afficher la page prédiction
    """
    navigation("Prédire et expliquer", data_fixture)
    captured = capsys.readouterr()
    assert "display_predict_page" in captured.out


def test_navigation_invalid():
    """
    test de la fonction navigation
    choix invalide
    """
    with pytest.raises(ValueError):
        navigation("Invalid Selection", {})
