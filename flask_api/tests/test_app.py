"""
    tests de l'app flask
"""
import pytest
from utils.controllers import home
from app import app

app.config["TESTING"] = True


@pytest.fixture(scope="module", name="streamlit_client")
def fixture_client():
    """
    fixture client pour notre api
    """
    with app.test_client() as streamlit_client:
        yield streamlit_client


def test_homepage(streamlit_client):
    """
    test route pour get method vers accueil api
    """
    response = streamlit_client.get("/")
    assert response.status_code == 200
    assert response.get_data(as_text=True) == home()


if __name__ == "__main__":
    pytest.main([__file__])
