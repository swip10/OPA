import pytest
import requests


@pytest.fixture
def json_data():
    x = [float(i) for i in range(0, 59)]
    return {
        'price': x,
        'volume': x,
        'currency': 'string',
    }


def test_get_model_api_currency():
    request = "http://127.0.0.1:8000/currency"

    x = requests.get(request).json()
    assert x["currency"] == "ETHBTC"


@pytest.mark.parametrize("next_hours", [8, 16, 1])
def test_get_model_api_prediction(json_data, next_hours):
    url = "http://127.0.0.1:8000/prediction"

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    params = {
        'next_hours': str(next_hours),
    }

    r = requests.post(url, params=params, headers=headers, json=json_data)

    if next_hours == 1:
        print(r.status_code)
        assert r.status_code == 204
    else:
        response = r.json()
        assert isinstance(response["price"], float)
        assert isinstance(response["volume"], float)
