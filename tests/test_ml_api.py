import copy
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
import pytest

from app_ml.config import get_ml_settings
from app_ml.main import app


settings = get_ml_settings()
AUTH_HEADERS = {"X-API-Key": settings.ml_api_key} if settings.ml_api_key else {}
client = TestClient(app, headers=AUTH_HEADERS)


def construir_payload_valido() -> dict:
    tz = ZoneInfo("America/Sao_Paulo")
    inicio = datetime(2026, 10, 18, 14, 0, 0, tzinfo=tz)

    observations = []
    for i in range(24):
        ts = inicio + timedelta(hours=i)
        # Curva diurna sintética plausível
        hora = ts.hour
        temp = 24.0 + 5.0 * (1 if 11 <= hora <= 17 else (-0.5 if hora <= 6 else 0.2))
        umid = 62.0 - 12.0 * (1 if 11 <= hora <= 17 else (-0.8 if hora <= 6 else 0.0))
        observations.append(
            {
                "timestamp": ts.isoformat(),
                "temperature_c": round(temp, 1),
                "humidity_pct": round(umid, 1),
                "quality": "observed",
            }
        )

    return {
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "location": {
            "id": "hg-woeid-458410",
            "woeid": 458410,
            "city": "Álvares Machado",
            "state": "SP",
            "country": "Brasil",
            "timezone": "America/Sao_Paulo",
        },
        "source": {
            "provider": "hg_brasil",
            "dataset": "current_conditions_collected_hourly",
        },
        "observations": observations,
        "options": {
            "include_summary": True,
            "include_indicators": True,
            "temperature_threshold_c": 28.0,
            "humidity_threshold_pct": 75.0,
            "reference_time": "2026-10-19T13:15:00-03:00",
        },
    }


def test_health_and_model_info():
    r_health = client.get("/health")
    assert r_health.status_code == 200
    body_h = r_health.json()
    assert body_h["status"] == "ok"
    assert body_h["model"]["id"] == "hgb_tu_24h"
    assert body_h["model"]["estimators_count"] == 48

    r_info = client.get("/v1/model/info")
    assert r_info.status_code == 200
    body_i = r_info.json()
    assert body_i["model"]["feature_set"] == "TU"
    assert body_i["training_metadata"]["history_hours"] == 24


def test_forecast_24h_sucesso_completo():
    payload = construir_payload_valido()
    response = client.post("/v1/forecast/24h", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["request_id"] == "550e8400-e29b-41d4-a716-446655440000"
    assert data["input_window"]["records"] == 24
    assert data["input_window"]["continuity"] == "complete"
    assert data["input_window"]["latest_observation_age_minutes"] == 15

    assert len(data["forecast"]) == 24
    assert data["forecast"][0]["horizon_hours"] == 1
    assert data["forecast"][0]["timestamp"] == "2026-10-19T14:00:00-03:00"
    assert data["forecast"][-1]["horizon_hours"] == 24
    assert data["forecast"][-1]["timestamp"] == "2026-10-20T13:00:00-03:00"

    for pt in data["forecast"]:
        assert -50.0 <= pt["temperature_c"] <= 60.0
        assert 0.0 <= pt["humidity_pct"] <= 100.0

    # Verifica summary e blocos por cultura
    summary = data["summary"]
    assert summary is not None
    assert "minimum_c" in summary["temperature"]
    assert "maximum_c" in summary["temperature"]
    assert "hours_above_configured_threshold" in summary["temperature"]
    assert "longest_sequence_above_threshold" in summary["temperature"]

    ctx = summary["thresholds_by_context"]
    assert "geral_viveiro" in ctx
    assert "alface_crespa" in ctx
    assert "rosa_do_deserto" in ctx
    assert "salsinha" in ctx

    # Verifica indicators e warnings
    indicators = data["indicators"]
    assert indicators["temperature_trend"] in ("rising", "falling", "stable")
    assert indicators["humidity_trend"] in ("rising", "falling", "stable")
    assert indicators["temperature_amplitude_c"] >= 0.0

    warning_codes = {w["code"] for w in data["warnings"]}
    assert "EXTERNAL_REFERENCE_CONDITIONS" in warning_codes
    assert "SOURCE_DOMAIN_SHIFT" in warning_codes


def test_rejeita_quantidade_diferente_de_24():
    payload = construir_payload_valido()
    payload["observations"] = payload["observations"][:23]

    response = client.post("/v1/forecast/24h", json=payload)
    assert response.status_code == 422
    err = response.json()["error"]
    assert err["code"] == "INVALID_OBSERVATION_COUNT"


def test_rejeita_timestamps_duplicados():
    payload = construir_payload_valido()
    payload["observations"][10]["timestamp"] = payload["observations"][9]["timestamp"]

    response = client.post("/v1/forecast/24h", json=payload)
    assert response.status_code == 422
    err = response.json()["error"]
    assert err["code"] == "DUPLICATE_TIMESTAMPS"


def test_rejeita_ordem_nao_crescente():
    payload = construir_payload_valido()
    payload["observations"][5], payload["observations"][6] = (
        payload["observations"][6],
        payload["observations"][5],
    )

    response = client.post("/v1/forecast/24h", json=payload)
    assert response.status_code == 422
    err = response.json()["error"]
    assert err["code"] == "NON_MONOTONIC_TIMESTAMPS"


def test_rejeita_horas_nao_consecutivas():
    payload = construir_payload_valido()
    # Introduz um salto de 2h no último registro
    payload["observations"][-1]["timestamp"] = "2026-10-19T14:00:00-03:00"
    payload["options"]["reference_time"] = "2026-10-19T14:15:00-03:00"

    response = client.post("/v1/forecast/24h", json=payload)
    assert response.status_code == 422
    err = response.json()["error"]
    assert err["code"] == "NON_CONSECUTIVE_HOURS"


@pytest.mark.parametrize("campo_nulo", ["temperature_c", "humidity_pct"])
def test_rejeita_temperatura_ou_umidade_ausente(campo_nulo: str):
    payload = construir_payload_valido()
    payload["observations"][12][campo_nulo] = None

    response = client.post("/v1/forecast/24h", json=payload)
    assert response.status_code == 422
    err = response.json()["error"]
    assert err["code"] == "MISSING_METEOROLOGICAL_VALUES"


def test_rejeita_localidade_ou_timezone_ausente():
    payload = copy.deepcopy(construir_payload_valido())
    payload["location"]["timezone"] = ""

    response = client.post("/v1/forecast/24h", json=payload)
    assert response.status_code == 422
    err = response.json()["error"]
    assert err["code"] == "MISSING_LOCATION_OR_TIMEZONE"


def test_rejeita_registro_mais_recente_excessivamente_atrasado():
    payload = construir_payload_valido()
    # Define o horário de referência 6 horas após o último registro (limite padrão: 180 min)
    payload["options"]["reference_time"] = "2026-10-19T19:30:00-03:00"

    response = client.post("/v1/forecast/24h", json=payload)
    assert response.status_code == 422
    err = response.json()["error"]
    assert err["code"] == "STALE_LATEST_OBSERVATION"


@pytest.mark.parametrize(
    "campo,valor_invalido",
    [
        ("temperature_c", 75.0),
        ("temperature_c", -60.0),
        ("humidity_pct", 105.0),
        ("humidity_pct", -5.0),
    ],
)
def test_rejeita_valores_fora_das_faixas_fisicas(campo: str, valor_invalido: float):
    payload = construir_payload_valido()
    payload["observations"][3][campo] = valor_invalido

    response = client.post("/v1/forecast/24h", json=payload)
    assert response.status_code == 422
    err = response.json()["error"]
    assert err["code"] == "OUT_OF_PHYSICAL_BOUNDS"


def test_autenticacao_chave_api():
    unauth_client = TestClient(app)

    # 1. /health é sempre público
    r_pub = unauth_client.get("/health")
    assert r_pub.status_code == 200

    if settings.validar_ml_api_key:
        # Sem chave -> 401
        r_sem = unauth_client.get("/v1/model/info")
        assert r_sem.status_code == 401

        # Chave inválida -> 401
        r_inv = unauth_client.get(
            "/v1/model/info", headers={"X-API-Key": "chave_incorreta"}
        )
        assert r_inv.status_code == 401

        # Chave correta via header -> 200
        r_head = unauth_client.get(
            "/v1/model/info", headers={"X-API-Key": settings.ml_api_key}
        )
        assert r_head.status_code == 200

        # Chave correta via query string ?api_key= -> 200
        from urllib.parse import quote

        r_query = unauth_client.get(
            f"/v1/model/info?api_key={quote(settings.ml_api_key)}"
        )
        assert r_query.status_code == 200


