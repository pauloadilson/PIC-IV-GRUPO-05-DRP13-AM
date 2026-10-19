from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app_ml.config import RAIZ_PROJETO, get_ml_settings
from app_ml.schemas import ForecastPoint
from app_ml.validation import ValidatedForecastInput


def _atributos_temporais_ciclicos(dt_utc: datetime) -> tuple[float, float, float, float]:
    """
    Calcula (hora_sin, hora_cos, dia_ano_sin, dia_ano_cos) usando o horário UTC,
    idêntico ao protocolo de 01_preparacao_dados_multivariados.ipynb.
    """
    hora = float(dt_utc.hour)
    dia_ano = float(dt_utc.timetuple().tm_yday)

    hora_sin = float(np.sin(2.0 * np.pi * hora / 24.0))
    hora_cos = float(np.cos(2.0 * np.pi * hora / 24.0))
    dia_ano_sin = float(np.sin(2.0 * np.pi * dia_ano / 365.25))
    dia_ano_cos = float(np.cos(2.0 * np.pi * dia_ano / 365.25))
    return hora_sin, hora_cos, dia_ano_sin, dia_ano_cos


class HGBForecastPredictor:
    """Carrega o pacote de 48 estimadores HistGradientBoostingRegressor (TU) e executa inferência 24h."""

    def __init__(self, package_path: str | Path):
        p = Path(package_path)
        if not p.is_absolute() and not p.exists():
            p_rel = RAIZ_PROJETO / p
            if p_rel.exists():
                p = p_rel
        self.package_path = p
        if not self.package_path.exists():
            raise FileNotFoundError(
                f"Pacote do modelo não encontrado em: {self.package_path}"
            )

        pacote = joblib.load(self.package_path)
        self.modelos: dict[tuple[int, int], Any] = pacote["modelos"]
        self.configuracao: str = pacote.get("configuracao", "TU")
        self.hiperparametros: dict[str, Any] = pacote.get("hiperparametros", {})
        self.metadados_preparacao: dict[str, Any] = pacote.get("metadados_preparacao", {})

        media_x = self.metadados_preparacao.get(
            "media_scaler_x",
            [24.035498777651895, 62.11299266591139],
        )
        escala_x = self.metadados_preparacao.get(
            "escala_scaler_x",
            [4.9688930043454, 19.849507657378844],
        )
        self.mean_x = np.asarray(media_x, dtype=np.float64)
        self.scale_x = np.asarray(escala_x, dtype=np.float64)

        self.tau = int(self.metadados_preparacao.get("tau", 24))
        self.horizonte = int(self.metadados_preparacao.get("horizonte", 24))

        if len(self.modelos) != self.horizonte * 2:
            raise ValueError(
                f"Quantidade inesperada de estimadores no pacote: {len(self.modelos)}"
            )

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "id": "hgb_tu_24h",
            "version": "1.0.0",
            "algorithm": "HistGradientBoostingRegressor",
            "estimators_count": len(self.modelos),
            "feature_set": self.configuracao,
            "input_features_count": 240,
            "history_hours": self.tau,
            "forecast_hours": self.horizonte,
            "targets": ["temperature_c", "humidity_pct"],
            "inference_device": "cpu",
        }

    def construir_vetor_tabular(self, entrada: ValidatedForecastInput) -> np.ndarray:
        """
        Constrói a matriz tabular (1, 240) float32 exatamente como no treinamento:
        - X_historico: (1, 24, 6) -> [temp_norm, umid_norm, hora_sin, hora_cos, dia_ano_sin, dia_ano_cos]
        - X_futuro_temporal: (1, 24, 4) -> [hora_sin, hora_cos, dia_ano_sin, dia_ano_cos]
        """
        x_hist = np.empty((1, self.tau, 6), dtype=np.float32)
        x_fut = np.empty((1, self.horizonte, 4), dtype=np.float32)

        for i, obs in enumerate(entrada.observations):
            temp_norm = (obs.temperature_c - self.mean_x[0]) / self.scale_x[0]
            umid_norm = (obs.humidity_pct - self.mean_x[1]) / self.scale_x[1]
            h_sin, h_cos, d_sin, d_cos = _atributos_temporais_ciclicos(obs.timestamp_utc)

            x_hist[0, i, 0] = np.float32(temp_norm)
            x_hist[0, i, 1] = np.float32(umid_norm)
            x_hist[0, i, 2] = np.float32(h_sin)
            x_hist[0, i, 3] = np.float32(h_cos)
            x_hist[0, i, 4] = np.float32(d_sin)
            x_hist[0, i, 5] = np.float32(d_cos)

        ultimo_utc = entrada.observations[-1].timestamp_utc
        for h in range(1, self.horizonte + 1):
            fut_utc = ultimo_utc + timedelta(hours=h)
            h_sin, h_cos, d_sin, d_cos = _atributos_temporais_ciclicos(fut_utc)
            x_fut[0, h - 1, 0] = np.float32(h_sin)
            x_fut[0, h - 1, 1] = np.float32(h_cos)
            x_fut[0, h - 1, 2] = np.float32(d_sin)
            x_fut[0, h - 1, 3] = np.float32(d_cos)

        x_tab = np.concatenate(
            [x_hist.reshape(1, -1), x_fut.reshape(1, -1)],
            axis=1,
        ).astype(np.float32)

        return x_tab

    def prever_24h(self, entrada: ValidatedForecastInput) -> list[ForecastPoint]:
        x_tab = self.construir_vetor_tabular(entrada)
        ultimo_local = entrada.observations[-1].timestamp_local

        pontos: list[ForecastPoint] = []
        for h in range(1, self.horizonte + 1):
            temp_raw = float(self.modelos[(h, 0)].predict(x_tab)[0])
            umid_raw = float(self.modelos[(h, 1)].predict(x_tab)[0])

            temp_clamped = round(min(60.0, max(-50.0, temp_raw)), 1)
            umid_clamped = round(min(100.0, max(0.0, umid_raw)), 1)

            ts_futuro = (ultimo_local + timedelta(hours=h)).isoformat()
            pontos.append(
                ForecastPoint(
                    horizon_hours=h,
                    timestamp=ts_futuro,
                    temperature_c=temp_clamped,
                    humidity_pct=umid_clamped,
                )
            )

        return pontos


_PREDICTOR_INSTANCE: HGBForecastPredictor | None = None


def get_predictor() -> HGBForecastPredictor:
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        settings = get_ml_settings()
        _PREDICTOR_INSTANCE = HGBForecastPredictor(settings.ml_model_package_path)
    return _PREDICTOR_INSTANCE

