from functools import lru_cache
import os
from pathlib import Path

from pydantic import AliasChoices, Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - fallback caso pydantic-settings não esteja instalado
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore


RAIZ_PROJETO = Path(__file__).resolve().parent.parent
CAMINHO_PACOTE_PADRAO = (
    RAIZ_PROJETO
    / "resultados_modelagem"
    / "comparacao_final_TU"
    / "pacote_hgb_TU_48_api.joblib"
)


class MLSettings(BaseSettings):
    """Configurações da API de Aprendizagem de Máquina obtidas via variáveis de ambiente."""

    ml_model_package_path: str = str(CAMINHO_PACOTE_PADRAO)
    ml_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ML_API_KEY", "API_KEY", "ml_api_key"),
    )
    validar_ml_api_key: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "VALIDAR_ML_API_KEY", "validar_ml_api_key"
        ),
    )

    porta_ml_api: int = Field(
        default=8001,
        validation_alias=AliasChoices(
            "PORTA_ML_API", "porta_ml_api", "PORTA", "porta", "PORT", "port"
        ),
    )

    ml_max_observation_age_minutes: int = 180
    ml_max_future_tolerance_minutes: int = 15
    ml_enforce_observation_age: bool = True

    ml_min_temperature_c: float = -50.0
    ml_max_temperature_c: float = 60.0
    ml_min_humidity_pct: float = 0.0
    ml_max_humidity_pct: float = 100.0

    ml_default_temp_threshold_c: float = 28.0
    ml_default_humidity_threshold_pct: float = 75.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_ml_settings() -> MLSettings:
    return MLSettings()

