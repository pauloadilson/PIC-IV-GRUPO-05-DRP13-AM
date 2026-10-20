from functools import lru_cache
import os
from pathlib import Path

from pydantic import AliasChoices, Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - fallback caso pydantic-settings não esteja instalado
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore


DIRETORIO_APP_ML = Path(__file__).resolve().parent
RAIZ_PROJETO = DIRETORIO_APP_ML.parent
DIRETORIO_ARTEFATOS = DIRETORIO_APP_ML / "artefatos"


class MLSettings(BaseSettings):
    """Configurações da API de Aprendizagem de Máquina obtidas via variáveis de ambiente."""

    ml_model_package: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "ML_MODEL_PACKAGE",
            "ml_model_package",
            "NOME_MODELO",
            "nome_modelo",
            "ML_MODEL_PACKAGE_PATH",
            "ml_model_package_path",
        ),
        description="Nome ou caminho do arquivo do modelo configurado via .env (ex: ML_MODEL_PACKAGE=nome_do_modelo.joblib).",
    )

    def obter_caminho_modelo(self) -> Path:
        """
        Resolve dinamicamente o arquivo do modelo com base na variável ML_MODEL_PACKAGE:
        1. Se ML_MODEL_PACKAGE for informado:
           - Procura dentro de app_ml/artefatos/<ML_MODEL_PACKAGE>;
           - Procura em relação à raiz do projeto ou caminho absoluto.
        2. Fallback caso a variável não esteja no .env:
           - Localiza automaticamente o primeiro arquivo .joblib em app_ml/artefatos/.
        """
        if self.ml_model_package:
            p = Path(self.ml_model_package.strip())
            if p.is_absolute() and p.exists():
                return p
            cand_artefatos = DIRETORIO_ARTEFATOS / p
            if cand_artefatos.exists():
                return cand_artefatos
            cand_raiz = RAIZ_PROJETO / p
            if cand_raiz.exists():
                return cand_raiz
            if p.exists():
                return p
            return cand_artefatos

        # Fallback de autodescoberta: procura qualquer .joblib em app_ml/artefatos/
        if DIRETORIO_ARTEFATOS.exists():
            joblibs = sorted(DIRETORIO_ARTEFATOS.glob("*.joblib"))
            if joblibs:
                return joblibs[0]

        raise FileNotFoundError(
            "Nenhum pacote de modelo foi configurado ou encontrado. "
            "Por favor, defina a variável ML_MODEL_PACKAGE no arquivo .env."
        )

    @property
    def ml_model_package_path(self) -> str:
        """Propriedade de compatibilidade que retorna o caminho resolvido do modelo."""
        return str(self.obter_caminho_modelo())
    
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

