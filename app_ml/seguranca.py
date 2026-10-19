import hmac
from typing import Annotated

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

from app_ml.config import get_ml_settings


api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="APIKeyHeader",
    description="Chave utilizada para autorizar o acesso aos endpoints da API de Aprendizagem de Máquina (cabeçalho HTTP padrão 'X-API-Key').",
    auto_error=False,
)

api_key_header_alt = APIKeyHeader(
    name="ML_API_KEY",
    scheme_name="MLApiKeyHeader",
    description="Chave de API enviada alternativamente com o nome do cabeçalho 'ml_api_key'.",
    auto_error=False,
)

api_key_query = APIKeyQuery(
    name="api_key",
    scheme_name="APIKeyQuery",
    description="Chave de API fornecida como parâmetro de URL (?api_key=...), útil para testes no navegador.",
    auto_error=False,
)


def validar_ml_api_key(
    chave_header: Annotated[str | None, Security(api_key_header)] = None,
    chave_header_alt: Annotated[str | None, Security(api_key_header_alt)] = None,
    chave_query: Annotated[str | None, Security(api_key_query)] = None,
) -> str | None:
    settings = get_ml_settings()

    if not settings.validar_ml_api_key:
        return None

    if not settings.ml_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A autenticação por chave de API não está configurada no servidor (ML_API_KEY ausente).",
        )

    chave_recebida = chave_header or chave_header_alt or chave_query

    if not chave_recebida:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Chave de API não informada. Envie o cabeçalho 'X-API-Key: sua_chave' ou o parâmetro '?api_key=sua_chave'.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    if not hmac.compare_digest(chave_recebida.strip(), settings.ml_api_key.strip()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Chave de API inválida.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    return chave_recebida

