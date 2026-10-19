from contextlib import asynccontextmanager
import uuid

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app_ml.config import get_ml_settings
from app_ml.metrics import calcular_resumo_e_indicadores
from app_ml.predictor import get_predictor
from app_ml.schemas import ForecastRequest, ForecastResponse
from app_ml.seguranca import validar_ml_api_key
from app_ml.validation import ForecastValidationError, validar_requisicao_previsao


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_predictor()
    yield


app = FastAPI(
    title="API de Aprendizagem de Máquina - Viveiro Bioterra (PIC-IV)",
    description=(
        "Serviço de previsão meteorológica horária (24h) de temperatura e umidade relativa (TU) "
        "baseado em 48 estimadores HistGradientBoostingRegressor."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(ForecastValidationError)
async def forecast_validation_error_handler(
    request: Request,
    exc: ForecastValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        },
    )


@app.exception_handler(RequestValidationError)
async def pydantic_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error": {
                "code": "INVALID_REQUEST_SCHEMA",
                "message": "O corpo da requisição possui formato estrutural inválido.",
                "details": {"errors": exc.errors()},
            },
        },
    )


@app.get("/health")
def health():
    predictor = get_predictor()
    return {
        "status": "ok",
        "aplicacao": "api-aprendizagem-maquina-pic-iv",
        "model": predictor.get_model_metadata(),
    }


@app.get(
    "/v1/model/info",
    dependencies=[Depends(validar_ml_api_key)],
)
def model_info():
    predictor = get_predictor()
    return {
        "status": "ok",
        "model": predictor.get_model_metadata(),
        "training_metadata": {
            "feature_set": predictor.configuracao,
            "history_hours": predictor.tau,
            "forecast_hours": predictor.horizonte,
            "years_train": predictor.metadados_preparacao.get("anos_treino", []),
            "year_validation": predictor.metadados_preparacao.get("ano_validacao", 2024),
            "year_test": predictor.metadados_preparacao.get("ano_teste", 2025),
            "scaler_x_mean": predictor.mean_x.tolist(),
            "scaler_x_scale": predictor.scale_x.tolist(),
        },
    }


@app.post(
    "/v1/forecast/24h",
    response_model=ForecastResponse,
    dependencies=[Depends(validar_ml_api_key)],
)
@app.post(
    "/predict",
    response_model=ForecastResponse,
    dependencies=[Depends(validar_ml_api_key)],
)
def forecast_24h(payload: ForecastRequest):
    settings = get_ml_settings()
    req_id = (payload.request_id or "").strip() or str(uuid.uuid4())

    entrada = validar_requisicao_previsao(
        req=payload,
        settings=settings,
        request_id=req_id,
    )

    predictor = get_predictor()
    forecast_points = predictor.prever_24h(entrada)

    summary_block, indicators_block, alerts, warnings = calcular_resumo_e_indicadores(
        entrada=entrada,
        forecast=forecast_points,
        options=payload.options,
    )

    primeira_obs = entrada.observations[0]
    ultima_obs = entrada.observations[-1]
    gerado_em = entrada.reference_now_local.replace(microsecond=0).isoformat()

    return ForecastResponse(
        status="success",
        request_id=req_id,
        generated_at=gerado_em,
        location=entrada.location,
        model=predictor.get_model_metadata(),
        input_window={
            "start": primeira_obs.timestamp_iso,
            "end": ultima_obs.timestamp_iso,
            "records": len(entrada.observations),
            "continuity": "complete",
            "latest_observation_age_minutes": entrada.latest_observation_age_minutes,
            "source": entrada.source_provider,
            "dataset": entrada.source_dataset,
            "last_observed": {
                "timestamp": ultima_obs.timestamp_iso,
                "temperature_c": ultima_obs.temperature_c,
                "humidity_pct": ultima_obs.humidity_pct,
            },
        },
        forecast=forecast_points,
        summary=summary_block,
        indicators=indicators_block,
        alerts=alerts,
        warnings=warnings,
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_ml_settings()
    print(f"Iniciando API de Aprendizagem de Máquina na porta {settings.porta_ml_api}...")
    uvicorn.run(
        "app_ml.main:app",
        host="127.0.0.1",
        port=settings.porta_ml_api,
        reload=True,
    )


