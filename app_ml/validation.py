from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app_ml.config import MLSettings
from app_ml.schemas import ForecastRequest


class ForecastValidationError(Exception):
    """Exceção de validação de regras de negócio para a janela de entrada de 24 horas."""

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        status_code: int = 422,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code


@dataclass
class ValidatedObservation:
    timestamp_local: datetime
    timestamp_utc: datetime
    timestamp_iso: str
    temperature_c: float
    humidity_pct: float
    quality: str


@dataclass
class ValidatedForecastInput:
    request_id: str
    location: dict[str, Any]
    tz: ZoneInfo | timezone
    timezone_name: str
    source_provider: str
    source_dataset: str
    observations: list[ValidatedObservation]
    latest_observation_age_minutes: int
    reference_now_local: datetime


def interpretar_timezone(valor: str) -> ZoneInfo | timezone:
    """Interpreta identificador IANA (ex: America/Sao_Paulo) ou deslocamento UTC (ex: -03:00)."""
    valor = valor.strip()
    if not valor:
        raise ValueError("Timezone vazio.")

    deslocamento = re.fullmatch(
        r"(?P<sinal>[+-])(?P<horas>\d{2}):(?P<minutos>\d{2})",
        valor,
    )
    if deslocamento:
        horas = int(deslocamento.group("horas"))
        minutos = int(deslocamento.group("minutos"))
        if horas > 23 or minutos > 59:
            raise ValueError(f"Deslocamento UTC inválido: {valor}")
        total = timedelta(hours=horas, minutes=minutos)
        if deslocamento.group("sinal") == "-":
            total = -total
        return timezone(total)

    return ZoneInfo(valor)


def validar_requisicao_previsao(
    req: ForecastRequest,
    settings: MLSettings,
    request_id: str,
) -> ValidatedForecastInput:
    """
    Valida rigorosamente as 8 regras de negócio exigidas pela API de AM:
    1. localidade e timezone presentes e válidos;
    2. exatamente 24 registros em observations;
    3. temperatura e umidade presentes e finitas;
    4. valores dentro das faixas físicas adotadas;
    5. timestamps válidos e sem duplicidade;
    6. ordem estritamente crescente;
    7. horas consecutivas (passo de 1 hora);
    8. registro mais recente não excessivamente atrasado (nem no futuro).
    """
    # 1. Localidade ou timezone ausente
    if req.location is None:
        raise ForecastValidationError(
            code="MISSING_LOCATION_OR_TIMEZONE",
            message="O bloco 'location' é obrigatório na requisição.",
            details={"missing_field": "location"},
        )

    loc_id = (req.location.id or "").strip()
    loc_city = (req.location.city or "").strip()
    tz_name = (req.location.timezone or "").strip()

    campos_ausentes = []
    if not loc_id and req.location.woeid is None:
        campos_ausentes.append("location.id")
    if not loc_city:
        campos_ausentes.append("location.city")
    if not tz_name:
        campos_ausentes.append("location.timezone")

    if campos_ausentes:
        raise ForecastValidationError(
            code="MISSING_LOCATION_OR_TIMEZONE",
            message="Localidade ou timezone ausente na requisição.",
            details={"missing_fields": campos_ausentes},
        )

    try:
        tz = interpretar_timezone(tz_name)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ForecastValidationError(
            code="MISSING_LOCATION_OR_TIMEZONE",
            message=f"Timezone inválido informado em location.timezone: '{tz_name}'.",
            details={"timezone": tz_name, "error": str(exc)},
        ) from exc

    # 2. Exatamente 24 registros
    obs_list = req.observations or []
    if len(obs_list) != 24:
        raise ForecastValidationError(
            code="INVALID_OBSERVATION_COUNT",
            message=f"A janela de entrada deve conter exatamente 24 registros horários (recebidos: {len(obs_list)}).",
            details={"expected_records": 24, "received_records": len(obs_list)},
        )

    parsed_obs: list[ValidatedObservation] = []
    seen_utc_timestamps: set[datetime] = set()
    duplicate_timestamps: list[str] = []

    for idx, item in enumerate(obs_list):
        # 3. Temperatura ou umidade ausentes
        if item.temperature_c is None or item.humidity_pct is None:
            raise ForecastValidationError(
                code="MISSING_METEOROLOGICAL_VALUES",
                message=f"Temperatura ou umidade ausente na observação de índice {idx}.",
                details={
                    "index": idx,
                    "timestamp": item.timestamp,
                    "temperature_c": item.temperature_c,
                    "humidity_pct": item.humidity_pct,
                },
            )

        try:
            temp_val = float(item.temperature_c)
            umid_val = float(item.humidity_pct)
        except (TypeError, ValueError) as exc:
            raise ForecastValidationError(
                code="MISSING_METEOROLOGICAL_VALUES",
                message=f"Temperatura ou umidade não numérica na observação de índice {idx}.",
                details={"index": idx, "timestamp": item.timestamp},
            ) from exc

        if not (math.isfinite(temp_val) and math.isfinite(umid_val)):
            raise ForecastValidationError(
                code="MISSING_METEOROLOGICAL_VALUES",
                message=f"Temperatura ou umidade contém NaN/Infinito na observação de índice {idx}.",
                details={"index": idx, "timestamp": item.timestamp},
            )

        # 4. Faixas físicas adotadas
        fora_faixa = []
        if not (settings.ml_min_temperature_c <= temp_val <= settings.ml_max_temperature_c):
            fora_faixa.append(
                f"temperature_c={temp_val} fora de [{settings.ml_min_temperature_c}, {settings.ml_max_temperature_c}]"
            )
        if not (settings.ml_min_humidity_pct <= umid_val <= settings.ml_max_humidity_pct):
            fora_faixa.append(
                f"humidity_pct={umid_val} fora de [{settings.ml_min_humidity_pct}, {settings.ml_max_humidity_pct}]"
            )
        if fora_faixa:
            raise ForecastValidationError(
                code="OUT_OF_PHYSICAL_BOUNDS",
                message=f"Valores meteorológicos fora das faixas físicas adotadas no índice {idx}: {'; '.join(fora_faixa)}.",
                details={
                    "index": idx,
                    "timestamp": item.timestamp,
                    "temperature_c": temp_val,
                    "humidity_pct": umid_val,
                    "bounds": {
                        "temperature_c": [settings.ml_min_temperature_c, settings.ml_max_temperature_c],
                        "humidity_pct": [settings.ml_min_humidity_pct, settings.ml_max_humidity_pct],
                    },
                },
            )

        # Timestamp presente e válido
        raw_ts = (item.timestamp or "").strip()
        if not raw_ts:
            raise ForecastValidationError(
                code="INVALID_TIMESTAMP",
                message=f"Timestamp ausente na observação de índice {idx}.",
                details={"index": idx},
            )

        try:
            dt = datetime.fromisoformat(raw_ts)
        except ValueError as exc:
            raise ForecastValidationError(
                code="INVALID_TIMESTAMP",
                message=f"Timestamp em formato inválido no índice {idx}: '{raw_ts}'.",
                details={"index": idx, "timestamp": raw_ts},
            ) from exc

        if dt.tzinfo is None:
            dt_local = dt.replace(tzinfo=tz)
        else:
            dt_local = dt.astimezone(tz)

        dt_utc = dt_local.astimezone(timezone.utc)

        # 5. Timestamps duplicados
        if dt_utc in seen_utc_timestamps:
            duplicate_timestamps.append(dt_local.isoformat())
        seen_utc_timestamps.add(dt_utc)

        parsed_obs.append(
            ValidatedObservation(
                timestamp_local=dt_local,
                timestamp_utc=dt_utc,
                timestamp_iso=dt_local.isoformat(),
                temperature_c=temp_val,
                humidity_pct=umid_val,
                quality=item.quality or "observed",
            )
        )

    if duplicate_timestamps:
        raise ForecastValidationError(
            code="DUPLICATE_TIMESTAMPS",
            message="Foram encontrados timestamps duplicados na janela de 24 observações.",
            details={"duplicates": duplicate_timestamps},
        )

    # 6. Ordem estritamente crescente
    for i in range(1, len(parsed_obs)):
        if parsed_obs[i].timestamp_utc <= parsed_obs[i - 1].timestamp_utc:
            raise ForecastValidationError(
                code="NON_MONOTONIC_TIMESTAMPS",
                message=(
                    "As observações não estão em ordem temporal estritamente crescente "
                    f"(índice {i - 1}: {parsed_obs[i - 1].timestamp_iso} -> índice {i}: {parsed_obs[i].timestamp_iso})."
                ),
                details={
                    "index": i,
                    "previous_timestamp": parsed_obs[i - 1].timestamp_iso,
                    "current_timestamp": parsed_obs[i].timestamp_iso,
                },
            )

    # 7. Horas consecutivas (diferença exata de 1h entre registros adjacentes)
    lacunas = []
    for i in range(1, len(parsed_obs)):
        delta = parsed_obs[i].timestamp_utc - parsed_obs[i - 1].timestamp_utc
        if delta != timedelta(hours=1):
            lacunas.append(
                {
                    "index": i,
                    "previous_timestamp": parsed_obs[i - 1].timestamp_iso,
                    "current_timestamp": parsed_obs[i].timestamp_iso,
                    "interval_hours": round(delta.total_seconds() / 3600.0, 4),
                }
            )
    if lacunas:
        raise ForecastValidationError(
            code="NON_CONSECUTIVE_HOURS",
            message="As 24 observações não formam uma sequência horária contínua (intervalo diferente de 1 hora).",
            details={"gaps": lacunas},
        )

    # 8. Verificar se o registro mais recente está excessivamente atrasado ou no futuro
    if req.options and req.options.reference_time:
        try:
            ref_dt = datetime.fromisoformat(req.options.reference_time.strip())
            ref_now_local = ref_dt.replace(tzinfo=tz) if ref_dt.tzinfo is None else ref_dt.astimezone(tz)
        except ValueError as exc:
            raise ForecastValidationError(
                code="INVALID_REFERENCE_TIME",
                message=f"options.reference_time inválido: '{req.options.reference_time}'.",
            ) from exc
    else:
        ref_now_local = datetime.now(tz)

    latest_obs = parsed_obs[-1]
    age_seconds = (ref_now_local.astimezone(timezone.utc) - latest_obs.timestamp_utc).total_seconds()
    age_minutes = int(round(age_seconds / 60.0))

    if settings.ml_enforce_observation_age:
        if age_minutes > settings.ml_max_observation_age_minutes:
            raise ForecastValidationError(
                code="STALE_LATEST_OBSERVATION",
                message=(
                    f"O registro mais recente ({latest_obs.timestamp_iso}) está excessivamente atrasado: "
                    f"{age_minutes} minutos (limite máximo permitido: {settings.ml_max_observation_age_minutes} minutos)."
                ),
                details={
                    "latest_observation_timestamp": latest_obs.timestamp_iso,
                    "reference_time": ref_now_local.isoformat(),
                    "age_minutes": age_minutes,
                    "max_allowed_age_minutes": settings.ml_max_observation_age_minutes,
                },
            )
        if age_minutes < -settings.ml_max_future_tolerance_minutes:
            raise ForecastValidationError(
                code="STALE_LATEST_OBSERVATION",
                message=(
                    f"O registro mais recente ({latest_obs.timestamp_iso}) está no futuro além da tolerância permitida "
                    f"({abs(age_minutes)} minutos à frente)."
                ),
                details={
                    "latest_observation_timestamp": latest_obs.timestamp_iso,
                    "reference_time": ref_now_local.isoformat(),
                    "age_minutes": age_minutes,
                },
            )

    woeid_val = req.location.woeid
    try:
        woeid_norm: int | str | None = int(woeid_val) if woeid_val is not None else 458410
    except (TypeError, ValueError):
        woeid_norm = woeid_val

    location_dict = {
        "id": loc_id or f"hg-woeid-{woeid_norm}",
        "woeid": woeid_norm,
        "city": loc_city,
        "state": req.location.state or "SP",
        "country": req.location.country or "Brasil",
        "timezone": tz_name,
    }

    return ValidatedForecastInput(
        request_id=request_id,
        location=location_dict,
        tz=tz,
        timezone_name=tz_name,
        source_provider=(req.source.provider if req.source else "hg_brasil") or "hg_brasil",
        source_dataset=(req.source.dataset if req.source else "current_conditions_collected_hourly")
        or "current_conditions_collected_hourly",
        observations=parsed_obs,
        latest_observation_age_minutes=max(0, age_minutes),
        reference_now_local=ref_now_local,
    )

