from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class LocationInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = Field(default=None, examples=["hg-woeid-458410"])
    woeid: int | str | None = Field(default=458410, examples=[458410])
    city: str | None = Field(default=None, examples=["Álvares Machado"])
    state: str | None = Field(default="SP", examples=["SP"])
    country: str | None = Field(default="Brasil", examples=["Brasil"])
    timezone: str | None = Field(default=None, examples=["America/Sao_Paulo"])


class SourceInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = Field(default="hg_brasil", examples=["hg_brasil"])
    dataset: str = Field(
        default="current_conditions_collected_hourly",
        examples=["current_conditions_collected_hourly"],
    )


class ObservationInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: str | None = Field(
        default=None,
        description="Timestamp ISO-8601 com fuso horário (ex: 2026-10-18T14:00:00-03:00).",
    )
    temperature_c: float | None = Field(
        default=None,
        description="Temperatura horária observada em °C.",
    )
    humidity_pct: float | None = Field(
        default=None,
        description="Umidade relativa horária observada em %.",
    )
    quality: str | None = Field(default="observed", examples=["observed"])


class ForecastOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    include_summary: bool = True
    include_indicators: bool = True
    temperature_threshold_c: float | None = Field(
        default=28.0,
        description="Limiar configurável de temperatura (°C) para cálculo de exposição no bloco summary.",
    )
    humidity_threshold_pct: float | None = Field(
        default=75.0,
        description="Limiar configurável de umidade (%) para cálculo de exposição no bloco summary.",
    )
    reference_time: str | None = Field(
        default=None,
        description="Instante de referência ISO-8601 opcional para avaliação de atraso (útil para testes determinísticos).",
    )


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str | None = Field(
        default=None,
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    location: LocationInput | None = None
    source: SourceInput = Field(default_factory=SourceInput)
    observations: list[ObservationInput] = Field(default_factory=list)
    options: ForecastOptions = Field(default_factory=ForecastOptions)


class ForecastPoint(BaseModel):
    horizon_hours: int
    timestamp: str
    temperature_c: float
    humidity_pct: float


class ThresholdEvaluation(BaseModel):
    condition: str
    direction: Literal["above", "above_or_equal", "below", "below_or_equal"]
    threshold: float
    unit: str
    hours_count: int
    longest_sequence_hours: int
    first_event_at: str | None = None
    first_event_horizon_hours: int | None = None
    peak_value: float
    peak_at: str


class VariableSummary(BaseModel):
    minimum: float
    maximum: float
    mean: float
    minimum_at: str
    maximum_at: str
    configured_threshold: float | None = None
    hours_above_configured_threshold: int | None = None
    longest_sequence_above_threshold: int | None = None
    first_event_at: str | None = None


class TemperatureSummary(BaseModel):
    minimum_c: float
    maximum_c: float
    mean_c: float
    minimum_at: str
    maximum_at: str
    configured_threshold_c: float | None = None
    hours_above_configured_threshold: int | None = None
    longest_sequence_above_threshold: int | None = None
    first_event_at: str | None = None


class HumiditySummary(BaseModel):
    minimum_pct: float
    maximum_pct: float
    mean_pct: float
    minimum_at: str
    maximum_at: str
    configured_threshold_pct: float | None = None
    hours_above_configured_threshold: int | None = None
    longest_sequence_above_threshold: int | None = None
    first_event_at: str | None = None


class SummaryBlock(BaseModel):
    temperature: TemperatureSummary
    humidity: HumiditySummary
    thresholds_by_context: dict[str, Any]


class CombinedPeriod(BaseModel):
    start: str
    end: str
    hours: int
    max_temperature_c: float
    max_humidity_pct: float


class IndicatorsBlock(BaseModel):
    temperature_trend: Literal["rising", "falling", "stable"]
    humidity_trend: Literal["rising", "falling", "stable"]
    temperature_short_term_delta_c: float
    humidity_short_term_delta_pct: float
    temperature_amplitude_c: float
    humidity_amplitude_pct: float
    combined_heat_humidity_periods: list[CombinedPeriod]


class AlertItem(BaseModel):
    code: str
    crop_id: str
    crop_name: str
    severity: Literal["info", "warning", "danger"]
    title: str
    message: str
    metric: Literal["temperature_c", "humidity_pct", "combined"]
    threshold: float | None = None
    unit: str | None = None
    hours_affected: int
    longest_sequence_hours: int
    first_event_at: str | None = None
    peak_value: float | None = None
    peak_at: str | None = None


class WarningItem(BaseModel):
    code: str
    message: str


class ForecastResponse(BaseModel):
    status: Literal["success"] = "success"
    request_id: str
    generated_at: str
    location: dict[str, Any]
    model: dict[str, Any]
    input_window: dict[str, Any]
    forecast: list[ForecastPoint]
    summary: SummaryBlock | None = None
    indicators: IndicatorsBlock | None = None
    alerts: list[AlertItem] = Field(default_factory=list)
    warnings: list[WarningItem] = Field(default_factory=list)

