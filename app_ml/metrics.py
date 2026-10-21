from typing import Any, Literal

from app_ml.schemas import (
    AlertItem,
    CombinedPeriod,
    ForecastOptions,
    ForecastPoint,
    HumiditySummary,
    IndicatorsBlock,
    SummaryBlock,
    TemperatureSummary,
    WarningItem,
)
from app_ml.validation import ValidatedForecastInput


def _avaliar_sequencia_limiar(
    forecast: list[ForecastPoint],
    campo: Literal["temperature_c", "humidity_pct"],
    threshold: float,
    operador: Literal["gt", "ge", "lt", "le"],
    unidade: str,
) -> dict[str, Any]:
    """
    Calcula para os 24 pontos previstos:
    - quantidade de horas que satisfazem a condição;
    - maior sequência contínua de horas;
    - horário inicial do evento (first_event_at e first_event_horizon_hours);
    - valor extremo (pico ou vale) e seu horário.
    """
    valores = [getattr(p, campo) for p in forecast]

    if operador == "gt":
        mascara = [v > threshold for v in valores]
        simbolo = ">"
        direcao = "above"
    elif operador == "ge":
        mascara = [v >= threshold for v in valores]
        simbolo = ">="
        direcao = "above_or_equal"
    elif operador == "lt":
        mascara = [v < threshold for v in valores]
        simbolo = "<"
        direcao = "below"
    else:
        mascara = [v <= threshold for v in valores]
        simbolo = "<="
        direcao = "below_or_equal"

    horas_acima = int(sum(1 for m in mascara if m))

    maior_seq = 0
    seq_atual = 0
    primeiro_idx: int | None = None

    for idx, ativo in enumerate(mascara):
        if ativo:
            if primeiro_idx is None:
                primeiro_idx = idx
            seq_atual += 1
            if seq_atual > maior_seq:
                maior_seq = seq_atual
        else:
            seq_atual = 0

    if operador in ("gt", "ge"):
        extremo_val = max(valores)
    else:
        extremo_val = min(valores)
    extremo_idx = valores.index(extremo_val)

    return {
        "condition": f"{campo} {simbolo} {threshold:.1f}",
        "direction": direcao,
        "threshold": round(float(threshold), 1),
        "unit": unidade,
        "hours_count": horas_acima,
        "longest_sequence_hours": maior_seq,
        "first_event_at": forecast[primeiro_idx].timestamp if primeiro_idx is not None else None,
        "first_event_horizon_hours": forecast[primeiro_idx].horizon_hours if primeiro_idx is not None else None,
        "peak_value": round(float(extremo_val), 1),
        "peak_at": forecast[extremo_idx].timestamp,
    }


def _classificar_tendencia(
    delta: float,
    tolerancia: float,
) -> Literal["rising", "falling", "stable"]:
    if delta > tolerancia:
        return "rising"
    if delta < -tolerancia:
        return "falling"
    return "stable"


def _identificar_periodos_combinados(
    forecast: list[ForecastPoint],
    temp_min_c: float = 26.0,
    umid_min_pct: float = 75.0,
) -> list[CombinedPeriod]:
    """Identifica blocos contínuos em que calor (>= temp_min_c) e alta umidade (>= umid_min_pct) coocorrem."""
    periodos: list[CombinedPeriod] = []
    bloco_atual: list[ForecastPoint] = []

    for pt in forecast:
        if pt.temperature_c >= temp_min_c and pt.humidity_pct >= umid_min_pct:
            bloco_atual.append(pt)
        else:
            if bloco_atual:
                periodos.append(
                    CombinedPeriod(
                        start=bloco_atual[0].timestamp,
                        end=bloco_atual[-1].timestamp,
                        hours=len(bloco_atual),
                        max_temperature_c=round(max(p.temperature_c for p in bloco_atual), 1),
                        max_humidity_pct=round(max(p.humidity_pct for p in bloco_atual), 1),
                    )
                )
                bloco_atual = []

    if bloco_atual:
        periodos.append(
            CombinedPeriod(
                start=bloco_atual[0].timestamp,
                end=bloco_atual[-1].timestamp,
                hours=len(bloco_atual),
                max_temperature_c=round(max(p.temperature_c for p in bloco_atual), 1),
                max_humidity_pct=round(max(p.humidity_pct for p in bloco_atual), 1),
            )
        )

    return periodos


def calcular_resumo_e_indicadores(
    entrada: ValidatedForecastInput,
    forecast: list[ForecastPoint],
    options: ForecastOptions,
) -> tuple[SummaryBlock | None, IndicatorsBlock | None, list[AlertItem], list[WarningItem]]:
    temps = [p.temperature_c for p in forecast]
    umids = [p.humidity_pct for p in forecast]

    t_min = min(temps)
    t_max = max(temps)
    t_mean = round(sum(temps) / len(temps), 1)
    t_min_at = forecast[temps.index(t_min)].timestamp
    t_max_at = forecast[temps.index(t_max)].timestamp

    u_min = min(umids)
    u_max = max(umids)
    u_mean = round(sum(umids) / len(umids), 1)
    u_min_at = forecast[umids.index(u_min)].timestamp
    u_max_at = forecast[umids.index(u_max)].timestamp

    limiar_temp_cfg = options.temperature_threshold_c if options.temperature_threshold_c is not None else 28.0
    limiar_umid_cfg = options.humidity_threshold_pct if options.humidity_threshold_pct is not None else 75.0

    av_temp_cfg = _avaliar_sequencia_limiar(forecast, "temperature_c", limiar_temp_cfg, "gt", "°C")
    av_umid_cfg = _avaliar_sequencia_limiar(forecast, "humidity_pct", limiar_umid_cfg, "gt", "%")

    # Blocos detalhados por cultura e referência geral do viveiro
    geral_calor_28 = _avaliar_sequencia_limiar(forecast, "temperature_c", 28.0, "ge", "°C")
    geral_calor_32 = _avaliar_sequencia_limiar(forecast, "temperature_c", 32.0, "ge", "°C")

    alface_calor_26 = _avaliar_sequencia_limiar(forecast, "temperature_c", 26.0, "gt", "°C")
    alface_umid_alta_80 = _avaliar_sequencia_limiar(forecast, "humidity_pct", 80.0, "gt", "%")
    alface_umid_baixa_60 = _avaliar_sequencia_limiar(forecast, "humidity_pct", 60.0, "lt", "%")

    rosa_umid_60 = _avaliar_sequencia_limiar(forecast, "humidity_pct", 60.0, "gt", "%")
    rosa_frio_14 = _avaliar_sequencia_limiar(forecast, "temperature_c", 14.0, "lt", "°C")

    salsinha_umid_75 = _avaliar_sequencia_limiar(forecast, "humidity_pct", 75.0, "gt", "%")
    salsinha_calor_24 = _avaliar_sequencia_limiar(forecast, "temperature_c", 24.0, "gt", "°C")
    salsinha_calor_28 = _avaliar_sequencia_limiar(forecast, "temperature_c", 28.0, "gt", "°C")

    thresholds_by_context = {
        "geral_viveiro": {
            "label": "Referência Geral do Viveiro",
            "heat_attention_ge_28c": geral_calor_28,
            "heat_critical_ge_32c": geral_calor_32,
        },
        "alface_crespa": {
            "crop_id": "alface-crespa",
            "crop_name": "Alface Crespa",
            "optimal_temperature_range": "15°C - 25°C",
            "optimal_humidity_range": "60% - 80%",
            "heat_stress_gt_26c": alface_calor_26,
            "humidity_high_gt_80pct": alface_umid_alta_80,
            "humidity_low_lt_60pct": alface_umid_baixa_60,
        },
        "rosa_do_deserto": {
            "crop_id": "Rosa-do-deserto",
            "crop_name": "Rosa do Deserto",
            "optimal_temperature_range": "25°C - 35°C",
            "optimal_humidity_range": "30% - 60%",
            "humidity_excess_gt_60pct": rosa_umid_60,
            "cold_stress_lt_14c": rosa_frio_14,
        },
        "salsinha": {
            "crop_id": "salsinha",
            "crop_name": "Salsinha",
            "optimal_temperature_range": "10°C - 24°C",
            "optimal_humidity_range": "60% - 75%",
            "humidity_excess_gt_75pct": salsinha_umid_75,
            "heat_attention_gt_24c": salsinha_calor_24,
            "heat_stress_gt_28c": salsinha_calor_28,
        },
    }

    summary_block = SummaryBlock(
        temperature=TemperatureSummary(
            minimum_c=round(t_min, 1),
            maximum_c=round(t_max, 1),
            mean_c=t_mean,
            minimum_at=t_min_at,
            maximum_at=t_max_at,
            configured_threshold_c=round(limiar_temp_cfg, 1),
            hours_above_configured_threshold=av_temp_cfg["hours_count"],
            longest_sequence_above_threshold=av_temp_cfg["longest_sequence_hours"],
            first_event_at=av_temp_cfg["first_event_at"],
        ),
        humidity=HumiditySummary(
            minimum_pct=round(u_min, 1),
            maximum_pct=round(u_max, 1),
            mean_pct=u_mean,
            minimum_at=u_min_at,
            maximum_at=u_max_at,
            configured_threshold_pct=round(limiar_umid_cfg, 1),
            hours_above_configured_threshold=av_umid_cfg["hours_count"],
            longest_sequence_above_threshold=av_umid_cfg["longest_sequence_hours"],
            first_event_at=av_umid_cfg["first_event_at"],
        ),
        thresholds_by_context=thresholds_by_context,
    )

    # Indicadores de tendência (média das próximas 6h previstas vs últimas 6h observadas)
    obs_ultimas_6_temp = [o.temperature_c for o in entrada.observations[-6:]]
    obs_ultimas_6_umid = [o.humidity_pct for o in entrada.observations[-6:]]
    prev_primeiras_6_temp = temps[:6]
    prev_primeiras_6_umid = umids[:6]

    delta_temp_curto = round(
        (sum(prev_primeiras_6_temp) / len(prev_primeiras_6_temp))
        - (sum(obs_ultimas_6_temp) / len(obs_ultimas_6_temp)),
        1,
    )
    delta_umid_curto = round(
        (sum(prev_primeiras_6_umid) / len(prev_primeiras_6_umid))
        - (sum(obs_ultimas_6_umid) / len(obs_ultimas_6_umid)),
        1,
    )

    indicators_block = IndicatorsBlock(
        temperature_trend=_classificar_tendencia(delta_temp_curto, tolerancia=0.8),
        humidity_trend=_classificar_tendencia(delta_umid_curto, tolerancia=3.0),
        temperature_short_term_delta_c=delta_temp_curto,
        humidity_short_term_delta_pct=delta_umid_curto,
        temperature_amplitude_c=round(t_max - t_min, 1),
        humidity_amplitude_pct=round(u_max - u_min, 1),
        combined_heat_humidity_periods=_identificar_periodos_combinados(forecast),
    )

    # Alertas preditivos acionados para o viveiro e por cultura
    alerts: list[AlertItem] = []

    if geral_calor_32["hours_count"] > 0:
        alerts.append(
            AlertItem(
                code="VIVEIRO_EXTREME_HEAT_FORECAST",
                crop_id="geral",
                crop_name="Viveiro Bioterra (Geral)",
                severity="danger",
                title="Calor Crítico Previsto nas Próximas 24h",
                message=(
                    f"Modelo prevê {geral_calor_32['hours_count']}h com temperatura >= 32°C "
                    f"(maior sequência: {geral_calor_32['longest_sequence_hours']}h; "
                    f"pico de {t_max:.1f}°C). Reforçar sombreamento e ventilação."
                ),
                metric="temperature_c",
                threshold=32.0,
                unit="°C",
                hours_affected=geral_calor_32["hours_count"],
                longest_sequence_hours=geral_calor_32["longest_sequence_hours"],
                first_event_at=geral_calor_32["first_event_at"],
                peak_value=t_max,
                peak_at=t_max_at,
            )
        )
    elif geral_calor_28["hours_count"] > 0:
        alerts.append(
            AlertItem(
                code="VIVEIRO_MODERATE_HEAT_FORECAST",
                crop_id="geral",
                crop_name="Viveiro Bioterra (Geral)",
                severity="warning",
                title="Atenção Térmica nas Próximas 24h",
                message=(
                    f"Previstas {geral_calor_28['hours_count']}h com temperatura >= 28°C "
                    f"(sequência máxima: {geral_calor_28['longest_sequence_hours']}h; pico de {t_max:.1f}°C)."
                ),
                metric="temperature_c",
                threshold=28.0,
                unit="°C",
                hours_affected=geral_calor_28["hours_count"],
                longest_sequence_hours=geral_calor_28["longest_sequence_hours"],
                first_event_at=geral_calor_28["first_event_at"],
                peak_value=t_max,
                peak_at=t_max_at,
            )
        )

    if alface_calor_26["hours_count"] > 0:
        alerts.append(
            AlertItem(
                code="ALFACE_HEAT_STRESS_FORECAST",
                crop_id="alface-crespa",
                crop_name="Alface Crespa",
                severity="danger" if alface_calor_26["longest_sequence_hours"] >= 4 or t_max >= 30.0 else "warning",
                title="Risco Preditivo de Tipburn / Estresse Térmico (Alface Crespa)",
                message=(
                    f"Previstas {alface_calor_26['hours_count']}h acima de 26°C nas próximas 24h "
                    f"(maior sequência: {alface_calor_26['longest_sequence_hours']}h, "
                    f"pico de {t_max:.1f}°C). Atenção ao manejo de sombreamento e irrigação."
                ),
                metric="temperature_c",
                threshold=26.0,
                unit="°C",
                hours_affected=alface_calor_26["hours_count"],
                longest_sequence_hours=alface_calor_26["longest_sequence_hours"],
                first_event_at=alface_calor_26["first_event_at"],
                peak_value=t_max,
                peak_at=t_max_at,
            )
        )

    if rosa_umid_60["hours_count"] > 0:
        alerts.append(
            AlertItem(
                code="ROSA_HUMIDITY_EXCESS_FORECAST",
                crop_id="Rosa-do-deserto",
                crop_name="Rosa do Deserto",
                severity="danger" if rosa_umid_60["longest_sequence_hours"] >= 8 or u_max >= 80.0 else "warning",
                title="Excesso de Umidade Previsto (Rosa do Deserto)",
                message=(
                    f"Previstas {rosa_umid_60['hours_count']}h com umidade > 60% "
                    f"(maior sequência: {rosa_umid_60['longest_sequence_hours']}h, "
                    f"máxima de {u_max:.1f}%). Manter substrato drenado e ventilação ativa."
                ),
                metric="humidity_pct",
                threshold=60.0,
                unit="%",
                hours_affected=rosa_umid_60["hours_count"],
                longest_sequence_hours=rosa_umid_60["longest_sequence_hours"],
                first_event_at=rosa_umid_60["first_event_at"],
                peak_value=u_max,
                peak_at=u_max_at,
            )
        )

    if rosa_frio_14["hours_count"] > 0:
        alerts.append(
            AlertItem(
                code="ROSA_COLD_STRESS_FORECAST",
                crop_id="Rosa-do-deserto",
                crop_name="Rosa do Deserto",
                severity="danger",
                title="Alerta de Frio Previsto (Rosa do Deserto)",
                message=(
                    f"Previstas {rosa_frio_14['hours_count']}h abaixo de 14°C "
                    f"(mínima prevista de {t_min:.1f}°C). Proteger lotes sensíveis."
                ),
                metric="temperature_c",
                threshold=14.0,
                unit="°C",
                hours_affected=rosa_frio_14["hours_count"],
                longest_sequence_hours=rosa_frio_14["longest_sequence_hours"],
                first_event_at=rosa_frio_14["first_event_at"],
                peak_value=t_min,
                peak_at=t_min_at,
            )
        )

    if salsinha_umid_75["hours_count"] > 0:
        alerts.append(
            AlertItem(
                code="SALSINHA_HUMIDITY_EXCESS_FORECAST",
                crop_id="salsinha",
                crop_name="Salsinha",
                severity="warning" if salsinha_umid_75["longest_sequence_hours"] < 8 else "danger",
                title="Umidade Elevada Prevista (Salsinha)",
                message=(
                    f"Previstas {salsinha_umid_75['hours_count']}h com umidade > 75% "
                    f"(maior sequência: {salsinha_umid_75['longest_sequence_hours']}h, "
                    f"pico de {u_max:.1f}%). Evitar rega foliar noturna para prevenir míldio."
                ),
                metric="humidity_pct",
                threshold=75.0,
                unit="%",
                hours_affected=salsinha_umid_75["hours_count"],
                longest_sequence_hours=salsinha_umid_75["longest_sequence_hours"],
                first_event_at=salsinha_umid_75["first_event_at"],
                peak_value=u_max,
                peak_at=u_max_at,
            )
        )

    if salsinha_calor_28["hours_count"] > 0:
        alerts.append(
            AlertItem(
                code="SALSINHA_HEAT_STRESS_FORECAST",
                crop_id="salsinha",
                crop_name="Salsinha",
                severity="warning",
                title="Calor Acima da Faixa Tolerada (Salsinha)",
                message=(
                    f"Previstas {salsinha_calor_28['hours_count']}h acima de 28°C "
                    f"(maior sequência: {salsinha_calor_28['longest_sequence_hours']}h, pico de {t_max:.1f}°C)."
                ),
                metric="temperature_c",
                threshold=28.0,
                unit="°C",
                hours_affected=salsinha_calor_28["hours_count"],
                longest_sequence_hours=salsinha_calor_28["longest_sequence_hours"],
                first_event_at=salsinha_calor_28["first_event_at"],
                peak_value=t_max,
                peak_at=t_max_at,
            )
        )

    warnings = [
        WarningItem(
            code="EXTERNAL_REFERENCE_CONDITIONS",
            message="A previsão representa condições meteorológicas externas de referência para o viveiro.",
        ),
        WarningItem(
            code="SOURCE_DOMAIN_SHIFT",
            message="O modelo foi treinado com dados históricos do INMET e recebe, na operação, observações coletadas da HG Brasil.",
        ),
    ]

    if entrada.latest_observation_age_minutes > 60:
        warnings.append(
            WarningItem(
                code="OBSERVATION_DELAY_NOTICE",
                message=(
                    f"A última observação utilizada possui idade de {entrada.latest_observation_age_minutes} minutos "
                    "em relação ao instante de geração da previsão."
                ),
            )
        )

    return (
        summary_block if options.include_summary else None,
        indicators_block if options.include_indicators else None,
        alerts,
        warnings,
    )

