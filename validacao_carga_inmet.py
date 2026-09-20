from __future__ import annotations

import datetime
from pathlib import Path
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

COLUNA_ALVO = "umidade_max_hora_anterior_pct"

MAPA_COLUNAS = {
    "DATA (YYYY-MM-DD)": "data",
    "Data": "data",
    "DATA": "data",
    "HORA (UTC)": "hora_utc",
    "Hora UTC": "hora_utc",
    "HORA UTC": "hora_utc",
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "precipitacao_total_horario_mm",
    "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": "pressao_estacao_horaria_mb",
    "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)": "pressao_max_hora_anterior_mb",
    "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)": "pressao_min_hora_anterior_mb",
    "RADIACAO GLOBAL (KJ/m²)": "radiacao_global_kj_m2",
    "RADIACAO GLOBAL (Kj/m²)": "radiacao_global_kj_m2",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "temperatura_ar_horaria_c",
    "TEMPERATURA DO PONTO DE ORVALHO (°C)": "temperatura_ponto_orvalho_c",
    "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)": "temperatura_max_hora_anterior_c",
    "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)": "temperatura_min_hora_anterior_c",
    "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)": "temperatura_orvalho_max_hora_anterior_c",
    "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)": "temperatura_orvalho_min_hora_anterior_c",
    "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)": "umidade_max_hora_anterior_pct",
    "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)": "umidade_min_hora_anterior_pct",
    "UMIDADE RELATIVA DO AR, HORARIA (%)": "umidade_horaria_pct",
    "VENTO, DIREÇÃO HORARIA (gr) (° (gr))": "vento_direcao_graus",
    "VENTO, RAJADA MAXIMA (m/s)": "vento_rajada_max_ms",
    "VENTO, VELOCIDADE HORARIA (m/s)": "vento_velocidade_horaria_ms",
}

COLUNAS_OBRIGATORIAS = set(MAPA_COLUNAS.values())
COLUNAS_OBRIGATORIAS.update({"data", "hora_utc"})

FAIXAS_BASICAS = {
    "precipitacao_total_horario_mm": (0, 500),
    "pressao_estacao_horaria_mb": (850, 1100),
    "pressao_max_hora_anterior_mb": (850, 1100),
    "pressao_min_hora_anterior_mb": (850, 1100),
    "temperatura_ar_horaria_c": (-20, 60),
    "temperatura_ponto_orvalho_c": (-40, 40),
    "temperatura_max_hora_anterior_c": (-20, 60),
    "temperatura_min_hora_anterior_c": (-20, 60),
    "temperatura_orvalho_max_hora_anterior_c": (-40, 40),
    "temperatura_orvalho_min_hora_anterior_c": (-40, 40),
    "umidade_max_hora_anterior_pct": (0, 100),
    "umidade_min_hora_anterior_pct": (0, 100),
    "umidade_horaria_pct": (0, 100),
    "vento_direcao_graus": (0, 360),
    "vento_rajada_max_ms": (0, 100),
    "vento_velocidade_horaria_ms": (0, 75),
}


def normalizar_chave_metadado(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.upper().strip().rstrip(":")
    texto = re.sub(r"\s+", " ", texto)
    return texto


def ler_metadados(caminho: Path) -> dict[str, str]:
    linhas = caminho.read_text(encoding="latin-1").splitlines()[:8]
    metadados: dict[str, str] = {}
    for linha in linhas:
        partes = linha.split(";", 1)
        if len(partes) == 2:
            chave = normalizar_chave_metadado(partes[0])
            metadados[chave] = partes[1].strip()
    return metadados


def obter_wmo(metadados: dict[str, str]) -> str | None:
    for chave, valor in metadados.items():
        if "WMO" in chave:
            return valor.strip()
    return None


def criar_datetime(data: pd.Series, hora: pd.Series) -> pd.Series:
    data_convertida = pd.to_datetime(data, errors="coerce")
    hora_limpa = (
        hora.astype(str)
        .str.replace(" UTC", "", regex=False)
        .str.replace(":", "", regex=False)
        .str.strip()
        .str.zfill(4)
    )
    texto = data_convertida.dt.strftime("%Y-%m-%d") + " " + hora_limpa
    return pd.to_datetime(
        texto,
        format="%Y-%m-%d %H%M",
        errors="coerce",
        utc=True,
    )


def maior_bloco_ausente(serie: pd.Series) -> int:
    ausente = serie.isna()
    if not ausente.any():
        return 0
    grupos = ausente.ne(ausente.shift()).cumsum()
    tamanhos = ausente.groupby(grupos).sum()
    return int(tamanhos.max())


def ler_e_validar_arquivo(
    caminho: str | Path,
    wmo_esperado: str = "A707",
    limite_ausencia_alvo: float = 0.20,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    caminho = Path(caminho)
    metadados = ler_metadados(caminho)
    wmo = obter_wmo(metadados)

    df = pd.read_csv(
        caminho,
        encoding="latin-1",
        skiprows=8,
        sep=";",
        decimal=",",
        na_values=["-9999", "-9999,0", -9999, -9999.0],
        low_memory=False,
    )

    df = df.drop(
        columns=[c for c in df.columns if str(c).startswith("Unnamed:")],
        errors="ignore",
    )

    colunas_originais = list(df.columns)
    desconhecidas = [c for c in colunas_originais if c not in MAPA_COLUNAS]
    df = df.rename(columns=MAPA_COLUNAS)

    duplicadas_canonicas = df.columns[df.columns.duplicated()].tolist()
    presentes = set(df.columns)
    faltantes = sorted(COLUNAS_OBRIGATORIAS - presentes)

    erro_estrutural = bool(
        wmo != wmo_esperado
        or desconhecidas
        or duplicadas_canonicas
        or faltantes
    )

    if "data" in df.columns and "hora_utc" in df.columns:
        df["datetime"] = criar_datetime(df["data"], df["hora_utc"])
    else:
        df["datetime"] = pd.NaT

    colunas_numericas = [
        c for c in df.columns
        if c not in {"data", "hora_utc", "datetime"}
    ]
    for coluna in colunas_numericas:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    timestamps_validos = pd.DatetimeIndex(df["datetime"].dropna())
    duplicatas = int(df["datetime"].duplicated().sum())
    timestamps_invalidos = int(df["datetime"].isna().sum())

    if len(timestamps_validos):
        inicio = timestamps_validos.min()
        fim = timestamps_validos.max()
        grade = pd.date_range(inicio, fim, freq="h")
        lacunas = grade.difference(timestamps_validos.unique())
    else:
        inicio = pd.NaT
        fim = pd.NaT
        lacunas = pd.DatetimeIndex([])

    alvo_ausente = (
        float(df[COLUNA_ALVO].isna().mean())
        if COLUNA_ALVO in df.columns and len(df)
        else 1.0
    )
    maior_bloco_alvo = (
        maior_bloco_ausente(df[COLUNA_ALVO])
        if COLUNA_ALVO in df.columns
        else len(df)
    )

    violacoes_total = 0
    linhas_colunas = []
    for coluna in colunas_numericas:
        serie = df[coluna]
        minimo = serie.min(skipna=True)
        maximo = serie.max(skipna=True)
        violacoes = 0
        if coluna in FAIXAS_BASICAS:
            limite_min, limite_max = FAIXAS_BASICAS[coluna]
            violacoes = int(
                ((serie < limite_min) | (serie > limite_max)).sum()
            )
        violacoes_total += violacoes
        linhas_colunas.append({
            "arquivo": caminho.name,
            "coluna": coluna,
            "n_registros": len(df),
            "n_validos": int(serie.notna().sum()),
            "n_ausentes": int(serie.isna().sum()),
            "percentual_ausente": float(serie.isna().mean() * 100),
            "minimo": minimo,
            "maximo": maximo,
            "violacoes_faixa": violacoes,
            "maior_bloco_ausente": maior_bloco_ausente(serie),
        })

    aprovado_modelagem = bool(
        not erro_estrutural
        and timestamps_invalidos == 0
        and alvo_ausente <= limite_ausencia_alvo
        and violacoes_total == 0
    )

    resumo = {
        "arquivo": caminho.name,
        "wmo": wmo,
        "estacao": metadados.get("ESTACAO"),
        "latitude": metadados.get("LATITUDE"),
        "longitude": metadados.get("LONGITUDE"),
        "altitude": metadados.get("ALTITUDE"),
        "n_linhas": len(df),
        "n_colunas_originais": len(colunas_originais),
        "inicio": inicio,
        "fim": fim,
        "timestamps_invalidos": timestamps_invalidos,
        "timestamps_duplicados": duplicatas,
        "lacunas_horarias": len(lacunas),
        "ausencia_alvo_pct": alvo_ausente * 100,
        "maior_bloco_ausente_alvo_horas": maior_bloco_alvo,
        "colunas_desconhecidas": " | ".join(desconhecidas),
        "colunas_faltantes": " | ".join(faltantes),
        "colunas_canonicas_duplicadas": " | ".join(duplicadas_canonicas),
        "violacoes_faixa_total": violacoes_total,
        "erro_estrutural": erro_estrutural,
        "aprovado_modelagem": aprovado_modelagem,
    }

    df["arquivo_origem"] = caminho.name
    return df, resumo, pd.DataFrame(linhas_colunas)


def carregar_e_validar_diretorio(
    pasta: str | Path = ".\\estacao_A707",
    padrao: str = "INMET*.CSV",
    wmo_esperado: str = "A707",
    limite_ausencia_alvo: float = 0.20,
    excluir_anos_incompletos: bool = True,
) -> dict[str, Any]:
    pasta = Path(pasta)
    arquivos = sorted(pasta.glob(padrao))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado em {pasta.resolve()} com {padrao}"
        )

    dados_aprovados = []
    resumos = []
    relatorios_colunas = []

    for caminho in arquivos:
        df, resumo, rel_colunas = ler_e_validar_arquivo(
            caminho,
            wmo_esperado=wmo_esperado,
            limite_ausencia_alvo=limite_ausencia_alvo,
        )
        resumos.append(resumo)
        relatorios_colunas.append(rel_colunas)

        incluir = resumo["aprovado_modelagem"] or not excluir_anos_incompletos
        if incluir and not resumo["erro_estrutural"]:
            dados_aprovados.append(df)

    relatorio_arquivos = pd.DataFrame(resumos)
    relatorio_colunas = pd.concat(relatorios_colunas, ignore_index=True)

    if dados_aprovados:
        dados = pd.concat(dados_aprovados, ignore_index=True, sort=False)
        dados = dados.sort_values("datetime").reset_index(drop=True)
        duplicatas_globais = dados["datetime"].duplicated(keep=False)
        if duplicatas_globais.any():
            exemplos = dados.loc[
                duplicatas_globais,
                ["datetime", "arquivo_origem"],
            ].head(20)
            raise ValueError(
                "Foram encontrados timestamps repetidos entre arquivos:\n"
                + exemplos.to_string(index=False)
            )
    else:
        dados = pd.DataFrame()

    return {
        "dados": dados,
        "relatorio_arquivos": relatorio_arquivos,
        "relatorio_colunas": relatorio_colunas,
    }


def main() -> None:
    resultado = carregar_e_validar_diretorio()
    rel_arq = resultado["relatorio_arquivos"]
    rel_col = resultado["relatorio_colunas"]
    dados = resultado["dados"]

    rel_arq.to_csv(
        f"relatorio_arquivos_inmet_{datetime.date.today().strftime('%Y-%m-%d')}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rel_col.to_csv(
        f"relatorio_colunas_inmet_{datetime.date.today().strftime('%Y-%m-%d')}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not dados.empty:
        dados.to_csv(
            f"dados_inmet_consolidados_{datetime.date.today().strftime('%Y-%m-%d')}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(rel_arq.to_string(index=False))
    print(f"\nLinhas consolidadas aprovadas: {len(dados)}")


if __name__ == "__main__":
    main()
