from __future__ import annotations

import datetime
from io import StringIO
from pathlib import Path
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

# Coluna usada apenas quando o usuario quiser avaliar a elegibilidade
# dos arquivos para um alvo especifico. A validacao estrutural e geral.
COLUNA_ALVO_PADRAO = "umidade_max_hora_anterior_pct"

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

COLUNAS_TEMPORAIS = {"data", "hora_utc"}
COLUNAS_METEOROLOGICAS = set(MAPA_COLUNAS.values()) - COLUNAS_TEMPORAIS
COLUNAS_OBRIGATORIAS = set(MAPA_COLUNAS.values())

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

# Corrige somente campos cujo conteudo comeca por virgula e e seguido por digito.
# Exemplo: ;,3; -> ;0,3;. Nao altera 1,3 ou campos vazios.
PADRAO_DECIMAL_SEM_ZERO = re.compile(r"(?<=;),(?=\d)")


def normalizar_chave_metadado(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.upper().strip().rstrip(":")
    return re.sub(r"\s+", " ", texto)


def ler_texto_e_corrigir_decimais(
    caminho: Path,
    encoding: str = "latin-1",
) -> tuple[str, int, list[int]]:
    """Le o CSV e converte campos ',numero' em '0,numero' nas linhas de dados."""
    linhas = caminho.read_text(encoding=encoding).splitlines()
    linhas_corrigidas = linhas[:9]
    total_correcoes = 0
    linhas_afetadas: list[int] = []

    for numero_linha, linha in enumerate(linhas[9:], start=10):
        linha_corrigida, quantidade = PADRAO_DECIMAL_SEM_ZERO.subn("0,", linha)
        linhas_corrigidas.append(linha_corrigida)
        if quantidade:
            total_correcoes += quantidade
            linhas_afetadas.append(numero_linha)

    return "\n".join(linhas_corrigidas) + "\n", total_correcoes, linhas_afetadas


def salvar_copia_preprocessada(
    caminho_original: Path,
    texto_preprocessado: str,
    pasta_destino: str | Path,
    encoding: str = "latin-1",
) -> Path:
    """Salva uma copia auditavel sem alterar o arquivo bruto original."""
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)
    destino = pasta_destino / caminho_original.name
    destino.write_text(texto_preprocessado, encoding=encoding, newline="")
    return destino


def ler_metadados_do_texto(texto: str) -> dict[str, str]:
    metadados: dict[str, str] = {}
    for linha in texto.splitlines()[:8]:
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
    coluna_alvo_criterio: str | None = None,
    limite_ausencia_alvo: float = 0.20,
    encoding: str = "latin-1",
    pasta_copias_preprocessadas: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    caminho = Path(caminho)

    texto, n_correcoes, linhas_corrigidas = ler_texto_e_corrigir_decimais(
        caminho,
        encoding=encoding,
    )
    caminho_copia_preprocessada = None
    if pasta_copias_preprocessadas is not None:
        caminho_copia_preprocessada = salvar_copia_preprocessada(
            caminho,
            texto,
            pasta_copias_preprocessadas,
            encoding=encoding,
        )

    metadados = ler_metadados_do_texto(texto)
    wmo = obter_wmo(metadados)

    df = pd.read_csv(
        StringIO(texto),
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
    faltantes = sorted(COLUNAS_OBRIGATORIAS - set(df.columns))

    erro_estrutural = bool(
        wmo != wmo_esperado
        or desconhecidas
        or duplicadas_canonicas
        or faltantes
    )

    if COLUNAS_TEMPORAIS.issubset(df.columns):
        df["datetime"] = criar_datetime(df["data"], df["hora_utc"])
    else:
        df["datetime"] = pd.NaT

    colunas_numericas = [
        c for c in df.columns if c not in {"data", "hora_utc", "datetime"}
    ]
    for coluna in colunas_numericas:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    timestamps_validos = pd.DatetimeIndex(df["datetime"].dropna())
    timestamps_invalidos = int(df["datetime"].isna().sum())
    duplicatas = int(df["datetime"].duplicated().sum())

    if len(timestamps_validos):
        inicio = timestamps_validos.min()
        fim = timestamps_validos.max()
        grade = pd.date_range(inicio, fim, freq="h")
        lacunas = grade.difference(timestamps_validos.unique())
    else:
        inicio = pd.NaT
        fim = pd.NaT
        lacunas = pd.DatetimeIndex([])

    linhas_colunas = []
    violacoes_total = 0
    colunas_vazias_arquivo = []

    for coluna in colunas_numericas:
        serie = df[coluna]
        n_validos = int(serie.notna().sum())
        vazia = n_validos == 0
        if vazia:
            colunas_vazias_arquivo.append(coluna)

        violacoes = 0
        if coluna in FAIXAS_BASICAS:
            limite_min, limite_max = FAIXAS_BASICAS[coluna]
            violacoes = int(((serie < limite_min) | (serie > limite_max)).sum())
        violacoes_total += violacoes

        linhas_colunas.append({
            "arquivo": caminho.name,
            "coluna": coluna,
            "n_registros": len(df),
            "n_validos": n_validos,
            "n_ausentes": int(serie.isna().sum()),
            "percentual_ausente": float(serie.isna().mean() * 100),
            "minimo": serie.min(skipna=True),
            "maximo": serie.max(skipna=True),
            "violacoes_faixa": violacoes,
            "maior_bloco_ausente": maior_bloco_ausente(serie),
            "coluna_vazia_no_arquivo": vazia,
        })

    aprovado_estrutural = bool(
        not erro_estrutural
        and timestamps_invalidos == 0
        and violacoes_total == 0
    )

    ausencia_alvo_pct = np.nan
    maior_bloco_alvo = np.nan
    aprovado_modelagem_alvo: bool | None = None
    if coluna_alvo_criterio is not None:
        if coluna_alvo_criterio in df.columns:
            serie_alvo = df[coluna_alvo_criterio]
            ausencia_alvo = float(serie_alvo.isna().mean())
            ausencia_alvo_pct = ausencia_alvo * 100
            maior_bloco_alvo = maior_bloco_ausente(serie_alvo)
            aprovado_modelagem_alvo = bool(
                aprovado_estrutural
                and ausencia_alvo <= limite_ausencia_alvo
                and serie_alvo.notna().any()
            )
        else:
            aprovado_modelagem_alvo = False

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
        "n_correcoes_decimal_sem_zero": n_correcoes,
        "n_linhas_com_correcao_decimal": len(linhas_corrigidas),
        "colunas_vazias_no_arquivo": " | ".join(colunas_vazias_arquivo),
        "n_colunas_vazias_no_arquivo": len(colunas_vazias_arquivo),
        "coluna_alvo_criterio": coluna_alvo_criterio,
        "ausencia_alvo_pct": ausencia_alvo_pct,
        "maior_bloco_ausente_alvo_horas": maior_bloco_alvo,
        "colunas_desconhecidas": " | ".join(desconhecidas),
        "colunas_faltantes": " | ".join(faltantes),
        "colunas_canonicas_duplicadas": " | ".join(duplicadas_canonicas),
        "violacoes_faixa_total": violacoes_total,
        "erro_estrutural": erro_estrutural,
        "aprovado_estrutural": aprovado_estrutural,
        "aprovado_modelagem_alvo": aprovado_modelagem_alvo,
    }

    registro_preprocessamento = {
        "arquivo": caminho.name,
        "etapa": "correcao_decimal_sem_zero_inicial",
        "regra": "campo no formato ,numero convertido para 0,numero",
        "regex": r"(?<=;),(?=\d)",
        "n_valores_corrigidos": n_correcoes,
        "n_linhas_afetadas": len(linhas_corrigidas),
        "primeiras_linhas_afetadas": " | ".join(map(str, linhas_corrigidas[:20])),
        "arquivo_original_alterado": False,
        "copia_preprocessada": (
            str(caminho_copia_preprocessada)
            if caminho_copia_preprocessada is not None
            else ""
        ),
    }

    df["arquivo_origem"] = caminho.name
    return df, resumo, pd.DataFrame(linhas_colunas), registro_preprocessamento


def resumir_consistencia_global_colunas(
    relatorio_colunas: pd.DataFrame,
) -> pd.DataFrame:
    linhas = []
    for coluna, grupo in relatorio_colunas.groupby("coluna", sort=True):
        n_total = int(grupo["n_registros"].sum())
        n_validos = int(grupo["n_validos"].sum())
        n_ausentes = int(grupo["n_ausentes"].sum())
        n_arquivos = int(grupo["arquivo"].nunique())
        n_arquivos_vazios = int(grupo["coluna_vazia_no_arquivo"].sum())
        percentual = 100 * n_ausentes / n_total if n_total else 100.0

        if n_validos == 0:
            sugestao = "EXCLUIR: coluna totalmente vazia em todos os arquivos carregados"
        elif n_arquivos_vazios == n_arquivos:
            sugestao = "EXCLUIR: sem dados válidos em qualquer arquivo"
        elif percentual >= 80:
            sugestao = "REVISAR: ausência global muito elevada"
        elif n_arquivos_vazios > 0:
            sugestao = "MANTER COM RESSALVA: vazia em alguns anos; avaliar período do modelo"
        else:
            sugestao = "MANTER: possui dados válidos"

        linhas.append({
            "coluna": coluna,
            "n_arquivos": n_arquivos,
            "n_arquivos_com_dados": n_arquivos - n_arquivos_vazios,
            "n_arquivos_vazios": n_arquivos_vazios,
            "n_registros_total": n_total,
            "n_validos_total": n_validos,
            "n_ausentes_total": n_ausentes,
            "percentual_ausente_global": percentual,
            "minimo_global": grupo["minimo"].min(skipna=True),
            "maximo_global": grupo["maximo"].max(skipna=True),
            "violacoes_faixa_total": int(grupo["violacoes_faixa"].sum()),
            "sugestao": sugestao,
        })

    return pd.DataFrame(linhas)


def colunas_totalmente_vazias(dados: pd.DataFrame) -> list[str]:
    colunas = [c for c in COLUNAS_METEOROLOGICAS if c in dados.columns]
    return [c for c in sorted(colunas) if dados[c].notna().sum() == 0]


def carregar_e_validar_diretorio(
    pasta: str | Path = ".\\estacao_A707",
    padrao: str = "INMET*.CSV",
    wmo_esperado: str = "A707",
    coluna_alvo_criterio: str | None = None,
    limite_ausencia_alvo: float = 0.20,
    excluir_arquivos_com_erro_estrutural: bool = True,
    encoding: str = "latin-1",
    salvar_copias_preprocessadas: bool = True,
) -> dict[str, Any]:
    pasta = Path(pasta)
    arquivos = sorted(pasta.glob(padrao))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado em {pasta.resolve()} com {padrao}"
        )

    pasta_copias_preprocessadas = (
        pasta / "preprocessados"
        if salvar_copias_preprocessadas
        else None
    )

    dados_incluidos = []
    resumos = []
    relatorios_colunas = []
    registros_preprocessamento = []

    for caminho in arquivos:
        df, resumo, rel_colunas, registro = ler_e_validar_arquivo(
            caminho,
            wmo_esperado=wmo_esperado,
            coluna_alvo_criterio=coluna_alvo_criterio,
            limite_ausencia_alvo=limite_ausencia_alvo,
            encoding=encoding,
            pasta_copias_preprocessadas=pasta_copias_preprocessadas,
        )
        resumos.append(resumo)
        relatorios_colunas.append(rel_colunas)
        registros_preprocessamento.append(registro)

        incluir = resumo["aprovado_estrutural"] or not excluir_arquivos_com_erro_estrutural
        if incluir:
            dados_incluidos.append(df)

    relatorio_arquivos = pd.DataFrame(resumos)
    relatorio_colunas = pd.concat(relatorios_colunas, ignore_index=True)
    relatorio_preprocessamento = pd.DataFrame(registros_preprocessamento)
    relatorio_global_colunas = resumir_consistencia_global_colunas(relatorio_colunas)

    if dados_incluidos:
        dados = pd.concat(dados_incluidos, ignore_index=True, sort=False)
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

    colunas_vazias_globais = colunas_totalmente_vazias(dados) if not dados.empty else []

    return {
        "dados": dados,
        "relatorio_arquivos": relatorio_arquivos,
        "relatorio_colunas": relatorio_colunas,
        "relatorio_global_colunas": relatorio_global_colunas,
        "relatorio_preprocessamento": relatorio_preprocessamento,
        "colunas_vazias_globais": colunas_vazias_globais,
    }


def main() -> None:
    resultado = carregar_e_validar_diretorio()
    data_execucao = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    saidas = {
        f"relatorio_arquivos_inmet_{data_execucao}.csv": resultado["relatorio_arquivos"],
        f"relatorio_colunas_inmet_{data_execucao}.csv": resultado["relatorio_colunas"],
        f"relatorio_global_colunas_inmet_{data_execucao}.csv": resultado["relatorio_global_colunas"],
        f"registro_preprocessamento_inmet_{data_execucao}.csv": resultado["relatorio_preprocessamento"],
    }
    for nome, tabela in saidas.items():
        tabela.to_csv(nome, index=False, sep=";", encoding="utf-8-sig")

    dados = resultado["dados"]
    if not dados.empty:
        dados.to_csv(
            f"dados_inmet_consolidados_{data_execucao}.csv",
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )

    print(resultado["relatorio_arquivos"].to_string(index=False))
    print("\nConsistencia global das colunas:")
    print(resultado["relatorio_global_colunas"].to_string(index=False))
    print(f"\nLinhas consolidadas: {len(dados)}")
    print("Colunas totalmente vazias no consolidado:", resultado["colunas_vazias_globais"])


if __name__ == "__main__":
    main()
