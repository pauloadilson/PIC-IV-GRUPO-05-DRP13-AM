import requests
import pandas as pd
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def obter_historico_inmet(
    estacao: str = "A707",
    data_inicio: str = "2024-01-01",
    data_fim: str = "2024-01-15"
) -> pd.DataFrame:
    url = f"https://apitempo.inmet.gov.br/estacao/{data_inicio}/{data_fim}/{estacao}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; Python-Request/3.0)",
    }
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))

    try:
        response = session.get(url, headers=headers, timeout=(10, 30))
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Falha ao consultar o INMET em {url}: {exc}") from exc

    if response.status_code == 204 or not response.content.strip():
        raise ValueError(
            "O INMET não retornou dados para a estação e o período especificados."
        )

    try:
        dados = response.json()
    except ValueError as exc:
        raise RuntimeError("A resposta do INMET não contém um JSON válido.") from exc
    
    if not dados:
        raise ValueError("Nenhum registro retornado pelo INMET para o período especificado.")
        
    df = pd.DataFrame(dados)
    
    # Tratamento da data e hora (HR_MEDICAO vem em formato UTC como '0000', '0100', etc.)
    df["data_hora_utc"] = pd.to_datetime(
        df["DT_MEDICAO"] + " " + df["HR_MEDICAO"].str.zfill(4),
        format="%Y-%m-%d %H%M"
    )
    # Conversão para o fuso local (UTC-3)
    df["data_hora_local"] = df["data_hora_utc"].dt.tz_localize("UTC").dt.tz_convert("America/Sao_Paulo")
    
    # Conversão de colunas numéricas (INMET retorna strings ou nulos)
    colunas_numericas = {
        "TEM_INS": "temperatura_c",
        "UMD_INS": "umidade_relativa_pct",
        "CHUVA": "precipitacao_mm",
        "RAD_GLO": "radiacao_global_kj_m2",
        "VEN_VEL": "vento_vel_ms",
        "PRE_INS": "pressao_hpa"
    }
    
    for col_origem, col_nova in colunas_numericas.items():
        if col_origem in df.columns:
            df[col_nova] = pd.to_numeric(df[col_origem], errors="coerce")
            
    colunas_finais = ["data_hora_local"] + list(colunas_numericas.values())
    return df[[col for col in colunas_finais if col in df.columns]].sort_values("data_hora_local").reset_index(drop=True)

if __name__ == "__main__":
    try:
        df_inmet = obter_historico_inmet()
    except (RuntimeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)

    print(df_inmet.head())