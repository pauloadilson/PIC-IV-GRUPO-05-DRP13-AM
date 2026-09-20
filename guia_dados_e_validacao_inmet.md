# Guia dos dados e validação dos arquivos anuais do INMET

## Projeto de previsão de umidade da estação A707

Este documento descreve a estrutura dos arquivos CSV horários do INMET utilizados no projeto, registra as diferenças observadas entre os arquivos de 2003 e 2025 e define uma etapa de validação que deve ser executada **antes da concatenação e antes do treinamento dos modelos**.

Arquivos examinados:

- `INMET_SE_SP_A707_PRESIDENTE PRUDENTE_04-02-2003_A_31-12-2003.CSV`
- `INMET_SE_SP_A707_PRESIDENTE PRUDENTE_01-01-2025_A_31-12-2025.CSV`
- script atual: `inmet_modeling.py`

---

# 1. Por que validar antes de modelar

A validação deve preceder o código dos modelos. Se arquivos anuais com colunas equivalentes, mas nomes ou formatos diferentes, forem concatenados diretamente, o pandas poderá criar colunas separadas e preencher grandes trechos com valores ausentes.

Exemplo real observado:

```text
2003: DATA (YYYY-MM-DD)
2025: Data
```

Essas duas colunas têm a mesma função, mas nomes diferentes. Uma concatenação sem padronização produziria duas colunas temporais distintas.

Outro exemplo:

```text
2003: RADIACAO GLOBAL (KJ/m²)
2025: RADIACAO GLOBAL (Kj/m²)
```

A diferença entre `KJ` e `Kj` é suficiente para o pandas tratá-las como colunas diferentes.

A etapa de validação deve verificar, pelo menos:

1. presença das oito linhas de metadados;
2. código WMO da estação;
3. quantidade e nomes das colunas;
4. equivalência semântica depois da normalização dos nomes;
5. formatos de data e hora;
6. conversão das variáveis meteorológicas para números;
7. códigos sentinela, especialmente `-9999`;
8. timestamps inválidos ou duplicados;
9. lacunas na grade horária;
10. percentual de ausências por arquivo e variável;
11. faixas físicas básicas;
12. anos muito incompletos para o alvo.

O pipeline recomendado é:

```text
localizar arquivos
    -> ler metadados
    -> padronizar esquema
    -> validar cada arquivo isoladamente
    -> gerar relatório de qualidade
    -> decidir quais anos podem ser usados
    -> concatenar
    -> reindexar para frequência horária
    -> definir treino, validação e teste
    -> imputar sem vazamento
    -> normalizar com estatísticas do treino
    -> criar janelas
    -> treinar e avaliar modelos
```

---

# 2. Estrutura física dos CSVs

Os arquivos avaliados apresentam:

- codificação compatível com `latin-1`;
- separador `;`;
- vírgula como separador decimal;
- oito linhas iniciais de metadados;
- cabeçalho tabular na nona linha;
- uma coluna vazia adicional no final, causada pelo `;` final de cada linha;
- frequência nominal horária;
- horário em UTC.

Leitura básica:

```python
df = pd.read_csv(
    caminho,
    encoding="latin-1",
    skiprows=8,
    sep=";",
    decimal=",",
    na_values=["-9999", "-9999,0"]
)
```

A coluna final vazia pode ser removida por:

```python
df = df.drop(
    columns=[
        coluna
        for coluna in df.columns
        if str(coluna).startswith("Unnamed:")
    ],
    errors="ignore"
)
```

---

# 3. Metadados da estação

Os oito primeiros registros descrevem a estação.

| Informação | 2003 | 2025 | Observação |
|---|---:|---:|---|
| Região | SE | SE | Mesmo valor |
| UF | SP | SP | Mesmo valor |
| Estação | PRESIDENTE PRUDENTE | PRESIDENTE PRUDENTE | Mesmo valor |
| Código WMO | A707 | A707 | Identifica a mesma estação |
| Latitude | -22,11999999 | -22,11999999 | Mesmo valor |
| Longitude | -51,4 | -51,40861111 | Diferença nos metadados |
| Altitude | 435,55 m | 431,92 m | Diferença nos metadados |
| Fundação | 2003-02-04 | 04/02/03 | Mesma data em formatos diferentes |

Também há diferenças de grafia:

```text
2003: REGIÃO, ESTAÇÃO, DATA DE FUNDAÇÃO
2025: REGIAO, ESTACAO, DATA DE FUNDACAO
```

As diferenças de longitude e altitude não devem ser corrigidas silenciosamente. Elas podem decorrer de revisão cadastral, precisão de coordenadas ou alteração física. O relatório de qualidade deve registrá-las. Para o modelo de uma única estação, esses metadados não precisam entrar como variáveis, pois permanecem constantes dentro de cada período.

---

# 4. Colunas temporais

## 4.1 Data

Foram observadas duas convenções:

```text
2003
coluna: DATA (YYYY-MM-DD)
valor:  2003-02-04

2025
coluna: Data
valor:  2025/01/01
```

A data deve ser convertida com `pd.to_datetime(..., errors="coerce")`.

## 4.2 Hora

Foram observadas duas convenções:

```text
2003
coluna: HORA (UTC)
valor:  00:00

2025
coluna: Hora UTC
valor:  0000 UTC
```

Depois da padronização, ambas devem produzir uma coluna canônica:

```text
datetime
```

com fuso UTC.

A representação temporal canônica recomendada é:

```python
2025-01-01 00:00:00+00:00
```

---

# 5. Dicionário das variáveis meteorológicas

## 5.1 Precipitação

### `PRECIPITAÇÃO TOTAL, HORÁRIO (mm)`

Precipitação total acumulada no período horário, em milímetros.

Considerações:

- valor zero representa ausência de precipitação na hora;
- valores negativos não são fisicamente válidos;
- a distribuição costuma ter muitos zeros e poucos valores altos;
- não se deve substituir automaticamente faltantes por zero, pois ausência de medição não equivale a ausência de chuva.

## 5.2 Pressão atmosférica

### `PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)`

Pressão atmosférica observada no nível da estação, em milibares. Numericamente, 1 mB equivale a 1 hPa.

### `PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)`

Maior pressão registrada na hora anterior.

### `PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)`

Menor pressão registrada na hora anterior.

As três colunas são fortemente relacionadas. Em futura seleção multivariada, deve-se avaliar se todas acrescentam informação útil ou apenas redundância.

## 5.3 Radiação solar

### `RADIACAO GLOBAL (KJ/m²)` ou `RADIACAO GLOBAL (Kj/m²)`

Energia de radiação solar global acumulada por unidade de área.

Considerações:

- a capitalização da unidade muda entre os arquivos;
- o nome precisa ser padronizado antes da concatenação;
- valores noturnos podem ser zero ou ausentes, dependendo do arquivo e do processamento;
- a alta proporção de ausências não deve ser tratada automaticamente por `ffill` sem investigar o padrão dia/noite.

## 5.4 Temperatura do ar

### `TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)`

Temperatura instantânea do ar na observação horária.

### `TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)`

Maior temperatura registrada na hora anterior.

### `TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)`

Menor temperatura registrada na hora anterior.

Essas variáveis têm relação física importante com a umidade relativa.

## 5.5 Ponto de orvalho

### `TEMPERATURA DO PONTO DE ORVALHO (°C)`

Temperatura na qual o ar atingiria saturação, mantida aproximadamente a quantidade de vapor d'água.

### `TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)`

Maior temperatura do ponto de orvalho registrada na hora anterior.

### `TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)`

Menor temperatura do ponto de orvalho registrada na hora anterior.

O ponto de orvalho é uma variável candidata relevante para previsão de umidade, mas pode ter alta associação com o próprio alvo.

## 5.6 Umidade relativa

### `UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)`

Maior umidade relativa registrada durante a hora anterior. Esta é a variável-alvo usada na etapa atual do projeto.

### `UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)`

Menor umidade relativa registrada durante a hora anterior.

### `UMIDADE RELATIVA DO AR, HORARIA (%)`

Umidade relativa observada no horário de referência.

A faixa física geral é de 0% a 100%. Valores fora desse intervalo devem ser sinalizados.

## 5.7 Vento

### `VENTO, DIREÇÃO HORARIA (gr) (° (gr))`

Direção do vento em graus, normalmente na faixa de 1° a 360°.

É uma variável circular. Em futura modelagem multivariada, a codificação recomendada é:

```python
vento_direcao_sin = np.sin(2 * np.pi * direcao / 360)
vento_direcao_cos = np.cos(2 * np.pi * direcao / 360)
```

### `VENTO, RAJADA MAXIMA (m/s)`

Maior velocidade de rajada registrada no período, em metros por segundo.

### `VENTO, VELOCIDADE HORARIA (m/s)`

Velocidade horária do vento, em metros por segundo.

Velocidades negativas devem ser sinalizadas como inválidas.

---

# 6. Diferenças de esquema observadas

Os arquivos têm 19 colunas meteorológicas e temporais após a remoção da coluna vazia final. Entretanto, os nomes não são exatamente iguais.

Diferenças confirmadas:

```text
2003: DATA (YYYY-MM-DD)
2025: Data
```

```text
2003: HORA (UTC)
2025: Hora UTC
```

```text
2003: RADIACAO GLOBAL (KJ/m²)
2025: RADIACAO GLOBAL (Kj/m²)
```

Por isso, a validação não deve exigir igualdade literal antes da normalização. O procedimento correto é:

1. guardar os nomes originais para auditoria;
2. mapear variações conhecidas para nomes canônicos;
3. verificar se o conjunto canônico é consistente;
4. interromper o processamento se aparecer uma coluna desconhecida ou faltar uma coluna obrigatória.

---

# 7. Qualidade observada nos dois arquivos

## 7.1 Arquivo de 2003

Período:

```text
04/02/2003 00:00 UTC até 31/12/2003 23:00 UTC
```

Resumo:

- 7.944 timestamps horários;
- nenhum timestamp inválido;
- nenhuma duplicidade temporal;
- nenhuma hora ausente na grade temporal do período coberto;
- somente 2.092 valores válidos para as principais variáveis;
- 5.852 valores ausentes na variável-alvo;
- aproximadamente 73,67% de ausência na variável-alvo;
- radiação com aproximadamente 85,75% de ausência;
- a variável-alvo válida variou de 26% a 95%.

Esse ano é temporalmente contínuo, mas meteorologicamente muito incompleto. Preencher automaticamente um bloco tão grande com `ffill` criaria uma sequência artificial e não é recomendado.

## 7.2 Arquivo de 2025

Período:

```text
01/01/2025 00:00 UTC até 31/12/2025 23:00 UTC
```

Resumo:

- 8.760 timestamps horários;
- nenhum timestamp inválido;
- nenhuma duplicidade temporal;
- nenhuma hora ausente na grade anual;
- 45 valores ausentes na variável-alvo;
- aproximadamente 0,51% de ausência na variável-alvo;
- radiação com aproximadamente 46,66% de ausência;
- a variável-alvo válida variou de 10% a 99%.

O arquivo de 2025 é muito mais completo para a variável-alvo.

---

# 8. Códigos de ausência

Foram observadas duas convenções:

```text
2003: -9999
2025: campo vazio
```

Na leitura, ambas devem ser convertidas para `NaN`:

```python
na_values = [
    "-9999",
    "-9999,0",
    -9999,
    -9999.0
]
```

A conversão deve ocorrer antes da validação das faixas físicas e antes da imputação.

---

# 9. Critérios sugeridos para aceitar um ano no modelo de umidade

Os limites abaixo são regras iniciais para triagem e devem ser registrados como decisões metodológicas.

## 9.1 Rejeição estrutural

Interromper a carga do arquivo se houver:

- código WMO diferente de `A707`;
- ausência das colunas temporais;
- ausência da variável-alvo;
- coluna canônica duplicada depois da padronização;
- mais ou menos colunas desconhecidas sem mapeamento explícito;
- timestamps majoritariamente inválidos.

## 9.2 Alerta de qualidade

Emitir alerta se houver:

- timestamps duplicados;
- lacunas na grade horária;
- valores de umidade fora de 0% a 100%;
- mais de 5% de ausência no alvo;
- sequências consecutivas longas de ausência;
- metadados divergentes para latitude, longitude ou altitude.

## 9.3 Exclusão recomendada do treinamento

Como regra inicial, não usar automaticamente um ano se a variável-alvo tiver mais de 20% de ausência ou se houver blocos contínuos muito longos sem observação.

Pelo critério de 20%, o arquivo de 2003 seria marcado para revisão ou exclusão, pois apresenta aproximadamente 73,67% de ausência no alvo.

Não é recomendável preencher 73,67% do ano com o último valor observado.

---

# 10. Padronização canônica das colunas

Nomes canônicos sugeridos:

```python
COLUNAS_CANONICAS = {
    "data": "data",
    "hora_utc": "hora_utc",
    "precipitacao_total_horario_mm": "precipitacao_total_horario_mm",
    "pressao_estacao_horaria_mb": "pressao_estacao_horaria_mb",
    "pressao_max_hora_anterior_mb": "pressao_max_hora_anterior_mb",
    "pressao_min_hora_anterior_mb": "pressao_min_hora_anterior_mb",
    "radiacao_global_kj_m2": "radiacao_global_kj_m2",
    "temperatura_ar_horaria_c": "temperatura_ar_horaria_c",
    "temperatura_ponto_orvalho_c": "temperatura_ponto_orvalho_c",
    "temperatura_max_hora_anterior_c": "temperatura_max_hora_anterior_c",
    "temperatura_min_hora_anterior_c": "temperatura_min_hora_anterior_c",
    "temperatura_orvalho_max_hora_anterior_c": "temperatura_orvalho_max_hora_anterior_c",
    "temperatura_orvalho_min_hora_anterior_c": "temperatura_orvalho_min_hora_anterior_c",
    "umidade_max_hora_anterior_pct": "umidade_max_hora_anterior_pct",
    "umidade_min_hora_anterior_pct": "umidade_min_hora_anterior_pct",
    "umidade_horaria_pct": "umidade_horaria_pct",
    "vento_direcao_graus": "vento_direcao_graus",
    "vento_rajada_max_ms": "vento_rajada_max_ms",
    "vento_velocidade_horaria_ms": "vento_velocidade_horaria_ms"
}
```

Para o modelo univariado atualizado, a variável-alvo canônica será:

```python
COLUNA_ALVO = "umidade_max_hora_anterior_pct"
```

---

# 11. Uso do módulo de validação fornecido

Foi criado, junto com este documento, o arquivo:

```text
validacao_carga_inmet.py
```

O módulo:

- localiza todos os CSVs anuais do INMET em uma pasta;
- lê os metadados;
- normaliza os nomes das colunas;
- converte `-9999` e campos vazios para `NaN`;
- cria a coluna `datetime` em UTC;
- compara o esquema canônico;
- calcula ausências e cobertura;
- identifica duplicidades e lacunas;
- valida faixas físicas básicas;
- gera relatórios CSV;
- concatena apenas os arquivos que passam pela validação estrutural.

Execução:

```bash
python validacao_carga_inmet.py
```

Por padrão, o programa procura:

```text
INMET*.CSV
```

na pasta atual.

Arquivos de saída:

```text
relatorio_arquivos_inmet.csv
relatorio_colunas_inmet.csv
dados_inmet_consolidados.csv
```

---

# 12. Como integrar com `inmet_modeling.py`

Substitua a leitura de um único arquivo:

```python
arquivo = "...2025.CSV"
df = pd.read_csv(...)
```

por:

```python
from validacao_carga_inmet import carregar_e_validar_diretorio

resultado_carga = carregar_e_validar_diretorio(
    pasta=".",
    padrao="INMET*.CSV",
    wmo_esperado="A707",
    limite_ausencia_alvo=0.20,
    excluir_anos_incompletos=True
)

df = resultado_carga["dados"]
relatorio_arquivos = resultado_carga["relatorio_arquivos"]
relatorio_colunas = resultado_carga["relatorio_colunas"]

COLUNA_ALVO = "umidade_max_hora_anterior_pct"
```

Antes de prosseguir:

```python
print(relatorio_arquivos.to_string(index=False))

assert not df.empty
assert df["datetime"].is_monotonic_increasing
assert not df["datetime"].duplicated().any()
assert COLUNA_ALVO in df.columns
```

---

# 13. Divisão temporal recomendada para 2003 a 2026

Após aprovação do relatório de qualidade, a divisão recomendada é:

```text
Treinamento:  arquivos aprovados até 31/12/2022
Validação:    01/01/2023 a 31/12/2024
Teste final:  01/01/2025 a 31/12/2025
Teste recente:2026 parcial
```

Implementação:

```python
data_inicio_validacao = pd.Timestamp(
    "2023-01-01",
    tz="UTC"
)

data_inicio_teste = pd.Timestamp(
    "2025-01-01",
    tz="UTC"
)

data_fim_teste = pd.Timestamp(
    "2026-01-01",
    tz="UTC"
)

indice_fim_treino = int(
    (df["datetime"] < data_inicio_validacao).sum()
)

indice_fim_validacao = int(
    (df["datetime"] < data_inicio_teste).sum()
)

indice_fim_teste = int(
    (df["datetime"] < data_fim_teste).sum()
)
```

IMPORTANTE: se anos inteiros forem excluídos por baixa qualidade, a série consolidada terá grandes saltos temporais. Não crie janelas atravessando esses saltos. A criação das janelas deverá verificar se todos os timestamps de cada janela são consecutivos e separados por uma hora.

---

# 14. Imputação para a série multiano

O `ffill()` ilimitado usado no script de um único ano não é seguro para a série multiano. Um bloco extenso ausente poderia repetir o mesmo valor por meses.

Use limite curto, por exemplo:

```python
serie_alvo = df[COLUNA_ALVO].ffill(limit=3)
```

Depois, não crie uma janela caso a entrada ou o alvo ainda contenha `NaN`.

Isso é preferível a preencher blocos longos artificialmente.

A função de janelas deve:

1. exigir timestamps horários consecutivos;
2. exigir todos os 24 valores de entrada válidos;
3. exigir alvo válido;
4. registrar quantas janelas foram descartadas.

---

# 15. Conclusão

A validação deve ser implementada antes do treinamento. Os dois arquivos examinados provam que mudanças reais existem entre anos, tanto nos nomes das colunas quanto nos formatos de data, hora, metadados e códigos de ausência.

O arquivo de 2003 possui grade temporal completa, mas aproximadamente 73,67% de ausência na variável-alvo. Portanto, não deve ser incorporado automaticamente ao treinamento por meio de preenchimento ilimitado. O arquivo de 2025 possui apenas aproximadamente 0,51% de ausência no alvo e é substancialmente mais adequado.

A próxima etapa recomendada é executar `validacao_carga_inmet.py` sobre todos os arquivos de 2003 a 2026, examinar `relatorio_arquivos_inmet.csv` e somente então definir quais anos formarão o treinamento, a validação e o teste.
