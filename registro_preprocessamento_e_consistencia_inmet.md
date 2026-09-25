# Registro de pré-processamento e consistência dos dados INMET

## 1. Finalidade

Este documento registra as decisões de pré-processamento adotadas na carga dos arquivos horários da estação INMET A707, a partir de 2004, e estabelece critérios para verificar todas as variáveis meteorológicas antes da escolha da variável-alvo.

A validação deixa de estar vinculada exclusivamente à umidade. O conjunto consolidado deverá permitir que uma etapa posterior utilize como alvo, por exemplo:

- umidade máxima horária;
- umidade horária;
- temperatura do ar;
- temperatura máxima ou mínima;
- ponto de orvalho;
- pressão atmosférica;
- precipitação;
- velocidade do vento.

A escolha do alvo passa a ser uma decisão da etapa de modelagem, posterior à validação estrutural e de qualidade.

---

## 2. Correção de números decimais sem zero inicial

### 2.1 Problema identificado

Alguns arquivos apresentam campos numéricos no formato:

```text
,3
,8
,2
```

Como os arquivos utilizam vírgula decimal, esses valores representam:

```text
0,3
0,8
0,2
```

Sem a correção, `pandas.to_numeric(..., errors="coerce")` pode converter esses campos em valores ausentes, reduzindo artificialmente a completude da coluna.

### 2.2 Regra adotada

Durante a etapa de pré-processamento, todo campo que:

1. começa imediatamente depois do separador `;`;
2. inicia com vírgula;
3. possui um dígito depois da vírgula;

é convertido mediante inclusão de zero à esquerda.

Exemplo:

```text
Antes:  ...;163;1,5;,3;
Depois: ...;163;1,5;0,3;
```

A expressão regular utilizada é:

```python
r"(?<=;),(?=\d)"
```

Substituição:

```python
"0,"
```

A regra não modifica:

- valores já completos, como `1,5`;
- números inteiros, como `163`;
- campos vazios, como `;;`;
- valores ausentes codificados como `-9999`;
- textos do cabeçalho.

### 2.3 Rastreabilidade

O arquivo bruto original não é alterado. O pipeline:

1. lê o arquivo original;
2. aplica a correção em memória;
3. salva uma cópia corrigida na subpasta `preprocessados`;
4. registra a quantidade de valores corrigidos;
5. registra a quantidade de linhas afetadas;
6. registra exemplos dos números das linhas corrigidas;
7. utiliza o conteúdo corrigido na validação e consolidação.

O relatório gerado é:

```text
registro_preprocessamento_inmet_YYYY-MM-DD.csv
```

Campos principais:

- `arquivo`;
- `etapa`;
- `regra`;
- `regex`;
- `n_valores_corrigidos`;
- `n_linhas_afetadas`;
- `primeiras_linhas_afetadas`;
- `arquivo_original_alterado`;
- `copia_preprocessada`.

### 2.4 Verificação realizada em 2019

No arquivo corrigido de 2019, a regra encontrou:

```text
2.785 valores corrigidos
2.687 linhas afetadas
```

A primeira observação do ano, cuja velocidade horária do vento aparecia como `,3`, passou a ser interpretada corretamente como `0,3 m/s`.

---

## 3. Separação entre validação estrutural e elegibilidade para um alvo

A versão anterior classificava o arquivo como aprovado ou reprovado usando a ausência da variável de umidade máxima. Esse critério é adequado apenas quando a umidade máxima é obrigatoriamente o alvo.

Como o projeto deverá permitir diferentes alvos, foram separados dois conceitos.

### 3.1 Aprovação estrutural

Um arquivo é estruturalmente aprovado quando:

- pertence à estação WMO esperada;
- contém todas as colunas canônicas esperadas;
- não contém colunas desconhecidas;
- não produz nomes canônicos duplicados;
- possui timestamps válidos;
- não apresenta valores fora das faixas físicas básicas definidas.

A aprovação estrutural permite que o arquivo integre o conjunto consolidado, ainda que determinada variável esteja vazia naquele ano.

### 3.2 Elegibilidade para um alvo específico

A elegibilidade para modelagem deve ser calculada depois que a variável-alvo for escolhida.

Exemplo:

```python
coluna_alvo_criterio="temperatura_ar_horaria_c"
```

Nesse caso, o limite de ausência será avaliado sobre temperatura, e não sobre umidade.

Se `coluna_alvo_criterio=None`, a carga realiza somente a validação estrutural e não exclui arquivos por ausência em um alvo previamente escolhido.

---

## 4. Validação de todas as colunas

Para cada variável meteorológica de cada arquivo anual, o relatório registra:

- quantidade total de registros;
- quantidade de valores válidos;
- quantidade de valores ausentes;
- percentual de ausência;
- valor mínimo;
- valor máximo;
- quantidade de violações das faixas básicas;
- maior bloco consecutivo de ausências;
- indicador de coluna totalmente vazia naquele arquivo.

O relatório detalhado é:

```text
relatorio_colunas_inmet_YYYY-MM-DD.csv
```

Essa análise permite escolher qualquer variável-alvo posteriormente e verificar quais anos oferecem dados suficientes para essa variável.

---

## 5. Consistência global das colunas

Além do relatório por arquivo, foi incluído um resumo global:

```text
relatorio_global_colunas_inmet_YYYY-MM-DD.csv
```

Para cada coluna, o resumo informa:

- número de arquivos avaliados;
- número de arquivos com pelo menos um valor válido;
- número de arquivos em que a coluna está totalmente vazia;
- número total de registros;
- número total de valores válidos;
- número total de ausências;
- percentual global de ausência;
- mínimo e máximo globais;
- total de violações das faixas físicas;
- sugestão de manutenção, revisão ou exclusão.

---

## 6. Critério de exclusão de colunas vazias

### 6.1 Coluna vazia em todos os arquivos

Se uma coluna não possuir nenhum valor válido em todo o período carregado, a sugestão será:

```text
EXCLUIR: coluna totalmente vazia em todos os arquivos carregados
```

Essa coluna não fornece informação ao modelo e deve ser removida antes da normalização e da criação das janelas.

Código para exclusão:

```python
colunas_vazias = resultado_carga["colunas_vazias_globais"]

df = df.drop(
    columns=colunas_vazias,
    errors="ignore"
)
```

### 6.2 Coluna vazia somente em alguns anos

Uma coluna vazia em determinados anos não deverá ser excluída automaticamente de todo o projeto.

Exemplo observado nos relatórios anteriores:

- as colunas de pressão estavam vazias em vários anos antigos;
- passaram a possuir dados em anos posteriores.

Nesse cenário, a sugestão é:

```text
MANTER COM RESSALVA: vazia em alguns anos; avaliar período do modelo
```

As opções metodológicas são:

1. usar a variável apenas em modelos treinados no período em que está disponível;
2. excluir a variável do conjunto multivariado para preservar uma série histórica mais longa;
3. comparar um modelo longo com menos variáveis e um modelo recente com mais variáveis;
4. nunca preencher anos inteiros artificialmente apenas para conservar a coluna.

### 6.3 Coluna com ausência global muito elevada

Uma coluna com pelo menos 80% de ausência global recebe a sugestão:

```text
REVISAR: ausência global muito elevada
```

A coluna não é excluída automaticamente, pois pode conter períodos contínuos úteis ou representar uma variável importante. A decisão final deve considerar:

- distribuição temporal das ausências;
- maior bloco ausente;
- período de treino desejado;
- relevância física;
- disponibilidade futura durante a previsão.

---

## 7. Uso recomendado no script de modelagem

Carga sem vincular a aprovação a um único alvo:

```python
from validacao_carga_inmet_v2 import (
    carregar_e_validar_diretorio
)

resultado_carga = carregar_e_validar_diretorio(
    pasta="./estacao_A707",
    padrao="INMET*.CSV",
    wmo_esperado="A707",
    coluna_alvo_criterio=None,
    excluir_arquivos_com_erro_estrutural=True,
    salvar_copias_preprocessadas=True
)

df = resultado_carga["dados"]
relatorio_arquivos = resultado_carga["relatorio_arquivos"]
relatorio_colunas = resultado_carga["relatorio_colunas"]
relatorio_global = resultado_carga["relatorio_global_colunas"]
registro_preprocessamento = resultado_carga[
    "relatorio_preprocessamento"
]
```

Exclusão somente das colunas globalmente vazias:

```python
colunas_vazias = resultado_carga[
    "colunas_vazias_globais"
]

df = df.drop(
    columns=colunas_vazias,
    errors="ignore"
)
```

Escolha posterior do alvo:

```python
COLUNA_ALVO = "temperatura_ar_horaria_c"
```

ou:

```python
COLUNA_ALVO = "umidade_max_hora_anterior_pct"
```

Antes do treino, verificar a disponibilidade do alvo por ano e período:

```python
qualidade_alvo = (
    relatorio_colunas
    .loc[
        relatorio_colunas["coluna"] == COLUNA_ALVO
    ]
    .sort_values("arquivo")
)

print(
    qualidade_alvo[
        [
            "arquivo",
            "n_validos",
            "percentual_ausente",
            "maior_bloco_ausente"
        ]
    ].to_string(index=False)
)
```

---

## 8. Texto sugerido para o relatório acadêmico

> Durante o pré-processamento, foram identificados campos numéricos decimais sem zero inicial, representados, por exemplo, como `,3` em vez de `0,3`. Esses campos foram normalizados por uma regra determinística aplicada somente quando a vírgula ocorria no início de um campo numérico. Os arquivos brutos foram preservados, e cópias pré-processadas foram gravadas separadamente. A quantidade de valores e linhas alterados foi registrada por arquivo para assegurar rastreabilidade.

> A validação de qualidade foi aplicada a todas as variáveis meteorológicas, independentemente da variável-alvo inicialmente utilizada. Para cada arquivo e coluna, foram calculados completude, faixa de valores, violações físicas e maior sequência de ausências. Colunas totalmente vazias em todo o período consolidado foram indicadas para exclusão. Colunas vazias apenas em determinados anos foram mantidas com ressalva, pois poderiam ser úteis em modelos restritos a períodos mais recentes.

> A aprovação estrutural dos arquivos foi separada da elegibilidade de cada variável como alvo. Essa separação permitiu preservar arquivos úteis para diferentes problemas de previsão, sem condicionar toda a base à disponibilidade de uma única variável, como umidade ou temperatura.

---

## 9. Decisões metodológicas registradas

1. Os arquivos brutos não serão sobrescritos.
2. Valores no formato `,numero` serão convertidos em `0,numero`.
3. A correção será aplicada somente no início de campos numéricos.
4. Cada alteração será documentada em relatório de pré-processamento.
5. Todas as variáveis meteorológicas serão validadas.
6. A aprovação estrutural não dependerá de um alvo fixo.
7. Colunas globalmente vazias serão sugeridas para exclusão.
8. Colunas vazias somente em alguns anos não serão excluídas automaticamente.
9. A escolha da variável-alvo ocorrerá depois da análise de qualidade.
10. Anos e janelas com ausências relevantes para o alvo escolhido poderão ser descartados na etapa de criação das amostras, sem imputação de blocos extensos.
