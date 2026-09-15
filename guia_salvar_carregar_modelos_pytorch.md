# Guia para salvar, carregar e reutilizar modelos PyTorch

Este guia mostra como salvar os modelos **Linear** e **GRU** treinados com diferentes sementes, liberar a memória da GPU e carregá-los posteriormente para previsões e avaliações sem repetir o treinamento.

## 1. O que será salvo

Cada arquivo de checkpoint armazenará:

- tipo do modelo: `Linear` ou `GRU`;
- pesos treinados, por meio do `state_dict`;
- semente utilizada;
- `tau`, isto é, o tamanho da janela temporal;
- média e desvio-padrão calculados no conjunto de treinamento;
- hiperparâmetros da arquitetura;
- histórico das perdas de treinamento e validação;
- data e hora da execução.

Salvar a média, o desvio-padrão e o `tau` é essencial porque as previsões futuras precisam usar exatamente o mesmo pré-processamento empregado no treinamento.

---

## 2. Imports necessários

```python
from pathlib import Path
from datetime import datetime
import gc
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
```

---

## 3. Definição das arquiteturas

A classe da GRU e a função que cria o modelo linear precisam estar disponíveis antes de carregar os checkpoints.

```python
class GRURegressor(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Entrada original: (batch_size, tau)
        # Entrada da GRU: (batch_size, tau, 1)
        x = x.unsqueeze(-1)
        output, _ = self.gru(x)
        return self.output(output[:, -1, :])


def criar_modelo_linear(tau):
    return nn.Linear(tau, 1)


def criar_modelo_gru(
    hidden_size=64,
    num_layers=2,
    dropout=0.2
):
    return GRURegressor(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
```

---

## 4. Criar pasta e identificação temporal da execução

Gere uma única identificação temporal para toda a rodada de treinamento. Assim, os modelos das diferentes sementes permanecem associados à mesma execução experimental.

```python
PASTA_MODELOS = Path("modelos_salvos")
PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

ID_EXECUCAO = datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)

print("Identificação da execução:", ID_EXECUCAO)
```

Exemplo de identificação:

```text
2026-09-14_15-20-35
```

---

## 5. Função para salvar um checkpoint

```python
def salvar_checkpoint(
    modelo,
    nome_modelo,
    semente,
    tau,
    media_treino,
    desvio_treino,
    id_execucao,
    pasta_modelos,
    hiperparametros=None,
    historico=None,
    metadados_adicionais=None
):
    pasta_modelos = Path(pasta_modelos)
    pasta_modelos.mkdir(parents=True, exist_ok=True)

    nome_arquivo = (
        f"{nome_modelo.lower()}_"
        f"semente-{int(semente)}_"
        f"{id_execucao}.pth"
    )

    caminho = pasta_modelos / nome_arquivo

    checkpoint = {
        "versao_checkpoint": 1,
        "nome_modelo": str(nome_modelo),
        "semente": int(semente),
        "tau": int(tau),
        "media_treino": float(media_treino),
        "desvio_treino": float(desvio_treino),
        "id_execucao": str(id_execucao),
        "hiperparametros": hiperparametros or {},
        "historico": historico,
        "metadados_adicionais": metadados_adicionais or {},
        "model_state_dict": {
            nome: tensor.detach().cpu()
            for nome, tensor in modelo.state_dict().items()
        }
    }

    torch.save(checkpoint, caminho)

    print(f"Checkpoint salvo: {caminho}")
    return caminho
```

Os tensores são movidos para a CPU antes da gravação. Isso facilita o carregamento posterior em qualquer computador, mesmo sem GPU CUDA.

---

## 6. Definir os hiperparâmetros usados

```python
hiperparametros_linear = {
    "arquitetura": "nn.Linear",
    "entrada": tau,
    "saida": 1,
    "lr": 0.001,
    "max_epocas": 200,
    "paciencia": 20,
    "batch_size": 32
}

hiperparametros_gru = {
    "arquitetura": "GRURegressor",
    "input_size": 1,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "saida": 1,
    "lr": 0.001,
    "max_epocas": 200,
    "paciencia": 20,
    "batch_size": 32
}
```

Se desejar registrar os limites temporais do experimento:

```python
metadados_experimento = {
    "fim_treino": int(fim_treino),
    "fim_validacao": int(fim_validacao),
    "tamanho_total": int(len(serie)),
    "coluna_alvo": COLUNA_ALVO,
    "proporcao_treino": 0.70,
    "proporcao_validacao": 0.15,
    "proporcao_teste": 0.15
}
```

---

## 7. Salvar todos os modelos treinados

Este bloco pressupõe a existência das estruturas:

```python
modelos_treinados = {
    "Linear": [modelo_semente_10, ..., modelo_semente_50],
    "GRU": [modelo_semente_10, ..., modelo_semente_50]
}

historicos = {
    "Linear": [historico_10, ..., historico_50],
    "GRU": [historico_10, ..., historico_50]
}

SEMENTES = [10, 20, 30, 40, 50]
```

Salvamento:

```python
caminhos_modelos = {
    "Linear": [],
    "GRU": []
}

for nome_modelo in ["Linear", "GRU"]:
    for indice, modelo in enumerate(
        modelos_treinados[nome_modelo]
    ):
        semente = SEMENTES[indice]

        if nome_modelo == "Linear":
            hiperparametros = hiperparametros_linear
        else:
            hiperparametros = hiperparametros_gru

        caminho = salvar_checkpoint(
            modelo=modelo,
            nome_modelo=nome_modelo,
            semente=semente,
            tau=tau,
            media_treino=media_treino,
            desvio_treino=desvio_treino,
            id_execucao=ID_EXECUCAO,
            pasta_modelos=PASTA_MODELOS,
            hiperparametros=hiperparametros,
            historico=historicos[nome_modelo][indice],
            metadados_adicionais=metadados_experimento
        )

        caminhos_modelos[nome_modelo].append(caminho)
```

Os arquivos terão nomes semelhantes a:

```text
modelos_salvos/
├── linear_semente-10_2026-09-14_15-20-35.pth
├── linear_semente-20_2026-09-14_15-20-35.pth
├── linear_semente-30_2026-09-14_15-20-35.pth
├── linear_semente-40_2026-09-14_15-20-35.pth
├── linear_semente-50_2026-09-14_15-20-35.pth
├── gru_semente-10_2026-09-14_15-20-35.pth
├── gru_semente-20_2026-09-14_15-20-35.pth
├── gru_semente-30_2026-09-14_15-20-35.pth
├── gru_semente-40_2026-09-14_15-20-35.pth
└── gru_semente-50_2026-09-14_15-20-35.pth
```

---

## 8. Verificar se os arquivos foram gravados

```python
for nome_modelo, caminhos in caminhos_modelos.items():
    print(f"\n{nome_modelo}:")

    for caminho in caminhos:
        assert caminho.exists(), (
            f"Arquivo não encontrado: {caminho}"
        )

        tamanho_mb = caminho.stat().st_size / 1024**2

        print(
            f"{caminho.name} | "
            f"{tamanho_mb:.3f} MB"
        )
```

Não libere os modelos da memória antes de confirmar que todos os arquivos foram criados.

---

## 9. Liberar a memória da GPU

```python
# Move os modelos para a CPU antes de apagar as referências.
for nome_modelo in modelos_treinados:
    for modelo in modelos_treinados[nome_modelo]:
        modelo.to("cpu")

# Remove as referências mantidas pelo dicionário.
modelos_treinados.clear()
del modelos_treinados

# Remove possíveis referências temporárias.
for nome_variavel in [
    "modelo",
    "modelo_linear",
    "modelo_gru"
]:
    if nome_variavel in globals():
        del globals()[nome_variavel]

# Executa a coleta de lixo e libera o cache CUDA desocupado.
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

    print(
        "Memória CUDA alocada: "
        f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )

    print(
        "Memória CUDA reservada: "
        f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
    )
```

`torch.cuda.empty_cache()` não remove objetos que ainda possuem referências no Python. Por isso, a ordem correta é:

1. mover os modelos para a CPU;
2. apagar referências;
3. executar `gc.collect()`;
4. executar `torch.cuda.empty_cache()`.

---

## 10. Função para carregar um checkpoint

```python
def carregar_checkpoint(
    caminho,
    dispositivo=None
):
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {caminho}"
        )

    if dispositivo is None:
        dispositivo = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        dispositivo = torch.device(dispositivo)

    # map_location="cpu" evita ocupar a GPU durante a leitura.
    checkpoint = torch.load(
        caminho,
        map_location="cpu",
        weights_only=False
    )

    nome_modelo = checkpoint["nome_modelo"]
    tau_checkpoint = int(checkpoint["tau"])
    hiperparametros = checkpoint.get(
        "hiperparametros",
        {}
    )

    if nome_modelo == "Linear":
        modelo = criar_modelo_linear(
            tau=tau_checkpoint
        )

    elif nome_modelo == "GRU":
        modelo = criar_modelo_gru(
            hidden_size=hiperparametros.get(
                "hidden_size",
                64
            ),
            num_layers=hiperparametros.get(
                "num_layers",
                2
            ),
            dropout=hiperparametros.get(
                "dropout",
                0.2
            )
        )

    else:
        raise ValueError(
            f"Tipo de modelo desconhecido: {nome_modelo}"
        )

    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )

    modelo = modelo.to(dispositivo)
    modelo.eval()

    metadados = {
        chave: valor
        for chave, valor in checkpoint.items()
        if chave != "model_state_dict"
    }

    print(
        f"Modelo carregado: {nome_modelo} | "
        f"semente: {checkpoint['semente']} | "
        f"execução: {checkpoint['id_execucao']} | "
        f"dispositivo: {dispositivo}"
    )

    return modelo, metadados
```

### Observação sobre `weights_only`

Este guia usa `weights_only=False` porque o checkpoint contém, além dos pesos, números, dicionários e históricos. Carregue somente checkpoints criados por você ou provenientes de fonte confiável.

---

## 11. Carregar um modelo diretamente na CPU

Carregar na CPU evita ocupar a memória da GPU.

```python
caminho_linear = (
    PASTA_MODELOS
    / "linear_semente-10_2026-09-14_15-20-35.pth"
)

modelo_linear, meta_linear = carregar_checkpoint(
    caminho=caminho_linear,
    dispositivo="cpu"
)
```

Para a GRU:

```python
caminho_gru = (
    PASTA_MODELOS
    / "gru_semente-10_2026-09-14_15-20-35.pth"
)

modelo_gru, meta_gru = carregar_checkpoint(
    caminho=caminho_gru,
    dispositivo="cpu"
)
```

Verificação:

```python
print("Linear em modo de treino?", modelo_linear.training)
print("GRU em modo de treino?", modelo_gru.training)
```

O resultado esperado é `False`, pois os modelos foram colocados em modo de avaliação com `eval()`.

---

## 12. Localizar automaticamente o checkpoint mais recente

```python
def localizar_checkpoint_mais_recente(
    pasta,
    nome_modelo,
    semente=None
):
    pasta = Path(pasta)

    if semente is None:
        padrao = f"{nome_modelo.lower()}_semente-*_*.pth"
    else:
        padrao = (
            f"{nome_modelo.lower()}_"
            f"semente-{int(semente)}_*.pth"
        )

    arquivos = list(pasta.glob(padrao))

    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum checkpoint corresponde a: {padrao}"
        )

    return max(
        arquivos,
        key=lambda arquivo: arquivo.stat().st_mtime
    )
```

Uso:

```python
caminho_linear = localizar_checkpoint_mais_recente(
    pasta=PASTA_MODELOS,
    nome_modelo="Linear",
    semente=10
)

caminho_gru = localizar_checkpoint_mais_recente(
    pasta=PASTA_MODELOS,
    nome_modelo="GRU",
    semente=10
)

modelo_linear, meta_linear = carregar_checkpoint(
    caminho_linear,
    dispositivo="cpu"
)

modelo_gru, meta_gru = carregar_checkpoint(
    caminho_gru,
    dispositivo="cpu"
)
```

---

## 13. Preparar a série com os metadados salvos

Use sempre a média, o desvio-padrão e o `tau` do checkpoint.

```python
tau_linear = int(meta_linear["tau"])
media_linear = float(meta_linear["media_treino"])
desvio_linear = float(meta_linear["desvio_treino"])

serie_normalizada_linear = (
    (serie - media_linear) / desvio_linear
).astype(np.float32)
```

Para a GRU:

```python
tau_gru = int(meta_gru["tau"])
media_gru = float(meta_gru["media_treino"])
desvio_gru = float(meta_gru["desvio_treino"])

serie_normalizada_gru = (
    (serie - media_gru) / desvio_gru
).astype(np.float32)
```

Se os dois modelos vierem da mesma execução, confira a compatibilidade:

```python
assert tau_linear == tau_gru, (
    "Os modelos foram treinados com valores de tau diferentes."
)

assert np.isclose(media_linear, media_gru), (
    "As médias de treinamento não coincidem."
)

assert np.isclose(desvio_linear, desvio_gru), (
    "Os desvios-padrão de treinamento não coincidem."
)
```

---

## 14. Função de previsão recursiva

```python
def obter_dispositivo(modelo):
    return next(modelo.parameters()).device


def previsao_recursiva(
    modelo,
    historico_normalizado,
    tau,
    horizonte
):
    modelo.eval()
    dispositivo = obter_dispositivo(modelo)

    if len(historico_normalizado) < tau:
        raise ValueError(
            "O histórico possui menos observações que tau."
        )

    janela = torch.tensor(
        historico_normalizado[-tau:],
        dtype=torch.float32,
        device=dispositivo
    )

    previsoes = []

    with torch.inference_mode():
        for _ in range(horizonte):
            entrada = janela.reshape(1, tau)
            proximo = modelo(entrada).squeeze()

            previsoes.append(float(proximo.item()))

            janela = torch.cat([
                janela[1:],
                proximo.reshape(1)
            ])

    return np.asarray(
        previsoes,
        dtype=np.float32
    )
```

---

## 15. Função do baseline sazonal de 24 horas

```python
def previsao_sazonal(
    historico_original,
    horizonte,
    periodo=24
):
    historico_original = np.asarray(
        historico_original,
        dtype=np.float32
    )

    if len(historico_original) < periodo:
        raise ValueError(
            "Histórico insuficiente para o período sazonal."
        )

    padrao = historico_original[-periodo:]

    repeticoes = int(
        np.ceil(horizonte / periodo)
    )

    return np.tile(
        padrao,
        repeticoes
    )[:horizonte]
```

---

## 16. Fazer uma previsão de 72 horas com modelos carregados

Escolha uma origem. Para prever as primeiras 72 horas do teste final:

```python
origem = fim_validacao
horizonte = 72
```

### Modelo Linear

```python
historico_linear = serie_normalizada_linear[:origem]

pred_linear_norm = previsao_recursiva(
    modelo=modelo_linear,
    historico_normalizado=historico_linear,
    tau=tau_linear,
    horizonte=horizonte
)

pred_linear_origem = (
    pred_linear_norm * desvio_linear
    + media_linear
)
```

### GRU

```python
historico_gru = serie_normalizada_gru[:origem]

pred_gru_norm = previsao_recursiva(
    modelo=modelo_gru,
    historico_normalizado=historico_gru,
    tau=tau_gru,
    horizonte=horizonte
)

pred_gru_origem = (
    pred_gru_norm * desvio_gru
    + media_gru
)
```

### Baseline e valores reais

```python
pred_baseline_origem = previsao_sazonal(
    historico_original=serie[:origem],
    horizonte=horizonte,
    periodo=24
)

real_origem = serie[
    origem:origem + horizonte
]

datas_origem = datas[
    origem:origem + horizonte
]
```

### Verificação

```python
assert len(datas_origem) == horizonte
assert len(real_origem) == horizonte
assert len(pred_linear_origem) == horizonte
assert len(pred_gru_origem) == horizonte
assert len(pred_baseline_origem) == horizonte
```

As variáveis estão prontas para o gráfico comparativo:

```python
datas_origem
real_origem
pred_linear_origem
pred_gru_origem
pred_baseline_origem
```

---

## 17. Gráfico comparativo de 72 horas

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))

plt.plot(
    datas_origem,
    real_origem,
    color="black",
    linewidth=2.5,
    label="Umidade real"
)

plt.plot(
    datas_origem,
    pred_baseline_origem,
    linestyle=":",
    linewidth=2,
    label="Baseline sazonal"
)

plt.plot(
    datas_origem,
    pred_linear_origem,
    linewidth=1.8,
    label="Modelo Linear"
)

plt.plot(
    datas_origem,
    pred_gru_origem,
    linewidth=1.8,
    label="GRU"
)

plt.axvline(
    datas_origem[23],
    color="gray",
    linestyle="--",
    linewidth=1
)

plt.axvline(
    datas_origem[47],
    color="gray",
    linestyle="--",
    linewidth=1
)

plt.xlabel("Data e hora UTC")
plt.ylabel("Umidade relativa máxima (%)")
plt.title(
    "Comparação das trajetórias previstas "
    "para as próximas 72 horas"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## 18. Avaliar um modelo carregado em múltiplas origens

Este bloco pressupõe que sua função `avaliar_horizontes_acumulados()` esteja definida.

### Modelo Linear

```python
resultados_linear_carregado = avaliar_horizontes_acumulados(
    modelo=modelo_linear,
    serie_normalizada=serie_normalizada_linear,
    serie_original=serie,
    indice_inicio_avaliacao=fim_validacao,
    indice_fim_avaliacao=len(serie),
    tau=tau_linear,
    media_treino=media_linear,
    desvio_treino=desvio_linear,
    horizontes=(12, 24, 36, 48, 60, 72),
    passo_origem=12
)
```

### GRU

```python
resultados_gru_carregado = avaliar_horizontes_acumulados(
    modelo=modelo_gru,
    serie_normalizada=serie_normalizada_gru,
    serie_original=serie,
    indice_inicio_avaliacao=fim_validacao,
    indice_fim_avaliacao=len(serie),
    tau=tau_gru,
    media_treino=media_gru,
    desvio_treino=desvio_gru,
    horizontes=(12, 24, 36, 48, 60, 72),
    passo_origem=12
)
```

Resumos:

```python
resumo_linear_carregado = resumir_resultados(
    resultados_linear_carregado
)

resumo_gru_carregado = resumir_resultados(
    resultados_gru_carregado
)

print("Modelo Linear")
print(resumo_linear_carregado)

print("\nGRU")
print(resumo_gru_carregado)
```

---

## 19. Carregar todos os modelos de uma execução

```python
def carregar_modelos_da_execucao(
    pasta,
    id_execucao,
    dispositivo="cpu"
):
    pasta = Path(pasta)

    modelos = {
        "Linear": [],
        "GRU": []
    }

    metadados = {
        "Linear": [],
        "GRU": []
    }

    for nome_modelo in ["Linear", "GRU"]:
        arquivos = list(
            pasta.glob(
                f"{nome_modelo.lower()}_"
                f"semente-*_{id_execucao}.pth"
            )
        )

        arquivos = sorted(
            arquivos,
            key=lambda caminho: int(
                caminho.name
                .split("semente-")[1]
                .split("_")[0]
            )
        )

        if not arquivos:
            raise FileNotFoundError(
                f"Nenhum checkpoint de {nome_modelo} "
                f"encontrado para a execução {id_execucao}."
            )

        for caminho in arquivos:
            modelo, meta = carregar_checkpoint(
                caminho=caminho,
                dispositivo=dispositivo
            )

            modelos[nome_modelo].append(modelo)
            metadados[nome_modelo].append(meta)

    return modelos, metadados
```

Uso:

```python
modelos_treinados, metadados_modelos = (
    carregar_modelos_da_execucao(
        pasta=PASTA_MODELOS,
        id_execucao="2026-09-14_15-20-35",
        dispositivo="cpu"
    )
)

print(
    "Modelos lineares carregados:",
    len(modelos_treinados["Linear"])
)

print(
    "Modelos GRU carregados:",
    len(modelos_treinados["GRU"])
)
```

---

## 20. Avaliar um checkpoint por vez para economizar memória

Esta é a estratégia mais econômica. Somente um modelo permanece carregado por vez, e os resultados numéricos são preservados.

```python
def extrair_semente_do_nome(caminho):
    return int(
        caminho.name
        .split("semente-")[1]
        .split("_")[0]
    )


resumos_carregados = []

for nome_modelo in ["Linear", "GRU"]:
    arquivos = list(
        PASTA_MODELOS.glob(
            f"{nome_modelo.lower()}_"
            f"semente-*_{ID_EXECUCAO}.pth"
        )
    )

    arquivos = sorted(
        arquivos,
        key=extrair_semente_do_nome
    )

    for caminho in arquivos:
        modelo, meta = carregar_checkpoint(
            caminho=caminho,
            dispositivo="cpu"
        )

        serie_norm = (
            (serie - meta["media_treino"])
            / meta["desvio_treino"]
        ).astype(np.float32)

        resultados = avaliar_horizontes_acumulados(
            modelo=modelo,
            serie_normalizada=serie_norm,
            serie_original=serie,
            indice_inicio_avaliacao=fim_validacao,
            indice_fim_avaliacao=len(serie),
            tau=meta["tau"],
            media_treino=meta["media_treino"],
            desvio_treino=meta["desvio_treino"],
            horizontes=(12, 24, 36, 48, 60, 72),
            passo_origem=12
        )

        resumo = resumir_resultados(resultados)
        resumo["modelo"] = nome_modelo
        resumo["semente"] = meta["semente"]
        resumo["arquivo"] = caminho.name

        resumos_carregados.append(resumo)

        # Remove o modelo antes de carregar o próximo.
        modelo.to("cpu")
        del modelo
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

resultados_execucoes_carregados = pd.concat(
    resumos_carregados,
    ignore_index=True
)

print(resultados_execucoes_carregados)
```

---

## 21. Carregar temporariamente na GPU

Se desejar acelerar uma avaliação específica:

```python
modelo_gru, meta_gru = carregar_checkpoint(
    caminho=caminho_gru,
    dispositivo="cuda"
)
```

Ao terminar:

```python
modelo_gru.to("cpu")
del modelo_gru
gc.collect()

torch.cuda.empty_cache()
```

Carregar na CPU é suficiente para gráficos e avaliações pequenas. Use a GPU quando o ganho de velocidade justificar a ocupação da memória.

---

## 22. Salvar também as métricas em CSV

Os modelos preservam os pesos. Para reprodutibilidade, também é útil salvar as tabelas de métricas.

```python
caminho_metricas = (
    PASTA_MODELOS
    / f"metricas_{ID_EXECUCAO}.csv"
)

resultados_execucoes.to_csv(
    caminho_metricas,
    index=False,
    encoding="utf-8-sig"
)

print("Métricas salvas em:", caminho_metricas)
```

Se quiser salvar o resumo final:

```python
caminho_resumo = (
    PASTA_MODELOS
    / f"resumo_final_{ID_EXECUCAO}.csv"
)

resumo_final.to_csv(
    caminho_resumo,
    index=False,
    encoding="utf-8-sig"
)
```

---

## 23. Checklist de segurança e consistência

Antes de usar um checkpoint, verifique:

```python
assert meta_linear["tau"] > 0
assert meta_linear["desvio_treino"] > 0
assert meta_gru["tau"] > 0
assert meta_gru["desvio_treino"] > 0
```

Confirme que o arquivo pertence ao experimento desejado:

```python
print(meta_linear["id_execucao"])
print(meta_linear["semente"])
print(meta_linear["hiperparametros"])
```

Confirme que os modelos estão em avaliação:

```python
assert modelo_linear.training is False
assert modelo_gru.training is False
```

Confirme que as previsões não contêm valores inválidos:

```python
assert np.isfinite(pred_linear_origem).all()
assert np.isfinite(pred_gru_origem).all()
assert np.isfinite(pred_baseline_origem).all()
```

Opcionalmente, para umidade relativa, verifique valores fora da faixa física:

```python
print(
    "Linear fora de 0 a 100:",
    np.sum(
        (pred_linear_origem < 0)
        | (pred_linear_origem > 100)
    )
)

print(
    "GRU fora de 0 a 100:",
    np.sum(
        (pred_gru_origem < 0)
        | (pred_gru_origem > 100)
    )
)
```

Não aplique `clip(0, 100)` silenciosamente durante a comparação. Se decidir restringir as previsões à faixa física, registre essa decisão e aplique o mesmo procedimento a todos os modelos comparáveis.

---

## 24. Fluxo recomendado

### Após o treinamento

```text
Treinar os modelos na GPU
        ↓
Salvar todos os checkpoints
        ↓
Verificar os arquivos gravados
        ↓
Mover os modelos para a CPU
        ↓
Apagar referências
        ↓
Liberar o cache CUDA
```

### Para previsão posterior

```text
Carregar somente o checkpoint necessário
        ↓
Recuperar tau, média e desvio-padrão
        ↓
Normalizar a série com os metadados salvos
        ↓
Executar previsão recursiva
        ↓
Desnormalizar a previsão
        ↓
Gerar gráfico ou métricas
```

### Para comparar todas as sementes

```text
Carregar um checkpoint por vez
        ↓
Avaliar e guardar os resultados numéricos
        ↓
Apagar o modelo da memória
        ↓
Carregar o próximo checkpoint
        ↓
Consolidar médias e desvios-padrão
```

---

## 25. Observação sobre reprodução futura

O checkpoint permite reconstruir o modelo e reproduzir as previsões, desde que também permaneçam disponíveis:

- a definição da classe `GRURegressor`;
- o arquivo de dados ou uma versão identificada dele;
- as funções de pré-processamento;
- as funções `previsao_recursiva()` e `avaliar_horizontes_acumulados()`;
- os índices ou datas que definem treino, validação e teste.

Para uma rastreabilidade ainda maior, registre no checkpoint:

- nome e hash do CSV;
- versão do Python;
- versão do PyTorch;
- versão do NumPy;
- dispositivo usado no treinamento;
- commit do Git correspondente ao código.

Exemplo:

```python
import hashlib
import platform


def calcular_sha256(caminho, tamanho_bloco=1024 * 1024):
    sha256 = hashlib.sha256()

    with open(caminho, "rb") as arquivo_aberto:
        while True:
            bloco = arquivo_aberto.read(tamanho_bloco)

            if not bloco:
                break

            sha256.update(bloco)

    return sha256.hexdigest()


metadados_experimento.update({
    "arquivo_csv": str(arquivo),
    "sha256_csv": calcular_sha256(arquivo),
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "numpy": np.__version__,
    "dispositivo_treinamento": str(device)
})
```

Esses metadados ajudam a demonstrar a reprodutibilidade metodológica do experimento no projeto.
