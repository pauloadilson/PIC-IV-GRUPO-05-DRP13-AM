#!/usr/bin/env python
# coding: utf-8

# # PROCESSO EM MENOS ETAPAS

# ## 1. Preparação da série

# In[39]:


import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

arquivo = (
    "INMET_SE_SP_A707_PRESIDENTE PRUDENTE_"
    "01-01-2025_A_31-12-2025.CSV"
)

COLUNA_ALVO = "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)"

df = pd.read_csv(
    arquivo,
    encoding="latin-1",
    skiprows=8,
    sep=";",
    decimal=","
)

df = df.drop(columns=["Unnamed: 19"], errors="ignore")

df["datetime"] = pd.to_datetime(
    df["Data"].astype(str)
    + " "
    + df["Hora UTC"].astype(str).str.replace(
        " UTC", "", regex=False
    ),
    format="%Y/%m/%d %H%M",
    errors="coerce",
    utc=True
)

df[COLUNA_ALVO] = pd.to_numeric(
    df[COLUNA_ALVO],
    errors="coerce"
)

df = (
    df.sort_values("datetime")
      .drop_duplicates(subset="datetime")
      .reset_index(drop=True)
)

# Mantém a frequência horária
serie_bruta = df[COLUNA_ALVO].copy()

# Para o relatório, informe que os valores ausentes foram preenchidos pelo último valor observado, preservando a causalidade temporal.
serie_bruta = serie_bruta.ffill()

if serie_bruta.isna().any():
    mediana_inicial = serie_bruta.dropna().iloc[:24].median()
    serie_bruta = serie_bruta.fillna(mediana_inicial)

df[COLUNA_ALVO] = serie_bruta

serie = df[COLUNA_ALVO].to_numpy(dtype=np.float32)
datas = df["datetime"].to_numpy()

n_total = len(serie)

fim_treino = int(n_total * 0.70)
fim_validacao = int(n_total * 0.85)

serie_treino = serie[:fim_treino]
serie_validacao = serie[fim_treino:fim_validacao]
serie_teste = serie[fim_validacao:]

datas_treino = datas[:fim_treino]
datas_validacao = datas[fim_treino:fim_validacao]
datas_teste = datas[fim_validacao:]

print(f"Total: {n_total}")
print(f"Treino: {len(serie_treino)}")
print(f"Validação: {len(serie_validacao)}")
print(f"Teste final: {len(serie_teste)}")

print(f"Treino: {datas_treino[0]} até {datas_treino[-1]}")
print(
    f"Validação: {datas_validacao[0]} até {datas_validacao[-1]}"
)
print(f"Teste: {datas_teste[0]} até {datas_teste[-1]}")

media_treino = serie_treino.mean()
desvio_treino = serie_treino.std()

serie_normalizada = (
    (serie - media_treino) / desvio_treino
).astype(np.float32)


# In[49]:


# ÍNDICE DA VALIDAÇÃO

indice_inicio_validacao = fim_treino
indice_fim_validacao = fim_validacao

# ÍNDICE DO TESTE

indice_inicio_teste = fim_validacao
indice_fim_teste = len(serie)


# # 2. Criação das janelas
# 
# Vamos usar as últimas 24 horas para prever a hora seguinte:

# In[41]:


tau = 24

def criar_janelas(serie, tau):
    X = []
    y = []

    for i in range(tau, len(serie)):
        X.append(serie[i - tau:i])
        y.append(serie[i])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32).reshape(-1, 1)
    )

X, y = criar_janelas(
    serie_normalizada,
    tau
)

datas_y = datas[tau:]

corte_treino_janelas = fim_treino - tau
corte_validacao_janelas = fim_validacao - tau

X_train = X[:corte_treino_janelas]
y_train = y[:corte_treino_janelas]

X_val = X[
    corte_treino_janelas:corte_validacao_janelas
]
y_val = y[
    corte_treino_janelas:corte_validacao_janelas
]

X_test = X[corte_validacao_janelas:]
y_test = y[corte_validacao_janelas:]

datas_val = datas_y[
    corte_treino_janelas:corte_validacao_janelas
]

datas_test = datas_y[
    corte_validacao_janelas:
]

print(f"Treino: {X_train.shape}, {y_train.shape}")
print(f"Validação: {X_val.shape}, {y_val.shape}")
print(f"Teste: {X_test.shape}, {y_test.shape}")


# # 3. Treinamento do modelo

# In[43]:


# Preparação dos tensores e DataLoader

def criar_dataloader(
    X,
    y,
    batch_size=32,
    shuffle=False
):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available()
    )

    return dataset, loader


train_dataset, train_loader = criar_dataloader(
    X_train,
    y_train,
    batch_size=32,
    shuffle=True
)

val_dataset, val_loader = criar_dataloader(
    X_val,
    y_val,
    batch_size=128,
    shuffle=False
)

test_dataset, test_loader = criar_dataloader(
    X_test,
    y_test,
    batch_size=128,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Dispositivo:", device)


# In[50]:


def treinar_modelo(
    modelo,
    train_loader,
    val_loader,
    device,
    epocas=100,
    lr=0.001,
    paciencia=15
):
    modelo = modelo.to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        modelo.parameters(),
        lr=lr
    )

    historico = {
        "treino": [],
        "validacao": []
    }

    melhor_val_loss = np.inf
    melhor_estado = None
    epocas_sem_melhora = 0

    for epoch in range(epocas):
        modelo.train()
        soma_treino = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            pred = modelo(X_batch)

            loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()

            soma_treino += (
                loss.item() * len(X_batch)
            )

        loss_treino = (
            soma_treino / len(train_loader.dataset)
        )

        modelo.eval()
        soma_validacao = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                pred = modelo(X_batch)

                loss = loss_fn(pred, y_batch)

                soma_validacao += (
                    loss.item() * len(X_batch)
                )

        loss_validacao = (
            soma_validacao / len(val_loader.dataset)
        )

        historico["treino"].append(loss_treino)
        historico["validacao"].append(
            loss_validacao
        )

        if loss_validacao < melhor_val_loss:
            melhor_val_loss = loss_validacao

            melhor_estado = {
                nome: tensor.detach().cpu().clone()
                for nome, tensor
                in modelo.state_dict().items()
            }

            epocas_sem_melhora = 0
        else:
            epocas_sem_melhora += 1

        if epocas_sem_melhora >= paciencia:
            break

        if (epoch + 1) % 10 == 0:
            print(
                f"Época {epoch + 1:03d} | "
                f"MSE: {perda_media:.6f}"
            )

    modelo.load_state_dict(melhor_estado)

    return modelo, historico


# In[45]:


SEMENTES = [10, 20, 30, 40, 50]

import random

def configurar_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[46]:


class GRURegressor(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2):
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
        x = x.unsqueeze(-1)
        output, _ = self.gru(x)
        return self.output(output[:, -1, :])


# In[47]:


def criar_modelo_linear(tau):
    return nn.Linear(tau, 1)


def criar_modelo_gru():
    return GRURegressor(
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )


# In[48]:


modelos_treinados = {
    "Linear": [],
    "GRU": []
}

historicos = {
    "Linear": [],
    "GRU": []
}

for semente in SEMENTES:
    configurar_semente(semente)

    modelo_linear = criar_modelo_linear(tau)

    modelo_linear, hist_linear = treinar_modelo(
        modelo=modelo_linear,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epocas=200,
        lr=0.001,
        paciencia=20
    )

    modelos_treinados["Linear"].append(
        modelo_linear
    )

    historicos["Linear"].append(
        hist_linear
    )

    configurar_semente(semente)

    modelo_gru = criar_modelo_gru()

    modelo_gru, hist_gru = treinar_modelo(
        modelo=modelo_gru,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epocas=200,
        lr=0.001,
        paciencia=20
    )

    modelos_treinados["GRU"].append(
        modelo_gru
    )

    historicos["GRU"].append(
        hist_gru
    )


# # 4. Avaliação de uma hora à frente

# In[53]:


for model in ["Linear", "GRU"]:
    print(f"\nMétricas para o modelo {model}:")

    for i, (modelo, hist) in enumerate(
        zip(
            modelos_treinados[model],
            historicos[model]
        )
    ):
        print(f"\nSemente: {SEMENTES[i]}")

        perda_treino = hist["treino"][-1]
        perda_validacao = hist["validacao"][-1]

        print(f"Perda de treino: {perda_treino:.6f}")
        print(f"Perda de validação: {perda_validacao:.6f}")

        X_test_tensor = torch.tensor(
            X_test,
            dtype=torch.float32,
            device=device
        )

        modelo.eval()

        with torch.no_grad():
            pred_one_step_norm = (
                modelo(X_test_tensor)
                .squeeze(1)
                .cpu()
                .numpy()
            )

        pred_one_step = (
            pred_one_step_norm * desvio_treino
            + media_treino
        )

        real_one_step = (
            y_test.squeeze(1) * desvio_treino
            + media_treino
        )

        # métricas

        mae = np.mean(
            np.abs(real_one_step - pred_one_step)
        )

        rmse = np.sqrt(
            np.mean((real_one_step - pred_one_step) ** 2)
        )

        print(f"MAE de 1 hora: {mae:.3f}")
        print(f"RMSE de 1 hora: {rmse:.3f}")

        # GRÁFICO 

        n_plot = 24 * 14

        plt.figure(figsize=(14, 5))

        plt.plot(
            datas_test[-n_plot:],
            real_one_step[-n_plot:],
            label="Umidade real"
        )

        plt.plot(
            datas_test[-n_plot:],
            pred_one_step[-n_plot:],
            label="Previsão de 1 hora"
        )

        plt.xlabel("Data e hora UTC")
        plt.ylabel("Umidade relativa máxima (%)")
        plt.title("Avaliação de uma hora à frente")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# # 5. Previsão recursiva para 48 E 72 horas
# 
# Usaremos como origem o último ponto do treinamento. O modelo não terá acesso aos valores reais do teste durante a geração.

# In[ ]:


def obter_dispositivo(modelo):
    return next(modelo.parameters()).device


# In[ ]:


def previsao_recursiva(
    modelo,
    historico_normalizado,
    tau,
    horizonte
):
    modelo.eval()

    device_modelo = obter_dispositivo(modelo)

    janela = torch.tensor(
        historico_normalizado[-tau:],
        dtype=torch.float32,
        device=device_modelo
    )

    previsoes = []

    with torch.no_grad():
        for _ in range(horizonte):
            entrada = janela.reshape(1, tau)

            proximo = modelo(entrada).squeeze()

            previsoes.append(proximo.item())

            janela = torch.cat([
                janela[1:],
                proximo.reshape(1)
            ])

    return np.array(previsoes, dtype=np.float32)


# In[13]:


def calcular_metricas(real, previsto):
    mae = np.mean(np.abs(real - previsto))
    rmse = np.sqrt(np.mean((real - previsto) ** 2))

    return mae, rmse


# ## 5.1. Previsão recursiva para 72

# In[58]:


# GERANDO 72 HORAS DE PREVISÃO

historico_treino_norm = serie_normalizada[:indice_inicio_validacao]

for model in ["Linear", "GRU"]:
    print(f"\nMétricas para o modelo {model}:")

    for i, (modelo, hist) in enumerate(
        zip(
            modelos_treinados[model],
            historicos[model]
        )
    ):

        pred_72_norm = previsao_recursiva(
            modelo=modelo,
            historico_normalizado=historico_treino_norm,
            tau=tau,
            horizonte=72
        )


        pred_72 = (
            pred_72_norm * desvio_treino
            + media_treino
        )

        real_72 = serie_teste[:72]
        datas_72 = datas_teste[:72]

        # GRÁFICO 72 HORAS

        plt.figure(figsize=(14, 5))

        plt.plot(
            datas_72,
            real_72,
            marker="o",
            markersize=3,
            label="Umidade real"
        )

        plt.plot(
            datas_72,
            pred_72,
            marker="o",
            markersize=3,
            label="Previsão recursiva"
        )

        plt.axvline(
            datas_72[23],
            color="gray",
            linestyle=":",
            label="24 horas"
        )

        plt.axvline(
            datas_72[47],
            color="orange",
            linestyle=":",
            label="48 horas"
        )

        plt.xlabel("Data e hora UTC")
        plt.ylabel("Umidade relativa máxima (%)")
        plt.title("Previsão recursiva para as próximas 72 horas")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ### 5.2.1. Como comparar o desempenho dentro das primeiras 12h, 24h, 36h, 48h, 60h e 72h
# 
# Isso mede a qualidade da trajetória completa até cada horizonte.

# In[59]:


for horizonte in [12, 24, 36, 48, 60, 72]:
    mae_h, rmse_h = calcular_metricas(
        real_72[:horizonte],
        pred_72[:horizonte]
    )

    print(
        f"Horizonte acumulado de {horizonte:02d} horas | "
        f"MAE = {mae_h:.3f} | "
        f"RMSE = {rmse_h:.3f}"
    )


# # 6. Avaliação de múltiplas origens em várias janelas de 12h, 24h, 36h, 48h, 60h e 72h

# In[60]:


def avaliar_horizontes_acumulados(
    modelo,
    serie_normalizada,
    serie_original,
    indice_inicio_avaliacao,
    indice_fim_avaliacao,
    tau,
    media_treino,
    desvio_treino,
    horizontes=(12, 24, 48, 72),
    passo_origem=12
):
    horizonte_maximo = max(horizontes)

    resultados = {
        h: {
            "residuos": [],
            "mae_por_origem": [],
            "rmse_por_origem": [],
            "origens": []
        }
        for h in horizontes
    }

    ultimo_inicio = (
        indice_fim_avaliacao
        - horizonte_maximo
    )

    for origem in range(
        indice_inicio_avaliacao,
        ultimo_inicio + 1,
        passo_origem
    ):
        historico = serie_normalizada[:origem]

        pred_norm = previsao_recursiva(
            modelo=modelo,
            historico_normalizado=historico,
            tau=tau,
            horizonte=horizonte_maximo
        )

        pred = (
            pred_norm * desvio_treino
            + media_treino
        )

        real = serie_original[
            origem:origem + horizonte_maximo
        ]

        for h in horizontes:
            residuos_h = (
                real[:h] - pred[:h]
            )

            resultados[h]["residuos"].extend(
                residuos_h.tolist()
            )

            resultados[h][
                "mae_por_origem"
            ].append(
                np.mean(np.abs(residuos_h))
            )

            resultados[h][
                "rmse_por_origem"
            ].append(
                np.sqrt(
                    np.mean(residuos_h ** 2)
                )
            )

            resultados[h]["origens"].append(
                origem
            )

    return resultados


# ## 6.1. Baseline sazonal 

# In[62]:


def previsao_sazonal(
    historico_original,
    horizonte,
    periodo=24
):
    padrao = np.asarray(
        historico_original[-periodo:],
        dtype=np.float32
    )

    repeticoes = int(
        np.ceil(horizonte / periodo)
    )

    return np.tile(
        padrao,
        repeticoes
    )[:horizonte]


# ### 6.1.1. Avaliação do baseline com as mesmas origens

# In[63]:


def avaliar_baseline_sazonal(
    serie_original,
    indice_inicio_avaliacao,
    indice_fim_avaliacao,
    horizontes=(12, 24, 36, 48, 60, 72),
    passo_origem=12,
    periodo=24
):
    horizonte_maximo = max(horizontes)

    resultados = {
        h: {
            "residuos": [],
            "mae_por_origem": [],
            "rmse_por_origem": [],
            "origens": []
        }
        for h in horizontes
    }

    ultimo_inicio = (
        indice_fim_avaliacao
        - horizonte_maximo
    )

    for origem in range(
        indice_inicio_avaliacao,
        ultimo_inicio + 1,
        passo_origem
    ):
        historico = serie_original[:origem]

        pred = previsao_sazonal(
            historico_original=historico,
            horizonte=horizonte_maximo,
            periodo=periodo
        )

        real = serie_original[
            origem:origem + horizonte_maximo
        ]

        for h in horizontes:
            residuos_h = (
                real[:h] - pred[:h]
            )

            resultados[h]["residuos"].extend(
                residuos_h.tolist()
            )

            resultados[h][
                "mae_por_origem"
            ].append(
                np.mean(np.abs(residuos_h))
            )

            resultados[h][
                "rmse_por_origem"
            ].append(
                np.sqrt(
                    np.mean(residuos_h ** 2)
                )
            )

            resultados[h]["origens"].append(
                origem
            )

    return resultados


# In[65]:


resultados_baseline_teste = (
    avaliar_baseline_sazonal(
        serie_original=serie,
        indice_inicio_avaliacao=fim_validacao,
        indice_fim_avaliacao=len(serie),
        horizontes=(12, 24, 36, 48, 60, 72),
        passo_origem=12,
        periodo=24
    )
)


# ## 6.1. Como resumir os resultados
# 
# Há duas formas legítimas de agregar os resultados.
# 
# 

# In[67]:


def resumir_resultados(resultados):
    resumo = []

    for horizonte, dados in resultados.items():
        residuos = np.asarray(
            dados["residuos"],
            dtype=float
        )

        mae = np.mean(np.abs(residuos))

        rmse = np.sqrt(
            np.mean(residuos ** 2)
        )

        mediana_abs = np.median(
            np.abs(residuos)
        )

        resumo.append({
            "horizonte": horizonte,
            "mae": mae,
            "rmse": rmse,
            "mediana_absoluta": mediana_abs,
            "n_origens": len(
                dados["mae_por_origem"]
            ),
            "n_previsoes": len(residuos)
        })

    return pd.DataFrame(resumo)


# In[68]:


registros_sementes = []

for nome_modelo in ["Linear", "GRU"]:
    for semente, modelo in zip(
        SEMENTES,
        modelos_treinados[nome_modelo]
    ):
        resultados = avaliar_horizontes_acumulados(
            modelo=modelo,
            serie_normalizada=serie_normalizada,
            serie_original=serie,
            indice_inicio_avaliacao=fim_validacao,
            indice_fim_avaliacao=len(serie),
            tau=tau,
            media_treino=media_treino,
            desvio_treino=desvio_treino,
            horizontes=(12, 24, 36, 48, 60, 72),
            passo_origem=12
        )

        resumo = resumir_resultados(resultados)

        resumo["modelo"] = nome_modelo
        resumo["semente"] = semente

        registros_sementes.append(resumo)

resultados_execucoes = pd.concat(
    registros_sementes,
    ignore_index=True
)


# ### 6.1.1. Forma A: erro global de todas as previsões
# 
# Junta todos os resíduos de todas as origens:

# In[69]:


resumo_final = (
    resultados_execucoes
    .groupby(["modelo", "horizonte"])
    .agg(
        mae_media=("mae", "mean"),
        mae_dp=("mae", "std"),
        rmse_media=("rmse", "mean"),
        rmse_dp=("rmse", "std"),
        execucoes=("semente", "nunique")
    )
    .reset_index()
)

print(resumo_final)


# # 7. Teste de hipótese entre Linear e GRU
# 
# ### Hipóteses
# $$
# H_0: \text{não há diferença de desempenho entre os modelos}
# $$
# $$
# H_1: \text{há diferença de desempenho entre os modelos}
# $$

# ## 7.1. Teste t pareado

# In[ ]:


from scipy.stats import ttest_rel

estatistica, p_valor = ttest_rel(
    mae_linear_por_origem,
    mae_gru_por_origem
)


# ## 7.2. Wilcoxon pareado

# In[ ]:


from scipy.stats import wilcoxon

estatistica, p_valor = wilcoxon(
    mae_linear_por_origem,
    mae_gru_por_origem
)


# # 8. Gráficos

# ## 8.1. MAE médio por horizonte

# In[71]:


fig, ax = plt.subplots(figsize=(10, 5))

for modelo, grupo in resumo_final.groupby("modelo"):
    ax.errorbar(
        grupo["horizonte"],
        grupo["mae_media"],
        yerr=grupo["mae_dp"],
        marker="o",
        capsize=4,
        label=modelo
    )

ax.set_xlabel("Horizonte acumulado, em horas")
ax.set_ylabel(
    "MAE, em pontos percentuais de umidade"
)
ax.set_title(
    "MAE por modelo e horizonte acumulado"
)
ax.set_xticks([12, 24, 36, 48, 60, 72])
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()


# ### 8.2 RMSE médio por horizonte

# In[72]:


fig, ax = plt.subplots(figsize=(10, 5))

for modelo, grupo in resumo_final.groupby("modelo"):
    ax.errorbar(
        grupo["horizonte"],
        grupo["rmse_media"],
        yerr=grupo["rmse_dp"],
        marker="o",
        capsize=4,
        label=modelo
    )

ax.set_xlabel("Horizonte acumulado, em horas")
ax.set_ylabel(
    "RMSE, em pontos percentuais de umidade"
)
ax.set_title(
    "RMSE por modelo e horizonte acumulado"
)
ax.set_xticks([12, 24, 48, 72])
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()


# ## 8.3. Boxplot do MAE por origem

# In[73]:


dados_boxplot = []

resultados_linear = (
    resultados_execucoes[
        resultados_execucoes["modelo"] == "Linear"
    ]
    .groupby("horizonte")
    .agg(
        mae_por_origem=("mae", lambda x: list(x))
    )
    .to_dict(orient="index")
)

resultados_gru = (
    resultados_execucoes[
        resultados_execucoes["modelo"] == "GRU"
    ]
    .groupby("horizonte")
    .agg(
        mae_por_origem=("mae", lambda x: list(x))
    )
    .to_dict(orient="index")
)

for nome, resultados in [
    ("Linear", resultados_linear),
    ("GRU", resultados_gru),
    ("Baseline", resultados_baseline_teste)
]:
    for horizonte in [12, 24, 48, 72]:
        for valor in resultados[
            horizonte
        ]["mae_por_origem"]:
            dados_boxplot.append({
                "modelo": nome,
                "horizonte": horizonte,
                "mae": valor
            })

df_boxplot = pd.DataFrame(dados_boxplot)


# In[74]:


for horizonte in [12, 24, 48, 72]:
    dados_h = df_boxplot[
        df_boxplot["horizonte"] == horizonte
    ]

    grupos = [
        dados_h.loc[
            dados_h["modelo"] == modelo,
            "mae"
        ].to_numpy()
        for modelo in ["Baseline", "Linear", "GRU"]
    ]

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        grupos,
        tick_labels=["Baseline", "Linear", "GRU"]
    )

    plt.ylabel(
        "MAE por origem, em pontos percentuais"
    )

    plt.title(
        f"Distribuição do MAE até {horizonte} horas"
    )

    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


# ## 8.4. Trajetória comparativa de 72 horas

# In[77]:


resultados_execucoes[
        (resultados_execucoes["modelo"] == "Linear")
        & (resultados_execucoes["horizonte"] == 72)
    ]


# In[78]:


resultados_execucoes[
        (resultados_execucoes["modelo"] == "GRU")
        & (resultados_execucoes["horizonte"] == 72)
    ]


# In[80]:


resultados_execucoes[
        (resultados_execucoes["modelo"] == "Baseline")
        & (resultados_execucoes["horizonte"] == 72)
    ]


# In[ ]:


datas_origem = datas_teste[:-72]
real_origem = serie_teste[:-72]
pred_linear_origem = (
    resultados_execucoes[
        (resultados_execucoes["modelo"] == "Linear")
        & (resultados_execucoes["horizonte"] == 72)
    ]
    .sort_values("semente")["mae"]
    .explode()
    .to_numpy(dtype=np.float32)
)
pred_gru_origem = (
    resultados_execucoes[
        (resultados_execucoes["modelo"] == "GRU")
        & (resultados_execucoes["horizonte"] == 72)
    ]
    .sort_values("semente")["mae"]
    .explode()
    .to_numpy(dtype=np.float32)
)
pred_baseline_origem = (
    resultados_execucoes[
        (resultados_execucoes["modelo"] == "Baseline")
        & (resultados_execucoes["horizonte"] == 72)
    ]
    .sort_values("semente")["mae"]
    .explode()
    .to_numpy(dtype=np.float32)
)



# In[85]:


plt.figure(figsize=(14, 5))

plt.plot(
    datas_origem,
    real_origem,
    linewidth=2,
    label="Real"
)

plt.plot(
    datas_origem,
    pred_linear_origem,
    label="Linear"
)

plt.plot(
    datas_origem,
    pred_gru_origem,
    label="GRU"
)

plt.plot(
    datas_origem,
    pred_baseline_origem,
    linestyle=":",
    label="Baseline sazonal"
)

plt.axvline(
    datas_origem[23],
    color="gray",
    linestyle=":"
)

plt.axvline(
    datas_origem[47],
    color="gray",
    linestyle=":"
)

plt.xlabel("Data e hora UTC")
plt.ylabel("Umidade relativa máxima (%)")
plt.title(
    "Comparação das trajetórias previstas em 72 horas"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

