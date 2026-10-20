# Comparação final com atributos TU: Ridge, HGB e GRU

Este notebook repete a comparação principal utilizando exclusivamente **temperatura e umidade**, conjunto compatível com a futura coleta operacional da HG Brasil.

## Objetivo

Comparar, sob o mesmo protocolo temporal:

- baseline sazonal de 24 horas;
- Ridge multialvo direta;
- HistGradientBoostingRegressor com 48 estimadores;
- GRU_B direta multialvo em cinco sementes.

## Protocolo

- Entradas meteorológicas: temperatura e umidade;
- Atributos temporais: hora e dia do ano em seno e cosseno;
- Histórico: 24 horas;
- Saídas: temperatura e umidade de t+1 a t+24;
- Treino: 2011–2023, excluindo 2022;
- Validação: 2024;
- Teste: 2025;
- Avaliação adicional: 2026 parcial;
- Sem covariáveis meteorológicas futuras reais;
- Ridge e HGB em CPU;
- GRU em CUDA e inferência compatível com CPU.

O notebook não usa o ponto de orvalho. Isso permite que o modelo posteriormente receba diretamente os campos de temperatura e umidade coletados pela HG Brasil.

## 1. Importações e configuração


```python
from pathlib import Path
import copy,json,platform,random,sys,time
import joblib,matplotlib.pyplot as plt,numpy as np,pandas as pd,torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from torch import nn
from torch.utils.data import DataLoader,Dataset

DIRETORIO_TU=Path("dados_processados_multivariados/TU")
DIRETORIO_SAIDA=Path("resultados_modelagem/comparacao_final_TU")
DIRETORIO_MODELOS=DIRETORIO_SAIDA/"modelos"
DIRETORIO_SAIDA.mkdir(parents=True,exist_ok=True); DIRETORIO_MODELOS.mkdir(parents=True,exist_ok=True)
SPLITS=["treino","validacao","teste","avaliacao_2026"]
SEMENTES=[10,20,30,40,50]
ALPHAS=[0.0,0.01,0.1,1.0,10.0,100.0]
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
USAR_AMP=DEVICE.type=="cuda"
HGB_B=dict(loss="squared_error",learning_rate=0.05,max_iter=250,max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=1.0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=15,tol=1e-5,random_state=42)
CONFIG_GRU=dict(n_hist=6,n_fut=4,hidden_size=96,num_layers=2,dropout=0.15,decoder_hidden=192)
print("Dispositivo GRU:",DEVICE,"| TU | sementes:",SEMENTES)
```

    Dispositivo GRU: cuda | TU | sementes: [10, 20, 30, 40, 50]
    

## 2. Carregamento TU


```python
def load_npz(path):
 with np.load(path) as a:return {k:a[k] for k in a.files}
splits={s:load_npz(DIRETORIO_TU/f"{s}.npz") for s in SPLITS}
scaler_y=joblib.load(DIRETORIO_TU/"scaler_y.joblib")
with (DIRETORIO_TU/"metadados.json").open(encoding="utf-8") as f:meta=json.load(f)
assert meta["configuracao"]=="TU" and meta["usa_covariaveis_meteorologicas_futuras"] is False
for s in SPLITS:print(s,splits[s]["X_historico"].shape,splits[s]["y"].shape)
```

    treino (84744, 24, 6) (84744, 24, 2)
    validacao (7837, 24, 6) (7837, 24, 2)
    teste (7970, 24, 6) (7970, 24, 2)
    avaliacao_2026 (4551, 24, 6) (4551, 24, 2)
    

## 3. Matrizes tabulares e utilidades


```python
def Xtab(d):return np.concatenate([d["X_historico"].reshape(len(d["y"]),-1),d["X_futuro_temporal"].reshape(len(d["y"]),-1)],axis=1).astype(np.float32)
def inv(y):
 sh=y.shape;return scaler_y.inverse_transform(y.reshape(-1,2)).reshape(sh)
X={s:Xtab(splits[s]) for s in SPLITS}; yn={s:splits[s]["y"] for s in SPLITS}; y={s:inv(yn[s]) for s in SPLITS}
assert X["treino"].shape[1]==240
ALVOS={0:("temperatura_c","°C"),1:("umidade_pct","p.p.")}
def metrica(real,pred,modelo,split,semente=None):
 out=[]
 for h in range(24):
  for j,(a,u) in ALVOS.items():
   r=real[:,h,j];p=pred[:,h,j]
   out.append(dict(modelo=modelo,semente=semente,split=split,horizonte_horas=h+1,alvo=a,unidade=u,mae=mean_absolute_error(r,p),rmse=mean_squared_error(r,p)**.5,vies=float(np.mean(p-r)),n=len(r)))
 return pd.DataFrame(out)
```

## 4. Baseline sazonal e Ridge


```python
# Baseline: repete as 24 horas históricas dos canais temperatura/umidade.
pred_saz={s:inv(splits[s]["X_historico"][:,:,:2]) for s in ["validacao","teste","avaliacao_2026"]}

sel=[];mods={}
yv=y["validacao"]
for a in ALPHAS:
 print("Treinando Ridge alpha",a)
 m=Ridge(alpha=a,solver="lsqr",tol=1e-4).fit(X["treino"],yn["treino"].reshape(len(yn["treino"]),-1))
 pv=inv(m.predict(X["validacao"]).reshape(-1,24,2))
 mt=mean_absolute_error(yv[:,:,0].ravel(),pv[:,:,0].ravel());mu=mean_absolute_error(yv[:,:,1].ravel(),pv[:,:,1].ravel())
 crit=((mt/scaler_y.scale_[0])+(mu/scaler_y.scale_[1]))/2
 sel.append(dict(alpha=a,criterio=crit,mae_temp_c=mt,mae_umidade_pp=mu));mods[a]=m
sel=pd.DataFrame(sel).sort_values("criterio");MELHOR_ALPHA=float(sel.iloc[0].alpha);ridge=mods[MELHOR_ALPHA]
print("Alpha Ridge:",MELHOR_ALPHA);display(sel)
```

    Treinando Ridge alpha 0.0
    Treinando Ridge alpha 0.01
    Treinando Ridge alpha 0.1
    Treinando Ridge alpha 1.0
    Treinando Ridge alpha 10.0
    Treinando Ridge alpha 100.0
    Alpha Ridge: 1.0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alpha</th>
      <th>criterio</th>
      <th>mae_temp_c</th>
      <th>mae_umidade_pp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>1.00</td>
      <td>0.378751</td>
      <td>1.696510</td>
      <td>8.258918</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.10</td>
      <td>0.378760</td>
      <td>1.696639</td>
      <td>8.258741</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>0.378761</td>
      <td>1.696652</td>
      <td>8.258740</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.378761</td>
      <td>1.696654</td>
      <td>8.258739</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.00</td>
      <td>0.378908</td>
      <td>1.698042</td>
      <td>8.259017</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100.00</td>
      <td>0.379321</td>
      <td>1.701614</td>
      <td>8.261151</td>
    </tr>
  </tbody>
</table>
</div>


## 5. HGB completo TU em 48 estimadores


```python
modelos_hgb={};log_hgb=[]
for h in range(1,25):
 for j,(alvo,unidade) in ALVOS.items():
  path=DIRETORIO_MODELOS/f"hgb_TU_{alvo}_h{h:02d}.joblib"
  if path.exists():m=joblib.load(path);origem="carregado";dt=0
  else:
   print("HGB",alvo,"t+",h);m=HistGradientBoostingRegressor(**HGB_B);t=time.perf_counter();m.fit(X["treino"],y["treino"][:,h-1,j]);dt=time.perf_counter()-t;joblib.dump(m,path);origem="treinado"
  modelos_hgb[(h,j)]=m;log_hgb.append(dict(horizonte=h,alvo=alvo,origem=origem,tempo_s=dt,n_iter=m.n_iter_))
def pred_hgb(xx):
 o=np.empty((len(xx),24,2),np.float32)
 for h in range(1,25):
  for j in ALVOS:o[:,h-1,j]=modelos_hgb[(h,j)].predict(xx)
 return o
pred_h={s:pred_hgb(X[s]) for s in ["validacao","teste","avaliacao_2026"]}
```

    HGB temperatura_c t+ 1
    HGB umidade_pct t+ 1
    HGB temperatura_c t+ 2
    HGB umidade_pct t+ 2
    HGB temperatura_c t+ 3
    HGB umidade_pct t+ 3
    HGB temperatura_c t+ 4
    HGB umidade_pct t+ 4
    HGB temperatura_c t+ 5
    HGB umidade_pct t+ 5
    HGB temperatura_c t+ 6
    HGB umidade_pct t+ 6
    HGB temperatura_c t+ 7
    HGB umidade_pct t+ 7
    HGB temperatura_c t+ 8
    HGB umidade_pct t+ 8
    HGB temperatura_c t+ 9
    HGB umidade_pct t+ 9
    HGB temperatura_c t+ 10
    HGB umidade_pct t+ 10
    HGB temperatura_c t+ 11
    HGB umidade_pct t+ 11
    HGB temperatura_c t+ 12
    HGB umidade_pct t+ 12
    HGB temperatura_c t+ 13
    HGB umidade_pct t+ 13
    HGB temperatura_c t+ 14
    HGB umidade_pct t+ 14
    HGB temperatura_c t+ 15
    HGB umidade_pct t+ 15
    HGB temperatura_c t+ 16
    HGB umidade_pct t+ 16
    HGB temperatura_c t+ 17
    HGB umidade_pct t+ 17
    HGB temperatura_c t+ 18
    HGB umidade_pct t+ 18
    HGB temperatura_c t+ 19
    HGB umidade_pct t+ 19
    HGB temperatura_c t+ 20
    HGB umidade_pct t+ 20
    HGB temperatura_c t+ 21
    HGB umidade_pct t+ 21
    HGB temperatura_c t+ 22
    HGB umidade_pct t+ 22
    HGB temperatura_c t+ 23
    HGB umidade_pct t+ 23
    HGB temperatura_c t+ 24
    HGB umidade_pct t+ 24
    

## 6. GRU_B TU com cinco sementes


```python
class DS(Dataset):
 def __init__(self,d):self.xh=torch.from_numpy(d["X_historico"]).float();self.xf=torch.from_numpy(d["X_futuro_temporal"]).float();self.y=torch.from_numpy(d["y"]).float()
 def __len__(self):return len(self.y)
 def __getitem__(self,i):return self.xh[i],self.xf[i],self.y[i]
class GRU(nn.Module):
 def __init__(self,n_hist=6,n_fut=4,hidden_size=96,num_layers=2,dropout=.15,decoder_hidden=192):
  super().__init__();self.config=locals().copy();self.config.pop("self")
  self.gru=nn.GRU(n_hist,hidden_size,num_layers,batch_first=True,dropout=dropout)
  self.dec=nn.Sequential(nn.Linear(hidden_size+n_fut,decoder_hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(decoder_hidden,2))
 def forward(self,xh,xf):
  _,h=self.gru(xh);c=h[-1].unsqueeze(1).expand(-1,24,-1);return self.dec(torch.cat([c,xf],-1))
dsets={s:DS(splits[s]) for s in SPLITS}
def seed_all(sd):
 random.seed(sd);np.random.seed(sd);torch.manual_seed(sd)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(sd)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def loaders(sd):
 g=torch.Generator();g.manual_seed(sd)
 return {s:DataLoader(dsets[s],128,shuffle=s=="treino",generator=g if s=="treino" else None,pin_memory=DEVICE.type=="cuda") for s in SPLITS}
def epoch(m,l,opt=None,sc=None):
 tr=opt is not None;m.train(tr);tot=n=0
 for xh,xf,yy in l:
  xh,xf,yy=xh.to(DEVICE),xf.to(DEVICE),yy.to(DEVICE)
  if tr:opt.zero_grad(set_to_none=True)
  with torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=USAR_AMP):pr=m(xh,xf);loss=nn.functional.mse_loss(pr,yy)
  if tr:sc.scale(loss).backward();sc.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),1);sc.step(opt);sc.update()
  tot+=loss.item()*len(yy);n+=len(yy)
 return tot/n
@torch.no_grad()
def predict(m,l):
 m.eval();o=[]
 for xh,xf,_ in l:
  with torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=USAR_AMP):o.append(m(xh.to(DEVICE),xf.to(DEVICE)).float().cpu().numpy())
 return inv(np.concatenate(o))
pred_g={};hist=[]
```


```python
for sd in SEMENTES:
 seed_all(sd);ls=loaders(sd);path=DIRETORIO_MODELOS/f"gru_B_TU_seed_{sd}.pt";m=GRU(**CONFIG_GRU).to(DEVICE)
 if path.exists():m.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True));print("GRU carregada",sd)
 else:
  opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-5);sc=torch.amp.GradScaler("cuda",enabled=USAR_AMP);best=np.inf;state=None;wait=0
  for e in range(1,61):
   tr=epoch(m,ls["treino"],opt,sc);va=epoch(m,ls["validacao"]);hist.append(dict(semente=sd,epoca=e,treino=tr,validacao=va));print(sd,e,tr,va)
   if va<best-1e-6:best=va;state=copy.deepcopy(m.state_dict());wait=0
   else:wait+=1
   if wait>=8:break
  m.load_state_dict(state);torch.save({k:v.cpu() for k,v in m.state_dict().items()},path)
 pred_g[sd]={s:predict(m,ls[s]) for s in ["validacao","teste","avaliacao_2026"]};del m
 if DEVICE.type=="cuda":torch.cuda.empty_cache()
```

    10 1 0.2640623377271564 0.26099829373985084
    10 2 0.2241561569404998 0.25065205043142086
    10 3 0.21866562792321478 0.25083768425854674
    10 4 0.2139905636735738 0.25570414471538055
    10 5 0.20990333079526505 0.25935342361542346
    10 6 0.2064032243872616 0.2453815161904252
    10 7 0.20168135423706643 0.24677541992034446
    10 8 0.19612854226536042 0.2483787247078051
    10 9 0.18862164233653209 0.250437685082545
    10 10 0.18153086248959335 0.2599650975395195
    10 11 0.17356116456259843 0.2701363332224498
    10 12 0.16597964453773728 0.2740250241738186
    10 13 0.16010531763550828 0.26797818734004086
    10 14 0.15376399619928402 0.2760842390908899
    20 1 0.2637196652620066 0.26108418184970583
    20 2 0.2245134230512156 0.24707556002557893
    20 3 0.21913025394767133 0.24788935438294762
    20 4 0.21515537301136642 0.2493869415342359
    20 5 0.21053066661914258 0.24205861524692432
    20 6 0.20636703407928736 0.2508393701242445
    20 7 0.2011168218869883 0.24854578369465716
    20 8 0.1950082882632505 0.25522830938314073
    20 9 0.188596546452995 0.2488289818593597
    20 10 0.18160594647834133 0.26083631128742024
    20 11 0.17481632820750548 0.2628651426503043
    20 12 0.16738181387152629 0.2566387852492481
    20 13 0.16081149210889809 0.2857238348971311
    30 1 0.26304573592941133 0.2567670848299019
    30 2 0.22427348531891647 0.25481449863607825
    30 3 0.2183484313317627 0.24435787935547185
    30 4 0.21436132635665572 0.2534152045132966
    30 5 0.21099012432361272 0.24089439864310436
    30 6 0.20661878363838346 0.26449856501753816
    30 7 0.20283803287364263 0.2506869315988892
    30 8 0.19721182562564865 0.2487235842083689
    30 9 0.19145848058156384 0.2471261644165279
    30 10 0.1851041692293344 0.25340741842935716
    30 11 0.1775179057835092 0.2568633695766073
    30 12 0.17094533823878572 0.2668200500915059
    30 13 0.1638313386957492 0.26251685792285295
    40 1 0.2654720289963428 0.2602862249748491
    40 2 0.22431737874369664 0.25893469518664936
    40 3 0.21809753268059665 0.2515076194902084
    40 4 0.21415012349580936 0.2484344188820324
    40 5 0.21007146832401213 0.25434639919404245
    40 6 0.20550371572286674 0.2502370019391877
    40 7 0.20027315128342485 0.2490052753298642
    40 8 0.19397937185920963 0.2494894201985287
    40 9 0.18718969839935679 0.25954268140038833
    40 10 0.18043978659427842 0.2544140671448195
    40 11 0.1735123325155813 0.2614435403778554
    40 12 0.16549169744272516 0.26964694725584126
    50 1 0.2661993734874053 0.26189081838809136
    50 2 0.22499599453579835 0.2559273575094033
    50 3 0.21853529662175183 0.24593944248504612
    50 4 0.21441157685340795 0.2501036149800333
    50 5 0.21087010193116423 0.241268056979232
    50 6 0.20597112421598715 0.25089227301122674
    50 7 0.20136099546531616 0.2501056590202482
    50 8 0.19591053083354731 0.2530028809333039
    50 9 0.18924081817312932 0.25516020977403553
    50 10 0.18207264567168746 0.2542606668756661
    50 11 0.17459073885016296 0.26782975203626325
    50 12 0.16738560423430804 0.2688951184560758
    50 13 0.15991535505835017 0.27767353663756955
    

## 7. Consolidação e seleção


```python
frames=[]
for s in ["validacao","teste","avaliacao_2026"]:
 frames+=[metrica(y[s],pred_saz[s],"sazonal",s),metrica(y[s],inv(ridge.predict(X[s]).reshape(-1,24,2)),"ridge",s),metrica(y[s],pred_h[s],"hgb",s)]
 for sd in SEMENTES:frames.append(metrica(y[s],pred_g[sd][s],"gru",s,sd))
metricas=pd.concat(frames,ignore_index=True)
glob=(metricas.groupby(["modelo","semente","split","alvo","unidade"],dropna=False).agg(mae=("mae","mean"),rmse=("rmse","mean"),vies=("vies","mean")).reset_index())
# Média das sementes GRU e comparação em 2024.
gru_ag=(glob[glob.modelo=="gru"].groupby(["split","alvo","unidade"]).agg(mae_medio=("mae","mean"),mae_dp=("mae","std"),rmse_medio=("rmse","mean"),vies_medio=("vies","mean"),sementes=("semente","count")).reset_index())
val=glob[glob.split=="validacao"]
ref=val[val.modelo.isin(["ridge","hgb"])][["modelo","alvo","mae"]]
print("Métricas globais:");display(glob)
print("GRU agregada:");display(gru_ag)
print("Referências 2024:");display(ref)
```

    Métricas globais:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>modelo</th>
      <th>semente</th>
      <th>split</th>
      <th>alvo</th>
      <th>unidade</th>
      <th>mae</th>
      <th>rmse</th>
      <th>vies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>gru</td>
      <td>10.0</td>
      <td>avaliacao_2026</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.356313</td>
      <td>1.860504</td>
      <td>0.030939</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gru</td>
      <td>10.0</td>
      <td>avaliacao_2026</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.059658</td>
      <td>9.664707</td>
      <td>-0.874945</td>
    </tr>
    <tr>
      <th>2</th>
      <td>gru</td>
      <td>10.0</td>
      <td>teste</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.553386</td>
      <td>2.153435</td>
      <td>-0.011132</td>
    </tr>
    <tr>
      <th>3</th>
      <td>gru</td>
      <td>10.0</td>
      <td>teste</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.999602</td>
      <td>10.985608</td>
      <td>-0.246822</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gru</td>
      <td>10.0</td>
      <td>validacao</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.580485</td>
      <td>2.111374</td>
      <td>-0.236904</td>
    </tr>
    <tr>
      <th>5</th>
      <td>gru</td>
      <td>10.0</td>
      <td>validacao</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>8.089357</td>
      <td>10.691776</td>
      <td>0.841049</td>
    </tr>
    <tr>
      <th>6</th>
      <td>gru</td>
      <td>20.0</td>
      <td>avaliacao_2026</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.361636</td>
      <td>1.888621</td>
      <td>0.089293</td>
    </tr>
    <tr>
      <th>7</th>
      <td>gru</td>
      <td>20.0</td>
      <td>avaliacao_2026</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>6.978648</td>
      <td>9.691463</td>
      <td>-0.839818</td>
    </tr>
    <tr>
      <th>8</th>
      <td>gru</td>
      <td>20.0</td>
      <td>teste</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.556974</td>
      <td>2.190473</td>
      <td>0.188372</td>
    </tr>
    <tr>
      <th>9</th>
      <td>gru</td>
      <td>20.0</td>
      <td>teste</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.913790</td>
      <td>11.091674</td>
      <td>-0.840870</td>
    </tr>
    <tr>
      <th>10</th>
      <td>gru</td>
      <td>20.0</td>
      <td>validacao</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.551309</td>
      <td>2.104053</td>
      <td>-0.125792</td>
    </tr>
    <tr>
      <th>11</th>
      <td>gru</td>
      <td>20.0</td>
      <td>validacao</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.820318</td>
      <td>10.582238</td>
      <td>0.268805</td>
    </tr>
    <tr>
      <th>12</th>
      <td>gru</td>
      <td>30.0</td>
      <td>avaliacao_2026</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.363039</td>
      <td>1.877096</td>
      <td>0.032550</td>
    </tr>
    <tr>
      <th>13</th>
      <td>gru</td>
      <td>30.0</td>
      <td>avaliacao_2026</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>6.976006</td>
      <td>9.696543</td>
      <td>-1.547195</td>
    </tr>
    <tr>
      <th>14</th>
      <td>gru</td>
      <td>30.0</td>
      <td>teste</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.565898</td>
      <td>2.202149</td>
      <td>0.151684</td>
    </tr>
    <tr>
      <th>15</th>
      <td>gru</td>
      <td>30.0</td>
      <td>teste</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.902086</td>
      <td>11.186597</td>
      <td>-1.645772</td>
    </tr>
    <tr>
      <th>16</th>
      <td>gru</td>
      <td>30.0</td>
      <td>validacao</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.555145</td>
      <td>2.110730</td>
      <td>-0.111557</td>
    </tr>
    <tr>
      <th>17</th>
      <td>gru</td>
      <td>30.0</td>
      <td>validacao</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.668289</td>
      <td>10.526023</td>
      <td>-0.593391</td>
    </tr>
    <tr>
      <th>18</th>
      <td>gru</td>
      <td>40.0</td>
      <td>avaliacao_2026</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.404255</td>
      <td>1.906684</td>
      <td>-0.141466</td>
    </tr>
    <tr>
      <th>19</th>
      <td>gru</td>
      <td>40.0</td>
      <td>avaliacao_2026</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.129384</td>
      <td>9.821563</td>
      <td>-0.902120</td>
    </tr>
    <tr>
      <th>20</th>
      <td>gru</td>
      <td>40.0</td>
      <td>teste</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.574309</td>
      <td>2.178760</td>
      <td>-0.060378</td>
    </tr>
    <tr>
      <th>21</th>
      <td>gru</td>
      <td>40.0</td>
      <td>teste</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.935124</td>
      <td>11.149499</td>
      <td>-0.984342</td>
    </tr>
    <tr>
      <th>22</th>
      <td>gru</td>
      <td>40.0</td>
      <td>validacao</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.638871</td>
      <td>2.162596</td>
      <td>-0.344806</td>
    </tr>
    <tr>
      <th>23</th>
      <td>gru</td>
      <td>40.0</td>
      <td>validacao</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.805265</td>
      <td>10.615133</td>
      <td>0.041949</td>
    </tr>
    <tr>
      <th>24</th>
      <td>gru</td>
      <td>50.0</td>
      <td>avaliacao_2026</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.358938</td>
      <td>1.894117</td>
      <td>0.196191</td>
    </tr>
    <tr>
      <th>25</th>
      <td>gru</td>
      <td>50.0</td>
      <td>avaliacao_2026</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.028897</td>
      <td>9.730263</td>
      <td>-1.384726</td>
    </tr>
    <tr>
      <th>26</th>
      <td>gru</td>
      <td>50.0</td>
      <td>teste</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.548643</td>
      <td>2.198943</td>
      <td>0.312468</td>
    </tr>
    <tr>
      <th>27</th>
      <td>gru</td>
      <td>50.0</td>
      <td>teste</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.927192</td>
      <td>11.114846</td>
      <td>-1.317381</td>
    </tr>
    <tr>
      <th>28</th>
      <td>gru</td>
      <td>50.0</td>
      <td>validacao</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.521762</td>
      <td>2.104068</td>
      <td>0.078725</td>
    </tr>
    <tr>
      <th>29</th>
      <td>gru</td>
      <td>50.0</td>
      <td>validacao</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.747850</td>
      <td>10.551398</td>
      <td>-0.165781</td>
    </tr>
    <tr>
      <th>30</th>
      <td>hgb</td>
      <td>NaN</td>
      <td>avaliacao_2026</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.389632</td>
      <td>1.914787</td>
      <td>-0.003202</td>
    </tr>
    <tr>
      <th>31</th>
      <td>hgb</td>
      <td>NaN</td>
      <td>avaliacao_2026</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.079983</td>
      <td>9.784586</td>
      <td>-1.169025</td>
    </tr>
    <tr>
      <th>32</th>
      <td>hgb</td>
      <td>NaN</td>
      <td>teste</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.579619</td>
      <td>2.199491</td>
      <td>0.040733</td>
    </tr>
    <tr>
      <th>33</th>
      <td>hgb</td>
      <td>NaN</td>
      <td>teste</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.980622</td>
      <td>11.172543</td>
      <td>-0.738234</td>
    </tr>
    <tr>
      <th>34</th>
      <td>hgb</td>
      <td>NaN</td>
      <td>validacao</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.603409</td>
      <td>2.146750</td>
      <td>-0.267912</td>
    </tr>
    <tr>
      <th>35</th>
      <td>hgb</td>
      <td>NaN</td>
      <td>validacao</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.903512</td>
      <td>10.642803</td>
      <td>0.500278</td>
    </tr>
    <tr>
      <th>36</th>
      <td>ridge</td>
      <td>NaN</td>
      <td>avaliacao_2026</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.514856</td>
      <td>2.092944</td>
      <td>0.036787</td>
    </tr>
    <tr>
      <th>37</th>
      <td>ridge</td>
      <td>NaN</td>
      <td>avaliacao_2026</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.307822</td>
      <td>10.064575</td>
      <td>-1.140046</td>
    </tr>
    <tr>
      <th>38</th>
      <td>ridge</td>
      <td>NaN</td>
      <td>teste</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.700491</td>
      <td>2.346387</td>
      <td>0.038398</td>
    </tr>
    <tr>
      <th>39</th>
      <td>ridge</td>
      <td>NaN</td>
      <td>teste</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>8.251848</td>
      <td>11.366963</td>
      <td>-0.536772</td>
    </tr>
    <tr>
      <th>40</th>
      <td>ridge</td>
      <td>NaN</td>
      <td>validacao</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.696510</td>
      <td>2.273055</td>
      <td>-0.227446</td>
    </tr>
    <tr>
      <th>41</th>
      <td>ridge</td>
      <td>NaN</td>
      <td>validacao</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>8.258917</td>
      <td>11.099002</td>
      <td>0.346578</td>
    </tr>
    <tr>
      <th>42</th>
      <td>sazonal</td>
      <td>NaN</td>
      <td>avaliacao_2026</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.920754</td>
      <td>2.704801</td>
      <td>0.054422</td>
    </tr>
    <tr>
      <th>43</th>
      <td>sazonal</td>
      <td>NaN</td>
      <td>avaliacao_2026</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>9.586437</td>
      <td>13.479269</td>
      <td>0.085174</td>
    </tr>
    <tr>
      <th>44</th>
      <td>sazonal</td>
      <td>NaN</td>
      <td>teste</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>2.210932</td>
      <td>3.100812</td>
      <td>0.014382</td>
    </tr>
    <tr>
      <th>45</th>
      <td>sazonal</td>
      <td>NaN</td>
      <td>teste</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>11.175094</td>
      <td>15.777282</td>
      <td>-0.345159</td>
    </tr>
    <tr>
      <th>46</th>
      <td>sazonal</td>
      <td>NaN</td>
      <td>validacao</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>2.085084</td>
      <td>2.958638</td>
      <td>-0.011233</td>
    </tr>
    <tr>
      <th>47</th>
      <td>sazonal</td>
      <td>NaN</td>
      <td>validacao</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>10.837693</td>
      <td>15.206178</td>
      <td>-0.238505</td>
    </tr>
  </tbody>
</table>
</div>


    GRU agregada:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>split</th>
      <th>alvo</th>
      <th>unidade</th>
      <th>mae_medio</th>
      <th>mae_dp</th>
      <th>rmse_medio</th>
      <th>vies_medio</th>
      <th>sementes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>avaliacao_2026</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.368836</td>
      <td>0.019967</td>
      <td>1.885405</td>
      <td>0.041501</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>avaliacao_2026</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.034519</td>
      <td>0.063656</td>
      <td>9.720908</td>
      <td>-1.109761</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>teste</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.559842</td>
      <td>0.010263</td>
      <td>2.184752</td>
      <td>0.116203</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>teste</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.935559</td>
      <td>0.037967</td>
      <td>11.105645</td>
      <td>-1.007038</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>validacao</td>
      <td>temperatura_c</td>
      <td>°C</td>
      <td>1.569514</td>
      <td>0.044014</td>
      <td>2.118564</td>
      <td>-0.148067</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>validacao</td>
      <td>umidade_pct</td>
      <td>p.p.</td>
      <td>7.826216</td>
      <td>0.158744</td>
      <td>10.593313</td>
      <td>0.078526</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


    Referências 2024:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>modelo</th>
      <th>alvo</th>
      <th>mae</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>hgb</td>
      <td>temperatura_c</td>
      <td>1.603409</td>
    </tr>
    <tr>
      <th>35</th>
      <td>hgb</td>
      <td>umidade_pct</td>
      <td>7.903512</td>
    </tr>
    <tr>
      <th>40</th>
      <td>ridge</td>
      <td>temperatura_c</td>
      <td>1.696510</td>
    </tr>
    <tr>
      <th>41</th>
      <td>ridge</td>
      <td>umidade_pct</td>
      <td>8.258917</td>
    </tr>
  </tbody>
</table>
</div>


## 8. Regra final

O modelo de API será escolhido somente com 2024. Entre Ridge e HGB, vence o menor MAE médio padronizado. A GRU somente vence se sua média superar o melhor tabular nos dois alvos, pelo menos quatro sementes vencerem em cada alvo e o CV do MAE for inferior a 2%.


```python
# critérios tabulares
scales={"temperatura_c":scaler_y.scale_[0],"umidade_pct":scaler_y.scale_[1]}
def crit_tab(nome):
 d=val[val.modelo==nome];return np.mean([r.mae/scales[r.alvo] for _,r in d.iterrows()])
TABULAR="hgb" if crit_tab("hgb")<crit_tab("ridge") else "ridge"
ref_tab=val[val.modelo==TABULAR].set_index("alvo")
gv=glob[(glob.modelo=="gru")&(glob.split=="validacao")]
cond_media=all(gru_ag[(gru_ag.split=="validacao")&(gru_ag.alvo==a)].mae_medio.iloc[0]<ref_tab.loc[a,"mae"] for a in scales)
cond_4=all((gv[gv.alvo==a].mae<ref_tab.loc[a,"mae"]).sum()>=4 for a in scales)
cond_cv=all(100*gv[gv.alvo==a].mae.std()/gv[gv.alvo==a].mae.mean()<2 for a in scales)
MODELO_API="gru" if cond_media and cond_4 and cond_cv else TABULAR
print("Melhor tabular:",TABULAR,"| média GRU melhor nos dois:",cond_media,"| 4/5:",cond_4,"| CV<2%:",cond_cv)
print("MODELO TU RECOMENDADO PARA API:",MODELO_API)
```

    Melhor tabular: hgb | média GRU melhor nos dois: True | 4/5: True | CV<2%: False
    MODELO TU RECOMENDADO PARA API: hgb
    

## 9. Salvamento


```python
pd.DataFrame(log_hgb).to_csv(DIRETORIO_SAIDA/"log_hgb_TU.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(hist).to_csv(DIRETORIO_SAIDA/"historico_gru_TU.csv",index=False,encoding="utf-8-sig")
metricas.to_csv(DIRETORIO_SAIDA/"metricas_por_horizonte_TU.csv",index=False,encoding="utf-8-sig")
glob.to_csv(DIRETORIO_SAIDA/"metricas_globais_TU.csv",index=False,encoding="utf-8-sig")
gru_ag.to_csv(DIRETORIO_SAIDA/"metricas_agregadas_gru_TU.csv",index=False,encoding="utf-8-sig")
sel.to_csv(DIRETORIO_SAIDA/"selecao_alpha_ridge_TU.csv",index=False,encoding="utf-8-sig")
joblib.dump({"modelos":modelos_hgb,"configuracao":"TU","hiperparametros":HGB_B,"metadados_preparacao":meta},DIRETORIO_SAIDA/"pacote_hgb_TU_48_api.joblib",compress=3)
with (DIRETORIO_SAIDA/"decisao_modelo_TU.json").open("w",encoding="utf-8") as f:json.dump({"modelo_recomendado":MODELO_API,"melhor_tabular":TABULAR,"atributos":"TU","usa_covar_meteorologica_futura":False},f,ensure_ascii=False,indent=2)
assert np.isfinite(metricas[["mae","rmse","vies"]].to_numpy()).all()
print("VERIFICAÇÃO FINAL: OK")
print("MODELO TU RECOMENDADO PARA API:",MODELO_API)
```

    VERIFICAÇÃO FINAL: OK
    MODELO TU RECOMENDADO PARA API: hgb
    

## Encerramento

Após a execução, enviar `metricas_globais_TU.csv`, `metricas_agregadas_gru_TU.csv` e `decisao_modelo_TU.json`. Esses resultados serão usados para redigir as novas seções de Metodologia e Resultados do Relatório Final, preservando integralmente o experimento univariado apresentado no Relatório Parcial.
