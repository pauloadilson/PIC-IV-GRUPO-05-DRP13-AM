# Contexto para a etapa de implementação da API e coleta meteorológica

## 1. Objetivo da nova etapa

Implementar a infraestrutura de coleta e persistência das condições meteorológicas atuais fornecidas pela API HG Brasil para Álvares Machado/SP. Nesta primeira fase, o objetivo é formar uma base horária própria, consultar as últimas 24 e 72 horas e validar a continuidade temporal. A integração do modelo preditivo será feita depois, em contexto complementar, quando o modelo final baseado no conjunto TU estiver definido.

## 2. Localidade e limites da fonte

- Cidade: Álvares Machado, SP
- WOEID: 458410
- Plano informado: gratuito
- Limite informado: 5 cidades e 400 requisições por dia
- Chave de acesso: armazenada exclusivamente em variável de ambiente
- Dados operacionais previstos para esta fase: condições meteorológicas atuais, especialmente temperatura e umidade relativa
- Dados históricos da HG Brasil: não contratados

A aplicação deverá construir seu próprio histórico por meio da coleta recorrente das condições atuais. Os registros antigos não devem ser automaticamente excluídos. A operação consultará recortes recentes, mas a retenção integral permitirá auditoria, monitoramento e avaliação futura das previsões.

## 3. Branch de desenvolvimento

Criar uma branch específica:

```bash
git switch -c feat/implementacao-api
```

Caso o repositório utilize nomes simples em português, `implementacao_api` também é aceitável, mas o nome recomendado é `feat/implementacao-api`.

## 4. Sequência de implementação desta fase

### 4.1 Configurar SQLite e a tabela de observações

Utilizar SQLite na primeira versão, por ser suficiente para uma única localidade, coleta horária e pequeno volume de dados.

Tabela principal sugerida:

```sql
CREATE TABLE observacoes_meteorologicas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    localidade_id TEXT NOT NULL,
    cidade TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    timezone TEXT NOT NULL,
    instante_fonte TEXT NOT NULL,
    instante_coleta_utc TEXT NOT NULL,
    hora_referencia TEXT NOT NULL,
    temperatura_c REAL NOT NULL,
    umidade_pct REAL NOT NULL,
    resposta_original_json TEXT NOT NULL,
    origem TEXT NOT NULL DEFAULT 'hg_brasil',
    valido_modelo INTEGER NOT NULL DEFAULT 1,
    motivo_invalidacao TEXT,
    criado_em TEXT NOT NULL,
    UNIQUE (localidade_id, hora_referencia)
);
```

Nesta fase, não incluir ponto de orvalho como campo obrigatório. O conjunto operacional previsto é TU: temperatura e umidade.

### 4.2 Implementar cliente da HG Brasil

Responsabilidades:

- ler a chave de variável de ambiente;
- consultar a localidade pelo WOEID 458410;
- aplicar timeout;
- tratar falhas HTTP e respostas inválidas;
- validar a cidade retornada;
- extrair data, hora, temperatura, umidade, timezone, latitude e longitude quando disponíveis;
- devolver estrutura normalizada ao serviço de coleta;
- nunca registrar a chave em logs.

Variáveis de ambiente sugeridas:

```env
HG_BRASIL_KEY=
HG_BRASIL_WOEID=458410
HG_BRASIL_CITY=Alvares Machado, SP
APP_TIMEZONE=America/Sao_Paulo
DATABASE_URL=sqlite:///./data/meteorologia.db
```

### 4.3 Gravar resposta bruta e campos normalizados

Para cada resposta válida:

- preservar o JSON original;
- registrar o instante informado pela fonte;
- registrar separadamente o instante UTC da coleta;
- construir `hora_referencia` truncada para a hora da fonte;
- normalizar temperatura e umidade para tipos numéricos;
- guardar cidade, timezone e coordenadas retornadas;
- indicar se o registro está válido para uso futuro pelo modelo.

### 4.4 Implementar deduplicação por localidade e hora

A chave lógica será:

```text
localidade_id + hora_referencia
```

Chamadas repetidas dentro da mesma hora não devem criar novas observações. Implementar inserção ou atualização idempotente. O horário do agendador não deve ser usado como substituto do horário informado pela fonte.

### 4.5 Implementar consulta das últimas 24 e 72 horas

Criar operações de repositório para:

- obter as últimas 24 observações válidas;
- obter as últimas 72 observações válidas;
- retornar os registros em ordem cronológica crescente;
- informar quantidade de observações encontradas;
- informar o primeiro e o último horário do intervalo.

A consulta de 24 horas será usada futuramente na inferência. A consulta de 72 horas servirá para exibição, diagnóstico e tolerância operacional.

### 4.6 Implementar verificação de continuidade

Não basta possuir 24 linhas. Elas devem representar 24 horas consecutivas.

A validação deverá detectar:

- horas ausentes;
- duplicidades;
- registros fora de ordem;
- intervalo maior ou menor que uma hora;
- campos nulos ou fora das faixas esperadas.

Status sugeridos:

```text
INSUFICIENTE: menos de 24 horas válidas
LACUNA: existem 24 registros, mas não consecutivos
PRONTO: 24 horas consecutivas válidas
```

A primeira previsão só poderá ser executada no estado `PRONTO`.

### 4.7 Começar a coleta real

Após concluir cliente, banco, deduplicação e continuidade:

- iniciar a coleta recorrente;
- monitorar erros e atrasos;
- acompanhar se a fonte atualiza exatamente a cada hora;
- confirmar cidade, timezone e coordenadas;
- manter o histórico integral;
- atingir no mínimo 24 horas consecutivas antes da primeira inferência;
- acumular 72 horas para estabilização operacional e visualização.

## 5. Frequência de coleta

O limite de 400 requisições por dia corresponde matematicamente a aproximadamente 16,67 requisições por hora, se fosse distribuído uniformemente. Isso não significa que seja recomendável consultar nessa frequência.

Para uma fonte atualizada aproximadamente de hora em hora, iniciar com uma requisição por hora, preferencialmente alguns minutos após a virada. Se for necessário tolerar atrasos, poderá ser adotada coleta a cada 15 ou 20 minutos, desde que confirmada a compatibilidade com os limites e termos do plano. A deduplicação impedirá múltiplos registros da mesma hora.

## 6. Estrutura sugerida do código

```text
app/
├── coleta/
│   └── hg_brasil.py
├── banco/
│   ├── conexao.py
│   └── modelos.py
├── repositorios/
│   └── observacoes.py
├── servicos/
│   ├── coleta.py
│   └── continuidade.py
├── api/
│   └── rotas_coleta.py
└── config.py

tests/
├── test_cliente_hg_brasil.py
├── test_deduplicacao.py
├── test_consulta_janelas.py
└── test_continuidade.py
```

## 7. Escopo desta conversa nova

Nesta primeira conversa de API, implementar somente:

1. branch de desenvolvimento;
2. configuração e variáveis de ambiente;
3. banco SQLite;
4. cliente da HG Brasil;
5. persistência da resposta bruta e normalizada;
6. deduplicação;
7. consulta de 24 e 72 horas;
8. continuidade temporal;
9. início da coleta real.

Não incluir ainda:

- modelo de aprendizagem de máquina;
- endpoint de previsão experimental;
- regras agronômicas de alertas;
- integração com o frontend;
- ponto de orvalho derivado;
- sensores locais;
- dados de microclima de estufa.

## 8. Decisões já consolidadas

- As informações representam condições meteorológicas externas de referência para o viveiro.
- Não serão instalados sensores locais.
- Não haverá controle individual por estufa.
- A ausência de dados internos será registrada como limitação.
- A fonte operacional será HG Brasil.
- O histórico recente será construído pela própria aplicação.
- A entrada operacional prevista para o modelo será TU: temperatura e umidade.
- O modelo final será informado em contexto posterior.
- A inferência será executada em CPU.
- Os alertas não afirmarão ocorrência de fungos, apodrecimento ou outras condições não observadas.

## 9. Critérios de conclusão desta fase

A fase estará concluída quando:

- a chave estiver fora do código-fonte;
- a consulta ao WOEID 458410 estiver funcional;
- a resposta bruta estiver preservada;
- temperatura e umidade estiverem normalizadas;
- chamadas repetidas não gerarem duplicidades;
- consultas de 24 e 72 horas funcionarem;
- a continuidade horária for validada automaticamente;
- o status informar quando a série estiver pronta para previsão;
- a coleta real estiver em execução;
- os testes principais estiverem aprovados.

## 10. Commit inicial sugerido

```text
feat(coleta): adiciona persistência horária da API HG Brasil
```

Corpo opcional:

```text
feat(coleta): adiciona persistência horária da API HG Brasil

- configura coleta para Álvares Machado pelo WOEID 458410
- lê chave de acesso por variável de ambiente
- persiste observações em SQLite
- armazena resposta original e campos normalizados
- evita duplicidade por localidade e hora
- consulta janelas recentes de 24 e 72 horas
- valida continuidade temporal das observações
```
