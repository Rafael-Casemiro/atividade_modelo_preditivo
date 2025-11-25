# Importação das Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Gerando o DataFrame
np.random.seed(42)
n = 300 # número de registros no DataFrame

idade = np.random.randint(18, 70, n) # Gerando Idades aleatórias entre 18 e 70 anos
tempo_cadastro = np.random.randint(1, 2000, n) # Gerando tempo de cadastro entre 1 a 2000 dias

# Cada pessoa tem 60% de chance de abrir o email
email_aberto = np.random.binomial(1, 0.6, n)

# 'clicou_no_link' depende de 'email_aberto':
# Se a pessoa abriu o e-mail → chance de clicar = 50%
# Se não abriu → chance de clicar = 10%
clicou_no_link = np.where(email_aberto == 1,
                          np.random.binomial(1, 0.5, n),
                          np.random.binomial(1, 0.1, n))

# Criando a probabilidade de compra dependendo de 'email_aberto', 'clicou_no_link' e 'tempo_cadastro'
prob_compra = (
    0.1 +
    0.25 * email_aberto +
    0.45 * clicou_no_link +
    0.2 * (tempo_cadastro / tempo_cadastro.max())
)

# Garante que a probabilidade fique entre 0 e 1
prob_compra = prob_compra.clip(0, 1)

comprou = np.random.binomial(1, prob_compra) # Comprou é gerado com base na probabilidade calculada acima

dados = pd.DataFrame({
    "idade": idade,
    "tempo_cadastro_dias": tempo_cadastro,
    "email_aberto": email_aberto,
    "clicou_no_link": clicou_no_link,
    "comprou": comprou
})

dados

# X = variáveis independentes (todas as colunas menos 'comprou')
# y = variável dependente (o que queremos prever)
X = dados.drop('comprou', axis=1)
y = dados['comprou']

# separando os dados em conjunto de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier(max_depth=5, random_state=60) # escolhendo o modelo  Árvore de Decisão
modelo.fit(X_treino, y_treino) # treinamento do modelo utilizando o método fit


previsoes = modelo.predict(X_teste) # utilizando o método predict para prever o comportamento no conjunto de teste

acuracia = accuracy_score(y_teste, previsoes) # calculando a acurácia do modelo
print(acuracia)