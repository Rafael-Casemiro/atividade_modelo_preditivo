# Atividade de Modelo Preditivo

Este projeto consiste num exercício prático de *Machine Learning* que simula um cenário de marketing digital. O objetivo é criar um conjunto de dados sintético (fictício) e treinar um modelo de classificação (Árvore de Decisão) para prever se um utilizador irá realizar uma compra com base no seu comportamento de interação com e-mails.

## 📋 Sobre o Projeto

O projeto está dividido em duas etapas principais:
1.  **Geração de Dados:** Criação de um *DataFrame* com 300 registos simulados, onde a probabilidade de compra é influenciada por variáveis como a idade, o tempo de cadastro e a interação com a campanha de e-mail.
2.  **Modelação Preditiva:** Utilização da biblioteca `scikit-learn` para treinar um classificador `DecisionTreeClassifier` que aprende os padrões dos dados gerados para prever a variável alvo (`comprou`).

## 🛠️ Tecnologias Utilizadas

* **Python 3**
* **Pandas:** Manipulação e análise de dados.
* **NumPy:** Geração de números aleatórios e operações matemáticas.
* **Scikit-learn:** Criação do modelo de *Machine Learning*, divisão de dados e métricas de avaliação.

## 📊 Dicionário de Dados

Os dados são gerados aleatoriamente e contêm as seguintes variáveis:

* `idade`: Idade do utilizador (gerada entre 18 e 70 ou 100 anos, dependendo do script).
* `tempo_cadastro_dias`: Há quantos dias o utilizador está registado (1 a 2000 dias).
* `email_aberto`: Se o utilizador abriu o e-mail de marketing (0 = Não, 1 = Sim).
* `clicou_no_link`: Se o utilizador clicou no link do e-mail.
    * *Nota:* A probabilidade de clicar é maior (50%) se o e-mail for aberto, comparado com quem não abriu (10%).
* `comprou`: Variável alvo (Target). Indica se a compra foi efetuada.
    * *Lógica:* A probabilidade de compra é calculada com base na abertura do e-mail, clique no link e tempo de cadastro.

## 🚀 Como Executar

### Pré-requisitos

Certifique-se de que tem o Python instalado e as bibliotecas necessárias:

```bash
pip install pandas numpy scikit-learn jupyter
