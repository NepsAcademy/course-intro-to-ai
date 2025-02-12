import joblib  # Salvamento e carregamento do modelo
import matplotlib.pyplot as plt  # Plotagem de gráficos
import pandas as pd  # Manipulação de dados
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)  # Divisão dos dados em treino e teste
from sklearn.preprocessing import LabelEncoder  # Codificação de variáveis categóricas
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
)  # Modelo e visualização da árvore

# Carrega os dados do arquivo CSV do Titanic
df = pd.read_csv("titanic.csv")

# Mostra as primeiras e últimas linhas para inspeção inicial
print(df.head(10))
print(df.tail(7))

# Exibe informações gerais sobre o DataFrame (como tipos de dados e valores nulos)
print(df.info())

# Exibe estatísticas descritivas para colunas numéricas
print(df.describe())

# Substitui valores "?" por NaN para identificar valores ausentes
df = df.replace("?", pd.NA)

# Converte colunas numéricas (age e fare) para números, tratando erros como NaN
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["fare"] = pd.to_numeric(df["fare"], errors="coerce")

# Preenche valores ausentes em colunas numéricas com a média da coluna
numerical_features = ["age", "fare"]
for column in numerical_features:
    if column in df.columns:
        df[column] = df[column].fillna(df[column].mean())

# Preenche valores ausentes em colunas categóricas com o valor mais frequente (moda)
categorical_features = ["sex", "embarked"]
for column in categorical_features:
    if column in df.columns:
        df[column] = df[column].fillna(df[column].mode()[0])

# Mostra os dados tratados
print(df)

# Calcula a média de idade e tarifa agrupadas por classe e sexo
stats_by_class_sex = df.groupby(["pclass", "sex"]).agg({"age": "mean", "fare": "mean"})
print(stats_by_class_sex)

# Define as colunas de entrada (features) e a coluna alvo (target)
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target = "survived"

# Cria os DataFrames de entrada (X) e saída (y)
X = df[features]
y = df[target]

print("DataFrame com as features")
print(X)
print("DataFrame com a variável alvo")
print(y)

# Divide os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preenche valores ausentes em variáveis categóricas nos conjuntos de treino e teste
for column in categorical_features:
    if column in X_train.columns:
        X_train[column] = X_train[column].fillna(X_train[column].mode()[0])
        X_test[column] = X_test[column].fillna(X_test[column].mode()[0])

# Codifica variáveis categóricas em formato numérico
le = LabelEncoder()
for column in categorical_features:
    if column in X_train.columns:
        X_train[column] = le.fit_transform(X_train[column])
        X_test[column] = le.transform(X_test[column])

# Define o modelo de árvore de decisão
dt_classifier = DecisionTreeClassifier(
    criterion="entropy", splitter="random", max_depth=3, random_state=42
)

# Treina o modelo com os dados de treinamento
dt_classifier.fit(X_train, y_train)

# Exemplo de entrada para previsão
test_data = pd.DataFrame(
    {
        "pclass": [2],
        "sex": ["male"],
        "age": [30],
        "sibsp": [0],
        "parch": [0],
        "fare": [12.5],
        "embarked": ["C"],
    }
)

# Codifica os dados de entrada no mesmo formato do treinamento
for column in categorical_features:
    if column in test_data.columns:
        if column == "sex":
            le.classes_ = pd.Index(["female", "male"])
        elif column == "embarked":
            le.classes_ = pd.Index(["C", "Q", "S"])
        test_data[column] = le.transform(test_data[column])

# Faz a previsão
prediction = dt_classifier.predict(test_data)
print("Previsão:", prediction)

# Decodifica o resultado
resultado = "Survived" if prediction[0] == 1 else "Not Survived"
print("Resultado interpretado:", resultado)

# Salva o modelo treinado em um arquivo
dt_model_filename = "decision_tree_model.pkl"
joblib.dump(dt_classifier, dt_model_filename)
print(f"Modelo salvo em {dt_model_filename}")

# Configura o tamanho da figura para a visualização
# plt.figure(figsize=(20, 10))

# Plota a árvore de decisão
# plot_tree(
#     dt_classifier,
#     feature_names=features,
#     class_names=["Not Survived", "Survived"],
#     filled=True,
# )

# Exibe a árvore de decisão.
# plt.show()


# Supondo que y_test contenha os rótulos verdadeiros e y_pred as previsões do modelo
y_pred = dt_classifier.predict(X_test)

# Cálculo da acurácia
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)

# Geração do relatório de classificação com precisão, recall e f1-score
report = classification_report(y_test, y_pred)
print("Relatório de Classificação:\n", report)

# Definição dos hiperparâmetros a serem testados para a Decision Tree
param_distributions = {
    "max_depth": [None, 1, 2, 3, 5, 10, 15],
    "min_samples_split": [2, 3, 5, 10],
    "min_samples_leaf": [1, 2, 4, 6],
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
}

# Configuração do RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=dt_classifier,
    param_distributions=param_distributions,
    n_iter=50,  # Número de combinações aleatórias a serem testadas
    scoring="accuracy",  # Métrica a ser otimizada
    cv=5,  # Validação cruzada com 5 folds
    verbose=2,
    random_state=42,
    n_jobs=-1,  # Utiliza todos os núcleos disponíveis
)

# Ajuste do RandomizedSearchCV aos dados de treinamento
random_search.fit(X_train, y_train)

# Exibição dos melhores parâmetros e da melhor acurácia obtida na validação
print("Melhores Parâmetros:", random_search.best_params_)
print("Melhor Acurácia:", random_search.best_score_)

# Seleciona o melhor classificador encontrado durante a busca
best_dt = random_search.best_estimator_

# Faz as previsões com o melhor classificador usando o conjunto de teste
y_pred = best_dt.predict(X_test)

# Recalcula a acurácia com base nos dados de teste
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)

# Gera o relatório de classificação com precisão, recall e f1-score
report = classification_report(y_test, y_pred)
print("Relatório de Classificação:\n", report)
