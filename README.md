# 🚢 Predição de Sobrevivência no Titanic

Este projeto aplica conceitos práticos de Machine Learning para prever a sobrevivência de passageiros no naufrágio do Titanic. O desenvolvimento e a estruturação do código foram baseados nas práticas ensinadas no livro **"Machine Learning: Guia de Referência Rápida" (Matt Harrison)**, abordando desde a análise exploratória até a validação do modelo.

## 🗂️ Sobre os Dados
O conjunto de dados utilizado é o clássico `train.csv` do Titanic. Ele contém informações demográficas e da viagem dos passageiros, como:
- **Classe** (`Pclass`)
- **Gênero** (`Sex`)
- **Idade** (`Age`)
- **Tarifa** (`Fare`)
- **Familiares a bordo** (`SibSp`, `Parch`)
- **Variável Alvo**: `Survived` (0 = Não sobreviveu, 1 = Sobreviveu)

## 🛠️ Pré-processamento e Refatoração (Pipeline)
Seguindo as boas práticas para evitar *Data Leakage* (vazamento de dados), as etapas de transformação foram aplicadas após a divisão de treino e teste (70/30):
1. **Limpeza**: Remoção de colunas com alta cardinalidade ou sem poder preditivo direto (`Name`, `Ticket`, `Cabin`, `PassengerId`).
2. **One-Hot Encoding**: Criação de variáveis *dummy* para `Sex` e `Embarked`, com a exclusão da primeira categoria (`drop_first=True`) para evitar multicolinearidade.
3. **Imputação de Dados Ausentes**: Utilização do `SimpleImputer` (estratégia da mediana) para preencher valores nulos (ex: `Age`), treinado apenas na amostra de treino.
4. **Normalização**: Aplicação do `StandardScaler` para padronizar as escalas das variáveis numéricas.
5. **Boas Práticas de Código**: Refatoração das etapas de limpeza e separação em funções modulares (`tweak_titanic` e `get_train_test_X_y`) para facilitar a reutilização e testes futuros.

## 🤖 Avaliação de Modelos e Seleção
Inicialmente, um modelo de *baseline* (`DummyClassifier` estratificado) foi criado, obtendo ~49.6% de acurácia. Em seguida, diversos algoritmos foram avaliados utilizando **Cross-Validation (10 folds)** com a métrica ROC AUC:

* DummyClassifier (AUC: 0.500)
* LogisticRegression (AUC: 0.848)
* DecisionTreeClassifier (AUC: 0.775)
* KNeighborsClassifier (AUC: 0.838)
* GaussianNB (AUC: 0.827)
* SVC (AUC: 0.844)
* **RandomForestClassifier (AUC: 0.859)**
* XGBClassifier (AUC: 0.862)

Um modelo **StackingClassifier** (combinando os modelos acima com uma Regressão Logística no final) também foi testado, alcançando um AUC de **0.870**. 

Para equilibrar desempenho e interpretabilidade, o **Random Forest** foi selecionado para otimização.

## 📈 Otimização (Hyperparameter Tuning)
O modelo Random Forest foi otimizado utilizando o `GridSearchCV`, resultando nos seguintes hiperparâmetros ideais:
* `n_estimators`: 200
* `max_features`: 'sqrt'
* `min_samples_leaf`: 1

**Métricas Finais no Conjunto de Teste:**
* **Acurácia**: 78,35%
* **Precisão**: 75,23%

### 🔑 Importância das Variáveis (Feature Importances)
O modelo revelou que os fatores mais decisivos para a previsão foram:
1. `Fare` (Tarifa) - 27.1%
2. `Sex_male` (Gênero) - 25.6%
3. `Age` (Idade) - 25.5%
4. `Pclass` (Classe) - 8.6%

## 📊 Visualização de Modelos e Dados
O projeto faz uso extenso de visualizações para validar o comportamento dos dados e do modelo:
* **Curva ROC**: Visualização da taxa de verdadeiros/falsos positivos do Random Forest.
* **Curva de Aprendizado (Learning Curve)**: Avaliação de *overfitting*/*underfitting* conforme o volume de dados de treino aumenta.
* **Matriz de Confusão**: Análise detalhada dos acertos e erros de classificação.
* **Missingno**: Matrizes, gráficos de barras e heatmaps para analisar os padrões de dados ausentes (`Age`, `Cabin`, `Embarked`), conforme as técnicas discutidas no Capítulo 4 do livro base.

## 💻 Tecnologias
* `Python` (Jupyter Notebook)
* `Pandas` e `NumPy`
* `Scikit-Learn` e `XGBoost`
* `Matplotlib` e `Missingno`
