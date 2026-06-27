import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Baixando dados da B3 e treinando o modelo...")

dados = yf.download('PETR4.SA', start='2024-01-01', end='2026-06-25', progress=False)

delta = dados['Close'].diff()
ganho = (delta.where(delta > 0, 0)).rolling(window=14).mean()
perda = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = ganho / perda
dados['IFR'] = 100 - (100 / (1 + rs))

dados['BB_Media'] = dados['Close'].rolling(window=20).mean()
dados['BB_Desvio'] = dados['Close'].rolling(window=20).std()
dados['BB_Superior'] = dados['BB_Media'] + (2 * dados['BB_Desvio'])
dados['BB_Inferior'] = dados['BB_Media'] - (2 * dados['BB_Desvio'])

dados['Retorno'] = dados['Close'].pct_change()
dados['Media_5dias'] = dados['Close'].rolling(window=5).mean()
dados['Alvo'] = (dados['Close'].shift(-1) > dados['Close']).astype(int)
dados = dados.dropna()

recursos = ['Close', 'Volume', 'Retorno', 'Media_5dias', 'IFR', 'BB_Inferior', 'BB_Media', 'BB_Superior']
X = dados[recursos]
y = dados['Alvo']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

modelo = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_split=10, class_weight='balanced', random_state=42)
modelo.fit(X_treino, y_treino)

with open('modelo_petr4.pkl', 'wb') as arquivo:
    pickle.dump(modelo, arquivo)
print("Cérebro da IA salvo com sucesso no arquivo: modelo_petr4.pkl")

previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)
print(f"Acuracia nos testes: {acuracia * 100:.2f}%")

dados_recentes = X.tail(1)
previsao_amanha = modelo.predict(dados_recentes)
probabilidades = modelo.predict_proba(dados_recentes)

print("PREVISAO PARA O PROXIMO DIA UTIL")
if previsao_amanha == 1:
    print(f"TENDENCIA: ALTA")
else:
    print(f"TENDENCIA: BAIXA OU NEUTRA")

plt.figure(figsize=(10, 5))
plt.plot(dados.index, dados['Close'], label='Preco Fechamento (PETR4)', color='blue')
plt.plot(dados.index, dados['BB_Media'], label='Media de 20d', color='orange')
plt.plot(dados.index, dados['BB_Superior'], color='red', alpha=0.3, label='Banda Superior')
plt.plot(dados.index, dados['BB_Inferior'], color='green', alpha=0.3, label='Banda Inferior')
plt.title('PETR4 - Random Forest Otimizado para Tendencias')
plt.legend()
plt.grid(True)
plt.show()
