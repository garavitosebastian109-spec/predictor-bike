import pandas as pd

df = pd.read_csv("dataset.csv")

# Crear la columna target (igual que en el entrenamiento)
mediana = df["cnt"].median()
df["target"] = (df["cnt"] > mediana).astype(int)

# Quedarse solo con las columnas necesarias
columnas = ["season","yr","mnth","holiday","weekday","workingday",
            "weathersit","temp","atemp","hum","windspeed","target"]

df[columnas].to_csv("lotes_test.csv", index=False)
print("✓ Archivo lotes_test.csv generado")