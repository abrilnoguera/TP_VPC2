from scipy.stats import skew, kurtosis
import pandas as pd

def calcular_metricas_avanzadas(base_dir, split="train"):
    """
    Calcula métricas de intensidad por clase y devuelve un DataFrame
    """
    split_dir = os.path.join(base_dir, split)
    resultados = []

    for clase in os.listdir(split_dir):
        clase_dir = os.path.join(split_dir, clase)
        if not os.path.isdir(clase_dir):
            continue

        for img_file in os.listdir(clase_dir):
            img_path = os.path.join(clase_dir, img_file)
            try:
                img = Image.open(img_path).convert("L")
                arr = np.array(img, dtype=np.float32).ravel()

                resultados.append({
                    "clase": clase,
                    "mean": np.mean(arr),
                    "median": np.median(arr),
                    "std": np.std(arr),
                    "skewness": skew(arr),
                    "kurtosis": kurtosis(arr),
                    "p25": np.percentile(arr, 25),
                    "p75": np.percentile(arr, 75),
                    "iqr": np.percentile(arr, 75) - np.percentile(arr, 25)
                })
            except:
                pass

    return pd.DataFrame(resultados)

# Uso
df_metricas = calcular_metricas_avanzadas(data_dir, split="train")

# Promedios por clase
df_resumen = df_metricas.groupby("clase").mean()
print(df_resumen)





### Dimensión de Imágenes y Relación de Aspecto
# Visualización con boxplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
metricas_a_graficar = ['mean', 'std', 'skewness', 'kurtosis']

for i, metrica in enumerate(metricas_a_graficar):
    ax = axes[i//2, i%2]
    sns.boxplot(data=df_metricas, x='clase', y=metrica, ax=ax, palette='Set2')
    ax.set_title(f'Distribución de {metrica} por clase')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Boxplot adicional para percentiles
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(data=df_metricas, x='clase', y='median', ax=axes[0], palette='Set2')
axes[0].set_title('Mediana de intensidad por clase')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

sns.boxplot(data=df_metricas, x='clase', y='iqr', ax=axes[1], palette='Set2')
axes[1].set_title('Rango intercuartílico (IQR) por clase')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
