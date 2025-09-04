# Cargar los datos 
# Usar un dataset (ejemplo: iris, tips de Seaborn, o un CSV ).
# Importar librerias de sklearn y pandas

from sklearn.datasets import load_iris
import pandas as pd 


print("\n1. Cargar los datos")
# Cargar el dataset Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Agregar la columna target al DataFrame
df['target'] = iris.target
print(iris.target_names)

# Enumerar nombres de las especies, para comprender mejor el dataset
df['Especies'] = iris.target
df['Especies'] = df['Especies'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Exploración inicial del dataset
# Mostrar las primeras filas del dataset.
print("\n2. Exploracion inicial")
print("\nPrimeras filas del dataset:")
print(df.head())

# Ver cuántas filas y columnas tiene.
print("\nDimensiones del dataset:")
print(df.shape)

# Identificar valores faltantes.
print("\nValores faltantes en cada columna:")
print(df.isnull().sum())

#--------------------------------------------------------

# Limpieza de datos
#Insertare algunos nulos en el dataset para el ejercicio
import numpy as np


df.loc[5:7, 'sepal length (cm)'] = np.nan
df.at[2, 'sepal width (cm)'] = np.nan
df.at[10, 'petal length (cm)'] = np.nan
df.loc[18:20, "Especies"] = np.nan

#--------------------------------------------------------

# Identificar valores faltantes.
print("\n3.Limpieza de datos")
print("\nSe agrego valores nulos:")
print("Nuevos valores faltantes en cada columna):")
print(df.isnull().sum())

#--------------------------------------------------------

# Eliminar valores nulos.
df.dropna(inplace=True)
print("\nDimensiones del dataset despues de eliminar nulos:")
print(df.shape)

#df = df.dropna()
#print("\nDimensiones del dataset después de eliminar nulos:")
#print(df.shape)

#--------------------------------------------------------

# Reemplazar valores nulos.
df['sepal length (cm)'] = df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean()) #Promedio
df['sepal width (cm)']  = df['sepal width (cm)'].fillna(df['sepal width (cm)'].mean()) #Promedio
df['petal length (cm)'] = df['petal length (cm)'].fillna(df['petal length (cm)'].mean()) #Promedio
df['Especies'] = df['Especies'].fillna(df['Especies'].mode()[0]) #Moda 

print("\nValores faltantes despues de la limpieza:")
print(df.isnull().sum())    
print("\nDimensiones del dataset despues de la limpieza:")
print(df.shape)

#--------------------------------------------------------

# Normalizar o renombrar columnas si es necesario. Renombrare las columnas en español 
df.rename(columns = {
    'sepal length (cm)': 'Largo_sepalo',
    'sepal width (cm)': 'Ancho_sepalo',
    'petal length (cm)': 'Largo_petalo',
    'petal width (cm)': 'Ancho_petalo',
    "Especies": "Especies"
}, inplace = True)
print("\nNombres de las columnas despues de la normalizacion:")
print(df.columns)
print("\nColumnas del Dataset actualizados:")
print(df.head())

#--------------------------------------------------------

# Filtros y selección
# Filtrar filas según una condición 
# Filtrare las flores que son de la especie 'setosa'
print("\n4. Filtros y seleccion")
print("\nFiltro de las filas Especies que son setosa:")
print(df[df['Especies'] == 'setosa'])
# Filtrare las flores que tienen el largo del petalo mayor a 1.5 cm
print("\nFiltro de las filas segun el Largo del petalo:")
print(df[df['Largo_petalo'] > 1.5])

#--------------------------------------------------------

# Seleccionar solo algunas columnas
# Seleccionare las columnas de Largo_petalo y Especies
print("\nFiltro de las columnas Largo_petalo y Especies:")
print(df[['Largo_petalo', 'Especies']])

#--------------------------------------------------------

# Creación de nuevas columnas
# Crear una nueva columna que sea el área del petalo (largo * ancho)
print("\n5. Creacion de nuevas columnas") 
df['Area_petalo'] = df['Largo_petalo'] * df['Ancho_petalo']
print("\nDataFrame con la nueva columna Area_petalo:") 
print(df.head())   

#--------------------------------------------------------

# Estadísticas descriptivas - Usare Pandas
# Promedio de las columnas numericas
print("\n6. Estadisticas descriptivas")
print("\nPromedio de las columnas numericas :")
print("\nPromedio de Largo de petalo:")
print(df['Largo_petalo'].mean())   

print("\nPromedio de Ancho de petalo:")
print(df['Ancho_petalo'].mean())   

print("\nPromedio de Largo de sepalo:")
print(df['Largo_sepalo'].mean()) 
   
print("\nPromedio de Ancho de sepalo:")
print(df['Ancho_sepalo'].mean())   

#--------------------------------------------------------

print("\nMaximo de las columnas numericas :")
print("\nMaximo de Largo de petalo:")
print(df['Largo_petalo'].max())

print("\nMaximo de Ancho de petalo:")
print(df['Ancho_petalo'].max()) 

print("\nMaximo de Largo de sepalo:")
print(df['Largo_sepalo'].max())

print("\nMaximo de Ancho de sepalo:")
print(df['Ancho_sepalo'].max())

#--------------------------------------------------------

print("\nMinimo de las columnas numericas :")
print("\nMinimo de Largo de petalo:")
print(df['Largo_petalo'].min())

print("\nMinimo de Ancho de petalo:")
print(df['Ancho_petalo'].min())

print("\nMinimo de Largo de sepalo:")
print(df['Largo_sepalo'].min())

print("\nMinimo de Ancho de sepalo:")
print(df['Ancho_sepalo'].min())

#--------------------------------------------------------

print("\nDesviacion estandar de las columnas numericas :")
print("\nDesviacion estandar de Largo de petalo:")
print(df['Largo_petalo'].std())

print("\nDesviacion estandar de Ancho de petalo:")
print(df['Ancho_petalo'].std())

print("\nDesviacion estandar de Largo de sepalo:")
print(df['Largo_sepalo'].std())

print("\nDesviacion estandar de Ancho de sepalo:")
print(df['Ancho_sepalo'].std())

#--------------------------------------------------------

# Visualizaciones 
import matplotlib.pyplot as plt
import seaborn as sns

print("\n7. Visualizaciones")
#plt.hist(df['Largo_petalo'], bins=10, color='green', edgecolor='black')
#plt.title('Histograma del Largo del Petalo')
#plt.xlabel('Largo del Petalo (cm)') 
#plt.ylabel('Frecuencia')
#plt.show()

#sns.barplot(x='Especies', y='Largo_petalo', data=df, palette='pastel')
#plt.title('Largo del Petalo por Especies')
#plt.xlabel('Especies')
#plt.ylabel('Largo del Petalo (cm)')
#plt.show()

#sns.scatterplot(x='Largo_petalo', y='Ancho_petalo', hue='Especies', data=df, palette='deep')
#plt.title('Grafico de dispersion: Largo vs Ancho del Petalo por Especies')
#plt.xlabel('Largo del Petalo (cm)')
#plt.ylabel('Ancho del Petalo (cm)')
#plt.show()


#plt.plot(df['Largo_sepalo'], color='pink', marker='o', linestyle='-', markersize=5)
#plt.title('Grafico de linea del Largo del Sepalo')
#plt.xlabel('Indice de la muestra')
#plt.ylabel('Largo del Sepalo (cm)')
#plt.show()

sns.lineplot(x='Largo_sepalo', y='Ancho_sepalo', hue='Especies', data=df, palette='bright')
plt.title('Grafico de linea: Largo vs Ancho del Sepalo por Especies')
plt.xlabel('Largo del Sepalo (cm)')
plt.ylabel('Ancho del Sepalo (cm)')
plt.show() 

# Los graficos de seaborn permiten ver mejor las diferencias entre las especies y comprender mejor los graficos. 










