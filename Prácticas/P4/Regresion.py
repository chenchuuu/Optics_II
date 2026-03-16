import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import sys


# ==========================================
# 1. Definición de los modelos matemáticos
# ==========================================
def modelo_ax_b(x, a, b):
    return a * x + b


def modelo_ax(x, a):
    return a * x


# ==========================================
# 2. Función para leer datos del archivo
# ==========================================
def cargar_datos(nombre_archivo):
    """
    Lee un archivo .dat o de texto donde las columnas están separadas
    por espacios o tabulaciones.
    """
    try:
        # sep=r'\s+' detecta cualquier cantidad de espacios o tabulaciones
        datos = pd.read_csv(nombre_archivo, sep=r'\s+')

        columnas_requeridas = ['x', 'y', 'dx', 'dy']
        # Convertir nombres de columnas a minúsculas para evitar errores tipográficos
        datos.columns = datos.columns.str.lower()

        if not all(col in datos.columns for col in columnas_requeridas):
            print(f"Error: El archivo debe contener un encabezado con las columnas: {', '.join(columnas_requeridas)}")
            sys.exit(1)

        return datos['x'].values, datos['y'].values, datos['dx'].values, datos['dy'].values

    except FileNotFoundError:
        print(
            f"Error: No se encontró el archivo '{nombre_archivo}'. Asegúrate de que esté en la misma carpeta que este script.")
        sys.exit(1)
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer el archivo: {e}")
        sys.exit(1)


# ==========================================
# 3. Función principal de regresión
# ==========================================
def realizar_regresion(x, y, dy, tipo_modelo):
    if tipo_modelo == '1':
        modelo = modelo_ax_b
        nombre_mod = "Ax + B"
    elif tipo_modelo == '2':
        modelo = modelo_ax
        nombre_mod = "Ax"
    else:
        print("Opción de modelo no válida.")
        sys.exit(1)

    # Prevenir división por cero si algún dy es exactamente 0 (útil para el cálculo de chi^2)
    dy_seguro = np.where(dy == 0, 1e-10, dy)

    # Ajuste de curva ponderado por las incertidumbres en el eje Y
    popt, pcov = curve_fit(modelo, x, y, sigma=dy_seguro, absolute_sigma=True)

    # Incertidumbres (raíz cuadrada de la diagonal de la matriz de covarianza)
    p_err = np.sqrt(np.diag(pcov))

    # Cálculo de Chi Cuadrado (chi^2)
    y_ajuste = modelo(x, *popt)
    residuos = y - y_ajuste
    chi_cuadrado = np.sum((residuos / dy_seguro) ** 2)

    # Cálculo del Coeficiente de Pearson (r)
    pearson_r, _ = pearsonr(x, y)

    # Visualización de resultados en consola
    print(f"\n--- Resultados de la Regresión (Modelo: {nombre_mod}) ---")
    if tipo_modelo == '1':
        print(f"Pendiente (A)     : {popt[0]:.5e} ± {p_err[0]:.5e}")
        print(f"Intersección (B)  : {popt[1]:.5e} ± {p_err[1]:.5e}")
    else:
        print(f"Pendiente (A)     : {popt[0]:.5e} ± {p_err[0]:.5e}")

    print(f"Chi^2             : {chi_cuadrado:.5f}")
    print(f"Coef. Pearson (r) : {pearson_r:.5f}")
    print("-" * 50)

    return popt, pcov, chi_cuadrado, pearson_r


# ==========================================
# 4. Función de graficación
# ==========================================
def graficar_resultados(x, y, dx, dy, popt, tipo_modelo):
    plt.figure(figsize=(10, 6))

    # Gráfico de datos experimentales con barras de error
    plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='o', color='black',
                 ecolor='gray', elinewidth=1.5, capsize=4, label='Datos Experimentales')

    # Línea de ajuste
    x_grid = np.linspace(min(x), max(x), 100)

    if tipo_modelo == '1':
        y_fit = modelo_ax_b(x_grid, *popt)
        label_fit = f'Ajuste: y = ({popt[0]:.2f})x + ({popt[1]:.2f})'
    else:
        y_fit = modelo_ax(x_grid, *popt)
        label_fit = f'Ajuste: y = ({popt[0]:.2f})x'

    plt.plot(x_grid, y_fit, color='red', linewidth=2, label=label_fit)

    # Configuración de la gráfica
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.title('Regresión Lineal con Barras de Error')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    print("\nGenerando gráfica... Cierre la ventana de la gráfica para terminar el programa.")
    plt.show()


# ==========================================
# 5. Ejecución principal
# ==========================================
if __name__ == "__main__":
    print("--- Programa de Regresión Lineal ---")

    # Pedimos el archivo .dat
    archivo = input("Introduce el nombre del archivo de datos (ej: datos.dat): ")
    x, y, dx, dy = cargar_datos(archivo)
    print(f"-> Se han cargado {len(x)} puntos experimentales.")

    # Selección de modelo
    print("\nElija el tipo de regresión:")
    print("  1) Modelo Completo: y = Ax + B")
    print("  2) Por el Origen:   y = Ax")
    seleccion = input("Seleccione (1 o 2): ")

    # Cálculos y gráfica
    popt, pcov, chi2, r = realizar_regresion(x, y, dy, seleccion)
    graficar_resultados(x, y, dx, dy, popt, seleccion)

    print("\nPrograma finalizado correctamente.")