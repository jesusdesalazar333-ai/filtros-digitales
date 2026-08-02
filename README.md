# filtros-digitales
# Actividad Formativa 3
# Implementación y evaluación de filtros digitales

# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import firwin
from scipy.signal import freqz


# ------------------------------------------------------------
# 1. CREACIÓN DE LA SEÑAL
# ------------------------------------------------------------

# Frecuencia de muestreo
fs = 2000

# Duración de la señal en segundos
duracion = 1

# Vector de tiempo
tiempo = np.arange(0, duracion, 1 / fs)

# Creamos tres señales con diferentes frecuencias
senal_50 = np.sin(2 * np.pi * 50 * tiempo)
senal_200 = 0.7 * np.sin(2 * np.pi * 200 * tiempo)
senal_400 = 0.5 * np.sin(2 * np.pi * 400 * tiempo)

# Sumamos las tres señales
senal_original = senal_50 + senal_200 + senal_400

# Creamos ruido blanco
np.random.seed(5)
ruido = 0.4 * np.random.randn(len(tiempo))

# Agregamos el ruido a la señal
senal_con_ruido = senal_original + ruido


# ------------------------------------------------------------
# 2. FILTROS IIR BUTTERWORTH
# ------------------------------------------------------------

# Filtro IIR pasa bajos
# Conserva principalmente la frecuencia de 50 Hz
b_bajos, a_bajos = butter(
    4,
    100,
    btype="lowpass",
    fs=fs
)

senal_pasa_bajos = filtfilt(
    b_bajos,
    a_bajos,
    senal_con_ruido
)

# Filtro IIR pasa altos
# Conserva principalmente la frecuencia de 400 Hz
b_altos, a_altos = butter(
    4,
    300,
    btype="highpass",
    fs=fs
)

senal_pasa_altos = filtfilt(
    b_altos,
    a_altos,
    senal_con_ruido
)

# Filtro IIR pasa bandas
# Conserva principalmente la frecuencia de 200 Hz
b_bandas, a_bandas = butter(
    4,
    [150, 250],
    btype="bandpass",
    fs=fs
)

senal_pasa_bandas = filtfilt(
    b_bandas,
    a_bandas,
    senal_con_ruido
)


# ------------------------------------------------------------
# 3. FILTROS FIR CON VENTANA DE HAMMING
# ------------------------------------------------------------

# Número de coeficientes de los filtros FIR
numero_coeficientes = 101

# Filtro FIR pasa bajos
fir_bajos = firwin(
    numero_coeficientes,
    100,
    fs=fs,
    pass_zero="lowpass",
    window="hamming"
)

senal_fir_bajos = filtfilt(
    fir_bajos,
    1,
    senal_con_ruido
)

# Filtro FIR pasa altos
fir_altos = firwin(
    numero_coeficientes,
    300,
    fs=fs,
    pass_zero="highpass",
    window="hamming"
)

senal_fir_altos = filtfilt(
    fir_altos,
    1,
    senal_con_ruido
)

# Filtro FIR pasa bandas
fir_bandas = firwin(
    numero_coeficientes,
    [150, 250],
    fs=fs,
    pass_zero="bandpass",
    window="hamming"
)

senal_fir_bandas = filtfilt(
    fir_bandas,
    1,
    senal_con_ruido
)


# ------------------------------------------------------------
# 4. GRÁFICA DE LA SEÑAL ORIGINAL
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)

plt.plot(
    tiempo,
    senal_original,
    color="blue"
)

plt.title("Señal original")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid()

plt.subplot(2, 1, 2)

plt.plot(
    tiempo,
    senal_con_ruido,
    color="gray"
)

plt.title("Señal con ruido blanco")
plt.xlabel("Tiempo en segundos")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid()

plt.tight_layout()
plt.savefig("senal_original_y_ruido.png")
plt.show()


# ------------------------------------------------------------
# 5. RESULTADOS DE LOS FILTROS IIR
# ------------------------------------------------------------

plt.figure(figsize=(10, 9))

plt.subplot(4, 1, 1)

plt.plot(
    tiempo,
    senal_con_ruido,
    color="gray"
)

plt.title("Señal con ruido")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid()

plt.subplot(4, 1, 2)

plt.plot(
    tiempo,
    senal_pasa_bajos,
    color="blue"
)

plt.title("Filtro IIR pasa bajos")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid()

plt.subplot(4, 1, 3)

plt.plot(
    tiempo,
    senal_pasa_altos,
    color="red"
)

plt.title("Filtro IIR pasa altos")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid()

plt.subplot(4, 1, 4)

plt.plot(
    tiempo,
    senal_pasa_bandas,
    color="green"
)

plt.title("Filtro IIR pasa bandas")
plt.xlabel("Tiempo en segundos")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid()

plt.tight_layout()
plt.savefig("resultados_filtros_iir.png")
plt.show()


# ------------------------------------------------------------
# 6. RESULTADOS DE LOS FILTROS FIR
# ------------------------------------------------------------

plt.figure(figsize=(10, 9))

plt.subplot(3, 1, 1)

plt.plot(
    tiempo,
    senal_fir_bajos,
    color="blue"
)

plt.title("Filtro FIR pasa bajos")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid()

plt.subplot(3, 1, 2)

plt.plot(
    tiempo,
    senal_fir_altos,
    color="red"
)

plt.title("Filtro FIR pasa altos")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid()

plt.subplot(3, 1, 3)

plt.plot(
    tiempo,
    senal_fir_bandas,
    color="green"
)

plt.title("Filtro FIR pasa bandas")
plt.xlabel("Tiempo en segundos")
plt.ylabel("Amplitud")
plt.xlim(0, 0.1)
plt.grid()

plt.tight_layout()
plt.savefig("resultados_filtros_fir.png")
plt.show()


# ------------------------------------------------------------
# 7. RESPUESTA EN FRECUENCIA
# ------------------------------------------------------------

# Calculamos la respuesta del filtro pasa bajos
frecuencias_bajos, respuesta_bajos = freqz(
    b_bajos,
    a_bajos,
    fs=fs
)

# Calculamos la respuesta del filtro pasa altos
frecuencias_altos, respuesta_altos = freqz(
    b_altos,
    a_altos,
    fs=fs
)

# Calculamos la respuesta del filtro pasa bandas
frecuencias_bandas, respuesta_bandas = freqz(
    b_bandas,
    a_bandas,
    fs=fs
)

plt.figure(figsize=(10, 6))

plt.plot(
    frecuencias_bajos,
    abs(respuesta_bajos),
    color="blue",
    label="Pasa bajos"
)

plt.plot(
    frecuencias_altos,
    abs(respuesta_altos),
    color="red",
    label="Pasa altos"
)

plt.plot(
    frecuencias_bandas,
    abs(respuesta_bandas),
    color="green",
    label="Pasa bandas"
)

plt.title("Respuesta en frecuencia de los filtros")
plt.xlabel("Frecuencia en Hz")
plt.ylabel("Magnitud")
plt.xlim(0, 500)
plt.grid()
plt.legend()

plt.savefig("respuesta_en_frecuencia.png")
plt.show()


# ------------------------------------------------------------
# 8. MENSAJES FINALES
# ------------------------------------------------------------

print("Actividad Formativa 3")
print("Los filtros se aplicaron correctamente.")

print("\nResultados:")

print(
    "El filtro pasa bajos conservó principalmente "
    "la frecuencia de 50 Hz."
)

print(
    "El filtro pasa altos conservó principalmente "
    "la frecuencia de 400 Hz."
)

print(
    "El filtro pasa bandas conservó principalmente "
    "la frecuencia de 200 Hz."
)

print("\nSe crearon las siguientes imágenes:")
print("- senal_original_y_ruido.png")
print("- resultados_filtros_iir.png")
print("- resultados_filtros_fir.png")
print("- respuesta_en_frecuencia.png")
