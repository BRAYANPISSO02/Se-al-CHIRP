# -*- coding: utf-8 -*-
"""
Exploración de una señal chirp con barrido de frecuencia lineal.
Este script genera una señal con frecuencia creciente (chirp lineal),
y realiza su análisis en el dominio del tiempo y la frecuencia.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Configuración básica
tasa_muestreo = 1000           # [Hz] frecuencia de muestreo
duracion_total = 2             # [s] duración de la señal
tiempo = np.linspace(0, duracion_total, int(tasa_muestreo * duracion_total), endpoint=False)

# Especificación de frecuencia inicial y final
inicio_freq = 5                # [Hz] frecuencia inicial
fin_freq = 100                # [Hz] frecuencia final

# Construcción del chirp (barrido lineal de frecuencia)
pendiente_frec = (fin_freq - inicio_freq) / duracion_total
fase_inst = 2 * np.pi * (inicio_freq * tiempo + 0.5 * pendiente_frec * tiempo**2)
senal_chirp = np.sin(fase_inst)

# Cálculo de la transformada rápida de Fourier
num_muestras = len(senal_chirp)
espectro = fft(senal_chirp)
frecuencias = fftfreq(num_muestras, 1 / tasa_muestreo)

# Solo se considera el espectro positivo
frecs_positivas = frecuencias[:num_muestras // 2]
magnitud_positiva = 2.0 / num_muestras * np.abs(espectro[:num_muestras // 2])

# Visualización
plt.figure(figsize=(12, 5))

# Señal en el tiempo
plt.subplot(1, 2, 1)
plt.plot(tiempo, senal_chirp)
plt.title("Chirp con Frecuencia Linealmente Ascendente")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")

# Transformada de Fourier
plt.subplot(1, 2, 2)
plt.plot(frecs_positivas, magnitud_positiva)
plt.title("Espectro de la Señal Chirp")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")

plt.tight_layout()
plt.show()
