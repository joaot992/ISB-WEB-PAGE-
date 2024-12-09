from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pywt
from scipy.signal import butter, filtfilt

app = Flask(__name__)

# Función para aplicar filtrado Wavelet Denoising
def aplicar_wavelet_denoising(datos, wavelet='db6', nivel=4):
    coeficientes = pywt.wavedec(datos, wavelet, level=nivel)
    sigma = np.std(coeficientes[-1])  # Estimar ruido
    umbral = sigma * np.sqrt(2 * np.log(len(datos)))  # Umbral para los coeficientes
    coeficientes[1:] = [pywt.threshold(c, value=umbral, mode='soft') for c in coeficientes[1:]]
    return pywt.waverec(coeficientes, wavelet)

# Función para aplicar filtro de Butterworth
def aplicar_filtro_butterworth(datos, orden=4, frecuencia_corte=50, fs=500):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = frecuencia_corte / nyquist
    b, a = butter(orden, normal_cutoff, btype='low', analog=False)  # Filtro pasabajos
    return filtfilt(b, a, datos)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Leer el archivo CSV
        df = pd.read_csv(file)
        
        # Validar si la columna "Voltaje (mV)" está presente
        if "Voltaje (mV)" not in df.columns:
            return jsonify({'error': '"Voltaje (mV)" column not found in the file'}), 400
        
        # Obtener la columna "Voltaje (mV)"
        voltaje_data = df["Voltaje (mV)"].astype(float).fillna(0)

        # Aplicar filtrado Wavelet Denoising
        filtrada_wavelet = aplicar_wavelet_denoising(voltaje_data.values)

        # Aplicar filtrado con Butterworth
        filtrada_butter = aplicar_filtro_butterworth(filtrada_wavelet)

        # Calcular estadísticas de la señal filtrada
        mean_voltage = np.mean(filtrada_butter)
        max_voltage = np.max(filtrada_butter)
        
        # Convertir la señal filtrada a lista para devolverla como JSON
        result = filtrada_butter.tolist()
        return jsonify({
            'data': result,
            'mean_voltage': mean_voltage,
            'max_voltage': max_voltage
        })
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'The uploaded file is empty or invalid'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)



