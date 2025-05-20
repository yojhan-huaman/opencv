from flask import Flask, render_template, request, Response, jsonify
import base64
import numpy as np
import cv2
import os
from detector import DetectorFormas  # Asegúrate de que esta clase exista

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Crear la carpeta 'static' si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Instancia global del detector
detector = DetectorFormas()

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = {}
    if request.method == 'POST':
        file = request.files['imagen']
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.png')
            file.save(path)

            # Cargar imagen
            imagen = cv2.imread(path)

            # Procesar imagen
            imagen_resultado = detector.detectar_formas(imagen)
            resultado = detector.obtener_estadisticas()

            # Guardar imagen procesada
            salida_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
            cv2.imwrite(salida_path, imagen_resultado)

    return render_template('index.html', resultado=resultado)


@app.route('/camara')
def camara():
    """Página que muestra el video en tiempo real."""
    return render_template('camara.html')


def generar_video():
    """Generador de frames procesados desde la cámara."""
    cap = cv2.VideoCapture(0)  # 0 es la cámara por defecto

    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara. Asegúrate de que esté conectada y libre de otros procesos.")

    while True:
        success, frame = cap.read()
        if not success:
            break  # Si no se pudo leer un frame, salir del loop

        # Procesar el frame para detectar formas
        resultado = detector.detectar_formas(frame)

        # Convertir el frame a formato JPEG
        _, buffer = cv2.imencode('.jpg', resultado)
        frame = buffer.tobytes()

        # Enviar el frame procesado como parte del flujo MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/procesar_imagen', methods=['POST'])
def procesar_imagen():
    try:
        archivo = request.files['imagen']
        imagen_np = np.frombuffer(archivo.read(), np.uint8)
        imagen = cv2.imdecode(imagen_np, cv2.IMREAD_COLOR)

        resultado = detector.detectar_formas(imagen)

        _, buffer = cv2.imencode('.jpg', resultado)
        imagen_base64 = base64.b64encode(buffer).decode('utf-8')

        estadisticas = detector.obtener_estadisticas()
        print("Estadísticas detectadas:", estadisticas)

        return jsonify({
            'imagen': imagen_base64,
            'estadisticas': estadisticas
        })
    except Exception as e:
        print("Error procesando imagen:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/video_feed')
def video_feed():
    """Stream de la cámara procesada con detección de formas."""
    return Response(generar_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/estadisticas')
def estadisticas():
    """Endpoint que devuelve las estadísticas de formas detectadas en formato JSON."""
    return jsonify(detector.obtener_estadisticas())

if __name__ == '__main__':
    app.run(debug=True)
