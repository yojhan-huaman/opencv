from flask import Flask, render_template, request, Response
import cv2
import os
from detector import DetectorFormas

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
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Procesar frame con detección de formas
        resultado = detector.detectar_formas(frame)

        # Convertir a JPEG
        _, buffer = cv2.imencode('.jpg', resultado)
        frame = buffer.tobytes()

        # Enviar frame como parte del stream MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    """Stream de la cámara procesada con detección de formas."""
    return Response(generar_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)