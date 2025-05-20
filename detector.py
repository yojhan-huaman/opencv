import cv2
import numpy as np

# Variable global para el área mínima
AREA_MINIMA_GLOBAL = 1000

class DetectorFormas:
    def __init__(self, max_formas=5, area_minima=AREA_MINIMA_GLOBAL, umbral_otros=6):
        """Inicializa el detector de formas geométricas con un límite de formas a detectar."""
        self.contador_formas = {
            "Circulo": 0,
            "Triangulo": 0,
            "Cuadrado": 0,
            "Rectangulo": 0,
            "Pentagono": 0,
            "Hexagono": 0,
            "Otros": 0
        }
        self.max_formas = max_formas
        self.area_minima = area_minima  # Área mínima para considerar un contorno
        self.umbral_otros = umbral_otros  # Número máximo de vértices que se permiten para "Otros"

    def preprocesar_imagen(self, imagen):
        """Preprocesa la imagen para detección de contornos."""
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        desenfocada = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(desenfocada, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        bordes_dilatados = cv2.dilate(bordes, kernel, iterations=1)
        return bordes_dilatados

    def es_cuadrado(self, aprox):
        """Determina si la figura es un cuadrado basado en la relación de sus lados."""
        if len(aprox) != 4:
            return False
        lados = []
        for i in range(4):
            pt1 = aprox[i][0]
            pt2 = aprox[(i + 1) % 4][0]
            lado = np.linalg.norm(pt1 - pt2)
            lados.append(lado)
        max_lado = max(lados)
        min_lado = min(lados)
        if min_lado / max_lado < 0.9:  # tolerancia de tamaño
            return False
        return True

    def detectar_formas(self, imagen):
        # Reiniciar contador por frame
        self.contador_formas = {clave: 0 for clave in self.contador_formas}

        imagen_resultado = imagen.copy()
        bordes = self.preprocesar_imagen(imagen)
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            if cv2.contourArea(contorno) < self.area_minima:
                continue  # Ignorar contornos pequeños

            perimetro = cv2.arcLength(contorno, True)
            aprox = cv2.approxPolyDP(contorno, 0.04 * perimetro, True)
            x, y, w, h = cv2.boundingRect(contorno)
            vertices = len(aprox)

            forma = None
            color = None

            if vertices > self.umbral_otros:
                forma = "Otros"
                color = (128, 128, 128)
            elif vertices == 3:
                forma = "Triangulo"
                color = (0, 255, 0)
            elif vertices == 4:
                if self.es_cuadrado(aprox):
                    forma = "Cuadrado"
                    color = (255, 0, 0)
                else:
                    forma = "Rectangulo"
                    color = (0, 0, 255)
            elif vertices == 5:
                forma = "Pentagono"
                color = (255, 165, 0)
            elif vertices == 6:
                forma = "Hexagono"
                color = (0, 255, 255)
            else:
                area = cv2.contourArea(contorno)
                perimetro = cv2.arcLength(contorno, True)
                if perimetro == 0:
                    continue
                circularidad = 4 * np.pi * (area / (perimetro * perimetro))
                if 0.7 < circularidad <= 1.2:
                    forma = "Circulo"
                    color = (0, 255, 255)
                else:
                    forma = "Otros"
                    color = (128, 128, 128)

            if forma:
                # Dibuja contorno siempre
                cv2.drawContours(imagen_resultado, [contorno], 0, color, 2)

                # Dibuja vértices
                for punto in aprox:
                    cv2.circle(imagen_resultado, tuple(punto[0]), 5, (0, 0, 255), -1)

                # Solo contar y mostrar texto si no excede el límite
                if self.contador_formas[forma] < self.max_formas:
                    self.contador_formas[forma] += 1
                    cv2.putText(imagen_resultado, forma, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return imagen_resultado

    def _es_circulo(self, contorno):
        """Determina si un contorno es un círculo basado en su circularidad."""
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if perimetro == 0:
            return False
        circularidad = 4 * np.pi * (area / (perimetro * perimetro))
        return 0.7 < circularidad <= 1.2

    def obtener_estadisticas(self):
        """Devuelve un resumen de las formas detectadas."""
        return self.contador_formas

def ejecutar_camara():
    """Ejecuta la detección de formas en tiempo real con la cámara."""
    detector = DetectorFormas(max_formas=5, area_minima=10000, umbral_otros=6)  # Limitar a máximo 5 formas de cada tipo
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break

        resultado = detector.detectar_formas(frame)

        y_pos = 30
        for forma, cantidad in detector.obtener_estadisticas().items():
            texto = f"{forma}: {cantidad}"
            cv2.putText(resultado, texto, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30

        cv2.imshow("Detector de Formas Geométricas", resultado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def procesar_imagen(ruta_imagen):
    """Procesa una imagen estática desde un archivo."""
    detector = DetectorFormas(max_formas=5, area_minima=10000, umbral_otros=6)
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen '{ruta_imagen}'")
        return

    resultado = detector.detectar_formas(imagen)

    print("Formas detectadas:")
    for forma, cantidad in detector.obtener_estadisticas().items():
        if cantidad > 0:
            print(f"- {forma}: {cantidad}")

    cv2.imshow("Resultado", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        procesar_imagen(sys.argv[1])
    else:
        ejecutar_camara()
