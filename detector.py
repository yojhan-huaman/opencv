import cv2
import numpy as np

class DetectorFormas:
    def __init__(self):
        """Inicializa el detector de formas geométricas."""
        # Diccionario para contar las formas detectadas
        self.contador_formas = {
            "Circulo": 0,
            "Triangulo": 0,
            "Cuadrado": 0,
            "Rectangulo": 0,
            "Pentagono": 0,
            "Hexagono": 0,
            "Ovalo": 0,
            "Estrella": 0,
            "Otro": 0
        }
        
    def preprocesar_imagen(self, imagen):
        """Preprocesa la imagen para detección de contornos."""
        # Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Aplicar desenfoque para reducir ruido
        desenfocada = cv2.GaussianBlur(gris, (5, 5), 0)
        
        # Detectar bordes
        bordes = cv2.Canny(desenfocada, 50, 150)
        
        # Dilatar para cerrar posibles brechas en los contornos
        kernel = np.ones((3, 3), np.uint8)
        bordes_dilatados = cv2.dilate(bordes, kernel, iterations=1)
        
        return bordes_dilatados
    
    def detectar_formas(self, imagen):
        """Detecta y clasifica formas geométricas en la imagen."""
        # Crear una copia de la imagen para dibujar
        imagen_resultado = imagen.copy()
        
        # Preprocesar la imagen
        bordes = self.preprocesar_imagen(imagen)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reiniciar contador
        self.contador_formas = {k: 0 for k in self.contador_formas}
        
        # Analizar cada contorno
        for contorno in contornos:
            # Ignorar contornos pequeños (posible ruido)
            if cv2.contourArea(contorno) < 100:
                continue
                
            # Obtener perímetro para aproximar contorno
            perimetro = cv2.arcLength(contorno, True)
            # Aproximar el contorno a una forma geométrica
            aprox = cv2.approxPolyDP(contorno, 0.04 * perimetro, True)
            
            # Obtener propiedades de la forma
            x, y, w, h = cv2.boundingRect(contorno)
            
            # Clasificar según número de vértices
            vertices = len(aprox)
            
            # Determinar la forma según el número de vértices
            if vertices == 3:
                forma = "Triangulo"
                color = (0, 255, 0)  # Verde
            elif vertices == 4:
                # Distinguir entre cuadrado y rectángulo
                proporcion = float(w) / h
                if 0.95 <= proporcion <= 1.05:
                    forma = "Cuadrado"
                    color = (255, 0, 0)  # Azul
                else:
                    forma = "Rectangulo"
                    color = (0, 0, 255)  # Rojo
            elif vertices == 5:
                forma = "Pentagono"
                color = (255, 165, 0)  # Naranja
            elif vertices == 6:
                forma = "Hexagono"
                color = (0, 255, 255)  # Amarillo
            elif vertices > 6:
                # Detectar Estrella
                if self.es_estrella(contorno):
                    forma = "Estrella"
                    color = (255, 255, 0)  # Cian
                else:
                    forma = "Otro"
                    color = (128, 128, 128)  # Gris
            else:
                # Comprobar si es un círculo u óvalo
                area = cv2.contourArea(contorno)
                radio = w / 2
                # Detectar círculo
                if abs((w - h) / (w + h)) < 0.2 and abs(area / (np.pi * radio * radio) - 1) < 0.2:
                    forma = "Circulo"
                    color = (0, 255, 255)  # Amarillo
                # Detectar óvalo
                elif abs(w - h) > 10:
                    forma = "Ovalo"
                    color = (255, 0, 255)  # Magenta
                else:
                    forma = "Otro"
                    color = (128, 128, 128)  # Gris
            
            # Incrementar contador
            self.contador_formas[forma] += 1
            
            # Dibujar contorno y etiqueta
            cv2.drawContours(imagen_resultado, [contorno], 0, color, 2)
            cv2.putText(imagen_resultado, forma, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return imagen_resultado
    
    def es_estrella(self, contorno):
        """Función para determinar si el contorno es una estrella."""
        # Si el contorno tiene muchos vértices (por ejemplo, más de 8), es posible que sea una estrella
        # Se puede mejorar con un análisis más complejo
        area = cv2.contourArea(contorno)
        if area > 1000 and len(contorno) > 8:
            return True
        return False
    
    def obtener_estadisticas(self):
        """Devuelve un resumen de las formas detectadas."""
        return self.contador_formas


def ejecutar_camara():
    """Ejecuta la detección de formas en tiempo real con la cámara."""
    detector = DetectorFormas()
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
            
        # Detectar formas
        resultado = detector.detectar_formas(frame)
        
        # Mostrar estadísticas
        y_pos = 30
        for forma, cantidad in detector.obtener_estadisticas().items():
            texto = f"{forma}: {cantidad}"
            cv2.putText(resultado, texto, (10, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
        
        # Mostrar resultado
        cv2.imshow("Detector de Formas Geométricas", resultado)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


def procesar_imagen(ruta_imagen):
    """Procesa una imagen estática desde un archivo."""
    detector = DetectorFormas()
    
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen '{ruta_imagen}'")
        return
        
    # Detectar formas
    resultado = detector.detectar_formas(imagen)
    
    # Mostrar estadísticas
    print("Formas detectadas:")
    for forma, cantidad in detector.obtener_estadisticas().items():
        if cantidad > 0:
            print(f"- {forma}: {cantidad}")
    
    # Mostrar resultado
    cv2.imshow("Resultado", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Modo de imagen estática
        procesar_imagen(sys.argv[1])
    else:
        # Modo de cámara
        ejecutar_camara()
