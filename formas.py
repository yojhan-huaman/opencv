import cv2
import numpy as np
import random
import time

class JuegoFormasGeometricas:
    def __init__(self):
        """Inicializa el juego educativo de formas geométricas."""
        self.detector = DetectorFormas()
        self.nivel = 1
        self.puntuacion = 0
        self.forma_objetivo = None
        self.tiempo_inicio = None
        self.tiempo_limite = 10  # segundos por ronda
        self.formas_disponibles = ["Circulo", "Triangulo", "Cuadrado", "Rectangulo", "Pentagono", "Hexagono", "Ovalo", "Paralelogramo", "Estrella"]
        self.seleccionar_forma_objetivo()
        
    def seleccionar_forma_objetivo(self):
        """Selecciona aleatoriamente una forma como objetivo."""
        self.forma_objetivo = random.choice(self.formas_disponibles)
        self.tiempo_inicio = time.time()
        
    def verificar_objetivo(self, estadisticas):
        """Verifica si se ha mostrado la forma objetivo."""
        if estadisticas[self.forma_objetivo] > 0:
            # Calcular puntos basados en el tiempo restante
            tiempo_transcurrido = time.time() - self.tiempo_inicio
            tiempo_restante = max(0, self.tiempo_limite - tiempo_transcurrido)
            puntos = int(tiempo_restante * 10) + 50
            
            self.puntuacion += puntos
            self.nivel += 1
            self.tiempo_limite = max(3, 10 - (self.nivel // 2))  # Reducir tiempo con cada nivel
            
            # Seleccionar nueva forma objetivo
            self.seleccionar_forma_objetivo()
            return True, puntos
        return False, 0
        
    def tiempo_restante(self):
        """Devuelve el tiempo restante para la ronda actual."""
        tiempo_transcurrido = time.time() - self.tiempo_inicio
        return max(0, self.tiempo_limite - tiempo_transcurrido)
        
    def tiempo_agotado(self):
        """Verifica si se ha agotado el tiempo para la ronda actual."""
        return self.tiempo_restante() <= 0


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
            "Paralelogramo": 0,
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
                # Distinguir entre cuadrado, rectángulo y paralelogramo
                proporcion = float(w) / h
                if 0.95 <= proporcion <= 1.05:
                    forma = "Cuadrado"
                    color = (255, 0, 0)  # Azul
                else:
                    # Detectar paralelogramo mediante el ángulo
                    angulo = cv2.minAreaRect(contorno)[2]
                    if abs(angulo) > 10 and abs(angulo) < 80:
                        forma = "Paralelogramo"
                        color = (255, 255, 0)  # Azul claro
                    else:
                        forma = "Rectangulo"
                        color = (0, 0, 255)  # Rojo
            elif vertices == 5:
                forma = "Pentagono"
                color = (255, 165, 0)  # Naranja
            elif vertices == 6:
                forma = "Hexagono"
                color = (0, 255, 255)  # Amarillo
            elif vertices > 6 and vertices < 12:
                # Detectar estrellas por el número de vértices
                if vertices % 2 == 0:
                    forma = "Estrella"
                    color = (255, 0, 255)  # Rosa
                else:
                    forma = "Otro"
                    color = (128, 128, 128)  # Gris
            else:
                # Comprobar si es un óvalo
                area = cv2.contourArea(contorno)
                radio = w / 2
                if abs((w - h) / (w + h)) > 0.3 and abs(area / (np.pi * radio * radio) - 1) > 0.3:
                    forma = "Ovalo"
                    color = (0, 255, 255)  # Amarillo
                else:
                    forma = "Circulo"
                    color = (0, 255, 0)  # Verde
            
            # Incrementar contador
            self.contador_formas[forma] += 1
            
            # Dibujar contorno y etiqueta
            cv2.drawContours(imagen_resultado, [contorno], 0, color, 2)
            cv2.putText(imagen_resultado, forma, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return imagen_resultado
    
    def obtener_estadisticas(self):
        """Devuelve un resumen de las formas detectadas."""
        return self.contador_formas


def ejecutar_juego():
    """Ejecuta el juego educativo de formas geométricas."""
    juego = JuegoFormasGeometricas()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return
        
    print("¡Juego de Formas Geométricas!")
    print("Muestra la forma que se indica en pantalla antes de que se acabe el tiempo.")
    print("Presiona 'q' para salir")
    
    game_over = False
    mensaje = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break
            
        # Detectar formas
        resultado = juego.detector.detectar_formas(frame)
        
        # Información del juego
        # Crear una barra superior para la información del juego
        info_bar = np.zeros((100, resultado.shape[1], 3), dtype=np.uint8)
        
        # Mostrar nivel y puntuación
        cv2.putText(info_bar, f"Nivel: {juego.nivel}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_bar, f"Puntuación: {juego.puntuacion}", (10, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar forma objetivo
        if not game_over:
            cv2.putText(info_bar, f"Muestra un: {juego.forma_objetivo}", (resultado.shape[1] // 2 - 100, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Mostrar tiempo restante
            tiempo_restante = juego.tiempo_restante()
            cv2.putText(info_bar, f"Tiempo: {tiempo_restante:.1f}s", (resultado.shape[1] // 2 - 100, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Verificar si se ha mostrado la forma objetivo
            exito, puntos = juego.verificar_objetivo(juego.detector.obtener_estadisticas())
            if exito:
                mensaje = f"¡Correcto! +{puntos} puntos"
            
            # Verificar si se ha agotado el tiempo
            if juego.tiempo_agotado():
                game_over = True
                mensaje = "¡Tiempo agotado! Juego terminado."
        else:
            # Mostrar mensaje de game over
            cv2.putText(info_bar, "JUEGO TERMINADO", (resultado.shape[1] // 2 - 100, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(info_bar, f"Puntuación final: {juego.puntuacion}", (resultado.shape[1] // 2 - 100, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar mensaje
        if mensaje:
            cv2.putText(info_bar, mensaje, (10, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combinar barra de información con el resultado
        pantalla = np.vstack([info_bar, resultado])
        
        # Mostrar resultado
        cv2.imshow("Juego de Formas Geométricas", pantalla)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Reducir la frecuencia de mensaje después de un tiempo
        if mensaje and (time.time() - juego.tiempo_inicio) > 2:
            mensaje = ""
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ejecutar_juego()