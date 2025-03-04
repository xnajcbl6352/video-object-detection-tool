# Herramienta de Detección de Objetos en Video

Esta aplicación permite subir videos y detectar objetos en movimiento utilizando Roboflow Inference.

## Características

- Interfaz web para subir videos
- Procesamiento de video con detección de objetos
- Visualización del video procesado con las detecciones marcadas

## Requisitos

- Python 3.7+
- Una cuenta de Roboflow y API Key

## Instalación

1. Clona el repositorio:
   ```
   git clone https://github.com/xnajcbl6352/video-object-detection-tool.git
   cd video-object-detection-tool
   ```

2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

3. Crea un archivo `.env` a partir del ejemplo:
   ```
   cp .env.example .env
   ```

4. Edita el archivo `.env` y añade tu API Key de Roboflow.

## Uso

1. Inicia la aplicación:
   ```
   python app.py
   ```

2. Abre un navegador y ve a `http://localhost:5000`

3. Sube un video para detectar objetos

## Funcionamiento

La aplicación hace lo siguiente:

1. El usuario sube un video a través de la interfaz web
2. El servidor procesa el video frame por frame
3. Cada frame se analiza con el modelo de detección de objetos de Roboflow
4. Se dibujan rectángulos alrededor de los objetos detectados en cada frame
5. Se genera un nuevo video con las detecciones marcadas
6. El video procesado se muestra al usuario

## Personalización

Puedes cambiar el modelo de detección de objetos modificando la función `init_roboflow()` en `app.py`. Por defecto, se utiliza el modelo COCO estándar, pero puedes usar cualquier otro modelo disponible en Roboflow.

## Obtener una API Key de Roboflow

1. Regístrate en [Roboflow](https://roboflow.com/) (es gratuito)
2. Ve a tu configuración de cuenta
3. Copia tu API Key

## Licencia

MIT