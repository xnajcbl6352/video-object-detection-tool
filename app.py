import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
import tempfile
from roboflow import Roboflow
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de Roboflow (necesitarás una API key de Roboflow)
rfAPI_KEY = os.getenv('ROBOFLOW_API_KEY')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limitar a 50MB

# Inicializar Roboflow - usamos COCO por defecto, pero puedes cambiar a otro modelo
def init_roboflow():
    rf = Roboflow(api_key=rfAPI_KEY)
    workspace = rf.workspace("roboflow-jvuqo")
    model = workspace.project("coco-object-detection").version(1).model
    return model

# Ruta principal para mostrar la interfaz de carga
@app.route('/')
def index():
    return render_template('index.html')

# Procesar video y detectar objetos
@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No se ha subido ningún archivo'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No se ha seleccionado ningún archivo'}), 400
    
    # Guardar el archivo de video subido temporalmente
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Crear un archivo temporal para el video procesado
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    # Procesar el video con detección de objetos
    try:
        model = init_roboflow()
        process_video_with_detection(filepath, output_path, model)
        
        # Devolver el video procesado
        return jsonify({
            'success': True,
            'message': 'Video procesado correctamente',
            'video_url': f'/processed/{output_filename}'
        })
    except Exception as e:
        return jsonify({'error': f'Error al procesar el video: {str(e)}'}), 500

# Ruta para servir el video procesado
@app.route('/processed/<filename>')
def processed_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    def generate():
        with open(filepath, 'rb') as f:
            data = f.read(1024)
            while data:
                yield data
                data = f.read(1024)
    
    return Response(generate(), mimetype='video/mp4')

# Función para procesar el video y detectar objetos
def process_video_with_detection(input_path, output_path, model):
    # Abrir el video
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        raise Exception("No se pudo abrir el archivo de video")
    
    # Obtener información del video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configurar el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Procesar cada frame
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Solo procesar cada 2 frames para acelerar (esto es ajustable)
        if frame_count % 2 == 0:
            # Guardar frame temporalmente
            temp_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_frame.jpg')
            cv2.imwrite(temp_frame_path, frame)
            
            # Realizar la predicción con Roboflow
            prediction = model.predict(temp_frame_path, confidence=40, overlap=30)
            
            # Dibujar las detecciones en el frame
            for detection in prediction.json()['predictions']:
                x1 = detection['x'] - detection['width'] / 2
                y1 = detection['y'] - detection['height'] / 2
                x2 = detection['x'] + detection['width'] / 2
                y2 = detection['y'] + detection['height'] / 2
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 
                          f"{detection['class']} {int(detection['confidence'] * 100)}%", 
                          (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Eliminar el archivo temporal
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
        
        # Escribir el frame en el video de salida
        out.write(frame)
        frame_count += 1
    
    # Liberar recursos
    video.release()
    out.release()

if __name__ == '__main__':
    app.run(debug=True)