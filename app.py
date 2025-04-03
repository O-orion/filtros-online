from flask import Flask, render_template, request, url_for, send_from_directory
import os
import cv2
import numpy as np
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def apply_filter(image, filter_type, intensity=15):
    if image is None:
        print(f"Erro: Imagem nula ao aplicar filtro {filter_type}")
        return None
    if filter_type == 'bw':
        filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    elif filter_type == 'sepia':
        sepia_matrix = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia_image = cv2.transform(image, sepia_matrix)
        return np.clip(sepia_image, 0, 255)
    elif filter_type == 'blur':
        ksize = int(intensity) | 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif filter_type == 'negative':
        return cv2.bitwise_not(image)
    elif filter_type == 'edges':
        filtered = cv2.Canny(image, 100, 200)
        return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    elif filter_type == 'bright':
        return cv2.convertScaleAbs(image, alpha=1.5, beta=50)
    elif filter_type == 'vintage':
        sepia_matrix = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia_image = cv2.transform(image, sepia_matrix)
        sepia_image = np.clip(sepia_image, 0, 255)
        noise = np.random.normal(0, 25, sepia_image.shape)
        return np.clip(sepia_image + noise, 0, 255)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    filtered_image = None
    filter_name = None
    original_image = None
    error = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error = 'Nenhuma imagem enviada'
        else:
            file = request.files['image']
            if file.filename == '':
                error = 'Nenhum arquivo selecionado'
            else:
                original_filename = 'original_' + os.path.splitext(file.filename)[0] + '.png'
                original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
                file.save(original_path)

                image = cv2.imread(original_path)
                if image is None:
                    error = 'Erro ao carregar a imagem'
                else:
                    image = cv2.resize(image, (800, 600))
                    image = image.astype(np.uint8)
                    cv2.imwrite(original_path, image)
                    original_image = original_filename

                    filter_types = request.form.getlist('filter')
                    intensity = request.form.get('intensity', 15)

                    filter_names = {
                        'bw': 'Preto e Branco', 'sepia': 'Sépia', 'blur': 'Blur',
                        'negative': 'Negativo', 'edges': 'Bordas', 'bright': 'Brilho',
                        'vintage': 'Vintage'
                    }
                    applied_filters = [filter_names.get(f, 'Desconhecido') for f in filter_types]
                    filter_name = ' + '.join(applied_filters) if applied_filters else 'Nenhum'

                    filtered = image.copy()
                    print(f"Aplicando filtros: {filter_types}")
                    for filter_type in filter_types:
                        filtered = apply_filter(filtered, filter_type, intensity)
                        if filtered is None:
                            error = f"Erro ao aplicar o filtro {filter_type}"
                            break
                        print(f"Filtro {filter_type} aplicado. Formato: {filtered.shape}, Tipo: {filtered.dtype}")

                    if filtered is not None:
                        filtered = filtered.astype(np.uint8)
                        filtered_filename = 'filtered_' + os.path.splitext(file.filename)[0] + '.png'
                        filtered_path = os.path.join(app.config['UPLOAD_FOLDER'], filtered_filename)
                        print(f"Salvando imagem filtrada em: {filtered_path}")
                        success = cv2.imwrite(filtered_path, filtered)
                        if not success:
                            error = "Erro ao salvar a imagem filtrada"
                        else:
                            filtered_image = filtered_filename

                    # Limpeza de arquivos antigos
                    files = sorted(os.listdir(UPLOAD_FOLDER), key=lambda x: os.path.getctime(os.path.join(UPLOAD_FOLDER, x)))
                    if len(files) > 10:
                        for old_file in files[:-10]:
                            os.remove(os.path.join(UPLOAD_FOLDER, old_file))

    return render_template('index.html', filtered_image=filtered_image, filter_name=filter_name,
                          original_image=original_image, error=error)

@app.route('/uploads/<filename>')
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
