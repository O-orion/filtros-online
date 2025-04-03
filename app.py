from flask import Flask, render_template, request, url_for, send_from_directory
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def apply_filter(image, filter_type):
    if filter_type == 'bw':
        filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        return filtered
    elif filter_type == 'sepia':
        sepia_matrix = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia_image = cv2.transform(image, sepia_matrix)
        sepia_image = np.clip(sepia_image, 0, 255)
        return sepia_image
    elif filter_type == 'blur':
        return cv2.GaussianBlur(image, (15, 15), 0)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    filtered_image = None
    filter_name = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return 'Nenhuma imagem enviada', 400

        file = request.files['image']
        if file.filename == '':
            return 'Nenhum arquivo selecionado', 400

        if file:
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + file.filename)
            file.save(original_path)

            image = cv2.imread(original_path)
            if image is None:
                return 'Erro ao carregar a imagem', 400

            filter_type = request.form['filter']
            filter_names = {'bw': 'Preto e Branco', 'sepia': 'Sépia', 'blur': 'Blur'}
            filter_name = filter_names.get(filter_type, 'Desconhecido')

            filtered = apply_filter(image, filter_type)

            # Garante que a imagem seja uint8
            filtered = filtered.astype(np.uint8)

            # Força a extensão como .png
            filtered_filename = 'filtered_' + os.path.splitext(file.filename)[0] + '.png'
            filtered_path = os.path.join(app.config['UPLOAD_FOLDER'], filtered_filename)

            # Depuração: imprime informações antes de salvar
            print(f"Salvando em: {filtered_path}")
            print(f"Formato da imagem: {filtered.shape}, Tipo: {filtered.dtype}")

            cv2.imwrite(filtered_path, filtered)

            filtered_image = filtered_filename

    return render_template('index.html', filtered_image=filtered_image, filter_name=filter_name)

@app.route('/uploads/<filename>')
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
