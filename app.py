from flask import Flask, render_template, jsonify
import torch
from model import Generator  # Import your Generator class from model.py
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the pre-trained GAN model
NOISE_DIM = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(NOISE_DIM).to(device)
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.eval()

# Convert generated images to base64 to send to the frontend
def image_to_base64(image_tensor):
    image_tensor = (image_tensor + 1) * 127.5  # Rescale to [0, 255]
    image_array = image_tensor.numpy().astype(np.uint8)
    im = Image.fromarray(image_array)
    buffered = BytesIO()
    im.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Route to serve the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to generate images
@app.route('/generate', methods=['GET'])
def generate_images():
    noise = torch.randn(16, NOISE_DIM, device=device)  # Generate 16 random images
    with torch.no_grad():
        fake_images = generator(noise).cpu().view(-1, 28, 28)

    # Convert images to base64 and send them back to the frontend
    images = []
    for img in fake_images:
        images.append(image_to_base64(img))
    
    return jsonify({'images': images})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
