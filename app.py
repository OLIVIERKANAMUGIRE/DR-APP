# Libraries 
#*************************************
from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
import os
from models import network, NODEs, small_NODEs, dataset


#Instantiate the flask app
#****************************************
app = Flask(__name__)

# set the device
device = torch.device("cpu")

# Load the models 
model = network.Autoencoder(input_channels=3, latent_channels=16)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()
##############################
#Preprocessing functions

def load_and_preprocess(image_path, target_size=(512, 512), device="cpu"):
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[1], target_size[0]))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = [clahe.apply(img[:, :, i]) for i in range(3)]
    enhanced = np.stack(channels, axis=-1).astype(np.float32) / 255.0

    img_pil = Image.fromarray((enhanced * 255).astype(np.uint8))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    return img_tensor

def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


###########################################

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    imagefile = request.files.get("imagefile")
    if not imagefile or imagefile.filename == "":
        return "No file uploaded", 400

    # Save uploaded file
    upload_folder = "./images"
    os.makedirs(upload_folder, exist_ok=True)
    imagepath = os.path.join(upload_folder, imagefile.filename)
    imagefile.save(imagepath)

    # Preprocess
    img_tensor = load_and_preprocess(imagepath, device=device)

    # Encoder model
    with torch.no_grad():
        _, z0 = model(img_tensor)

    # Latent ODE
    channels = z0.shape[1]
    ode_func = small_NODEs.ConvLatentODEFunc(channels=channels)
    latent_ode = NODEs.LatentODEModel(ode_func, method="dopri5", rtol=1e-5, atol=1e-5)
    t = torch.linspace(0, 0.5, steps=5)
    with torch.no_grad():
        z_t = latent_ode(z0, t)
        z_final = z_t[-1]

        # Decode
        x_final = model.decoder(model.from_latent(z_final))
        x_final = x_final.squeeze(0).permute(1, 2, 0)
        x_final = (x_final * 255).byte().cpu().numpy()
        recon_image = Image.fromarray(x_final)

    # Convert processed input to base64
    processed_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    processed_image = Image.fromarray(processed_np)
    processed_b64 =pil_to_base64(processed_image)

    recon_b64 = pil_to_base64(recon_image)

    return render_template("index.html",
                           processed_image=processed_b64,
                           reconstructed_image=recon_b64)

####################################################################
if __name__ == "__main__":
    app.run(port=3000, debug=True)
