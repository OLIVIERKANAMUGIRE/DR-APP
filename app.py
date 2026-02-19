# Libraries 
from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models import network, NODEs, small_NODEs, dataset

app = Flask(__name__)
device = torch.device("cpu")

# Load model
model = network.Autoencoder(input_channels=3, latent_channels=16)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

# ─── Preprocessing ────────────────────────────────────────────────────────────

def load_and_preprocess(image_path, target_size=(512, 512), device="cpu"):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[1], target_size[0]))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = [clahe.apply(img[:, :, i]) for i in range(3)]
    enhanced = np.stack(channels, axis=-1).astype(np.float32) / 255.0
    img_pil = Image.fromarray((enhanced * 255).astype(np.uint8))
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img_pil).unsqueeze(0).to(device)

def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def fig_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="PNG", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def tensor_to_pil(t):
    """Convert a [C, H, W] tensor (0-1) to a PIL Image."""
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# ─── Analysis helpers ─────────────────────────────────────────────────────────

def make_progression_strip(model, z_t):
    """
    z_t: [T, 1, C, H, W]  — all ODE time steps
    Returns base64 PNG of a horizontal strip of decoded frames.
    """
    n_steps = z_t.shape[0]
    frames = []
    with torch.no_grad():
        for i in range(n_steps):
            z_i = z_t[i]                              # [1, C, H, W]
            x_i = model.decoder(model.from_latent(z_i))  # [1, 3, H, W]
            frames.append(tensor_to_pil(x_i.squeeze(0)))

    # Build a single wide image
    w, h = frames[0].size
    thumb_w = 220
    thumb_h = int(h * thumb_w / w)
    strip_img = Image.new("RGB", (thumb_w * n_steps + 10 * (n_steps - 1), thumb_h), (20, 20, 30))
    for idx, frame in enumerate(frames):
        strip_img.paste(frame.resize((thumb_w, thumb_h)), (idx * (thumb_w + 10), 0))

    return pil_to_base64(strip_img), [pil_to_base64(f.resize((thumb_w, thumb_h))) for f in frames]


def make_latent_pca(z0):
    """
    z0: [1, C, H, W] latent tensor
    Runs PCA on the C channel maps (each flattened to H*W points).
    Returns base64 PNG of a styled PCA scatter plot.
    """
    z_np = z0.squeeze(0).cpu().numpy()  # [C, H, W]
    C, H, W = z_np.shape

    # Each spatial location is a point in C-dim space
    spatial_vectors = z_np.reshape(C, -1).T  # [H*W, C]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(spatial_vectors)  # [H*W, 2]

    # Color by distance from origin in latent space (proxy for "activation strength")
    magnitudes = np.linalg.norm(spatial_vectors, axis=1)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    sc = ax.scatter(
        coords[::4, 0], coords[::4, 1],   # downsample for speed
        c=magnitudes[::4],
        cmap="plasma",
        s=3,
        alpha=0.7,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Activation magnitude", color="#aaaacc", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#aaaacc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaacc")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC 1  ({var[0]*100:.1f}% var)", color="#aaaacc", fontsize=10)
    ax.set_ylabel(f"PC 2  ({var[1]*100:.1f}% var)", color="#aaaacc", fontsize=10)
    ax.set_title("Latent Space — PCA of z₀", color="#e0e0ff", fontsize=12, pad=10)
    ax.tick_params(colors="#555577")
    for spine in ax.spines.values():
        spine.set_edgecolor("#222244")

    fig.tight_layout()
    return fig_to_base64(fig), round(float(var[0]) * 100, 1), round(float(var[1]) * 100, 1)


def make_channel_heatmaps(z0, n_show=8):
    """
    Show the first n_show latent channel activations as heatmaps.
    Returns base64 PNG.
    """
    z_np = z0.squeeze(0).cpu().numpy()  # [C, H, W]
    n_show = min(n_show, z_np.shape[0])
    cols = 4
    rows = int(np.ceil(n_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), facecolor="#0d0d1a")
    axes = axes.flatten()

    for i in range(n_show):
        ax = axes[i]
        ch = z_np[i]
        im = ax.imshow(ch, cmap="inferno", aspect="auto")
        ax.set_title(f"ch {i}", color="#aaaacc", fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6, colors="#777799")

    for j in range(n_show, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Latent Channel Activations (z₀)", color="#e0e0ff", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig_to_base64(fig)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    imagefile = request.files.get("imagefile")
    if not imagefile or imagefile.filename == "":
        return "No file uploaded", 400

    upload_folder = "./images"
    os.makedirs(upload_folder, exist_ok=True)
    imagepath = os.path.join(upload_folder, imagefile.filename)
    imagefile.save(imagepath)

    img_tensor = load_and_preprocess(imagepath, device=device)

    with torch.no_grad():
        _, z0 = model(img_tensor)

    # ── Latent ODE ─────────────────────────────────────────────────────────
    channels = z0.shape[1]
    ode_func = small_NODEs.ConvLatentODEFunc(channels=channels)
    latent_ode = NODEs.LatentODEModel(ode_func, method="dopri5", rtol=1e-5, atol=1e-5)
    t = torch.linspace(0, 0.5, steps=5)

    with torch.no_grad():
        z_t = latent_ode(z0, t)   # [T, 1, C, H, W]
        z_final = z_t[-1]
        x_final = model.decoder(model.from_latent(z_final))
        recon_image = tensor_to_pil(x_final.squeeze(0))

    # ── Base outputs ────────────────────────────────────────────────────────
    processed_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    processed_b64 = pil_to_base64(Image.fromarray(processed_np))
    recon_b64 = pil_to_base64(recon_image)

    # ── Analysis ─────────────────────────────────────────────────────────────
    _, frame_b64s = make_progression_strip(model, z_t)          # per-frame list
    pca_b64, pc1_var, pc2_var = make_latent_pca(z0)
    heatmap_b64 = make_channel_heatmaps(z0)

    t_labels = [f"t={v:.2f}" for v in torch.linspace(0, 0.5, steps=5).tolist()]

    progression_data = list(zip(frame_b64s, t_labels))

    return render_template(
        "index.html",
        processed_image=processed_b64,
        reconstructed_image=recon_b64,
        # progression
        progression_data=progression_data,
        # latent
        pca_plot=pca_b64,
        pc1_var=pc1_var,
        pc2_var=pc2_var,
        heatmap_plot=heatmap_b64,
    )


if __name__ == "__main__":
    app.run(port=3000, debug=True)
