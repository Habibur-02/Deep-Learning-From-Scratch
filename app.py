import gradio as gr
import torch
import timm
import torchvision.transforms as T
from PIL import Image

# ─── Model Loading ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("davit_tiny", pretrained=False, num_classes=2)

# Hugging Face Space এ model সরাসরি same folder এ থাকবে
checkpoint = torch.load("epoch10_model.pth", map_location=device)

model.load_state_dict(
    checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
)

model.to(device)
model.eval()

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


def predict_image(img):
    if img is None:
        return {"Line Plot": 0.0, "Spider Plot": 0.0}
    img = img.convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

    labels = ["Line Plot", "Spider Plot"]
    return {labels[i]: float(probs[0][i]) for i in range(len(labels))}


# ─── Custom CSS ───────────────────────────────────────────────────────────────
custom_css = """

/* ── Page background ── */
body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
    font-family: 'Segoe UI', sans-serif !important;
}

/* ── Main card: transparent ── */
.main-card {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 20px !important;
    padding: 24px 32px 28px 32px !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.55) !important;
    max-width: 880px !important;
    margin: 24px auto !important;
}

/* ── Logo row ── */
.header-logo {
    display: flex !important;
    justify-content: center !important;
    margin-bottom: 6px !important;
}

/* ── Title ── */
.app-title {
    text-align: center;
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
}

/* ── Subtitle ── */
.app-subtitle {
    text-align: center;
    color: rgba(255, 255, 255, 0.50) !important;
    font-size: 0.90rem !important;
    margin: 0 0 14px 0 !important;
    padding: 0 !important;
    line-height: 1.3 !important;
}

/* ── Divider ── */
.divider {
    border: none !important;
    border-top: 1px solid rgba(255, 255, 255, 0.08) !important;
    margin: 10px 0 16px 0 !important;
}

/* ── Section labels ── */
label span, .block > label > span {
    color: rgba(255, 255, 255, 0.70) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Upload box ── */
[data-testid="image"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 2px dashed rgba(167, 139, 250, 0.45) !important;
    border-radius: 14px !important;
    transition: border-color 0.3s ease !important;
}
[data-testid="image"]:hover {
    border-color: rgba(167, 139, 250, 0.85) !important;
}

/* ── Analyze button ── */
#analyze-btn, button.primary, button[variant="primary"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    margin-top: 10px !important;
    cursor: pointer !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.40) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
#analyze-btn:hover, button.primary:hover, button[variant="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124, 58, 237, 0.65) !important;
}

/* ── Label output — ALL text white ── */
.label-container,
[data-testid="label"],
.output-class {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

.label-container *,
[data-testid="label"] * {
    color: #ffffff !important;
}

.label-container svg text,
[data-testid="label"] svg text,
.label-container tspan,
[data-testid="label"] tspan {
    fill: #ffffff !important;
    color: #ffffff !important;
}

/* ── Confidence bar ── */
.label-container .bar,
[data-testid="label"] .bar {
    background: linear-gradient(90deg, #7c3aed, #60a5fa) !important;
    border-radius: 4px !important;
}

/* ── Footer ── */
.footer-text {
    text-align: center;
    color: rgba(255, 255, 255, 0.25) !important;
    font-size: 0.76rem !important;
    margin-top: 18px !important;
    padding-top: 6px !important;
}

"""


# ─── UI Layout ────────────────────────────────────────────────────────────────
with gr.Blocks(css=custom_css) as demo:

    with gr.Column(elem_classes="main-card"):

        # ── Logo ──
        with gr.Row(elem_classes="header-logo"):
            gr.Image(
                "delineate.png",          # same folder এ থাকবে
                width=220,
                show_label=False,
                interactive=False,
                container=False,
            )

        # ── Title & Subtitle ──
        gr.HTML("<h1 class='app-title'>Chart Analysis Classifier</h1>")
        gr.HTML("<p class='app-subtitle'>Powered by Delineate AI Technology</p>")

        gr.HTML("<hr class='divider'>")

        # ── Two-column layout ──
        with gr.Row():

            with gr.Column(scale=1):
                input_img = gr.Image(
                    type="pil",
                    label="Upload Chart Image",
                    height=300,
                )
                predict_btn = gr.Button(
                    "✦  Analyze Image",
                    variant="primary",
                    elem_id="analyze-btn",
                )

            with gr.Column(scale=1):
                output_label = gr.Label(
                    num_top_classes=2,
                    label="Prediction Result",
                )

        gr.HTML("<hr class='divider'>")

        gr.HTML(
            "<p class='footer-text'>© 2026 Delineate AI · Chart Analysis Classifier</p>"
        )

    predict_btn.click(
        fn=predict_image,
        inputs=input_img,
        outputs=output_label,
    )

demo.launch()
