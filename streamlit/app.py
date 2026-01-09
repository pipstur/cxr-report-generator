import rootutils

root = rootutils.setup_root(
    search_from=__file__,
    indicator=[".pre-commit-config.yaml", ".git", ".github"],
    pythonpath=True,
    dotenv=True,
)
import os
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import torch
from groq import Groq
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import streamlit as st
from training.src.models.efficientformer import EfficientFormer

MODEL_PATH_CKPT = "models/epoch_001.ckpt"
MODEL_FOLDER_ONNX = "models/"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
PROVIDERS = ["CPUExecutionProvider"]


@st.cache_resource
def load_torch_model(ckpt_path: str) -> EfficientFormer:
    model = EfficientFormer.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad = True
    return model


@st.cache_resource
def load_onnx_models(models_folder: str) -> List[ort.InferenceSession]:
    sessions = []
    for model_name in os.listdir(models_folder):
        if model_name.endswith(".onnx"):
            path = os.path.join(models_folder, model_name)
            sessions.append(ort.InferenceSession(path, providers=PROVIDERS))
    return sessions


def preprocess_image(image: Image.Image) -> Image.Image:
    return image.convert("L").resize(IMAGE_SIZE)


def prepare_torch_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def prepare_onnx_tensor(image: Image.Image) -> np.ndarray:
    return prepare_torch_tensor(image).numpy()


def predict_image(sessions: List[ort.InferenceSession], tensor: np.ndarray) -> List[np.ndarray]:
    predictions = []
    for session in sessions:
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: tensor})
        predictions.append(outputs[0].squeeze())
    return predictions


def safe_sigmoid(x, clip_value=50):
    x = np.clip(x, -clip_value, clip_value)  # prevents huge exponentials
    return 1 / (1 + np.exp(-x))


def predict_multilabel_single(
    logits: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    probs = safe_sigmoid(logits)
    preds = probs >= threshold
    return preds, probs


def generate_gradcam(
    model: EfficientFormer, tensor: torch.Tensor, target_class: int, target_layer: torch.nn.Module
) -> np.ndarray:
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale = cam(tensor, targets=[ClassifierOutputTarget(target_class)])
    return grayscale[0]


def overlay_cam(image: Image.Image, cam: np.ndarray) -> np.ndarray:
    img_np = np.array(image.resize(IMAGE_SIZE)) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    return show_cam_on_image(img_np, cam, use_rgb=True)


def generate_report_with_groq(
    probs: np.ndarray, model: str = "llama-3.1-8b-instant"
) -> str | None:
    print(probs)
    client = Groq()
    class_info = [f"{CLASS_NAMES[i]} (Probability: {prob:.2f})" for i, prob in enumerate(probs)]
    prompt = (
        "You are a radiology assistant. Given the following model predictions for a chest X-ray, "
        "write a concise, human-readable report. Include the most important findings, "
        "highlight any serious conditions, and explain in plain language if possible.\n\n"
        f"Predictions:\n{class_info}"
    )

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
    )
    return completion.choices[0].message.content


def main():
    st.title("🔬 14-Class Image Classification with Grad-CAM")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    show_explanation = st.checkbox("Show Grad-CAM explanation")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width="stretch")

        onnx_tensor = prepare_onnx_tensor(preprocess_image(image))
        sessions = load_onnx_models(MODEL_FOLDER_ONNX)
        logits = predict_image(sessions, onnx_tensor)[0]
        preds, probs = predict_multilabel_single(logits, threshold=0.5)

        st.write("### Predictions")
        for i, (p, prob) in enumerate(zip(preds, probs)):
            if p:
                st.write(f"- **{CLASS_NAMES[i]}**: {prob * 100:.2f}%")

        if show_explanation:
            torch_model = load_torch_model(MODEL_PATH_CKPT)
            torch_tensor = prepare_torch_tensor(preprocess_image(image))
            target_layer = torch_model.feature_extractor.stages[3].blocks[-1].token_mixer.proj.conv

            positive_classes = np.where(preds)[0]
            cams = [
                generate_gradcam(torch_model, torch_tensor, c, target_layer)
                for c in positive_classes
            ]
            if cams:
                agg_cam = np.mean(cams, axis=0)
                visualization = overlay_cam(image, agg_cam)
                st.image(visualization, caption="Grad-CAM (aggregated)", width="stretch")

        if st.button("Generate Radiology Report"):
            with st.spinner("Generating report..."):
                report = generate_report_with_groq(probs)
            st.write("### Radiology Report")
            st.text(report)


if __name__ == "__main__":
    main()
