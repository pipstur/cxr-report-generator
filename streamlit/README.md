# Chest X-ray AI Assistant (Streamlit)

This repository contains a **Streamlit-based web application** for analyzing **chest X-ray images** using deep learning.
The app performs **multi-label classification**, visual explanation with **Grad-CAM**, and generates a **human-readable summary report** using an LLM.

> ⚠️ **Disclaimer**
> This tool is intended for **research and decision support only**.
> It is **not a medical device** and must not be used as a standalone diagnostic system.

---

## Features

- **14-class multi-label chest X-ray classification**
- **ONNX inference** for fast and portable model execution
- **Grad-CAM visual explanations** highlighting image regions influencing predictions
- **Plain-language explanations** for detected findings
- **AI-generated summary report** using Groq LLM
- Simple, clean **Streamlit UI**

---

## Supported Findings

The model predicts the presence (or absence) of the following findings:

- No Finding
- Enlarged Cardiomediastinum
- Cardiomegaly
- Lung Opacity
- Lung Lesion
- Edema
- Consolidation
- Pneumonia
- Atelectasis
- Pneumothorax
- Pleural Effusion
- Pleural Other
- Fracture
- Support Devices

This is a **multi-label problem**, meaning **multiple findings may be present simultaneously**.

---

## How the App Works

### 1. Image Upload
You upload a chest X-ray image (`.jpg`, `.jpeg`, `.png`) via the web interface.

📸 *Screenshot: Image upload screen*
![Image upload screen](https://i.imgur.com/eR01xYH.png)

---

### 2. Model Inference
- The image is resized and normalized
- Inference is performed using an **ONNX model**
- Each class outputs a probability between 0 and 1
- A sigmoid + threshold (default: 0.5) determines detected findings

📸 *Screenshot: Prediction results*
![Prediction results](https://i.imgur.com/4DtCzom.png)

---

### 3. Prediction Display
For each detected finding, the app shows:
- Finding name
- Probability (%)
- A short, plain-language explanation

This makes results easier to understand without requiring radiology expertise.

---

### 4. Grad-CAM Explanation (Optional)
When enabled, the app:
- Generates Grad-CAM heatmaps for all positive predictions
- Aggregates them into a single visualization
- Overlays the heatmap on the original image

This highlights **which regions of the image influenced the model’s decision**.

📸 *Screenshot: Grad-CAM visualization*
![Grad-CAM visualization](https://i.imgur.com/4vnTSKr.png)

---

### 5. AI-Generated Report
By clicking **“Generate Summary Report”**, the app:
- Sends prediction probabilities to a Groq-hosted LLM
- Produces a concise, structured report with:
  - Key impressions
  - Simple explanations
  - A safety disclaimer

📸 *Screenshot: Generated report*
![Generated Report](https://i.imgur.com/fJaHAdc.png)



## Running the App
While you can always access the app [online](https://chest-x-ray-report-generator.streamlit.app/), you can also run it locally:

0. Get your Groq API key from https://console.groq.com/, export it as an environment variable:
```bash
export GROQ_API_KEY="your_groq_api_key"
```

1. Clone the repository:
```bash
git clone https://github.com/pipstur/cxr-report-generator.git
cd cxr-report-generator/streamlit
```

2. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit/app.py
```

4. Open the provided local URL in your web browser to access the app.
