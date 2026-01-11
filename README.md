# Chest X-Ray report generator
This project represents an experimental system for automatic radiology report generation from chest X-ray images. It combines a vision encoder for medical image understanding with a language model (GROQ LLM) to generate human-readable clinical reports.

# Project structure
1. `training/` - Contains code and scripts for training of the ViT encoder on chest X-ray images.
2. `data_utils/` - Utilities for data acquirement, loading, preprocessing, and augmentation.
3. `inference/` - Code for running inference with the trained model to generate reports from new chest X-ray images.
4. `streamlit/` - A simple web application for interactive report generation using [Streamlit](https://chest-x-ray-report-generator.streamlit.app/).

# Project setup
The project is intended to be run in a Python environment (3.10). Follow these steps to set up the project:
1. Clone the repository:
   ```bash
   git clone https://github.com/pipstur/cxr-report-generator.git
   ```
2. Install the required dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc
   uv pip install --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match -r training/requirements.txt
   ```
3. Download the models using the provided script:
   ```bash
   bash download_models.sh
   ```

# Project usage
For training, inference, and running the Streamlit app, refer to the respective README files in each subdirectory for detailed instructions.

# Project status
This project is currently in an experimental stage and is not intended for clinical use. Further validation and testing are required before deployment in a healthcare setting.
