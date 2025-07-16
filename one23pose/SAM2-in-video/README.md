[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h9oG5QafmFBuJEvdfVAXFT_Iawpb5HOP?usp=sharing)
# SAM2-in-Video

This repository contains code for deploying a Gradio application using the SAM2 model for video processing. The application allows users to interact with the model through a user-friendly web interface.

## Demo 
<p align="center">
  <img src="assets/demo.gif" alt="demo" width="80%">
</p>

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

To set up the project locally, follow these steps:

1. Clone the GitHub repository:
    ```bash
    git clone https://github.com/sujanshresstha/sam2-in-video.git
    cd sam2-in-video
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -e .
    pip install -q -r requirements.txt
    ```

3. Build the project:
    ```bash
    python setup.py build_ext --inplace
    ```

4. Download the necessary model checkpoints:
    ```bash
    cd checkpoints
    ./download_ckpts.sh
    cd ..
    ```

## Usage

To run the Gradio application:

1. Make sure you are in the root directory of the cloned repository.
2. Run the `gradio_app.py` script:
    ```bash
    python gradio_app.py
    ```

This will launch the Gradio interface, where you can interact with the SAM2 model.

### Running the Jupyter Notebook

Alternatively, you can run the entire setup and launch the Gradio app through the provided Jupyter Notebook:

1. Open the `gradio_app.ipynb` file in Jupyter Notebook or JupyterLab.
2. Execute the cells sequentially to set up the environment and launch the app.

## Files

- **`gradio_app.ipynb`**: A Jupyter Notebook containing steps to set up the environment and run the Gradio application.
- **`gradio_app.py`**: A Python script that contains the Gradio interface and the necessary backend logic for model inference.

## Dependencies

This project requires the following dependencies:

- Python 3.7 or higher
- PyTorch
- Gradio
- OpenCV
- Matplotlib
- PIL



