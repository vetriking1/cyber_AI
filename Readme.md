# Cyber Crime Classification using DistilBERT

This project uses the `DistilBERT` model to classify different types of cybercrimes based on textual descriptions. The classification is divided into four categories:
- Digital Infrastructure Crimes
- Financial Crimes
- Other Cyber Crimes
- Sexual Crimes

## Model Overview
The model leverages the **DistilBERT** transformer architecture fine-tuned for sequence classification. It uses textual descriptions of crimes to predict one of the four categories.

## Training Results
- **Training Loss**: 0.7020
- **Validation Loss**: 0.6105
- **Validation Accuracy**: 74.67%

### Classification Report
| Class                           | Precision | Recall | F1-Score | Support |
|----------------------------------|-----------|--------|----------|---------|
| Digital Infrastructure Crimes    | 0.92      | 0.68   | 0.78     | 65      |
| Financial Crimes                 | 0.74      | 0.98   | 0.84     | 582     |
| Other Cyber Crimes               | 0.63      | 0.34   | 0.44     | 273     |
| Sexual Crimes                    | 1.00      | 0.39   | 0.56     | 80      |

**Overall Accuracy**: 73.6%

## Setup and Requirements

To set up this project and run it locally, ensure you have Python 3.6+ installed. You can create a virtual environment to manage dependencies.

### Steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/cybercrime-classification.git
    cd cybercrime-classification
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - **Windows**:
      ```bash
      .\venv\Scripts\activate
      ```
    - **Linux/macOS**:
      ```bash
      source venv/bin/activate
      ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5. To train the model, run `train.py`:
    ```bash
    python train.py
    ```

6. To perform inference with a trained model, run `test.py`:
    ```bash
    python test.py
    ```

### Files
- `train.py`: Script for training the model.
- `test.py`: Script for performing inference.
- `model.pth`: Saved model file.
- `data/`: Directory for data preprocessing and storage.
- `utils.py`: Utility functions for text processing and data handling.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
