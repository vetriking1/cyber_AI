import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def predict(text, model, tokenizer, device, max_length=128):
    """
    Performs inference on a single input text using the trained model.
    """
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class

# Loading the model
def load_model(model, model_path, device):
    """
    Loads the model weights from a specified path.
    """
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f"Model loaded from {model_path}")
    return model



if __name__ == "__main__":
    # Load the model and tokenizer
    # Example usage in test.py for inference
    model_path = "trained_model.pth"  # Path to the saved model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Load the model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


    # Perform inference
    sample_text = "Sample text describing the cyber crime"  # Replace with your text input
    prediction = predict(sample_text, model, tokenizer, device)
    print(f"Predicted Class: {prediction}")
