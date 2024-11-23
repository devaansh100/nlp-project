import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

def extract_token_embeddings(model, tokenizer, tokens, labels, device):
    embeddings = []
    flattened_labels = []

    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(tokens), batch_size)):
            batch_labels = labels[i:i+batch_size]
            inputs = torch.from_numpy(tokens[i:i+batch_size]).to(device)
            attention_mask = (inputs != tokenizer.eos_token_id).long().to(device)
            
            outputs = model(inputs, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            valid_mask = batch_labels != -1
            embeddings.append(token_embeddings[valid_mask].cpu().numpy())
            flattened_labels.append(batch_labels[valid_mask])

    return np.concatenate(embeddings), np.concatenate(flattened_labels)

def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B")
    model.eval()

    # Use the appropriate device for Apple Silicon
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print(f"Using device: {device}")
    model.to(device)

    # Load annotated data
    train_tokens = np.load('data/train_tokens.npy')
    train_labels = np.load('data/train_labels.npy')
    val_tokens = np.load('data/val_tokens.npy')
    val_labels = np.load('data/val_labels.npy')

    # Extract token embeddings and labels for training data
    print("Extracting token embeddings for training data...")
    train_features, train_targets = extract_token_embeddings(model, tokenizer, train_tokens, train_labels, device)

    # # Extract token embeddings and labels for validation data
    # print("Extracting token embeddings for validation data...")
    # val_features, val_targets = extract_token_embeddings(model, tokenizer, val_tokens, val_labels, device)

    # Check for any remaining -1 labels
    print(f"Unique labels in train set: {np.unique(train_targets)}")
    # print(f"Unique labels in val set: {np.unique(val_targets)}")

    # Ensure no -1 labels
    assert -1 not in train_targets, "Found -1 labels in training data"
    # assert -1 not in val_targets, "Found -1 labels in validation data"

    # Train a logistic regression classifier
    print("Training logistic regression model...")
    classifier = LogisticRegression(max_iter=100)
    classifier.fit(train_features, train_targets)

    # # Validate the classifier
    # val_predictions = classifier.predict(val_features)
    # print("Validation Accuracy:", accuracy_score(val_targets, val_predictions))
    # print(classification_report(val_targets, val_predictions))

    # Save the classifier's weights and biases
    print("Saving the classifier's weights and biases...")
    np.save('models/logistic_regression_weights.npy', classifier.coef_)
    np.save('models/logistic_regression_biases.npy', classifier.intercept_)

    # # Optionally save the entire classifier model for later use
    # print("Saving the entire classifier...")
    # import pickle
    # with open('logistic_classifier.pkl', 'wb') as f:
    #     pickle.dump(classifier, f)

if __name__ == "__main__":
    main()