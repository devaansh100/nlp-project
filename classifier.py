import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def extract_token_embeddings(model, tokenizer, tokens, labels, device):
    """
    Extract token embeddings and their corresponding labels efficiently.
    Assumes all input sequences are of the same length.
    """
    embeddings = []
    flattened_labels = []

    # Process tokens in batches
    batch_size = 8  # Adjust based on your GPU memory
    
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch_tokens = tokens[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Convert to tensors
            inputs = torch.tensor(batch_tokens, dtype=torch.long).to(device)
            
            # Create attention mask
            if tokenizer.pad_token_id is None:
                # If there's no pad token, assume all tokens are valid
                attention_mask = torch.ones_like(inputs, dtype=torch.long).to(device)
            else:
                attention_mask = (inputs != tokenizer.eos_token_id).long().to(device)
            
            # Forward pass
            outputs = model(inputs, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            
            # Process each sequence in the batch
            for emb, label, mask in zip(token_embeddings, batch_labels, attention_mask):
                mask = mask.bool()
                embeddings.append(emb[mask].cpu().numpy())
                flattened_labels.append(label[mask.cpu().numpy()])

    return np.concatenate(embeddings), np.concatenate(flattened_labels)


def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B")
    model.eval()

    # Use the appropriate device for Apple Silicon
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Load annotated data
    train_tokens = np.load('train_tokens.npy')
    train_labels = np.load('train_labels.npy')
    val_tokens = np.load('val_tokens.npy')
    val_labels = np.load('val_labels.npy')

    # Extract token embeddings and labels for training data
    print("Extracting token embeddings for training data...")
    train_features, train_targets = extract_token_embeddings(model, tokenizer, train_tokens, train_labels, device)

    # Extract token embeddings and labels for validation data
    print("Extracting token embeddings for validation data...")
    val_features, val_targets = extract_token_embeddings(model, tokenizer, val_tokens, val_labels, device)

    # Train a logistic regression classifier
    print("Training logistic regression model...")
    classifier = LogisticRegression(max_iter=100)
    classifier.fit(train_features, train_targets)

    # Validate the classifier
    val_predictions = classifier.predict(val_features)
    print("Validation Accuracy:", accuracy_score(val_targets, val_predictions))
    print(classification_report(val_targets, val_predictions))

    # Save the classifier's weights and biases
    print("Saving the classifier's weights and biases...")
    np.save('logistic_regression_weights.npy', classifier.coef_)
    np.save('logistic_regression_biases.npy', classifier.intercept_)

    # Optionally save the entire classifier model for later use
    print("Saving the entire classifier...")
    import pickle
    with open('logistic_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)

if __name__ == "__main__":
    main()