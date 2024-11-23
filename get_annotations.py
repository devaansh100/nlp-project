from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def annotate(question, answer, is_true, tokenizer):
    question_tokens = [tokenizer.bos_token_id] + tokenizer(question, add_special_tokens=False).input_ids
    answer_tokens = tokenizer(answer, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    labels = [1] * len(question_tokens) + [int(is_true)] * len(answer_tokens)
    return torch.tensor(question_tokens + answer_tokens).long(), torch.tensor(labels).long()

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # Load and preprocess dataset
    dataset = load_dataset("truthful_qa", "generation")['validation']
        
    # Split data into train/val
    test_indices = torch.rand(len(dataset)) < 0.8
    indices = {
        'test': torch.where(test_indices)[0],
        'train': torch.where(~test_indices)[0]
    }
    train_tokens, train_labels = [], []
    val_tokens, val_labels = [], []
    for split in indices:
        for i in tqdm(indices[split], desc = f'Tokenizing {split}'):
            for answer, is_true in zip(dataset['correct_answers'][i] + dataset['incorrect_answers'][i], [True] * len(dataset['correct_answers'][i]) + [False] * len(dataset['incorrect_answers'][i])):
                tokens, labels = annotate(dataset['question'][i], answer, is_true, tokenizer)
                if split == 'train':
                    train_tokens.append(tokens)
                    train_labels.append(labels)
                else:
                    val_tokens.append(tokens)
                    val_labels.append(labels)

    train_tokens, train_labels = pad_sequence(train_tokens, batch_first=True, padding_value=tokenizer.eos_token_id), pad_sequence(train_labels, batch_first=True, padding_value=-1)
    val_tokens, val_labels = pad_sequence(val_tokens, batch_first=True, padding_value=tokenizer.eos_token_id), pad_sequence(val_labels, batch_first=True, padding_value=-1)

    np.save('data/train_tokens.npy', train_tokens.numpy())
    np.save('data/train_labels.npy', train_labels.numpy())
    np.save('data/val_tokens.npy', val_tokens.numpy())
    np.save('data/val_labels.npy', val_labels.numpy())


if __name__ == "__main__":
    main()
