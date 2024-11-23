import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from truthful_model import TruthfulLlama
from datasets import load_dataset


model = TruthfulLlama(LlamaForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B'))
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
dataset = load_dataset("truthful_qa", "generation")['validation']
