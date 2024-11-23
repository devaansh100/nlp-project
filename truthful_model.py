import torch
import torch.nn as nn
import numpy as np

class TruthfulLlama(nn.Module):
	def __init__(self, model):
		super(TruthfulLlama, self).__init__()
		self.model = model
		weight = torch.from_numpy(np.load('models/logistic_regression_weights.npy')).float()
		bias = torch.from_numpy(np.load('models/logistic_regression_biases.npy')).float()
		self.truthful_head = torch.nn.Linear(self.model.config.hidden_size, 1)
		self.truthful_head.weight.data.copy_(weight)
		self.truthful_head.bias.data.copy_(bias)		

	def forward(self, *args, **kwargs):
		outputs = self.model(*args, **kwargs, output_hidden_states=True)
		outputs.logits += self.truthful_head(outputs.hidden_states[-1])
		return outputs

from transformers import LlamaForCausalLM, AutoTokenizer
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
model = TruthfulLlama(model)
model(tokenizer('test', return_tensors='pt').input_ids)
