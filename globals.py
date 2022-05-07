import torch

max_seq_length = 50
train_batch_size = 32
dev_batch_size = 128
test_batch_size = 128
n_epochs = 30
gradient_accumulation_step = 1
# learning_rate = 1e-3
warmup_proportion = 0.1

ACOUSTIC_DIM = 74
VISUAL_DIM = 47
TEXT_DIM = 768
DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cpu")