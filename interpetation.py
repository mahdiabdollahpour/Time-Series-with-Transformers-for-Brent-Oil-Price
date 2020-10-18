import torch
from models.custom_transformer import CustomTransformer

save_path = 'saved/transformer_checkpoint.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

from models.helpers import get_model

model_class = CustomTransformer
checkpoint = torch.load(save_path)
net, optimizer = get_model(model_class, checkpoint['args'], device, checkpoint['lr'])

net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
