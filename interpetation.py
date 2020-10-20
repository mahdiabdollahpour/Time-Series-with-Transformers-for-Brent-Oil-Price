import torch
from models.custom_transformer import CustomTransformer
import seaborn as sns;

save_path = 'saved/transformer_checkpoint.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
from matplotlib import pyplot as plt
from models.helpers import get_model
import numpy as np
from util import getdata_energy_after_Corona, plot_result, all_energy_data, full_report, energy_return_data, plot_window

# data,_ = getdata_energy_after_Corona(window=10)
data, _, scaler = energy_return_data(window=25)

model_class = CustomTransformer
checkpoint = torch.load(save_path)
net, optimizer = get_model(model_class, checkpoint['args'], device, checkpoint['lr'])

net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# torch.save(checkpoint, save_path)
data = data.astype(np.float32)
i = 1440
a_window = data[i:i + 1, :, :-1]
plot_window(a_window)
a_window = torch.tensor(a_window, device=device)
net.eval()
netout = net(a_window)
# Select first encoding layer
encoder = net.layers_encoding[0]
#
# # Get the first attention map
attn_map = encoder.attention_map[0].detach().cpu()
# from matplotlib import pyplot as plt
# # Plot
from scipy.special import softmax

#
print(attn_map[0,:])
# attn_map = softmax(attn_map, axis=0)
# print(attn_map)
plt.figure(figsize=(20, 20))
sns.heatmap(attn_map)
plt.show()
plt.savefig("attention_map.jpg")
