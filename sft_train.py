from module import *

model = Transformer(Config('params/config.json'))
print(f"model parameter size: {model.parameter_size():,}")

bsz = 1
n_dim = 1024
m_len = 512

x = torch.randint(0, 100, size=(bsz, m_len))
y = model(x)
