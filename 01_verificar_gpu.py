import torch
print(torch.cuda.is_available())  # Debería decir "True"
print(torch.cuda.get_device_name(0)) # Debería decir "NVIDIA GeForce RTX 4060"