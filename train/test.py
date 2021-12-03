import torch


func_mse_1 = torch.nn.MSELoss()
func_mse_2 = torch.nn.functional.mse_loss

b = 3
dim = 1000

x1 = torch.rand((b, dim, dim))
x2 = torch.rand((b, dim, dim))


y1 = func_mse_1(x1, x2)
y2 = func_mse_2(x1, x2)

sample_loss = 0
for x,y in zip(x1, x2):
	sample_loss += (((x - y)**2).sum() / (dim**2))
sample_loss /= b