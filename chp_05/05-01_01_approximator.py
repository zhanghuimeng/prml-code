# Figure 5.3: Using a MLP to approximate 4 functions

import numpy as np
import torch
import matplotlib.pyplot as plt

def f1(x): return x**2

def f2(x): return torch.sin(x)

def f3(x): return torch.abs(x)

def f4(x):
    # x = x.data.numpy()
    return np.heaviside(x, 0)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        y1 = torch.tanh(self.hidden(x))
        y2 = self.predict(y1)
        return y1, y2

x = torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1)
for f in [f1, f2, f3, f4]:
    # Generate data
    y = f(x)
    # Build Network
    net = Net(n_feature=1, n_hidden=3, n_output=1)
    print(net)
    # Training
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()
    for i in range(500):
        _, y2 = net(x)
        loss = loss_func(y, y2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Visualization
    y1, y2 = net(x)
    plt.cla()
    plt.plot(x.data.numpy(), y1.data.numpy().take(0, 1), "--", color="green", linewidth=1)
    plt.plot(x.data.numpy(), y1.data.numpy().take(1, 1), "--", color="yellow", linewidth=1)
    plt.plot(x.data.numpy(), y1.data.numpy().take(2, 1), "--", color="magenta", linewidth=1)
    plt.plot(x.data.numpy(), y2.data.numpy(), color="red", linewidth=1)
    plt.scatter(x.data.numpy(), y.data.numpy(), color="blue")
    plt.show()
