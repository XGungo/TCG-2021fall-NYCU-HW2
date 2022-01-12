import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter



class NTupleNet(nn.Module):
    def __init__(self, num_ft=4*8, num_ft_elem=5):
        super(NTupleNet, self).__init__()
        self.num_ft = num_ft
        self.fcs = nn.ModuleList([nn.Sequential(nn.Linear(num_ft_elem, 64), nn.Sigmoid()) for _ in range(num_ft)])
        self.fc = nn.Sequential(nn.Linear(64 * num_ft, 1), nn.ReLU())

    def forward(self, inp):
        out = [self.fcs[i](inp[i]) for i in range(self.num_ft)]
        # out = self.fcs(inp)
        out = torch.cat(out).flatten()
        out = self.fc(out)
        return out


if __name__ == "__main__":
    writer = SummaryWriter('runs/tuple_net')

    net = NTupleNet()
    a = torch.randint(0, 25, (4*8, 5)).float()
    print(a.shape)
    print(net(a))
    if torch.cuda.is_available():
        net.cuda()
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
    sm = torch.jit.trace(net, a)
    sm.save("model/basic.pt")
    writer.close()
