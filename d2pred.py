import collections, torch
from matches import *

HEROES_LIST = list(collections.OrderedDict(sorted(HEROES.items(), key=lambda t: t[0])).keys())

class D2Pred(torch.nn.Module):
    def __init__(self):
        super(D2Pred, self).__init__()
        self.hid1 = torch.nn.Sequential(
            torch.nn.Linear(N_HEROES * 2, 512),
            torch.nn.Tanh()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(512 + (N_HEROES * 2), 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        hid1_out = self.hid1(x)
        out_input = torch.cat((x, hid1_out), dim=1)
        output = self.out(out_input)
        return output.reshape(len(x[:,0]))
    
    def init_weights(self, a, b):
        for m in self.hid1.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(a, b)
        
        for m in self.out.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(a, b)

def get_hero_index(id, radiant=True):
    return HEROES_LIST.index(int(id)) + (0 if radiant else N_HEROES)

def get_normalised_output(output):
    norm = (output - 0.5) * 2
    return ("+%7.4f" % norm) if norm > 0 else ("%7.4f" % norm)
