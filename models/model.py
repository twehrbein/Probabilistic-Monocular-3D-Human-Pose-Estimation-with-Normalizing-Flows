from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

import torch
import torch.nn as nn
import config as c


class poseINN(nn.Module):

    def __init__(self):
        super().__init__()
        self.cond_in_size = c.COND_JOINTS * c.COND_LENGTH
        self.cond_out_size = 56
        self.inn = self.build_inn()

        self.trainable_parameters = [p for p in self.inn.parameters() if p.requires_grad]
        for param in self.trainable_parameters:
            param.data = 0.05 * torch.randn_like(param)

        self.cond_net = self.subnet_cond(self.cond_out_size)
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=c.lr, betas=(0.5, 0.9))
        self.optimizer.add_param_group({"params": self.cond_net.parameters(), "lr": c.lr})
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=150, gamma=0.5)

    def subnet_cond(self, c_out):
        # takes as input a batch of fitted Gaussians
        return nn.Sequential(nn.Linear(self.cond_in_size, 256), nn.ReLU(),
                             nn.Linear(256, c_out))

    def build_inn(self):

        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, 1028), nn.ReLU(),
                                 nn.Linear(1028, c_out))

        nodes = [InputNode(c.ndim_x, name='input')]
        cond = ConditionNode(self.cond_out_size, name='condition')

        for k in range(8):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock,
                              {'subnet_constructor': subnet_fc, 'clamp': c.clamp},
                              conditions=cond,
                              name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed': k},
                              name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name='output'))

        return ReversibleGraphNet(nodes + [cond], verbose=False)

    def save(self, path):
        # do not save unnecessary tmp variables..
        filtered_dict = {k: v for k, v in self.state_dict().items() if 'tmp_var' not in k}
        torch.save({'net': filtered_dict}, path)

    def load(self, path, device):
        state_dicts = torch.load(path, map_location=device)
        network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
        self.load_state_dict(network_state_dict)
        print("weights of trained model loaded")
        try:
            self.optimizer.load_state_dict(state_dicts['opt'])
        except:
            print('Cannot load optimizer for some reason or other')

    def forward(self, x, cond):
        return self.inn(x, self.cond_net(cond))

    def reverse(self, y_rev, cond):
        return self.inn(y_rev, self.cond_net(cond), rev=True)
