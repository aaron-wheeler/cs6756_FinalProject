import torch
import torch.nn.functional as F # for DQN

# class ImitationNet(torch.nn.Module):
#     def __init__(self, features):
#         super(ImitationNet, self).__init__()

#         self.linear1 = torch.nn.Linear(features, 128)
#         self.linear2 = torch.nn.Linear(128, 128)
#         self.linear3 = torch.nn.Linear(128, 9)
#         # 2-d output, if action = 0 -> [1,0] elif action = 1 -> [0,1]

#         # self.dropout = torch.nn.Dropout()

#     def forward(self, x):
#         x = self.linear1(x)
#         x = torch.relu(x)
#         # x = self.dropout(x)

#         x = self.linear2(x)
#         x = torch.relu(x)
#         # x = self.dropout(x)

#         x = self.linear3(x)
#         return x

class ImitationNet(torch.nn.Module):
    def __init__(self, features):
        super(ImitationNet, self).__init__()

        self.linear1 = torch.nn.Linear(features, 64)
        self.linear2 = torch.nn.Linear(64, 64)
        self.linear3 = torch.nn.Linear(64, 64)
        self.linear4 = torch.nn.Linear(64, 64)
        self.linear5 = torch.nn.Linear(64, 9)

        self.dropout = torch.nn.Dropout(p=0.15) # Define proportion or neurons to dropout

    def forward(self, features):
        activation1 = F.relu(self.linear1(features))
        activation1 = self.dropout(activation1)
        activation2 = F.relu(self.linear2(activation1))
        activation2 = self.dropout(activation2)
        activation3 = F.relu(self.linear3(activation2))
        activation3 = self.dropout(activation3)
        activation4 = F.relu(self.linear4(activation3))
        output = self.linear5(activation4)
        return output

# step functions
def learn(net, loss_fn, opt, states, actions):
    ''' Implement BC method for initial policy and dataset'''
    # forward pass
    pred = net(states)
    # compute prediction error
    loss = loss_fn(pred, actions)
    # backpropagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


# class DQN(torch.nn.Module):
#     def __init__(self, input_dim):
#         super(DQN, self).__init__()
#         self.linear1 = torch.nn.Linear(input_dim, 128)
#         self.linear2 = torch.nn.Linear(128, 128) 
#         self.linear3 = torch.nn.Linear(128, 9)


#     def forward(self, features):
#         activation1 = F.relu(self.linear1(features))
#         activation2 = F.relu(self.linear2(activation1))
#         output = self.linear3(activation2)
#         return output

class DQN(torch.nn.Module):
    def __init__(self, features):
        super(DQN, self).__init__()

        self.linear1 = torch.nn.Linear(features, 64)
        self.linear2 = torch.nn.Linear(64, 64)
        self.linear3 = torch.nn.Linear(64, 64)
        self.linear4 = torch.nn.Linear(64, 64)
        self.linear5 = torch.nn.Linear(64, 9)

    def forward(self, features):
        activation1 = F.relu(self.linear1(features))
        activation2 = F.relu(self.linear2(activation1))
        activation3 = F.relu(self.linear3(activation2))
        activation4 = F.relu(self.linear4(activation3))
        output = self.linear5(activation4)
        return output