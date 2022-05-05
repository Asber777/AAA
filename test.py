import torch as t
import torch.nn.functional as F
other_input = t.randn(1, 15)
other_p = t.zeros([15, 10], requires_grad=True, dtype=t.float32)
other_optimizer = t.optim.Adam([other_p], lr = 0.01)
other_target = t.empty(1,dtype = t.long).random_(10) 
other_probs = F.softmax(other_input @ other_p , dim=1)

input = t.randn(1, 15)
p = t.zeros([15, 10], requires_grad=True, dtype=t.float32)
optimizer = t.optim.Adam([p], lr = 0.01)
target = t.empty(1,dtype = t.long).random_(10) 
probs = F.softmax(input @ p , dim=1)
loss = F.cross_entropy(probs, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()


other_loss = F.cross_entropy(other_probs, other_target)
other_optimizer.zero_grad()
other_loss.backward()
other_optimizer.step()