from torch.autograd import grad_mode
from torch.autograd import Function
import torch.nn.functional as F
import torch

class TemperatureSoftmax(Function):

    @staticmethod
    def forward(ctx, inputs, dim=None, temperature=1):
        output = F.softmax(inputs*temperature, dim=dim).requires_grad_()
        softmax_grad = torch.autograd.grad(output, inputs, grad_outputs=(torch.ones_like(output)), retain_graph=True)
        softmax_grad_temperature = softmax_grad / temperature
        ctx.save_for_backward(inputs, output, softmax_grad_temperature)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, output, softmax_grad_temperature= ctx.saved_tensors
        grad_input = grad_dim = grad_temperature = None
        return  grad_input, grad_dim, grad_temperature

temperature_softmax = TemperatureSoftmax.apply

if __name__ == "__main__":
    a = torch.randn(32, 64).requires_grad_()
    label = torch.randint(10, (32,))
    b = torch.randn(64, 10).requires_grad_()
    c = a.mm(b)
    logits = torch.log(temperature_softmax(c)) # 等价于log_softmax
    loss_fn = torch.nn.NLLLoss()
    loss = loss_fn(logits, label)
    
    loss2 = F.cross_entropy(c, label)
    loss2.backward()
    print()