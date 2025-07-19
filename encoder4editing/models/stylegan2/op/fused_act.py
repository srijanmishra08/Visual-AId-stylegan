import os
import torch
import torch.nn as nn
import torch.nn.functional as F

fused = None
if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load
        module_path = os.path.dirname(__file__)
        fused = load(
            name="fused",
            sources=[
                os.path.join(module_path, "fused_bias_act.cpp"),
                os.path.join(module_path, "fused_bias_act_kernel.cu"),
            ],
        )
    except Exception as e:
        print(f"Warning: Could not load CUDA extensions, falling back to CPU: {e}")
        fused = None

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))
        else:
            self.bias = None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if fused is not None and input.is_cuda:
        return FusedLeakyReLUFunction.apply(
            input, bias, negative_slope, scale
        )
    else:
        # CPU fallback
        if bias is not None:
            rest_dim = [1] * (input.ndim - bias.ndim - 1)
            input = input + bias.view(1, bias.shape[0], *rest_dim)
        return F.leaky_relu(input, negative_slope) * scale

class FusedLeakyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        ctx.bias = bias is not None
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        args = [input]
        if bias is not None:
            args.append(bias)
        args.append(empty)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(*args)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        args = ctx.saved_tensors
        input, bias, empty = args[0], args[1] if ctx.bias else None, args[-1]
        grad_input = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = empty.new_empty(input.shape)
        if ctx.bias and ctx.needs_input_grad[1]:
            grad_bias = empty.new_empty(bias.shape)
        fused.fused_bias_act_backward(
            grad_output, grad_input, grad_bias, input, bias, empty, 3, 0, ctx.negative_slope, ctx.scale
        )
        return grad_input, grad_bias, None, None
