import os
import torch
import torch.nn.functional as F

upfirdn2d_op = None
if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load
        module_path = os.path.dirname(__file__)
        upfirdn2d_op = load(
            name="upfirdn2d_op",
            sources=[
                os.path.join(module_path, "upfirdn2d.cpp"),
                os.path.join(module_path, "upfirdn2d_kernel.cu"),
            ],
        )
    except Exception as e:
        print(f"Warning: Could not load CUDA upfirdn2d extension, falling back to CPU: {e}")
        upfirdn2d_op = None

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if upfirdn2d_op is not None and input.is_cuda:
        return upfirdn2d_op.upfirdn2d(input, kernel, up, down, pad)
    else:
        # CPU fallback supporting up/down factors
        # Upsample
        if up > 1:
            input = input.reshape(input.shape[0], input.shape[1], input.shape[2], 1, input.shape[3], 1)
            input = torch.nn.functional.pad(input, (0, up - 1, 0, 0, 0, up - 1, 0, 0))
            input = input.view(input.shape[0], input.shape[1], input.shape[2] * up, input.shape[4] * up)
        # Pad
        if isinstance(pad, int):
            pad = (pad, pad, pad, pad)
        elif len(pad) == 2:
            pad = (pad[0], pad[1], pad[0], pad[1])
        input = F.pad(input, (pad[0], pad[1], pad[2], pad[3]))
        b, c, h, w = input.shape
        kh, kw = kernel.shape
        kernel_flip = kernel.flip([0, 1]).unsqueeze(0).unsqueeze(0)
        kernel_flip = kernel_flip.to(input.device, dtype=input.dtype)
        out = F.conv2d(input, kernel_flip.expand(c, 1, kh, kw), groups=c)
        # Downsample
        if down > 1:
            out = out[:, :, ::down, ::down]
        return out


def upfirdn2d_native(
        input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
          :,
          max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
          max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
          :,
          ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)

    return out[:, ::down_y, ::down_x, :]
