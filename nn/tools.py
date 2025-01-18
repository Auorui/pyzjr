"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used to provide summary information of a PyTorch model.
"""
import sys
import numpy as np
import torch
from collections import OrderedDict

layer_modules = (torch.nn.MultiheadAttention,)
import time
import torch.nn as nn
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

def profile(input, ops, n=10, cuda=True):
    """
    Perform performance analysis on the given model and input data.
    Usage:
        model = MyModel().to('cuda')
        input_tensor = torch.randn(1, 3, 512, 512).to('cuda')
        profile(input_tensor, model, n=10)
    """
    results = []
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")
    if torch.cuda.is_available():
        device = "cuda" if cuda else "cpu"
    else:
        device = "cpu"
    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x, ), verbose=False)[0] / 1E9 * 2  # GFLOPs
            except Exception:
                flops = 0
            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception as e:  # no backward method
                        print(e)
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else 'list' for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def summary_1(model, input_data=None, input_data_args=None, input_size=None, input_dtype=torch.FloatTensor,
            batch_size=-1, *args, **kwargs):
    """
    give example input data as least one way like below:
    1. input_data ---> model.forward(input_data)
    2. input_data_args ---> model.forward(*input_data_args)
    3. input_size & input_dtype ---> model.forward(*[torch.rand(2, *size).type(input_dtype) for size in input_size])
    """
    hooks = []
    summary = OrderedDict()

    def register_hook(module):
        def hook(module, inputs, outputs):

            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            key = "%s-%i" % (class_name, module_idx + 1)

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = [batch_size] + list(outputs[0].size())[1:]
                except AttributeError:
                    # pack_padded_seq and pad_packed_seq store feature into data attribute
                    info["out"] = [batch_size] + list(outputs[0].data.size())[1:]
            else:
                info["out"] = [batch_size] + list(outputs.size())[1:]

            info["params_nt"], info["params"] = 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

            summary[key] = info

        # ignore Sequential and ModuleList and other containers
        if isinstance(module, layer_modules) or not module._modules:
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    if input_data is not None:
        x = [input_data]
    elif input_size is not None:
        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *size).type(input_dtype) for size in input_size]
    elif input_data_args is not None:
        x = input_data_args
    else:
        x = []
    try:
        with torch.no_grad():
            # Check the device of the model and move input_data to the same device
            device = next(model.parameters()).device  # Get model device (cuda or cpu)
            if input_data is not None:
                x = [data.to(device) if isinstance(data, torch.Tensor) else data for data in x]
            else:
                x = [torch.rand(2, *size).type(input_dtype).to(device) for size in input_size]
            model(*x) if not (kwargs or args) else model(*x, *args, **kwargs)
    except Exception:
        # This can be usefull for debugging
        print("Failed to run summary...")
        raise
    finally:
        for hook in hooks:
            hook.remove()
    summary_logs = []
    summary_logs.append("--------------------------------------------------------------------------")
    line_new = "{:<30}  {:>20} {:>20}".format("Layer (type)", "Output Shape", "Param #")
    summary_logs.append(line_new)
    summary_logs.append("==========================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # layer, output_shape, params
        line_new = "{:<30}  {:>20} {:>20}".format(
            layer,
            str(summary[layer]["out"]),
            "{0:,}".format(summary[layer]["params"] + summary[layer]["params_nt"])
        )
        total_params += (summary[layer]["params"] + summary[layer]["params_nt"])
        total_output += np.prod(summary[layer]["out"])
        trainable_params += summary[layer]["params"]
        summary_logs.append(line_new)

    # assume 4 bytes/number
    if input_data is not None:
        total_input_size = abs(sys.getsizeof(input_data) / (1024 ** 2.))
    elif input_size is not None:
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    else:
        total_input_size = 0.0
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_logs.append("==========================================================================")
    summary_logs.append("Total params: {0:,}".format(total_params))
    summary_logs.append("Trainable params: {0:,}".format(trainable_params))
    summary_logs.append("Non-trainable params: {0:,}".format(total_params - trainable_params))
    summary_logs.append("--------------------------------------------------------------------------")
    summary_logs.append("Input size (MB): %0.6f" % total_input_size)
    summary_logs.append("Forward/backward pass size (MB): %0.6f" % total_output_size)
    summary_logs.append("Params size (MB): %0.6f" % total_params_size)
    summary_logs.append("Estimated Total Size (MB): %0.6f" % total_size)
    summary_logs.append("--------------------------------------------------------------------------")

    summary_info = "\n".join(summary_logs)

    print(summary_info)
    return summary_info

def summary_2(model, input_size, *args, **kwargs):
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)

    Args:
        model (Module): Model to summarize
        input_size (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function
    """
    def register_hook(module):
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            key = "{}_{}".format(module_idx, cls_name)

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                info["out"] = list(outputs[0].size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params"], info["macs"] = 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement()

                if name == "weight":
                    ksize = list(param.size())
                    # to make [in_shape, out_shape, ksize, ksize]
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    # ignore N, C when calculate Mult-Adds in ConvNd
                    if "Conv" in cls_name:
                        info["macs"] += int(param.nelement() * np.prod(info["out"][2:]))
                    else:
                        info["macs"] += param.nelement()

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        # ignore Sequential and ModuleList
        if not module._modules:
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)
    with torch.no_grad():
        model(input_size) if not (kwargs or args) else model(input_size, *args, **kwargs)

    for hook in hooks:
        hook.remove()

    print("-"*100)
    print("{:<15} {:>20} {:>20} {:>20} {:>20}"
          .format("Layer", "Kernel Shape", "Output Shape",
                  "# Params (K)", "# Mult-Adds (M)"))
    print("="*100)

    total_params, total_macs = 0, 0
    for layer, info in summary.items():
        repr_ksize = str(info["ksize"])
        repr_out = str(info["out"])
        repr_params = info["params"]
        repr_macs = info["macs"]

        if isinstance(repr_params, (int, float)):
            total_params += repr_params
            repr_params = "{0:,.2f}".format(repr_params/1000)
        if isinstance(repr_macs, (int, float)):
            total_macs += repr_macs
            repr_macs = "{0:,.2f}".format(repr_macs/1000000)

        print("{:<15} {:>20} {:>20} {:>20} {:>20}"
              .format(layer, repr_ksize, repr_out, repr_params, repr_macs))

        # for RNN, describe inner weights (i.e. w_hh, w_ih)
        for inner_name, inner_shape in info["inner"].items():
            print("  {:<13} {:>20}".format(inner_name, str(inner_shape)))

    print("="*100)
    print("# Params:    {0:,.2f}K".format(total_params/1000))
    print("# Mult-Adds: {0:,.2f}M".format(total_macs/1000000))
    print("-"*100)



if __name__ == "__main__":
    import torchvision.models.resnet as models
    model = models.resnet18()
    input_size = (3, 224, 224)
    summary_1(model, input_size=input_size)
    summary_2(model, input_size=torch.ones((1, 3, 224, 224)))
    model = model.to('cpu')
    input_tensor = torch.randn(1, 3, 512, 512).to('cpu')
    results = profile(input_tensor, model, n=10)