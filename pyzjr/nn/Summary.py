import sys
import numpy as np
import torch
from collections import OrderedDict

layer_modules = (torch.nn.MultiheadAttention,)

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
