import torch
import sys
from pyzjr.nn.models.backbone import *
from pyzjr.nn.devices import load_owned_device

def get_clsnetwork(name, num_classes=None, weights='', device=load_owned_device()):
    if name == 'LeNet':
        net = LeNet(num_classes=10)
    else:
        network_names = [
            'ZFNet', 'AlexNet', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
            'googlenet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'conv2former_n', 'conv2former_t', 'conv2former_s', 'conv2former_l', 'conv2former_b',
            'se_resnet18', 'se_resnet34', 'se_resnet50', 'se_resnet101', 'se_resnet152',
            'squeezenet1_0', 'squeezenet1_1', 'darknet19', 'darknet53',
            'MobileNetV1', 'MobileNetV2', 'MobileNetV3_Large', 'MobileNetV3_Small',
            'densenet121', 'densenet161', 'densenet169', 'densenet201',
            'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'shufflenet_v1_g1',
            'shufflenet_v1_g2', 'shufflenet_v1_g3', 'shufflenet_v1_g4', 'shufflenet_v1_g8',
            'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5',
            'shufflenet_v2_x2_0', 'Xception', 'drn_c_26', 'drn_c_42', 'drn_c_58', 'drn_d_22',
            'drn_d_24', 'drn_d_38', 'drn_d_40', 'drn_d_54', 'drn_d_56', 'drn_d_105', 'drn_d_107',
            'ghostnetv1', 'ghostnetv2', 'g_ghost_regnetx_002', 'g_ghost_regnetx_004',
            'g_ghost_regnetx_006', 'g_ghost_regnetx_008', 'g_ghost_regnetx_016',
            'g_ghost_regnetx_032', 'g_ghost_regnetx_040', 'g_ghost_regnetx_064',
            'g_ghost_regnetx_080', 'g_ghost_regnetx_120', 'g_ghost_regnetx_160',
            'g_ghost_regnetx_320', 'regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008',
            'regnetx_016', 'regnetx_032', 'regnetx_040', 'regnetx_064', 'regnetx_080',
            'regnetx_120', 'regnetx_160', 'regnetx_320', 'fasternet_t0', 'fasternet_t1',
            'fasternet_t2', 'fasternet_s', 'fasternet_m', 'fasternet_l',  'vit_b_16', 'vit_b_32',
            'vit_l_16', 'vit_l_32', 'vit_h_14', 'swin_t', 'swin_s', 'swin_b', 'swin_l'
        ]

        if name not in network_names:
            print('the network name you have entered is not supported yet')
            sys.exit()

        network_class = getattr(sys.modules[__name__], name)
        net = network_class(num_classes=num_classes)

    net.to(device)
    if weights != '':
        print('Load weights {}.'.format(weights))
        model_dict = net.state_dict()
        pretrained_dict = torch.load(weights, map_location=device)

        matched_keys = set(model_dict.keys()) & set(pretrained_dict.keys())
        mismatched_keys = set(model_dict.keys()) - set(pretrained_dict.keys())

        for key in matched_keys:
            if model_dict[key].shape == pretrained_dict[key].shape:
                model_dict[key] = pretrained_dict[key]
            else:
                mismatched_keys.add(key)

        if mismatched_keys:
            print("The following keys have mismatched shapes or are not present in the model:")
            for key in mismatched_keys:
                print(f"- {key}")
        else:
            print("All weights were successfully loaded.")
            net.load_state_dict(model_dict)
    else:
        print("\033[31mNo weights, training will start from scratch")

    return net

if __name__=="__main__":
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = get_clsnetwork(name='vit_l_16', num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))