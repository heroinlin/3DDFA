import torch
from collections import OrderedDict


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for name, module1 in module._modules.items():
            recursion_change_bn(module1)
    return module


def pth_to_jit(model, save_path, device="cuda:0"):
    model.eval()
    input_x = torch.randn(1, 3, 160, 160).to(device)
    new_model = torch.jit.trace(model, input_x)
    torch.jit.save(new_model, save_path)


def jit_to_onnx(jit_model_path, onnx_model_path):
    model = torch.jit.load(jit_model_path, map_location=torch.device('cuda:0'))
    model.eval()
    example_input = torch.randn(1, 3, 160, 160).to("cuda:0")
    example_output = torch.rand(1, 805, 1).to("cuda:0"), torch.randn(1, 805, 4).to("cuda:0")
    torch.onnx._export(model, example_input, onnx_model_path, example_outputs=example_output, verbose=True)


if __name__ == '__main__':
    if torch.__version__ < "1.0.0":
        print("pytorch version is not  1.0.0, please check it!")
        exit(-1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint_file_path = r"./models/mobilenet_v2_0.25_43_0.1162.pth"
    save_path = r"./models/mobilenet_v2_0.25_43_0.1162_jit.pth"
    check_point = torch.load(checkpoint_file_path, map_location=device)
    # mapped_state_dict = OrderedDict()
    model = check_point['net'].to(device)
    recursion_change_bn(model)
    # for key, value in model.state_dict().items():
    #     print(key)
    #     mapped_key = key
    #     mapped_state_dict[mapped_key] = value
    #     if 'running_var' in key:
    #         mapped_state_dict[key.replace('running_var', 'num_batches_tracked')] = torch.zeros(1).to(device)
    # model.load_state_dict(mapped_state_dict, strict=True)
    pth_to_jit(model, save_path, device)

