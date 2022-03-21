# Compile efficientnet model into torchscript

import torch as th
from torchvision.models import efficientnet_b7


def main():
    model = efficientnet_b7(pretrained=True)
    ts_model = th.jit.script(model)

    ts_model.save('efficientnet_b7.pt')


if __name__ == '__main__':
    main()
