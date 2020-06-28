# related third party imports
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from retro.retro_env import RetroEnv


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_screen(device: torch.device, env: RetroEnv) -> torch.Tensor:
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


def show_example_screen(device: torch.device, env: RetroEnv) -> None:
    env.reset()
    plt.figure()
    plt.imshow(get_screen(device, env).cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
