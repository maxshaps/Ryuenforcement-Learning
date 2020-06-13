import retro
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from common.actions import move_index_to_action_array as mi2aa

def get_screen(env):
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[all_available_moves]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
    def select_inference_action(self, state):
        return self.forward(state).max(1)[1].view(1, 1)
   

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape
    n_actions = 11
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    policy_net.load_state_dict(torch.load('hadoooouken.pt'))
    policy_net.eval()
    obs = env.reset()
    screen = get_screen(env)
    count = 0
    while True:
        print(count)
        action = mi2aa(torch.argmax(policy_net(screen)).item())
        obs, rew, done, info = env.step(action)
        get_screen(env)
        count +=1
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()