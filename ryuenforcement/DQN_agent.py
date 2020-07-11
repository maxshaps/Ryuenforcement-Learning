import os
import retro
import torch
from ryuenforcement.agent import select_action, DQN
from ryuenforcement.environment import get_screen
from ryuenforcement.actions import move_index_to_action_array as mi2aa


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = 11
    eps_start = 0
    eps_end = 0
    eps_decay = 1

    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    steps_done = 0

    env.reset()
    last_screen = get_screen(device, env)
    current_screen = get_screen(device, env)
    state = current_screen - last_screen

    _, _, screen_height, screen_width = state.shape
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    policy_net.load_state_dict(torch.load(os.path.join('saved_files','hadouken_2020-07-07.pt')))
    policy_net.eval()
    while True:
        if steps_done % 2 == 0:
            action, steps_done = select_action(device, state, policy_net, n_actions, eps_start, eps_end, eps_decay,
                                               steps_done)
        else:
            action = torch.tensor([[0]], device=device, dtype=torch.long)
            steps_done += 1
        _, rew, done, _ = env.step(mi2aa(action.item()))
        env.render()
        if done:
            _ = env.reset()
            break
        last_screen = current_screen
        current_screen = get_screen(device, env)
        state = current_screen - last_screen
    env.close()


if __name__ == "__main__":
    main()
