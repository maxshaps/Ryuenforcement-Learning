from typing import List
import numpy as np

# Move List
MOVE_STR_LIST = ['NONE', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'LIGHT_KICK', 'MEDIUM_KICK', 'HARD_KICK', 'LIGHT_PUNCH',
                 'MEDIUM_PUNCH', 'HARD_PUNCH']

# Move to action
MOVE_STR_TO_ACTION_ARRAY = {'NONE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            'UP': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            'DOWN': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            'LEFT': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            'RIGHT': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            'LIGHT_KICK': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A
                            'MEDIUM_KICK': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B
                            'HARD_KICK': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # C
                            'LIGHT_PUNCH': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # X
                            'MEDIUM_PUNCH': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Y
                            'HARD_PUNCH': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # Z
                           }

# Special Moves
HADOUKEN_RIGHT = [MOVE_STR_TO_ACTION_ARRAY['DOWN'], np.add(MOVE_STR_TO_ACTION_ARRAY['DOWN'], MOVE_STR_TO_ACTION_ARRAY['RIGHT']), np.add(MOVE_STR_TO_ACTION_ARRAY['RIGHT'], MOVE_STR_TO_ACTION_ARRAY['LIGHT_PUNCH'])]
SHORYUKEN_RIGHT = [MOVE_STR_TO_ACTION_ARRAY['RIGHT'], MOVE_STR_TO_ACTION_ARRAY['DOWN'], np.add(np.add(MOVE_STR_TO_ACTION_ARRAY['DOWN'], MOVE_STR_TO_ACTION_ARRAY['RIGHT']), MOVE_STR_TO_ACTION_ARRAY['LIGHT_PUNCH'])]
TATSUMAKI_SENPUKYAKU_RIGHT = [MOVE_STR_TO_ACTION_ARRAY['DOWN'], np.add(MOVE_STR_TO_ACTION_ARRAY['DOWN'], MOVE_STR_TO_ACTION_ARRAY['LEFT']), np.add(MOVE_STR_TO_ACTION_ARRAY['LEFT'], MOVE_STR_TO_ACTION_ARRAY['LIGHT_KICK'])]


def move_index_to_action_array(move_str_index: int) -> List[int]:
    move_str = MOVE_STR_LIST[move_str_index]
    action_array = MOVE_STR_TO_ACTION_ARRAY[move_str]
    return action_array


# Move list from
# https://www.ign.com/faqs/2006/playtv-legends-sega-genesis-street-fighter-ii-special-champion-edition-move-list-522302

