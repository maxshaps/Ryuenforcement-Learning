import numpy as np

MOVE = {'NONE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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

ACITON_TO_MOVE = ['NONE', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'LIGHT_KICK', 'MEDIUM_KICK', 'HARD_KICK', 'LIGHT_PUNCH',
                  'MEDIUM_PUNCH', 'HARD_PUNCH']

# Move list from
# https://www.ign.com/faqs/2006/playtv-legends-sega-genesis-street-fighter-ii-special-champion-edition-move-list-522302

HADOUKEN_RIGHT = [MOVE['DOWN'], np.add(MOVE['DOWN'], MOVE['RIGHT']), np.add(MOVE['RIGHT'], MOVE['LIGHT_PUNCH'])]
SHORYUKEN_RIGHT = [MOVE['RIGHT'], MOVE['DOWN'], np.add(np.add(MOVE['DOWN'], MOVE['RIGHT']), MOVE['LIGHT_PUNCH'])]
TATSUMAKI_SENPUKYAKU_RIGHT = [MOVE['DOWN'], np.add(MOVE['DOWN'], MOVE['LEFT']), np.add(MOVE['LEFT'], MOVE['LIGHT_KICK'])]