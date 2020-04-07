import pytest
from ryuenforcement.common.actions import MOVE, HADOUKEN_RIGHT, SHORYUKEN_RIGHT, TATSUMAKI_SENPUKYAKU_RIGHT


def test_moves():
    """All actions are 1-hot encodings of length 12 (having all 0s is also a valid move for 'do nothing')"""
    for _, value in MOVE.items():
        assert len(value) == 12
        assert (sum(value) == 1 or sum(value) == 0)
        for num in value:
            assert (num == 0 or num == 1)


@pytest.mark.parametrize("special_move", [HADOUKEN_RIGHT, SHORYUKEN_RIGHT, TATSUMAKI_SENPUKYAKU_RIGHT])
def test_special_moves(special_move):
    assert len(special_move) == 3
    for move in special_move:
        for value in move:
            assert value == 1 or value == 0
