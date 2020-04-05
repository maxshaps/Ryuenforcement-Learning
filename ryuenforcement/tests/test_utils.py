from common.utils import ACTION


def test_action_list():
    """All actions are 1-hot encodings of length 12 (having all 0s is also a valid move for 'do nothing')"""
    for _, value in ACTION.items():
        assert len(value) == 12
        assert (sum(value) == 1 or sum(value) == 0)
        for num in value:
            assert (num == 0 or num == 1)
