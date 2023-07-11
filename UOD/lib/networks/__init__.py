from .uvgg import UVGG
from .DATR import DATR


def get_net(s):
    return {
            'uvgg': UVGG,
            'datr': DATR,
           }[s.lower()]
