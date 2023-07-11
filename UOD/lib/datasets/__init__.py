from .head import Head
from .hand import Hand
from .jsrt import JSRT
from .jsrt_ssl import JSRT_SSL, Testset_JSRT_SSL
from .head_ssl import Head_SSL, Testset_Head_SSL
from .hand_ssl import Hand_SSL, Testset_Hand_SSL


def get_dataset(s):
    return {
            'head': Head,
            'head_ssl': Head_SSL,
            'testset_head_ssl': Testset_Head_SSL,

            'jsrt': JSRT,
            'jsrt_ssl': JSRT_SSL,
            'testset_jsrt_ssl': Testset_JSRT_SSL,

            'hand': Hand,
            'hand_ssl': Hand_SSL,
            'testset_hand_ssl': Testset_Hand_SSL,

           }[s.lower()]
