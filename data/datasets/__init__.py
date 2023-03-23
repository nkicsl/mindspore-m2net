# encoding: utf-8

from .duke import Duke
from .duke_conb import DukeCM
from .market import Market
from .market_conb import MarketCM
from .dataset_loader import ImageDataset, CMImageDataset
from .nkupv2 import NKUPv2
from .nkupv2_conb import NKUPv2CM

__factory = {
    'market': Market,
    'market_cm': MarketCM,
    'duke': Duke,
    'duke_cm': DukeCM,
    'nkupv2': NKUPv2,
    'nkupv2_cm': NKUPv2CM,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
