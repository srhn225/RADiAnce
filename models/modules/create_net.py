#!/usr/bin/python
# -*- coding:utf-8 -*-

from .GET.get import GET
from .EPT.ept import XTransEncoderAct as EPT
from .EPT.ept import XTransEncoderActrag as EPTrag
from .EPT.ept import XTransEncoderActragfull as EPTragfull
from .EPT.ept import XTransEncoderActadaLN as EPTragadaLN
from .EPT.ept import XTransEncoderActincontext as EPTragincontext
from .EPT.ept import XTransEncoderActadaLNAttn as EPTragadaLNAttn
from .EPT.ept import XTransEncoderActragmask as EPTragmask
from .BAT.encoder import BATEncoder as BAT
from .EGNN.encoder import EGNNEncoder as EGNN


def create_net(
    name, # GET
    hidden_size,
    edge_size,
    opt={}
):
    if name == 'GET':
        kargs = {
            'd_hidden': hidden_size,
            'd_radial': hidden_size // 4,
            'n_channel': 1,
            'n_rbf': 32,
            'd_edge': edge_size
        }
        kargs.update(opt)
        return GET(**kargs)
    elif name == 'EPT':
        kargs = {
            'hidden_size': hidden_size,
            'ffn_size': hidden_size,
            'edge_size': edge_size
        }
        kargs.update(opt)
        return EPT(**kargs)
    elif name == 'BAT':
        kwargs = {
            'hidden_size': hidden_size,
            'n_rbf': 32,
            'cutoff': 10.0,
            'edge_size': edge_size
        }
        return BAT(**kwargs)
    elif name == 'EGNN':
        kwargs = {
            'hidden_size': hidden_size,
            'edge_size': edge_size,
        }
        kwargs.update(opt)
        return EGNN(**kwargs)
    elif name == 'EPTrag':
        kargs = {
            'hidden_size': hidden_size,
            'ffn_size': hidden_size,
            'edge_size': edge_size
        }
        kargs.update(opt)
        return EPTrag(**kargs)        
    elif name == 'EPTragfull':
        kargs = {
            'hidden_size': hidden_size,
            'ffn_size': hidden_size,
            'edge_size': edge_size
        }
        kargs.update(opt)
        return EPTragfull(**kargs)  
    elif name == 'EPTragadaLN':
        kargs = {
            'hidden_size': hidden_size,
            'ffn_size': hidden_size,
            'edge_size': edge_size
        }
        kargs.update(opt)
        return EPTragadaLN(**kargs)
    elif name == 'EPTragincontext':
        kargs = {
            'hidden_size': hidden_size,
            'ffn_size': hidden_size,
            'edge_size': edge_size
        }
        kargs.update(opt)
        return EPTragincontext(**kargs)     
    elif name == 'EPTragadaLNAttn':
        kargs = {
            'hidden_size': hidden_size,
            'ffn_size': hidden_size,
            'edge_size': edge_size
        }
        kargs.update(opt)
        return EPTragadaLNAttn(**kargs)
    elif name == 'EPTragmask':
        kargs = {
            'hidden_size': hidden_size,
            'ffn_size': hidden_size,
            'edge_size': edge_size
        }
        kargs.update(opt)
        return EPTragmask(**kargs)
    else:
        raise NotImplementedError(f'{name} not implemented')