import functools

import torch
import torch.nn as nn
from torch.nn import init
from networks.modules import architecture as arch



####################
# initialize
####################

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.constant_(m.weight.data, 1.0)
        m.weight.data *= scale
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

####################
# define network
####################

def create_model(opt):
    if opt['mode'] == 'sr':
        net = define_net(opt['networks'])
        return net
    else:
        raise NotImplementedError("The mode [%s] of networks is not recognized." % opt['mode'])

# choose one network
def define_net(opt):

    which_model = opt['which_model'].upper()
    print('===> Building network [%s]...'%which_model)

    if which_model == 'DBPN':
        from .dbpn_arch import DBPN
        net = DBPN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                         num_features=opt['num_features'], bp_stages=opt['num_blocks'],
                         upscale_factor=opt['scale'])

    elif which_model == 'D-DBPN':
        from .dbpn_arch import D_DBPN
        net = D_DBPN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                           num_features=opt['num_features'], bp_stages=opt['num_blocks'],
                           upscale_factor=opt['scale'])

    elif which_model.find('SRFBN') >= 0:
        from .srfbn_arch import SRFBN
        net = SRFBN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                                  num_features=opt['num_features'], num_steps=opt['num_steps'], num_groups=opt['num_groups'],
                                  upscale_factor=opt['scale'])

    elif which_model.find('CAON') >= 0:
        from .caon_arch import CAON
        net = CAON(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                                  num_features=opt['num_features'], num_steps=opt['num_steps'], num_groups=opt['num_groups'],
                                  upscale_factor=opt['scale'])

    elif which_model.find('GMFN') >= 0:
        from .gmfn_arch import GMFN
        net = GMFN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                               num_features=opt['num_features'], num_steps=opt['num_steps'], num_blocks=opt['num_blocks'],
                               num_reroute_feats=opt['num_reroute_feats'], num_refine_feats=opt['num_refine_feats'],
                               upscale_factor=opt['scale'])

    elif which_model.find('RDN') >= 0:
        from .rdn_arch import RDN
        net = RDN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_blocks = opt['num_blocks'], num_layers = opt['num_layers'],
                             upscale_factor=opt['scale'])

    elif which_model.find('EDSR') >= 0:
        from .edsr_arch import EDSR
        net = EDSR(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_blocks = opt['num_blocks'], res_scale=opt['res_scale'],
                             upscale_factor=opt['scale'])

    else:
        raise NotImplementedError("Network [%s] is not recognized." % which_model)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    return net

def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']
    HR_size = opt['datasets']['train']['HR_size']

    if which_model == 'discriminator_vgg_128':
        netD = arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'],
            HR_size=HR_size)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized, only support discriminator_vgg_128'.format(which_model))
    '''
    elif which_model == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        netD = sft_arch.ACD_VGG_BN_96()

    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    '''
    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD.cuda())
    return netD

