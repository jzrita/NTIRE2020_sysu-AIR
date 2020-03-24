import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable

import os

class myloss_func(object):
    def __init__(self, opt):
        self.opt = opt
        self.pixel_loss = nn.L1Loss().cuda()  # if want to use l2loss , change it to nn.MSELoss()
        self.netF = define_F(opt)
        self.style_loss = style_loss
        self.tv_loss = TVLoss()
        self.fft_loss = fft_loss
        self.cri_gan = GANLoss(opt['solver']['gan_type'], 1.0, 0.0).cuda()

    def compute_D_loss(self, sr, hr, discriminator):
        self.l_d_total = 0
        for step, subiter_sr in enumerate(sr):
            # I find if we compute ganloss for all 4 steps, it cost more than double cost of GPU memory so that only compute for the last step
            if step != len(sr)-1:
                continue
            self.pred_d_real = discriminator(hr)
            self.pred_d_fake = discriminator(subiter_sr.detach())  # detach to avoid BP to G
            self.l_d_real = self.cri_gan(self.pred_d_real - torch.mean(self.pred_d_fake), True)
            self.l_d_fake = self.cri_gan(self.pred_d_fake - torch.mean(self.pred_d_real), False)

            self.l_d_total += (self.l_d_real + self.l_d_fake) / 2
        return self.l_d_total

    def compute_loss(self, sr, hr, discriminator=None):
        # sr has 4 iter

        l_total = pix_loss = f_loss = tvloss = styleloss = fftloss = self.ganloss = 0       
        for step, subiter_sr in enumerate(sr):
            # pixel loss
            pix_loss += self.pixel_loss(subiter_sr, hr) * self.opt['solver']['cl_weights'][step]
            # feature loss
            feature_sr = self.netF(subiter_sr)
            feature_hr = self.netF(hr).detach()
            f_loss += self.pixel_loss(feature_sr, feature_hr) * self.opt['solver']['feature_loss_weights'][step]
            # tv loss
            tvloss += self.tv_loss(subiter_sr) * self.opt['solver']['tv_loss_weights'][step]
            # style loss
            styleloss += self.style_loss(feature_sr, feature_hr, self.opt['gpu_ids']) * self.opt['solver']['style_loss_weights'][step]
            # fft loss
            fftloss += self.fft_loss(subiter_sr, hr) * self.opt['solver']['fft_loss_weights'][step]
            if not discriminator is None:
                # I find if we compute ganloss for all 4 steps, it cost more than double cost of GPU memory so that only compute for the last step
                if step != len(sr)-1:
                    continue
                # G gan loss
                self.pred_g_fake = discriminator(subiter_sr)
                self.pred_d_real = discriminator(hr).detach()

                self.ganloss += self.opt['solver']['gan_weight'][step] * (self.cri_gan(self.pred_d_real - torch.mean(self.pred_g_fake), False) +
                                        self.cri_gan(self.pred_g_fake - torch.mean(self.pred_d_real), True)) / 2        
        # loss log
        l_total = pix_loss + f_loss + tvloss + styleloss + fftloss + self.ganloss
        loss_log = {
            "pixel_loss":pix_loss.item(),
            "feature_loss":f_loss.item(),
            "tv_loss":tvloss.item(),
            "style_loss":styleloss.item(),
            "fft_loss":fftloss.item(),
            "generator_vanilla_loss":self.ganloss.item()
        }
        
        return l_total, loss_log


def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, device=device).cuda()
    # netF = arch.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF


class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def fft_loss(sr, hr):  
    if_normalized = False   # fft是否归一化
    L1 = True               # 是否算L1Loss，否则算MSELoss
    if_amp = False          # 是否对能量谱算loss，否则直接复数求差
    if_db = False          # 是否取对数，仅对能量谱有效

    dist_func = nn.L1Loss().cuda() if L1 else nn.MSELoss()
    sr_fft_complex = torch.rfft(sr,2,normalized=if_normalized, onesided=False)
    hr_fft_complex = torch.rfft(hr,2,normalized=if_normalized, onesided=False)

    if if_amp:
        sr_fft_real = sr_fft_complex[:,:,:,:,0]
        sr_fft_Im = sr_fft_complex[:,:,:,:,1]
        sr_fft_amp = sr_fft_real.mul(sr_fft_real) + sr_fft_Im.mul(sr_fft_Im)
        hr_fft_real = hr_fft_complex[:,:,:,:,0]
        hr_fft_Im = hr_fft_complex[:,:,:,:,1]
        hr_fft_amp = hr_fft_real.mul(hr_fft_real) + hr_fft_Im.mul(hr_fft_Im)
        if if_db:
            sr_fft_amp = torch.log(sr_fft_amp)
            hr_fft_amp = torch.log(hr_fft_amp)
        return dist_func(sr_fft_amp, hr_fft_amp)
    else:
        return dist_func(sr_fft_complex, hr_fft_complex)



def style_loss(feature_sr, feature_hr, gpu_ids):
    """
    Compute style loss

     para:
     feature_sr: tensor after using vgg on sr imgs 
     feature_hr: tensor after using vgg on hr imgs 
     gpu_ids: if None than run style_loss on cpu
    """
    sr_style = gram_matrix(feature_sr, gpu_ids)
    hr_style = gram_matrix(feature_hr, gpu_ids)
    return nn.L1Loss()(sr_style, hr_style)

def gram_matrix(input_tensor, gpu_ids):
    """
    Compute Gram matrix

    :param input_tensor: input tensor with shape
     (batch_size, nbr_channels, height, width)
    :return: Gram matrix of y
    """
    (b, ch, h, w) = input_tensor.size()
    features = input_tensor.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    
    # more efficient and formal way to avoid underflow for mixed precision training
    input = torch.zeros(b, ch, ch).type(features.type())
    if not gpu_ids is None:
        input = input.cuda()
    gram = torch.baddbmm(input, features, features_t, beta=0, alpha=1./(ch * h * w), out=None)
    
    # naive way to avoid underflow for mixed precision training
    # features = features / (ch * h)
    # gram = features.bmm(features_t) / w

    # for fp32 training, it is also safe to use the following:
    # gram = features.bmm(features_t) / (ch * h * w)

    return gram


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

