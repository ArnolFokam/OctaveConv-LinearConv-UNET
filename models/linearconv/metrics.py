import torch


def l1Loss(feat):
    loss = 0
    param = {}
    for i in feat.named_parameters():
        if 'linear_weights' in i[0]:
            dat = i[1]
            loss += torch.sum(torch.abs(dat))
    return loss


def corrcoef(x):
    mean_x = torch.mean(x, dim=1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c/stddev[:,None]
    c = c/stddev[None,:]
    c = torch.clamp(c, -1.0, 1.0)
    return c


def corrLoss(feat, device=torch.device('cpu')):
    loss = 0
    for i in feat.named_parameters():
        if 'conv_weights' in i[0]:
            dat = i[1]
            corr = corrcoef(dat.reshape(dat.shape[0], -1))
            loss += torch.sum(torch.abs(corr - torch.eye(corr.shape[0]).to(device)))
    return loss
