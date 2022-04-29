# third party libraries
import torch


def spread_loss_buggy(y_pred, y_true, m, device):
    print(y_pred, y_true.shape)
    at = torch.zeros(y_true.shape).to(device)
    zr = torch.zeros((y_pred.shape[0],y_pred.shape[1]-1)).to(device)
    #create at
    for i, cl in enumerate(y_true):
        at[i] = y_pred[i][cl]
    at = at.unsqueeze(1).repeat(1,y_pred.shape[1])
    print(zr.shape, at.shape)
    ai = y_pred[y_pred!=at].view(y_pred.shape[0],-1)
    print(ai.shape)

    loss = ((torch.max( m-(at[:,:-1] - ai), zr))**2).sum(dim=1)
    return loss.mean()


def spread_loss(y_pred, y_true, m, device):
    at = torch.zeros(y_true.shape).to(device)
    zr = torch.zeros((y_pred.shape[0],y_pred.shape[1])).to(device)
    #create at
    for i, cl in enumerate(y_true):
        at[i] = y_pred[i][cl]
    at = at.unsqueeze(1).repeat(1,y_pred.shape[1])
    
    loss = torch.max(m - (at - y_pred), zr)
    loss = loss**2
    loss = loss.sum() / y_true.shape[0] - m**2    

    return loss


def func_margin_hinton(step_abs, m_max, m_min):
    # AG 31/07/2018: function for margin of loss func
    # on Hinton's response to questions on OpenReview.net: 
    # https://openreview.net/forum?id=HJWLfGWRb
    # !!! I actually do not understand the fix hyper-parameter
    # !!! They only makes sence with the right size of steps...
    return (m_min + (m_max - m_min - 0.01) * torch.sigmoid(torch.min(torch.tensor([10.0, step_abs / 50000.0 - 4])))).item()


def func_margin_linear(step_rel, m_max, m_min):
    return m_min + (m_max - m_min)*step_rel


def func_step_rel(num_epochs, dl_len, epoch_idx, idx):
    return (1.*idx + (epoch_idx-1)*dl_len) / (num_epochs*dl_len)


def func_step_abs(dl_len, epoch_idx, idx):
    return (1. + idx + (epoch_idx-1)*dl_len)


def exp_lr_decay(optimizer, global_step, init_lr = 3e-3, decay_steps = 20000,
                                        decay_rate = 0.96, lr_clip = 3e-3 ,staircase=False):
    
    ''' decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)  '''
    
    if staircase:
        lr = (init_lr * decay_rate**(global_step // decay_steps)) 
    else:
        lr = (init_lr * decay_rate**(global_step / decay_steps)) 
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def func_acc(y_pred, y_true):
    y_pr_val, y_pr_idx = torch.topk(input=y_pred, k=1, dim=1, largest=True, sorted=True)
    acc = (y_pr_idx.squeeze(-1) == y_true).sum() / y_true.shape[0]
    return acc


if __name__ == '__main__':
    import torch.nn.functional as F

    loss_dev = torch.device("cpu")
    bs = 8
    num_epochs = 1000
    batch_size = 60
    epoch_idx = 90
    idx = 900
    m_max = 0.9
    m_min = 0.2
    y_true = torch.randint(0, 9, (bs,), requires_grad=False).to(loss_dev)
    y_pred = torch.rand(bs,10, requires_grad=True).to(loss_dev)

    # Loss with margin Hinton
    st_abs = func_step_abs(batch_size, epoch_idx, idx)
    marg_hinton = func_margin_hinton(st_abs, m_max, m_min)
    loss_hi = spread_loss(y_pred, y_true, m=marg_hinton, device=loss_dev)

    print(st_abs, marg_hinton, loss_hi)

    # Loss with margin Hinton
    st_rel = func_step_rel(num_epochs, batch_size, epoch_idx, idx)
    marg_linear = func_margin_linear(st_rel, m_max, m_min)
    loss_li = spread_loss(y_pred, y_true, m=marg_linear, device=loss_dev)
    
    print(st_rel, marg_linear, loss_li)

    