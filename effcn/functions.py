import torch
import torch.nn.functional as F


def squash_func(x, eps=10e-21):
    """
        IN: 
            x (b, n, d)
        OUT: 
            squash(x) (b, n, d)
    """
    x_norm = torch.norm(x, dim=2, keepdim=True)
    return (1 - 1 / (torch.exp(x_norm) + eps)) * (x / (x_norm + eps))


def margin_loss(u, y_true, lbd=0.5, m_plus=0.9, m_minus=0.1):
    """
    IN:
        u      (b,n,d)  ... capsules with n equals the numbe of classes
        y_true (b,n)    .... labels vector, categorical representation
    OUT:
        loss, scalar  
    """

    u_norm = torch.norm(u, dim=2)
    term_left = torch.square(F.relu(m_plus - u_norm))
    term_right = torch.square(F.relu(u_norm - m_minus))
    #
    loss = y_true * term_left + lbd * (1.0 - y_true) * term_right
    loss = loss.sum(dim=1).mean()
    return loss


def margin_loss2(u, y_true, lbd=0.5, m_plus=0.9, m_minus=0.1):
    """
    Input:  u      (b,n,d)  ... capsules with n equals the numbe of classes
            y_true (b,n)    ... labels vector, categorical representation
    Output:
        loss, scalar  
    """
    
    u_norm = torch.norm(u, dim=-1)
    p_true = torch.square(F.relu(m_plus - u_norm))     #square is the difference to margin_loss!
    p_false = torch.square(F.relu(u_norm - m_minus))

    loss = y_true * p_true + lbd * (1-y_true) * p_false
    loss = loss.sum(dim=1).mean()
    
    return loss


def max_norm_masking(u):
    """
    IN:
        u (b, n d) ... capsules
    OUT:
        masked(u)  (b, n, d) where:
        - normalise over dimension d of u
        - keep largest vector in dimension n
        - mask out everything else
    """
    _, n_classes, _ = u.shape
    u_norm = torch.norm(u, dim=2)
    mask = F.one_hot(torch.argmax(u_norm, 1), num_classes=n_classes)
    return torch.einsum('bnd,bn->bnd', u, mask)


def masking_max_norm(u):
    """
    IN:
        u (b, n d) ... capsules
    OUT:
        masked(u)  (b, n, d) where:
        - normalise over dimension d of u
        - keep largest vector in dimension n
        - mask out everything else
    """
    _, n_classes, _ = u.shape
    u_norm = torch.norm(u, dim=2)
    mask = F.one_hot(torch.argmax(u_norm, 1), num_classes=n_classes)
    return torch.einsum('bnd,bn->bnd', u, mask)


def masking_y_true(u, y_true):
    """
    IN:
        u (b, n d) ... capsules
        y_true (b,)  ... classification value (skalar)
    OUT:
        masked(u)  (b, n, d) where:
        - normalise over dimension d of u
        - keep vector in dimension n with y_true
        - mask out everything else
    """
    _, n_classes, _ = u.shape
    u_norm = torch.norm(u, dim=2)
    mask = F.one_hot(y_true, num_classes=n_classes)
    return torch.einsum('bnd,bn->bnd', u, mask)