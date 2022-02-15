import torch


def pos_tanh_embedding(h, w, t_freq=2, t_symm=0.5):
    pe = torch.zeros(4, h, w)
    pe[0] = (1 - torch.tanh(t_freq * (torch.linspace(0, 1, w) - t_symm)
                            ).unsqueeze(1).repeat(1, h)) * 0.5
    pe[1] = (1 - torch.tanh(t_freq * (torch.linspace(1, 0, w) - t_symm)
                            ).unsqueeze(1).repeat(1, h)) * 0.5
    pe[2] = (1 - torch.tanh(t_freq *
                            (torch.linspace(0, 1, h) - t_symm)).T.repeat(w, 1)) * 0.5
    pe[3] = (1 - torch.tanh(t_freq *
                            (torch.linspace(1, 0, h) - t_symm)).T.repeat(w, 1)) * 0.5
    return pe
