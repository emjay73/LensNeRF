import torch

def get_length_with_min_dist(o1, d1, o2, d2):
    d1 = d1/torch.linalg.norm(d1, dim=-1, keepdim=True)
    d2 = d2/torch.linalg.norm(d2, dim=-1, keepdim=True)

    n = torch.cross(d1, d2)
    n = n/torch.linalg.norm(n, dim=-1, keepdim=True)
    
    n1 = torch.cross(d1, n)
    n1 = n1/torch.linalg.norm(n1, dim=-1, keepdim=True)
    
    n2 = torch.cross(d2, n)
    n2 = n2/torch.linalg.norm(n2, dim=-1, keepdim=True)

    #t1 = torch.dot((o2-o1), n2)/torch.dot(d1, n2)
    #t2 = torch.dot((o1-o2), n1)/torch.dot(d2, n1)
    t1 = torch.sum((o2-o1)*n2, dim=1)/torch.sum(d1 * n2, dim=1)
    t2 = torch.sum((o1-o2)*n1, dim=1)/torch.sum(d2 * n1, dim=1)

    return t1, t2