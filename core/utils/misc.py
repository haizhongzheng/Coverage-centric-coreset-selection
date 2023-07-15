import torch
import os

def prediction_correct(true, preds):
    """
    Computes prediction_hit.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Prediction_hit for each img.
    """
    rst = (torch.softmax(preds, dim=1).argmax(dim=1) == true)
    return rst.detach().cpu().type(torch.int)

def get_model_directory(base_dir, model_name):
    model_dir = os.join(base_dir, model_name)
    ckpt_dir = os.join(model_dir, 'ckpt')
    data_dir = os.join(model_dir, 'data')
    log_dir = os.join(model_dir, 'log')

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return ckpt_dir, data_dir, log_dir

def l2_distance(tensor1, tensor2):
    dist = (tensor1 - tensor2).pow(2).sum().sqrt()
    return dist