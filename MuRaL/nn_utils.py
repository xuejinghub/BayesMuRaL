import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from bayesian_torch.ao.quantization.quantize import enable_prepare, convert



def weights_init(m):
    """Initialize network layers"""
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight)
        
        if m.bias is not None:
            nn.init.normal_(m.bias)
            nn.init.constant_(m.bias, 0)
        
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
            
        if m.bias is not None:
            nn.init.normal_(m.bias)
            nn.init.constant_(m.bias, 0)
        
    elif classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.xavier_uniform_(m.__getattr__(p))

def model_predict_m(model, dataloader, criterion, device, n_class, num_monte_carlo=10, distal=True):
    """Do model prediction using dataloader"""
    model.to(device)
    model.eval()
    
    pred_y = torch.empty(0, n_class).to(device)
    pred_y_std = torch.empty(0, n_class).to(device)
        
    total_loss = 0
    
    with torch.no_grad():
        for y, cont_x, cat_x, distal_x in dataloader:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            distal_x = distal_x.to(device)
            y  = y.to(device)
            
            pred_results = []
            output_mc = []

            for i in range(num_monte_carlo):

                if distal:
                    preds = model.forward((cont_x, cat_x), distal_x)
                else:
                    preds = model.forward(cont_x, cat_x)

                pred_results.append(preds)
                preds_pro = F.softmax(preds, dim=1)
                output_mc.append(preds_pro)

            pred_results = torch.stack(pred_results)
            output_mc = torch.stack(output_mc)
            means = output_mc.mean(axis=0)
            stds = output_mc.std(axis=0)

            pred_y = torch.cat((pred_y, means), dim=0)
            pred_y_std = torch.cat((pred_y_std, stds), dim=0)
                
            loss = criterion(pred_results.mean(axis=0), y.long().squeeze(1))
            total_loss += loss.item()
            
            if device == torch.device('cpu'):
                if  np.random.uniform(0,1) < 0.0001:
                    #print('in the model_predict_m:', device)
                    sys.stdout.flush()

    return pred_y, pred_y_std, total_loss

def quantize(model, calib_loader):
    model.eval()
    model.cpu()
    model.qconfig = torch.quantization.get_default_qconfig("onednn")
    model.emb_layer.qconfig = None
    print('Preparing model for quantization....')
    enable_prepare(model)
    prepared_model = torch.quantization.prepare(model)
    print('Calibrating...')
    with torch.no_grad():
        for y, cont_x, cat_x, distal_x in calib_loader:
            cat_x = cat_x.cpu()
            cont_x = cont_x.cpu()
            distal_x = distal_x.cpu()
            _ = prepared_model((cont_x, cat_x), distal_x)
            # if distal:
            #     _ = prepared_model((cont_x, cat_x), distal_x)
            # else:
            #     _ = prepared_model(cont_x, cat_x)
    print('Calibration complete....')
    quantized_model = convert(prepared_model)
    return quantized_model
