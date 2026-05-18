
# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model_utils import AverageMeter, BestMeter, load_model_dict, save_model_dict
from model_enz import bottle_view_graph
from dataset_graphkcat_chai1 import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
# from utils import *
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error
torch.multiprocessing.set_sharing_strategy('file_system')  # solve the issue of multiprocessing

# %%

def compute_kcat_loss_and_pcc(pred_kcat, y_kcat):

    if isinstance(pred_kcat, np.ndarray):
        pred_kcat = torch.from_numpy(pred_kcat).float()
    if isinstance(y_kcat, np.ndarray):
        y_kcat = torch.from_numpy(y_kcat).float()
    # 
    if isinstance(y_kcat, list):
        y_kcat = torch.tensor([float('nan') if x is None else x for x in y_kcat], dtype=torch.float32)

    # 
    mask_kcat = ~torch.isnan(y_kcat)

    # 
    loss_kcat = 0
    pcc_kcat = None
    if mask_kcat.any():  # 
        valid_pred_kcat = pred_kcat[mask_kcat]  # 
        valid_y_kcat = y_kcat[mask_kcat]        # 
        loss_kcat = F.mse_loss(valid_pred_kcat, valid_y_kcat)  # MSE 

        # Pearson 
        if valid_pred_kcat.size(0) > 1:  # 
            mean_pred = valid_pred_kcat.mean()
            mean_y = valid_y_kcat.mean()
            diff_pred = valid_pred_kcat - mean_pred
            diff_y = valid_y_kcat - mean_y
            numerator = torch.sum(diff_pred * diff_y)
            denominator = torch.sqrt(torch.sum(diff_pred ** 2) * torch.sum(diff_y ** 2))
            pcc_kcat = numerator / denominator if denominator != 0 else 0.0
        else:
            pcc_kcat = None  

    return loss_kcat, pcc_kcat

def compute_km_loss_and_pcc(pred_km, y_km):

    if isinstance(pred_km, np.ndarray):
        pred_km = torch.from_numpy(pred_km).float()
    if isinstance(y_km, list):
        y_km = torch.tensor([float('nan') if x is None else x for x in y_km], dtype=torch.float32)
    elif isinstance(y_km, np.ndarray):
        y_km = torch.from_numpy(y_km).float()
    
    
    mask_km = ~torch.isnan(y_km)
    
    loss_km, pcc_km = 0.0, None
    
    if mask_km.any():
        
        valid_pred_km = pred_km[mask_km].flatten()
        valid_y_km = y_km[mask_km].flatten()
        
        
        loss_km = F.mse_loss(valid_pred_km, valid_y_km)
        
        # 
        if valid_pred_km.size(0) > 1:
            mean_pred = valid_pred_km.mean()
            mean_y = valid_y_km.mean()
            diff_pred = valid_pred_km - mean_pred
            diff_y = valid_y_km - mean_y
            
            numerator = torch.sum(diff_pred * diff_y)
            denominator = torch.sqrt(torch.sum(diff_pred ** 2) * torch.sum(diff_y ** 2))
            
            if denominator == 0:
                # 
                if torch.all(diff_pred == 0) and torch.all(diff_y == 0):
                    pcc_km = 1.0  # 
                else:
                    pcc_km = None  # 
            else:
                pcc_km = (numerator / denominator).item()
    
    return loss_km, pcc_km
def val(model, dataloader, device):
    model.eval()

    pred_kcat_list = []
    label_kcat_list = []
    pred_km_list = []
    label_km_list = []
    for data in dataloader:
        
        with torch.no_grad():
            data = [data[i].to(device) for i in range(len(data)) if data[i] != "label"]
            # label = data["label"]
            label_kcat = [data[i]["label_kcat"] for i in range(len(data))] # list of tensor
            label_kcat = torch.cat(label_kcat, dim=0).to(device)

            label_km = [data[i]["label_km"] for i in range(len(data))] # list of tensor
            label_km = torch.cat(label_km, dim=0).to(device)


            pred_kcat, pred_km,_,_,_,_,_,_,_,_,_,_ = model(data)
            

            pred_kcat_list.append(pred_kcat.detach().cpu().numpy())
            label_kcat_list.append(label_kcat.detach().cpu().numpy())
            pred_km_list.append(pred_km.detach().cpu().numpy())
            label_km_list.append(label_km.detach().cpu().numpy())
            
            
    # pred = np.concatenate(pred_list, axis=0)
    # label = np.concatenate(label_list, axis=0)
    pred_kcat = np.concatenate(pred_kcat_list, axis=0)
    label_kcat = np.concatenate(label_kcat_list, axis=0)
    pred_km = np.concatenate(pred_km_list, axis=0)
    label_km = np.concatenate(label_km_list, axis=0)

    print(label_kcat)


    # coff = np.corrcoef(pred, label)[0, 1]
    # coff_kcat = np.corrcoef(pred_kcat, label_kcat)[0, 1]
    # coff_km = np.corrcoef(pred_km, label_km)[0, 1]
    # # rmse = np.sqrt(mean_squared_error(label, pred))
    # rmse_kcat = np.sqrt(mean_squared_error(label_kcat, pred_kcat))
    # rmse_km = np.sqrt(mean_squared_error(label_km, pred_km))

    mse_kcat, coff_kcat = compute_km_loss_and_pcc(pred_kcat, label_kcat)
    mse_km, coff_km = compute_km_loss_and_pcc(pred_km, label_km)
    rmse_kcat = np.sqrt(mse_kcat)
    rmse_km = np.sqrt(mse_km)

    model.train()

    return rmse_kcat, coff_kcat, rmse_km, coff_km

# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_kcat_enz'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    hidden_dim = args.get("hidden_dim")
    lr = args.get("lr")
    # share_score = args.get("share_score")
    pooling = args.get("pooling")
    vocab_size = args.get("vocab_size")
    num_layers = args.get("num_layers")
    dropout = args.get("dropout")
    # cross_attention = args.get("cross_attention")
    # num_heads = args.get("num_heads")
    ligand_nn_embedding = args.get("ligand_nn_embedding")
    HeteroGNN_layers = args.get("HeteroGNN_layers")
    num_fc_layers = args.get("num_fc_layers")
    fc_hidden_dim = args.get("fc_hidden_dim")  # hidden_dim *  fc_hidden_dim
    share_fc = args.get("share_fc")

    for repeat in range(repeats):
        args['repeat'] = repeat


        data_dir = os.path.join(data_root, 'structure_enzyme')

        train_df = pd.read_csv(os.path.join(data_root, "modeling-datasets", f"train_dataset_clean_no_structure.csv" ))#/  filter_train_2020 refined_2016 data_training filter_train_2020 train_2016 val_2016
        valid_df = pd.read_csv(os.path.join(data_root, "modeling-datasets", f"valid_dataset_clean_no_structure.csv" ))# /  filter_val_2020 val_2016 data_validation.val_new.csv
        test2016_df = pd.read_csv(os.path.join(data_root, "modeling-datasets", f"test_dataset_clean_no_structure.csv" ))


        train_set = GraphDataset(data_dir, train_df, data_type="train", graph_type=graph_type, dis_threshold=8, create=False)
        valid_set = GraphDataset(data_dir, valid_df, data_type="valid", graph_type=graph_type, dis_threshold=8, create=False)
        test2016_set = GraphDataset(data_dir, test2016_df, data_type="test", graph_type=graph_type,dis_threshold=8, create=False)

        train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
        test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=0)

        logger = TrainLogger(args, cfg, create=True)
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"test2016 data: {len(test2016_set)}")


        device = torch.device('cuda:0')
        model = bottle_view_graph(node_dim=35,
                                hidden_dim=hidden_dim,
                                HeteroGNN_layers=HeteroGNN_layers,
                                # share_score=share_score,
                                pooling=pooling,
                                vocab_size=vocab_size,
                                num_layers=num_layers,
                                dropout=dropout,
                                # cross_attention=cross_attention,
                                # num_heads=num_heads,
                                ligand_nn_embedding=ligand_nn_embedding,
                                num_fc_layers=num_fc_layers,
                                fc_hidden_dim=fc_hidden_dim,
                                share_fc=share_fc
                                )
        model = model.to(device)
        # load_model_dict(model, f'../unsupervised/model/20230817_110144_Con_BAP_repeat3/model/contrastive_1.pt')
        # model = downstream_affinity(model, 256).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6) #1e-4
        criterion = nn.MSELoss()

        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []
        
        model.train()
        for epoch in range(epochs):
            for data in train_loader:
                # data['ligand_features'] = data['ligand_features'].to(device)
                # # data['atom_pocket_features'] = data['atom_pocket_features'].to(device)
                # # data['amino_acid_features'] = data['amino_acid_features'].to(device)
                # data['complex_features'] = data['complex_features'].to(device)
                data = [data[i].to(device) for i in range(len(data))]
                pred_kcat, pred_km = model(data)#, log_var1, log_var2
                label_kcat = [data[i]["label_kcat"] for i in range(len(data))] # list of tensor
                # print(label)
                label_kcat = torch.cat(label_kcat, dim=0).to(device)

                label_km = [data[i]["label_km"] for i in range(len(data))] # list of tensor
                label_km = torch.cat(label_km, dim=0).to(device)
                
                

                # loss = criterion(pred, label)
                # loss = compute_loss(pred_kcat, pred_km, label_kcat, label_km)
                loss_kcat,_ = compute_km_loss_and_pcc(pred_kcat,label_kcat)
                loss_km,_ = compute_km_loss_and_pcc(pred_km, label_km)
                # 加权损失
                # weighted_loss_kcat = 0.5 * torch.exp(-log_var1) * loss_kcat 
                # weighted_loss_km = 0.5 * torch.exp(-log_var2) * loss_km
                # reg_term = 0.5 * (log_var1 + log_var2)

                # # # cal L2 regularization
                # # l2_reg = torch.tensor(0.).to(device)
                # # for name, param in model.named_parameters():
                # #     if "log_var" not in name:
                # #         l2_reg += torch.sum(param ** 2)
                # loss =  weighted_loss_kcat + weighted_loss_km + reg_term #+ 0.001 * l2_reg

                # loss =  weighted_loss_kcat + weighted_loss_km + reg_term
                loss =  loss_kcat + loss_km#
                # 
                mask_kcat = ~torch.isnan(label_kcat)  # kcat 
                mask_km = ~torch.isnan(label_km)      # km 
                valid_samples = (mask_kcat | mask_km).sum().item()  # 

                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), valid_samples) 

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            # start validating
            valid_rmse_kcat, valid_pr_kcat, valid_rmse_km, valid_pr_km = val(model, valid_loader, device)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse_kcat-%.4f, valid_pr_kcat-%.4f, valid_rmse_km-%.4f, valid_pr_km-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse_kcat, valid_pr_kcat, valid_rmse_km, valid_pr_km)
            logger.info(msg)

            # if valid_rmse < running_best_mse.get_best():
            running_best_mse.update(valid_rmse_kcat)
            if save_model:
                msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse_kcat-%.4f, valid_pr_kcat-%.4f, valid_rmse_km-%.4f, valid_pr_km-%.4f" \
                        % (epoch, epoch_loss, epoch_rmse, valid_rmse_kcat, valid_pr_kcat, valid_rmse_km, valid_pr_km)
                model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                best_model_list.append(model_path)
                save_model_dict(model, logger.get_model_dir(), msg)
            else:
                count = running_best_mse.counter()
                if count > early_stop_epoch:
                    best_mse = running_best_mse.get_best()
                    msg = "best_rmse: %.4f" % best_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break

        # final testing
        load_model_dict(model, best_model_list[-1])
        valid_rmse, valid_pr = val(model, valid_loader, device)
        test2016_rmse, test2016_pr = val(model, test2016_loader, device)


        # msg = "valid_rmse-%.4f, valid_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f," \
        #             % (valid_rmse, valid_pr, test2016_rmse, test2016_pr)
        msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse_kcat-%.4f, valid_pr_kcat-%.4f, valid_rmse_km-%.4f, valid_pr_km-%.4f" \
        % (epoch, epoch_loss, epoch_rmse, valid_rmse_kcat, valid_pr_kcat, valid_rmse_km, valid_pr_km)


        logger.info(msg)
        

# %%