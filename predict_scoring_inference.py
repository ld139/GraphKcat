
# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pandas as pd
import torch
from model_enz import bottle_view_graph
from dataset_graphkcat_chai1 import GraphDataset, PLIDataLoader
import numpy as np
from model_utils import *
from config.config_dict import Config
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
# from tmtools import tm_align
import esm
from unimol_tools import UniMolRepr
from preprocessing_inference import extract_pocket_and_ligand, get_pocket_by_sdf, \
      transfer_conformation, extract_sequence_from_pdb, get_unimol2_embedding, get_esm2_embeddings, three_to_one
# %%

def preprocessing(df, clf, esm_model,alphabet, batch_converter, cutoff=8):
    for i, row in df.iterrows():
        cid = row["id"]
        has_complex = 'complex' in row.index and row['complex'] and not pd.isna(row['complex'])
        if has_complex:
            data_dir = os.path.dirname(row["complex"])
            complex = row["complex"]
            complex_path = complex
            if not os.path.exists(complex_path):
                print(f"Complex {complex} does not exist in {data_dir}")
                continue
            extract_pocket_and_ligand(complex_path, cutoff=cutoff)
        else:
            data_dir = os.path.dirname(row["ligand"])
            ligand_path = row["ligand"]
            protein_path = row["protein"]
            if not os.path.exists(ligand_path):
                print(f"Ligand {ligand_path} does not exist")
                continue
            if not os.path.exists(protein_path):
                print(f"Protein {protein_path} does not exist")
                continue
            pocket_path = get_pocket_by_sdf(ligand_path, protein_path, distance=cutoff)
            # df.at[i, "pocket"] = pocket_path
        if not os.path.exists(os.path.join(data_dir, f"{cid}_ligand.sdf")):
            smiles = row["Smiles"]
            transfer_conformation(os.path.join(data_dir, f"{cid}_ligand.pdb"),
                                   smiles, os.path.join(data_dir, f"{cid}_ligand.sdf"))
            
        seq = extract_sequence_from_pdb(protein_path)

        print(f"Processing {cid} with sequence length {len(seq)}")
        if os.path.exists(os.path.join(data_dir, f"{cid}_unimol_1b.pt")) and \
           os.path.exists(os.path.join(data_dir, f"{cid}_esm2_3b.pt")):
            print(f"Embeddings for {cid} already exist, skipping...")
            continue

        mol_embedding = get_unimol2_embedding(clf, row["Smiles"], embedding_type="atomic_reprs")
        esm_embedding = get_esm2_embeddings(esm_model, alphabet, batch_converter, seq, mean=False)

        torch.save(mol_embedding, os.path.join(data_dir, f"{cid}_unimol_1b.pt"))
        torch.save(esm_embedding, os.path.join(data_dir, f"{cid}_esm2_3b.pt"))
    return pocket_path, seq


def compute_km_loss_and_pcc(pred_km, y_km):

    if isinstance(pred_km, np.ndarray):
        pred_km = torch.from_numpy(pred_km).float()
    if isinstance(y_km, list):
        y_km = torch.tensor([float('nan') if x is None else x for x in y_km], dtype=torch.float32)
    elif isinstance(y_km, np.ndarray):
        y_km = torch.from_numpy(y_km).float()
    
    # 生成有效样本的掩码
    mask_km = ~torch.isnan(y_km)
    
    loss_km, pcc_km = 0.0, None
    
    if mask_km.any():
        # 提取有效样本并确保一维形状
        valid_pred_km = pred_km[mask_km].flatten()
        valid_y_km = y_km[mask_km].flatten()
        
        # 计算 MSE 损失
        loss_km = F.mse_loss(valid_pred_km, valid_y_km)
        
        # 计算 Pearson 相关系数
        if valid_pred_km.size(0) > 1:
            mean_pred = valid_pred_km.mean()
            mean_y = valid_y_km.mean()
            diff_pred = valid_pred_km - mean_pred
            diff_y = valid_y_km - mean_y
            
            numerator = torch.sum(diff_pred * diff_y)
            denominator = torch.sqrt(torch.sum(diff_pred ** 2) * torch.sum(diff_y ** 2))
            
            if denominator == 0:
                # 处理分母为零的情况
                if torch.all(diff_pred == 0) and torch.all(diff_y == 0):
                    pcc_km = 1.0  # 两者均为常数，视为完全相关
                else:
                    pcc_km = None  # 无法计算 PCC
            else:
                pcc_km = (numerator / denominator).item()

    
            # np.save("valid_pred_kcat.npy", valid_pred_km.detach().cpu().numpy())
            # np.save("valid_y_kcat.npy", valid_y_km.detach().cpu().numpy())
    return loss_km, pcc_km, pred_km
def val(model, dataloader, device):
    model.eval()

    pred_kcat_list = []
    pred_km_list = []
    for data in dataloader:
        
        with torch.no_grad():
            data = [data[i].to(device) for i in range(len(data))]

            pred_kcat, pred_km,_,_,_,_,_,_,_,_,_,_ = model(data)
            

            pred_kcat_list.append(pred_kcat.detach().cpu().numpy())
            pred_km_list.append(pred_km.detach().cpu().numpy())
            
            
    # pred = np.concatenate(pred_list, axis=0)
    # label = np.concatenate(label_list, axis=0)
    pred_kcat = np.concatenate(pred_kcat_list, axis=0)
    pred_km = np.concatenate(pred_km_list, axis=0)
    log_pred_kcat = pred_kcat
    log_pred_km = pred_km 
    log_pred_kcat_km = log_pred_kcat - log_pred_km 
    return log_pred_kcat, log_pred_km, log_pred_kcat_km


def get_ca_coords_and_sequence(pdb_file, chain_id):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    # 只考虑第一个模型
    model = structure[0]
    
    # 尝试获取指定链
    try:
        chain = model[chain_id]
    except KeyError:
        available_chains = list(model.child_dict.keys())
        raise ValueError(f"链 {chain_id} 不存在。可用链: {available_chains}")
    

    coords = []
    seq = []
    for residue in chain:

        if residue.id[0].strip() != "":
            continue
            
        resname = residue.resname.strip()
        

        if is_aa(residue) and residue.has_id('CA'):
            ca = residue['CA']
            coords.append(ca.coord)
            

            try:
                aa_code = three_to_one(resname)
                seq.append(aa_code)
            except:

                seq.append('X')
    
    return np.array(coords, dtype=np.float32), ''.join(seq)


    
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
    share_score = args.get("share_score")
    pooling = args.get("pooling")
    vocab_size = args.get("vocab_size")
    num_layers = args.get("num_layers")
    dropout = args.get("dropout")
    cross_attention = args.get("cross_attention")
    num_heads = args.get("num_heads")
    ligand_nn_embedding = args.get("ligand_nn_embedding")
    HeteroGNN_layers = args.get("HeteroGNN_layers")
    num_fc_layers = args.get("num_fc_layers")
    fc_hidden_dim = args.get("fc_hidden_dim")  # hidden_dim *  fc_hidden_dim
    share_fc = args.get("share_fc")
    # data_dir = os.path.join(data_root, 'structure_enzyme')
    # data_root = '/export/home/luod111/PLmodel/supervised/data/pdbbind'
    graph_type = 'graphkcat'
    batch_size =1
    # valid_dir = os.path.join(data_root, 'valid')
    clf = UniMolRepr(data_type='molecule', 
                remove_hs=False,
                model_name='unimolv2', # avaliable: unimolv1, unimolv2
                model_size='1.1B', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                )
    # data_type = "output_enz_miner_rzs"
    # test2016_df = pd.read_csv(os.path.join(data_root, "modeling-datasets", f"test_dataset_clean_no_structure.csv" ))
    model_esm, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model_esm.eval()

    test_df = pd.read_csv("test.csv")
    organism_set = "./sub_utils/all_organism_set.npy"
    temp_set = "./sub_utils/temp_set.npy"
    preprocessing(test_df, clf, model_esm, alphabet, batch_converter, cutoff=8)
    
    test2016_set = GraphDataset(test_df, organism_set, temp_set, dis_threshold=8)
    test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    load_model_dict(model, f"./checkpoint/paper.pt")
# load_model_dict(model, )
    model = model.to(device)

    # model.eval()
    # valid_rmse_kcat, valid_pr_kcat, valid_rmse_km, valid_pr_km = val(model, valid_loader, device)
    os.makedirs("./inference_results",exist_ok=True)
    log_pred_kcat, log_pred_km, log_pred_kcat_km = val(model, test2016_loader, device)
    test_df["pred_log_kcat_graphkcat"] = log_pred_kcat
    test_df["pred_log_km_graphkcat"] = log_pred_km
    # test2016_df["pred_log_kcat_km_graphkcat"] = pred_kcat_km
    test_df["pred_log_kcat_km_graphkcat"] =  log_pred_kcat_km
    # query_pdb = "inference_results/ref_rzs.pdb"
    # mining_pdbs_path = os.path.join(data_dir,data_type)
    # tm_score_list = []
    # rmsd_list = []
    # for pdb_name in test2016_df["id"].values.tolist():
    #     mining_pdb_path = os.path.join(mining_pdbs_path, pdb_name,  f"{pdb_name}_ligand_complex.pdb")
    #     tm_score_results = calculate_tm_score(query_pdb, mining_pdb_path, chain_id1='A', chain_id2='A')
    #     tm_score_list.append(tm_score_results['tm_norm_1'])
    #     rmsd_list.append(tm_score_results['rmsd'])
    # test2016_df["tm_score_vs_query"] = tm_score_list
    # test2016_df["rmsd_vs_query"] = rmsd_list
    test_df.to_csv(f"./inference_results.csv", index=False)
    # msg2 = "test_rmse_kcat-%.4f, test_pr_kcat-%.4f, test_rmse_km-%.4f, test_pr_km-%.4f " \
    #          % (test2016_valid_rmse_kcat, test2016_valid_pr_kcat, test2016_valid_rmse_km, test2016_valid_pr_km)
            
    # print(msg1)
    # print(msg2)
    # %%
