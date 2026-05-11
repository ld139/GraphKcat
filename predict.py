import os
import shutil
import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from tqdm import tqdm
import torch.nn.functional as F
import gc
# Model and Utils imports
from model_enz import bottle_view_graph
from dataset_graphkcat_chai1 import GraphDataset, PLIDataLoader
from config.config_dict import Config
import esm
from unimol_tools import UniMolRepr
import csv
import traceback

from preprocessing_inference import (
    extract_pocket_and_ligand, 
    get_pocket_by_sdf, 
    transfer_conformation, 
    extract_sequence_from_pdb, 
    get_unimol2_embedding, 
    get_esm2_embeddings,
    three_to_one
)

def preprocessing(df, clf, esm_model, alphabet, batch_converter, device, cutoff=8):
    
    print(f"Starting preprocessing for {len(df)} samples...")
    
    
    # esm_model = esm_model.to(device)
    esm_model = esm_model.to('cpu')
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        cid = str(row["id"])
        
        
        if 'complex' in row.index and row['complex'] and not pd.isna(row['complex']):
            data_dir = os.path.dirname(str(row["complex"]))
            has_complex = True
        else:
            data_dir = os.path.dirname(str(row["ligand"]))
            has_complex = False
            

        protein_path = None
        try:
            if has_complex:
                complex_path = str(row["complex"])
                if os.path.exists(complex_path):
                    extract_pocket_and_ligand(complex_path, cutoff=cutoff)
                    protein_path = complex_path
            else:
                ligand_path = str(row["ligand"])
                original_protein_path = str(row["protein"])
                

                expected_protein_path = os.path.join(data_dir, f"{cid}_protein.pdb")
                
                if not os.path.exists(expected_protein_path):
                    if os.path.exists(original_protein_path):
                        try:
                            shutil.copy(original_protein_path, expected_protein_path)
                        except shutil.SameFileError:
                            pass
                        except Exception as e:
                            print(f"Copy warning: {e}")
                
                
                if os.path.exists(ligand_path) and os.path.exists(original_protein_path):
                    if not os.path.exists(os.path.join(data_dir, f"Pocket_{cutoff}A.pdb")) and \
                       not os.path.exists(os.path.join(data_dir, f"Pocket_clean_{cutoff}A.pdb")):
                        get_pocket_by_sdf(ligand_path, original_protein_path, distance=cutoff)
                
                protein_path = original_protein_path

            
            target_sdf_path = os.path.join(data_dir, f"{cid}_ligand.sdf")
            target_pdb_path = os.path.join(data_dir, f"{cid}_ligand.pdb") 
            if not os.path.exists(target_sdf_path):
                if has_complex and os.path.exists(target_pdb_path):
                    
                    try:
                        print(f"[{cid}] Transferring conformation from {target_pdb_path} to {target_sdf_path}...")
                        transfer_conformation(target_pdb_path, row["Smiles"], target_sdf_path)
                    except Exception as e:
                        print(f"[{cid}] conformation transfer error: {e}")
                        
                elif not has_complex and str(row["ligand"]).endswith('.sdf') and os.path.exists(str(row["ligand"])):
                    shutil.copy(str(row["ligand"]), target_sdf_path)

        except Exception as e:
            print(f"File setup error for {cid}: {e}")


        unimol_path = os.path.join(data_dir, f"{cid}_unimol_1b.pt")
        esm_path = os.path.join(data_dir, f"{cid}_esm2_3b.pt")
        
        if os.path.exists(unimol_path) and os.path.exists(esm_path):
            continue 

        try:
            
            seq = ""
            if protein_path and os.path.exists(protein_path):
                seq = extract_sequence_from_pdb(protein_path)
            
            # 2. UniMol
            if not os.path.exists(unimol_path):
                mol_embedding = get_unimol2_embedding(clf, row["Smiles"], embedding_type="atomic_reprs")
                torch.save(mol_embedding, unimol_path)
            
            # 3. ESM
            if not os.path.exists(esm_path):
                esm_embedding = get_esm2_embeddings(esm_model, alphabet, batch_converter, seq, mean=False)
                torch.save(esm_embedding, esm_path)

                
        except Exception as e:
            print(f"Embedding error for {cid}: {e}")
            traceback.print_exc()
            if os.path.exists(unimol_path): os.remove(unimol_path)
            if os.path.exists(esm_path): os.remove(esm_path)
            continue

    return

# ================= NEW VAL FUNCTION =================
def val(model, dataloader, device, result_csv_path, skipped_ids_path):
    model.eval()
    
    
    f_res = open(result_csv_path, 'a', newline='')
    writer = csv.writer(f_res)
    
    f_skip = open(skipped_ids_path, 'a')
    
    
    pbar = tqdm(dataloader, desc="Inference (Stream Mode)")
    
    with torch.inference_mode():
        for step, (batch_data_list, batch_ids) in enumerate(pbar):
            
            
            if len(batch_data_list) == 0:
                print(f"\n[Warning] Batch {step} is empty. Skipping...")
                continue

            try:
                
                data_gpu = [d.to(device, non_blocking=True) for d in batch_data_list]
                
                
                pred_kcat, pred_km, _, _, _, _, _, _, _, _, _, _ = model(data_gpu)
                
                
                kcat_vals = pred_kcat.detach().cpu().numpy().flatten().tolist()
                km_vals = pred_km.detach().cpu().numpy().flatten().tolist()
                
                # 5. 写入 CSV
                for i, cid in enumerate(batch_ids):
                    k = kcat_vals[i]
                    m = km_vals[i]
                    eff = k - m
                    writer.writerow([cid, k, m, eff])
                
                
                del pred_kcat, pred_km, data_gpu, kcat_vals, km_vals
                
                
                if step % 5 == 0:
                    f_res.flush() #
                    torch.cuda.empty_cache()
                    gc.collect()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    pbar.write(f"[Warning] Batch {step} OOM. Skipping...")
                    for cid in batch_ids:
                        f_skip.write(f"{cid}\n")
                    f_skip.flush()
                    
                    
                    if 'data_gpu' in locals(): del data_gpu
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise e
                    
    f_res.close()
    f_skip.close()
    return
# ====================================================

def load_model_dict(model, ckpt_path):
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = None
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unknown checkpoint format. Type is {type(checkpoint)}")

    new_state_dict = {}
    is_module_prefix = False
    for k, v in state_dict.items():
        if k.startswith('module.'):
            is_module_prefix = True
            new_state_dict[k[7:]] = v 
        else:
            new_state_dict[k] = v
    
    if is_module_prefix:
        state_dict = new_state_dict

    try:
        msg = model.load_state_dict(state_dict, strict=True)
        print(f"Model loaded successfully. {msg}")
    except RuntimeError as e:
        print("Strict loading failed. Trying with strict=False...")
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded with strict=False. Missing keys: {msg.missing_keys}")

def parse_args():
    parser = argparse.ArgumentParser(description="GraphKcat Prediction")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--cpkt_path', type=str, default='./checkpoint/paper.pt', help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--cfg', type=str, default='TrainConfig_kcat_enz', help='Configuration file name')
    parser.add_argument('--organism_set_path', type=str, default='./sub_utils/all_organism_set.npy', help='Path to the organism set file')
    parser.add_argument('--temp_set_path', type=str, default='./sub_utils/temp_set.npy', help='Path to the temporary set file')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    return parser.parse_args()

# ================= NEW MAIN FUNCTION =================
def main():
    args = parse_args()
    
    
    cfg = args.cfg
    config = Config(cfg)
    config = config.get_config()   
    
    batch_size = args.batch_size 
    
    
    hidden_dim = config.get("hidden_dim")
    pooling = config.get("pooling")
    vocab_size = config.get("vocab_size")
    num_layers = config.get("num_layers")
    dropout = config.get("dropout")
    ligand_nn_embedding = config.get("ligand_nn_embedding")
    HeteroGNN_layers = config.get("HeteroGNN_layers")
    num_fc_layers = config.get("num_fc_layers")
    fc_hidden_dim = config.get("fc_hidden_dim")
    share_fc = config.get("share_fc")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ================= [Phase 1] Feature Engineering =================
    print("Initializing UniMol...")
    clf = UniMolRepr(data_type='molecule', remove_hs=False, model_name='unimolv2', model_size='1.1B')

    print("Initializing ESM-2...")
    model_esm, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model_esm.eval()
    # model_esm = model_esm.to(device)
    model_esm = model_esm.to('cpu')

    # Load Data
    test_df = pd.read_csv(args.csv_file)
    test_df['id'] = test_df['id'].astype(str)
    
    # Run Preprocessing
    preprocessing(test_df, clf, model_esm, alphabet, batch_converter, device, cutoff=8)
    
    
    print("Preprocessing done. Freeing VRAM...")
    del model_esm
    del clf
    del batch_converter
    gc.collect()
    torch.cuda.empty_cache()
    print(f"VRAM freed. Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    # ==============================================================

    # ================= [Phase 2] Inference =================
    
    organism_set = args.organism_set_path
    temp_set = args.temp_set_path
    
    
    test2016_set = GraphDataset(test_df, organism_set, temp_set, dis_threshold=8)
    test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"\n[Debug] Dataset loaded {len(test2016_set)} items.")
    # Initialize Main Model
    model = bottle_view_graph(node_dim=35,
                             hidden_dim=hidden_dim,
                             HeteroGNN_layers=HeteroGNN_layers,
                             pooling=pooling,
                             vocab_size=vocab_size,
                             num_layers=num_layers,
                             dropout=dropout,
                             ligand_nn_embedding=ligand_nn_embedding,
                             num_fc_layers=num_fc_layers,
                             fc_hidden_dim=fc_hidden_dim,
                             share_fc=share_fc)
                             
    load_model_dict(model, args.cpkt_path)
    model = model.to(device)
    
    
    temp_res_path = output_dir / "temp_predictions.csv"
    skipped_ids_path = output_dir / "skipped_ids.txt"
    
   
    with open(temp_res_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'pred_log_kcat_graphkcat', 'pred_log_km_graphkcat', 'pred_log_kcat_km_graphkcat'])

    # Run Inference (Stream Mode)
    print("Running inference (Stream Mode)...")
    val(model, test2016_loader, device, temp_res_path, skipped_ids_path)
    
    # ================= [Phase 3] Post-Processing Merge =================

    if os.path.exists(temp_res_path):
        print("Inference finished. Merging results...")
        pred_df = pd.read_csv(temp_res_path)
        pred_df['id'] = pred_df['id'].astype(str) #
        
        
        final_df = pd.merge(test_df, pred_df, on='id', how='inner')
        input_name = Path(args.csv_file).stem
        save_path = output_dir / f"{input_name}_predictions.csv"
        final_df.to_csv(save_path, index=False)
        print(f"Successfully saved {len(final_df)} results to {save_path}")
        
        
        # os.remove(temp_res_path)
    else:
        print("Error: No predictions generated.")

if __name__ == '__main__':
    main()