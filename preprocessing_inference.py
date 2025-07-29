# %%
import os
from rdkit import Chem
from rdkit.Chem import rdFMCS
import pandas as pd
from tqdm import tqdm
import pymol
from pymol import cmd
from rdkit import RDLogger
# from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import PDBParser
# import warnings
import torch
RDLogger.DisableLog('rdApp.*')

# %%

# import pymol
# from pymol import cmd
# import os
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N',
    'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'SEC': 'U', 'PYL': 'O'
}
def transfer_conformation(pdb_file, smiles, output_sdf):
    # 生成正确键连的基础分子（不含氢）
    base_mol = Chem.MolFromSmiles(smiles)
    if not base_mol:
        raise ValueError("Invalid SMILES string")
    base_mol = Chem.RemoveHs(base_mol)

    # 读取PDB文件（不含氢）
    pdb_mol = Chem.MolFromPDBFile(pdb_file, removeHs=True)
    if not pdb_mol:
        raise ValueError(f"Failed to read PDB file of {pdb_file}")

    # 验证原子数量一致
    if base_mol.GetNumAtoms() != pdb_mol.GetNumAtoms():
        raise ValueError("Atom count mismatch between SMILES and PDB")

    # 寻找最大公共子结构进行原子映射
    mcs = rdFMCS.FindMCS(
        [base_mol, pdb_mol],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=False,
        timeout=60
    )
    
    if mcs.numAtoms != base_mol.GetNumAtoms():
        raise ValueError("Failed to find complete atom mapping")

    # 获取原子映射关系
    mcs_pattern = Chem.MolFromSmarts(mcs.smartsString)
    base_match = base_mol.GetSubstructMatch(mcs_pattern)
    pdb_match = pdb_mol.GetSubstructMatch(mcs_pattern)

    # 创建新构象并转移坐标
    conformer = Chem.Conformer(base_mol.GetNumAtoms())
    pdb_conf = pdb_mol.GetConformer()
    for base_idx, pdb_idx in zip(base_match, pdb_match):
        position = pdb_conf.GetAtomPosition(pdb_idx)
        conformer.SetAtomPosition(base_idx, position)
    base_mol.AddConformer(conformer)

    # 添加氢原子并优化
    mol_with_h = Chem.AddHs(base_mol, addCoords=True)
    
    # 设置力场优化参数（固定重原子）
    heavy_atoms = [a.GetIdx() for a in mol_with_h.GetAtoms() if a.GetAtomicNum() > 1]
    try:
        # 尝试使用UFF力场
        ff = AllChem.UFFGetMoleculeForceField(mol_with_h)
        for idx in heavy_atoms:
            ff.AddFixedPoint(idx)
        ff.Minimize(maxIts=200)
    except:
        # 回退到MMFF力场
        try:
            AllChem.MMFFSanitizeMolecule(mol_with_h)
            mp = AllChem.MMFFGetMoleculeProperties(mol_with_h)
            ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, mp)
            for idx in heavy_atoms:
                ff.AddFixedPoint(idx)
            ff.Minimize(maxIts=200)
        except:
            pass  # 如果优化失败则保留原始坐标

    # 写入SDF文件
    writer = Chem.SDWriter(output_sdf)
    writer.write(mol_with_h)
    writer.close()

def extract_pocket_and_ligand(complex, cutoff=8):
    # ids = data_df['id'].values.tolist()
    # for i, row in data_df.iterrows():
        # id = row['id']
        pdbfile_path = complex

        
        # 初始化PyMOL
        pymol.finish_launching(['pymol', '-cq'])
        
        filepath = os.path.dirname(pdbfile_path)
        file_name = os.path.basename(pdbfile_path).split('.')[0]

        # 加载PDB文件
        cmd.load(pdbfile_path, "complex")

        # 创建保存路径
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # 选择并去除氢原子
        cmd.remove("hydrogen")

        # 提取配体（LIG）部分
        if os.path.exists(os.path.join(filepath, f"{file_name}_ligand.pdb")):
            os.remove(os.path.join(filepath, f"{file_name}_ligand.pdb"))
        cmd.select("ligand", "resn UNK")
        cmd.save(os.path.join(filepath, f"{file_name}_ligand.pdb"), "ligand")

        # 提取蛋白质部分（不包含配体）
        cmd.select("protein", "not resn UNK")
        cmd.save(os.path.join(filepath, f"{file_name}_protein.pdb"), "protein")

        # 提取口袋（蛋白质中距离配体cutoff范围内的氨基酸）
        cmd.select("pocket", f"byres (protein within {cutoff} of ligand)")  # 关键修改点
        # if os.path.exists(os.path.join(filepath, f"pocket_{cutoff}A.pdb")):
        #     os.remove(os.path.join(filepath, f"pocket_{cutoff}A.pdb"))
            
        cmd.save(os.path.join(filepath, f"Pocket_{cutoff}A.pdb"), "pocket")

        # 关闭PyMOL
        cmd.delete("all")
        # cmd.quit()
def get_pocket_by_sdf(ref_ligand, protein, distance=10):
    """
    This function gets the pocket of the protein using pymol
    """
    import pymol
    pymol.cmd.load(protein, "protein")
    pymol.cmd.remove("resn HOH")
    pymol.cmd.remove("not polymer.protein")
    pymol.cmd.load(ref_ligand, "ligand")
    pymol.cmd.remove("hydrogens")
    pymol.cmd.select("Pocket", f"byres ligand around {distance}")

    out_dir = os.path.dirname(ref_ligand)
    pocket_path = os.path.join(out_dir, f"Pocket_{distance}A.pdb")
    pymol.cmd.save(pocket_path, "Pocket")
    pymol.cmd.delete("all")

    return pocket_path

def generate_sdf_from_pdb(data_df, data_dir):
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        cid = row['id']
        pdbfile_path = os.path.join(data_dir, cid, f'{cid}_ligand.pdb')
        smiles = row['Smiles']
        # print(pdbfile_path)
        # print(smiles)
        # print(cid)
        output_sdf = os.path.join(data_dir, cid, f'{cid}_ligand.sdf')
        try:
            transfer_conformation(pdbfile_path, smiles, output_sdf)
        except Exception as e:
            print(f"Failed to process {cid}: {e}")
            data_df.drop(i, inplace=True)
            print(f"Drop {cid}")
        pbar.update(1)
    return data_df

def extract_sequence_from_pdb(pdb_file):
    """
    Extracts the sequence of the first chain in the first model of a PDB file.
    Returns the single-letter sequence as a string.
    """
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # 获取第一个模型
    model = next(structure.get_models(), None)
    if model is None:
        return ""  # 没有模型
    
    # 获取第一条链
    chain = next(model.get_chains(), None)
    if chain is None:
        return ""  # 没有链
    
    sequence = []
    for residue in chain:
        if residue.id[0] == ' ':  # 只处理标准氨基酸残基
            resname = residue.resname.strip()
            if resname in three_to_one:
                sequence.append(three_to_one[resname])
            else:
                sequence.append('X')
    return ''.join(sequence)

def get_esm2_embeddings(model,alphabet,batch_converter, seq, mean = False):
    data = [(f"protein1", seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        if mean:
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))  # [2560]
        else:
        # sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1])  # [seq_len, 2560]
    return sequence_representations

def get_unimol2_embedding(clf, smiles,embedding_type = "atomic_reprs"):
    """
    Get unimol2 embedding from smiles
    """
    # smiles = "C1=CC=C(C=C1)C(=O)O"
    smiles_list = [smiles]
    unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)

    # print(np.array(unimol_repr['cls_repr']).shape)
    # atomic level repr, align with rdkit mol.GetAtoms()
    # print(np.array(unimol_repr['atomic_reprs']).shape)
    # print(type(unimol_repr['cls_repr'])) # list
    # print(len(unimol_repr['cls_repr']))
    uni_mol2_embedding = torch.tensor(unimol_repr[embedding_type])
    return uni_mol2_embedding

if __name__ == '__main__':
    distance = 8
    # pdbfile_path = '/export/home/luod111/chai1/sturcture_enzyme/test/mpek0/mpekmpek0.pdb'
    # data_type = 'train'
    # data_df = pd.read_csv(f'/export/home/luod111/chai1/modeling-datasets/{data_type}_dataset_clean_no_structure.csv')
    # data_dir = f'/export/home/luod111/chai1/structure_enzyme/{data_type}/'

    # extract_pocket_and_ligand(data_df, data_dir, cutoff=distance)

    # data_df = generate_sdf_from_pdb(data_df, data_dir)
    # data_df.to_csv(f'/export/home/luod111/chai1/modeling-datasets/{data_type}_dataset_clean_no_structure.csv', index=False)
    
    # data_df = generate_complex_v1(data_dir, data_df, distance=distance, input_ligand_format='sdf')
    # data_df.to_csv(f'/export/home/luod111/chai1/modeling-datasets/{data_type}_dataset_clean_no_structure.csv', index=False)
    data_type ="enz_miner_rzs"
    data_df = pd.read_csv("/export/home/luod111/chai1/modeling-datasets/enz_miner_rzs.csv")
    data_dir = f'/export/home/luod111/chai1/structure_enzyme/output_{data_type}/'
    extract_pocket_and_ligand(data_df, data_dir, cutoff=distance)
    data_df = generate_sdf_from_pdb(data_df, data_dir)
    data_df = generate_complex_v1(data_dir, data_df, distance=distance, input_ligand_format='sdf')
    data_df.to_csv(f'/export/home/luod111/chai1/modeling-datasets/{data_type}_final.csv', index=False)


   


# %%
