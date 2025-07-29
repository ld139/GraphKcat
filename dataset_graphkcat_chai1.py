# %%
import os
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import networkx as nx
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import SanitizeMol, SanitizeFlags
from torch_geometric.data import HeteroData
from Bio.PDB import PDBParser, PDBParser, Structure, Model, Chain,PDBIO
# from Bio.PDB.PDBIO import PDBIO
from model_utils import get_clean_res_list, vocab_dict
from preprocessing_inference import three_to_one
from mol_bpe import Tokenizer
import torch_cluster
import warnings
# sys.path.append("ps")
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

# %%
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))


def get_edge_index(mol, graph):
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feats = bond_features(bond)
        graph.add_edge(i, j)
        edge_features.append(feats)
        edge_features.append(feats)

    return torch.stack(edge_features)

def bond_features(bond):
    # 返回一个[1,6]的张量，表示一键的各种信息是否存在
    bt = bond.GetBondType() # 获取键的类型
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
             bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing(),bond.GetIsConjugated()]
    return torch.Tensor(fbond)

def bond_features_sub_graph(edge_index,pos_sub_graph):
    # print(pos_sub_graph)
    pos_matrix = distance_matrix(pos_sub_graph, pos_sub_graph)
    edge_features = []
    # print(edge_index)
    if len(pos_sub_graph) == 1:
        return torch.zeros((1, 16)) # 1 * 16

    for i,j in edge_index:
        # print(i,j)
        edge_features.append(get_rbfs(pos_matrix[i,j], cg="ligand"))

    # if len(edge_features) == 0:
    #     print("edge_index:", edge_index)
    #     raise ValueError("No edge features")
    return torch.stack(edge_features)
    


def mol2graph(mol):
    # print(mol.GetNumAtoms())
    graph = nx.Graph()
    if mol.GetNumAtoms() == 1:
        atom = mol.GetAtomWithIdx(0)
        
        atom_features(mol, graph)
        # graph.add_node(0, feats=torch.from_numpy(atom_feats))
        # ture or false to 0 or 1
        # self loop
        edge_index = torch.tensor([0, 0]).unsqueeze(0).T
        edge_features = torch.tensor([0, 0, 0, 0, 0, 0]).unsqueeze(0)
        graph.add_edge(0, 0)
        # x = torch.tensor([atom_feats])
        graph = graph.to_directed()
        x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
        edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
        # print(x.shape, edge_index.shape, edge_features.shape)
        
    else:
        atom_features(mol, graph)
        edge_features = get_edge_index(mol, graph)
    # atom_features(mol, graph)
    # edge_features = get_edge_index(mol, graph)
        graph = graph.to_directed()
        x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
        edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    # degrees = torch.tensor([graph.degree(node) for node in graph.nodes()])

    return x, edge_index, edge_features

def get_rbfs(distances, num_rbf=16, cg="protein"):
    # distances: [1]
    if cg =="protein":
        d_min, d_max = 2, 22
    else:
        d_min, d_max = 2, 5
    d = torch.linspace(d_min, d_max, num_rbf)
    d_sigma = (d_max - d_min) / num_rbf
    rbf = torch.exp(-((distances - d) / d_sigma) ** 2) # [num_rbf]
    # print(type(rbf))
    return rbf
    

def inter_graph_cg(pos_sub_graph, pos_protein, dis_threshold=8.0):
    """
    Construct ligand-protein interaction graph.

    Args:
        pos_sub_graph (np.ndarray): Positions of ligand nodes, shape [num_ligand_nodes, 3].
        pos_protein (np.ndarray): Positions of protein nodes, shape [num_protein_nodes, 3].
        dis_threshold (float): Distance threshold for edges.

    Returns:
        edge_index_inter (torch.Tensor): Edge indices, shape [2, num_edges].
        edge_attrs_inter (torch.Tensor): Edge attributes, shape [num_edges, 16].
    """
    # Compute distance matrix between ligand and protein nodes
    dis_matrix_lp = distance_matrix(pos_sub_graph, pos_protein)

    # Find pairs of nodes within the distance threshold
    node_idx_lp = np.where(dis_matrix_lp < dis_threshold)
    src = node_idx_lp[0]  # Ligand node indices (source)
    dst = node_idx_lp[1]  # Protein node indices (target)

    # Generate edge index
    edge_index_inter = torch.tensor([src, dst+pos_sub_graph.shape[0]], dtype=torch.long) # the index of protein nodes should be added by the number of ligand nodes

    edge_index_inter = torch.cat([edge_index_inter, edge_index_inter.flip(0)], dim=1)  # 添加反向边
    # src = torch.cat([src, dst]) # 
    # Generate edge attributes
    edge_attrs_inter = torch.stack([get_rbfs(dis_matrix_lp[i, j], cg="protein") for i, j in zip(src, dst)])

    edge_attrs_inter = torch.cat([edge_attrs_inter, edge_attrs_inter], dim=0)  # 添加反向边的属性
    # Sanity checks
    assert edge_index_inter.dtype == torch.long, "edge_index_inter must be torch.long"
    assert edge_attrs_inter.dtype == torch.float32, "edge_attrs_inter must be torch.float32"
    assert edge_index_inter.shape[0] == 2, f"edge_index_inter must have shape [2, num_edges], got {edge_index_inter.shape}"
    assert edge_attrs_inter.shape[0] == edge_index_inter.shape[1], (
        "Number of edges in edge_attrs_inter must match edge_index_inter"
    )

    return edge_index_inter, edge_attrs_inter

def _dihedrals(X, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features

def one_hot_seq(seq):
    three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
    letter = 'ACDEFGHIKLMNPQRSTVWY'
    one_hot = np.zeros((len(seq), 20))
    for i, aa in enumerate(seq):
        if aa in letter:
            one_hot[i, letter.index(aa)] = 1
        else:
            one_hot[i, letter.index(three_to_one[aa])] = 1
    one_hot = torch.tensor(one_hot).float()
    return one_hot

def get_pro_node_batch(rdkit_mol, res_list_coords):
    pos_mol = rdkit_mol.GetConformer().GetPositions()
    node_batch = []
    # for i in range(len(pos_mol)):
    all_res_atoms = []
    for res in res_list_coords:
        all_res_atoms.extend(res)
    # print(len(all_res_atoms))
    for j, res in enumerate(res_list_coords):
        node_batch.extend([j] * len(res))

    # print(len(node_batch), len(pos_mol))
    assert len(node_batch) == len(pos_mol)

    return node_batch
            

def get_protein_graph(protein, pocket, saprot_feature=None, topk=16):


    protein_res_list = get_clean_res_list(protein.get_residues(), verbose=False, ensure_ca_exist=True)
    pocket_res_list = get_clean_res_list(pocket.get_residues(), verbose=False, ensure_ca_exist=True)
    pocket_coords_full, pocket_coords, pocket_coords_main = get_pro_coord(pocket_res_list)
    # print(protein_res_list)
    # print(pocket_res_list)
    # print(len(all_res_atoms))
    # saprot_feature = torch.load(saprot_feature, map_location='cpu')
    node = _dihedrals(pocket_coords_full)
    seq = "".join([three_to_one.get(res.resname) for res in pocket_res_list])
    feat_one_hot = one_hot_seq(seq)
    if saprot_feature is not None:
        saprot_feature_pocket = get_pcoket_saprot_feature(protein, pocket, saprot_feature)

        node = torch.cat([node, feat_one_hot, saprot_feature_pocket], dim=1)
    else:
        node = torch.cat([node, feat_one_hot], dim=1)  # 20 +

    coord_ca = pocket_coords_main[:, 1]
    edge_index_pro = torch_cluster.knn_graph(coord_ca, k=topk)
    edge_index_pro_T = edge_index_pro.T
    position_embedding_pro = positional_embeddings(edge_index_pro)
    rbfs= []
    for i, j in edge_index_pro_T:
        dis = torch.norm(coord_ca[i] - coord_ca[j]) # l2 norm = sqrt(sum(x_i - y_i)^2)
        rbfs.append(get_rbfs(dis, cg="protein"))
    edge_attr_rbf = torch.stack(rbfs)
    edge_attr = torch.cat([edge_attr_rbf, position_embedding_pro], dim=1) # 16+16
    # print(edge_attr.shape, node.shape, edge_index_pro.shape)
    return node, edge_index_pro, edge_attr, coord_ca, pocket_coords


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def positional_embeddings(edge_index, 
                            num_embeddings=16,
                            period_range=[2, 1000],device='cpu'):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_embeddings = num_embeddings
    d = edge_index[0] - edge_index[1]
    
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

def get_organism_index(organism_set, organism):
    # 
    if isinstance(organism_set, np.ndarray):
        organism_set = organism_set.item()  
    
    if not isinstance(organism_set, set):
        raise TypeError(" input must be a set or a NumPy array")
    
    # 
    sorted_organisms = sorted(organism_set - {'unknown'})  #
    sorted_organisms.append('unknown')  #
    
    organism_dict = {org: idx for idx, org in enumerate(sorted_organisms)}

    return organism_dict.get(organism, organism_dict['unknown'])



def bin_encoding(value, method='onehot'):
    """
    支持非均匀分箱的编码
    value: 输入标量 (如pH=6.3)
    bins: 递增分箱边界，如generate_ph_bins()的输出
    method: 编码方式
    """
    # 确定所属区间
    def generate_ph_bins():
        """ 生成包含强酸、0.5间隔、强碱的分箱边界 """
        strong_acid = [0, 2]                # 强酸区间: 0 <= pH < 2
        middle = np.arange(2, 12, 0.5)      # 中间区间: 2 <= pH <12，每0.5一个bin
        strong_base = [12, 14]              # 强碱区间: 12 <= pH <=14
        return np.concatenate([strong_acid, middle, strong_base])

    bins = generate_ph_bins()
    bin_idx = np.digitize(value, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins)-2)  # 约束索引范围
    
    if method == 'ordinal':
        return bin_idx
    elif method == 'onehot':
        onehot = np.zeros(len(bins)-1)
        onehot[bin_idx] = 1
        return onehot
    else:
        raise ValueError("Method must be 'onehot' or 'ordinal'")


def generate_temp_bins(train_temps, bin_width=1.0):
    """
    生成分箱边界，包含低于min和高于max的特殊箱
    train_temps: 训练集温度Tensor
    bin_width: 分箱宽度（如1.0或2.0）
    """
    min_temp = train_temps.min().item()
    max_temp = train_temps.max().item()
    
    # 生成主分箱（不包含两端）
    lower = min_temp
    upper = max_temp
    num_main_bins = int((upper - lower) // bin_width) + 1
    
    main_bins = torch.arange(
        start=lower,
        end=lower + num_main_bins*bin_width + 1e-6,  # 避免浮点误差
        step=bin_width
    )
    
    # 添加两端特殊分箱
    bins = torch.cat([
        torch.tensor([-torch.inf]),  # 左端：<min_temp
        main_bins,
        torch.tensor([torch.inf])    # 右端：>max_temp
    ])
    
    return bins

def generate_temp_bins_from_set(train_temp_set, bin_width=1.0):
    """
    从温度集合生成分箱边界，包含低于min和高于max的特殊箱
    train_temp_set: 训练集温度集合（Python set）或包含集合的0维NumPy数组
    bin_width: 分箱宽度（如1.0或2.0）
    """
    # 修复1：处理NumPy数组包装的集合
    if isinstance(train_temp_set, np.ndarray):
        train_temp_set = train_temp_set.item()  # 提取真正的Python集合
    
    # 修复2：确保输入为可迭代对象
    if not isinstance(train_temp_set, (set, list, tuple)):
        raise TypeError("输入必须是Python集合、列表或元组")
    
    # 转换为排序后的Tensor
    sorted_temps = torch.tensor(sorted(train_temp_set), dtype=torch.float32)
    
    # 后续逻辑保持不变...
    min_temp = sorted_temps[0].item()
    max_temp = sorted_temps[-1].item()
    
    # 生成主分箱（不包含两端）
    lower = min_temp
    upper = max_temp
    num_main_bins = int((upper - lower) // bin_width) + 1
    
    main_bins = torch.arange(
        start=lower,
        end=lower + num_main_bins*bin_width + 1e-6,  # 避免浮点误差
        step=bin_width
    )
    
    # 添加两端特殊分箱
    bins = torch.cat([
        torch.tensor([-torch.inf]),  # 左端：<min_temp
        main_bins,
        torch.tensor([torch.inf])    # 右端：>max_temp
    ])
    
    return bins

def temp_bin_encoding(values, bins, method='onehot', device='cpu'):
    """
    处理标量输入的兼容版本
    """
    # 转换输入为至少1维张量
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float32)
    values = values.to(device).view(-1)  # 确保至少1维
    
    bins = bins.to(device)
    
    # 确定分箱索引
    bin_indices = torch.bucketize(values, bins, right=False) - 1
    bin_indices = torch.clamp(bin_indices, 0, len(bins)-2)
    
    # 编码输出
    n_bins = len(bins) - 1
    if method == 'ordinal':
        return bin_indices.to(torch.long).squeeze()
    elif method == 'onehot':
        onehot = torch.zeros(len(values), n_bins, device=device)
        onehot.scatter_(1, bin_indices.unsqueeze(1), 1)
        return onehot.squeeze(0)  # 输入为标量时返回1D张量
    else:
        raise ValueError("method必须为'onehot'或'ordinal'")

def inter_graph_all_atom(ligand, pocket, dis_threshold=5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformer().GetPositions()
    pos_p = pocket.GetConformer().GetPositions()

    pos = torch.cat([torch.FloatTensor(pos_l), torch.FloatTensor(pos_p)], dim=0)

    # 添加配体-配体和口袋-口袋之间的边
    dis_matrix_l = distance_matrix(pos_l, pos_l)
    dis_matrix_p = distance_matrix(pos_p, pos_p)
    dis_matrix_lp = distance_matrix(pos_l, pos_p)

    node_idx_l = np.where(dis_matrix_l < dis_threshold)
    for i, j in zip(node_idx_l[0], node_idx_l[1]):

        if i == j:
            continue
        # print(dis_matrix_l[i, j])
        # print(get_rbfs(dis_matrix_l[i, j], cg="ligand"))
        # feats = torch.cat([torch.tensor([1, 0, 0]), get_rbfs(dis_matrix_l[i, j], cg="ligand")])
        feats=torch.tensor([0, 0, 1, dis_matrix_l[i, j]])
        graph_inter.add_edge(i, j, feats=feats)



    node_idx_p = np.where(dis_matrix_p < dis_threshold)
    for i, j in zip(node_idx_p[0], node_idx_p[1]):
        if i == j:
            continue
        # feats = torch.cat([torch.tensor([0, 1, 0]), get_rbfs(dis_matrix_p[i, j], cg="ligand")])
        # feats = torch.cat([torch.tensor([0, 1, 0]), dis_matrix_p[i, j].reshape(1)])
        feats = torch.tensor([0, 1, 0, dis_matrix_p[i, j]])
        graph_inter.add_edge(i + atom_num_l, j + atom_num_l, feats=feats)

    node_idx_lp = np.where(dis_matrix_lp < dis_threshold)
    for i, j in zip(node_idx_lp[0], node_idx_lp[1]):
        # feats = torch.cat([torch.tensor([0, 0, 1]), get_rbfs(dis_matrix_lp[i, j], cg="ligand")])
        # feats = torch.cat([torch.tensor([0, 0, 1]), dis_matrix_lp[i, j].reshape(1)])
        feats = torch.tensor([1, 0, 0, dis_matrix_lp[i, j]])
        graph_inter.add_edge(i, j + atom_num_l, feats=feats)

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v, _ in graph_inter.edges(data=True)]).T
    edge_attrs_inter = torch.stack([feats['feats'] for _, _, feats in graph_inter.edges(data=True)]).float()

    return edge_index_inter, edge_attrs_inter, pos

def get_inter_graph_all_atom(ligand, pocket, dis_threshold=5.):

    x_l, _,_ = mol2graph(ligand)
    x_p, _,_ = mol2graph(pocket) 
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_inter, edge_attrs_inter, pos = inter_graph_all_atom(ligand, pocket, dis_threshold)

    return x, edge_index_inter, edge_attrs_inter, pos

def get_sub_graph_dict(vocab_txt):
    with open(vocab_txt, 'r') as f:
        vocab = f.readlines()
    vocab = [i.split()[0] for i in vocab if not i.startswith('{')] # remove the special tokens
    vocab_dict = {vocab[i]: i for i in range(len(vocab))}
    # print(vocab_dict)
    return vocab_dict

def get_subgraph_mol(mol):
    tokenizer = Tokenizer('./sub_utils/zinc_350.txt')
    # print(mol.GetNumAtoms())
    # smi = Chem.MolToSmiles(mol)

    # print(smi)
    total_mol = mol
    sub_mols = tokenizer.tokenize(mol)
    # print(sub_mols.get_node(0).smiles)
    # smiles = sub_mols.get_node(0).smiles
    # print(vocab_dict[smiles])
    len_sub_mols = len(sub_mols)
    # print(len_sub_mols)
    all_atom_pos = mol.GetConformer().GetPositions()
    # print(sub_mols)
    sub_mols_directed = sub_mols.to_directed()
    edges = sub_mols_directed.edges
    # print(edges)
    if len(edges) == 0:
        # self loop
        edges = [(0, 0)]        
    edge_index = torch.tensor(list(edges)).T
    # edge_features = bond_features_sub_graph(sub_mols, edges)
    
    subgraphs, subgraph_pos_mean, all_atom_index, node_class,subgraph_pos = [], [], [], [], []
    for i in range(len(sub_mols)):
        node = sub_mols.get_node(i)
        mol = node.mol
        # print(mol.smiles)
        sub_type = vocab_dict[node.smiles]
        # print(sub_type)
        SanitizeMol(mol, SanitizeFlags.SANITIZE_SETAROMATICITY)  # 保证分子的完整性
        subgraphs.append(mol2graph(mol))
        
        # single_atom.append(1)
        node_poses_index = sub_mols.get_node(i).atom_mapping
        node_poses =[all_atom_pos[i] for i in node_poses_index.keys()] #  atom map: {7: 4, 13: 6, 8: 5, 6: 3, 5: 2, 4: 1, 3: 0}
        # reverse node_poses_list as the order of atom in mol is reversed
        node_poses = [node_poses[i] for i in range(len(node_poses)-1, -1, -1)]
        

    
        node_pos = np.mean(node_poses, axis=0)
        subgraph_pos_mean.append(torch.FloatTensor(node_pos))
        all_atom_index.append(node_poses_index)
        subgraph_pos.append(torch.FloatTensor(node_poses))
        node_class.append(sub_type)
    pos = torch.stack(subgraph_pos_mean)
    node_class = torch.tensor(node_class)
    # print(len(subgraphs))
    # print(len(node_class))
    # single_atom = torch.tensor(single_atom)
    edge_features = bond_features_sub_graph(edges, pos)
    # print(pos.shape)
    # print(edge_features.shape)
    # print(edge_index.shape)

    node_batch = [] # 
    for i in range(len(all_atom_pos)):
        # i: 1,2,3,4,5,6,7,8,9,10
        for j, k in enumerate(all_atom_index):
            if i in k.keys():
                node_batch.append(j)  # get the index of the subgraph
    # print(node_batch)
                
                

    assert len(subgraphs) == len(node_class) == len(pos)

    return subgraphs, pos, node_batch, node_class, edge_features, edge_index, subgraph_pos

def get_CA_coord(res):
    # 获取残基的CA原子的坐标
    for atom in [res['N'], res['CA'], res['C'], res['O']]:
            if atom == res['CA']:
                return atom.coord

def get_pro_coord(pro,max_atom_num=24):
    # 获取残基的所有原子的坐标，包括side chain
    coords_pro = [] # 所有原子的坐标 list
    coords_pro_full = [] # full的坐标，不足的用nan填充  
    coords_main_pro = [] # 只包含主链N,CA,C原子的坐标

    for res in pro:
        # print(len(res.get_atoms()))
        # break
        atoms = res.get_atoms()
        # print(len(atoms))
        # print(dir(atoms))
        # break
        coords_res = []
        coords_res_main = []
        for atom in atoms:
            # print(atom.name)
            # break
            if atom.name in ['N', 'CA', 'C']:
                coords_res_main.append(atom.coord)
            coords_res.append(atom.coord)
            
        coords_res_main = np.array(coords_res_main)
        coords_res = np.array(coords_res)
        # print(len(coords_res))
        coords_res_full = np.concatenate([coords_res, np.full((max_atom_num - len(coords_res), 3),np.nan)], axis=0)
        # print(coords.shape)
        # break
        coords_main_pro.append(coords_res_main)
        coords_pro.append(coords_res) 
        coords_pro_full.append(coords_res_full)
    # print(coords_pro[0].shape)
        
    coords_pro = [torch.FloatTensor(coords) for coords in coords_pro]
    # coords_pro_full = torch.stack([torch.FloatTensor(coords) for coords in coords_pro_full])
    # get the batch of coord,like :[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
    # coords_pro_batch = torch.cat([torch.full((len(coords),), i) for i, coords in enumerate(coords_pro)])
    coords_main_pro = torch.tensor(coords_main_pro)
    coords_pro_full = torch.tensor(coords_pro_full)
    # print(coords_pro.shape)
    # print(coords_main_pro.shape)
    # print(coords_pro[0], coords_pro[1])
    return coords_pro_full, coords_pro, coords_main_pro
    
def res2pdb(res_list, pdbid, save_path):
    # 创建一个新的结构对象
    new_structure = Structure.Structure("New_Structure")
    model = Model.Model(0)
    new_structure.add(model)
    # 创建链的字典来存储不同链
    chains = {}

    for res in res_list:
        chain_id = res.get_full_id()[2]  # 获取原始链ID
        if chain_id not in chains:
            chain = Chain.Chain(chain_id)
            model.add(chain)
            chains[chain_id] = chain
        chains[chain_id].add(res)

    # 创建PDBIO对象并写入文件
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(save_path)


# %%
def mols2graphs(complex_path, pdbid, organism_set, temp_set, \
    organism, ph, temp, dis_threshold = 5):
    parser = PDBParser(QUIET=True)    

    organism_index = get_organism_index(organism_set, organism) # [1]
    organism_index = torch.tensor([organism_index]) # [1]

    ph_encoding = bin_encoding(ph)  # [3]
    ph_encoding = torch.FloatTensor(ph_encoding)  #[23]
    # temp_set = np.load('temp_set.npy', allow_pickle=True) 
    temp_bins = generate_temp_bins_from_set(temp_set)
    # print(temp_bins)
    temp_encoding = temp_bin_encoding(temp, temp_bins) #[103]
    # print(temp_encoding)
    res_list_pdb_dir = os.path.join(
                        complex_path,
                        f'Pocket_clean_{dis_threshold}A.pdb'
                        )
    # read sdf ligand file
    ligand_path = os.path.join(
                        complex_path,
                        f'{pdbid}_ligand.sdf'
                        )
    ligand = Chem.MolFromMolFile(ligand_path, sanitize=True, removeHs=True)
    if any([atom.GetSymbol() == 'Si' for atom in ligand.GetAtoms()]):
        print(f"Silicon found in {pdbid}, skipping...")
        return

    protein = os.path.join(
                        complex_path,
                        f'{pdbid}_protein.pdb'
                        )
    pocket = os.path.join(
                        complex_path,
                        f'Pocket_{dis_threshold}A.pdb')

    esm2_3b = os.path.join(
                        complex_path,
                        f'{pdbid}_esm2_3b.pt'
                        )
    esm2_3b = torch.load(esm2_3b, map_location='cpu')[0] #when using esm2_3b

    unimol_1b = os.path.join(
                        complex_path,
                        f'{pdbid}_unimol_1b.pt'
                        )
    unimol_1b = torch.load(unimol_1b, map_location='cpu')
    unimol_1b = torch.squeeze(unimol_1b) #

    res_list = get_clean_res_list(parser.get_structure(pdbid, pocket).get_residues(), verbose=False, ensure_ca_exist=True)
    # print(len(res_list))

    if not os.path.exists(res_list_pdb_dir):
        res2pdb(res_list, pdbid, res_list_pdb_dir)

    pocket = res_list_pdb_dir
    pocket_mol = Chem.MolFromPDBFile(pocket, sanitize=True, removeHs=True)
    # print(pocket_mol.GetNumAtoms())
    subgraphs_node, pos_sub_graph, node_batch_lig, node_class, edge_features_l, edge_index_l,subgraphs_node_pos = get_subgraph_mol(ligand)

    protein_node, edge_index_pro, edge_features_pro, coord_ca, coord_all = get_protein_graph(parser.get_structure(pdbid, protein), \
    parser.get_structure(pdbid, pocket))#, protein_saprot_feature)
    node_batch_pro = get_pro_node_batch(pocket_mol, coord_all)
    edge_index_inter, edge_attrs_inter = inter_graph_cg(pos_sub_graph, coord_ca, dis_threshold=dis_threshold)

    x_all_atom, edge_index_inter_all_atom, edge_attrs_inter_all_atom,pos_all_atom = get_inter_graph_all_atom(ligand, pocket_mol, dis_threshold=5.)


    data = HeteroData()
    # protein-protein node, protein-protein edge, protein-protein attr
    data["protein"].x = protein_node
    data["protein", "protein_edge", "protein"].edge_index = edge_index_pro
    data["protein", "protein_edge", "protein"].edge_attr = edge_features_pro
    data["protein"].pos = torch.FloatTensor(coord_ca)

    data["ligand"].x = node_class # [len(subgraphs_node)] ,will be processed by the nn.Embedding layer
    data["ligand", "ligand_edge", "ligand"].edge_index = edge_index_l
    data["ligand", "ligand_edge", "ligand"].edge_attr = edge_features_l
    data["ligand"].pos = torch.FloatTensor(pos_sub_graph)

    data["subnodes"] = subgraphs_node



    data["ligand", "inter_edge", "protein"].edge_index = edge_index_inter
    data["ligand", "inter_edge", "protein"].edge_attr = edge_attrs_inter

    data["complex"].x = x_all_atom  # [num_nodes, 35]
    data["complex", "inter_edge", "complex"].edge_index = edge_index_inter_all_atom # [2, num_edges]
    data["complex", "inter_edge", "complex"].edge_attr = edge_attrs_inter_all_atom   # [num_edges, 16rbf + 3type]
    data["complex"].pos = pos_all_atom # [num_nodes, 3]

    data["node_batch_lig"] = node_batch_lig
    data["node_batch_pro"] = node_batch_pro

    assert len(data["node_batch_lig"]) + len(data["node_batch_pro"]) == data["complex"].x.shape[0]
    data["esm_feature"] = esm2_3b
    data["unimol_feature"] = unimol_1b
    data["organism"] = organism_index
    data["ph_encoding"] = ph_encoding
    data["temp_encoding"] = temp_encoding
    return data

# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data,  collate_fn=data.collate_fn, **kwargs)#

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self,  data_df, organism_set, temp_set, dis_threshold=8):
        # self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.organism_set = organism_set
        self.temp_set = temp_set
        self._pre_process()

    def _pre_process(self):
        # data_dir = self.data_dir
        data_df = self.data_df
        dis_threshold = self.dis_threshold
        organism_set = np.load(self.organism_set, allow_pickle=True)
        temp_set = np.load(self.temp_set, allow_pickle=True)
        # dis_thresholds = repeat(self.dis_threshold, len(data_df))
        # dis_thresholds = list(dis_thresholds)
        graph_data_list = []
        # complex_id_list = []
        for i, row in data_df.iterrows():
            cid, organism, ph, temp = row['id'], str(row['Organism']), row['pH'], row['Temp']
            if type(cid) != str:
                cid = str(int(cid))
            has_complex = 'complex' in row.index and row['complex'] and not pd.isna(row['complex'])
            if has_complex:
                complex_dir = os.path.dirname(row['complex'])
            else:
                complex_dir = os.path.dirname(row['ligand'])
            data = mols2graphs(complex_dir, cid, organism_set, temp_set, \
                             organism, ph, temp, dis_threshold=dis_threshold)
            if data is None:
                print(f"Error: {cid} has no data, skipping...")
                continue
            graph_data_list.append(data)


        self.graph_data_list = graph_data_list

        # self.complex_ids = complex_id_list

    def __getitem__(self, idx):
        
        
        return self.graph_data_list[idx] #, self.complex_ids[idx], self.pKa_list[idx]
            
    

    def collate_fn(self, data_list):

        return data_list

    def __len__(self):
        return len(self.data_df)


if __name__ == '__main__':


    data_root = '/export/home/luod111/chai1'
    
    
    data_type = 'output_enz_miner_rzs'
    # data_df = pd.read_csv(os.path.join(data_root, "modeling-datasets", f"{data_type}_dataset_clean_no_structure.csv" ))
    # data_df = pd.read_csv(os.path.join(data_root, "modeling-datasets", f"{data_type}_dataset_clean_no_structure.csv" ))
    data_df = pd.read_csv(os.path.join(data_root, "modeling-datasets", f"enz_miner_rzs.csv" ))
    # # # three hours
    data_dir = os.path.join(data_root, "structure_enzyme")
    toy_set = GraphDataset(data_dir, data_df, data_type, graph_type='graphkcat', dis_threshold=8, num_process=30, create=True)
    print('finish!')


    # parser = PDBParser(QUIET=True)
    # protein = "/export/home/luod111/PLmodel/supervised/data/pdbbind/protein_remove_extra_chains_10A/2uy4_protein.pdb"
    # pocket_biopy = "/export/home/luod111/PLmodel/supervised/data/pdbbind/v2020-other-PL/2uy4/Pocket_8A.pdb"
    # protein_saprot_feature = "/export/home/luod111/PLmodel/supervised/data/pdbbind/embeddings/2uy4_embedding.pt"
    # get_pcoket_saprot_feature(parser.get_structure('2uy4', protein), parser.get_structure('2uy4', pocket_biopy), protein_saprot_feature)
    # pdbid = '4j48'
    # test_file = f"/export/home/luod111/PLmodel/supervised/data/pdbbind/v2020-other-PL/{pdbid}/{pdbid}_8A.rdkit"
    # with open(test_file, 'rb') as f:
    #     ligand, pocket = pickle.load(f)
    # pocket_biopy = parser.get_structure('1a30', pocket_biopy)

    # # # # interact_matrix = get_interaction_matrix(ligand, pocket.get_residues())
    # # # # print(interact_matrix)
    # pocket_res_list = get_clean_res_list(pocket_biopy.get_residues(), verbose=False, ensure_ca_exist=True)
    # _,pocket_coords, pocket_coords_main = get_pro_coord(pocket_res_list)
    # all_atom_pos_res_list = []
    # for i in pocket_coords:
    #     all_atom_pos_res_list.extend(i)
    # print(len(all_atom_pos_res_list))
    # coords = pocket.GetConformer().GetPositions()
    # print(len(coords))
    # print(coords)
    # get_subgraph_mol(ligand)
    # # get_protein_res_class(parser.get_structure('1a1c', protein), parser.get_structure('1a1c', pocket))
    # # get_protein_graph(parser.get_structure('1a1c', protein), parser.get_structure('1a1c', pocket))

    # mols2graphs(test_file, f'{pdbid}', 1, 'test_l.pyg', pocket_dis = 8)
    # vocab_txt = "/export/home/luod111/kcat/PS-VAE/data/zinc250k/zinc_350.txt"
    # get_sub_graph_dict(vocab_txt)

# %%
