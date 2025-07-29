import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
# from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn import global_mean_pool,global_add_pool
# from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch
from torch_geometric.data.batch import Batch
from egnn_clean import EGNN

def dense_to_sparse(dense_features, mask):
    """
    将 dense 格式的特征重新转回稀疏批次图。

    Args:
        dense_features (torch.Tensor): 形状为 [batch_size, max_nodes, hidden] 的特征张量。
        mask (torch.Tensor): 形状为 [batch_size, max_nodes] 的布尔掩码，指示哪些节点有效。

    Returns:
        torch.Tensor: 恢复的节点特征，形状为 [num_nodes, hidden]。
        torch.Tensor: 节点所属的 batch 索引，形状为 [num_nodes]。
    """
    # 提取有效节点
    valid_nodes = mask.view(-1)
    valid_features = dense_features.view(-1, dense_features.size(-1))[valid_nodes]
    
    # 构建每个节点的 batch 索引
    num_batches, max_nodes = mask.size()
    batch_indices = torch.arange(num_batches, device=mask.device).repeat_interleave(max_nodes)[valid_nodes]

    return valid_features, batch_indices


class AttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super(AttentionPooling, self).__init__()
        self.in_channels = in_channels

        # 定义注意力机制中的权重矩阵和全局上下文向量
        self.attn_fc = nn.Linear(in_channels, 1)  # 线性变换，用于计算注意力分数

    def forward(self, x):
        """
        x: [num_nodes, in_channels] - 节点特征
        """
        # 计算注意力分数 (未归一化)
        attn_scores = self.attn_fc(x)  # [num_nodes, 1]
        attn_scores = F.leaky_relu(attn_scores)  # 使用 LeakyReLU 激活

        # 对注意力分数进行归一化（softmax）
        attn_scores = F.softmax(attn_scores, dim=0)  # 归一化 [num_nodes, 1]

        # 加权聚合节点特征
        x_weighted = x * attn_scores  # [num_nodes, in_channels]
        x_pooled = torch.sum(x_weighted, dim=0)  # [in_channels]，对节点特征加权求和
        return x_pooled

class EGNN_complex(nn.Module):
    def __init__(self, hid_dim,  edge_dim, n_layers, attention=False, normalize=False, tanh=False):
        super(EGNN_complex, self).__init__()
        self.hid_dim = hid_dim
        self.edge_dim = edge_dim
        self.n_layers = n_layers
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh

        
        self.egnn=EGNN(hid_dim, hid_dim, hid_dim, in_edge_nf=edge_dim, n_layers=n_layers, residual=True, attention=attention, normalize=normalize, tanh=tanh)

        

       
    
    def forward(self, complex_x, complex_pos, complex_edge_index, complex_edge_attr):
        
        # complex_x_list = []
        # for i in range(len(data_complex)):
        #     complex_x =data_complex[i].x
        #     complex_edge_attr = data_complex[i].edge_attr
        #     complex_edge_index =data_complex[i].edge_index
        #     complex_pos = data_complex[i].pos
                       
        #     complex_x, complex_pos = self.egnn(complex_x, complex_pos,complex_edge_index, complex_edge_attr)
            

        #     complex_x_list.append(complex_x)
        # complex_x = torch.cat(complex_x_list, dim=0)  # [num_atoms, hid_dim]
        complex_x, complex_pos = self.egnn(complex_x, complex_pos,complex_edge_index, complex_edge_attr)
 
        return complex_x,complex_pos


class all_atom_view_graph_opt(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_layers):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lin_node_lig = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.egnn = EGNN_complex(hidden_dim, edge_dim=4, n_layers=n_layers, attention=False, normalize=False, tanh=False)
        # self.egnn_lig_pool = EGNN_complex(hidden_dim, edge_dim=0, n_layers=n_layers, attention=False, normalize=False, tanh=False)
        # self.egnn_pro_pool = EGNN_complex(hidden_dim, edge_dim=0, n_layers=n_layers, attention=False, normalize=False, tanh=False)
        self.gat_lig_pool = GATv2Conv(hidden_dim, hidden_dim)
        # self.gat_lig_pool = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.gat_pro_pool = GATv2Conv(hidden_dim, hidden_dim)
        # self.gat_pro_pool = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim) for _ in range(n_layers)])

    def forward(self, data):
        data_x_all_atom = data["complex"].x
        data_x_all_atom_edge_index = data["complex", "inter_edge", "complex"].edge_index
        data_x_all_atom_edge_attr = data["complex", "inter_edge", "complex"].edge_attr
        
        x = self.lin_node_lig(data_x_all_atom)
        x, pos = self.egnn(x, data["complex"].pos, data_x_all_atom_edge_index, data_x_all_atom_edge_attr)
        data["complex"].x = x
        data_list = data.to_data_list()
        batch_lig = [d["node_batch_lig"] for d in data_list]
        batch_pro = [d["node_batch_pro"] for d in data_list]

        # print(len(batch_lig[0]),len(batch_lig[1]))
        # print(len(batch_pro[0]),len(batch_pro[1]))
        
        batch_lig = [torch.tensor(b).to(x.device) for b in batch_lig]
        batch_pro = [torch.tensor(b).to(x.device) for b in batch_pro]
        # print(len(batch_lig[0]),len(batch_lig[1]))
        # print(len(batch_pro[0]),len(batch_pro[1]))
        

        # Precompute necessary information for batch processing
        lig_atom_counts, pro_atom_counts = [], []
        num_cg_lig_list, num_cg_pro_list = [], []
        batch_lig_global_list, batch_pro_global_list = [], []
        offsets_lig, offsets_pro = 0, 0

        for bl, bp in zip(batch_lig, batch_pro):
            lig_atom_count = len(bl)
            pro_atom_count = len(bp)
            lig_atom_counts.append(lig_atom_count)
            pro_atom_counts.append(pro_atom_count)

            num_cg_lig = bl.max().item() + 1 if lig_atom_count > 0 else 0
            num_cg_pro = bp.max().item() + 1 if pro_atom_count > 0 else 0
            num_cg_lig_list.append(num_cg_lig)
            num_cg_pro_list.append(num_cg_pro)

            batch_lig_global = bl + offsets_lig if lig_atom_count > 0 else bl
            batch_pro_global = bp + offsets_pro if pro_atom_count > 0 else bp
            batch_lig_global_list.append(batch_lig_global)
            batch_pro_global_list.append(batch_pro_global)

            offsets_lig += num_cg_lig
            offsets_pro += num_cg_pro

        # Combine all ligand and protein data
        batch_lig_all = torch.cat(batch_lig_global_list) if batch_lig_global_list else torch.tensor([], device=x.device)
        batch_pro_all = torch.cat(batch_pro_global_list) if batch_pro_global_list else torch.tensor([], device=x.device)

        x_lig_all, pos_lig_all, x_pro_all, pos_pro_all = [], [], [], []
        for data_sub, bl, bp in zip(data_list, batch_lig, batch_pro):
            split_sizes = [len(bl), len(bp)]
            x_split = torch.split(data_sub["complex"].x, split_sizes)
            pos_split = torch.split(data_sub["complex"].pos, split_sizes)
            x_lig_all.append(x_split[0])
            x_pro_all.append(x_split[1])
            pos_lig_all.append(pos_split[0])
            pos_pro_all.append(pos_split[1])

        x_lig_all = torch.cat(x_lig_all) if x_lig_all else torch.tensor([], device=x.device)
        pos_lig_all = torch.cat(pos_lig_all) if pos_lig_all else torch.tensor([], device=x.device)
        x_pro_all = torch.cat(x_pro_all) if x_pro_all else torch.tensor([], device=x.device)
        pos_pro_all = torch.cat(pos_pro_all) if pos_pro_all else torch.tensor([], device=x.device)

        # Compute coarse-grained positions using scatter_mean
        sum_L, sum_P = x_lig_all.size(0), x_pro_all.size(0)
        sum_C_lig, sum_C_pro = sum(num_cg_lig_list), sum(num_cg_pro_list)

        cg_pos_lig = scatter_mean(pos_lig_all, batch_lig_all, dim=0, dim_size=sum_C_lig) if sum_L > 0 else None
        cg_pos_pro = scatter_mean(pos_pro_all, batch_pro_all, dim=0, dim_size=sum_C_pro) if sum_P > 0 else None

        # Prepare combined features and positions
        x_lig_combined = torch.cat([x_lig_all, torch.zeros(sum_C_lig, self.hidden_dim, device=x.device)], dim=0) if sum_L + sum_C_lig > 0 else torch.tensor([], device=x.device)
        pos_lig_combined = torch.cat([pos_lig_all, cg_pos_lig], dim=0) if sum_L + sum_C_lig > 0 else torch.tensor([], device=x.device)
        x_pro_combined = torch.cat([x_pro_all, torch.zeros(sum_C_pro, self.hidden_dim, device=x.device)], dim=0) if sum_P + sum_C_pro > 0 else torch.tensor([], device=x.device)
        pos_pro_combined = torch.cat([pos_pro_all, cg_pos_pro], dim=0) if sum_P + sum_C_pro > 0 else torch.tensor([], device=x.device)

        # Build edge indices
        if sum_L > 0:
            src_lig = torch.arange(sum_L, device=x.device)
            dst_lig = src_lig + sum_L  # Adjust for combined tensor
            edge_index_lig = torch.stack([src_lig, dst_lig[batch_lig_all]], dim=0)
        else:
            edge_index_lig = torch.tensor([], device=x.device, dtype=torch.long)

        if sum_P > 0:
            src_pro = torch.arange(sum_P, device=x.device)
            dst_pro = src_pro + sum_P  # Adjust for combined tensor
            edge_index_pro = torch.stack([src_pro, dst_pro[batch_pro_all]], dim=0)
        else:
            edge_index_pro = torch.tensor([], device=x.device, dtype=torch.long)

        # Process ligand and protein through their respective EGNN layers
        if len(x_lig_combined) > 0:
            # x_lig_updated, _ = self.egnn_lig_pool(x_lig_combined, pos_lig_combined, edge_index_lig, None)
            x_lig_updated = self.gat_lig_pool(x_lig_combined, edge_index_lig)
            # for i in range(self.n_layers):
            #     x_lig_combined = self.gat_lig_pool[i](x_lig_combined, edge_index_lig)

            # x_lig_updated = x_lig_combined
            x_lig_cg = x_lig_updated[sum_L: sum_L + sum_C_lig]
        else:
            x_lig_cg = torch.tensor([], device=x.device)

        if len(x_pro_combined) > 0:
            # x_pro_updated, _ = self.egnn_pro_pool(x_pro_combined, pos_pro_combined, edge_index_pro, None)
            x_pro_updated = self.gat_pro_pool(x_pro_combined, edge_index_pro)
            # for i in range(self.n_layers):
            #     x_pro_combined = self.gat_pro_pool[i](x_pro_combined, edge_index_pro)

            # x_pro_updated = x_pro_combined
            x_pro_cg = x_pro_updated[sum_P: sum_P + sum_C_pro]
        else:
            x_pro_cg = torch.tensor([], device=x.device)

        # Split results into subgraphs
        x_list_lig_sub = list(torch.split(x_lig_cg, num_cg_lig_list, dim=0)) if sum_C_lig > 0 else []
        x_list_pro_sub = list(torch.split(x_pro_cg, num_cg_pro_list, dim=0)) if sum_C_pro > 0 else []
        pos_lig_list = list(torch.split(pos_lig_all, lig_atom_counts, dim=0)) if sum_L > 0 else []
        pos_pro_list = list(torch.split(pos_pro_all, pro_atom_counts, dim=0)) if sum_P > 0 else []
        return x_list_lig_sub, x_list_pro_sub, pos_lig_list, pos_pro_list


class all_atom_view_graph(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_layers):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lin_node_lig = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.egnn = EGNN_complex(hidden_dim, edge_dim=4, n_layers=n_layers, attention=False, normalize=False, tanh=False)
        self.egnn_lig_pool = EGNN_complex(hidden_dim, edge_dim=0, n_layers=n_layers, attention=False, normalize=False, tanh=False)
        self.egnn_pro_pool = EGNN_complex(hidden_dim, edge_dim=0, n_layers=n_layers, attention=False, normalize=False, tanh=False)

    def forward(self, data, batch_lig, batch_pro):
        data_x_all_atom = data["complex"].x
        data_x_all_atom_edge_index = data["complex", "inter_edge", "complex"].edge_index
        data_x_all_atom_edge_attr = data["complex", "inter_edge", "complex"].edge_attr
        
        # 初始节点特征线性映射
        x = self.lin_node_lig(data_x_all_atom)
        
        # 更新节点特征和位置
        x, pos = self.egnn(x, data["complex"].pos, data_x_all_atom_edge_index, data_x_all_atom_edge_attr)
        data["complex"].x = x
        data_list = data.to_data_list()
        # print(len(data_list))
        # x_list = [data["complex"].x for data in data_list]
        x_list = []
        pos_lig_list = []
        pos_pro_list = []
        pos_all_list = []
        for data in data_list:
            pos_lig_list.append(data["ligand"].pos)
            pos_pro_list.append(data["protein"].pos)
            x_list.append(data["complex"].x)
            pos_all_list.append(data["complex"].pos)

            

        x_list_lig_sub, x_list_pro_sub = [], []

        # 预先将 batch_lig 和 batch_pro 转换为 Tensor
        batch_lig = [torch.tensor(batch_lig_sub).to(x.device) for batch_lig_sub in batch_lig]
        batch_pro = [torch.tensor(batch_pro_sub).to(x.device) for batch_pro_sub in batch_pro]
        for x_sub, batch_lig_sub, batch_pro_sub,pos_lig_sub, pos_pro_sub, pos_all in zip \
            (x_list, batch_lig, batch_pro,pos_lig_list,pos_pro_list,pos_all_list):
            # 计算 split_sizes_sub，只需要一次计算
            split_sizes_sub = [len(batch_lig_sub), len(batch_pro_sub)]
            x_lig_sub, x_pro_sub = torch.split(x_sub, split_sizes_sub, dim=0)
            x_lig_pos, x_pro_pos = torch.split(pos_all, split_sizes_sub, dim=0)

            # 粗粒化节点的特征初始化
            len_sub_node_lig = len(torch.unique(batch_lig_sub))
            len_sub_node_pro = len(torch.unique(batch_pro_sub))

            sub_node_lig = torch.zeros(len_sub_node_lig, x_lig_sub.size(1), device=x_lig_sub.device)
            sub_node_pro = torch.zeros(len_sub_node_pro, x_pro_sub.size(1), device=x_pro_sub.device)

            # 拼接坐标
            # sub_pos_lig = torch.zeros(len_sub_node_lig, x_lig_pos.size(1), device=x_lig_pos.device)
            # sub_pos_pro = torch.zeros(len_sub_node_pro, x_pro_sub.size(1), device=x_pro_sub.device)

            x_lig_pos = torch.cat([x_lig_pos, pos_lig_sub], dim=0)
            x_pro_pos = torch.cat([x_pro_pos, pos_pro_sub], dim=0)

            # 拼接粗粒化节点到全原子节点的特征
            x_lig_sub = torch.cat([x_lig_sub, sub_node_lig], dim=0)
            x_pro_sub = torch.cat([x_pro_sub, sub_node_pro], dim=0)

            # 创建粗粒化节点的批次索引
            # 这里我们假设 batch_lig_sub 和 batch_pro_sub 每个元素对应着一个粗粒化节点
            # batch_lig_sub 和 batch_pro_sub 是每个原子节点所属的粗粒化节点的索引
            
            # 计算粗粒化边索引
            lig_to_cg_edges = torch.stack([batch_lig_sub, torch.arange(len(batch_lig_sub), device=x_lig_sub.device)], dim=0)
            pro_to_cg_edges = torch.stack([batch_pro_sub, torch.arange(len(batch_pro_sub), device=x_pro_sub.device)], dim=0)

            # 确保边索引是整数类型
            lig_to_cg_edges = lig_to_cg_edges.long()
            pro_to_cg_edges = pro_to_cg_edges.long()

            # 使用egnn层更新粗粒化节点
            # for i in range(self.n_layers):
            #     x_lig_sub = self.gconv_l[i](x_lig_sub, lig_to_cg_edges)
            #     x_pro_sub = self.gconv_pro[i](x_pro_sub, pro_to_cg_edges)
            x_lig_sub, _ = self.egnn_lig_pool(x_lig_sub, x_lig_pos, lig_to_cg_edges, None)
            x_pro_sub, _ = self.egnn_pro_pool(x_pro_sub, x_pro_pos, pro_to_cg_edges, None)

            # 从拼接后的特征中提取粗粒化节点
            x_lig_cg = x_lig_sub[-len_sub_node_lig:]  # 最后几个是粗粒化节点
            x_pro_cg = x_pro_sub[-len_sub_node_pro:]  # 最后几个是粗粒化节点

            # 将粗粒化节点的特征存入列表
            x_list_lig_sub.append(x_lig_cg)
            x_list_pro_sub.append(x_pro_cg)

            # x_list_lig_sub.append(x_lig_sub)
            # x_list_pro_sub.append(x_pro_sub)


        return x_list_lig_sub, x_list_pro_sub, pos_lig_list, pos_pro_list



class HeteroGNN(nn.Module):
    def __init__(self, hidden_dim, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layernorm = nn.LayerNorm(hidden_dim)

        self.pro_gconvs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=32) for _ in range(num_layers)
        ])
        self.lig_gconvs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=16) for _ in range(num_layers)
        ])

        self.lig_pro_gconvs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=16) for _ in range(num_layers)
        ])

        self.pro_lig_gconvs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=16) for _ in range(num_layers)
        ])

    def forward(self, data_list):
        # 批量处理同构图部分（ligand和protein各自的图卷积）
        batch = Batch.from_data_list(data_list)
        x_lig = batch['ligand'].x.float()
        x_pro = batch['protein'].x.float()
        
        # 预计算各样本的节点数用于后续分割
        num_ligs = [data['ligand'].num_nodes for data in data_list]
        num_pros = [data['protein'].num_nodes for data in data_list]
        # print(batch['ligand', 'ligand_edge', 'ligand'].edge_index.shape)
        edge_index_list = [data['protein'].num_nodes for data in data_list]
        for i in range(self.num_layers):
            # 批量处理ligand图卷积
            x_lig = self.lig_gconvs[i](
                x_lig,
                batch['ligand', 'ligand_edge', 'ligand'].edge_index,
                batch['ligand', 'ligand_edge', 'ligand'].edge_attr
            )
            
            # 批量处理protein图卷积
            x_pro = self.pro_gconvs[i](
                x_pro,
                batch['protein', 'protein_edge', 'protein'].edge_index,
                batch['protein', 'protein_edge', 'protein'].edge_attr
            )
            
            # 按样本分割特征以处理跨图卷积
            x_lig_split = x_lig.split(num_ligs)
            x_pro_split = x_pro.split(num_pros)
            
            updated_lig, updated_pro = [], []
            for j in range(len(data_list)):
                # 拼接单个样本的ligand和protein特征
                x_lig_pro = torch.cat([x_lig_split[j], x_pro_split[j]], dim=0)
                
                # 获取该样本的跨图边信息
                edge_index = data_list[j]['ligand', 'inter_edge', 'protein'].edge_index
                edge_attr = data_list[j]['ligand', 'inter_edge', 'protein'].edge_attr
                
                # 处理跨图卷积
                x_lig_pro = self.lig_pro_gconvs[i](x_lig_pro, edge_index, edge_attr)
                
                # 重新分割特征
                updated_lig.append(x_lig_pro[:num_ligs[j]])
                updated_pro.append(x_lig_pro[num_ligs[j]:])
            
            # 合并批量特征
            x_lig = torch.cat(updated_lig, dim=0)
            x_pro = torch.cat(updated_pro, dim=0)
        
        return x_pro, x_lig

class AttentionBlock(nn.Module):
    def __init__(self, hid_dim, num_heads, atten_active_fuc, dropout):
        super(AttentionBlock, self).__init__()
        assert hid_dim % num_heads == 0, "hid_dim must be divisible by num_heads"

        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.atten_active_fuc = atten_active_fuc
        self.head_dim = hid_dim // num_heads

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.sqrt_dk = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hid_dim, hid_dim)

    def forward(self, query, key, value, mask=None, return_attn=None):
        B = query.shape[0]

        # Transform Q, K, V
        q = self.f_q(query).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, N_q, head_dim]
        k = self.f_k(key).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N_k, head_dim]

        v = self.f_v(value).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, N_k, head_dim]


        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_dk  # [B, num_heads, N_q, N_k]
        if mask is not None:
            mask = mask.unsqueeze(1)
            # print(mask.shape)
            # print(scores.shape)
            scores = scores.masked_fill(mask == 0, float('-inf'))
       
        # Apply softmax and dropout
        if self.atten_active_fuc == "softmax":
            attention_weights = F.softmax(scores, dim=-1)
        elif self.atten_active_fuc == "sigmoid":
            attention_weights = F.sigmoid(scores)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to the values
        weighted = torch.matmul(attention_weights, v)  # [B, num_heads, N_q, head_dim]
        # print(weighted.shape)
        weighted = weighted.permute(0, 2, 1, 3).contiguous() # [B, N_q, num_heads, head_dim]

        # Concatenate heads and put through final linear layer
        weighted = weighted.view(B, -1, self.hid_dim)   # [B, N_q, hid_dim]
        output = self.fc_out(weighted)
                # Determine if returning attention weights
        if return_attn is None:
            return_attn = not self.training  # 默认训练不返回，推理返回

        if return_attn:
            return output, attention_weights
        else:
            return output


class CrossAttentionBlock(nn.Module):
    def __init__(self, hid_dim, dropout, atten_active_fuc, num_heads=4):
        super(CrossAttentionBlock, self).__init__()
        self.att = AttentionBlock(hid_dim=hid_dim, num_heads=num_heads, atten_active_fuc=atten_active_fuc, dropout=dropout)
        
        self.linear_res = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )

        self.linear_lig = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.norm_aa = nn.LayerNorm(hid_dim)
        self.norm_lig = nn.LayerNorm(hid_dim)

    def forward(self, ligand_features, aa_features, mask_l, mask_aa):
        # 生成掩码以匹配attention分数的维度
        mask_l_expanded = mask_l.unsqueeze(1)  # [B, 1, N_l]
        mask_aa_expanded = mask_aa.unsqueeze(1)  # [B, 1, N_aa]

        # 交叉注意力计算
        if self.training:
            aa_att = self.att(aa_features, ligand_features, ligand_features, mask=mask_l_expanded)
            lig_att = self.att(ligand_features, aa_features, aa_features, mask=mask_aa_expanded)
        else:
            aa_att, atten_score_aa_lig = self.att(aa_features, ligand_features, ligand_features, mask=mask_l_expanded)
            lig_att, atten_score_lig_aa = self.att(ligand_features, aa_features, aa_features, mask=mask_aa_expanded)
        # 线性变换与残差连接
        aa_features = self.linear_res(aa_att) + aa_features
        # aa_features = aa_att + aa_features
        aa_features = self.norm_aa(aa_features)
        
        
        ligand_features = self.linear_lig(lig_att)+ ligand_features #
        # ligand_features = lig_att + ligand_features

        ligand_features = self.norm_lig(ligand_features)

        if self.training:
            return ligand_features, aa_features
        else:
            return ligand_features, aa_features, atten_score_aa_lig, atten_score_lig_aa

       
class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                # self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                # self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h
        
class bottle_view_graph(nn.Module):
    def __init__(self, 
        node_dim, 
        hidden_dim,
        HeteroGNN_layers=True,
        # share_score=True,
        pooling = "sum",
        vocab_size=350,
        num_layers=3,
        dropout=0.1,
        # cross_attention=True,
        # num_heads=4,
        ligand_nn_embedding=True,
        num_fc_layers=2,
        fc_hidden_dim=2,
        share_fc=False
        ):
        super().__init__()
        # self.lin_node_lig = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        # self.top_view = top_view_graph(node_dim, hidden_dim)
        self.dropout = dropout
        # self.share_score = share_score
        self.pooling = pooling
        # self.cross_attention = cross_attention
        self.ligand_nn_embedding = ligand_nn_embedding
        self.HeteroGNN_layers = HeteroGNN_layers
        self.share_fc = share_fc
        # self.log_var1 = nn.Parameter(torch.zeros(1))  # 初始化为0
        # self.log_var2 = nn.Parameter(torch.zeros(1))

        self.all_atom_view = all_atom_view_graph_opt(node_dim, hidden_dim, num_layers)
        # self.att_pool = AttentionPooling(hidden_dim)
        
        if self.share_fc:
            self.fc = FC(hidden_dim*6, hidden_dim*fc_hidden_dim, num_fc_layers, dropout, 1)
        else:
            self.fc_kcat = FC(hidden_dim*5+23+103, hidden_dim*fc_hidden_dim, num_fc_layers, dropout, 1)#
            self.fc_km = FC(hidden_dim*5+23+103, hidden_dim*fc_hidden_dim, num_fc_layers, dropout, 1)#
            # self.fc_kcat = FC(hidden_dim*2, hidden_dim*fc_hidden_dim, num_fc_layers, dropout, 1)
            # self.fc_km = FC(hidden_dim*2, hidden_dim*fc_hidden_dim, num_fc_layers, dropout, 1)

        # self.fc_str= nn.Sequential(Linear(hidden_dim*2, hidden_dim*2), nn.LeakyReLU())
        # self.fc_seq = nn.Sequential(Linear(hidden_dim*16, hidden_dim*8), nn.LeakyReLU())

        self.fc_esm = nn.Sequential(Linear(hidden_dim*10, hidden_dim), nn.LeakyReLU()) # esm2: *10 SaProt: *5 prot5: *4
        self.fc_unimol = nn.Sequential(Linear(hidden_dim*6, hidden_dim), nn.LeakyReLU())

        if self.HeteroGNN_layers:
            self.HeteroGNN = HeteroGNN(hidden_dim, num_layers)

        if self.ligand_nn_embedding:
            self.lin_node_lig = nn.Sequential(Linear(hidden_dim*2, hidden_dim), nn.LeakyReLU())
            self.layer_norm_lig = nn.LayerNorm(hidden_dim*2)


        self.lin_node_pro = nn.Sequential(Linear(hidden_dim+26, hidden_dim), nn.LeakyReLU())#26+
        self.layer_norm_pro = nn.LayerNorm(hidden_dim+26)#+

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.embedding_organism = nn.Embedding(1428, hidden_dim)
        # if self.cross_attention:
        self.cross_atten_lig_pro = CrossAttentionBlock(hidden_dim, dropout, atten_active_fuc="softmax")
        self.cross_atten_llmlig_llmpro = CrossAttentionBlock(hidden_dim, dropout, atten_active_fuc="softmax")
        self.cross_atten_lig_llmlig = CrossAttentionBlock(hidden_dim, dropout, atten_active_fuc="softmax")
        self.cross_atten_pro_llmpro = CrossAttentionBlock(hidden_dim, dropout, atten_active_fuc= "softmax")
        self.cross_atten_pro_mmlig = CrossAttentionBlock(hidden_dim, dropout, atten_active_fuc="softmax")
        self.cross_atten_lig_mmpro = CrossAttentionBlock(hidden_dim, dropout, atten_active_fuc="softmax")

    def forward(self, data):
        batch_size = len(data)
        data_batch = Batch.from_data_list(data)

        # 特征转换
        # print(data_batch["esm_feature"].shape, data_batch["unimol_feature"].shape)
        esm_feature = self.fc_esm(data_batch["esm_feature"])
        unimol_feature = self.fc_unimol(data_batch["unimol_feature"])

        # 优化批次索引
        device = esm_feature.device
        m_list = [d["esm_feature"].shape[0] for d in data]  # 建议预计算
        n_list = [d["unimol_feature"].shape[0] for d in data]
        batch_index_esm = torch.repeat_interleave(torch.arange(batch_size, device=device), torch.tensor(m_list, device=device))
        batch_index_unimol = torch.repeat_interleave(torch.arange(batch_size, device=device), torch.tensor(n_list, device=device))

        # 图特征提取
        x_list_lig_sub, x_list_pro_sub, _, _ = self.all_atom_view(data_batch)

        # 批量处理嵌入
        if self.ligand_nn_embedding:
            sub_node_type_embs = torch.cat([d["ligand"].x for d in data], dim=0)
            sub_node_type_embs = self.embedding(sub_node_type_embs)
            x_lig_subs = torch.cat(x_list_lig_sub, dim=0)
            all_emb_ligs = self.lin_node_lig(self.layer_norm_lig(torch.cat([sub_node_type_embs, x_lig_subs], dim=1).float()))
        else:
            all_emb_ligs = torch.cat(x_list_lig_sub, dim=0)

        pro_xs = torch.cat([d["protein"].x for d in data], dim=0)
        x_pro_subs = torch.cat(x_list_pro_sub, dim=0)
        all_emb_pros = self.lin_node_pro(self.layer_norm_pro(torch.cat([pro_xs, x_pro_subs], dim=1).float()))

        # 分配嵌入
        start_lig, start_pro = 0, 0
        for i in range(batch_size):
            num_lig = data[i]["ligand"].x.shape[0]
            num_pro = data[i]["protein"].x.shape[0]
            data[i]["ligand"].x = all_emb_ligs[start_lig:start_lig + num_lig]
            data[i]["protein"].x = all_emb_pros[start_pro:start_pro + num_pro]
            start_lig += num_lig
            start_pro += num_pro

        # 异构图处理
        if self.HeteroGNN_layers:
            x_pro, x_lig = self.HeteroGNN(data)
        else:
            data = Batch.from_data_list(data)
            x_lig = data.x_dict['ligand']
            x_pro = data.x_dict['protein']

        # 转换为密集张量
        esm_feature, mask_esm = to_dense_batch(esm_feature, batch_index_esm)
        unimol_feature, mask_unimol = to_dense_batch(unimol_feature, batch_index_unimol)
        x_pro, mask_pro = to_dense_batch(x_pro, data_batch.batch_dict['protein'])
        x_lig, mask_lig = to_dense_batch(x_lig, data_batch.batch_dict['ligand'])

        # 交叉注意力
        # if self.cross_attention:
        if self.training:
                
            x_lig, x_pro = self.cross_atten_lig_pro(x_lig, x_pro, mask_lig, mask_pro)
            x_lig, x_llmlig = self.cross_atten_lig_llmlig(x_lig, unimol_feature, mask_lig, mask_unimol)
            x_pro, x_llmpro = self.cross_atten_pro_llmpro(x_pro, esm_feature, mask_pro, mask_esm)
            x_pro, x_llmlig = self.cross_atten_pro_mmlig(x_pro, x_llmlig, mask_pro, mask_unimol)
            x_lig, x_llmpro = self.cross_atten_lig_mmpro(x_lig, x_llmpro, mask_lig, mask_esm)
            x_llmlig, x_llmpro = self.cross_atten_llmlig_llmpro(x_llmlig, x_llmpro, mask_unimol, mask_esm)

        else:
            x_lig, x_pro, att_lig_pro, att_pro_lig = self.cross_atten_lig_pro(x_lig, x_pro, mask_lig, mask_pro)
            x_lig, x_llmlig, att_lig_llmlig, att_llmlig_lig= self.cross_atten_lig_llmlig(x_lig, unimol_feature, mask_lig, mask_unimol)
            x_pro, x_llmpro, att_pro_llmpro, att_llmpro_pro = self.cross_atten_pro_llmpro(x_pro, esm_feature, mask_pro, mask_esm)
            x_pro, x_llmlig, att_pro_llmlig, att_llmlig_pro = self.cross_atten_pro_mmlig(x_pro, x_llmlig, mask_pro, mask_unimol)
            x_lig, x_llmpro, att_lig_llmpro, att_llmpro_lig= self.cross_atten_lig_mmpro(x_lig, x_llmpro, mask_lig, mask_esm)
            x_llmlig, x_llmpro, att_llmlig_llmpro, att_llmpro_llmlig = self.cross_atten_llmlig_llmpro(x_llmlig, x_llmpro, mask_unimol, mask_esm)

        x_lig_batch, _ = dense_to_sparse(x_lig, mask_lig)
        x_pro_batch, _ = dense_to_sparse(x_pro, mask_pro)
        x_llmlig_batch, _ = dense_to_sparse(x_llmlig, mask_unimol)
        x_llmpro_batch, _ = dense_to_sparse(x_llmpro, mask_esm)
        # x_llmlig_batch,_ = dense_to_sparse(unimol_feature, mask_unimol)
        # x_llmpro_batch,_ = dense_to_sparse(esm_feature, mask_esm)
       


        if self.pooling == "sum":
            x_lig = global_add_pool(x_lig_batch, data_batch.batch_dict['ligand'])
            x_pro = global_add_pool(x_pro_batch, data_batch.batch_dict['protein'])
            x_llmlig = global_add_pool(x_llmlig_batch, batch_index_unimol)
            x_llmpro = global_add_pool(x_llmpro_batch, batch_index_esm)
        else:
            x_lig = global_mean_pool(x_lig_batch, data_batch.batch_dict['ligand'])
            x_pro = global_mean_pool(x_pro_batch, data_batch.batch_dict['protein'])
            x_llmlig = global_mean_pool(x_llmlig_batch, batch_index_unimol)
            x_llmpro = global_mean_pool(x_llmpro_batch, batch_index_esm)

        # # 拼接并输出
        org_emb = self.embedding_organism(data_batch["organism"])
        # # # # RESHAPE TO [B,D]
        ph_emb = data_batch["ph_encoding"].reshape(-1, 23)  # 23dim
        temp_emb = data_batch["temp_encoding"].reshape(-1, 103) # 103dim
        # x = torch.cat([x_llmlig, x_llmpro, x_lig, x_pro], dim=1)
        x = torch.cat([x_llmlig, x_llmpro, org_emb, ph_emb, temp_emb, x_lig, x_pro], dim=1)

        if self.share_fc:
            return self.fc(x), self.fc(x)
        else:
            if self.training:
                return self.fc_kcat(x), self.fc_km(x)         
            else:
                return self.fc_kcat(x), self.fc_km(x), att_lig_pro, att_pro_lig, att_lig_llmlig, att_llmlig_lig, \
                att_pro_llmpro, att_llmpro_pro, att_pro_llmlig, att_llmlig_pro, att_llmlig_llmpro, att_llmpro_llmlig
            # return self.fc_kcat(x), self.fc_km(x)            




if __name__ == "__main__":
    # test
    import os
    import pandas as pd
    from dataset_graphkcat_chai1 import GraphDataset, PLIDataLoader
    data_root = '/export/home/luod111/chai1'
    data_dir = os.path.join(data_root, 'structure_enzyme')
    batch_size = 8
    data_df = pd.read_csv(os.path.join(data_root, "modeling-datasets", f"train_dataset_clean_no_structure.csv" ))[:10]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    toy_set = GraphDataset(data_dir, data_df, data_type="train", graph_type="graphkcat",dis_threshold=8, create=False)
    toy_loader = PLIDataLoader(toy_set, batch_size=batch_size, shuffle=True)
    # model_pdbbind = bottle_view_graph_pdbbind(node_dim=35, hidden_dim=256,vocab_size=350, num_layers=3,dropout=0.1)
    # load_model_dict(model_pdbbind, "/export/home/luod111/software/kcat/kcat_model/model/20250323_231535_graphkcat_repeat0/model/epoch-699, train_loss-0.1504, train_rmse-0.3878, valid_rmse-1.3859, valid_pr-0.7777.pt")
    model = bottle_view_graph(node_dim=35, hidden_dim=256,vocab_size=350, num_layers=3,dropout=0.1,)
    model = model.to(device)
    for data in toy_loader:
        # print
        # subnodes = [data[i]["subnodes"] for i in range(4)]
        data = [data[i].to(device) for i in range(len(data)) if data[i] != "label"]
        # print(data[0].keys)
        out = model(data)
        # print(out.size())
        # break
            



  