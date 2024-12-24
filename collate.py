from torch_geometric.data import Batch
from utils import *
def collate(data_list):

    for data in data_list:

        data[0].attn_bias = pad_attn_bias_unsqueeze(data[0].attn_bias, 48)
        data[0].spatial_pos = pad_spatial_pos_unsqueeze(data[0].spatial_pos, 48)
        data[0].node = pad_2d_unsqueeze(data[0].node, 48)
        data[0].in_degree = pad_1d_unsqueeze(data[0].in_degree,48)
        data[0].out_degree = pad_1d_unsqueeze(data[0].out_degree, 48)
        data[0].edge_input = pad_4d_unsqueeze(data[0].edge_input, 48, 48, 10)
    
    # Tạo batch cho Protein và SMILES
    protein_batch = Batch.from_data_list([data[1] for data in data_list])
    smiles_batch = Batch.from_data_list([data[0] for data in data_list])

    return smiles_batch, protein_batch
