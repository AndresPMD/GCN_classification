import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import *


def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim) # Adjust last
    assert sum(sizes_list) == dim
    if sizes_list[-1]<0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j-1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list

def get_chunks(x,sizes):
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1,begin,s)
        out.append(y)
        begin += s
    return out


class Block(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1600,
            chunks=20,
            rank=15,
            shared=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.,
            pos_norm='before_cat'):
        super(Block, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert(pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        merge_linears0, merge_linears1 = [], []
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = nn.Linear(size, size*rank)
            merge_linears0.append(ml0)
            if self.shared:
                ml1 = ml0
            else:
                ml1 = nn.Linear(size, size*rank)
            merge_linears1.append(ml1)
        self.merge_linears0 = nn.ModuleList(merge_linears0)
        self.merge_linears1 = nn.ModuleList(merge_linears1)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bsize = x1.size(0)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, m0, m1 in zip(range(len(self.sizes_list)),
                                    self.merge_linears0,
                                    self.merge_linears1):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            m = m0(x0_c) * m1(x1_c) # bsize x split_size*rank
            m = m.view(bsize, self.rank, -1)
            z = torch.sum(m, 1)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z,p=2)
            zs.append(z)
        z = torch.cat(zs,1)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class BlockTucker(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1600,
            chunks=20,
            shared=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.,
            pos_norm='before_cat'):
        super(BlockTucker, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert(pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if self.shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)

        self.sizes_list = get_sizes_list(mm_dim, chunks)
        bilinears = []
        for size in self.sizes_list:
            bilinears.append(
                nn.Bilinear(size, size, size)
            )
        self.bilinears = nn.ModuleList(bilinears)
        self.linear_out = nn.Linear(self.mm_dim, self.output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bsize = x1.size(0)
        if self.dropout_input:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, bilinear in enumerate(self.bilinears):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            z = bilinear(x0_c, x1_c)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z,p=2)
            zs.append(z)
        z = torch.cat(zs, 1)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class Mutan(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1600,
            rank=15,
            shared=False,
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        super(Mutan, self).__init__()
        self.input_dims = input_dims
        self.shared = shared
        self.mm_dim = mm_dim
        self.rank = rank
        self.output_dim = output_dim
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.normalize = normalize
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.merge_linear0 = nn.Linear(mm_dim, mm_dim*rank)
        if self.shared:
            self.linear1 = self.linear0
            self.merge_linear1 = self.merge_linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
            self.merge_linear1 = nn.Linear(mm_dim, mm_dim*rank)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        m0 = self.merge_linear0(x0)
        m1 = self.merge_linear1(x1)
        m = m0 * m1
        m = m.view(-1, self.rank, self.mm_dim)
        z = torch.sum(m, 1)
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class Tucker(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1600,
            shared=False,
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        super(Tucker, self).__init__()
        self.input_dims = input_dims
        self.shared = shared
        self.mm_dim = mm_dim
        self.output_dim = output_dim
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.bilinear = nn.Bilinear(mm_dim, mm_dim, mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z = self.bilinear(x0, x1)

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class MLB(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1200,
            activ_input='relu',
            activ_output='relu',
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        super(MLB, self).__init__()
        self.input_dims = input_dims
        self.mm_dim = mm_dim
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z = x0 * x1

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class MFB(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1200,
            factor=2,
            activ_input='relu',
            activ_output='relu',
            normalize=False,
            dropout_input=0.,
            dropout_pre_norm=0.,
            dropout_output=0.):
        super(MFB, self).__init__()
        self.input_dims = input_dims
        self.mm_dim = mm_dim
        self.factor = factor
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_norm = dropout_pre_norm
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim*factor)
        self.linear1 = nn.Linear(input_dims[1], mm_dim*factor)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z = x0 * x1

        if self.dropout_pre_norm > 0:
            z = F.dropout(z, p=self.dropout_pre_norm, training=self.training)

        z = z.view(z.size(0), self.mm_dim, self.factor)
        z = z.sum(2)

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)

        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class MFH(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1200,
            factor=2,
            activ_input='relu',
            activ_output='relu',
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        super(MFH, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.factor = factor
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0_0 = nn.Linear(input_dims[0], mm_dim*factor)
        self.linear1_0 = nn.Linear(input_dims[1], mm_dim*factor)
        self.linear0_1 = nn.Linear(input_dims[0], mm_dim*factor)
        self.linear1_1 = nn.Linear(input_dims[1], mm_dim*factor)
        self.linear_out = nn.Linear(mm_dim*2, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0_0(x[0])
        x1 = self.linear1_0(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z_0_skip = x0 * x1

        if self.dropout_pre_lin:
            z_0_skip = F.dropout(z_0_skip, p=self.dropout_pre_lin, training=self.training)

        z_0 = z_0_skip.view(z_0_skip.size(0), self.mm_dim, self.factor)
        z_0 = z_0.sum(2)

        if self.normalize:
            z_0 = torch.sqrt(F.relu(z_0)) - torch.sqrt(F.relu(-z_0))
            z_0 = F.normalize(z_0, p=2)

        #
        x0 = self.linear0_1(x[0])
        x1 = self.linear1_1(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z_1 = x0 * x1 * z_0_skip

        if self.dropout_pre_lin > 0:
            z_1 = F.dropout(z_1, p=self.dropout_pre_lin, training=self.training)

        z_1 = z_1.view(z_1.size(0), self.mm_dim, self.factor)
        z_1 = z_1.sum(2)

        if self.normalize:
            z_1 = torch.sqrt(F.relu(z_1)) - torch.sqrt(F.relu(-z_1))
            z_1 = F.normalize(z_1, p=2)

        #
        cat_dim = z_0.dim() - 1
        z = torch.cat([z_0, z_1], cat_dim)
        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z

#
# class MCB(nn.Module):
#
#     def __init__(self,
#             input_dims,
#             output_dim,
#             mm_dim=16000,
#             activ_output='relu',
#             dropout_output=0.):
#         super(MCB, self).__init__()
#         # compatible with pytorch 0.3 and 0.4, not 1.0
#         from . import compactbilinearpooling as cbp
#         self.input_dims = input_dims
#         self.output_dim = output_dim
#         self.mm_dim = mm_dim
#         self.activ_output = activ_output
#         self.dropout_output = dropout_output
#         # Modules
#         self.mcb = cbp.CompactBilinearPooling(input_dims[0], input_dims[1], mm_dim)
#         self.linear_out = nn.Linear(mm_dim, output_dim)
#         self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#
#     def forward(self, x):
#         z = self.mcb(x[0], x[1])
#         z = self.linear_out(z)
#         if self.activ_output:
#             z = getattr(F, self.activ_output)(z)
#         if self.dropout_output > 0:
#             z = F.dropout(z, p=self.dropout_output, training=self.training)
#         return z


class LinearSum(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1200,
            activ_input='relu',
            activ_output='relu',
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        super(LinearSum, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z = x0 + x1

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)

        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class ConcatMLP(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            dimensions=[500,500],
            activation='relu',
            dropout=0.):
        super(ConcatMLP, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.input_dim = sum(input_dims)
        self.dimensions = dimensions + [output_dim]
        self.activation = activation
        self.dropout = dropout
        # Modules

        self.mlp = MLP(
            self.input_dim,
            self.dimensions,
            self.activation,
            self.dropout)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if x[0].dim() == 3 and x[1].dim() == 2:
            x[1] = x[1].unsqueeze(1).reshape_as(x[0])
        if x[1].dim() == 3 and x[0].dim() == 2:
            x[0] = x[0].unsqueeze(1).reshape_as(x[1])
        z = torch.cat(x, dim=x[0].dim()-1)
        z = self.mlp(z)
        return z

class RelationalNetwork(nn.Module):

    def __init__(self, input_dims):
        super(RelationalNetwork, self).__init__()
        self.input_dims = input_dims

        # Modules
        self.bn_g = nn.Linear(input_dims, input_dims)
        self.g = nn.Linear(input_dims, input_dims)
        self.bn_h = nn.Linear(input_dims, input_dims)
        self.h = nn.Linear(input_dims, input_dims)


    def forward(self, vf, tf):

        bs = vf.size[0]
        vf_g = F.relu(self.g(self.bn_g(vf)))
        tf_g = F.relu(self.g(self.bn_g(tf)))

        x = vf_g + tf_g
        x = torch.sigmoid(self.h(self.bn_h(x)))
        x = vf * x

        return x

