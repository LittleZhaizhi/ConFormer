import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath
from timm.models.vision_transformer import Mlp

from model.tky_dual_axis import DualAxisAttention


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        c_dim,
        feed_forward_dim=2048,
        num_heads=8,
        dropout=0,
    ):
        super().__init__()

        self.attn = DualAxisAttention(model_dim, c_dim, num_heads)
        self.ln1 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.ln2 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.feed_forward = Mlp(
            in_features=model_dim,
            hidden_features=feed_forward_dim,
            act_layer=nn.ReLU,
            drop=dropout,
        )
        self.dropout = DropPath(dropout)
        self.GLN = nn.Sequential(nn.ReLU(), nn.Linear(c_dim, 6 * model_dim, bias=True))

    def forward(self, x, c):
        c = c.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.GLN(
            c
        ).chunk(6, dim=-1)
        normalized = modulate(self.ln1(x), shift_msa, scale_msa)
        attention = self.attn(normalized, c)
        x = x + self.dropout(gate_msa * attention)
        x = x + self.dropout(
            gate_mlp * self.feed_forward(modulate(self.ln2(x), shift_mlp, scale_mlp))
        )
        return x


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("nvc,vw->nwc", (x, A.to(x)))
        return x.contiguous()


class GCN(nn.Module):
    def __init__(self, c_in, node_num, supports=[], adp=1, order=2, dropout=0.0):
        super(GCN, self).__init__()
        self.nconv = nconv()
        self.support_len = len(supports)
        self.supports = supports
        self.adp = adp
        self.dropout = dropout
        if self.support_len > 0:
            c_out = c_in
            c_in = (order * (self.support_len + adp) + 1) * c_in
            self.mlp = nn.Sequential(nn.Linear(c_in, c_out, bias=True), nn.ReLU())
            self.order = order
        if self.adp:
            self.nodevec1 = nn.Parameter(torch.randn(node_num, 10), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(10, node_num), requires_grad=True)

    def forward(self, x):
        supports = self.supports
        if self.support_len:
            if self.adp:
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                supports = self.supports + [adp]
            out = [x]
            for a in supports:
                x1 = self.nconv(x, a)
                out.append(x1)
                for k in range(2, self.order + 1):
                    x2 = self.nconv(x1, a)
                    out.append(x2)
                    x1 = x2
            h = self.mlp(torch.cat(out, dim=-1))
            h = F.dropout(h, self.dropout, training=self.training)
        else:
            h = x
        return h


class ConFormer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=0,
        dow_embedding_dim=0,
        node_embedding_dim=0,
        acc_embedding_dim=0,
        reg_embedding_dim=0,
        feed_forward_dim=256,
        num_heads=4,
        workday=False,
        num_layers=3,
        dropout=0.1,
        adp_dropout=0,
        dow_dropout=0,
        tod_dropout=0,
        adp=1,
        supports=1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.acc_embedding_dim = acc_embedding_dim
        self.reg_embedding_dim = reg_embedding_dim
        self.workday = workday
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + node_embedding_dim
            + acc_embedding_dim
            + reg_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.adp_dropout = adp_dropout
        self.dow_dropout = dow_dropout
        self.tod_dropout = tod_dropout
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        self.graph_propagate = GCN(
            self.model_dim - input_embedding_dim,
            self.num_nodes,
            supports,
            adp=adp,
            dropout=dropout,
        )

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if node_embedding_dim > 0:
            self.node_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
            )
        if acc_embedding_dim > 0:
            self.acc_embedding = nn.Embedding(2, acc_embedding_dim)
        if reg_embedding_dim > 0:
            self.reg_embedding = nn.Embedding(2, reg_embedding_dim)

        self.output_proj = nn.Linear(self.model_dim * in_steps, out_steps * output_dim)

        self.attn_layers = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    self.model_dim - input_embedding_dim,
                    feed_forward_dim,
                    num_heads,
                    dropout,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim)
        batch_size = x.shape[0]
        features = []
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (x[:, -1, :, 1] * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            tod_emb = F.dropout(tod_emb, self.tod_dropout, training=self.training)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            if self.workday:
                dow_emb = self.dow_embedding((x[:, -1, :, 2] >= 5).long())
            else:
                dow_emb = self.dow_embedding(x[:, -1, :, 2].long())
            dow_emb = F.dropout(dow_emb, self.dow_dropout, training=self.training)
            features.append(dow_emb)
        if self.node_embedding_dim > 0:
            node_emb = self.node_embedding.expand(
                size=(batch_size, *self.node_embedding.shape)
            )
            node_emb = F.dropout(node_emb, self.adp_dropout, training=self.training)
            features.append(node_emb)
        if self.acc_embedding_dim > 0:
            acc_emb = self.acc_embedding((x[..., 3].sum(1) > 0).long())
            features.append(acc_emb)
        if self.reg_embedding_dim > 0:
            reg_emb = self.reg_embedding((x[..., 4].sum(1) > 0).long())
            features.append(reg_emb)

        traffic = x[..., : self.input_dim]
        node_features = torch.cat(features, dim=-1)
        c = self.graph_propagate(node_features)
        history_features = node_features.unsqueeze(1).expand(-1, self.in_steps, -1, -1)
        x = torch.cat((self.input_proj(traffic), history_features), dim=-1)
        for attn in self.attn_layers:
            x = attn(x, c)
        x = x.transpose(1, 2).flatten(2)
        out = self.output_proj(x).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        return out
