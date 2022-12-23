import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.pointnet import PointNetEncoder
from models.dgcnn import DGCNN, DGCNN_rel
from models.gcn import GraphTripleConv, GraphTripleConvNet
from models.gat import GAT
from models.layers import build_mlp
from lib.config import CONF

PRETRAINED_DGCNN = "/home/vrlab725_wff/Documents/segmentation/dgcnn/pytorch/pretrained/model.1024.t7"
DEVICE = "cuda"

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def load_pretained_cls_model(model):
    # load pretrained pointnet_cls model [ relPointNet ver. ]
    pretrained_dict = torch.load(CONF.PATH.PRETRAINED)["model_state_dict"]
    net_state_dict = model.state_dict()
    pretrained_dict_ = {k[5:]: v for k, v in pretrained_dict.items() if 'feat' in k and v.size() == net_state_dict[k[5:]].size()}
    net_state_dict.update(pretrained_dict_)
    model.load_state_dict(net_state_dict)


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature_edge(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)

    idx_base = torch.arange(0, batch_size, device=DEVICE).view(-1, 1, 1) * num_points
    indices = idx + idx_base

    indices = indices.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[indices, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = (feature - x).permute(0, 3, 1, 2).contiguous()
    base = torch.repeat_interleave(torch.arange(0, 61, device=DEVICE), repeats=k)
    base = torch.repeat_interleave(base.unsqueeze(dim=1), repeats=batch_size, dim=1).T
    idx = torch.stack((base, idx.permute(0, 2, 1).reshape(batch_size, -1)), dim=2)
    return feature, idx  # (batch_size, num_dims, num_points, k)


class SGPN(nn.Module):
    def __init__(self, use_pretrained_cls, feature_extractor='pointnet', graph_gen='gcn', gnn_dim=256, gnn_hidden_dim=512,
                 gnn_pooling='avg', gnn_num_layers=5, mlp_normalization='none',
                 obj_cat_num=160, pred_cat_num=27):
        super().__init__()

        # ObjPointNet and RelPointNet
        assert feature_extractor == 'pointnet' or feature_extractor == 'dgcnn'
        self.feature_extractor = feature_extractor
        if feature_extractor == 'pointnet':
            self.objExtractor = PointNetEncoder(global_feat=True, feature_transform=True, channel=6)  # (x,y,z)
            self.relExtractor = PointNetEncoder(global_feat=True, feature_transform=True,
                                                channel=3)  # (x,y,z,M) M-> class-agnostic instance segmentation
            if use_pretrained_cls:
                print("Loading pretrained classifier ...")
                load_pretained_cls_model(self.objExtractor)
                load_pretained_cls_model(self.relExtractor)
                print("Loaded pretrained classifier.")
            # print('objPointNet params:', get_num_params(self.objPointNet))
            # print('relPointNet params:', get_num_params(self.relPointNet))
        else:
            self.objExtractor = DGCNN()
            self.objExtractor.load_state_dict(torch.load(PRETRAINED_DGCNN), strict=False)
            for p in self.objExtractor.parameters():
                p.requires_grad = False
            self.relExtractor = DGCNN_rel(input_channel=4)

        assert graph_gen == 'gcn' or graph_gen == 'gat'
        self.graph_gen = graph_gen
        if graph_gen == 'gcn':
            # GCN module
            if gnn_num_layers == 0:
                self.gconv = nn.Linear(1024, gnn_dim)   # final feature of the pointNet2
            elif gnn_num_layers > 0:
                gconv_kwargs = {
                    'input_dim': 1024, 'output_dim': gnn_dim, 'hidden_dim': gnn_hidden_dim, 'pooling': gnn_pooling,
                    'mlp_normalization': mlp_normalization,
                }
                self.gconv = GraphTripleConv(**gconv_kwargs)

            self.gconv_net = None
            if gnn_num_layers > 1:
                gconv_kwargs = {
                    'input_dim': gnn_dim, 'hidden_dim': gnn_hidden_dim, 'pooling': gnn_pooling, 'num_layers': gnn_num_layers - 1,
                    'mlp_normalization': mlp_normalization,
                }
                self.gconv_net = GraphTripleConvNet(**gconv_kwargs)
            # print('gconv params:', get_num_params(self.gconv_net) + get_num_params(self.gconv))
        else:
            # GAT Module
            self.gat_trans_obj = nn.Linear(1024, gnn_dim)
            self.gat_trans_pred = nn.Linear(1024, gnn_dim)
            self.gat = None
            if gnn_num_layers > 0:
                self.gat = GAT(gnn_dim)
            # print('gat params:', get_num_params(self.gat))

        # MLP for classification
        obj_classifier_layer = [gnn_dim, 256, obj_cat_num]
        self.obj_classifier = build_mlp(obj_classifier_layer, batch_norm=mlp_normalization)

        rel_classifier_layer = [gnn_dim, 256, pred_cat_num]
        self.rel_classifier = build_mlp(rel_classifier_layer, batch_norm=mlp_normalization)

        self.pad_human_centroid = torch.nn.ConstantPad1d((0, 3), 0)

    def forward(self, data_dict):
        # objects_id = data_dict["objects_id"]
        objects_pc = data_dict["objects_pc"]
        batch_size = objects_pc.size(0) // 60
        
        # objects_count = []
        # predicate_count = []
        # for i in range(batch_size):
        #     objects_count.append(data_dict["objects_cat"][i].unique().shape[0])
        #     predicate_count.append(data_dict["triples"][i][:,-1].unique().shape[0])
        
        predicate_pc_flag = data_dict["poses"] # 1, 3, 25, 25
        trans_feat = []
        
        # point cloud pass feature extractor
        objects_pc = objects_pc.permute(0, 2, 1)
        predicate_pc_flag = predicate_pc_flag.flatten(2)
        # predicate_pc_flag = predicate_pc_flag.permute(0, 2, 1)
        if self.feature_extractor == 'pointnet':
            obj_vecs, _, tf1 = self.objExtractor(objects_pc) # 120, 1024 | 120, 64, 64
            pred_vecs, _, tf2 = self.relExtractor(predicate_pc_flag) # 2, 1024 (only works with batch size > 1)
            obj_vecs = torch.repeat_interleave(obj_vecs[:60, :].unsqueeze(0), batch_size, dim=0).transpose(2, 1) # 2, 1024, 60
            combined_feats = torch.concat((obj_vecs, pred_vecs.unsqueeze(2)), dim=2)
            edge_feats, edge_indices = get_graph_feature_edge(combined_feats, k=60) # k = all clusters i.e. edges between everything
            edge_feats = edge_feats.flatten(2)
            trans_feat.append(tf1)
            trans_feat.append(tf2)
        else:
            obj_vecs = self.objExtractor(objects_pc)
            pred_vecs = self.relExtractor(predicate_pc_flag)

        data_dict["trans_feat"] = trans_feat
        # obj_vecs and rel_vecs pass GNN module
        obj_vecs_list = []
        pred_vecs_list = []
        for i in range(batch_size):
            if self.graph_gen == 'gcn':
                # GCN Module
                o_vecs, p_vecs = self.gconv(combined_feats[i].T, edge_feats[i].T, edge_indices[i])
                if self.gconv_net is not None:
                    o_vecs, p_vecs = self.gconv_net(o_vecs, p_vecs, edge_indices[i])
            obj_vecs_list.append(o_vecs)
            pred_vecs_list.append(p_vecs)
        # for i in range(batch_size):
        #     if self.graph_gen == 'gcn':
        #         # GCN Module
        #         if isinstance(self.gconv, nn.Linear):
        #             o_vecs = self.gconv(obj_vecs[object_num*i: object_num*(i+1)])
        #         else:
        #             o_vecs, p_vecs = self.gconv(obj_vecs[object_num*i: object_num*(i+1)], pred_vecs[predicate_num*i: predicate_num*(i+1)], edge_idx[i])
        #         if self.gconv_net is not None:
        #             o_vecs, p_vecs = self.gconv_net(o_vecs, p_vecs, edge_idx[i])
        #     else:
        #         # GAT Module
        #         o_vecs = self.gat_trans_obj(obj_vecs[object_num*i: object_num*(i+1)])
        #         p_vecs = self.gat_trans_pred(pred_vecs[predicate_num*i: predicate_num*(i+1)])
        #         if self.gat is not None:
        #             o_vecs, p_vecs = self.gat(o_vecs, p_vecs, edge_idx[i])

        #     obj_vecs_list.append(o_vecs)
        #     pred_vecs_list.append(p_vecs)

        obj_pred_list = []
        rel_pred_list = []
        for o_vec in obj_vecs_list:
            obj_pred = self.obj_classifier(o_vec)
            obj_pred_list.append(obj_pred)
        for p_vec in pred_vecs_list:
            rel_pred = self.rel_classifier(p_vec)
            rel_pred_list.append(rel_pred)
        # ATTENTION: as batch_size > 1, the value that corresponds to the "predict" key is a list
        data_dict["objects_predict"] = obj_pred_list
        data_dict["predicate_predict"] = rel_pred_list

        return data_dict

# predicate_count = 0
# for b in range(data_dict["triples"].shape[0]):
#     predicate_count += data_dict["triples"][b][:, 2].unique()
