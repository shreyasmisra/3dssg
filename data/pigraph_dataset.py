import os
import sys
import time
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
from tqdm import tqdm
import glob
import itertools
import pclpy
from pclpy import pcl
from collections import Counter

sys.path.append(os.path.join(os.getcwd())) # HACK add the lib folder
from lib.config import CONF

# yield from the relationships_train.json
MAX_OBJECTS_NUM = 62
MAX_REL_NUM = 62

class D3SemanticSceneGraphDataset(Dataset):

    def __init__(self, relationships, all_scan_id,
                 split="train",
                 augment=False):
        ''' target: obtain all data path and split into train/val/test set '''
        self.relationships = relationships # all relationships and classes
        # all scan id, include split id in scan id
        self.all_scan_id = all_scan_id  # useless
        self.split = split
        self.augment = augment
        self.scene_data = []
        self.num_action_classes = 14
        self._load_data()
        # self.index_map = {}
        # index = 0
        # for s_idx in range(len(self.scene_data)):
        #     scene_index = s_idx
        #     import pdb; pdb.set_trace()
        #     for t_idx in (self.scene_data[s_idx]["predicate_cat"].keys()):
        #         timeseries_index = int(t_idx)
        #         self.index_map[index] = (scene_index, timeseries_index)
        #         index += 1
        print('length of index map', len(self.scene_data))

    def __len__(self):
        return len(self.scene_data)

    def __getitem__(self, idx):
        """ build_in function: make this class can be indexed like list in python """
        start = time.time()
        # scene_index, timeseries_index = self.index_map[idx.item()]
        data_dict = self.scene_data[idx].copy()

        pc = np.array(data_dict['points']).astype(np.float32)
        labels = np.array(data_dict['labels']).astype(np.int64)
        data_dict['objects_pc'], data_dict['objects_cat'] = self.get_clusters(pc, labels)
        # data_dict['objects_pc'] = np.array(data_dict['points']).astype(np.float32)
        # data_dict['objects_cat'] = np.array(data_dict['labels']).astype(np.int64)
        merged = [[el] if type(el) != list else el for el in data_dict['triples']]
        merged = list(itertools.chain(*merged))
        data_dict['triples'] = np.asarray(merged).reshape(-1, 4)[:, :3].astype(np.int64)
        
        # Extract action labels from triples and make a separate vector.
        action_labels = self._gather_action_labels(data_dict["triples"])
        data_dict['action_labels'] = action_labels

        # data_dict['triples'] = np.array(data_dict['triples'][:3]).astype(np.int64)
        data_dict['poses'] = np.array(data_dict['poses']).astype(np.float32)
        data_dict["load_time"] = time.time() - start

        return data_dict

    def _load_data(self):
        print("\nloading {} data into memory...".format(self.split))
        relations = tqdm(self.relationships)
        for relationship in relations:
            for key in relationship['relationships'].keys():
                try:
                    path = CONF.PATH.DATA + "processed/data_dict/" + (relationship["scan"]).split("/")[-1].split(".")[0] + "/data_dict_{}.json".format(key)
                    data_dict = json.load(open(path))
                    if len(data_dict["points"]) > 1024:
                        self.scene_data.append(data_dict)
                    del data_dict
                except:
                    pass


    def _pad_dict(self, data, key):
        # align all values in the DATA dict except for the point cloud key/value pair
        batch_size = len(data)
        assert batch_size > 0
        new_value_list = []
        for i in range(batch_size):
            if key not in ['triples', 'labels', 'points']:
                new_value_list.append(data[i][key])
            elif key in ['points']:
                if len(data[i][key])!=4096:
                    new_value_list.append((data[i][key] + [[0, 0, 0, 0, 0, 0]] * (4096 - len(data[i][key]))))
                else:
                    new_value_list.append(data[i][key])
            # elif key in ['objects_pc']:
            #     if len(data[i][key])!=2048:
            #         tmp = np.zeros((2048, 6))
            #         tmp[:len(data[i][key]), :] = data[i][key]
            #         new_value_list.append(tmp)
            #     else:
            #         new_value_list.append(data[i][key])
            # elif key in ['objects_cat']:
            #     if len(data[i][key])!=2048:
            #         tmp = np.zeros((2048,))
            #         tmp[:len(data[i][key])] = data[i][key]
            #         new_value_list.append(tmp)
            #     else:
            #         new_value_list.append(data[i][key])
            elif key == 'labels':
                if len(data[i][key])!=4096:
                    new_value_list.append((data[i][key] + [0] * (4096 - len(data[i][key]))))
                else:
                    new_value_list.append(data[i][key])
            else:
                zeros = np.zeros((5, 3))
                zeros[:len(data[i][key]), :] = data[i][key]
                new_value_list.append(zeros)
        # import pdb; pdb.set_trace()
        try:
            batch_values = np.stack(new_value_list)
        except:
            import pdb; pdb.set_trace()
        if key not in ['scan_id']:
            batch_values = torch.from_numpy(batch_values).type(torch.FloatTensor)
        if key in ['objects_pc']:
            batch_values = batch_values.view(batch_values.shape[0]*batch_values.shape[1], 256, -1)
        return batch_values

    def get_clusters(self, point_set, point_labels):
        cloud = pcl.PointCloud.PointXYZRGBA.from_array(point_set[:, :3], point_set[:, 3:])
        normals = cloud.compute_normals(radius=0.1)

        disable_transform = True
        voxel_resolution = 0.45
        seed_resolution = 0.1
        color_importance = 0.2
        spatial_importance = 0.4
        normal_importance = 1.0

        sv = pcl.segmentation.SupervoxelClustering.PointXYZRGBA(voxel_resolution, seed_resolution)
        if disable_transform:
            sv.setUseSingleCameraTransform(False)

        sv.setInputCloud(cloud)
        sv.setNormalCloud(normals)
        sv.setColorImportance(color_importance)
        sv.setSpatialImportance(spatial_importance)
        sv.setNormalImportance(normal_importance)
        supervoxel_clusters = pcl.vectors.map_uint32t_PointXYZRGBA()
        # print('Extracting supervoxels!')
        sv.extract(supervoxel_clusters)
        labeled_cloud = sv.getLabeledCloud()
        labels = np.asarray(labeled_cloud.label)
        # print(labels.max())
        point_list = []
        label_list = []
        rank = []
        for c in range(1, labels.max()+1):
            indices = np.where(labels==c)
            points = point_set[indices]
            plabels = point_labels[indices]
            if len(points)>256:
                index = np.random.choice(points.shape[0], 256, replace=False)  
                points = points[index]
                plabels = plabels[index]
                c = Counter(plabels)
                value, _ = c.most_common()[0]
                point_list.append(points)
                label_list.append(value)
                rank.append(len(points))
            elif len(points>10) and len(points)<=256:
                tmp = np.zeros((256, 6))
                tmp[:len(points)] = points
                c = Counter(plabels)
                value, _ = c.most_common()[0]
                point_list.append(tmp)
                label_list.append(value)
                rank.append(len(points))
            else:
                continue
        sort_index = np.argsort(np.asarray(rank))
        point_list = [point_list[i] for i in sort_index]
        label_list = [label_list[i] for i in sort_index]

        if len(sort_index)<60:
            for diff in range(60-len(sort_index)):
                point_list.append(np.zeros((256, 6)))
                label_list.append(1)
        else:
            point_list = point_list[:60]
            label_list = label_list[:60]
        return point_list, label_list

    def _pad_object_pc(self, data, key):
        batch_size = len(data)
        assert batch_size > 0
        prefix = key.split('_')[0]
        num_key = prefix + '_num'

        # gather all objects' point number
        l_list = []
        for one_line in data:
            l_list.extend(one_line[num_key])
        max_l = np.array(l_list).max()
        num_lines = MAX_OBJECTS_NUM if 'object' in key else MAX_REL_NUM  # align the 'batch_size' dim of input to the pointnet

        # align objects' point cloud
        new_pc_list = []
        for i in range(batch_size):
            start = 0
            dim2 = data[i][key].shape[1]
            for j in range(len(data[i][num_key])):
                line = np.expand_dims(np.repeat(0, dim2), 0)
                num = data[i][num_key][j]
                lines = np.repeat(line, max_l-num, axis=0)
                new_pc_list.append(np.concatenate((data[i][key][start: start+num], lines), axis=0))
                start = start + num
            for j in range(len(data[i][num_key]), num_lines):
                line = np.expand_dims(np.repeat(0, dim2), 0)
                lines = np.repeat(line, max_l, axis=0)
                new_pc_list.append(lines)

        batch_pc = np.stack(new_pc_list)
        batch_pc = torch.from_numpy(batch_pc).type(torch.FloatTensor)
        return batch_pc
    
    def _gather_action_labels(self, triplet):
        """
        Gathers action (relationship) labels from triplet.

        Triplet: <np.ndarray>. A nx3 matrix with labels in the last column.
        """
        action_labels = np.zeros((self.num_action_classes), dtype=np.float32)
        action_labels[np.unique(triplet[:,-1])] = 1.
        return action_labels

    def collate_fn(self, data):
        data_dict = {}
        keys = data[0].keys()
        pc_keys = ["objects_pc"]
        ignore_keys = ["scan_id", "objects_num", "predicate_num"]
        scan_id = []
        predicate_pc = []
        # predicate_num = []
        for key in keys:
            data_dict[key] = self._pad_dict(data, key)
        # data_dict["scan_id"] = scan_id
        return data_dict

if __name__ == "__main__":
    scans = json.load(open(os.path.join(CONF.PATH.DATA, "/home/shreyasm/pigraph/data/processed/HOI_all.json")))["scans"]
    import pdb;pdb.set_trace();
    max_objects_num = 0
    max_rel_num = 0
    min_objects_num = 10
    min_rel_num = 200
    for scan in scans:
        if len(scan["relationships"]) == 0:
            print(scan)
        max_objects_num = max(max_objects_num, len(scan["objects"]))
        max_rel_num = max(max_rel_num, len(scan["relationships"]))
        min_objects_num = min(min_objects_num, len(scan["objects"]))
        min_rel_num = min(min_rel_num, len(scan["relationships"]))
    print(max_objects_num, max_rel_num)
    print(min_objects_num, min_rel_num)
