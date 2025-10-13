from numpy import dtype
import numpy as np
import random
from sympy import false
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch import optim
import torch.nn.functional as F

torch.set_printoptions(precision=3, sci_mode=False)

from PIL import Image
import torchvision.transforms.functional as TF

from copy import deepcopy
import itertools
from matplotlib import pyplot as plt
from GNN.discretization import *
from GNN.tools import (
    calculate_area_under_curve,
    under_prediction_score,
    over_prediction_score,
    iou_score,
    evaluate_metrics,
    calculate_ic95,
)
from GNN.config import graph_id_index, departement_index
from sklearn.metrics import f1_score, jaccard_score

import dgl

from GNN.graph_builder import *
from GNN.tools import check_and_create_path, save_object, read_object

from tqdm import tqdm

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

g = torch.Generator()
g.manual_seed(42)

def plot_score_per_epochs(score_per_epoch, dir_output, name):
    plt.figure(figsize=(15,5))
    scores = score_per_epoch['score']
    epochs = score_per_epoch['epoch']
    plt.plot(epochs, scores)
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.savefig(dir_output / f'{name}.png')
    plt.close('all')

class InplaceGraphDataset(Dataset):
    def __init__(self, X : list, Y : list, edges : list, leni : int, device : torch.device) -> None:
        self.X = X
        self.Y = Y
        self.device = device
        self.edges = edges
        self.leni = leni

    def __getitem__(self, index) -> tuple:
        x = self.X[index]
        y = self.Y[index]

        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = []
        
        return torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device),  \

    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

def construct_dataset(date_ids, x_data, y_data, graph, ids_columns, ks, horizon, use_temporal_as_edges, isNotmesh=False):
    Xs, Ys, Es = [], [], []
    
    # Traiter par date
    print(ks, horizon)
    for id in date_ids:
        if use_temporal_as_edges is None:
            x, y = construct_time_series(id, x_data, y_data, ks, horizon, len(ids_columns))
            if x is not None and isNotmesh:
                for i in range(x.shape[0]):
                    Xs.append(x[i])
                    Ys.append(y[i])
            elif x is not None:
                Xs.append(x)
                Ys.append(y)
            continue
        elif use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, x_data, y_data, ks, horizon, len(ids_columns))
        else:
            x, y, e = construct_graph_with_time_series(graph, id, x_data, y_data, ks, horizon, len(ids_columns))

        if x is None:
            continue

        if x.shape[0] == 0:
            continue
        
        Xs.append(x)
        Ys.append(y)
        Es.append(e)
    
    return Xs, Ys, Es

def create_dataset(graph,
                    df_train,
                    df_val,
                    df_test,
                    features_name,
                    target_name,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int,
                    horizon: int,
                    graph_mesh=None,
                    gridh2mesh=None,
                    mesh2graph=None
                    ):
    
    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns + [target_name]].values
    
    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns + [target_name]].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns + [target_name]].values

    dateTrain = np.sort(np.unique(y_train[y_train[:, weight_index] > 0, date_index]))
    dateVal = np.sort(np.unique(y_val[y_val[:, weight_index] > 0, date_index]))
    dateTest = np.sort(np.unique(y_test[y_test[:, weight_index] > 0, date_index]))

    logger.info(f'{dateTrain.shape}, {dateVal.shape}, {dateTest.shape}')

    logger.info(f'Constructing train Dataset')
    Xst, Yst, Est = construct_dataset(dateTrain, x_train, y_train, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None)

    logger.info(f'Constructing val Dataset')
    XsV, YsV, EsV = construct_dataset(dateVal, x_val, y_val, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None)

    logger.info(f'Constructing test Dataset')
    XsTe, YsTe, EsTe = construct_dataset(dateTest, x_test, y_test, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None)

    # Assurez-vous que les ensembles ne sont pas vides
    assert len(Xst) > 0, "Le jeu de données d'entraînement est vide"
    assert len(XsV) > 0, "Le jeu de données de validation est vide"
    assert len(XsTe) > 0, "Le jeu de données de test est vide"

    if graph_mesh is None:
        # Création des datasets finaux
        print('uzbdkazdkjzan')
        train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
        val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
        test_dataset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    elif graph_mesh is not None:
        train_dataset = InplaceMeshGraphDatasetInplace(Xst, Yst, Est, len(Xst), device, graph_mesh, gridh2mesh, mesh2graph)
        val_dataset = InplaceMeshGraphDatasetInplace(XsV, YsV, EsV, len(XsV), device, graph_mesh, gridh2mesh, mesh2graph)
        test_dataset = InplaceMeshGraphDatasetInplace(XsTe, YsTe, EsTe, len(XsTe), device, graph_mesh, gridh2mesh, mesh2graph)

    return train_dataset, val_dataset, test_dataset

def create_train_dataset(graph,
                    df_train,
                    features_name,
                    target_name,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int,
                    horizon:int,
                    graph_mesh=None,
                    gridh2mesh=None,
                    mesh2graph=None):

    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns + [target_name]].values
    
    print('weight', df_train['weight'].unique())

    dateTrain = np.sort(np.unique(y_train[y_train[:, weight_index] > 0, date_index]))

    logger.info(f'{dateTrain.shape}')

    logger.info(f'Constructing train Dataset')
    Xst, Yst, Est = construct_dataset(dateTrain, x_train, y_train, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None)

    # Assurez-vous que les ensembles ne sont pas vides
    assert len(Xst) > 0, "Le jeu de données d'entraînement est vide"

    if graph_mesh is None:
        # Création des datasets finaux
        print('uzbdkazdkjzan')
        train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
    elif graph_mesh is not None:
        train_dataset = InplaceMeshGraphDatasetInplace(Xst, Yst, Est, len(Xst), device, graph_mesh, gridh2mesh, mesh2graph)
    #elif mesh == 'mygraph':
    #    train_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, Xst, Yst, Est, len(Xst), device)

    return train_dataset

def create_test_val_dataset(graph,
                    df_val,
                    df_test,
                    features_name,
                    target_name,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int,
                    horizon: int,
                    graph_mesh=None,
                    gridh2mesh=None,
                    mesh2graph=None):
        
    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns + [target_name]].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns + [target_name]].values

    dateVal = np.sort(np.unique(y_val[y_val[:, weight_index] > 0, date_index]))
    dateTest = np.sort(np.unique(y_test[y_test[:, weight_index] > 0, date_index]))

    logger.info(f'{dateVal.shape}, {dateTest.shape}')

    logger.info(f'Constructing val Dataset')
    XsV, YsV, EsV = construct_dataset(dateVal, x_val, y_val, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None)

    logger.info(f'Constructing test Dataset')
    XsTe, YsTe, EsTe = construct_dataset(dateTest, x_test, y_test, graph, ids_columns, horizon, ks, use_temporal_as_edges, graph_mesh is None)

    # Assurez-vous que les ensembles ne sont pas vides
    assert len(XsV) > 0, "Le jeu de données de validation est vide"
    assert len(XsTe) > 0, "Le jeu de données de test est vide"

    if gridh2mesh is None:
        # Création des datasets finaux
        print('uzbdkazdkjzan')
        val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
        test_dataset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    elif gridh2mesh is not None:
        val_dataset = InplaceMeshGraphDatasetInplace(XsV, YsV, EsV, len(XsV), device, graph_mesh, gridh2mesh, mesh2graph)
        test_dataset = InplaceMeshGraphDatasetInplace(XsTe, YsTe, EsTe, len(XsTe), device, graph_mesh, gridh2mesh, mesh2graph)
    #elif mesh == 'mygraph':
    #    val_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, XsV, YsV, EsV, len(XsV), device)
    #    test_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, XsTe, YsTe, EsTe, len(XsTe), device)

    return val_dataset, test_dataset

def get_numpy_data(graph, df,
                       features_name,
                       use_temporal_as_edges : bool,
                       ks :int):

    Xset = df[ids_columns + features_name].values

    X = []
    E = []
    Yset = None
    graphId = np.unique(Xset[:, date_index])
    for date in graphId:
        if use_temporal_as_edges is None:
            x, _ = construct_time_series(date, Xset, Yset, ks, len(ids_columns))
            if x is not None:
                for i in range(x.shape[0]):
                    X.append(x[i])
            continue
        elif use_temporal_as_edges:
            x, _, e = construct_graph_set(graph, date, Xset, Yset, ks, len(ids_columns))
        else:
            x, _, e = construct_graph_with_time_series(graph, date, Xset, Yset, ks, len(ids_columns))

        if x is None:
            continue

        if x.shape[0] == 0:
            continue

        X.append(x)
        if 'e' in locals():
            E.append(e)

    return np.asarray(X), np.asarray(E)

def create_test_loader(graph, df,
                       features_name,
                       device : torch.device,
                       use_temporal_as_edges : bool,
                       target_name,
                       ks :int,
                       horizon:int,
                       graph_mesh=None,
                        gridh2mesh=None,
                        mesh2graph=None):
    
    Xset, Yset = df[ids_columns + features_name].values, df[ids_columns + targets_columns + [target_name]].values

    X = []
    Y = []
    E = []

    graphId = np.unique(Xset[:, date_index])
    for date in graphId:
        if use_temporal_as_edges is None:
            x, y = construct_time_series(date, Xset, Yset, ks, horizon, len(ids_columns))
            if x is not None:
                for i in range(x.shape[0]):
                    X.append(x[i])
                    Y.append(y[i])
            continue
        elif use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, date, Xset, Yset, ks, horizon, len(ids_columns))
        else:
            x, y, e = construct_graph_with_time_series(graph, date, Xset, Yset, ks, horizon,len(ids_columns))

        if x is None:
            continue

        if x.shape[0] == 0:
            continue

        X.append(x)
        Y.append(y)
        E.append(e)

    if gridh2mesh is None:
        dataset = InplaceGraphDataset(X, Y, E, len(X), device)
        collate = graph_collate_fn
    elif gridh2mesh is not None:
        dataset = InplaceMeshGraphDatasetInplace(X, Y, E, len(X), device, graph_mesh, gridh2mesh, mesh2graph)
        collate = graph_collate_fn_mesh
    #elif mesh == 'mygraph':
    #    dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, X, Y, E, len(X), device)
    #    collate = graph_collate_fn_multiple_graph
    else:
        raise ValueError(f'{mesh} is not a known mesh value')

    if use_temporal_as_edges is None:
        loader = DataLoader(dataset, dataset.__len__(), False, worker_init_fn=seed_worker,
            generator=g)
    else:
        loader = DataLoader(dataset, dataset.__len__(), False, collate_fn=collate,
                            worker_init_fn=seed_worker,
        generator=g)

    return loader

class WrapperModel(torch.nn.Module):
    def __init__(self, original_model, F, T, edges, horizon=0):
        super().__init__()
        self.model = original_model
        self.F = F
        self.T = T
        self.edges = edges

        self.horizon = horizon

    def forward(self, x_flat):
        # reshape x_flat (B, F*T) vers (B, F, T)
        x_orig = x_flat.reshape(-1, self.F, self.T)
        return self.model(x_orig, self.edges)

class Training():
    def __init__(self, model_name, nbfeatures, batch_size, lr, target_name, task_type,
                 features_name, ks, out_channels, dir_log,
                 loss='mse', name='Training', device='cpu', under_sampling='full', over_sampling='full', n_run=1,
                 horizon=0):
        
        self.model_name = model_name
        self.name = name
        self.loss = loss
        self.device = device
        self.model = None
        self.optimizer = None
        self.batch_size = batch_size
        self.target_name = target_name
        self.features_name = [str(fet) for fet in features_name]
        self.ks = int(ks)
        self.lr = lr
        self.out_channels = out_channels
        self.dir_log = dir_log
        self.task_type = task_type
        self.model_params = None
        self.under_sampling = under_sampling
        self.over_sampling = over_sampling
        self.find_log = False
        self.nbfeatures = nbfeatures
        self.student_train = False
        self.use_temporal_as_edges = None
        self.n_run = n_run
        self.metrics = {}
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.constrastive = False
        self.use_prototypes = False
        self.prototype_weight = 1.0
        self.prototypes = None
        self.ALATraining = False
        self.area_parameters = None
        # Distillation tracking (best/worst losses per epoch)
        self.distill_best_log = []   # list of dicts: {epoch, graph_id, loss}
        self.distill_worst_log = []  # list of dicts: {epoch, graph_id, loss}
        self.criterion_params = []
        self._current_epoch = None
        self.seed = None
        self.horizon = horizon
        self.seed = None

        if 'Past_risk' in self.features_name:
            self.id_past_risk = features_name.index('Past_risk')
        else:
            self.id_past_risk = None

        if 'Past_burnedarea' in self.features_name:
            self.id_past_ba = features_name.index('Past_burnedarea')
        else:
            self.id_past_ba = None

        self.prev_idx = []

        if self.task_type == "classification":
            # Pour classification : les colonnes one-hot sont du type f"{colunm}_prev_<classe>"
            new_features = [f"{self.target_name}_prev_{i}" for i in range(self.out_channels)]
            self.prev_idx = [self.features_name.index(f) for f in new_features if f in self.features_name]

        elif self.task_type == "binary":
            # Pour binaire : on a colunm_prev_bin, colunm_prev_bin_0 et colunm_prev_bin_1
            new_features = [f"{self.target_name}_prev_bin"] + [f"{self.target_name}_prev_bin_{i}" for i in range(self.out_channels)]
            self.prev_idx = [self.features_name.index(f) for f in new_features if f in self.features_name]

        elif self.task_type == "regression" and self.target_name in ["nbsinister", "burnedarea"]:
            # Pour régression : une seule feature ajoutée
            new_features = [f"{self.target_name}_prev"]
            self.prev_idx = [self.features_name.index(f) for f in new_features if f in self.features_name]
        
        if len(self.prev_idx) == 0:
            self.prev_idx = None

    def compute_weights_and_target(self, labels, band, ids_columns, is_grap_or_node, graphs, H):
        weight_idx = ids_columns.index('weight')
        target_is_binary = self.task_type == 'binary'

        if len(labels.shape) == 3:
            weights = labels[:, weight_idx, H]
            target = (labels[:, band, H] > 0).long() if target_is_binary else labels[:, band, H]

        elif len(labels.shape) == 5:
            weights = labels[:, :, :, weight_idx, H]
            target = (labels[:, :, :, band, H] > 0).long() if target_is_binary else labels[:, :, :, band, H]

        elif len(labels.shape) == 4:
            weights = labels[:, :, :, weight_idx,]
            target = (labels[:, :, :, band] > 0).long() if target_is_binary else labels[:, :, :, band]

        else:
            weights = labels[:, weight_idx]
            target = (labels[:, band] > 0).long() if target_is_binary else labels[:, band]
        
        if is_grap_or_node:
            unique_elements = torch.unique(graphs, return_inverse=False, return_counts=False, sorted=True)
            first_indices = torch.tensor([torch.nonzero(graphs == u, as_tuple=True)[0][0] for u in unique_elements])
            weights = weights[first_indices]
            target = target[first_indices]

        return target, weights
    
    def compute_inputs(self, inputs, H, time_steps):
        if H + 1 == 0:
            if len(inputs.shape) == 3:
                inputs_horizon = inputs[:, :, -(self.ks + 1):]

            elif len(inputs.shape) == 5:
                inputs_horizon = inputs[:, :, :, :, -(self.ks + 1):]

            elif len(inputs.shape) == 4:
                inputs_horizon = inputs
            else:
                inputs_horizon = inputs
        
        else:
            if len(inputs.shape) == 3:
                inputs_horizon = inputs[:, :, H - self.ks:H + 1]

            elif len(inputs.shape) == 5:
                inputs_horizon = inputs[:, :, :, :, H - self.ks:H + 1]

            elif len(inputs.shape) == 4:
                inputs_horizon = inputs
            else:
                inputs_horizon = inputs

        if inputs_horizon.ndim % 2 == 0:
                inputs_horizon = inputs_horizon[:, :, None]
            
        return inputs_horizon
    
    def compute_single_loss(self, out, tar, wei, cluster_ids=None, tolong=False, criterion=None):
        if self.task_type == 'regression':
            tar = tar.view(out.shape[0])
            wei = wei.view(out.shape[0])

            tar = torch.masked_select(tar, wei.gt(0))
            out = out[wei.gt(0)]
            wei = torch.masked_select(wei, wei.gt(0))
        else:
            wei = wei.long()
            if not self.student_train: # works on probability
                tar = tar.long()

            tar = tar[wei.gt(0)]
            out = out[wei.gt(0)]

            if cluster_ids is not None:
                cluster_ids = cluster_ids[wei.gt(0)]

            wei = torch.masked_select(wei, wei.gt(0))

            if tolong:
                tar = tar.long()

        if cluster_ids is not None:
            return criterion(out, tar, cluster_ids=cluster_ids)
        else:
            return criterion(out, tar)

    def loss_distill(
        self,
        output: torch.Tensor,          # [B, C] logits ou sorties du modèle (si compute_single_loss en a besoin)
        target: torch.Tensor,          # [B, ...] doit contenir les IDs de région en colonne graph_id_index
        weight: torch.Tensor,          # 
        label : torch.Tensor,
        hidden: torch.Tensor,          # [B, D] embeddings (features) par échantillon
        percent_less: float,           # ex: 0.10 pour 10% pires régions
        percent_high: float,           # ex: 0.10 pour 10% meilleures régions
        graph_id_index: int,           # index de la colonne dans target contenant l'ID de région
        lambda_kd: float = 1.0,        # poids du terme de distillation
        use_cosine: bool = True,       # True = 1 - cos, False = MSE
        tolong=False,
        cluster_ids=None,
        criterion=None
    ):
        """
        Calcule un terme de distillation d'embeddings des régions 'fortes' (meilleures) vers les 'faibles' (pires),
        en se basant sur la loss par région. Retourne:
        - kd_loss: le terme de distillation,
        - region_losses: dict {region_id: loss_scalar} (detach) pour inspection,
        - best_ids / worst_ids: listes d'IDs sélectionnés.

        On suppose l'existence d'une fonction globale:
            compute_single_loss(output_subset, target_subset) -> scalaire (Tensor)
        """

        assert 0 < percent_less <= 1 and 0 < percent_high <= 1, "percentages doivent être dans (0,1]"
        device = output.device
        region_ids = label[:, graph_id_index, -1]
        unique_ids = torch.unique(region_ids)

        # 1) Loss par région (pour le tri)
        region_losses = {}
        for rid in unique_ids:
            mask = (region_ids == rid)
            # IMPORTANT: compute_single_loss peut s'attendre à des shapes [N, C] / [N, ...]
            loss_i = self.compute_single_loss(output[mask], target[mask], weight[mask], cluster_ids, tolong, criterion)
            # on détache pour le tri (ne pas backprop à travers la sélection)
            region_losses[int(rid.item())] = loss_i.detach()

        # 2) Tri des régions par loss (croissant: meilleures d'abord)
        sorted_items = sorted(region_losses.items(), key=lambda kv: kv[1].item())
        n_regions = len(sorted_items)
        k_high = max(1, int(round(percent_high * n_regions)))
        k_low  = max(1, int(round(percent_less * n_regions)))

        best_ids  = [rid for rid, _ in sorted_items[:k_high]]           # meilleures (loss faible)
        worst_ids = [rid for rid, _ in sorted_items[-k_low:]]           # pires (loss élevée)

        # 3) Prototype enseignant = moyenne des embeddings des meilleures régions
        best_mask = torch.zeros_like(region_ids, dtype=torch.bool)
        for rid in best_ids:
            best_mask |= (region_ids == rid)

        # S'il n'y a pas d'échantillon (cas pathologique), on protège
        if best_mask.any():
            teacher_proto = hidden[best_mask].mean(dim=0, keepdim=True)  # [1, D]
        else:
            # fallback: moyenne globale
            teacher_proto = hidden.mean(dim=0, keepdim=True)

        # On "coupe" le gradient côté enseignant (on ne veut pas déplacer les meilleures)
        teacher_proto = teacher_proto.detach()

        # 4) Distillation: on pousse les embeddings des pires vers le prototype enseignant
        worst_mask = torch.zeros_like(region_ids, dtype=torch.bool)
        for rid in worst_ids:
            worst_mask |= (region_ids == rid)

        if worst_mask.any():
            student_emb = hidden[worst_mask]                 # [N_w, D]

            if use_cosine:
                # 1 - cos(sim)  (plus stable d'échelle que MSE)
                student = F.normalize(student_emb, dim=-1)
                teacher = F.normalize(teacher_proto, dim=-1)
                kd = 1.0 - (student @ teacher.T).squeeze(-1) # [N_w]
                kd_loss = kd.mean()
            else:
                # MSE sur embeddings non normalisés
                kd_loss = F.mse_loss(student_emb, teacher_proto.expand_as(student_emb))
        else:
            # aucune région "pire" sélectionnée
            kd_loss = torch.tensor(0.0, device=device)

        # 5) Pondération du terme de distillation
        kd_loss = lambda_kd * kd_loss

        return kd_loss, region_losses, best_ids, worst_ids

    def calculate_loss(self, criterion, output, target, weights, label, tolong=True):

        if 'cluster_ids' in required_params(criterion.forward):
            id_mask = label[:, criterion.id, -1]
            self.cluster_id_index = criterion.id
        else:
            id_mask = None
            
        base_loss = self.compute_single_loss(output, target, weights, id_mask, tolong, criterion)

        if 'area' in self.loss: # Calculate area loss (specify loss-area)
            area_mask = label[:, graph_id_index, -1]
            unique_ids = torch.unique(area_mask)
            values = []
            active_idx = []
            for aid in unique_ids:
                m = area_mask == aid
                if m.sum() == 0:
                    continue
                active_idx.append(int(aid))
                l = self.compute_single_loss(output[m], target[m], weights[m], None, tolong, criterion)
                #print(f'{aid}, {l}')
                values.append(l)
            if len(values) > 0:
                active_idx = torch.as_tensor(active_idx, dtype=torch.long)
                mask = torch.zeros_like(self.area_parameters, dtype=self.area_parameters.dtype)
                mask.index_fill_(0, active_idx, 1.0)
                vals_active = torch.as_tensor(values, device=self.area_parameters.device,
                              dtype=self.area_parameters.dtype)
                vals_full = torch.zeros_like(self.area_parameters)
                vals_full.index_copy_(0, active_idx, vals_active)
                ap_active = self.area_parameters * mask
                eps = 1e-8
                ap_active = torch.log(torch.nn.functional.softplus(ap_active))
                area_loss = (vals_full * ap_active).sum() / ap_active.sum().clamp_min(eps)
            else:
                area_loss = torch.as_tensor(0.0, device=output.device)

            if 'area-global' in self.loss:  # Calculate area * global (classic) loss  (specify loss-area-global)
                loss = area_loss + base_loss
                logger.info(f'area_loss : {area_loss}, {base_loss}, {loss}')
            else:
                loss = area_loss
        else:
            loss = base_loss

        return loss
    
    def calculate_contrastive_moon_loss(self, z, zprev, zglob, temperature=0.5):
        """
        Computes the MOON contrastive loss.

        Args:
            z       : Tensor of shape [batch_size, dim] from current local model.
            zprev   : Tensor of shape [batch_size, dim] from previous local model.
            zglob   : Tensor of shape [batch_size, dim] from global model.
            temperature (float): Temperature parameter τ for scaling similarities.

        Returns:
            loss (Tensor): Scalar contrastive loss for the batch.
        """
        # Normalize representations to compute cosine similarity
        z = F.normalize(z, dim=1)
        zprev = F.normalize(zprev, dim=1)
        zglob = F.normalize(zglob, dim=1)

        # Cosine similarities
        sim_pos = torch.sum(z * zglob, dim=1) / temperature  # similarity with global (positive)
        sim_neg = torch.sum(z * zprev, dim=1) / temperature   # similarity with previous (negative)

        # Contrastive loss per sample
        logits = torch.stack([sim_pos, sim_neg], dim=1)  # shape: [batch_size, 2]
        labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)  # positive is at index 0

        # Use cross-entropy to compute: -log( exp(sim_pos) / (exp(sim_pos) + exp(sim_neg)) )
        loss = F.cross_entropy(logits, labels)
        
        return loss

    def calculate_prototype_loss(self, hidden, target, prototypes):
        """Compute prototype alignment loss."""
        loss = 0.0
        classes = torch.unique(target)
        for cls in classes:
            cls_idx = int(cls.item())
            if prototypes is None or cls_idx not in prototypes:
                continue
            proto = prototypes[cls_idx].to(hidden.device)
            mask = target == cls
            if mask.sum() == 0:
                continue
            diff = hidden[mask] - proto
            loss += torch.mean(torch.norm(diff, dim=1))
        return loss

    def calculate_prototype_alignment_loss(self, hidden, target, prototypes):
        """
        Compute prototype alignment loss:
        Sum over classes of L2 distance squared between local and global prototypes.

        Arguments:
            hidden (Tensor): Embeddings of shape [batch_size, embedding_dim].
            target (Tensor): Class labels of shape [batch_size].
            prototypes (dict): {class_id: global_prototype_tensor}

        Returns:
            loss (Tensor): Scalar tensor representing the total alignment loss.
        """
        loss = 0.0
        classes = torch.unique(target)
        for cls in classes:
            cls_idx = int(cls.item())
            if prototypes is None or cls_idx not in prototypes:
                continue
            # Global prototype
            proto_global = prototypes[cls_idx].to(hidden.device)
            
            # Local prototype for class cls
            mask = target == cls
            if mask.sum() == 0:
                continue
            proto_local = hidden[mask].mean(dim=0)
            
            # L2 distance squared between local and global prototype
            diff = proto_local - proto_global
            loss += torch.sum(diff ** 2)
            
        return loss
    
    def launch_batch(self, data, criterion, batch_type, do_update):
        inputs, labels, _ = data
        graphs = None

        if inputs.shape[0] == 1:
            return 0

        band = -1
        
        hidden_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)
        output_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)

        for H in range(self.horizon + 1):

            if hasattr(self.model, 'is_graph_or_node'):
                is_graph_or_node = self.model.is_graph_or_node
            else:
                is_graph_or_node = False

            target, weights = self.compute_weights_and_target(labels, band, ids_columns, is_graph_or_node, graphs,  -1 - (self.horizon - H))

            if self.loss not in ['kldivloss']: # works on probability
                target = target.long()

            inputs_horizon = self.compute_inputs(inputs,  -1 - (self.horizon - H), "current" if H == 0 else "futur")

            if H == 0:
                z_prev = None
            else:
                if self.ks > 0:
                    # on prend les ks derniers états cachés déjà vus
                    history = hidden_past[-(self.ks + 1):]
                    # empilement (B, D, L) avec L = len(history)
                    z_prev = torch.stack(history, dim=2)  # (B, D, L)

                    # padding à gauche si L < ks
                    L = z_prev.size(2)
                    if L < (self.ks + 1):
                        B, D = z_prev.size(0), z_prev.size(1)
                        pad = torch.zeros(
                            (B, D, self.ks + 1 - L),
                            device=z_prev.device,
                            dtype=z_prev.dtype
                        )
                        z_prev = torch.cat([pad, z_prev], dim=2)  # (B, D, ks)
                else:
                    z_prev = hidden_past[-1]
            if H == 0:
                output, logits, hidden = self.model(inputs_horizon, z_prev=None)
                if batch_type == 'train' and do_update:
                    if has_method(criterion, 'update_after_batch'):
                        criterion.update_after_batch(logits, target)
            else:
                if self.id_past_risk is not None:
                    inputs_horizon[:, self.id_past_risk, -H:] = 0
                if self.id_past_ba is not None:
                    inputs_horizon[:, self.id_past_ba, -H:] = 0
                if self.prev_idx is not None:
                    inputs_horizon[:, self.prev_idx, -H:] = torch.stack(output_past, dim=2)[(-inputs_horizon.shape[-1]):]

                output, logits, hidden = self.model(inputs_horizon, z_prev=z_prev)
            
            hidden_past.append(hidden)
            output_past.append(output)
            
            loss = self.calculate_loss(criterion, logits, target, weights, labels)
            
            if self.student_train: # distallation traning
                criterion_teacher = self.get_loss('kldivloss')
                df_test = pd.DataFrame(inputs_horizon[:, :, -1], columns=self.features_name)
                df_test.columns = df_test.columns.astype(str)
                if self.top_model != 'task':
                    teacher_logits = self.teacher.predict(df_test,
                                                                weights_average=self.weights_average,
                                                                top_model=self.top_model, id_col=(None, None),
                                                                prediction_type='RawFormulaVal')
                else:
                    teacher_logits = self.teacher.predict_with_tasks(df_test,
                                                                    weights_average=self.weights_average,
                                                                    id_col=(None, None), proba='RawFormulaVal')
                
                teacher_logits = torch.Tensor(teacher_logits, device=inputs.device).to(torch.float32)

                T = torch.nn.functional.softplus(self.temperature_value) + 1e-6
                
                p_teacher = F.softmax(teacher_logits / T, dim=1)
                p_student = F.log_softmax(logits / T, dim=1)

                target = target / T

                kl_div_loss = self.calculate_loss(criterion_teacher, p_student, p_teacher, weights, labels, tolong=False) * (T * T)

                loss = self.alpha_value * kl_div_loss + (1 - self.alpha_value) * loss + 1e-3 * (torch.log(T) ** 2)
            
            if self.constrastive: # MOON federated training
                _, _, zprev = self.prev_model(inputs)
                _, _, zglob = self.global_model(inputs)
                loss_constrastive = self.calculate_contrastive_moon_loss(hidden, zprev, zglob, self.moon_temperature_value)
                loss = loss + self.smooth_value * loss_constrastive

            if self.use_prototypes and self.prototypes is not None:
                if not self.model.return_hidden:
                    raise ValueError('Model must return hidden states for prototype training')
                
                proto_loss = self.calculate_prototype_alignment_loss(hidden, target, self.prototypes)
                loss = loss + self.prototype_weight * proto_loss

            if 'distillation' in self.loss:
                distill_loss, region_losses, best, worst = self.loss_distill(
                    output, target, weights, labels, hidden, 0.10, 0.10, graph_id_index,
                    lambda_kd=1, use_cosine=True, tolong=False, cluster_ids=None, criterion=criterion
                )
                loss = loss + distill_loss

                # Update per-epoch best/worst trackers using region_losses
                try:
                    # region_losses is a dict {rid: tensor_loss}
                    if isinstance(region_losses, dict) and len(region_losses) > 0:
                        # Best: smallest loss among reported best IDs
                        if len(best) > 0:
                            best_pair = min(((rid, region_losses[rid].item()) for rid in best if rid in region_losses),
                                            key=lambda kv: kv[1], default=None)
                            if best_pair is not None:
                                rid_b, loss_b = best_pair
                                if hasattr(self, '_epoch_distill_best'):
                                    if loss_b < self._epoch_distill_best['loss']:
                                        self._epoch_distill_best['loss'] = float(loss_b)
                                        self._epoch_distill_best['graph_id'] = int(rid_b)
                        # Worst: largest loss among reported worst IDs
                        if len(worst) > 0:
                            worst_pair = max(((rid, region_losses[rid].item()) for rid in worst if rid in region_losses),
                                            key=lambda kv: kv[1], default=None)
                            if worst_pair is not None:
                                rid_w, loss_w = worst_pair
                                if hasattr(self, '_epoch_distill_worst'):
                                    if loss_w > self._epoch_distill_worst['loss']:
                                        self._epoch_distill_worst['loss'] = float(loss_w)
                                        self._epoch_distill_worst['graph_id'] = int(rid_w)
                except Exception as _e:
                    # Never break training because of logging
                    pass

            if self.model_name in ['BayesianMLP', 'BayesianCNN', 'BayesianRNN']:
                loss += self.model.kl_loss()
            
            if 'total_loss' not in locals():
                total_loss = loss
            else:
                total_loss += loss

        return total_loss
    
    def launch_train_loader(self, loader, criterion, optimizer, do_update):

        self.model.train()
        
        if has_method(criterion, 'get_learnable_parameters'):
            criterion.train()

        # Initialize per-epoch aggregation for distillation best/worst
        if 'distillation' in self.loss:
            self._epoch_distill_best = {'loss': float('inf'), 'graph_id': None}
            self._epoch_distill_worst = {'loss': float('-inf'), 'graph_id': None}
        
        for i, data in enumerate(loader, 0):

            loss = self.launch_batch(data, criterion, 'train', do_update)

            if isinstance(loss, int):
                continue
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
            
            if 'res_loss' in locals():
                res_loss += loss.item()
            else:
                res_loss = loss.item()

            if self.ALATraining:
                
                # Mises à jour SANS autograd
                with torch.no_grad():
                    # 1) update des weights
                    for p_t, p_prev, p_g, w in zip(
                            self.params_p, self.params_tp, self.params_gp, self.weights):
                        upd = w - self.eta * ((p_g - p_prev) * p_t.grad)
                        w.copy_(torch.clamp(upd, 0.0, 1.0))

                    if not self.ala_weight_only:
                        # 2) calcul des params interpolés
                        for p_t, p_prev, p_g, w in zip(
                                self.params_p, self.params_tp, self.params_gp, self.weights):
                            p_t.copy(p_t - self.eta * (p_g - p_prev) * (p_t.grad))

            if optimizer is not None:
                optimizer.step()

                #self.update_weight()
        # After finishing the epoch, persist the best/worst entries for this epoch
        if 'distillation' in self.loss:
            if getattr(self, '_epoch_distill_best', None) is not None and self._epoch_distill_best['graph_id'] is not None:
                self.distill_best_log.append({
                    'epoch': self._current_epoch if self._current_epoch is not None else -1,
                    'graph_id': int(self._epoch_distill_best['graph_id']),
                    'loss': float(self._epoch_distill_best['loss'])
                })
            if getattr(self, '_epoch_distill_worst', None) is not None and self._epoch_distill_worst['graph_id'] is not None:
                self.distill_worst_log.append({
                    'epoch': self._current_epoch if self._current_epoch is not None else -1,
                    'graph_id': int(self._epoch_distill_worst['graph_id']),
                    'loss': float(self._epoch_distill_worst['loss'])
                })

        if has_method(criterion, 'get_attribute'):
            params = criterion.get_attribute()
            dict_params = {'epoch': self._current_epoch}
            for par in params:
                name = par[0]
                value = par[1]
                dict_params[name] = copy.deepcopy(value.detach().cpu().numpy())
            
            self.criterion_params.append(dict_params)

        if self.student_train:
            self.distillation_log.append({'epoch' : self._current_epoch,
                                 'temperature' : copy.deepcopy(self.temperature_value.detach().cpu().numpy()),
                                  'alpha' : copy.deepcopy(self.alpha_value.detach().cpu().numpy())})

        return res_loss

    def launch_val_test_loader(self, loader, criterion, teacher=None):

        self.model.eval()

        if has_method(criterion, 'get_learnable_parameters'):
            criterion.eval()

        total_loss = 0.0

        with torch.no_grad():

            for i, data in enumerate(loader, 0):
                loss = self.launch_batch(data, criterion, 'val', do_update=False)

                total_loss += loss.item()
            
        if 'learnable-area' in self.loss:
            if hasattr(self, 'area_parameters_log'):
                self.area_parameters_log.append(self.area_parameters)
            else:
                self.area_parameters_log = []
                self.area_parameters_log.append(self.area_parameters)

        return total_loss
    
    def make_model(self, graph, custom_model_params):
        model, params = make_model(self.model_name, len(self.features_name), len(self.features_name),
                                graph, dropout, activation,
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params, horizon=self.horizon)

        if self.model_params is None:
            self.model_params = params

        return model, params

    def func_epoch(self, train_loader, val_loader, optimizer, criterion, do_update):

        train_loss = self.launch_train_loader(train_loader, criterion, optimizer, do_update)

        if val_loader is not None:
            val_loss = self.launch_val_test_loader(val_loader, criterion)
        else:
            val_loss = train_loss.item()

        return val_loss, train_loss

    def train(self, graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=True, custom_model_params=None, new_model=True, min_epochs=1):
        """
        Train neural network model
        """

        self.score_per_epochs = {}

        if MLFLOW:
            existing_run = get_existing_run(f'{self.model_name}_')
            if existing_run:
                mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
            else:
                mlflow.start_run(run_name=f'{self.model_name}_', nested=True)

        assert self.train_loader is not None and self.val_loader is not None

        check_and_create_path(self.dir_log)

        criterion = self.get_loss(self.loss)

        static_idx, temporal_idx = get_static_temporal_idx(self.features_name)

        if self.model_name in ['SepGRUGNN']:
            if custom_model_params is None:
                custom_model_params = {'static_idx': static_idx, 'temporal_idx' : temporal_idx}
            else:
                custom_model_params.update({'static_idx': static_idx, 'temporal_idx' : temporal_idx})

        if new_model or self.model is None:
            self.model, _ = self.make_model(graph, custom_model_params)
        
        optimizer = self.get_optimizer(criterion)

        BEST_VAL_LOSS = math.inf
        BEST_MODEL_PARAMS = None
        best_epoch = 0
        patience_cnt = 0

        val_loss_list = []
        train_loss_list = []
        epochs_list = []

        #if (self.dir_log / 'best.pt').is_file():
        if False:
            self._load_model_from_path(self.dir_log / 'best.pt', self.model)
        else:
            for epoch in tqdm(range(epochs), disable=not verbose):
                # Expose current epoch to subroutines for logging
                self._current_epoch = epoch
                val_loss, train_loss = self.func_epoch(train_loader=self.train_loader, val_loader=self.val_loader,
                                                    optimizer=optimizer, criterion=criterion, do_update=epoch < min_epochs)

                val_loss_list.append(round(val_loss, 3))
                train_loss_list.append(round(train_loss, 3))
                epochs_list.append(epoch)
                if val_loss < BEST_VAL_LOSS and epoch > min_epochs:
                    BEST_VAL_LOSS = val_loss
                    BEST_MODEL_PARAMS = self.model.state_dict()
                    patience_cnt = 0
                    best_epoch = epoch
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE_CNT and epoch >= min_epochs:
                        logger.info(f'Loss has not increased for {patience_cnt} epochs. Last best val loss {BEST_VAL_LOSS}, current val loss {val_loss}')
                        save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
                        save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
                        plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
                        if MLFLOW:
                            mlflow.end_run()
                        break
                if MLFLOW:
                    mlflow.log_metric('loss', val_loss, step=epoch)
                if epoch % CHECKPOINT == 0 and verbose:
                    logger.info(f'epochs {epoch}, Val loss {val_loss}')
                    logger.info(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')
                    save_object_torch(self.model.state_dict(), str(epoch)+'.pt', self.dir_log)

            logger.info(f'Last val loss {val_loss}')
            save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
            save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
            plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
        
        self.best_epoch = best_epoch
        logger.info(f'Best epoch {best_epoch}, Best val loss {BEST_VAL_LOSS}')
        ##################################### TEST #################################################
        test_output_, y_ = self._predict_test_loader(self.test_loader, output_pdf='test')
        test_output_ = test_output_.detach().cpu().numpy()
        y_ = y_.detach().cpu().numpy()

        for H in range(self.horizon + 1):
            y = y_[:, :, -1 - (self.horizon - H)]
            test_output = test_output_[:, -1 - (self.horizon - H)]

            if np.any(y[:, -1] > 0) or np.any(test_output > 0):

                under_prediction_score_value = under_prediction_score(y[:, -1], test_output)
                over_prediction_score_value = over_prediction_score(y[:, -1], test_output)
                
                iou = iou_score(y[:, -1], test_output)
                f1 = f1_score((test_output > 0).astype(int), (y[:, -1] > 0).astype(int), zero_division=0)
                iou_area, f1_area = self.compute_area_score(test_output, y[:, -1], y[:, graph_id_index])

                print(f'Horizon {H} -> Test -> Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, IoU {iou}, f1 {f1}, IoU_area {iou_area}, f1_area {f1_area}')
        
        test_output_, y_ = self._predict_test_loader(self.val_loader, output_pdf='test')
        test_output_ = test_output_.detach().cpu().numpy()
        y_ = y_.detach().cpu().numpy()
        
        for H in range(self.horizon + 1):
            check_and_create_path(self.dir_log / f"H{H}")

            y = y_[:, :, -1 - (self.horizon - H)]
            test_output = test_output_[:, -1 - (self.horizon - H)]
            
            under_prediction_score_value = under_prediction_score(y[:, -1], test_output)
            over_prediction_score_value = over_prediction_score(y[:, -1], test_output)
            
            iou = iou_score(y[:, -1], test_output)
            f1 = f1_score((test_output > 0).astype(int), (y[:, -1] > 0).astype(int), zero_division=0)
            iou_area, f1_area = self.compute_area_score(test_output, y[:, -1], y[:, graph_id_index])

            print(f'Horizon {H} -> Val {y.shape} -> Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, IoU {iou} f1 {f1}, IoU_area {iou_area}, f1_area {f1_area}')

            plt.figure(figsize=(15,5))
            plt.plot(y[y[:, departement_index] == 13, -1])
            plt.plot(test_output[y[:, departement_index] == 13])
            plt.savefig(self.dir_log / f"H{H}" / 'test_13.png')

            plt.figure(figsize=(15,5))
            plt.plot(y[y[:, departement_index] == 6, -1])
            plt.plot(test_output[y[:, departement_index] == 6])
            plt.savefig(self.dir_log / f"H{H}" / 'test_6.png')
            plt.close('all')

        if BEST_MODEL_PARAMS is not None:
            self.update_weight(BEST_MODEL_PARAMS)
        
        if 'learnable-area' in self.loss:
            ids = y[:, 0]                       # première colonne
            values = y[:, 1:]

            # Somme groupée par id
            unique_ids, inverse = np.unique(ids, return_inverse=True)
            sums = np.zeros((len(unique_ids), values.shape[1]), dtype=values.dtype)
            np.add.at(sums, inverse, values)
            
            self.plot_area_parameter(epochs_list, y[:, 0], sums[:, -1])
            save_object(self.area_parameters_log, 'area_parameters_log.pkl' ,self.dir_log)

        if has_method(criterion, 'plot_params'):
            if has_method(criterion, 'update_params'):
                criterion.update_params(self.criterion_params[self.best_epoch])
            criterion.plot_params(self.criterion_params, self.dir_log)

        # Save distillation best/worst logs and 3D plot at the end of training
        if 'distillation' in self.loss:
            try:
                self._save_distill_logs_and_plot()
            except Exception as _e:
                # Keep training flow robust even if plotting fails
                logger.info(f"Distillation log/plot skipped: {_e}")

        self.params = BEST_MODEL_PARAMS

    def _save_distill_logs_and_plot(self):
        """Persist best/worst per-epoch logs and save a 3D scatter plot.
        Axes: X=epoch, Y=loss, Z=graph_id. Two series: best (green) and worst (red).
        """
        # Persist raw logs
        logs = {
            'best': self.distill_best_log,
            'worst': self.distill_worst_log,
        }
        save_object(logs, 'distill_best_worst.pkl', self.dir_log)

        if len(self.distill_best_log) == 0 and len(self.distill_worst_log) == 0:
            return

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

        # Prepare arrays for plotting
        bx = [d['epoch'] for d in self.distill_best_log]
        by = [d['loss'] for d in self.distill_best_log]
        bz = [d['graph_id'] for d in self.distill_best_log]

        wx = [d['epoch'] for d in self.distill_worst_log]
        wy = [d['loss'] for d in self.distill_worst_log]
        wz = [d['graph_id'] for d in self.distill_worst_log]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        if len(bx) > 0:
            ax.scatter(bx, by, bz, c='green', marker='o', label='best')
        if len(wx) > 0:
            ax.scatter(wx, wy, wz, c='red', marker='^', label='worst')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_zlabel('Graph ID')
        ax.set_title('Distillation Best/Worst per Epoch')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.dir_log / 'distill_best_worst_3d.png')
        plt.close('all')

    def plot_area_parameter(self, epochs_list, ids, sinisters):
        """
        Plot self.area_parameter (2D array: epochs x parameters) in 3D.
        X = epochs_list
        Y = parameter index
        Z = value of area_parameter
        """
    
        from mpl_toolkits.mplot3d import Axes3D

        sort_ids = np.argsort(sinisters)

        # Vérifier dimensions
        area_param = np.asarray([params.detach().numpy()[sort_ids] for params in self.area_parameters_log])  # doit être (epochs, n_params)
        
        logger.info(f'Last aera params -> {area_param[-1]}')

        # Créer grilles X et Y
        X = epochs_list
        Y = sinisters[sort_ids]
        X, Y = np.meshgrid(X, Y)
        
        # Z = valeurs de self.area_parameter transposées pour correspondre à la grille
        Z = area_param.T  
        
        # Plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Parameter Index')
        ax.set_zlabel('Area Parameter Value')
        ax.set_title('Evolution of Area Parameter during Training')
        
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.savefig(self.dir_log / 'area_parameters.png')
        plt.close()

    def split_dataset(self, dataset, nb, reset=True):
        # Separate the positive and zero classes based on y
        positive_mask = dataset[self.target_name] > 0
        non_fire_mask = dataset[self.target_name] == 0

        # Filtrer les données positives et non feu
        df_positive = dataset[positive_mask]
        df_non_fire = dataset[non_fire_mask]

        # Échantillonner les données non feu
        nb = min(len(df_non_fire), nb)

        if self.n_run == 1:
            seed = self.seed if self.seed is not None else 42
            sampled_indices = np.random.RandomState(seed).choice(len(df_non_fire), nb, replace=False)
        else:
            sampled_indices = np.random.RandomState().choice(len(df_non_fire), nb, replace=False)
            
        df_non_fire_sampled = df_non_fire.iloc[sampled_indices]

        # Combiner les données positives et non feu échantillonnées
        df_combined = pd.concat([df_positive, df_non_fire_sampled])
        # Réinitialiser les index du DataFrame combiné
        if reset:
            df_combined.reset_index(drop=True, inplace=True)
        return df_combined
    
    def add_ordinal_class(self, X, y, limit):        
        pass

    def calculcate_score(self, pred, y, id_mask=None):
        if id_mask is None:
            under_prediction_score_value = under_prediction_score(y, pred)
            over_prediction_score_value = over_prediction_score(y, pred)

            iou = iou_score(y, pred)
            return under_prediction_score_value, over_prediction_score_value, iou
        else:
            uids = np.unique(id_mask)
            under_prediction_score_value = []
            over_prediction_score_value = []
            iou = []
            
            for id in uids:
                mask = (id_mask == id)
                pred_mask = pred[mask]
                y_mask = y[mask]
                
                if np.any(y_mask > 0):

                    under_score = under_prediction_score(y_mask, pred_mask)
                    over_score = over_prediction_score(y_mask, pred_mask)
                    iou_val = iou_score(y_mask, pred_mask)

                    # Stocker les valeurs
                    under_prediction_score_value.append(under_score)
                    over_prediction_score_value.append(over_score)
                    iou.append(iou_val)

                    # Log propre
                    logger.info(
                        f'Id {id} -> under_prediction_score: {under_score}, over_prediction_score: {over_score}, iou: {iou_val}'
                    )
                
                else:
                     logger.info(
                        f'Id {id} -> No fire'
                    )

            under_prediction_score_value = np.trapz(under_prediction_score_value)
            over_prediction_score_value = np.trapz(over_prediction_score_value)
            iou = np.trapz(iou)

            return under_prediction_score_value, over_prediction_score_value, iou

    def compute_area_score(self, pred, y_true, graph_ids):
        unique_graphs = np.unique(graph_ids)
        graph_sums = {gid: y_true[graph_ids == gid].sum() for gid in unique_graphs}
        sorted_graphs = sorted(graph_sums, key=graph_sums.get, reverse=True)

        iou_scores = []
        f1_scores = []

        for gid in sorted_graphs:
            mask = graph_ids == gid
            y_g = y_true[mask]
            pred_g = pred[mask]
            if np.any(y_g > 0):
                iou = jaccard_score((y_g > 0).astype(int), (pred_g > 0).astype(int), zero_division=0)
                f1 = f1_score((y_g > 0).astype(int), (pred_g > 0).astype(int), zero_division=0)
                iou_scores.append(iou)
                f1_scores.append(f1)

        if len(iou_scores) == 0:
            return 0.0, 0.0

        max_area = np.trapz(np.ones(np.unique(graph_ids[y_true > 0]).shape[0]))
        IoU_area = calculate_area_under_curve(iou_scores)
        F1_area = calculate_area_under_curve(f1_scores)
        if max_area == 0:
            return 0, 0
        return IoU_area / max_area, F1_area / max_area

    def search_samples_proportion(self, graph, df_train, df_val, df_test, is_unknowed_risk, epochs, PATIENCE_CNT, CHECKPOINT, reset=True, custom_model_params=None, use_log=True):
        
        check_and_create_path(self.dir_log)

        if not is_unknowed_risk:
                test_percentage = np.round(np.arange(0.1, 1.05, 0.05), 2)
        else:
            test_percentage = np.arange(0.0, 1.05, 0.05)

        if 'MultiScale' in self.model_name:
            test_percentage = np.arange(0.5, 1.05, 0.05)

        under_prediction_score_scores = []
        over_prediction_score_scores = []
        iou_scores = []
        data_log = None
        find_log = False

        self.metrics['test_percentage'] = []
        self.metrics['under_prediction_scores'] = []
        self.metrics['over_predictio_scores'] = []
        self.metrics['iou_scores'] = []

        if use_log:
        #if False:
            if False:
                if (self.dir_log / 'unknowned_scores_per_percentage.pkl').is_file():
                    data_log = read_object('unknowned_scores_per_percentage.pkl', self.dir_log)
            else:
                if (self.dir_log / 'metrics.pkl').is_file():
                    print(f'Load metrics')
                    find_log = True
                    data_log = read_object('metrics.pkl', self.dir_log)
                else:
                    xs = [0, 10]
                    for x in xs:
                        other_model = f'{self.model_name}_search_full_{x}_all_one_{self.target_name}_{self.task_type}_{self.loss}'
                        print(f'{self.dir_log / ".."/ other_model / "metrics.pkl"}')
                        if (self.dir_log / '..'/ other_model / 'metrics.pkl').is_file():
                            data_log = read_object('metrics.pkl', self.dir_log / '..'/ other_model)
                        if data_log is not None:
                            break
                        
                    if data_log is None:
                        xs = [25]
                        for x in xs:
                            other_model = f'{self.model_name}_search_full_{self.ks}_{x}_one_{self.target_name}_{self.task_type}_{self.loss}'
                            print(f'{self.dir_log / ".."/ other_model / "metrics.pkl"}')
                            if (self.dir_log / '..'/ other_model / 'metrics.pkl').is_file():
                                data_log = read_object('metrics.pkl', self.dir_log / '..'/ other_model)
                            if data_log is not None:
                                break

        print(f'data_log : {data_log}')
        if data_log is not None:
            try:
                self.metrics = data_log
                #test_percentage = self.metrics['test_percentage']
                #under_prediction_score_scores = self.metrics['under_prediction_scores']
                #over_prediction_score_scores = self.metrics['over_prediction_scores']
            except Exception as e:
                print(e)
                self.metrics = {}
                data_log = None
                pass

        doSearch = True
        if data_log is not None: #and self.n_run == data_log['n_run']:
            test_percentage = np.asarray(self.metrics['test_percentage'])
            start_test = -1
            for i in range(0, len(test_percentage) - 1):
                start_test = i

                if test_percentage[i] in self.metrics.keys():
                    last_keys = test_percentage[i]
                    val_1 = np.mean(data_log[test_percentage[i]]['iou_val'])
                    val_2 = np.mean(data_log[test_percentage[i + 1]]['iou_val'])

                    std_1 = np.std(data_log[test_percentage[i]]['iou_val'])
                    std_2 = np.std(data_log[test_percentage[i + 1]]['iou_val'])

                    print('#########################"')
                    print(f'{test_percentage[i]} -> {val_1} -> {std_1}')
                    print(f'{test_percentage[i + 1]} -> {val_2} -> {std_2}')

                    try:
                        if  val_1 >  val_2 or ((val_1 == val_2) and (std_1 < std_2)):
                            print(f"Last score {val_1} current score {val_2}")
                            print(f"Last std {std_1} current std {std_2}")
                            doSearch = False
                            tp = test_percentage[i]
                            break
                    except Exception as e:
                        print(e)
                        doSearch = True
                        break

            start_test += 1

        else:
            start_test = 0
        
        if doSearch:
            last_score = -math.inf if start_test == 0 else np.mean(self.metrics[last_keys]['iou_val'])
            y_ori = df_train[self.target_name].values
            for i in range(start_test, test_percentage.shape[0]):
                tp = round(test_percentage[i], 2)

                if tp in self.metrics.keys():
                    continue
                
                df_train_copy = df_train.copy(deep=True)
                
                if not is_unknowed_risk:
                    nb = int(tp * y_ori[y_ori == 0].shape[0])
                else:
                    nb = int(tp * len(X[(X['potential_risk'] > 0) & (y_ori == 0)]))

                logger.info(f'Trained with {tp} -> {nb} sample of class 0')

                for run in range(self.n_run):

                    df_combined = self.split_dataset(df_train_copy, nb, reset=False)

                    # Mettre à jour df_train pour l'entraînement
                    df_train_copy['weight'] = 0
                    df_train_copy.loc[df_combined.index, 'weight'] = 1

                    copy_model = deepcopy(self)
                    copy_model.under_sampling = 'full'
                    copy_model.horizon = 0
                    copy_model.create_train_val_test_loader(graph, df_train_copy, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT, features_importance=False, custom_model_params=custom_model_params)
                    copy_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=False, custom_model_params=custom_model_params)
                    
                    ############################# On set val ##############################
                    test_output, y = copy_model._predict_test_loader(copy_model.val_loader, output_pdf='Val')

                    test_output = test_output[:, 0]
                    y = y[:, :, 0]

                    prediction = test_output.detach().cpu().numpy()
                    
                    y = y.detach().cpu().numpy()
                    if 'MultiScale' in self.model_name:
                        id_mask = y[:, scale_index]
                    else:
                        id_mask = y[:, departement_index]
                        id_mask = None
                
                    dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
                    dff['departement'] = y[:, departement_index]
                    dff[self.target_name] = y[:, -1]
                    y = y[:, -1] > 0 if self.task_type == 'binary' else y[:, -1]

                    metrics_run = evaluate_metrics(dff, self.target_name, prediction)
                    metrics_run = round_floats(metrics_run)
                    under_prediction_score_value = under_prediction_score(y, prediction)
                    over_prediction_score_value = over_prediction_score(y, prediction)
                    update_metrics_as_arrays(self, tp, metrics_run, 'val')

                    ############################# On set test ##############################
                    test_output, y = copy_model._predict_test_loader(copy_model.test_loader, output_pdf='test')

                    test_output = test_output[:, 0]
                    y = y[:, :, 0]

                    prediction = test_output.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()

                    if 'MultiScale' in self.model_name:
                        id_mask = y[:, scale_index]
                    else:
                        id_mask = y[:, departement_index]
                        id_mask = None
                
                    dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
                    dff['departement'] = y[:, departement_index]
                    dff[self.target_name] = y[:, -1]
                    y = y[:, -1] > 0 if self.task_type == 'binary' else y[:, -1]

                    metrics_run = evaluate_metrics(dff, self.target_name, prediction)
                    metrics_run = round_floats(metrics_run)
                    update_metrics_as_arrays(self, tp, metrics_run, 'test')
                
                self.metrics[tp] = add_ic95_to_dict(self.metrics[tp], None, "_ic95")

                iou = np.mean(self.metrics[tp]['iou_val'])
                std_iou = np.std(self.metrics[tp]['iou_val'])

                save_object(self.metrics, 'metrics.pkl', self.dir_log)
                
                print(f'Metrics achieved : {self.metrics[tp]}')

                if iou >= last_score:
                    last_score = iou
                else:
                    print(f'Last score {last_score} current score {iou}')
                    break
        
        keys_array = []
        eps = 1e-9  # tolérance pour considérer deux moyennes égales

        iou_means = []
        iou_stds = []
        for k in self.metrics.keys():
            if isinstance(k, float) and "iou_val" in self.metrics[k]:
                vals = np.asarray(self.metrics[k]["iou_val"], dtype=float)
                mean_iou = float(np.nanmean(vals)) if vals.size else -np.inf
                std_iou = np.std(vals)

                print(f"{k} -> mean={mean_iou:.6f}, std={std_iou:.6f}")
                keys_array.append(k)
                iou_means.append(mean_iou)
                iou_stds.append(std_iou)

        if not keys_array:
            raise ValueError("Aucune clé float avec 'iou_val' trouvée dans self.metrics.")

        # Sélection avec tie-break: max(mean), puis min(std)
        iou_means = np.array(iou_means, dtype=float)
        iou_stds  = np.array(iou_stds,  dtype=float)

        max_mean = np.nanmax(iou_means)
        candidates = np.where(np.isclose(iou_means, max_mean, atol=eps))[0]
        if candidates.size == 1:
            index_max = int(candidates[0])
        else:
            # parmi les ex aequo en moyenne, prendre le plus petit std
            index_max = int(candidates[np.nanargmin(iou_stds[candidates])])

        best_tp = keys_array[index_max]
        logger.info(f'Best tp {best_tp}')
        self.metrics['iou_score'] = iou_scores
        self.metrics['test_percentage'] = test_percentage
        self.metrics['under_prediction_scores'] = under_prediction_score_scores
        self.metrics['over_prediction_scores'] = over_prediction_score_scores
        self.metrics['best_tp'] = best_tp
        self.metrics['run'] = self.n_run

        logger.info(f'{self.metrics[best_tp]}')

        save_object(self.metrics, 'metrics.pkl', self.dir_log)

        return best_tp, find_log

    def search_samples_limit(self, X, y, X_val, y_val, X_test, y_test):
        pass

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions, y = self.predict(X, return_y=True)
        predictions = predictions[:, -1]
        y = y[:, -1, 0]
        return self.score_with_prediction(predictions, y, sample_weight)

    def score_with_prediction(self, y_pred, y, sample_weight=None):
        
        return iou_score(y, y_pred)

    def _predict_test_loader(self, X: DataLoader, prediction_type='Class', output_pdf="test") -> torch.tensor:
            assert self.model is not None
            self.model.eval()
            if len(self.criterion_params) > 0:
                criterion = self.get_loss(self.loss)
                if has_method(criterion, 'update_params'):
                    criterion.update_params(self.criterion_params[self.best_epoch])
                    criterion.eval()

            with torch.no_grad():
                pred = []
                y = []

                for i, data in enumerate(X, 0):
                    
                    inputs, orilabels_, _ = data

                    orilabels_ = orilabels_.to(device)
                    pred_horizon = []
                    labels_horizon = []

                    hidden_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)
                    output_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)
                    for H in range(self.horizon + 1):

                        orilabels = orilabels_[:, :, -1 - (self.horizon - H)]
                        orilabels[:, -1] = orilabels[:,  -1 ] > 0 if self.task_type == 'binary' else orilabels[:,  -1 ]
                        inputs_horizon = self.compute_inputs(inputs,  -1 - (self.horizon - H), "current" if H == 0 else "futur")
                        
                        if H == 0:
                            z_prev = None
                        else:
                            if self.ks > 0:
                                # on prend les ks derniers états cachés déjà vus
                                history = hidden_past[-(self.ks + 1):]
                                # empilement (B, D, L) avec L = len(history)
                                z_prev = torch.stack(history, dim=2)  # (B, D, L)

                                # padding à gauche si L < ks
                                L = z_prev.size(2)
                                if L < (self.ks + 1):
                                    B, D = z_prev.size(0), z_prev.size(1)
                                    pad = torch.zeros(
                                        (B, D, self.ks + 1 - L),
                                        device=z_prev.device,
                                        dtype=z_prev.dtype
                                    )
                                    z_prev = torch.cat([pad, z_prev], dim=2)  # (B, D, ks)
                            else:
                                z_prev = hidden_past[-1]
                        if H == 0:
                            output, logits, hidden = self.model(inputs_horizon, z_prev=None)
                        else:
                            if self.id_past_risk is not None:
                                inputs_horizon[:, self.id_past_risk, -H:] = 0
                            if self.id_past_ba is not None:
                                inputs_horizon[:, self.id_past_ba, -H:] = 0
                            if self.prev_idx is not None:
                                inputs_horizon[:, self.prev_idx, -H:] = torch.stack(output_past, dim=2)[(-inputs_horizon.shape[-1]):]
                            
                            output, logits, hidden = self.model(inputs_horizon, z_prev=z_prev)
                        
                        hidden_past.append(hidden)
                        output_past.append(output)

                        if hasattr(self, 'occ_model'):
                            zeros_col = torch.zeros(inputs_horizon.size(0), 1, inputs_horizon.size(2), device=inputs_horizon.device, dtype=inputs_horizon.dtype)
                            inputs_horizon_occ = torch.cat([inputs_horizon, zeros_col], dim=1)  # -> (5, 4)
                            proba_output, proba_logits, proba_hidden = self.occ_model.model(inputs_horizon_occ)
                            logits = [logits, proba_logits]

                        if 'criterion' in locals() and hasattr(criterion, 'transform'):
                            params = {'inputs' : logits}
                            if 'cluster_ids' in required_params(criterion.transform):
                                cluster_ids = orilabels[:, self.cluster_id_index].long()
                                params['cluster_ids'] = cluster_ids
                            if 'output_pdf' in required_params(criterion.transform):
                                assert output_pdf is not None and self.dir_log is not None
                                params['output_pdf'] = output_pdf
                                params['dir_output'] = self.dir_log

                            output = criterion.transform(**params)

                        if prediction_type == 'Class':

                            if self.task_type == 'classification' or self.task_type == 'binary':
                                output = torch.argmax(output, dim=1)

                            elif self.task_type == 'regression' and output.ndim > 1 and output.shape[1] > 1:
                                output = torch.argmax(output, dim=1)

                        elif prediction_type == 'RawFormulaVal':
                            output = logits

                        pred_horizon.append(output[:, None])
                        labels_horizon.append(orilabels[:, :, None])

                pred_horizon = torch.cat(pred_horizon, dim=1)
                labels_horizon = torch.cat(labels_horizon, dim=2)
                pred.append(pred_horizon)
                y.append(labels_horizon)

                y = torch.cat(y, 0)
                pred = torch.cat(pred, 0)

                #if self.task_type == 'regression' and prediction_type == 'Class':
                if prediction_type == 'Class' and pred.dtype != torch.long:
                #if pred.dtype != torch.long:
                    pred = torch.round(pred, decimals=1)

                return pred, y

    def fit(self, graph, X, y, X_val, y_val, X_test, y_test, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=None, use_log=True):
        
        X = X.set_index(ids_columns[:-1]).join(y.set_index(ids_columns[:-1])[targets_columns + [self.target_name]], on=ids_columns[:-1], how='left').reset_index()
        X_val = X_val.set_index(ids_columns[:-1]).join(y_val.set_index(ids_columns[:-1])[targets_columns + [self.target_name]], on=ids_columns[:-1], how='left').reset_index()
        X_test = X_test.set_index(ids_columns[:-1]).join(y_test.set_index(ids_columns[:-1])[targets_columns  + [self.target_name]], on=ids_columns[:-1], how='left').reset_index()
        #if (self.dir_log / 'last.pt').is_file():
        #    self.graph = graph
        #    self._load_model_from_path(self.dir_log / 'best.pt', self.model)
        #else:
        self.create_train_val_test_loader(graph, X, X_val, X_test, epochs, PATIENCE_CNT, CHECKPOINT, custom_model_params=custom_model_params, use_log=use_log)
        self.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=custom_model_params)
            
    def filtering_pred(self, df, predTensor, y, graph, return_y = False):
        
        y = y.detach().cpu().numpy()
        predTensor = predTensor.detach().cpu().numpy()

        # Extraire les paires de test_dataset_dept
        test_pairs = set(zip(df['date'], df['graph_id'], df['scale']))

        # Normaliser les valeurs dans YTensor
        date_values = [item for item in y[:, date_index]]
        graph_id_values = [item for item in y[:, graph_id_index]]
        scale_values = [item for item in y[:, scale_index]]

        # Filtrer les lignes de YTensor correspondant aux paires présentes dans test_dataset_dept
        filtered_indices = [
            i for i, (date, graph_id, scale) in enumerate(zip(date_values, graph_id_values, scale_values)) 
            if (date, graph_id, scale) in test_pairs
            ]

        # Créer YTensor filtré
        y = y[filtered_indices]

        # Créer des paires et les convertir en set
        ytensor_pairs = set(zip(date_values, graph_id_values, scale_values))

        # Filtrer les lignes en vérifiant si chaque couple (date, graph_id) appartient à ytensor_pairs
        df = df[
            df.apply(lambda row: (row['date'], row['graph_id'], row['scale']) in ytensor_pairs, axis=1)
        ].reset_index(drop=True)
        
        if graph.graph_method == 'graph':
            def keep_one_per_pair(dataset):
                # Supprime les doublons en gardant uniquement la première occurrence par paire (graph_id, date)
                return dataset.drop_duplicates(subset=['graph_id', 'date'], keep='first')
            
            def get_unique_pair_indices(array, graph_id_index, date_index):
                """
                Retourne les indices des lignes uniques basées sur les paires (graph_id, date).
                :param array: Liste de listes (tableau Python)
                :param graph_id_index: Index de la colonne `graph_id`
                :param date_index: Index de la colonne `date`
                :return: Liste des indices correspondant aux lignes uniques
                """
                seen_pairs = set()
                unique_indices = []
                for i, row in enumerate(array):
                    pair = (row[graph_id_index], row[date_index])
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        unique_indices.append(i)
                return unique_indices

            unique_indices = get_unique_pair_indices(y, graph_id_index=graph_id_index, date_index=date_index)
            #predTensor = predTensor[unique_indices]
            y = y[unique_indices]
            df = keep_one_per_pair(df)

        df.sort_values(['graph_id', 'date', 'scale'], inplace=True)
        ind = np.lexsort((y[:, scale_index], y[:,0], y[:,4]))
        y = y[ind]
        predTensor = predTensor[ind]
        
        if self.target_name == 'binary' or self.target_name == 'nbsinister':
            band = -2
        else:
            band = -1

        pred = np.full((predTensor.shape[0], 1), fill_value=np.nan)
        if name in ['Unet', 'ULSTM']:
            pred = np.full((y.shape[0], 2), fill_value=np.nan)
            pred_2D = predTensor
            Y_2D = y
            udates = np.unique(df['date'].values)
            ugraph = np.unique(df['graph_id'].values)
            for graph in ugraph:
                for date in udates:
                    mask_2D = np.argwhere((Y_2D[:, graph_id_index] == graph) & (Y_2D[:, date_index] == date))
                    mask = np.argwhere((y[:, graph_id_index] == graph) & (y[:, date_index] == date))
                    if mask.shape[0] == 0:
                        continue
                    pred[mask[:, 0]] = pred_2D[mask_2D[:, 0], band, mask_2D[:, 1], mask_2D[:, 2]]
        else:
            pred = predTensor

        if return_y:
            return pred, y
        return pred

    def predict(self, df, graph=None, return_y=False, prediction_type='Class'):
        if graph is None and hasattr(self, 'graph'):
            graph = self.graph

        if self.target_name not in list(df.columns):
            df[self.target_name] = 0

        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       None,
                       self.target_name,
                       self.ks,
                       self.horizon)
        
        predTensor, YTensor = self._predict_test_loader(loader, prediction_type='Class')

        if return_y:
        #    pred, y = self.filtering_pred(df, predTensor, YTensor, graph, return_y=return_y)
            return predTensor, YTensor
        
        #pred = self.filtering_pred(df, predTensor, YTensor, graph, return_y=return_y)
        return predTensor
    
    def predict_proba(self, df, graph=None, return_y=False):
        if graph is None:
            graph = self.graph

        if self.target_name not in list(df.columns):
            df[self.target_name] = 0

        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       None,
                       self.target_name,
                       self.ks,
                       self.horizon)
        
        predTensor, YTensor = self._predict_test_loader(loader, True)
        if return_y:
            pred, y = self.filtering_pred(df, predTensor, YTensor, graph, return_y=return_y)
            return pred, y
        pred = self.filtering_pred(df, predTensor, YTensor, graph, return_y=return_y)
        return pred
        
    def plot_train_val_loss(self, epochs, train_loss_list, val_loss_list, dir_log):
        # Création de la figure et des axes
        plt.figure(figsize=(10, 6))

        # Tracé de la courbe de val_loss
        plt.plot(epochs, val_loss_list, label='Validation Loss', color='blue')

        # Ajout de la légende
        plt.legend()

        # Ajout des labels des axes
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Ajout d'un titre
        plt.title('Validation Loss over Epochs')
        plt.savefig(dir_log / 'Validation.png')
        plt.close('all')

        # Tracé de la courbe de train_loss
        plt.plot(epochs, train_loss_list, label='Training Loss', color='red')

        # Ajout de la légende
        plt.legend()

        # Ajout des labels des axes
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Ajout d'un titre
        plt.title('Training Loss over Epochs')
        plt.savefig(dir_log / 'Training.png')
        plt.close('all')

    def _load_model_from_path(self, path : Path, model) -> None:
        model, _ = self.make_model(self.graph, None)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=False)
        self.model = model
        
    def update_weight(self, weight):
        """
        Update the model's weights with the given state dictionary.

        Parameters:
        - weight (dict): State dictionary containing the new weights.
        """

        assert self.model is not None

        if not isinstance(weight, dict):
            raise ValueError("The provided weight must be a dictionary containing model parameters.")

        model_state_dict = self.model.state_dict()

        # Vérification que toutes les clés existent dans le modèle
        missing_keys = [key for key in weight.keys() if key not in model_state_dict]
        if missing_keys:
            raise KeyError(f"Some keys in the provided weights do not match the model's parameters: {missing_keys}")

        # Charger les poids dans le modèle
        self.model.load_state_dict(weight)
    
    def update_model(self, model):
        self.model = deepcopy(model)

    def get_loss(self, loss_name):
        loss_params = {'num_classes' : 5}
        return get_loss_function(loss_name, **loss_params)

    def get_learnable_parameters(self, criterion):
        """
        Retourne les paramètres apprenables (list/param groups) pour l'optimizer.
        Tous les objets doivent être nn.Parameter avec requires_grad=True.
        """
        params = list(self.model.parameters())

        # Ajouter paramètres spécifiques à la loss (s'ils existent)
        if has_method(criterion, 'get_learnable_parameters'):
            logger.info(f'Adding {self.loss} parameter(s)')
            # On s'assure que ce sont bien des nn.Parameter
            loss_params = []
            for p in criterion.get_learnable_parameters().values():
                if isinstance(p, torch.nn.Parameter):
                    loss_params.append(p)
                else:
                    # Convertir un tensor en Parameter si besoin
                    loss_params.append(torch.nn.Parameter(p, requires_grad=True))
            params.extend(loss_params)

        # Ajouter les area_parameters si la loss l'exige
        if 'learnable-area' in self.loss:
            logger.info("Adding learnable area parameters")
            # self.area_parameters est déjà nn.Parameter
            params.append(self.area_parameters)

        if self.student_train and self.temperature == 'seach':
            params.append(self.temperature_value)

        if self.student_train and self.alpha == 'seach':
            params.append(self.alpha_value)

        return params

    def get_optimizer(self, criterion,):
        parameters = self.get_learnable_parameters(criterion)
        optimizer = optim.Adam(parameters, lr=self.lr)
        return optimizer

    def shapley_additive_explanation(self, df, outname, dir_output, mode='bar', figsize=(50, 25), samples=None, samples_name=None):
        """
        Visualisation des valeurs SHAP pour expliquer les prédictions.
        :param df_set: DataFrame des caractéristiques d'entrée.
        :param outname: Nom de sortie pour le fichier d'image.
        :param dir_output: Répertoire où enregistrer les résultats.
        :param mode: Mode de visualisation ('bar' ou 'beeswarm').
        :param figsize: Taille de la figure.
        :param samples: Échantillons spécifiques à analyser.
        :param samples_name: Noms des échantillons à afficher.
        """
        if hasattr(self, 'use_temporal_as_edges'):
            use_temporal_as_edges = self.use_temporal_as_edges
        else:
            use_temporal_as_edges = None

        Xst, e = get_numpy_data(self.graph, df, self.features_name, use_temporal_as_edges, self.ks)
        Xst = torch.Tensor(Xst).to(self.device)

        B, F, T = Xst.shape

        Xst_flat = Xst.reshape((B, F*T))
        df_features = []
        # SHAP DeepExplainer avec wrapper du modèle
        explainer = shap.DeepExplainer(WrapperModel(self.model, F, T, e).to(self.device), Xst_flat)
        shap_values = explainer.shap_values(Xst_flat)

        n_classes = self.out_channels

        # Vérifier si la sortie SHAP est multi-classes
        if n_classes == 1:
            shap_values = shap_values[:, :, np.newaxis]
        
        shap_values = np.asarray(shap_values)
        shap_values = np.reshape(shap_values, (n_classes, B, F, T))
        shap_values = shap_values[:, :, :, -1]
        shap_values = np.moveaxis(shap_values, 0, 2)
        #shap_values = shap_values.values

        # Pour chaque classe, calculer et sauvegarder les résultats SHAP
        for class_idx in range(n_classes):
            # Calcul des valeurs SHAP moyennes et écarts-types
            shap_mean_abs = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
            shap_std_abs = np.std(np.abs(shap_values[:, :, class_idx]), axis=0)
        
            df_shap = pd.DataFrame({
                "mean_abs_shap": shap_mean_abs,
                "stdev_abs_shap": shap_std_abs,
                "name": self.features_name
            }).sort_values("mean_abs_shap", ascending=False)

            df_shap['class'] = class_idx
            df_features.append(df_shap)

            # Visualisation globale (summary_plot) pour chaque classe
            plt.figure(figsize=figsize)
            """if mode == 'bar':
                shap.summary_plot(
                    shap_values[:, :, class_idx],
                    features=Xst_flat, 
                    feature_names=self.features_name,
                    plot_type='bar',
                    show=False
                )
            elif mode == 'beeswarm':
                #print(shap_values[:, :, class_idx].shape, df.values.shape, len(self.features_name))
                fig, ax = plt.subplots(figsize=(10, 6))

                # Générer le graphique SHAP pour une classe spécifique (class_idx)
                shap.summary_plot(
                    shap_values[:, :, class_idx],
                    features=Xst_flat,
                    feature_names=self.features_name,
                    show=False,
                    plot_type="dot",  # Vous pouvez choisir 'dot', 'bar', ou 'violin' comme type de plot
                    ax=ax
                )

                # Ajouter explicitement la colorbar
                plt.colorbar(ax.collections[0], ax=ax)
                """
                #plt.show()

            #print(dir_output / f"{outname}_class_{class_idx}_shapley.png")
            #plt.savefig(dir_output / f"{outname}_class_{class_idx}_shapley.png")
            #plt.close()

            # Visualisations spécifiques aux échantillons (force_plot)
            if samples is not None and samples_name is not None:

                for i, sample in enumerate(samples):
                    plt.figure(figsize=figsize)
                    shap.force_plot(
                        explainer.expected_value[class_idx],
                        shap_values[sample, :, class_idx],
                        features=df.iloc[sample].values,
                        feature_names=self.features_name,
                        matplotlib=True,
                        show=False
                    )

                    plt.savefig(
                        dir_output / f"{outname}_class_{class_idx}_{samples_name[i]}_shapley.png",
                        bbox_inches='tight'
                    )
                    plt.close()

        df_features = pd.concat(df_features)
        save_object(df_features, 'features_importance.pkl', dir_output)




    
    


class Model_Torch(Training):
    def __init__(self, model_name, nbfeatures, batch_size, lr, target_name, task_type, out_channels,
                 dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling, n_run,
                 training_mode='normal', federated_cluster='', cut_layer_name='', input_server_model=0,
                 horizon=0):

        #federated_cluster, model_name, nbfeatures, batch_size, lr, target_name, task_type, out_channels,
        #         dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling, n_run

        super().__init__(model_name=model_name, nbfeatures=nbfeatures, batch_size=batch_size, lr=lr,
                         target_name=target_name, task_type=task_type, features_name=features_name, ks=ks,
                         out_channels=out_channels, dir_log=dir_log, loss=loss, name=name, device=device,
                         under_sampling=under_sampling, over_sampling=over_sampling, n_run=n_run,
                         horizon=horizon)

        self.training_mode = training_mode
        self.federated_cluster = federated_cluster
        self.cut_layer_name = cut_layer_name
        self.input_server_model = input_server_model

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT, features_importance=True, custom_model_params=None, use_log=True):
        self.graph = graph

        if 'learnable-area' in self.loss:
            area_parameters = np.sort(df_train['graph_id'].unique())
            self.area_parameters = torch.nn.Parameter(torch.Tensor(torch.ones_like(torch.tensor(area_parameters))))
        elif 'area' in self.loss:
            area_parameters = np.sort(df_train['graph_id'].unique())
            self.area_parameters = torch.ones_like(torch.tensor(area_parameters))
        
        if False:
            ##################################### Select features #########################################
            importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
            #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
            features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)

            if self.nbfeatures != 'all':
                self.features_name = featuresAll[:int(self.nbfeatures)]
            else:
                self.features_name = featuresAll

        if self.val_loader is None:
            val_dataset, test_dataset = create_test_val_dataset(graph,
                                                                df_val,
                                                                df_test,
                                                                self.features_name,
                                                                self.target_name,
                                                                None,
                                                                self.device, self.ks,
                                                                self.horizon,
                                                                graph_mesh=None,
                                                                gridh2mesh=None,
                                                                mesh2graph=None)
            
            val_loader = DataLoader(val_dataset, val_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, test_dataset.__len__(), False, worker_init_fn=seed_worker, generator=g)
            self.val_loader = val_loader
            self.test_loader = test_loader
        ##################################### Define percentage of 0 samples #########################################
        if self.under_sampling != 'full':
            old_shape = df_train.shape
            y = df_train[self.target_name]
            if 'binary' in self.under_sampling:
                vec = self.under_sampling.split('-')
                try:
                    nb = int(vec[-1]) * len(df_train[df_train[self.target_name] > 0])
                except:
                    logger.info(f'{self.under_sampling} with undefined factor, set to 1 -> {len(df_train[df_train[self.target_name] > 0])}')
                    nb = len(df_train[df_train[self.target_name] > 0])

                df_combined = self.split_dataset(df_train, nb)

                # Mettre à jour df_train pour l'entraînement
                df_train = df_combined

                logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')

            elif self.under_sampling == 'search' or 'percentage' in self.under_sampling:
                if self.training_mode == 'splittraining':
                    raise NotImplementedError('Split training mode is no longer supported.')

                if self.under_sampling == 'search':
                    best_tp, find_log = self.search_samples_proportion(
                        graph, df_train, df_val, df_test, is_unknowed_risk=False,
                        epochs=epochs, PATIENCE_CNT=PATIENCE_CNT, CHECKPOINT=CHECKPOINT,
                        custom_model_params=custom_model_params, use_log=use_log
                    )
                    self.find_log = find_log
                else:
                    vec = self.under_sampling.split('-')
                    try:
                        best_tp = float(vec[-1])
                    except ValueError:
                        logger.info(f'{self.under_sampling} with undefined factor, set to 0.3 -> {0.3 * len(y[y == 0])}')
                        best_tp = 0.3

                nb = int(best_tp * len(y[y == 0]))

                df_combined = self.split_dataset(df_train, nb, reset=False)
                df_train['weight'] = 0

                # Mettre à jour df_train pour l'entraînement
                df_train.loc[df_combined.index, 'weight'] = 1
                logger.info(f'Train mask df_train shape: {old_shape} -> {df_train.shape}')

        if 'smote' in self.over_sampling:

            model_name_lower = getattr(self, 'model_name', '').lower()
            if 'cnn' in model_name_lower or 'gnn' in model_name_lower:
                raise ValueError('Smote is not adaptable to images or GNN models')
            
            if self.ks > 0:
                raise ValueError(f'Smote is not adaptbale to time series')

            # Exemple : si ta cible est dans une colonne 'target'
            matching_ids_columns = [col for col in ids_columns if col != 'date']  # on exclut 'date' du matching

            # 2. Construire X (features + ids_columns), y (target)
            X_full = df_train[self.features_name + matching_ids_columns].copy()
            y = df_train[self.target_name].copy()

            smote_coef = int(self.over_sampling.split('-')[1])
            y_negative = y[y == 0].shape[0]
            
            y_one = min(y[y == 1].shape[0] * smote_coef, y_negative)
            y_two = min(y[y == 2].shape[0] * smote_coef, y_negative)
            y_three = min(y[y == 3].shape[0] * smote_coef, y_negative)
            y_four = min(y[y == 4].shape[0] * smote_coef, y_negative)

            if self.task_type in ['classification', 'ordinal-classification']:
                sampling_strategy = {
                    0: y_negative,
                    1: y_one,
                    2: y_two,
                    3: y_three,
                    4: y_four
                }
                smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
            
            elif self.task_type == 'binary':
                smote = SMOTE(random_state=42, sampling_strategy='auto')

            X_resampled, y_resampled = smote.fit_resample(X_full, y)

            df_train = X_resampled
            df_train[self.target_name] = y_resampled

            for uy in np.unique(df_train[self.target_name]):
                    print(f'Number of {uy} class : {df_train[df_train[self.target_name] == uy].shape}') 

        self.df_train = df_train
        self.df_test = df_test
        self.df_val = df_val

        print(df_train.shape, df_test.shape, df_val.shape)
        print(df_train[df_train['weight'] > 0].shape, df_val[df_val['weight'] > 0].shape)

        ##################################### Create loader #########################################        
        if False:
            train_dataset = read_object('train_dataset.pkl', self.dir_log)
            val_dataset = read_object('val_dataset.pkl', self.dir_log)
            test_dataset = read_object('test_dataset.pkl', self.dir_log)
        else:
            train_dataset = create_train_dataset(graph,
                                                df_train,
                                                self.features_name,
                                                self.target_name,
                                                None,
                                                self.device, self.ks, self.horizon,
                                                graph_mesh=None,
                                                gridh2mesh=None,
                                                mesh2graph=None)
    
            #save_object_torch(train_dataset, 'train_dataset.pkl', self.dir_log)
            #save_object_torch(val_dataset, 'val_dataset.pkl', self.dir_log)
            #save_object_torch(test_dataset, 'test_dataset.pkl', self.dir_log)

            train_loader = DataLoader(train_dataset, batch_size, True, worker_init_fn=seed_worker, generator=g)

            self.train_loader = train_loader

    def create_test_loader(self, graph, df):
        loader = create_test_loader(graph, df,
                       self.features_name,
                       self.device,
                       None,
                       self.target_name,
                       self.ks,
                       self.horizon,
                        graph_mesh=None,
                        gridh2mesh=None,
                        mesh2graph=None)

        return loader

########################################## Federated Learning #########################################

class FederatedLearningModel(RegressorMixin, ClassifierMixin):
    def __init__(self, federated_model, features, federated_cluster='departement', loss='mse',
                 name='FederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification',
                 aggregation_method='max', nbfeatures='all', n_run=1, horizon=0):
        """
        Initialize the Federated Learning Model.

        Parameters:
        - federated_model: The base model to be used across all clusters.
        - federated_cluster: Column name used to identify clusters (default: 'departement').
        - aggregation_method: Method to aggregate local models ('mean', 'median', 'weighted', etc.).
        """
        super().__init__()
        self.features_name = features
        self.federated_cluster = federated_cluster
        self.name = name
        self.loss = loss
        self.dir_log = dir_log
        self.under_sampling = under_sampling
        self.over_sampling = over_sampling
        self.target_name = target_name
        self.post_process = post_process
        self.task_type = task_type
        self.aggregation_method = aggregation_method  # Méthode d'agrégation
        self.global_model = deepcopy(federated_model)  # Modèle global
        self.nbfeatures = nbfeatures
        self.n_run = n_run

        self.horizon = horizon

    def fit(self, df_train, df_val, df_test, graph, args):
        """
        Train local models for each federated cluster, aggregate them into a global model, 
        and stop training once the global score does not improve for patience_count_global epochs.
        """

        #importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        #if self.nbfeatures != 'all':
        #    self.features_name = featuresAll[:int(self.nbfeatures)]
        #else:
            #features_name = featuresAll

        self.global_model.features_name = self.features_name
        self.global_model.nbfeatures = 'all'
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params=None)
        
        # Vérifier que la méthode d'agrégation est implémentée
        if self.aggregation_method not in ['mean', 'median', 'weighted', 'max']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")
        
        print(args)
        # Récupération des paramètres d'entraînement
        global_epochs = args.get('global_epochs')
        local_epochs = args.get('local_epochs')
        patience_count_global = args.get('patience_count_global')
        patience_count_local = args.get('patience_count_local')

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")
        
        print(f"\n--- Training Federated Model for {global_epochs} global epochs ---")
        
        self.metrics = {}

        tp = 'client-based'

        for run in range(self.n_run):

            self.global_model.model = deepcopy(initiate_model)
            self.global_model.model_params = deepcopy(model_params)
            local_models = {}
            self.score_per_epochs = {}
            self.score_per_epochs['epoch'] = []
            self.score_per_epochs['score'] = []

            seed = int(random.random())

            best_global_score = float('-inf')
            patience_counter = 0  # Compteur pour l'arrêt anticipé
            for epoch in range(global_epochs):
                print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

                local_weights = []
                sample_counts = []
                
                for cluster in clusters:

                    if self.federated_cluster == 'departement':
                        if not is_below_threshold():   # threshold par défaut = 0.35
                            print(f"\nSkip: {cluster}")
                            continue

                    print(f"\nTraining local model for cluster: {cluster}")

                    # Création des datasets pour le cluster fédéré
                    df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                        df_train, df_val, df_test, cluster
                    )

                    if np.all(df_train[self.global_model.target_name].values == 0) or np.all(df_val[self.global_model.target_name].values == 0) or np.all(df_test[self.global_model.target_name].values == 0):
                        print(f'Skipping {cluster} due to no positif samples')
                        continue
                    
                    if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                        print(f'Skipping {cluster} due to empty dataset')
                        continue

                    if cluster in local_models.keys():
                        local_model = local_models[cluster]
                        local_model.model.load_state_dict(self.global_model.model.state_dict())
                    else:
                        # Initialisation du modèle local
                        local_model = deepcopy(self.global_model)
                        local_model.seed = seed
                        local_model.name = f'{self.federated_cluster}_{cluster}_{self.global_model.name}'
                        local_model.dir_log = self.dir_log / local_model.name
                        if epoch == 0:
                            check_and_create_path(local_model.dir_log)
                        local_model.features_name = self.features_name
                        local_model.nbfeatures = 'all'

                    if epoch == 0 or cluster not in local_models.keys():
                        local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster, local_epochs, patience_count_local, CHECKPOINT, False)
                        
                    # Entraînement du modèle local
                    local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params=None, new_model=False)
                    
                    local_models[cluster] = local_model

                    # Stocker les poids des modèles locaux
                    local_weights.append(deepcopy(local_model.model.state_dict()))
                    sample_counts.append(len(df_train_cluster))

                # Agréger les modèles locaux dans le modèle global
                self.aggregate_models(local_weights, sample_counts)

                # Évaluer le modèle global
                global_score = self.global_model.score(df_val, df_val[self.target_name])
                print(f"\nGlobal Model Score after epoch {epoch + 1}: {global_score:.4f}")
                self.score_per_epochs['epoch'].append(epoch)
                self.score_per_epochs['score'].append(global_score)

                # Vérifier si le score s'est amélioré
                if global_score > best_global_score:
                    best_global_score = global_score
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Arrêt anticipé si le score global ne s'améliore plus
                if patience_counter >= patience_count_global:
                    print("\nEarly stopping: Global model score did not improve.")
                    break
            
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            ###################### TEST SET ########################
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)

            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')

            ###################### VAL SET ########################
            loader = self.create_test_loader(graph, df_val)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'val')
            plot_score_per_epochs(self.score_per_epochs, self.dir_log, f'score_per_epoch_run_{run}')

        self.metrics['best_tp'] = tp

        self.is_fitted_ = True
        print("\n--- Federated Learning Training Complete ---")

    def create_cluster_set(self, df_train, df_val, df_test, cluster):
        """
        Create training, validation, and test datasets for a given cluster.
        """
        if self.federated_cluster in np.unique(df_train.columns):
            X_cluster = df_train[df_train[self.federated_cluster] == cluster].reset_index(drop=True)
            X_val_cluster = df_val[df_val[self.federated_cluster] == cluster].reset_index(drop=True)
            X_test_cluster = df_test[df_test[self.federated_cluster] == cluster].reset_index(drop=True)

            return X_cluster, X_val_cluster, X_test_cluster
        
        raise NotImplementedError(f"Federated clustering method '{self.federated_cluster}' is not implemented.")

    def aggregate_models(self, local_weights, sample_counts=None):
        """
        Aggregate local models into the global model using the chosen method.
        """
        print(f"\n--- Aggregating models using {self.aggregation_method} ---")

        param_keys = local_weights[0].keys()
        new_state_dict = {}

        for key in param_keys:
            stacked_params = torch.stack([weights[key] for weights in local_weights])

            if self.aggregation_method == 'mean':
                new_state_dict[key] = torch.mean(stacked_params, dim=0)
            elif self.aggregation_method == 'median':
                new_state_dict[key] = torch.median(stacked_params, dim=0)[0]
            elif self.aggregation_method == 'max':
                new_state_dict[key] = torch.max(stacked_params, dim=0)[0]
            elif self.aggregation_method == 'weighted':
                if sample_counts is not None and len(sample_counts) == len(local_weights):
                    weights = torch.tensor(sample_counts, dtype=torch.float32)
                    weights = weights / weights.sum()
                else:
                    weights = torch.tensor([1 / len(local_weights)] * len(local_weights), dtype=torch.float32)
                view_shape = [len(local_weights)] + [1] * (stacked_params.dim() - 1)
                new_state_dict[key] = torch.sum(stacked_params * weights.view(*view_shape), dim=0)

        # Mettre à jour les poids du modèle global
        self.global_model.update_weight(new_state_dict)
        print("\n--- Global Model Weights Updated ---")

    def predict(self, X, graph=None, return_y=False):
        """
        Predict using the aggregated global model.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Please train the model before predicting.")

        print(f'Predicting using Global Model')
        return self.global_model.predict(X, graph, return_y)

    def predict_proba(self, X):
        """
        Predict probabilities using the aggregated global model.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Please train the model before predicting.")

        if hasattr(self.global_model, "predict_proba"):
            return self.global_model.predict_proba(X)
        else:
            raise AttributeError("The global model does not support predict_proba.")
        
    def make_model(self, graph, custom_model_params):
        return self.global_model.make_model(graph, custom_model_params)
    
    def create_test_loader(self, graph, df):
        return self.global_model.create_test_loader(graph, df)
    
    def _predict_test_loader(self, X):
        return self.global_model._predict_test_loader(X)

############################################ ALA Federated Model ##############################################################
class FederatedALA(FederatedLearningModel):
    """Federated learning strategy relying on the :class:`Training` class for local training."""

    def __init__(self, federated_model, eta, features, federated_cluster='departement', loss='mse',
                 name='FederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification',
                 aggregation_method='max', nbfeatures='all', n_run=1, params_to_update=['linear2'], horizon=0):


        super().__init__(federated_model=federated_model, features=features, federated_cluster=federated_cluster,
                         loss=loss, name=name, dir_log=dir_log, under_sampling=under_sampling,
                         over_sampling=over_sampling, target_name=target_name, post_process=post_process,
                         task_type=task_type, aggregation_method=aggregation_method, nbfeatures=nbfeatures,
                         n_run=n_run, horizon=horizon)
        self.eta = eta
        self.weight = 0.5
        self.params_to_update = params_to_update
        self.horizon = horizon

    def pick_params_by_name(self, model):
        names, params = [], []
        for n, p in model.named_parameters():
            for pick_parm in self.params_to_update:
                if pick_parm in n:
                    names.append(n); params.append(p)
        return names, params
    
    def pick_params_to_replace(self, model):
        names, params = [], []
        for n, p in model.named_parameters():
            add = True
            for pick_parm in self.params_to_update:
                if pick_parm in n:
                    add = False
                    break
            if add:
                names.append(n); params.append(p)
        return names, params

    def fedALA_params(self, local_model):
        with torch.no_grad():
            for p_t, p_g in zip(local_model.params_erase,
                                local_model.params_gp_erase,
                                ):
                p_t.copy_(p_g)

            for p_t, p_prev, p_g, w in zip(local_model.params_p,
                                        local_model.params_tp,
                                        local_model.params_gp,
                                        local_model.weights):
                # met tout sur le bon device/dtype
                p_prev = p_prev.to(local_model.device, dtype=p_t.dtype)
                p_g    = p_g.detach().to(local_model.device, dtype=p_t.dtype)
                w      = w.to(local_model.device, dtype=p_t.dtype)
                w = w.clamp_(0, 1)
                p_t.copy_(p_prev + (p_g - p_prev) * w)

    def fit(self, df_train, df_val, df_test, graph, args):
        """
        Train local models for each federated cluster, aggregate them into a global model, 
        and stop training once the global score does not improve for patience_count_global epochs.
        """

        importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        #if self.nbfeatures != 'all':
        #    self.features_name = featuresAll[:int(self.nbfeatures)]
        #else:
        #    self.features_name = featuresAll

        self.global_model.features_name = self.features_name
        self.global_model.nbfeatures = 'all'
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params=None)

        # Vérifier que la méthode d'agrégation est implémentée
        if self.aggregation_method not in ['mean', 'median', 'weighted', 'max']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")
        
        # Récupération des paramètres d'entraînement
        global_epochs = args.get('global_epochs')
        local_epochs = args.get('local_epochs')
        patience_count_global = args.get('patience_count_global')
        patience_count_local = args.get('patience_count_local')

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")

        print(f"\n--- Training ALA Federated Model for {global_epochs} global epochs ---")

        tp = 'client-based'
        self.metrics = {}
        
        for run in range(self.n_run):
            
            self.score_per_epochs = {}
            self.model_params = deepcopy(model_params)
            self.global_model.model = deepcopy(initiate_model)
            self.global_model.model_params = deepcopy(model_params)
            self.global_model.eta = self.eta
            local_models = {}
            self.score_per_epochs['epoch'] = []
            self.score_per_epochs['score'] = []
            
            best_global_score = float('-inf')
            patience_counter = 0  # Compteur pour l'arrêt anticipé
            seed = int(random.random())

            for epoch in range(global_epochs):
                print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

                local_weights = []
                sample_counts = []
                
                for cluster in clusters:

                    if self.federated_cluster == 'departement':
                        if not is_below_threshold():   # threshold par défaut = 0.35
                            print(f"\nSkip: {cluster}")
                            continue

                    print(f"\nTraining local model for cluster: {cluster}")
                    # Création des datasets pour le cluster fédéré
                    df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                        df_train, df_val, df_test, cluster
                    )

                    if np.all(df_train[self.global_model.target_name].values == 0) or np.all(df_val[self.global_model.target_name].values == 0) or np.all(df_test[self.global_model.target_name].values == 0):
                        print(f'Skipping {cluster} due to no positif samples')
                        continue

                    if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                        print(f'Skipping {cluster} due to empty dataset')
                        continue

                    # Initialisation du modèle local
                    if cluster not in local_models.keys():
                        local_model = deepcopy(self.global_model)
                        local_model.ALATraining = False
                        local_model.seed = seed
                        local_model.features_name = self.features_name
                        local_model.nbfeatures = 'all'
                        self.metrics[cluster] = local_model.metrics
                        local_model.name = f'{self.federated_cluster}_{cluster}_{self.global_model.name}'
                        local_model.dir_log = self.dir_log / local_model.name
                        local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster, local_epochs, patience_count_local, CHECKPOINT, False)
                        check_and_create_path(local_model.dir_log)
                    else:
                        local_model = local_models[cluster]
                        local_model.ALAtraining = True
                        local_model.ala_weight_only = False

                        local_model.global_model = self.global_model.model
                        _, params = self.pick_params_by_name(local_model.model)
                        _, params_g = self.pick_params_by_name(self.global_model.model)

                        _, params_erase = self.pick_params_to_replace(local_model.model)
                        _, params_gp_erase = self.pick_params_to_replace(self.global_model.model)

                        assert len(params) > 0
                        
                        local_model.params_p  = params
                        local_model.params_gp = params_g
                        local_model.params_erase = params_erase
                        local_model.params_gp_erase = params_gp_erase

                        # snapshot des poids locaux précédents (gelés)
                        local_model.params_tp = [p.detach().clone() for p in local_model.params_p]
                        
                        for param in local_model.params_tp:
                            param.requires_grad = False
                        
                        if not hasattr(local_model, "weights") and cluster in local_models.keys():
                            local_model_log_params = deepcopy(local_model.model.state_dict())
                            w_t = torch.as_tensor(self.weight, device=local_model.device, dtype=local_model.params_p[0].dtype)
                            local_model.weights = [
                                w_t.expand_as(p).clone() for p in local_model.params_p
                            ]
                            local_model.ala_weight_only = True
                            local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params={'return_hidden' : True}, new_model=False)
                            local_model.model.load_state_dict(local_model_log_params)

                        # Fed ala params
                        self.fedALA_params(local_model)

                    # Entraînement du modèle local
                    local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs if not hasattr(local_model, "weights") else 1, verbose=False, custom_model_params={'return_hidden' : True}, new_model=False)
                    
                    local_models[cluster] = local_model
                    
                    # Stocker les poids des modèles locaux
                    local_weights.append(deepcopy(local_model.model.state_dict()))
                    sample_counts.append(len(df_train_cluster))

                # Agréger les modèles locaux dans le modèle global
                self.aggregate_models(local_weights, sample_counts)

                # Évaluer le modèle global
                global_score = self.global_model.score(df_val, df_val[self.target_name])
                self.score_per_epochs['score'].append(global_score)
                self.score_per_epochs['epoch'].append(epoch)
                print(f"\nGlobal Model Score after epoch {epoch + 1}: {global_score:.4f}")

                # Vérifier si le score s'est amélioré
                if global_score > best_global_score:
                    best_global_score = global_score
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Arrêt anticipé si le score global ne s'améliore plus
                if patience_counter >= patience_count_global:
                    print("\nEarly stopping: Global model score did not improve.")
                    break

            ###################### TEST SET ########################
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)

            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')

            ###################### VAL SET ########################
            loader = self.create_test_loader(graph, df_val)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'val')
            plot_score_per_epochs(self.score_per_epochs, self.dir_log, f'score_per_epoch_run_{run}')

        self.metrics['best_tp'] = tp

        self.is_fitted_ = True
        print("\n--- ALA Federated Learning Training Complete ---")

############################################ MOON Federated Model ##############################################################

class MOONFederatedLearning(FederatedLearningModel):
    def __init__(self, federated_model, features, federated_cluster='departement', loss='mse',
                 name='MoonFederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification',
                 aggregation_method='max', nbfeatures='all', n_run=1, temperature=1, smooth=0, horizon=0):

        super().__init__(federated_model=federated_model, features=features, federated_cluster=federated_cluster, loss=loss,
                         name=name, dir_log=dir_log, under_sampling=under_sampling, over_sampling=over_sampling,
                         target_name=target_name, post_process=post_process, task_type=task_type,
                         aggregation_method=aggregation_method, nbfeatures=nbfeatures, n_run=n_run, horizon=horizon)

        self.moon_temperature_value = temperature
        self.smooth_value = smooth

        self.horizon = horizon
    def __init__(self, federated_model, features, federated_cluster='departement', loss='mse', 
                 name='MoonFederatedModel', dir_log=Path('../'), under_sampling='full', over_sampling='full',
                 target_name='nbsinister', post_process=None, task_type='classification', 
                 aggregation_method='max', nbfeatures='all', n_run=1, temperature=1, smooth=0):
        
        super().__init__(federated_model=federated_model, features=features, federated_cluster=federated_cluster, loss=loss,
                         name=name, dir_log=dir_log, under_sampling=under_sampling, over_sampling=over_sampling,
                         target_name=target_name, post_process=post_process, task_type=task_type,
                         aggregation_method=aggregation_method, nbfeatures=nbfeatures, n_run=n_run)
        
        self.moon_temperature_value = temperature
        self.smooth_value = smooth
    
    def fit(self, df_train, df_val, df_test, graph, args):
        """
        Train local models for each federated cluster, aggregate them into a global model, 
        and stop training once the global score does not improve for patience_count_global epochs.
        """

        #importance_df = calculate_and_plot_feature_importance(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #importance_df = calculate_and_plot_feature_importance_shapley(df_train[self.features_name], df_train[self.target_name], self.features_name, self.dir_log / '../importance', self.target_name)
        #features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        #if self.nbfeatures != 'all':
        #    self.features_name = featuresAll[:int(self.nbfeatures)]
        #else:
        #    self.features_name = featuresAll

        self.global_model.features_name = self.features_name
        self.global_model.nbfeatures = 'all'
        self.global_model.graph = graph

        initiate_model, model_params = self.global_model.make_model(graph, custom_model_params={'return_hidden' : True})

        # Vérifier que la méthode d'agrégation est implémentée
        if self.aggregation_method not in ['mean', 'median', 'weighted', 'max']:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' is not implemented.")

        # Récupération des paramètres d'entraînement
        global_epochs = args.get('global_epochs')
        local_epochs = args.get('local_epochs')
        patience_count_global = args.get('patience_count_global')
        patience_count_local = args.get('patience_count_local')

        if self.federated_cluster in np.unique(df_train.columns):
            clusters = df_train[self.federated_cluster].unique()
        else:
            raise NotImplementedError(f"Aggregation method '{self.federated_cluster}' is not implemented.")
        
        print(f"\n--- Training Federated Model for {global_epochs} global epochs ---")

        tp = 'client-based'
        
        self.metrics = {}
        for run in range(self.n_run):

            self.score_per_epochs = {}
            self.global_model.model = deepcopy(initiate_model)
            self.global_model.model_params = deepcopy(model_params)
            local_models = {}
            self.score_per_epochs['epoch'] = []
            self.score_per_epochs['score'] = []
            
            best_global_score = float('-inf')
            patience_counter = 0  # Compteur pour l'arrêt anticipé
            seed = int(random.random())

            for epoch in range(global_epochs):
                print(f"\n--- Global Epoch {epoch + 1}/{global_epochs} ---")

                local_weights = []
                sample_counts = []
                
                for cluster in clusters:

                    if self.federated_cluster == 'departement':
                        if not is_below_threshold():   # threshold par défaut = 0.35
                            print(f"\nSkip: {cluster}")
                            continue

                    print(f"\nTraining local model for cluster: {cluster}")
                    # Création des datasets pour le cluster fédéré
                    df_train_cluster, df_val_cluster, df_test_cluster = self.create_cluster_set(
                        df_train, df_val, df_test, cluster
                    )

                    if np.all(df_train[self.global_model.target_name].values == 0) or np.all(df_val[self.global_model.target_name].values == 0) or np.all(df_test[self.global_model.target_name].values == 0):
                        print(f'Skipping {cluster} due to no positif samples')
                        continue

                    if df_val_cluster.shape[0] == 0 or df_train_cluster.shape[0] == 0:
                        print(f'Skipping {cluster} due to empty dataset')
                        continue

                    # Initialisation du modèle local
                    if cluster in local_models.keys():
                        local_model = local_models[cluster]
                        local_model.model.load_state_dict(self.global_model.model.state_dict())
                    else:
                        local_model = deepcopy(self.global_model)
                        local_model.seed = seed
                        local_model.name = f'{self.federated_cluster}_{cluster}_{self.global_model.name}'
                        local_model.dir_log = self.dir_log / local_model.name
                        print(local_model.dir_log)
                        local_model.global_model = self.global_model
                        local_model.moon_temperature_value = self.moon_temperature_value
                        local_model.smooth_value = self.smooth_value
                        local_model.constrastive = True

                    if epoch == 0 or cluster not in local_models.keys():
                        local_model.prev_model = self.global_model.model
                        local_model.constrastive = False
                        check_and_create_path(local_model.dir_log)
                    else:
                        local_model.prev_model = deepcopy(local_models[cluster].model)
                    
                    local_model.global_model = deepcopy(self.global_model.model)

                    local_model.features_name = self.features_name
                    local_model.nbfeatures = 'all'

                    if epoch == 0 or cluster not in local_models.keys():
                        local_model.create_train_val_test_loader(graph, df_train_cluster, df_val_cluster, df_test_cluster, local_epochs, patience_count_local, CHECKPOINT, False, custom_model_params={'return_hidden' : True})
                        self.metrics[cluster] = local_model.metrics

                    local_model.train(graph, patience_count_local, CHECKPOINT, local_epochs, verbose=False, custom_model_params=None, new_model=False)
                    
                    local_models[cluster] = local_model

                    # Stocker les poids des modèles locaux
                    local_weights.append(deepcopy(local_model.model.state_dict()))
                    sample_counts.append(len(df_train_cluster))

                # Agréger les modèles locaux dans le modèle global
                self.aggregate_models(local_weights, sample_counts)

                # Évaluer le modèle global
                global_score = self.global_model.score(df_val, df_val[self.target_name])
                print(f"\nGlobal Model Score after epoch {epoch + 1}: {global_score:.4f}")
                self.score_per_epochs['epoch'].append(epoch)
                self.score_per_epochs['score'].append(global_score)

                # Vérifier si le score s'est amélioré
                if global_score > best_global_score:
                    best_global_score = global_score
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Arrêt anticipé si le score global ne s'améliore plus
                if patience_counter >= patience_count_global:
                    print("\nEarly stopping: Global model score did not improve.")
                    break
            
            loader = self.create_test_loader(graph, df_test)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')
            
            loader = self.create_test_loader(graph, df_val)
            test_output, y = self._predict_test_loader(loader)
            test_output = test_output.detach().cpu().numpy()
            
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]
            
            metrics_run = evaluate_metrics(dff, self.target_name, test_output)
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'val')
            plot_score_per_epochs(self.score_per_epochs, self.dir_log, f'score_per_epoch_run_{run}')

        self.metrics['best_tp'] = tp
        self.is_fitted_ = True
        print("\n--- Federated Learning Training Complete ---")

############################################ Proto Federated Learning ############################################


############################################ KNOWNLEDEG DISTILLATION ##############################################################


############################################ VOTING MODEL ##############################################################

class ModelVotingPytorchAndSklearn(RegressorMixin, ClassifierMixin):
    def __init__(self, models, features, loss='mse', name='ModelVoting', dir_log=Path('../'), under_sampling='full', target_name='nbsinister', post_process=None, task_type='classification', horizon=0):
        """
        Initialize the ModelVoting class.

        Parameters:
        - models: A list of base models to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', 'mse', 'rmse', etc.).
        """
        super().__init__()
        self.best_estimator_ = models  # Now a list of models
        self.feature_names = features
        self.name = name
        self.loss = loss
        self.is_fitted_ = [False] * len(models)  # Keep track of fitted models
        self.features_per_model = []
        self.dir_log = dir_log
        self.post_process = post_process
        self.under_sampling = under_sampling
        self.target_name = target_name
        self.task_type = task_type
        self.horizon = horizon

    def fit(self, X, y, X_val, y_val, X_test, y_test, args, use_log=True):
        """
        Train each model on the corresponding data.

        Parameters:
        - X_list: List of training data for each model.
        - y_list: List of labels for the training data for each model.
        - optimization: Optimization method to use ('grid' or 'bayes').
        - grid_params_list: List of parameters to optimize for each model.
        - fit_params_list: List of additional parameters for the fit function for each model.
        - cv_folds: Number of cross-validation folds.
        """

        ######## Pytorch params
        PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params_list = args['PATIENCE_CNT'], args['CHECKPOINT'], args['epochs'], args['custom_model_params_list']
        graph = args['graph']
        training_mode = args['training_mode']
        optimization = args['optimization']
        grid_params_list = args['grid_params_list']
        fit_params_list = args['fit_params_list']
        cv_folds = args['cv_folds']
        
        df_val = X_val.copy(deep=True)
        df_val[y_val.columns] = y_val

        self.cv_results_ = []
        self.is_fitted_ = [True] * len(self.best_estimator_)
        self.weights_for_model = []
        for i, model in enumerate(self.best_estimator_):
            model.dir_log = self.dir_log / '..' / model.name
            print(f'Fitting model -> {model.name}')

            #if issubclass(model, Model_Torch):
            model.fit(graph, X, y, X_val, y_val, X_test, y_test, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=None, use_log=use_log)
            #else:
            #    model.fit(graph, X, y[targets[i]], X_val, y_val[targets[i]], training_mode=training_mode, optimization=optimization, grid_params=grid_params_list[i], fit_params=fit_params_list[i], cv_folds=cv_folds)
            target_name_model = model.target_name
            model.target_name = self.target_name
            print(f'Change target name {target_name_model} to {model.target_name}')
            test_loader = model.create_test_loader(graph, df_val)
            test_output, y_test_val = model._predict_test_loader(test_loader)
            test_output = test_output.detach().cpu().numpy()
            y_test_val = y_test_val.detach().cpu().numpy()[:, -1]
                
            model.target_name = target_name_model

            score_model = self.score_with_prediction(y_test_val, test_output)
            self.weights_for_model.append(score_model)

        self.weights_for_model = np.asarray(self.weights_for_model)
        # Affichage des poids et des modèles
        print("\n--- Final Model Weights ---")
        for model, weight in zip(self.best_estimator_, self.weights_for_model):
            print(f"Model: {model.name}, Weight: {weight:.4f}")

        # Plot des poids des modèles
        model_names = [model.name for model in self.best_estimator_]
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, self.weights_for_model, color='skyblue', edgecolor='black')
        plt.title('Model Weights', fontsize=16)
        plt.xlabel('Models', fontsize=14)
        plt.ylabel('Weights', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.tight_layout()
        plt.savefig(self.dir_log / 'weights_of_models.png')
        plt.close('all')

        save_object([model_names, self.weights_for_model], f'weights.pkl', self.dir_log)

    def predict_nbsinister(self, X, ids=None, preprocessor_ids=None, hard_or_soft='soft', weights_average=True):
        
        if self.target_name == 'nbsinister':
            return self.predict(X)
        else:
            assert self.post_process is not None
            predict = self.predict(X)
            return self.post_process.predict_nbsinister(predict, ids)
    
    def predict_risk(self, X, ids=None, preprocessor_ids=None, hard_or_soft='soft', weights_average=True):

        if self.task_type == 'classification':
            return self.predict(X)
        
        elif self.task_type == 'binary':
                assert self.post_process is not None
                predict = self.predict_proba(X, return_y=False)[:, 1]

                if isinstance(ids, pd.Series):
                    ids = ids.values
                if isinstance(preprocessor_ids, pd.Series):
                    preprocessor_ids = preprocessor_ids.values
    
                return self.post_process.predict_risk(predict, None, ids, preprocessor_ids)
        else:
            assert self.post_process is not None
            predict = self.predict(X)

            if isinstance(ids, pd.Series):
                ids = ids.values
            if isinstance(preprocessor_ids, pd.Series):
                preprocessor_ids = preprocessor_ids.values

            return self.post_process.predict_risk(predict, None, ids, preprocessor_ids)

    def predict_with_weight(self, X, hard_or_soft='soft', weights_average='weight', weights2use=[], top_model='all', prediction_type="Class"):
        
        models_list = np.asarray([estimator.name for estimator in self.best_estimator_])
        weights2use = np.asarray(weights2use)
        
        if hard_or_soft == 'hard' or prediction_type == 'RawFormulaVal':
            if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(weights2use)
                models_list = models_list[key]
                models_list = models_list[-top_model:]
                #weights2use = weights2use[np.asarray(key)]
                #weights2use = weights2use[-top_model:]
            else:
                key = np.arange(0, len(self.best_estimator_))

            models_to_mean = []
            predictions = []
            for i, estimator in enumerate(self.best_estimator_):
                if estimator.target_name == self.target_name:
                    pred, y = estimator.predict(X, return_y=True, prediction_type=prediction_type)
                if estimator.name not in models_list:
                    continue
                else:
                    if estimator.target_name != self.target_name:
                        pred = estimator.predict(X, return_y=False, prediction_type=prediction_type)

                    predictions.append(pred.detach().cpu().numpy())

                models_to_mean.append(key[i])

            try:
                weights2use = weights2use[models_to_mean]
            except:
                pass
            # Aggregate predictions
            aggregated_pred = self.aggregate_predictions(predictions, models_to_mean, weights2use)
            #print(aggregated_pred)
            #print(y)
            return aggregated_pred, y.detach().cpu().numpy()
        elif hard_or_soft == 'None':
            top_model = int(top_model)
            key = np.argsort(weights2use)
            idx = key[-top_model]
            estimator = self.best_estimator_[idx]
            if estimator.target_name == self.target_name:
                pred, y = estimator.predict(X, return_y=True)
                return pred.detach().cpu().numpy(), y.detach().cpu().numpy(),
            else:
                pred = estimator.predict(X, return_y=False)
                y = None
                for estimator in self.best_estimator_:
                    if estimator.target_name == self.target_name:
                        _, y = estimator.predict(X, return_y=True)
                        return pred.detach().cpu().numpy(), y.detach().cpu().numpy(),
        else:
            aggregated_pred, y = self.predict_proba_with_weights(X, weights_average=weights_average, top_model=top_model, weights2use=weights2use)
            predictions = np.argmax(aggregated_pred, axis=1)
            return predictions, y

    def predict_proba_with_weights(self, X, hard_or_soft='soft', weights_average='weight', top_model='all', weights2use=[], id_col=(None, None)):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.
        
        Returns:
        - Aggregated predicted probabilities.
        """
        models_list = np.asarray([estimator.name for estimator in self.best_estimator_])
        weights2use = np.asarray(weights2use)

        if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(weights2use)
                models_list = models_list[np.asarray(key)]
                models_list = models_list[-top_model:]
                #weights2use = weights2use[np.asarray(key)]
                #weights2use = weights2use[-top_model:]
        else:
            key = np.arange(0, len(self.best_estimator_))
        
        probas = []
        models_to_mean = []
        print(models_list)

        if hard_or_soft == 'None':
            top_model = int(top_model)
            idx = np.argsort(weights2use)[-top_model]
            estimator = self.best_estimator_[idx]
            if estimator.target_name == self.target_name:
                pred, y = estimator.predict_proba(X, return_y=True)
                return pred.detach().cpu().numpy(), y.detach().cpu().numpy()
            else:
                pred = estimator.predict(X, return_y=False)
                y = None
                for estimator in self.best_estimator_:
                    if estimator.target_name == self.target_name:
                        _, y = estimator.predict_proba(X, return_y=True)
                        return pred.detach().cpu().numpy(), y.detach().cpu().numpy()

        for i, estimator in enumerate(self.best_estimator_):
            X_ = X
            if estimator.target_name == self.target_name:
                proba, y = estimator.predict_proba(X, return_y=True)
            if estimator.name not in models_list:
                continue
            else:
                if estimator.target_name != self.target_name:
                    proba = estimator.predict_proba(X_, return_y=False)
            if proba.shape[1] != 5:
                continue
            #print(estimator.name, np.asarray(probas).shape)
            models_to_mean.append(key[i])
            probas.append(proba)
        try:
            weights2use = weights2use[models_to_mean]
        except:
            pass
        # Aggregate probabilities
        aggregated_proba = self.aggregate_probabilities(probas, models_to_mean, weights2use)
        return aggregated_proba, y
    
    def predict_with_tasks(
        self,
        X,
        hard_or_soft="soft",
        weights_average="weight",
        model_per_task=None,
        generalized_departement=None,
        id_col=(None, None),
        prediction_type='Class'
    ):
        """Predict with a specific ``top_model`` per task.

        Parameters
        ----------
        X : pandas.DataFrame
            Input dataframe.
        hard_or_soft : str
            Mode passed to :func:`predict_with_weight`.
        weights_average : str
            Weighting mode for aggregation.
        model_per_task : dict
            Mapping of task names to number of models to use.
        generalized_departement : list
            Departments for which to apply ``generalized_prediction`` task.
        id_col : tuple
            Id column information for weighted predictions.
        """

        if model_per_task is None:
            model_per_task = {}
        
        if model_per_task == 'default':
            model_per_task={'normal_predictions' : 4,
                                'generalized_prediction' : 12,
                                'class_value_2_predictions' : 12,
                                'class_value_3_predictions' : 20,
                                'class_value_4_predictions' : 20
                                }
            
            generalized_departement = [
                1.,  2.,  3.,  4.,  5.,  8.,  9., 10., 12., 14.,
                15., 16., 17., 18., 19., 21., 22., 23., 24., 25.,
                26., 27., 28., 29., 31., 32., 35., 36., 37., 38.,
                39., 41., 42., 43., 44., 45., 46., 47., 48., 49.,
                50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
                60., 61., 62., 63., 64., 65., 67., 68., 69., 70.,
                71., 72., 73., 74., 75., 76., 77., 78., 79., 80.,
                81., 82., 85., 86., 87., 88., 89., 90., 91., 92.,
                93., 94., 95.
            ]
        

        # Normal prediction for all samples
        top_model = model_per_task.get("normal_predictions", "all")
        if prediction_type == 'Class' or prediction_type == 'RawFormulaVal':
            predictions = self.predict_with_weight(
                X,
                hard_or_soft=hard_or_soft,
                weights_average=weights_average,
                weights2use=self.weights_for_model,
                top_model=top_model,
                prediction_type=prediction_type
            )
        else:
            predictions = self.predict_proba_with_weights(
                X,
                hard_or_soft=hard_or_soft,
                weights_average=weights_average,
                weights2use=self.weights_for_model,
                top_model=top_model,
                prediction_type=prediction_type
            )

        predictions = np.asarray(predictions)

        # Generalized prediction
        if (
            model_per_task.get("generalized_prediction") is not None
            and generalized_departement is not None
            and "departement" in X.columns
        ):
            mask = X["departement"].isin(generalized_departement).values
            if mask.any():
                if prediction_type == 'Class' or prediction_type == 'RawFormulaVal':
                    preds_gen  = self.predict_with_weight(
                        X[mask],
                        hard_or_soft=hard_or_soft,
                        weights_average=weights_average,
                        weights2use=self.weights_for_model,
                        top_model=model_per_task["generalized_prediction"],
                        prediction_type=prediction_type
                    )
                else:
                    preds_gen  = self.predict_proba_with_weights(
                        X[mask],
                        hard_or_soft=hard_or_soft,
                        weights_average=weights_average,
                        weights2use=self.weights_for_model,
                        top_model=model_per_task["generalized_prediction"],
                        prediction_type=prediction_type
                    )
                mask = np.isin(y[:, 4], generalized_departement)
                predictions[mask] = preds_gen

        for val in [2, 3, 4]:
            task_name = f"class_value_{val}_predictions"
            if task_name in model_per_task:
                if prediction_type == 'Class' or prediction_type == 'RawFormulaVal':
                    preds_cls = self.predict_with_weight(
                        X,
                        hard_or_soft=hard_or_soft,
                        weights_average=weights_average,
                        weights2use=self.weights_for_model,
                        top_model=model_per_task[task_name],
                        prediction_type=prediction_type
                    )
                    mask = (preds_cls >= val) | (predictions >= val)
                    if mask.any():
                        predictions[mask] = preds_cls[mask]
                else:
                    preds_cls = self.predict_proba_with_weights(
                        X,
                        hard_or_soft=hard_or_soft,
                        weights_average=weights_average,
                        weights2use=self.weights_for_model,
                        top_model=model_per_task[task_name],
                        prediction_type=prediction_type
                    )
                    # prendre les lignes où la classe la plus probable est >= val
                    mask = (np.argmax(preds_cls, axis=1) >= val) | (np.argmax(predictions, axis=1) >= val)
                    if mask.any():
                        predictions[mask] = preds_cls[mask]

        return predictions

    def predict(self, X, hard_or_soft='soft', weights_average='weight', top_model='all', id_col=(None, None), prediction_type="Class"):
        """
        Predict labels for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict labels for.

        Returns:
        - Aggregated predicted labels.
        """

        if weights_average not in ['None', 'weight']:
            assert id_col[0] is not None and id_col[1] is not None
            vals = id_col[1]
            unique_ids = np.unique(vals)
            prediction = np.empty(X.shape[0], dtype=int)
            y = np.empty(X.shape[0], dtype=int)
            for id in unique_ids:
                print(f'Prediction for {id_col[0]} {id}')
                mask = (id_col[1] == id)
                prediction[mask], y[mask] = self.predict_with_weight(X[mask], hard_or_soft=hard_or_soft, weights_average='weight', weights2use=self.weights_id_model[id_col[0]][id], top_model=top_model, prediction_type=prediction_type)
            return prediction, y
        else:
            return self.predict_with_weight(X, hard_or_soft=hard_or_soft, weights_average='weight', weights2use=self.weights_for_model, top_model=top_model,  prediction_type=prediction_type)

        """print(f'Predict with {hard_or_soft} and weighs at {weights_average}')
        if hard_or_soft == 'hard':
            if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(self.weights_for_model)
                models_list = models_list[key]
                models_list = models_list[-top_model:]
            else:
                key = np.arange(0, len(self.best_estimator_))

            predictions = []
            for i, estimator in enumerate(self.best_estimator_):
                if estimator.name not in models_list:
                    continue
                else:
                    pred = estimator.predict(X)
                    predictions.append(pred)

                models_to_mean.append(key[i])

            # Aggregate predictions
            aggregated_pred = self.aggregate_predictions(predictions, models_to_mean, weights_average)
            return aggregated_pred
        else:
            aggregated_pred = self.predict_proba(X, weights_average, top_model)
            predictions = np.argmax(aggregated_pred, axis=1)
            return predictions"""

    def predict_proba(self, X, weights_average='weight', top_model='all', id_col=(None, None)):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.
        
        Returns:
        - Aggregated predicted probabilities.
        """

        if weights_average not in ['None', 'weight']:
            assert id_col[0] is not None and id_col[1] is not None
            vals = id_col[1]
            unique_ids = np.unique(vals)
            prediction = np.empty(X.shape[0], dtype=int)
            y = np.empty(X.shape[0], dtype=int)
            for id in unique_ids:
                print(f'Prediction for {id_col[0]} {id}')
                mask = (id_col[1] == id)
                prediction[mask], y[mask] = self.predict_proba_with_weights(X[mask], hard_or_soft='soft', weights_average='weight', weights2use=self.weights_id_model[id_col[0]][id], top_model=top_model)
            return prediction, y

        else:
            return self.predict_proba_with_weights(X, hard_or_soft='soft', weights_average='weight', weights2use=self.weights_for_model, top_model=top_model)

        """models_list = np.asarray([estimator.name for estimator in self.best_estimator_])

        if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(self.weights_for_model)
                models_list = models_list[np.asarray(key)]
                models_list = models_list[-top_model:]
        else:
            key = np.arange(0, len(self.best_estimator_))
        
        print(models_list)
        probas = []
        models_to_mean = []
        for i, estimator in enumerate(self.best_estimator_):
            if estimator.name not in models_list:
                continue
            X_ = X
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X_)
                if proba.shape[1] != 5:
                    continue
                #print(estimator.name, np.asarray(probas).shape)
                models_to_mean.append(key[i])
                probas.append(proba)
            else:
                raise AttributeError(f"The model at index {i} does not support predict_proba.")
            
        # Aggregate probabilities
        aggregated_proba = self.aggregate_probabilities(probas, models_to_mean, weights_average)
        return aggregated_proba"""

    def aggregate_predictions(self, predictions_list, models_to_mean, weight2use=[], id_col=(None, None), prediction_type="RawFormulaVal"):
        """
        Aggregate predictions from multiple models with weights.

        Parameters:
        - predictions_list: List of predictions from each model.

        Returns:
        - Aggregated predictions.
        """
        
        if prediction_type == 'RawFormulaVal' or prediction_type == 'Probability':
            return self.aggregate_probabilities(predictions_list, models_to_mean, weight2use=weight2use, id_col=id_col)

        predictions_array = np.array(predictions_list)
        if len(weight2use) == 0 or weight2use is None:
            weight2use = np.ones_like(self.weights_for_model)[models_to_mean]

        if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
            # Weighted vote for classification
            unique_classes = np.arange(0, 5)
            weighted_votes = np.zeros((len(unique_classes), predictions_array.shape[1]))

            for i, cls in enumerate(unique_classes):
                mask = (predictions_array == cls)
                weighted_votes[i] = np.sum(mask * weight2use.reshape(mask.shape[0], 1), axis=0)

            aggregated_pred = unique_classes[np.argmax(weighted_votes, axis=0)]
        else:
            # Weighted average for regression
            weighted_sum = np.sum(predictions_array * weight2use[:, None], axis=0)
            aggregated_pred = weighted_sum / np.sum(weight2use)
            #aggregated_pred = np.max(predictions_array * weight2use[:, None], axis=0)
        
        return aggregated_pred

    """def aggregate_predictions_id(self, predictions_array, models_to_mean, id_col=(None, None)):
        assert id_col[0] is not None and id_col[1] is not None
        id = id_col[0]
        vals = id_col[1]
        uvals = np.unique(vals)

        weight2use = np.zeros((len(models_to_mean), predictions_array.shape[1]))
        for val in uvals:
            mask = (vals == val)
            weight2use[:, mask] = self.weights_id_model[id][val][models_to_mean]
            if np.all(weight2use[:, mask] == 0):
                weight2use[:, mask] = self.weights_for_model[models_to_mean]

        unique_classes = np.arange(0, 5)
        weighted_votes = np.zeros((len(unique_classes), predictions_array.shape[1]))

        for i, cls in enumerate(unique_classes):
            mask = (predictions_array == cls)
            weighted_votes[i] = np.sum(mask * weight2use, axis=0)

        aggregated_pred = unique_classes[np.argmax(weighted_votes, axis=0)]
        return aggregated_pred"""   

    def aggregate_probabilities(self, probas_list, models_to_mean, weight2use=[], id_col=(None, None)):
        """
        Aggregate probabilities from multiple models with weights.

        Parameters:
        - probas_list: List of probability predictions from each model.

        Returns:
        - Aggregated probabilities.
        """
        probas_array = np.array(probas_list)
        if weight2use is None or len(weight2use) == 0:
            weight2use = np.ones_like(self.weights_for_model)[models_to_mean]
        
        # Weighted average for probabilities
        weighted_sum = np.sum(probas_array * weight2use[:, None, None], axis=0)
        aggregated_proba = weighted_sum / np.sum(weight2use)
        #aggregated_proba = np.max(probas_array * weight2use[:, None, None], axis=0)
        return aggregated_proba

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions = self.predict(X)
        return self.score_with_prediction(predictions, y, sample_weight)
    
    def score_with_prediction(self, y_pred, y, sample_weight=None):
        
        return iou_score(y, y_pred)
    
class ModelPerID(RegressorMixin, ClassifierMixin):
    def __init__(self, model, dir_log, cluster="departement", horizon=0):
        self.base_model = model
        self.cluster_col = cluster
        self.models = {}
        self.is_fitted_ = False
        self.name = f'unique-{cluster}-{model.name}'

        self.horizon = horizon
        self.dir_log = dir_log

    def fit(self, df_train, df_val, df_test, graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params, **args):
        """Train one model per cluster value."""
        self.models = {}
        values = df_train[self.cluster_col].unique()
        for val in values:
            train_subset = df_train[df_train[self.cluster_col] == val].reset_index(drop=True)
            val_subset = df_val[df_val[self.cluster_col] == val].reset_index(drop=True)
            test_subset = df_test[df_test[self.cluster_col] == val].reset_index(drop=True)
            model = deepcopy(self.base_model)
            model.dir_log = self.base_model.dir_log / f'unique-{val}-{self.base_model.name}'
            if hasattr(model, "name"):
                model.name = f"{val}_{model.name}"
            if hasattr(model, "fit"):
                args['PATIENCE_CNT'] = PATIENCE_CNT
                args['CHECKPOINT'] = CHECKPOINT
                args['epochs'] = epochs
                args['custom_model_params'] = custom_model_params
                model.fit(train_subset, val_subset, test_subset, graph, **args)
            else:
                raise ValueError("Base model must implement fit method")
            self.models[val] = model
        self.is_fitted_ = True

    def predict_proba(self, X, **kwargs):
        proba = None
        result = []
        for val, model in self.models.items():
            mask = X[self.cluster_col] == val
            if mask.any():
                preds = model.predict_proba(X[mask], **kwargs)
                if proba is None:
                    proba = np.zeros((len(X), preds.shape[1]))
                proba[mask] = preds
        return proba

    def predict(self, X, **kwargs):
        preds = np.zeros(len(X))
        for val, model in self.models.items():
            mask = X[self.cluster_col] == val
            if mask.any():
                preds[mask] = model.predict(X[mask], **kwargs)
        return preds

    def _predict_test_loader(self, loader):
        preds_list = []
        ys_list = []
        for val, model in self.models.items():
            pred, y = model._predict_test_loader(loader)
            preds_list.append(pred)
            ys_list.append(y)
        return torch.cat(preds_list, 0), torch.cat(ys_list, 0)

