import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from torch_geometric.explain import Explainer, GNNExplainer

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class AtomEncoder2(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder2, self).__init__()
        print("Using this encoder!!!!!!!!")
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        # print('in atomEncoder2')
        # print('x.shape', x.shape)
        for i in range(x.shape[1]):
            # print('x_embedding before:')
            # print(x_embedding)

            x_embedding += self.atom_embedding_list[i](x[:,i])

            # print('x:')
            # print(x[:, i][0:3], x[:, i].shape)

            # print('+')
            # add_list = self.atom_embedding_list[i](x[:,i])
            # print(add_list[0][0:3])

            # print('x_embedding after +=')
            # print(x_embedding[0][0:3], x_embedding.shape)

        return x_embedding

def train(model, atom_encoder, device, loader, optimizer, task_type):
    model.train()
    atom_encoder.train()
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        x, edge_index, edge_attr, batched_data = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            node_features = atom_encoder(x)
            pred = model(node_features, edge_index, batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, atom_encoder, device, loader, evaluator):
    model.eval()
    atom_encoder.eval()
    y_true = []
    y_pred = []
    
    
    

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        #print("the value of the baatch is:", batch)
        x, edge_index, edge_attr, batched_data = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        #atom_encoder = AtomEncoder2(100)
        #atom_encoder = atom_encoder.to(device)
        node_features = atom_encoder(x)
        #node_features = node_features.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            
            #explanation = explainer( x = node_features.detach(), edge_index = edge_index, batched_data = batch.detach())
            # print(explanation.edge_mask)
            # print(explanation.node_mask)
            pred = model(node_features, edge_index, batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def eval_with_explainer(model, atom_encoder, device, loader, evaluator):
    model.eval()
    atom_encoder.eval()
    y_true = []
    y_pred = []
    
    
    explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='log_probs',  # Model returns log probabilities.
    ),
)

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        #print("the value of the baatch is:", batch)
        x, edge_index, edge_attr, batched_data = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        #atom_encoder = AtomEncoder2(100)
        #atom_encoder = atom_encoder.to(device)
        node_features = atom_encoder(x)
        #node_features = node_features.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            
            explanation = explainer( x = node_features.detach(), edge_index = edge_index, batched_data = batch.detach())
            # print(explanation.edge_mask)
            # print(explanation.node_mask)
            pred = model(node_features, edge_index, batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=100,
                        help='dimensionality of hidden units in GNNs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument("--use_explainer", choices=['true', 'false'], default = 'false',help="Set the explainer option (true/false)")
    
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    if (args.use_explainer == 'true'):
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=1, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=1, shuffle=False, num_workers = args.num_workers)
    else:
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    #model = torch.load('gin_ep1.pt')
    atom_encoder = AtomEncoder2(100).to(device)
    
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, atom_encoder, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        #train_perf = eval(model, atom_encoder, device, train_loader, evaluator)
        if (args.use_explainer == 'true'):
            valid_perf = eval_with_explainer(model, atom_encoder,device, valid_loader, evaluator)
            test_perf = eval_with_explainer(model, atom_encoder, device, test_loader, evaluator)
        else:
            
            valid_perf = eval(model, atom_encoder,device, valid_loader, evaluator)
            test_perf = eval(model, atom_encoder, device, test_loader, evaluator)

        #print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        print({ 'Validation': valid_perf, 'Test': test_perf})
        #train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        torch.save(atom_encoder.state_dict(), './GIN_ogbgmol_weights_new/%s_ep%d_dim%d_atom_encoder.pt' % (args.gnn, epoch, args.emb_dim))
        torch.save(model.state_dict(), './GIN_ogbgmol_weights_new/%s_ep%d_dim%d.pt' % (args.gnn, epoch, args.emb_dim))

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        #best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        #best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))



if __name__ == "__main__":
    main()

