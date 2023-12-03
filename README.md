# CSE8803

Run this command if you want to use explainer after training finishes

python main_pyg.py --gnn gcn --use_explainer true

If you do not want to use explainer, then rrun

python main_pyg.py --gnn gcn --use_explainer false


## The updated version of GNN fault injection framework:

In the folder Golden_GCN_v2

If you want to train the GNN, then run this command:

python main_pyg.py --gnn gcn --use_explainer false

If you want to train the GIN, then run this command:

python main_pyg.py --gnn gin --use_explainer false

If you want to test the training results for GCN:

python main_pyg_test.py --gnn gcn --use_explainer false

For GIN:

python main_pyg_test.py --gnn gin --use_explainer false

If you want to save the explanattion results for GCN:

python main_pyg_bit_flipping.py --gnn gcn --use_explainer true

For GIN:

python main_pyg_bit_flipping.py --gnn gin --use_explainer true

If you want to insert faults for GCN:

python main_pyg_validate.py --gnn gcn --use_explainer false

For GIN:

python main_pyg_validate.py --gnn gin --use_explainer false


If want to insert faults to GNN hidden layers, go into folder Insert_faults_to_hidden_layers_code,

and then run python main_pyg_validate.py --gnn gcn --use_explainer false for GCN and python main_pyg_validate.py --gnn gin --use_explainer false for GIN with correct configurations. 

