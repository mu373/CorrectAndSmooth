# Correct and Smooth (C&S) OGB submissions

Paper: https://arxiv.org/abs/2010.13993

This directory contains OGB submissions. All hyperparameters were tuned on the validation set with optuna, except for products, which was hand tuned. All experiments were run with a RTX 2080 TI with 11GB.

## Some Tips 
- In general, the more complex and "smooth" your GNN is, the less likely it'll be that applying the "Correct" portion helps performance. In those cases, you may consider just applying the "smooth" portion, like we do on the GAT. In almost all cases, applying the "smoothing" component will improve performance. For Linear/MLP models, applying the "Correct" portion is almost always essential for obtaining good performance.

- In a similar vein, an improvement of performance of your model may not correspond to an improvement after applying C&S. Considering that C&S learns no parameters over your data, our intuition is that C&S "levels" the playing field, allowing models that learn interesting features to shine (as opposed to learning how to be smooth).
     - Even though GAT (73.57) is outperformed by GAT + labels (73.65), when we apply C&S, we see that GAT + C&S (73.86) performs better than GAT + labels + C&S (~73.70) , 
     - Even though a 6 layer GCN performs on par with a 2 layer GCN with Node2Vec features, C&S improves performance of the 2 layer GCN with Node2Vec features substantially more.
     - Even though MLP + Node2Vec outperforms MLP + Spectral in both arxiv and products, the performance ordering flips after we apply C&S.
     - On Products, the MLP (74%) is substantially outperformed by ClusterGCN (80%). However, MLP + C&S (84.1%) substantially outperforms ClusterGCN + C&S (82.4%).

- In general, autoscale works more reliably than fixedscale, even though fixedscale may make more sense...

## Setup

### Environments

**Paperspace gradient**
Setup a notebook inside Paperspace gradient with PyTorch 1.12 template.
- Image (default): `paperspace/gradient-base:pt211-tf215-cudatk120-py311-20240202`

**Runpod**
- GPU: A40 (VRAM 48GB)
- Template: Runpod Pytorch 2.8.0 (runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404)
- https://console.runpod.io/deploy?gpu=A40&count=1&template=runpod-torch-v280


### Installation/Running
```sh
git clone --branch pytorch https://github.com/mu373/CorrectAndSmooth.git
cd CorrectAndSmooth
mkdir -p embeddings

# Install requirements
pip install -r requirements.txt
# pip install -r requirements.lock

# For data loading
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

## Arxiv

### Label Propagation (0 params):
```
python run_experiments.py --dataset arxiv --method lp

Valid acc: 0.7013658176448874
Test acc: 0.6832294302820814
```

### Plain Linear + C&S (5160 params, 52.5% base accuracy)
```
python gen_models.py --dataset arxiv --model plain --epochs 1000    
python run_experiments.py --dataset arxiv --method plain

Valid acc -> Test acc
Args []: 73.00 ± 0.01 -> 71.26 ± 0.01
```

### Linear + C&S (15400 params, 70.11% base accuracy)
```
python gen_models.py --dataset arxiv --model linear --use_embeddings --epochs 1000 
python run_experiments.py --dataset arxiv --method linear

Valid acc -> Test acc
Args []: 73.68 ± 0.04 -> 72.22 ± 0.02;
```

### MLP + C&S (175656 params, 71.44% base accuracy)
```
python gen_models.py --dataset arxiv --model mlp --use_embeddings
python run_experiments.py --dataset arxiv --method mlp

Valid acc -> Test acc
Args []: 73.91 ± 0.15 -> 73.12 ± 0.12
```

### GAT + C&S (1567000 params, 73.56% base accuracy)
```
python gat/gat.py --use-norm
python run_experiments.py --dataset arxiv --method gat

Valid acc -> Test acc
Args []: 74.84 ± 0.07 -> 73.86 ± 0.14
```

### Notes
As opposed to the paper's results, which only use spectral embeddings, here we use spectral *and* diffusion embeddings, which we find improves Arxiv performance.

## Custom Datasets

You can run C&S on custom graph data using `--dataset custom --dataname <name>`.

### Data Format
```
dataset/{prefix}/{dataname}/
├── edge.csv(.gz)       # source,target pairs (no header)
├── node-feat.csv(.gz)  # node features, comma-separated
├── node-label.csv(.gz) # one integer label per line
└── split/              # optional (auto-generated 60/20/20 if missing)
    ├── train.csv(.gz)
    ├── valid.csv(.gz)
    └── test.csv(.gz)
```

Prefix is extracted from dataname: `ba001` → `ba`, `ws001` → `ws`.

### Output Paths
- Models: `models/{prefix}/{dataname}-{model}/`
- Embeddings: `embeddings/{prefix}/{dataname}-spectral.pt`
- Results: `results/gen_models.csv`, `results/run_experiments.csv`

### Results CSV Format
One row per run, appended across experiments.

**gen_models.csv** (base model training):
```
dataname,model,epochs,hidden_channels,use_embeddings,run,train,valid,test
ba001,mlp,300,256,True,0,0.923,0.861,0.840
```

**run_experiments.csv** (C&S post-processing):
```
dataname,method,normalizer,adjacency,norm_style,run,orig_valid,orig_test,cs_valid,cs_test
ba001,mlp,degree,standard,symmetric,0,0.861,0.840,0.883,0.869
```

### Example
```bash
# Train MLP
python gen_models.py --dataset custom --dataname ba001 --model mlp --epochs 300

# With spectral embeddings
python gen_models.py --dataset custom --dataname ba001 --model mlp --epochs 300 --use_embeddings

# Run C&S
python run_experiments.py --dataset custom --dataname ba001 --method mlp
```

## BA

Evaluate the effect of hubs in the dataset. Vary `m` to control degree heterogeneity (hubbiness).

```bash
python generate_synthetic_graph_ba.py --n_nodes 10000 --labeling louvain --n_classes 40 --sigma 5 --m 1
python gen_models.py --dataset custom --model mlp --epochs 300 --dataname ba001
python run_experiments.py --dataset custom --method mlp --dataname ba001
```

## WS

Evaluate small-world rewiring and contiguous ring sectors. Vary `p` to control clustering coefficient.

```bash
python generate_synthetic_graph_ws.py --n_nodes 10000 --k 10 --p 0.1 --labeling equal --n_classes 40 --feature_noise_sigma 5
python gen_models.py --dataset custom --model mlp --epochs 300 --dataname ws001
python run_experiments.py --dataset custom --method mlp --dataname ws001
```

## Aggregating Metadata

Combine metadata.json files from multiple datasets into a single CSV for analysis:

```bash
python aggregate_dataset_metadata.py --datatype ba
python aggregate_dataset_metadata.py --datatype ws
```

Output: `{datatype}_metadata.csv` containing graph properties (n_nodes, n_edges, avg_clustering, centrality metrics, etc.) for all datasets of that type.

## Products

### Label Propagation (0 params):
```
python run_experiments.py --dataset products --method lp 

Valid acc:  0.9090608549703736
Test acc: 0.7434145274640762
```

### Plain Linear + C&S (4747 params, 47.73% base accuracy)
```
python gen_models.py --dataset products --model plain --epochs 1000 --lr 0.1
python run_experiments.py --dataset products --method plain

Valid acc -> Test acc
Args []: 91.03 ± 0.01 -> 82.54 ± 0.03
```

### Linear + C&S (10763 params, 50.05% base accuracy)
```
python gen_models.py --dataset products --model linear --use_embeddings --epochs 1000 --lr 0.1
python run_experiments.py --dataset products --method linear

Valid acc -> Test acc
Args []: 91.34 ± 0.01 -> 83.01 ± 0.01
```

### MLP + C&S (96247 params, 63.41% base accuracy)
```
python gen_models.py --dataset products --model mlp --hidden_channels 200 --use_embeddings
python run_experiments.py --dataset products --method mlp

Valid acc -> Test acc
Args []: 91.47 ± 0.09 -> 84.18 ± 0.07
```
