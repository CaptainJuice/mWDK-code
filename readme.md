
#### TL; DR

* **(mWDK)** We analyze how node features and graph structure influence the density distribution in embedding space. Based on these insights, we propose mWDK, a learning-free method that alleviates density imbalances through iterative kernel propagation.

* **(Performance)** Experiments show that mWDK achieves state-of-the-art performance in unsupervised node clustering. As a matrix-based approach, it is highly amenable to GPU acceleration.


### Project
```
mWDK/
│
├── main.py # Main entry script
│
├── algos/ # Graph embedding algorithms
│ ├── init.py
│ ├── algo.py # Core Algo class for WL, WDK, and mWDK
│ └── base.py # Basic operations: adjacency normalization, propagation
│
├── clustering/ # Clustering modules
│ ├── clustering_methods.py # Clustering wrappers (e.g., sc_linear, km)
│ ├── clustering_metric.py # Evaluation: ACC, NMI, F1, label matching
│ ├── DensityPeak.py # Density Peak Clustering (DPC)
│ └── IKBC.py 
│
├── datasets/ 
│ ├── cora/
│ ├── citeseer/
│ ├── pubmed/
│ └── ···
│ └── raw_data/ # e.g., .npz or .txt files
│
├── utils/ 
│ ├── data2mat.py # Convert raw data to .mat
│ ├── load_data.py
│ ├── synthetic_graph_demo.py
│ └── plot.py 
│
├── requirements.txt # Required Python dependencies
├── README.md # This file
```
## Usage
### 1. Dependencies

```bash
requirements.txt:
{
    matplotlib
    munkres
    networkx
    numpy
    scikit_learn
    scipy
}

pip install -r requirements.txt

```

### 2. Run Example
  ```
python main.py --m mWDK --dataset cora --psi 64 --t 150 --h 10 --r 1
  ```
