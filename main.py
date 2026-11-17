import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'clustering'))
sys.path.append(os.path.join(project_root, 'utils'))
import numpy as np
import time
from utils import load_data
import argparse
import warnings
from algos.algo import Algo
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

warnings.filterwarnings('ignore')


def runner(dataset, method, psi, t, h, path, repetition):
    """Run embedding & clustering."""
    logger.info("[Loading dataset]: {}".format(dataset))
    try:
        adj_mat, node_features, true_labels = load_data(path, dataset)
    except Exception as e:
        logger.error("Failed to load dataset {}: {}".format(dataset, str(e)))
        return

    num_of_clusters = len(np.unique(true_labels))
    time_start = time.perf_counter()
    algo = Algo(dataset=dataset, algorithm=method, psi=psi, t=t, h=h, repetition=repetition)
    embeddings = algo.get_embedding(adj_mat, node_features)

    results = algo.do_clustering(
        embeddings=embeddings,
        num_of_clusters=num_of_clusters,
        true_labels=true_labels,
        repetition=repetition
    )
    total_time = time.perf_counter() - time_start
    logger.info("{} experiment done. Total time: {:.4f}s".format(method, total_time))

    return results


if __name__ == '__main__':
    default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets"))

    parser = argparse.ArgumentParser(description="Run WDK/mWDK/WL experiments")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name (e.g., cora, citeseer)")
    parser.add_argument("--m", type=str, default='mWDK', choices=['WL', 'WDK', 'mWDK'], help="Embedding method")
    parser.add_argument("--psi", type=int, default=64, help="param of IK")
    parser.add_argument("--t", type=int, default=150, help="param of IK")
    parser.add_argument("--h", type=int, default=7, help="Number of iterations")
    parser.add_argument("--path", type=str, default=default_path, help="Path to datasets")
    parser.add_argument("--r", type=int, default=1, help="Number of repetitions")

    args = parser.parse_args()

    runner(
        dataset=args.dataset,
        method=args.m,
        psi=args.psi,
        t=args.t,
        h=args.h,
        path=args.path,
        repetition=args.r
    )
