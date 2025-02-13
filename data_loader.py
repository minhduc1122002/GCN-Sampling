# code from tgn/preprocess_data.py
import numpy as np
import random
import pandas as pd

class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes) + 1

def get_data_node_classification(dataset_name, use_validation=False):
    # Load CSV and feature files from ./data/
    graph_df = pd.read_csv(f'./data/{dataset_name}/ml_{dataset_name}.csv')
    edge_features = np.load(f'./data/{dataset_name}/ml_{dataset_name}.npy')
    node_features = np.load(f'./data/{dataset_name}/ml_{dataset_name}_node.npy')
    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values
    random.seed(2020)
    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])
    return full_data, node_features, edge_features, train_data, val_data, test_data

def get_data(dataset_name, val_ratio, test_ratio, different_new_nodes_between_val_and_test=False,
             randomize_features=False):
    # This function is used for link prediction.
    graph_df = pd.read_csv(f'./data/{dataset_name}/ml_{dataset_name}.csv')
    edge_features = np.load(f'./data/{dataset_name}/ml_{dataset_name}.npy')
    node_features = np.load(f'./data/{dataset_name}/ml_{dataset_name}_node.npy')
    
    # For some datasets, pad features to fixed dimension
    if dataset_name in ['enron', 'socialevolve', 'uci']:
        node_zero_padding = np.zeros((node_features.shape[0], 172 - node_features.shape[1]))
        node_features = np.concatenate([node_features, node_zero_padding], axis=1)
        edge_zero_padding = np.zeros((edge_features.shape[0], 172 - edge_features.shape[1]))
        edge_features = np.concatenate([edge_features, edge_zero_padding], axis=1)
    if randomize_features:
        node_features = np.random.rand(node_features.shape[0], node_features.shape[1])
    # Split by time using provided ratios.
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    random.seed(2020)
    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)
    test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))

    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    train_node_set = set(train_data.sources) | set(train_data.destinations)
    new_node_set = node_set - train_node_set
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time
    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])
        edge_contains_new_val_node_mask = np.array([(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array([(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
    else:
        edge_contains_new_node_mask = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask], timestamps[new_node_val_mask],
                             edge_idxs[new_node_val_mask], labels[new_node_val_mask])
    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask], timestamps[new_node_test_mask],
                              edge_idxs[new_node_test_mask], labels[new_node_test_mask])
    print("Full dataset: {} interactions, {} nodes".format(full_data.n_interactions, full_data.n_unique_nodes))
    print("Train: {} interactions, {} nodes".format(train_data.n_interactions, train_data.n_unique_nodes))
    print("Validation: {} interactions, {} nodes".format(val_data.n_interactions, val_data.n_unique_nodes))
    print("Test: {} interactions, {} nodes".format(test_data.n_interactions, test_data.n_unique_nodes))
    print("New node Val: {} interactions, {} nodes".format(new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    print("New node Test: {} interactions, {} nodes".format(new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
    print("{} nodes used for inductive testing (never seen during training)".format(len(new_test_node_set)))
    return node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data

def main():
    import sys
    parser = argparse.ArgumentParser(description="Unified Data Loader for Link Prediction")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., mooc, reddit, wiki, uci, socialevo, enron)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--different_new_nodes", action="store_true", help="Use different new nodes between val and test")
    parser.add_argument("--randomize_features", action="store_true", help="Randomize node features")
    parser.add_argument("--in_dim", type=int, default=16, help="Input feature dimension")
    args = parser.parse_args()
    
    node_features, edge_features, full_data, train_data, val_data, test_data, new_val, new_test = get_data(
        args.dataset_name, args.val_ratio, args.test_ratio, different_new_nodes_between_val_and_test=args.different_new_nodes,
        randomize_features=args.randomize_features
    )
    print("Data loaded for dataset:", args.dataset_name)
    print("Full interactions:", full_data.n_interactions, "Unique nodes:", full_data.n_unique_nodes)

if __name__ == "__main__":
    main()
