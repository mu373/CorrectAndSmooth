import argparse
import glob
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from custom_dataset import CustomDataset, get_prefix
from custom_evaluator import CustomEvaluator
from custom_presets import get_preset, get_fn_name
from outcome_correlation import *
from results_logger import save_run_experiments_result
from normalizers import DegreeNormalizer, PageRankNormalizer, DegreePageRankNormalizer
from adjacency import (
    StandardAdjacency,
    TwoHopAdjacency,
    SignlessLaplacian,
    Laplacian,
    KatzAdjacency,
)
from cache import MatrixCache


def create_normalizer(args):
    """Create normalizer from CLI arguments."""
    if args.normalizer == "degree":
        return DegreeNormalizer()
    elif args.normalizer == "pagerank":
        return PageRankNormalizer(damping=args.pagerank_damping)
    elif args.normalizer == "degree_pagerank":
        print(
            f"Creating DegreePageRankNormalizer with alpha={args.degree_pagerank_alpha}, damping={args.pagerank_damping}"
        )
        return DegreePageRankNormalizer(
            alpha=args.degree_pagerank_alpha, damping=args.pagerank_damping
        )
    else:
        raise ValueError(f"Unknown normalizer: {args.normalizer}")


def create_adjacency(args):
    """Create adjacency transformer from CLI arguments."""
    if args.adjacency == "standard":
        return StandardAdjacency()
    elif args.adjacency == "2hop":
        return TwoHopAdjacency()
    elif args.adjacency == "signless_laplacian":
        return SignlessLaplacian()
    elif args.adjacency == "laplacian":
        return Laplacian()
    elif args.adjacency == "katz":
        return KatzAdjacency(beta=args.katz_beta, k=args.katz_k)
    else:
        raise ValueError(f"Unknown adjacency type: {args.adjacency}")


def main():
    parser = argparse.ArgumentParser(description="Outcome Correlations)")
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--dataname",
        type=str,
        default=None,
        help="Dataset name for custom datasets (e.g., ba001)",
    )
    parser.add_argument("--method", type=str)

    # Node weighting (normalization)
    parser.add_argument(
        "--normalizer",
        type=str,
        default="degree",
        choices=["degree", "pagerank", "degree_pagerank"],
        help="Node weighting strategy (default: degree)",
    )
    parser.add_argument(
        "--pagerank-damping",
        type=float,
        default=0.85,
        help="Damping factor for PageRank normalizer (default: 0.85)",
    )
    parser.add_argument(
        "--degree-pagerank-alpha",
        type=float,
        default=0.5,
        help="Weight for degree in degree_pagerank normalizer (default: 0.5)",
    )

    # Adjacency type
    parser.add_argument(
        "--adjacency",
        type=str,
        default="standard",
        choices=["standard", "2hop", "signless_laplacian", "laplacian", "katz"],
        help="Adjacency matrix type (default: standard)",
    )
    parser.add_argument(
        "--katz-beta",
        type=float,
        default=0.1,
        help="Beta decay factor for Katz adjacency (default: 0.1)",
    )
    parser.add_argument(
        "--katz-k",
        type=int,
        default=3,
        help="Number of hops for Katz adjacency (default: 3)",
    )

    # Caching
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching matrices (default: None, no caching)",
    )

    # Normalization style
    parser.add_argument(
        "--norm-style",
        type=str,
        default="symmetric",
        choices=["symmetric", "left", "right"],
        help="Normalization style: symmetric (D^{-1/2} A D^{-1/2}), left (D^{-1/2} A), right (A D^{-1/2})",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.dataset == "custom" and args.dataname is None:
        parser.error("--dataname is required when --dataset is 'custom'")

    # Create normalizer and adjacency transformer
    normalizer = create_normalizer(args)
    adjacency = create_adjacency(args)

    # Create cache if specified
    cache = MatrixCache(args.cache_dir) if args.cache_dir else None

    print(f"Using normalizer: {normalizer.name}")
    print(f"Using adjacency: {adjacency.name}")
    print(f"Using norm-style: {args.norm_style}")
    if cache:
        print(f"Using cache: {args.cache_dir}")

    # Load dataset
    if args.dataset == "custom":
        dataset = CustomDataset(args.dataname)
        evaluator = CustomEvaluator()
        prefix = get_prefix(args.dataname)
        print(f"Using custom dataset: {args.dataname} (category: {prefix})")
    else:
        dataset = PygNodePropPredDataset(name=f"ogbn-{args.dataset}")
        evaluator = Evaluator(name=f"ogbn-{args.dataset}")

    data = dataset[0]

    adj, norm_result = process_adj(
        data, normalizer=normalizer, adjacency=adjacency, cache=cache
    )

    # Generate matrix with the specified normalization style
    A_styled = gen_normalized_adj_by_style(adj, norm_result, style=args.norm_style)

    split_idx = dataset.get_idx_split()

    def eval_test(result, idx=split_idx["test"]):
        return evaluator.eval(
            {
                "y_true": data.y[idx],
                "y_pred": result[idx].argmax(dim=-1, keepdim=True),
            }
        )["acc"]

    if args.dataset == "arxiv":
        lp_dict = {
            "idxs": ["train"],
            "alpha": 0.9,
            "num_propagations": 50,
            "A": A_styled,
        }
        plain_dict = {
            "train_only": True,
            "alpha1": 0.87,
            "A1": A_styled,
            "num_propagations1": 50,
            "alpha2": 0.81,
            "A2": A_styled,
            "num_propagations2": 50,
            "display": False,
        }
        plain_fn = double_correlation_autoscale

        """
        If you tune hyperparameters on test set
        {'alpha1': 0.9988673963255859, 'alpha2': 0.7942279952481052, 'A1': 'DA', 'A2': 'AD'}
        gets you to 72.64
        """
        linear_dict = {
            "train_only": True,
            "alpha1": 0.98,
            "alpha2": 0.65,
            "A1": A_styled,
            "A2": A_styled,
            "num_propagations1": 50,
            "num_propagations2": 50,
            "display": False,
        }
        linear_fn = double_correlation_autoscale

        """
        If you tune hyperparameters on test set
        {'alpha1': 0.9956668128133523, 'alpha2': 0.8542393515434346, 'A1': 'DA', 'A2': 'AD'}
        gets you to 73.35
        """
        mlp_dict = {
            "train_only": True,
            "alpha1": 0.9791632871592579,
            "alpha2": 0.7564990804200602,
            "A1": A_styled,
            "A2": A_styled,
            "num_propagations1": 50,
            "num_propagations2": 50,
            "display": False,
        }
        mlp_fn = double_correlation_autoscale

        gat_dict = {
            "labels": ["train"],
            "alpha": 0.8,
            "A": A_styled,
            "num_propagations": 50,
            "display": False,
        }
        gat_fn = only_outcome_correlation

    elif args.dataset == "products":
        lp_dict = {
            "idxs": ["train"],
            "alpha": 0.5,
            "num_propagations": 50,
            "A": A_styled,
        }

        plain_dict = {
            "train_only": True,
            "alpha1": 1.0,
            "alpha2": 0.9,
            "scale": 20.0,
            "A1": A_styled,
            "A2": A_styled,
            "num_propagations1": 50,
            "num_propagations2": 50,
        }
        plain_fn = double_correlation_fixed

        linear_dict = {
            "train_only": True,
            "alpha1": 1.0,
            "alpha2": 0.9,
            "scale": 20.0,
            "A1": A_styled,
            "A2": A_styled,
            "num_propagations1": 50,
            "num_propagations2": 50,
        }
        linear_fn = double_correlation_fixed

        mlp_dict = {
            "train_only": True,
            "alpha1": 1.0,
            "alpha2": 0.8,
            "scale": 10.0,
            "A1": A_styled,
            "A2": A_styled,
            "num_propagations1": 50,
            "num_propagations2": 50,
        }
        mlp_fn = double_correlation_fixed

    elif args.dataset == "custom":
        # Get presets based on dataset category (prefix)
        preset = get_preset(prefix)
        fn_name = get_fn_name(prefix)

        # Select correlation function
        if fn_name == "double_correlation_fixed":
            correlation_fn = double_correlation_fixed
        else:
            correlation_fn = double_correlation_autoscale

        lp_preset = preset.get("lp", {})
        lp_dict = {
            "idxs": ["train"],
            "alpha": lp_preset.get("alpha", 0.9),
            "num_propagations": lp_preset.get("num_propagations", 50),
            "A": A_styled,
        }

        plain_preset = preset.get("plain", {})
        plain_dict = {
            "train_only": True,
            "alpha1": plain_preset.get("alpha1", 0.87),
            "alpha2": plain_preset.get("alpha2", 0.81),
            "A1": A_styled,
            "A2": A_styled,
            "num_propagations1": plain_preset.get("num_propagations1", 50),
            "num_propagations2": plain_preset.get("num_propagations2", 50),
            "display": False,
        }
        plain_fn = correlation_fn

        linear_preset = preset.get("linear", {})
        linear_dict = {
            "train_only": True,
            "alpha1": linear_preset.get("alpha1", 0.98),
            "alpha2": linear_preset.get("alpha2", 0.65),
            "A1": A_styled,
            "A2": A_styled,
            "num_propagations1": linear_preset.get("num_propagations1", 50),
            "num_propagations2": linear_preset.get("num_propagations2", 50),
            "display": False,
        }
        linear_fn = correlation_fn

        mlp_preset = preset.get("mlp", {})
        mlp_dict = {
            "train_only": True,
            "alpha1": mlp_preset.get("alpha1", 0.979),
            "alpha2": mlp_preset.get("alpha2", 0.756),
            "A1": A_styled,
            "A2": A_styled,
            "num_propagations1": mlp_preset.get("num_propagations1", 50),
            "num_propagations2": mlp_preset.get("num_propagations2", 50),
            "display": False,
        }
        mlp_fn = correlation_fn

        gat_preset = preset.get("gat", {})
        gat_dict = {
            "labels": ["train"],
            "alpha": gat_preset.get("alpha", 0.8),
            "A": A_styled,
            "num_propagations": gat_preset.get("num_propagations", 50),
            "display": False,
        }
        gat_fn = only_outcome_correlation

    # Model outputs path
    if args.dataset == "custom":
        model_outs = glob.glob(f"models/{prefix}/{args.dataname}-{args.method}/*.pt")
    else:
        model_outs = glob.glob(f"models/{args.dataset}_{args.method}/*.pt")

    if args.method == "lp":
        out = label_propagation(data, split_idx, **lp_dict)
        print("Valid acc: ", eval_test(out, split_idx["valid"]))
        print("Test acc:", eval_test(out, split_idx["test"]))
        return

    orig_results = get_orig_acc(data, eval_test, model_outs, split_idx)

    # Get C&S results
    if args.method == "plain":
        _, cs_results = evaluate_params(
            data, eval_test, model_outs, split_idx, plain_dict, fn=plain_fn
        )
    elif args.method == "linear":
        _, cs_results = evaluate_params(
            data, eval_test, model_outs, split_idx, linear_dict, fn=linear_fn
        )
    elif args.method == "mlp":
        _, cs_results = evaluate_params(
            data, eval_test, model_outs, split_idx, mlp_dict, fn=mlp_fn
        )
    elif args.method == "gat":
        _, cs_results = evaluate_params(
            data, eval_test, model_outs, split_idx, gat_dict, fn=gat_fn
        )

    # Save results to CSV
    dataname = args.dataname if args.dataset == "custom" else args.dataset
    # Match orig and cs results by run number
    orig_by_run = {r[0]: (r[1], r[2]) for r in orig_results}
    for run, cs_valid, cs_test in cs_results:
        orig_valid, orig_test = orig_by_run.get(run, (0, 0))
        save_run_experiments_result(
            dataname, args, run, orig_valid, orig_test, cs_valid, cs_test
        )


#     name = f'{args.experiment}_{args.search_type}_{args.model_dir}'
#     setup_experiments(data, eval_test, model_outs, split_idx, normalized_adjs, args.experiment, args.search_type, name, num_iters=300)

#     return


if __name__ == "__main__":
    main()
