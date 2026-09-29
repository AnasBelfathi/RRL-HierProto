"""
Aggregate baseline vs PCM (3 strategies) vs PBR vs Mind-Your-Neighbours
results for a given task, across a given seed list, with significance
testing against baseline.

Each training run (baseline_run.py) already writes its own results.csv (via
eval_run.eval_and_save_metrics) inside its seed_<N>/ folder, but only for that
one seed. This script reads the per-seed results.csv files for each
experiment and reports mean +/- std of test weighted-F1 / macro-F1 / accuracy.
Experiments/seeds whose results.csv don't exist yet are skipped, not fatal.

Significance: seeds are matched 1:1 across experiments (same seed number ==
same init/shuffling for both runs), so each experiment vs baseline is a
paired comparison. We run a paired t-test (parametric) and a Wilcoxon
signed-rank test (non-parametric, more robust with few seeds but with very
low power below ~6 pairs) on the seed-matched weighted-F1 and macro-F1
values. With few seeds (3-5, as here) these p-values are indicative, not
proof -- report them as such, don't over-interpret a single significant p.

Run from the project root (after the jobs have completed):
    python slurms/jeanzay/pipeline/aggregate_results.py --task legal-eval-v2 --seeds 1 2 3
    python slurms/jeanzay/pipeline/aggregate_results.py --task legal-eval-v2 --seeds 1 2 3 4 5
    python slurms/jeanzay/pipeline/aggregate_results.py --task scotus-rhetorical_function --seeds 1 2 3
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


def experiments_for(task):
    return [
        ("baseline", f"output-training-{task}", "full"),
        ("PCM none (corpus-level, no doc clustering)", f"new-output-training-{task}", "none_mean_pre_concat_proj"),
        ("PCM random-clusters (control)", f"new-output-training-{task}", "random-clusters_mean_pre_concat_proj"),
        ("PCM supervised-clustering (OpenAI emb + KMeans)", f"new-output-training-{task}", "supervised-clustering_mean_pre_concat_proj"),
        ("PBR (8 prototypes, joint, euclidean)", f"pbr-output-training-{task}", "pbr"),
        ("Mind-Your-Neighbours (single prototype)", f"mind-output-training-{task}", "mind_proto"),
    ]


def collect(task, output_dir, variant, seeds):
    rows = []
    for seed in seeds:
        path = Path(output_dir) / task / variant / f"seed_{seed}" / "results.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        mean_row = df[df["task"].str.endswith("mean")].iloc[0]
        rows.append({
            "seed": seed,
            "test_weighted_f1": mean_row["test weighted-f1"],
            "test_macro_f1": mean_row["test macro-f1"],
            "test_accuracy": mean_row["test accuracy"],
        })
    return pd.DataFrame(rows)


def stars(p):
    if p is None:
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def significance(exp_df, base_df, metric):
    """Paired t-test + Wilcoxon signed-rank on seed-matched `metric` values.
    Returns (p_ttest, p_wilcoxon, n_pairs), any of which may be None if there
    aren't enough matched/non-identical pairs to run the test.
    """
    merged = exp_df[["seed", metric]].merge(base_df[["seed", metric]], on="seed", suffixes=("_exp", "_base"))
    n = len(merged)
    if n < 2:
        return None, None, n
    a, b = merged[f"{metric}_exp"], merged[f"{metric}_base"]
    p_t = stats.ttest_rel(a, b).pvalue if n >= 2 and not (a == b).all() else (1.0 if (a == b).all() else None)
    try:
        p_w = stats.wilcoxon(a, b).pvalue if n >= 2 and not (a == b).all() else (1.0 if (a == b).all() else None)
    except ValueError:
        p_w = None  # e.g. all differences are zero, or n too small for wilcoxon
    return p_t, p_w, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="legal-eval-v2")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()

    print(f"### task={args.task}  seeds={args.seeds}")

    summaries = {}
    for label, output_dir, variant in experiments_for(args.task):
        df = collect(args.task, output_dir, variant, args.seeds)
        if df.empty:
            print(f"\n=== {label} === (pas encore de résultats, ignoré)")
            continue
        summaries[label] = df
        print(f"\n=== {label} === ({len(df)}/{len(args.seeds)} seeds)")
        print(df.to_string(index=False))
        print(
            f"test weighted-F1 = {df['test_weighted_f1'].mean():.2f} +/- {df['test_weighted_f1'].std():.2f}   "
            f"test macro-F1 = {df['test_macro_f1'].mean():.2f} +/- {df['test_macro_f1'].std():.2f}   "
            f"test accuracy = {df['test_accuracy'].mean():.2f} +/- {df['test_accuracy'].std():.2f}"
        )

    if "baseline" in summaries:
        base_df = summaries["baseline"]
        base_wf1 = base_df["test_weighted_f1"].mean()
        base_mf1 = base_df["test_macro_f1"].mean()
        print("\n--- Delta vs baseline (+ significance, paired t-test / Wilcoxon on matched seeds) ---")
        for label, _, _ in experiments_for(args.task)[1:]:
            if label not in summaries:
                continue
            exp_df = summaries[label]
            wf1 = exp_df["test_weighted_f1"].mean()
            mf1 = exp_df["test_macro_f1"].mean()

            pt_w, pw_w, n_w = significance(exp_df, base_df, "test_weighted_f1")
            pt_m, pw_m, n_m = significance(exp_df, base_df, "test_macro_f1")

            def fmt(p):
                return f"p={p:.3f}{stars(p)}" if p is not None else "p=n/a"

            print(
                f"{label} (n={n_w} paired seeds):\n"
                f"  weighted-F1 {wf1 - base_wf1:+.2f}   t-test {fmt(pt_w)}   wilcoxon {fmt(pw_w)}\n"
                f"  macro-F1    {mf1 - base_mf1:+.2f}   t-test {fmt(pt_m)}   wilcoxon {fmt(pw_m)}"
            )
        print("\n(* p<0.05, ** p<0.01 -- indicative only with this few seeds, not proof)")


if __name__ == "__main__":
    main()
