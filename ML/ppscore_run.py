#!/usr/bin/env python3
"""ppscore_run.py

Run Predictive Power Score (PPS) on a real dataset and save results.

Default dataset: ../Datasets/hotel_bookings.csv (relative to the repo root)

Usage examples:
    python ppscore_run.py <target> [data_path]
    python ppscore_run.py is_canceled
    python ppscore_run.py "is_canceled" "C:/path/to/file.csv"
"""

from pathlib import Path
import sys
import os

# Try to import rich for better output globally
try:
    from rich import print as rich_print
    from rich.console import Console
    console = Console()
except ImportError:
    rich_print = print
    console = None


def find_default_dataset():
    # file is placed in ML/, so parent of parent is repo root; Datasets is under repo root
    this = Path(__file__).resolve()
    repo_root = this.parents[1]
    candidate = repo_root / "Datasets" / "hotel_bookings.csv"
    return candidate


def ensure_imports():
    try:
        import pandas as pd  # noqa: F401
    except Exception as e:
        print("Missing dependency: pandas. Install with: pip install pandas")
        raise
    try:
        import ppscore as pps  # noqa: F401
    except Exception:
        print("Missing dependency: ppscore. Install with: pip install -U ppscore")
        raise


def main():
    """
    Simple runner without argparse.

    Usage (positional overrides):
        python ppscore_run.py <target> [data_path]

    Or edit the defaults below or set environment variables:
        PPS_DATA, PPS_TARGET, PPS_SAMPLE, PPS_SEED, PPS_PLOT, PPS_TOP, PPS_OUTPUT
    """

    # Defaults (edit here or use environment variables / positional args)
    DEFAULT_TARGET = os.environ.get("PPS_TARGET", None)
    DEFAULT_DATA = os.environ.get("PPS_DATA", None)
    DEFAULT_SAMPLE = os.environ.get("PPS_SAMPLE", None)
    DEFAULT_SEED = int(os.environ.get("PPS_SEED", 123))
    DEFAULT_PLOT = os.environ.get("PPS_PLOT", "0") in ("1", "true", "True")
    DEFAULT_TOP = int(os.environ.get("PPS_TOP", 20))
    DEFAULT_OUTPUT = os.environ.get("PPS_OUTPUT", None)

    # positional args: target [data_path]
    target = None
    data_path_arg = None
    if len(sys.argv) > 1:
        target = sys.argv[1]
    if len(sys.argv) > 2:
        data_path_arg = sys.argv[2]

    target = target or DEFAULT_TARGET

    if data_path_arg:
        data_path = Path(data_path_arg)
    elif DEFAULT_DATA:
        data_path = Path(DEFAULT_DATA)
    else:
        data_path = find_default_dataset()

    sample = int(DEFAULT_SAMPLE) if DEFAULT_SAMPLE else None
    seed = DEFAULT_SEED
    do_plot = DEFAULT_PLOT
    top = DEFAULT_TOP
    out_path = Path(DEFAULT_OUTPUT) if DEFAULT_OUTPUT else None

    if not data_path.exists():
        rich_print(f"[bold red]Dataset not found:[/] {data_path}")
        rich_print("[yellow]Either provide a data file path as 2nd positional arg or set environment variable PPS_DATA.[/]")
        sys.exit(1)

    # imports (fail with a friendly message)
    try:
        import pandas as pd
        import ppscore as pps
    except Exception:
        rich_print("[bold red]One or more Python packages are missing. Please install requirements:[/]")
        rich_print("[yellow]  pip install pandas ppscore rich[/]")
        sys.exit(2)

    rich_print(f"[cyan]Loading dataset:[/] {data_path} (may take a moment) ...")
    df = pd.read_csv(data_path)
    rich_print(f"[green]Data shape:[/] {df.shape}")

    if sample and sample > 0 and sample < len(df):
        rich_print(f"[cyan]Sampling {sample} rows (seed={seed}) ...[/]")
        df = df.sample(n=sample, random_state=seed)

    if target not in df.columns:
        rich_print(f"[yellow]Target column '{target}' not found in dataset columns.[/]")
        # try common target column names
        candidates = ["is_canceled", "is_cancelled", "canceled", "cancelled", "booking_status", "status"]
        picked = None
        for c in candidates:
            if c in df.columns:
                picked = c
                rich_print(f"[green]Detected plausible target column '{c}' and will use it.[/]")
                target = c
                break

        if picked is None:
            # fallback: pick a low-cardinality non-ID column
            for col in df.columns:
                if "id" in col.lower():
                    continue
                try:
                    nunique = df[col].nunique(dropna=True)
                except Exception:
                    nunique = None
                if nunique is not None and nunique <= 10:
                    picked = col
                    target = col
                    rich_print(f"[yellow]No standard target found. Falling back to column '{col}' (unique={nunique}).[/]")
                    break

        if picked is None:
            rich_print("[bold red]Could not auto-detect a suitable target column. Available columns:\n[/]" + str(list(df.columns[:50])))
            rich_print("[yellow]Please re-run with the desired target as the first positional argument or set PPS_TARGET environment variable.[/]")
            sys.exit(3)

    rich_print(f"[cyan]Computing PPS predictors for target '{target}' ...[/]")
    try:
        predictors_df = pps.predictors(df, y=target)
    except Exception as e:
        rich_print(f"[bold red]pps.predictors failed:[/] {e}")
        rich_print("[yellow]You can try a smaller sample (set PPS_SAMPLE) or clean the data.[/]")
        sys.exit(4)

    # Order by ppscore desc
    predictors_df = predictors_df.sort_values("ppscore", ascending=False)
    rich_print("[bold green]Top predictors:[/]")
    if console:
        console.print(predictors_df.head(top), justify="left")
    else:
        print(predictors_df.head(top).to_string(index=False))

    if out_path is None:
        out_path = Path.cwd() / f"pps_{target}.csv"

    predictors_df.to_csv(out_path, index=False)
    rich_print(f"[green]Saved predictors to:[/] {out_path}")

    if do_plot:
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except Exception:
            rich_print("[yellow]To plot results install: pip install seaborn matplotlib[/]")
        else:
            top_df = predictors_df.head(top)
            plt.figure(figsize=(10, max(4, top * 0.35)))
            sns.barplot(data=top_df, x="ppscore", y="x")
            plt.xlabel("PPS (predictive power score)")
            plt.ylabel("Predictor")
            plt.title(f"Top {top} predictors for target '{target}'")
            plt.tight_layout()
            fig_path = out_path.with_suffix(".png")
            plt.savefig(fig_path)
            rich_print(f"[green]Saved plot to:[/] {fig_path}")



if __name__ == "__main__":
    main()
