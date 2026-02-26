import marimo

__generated_with = "0.20.1"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3
    return pd, plt, venn3


@app.cell
def _(pd):
    df = pd.read_csv("enzyme_discovery.csv", sep="\t")
    return (df,)


@app.cell
def _(venn3):
    def plot_venn(row, ax):
        # venn3 subsets order: (100, 010, 110, 001, 101, 011, 111)
        # A = Pretrained, B = HEV, C = UCB
        subsets = (
            int(row["Pretrained only"]),  # 100
            int(row["HEV only"]),         # 010
            int(row["Pre ? HEV"]),        # 110
            int(row["UCB only"]),         # 001
            int(row["Pre ? UCB"]),        # 101
            int(row["HEV ? UCB"]),        # 011
            int(row["All three"]),        # 111
        )
        venn3(subsets=subsets, set_labels=("Pretrained", "HEV", "UCB"), ax=ax)
        ax.set_title(row["Task"])
    return (plot_venn,)


@app.cell
def _(df, plot_venn, plt):
    fig, axes = plt.subplots(1, len(df), figsize=(6 * len(df), 5))
    for ax, (_, row) in zip(axes, df.iterrows()):
        plot_venn(row, ax)
    plt.tight_layout()
    fig
    return (fig,)


if __name__ == "__main__":
    app.run()
