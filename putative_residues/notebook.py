import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Protein Feature Importance Viewer
    """)
    return


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _(mo):
    pdb_id_input = mo.ui.text(value="1CRN", label="PDB ID")
    pdb_id_input
    return (pdb_id_input,)


@app.cell
def _(mo, np, pdb_id_input):
    import urllib.request
    import py3Dmol

    pdb_id = pdb_id_input.value.strip().upper()

    # Fetch PDB data
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    with urllib.request.urlopen(url) as resp:
        pdb_data = resp.read().decode("utf-8")

    # Extract unique residue numbers from ATOM records
    residue_numbers = sorted(
        {
            int(line[22:26])
            for line in pdb_data.splitlines()
            if line.startswith("ATOM")
        }
    )
    n_residues = len(residue_numbers)

    # Placeholder: random importance scores (replace with real data)
    rng = np.random.default_rng(42)
    importance = rng.uniform(0, 1, size=n_residues)

    # Map importance to colors (blue=low, red=high)
    def importance_to_hex(val):
        r = int(255 * val)
        b = int(255 * (1 - val))
        return f"#{r:02x}00{b:02x}"

    def build_viewer_html(pdb_data, residue_numbers, importance):
        """Build py3Dmol viewer and return its HTML. Keeps the view object
        local to this function so marimo's datasource introspection never sees it."""
        viewer = py3Dmol.view(width=800, height=500)
        viewer.addModel(pdb_data, "pdb")
        viewer.setStyle({"cartoon": {"color": "grey"}})
        for resi, score in zip(residue_numbers, importance):
            viewer.setStyle({"resi": resi}, {"cartoon": {"color": importance_to_hex(score)}})
        viewer.zoomTo()
        return viewer._make_html()

    import html as _html
    raw_html = build_viewer_html(pdb_data, residue_numbers, importance)
    viewer_iframe = mo.Html(
        f'<iframe srcdoc="{_html.escape(raw_html)}" '
        f'width="820" height="520" style="border:none;"></iframe>'
    )
    info = mo.md(f"**{pdb_id}** — {n_residues} residues, colored by importance (blue=low, red=high)")
    mo.vstack([info, viewer_iframe])
    return


if __name__ == "__main__":
    app.run()
