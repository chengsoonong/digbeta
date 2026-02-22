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
    pdb_id_input = mo.ui.text(value="1ACJ", label="PDB ID")
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

    # Catalytic triad of acetylcholinesterase (chain A)
    catalytic_triad = {("A", 200), ("A", 440), ("A", 327)}  # Ser200, His440, Glu327

    # Parse CA atom coordinates per residue (chain, resnum) -> (x, y, z)
    ca_coords = {}
    residue_set = set()
    for line in pdb_data.splitlines():
        if not line.startswith("ATOM"):
            continue
        chain = line[21]
        resi = int(line[22:26])
        residue_set.add(resi)
        atom_name = line[12:16].strip()
        if atom_name == "CA":
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            ca_coords[(chain, resi)] = np.array([x, y, z])

    residue_numbers = sorted(residue_set)
    n_residues = len(residue_numbers)

    # Get catalytic triad CA positions
    triad_positions = [ca_coords[key] for key in catalytic_triad if key in ca_coords]

    # Score each residue by proximity to catalytic triad
    decay_radius = 10.0  # Angstroms
    importance = np.zeros(n_residues)
    for i, resi in enumerate(residue_numbers):
        # Check if this residue is in the triad (any chain)
        if any((ch, resi) in catalytic_triad for ch in "A"):
            importance[i] = 1.0
            continue
        # Find CA coord for this residue (prefer chain A)
        coord = ca_coords.get(("A", resi))
        if coord is None:
            continue
        min_dist = min(np.linalg.norm(coord - tp) for tp in triad_positions)
        if min_dist < decay_radius:
            importance[i] = max(0.0, 1.0 - min_dist / decay_radius)

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
    info = mo.md(
        f"**{pdb_id}** — {n_residues} residues, colored by proximity to catalytic triad "
        f"(Ser200, His440, Glu327). Red=catalytic site, blue=distant."
    )
    mo.vstack([info, viewer_iframe])
    return


if __name__ == "__main__":
    app.run()
