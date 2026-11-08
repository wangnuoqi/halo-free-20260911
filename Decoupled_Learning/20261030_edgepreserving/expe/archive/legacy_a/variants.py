"""Original fixed-mean A1/A2 definitions retained for checkpoint loading."""

LEGACY_A_VARIANTS = {
    "viv_gf_a1": {},
    "viv_gf_a2_grad": dict(use_edges=True, edge_mode="gradient"),
    "viv_gf_a2_otsu": dict(use_edges=True, edge_mode="otsu"),
    "viv_gf_a2_input": dict(use_edges=True, edge_mode="input"),
    "viv_gf_a2": dict(use_edges=True),
}
