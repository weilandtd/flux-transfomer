import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from sigmaflux.viz import (
    bipartite_graph,
    common_subgraph,
    hubs,
    plot_bipartite_graph,
    relabel,
)

S, T = "source", "target"


def toy():
    # rows are targets x, y, z; columns sources a, b, c
    return np.array([[1.0, 0.95, 0.1],
                     [0.2, 2.0, 0.0],
                     [0.0, 0.0, 0.0]])


def test_threshold_keeps_edges_relative_to_each_target_max():
    g = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0.9)
    assert set(map(frozenset, g.edges)) == {
        frozenset({(S, "a"), (T, "x")}), frozenset({(S, "b"), (T, "x")}),
        frozenset({(S, "b"), (T, "y")})}
    assert g.edges[(S, "b"), (T, "x")]["weight"] == pytest.approx(0.95)
    assert (S, "c") not in g and (T, "z") not in g


def test_zero_threshold_keeps_every_positive_weight_only():
    g = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0)
    assert g.number_of_edges() == 5


def test_absolute_threshold_cuts_raw_weights():
    g = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0.96, relative=False)
    assert set(map(frozenset, g.edges)) == {frozenset({(S, "a"), (T, "x")}),
                                            frozenset({(S, "b"), (T, "y")})}
    assert g.edges[(S, "b"), (T, "y")]["weight"] == pytest.approx(2.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        bipartite_graph(toy(), ["a", "b"], ["x", "y", "z"])


def test_relabel_merges_with_max_and_drops_unmapped():
    g = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0.9)
    r = relabel(g, source_map={"a": "ab", "b": "ab"}, target_map=lambda t: None if t == "y" else t)
    assert list(r.edges(data="weight")) in ([((S, "ab"), (T, "x"), 1.0)],
                                            [((T, "x"), (S, "ab"), 1.0)])


def test_common_subgraph_keeps_shared_edges_with_min_weight():
    g1 = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0.9)
    g2 = bipartite_graph(np.array([[0.5, 1.0]]), ["a", "b"], ["x"], threshold=0.4)
    c = common_subgraph([g1, g2])
    assert set(map(frozenset, c.edges)) == {frozenset({(S, "a"), (T, "x")}),
                                            frozenset({(S, "b"), (T, "x")})}
    assert c.edges[(S, "a"), (T, "x")]["weight"] == pytest.approx(0.5)
    assert c.edges[(S, "a"), (T, "x")]["weights"] == pytest.approx([1.0, 0.5])


def test_hubs_rank_by_weighted_degree():
    g = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0.9)
    assert hubs(g, n=2) == [(S, "b"), (S, "a")]
    assert hubs(g, side=T, n=1) == [(T, "x")]
    assert hubs(g, n=None, min_strength=1.5) == [(S, "b")]


def test_plot_returns_axes():
    g = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0.9)
    ax = plot_bipartite_graph(g, node_colors={(T, "x"): "red"}, hub_labels={(S, "b"): "b"},
                              legend={"x": "red"}, bold={(S, "b")})
    assert ax.texts[0].get_text() == "b" and ax.texts[0].get_fontweight() == "bold"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["x"]


@pytest.mark.parametrize("layout", ["spring", "biplot", "columns"])
def test_layouts_place_every_node(layout):
    g = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0.1)
    ax = plot_bipartite_graph(g, layout=layout, hub_labels={(S, "a"): "a", (T, "y"): "y"},
                              edges=layout != "biplot")
    assert {t.get_text() for t in ax.texts} == {"a", "y"}


def test_legend_annotation_lists_named_nodes_by_color():
    g = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0.9)
    ax = plot_bipartite_graph(g, node_colors={(S, "b"): "red"}, hub_labels={(S, "b"): "b"},
                              legend={"tokens": "green"}, annotate="legend")
    assert not ax.texts
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["tokens", "b"]
    assert ax.get_legend().legend_handles[1].get_facecolor()[:3] == (1.0, 0.0, 0.0)


def test_unknown_layout_raises():
    g = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"])
    with pytest.raises(ValueError):
        plot_bipartite_graph(g, layout="circle")


def test_edges_can_come_from_another_graph():
    full = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0)
    strong = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0.9)
    ax = plot_bipartite_graph(full, layout="biplot", edges=strong)
    assert len(ax.collections[0].get_segments()) == strong.number_of_edges()


def test_legend_none_color_is_a_bold_header():
    g = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0.9)
    ax = plot_bipartite_graph(g, legend={"Group": None, "x": "red"})
    header, entry = ax.get_legend().get_texts()
    assert header.get_fontweight() == "bold" and entry.get_fontweight() == "normal"


def test_node_styles_hatch_nodes_and_legend_swatches():
    g = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0.9)
    striped = {"facecolor": "white", "edgecolor": "black", "hatch": "////"}
    ax = plot_bipartite_graph(g, node_styles={(S, "a"): striped}, legend={"striped": striped})
    assert "////" in {c.get_hatch() for c in ax.collections}
    assert ax.get_legend().legend_handles[0].get_hatch() == "////"


def test_source_gap_separates_sources():
    # a and b have the same targets, so the spring layout puts them close together
    g = bipartite_graph(np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                        ["a", "b", "c"], ["x", "y", "z"], threshold=0.5)
    labels = {(S, "a"): "a", (S, "b"): "b"}
    gap = [np.linalg.norm(np.subtract(*(t.xy for t in ax.texts)))
           for ax in (plot_bipartite_graph(g, hub_labels=labels),
                      plot_bipartite_graph(g, hub_labels=labels, source_gap=0.5))]
    assert gap[1] > 2 * gap[0]



def test_edge_colors_color_the_named_edges_either_way_round():
    g = bipartite_graph(toy(), ["a", "b", "c"], ["x", "y", "z"], threshold=0.9)
    ax = plot_bipartite_graph(g, edge_colors={((S, "a"), (T, "x")): "red"})
    rgb = {tuple(c[:3]) for c in ax.collections[0].get_colors()}
    assert rgb == {(1.0, 0.0, 0.0), (0.3, 0.3, 0.3)}
